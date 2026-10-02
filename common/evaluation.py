"""Validation/test metrics and model-resource measurements."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.mapping import (
    PersistentSemanticMap,
    build_predicted_contexts,
    extract_regions,
    register_detection,
)
from common.models import build_model
from common.registry import POI_CLASSES, SCENE_CLASSES, SEMANTIC_CLASSES
from common.runtime import benchmark_model, file_sha256, load_checkpoint, save_json, select_device
from common.training.engine import attach_map_context, forward_model, model_mode, move_targets
from common.training.metrics import BinaryCounts, CategoricalMetrics, POIMetrics
from data.adapter import ImaginationDataset, load_manifest, validate_manifest


def _match_regions(predicted, truth, maximum_distance: float = 2.0):
    used = set()
    matches = []
    for detection in predicted:
        candidates = [
            (
                np.hypot(
                    detection.grid_xy[0] - target.grid_xy[0],
                    detection.grid_xy[1] - target.grid_xy[1],
                ),
                index,
            )
            for index, target in enumerate(truth)
            if index not in used and detection.class_name == target.class_name
        ]
        if candidates:
            distance, index = min(candidates)
            if distance <= maximum_distance:
                used.add(index)
                matches.append(float(distance))
    return len(matches), len(predicted) - len(matches), len(truth) - len(matches), matches


def _model_inputs(batch, family: str):
    state = batch["vehicle_state"][:1]
    state_validity = batch["state_validity"][:1]
    if family == "imf_htransformer":
        return batch["analytical"][:1], batch["validity"][:1], state, state_validity
    return batch["rgb"][:1], state, state_validity


def evaluate(checkpoint_path, manifest, split, output_directory, device_name="auto"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    family = config["model"]["family"]
    semantic_config = (
        Path(__file__).resolve().parents[1] / "data" / "configs" / "semantic_rules.yaml"
    )
    if config["model"].get("map_context", True):
        expected_map_hash = checkpoint.get("map_policy_hash")
        actual_map_hash = file_sha256(semantic_config)
        if expected_map_hash != actual_map_hash:
            raise ValueError("Checkpoint semantic-map policy differs from the current map rules")
    device = select_device(device_name)
    model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    poi_classes = config.get("poi", {}).get("classes", POI_CLASSES)
    needs_geometry = family == "imf_htransformer" or bool(config["model"].get("map_context", True))
    validate_manifest(
        manifest,
        check_files=True,
        require_analytical=needs_geometry,
        require_candidate_depth=bool(config["model"].get("map_context", True)),
    )
    dataset = ImaginationDataset(manifest, split, model_mode(family), poi_classes)
    loader = DataLoader(
        dataset, batch_size=int(config["training"].get("batch_size", 8)), shuffle=False
    )
    map_contexts = None
    if config["model"].get("map_context", True):
        context_visual = ImaginationDataset(
            manifest, split, model_mode(family), poi_classes, include_targets=False
        )
        context_geometry = (
            context_visual
            if family == "imf_htransformer"
            else ImaginationDataset(manifest, split, "analytical", include_targets=False)
        )
        map_contexts = build_predicted_contexts(
            model,
            family,
            context_visual,
            context_geometry,
            device,
            semantic_config,
            map_overrides=config.get("mapping"),
        )
    thresholds = config.get("thresholds", {})

    hazard = BinaryCounts("hazard")
    landing = BinaryCounts("landing")
    poi = POIMetrics(list(poi_classes))
    semantic = CategoricalMetrics(SEMANTIC_CLASSES)
    scene = CategoricalMetrics(SCENE_CLASSES)
    candidate_risk = CategoricalMetrics(SCENE_CLASSES)
    candidate_landing = BinaryCounts("candidate_landing")
    by_episode = defaultdict(lambda: BinaryCounts("hazard"))
    landing_counts = {"tp": 0, "fp": 0, "fn": 0, "errors": []}
    poi_counts = {"tp": 0, "fp": 0, "fn": 0, "errors": []}
    local_maps = defaultdict(PersistentSemanticMap)
    resolved_regions = unresolved_regions = 0
    total_samples = 0
    started = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            move_targets(batch, device)
            if map_contexts is not None:
                attach_map_context(batch, map_contexts)
            outputs = forward_model(model, batch, family, device)
            hazard.update(
                outputs["hazard_logits"],
                batch["hazard_target"],
                batch["hazard_validity"],
                thresholds.get("hazard", 0.5),
            )
            landing.update(
                outputs["landing_logits"],
                batch["landing_target"],
                batch["landing_validity"],
                thresholds.get("landing", 0.5),
            )
            poi.update(
                outputs["poi"]["class_logits"],
                batch["poi_target"],
                batch["poi_validity"],
                thresholds.get("poi", 0.5),
            )
            semantic.update(
                outputs["semantic_logits"], batch["semantic_target"], batch["semantic_validity"]
            )
            scene.update(outputs["scene_risk_logits"], batch["scene_target"])
            if "candidate_risk_target" in batch:
                candidate_mask = batch["candidate_target_validity"]
                candidate_risk.update(
                    outputs["candidate"]["risk_logits"].transpose(1, 2),
                    batch["candidate_risk_target"],
                    candidate_mask,
                )
                candidate_landing.update(
                    outputs["candidate"]["landing_safe_logits"],
                    batch["candidate_landing_target"],
                    candidate_mask,
                    thresholds.get("candidate_landing", 0.5),
                )
            total_samples += int(batch["hazard_target"].shape[0])

            landing_probability = outputs["landing_logits"].sigmoid().cpu().numpy()
            poi_probability = outputs["poi"]["class_logits"].sigmoid().cpu().numpy()
            for row in range(landing_probability.shape[0]):
                episode_id = batch["metadata"]["episode_id"][row]
                by_episode[episode_id].update(
                    outputs["hazard_logits"][row : row + 1],
                    batch["hazard_target"][row : row + 1],
                    batch["hazard_validity"][row : row + 1],
                    thresholds.get("hazard", 0.5),
                )
                predicted_landing = extract_regions(
                    landing_probability[row],
                    thresholds.get("landing", 0.5),
                    "landing",
                    min_cells=int(config.get("postprocess", {}).get("landing_min_cells", 4)),
                )
                target_landing = extract_regions(
                    batch["landing_target"][row].cpu().numpy(), 0.5, "landing"
                )
                tp, fp, fn, errors = _match_regions(predicted_landing, target_landing)
                landing_counts["tp"] += tp
                landing_counts["fp"] += fp
                landing_counts["fn"] += fn
                landing_counts["errors"].extend(errors)

                predicted_poi = extract_regions(
                    poi_probability[row], thresholds.get("poi", 0.5), "poi", list(poi_classes), 1
                )
                target_poi = extract_regions(
                    batch["poi_target"][row].cpu().numpy(), 0.5, "poi", list(poi_classes), 1
                )
                tp, fp, fn, errors = _match_regions(predicted_poi, target_poi)
                poi_counts["tp"] += tp
                poi_counts["fp"] += fp
                poi_counts["fn"] += fn
                poi_counts["errors"].extend(errors)

                if family == "imf_htransformer":
                    calibration_values = batch["calibration"][row].numpy()
                    calibration = dict(
                        zip(("width", "height", "fx", "fy", "cx", "cy"), calibration_values)
                    )
                    pose = (
                        batch["camera_to_local_map"][row].numpy()
                        if bool(batch["camera_pose_valid"][row])
                        else np.empty((0, 0))
                    )
                    depth = batch["analytical"][row, 22].numpy()
                    depth_validity = batch["validity"][row, 22].numpy()
                    scale = float(batch["depth_scale_m"][row])
                    timestamp = float(batch["metadata"]["timestamp"][row])
                    local_map = local_maps[episode_id]
                    for detection in predicted_landing + predicted_poi:
                        mapped = register_detection(
                            detection, depth, depth_validity, scale, calibration, pose, timestamp
                        )
                        if mapped.resolved:
                            local_map.update_region(
                                mapped.class_name,
                                mapped.map_xyz_m,
                                mapped.radius_m or 0.0,
                                mapped.probability,
                                timestamp,
                                source_observation=f"{mapped.kind}:{timestamp:.6f}:{mapped.source_grid_xy}",
                            )
                            resolved_regions += 1
                        else:
                            local_map.retain_unresolved(
                                mapped.class_name,
                                mapped.source_grid_xy,
                                mapped.probability,
                                timestamp,
                                f"{mapped.kind}:{timestamp:.6f}",
                            )
                            unresolved_regions += 1

    elapsed = time.perf_counter() - started
    metrics = {
        **hazard.result(),
        **landing.result(),
        **poi.result(),
        **semantic.result("semantic"),
        **scene.result("scene"),
        **candidate_risk.result("candidate_risk"),
        **candidate_landing.result(),
    }
    ratio = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    metrics.update(
        {
            "landing_candidate_detection_rate": ratio(
                landing_counts["tp"], landing_counts["tp"] + landing_counts["fn"]
            ),
            "landing_false_proposal_rate": ratio(
                landing_counts["fp"], landing_counts["tp"] + landing_counts["fp"]
            ),
            "landing_2d_localization_error_cells": float(np.mean(landing_counts["errors"]))
            if landing_counts["errors"]
            else None,
            "poi_detection_recall": ratio(poi_counts["tp"], poi_counts["tp"] + poi_counts["fn"]),
            "poi_detection_precision": ratio(poi_counts["tp"], poi_counts["tp"] + poi_counts["fp"]),
            "poi_2d_localization_error_cells": float(np.mean(poi_counts["errors"]))
            if poi_counts["errors"]
            else None,
            "metric_xyz_resolution_rate": ratio(
                resolved_regions, resolved_regions + unresolved_regions
            ),
            "unresolved_detection_rate": ratio(
                unresolved_regions, resolved_regions + unresolved_regions
            ),
            "poi_3d_position_error_m": None,
            "landing_3d_position_error_m": None,
            "duplicate_association_rate": None,
        }
    )

    first = next(iter(loader))
    if map_contexts is not None:
        attach_map_context(first, map_contexts)
    inputs = _model_inputs(first, family) + (
        first["relational"][:1],
        first["relational_validity"][:1],
        first["candidates"][:1],
        first["candidate_validity"][:1],
        first["candidate_feature_validity"][:1],
    )
    resources = benchmark_model(model, inputs, device, warmup=2, repetitions=5)
    resources.update(
        {
            "dataset_inference_seconds": elapsed,
            "pre_extraction_latency_ms": None,
            "end_to_end_latency_ms": None,
            "average_power_w": None,
            "energy_joules_per_frame": None,
            "measurement_note": "Frame model measured separately from C++ IMF and map post-processing.",
        }
    )
    summary = {
        "model": family,
        "split": split,
        "samples": len(dataset),
        "metrics": metrics,
        "per_episode": {key: value.result() for key, value in by_episode.items()},
        "resources": resources,
        "map_entities": sum(len(value.entities) for value in local_maps.values()),
        "map_unresolved_observations": sum(
            len(value.unresolved_observations) for value in local_maps.values()
        ),
        "label_provenance": "independent simulator class masks; labels are not used as inference inputs",
        "energy": "NOT_MEASURED",
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "metrics.json", summary)
    save_json(output / "resource_metrics.json", resources)
    save_json(
        output / "mapped_detections.json",
        {key: value.records() for key, value in local_maps.items()},
    )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        writer.writerows(metrics.items())
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.checkpoint, args.manifest, args.split, args.output, args.device), indent=2
        )
    )


if __name__ == "__main__":
    main()
