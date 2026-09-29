"""Evaluate checkpoints and compare analytical and RGB models on one split."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.common import (
    benchmark_model,
    file_sha256,
    load_checkpoint,
    save_json,
    select_device,
)
from training.data import ImaginationDataset, validate_manifest
from training.metrics import HazardCounts, NavigationMetrics, non_inferiority
from training.models import build_model
from training.train import model_forward, move_batch


def _single_inputs(batch: dict[str, Any], model_type: str) -> tuple[torch.Tensor, ...]:
    state = batch["vehicle_state"][:1]
    state_validity = batch["state_validity"][:1]
    if model_type == "analytical":
        return batch["analytical"][:1], batch["validity"][:1], state, state_validity
    return batch["rgb"][:1], state, state_validity


def _metric_groups():
    return defaultdict(lambda: (HazardCounts(), NavigationMetrics()))


@torch.inference_mode()
def evaluate(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    split: str,
    output_directory: str | Path,
    device_name: str = "auto",
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_type = config["model"]["type"]
    if checkpoint.get("manifest_hash") != file_sha256(manifest_path):
        raise ValueError("Checkpoint and evaluation dataset manifests differ")
    validate_manifest(manifest_path)
    device = select_device(device_name)
    model = build_model(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    dataset = ImaginationDataset(
        manifest_path, split, "analytical" if model_type == "analytical" else "baseline"
    )
    loader = DataLoader(dataset, batch_size=int(config["training"].get("batch_size", 8)),
                        shuffle=False, num_workers=0)
    overall_hazard = HazardCounts()
    hazard_frames = HazardCounts()
    overall_navigation = NavigationMetrics()
    environment = _metric_groups()
    episode = _metric_groups()
    prediction_ids, hazard_predictions, waypoint_predictions = [], [], []
    first_batch = None

    for batch in loader:
        batch = move_batch(batch, device)
        if first_batch is None:
            first_batch = batch
        outputs = model_forward(model, batch, model_type)
        overall_hazard.update(outputs["hazard_logits"], batch["hazard_target"],
                              batch["hazard_validity"])
        overall_navigation.update(outputs["waypoint"], batch["waypoint_target"])
        metadata = batch["metadata"]
        for index, sample_id in enumerate(metadata["sample_id"]):
            logits = outputs["hazard_logits"][index:index + 1]
            target = batch["hazard_target"][index:index + 1]
            validity = batch["hazard_validity"][index:index + 1]
            waypoint = outputs["waypoint"][index:index + 1]
            waypoint_target = batch["waypoint_target"][index:index + 1]
            if bool(((target >= 0.5) & (validity >= 0.5)).any()):
                hazard_frames.update(logits, target, validity)
            for groups, key in ((environment, metadata["environment_id"][index]),
                                (episode, metadata["episode_id"][index])):
                groups[key][0].update(logits, target, validity)
                groups[key][1].update(waypoint, waypoint_target)
            prediction_ids.append(sample_id)
            hazard_predictions.append(logits.sigmoid().cpu().numpy())
            waypoint_predictions.append(waypoint.cpu().numpy())

    if first_batch is None:
        raise ValueError("Evaluation split is empty")
    resources = benchmark_model(model, _single_inputs(first_batch, model_type), device)
    resources["checkpoint_bytes"] = Path(checkpoint_path).stat().st_size
    resources["analytical_preextraction_latency_ms"] = None
    resources["end_to_end_latency_ms"] = None
    resources["energy_joules"] = None
    overall = {**overall_hazard.result(), **overall_navigation.result()}
    grouped = {
        "environment": {
            key: {**values[0].result(), **values[1].result()}
            for key, values in environment.items()
        },
        "episode": {
            key: {**values[0].result(), **values[1].result()}
            for key, values in episode.items()
        },
    }
    summary = {
        "model": model_type,
        "split": split,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "overall": overall,
        "hazard_containing_frames": hazard_frames.result(),
        "groups": grouped,
        "resources": resources,
    }
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "summary.json", summary)
    np.savez_compressed(
        output / "predictions.npz",
        sample_ids=np.asarray(prediction_ids),
        hazard_probability=np.concatenate(hazard_predictions),
        waypoint=np.concatenate(waypoint_predictions),
    )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("scope", "id", *overall.keys()))
        writer.writerow(("all", "all", *overall.values()))
        for scope, groups in grouped.items():
            for identity, metrics in groups.items():
                writer.writerow((scope, identity, *(metrics[key] for key in overall)))
    return summary


def compare(
    analytical_checkpoint: str | Path,
    baseline_checkpoint: str | Path,
    manifest: str | Path,
    output: str | Path,
    fnr_epsilon: float,
    navigation_delta: float,
    device: str = "auto",
) -> dict[str, Any]:
    destination = Path(output)
    analytical_checkpoint_data = torch.load(
        analytical_checkpoint, map_location="cpu", weights_only=False
    )
    baseline_checkpoint_data = torch.load(
        baseline_checkpoint, map_location="cpu", weights_only=False
    )
    analytical_config = analytical_checkpoint_data["config"]
    baseline_config = baseline_checkpoint_data["config"]
    if analytical_config["htransformer"] != baseline_config["htransformer"]:
        raise ValueError("Fair comparison requires identical HTransformer configurations")
    if analytical_config["losses"] != baseline_config["losses"]:
        raise ValueError("Fair comparison requires identical loss configurations")
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "seed"):
        if analytical_config["training"].get(key) != baseline_config["training"].get(key):
            raise ValueError(f"Fair comparison requires identical training.{key}")
    if analytical_config["model"].get("state_dim", 10) != \
       baseline_config["model"].get("state_dim", 10):
        raise ValueError("Fair comparison requires identical vehicle-state dimensions")
    analytical = evaluate(
        analytical_checkpoint, manifest, "test", destination / "analytical", device
    )
    baseline = evaluate(
        baseline_checkpoint, manifest, "test", destination / "baseline", device
    )
    criterion = non_inferiority(
        analytical["overall"], baseline["overall"], fnr_epsilon, navigation_delta
    )
    result = {"analytical": analytical, "baseline": baseline,
              "configured_non_inferiority": criterion}
    save_json(destination / "comparison.json", result)
    columns = (
        "model", "hazard_fnr", "hazard_recall", "hazard_precision", "hazard_iou",
        "waypoint_translation_mae_m", "parameters", "latency_mean_ms",
        "estimated_macs_per_sample",
        "analytical_preextraction_latency_ms", "end_to_end_latency_ms",
        "peak_inference_memory_bytes", "energy_joules",
    )
    with (destination / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for summary in (analytical, baseline):
            writer.writerow({
                "model": summary["model"],
                **{key: summary["overall"][key] for key in columns if key in summary["overall"]},
                **{key: summary["resources"][key] for key in columns if key in summary["resources"]},
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--compare", nargs=2, metavar=("ANALYTICAL", "BASELINE"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fnr-epsilon", type=float)
    parser.add_argument("--navigation-delta", type=float)
    arguments = parser.parse_args()
    if bool(arguments.checkpoint) == bool(arguments.compare):
        parser.error("choose exactly one of --checkpoint or --compare")
    if arguments.compare:
        if arguments.fnr_epsilon is None or arguments.navigation_delta is None:
            parser.error("comparison requires explicit --fnr-epsilon and --navigation-delta")
        compare(*arguments.compare, arguments.manifest, arguments.output,
                arguments.fnr_epsilon, arguments.navigation_delta, arguments.device)
    else:
        evaluate(arguments.checkpoint, arguments.manifest, arguments.split,
                 arguments.output, arguments.device)


if __name__ == "__main__":
    main()
