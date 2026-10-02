"""Synthetic microbenchmarks for uncertainty-aware IMF mapping.

This measures implementation cost and controlled association behavior only. It
does not use the research test split and is not a map-accuracy result.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from common.mapping.relations import compute_relations
from common.mapping.semantic_map import PersistentSemanticMap, SemanticObservation
from common.mapping.uncertainty import (
    approximate_pose_covariance,
    pixel_depth_covariance,
    propagate_camera_to_map,
)
from common.mapping.visibility import Visibility, classify_map_point


def _percentiles(samples: list[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(values.mean() * 1000),
        "p50_ms": float(np.percentile(values, 50) * 1000),
        "p95_ms": float(np.percentile(values, 95) * 1000),
        "std_ms": float(values.std() * 1000),
        "sample_count": int(values.size),
    }


def _make_observations(seed: int, pairs: int, frames: int) -> tuple[list[SemanticObservation], int]:
    rng = np.random.default_rng(seed)
    observations = []
    truth_count = pairs * 2
    centers = []
    for pair in range(pairs):
        base_x = (pair % 5 - 2) * 1.6
        base_y = (pair // 5 - 3.5) * 1.4
        centers.extend(((base_x, base_y, 0.0), (base_x + 0.35, base_y, 0.0)))
    for frame in range(frames):
        for object_id, center in enumerate(centers):
            measured = np.asarray(center) + rng.normal(0.0, 0.005, size=3)
            label = "cone" if (object_id // 2) % 2 == 0 else "rock"
            other = "rock" if label == "cone" else "cone"
            observations.append(
                SemanticObservation(
                    xyz_m=tuple(measured.tolist()),
                    class_probabilities={label: 0.9, other: 0.1},
                    confidence=0.9,
                    timestamp_s=frame * 0.1,
                    extent_m=0.03,
                    source="predicted",
                    source_observation=f"object-{object_id:04d}:frame-{frame:03d}",
                    position_covariance_m2=(np.eye(3) * 25e-6).tolist(),
                    covariance_source="synthetic_known_noise",
                )
            )
    return observations, truth_count


def _association_metrics(local_map: PersistentSemanticMap, truth_count: int) -> dict:
    objects_by_entity = []
    entities_by_object: dict[str, int] = {}
    for entity in local_map.entities:
        object_ids = {
            item.split(":", 1)[0]
            for item in entity.source_observations
            if item.startswith("object-")
        }
        objects_by_entity.append(object_ids)
        for object_id in object_ids:
            entities_by_object[object_id] = entities_by_object.get(object_id, 0) + 1
    merged = sum(len(object_ids) > 1 for object_ids in objects_by_entity)
    incorrectly_merged_objects = {
        object_id
        for object_ids in objects_by_entity
        if len(object_ids) > 1
        for object_id in object_ids
    }
    duplicated = sum(count > 1 for count in entities_by_object.values())
    return {
        "ground_truth_objects": truth_count,
        "map_entities": len(local_map.entities),
        "duplicate_object_fraction": duplicated / max(truth_count, 1),
        "incorrectly_merged_object_fraction": len(incorrectly_merged_objects) / max(truth_count, 1),
        "mixed_entity_fraction": merged / max(len(local_map.entities), 1),
        "association_counts": dict(local_map.association_stats),
        "serialized_map_bytes": len(
            json.dumps(local_map.records(), separators=(",", ":")).encode()
        ),
        "mean_semantic_entropy": float(
            np.mean([item.semantic_entropy for item in local_map.entities])
        )
        if local_map.entities
        else 0.0,
    }


def run(repeats: int = 3, pairs: int = 40, frames: int = 8) -> dict:
    observations, truth_count = _make_observations(20261001, pairs, frames)
    measurements = {}
    retained_maps = {}
    for association in ("euclidean", "mahalanobis"):
        for fusion in ("arithmetic", "bayesian"):
            timings = []
            result_map = None
            for repeat in range(repeats):
                local_map = PersistentSemanticMap(
                    association_m=0.75,
                    voxel_m=0.2,
                    max_entities=truth_count * 4,
                    max_age_s=1000.0,
                    association_mode=association,
                    semantic_fusion=fusion,
                    evidence_correlation_time_s=1.0,
                )
                sample_times = []
                for observation in observations:
                    start = time.perf_counter()
                    local_map.update(observation)
                    sample_times.append(time.perf_counter() - start)
                if repeat:
                    timings.extend(sample_times)
                result_map = local_map
            measurements[f"{association}_{fusion}"] = {
                "per_observation": _percentiles(timings),
                **_association_metrics(result_map, truth_count),
            }
            retained_maps[association + "_" + fusion] = result_map

    selected_map = retained_maps["mahalanobis_bayesian"]
    # Camera at z=10 m, looking down: OpenCV +z maps to local-map -z.
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = (0.0, 0.0, 10.0)
    pose_covariance = approximate_pose_covariance(0.7)
    intrinsics = {
        "width": 640.0,
        "height": 480.0,
        "fx": 440.0,
        "fy": 440.0,
        "cx": 319.5,
        "cy": 239.5,
    }
    covariance_samples = []
    for _ in range(1000):
        start = time.perf_counter()
        covariance = pixel_depth_covariance(
            (320, 240), 3.0, intrinsics, pixel_sigma=1.0, depth_sigma_m=0.05
        )
        propagate_camera_to_map((0.0, 0.0, 3.0), covariance, transform, pose_covariance)
        covariance_samples.append(time.perf_counter() - start)

    depth = np.full((32, 32), 10.0, dtype=np.float32)
    depth_valid = np.ones_like(depth)
    visibility_samples = []
    entities = selected_map.entities[: min(64, len(selected_map.entities))]
    visible_count = 0
    for _ in range(10):
        for entity in entities:
            start = time.perf_counter()
            code, _, _ = classify_map_point(entity.xyz_m, transform, intrinsics, depth, depth_valid)
            visibility_samples.append(time.perf_counter() - start)
            visible_count += code == Visibility.VISIBLE

    relation_samples = []
    visibility = {item.entity_id: int(Visibility.VISIBLE) for item in selected_map.entities}
    for _ in range(250):
        start = time.perf_counter()
        compute_relations(
            selected_map,
            timestamp_s=frames * 0.1,
            position_m=(0.0, 0.0, 0.0),
            yaw_rad=0.3,
            semantic_rules={
                "track": {"fly_allowed": False, "land_allowed": False},
                "cone": {"land_allowed": False},
                "rock": {"land_allowed": False},
            },
            visibility_by_entity=visibility,
        )
        relation_samples.append(time.perf_counter() - start)

    return {
        "schema": "imagination-imf-revision-microbenchmark-v1",
        "purpose": "synthetic algorithm-cost and controlled association diagnostic; not research accuracy",
        "platform": platform.platform(),
        "seed": 20261001,
        "repeats": repeats,
        "timed_repeats_after_one_warmup": max(repeats - 1, 0),
        "synthetic_objects": truth_count,
        "frames_per_object": frames,
        "observations_per_configuration": len(observations),
        "map_update_results": measurements,
        "other_stages": {
            "covariance_propagation": _percentiles(covariance_samples),
            "visibility_per_entity": _percentiles(visibility_samples),
            "visible_checks": int(visible_count),
            "relation_and_token_construction": _percentiles(relation_samples),
        },
        "limitations": [
            "Synthetic point layout and known covariance do not estimate real-scene association quality.",
            "Map-update timing includes spatial lookup, association, evidence fusion and entity maintenance.",
            "No target-test data, simulator labels or rendered dataset files are accessed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/benchmarks/imf_revision_microbench.json"
    )
    args = parser.parse_args()
    if min(args.pairs, args.frames) < 1 or args.repeats < 2:
        parser.error("pairs and frames must be positive; repeats must be at least 2")
    result = run(args.repeats, args.pairs, args.frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
