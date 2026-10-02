#!/usr/bin/env python3
"""Validate structure, geometry/sensor continuity, labels, and seed reproducibility."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generator.specs import build_episode_spec


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numbered_files(path: Path, suffix: str) -> tuple[list[Path], list[int]]:
    files = sorted(path.glob(f"*{suffix}"))
    indices = []
    for item in files:
        try:
            indices.append(int(item.stem))
        except ValueError:
            indices.append(-1)
    return files, indices


def quaternion_step_degrees(a: dict[str, str], b: dict[str, str]) -> float:
    qa = [float(a[key]) for key in ("qw", "qx", "qy", "qz")]
    qb = [float(b[key]) for key in ("qw", "qx", "qy", "qz")]
    dot = abs(sum(x * y for x, y in zip(qa, qb)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def validate_episode(episode: Path) -> tuple[list[str], dict]:
    errors: list[str] = []
    prefix = episode.name

    def check(condition: bool, message: str):
        if not condition:
            errors.append(f"{prefix}: {message}")

    required = [
        "metadata.json",
        "objects.json",
        "sensors.csv",
        "sensors_clean.csv",
        "poses.csv",
        "navigation_labels.csv",
    ]
    for name in required:
        check((episode / name).is_file(), f"missing {name}")
    if errors:
        return errors, {}
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    objects = json.loads((episode / "objects.json").read_text(encoding="utf-8"))
    sensors = read_csv(episode / "sensors.csv")
    clean = read_csv(episode / "sensors_clean.csv")
    poses = read_csv(episode / "poses.csv")
    navigation = read_csv(episode / "navigation_labels.csv")
    count = int(metadata.get("frame_count", -1))
    check(count > 0, "invalid frame_count")
    check(len(sensors) == count, f"sensors.csv rows {len(sensors)} != {count}")
    check(len(clean) == count, f"sensors_clean.csv rows {len(clean)} != {count}")
    check(len(poses) == count, f"poses.csv rows {len(poses)} != {count}")
    check(len(navigation) == count, f"navigation_labels.csv rows {len(navigation)} != {count}")

    rgb, rgb_indices = numbered_files(episode / "rgb", ".jpg")
    depth, depth_indices = numbered_files(episode / "depth", ".npy")
    segmentation, segmentation_indices = numbered_files(episode / "segmentation", ".png")
    segmentation_color, segmentation_color_indices = numbered_files(
        episode / "segmentation_color", ".png"
    )
    landing, landing_indices = numbered_files(episode / "landing_suitability", ".png")
    landing_color, landing_color_indices = numbered_files(episode / "landing_labels", ".png")
    expected_indices = list(range(max(count, 0)))
    check(rgb_indices == expected_indices, "RGB numbering is missing or non-sequential")
    check(depth_indices == expected_indices, "depth numbering is missing or non-sequential")
    check(
        segmentation_indices == expected_indices,
        "segmentation numbering is missing or non-sequential",
    )
    check(
        segmentation_color_indices == expected_indices,
        "color segmentation numbering is missing or non-sequential",
    )
    check(
        landing_indices == expected_indices,
        "landing-suitability numbering is missing or non-sequential",
    )
    check(
        landing_color_indices == expected_indices,
        "color landing-label numbering is missing or non-sequential",
    )
    for path in rgb:
        with Image.open(path) as image:
            check(image.size == (640, 480), f"{path.name} RGB size {image.size} != 640x480")
            check(image.mode == "RGB", f"{path.name} RGB mode is {image.mode}")
    mask_classes: set[int] = set()
    for path in segmentation:
        with Image.open(path) as image:
            check(image.size == (640, 480), f"{path.name} mask size {image.size} != 640x480")
            values = np.asarray(image)
            mask_classes.update(int(v) for v in np.unique(values))
            check(np.all((values >= 0) & (values <= 10)), f"{path.name} has invalid semantic IDs")
    for path in landing:
        with Image.open(path) as image:
            check(
                image.size == (640, 480), f"{path.name} landing mask size {image.size} != 640x480"
            )
            values = np.asarray(image)
            unique = set(int(v) for v in np.unique(values))
            check(
                unique.issubset({0, 1, 2}),
                f"{path.name} landing mask has invalid IDs {sorted(unique)}",
            )
    for kind, paths in (
        ("color segmentation", segmentation_color),
        ("color landing label", landing_color),
    ):
        for path in paths:
            with Image.open(path) as image:
                check(image.size == (640, 480), f"{path.name} {kind} size {image.size} != 640x480")
                check(image.mode == "RGB", f"{path.name} {kind} mode {image.mode} != RGB")
                values = np.asarray(image)
                check(bool(np.any(values > 32)), f"{path.name} {kind} is effectively black")
    min_positive = 1.0
    for path in depth:
        values = np.load(path, allow_pickle=False)
        check(values.shape == (480, 640), f"{path.name} depth shape {values.shape} != (480,640)")
        check(values.dtype == np.float32, f"{path.name} depth dtype {values.dtype} != float32")
        check(bool(np.all(np.isfinite(values))), f"{path.name} depth contains non-finite values")
        check(bool(np.all(values >= 0.0)), f"{path.name} depth contains negative values")
        fraction = float(np.mean(values > 0.0))
        min_positive = min(min_positive, fraction)
        check(fraction > 0.95, f"{path.name} only {fraction:.3f} positive depth")

    if sensors:
        timestamps = [float(row["timestamp_s"]) for row in sensors]
        check(
            all(b > a for a, b in zip(timestamps, timestamps[1:])),
            "timestamps are not strictly increasing",
        )
        for row_index, row in enumerate(sensors):
            check(
                int(row["frame_index"]) == row_index,
                f"sensor frame index mismatch at row {row_index}",
            )
            for key, value in row.items():
                try:
                    numeric = float(value)
                    check(math.isfinite(numeric), f"non-finite sensor {key} at row {row_index}")
                except (TypeError, ValueError):
                    errors.append(f"{prefix}: non-numeric sensor {key} at row {row_index}")
            clearance = float(row["ground_clearance_m"])
            check(
                1.0 <= clearance <= 3.0,
                f"ground clearance {clearance:.4f} outside [1,3] at row {row_index}",
            )
        validation_cfg = metadata["config"]["validation"]
        max_position_step = float(validation_cfg["max_position_step_m"])
        max_orientation_step = float(validation_cfg["max_orientation_step_degrees"])
        for index, (a, b) in enumerate(zip(sensors, sensors[1:]), start=1):
            distance = math.sqrt(sum((float(b[k]) - float(a[k])) ** 2 for k in ("x", "y", "z")))
            check(
                distance <= max_position_step,
                f"position step {distance:.4f}m exceeds {max_position_step} at frame {index}",
            )
            angle = quaternion_step_degrees(a, b)
            check(
                angle <= max_orientation_step,
                f"orientation step {angle:.3f}deg exceeds {max_orientation_step} at frame {index}",
            )

    valid_navigation = {
        "grass_interior": 0,
        "near_track": 1,
        "grass_track_boundary": 2,
        "track_surface": 3,
    }
    for row_index, row in enumerate(navigation):
        check(
            int(row["frame_index"]) == row_index,
            f"navigation frame index mismatch at row {row_index}",
        )
        name = row["navigation_class"]
        check(name in valid_navigation, f"unknown navigation class {name!r} at row {row_index}")
        if name in valid_navigation:
            check(
                int(row["navigation_class_id"]) == valid_navigation[name],
                f"navigation class ID/name mismatch at row {row_index}",
            )
        fractions = [
            float(row[key])
            for key in (
                "landing_unsafe_fraction",
                "landing_caution_fraction",
                "landing_safe_fraction",
            )
        ]
        check(
            all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in fractions),
            f"invalid landing fractions at row {row_index}",
        )
        check(
            abs(sum(fractions) - 1.0) < 1e-5,
            f"landing fractions do not sum to one at row {row_index}",
        )
        check(
            0.0 <= float(row["risk_score"]) <= 1.0, f"risk score outside [0,1] at row {row_index}"
        )

    required_metadata = [
        "schema_version",
        "episode_seed",
        "image_resolution",
        "fps",
        "terrain",
        "lighting",
        "render",
        "sensor_configuration",
        "semantic_classes",
        "segmentation_color_format",
        "landing_suitability_format",
        "landing_labels_format",
        "navigation_classes",
        "episode_spec_sha256",
        "generation",
        "config",
    ]
    for key in required_metadata:
        check(key in metadata, f"metadata missing {key}")
    check(
        metadata.get("image_resolution") == [640, 480], "metadata image resolution is not 640x480"
    )
    check(
        objects.get("episode_seed") == metadata.get("episode_seed"),
        "objects seed does not match metadata",
    )
    object_rows = objects.get("objects", [])
    check(bool(object_rows), "clutter list is empty")
    ids = [row.get("object_id") for row in object_rows]
    check(len(ids) == len(set(ids)), "clutter object IDs are not unique")
    # A single immutable object table per episode is the authoritative fixed layout.
    for row in object_rows:
        position = row.get("position", [])
        check(
            len(position) == 3 and all(math.isfinite(float(v)) for v in position),
            f"invalid fixed clutter position for {row.get('object_id')}",
        )

    try:
        reproduced = build_episode_spec(int(metadata["episode_seed"]), metadata["config"])
        reproduced_hash = canonical_hash(reproduced)
        check(
            reproduced_hash == metadata["episode_spec_sha256"],
            "episode seed does not reproduce identical specification",
        )
        check(reproduced["objects"] == object_rows, "reproduced clutter differs from objects.json")
    except Exception as exc:  # report rather than abort the rest of the dataset
        errors.append(f"{prefix}: reproducibility check raised {exc!r}")

    stats = {
        "frame_count": count,
        "episode_seed": metadata.get("episode_seed"),
        "object_count": len(object_rows),
        "mask_classes_seen": sorted(mask_classes),
        "min_positive_depth_fraction": min_positive,
        "objects_sha256": canonical_hash(object_rows),
    }
    return errors, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default=str(PROJECT_ROOT / "dataset"))
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    episodes = sorted(path for path in root.glob("episode_*") if path.is_dir())
    errors: list[str] = []
    episode_stats = []
    if not episodes:
        errors.append(f"No episode directories found in {root}")
    for episode in episodes:
        episode_errors, stats = validate_episode(episode)
        errors.extend(episode_errors)
        if stats:
            episode_stats.append({"episode": episode.name, **stats})
    seeds = [row["episode_seed"] for row in episode_stats]
    object_hashes = [row["objects_sha256"] for row in episode_stats]
    if len(seeds) > 1:
        if len(set(seeds)) != len(seeds):
            errors.append("Episode seeds are not unique")
        if len(set(object_hashes)) != len(object_hashes):
            errors.append("Different episode seeds did not produce different clutter")
    report = {
        "status": "PASS" if not errors else "FAIL",
        "dataset": str(root),
        "episode_count": len(episodes),
        "validated_frames": sum(row.get("frame_count", 0) for row in episode_stats),
        "errors": errors,
        "episodes": episode_stats,
    }
    report_path = Path(args.report).resolve() if args.report else root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
