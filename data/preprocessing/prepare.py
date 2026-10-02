"""Adapt existing rendered episodes and cache the C++ IMF output once."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from common.registry import (
    ANALYTICAL_CHANNELS,
    CHANNEL_REGISTRY_VERSION,
    DATASET_MANIFEST_VERSION,
    POI_CLASSES,
    STATE_NAMES,
    TRAINING_STATISTICS_VERSION,
)
from common.feature_spec import FEATURE_SPEC_VERSION, feature_specification
from common.runtime import file_sha256, save_json, stable_hash
from data.adapter.dataset import validate_manifest
from data.preprocessing.targets import targets_from_sources


SENSOR_COLUMNS = (
    "imu_ax",
    "imu_ay",
    "imu_az",
    "wx",
    "wy",
    "wz",
    "roll",
    "pitch",
    "yaw",
    "mx",
    "my",
    "mz",
    "ultrasonic_range_m",
)


def _relative(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _sensor_value(row: dict[str, str], name: str) -> tuple[float, int]:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        return 0.0, 0
    return (value, 1) if math.isfinite(value) else (0.0, 0)


def create_calibration(dataset_root: Path, output: Path) -> dict[str, Any]:
    """Recover one canonical pinhole camera calibration from Blender metadata."""
    episodes = sorted(dataset_root.glob("episode_*"))
    if not episodes:
        raise FileNotFoundError(f"No rendered episodes under {dataset_root}")
    metadata = json.loads((episodes[0] / "metadata.json").read_text(encoding="utf-8"))
    width, height = map(int, metadata["image_resolution"])
    fov_degrees = float(metadata["render"]["camera_fov_degrees"])
    for episode in episodes:
        current = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        current_fov = float(current["render"]["camera_fov_degrees"])
        if current["image_resolution"] != [width, height] or current_fov != fov_degrees:
            raise ValueError("Episodes do not share the canonical camera calibration")

    focal = width / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "%YAML:1.0\n---\n"
        f"camera: {{ width: {width}, height: {height}, fx: {focal:.12g}, "
        f"fy: {focal:.12g}, cx: {center_x:.12g}, cy: {center_y:.12g} }}\n"
        "distortion: [ 0., 0., 0., 0., 0. ]\n",
        encoding="utf-8",
    )
    record = {
        "id": file_sha256(output),
        "width": width,
        "height": height,
        "fx": focal,
        "fy": focal,
        "cx": center_x,
        "cy": center_y,
        "horizontal_fov_degrees": fov_degrees,
        "fov_convention": "Blender horizontal camera angle for 4:3 sensor fit",
        "camera_coordinates": "OpenCV x right, y down, z forward after exporter conversion",
        "distortion": "none (ideal Blender pinhole)",
    }
    save_json(output.with_suffix(".json"), record)
    return record


def build_manifests(dataset_root: Path, manifest_dir: Path, seed: int = 24051991) -> dict[str, Any]:
    """Write deterministic episode-grouped manifests without copying frames."""
    episodes = sorted(dataset_root.glob("episode_*"))
    if not episodes:
        raise FileNotFoundError(f"No episode_* directories under {dataset_root}")
    calibration_path = manifest_dir.parent / "calibration" / "blender_camera.yaml"
    calibration = create_calibration(dataset_root, calibration_path)

    generator = np.random.default_rng(seed)
    shuffled = list(episodes)
    generator.shuffle(shuffled)
    train_count = round(len(shuffled) * 0.70)
    validation_count = round(len(shuffled) * 0.15)
    assignments = {
        episode.name: (
            "train"
            if index < train_count
            else "val"
            if index < train_count + validation_count
            else "test"
        )
        for index, episode in enumerate(shuffled)
    }

    manifest_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for episode in episodes:
        sensors = _read_rows(episode / "sensors_clean.csv")
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        split = assignments[episode.name]
        rgb_files = sorted((episode / "rgb").glob("*.jpg"))
        semantic_files = sorted((episode / "segmentation").glob("*.png"))
        landing_files = sorted((episode / "landing_suitability").glob("*.png"))
        frame_count = int(metadata["frame_count"])
        if not (
            len(rgb_files)
            == len(semantic_files)
            == len(landing_files)
            == len(sensors)
            == frame_count
        ):
            raise ValueError(f"Episode {episode.name} has an incomplete RGB/label/sensor sequence")

        for index, (rgb, semantic, landing, sensor_row) in enumerate(
            zip(rgb_files, semantic_files, landing_files, sensors)
        ):
            state_pairs = [_sensor_value(sensor_row, name) for name in SENSOR_COLUMNS]
            state_values = [value for value, _ in state_pairs]
            state_validity = [valid for _, valid in state_pairs]
            records.append(
                {
                    "sample_id": f"{episode.name}:{index:06d}",
                    "episode_id": episode.name,
                    "environment_id": f"layout:{metadata['episode_spec_sha256']}",
                    "trajectory_id": f"{episode.name}:{metadata['trajectory_type']}",
                    "frame_index": index,
                    "timestamp": float(sensor_row["timestamp_s"]),
                    "rgb_path": _relative(rgb, manifest_dir),
                    "previous_rgb_path": _relative(rgb_files[max(index - 1, 0)], manifest_dir),
                    "depth_path": _relative(episode / "depth" / f"{index:06d}.npy", manifest_dir),
                    "semantic_path": _relative(semantic, manifest_dir),
                    "landing_source_path": _relative(landing, manifest_dir),
                    "objects_path": _relative(episode / "objects.json", manifest_dir),
                    "pose_path": _relative(episode / "poses.csv", manifest_dir),
                    "analytical_path": _relative(
                        manifest_dir.parent / "cache" / episode.name / f"frame_{index:06d}.npz",
                        manifest_dir,
                    ),
                    "calibration_path": _relative(calibration_path, manifest_dir),
                    "calibration_id": calibration["id"],
                    "state_names": list(STATE_NAMES),
                    "state_values": state_values,
                    "state_validity": state_validity,
                    "ultrasonic_axis_range_m": state_values[-1],
                    "geometry_height_m": float(sensor_row["ground_clearance_m"]),
                    "geometry_height_source": "simulator_vertical_ground_clearance",
                    "hazard_source_ids": [0, 1],
                    "landing_source_ids": [2],
                    "hazard_fraction": 0.05,
                    "landing_safe_fraction": 0.98,
                    "poi_classes": POI_CLASSES,
                    "split": split,
                    "source": "synthetic",
                    "label_provenance": "independent Blender world-truth class masks",
                }
            )

    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in records if row["split"] == split
            ),
            encoding="utf-8",
        )
    all_path = manifest_dir / "all.jsonl"
    all_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )

    hashes = {
        name: file_sha256(manifest_dir / f"{name}.jsonl")
        for name in ("all", "train", "val", "test")
    }
    summary = {
        "schema": DATASET_MANIFEST_VERSION,
        "seed": seed,
        "episodes": len(episodes),
        "frames": len(records),
        "split_episodes": {
            split: sum(value == split for value in assignments.values())
            for split in ("train", "val", "test")
        },
        "split_frames": {
            split: sum(row["split"] == split for row in records)
            for split in ("train", "val", "test")
        },
        "hashes": hashes,
        "calibration": calibration,
    }
    save_json(manifest_dir / "manifest_lock.json", summary)
    return summary


def compute_training_statistics(manifest: Path) -> dict[str, Any]:
    """Calculate positive weights from training labels only."""
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    class_count = len(POI_CLASSES)
    positives = {
        "hazard": 0,
        "landing": 0,
        "poi": np.zeros(class_count, dtype=np.int64),
    }
    totals = {"hazard": 0, "landing": 0, "poi": np.zeros(class_count, dtype=np.int64)}
    training_samples = 0
    for record in records:
        if record["split"] != "train":
            continue
        training_samples += 1
        landing_ids = np.asarray(
            Image.open((manifest.parent / record["landing_source_path"]).resolve()).convert("L")
        )
        semantic_ids = np.asarray(
            Image.open((manifest.parent / record["semantic_path"]).resolve()).convert("L")
        )
        hazard, landing, poi = targets_from_sources(landing_ids, semantic_ids)
        positives["hazard"] += int(hazard.sum())
        positives["landing"] += int(landing.sum())
        positives["poi"] += poi.sum(axis=(1, 2)).astype(np.int64)
        totals["hazard"] += hazard.size
        totals["landing"] += landing.size
        totals["poi"] += poi.shape[1] * poi.shape[2]

    def positive_weight(positive: int, total: int) -> float:
        return min(50.0, max(1.0, (total - positive) / max(positive, 1)))

    result = {
        "schema": TRAINING_STATISTICS_VERSION,
        "source_split": "train",
        "manifest_path": str(manifest),
        "manifest_hash": file_sha256(manifest),
        "train_split_hash": stable_hash(
            [row["sample_id"] for row in records if row["split"] == "train"]
        ),
        "statistic_version": "target-positive-weights-v2",
        "samples": training_samples,
        "positive_cells": {
            "hazard": positives["hazard"],
            "landing": positives["landing"],
            "poi": positives["poi"].tolist(),
        },
        "total_cells": {
            "hazard": totals["hazard"],
            "landing": totals["landing"],
            "poi": totals["poi"].tolist(),
        },
        "positive_weights": {
            "hazard": positive_weight(positives["hazard"], totals["hazard"]),
            "landing": positive_weight(positives["landing"], totals["landing"]),
            "poi": [
                positive_weight(int(pos), int(total))
                for pos, total in zip(positives["poi"], totals["poi"])
            ],
        },
    }
    save_json(manifest.parent / "training_statistics.json", result)
    return result


def read_cpp_export(path: Path):
    """Read the versioned FileStorage output emitted by the C++ IMF executable."""
    import cv2

    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"Cannot read C++ analytical export: {path}")
    analytical = storage.getNode("analytical")
    features = analytical.getNode("tensor").mat()
    validity = analytical.getNode("valid").mat()
    name_node = analytical.getNode("channel_names")
    names = [name_node.at(index).string() for index in range(name_node.size())]
    pose_node = analytical.getNode("camera_to_local_map")
    pose = pose_node.mat() if not pose_node.empty() else np.empty((0, 0), np.float64)
    metadata = {
        "schema_version": int(storage.getNode("schema_version").real()),
        "timestamp": float(storage.getNode("timestamp_s").real()),
        "geometry_valid": bool(analytical.getNode("geometry_valid").real()),
        "dense_flow_valid": bool(analytical.getNode("dense_flow_valid").real()),
        "depth_scale_m": float(analytical.getNode("depth_scale_m").real()),
        "camera_to_local_map": pose,
    }
    storage.release()
    if features.ndim == 4:
        features, validity = features[0], validity[0]
    return features, validity, names, metadata


def candidate_depth_cache_path(manifest_dir: Path, record: dict) -> Path:
    analytical = (manifest_dir / record["analytical_path"]).resolve()
    return analytical.with_name(analytical.stem + ".target_depth.npz")


def ensure_candidate_depth_cache(manifest_dir: Path, record: dict) -> bool:
    """Cache sparse 32×32 simulator depth targets used only for token supervision.

    This is derived from the already-rendered depth pass. It is not read by the
    model or predicted-map update. Sampling the center of each output cell keeps
    this inexpensive and preserves metric camera-Z units.
    """
    source_value = record.get("depth_path")
    if not source_value:
        return False
    source = (manifest_dir / source_value).resolve()
    destination = candidate_depth_cache_path(manifest_dir, record)
    stat = source.stat()
    signature = stable_hash(
        {
            "version": "candidate-depth-center-grid-v1",
            "sample_id": record["sample_id"],
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
        }
    )
    if destination.is_file():
        try:
            with np.load(destination, allow_pickle=False) as cache:
                if str(cache["signature"].item()) == signature:
                    return False
        except (OSError, KeyError, ValueError):
            pass

    depth = np.load(source, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2-D camera-Z depth image: {source}")
    height, width = depth.shape
    xs = np.rint((np.arange(32) + 0.5) * width / 32.0 - 0.5).astype(np.int64)
    ys = np.rint((np.arange(32) + 0.5) * height / 32.0 - 0.5).astype(np.int64)
    grid = np.asarray(depth[np.ix_(ys, xs)], dtype=np.float32)
    valid = np.isfinite(grid) & (grid > 0.0)
    grid[~valid] = 0.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        depth_m=grid,
        validity=valid.astype(np.uint8),
        signature=np.asarray(signature),
        source_units=np.asarray("camera_z_metres"),
    )
    os.replace(temporary, destination)
    return True


def ensure_candidate_depth_caches(manifest: Path, progress=None, episode_ids=None) -> int:
    """Create/update compact token-supervision depth grids without rerendering."""
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if episode_ids is not None:
        allowed = set(episode_ids)
        records = [record for record in records if record["episode_id"] in allowed]
    created = 0
    for index, record in enumerate(records, start=1):
        created += int(ensure_candidate_depth_cache(manifest.parent, record))
        if progress:
            progress(index, len(records))
    return created


def write_sequence(records: list[dict], manifest_dir: Path, path: Path) -> None:
    """Write timestamped inputs for the existing stateful C++ extractor."""
    frames = []
    for record in records:
        # The legacy C++ API calls its vertical geometry-height sample altitude_m.
        # It is simulator ground clearance here, not the ultrasonic axis range.
        frames.append(
            {
                "image": str((manifest_dir / record["rgb_path"]).resolve()),
                "timestamp_s": record["timestamp"],
                "altitude_m": record["geometry_height_m"],
                "range_timestamp_s": record["timestamp"],
                "imu": [
                    record["state_values"][6],
                    record["state_values"][7],
                    record["state_values"][8],
                ],
                "imu_timestamp_s": record["timestamp"],
            }
        )
    path.write_text(json.dumps({"frames": frames}), encoding="utf-8")


def cache_episode(
    records: list[dict],
    manifest_dir: Path,
    executable: Path,
    config: Path,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Extract one sequence to a bounded temporary folder, then atomically cache frames."""
    records = sorted(records, key=lambda row: row["frame_index"])
    if not records:
        return 0
    target_dir = (manifest_dir / records[0]["analytical_path"]).resolve().parent
    marker = target_dir / "complete.json"
    source_signature = []
    for record in records:
        source = (manifest_dir / record["rgb_path"]).resolve()
        stat = source.stat()
        source_signature.append(
            (
                record["sample_id"],
                stat.st_size,
                stat.st_mtime_ns,
                record["timestamp"],
                record["state_values"],
                record["state_validity"],
                record["geometry_height_m"],
            )
        )
    signature = stable_hash(
        {
            "executable": file_sha256(executable),
            "config": file_sha256(config),
            "calibration": records[0]["calibration_id"],
            "registry": CHANNEL_REGISTRY_VERSION,
            "feature_spec_version": FEATURE_SPEC_VERSION,
            "feature_spec_hash": stable_hash(feature_specification()),
            "sources": source_signature,
        }
    )
    cached_files = [(manifest_dir / row["analytical_path"]).resolve() for row in records]
    complete = marker.is_file() and all(path.is_file() for path in cached_files)
    if (
        complete
        and not force
        and json.loads(marker.read_text(encoding="utf-8")).get("signature") == signature
    ):
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="imagination_extract_") as temporary:
        temporary_dir = Path(temporary)
        sequence_path = temporary_dir / "sequence.json"
        export_dir = temporary_dir / "exports"
        write_sequence(records, manifest_dir, sequence_path)
        subprocess.run(
            [
                str(executable),
                "pre_extract",
                str(sequence_path),
                str(export_dir),
                "--sequence",
                "--config",
                str(config),
                "--calibration",
                str((manifest_dir / records[0]["calibration_path"]).resolve()),
                "--frames",
                str(len(records)),
                "--tensor-data",
            ],
            check=True,
        )
        for position, record in enumerate(records):
            features, validity, names, metadata = read_cpp_export(
                export_dir / f"frame_{position}.yml.gz"
            )
            if tuple(names) != ANALYTICAL_CHANNELS or features.shape != (28, 32, 32):
                raise ValueError(
                    "C++ analytical channel registry or tensor shape differs from the frozen contract"
                )
            destination = (manifest_dir / record["analytical_path"]).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary_cache = destination.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary_cache,
                features=features.astype(np.float32),
                validity=(validity > 0).astype(np.uint8),
                channel_names=np.asarray(names),
                registry_version=np.asarray(CHANNEL_REGISTRY_VERSION),
                feature_spec_version=np.asarray(FEATURE_SPEC_VERSION),
                cpp_schema_version=np.asarray(metadata["schema_version"]),
                timestamp=np.asarray(metadata["timestamp"]),
                geometry_valid=np.asarray(metadata["geometry_valid"]),
                dense_flow_valid=np.asarray(metadata["dense_flow_valid"]),
                depth_scale_m=np.asarray(metadata["depth_scale_m"]),
                camera_to_local_map=metadata["camera_to_local_map"],
                cache_signature=np.asarray(signature),
                source_sample_id=np.asarray(record["sample_id"]),
            )
            os.replace(temporary_cache, destination)
            if progress:
                progress(position + 1, len(records))
    save_json(marker, {"signature": signature, "frames": len(records)})
    return len(records)


def cache_manifest(
    manifest: Path,
    executable: Path,
    config: Path,
    limit_episodes: int | None = None,
    progress=None,
    force: bool = False,
) -> int:
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episodes: dict[str, list[dict]] = {}
    for record in records:
        episodes.setdefault(record["episode_id"], []).append(record)
    created_total = 0
    selected = sorted(episodes)
    if limit_episodes is not None:
        selected = selected[:limit_episodes]
    ensure_candidate_depth_caches(manifest, episode_ids=selected)
    total_frames = sum(len(episodes[episode]) for episode in selected)
    completed_frames = 0
    for episode in selected:
        episode_records = episodes[episode]

        def episode_progress(done: int, _episode_total: int) -> None:
            if progress:
                progress(completed_frames + done, total_frames)

        created = cache_episode(
            episode_records,
            manifest.parent,
            executable,
            config,
            force,
            episode_progress,
        )
        created_total += created
        if created == 0 and progress:
            progress(completed_frames + len(episode_records), total_frames)
        completed_frames += len(episode_records)
    return created_total


def generate_smoke_dataset(root: Path, episodes_per_split: int = 8) -> Path:
    """Create disposable 16-frame-per-split fixtures; never research data."""
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    calibration = root / "calibration.yaml"
    calibration.write_text(
        "%YAML:1.0\n---\ncamera: { width: 640, height: 480, fx: 440., fy: 440., cx: 319.5, cy: 239.5 }\n",
        encoding="utf-8",
    )
    random = np.random.default_rng(4)
    records = []
    for split in ("train", "val", "test"):
        for episode_index in range(episodes_per_split):
            episode_id = f"{split}_{episode_index}"
            directory = root / episode_id
            directory.mkdir(parents=True, exist_ok=True)
            for frame_index in range(2):
                rgb = (random.random((64, 64, 3)) * 255).astype(np.uint8)
                semantic = np.zeros((64, 64), dtype=np.uint8)
                semantic[12:24, 10:20] = 3
                landing = np.full((64, 64), 2, dtype=np.uint8)
                landing[:24] = 0
                landing[24:32] = 1
                rgb_path = directory / f"rgb_{frame_index}.jpg"
                semantic_path = directory / f"sem_{frame_index}.png"
                landing_path = directory / f"landing_{frame_index}.png"
                Image.fromarray(rgb).save(rgb_path)
                Image.fromarray(semantic).save(semantic_path)
                Image.fromarray(landing).save(landing_path)
                analytical_path = directory / f"analytic_{frame_index}.npz"
                np.savez_compressed(
                    analytical_path,
                    features=np.zeros((28, 32, 32), dtype=np.float32),
                    validity=np.ones((28, 32, 32), dtype=np.uint8),
                    channel_names=np.asarray(ANALYTICAL_CHANNELS),
                    registry_version=np.asarray(CHANNEL_REGISTRY_VERSION),
                    feature_spec_version=np.asarray(FEATURE_SPEC_VERSION),
                    cpp_schema_version=np.asarray(1),
                )
                records.append(
                    {
                        "sample_id": f"{episode_id}:{frame_index}",
                        "episode_id": episode_id,
                        "environment_id": episode_id,
                        "trajectory_id": episode_id,
                        "frame_index": frame_index,
                        "timestamp": frame_index * 0.1,
                        "rgb_path": str(rgb_path),
                        "previous_rgb_path": str(directory / f"rgb_{max(0, frame_index - 1)}.jpg"),
                        "semantic_path": str(semantic_path),
                        "landing_source_path": str(landing_path),
                        "analytical_path": str(analytical_path),
                        "calibration_path": str(calibration),
                        "calibration_id": "smoke",
                        "state_names": list(STATE_NAMES),
                        "state_values": [0.0] * 13,
                        "state_validity": [0] * 13,
                        "ultrasonic_axis_range_m": 0.0,
                        "geometry_height_m": 2.0,
                        "geometry_height_source": "simulator_vertical_ground_clearance",
                        "poi_classes": POI_CLASSES,
                        "hazard_fraction": 0.05,
                        "landing_safe_fraction": 0.98,
                        "split": split,
                        "source": "synthetic-smoke",
                    }
                )
            save_json(
                directory / "complete.json",
                {"signature": stable_hash(episode_id), "frames": 2},
            )

    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in records if row["split"] == split),
            encoding="utf-8",
        )
    all_path = manifest_dir / "all.jsonl"
    all_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return all_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/dataset")
    parser.add_argument("--manifests", default="data/manifests")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--executable", default="artifacts/build/imagination")
    parser.add_argument("--config", default="data/configs/pre_extraction.yaml")
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = build_manifests(Path(args.dataset), Path(args.manifests))
    print(json.dumps(summary, indent=2))
    validate_manifest(
        Path(args.manifests) / "all.jsonl", check_files=True, require_analytical=False
    )
    if args.cache:
        count = cache_manifest(
            Path(args.manifests) / "all.jsonl",
            Path(args.executable),
            Path(args.config),
            args.limit_episodes,
            force=args.force,
        )
        print(f"New analytical cache frames: {count}")


if __name__ == "__main__":
    main()
