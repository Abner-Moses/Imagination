"""Invoke the existing C++ extractor once, then build a stable NPZ cache."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from training.common import (
    ANALYTICAL_CHANNELS,
    CHANNEL_REGISTRY_VERSION,
    DEFAULT_STATE_NAMES,
    file_sha256,
    load_config,
    save_json,
    stable_hash,
)
from training.data.dataset import validate_manifest


def read_cpp_export(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Dataset preparation requires opencv-python for FileStorage") from error
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"Cannot read C++ export: {path}")
    analytical = storage.getNode("analytical")
    features = analytical.getNode("tensor").mat()
    validity = analytical.getNode("valid").mat()
    if features is not None and features.ndim == 4 and features.shape[0] == 1:
        features = features[0]
        validity = validity[0]
    names_node = analytical.getNode("channel_names")
    names = [names_node.at(i).string() for i in range(names_node.size())]
    metadata = {
        "schema_version": int(storage.getNode("schema_version").real()),
        "timestamp": float(storage.getNode("timestamp_s").real()),
        "geometry_valid": bool(analytical.getNode("geometry_valid").real()),
        "dense_flow_valid": bool(analytical.getNode("dense_flow_valid").real()),
    }
    storage.release()
    return features, validity, names, metadata


def expand_ablated_channels(
    features: np.ndarray, validity: np.ndarray, names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    if any(name not in ANALYTICAL_CHANNELS for name in names):
        raise ValueError("C++ export contains an unknown analytical channel")
    expected_subset = [name for name in ANALYTICAL_CHANNELS if name in names]
    if names != expected_subset:
        raise ValueError("C++ analytical channels are not in registry order")
    expanded = np.zeros((28, 32, 32), dtype=np.float32)
    expanded_validity = np.zeros_like(expanded, dtype=np.uint8)
    for source, name in enumerate(names):
        destination = ANALYTICAL_CHANNELS.index(name)
        expanded[destination] = features[source]
        expanded_validity[destination] = validity[source] > 0
    return expanded, expanded_validity


def waypoint_vector(value: list[float]) -> list[float]:
    if len(value) == 5:
        result = np.asarray(value, dtype=np.float32)
        norm = np.linalg.norm(result[3:5])
        if norm <= 1e-9:
            raise ValueError("Waypoint yaw vector cannot be zero")
        result[3:5] /= norm
        return result.tolist()
    if len(value) != 4:
        raise ValueError("Waypoint must be [dx,dy,dz,yaw] or five-value sine/cosine form")
    return [float(value[0]), float(value[1]), float(value[2]),
            math.sin(float(value[3])), math.cos(float(value[3]))]


def write_sequence(path: Path, frames: list[dict[str, Any]], episode: Path) -> None:
    sequence = []
    for frame in frames:
        item: dict[str, Any] = {
            "image": str((episode / frame["rgb_path"]).resolve()),
            "timestamp_s": float(frame["timestamp"]),
        }
        sensors = frame.get("sensors", {})
        if sensors.get("ultrasonic_m_valid", sensors.get("ultrasonic_valid", False)):
            item["altitude_m"] = float(sensors["ultrasonic_m"])
            item["range_timestamp_s"] = float(sensors.get("timestamp", frame["timestamp"]))
        if all(sensors.get(f"{axis}_rad_valid", sensors.get(f"{axis}_valid", False))
               for axis in ("roll", "pitch", "yaw")):
            item["imu"] = [float(sensors[f"{axis}_rad"]) for axis in ("roll", "pitch", "yaw")]
            item["imu_timestamp_s"] = float(sensors.get("timestamp", frame["timestamp"]))
        sequence.append(item)
    path.write_text(json.dumps({"frames": sequence}, indent=2), encoding="utf-8")


def state_vector(frame: dict[str, Any]) -> tuple[list[float], list[int]]:
    sensors = frame.get("sensors", {})
    values, validity = [], []
    for name in DEFAULT_STATE_NAMES:
        value = sensors.get(name)
        valid = value is not None and bool(sensors.get(f"{name}_valid", True))
        values.append(float(value) if valid else 0.0)
        validity.append(int(valid))
    return values, validity


def prepare(config_path: str | Path) -> Path:
    config = load_config(config_path)
    data = config["data"]
    raw_root = Path(data["raw_root"]).resolve()
    output_root = Path(data["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    executable = Path(data.get("extractor", "build/imagination")).resolve()
    extractor_config = Path(data.get("extractor_config", "configs/pre_extraction.yaml")).resolve()
    records: list[dict[str, Any]] = []

    for episode in sorted((raw_root / "episodes").iterdir()):
        if not episode.is_dir():
            continue
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        frames = [json.loads(line) for line in (episode / "frames.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        if not frames:
            raise ValueError(f"No frames in {episode}")
        frames.sort(key=lambda item: int(item["frame_index"]))
        episode_output = output_root / "analytical" / episode.name
        episode_output.mkdir(parents=True, exist_ok=True)
        sequence_path = episode_output / "sequence.json"
        write_sequence(sequence_path, frames, episode)
        calibration = episode / "calibration.yaml"
        command = [
            str(executable), "pre_extract", str(sequence_path), str(episode_output / "cpp"),
            "--sequence", "--config", str(extractor_config), "--calibration",
            str(calibration), "--frames", str(len(frames)), "--data",
        ]
        input_hashes = []
        for frame in frames:
            paths = {
                "rgb": episode / frame["rgb_path"],
                "hazard": episode / frame["hazard_path"],
            }
            if frame.get("hazard_validity_path"):
                paths["hazard_validity"] = episode / frame["hazard_validity_path"]
            input_hashes.append({name: file_sha256(path) for name, path in paths.items()})
        cache_signature = stable_hash({
            "frames": frames,
            "metadata": metadata,
            "input_sha256": input_hashes,
            "calibration_sha256": file_sha256(calibration),
            "extractor_sha256": file_sha256(executable),
            "extractor_config_sha256": file_sha256(extractor_config),
            "registry": CHANNEL_REGISTRY_VERSION,
        })
        cache_file = episode_output / "cache.json"
        cached = cache_file.exists() and json.loads(cache_file.read_text()).get("signature") == cache_signature
        if not cached:
            subprocess.run(command, check=True)

        for position, frame in enumerate(frames):
            export_path = episode_output / "cpp" / f"frame_{position}.yml.gz"
            features, validity, channels, cpp_metadata = read_cpp_export(export_path)
            if features.ndim != 3 or features.shape[1:] != (32, 32) or validity.shape != features.shape:
                raise ValueError(f"C++ tensor shape mismatch in {export_path}")
            computed_channels = list(channels)
            if tuple(channels) != ANALYTICAL_CHANNELS:
                if not bool(data.get("allow_feature_ablation", False)):
                    raise ValueError(f"C++ channel registry mismatch in {export_path}")
                features, validity = expand_ablated_channels(features, validity, channels)
            destination = episode_output / f"{int(frame['frame_index']):06d}.npz"
            np.savez_compressed(
                destination,
                features=features.astype(np.float32),
                validity=(validity > 0).astype(np.uint8),
                channel_names=np.asarray(ANALYTICAL_CHANNELS),
                computed_channel_names=np.asarray(computed_channels),
                registry_version=np.asarray(CHANNEL_REGISTRY_VERSION),
                cpp_schema_version=np.asarray(cpp_metadata["schema_version"]),
                preprocessing_hash=np.asarray(cache_signature),
                timestamp=np.asarray(cpp_metadata["timestamp"]),
            )
            state_values, state_validity = state_vector(frame)
            records.append({
                "sample_id": f"{episode.name}:{int(frame['frame_index']):06d}",
                "episode_id": metadata.get("episode_id", episode.name),
                "environment_id": metadata["environment_id"],
                "trajectory_id": metadata["trajectory_id"],
                "frame_index": int(frame["frame_index"]),
                "timestamp": float(frame["timestamp"]),
                "rgb_path": str((episode / frame["rgb_path"]).resolve()),
                "analytical_path": str(destination),
                "hazard_path": str((episode / frame["hazard_path"]).resolve()),
                "hazard_validity_path": str((episode / frame["hazard_validity_path"]).resolve())
                if frame.get("hazard_validity_path") else None,
                "waypoint": waypoint_vector(frame["waypoint"]),
                "state_names": list(DEFAULT_STATE_NAMES),
                "state_values": state_values,
                "state_validity": state_validity,
                "split": metadata["split"],
                "source": metadata["source"],
                "label_provenance": metadata["label_provenance"],
                "calibration_id": file_sha256(calibration),
                "calibration_path": str(calibration.resolve()),
                "has_geometry": cpp_metadata["geometry_valid"],
                "has_flow": cpp_metadata["dense_flow_valid"],
            })
        save_json(cache_file, {"signature": cache_signature, "command": command})

    manifest = output_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    validate_manifest(manifest)
    save_json(output_root / "dataset.json", {
        "channel_registry_version": CHANNEL_REGISTRY_VERSION,
        "channel_names": list(ANALYTICAL_CHANNELS),
        "manifest_sha256": file_sha256(manifest),
        "sample_count": len(records),
    })
    return manifest


def generate_smoke_dataset(output: str | Path, frames_per_split: int = 4) -> Path:
    """Create non-research synthetic data solely for pipeline tests."""
    root = Path(output).resolve()
    records = []
    rng = np.random.default_rng(7)
    for split_index, split in enumerate(("train", "val", "test")):
        episode = f"smoke_{split}"
        directory = root / episode
        directory.mkdir(parents=True, exist_ok=True)
        for frame_index in range(frames_per_split):
            rgb_path = directory / f"rgb_{frame_index:06d}.png"
            hazard_path = directory / f"hazard_{frame_index:06d}.npy"
            analytical_path = directory / f"analytical_{frame_index:06d}.npz"
            image = Image.new("RGB", (256, 256), (30 + 20 * split_index, 50, 80))
            ImageDraw.Draw(image).rectangle((60 + frame_index, 60, 130, 150), fill=(180, 90, 40))
            image.save(rgb_path)
            hazard = np.zeros((1, 32, 32), dtype=np.float32)
            hazard[:, 8:18, 8 + frame_index:18 + frame_index] = 1
            np.save(hazard_path, hazard)
            features = rng.normal(0, 0.1, (28, 32, 32)).astype(np.float32)
            validity = np.ones_like(features, dtype=np.uint8)
            np.savez_compressed(
                analytical_path, features=features, validity=validity,
                channel_names=np.asarray(ANALYTICAL_CHANNELS),
                registry_version=np.asarray(CHANNEL_REGISTRY_VERSION),
                cpp_schema_version=np.asarray(1),
                preprocessing_hash=np.asarray("synthetic-smoke-only"),
            )
            records.append({
                "sample_id": f"{episode}:{frame_index:06d}",
                "episode_id": episode,
                "environment_id": f"smoke_environment_{split}",
                "trajectory_id": f"smoke_trajectory_{split}",
                "frame_index": frame_index,
                "timestamp": frame_index / 10,
                "rgb_path": str(rgb_path),
                "analytical_path": str(analytical_path),
                "hazard_path": str(hazard_path),
                "waypoint": [0.2, 0.0, 0.0, 0.0, 1.0],
                "state_names": list(DEFAULT_STATE_NAMES),
                "state_values": [0.0] * len(DEFAULT_STATE_NAMES),
                "state_validity": [0] * len(DEFAULT_STATE_NAMES),
                "split": split,
                "source": "synthetic_smoke_test_only",
                "label_provenance": "programmatic smoke-test fixture; not research data",
                "calibration_id": f"synthetic-smoke-calibration-{split}",
            })
    manifest = root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    validate_manifest(manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--synthetic-smoke", metavar="DIRECTORY")
    parser.add_argument("--frames-per-split", type=int, default=4)
    arguments = parser.parse_args()
    if bool(arguments.config) == bool(arguments.synthetic_smoke):
        parser.error("choose exactly one of --config or --synthetic-smoke")
    manifest = prepare(arguments.config) if arguments.config else generate_smoke_dataset(
        arguments.synthetic_smoke, arguments.frames_per_split
    )
    print(manifest)


if __name__ == "__main__":
    main()
