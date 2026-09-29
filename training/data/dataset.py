"""Dataset contract, channel checks, and sequence-level leakage validation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.common import (
    ANALYTICAL_CHANNELS,
    CHANNEL_REGISTRY_VERSION,
    DEFAULT_STATE_NAMES,
)


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
    if not records:
        raise ValueError("Dataset manifest is empty")
    return records


def _path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def verify_split_leakage(records: list[dict[str, Any]]) -> None:
    """Reject correlated sequence identities appearing in multiple splits."""
    owners: dict[tuple[str, str], str] = {}
    for record in records:
        split = record["split"]
        identities = {
            ("episode", str(record["episode_id"])),
            ("trajectory", str(record["trajectory_id"])),
        }
        environment = str(record.get("environment_id", ""))
        if environment:
            identities.add(("environment", environment))
        for identity in identities:
            previous = owners.setdefault(identity, split)
            if previous != split:
                raise ValueError(
                    f"Split leakage: {identity[0]} {identity[1]} occurs in "
                    f"both {previous} and {split}"
                )


def validate_manifest(path: str | Path, check_files: bool = True) -> dict[str, int]:
    manifest = Path(path)
    base = manifest.parent
    records = load_manifest(manifest)
    verify_split_leakage(records)
    seen_ids: set[str] = set()
    episodes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_counts: dict[str, int] = defaultdict(int)

    for record in records:
        required = {
            "sample_id", "episode_id", "environment_id", "trajectory_id",
            "frame_index", "timestamp", "rgb_path", "analytical_path",
            "hazard_path", "waypoint", "split", "source", "label_provenance",
            "calibration_id",
        }
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{record.get('sample_id', '?')} lacks {sorted(missing)}")
        sample_id = str(record["sample_id"])
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        if record["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {sample_id}: {record['split']}")
        waypoint = np.asarray(record["waypoint"], dtype=np.float32)
        if waypoint.shape != (5,) or not np.isfinite(waypoint).all():
            raise ValueError(f"Invalid five-value waypoint for {sample_id}")
        if abs(float(np.linalg.norm(waypoint[3:5])) - 1) > 1e-3:
            raise ValueError(f"Waypoint yaw sine/cosine is not normalized: {sample_id}")
        if np.linalg.norm(waypoint[:3]) > float(record.get("maximum_waypoint_m", 100)):
            raise ValueError(f"Implausible waypoint distance: {sample_id}")

        episodes[str(record["episode_id"])].append(record)
        split_counts[record["split"]] += 1
        if check_files:
            _validate_files(base, record)

    for episode_id, frames in episodes.items():
        ordered = sorted(frames, key=lambda item: int(item["frame_index"]))
        frame_indices = [int(item["frame_index"]) for item in ordered]
        if any(second != first + 1 for first, second in zip(frame_indices, frame_indices[1:])):
            raise ValueError(f"Missing or repeated frame index in episode {episode_id}")
        timestamps = [float(item["timestamp"]) for item in ordered]
        if any(not math.isfinite(value) for value in timestamps):
            raise ValueError(f"Nonfinite timestamp in episode {episode_id}")
        if any(second <= first for first, second in zip(timestamps, timestamps[1:])):
            raise ValueError(f"Nonmonotonic timestamps in episode {episode_id}")
        calibrations = {item["calibration_id"] for item in ordered}
        if len(calibrations) != 1:
            raise ValueError(f"Calibration mismatch in episode {episode_id}")
    return dict(split_counts)


def _validate_files(base: Path, record: dict[str, Any]) -> None:
    sample_id = record["sample_id"]
    rgb_path = _path(base, record["rgb_path"])
    try:
        with Image.open(rgb_path) as image:
            image.verify()
    except Exception as error:
        raise ValueError(f"Corrupt RGB image for {sample_id}: {rgb_path}") from error

    analytical_path = _path(base, record["analytical_path"])
    with np.load(analytical_path, allow_pickle=False) as data:
        features = data["features"]
        validity = data["validity"]
        channels = tuple(str(value) for value in data["channel_names"].tolist())
        registry = str(data["registry_version"].item())
        schema_version = int(data["cpp_schema_version"].item())
        if features.shape != (28, 32, 32) or features.dtype != np.float32:
            raise ValueError(f"Wrong analytical tensor for {sample_id}: {features.shape}")
        if validity.shape != features.shape:
            raise ValueError(f"Wrong validity tensor for {sample_id}: {validity.shape}")
        if channels != ANALYTICAL_CHANNELS or registry != CHANNEL_REGISTRY_VERSION or schema_version != 1:
            raise ValueError(f"Analytical channel registry mismatch for {sample_id}")
        if not np.isfinite(features).all():
            raise ValueError(f"NaN/inf analytical data for {sample_id}")

    hazard = _load_map(_path(base, record["hazard_path"]), binary=True)
    if hazard.shape != (1, 32, 32):
        raise ValueError(f"Wrong hazard shape for {sample_id}: {hazard.shape}")
    if record.get("expect_hazard", False) and not hazard.any():
        raise ValueError(f"Expected hazards but label is empty for {sample_id}")
    validity_path = record.get("hazard_validity_path")
    if validity_path and _load_map(_path(base, validity_path), binary=True).shape != (1, 32, 32):
        raise ValueError(f"Wrong hazard validity shape for {sample_id}")


def _load_map(path: Path, binary: bool = False) -> np.ndarray:
    if path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            values = data["values"] if "values" in data else data[data.files[0]]
    else:
        with Image.open(path) as image:
            values = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.NEAREST))
    if values.ndim == 2:
        values = values[None]
    values = values.astype(np.float32)
    if binary:
        values = (values > (0.5 if values.max(initial=0) <= 1 else 127)).astype(np.float32)
    return values


class ImaginationDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str, mode: str = "analytical"):
        if mode not in {"analytical", "baseline"}:
            raise ValueError("Dataset mode must be analytical or baseline")
        self.manifest = Path(manifest)
        self.base = self.manifest.parent
        self.mode = mode
        all_records = load_manifest(self.manifest)
        verify_split_leakage(all_records)
        self.records = [record for record in all_records if record["split"] == split]
        if not self.records:
            raise ValueError(f"No samples for split {split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample: dict[str, Any] = {}
        if self.mode == "analytical":
            with np.load(_path(self.base, record["analytical_path"]), allow_pickle=False) as data:
                channels = tuple(str(value) for value in data["channel_names"].tolist())
                registry = str(data["registry_version"].item())
                schema_version = int(data["cpp_schema_version"].item())
                if channels != ANALYTICAL_CHANNELS or registry != CHANNEL_REGISTRY_VERSION or schema_version != 1:
                    raise ValueError(f"Channel registry mismatch: {record['sample_id']}")
                sample["analytical"] = torch.from_numpy(data["features"].astype(np.float32))
                sample["validity"] = torch.from_numpy(
                    (data["validity"] > 0).astype(np.float32)
                )
        else:
            with Image.open(_path(self.base, record["rgb_path"])) as image:
                image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255
            sample["rgb"] = torch.from_numpy(rgb.copy())

        state_names = tuple(record.get("state_names", DEFAULT_STATE_NAMES))
        values = np.asarray(record.get("state_values", [0] * len(state_names)), dtype=np.float32)
        validity = np.asarray(record.get("state_validity", [0] * len(state_names)), dtype=np.float32)
        if values.shape != validity.shape or values.shape != (len(state_names),):
            raise ValueError(f"Invalid state vectors: {record['sample_id']}")
        sample["vehicle_state"] = torch.from_numpy(values)
        sample["state_validity"] = torch.from_numpy(validity)
        sample["hazard_target"] = torch.from_numpy(
            _load_map(_path(self.base, record["hazard_path"]), binary=True)
        )
        validity_path = record.get("hazard_validity_path")
        sample["hazard_validity"] = torch.from_numpy(
            _load_map(_path(self.base, validity_path), binary=True)
            if validity_path else np.ones((1, 32, 32), dtype=np.float32)
        )
        sample["waypoint_target"] = torch.tensor(record["waypoint"], dtype=torch.float32)
        sample["metadata"] = {
            key: record[key] for key in (
                "sample_id", "episode_id", "environment_id", "trajectory_id",
                "frame_index", "timestamp", "source", "label_provenance",
                "calibration_id",
            )
        }
        return sample
