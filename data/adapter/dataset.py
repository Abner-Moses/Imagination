"""Dataset adapter for rendered episodes and cached IMF observations."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common.feature_spec import FEATURE_SPEC_VERSION
from common.registry import (
    ANALYTICAL_CHANNELS,
    CANDIDATE_FEATURE_NAMES,
    CHANNEL_REGISTRY_VERSION,
    MAX_CANDIDATES,
    POI_CLASSES as DEFAULT_POI_CLASSES,
    RELATIONAL_NAMES,
    STATE_NAMES,
)
from data.preprocessing.targets import (
    landing_class_grid,
    scene_verdict,
    semantic_grid,
    targets_from_sources,
)


POI_CLASSES = DEFAULT_POI_CLASSES
HAZARD_SOURCE_IDS = (0, 1)
LANDING_SOURCE_IDS = (2,)


@lru_cache(maxsize=8)
def _load_manifest_cached(path: str, modified_ns: int, size: int) -> tuple[dict, ...]:
    del modified_ns, size  # Cache-key fields invalidate changed manifests.
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL line {number}: {error}") from error
    if not records:
        raise ValueError("Dataset manifest is empty")
    return tuple(records)


def load_manifest(path: str | Path) -> list[dict]:
    """Load a manifest once per process and invalidate the cache on file change."""
    manifest = Path(path).resolve()
    stat = manifest.stat()
    return list(_load_manifest_cached(str(manifest), stat.st_mtime_ns, stat.st_size))


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def verify_split_leakage(records: list[dict]) -> None:
    owners: dict[tuple[str, str], str] = {}
    for record in records:
        identities = (
            ("episode", record["episode_id"]),
            ("trajectory", record["trajectory_id"]),
            ("environment", record.get("environment_id")),
        )
        for kind, value in identities:
            if not value:
                continue
            key = kind, str(value)
            previous = owners.setdefault(key, record["split"])
            if previous != record["split"]:
                raise ValueError(
                    f"Split leakage: {kind} {value} occurs in both {previous} and {record['split']}"
                )


def _read_ids(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _load_analytical(path: Path, sample_id: str = "?") -> dict[str, np.ndarray | float]:
    """Load and validate one cached IMF observation with a single NPZ open."""
    with np.load(path, allow_pickle=False) as data:
        features = data["features"].astype(np.float32, copy=True)
        validity = (data["validity"] > 0).astype(np.float32, copy=True)
        names = tuple(str(value) for value in data["channel_names"].tolist())
        registry_version = str(data["registry_version"].item())
        feature_spec_version = (
            str(data["feature_spec_version"].item()) if "feature_spec_version" in data else None
        )
        depth_scale_m = float(data["depth_scale_m"].item()) if "depth_scale_m" in data else 1.0
        pose = (
            data["camera_to_local_map"].astype(np.float64, copy=True)
            if "camera_to_local_map" in data
            else np.empty((0, 0), dtype=np.float64)
        )

    if features.shape != (28, 32, 32) or validity.shape != features.shape:
        raise ValueError(f"Wrong analytical tensor for {sample_id}: {features.shape}")
    if names != ANALYTICAL_CHANNELS or registry_version != CHANNEL_REGISTRY_VERSION:
        raise ValueError(f"Analytical channel registry mismatch for {sample_id}")
    if feature_spec_version not in (None, FEATURE_SPEC_VERSION):
        raise ValueError(f"Analytical normalization contract mismatch for {sample_id}")
    if not np.isfinite(features).all():
        raise ValueError(f"NaN/inf analytical data for {sample_id}")
    return {
        "features": features,
        "validity": validity,
        "depth_scale_m": depth_scale_m,
        "camera_to_local_map": pose,
    }


def validate_analytical(path: Path, sample_id: str = "?") -> None:
    _load_analytical(path, sample_id)


def validate_manifest(
    path: str | Path,
    check_files: bool = True,
    require_analytical: bool = True,
    require_candidate_depth: bool = False,
) -> dict[str, int]:
    manifest = Path(path)
    records = load_manifest(manifest)
    verify_split_leakage(records)
    seen: set[str] = set()
    episodes: defaultdict[str, list[dict]] = defaultdict(list)
    counts: defaultdict[str, int] = defaultdict(int)
    required = {
        "sample_id",
        "episode_id",
        "environment_id",
        "trajectory_id",
        "frame_index",
        "timestamp",
        "rgb_path",
        "semantic_path",
        "landing_source_path",
        "analytical_path",
        "split",
        "source",
        "calibration_id",
        "calibration_path",
        "state_names",
        "state_values",
        "state_validity",
        "geometry_height_m",
        "geometry_height_source",
    }
    classes_seen: set[int] = set()
    hazard_values: set[float] = set()
    landing_values: set[float] = set()
    for record in records:
        missing = required - record.keys()
        if missing:
            raise ValueError(f"{record.get('sample_id', '?')} lacks {sorted(missing)}")
        sample_id = record["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)
        counts[record["split"]] += 1
        episodes[record["episode_id"]].append(record)
        if record["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {record['split']}")
        if (
            tuple(record["state_names"]) != STATE_NAMES
            or len(record["state_values"]) != 13
            or len(record["state_validity"]) != 13
        ):
            raise ValueError(f"Invalid 13-value state contract: {sample_id}")
        if record["geometry_height_source"] != "simulator_vertical_ground_clearance":
            raise ValueError(
                "Geometry height must be explicit; ultrasonic axis range is not altitude"
            )
        if not check_files:
            continue
        base = manifest.parent
        for key in (
            "rgb_path",
            "semantic_path",
            "landing_source_path",
            "calibration_path",
        ):
            if not resolve_path(base, record[key]).is_file():
                raise ValueError(f"Missing {key}: {sample_id}")
        if require_analytical:
            validate_analytical(resolve_path(base, record["analytical_path"]), sample_id)
        if require_candidate_depth and record.get("depth_path"):
            analytical = resolve_path(base, record["analytical_path"])
            target_depth = analytical.with_name(analytical.stem + ".target_depth.npz")
            if not target_depth.is_file():
                raise ValueError(f"Candidate supervision depth cache missing: {sample_id}")
            with np.load(target_depth, allow_pickle=False) as depth_cache:
                if depth_cache["depth_m"].shape != (32, 32) or depth_cache["validity"].shape != (
                    32,
                    32,
                ):
                    raise ValueError(f"Invalid candidate depth grid: {sample_id}")
                if str(depth_cache["source_units"].item()) != "camera_z_metres":
                    raise ValueError(f"Invalid candidate depth units: {sample_id}")

    for episode, frames in episodes.items():
        ordered = sorted(frames, key=lambda item: item["frame_index"])
        frame_ids = [item["frame_index"] for item in ordered]
        timestamps = [item["timestamp"] for item in ordered]
        if any(second != first + 1 for first, second in zip(frame_ids, frame_ids[1:])):
            raise ValueError(f"Missing/repeated frame in {episode}")
        if any(not math.isfinite(value) for value in timestamps) or any(
            second <= first for first, second in zip(timestamps, timestamps[1:])
        ):
            raise ValueError(f"Bad timestamps in {episode}")
        if not check_files:
            continue
        # Every path is checked above. Decode three samples per episode here;
        # the source validation report separately covers all rendered frames.
        sample_indices = sorted({0, (len(ordered) - 1) // 2, len(ordered) - 1})
        for index in sample_indices:
            record = ordered[index]
            rgb_path = resolve_path(manifest.parent, record["rgb_path"])
            with Image.open(rgb_path) as image:
                image.verify()
            semantic = _read_ids(resolve_path(manifest.parent, record["semantic_path"]))
            landing_ids = _read_ids(resolve_path(manifest.parent, record["landing_source_path"]))
            classes_seen.update(np.unique(semantic).tolist())
            hazard, landing, _ = targets_from_sources(landing_ids, semantic)
            hazard_values.update(np.unique(hazard).tolist())
            landing_values.update(np.unique(landing).tolist())

    if check_files and (hazard_values != {0.0, 1.0} or landing_values != {0.0, 1.0}):
        raise ValueError("Hazard and landing targets must both contain positive and negative cells")
    return {
        **dict(counts),
        "episodes": len(episodes),
        "semantic_classes_seen": len(classes_seen),
    }


def _rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        values = (
            np.asarray(
                image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ).transpose(2, 0, 1)
            / 255
        )
    return torch.from_numpy(values.copy())


def normalize_state(values: list[float]) -> torch.Tensor:
    """Apply fixed unit scales; the sensor validity mask remains separate."""
    array = np.asarray(values, dtype=np.float32).copy()
    if array.shape != (13,):
        raise ValueError("Vehicle state must have 13 sensor fields")
    scales = np.asarray(
        (
            9.81,
            9.81,
            9.81,
            math.pi,
            math.pi,
            math.pi,
            math.pi,
            math.pi,
            math.pi,
            100.0,
            100.0,
            100.0,
            20.0,
        ),
        dtype=np.float32,
    )
    return torch.from_numpy(array / scales)


@lru_cache(maxsize=8)
def _read_calibration(path: str, modified_ns: int) -> dict[str, float]:
    del modified_ns
    import cv2

    storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"Cannot read camera calibration: {path}")
    camera = storage.getNode("camera")
    calibration = {
        name: float(camera.getNode(name).real())
        for name in ("width", "height", "fx", "fy", "cx", "cy")
    }
    storage.release()
    return calibration


class ImaginationDataset(Dataset):
    """Expose one shared sample contract for RGB and analytical model families."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        mode: str = "analytical",
        poi_classes: dict[str, int] | None = None,
        include_targets: bool = True,
    ):
        aliases = {"baseline": "rgb_current"}
        self.mode = aliases.get(mode, mode)
        if self.mode not in {"analytical", "rgb_current", "rgb_temporal"}:
            raise ValueError(f"Unknown dataset mode: {mode!r}")
        self.manifest = Path(manifest)
        self.base = self.manifest.parent
        all_records = load_manifest(manifest)
        verify_split_leakage(all_records)
        self.include_targets = include_targets
        self.records = [record for record in all_records if record["split"] == split]
        self.poi_classes = poi_classes or POI_CLASSES
        if not self.records:
            raise ValueError(f"No samples for split {split}")
        calibration_path = resolve_path(self.base, self.records[0]["calibration_path"])
        self.calibration = _read_calibration(
            str(calibration_path), calibration_path.stat().st_mtime_ns
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        sample: dict = {}
        if self.mode == "analytical":
            path = resolve_path(self.base, record["analytical_path"])
            analytical = _load_analytical(path, record["sample_id"])
            sample["analytical"] = torch.from_numpy(analytical["features"])
            sample["validity"] = torch.from_numpy(analytical["validity"])
            sample["depth_scale_m"] = torch.tensor(analytical["depth_scale_m"], dtype=torch.float64)
            pose = analytical["camera_to_local_map"]
            pose_valid = pose.shape == (4, 4)
            sample["camera_to_local_map"] = (
                torch.from_numpy(pose) if pose_valid else torch.zeros(4, 4, dtype=torch.float64)
            )
            sample["camera_pose_valid"] = torch.tensor(pose_valid)
        else:
            sample["rgb"] = _rgb(resolve_path(self.base, record["rgb_path"]))
            if self.mode == "rgb_temporal":
                sample["previous_rgb"] = _rgb(resolve_path(self.base, record["previous_rgb_path"]))

        if tuple(record["state_names"]) != STATE_NAMES:
            raise ValueError(f"State registry mismatch: {record['sample_id']}")
        sample["vehicle_state"] = normalize_state(record["state_values"])
        sample["state_validity"] = torch.tensor(record["state_validity"], dtype=torch.float32)
        sample["calibration"] = torch.tensor(
            [self.calibration[name] for name in ("width", "height", "fx", "fy", "cx", "cy")],
            dtype=torch.float64,
        )

        # Inference mode deliberately avoids simulator label files.
        sample["relational"] = torch.zeros(len(RELATIONAL_NAMES), dtype=torch.float32)
        sample["relational_validity"] = torch.zeros(len(RELATIONAL_NAMES), dtype=torch.float32)
        sample["candidates"] = torch.zeros(
            MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES), dtype=torch.float32
        )
        sample["candidate_validity"] = torch.zeros(MAX_CANDIDATES, dtype=torch.float32)
        sample["candidate_feature_validity"] = torch.zeros_like(sample["candidates"])
        sample["metadata"] = {
            key: record[key]
            for key in (
                "sample_id",
                "episode_id",
                "environment_id",
                "trajectory_id",
                "frame_index",
                "timestamp",
                "source",
                "calibration_id",
            )
        }
        if not self.include_targets:
            return sample

        landing_ids = _read_ids(resolve_path(self.base, record["landing_source_path"]))
        semantic_ids = _read_ids(resolve_path(self.base, record["semantic_path"]))
        target_depth_path = resolve_path(self.base, record["analytical_path"])
        target_depth_path = target_depth_path.with_name(
            target_depth_path.stem + ".target_depth.npz"
        )
        if target_depth_path.is_file():
            with np.load(target_depth_path, allow_pickle=False) as depth_cache:
                if str(depth_cache["source_units"].item()) != "camera_z_metres":
                    raise ValueError("Candidate target depth must be camera-Z metres")
                candidate_depth = torch.from_numpy(
                    depth_cache["depth_m"].astype(np.float32, copy=True)
                )
                candidate_depth_validity = torch.from_numpy(
                    (depth_cache["validity"] > 0).astype(np.float32, copy=True)
                )
        else:
            candidate_depth = torch.zeros(32, 32, dtype=torch.float32)
            candidate_depth_validity = torch.zeros(32, 32, dtype=torch.float32)

        hazard, landing, poi = targets_from_sources(
            landing_ids,
            semantic_ids,
            self.poi_classes,
            float(record.get("hazard_fraction", 0.05)),
            float(record.get("landing_safe_fraction", 0.98)),
        )
        semantic = semantic_grid(semantic_ids)
        poi_tensor = torch.from_numpy(poi)
        sample.update(
            hazard_target=torch.from_numpy(hazard),
            hazard_validity=torch.ones(1, 32, 32),
            landing_target=torch.from_numpy(landing),
            landing_validity=torch.ones(1, 32, 32),
            landing_class_target=torch.from_numpy(landing_class_grid(landing_ids)),
            candidate_depth_target_m=candidate_depth,
            candidate_depth_validity=candidate_depth_validity,
            poi_target=poi_tensor,
            poi_validity=torch.ones_like(poi_tensor),
            semantic_target=torch.from_numpy(semantic),
            semantic_validity=torch.ones(32, 32, dtype=torch.float32),
            scene_target=torch.tensor(scene_verdict(landing_ids, semantic_ids), dtype=torch.long),
        )
        return sample
