"""World-truth supervision conversion kept separate from runtime feature inputs."""

from __future__ import annotations
import numpy as np
from PIL import Image
from common.registry import SEMANTIC_SOURCE_TO_TRAIN

POI_CLASSES = {"cone": 3, "rock": 4, "football": 5, "bag": 6, "box": 7, "pole": 8, "chair": 9}
HAZARD_SOURCE_IDS = (0, 1)
LANDING_SOURCE_IDS = (2,)


def cell_fraction(values: np.ndarray, size=32) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    h, w = values.shape
    if h % size == 0 and w % size == 0:
        return values.reshape(size, h // size, size, w // size).mean(axis=(1, 3), dtype=np.float32)
    import cv2

    return cv2.resize(values, (size, size), interpolation=cv2.INTER_AREA)


def semantic_grid(semantic_ids: np.ndarray, size=32, num_classes=6) -> np.ndarray:
    source = np.asarray(semantic_ids, dtype=np.uint8)
    scores = []
    for class_id in range(num_classes):
        ids = [
            source_id
            for source_id, target_id in SEMANTIC_SOURCE_TO_TRAIN.items()
            if target_id == class_id
        ]
        scores.append(cell_fraction(np.isin(source, ids), size))
    return np.stack(scores).argmax(axis=0).astype(np.int64)


def landing_class_grid(landing_ids: np.ndarray, size=32) -> np.ndarray:
    """Return one simulator landing class per cell (0 unsafe, 1 caution, 2 safe).

    Area fractions are compared and ties prefer the lower, more conservative
    class ID. This map is supervision only; it is never a runtime model input.
    """
    source = np.asarray(landing_ids, dtype=np.uint8)
    scores = np.stack([cell_fraction(source == class_id, size) for class_id in range(3)])
    return scores.argmax(axis=0).astype(np.int64)


def scene_verdict(landing_ids: np.ndarray, semantic_ids: np.ndarray) -> int:
    """World-truth rules: track overlap is dangerous; poor/caution surface is CAUTION."""
    landing = np.asarray(landing_ids)
    semantic = np.asarray(semantic_ids)
    if landing.ndim != 2 or semantic.ndim != 2:
        raise ValueError("Scene targets need class-ID masks")
    track_fraction = float(np.mean(semantic == 2))
    unsafe_fraction = float(np.mean(landing == 0))
    caution_fraction = float(np.mean(landing == 1))
    if track_fraction >= 0.002 or unsafe_fraction >= 0.08:
        return 2
    if caution_fraction >= 0.08:
        return 1
    return 0


def targets_from_sources(
    landing_ids: np.ndarray,
    semantic_ids: np.ndarray,
    poi_classes: dict[str, int] | None = None,
    hazard_fraction: float = 0.05,
    landing_safe_fraction: float = 0.98,
):
    poi_classes = poi_classes or POI_CLASSES
    if landing_ids.ndim != 2 or semantic_ids.ndim != 2:
        raise ValueError("Source masks must be 2-D class IDs")
    hazard = (cell_fraction(np.isin(landing_ids, HAZARD_SOURCE_IDS)) >= hazard_fraction).astype(
        np.float32
    )[None]
    landing = (
        cell_fraction(np.isin(landing_ids, LANDING_SOURCE_IDS)) >= landing_safe_fraction
    ).astype(np.float32)[None]
    poi = np.stack(
        [
            (cell_fraction(semantic_ids == class_id) > 0).astype(np.float32)
            for class_id in poi_classes.values()
        ]
    )
    return hazard, landing, poi


def load_targets(
    landing_path, semantic_path, poi_classes=None, hazard_fraction=0.05, landing_safe_fraction=0.98
):
    with Image.open(landing_path) as image:
        landing = np.asarray(image.convert("L"), dtype=np.uint8)
    with Image.open(semantic_path) as image:
        semantic = np.asarray(image.convert("L"), dtype=np.uint8)
    hazard, landing_target, poi = targets_from_sources(
        landing, semantic, poi_classes, hazard_fraction, landing_safe_fraction
    )
    return {
        "hazard": hazard,
        "landing": landing_target,
        "poi": poi,
        "semantic": semantic_grid(semantic),
        "scene": scene_verdict(landing, semantic),
    }
