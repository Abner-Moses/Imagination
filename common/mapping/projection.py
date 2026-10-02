"""Deterministic 2-D region extraction and registration in the C++ local-map frame."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .uncertainty import pixel_depth_covariance, propagate_camera_to_map


@dataclass
class Detection2D:
    kind: str
    class_name: str
    grid_xy: tuple[float, float]
    area_cells: int
    probability: float


@dataclass
class MapFeature:
    id: str
    kind: str
    class_name: str
    map_xyz_m: list[float] | None
    radius_m: float | None
    probability: float
    observation_count: int
    last_seen_timestamp: float
    resolved: bool
    source_grid_xy: list[float]
    attributes: dict[str, float] = field(default_factory=dict)
    position_covariance_m2: list[list[float]] | None = None
    covariance_source: str = "unavailable"


def extract_regions(
    logits_or_probability: np.ndarray,
    threshold: float,
    kind: str,
    class_names: list[str] | None = None,
    min_cells: int = 1,
) -> list[Detection2D]:
    import cv2

    values = np.asarray(logits_or_probability, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.min() < 0 or values.max() > 1:
        values = 1 / (1 + np.exp(-np.clip(values, -40, 40)))
    names = class_names or [kind]
    if len(names) != values.shape[0]:
        raise ValueError("class_names and map channels differ")
    result = []
    for channel, name in enumerate(names):
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (values[channel] >= threshold).astype(np.uint8), 8
        )
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_cells:
                continue
            mask = labels == label
            result.append(
                Detection2D(
                    kind,
                    name,
                    tuple(float(x) for x in centroids[label]),
                    area,
                    float(values[channel][mask].mean()),
                )
            )
    return result


def register_detection(
    detection: Detection2D,
    depth_normalized: np.ndarray,
    depth_valid: np.ndarray,
    depth_scale_m: float,
    calibration: dict[str, float],
    camera_to_local_map: np.ndarray,
    timestamp: float,
    search_radius: int = 2,
    attribute_maps: dict[str, np.ndarray] | None = None,
    roughness_scale_m: float = 0.1,
    pose_covariance=None,
    pixel_sigma: float = 1.0,
    depth_sigma_base_m: float = 0.02,
    depth_sigma_relative: float = 0.01,
) -> MapFeature:
    x, y = detection.grid_xy
    xi, yi = int(round(x)), int(round(y))
    depth = np.asarray(depth_normalized)
    valid = np.asarray(depth_valid) > 0
    candidates = []
    for radius in range(search_radius + 1):
        for yy in range(max(0, yi - radius), min(32, yi + radius + 1)):
            for xx in range(max(0, xi - radius), min(32, xi + radius + 1)):
                if valid[yy, xx] and np.isfinite(depth[yy, xx]) and depth[yy, xx] > 0:
                    candidates.append(((xx - x) ** 2 + (yy - y) ** 2, xx, yy))
        if candidates:
            break
    unresolved = MapFeature(
        "",
        detection.kind,
        detection.class_name,
        None,
        None,
        detection.probability,
        1,
        timestamp,
        False,
        list(detection.grid_xy),
    )
    if not candidates or np.asarray(camera_to_local_map).shape != (4, 4):
        return unresolved
    _, xx, yy = min(candidates)
    z = float(depth[yy, xx]) * float(depth_scale_m)
    fx = float(calibration["fx"]) * 32 / float(calibration["width"])
    fy = float(calibration["fy"]) * 32 / float(calibration["height"])
    cx = (float(calibration["cx"]) + 0.5) * 32 / float(calibration["width"]) - 0.5
    cy = (float(calibration["cy"]) + 0.5) * 32 / float(calibration["height"]) - 0.5
    camera_point = np.array([(x - cx) * z / fx, (y - cy) * z / fy, z], dtype=np.float64)
    mapped = np.asarray(camera_to_local_map, dtype=np.float64) @ np.append(camera_point, 1.0)
    radius_cells = np.sqrt(detection.area_cells / np.pi)
    radius_m = float(radius_cells * z / np.sqrt(fx * fy))
    grid_intrinsics = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    point_covariance = pixel_depth_covariance(
        (xx, yy),
        z,
        grid_intrinsics,
        pixel_sigma=pixel_sigma,
        depth_sigma_m=depth_sigma_base_m + depth_sigma_relative * z,
    )
    map_covariance = (
        propagate_camera_to_map(
            camera_point, point_covariance, camera_to_local_map, pose_covariance
        )
        if point_covariance is not None and pose_covariance is not None
        else None
    )
    attributes = {}
    for name, plane in (attribute_maps or {}).items():
        values = np.asarray(plane)
        if values.shape == depth.shape and np.isfinite(values[yy, xx]):
            scale = (
                np.pi / 2
                if name == "slope_rad"
                else roughness_scale_m
                if name == "roughness_m"
                else 1.0
            )
            attributes[name] = float(values[yy, xx]) * scale
    return MapFeature(
        "",
        detection.kind,
        detection.class_name,
        mapped[:3].tolist(),
        radius_m,
        detection.probability,
        1,
        timestamp,
        True,
        list(detection.grid_xy),
        attributes,
        None if map_covariance is None else map_covariance.tolist(),
        "first_order_approximation" if map_covariance is not None else "unavailable",
    )
