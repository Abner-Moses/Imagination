"""Deterministic current-camera visibility tests using IMF geometry only."""

from __future__ import annotations

from enum import IntEnum
import math

import numpy as np


class Visibility(IntEnum):
    UNKNOWN = 0
    VISIBLE = 1
    BEHIND_CAMERA = 2
    OUTSIDE_FOV = 3
    OCCLUDED = 4
    DEPTH_INCONSISTENT = 5


def classify_map_point(
    point_map,
    camera_to_map,
    intrinsics: dict[str, float],
    depth_m=None,
    depth_validity=None,
    *,
    tolerance_m: float = 0.25,
) -> tuple[Visibility, tuple[int, int] | None, float | None]:
    """Project one map point and compare it with reconstructed current-view Z.

    A missing depth sample yields UNKNOWN, never VISIBLE by assumption. Depth is
    the C++ IMF current-camera optical-axis measurement, not target-only depth.
    """
    point = np.asarray(point_map, dtype=np.float64)
    transform = np.asarray(camera_to_map, dtype=np.float64)
    if (
        point.shape != (3,)
        or transform.shape != (4, 4)
        or not np.isfinite(point).all()
        or not np.isfinite(transform).all()
    ):
        return Visibility.UNKNOWN, None, None
    try:
        camera = np.linalg.solve(transform, np.append(point, 1.0))[:3]
    except np.linalg.LinAlgError:
        return Visibility.UNKNOWN, None, None
    x_cam, y_cam, z_cam = map(float, camera)
    if not math.isfinite(z_cam):
        return Visibility.UNKNOWN, None, None
    if z_cam <= 0:
        return Visibility.BEHIND_CAMERA, None, z_cam
    width, height = float(intrinsics["width"]), float(intrinsics["height"])
    gx_f, gy_f = float(intrinsics["fx"]) * 32 / width, float(intrinsics["fy"]) * 32 / height
    cx = (float(intrinsics["cx"]) + 0.5) * 32 / width - 0.5
    cy = (float(intrinsics["cy"]) + 0.5) * 32 / height - 0.5
    gx = int(round(gx_f * x_cam / z_cam + cx))
    gy = int(round(gy_f * y_cam / z_cam + cy))
    if not (0 <= gx < 32 and 0 <= gy < 32):
        return Visibility.OUTSIDE_FOV, None, z_cam
    if depth_m is None or depth_validity is None:
        return Visibility.UNKNOWN, (gx, gy), z_cam
    depth = np.asarray(depth_m)
    validity = np.asarray(depth_validity) > 0
    if depth.shape != (32, 32) or validity.shape != depth.shape:
        raise ValueError("Visibility depth and validity must be 32x32 current-view maps")
    measured = float(depth[gy, gx])
    if not validity[gy, gx] or not math.isfinite(measured) or measured <= 0:
        return Visibility.UNKNOWN, (gx, gy), z_cam
    tolerance = max(float(tolerance_m), 0.05 * measured)
    difference = z_cam - measured
    if difference > tolerance:
        return Visibility.OCCLUDED, (gx, gy), z_cam
    if difference < -tolerance:
        return Visibility.DEPTH_INCONSISTENT, (gx, gy), z_cam
    return Visibility.VISIBLE, (gx, gy), z_cam
