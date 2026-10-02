"""Explicit transforms between the local-map and yaw-aligned ground frame."""

from __future__ import annotations

import math
import numpy as np


def map_to_egocentric(point_map, origin_map, yaw_rad: float) -> np.ndarray:
    """Return body-aligned XYZ with +x forward, +y left, +z up.

    The map follows the C++ nadir local frame (+Z up). Only yaw rotates the
    horizontal ground plane; roll/pitch do not tilt terrain relationships.
    This is a 2.5-D navigation frame, not the full aircraft body frame.
    """
    point = np.asarray(point_map, dtype=np.float64)
    origin = np.asarray(origin_map, dtype=np.float64)
    if (
        point.shape != (3,)
        or origin.shape != (3,)
        or not np.isfinite((*point, *origin, yaw_rad)).all()
    ):
        raise ValueError("Map/egocentric transform requires finite XYZ and yaw")
    dx, dy, dz = point - origin
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    return np.array([c * dx + s * dy, -s * dx + c * dy, dz], dtype=np.float64)


def egocentric_to_map(point_ego, origin_map, yaw_rad: float) -> np.ndarray:
    """Inverse of `map_to_egocentric` under the same yaw-aligned convention."""
    point = np.asarray(point_ego, dtype=np.float64)
    origin = np.asarray(origin_map, dtype=np.float64)
    if (
        point.shape != (3,)
        or origin.shape != (3,)
        or not np.isfinite((*point, *origin, yaw_rad)).all()
    ):
        raise ValueError("Egocentric/map transform requires finite XYZ and yaw")
    x, y, z = point
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    return origin + np.array([c * x - s * y, s * x + c * y, z], dtype=np.float64)


def map_vector_to_egocentric(vector_map, yaw_rad: float) -> np.ndarray:
    """Rotate a local-map vector into +x-forward/+y-left coordinates.

    Unlike point transforms this never applies translation. The local-map
    convention is +x east/+y north/+z up and positive yaw rotates +x toward +y.
    """
    vector = np.asarray(vector_map, dtype=np.float64)
    if (
        vector.shape not in {(2,), (3,)}
        or not np.isfinite(vector).all()
        or not math.isfinite(yaw_rad)
    ):
        raise ValueError("Map vector transform requires a finite XY/XYZ vector and yaw")
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    result = vector.copy()
    result[0] = c * vector[0] + s * vector[1]
    result[1] = -s * vector[0] + c * vector[1]
    return result


def egocentric_vector_to_map(vector_ego, yaw_rad: float) -> np.ndarray:
    """Inverse rotation of :func:`map_vector_to_egocentric`, no translation."""
    vector = np.asarray(vector_ego, dtype=np.float64)
    if (
        vector.shape not in {(2,), (3,)}
        or not np.isfinite(vector).all()
        or not math.isfinite(yaw_rad)
    ):
        raise ValueError("Egocentric vector transform requires a finite XY/XYZ vector and yaw")
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    result = vector.copy()
    result[0] = c * vector[0] - s * vector[1]
    result[1] = s * vector[0] + c * vector[1]
    return result
