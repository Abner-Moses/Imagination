"""Small first-order covariance helpers for local-map observations.

The current IMF pose/geometry APIs do not publish a calibrated covariance. The
support-to-pose model below is explicitly approximate and must not be reported
as calibrated uncertainty.
"""

from __future__ import annotations

import numpy as np


def stabilize_covariance(
    value, dimension: int = 3, *, floor: float = 1e-8, negative_tolerance: float = 1e-7
) -> np.ndarray | None:
    """Validate symmetry/PSD and floor near-zero eigenvalues for stable solves."""
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (dimension, dimension) or not np.isfinite(covariance).all():
        return None
    if not np.allclose(covariance, covariance.T, atol=1e-7, rtol=1e-6):
        return None
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues.min()) < -negative_tolerance:
        return None
    eigenvalues = np.maximum(eigenvalues, floor)
    stabilized = (eigenvectors * eigenvalues) @ eigenvectors.T
    return (stabilized + stabilized.T) * 0.5


def pixel_depth_covariance(
    pixel_xy,
    depth_m: float,
    intrinsics: dict[str, float],
    *,
    pixel_sigma: float,
    depth_sigma_m: float,
) -> np.ndarray | None:
    """Propagate pixel/depth noise through the OpenCV pinhole back-projection."""
    x, y = map(float, pixel_xy)
    z = float(depth_m)
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    if (
        min(fx, fy, z, pixel_sigma, depth_sigma_m) <= 0
        or not np.isfinite((x, y, z, fx, fy, cx, cy)).all()
    ):
        return None
    jacobian = np.array(
        [
            [z / fx, 0.0, (x - cx) / fx],
            [0.0, z / fy, (y - cy) / fy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    input_covariance = np.diag((pixel_sigma**2, pixel_sigma**2, depth_sigma_m**2))
    return stabilize_covariance(jacobian @ input_covariance @ jacobian.T)


def approximate_pose_covariance(
    support: float | None,
    *,
    translation_floor_m: float = 0.01,
    translation_at_zero_m: float = 0.5,
    rotation_floor_rad: float = 0.002,
    rotation_at_zero_rad: float = 0.15,
) -> np.ndarray | None:
    """Create a configurable approximate 6-D pose covariance from support.

    `support` is an IMF geometry support score, not a probability. This model is
    a conservative heuristic until pose residuals are exposed and calibrated.
    """
    if support is None or not np.isfinite(support):
        return None
    quality = float(np.clip(support, 0.0, 1.0))
    translation_sigma = translation_floor_m + (1.0 - quality) * translation_at_zero_m
    rotation_sigma = rotation_floor_rad + (1.0 - quality) * rotation_at_zero_rad
    return np.diag([translation_sigma**2] * 3 + [rotation_sigma**2] * 3)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = map(float, vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def propagate_camera_to_map(
    point_camera, covariance_camera, camera_to_map, pose_covariance=None
) -> np.ndarray | None:
    """Propagate point and optional small-pose uncertainty into the map frame."""
    point = np.asarray(point_camera, dtype=np.float64)
    transform = np.asarray(camera_to_map, dtype=np.float64)
    point_covariance = stabilize_covariance(covariance_camera)
    if point.shape != (3,) or transform.shape != (4, 4) or point_covariance is None:
        return None
    if not np.isfinite(point).all() or not np.isfinite(transform).all():
        return None
    rotation = transform[:3, :3]
    covariance_map = rotation @ point_covariance @ rotation.T
    if pose_covariance is not None:
        pose_cov = stabilize_covariance(pose_covariance, dimension=6)
        if pose_cov is None:
            return None
        jacobian_pose = np.concatenate((np.eye(3), -rotation @ _skew(point)), axis=1)
        covariance_map += jacobian_pose @ pose_cov @ jacobian_pose.T
    return stabilize_covariance(covariance_map)


def mahalanobis_squared(delta, covariance, *, regularization: float = 1e-6) -> float | None:
    """Solve the covariance system without explicitly inverting it."""
    vector = np.asarray(delta, dtype=np.float64)
    matrix = stabilize_covariance(covariance, floor=regularization)
    if vector.shape != (3,) or matrix is None or not np.isfinite(vector).all():
        return None
    try:
        solved = np.linalg.solve(matrix, vector)
    except np.linalg.LinAlgError:
        return None
    value = float(vector @ solved)
    return value if np.isfinite(value) and value >= 0 else None


def fuse_gaussians(mean_a, covariance_a, mean_b, covariance_b):
    """Information-form fusion with stable solves; returns None on invalid input."""
    a = np.asarray(mean_a, dtype=np.float64)
    b = np.asarray(mean_b, dtype=np.float64)
    ca = stabilize_covariance(covariance_a)
    cb = stabilize_covariance(covariance_b)
    if a.shape != (3,) or b.shape != (3,) or ca is None or cb is None:
        return None
    try:
        precision_a = np.linalg.solve(ca, np.eye(3))
        precision_b = np.linalg.solve(cb, np.eye(3))
        precision = precision_a + precision_b
        covariance = stabilize_covariance(np.linalg.solve(precision, np.eye(3)))
        if covariance is None:
            return None
        mean = covariance @ (precision_a @ a + precision_b @ b)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(mean).all():
        return None
    return mean, covariance
