"""Operator-specific diagnostics over allowed cached calibration frames."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from common.mapping.semantic_map import PersistentSemanticMap, SemanticObservation
from common.mapping.visibility import Visibility, classify_map_point
from data.adapter import resolve_path

from .contracts import DiagnosticEvidence, EvidenceType, OperatorSpec


@dataclass
class AuditFrame:
    record: dict
    features: np.ndarray | None
    validity: np.ndarray | None
    depth_scale_m: float | None
    camera_to_map: np.ndarray | None
    reference_depth_m: np.ndarray | None
    reference_depth_validity: np.ndarray | None

    @property
    def pose_valid(self) -> bool:
        return (
            self.camera_to_map is not None
            and self.camera_to_map.shape == (4, 4)
            and np.isfinite(self.camera_to_map).all()
        )


def load_audit_frames(records: list[dict], manifest: Path) -> list[AuditFrame]:
    frames = []
    for record in records:
        cache_path = resolve_path(manifest.parent, record["analytical_path"])
        features = validity = pose = None
        depth_scale = None
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cache:
                features = cache["features"].astype(np.float64)
                validity = cache["validity"] > 0
                depth_scale = float(cache["depth_scale_m"].item())
                candidate_pose = cache["camera_to_local_map"]
                if candidate_pose.shape == (4, 4) and np.isfinite(candidate_pose).all():
                    pose = candidate_pose.astype(np.float64)
        target_path = cache_path.with_name(cache_path.stem + ".target_depth.npz")
        reference = reference_validity = None
        if target_path.is_file():
            with np.load(target_path, allow_pickle=False) as target:
                if str(target["source_units"].item()) != "camera_z_metres":
                    raise ValueError(f"Unsupported calibration depth units: {target_path}")
                reference = target["depth_m"].astype(np.float64)
                reference_validity = target["validity"] > 0
        frames.append(
            AuditFrame(record, features, validity, depth_scale, pose, reference, reference_validity)
        )
    return frames


def _consecutive(left: AuditFrame, right: AuditFrame) -> bool:
    return (
        left.record["episode_id"] == right.record["episode_id"]
        and int(right.record["frame_index"]) == int(left.record["frame_index"]) + 1
        and float(right.record["timestamp"]) > float(left.record["timestamp"])
    )


def _sample(values: np.ndarray, limit: int = 4096) -> list[float]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size > limit:
        finite = finite[np.linspace(0, finite.size - 1, limit, dtype=int)]
    return finite.tolist()


def _finalize(evidence: DiagnosticEvidence, start: float, diagnostic: str) -> DiagnosticEvidence:
    evidence.metadata.update(
        {
            "diagnostic": diagnostic,
            "auditor_runtime_ms": (perf_counter() - start) * 1000.0,
            "sample_counts": {
                "error": len(evidence.error_samples),
                "uncertainty": len(evidence.uncertainty_samples),
                "failure": len(evidence.failure_samples),
                "deployment_cost": len(evidence.cost_samples),
            },
            "deployment_operator_cost": "NOT_MEASURED",
        }
    )
    return evidence


def optical_flow(frames: list[AuditFrame], _: dict) -> DiagnosticEvidence:
    start = perf_counter()
    evidence = DiagnosticEvidence(
        frames_observed=len(frames),
        evidence_type=EvidenceType.SELF_CONSISTENCY,
        evidence_strength="MODERATE",
    )
    meaningful = 0
    for previous, current in zip(frames, frames[1:]):
        if (
            not _consecutive(previous, current)
            or previous.features is None
            or current.features is None
        ):
            evidence.failure_samples.append(1.0)
            continue
        valid = current.validity[20] & current.validity[21]
        valid_fraction = float(valid.mean())
        evidence.failure_samples.append(1.0 - valid_fraction)
        if not valid.any():
            continue
        y, x = np.indices((32, 32), dtype=np.float32)
        # Flow is stored in 256-pixel working-image units and aligned to current endpoints.
        u = current.features[20].astype(np.float32) * 16.0 / 8.0
        v = current.features[21].astype(np.float32) * 16.0 / 8.0
        warped = cv2.remap(
            previous.features[0].astype(np.float32),
            x - u,
            y - v,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        residual = np.abs(current.features[0] - warped)
        supported = valid & np.isfinite(residual)
        values = residual[supported]
        if not values.size:
            continue
        evidence.error_samples.extend(_sample(values))
        evidence.uncertainty_samples.append(float(np.median(np.abs(values - np.median(values)))))
        texture = current.features[5][supported]
        motion = np.hypot(u[supported], v[supported])
        challenged = float(np.mean((texture > 0.02) & (motion > 0.02)))
        meaningful += challenged >= 0.05
        evidence.applicable_frames += 1
    pairs = max(1, len(frames) - 1)
    evidence.coverage = meaningful / pairs
    if meaningful == 0:
        evidence.warnings.append("No frame pair combined meaningful texture and motion.")
    return _finalize(evidence, start, "photometric_warp_consistency")


def depth_reconstruction(frames: list[AuditFrame], _: dict) -> DiagnosticEvidence:
    start = perf_counter()
    evidence = DiagnosticEvidence(
        frames_observed=len(frames),
        evidence_type=EvidenceType.SUPERVISED_REFERENCE,
        evidence_strength="STRONG",
    )
    reference_count = 0
    supported_count = 0
    for frame in frames:
        if frame.features is None or frame.reference_depth_m is None:
            evidence.failure_samples.append(1.0)
            continue
        reference_valid = (
            frame.reference_depth_validity
            & np.isfinite(frame.reference_depth_m)
            & (frame.reference_depth_m > 0)
        )
        analytic_valid = frame.validity[22]
        reference_count += int(reference_valid.sum())
        supported = reference_valid & analytic_valid
        supported_count += int(supported.sum())
        evidence.failure_samples.append(
            1.0 - float(supported.sum()) / max(1, int(reference_valid.sum()))
        )
        if not supported.any():
            continue
        depth_m = frame.features[22] * frame.depth_scale_m
        error = np.abs(depth_m[supported] - frame.reference_depth_m[supported])
        evidence.error_samples.extend(_sample(error))
        support = np.clip(frame.features[27][supported], 0.0, 1.0)
        # GeometryConfidence is a support score, explicitly not calibrated covariance.
        evidence.uncertainty_samples.extend(_sample(1.0 - support))
        evidence.applicable_frames += 1
    evidence.coverage = supported_count / max(1, reference_count)
    evidence.metadata["reference"] = (
        "simulator camera-Z depth; calibration-only supervised diagnostic"
    )
    if not reference_count:
        evidence.warnings.append("No declared calibration depth reference was available.")
    return _finalize(evidence, start, "calibration_depth_error")


def _backproject(
    grid_x: np.ndarray, grid_y: np.ndarray, depth: np.ndarray, intrinsics: dict[str, float]
) -> np.ndarray:
    fx = intrinsics["fx"] * 32.0 / intrinsics["width"]
    fy = intrinsics["fy"] * 32.0 / intrinsics["height"]
    cx = (intrinsics["cx"] + 0.5) * 32.0 / intrinsics["width"] - 0.5
    cy = (intrinsics["cy"] + 0.5) * 32.0 / intrinsics["height"] - 0.5
    return np.stack(((grid_x - cx) * depth / fx, (grid_y - cy) * depth / fy, depth), axis=-1)


def _repeated_points(frames: list[AuditFrame], settings: dict):
    intrinsics = settings["intrinsics"]
    pairs = []
    attempted = 0
    for previous, current in zip(frames, frames[1:]):
        if not _consecutive(previous, current) or not previous.pose_valid or not current.pose_valid:
            continue
        if previous.features is None or current.features is None:
            continue
        current_valid = current.validity[20] & current.validity[21] & current.validity[22]
        x, y = np.meshgrid(np.arange(32), np.arange(32))
        previous_x = np.rint(x - current.features[20] * 2.0).astype(int)
        previous_y = np.rint(y - current.features[21] * 2.0).astype(int)
        inside = (previous_x >= 0) & (previous_x < 32) & (previous_y >= 0) & (previous_y < 32)
        mask = current_valid & inside
        attempted += int(current_valid.sum())
        if not mask.any():
            continue
        cy, cx = np.nonzero(mask)
        py, px = previous_y[mask], previous_x[mask]
        previous_good = previous.validity[22, py, px]
        cy, cx, py, px = cy[previous_good], cx[previous_good], py[previous_good], px[previous_good]
        if not len(cx):
            continue
        current_depth = current.features[22, cy, cx] * current.depth_scale_m
        previous_depth = previous.features[22, py, px] * previous.depth_scale_m
        current_camera = _backproject(cx, cy, current_depth, intrinsics)
        previous_camera = _backproject(px, py, previous_depth, intrinsics)
        current_map = (current.camera_to_map[:3, :3] @ current_camera.T).T + current.camera_to_map[
            :3, 3
        ]
        previous_map = (
            previous.camera_to_map[:3, :3] @ previous_camera.T
        ).T + previous.camera_to_map[:3, 3]
        residual = np.linalg.norm(current_map - previous_map, axis=1)
        confidence = np.minimum(current.features[27, cy, cx], previous.features[27, py, px])
        pairs.append((residual, confidence))
    return pairs, attempted


def repeated_3d(frames: list[AuditFrame], settings: dict) -> DiagnosticEvidence:
    start = perf_counter()
    evidence = DiagnosticEvidence(
        frames_observed=len(frames),
        evidence_type=EvidenceType.SELF_CONSISTENCY,
        evidence_strength="MODERATE",
    )
    pairs, attempted = _repeated_points(frames, settings)
    valid = 0
    for residual, _ in pairs:
        evidence.error_samples.extend(_sample(residual))
        evidence.uncertainty_samples.append(
            float(np.median(np.abs(residual - np.median(residual))))
        )
        valid += len(residual)
        evidence.applicable_frames += 1
    evidence.failure_samples.append(1.0 - valid / max(1, attempted))
    evidence.coverage = valid / max(1, attempted)
    if not pairs:
        evidence.warnings.append("No flow-linked repeated 3-D points were available.")
    return _finalize(evidence, start, "flow_linked_map_residual")


def map_association(frames: list[AuditFrame], settings: dict) -> DiagnosticEvidence:
    start = perf_counter()
    evidence = DiagnosticEvidence(
        frames_observed=len(frames),
        evidence_type=EvidenceType.SELF_CONSISTENCY,
        evidence_strength="MODERATE",
    )
    pairs, attempted = _repeated_points(frames, settings)
    usable = 0
    for residual, confidence in pairs:
        sigma = 0.02 + (1.0 - np.clip(confidence, 0.0, 1.0)) * 0.5
        combined_sigma = np.sqrt(2.0) * sigma
        good = np.isfinite(residual) & np.isfinite(combined_sigma) & (combined_sigma > 0)
        if not good.any():
            continue
        innovation = residual[good] / combined_sigma[good]
        evidence.error_samples.extend(_sample(innovation))
        evidence.uncertainty_samples.extend(_sample(combined_sigma[good]))
        usable += int(good.sum())
        evidence.applicable_frames += 1
    evidence.failure_samples.append(1.0 - usable / max(1, attempted))
    evidence.coverage = usable / max(1, attempted)
    evidence.metadata["limitation"] = (
        "Flow-linked proxy associations; duplicate/incorrect merge truth unavailable."
    )
    return _finalize(evidence, start, "mahalanobis_innovation")


def semantic_fusion(frames: list[AuditFrame], _: dict) -> DiagnosticEvidence:
    start = perf_counter()
    evidence = DiagnosticEvidence(
        frames_observed=len(frames),
        evidence_type=EvidenceType.ANALYTICAL_IDENTITY,
        evidence_strength="WEAK",
    )
    semantic_map = PersistentSemanticMap(max_semantic_evidence=10.0)
    for index in range(max(1, len(frames))):
        probabilities = (
            {"track": 0.8, "grass": 0.2} if index % 4 else {"track": 0.35, "grass": 0.65}
        )
        entity = semantic_map.update(
            SemanticObservation(
                (0.0, 0.0, 0.0),
                probabilities,
                0.9,
                float(index),
                0.2,
                position_covariance_m2=(np.eye(3) * 0.01).tolist(),
            )
        )
        if entity is None:
            evidence.failure_samples.append(1.0)
            continue
        values = np.asarray(list(entity.class_probabilities.values()), dtype=np.float64)
        evidence.error_samples.append(abs(float(values.sum()) - 1.0))
        evidence.uncertainty_samples.append(entity.semantic_entropy)
        evidence.failure_samples.append(
            float(not np.isfinite(values).all() or entity.semantic_support > 10.0 + 1e-6)
        )
        evidence.applicable_frames += 1
    evidence.coverage = 0.5
    evidence.warnings.append(
        "Analytical identities do not establish deployment sufficiency without model-prediction sequences."
    )
    evidence.metadata["limitation"] = "Controlled identity test; no deployment prediction sequence."
    return _finalize(evidence, start, "probability_and_evidence_identities")


def visibility(frames: list[AuditFrame], settings: dict) -> DiagnosticEvidence:
    start = perf_counter()
    evidence = DiagnosticEvidence(
        frames_observed=len(frames),
        evidence_type=EvidenceType.SUPERVISED_REFERENCE,
        evidence_strength="MODERATE",
    )
    intrinsics = settings["intrinsics"]
    for frame in frames:
        if not frame.pose_valid or frame.reference_depth_m is None or frame.features is None:
            evidence.failure_samples.append(1.0)
            continue
        depth_m = frame.features[22] * frame.depth_scale_m
        valid = frame.reference_depth_validity & (frame.reference_depth_m > 0)
        ys, xs = np.nonzero(valid)
        if len(xs) > 64:
            selected = np.linspace(0, len(xs) - 1, 64, dtype=int)
            ys, xs = ys[selected], xs[selected]
        disagreements = unknown = 0
        for y, x in zip(ys, xs):
            camera = _backproject(
                np.asarray(x), np.asarray(y), np.asarray(frame.reference_depth_m[y, x]), intrinsics
            )
            point_map = frame.camera_to_map[:3, :3] @ camera + frame.camera_to_map[:3, 3]
            code, _, _ = classify_map_point(
                point_map, frame.camera_to_map, intrinsics, depth_m, frame.validity[22]
            )
            disagreements += code != Visibility.VISIBLE
            unknown += code == Visibility.UNKNOWN
            evidence.error_samples.append(float(code != Visibility.VISIBLE))
            evidence.uncertainty_samples.append(
                abs(depth_m[y, x] - frame.reference_depth_m[y, x])
                if frame.validity[22, y, x]
                else frame.reference_depth_m[y, x]
            )
        if len(xs):
            evidence.failure_samples.append(unknown / len(xs))
            evidence.applicable_frames += 1
    # The available point construction challenges visible agreement only.
    evidence.coverage = 0.5 if evidence.applicable_frames else 0.0
    evidence.warnings.append(
        "Calibration window contains no explicit occluded/out-of-FOV reference cases."
    )
    return _finalize(evidence, start, "calibration_depth_visibility_agreement")


def drift(frames: list[AuditFrame], _: dict) -> DiagnosticEvidence:
    start = perf_counter()
    evidence = DiagnosticEvidence(
        frames_observed=len(frames),
        evidence_type=EvidenceType.CROSS_SENSOR_REFERENCE,
        evidence_strength="STRONG",
    )
    attempted = max(0, len(frames) - 2)
    residuals = []
    for first, second, third in zip(frames, frames[1:], frames[2:]):
        if not (
            _consecutive(first, second)
            and _consecutive(second, third)
            and first.pose_valid
            and second.pose_valid
            and third.pose_valid
        ):
            evidence.failure_samples.append(1.0)
            continue
        dt1 = float(second.record["timestamp"]) - float(first.record["timestamp"])
        dt2 = float(third.record["timestamp"]) - float(second.record["timestamp"])
        if min(dt1, dt2) <= 0:
            evidence.failure_samples.append(1.0)
            continue
        velocity = (second.camera_to_map[:2, 3] - first.camera_to_map[:2, 3]) / dt1
        predicted = velocity * dt2
        observed = third.camera_to_map[:2, 3] - second.camera_to_map[:2, 3]
        residuals.append(float(np.linalg.norm(predicted - observed)))
        evidence.failure_samples.append(0.0)
        evidence.applicable_frames += 1
    evidence.error_samples.extend(residuals)
    if residuals:
        center = float(np.median(residuals))
        evidence.uncertainty_samples.extend(abs(value - center) for value in residuals)
    evidence.coverage = evidence.applicable_frames / max(1, attempted)
    if not residuals:
        evidence.warnings.append("No valid pose triples were available for drift evaluation.")
    return _finalize(evidence, start, "trajectory_displacement_residual")


DIAGNOSTICS: dict[str, Callable[[list[AuditFrame], dict], DiagnosticEvidence]] = {
    "optical_flow": optical_flow,
    "depth_reconstruction": depth_reconstruction,
    "repeated_3d_consistency": repeated_3d,
    "map_association": map_association,
    "semantic_evidence_fusion": semantic_fusion,
    "visibility": visibility,
    "drift_prediction": drift,
}


def run_diagnostic(
    spec: OperatorSpec, frames: list[AuditFrame], settings: dict
) -> DiagnosticEvidence:
    diagnostic = DIAGNOSTICS.get(spec.id)
    if diagnostic is None or spec.diagnostic_method is None:
        evidence = DiagnosticEvidence(
            frames_observed=len(frames),
            evidence_type=spec.reference_type,
            evidence_strength="NONE",
            warnings=["No meaningful diagnostic is registered."],
        )
    else:
        evidence = diagnostic(frames, settings)
    measured_cost = settings.get("operator_config", {}).get("measured_cost")
    if measured_cost and measured_cost.get("value") is not None:
        value = float(measured_cost["value"])
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"Invalid measured deployment cost for {spec.id}")
        evidence.cost_samples.append(value)
        evidence.metadata["deployment_cost_measurement"] = dict(measured_cost)
    return evidence
