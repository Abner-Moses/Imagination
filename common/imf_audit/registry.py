"""Auditable IMF responsibility registry and dependency graph."""

from __future__ import annotations

from typing import Any

from .contracts import EvidenceType, OperatorSpec, Threshold


LBA_SCHEMA = "imagination-learning-boundary-v1"


def _thresholds(config: dict[str, Any]) -> dict[str, Threshold]:
    result = {}
    for name in ("error", "uncertainty", "failure", "cost"):
        item = config.get("thresholds", {}).get(name, {}) or {}
        value = item.get("value")
        if value is not None and float(value) <= 0:
            raise ValueError(f"IMF-LBA threshold '{name}' must be positive or null")
        result[name] = Threshold(
            None if value is None else float(value),
            str(item.get("units", "UNSPECIFIED")),
            str(item.get("source", "REQUIRED")),
            str(item.get("note", "Must be frozen before primary evaluation.")),
        )
    return result


_DEFINITIONS = {
    "color_conversion": dict(
        display_name="Luminance/chrominance conversion",
        description="Deterministic RGB to YCbCr conversion.",
        deps=(),
        outputs=("Y", "Cb", "Cr"),
    ),
    "image_gradients": dict(
        display_name="Image gradients",
        description="Sobel spatial derivatives and magnitude.",
        deps=("color_conversion",),
        outputs=("Gx", "Gy", "GradientMagnitude"),
    ),
    "orientation_structure": dict(
        display_name="Local orientation structure",
        description="HOG orientation-energy representation.",
        deps=("image_gradients",),
        outputs=tuple(f"HOG_{i}" for i in range(9)),
    ),
    "corner_detection": dict(
        display_name="Corner detection",
        description="Harris keypoint response.",
        deps=("image_gradients",),
        outputs=("HarrisResponse",),
    ),
    "edge_extraction": dict(
        display_name="Edge and contour extraction",
        description="Canny and contour boundary maps.",
        deps=("image_gradients",),
        outputs=("CannyEdge", "ContourMap"),
    ),
    "chroma_transitions": dict(
        display_name="Chroma transitions",
        description="Chrominance-gradient magnitude maps.",
        deps=("color_conversion",),
        outputs=("ChromaGradientCb", "ChromaGradientCr"),
    ),
    "optical_flow": dict(
        display_name="Optical motion estimation",
        description="Dense current-aligned image motion from consecutive luminance frames.",
        deps=(),
        outputs=("OpticalFlowU", "OpticalFlowV"),
        diagnostic="photometric_warp_consistency",
        error="absolute current-versus-warped-previous luminance residual",
        uncertainty="per-frame robust residual dispersion",
        failure="invalid dense-flow fraction",
        reference=EvidenceType.SELF_CONSISTENCY,
        conditions=("consecutive frames", "texture support", "non-trivial motion"),
        relevance="Supplies explicit motion and supports geometry.",
        upstream=("luminance",),
    ),
    "depth_reconstruction": dict(
        display_name="Metric depth reconstruction",
        description="Current-view metric depth reconstructed by multi-view IMF geometry.",
        deps=("optical_flow",),
        outputs=("Depth",),
        diagnostic="calibration_depth_error",
        error="absolute metric depth error against declared calibration reference",
        uncertainty="geometry-support-derived uncertainty proxy; not calibrated covariance",
        failure="reference-valid pixels without reconstructed depth",
        reference=EvidenceType.SUPERVISED_REFERENCE,
        conditions=("valid calibration depth", "valid pose", "geometry support"),
        relevance="Supports metric projection, terrain and visibility.",
        upstream=("appearance", "optical_flow", "state", "camera_calibration"),
    ),
    "repeated_3d_consistency": dict(
        display_name="Repeated 3-D consistency",
        description="Agreement of flow-linked reconstructed points in the local map frame.",
        deps=("optical_flow", "depth_reconstruction"),
        outputs=(),
        diagnostic="flow_linked_map_residual",
        error="Euclidean local-map residual for repeated flow-linked points",
        uncertainty="robust dispersion of repeated-point residuals",
        failure="fraction of attempted links lacking valid geometry/pose",
        reference=EvidenceType.SELF_CONSISTENCY,
        conditions=("consecutive poses", "valid flow", "valid depth in both frames"),
        relevance="Checks temporal metric consistency without simulator semantics.",
        upstream=("optical_flow", "depth", "camera_pose"),
    ),
    "map_association": dict(
        display_name="Map association",
        description="Uncertainty-aware correspondence of repeated metric entities.",
        deps=("repeated_3d_consistency",),
        outputs=(),
        diagnostic="mahalanobis_innovation",
        error="normalized Mahalanobis innovation of flow-linked repeated points",
        uncertainty="combined support-derived positional sigma",
        failure="fraction of repeated points without usable uncertainty",
        reference=EvidenceType.SELF_CONSISTENCY,
        conditions=("repeated observations", "finite covariance approximation"),
        relevance="Prevents duplicate entities and incorrect temporal merges.",
        upstream=("repeated_geometry", "position_uncertainty", "semantic_posterior"),
    ),
    "semantic_evidence_fusion": dict(
        display_name="Semantic evidence accumulation",
        description="Bounded correlation-aware Dirichlet-style temporal evidence fusion.",
        deps=("map_association",),
        outputs=(),
        diagnostic="probability_and_evidence_identities",
        error="posterior normalization and finiteness residual",
        uncertainty="posterior entropy under controlled consistent/contradictory evidence",
        failure="non-finite or evidence-bound violation rate",
        reference=EvidenceType.ANALYTICAL_IDENTITY,
        conditions=("predicted probability sequence for deployment sufficiency"),
        relevance="Replaces learned temporal averaging while retaining uncertainty.",
        upstream=("model_semantic_probabilities", "association", "timestamps"),
    ),
    "visibility": dict(
        display_name="Visibility determination",
        description="Frustum/depth-order visibility of persistent map entities.",
        deps=("depth_reconstruction",),
        outputs=(),
        diagnostic="calibration_depth_visibility_agreement",
        error="visibility disagreement against calibration-depth projection reference",
        uncertainty="absolute reconstructed-versus-reference depth discrepancy",
        failure="unknown visibility fraction where reference visibility is defined",
        reference=EvidenceType.SUPERVISED_REFERENCE,
        conditions=("valid pose", "valid depth", "occluded and visible cases"),
        relevance="Gates current-view relations without erasing persistent memory.",
        upstream=("depth", "camera_pose", "intrinsics", "map_entities"),
    ),
    "drift_prediction": dict(
        display_name="Landing drift prediction",
        description="Constant-horizontal-velocity displacement estimate over a configured horizon.",
        deps=(),
        outputs=(),
        diagnostic="trajectory_displacement_residual",
        error="predicted-versus-observed horizontal displacement error",
        uncertainty="robust dispersion of displacement residuals",
        failure="missing/non-monotonic pose triple fraction",
        reference=EvidenceType.CROSS_SENSOR_REFERENCE,
        conditions=("three consecutive poses", "non-zero time intervals", "motion variation"),
        relevance="Provides deterministic landing-envelope displacement.",
        upstream=("camera_or_body_pose", "timestamps"),
    ),
    "depth_gradient": dict(
        display_name="Depth gradients",
        description="Metric depth derivatives.",
        deps=("depth_reconstruction",),
        outputs=("DepthGradientX", "DepthGradientY"),
    ),
    "surface_slope": dict(
        display_name="Surface slope",
        description="Local terrain inclination.",
        deps=("depth_gradient",),
        outputs=("Slope",),
    ),
    "surface_roughness": dict(
        display_name="Surface roughness",
        description="Local surface variation.",
        deps=("depth_reconstruction", "surface_slope"),
        outputs=("Roughness",),
    ),
    "geometry_confidence": dict(
        display_name="Geometry support confidence",
        description="Bounded reconstruction support score, not a calibrated probability.",
        deps=("depth_reconstruction",),
        outputs=("GeometryConfidence",),
    ),
    "position_uncertainty": dict(
        display_name="Position uncertainty",
        description="First-order metric position covariance.",
        deps=("depth_reconstruction",),
        outputs=(),
    ),
    "obstacle_density": dict(
        display_name="Obstacle density",
        description="Metric neighborhood entity density.",
        deps=("map_association",),
        outputs=(),
    ),
    "restricted_zone_distance": dict(
        display_name="Restricted-zone distance",
        description="Point-to-region boundary approximation.",
        deps=("map_association",),
        outputs=(),
    ),
    "landing_clearance": dict(
        display_name="Landing clearance",
        description="Candidate-centered obstacle/restriction/drift clearance.",
        deps=("restricted_zone_distance", "obstacle_density", "drift_prediction"),
        outputs=(),
    ),
}


def operator_registry(config: dict[str, Any]) -> dict[str, OperatorSpec]:
    operator_config = config.get("operators", {})
    downstream = {name: [] for name in _DEFINITIONS}
    for name, definition in _DEFINITIONS.items():
        for dependency in definition.get("deps", ()):
            downstream.setdefault(dependency, []).append(name)
    registry = {}
    for name, definition in _DEFINITIONS.items():
        item = operator_config.get(name, {}) or {}
        diagnostic = definition.get("diagnostic") if item.get("enabled", True) else None
        registry[name] = OperatorSpec(
            id=name,
            display_name=definition["display_name"],
            description=definition["description"],
            input_dependencies=tuple(definition.get("deps", ())),
            output_features=tuple(definition.get("outputs", ())),
            diagnostic_method=diagnostic,
            error_metric=definition.get("error"),
            uncertainty_metric=definition.get("uncertainty"),
            failure_definition=definition.get("failure"),
            cost_metric="deployment operator latency in milliseconds",
            thresholds=_thresholds(item),
            reference_type=definition.get("reference", EvidenceType.NONE),
            minimum_samples=int(item.get("minimum_samples", 1)),
            minimum_frames=int(item.get("minimum_frames", config.get("default_min_frames", 10))),
            minimum_coverage=float(item.get("minimum_coverage", 0.0)),
            applicability_conditions=tuple(definition.get("conditions", ())),
            downstream_dependencies=tuple(downstream.get(name, ())),
            deployment_relevance=definition.get(
                "relevance", "Diagnostic placeholder; task utility requires ablation."
            ),
            available_upstream_inputs=tuple(definition.get("upstream", ())),
            version=str(item.get("version", "1")),
        )
    return registry


def dependency_edges(registry: dict[str, OperatorSpec]) -> list[tuple[str, str]]:
    return [
        (dependency, name)
        for name, spec in registry.items()
        for dependency in spec.input_dependencies
    ]
