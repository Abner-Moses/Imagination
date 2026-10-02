"""Versioned numerical contracts shared by data, IMF, models, and checkpoints."""

from __future__ import annotations

ANALYTICAL_CHANNELS = (
    "Y",
    "Cb",
    "Cr",
    "Gx",
    "Gy",
    "GradientMagnitude",
    "HOG_0",
    "HOG_1",
    "HOG_2",
    "HOG_3",
    "HOG_4",
    "HOG_5",
    "HOG_6",
    "HOG_7",
    "HOG_8",
    "HarrisResponse",
    "CannyEdge",
    "ContourMap",
    "ChromaGradientCb",
    "ChromaGradientCr",
    "OpticalFlowU",
    "OpticalFlowV",
    "Depth",
    "DepthGradientX",
    "DepthGradientY",
    "Slope",
    "Roughness",
    "GeometryConfidence",
)
# Renaming the subsystem to IMF does not change its numerical cache contract.
# Keeping v1 lets the validated cache remain reproducible without recomputation.
CHANNEL_REGISTRY_VERSION = "imagination-analytical-v1"
MODEL_ARCHITECTURE_VERSION = "imagination-imf-math-assisted-v7"
MAP_SCHEMA_VERSION = "imagination-semantic-map-v3"
FUSION_CONTRACT_VERSION = "family-fusion-v2"
SEMANTIC_FUSION_VERSION = "dirichlet-evidence-terrain-contradiction-v3"
METRIC_ATTENTION_VERSION = "imf-stagewise-activequery-relational-v5"
CANDIDATE_FEATURE_VERSION = "imagination-candidate-mask-contract-v2"
DATASET_MANIFEST_VERSION = "imagination-perception-v2"
CHECKPOINT_SCHEMA_VERSION = "imagination-checkpoint-v1"
TRAINING_PLAN_SCHEMA_VERSION = "imagination-primary-training-plan-v1"
TRAINING_READINESS_SCHEMA_VERSION = "imagination-training-readiness-v1"
TRAINING_STATISTICS_VERSION = "imagination-training-statistics-v2"


def contract_versions() -> dict[str, str]:
    """Return serialized contract versions recorded with research artifacts."""
    return {
        "analytical_features": CHANNEL_REGISTRY_VERSION,
        "model_architecture": MODEL_ARCHITECTURE_VERSION,
        "map_schema": MAP_SCHEMA_VERSION,
        "fusion": FUSION_CONTRACT_VERSION,
        "semantic_fusion": SEMANTIC_FUSION_VERSION,
        "metric_attention": METRIC_ATTENTION_VERSION,
        "candidate_features": CANDIDATE_FEATURE_VERSION,
        "dataset_manifest": DATASET_MANIFEST_VERSION,
        "checkpoint": CHECKPOINT_SCHEMA_VERSION,
        "training_plan": TRAINING_PLAN_SCHEMA_VERSION,
        "training_readiness": TRAINING_READINESS_SCHEMA_VERSION,
        "training_statistics": TRAINING_STATISTICS_VERSION,
    }


STATE_NAMES = (
    "accel_x_mps2",
    "accel_y_mps2",
    "accel_z_mps2",
    "gyro_x_radps",
    "gyro_y_radps",
    "gyro_z_radps",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "mag_x_uT",
    "mag_y_uT",
    "mag_z_uT",
    "ultrasonic_axis_range_m",
)
STATE_DIM = len(STATE_NAMES)
SEMANTIC_CLASSES = ("background", "grass", "track", "cone", "rock", "other_obstacle")
SEMANTIC_SOURCE_TO_TRAIN = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 5,
    7: 5,
    8: 5,
    9: 5,
    10: 0,
}
POI_CLASSES = {
    "cone": 3,
    "rock": 4,
    "football": 5,
    "bag": 6,
    "box": 7,
    "pole": 8,
    "chair": 9,
}
SCENE_CLASSES = ("SAFE", "CAUTION", "DANGEROUS")
RELATIONAL_NAMES = (
    "nearest_track_m",
    "track_bearing_sin",
    "track_bearing_cos",
    "restricted_zone_clearance_m",
    "nearest_obstacle_m",
    "obstacle_density_2m",
    "cone_density_1m",
    "cone_density_2m",
    "cone_density_4m",
    "maximum_cone_density",
    "maximum_density_bearing_sin",
    "maximum_density_bearing_cos",
    "free_space_radius_m",
    "best_landing_clearance_m",
    "drift_x_m",
    "drift_y_m",
    "horizontal_speed_mps",
    "drift_clearance_m",
    "landing_margin_m",
    "map_coverage",
    "geometry_confidence",
    "mapped_poi_count_scaled",
    "nearest_poi_m",
    "geometry_freshness",
    "map_semantic_confidence",
    "map_position_sigma_m",
    "map_semantic_entropy",
    "map_visible_fraction",
    "map_evidence_support_scaled",
    "orientation_valid",
)
CANDIDATE_TYPES = (
    "landing",
    "obstacle_cluster",
    "restricted_boundary",
    "poi",
    "high_risk",
    "terrain_region",
)
CANDIDATE_FEATURE_NAMES = (
    "egocentric_x_m",
    "egocentric_y_m",
    "egocentric_z_m",
    "distance_m",
    "bearing_sin",
    "bearing_cos",
    *(f"type_{name}" for name in CANDIDATE_TYPES),
    "semantic_confidence",
    "extent_m",
    "obstacle_density_2m_entities_per_m2",
    "slope_rad",
    "roughness_m",
    "free_radius_m",
    "restricted_boundary_clearance_m",
    "age_s",
    "horizontal_sigma_m",
    "vertical_sigma_m",
    "semantic_entropy",
    "evidence_support_scaled",
    "visibility_unknown",
    "visibility_visible",
    "visibility_behind_camera",
    "visibility_outside_fov",
    "visibility_occluded",
    "visibility_depth_inconsistent",
    "orientation_valid",
    "importance_score",
    "track_boundary_clearance_m",
    "nearest_obstacle_clearance_m",
    "cone_density_2m_entities_per_m2",
    "drifted_track_clearance_m",
    "drifted_obstacle_clearance_m",
    "drifted_restricted_clearance_m",
    "drifted_landing_margin_m",
    "drift_valid",
)
MAX_CANDIDATES = 32
DEFAULT_STATE_NAMES = STATE_NAMES
