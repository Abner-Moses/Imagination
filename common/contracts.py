"""Shared input and output names for the four-model experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


COMMON_INPUT_FIELDS = (
    "vehicle_state",
    "state_validity",
    "relational",
    "relational_validity",
    "candidates",
    "candidate_validity",
    "candidate_feature_validity",
    "metadata",
)
RGB_INPUT_FIELDS = ("rgb",)
ANALYTICAL_INPUT_FIELDS = (
    "analytical",
    "validity",
    "calibration",
    "depth_scale_m",
)
TARGET_FIELDS = (
    "hazard_target",
    "hazard_validity",
    "landing_target",
    "landing_validity",
    "semantic_target",
    "semantic_validity",
    "poi_target",
    "poi_validity",
    "scene_target",
)
MODEL_OUTPUT_FIELDS = (
    "hazard_logits",
    "landing_logits",
    "semantic_logits",
    "poi",
    "scene_risk_logits",
    "candidate",
)
CANDIDATE_OUTPUT_FIELDS = ("risk_logits", "landing_safe_logits", "validity")


def required_input_fields(observation: str, *, include_targets: bool = True) -> tuple[str, ...]:
    """Return the batch fields required by one registered observation contract."""
    if observation == "rgb":
        observation_fields = RGB_INPUT_FIELDS
    elif observation == "analytical":
        observation_fields = ANALYTICAL_INPUT_FIELDS
    else:
        raise ValueError(f"Unknown observation contract: {observation!r}")
    targets = TARGET_FIELDS if include_targets else ()
    return (*observation_fields, *COMMON_INPUT_FIELDS, *targets)


def validate_model_outputs(outputs: Mapping[str, Any]) -> None:
    """Reject output-name drift before losses and metrics silently diverge."""
    missing = set(MODEL_OUTPUT_FIELDS) - outputs.keys()
    if missing:
        raise ValueError(f"Model output contract is missing {sorted(missing)}")
    poi = outputs["poi"]
    if not isinstance(poi, Mapping) or "class_logits" not in poi:
        raise ValueError("Model output 'poi' must contain 'class_logits'")
    candidate = outputs["candidate"]
    if not isinstance(candidate, Mapping):
        raise ValueError("Model output 'candidate' must be a mapping")
    missing_candidate = set(CANDIDATE_OUTPUT_FIELDS) - candidate.keys()
    if missing_candidate:
        raise ValueError(f"Candidate output contract is missing {sorted(missing_candidate)}")
