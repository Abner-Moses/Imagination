"""Typed, serializable IMF-LBA contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Decision(str, Enum):
    ANALYTICAL = "ANALYTICAL"
    LEARNING_CANDIDATE = "LEARNING_CANDIDATE"
    UNDECIDED = "UNDECIDED"
    DEPENDENCY_BLOCKED = "DEPENDENCY_BLOCKED"


class EvidenceType(str, Enum):
    SELF_CONSISTENCY = "SELF_CONSISTENCY"
    SUPERVISED_REFERENCE = "SUPERVISED_REFERENCE"
    CROSS_SENSOR_REFERENCE = "CROSS_SENSOR_REFERENCE"
    ANALYTICAL_IDENTITY = "ANALYTICAL_IDENTITY"
    NONE = "NONE"


@dataclass(frozen=True)
class MaskedValue:
    """Tensor-level MSK semantics: placeholder value plus explicit validity."""

    value: float = 0.0
    valid: bool = False


MSK = MaskedValue()


@dataclass(frozen=True)
class Threshold:
    value: float | None
    units: str
    source: str = "REQUIRED"
    note: str = "Must be frozen before primary evaluation."


@dataclass(frozen=True)
class OperatorSpec:
    id: str
    display_name: str
    description: str
    input_dependencies: tuple[str, ...]
    output_features: tuple[str, ...]
    diagnostic_method: str | None
    error_metric: str | None
    uncertainty_metric: str | None
    failure_definition: str | None
    cost_metric: str
    thresholds: dict[str, Threshold]
    reference_type: EvidenceType
    minimum_samples: int
    minimum_frames: int
    minimum_coverage: float
    applicability_conditions: tuple[str, ...]
    downstream_dependencies: tuple[str, ...]
    deployment_relevance: str
    available_upstream_inputs: tuple[str, ...] = ()
    version: str = "1"


@dataclass
class DiagnosticEvidence:
    error_samples: list[float] = field(default_factory=list)
    uncertainty_samples: list[float] = field(default_factory=list)
    failure_samples: list[float] = field(default_factory=list)
    cost_samples: list[float] = field(default_factory=list)
    frames_observed: int = 0
    applicable_frames: int = 0
    coverage: float = 0.0
    evidence_type: EvidenceType = EvidenceType.NONE
    evidence_strength: str = "NONE"
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_samples(self) -> int:
        if self.error_samples:
            return len(self.error_samples)
        return max((len(self.uncertainty_samples), len(self.failure_samples)), default=0)


@dataclass
class OperatorResult:
    operator_id: str
    window_size: int
    valid_samples: int
    frames_observed: int
    applicable_frames: int
    coverage: float
    error_value: float | None
    error_threshold: float | None
    normalized_error: float | None
    uncertainty_value: float | None
    uncertainty_threshold: float | None
    normalized_uncertainty: float | None
    failure_rate: float | None
    failure_threshold: float | None
    normalized_failure: float | None
    cost_value: float | None
    cost_threshold: float | None
    normalized_cost: float | None
    ratio: float | None
    confidence_interval: tuple[float, float] | None
    confidence_level: float
    confidence: str
    decision: Decision
    decision_reasons: list[str]
    evidence_type: EvidenceType
    evidence_strength: str
    dependencies: tuple[str, ...]
    blocked_by: list[str]
    diagnostic_metadata: dict[str, Any]
    warnings: list[str]
    secondary_diagnostic_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decision"] = self.decision.value
        value["evidence_type"] = self.evidence_type.value
        value["R"] = value.pop("ratio")
        return value
