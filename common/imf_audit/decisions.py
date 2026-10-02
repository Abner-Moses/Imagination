"""Safety-oriented sufficiency statistic, confidence interval, and decisions."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .contracts import Decision, DiagnosticEvidence, OperatorResult, OperatorSpec


def analytical_sufficiency_ratio(components: Iterable[float]) -> float:
    """Return max normalized constraint; favorable terms never cancel failure."""
    values = [float(value) for value in components]
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Sufficiency components must be finite, non-negative, and non-empty")
    return max(values)


def decide_from_interval(lower: float, upper: float) -> Decision:
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        raise ValueError("Invalid sufficiency confidence interval")
    if upper <= 1.0:
        return Decision.ANALYTICAL
    if lower > 1.0:
        return Decision.LEARNING_CANDIDATE
    return Decision.UNDECIDED


def mask_stability(left: dict[str, Decision], right: dict[str, Decision]) -> float | None:
    resolved = {Decision.ANALYTICAL, Decision.LEARNING_CANDIDATE}
    shared = [name for name in left if left[name] in resolved and right.get(name) in resolved]
    return None if not shared else sum(left[name] == right[name] for name in shared) / len(shared)


def _aggregate(values: list[float], kind: str) -> float | None:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return None
    return float(np.mean(finite) if kind == "failure" else np.median(finite))


def _bootstrap_ratio(
    evidence: DiagnosticEvidence, thresholds: dict, confidence: float, samples: int, seed: int
) -> tuple[float, float] | None:
    arrays = {
        "error": evidence.error_samples,
        "uncertainty": evidence.uncertainty_samples,
        "failure": evidence.failure_samples,
        "cost": evidence.cost_samples,
    }
    if any(not arrays[name] or thresholds[name].value is None for name in arrays):
        return None
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(samples):
        normalized = []
        for name, values in arrays.items():
            data = np.asarray(values, dtype=np.float64)
            draw = data[rng.integers(0, len(data), len(data))]
            value = float(np.mean(draw) if name == "failure" else np.median(draw))
            normalized.append(value / thresholds[name].value)
        ratios.append(max(normalized))
    tail = (1.0 - confidence) * 0.5
    return float(np.quantile(ratios, tail)), float(np.quantile(ratios, 1.0 - tail))


def evaluate_operator(
    spec: OperatorSpec,
    evidence: DiagnosticEvidence,
    *,
    window_size: int,
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
    dependency_results: dict[str, OperatorResult],
) -> OperatorResult:
    blocked = [
        name
        for name in spec.input_dependencies
        if name in dependency_results
        and dependency_results[name].decision
        in {Decision.LEARNING_CANDIDATE, Decision.DEPENDENCY_BLOCKED}
    ]
    values = {
        "error": _aggregate(evidence.error_samples, "error"),
        "uncertainty": _aggregate(evidence.uncertainty_samples, "uncertainty"),
        "failure": _aggregate(evidence.failure_samples, "failure"),
        "cost": _aggregate(evidence.cost_samples, "cost"),
    }
    normalized = {
        name: (
            None
            if value is None or spec.thresholds[name].value is None
            else value / spec.thresholds[name].value
        )
        for name, value in values.items()
    }
    reasons: list[str] = []
    ratio = None
    interval = None
    confidence = "LOW"
    if blocked:
        decision = Decision.DEPENDENCY_BLOCKED
        reasons.append("DEPENDENCY_INSUFFICIENT")
    elif spec.diagnostic_method is None:
        decision = Decision.UNDECIDED
        reasons.append("DIAGNOSTIC_NOT_IMPLEMENTED")
    elif evidence.frames_observed < spec.minimum_frames:
        decision = Decision.UNDECIDED
        reasons.append("INSUFFICIENT_FRAMES")
    elif evidence.valid_samples < spec.minimum_samples:
        decision = Decision.UNDECIDED
        reasons.append("INSUFFICIENT_DIAGNOSTIC_SAMPLES")
    elif evidence.coverage < spec.minimum_coverage:
        decision = Decision.UNDECIDED
        reasons.append("INSUFFICIENT_APPLICABILITY_COVERAGE")
    elif any(spec.thresholds[name].value is None for name in normalized):
        decision = Decision.UNDECIDED
        reasons.append("THRESHOLD_REQUIRED")
    elif any(normalized[name] is None for name in normalized):
        decision = Decision.UNDECIDED
        reasons.append("METRIC_NOT_MEASURED")
    else:
        ratio = analytical_sufficiency_ratio(normalized.values())
        interval = _bootstrap_ratio(
            evidence, spec.thresholds, confidence_level, bootstrap_samples, seed
        )
        if interval is None:
            decision = Decision.UNDECIDED
            reasons.append("CONFIDENCE_INTERVAL_UNAVAILABLE")
        else:
            decision = decide_from_interval(*interval)
            confidence = (
                "HIGH" if evidence.frames_observed >= max(25, spec.minimum_frames) else "MODERATE"
            )
            if decision == Decision.LEARNING_CANDIDATE:
                labels = {
                    "error": "ERROR_THRESHOLD_EXCEEDED",
                    "uncertainty": "UNCERTAINTY_THRESHOLD_EXCEEDED",
                    "failure": "FAILURE_THRESHOLD_EXCEEDED",
                    "cost": "COST_BUDGET_EXCEEDED",
                }
                reasons.extend(
                    labels[name]
                    for name, value in normalized.items()
                    if value is not None and value > 1
                )
            elif decision == Decision.UNDECIDED:
                reasons.append("CONFIDENCE_INTERVAL_CROSSES_BOUNDARY")
    secondary = None
    known = [value for value in normalized.values() if value is not None]
    if known:
        secondary = float(sum(known) / len(known))
    return OperatorResult(
        operator_id=spec.id,
        window_size=window_size,
        valid_samples=evidence.valid_samples,
        frames_observed=evidence.frames_observed,
        applicable_frames=evidence.applicable_frames,
        coverage=evidence.coverage,
        error_value=values["error"],
        error_threshold=spec.thresholds["error"].value,
        normalized_error=normalized["error"],
        uncertainty_value=values["uncertainty"],
        uncertainty_threshold=spec.thresholds["uncertainty"].value,
        normalized_uncertainty=normalized["uncertainty"],
        failure_rate=values["failure"],
        failure_threshold=spec.thresholds["failure"].value,
        normalized_failure=normalized["failure"],
        cost_value=values["cost"],
        cost_threshold=spec.thresholds["cost"].value,
        normalized_cost=normalized["cost"],
        ratio=ratio,
        confidence_interval=interval,
        confidence_level=confidence_level,
        confidence=confidence,
        decision=decision,
        decision_reasons=reasons,
        evidence_type=evidence.evidence_type,
        evidence_strength=evidence.evidence_strength,
        dependencies=spec.input_dependencies,
        blocked_by=blocked,
        diagnostic_metadata=evidence.metadata,
        warnings=evidence.warnings,
        secondary_diagnostic_score=secondary,
    )
