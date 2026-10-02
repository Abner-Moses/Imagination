"""Focused scientific-contract tests for IMF Learning-Boundary Auditor."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from common.imf_audit.contracts import (
    Decision,
    DiagnosticEvidence,
    EvidenceType,
    MSK,
    MaskedValue,
    OperatorSpec,
    Threshold,
)
from common.imf_audit.decisions import (
    analytical_sufficiency_ratio,
    decide_from_interval,
    evaluate_operator,
    mask_stability,
)
from common.imf_audit.registry import operator_registry
from common.imf_audit.runner import _records


def _spec(**overrides) -> OperatorSpec:
    base = OperatorSpec(
        id="candidate",
        display_name="Candidate",
        description="test",
        input_dependencies=(),
        output_features=("Depth",),
        diagnostic_method="test",
        error_metric="error",
        uncertainty_metric="uncertainty",
        failure_definition="failure",
        cost_metric="cost",
        thresholds={
            name: Threshold(1.0, "unit", "test")
            for name in ("error", "uncertainty", "failure", "cost")
        },
        reference_type=EvidenceType.SELF_CONSISTENCY,
        minimum_samples=2,
        minimum_frames=2,
        minimum_coverage=0.5,
        applicability_conditions=(),
        downstream_dependencies=(),
        deployment_relevance="test",
        available_upstream_inputs=("appearance",),
    )
    return replace(base, **overrides)


def _evidence(error=0.5, uncertainty=0.5, failure=0.5, cost=0.5, coverage=1.0):
    return DiagnosticEvidence(
        error_samples=[error] * 20,
        uncertainty_samples=[uncertainty] * 20,
        failure_samples=[failure] * 20,
        cost_samples=[cost] * 20,
        frames_observed=20,
        applicable_frames=20,
        coverage=coverage,
        evidence_type=EvidenceType.SELF_CONSISTENCY,
        evidence_strength="STRONG",
    )


def test_primary_ratio_is_maximum_and_never_compensates():
    assert analytical_sufficiency_ratio((0.5, 0.7, 0.4, 1.3)) == 1.3
    assert analytical_sufficiency_ratio((2.0, 0.01, 0.01, 0.01)) == 2.0


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ((0.6, 0.9), Decision.ANALYTICAL),
        ((1.01, 1.4), Decision.LEARNING_CANDIDATE),
        ((0.8, 1.2), Decision.UNDECIDED),
    ],
)
def test_three_way_confidence_decision(interval, expected):
    assert decide_from_interval(*interval) == expected


def test_clear_constraint_failure_is_learning_candidate():
    result = evaluate_operator(
        _spec(),
        _evidence(error=2.0, uncertainty=0.01, failure=0.01, cost=0.01),
        window_size=20,
        confidence_level=0.95,
        bootstrap_samples=100,
        seed=7,
        dependency_results={},
    )
    assert result.ratio == 2.0
    assert result.decision == Decision.LEARNING_CANDIDATE
    assert "ERROR_THRESHOLD_EXCEEDED" in result.decision_reasons


def test_dependency_failure_blocks_downstream_without_calling_it_failed():
    parent = evaluate_operator(
        _spec(id="depth"),
        _evidence(error=2.0),
        window_size=20,
        confidence_level=0.95,
        bootstrap_samples=100,
        seed=7,
        dependency_results={},
    )
    child_spec = _spec(id="slope", input_dependencies=("depth",))
    child = evaluate_operator(
        child_spec,
        _evidence(),
        window_size=20,
        confidence_level=0.95,
        bootstrap_samples=100,
        seed=8,
        dependency_results={"depth": parent},
    )
    assert parent.decision == Decision.LEARNING_CANDIDATE
    assert child.decision == Decision.DEPENDENCY_BLOCKED
    assert child.blocked_by == ["depth"]


def test_insufficient_coverage_is_undecided():
    result = evaluate_operator(
        _spec(minimum_coverage=0.8),
        _evidence(coverage=0.1),
        window_size=20,
        confidence_level=0.95,
        bootstrap_samples=100,
        seed=7,
        dependency_results={},
    )
    assert result.decision == Decision.UNDECIDED
    assert "INSUFFICIENT_APPLICABILITY_COVERAGE" in result.decision_reasons


def test_unfrozen_threshold_prevents_decision():
    thresholds = {
        name: Threshold(None, "unit") for name in ("error", "uncertainty", "failure", "cost")
    }
    result = evaluate_operator(
        _spec(thresholds=thresholds),
        _evidence(),
        window_size=20,
        confidence_level=0.95,
        bootstrap_samples=100,
        seed=7,
        dependency_results={},
    )
    assert result.decision == Decision.UNDECIDED
    assert result.ratio is None
    assert "THRESHOLD_REQUIRED" in result.decision_reasons


def test_msk_is_distinct_from_legitimate_zero():
    measured_zero = MaskedValue(0.0, True)
    assert measured_zero.value == MSK.value == 0.0
    assert measured_zero.valid and not MSK.valid


def test_final_test_guard_rejects_architecture_selection(tmp_path: Path):
    manifest = tmp_path / "test.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "sample_id": "held-out:0",
                "episode_id": "held-out",
                "frame_index": 0,
                "split": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="final held-out test"):
        _records(manifest, "test", 1)


def test_mask_stability_excludes_unresolved_decisions():
    left = {
        "flow": Decision.UNDECIDED,
        "depth": Decision.ANALYTICAL,
        "drift": Decision.LEARNING_CANDIDATE,
    }
    right = {
        "flow": Decision.ANALYTICAL,
        "depth": Decision.ANALYTICAL,
        "drift": Decision.LEARNING_CANDIDATE,
    }
    assert mask_stability(left, right) == 1.0


def test_progressive_boundary_can_change_then_stabilize():
    five = {"flow": Decision.UNDECIDED}
    ten = {"flow": Decision.ANALYTICAL}
    twenty_five = {"flow": Decision.ANALYTICAL}
    fifty = {"flow": Decision.ANALYTICAL}
    assert mask_stability(five, ten) is None
    assert mask_stability(ten, twenty_five) == 1.0
    assert mask_stability(twenty_five, fifty) == 1.0


def test_short_window_is_low_confidence_and_undecided():
    evidence = _evidence()
    evidence.frames_observed = 5
    result = evaluate_operator(
        _spec(minimum_frames=10),
        evidence,
        window_size=5,
        confidence_level=0.95,
        bootstrap_samples=100,
        seed=7,
        dependency_results={},
    )
    assert result.decision == Decision.UNDECIDED
    assert result.confidence == "LOW"
    assert "INSUFFICIENT_FRAMES" in result.decision_reasons


def test_registry_models_depth_dependency_chain_and_placeholders():
    registry = operator_registry({"default_min_frames": 10, "operators": {}})
    assert registry["depth_gradient"].input_dependencies == ("depth_reconstruction",)
    assert "depth_gradient" in registry["surface_slope"].input_dependencies
    assert registry["image_gradients"].diagnostic_method is None
