"""Export the learned perception model with fixed batch-one deployment shapes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn

from common.models import build_model
from common.registry import (
    CANDIDATE_FEATURE_NAMES,
    CANDIDATE_FEATURE_VERSION,
    MAX_CANDIDATES,
    METRIC_ATTENTION_VERSION,
    MODEL_ARCHITECTURE_VERSION,
    RELATIONAL_NAMES,
    STATE_DIM,
)
from common.runtime import load_checkpoint, save_json


class ExportWrapper(nn.Module):
    """Flatten the shared Python dictionary into stable ONNX output names."""

    def __init__(self, model: nn.Module, family: str):
        super().__init__()
        self.model = model
        self.family = family

    def forward(self, *inputs):
        if self.family == "imf_htransformer":
            outputs = self.model(
                *inputs[:9],
                calibration=inputs[9],
                depth_scale_m=inputs[10],
                candidate_grid=inputs[11],
                candidate_projected_depth_m=inputs[12],
                candidate_projection_validity=inputs[13],
                candidate_visibility=inputs[14],
            )
        else:
            outputs = self.model(*inputs)
        return (
            outputs["hazard_logits"],
            outputs["landing_logits"],
            outputs["semantic_logits"],
            outputs["poi"]["class_logits"],
            outputs["scene_risk_logits"],
            outputs["candidate"]["risk_logits"],
            outputs["candidate"]["landing_safe_logits"],
        )


def make_inputs(family: str):
    state = torch.zeros(1, STATE_DIM)
    state_validity = torch.zeros_like(state)
    relation = torch.zeros(1, len(RELATIONAL_NAMES))
    relation_validity = torch.zeros_like(relation)
    candidates = torch.zeros(1, MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES))
    candidate_validity = torch.zeros(1, MAX_CANDIDATES)
    candidate_feature_validity = torch.zeros_like(candidates)
    if family == "imf_htransformer":
        analytical = torch.zeros(1, 28, 32, 32)
        analytical[:, 22] = 0.1
        analytical_validity = torch.zeros_like(analytical)
        analytical_validity[:, 22] = 1.0
        calibration = torch.tensor([[256.0, 256.0, 220.0, 220.0, 127.5, 127.5]])
        return (
            analytical,
            analytical_validity,
            state,
            state_validity,
            relation,
            relation_validity,
            candidates,
            candidate_validity,
            candidate_feature_validity,
            calibration,
            torch.tensor([20.0]),
            torch.zeros(1, MAX_CANDIDATES, 2),
            torch.zeros(1, MAX_CANDIDATES),
            torch.zeros(1, MAX_CANDIDATES),
            torch.zeros(1, MAX_CANDIDATES, dtype=torch.long),
        ), (
            "analytical",
            "analytical_validity",
            "vehicle_state",
            "state_validity",
            "relations",
            "relation_validity",
            "candidates",
            "candidate_validity",
            "candidate_feature_validity",
            "calibration_wh_fx_fy_cx_cy",
            "depth_scale_m",
            "candidate_grid_xy",
            "candidate_projected_depth_m",
            "candidate_projection_validity",
            "candidate_visibility",
        )
    return (
        torch.zeros(1, 3, 256, 256),
        state,
        state_validity,
        relation,
        relation_validity,
        candidates,
        candidate_validity,
        candidate_feature_validity,
    ), (
        "rgb",
        "vehicle_state",
        "state_validity",
        "relations",
        "relation_validity",
        "candidates",
        "candidate_validity",
        "candidate_feature_validity",
    )


def export(checkpoint_path: str | Path, output_path: str | Path, verify: bool = True) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    family = config["model"]["family"]
    model = build_model(config).eval()
    load_checkpoint(checkpoint_path, model)
    wrapper = ExportWrapper(model, family).eval()
    inputs, input_names = make_inputs(family)
    output_names = (
        "hazard_logits",
        "landing_logits",
        "semantic_logits",
        "poi_logits",
        "scene_risk_logits",
        "candidate_risk_logits",
        "candidate_landing_logits",
    )

    with torch.inference_mode():
        expected = wrapper(*inputs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        selection = None
        if family == "imf_htransformer":
            from common.models.htransformer import HTransformerBlock

            sparse_blocks = [
                module
                for module in model.modules()
                if isinstance(module, HTransformerBlock) and module.attention_mode == "sparse"
            ]
            if any(module.sparse_selection == "metric" for module in sparse_blocks):
                # The legacy ONNX exporter cannot represent the data-dependent
                # argsort used by metric neighbor selection. Keep the metric
                # attention bias, but export the deterministic fixed-grid graph.
                for module in sparse_blocks:
                    module.sparse_selection = "grid"
                selection = "grid_fallback_metric_bias_preserved"
        try:
            torch.onnx.export(
                wrapper,
                inputs,
                str(temporary),
                input_names=list(input_names),
                output_names=list(output_names),
                opset_version=18,
                dynamo=False,
            )
            expected = wrapper(*inputs)
        except Exception:
            if family != "imf_htransformer" or selection is not None:
                raise
            # Explicit deployment fallback: preserve metric attention bias and
            # gathered K-neighbor computation, but use the static grid graph
            # when the portable ONNX exporter cannot encode dynamic metric sort.
            for module in model.modules():
                if isinstance(module, HTransformerBlock) and module.attention_mode == "sparse":
                    module.sparse_selection = "grid"
            selection = "grid_fallback_metric_bias_preserved"
            torch.onnx.export(
                wrapper,
                inputs,
                str(temporary),
                input_names=list(input_names),
                output_names=list(output_names),
                opset_version=18,
                dynamo=False,
            )
            expected = wrapper(*inputs)
        os.replace(temporary, output_path)

    verified = False
    if verify:
        try:
            import onnx

            onnx.checker.check_model(onnx.load(str(output_path)))
            import onnxruntime as ort
        except ImportError:
            ort = None
        if ort is not None:
            session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            feeds = {name: value.cpu().numpy() for name, value in zip(input_names, inputs)}
            actual = session.run(list(output_names), feeds)
            for expected_value, actual_value in zip(expected, actual):
                torch.testing.assert_close(
                    torch.from_numpy(actual_value), expected_value, rtol=1e-3, atol=1e-4
                )
            verified = True

    save_json(
        output_path.with_suffix(".json"),
        {
            "schema": "imagination-onnx-v2",
            "family": family,
            "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
            "candidate_feature_version": CANDIDATE_FEATURE_VERSION,
            "metric_attention_version": METRIC_ATTENTION_VERSION,
            "fixed_batch": 1,
            "input_names": input_names,
            "output_names": output_names,
            "onnxruntime_numerical_check": "PASS" if verified else "NOT_AVAILABLE",
            "sparse_selection": selection
            or ("metric" if family == "imf_htransformer" else "not_applicable"),
            "metric_attention_bias": "preserved"
            if family == "imf_htransformer"
            else "not_applicable",
            "note": "C++ IMF extraction and persistent map are outside this model export; IMF exports take fixed-shape calibration and projected candidate-map inputs.",
        },
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    print(export(args.checkpoint, args.output, verify=not args.no_verify))


if __name__ == "__main__":
    main()
