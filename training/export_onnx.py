"""Export only the learned analytical model; C++ preprocessing stays outside ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training.common import load_checkpoint
from training.models import build_model


class OnnxOutputs(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, analytical, validity, state_values, state_validity):
        result = self.model(analytical, validity, state_values, state_validity)
        return result["hazard_logits"], result["waypoint"]


def export(checkpoint_path: str | Path, destination: str | Path) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["model"]["type"] != "analytical":
        raise ValueError("This deployment export is for the analytical learned model")
    model = build_model(config).eval()
    load_checkpoint(checkpoint_path, model)
    wrapper = OnnxOutputs(model).eval()
    state_dim = int(config["model"].get("state_dim", 10))
    inputs = (
        torch.zeros(1, 28, 32, 32),
        torch.ones(1, 28, 32, 32),
        torch.zeros(1, state_dim),
        torch.zeros(1, state_dim),
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, inputs, destination,
        input_names=("analytical", "validity", "state_values", "state_validity"),
        output_names=("hazard_logits", "waypoint"),
        opset_version=18,
        dynamic_axes={
            "analytical": {0: "batch"}, "validity": {0: "batch"},
            "state_values": {0: "batch"}, "state_validity": {0: "batch"},
            "hazard_logits": {0: "batch"}, "waypoint": {0: "batch"},
        },
        dynamo=False,
    )
    try:
        import onnxruntime as runtime
    except ImportError:
        print("onnxruntime unavailable; skipped numerical export validation")
        return destination
    session = runtime.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    feed = {name: value.numpy() for name, value in zip(
        ("analytical", "validity", "state_values", "state_validity"), inputs
    )}
    onnx_outputs = session.run(None, feed)
    with torch.inference_mode():
        torch_outputs = wrapper(*inputs)
    for expected, actual in zip(torch_outputs, onnx_outputs):
        np.testing.assert_allclose(expected.numpy(), actual, rtol=1e-4, atol=1e-5)
    print("ONNX Runtime output matches PyTorch")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    print(export(arguments.checkpoint, arguments.output))


if __name__ == "__main__":
    main()
