"""Validity-aware learned model following the existing C++ pre-extractor."""

from __future__ import annotations

import torch
from torch import nn

from training.common import ANALYTICAL_CHANNELS
from .htransformer import HTransformerBackbone
from .heads import TaskHeads


FAMILY_CHANNELS = {
    "appearance": ("Y", "Cb", "Cr"),
    "gradients": ("Gx", "Gy", "GradientMagnitude"),
    "hog": tuple(f"HOG_{index}" for index in range(9)),
    "harris": ("HarrisResponse",),
    "canny": ("CannyEdge",),
    "contours": ("ContourMap",),
    "chroma": ("ChromaGradientCb", "ChromaGradientCr"),
    "flow": ("OpticalFlowU", "OpticalFlowV"),
    "geometry": (
        "Depth", "DepthGradientX", "DepthGradientY", "Slope", "Roughness",
        "GeometryConfidence",
    ),
    "depth_gradients": ("DepthGradientX", "DepthGradientY"),
    "slope": ("Slope",),
    "roughness": ("Roughness",),
}


class AnalyticalModel(nn.Module):
    def __init__(
        self,
        htransformer: dict,
        state_dim: int = 10,
        disabled_families: list[str] | None = None,
    ):
        super().__init__()
        enabled = torch.ones(1, 28, 1, 1)
        for family in disabled_families or []:
            if family not in FAMILY_CHANNELS:
                raise ValueError(f"Unknown analytical ablation family: {family}")
            for channel in FAMILY_CHANNELS[family]:
                enabled[:, ANALYTICAL_CHANNELS.index(channel)] = 0
        self.register_buffer("ablation_mask", enabled, persistent=True)
        self.adapter = nn.Sequential(
            nn.Conv2d(56, 192, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
        )
        self.backbone = HTransformerBackbone(htransformer)
        self.heads = TaskHeads(self.backbone.output_channels, state_dim)

    @staticmethod
    def mask_aware_input(
        analytical: torch.Tensor, validity: torch.Tensor
    ) -> torch.Tensor:
        if analytical.shape != validity.shape or analytical.shape[1:] != (28, 32, 32):
            raise ValueError("Analytical features and validity must be Bx28x32x32")
        validity = validity.to(analytical.dtype)
        return torch.cat((analytical * validity, validity), dim=1)

    def forward(
        self,
        analytical: torch.Tensor,
        validity: torch.Tensor,
        state_values: torch.Tensor | None = None,
        state_validity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        validity = validity * self.ablation_mask
        adapted = self.adapter(self.mask_aware_input(analytical, validity))
        return self.heads(self.backbone(adapted), state_values, state_validity)

    @classmethod
    def from_config(cls, config: dict) -> "AnalyticalModel":
        model = config["model"]
        return cls(
            config["htransformer"], int(model.get("state_dim", 10)),
            list(model.get("disabled_families", [])),
        )
