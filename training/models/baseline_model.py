"""RGB learned-front-end baseline with the shared downstream architecture."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import MBConv
from .htransformer import HTransformerBackbone
from .heads import TaskHeads


class LearnedRgbFrontEnd(nn.Module):
    """Learned 256x256 RGB -> 192x32x32 low-level feature extractor."""

    def __init__(self, config: dict):
        super().__init__()
        expansion = float(config.get("mbconv_expansion", 2.0))
        kernel = int(config.get("mbconv_kernel", 3))
        se_ratio = float(config.get("se_ratio", 0.25))
        dropout = float(config.get("dropout", 0.0))
        stochastic_depth = float(config.get("stochastic_depth", 0.0))
        self.layers = nn.Sequential(
            nn.Conv2d(3, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.SiLU(),
            MBConv(48, 96, expansion, kernel, 2, se_ratio, dropout, stochastic_depth),
            MBConv(96, 192, expansion, kernel, 2, se_ratio, dropout, stochastic_depth),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.shape[1:] != (3, 256, 256):
            raise ValueError("RGB input must be Bx3x256x256")
        return self.layers(rgb)


class BaselineModel(nn.Module):
    def __init__(self, htransformer: dict, state_dim: int = 10):
        super().__init__()
        self.front_end = LearnedRgbFrontEnd(htransformer)
        self.backbone = HTransformerBackbone(htransformer)
        self.heads = TaskHeads(self.backbone.output_channels, state_dim)

    def forward(
        self,
        rgb: torch.Tensor,
        state_values: torch.Tensor | None = None,
        state_validity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.heads(
            self.backbone(self.front_end(rgb)), state_values, state_validity
        )

    @classmethod
    def from_config(cls, config: dict) -> "BaselineModel":
        return cls(config["htransformer"], int(config["model"].get("state_dim", 10)))
