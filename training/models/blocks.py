"""Reusable convolutional blocks for local learned processing."""

from __future__ import annotations

import torch
from torch import nn


class StochasticDepth(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = probability

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return value
        keep = 1 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mask = torch.empty(shape, device=value.device).bernoulli_(keep)
        return value * mask / keep


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, ratio: float):
        super().__init__()
        hidden = max(1, int(channels * ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.gate(self.pool(value))


class MBConv(nn.Module):
    """Mobile inverted bottleneck with required depthwise local processing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 2.0,
        kernel_size: int = 3,
        stride: int = 1,
        se_ratio: float = 0.25,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("MBConv kernel size must be odd")
        expanded = max(in_channels, int(round(in_channels * expansion)))
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.SiLU(),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                expanded, expanded, kernel_size, stride=stride,
                padding=kernel_size // 2, groups=expanded, bias=False,
            ),
            nn.BatchNorm2d(expanded),
            nn.SiLU(),
        )
        self.se = SqueezeExcitation(expanded, se_ratio)
        self.project = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(expanded, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.drop_path = StochasticDepth(stochastic_depth)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        local = self.project(self.se(self.depthwise(self.expand(value))))
        return value + self.drop_path(local) if self.use_residual else local
