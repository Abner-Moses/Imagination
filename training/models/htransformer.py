"""CoHAtNet-style attention: linear Q/K and spatial MBConv-derived V."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import MBConv, StochasticDepth


class RelativePositionBias(nn.Module):
    def __init__(self, height: int, width: int, heads: int):
        super().__init__()
        self.height = height
        self.width = width
        self.table = nn.Parameter(torch.zeros(heads, (2 * height - 1) * (2 * width - 1)))
        coordinates = torch.stack(torch.meshgrid(
            torch.arange(height), torch.arange(width), indexing="ij"
        )).flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative[0] += height - 1
        relative[1] += width - 1
        index = relative[0] * (2 * width - 1) + relative[1]
        self.register_buffer("index", index, persistent=False)
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self) -> torch.Tensor:
        return self.table[:, self.index.reshape(-1)].reshape(
            self.table.shape[0], self.height * self.width, self.height * self.width
        )


class HTransformerBlock(nn.Module):
    """Q/K select global relations; V is tokenized MBConv spatial output."""

    def __init__(
        self,
        channels: int,
        spatial_size: tuple[int, int],
        heads: int,
        expansion: float = 2.0,
        kernel_size: int = 3,
        se_ratio: float = 0.25,
        dropout: float = 0.0,
        stochastic_depth: float = 0.0,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        if channels % heads:
            raise ValueError("HTransformer channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.norm_attention = nn.GroupNorm(1, channels)
        self.query = nn.Linear(channels, channels, bias=True)
        self.key = nn.Linear(channels, channels, bias=True)
        self.value_local = MBConv(
            channels, channels, expansion, kernel_size, 1, se_ratio,
            dropout, stochastic_depth,
        )
        self.relative_bias = RelativePositionBias(*spatial_size, heads)
        self.output = nn.Sequential(nn.Linear(channels, channels), nn.Dropout(dropout))
        self.drop_path = StochasticDepth(stochastic_depth)
        hidden = int(channels * mlp_ratio)
        self.norm_mlp = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, 1), nn.Dropout(dropout),
        )

    def _heads(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, _ = tokens.shape
        return tokens.reshape(batch, count, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        tokens = self.norm_attention(feature_map).flatten(2).transpose(1, 2)
        query = self._heads(self.query(tokens))
        key = self._heads(self.key(tokens))

        # There is deliberately no Linear(V). MBConv(F), including its
        # depthwise convolution and SE gate, is the sole Value source.
        local_map = self.value_local(feature_map)
        value = self._heads(local_map.flatten(2).transpose(1, 2))
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.head_dim ** -0.5
        scores = scores + self.relative_bias().unsqueeze(0)
        attended = torch.matmul(scores.softmax(dim=-1), value)
        attended = attended.transpose(1, 2).reshape(tokens.shape)
        attended = self.output(attended).transpose(1, 2).reshape_as(feature_map)
        feature_map = feature_map + self.drop_path(attended)
        return feature_map + self.drop_path(self.mlp(self.norm_mlp(feature_map)))


class HTransformerBackbone(nn.Module):
    """Shared 192x32x32 -> 384x16x16 -> 768x8x8 backbone."""

    def __init__(self, config: dict):
        super().__init__()
        stage3_channels = int(config.get("stage3_channels", 384))
        stage4_channels = int(config.get("stage4_channels", 768))
        expansion = float(config.get("mbconv_expansion", 2.0))
        kernel = int(config.get("mbconv_kernel", 3))
        se_ratio = float(config.get("se_ratio", 0.25))
        dropout = float(config.get("dropout", 0.0))
        drop_path = float(config.get("stochastic_depth", 0.0))
        stage3_depth = int(config.get("stage3_depth", 1))
        stage4_depth = int(config.get("stage4_depth", 1))
        self.stage3_downsample = MBConv(192, stage3_channels, expansion, kernel, 2, se_ratio)
        self.stage3 = nn.Sequential(*[
            HTransformerBlock(
                stage3_channels, (16, 16), int(config.get("stage3_heads", 6)),
                expansion, kernel, se_ratio, dropout, drop_path,
            ) for _ in range(stage3_depth)
        ])
        self.stage4_downsample = MBConv(
            stage3_channels, stage4_channels, expansion, kernel, 2, se_ratio
        )
        self.stage4 = nn.Sequential(*[
            HTransformerBlock(
                stage4_channels, (8, 8), int(config.get("stage4_heads", 12)),
                expansion, kernel, se_ratio, dropout, drop_path,
            ) for _ in range(stage4_depth)
        ])
        self.output_channels = stage4_channels

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        feature_map = self.stage3(self.stage3_downsample(feature_map))
        return self.stage4(self.stage4_downsample(feature_map))
