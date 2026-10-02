"""Conventional Transformer blocks whose Q, K, and V are all linear projections."""

from __future__ import annotations
import torch
from torch import nn
from .blocks import MBConv, StochasticDepth


class ConventionalViTBlock(nn.Module):
    def __init__(
        self, channels: int, heads: int, dropout: float = 0.0, stochastic_depth: float = 0.0
    ):
        super().__init__()
        if channels % heads:
            raise ValueError("ViT channels must divide by heads")
        self.channels, self.heads, self.head_dim = channels, heads, channels // heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.output = nn.Linear(channels, channels)
        self.drop = StochasticDepth(stochastic_depth)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, feature_map):
        shape = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)
        q, k, v = self.qkv(self.norm1(tokens)).chunk(3, dim=-1)
        heads = lambda x: x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(
            1, 2
        )
        q, k, v = heads(q), heads(k), heads(v)
        attended = ((q @ k.transpose(-2, -1)) * self.head_dim**-0.5).softmax(-1) @ v
        attended = attended.transpose(1, 2).reshape_as(tokens)
        tokens = tokens + self.drop(self.output(attended))
        tokens = tokens + self.drop(self.mlp(self.norm2(tokens)))
        return tokens.transpose(1, 2).reshape(shape)


class ViTBackbone(nn.Module):
    def __init__(
        self,
        embedding: int,
        stage3: int,
        stage4: int,
        heads3: int,
        heads4: int,
        depth3: int = 1,
        depth4: int = 1,
    ):
        super().__init__()
        self.down3 = MBConv(embedding, stage3, stride=2)
        self.stage3 = nn.Sequential(*[ConventionalViTBlock(stage3, heads3) for _ in range(depth3)])
        self.down4 = MBConv(stage3, stage4, stride=2)
        self.stage4 = nn.Sequential(*[ConventionalViTBlock(stage4, heads4) for _ in range(depth4)])
        self.output_channels = stage4

    def forward(self, value):
        batch = value.shape[0]
        value = self.stage3(self.down3(value))
        stage3_tokens = value.shape[2] * value.shape[3]
        value = self.stage4(self.down4(value))
        stage4_tokens = value.shape[2] * value.shape[3]
        stage3_blocks = list(self.stage3)
        stage4_blocks = list(self.stage4)
        pairs = sum(
            batch * block.heads * stage3_tokens * stage3_tokens for block in stage3_blocks
        ) + sum(batch * block.heads * stage4_tokens * stage4_tokens for block in stage4_blocks)
        self.last_attention_stats = {
            "attention_pairs": pairs,
            "dense_reference_pairs": pairs,
            "context_tokens": 0,
            "metric_neighbor_distance_comparisons": 0,
            "neighbors_per_query": max(stage3_tokens, stage4_tokens),
        }
        return value
