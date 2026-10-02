"""CoHAtNet-style attention: linear Q/K and spatial MBConv-derived V."""

from __future__ import annotations

from time import perf_counter
import torch
from torch import nn

from .blocks import MBConv, StochasticDepth
from .metric_attention import (
    downsample_imf_descriptors,
    grid_neighborhood,
    metric_neighborhood,
    spatial_attention,
)
from .query_selection import active_query_indices, downsample_importance, downsample_mandatory
from common.registry import CANDIDATE_FEATURE_NAMES


def _aggregate_stage_stats(blocks):
    keys = (
        "attention_pairs",
        "dense_reference_pairs",
        "metric_neighbor_distance_comparisons",
        "active_queries",
        "imf_dissimilarity_pairs",
        "query_selection_ms",
        "q_projection_ms",
        "k_projection_ms",
        "mbconv_value_ms",
        "neighbor_selection_ms",
        "sparse_gather_ms",
        "context_projection_ms",
        "context_attention_ms",
        "attention_ms",
        "ffn_ms",
    )
    result = {
        key: sum(float(block.last_attention_stats.get(key, 0.0)) for block in blocks)
        for key in keys
    }
    result["neighbors_per_query"] = max(
        (int(block.last_attention_stats.get("neighbors_per_query", 0)) for block in blocks),
        default=0,
    )
    result["context_tokens"] = max(
        (int(block.last_attention_stats.get("context_tokens", 0)) for block in blocks),
        default=0,
    )
    return result


class RelativePositionBias(nn.Module):
    def __init__(self, height: int, width: int, heads: int):
        super().__init__()
        self.height = height
        self.width = width
        self.table = nn.Parameter(torch.zeros(heads, (2 * height - 1) * (2 * width - 1)))
        coordinates = (
            torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij"))
            .flatten(1)
            .transpose(0, 1)
        )
        self.register_buffer("coordinates", coordinates, persistent=False)
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(
        self, neighbors: torch.Tensor | None = None, query_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        if neighbors is None and query_indices is None:
            relative = self.coordinates[:, None, :] - self.coordinates[None, :, :]
            row = relative[..., 0] + self.height - 1
            column = relative[..., 1] + self.width - 1
            index = row * (2 * self.width - 1) + column
            return self.table[:, index.reshape(-1)].reshape(
                self.table.shape[0], self.height * self.width, index.shape[-1]
            )
        if query_indices is not None:
            query_coordinates = self.coordinates[query_indices]
            if neighbors is None:
                key_coordinates = self.coordinates[None, None, :, :]
            else:
                key_coordinates = self.coordinates[neighbors]
            relative = query_coordinates[:, :, None, :] - key_coordinates
        elif neighbors.ndim == 3:
            query_coordinates = self.coordinates[None, :, None, :]
            relative = query_coordinates - self.coordinates[neighbors]
        else:
            relative = self.coordinates[:, None, :] - self.coordinates[neighbors]
        row = relative[..., 0] + self.height - 1
        column = relative[..., 1] + self.width - 1
        index = row * (2 * self.width - 1) + column
        if query_indices is not None:
            return (
                self.table[:, index.reshape(-1)]
                .reshape(self.table.shape[0], *index.shape)
                .permute(1, 0, 2, 3)
            )
        if neighbors.ndim == 3:
            return (
                self.table[:, index.reshape(-1)]
                .reshape(self.table.shape[0], *index.shape)
                .permute(1, 0, 2, 3)
            )
        return self.table[:, index.reshape(-1)].reshape(
            self.table.shape[0], self.height * self.width, index.shape[-1]
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
        attention_mode: str = "dense",
        qk_channels: int | None = None,
        position_bias_mode: str = "relative",
        metric_weight: float = 0.5,
        metric_scale_m: float = 4.0,
        sparse_neighbors: int = 32,
        qk_normalize: bool = False,
        context_features: int = 0,
        sparse_selection: str = "metric",
        capture_attention: bool = False,
        profile_timing: bool = False,
        active_query_fraction: float = 1.0,
        active_query_coverage_bins: tuple[int, int] = (4, 4),
        imf_dissimilarity_weight: float = 0.0,
        imf_dissimilarity_scales=(0.25, 0.25, 0.25, 0.25),
        imf_dissimilarity_weights=(1.0, 0.5, 0.5, 0.5),
    ):
        super().__init__()
        if channels % heads:
            raise ValueError("HTransformer channels must be divisible by heads")
        qk_channels = channels if qk_channels is None else int(qk_channels)
        if qk_channels < 1 or qk_channels % heads:
            raise ValueError("Q/K channels must be positive and divisible by attention heads")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.qk_channels = qk_channels
        self.qk_head_dim = qk_channels // heads
        self.spatial_size = spatial_size
        self.attention_mode = attention_mode
        self.position_bias_mode = position_bias_mode
        self.metric_weight = float(metric_weight)
        self.metric_scale_m = float(metric_scale_m)
        self.sparse_neighbors = int(sparse_neighbors)
        self.qk_normalize = bool(qk_normalize)
        if sparse_selection not in {"metric", "grid"}:
            raise ValueError("sparse_selection must be metric or grid")
        self.sparse_selection = sparse_selection
        self.capture_attention = bool(capture_attention)
        self.profile_timing = bool(profile_timing)
        self.active_query_fraction = float(active_query_fraction)
        if not 0.0 < self.active_query_fraction <= 1.0:
            raise ValueError("active_query_fraction must be in (0,1]")
        self.active_query_coverage_bins = tuple(int(value) for value in active_query_coverage_bins)
        if imf_dissimilarity_weight < 0:
            raise ValueError("IMF dissimilarity weight must be nonnegative")
        self.imf_dissimilarity_weight = float(imf_dissimilarity_weight)
        self.imf_dissimilarity_scales = tuple(float(v) for v in imf_dissimilarity_scales)
        self.imf_dissimilarity_weights = tuple(float(v) for v in imf_dissimilarity_weights)
        self.last_attention_debug = None
        # Map tokens are an unordered set, not a spatial lattice. Their key
        # and value projections are independent pointwise maps; spatial V alone
        # uses MBConv over a regular image grid.
        self.context_key_encoder = (
            nn.Linear(context_features, qk_channels) if context_features else None
        )
        self.context_value_encoder = (
            nn.Linear(context_features, channels) if context_features else None
        )
        self.norm_attention = nn.GroupNorm(1, channels)
        self.query = nn.Linear(channels, qk_channels, bias=True)
        self.key = nn.Linear(channels, qk_channels, bias=True)
        self.value_local = MBConv(
            channels,
            channels,
            expansion,
            kernel_size,
            1,
            se_ratio,
            dropout,
            stochastic_depth,
        )
        self.relative_bias = (
            RelativePositionBias(*spatial_size, heads)
            if position_bias_mode in {"relative", "both"}
            else None
        )
        if attention_mode == "sparse":
            total = spatial_size[0] * spatial_size[1]
            pool = min(total, max(self.sparse_neighbors, self.sparse_neighbors * 4, 8))
            self.register_buffer(
                "neighbor_indices",
                grid_neighborhood(*spatial_size, pool),
                persistent=False,
            )
        elif attention_mode == "dense":
            self.register_buffer("neighbor_indices", None, persistent=False)
        else:
            raise ValueError("attention_mode must be dense or sparse")
        self.output = nn.Sequential(nn.Linear(channels, channels), nn.Dropout(dropout))
        self.drop_path = StochasticDepth(stochastic_depth)
        hidden = int(channels * mlp_ratio)
        self.norm_mlp = nn.GroupNorm(1, channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden, channels, 1),
            nn.Dropout(dropout),
        )

    def _heads(self, tokens: torch.Tensor, head_dim: int | None = None) -> torch.Tensor:
        batch, count, _ = tokens.shape
        head_dim = self.head_dim if head_dim is None else int(head_dim)
        return tokens.reshape(batch, count, self.heads, head_dim).transpose(1, 2)

    @staticmethod
    def _clock(device, enabled):
        if not enabled:
            return 0.0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return perf_counter()

    def forward(
        self,
        feature_map: torch.Tensor,
        metric_positions=None,
        metric_validity=None,
        context_tokens=None,
        context_positions=None,
        context_validity=None,
        query_importance=None,
        query_mandatory=None,
        imf_descriptors=None,
        imf_validity=None,
        imf_reliability=None,
    ) -> torch.Tensor:
        timing = {}
        tokens = self.norm_attention(feature_map).flatten(2).transpose(1, 2)
        query_indices = query_validity = None
        if self.active_query_fraction < 1.0:
            if query_importance is None:
                raise ValueError("Active-query attention requires IMF query-importance maps")
            t = self._clock(feature_map.device, self.profile_timing)
            query_indices, query_validity = active_query_indices(
                query_importance,
                query_mandatory,
                self.active_query_fraction,
                coverage_bins=self.active_query_coverage_bins,
            )
            if self.profile_timing:
                timing["query_selection_ms"] = (self._clock(feature_map.device, True) - t) * 1000.0
            batch_ids = torch.arange(tokens.shape[0], device=tokens.device)[:, None]
            query_tokens = tokens[batch_ids, query_indices]
        else:
            query_tokens = tokens
            query_indices = torch.arange(tokens.shape[1], device=tokens.device)[None, :].expand(
                tokens.shape[0], -1
            )
            query_validity = tokens.new_ones(tokens.shape[:2])
        t = self._clock(feature_map.device, self.profile_timing)
        query = self._heads(self.query(query_tokens), self.qk_head_dim)
        if self.profile_timing:
            timing["q_projection_ms"] = (self._clock(feature_map.device, True) - t) * 1000.0
        t = self._clock(feature_map.device, self.profile_timing)
        key = self._heads(self.key(tokens), self.qk_head_dim)
        if self.profile_timing:
            timing["k_projection_ms"] = (self._clock(feature_map.device, True) - t) * 1000.0

        # There is deliberately no Linear(V). MBConv(F), including its
        # depthwise convolution and SE gate, is the sole Value source.
        t = self._clock(feature_map.device, self.profile_timing)
        local_map = self.value_local(feature_map)
        value = self._heads(local_map.flatten(2).transpose(1, 2))
        if self.profile_timing:
            timing["mbconv_value_ms"] = (self._clock(feature_map.device, True) - t) * 1000.0
        context_key = context_value = None
        if context_tokens is not None:
            if self.context_key_encoder is None or self.context_value_encoder is None:
                raise ValueError(
                    "Map context tokens were supplied to a backbone without a context adapter"
                )
            t = self._clock(feature_map.device, self.profile_timing)
            context_key = self._heads(self.context_key_encoder(context_tokens), self.qk_head_dim)
            context_value = self._heads(self.context_value_encoder(context_tokens), self.head_dim)
            if self.profile_timing:
                timing["context_projection_ms"] = (
                    self._clock(feature_map.device, True) - t
                ) * 1000.0
        neighbor_indices = self.neighbor_indices
        t = self._clock(feature_map.device, self.profile_timing)
        if self.attention_mode == "sparse":
            if metric_positions is not None and self.sparse_selection == "metric":
                flattened_positions = metric_positions.reshape(tokens.shape[0], -1, 3)
                if metric_validity is None:
                    flattened_validity = flattened_positions.new_ones(flattened_positions.shape[:2])
                else:
                    flattened_validity = metric_validity.reshape(tokens.shape[0], -1)
                neighbor_indices = metric_neighborhood(
                    flattened_positions,
                    flattened_validity,
                    self.neighbor_indices,
                    self.sparse_neighbors,
                    grid_shape=self.spatial_size,
                    query_indices=query_indices if self.active_query_fraction < 1.0 else None,
                )
            else:
                neighbor_indices = self.neighbor_indices[query_indices, : self.sparse_neighbors]
        if self.profile_timing:
            timing["neighbor_selection_ms"] = (self._clock(feature_map.device, True) - t) * 1000.0
        bias = None
        if self.relative_bias is not None:
            bias = self.relative_bias(
                neighbor_indices if self.attention_mode == "sparse" else None,
                query_indices if self.active_query_fraction < 1.0 else None,
            )
        attention_debug = {} if self.capture_attention else None
        attended, stats = spatial_attention(
            query,
            key,
            value,
            grid_shape=self.spatial_size,
            mode=self.attention_mode,
            positions=metric_positions,
            position_validity=metric_validity,
            relative_bias=bias,
            position_bias_mode=self.position_bias_mode,
            metric_weight=self.metric_weight,
            metric_scale_m=self.metric_scale_m,
            neighbor_indices=neighbor_indices,
            qk_normalize=self.qk_normalize,
            context_key=context_key,
            context_value=context_value,
            context_positions=context_positions,
            context_validity=context_validity,
            attention_debug=attention_debug,
            profile_timing=self.profile_timing,
            query_indices=query_indices if self.active_query_fraction < 1.0 else None,
            query_validity=query_validity if self.active_query_fraction < 1.0 else None,
            imf_descriptors=imf_descriptors,
            imf_validity=imf_validity,
            imf_reliability=imf_reliability,
            imf_scales=self.imf_dissimilarity_scales,
            imf_weights=self.imf_dissimilarity_weights,
            imf_bias_weight=self.imf_dissimilarity_weight,
        )
        if self.profile_timing:
            timing["attention_ms"] = stats.pop("attention_ms", 0.0)
            timing["context_attention_ms"] = stats.pop("context_attention_ms", 0.0)
        stats["neighbors_per_query"] = (
            int(neighbor_indices.shape[-1])
            if neighbor_indices is not None
            else int(tokens.shape[1])
        )
        stats["metric_neighbor_distance_comparisons"] = (
            int(tokens.shape[0] * query_tokens.shape[1] * self.neighbor_indices.shape[-1])
            if self.attention_mode == "sparse"
            and self.sparse_selection == "metric"
            and metric_positions is not None
            else 0
        )
        stats["active_queries"] = (
            int(query_validity.sum().item())
            if self.active_query_fraction < 1.0
            else int(tokens.shape[0] * tokens.shape[1])
        )
        self.last_attention_stats = stats
        self.last_attention_stats.update(timing)
        if attention_debug is not None:
            attention_debug.update(
                {
                    "grid_shape": self.spatial_size,
                    "mode": self.attention_mode,
                    "selection": self.sparse_selection,
                    "position_bias_mode": self.position_bias_mode,
                    "metric_weight": self.metric_weight,
                    "metric_scale_m": self.metric_scale_m,
                    "neighbor_indices": None
                    if neighbor_indices is None
                    else neighbor_indices.detach().cpu(),
                    "positions": None
                    if metric_positions is None
                    else metric_positions.detach().float().cpu(),
                    "position_validity": None
                    if metric_validity is None
                    else metric_validity.detach().float().cpu(),
                    "context_positions": None
                    if context_positions is None
                    else context_positions.detach().float().cpu(),
                    "context_validity": None
                    if context_validity is None
                    else context_validity.detach().float().cpu(),
                }
            )
            self.last_attention_debug = attention_debug
        attended = attended.transpose(1, 2).reshape(query_tokens.shape)
        attended = self.output(attended) * query_validity.unsqueeze(-1)
        if self.active_query_fraction < 1.0:
            scattered = torch.zeros_like(tokens)
            scatter_index = query_indices[..., None].expand(-1, -1, tokens.shape[-1])
            scattered.scatter_add_(1, scatter_index, attended)
            attended = scattered
        attended = attended.transpose(1, 2).reshape_as(feature_map)
        feature_map = feature_map + self.drop_path(attended)
        t = self._clock(feature_map.device, self.profile_timing)
        feature_map = feature_map + self.drop_path(self.mlp(self.norm_mlp(feature_map)))
        if self.profile_timing:
            self.last_attention_stats["ffn_ms"] = (
                self._clock(feature_map.device, True) - t
            ) * 1000.0
        return feature_map


class HTransformerBackbone(nn.Module):
    """Configurable 32x32 -> 16x16 -> 8x8 MBConv-Value backbone."""

    def __init__(self, config: dict, input_channels: int | None = None):
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
        attention_mode = str(config.get("attention_mode", "dense"))
        stage3_attention_mode = str(config.get("stage3_attention_mode", attention_mode))
        stage4_attention_mode = str(config.get("stage4_attention_mode", attention_mode))
        stage3_sparse_neighbors = int(
            config.get("stage3_sparse_neighbors", config.get("sparse_neighbors", 32))
        )
        stage4_sparse_neighbors = int(
            config.get("stage4_sparse_neighbors", config.get("sparse_neighbors", 32))
        )
        stage3_active_query_fraction = float(config.get("stage3_active_query_fraction", 1.0))
        stage4_active_query_fraction = float(config.get("stage4_active_query_fraction", 1.0))
        imf_dissimilarity_weight = float(config.get("imf_dissimilarity_weight", 0.0))
        imf_dissimilarity_scales = tuple(
            config.get("imf_dissimilarity_scales", (0.25, 0.25, 0.25, 0.25))
        )
        imf_dissimilarity_weights = tuple(
            config.get("imf_dissimilarity_weights", (1.0, 0.5, 0.5, 0.5))
        )
        position_bias_mode = str(config.get("position_bias_mode", "relative"))
        metric_weight = float(config.get("metric_weight", 0.5))
        metric_scale_m = float(config.get("metric_scale_m", 4.0))
        qk_normalize = bool(config.get("qk_normalize", False))
        sparse_selection = str(config.get("sparse_selection", "metric"))
        capture_attention = bool(config.get("capture_attention", False))
        profile_timing = bool(config.get("profile_timing", False))
        context_features = int(config.get("context_features", 0))
        self.stage3_use_map_context = bool(config.get("stage3_use_map_context", True))
        self.stage4_use_map_context = bool(config.get("stage4_use_map_context", True))
        input_channels = int(input_channels or config.get("embedding", 192))
        self.stage3_downsample = MBConv(
            input_channels, stage3_channels, expansion, kernel, 2, se_ratio
        )
        self.stage3 = nn.Sequential(
            *[
                HTransformerBlock(
                    stage3_channels,
                    (16, 16),
                    int(config.get("stage3_heads", 6)),
                    expansion,
                    kernel,
                    se_ratio,
                    dropout,
                    drop_path,
                    attention_mode=stage3_attention_mode,
                    qk_channels=int(config.get("stage3_qk_channels", stage3_channels)),
                    mlp_ratio=float(config.get("stage3_mlp_ratio", config.get("mlp_ratio", 2.0))),
                    position_bias_mode=position_bias_mode,
                    metric_weight=metric_weight,
                    metric_scale_m=metric_scale_m,
                    sparse_neighbors=stage3_sparse_neighbors,
                    qk_normalize=qk_normalize,
                    context_features=context_features if self.stage3_use_map_context else 0,
                    sparse_selection=str(config.get("stage3_sparse_selection", sparse_selection)),
                    capture_attention=capture_attention,
                    profile_timing=profile_timing,
                    active_query_fraction=stage3_active_query_fraction,
                    active_query_coverage_bins=tuple(
                        config.get("stage3_active_query_coverage_bins", (4, 4))
                    ),
                    imf_dissimilarity_weight=imf_dissimilarity_weight,
                    imf_dissimilarity_scales=imf_dissimilarity_scales,
                    imf_dissimilarity_weights=imf_dissimilarity_weights,
                )
                for _ in range(stage3_depth)
            ]
        )
        self.stage4_downsample = MBConv(
            stage3_channels, stage4_channels, expansion, kernel, 2, se_ratio
        )
        self.stage4 = nn.Sequential(
            *[
                HTransformerBlock(
                    stage4_channels,
                    (8, 8),
                    int(config.get("stage4_heads", 12)),
                    expansion,
                    kernel,
                    se_ratio,
                    dropout,
                    drop_path,
                    attention_mode=stage4_attention_mode,
                    qk_channels=int(config.get("stage4_qk_channels", stage4_channels)),
                    mlp_ratio=float(config.get("stage4_mlp_ratio", config.get("mlp_ratio", 2.0))),
                    position_bias_mode=position_bias_mode,
                    metric_weight=metric_weight,
                    metric_scale_m=metric_scale_m,
                    sparse_neighbors=stage4_sparse_neighbors,
                    qk_normalize=qk_normalize,
                    context_features=context_features if self.stage4_use_map_context else 0,
                    sparse_selection=str(config.get("stage4_sparse_selection", sparse_selection)),
                    capture_attention=capture_attention,
                    profile_timing=profile_timing,
                    active_query_fraction=stage4_active_query_fraction,
                    active_query_coverage_bins=tuple(
                        config.get("stage4_active_query_coverage_bins", (4, 4))
                    ),
                    imf_dissimilarity_weight=imf_dissimilarity_weight,
                    imf_dissimilarity_scales=imf_dissimilarity_scales,
                    imf_dissimilarity_weights=imf_dissimilarity_weights,
                )
                for _ in range(stage4_depth)
            ]
        )
        self.output_channels = stage4_channels

    def forward(
        self,
        feature_map: torch.Tensor,
        metric_positions=None,
        metric_validity=None,
        context_tokens=None,
        context_positions=None,
        context_validity=None,
        query_importance=None,
        query_mandatory=None,
        imf_descriptors=None,
        imf_descriptor_validity=None,
        imf_reliability=None,
    ) -> torch.Tensor:
        feature_map = self.stage3_downsample(feature_map)
        stage3_positions = stage3_validity = None
        if metric_positions is not None:
            from .metric_attention import downsample_metric_positions

            stage3_positions, stage3_validity = downsample_metric_positions(
                metric_positions, metric_validity, (16, 16)
            )
        for block in self.stage3:
            descriptor_args = (None, None, None)
            if imf_descriptors is not None:
                descriptor_args = downsample_imf_descriptors(
                    imf_descriptors, imf_descriptor_validity, imf_reliability, (16, 16)
                )
            feature_map = block(
                feature_map,
                stage3_positions,
                stage3_validity,
                context_tokens if self.stage3_use_map_context else None,
                context_positions if self.stage3_use_map_context else None,
                context_validity if self.stage3_use_map_context else None,
                downsample_importance(query_importance, (16, 16))
                if query_importance is not None
                else None,
                downsample_mandatory(query_mandatory, (16, 16))
                if query_mandatory is not None
                else None,
                *descriptor_args,
            )

        feature_map = self.stage4_downsample(feature_map)
        stage4_positions = stage4_validity = None
        if metric_positions is not None:
            from .metric_attention import downsample_metric_positions

            stage4_positions, stage4_validity = downsample_metric_positions(
                metric_positions, metric_validity, (8, 8)
            )
        for block in self.stage4:
            descriptor_args = (None, None, None)
            if imf_descriptors is not None:
                descriptor_args = downsample_imf_descriptors(
                    imf_descriptors, imf_descriptor_validity, imf_reliability, (8, 8)
                )
            feature_map = block(
                feature_map,
                stage4_positions,
                stage4_validity,
                context_tokens if self.stage4_use_map_context else None,
                context_positions if self.stage4_use_map_context else None,
                context_validity if self.stage4_use_map_context else None,
                downsample_importance(query_importance, (8, 8))
                if query_importance is not None
                else None,
                downsample_mandatory(query_mandatory, (8, 8))
                if query_mandatory is not None
                else None,
                *descriptor_args,
            )
        blocks = [*self.stage3, *self.stage4]
        self.last_attention_stats = {
            "attention_pairs": sum(
                block.last_attention_stats["attention_pairs"] for block in blocks
            ),
            "dense_reference_pairs": sum(
                block.last_attention_stats["dense_reference_pairs"] for block in blocks
            ),
            "context_tokens": sum(block.last_attention_stats["context_tokens"] for block in blocks),
            "metric_neighbor_distance_comparisons": sum(
                block.last_attention_stats.get("metric_neighbor_distance_comparisons", 0)
                for block in blocks
            ),
            "neighbors_per_query": max(
                (block.last_attention_stats.get("neighbors_per_query", 0) for block in blocks),
                default=0,
            ),
        }
        self.last_attention_stats["stage3"] = _aggregate_stage_stats(list(self.stage3))
        self.last_attention_stats["stage4"] = _aggregate_stage_stats(list(self.stage4))
        self.last_attention_debug = {
            "stage3": [block.last_attention_debug for block in self.stage3],
            "stage4": [block.last_attention_debug for block in self.stage4],
        }
        return feature_map
