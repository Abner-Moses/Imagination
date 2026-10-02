"""Validity-aware IMF input with compact map context and HTransformer stages."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from common.models.heads import PerceptionHeads
from common.models.htransformer import HTransformerBackbone
from common.models.profiles import profile_config
from common.feature_spec import FUSION_FAMILIES
from common.models.metric_attention import (
    downsample_metric_positions,
    positions_from_analytical,
    positions_from_candidate_projection,
    topology_smoothness_loss,
)
from common.registry import (
    ANALYTICAL_CHANNELS,
    CANDIDATE_FEATURE_NAMES,
    POI_CLASSES,
    SEMANTIC_CLASSES,
    STATE_DIM,
)


FEATURE_FAMILIES = {
    "appearance": ("Y", "Cb", "Cr"),
    "gradients": ("Gx", "Gy", "GradientMagnitude"),
    "hog": tuple(f"HOG_{index}" for index in range(9)),
    "harris": ("HarrisResponse",),
    "canny": ("CannyEdge",),
    "contours": ("ContourMap",),
    "chroma": ("ChromaGradientCb", "ChromaGradientCr"),
    "flow": ("OpticalFlowU", "OpticalFlowV"),
    "geometry": (
        "Depth",
        "DepthGradientX",
        "DepthGradientY",
        "Slope",
        "Roughness",
        "GeometryConfidence",
    ),
}


class FeatureFamilyAdapter(nn.Module):
    """Fuse one homogeneous feature family using a cheap local depthwise block."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels * 2, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
            nn.Conv2d(
                output_channels, output_channels, 3, padding=1, groups=output_channels, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        validity = validity.to(dtype=values.dtype)
        return self.layers(torch.cat((values * validity, validity), dim=1))


class IMFHTransformer(nn.Module):
    family = "imf_htransformer"

    def __init__(
        self,
        profile: str = "research",
        disabled_families=(),
        disabled_fusion_families=(),
        use_state: bool = True,
        fusion: str = "family",
        overrides: dict | None = None,
        poi_classes: int = len(POI_CLASSES),
    ):
        super().__init__()
        widths = profile_config(profile, overrides)
        self.profile = profile
        self.fusion = fusion
        self.attention = dict((overrides or {}).get("attention", {}))
        dissimilarity = self.attention.get("imf_dissimilarity", {})
        if not isinstance(dissimilarity, dict):
            raise ValueError("attention.imf_dissimilarity must be a mapping")
        self.imf_dissimilarity_weight = (
            float(dissimilarity.get("weight", 0.0)) if dissimilarity.get("enabled", False) else 0.0
        )
        self.imf_dissimilarity_scales = tuple(dissimilarity.get("scales", (0.25, 0.25, 0.25, 0.25)))
        self.imf_dissimilarity_weights = tuple(dissimilarity.get("weights", (1.0, 0.5, 0.5, 0.5)))
        self.stage3_use_map_context = bool(self.attention.get("stage3_use_map_context", True))
        self.stage4_use_map_context = bool(self.attention.get("stage4_use_map_context", True))
        self.topology_loss_weight = float((overrides or {}).get("topology_loss_weight", 0.0))
        if self.topology_loss_weight < 0:
            raise ValueError("topology_loss_weight must be nonnegative")
        if fusion not in {"family", "flat"}:
            raise ValueError("fusion must be 'family' or 'flat'")

        enabled_channels = torch.ones(1, len(ANALYTICAL_CHANNELS), 1, 1)
        for family in disabled_families:
            if family not in FEATURE_FAMILIES:
                raise ValueError(f"Unknown IMF feature family: {family}")
            for channel in FEATURE_FAMILIES[family]:
                channel_index = ANALYTICAL_CHANNELS.index(channel)
                enabled_channels[:, channel_index] = 0
        unknown_fusion = set(disabled_fusion_families) - set(FUSION_FAMILIES)
        if unknown_fusion:
            raise ValueError(f"Unknown IMF fusion families: {sorted(unknown_fusion)}")
        for family in disabled_fusion_families:
            enabled_channels[:, list(FUSION_FAMILIES[family])] = 0
        self.register_buffer("ablation_mask", enabled_channels)
        self.enabled_fusion_families = {
            name
            for name, indices in FUSION_FAMILIES.items()
            if bool(enabled_channels[:, list(indices)].any())
        }

        self.family_adapters = nn.ModuleDict()
        self.flat_adapter = None
        self.family_widths = {}
        if fusion == "flat":
            self.flat_adapter = nn.Sequential(
                nn.Conv2d(56, widths["embedding"], kernel_size=1, bias=False),
                nn.BatchNorm2d(widths["embedding"]),
                nn.GELU(),
            )
        else:
            family_widths = {
                "appearance": widths["fusion_appearance"],
                "motion": widths["fusion_motion"],
                "geometry": widths["fusion_geometry"],
            }
            self.family_widths = family_widths
            if sum(family_widths.values()) != widths["embedding"]:
                raise ValueError(
                    f"Family adapter widths {family_widths} must sum to embedding "
                    f"{widths['embedding']}"
                )
            self.family_adapters = nn.ModuleDict(
                {
                    name: FeatureFamilyAdapter(len(indices), family_widths[name])
                    for name, indices in FUSION_FAMILIES.items()
                    if name in self.enabled_fusion_families
                }
            )
        backbone_config = {
            "embedding": widths["embedding"],
            "stage3_channels": widths["stage3"],
            "stage4_channels": widths["stage4"],
            "stage3_heads": widths["heads3"],
            "stage4_heads": widths["heads4"],
            "stage3_depth": widths["depth3"],
            "stage4_depth": widths["depth4"],
            "stage3_heads": int(self.attention.get("stage3_heads", widths["heads3"])),
            "stage4_heads": int(self.attention.get("stage4_heads", widths["heads4"])),
            "stage3_qk_channels": int(
                self.attention.get(
                    "stage3_qk_dim", self.attention.get("stage3_qk_channels", widths["stage3"])
                )
            ),
            "stage4_qk_channels": int(
                self.attention.get(
                    "stage4_qk_dim", self.attention.get("stage4_qk_channels", widths["stage4"])
                )
            ),
            "stage3_mlp_ratio": float(
                self.attention.get("stage3_ffn_ratio", self.attention.get("stage3_mlp_ratio", 2.0))
            ),
            "stage4_mlp_ratio": float(
                self.attention.get("stage4_ffn_ratio", self.attention.get("stage4_mlp_ratio", 2.0))
            ),
            "mbconv_expansion": 2.0,
            "stochastic_depth": 0.05,
            "attention_mode": self.attention.get("mode", "dense"),
            "stage3_attention_mode": self.attention.get(
                "stage3_mode", self.attention.get("mode", "dense")
            ),
            "stage4_attention_mode": self.attention.get(
                "stage4_mode", self.attention.get("mode", "dense")
            ),
            "stage3_sparse_neighbors": int(
                self.attention.get("stage3_sparse_neighbors", self.attention.get("neighbors", 32))
            ),
            "stage4_sparse_neighbors": int(
                self.attention.get("stage4_sparse_neighbors", self.attention.get("neighbors", 32))
            ),
            "stage3_active_query_fraction": float(
                self.attention.get("stage3_active_query_fraction", 1.0)
            ),
            "stage4_active_query_fraction": float(
                self.attention.get("stage4_active_query_fraction", 1.0)
            ),
            "stage3_active_query_coverage_bins": self.attention.get(
                "stage3_active_query_coverage_bins", (4, 4)
            ),
            "stage4_active_query_coverage_bins": self.attention.get(
                "stage4_active_query_coverage_bins", (4, 4)
            ),
            "imf_dissimilarity_weight": self.imf_dissimilarity_weight,
            "imf_dissimilarity_scales": self.imf_dissimilarity_scales,
            "imf_dissimilarity_weights": self.imf_dissimilarity_weights,
            "stage3_sparse_selection": self.attention.get(
                "stage3_sparse_selection", self.attention.get("selection", "metric")
            ),
            "stage4_sparse_selection": self.attention.get(
                "stage4_sparse_selection", self.attention.get("selection", "metric")
            ),
            "stage3_use_map_context": bool(self.attention.get("stage3_use_map_context", True)),
            "stage4_use_map_context": bool(self.attention.get("stage4_use_map_context", True)),
            "profile_timing": bool(self.attention.get("profile_timing", False)),
            "position_bias_mode": self.attention.get("position_bias", "metric"),
            "metric_weight": self.attention.get("weight", 0.5),
            "metric_scale_m": self.attention.get("scale_m", 4.0),
            "sparse_neighbors": self.attention.get("neighbors", 32),
            "sparse_selection": self.attention.get("selection", "metric"),
            "qk_normalize": self.attention.get("qk_normalize", False),
            "capture_attention": self.attention.get("capture_diagnostics", False),
            "context_features": len(CANDIDATE_FEATURE_NAMES) * 2,
        }
        self.backbone = HTransformerBackbone(backbone_config, widths["embedding"])
        self.heads = PerceptionHeads(
            widths["stage4"],
            widths["decoder"],
            STATE_DIM,
            poi_classes,
            len(SEMANTIC_CLASSES),
            use_state,
        )

    @staticmethod
    def mask_aware_input(features: torch.Tensor, validity: torch.Tensor) -> torch.Tensor:
        expected_shape = (28, 32, 32)
        if features.shape != validity.shape or features.shape[1:] != expected_shape:
            raise ValueError("IMF tensors must both have shape Bx28x32x32")
        mask = validity.to(dtype=features.dtype)
        return torch.cat((features * mask, mask), dim=1)

    def _active_query_evidence(
        self,
        analytical,
        validity,
        candidates=None,
        candidate_validity=None,
        candidate_feature_validity=None,
        candidate_grid=None,
        candidate_projection_validity=None,
        candidate_visibility=None,
    ):
        """Create deterministic spatial-query importance from current IMF only."""
        device = analytical.device
        weights = self.attention.get("active_query_weights", {})
        gradient = analytical[:, 5] * validity[:, 5]
        edge = analytical[:, 16] * validity[:, 16]
        geometry_supported = validity[:, 27] > 0
        geometry_uncertainty = geometry_supported.to(analytical.dtype) * (
            1.0 - analytical[:, 27].clamp(0, 1)
        )
        flow_supported = (validity[:, 20] > 0) & (validity[:, 21] > 0)
        flow_x = analytical[:, 20]
        flow_y = analytical[:, 21]
        flow_discontinuity = F.pad((flow_x[:, :, 1:] - flow_x[:, :, :-1]).abs(), (0, 1, 0, 0))
        flow_discontinuity += F.pad((flow_y[:, 1:, :] - flow_y[:, :-1, :]).abs(), (0, 0, 0, 1))
        flow_discontinuity *= flow_supported.to(analytical.dtype)
        importance = (
            float(weights.get("structure", 1.0)) * gradient
            + float(weights.get("edge", 1.0)) * edge
            + float(weights.get("geometry_uncertainty", 0.5)) * geometry_uncertainty
            + float(weights.get("motion_discontinuity", 0.5)) * flow_discontinuity
        )
        mandatory = (
            (edge >= float(self.attention.get("active_edge_threshold", 0.8)))
            | (
                geometry_uncertainty
                >= float(self.attention.get("active_uncertainty_threshold", 0.8))
            )
            | (flow_discontinuity >= float(self.attention.get("active_motion_threshold", 0.8)))
        )
        if (
            candidates is not None
            and candidate_grid is not None
            and candidate_projection_validity is not None
            and candidate_visibility is not None
        ):
            batch, height, width = importance.shape
            columns = {name: index for index, name in enumerate(CANDIDATE_FEATURE_NAMES)}
            relevant = torch.zeros_like(candidate_visibility, dtype=torch.bool)
            relevant_feature_names = (
                "type_landing",
                "type_restricted_boundary",
                "type_poi",
                "type_high_risk",
            )
            for name in relevant_feature_names:
                index = columns[name]
                if candidate_feature_validity is None:
                    relevant |= candidates[..., index] > 0.5
                else:
                    relevant |= (candidates[..., index] > 0.5) & (
                        candidate_feature_validity[..., index] > 0
                    )
            if candidate_validity is not None:
                relevant &= candidate_validity > 0
            relevant &= (candidate_projection_validity > 0) & (candidate_visibility == 1)
            for batch_index in range(batch):
                for candidate_index in torch.where(relevant[batch_index])[0].tolist():
                    x = int(torch.round(candidate_grid[batch_index, candidate_index, 0]).item())
                    y = int(torch.round(candidate_grid[batch_index, candidate_index, 1]).item())
                    if 0 <= x < width and 0 <= y < height:
                        mandatory[batch_index, y, x] = True
                        importance[batch_index, y, x] = torch.maximum(
                            importance[batch_index, y, x], importance.new_tensor(1.0)
                        )
        return importance, mandatory

    def forward(
        self,
        analytical,
        validity,
        state_values=None,
        state_validity=None,
        relational=None,
        relational_validity=None,
        candidates=None,
        candidate_validity=None,
        candidate_feature_validity=None,
        calibration=None,
        depth_scale_m=None,
        metric_positions=None,
        metric_position_validity=None,
        candidate_grid=None,
        candidate_projected_depth_m=None,
        candidate_projection_validity=None,
        candidate_visibility=None,
    ):
        validity = validity * self.ablation_mask
        if self.fusion == "flat":
            model_input = self.mask_aware_input(analytical, validity)
            embedded = self.flat_adapter(model_input)
        else:
            family_features = []
            for name, indices in FUSION_FAMILIES.items():
                if name not in self.enabled_fusion_families:
                    family_features.append(
                        analytical.new_zeros(
                            analytical.shape[0],
                            self.family_widths[name],
                            analytical.shape[2],
                            analytical.shape[3],
                        )
                    )
                    continue
                index = torch.as_tensor(indices, device=analytical.device)
                family_features.append(
                    self.family_adapters[name](
                        analytical.index_select(1, index), validity.index_select(1, index)
                    )
                )
            embedded = torch.cat(family_features, dim=1)
        if metric_positions is None and calibration is not None and depth_scale_m is not None:
            metric_positions, metric_position_validity = positions_from_analytical(
                analytical, validity, calibration, depth_scale_m
            )
        context_positions = context_validity = context_tokens = None
        if (
            (self.stage3_use_map_context or self.stage4_use_map_context)
            and candidates is not None
            and candidate_grid is not None
            and candidate_projected_depth_m is not None
            and candidate_projection_validity is not None
            and candidate_visibility is not None
            and calibration is not None
        ):
            if candidate_feature_validity is None:
                candidate_feature_validity = (
                    candidate_validity[..., None].expand_as(candidates)
                    if candidate_validity is not None
                    else torch.ones_like(candidates)
                )
            candidate_feature_validity = candidate_feature_validity.to(candidates.dtype)
            context_tokens = torch.cat(
                (candidates * candidate_feature_validity, candidate_feature_validity), dim=-1
            )
            context_positions, context_validity = positions_from_candidate_projection(
                candidate_grid,
                candidate_projected_depth_m,
                candidate_projection_validity,
                candidate_visibility,
                calibration,
            )
            if candidate_validity is not None:
                context_validity = context_validity * candidate_validity.to(context_validity.dtype)
        query_importance = query_mandatory = None
        if (
            float(self.attention.get("stage3_active_query_fraction", 1.0)) < 1.0
            or float(self.attention.get("stage4_active_query_fraction", 1.0)) < 1.0
        ):
            query_importance, query_mandatory = self._active_query_evidence(
                analytical,
                validity,
                candidates,
                candidate_validity,
                candidate_feature_validity,
                candidate_grid,
                candidate_projection_validity,
                candidate_visibility,
            )
        imf_descriptors = imf_descriptor_validity = imf_reliability = None
        if self.imf_dissimilarity_weight > 0:
            descriptor_indices = torch.tensor((20, 21, 25, 26), device=analytical.device)
            imf_descriptors = analytical.index_select(1, descriptor_indices).permute(0, 2, 3, 1)
            imf_descriptor_validity = validity.index_select(1, descriptor_indices).permute(
                0, 2, 3, 1
            )
            geometry_reliability = analytical[:, 27].clamp(0, 1) * validity[:, 27]
            motion_reliability = ((validity[:, 20] > 0) & (validity[:, 21] > 0)).to(
                analytical.dtype
            )
            imf_reliability = torch.maximum(geometry_reliability, motion_reliability)
        features = self.backbone(
            embedded,
            metric_positions,
            metric_position_validity,
            context_tokens,
            context_positions,
            context_validity,
            query_importance,
            query_mandatory,
            imf_descriptors,
            imf_descriptor_validity,
            imf_reliability,
        )
        outputs = self.heads(
            features,
            state_values,
            state_validity,
            relational,
            relational_validity,
            candidates,
            candidate_validity,
            candidate_feature_validity,
        )
        outputs["aux"]["attention"] = dict(self.backbone.last_attention_stats)
        topology_loss = features.sum() * 0.0
        if self.topology_loss_weight > 0 and metric_positions is not None:
            positions_8, valid_8 = downsample_metric_positions(
                metric_positions, metric_position_validity, (8, 8)
            )
            flat_positions = positions_8.flatten(1, 2)
            squared = torch.cdist(flat_positions.float(), flat_positions.float()).square()
            affinity = torch.exp(-0.5 * squared / float(self.attention.get("scale_m", 4.0)) ** 2)
            topology_loss = topology_smoothness_loss(
                features.flatten(2).transpose(1, 2),
                affinity,
                valid_8.flatten(1),
            )
        outputs["aux"]["topology_loss"] = topology_loss
        return outputs

    @classmethod
    def from_config(cls, config: dict) -> "IMFHTransformer":
        model = config.get("model", {})
        return cls(
            profile=model.get("profile", "research"),
            disabled_families=model.get("disabled_families", []),
            disabled_fusion_families=model.get("disabled_fusion_families", []),
            use_state=model.get("state_conditioning", True),
            fusion=model.get("fusion", "family"),
            overrides=model,
            poi_classes=len(config.get("poi", {}).get("classes", POI_CLASSES)),
        )


AnalyticalModel = IMFHTransformer
