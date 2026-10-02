"""Shared lightweight spatial decoder and multimodal perception heads."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from common.registry import (
    CANDIDATE_FEATURE_NAMES,
    MAX_CANDIDATES,
    RELATIONAL_NAMES,
    SCENE_CLASSES,
    SEMANTIC_CLASSES,
    STATE_DIM,
)


class MaskedEncoder(nn.Module):
    """Encode values together with masks so missing data is not mistaken for zero."""

    def __init__(self, dimensions: int, embedding: int):
        super().__init__()
        self.dimensions = dimensions
        self.network = nn.Sequential(
            nn.Linear(dimensions * 2, embedding),
            nn.GELU(),
            nn.Linear(embedding, embedding),
            nn.GELU(),
        )

    def forward(self, values, validity, batch_size: int, device):
        if values is None:
            values = torch.zeros(batch_size, self.dimensions, device=device)
        if validity is None:
            validity = torch.zeros_like(values)
        validity = validity.to(dtype=values.dtype)
        if values.shape != (batch_size, self.dimensions):
            raise ValueError(f"Expected Bx{self.dimensions} values, got {tuple(values.shape)}")
        if validity.shape != values.shape:
            raise ValueError("Value and validity shapes must match")
        return self.network(torch.cat((values * validity, validity), dim=1))


class GlobalFiLM(nn.Module):
    """Condition spatial perception on separately encoded state/map relations."""

    def __init__(
        self,
        channels: int,
        state_dim: int = STATE_DIM,
        relation_dim: int = len(RELATIONAL_NAMES),
        embedding: int = 64,
        use_state: bool = True,
    ):
        super().__init__()
        self.use_state = use_state
        self.state_encoder = MaskedEncoder(state_dim, embedding)
        self.relation_encoder = MaskedEncoder(relation_dim, embedding)
        self.affine = nn.Linear(embedding * 2, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, visual, state, state_validity, relation, relation_validity):
        batch = visual.shape[0]
        state_embedding = self.state_encoder(
            state if self.use_state else None,
            state_validity if self.use_state else None,
            batch,
            visual.device,
        )
        relation_embedding = self.relation_encoder(
            relation, relation_validity, batch, visual.device
        )
        scale, shift = self.affine(torch.cat((state_embedding, relation_embedding), dim=1)).chunk(
            2, dim=1
        )
        conditioned = visual * (1.0 + torch.tanh(scale)[:, :, None, None]) + shift[:, :, None, None]
        return conditioned, state_embedding, relation_embedding


class SharedSpatialDecoder(nn.Module):
    """One bilinear upsample and depthwise refinement shared by spatial tasks."""

    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(
                output_channels, output_channels, 3, padding=1, groups=output_channels, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
        )

    def forward(self, value):
        value = self.project(value)
        value = F.interpolate(value, size=(32, 32), mode="bilinear", align_corners=False)
        return self.refine(value)


class PerceptionHeads(nn.Module):
    """Hazard, landing, semantic, POI, scene, and map-candidate outputs."""

    def __init__(
        self,
        visual_channels: int,
        decoder_channels: int,
        state_dim: int = STATE_DIM,
        poi_classes: int = 7,
        semantic_classes: int = len(SEMANTIC_CLASSES),
        use_state: bool = True,
    ):
        super().__init__()
        self.condition = GlobalFiLM(visual_channels, state_dim, use_state=use_state)
        self.decoder = SharedSpatialDecoder(visual_channels, decoder_channels)
        self.hazard = nn.Conv2d(decoder_channels, 1, 1)
        self.landing = nn.Conv2d(decoder_channels, 1, 1)
        self.semantic = nn.Conv2d(decoder_channels, semantic_classes, 1)
        self.poi = nn.Conv2d(decoder_channels, poi_classes, 1)

        candidate_width = 64
        self.candidate_encoder = nn.Sequential(
            nn.Linear(len(CANDIDATE_FEATURE_NAMES) * 2, candidate_width),
            nn.GELU(),
            nn.Linear(candidate_width, candidate_width),
            nn.GELU(),
        )
        # Candidate verdicts need the same global state/map facts as scene
        # verdicts (notably drift, vehicle state, and track relations).
        # Encode those separately, then condition each candidate compactly.
        self.candidate_condition = nn.Sequential(
            nn.Linear(candidate_width + 128, candidate_width),
            nn.GELU(),
        )
        self.candidate_risk = nn.Linear(candidate_width, len(SCENE_CLASSES))
        self.candidate_landing = nn.Linear(candidate_width, 1)
        self.scene = nn.Sequential(
            nn.Linear(visual_channels + candidate_width * 3, 128),
            nn.GELU(),
            nn.Linear(128, len(SCENE_CLASSES)),
        )

    @staticmethod
    def mask_candidate_input(candidates, feature_validity):
        if candidates.shape != feature_validity.shape:
            raise ValueError("Candidate values and per-feature validity must match")
        feature_validity = feature_validity.to(candidates.dtype)
        return torch.cat((candidates * feature_validity, feature_validity), dim=-1)

    def forward(
        self,
        visual,
        state_values=None,
        state_validity=None,
        relational=None,
        relational_validity=None,
        candidates=None,
        candidate_validity=None,
        candidate_feature_validity=None,
    ):
        conditioned, state_embedding, relation_embedding = self.condition(
            visual, state_values, state_validity, relational, relational_validity
        )
        spatial = self.decoder(conditioned)
        batch = conditioned.shape[0]
        if candidates is None:
            candidates = conditioned.new_zeros(batch, MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES))
        if candidate_validity is None:
            candidate_validity = conditioned.new_zeros(batch, MAX_CANDIDATES)
        if candidate_feature_validity is None:
            candidate_feature_validity = candidate_validity[..., None].expand_as(candidates)
        if candidate_feature_validity.shape != candidates.shape:
            raise ValueError("candidate_feature_validity must match candidate feature shape")
        candidate_input = self.mask_candidate_input(candidates, candidate_feature_validity)
        encoded = self.candidate_encoder(candidate_input)
        candidate_global = torch.cat((state_embedding, relation_embedding), dim=1)
        candidate_global = candidate_global[:, None, :].expand(-1, encoded.shape[1], -1)
        encoded = self.candidate_condition(torch.cat((encoded, candidate_global), dim=-1))
        mask = candidate_validity.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        scene_features = torch.cat(
            (conditioned.mean(dim=(2, 3)), state_embedding, relation_embedding, pooled), dim=1
        )
        return {
            "hazard_logits": self.hazard(spatial),
            "landing_logits": self.landing(spatial),
            "semantic_logits": self.semantic(spatial),
            "poi": {"class_logits": self.poi(spatial)},
            "scene_risk_logits": self.scene(scene_features),
            "candidate": {
                "risk_logits": self.candidate_risk(encoded),
                "landing_safe_logits": self.candidate_landing(encoded).squeeze(-1),
                "validity": candidate_validity,
            },
            "aux": {
                "candidate_validity": candidate_validity,
                "candidate_feature_validity": candidate_feature_validity,
            },
        }


TaskHeads = PerceptionHeads
