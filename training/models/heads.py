"""Task heads shared by both experimental models."""

from __future__ import annotations

import torch
from torch import nn


class StateEncoder(nn.Module):
    def __init__(self, state_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.network = nn.Sequential(
            nn.Linear(2 * state_dim, embedding_dim), nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim), nn.GELU(),
        ) if state_dim else None

    def forward(
        self,
        values: torch.Tensor | None,
        validity: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.network is None:
            return torch.zeros(batch_size, self.embedding_dim, device=device)
        if values is None or validity is None:
            values = torch.zeros(batch_size, self.state_dim, device=device)
            validity = torch.zeros_like(values)
        validity = validity.to(values.dtype)
        return self.network(torch.cat((values * validity, validity), dim=1))


class HazardHead(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        middle = min(384, max(16, input_channels // 2))
        low = min(192, max(8, middle // 2))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_channels, middle, 4, stride=2, padding=1),
            nn.BatchNorm2d(middle), nn.GELU(),
            nn.ConvTranspose2d(middle, low, 4, stride=2, padding=1),
            nn.BatchNorm2d(low), nn.GELU(),
            nn.Conv2d(low, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


class NavigationHead(nn.Module):
    def __init__(self, visual_channels: int, state_embedding: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.network = nn.Sequential(
            nn.Linear(visual_channels + state_embedding, 256), nn.GELU(),
            nn.Linear(256, 5),
        )

    def forward(self, visual: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(visual).flatten(1)
        prediction = self.network(torch.cat((pooled, state), dim=1))
        yaw = torch.nn.functional.normalize(prediction[:, 3:5], dim=1, eps=1e-6)
        return torch.cat((prediction[:, :3], yaw), dim=1)


class TaskHeads(nn.Module):
    def __init__(self, visual_channels: int, state_dim: int, state_embedding: int = 64):
        super().__init__()
        self.state = StateEncoder(state_dim, state_embedding)
        self.hazard = HazardHead(visual_channels)
        self.navigation = NavigationHead(visual_channels, state_embedding)

    def forward(
        self,
        visual: torch.Tensor,
        state_values: torch.Tensor | None,
        state_validity: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        state = self.state(
            state_values, state_validity, visual.shape[0], visual.device
        )
        return {
            "hazard_logits": self.hazard(visual),
            "waypoint": self.navigation(visual, state),
            "aux": {"state_embedding": state},
        }
