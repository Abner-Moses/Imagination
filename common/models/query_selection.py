"""Deterministic analytical selection of spatial queries for optional attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def active_query_indices(
    importance: torch.Tensor,
    mandatory: torch.Tensor | None,
    fraction: float,
    *,
    coverage_bins: tuple[int, int] = (4, 4),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a deterministic top set plus safety and coverage queries.

    Returns padded ``indices[B,Q]`` and a validity mask. Ties resolve by
    row-major token index. Mandatory safety queries and one maximum-score query
    per coverage bin may raise Q above the nominal fraction budget.
    """
    if importance.ndim != 3 or not importance.is_floating_point():
        raise ValueError("importance must be a floating BxHxW tensor")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("active query fraction must be in (0,1]")
    batch, height, width = importance.shape
    if mandatory is None:
        mandatory = torch.zeros_like(importance, dtype=torch.bool)
    if mandatory.shape != importance.shape:
        raise ValueError("mandatory query mask must match importance map")
    bins_y, bins_x = coverage_bins
    if bins_y < 1 or bins_x < 1 or bins_y > height or bins_x > width:
        raise ValueError("coverage bins must fit inside the spatial grid")
    flat_importance = torch.nan_to_num(
        importance.reshape(batch, -1), nan=0.0, posinf=1e6, neginf=0.0
    )
    flat_mandatory = mandatory.reshape(batch, -1).bool()
    budget = max(1, int(math.ceil(height * width * fraction)))
    selections = []
    for batch_index in range(batch):
        selected = set(
            torch.nonzero(flat_mandatory[batch_index], as_tuple=False).flatten().tolist()
        )
        for bin_y in range(bins_y):
            y0, y1 = bin_y * height // bins_y, (bin_y + 1) * height // bins_y
            for bin_x in range(bins_x):
                x0, x1 = bin_x * width // bins_x, (bin_x + 1) * width // bins_x
                ids = [y * width + x for y in range(y0, y1) for x in range(x0, x1)]
                local_scores = flat_importance[batch_index, ids]
                selected.add(ids[int(torch.argmax(local_scores))])
        target = min(height * width, max(budget, len(selected)))
        order = torch.argsort(flat_importance[batch_index], descending=True, stable=True).tolist()
        for token in order:
            if len(selected) >= target:
                break
            selected.add(int(token))
        selections.append(sorted(selected))
    max_count = max(map(len, selections))
    indices = torch.zeros((batch, max_count), device=importance.device, dtype=torch.long)
    validity = torch.zeros((batch, max_count), device=importance.device, dtype=importance.dtype)
    for batch_index, selected in enumerate(selections):
        indices[batch_index, : len(selected)] = torch.tensor(selected, device=importance.device)
        validity[batch_index, : len(selected)] = 1.0
        if len(selected) < max_count:
            indices[batch_index, len(selected) :] = selected[0]
    return indices, validity


def downsample_importance(importance: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if importance.ndim != 3:
        raise ValueError("importance must have shape BxHxW")
    return F.adaptive_max_pool2d(importance[:, None], size)[:, 0]


def downsample_mandatory(mandatory: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if mandatory.ndim != 3:
        raise ValueError("mandatory must have shape BxHxW")
    return F.adaptive_max_pool2d(mandatory.float()[:, None], size)[:, 0] > 0
