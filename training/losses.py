"""Mask-aware safety and local-waypoint training objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def hazard_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    validity: torch.Tensor,
    positive_weight: float = 3.0,
    overlap_weight: float = 0.5,
    overlap: str = "tversky",
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> torch.Tensor:
    validity = validity.to(logits.dtype)
    target = target.to(logits.dtype)
    pixel_loss = functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none",
        pos_weight=torch.as_tensor(positive_weight, device=logits.device),
    )
    denominator = validity.sum().clamp_min(1)
    weighted_bce = (pixel_loss * validity).sum() / denominator
    probability = logits.sigmoid() * validity
    truth = target * validity
    true_positive = (probability * truth).sum()
    false_positive = (probability * (1 - truth) * validity).sum()
    false_negative = ((1 - probability) * truth).sum()
    smooth = 1e-6
    if overlap == "dice":
        score = (2 * true_positive + smooth) / (
            probability.sum() + truth.sum() + smooth
        )
    elif overlap == "tversky":
        score = (true_positive + smooth) / (
            true_positive + tversky_alpha * false_positive +
            tversky_beta * false_negative + smooth
        )
    else:
        raise ValueError("overlap must be dice or tversky")
    return weighted_bce + overlap_weight * (1 - score)


def waypoint_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    yaw_weight: float = 1.0,
) -> torch.Tensor:
    translation = functional.smooth_l1_loss(prediction[:, :3], target[:, :3])
    predicted_yaw = functional.normalize(prediction[:, 3:5], dim=1, eps=1e-6)
    target_yaw = functional.normalize(target[:, 3:5], dim=1, eps=1e-6)
    yaw = functional.smooth_l1_loss(predicted_yaw, target_yaw)
    return translation + yaw_weight * yaw


def multitask_loss(outputs: dict, batch: dict, config: dict) -> dict[str, torch.Tensor]:
    hazard = hazard_loss(
        outputs["hazard_logits"], batch["hazard_target"], batch["hazard_validity"],
        float(config.get("positive_class_weight", 3.0)),
        float(config.get("overlap_weight", 0.5)),
        str(config.get("overlap", "tversky")),
        float(config.get("tversky_alpha", 0.3)),
        float(config.get("tversky_beta", 0.7)),
    )
    waypoint = waypoint_loss(
        outputs["waypoint"], batch["waypoint_target"],
        float(config.get("yaw_weight", 1.0)),
    )
    total = float(config.get("hazard_weight", 1.0)) * hazard + \
        float(config.get("waypoint_weight", 1.0)) * waypoint
    return {"total": total, "hazard": hazard, "waypoint": waypoint}
