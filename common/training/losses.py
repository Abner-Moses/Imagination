"""Losses for spatial perception and scene-level context outputs."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def segmentation_loss(
    logits,
    target,
    validity,
    positive_weight=1.0,
    overlap_weight: float = 0.5,
    false_negative_weight: float = 0.7,
):
    """Masked weighted BCE plus a soft Tversky overlap term."""
    target = target.to(dtype=logits.dtype)
    validity = validity.to(dtype=logits.dtype)
    positive_weight = torch.as_tensor(positive_weight, dtype=logits.dtype, device=logits.device)
    if positive_weight.ndim == 1:
        positive_weight = positive_weight.view(1, -1, 1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = bce * (1.0 + (positive_weight - 1.0) * target)
    bce = (bce * validity).sum() / validity.sum().clamp_min(1.0)

    probability = logits.sigmoid()
    axes = (0, 2, 3)
    true_positive = (probability * target * validity).sum(axes)
    false_positive = (probability * (1.0 - target) * validity).sum(axes)
    false_negative = ((1.0 - probability) * target * validity).sum(axes)
    score = (true_positive + 1e-6) / (
        true_positive
        + (1.0 - false_negative_weight) * false_positive
        + false_negative_weight * false_negative
        + 1e-6
    )
    return bce + overlap_weight * (1.0 - score.mean())


def _semantic_loss(logits, target, validity):
    pixel_loss = F.cross_entropy(logits, target.long(), reduction="none")
    mask = validity.to(dtype=pixel_loss.dtype)
    return (pixel_loss * mask).sum() / mask.sum().clamp_min(1.0)


def multitask_loss(outputs, batch, config):
    hazard = segmentation_loss(
        outputs["hazard_logits"],
        batch["hazard_target"],
        batch["hazard_validity"],
        config.get("hazard_positive_weight", 3.0),
        config.get("overlap_weight", 0.5),
        config.get("hazard_false_negative_weight", 0.7),
    )
    landing = segmentation_loss(
        outputs["landing_logits"],
        batch["landing_target"],
        batch["landing_validity"],
        config.get("landing_positive_weight", 2.0),
        config.get("overlap_weight", 0.5),
        0.5,
    )
    poi = segmentation_loss(
        outputs["poi"]["class_logits"],
        batch["poi_target"],
        batch["poi_validity"],
        config.get("poi_positive_weight", 8.0),
        config.get("overlap_weight", 0.5),
        0.5,
    )
    semantic = _semantic_loss(
        outputs["semantic_logits"], batch["semantic_target"], batch["semantic_validity"]
    )
    scene = F.cross_entropy(outputs["scene_risk_logits"], batch["scene_target"].long())

    candidate_risk = outputs["candidate"]["risk_logits"]
    candidate_landing = outputs["candidate"]["landing_safe_logits"]
    candidate_validity = batch.get("candidate_target_validity", outputs["candidate"]["validity"])
    if "candidate_risk_target" in batch:
        mask = candidate_validity.to(candidate_risk.dtype)
        risk_loss = F.cross_entropy(
            candidate_risk.flatten(0, 1),
            batch["candidate_risk_target"].long().flatten(),
            reduction="none",
        ).view_as(mask)
        candidate_risk_loss = (risk_loss * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        candidate_risk_loss = candidate_risk.sum() * 0.0
    if "candidate_landing_target" in batch:
        mask = candidate_validity.to(candidate_landing.dtype)
        target = batch["candidate_landing_target"].to(candidate_landing.dtype)
        raw = F.binary_cross_entropy_with_logits(candidate_landing, target, reduction="none")
        candidate_landing_loss = (raw * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        candidate_landing_loss = candidate_landing.sum() * 0.0
    candidate = candidate_risk_loss + candidate_landing_loss
    topology = outputs.get("aux", {}).get("topology_loss", hazard.sum() * 0.0)

    components = {
        "hazard": hazard,
        "landing": landing,
        "poi": poi,
        "semantic": semantic,
        "scene": scene,
        "candidate": candidate,
        "topology": topology,
    }
    weights = {
        "hazard": float(config.get("hazard_weight", 1.0)),
        "landing": float(config.get("landing_weight", 1.0)),
        "poi": float(config.get("poi_weight", 1.0)),
        "semantic": float(config.get("semantic_weight", 0.5)),
        "scene": float(config.get("scene_weight", 0.25)),
        "candidate": float(config.get("candidate_weight", 0.25)),
        "topology": float(config.get("topology_loss_weight", 0.0)),
    }
    components["total"] = sum(components[name] * weights[name] for name in weights)
    return components
