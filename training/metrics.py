"""Safety, navigation, resource, and configured non-inferiority metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class HazardCounts:
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def update(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        validity: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        predicted = logits.sigmoid() >= threshold
        truth = target >= 0.5
        valid = validity >= 0.5
        self.true_positive += int((predicted & truth & valid).sum())
        self.true_negative += int((~predicted & ~truth & valid).sum())
        self.false_positive += int((predicted & ~truth & valid).sum())
        self.false_negative += int((~predicted & truth & valid).sum())

    def result(self) -> dict[str, float | int]:
        tp, tn, fp, fn = (
            self.true_positive, self.true_negative,
            self.false_positive, self.false_negative,
        )
        ratio = lambda numerator, denominator: numerator / denominator if denominator else 0.0
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        return {
            "true_positive": tp, "true_negative": tn,
            "false_positive": fp, "false_negative": fn,
            "hazard_precision": precision,
            "hazard_recall": recall,
            "hazard_fnr": ratio(fn, tp + fn),
            "hazard_fpr": ratio(fp, fp + tn),
            "hazard_f1": ratio(2 * precision * recall, precision + recall),
            "hazard_iou": ratio(tp, tp + fp + fn),
        }


class NavigationMetrics:
    def __init__(self):
        self.translation_absolute = 0.0
        self.translation_squared = 0.0
        self.heading_absolute = 0.0
        self.samples = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        error = prediction[:, :3] - target[:, :3]
        distance = torch.linalg.vector_norm(error, dim=1)
        predicted_angle = torch.atan2(prediction[:, 3], prediction[:, 4])
        target_angle = torch.atan2(target[:, 3], target[:, 4])
        angle = torch.atan2(
            torch.sin(predicted_angle - target_angle),
            torch.cos(predicted_angle - target_angle),
        ).abs()
        self.translation_absolute += float(distance.sum())
        self.translation_squared += float((distance * distance).sum())
        self.heading_absolute += float(angle.sum())
        self.samples += prediction.shape[0]

    def result(self) -> dict[str, float]:
        count = max(self.samples, 1)
        return {
            "waypoint_translation_mae_m": self.translation_absolute / count,
            "waypoint_translation_rmse_m": math.sqrt(self.translation_squared / count),
            "waypoint_heading_mae_rad": self.heading_absolute / count,
        }


def non_inferiority(
    analytical: dict[str, float],
    baseline: dict[str, float],
    fnr_epsilon: float,
    navigation_delta: float,
) -> dict[str, float | bool]:
    delta_fnr = analytical["hazard_fnr"] - baseline["hazard_fnr"]
    delta_navigation = (
        analytical["waypoint_translation_mae_m"] -
        baseline["waypoint_translation_mae_m"]
    )
    return {
        "delta_fnr": delta_fnr,
        "configured_fnr_epsilon": fnr_epsilon,
        "fnr_criterion_met": delta_fnr <= fnr_epsilon,
        "delta_navigation_mae_m": delta_navigation,
        "configured_navigation_delta_m": navigation_delta,
        "navigation_criterion_met": delta_navigation <= navigation_delta,
        "configured_joint_criterion_met":
            delta_fnr <= fnr_epsilon and delta_navigation <= navigation_delta,
        "statistical_significance_tested": False,
    }
