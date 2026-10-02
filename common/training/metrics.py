"""Streaming pixel metrics and paired episode bootstrap intervals."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class BinaryCounts:
    prefix: str = "hazard"
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def update(self, logits, target, validity, threshold=0.5):
        predicted = logits.sigmoid() >= threshold
        truth = target >= 0.5
        valid = validity >= 0.5
        self.true_positive += int((predicted & truth & valid).sum())
        self.true_negative += int((~predicted & ~truth & valid).sum())
        self.false_positive += int((predicted & ~truth & valid).sum())
        self.false_negative += int((~predicted & truth & valid).sum())

    def result(self):
        tp, tn, fp, fn = (
            self.true_positive,
            self.true_negative,
            self.false_positive,
            self.false_negative,
        )
        ratio = lambda a, b: a / b if b else 0.0
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        p = self.prefix
        return {
            f"{p}_true_positive": tp,
            f"{p}_true_negative": tn,
            f"{p}_false_positive": fp,
            f"{p}_false_negative": fn,
            f"{p}_accuracy": ratio(tp + tn, tp + tn + fp + fn),
            f"{p}_precision": precision,
            f"{p}_recall": recall,
            f"{p}_fnr": ratio(fn, tp + fn),
            f"{p}_fpr": ratio(fp, fp + tn),
            f"{p}_f1": ratio(2 * precision * recall, precision + recall),
            f"{p}_iou": ratio(tp, tp + fp + fn),
        }


HazardCounts = BinaryCounts


class POIMetrics:
    def __init__(self, names):
        self.counts = [BinaryCounts(f"poi_{name}") for name in names]

    def update(self, logits, target, validity, threshold=0.5):
        for i, count in enumerate(self.counts):
            count.update(
                logits[:, i : i + 1], target[:, i : i + 1], validity[:, i : i + 1], threshold
            )

    def result(self):
        result = {}
        f1 = []
        for count in self.counts:
            current = count.result()
            result.update(current)
            f1.append(current[f"{count.prefix}_f1"])
        result["poi_macro_f1"] = float(np.mean(f1)) if f1 else 0.0
        return result


class CategoricalMetrics:
    def __init__(self, names):
        self.names = tuple(names)
        self.matrix = np.zeros((len(names), len(names)), dtype=np.int64)

    def update(self, logits, target, validity=None):
        prediction = logits.argmax(dim=1).detach().cpu().numpy().reshape(-1)
        truth = target.detach().cpu().numpy().reshape(-1)
        valid = (
            np.ones_like(truth, dtype=bool)
            if validity is None
            else validity.detach().cpu().numpy().reshape(-1) >= 0.5
        )
        keep = valid & (truth >= 0) & (truth < len(self.names)) & (prediction < len(self.names))
        np.add.at(self.matrix, (truth[keep], prediction[keep]), 1)

    def result(self, prefix):
        result = {}
        ious = []
        recalls = []
        for i, name in enumerate(self.names):
            tp = int(self.matrix[i, i])
            fn = int(self.matrix[i].sum() - tp)
            fp = int(self.matrix[:, i].sum() - tp)
            union = tp + fn + fp
            iou = tp / union if union else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            result[f"{prefix}_{name.lower()}_iou"] = iou
            result[f"{prefix}_{name.lower()}_recall"] = recall
            ious.append(iou)
            recalls.append(recall)
        result[f"{prefix}_mean_iou"] = float(np.mean(ious)) if ious else 0.0
        result[f"{prefix}_mean_recall"] = float(np.mean(recalls)) if recalls else 0.0
        result[f"{prefix}_confusion_matrix"] = self.matrix.tolist()
        return result


def paired_bootstrap_delta(
    left_by_episode, right_by_episode, metric, seed=24051991, repetitions=2000
):
    keys = sorted(set(left_by_episode) & set(right_by_episode))
    if not keys:
        return {"mean_delta": None, "ci95": [None, None], "episodes": 0}
    values = np.asarray([left_by_episode[k][metric] - right_by_episode[k][metric] for k in keys])
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)]
    )
    return {
        "mean_delta": float(values.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "episodes": len(keys),
    }
