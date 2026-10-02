"""Calibration-window orchestration and CLI for IMF-LBA."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from common.runtime import file_sha256, git_commit, load_config, stable_hash
from data.adapter import load_manifest, resolve_path

from .contracts import Decision, OperatorResult
from .decisions import evaluate_operator, mask_stability
from .diagnostics import load_audit_frames, run_diagnostic
from .registry import LBA_SCHEMA, operator_registry
from .reporting import write_report


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "common" / "configs" / "imf_audit.yaml"
DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "validation_episode.jsonl"


def _intrinsics(path: Path) -> dict[str, float]:
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise ValueError(f"Cannot open calibration: {path}")
    camera = storage.getNode("camera")
    result = {
        name: float(camera.getNode(name).real())
        for name in ("width", "height", "fx", "fy", "cx", "cy")
    }
    storage.release()
    if min(result.values()) <= 0:
        raise ValueError(f"Invalid calibration: {path}")
    return result


def _records(manifest: Path, split: str, count: int) -> list[dict]:
    records = load_manifest(manifest)
    if any(row.get("split") == "test" for row in records) and split == "test":
        raise PermissionError(
            "IMF-LBA architecture-selection mode refuses the final held-out test split"
        )
    selected = [row for row in records if row.get("split") == split]
    if not selected:
        raise ValueError(f"No records for allowed calibration split '{split}'")
    if any(row.get("split") == "test" for row in selected):
        raise PermissionError(
            "Final held-out test records are forbidden in IMF-LBA design-time audits"
        )
    selected.sort(key=lambda row: (row["episode_id"], int(row["frame_index"])))
    return selected[:count]


def _result_counts(results: dict[str, OperatorResult]) -> dict[str, int]:
    counts = Counter(result.decision.value for result in results.values())
    return {decision.value: counts.get(decision.value, 0) for decision in Decision}


def _print_table(results: dict[str, OperatorResult]) -> None:
    print(f"{'Operator':<30} {'R':>8}  {'CI':<20} {'Decision':<22} Primary reason")
    print("-" * 110)
    for name, result in results.items():
        ratio = "N/A" if result.ratio is None else f"{result.ratio:.3f}"
        interval = (
            "N/A"
            if result.confidence_interval is None
            else (f"[{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
        )
        reason = result.decision_reasons[0] if result.decision_reasons else "—"
        print(f"{name:<30} {ratio:>8}  {interval:<20} {result.decision.value:<22} {reason}")


def run_audit(
    *,
    manifest: str | Path = DEFAULT_MANIFEST,
    split: str = "val",
    frames: int | None = None,
    windows: list[int] | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    output_root: str | Path | None = None,
    confidence_level: float | None = None,
    seed: int | None = None,
) -> Path:
    """Run a design-time audit; never mutates models, caches, data, or training."""
    started = perf_counter()
    manifest = Path(manifest).resolve()
    config_path = Path(config_path).resolve()
    config = load_config(config_path).get("imf_audit", {})
    confidence = float(confidence_level or config.get("confidence_level", 0.95))
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level must be between 0.5 and 1")
    bootstrap_seed = int(seed if seed is not None else config.get("bootstrap_seed", 24051991))
    requested = windows or (
        [int(frames)]
        if frames is not None
        else list(config.get("progressive_windows", (5, 10, 25, 50)))
    )
    requested = sorted(set(int(value) for value in requested))
    if not requested or requested[0] < 1:
        raise ValueError("Calibration windows must contain positive frame counts")
    records = _records(manifest, split, max(requested))
    if len(records) < max(requested):
        requested = [value for value in requested if value <= len(records)] + [len(records)]
        requested = sorted(set(requested))
    first_calibration = resolve_path(manifest.parent, records[0]["calibration_path"])
    intrinsics = _intrinsics(first_calibration)
    registry = operator_registry(config)
    histories: dict[int, dict[str, OperatorResult]] = {}
    ids_by_window = {}
    runtime_by_window = {}
    for window in requested:
        window_started = perf_counter()
        selected = records[:window]
        ids_by_window[str(window)] = [row["sample_id"] for row in selected]
        loaded = load_audit_frames(selected, manifest)
        results: dict[str, OperatorResult] = {}
        settings = {"intrinsics": intrinsics}
        for index, (name, spec) in enumerate(registry.items()):
            evidence = run_diagnostic(
                spec,
                loaded,
                {
                    **settings,
                    "operator_config": config.get("operators", {}).get(name, {}),
                },
            )
            results[name] = evaluate_operator(
                spec,
                evidence,
                window_size=window,
                confidence_level=confidence,
                bootstrap_samples=int(config.get("bootstrap_samples", 1000)),
                seed=bootstrap_seed + index + window * 1009,
                dependency_results=results,
            )
        histories[window] = results
        runtime_by_window[str(window)] = perf_counter() - window_started
    stability = []
    ordered = list(histories)
    for left_window, right_window in zip(ordered, ordered[1:]):
        left = {name: result.decision for name, result in histories[left_window].items()}
        right = {name: result.decision for name, result in histories[right_window].items()}
        stable = mask_stability(left, right)
        resolved = sum(
            value in {Decision.ANALYTICAL, Decision.LEARNING_CANDIDATE} for value in right.values()
        )
        stability.append(
            {
                "from_window": left_window,
                "to_window": right_window,
                "stability": stable,
                "resolved_fraction": resolved / max(1, len(right)),
            }
        )
    config_hash = stable_hash(config)
    manifest_hash = file_sha256(manifest)
    run_hash = stable_hash(
        {
            "schema": LBA_SCHEMA,
            "config": config_hash,
            "manifest": manifest_hash,
            "split": split,
            "windows": requested,
            "ids": ids_by_window,
        }
    )[:10]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(output_root or config.get("output_root", "artifacts/results/imf_audit"))
    if not root.is_absolute():
        root = ROOT / root
    output = root / f"{stamp}_{run_hash}"
    suffix = 2
    while output.exists():
        output = root / f"{stamp}_{run_hash}_{suffix}"
        suffix += 1
    metadata: dict[str, Any] = {
        "schema": LBA_SCHEMA,
        "design_time_only": True,
        "automatic_architecture_mutation": False,
        "automatic_training": False,
        "final_test_accessed": False,
        "dataset_identifier": records[0].get("source", "unknown"),
        "split": split,
        "episode_ids": sorted({row["episode_id"] for row in records}),
        "calibration_frame_ids": ids_by_window,
        "manifest": str(manifest),
        "manifest_hash": manifest_hash,
        "calibration_hash": file_sha256(first_calibration),
        "config": str(config_path),
        "configuration_hash": config_hash,
        "git_commit": git_commit(ROOT),
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": confidence,
        "windows": requested,
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "operator_versions": {name: spec.version for name, spec in registry.items()},
        "window_runtime_seconds": runtime_by_window,
        "total_runtime_seconds": perf_counter() - started,
        "deployment_cost_scope": "NOT_MEASURED_FROM_CACHED_AUDIT",
    }
    write_report(output, metadata, registry, histories, stability)
    final = histories[max(histories)]
    _print_table(final)
    print("\nDecision counts:", json.dumps(_result_counts(final), sort_keys=True))
    print(f"Audit artifacts: {output}")
    return output


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python run.py audit-imf", description="IMF Learning-Boundary Auditor"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", default="val", choices=("train", "val", "calibration", "test"))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--frames", type=int)
    group.add_argument("--windows", type=int, nargs="+")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args(argv)
    run_audit(
        manifest=arguments.manifest,
        split=arguments.split,
        frames=arguments.frames,
        windows=arguments.windows,
        config_path=arguments.config,
        output_root=arguments.output,
        confidence_level=arguments.confidence,
        seed=arguments.seed,
    )
    return 0
