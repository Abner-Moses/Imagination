"""Sequential, resumable execution of the frozen pre-LBA primary experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from common.artifacts import ARTIFACTS
from common.evaluation import evaluate
from common.models.registry import MODEL_SPECS, PRIMARY_TRAINING_ORDER
from common.registry import MODEL_ARCHITECTURE_VERSION
from common.runner import _effective_training_config, experiment_config
from common.runtime import file_sha256, save_json, stable_hash
from common.training.engine import train
from common.training.readiness import (
    ARTIFACT_DIR,
    MANIFEST_DIR,
    PRIMARY_SEEDS,
    ROOT,
    prepare_training,
)

PRIMARY_ORDER = PRIMARY_TRAINING_ORDER
DISPLAY_NAMES = {model_id: spec.display_name for model_id, spec in MODEL_SPECS.items()}


def _flatten_run(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    resources = summary.get("resources", {})
    fields = (
        "hazard_fnr",
        "hazard_recall",
        "hazard_precision",
        "hazard_iou",
        "landing_iou",
        "landing_candidate_detection_rate",
        "poi_macro_f1",
        "semantic_mean_iou",
        "scene_accuracy",
    )
    return {
        "model": summary["model"],
        "display_name": summary["display_name"],
        "seed": summary["seed"],
        "training_plan_hash": summary["training_plan_hash"],
        **{name: metrics.get(name) for name in fields},
        "parameters": resources.get("parameters"),
        "latency_p50_ms": resources.get("latency_p50_ms"),
        "latency_p95_ms": resources.get("latency_p95_ms"),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = sorted(
        {
            key
            for row in records
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "seed"
        }
    )
    result = {}
    for key in numeric:
        values = np.asarray(
            [row[key] for row in records if row.get(key) is not None], dtype=np.float64
        )
        if values.size:
            result[key] = {
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }
    return result


def _write_records(directory: Path, title: str, records: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    aggregates = _aggregate(records)
    payload = {
        "schema": "imagination-primary-summary-v1",
        "title": title,
        "split": "validation",
        "test_split_evaluated": False,
        "runs": records,
        "aggregate": aggregates,
    }
    save_json(directory / "summary.json", payload)
    columns = sorted({key for row in records for key in row})
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    lines = [
        f"# {title}",
        "",
        "Validation-only descriptive summary. The final test split remains sealed.",
        "",
    ]
    if records:
        key_metrics = (
            "hazard_fnr",
            "hazard_iou",
            "landing_iou",
            "poi_macro_f1",
            "semantic_mean_iou",
            "scene_accuracy",
        )
        lines.extend(
            (
                "| Model | Seed | " + " | ".join(key_metrics) + " |",
                "|---|---:|" + "---:|" * len(key_metrics),
            )
        )
        for row in records:
            lines.append(
                "| "
                + str(row["model"])
                + " | "
                + str(row["seed"])
                + " | "
                + " | ".join(str(row.get(key)) for key in key_metrics)
                + " |"
            )
        lines.extend(
            (
                "",
                "## Mean and standard deviation",
                "",
                "```json",
                json.dumps(aggregates, indent=2),
                "```",
                "",
            )
        )
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _family_summary(result_root: Path, family: str) -> None:
    records = []
    for path in sorted((result_root / family).glob("seed_*/run_summary.json")):
        records.append(_flatten_run(json.loads(path.read_text(encoding="utf-8"))))
    family_dir = result_root / family
    _write_records(family_dir, f"{DISPLAY_NAMES[family]} family summary", records)
    for suffix in ("json", "csv", "md"):
        source = family_dir / f"summary.{suffix}"
        target = family_dir / f"model_family_summary.{suffix}"
        shutil.copy2(source, target)


def _primary_summary(result_root: Path) -> None:
    records = []
    for family in PRIMARY_ORDER:
        for path in sorted((result_root / family).glob("seed_*/run_summary.json")):
            records.append(_flatten_run(json.loads(path.read_text(encoding="utf-8"))))
    _write_records(result_root, "Primary suite summary", records)


def _progress(run_number: int, total_runs: int, family: str, seed: int, suite_started: float):
    run_started = time.monotonic()

    def callback(values: dict[str, Any]) -> None:
        elapsed = time.monotonic() - suite_started
        run_elapsed = time.monotonic() - run_started
        batch_eta = float(values.get("batch_eta_seconds", 0.0))
        line = (
            f"Run {run_number}/{total_runs} | {DISPLAY_NAMES[family]} | seed {seed} | "
            f"epoch {int(values.get('epoch', 0)) + 1}/{values.get('epochs')} | "
            f"batch {values.get('batch')}/{values.get('batches')} | step {values.get('global_step')} | "
            f"loss {values.get('training_loss', 'n/a')} | device {values.get('device', 'n/a')} | "
            f"micro {values.get('micro_batch', 'n/a')} × accum {values.get('gradient_accumulation', 'n/a')} "
            f"= effective {values.get('effective_batch', 'n/a')} | "
            f"elapsed {elapsed / 3600:.2f}h | epoch ETA {batch_eta / 60:.1f}m"
        )
        if (
            values.get("batch", 0) % 50 == 0
            or values.get("batch") == values.get("batches")
            or values.get("stage")
        ):
            print(line, flush=True)
        save_json(
            ARTIFACTS.progress,
            {
                "stage": values.get("stage", "Training"),
                "model": family,
                "display_name": DISPLAY_NAMES[family],
                "seed": seed,
                "run": run_number,
                "total_runs": total_runs,
                "epoch": values.get("epoch"),
                "epochs": values.get("epochs"),
                "batch": values.get("batch"),
                "batches": values.get("batches"),
                "global_step": values.get("global_step"),
                "training_loss": values.get("training_loss"),
                "elapsed_seconds": elapsed,
                "run_elapsed_seconds": run_elapsed,
                "epoch_eta_seconds": batch_eta,
                "estimated_finish": (datetime.now() + timedelta(seconds=batch_eta)).isoformat(),
            },
        )

    return callback


def _load_ready_plan() -> dict[str, Any]:
    readiness_path = ARTIFACT_DIR / "readiness.json"
    plan_path = ARTIFACT_DIR / "training_plan.yaml"
    if (
        not readiness_path.is_file()
        or json.loads(readiness_path.read_text(encoding="utf-8")).get("status") != "READY"
    ):
        raise RuntimeError("Training readiness is not READY")
    if not plan_path.is_file():
        raise RuntimeError("Frozen training plan is missing")
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    if (
        plan["model_order"] != list(PRIMARY_ORDER)
        or plan["seeds"] != list(PRIMARY_SEEDS)
        or plan["max_epochs"] != 80
    ):
        raise RuntimeError(
            "Frozen training plan does not match the required primary order/seeds/epoch budget"
        )
    return plan


def run_sequence(
    *,
    mode: str = "primary",
    prepare_only: bool = False,
    force: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    readiness = prepare_training(device_name=device, force=force)
    if readiness["status"] != "READY":
        return {"status": "NOT_READY", "training_started": False}
    if prepare_only:
        return {
            "status": "READY",
            "training_started": False,
            "training_plan_hash": readiness["training_plan_hash"],
        }

    plan = _load_ready_plan()
    manifest = MANIFEST_DIR / "all.jsonl"
    plan_hash = plan["training_plan_hash"]
    if mode == "primary":
        seeds, epochs, namespace = PRIMARY_SEEDS, 80, "primary"
    else:
        readiness_config = yaml.safe_load(
            (ROOT / "common" / "configs" / "training_readiness.yaml").read_text(encoding="utf-8")
        )
        seeds = (int(readiness_config["development"]["seed"]),)
        epochs = int(readiness_config["development"]["max_epochs"])
        namespace = "development"
    suite = {"name": namespace, "profile": "research", "epochs": epochs}
    planned = [(family, seed) for family in PRIMARY_ORDER for seed in seeds]
    result_root = ARTIFACTS.results / namespace
    suite_started = time.monotonic()
    completed = 0

    for run_number, (family, seed) in enumerate(planned, start=1):
        config = experiment_config(family, seed, suite, family, manifest, {"family": family})
        batch = plan["batch_plan"][family]
        config["training"].update(
            batch_size=int(batch["micro_batch"]),
            gradient_accumulation=int(batch["gradient_accumulation"]),
            device=device,
            max_epochs=epochs,
        )
        config["experiment"] = {
            "mode": mode,
            "display_name": DISPLAY_NAMES[family],
            "training_plan_hash": plan_hash,
            "analytical_boundary": "human_designed_pre_lba",
            "test_split_evaluated": False,
        }
        checkpoint_dir = Path(config["training"]["output"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config_path = checkpoint_dir / "config.yaml"
        effective = _effective_training_config(config, manifest)
        expected_hash = stable_hash(effective)
        complete_path = checkpoint_dir / "complete.json"
        result_dir = result_root / family / f"seed_{seed}"
        summary_path = result_dir / "run_summary.json"

        if complete_path.is_file() and not force:
            completion = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                completion.get("config_hash") != expected_hash
                or completion.get("training_plan_hash") != plan_hash
            ):
                raise RuntimeError(
                    f"Completed run {family}/seed_{seed} is incompatible with the frozen plan"
                )
            if completion.get("complete") and summary_path.is_file():
                print(f"SKIP complete matching run: {family} seed {seed}", flush=True)
                completed += 1
                continue
        if force and (complete_path.exists() or (checkpoint_dir / "last.pt").exists()):
            print(
                f"FORCE will replace checkpoints/history for {family} seed {seed}: {checkpoint_dir}",
                flush=True,
            )
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        print(
            f"\nStarting {run_number}/{len(planned)}: {DISPLAY_NAMES[family]} seed {seed}; "
            f"micro={batch['micro_batch']}, accumulation={batch['gradient_accumulation']}, "
            f"effective={batch['effective_batch']}, epochs<= {epochs}",
            flush=True,
        )
        try:
            checkpoint = train(
                config_path,
                force=force,
                progress_callback=_progress(run_number, len(planned), family, seed, suite_started),
            )
            # A caught SIGINT/SIGTERM returns a safe last checkpoint without a
            # completion marker. Stop the family sequence rather than evaluating
            # or advancing to another model.
            if not complete_path.is_file():
                print(
                    "Training interrupted after a safe checkpoint; rerun the same command to resume.",
                    flush=True,
                )
                return {
                    "status": "INTERRUPTED",
                    "completed_runs": completed,
                    "last_checkpoint": str(checkpoint),
                }
            validation_dir = result_dir / "validation"
            summary = evaluate(checkpoint, manifest, "val", validation_dir, device_name=device)
            result_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, result_dir / "config.yaml")
            for name in ("metrics.csv", "resource_metrics.json"):
                source = validation_dir / name
                if source.is_file():
                    shutil.copy2(source, result_dir / name)
            history = checkpoint_dir / "training_history.csv"
            if history.is_file():
                shutil.copy2(history, result_dir / "training_history.csv")
            record = {
                **summary,
                "model": family,
                "display_name": DISPLAY_NAMES[family],
                "seed": seed,
                "mode": mode,
                "split": "validation",
                "training_plan_hash": plan_hash,
                "analytical_boundary": "human_designed_pre_lba",
                "test_split_evaluated": False,
                "checkpoint": os.path.relpath(checkpoint, result_dir),
            }
            save_json(summary_path, record)
            completed += 1
        except BaseException as error:
            failure_dir = ARTIFACTS.failures
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure_path = (
                failure_dir
                / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{family}_seed_{seed}.json"
            )
            save_json(
                failure_path,
                {
                    "model": family,
                    "seed": seed,
                    "exception": repr(error),
                    "traceback": traceback.format_exc(),
                    "config_hash": expected_hash,
                    "training_plan_hash": plan_hash,
                    "device": device,
                    "last_checkpoint": str(checkpoint_dir / "last.pt")
                    if (checkpoint_dir / "last.pt").exists()
                    else None,
                },
            )
            print(f"Run failed; report saved to {failure_path}", flush=True)
            raise

        # Family summary is written only after all configured seeds have
        # finished because model families must remain block-sequential.
        if all(
            (result_root / family / f"seed_{value}" / "run_summary.json").is_file()
            for value in seeds
        ):
            _family_summary(result_root, family)

    _primary_summary(result_root)
    return {
        "status": "COMPLETE",
        "mode": mode,
        "completed_runs": completed,
        "planned_runs": len(planned),
        "training_plan_hash": plan_hash,
        "test_split_evaluated": False,
    }


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primary", "development"), default="primary")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args(argv)
    # Resume is deliberately the default. --no-resume is accepted for CLI
    # clarity but cannot erase matching work; use --force for an intentional restart.
    if not args.resume and not args.force:
        parser.error(
            "--no-resume requires --force; normal execution protects and resumes existing work"
        )
    result = run_sequence(
        mode=args.mode,
        prepare_only=args.prepare_only,
        force=args.force,
        device=args.device,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"READY", "COMPLETE", "INTERRUPTED"} else 2
