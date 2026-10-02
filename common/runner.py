"""Cross-platform repository entry point shared by run.py and model scripts."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch
import yaml
import numpy as np

from common.artifacts import ARTIFACTS as ARTIFACT_LAYOUT
from common.evaluation import evaluate
from common.models import build_model
from common.models.registry import MODEL_IDS, model_spec
from common.registry import (
    CANDIDATE_FEATURE_VERSION,
    CHANNEL_REGISTRY_VERSION,
    METRIC_ATTENTION_VERSION,
    MODEL_ARCHITECTURE_VERSION,
    SEMANTIC_FUSION_VERSION,
    contract_versions,
)
from common.runtime import (
    benchmark_model,
    file_sha256,
    load_config,
    merge_dicts,
    save_json,
    select_device,
    stable_hash,
)
from common.training.engine import train
from data.adapter import load_manifest, validate_manifest
from data.preprocessing.prepare import (
    build_manifests,
    cache_manifest,
    compute_training_statistics,
    ensure_candidate_depth_caches,
    generate_smoke_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ARTIFACT_LAYOUT.root
MANIFESTS = ROOT / "data" / "manifests"
MODEL_FAMILIES = MODEL_IDS


def _run(command: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _config(name: str) -> dict:
    return load_config(ROOT / "common" / "configs" / name)


def family_config(family: str) -> dict:
    """Load one model family on top of the shared model contract."""
    spec = model_spec(family)
    return merge_dicts(
        _config("model_common.yaml"),
        load_config(spec.config_path(ROOT)),
    )


def ensure_manifests(force: bool = False) -> Path:
    all_manifest = MANIFESTS / "all.jsonl"
    lock_path = MANIFESTS / "manifest_lock.json"
    if all_manifest.exists() and lock_path.exists() and not force:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        expected = lock.get("hashes", {})
        for split in ("all", "train", "val", "test"):
            path = MANIFESTS / f"{split}.jsonl"
            if not path.is_file() or file_sha256(path) != expected.get(split):
                raise RuntimeError(
                    "Frozen data manifests changed; stop and review data/manifests/manifest_lock.json"
                )
        print("Using frozen episode-level manifests.")
        return all_manifest
    dataset = ROOT / "data" / "dataset"
    if not dataset.is_dir():
        raise FileNotFoundError(f"Existing rendered dataset not found: {dataset}")
    result = build_manifests(dataset, MANIFESTS)
    print(
        f"Created grouped manifests: {result['split_episodes']} episodes, {result['split_frames']} frames"
    )
    return all_manifest


def validate_repository(*, run_cpp: bool = True) -> dict:
    for module in ("torch", "numpy", "cv2", "PIL", "yaml"):
        __import__(module)
    raw_root = ROOT / "data" / "dataset"
    report_path = raw_root / "validation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError("The existing raw dataset validation report is missing")
    raw_report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        raw_report.get("status") != "PASS"
        or raw_report.get("episode_count") != 250
        or raw_report.get("validated_frames") != 20000
    ):
        raise RuntimeError(
            "Existing 20k-frame raw dataset report does not match the expected validated source"
        )
    manifest = ensure_manifests()
    manifest_report = validate_manifest(manifest, check_files=True, require_analytical=False)
    if (
        manifest_report.get("episodes") != 250
        or sum(manifest_report.get(k, 0) for k in ("train", "val", "test")) != 20000
    ):
        raise RuntimeError(f"Manifest counts do not match the 20k source: {manifest_report}")
    if run_cpp:
        _run(
            [
                "cmake",
                "-S",
                str(ROOT),
                "-B",
                str(ARTIFACTS / "build"),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_TESTING=ON",
            ]
        )
        _run(["cmake", "--build", str(ARTIFACTS / "build"), "--parallel", "2"])
        _run(["ctest", "--test-dir", str(ARTIFACTS / "build"), "--output-on-failure"])
    _run([sys.executable, "-m", "pytest", "-q", "tests/python"])
    return {"raw_dataset": raw_report["status"], "manifest": manifest_report, "cpp": run_cpp}


def _cache_estimate(frames: int) -> int:
    # Uncompressed float32 28×32×32 features plus one-byte validity, with headroom.
    return int(frames * (28 * 32 * 32 * 5) * 1.5)


def _prepare_candidate_depth(manifest: Path, episode_ids=None) -> int:
    """Prepare a compact target-only camera-Z grid without touching rendered RGB."""
    records = load_manifest(manifest)
    if episode_ids is not None:
        allowed = set(episode_ids)
        records = [row for row in records if row["episode_id"] in allowed]
    if not any(row.get("depth_path") for row in records):
        return 0
    started = time.monotonic()
    last_print = {"frame": 0}

    def progress(done: int, total: int):
        if done == total or done - last_print["frame"] >= 1000:
            elapsed = time.monotonic() - started
            remaining = elapsed / max(done, 1) * max(total - done, 0)
            save_json(
                ARTIFACTS / "progress.json",
                {
                    "stage": "Candidate supervision depth cache",
                    "frames": done,
                    "total_frames": total,
                    "elapsed_seconds": elapsed,
                    "estimated_remaining_seconds": remaining,
                    "estimated_finish": (datetime.now() + timedelta(seconds=remaining)).isoformat(),
                },
            )
            print(
                f"Candidate target depth: {done:,}/{total:,}; "
                f"elapsed {elapsed / 60:.1f} min; ETA {remaining / 60:.1f} min",
                flush=True,
            )
            last_print["frame"] = done

    created = ensure_candidate_depth_caches(manifest, progress=progress, episode_ids=episode_ids)
    print(f"Candidate target-depth grids created/updated: {created:,}", flush=True)
    return created


def preextract(limit_episodes: int | None = None, force: bool = False) -> dict:
    manifest = ensure_manifests()
    build = ARTIFACTS / "build"
    executable = build / ("imagination.exe" if sys.platform == "win32" else "imagination")
    if not executable.exists():
        _run(
            [
                "cmake",
                "-S",
                str(ROOT),
                "-B",
                str(build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DBUILD_TESTING=ON",
            ]
        )
        _run(["cmake", "--build", str(build), "--parallel", "2"])
    records = load_manifest(manifest)
    episodes = sorted({record["episode_id"] for record in records})
    chosen = episodes if limit_episodes is None else episodes[:limit_episodes]
    frames = sum(record["episode_id"] in set(chosen) for record in records)
    expected_bytes = _cache_estimate(frames)
    free = shutil.disk_usage(ROOT).free
    print(
        f"Analytical cache: {frames:,} frames; upper estimate {expected_bytes / 1e9:.2f} GB; free {free / 1e9:.2f} GB"
    )
    if free < expected_bytes * 1.15:
        raise RuntimeError("Insufficient free disk for the requested resumable analytical cache")
    candidate_depth_created = _prepare_candidate_depth(manifest, chosen)
    start = time.monotonic()
    progress = {"done": 0, "last_print": 0}

    def show(done: int, total: int):
        progress["done"] = done
        elapsed = time.monotonic() - start
        remaining = elapsed / max(done, 1) * max(total - done, 0)
        save_json(
            ARTIFACTS / "progress.json",
            {
                "stage": "Analytical cache",
                "frames": done,
                "total_frames": total,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": remaining,
                "estimated_finish": (datetime.now() + timedelta(seconds=remaining)).isoformat(),
            },
        )
        if done == total or done - progress["last_print"] >= 1000:
            print(
                f"Analytical cache: {done:,}/{total:,}; elapsed {elapsed / 60:.1f} min; "
                f"ETA {remaining / 60:.1f} min",
                flush=True,
            )
            progress["last_print"] = done

    completed = cache_manifest(
        manifest,
        executable,
        ROOT / "data" / "configs" / "pre_extraction.yaml",
        limit_episodes=limit_episodes,
        progress=show,
        force=force,
    )
    return {
        "requested_frames": frames,
        "newly_extracted_frames": completed,
        "candidate_depth_created_or_updated": candidate_depth_created,
        "seconds": time.monotonic() - start,
    }


def _effective_training_config(config: dict, manifest: Path) -> dict:
    """Apply train-split weights exactly as the engine does before hashing a run."""
    result = copy.deepcopy(config)
    statistics_path = manifest.parent / "training_statistics.json"
    statistics = (
        json.loads(statistics_path.read_text(encoding="utf-8")) if statistics_path.is_file() else {}
    )
    weights = statistics.get("positive_weights", {})
    for key, source in (
        ("hazard_positive_weight", "hazard"),
        ("landing_positive_weight", "landing"),
        ("poi_positive_weight", "poi"),
    ):
        if result.get("losses", {}).get(key) == "auto":
            result["losses"][key] = weights.get(source, 1.0)
    return result


def experiment_config(
    family: str, seed: int, suite: dict, run_name: str, manifest: Path, run_spec: dict
) -> dict:
    config = family_config(family)
    config["model"]["family"] = family
    config["model"]["profile"] = suite.get("profile", "research")
    config["training"]["seed"] = seed
    config["training"]["max_epochs"] = int(suite.get("epochs", 80))
    config["training"]["output"] = str(
        ARTIFACT_LAYOUT.run_checkpoints(suite["name"], MODEL_ARCHITECTURE_VERSION, run_name, seed)
    )
    config["data"]["manifest"] = str(manifest)
    if run_spec.get("disabled_families"):
        config["model"]["disabled_families"] = run_spec["disabled_families"]
    if "state_conditioning" in run_spec:
        config["model"]["state_conditioning"] = run_spec["state_conditioning"]
    if "map_context" in run_spec:
        config["model"]["map_context"] = run_spec["map_context"]
    if "depths" in run_spec:
        config["model"]["stage3_depth"], config["model"]["stage4_depth"] = run_spec["depths"]
    if "fusion" in run_spec:
        config["model"]["fusion"] = run_spec["fusion"]
    if "attention" in run_spec:
        config["model"]["attention"].update(run_spec["attention"])
    if "mapping" in run_spec:
        config["mapping"] = merge_dicts(config.get("mapping", {}), run_spec["mapping"])
    return config


def _planned_runs(suite: dict) -> list[dict]:
    runs = [
        {"family": family, "name": family, "seed": seed}
        for family in suite["models"]
        for seed in suite["seeds"]
    ]
    for ablation in suite.get("ablations", []):
        for seed in ablation.get("seeds", suite["seeds"]):
            runs.append(
                {**ablation, "family": ablation["base"], "name": ablation["name"], "seed": seed}
            )
    sensitivity = suite.get("sensitivity", {})
    if sensitivity.get("enabled"):
        for seed in sensitivity.get("seeds", suite["seeds"]):
            runs.append(
                {
                    "family": sensitivity["base"],
                    "name": "imf_depth_5_2",
                    "seed": seed,
                    "depths": sensitivity.get("depths", [5, 2]),
                }
            )
    return runs


def _probe(config: dict, manifest: Path) -> dict:
    """Measure warm training and validation batches at the configured batch size."""
    import gc
    from torch.utils.data import DataLoader, Subset
    from common.mapping import build_predicted_contexts
    from common.training.engine import forward_model, model_mode, move_targets
    from common.training.losses import multitask_loss
    from data.adapter import ImaginationDataset

    family = config["model"]["family"]
    device = select_device(config["training"].get("device", "auto"))
    data = ImaginationDataset(
        manifest, "train", model_mode(family), config.get("poi", {}).get("classes")
    )
    batch_size = int(config["training"].get("batch_size", 8))
    requested_batch_size = batch_size
    model = None
    while True:
        try:
            # Three warm-up steps plus up to 25 measured steps. Smoke fixtures
            # stay short; real manifests give a more stable estimate.
            training_subset = Subset(data, range(min(len(data), batch_size * 28)))
            loader = DataLoader(training_subset, batch_size=batch_size)
            validation_data = ImaginationDataset(
                manifest, "val", model_mode(family), config.get("poi", {}).get("classes")
            )
            validation_loader = DataLoader(
                Subset(validation_data, range(min(len(validation_data), batch_size * 4))),
                batch_size=batch_size,
            )
            model = build_model(config).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=float(config["training"]["learning_rate"])
            )
            train_times = []
            loader_waits = []
            warmup = min(3, max(0, len(loader) - 1))
            model.train()
            last_step_end = time.perf_counter()
            for index, batch in enumerate(loader):
                ready = time.perf_counter()
                loader_waits.append(ready - last_step_end)
                move_targets(batch, device)
                started = time.perf_counter()
                output = forward_model(model, batch, family, device)
                losses = multitask_loss(output, batch, config["losses"])
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                last_step_end = time.perf_counter()
                if index >= warmup:
                    train_times.append(last_step_end - started)
                if len(train_times) >= 25:
                    break

            model.eval()
            validation_times = []
            with torch.inference_mode():
                for batch in validation_loader:
                    move_targets(batch, device)
                    started = time.perf_counter()
                    output = forward_model(model, batch, family, device)
                    multitask_loss(output, batch, config["losses"])
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    validation_times.append(time.perf_counter() - started)
                    if len(validation_times) >= 3:
                        break
            break
        except RuntimeError as error:
            if "out of memory" not in str(error).lower() or batch_size <= 1:
                raise
            if model is not None:
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(
                f"Timing probe ran out of memory; retrying with batch size {batch_size}.",
                flush=True,
            )

    if batch_size != requested_batch_size:
        config["training"]["batch_size"] = batch_size
    median_step = float(np.median(train_times)) if train_times else 0.0
    median_loader_wait = float(np.median(loader_waits[warmup:])) if loader_waits else 0.0
    median_validation = float(np.median(validation_times)) if validation_times else 0.0
    train_steps = math.ceil(len(data) / batch_size)
    validation_steps = math.ceil(len(validation_data) / batch_size)
    train_epoch = (median_step + median_loader_wait) * train_steps
    validation_epoch = median_validation * validation_steps
    map_context = bool(config["model"].get("map_context", True))
    context_seconds_per_frame = 0.0
    context_train_seconds = context_validation_seconds = 0.0
    context_probe_frames = 0
    if map_context:
        train_context_visual = ImaginationDataset(
            manifest,
            "train",
            model_mode(family),
            config.get("poi", {}).get("classes"),
            include_targets=False,
        )
        val_context_visual = ImaginationDataset(
            manifest,
            "val",
            model_mode(family),
            config.get("poi", {}).get("classes"),
            include_targets=False,
        )
        train_geometry = (
            train_context_visual
            if family == "imf_htransformer"
            else ImaginationDataset(manifest, "train", "analytical", include_targets=False)
        )
        val_geometry = (
            val_context_visual
            if family == "imf_htransformer"
            else ImaginationDataset(manifest, "val", "analytical", include_targets=False)
        )
        map_config = ROOT / "data" / "configs" / "semantic_rules.yaml"
        context_start = time.perf_counter()
        train_contexts = build_predicted_contexts(
            model,
            family,
            train_context_visual,
            train_geometry,
            device,
            map_config,
            max_samples=min(64, len(train_context_visual)),
            map_overrides=config.get("mapping"),
        )
        train_context_probe = time.perf_counter() - context_start
        context_start = time.perf_counter()
        validation_contexts = build_predicted_contexts(
            model,
            family,
            val_context_visual,
            val_geometry,
            device,
            map_config,
            max_samples=min(32, len(val_context_visual)),
            map_overrides=config.get("mapping"),
        )
        validation_context_probe = time.perf_counter() - context_start
        context_probe_frames = len(train_contexts) + len(validation_contexts)
        context_seconds_per_frame = (train_context_probe + validation_context_probe) / max(
            context_probe_frames, 1
        )
        context_train_seconds = context_seconds_per_frame * len(data)
        context_validation_seconds = context_seconds_per_frame * len(validation_data)
    return {
        "requested_batch_size": requested_batch_size,
        "batch_size": batch_size,
        "timed_training_batches": len(train_times),
        "median_training_step_seconds": median_step,
        "median_loader_wait_seconds": median_loader_wait,
        "median_validation_batch_seconds": median_validation,
        "estimated_train_epoch_seconds": train_epoch,
        "estimated_validation_epoch_seconds": validation_epoch,
        "context_probe_frames": context_probe_frames,
        "context_seconds_per_frame": context_seconds_per_frame,
        "estimated_train_context_seconds": context_train_seconds,
        "estimated_validation_context_seconds": context_validation_seconds,
        "estimated_epoch_seconds": train_epoch
        + validation_epoch
        + context_train_seconds
        + context_validation_seconds,
    }


def _progress_writer(suite_name: str, total_runs: int):
    path = ARTIFACTS / "progress.json"
    start = time.monotonic()
    history = []

    def report(run_index: int, family: str, seed: int, run_elapsed_before: float, values: dict):
        history.append(values["elapsed_seconds"])
        elapsed = time.monotonic() - start
        remaining = max(0.0, values.get("suite_eta_seconds", 0.0))
        payload = {
            "stage": "Training",
            "suite": suite_name,
            "model": family,
            "seed": seed,
            "run": run_index,
            "total_runs": total_runs,
            "epoch": values["epoch"] + 1,
            "epochs": values["epochs"],
            "batch": values["batch"],
            "batches": values["batches"],
            "global_step": values["global_step"],
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": remaining,
            "estimated_finish": (datetime.now() + timedelta(seconds=remaining)).isoformat(),
        }
        save_json(path, payload)
        if values["batch"] % 50 == 0 or values["batch"] == values["batches"]:
            print(
                f"{suite_name}: run {run_index}/{total_runs} {family} seed={seed}; "
                f"epoch {values['epoch'] + 1}/{values['epochs']} batch {values['batch']}/{values['batches']}; "
                f"elapsed {elapsed / 3600:.2f}h, suite ETA {remaining / 3600:.2f}h",
                flush=True,
            )

    return report


def _write_suite_summary(result_root: Path) -> list[dict]:
    """Flatten completed validation records for review without opening checkpoints."""
    records = []
    for path in sorted(result_root.glob("**/run_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary.get("metrics", {})
        resources = summary.get("resources", {})
        records.append(
            {
                "model": summary.get("model"),
                "run_name": summary.get("run_name"),
                "seed": summary.get("seed"),
                "hazard_fnr": metrics.get("hazard_fnr"),
                "hazard_recall": metrics.get("hazard_recall"),
                "hazard_precision": metrics.get("hazard_precision"),
                "hazard_iou": metrics.get("hazard_iou"),
                "landing_iou": metrics.get("landing_iou"),
                "landing_candidate_detection_rate": metrics.get("landing_candidate_detection_rate"),
                "poi_macro_f1": metrics.get("poi_macro_f1"),
                "semantic_mean_iou": metrics.get("semantic_mean_iou"),
                "scene_accuracy": metrics.get("scene_accuracy"),
                "parameters": resources.get("parameters"),
                "macs_per_sample": resources.get("estimated_macs_per_sample"),
                "model_latency_p50_ms": resources.get("latency_p50_ms"),
                "model_latency_p95_ms": resources.get("latency_p95_ms"),
                "pre_extraction_latency_ms": resources.get("pre_extraction_latency_ms"),
                "end_to_end_latency_ms": resources.get("end_to_end_latency_ms"),
                "average_power_w": resources.get("average_power_w"),
                "energy_joules_per_frame": resources.get("energy_joules_per_frame"),
            }
        )

    result_root.mkdir(parents=True, exist_ok=True)
    columns = (
        list(records[0])
        if records
        else [
            "model",
            "run_name",
            "seed",
            "hazard_fnr",
            "landing_iou",
            "poi_macro_f1",
            "parameters",
            "macs_per_sample",
            "model_latency_p50_ms",
            "model_latency_p95_ms",
        ]
    )
    with (result_root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    lines = [
        "# Experiment summary",
        "",
        "Validation results only. Test results are not included unless a separate frozen evaluation is explicitly run.",
        "Energy and end-to-end fields remain empty until measured.",
        "",
        "| Model/run | Seed | Hazard FNR | Landing IoU | POI macro F1 | Parameters | Model p50 ms | Model p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in records:
        lines.append(
            f"| {row['run_name']} | {row['seed']} | {row['hazard_fnr']} | {row['landing_iou']} | "
            f"{row['poi_macro_f1']} | {row['parameters']} | {row['model_latency_p50_ms']} | "
            f"{row['model_latency_p95_ms']} |"
        )
    (result_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records


def run_suite(name: str, *, force: bool = False, resume: bool = False) -> dict:
    suite = _config(f"suite_{name}.yaml")
    runs = _planned_runs(suite)
    if name == "smoke":
        _run([sys.executable, "-m", "pytest", "-q", "tests/python"])
    else:
        validate_repository(run_cpp=True)
    manifest = Path(suite.get("manifest", "data/manifests/all.jsonl"))
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    manifest.parent.mkdir(parents=True, exist_ok=True)

    if name == "smoke":
        manifest = generate_smoke_dataset(ARTIFACTS / "smoke_data")
    else:
        manifest = ensure_manifests()
        validate_manifest(manifest, check_files=True, require_analytical=False)

    print(
        f"Suite: {name}; planned runs: {len(runs)}; models: {', '.join(suite['models'])}; seeds: {suite['seeds']}"
    )
    if not runs:
        raise ValueError("The selected experiment suite contains no runs")

    if name != "smoke" and any(
        run["family"] == "imf_htransformer" or run.get("map_context", True) for run in runs
    ):
        cache_complete = all(
            (manifest.parent / record["analytical_path"]).resolve().is_file()
            for record in load_manifest(manifest)
        )
        if not cache_complete:
            print("IMF cache is missing; pre-extracting the existing rendered frames.")
            preextract()
        validate_manifest(manifest, check_files=True, require_analytical=True)
        if any(run.get("map_context", True) for run in runs):
            _prepare_candidate_depth(manifest)
    if not (manifest.parent / "training_statistics.json").is_file():
        compute_training_statistics(manifest)

    from tools.preflight import run_preflight

    preflight_path = ARTIFACTS / "preflight" / f"{name}.json"
    preflight_key = stable_hash(
        {
            "manifest": file_sha256(manifest),
            "registry": CHANNEL_REGISTRY_VERSION,
            "model_architecture": MODEL_ARCHITECTURE_VERSION,
            "candidate_features": CANDIDATE_FEATURE_VERSION,
            "metric_attention": METRIC_ATTENTION_VERSION,
            "semantic_fusion": SEMANTIC_FUSION_VERSION,
            "models": suite["models"],
            "profiles": suite.get("profile", "research"),
        }
    )
    preflight = (
        json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.exists() else {}
    )
    if force or preflight.get("key") != preflight_key:
        print("Running Q/K/V, mapping, checkpoint, and small-sample overfit preflight…", flush=True)
        result = run_preflight(manifest, ARTIFACTS / "preflight" / name)
        save_json(preflight_path, {"key": preflight_key, "result": result})
    else:
        print("Using matching passing preflight record.")

    run_specs = []
    probes = {}
    for spec in runs:
        config = experiment_config(
            spec["family"], spec["seed"], suite, spec["name"], manifest, spec
        )
        # The trainer resolves `auto` weights from the train split before loss
        # construction. The timing probe uses the same loss path, so resolve
        # them here too rather than passing the YAML sentinel string to torch.
        config = _effective_training_config(config, manifest)
        run_specs.append((spec, config))
    print("Timing a few real forward/backward batches before estimating runtime…", flush=True)
    for _, config in run_specs:
        family = config["model"]["family"]
        if family not in probes:
            probes[family] = _probe(config, manifest)
    for _, config in run_specs:
        family = config["model"]["family"]
        config["training"]["batch_size"] = probes[family]["batch_size"]
    total_seconds = sum(
        probes[config["model"]["family"]]["estimated_epoch_seconds"]
        * int(config["training"]["max_epochs"])
        for _, config in run_specs
    )
    available = shutil.disk_usage(ROOT).free
    print(
        f"Measured-probe PRIMARY estimate: {total_seconds / 3600:.2f} h; available disk: {available / 1e9:.2f} GB"
    )
    print(
        "This is an extrapolation from a short local throughput probe; early stopping may shorten it."
    )

    progress = _progress_writer(name, len(runs))
    completed = []
    start = time.monotonic()
    for index, (spec, config) in enumerate(run_specs, start=1):
        checkpoint_dir = Path(config["training"]["output"])
        complete_path = checkpoint_dir / "complete.json"
        expected_hash = stable_hash(_effective_training_config(config, manifest))
        result_dir = ARTIFACTS / "results" / name / spec["name"] / f"seed_{spec['seed']}"
        summary_path = result_dir / "run_summary.json"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config_path = checkpoint_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        checkpoint = None
        if complete_path.exists() and not force:
            record = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                record.get("complete")
                and record.get("config_hash") == expected_hash
                and record.get("manifest_hash") == file_sha256(manifest)
            ):
                if summary_path.is_file():
                    print(f"Skip complete run {spec['name']} seed {spec['seed']}")
                    completed.append(spec["name"])
                    continue
                checkpoint = checkpoint_dir / "best.pt"
                if not checkpoint.is_file():
                    checkpoint = checkpoint_dir / "last.pt"
                print(
                    f"Run {spec['name']} seed {spec['seed']} trained; completing validation report."
                )
        estimated_remaining = total_seconds - (time.monotonic() - start)

        def callback(values):
            values["suite_eta_seconds"] = max(0.0, estimated_remaining)
            progress(index, spec["family"], spec["seed"], 0.0, values)

        if checkpoint is None:
            checkpoint = train(config_path, force=force, progress_callback=callback)
        validation_dir = result_dir / "validation"
        summary = evaluate(checkpoint, manifest, "val", validation_dir)
        shutil.copy2(config_path, result_dir / "config.yaml")
        shutil.copy2(validation_dir / "metrics.csv", result_dir / "metrics.csv")
        shutil.copy2(validation_dir / "resource_metrics.json", result_dir / "resource_metrics.json")
        history_path = checkpoint_dir / "training_history.csv"
        if history_path.is_file():
            shutil.copy2(history_path, result_dir / "training_history.csv")
        relative_checkpoint = Path(os.path.relpath(checkpoint, result_dir))
        (result_dir / "best_checkpoint_reference.txt").write_text(
            str(relative_checkpoint) + "\n", encoding="utf-8"
        )
        run_record = {**summary, "suite": name, "run_name": spec["name"], "seed": spec["seed"]}
        save_json(result_dir / "run_summary.json", run_record)
        (result_dir / "run_summary.md").write_text(
            f"# {spec['name']} seed {spec['seed']}\n\n"
            f"Validation samples: {summary['samples']}\n\n"
            f"Hazard FNR: {summary['metrics'].get('hazard_fnr')}  \n"
            f"Landing IoU: {summary['metrics'].get('landing_iou')}  \n"
            f"POI macro F1: {summary['metrics'].get('poi_macro_f1')}\n\n"
            f"Checkpoint: `{relative_checkpoint}`\n\n"
            "This record uses the validation split. It makes no test, energy or end-to-end claim.\n",
            encoding="utf-8",
        )
        _write_suite_summary(ARTIFACTS / "results" / name)
        completed.append(spec["name"])
    _write_suite_summary(ARTIFACTS / "results" / name)
    save_json(
        ARTIFACTS / "progress.json",
        {
            "stage": "Complete",
            "suite": name,
            "completed_runs": completed,
            "total_runs": len(runs),
            "elapsed_seconds": time.monotonic() - start,
        },
    )
    return {
        "suite": name,
        "runs": len(runs),
        "completed": len(completed),
        "estimated_seconds": total_seconds,
    }


def train_family(
    family: str, seed: int = 7, profile: str = "research", force: bool = False
) -> Path:
    suite = {"name": "manual", "profile": profile, "epochs": 80}
    manifest = ensure_manifests()
    config = experiment_config(family, seed, suite, family, manifest, {"family": family})
    if family == "imf_htransformer" or config["model"].get("map_context", False):
        records = load_manifest(manifest)
        if not all(
            (manifest.parent / row["analytical_path"]).resolve().is_file() for row in records
        ):
            preextract()
        if config["model"].get("map_context", False):
            _prepare_candidate_depth(manifest)
        validate_manifest(manifest, check_files=True, require_analytical=True)
    checkpoint_dir = Path(config["training"]["output"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_path = checkpoint_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return train(config_path, force=force)


def benchmark_models(profile: str = "research") -> dict:
    from common.registry import CANDIDATE_FEATURE_NAMES, MAX_CANDIDATES, RELATIONAL_NAMES, STATE_DIM

    device = select_device()
    results = {}
    for family in MODEL_FAMILIES:
        config = family_config(family)
        config["model"]["family"] = family
        config["model"]["profile"] = profile
        model = build_model(config).to(device).eval()
        state = torch.zeros(1, STATE_DIM, device=device)
        state_validity = torch.zeros_like(state)
        relation = torch.zeros(1, len(RELATIONAL_NAMES), device=device)
        relation_validity = torch.zeros_like(relation)
        candidates = torch.zeros(1, MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES), device=device)
        candidate_validity = torch.zeros(1, MAX_CANDIDATES, device=device)
        candidate_feature_validity = torch.zeros_like(candidates)
        if family == "imf_htransformer":
            analytical = torch.zeros(1, 28, 32, 32, device=device)
            analytical[:, 22] = 0.1  # Two metres under the configured 20 m scale.
            analytical_validity = torch.zeros_like(analytical)
            analytical_validity[:, 22] = 1.0
            grid_index = torch.arange(MAX_CANDIDATES, device=device)
            candidates[:, :, 0] = grid_index.float() / MAX_CANDIDATES
            candidate_validity.fill_(1.0)
            candidate_feature_validity.fill_(1.0)
            inputs = (
                analytical,
                analytical_validity,
                state,
                state_validity,
                relation,
                relation_validity,
                candidates,
                candidate_validity,
                candidate_feature_validity,
                torch.tensor([[256.0, 256.0, 220.0, 220.0, 127.5, 127.5]], device=device),
                torch.tensor([20.0], device=device),
                None,
                None,
                torch.stack((grid_index % 32, grid_index // 32), dim=-1).float()[None],
                torch.full((1, MAX_CANDIDATES), 2.0, device=device),
                torch.ones(1, MAX_CANDIDATES, device=device),
                torch.ones(1, MAX_CANDIDATES, device=device, dtype=torch.long),
            )
        else:
            inputs = (
                torch.zeros(1, 3, 256, 256, device=device),
                state,
                state_validity,
                relation,
                relation_validity,
                candidates,
                candidate_validity,
                candidate_feature_validity,
            )
        result = {
            "profile": profile,
            "device": str(device),
            **benchmark_model(model, inputs, device),
        }
        backbone = getattr(model, "backbone", None)
        attention_stats = getattr(backbone, "last_attention_stats", None)
        if attention_stats:
            pairs = attention_stats["attention_pairs"]
            dense_pairs = attention_stats["dense_reference_pairs"]
            result["attention"] = {
                **attention_stats,
                "pair_reduction_percent": 100.0 * (1.0 - pairs / max(dense_pairs, 1)),
                "map_context_benchmark_tokens": MAX_CANDIDATES
                if family == "imf_htransformer"
                else 0,
            }
        results[family] = result
    output = ARTIFACTS / "benchmarks" / f"models_{profile}.json"
    save_json(output, results)
    print(f"Model benchmarks saved to {output}")
    return results


def health_check() -> dict:
    """Run fast, dataset-independent checks for local development and CI."""
    suite = _config("suite_primary.yaml")
    expected_order = ["cnn_htransformer", "cnn_vit", "cnn", "imf_htransformer"]
    if suite["models"] != expected_order:
        raise RuntimeError(
            f"Primary model order changed: expected {expected_order}, observed {suite['models']}"
        )
    models = {}
    for family in MODEL_FAMILIES:
        config = family_config(family)
        config["model"].update(family=family, profile="tiny")
        model = build_model(config)
        models[family] = {
            "display_name": model_spec(family).display_name,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "observation": model_spec(family).observation,
        }
    return {
        "status": "PASS",
        "models": models,
        "primary_order": expected_order,
        "contracts": contract_versions(),
        "dataset_required": False,
    }


def status() -> None:
    raw_report = ROOT / "data" / "dataset" / "validation_report.json"
    if raw_report.exists():
        record = json.loads(raw_report.read_text(encoding="utf-8"))
        print(
            f"Raw data: {record.get('episode_count')} episodes, {record.get('validated_frames')} frames, {record.get('status')}"
        )
    manifest = MANIFESTS / "manifest_lock.json"
    print(f"Frozen manifests: {'present' if manifest.exists() else 'not generated'}")
    cache_count = len(list((ROOT / "data" / "cache").glob("episode_*/frame_*.npz")))
    print(f"Analytical cache: {cache_count:,} frames")
    print("\nCommands:")
    print("  python run.py --check")
    print("  python run.py --validate")
    print("  python run.py --prepare-training")
    print("  python run.py --preextract [--limit-episodes 1]")
    for family in MODEL_FAMILIES:
        print(f"  python run.py --train {family}")
    print("  python run.py --suite smoke|primary|full")
    print("  python run.py --benchmark")
    print("  python run.py --figures")
    print("  python run.py --resume")
    print("No full experiment starts unless --train or --suite is given.")


def main(argv=None) -> int:
    arguments_list = list(sys.argv[1:] if argv is None else argv)
    if arguments_list and arguments_list[0] == "audit-imf":
        from common.imf_audit.runner import cli as audit_cli

        return audit_cli(arguments_list[1:])
    parser = argparse.ArgumentParser(description="Imagination research runner")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--prepare-training", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--preextract", action="store_true")
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--train", choices=MODEL_FAMILIES)
    parser.add_argument("--suite", choices=("smoke", "primary", "full"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile", choices=("research", "small", "tiny"), default="research")
    arguments = parser.parse_args(arguments_list)

    selected = any(
        (
            arguments.check,
            arguments.validate,
            arguments.prepare_training,
            arguments.preextract,
            arguments.train,
            arguments.suite,
            arguments.resume,
            arguments.benchmark,
            arguments.figures,
        )
    )
    if not selected:
        status()
        return 0
    if arguments.check:
        print(json.dumps(health_check(), indent=2))
    if arguments.validate:
        print(json.dumps(validate_repository(), indent=2))
    if arguments.prepare_training:
        from common.training.readiness import prepare_training

        result = prepare_training(device_name=arguments.device, force=arguments.force)
        print(json.dumps(result, indent=2))
        if result["status"] != "READY":
            return 2
    if arguments.preextract:
        print(json.dumps(preextract(arguments.limit_episodes, arguments.force), indent=2))
    if arguments.train:
        print(train_family(arguments.train, arguments.seed, arguments.profile, arguments.force))
    if arguments.suite:
        print(
            json.dumps(
                run_suite(arguments.suite, force=arguments.force, resume=arguments.resume), indent=2
            )
        )
    elif arguments.resume:
        print(json.dumps(run_suite("primary", force=False, resume=True), indent=2))
    if arguments.benchmark:
        print(json.dumps(benchmark_models(arguments.profile), indent=2))
    if arguments.figures:
        _run(
            [
                sys.executable,
                "-m",
                "tools.generate_figures",
                "--manifest",
                str(MANIFESTS / "all.jsonl"),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
