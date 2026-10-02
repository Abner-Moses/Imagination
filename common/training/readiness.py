"""Repeatable gates that freeze the pre-LBA primary training experiment."""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Subset

from common.artifacts import ARTIFACTS
from common.mapping import build_predicted_contexts
from common.models import build_model
from common.models.registry import MODEL_SPECS, PRIMARY_TRAINING_ORDER
from common.registry import (
    ANALYTICAL_CHANNELS,
    CANDIDATE_FEATURE_NAMES,
    CHANNEL_REGISTRY_VERSION,
    MODEL_ARCHITECTURE_VERSION,
    POI_CLASSES,
    RELATIONAL_NAMES,
    SEMANTIC_CLASSES,
    TRAINING_PLAN_SCHEMA_VERSION,
    TRAINING_READINESS_SCHEMA_VERSION,
    TRAINING_STATISTICS_VERSION,
)
from common.runtime import (
    file_sha256,
    git_commit,
    load_checkpoint,
    load_config,
    parameter_count,
    save_checkpoint,
    save_json,
    select_device,
    set_seed,
    source_tree_hash,
    stable_hash,
)
from common.training.engine import (
    attach_map_context,
    forward_model,
    model_mode,
    move_targets,
)
from common.training.losses import multitask_loss
from data.adapter import (
    ImaginationDataset,
    load_manifest,
    resolve_path,
    validate_manifest,
)
from data.preprocessing.prepare import (
    candidate_depth_cache_path,
    compute_training_statistics,
    ensure_candidate_depth_cache,
)
from data.preprocessing.targets import (
    targets_from_sources,
    semantic_grid,
    scene_verdict,
)


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ARTIFACTS.training_readiness
MANIFEST_DIR = ROOT / "data" / "manifests"
PRIMARY_ORDER = PRIMARY_TRAINING_ORDER
DISPLAY_NAMES = {model_id: spec.display_name for model_id, spec in MODEL_SPECS.items()}
PRIMARY_SEEDS = (7, 17, 29)


@dataclass
class Gate:
    name: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _run(command: list[str], *, timeout: float | None = None) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "command": command,
        "returncode": process.returncode,
        "seconds": time.monotonic() - started,
        "stdout_tail": process.stdout[-4000:],
        "stderr_tail": process.stderr[-4000:],
    }


def device_metadata(requested: str = "auto") -> tuple[torch.device, dict[str, Any]]:
    device = select_device(requested)
    metadata: dict[str, Any] = {
        "requested": requested,
        "selected": str(device),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "mixed_precision": device.type == "cuda",
    }
    if device.type == "cuda":
        metadata.update(
            name=torch.cuda.get_device_name(device),
            memory_bytes=torch.cuda.get_device_properties(device).total_memory,
        )
    elif device.type == "mps":
        metadata.update(name="Apple Metal Performance Shaders", memory_bytes=None)
    else:
        metadata.update(name=platform.processor() or "CPU", memory_bytes=None)
    return device, metadata


def validate_source_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Validate actual source paths and episode structure; the old report is evidence only."""
    root = ROOT / "data" / "dataset"
    episodes = sorted(path for path in root.glob("episode_*") if path.is_dir())
    expected_episodes = int(config["expected_episodes"])
    expected_frames = int(config["expected_frames"])
    if len(episodes) != expected_episodes:
        raise RuntimeError(f"Expected {expected_episodes} source episodes; found {len(episodes)}")

    records = load_manifest(MANIFEST_DIR / "all.jsonl")
    if len(records) != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} manifest frames; found {len(records)}")
    seen: set[str] = set()
    episode_counts = Counter()
    malformed: list[str] = []
    required_record_paths = (
        "rgb_path",
        "semantic_path",
        "landing_source_path",
        "depth_path",
        "objects_path",
        "pose_path",
        "calibration_path",
    )
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in seen:
            raise RuntimeError(f"Missing/duplicate source frame identifier: {sample_id!r}")
        seen.add(sample_id)
        episode_counts[record["episode_id"]] += 1
        for key in required_record_paths:
            value = record.get(key)
            if not value or not resolve_path(MANIFEST_DIR, value).is_file():
                malformed.append(f"{sample_id}:{key}")
                if len(malformed) >= 20:
                    break
        if malformed:
            break
    if malformed:
        raise RuntimeError(f"Missing required source files: {malformed[:20]}")

    frame_total = 0
    for episode in episodes:
        metadata_path = episode / "metadata.json"
        required = (
            metadata_path,
            episode / "objects.json",
            episode / "poses.csv",
            episode / "sensors.csv",
            episode / "sensors_clean.csv",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Episode {episode.name} lacks metadata/sensors: {missing}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frame_count = int(metadata.get("frame_count", -1))
        counts = {
            "rgb": len(list((episode / "rgb").glob("*.jpg"))),
            "semantic": len(list((episode / "segmentation").glob("*.png"))),
            "landing": len(list((episode / "landing_suitability").glob("*.png"))),
            "depth": len(list((episode / "depth").glob("*.npy"))),
        }
        with (episode / "sensors_clean.csv").open(newline="", encoding="utf-8") as stream:
            sensor_rows = sum(1 for _ in csv.DictReader(stream))
        if any(value != frame_count for value in (*counts.values(), sensor_rows)):
            raise RuntimeError(
                f"Inconsistent source episode {episode.name}: declared={frame_count}, "
                f"files={counts}, sensor_rows={sensor_rows}"
            )
        if episode_counts[episode.name] != frame_count:
            raise RuntimeError(f"Manifest/source mismatch for {episode.name}")
        frame_total += frame_count
    if frame_total != expected_frames:
        raise RuntimeError(
            f"Actual source frame total is {frame_total}, expected {expected_frames}"
        )

    old_report_path = root / "validation_report.json"
    old_report = (
        json.loads(old_report_path.read_text(encoding="utf-8")) if old_report_path.is_file() else {}
    )
    report_matches = (
        old_report.get("status") == "PASS"
        and old_report.get("episode_count") == len(episodes)
        and old_report.get("validated_frames") == frame_total
    )
    if not report_matches:
        raise RuntimeError("Existing validation_report.json disagrees with actual source files")
    return {
        "episodes": len(episodes),
        "frames": frame_total,
        "files_checked": len(records) * len(required_record_paths),
        "validation_report_verified": True,
    }


def verify_frozen_manifests() -> dict[str, Any]:
    lock_path = MANIFEST_DIR / "manifest_lock.json"
    if not lock_path.is_file():
        raise RuntimeError("Frozen manifest lock is absent; generate and review manifests first")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    hashes = {}
    split_records: dict[str, list[dict]] = {}
    for split in ("all", "train", "val", "test"):
        path = MANIFEST_DIR / f"{split}.jsonl"
        hashes[split] = file_sha256(path)
        if hashes[split] != lock.get("hashes", {}).get(split):
            raise RuntimeError(f"Frozen {split} manifest hash differs from manifest_lock.json")
        split_records[split] = load_manifest(path)
    validate_manifest(MANIFEST_DIR / "all.jsonl", check_files=True, require_analytical=False)

    all_ids = {row["sample_id"] for row in split_records["all"]}
    selected_sets = {
        name: {row["sample_id"] for row in split_records[name]} for name in ("train", "val", "test")
    }
    if set.union(*selected_sets.values()) != all_ids:
        raise RuntimeError("Split manifests do not partition all.jsonl")
    if any(
        selected_sets[a] & selected_sets[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise RuntimeError("Frame overlap exists between train/validation/test manifests")
    owner: dict[tuple[str, str], str] = {}
    episode_counts, frame_counts = {}, {}
    for split in ("train", "val", "test"):
        rows = split_records[split]
        frame_counts[split] = len(rows)
        episode_counts[split] = len({row["episode_id"] for row in rows})
        for row in rows:
            for kind, key in (
                ("episode", "episode_id"),
                ("trajectory", "trajectory_id"),
                ("environment", "environment_id"),
            ):
                value = row.get(key)
                if not value:
                    continue
                identity = (kind, str(value))
                previous = owner.setdefault(identity, split)
                if previous != split:
                    raise RuntimeError(f"{kind} leakage between {previous} and {split}: {value}")
    return {
        "hashes": hashes,
        "frame_counts": frame_counts,
        "episode_counts": episode_counts,
        "grouping": ["episode", "trajectory", "environment"],
        "test_policy": "existence/hash/overlap only; no test labels or model metrics opened",
    }


def _cache_inventory(records: list[dict]) -> dict[str, int]:
    counts = Counter()
    for row in records:
        path = resolve_path(MANIFEST_DIR, row["analytical_path"])
        if not path.is_file():
            counts["missing"] += 1
            continue
        try:
            with np.load(path, allow_pickle=False) as cache:
                valid = (
                    cache["features"].shape == (28, 32, 32)
                    and cache["validity"].shape == (28, 32, 32)
                    and tuple(str(value) for value in cache["channel_names"].tolist())
                    == ANALYTICAL_CHANNELS
                    and str(cache["registry_version"].item()) == CHANNEL_REGISTRY_VERSION
                    and str(cache["source_sample_id"].item()) == row["sample_id"]
                    and np.isfinite(cache["features"]).all()
                )
            counts["valid" if valid else "invalid"] += 1
        except (OSError, ValueError, KeyError):
            counts["invalid"] += 1
    return {
        "valid": counts["valid"],
        "missing": counts["missing"],
        "invalid": counts["invalid"],
    }


def prepare_and_verify_caches() -> tuple[dict[str, Any], dict[str, Any]]:
    from common.runner import preextract

    manifest = MANIFEST_DIR / "all.jsonl"
    records = load_manifest(manifest)
    before = _cache_inventory(records)
    extraction = preextract()
    after = _cache_inventory(records)
    if after["missing"] or after["invalid"] or after["valid"] != len(records):
        raise RuntimeError(f"Analytical cache remains incomplete: {after}")
    analytical = {
        "expected_frames": len(records),
        "before": before,
        "after": after,
        "regenerated_frames": int(extraction["newly_extracted_frames"]),
        "seconds": extraction["seconds"],
    }

    depth_counts = Counter()
    for row in records:
        path = candidate_depth_cache_path(MANIFEST_DIR, row)
        if not path.is_file():
            depth_counts["missing"] += 1
            continue
        try:
            with np.load(path, allow_pickle=False) as cache:
                valid = (
                    cache["depth_m"].shape == (32, 32)
                    and cache["validity"].shape == (32, 32)
                    and str(cache["source_units"].item()) == "camera_z_metres"
                    and np.isfinite(cache["depth_m"]).all()
                )
            depth_counts["valid" if valid else "invalid"] += 1
        except (OSError, ValueError, KeyError):
            depth_counts["invalid"] += 1
    depth = {
        "expected_grids": len(records),
        "valid_grids": depth_counts["valid"],
        "generated_or_updated_grids": int(extraction.get("candidate_depth_created_or_updated", 0)),
        "missing_grids": depth_counts["missing"],
        "invalid_grids": depth_counts["invalid"],
        "use": "target-only candidate supervision",
    }
    if depth_counts["missing"] or depth_counts["invalid"] or depth_counts["valid"] != len(records):
        raise RuntimeError(f"Candidate target-depth cache remains incomplete: {depth}")
    return analytical, depth


def ensure_training_statistics() -> dict[str, Any]:
    manifest = MANIFEST_DIR / "all.jsonl"
    path = MANIFEST_DIR / "training_statistics.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    expected_hash = file_sha256(manifest)
    train_ids = [row["sample_id"] for row in load_manifest(manifest) if row["split"] == "train"]
    expected_split_hash = stable_hash(train_ids)
    reused = (
        existing.get("schema") == TRAINING_STATISTICS_VERSION
        and existing.get("manifest_hash") == expected_hash
        and existing.get("train_split_hash") == expected_split_hash
    )
    result = existing if reused else compute_training_statistics(manifest)
    if result.get("source_split") != "train" or result.get("samples") != len(train_ids):
        raise RuntimeError("Automatic weights were not derived exclusively from the training split")
    return {**result, "reused": reused}


def _finite_tensor_contract(sample: dict[str, Any], family: str) -> None:
    required = {
        "vehicle_state": (13,),
        "state_validity": (13,),
        "relational": (len(RELATIONAL_NAMES),),
        "relational_validity": (len(RELATIONAL_NAMES),),
        "candidates": (32, len(CANDIDATE_FEATURE_NAMES)),
        "candidate_validity": (32,),
        "hazard_target": (1, 32, 32),
        "landing_target": (1, 32, 32),
        "semantic_target": (32, 32),
        "poi_target": (len(POI_CLASSES), 32, 32),
    }
    if family == "imf_htransformer":
        required.update(analytical=(28, 32, 32), validity=(28, 32, 32))
    else:
        required["rgb"] = (3, 256, 256)
    for name, shape in required.items():
        value = sample.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise RuntimeError(
                f"{family} field {name} has {getattr(value, 'shape', None)}, expected {shape}"
            )
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError(f"{family} field {name} contains NaN/Inf")


def audit_data_contracts(config: dict[str, Any]) -> dict[str, Any]:
    count = int(config["contract_samples_per_split"])
    report: dict[str, Any] = {}
    for family in PRIMARY_ORDER:
        mode = model_mode(family)
        family_report = {}
        for split in ("train", "val"):
            dataset = ImaginationDataset(MANIFEST_DIR / "all.jsonl", split, mode)
            indices = np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int)
            for index in indices:
                _finite_tensor_contract(dataset[int(index)], family)
            family_report[split] = {
                "samples_checked": len(indices),
                "records": len(dataset),
            }
        report[family] = family_report
    return report


def target_sanity() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "val"):
        rows = [row for row in load_manifest(MANIFEST_DIR / "all.jsonl") if row["split"] == split]
        hazard = Counter()
        landing = Counter()
        semantic = Counter()
        poi = Counter()
        scene = Counter()
        for row in rows:
            landing_ids = np.asarray(
                Image.open(resolve_path(MANIFEST_DIR, row["landing_source_path"])).convert("L")
            )
            semantic_ids = np.asarray(
                Image.open(resolve_path(MANIFEST_DIR, row["semantic_path"])).convert("L")
            )
            hazard_target, landing_target, poi_target = targets_from_sources(
                landing_ids, semantic_ids
            )
            hazard["positive"] += int(hazard_target.sum())
            hazard["total"] += int(hazard_target.size)
            landing["positive"] += int(landing_target.sum())
            landing["total"] += int(landing_target.size)
            semantic_grid_values = semantic_grid(semantic_ids)
            for index, amount in enumerate(
                np.bincount(semantic_grid_values.ravel(), minlength=len(SEMANTIC_CLASSES))
            ):
                semantic[str(SEMANTIC_CLASSES[index])] += int(amount)
            for index, name in enumerate(POI_CLASSES):
                poi[name] += int(poi_target[index].sum())
            scene[str(scene_verdict(landing_ids, semantic_ids))] += 1
        hazard["negative"] = hazard["total"] - hazard["positive"]
        landing["negative"] = landing["total"] - landing["positive"]
        supervised_nonzero = (
            hazard["positive"]
            and landing["positive"]
            and sum(poi.values())
            and sum(semantic.values())
        )
        if not supervised_nonzero:
            raise RuntimeError(f"Structurally empty target family in {split}")
        result[split] = {
            "frames": len(rows),
            "hazard": dict(hazard),
            "landing": dict(landing),
            "semantic": dict(semantic),
            "poi": dict(poi),
            "scene": dict(scene),
        }
    return result


def _subset_datasets(family: str, split: str, samples: int):
    manifest = MANIFEST_DIR / "all.jsonl"
    visual = ImaginationDataset(manifest, split, model_mode(family), include_targets=False)
    geometry = (
        visual
        if family == "imf_htransformer"
        else ImaginationDataset(manifest, split, "analytical", include_targets=False)
    )
    target = ImaginationDataset(manifest, split, model_mode(family), include_targets=True)
    return visual, geometry, target, min(samples, len(target))


def _build_context_subset(model, family: str, split: str, samples: int, device: torch.device):
    visual, geometry, target, count = _subset_datasets(family, split, samples)
    contexts = build_predicted_contexts(
        model,
        family,
        visual,
        geometry,
        device,
        ROOT / "data" / "configs" / "semantic_rules.yaml",
        max_samples=count,
    )
    indices = [index for index, row in enumerate(target.records) if row["sample_id"] in contexts]
    return contexts, Subset(target, indices), indices


def candidate_supervision_coverage(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    frames = int(config["candidate_coverage_frames_per_split"])
    family = "cnn_htransformer"
    from common.runner import family_config

    model_config = family_config(family)
    model_config["model"]["family"] = family
    model_config["model"]["profile"] = "research"
    set_seed(7, True)
    model = build_model(model_config).to(device).eval()
    report = {
        "basis": "deterministic fresh CNN-HTransformer predictions; targets used only after context construction"
    }
    for split in ("train", "val"):
        contexts, dataset, _ = _build_context_subset(model, family, split, frames, device)
        counts = Counter()
        type_counts = Counter()
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        for batch in loader:
            attach_map_context(batch, contexts)
            candidates = batch["candidates"]
            token_valid = batch["candidate_validity"] > 0
            projection_valid = batch["candidate_projection_validity"] > 0
            target_valid = batch["candidate_target_validity"] > 0
            counts["slots_considered"] += token_valid.numel()
            counts["valid_candidate_tokens"] += int(token_valid.sum())
            counts["projected_candidates"] += int((token_valid & projection_valid).sum())
            counts["valid_projected_supervision"] += int(target_valid.sum())
            counts["candidate_risk_targets"] += int(target_valid.sum())
            counts["candidate_landing_targets"] += int(target_valid.sum())
            counts["invalid_candidate"] += int((~token_valid).sum())
            counts["projection_invalid_or_unresolved"] += int(
                (token_valid & ~projection_valid).sum()
            )
            grid = batch["candidate_grid"].long()
            x, y = grid[..., 0].clamp(0, 31), grid[..., 1].clamp(0, 31)
            batch_index = torch.arange(grid.shape[0]).unsqueeze(1)
            depth_valid = batch["candidate_depth_validity"][batch_index, y, x] > 0
            counts["missing_target_depth"] += int(
                (token_valid & projection_valid & ~depth_valid).sum()
            )
            counts["depth_mismatch"] += int(
                (token_valid & projection_valid & depth_valid & ~target_valid).sum()
            )
            for name, index in (
                ("landing", 6),
                ("obstacle", 7),
                ("restricted", 8),
                ("poi", 9),
                ("high_risk", 10),
                ("terrain", 11),
            ):
                typed = token_valid & (candidates[..., index] > 0.5)
                type_counts[f"{name}_tokens"] += int(typed.sum())
                type_counts[f"{name}_supervised"] += int((typed & target_valid).sum())
        valid_tokens = counts["valid_candidate_tokens"]
        supervised = counts["valid_projected_supervision"]
        counts["valid_supervision_fraction"] = supervised / max(valid_tokens, 1)
        report[split] = {
            "sampled_frames": len(contexts),
            **dict(counts),
            "by_type": dict(type_counts),
        }
        if supervised == 0:
            raise RuntimeError(
                f"Candidate heads are enabled but {split} has zero valid candidate supervision"
            )
    del model
    gc.collect()
    return report


def run_software_tests() -> dict[str, Any]:
    python = _run([sys.executable, "-m", "pytest", "-q", "tests/python"])
    if python["returncode"]:
        raise RuntimeError(f"Python tests failed:\n{python['stderr_tail']}")
    build = ARTIFACT_DIR / "cpp_build"
    configure = _run(
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
    if configure["returncode"]:
        raise RuntimeError(f"C++ configure failed:\n{configure['stderr_tail']}")
    compile_result = _run(["cmake", "--build", str(build), "--parallel", "2"])
    if compile_result["returncode"]:
        raise RuntimeError(f"C++ build failed:\n{compile_result['stderr_tail']}")
    cpp = _run(["ctest", "--test-dir", str(build), "--output-on-failure"])
    if cpp["returncode"]:
        raise RuntimeError(f"C++ tests failed:\n{cpp['stdout_tail']}\n{cpp['stderr_tail']}")
    return {
        "python": python,
        "cpp_configure": configure,
        "cpp_build": compile_result,
        "cpp_tests": cpp,
    }


def run_smoke_suite() -> dict[str, Any]:
    from common.runner import run_suite

    return run_suite("smoke", force=False, resume=True)


def _resolved_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    statistics = json.loads((MANIFEST_DIR / "training_statistics.json").read_text(encoding="utf-8"))
    result = dict(config["losses"])
    for key, source in (
        ("hazard_positive_weight", "hazard"),
        ("landing_positive_weight", "landing"),
        ("poi_positive_weight", "poi"),
    ):
        if result.get(key) == "auto":
            result[key] = statistics["positive_weights"][source]
    return result


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _clear_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def probe_models(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Real training data forward/backward, microbatch safety, runtime, and gradients."""
    from common.runner import family_config

    target_effective = int(config["target_effective_batch"])
    result: dict[str, Any] = {}
    for family in PRIMARY_ORDER:
        family_cfg = family_config(family)
        family_cfg["model"]["family"] = family
        family_cfg["model"]["profile"] = "research"
        loss_config = _resolved_loss_config(family_cfg)
        selected_batch = None
        chosen = None
        model = None
        contexts = None
        dataset = None
        errors = []
        for micro_batch in config["micro_batch_candidates"]:
            if target_effective % int(micro_batch):
                continue
            try:
                set_seed(7, True)
                model = build_model(family_cfg).to(device)
                contexts, dataset, _ = _build_context_subset(
                    model, family, "train", max(16, int(micro_batch)), device
                )
                loader = DataLoader(
                    dataset, batch_size=int(micro_batch), shuffle=False, num_workers=0
                )
                selected_batch = next(iter(loader))
                move_targets(selected_batch, device)
                attach_map_context(selected_batch, contexts)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(family_cfg["training"]["learning_rate"]),
                )
                optimizer.zero_grad(set_to_none=True)
                started = time.perf_counter()
                output = forward_model(model, selected_batch, family, device)
                losses = multitask_loss(output, selected_batch, loss_config)
                losses["total"].backward()
                _device_sync(device)
                elapsed = time.perf_counter() - started
                finite_gradients = [
                    parameter.grad for parameter in model.parameters() if parameter.grad is not None
                ]
                if (
                    not torch.isfinite(losses["total"])
                    or not finite_gradients
                    or not all(torch.isfinite(value).all() for value in finite_gradients)
                ):
                    raise RuntimeError("non-finite loss/gradient")
                optimizer.step()
                chosen = int(micro_batch)
                break
            except RuntimeError as error:
                errors.append({"micro_batch": int(micro_batch), "error": str(error)})
                del model
                model = None
                _clear_device(device)
                if (
                    "out of memory" not in str(error).lower()
                    and "mps backend out of memory" not in str(error).lower()
                ):
                    raise
        if chosen is None or model is None or selected_batch is None:
            raise RuntimeError(f"{family} failed even at micro-batch 1: {errors}")

        head_gradients = {}
        heads = getattr(model, "heads", None)
        if heads is not None:
            for name in (
                "hazard",
                "landing",
                "semantic",
                "poi",
                "scene",
                "candidate_risk",
                "candidate_landing",
            ):
                module = getattr(heads, name, None)
                if module is not None:
                    head_gradients[name] = any(
                        parameter.grad is not None and torch.isfinite(parameter.grad).all()
                        for parameter in module.parameters()
                    )
        candidate_valid = int(selected_batch.get("candidate_target_validity", torch.zeros(1)).sum())
        for name, received in head_gradients.items():
            if name.startswith("candidate") and candidate_valid == 0:
                continue
            if not received:
                raise RuntimeError(f"{family} supervised {name} head has no finite gradient")

        # Measure a few repeated real-data steps after warmup. This includes the
        # model/loss/backward path; map rollout is measured separately below.
        times = []
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(family_cfg["training"]["learning_rate"])
        )
        for _ in range(int(config["runtime_probe"]["training_batches"]) + 1):
            optimizer.zero_grad(set_to_none=True)
            _device_sync(device)
            started = time.perf_counter()
            output = forward_model(model, selected_batch, family, device)
            current_losses = multitask_loss(output, selected_batch, loss_config)
            current_losses["total"].backward()
            optimizer.step()
            _device_sync(device)
            times.append(time.perf_counter() - started)
        times = times[1:]
        params = parameter_count(model)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as stream:
            state_path = Path(stream.name)
        torch.save(model.state_dict(), state_path)
        checkpoint_bytes = state_path.stat().st_size
        state_path.unlink()
        context_started = time.perf_counter()
        context_frames = int(config["runtime_probe"]["map_context_frames"])
        build_predicted_contexts(
            model,
            family,
            ImaginationDataset(
                MANIFEST_DIR / "all.jsonl",
                "train",
                model_mode(family),
                include_targets=False,
            ),
            ImaginationDataset(
                MANIFEST_DIR / "all.jsonl", "train", "analytical", include_targets=False
            )
            if family != "imf_htransformer"
            else ImaginationDataset(
                MANIFEST_DIR / "all.jsonl", "train", "analytical", include_targets=False
            ),
            device,
            ROOT / "data" / "configs" / "semantic_rules.yaml",
            max_samples=context_frames,
        )
        _device_sync(device)
        context_seconds = time.perf_counter() - context_started
        micro_steps_per_epoch = math.ceil(14000 / chosen)
        model_seconds_per_epoch = float(np.median(times)) * micro_steps_per_epoch
        context_seconds_per_frame = context_seconds / max(context_frames, 1)
        context_epoch_seconds = context_seconds_per_frame * (14000 + 3040)
        estimated_epoch = model_seconds_per_epoch + context_epoch_seconds
        result[family] = {
            "micro_batch": chosen,
            "gradient_accumulation": target_effective // chosen,
            "effective_batch": target_effective,
            "total_loss": float(losses["total"].detach()),
            "finite_gradients": True,
            "head_gradients": head_gradients,
            "candidate_supervised_in_batch": candidate_valid,
            "parameters": params,
            "checkpoint_state_bytes": checkpoint_bytes,
            "median_training_microbatch_seconds": float(np.median(times)),
            "p95_training_microbatch_seconds": float(np.percentile(times, 95)),
            "map_context_seconds_per_frame": context_seconds_per_frame,
            "estimated_model_train_epoch_seconds": model_seconds_per_epoch,
            "estimated_map_train_and_validation_seconds": context_epoch_seconds,
            "estimated_epoch_seconds": estimated_epoch,
            "estimated_run_seconds_80_epochs": estimated_epoch * 80,
            "requested_microbatch_errors": errors,
        }
        del model, selected_batch, contexts, dataset
        _clear_device(device)
    return result


def tiny_real_overfit(
    config: dict[str, Any], device: torch.device, probes: dict[str, Any]
) -> dict[str, Any]:
    from common.runner import family_config

    settings = config["overfit"]
    result = {}
    for family in PRIMARY_ORDER:
        set_seed(7, True)
        family_cfg = family_config(family)
        family_cfg["model"]["family"] = family
        family_cfg["model"]["profile"] = "research"
        model = build_model(family_cfg).to(device)
        contexts, dataset, _ = _build_context_subset(
            model, family, "train", int(settings["samples"]), device
        )
        loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False, num_workers=0)
        batches = list(loader)
        loss_config = _resolved_loss_config(family_cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]))
        initial = best = final = None
        initial_parts = final_parts = None
        finite = True
        for step in range(int(settings["steps"])):
            batch = batches[step % len(batches)]
            move_targets(batch, device)
            attach_map_context(batch, contexts)
            optimizer.zero_grad(set_to_none=True)
            outputs = forward_model(model, batch, family, device)
            losses = multitask_loss(outputs, batch, loss_config)
            losses["total"].backward()
            gradients = [
                parameter.grad for parameter in model.parameters() if parameter.grad is not None
            ]
            finite = (
                finite
                and bool(gradients)
                and all(bool(torch.isfinite(value).all()) for value in gradients)
            )
            optimizer.step()
            current = float(losses["total"].detach())
            parts = {name: float(value.detach()) for name, value in losses.items()}
            if initial is None:
                initial, initial_parts = current, parts
            best = current if best is None else min(best, current)
            final, final_parts = current, parts
        drop = (initial - best) / max(abs(initial), 1e-12)
        if not finite or drop < float(settings["minimum_relative_loss_drop"]):
            raise RuntimeError(
                f"{family} real-data overfit failed: initial={initial:.5f}, best={best:.5f}, drop={drop:.3f}"
            )
        result[family] = {
            "samples": len(dataset),
            "steps": int(settings["steps"]),
            "initial_loss": initial,
            "best_loss": best,
            "final_loss": final,
            "relative_best_loss_drop": drop,
            "minimum_required_drop": settings["minimum_relative_loss_drop"],
            "criterion_note": settings["criterion_note"],
            "finite_gradients": finite,
            "initial_per_head": initial_parts,
            "final_per_head": final_parts,
        }
        del model, contexts, dataset, batches
        _clear_device(device)
    return result


def checkpoint_resume_test(device: torch.device) -> dict[str, Any]:
    from common.runner import family_config

    family = "cnn"
    config = family_config(family)
    config["model"].update(family=family, profile="tiny")
    set_seed(7, True)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint = ARTIFACT_DIR / "checkpoint_resume" / "last.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        0,
        3,
        config,
        {"early_stopping_bad_epochs": 2},
        file_sha256(MANIFEST_DIR / "all.jsonl"),
        scaler=scaler,
        batch_in_epoch=3,
        best_metric=1.25,
    )
    temporary_absent = not checkpoint.with_name(checkpoint.name + ".tmp").exists()
    restored = build_model(config).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, 2)
    restored_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    payload = load_checkpoint(
        checkpoint,
        restored,
        restored_optimizer,
        restored_scheduler,
        restored_scaler,
        device,
    )
    if (
        payload["global_step"] != 3
        or payload["batch_in_epoch"] != 3
        or payload["best_validation_metric"] != 1.25
    ):
        raise RuntimeError("Checkpoint did not restore epoch/step/best state")
    # Prove continuation rather than reset with one fresh real training step.
    dataset = ImaginationDataset(MANIFEST_DIR / "all.jsonl", "train", "rgb_current")
    batch = next(iter(DataLoader(Subset(dataset, range(1)), batch_size=1)))
    move_targets(batch, device)
    restored_optimizer.zero_grad(set_to_none=True)
    losses = multitask_loss(
        forward_model(restored, batch, family, device),
        batch,
        _resolved_loss_config(config),
    )
    losses["total"].backward()
    restored_optimizer.step()
    continued_step = int(payload["global_step"]) + 1
    return {
        "model": family,
        "restored_epoch": payload["epoch"],
        "restored_global_step": payload["global_step"],
        "continued_global_step": continued_step,
        "optimizer": True,
        "scheduler": True,
        "scaler": device.type == "cuda",
        "rng_state": "rng_state" in payload,
        "best_metric": payload["best_validation_metric"],
        "atomic_temp_absent": temporary_absent,
        "continued_loss": float(losses["total"].detach()),
    }


def export_compatibility_test() -> dict[str, Any]:
    """Export each smoke checkpoint and run the installed ONNX checker."""
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="imagination_onnx_") as temporary:
        output_root = Path(temporary)
        for family in PRIMARY_ORDER:
            checkpoint = (
                ARTIFACTS.run_checkpoints("smoke", MODEL_ARCHITECTURE_VERSION, family, 7)
                / "best.pt"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Smoke checkpoint required for export is missing: {checkpoint}"
                )
            output = output_root / f"{family}.onnx"
            process = _run(
                [
                    sys.executable,
                    "-m",
                    "tools.export_onnx",
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(output),
                ],
                timeout=300,
            )
            if process["returncode"]:
                raise RuntimeError(
                    f"ONNX export failed for {family}:\n{process['stdout_tail']}\n"
                    f"{process['stderr_tail']}"
                )
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            results[family] = {
                "bytes": output.stat().st_size,
                "onnx_checker": "PASS",
                "onnxruntime_numerical_check": metadata["onnxruntime_numerical_check"],
                "sparse_selection": metadata["sparse_selection"],
            }
    return results


def causal_map_guard() -> dict[str, Any]:
    source = (ROOT / "common" / "mapping" / "context.py").read_text(encoding="utf-8")
    forbidden = (
        "semantic_target",
        "landing_target",
        "scene_target",
        "candidate_depth_target",
    )
    used = [name for name in forbidden if name in source]
    if used:
        raise RuntimeError(f"Predicted-context builder mentions target-only fields: {used}")
    # The builder snapshots result before OnlinePerceptionMap.process applies
    # the current prediction update; the invariant also has dedicated unit tests.
    online = (ROOT / "common" / "mapping" / "online.py").read_text(encoding="utf-8")
    relation_position = online.find("relation_features = compute_relations(")
    update_position = online.find("self.local_map.update_region(")
    if relation_position < 0 or update_position < 0 or relation_position > update_position:
        raise RuntimeError(
            "Could not verify context-before-current-update ordering in OnlinePerceptionMap"
        )
    return {
        "frame_t_context": "predictions through frame t-1 plus permitted analytical geometry/state",
        "frame_t_update": "after model prediction",
        "target_fields_in_context_builder": used,
        "test_split_used": False,
    }


def disk_readiness(probes: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    free = shutil.disk_usage(ROOT).free
    checkpoint_bytes = (
        sum(item["checkpoint_state_bytes"] for item in probes.values()) * 2 * len(PRIMARY_SEEDS)
    )
    validation = int(float(config["disk"]["validation_artifacts_gb"]) * 1e9)
    cache_remaining = sum(
        path.stat().st_size for path in (ROOT / "data" / "cache").glob("episode_*/*.npz")
    )
    required = checkpoint_bytes + validation
    safety = float(config["disk"]["safety_multiplier"])
    if free < required * safety:
        raise RuntimeError(
            f"Free disk {free} is below required safety allocation {required * safety}"
        )
    return {
        "free_bytes": free,
        "measured_current_cache_bytes": cache_remaining,
        "estimated_checkpoint_bytes": checkpoint_bytes,
        "estimated_validation_and_summary_bytes": validation,
        "estimated_requirement_bytes": required,
        "safety_multiplier": safety,
        "remaining_after_requirement_bytes": free - int(required * safety),
    }


def runtime_estimate(probes: dict[str, Any]) -> dict[str, Any]:
    models = {}
    total = 0.0
    for family in PRIMARY_ORDER:
        one = float(probes[family]["estimated_run_seconds_80_epochs"])
        block = one * len(PRIMARY_SEEDS)
        total += block
        models[family] = {
            "estimated_epoch_seconds": probes[family]["estimated_epoch_seconds"],
            "estimated_one_run_seconds_80_epoch_max": one,
            "estimated_three_seed_seconds": block,
            "map_context_seconds_per_frame": probes[family]["map_context_seconds_per_frame"],
        }
    result = {
        "schema": "imagination-runtime-estimate-v1",
        "label": "ESTIMATE",
        "basis": "short real-training-data forward/backward and causal-map timing probes",
        "models": models,
        "estimated_primary_suite_seconds": total,
        "early_stopping_may_reduce_runtime": True,
    }
    save_json(ARTIFACT_DIR / "runtime_estimate.json", result)
    return result


def _cache_contract_hash() -> str:
    markers = []
    for path in sorted((ROOT / "data" / "cache").glob("episode_*/complete.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        markers.append((path.parent.name, record.get("signature")))
    return stable_hash(markers)


def freeze_training_plan(
    manifest_report: dict[str, Any],
    probes: dict[str, Any],
    device: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    from common.runner import family_config

    plan = {
        "schema": TRAINING_PLAN_SCHEMA_VERSION,
        "analytical_boundary": "human_designed_pre_lba",
        "architecture_version": MODEL_ARCHITECTURE_VERSION,
        "dataset_manifest_hashes": manifest_report["hashes"],
        "analytical_cache_contract_hash": _cache_contract_hash(),
        "extractor_configuration_hash": file_sha256(
            ROOT / "data" / "configs" / "pre_extraction.yaml"
        ),
        "model_order": list(PRIMARY_ORDER),
        "model_display_names": DISPLAY_NAMES,
        "seeds": list(PRIMARY_SEEDS),
        "profile": "research",
        "max_epochs": 80,
        "early_stopping": {"patience": 12, "minimum_epochs": 8, "minimum_delta": 0.001},
        "target_effective_batch": 8,
        "batch_plan": {
            family: {
                "micro_batch": probes[family]["micro_batch"],
                "gradient_accumulation": probes[family]["gradient_accumulation"],
                "effective_batch": probes[family]["effective_batch"],
            }
            for family in PRIMARY_ORDER
        },
        "device": device,
        "optimizer": {
            key: family_config("cnn_htransformer")["training"][key]
            for key in (
                "learning_rate",
                "weight_decay",
                "gradient_clip",
                "mixed_precision",
                "deterministic",
            )
        },
        "loss_settings": family_config("cnn_htransformer")["losses"],
        "model_config_hashes": {
            family: stable_hash(family_config(family)) for family in PRIMARY_ORDER
        },
        "git_commit": git_commit(),
        "source_tree_hash": source_tree_hash(ROOT),
        "test_split_policy": "sealed: existence/hash/overlap checks only until deliberate final evaluation",
    }
    plan_hash = stable_hash(plan)
    plan["training_plan_hash"] = plan_hash
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    canonical = ARTIFACT_DIR / "training_plan.yaml"
    versioned = ARTIFACT_DIR / f"training_plan_{plan_hash[:12]}.yaml"
    if canonical.is_file():
        existing = yaml.safe_load(canonical.read_text(encoding="utf-8"))
        if existing.get("training_plan_hash") != plan_hash:
            if not force:
                raise RuntimeError(
                    "Frozen training_plan.yaml differs from current readiness result. "
                    "Review the change and rerun preparation with --force to create a new versioned plan."
                )
            old_hash = existing.get("training_plan_hash", "unknown")
            archive = ARTIFACT_DIR / f"training_plan_superseded_{str(old_hash)[:12]}.yaml"
            if not archive.exists():
                shutil.copy2(canonical, archive)
    text = yaml.safe_dump(plan, sort_keys=False)
    if not versioned.exists():
        versioned.write_text(text, encoding="utf-8")
    canonical.write_text(text, encoding="utf-8")
    return plan


def _write_readiness(report: dict[str, Any]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(ARTIFACT_DIR / "readiness.json", _jsonable(report))
    lines = [
        "# Training readiness",
        "",
        f"Final status: **{report['status']}**",
        "",
        "The final test split remained sealed. This pass performed existence, hash, and overlap checks only.",
        "",
    ]
    for gate in report["gates"]:
        lines.extend((f"## {gate['name']}", "", f"**{gate['status']}**", ""))
        if gate.get("error"):
            lines.extend((f"Error: `{gate['error']}`", ""))
        lines.extend(
            (
                "```json",
                json.dumps(gate.get("details", {}), indent=2, default=str),
                "```",
                "",
            )
        )
    if report.get("warnings"):
        lines.extend(("## Warnings", ""))
        lines.extend(f"- {warning}" for warning in report["warnings"])
        lines.append("")
    lines.extend(("## Final status", "", f"**{report['status']}**", ""))
    (ARTIFACT_DIR / "readiness.md").write_text("\n".join(lines), encoding="utf-8")


def prepare_training(*, device_name: str = "auto", force: bool = False) -> dict[str, Any]:
    """Run every mandatory gate and return READY/NOT_READY without starting training."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "common" / "configs" / "training_readiness.yaml")
    gates: list[Gate] = []
    warnings: list[str] = []
    values: dict[str, Any] = {}

    def gate(name: str, action: Callable[[], Any], *, mandatory: bool = True) -> Any:
        print(f"\n[training readiness] {name}", flush=True)
        try:
            result = action()
            values[name] = result
            gates.append(Gate(name, "PASS", _jsonable(result)))
            return result
        except Exception as error:  # Keep a complete, reviewable failure report.
            gates.append(Gate(name, "FAIL" if mandatory else "WARNING", error=str(error)))
            if not mandatory:
                warnings.append(f"{name}: {error}")
            return None

    source = gate("DATASET", lambda: validate_source_dataset(config))
    manifests = gate("SPLITS", verify_frozen_manifests) if source else None
    cache_pair = gate("CACHES", prepare_and_verify_caches) if manifests else None
    if cache_pair:
        analytical, candidate_depth = cache_pair
        gates[-1].details = {
            "analytical": analytical,
            "candidate_target_depth": candidate_depth,
        }
    statistics = gate("TRAIN STATISTICS", ensure_training_statistics) if cache_pair else None
    contracts = gate("DATA CONTRACT", lambda: audit_data_contracts(config)) if cache_pair else None
    targets = gate("TARGET SANITY", target_sanity) if contracts else None
    tests = gate("SOFTWARE TESTS", run_software_tests)
    causal = gate("CAUSAL MAP LEAKAGE", causal_map_guard)
    smoke = gate("SMOKE SUITE", run_smoke_suite) if tests else None
    device_pair = gate("DEVICE", lambda: device_metadata(device_name))
    device, hardware = device_pair if device_pair else (torch.device("cpu"), {})
    if device.type == "mps":
        warnings.append(
            "Apple MPS is selected. Mixed precision is disabled, and PyTorch reports "
            "that some backward kernels are not bitwise deterministic on this backend."
        )
    elif device.type == "cpu":
        warnings.append("CPU training is supported but the 12-run primary suite will be slow.")
    coverage = (
        gate(
            "CANDIDATE SUPERVISION",
            lambda: candidate_supervision_coverage(config, device),
        )
        if cache_pair
        else None
    )
    if coverage:
        for split in ("train", "val"):
            fraction = coverage[split]["valid_supervision_fraction"]
            if 0 < fraction < 0.05:
                warnings.append(f"Low nonzero candidate supervision in {split}: {fraction:.3%}")
    probes = (
        gate("REAL-DATA FWD/BWD + BATCH PLAN", lambda: probe_models(config, device))
        if cache_pair and statistics
        else None
    )
    overfit = (
        gate("REAL-DATA TINY OVERFIT", lambda: tiny_real_overfit(config, device, probes))
        if probes
        else None
    )
    resume = gate("CHECKPOINT RESUME", lambda: checkpoint_resume_test(device)) if probes else None
    if smoke:
        gate("EXPORT", export_compatibility_test, mandatory=False)
    disk = gate("DISK", lambda: disk_readiness(probes, config)) if probes else None
    runtime = gate("RUNTIME ESTIMATE", lambda: runtime_estimate(probes)) if probes else None
    plan = (
        gate(
            "FROZEN TRAINING PLAN",
            lambda: freeze_training_plan(manifests, probes, hardware, force=force),
        )
        if manifests and probes and disk and runtime
        else None
    )

    mandatory_failed = any(item.status == "FAIL" for item in gates)
    status = "NOT_READY" if mandatory_failed else "READY"
    report = {
        "schema": TRAINING_READINESS_SCHEMA_VERSION,
        "status": status,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git_commit(),
        "gates": [_jsonable(gate.__dict__) for gate in gates],
        "warnings": warnings,
        "test_split_evaluated": False,
        "full_primary_training_started": False,
        "training_plan_hash": plan.get("training_plan_hash") if plan else None,
    }
    _write_readiness(report)
    print(f"\nFINAL STATUS: {status}", flush=True)
    return report
