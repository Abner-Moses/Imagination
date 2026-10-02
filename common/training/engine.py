"""Shared, resumable training loop for the four perception model families."""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from common.contracts import validate_model_outputs
from common.models import build_model
from common.mapping import build_predicted_contexts
from common.registry import POI_CLASSES, SCENE_CLASSES, SEMANTIC_CLASSES
from common.runtime import (
    file_sha256,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_json,
    select_device,
    set_seed,
    stable_hash,
)
from common.training.losses import multitask_loss
from common.training.metrics import BinaryCounts, CategoricalMetrics, POIMetrics
from data.adapter import ImaginationDataset, validate_manifest


class EpochSampler(Sampler):
    """Shuffle identically after resume, independent of prior epoch iteration."""

    def __init__(self, size: int, seed: int):
        self.size = size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self):
        return self.size


def model_mode(family: str) -> str:
    return "analytical" if family == "imf_htransformer" else "rgb_current"


def forward_model(model, batch, family: str, device):
    state = batch["vehicle_state"].to(device, non_blocking=True)
    state_validity = batch["state_validity"].to(device, non_blocking=True)
    relation = batch["relational"].to(device, non_blocking=True)
    relation_validity = batch["relational_validity"].to(device, non_blocking=True)
    candidates = batch["candidates"].to(device, non_blocking=True)
    candidate_validity = batch["candidate_validity"].to(device, non_blocking=True)
    candidate_feature_validity = batch.get("candidate_feature_validity")
    if candidate_feature_validity is not None:
        candidate_feature_validity = candidate_feature_validity.to(device, non_blocking=True)
    if family == "imf_htransformer":
        outputs = model(
            batch["analytical"].to(device, non_blocking=True),
            batch["validity"].to(device, non_blocking=True),
            state,
            state_validity,
            relation,
            relation_validity,
            candidates,
            candidate_validity,
            calibration=batch.get("calibration").to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            if batch.get("calibration") is not None
            else None,
            depth_scale_m=batch.get("depth_scale_m").to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            if batch.get("depth_scale_m") is not None
            else None,
            candidate_grid=batch.get("candidate_grid").to(device, non_blocking=True)
            if batch.get("candidate_grid") is not None
            else None,
            candidate_projected_depth_m=batch.get("candidate_projected_depth_m").to(
                device, non_blocking=True
            )
            if batch.get("candidate_projected_depth_m") is not None
            else None,
            candidate_projection_validity=batch.get("candidate_projection_validity").to(
                device, non_blocking=True
            )
            if batch.get("candidate_projection_validity") is not None
            else None,
            candidate_visibility=batch.get("candidate_visibility").to(device, non_blocking=True)
            if batch.get("candidate_visibility") is not None
            else None,
            candidate_feature_validity=candidate_feature_validity,
        )
    else:
        outputs = model(
            batch["rgb"].to(device, non_blocking=True),
            state,
            state_validity,
            relation,
            relation_validity,
            candidates,
            candidate_validity,
            candidate_feature_validity,
        )
    validate_model_outputs(outputs)
    return outputs


def attach_map_context(batch: dict, contexts: dict) -> None:
    """Add the prior-prediction map snapshot matching each sample in a batch."""
    sample_ids = batch["metadata"]["sample_id"]
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]
    missing = [sample_id for sample_id in sample_ids if sample_id not in contexts]
    if missing:
        raise KeyError(f"Predicted map context is missing samples: {missing[:3]}")
    target = batch.get("landing_class_target", batch.get("hazard_target"))
    device = target.device if target is not None else torch.device("cpu")
    names = (
        "relational",
        "relational_validity",
        "candidates",
        "candidate_validity",
        "candidate_feature_validity",
        "candidate_grid",
        "candidate_projected_depth_m",
        "candidate_projection_validity",
        "candidate_visibility",
    )
    for name in names:
        batch[name] = torch.stack([contexts[sample_id][name] for sample_id in sample_ids]).to(
            device
        )
    if "landing_class_target" in batch:
        grid = batch["candidate_grid"].long()
        x = grid[..., 0].clamp(0, 31)
        y = grid[..., 1].clamp(0, 31)
        batch_index = torch.arange(len(sample_ids), device=grid.device).unsqueeze(1)
        source_class = batch["landing_class_target"][batch_index, y, x]
        # Source classes are 0=unsafe, 1=caution, 2=suitable. The model's
        # scene order is 0=SAFE, 1=CAUTION, 2=DANGEROUS.
        risk_target = torch.where(source_class == 2, 0, torch.where(source_class == 1, 1, 2))
        batch["candidate_risk_target"] = risk_target
        batch["candidate_landing_target"] = (source_class == 2).float()
        depth_at_cell = batch.get("candidate_depth_target_m")
        depth_validity = batch.get("candidate_depth_validity")
        if depth_at_cell is None or depth_validity is None:
            batch["candidate_target_validity"] = torch.zeros_like(
                batch["candidate_projection_validity"]
            )
        else:
            measured = depth_at_cell[batch_index, y, x]
            measured_valid = depth_validity[batch_index, y, x] > 0
            projected = batch["candidate_projected_depth_m"]
            tolerance = torch.maximum(torch.full_like(projected, 0.25), projected * 0.15)
            consistent = torch.abs(measured - projected) <= tolerance
            batch["candidate_target_validity"] = batch["candidate_projection_validity"] * (
                measured_valid & consistent
            ).to(projected.dtype)


def move_targets(batch, device):
    tensor_keys = (
        "hazard_target",
        "hazard_validity",
        "landing_target",
        "landing_validity",
        "poi_target",
        "poi_validity",
        "semantic_target",
        "semantic_validity",
        "scene_target",
        "landing_class_target",
        "candidate_depth_target_m",
        "candidate_depth_validity",
    )
    for key in tensor_keys:
        batch[key] = batch[key].to(device, non_blocking=True)
    for key in ("candidate_risk_target", "candidate_landing_target", "candidate_target_validity"):
        if key in batch:
            batch[key] = batch[key].to(device, non_blocking=True)


def evaluate_loader(model, loader, family, device, loss_config, thresholds, map_contexts=None):
    model.eval()
    totals = {
        key: 0.0
        for key in (
            "total",
            "hazard",
            "landing",
            "poi",
            "semantic",
            "scene",
            "candidate",
            "topology",
        )
    }
    samples = 0
    hazard = BinaryCounts("hazard")
    landing = BinaryCounts("landing")
    poi_names = list(loader.dataset.poi_classes)
    poi = POIMetrics(poi_names)
    semantic = CategoricalMetrics(SEMANTIC_CLASSES)
    scene = CategoricalMetrics(SCENE_CLASSES)
    candidate_risk = CategoricalMetrics(SCENE_CLASSES)
    candidate_landing = BinaryCounts("candidate_landing")

    with torch.inference_mode():
        for batch in loader:
            move_targets(batch, device)
            if map_contexts is not None:
                attach_map_context(batch, map_contexts)
            outputs = forward_model(model, batch, family, device)
            losses = multitask_loss(outputs, batch, loss_config)
            count = int(batch["hazard_target"].shape[0])
            samples += count
            for key in totals:
                totals[key] += float(losses[key]) * count
            hazard.update(
                outputs["hazard_logits"],
                batch["hazard_target"],
                batch["hazard_validity"],
                thresholds.get("hazard", 0.5),
            )
            landing.update(
                outputs["landing_logits"],
                batch["landing_target"],
                batch["landing_validity"],
                thresholds.get("landing", 0.5),
            )
            poi.update(
                outputs["poi"]["class_logits"],
                batch["poi_target"],
                batch["poi_validity"],
                thresholds.get("poi", 0.5),
            )
            semantic.update(
                outputs["semantic_logits"], batch["semantic_target"], batch["semantic_validity"]
            )
            scene.update(outputs["scene_risk_logits"], batch["scene_target"])
            if "candidate_risk_target" in batch:
                candidate_mask = batch["candidate_target_validity"]
                candidate_risk.update(
                    outputs["candidate"]["risk_logits"].transpose(1, 2),
                    batch["candidate_risk_target"],
                    candidate_mask,
                )
                candidate_landing.update(
                    outputs["candidate"]["landing_safe_logits"],
                    batch["candidate_landing_target"],
                    candidate_mask,
                    thresholds.get("candidate_landing", 0.5),
                )

    return {
        **{f"val_{key}_loss": value / max(samples, 1) for key, value in totals.items()},
        **hazard.result(),
        **landing.result(),
        **poi.result(),
        **semantic.result("semantic"),
        **scene.result("scene"),
        **candidate_risk.result("candidate_risk"),
        **candidate_landing.result(),
    }


def _autoweights(config: dict, manifest: Path) -> None:
    statistics = manifest.parent / "training_statistics.json"
    weights = (
        json.loads(statistics.read_text(encoding="utf-8"))["positive_weights"]
        if statistics.exists()
        else {}
    )
    losses = config.setdefault("losses", {})
    for key, source in (
        ("hazard_positive_weight", "hazard"),
        ("landing_positive_weight", "landing"),
        ("poi_positive_weight", "poi"),
    ):
        if losses.get(key) == "auto":
            losses[key] = weights.get(source, 1.0)


def _cache_hash(manifest: Path, required: bool) -> str | None:
    if not required:
        return None
    records = []
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            marker = (
                manifest.parent / record["analytical_path"]
            ).resolve().parent / "complete.json"
            if not marker.is_file():
                raise FileNotFoundError(f"Analytical cache completion marker missing: {marker}")
            records.append(
                (record["episode_id"], json.loads(marker.read_text(encoding="utf-8"))["signature"])
            )
    return stable_hash(sorted(set(records)))


def train(config_path, *, force: bool = False, progress_callback=None):
    config = load_config(config_path) if not isinstance(config_path, dict) else config_path
    model_config = config["model"]
    family = model_config["family"]
    training = config["training"]
    seed = int(training.get("seed", 7))
    set_seed(seed, bool(training.get("deterministic", True)))
    device = select_device(training.get("device", "auto"))
    output_dir = Path(training["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = Path(config["data"]["manifest"])
    map_context_enabled = bool(model_config.get("map_context", True))
    map_config_path = (
        Path(__file__).resolve().parents[2] / "data" / "configs" / "semantic_rules.yaml"
    )
    map_policy_hash = file_sha256(map_config_path) if map_context_enabled else None
    uses_analytical_cache = family == "imf_htransformer" or map_context_enabled
    validate_manifest(
        manifest,
        check_files=bool(config["data"].get("validate_files", True)),
        require_analytical=uses_analytical_cache,
        require_candidate_depth=map_context_enabled,
    )
    _autoweights(config, manifest)
    config_hash = stable_hash(config)
    cache_hash = _cache_hash(manifest, uses_analytical_cache)

    poi_classes = config.get("poi", {}).get("classes", POI_CLASSES)
    train_data = ImaginationDataset(manifest, "train", model_mode(family), poi_classes)
    validation_data = ImaginationDataset(manifest, "val", model_mode(family), poi_classes)
    context_datasets = None
    if map_context_enabled:
        train_context_visual = ImaginationDataset(
            manifest, "train", model_mode(family), poi_classes, include_targets=False
        )
        val_context_visual = ImaginationDataset(
            manifest, "val", model_mode(family), poi_classes, include_targets=False
        )
        if family == "imf_htransformer":
            train_context_geometry = train_context_visual
            val_context_geometry = val_context_visual
        else:
            train_context_geometry = ImaginationDataset(
                manifest, "train", "analytical", include_targets=False
            )
            val_context_geometry = ImaginationDataset(
                manifest, "val", "analytical", include_targets=False
            )
        context_datasets = (
            train_context_visual,
            train_context_geometry,
            val_context_visual,
            val_context_geometry,
        )
    sampler = EpochSampler(len(train_data), seed)
    batch_size = int(training.get("batch_size", 8))
    workers = int(training.get("workers", 0))
    loader_args = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, sampler=sampler, **loader_args)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_args)

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("max_epochs", 80))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1))
    amp_enabled = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    start_epoch = 0
    skip_batches = 0
    global_step = 0
    best_metric = float("inf")
    bad_epochs = 0
    if last_path.exists() and not force:
        checkpoint = load_checkpoint(last_path, model, optimizer, scheduler, scaler, device)
        if checkpoint.get("manifest_hash") != file_sha256(manifest):
            raise ValueError("Checkpoint belongs to a different dataset manifest")
        if checkpoint.get("config_hash") != config_hash:
            raise ValueError("Checkpoint configuration differs; use the matching config or --force")
        if checkpoint.get("analytical_cache_hash") != cache_hash:
            raise ValueError("Checkpoint analytical cache differs from the current episode cache")
        if checkpoint.get("map_policy_hash") != map_policy_hash:
            raise ValueError("Checkpoint semantic-map policy differs from the current map rules")
        start_epoch = int(checkpoint["epoch"])
        skip_batches = int(checkpoint.get("batch_in_epoch", 0))
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint.get("best_validation_metric") or float("inf"))
        bad_epochs = int(checkpoint.get("metrics", {}).get("early_stopping_bad_epochs", 0))
        if skip_batches == 0:
            start_epoch += 1

    stop_requested = False

    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {signal.SIGINT: signal.signal(signal.SIGINT, request_stop)}
    if hasattr(signal, "SIGTERM"):
        previous_handlers[signal.SIGTERM] = signal.signal(signal.SIGTERM, request_stop)
    history_path = output_dir / "training_history.csv"
    if force and history_path.exists():
        history_path.unlink()
    history_exists = history_path.exists() and history_path.stat().st_size > 0
    started = time.monotonic()
    try:
        accumulation = max(1, int(training.get("gradient_accumulation", 1)))
        checkpoint_every = max(0, int(training.get("checkpoint_every_batches", 250)))
        patience = int(training.get("early_stopping_patience", 12))
        min_epochs = int(training.get("early_stopping_min_epochs", 8))
        min_delta = float(training.get("early_stopping_min_delta", 0.001))

        for epoch in range(start_epoch, epochs):
            sampler.set_epoch(epoch)
            epoch_started = time.monotonic()
            map_context_started = time.monotonic()
            train_contexts = None
            if context_datasets is not None:
                context_train_visual, context_train_geometry = context_datasets[:2]

                def report_context(done, total):
                    if progress_callback:
                        progress_callback(
                            {
                                "stage": "Predicted map context",
                                "epoch": epoch,
                                "epochs": epochs,
                                "batch": done,
                                "batches": total,
                                "global_step": global_step,
                                "batch_eta_seconds": 0.0,
                                "elapsed_seconds": time.monotonic() - started,
                            }
                        )

                train_contexts = build_predicted_contexts(
                    model,
                    family,
                    context_train_visual,
                    context_train_geometry,
                    device,
                    map_config_path,
                    report_context if progress_callback else None,
                    map_overrides=config.get("mapping"),
                )
            train_map_context_seconds = time.monotonic() - map_context_started

            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals = {
                key: 0.0
                for key in (
                    "total",
                    "hazard",
                    "landing",
                    "poi",
                    "semantic",
                    "scene",
                    "candidate",
                    "topology",
                )
            }
            seen = 0
            last_batch_end = time.monotonic()
            loader_wait_seconds = 0.0
            step_seconds = 0.0
            for batch_index, batch in enumerate(train_loader):
                if epoch == start_epoch and batch_index < skip_batches:
                    continue
                batch_ready = time.monotonic()
                loader_wait_seconds += batch_ready - last_batch_end
                step_started = batch_ready
                move_targets(batch, device)
                if train_contexts is not None:
                    attach_map_context(batch, train_contexts)
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = forward_model(model, batch, family, device)
                        losses = multitask_loss(outputs, batch, config.get("losses", {}))
                else:
                    outputs = forward_model(model, batch, family, device)
                    losses = multitask_loss(outputs, batch, config.get("losses", {}))
                scaled_loss = losses["total"] / accumulation
                if not torch.isfinite(scaled_loss):
                    raise RuntimeError("Nonfinite training loss")
                scaler.scale(scaled_loss).backward()

                completed_batch = batch_index + 1
                if completed_batch % accumulation == 0 or completed_batch == len(train_loader):
                    if training.get("gradient_clip"):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(training["gradient_clip"])
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                step_seconds += time.monotonic() - step_started

                count = int(batch["hazard_target"].shape[0])
                seen += count
                global_step += 1
                for key, value in losses.items():
                    totals[key] += float(value.detach()) * count

                elapsed = time.monotonic() - epoch_started
                eta = elapsed / max(completed_batch, 1) * (len(train_loader) - completed_batch)
                if progress_callback:
                    progress_callback(
                        {
                            "epoch": epoch,
                            "epochs": epochs,
                            "batch": completed_batch,
                            "batches": len(train_loader),
                            "global_step": global_step,
                            "training_loss": totals["total"] / max(seen, 1),
                            "device": str(device),
                            "micro_batch": batch_size,
                            "gradient_accumulation": accumulation,
                            "effective_batch": batch_size * accumulation,
                            "batch_eta_seconds": eta,
                            "elapsed_seconds": time.monotonic() - started,
                        }
                    )

                safe_boundary = completed_batch % accumulation == 0 or completed_batch == len(
                    train_loader
                )
                if checkpoint_every and global_step % checkpoint_every == 0 and safe_boundary:
                    save_checkpoint(
                        last_path,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        config,
                        {"train_loss": totals["total"] / max(seen, 1)},
                        file_sha256(manifest),
                        scaler=scaler,
                        batch_in_epoch=completed_batch,
                        best_metric=best_metric,
                        analytical_cache_hash=cache_hash,
                        map_policy_hash=map_policy_hash,
                    )
                last_batch_end = time.monotonic()
                if stop_requested:
                    save_checkpoint(
                        last_path,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        config,
                        {"train_loss": totals["total"] / max(seen, 1)},
                        file_sha256(manifest),
                        scaler=scaler,
                        batch_in_epoch=completed_batch,
                        best_metric=best_metric,
                        analytical_cache_hash=cache_hash,
                        map_policy_hash=map_policy_hash,
                    )
                    print("\nSafe to resume with:\npython run.py --resume", flush=True)
                    return last_path

            skip_batches = 0
            train_epoch_seconds = time.monotonic() - epoch_started
            validation_started = time.monotonic()
            validation_context_seconds = 0.0
            validation_contexts = None
            if context_datasets is not None:
                context_val_visual, context_val_geometry = context_datasets[2:]
                map_started = time.monotonic()
                validation_contexts = build_predicted_contexts(
                    model,
                    family,
                    context_val_visual,
                    context_val_geometry,
                    device,
                    map_config_path,
                    map_overrides=config.get("mapping"),
                )
                validation_context_seconds = time.monotonic() - map_started
            validation = evaluate_loader(
                model,
                validation_loader,
                family,
                device,
                config.get("losses", {}),
                config.get("thresholds", {}),
                validation_contexts,
            )
            validation_seconds = time.monotonic() - validation_started
            validation["train_loss"] = totals["total"] / max(seen, 1)
            for key in ("hazard", "landing", "poi", "semantic", "scene", "candidate", "topology"):
                validation[f"train_{key}_loss"] = totals[key] / max(seen, 1)
            validation["epoch"] = epoch + 1
            validation["learning_rate"] = optimizer.param_groups[0]["lr"]
            validation["train_epoch_seconds"] = train_epoch_seconds
            validation["train_loader_wait_seconds"] = loader_wait_seconds
            validation["train_step_seconds"] = step_seconds
            validation["train_map_context_seconds"] = train_map_context_seconds
            validation["validation_map_context_seconds"] = validation_context_seconds
            validation["validation_seconds"] = validation_seconds
            validation["train_samples_per_second"] = seen / max(train_epoch_seconds, 1e-9)
            current = float(validation["val_total_loss"])
            improved = current < best_metric - min_delta
            bad_epochs = 0 if improved else bad_epochs + 1
            validation["early_stopping_bad_epochs"] = bad_epochs
            scheduler.step()

            if improved:
                best_metric = current
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    config,
                    validation,
                    file_sha256(manifest),
                    scaler=scaler,
                    best_metric=best_metric,
                    analytical_cache_hash=cache_hash,
                    map_policy_hash=map_policy_hash,
                )
            save_checkpoint(
                last_path,
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
                validation,
                file_sha256(manifest),
                scaler=scaler,
                best_metric=best_metric,
                analytical_cache_hash=cache_hash,
                map_policy_hash=map_policy_hash,
            )

            history_fields = list(validation)
            if history_exists:
                with history_path.open("r", newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    previous_fields = reader.fieldnames or []
                    previous_rows = list(reader) if previous_fields != history_fields else None
                if previous_rows is not None:
                    # Keep valid historical metrics when a newer code version
                    # adds columns to the training log.
                    merged_fields = list(dict.fromkeys(previous_fields + history_fields))
                    temporary_history = history_path.with_name(history_path.name + ".tmp")
                    with temporary_history.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=merged_fields)
                        writer.writeheader()
                        writer.writerows(previous_rows)
                    temporary_history.replace(history_path)
                    history_fields = merged_fields
            with history_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=history_fields)
                if not history_exists:
                    writer.writeheader()
                    history_exists = True
                writer.writerow(validation)
            print(
                f"Epoch {epoch + 1}/{epochs}: train={validation['train_loss']:.4f} "
                f"val={current:.4f} hazard FNR={validation['hazard_fnr']:.4f} "
                f"landing IoU={validation['landing_iou']:.4f} "
                f"semantic mIoU={validation['semantic_mean_iou']:.4f} "
                f"POI macro-F1={validation['poi_macro_f1']:.4f} "
                f"plateau={bad_epochs}/{patience} "
                f"train={train_epoch_seconds:.2f}s val={validation_seconds:.2f}s "
                f"throughput={validation['train_samples_per_second']:.1f} samples/s",
                flush=True,
            )
            if epoch + 1 >= min_epochs and patience > 0 and bad_epochs >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1}: validation loss plateaued.", flush=True
                )
                break

        record = {
            "complete": True,
            "epochs": epoch + 1 if epochs > start_epoch else start_epoch,
            "best_validation_loss": best_metric,
            "config_hash": config_hash,
            "manifest_hash": file_sha256(manifest),
            "training_plan_hash": config.get("experiment", {}).get("training_plan_hash"),
            "analytical_boundary": config.get("experiment", {}).get("analytical_boundary"),
        }
        save_json(output_dir / "complete.json", record)
        return best_path if best_path.exists() else last_path
    finally:
        for handled_signal, previous_handler in previous_handlers.items():
            signal.signal(handled_signal, previous_handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    print(train(arguments.config, force=arguments.force))


if __name__ == "__main__":
    main()
