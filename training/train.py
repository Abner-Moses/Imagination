"""Config-driven training with deterministic splits, resume, and checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from training.common import (
    environment_record,
    file_sha256,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_json,
    select_device,
    set_seed,
)
from training.data import ImaginationDataset, validate_manifest
from training.losses import multitask_loss
from training.metrics import HazardCounts, NavigationMetrics
from training.models import build_model


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_forward(model: torch.nn.Module, batch: dict[str, Any], model_type: str):
    state = batch["vehicle_state"]
    state_validity = batch["state_validity"]
    if model_type == "analytical":
        return model(batch["analytical"], batch["validity"], state, state_validity)
    return model(batch["rgb"], state, state_validity)


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
    loss_config: dict,
) -> dict[str, float]:
    model.eval()
    counts = HazardCounts()
    navigation = NavigationMetrics()
    total_loss = 0.0
    samples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model_forward(model, batch, model_type)
        losses = multitask_loss(outputs, batch, loss_config)
        batch_size = batch["waypoint_target"].shape[0]
        total_loss += float(losses["total"]) * batch_size
        samples += batch_size
        counts.update(outputs["hazard_logits"], batch["hazard_target"],
                      batch["hazard_validity"])
        navigation.update(outputs["waypoint"], batch["waypoint_target"])
    return {"loss": total_loss / max(samples, 1), **counts.result(), **navigation.result()}


def train(config_path: str | Path, resume: str | Path | None = None) -> Path:
    config = load_config(config_path)
    training = config["training"]
    model_type = config["model"]["type"]
    seed = int(training.get("seed", 7))
    set_seed(seed, bool(training.get("deterministic", True)))
    device = select_device(str(training.get("device", "auto")))
    manifest = Path(config["data"]["manifest"])
    validate_manifest(manifest, check_files=bool(config["data"].get("validate_files", True)))
    mode = "analytical" if model_type == "analytical" else "baseline"
    train_data = ImaginationDataset(manifest, "train", mode)
    val_data = ImaginationDataset(manifest, "val", mode)
    loader_options = {
        "batch_size": int(training.get("batch_size", 8)),
        "num_workers": int(training.get("workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    epochs = int(training.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = global_step = 0
    if resume:
        restored = load_checkpoint(resume, model, optimizer, scheduler, device)
        start_epoch = int(restored["epoch"]) + 1
        global_step = int(restored["global_step"])

    output = Path(training.get("output", "training/runs/default"))
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "run_environment.json", {
        **environment_record(device), "seed": seed, "config": config,
        "manifest_sha256": file_sha256(manifest),
    })
    log_path = output / "metrics.csv"
    best_loss = float("inf")
    stale_epochs = 0
    patience = int(training.get("early_stopping_patience", 0))
    with log_path.open("a" if resume else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "epoch", "train_loss", "val_loss", "hazard_fnr", "hazard_recall",
            "hazard_precision", "hazard_iou", "waypoint_translation_mae_m",
            "waypoint_heading_mae_rad", "learning_rate",
        ])
        if not resume:
            writer.writeheader()
        for epoch in range(start_epoch, epochs):
            model.train()
            total = 0.0
            seen = 0
            for batch in train_loader:
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model_forward(model, batch, model_type)
                    losses = multitask_loss(outputs, batch, config["losses"])
                scaler.scale(losses["total"]).backward()
                clip = float(training.get("gradient_clip", 0))
                if clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                batch_size = batch["waypoint_target"].shape[0]
                total += float(losses["total"].detach()) * batch_size
                seen += batch_size
                global_step += 1
            scheduler.step()
            metrics = validate(model, val_loader, device, model_type, config["losses"])
            row = {
                "epoch": epoch,
                "train_loss": total / max(seen, 1),
                "val_loss": metrics["loss"],
                "hazard_fnr": metrics["hazard_fnr"],
                "hazard_recall": metrics["hazard_recall"],
                "hazard_precision": metrics["hazard_precision"],
                "hazard_iou": metrics["hazard_iou"],
                "waypoint_translation_mae_m": metrics["waypoint_translation_mae_m"],
                "waypoint_heading_mae_rad": metrics["waypoint_heading_mae_rad"],
                "learning_rate": scheduler.get_last_lr()[0],
            }
            writer.writerow(row)
            stream.flush()
            save_checkpoint(
                output / "last.pt", model, optimizer, scheduler, epoch, global_step,
                config, metrics, file_sha256(manifest),
            )
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                stale_epochs = 0
                save_checkpoint(
                    output / "best.pt", model, optimizer, scheduler, epoch, global_step,
                    config, metrics, file_sha256(manifest),
                )
            else:
                stale_epochs += 1
            print(f"epoch={epoch} train={row['train_loss']:.5f} val={metrics['loss']:.5f} "
                  f"fnr={metrics['hazard_fnr']:.5f}")
            if patience and stale_epochs >= patience:
                break
    return output / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    arguments = parser.parse_args()
    print(train(arguments.config, arguments.resume))


if __name__ == "__main__":
    main()
