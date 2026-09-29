"""Shared configuration, reproducibility, checkpoint, and resource helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ANALYTICAL_CHANNELS = (
    "Y", "Cb", "Cr", "Gx", "Gy", "GradientMagnitude",
    "HOG_0", "HOG_1", "HOG_2", "HOG_3", "HOG_4", "HOG_5", "HOG_6",
    "HOG_7", "HOG_8", "HarrisResponse", "CannyEdge", "ContourMap",
    "ChromaGradientCb", "ChromaGradientCr", "OpticalFlowU", "OpticalFlowV",
    "Depth", "DepthGradientX", "DepthGradientY", "Slope", "Roughness",
    "GeometryConfidence",
)
CHANNEL_REGISTRY_VERSION = "imagination-analytical-v1"
DEFAULT_STATE_NAMES = (
    "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
    "gyro_x_radps", "gyro_y_radps", "gyro_z_radps",
    "roll_rad", "pitch_rad", "yaw_rad", "ultrasonic_m",
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def save_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def git_commit(repository: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "platform": platform.platform(),
    }


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    manifest_hash: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "metrics": metrics,
        "channel_registry_version": CHANNEL_REGISTRY_VERSION,
        "channel_names": list(ANALYTICAL_CHANNELS),
        "manifest_hash": manifest_hash,
        "git_commit": git_commit(),
    }, destination)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("channel_registry_version") != CHANNEL_REGISTRY_VERSION:
        raise ValueError("Checkpoint analytical channel registry is incompatible")
    model.load_state_dict(checkpoint["model"])
    if optimizer and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


@torch.inference_mode()
def estimate_macs(model: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> int:
    """Count Conv/Linear and explicit attention multiply-accumulates."""
    total = 0
    handles = []

    def module_hook(module, arguments, output):
        nonlocal total
        result = output[0] if isinstance(output, tuple) else output
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            total += result.numel() * (module.in_channels // module.groups) * kernel
        elif isinstance(module, torch.nn.Linear):
            total += result.numel() * module.in_features

    def attention_hook(module, arguments, output):
        nonlocal total
        batch, _, height, width = arguments[0].shape
        tokens = height * width
        # QK^T and attention-times-V; projections are counted as Linear modules.
        total += 2 * batch * module.heads * tokens * tokens * module.head_dim

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(module_hook))
        if module.__class__.__name__ == "HTransformerBlock":
            handles.append(module.register_forward_hook(attention_hook))
    try:
        model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    return int(total / inputs[0].shape[0])


@torch.inference_mode()
def benchmark_model(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    device: torch.device,
    warmup: int = 5,
    repetitions: int = 20,
) -> dict[str, float | int]:
    model.eval()
    inputs = tuple(value.to(device) for value in inputs)
    for _ in range(warmup):
        model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "parameters": parameter_count(model),
        "estimated_macs_per_sample": estimate_macs(model, inputs),
        "latency_mean_ms": float(np.mean(samples)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "peak_inference_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else 0,
    }
