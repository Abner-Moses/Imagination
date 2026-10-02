"""Shared configuration, reproducibility, checkpoint, and resource helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from common.registry import (
    ANALYTICAL_CHANNELS,
    CHANNEL_REGISTRY_VERSION,
    CANDIDATE_FEATURE_VERSION,
    STATE_DIM,
    STATE_NAMES as DEFAULT_STATE_NAMES,
    CHECKPOINT_SCHEMA_VERSION,
    FUSION_CONTRACT_VERSION,
    MAP_SCHEMA_VERSION,
    METRIC_ATTENTION_VERSION,
    MODEL_ARCHITECTURE_VERSION,
    SEMANTIC_FUSION_VERSION,
)

try:  # `resource` does not exist on Windows.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised by Windows CI
    _resource = None


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a recursive configuration overlay without mutating either input."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load YAML with optional relative ``extends`` overlays.

    Multiple bases are merged in order, then the current file overrides them.
    This keeps ablation configs small while making them usable by the existing
    runner. Paths are resolved relative to the containing YAML file.
    """
    source = Path(path).resolve()
    if source in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, source))
        raise ValueError(f"Configuration inheritance cycle: {chain}")
    with source.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {source}")
    bases = config.pop("extends", ())
    if isinstance(bases, (str, Path)):
        bases = (bases,)
    if not isinstance(bases, (list, tuple)):
        raise ValueError("Configuration 'extends' must be a path or list of paths")
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = source.parent / base_path
        merged = merge_dicts(merged, load_config(base_path, (*_stack, source)))
    return merge_dicts(merged, config)


def save_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_hash(root: str | Path = ".") -> str:
    """Hash research source/config files while excluding data and generated artifacts."""
    repository = Path(root).resolve()
    excluded = {
        ".git",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "artifacts",
        "cache",
        "dataset",
        ".venv",
        ".runtime",
    }
    suffixes = {".py", ".yaml", ".yml", ".json", ".cpp", ".hpp", ".toml", ".txt"}
    names = {"CMakeLists.txt", "CMakePresets.json"}
    digest = hashlib.sha256()
    for path in sorted(repository.rglob("*")):
        relative = path.relative_to(repository)
        if not path.is_file() or any(
            part in excluded or part.startswith(".venv-") for part in relative.parts
        ):
            continue
        if path.suffix not in suffixes and path.name not in names:
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
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
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "numpy": np.__version__,
        "device": str(device),
        "platform": platform.platform(),
    }


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore saved generators after model/data setup and before the next batch."""
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


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
    *,
    scaler: Any = None,
    batch_in_epoch: int = 0,
    best_metric: float | None = None,
    analytical_cache_hash: str | None = None,
    map_policy_hash: str | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "config_hash": stable_hash(config),
        "metrics": metrics,
        "best_validation_metric": best_metric,
        "batch_in_epoch": batch_in_epoch,
        "amp_scaler": scaler.state_dict() if scaler is not None else None,
        "rng_state": _rng_state(),
        "channel_registry_version": CHANNEL_REGISTRY_VERSION,
        "architecture_contract": {
            "model": MODEL_ARCHITECTURE_VERSION,
            "map": MAP_SCHEMA_VERSION,
            "fusion": FUSION_CONTRACT_VERSION,
            "semantic_fusion": SEMANTIC_FUSION_VERSION,
            "metric_attention": METRIC_ATTENTION_VERSION,
            "candidate_features": CANDIDATE_FEATURE_VERSION,
        },
        "channel_names": list(ANALYTICAL_CHANNELS),
        "manifest_hash": manifest_hash,
        "analytical_cache_hash": analytical_cache_hash,
        "map_policy_hash": map_policy_hash,
        "git_commit": git_commit(),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    try:
        descriptor = os.open(destination.parent, os.O_DIRECTORY)
        os.fsync(descriptor)
        os.close(descriptor)
    except (AttributeError, OSError):
        pass


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("checkpoint_schema", CHECKPOINT_SCHEMA_VERSION) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema {checkpoint.get('checkpoint_schema')!r} is incompatible; "
            f"expected {CHECKPOINT_SCHEMA_VERSION!r}."
        )
    if checkpoint.get("channel_registry_version") != CHANNEL_REGISTRY_VERSION:
        raise ValueError("Checkpoint analytical channel registry is incompatible")
    expected_contract = {
        "model": MODEL_ARCHITECTURE_VERSION,
        "map": MAP_SCHEMA_VERSION,
        "fusion": FUSION_CONTRACT_VERSION,
        "semantic_fusion": SEMANTIC_FUSION_VERSION,
        "metric_attention": METRIC_ATTENTION_VERSION,
        "candidate_features": CANDIDATE_FEATURE_VERSION,
    }
    if checkpoint.get("architecture_contract") != expected_contract:
        raise ValueError(
            "Checkpoint architecture/map contract is incompatible with the current "
            "IMF revision. Start a new run or use a matching older checkout; do "
            "not resume across fusion, map-schema, evidence, or attention changes."
        )
    model.load_state_dict(checkpoint["model"])
    if optimizer and checkpoint.get("optimizer"):
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler and checkpoint.get("scheduler"):
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint.get("amp_scaler"):
        scaler.load_state_dict(checkpoint["amp_scaler"])
    _restore_rng_state(checkpoint.get("rng_state"))
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
        # Count the actually executed gathered/dense pairs. Q/K and MBConv
        # projections are counted through their registered Conv/Linear modules.
        stats = getattr(module, "last_attention_stats", {})
        pairs = stats.get("attention_pairs", batch * module.heads * tokens * tokens)
        total += 2 * pairs * module.head_dim

    def vit_attention_hook(module, arguments, output):
        nonlocal total
        tokens = arguments[0].shape[2] * arguments[0].shape[3]
        total += 2 * arguments[0].shape[0] * module.heads * tokens * tokens * module.head_dim

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
            handles.append(module.register_forward_hook(module_hook))
        if module.__class__.__name__ == "HTransformerBlock":
            handles.append(module.register_forward_hook(attention_hook))
        elif module.__class__.__name__ == "ConventionalViTBlock":
            handles.append(module.register_forward_hook(vit_attention_hook))
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
    inputs = tuple(
        value.to(device) if isinstance(value, torch.Tensor) else value for value in inputs
    )
    for _ in range(warmup):
        model(*inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    memory_samples = []
    for _ in range(repetitions):
        start = time.perf_counter()
        model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
        if device.type == "cuda":
            memory_samples.append(torch.cuda.max_memory_allocated(device))
        elif device.type == "mps":
            memory_samples.append(torch.mps.current_allocated_memory())
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    rss = None
    if _resource is not None:
        rss = int(
            _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024)
        )
    macs = estimate_macs(model, inputs)
    mean = float(np.mean(samples))
    return {
        "parameters": parameter_count(model),
        "serialized_state_bytes": stream.tell(),
        "estimated_macs_per_sample": macs,
        "estimated_flops_per_sample": 2 * macs,
        "flop_convention": "2 FLOPs per multiply-accumulate; nonlinear/library ops omitted",
        "latency_mean_ms": mean,
        "latency_p50_ms": float(np.percentile(samples, 50)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "latency_std_ms": float(np.std(samples)),
        "fps": 1000.0 / mean if mean else None,
        "peak_inference_memory_bytes": int(max(memory_samples)) if memory_samples else None,
        "process_peak_rss_bytes": rss,
        "energy_joules_per_frame": None,
        "energy_status": "NOT_MEASURED",
    }
