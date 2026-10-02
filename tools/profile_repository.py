"""Profile representative data, model, and causal-map paths on development data."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common.mapping import build_predicted_contexts
from common.models import build_model
from common.runner import family_config
from common.runtime import parameter_count, save_json, select_device, set_seed
from common.training.engine import forward_model, model_mode, move_targets
from common.training.losses import multitask_loss
from data.adapter import ImaginationDataset


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("cnn_htransformer", "cnn_vit", "cnn", "imf_htransformer")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)


def _data_profile(manifest: Path, samples: int) -> dict:
    report = {}
    for name, mode in (("rgb", "rgb_current"), ("analytical", "analytical")):
        started = time.perf_counter()
        dataset = ImaginationDataset(manifest, "train", mode)
        initialization = time.perf_counter() - started
        started = time.perf_counter()
        for index in range(min(samples, len(dataset))):
            dataset[index]
        elapsed = time.perf_counter() - started
        report[name] = {
            "dataset_initialization_ms": initialization * 1000,
            "samples": min(samples, len(dataset)),
            "sample_loading_seconds": elapsed,
            "samples_per_second": min(samples, len(dataset)) / max(elapsed, 1e-12),
        }
    return report


def _loss_config(config: dict) -> dict:
    result = dict(config["losses"])
    for name in ("hazard_positive_weight", "landing_positive_weight", "poi_positive_weight"):
        if result.get(name) == "auto":
            result[name] = 1.0
    return result


def _model_profile(manifest: Path, family: str, device: torch.device, repetitions: int) -> dict:
    set_seed(7, True)
    config = family_config(family)
    config["model"].update(family=family, profile="research")
    dataset = ImaginationDataset(manifest, "train", model_mode(family))
    batch = next(iter(DataLoader(Subset(dataset, range(2)), batch_size=2, num_workers=0)))
    move_targets(batch, device)
    model = build_model(config).to(device)
    losses_config = _loss_config(config)

    model.eval()
    with torch.inference_mode():
        forward_model(model, batch, family, device)
        forward_times = []
        for _ in range(repetitions):
            _sync(device)
            started = time.perf_counter()
            forward_model(model, batch, family, device)
            _sync(device)
            forward_times.append(time.perf_counter() - started)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    training_times = []
    for _ in range(repetitions):
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        started = time.perf_counter()
        outputs = forward_model(model, batch, family, device)
        losses = multitask_loss(outputs, batch, losses_config)
        losses["total"].backward()
        optimizer.step()
        _sync(device)
        training_times.append(time.perf_counter() - started)

    visual = ImaginationDataset(manifest, "train", model_mode(family), include_targets=False)
    geometry = (
        visual
        if family == "imf_htransformer"
        else ImaginationDataset(manifest, "train", "analytical", include_targets=False)
    )
    map_frames = 16
    started = time.perf_counter()
    build_predicted_contexts(
        model,
        family,
        visual,
        geometry,
        device,
        ROOT / "data/configs/semantic_rules.yaml",
        max_samples=map_frames,
    )
    _sync(device)
    map_seconds = time.perf_counter() - started
    result = {
        "parameters": parameter_count(model),
        "forward_batch_size": 2,
        "forward_p50_ms": float(np.median(forward_times) * 1000),
        "forward_p95_ms": float(np.percentile(forward_times, 95) * 1000),
        "training_step_p50_ms": float(np.median(training_times) * 1000),
        "training_step_p95_ms": float(np.percentile(training_times, 95) * 1000),
        "map_frames": map_frames,
        "map_context_ms_per_frame": map_seconds / map_frames * 1000,
        "peak_process_rss_bytes": _rss_bytes(),
    }
    del model, batch, dataset, visual, geometry
    gc.collect()
    return result


def profile(
    output: Path, samples: int = 64, repetitions: int = 5, device_name: str = "auto"
) -> dict:
    manifest = ROOT / "data/manifests/all.jsonl"
    device = select_device(device_name)
    result = {
        "schema": "imagination-repository-profile-v1",
        "hardware": {
            "platform": platform.platform(),
            "device": str(device),
            "torch": str(torch.__version__),
        },
        "workload": {"data_samples": samples, "model_repetitions": repetitions, "batch_size": 2},
        "data": _data_profile(manifest, samples),
        "models": {
            family: _model_profile(manifest, family, device, repetitions) for family in FAMILIES
        },
    }
    save_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    print(json.dumps(profile(args.output, args.samples, args.repetitions, args.device), indent=2))


if __name__ == "__main__":
    main()
