"""Measure learned-model inference and summarize C++ timing CSV exports."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from common.models import build_model
from common.registry import CANDIDATE_FEATURE_NAMES, MAX_CANDIDATES, RELATIONAL_NAMES, STATE_DIM
from common.runtime import benchmark_model, load_config, parameter_count, save_json, select_device
from data.adapter import ImaginationDataset
from common.training.engine import model_mode


def benchmark_family(
    config_path: str | Path, manifest: str | Path | None, device_name: str = "auto"
) -> dict:
    config = load_config(config_path)
    family = config["model"]["family"]
    device = select_device(device_name)
    model = build_model(config).to(device).eval()
    if manifest:
        dataset = ImaginationDataset(
            manifest, "val", model_mode(family), config.get("poi", {}).get("classes")
        )
        sample = dataset[0]
        state = sample["vehicle_state"].unsqueeze(0)
        state_validity = sample["state_validity"].unsqueeze(0)
        relation = sample["relational"].unsqueeze(0)
        relation_validity = sample["relational_validity"].unsqueeze(0)
        candidates = sample["candidates"].unsqueeze(0)
        candidate_validity = sample["candidate_validity"].unsqueeze(0)
        candidate_feature_validity = sample["candidate_feature_validity"].unsqueeze(0)
        if family == "imf_htransformer":
            inputs = (
                sample["analytical"].unsqueeze(0),
                sample["validity"].unsqueeze(0),
                state,
                state_validity,
                relation,
                relation_validity,
                candidates,
                candidate_validity,
                candidate_feature_validity,
            )
        else:
            inputs = (
                sample["rgb"].unsqueeze(0),
                state,
                state_validity,
                relation,
                relation_validity,
                candidates,
                candidate_validity,
                candidate_feature_validity,
            )
    else:
        state = torch.zeros(1, STATE_DIM)
        state_validity = torch.zeros_like(state)
        relation = torch.zeros(1, len(RELATIONAL_NAMES))
        relation_validity = torch.zeros_like(relation)
        candidates = torch.zeros(1, MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES))
        candidate_validity = torch.zeros(1, MAX_CANDIDATES)
        candidate_feature_validity = torch.zeros_like(candidates)
        if family == "imf_htransformer":
            inputs = (
                torch.zeros(1, 28, 32, 32),
                torch.zeros(1, 28, 32, 32),
                state,
                state_validity,
                relation,
                relation_validity,
                candidates,
                candidate_validity,
                candidate_feature_validity,
            )
        else:
            inputs = (
                torch.zeros(1, 3, 256, 256),
                state,
                state_validity,
                relation,
                relation_validity,
                candidates,
                candidate_validity,
                candidate_feature_validity,
            )

    measured = benchmark_model(model, inputs, device)
    return {
        "family": family,
        "profile": config["model"].get("profile", "research"),
        "device": str(device),
        "parameters": parameter_count(model),
        "input_source": "validation sample" if manifest else "zero-valued shape probe",
        **measured,
        "pre_extraction_latency_ms": None,
        "end_to_end_latency_ms": None,
        "average_power_w": None,
        "energy_joules_per_frame": None,
        "energy_status": "NOT_MEASURED",
    }


def summarize_preextraction(path: str | Path) -> dict:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Pre-extraction timing CSV is empty")
    # C++ writes one row per stage/frame. Preserve total time once per frame.
    frame_times: dict[int, float] = {}
    stage_times: dict[str, list[float]] = {}
    for row in rows:
        frame = int(row["frame"])
        frame_times[frame] = float(row["total_ms"])
        stage_times.setdefault(row["stage"], []).append(
            float(row["stage_ms"]) if row["ran"].lower() in {"1", "true"} else 0.0
        )
    values = list(frame_times.values())
    return {
        "sample_count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": float(np.percentile(values, 95)),
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "stage_mean_ms": {
            name: statistics.fmean(samples) for name, samples in sorted(stage_times.items())
        },
        "throughput_fps": 1000.0 / statistics.fmean(values) if statistics.fmean(values) else None,
        "timing_file": str(path),
        "includes_filesystem_io": False,
    }


def integrate_power(path: str | Path, frame_count: int) -> dict:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2 or frame_count < 1:
        raise ValueError(
            "Power input requires at least two timestamp_s,power_w samples and a positive frame count"
        )
    times = np.asarray([float(row["timestamp_s"]) for row in rows])
    watts = np.asarray([float(row["power_w"]) for row in rows])
    if not np.isfinite(times).all() or not np.isfinite(watts).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Power CSV timestamps must increase and readings must be finite")
    joules = float(np.trapezoid(watts, times))
    duration = float(times[-1] - times[0])
    return {
        "average_power_w": joules / duration,
        "energy_joules_per_frame": joules / frame_count,
        "frame_count": frame_count,
        "source": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="One model YAML configuration")
    parser.add_argument("--manifest", help="Optional manifest; benchmarks one validation sample")
    parser.add_argument("--preextraction-csv", help="C++ pre_extract timings.csv")
    parser.add_argument("--power-csv", help="Hardware log with timestamp_s,power_w")
    parser.add_argument("--frames", type=int, help="Frames represented in --power-csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not any((args.config, args.preextraction_csv, args.power_csv)):
        parser.error("Choose a model config, pre-extraction timing CSV, or power CSV")
    results = {}
    if args.config:
        results["model"] = benchmark_family(args.config, args.manifest, args.device)
    if args.preextraction_csv:
        results["pre_extraction"] = summarize_preextraction(args.preextraction_csv)
    if args.power_csv:
        if args.frames is None:
            parser.error("--power-csv requires --frames")
        results["measured_power"] = integrate_power(args.power_csv, args.frames)
    else:
        results["measured_power"] = {
            "average_power_w": None,
            "energy_joules_per_frame": None,
            "status": "NOT_MEASURED",
        }
    save_json(args.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
