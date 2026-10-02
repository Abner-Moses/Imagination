"""Development-only IMF HTransformer resource matrix using synthetic tensors.

No manifest, dataset, validation data, or held-out test data is opened. Results
measure implementation cost and must not be interpreted as task quality.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

try:
    import psutil
except ImportError:  # memory sampling is an optional diagnostic
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
from common.models import build_model
from common.registry import CANDIDATE_FEATURE_NAMES, MAX_CANDIDATES, RELATIONAL_NAMES, STATE_DIM
from common.runtime import benchmark_model, load_config, save_json


CONFIGS = {
    "reference_sparse_sparse_full_qk_context_both": "attention_policy_sparse_sparse.yaml",
    "A_sparse_dense_context_both": "attention_policy_sparse_dense.yaml",
    "B_sparse_dense_context_stage4": "attention_policy_stage4_context.yaml",
    "C_B_reduced_qk_medium": "attention_qk_medium.yaml",
    "D_C_reduced_ffn": "attention_ffn_reduced.yaml",
    "E_small_qk_reduced_heads": "attention_qk_small.yaml",
    "F_optional_active_queries": "active_query_experimental.yaml",
    "policy_dense_dense": "attention_policy_dense_dense.yaml",
    "policy_dense_sparse": "attention_policy_dense_sparse.yaml",
}


def _synthetic_inputs(model, device: torch.device):
    batch = 1
    analytical = torch.rand(batch, 28, 32, 32)
    validity = torch.ones_like(analytical)
    # Keep geometry physically bounded and provide enough support to exercise
    # metric selection without loading any cache or simulator target.
    analytical[:, 22] = 0.4 + 0.5 * analytical[:, 22]
    analytical[:, 27] = 0.8
    analytical[:, 16] = (torch.rand_like(analytical[:, 16]) < 0.05).to(analytical.dtype)
    analytical[:, 20:22] = torch.rand_like(analytical[:, 20:22]) * 0.05
    state = torch.zeros(batch, STATE_DIM)
    state_validity = torch.ones_like(state)
    relations = torch.zeros(batch, len(RELATIONAL_NAMES))
    relation_validity = torch.ones_like(relations)
    candidates = torch.zeros(batch, MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES))
    candidate_validity = torch.zeros(batch, MAX_CANDIDATES)
    candidate_feature_validity = torch.zeros_like(candidates)
    candidate_grid = torch.full((batch, MAX_CANDIDATES, 2), 16.0)
    projected_depth = torch.full((batch, MAX_CANDIDATES), 2.0)
    projection_validity = torch.zeros(batch, MAX_CANDIDATES)
    visibility = torch.zeros(batch, MAX_CANDIDATES, dtype=torch.long)
    candidate_validity[:, :8] = 1.0
    candidate_feature_validity[:, :8] = 1.0
    projection_validity[:, :8] = 1.0
    visibility[:, :8] = 1  # deterministic visible state
    calibration = torch.tensor([[32.0, 32.0, 20.0, 20.0, 15.5, 15.5]])
    depth_scale = torch.tensor([10.0])
    values = (
        analytical,
        validity,
        state,
        state_validity,
        relations,
        relation_validity,
        candidates,
        candidate_validity,
        candidate_feature_validity,
        calibration,
        depth_scale,
        None,
        None,
        candidate_grid,
        projected_depth,
        projection_validity,
        visibility,
    )
    return tuple(value.to(device) if isinstance(value, torch.Tensor) else value for value in values)


def _timing_summary(blocks) -> dict:
    stage = {}
    for stage_name, sequential in (("stage3", blocks.stage3), ("stage4", blocks.stage4)):
        keys = (
            "q_projection_ms",
            "k_projection_ms",
            "mbconv_value_ms",
            "query_selection_ms",
            "neighbor_selection_ms",
            "sparse_gather_ms",
            "context_projection_ms",
            "attention_ms",
            "context_attention_ms",
            "ffn_ms",
            "attention_pairs",
            "active_queries",
            "imf_dissimilarity_pairs",
            "dense_reference_pairs",
            "metric_neighbor_distance_comparisons",
        )
        stage[stage_name] = {
            key: sum(float(block.last_attention_stats.get(key, 0)) for block in sequential)
            for key in keys
        }
    return stage


class _RssSampler:
    """Sample this process RSS during one model run when psutil is available."""

    def __init__(self):
        self.process = psutil.Process() if psutil is not None else None
        self.baseline = self.process.memory_info().rss if self.process else None
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = None

    def _sample(self):
        while not self.stop_event.wait(0.002):
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except (psutil.Error, AttributeError):
                return

    def __enter__(self):
        if self.process:
            self.thread = threading.Thread(target=self._sample, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)
            self.peak = max(self.peak, self.process.memory_info().rss)

    @property
    def delta(self):
        return None if self.baseline is None else max(0, self.peak - self.baseline)


def _run_one(label, filename, device_name="cpu", warmup=5, repetitions=20):
    device = torch.device(device_name)
    config = load_config(ROOT / "IMF_HTransformer" / "configs" / filename)
    config["model"].setdefault("attention", {})["profile_timing"] = False
    model = build_model(config).to(device).eval()
    inputs = _synthetic_inputs(model, device)
    with _RssSampler() as rss_sampler:
        measured = benchmark_model(model, inputs, device, warmup=warmup, repetitions=repetitions)
    # Component timing is collected in one separate diagnostic forward pass
    # so per-operation instrumentation does not contaminate latency percentiles.
    for block in (*model.backbone.stage3, *model.backbone.stage4):
        block.profile_timing = True
    adapter_times = {}
    hook_handles = []
    for name, adapter in model.family_adapters.items():
        state = {"start": 0.0, "ms": 0.0}
        adapter_times[name] = state
        hook_handles.append(
            adapter.register_forward_pre_hook(
                lambda _module, _args, state=state: state.update(start=time.perf_counter())
            )
        )
        hook_handles.append(
            adapter.register_forward_hook(
                lambda _module, _args, _out, state=state: state.update(
                    ms=(time.perf_counter() - state["start"]) * 1000.0
                )
            )
        )
    decoder_time = {"start": 0.0, "ms": 0.0}
    decoder = model.heads.decoder
    hook_handles.append(
        decoder.register_forward_pre_hook(
            lambda _module, _args: decoder_time.update(start=time.perf_counter())
        )
    )
    hook_handles.append(
        decoder.register_forward_hook(
            lambda _module, _args, _out: decoder_time.update(
                ms=(time.perf_counter() - decoder_time["start"]) * 1000.0
            )
        )
    )
    with torch.inference_mode():
        model(*inputs)
    for handle in hook_handles:
        handle.remove()
    stage_stats = _timing_summary(model.backbone)
    widths = model.backbone.stage3[0].channels
    width4 = model.backbone.stage4[0].channels
    qk3 = model.backbone.stage3[0].qk_channels
    qk4 = model.backbone.stage4[0].qk_channels
    active3 = int(stage_stats["stage3"]["active_queries"] or 256)
    active4 = int(stage_stats["stage4"]["active_queries"] or 64)
    stage_stats["stage3"]["q_projection_macs"] = active3 * widths * qk3
    stage_stats["stage3"]["k_projection_macs"] = 256 * widths * qk3
    stage_stats["stage4"]["q_projection_macs"] = active4 * width4 * qk4
    stage_stats["stage4"]["k_projection_macs"] = 64 * width4 * qk4
    return {
        "configuration": label,
        "config_file": filename,
        "parameters": measured["parameters"],
        "estimated_macs_per_sample": measured["estimated_macs_per_sample"],
        "estimated_flops_per_sample": measured["estimated_flops_per_sample"],
        "serialized_state_bytes": measured["serialized_state_bytes"],
        "latency_p50_ms": measured["latency_p50_ms"],
        "latency_p95_ms": measured["latency_p95_ms"],
        "latency_mean_ms": measured["latency_mean_ms"],
        "process_peak_rss_bytes": measured["process_peak_rss_bytes"],
        "inference_rss_peak_delta_bytes": rss_sampler.delta,
        "device_memory_bytes": measured["peak_inference_memory_bytes"],
        "family_adapter_profile_forward_ms": {
            name: state["ms"] for name, state in adapter_times.items()
        },
        "decoder_profile_forward_ms": decoder_time["ms"],
        "stages": stage_stats,
    }


def run_matrix(device_name="cpu", warmup=5, repetitions=20):
    results = []
    for label, filename in CONFIGS.items():
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            label,
            "--device",
            device_name,
            "--warmup",
            str(warmup),
            "--repetitions",
            str(repetitions),
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(
                f"Resource probe {label} failed ({completed.returncode}):\n{completed.stderr}"
            )
        marker = "IMAGINATION_BENCHMARK_RESULT="
        lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
        if not lines:
            raise RuntimeError(f"Resource probe {label} returned no result JSON")
        results.append(json.loads(lines[-1][len(marker) :]))
    return {
        "schema": "imagination-attention-resource-matrix-v1",
        "purpose": "synthetic engineering benchmark only; no task-quality or test-split measurements",
        "device": str(device_name),
        "warmup": warmup,
        "repetitions": repetitions,
        "models": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/benchmarks/imf_attention_matrix.json"
    )
    parser.add_argument("--single", choices=tuple(CONFIGS), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 1 or args.repetitions < 2:
        parser.error("warmup must be positive and repetitions at least two")
    if args.single:
        print(
            "IMAGINATION_BENCHMARK_RESULT="
            + json.dumps(
                _run_one(
                    args.single, CONFIGS[args.single], args.device, args.warmup, args.repetitions
                )
            )
        )
        return
    result = run_matrix(args.device, args.warmup, args.repetitions)
    save_json(args.output, result)
    print(json.dumps(result, indent=2))
    print(f"Saved development-only matrix to {args.output}")


if __name__ == "__main__":
    main()
