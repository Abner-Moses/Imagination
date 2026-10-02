"""Run a trained model through one ordered episode with predicted-map feedback.

This diagnostic never opens simulator label masks. The current trainer is still
frame-based, so its output must not be reported as evidence of learned map use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common.mapping.online import OnlinePerceptionMap
from common.models import build_model
from common.registry import POI_CLASSES
from common.runtime import load_checkpoint, load_config, save_json, select_device
from common.training.engine import model_mode
from data.adapter import ImaginationDataset, load_manifest


def run_episode(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    episode_id: str,
    output_path: str | Path,
    device_name: str = "auto",
    attention_debug_output: str | Path | None = None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    family = config["model"]["family"]
    device = select_device(device_name)
    model = build_model(config).to(device).eval()
    load_checkpoint(checkpoint_path, model, map_location=device)
    if attention_debug_output is not None:
        from common.models.htransformer import HTransformerBlock

        for module in model.modules():
            if isinstance(module, HTransformerBlock):
                module.capture_attention = True

    manifest = Path(manifest_path).resolve()
    records = [row for row in load_manifest(manifest) if row["episode_id"] == episode_id]
    records.sort(key=lambda row: row["frame_index"])
    if not records:
        raise ValueError(f"Episode {episode_id!r} is absent from {manifest}")
    split = records[0]["split"]
    visual_data = ImaginationDataset(
        manifest,
        split,
        model_mode(family),
        config.get("poi", {}).get("classes", POI_CLASSES),
        include_targets=False,
    )
    geometry_data = ImaginationDataset(manifest, split, "analytical", include_targets=False)
    by_sample = {row["sample_id"]: index for index, row in enumerate(visual_data.records)}
    semantic_config = (
        Path(__file__).resolve().parents[1] / "data" / "configs" / "semantic_rules.yaml"
    )
    stream = OnlinePerceptionMap.from_config(semantic_config)
    frame_results = []

    with torch.inference_mode():
        for record in records:
            index = by_sample[record["sample_id"]]
            visual = visual_data[index]
            geometry = geometry_data[index]
            state = visual["vehicle_state"].unsqueeze(0).to(device)
            state_validity = visual["state_validity"].unsqueeze(0).to(device)
            analytical = geometry["analytical"].unsqueeze(0)
            geometry_validity = geometry["validity"].unsqueeze(0)
            pose = geometry["camera_to_local_map"].numpy()
            pose_valid = bool(geometry["camera_pose_valid"])
            calibration_values = visual["calibration"].tolist()
            calibration = dict(zip(("width", "height", "fx", "fy", "cx", "cy"), calibration_values))
            confidence_mask = geometry_validity[0, 27].numpy() > 0
            confidence = (
                float(analytical[0, 27].numpy()[confidence_mask].mean())
                if confidence_mask.any()
                else None
            )

            result = stream.process(
                model,
                family=family,
                visual_input=analytical
                if family == "imf_htransformer"
                else visual["rgb"].unsqueeze(0),
                validity=geometry_validity if family == "imf_htransformer" else None,
                vehicle_state=state,
                state_validity=state_validity,
                timestamp_s=float(record["timestamp"]),
                intrinsics=calibration,
                camera_to_local_map=pose if pose_valid else None,
                depth_normalized=analytical[0, 22].numpy(),
                depth_validity=geometry_validity[0, 22].numpy(),
                depth_scale_m=float(geometry["depth_scale_m"]),
                device=device,
                geometry_confidence=confidence,
            )
            frame_results.append(
                {
                    "sample_id": record["sample_id"],
                    "timestamp": record["timestamp"],
                    "camera_pose_valid": pose_valid,
                    "mapped_semantic_points": result.mapped_semantic_points,
                    "resolved_regions": result.resolved_regions,
                    "unresolved_regions": result.unresolved_regions,
                    "map_entities": result.map_size,
                    "relational_values": result.relation_values,
                    "relational_validity": result.relation_validity,
                    "scene_risk_probabilities": result.outputs["scene_risk_logits"][0]
                    .softmax(-1)
                    .cpu()
                    .tolist(),
                }
            )

    summary = {
        "episode_id": episode_id,
        "model_family": family,
        "map_mode": stream.local_map.mode,
        "oracle_labels_opened": False,
        "map_entities": stream.local_map.records(),
        "unresolved_observations": stream.local_map.unresolved_observations,
        "frames": frame_results,
        "interpretation": "Streaming predicted-map diagnostic only; frame-based training has not taught map-context use.",
    }
    save_json(output_path, summary)
    if attention_debug_output is not None:
        debug = getattr(getattr(model, "backbone", None), "last_attention_debug", None)
        if not debug:
            raise ValueError(
                "Attention diagnostics requested but this architecture exposed no HTransformer debug tensors"
            )
        arrays = {"sample_id": np.asarray(records[-1]["sample_id"])}
        for stage, blocks in debug.items():
            for block_index, block in enumerate(blocks):
                if not block:
                    continue
                for name, value in block.items():
                    if isinstance(value, torch.Tensor):
                        arrays[f"{stage}_block{block_index}_{name}"] = value.numpy()
                    elif value is not None:
                        arrays[f"{stage}_block{block_index}_{name}"] = np.asarray(value)
        destination = Path(attention_debug_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifests/all.jsonl")
    parser.add_argument("--episode", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--attention-debug-output",
        help="Optional NPZ for the final frame's attention/metric/visibility diagnostics",
    )
    args = parser.parse_args()
    summary = run_episode(
        args.checkpoint,
        args.manifest,
        args.episode,
        args.output,
        args.device,
        args.attention_debug_output,
    )
    print(
        f"Processed {len(summary['frames'])} ordered frames; map has {len(summary['map_entities'])} entities."
    )


if __name__ == "__main__":
    main()
