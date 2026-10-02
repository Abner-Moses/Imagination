"""Fast architecture, mapping, and small-sample learning checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common.mapping import Detection2D, PersistentSemanticMap, register_detection
from common.models import build_model
from common.models.blocks import MBConv
from common.models.htransformer import HTransformerBlock
from common.runner import family_config
from common.training.engine import forward_model, model_mode, move_targets
from common.training.losses import multitask_loss
from data.adapter import ImaginationDataset
from data.preprocessing.prepare import generate_smoke_dataset


def check_attention_contract() -> None:
    block = HTransformerBlock(24, (8, 8), 3)
    assert block.query is not block.key
    assert isinstance(block.value_local, MBConv)
    assert not hasattr(block, "value")
    assert block.value_local.depthwise[0].groups == block.value_local.depthwise[0].in_channels


def check_map_projection() -> None:
    camera = {"width": 640, "height": 480, "fx": 440.0, "fy": 440.0, "cx": 319.5, "cy": 239.5}
    depth = np.zeros((32, 32), np.float32)
    valid = np.zeros_like(depth)
    depth[16, 16] = 0.5
    valid[16, 16] = 1
    detection = Detection2D("poi", "rock", (16.0, 16.0), 4, 0.9)
    mapped = register_detection(detection, depth, valid, 4.0, camera, np.eye(4), 0.0)
    assert mapped.resolved and abs(mapped.map_xyz_m[2] - 2.0) < 1e-6
    local_map = PersistentSemanticMap()
    first = local_map.update_region(
        "rock", mapped.map_xyz_m, mapped.radius_m, mapped.probability, 0.0
    )
    second = local_map.update_region(
        "rock", mapped.map_xyz_m, mapped.radius_m, mapped.probability, 1.0
    )
    assert first.entity_id == second.entity_id and second.observation_count == 2
    unresolved = register_detection(
        detection, depth, np.zeros_like(valid), 4.0, camera, np.eye(4), 2.0
    )
    assert not unresolved.resolved and unresolved.map_xyz_m is None


def _tiny_config(family: str, output: Path) -> dict:
    config = family_config(family)
    config["model"].update(
        {
            "family": family,
            "profile": "tiny",
            "stage3_depth": 0,
            "stage4_depth": 0,
        }
    )
    config["training"].update({"seed": 7, "output": str(output), "max_epochs": 1})
    config["data"]["validate_files"] = True
    for name in ("hazard_positive_weight", "landing_positive_weight", "poi_positive_weight"):
        config["losses"][name] = 1.0
    return config


def _overfit(manifest: Path, family: str, work: Path, steps: int = 10) -> dict:
    from common.registry import POI_CLASSES

    data = ImaginationDataset(manifest, "train", model_mode(family), POI_CLASSES)
    samples = min(16, len(data))
    loader = DataLoader(Subset(data, range(samples)), batch_size=min(4, samples), shuffle=False)
    config = _tiny_config(family, work / family)
    config["data"]["manifest"] = str(manifest)
    model = build_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = last = None
    modules = ("hazard", "landing", "semantic", "poi")

    batches = list(loader)
    for step in range(steps):
        batch = batches[step % len(batches)]
        move_targets(batch, torch.device("cpu"))
        outputs = forward_model(model, batch, family, torch.device("cpu"))
        losses = multitask_loss(outputs, batch, {"overlap_weight": 0.0})
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        for name in modules:
            head = getattr(model.heads, name)
            if not any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in head.parameters()
            ):
                raise RuntimeError(f"{family} {name} head did not receive finite gradients")
        optimizer.step()
        current = float(losses["total"].detach())
        first = current if first is None else first
        last = current
    if not last < first * 0.995:
        raise RuntimeError(
            f"{family} tiny overfit probe did not reduce loss: {first:.5f} -> {last:.5f}"
        )
    return {
        "family": family,
        "samples": samples,
        "steps": steps,
        "initial_loss": first,
        "final_loss": last,
    }


def run_preflight(manifest: Path, work: Path | None = None) -> dict:
    check_attention_contract()
    check_map_projection()
    if work is None:
        work = Path("artifacts") / "preflight"
    return {
        "qk_mbconv_value": "PASS",
        "metric_projection_and_map_association": "PASS",
        "overfit": {
            family: _overfit(manifest, family, work)
            for family in ("cnn", "cnn_vit", "cnn_htransformer", "imf_htransformer")
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = (
        Path(args.manifest)
        if args.manifest
        else generate_smoke_dataset(Path("artifacts") / "smoke_data")
    )
    print(run_preflight(manifest))


if __name__ == "__main__":
    main()
