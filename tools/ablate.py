"""Run one named IMF representation ablation through the shared trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
from common.runtime import load_config, merge_dicts
from common.training.engine import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="IMF_HTransformer/config.yaml")
    parser.add_argument("--disable", nargs="+", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = merge_dicts(
        load_config(ROOT / "common/configs/model_common.yaml"),
        load_config(ROOT / args.config),
    )
    config["model"]["disabled_families"] = args.disable
    config["experiment_kind"] = "representation_ablation"
    output = ROOT / "artifacts/checkpoints/ablations" / args.name
    config["training"]["output"] = str(output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(train(config_path, force=args.force))


if __name__ == "__main__":
    main()
