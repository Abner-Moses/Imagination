"""Create clearly labeled representation or end-to-end analytical ablations."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from training.common import load_config
from training.models.analytical_model import FAMILY_CHANNELS
from training.train import train


def create_ablation(
    config_path: str | Path,
    output_directory: str | Path,
    families: list[str],
    mode: str,
    extractor_config: str | Path | None = None,
    dataset_config: str | Path | None = None,
) -> Path:
    unknown = set(families) - FAMILY_CHANNELS.keys()
    if unknown:
        raise ValueError(f"Unknown feature families: {sorted(unknown)}")
    config = copy.deepcopy(load_config(config_path))
    if config["model"]["type"] != "analytical":
        raise ValueError("Analytical feature ablation requires the analytical model")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    config["training"]["output"] = str(output / "run")
    config["experiment"] = {
        "ablation_mode": mode,
        "disabled_families": families,
        "claim": "representation-only" if mode == "representation" else
                 "end-to-end only after regenerating the C++ cache",
    }
    if mode == "representation":
        config["model"]["disabled_families"] = families
    elif mode == "end_to_end":
        if not extractor_config or not dataset_config:
            raise ValueError(
                "End-to-end ablation requires --extractor-config and --dataset-config"
            )
        source = Path(extractor_config).read_text(encoding="utf-8")
        # Geometry is one model family but five independently switchable C++ families.
        cxx_families = [
            family for family in (
                "appearance", "gradients", "hog", "harris", "canny", "contours",
                "chroma", "flow", "depth", "depth_gradients", "slope", "roughness",
                "geometry_confidence",
            ) if not (
                family in families or
                ("geometry" in families and family in {
                    "depth", "depth_gradients", "slope", "roughness", "geometry_confidence"
                })
            )
        ]
        lines = []
        for line in source.splitlines():
            lines.append(f'features: "{",".join(cxx_families)}"' if line.startswith("features:") else line)
        (output / "pre_extraction.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        preparation = load_config(dataset_config)
        preparation["data"]["allow_feature_ablation"] = True
        preparation["data"]["extractor_config"] = str(output / "pre_extraction.yaml")
        (output / "dataset.yaml").write_text(
            yaml.safe_dump(preparation, sort_keys=False), encoding="utf-8"
        )
        config["model"].pop("disabled_families", None)
    else:
        raise ValueError("Ablation mode must be representation or end_to_end")
    destination = output / "experiment.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--without", required=True,
                        help="comma-separated analytical families")
    parser.add_argument("--mode", choices=("representation", "end_to_end"),
                        default="representation")
    parser.add_argument("--extractor-config")
    parser.add_argument("--dataset-config")
    parser.add_argument("--train", action="store_true")
    arguments = parser.parse_args()
    generated = create_ablation(
        arguments.config, arguments.output,
        [item for item in arguments.without.split(",") if item],
        arguments.mode, arguments.extractor_config, arguments.dataset_config,
    )
    print(generated)
    if arguments.train:
        print(train(generated))


if __name__ == "__main__":
    main()
