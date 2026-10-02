"""Authoritative registry for the four model families in the experiment."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from common.registry import MODEL_ARCHITECTURE_VERSION


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    module: str
    class_name: str
    config_directory: str
    observation: str
    profiles: tuple[str, ...] = ("research", "small", "tiny")
    architecture_version: str = MODEL_ARCHITECTURE_VERSION

    @property
    def uses_analytical_observation(self) -> bool:
        return self.observation == "analytical"

    def build(self, config: dict[str, Any]):
        model_class = getattr(import_module(self.module), self.class_name)
        return model_class.from_config(config)

    def config_path(self, root: Path) -> Path:
        return root / self.config_directory / "config.yaml"


MODEL_SPECS = {
    spec.id: spec
    for spec in (
        ModelSpec("cnn", "CNN", "CNN.model", "CNNModel", "CNN", "rgb"),
        ModelSpec("cnn_vit", "CNN-ViT", "CNN_ViT.model", "CNNViTModel", "CNN_ViT", "rgb"),
        ModelSpec(
            "cnn_htransformer",
            "CNN-HTransformer (CoHAtNet-inspired)",
            "CNN_HTransformer.model",
            "CNNHTransformer",
            "CNN_HTransformer",
            "rgb",
        ),
        ModelSpec(
            "imf_htransformer",
            "Imagination IMF-HTransformer",
            "IMF_HTransformer.model",
            "IMFHTransformer",
            "IMF_HTransformer",
            "analytical",
        ),
    )
}

MODEL_IDS = tuple(MODEL_SPECS)
PRIMARY_TRAINING_ORDER = ("cnn_htransformer", "cnn_vit", "cnn", "imf_htransformer")


def model_spec(model_id: str) -> ModelSpec:
    try:
        return MODEL_SPECS[model_id]
    except KeyError as error:
        raise ValueError(f"Unknown model family {model_id!r}; choose from {MODEL_IDS}") from error
