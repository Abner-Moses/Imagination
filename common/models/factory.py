"""Build one of the four registered model families."""

from __future__ import annotations

from typing import Any

from .registry import model_spec


def build_model(config: dict[str, Any]):
    family = config["model"]["family"]
    return model_spec(family).build(config)
