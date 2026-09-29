"""Model factory for the controlled analytical-versus-RGB comparison."""

from .analytical_model import AnalyticalModel
from .baseline_model import BaselineModel


def build_model(config: dict):
    model_type = config["model"]["type"]
    if model_type == "analytical":
        return AnalyticalModel.from_config(config)
    if model_type == "baseline":
        return BaselineModel.from_config(config)
    raise ValueError(f"Unknown model type: {model_type}")


__all__ = ["AnalyticalModel", "BaselineModel", "build_model"]
