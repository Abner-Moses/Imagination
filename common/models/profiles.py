"""Explicit capacity profiles; widths change together rather than ad hoc."""

from __future__ import annotations
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    stem: int
    embedding: int
    stage3: int
    stage4: int
    heads3: int
    heads4: int
    decoder: int
    depth3: int = 1
    depth4: int = 1
    fusion_appearance: int = 0
    fusion_motion: int = 0
    fusion_geometry: int = 0


PROFILES = {
    "research": ModelProfile("research", 48, 192, 384, 768, 6, 12, 64, 1, 1, 96, 32, 64),
    "small": ModelProfile("small", 32, 128, 256, 384, 4, 6, 48, 1, 1, 64, 24, 40),
    "tiny": ModelProfile("tiny", 24, 64, 128, 192, 4, 6, 32, 1, 1, 32, 16, 16),
}


def profile_config(name: str, overrides: dict | None = None) -> dict:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile {name!r}; choose {sorted(PROFILES)}")
    values = asdict(PROFILES[name])
    aliases = {"stage3_depth": "depth3", "stage4_depth": "depth4"}
    for key, value in (overrides or {}).items():
        values[aliases.get(key, key)] = value
    return values
