"""Canonical locations for generated research artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path = PROJECT_ROOT / "artifacts"

    @property
    def build(self) -> Path:
        return self.root / "build"

    @property
    def benchmarks(self) -> Path:
        return self.root / "benchmarks"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def failures(self) -> Path:
        return self.root / "failures"

    @property
    def figures(self) -> Path:
        return self.root / "paper_figures"

    @property
    def preflight(self) -> Path:
        return self.root / "preflight"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def training_readiness(self) -> Path:
        return self.root / "training_readiness"

    @property
    def progress(self) -> Path:
        return self.root / "progress.json"

    def run_checkpoints(self, suite: str, architecture: str, model: str, seed: int) -> Path:
        return self.checkpoints / suite / architecture / model / f"seed_{seed}"

    def run_results(self, suite: str, model: str, seed: int) -> Path:
        return self.results / suite / model / f"seed_{seed}"


ARTIFACTS = ArtifactLayout()
