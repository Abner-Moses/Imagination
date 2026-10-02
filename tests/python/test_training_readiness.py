"""Contracts for the frozen pre-LBA training sequence."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml

from common.runner import _planned_runs
from common.training.readiness import PRIMARY_ORDER, PRIMARY_SEEDS


ROOT = Path(__file__).resolve().parents[2]


class TrainingReadinessContracts(unittest.TestCase):
    def test_primary_suite_is_family_block_sequential(self):
        suite = yaml.safe_load((ROOT / "common/configs/suite_primary.yaml").read_text())
        runs = [(row["family"], row["seed"]) for row in _planned_runs(suite)]
        expected = [(family, seed) for family in PRIMARY_ORDER for seed in PRIMARY_SEEDS]
        self.assertEqual(runs, expected)
        self.assertEqual(len(runs), 12)

    def test_primary_budget_and_early_stopping_are_frozen(self):
        suite = yaml.safe_load((ROOT / "common/configs/suite_primary.yaml").read_text())
        shared = yaml.safe_load((ROOT / "common/configs/model_common.yaml").read_text())
        self.assertEqual(suite["epochs"], 80)
        self.assertEqual(tuple(suite["seeds"]), PRIMARY_SEEDS)
        self.assertEqual(shared["training"]["early_stopping_patience"], 12)
        self.assertEqual(shared["training"]["early_stopping_min_epochs"], 8)
        self.assertEqual(shared["training"]["early_stopping_min_delta"], 0.001)

    def test_readiness_effective_batch_candidates_are_exact(self):
        config = yaml.safe_load((ROOT / "common/configs/training_readiness.yaml").read_text())
        effective = config["target_effective_batch"]
        self.assertEqual(effective, 8)
        for micro in config["micro_batch_candidates"]:
            self.assertEqual(micro * (effective // micro), effective)

    def test_canonical_entry_points_exist(self):
        self.assertTrue((ROOT / "train_research_sequence.py").is_file())
        self.assertTrue((ROOT / "train_research_sequence.sh").is_file())

    def test_sequence_evaluates_validation_only(self):
        source = (ROOT / "common/training/sequence.py").read_text(encoding="utf-8")
        self.assertRegex(
            source,
            re.compile(r'evaluate\(\s*checkpoint,\s*manifest,\s*"val"', re.MULTILINE),
        )
        self.assertNotRegex(
            source,
            re.compile(r'evaluate\([^)]*"test"', re.MULTILINE | re.DOTALL),
        )


if __name__ == "__main__":
    unittest.main()
