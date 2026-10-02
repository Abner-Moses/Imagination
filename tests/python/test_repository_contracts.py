"""Repository-level contracts shared across model families and entry points."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch

from common.contracts import required_input_fields, validate_model_outputs
from common.models.registry import MODEL_IDS, PRIMARY_TRAINING_ORDER, model_spec
from common.registry import (
    CANDIDATE_FEATURE_NAMES,
    MAP_SCHEMA_VERSION,
    contract_versions,
)
from common.mapping.semantic_map import MAP_SCHEMA
from common.runtime import source_tree_hash


class RepositoryContracts(unittest.TestCase):
    def test_model_registry_is_authoritative_for_primary_families(self):
        self.assertEqual(
            MODEL_IDS,
            ("cnn", "cnn_vit", "cnn_htransformer", "imf_htransformer"),
        )
        self.assertEqual(
            PRIMARY_TRAINING_ORDER,
            ("cnn_htransformer", "cnn_vit", "cnn", "imf_htransformer"),
        )
        self.assertEqual(
            model_spec("cnn_htransformer").display_name,
            "CNN-HTransformer (CoHAtNet-inspired)",
        )
        self.assertTrue(model_spec("imf_htransformer").uses_analytical_observation)

    def test_input_contract_keeps_state_and_analytical_maps_separate(self):
        analytical = required_input_fields("analytical")
        self.assertIn("analytical", analytical)
        self.assertIn("vehicle_state", analytical)
        self.assertIn("candidate_feature_validity", analytical)
        self.assertNotEqual(analytical.index("analytical"), analytical.index("vehicle_state"))
        self.assertEqual(len(CANDIDATE_FEATURE_NAMES), 40)

    def test_output_contract_accepts_shared_head_names(self):
        outputs = {
            "hazard_logits": torch.zeros(1, 1, 32, 32),
            "landing_logits": torch.zeros(1, 1, 32, 32),
            "semantic_logits": torch.zeros(1, 6, 32, 32),
            "poi": {"class_logits": torch.zeros(1, 7, 32, 32)},
            "scene_risk_logits": torch.zeros(1, 3),
            "candidate": {
                "risk_logits": torch.zeros(1, 32, 3),
                "landing_safe_logits": torch.zeros(1, 32),
                "validity": torch.zeros(1, 32),
            },
        }
        validate_model_outputs(outputs)
        with self.assertRaisesRegex(ValueError, "scene_risk_logits"):
            validate_model_outputs(
                {key: value for key, value in outputs.items() if key != "scene_risk_logits"}
            )

    def test_map_schema_uses_central_version_contract(self):
        self.assertEqual(MAP_SCHEMA, MAP_SCHEMA_VERSION)
        versions = contract_versions()
        self.assertEqual(versions["map_schema"], MAP_SCHEMA_VERSION)
        self.assertIn("checkpoint", versions)
        self.assertIn("training_plan", versions)

    def test_source_tree_hash_ignores_generated_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "module.py").write_text("value = 1\n", encoding="utf-8")
            first = source_tree_hash(root)
            generated = root / "artifacts"
            generated.mkdir()
            (generated / "metrics.json").write_text('{"value": 2}\n', encoding="utf-8")
            self.assertEqual(source_tree_hash(root), first)
            (root / "module.py").write_text("value = 2\n", encoding="utf-8")
            self.assertNotEqual(source_tree_hash(root), first)


if __name__ == "__main__":
    unittest.main()
