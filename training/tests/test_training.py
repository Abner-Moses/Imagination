from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from training.common import load_checkpoint, save_checkpoint
from training.data import ImaginationDataset, load_manifest, validate_manifest
from training.data.prepare_dataset import generate_smoke_dataset
from training.losses import multitask_loss
from training.metrics import HazardCounts
from training.models import AnalyticalModel, BaselineModel
from training.models.blocks import MBConv
from training.models.htransformer import HTransformerBlock
from training.models.heads import HazardHead, NavigationHead
from training.train import train


SMALL_BACKBONE = {
    "stage3_channels": 24,
    "stage4_channels": 48,
    "stage3_heads": 3,
    "stage4_heads": 6,
    "stage3_depth": 1,
    "stage4_depth": 1,
    "mbconv_expansion": 1.25,
    "mbconv_kernel": 3,
    "se_ratio": 0.25,
    "dropout": 0.0,
    "stochastic_depth": 0.0,
}


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.manifest = generate_smoke_dataset(cls.root / "data", 2)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_dataset_modes_and_contract(self):
        analytical = ImaginationDataset(self.manifest, "train", "analytical")[0]
        baseline = ImaginationDataset(self.manifest, "train", "baseline")[0]
        self.assertEqual(analytical["analytical"].shape, (28, 32, 32))
        self.assertEqual(analytical["validity"].shape, (28, 32, 32))
        self.assertEqual(baseline["rgb"].shape, (3, 256, 256))
        self.assertTrue(torch.equal(analytical["hazard_target"], baseline["hazard_target"]))
        self.assertTrue(torch.equal(analytical["waypoint_target"], baseline["waypoint_target"]))

    def test_mask_aware_input(self):
        values = torch.ones(2, 28, 32, 32)
        validity = torch.ones_like(values)
        validity[:, 0] = 0
        combined = AnalyticalModel.mask_aware_input(values, validity)
        self.assertEqual(combined.shape, (2, 56, 32, 32))
        self.assertEqual(int(combined[:, 0].count_nonzero()), 0)
        self.assertEqual(int(combined[:, 28].count_nonzero()), 0)

    def test_strict_htransformer_value_path(self):
        block = HTransformerBlock(24, (8, 8), 3)
        self.assertIsNot(block.query, block.key)
        self.assertIsInstance(block.value_local, MBConv)
        depthwise = block.value_local.depthwise[0]
        self.assertEqual(depthwise.groups, depthwise.in_channels)
        output = block(torch.randn(2, 24, 8, 8))
        self.assertEqual(output.shape, (2, 24, 8, 8))

    def test_model_forward_shapes_and_shared_heads(self):
        analytical = AnalyticalModel(SMALL_BACKBONE, state_dim=10).eval()
        baseline = BaselineModel(SMALL_BACKBONE, state_dim=10).eval()
        state = torch.zeros(1, 10)
        state_validity = torch.zeros_like(state)
        with torch.inference_mode():
            analytical_output = analytical(
                torch.randn(1, 28, 32, 32), torch.ones(1, 28, 32, 32),
                state, state_validity,
            )
            baseline_output = baseline(torch.randn(1, 3, 256, 256), state, state_validity)
        for output in (analytical_output, baseline_output):
            self.assertEqual(output["hazard_logits"].shape, (1, 1, 32, 32))
            self.assertEqual(output["waypoint"].shape, (1, 5))
        self.assertIs(type(analytical.heads.hazard), type(baseline.heads.hazard))
        self.assertIs(type(analytical.heads.navigation), type(baseline.heads.navigation))
        self.assertIsInstance(analytical.heads.hazard, HazardHead)
        self.assertIsInstance(analytical.heads.navigation, NavigationHead)

    def test_losses_are_finite_and_backward(self):
        outputs = {
            "hazard_logits": torch.randn(2, 1, 32, 32, requires_grad=True),
            "waypoint": torch.randn(2, 5, requires_grad=True),
        }
        batch = {
            "hazard_target": torch.randint(0, 2, (2, 1, 32, 32)).float(),
            "hazard_validity": torch.ones(2, 1, 32, 32),
            "waypoint_target": torch.tensor([[0, 0, 0, 0, 1]] * 2).float(),
        }
        losses = multitask_loss(outputs, batch, {})
        self.assertTrue(torch.isfinite(losses["total"]))
        losses["total"].backward()
        self.assertIsNotNone(outputs["hazard_logits"].grad)

    def test_false_negative_rate(self):
        counts = HazardCounts()
        probabilities = torch.tensor([[[[0.9, 0.1, 0.1, 0.9]]]])
        logits = torch.logit(probabilities)
        target = torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]])
        counts.update(logits, target, torch.ones_like(target))
        result = counts.result()
        self.assertEqual(result["false_negative"], 1)
        self.assertAlmostEqual(result["hazard_fnr"], 0.5)

    def test_split_leakage_detection(self):
        records = load_manifest(self.manifest)
        records[-1]["episode_id"] = records[0]["episode_id"]
        bad = self.root / "leaked.jsonl"
        bad.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Split leakage"):
            validate_manifest(bad, check_files=False)

    def test_channel_order_mismatch_detection(self):
        records = load_manifest(self.manifest)
        source = Path(records[0]["analytical_path"])
        with np.load(source, allow_pickle=False) as data:
            content = {name: data[name] for name in data.files}
        content["channel_names"] = content["channel_names"][::-1]
        bad_tensor = self.root / "bad_channels.npz"
        np.savez_compressed(bad_tensor, **content)
        records[0]["analytical_path"] = str(bad_tensor)
        bad_manifest = self.root / "bad_channels.jsonl"
        bad_manifest.write_text("\n".join(json.dumps(record) for record in records),
                                encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "registry mismatch"):
            validate_manifest(bad_manifest)

    def test_checkpoint_round_trip(self):
        model = AnalyticalModel(SMALL_BACKBONE, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        checkpoint = self.root / "round_trip.pt"
        save_checkpoint(checkpoint, model, optimizer, None, 2, 9, {
            "model": {"type": "analytical"}, "htransformer": SMALL_BACKBONE,
        }, {"loss": 1.0}, "manifest-hash")
        restored = AnalyticalModel(SMALL_BACKBONE, 10)
        data = load_checkpoint(checkpoint, restored)
        self.assertEqual(data["epoch"], 2)
        for left, right in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(left, right))

    def test_two_epoch_smoke_training(self):
        output = self.root / "run"
        config = {
            "model": {"type": "analytical", "state_dim": 10},
            "htransformer": {**SMALL_BACKBONE, "stage3_depth": 0, "stage4_depth": 0},
            "training": {
                "output": str(output), "epochs": 2, "batch_size": 2,
                "workers": 0, "learning_rate": 1e-3, "seed": 3,
                "device": "cpu", "mixed_precision": False,
                "deterministic": False,
            },
            "losses": {},
            "data": {"manifest": str(self.manifest), "validate_files": True},
        }
        config_path = self.root / "smoke.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        checkpoint = train(config_path)
        self.assertTrue(checkpoint.is_file())
        self.assertTrue((output / "last.pt").is_file())


if __name__ == "__main__":
    unittest.main()
