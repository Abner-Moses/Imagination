from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common.mapping import (
    Detection2D,
    PersistentSemanticMap,
    SemanticObservation,
    compute_relations,
    extract_regions,
    register_detection,
)
from common.mapping.context import _project_candidates
from common.models import build_model
from common.models.blocks import MBConv
from common.models.htransformer import HTransformerBlock, HTransformerBackbone
from IMF_HTransformer.model import FeatureFamilyAdapter
from common.models.metric_attention import (
    grid_neighborhood,
    metric_neighborhood,
    positions_from_analytical,
    spatial_attention,
)
from common.feature_spec import FUSION_FAMILIES, validate_feature_batch
from common.mapping.coordinates import (
    map_to_egocentric,
    egocentric_to_map,
    map_vector_to_egocentric,
    egocentric_vector_to_map,
)
from common.mapping.uncertainty import (
    approximate_pose_covariance,
    mahalanobis_squared,
    pixel_depth_covariance,
    propagate_camera_to_map,
    stabilize_covariance,
)
from common.mapping.visibility import Visibility, classify_map_point
from common.registry import (
    ANALYTICAL_CHANNELS,
    CANDIDATE_FEATURE_NAMES,
    MAX_CANDIDATES,
    RELATIONAL_NAMES,
    SCENE_CLASSES,
    SEMANTIC_CLASSES,
    STATE_NAMES,
)
from common.runtime import estimate_macs, load_checkpoint, save_checkpoint
from common.training.engine import attach_map_context, forward_model, model_mode, move_targets
from common.training.losses import multitask_loss
from common.training.metrics import BinaryCounts, CategoricalMetrics
from data.adapter import ImaginationDataset, load_manifest, validate_manifest
from data.preprocessing.prepare import generate_smoke_dataset
from data.preprocessing.prepare import ensure_candidate_depth_cache
from data.preprocessing.targets import landing_class_grid, scene_verdict, targets_from_sources


TINY = {
    "model": {
        "profile": "tiny",
        "state_dim": 13,
        "state_conditioning": True,
        "stage3_depth": 0,
        "stage4_depth": 0,
        "disabled_families": [],
    },
    "training": {"seed": 7},
    "losses": {},
    "poi": {
        "classes": {
            "cone": 3,
            "rock": 4,
            "football": 5,
            "bag": 6,
            "box": 7,
            "pole": 8,
            "chair": 9,
        }
    },
}
FAMILIES = ("cnn", "cnn_vit", "cnn_htransformer", "imf_htransformer")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.manifest = generate_smoke_dataset(cls.root / "smoke")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def config(self, family):
        return {**TINY, "model": {**TINY["model"], "family": family}}

    def test_fixed_registries(self):
        self.assertEqual(len(ANALYTICAL_CHANNELS), 28)
        self.assertEqual(ANALYTICAL_CHANNELS[0], "Y")
        self.assertEqual(ANALYTICAL_CHANNELS[-1], "GeometryConfidence")
        self.assertEqual(len(STATE_NAMES), 13)
        self.assertEqual(STATE_NAMES[-1], "ultrasonic_axis_range_m")
        self.assertEqual(len(RELATIONAL_NAMES), 30)
        self.assertEqual(len(CANDIDATE_FEATURE_NAMES), 40)
        self.assertEqual(tuple(FUSION_FAMILIES), ("appearance", "motion", "geometry"))
        self.assertEqual(sum(len(indices) for indices in FUSION_FAMILIES.values()), 28)
        self.assertEqual(MAX_CANDIDATES, 32)

    def test_dataset_shapes_and_masked_unknown_geometry(self):
        row = ImaginationDataset(self.manifest, "train", "analytical")[0]
        self.assertEqual(tuple(row["analytical"].shape), (28, 32, 32))
        self.assertEqual(tuple(row["validity"].shape), (28, 32, 32))
        self.assertEqual(tuple(row["vehicle_state"].shape), (13,))
        self.assertEqual(tuple(row["relational"].shape), (30,))
        self.assertEqual(tuple(row["candidates"].shape), (32, len(CANDIDATE_FEATURE_NAMES)))
        self.assertFalse(bool(row["camera_pose_valid"]))
        self.assertEqual(tuple(row["poi_target"].shape), (7, 32, 32))
        rgb = ImaginationDataset(self.manifest, "train", "rgb_current")[0]
        self.assertEqual(tuple(rgb["rgb"].shape), (3, 256, 256))
        self.assertTrue(torch.all(rgb["relational_validity"] == 0))
        inference = ImaginationDataset(
            self.manifest, "train", "rgb_current", include_targets=False
        )[0]
        self.assertNotIn("hazard_target", inference)

    def test_optimized_analytical_loader_preserves_cached_values(self):
        dataset = ImaginationDataset(self.manifest, "train", "analytical")
        row = dataset[0]
        record = dataset.records[0]
        path = (dataset.base / record["analytical_path"]).resolve()
        with np.load(path, allow_pickle=False) as cached:
            expected_features = cached["features"].astype(np.float32)
            expected_validity = (cached["validity"] > 0).astype(np.float32)
        np.testing.assert_array_equal(row["analytical"].numpy(), expected_features)
        np.testing.assert_array_equal(row["validity"].numpy(), expected_validity)

    def test_source_class_id_target_mapping(self):
        landing = np.full((32, 32), 2, dtype=np.uint8)
        landing[:8] = 1
        landing[:4] = 0
        semantic = np.zeros((32, 32), dtype=np.uint8)
        semantic[10:14, 10:14] = 4
        hazard, suitable, poi = targets_from_sources(landing, semantic)
        self.assertGreater(float(hazard.sum()), 0.0)
        self.assertGreater(float(suitable.sum()), 0.0)
        self.assertGreater(float(poi[1].sum()), 0.0)
        self.assertEqual(scene_verdict(landing, semantic), 2)

    def test_landing_class_grid_uses_conservative_area_majority(self):
        source = np.full((64, 64), 2, dtype=np.uint8)
        source[:32, :32] = 1
        source[:16, :32] = 0
        grid = landing_class_grid(source)
        self.assertEqual(grid[4, 4], 0)
        self.assertEqual(grid[10, 4], 1)
        self.assertEqual(grid[20, 20], 2)

    def test_target_depth_cache_is_compact_metric_supervision(self):
        directory = self.root / "depth_fixture"
        directory.mkdir()
        source = directory / "depth.npy"
        destination = directory / "frame_0.npz"
        np.save(source, np.full((64, 96), 3.25, dtype=np.float32))
        record = {
            "sample_id": "depth_fixture:0",
            "depth_path": str(source),
            "analytical_path": str(destination),
        }
        self.assertTrue(ensure_candidate_depth_cache(directory, record))
        self.assertFalse(ensure_candidate_depth_cache(directory, record))
        target = destination.with_name(destination.stem + ".target_depth.npz")
        with np.load(target, allow_pickle=False) as cache:
            self.assertEqual(cache["depth_m"].shape, (32, 32))
            self.assertTrue(np.all(cache["validity"] == 1))
            self.assertTrue(np.allclose(cache["depth_m"], 3.25))
            self.assertEqual(str(cache["source_units"].item()), "camera_z_metres")

    def test_candidate_projection_requires_matching_visible_depth(self):
        candidates = np.zeros((MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32)
        candidates[0, :3] = (0.0, 0.0, 2.0)
        candidate_validity = np.zeros(MAX_CANDIDATES, dtype=np.float32)
        candidate_validity[0] = 1.0
        result = SimpleNamespace(
            candidate_values=tuple(tuple(row) for row in candidates),
            candidate_validity=tuple(candidate_validity),
            candidate_map_xyz=tuple(
                (0.0, 0.0, 2.0) if i == 0 else (0.0, 0.0, 0.0) for i in range(MAX_CANDIDATES)
            ),
        )
        features = torch.zeros(28, 32, 32)
        validity = torch.zeros_like(features)
        features[22, 16, 16] = 0.1  # 2 m under the configured 20 m scale.
        validity[22, 16, 16] = 1.0
        geometry = {
            "camera_pose_valid": torch.tensor(True),
            "camera_to_local_map": torch.eye(4, dtype=torch.float64),
            "calibration": torch.tensor((640, 480, 440, 440, 319.5, 239.5), dtype=torch.float64),
            "depth_scale_m": torch.tensor(20.0),
            "analytical": features,
            "validity": validity,
        }
        grid, projected_depth, projected, visibility = _project_candidates(result, geometry)
        self.assertEqual(projected[0].item(), 1.0)
        self.assertEqual(tuple(grid[0].tolist()), (16.0, 16.0))
        self.assertEqual(projected_depth[0].item(), 2.0)
        self.assertEqual(visibility[0].item(), int(Visibility.VISIBLE))
        occluded_geometry = dict(geometry)
        occluded_geometry["analytical"] = features.clone()
        occluded_geometry["analytical"][22, 16, 16] = 0.05
        _, _, _, occluded = _project_candidates(result, occluded_geometry)
        _, _, _, ungated = _project_candidates(result, occluded_geometry, use_visibility=False)
        self.assertEqual(occluded[0].item(), int(Visibility.OCCLUDED))
        self.assertEqual(ungated[0].item(), int(Visibility.VISIBLE))

    def test_candidate_verdict_targets_are_separate_from_map_inputs(self):
        sample_id = "episode_000001:000000"
        batch = {
            "metadata": {"sample_id": [sample_id]},
            "landing_class_target": torch.full((1, 32, 32), 1, dtype=torch.long),
            "candidate_depth_target_m": torch.full((1, 32, 32), 2.0),
            "candidate_depth_validity": torch.ones((1, 32, 32)),
        }
        context = {
            "relational": torch.zeros(len(RELATIONAL_NAMES)),
            "relational_validity": torch.ones(len(RELATIONAL_NAMES)),
            "candidates": torch.zeros(MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES)),
            "candidate_validity": torch.zeros(MAX_CANDIDATES),
            "candidate_feature_validity": torch.zeros(MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES)),
            "candidate_grid": torch.zeros(MAX_CANDIDATES, 2),
            "candidate_projected_depth_m": torch.zeros(MAX_CANDIDATES),
            "candidate_projection_validity": torch.zeros(MAX_CANDIDATES),
            "candidate_visibility": torch.zeros(MAX_CANDIDATES, dtype=torch.long),
        }
        context["candidates"][0, 0] = 1.0
        context["candidate_validity"][0] = 1.0
        context["candidate_grid"][0] = torch.tensor((4, 7))
        context["candidate_projected_depth_m"][0] = 2.0
        context["candidate_projection_validity"][0] = 1.0
        batch["landing_class_target"][0, 7, 4] = 2
        attach_map_context(batch, {sample_id: context})
        self.assertEqual(batch["candidate_risk_target"][0, 0].item(), 0)  # SAFE
        self.assertEqual(batch["candidate_landing_target"][0, 0].item(), 1.0)
        self.assertEqual(batch["candidate_target_validity"][0, 0].item(), 1.0)
        self.assertEqual(batch["candidate_target_validity"][0, 1].item(), 0.0)
        batch["candidate_depth_target_m"][0, 7, 4] = 5.0
        attach_map_context(batch, {sample_id: context})
        self.assertEqual(batch["candidate_target_validity"][0, 0].item(), 0.0)

    def test_grouped_split_contract(self):
        records = load_manifest(self.manifest)
        self.assertEqual(
            validate_manifest(self.manifest, check_files=True, require_analytical=True)["episodes"],
            24,
        )
        broken = list(records)
        broken[-1] = {**broken[-1], "episode_id": records[0]["episode_id"]}
        path = self.root / "leaked.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in broken), encoding="utf-8")
        with self.assertRaises(ValueError):
            validate_manifest(path, check_files=False, require_analytical=False)

    def test_q_k_and_mbconv_value_are_distinct(self):
        block = HTransformerBlock(24, (8, 8), 3)
        self.assertIsNot(block.query, block.key)
        self.assertIsInstance(block.value_local, MBConv)
        self.assertFalse(hasattr(block, "value"))
        depthwise = block.value_local.depthwise[0]
        self.assertEqual(depthwise.groups, depthwise.in_channels)
        self.assertEqual(block(torch.randn(1, 24, 8, 8)).shape, (1, 24, 8, 8))

    def test_feature_spec_ranges_and_binary_validity(self):
        features = torch.zeros(2, 28, 32, 32)
        validity = torch.ones_like(features)
        features[:, 3] = -1.0
        features[:, 16] = 1.0
        features[:, 22] = 0.5
        validate_feature_batch(features, validity)
        validity[:, 22] = 0
        validate_feature_batch(features, validity)
        with self.assertRaises(ValueError):
            validate_feature_batch(features, torch.full_like(validity, 0.5))

    def test_family_adapter_masks_invalid_values_and_skips_disabled_family(self):
        adapter = FeatureFamilyAdapter(2, 8).eval()
        validity = torch.tensor([[[[1.0]], [[0.0]]]])
        first = adapter(torch.tensor([[[[0.25]], [[100.0]]]]), validity)
        second = adapter(torch.tensor([[[[0.25]], [[-900.0]]]]), validity)
        self.assertTrue(torch.allclose(first, second))

        config = self.config("imf_htransformer")
        config["model"]["disabled_fusion_families"] = ["geometry"]
        model = build_model(config).eval()
        self.assertNotIn("geometry", model.family_adapters)
        self.assertEqual(sum(model.family_widths.values()), 64)
        outputs = model(torch.zeros(1, 28, 32, 32), torch.ones(1, 28, 32, 32))
        self.assertEqual(tuple(outputs["hazard_logits"].shape), (1, 1, 32, 32))

    def test_metric_positions_use_measured_depth_in_camera_frame(self):
        features = torch.zeros(1, 28, 32, 32)
        masks = torch.zeros_like(features)
        features[:, 22] = 0.1
        masks[:, 22] = 1.0
        calibration = torch.tensor([[640.0, 480.0, 440.0, 440.0, 319.5, 239.5]])
        points, valid = positions_from_analytical(features, masks, calibration, 20.0)
        self.assertTrue(torch.all(valid == 1))
        self.assertTrue(torch.allclose(points[0, 16, 16, 2], torch.tensor(2.0)))
        self.assertLess(abs(float(points[0, 16, 16, 0])), 0.06)
        self.assertLess(abs(float(points[0, 16, 16, 1])), 0.06)

    def test_sparse_attention_equivalence_and_actual_pair_count(self):
        torch.manual_seed(11)
        batch, heads, count, dim = 1, 2, 16, 4
        query, key, value = [
            torch.randn(batch, heads, count, dim, requires_grad=True) for _ in range(3)
        ]
        positions = torch.randn(batch, 4, 4, 3)
        valid = torch.ones(batch, 4, 4)
        context_key = torch.randn(batch, heads, 3, dim, requires_grad=True)
        context_value = torch.randn(batch, heads, 3, dim, requires_grad=True)
        context_positions = torch.randn(batch, 3, 3)
        context_validity = torch.tensor([[1.0, 0.0, 1.0]])
        all_neighbors = grid_neighborhood(4, 4, count)
        dense, dense_stats = spatial_attention(
            query,
            key,
            value,
            grid_shape=(4, 4),
            mode="dense",
            positions=positions,
            position_validity=valid,
            position_bias_mode="metric",
            metric_weight=0.4,
            context_key=context_key,
            context_value=context_value,
            context_positions=context_positions,
            context_validity=context_validity,
        )
        sparse, sparse_stats = spatial_attention(
            query,
            key,
            value,
            grid_shape=(4, 4),
            mode="sparse",
            positions=positions,
            position_validity=valid,
            position_bias_mode="metric",
            metric_weight=0.4,
            neighbor_indices=all_neighbors,
            context_key=context_key,
            context_value=context_value,
            context_positions=context_positions,
            context_validity=context_validity,
        )
        self.assertTrue(torch.allclose(dense, sparse, atol=1e-6, rtol=1e-6))
        self.assertEqual(dense_stats["attention_pairs"], batch * heads * count * (count + 3))
        self.assertEqual(sparse_stats["attention_pairs"], batch * heads * count * (count + 3))
        five_neighbors = grid_neighborhood(4, 4, 5)
        local, stats = spatial_attention(
            query,
            key,
            value,
            grid_shape=(4, 4),
            mode="sparse",
            positions=positions,
            position_validity=valid,
            position_bias_mode="metric",
            metric_weight=0.4,
            neighbor_indices=five_neighbors,
            context_key=context_key,
            context_value=context_value,
            context_positions=context_positions,
            context_validity=context_validity,
        )
        self.assertEqual(stats["attention_pairs"], batch * heads * count * (5 + 3))
        self.assertFalse(torch.allclose(dense, local))
        local.square().mean().backward()
        self.assertTrue(all(torch.isfinite(tensor.grad).all() for tensor in (query, key, value)))
        self.assertTrue(torch.isfinite(context_key.grad).all())
        self.assertTrue(torch.isfinite(context_value.grad).all())

    def test_sparse_attention_supports_reduced_qk_width_with_mbconv_value_width(self):
        block = HTransformerBlock(
            24,
            (4, 4),
            3,
            qk_channels=12,
            attention_mode="sparse",
            sparse_neighbors=6,
            mlp_ratio=1.0,
        )
        feature = torch.randn(2, 24, 4, 4, requires_grad=True)
        output = block(feature)
        self.assertEqual(tuple(output.shape), tuple(feature.shape))
        self.assertEqual(block.query.out_features, 12)
        self.assertEqual(block.key.out_features, 12)
        self.assertEqual(block.value_local.project[1].in_channels, 48)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(feature.grad).all())

    def test_context_tokens_are_permutation_invariant(self):
        block = HTransformerBlock(
            24,
            (4, 4),
            3,
            qk_channels=12,
            attention_mode="sparse",
            sparse_neighbors=8,
            position_bias_mode="metric",
            context_features=10,
        ).eval()
        spatial = torch.randn(1, 24, 4, 4)
        positions = torch.randn(1, 4, 4, 3)
        position_validity = torch.ones(1, 4, 4)
        context = torch.randn(1, 5, 10)
        context_positions = torch.randn(1, 5, 3)
        context_validity = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0]])
        with torch.inference_mode():
            output_a = block(
                spatial,
                positions,
                position_validity,
                context,
                context_positions,
                context_validity,
            )
            permutation = torch.tensor([3, 0, 4, 1, 2])
            output_b = block(
                spatial,
                positions,
                position_validity,
                context[:, permutation],
                context_positions[:, permutation],
                context_validity[:, permutation],
            )
        self.assertTrue(torch.allclose(output_a, output_b, atol=1e-6, rtol=1e-6))
        self.assertIsInstance(block.value_local, MBConv)
        self.assertIsInstance(block.context_value_encoder, torch.nn.Linear)

    def test_stage_specific_attention_context_and_qk_policies_forward_backward(self):
        policies = (
            ("sparse", "sparse", True, True),
            ("sparse", "dense", False, True),
            ("dense", "dense", False, False),
            ("dense", "sparse", True, False),
        )
        for stage3_mode, stage4_mode, context3, context4 in policies:
            with self.subTest(stage3=stage3_mode, stage4=stage4_mode):
                backbone = HTransformerBackbone(
                    {
                        "embedding": 8,
                        "stage3_channels": 12,
                        "stage4_channels": 16,
                        "stage3_heads": 2,
                        "stage4_heads": 2,
                        "stage3_qk_channels": 6,
                        "stage4_qk_channels": 8,
                        "stage3_depth": 1,
                        "stage4_depth": 1,
                        "stage3_attention_mode": stage3_mode,
                        "stage4_attention_mode": stage4_mode,
                        "stage3_sparse_neighbors": 8,
                        "stage4_sparse_neighbors": 8,
                        "stage3_sparse_selection": "grid",
                        "stage4_sparse_selection": "grid",
                        "stage3_use_map_context": context3,
                        "stage4_use_map_context": context4,
                        "context_features": 10,
                        "position_bias_mode": "none",
                    },
                    input_channels=8,
                )
                image = torch.randn(1, 8, 32, 32, requires_grad=True)
                context = torch.randn(1, 3, 10)
                context_positions = torch.randn(1, 3, 3)
                context_validity = torch.ones(1, 3)
                output = backbone(
                    image,
                    context_tokens=context,
                    context_positions=context_positions,
                    context_validity=context_validity,
                )
                self.assertEqual(tuple(output.shape), (1, 16, 8, 8))
                self.assertEqual(backbone.stage3[0].query.out_features, 6)
                self.assertEqual(backbone.stage4[0].query.out_features, 8)
                self.assertEqual(
                    backbone.last_attention_stats["stage3"]["attention_pairs"],
                    2 * 256 * (8 if stage3_mode == "sparse" else 256)
                    + 2 * 256 * (3 if context3 else 0),
                )
                self.assertEqual(
                    backbone.last_attention_stats["stage4"]["attention_pairs"],
                    2 * 64 * (8 if stage4_mode == "sparse" else 64)
                    + 2 * 64 * (3 if context4 else 0),
                )
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(image.grad).all())

    def test_candidate_missing_feature_mask_differs_from_measured_zero(self):
        from common.models.heads import PerceptionHeads

        actual_zero = torch.zeros(1, 1, len(CANDIDATE_FEATURE_NAMES))
        measured_validity = torch.ones_like(actual_zero)
        missing_validity = measured_validity.clone()
        missing_validity[..., CANDIDATE_FEATURE_NAMES.index("slope_rad")] = 0
        measured = PerceptionHeads.mask_candidate_input(actual_zero, measured_validity)
        missing = PerceptionHeads.mask_candidate_input(actual_zero, missing_validity)
        self.assertFalse(torch.equal(measured, missing))

    def test_backbone_supports_sparse_stage3_and_dense_stage4(self):
        from common.models.htransformer import HTransformerBackbone

        backbone = HTransformerBackbone(
            {
                "embedding": 8,
                "stage3_channels": 12,
                "stage4_channels": 16,
                "stage3_heads": 2,
                "stage4_heads": 2,
                "stage3_qk_channels": 6,
                "stage4_qk_channels": 8,
                "stage3_depth": 1,
                "stage4_depth": 1,
                "stage3_attention_mode": "sparse",
                "stage4_attention_mode": "dense",
                "sparse_neighbors": 8,
                "position_bias_mode": "none",
            },
            input_channels=8,
        )
        result = backbone(torch.randn(1, 8, 32, 32))
        self.assertEqual(tuple(result.shape), (1, 16, 8, 8))
        self.assertEqual(backbone.stage3[0].attention_mode, "sparse")
        self.assertEqual(backbone.stage4[0].attention_mode, "dense")
        self.assertEqual(backbone.stage3[0].query.out_features, 6)
        self.assertEqual(backbone.stage4[0].query.out_features, 8)

    def test_metric_neighborhood_selects_physical_neighbors_from_bounded_pool(self):
        positions = torch.full((1, 16, 3), 100.0)
        positions[0, 0] = torch.tensor([0.0, 0.0, 0.0])
        positions[0, 15] = torch.tensor([0.01, 0.0, 0.0])
        valid = torch.ones(1, 16)
        pool = grid_neighborhood(4, 4, 16)
        neighbors = metric_neighborhood(
            positions,
            valid,
            pool,
            4,
            grid_shape=(4, 4),
            global_anchors=0,
        )
        self.assertEqual(int(neighbors[0, 0, 0]), 0)
        self.assertIn(15, neighbors[0, 0].tolist())

    def test_sparse_metric_relative_bias_has_finite_gradient(self):
        block = HTransformerBlock(
            24,
            (4, 4),
            3,
            attention_mode="sparse",
            position_bias_mode="both",
            sparse_neighbors=6,
        )
        image = torch.randn(1, 24, 4, 4, requires_grad=True)
        positions = torch.randn(1, 4, 4, 3)
        valid = torch.ones(1, 4, 4)
        output = block(image, positions, valid)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(output).all())
        self.assertIsNotNone(block.relative_bias.table.grad)

    def test_active_query_selection_is_deterministic_and_keeps_safety_coverage(self):
        from common.models.query_selection import active_query_indices

        importance = torch.zeros(1, 8, 8)
        importance[0, 7, 7] = 10.0
        mandatory = torch.zeros_like(importance, dtype=torch.bool)
        mandatory[0, 0, 0] = True
        first, first_validity = active_query_indices(
            importance,
            mandatory,
            0.25,
            coverage_bins=(2, 2),
        )
        second, second_validity = active_query_indices(
            importance,
            mandatory,
            0.25,
            coverage_bins=(2, 2),
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first_validity, second_validity))
        chosen = first[0, first_validity[0] > 0].tolist()
        self.assertIn(0, chosen)
        self.assertIn(63, chosen)
        self.assertGreaterEqual(len(chosen), 4)

    def test_active_query_attention_saves_query_pairs_and_keeps_dense_grid_output(self):
        block = HTransformerBlock(
            24,
            (4, 4),
            3,
            qk_channels=12,
            attention_mode="sparse",
            sparse_neighbors=5,
            position_bias_mode="both",
            active_query_fraction=0.25,
            active_query_coverage_bins=(2, 2),
        )
        image = torch.randn(1, 24, 4, 4, requires_grad=True)
        importance = torch.rand(1, 4, 4)
        mandatory = torch.zeros_like(importance, dtype=torch.bool)
        mandatory[0, 0, 0] = True
        metric_positions = torch.randn(1, 4, 4, 3)
        metric_validity = torch.ones(1, 4, 4)
        output = block(
            image,
            metric_positions,
            metric_validity,
            query_importance=importance,
            query_mandatory=mandatory,
        )
        self.assertEqual(tuple(output.shape), tuple(image.shape))
        self.assertLess(block.last_attention_stats["active_queries"], 16)
        self.assertEqual(
            block.last_attention_stats["attention_pairs"],
            3 * block.last_attention_stats["active_queries"] * 5,
        )
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_imf_relational_dissimilarity_dense_sparse_equivalence_and_gradient(self):
        from common.models.metric_attention import grid_neighborhood, spatial_attention

        torch.manual_seed(903)
        query = torch.randn(1, 2, 16, 4, requires_grad=True)
        key = torch.randn(1, 2, 16, 4, requires_grad=True)
        value = torch.randn(1, 2, 16, 6, requires_grad=True)
        descriptors = torch.rand(1, 16, 4, requires_grad=True)
        descriptor_validity = torch.ones_like(descriptors)
        descriptor_validity[:, 5, 2] = 0
        reliability = torch.rand(1, 16)
        shared = dict(
            grid_shape=(4, 4),
            position_bias_mode="none",
            imf_descriptors=descriptors,
            imf_validity=descriptor_validity,
            imf_reliability=reliability,
            imf_scales=(0.25, 0.25, 0.25, 0.25),
            imf_weights=(1.0, 1.0, 0.5, 0.5),
            imf_bias_weight=0.4,
        )
        dense, _ = spatial_attention(query, key, value, mode="dense", **shared)
        sparse, stats = spatial_attention(
            query,
            key,
            value,
            mode="sparse",
            neighbor_indices=grid_neighborhood(4, 4, 16),
            **shared,
        )
        self.assertTrue(torch.allclose(dense, sparse, atol=1e-6, rtol=1e-6))
        self.assertEqual(stats["imf_dissimilarity_pairs"], 2 * 16 * 16)
        sparse.square().mean().backward()
        self.assertTrue(torch.isfinite(descriptors.grad).all())

    def test_optional_imf_assistance_controls_integrate_without_changing_dense_outputs(self):
        from IMF_HTransformer.model import IMFHTransformer

        model = IMFHTransformer(
            profile="tiny",
            overrides={
                "attention": {
                    "mode": "sparse",
                    "stage3_mode": "sparse",
                    "stage4_mode": "dense",
                    "stage3_sparse_neighbors": 8,
                    "stage3_use_map_context": False,
                    "stage4_use_map_context": False,
                    "stage3_active_query_fraction": 0.25,
                    "stage4_active_query_fraction": 0.5,
                    "position_bias": "metric",
                    "imf_dissimilarity": {
                        "enabled": True,
                        "weight": 0.2,
                        "scales": [0.25, 0.25, 0.25, 0.25],
                        "weights": [1.0, 1.0, 0.5, 0.5],
                    },
                },
            },
        )
        analytical = torch.rand(1, 28, 32, 32)
        analytical[:, 22] = 0.5
        analytical[:, 27] = 0.8
        validity = torch.ones_like(analytical)
        state = torch.zeros(1, len(STATE_NAMES))
        state_validity = torch.ones_like(state)
        relation = torch.zeros(1, len(RELATIONAL_NAMES))
        relation_validity = torch.ones_like(relation)
        output = model(
            analytical,
            validity,
            state,
            state_validity,
            relation,
            relation_validity,
        )
        self.assertEqual(tuple(output["hazard_logits"].shape), (1, 1, 32, 32))
        self.assertLess(output["aux"]["attention"]["stage3"]["active_queries"], 256)
        self.assertGreater(output["aux"]["attention"]["stage3"]["imf_dissimilarity_pairs"], 0)
        output["hazard_logits"].mean().backward()
        self.assertTrue(
            all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        )

    def test_optional_attention_diagnostics_and_figure_smoke(self):
        from tools.imf_figure_diagnostics import attention_diagnostics, bayesian_evidence_demo
        from tools.generate_figures import _map_figure

        config = self.config("imf_htransformer")
        config["model"]["stage3_depth"] = 1
        config["model"]["stage4_depth"] = 1
        config["model"]["attention"] = {
            "mode": "sparse",
            "selection": "grid",
            "position_bias": "none",
            "neighbors": 8,
            "capture_diagnostics": True,
        }
        model = build_model(config).eval()
        with torch.inference_mode():
            model(
                torch.randn(1, 28, 32, 32),
                torch.ones(1, 28, 32, 32),
                torch.zeros(1, 13),
                torch.ones(1, 13),
                torch.zeros(1, len(RELATIONAL_NAMES)),
                torch.ones(1, len(RELATIONAL_NAMES)),
                torch.zeros(1, 32, len(CANDIDATE_FEATURE_NAMES)),
                torch.zeros(1, 32),
            )
        debug = model.backbone.last_attention_debug["stage3"][0]
        self.assertEqual(tuple(debug["weights"].shape), (1, 4, 256, 8))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "attention.npz"
            arrays = {"sample_id": np.asarray("fixture")}
            for stage, blocks in model.backbone.last_attention_debug.items():
                for index, block in enumerate(blocks):
                    if block:
                        for name, value in block.items():
                            if isinstance(value, torch.Tensor):
                                arrays[f"{stage}_block{index}_{name}"] = value.numpy()
                            elif value is not None:
                                arrays[f"{stage}_block{index}_{name}"] = np.asarray(value)
            np.savez_compressed(path, **arrays)
            rendered = attention_diagnostics(path)
            self.assertGreater(rendered.width, 1000)
            rendered.save(Path(temp_dir) / "attention.png")
            map_path = Path(temp_dir) / "map.json"
            map_path.write_text(
                json.dumps(
                    {
                        "map_entities": [
                            {
                                "entity_id": "fixture",
                                "semantic_class": "cone",
                                "xyz_m": [0.0, 0.0, 1.0],
                                "valid": True,
                                "covariance_valid": True,
                                "position_covariance_m2": (np.eye(3) * 0.01).tolist(),
                            }
                        ]
                    }
                )
            )
            _map_figure(map_path, Path(temp_dir))
            self.assertTrue((Path(temp_dir) / "map_local_semantic.svg").is_file())
        self.assertGreater(bayesian_evidence_demo().width, 1000)

    def test_resource_estimator_counts_sparse_executed_attention_pairs(self):
        image = torch.randn(1, 24, 8, 8)
        dense = HTransformerBlock(24, (8, 8), 3, attention_mode="dense")
        sparse = HTransformerBlock(
            24,
            (8, 8),
            3,
            attention_mode="sparse",
            sparse_neighbors=8,
        )
        dense_macs = estimate_macs(dense, (image,))
        sparse_macs = estimate_macs(sparse, (image,))
        self.assertLess(sparse_macs, dense_macs)
        self.assertEqual(sparse.last_attention_stats["attention_pairs"], 3 * 64 * 8)

    def test_htransformer_spatial_values_are_mbconv_and_context_is_pointwise_masked(self):
        block = HTransformerBlock(
            24,
            (8, 8),
            3,
            attention_mode="sparse",
            position_bias_mode="metric",
            sparse_neighbors=8,
            context_features=len(CANDIDATE_FEATURE_NAMES),
        )
        spatial = torch.randn(1, 24, 8, 8, requires_grad=True)
        positions = torch.randn(1, 8, 8, 3)
        position_validity = torch.ones(1, 8, 8)
        candidates = torch.randn(1, 4, len(CANDIDATE_FEATURE_NAMES), requires_grad=True)
        candidate_positions = torch.randn(1, 4, 3)
        candidate_validity = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        output = block(
            spatial,
            positions,
            position_validity,
            candidates,
            candidate_positions,
            candidate_validity,
        )
        output.mean().backward()
        self.assertEqual(tuple(output.shape), tuple(spatial.shape))
        self.assertEqual(block.last_attention_stats["attention_pairs"], 1 * 3 * 64 * (8 + 4))
        self.assertTrue(torch.isfinite(candidates.grad).all())
        self.assertIsInstance(block.value_local, MBConv)
        self.assertIsInstance(block.context_key_encoder, torch.nn.Linear)
        self.assertIsInstance(block.context_value_encoder, torch.nn.Linear)
        self.assertFalse(hasattr(block, "value"))

    def test_candidate_projection_values_and_map_visibility_gate(self):
        from common.models.metric_attention import positions_from_candidate_projection

        candidate_grid = torch.tensor([[[16.0, 16.0], [16.0, 16.0]]])
        depth = torch.tensor([[2.0, 3.0]])
        projection_validity = torch.ones(1, 2)
        visibility = torch.tensor([[int(Visibility.VISIBLE), int(Visibility.OCCLUDED)]])
        calibration = torch.tensor([[32.0, 32.0, 20.0, 20.0, 15.5, 15.5]])
        position, mask = positions_from_candidate_projection(
            candidate_grid,
            depth,
            projection_validity,
            visibility,
            calibration,
        )
        self.assertEqual(mask.tolist(), [[1.0, 0.0]])
        self.assertTrue(torch.allclose(position[0, 0], torch.tensor([0.05, 0.05, 2.0])))

    def test_config_inheritance_and_checkpoint_contract_rejection(self):
        from common.runtime import load_config

        config = load_config("IMF_HTransformer/configs/family_fusion_metric_sparse_attention.yaml")
        self.assertEqual(config["model"]["family"], "imf_htransformer")
        self.assertEqual(config["model"]["attention"]["mode"], "sparse")
        model = build_model(self.config("cnn"))
        path = self.root / "old_contract.pt"
        save_checkpoint(path, model, None, None, 1, 1, {}, {}, "manifest")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.pop("architecture_contract")
        torch.save(payload, path)
        with self.assertRaisesRegex(ValueError, "architecture/map contract"):
            load_checkpoint(path, build_model(self.config("cnn")))

    def test_covariance_propagation_and_mahalanobis_are_finite(self):
        intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0}
        pixel_cov = pixel_depth_covariance(
            (60, 55), 3.0, intrinsics, pixel_sigma=1.0, depth_sigma_m=0.1
        )
        pose_cov = approximate_pose_covariance(0.8)
        mapped_cov = propagate_camera_to_map((0.3, 0.15, 3.0), pixel_cov, np.eye(4), pose_cov)
        self.assertIsNotNone(stabilize_covariance(mapped_cov))
        self.assertTrue(np.allclose(mapped_cov, mapped_cov.T, atol=1e-9))
        self.assertGreaterEqual(np.linalg.eigvalsh(mapped_cov).min(), 0.0)
        self.assertLess(
            mahalanobis_squared((0.1, 0, 0), np.eye(3) * 0.1),
            mahalanobis_squared((1.0, 0, 0), np.eye(3) * 0.1),
        )
        self.assertIsNone(stabilize_covariance(np.diag([1.0, 1.0, -1.0])))

    def test_mahalanobis_association_uses_uncertainty_and_class(self):
        covariance = np.eye(3) * 0.001
        broad = np.eye(3) * 0.5
        uncertain = PersistentSemanticMap(association_m=0.75, association_mode="mahalanobis")
        first = uncertain.update(
            SemanticObservation(
                (0.0, 0.0, 0.0),
                {"rock": 1.0},
                0.9,
                1.0,
                position_covariance_m2=covariance.tolist(),
                covariance_source="test",
            )
        )
        merged = uncertain.update(
            SemanticObservation(
                (0.5, 0.0, 0.0),
                {"rock": 1.0},
                0.9,
                2.0,
                position_covariance_m2=broad.tolist(),
                covariance_source="test",
            )
        )
        self.assertEqual(len(uncertain.entities), 1)
        self.assertEqual(merged.entity_id, first.entity_id)
        self.assertGreater(uncertain.association_stats["mahalanobis"], 0)

        precise = PersistentSemanticMap(association_m=0.75, association_mode="mahalanobis")
        precise.update(
            SemanticObservation(
                (0.0, 0.0, 0.0),
                {"rock": 1.0},
                0.9,
                1.0,
                position_covariance_m2=covariance.tolist(),
            )
        )
        precise.update(
            SemanticObservation(
                (0.5, 0.0, 0.0),
                {"rock": 1.0},
                0.9,
                2.0,
                position_covariance_m2=covariance.tolist(),
            )
        )
        self.assertEqual(len(precise.entities), 2)
        precise.update(SemanticObservation((0.0, 0.0, 0.0), {"cone": 1.0}, 0.9, 3.0))
        self.assertEqual(len(precise.entities), 3)

    def test_bayesian_semantic_evidence_is_bounded_and_contradictions_change_posterior(self):
        local_map = PersistentSemanticMap(
            evidence_correlation_time_s=1.0,
            max_semantic_evidence=4.0,
        )
        entity = local_map.update(
            SemanticObservation(
                (0.0, 0.0, 1.0),
                {"rock": 0.9, "track": 0.1},
                0.95,
                0.0,
            )
        )
        initial = entity.class_probabilities["rock"]
        for index in range(1, 40):
            entity = local_map.update(
                SemanticObservation(
                    (0.0, 0.0, 1.0),
                    {"rock": 0.9, "track": 0.1},
                    0.95,
                    index * 0.05,
                )
            )
        self.assertLessEqual(entity.semantic_support, 4.0 + 1e-6)
        before_contradiction = entity.class_probabilities["rock"]
        entity = local_map.update(
            SemanticObservation(
                (0.0, 0.0, 1.0),
                {"rock": 0.51, "track": 0.49},
                0.95,
                4.0,
            )
        )
        self.assertLess(entity.class_probabilities["rock"], before_contradiction)
        self.assertGreater(initial, 0.5)
        self.assertTrue(np.isfinite(entity.semantic_entropy))

    def test_soft_contradictory_semantics_associate_but_disjoint_classes_do_not(self):
        local_map = PersistentSemanticMap(
            association_m=0.5,
            semantic_overlap_threshold=0.1,
            evidence_correlation_time_s=0.1,
        )
        entity = local_map.update(
            SemanticObservation(
                (0.0, 0.0, 1.0),
                {"rock": 1.0},
                0.9,
                0.0,
            )
        )
        initial_rock_probability = entity.class_probabilities["rock"]

        # The new prediction has a different argmax, but enough shared class
        # posterior to be credible evidence about the same spatial entity.
        entity = local_map.update(
            SemanticObservation(
                (0.02, 0.0, 1.0),
                {"rock": 0.25, "cone": 0.75},
                0.9,
                2.0,
            )
        )
        self.assertEqual(len(local_map.entities), 1)
        self.assertLess(entity.class_probabilities["rock"], initial_rock_probability)
        self.assertGreater(entity.class_probabilities["cone"], 0.0)

        # A disjoint posterior stays a separate class-consistent map entity.
        local_map.update(
            SemanticObservation(
                (0.03, 0.0, 1.0),
                {"grass": 1.0},
                0.9,
                4.0,
            )
        )
        self.assertEqual(len(local_map.entities), 2)
        self.assertGreater(local_map.association_stats["semantic_incompatible"], 0)

    def test_configured_terrain_class_flip_updates_one_bayesian_entity(self):
        local_map = PersistentSemanticMap(
            association_m=0.5,
            semantic_overlap_threshold=0.1,
            semantic_association_groups=(("background", "grass", "track"),),
            evidence_correlation_time_s=0.1,
        )
        entity = local_map.update(SemanticObservation((1.0, 2.0, 0.0), {"track": 1.0}, 0.9, 0.0))
        initial_track = entity.class_probabilities["track"]
        local_map.update(SemanticObservation((1.01, 2.0, 0.0), {"grass": 1.0}, 0.9, 2.0))
        entity = local_map.entities[0]
        self.assertEqual(len(local_map.entities), 1)
        self.assertLess(entity.class_probabilities["track"], initial_track)
        self.assertGreater(entity.class_probabilities["grass"], 0.0)
        for index in range(2, 6):
            local_map.update(
                SemanticObservation((1.01, 2.0, 0.0), {"grass": 1.0}, 0.9, index * 2.0)
            )
        relation = compute_relations(
            local_map,
            timestamp_s=12.0,
            position_m=(0.0, 2.0, 0.0),
            yaw_rad=0.0,
            semantic_rules={"track": {"fly_allowed": False, "land_allowed": False}},
        )
        self.assertEqual(relation.validity[RELATIONAL_NAMES.index("nearest_track_m")], 1.0)

    def test_map_schema_v3_persistence_and_v1_v2_migration(self):
        local_map = PersistentSemanticMap()
        entity = local_map.update(
            SemanticObservation(
                (1.0, 2.0, 3.0),
                {"rock": 1.0},
                0.8,
                1.0,
                position_covariance_m2=(np.eye(3) * 0.1).tolist(),
                covariance_source="test",
            )
        )
        path = self.root / "schema_v3.json"
        local_map.save(path)
        restored = PersistentSemanticMap.load(path)
        self.assertEqual(restored.entities[0].position_covariance_m2, entity.position_covariance_m2)
        self.assertEqual(restored.semantic_overlap_threshold, local_map.semantic_overlap_threshold)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("semantic_overlap_threshold")
        payload.pop("semantic_association_groups", None)
        payload["schema"] = "imagination-semantic-map-v2"
        old_v2 = self.root / "schema_v2_without_overlap_setting.json"
        old_v2.write_text(json.dumps(payload), encoding="utf-8")
        self.assertEqual(PersistentSemanticMap.load(old_v2).semantic_overlap_threshold, 0.1)
        payload["schema"] = "imagination-semantic-map-v1"
        for row in payload["entities"]:
            for key in (
                "position_covariance_m2",
                "covariance_valid",
                "covariance_source",
                "semantic_dirichlet_alpha",
                "semantic_evidence_valid",
            ):
                row.pop(key, None)
        legacy = self.root / "schema_v1.json"
        legacy.write_text(json.dumps(payload), encoding="utf-8")
        migrated = PersistentSemanticMap.load(legacy)
        self.assertFalse(migrated.entities[0].covariance_valid)
        self.assertFalse(migrated.entities[0].semantic_evidence_valid)
        self.assertIn("v1_migration", migrated.entities[0].covariance_source)

    def test_egocentric_roundtrip_and_yaw_invariance(self):
        origin = np.array([2.0, -1.0, 0.5])
        point = np.array([5.0, 3.0, 1.5])
        ego = map_to_egocentric(point, origin, np.pi / 2)
        self.assertTrue(np.allclose(egocentric_to_map(ego, origin, np.pi / 2), point))
        self.assertTrue(
            np.allclose(
                map_to_egocentric(np.array([-3.0, 4.0, 1.5]), np.array([1.0, 1.0, 0.5]), np.pi / 2),
                map_to_egocentric(point, origin, 0.0),
            )
        )

    def test_visibility_distinguishes_camera_visibility_states(self):
        intrinsics = {"width": 32.0, "height": 32.0, "fx": 20.0, "fy": 20.0, "cx": 15.5, "cy": 15.5}
        pose = np.eye(4)
        self.assertEqual(
            classify_map_point((0, 0, -1), pose, intrinsics)[0], Visibility.BEHIND_CAMERA
        )
        self.assertEqual(
            classify_map_point((100, 0, 2), pose, intrinsics)[0], Visibility.OUTSIDE_FOV
        )
        depth = np.full((32, 32), 1.0)
        validity = np.ones((32, 32))
        self.assertEqual(
            classify_map_point((0, 0, 2), pose, intrinsics, depth, validity)[0],
            Visibility.OCCLUDED,
        )

    def test_all_four_models_share_outputs_and_backpropagate(self):
        state = torch.zeros(1, 13)
        state_validity = torch.ones_like(state)
        relation = torch.zeros(1, len(RELATIONAL_NAMES))
        relation_validity = torch.zeros_like(relation)
        candidates = torch.zeros(1, MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES))
        candidate_validity = torch.zeros(1, MAX_CANDIDATES)
        batch = next(
            iter(
                torch.utils.data.DataLoader(
                    ImaginationDataset(self.manifest, "train", "analytical"), batch_size=1
                )
            )
        )
        move_targets(batch, torch.device("cpu"))

        for family in FAMILIES:
            model = build_model(self.config(family))
            if family == "imf_htransformer":
                outputs = model(
                    torch.randn(1, 28, 32, 32),
                    torch.ones(1, 28, 32, 32),
                    state,
                    state_validity,
                    relation,
                    relation_validity,
                    candidates,
                    candidate_validity,
                    calibration=torch.tensor([[32.0, 32.0, 24.0, 24.0, 15.5, 15.5]]),
                    depth_scale_m=torch.tensor([20.0]),
                )
            else:
                outputs = model(
                    torch.randn(1, 3, 256, 256),
                    state,
                    state_validity,
                    relation,
                    relation_validity,
                    candidates,
                    candidate_validity,
                )
            self.assertEqual(tuple(outputs["hazard_logits"].shape), (1, 1, 32, 32))
            self.assertEqual(tuple(outputs["landing_logits"].shape), (1, 1, 32, 32))
            self.assertEqual(
                tuple(outputs["semantic_logits"].shape), (1, len(SEMANTIC_CLASSES), 32, 32)
            )
            self.assertEqual(tuple(outputs["poi"]["class_logits"].shape), (1, 7, 32, 32))
            self.assertEqual(tuple(outputs["scene_risk_logits"].shape), (1, len(SCENE_CLASSES)))
            self.assertEqual(tuple(outputs["candidate"]["risk_logits"].shape), (1, 32, 3))
            losses = multitask_loss(outputs, batch, {"overlap_weight": 0.0})
            self.assertTrue(torch.isfinite(losses["total"]))
            losses["total"].backward()
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.heads.hazard.parameters())
            )
            self.assertTrue(
                any(parameter.grad is not None for parameter in model.heads.scene.parameters())
            )

    def test_research_operation_allocation_contract(self):
        """Keep the four comparisons distinct and experimental IMF capacity opt-in."""
        from common.models.frontends import ConvolutionalBackbone, RGBFrontEnd
        from common.models.vit import ViTBackbone

        configs = {
            family: {
                **self.config(family),
                "model": {
                    **self.config(family)["model"],
                    "stage3_depth": 1,
                    "stage4_depth": 1,
                },
            }
            for family in FAMILIES
        }
        models = {family: build_model(config) for family, config in configs.items()}

        self.assertIsInstance(models["cnn"].frontend, RGBFrontEnd)
        self.assertIsInstance(models["cnn"].backbone, ConvolutionalBackbone)
        self.assertIsInstance(models["cnn_vit"].frontend, RGBFrontEnd)
        self.assertIsInstance(models["cnn_vit"].backbone, ViTBackbone)
        self.assertIsInstance(models["cnn_htransformer"].frontend, RGBFrontEnd)
        self.assertIsInstance(models["cnn_htransformer"].backbone, HTransformerBackbone)

        imf = models["imf_htransformer"]
        self.assertFalse(hasattr(imf, "frontend"))
        self.assertEqual(tuple(imf.family_adapters), ("appearance", "motion", "geometry"))
        self.assertEqual(model_mode("imf_htransformer"), "analytical")
        self.assertTrue(all(model_mode(family) == "rgb_current" for family in FAMILIES[:-1]))

        imf_blocks = (imf.backbone.stage3[0], imf.backbone.stage4[0])
        rgb_h_blocks = (
            models["cnn_htransformer"].backbone.stage3[0],
            models["cnn_htransformer"].backbone.stage4[0],
        )
        for block in (*imf_blocks, *rgb_h_blocks):
            self.assertIsInstance(block.value_local, MBConv)
            self.assertFalse(hasattr(block, "value"))
            self.assertIsNot(block.query, block.key)

        # IMF map context is pointwise and unordered. The RGB HTransformer
        # does not receive projected map tokens inside its attention blocks.
        for block in imf_blocks:
            self.assertIsInstance(block.context_key_encoder, torch.nn.Linear)
            self.assertIsInstance(block.context_value_encoder, torch.nn.Linear)
            self.assertEqual(block.qk_channels, block.channels)
            self.assertEqual(block.active_query_fraction, 1.0)
            self.assertEqual(block.imf_dissimilarity_weight, 0.0)
        for block in rgb_h_blocks:
            self.assertIsNone(block.context_key_encoder)
            self.assertIsNone(block.context_value_encoder)

    def test_binary_and_categorical_metrics(self):
        metric = BinaryCounts("hazard")
        metric.update(
            torch.tensor([[[[4.0, -4.0, -4.0, 4.0]]]]),
            torch.tensor([[[[1.0, 1.0, 0.0, 0.0]]]]),
            torch.ones(1, 1, 1, 4),
        )
        self.assertEqual(metric.result()["hazard_false_negative"], 1)
        self.assertEqual(metric.result()["hazard_false_positive"], 1)
        categorical = CategoricalMetrics(("a", "b"))
        categorical.update(torch.tensor([[[[0.0]], [[2.0]]]]), torch.tensor([[[1]]]))
        self.assertEqual(categorical.result("test")["test_b_iou"], 1.0)

    def test_map_projection_association_and_unresolved(self):
        camera = {"width": 640, "height": 480, "fx": 440.0, "fy": 440.0, "cx": 319.5, "cy": 239.5}
        depth = np.zeros((32, 32), np.float32)
        validity = np.zeros_like(depth)
        depth[16, 16] = 0.5
        validity[16, 16] = 1.0
        detection = Detection2D("poi", "rock", (16.0, 16.0), 4, 0.9)
        feature = register_detection(detection, depth, validity, 4.0, camera, np.eye(4), 1.0)
        self.assertTrue(feature.resolved)
        self.assertAlmostEqual(feature.map_xyz_m[2], 2.0)
        local_map = PersistentSemanticMap()
        first = local_map.update_region(
            "rock", feature.map_xyz_m, feature.radius_m, feature.probability, 1.0
        )
        second = local_map.update_region(
            "rock", feature.map_xyz_m, feature.radius_m, feature.probability, 2.0
        )
        self.assertEqual(first.entity_id, second.entity_id)
        self.assertEqual(second.observation_count, 2)
        unresolved = register_detection(
            detection, depth, np.zeros_like(validity), 4.0, camera, np.eye(4), 3.0
        )
        self.assertFalse(unresolved.resolved)
        self.assertIsNone(unresolved.map_xyz_m)

    def test_covariance_free_observation_does_not_erase_supported_covariance(self):
        local_map = PersistentSemanticMap(association_m=1.0)
        covariance = np.diag([0.01, 0.02, 0.03]).tolist()
        entity = local_map.update(
            SemanticObservation(
                (1.0, 2.0, 3.0),
                {"cone": 1.0},
                0.9,
                1.0,
                position_covariance_m2=covariance,
                covariance_source="first_order_approximation",
            )
        )
        before_xyz = np.asarray(entity.xyz_m).copy()
        before_covariance = np.asarray(entity.position_covariance_m2).copy()
        local_map.update(SemanticObservation((1.1, 2.0, 3.0), {"cone": 1.0}, 0.9, 2.0))
        self.assertTrue(entity.covariance_valid)
        self.assertTrue(np.allclose(entity.xyz_m, before_xyz))
        self.assertTrue(np.allclose(entity.position_covariance_m2, before_covariance))

    def test_grass_regions_survive_as_merged_terrain_tokens(self):
        local_map = PersistentSemanticMap()
        local_map.update(SemanticObservation((1.0, 0.0, 0.0), {"grass": 1.0}, 0.9, 1.0))
        local_map.update(SemanticObservation((1.1, 0.0, 0.0), {"grass": 1.0}, 0.9, 1.1))
        relations = compute_relations(local_map, timestamp_s=2.0, yaw_rad=0.0)
        valid_rows = np.asarray(relations.candidates)[np.asarray(relations.candidate_validity) > 0]
        terrain_index = CANDIDATE_FEATURE_NAMES.index("type_terrain_region")
        self.assertEqual(len(valid_rows), 1)
        self.assertEqual(float(valid_rows[0, terrain_index]), 1.0)

    def test_merged_region_positions_remain_absolute_away_from_map_origin(self):
        local_map = PersistentSemanticMap(association_m=0.05)
        local_map.update(SemanticObservation((11.1, 20.0, 2.0), {"grass": 1.0}, 0.9, 1.0))
        local_map.update(SemanticObservation((11.2, 20.0, 2.0), {"grass": 1.0}, 0.9, 1.1))
        result = compute_relations(
            local_map,
            timestamp_s=2.0,
            position_m=(10.0, 20.0, 2.0),
            yaw_rad=np.pi / 2,
            candidate_merge_cell_m=0.5,
            merge_region_classes=("grass",),
        )
        valid = np.asarray(result.candidate_validity) > 0
        terrain_index = CANDIDATE_FEATURE_NAMES.index("type_terrain_region")
        candidates = np.asarray(result.candidates)[valid]
        map_xyz = np.asarray(result.candidate_map_xyz)[valid]
        self.assertEqual(len(candidates), 1)
        self.assertTrue(np.allclose(map_xyz[0], (11.15, 20.0, 2.0)))
        self.assertTrue(np.allclose(candidates[0, :3], (0.0, -1.15, 0.0)))
        recovered = egocentric_to_map(candidates[0, :3], (10.0, 20.0, 2.0), np.pi / 2)
        self.assertTrue(np.allclose(recovered, map_xyz[0]))

    def test_map_vector_transforms_rotate_without_translation(self):
        vector = np.asarray([1.0, 0.0])
        self.assertTrue(np.allclose(map_vector_to_egocentric(vector, 0.0), (1.0, 0.0)))
        self.assertTrue(
            np.allclose(map_vector_to_egocentric(vector, np.pi / 2), (0.0, -1.0), atol=1e-7)
        )
        self.assertTrue(
            np.allclose(map_vector_to_egocentric(vector, -np.pi / 2), (0.0, 1.0), atol=1e-7)
        )
        arbitrary = map_vector_to_egocentric((0.3, -0.7, 2.0), 0.83)
        self.assertTrue(np.allclose(egocentric_vector_to_map(arbitrary, 0.83), (0.3, -0.7, 2.0)))

    def test_candidate_clearances_are_candidate_local_and_drift_is_egocentric(self):
        local_map = PersistentSemanticMap(association_m=0.4)
        local_map.update(
            SemanticObservation((2.0, 0.0, 0.0), {"landing": 1.0}, 0.9, 1.0, extent_m=0.8)
        )
        local_map.update(
            SemanticObservation((9.0, 0.0, 0.0), {"landing": 1.0}, 0.9, 1.0, extent_m=0.8)
        )
        local_map.update(
            SemanticObservation((0.0, 0.0, 0.0), {"track": 1.0}, 0.9, 1.0, extent_m=1.0)
        )
        local_map.update(
            SemanticObservation((20.0, 0.0, 0.0), {"rock": 1.0}, 0.9, 1.0, extent_m=0.5)
        )
        local_map.update(
            SemanticObservation((2.0, 1.5, 0.0), {"cone": 1.0}, 0.9, 1.0, extent_m=0.1)
        )
        rules = {
            "track": {"fly_allowed": False, "land_allowed": False},
            "rock": {"land_allowed": False},
            "cone": {"land_allowed": False},
        }
        result = compute_relations(
            local_map,
            timestamp_s=1.1,
            position_m=(0.0, 0.0, 0.0),
            velocity_xy_mps=(1.0, 0.0),
            yaw_rad=0.0,
            landing_time_s=2.0,
            vehicle_radius_m=0.2,
            safety_margin_m=0.1,
            semantic_rules=rules,
        )
        rows = np.asarray(result.candidates)
        valid = np.asarray(result.candidate_validity) > 0
        xyz = np.asarray(result.candidate_map_xyz)
        col = {name: index for index, name in enumerate(CANDIDATE_FEATURE_NAMES)}
        landing_rows = rows[:, col["type_landing"]] > 0.5
        first = rows[valid & landing_rows & np.isclose(xyz[:, 0], 2.0)][0]
        second = rows[valid & landing_rows & np.isclose(xyz[:, 0], 9.0)][0]
        self.assertLess(
            first[col["track_boundary_clearance_m"]], second[col["track_boundary_clearance_m"]]
        )
        self.assertAlmostEqual(first[col["restricted_boundary_clearance_m"]], 1.0, places=5)
        self.assertAlmostEqual(second[col["restricted_boundary_clearance_m"]], 8.0, places=5)
        self.assertGreater(
            second[col["nearest_obstacle_clearance_m"]], first[col["nearest_obstacle_clearance_m"]]
        )
        self.assertGreater(
            first[col["cone_density_2m_entities_per_m2"]],
            second[col["cone_density_2m_entities_per_m2"]],
        )
        self.assertLess(
            first[col["drifted_track_clearance_m"]], second[col["drifted_track_clearance_m"]]
        )
        self.assertAlmostEqual(first[col["free_radius_m"]], 0.8, places=5)
        self.assertAlmostEqual(second[col["free_radius_m"]], np.sqrt(51.25) - 0.3, places=5)
        # Candidate clearances are map quantities and do not change when only
        # the UAV pose/orientation changes; token egocentric coordinates do.
        moved = compute_relations(
            local_map,
            timestamp_s=1.1,
            position_m=(7.0, -3.0, 1.5),
            velocity_xy_mps=(1.0, 0.0),
            yaw_rad=0.73,
            landing_time_s=2.0,
            vehicle_radius_m=0.2,
            safety_margin_m=0.1,
            semantic_rules=rules,
        )
        moved_rows = np.asarray(moved.candidates)
        moved_valid = np.asarray(moved.candidate_validity) > 0
        moved_xyz = np.asarray(moved.candidate_map_xyz)
        moved_landing = moved_rows[:, col["type_landing"]] > 0.5
        moved_a = moved_rows[moved_valid & moved_landing & np.isclose(moved_xyz[:, 0], 2.0)][0]
        self.assertAlmostEqual(
            moved_a[col["restricted_boundary_clearance_m"]],
            first[col["restricted_boundary_clearance_m"]],
            places=6,
        )
        self.assertAlmostEqual(
            moved_a[col["nearest_obstacle_clearance_m"]],
            first[col["nearest_obstacle_clearance_m"]],
            places=6,
        )
        self.assertAlmostEqual(moved_a[col["free_radius_m"]], first[col["free_radius_m"]], places=6)
        rotated = compute_relations(
            PersistentSemanticMap(),
            timestamp_s=1.0,
            position_m=(12.0, 8.0, 0.0),
            velocity_xy_mps=(0.0, 1.0),
            yaw_rad=np.pi / 2,
            landing_time_s=2.0,
        )
        self.assertAlmostEqual(rotated.values[RELATIONAL_NAMES.index("drift_x_m")], 2.0, places=6)
        self.assertAlmostEqual(rotated.values[RELATIONAL_NAMES.index("drift_y_m")], 0.0, places=6)

    def test_candidate_verdicts_receive_state_and_relational_context(self):
        from common.models.heads import PerceptionHeads

        torch.manual_seed(81)
        heads = PerceptionHeads(16, 8).eval()
        state = torch.randn(1, 13, requires_grad=True)
        relation = torch.randn(1, len(RELATIONAL_NAMES), requires_grad=True)
        candidates = torch.randn(1, MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES))
        output = heads(
            torch.randn(1, 16, 8, 8),
            state,
            torch.ones_like(state),
            relation,
            torch.ones_like(relation),
            candidates,
            torch.ones(1, MAX_CANDIDATES),
        )
        output["candidate"]["landing_safe_logits"].sum().backward()
        self.assertIsNotNone(state.grad)
        self.assertIsNotNone(relation.grad)
        self.assertGreater(float(state.grad.abs().sum()), 0.0)
        self.assertGreater(float(relation.grad.abs().sum()), 0.0)

    def test_candidate_visibility_is_one_hot_not_ordinal(self):
        local_map = PersistentSemanticMap()
        entity = local_map.update(SemanticObservation((2.0, 0.0, 1.0), {"rock": 1.0}, 0.9, 1.0))
        result = compute_relations(
            local_map,
            timestamp_s=1.1,
            yaw_rad=0.0,
            visibility_by_entity={entity.entity_id: int(Visibility.OCCLUDED)},
        )
        candidate = np.asarray(result.candidates)[0]
        flags = [
            candidate[CANDIDATE_FEATURE_NAMES.index(name)]
            for name in (
                "visibility_unknown",
                "visibility_visible",
                "visibility_behind_camera",
                "visibility_outside_fov",
                "visibility_occluded",
                "visibility_depth_inconsistent",
            )
        ]
        self.assertEqual(flags, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def test_candidate_pruning_preserves_poi_hazard_and_landing_uncertainty(self):
        local_map = PersistentSemanticMap()
        for index in range(40):
            local_map.update(
                SemanticObservation(
                    (5.0 + index * 2.0, 5.0, 0.0),
                    {"grass": 1.0},
                    0.8,
                    1.0,
                    source_observation=f"grass-{index}",
                )
            )
        local_map.update(
            SemanticObservation(
                (0.5, 0.0, 0.0),
                {"grass": 1.0},
                0.8,
                1.0,
                position_covariance_m2=(np.eye(3) * 0.1).tolist(),
                source_observation="uncertain-near-landing",
            )
        )
        local_map.update(
            SemanticObservation(
                (0.0, 0.0, 0.0),
                {"landing": 1.0},
                0.95,
                1.0,
                extent_m=0.5,
                source_observation="landing-candidate",
            )
        )
        local_map.update(
            SemanticObservation(
                (0.0, 1.0, 0.0),
                {"poi": 1.0},
                0.95,
                1.0,
                source_observation="high-confidence-poi",
            )
        )
        local_map.update(
            SemanticObservation(
                (0.0, -1.0, 0.0),
                {"grass": 1.0},
                0.9,
                1.0,
                attributes={"hazard_probability": 0.9},
                source_observation="high-hazard-region",
            )
        )
        relations = compute_relations(local_map, timestamp_s=1.1, yaw_rad=0.0)
        valid = np.asarray(relations.candidate_validity) > 0
        mapped = np.asarray(relations.candidate_map_xyz)[valid]
        for expected in ((0.5, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)):
            self.assertTrue(any(np.allclose(row, expected) for row in mapped), expected)
        rows = np.asarray(relations.candidates)[valid]
        high_risk = CANDIDATE_FEATURE_NAMES.index("type_high_risk")
        self.assertTrue(np.any(rows[:, high_risk] == 1.0))

    def test_drift_clearance_includes_restricted_track_region(self):
        local_map = PersistentSemanticMap()
        local_map.update(
            SemanticObservation((0.0, 0.0, 0.0), {"landing": 1.0}, 0.9, 1.0, extent_m=0.5)
        )
        local_map.update(
            SemanticObservation((2.3, 0.0, 0.0), {"track": 1.0}, 0.9, 1.0, extent_m=0.4)
        )
        relation = compute_relations(
            local_map,
            timestamp_s=1.1,
            velocity_xy_mps=(1.0, 0.0),
            yaw_rad=0.0,
            landing_time_s=2.0,
            semantic_rules={"track": {"fly_allowed": False, "land_allowed": False}},
        )
        index = RELATIONAL_NAMES.index("drift_clearance_m")
        self.assertEqual(relation.validity[index], 1.0)
        self.assertLess(relation.values[index], 0.0)

    def test_semantic_map_policy_relations_and_roundtrip(self):
        local_map = PersistentSemanticMap()
        local_map.update(SemanticObservation((2.0, 0.0, 0.0), {"track": 1.0}, 0.9, 1.0, 0.5))
        local_map.update(SemanticObservation((1.0, 0.0, 0.0), {"cone": 1.0}, 0.8, 1.0, 0.2))
        rules = {"track": {"fly_allowed": False, "land_allowed": False}}
        relation = compute_relations(
            local_map, timestamp_s=1.1, semantic_rules=rules, velocity_xy_mps=(0.2, 0.0)
        )
        self.assertEqual(len(relation.values), 30)
        self.assertGreater(relation.values[RELATIONAL_NAMES.index("cone_density_2m")], 0.0)
        self.assertEqual(len(relation.candidates), 32)
        self.assertGreater(sum(relation.candidate_validity), 0.0)
        density_index = CANDIDATE_FEATURE_NAMES.index("obstacle_density_2m_entities_per_m2")
        self.assertTrue(any(row[density_index] > 0 for row in relation.candidates))
        path = self.root / "map.json"
        local_map.save(path)
        restored = PersistentSemanticMap.load(path)
        self.assertEqual(len(restored.entities), 2)
        self.assertEqual(restored.mode, "predicted_map")
        with self.assertRaises(ValueError):
            local_map.update(
                SemanticObservation((0.0, 0.0, 1.0), {"track": 1.0}, 1.0, 2.0, source="oracle")
            )

    def test_region_components_keep_separate_objects(self):
        heat = np.zeros((1, 32, 32), dtype=np.float32)
        heat[0, 2:4, 2:4] = 1.0
        heat[0, 20:22, 20:22] = 1.0
        self.assertEqual(len(extract_regions(heat, 0.5, "poi", ["rock"])), 2)

    def test_checkpoint_round_trip_keeps_rng_and_state(self):
        model = build_model(self.config("imf_htransformer"))
        optimizer = torch.optim.AdamW(model.parameters())
        path = self.root / "last.pt"
        random.seed(22)
        np.random.seed(22)
        torch.manual_seed(22)
        save_checkpoint(
            path,
            model,
            optimizer,
            None,
            1,
            5,
            self.config("imf_htransformer"),
            {"loss": 1.0},
            "manifest",
            map_policy_hash="semantic-rules-test-hash",
        )
        expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        restored = build_model(self.config("imf_htransformer"))
        checkpoint = load_checkpoint(path, restored)
        self.assertEqual(checkpoint["global_step"], 5)
        self.assertEqual(checkpoint["map_policy_hash"], "semantic-rules-test-hash")
        self.assertIn("rng_state", checkpoint)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(())))
        self.assertEqual(actual, expected)
        self.assertFalse(path.with_name("last.pt.tmp").exists())

    def test_semantic_map_loads_configured_domain_rules(self):
        from common.mapping.online import OnlinePerceptionMap

        online = OnlinePerceptionMap.from_config(Path("data/configs/semantic_rules.yaml"))
        self.assertFalse(online.semantic_rules["track"]["fly_allowed"])
        self.assertEqual(online.local_map.mode, "predicted_map")
        self.assertEqual(online.local_map.semantic_overlap_threshold, 0.1)
        self.assertIn(
            frozenset(("background", "grass", "track")),
            online.local_map.semantic_association_groups,
        )

    def test_shared_training_engine_resumes_from_saved_epoch(self):
        from common.training.engine import train

        output = self.root / "resume_run"
        config = self.config("cnn")
        config["data"] = {"manifest": str(self.manifest), "validate_files": True}
        config["training"].update(
            {
                "output": str(output),
                "max_epochs": 1,
                "batch_size": 2,
                "workers": 0,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "mixed_precision": False,
                "checkpoint_every_batches": 0,
                "early_stopping_patience": 0,
                "early_stopping_min_epochs": 1,
            }
        )
        config["losses"].update(
            {
                "hazard_positive_weight": 1.0,
                "landing_positive_weight": 1.0,
                "poi_positive_weight": [1.0] * 7,
            }
        )
        config_path = self.root / "resume.yaml"
        import yaml

        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        first = train(config_path)
        self.assertTrue(first.is_file())
        import csv

        with (output / "training_history.csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertGreater(float(row["train_epoch_seconds"]), 0.0)
        self.assertGreaterEqual(float(row["validation_seconds"]), 0.0)
        self.assertGreater(float(row["train_samples_per_second"]), 0.0)
        (output / "complete.json").unlink()
        resumed = train(config_path)
        self.assertEqual(first, resumed)
        self.assertTrue((output / "complete.json").is_file())
        last_checkpoint = output / "last.pt"
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        checkpoint["map_policy_hash"] = "stale-map-rules"
        torch.save(checkpoint, last_checkpoint)
        with self.assertRaisesRegex(ValueError, "semantic-map policy"):
            train(config_path)


if __name__ == "__main__":
    unittest.main()
