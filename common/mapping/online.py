"""Predicted-map inference loop with explicit, non-oracle geometry updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from common.mapping.projection import extract_regions, register_detection
from common.mapping.relations import compute_relations
from common.mapping.semantic_map import PersistentSemanticMap, update_from_semantic_prediction
from common.registry import POI_CLASSES, SEMANTIC_CLASSES
from common.mapping.uncertainty import approximate_pose_covariance
from common.mapping.visibility import classify_map_point, Visibility


@dataclass
class OnlineMapResult:
    outputs: dict
    mapped_semantic_points: int
    resolved_regions: int
    unresolved_regions: int
    map_size: int
    relation_values: tuple[float, ...]
    relation_validity: tuple[float, ...]
    candidate_values: tuple[tuple[float, ...], ...]
    candidate_validity: tuple[float, ...]
    candidate_feature_validity: tuple[tuple[float, ...], ...]
    candidate_map_xyz: tuple[tuple[float, float, float], ...]
    candidate_visibility: tuple[int, ...]


@dataclass
class OnlinePerceptionMap:
    """Use last frame's predicted map to condition current perception.

    Inputs describing labels are deliberately absent. Geometry may be supplied
    only as measured/IMF-produced depth with an independent validity map.
    """

    local_map: PersistentSemanticMap = field(default_factory=PersistentSemanticMap)
    semantic_rules: dict = field(default_factory=dict)
    max_local_radius_m: float = 20.0
    landing_time_s: float = 2.0
    vehicle_radius_m: float = 0.35
    safety_margin_m: float = 0.5
    mahalanobis_threshold: float = 11.34
    evidence_correlation_time_s: float = 1.0
    evidence_decay_per_second: float = 0.0
    max_semantic_evidence: float = 50.0
    pose_translation_floor_m: float = 0.01
    pose_translation_at_zero_m: float = 0.5
    pose_rotation_floor_rad: float = 0.002
    pose_rotation_at_zero_rad: float = 0.15
    observation_pixel_sigma: float = 1.0
    depth_sigma_base_m: float = 0.02
    depth_sigma_relative: float = 0.01
    association_mode: str = "mahalanobis"
    semantic_fusion: str = "bayesian"
    semantic_overlap_threshold: float = 0.1
    use_uncertainty: bool = True
    use_visibility: bool = True
    canonicalization: str = "egocentric"
    candidate_merge_cell_m: float = 0.5
    merge_region_classes: tuple[str, ...] = ("track", "grass")
    candidate_importance_weights: dict = field(default_factory=dict)
    protected_poi_threshold: float = 0.6
    protected_hazard_threshold: float = 0.5
    semantic_rule_probability_threshold: float = 0.1
    protected_uncertainty_sigma_m: float = 0.25
    landing_protection_radius_m: float = 1.0
    previous_position_m: np.ndarray | None = None
    previous_timestamp_s: float | None = None

    @classmethod
    def from_config(
        cls, path: str | Path, *, mode: str = "predicted_map", overrides: dict | None = None
    ):
        """Load domain rules and bounded-map/drift settings from the shared YAML."""
        import yaml

        with Path(path).open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        map_config = config.get("map", {})
        map_config = {**map_config, **(overrides or {})}
        relationship_config = config.get("relationships", {})
        local_map = PersistentSemanticMap(
            association_m=float(map_config.get("association_distance_m", 0.75)),
            voxel_m=float(map_config.get("voxel_size_m", 0.2)),
            max_entities=int(map_config.get("max_entities", 4096)),
            max_age_s=float(map_config.get("max_age_seconds", 120.0)),
            mode=mode,
            mahalanobis_threshold=float(map_config.get("mahalanobis_threshold_squared", 11.34)),
            evidence_correlation_time_s=float(
                map_config.get("evidence_correlation_time_seconds", 1.0)
            ),
            evidence_decay_per_second=float(map_config.get("evidence_decay_per_second", 0.0)),
            max_semantic_evidence=float(map_config.get("maximum_semantic_evidence", 50.0)),
            association_mode=str(map_config.get("association_mode", "mahalanobis")),
            semantic_fusion=str(map_config.get("semantic_fusion", "bayesian")),
            semantic_overlap_threshold=float(map_config.get("semantic_overlap_threshold", 0.1)),
            semantic_association_groups=map_config.get("semantic_association_groups", ()),
        )
        return cls(
            local_map=local_map,
            semantic_rules=config.get("semantic_rules", {}),
            max_local_radius_m=float(map_config.get("max_local_radius_m", 20.0)),
            landing_time_s=float(relationship_config.get("landing_time_seconds", 2.0)),
            vehicle_radius_m=float(relationship_config.get("vehicle_radius_m", 0.35)),
            safety_margin_m=float(relationship_config.get("safety_margin_m", 0.5)),
            mahalanobis_threshold=float(map_config.get("mahalanobis_threshold_squared", 11.34)),
            evidence_correlation_time_s=float(
                map_config.get("evidence_correlation_time_seconds", 1.0)
            ),
            evidence_decay_per_second=float(map_config.get("evidence_decay_per_second", 0.0)),
            max_semantic_evidence=float(map_config.get("maximum_semantic_evidence", 50.0)),
            pose_translation_floor_m=float(map_config.get("pose_translation_sigma_floor_m", 0.01)),
            pose_translation_at_zero_m=float(
                map_config.get("pose_translation_sigma_at_zero_support_m", 0.5)
            ),
            pose_rotation_floor_rad=float(map_config.get("pose_rotation_sigma_floor_rad", 0.002)),
            pose_rotation_at_zero_rad=float(
                map_config.get("pose_rotation_sigma_at_zero_support_rad", 0.15)
            ),
            observation_pixel_sigma=float(map_config.get("observation_pixel_sigma", 1.0)),
            depth_sigma_base_m=float(map_config.get("depth_sigma_base_m", 0.02)),
            depth_sigma_relative=float(map_config.get("depth_sigma_relative", 0.01)),
            association_mode=str(map_config.get("association_mode", "mahalanobis")),
            semantic_fusion=str(map_config.get("semantic_fusion", "bayesian")),
            semantic_overlap_threshold=float(map_config.get("semantic_overlap_threshold", 0.1)),
            use_uncertainty=bool(map_config.get("use_uncertainty", True)),
            use_visibility=bool(map_config.get("use_visibility", True)),
            canonicalization=str(map_config.get("canonicalization", "egocentric")),
            candidate_merge_cell_m=float(map_config.get("candidate_merge_cell_m", 0.5)),
            merge_region_classes=tuple(map_config.get("merge_region_classes", ("track", "grass"))),
            candidate_importance_weights=dict(map_config.get("candidate_importance_weights", {})),
            protected_poi_threshold=float(map_config.get("protected_poi_threshold", 0.6)),
            protected_hazard_threshold=float(map_config.get("protected_hazard_threshold", 0.5)),
            semantic_rule_probability_threshold=float(
                map_config.get("semantic_rule_probability_threshold", 0.1)
            ),
            protected_uncertainty_sigma_m=float(
                map_config.get("protected_uncertainty_sigma_m", 0.25)
            ),
            landing_protection_radius_m=float(map_config.get("landing_protection_radius_m", 1.0)),
        )

    def process(
        self,
        model,
        *,
        family: str,
        visual_input: torch.Tensor,
        validity: torch.Tensor | None,
        vehicle_state: torch.Tensor,
        state_validity: torch.Tensor,
        timestamp_s: float,
        intrinsics: dict[str, float],
        camera_to_local_map: np.ndarray | None,
        depth_normalized: np.ndarray | None,
        depth_validity: np.ndarray | None,
        depth_scale_m: float | None,
        device: torch.device,
        geometry_confidence: float | None = None,
    ) -> OnlineMapResult:
        """Infer from prior map context, then update map from current predictions."""
        if self.local_map.mode != "predicted_map":
            raise ValueError("Deployment inference requires mode='predicted_map'")

        pose = (
            None
            if camera_to_local_map is None
            else np.asarray(camera_to_local_map, dtype=np.float64)
        )
        pose_valid = pose is not None and pose.shape == (4, 4) and np.isfinite(pose).all()
        position = pose[:3, 3] if pose_valid else np.zeros(3, dtype=np.float64)
        state = vehicle_state.to(device).reshape(1, -1)
        state_mask = state_validity.to(device).reshape(1, -1)
        velocity = None
        if (
            pose_valid
            and self.previous_position_m is not None
            and timestamp_s > self.previous_timestamp_s
        ):
            velocity = (position[:2] - self.previous_position_m[:2]) / (
                timestamp_s - self.previous_timestamp_s
            )

        relation_map = self.local_map if pose_valid else PersistentSemanticMap()
        yaw = None
        if state.shape[-1] > 8 and bool(state_mask[0, 8] > 0):
            # Dataset/state contract stores yaw normalized by pi.
            yaw = float(state[0, 8].item()) * float(np.pi)
        pose_covariance = approximate_pose_covariance(
            geometry_confidence,
            translation_floor_m=self.pose_translation_floor_m,
            translation_at_zero_m=self.pose_translation_at_zero_m,
            rotation_floor_rad=self.pose_rotation_floor_rad,
            rotation_at_zero_rad=self.pose_rotation_at_zero_rad,
        )
        visibility = {}
        if pose_valid and self.use_visibility:
            depth_m = (
                None
                if depth_normalized is None or depth_scale_m is None
                else np.asarray(depth_normalized) * float(depth_scale_m)
            )
            for entity in relation_map.entities:
                code, _, _ = classify_map_point(
                    entity.xyz_m,
                    pose,
                    intrinsics,
                    depth_m,
                    depth_validity,
                    tolerance_m=float(
                        self.semantic_rules.get("visibility_depth_tolerance_m", 0.25)
                    ),
                )
                visibility[entity.entity_id] = int(code)
        relation_features = compute_relations(
            relation_map,
            timestamp_s=timestamp_s,
            position_m=position,
            velocity_xy_mps=velocity,
            yaw_rad=yaw,
            landing_time_s=self.landing_time_s,
            vehicle_radius_m=self.vehicle_radius_m,
            safety_margin_m=self.safety_margin_m,
            semantic_rules=self.semantic_rules,
            map_coverage=None,
            geometry_confidence=geometry_confidence,
            visibility_by_entity=visibility,
            use_uncertainty=self.use_uncertainty,
            use_visibility=self.use_visibility,
            canonicalization=self.canonicalization,
            candidate_merge_cell_m=self.candidate_merge_cell_m,
            merge_region_classes=self.merge_region_classes,
            importance_weights=self.candidate_importance_weights,
            protected_poi_threshold=self.protected_poi_threshold,
            protected_hazard_threshold=self.protected_hazard_threshold,
            protected_uncertainty_sigma_m=self.protected_uncertainty_sigma_m,
            landing_protection_radius_m=self.landing_protection_radius_m,
            semantic_rule_probability_threshold=self.semantic_rule_probability_threshold,
        )
        relation = torch.tensor(relation_features.values, dtype=torch.float32, device=device)[None]
        relation_validity = torch.tensor(
            relation_features.validity, dtype=torch.float32, device=device
        )[None]
        candidates = torch.tensor(relation_features.candidates, dtype=torch.float32, device=device)[
            None
        ]
        candidate_validity = torch.tensor(
            relation_features.candidate_validity, dtype=torch.float32, device=device
        )[None]
        feature_mask = np.asarray(relation_features.candidate_feature_validity, dtype=np.float32)
        if feature_mask.shape != np.asarray(relation_features.candidates).shape:
            feature_mask = np.zeros_like(np.asarray(relation_features.candidates), dtype=np.float32)
        candidate_feature_validity = torch.tensor(feature_mask, dtype=torch.float32, device=device)[
            None
        ]
        candidate_grid = np.zeros((len(relation_features.candidate_validity), 2), dtype=np.float32)
        candidate_projected_depth = np.zeros(
            len(relation_features.candidate_validity), dtype=np.float32
        )
        candidate_projection_validity = np.zeros(
            len(relation_features.candidate_validity), dtype=np.float32
        )
        candidate_visibility = np.asarray(relation_features.candidate_visibility, dtype=np.int64)
        if pose_valid:
            current_depth = (
                None
                if depth_normalized is None or depth_scale_m is None
                else np.asarray(depth_normalized) * float(depth_scale_m)
            )
            for index, map_point in enumerate(relation_features.candidate_map_xyz):
                if relation_features.candidate_validity[index] <= 0:
                    continue
                code, pixel, z_camera = classify_map_point(
                    map_point,
                    pose,
                    intrinsics,
                    current_depth if self.use_visibility else None,
                    depth_validity if self.use_visibility else None,
                    tolerance_m=float(
                        self.semantic_rules.get("visibility_depth_tolerance_m", 0.25)
                    ),
                )
                if not self.use_visibility and pixel is not None and z_camera is not None:
                    code = Visibility.VISIBLE
                candidate_visibility[index] = int(code)
                if pixel is not None and z_camera is not None:
                    candidate_grid[index] = pixel
                    candidate_projected_depth[index] = z_camera
                    candidate_projection_validity[index] = 1.0
        if family == "imf_htransformer":
            if validity is None:
                raise ValueError("IMF model requires analytical validity maps")
            outputs = model(
                visual_input.to(device),
                validity.to(device),
                state,
                state_mask,
                relation,
                relation_validity,
                candidates,
                candidate_validity,
                calibration=torch.tensor(
                    [intrinsics[name] for name in ("width", "height", "fx", "fy", "cx", "cy")],
                    dtype=torch.float32,
                    device=device,
                ),
                depth_scale_m=depth_scale_m,
                candidate_feature_validity=candidate_feature_validity,
                candidate_grid=torch.as_tensor(candidate_grid, device=device)[None],
                candidate_projected_depth_m=torch.as_tensor(
                    candidate_projected_depth, device=device
                )[None],
                candidate_projection_validity=torch.as_tensor(
                    candidate_projection_validity, device=device
                )[None],
                candidate_visibility=torch.as_tensor(candidate_visibility, device=device)[None],
            )
        else:
            outputs = model(
                visual_input.to(device),
                state,
                state_mask,
                relation,
                relation_validity,
                candidates,
                candidate_validity,
                candidate_feature_validity,
            )

        mapped_points = 0
        resolved = unresolved = 0
        spatial_attributes = None
        if family == "imf_htransformer":
            spatial_attributes = {
                "slope_rad": visual_input[0, 25].detach().cpu().numpy() * (np.pi / 2),
                "roughness_m": visual_input[0, 26].detach().cpu().numpy() * 0.1,
                "hazard_probability": outputs["hazard_logits"][0, 0]
                .detach()
                .float()
                .sigmoid()
                .cpu()
                .numpy(),
            }
        if (
            pose_valid
            and depth_normalized is not None
            and depth_validity is not None
            and depth_scale_m
        ):
            if depth_scale_m <= 0:
                raise ValueError("depth_scale_m must be positive")
            mapped_points = update_from_semantic_prediction(
                self.local_map,
                outputs["semantic_logits"][0].detach().float().cpu().numpy(),
                depth_normalized,
                depth_validity,
                intrinsics,
                pose,
                timestamp_s,
                SEMANTIC_CLASSES,
                source_prefix=f"semantic:{timestamp_s:.6f}",
                include_classes={"grass", "track", "cone", "rock", "other_obstacle"},
                attributes=spatial_attributes,
                pose_covariance=pose_covariance,
                pixel_sigma=self.observation_pixel_sigma,
                depth_sigma_base_m=self.depth_sigma_base_m,
                depth_sigma_relative=self.depth_sigma_relative,
            )

        landing_probability = outputs["landing_logits"][0].detach().float().sigmoid().cpu().numpy()
        landing_regions = extract_regions(landing_probability, 0.5, "landing", min_cells=4)
        poi_probability = outputs["poi"]["class_logits"][0].detach().float().sigmoid().cpu().numpy()
        poi_regions = extract_regions(poi_probability, 0.5, "poi", list(POI_CLASSES), min_cells=1)
        for detection in landing_regions + poi_regions:
            mapped = None
            if (
                pose_valid
                and depth_normalized is not None
                and depth_validity is not None
                and depth_scale_m is not None
            ):
                mapped = register_detection(
                    detection,
                    depth_normalized,
                    depth_validity,
                    depth_scale_m,
                    intrinsics,
                    pose,
                    timestamp_s,
                    attribute_maps={
                        "slope_rad": visual_input[0, 25].detach().cpu().numpy(),
                        "roughness_m": visual_input[0, 26].detach().cpu().numpy(),
                    }
                    if family == "imf_htransformer"
                    else None,
                    pose_covariance=pose_covariance,
                    pixel_sigma=self.observation_pixel_sigma,
                    depth_sigma_base_m=self.depth_sigma_base_m,
                    depth_sigma_relative=self.depth_sigma_relative,
                )
            if mapped is not None and mapped.resolved:
                self.local_map.update_region(
                    mapped.class_name,
                    mapped.map_xyz_m,
                    mapped.radius_m or 0.0,
                    mapped.probability,
                    timestamp_s,
                    source_observation=f"{mapped.kind}:{timestamp_s:.6f}:{mapped.source_grid_xy}",
                    attributes=mapped.attributes,
                    position_covariance_m2=mapped.position_covariance_m2,
                    covariance_source=mapped.covariance_source,
                )
                resolved += 1
            else:
                self.local_map.retain_unresolved(
                    detection.class_name,
                    detection.grid_xy,
                    detection.probability,
                    timestamp_s,
                    f"{detection.kind}:{timestamp_s:.6f}",
                )
                unresolved += 1

        if pose_valid:
            self.local_map.prune(timestamp_s, self.max_local_radius_m, position)
            self.previous_position_m = position.copy()
            self.previous_timestamp_s = timestamp_s

        return OnlineMapResult(
            outputs=outputs,
            mapped_semantic_points=mapped_points,
            resolved_regions=resolved,
            unresolved_regions=unresolved,
            map_size=len(self.local_map.entities),
            relation_values=relation_features.values,
            relation_validity=relation_features.validity,
            candidate_values=relation_features.candidates,
            candidate_validity=relation_features.candidate_validity,
            candidate_feature_validity=relation_features.candidate_feature_validity,
            candidate_map_xyz=relation_features.candidate_map_xyz,
            candidate_visibility=tuple(int(value) for value in candidate_visibility),
        )
