"""Bounded uncertainty-aware semantic map in the local metric frame."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from common.registry import MAP_SCHEMA_VERSION

from .uncertainty import fuse_gaussians, mahalanobis_squared, stabilize_covariance


MAP_SCHEMA = MAP_SCHEMA_VERSION


@dataclass
class SemanticObservation:
    xyz_m: tuple[float, float, float]
    class_probabilities: dict[str, float]
    confidence: float
    timestamp_s: float
    extent_m: float = 0.0
    source: str = "predicted"
    source_observation: str = ""
    valid: bool = True
    attributes: dict[str, float] = field(default_factory=dict)
    position_covariance_m2: list[list[float]] | None = None
    covariance_source: str = "unavailable"


@dataclass
class SemanticEntity:
    entity_id: str
    xyz_m: list[float]
    class_probabilities: dict[str, float]
    confidence: float
    observation_count: int
    first_seen_s: float
    last_seen_s: float
    extent_m: float
    source_observations: list[str] = field(default_factory=list)
    valid: bool = True
    attributes: dict[str, float] = field(default_factory=dict)
    position_covariance_m2: list[list[float]] | None = None
    covariance_valid: bool = False
    covariance_source: str = "unavailable"
    semantic_dirichlet_alpha: dict[str, float] = field(default_factory=dict)
    semantic_evidence_valid: bool = True

    @property
    def semantic_class(self) -> str:
        if not self.class_probabilities:
            return "unknown"
        label = max(self.class_probabilities, key=self.class_probabilities.get)
        if self.semantic_evidence_valid and (
            self.semantic_support < 0.5 or self.class_probabilities.get(label, 0.0) < 0.5
        ):
            return "unknown"
        return label

    @property
    def semantic_entropy(self) -> float:
        probabilities = np.asarray(list(self.class_probabilities.values()), dtype=np.float64)
        probabilities = probabilities[np.isfinite(probabilities) & (probabilities > 0)]
        return float(-np.sum(probabilities * np.log(probabilities))) if probabilities.size else 0.0

    @property
    def semantic_support(self) -> float:
        return float(sum(max(0.0, value) for value in self.semantic_dirichlet_alpha.values()))

    @property
    def position_sigma_m(self) -> float | None:
        if not self.covariance_valid or self.position_covariance_m2 is None:
            return None
        covariance = np.asarray(self.position_covariance_m2, dtype=np.float64)
        if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
            return None
        # Covariances are PSD-validated when observations enter the map and
        # when v2 records load. The trace summary needs no repeated eigensolve.
        return float(math.sqrt(max(0.0, float(np.trace(covariance)))))


class PersistentSemanticMap:
    """Deterministic map with bounded spatial association and evidence fusion."""

    def __init__(
        self,
        association_m: float = 0.75,
        voxel_m: float = 0.2,
        max_entities: int = 4096,
        max_age_s: float = 120.0,
        mode: str = "predicted_map",
        *,
        mahalanobis_threshold: float = 11.34,
        evidence_correlation_time_s: float = 1.0,
        evidence_decay_per_second: float = 0.0,
        max_semantic_evidence: float = 50.0,
        legacy_prior_strength: float = 1.0,
        association_mode: str = "mahalanobis",
        semantic_fusion: str = "bayesian",
        semantic_overlap_threshold: float = 0.1,
        semantic_association_groups: Iterable[Iterable[str]] = (),
    ):
        if (
            min(
                association_m,
                voxel_m,
                max_age_s,
                mahalanobis_threshold,
                evidence_correlation_time_s,
                max_semantic_evidence,
                legacy_prior_strength,
            )
            <= 0
        ):
            raise ValueError("Map bounds and evidence settings must be positive")
        if evidence_decay_per_second < 0 or max_entities < 1:
            raise ValueError("Invalid map/evidence settings")
        if mode not in {"predicted_map", "oracle_map"}:
            raise ValueError("Map mode must be predicted_map or oracle_map")
        if association_mode not in {"mahalanobis", "euclidean"}:
            raise ValueError("association_mode must be mahalanobis or euclidean")
        if semantic_fusion not in {"bayesian", "arithmetic"}:
            raise ValueError("semantic_fusion must be bayesian or arithmetic")
        if not 0.0 <= semantic_overlap_threshold <= 1.0:
            raise ValueError("semantic_overlap_threshold must be in [0,1]")
        self.association_m = float(association_m)
        self.voxel_m = float(voxel_m)
        self.max_entities = int(max_entities)
        self.max_age_s = float(max_age_s)
        self.mode = mode
        self.mahalanobis_threshold = float(mahalanobis_threshold)
        self.evidence_correlation_time_s = float(evidence_correlation_time_s)
        self.evidence_decay_per_second = float(evidence_decay_per_second)
        self.max_semantic_evidence = float(max_semantic_evidence)
        self.legacy_prior_strength = float(legacy_prior_strength)
        self.association_mode = association_mode
        self.semantic_fusion = semantic_fusion
        self.semantic_overlap_threshold = float(semantic_overlap_threshold)
        association_groups = []
        for group in semantic_association_groups:
            normalized = frozenset(str(name) for name in group)
            if len(normalized) > 1:
                association_groups.append(normalized)
        self.semantic_association_groups = tuple(association_groups)
        self.entities: list[SemanticEntity] = []
        self._next = 1
        self._voxel_entities: dict[tuple[int, int, int], list[int]] = {}
        self._index_cell_m = max(self.voxel_m, self.association_m)
        self.unresolved_observations: list[dict] = []
        self.association_stats = {
            "mahalanobis": 0,
            "euclidean_fallback": 0,
            "new_entity": 0,
            "semantic_incompatible": 0,
        }

    def _voxel(self, xyz) -> tuple[int, int, int]:
        return tuple(int(math.floor(float(value) / self.voxel_m)) for value in xyz)

    def _index_key(self, xyz) -> tuple[int, int, int]:
        return tuple(int(math.floor(float(value) / self._index_cell_m)) for value in xyz)

    def _index_add(self, index: int) -> None:
        key = self._index_key(self.entities[index].xyz_m)
        self._voxel_entities.setdefault(key, []).append(index)

    def _rebuild_index(self) -> None:
        self._voxel_entities.clear()
        for index, entity in enumerate(self.entities):
            if entity.valid:
                self._index_add(index)

    def _nearby_indices(self, xyz) -> list[int]:
        center = self._index_key(xyz)
        radius = int(math.ceil(self.association_m / self._index_cell_m))
        found = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    found.extend(
                        self._voxel_entities.get(
                            (center[0] + dx, center[1] + dy, center[2] + dz), ()
                        )
                    )
        return found

    @staticmethod
    def _normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
        values = {
            str(key): max(0.0, float(value))
            for key, value in probabilities.items()
            if math.isfinite(float(value))
        }
        total = sum(values.values())
        return {key: value / total for key, value in values.items()} if total > 0 else {}

    @staticmethod
    def _posterior_overlap(left: dict[str, float], right: dict[str, float]) -> float:
        """Return shared posterior mass (1 minus total-variation distance)."""
        classes = set(left) | set(right)
        return float(
            sum(
                min(max(0.0, left.get(name, 0.0)), max(0.0, right.get(name, 0.0)))
                for name in classes
            )
        )

    def _semantic_compatible(self, left: dict[str, float], right: dict[str, float]) -> bool:
        if self._posterior_overlap(left, right) >= self.semantic_overlap_threshold:
            return True
        left_label = max(left, key=left.get, default="unknown")
        right_label = max(right, key=right.get, default="unknown")
        return any(
            left_label in group and right_label in group
            for group in self.semantic_association_groups
        )

    def _effective_evidence(
        self,
        item: SemanticEntity,
        observation: SemanticObservation,
        probabilities: dict[str, float],
    ) -> dict[str, float]:
        if item.semantic_evidence_valid and item.semantic_dirichlet_alpha:
            alpha = dict(item.semantic_dirichlet_alpha)
        else:
            # v1 contained a posterior but no evidence history. Preserve that
            # posterior as an explicitly one-unit legacy prior; do not invent a
            # historical observation count or mark its evidence as measured.
            prior = self._normalize_probabilities(item.class_probabilities)
            alpha = {
                name: probability * self.legacy_prior_strength
                for name, probability in prior.items()
            }
        keys = set(alpha) | set(probabilities)
        elapsed = max(0.0, observation.timestamp_s - item.last_seen_s)
        if self.evidence_decay_per_second:
            decay = math.exp(-self.evidence_decay_per_second * elapsed)
            alpha = {name: value * decay for name, value in alpha.items()}
        temporal_weight = max(0.05, 1.0 - math.exp(-elapsed / self.evidence_correlation_time_s))
        weight = float(np.clip(observation.confidence, 0.0, 1.0)) * temporal_weight
        alpha = {
            name: alpha.get(name, 0.0) + weight * probabilities.get(name, 0.0) for name in keys
        }
        concentration = sum(alpha.values())
        if concentration > self.max_semantic_evidence:
            # Bound repeated-frame evidence while retaining its posterior.
            factor = self.max_semantic_evidence / concentration
            alpha = {name: value * factor for name, value in alpha.items()}
        return alpha

    def _correlated_measurement_covariance(
        self, item: SemanticEntity, covariance: np.ndarray | None, timestamp_s: float
    ) -> np.ndarray | None:
        """Inflate repeated near-time measurements to limit false independence."""
        if covariance is None:
            return None
        elapsed = max(0.0, timestamp_s - item.last_seen_s)
        weight = max(0.05, 1.0 - math.exp(-elapsed / self.evidence_correlation_time_s))
        return stabilize_covariance(covariance / weight)

    @staticmethod
    def _posterior(alpha: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, value) for value in alpha.values())
        if total <= 0:
            return {"unknown": 1.0}
        return {name: max(0.0, value) / total for name, value in sorted(alpha.items())}

    def update(self, observation: SemanticObservation) -> SemanticEntity | None:
        permitted_source = "predicted" if self.mode == "predicted_map" else "oracle"
        if observation.source != permitted_source:
            raise ValueError(f"{self.mode} accepts only {permitted_source!r} observations")
        xyz = np.asarray(observation.xyz_m, dtype=np.float64)
        if not observation.valid or xyz.shape != (3,) or not np.isfinite(xyz).all():
            return None
        probabilities = self._normalize_probabilities(observation.class_probabilities)
        if not probabilities:
            return None
        observed_covariance = stabilize_covariance(observation.position_covariance_m2)
        voxel = self._voxel(xyz)
        candidates = []
        for index in self._nearby_indices(xyz):
            item = self.entities[index]
            if not item.valid:
                continue
            if item.last_seen_s < observation.timestamp_s - self.max_age_s:
                continue
            delta = np.asarray(item.xyz_m, dtype=np.float64) - xyz
            euclidean = float(np.linalg.norm(delta))
            if euclidean > self.association_m:
                continue
            # Do not make association depend on only the current argmax. A
            # spatially supported observation with a partially overlapping
            # class posterior can be contradictory evidence about this same
            # entity; feed it to Bayesian fusion. Disjoint class hypotheses
            # remain separate, preventing obvious cross-class merges.
            if not self._semantic_compatible(item.class_probabilities, probabilities):
                self.association_stats["semantic_incompatible"] += 1
                continue
            item_covariance = (
                stabilize_covariance(item.position_covariance_m2) if item.covariance_valid else None
            )
            if (
                self.association_mode == "mahalanobis"
                and item_covariance is not None
                and observed_covariance is not None
            ):
                effective_covariance = self._correlated_measurement_covariance(
                    item, observed_covariance, observation.timestamp_s
                )
                distance2 = mahalanobis_squared(delta, item_covariance + effective_covariance)
                if distance2 is not None and distance2 <= self.mahalanobis_threshold:
                    candidates.append((distance2, index, item, "mahalanobis"))
            else:
                candidates.append((euclidean, index, item, "euclidean"))

        if candidates:
            _, index, item, association_method = min(candidates, key=lambda row: (row[0], row[1]))
            previous_key = self._index_key(item.xyz_m)
            if self.semantic_fusion == "bayesian":
                item.semantic_dirichlet_alpha = self._effective_evidence(
                    item, observation, probabilities
                )
                item.semantic_evidence_valid = True
                item.class_probabilities = self._posterior(item.semantic_dirichlet_alpha)
            else:
                old = item.class_probabilities
                keys = set(old) | set(probabilities)
                item.class_probabilities = self._normalize_probabilities(
                    {
                        name: old.get(name, 0.0) * item.observation_count
                        + probabilities.get(name, 0.0)
                        * float(np.clip(observation.confidence, 0.0, 1.0))
                        for name in keys
                    }
                )

            if observed_covariance is not None and item.covariance_valid:
                effective_covariance = self._correlated_measurement_covariance(
                    item, observed_covariance, observation.timestamp_s
                )
                fused = fuse_gaussians(
                    item.xyz_m, item.position_covariance_m2, xyz, effective_covariance
                )
                if fused is not None:
                    mean, covariance = fused
                    item.xyz_m = mean.tolist()
                    item.position_covariance_m2 = covariance.tolist()
                    item.covariance_valid = True
                    item.covariance_source = "fused_approximate"
            elif observed_covariance is not None:
                item.xyz_m = xyz.tolist()
                item.position_covariance_m2 = observed_covariance.tolist()
                item.covariance_valid = True
                item.covariance_source = observation.covariance_source
            elif item.covariance_valid:
                # A measurement without defensible covariance must not erase
                # previously supported position uncertainty or move the fused
                # estimate as though that measurement had known precision.
                pass
            else:
                # If neither position has covariance, retain a clearly
                # covariance-invalid Euclidean mean for the reference path.
                count = item.observation_count
                item.xyz_m = ((np.asarray(item.xyz_m) * count + xyz) / (count + 1)).tolist()
                item.covariance_valid = False
                item.position_covariance_m2 = None
                item.covariance_source = "unavailable"

            count = item.observation_count
            item.confidence = (
                item.confidence * count + float(np.clip(observation.confidence, 0, 1))
            ) / (count + 1)
            item.extent_m = (item.extent_m * count + max(0.0, observation.extent_m)) / (count + 1)
            for key, value in observation.attributes.items():
                if math.isfinite(float(value)):
                    old = item.attributes.get(key, float(value))
                    item.attributes[key] = (old * count + float(value)) / (count + 1)
            item.observation_count = count + 1
            item.last_seen_s = observation.timestamp_s
            if (
                observation.source_observation
                and observation.source_observation not in item.source_observations
            ):
                item.source_observations.append(observation.source_observation)
            item.source_observations = item.source_observations[-16:]
            self.association_stats[
                association_method if association_method == "mahalanobis" else "euclidean_fallback"
            ] += 1
            new_key = self._index_key(item.xyz_m)
            if new_key != previous_key:
                bucket = self._voxel_entities.get(previous_key, [])
                if index in bucket:
                    bucket.remove(index)
                if not bucket:
                    self._voxel_entities.pop(previous_key, None)
                self._index_add(index)
            return item

        alpha = {
            name: 0.05 + float(np.clip(observation.confidence, 0, 1)) * value
            for name, value in probabilities.items()
        }
        covariance = observed_covariance.tolist() if observed_covariance is not None else None
        entity = SemanticEntity(
            entity_id=f"local-{self._next:07d}",
            xyz_m=xyz.tolist(),
            class_probabilities=self._posterior(alpha),
            confidence=float(np.clip(observation.confidence, 0, 1)),
            observation_count=1,
            first_seen_s=observation.timestamp_s,
            last_seen_s=observation.timestamp_s,
            extent_m=max(0.0, observation.extent_m),
            source_observations=[observation.source_observation]
            if observation.source_observation
            else [],
            attributes={
                key: float(value)
                for key, value in observation.attributes.items()
                if math.isfinite(float(value))
            },
            position_covariance_m2=covariance,
            covariance_valid=covariance is not None,
            covariance_source=observation.covariance_source
            if covariance is not None
            else "unavailable",
            semantic_dirichlet_alpha=alpha,
        )
        self._next += 1
        self.entities.append(entity)
        self._index_add(len(self.entities) - 1)
        self.association_stats["new_entity"] += 1
        if len(self.entities) > self.max_entities:
            self.entities.sort(
                key=lambda item: (item.last_seen_s, item.confidence, item.entity_id), reverse=True
            )
            del self.entities[self.max_entities :]
            self._rebuild_index()
        return entity

    def update_region(
        self,
        class_name: str,
        xyz_m,
        radius_m: float,
        confidence: float,
        timestamp_s: float,
        source_observation: str = "",
        attributes: dict[str, float] | None = None,
        position_covariance_m2=None,
        covariance_source: str = "unavailable",
    ) -> SemanticEntity | None:
        """Insert a registered POI or landing region into the shared map."""
        return self.update(
            SemanticObservation(
                tuple(float(value) for value in xyz_m),
                {class_name: 1.0},
                confidence,
                timestamp_s,
                radius_m,
                "predicted",
                source_observation,
                True,
                attributes or {},
                None
                if position_covariance_m2 is None
                else np.asarray(position_covariance_m2).tolist(),
                covariance_source,
            )
        )

    def retain_unresolved(
        self,
        class_name: str,
        image_xy,
        confidence: float,
        timestamp_s: float,
        source_observation: str = "",
    ) -> None:
        self.unresolved_observations.append(
            {
                "class_name": class_name,
                "image_xy": [float(image_xy[0]), float(image_xy[1])],
                "confidence": float(confidence),
                "timestamp_s": float(timestamp_s),
                "source_observation": source_observation,
            }
        )
        self.unresolved_observations = self.unresolved_observations[-1024:]

    def prune(self, now_s: float, radius_m: float | None = None, origin=(0.0, 0.0, 0.0)) -> None:
        origin = np.asarray(origin, dtype=np.float64)
        self.entities = [
            entity
            for entity in self.entities
            if entity.valid
            and now_s - entity.last_seen_s <= self.max_age_s
            and (radius_m is None or np.linalg.norm(np.asarray(entity.xyz_m) - origin) <= radius_m)
        ]
        self._rebuild_index()

    def by_class(self, names: Iterable[str]) -> list[SemanticEntity]:
        allowed = set(names)
        return [
            entity for entity in self.entities if entity.valid and entity.semantic_class in allowed
        ]

    def records(self):
        return [asdict(entity) for entity in self.entities]

    def save(self, path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        payload = {
            "schema": MAP_SCHEMA,
            "mode": self.mode,
            "association_m": self.association_m,
            "voxel_m": self.voxel_m,
            "max_entities": self.max_entities,
            "max_age_s": self.max_age_s,
            "mahalanobis_threshold": self.mahalanobis_threshold,
            "evidence_correlation_time_s": self.evidence_correlation_time_s,
            "evidence_decay_per_second": self.evidence_decay_per_second,
            "max_semantic_evidence": self.max_semantic_evidence,
            "legacy_prior_strength": self.legacy_prior_strength,
            "association_mode": self.association_mode,
            "semantic_fusion": self.semantic_fusion,
            "semantic_overlap_threshold": self.semantic_overlap_threshold,
            "semantic_association_groups": [
                sorted(group) for group in self.semantic_association_groups
            ],
            "next_id": self._next,
            "association_stats": self.association_stats,
            "entities": self.records(),
            "unresolved_observations": self.unresolved_observations,
        }
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = payload.get("schema")
        if schema not in {
            "imagination-semantic-map-v1",
            "imagination-semantic-map-v2",
            MAP_SCHEMA,
        }:
            raise ValueError(f"Unknown semantic-map schema {schema!r}")
        result = cls(
            payload["association_m"],
            payload["voxel_m"],
            payload["max_entities"],
            payload["max_age_s"],
            payload["mode"],
            mahalanobis_threshold=payload.get("mahalanobis_threshold", 11.34),
            evidence_correlation_time_s=payload.get("evidence_correlation_time_s", 1.0),
            evidence_decay_per_second=payload.get("evidence_decay_per_second", 0.0),
            max_semantic_evidence=payload.get("max_semantic_evidence", 50.0),
            legacy_prior_strength=payload.get("legacy_prior_strength", 1.0),
            association_mode=payload.get(
                "association_mode",
                "euclidean" if schema == "imagination-semantic-map-v1" else "mahalanobis",
            ),
            semantic_fusion=payload.get(
                "semantic_fusion",
                "arithmetic" if schema == "imagination-semantic-map-v1" else "bayesian",
            ),
            semantic_overlap_threshold=payload.get("semantic_overlap_threshold", 0.1),
            semantic_association_groups=payload.get("semantic_association_groups", ()),
        )
        for record in payload.get("entities", []):
            item = dict(record)
            if schema == "imagination-semantic-map-v1":
                item.update(
                    position_covariance_m2=None,
                    covariance_valid=False,
                    covariance_source="unavailable_after_v1_migration",
                    semantic_dirichlet_alpha=PersistentSemanticMap._normalize_probabilities(
                        item.get("class_probabilities", {})
                    ),
                    semantic_evidence_valid=False,
                )
            else:
                item.setdefault("position_covariance_m2", None)
                item.setdefault("covariance_valid", False)
                item.setdefault("covariance_source", "unavailable")
                item.setdefault("semantic_dirichlet_alpha", {})
                item.setdefault("semantic_evidence_valid", True)
                if item.get("covariance_valid"):
                    covariance = stabilize_covariance(item.get("position_covariance_m2"))
                    if covariance is None:
                        item["position_covariance_m2"] = None
                        item["covariance_valid"] = False
                        item["covariance_source"] = "invalid_on_load"
                    else:
                        item["position_covariance_m2"] = covariance.tolist()
            result.entities.append(SemanticEntity(**item))
        result._next = int(payload["next_id"])
        result._rebuild_index()
        result.unresolved_observations = list(payload.get("unresolved_observations", []))[-1024:]
        result.association_stats.update(payload.get("association_stats", {}))
        return result


def update_from_semantic_prediction(
    local_map: PersistentSemanticMap,
    semantic_logits,
    depth_m,
    depth_valid,
    intrinsics: dict[str, float],
    camera_to_map,
    timestamp_s: float,
    class_names: tuple[str, ...],
    confidence_threshold: float = 0.55,
    sample_stride: int = 2,
    source_prefix: str = "prediction",
    include_classes: set[str] | None = None,
    attributes: dict[str, np.ndarray] | None = None,
    pose_covariance=None,
    pixel_sigma: float = 1.0,
    depth_sigma_base_m: float = 0.02,
    depth_sigma_relative: float = 0.01,
) -> int:
    """Project predicted labels and measured IMF depth; targets are not inputs."""
    from .uncertainty import pixel_depth_covariance, propagate_camera_to_map

    logits = np.asarray(semantic_logits, dtype=np.float32)
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.asarray(depth_valid) > 0
    if (
        logits.ndim != 3
        or depth.ndim != 2
        or valid.shape != depth.shape
        or logits.shape[1:] != depth.shape
    ):
        raise ValueError("Expected class×H×W logits plus same-sized metric depth and validity")
    transform = np.asarray(camera_to_map, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("camera_to_map must be a finite 4×4 matrix")
    h, w = depth.shape
    source_width, source_height = float(intrinsics["width"]), float(intrinsics["height"])
    scale_x, scale_y = w / source_width, h / source_height
    scaled_intrinsics = {
        "fx": float(intrinsics["fx"]) * scale_x,
        "fy": float(intrinsics["fy"]) * scale_y,
        "cx": (float(intrinsics["cx"]) + 0.5) * scale_x - 0.5,
        "cy": (float(intrinsics["cy"]) + 0.5) * scale_y - 0.5,
    }
    shifted = logits - logits.max(axis=0, keepdims=True)
    probability = np.exp(shifted)
    probability /= np.maximum(probability.sum(axis=0, keepdims=True), 1e-12)
    count = 0
    for y in range(0, h, max(1, sample_stride)):
        for x in range(0, w, max(1, sample_stride)):
            z = float(depth[y, x])
            if not valid[y, x] or not math.isfinite(z) or z <= 0:
                continue
            cls = int(probability[:, y, x].argmax())
            confidence = float(probability[cls, y, x])
            if cls >= len(class_names) or confidence < confidence_threshold:
                continue
            if include_classes is not None and class_names[cls] not in include_classes:
                continue
            point = np.array(
                [
                    (x - scaled_intrinsics["cx"]) * z / scaled_intrinsics["fx"],
                    (y - scaled_intrinsics["cy"]) * z / scaled_intrinsics["fy"],
                    z,
                ],
                dtype=np.float64,
            )
            mapped = transform @ np.append(point, 1.0)
            if not np.isfinite(mapped[:3]).all():
                continue
            depth_sigma = depth_sigma_base_m + depth_sigma_relative * z
            point_covariance = pixel_depth_covariance(
                (x, y),
                z,
                scaled_intrinsics,
                pixel_sigma=pixel_sigma,
                depth_sigma_m=depth_sigma,
            )
            map_covariance = (
                propagate_camera_to_map(point, point_covariance, transform, pose_covariance)
                if point_covariance is not None and pose_covariance is not None
                else None
            )
            extent = (
                z
                * max(1, sample_stride)
                / math.sqrt(scaled_intrinsics["fx"] * scaled_intrinsics["fy"])
            )
            distribution = {name: float(probability[i, y, x]) for i, name in enumerate(class_names)}
            measured = {}
            for key, plane in (attributes or {}).items():
                values = np.asarray(plane)
                if values.shape == depth.shape and np.isfinite(values[y, x]):
                    measured[key] = float(values[y, x])
            observation = SemanticObservation(
                tuple(mapped[:3]),
                distribution,
                confidence,
                timestamp_s,
                extent,
                "predicted",
                f"{source_prefix}:{x}:{y}",
                True,
                measured,
                None if map_covariance is None else map_covariance.tolist(),
                "first_order_approximation" if map_covariance is not None else "unavailable",
            )
            if local_map.update(observation) is not None:
                count += 1
    return count
