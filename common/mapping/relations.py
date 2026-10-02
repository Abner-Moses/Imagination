"""Compact egocentric relationships computed deterministically from the map."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from common.registry import (
    CANDIDATE_FEATURE_NAMES,
    CANDIDATE_TYPES,
    MAX_CANDIDATES,
    RELATIONAL_NAMES,
)
from .coordinates import map_to_egocentric, map_vector_to_egocentric
from .semantic_map import PersistentSemanticMap
from .visibility import Visibility


@dataclass(frozen=True)
class RelationalFeatures:
    names: tuple[str, ...]
    values: tuple[float, ...]
    validity: tuple[float, ...]
    candidates: tuple[tuple[float, ...], ...]
    candidate_validity: tuple[float, ...]
    candidate_map_xyz: tuple[tuple[float, float, float], ...] = ()
    candidate_visibility: tuple[int, ...] = ()
    candidate_tokens_before: int = 0
    candidate_tokens_after: int = 0
    merged_region_count: int = 0
    candidate_feature_validity: tuple[tuple[float, ...], ...] = ()


def _bearing(delta: np.ndarray) -> tuple[float, float]:
    angle = math.atan2(float(delta[1]), float(delta[0]))
    return math.sin(angle), math.cos(angle)


def _surface_clearance(distance: float, extent_m: float) -> float:
    """Point-to-disc approximation; negative values mean the origin is inside."""
    return float(distance - max(0.0, extent_m))


def _entity_sigma(entity) -> float:
    if not entity.covariance_valid or entity.position_covariance_m2 is None:
        return 0.0
    covariance = np.asarray(entity.position_covariance_m2, dtype=np.float64)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        return 0.0
    # Closed form for the largest eigenvalue of a symmetric 2x2 matrix avoids
    # repeated LAPACK setup for the many small covariances in a local map.
    a, b = float(covariance[0, 0]), float(covariance[0, 1])
    d = float(covariance[1, 1])
    largest = 0.5 * (a + d + math.sqrt(max(0.0, (a - d) ** 2 + 4.0 * b * b)))
    return math.sqrt(max(0.0, largest))


def _merge_region_tokens(rows, *, classes, cell_m: float, origin_map, yaw_rad=None):
    """Aggregate configured extended regions while retaining point objects."""
    groups = {}
    untouched = []
    for row in rows:
        if row[3].semantic_class not in classes:
            untouched.append(row)
            continue
        delta_ego, entity = row[2], row[3]
        key = (
            entity.semantic_class,
            math.floor(float(delta_ego[0]) / cell_m),
            math.floor(float(delta_ego[1]) / cell_m),
        )
        groups.setdefault(key, []).append(row)
    merged = list(untouched)
    removed = 0
    for key, group in sorted(groups.items()):
        if len(group) == 1:
            merged.append(group[0])
            continue
        removed += len(group) - 1
        weights = np.asarray([max(0.05, row[3].confidence) for row in group], dtype=np.float64)
        weights /= weights.sum()
        # row[1] is map-relative displacement. Persistent merged entities must
        # retain absolute local-map XYZ; the egocentric displacement is already
        # averaged separately below.
        center_map_xyz = sum(
            weight * np.asarray(row[3].xyz_m, dtype=np.float64)
            for weight, row in zip(weights, group)
        )
        center_delta_map = center_map_xyz - np.asarray(origin_map, dtype=np.float64)
        if yaw_rad is not None:
            center_ego = map_to_egocentric(center_map_xyz, origin_map, float(yaw_rad))
        else:
            center_ego = center_delta_map.copy()
        representative = group[0][3]
        extent = max(
            float(np.linalg.norm(row[2][:2] - center_ego[:2])) + row[3].extent_m for row in group
        )
        labels = set().union(*(row[3].class_probabilities for row in group))
        probabilities = {
            name: float(
                sum(
                    weight * row[3].class_probabilities.get(name, 0.0)
                    for weight, row in zip(weights, group)
                )
            )
            for name in labels
        }
        alpha_names = set().union(*(row[3].semantic_dirichlet_alpha for row in group))
        alpha = {
            name: float(sum(row[3].semantic_dirichlet_alpha.get(name, 0.0) for row in group))
            for name in alpha_names
        }
        cov_valid = all(row[3].covariance_valid for row in group)
        covariance = None
        if cov_valid:
            covariance = np.zeros((3, 3), dtype=np.float64)
            for weight, row in zip(weights, group):
                delta = np.asarray(row[3].xyz_m, dtype=np.float64) - center_map_xyz
                covariance += weight * (
                    np.asarray(row[3].position_covariance_m2) + np.outer(delta, delta)
                )
        attribute_names = set().union(*(row[3].attributes for row in group))
        attributes = {}
        for name in attribute_names:
            present = [
                (weight, row[3].attributes[name])
                for weight, row in zip(weights, group)
                if name in row[3].attributes
            ]
            total = sum(weight for weight, _ in present)
            attributes[name] = sum(weight * value for weight, value in present) / max(total, 1e-12)
        entity = replace(
            representative,
            entity_id=f"region:{key[0]}:{key[1]}:{key[2]}",
            xyz_m=center_map_xyz.tolist(),
            class_probabilities=probabilities,
            confidence=float(
                sum(weight * row[3].confidence for weight, row in zip(weights, group))
            ),
            observation_count=sum(row[3].observation_count for row in group),
            first_seen_s=min(row[3].first_seen_s for row in group),
            last_seen_s=max(row[3].last_seen_s for row in group),
            extent_m=extent,
            attributes=attributes,
            position_covariance_m2=covariance.tolist() if cov_valid else None,
            covariance_valid=cov_valid,
            covariance_source="aggregated_first_order" if cov_valid else "unavailable",
            semantic_dirichlet_alpha=alpha,
        )
        # Compute delta from the authoritative absolute map position. The
        # weighted egocentric average is equivalent for a common origin/yaw.
        center_delta_map = center_map_xyz - np.asarray(origin_map, dtype=np.float64)
        merged.append(
            (
                float(np.linalg.norm(center_delta_map[:2])),
                center_delta_map,
                center_ego,
                entity,
                max(row[4] for row in group),
            )
        )
    return merged, removed


def compute_relations(
    local_map: PersistentSemanticMap,
    *,
    timestamp_s: float,
    position_m=(0.0, 0.0, 0.0),
    velocity_xy_mps=None,
    yaw_rad: float | None = None,
    landing_time_s: float = 2.0,
    vehicle_radius_m: float = 0.35,
    safety_margin_m: float = 0.5,
    density_radii_m=(1.0, 2.0, 4.0),
    semantic_rules=None,
    map_coverage: float | None = None,
    geometry_confidence: float | None = None,
    visibility_by_entity: dict[str, int] | None = None,
    use_uncertainty: bool = True,
    use_visibility: bool = True,
    canonicalization: str = "egocentric",
    candidate_merge_cell_m: float = 0.5,
    merge_region_classes=("track", "grass"),
    importance_weights=None,
    protected_poi_threshold: float = 0.6,
    protected_hazard_threshold: float = 0.5,
    semantic_rule_probability_threshold: float = 0.1,
    protected_uncertainty_sigma_m: float = 0.25,
    landing_protection_radius_m: float = 1.0,
) -> RelationalFeatures:
    """Calculate distances, density, drift, uncertainty and map evidence.

    Entity extents are represented as discs in the local horizontal plane. Track
    and obstacle clearance is therefore point-to-disc, not an exact polygonal
    boundary distance. Bearings and token XYZ use a yaw-aligned ground frame:
    +x forward, +y left, +z up. Missing yaw leaves map-aligned values and marks
    orientation validity as false.
    """
    rules = semantic_rules or {}
    origin = np.asarray(position_m, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("position_m must be a finite local-map XYZ point")
    velocity = None if velocity_xy_mps is None else np.asarray(velocity_xy_mps, dtype=np.float64)
    velocity_valid = velocity is not None and velocity.shape == (2,) and np.isfinite(velocity).all()
    if canonicalization not in {"egocentric", "map"}:
        raise ValueError("canonicalization must be egocentric or map")
    if candidate_merge_cell_m <= 0:
        raise ValueError("candidate_merge_cell_m must be positive")
    if not 0.0 <= protected_poi_threshold <= 1.0 or not 0.0 <= protected_hazard_threshold <= 1.0:
        raise ValueError("Protected POI/hazard thresholds must be in [0,1]")
    if protected_uncertainty_sigma_m < 0 or landing_protection_radius_m < 0:
        raise ValueError("Protection distance thresholds must be nonnegative")
    if not 0.0 <= semantic_rule_probability_threshold <= 1.0:
        raise ValueError("semantic_rule_probability_threshold must be in [0,1]")
    yaw_valid = (
        canonicalization == "egocentric" and yaw_rad is not None and math.isfinite(float(yaw_rad))
    )
    visibility_by_entity = visibility_by_entity or {} if use_visibility else {}
    entities = [item for item in local_map.entities if item.valid]
    records = []
    sigma_by_entity = {
        item.entity_id: (_entity_sigma(item) if use_uncertainty else 0.0) for item in entities
    }

    def sigma_for(entity) -> float:
        if entity.entity_id in sigma_by_entity:
            return sigma_by_entity[entity.entity_id]
        return _entity_sigma(entity) if use_uncertainty else 0.0

    for entity in entities:
        point = np.asarray(entity.xyz_m, dtype=np.float64)
        delta_map = point - origin
        if delta_map.shape == (3,) and np.isfinite(delta_map).all():
            delta_ego = (
                map_to_egocentric(point, origin, float(yaw_rad)) if yaw_valid else delta_map.copy()
            )
            distance = float(np.linalg.norm(delta_map[:2]))
            records.append(
                (
                    entity,
                    delta_map,
                    delta_ego,
                    distance,
                    int(visibility_by_entity.get(entity.entity_id, Visibility.UNKNOWN)),
                )
            )

    values = np.zeros(len(RELATIONAL_NAMES), dtype=np.float32)
    validity = np.zeros_like(values)
    index = {name: i for i, name in enumerate(RELATIONAL_NAMES)}

    def put(name: str, value: float, known: bool = True) -> None:
        valid = known and math.isfinite(float(value))
        values[index[name]] = float(value) if valid else 0.0
        validity[index[name]] = float(valid)

    def semantic_mass(entity, classes):
        return float(sum(max(0.0, entity.class_probabilities.get(name, 0.0)) for name in classes))

    # A no-land object (for example a cone) is an obstacle, not a restricted
    # zone. Keep candidate restricted-boundary clearance distinct from obstacle
    # clearance; zones are configured explicitly or through a no-fly rule.
    restricted_classes = {
        name
        for name, rule in rules.items()
        if rule.get("restricted_zone", False) or rule.get("fly_allowed", True) is False
    }
    obstacle_classes = {"cone", "rock", "other_obstacle", "obstacle", "pole", "chair", "box"}
    poi_classes = {"poi", "football", "bag", "cone", "rock"}
    restricted = []
    obstacles = []
    cones = []
    pois = []
    landings = []
    for entity, delta_map, delta_ego, distance, visibility in records:
        name = entity.semantic_class
        row = (distance, delta_map, delta_ego, entity, visibility)
        if semantic_mass(entity, restricted_classes) >= semantic_rule_probability_threshold:
            restricted.append(row)
        if semantic_mass(entity, obstacle_classes) >= semantic_rule_probability_threshold:
            obstacles.append(row)
        if semantic_mass(entity, {"cone"}) >= semantic_rule_probability_threshold:
            cones.append(row)
        if semantic_mass(entity, poi_classes) >= semantic_rule_probability_threshold:
            pois.append(row)
        if semantic_mass(entity, {"landing"}) >= semantic_rule_probability_threshold:
            landings.append(row)

    tracks = [
        row
        for row in records
        if semantic_mass(row[0], {"track"}) >= semantic_rule_probability_threshold
    ]
    if tracks:
        entity, _, delta_ego, distance, _ = min(
            tracks,
            key=lambda row: (
                row[3] - row[0].extent_m - 2.0 * sigma_for(row[0]),
                row[0].entity_id,
            ),
        )
        track_clearance = _surface_clearance(distance, entity.extent_m)
        if use_uncertainty:
            track_clearance -= 2.0 * sigma_for(entity)
        put("nearest_track_m", track_clearance)
        sin_bearing, cos_bearing = _bearing(delta_ego)
        put("track_bearing_sin", sin_bearing, yaw_valid)
        put("track_bearing_cos", cos_bearing, yaw_valid)

    if restricted:
        _, _, delta_ego, entity, _ = min(
            restricted,
            key=lambda row: (
                row[0] - row[3].extent_m - 2.0 * sigma_for(row[3]),
                row[3].entity_id,
            ),
        )
        clearance = _surface_clearance(float(np.linalg.norm(delta_ego[:2])), entity.extent_m)
        if use_uncertainty:
            clearance -= 2.0 * _entity_sigma(entity)
        put("restricted_zone_clearance_m", clearance)

    if obstacles:
        obstacle_clearances = [
            (_surface_clearance(row[0], row[3].extent_m) - 2.0 * sigma_for(row[3]), row)
            for row in obstacles
        ]
        put("nearest_obstacle_m", min(clearance for clearance, _ in obstacle_clearances))
        for radius, name in zip(
            density_radii_m, ("cone_density_1m", "cone_density_2m", "cone_density_4m")
        ):
            count = sum(
                _surface_clearance(distance, entity.extent_m) - 2.0 * sigma_for(entity) <= radius
                for distance, _, _, entity, _ in cones
            )
            put(name, count / (math.pi * radius**2))
        radius = 2.0
        count = sum(clearance <= radius for clearance, _ in obstacle_clearances)
        put("obstacle_density_2m", count / (math.pi * radius**2))

    # A landing/free-space region must avoid every configured forbidden class,
    # including restricted track regions that are not object obstacles.
    landing_forbidden = {row[3].entity_id: row for row in obstacles + restricted}
    if landing_forbidden:
        forbidden_clearances = [
            _surface_clearance(row[0], row[3].extent_m) - 2.0 * sigma_for(row[3])
            for row in landing_forbidden.values()
        ]
        put("free_space_radius_m", max(0.0, min(forbidden_clearances) - vehicle_radius_m))

    if cones:
        buckets: dict[tuple[int, int], list] = {}
        for row in cones:
            delta = row[2]
            cell = (round(float(delta[0]) / 0.5), round(float(delta[1]) / 0.5))
            buckets.setdefault(cell, []).append(row)
        _, cluster = max(buckets.items(), key=lambda pair: (len(pair[1]), -pair[0][0], -pair[0][1]))
        center = np.mean([row[2][:2] for row in cluster], axis=0)
        sin_bearing, cos_bearing = _bearing(center)
        put("maximum_cone_density", len(cluster) / (math.pi * 0.25**2))
        put("maximum_density_bearing_sin", sin_bearing, yaw_valid)
        put("maximum_density_bearing_cos", cos_bearing, yaw_valid)

    if landings:
        _, _, _, best, _ = min(landings, key=lambda row: (-row[3].confidence, row[0]))
        put("best_landing_clearance_m", best.extent_m)

    drift_map = None
    drift_ego = None
    if velocity_valid:
        drift_map = velocity * max(0.0, float(landing_time_s))
        drift_ego = (
            map_vector_to_egocentric(drift_map, float(yaw_rad)) if yaw_valid else drift_map.copy()
        )
        put("drift_x_m", drift_ego[0])
        put("drift_y_m", drift_ego[1])
        put("horizontal_speed_mps", float(np.linalg.norm(velocity)))
        if landings and (obstacles or restricted):
            _, _, _, best, _ = min(landings, key=lambda row: (-row[3].confidence, row[0]))
            drifted_xy = np.asarray(best.xyz_m[:2]) - origin[:2] + drift_map
            drift_forbidden = {row[3].entity_id: row for row in obstacles + restricted}
            clearance = min(
                (
                    float(np.linalg.norm(drifted_xy - (np.asarray(row[3].xyz_m[:2]) - origin[:2])))
                    - row[3].extent_m
                    - 2.0 * sigma_for(row[3])
                    for row in drift_forbidden.values()
                ),
                default=0.0,
            )
            put("drift_clearance_m", clearance)
            put("landing_margin_m", clearance - vehicle_radius_m - safety_margin_m)

    if map_coverage is not None:
        put("map_coverage", float(np.clip(map_coverage, 0.0, 1.0)))
    if geometry_confidence is not None:
        put("geometry_confidence", float(np.clip(geometry_confidence, 0.0, 1.0)))
    put("orientation_valid", float(yaw_valid))
    if entities:
        put(
            "map_semantic_confidence",
            float(np.mean([max(item.class_probabilities.values()) for item in entities])),
        )
        sigmas = (
            [item.position_sigma_m for item in entities if item.position_sigma_m is not None]
            if use_uncertainty
            else []
        )
        if sigmas:
            put("map_position_sigma_m", float(np.mean(sigmas)))
        if use_uncertainty:
            put(
                "map_semantic_entropy", float(np.mean([item.semantic_entropy for item in entities]))
            )
        if use_uncertainty:
            put(
                "map_evidence_support_scaled",
                float(
                    np.mean(
                        [
                            min(item.semantic_support / local_map.max_semantic_evidence, 1.0)
                            for item in entities
                        ]
                    )
                ),
            )
        if use_visibility and visibility_by_entity:
            put(
                "map_visible_fraction",
                sum(value == int(Visibility.VISIBLE) for value in visibility_by_entity.values())
                / max(len(visibility_by_entity), 1),
            )
        age = max(0.0, timestamp_s - max(item.last_seen_s for item in entities))
        put("geometry_freshness", max(0.0, 1.0 - age / 10.0))
    if map_coverage is not None:
        put("map_coverage", float(np.clip(map_coverage, 0.0, 1.0)))
    elif entities:
        put("map_coverage", min(len(entities) / MAX_CANDIDATES, 1.0))
    put("mapped_poi_count_scaled", min(len(pois) / MAX_CANDIDATES, 1.0))
    if pois:
        put("nearest_poi_m", min(row[0] for row in pois))

    type_index = {name: i for i, name in enumerate(CANDIDATE_TYPES)}
    unique_sources = {}
    for item in restricted + obstacles + pois + landings:
        unique_sources.setdefault(item[3].entity_id, item)
    # Uniform predicted terrain regions (currently grass) remain represented as
    # compact merged tokens so pruning reduces redundancy without erasing the
    # semantic ground coverage around point-like detections.
    merge_classes = set(merge_region_classes)
    for entity, delta_map, delta_ego, distance, visibility in records:
        if entity.semantic_class in merge_classes:
            unique_sources.setdefault(
                entity.entity_id, (distance, delta_map, delta_ego, entity, visibility)
            )
    raw_candidate_rows = list(unique_sources.values())
    tokens_before = len(raw_candidate_rows)
    candidate_sources, merged_count = _merge_region_tokens(
        raw_candidate_rows,
        classes=set(merge_region_classes),
        cell_m=candidate_merge_cell_m,
        origin_map=origin,
        yaw_rad=float(yaw_rad) if yaw_valid else None,
    )
    configured_weights = {
        "restricted_proximity": 1.0,
        "uncertainty": 0.25,
        "novelty": 0.2,
        "poi_evidence": 0.5,
        "landing_evidence": 0.5,
    }
    configured_weights.update(importance_weights or {})
    if any(float(weight) < 0 for weight in configured_weights.values()):
        raise ValueError("Candidate importance weights must be nonnegative")
    configured_weights.setdefault("hazard_evidence", 1.0)
    landing_regions = [
        (np.asarray(row[1][:2], dtype=np.float64), max(0.0, row[3].extent_m)) for row in landings
    ]
    candidate_rows = []
    seen: set[str] = set()
    columns = {field: i for i, field in enumerate(CANDIDATE_FEATURE_NAMES)}

    def candidate_clearance(point_map, candidate_entity, source_rows, candidate_sigma):
        clearances = []
        for source in source_rows:
            other = source[3]
            if other.entity_id == candidate_entity.entity_id:
                continue
            delta = np.asarray(point_map[:2], dtype=np.float64) - np.asarray(
                other.xyz_m[:2], dtype=np.float64
            )
            uncertainty = (
                2.0 * math.hypot(candidate_sigma, sigma_for(other)) if use_uncertainty else 0.0
            )
            # A merged extended region already represents member points within
            # its own disc; do not treat those source observations as external
            # obstacles around the merged centroid.
            if (
                candidate_entity.entity_id.startswith("region:")
                and other.semantic_class == candidate_entity.semantic_class
                and np.linalg.norm(delta) <= candidate_entity.extent_m + other.extent_m
            ):
                continue
            clearances.append(float(np.linalg.norm(delta)) - max(0.0, other.extent_m) - uncertainty)
        return min(clearances) if clearances else None

    track_rows = [
        (row[3], row[1], row[2], row[0], row[4])
        for row in records
        if semantic_mass(row[0], {"track"}) >= semantic_rule_probability_threshold
    ]
    for distance, delta_map, delta_ego, entity, visibility in candidate_sources:
        if entity.entity_id in seen:
            continue
        seen.add(entity.entity_id)
        name = entity.semantic_class
        rule = rules.get(name, {})
        restricted_probability = semantic_mass(entity, restricted_classes)
        hazard_score = float(np.clip(entity.attributes.get("hazard_probability", 0.0), 0.0, 1.0))
        if (
            rule.get("restricted_zone", False)
            or rule.get("fly_allowed", True) is False
            or restricted_probability >= semantic_rule_probability_threshold
        ):
            kind, priority = "restricted_boundary", 0
        elif hazard_score >= protected_hazard_threshold:
            kind, priority = "high_risk", 0
        elif name == "landing":
            kind, priority = "landing", 1
        elif name in {"poi", "football", "bag"}:
            kind, priority = "poi", 3
        elif name in merge_classes:
            kind, priority = "terrain_region", 2
        else:
            kind, priority = "obstacle_cluster", 0
        row = np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float32)
        feature_validity = np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float32)

        def set_feature(feature_name, value, known=True):
            valid = bool(known) and math.isfinite(float(value))
            if valid:
                row[columns[feature_name]] = float(value)
                feature_validity[columns[feature_name]] = 1.0

        for feature_name, value in zip(
            ("egocentric_x_m", "egocentric_y_m", "egocentric_z_m"), delta_ego
        ):
            set_feature(feature_name, value)
        set_feature("distance_m", distance)
        bearing_valid = canonicalization == "map" or yaw_valid
        bearing_sin, bearing_cos = _bearing(delta_ego)
        set_feature("bearing_sin", bearing_sin, bearing_valid)
        set_feature("bearing_cos", bearing_cos, bearing_valid)
        for candidate_type in CANDIDATE_TYPES:
            set_feature(f"type_{candidate_type}", float(kind == candidate_type))
        if entity.class_probabilities:
            set_feature("semantic_confidence", max(entity.class_probabilities.values()))
        set_feature("extent_m", entity.extent_m)
        set_feature(
            "slope_rad", entity.attributes.get("slope_rad", 0.0), "slope_rad" in entity.attributes
        )
        set_feature(
            "roughness_m",
            entity.attributes.get("roughness_m", 0.0),
            "roughness_m" in entity.attributes,
        )
        entity_map = np.asarray(entity.xyz_m, dtype=np.float64)
        candidate_sigma = sigma_for(entity)
        restricted_clearance = candidate_clearance(entity_map, entity, restricted, candidate_sigma)
        obstacle_clearance = candidate_clearance(entity_map, entity, obstacles, candidate_sigma)
        track_clearance = candidate_clearance(entity_map, entity, track_rows, candidate_sigma)
        set_feature(
            "restricted_boundary_clearance_m",
            restricted_clearance or 0.0,
            restricted_clearance is not None,
        )
        set_feature(
            "track_boundary_clearance_m", track_clearance or 0.0, track_clearance is not None
        )
        set_feature(
            "nearest_obstacle_clearance_m",
            obstacle_clearance or 0.0,
            obstacle_clearance is not None,
        )
        # Free radius is meaningful only for mapped landing/suitable-terrain
        # candidates and only when both obstacle and restricted-zone evidence
        # exist. It is centered on this candidate, never on the UAV.
        if (
            kind in {"landing", "terrain_region"}
            and obstacle_clearance is not None
            and restricted_clearance is not None
        ):
            set_feature(
                "free_radius_m",
                max(0.0, min(obstacle_clearance, restricted_clearance) - vehicle_radius_m),
            )

        def density_at(point_map, source_rows, radius_m):
            if not source_rows:
                return None
            count = sum(
                float(np.linalg.norm(np.asarray(point_map[:2]) - np.asarray(source[3].xyz_m[:2])))
                - max(0.0, source[3].extent_m)
                - 2.0 * sigma_for(source[3])
                <= radius_m
                for source in source_rows
                if source[3].entity_id != entity.entity_id
            )
            return count / (math.pi * radius_m**2)

        set_feature(
            "cone_density_2m_entities_per_m2", density_at(entity_map, cones, 2.0), bool(cones)
        )
        set_feature(
            "obstacle_density_2m_entities_per_m2",
            density_at(entity_map, obstacles, 2.0),
            bool(obstacles),
        )
        if use_visibility:
            for code, flag_name in enumerate(
                (
                    "visibility_unknown",
                    "visibility_visible",
                    "visibility_behind_camera",
                    "visibility_outside_fov",
                    "visibility_occluded",
                    "visibility_depth_inconsistent",
                )
            ):
                set_feature(flag_name, float(int(visibility) == code))
        set_feature("age_s", max(0.0, timestamp_s - entity.last_seen_s))
        if use_uncertainty and entity.covariance_valid:
            covariance = np.asarray(entity.position_covariance_m2, dtype=np.float64)
            set_feature("horizontal_sigma_m", sigma_for(entity))
            set_feature("vertical_sigma_m", math.sqrt(max(0.0, float(covariance[2, 2]))))
        set_feature("semantic_entropy", entity.semantic_entropy, use_uncertainty)
        set_feature(
            "evidence_support_scaled",
            min(entity.semantic_support / local_map.max_semantic_evidence, 1.0),
            use_uncertainty,
        )
        set_feature("orientation_valid", float(yaw_valid))
        clearance = _surface_clearance(distance, entity.extent_m)
        if use_uncertainty:
            clearance -= 2.0 * sigma_for(entity)
        restricted_score = 1.0 / (1.0 + max(0.0, clearance)) if priority == 0 else 0.0
        uncertainty_score = min(sigma_for(entity) / 2.0, 1.0) if use_uncertainty else 0.0
        high_uncertainty_near_landing = (
            use_uncertainty
            and sigma_for(entity) >= protected_uncertainty_sigma_m
            and any(
                float(np.linalg.norm(delta_map[:2] - center))
                <= landing_protection_radius_m + extent
                for center, extent in landing_regions
            )
        )
        if (
            (kind == "poi" and entity.confidence >= protected_poi_threshold)
            or hazard_score >= protected_hazard_threshold
            or high_uncertainty_near_landing
        ):
            # Safety/mission-critical candidates rank ahead of ordinary tokens
            # when the fixed K budget prunes the remainder.
            priority = 0
        novelty_score = 1.0 - min(entity.semantic_support / local_map.max_semantic_evidence, 1.0)
        poi_score = entity.confidence if kind == "poi" else 0.0
        landing_score = entity.confidence if kind == "landing" else 0.0
        importance = (
            configured_weights["restricted_proximity"] * restricted_score
            + configured_weights["uncertainty"] * uncertainty_score
            + configured_weights["novelty"] * novelty_score
            + configured_weights["poi_evidence"] * poi_score
            + configured_weights["landing_evidence"] * landing_score
            + configured_weights["hazard_evidence"] * hazard_score
        )
        set_feature("importance_score", importance)
        if velocity_valid and kind in {"landing", "terrain_region"}:
            drifted_map = entity_map.copy()
            drifted_map[:2] += drift_map
            drifted_restricted = candidate_clearance(
                drifted_map, entity, restricted, candidate_sigma
            )
            drifted_track = candidate_clearance(drifted_map, entity, track_rows, candidate_sigma)
            drifted_obstacle = candidate_clearance(drifted_map, entity, obstacles, candidate_sigma)
            set_feature(
                "drifted_track_clearance_m", drifted_track or 0.0, drifted_track is not None
            )
            set_feature(
                "drifted_obstacle_clearance_m",
                drifted_obstacle or 0.0,
                drifted_obstacle is not None,
            )
            set_feature(
                "drifted_restricted_clearance_m",
                drifted_restricted or 0.0,
                drifted_restricted is not None,
            )
            if drifted_obstacle is not None and drifted_restricted is not None:
                margin = (
                    min(drifted_obstacle, drifted_restricted) - vehicle_radius_m - safety_margin_m
                )
                set_feature("drifted_landing_margin_m", margin)
        set_feature("drift_valid", float(velocity_valid), velocity_valid)
        candidate_rows.append(
            (
                priority,
                -importance,
                distance,
                -entity.confidence,
                entity.entity_id,
                row,
                feature_validity,
                np.asarray(entity.xyz_m, dtype=np.float64),
                visibility,
            )
        )

    candidate_rows.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
    matrix = np.zeros((MAX_CANDIDATES, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32)
    candidate_validity = np.zeros(MAX_CANDIDATES, dtype=np.float32)
    candidate_feature_validity = np.zeros_like(matrix)
    candidate_map_xyz = np.zeros((MAX_CANDIDATES, 3), dtype=np.float64)
    candidate_visibility = np.zeros(MAX_CANDIDATES, dtype=np.int64)
    selected_rows = candidate_rows[:MAX_CANDIDATES]
    for i, (_, _, _, _, _, row, feature_mask, map_xyz, visibility) in enumerate(selected_rows):
        matrix[i] = row
        candidate_validity[i] = 1.0
        candidate_feature_validity[i] = feature_mask
        candidate_map_xyz[i] = map_xyz
        candidate_visibility[i] = int(visibility)

    return RelationalFeatures(
        names=RELATIONAL_NAMES,
        values=tuple(values.tolist()),
        validity=tuple(validity.tolist()),
        candidates=tuple(tuple(row) for row in matrix.tolist()),
        candidate_validity=tuple(candidate_validity.tolist()),
        candidate_map_xyz=tuple(
            tuple(float(value) for value in row) for row in candidate_map_xyz.tolist()
        ),
        candidate_visibility=tuple(int(value) for value in candidate_visibility.tolist()),
        candidate_tokens_before=tokens_before,
        candidate_tokens_after=int(candidate_validity.sum()),
        merged_region_count=merged_count,
        candidate_feature_validity=tuple(tuple(row) for row in candidate_feature_validity.tolist()),
    )


def build_candidate_tokens(local_map: PersistentSemanticMap, **kwargs):
    result = compute_relations(local_map, **kwargs)
    return (
        np.asarray(result.candidates, dtype=np.float32),
        np.asarray(result.candidate_validity, dtype=np.float32),
        np.asarray(result.candidate_feature_validity, dtype=np.float32),
    )
