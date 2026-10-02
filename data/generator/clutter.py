"""Deterministic clutter layouts with separation and field-boundary constraints."""

from __future__ import annotations

import math
import random
from typing import Any

CLASS_IDS = {
    "cone": 3,
    **{f"rock_{i}": 4 for i in range(1, 8)},
    "football": 5,
    "backpack": 6,
    "box": 7,
    "pole": 8,
    "chair": 9,
}
RADII = {
    "cone": 0.26,
    "rock_1": 0.24,
    "rock_2": 0.16,
    "rock_3": 0.18,
    "rock_4": 0.16,
    "rock_5": 0.19,
    "rock_6": 0.18,
    "rock_7": 0.18,
    "football": 0.13,
    "backpack": 0.38,
    "box": 0.42,
    "pole": 0.16,
    "chair": 0.48,
}


def _inside(x: float, y: float) -> bool:
    return (x / 76.0) ** 2 + (y / 36.0) ** 2 < 1.0


def _candidate(rng: random.Random, style: str, index: int, count: int) -> tuple[float, float]:
    if style == "clusters":
        centers = [(-18.0, -8.0), (12.0, 10.0), (31.0, -6.0)]
        cx, cy = centers[index % len(centers)]
        return cx + rng.gauss(0, 3.4), cy + rng.gauss(0, 2.5)
    if style == "corridors":
        side = -1.0 if index % 2 else 1.0
        return rng.uniform(-42.0, 42.0), side * rng.uniform(5.0, 8.0) + rng.gauss(0, 0.45)
    if style == "slalom":
        x = -32.0 + 64.0 * (index + 0.5) / max(count, 1)
        return x + rng.gauss(0, 0.3), (3.8 if index % 2 else -3.8) + rng.gauss(0, 0.35)
    if style == "gates":
        gate = index // 2
        x = -30.0 + 60.0 * (gate + 0.5) / max((count + 1) // 2, 1)
        return x + rng.gauss(0, 0.2), (-3.0 if index % 2 else 3.0) + rng.gauss(0, 0.2)
    if style == "mixed":
        return _candidate(rng, ("clusters", "corridors", "random_scatter")[index % 3], index, count)
    for _ in range(100):
        x, y = rng.uniform(-66.0, 66.0), rng.uniform(-28.0, 28.0)
        if _inside(x, y):
            return x, y
    return 0.0, 0.0


def generate_clutter(
    seed: int,
    config: dict[str, Any],
    trajectory_poses: list[dict] | None = None,
    view_mode: str = "field_interior",
) -> tuple[str, str, list[dict[str, Any]]]:
    rng = random.Random(seed ^ 0xC1077E12)
    cfg = config["clutter"]
    styles = list(cfg["styles"])
    style = styles[seed % len(styles)]
    review = config.get("review", {})
    review_index = int(review.get("index", 0))
    profiles = list(cfg.get("density_profiles", ["mixed_medium"]))
    density_profile = (
        "visual_review" if review.get("enabled") else profiles[(seed ^ 0xD3517E) % len(profiles)]
    )
    ranges = cfg.get("profile_object_ranges", {})
    count_range = ranges.get(density_profile, [cfg["min_objects"], cfg["max_objects"]])
    count = rng.randint(int(count_range[0]), int(count_range[1]))
    if density_profile == "cone_dense":
        style = rng.choice(("clusters", "gates", "slalom"))
    types = list(cfg["asset_types"])
    min_sep = float(cfg["min_separation_m"])
    if density_profile in {"cone_dense", "mixed_dense"}:
        min_sep = min(min_sep, 0.62)
    if review.get("enabled"):
        trajectory_object_limit = 4 if review_index == 5 else 3
    elif view_mode in {"field_interior", "field_hazard"}:
        trajectory_object_limit = {
            "cone_dense": 8,
            "mixed_dense": 6,
            "cone_sparse": 2,
            "mixed_sparse": 2,
        }.get(density_profile, 4)
    else:
        trajectory_object_limit = 0
    objects: list[dict[str, Any]] = []
    attempts = 0
    failed_for_current_index = 0
    while len(objects) < count and attempts < count * 140:
        attempts += 1
        failed_for_current_index += 1
        index = len(objects)
        x, y = _candidate(rng, style, index, count)
        # Structured layouts can exhaust the local slot for one index. After
        # repeated collisions, retain the same density/type profile but find a
        # valid position elsewhere instead of deterministically deadlocking.
        if failed_for_current_index > 28:
            x, y = _candidate(rng, "random_scatter", index, count)
        fallback_x, fallback_y = x, y
        # Keep each flight informative without moving clutter: reserve a few
        # static obstacles near (not directly on) the planned flight corridor.
        if (
            trajectory_poses
            and index < min(trajectory_object_limit, count)
            and attempts <= max(24, trajectory_object_limit * 24)
        ):
            pose_index = min(
                len(trajectory_poses) - 1,
                round(
                    (index + 1) * (len(trajectory_poses) - 1) / max(trajectory_object_limit + 1, 1)
                ),
            )
            pose = trajectory_poses[pose_index]
            if review.get("enabled") and review_index == 5 and index < 4:
                offsets = ((-0.58, -0.46), (0.58, -0.46), (-0.58, 0.46), (0.58, 0.46))
                x = trajectory_poses[0]["x"] + offsets[index][0]
                y = trajectory_poses[0]["y"] + offsets[index][1]
                forward = lateral = None
            elif review.get("enabled") and review_index <= 4 and index == 0:
                # Put one representative asset inside a nadir camera footprint.
                # It remains a normal fixed scene object; only the review layout
                # deliberately guarantees that it is visible.
                forward = 0.10
                lateral = rng.choice((-1.0, 1.0)) * 0.12
            elif density_profile == "cone_dense" and index < 8:
                offsets = (
                    (-1.38, -0.46),
                    (-0.46, -0.46),
                    (0.46, -0.46),
                    (1.38, -0.46),
                    (-1.38, 0.46),
                    (-0.46, 0.46),
                    (0.46, 0.46),
                    (1.38, 0.46),
                )
                pose = trajectory_poses[len(trajectory_poses) // 2]
                x = pose["x"] + offsets[index][0]
                y = pose["y"] + offsets[index][1]
                forward = lateral = None
            else:
                forward = rng.uniform(-0.18, 0.28)
                lateral = rng.choice((-1.0, 1.0)) * rng.uniform(0.18, 0.42)
            if forward is not None:
                x = pose["x"] + forward * math.cos(pose["yaw"]) - lateral * math.sin(pose["yaw"])
                y = pose["y"] + forward * math.sin(pose["yaw"]) + lateral * math.cos(pose["yaw"])
            if not _inside(x, y):
                x, y = fallback_x, fallback_y
        if not _inside(x, y):
            continue
        asset_type = rng.choice(types)
        # Deliberately create cone-heavy exclusion/training zones in the
        # structured layouts while retaining mixed real-world clutter elsewhere.
        if density_profile == "cone_dense" and (index < 8 or rng.random() < 0.76):
            asset_type = "cone"
        elif density_profile == "cone_sparse" and rng.random() < 0.20:
            asset_type = "cone"
        elif style in {"slalom", "gates"} and rng.random() < 0.62:
            asset_type = "cone"
        elif style == "clusters" and rng.random() < 0.36:
            asset_type = "cone"
        if review.get("enabled") and review_index == 5 and index < 4:
            asset_type = "cone"
        elif review.get("enabled") and review_index <= 4 and index == 0:
            review_assets = [
                "cone",
                "football",
                "rock_1",
                "backpack",
                "box",
                "rock_4",
                "chair",
                "pole",
                "rock_6",
            ]
            asset_type = review_assets[review_index % len(review_assets)]
        scale = rng.uniform(0.84, 1.18)
        radius = RADII[asset_type] * scale
        if any(
            math.hypot(x - o["position"][0], y - o["position"][1])
            < (
                (0.18 if asset_type == "cone" and o["asset_type"] == "cone" else min_sep)
                + radius
                + o["radius_m"]
            )
            for o in objects
        ):
            continue
        objects.append(
            {
                "object_id": f"clutter_{index:03d}",
                "asset_type": asset_type,
                "class_id": CLASS_IDS[asset_type],
                "position": [round(x, 6), round(y, 6), 0.0],
                "yaw_rad": round(rng.uniform(-math.pi, math.pi), 8),
                "scale": round(scale, 6),
                "radius_m": round(radius, 6),
                "material_variant": rng.randrange(3),
            }
        )
        failed_for_current_index = 0
    if len(objects) < int(count_range[0]):
        raise RuntimeError(f"Could only place {len(objects)} separated clutter objects")
    return style, density_profile, objects
