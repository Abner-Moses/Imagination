"""Pure-Python episode specifications used by generation and reproducibility checks."""

from __future__ import annotations

import math
import random
from typing import Any

from generator.clutter import generate_clutter
from generator.trajectories import make_trajectory
from scene.terrain import ground_surface_height, make_terrain_parameters, terrain_height


def episode_seed(base_seed: int, episode_number: int) -> int:
    return int(base_seed) + int(episode_number) * 1_000_003


def _view_mode(seed: int, config: dict[str, Any]) -> str:
    modes = list(config.get("sampling", {}).get("view_modes", ["field_interior"]))
    return modes[(seed ^ 0x61A7C0DE) % len(modes)]


def _retarget_boundary_trajectory(
    poses: list[dict[str, float]],
    mode: str,
    terrain: dict[str, Any],
    config: dict[str, Any],
    rng: random.Random,
):
    """Move a smooth trajectory onto/near the oval boundary without teleporting."""
    if mode not in {"near_track", "boundary_mixed", "track_surface"}:
        return
    ax = float(config["field"]["grass_length_m"]) / 2.0
    ay = float(config["field"]["grass_width_m"]) / 2.0
    theta = rng.uniform(-math.pi, math.pi)
    direction = rng.choice((-1.0, 1.0))
    speed = rng.uniform(*config["trajectory"]["nominal_speed_mps"])
    base_offset = {"near_track": -1.35, "boundary_mixed": 0.0, "track_surface": 4.0}[mode]
    for pose in poses:
        bx, by = ax * math.cos(theta), ay * math.sin(theta)
        gx, gy = math.cos(theta) / ax, math.sin(theta) / ay
        magnitude = math.hypot(gx, gy)
        nx, ny = gx / magnitude, gy / magnitude
        offset = base_offset
        if mode == "boundary_mixed":
            offset += 0.32 * math.sin(0.31 * float(pose["timestamp_s"]) + rng.random() * 0.02)
        x, y = bx + nx * offset, by + ny * offset
        tx, ty = -ax * math.sin(theta) * direction, ay * math.cos(theta) * direction
        yaw = math.atan2(ty, tx)
        ground_z = ground_surface_height(x, y, terrain, config)
        pose.update(
            {
                "x": x,
                "y": y,
                "yaw": yaw,
                "ground_z": ground_z,
                "z": ground_z + float(pose["ground_clearance_m"]),
            }
        )
        tangent_length = max(math.hypot(-ax * math.sin(theta), ay * math.cos(theta)), 1e-6)
        theta += (
            direction
            * speed
            * float(pose["dt_s"] or (1.0 / config["timing"]["fps"]))
            / tangent_length
        )


def build_episode_spec(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(seed ^ 0x119AC011)
    terrain = make_terrain_parameters(seed, config)
    trajectory_type, poses = make_trajectory(
        seed, config, lambda x, y: terrain_height(x, y, terrain)
    )
    review = config.get("review", {})
    view_mode = "visual_review" if review.get("enabled") else _view_mode(seed, config)
    if not review.get("enabled"):
        _retarget_boundary_trajectory(poses, view_mode, terrain, config, rng)
        if view_mode == "field_hazard" and poses:
            anchor = poses[len(poses) // 2]
            terrain["broken_zones"].append(
                {
                    "x": float(anchor["x"]),
                    "y": float(anchor["y"]),
                    "radius_x": rng.uniform(1.1, 2.2),
                    "radius_y": rng.uniform(0.65, 1.35),
                    "angle": rng.uniform(-math.pi, math.pi),
                    "amplitude": rng.uniform(-0.06, -0.025),
                    "severity": rng.uniform(0.82, 1.0),
                }
            )
            for pose in poses:
                ground_z = terrain_height(pose["x"], pose["y"], terrain)
                pose["ground_z"] = ground_z
                pose["z"] = ground_z + pose["ground_clearance_m"]
    if review.get("enabled") and poses:
        review_index = int(review.get("index", 0))
        # Interior object views, an approach to the edge, two deliberately
        # mixed grass/track views, then track-dominant boundary violations.
        positions = [
            (0, 0, 0.15),
            (-34, -12, 0.55),
            (32, 13, -2.45),
            (-20, 18, math.pi / 2),
            (24, -18, -math.pi / 2),
            (-50, 10, math.pi),
            (50, -10, 0),
            (0, 38.2, 0.65),
            (56.4, 28.2, -0.9),
            (0, 42.0, 2.8),
            (83.0, 0, -0.35),
            (0, -44.0, 1.35),
        ]
        clearances = [1.05, 1.25, 1.48, 1.72, 1.95, 2.18, 2.42, 2.62, 2.82, 1.38, 2.08, 2.95]
        x, y, yaw = positions[review_index % len(positions)]
        if review_index == 6:
            terrain["broken_zones"].append(
                {
                    "x": float(x),
                    "y": float(y),
                    "radius_x": 1.45,
                    "radius_y": 0.85,
                    "angle": 0.42,
                    "amplitude": -0.055,
                    "severity": 1.0,
                }
            )
        clearance = clearances[review_index % len(clearances)]
        ground_z = ground_surface_height(x, y, terrain, config)
        poses[0].update(
            {
                "x": float(x),
                "y": float(y),
                "yaw": yaw,
                "roll": 0.0,
                "pitch": 0.0,
                "ground_z": ground_z,
                "ground_clearance_m": clearance,
                "z": ground_z + clearance,
            }
        )
    style, density_profile, objects = generate_clutter(seed, config, poses, view_mode)
    for record in objects:
        record["position"][2] = round(
            terrain_height(record["position"][0], record["position"][1], terrain), 6
        )
    conditions = ["bright_sun", "soft_daylight", "long_shadows", "overcast"]
    condition_index = int(review.get("index", seed)) if review.get("enabled") else seed
    condition = conditions[condition_index % len(conditions)]
    presets = {
        "bright_sun": (48.0, 2.35, 1.8, 0.065, 5600),
        "soft_daylight": (58.0, 1.55, 5.5, 0.09, 6100),
        "long_shadows": (19.0, 1.95, 2.8, 0.065, 4800),
        "overcast": (43.0, 0.55, 15.0, 0.13, 6700),
    }
    elevation, energy, angle, world_strength, kelvin = presets[condition]
    lighting = {
        "condition": condition,
        "sun_azimuth_deg": round(rng.uniform(20.0, 340.0), 5),
        "sun_elevation_deg": round(elevation + rng.uniform(-4.0, 4.0), 5),
        "sun_energy": round(energy * rng.uniform(0.9, 1.1), 5),
        "sun_angle_deg": angle,
        "world_strength": round(world_strength * rng.uniform(0.92, 1.08), 5),
        "color_temperature_k": kelvin + rng.randint(-180, 180),
        "turbidity": round(rng.uniform(2.2, 6.5) + (2.0 if condition == "overcast" else 0.0), 4),
        "exposure_ev": round(rng.uniform(-0.65, -0.28), 4),
    }
    review_focuses = [
        "training cone",
        "football",
        "rock",
        "bag",
        "box",
        "cone exclusion zone",
        "broken ground",
        "near track",
        "half grass / half track",
        "track north side",
        "track east side",
        "track south side",
    ]
    return {
        "seed": seed,
        "terrain": terrain,
        "clutter_style": style,
        "clutter_density_profile": density_profile,
        "view_mode": view_mode,
        "objects": objects,
        "trajectory_type": trajectory_type,
        "poses": poses,
        "lighting": lighting,
        "review_focus": review_focuses[int(review.get("index", 0)) % len(review_focuses)]
        if review.get("enabled")
        else None,
    }
