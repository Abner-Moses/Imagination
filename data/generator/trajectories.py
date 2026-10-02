"""Smooth, deterministic UAV trajectories expressed in an ENU world frame."""

from __future__ import annotations

import math
import random
from typing import Any, Callable


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def make_trajectory(
    seed: int, config: dict[str, Any], height_fn: Callable[[float, float], float]
) -> tuple[str, list[dict[str, float]]]:
    rng = random.Random(seed ^ 0x7A19EC70)
    tcfg = config["trajectory"]
    fps = float(config["timing"]["fps"])
    segments = int(config["dataset"]["trajectories_per_layout"])
    per_segment = int(config["dataset"]["frames_per_trajectory"])
    frames = segments * per_segment
    types = list(tcfg["types"])
    kind = types[seed % len(types)]
    speed = rng.uniform(*tcfg["nominal_speed_mps"])
    start_x, start_y = rng.uniform(-24.0, 24.0), rng.uniform(-12.0, 12.0)
    heading0 = rng.uniform(-math.pi, math.pi)
    target_clearance = rng.uniform(1.35, 2.65)
    phase = rng.uniform(-math.pi, math.pi)
    curve_omega = rng.choice((-1.0, 1.0)) * 0.12
    poses: list[dict[str, float]] = []
    previous_yaw = heading0
    for index in range(frames):
        t = index / fps
        if kind == "straight":
            x = start_x + speed * t * math.cos(heading0)
            y = start_y + speed * t * math.sin(heading0)
            dx, dy = speed * math.cos(heading0), speed * math.sin(heading0)
        elif kind == "diagonal":
            angle = (math.pi / 4.0) * (1 if math.sin(heading0) >= 0 else -1) + (
                0 if math.cos(heading0) >= 0 else math.pi
            )
            x, y = start_x + speed * t * math.cos(angle), start_y + speed * t * math.sin(angle)
            dx, dy = speed * math.cos(angle), speed * math.sin(angle)
        elif kind == "curved":
            omega = curve_omega
            angle = heading0 + omega * t
            radius = speed / abs(omega)
            sign = 1.0 if omega > 0 else -1.0
            x = start_x + sign * radius * (math.sin(angle) - math.sin(heading0))
            y = start_y - sign * radius * (math.cos(angle) - math.cos(heading0))
            dx, dy = speed * math.cos(angle), speed * math.sin(angle)
        elif kind == "lawnmower":
            # A smooth local S-turn; longer sequences naturally form repeated mowing lanes.
            x = start_x + speed * t * math.cos(heading0)
            lateral = 2.2 * math.sin(0.24 * t)
            y = start_y + speed * t * math.sin(heading0)
            x += -math.sin(heading0) * lateral
            y += math.cos(heading0) * lateral
            dl = 2.2 * 0.24 * math.cos(0.24 * t)
            dx = speed * math.cos(heading0) - math.sin(heading0) * dl
            dy = speed * math.sin(heading0) + math.cos(heading0) * dl
        elif kind == "smooth_random_waypoint":
            x = start_x + speed * t * math.cos(heading0) + 1.4 * math.sin(0.31 * t + phase)
            y = start_y + speed * t * math.sin(heading0) + 1.1 * math.sin(0.23 * t + phase * 0.7)
            dx = speed * math.cos(heading0) + 1.4 * 0.31 * math.cos(0.31 * t + phase)
            dy = speed * math.sin(heading0) + 1.1 * 0.23 * math.cos(0.23 * t + phase * 0.7)
        else:  # hover_rotate_translate
            transition = _smoothstep(min(1.0, max(0.0, (t - 0.6) / 1.2)))
            distance = speed * max(0.0, t - 0.6) * transition
            x = start_x + distance * math.cos(heading0)
            y = start_y + distance * math.sin(heading0)
            dx = max(0.03, speed * transition) * math.cos(heading0)
            dy = max(0.03, speed * transition) * math.sin(heading0)
        x = max(-66.0, min(66.0, x))
        y = max(-28.0, min(28.0, y))
        desired_yaw = math.atan2(dy, dx)
        if kind == "hover_rotate_translate" and t < 1.5:
            desired_yaw = heading0 + math.radians(18.0) * t
        delta = _wrap(desired_yaw - previous_yaw)
        max_delta = math.radians(float(tcfg["max_yaw_rate_degrees_s"])) / fps
        yaw = previous_yaw + max(-max_delta, min(max_delta, delta))
        previous_yaw = yaw
        clearance = target_clearance + 0.16 * math.sin(0.47 * t + phase)
        clearance = max(
            float(tcfg["min_ground_clearance_m"]) + 0.05,
            min(float(tcfg["max_ground_clearance_m"]) - 0.05, clearance),
        )
        ground_z = height_fn(x, y)
        roll = math.radians(2.2) * math.sin(0.37 * t + phase)
        pitch = math.radians(2.8) * math.sin(0.29 * t + phase * 0.5)
        poses.append(
            {
                "frame_index": index,
                "timestamp_s": index / fps,
                "dt_s": 0.0 if index == 0 else 1.0 / fps,
                "x": x,
                "y": y,
                "z": ground_z + clearance,
                "ground_z": ground_z,
                "ground_clearance_m": clearance,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw,
            }
        )
    return kind, poses
