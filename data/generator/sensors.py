"""Kinematically consistent clean and noisy UAV sensor samples."""

from __future__ import annotations

import math
import random


def _derivative(
    values: list[tuple[float, float, float]], index: int, dt: float
) -> tuple[float, float, float]:
    if len(values) == 1:
        return (0.0, 0.0, 0.0)
    if index == 0:
        a, b, scale = values[0], values[1], 1.0 / dt
    elif index == len(values) - 1:
        a, b, scale = values[-2], values[-1], 1.0 / dt
    else:
        a, b, scale = values[index - 1], values[index + 1], 0.5 / dt
    return tuple((b[j] - a[j]) * scale for j in range(3))


def compute_clean_states(poses: list[dict], config: dict) -> list[dict]:
    from mathutils import Euler, Vector

    dt = 1.0 / float(config["timing"]["fps"])
    gravity = float(config["sensors"]["gravity_mps2"])
    magnetic_world = Vector(config["sensors"]["magnetic_field_world_uT"])
    positions = [(p["x"], p["y"], p["z"]) for p in poses]
    velocities = [_derivative(positions, i, dt) for i in range(len(poses))]
    accelerations = [_derivative(velocities, i, dt) for i in range(len(poses))]
    quaternions = [Euler((p["roll"], p["pitch"], p["yaw"]), "XYZ").to_quaternion() for p in poses]
    states = []
    for index, pose in enumerate(poses):
        q = quaternions[index]
        if index == 0:
            omega = Vector((0.0, 0.0, 0.0))
        else:
            q_rel = quaternions[index - 1].conjugated() @ q
            if q_rel.w < 0.0:
                q_rel.negate()
            axis, angle = q_rel.to_axis_angle()
            omega = axis * (angle / dt)
        accel_world = Vector(accelerations[index])
        specific_force_body = q.conjugated() @ (accel_world - Vector((0.0, 0.0, -gravity)))
        magnetic_body = q.conjugated() @ magnetic_world
        states.append(
            {
                **pose,
                "qw": q.w,
                "qx": q.x,
                "qy": q.y,
                "qz": q.z,
                "vx": velocities[index][0],
                "vy": velocities[index][1],
                "vz": velocities[index][2],
                "world_ax": accelerations[index][0],
                "world_ay": accelerations[index][1],
                "world_az": accelerations[index][2],
                "imu_ax": specific_force_body.x,
                "imu_ay": specific_force_body.y,
                "imu_az": specific_force_body.z,
                "wx": omega.x,
                "wy": omega.y,
                "wz": omega.z,
                "mx": magnetic_body.x,
                "my": magnetic_body.y,
                "mz": magnetic_body.z,
                "ultrasonic_range_m": float("nan"),
            }
        )
    return states


def noisy_copy(states: list[dict], config: dict, seed: int) -> list[dict]:
    if not config["sensors"]["noise_enabled"]:
        return [dict(row) for row in states]
    rng = random.Random(seed ^ 0x5015E123)
    std = config["sensors"]["noise_std"]
    groups = {
        "position_m": ("x", "y", "z"),
        "velocity_mps": ("vx", "vy", "vz"),
        "acceleration_mps2": ("imu_ax", "imu_ay", "imu_az"),
        "gyro_radps": ("wx", "wy", "wz"),
        "magnetometer_uT": ("mx", "my", "mz"),
        "ultrasonic_m": ("ultrasonic_range_m",),
    }
    result = []
    for row in states:
        noisy = dict(row)
        for std_key, columns in groups.items():
            sigma = float(std[std_key])
            for column in columns:
                noisy[column] += rng.gauss(0.0, sigma)
        result.append(noisy)
    return result
