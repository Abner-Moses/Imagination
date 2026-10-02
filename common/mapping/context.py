"""Build causal, prediction-only map context for ordered episode frames."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math

import numpy as np
import torch

from .online import OnlinePerceptionMap
from .visibility import classify_map_point, Visibility


def _project_candidates(result, geometry, *, use_visibility: bool = True):
    """Project prior-map tokens into the current 32x32 camera grid.

    This prediction-only pass uses pose and calibration, never simulator depth
    or labels. Training later gates supervision against a separate cached
    simulator depth target; that target is not returned to the model.
    """
    coordinates = torch.zeros((len(result.candidate_validity), 2), dtype=torch.float32)
    projected_depth = torch.zeros(len(result.candidate_validity), dtype=torch.float32)
    valid = torch.zeros(len(result.candidate_validity), dtype=torch.float32)
    visibility = torch.full(
        (len(result.candidate_validity),), int(Visibility.UNKNOWN), dtype=torch.int64
    )
    if not bool(geometry["camera_pose_valid"]):
        return coordinates, projected_depth, valid, visibility

    pose = geometry["camera_to_local_map"].numpy().astype("float64", copy=False)
    if pose.shape != (4, 4):
        return coordinates, projected_depth, valid, visibility
    try:
        inverse_pose = np.linalg.inv(pose)
    except np.linalg.LinAlgError:
        return coordinates, projected_depth, valid, visibility
    calibration = geometry["calibration"].tolist()
    width, height, fx, fy, cx, cy = calibration
    # Model-facing candidate XYZ is egocentric; projection always uses the
    # separate map-frame coordinates carried in the typed result field.
    map_points = np.asarray(result.candidate_map_xyz, dtype=np.float64)
    candidate_validity = np.asarray(result.candidate_validity) > 0
    if map_points.shape != (len(candidate_validity), 3):
        raise ValueError("Candidate projection requires explicit map-frame XYZ for every token")
    fx_grid, fy_grid = fx * 32 / width, fy * 32 / height
    cx_grid, cy_grid = (cx + 0.5) * 32 / width - 0.5, (cy + 0.5) * 32 / height - 0.5

    geometry_depth = geometry["analytical"][22].numpy() * float(geometry["depth_scale_m"])
    geometry_valid = geometry["validity"][22].numpy()
    for index, map_point in enumerate(map_points):
        if not candidate_validity[index]:
            continue
        camera_point = inverse_pose @ np.append(map_point, 1.0)
        x_cam, y_cam, z_cam = camera_point[:3]
        if not math.isfinite(z_cam) or z_cam <= 0:
            continue
        pixel_x = fx_grid * x_cam / z_cam + cx_grid
        pixel_y = fy_grid * y_cam / z_cam + cy_grid
        grid_x, grid_y = int(round(pixel_x)), int(round(pixel_y))
        if not (0 <= grid_x < 32 and 0 <= grid_y < 32):
            continue
        coordinates[index] = torch.tensor((grid_x, grid_y), dtype=torch.float32)
        projected_depth[index] = float(z_cam)
        valid[index] = 1.0
        code, _, _ = classify_map_point(
            map_point,
            pose,
            dict(zip(("width", "height", "fx", "fy", "cx", "cy"), calibration)),
            geometry_depth if use_visibility else None,
            geometry_valid if use_visibility else None,
        )
        if not use_visibility and code == Visibility.UNKNOWN:
            code = Visibility.VISIBLE
        visibility[index] = int(code)
    return coordinates, projected_depth, valid, visibility


def process_dataset_frame(model, family, visual, geometry, online_map, device):
    """Run one frame through the predicted map without reading supervision labels."""
    state = visual["vehicle_state"].unsqueeze(0).to(device)
    state_validity = visual["state_validity"].unsqueeze(0).to(device)
    analytical = geometry["analytical"].unsqueeze(0)
    validity = geometry["validity"].unsqueeze(0)
    pose = geometry["camera_to_local_map"].numpy()
    pose_valid = bool(geometry["camera_pose_valid"])
    calibration_values = geometry["calibration"].tolist()
    intrinsics = dict(zip(("width", "height", "fx", "fy", "cx", "cy"), calibration_values))

    confidence_mask = validity[0, 27].numpy() > 0
    confidence = (
        float(analytical[0, 27].numpy()[confidence_mask].mean()) if confidence_mask.any() else None
    )
    visual_input = analytical if family == "imf_htransformer" else visual["rgb"].unsqueeze(0)
    return online_map.process(
        model,
        family=family,
        visual_input=visual_input,
        validity=validity if family == "imf_htransformer" else None,
        vehicle_state=state,
        state_validity=state_validity,
        timestamp_s=float(visual["metadata"]["timestamp"]),
        intrinsics=intrinsics,
        camera_to_local_map=pose if pose_valid else None,
        depth_normalized=analytical[0, 22].numpy(),
        depth_validity=validity[0, 22].numpy(),
        depth_scale_m=float(geometry["depth_scale_m"]),
        device=device,
        geometry_confidence=confidence,
    )


def build_predicted_contexts(
    model,
    family: str,
    visual_data,
    geometry_data,
    device: torch.device,
    map_config: str | Path,
    progress=None,
    max_samples: int | None = None,
    map_overrides: dict | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Roll each episode forward using only earlier model predictions.

    Context is a frozen, per-epoch snapshot. Labels are not opened by either
    dataset passed here; training and evaluation load them separately after this
    function has produced the causal relation inputs.
    """
    if [row["sample_id"] for row in visual_data.records] != [
        row["sample_id"] for row in geometry_data.records
    ]:
        raise ValueError("Visual and geometry datasets must have identical sample order")

    episodes = defaultdict(list)
    for index, record in enumerate(visual_data.records):
        episodes[record["episode_id"]].append((record["frame_index"], index, record))

    selected = []
    for episode_id in sorted(episodes):
        selected.extend(sorted(episodes[episode_id]))
        if max_samples is not None and len(selected) >= max_samples:
            selected = selected[:max_samples]
            break
    selected_indices = {row[1] for row in selected}

    contexts = {}
    model.eval()
    total = len(selected)
    completed = 0
    with torch.inference_mode():
        for episode_id in sorted(episodes):
            online_map = OnlinePerceptionMap.from_config(map_config, overrides=map_overrides)
            frames = sorted(episodes[episode_id])
            for _, index, record in frames:
                if index not in selected_indices:
                    continue
                visual = visual_data[index]
                geometry = geometry_data[index]
                result = process_dataset_frame(model, family, visual, geometry, online_map, device)
                (
                    candidate_grid,
                    candidate_projected_depth,
                    candidate_projection_validity,
                    candidate_visibility,
                ) = _project_candidates(
                    result,
                    geometry,
                    use_visibility=bool((map_overrides or {}).get("use_visibility", True)),
                )
                contexts[record["sample_id"]] = {
                    "relational": torch.tensor(result.relation_values, dtype=torch.float32),
                    "relational_validity": torch.tensor(
                        result.relation_validity, dtype=torch.float32
                    ),
                    "candidates": torch.tensor(result.candidate_values, dtype=torch.float32),
                    "candidate_validity": torch.tensor(
                        result.candidate_validity, dtype=torch.float32
                    ),
                    "candidate_feature_validity": torch.tensor(
                        result.candidate_feature_validity
                        if np.asarray(result.candidate_feature_validity).shape
                        == np.asarray(result.candidate_values).shape
                        else np.zeros_like(np.asarray(result.candidate_values), dtype=np.float32),
                        dtype=torch.float32,
                    ),
                    "candidate_grid": candidate_grid,
                    "candidate_projected_depth_m": candidate_projected_depth,
                    "candidate_projection_validity": candidate_projection_validity,
                    "candidate_visibility": candidate_visibility,
                }
                completed += 1
                if progress:
                    progress(completed, total)
    return contexts
