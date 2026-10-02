"""Compare analytical local-map trajectory/POIs with saved simulator world truth."""

from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np

from common.runtime import save_json
from data.adapter import load_manifest


def _world_from_local(first_pose):
    # The C++ local axes at the first nadir frame are +X camera-right,
    # +Y opposite OpenCV camera-down and +Z opposite OpenCV camera-forward.
    # Those are exactly Blender camera-local +X,+Y,+Z, so the saved Blender
    # camera-to-world quaternion directly maps the C++ local map into world ENU.
    world_from_camera = _quaternion_rotation(
        *[float(first_pose[key]) for key in ("camera_qw", "camera_qx", "camera_qy", "camera_qz")]
    )
    rotation = world_from_camera
    translation = np.array([float(first_pose[x]) for x in ("x", "y", "z")])
    return rotation, translation


def _quaternion_rotation(w, x, y, z):
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        float,
    )


def _truth_at_grid(record, pose, grid_xy, manifest_dir, calibration):
    if "depth_path" not in record:
        return None
    depth = np.load((manifest_dir / record["depth_path"]).resolve(), allow_pickle=False)
    x, y = grid_xy
    u = min(depth.shape[1] - 1, max(0, int(round((x + 0.5) * depth.shape[1] / 32 - 0.5))))
    v = min(depth.shape[0] - 1, max(0, int(round((y + 0.5) * depth.shape[0] / 32 - 0.5))))
    distance = float(depth[v, u])
    if not np.isfinite(distance) or distance <= 0:
        return None
    ray = np.array(
        [
            (u - calibration["cx"]) / calibration["fx"],
            -(v - calibration["cy"]) / calibration["fy"],
            -1.0,
        ]
    )
    ray = ray / np.linalg.norm(ray) * distance
    rotation = _quaternion_rotation(
        *[float(pose[key]) for key in ("camera_qw", "camera_qx", "camera_qy", "camera_qz")]
    )
    origin = np.array([float(pose[key]) for key in ("x", "y", "z")])
    return origin + rotation @ ray


def benchmark(manifest_path, mapped_path, output):
    manifest = Path(manifest_path)
    records = load_manifest(manifest)
    by_episode = {}
    for r in records:
        by_episode.setdefault(r["episode_id"], []).append(r)
    mapped = json.loads(Path(mapped_path).read_text())
    trajectory_errors = []
    projection_errors = []
    poi_errors = []
    landing_errors = []
    matched_objects = set()
    predicted_pois = 0
    valid_geometry = total_geometry = 0
    calibration = json.loads(
        (manifest.parent / "../calibration/blender_camera.json").resolve().read_text()
    )
    for episode, features in mapped.items():
        rows = sorted(by_episode[episode], key=lambda x: x["frame_index"])
        pose_file = (manifest.parent / rows[0]["pose_path"]).resolve()
        with pose_file.open(newline="") as stream:
            poses = list(csv.DictReader(stream))
        rotation, translation = _world_from_local(poses[0])
        objects = json.loads((manifest.parent / rows[0]["objects_path"]).resolve().read_text())[
            "objects"
        ]
        for row, truth in zip(rows, poses):
            cache = (manifest.parent / row["analytical_path"]).resolve()
            if not cache.exists():
                continue
            with np.load(cache, allow_pickle=False) as data:
                pose = data["camera_to_local_map"]
                valid_geometry += int(bool(data["geometry_valid"]))
                total_geometry += 1
            if pose.shape == (4, 4):
                predicted = rotation @ pose[:3, 3] + translation
                target = np.array([float(truth[x]) for x in ("x", "y", "z")])
                trajectory_errors.append(float(np.linalg.norm(predicted - target)))
        for feature in features:
            if not feature["resolved"]:
                continue
            world = rotation @ np.asarray(feature["map_xyz_m"]) + translation
            frame = int(
                np.argmin(
                    [abs(float(p["timestamp_s"]) - feature["last_seen_timestamp"]) for p in poses]
                )
            )
            truth_point = _truth_at_grid(
                rows[frame], poses[frame], feature["source_grid_xy"], manifest.parent, calibration
            )
            if truth_point is not None:
                error = float(np.linalg.norm(world - truth_point))
                projection_errors.append(error)
                if feature["kind"] == "landing":
                    landing_errors.append(error)
            if feature["kind"] != "poi":
                continue
            predicted_pois += 1
            candidates = [
                o
                for o in objects
                if int(o["class_id"]) == rows[0]["poi_classes"].get(feature["class_name"])
            ]
            if candidates:
                distances = [
                    float(np.linalg.norm(world - np.asarray(o["position"]))) for o in candidates
                ]
                index = int(np.argmin(distances))
                poi_errors.append(distances[index])
                matched_objects.add((episode, candidates[index]["object_id"]))
    duplicate_rate = 1 - len(matched_objects) / predicted_pois if predicted_pois else None
    result = {
        "trajectory_ate_mae_m": float(np.mean(trajectory_errors)) if trajectory_errors else None,
        "trajectory_ate_rmse_m": float(np.sqrt(np.mean(np.square(trajectory_errors))))
        if trajectory_errors
        else None,
        "sparse_geometry_frame_coverage": valid_geometry / total_geometry
        if total_geometry
        else 0.0,
        "projection_surface_mae_m": float(np.mean(projection_errors))
        if projection_errors
        else None,
        "poi_3d_position_mae_m": float(np.mean(poi_errors)) if poi_errors else None,
        "poi_3d_position_rmse_m": float(np.sqrt(np.mean(np.square(poi_errors))))
        if poi_errors
        else None,
        "landing_3d_position_mae_m": float(np.mean(landing_errors)) if landing_errors else None,
        "landing_3d_position_rmse_m": float(np.sqrt(np.mean(np.square(landing_errors))))
        if landing_errors
        else None,
        "duplicate_association_rate": duplicate_rate,
        "matched_poi_objects": len(matched_objects),
        "resolved_poi_entries": predicted_pois,
        "truth_method": "simulator depth plus exact saved camera quaternion; unresolved observations remain explicit",
    }
    save_json(output, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--mapped", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(benchmark(a.manifest, a.mapped, a.output), indent=2))


if __name__ == "__main__":
    main()
