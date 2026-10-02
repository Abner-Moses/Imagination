if False:
    import blenderproc as bproc  # BlenderProc launcher marker; see README compatibility note.

"""Blender-side entry point. Must begin with BlenderProc import for its launcher."""

import csv
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

# BlenderProc's launcher prepends its host-venv site-packages. The host venv is
# Python 3.14 while Blender 5.2 embeds 3.13, so compiled host wheels must not be
# visible inside Blender. Blender's own NumPy remains available afterwards.
sys.path[:] = [path for path in sys.path if not (".venv" in path and "site-packages" in path)]

import bpy
import numpy as np
from mathutils import Euler, Vector
from mathutils.bvhtree import BVHTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generator.labels import collect_auxiliary_outputs, configure_render_outputs
from generator.sensors import compute_clean_states, noisy_copy
from generator.specs import build_episode_spec, episode_seed
from scene.build_scene import build_episode_scene
from scene.terrain import _ellipse_signed_distance


SENSOR_COLUMNS = [
    "frame_index",
    "timestamp_s",
    "dt_s",
    "x",
    "y",
    "z",
    "ground_z",
    "ground_clearance_m",
    "qw",
    "qx",
    "qy",
    "qz",
    "roll",
    "pitch",
    "yaw",
    "vx",
    "vy",
    "vz",
    "world_ax",
    "world_ay",
    "world_az",
    "imu_ax",
    "imu_ay",
    "imu_az",
    "wx",
    "wy",
    "wz",
    "mx",
    "my",
    "mz",
    "ultrasonic_range_m",
]
POSE_COLUMNS = [
    "frame_index",
    "timestamp_s",
    "dt_s",
    "x",
    "y",
    "z",
    "ground_z",
    "ground_clearance_m",
    "qw",
    "qx",
    "qy",
    "qz",
    "roll",
    "pitch",
    "yaw",
    "camera_qw",
    "camera_qx",
    "camera_qy",
    "camera_qz",
]
NAVIGATION_COLUMNS = [
    "frame_index",
    "navigation_class_id",
    "navigation_class",
    "signed_grass_boundary_distance_m",
    "grass_pixel_fraction",
    "track_pixel_fraction",
    "landing_unsafe_fraction",
    "landing_caution_fraction",
    "landing_safe_fraction",
    "risk_score",
    "landing_permitted",
]


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})


def episode_is_complete(path: Path, expected_frames: int) -> bool:
    """Use the episode itself as the resume record; no side checkpoint file."""
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if int(metadata["frame_count"]) != expected_frames:
            return False
        for directory, suffix in (
            ("rgb", ".jpg"),
            ("depth", ".npy"),
            ("segmentation", ".png"),
            ("segmentation_color", ".png"),
            ("landing_suitability", ".png"),
            ("landing_labels", ".png"),
        ):
            if len(list((path / directory).glob(f"*{suffix}"))) != expected_frames:
                return False
        for filename in (
            "sensors.csv",
            "sensors_clean.csv",
            "poses.csv",
            "navigation_labels.csv",
            "objects.json",
        ):
            if not (path / filename).is_file():
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def configure_renderer(config: dict):
    scene = bpy.context.scene
    render = config["render"]
    requested_engine = render["engine"]
    try:
        scene.render.engine = requested_engine
    except TypeError:
        available = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
        raise RuntimeError(
            f"Render engine {requested_engine!r} unavailable; choices: {list(available)}"
        )
    scene.render.resolution_x = int(render["width"])
    scene.render.resolution_y = int(render["height"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.quality = int(render["jpeg_quality"])
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.look = "AgX - Medium High Contrast"
    if hasattr(scene, "eevee") and hasattr(scene.eevee, "taa_render_samples"):
        scene.eevee.taa_render_samples = int(render["samples"])
    return scene


def set_camera_pose(camera, state: dict, down_tilt_deg: float):
    body_q = Euler((state["roll"], state["pitch"], state["yaw"]), "XYZ").to_quaternion()
    tilt = math.radians(down_tilt_deg)
    view_direction_body = Vector((math.cos(tilt), 0.0, -math.sin(tilt)))
    mount_q = view_direction_body.to_track_quat("-Z", "Y")
    camera_q = body_q @ mount_q
    camera.location = (state["x"], state["y"], state["z"])
    camera.rotation_mode = "QUATERNION"
    camera.rotation_quaternion = camera_q
    state.update(
        {
            "camera_qw": camera_q.w,
            "camera_qx": camera_q.x,
            "camera_qy": camera_q.y,
            "camera_qz": camera_q.z,
        }
    )
    return body_q


def raycast_ultrasonic(scene, body_q, state: dict) -> float:
    origin = Vector((state["x"], state["y"], state["z"]))
    direction = body_q @ Vector((0.0, 0.0, -1.0))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    hit, location, _normal, _face_index, _obj, _matrix = scene.ray_cast(
        depsgraph, origin, direction, distance=12.0
    )
    if not hit:
        raise RuntimeError(f"Ultrasonic ray missed the scene at frame {state['frame_index']}")
    return float((location - origin).length)


def update_mesh_ground_clearance(surface_bvhs, state: dict):
    origin = Vector((state["x"], state["y"], state["z"] + 10.0))
    hits = []
    for surface_bvh in surface_bvhs:
        location, _normal, _index, _distance = surface_bvh.ray_cast(
            origin, Vector((0.0, 0.0, -1.0)), 30.0
        )
        if location is not None:
            hits.append(location)
    if not hits:
        raise RuntimeError(f"Vertical terrain ray missed at frame {state['frame_index']}")
    location = max(hits, key=lambda item: item.z)
    state["ground_z"] = float(location.z)
    state["ground_clearance_m"] = float(state["z"] - location.z)


def navigation_label(state: dict, aux: dict, config: dict) -> dict:
    field = config["field"]
    signed_distance = _ellipse_signed_distance(
        state["x"], state["y"], field["grass_length_m"] / 2.0, field["grass_width_m"] / 2.0
    )
    track_fraction = float(aux["track_fraction"])
    caution_distance = float(config.get("landing", {}).get("boundary_caution_m", 2.0))
    if track_fraction >= 0.80 or signed_distance < -0.75:
        class_id, class_name = 3, "track_surface"
    elif track_fraction >= 0.08:
        class_id, class_name = 2, "grass_track_boundary"
    elif signed_distance <= caution_distance:
        class_id, class_name = 1, "near_track"
    else:
        class_id, class_name = 0, "grass_interior"
    risk = max(0.0, min(1.0, (caution_distance - signed_distance) / max(caution_distance, 1e-6)))
    if class_id == 2:
        risk = max(risk, 0.75)
    elif class_id == 3:
        risk = 1.0
    return {
        "frame_index": int(state["frame_index"]),
        "navigation_class_id": class_id,
        "navigation_class": class_name,
        "signed_grass_boundary_distance_m": signed_distance,
        "grass_pixel_fraction": float(aux["grass_fraction"]),
        "track_pixel_fraction": track_fraction,
        "landing_unsafe_fraction": float(aux["landing_unsafe_fraction"]),
        "landing_caution_fraction": float(aux["landing_caution_fraction"]),
        "landing_safe_fraction": float(aux["landing_safe_fraction"]),
        "risk_score": risk,
        "landing_permitted": int(aux["landing_safe_fraction"] >= 0.70 and class_id == 0),
    }


def render_episode(
    episode_number: int, seed: int, config: dict, output_root: Path, overwrite: bool
) -> dict:
    episode_dir = output_root / f"episode_{episode_number:06d}"
    if episode_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{episode_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(episode_dir)
    for subdir in (
        "rgb",
        "depth",
        "segmentation",
        "segmentation_color",
        "landing_suitability",
        "landing_labels",
    ):
        (episode_dir / subdir).mkdir(parents=True, exist_ok=True)
    temp_root = episode_dir / ".render_tmp"

    spec = build_episode_spec(seed, config)
    scene_objects = build_episode_scene(spec, config)
    scene = configure_renderer(config)
    configure_render_outputs(scene, temp_root)
    states = compute_clean_states(spec["poses"], config)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    surface_bvhs = [BVHTree.FromObject(scene_objects[key], depsgraph) for key in ("grass", "track")]
    render_stats = []
    navigation_rows = []
    started = time.perf_counter()
    for state in states:
        index = int(state["frame_index"])
        scene.frame_set(index + 1)
        update_mesh_ground_clearance(surface_bvhs, state)
        body_q = set_camera_pose(
            scene_objects["camera"], state, float(config["render"]["camera_down_tilt_degrees"])
        )
        bpy.context.view_layer.update()
        state["ultrasonic_range_m"] = raycast_ultrasonic(scene, body_q, state)
        scene.render.filepath = str(episode_dir / "rgb" / f"{index:06d}.jpg")
        frame_started = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        aux = collect_auxiliary_outputs(
            scene,
            temp_root,
            episode_dir / "depth" / f"{index:06d}.npy",
            episode_dir / "segmentation" / f"{index:06d}.png",
            episode_dir / "landing_suitability" / f"{index:06d}.png",
            episode_dir / "segmentation_color" / f"{index:06d}.png",
            episode_dir / "landing_labels" / f"{index:06d}.png",
            state,
            config,
        )
        navigation_rows.append(navigation_label(state, aux, config))
        render_stats.append(
            {"frame_index": index, "render_seconds": time.perf_counter() - frame_started, **aux}
        )
    episode_seconds = time.perf_counter() - started
    shutil.rmtree(temp_root)

    clean_rows = [dict(row) for row in states]
    output_rows = noisy_copy(clean_rows, config, seed)
    write_csv(episode_dir / "sensors_clean.csv", clean_rows, SENSOR_COLUMNS)
    write_csv(episode_dir / "sensors.csv", output_rows, SENSOR_COLUMNS)
    write_csv(episode_dir / "poses.csv", clean_rows, POSE_COLUMNS)
    write_csv(episode_dir / "navigation_labels.csv", navigation_rows, NAVIGATION_COLUMNS)
    objects_payload = {
        "episode_seed": seed,
        "layout_style": spec["clutter_style"],
        "density_profile": spec["clutter_density_profile"],
        "objects": spec["objects"],
    }
    (episode_dir / "objects.json").write_text(
        json.dumps(objects_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    metadata = {
        "schema_version": "1.0.0",
        "project": "Imagination",
        "episode_number": episode_number,
        "episode_seed": seed,
        "base_seed": config["dataset"]["base_seed"],
        "coordinate_frame": "right-handed ENU world; x/east, y/north, z/up; metres, seconds, radians",
        "quaternion_order": "w,x,y,z",
        "image_resolution": [config["render"]["width"], config["render"]["height"]],
        "fps": config["timing"]["fps"],
        "frame_count": len(states),
        "frames_per_trajectory": config["dataset"]["frames_per_trajectory"],
        "trajectories_per_layout": config["dataset"]["trajectories_per_layout"],
        "trajectory_type": spec["trajectory_type"],
        "view_mode": spec["view_mode"],
        "clutter_style": spec["clutter_style"],
        "clutter_density_profile": spec["clutter_density_profile"],
        "review_focus": spec.get("review_focus"),
        "terrain": spec["terrain"],
        "lighting": spec["lighting"],
        "render": config["render"],
        "sensor_configuration": config["sensors"],
        "semantic_classes": {
            "0": "background",
            "1": "grass",
            "2": "track",
            "3": "cone",
            "4": "rock",
            "5": "football",
            "6": "bag",
            "7": "box",
            "8": "pole",
            "9": "chair",
            "10": "surrounding_ground",
        },
        "depth_format": "NumPy float32 .npy; Blender camera Z pass in metres; 0 indicates no finite hit",
        "segmentation_format": "8-bit PNG; pixel value equals semantic class ID",
        "segmentation_color_format": "visible RGB palette preview; training IDs remain in segmentation/",
        "landing_suitability_format": "8-bit PNG class IDs: 0 unsafe, 1 caution, 2 safe",
        "landing_labels_format": "visible RGB palette: red unsafe, amber caution, green safe",
        "navigation_classes": {
            "0": "grass_interior",
            "1": "near_track",
            "2": "grass_track_boundary",
            "3": "track_surface",
        },
        "episode_spec_sha256": canonical_hash(spec),
        "config": config,
        "generation": {
            "blender_version": bpy.app.version_string,
            "blenderproc_version": "2.8.0",
            "render_engine": scene.render.engine,
            "episode_seconds": episode_seconds,
            "mean_frame_render_seconds": sum(s["render_seconds"] for s in render_stats)
            / len(render_stats),
            "max_depth_invalid_pixels": max(s["depth_invalid_pixels"] for s in render_stats),
            "min_depth_positive_fraction": min(s["depth_positive_fraction"] for s in render_stats),
        },
    }
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def main():
    runtime_path = os.environ.get("IMAGINATION_RUNTIME_CONFIG")
    if not runtime_path:
        raise RuntimeError(
            "IMAGINATION_RUNTIME_CONFIG is not set; use generator/generate_dataset.py"
        )
    runtime = json.loads(Path(runtime_path).read_text(encoding="utf-8"))
    config = runtime["config"]
    output_root = Path(runtime["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = []
    expected_frames = int(config["dataset"]["trajectories_per_layout"]) * int(
        config["dataset"]["frames_per_trajectory"]
    )
    for zero_index in range(int(config["dataset"]["num_episodes"])):
        seed = episode_seed(config["dataset"]["base_seed"], zero_index)
        episode_config = copy.deepcopy(config)
        if episode_config.get("review", {}).get("enabled"):
            episode_config["review"]["index"] = zero_index
        episode_path = output_root / f"episode_{zero_index + 1:06d}"
        if runtime.get("resume") and episode_is_complete(episode_path, expected_frames):
            metadata.append(
                json.loads((episode_path / "metadata.json").read_text(encoding="utf-8"))
            )
            print(
                f"IMAGINATION_EPISODE_SKIPPED {zero_index + 1}/{config['dataset']['num_episodes']}"
            )
            continue
        metadata.append(
            render_episode(
                zero_index + 1,
                seed,
                episode_config,
                output_root,
                bool(runtime["overwrite"] or runtime.get("resume")),
            )
        )
        print(f"IMAGINATION_EPISODE_COMPLETE {zero_index + 1}/{config['dataset']['num_episodes']}")
    print(f"IMAGINATION_GENERATION_COMPLETE frames={sum(m['frame_count'] for m in metadata)}")


if __name__ == "__main__":
    main()
