"""Single-pass depth and object-index compositor outputs."""

from __future__ import annotations

from pathlib import Path


def configure_render_outputs(scene, temp_root: Path):
    """Configure depth, semantic, and landing-suitability outputs."""
    import bpy

    scene.view_layers[0].use_pass_z = True
    semantic_aov = next((aov for aov in scene.view_layers[0].aovs if aov.name == "Semantic"), None)
    if semantic_aov is None:
        semantic_aov = scene.view_layers[0].aovs.add()
        semantic_aov.name = "Semantic"
        semantic_aov.type = "VALUE"
    landing_aov = next(
        (aov for aov in scene.view_layers[0].aovs if aov.name == "LandingSuitability"), None
    )
    if landing_aov is None:
        landing_aov = scene.view_layers[0].aovs.add()
        landing_aov.name = "LandingSuitability"
        landing_aov.type = "VALUE"
    scene.use_nodes = True
    # Blender 5 moved the compositor tree from Scene.node_tree to an explicit
    # Scene.compositing_node_group. Creating it also works on a factory scene.
    tree = scene.compositing_node_group
    if tree is None:
        tree = bpy.data.node_groups.new("ImaginationCompositor", "CompositorNodeTree")
        scene.compositing_node_group = tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")

    depth = tree.nodes.new("CompositorNodeOutputFile")
    depth.name = "DepthOutput"
    if hasattr(depth, "base_path"):
        depth.base_path = str(temp_root / "depth")
        depth.file_slots[0].path = "raw_"
        depth_input = depth.inputs[0]
    else:
        depth.directory = str(temp_root / "depth")
        depth.file_name = "raw_"
        depth.file_output_items.new("FLOAT", "Depth")
        depth_input = depth.inputs["Depth"]
        depth.format.media_type = "IMAGE"
    depth.format.file_format = "OPEN_EXR"
    depth.format.color_depth = "32"
    depth.format.color_mode = "RGB"
    depth.save_as_render = False
    tree.links.new(layers.outputs["Depth"], depth_input)

    segmentation = tree.nodes.new("CompositorNodeOutputFile")
    segmentation.name = "SegmentationOutput"
    if hasattr(segmentation, "base_path"):
        segmentation.base_path = str(temp_root / "segmentation")
        segmentation.file_slots[0].path = "raw_"
        segmentation_input = segmentation.inputs[0]
    else:
        segmentation.directory = str(temp_root / "segmentation")
        segmentation.file_name = "raw_"
        segmentation.file_output_items.new("FLOAT", "Mask")
        segmentation_input = segmentation.inputs["Mask"]
        segmentation.format.media_type = "IMAGE"
    segmentation.format.file_format = "PNG"
    segmentation.format.color_mode = "BW"
    segmentation.format.color_depth = "8"
    segmentation.save_as_render = False
    tree.links.new(layers.outputs["Semantic"], segmentation_input)
    landing = tree.nodes.new("CompositorNodeOutputFile")
    landing.name = "LandingOutput"
    if hasattr(landing, "base_path"):
        landing.base_path = str(temp_root / "landing")
        landing.file_slots[0].path = "raw_"
        landing_input = landing.inputs[0]
    else:
        landing.directory = str(temp_root / "landing")
        landing.file_name = "raw_"
        landing.file_output_items.new("FLOAT", "Landing")
        landing_input = landing.inputs["Landing"]
        landing.format.media_type = "IMAGE"
    landing.format.file_format = "PNG"
    landing.format.color_mode = "BW"
    landing.format.color_depth = "8"
    landing.save_as_render = False
    tree.links.new(layers.outputs["LandingSuitability"], landing_input)
    (temp_root / "depth").mkdir(parents=True, exist_ok=True)
    (temp_root / "segmentation").mkdir(parents=True, exist_ok=True)
    (temp_root / "landing").mkdir(parents=True, exist_ok=True)


def _load_scalar_png(path: Path):
    import bpy
    import numpy as np

    image = bpy.data.images.load(str(path), check_existing=False)
    image.colorspace_settings.name = "Non-Color"
    width, height = image.size
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    values = np.flipud(pixels[:, :, 0]).copy()
    bpy.data.images.remove(image)
    return values


def _write_grayscale_png(path: Path, values):
    """Write an 8-bit single-channel PNG using Blender's standard library."""
    import struct
    import zlib

    height, width = values.shape
    raw = b"".join(b"\x00" + values[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _write_rgb_png(path: Path, values):
    """Write an 8-bit RGB PNG without requiring Pillow inside Blender."""
    import struct
    import zlib

    height, width, channels = values.shape
    if channels != 3:
        raise ValueError("RGB PNG input must have three channels")
    raw = b"".join(b"\x00" + values[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _dilate(mask, radius: int):
    """Fast square dilation implemented with an integral image."""
    import numpy as np

    if radius <= 0 or not np.any(mask):
        return mask.copy()
    padded = np.pad(mask.astype(np.uint8), ((radius, radius), (radius, radius)))
    integral = np.pad(padded, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    sums = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return sums > 0


def collect_auxiliary_outputs(
    scene,
    temp_root: Path,
    depth_path: Path,
    segmentation_path: Path,
    landing_path: Path,
    segmentation_color_path: Path,
    landing_color_path: Path,
    state: dict,
    config: dict,
) -> dict:
    """Convert the compositor EXR to portable float32 NPY and name mask deterministically."""
    import bpy
    import numpy as np

    frame_token = f"{scene.frame_current:04d}"
    depth_candidates = sorted((temp_root / "depth").glob(f"raw_*{frame_token}*.exr"))
    segmentation_candidates = sorted((temp_root / "segmentation").glob(f"raw_*{frame_token}*.png"))
    landing_candidates = sorted((temp_root / "landing").glob(f"raw_*{frame_token}*.png"))
    # Blender 5's new File Output node names by output item rather than frame;
    # files are consumed immediately, so a constant temporary name is safe.
    if not depth_candidates:
        depth_candidates = sorted((temp_root / "depth").glob("raw_*.exr"))
    if not segmentation_candidates:
        segmentation_candidates = sorted((temp_root / "segmentation").glob("raw_*.png"))
    if not landing_candidates:
        landing_candidates = sorted((temp_root / "landing").glob("raw_*.png"))
    if not depth_candidates or not segmentation_candidates or not landing_candidates:
        raise RuntimeError(f"Compositor output missing for Blender frame {scene.frame_current}")
    exr_path = depth_candidates[-1]
    seg_path = segmentation_candidates[-1]
    raw_landing_path = landing_candidates[-1]
    image = bpy.data.images.load(str(exr_path), check_existing=False)
    width, height = image.size
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
    depth = np.flipud(pixels[:, :, 0]).copy()
    invalid = (
        ~np.isfinite(depth) | (depth < 0.0) | (depth > float(scene.camera.data.clip_end) * 1.01)
    )
    invalid_count = int(np.count_nonzero(invalid))
    depth[invalid] = 0.0
    np.save(depth_path, depth, allow_pickle=False)
    bpy.data.images.remove(image)
    exr_path.unlink()
    segmentation = np.rint(_load_scalar_png(seg_path) * 255.0).astype(np.uint8)
    grass_fraction = float(np.mean(segmentation == 1))
    track_fraction = float(np.mean(segmentation == 2))
    seg_path.replace(segmentation_path)
    landing_values = _load_scalar_png(raw_landing_path)
    landing_classes = np.where(
        landing_values < 0.25, 0, np.where(landing_values < 0.75, 1, 2)
    ).astype(np.uint8)
    object_mask = (segmentation >= 3) & (segmentation <= 9)
    metres_per_pixel = (
        2.0
        * float(state["ground_clearance_m"])
        * np.tan(np.radians(float(config["render"]["camera_fov_degrees"])) / 2.0)
        / float(config["render"]["width"])
    )
    unsafe_radius = max(
        1,
        round(
            float(config.get("landing", {}).get("obstacle_clearance_m", 0.35))
            / max(metres_per_pixel, 1e-6)
        ),
    )
    caution_radius = max(
        unsafe_radius + 1,
        round(
            (float(config.get("landing", {}).get("obstacle_clearance_m", 0.35)) + 0.18)
            / max(metres_per_pixel, 1e-6)
        ),
    )
    unsafe_near_objects = _dilate(object_mask, unsafe_radius)
    caution_near_objects = _dilate(object_mask, caution_radius) & ~unsafe_near_objects
    landing_classes[caution_near_objects & (landing_classes == 2)] = 1
    landing_classes[unsafe_near_objects] = 0
    _write_grayscale_png(landing_path, landing_classes)
    semantic_palette = np.asarray(
        (
            (0, 0, 0),
            (45, 155, 60),
            (190, 45, 35),
            (245, 125, 20),
            (125, 125, 125),
            (235, 235, 225),
            (45, 85, 190),
            (160, 105, 55),
            (210, 45, 180),
            (35, 185, 190),
            (110, 75, 40),
        ),
        dtype=np.uint8,
    )
    landing_palette = np.asarray(((220, 35, 35), (245, 175, 35), (35, 190, 75)), dtype=np.uint8)
    _write_rgb_png(segmentation_color_path, semantic_palette[np.clip(segmentation, 0, 10)])
    _write_rgb_png(landing_color_path, landing_palette[landing_classes])
    raw_landing_path.unlink()
    return {
        "depth_invalid_pixels": invalid_count,
        "depth_positive_fraction": float(np.mean(depth > 0.0)),
        "grass_fraction": grass_fraction,
        "track_fraction": track_fraction,
        "landing_unsafe_fraction": float(np.mean(landing_classes == 0)),
        "landing_caution_fraction": float(np.mean(landing_classes == 1)),
        "landing_safe_fraction": float(np.mean(landing_classes == 2)),
    }
