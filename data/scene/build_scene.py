"""Assemble one fixed environment for an episode."""

from __future__ import annotations

import math

from scene.assets import AssetLibrary
from scene.materials import create_material_library
from scene.terrain import create_grass_blades, create_grass_mesh, create_track_mesh, terrain_height


def _kelvin_rgb(kelvin: float) -> tuple[float, float, float]:
    """Approximate daylight color in display RGB; sufficient for light tinting."""
    temperature = max(1000.0, min(40000.0, kelvin)) / 100.0
    if temperature <= 66:
        red = 255.0
        green = 99.4708025861 * math.log(temperature) - 161.1195681661
        blue = (
            0.0
            if temperature <= 19
            else 138.5177312231 * math.log(temperature - 10.0) - 305.0447927307
        )
    else:
        red = 329.698727446 * ((temperature - 60.0) ** -0.1332047592)
        green = 288.1221695283 * ((temperature - 60.0) ** -0.0755148492)
        blue = 255.0
    return tuple(max(0.0, min(255.0, value)) / 255.0 for value in (red, green, blue))


def clear_scene():
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def build_episode_scene(spec: dict, config: dict):
    import bpy

    clear_scene()
    materials = create_material_library(spec["terrain"]["wear_seed"])
    grass = create_grass_mesh(spec["terrain"], config)
    grass.data.materials.append(materials["grass"])
    blades = create_grass_blades(spec["terrain"], config, spec["poses"])
    blades.data.materials.append(materials["grass_blades"])
    track = create_track_mesh(spec["terrain"], config)
    track.data.materials.append(materials["track"])

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, -0.11))
    surround = bpy.context.object
    surround.name = "SurroundingGround"
    surround.scale = (115.0, 70.0, 1.0)
    surround.data.materials.append(materials["surround"])
    surround.pass_index = 10
    surround["semantic_class"] = "surround"

    assets = AssetLibrary(materials)
    created = []
    for record in spec["objects"]:
        ground_z = terrain_height(record["position"][0], record["position"][1], spec["terrain"])
        created.append(assets.instantiate(record, ground_z))

    camera_data = bpy.data.cameras.new("UAVCamera")
    camera = bpy.data.objects.new("UAVCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene = bpy.context.scene
    scene.camera = camera
    camera_data.lens_unit = "FOV"
    camera_data.angle = math.radians(float(config["render"]["camera_fov_degrees"]))
    camera_data.clip_start = float(config["render"]["clip_start_m"])
    camera_data.clip_end = float(config["render"]["clip_end_m"])

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (
        math.radians(spec["lighting"]["sun_elevation_deg"] - 90.0),
        0.0,
        math.radians(spec["lighting"]["sun_azimuth_deg"]),
    )
    sun_data.energy = spec["lighting"]["sun_energy"]
    sun_data.angle = math.radians(spec["lighting"]["sun_angle_deg"])
    sun_data.color = _kelvin_rgb(spec["lighting"]["color_temperature_k"])
    world = bpy.data.worlds.new("FieldWorld") if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    sky = nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(spec["lighting"]["sun_elevation_deg"])
    sky.sun_rotation = math.radians(spec["lighting"]["sun_azimuth_deg"])
    sky.altitude = 0.1
    sky.air_density = 1.0
    sky.aerosol_density = min(10.0, spec["lighting"]["turbidity"] * 0.42)
    sky.ozone_density = 1.0
    background.inputs["Strength"].default_value = spec["lighting"]["world_strength"]
    links.new(sky.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    scene.view_settings.exposure = spec["lighting"]["exposure_ev"]
    return {
        "camera": camera,
        "grass": grass,
        "grass_blades": blades,
        "track": track,
        "objects": created,
    }
