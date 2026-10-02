"""Deterministic, physically deformed sports-field terrain."""

from __future__ import annotations

import math
import random
from typing import Any


def make_terrain_parameters(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(seed ^ 0x54E22A11)
    tcfg = config["terrain"]
    broad_lo, broad_hi = tcfg["broad_amplitude_m"]
    fine_lo, fine_hi = tcfg["fine_amplitude_m"]
    count_lo, count_hi = tcfg["feature_count"]
    amp_lo, amp_hi = tcfg["feature_amplitude_m"]
    sigma_lo, sigma_hi = tcfg["feature_sigma_m"]
    broad = []
    for _ in range(3):
        broad.append(
            {
                "amplitude": rng.uniform(broad_lo, broad_hi) / 3.0,
                "kx": rng.uniform(0.025, 0.075),
                "ky": rng.uniform(0.025, 0.085),
                "phase": rng.uniform(-math.pi, math.pi),
            }
        )
    features = []
    for _ in range(rng.randint(int(count_lo), int(count_hi))):
        features.append(
            {
                "x": rng.uniform(-60.0, 60.0),
                "y": rng.uniform(-27.0, 27.0),
                "amplitude": rng.uniform(amp_lo, amp_hi),
                "sigma_x": rng.uniform(sigma_lo, sigma_hi),
                "sigma_y": rng.uniform(sigma_lo, sigma_hi),
                "angle": rng.uniform(-math.pi, math.pi),
            }
        )
    broken_zones = []
    for _ in range(rng.randint(3, 6)):
        broken_zones.append(
            {
                "x": rng.uniform(-62.0, 62.0),
                "y": rng.uniform(-26.0, 26.0),
                "radius_x": rng.uniform(0.7, 2.4),
                "radius_y": rng.uniform(0.45, 1.5),
                "angle": rng.uniform(-math.pi, math.pi),
                "amplitude": rng.uniform(-0.055, 0.025),
                "severity": rng.uniform(0.65, 1.0),
            }
        )
    return {
        "seed": seed,
        "broad_components": broad,
        "fine_amplitude": rng.uniform(fine_lo, fine_hi),
        "fine_phase_x": rng.uniform(-math.pi, math.pi),
        "fine_phase_y": rng.uniform(-math.pi, math.pi),
        "features": features,
        "broken_zones": broken_zones,
        "wear_seed": rng.randrange(1, 1_000_000),
    }


def terrain_height(x: float, y: float, params: dict[str, Any]) -> float:
    """Analytic height used both to build geometry and plan ground-relative flight."""
    z = 0.0
    for comp in params["broad_components"]:
        z += comp["amplitude"] * math.sin(comp["kx"] * x + comp["ky"] * y + comp["phase"])
    fa = params["fine_amplitude"]
    z += (
        fa
        * math.sin(0.37 * x + params["fine_phase_x"])
        * math.cos(0.43 * y + params["fine_phase_y"])
    )
    for feature in params["features"]:
        dx, dy = x - feature["x"], y - feature["y"]
        ca, sa = math.cos(feature["angle"]), math.sin(feature["angle"])
        u, v = ca * dx + sa * dy, -sa * dx + ca * dy
        exponent = -0.5 * ((u / feature["sigma_x"]) ** 2 + (v / feature["sigma_y"]) ** 2)
        z += feature["amplitude"] * math.exp(exponent)
    for zone in params.get("broken_zones", []):
        dx, dy = x - zone["x"], y - zone["y"]
        ca, sa = math.cos(zone["angle"]), math.sin(zone["angle"])
        u, v = ca * dx + sa * dy, -sa * dx + ca * dy
        exponent = -0.5 * ((u / zone["radius_x"]) ** 2 + (v / zone["radius_y"]) ** 2)
        z += zone["amplitude"] * math.exp(exponent)
    return z


def _ellipse_signed_distance(x: float, y: float, ax: float, ay: float) -> float:
    """Signed radial distance to an ellipse; positive is inside."""
    radius = math.hypot(x, y)
    if radius < 1e-9:
        return min(ax, ay)
    ux, uy = x / radius, y / radius
    boundary_radius = 1.0 / math.sqrt((ux / ax) ** 2 + (uy / ay) ** 2)
    return boundary_radius - radius


def landing_suitability(x: float, y: float, params: dict[str, Any], config: dict[str, Any]) -> int:
    """Return 0=unsafe, 1=caution, 2=safe for terrain-centred landing loss."""
    field = config["field"]
    boundary_distance = _ellipse_signed_distance(
        x, y, field["grass_length_m"] / 2.0, field["grass_width_m"] / 2.0
    )
    if boundary_distance < 0.0:
        return 0
    score = (
        1
        if boundary_distance < float(config.get("landing", {}).get("boundary_caution_m", 2.0))
        else 2
    )
    for zone in params.get("broken_zones", []):
        dx, dy = x - zone["x"], y - zone["y"]
        ca, sa = math.cos(zone["angle"]), math.sin(zone["angle"])
        u, v = ca * dx + sa * dy, -sa * dx + ca * dy
        normalized = math.sqrt((u / zone["radius_x"]) ** 2 + (v / zone["radius_y"]) ** 2)
        if normalized <= 0.72:
            return 0
        if normalized <= 1.18:
            score = min(score, 1)
    return score


def broken_ground_strength(x: float, y: float, params: dict[str, Any]) -> float:
    """Continuous 0..1 dirt/breakup weight for realistic soft-edged blending."""
    strength = 0.0
    for zone in params.get("broken_zones", []):
        dx, dy = x - zone["x"], y - zone["y"]
        ca, sa = math.cos(zone["angle"]), math.sin(zone["angle"])
        u, v = ca * dx + sa * dy, -sa * dx + ca * dy
        normalized_sq = (u / zone["radius_x"]) ** 2 + (v / zone["radius_y"]) ** 2
        strength = max(strength, float(zone["severity"]) * math.exp(-0.5 * normalized_sq))
    return min(1.0, strength)


def ground_surface_height(
    x: float, y: float, params: dict[str, Any], config: dict[str, Any]
) -> float:
    """Height of the visible walkable surface (grass, track, or surround)."""
    field = config["field"]
    grass_q = (x / (field["grass_length_m"] / 2.0)) ** 2 + (y / (field["grass_width_m"] / 2.0)) ** 2
    track_q = (x / (field["outer_track_length_m"] / 2.0)) ** 2 + (
        y / (field["outer_track_width_m"] / 2.0)
    ) ** 2
    if grass_q <= 1.0:
        return terrain_height(x, y, params)
    if track_q <= 1.0:
        return -0.025 + 0.12 * terrain_height(x, y, params)
    return -0.11


def create_grass_mesh(params: dict[str, Any], config: dict[str, Any]):
    import bpy

    field_cfg = config["field"]
    rings = int(field_cfg["terrain_rings"])
    segments = int(field_cfg["terrain_segments"])
    ax = field_cfg["grass_length_m"] / 2.0
    ay = field_cfg["grass_width_m"] / 2.0
    vertices = [(0.0, 0.0, terrain_height(0.0, 0.0, params))]
    for ring in range(1, rings + 1):
        radius = ring / rings
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            x, y = ax * radius * math.cos(theta), ay * radius * math.sin(theta)
            vertices.append((x, y, terrain_height(x, y, params)))
    faces = []
    for segment in range(segments):
        faces.append((0, 1 + segment, 1 + (segment + 1) % segments))
    for ring in range(1, rings):
        inner = 1 + (ring - 1) * segments
        outer = 1 + ring * segments
        for segment in range(segments):
            nxt = (segment + 1) % segments
            faces.append((inner + segment, outer + segment, outer + nxt, inner + nxt))
    mesh = bpy.data.meshes.new("GrassTerrainMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    landing = mesh.attributes.new(name="LandingScore", type="FLOAT", domain="POINT")
    brokenness = mesh.attributes.new(name="Brokenness", type="FLOAT", domain="POINT")
    for index, vertex in enumerate(mesh.vertices):
        landing.data[index].value = (
            landing_suitability(vertex.co.x, vertex.co.y, params, config) / 2.0
        )
        brokenness.data[index].value = broken_ground_strength(vertex.co.x, vertex.co.y, params)
    uv_layer = mesh.uv_layers.new(name="FieldUV")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index].co
            uv_layer.data[loop_index].uv = (vertex.x / 2.4, vertex.y / 2.4)
    obj = bpy.data.objects.new("GrassTerrain", mesh)
    bpy.context.collection.objects.link(obj)
    obj["semantic_class"] = "grass"
    obj.pass_index = 1
    return obj


def create_grass_blades(
    params: dict[str, Any], config: dict[str, Any], poses: list[dict[str, float]]
):
    """Create dense, static grass only in the swept camera corridor.

    Covering the complete 10,000 m2 field with individual blades would waste
    millions of polygons outside the nadir camera frustum.  The corridor is
    computed once from the complete trajectory, so the geometry never follows
    the camera or changes during an episode.
    """
    import bpy

    density = float(config["field"].get("grass_blades_per_m2", 850.0))
    rng = random.Random(int(params["seed"]) ^ 0xB1ADE551)
    ax = config["field"]["grass_length_m"] / 2.0
    ay = config["field"]["grass_width_m"] / 2.0
    maximum_clearance = max(float(pose["ground_clearance_m"]) for pose in poses)
    half_width = maximum_clearance * math.tan(
        math.radians(float(config["render"]["camera_fov_degrees"])) / 2.0
    )
    # Blender's camera.angle is horizontal for this 4:3 render.  Add a generous
    # margin for roll/pitch and anti-aliasing at the edge of the frame.
    margin_x = half_width + 0.65
    margin_y = (
        half_width * float(config["render"]["height"]) / float(config["render"]["width"]) + 0.65
    )
    min_x = max(-ax, min(float(p["x"]) for p in poses) - margin_x)
    max_x = min(ax, max(float(p["x"]) for p in poses) + margin_x)
    min_y = max(-ay, min(float(p["y"]) for p in poses) - margin_y)
    max_y = min(ay, max(float(p["y"]) for p in poses) + margin_y)
    count = max(1, round((max_x - min_x) * (max_y - min_y) * density))
    vertices = []
    faces = []
    for _ in range(count):
        x, y = rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)
        if (x / ax) ** 2 + (y / ay) ** 2 >= 0.998:
            continue
        if broken_ground_strength(x, y, params) > 0.42:
            continue
        z = terrain_height(x, y, params) + 0.002
        yaw = rng.random() * math.pi
        blade_half_width = rng.uniform(0.0011, 0.0024)
        height = rng.uniform(0.075, 0.135)
        # Long upright blades should read as a dense turf canopy from nadir,
        # not as isolated horizontal sticks.
        lean_x, lean_y = rng.uniform(-0.004, 0.004), rng.uniform(-0.004, 0.004)
        start = len(vertices)
        for angle in (yaw, yaw + math.pi / 2.0):
            dx, dy = math.cos(angle) * blade_half_width, math.sin(angle) * blade_half_width
            vertices.extend(
                ((x - dx, y - dy, z), (x + dx, y + dy, z), (x + lean_x, y + lean_y, z + height))
            )
        faces.extend(((start, start + 1, start + 2), (start + 3, start + 4, start + 5)))
    mesh = bpy.data.meshes.new("VisibleGrassBladeMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    landing = mesh.attributes.new(name="LandingScore", type="FLOAT", domain="POINT")
    for index, vertex in enumerate(mesh.vertices):
        landing.data[index].value = (
            landing_suitability(vertex.co.x, vertex.co.y, params, config) / 2.0
        )
    obj = bpy.data.objects.new("VisibleGrassBlades", mesh)
    bpy.context.collection.objects.link(obj)
    obj.pass_index = 1
    obj["semantic_class"] = "grass"
    if hasattr(obj, "visible_shadow"):
        obj.visible_shadow = False
    return obj


def create_track_mesh(params: dict[str, Any], config: dict[str, Any]):
    import bpy

    field_cfg = config["field"]
    segments = max(128, int(field_cfg["terrain_segments"]))
    inner_x = field_cfg["grass_length_m"] / 2.0
    inner_y = field_cfg["grass_width_m"] / 2.0
    outer_x = field_cfg["outer_track_length_m"] / 2.0
    outer_y = field_cfg["outer_track_width_m"] / 2.0
    vertices = []
    for ax, ay in ((inner_x, inner_y), (outer_x, outer_y)):
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            x, y = ax * math.cos(theta), ay * math.sin(theta)
            vertices.append((x, y, -0.025 + 0.12 * terrain_height(x, y, params)))
    faces = []
    for segment in range(segments):
        nxt = (segment + 1) % segments
        faces.append((segment, nxt, segments + nxt, segments + segment))
    mesh = bpy.data.meshes.new("RunningTrackMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("RunningTrack", mesh)
    bpy.context.collection.objects.link(obj)
    obj["semantic_class"] = "track"
    obj.pass_index = 2
    return obj
