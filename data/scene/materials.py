"""PBR materials for terrain and project-created visible assets."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYHAVEN = PROJECT_ROOT / "assets" / "third_party" / "polyhaven"


def ensure_semantic_aov(material, class_id: int):
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in list(nodes):
        if node.bl_idname == "ShaderNodeOutputAOV" and node.aov_name == "Semantic":
            nodes.remove(node)
    aov = nodes.new("ShaderNodeOutputAOV")
    aov.aov_name = "Semantic"
    aov.inputs["Value"].default_value = class_id / 255.0


def ensure_landing_aov(material, value: float = 0.0, socket=None):
    """Write normalized landing suitability: 0 unsafe, .5 caution, 1 safe."""
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    for node in list(nodes):
        if node.bl_idname == "ShaderNodeOutputAOV" and node.aov_name == "LandingSuitability":
            nodes.remove(node)
    aov = nodes.new("ShaderNodeOutputAOV")
    aov.aov_name = "LandingSuitability"
    if socket is None:
        aov.inputs["Value"].default_value = float(value)
    else:
        links.new(socket, aov.inputs["Value"])


def _principled(
    name: str, color: tuple[float, float, float, float], roughness: float, class_id: int
):
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    ensure_semantic_aov(mat, class_id)
    ensure_landing_aov(mat, 0.0)
    return mat


def _load_image(path: Path, non_color: bool = False):
    import bpy

    if not path.is_file():
        raise FileNotFoundError(
            f"Required PBR texture missing: {path}. Run scripts/download_assets.py"
        )
    image = bpy.data.images.load(str(path), check_existing=True)
    if non_color:
        image.colorspace_settings.name = "Non-Color"
    return image


def _pbr_set(nodes, links, asset_id: str, uv_output, x: float):
    base = POLYHAVEN / asset_id / "textures"
    textures = {}
    for suffix, non_color, y in (("diff", False, 260), ("rough", True, 20), ("nor_gl", True, -220)):
        node = nodes.new("ShaderNodeTexImage")
        node.label = f"{asset_id} {suffix}"
        node.image = _load_image(base / f"{asset_id}_{suffix}_1k.jpg", non_color)
        node.extension = "REPEAT"
        node.location = (x, y)
        links.new(uv_output, node.inputs["Vector"])
        textures[suffix] = node
    normal = nodes.new("ShaderNodeNormalMap")
    normal.space = "TANGENT"
    normal.inputs["Strength"].default_value = 0.72
    normal.location = (x + 210, -220)
    links.new(textures["nor_gl"].outputs["Color"], normal.inputs["Color"])
    textures["normal"] = normal
    return textures


def create_grass_material(wear_seed: int):
    import bpy

    mat = bpy.data.materials.new("PBR_Worn_Sports_Field")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (1050, 100)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (790, 100)
    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-1200, 160)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-1000, 160)
    mapping.inputs["Scale"].default_value = (0.72, 0.72, 1.0)
    mapping.inputs["Location"].default_value = (
        (wear_seed % 37) / 11.0,
        (wear_seed % 61) / 13.0,
        0.0,
    )
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
    healthy = _pbr_set(nodes, links, "grass_ground", mapping.outputs["Vector"], -780)
    dry = _pbr_set(nodes, links, "withered_grass", mapping.outputs["Vector"], -500)
    mud = _pbr_set(nodes, links, "brown_mud", mapping.outputs["Vector"], -220)
    green_grade = nodes.new("ShaderNodeMixRGB")
    green_grade.name = "HealthyGrassGreenGrade"
    green_grade.blend_type = "MULTIPLY"
    green_grade.inputs[0].default_value = 0.72
    green_grade.inputs[2].default_value = (0.46, 1.28, 0.39, 1.0)
    green_grade.location = (-500, 440)
    links.new(healthy["diff"].outputs["Color"], green_grade.inputs[1])

    large_noise = nodes.new("ShaderNodeTexNoise")
    large_noise.location = (-1000, -520)
    large_noise.inputs["Scale"].default_value = 0.055
    large_noise.inputs["Detail"].default_value = 3.2
    large_noise.inputs["Roughness"].default_value = 0.68
    large_noise.inputs["Distortion"].default_value = 0.18
    links.new(texcoord.outputs["Object"], large_noise.inputs["Vector"])
    dry_ramp = nodes.new("ShaderNodeValToRGB")
    dry_ramp.location = (-720, -500)
    dry_ramp.color_ramp.elements[0].position = 0.69
    dry_ramp.color_ramp.elements[1].position = 0.86
    links.new(large_noise.outputs["Fac"], dry_ramp.inputs["Fac"])

    patch_noise = nodes.new("ShaderNodeTexNoise")
    patch_noise.location = (-1000, -700)
    patch_noise.inputs["Scale"].default_value = 0.12
    patch_noise.inputs["Detail"].default_value = 2.0
    patch_noise.inputs["Roughness"].default_value = 0.62
    links.new(texcoord.outputs["Object"], patch_noise.inputs["Vector"])
    mud_ramp = nodes.new("ShaderNodeValToRGB")
    mud_ramp.location = (-720, -700)
    mud_ramp.color_ramp.elements[0].position = 0.80
    mud_ramp.color_ramp.elements[0].color = (0, 0, 0, 1)
    mud_ramp.color_ramp.elements[1].position = 0.93
    mud_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1)
    links.new(patch_noise.outputs["Fac"], mud_ramp.inputs["Fac"])

    def mix(name, factor, first, second, location):
        node = nodes.new("ShaderNodeMixRGB")
        node.name = name
        node.location = location
        links.new(factor, node.inputs[0])
        links.new(first, node.inputs[1])
        links.new(second, node.inputs[2])
        return node.outputs["Color"]

    color_gd = mix(
        "GrassDryColor",
        dry_ramp.outputs["Color"],
        green_grade.outputs["Color"],
        dry["diff"].outputs["Color"],
        (100, 310),
    )
    color = mix(
        "MudColor", mud_ramp.outputs["Color"], color_gd, mud["diff"].outputs["Color"], (360, 310)
    )
    links.new(color, bsdf.inputs["Base Color"])
    rough_gd = mix(
        "GrassDryRough",
        dry_ramp.outputs["Color"],
        healthy["rough"].outputs["Color"],
        dry["rough"].outputs["Color"],
        (100, 40),
    )
    rough = mix(
        "MudRough", mud_ramp.outputs["Color"], rough_gd, mud["rough"].outputs["Color"], (360, 40)
    )
    links.new(rough, bsdf.inputs["Roughness"])
    normal_gd = mix(
        "GrassDryNormal",
        dry_ramp.outputs["Color"],
        healthy["normal"].outputs["Normal"],
        dry["normal"].outputs["Normal"],
        (100, -210),
    )
    normal = mix(
        "MudNormal",
        mud_ramp.outputs["Color"],
        normal_gd,
        mud["normal"].outputs["Normal"],
        (360, -210),
    )
    links.new(normal, bsdf.inputs["Normal"])
    bsdf.inputs["Specular IOR Level"].default_value = 0.28
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    ensure_semantic_aov(mat, 1)
    landing_attr = nodes.new("ShaderNodeAttribute")
    landing_attr.attribute_name = "LandingScore"
    broken_attr = nodes.new("ShaderNodeAttribute")
    broken_attr.attribute_name = "Brokenness"
    broken_factor = nodes.new("ShaderNodeMath")
    broken_factor.operation = "POWER"
    broken_factor.inputs[1].default_value = 1.35
    links.new(broken_attr.outputs["Fac"], broken_factor.inputs[0])
    broken_color = mix(
        "BrokenGroundColor",
        broken_factor.outputs["Value"],
        color,
        mud["diff"].outputs["Color"],
        (560, 390),
    )
    broken_rough = mix(
        "BrokenGroundRough",
        broken_factor.outputs["Value"],
        rough,
        mud["rough"].outputs["Color"],
        (560, -10),
    )
    broken_normal = mix(
        "BrokenGroundNormal",
        broken_factor.outputs["Value"],
        normal,
        mud["normal"].outputs["Normal"],
        (560, -300),
    )
    links.new(broken_color, bsdf.inputs["Base Color"])
    links.new(broken_rough, bsdf.inputs["Roughness"])
    links.new(broken_normal, bsdf.inputs["Normal"])
    dry_penalty = nodes.new("ShaderNodeMath")
    dry_penalty.operation = "MULTIPLY_ADD"
    dry_penalty.inputs[1].default_value = -0.5
    dry_penalty.inputs[2].default_value = 1.0
    links.new(dry_ramp.outputs["Color"], dry_penalty.inputs[0])
    mud_penalty = nodes.new("ShaderNodeMath")
    mud_penalty.operation = "SUBTRACT"
    mud_penalty.inputs[0].default_value = 1.0
    links.new(mud_ramp.outputs["Color"], mud_penalty.inputs[1])
    landing_min = nodes.new("ShaderNodeMath")
    landing_min.operation = "MINIMUM"
    links.new(landing_attr.outputs["Fac"], landing_min.inputs[0])
    links.new(dry_penalty.outputs["Value"], landing_min.inputs[1])
    landing_final = nodes.new("ShaderNodeMath")
    landing_final.operation = "MINIMUM"
    links.new(landing_min.outputs["Value"], landing_final.inputs[0])
    links.new(mud_penalty.outputs["Value"], landing_final.inputs[1])
    ensure_landing_aov(mat, socket=landing_final.outputs["Value"])
    return mat


def create_cone_materials():
    import bpy

    image = _load_image(PROJECT_ROOT / "assets" / "generated" / "cone_weathered_orange_albedo.png")
    results = []
    for index, hue in enumerate((0.5, 0.51, 0.46)):
        mat = bpy.data.materials.new(f"WeatheredTrainingCone_{index}")
        mat.use_nodes = True
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        texcoord = nodes.new("ShaderNodeTexCoord")
        image_node = nodes.new("ShaderNodeTexImage")
        image_node.image = image
        image_node.extension = "REPEAT"
        hue_node = nodes.new("ShaderNodeHueSaturation")
        hue_node.inputs["Hue"].default_value = hue
        hue_node.inputs["Saturation"].default_value = 0.84
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 38.0
        noise.inputs["Detail"].default_value = 2.0
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.18
        bump.inputs["Distance"].default_value = 0.035
        links.new(texcoord.outputs["Generated"], image_node.inputs["Vector"])
        links.new(image_node.outputs["Color"], hue_node.inputs["Color"])
        links.new(hue_node.outputs["Color"], bsdf.inputs["Base Color"])
        links.new(texcoord.outputs["Object"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        bsdf.inputs["Roughness"].default_value = 0.62 + index * 0.05
        ensure_semantic_aov(mat, 3)
        ensure_landing_aov(mat, 0.0)
        results.append(mat)
    return results


def create_material_library(wear_seed: int) -> dict[str, object]:
    track = _principled("TrackRubber", (0.34, 0.022, 0.015, 1.0), 0.82, 2)
    nodes, links = track.node_tree.nodes, track.node_tree.links
    texcoord = nodes.new("ShaderNodeTexCoord")
    color_noise = nodes.new("ShaderNodeTexNoise")
    color_noise.inputs["Scale"].default_value = 2.8
    color_noise.inputs["Detail"].default_value = 5.0
    color_noise.inputs["Roughness"].default_value = 0.72
    color_ramp = nodes.new("ShaderNodeValToRGB")
    color_ramp.color_ramp.elements[0].position = 0.18
    color_ramp.color_ramp.elements[0].color = (0.16, 0.008, 0.005, 1.0)
    color_ramp.color_ramp.elements[1].position = 0.84
    color_ramp.color_ramp.elements[1].color = (0.42, 0.032, 0.020, 1.0)
    links.new(texcoord.outputs["Object"], color_noise.inputs["Vector"])
    links.new(color_noise.outputs["Fac"], color_ramp.inputs["Fac"])
    links.new(color_ramp.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 38.0
    noise.inputs["Detail"].default_value = 5.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.28
    bump.inputs["Distance"].default_value = 0.012
    links.new(texcoord.outputs["Object"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], nodes["Principled BSDF"].inputs["Normal"])
    grass_blades = _principled("GrassBlade", (0.035, 0.30, 0.012, 1.0), 0.91, 1)
    blade_attr = grass_blades.node_tree.nodes.new("ShaderNodeAttribute")
    blade_attr.attribute_name = "LandingScore"
    ensure_landing_aov(grass_blades, socket=blade_attr.outputs["Fac"])
    return {
        "grass": create_grass_material(wear_seed),
        "grass_blades": grass_blades,
        "track": track,
        "surround": _principled("SurroundingSoil", (0.065, 0.075, 0.028, 1.0), 0.98, 10),
        "cone_variants": create_cone_materials(),
        "pole_red": _principled("PoleRed", (0.55, 0.012, 0.008, 1.0), 0.52, 8),
        "pole_white": _principled("PoleWhite", (0.76, 0.73, 0.66, 1.0), 0.58, 8),
        "pole_base": _principled("PoleRubberBase", (0.018, 0.019, 0.017, 1.0), 0.86, 8),
    }
