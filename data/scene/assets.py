"""Moderate-poly visible assets with PBR materials and shared meshes."""

from __future__ import annotations

from pathlib import Path

from scene.materials import ensure_landing_aov, ensure_semantic_aov

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets" / "third_party" / "polyhaven"


class AssetLibrary:
    EXTERNAL = {
        "football": ("dirty_football", "dirty_football_LOD1", 1.0),
        "rock_1": ("rock_07", "rock_07_LOD1", 1.0),
        "rock_2": ("rock_09", "rock_09_LOD1", 1.45),
        "rock_3": ("stone_01", "stone_01_LOD1", 1.35),
        "rock_4": ("namaqualand_stones_01", "namaqualand_stones_01_a_LOD1", 2.0),
        "rock_5": ("namaqualand_stones_01", "namaqualand_stones_01_b_LOD1", 1.7),
        "rock_6": ("namaqualand_stones_01", "namaqualand_stones_01_c_LOD1", 1.9),
        "rock_7": ("namaqualand_stones_01", "namaqualand_stones_01_e_LOD1", 1.8),
        "backpack": ("trashbag", "trashbag", 0.78),
        "box": ("cardboard_box_01", "cardboard_box_01", 0.9),
        "chair": ("plastic_monobloc_chair_01", "plastic_monobloc_chair_01", 0.9),
    }

    def __init__(self, materials: dict[str, object]):
        self.materials = materials
        self.meshes: dict[str, object] = {}
        self.offsets: dict[str, float] = {}
        self.base_scales: dict[str, float] = {}
        self._load_external_assets()
        self._build_training_cone()
        self._build_training_pole()

    def _register_mesh(self, key: str, mesh, class_id: int, base_scale: float = 1.0):
        self.meshes[key] = mesh
        self.offsets[key] = -min(vertex.co.z for vertex in mesh.vertices)
        self.base_scales[key] = base_scale
        for material in mesh.materials:
            if material:
                ensure_semantic_aov(material, class_id)
                ensure_landing_aov(material, 0.0)

    def _load_external_assets(self):
        import bpy

        class_ids = {"football": 5, "backpack": 6, "box": 7, "chair": 9}
        loaded_sources = {}
        for key, (asset_id, object_name, base_scale) in self.EXTERNAL.items():
            blend_path = ASSET_ROOT / asset_id / f"{asset_id}_1k.blend"
            if not blend_path.is_file():
                raise FileNotFoundError(
                    f"Missing CC0 asset {blend_path}; run scripts/download_assets.py"
                )
            source_key = (blend_path, object_name)
            if source_key not in loaded_sources:
                with bpy.data.libraries.load(str(blend_path), link=False) as (source, target):
                    if object_name not in source.objects:
                        raise RuntimeError(f"{object_name} not found in {blend_path}")
                    target.objects = [object_name]
                source_obj = target.objects[0]
                mesh = source_obj.data.copy()
                mesh.name = f"Visible_{key}_Mesh"
                mesh.transform(source_obj.matrix_world)
                bpy.data.objects.remove(source_obj, do_unlink=True)
                loaded_sources[source_key] = mesh
            else:
                mesh = loaded_sources[source_key].copy()
            class_id = 4 if key.startswith("rock_") else class_ids[key]
            self._register_mesh(key, mesh, class_id, base_scale)

    def _build_training_cone(self):
        import bpy

        created = []
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.025))
        base = bpy.context.object
        base.name = "ConeWeightedBase"
        base.dimensions = (0.43, 0.43, 0.05)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        bevel = base.modifiers.new("RoundedBase", "BEVEL")
        bevel.width = 0.025
        bevel.segments = 4
        bpy.context.view_layer.objects.active = base
        bpy.ops.object.modifier_apply(modifier=bevel.name)
        created.append(base)
        bpy.ops.mesh.primitive_cone_add(
            vertices=64,
            radius1=0.145,
            radius2=0.035,
            depth=0.38,
            end_fill_type="NOTHING",
            location=(0.0, 0.0, 0.235),
        )
        shell = bpy.context.object
        shell.name = "ConeFlexibleShell"
        solidify = shell.modifiers.new("ConeWall", "SOLIDIFY")
        solidify.thickness = 0.006
        bevel = shell.modifiers.new("SoftEdges", "BEVEL")
        bevel.width = 0.004
        bevel.segments = 3
        bpy.context.view_layer.objects.active = shell
        bpy.ops.object.modifier_apply(modifier=solidify.name)
        bpy.ops.object.modifier_apply(modifier=bevel.name)
        created.append(shell)
        bpy.ops.mesh.primitive_torus_add(
            major_radius=0.035,
            minor_radius=0.0065,
            major_segments=48,
            minor_segments=8,
            location=(0.0, 0.0, 0.426),
        )
        created.append(bpy.context.object)
        for obj in created:
            obj.select_set(True)
            obj.data.materials.append(self.materials["cone_variants"][0])
        bpy.context.view_layer.objects.active = base
        bpy.ops.object.join()
        mesh = base.data
        mesh.name = "VisibleTrainingConeMesh"
        self._register_mesh("cone", mesh, 3)
        bpy.data.objects.remove(base, do_unlink=True)

    def _build_training_pole(self):
        import bpy

        parts = []
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48, radius=0.14, depth=0.035, location=(0, 0, 0.0175)
        )
        base = bpy.context.object
        base.data.materials.append(self.materials["pole_base"])
        parts.append(base)
        for segment in range(6):
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=32, radius=0.024, depth=0.16, location=(0, 0, 0.135 + segment * 0.16)
            )
            part = bpy.context.object
            part.data.materials.append(
                self.materials["pole_red" if segment % 2 == 0 else "pole_white"]
            )
            parts.append(part)
        for obj in parts:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = base
        bpy.ops.object.join()
        mesh = base.data
        mesh.name = "VisibleTrainingPoleMesh"
        self._register_mesh("pole", mesh, 8)
        bpy.data.objects.remove(base, do_unlink=True)

    def instantiate(self, record: dict, ground_z: float):
        import bpy

        asset_type = record["asset_type"]
        obj = bpy.data.objects.new(record["object_id"], self.meshes[asset_type])
        bpy.context.collection.objects.link(obj)
        scale = float(record["scale"]) * self.base_scales[asset_type]
        obj.scale = (scale, scale, scale)
        obj.rotation_euler[2] = float(record["yaw_rad"])
        obj.location = (
            record["position"][0],
            record["position"][1],
            ground_z + self.offsets[asset_type] * scale - 0.006,
        )
        obj.pass_index = int(record["class_id"])
        obj["semantic_class"] = "rock" if asset_type.startswith("rock_") else asset_type
        obj["object_id"] = record["object_id"]
        if asset_type == "cone":
            for slot in obj.material_slots:
                slot.link = "OBJECT"
                slot.material = self.materials["cone_variants"][record["material_variant"]]
        return obj
