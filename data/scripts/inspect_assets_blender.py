"""Developer utility: print appendable object names and dimensions from downloaded blends."""

from pathlib import Path

import bpy

root = Path(__file__).resolve().parents[1] / "assets" / "third_party" / "polyhaven"
for blend_path in sorted(root.glob("*/*.blend")):
    with bpy.data.libraries.load(str(blend_path), link=False) as (source, target):
        target.objects = list(source.objects)
    objects = [obj for obj in target.objects if obj is not None]
    for obj in objects:
        if obj.name not in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.update()
    print(
        "ASSET_INSPECT",
        blend_path.parent.name,
        [(obj.name, obj.type, tuple(round(v, 4) for v in obj.dimensions)) for obj in objects],
    )
    for obj in objects:
        bpy.data.objects.remove(obj, do_unlink=True)
