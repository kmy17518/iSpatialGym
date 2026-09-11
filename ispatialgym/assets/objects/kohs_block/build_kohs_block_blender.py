"""Build the iSpatialGym Kohs block (one-inch painted wooden cube) with Blender.

Follows the ARAT component pipeline (``Behavior-1K-arat/asset_pipeline/arat_v1``): a Blender source
mesh is exported as OBJ (visual + one convex collision box) and wrapped in a URDF in metres with the
model centred in X/Y and its bottom at Z = 0. Run (Blender 4.5, background)::

    python make_texture.py
    <blender> --background --python build_kohs_block_blender.py

Outputs ``model/<MODEL>/`` next to this file: ``meshes/visual.obj`` (+ ``.mtl`` + atlas png),
``meshes/collision/box.obj``, ``<MODEL>.urdf`` and ``build_manifest.json``.

Faces (block frame): +z red, -z blue, +x white, -x yellow, +y blue|yellow diagonal, -y red|white
diagonal. On the side faces the tile's "up" is the block's +z; on the top / bottom faces it is +y.
Mass: the ARAT wooden blocks (ARAT_specification.md) have 6.5 g at 2.5 cm; scaled by volume to
25.4 mm this gives 6.8 g (0.42 g/cm^3), rounded to 6.9 g.
"""

from __future__ import annotations

import json
import math
import sys
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path

import bmesh
import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_texture import tile_uv_rect  # noqa: E402

MODEL = "isgblk"
CATEGORY = "toy_dice"
EDGE = 0.0254  # one inch
MASS = 0.0069  # kg (see module docstring)
ATLAS = HERE / "kohs_block_atlas.png"
OUT = HERE / "model" / MODEL

# face -> (normal, tile id, corner order). Corners are listed counter-clockwise seen from outside,
# starting at the tile's bottom-left (u0, v0), then bottom-right (u1, v0), top-right (u1, v1), top-left (u0, v1).
h = EDGE / 2.0
FACES = {
    "+z": ((0, 0, 1), 0, [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]),  # tile up = +y
    "-z": ((0, 0, -1), 1, [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)]),  # tile up = +y (viewed from below)
    "+x": ((1, 0, 0), 2, [(h, -h, -h), (h, h, -h), (h, h, h), (h, -h, h)]),  # tile up = +z
    "-x": ((-1, 0, 0), 3, [(-h, h, -h), (-h, -h, -h), (-h, -h, h), (-h, h, h)]),  # tile up = +z
    "+y": ((0, 1, 0), 4, [(h, h, -h), (-h, h, -h), (-h, h, h), (h, h, h)]),  # tile up = +z
    "-y": ((0, -1, 0), 5, [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)]),  # tile up = +z
}
FACE_COLORS = {"+z": "red", "-z": "blue", "+x": "white", "-x": "yellow", "+y": "blue|yellow", "-y": "red|white"}


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(block):
            block.remove(item)


def make_material(name: str, image_path: Path):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(str(image_path), check_existing=True)
    tex.interpolation = "Closest"
    bsdf.inputs["Roughness"].default_value = 0.45
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def build_cube(name: str, textured: bool, z_offset: float):
    bm = bmesh.new()
    uv_layer = bm.loops.layers.uv.new("UVMap") if textured else None
    vcache = {}

    def vert(p):
        key = tuple(round(v, 9) for v in p)
        if key not in vcache:
            vcache[key] = bm.verts.new((p[0], p[1], p[2] + z_offset))
        return vcache[key]

    for face_id, (normal, tile, corners) in FACES.items():
        f = bm.faces.new([vert(c) for c in corners])
        # verify winding: face normal must point along `normal`
        f.normal_update()
        if sum(a * b for a, b in zip(f.normal, normal)) < 0:
            raise RuntimeError(f"face {face_id} winding is inverted")
        if textured:
            u0, v0, u1, v1 = tile_uv_rect(tile)
            for loop, uv in zip(f.loops, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]):
                loop[uv_layer].uv = uv
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def export_obj(obj, path: Path, materials: bool, uv: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=str(path),
        check_existing=False,
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        apply_modifiers=True,
        export_uv=uv,
        export_normals=True,
        export_materials=materials,
        export_pbr_extensions=materials,
        path_mode="COPY",
        export_triangulated_mesh=True,
    )


def write_urdf(path: Path):
    robot = ET.Element("robot", name=MODEL)
    link = ET.SubElement(robot, "link", name="base_link")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=f"0 0 {EDGE / 2:.9g}", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{MASS:.9g}")
    i = MASS * EDGE * EDGE / 6.0
    ET.SubElement(inertial, "inertia", ixx=f"{i:.9g}", ixy="0", ixz="0", iyy=f"{i:.9g}", iyz="0", izz=f"{i:.9g}")
    visual = ET.SubElement(link, "visual", name="visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename="meshes/visual.obj", scale="1 1 1")
    collision = ET.SubElement(link, "collision", name="box")
    ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename="meshes/collision/box.obj", scale="1 1 1")
    text = xml.dom.minidom.parseString(ET.tostring(robot, encoding="unicode")).toprettyxml(indent="  ")
    path.write_text(text, encoding="utf-8")


def main():
    if not ATLAS.is_file():
        raise FileNotFoundError(f"run make_texture.py first ({ATLAS})")
    clear_scene()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0

    visual = build_cube("kohs_block", textured=True, z_offset=EDGE / 2.0)
    visual.data.materials.append(make_material("kohs_paint", ATLAS))
    collision = build_cube("COL_box", textured=False, z_offset=EDGE / 2.0)

    export_obj(visual, OUT / "meshes" / "visual.obj", materials=True, uv=True)
    export_obj(collision, OUT / "meshes" / "collision" / "box.obj", materials=False, uv=False)
    write_urdf(OUT / f"{MODEL}.urdf")
    manifest = {
        "category": CATEGORY,
        "model": MODEL,
        "edge_m": EDGE,
        "mass_kg": MASS,
        "mass_note": "ARAT 2.5 cm wooden block (6.5 g) scaled by volume to 25.4 mm: 6.8 g, rounded to 6.9 g",
        "faces": FACE_COLORS,
        "opposite_pairs": [["+z", "-z"], ["+x", "-x"], ["+y", "-y"]],
        "tile_up_axis": {"+z": "+y", "-z": "+y", "+x": "+z", "-x": "+z", "+y": "+z", "-y": "+z"},
        "bbox_m": [EDGE, EDGE, EDGE],
        "urdf": str(OUT / f"{MODEL}.urdf"),
    }
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(HERE / "kohs_block.blend"))
    print(f"KOHS_BLOCK_BUILD_COMPLETE {OUT}")


if __name__ == "__main__":
    main()
