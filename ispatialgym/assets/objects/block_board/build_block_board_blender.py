"""Build the iSpatialGym block board - a tight-fit "puzzle board" for the 2 x 2 Kohs block design - with Blender.

Owner request 2026-09-20 22:39 / 22:50 / 23:00 (option 2 follow-up): "a tight fit board like a puzzle board so that it
wouldn't need pushing to organize the blocks, not the spacious one we have right now"; chamfers accepted; "make the space
between the pockets small so they don't look too much apart". Four chamfered pockets seat the 25.4 mm cubes without any
pushing; thin dividers leave the seated blocks 4 mm apart. Follows the ARAT component pipeline
(``Behavior-1K-arat/asset_pipeline/arat_v1``) like ``../kohs_block``: a Blender source mesh is exported as OBJ (visual
union + one convex OBJ per collision piece) and wrapped in a URDF in metres with the model centred in X/Y and its bottom
at Z = 0. Run (Blender 4.5, background)::

    <blender> --background --python build_block_board_blender.py

Geometry (metres, board frame; +x = the robot's right when the board is table-aligned, +y = away from the robot):

* plate ``PLATE`` square, ``PLATE_T`` thick; the pocket floor is the plate top (z = PLATE_T);
* 2 x 2 pockets: seat ``SEAT`` square at the floor (1 mm clearance per side of the cube), vertical wall ``WALL_H`` tall,
  then chamfers;
* outer rim: chamfer ``OUTER_CHAMFER_W`` wide x ``OUTER_CHAMFER_H`` tall from the wall top (10 x 12 mm = 50 degrees), a
  flat top ``RIM_FLAT`` wide, the rim top ``WALL_H + OUTER_CHAMFER_H`` above the floor (the seated cube protrudes ~7 mm);
* dividers: ``DIVIDER_W`` wide at the floor, vertical ``DIVIDER_H`` tall, then chamfered from both sides to a ridge
  ``DIVIDER_W / 2`` higher (a "house" profile); pocket pitch ``SEAT + DIVIDER_W``.

Tuned by ``drop_test.py`` (2026-09-20, drop test v5; the owner's initial 45-degree chamfers and 8 mm apex let a cube edge
LEAN on the chamfer or climb the low divider in a quarter of the asymmetric landings): the 50-degree outer chamfer and the
10 mm vertical divider face shed a tilted arrival, and ``convert_to_omnigibson.py`` bakes a 0.25-friction physics material
(min combine) into the USD so an edge on a chamfer slides down instead of resting. Result: a cube released with its bottom
4-6 mm above the rim, aimed 3.5 mm toward the pocket's outer corner, seats within 1 mm / 2 degrees for landing errors of
+-6 mm per axis or diagonal and +-2 degrees of yaw (50 / 50 cases at 4 and 5 mm).

Every collision piece is a CONVEX prism (a pentagon cross-section extruded along the wall) - the chamfers are part of
the collision geometry (no convex decomposition); the visual is the boolean union of the pieces. ``build_manifest.json``
carries the seat geometry the sampler reads (``block_design._board_geometry``): ``pocket_pitch_m``, ``seat_inner_m``,
``outer_mouth_m``, ``wall_h_m``, ``chamfer_h_m``, ``outer_chamfer_w_m``, ``divider_w_m``, ``ridge_apex_z_m``,
``rim_top_z_m``, ``plate_thickness_m``, ``pocket_centres`` and ``outer_corner_sign`` (board frame, 4 pockets).
"""

from __future__ import annotations

import json
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path

import bmesh
import bpy

HERE = Path(__file__).resolve().parent
MODEL = "isgboard"
CATEGORY = "tray"
EDGE = 0.0254  # the Kohs block
import os as _os

_f = lambda k, d: float(_os.environ.get(k, d))  # noqa: E731  - drop-test tuning knobs (build_manifest.json records the values)
SEAT = 0.0274  # seat at the floor: 1 mm clearance per side
WALL_H = _f("BOARD_WALL_H", 0.006)  # outer wall: vertical part above the floor before the chamfer starts
OUTER_CHAMFER_W = _f("BOARD_OUTER_CHAMFER_W", 0.010)  # outer rim chamfer width (drop test v5)
OUTER_CHAMFER_H = _f("BOARD_OUTER_CHAMFER_H", 0.012)  # outer rim chamfer height: 0.012 over 0.010 = 50 deg (45 deg let a cube edge lean on it, drop test 2026-09-20)
RIM_FLAT = 0.006  # flat rim top outside the chamfer
DIVIDER_W = 0.004  # divider width at the floor
DIVIDER_H = _f("BOARD_DIVIDER_H", 0.010)  # divider: vertical part above the floor (a tilted arrival hits the face instead of climbing a low roof), then a 45 deg roof to the apex
PLATE_T = 0.008
PITCH = SEAT + DIVIDER_W  # 0.0314: seated blocks 4 mm apart
HALF_POCKETS = PITCH / 2 + SEAT / 2  # 0.0294: the pocket area's half extent (inner face of the outer walls)
PLATE = 2 * HALF_POCKETS + 2 * (OUTER_CHAMFER_W + RIM_FLAT)  # 0.0908 at the default chamfer width
RIM_TOP = PLATE_T + WALL_H + OUTER_CHAMFER_H  # above the bottom (0.024 default)
APEX = PLATE_T + DIVIDER_H + DIVIDER_W / 2  # ridge apex above the bottom (0.016 default: 0.008 above the floor)
MASS = 1.5  # kg: deliberately heavy (a hardwood board) so nothing slides it on the table
OUT = HERE / "model" / MODEL
COLOR = (0.62, 0.45, 0.28, 1.0)  # a stained wood


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(block):
            block.remove(item)


def make_material(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = COLOR
    bsdf.inputs["Roughness"].default_value = 0.6
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mat.diffuse_color = COLOR
    return mat


# ----------------------------------------------------------------------------------------------------------------------
# convex prisms
# ----------------------------------------------------------------------------------------------------------------------


def prism_verts(section, axis: str, a0: float, a1: float):
    """3-D vertices of a prism: ``section`` = convex polygon [(u, z), ...] (counter-clockwise) in the plane perpendicular to
    ``axis`` ("x" or "y"; ``u`` is the other horizontal coordinate), extruded from ``a0`` to ``a1`` along ``axis``."""
    pts = []
    for a in (a0, a1):
        ring = []
        for u, z in section:
            ring.append((a, u, z) if axis == "x" else (u, a, z))
        pts.append(ring)
    return pts


def add_prism(bm, section, axis: str, a0: float, a1: float):
    rings = prism_verts(section, axis, a0, a1)
    v0 = [bm.verts.new(p) for p in rings[0]]
    v1 = [bm.verts.new(p) for p in rings[1]]
    n = len(section)
    faces = [bm.faces.new(v0), bm.faces.new(list(reversed(v1)))]
    for i in range(n):
        j = (i + 1) % n
        faces.append(bm.faces.new([v0[i], v0[j], v1[j], v1[i]]))
    return faces


def box_section(x0, x1, z0, z1):
    return [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]


def outer_wall_section(sign: float):
    """Pentagon (u, z) of an outer wall on the ``sign`` side: floor-level foot from the pocket area's edge to the board's edge,
    the vertical wall ``WALL_H`` tall at the inner face, the 45-degree chamfer up to the rim top, the flat rim top."""
    ui = sign * HALF_POCKETS  # inner face (the seat wall)
    uc = sign * (HALF_POCKETS + OUTER_CHAMFER_W)  # top of the chamfer
    uo = sign * PLATE / 2  # outer face
    z0, zw, zt = PLATE_T, PLATE_T + WALL_H, RIM_TOP
    pts = [(ui, z0), (uo, z0), (uo, zt), (uc, zt), (ui, zw)]
    return pts if sign > 0 else list(reversed(pts))


def divider_section():
    """The "house" pentagon (u, z) of a divider centred on u = 0: ``DIVIDER_W`` wide, vertical to the wall top, both sides
    chamfered 45 degrees to the ridge apex."""
    h = DIVIDER_W / 2
    z0, zw = PLATE_T, PLATE_T + DIVIDER_H
    return [(-h, z0), (h, z0), (h, zw), (0.0, APEX), (-h, zw)]


def pieces():
    """(name, section, axis, a0, a1) of every convex collision piece."""
    half = PLATE / 2
    out = [("plate", box_section(-half, half, 0.0, PLATE_T), "y", -half, half)]
    out += [
        ("wall_right", outer_wall_section(+1.0), "y", -half, half),  # +x
        ("wall_left", outer_wall_section(-1.0), "y", -half, half),  # -x
        ("wall_far", outer_wall_section(+1.0), "x", -half, half),  # +y
        ("wall_near", outer_wall_section(-1.0), "x", -half, half),  # -y
    ]
    ext = HALF_POCKETS + 0.0006  # the dividers run a hair into the outer walls (no gap at the junction)
    out += [
        ("divider_y", divider_section(), "y", -ext, ext),  # along y at x = 0
        ("divider_x", divider_section(), "x", -ext, ext),  # along x at y = 0
    ]
    return out


def build_mesh(name: str, parts):
    bm = bmesh.new()
    for _, section, axis, a0, a1 in parts:
        add_prism(bm, section, axis, a0, a1)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def union_visual(parts):
    """Boolean union of the pieces (exact solver) for the visual mesh."""
    objs = [build_mesh(f"VIS_{name}", [p]) for name, *_ in parts for p in [(name, *_)]]
    base = objs[0]
    for other in objs[1:]:
        mod = base.modifiers.new(name=f"union_{other.name}", type="BOOLEAN")
        mod.operation = "UNION"
        mod.solver = "EXACT"
        mod.object = other
        bpy.context.view_layer.objects.active = base
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(other, do_unlink=True)
    base.name = "block_board"
    return base


def export_obj(obj, path: Path, materials: bool):
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
        export_uv=False,
        export_normals=True,
        export_materials=materials,
        export_pbr_extensions=materials,
        path_mode="COPY",
        export_triangulated_mesh=True,
    )


def write_urdf(path: Path, collision_names):
    robot = ET.Element("robot", name=MODEL)
    link = ET.SubElement(robot, "link", name="base_link")
    inertial = ET.SubElement(link, "inertial")
    zc = 0.5 * RIM_TOP
    ET.SubElement(inertial, "origin", xyz=f"0 0 {zc:.9g}", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{MASS:.9g}")
    a, b, c = PLATE, PLATE, RIM_TOP
    ixx = MASS * (b * b + c * c) / 12.0
    iyy = MASS * (a * a + c * c) / 12.0
    izz = MASS * (a * a + b * b) / 12.0
    ET.SubElement(inertial, "inertia", ixx=f"{ixx:.9g}", ixy="0", ixz="0", iyy=f"{iyy:.9g}", iyz="0", izz=f"{izz:.9g}")
    visual = ET.SubElement(link, "visual", name="visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename="meshes/visual.obj", scale="1 1 1")
    for n in collision_names:
        collision = ET.SubElement(link, "collision", name=n)
        ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename=f"meshes/collision/{n}.obj", scale="1 1 1")
    text = xml.dom.minidom.parseString(ET.tostring(robot, encoding="unicode")).toprettyxml(indent="  ")
    path.write_text(text, encoding="utf-8")


def main():
    clear_scene()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0

    parts = pieces()
    visual = union_visual(parts)
    visual.data.materials.append(make_material("board_wood"))
    export_obj(visual, OUT / "meshes" / "visual.obj", materials=True)
    names = []
    for part in parts:
        name = part[0]
        col = build_mesh(f"COL_{name}", [part])
        export_obj(col, OUT / "meshes" / "collision" / f"{name}.obj", materials=False)
        names.append(name)
    write_urdf(OUT / f"{MODEL}.urdf", names)
    h = PITCH / 2
    # pocket order = the design's slot order: 0 far-left (-x, +y), 1 far-right (+x, +y), 2 near-left (-x, -y), 3 near-right (+x, -y)
    centres = [(-h, h), (h, h), (-h, -h), (h, -h)]
    manifest = {
        "category": CATEGORY,
        "model": MODEL,
        "kind": "puzzle_board",
        "edge_m": EDGE,
        "plate_m": PLATE,
        "plate_thickness_m": PLATE_T,
        "pocket_floor_z_m": PLATE_T,
        "pocket_pitch_m": PITCH,
        "seat_inner_m": SEAT,
        "outer_mouth_m": SEAT + 2 * OUTER_CHAMFER_W,
        "wall_h_m": WALL_H,
        "chamfer_h_m": OUTER_CHAMFER_H,
        "outer_chamfer_w_m": OUTER_CHAMFER_W,
        "rim_flat_m": RIM_FLAT,
        "divider_w_m": DIVIDER_W,
        "divider_h_m": DIVIDER_H,  # vertical part of the divider above the floor
        "ridge_apex_z_m": DIVIDER_H + DIVIDER_W / 2,  # above the pocket floor
        "rim_top_z_m": RIM_TOP,  # above the board's bottom
        "rim_top_above_floor_m": WALL_H + OUTER_CHAMFER_H,
        "wall_height_m": WALL_H + OUTER_CHAMFER_H,  # legacy key: rim top above the floor
        "pocket_centres": [[round(x, 6), round(y, 6)] for x, y in centres],
        "outer_corner_sign": [[1 if x > 0 else -1, 1 if y > 0 else -1] for x, y in centres],
        "mass_kg": MASS,
        "bbox_m": [PLATE, PLATE, RIM_TOP],
        "frame": "+x = the robot's right, +y = away from the robot when table-aligned; pocket order 0 far-left, 1 far-right, 2 near-left, 3 near-right",
        "collision": "compound convex prisms (plate, four chamfered outer walls, two chamfered dividers)",
        "urdf": str(OUT / f"{MODEL}.urdf"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(HERE / "block_board.blend"))
    print(f"BLOCK_BOARD_BUILD_COMPLETE {OUT}")


if __name__ == "__main__":
    main()
