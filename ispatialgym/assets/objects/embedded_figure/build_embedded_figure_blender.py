"""Build the iSpatialGym embedded figure (EFT item 11) - the context figure of 011a.png made only of its black lines, split
so that the target triangle of 011.png lifts out - with Blender.

Owner requests 2026-10-10: "One context & target shape for disembedding. 11a.png is context, 011 is target. [...] we can
grasp the part (the left upper one) that contains target shape and lift from the tray, the figure should be thick enough
to be grasped"; 18:07: "ONLY the black lines exist, at same thickness, and white space is just blank. And it should be
split into parts so that the target shape can be lifted (it should not contain lines like the target piece - those
lines should be separate parts)"; 18:46: the figure uses the same trays as every other disembedding pick; 2026-10-11 02:16: "it's
4 mm wide, so it can't be assisted grasp. I want it to be 12 mm wide, but the overall size should be the same".

Design (``figure_geometry.py`` holds every number and the part geometry):

* the 300 px drawing becomes a 20 cm square (line centres; the border bars' outer faces span 21.2 cm); every drawn line is
  a black bar 12 mm wide and 20 mm tall, the same for every line and every part, and the cells are empty;
* four parts: the target (the triangle's three sides only), the context frame (the border and every other bar outside
  the triangle, one rigid piece), and the two pieces of lines A and B inside the triangle (loose). Where a line meets or
  continues past the target it stops 0.3 mm short of the target's bars, so the target lifts straight up without touching
  anything; the empty cells beside its bars leave room for the fingers;
* the pieces are wood of the ARAT block's density (421 kg/m^3) painted black; the figure has no tray of its own: it lies
  on the standard disembedding source tray like every other disembedding pick.

Every part's collision is its convex pieces (one OBJ each: prisms of the clipped bar polygons); its visual is their
boolean union, and its mass, centre of mass and inertia come from that closed mesh. Exported like ``../kohs_block`` and
``../block_board`` (the ARAT component pipeline): a URDF in metres with the model centred in X/Y (bounding box) and its
bottom at Z = 0. Run (Blender 4.5, background)::

    <blender> --background --python build_embedded_figure_blender.py

Outputs ``model/<MODEL>/`` (``meshes/visual.obj`` + ``.mtl``, ``meshes/collision/*.obj``, ``<MODEL>.urdf``,
``build_manifest.json``) for the target ``isgefa``, the context frame ``isgefb`` and the inner pieces ``isgefc`` (line A)
and ``isgefd`` (line B), and ``model/assembly.json``: where each part's origin sits in the figure frame when the figure is
laid down (the square's centre on the surface it lies on, +x right, +y up the drawing, z up), the clearance check and a
grasp hint for the target.
"""

from __future__ import annotations

import json
import shutil
import sys
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path

import bmesh
import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_geometry as G  # noqa: E402

OUT = HERE / "model"
PIECE_CATEGORY = "jigsaw_puzzle_piece"
INK = (0.02, 0.02, 0.02, 1.0)
ROLES = {
    G.TARGET: ("target", "the target triangle of 011.png: its three sides only (G / C / E), no inner lines"),
    G.FRAME: ("context_frame", "the square's border and every bar outside the target triangle, one rigid piece"),
    G.INNER_A: ("context_inner_A", "the piece of line A inside the target triangle (loose)"),
    G.INNER_B: ("context_inner_B", "the piece of line B inside the target triangle (loose)"),
}


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(block):
            block.remove(item)


def color_material(name: str, rgba, roughness: float):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = roughness
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    mat.diffuse_color = rgba
    return mat


def link_object(name: str, bm):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def prism(name: str, poly, centre, height: float):
    """Convex prism over the CCW polygon ``poly`` (figure frame) from z = 0 to ``height``, shifted by ``-centre`` in x / y."""
    bm = bmesh.new()
    cx, cy = centre
    bottom = [bm.verts.new((x - cx, y - cy, 0.0)) for x, y in poly]
    top = [bm.verts.new((x - cx, y - cy, height)) for x, y in poly]
    n = len(poly)
    f_top, f_bottom = bm.faces.new(top), bm.faces.new(list(reversed(bottom)))
    for i in range(n):
        bm.faces.new([bottom[i], bottom[(i + 1) % n], top[(i + 1) % n], top[i]])
    for f, normal_z in ((f_top, 1.0), (f_bottom, -1.0)):
        f.normal_update()
        if f.normal.z * normal_z < 0.99:
            raise RuntimeError(f"{name}: cap winding is inverted")
    return link_object(name, bm)


def union(objs, name: str):
    """Boolean union (exact solver) of the objects into the first one."""
    base = objs[0]
    for other in objs[1:]:
        mod = base.modifiers.new(name="union", type="BOOLEAN")
        mod.operation = "UNION"
        mod.solver = "EXACT"
        mod.object = other
        bpy.context.view_layer.objects.active = base
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(other, do_unlink=True)
    bm = bmesh.new()
    bm.from_mesh(base.data)
    bmesh.ops.dissolve_limit(bm, angle_limit=0.0005, verts=bm.verts, edges=bm.edges)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(base.data)
    bm.free()
    base.name = name
    return base


def mesh_is_closed(obj) -> bool:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    ok = all(e.is_manifold for e in bm.edges)
    bm.free()
    return ok


def mesh_mass_properties(obj, density: float):
    """Mass, centre of mass and inertia about it of a closed, outward-oriented mesh (Eberly's polyhedral integrals)."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    co = [v.co for v in mesh.vertices]

    def sub(w0, w1, w2):
        t0 = w0 + w1
        f1 = t0 + w2
        t1 = w0 * w0
        t2 = t1 + w1 * t0
        f2 = t2 + w2 * f1
        f3 = w0 * t1 + w1 * t2 + w2 * f2
        return f1, f2, f3, f2 + w0 * (f1 + w0), f2 + w1 * (f1 + w1), f2 + w2 * (f1 + w2)

    s = [0.0] * 10
    for tri in mesh.loop_triangles:
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = (tuple(co[i]) for i in tri.vertices)
        a1, b1, c1, a2, b2, c2 = x1 - x0, y1 - y0, z1 - z0, x2 - x0, y2 - y0, z2 - z0
        d0, d1, d2 = b1 * c2 - b2 * c1, a2 * c1 - a1 * c2, a1 * b2 - a2 * b1
        f1x, f2x, f3x, g0x, g1x, g2x = sub(x0, x1, x2)
        f1y, f2y, f3y, g0y, g1y, g2y = sub(y0, y1, y2)
        f1z, f2z, f3z, g0z, g1z, g2z = sub(z0, z1, z2)
        s[0] += d0 * f1x
        s[1] += d0 * f2x
        s[2] += d1 * f2y
        s[3] += d2 * f2z
        s[4] += d0 * f3x
        s[5] += d1 * f3y
        s[6] += d2 * f3z
        s[7] += d0 * (y0 * g0x + y1 * g1x + y2 * g2x)
        s[8] += d1 * (z0 * g0y + z1 * g1y + z2 * g2y)
        s[9] += d2 * (x0 * g0z + x1 * g1z + x2 * g2z)
    s = [s[0] / 6.0] + [v / 24.0 for v in s[1:4]] + [v / 60.0 for v in s[4:7]] + [v / 120.0 for v in s[7:10]]
    vol = s[0]
    cx, cy, cz = s[1] / vol, s[2] / vol, s[3] / vol
    ixx = s[5] + s[6] - vol * (cy * cy + cz * cz)
    iyy = s[4] + s[6] - vol * (cz * cz + cx * cx)
    izz = s[4] + s[5] - vol * (cx * cx + cy * cy)
    ixy = -(s[7] - vol * cx * cy)
    iyz = -(s[8] - vol * cy * cz)
    ixz = -(s[9] - vol * cz * cx)
    inertia = [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
    return density * vol, vol, (cx, cy, cz), [[density * v for v in row] for row in inertia]


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


def write_urdf(path: Path, model: str, mass, com, inertia, collision_names):
    robot = ET.Element("robot", name=model)
    link = ET.SubElement(robot, "link", name="base_link")
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=" ".join(f"{v:.9g}" for v in com), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.9g}")
    ET.SubElement(inertial, "inertia", ixx=f"{inertia[0][0]:.9g}", ixy=f"{inertia[0][1]:.9g}", ixz=f"{inertia[0][2]:.9g}",
                  iyy=f"{inertia[1][1]:.9g}", iyz=f"{inertia[1][2]:.9g}", izz=f"{inertia[2][2]:.9g}")
    visual = ET.SubElement(link, "visual", name="visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename="meshes/visual.obj", scale="1 1 1")
    for n in collision_names:
        collision = ET.SubElement(link, "collision", name=n)
        ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(ET.SubElement(collision, "geometry"), "mesh", filename=f"meshes/collision/{n}.obj", scale="1 1 1")
    text = xml.dom.minidom.parseString(ET.tostring(robot, encoding="unicode")).toprettyxml(indent="  ")
    path.write_text(text, encoding="utf-8")


def r6(v):
    return round(float(v), 6)


def build_part(model: str, pieces, ink):
    out = OUT / model
    xs = [x for _, p in pieces for x, _ in p]
    ys = [y for _, p in pieces for _, y in p]
    centre = (0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)))
    names = []
    for name, poly in pieces:
        export_obj(prism(f"COL_{model}_{name}", poly, centre, G.BAR_H), out / "meshes" / "collision" / f"{name}.obj", materials=False)
        names.append(name)
    visual = union([prism(f"VIS_{model}_{name}", poly, centre, G.BAR_H) for name, poly in pieces], f"{model}_visual")
    if not mesh_is_closed(visual):
        raise RuntimeError(f"{model}: the visual union is not a closed mesh")
    visual.data.materials.append(ink)
    export_obj(visual, out / "meshes" / "visual.obj", materials=True)
    mass, vol, com, inertia = mesh_mass_properties(visual, G.DENSITY)
    write_urdf(out / f"{model}.urdf", model, mass, com, inertia, names)
    role, note = ROLES[model]
    manifest = {
        "category": PIECE_CATEGORY,
        "model": model,
        "kind": "embedded_figure_part",
        "figure": "EFT item 11 (context 011a.png, target 011.png), made of its lines only",
        "role": role,
        "role_note": note,
        "bar_width_m": G.BAR_W,
        "height_m": G.BAR_H,
        "mass_kg": r6(mass),
        "volume_m3": float(f"{vol:.6g}"),
        "density_kg_m3": G.DENSITY,
        "finish": "matte black paint",
        "pieces": {name: [[r6(x - centre[0]), r6(y - centre[1])] for x, y in poly] for name, poly in pieces},
        "origin_in_figure_m": [r6(centre[0]), r6(centre[1]), 0.0],
        "com_m": [r6(v) for v in com],
        "inertia_about_com_kg_m2": [[float(f"{v:.6g}") for v in row] for row in inertia],
        "bbox_m": [r6(max(xs) - min(xs)), r6(max(ys) - min(ys)), r6(G.BAR_H)],
        "frame": "part frame = figure frame shifted to the part's bounding-box centre; bottom at z = 0; +y up the drawing",
        "collision": f"{len(pieces)} convex prisms (the clipped bar pieces)",
        "urdf": str(out / f"{model}.urdf"),
    }
    if model == G.TARGET:
        manifest["triangle_angles_deg"] = [round(a, 3) for a in G.triangle_angles(G.target_triangle())]
        manifest["corners_cut_at_abs_xy_figure_m"] = r6(G.inner_limit())
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parts = G.parts()
    report = G.check(parts)
    clear_scene()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "METERS"
    scene.unit_settings.scale_length = 1.0
    if OUT.exists():
        shutil.rmtree(OUT)

    ink = color_material("bar_ink", INK, 0.85)
    built = {model: build_part(model, pieces, ink) for model, pieces in parts.items()}
    half = G.SQUARE_M / 2 + G.BAR_W / 2
    assembly = {
        "figure": "EFT item 11 (context 011a.png, target 011.png), made of its lines only",
        "square_m": G.SQUARE_M,
        "m_per_source_px": G.M_PER_PX,
        "bar_width_m": G.BAR_W,
        "height_m": G.BAR_H,
        "clearance_m": G.CLEARANCE,
        "footprint_m": [r6(2 * half), r6(2 * half)],
        "trays": "none of its own: lies on the standard disembedding source tray (and the target goes to the standard destination tray)",
        "parts": {m: {"category": PIECE_CATEGORY, "role": p["role"], "origin_in_figure_m": p["origin_in_figure_m"], "mass_kg": p["mass_kg"],
                      "bbox_m": p["bbox_m"]} for m, p in built.items()},
        "target": G.TARGET,
        "check": report,
        "grasp_hint": G.grasp_hint(parts),
        "figure_frame": "origin at the square's centre on the surface the figure lies on (the source tray's floor); x right, y up the drawing "
        "(away from the robot when the figure lies table-aligned in front of it), z up; every part's bottom at z = 0",
    }
    (OUT / "assembly.json").write_text(json.dumps(assembly, indent=2) + "\n", encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(HERE / "embedded_figure.blend"))
    print(f"EMBEDDED_FIGURE_BUILD_COMPLETE {OUT}")


if __name__ == "__main__":
    main()
