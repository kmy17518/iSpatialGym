"""Render preview PNGs of the embedded figure (``model/``) with Blender (Cycles on the CPU, no GPU needed).

The exported visual OBJs are re-imported and laid down at ``model/assembly.json``'s poses on a white sheet (a stand-in
for the source tray's floor, so the figure reads like the drawing), so the previews show the asset files themselves:

* ``previews/assembled_top.png`` - straight down, orthographic: the laid-down figure, to compare with 011a.png;
* ``previews/assembled_oblique.png`` - from the robot's side, as a head camera sees the figure on a table;
* ``previews/target_lifted.png`` - the target 6 cm above its place (as when grasped); the inner pieces stay put;
* ``previews/pieces_exploded.png`` - the four parts apart: the frame in place, the target and the two inner pieces in
  front of it;
* ``previews/target_top.png`` - the target alone from above, to compare with 011.png.

Run after the build::

    <blender> --background --python render_previews_blender.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
PREVIEWS = HERE / "previews"
SAMPLES = 96
SHEET = 0.26  # m: the white sheet the figure lies on
SHEET_T = 0.002


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.view_settings.view_transform = "AgX"
    return scene


def world_color(scene, rgb, strength):
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*rgb, 1.0)
    bg.inputs["Strength"].default_value = strength
    scene.world = world


def import_obj(path: Path, name: str):
    bpy.ops.wm.obj_import(filepath=str(path), forward_axis="Y", up_axis="Z")
    obj = bpy.context.selected_objects[0]
    obj.name = name
    return obj


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def area_light(name, location, target, power, size):
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = power
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(light)
    light.location = location
    look_at(light, target)
    return light


def perspective_camera(target, azimuth_deg, elevation_deg, distance, lens):
    """Looking at ``target`` from the robot's side (-y), turned ``azimuth_deg`` toward -x, raised ``elevation_deg``."""
    a, e = math.radians(azimuth_deg), math.radians(elevation_deg)
    data = bpy.data.cameras.new("cam")
    data.lens = lens
    cam = bpy.data.objects.new("cam", data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = Vector(target) + distance * Vector((-math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))
    look_at(cam, target)
    bpy.context.scene.camera = cam
    return cam


def top_camera(centre_xy, scale):
    data = bpy.data.cameras.new("top")
    data.type = "ORTHO"
    data.ortho_scale = scale
    cam = bpy.data.objects.new("top", data)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = (centre_xy[0], centre_xy[1], 1.0)
    cam.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.scene.camera = cam
    return cam


def table(rgb=(0.55, 0.42, 0.30)):
    bpy.ops.mesh.primitive_plane_add(size=3.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.object
    mat = bpy.data.materials.new("table")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.8
    plane.data.materials.append(mat)


def sheet():
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, SHEET_T / 2))
    obj = bpy.context.object
    obj.scale = (SHEET, SHEET, SHEET_T)
    mat = bpy.data.materials.new("sheet")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.90, 0.89, 0.86, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    obj.data.materials.append(mat)


def lay_down(assembly, offsets=None):
    """The white sheet at the origin, every part at its laid-down pose on it (+ an optional per-model offset)."""
    sheet()
    objs = {}
    for model, entry in assembly["parts"].items():
        obj = import_obj(MODEL_DIR / model / "meshes" / "visual.obj", model)
        x, y, z = entry["origin_in_figure_m"]
        dx, dy, dz = (offsets or {}).get(model, (0.0, 0.0, 0.0))
        obj.location = (x + dx, y + dy, z + SHEET_T + dz)
        objs[model] = obj
    return objs


def render(scene, name: str, res):
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(PREVIEWS / name)
    bpy.ops.render.render(write_still=True)


def studio_lights(centre, key=6.0, fill=1.5):
    area_light("key", (centre[0] - 0.35, centre[1] - 0.45, 0.75), centre, key, 0.5)
    area_light("fill", (centre[0] + 0.5, centre[1] - 0.2, 0.4), centre, fill, 0.6)


def main():
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    assembly = json.loads((MODEL_DIR / "assembly.json").read_text())
    target = assembly["target"]

    scene = reset()
    world_color(scene, (1.0, 1.0, 1.0), 0.9)
    table()
    lay_down(assembly)
    area_light("overhead", (0.0, 0.0, 0.8), (0.0, 0.0, 0.0), 8.0, 1.2)
    top_camera((0.0, 0.0), 0.275)
    render(scene, "assembled_top.png", (1000, 1000))

    scene = reset()
    world_color(scene, (0.92, 0.93, 0.95), 0.3)
    table()
    lay_down(assembly)
    studio_lights((0.0, 0.0, 0.02))
    perspective_camera((0.0, 0.0, 0.01), azimuth_deg=0.0, elevation_deg=50.0, distance=0.66, lens=40.0)
    render(scene, "assembled_oblique.png", (1200, 900))

    scene = reset()
    world_color(scene, (0.92, 0.93, 0.95), 0.3)
    table()
    lay_down(assembly, {target: (0.0, 0.0, 0.06)})
    studio_lights((0.0, 0.0, 0.03))
    perspective_camera((0.0, 0.0, 0.025), azimuth_deg=20.0, elevation_deg=40.0, distance=0.66, lens=40.0)
    render(scene, "target_lifted.png", (1200, 900))

    scene = reset()
    world_color(scene, (0.92, 0.93, 0.95), 0.3)
    table()
    objs = lay_down(assembly)
    inner = [m for m, e in assembly["parts"].items() if e["role"].startswith("context_inner")]
    for model, xy in zip([target] + inner, [(-0.04, -0.20), (0.075, -0.20), (0.115, -0.20)]):
        objs[model].location = (xy[0], xy[1], 0.0)
    studio_lights((0.0, -0.06, 0.03))
    perspective_camera((0.0, -0.05, 0.0), azimuth_deg=0.0, elevation_deg=55.0, distance=0.9, lens=40.0)
    render(scene, "pieces_exploded.png", (1200, 900))

    scene = reset()
    world_color(scene, (1.0, 1.0, 1.0), 0.9)
    table((0.92, 0.91, 0.88))
    obj = import_obj(MODEL_DIR / target / "meshes" / "visual.obj", target)
    obj.location = (0.0, 0.0, 0.0)
    area_light("overhead", (0.0, 0.0, 0.8), (0.0, 0.0, 0.0), 8.0, 1.2)
    top_camera((0.0, 0.0), 0.21)
    render(scene, "target_top.png", (800, 800))
    print(f"EMBEDDED_FIGURE_PREVIEWS_COMPLETE {PREVIEWS}")


if __name__ == "__main__":
    main()
