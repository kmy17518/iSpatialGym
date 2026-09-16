"""Render the init (and goal) state of generated instances for the review rounds.

**v2 layout** (``sample_isg`` / ``multiply_isg``, spec §1; detected per template by
:func:`ispatialgym.sampling.validate_instances.detect_layout`): every instance is loaded through the evaluator's loader
(:class:`ispatialgym.oracle.run_isg.InstanceLoader`, the same code path as ``run_isg`` and ``validate_instances``) and
rendered from the head camera with the ``isg-v1`` evaluation profile (1080 x 1080, 40 mm horizontal aperture):

* ``<template>/renders/<iid>-init.png`` -- the init view,
* ``<template>/renders/<iid>-goal.png`` -- with ``--goal``: the goal configuration materialised by teleport the way the
  sampler's goal check did (``validate_instances.materialise_goal``; skipped for targets a teleport cannot realise),
* ``<template>/renders/<iid>-topdown.png`` -- the traversability map around the task annotated from the sidecar: the
  recorded stances as base-pose arrows (init / manipulation / placement / goal), the recorded paths, the objects and the
  robot's actual base pose (``--no-topdown`` to skip),
* ``<template>/renders/montage.png`` -- one row per instance: init | goal | top-down.

**v1 layout** (``sample_instances``): the BEHAVIOR replay sheet (left wrist | head | right wrist over an external
shoulder camera) per instance next to the TRO and one contact sheet per problem under ``<out>/renders/``, unchanged.

Examples::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.render_instances --root /tmp/isg_v2 [--goal] [--alias ...] [--template ...]
    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.render_instances --out /tmp/isg_full11      # v1 layout
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from ispatialgym.sampling import validate_instances as V  # noqa: E402  (layout helpers, loader wiring, goal materialisation; simulator-free import)

log = logging.getLogger("ispatialgym.render")

CAM_RES = 640  # side of each robot camera tile in the v1 sheet (``--res``); 384 with 4 iterations looked blurred
RENDER_ITERS = 10  # frames rendered after a teleport before the sensors are read (``--render-iters``)
RENDERS_DIR = "renders"
MONTAGE_TILE = 360  # side of a montage tile (v2)
TOPDOWN_WINDOW_M = 3.0  # half-extent of the top-down crop around the stances
TOPDOWN_SCALE = 6  # pixels per traversability cell in the top-down overlay
ARROW_M = 0.35  # base-pose arrow length (m)
#: stance name -> arrow colour of the top-down overlay
STANCE_COLOURS: Dict[str, Tuple[int, int, int]] = {"init": (255, 220, 0), "manipulation": (255, 120, 0), "placement": (0, 200, 255), "goal": (255, 0, 200)}
ROBOT_COLOUR = (0, 220, 80)
OBJECT_COLOUR = (230, 40, 40)
PATH_COLOUR = (80, 120, 255)


def _to_uint8(img) -> np.ndarray:
    arr = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr[..., :3]


def _tile(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.BILINEAR))


def compose(left, head, right, external, res: int = CAM_RES) -> np.ndarray:
    """v1 BEHAVIOR replay layout: top row = left wrist | head | right wrist, bottom = the external shoulder camera."""
    top = np.concatenate([_tile(left, res, res), _tile(head, res, res), _tile(right, res, res)], axis=1)
    bottom = _tile(external, 3 * res, 2 * res)
    return np.concatenate([top, bottom], axis=0)


# ==========================================================================================
# v2: top-down stance overlay and montage (pure)
# ==========================================================================================


def topdown_overlay(free_map: np.ndarray, to_map: Callable[[float, float], Tuple[int, int]], stances: Mapping[str, Sequence[float]],
                    paths: Sequence[Sequence[Sequence[float]]] = (), objects: Optional[Mapping[str, Sequence[float]]] = None,
                    robot_pose2d: Optional[Sequence[float]] = None, resolution_m: float = 0.1, window_m: float = TOPDOWN_WINDOW_M,
                    scale: int = TOPDOWN_SCALE, title: str = "") -> np.ndarray:
    """Top-down annotation of the sidecar facts on the traversability map.

    ``free_map`` is the (H, W) map (255 = free floor); ``to_map(x, y) -> (row, col)`` the scene's world-to-map function
    (any axis convention: the canvas is flipped so that world +y points up); ``stances`` ``{name: [x, y, yaw]}`` are
    drawn as arrows (:data:`STANCE_COLOURS`), ``paths`` as polylines of ``[x, y, yaw]`` waypoints, ``objects``
    ``{name: [x, y]}`` as dots, ``robot_pose2d`` as the green arrow. The crop is at least ``window_m`` around the centre
    of the stances / waypoints / robot and grows to cover them all (+ 1 m), upscaled ``scale`` x."""
    fm = np.asarray(free_map)
    if fm.ndim == 3:
        fm = fm[..., 0]
    anchors = [tuple(float(v) for v in p[:2]) for p in stances.values()]
    if robot_pose2d is not None:
        anchors.append((float(robot_pose2d[0]), float(robot_pose2d[1])))
    for wps in paths:
        anchors += [(float(w[0]), float(w[1])) for w in wps]
    if not anchors and objects:
        anchors = [tuple(float(v) for v in p[:2]) for p in objects.values()]
    if not anchors:
        anchors = [(0.0, 0.0)]
    xs, ys = [a[0] for a in anchors], [a[1] for a in anchors]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    # the crop covers every stance / waypoint (a recorded path may be several metres long) with a 1 m margin
    window_m = max(float(window_m), 0.5 * max(max(xs) - min(xs), max(ys) - min(ys)) + 1.0)
    r0, c0 = to_map(cx, cy)
    half = max(1, int(round(window_m / max(resolution_m, 1e-6))))
    h, w = fm.shape
    rows = (max(0, r0 - half), min(h, r0 + half + 1))
    cols = (max(0, c0 - half), min(w, c0 + half + 1))
    crop = fm[rows[0]:rows[1], cols[0]:cols[1]]
    if crop.size == 0:
        crop = np.zeros((2 * half + 1, 2 * half + 1), dtype=np.uint8)
        rows, cols = (r0 - half, r0 + half + 1), (c0 - half, c0 + half + 1)
    # world +y must point up on the canvas: flip when the map row grows with y
    flip = to_map(cx, cy + 1.0)[0] > r0
    base = np.where(crop == 255, 235, 70).astype(np.uint8)
    if flip:
        base = base[::-1]
    canvas = Image.fromarray(np.stack([base] * 3, axis=-1)).resize((base.shape[1] * scale, base.shape[0] * scale), Image.NEAREST)
    draw = ImageDraw.Draw(canvas)
    n_rows = rows[1] - rows[0]

    def px(x: float, y: float) -> Tuple[float, float]:
        r, c = to_map(float(x), float(y))
        rr = r - rows[0]
        if flip:
            rr = n_rows - 1 - rr
        return (c - cols[0] + 0.5) * scale, (rr + 0.5) * scale

    def arrow(pose2d, colour, label):
        x, y, yaw = float(pose2d[0]), float(pose2d[1]), float(pose2d[2])
        tail = px(x, y)
        head = px(x + ARROW_M * math.cos(yaw), y + ARROW_M * math.sin(yaw))
        draw.line([tail, head], fill=colour, width=max(2, scale // 2))
        dx, dy = head[0] - tail[0], head[1] - tail[1]
        n = math.hypot(dx, dy) or 1.0
        ux, uy = dx / n, dy / n
        wing = 0.35 * n
        draw.polygon([head, (head[0] - wing * ux + 0.5 * wing * uy, head[1] - wing * uy - 0.5 * wing * ux),
                      (head[0] - wing * ux - 0.5 * wing * uy, head[1] - wing * uy + 0.5 * wing * ux)], fill=colour)
        rad = max(3, scale)
        draw.ellipse([tail[0] - rad, tail[1] - rad, tail[0] + rad, tail[1] + rad], outline=colour, width=2)
        draw.text((tail[0] + rad + 2, tail[1] - rad - 10), label, fill=colour)

    for wps in paths:
        pts = [px(w[0], w[1]) for w in wps]
        if len(pts) >= 2:
            draw.line(pts, fill=PATH_COLOUR, width=max(1, scale // 3))
        for p in pts:
            draw.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=PATH_COLOUR)
    for name, xy in (objects or {}).items():
        p = px(xy[0], xy[1])
        rad = max(3, scale // 2 + 1)
        draw.ellipse([p[0] - rad, p[1] - rad, p[0] + rad, p[1] + rad], fill=OBJECT_COLOUR)
        draw.text((p[0] + rad + 1, p[1] - 6), name.split(".n.")[0][:14], fill=OBJECT_COLOUR)
    if robot_pose2d is not None:  # drawn first: a stance that coincides with the robot's pose keeps its label on top
        arrow(robot_pose2d, ROBOT_COLOUR, "robot")
    for name, pose in stances.items():
        arrow(pose, STANCE_COLOURS.get(name, (200, 200, 200)), name)
    if title:
        draw.rectangle([0, 0, canvas.width, 14], fill=(0, 0, 0))
        draw.text((3, 1), title[:120], fill=(255, 255, 255))
    return np.asarray(canvas)


def montage(rows: Sequence[Sequence[Optional[np.ndarray]]], labels: Sequence[str] = (), tile: int = MONTAGE_TILE,
            headers: Sequence[str] = ("init", "goal", "top-down")) -> np.ndarray:
    """Grid: one row per instance, ``tile`` px square cells (missing cells grey), a header row and a label per row."""
    n_cols = max((len(r) for r in rows), default=len(headers))
    bar = 18
    sheet = Image.new("RGB", (n_cols * tile, bar + len(rows) * (tile + bar)), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for j, hname in enumerate(headers[:n_cols]):
        draw.text((j * tile + 4, 3), hname, fill=(255, 255, 255))
    for i, row in enumerate(rows):
        y0 = bar + i * (tile + bar)
        label = labels[i] if i < len(labels) else ""
        draw.text((4, y0 + 2), label[:200], fill=(255, 255, 0))
        for j in range(n_cols):
            img = row[j] if j < len(row) else None
            if img is None:
                cell = Image.new("RGB", (tile, tile), (70, 70, 70))
            else:
                cell = Image.fromarray(_to_uint8(img)).convert("RGB").resize((tile, tile), Image.BILINEAR)
            sheet.paste(cell, (j * tile, y0 + bar))
    return np.asarray(sheet)


def sidecar_overlay_facts(sidecar: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, List[float]], List[List[List[float]]]]:
    """Stances (base poses, the ``goal`` block included) and path waypoint lists of a sidecar dict for the overlay."""
    stances: Dict[str, List[float]] = {}
    paths: List[List[List[float]]] = []
    if not sidecar:
        return stances, paths
    for name, st in (sidecar.get("stances") or {}).items():
        if isinstance(st, dict) and len(st.get("base_pose2d") or []) == 3:
            stances[name] = [float(v) for v in st["base_pose2d"]]
    goal = sidecar.get("goal")
    if isinstance(goal, dict) and len(goal.get("base_pose2d") or []) == 3:
        stances["goal"] = [float(v) for v in goal["base_pose2d"]]
    for p in sidecar.get("paths") or []:
        wps = p.get("waypoints") if isinstance(p, dict) else None
        if wps:
            paths.append([[float(v) for v in w] for w in wps])
    return stances, paths


# ==========================================================================================
# v2: the simulator side
# ==========================================================================================


def _topdown_from_scene(scene, sidecar: Optional[Mapping[str, Any]], physical: Mapping[str, Any], robot, agent_instance: str, title: str) -> Optional[np.ndarray]:
    from ispatialgym.robot.conventions import yaw_of

    trav = getattr(scene, "trav_map", None)
    if trav is None or not getattr(trav, "floor_map", None):
        return None
    fm = trav.floor_map[0]
    fm = fm.cpu().numpy() if hasattr(fm, "cpu") else np.asarray(fm)

    def to_map(x: float, y: float) -> Tuple[int, int]:
        ij = trav.world_to_map(th.tensor([float(x), float(y)]))
        return int(ij[0]), int(ij[1])

    stances, paths = sidecar_overlay_facts(sidecar)
    objects = {}
    for inst, ent in physical.items():
        if inst == agent_instance or ent is robot or getattr(ent, "category", None) == "floors":
            continue
        p = ent.get_position_orientation()[0]
        objects[inst] = [float(p[0]), float(p[1])]
    rp, rq = robot.get_position_orientation()
    robot_pose2d = [float(rp[0]), float(rp[1]), float(yaw_of(rq))]
    return topdown_overlay(fm, to_map, stances, paths, objects, robot_pose2d, resolution_m=float(getattr(trav, "map_resolution", 0.1)), title=title)


def render_template_v2(template_dir: str, goal: bool = False, topdown: bool = True, render_iters: int = 3, image_size: int = V.DEFAULT_IMAGE_SIZE,
                       head_aperture: float = V.DEFAULT_HEAD_APERTURE_MM, tile: int = MONTAGE_TILE) -> List[str]:
    """Render every instance of a v2 template (see the module docstring); returns the PNGs written."""
    from ispatialgym.oracle import run_isg as RI  # imports the cuRobo planner before omnigibson (warp pinning)

    import omnigibson as og

    from ispatialgym.activity.problem import AGENT_INSTANCE

    template_dir = os.path.abspath(template_dir)
    insts = V.list_instances(template_dir)
    if not insts:
        return []
    out_dir = os.path.join(template_dir, RENDERS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    rows: List[List[Optional[np.ndarray]]] = []
    labels: List[str] = []
    env = cfg = None
    prev = None
    for inst in insts:
        loader = None
        try:
            loader = RI.InstanceLoader(inst.tro_path, image_size=image_size, head_aperture=head_aperture)
            if env is None:
                env = loader.build_env()
                cfg = loader.cfg
            else:
                V._release_previous(prev, env.robots[0], og)
                V._adopt_env(loader, env, cfg)
            try:
                loader.restore()
            except RI.LoadError as e:
                log.error("%s: loader contract violated: [%s] %s", inst.iid, e.reason, e.message)
                rows.append([None, None, None])
                labels.append(f"{inst.iid}  (load failed: {e.reason})")
                continue
            robot, task, cam_key, physical = loader.robot, loader.task, loader.cam_key, loader.physical
            sidecar = V._read_json(inst.sidecar_path) if inst.sidecar_path else None
            init_rgb = V._head_rgb(robot, cam_key, og, render_iters)
            p = os.path.join(out_dir, f"{inst.iid}-init.png")
            Image.fromarray(init_rgb).save(p)
            written.append(p)
            top = None
            if topdown:
                try:
                    top = _topdown_from_scene(env.scene, sidecar, physical, robot, AGENT_INSTANCE, inst.iid)
                except Exception as e:  # noqa: BLE001
                    log.warning("%s: top-down overlay failed: %r", inst.iid, e)
                if top is not None:
                    p = os.path.join(out_dir, f"{inst.iid}-topdown.png")
                    Image.fromarray(top).save(p)
                    written.append(p)
            goal_rgb = None
            note = ""
            if goal:
                g = V.materialise_goal(env, task, loader.tro, physical, robot, sidecar, og, V.GOAL_SETTLE_STEPS)
                if g["materialised"]:
                    goal_rgb = V._head_rgb(robot, cam_key, og, render_iters)
                    p = os.path.join(out_dir, f"{inst.iid}-goal.png")
                    Image.fromarray(goal_rgb).save(p)
                    written.append(p)
                    if g["binary_goal_after_teleport"] is False:
                        note = f"  goal not satisfied after teleport: {g['unsatisfied']}"
                else:
                    note = "  goal not materialised: " + "; ".join(g["skipped"])
            rows.append([init_rgb, goal_rgb, top])
            labels.append(inst.iid + note)
            log.info("%s rendered%s", inst.iid, note)
            env.reset()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            log.error("%s failed: %r", inst.iid, e)
            rows.append([None, None, None])
            labels.append(f"{inst.iid}  (error: {e!r})")
        prev = loader
    if rows:
        p = os.path.join(out_dir, "montage.png")
        Image.fromarray(montage(rows, labels, tile)).save(p)
        written.append(p)
        log.info("montage -> %s", p)
    og.clear()
    return written


# ==========================================================================================
# v1 (sample_instances layout), unchanged
# ==========================================================================================


def render_template(template_dir: str, robot_name: str = "robot_r1", res: int = None, render_iters: int = None):
    res = CAM_RES if res is None else int(res)
    render_iters = RENDER_ITERS if render_iters is None else int(render_iters)
    import omnigibson as og
    from omnigibson.macros import gm

    for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False)):
        if getattr(gm, name) != value:
            with gm.unlocked():
                setattr(gm, name, value)
    from ispatialgym.eval.group_entity import GroupEntity
    from ispatialgym.eval.tro import load_tro
    from ispatialgym.oracle import r1pro_model as M
    from ispatialgym.sampling.sim_utils import PARK_POSITION
    from ispatialgym.tasks import ISpatialGymTask  # noqa: F401

    with open(os.path.join(template_dir, "template_meta.json")) as f:
        meta = json.load(f)
    problem = meta["problem"]
    activity, did = problem.rsplit("-", 1)
    tros = sorted(glob.glob(os.path.join(template_dir, "instances", "*-tro_state.json")))
    if not tros:
        return []
    # BEHAVIOR replay external camera (joylo/scripts/replay_data.py): over the robot's shoulder, looking down-forward
    external_sensors = [
        {
            "sensor_type": "VisionSensor",
            "name": "external_sensor0",
            "relative_prim_path": f"/controllable__r1pro__{robot_name}/base_link/external_sensor0",
            "modalities": ["rgb"],
            "sensor_kwargs": {"image_height": res * 2, "image_width": res * 3, "horizontal_aperture": 40.0},
            "position": th.tensor([-0.4, 0.0, 2.0], dtype=th.float32),
            "orientation": th.tensor([0.2706, -0.2706, -0.6533, 0.6533], dtype=th.float32),
            "pose_frame": "parent",
        }
    ]
    cfg = {
        "env": {"action_frequency": 30, "physics_frequency": 120, "external_sensors": external_sensors},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": meta["scene"],
            "scene_file": os.path.join(template_dir, "template.json"),
            "load_room_instances": meta.get("load_room_instances") or None,
        },
        "robots": [
            M.joylo_robot_config(
                robot_name,
                obs_modalities=["rgb"],
                sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": res, "image_width": res}}},
                position=PARK_POSITION.tolist(),
            )
        ],
        "task": {
            "type": "ISpatialGymTask",
            "activity_name": activity,
            "activity_definition_id": int(did),
            "activity_definitions_root": os.path.join(template_dir, "activity_definitions"),
            "tro_path": tros[0],
            "termination_config": {"max_steps": 10 ** 6},
        },
    }
    env = og.Environment(configs=cfg)
    task = env.task
    robot = env.robots[0]
    head_key = [k for k in robot.sensors if k.endswith(M.HEAD_CAMERA_SENSOR_SUFFIX)][0]
    left_key = [k for k in robot.sensors if "left" in k][0]
    right_key = [k for k in robot.sensors if "right" in k][0]
    for k in (head_key, left_key, right_key):
        robot.sensors[k].horizontal_aperture = 40.0
    external = env.external_sensors["external_sensor0"]
    written = []
    for path in tros:
        iid = os.path.basename(path).replace("-tro_state.json", "")
        try:
            tro = load_tro(path)
            task.load_tro_instance(env, tro=tro, settle_steps=10)
            scope = task.object_scope
            # objects the loader cannot restore yet: pose them from the TRO with gravity off so the picture is the init
            extras = tro.meta.extras
            frozen = {c["child"] for c in (extras.get("constraints") or [])} | {g["object"] for g in (extras.get("grasps") or [])}
            for inst in frozen:
                ent = scope.get(inst)
                st = tro.object_states.get(inst)
                if ent is None or isinstance(ent, GroupEntity) or st is None:
                    continue
                root = st["root_link"]
                ent.set_position_orientation(th.tensor(root["pos"], dtype=th.float32), th.tensor(root["ori"], dtype=th.float32))
                ent.disable_gravity()
                ent.keep_still()
            robot.keep_still()
            og.sim.step_physics()
            for _ in range(render_iters):
                og.sim.render()
            obs, _ = robot.get_obs()
            ext_obs, _ = external.get_obs()
            sheet = compose(_to_uint8(obs[left_key]["rgb"]), _to_uint8(obs[head_key]["rgb"]), _to_uint8(obs[right_key]["rgb"]), _to_uint8(ext_obs["rgb"]), res)
            out = path.replace("-tro_state.json", "-init.png")
            Image.fromarray(sheet).save(out)
            written.append(out)
            log.info("%s -> %s", iid, out)
            env.reset()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            log.error("%s failed: %r", iid, e)
    og.clear()
    return written


# ==========================================================================================
# CLI
# ==========================================================================================


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="v2 output root (<root>/<alias>/<template_id>/...)")
    parser.add_argument("--out", default=None, help="v1 sampler output directory; same as --root for the v1 layout")
    parser.add_argument("--alias", action="append", default=None, help="restrict to these aliases / problem directories; repeatable")
    parser.add_argument("--template", action="append", default=None, help="restrict to these template ids or directories; repeatable")
    parser.add_argument("--problem", action="append", default=None, help="(v1) problem directories; alias of --alias")
    parser.add_argument("--goal", action="store_true", help="(v2) also render the goal configuration (teleport as the sampler's goal check)")
    parser.add_argument("--no-topdown", action="store_true", help="(v2) skip the top-down stance overlay")
    parser.add_argument("--tile", type=int, default=MONTAGE_TILE, help="(v2) montage tile side")
    parser.add_argument("--image-size", type=int, default=V.DEFAULT_IMAGE_SIZE, help="(v2) head camera resolution (isg-v1: 1080)")
    parser.add_argument("--res", type=int, default=CAM_RES, help="(v1) side of each robot camera tile")
    parser.add_argument("--render-iters", type=int, default=None, help="frames rendered before the sensors are read (v1: 10, v2: 3)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)
    root = args.root or args.out
    if not root:
        raise SystemExit("--root (v2) or --out (v1) is required")
    aliases = list(args.alias or []) + list(args.problem or []) or None
    templates = V.discover_templates(root, aliases, args.template)
    v2 = [t for t in templates if V.detect_layout(t) == "v2"]
    v1 = [t for t in templates if t not in v2]
    log.info("%d templates under %s (%d v2, %d v1)", len(templates), root, len(v2), len(v1))
    from ispatialgym.sampling import robot_utils  # noqa: F401  (cuRobo / warp pinning before omnigibson)

    import omnigibson as og

    report: Dict[str, Any] = {"v2": {}, "v1": {}}
    sheets: Dict[str, List[str]] = {}
    try:
        for tdir in v2:
            try:
                written = render_template_v2(tdir, goal=args.goal, topdown=not args.no_topdown, render_iters=args.render_iters or 3, image_size=args.image_size, tile=args.tile)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                log.error("template %s failed: %r", tdir, e)
                try:
                    og.clear()
                except Exception:  # noqa: BLE001
                    pass
                written = []
            report["v2"][os.path.relpath(tdir, root)] = len(written)
        for tdir in v1:
            pname = os.path.basename(os.path.dirname(tdir))
            try:
                written = render_template(tdir, res=args.res, render_iters=args.render_iters or RENDER_ITERS)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                log.error("template %s failed: %r", tdir, e)
                try:
                    og.clear()
                except Exception:  # noqa: BLE001
                    pass
                continue
            if written:
                sheets.setdefault(pname, []).append(written[0])
        # v1 contact sheet per problem: first instance of every template, stacked vertically
        if sheets:
            os.makedirs(os.path.join(root, RENDERS_DIR), exist_ok=True)
            for pname, paths in sheets.items():
                imgs = [np.asarray(Image.open(p)) for p in paths]
                Image.fromarray(np.concatenate(imgs, axis=0)).save(os.path.join(root, RENDERS_DIR, f"{pname}.png"))
            report["v1"] = {k: len(v) for k, v in sheets.items()}
        print("RENDER_REPORT " + json.dumps(report), flush=True)
    finally:
        if getattr(og, "app", None) is not None:
            og.shutdown()


if __name__ == "__main__":
    main()
