"""B7 — held-object visibility cache per (model, grasp) for object scaling / mental rotation (spec §6.3, §3.10).

For every target pick of ``object_scaling`` / ``mental_rotation`` in ``icl/roles/representatives.csv`` (slots
``representative``, ``variation_geometry``, ``heldout``, ``variation_texture``) and every admissible ``floating`` grasp
row of ``icl/require/grasps.csv`` (verdict ``ok`` or ``flag`` with flags ⊆ {``one_finger``}; the poses come from the
grasp library the csv indexes), the held pose is realised in an **empty scene** exactly as the v2 sampler does
(``sampling/v2/tasks/held_object.py``): R1Pro at the JoyLo reset trunk (0.5, 0) with the left arm zero, the right arm
at the **fixed hold** (pad centre on the head camera's optical axis at depth 0.50 m, approach = base +y, closing =
world down rolled 15° about the approach; one cuRobo IK, written into the sampler's hold cache), the object placed
at ``eef ∘ inv(T_obj→eef)`` of the library grasp with the fingers at the library's closed finger positions, and two
instance-segmentation renders of the head camera are taken: with the robot (visible pixels) and with the robot hidden
(reference silhouette). Recorded per (grasp key, grasp name): the visible fraction, whether the whole object (AABB of
its collision hull) is inside the image with the 5 % margin, the pixel counts, the library verdict / flags, and the
object's depth along the optical axis. The sampler reads ``visibility.json`` as a prior and skips rows below
``min_visible_fraction`` (0.40) before realising them (``held_object.prior_admits``).

Assisted grasping is *not* run here (the survey is a render prior; the sampler realises every grasp it uses, v1 §5.2),
so a row's realised pose may differ from the surveyed one by the few millimetres the attach pulls the object in.

Usage (local 4090, headless, one process; a few seconds per grasp after the scene load)::

    OMNIGIBSON_HEADLESS=1 python scripts/held_visibility/survey.py --out runlogs/held_visibility_20260915/visibility.json \
        [--subtasks object_scaling,mental_rotation] [--slots representative,variation_geometry,heldout,variation_texture] \
        [--models banana/znakxm,...] [--limit-models N] [--resume]
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch as th  # noqa: E402

from ispatialgym import ASSETS_ROOT, REPO_ROOT  # noqa: E402
from ispatialgym.sampling import robot_utils  # noqa: E402,F401  (warp pinning before omnigibson)

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.robot import conventions as C  # noqa: E402
from ispatialgym.sampling import geometry as G  # noqa: E402
from ispatialgym.sampling import grasp_library as GL  # noqa: E402
from ispatialgym.sampling.robot_utils import RobotKit  # noqa: E402
from ispatialgym.sampling.sim_utils import HEAD_APERTURE_MM, IMAGE_SIZE, PARK_POSITION, SimWorld  # noqa: E402
from ispatialgym.sampling.v2.tasks import held_object as HO  # noqa: E402

log = logging.getLogger("held_visibility")

ICL_ROOT = os.path.join(ASSETS_ROOT, "icl")
DEFAULT_OUT = os.path.join(REPO_ROOT, "runlogs", "held_visibility_20260915", "visibility.json")
SUBTASKS = ("object_scaling", "mental_rotation")
SLOTS = ("representative", "variation_geometry", "heldout", "variation_texture")
SCHEMA = "held-visibility-v1"


# ----------------------------------------------------------------------------------------------------------------
# picks and grasp rows (icl)
# ----------------------------------------------------------------------------------------------------------------


def target_picks(subtasks: Tuple[str, ...], slots: Tuple[str, ...], only: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Distinct (category, model) target picks with their slots / subtasks, grasp key and spawn scale."""
    picks: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(os.path.join(ICL_ROOT, "roles", "representatives.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["subtask"] not in subtasks or row["role"] != "target" or row["slot"] not in slots:
                continue
            key = (row["category"], row["model"])
            if only and f"{key[0]}/{key[1]}" not in only:
                continue
            p = picks.setdefault(key, {"category": key[0], "model": key[1], "slots": [], "subtasks": [], "picks": [], "grasp_key": row.get("grasp_key") or None,
                                       "scale": _parse_scale(row.get("scale"))})
            p["slots"].append(row["slot"])
            p["subtasks"].append(row["subtask"])
            p["picks"].append(row["pick"])
            if not p["grasp_key"] and row.get("grasp_key"):
                p["grasp_key"] = row["grasp_key"]
    out = list(picks.values())
    for p in out:
        p["slots"] = sorted(set(p["slots"]))
        p["subtasks"] = sorted(set(p["subtasks"]))
    return sorted(out, key=lambda p: (p["category"], p["model"]))


def _parse_scale(text: Optional[str]):
    if not text:
        return None
    try:
        v = json.loads(text)
    except ValueError:
        return None
    if isinstance(v, (int, float)):
        return [float(v)] * 3
    if isinstance(v, list) and len(v) == 3:
        return [float(x) for x in v]
    return None


def grasp_rows_index() -> Dict[Tuple[str, str], Dict[str, str]]:
    """``icl/require/grasps.csv`` rows of the floating pipeline by (key, name): the admissibility filter of §2.2."""
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with open(os.path.join(ICL_ROOT, "require", "grasps.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row.get("require") != "grasps:floating":
                continue
            out[(row["key"], row["name"])] = row
    return out


def admissible(row: Dict[str, str]) -> bool:
    flags = {t for t in (row.get("flags") or "").replace(",", " ").split() if t}
    return row.get("verdict") == "ok" or (row.get("verdict") == "flag" and flags <= GL.TOLERATED_FLAGS)


# ----------------------------------------------------------------------------------------------------------------
# empty world
# ----------------------------------------------------------------------------------------------------------------


class EmptyWorld(SimWorld):
    """An OmniGibson world with an empty ``Scene`` (ground plane) and the R1Pro; the ``SimWorld`` render / spawn
    helpers work unchanged on it (no rooms, no traversability map)."""

    def __init__(self, image_size: int = IMAGE_SIZE):  # noqa: D107 - deliberately not calling SimWorld.__init__
        for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False)):
            if getattr(gm, name) != value:
                with gm.unlocked():
                    setattr(gm, name, value)
        self.scene_model = "empty"
        self.rooms = None
        self.adjacent = []
        self.eval_rooms = None
        self.loaded_rooms = None
        self.robot_name = "robot_r1"
        self.scene_variant = {}
        cfg = {
            "env": {"action_frequency": 30, "physics_frequency": 120},
            "scene": {"type": "Scene"},
            "robots": [M.joylo_robot_config(self.robot_name, obs_modalities=["rgb", "seg_instance"],
                                            sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": image_size, "image_width": image_size}}},
                                            position=[0.0, 0.0, robot_utils.NOMINAL_BASE_Z])],
        }
        t0 = time.time()
        self.env = og.Environment(configs=cfg)
        self._set_measurement_render_settings()
        self.scene = self.env.scene
        self.robot = self.env.robots[0]
        self.cam_key = [k for k in self.robot.sensors if k.endswith(M.HEAD_CAMERA_SENSOR_SUFFIX)][0]
        self.cam = self.robot.sensors[self.cam_key]
        self.cam.horizontal_aperture = HEAD_APERTURE_MM
        self.camera_model = G.CameraModel.from_sensor(float(self.cam.focal_length), HEAD_APERTURE_MM, image_size, image_size)
        self.n_scene_objects = len(self.scene.objects)
        self.scene_object_names = {o.name for o in self.scene.objects}
        self.spawned = {}
        self.floor_z = 0.0
        self.render_hidden = []
        self.robot_visible = True
        self.task_bounds = None
        self.robot_park_xy = th.tensor([0.0, 0.0])
        self.robot.set_joint_positions(th.tensor(M.JOYLO_RESET_JOINT_POS, dtype=th.float32))
        self.robot.set_joint_velocities(th.zeros(self.robot.n_joints))
        self.robot.set_position_orientation(th.tensor([0.0, 0.0, robot_utils.NOMINAL_BASE_Z]), G.IDENTITY_QUAT)
        self.robot.keep_still()
        for _ in range(5):
            og.sim.step_physics()
            self.robot.keep_still()
        self.load_time = time.time() - t0
        log.info("empty world loaded in %.1fs (%d objects)", self.load_time, self.n_scene_objects)


# ----------------------------------------------------------------------------------------------------------------
# survey
# ----------------------------------------------------------------------------------------------------------------


def solve_hold(world: EmptyWorld, kit: RobotKit, hc: HO.HoldConstants) -> Dict[str, Any]:
    """The fixed hold at the robot's (origin) base pose; also seeds the sampler's hold cache with the same key."""
    x, y, yaw = kit.world_base_pose2d()
    base3 = (th.tensor([x, y, kit.floor_z + C.NOMINAL_BASE_Z]), C.quat_from_yaw(yaw))
    q0 = kit.q_at_base_pose(kit.q_config(arms=HO.IDLE_ARMS, translate=hc.translate, tilt=hc.tilt), x, y, yaw)
    cam = kit.camera_pose(q0)
    eef_w, pad_w = HO.hold_eef_pose(cam, yaw, hc)
    q = kit.ik(eef_w, q0, arm=HO.ARM)
    if q is None:
        raise SystemExit(f"no IK solution for the fixed hold {hc.to_json()}")
    if not bool(kit.collision_free(q, margin=0.01)[0]):
        raise SystemExit("the fixed hold self-collides")
    arm_idx = world.robot.arm_control_idx[HO.ARM]
    extras = {"idle_arms": HO.IDLE_ARMS, "contact_center_z": C.CONTACT_CENTER_Z, "cam_in_zed": [kit.cam_in_zed[0].tolist(), kit.cam_in_zed[1].tolist()], "robot_model": "r1pro"}
    key = HO.hold_cache_key("r1pro", hc, extras)
    eef_in_base = C.relative(eef_w, base3)
    cam_in_base = C.relative(cam, base3)
    sol = {"key": key, "robot": "r1pro", **hc.to_json(), "approach": "base +y", "closing": f"world down rolled {hc.roll_deg} deg about the approach",
           "eef_in_base": {"position": eef_in_base[0].tolist(), "orientation": eef_in_base[1].tolist()},
           "pad_in_base": C.relative((pad_w, C.IDENTITY_QUAT), base3)[0].tolist(),
           "camera_in_base": {"position": cam_in_base[0].tolist(), "orientation": cam_in_base[1].tolist()},
           "right_arm_q": q[arm_idx].tolist(), "idle_arms": HO.IDLE_ARMS}
    cache = HO._read_json(HO.HOLD_CACHE_PATH)
    if key not in cache:
        cache[key] = dict(sol)
        HO._write_json(HO.HOLD_CACHE_PATH, cache)
        log.info("hold written to %s (%s)", HO.HOLD_CACHE_PATH, key)
    return {"solution": sol, "q": q, "eef_w": eef_w, "cam": cam, "base": (x, y, yaw), "pad_w": pad_w}


def object_pixels_settled(world: EmptyWorld, obj) -> Tuple[int, int]:
    vis = world.object_pixels(obj)
    ref = world.object_pixels(obj, hidden=[world.robot])
    if ref == 0:
        world.ensure_segmented(obj)
        vis = world.object_pixels(obj)
        ref = world.object_pixels(obj, hidden=[world.robot])
    return int(vis), int(ref)


def survey_model(world: EmptyWorld, kit: RobotKit, hold: Dict[str, Any], pick: Dict[str, Any], rows: Dict[Tuple[str, str], Dict[str, str]],
                 in_frame_margin: float) -> List[Dict[str, Any]]:
    cat, model = pick["category"], pick["model"]
    t0 = time.time()
    try:
        key = HO.floating_key(cat, model, pick.get("grasp_key"))
        grasps = GL.load_grasps(key)
    except (FileNotFoundError, ValueError) as e:
        return [{"category": cat, "model": model, "grasp_key": pick.get("grasp_key"), "grasp_name": None, "error": f"missing_grasps:floating: {e}",
                 "slots": pick["slots"], "subtasks": pick["subtasks"]}]
    name = f"survey_{cat}_{model}"
    obj = world.spawn(name, cat, model, scale=pick.get("scale"))
    out: List[Dict[str, Any]] = []
    try:
        hull = world.local_hull(obj)
        q_hold = hold["q"]
        base = hold["base"]
        eef_w = hold["eef_w"]
        gidx = world.robot.gripper_control_idx[HO.ARM]
        for g in grasps:
            row = rows.get((key, g.name), {})
            t1 = time.time()
            rec: Dict[str, Any] = {"category": cat, "model": model, "scale": pick.get("scale"), "grasp_key": key, "grasp_name": g.name, "verdict": g.verdict,
                                   "flags": list(g.flags), "icl_admissible": admissible(row) if row else None, "slots": pick["slots"], "subtasks": pick["subtasks"]}
            obj_pose = C.compose(eef_w, C.invert(g.pose))
            kit.apply(q_hold, base)
            if g.finger_q and len(g.finger_q) == len(gidx):
                world.robot.set_joint_positions(th.tensor(g.finger_q, dtype=th.float32), indices=gidx, drive=False)
            obj.set_position_orientation(obj_pose[0], obj_pose[1])
            obj.disable_gravity()
            obj.wake()
            obj.keep_still()
            world.robot.keep_still()
            world.render(2)
            cam = world.camera_pose()
            corners = world.hull_at(hull, obj_pose)
            lo, hi = corners.min(dim=0).values, corners.max(dim=0).values
            in_frame = world.camera_model.all_inside(cam, G.aabb_corners(lo, hi), in_frame_margin)
            axis = world.camera_model.optical_axis(cam)
            depth = float(th.dot(G.t(obj_pose[0]) - G.t(cam[0]), axis))
            try:
                vis, ref = object_pixels_settled(world, obj)
            except Exception as e:  # noqa: BLE001 - one row must not stop the survey
                rec.update({"error": f"render: {e!r}", "visible_fraction": None, "in_frame": bool(in_frame)})
                out.append(rec)
                continue
            rec.update({"visible_pixels": vis, "reference_pixels": ref, "visible_fraction": HO.visible_fraction(vis, ref) if ref > 0 else None,
                        "in_frame": bool(in_frame), "root_depth_m": round(depth, 4), "seconds": round(time.time() - t1, 2)})
            if ref == 0:
                rec["error"] = "no segmentation pixels with the robot hidden"
            out.append(rec)
            log.info("%s/%s %s: visible %.2f (%d/%d px) in_frame=%s depth %.3f", cat, model, g.name, rec.get("visible_fraction") or 0.0, vis, ref, in_frame, depth)
    finally:
        world.remove(obj)
        kit.sync_world()
    log.info("%s/%s: %d grasps in %.1fs", cat, model, len(out), time.time() - t0)
    return out


def summarise(records: List[Dict[str, Any]], min_fraction: float) -> Dict[str, Any]:
    ok = [r for r in records if r.get("visible_fraction") is not None]
    models = {(r["category"], r["model"]) for r in records}
    per_model: Dict[str, Dict[str, Any]] = {}
    for (c, m) in sorted(models):
        rs = [r for r in ok if (r["category"], r["model"]) == (c, m)]
        adm = [r for r in rs if r["visible_fraction"] >= min_fraction and r.get("in_frame")]
        per_model[f"{c}/{m}"] = {"grasps": len([r for r in records if (r["category"], r["model"]) == (c, m)]), "measured": len(rs), "admitted": len(adm),
                                 "best_visible_fraction": max((r["visible_fraction"] for r in rs), default=None)}
    return {"models": len(models), "grasps": len(records), "measured": len(ok), "admitted": sum(1 for r in ok if r["visible_fraction"] >= min_fraction and r.get("in_frame")),
            "below_threshold": sum(1 for r in ok if r["visible_fraction"] < min_fraction), "out_of_frame": sum(1 for r in ok if not r.get("in_frame")),
            "errors": sum(1 for r in records if r.get("error")), "models_without_admitted_grasp": [k for k, v in per_model.items() if v["admitted"] == 0],
            "per_model": per_model}


def write_output(path: str, hold: Dict[str, Any], records: List[Dict[str, Any]], args, started: str, min_fraction: float):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {"schema": SCHEMA, "date": started, "finished": _dt.datetime.now().isoformat(timespec="seconds"), "robot": "r1pro", "image_size": IMAGE_SIZE,
            "in_frame_margin": args.in_frame_margin, "min_visible_fraction": min_fraction, "hold": hold, "method": "empty scene, hold pose from the sampler's "
            "hold_eef_pose + one cuRobo IK, object at eef ∘ inv(T_obj→eef) of the library grasp, fingers at the library finger_q, seg_instance renders "
            "with / without the robot (no assisted grasping)", "args": {k: v for k, v in vars(args).items()}, "summary": summarise(records, min_fraction),
            "records": records}
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--subtasks", default=",".join(SUBTASKS))
    parser.add_argument("--slots", default=",".join(SLOTS))
    parser.add_argument("--models", default=None, help="restrict to category/model,... (debug)")
    parser.add_argument("--limit-models", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="skip models already in --out")
    parser.add_argument("--in-frame-margin", type=float, default=0.05)
    parser.add_argument("--min-visible-fraction", type=float, default=None, help="default: generation_defaults object_scaling.min_visible_fraction")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)

    from ispatialgym.sampling import generation_defaults as GD

    defaults = GD.defaults_for("object_scaling")
    hc = HO.HoldConstants.from_generation(defaults["hold"])
    min_fraction = float(args.min_visible_fraction if args.min_visible_fraction is not None else defaults["min_visible_fraction"])
    subtasks = tuple(s for s in args.subtasks.split(",") if s)
    slots = tuple(s for s in args.slots.split(",") if s)
    only = [m for m in args.models.split(",") if m] if args.models else None
    picks = target_picks(subtasks, slots, only)
    if args.limit_models:
        picks = picks[: args.limit_models]
    rows = grasp_rows_index()
    records: List[Dict[str, Any]] = []
    done_models = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            records = json.load(f).get("records", [])
        done_models = {(r["category"], r["model"]) for r in records}
    picks = [p for p in picks if (p["category"], p["model"]) not in done_models]
    log.info("%d picks to survey (%d subtasks, slots %s), %d floating grasp rows indexed", len(picks), len(subtasks), slots, len(rows))
    started = _dt.datetime.now().isoformat(timespec="seconds")
    world = EmptyWorld()
    kit = RobotKit(world, prebuild=("right_arm",))
    try:
        hold = solve_hold(world, kit, hc)
        log.info("hold: %s", {k: v for k, v in hold["solution"].items() if k in ("key", "depth_m", "trunk", "roll_deg")})
        for i, p in enumerate(picks):
            t0 = time.time()
            recs = survey_model(world, kit, hold, p, rows, args.in_frame_margin)
            records.extend(recs)
            write_output(args.out, hold["solution"], records, args, started, min_fraction)
            log.info("[%d/%d] %s/%s done in %.1fs", i + 1, len(picks), p["category"], p["model"], time.time() - t0)
        write_output(args.out, hold["solution"], records, args, started, min_fraction)
        s = summarise(records, min_fraction)
        print("HELD_VISIBILITY_REPORT " + json.dumps({k: v for k, v in s.items() if k != "per_model"}), flush=True)
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
