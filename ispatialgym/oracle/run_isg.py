"""``run_isg`` -- v1 oracle driver: load one sampled iSpatialGym instance, run the oracle, record, evaluate.

TODO F1 / F4 "lite". One process = one instance = one episode::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.oracle.run_isg \\
        --instance <out>/<problem>/<template_id>/instances/<iid>-tro_state.json \\
        [--mode oracle] [--out <dir>] [--video] [--max-steps N]

What it does

1. Loads the instance exactly like the evaluator does (``sampling/validate_instances.py``): the template scene
   (``template.json``, partial rooms) with the R1Pro JoyLo configuration, an :class:`ISpatialGymTask` built from the
   template's own ``activity_definitions`` and the TRO applied with ``ISpatialGymTask.load_tro_instance``. TRO
   ``grasps`` (held-object tasks) are restored with OmniGibson's assisted-grasp joint, which the task loader does not
   do yet. The robot is spawned at the TRO base pose so that the holonomic base's virtual joints (and therefore the
   cuRobo base planner, whose derived URDF limits them to +-10 m) stay near zero.
2. Runs the v1 oracle of the instance's activity (``oracle/tasks``): ``camera_relocalization`` (goal = TRO
   ``pose_targets[agent]`` camera pose), ``object_scaling`` (goal depth of the object root along the head camera's
   optical axis, from ``pose_targets[target]``), ``mental_rotation`` (goal orientation). The executor here owns
   ``env.step``: the oracle generators only yield 23-D JoyLo actions.
3. After the oracle and a no-op settle window of ``max(--settle-steps, 20 %)`` steps the task's own evaluator decides:
   ``info["done"]["ispatialgym"]`` of the last step (binary goal, settled goal, measured errors).
4. Records per step into ``<out>/<iid>/demo.hdf5`` (actions, joint positions / velocities, base / eef / head-camera
   poses, task-object poses, evaluator flags and errors, phase) and, with ``--video``, the head camera
   (``<robot>:zed_link:Camera:0``) RGB as ``<out>/<iid>/head_rgb.mp4``; ``init_rgb.png`` / ``final_rgb.png`` are
   always written. ``meta.json`` carries the summary (instance id, alias, success flag, errors, wall times, step
   counts, oracle name, QA reason, traceback on error).
5. Prints one line ``RUN_ISG_RESULT {...}`` and exits with

   * ``0`` success (the task's binary goal was reached),
   * ``2`` oracle / QA failure (``reason`` in ``plan_failed``, ``no_motion``, ``collision``, ``timeout``,
     ``eval_false``, ``execution``, ``failed_grasp``),
   * ``3`` ``no_oracle``: the activity has no v1 oracle (disembedding, configuration_matching, world_alignment,
     inter_object_relation_satisfaction, ...); the instance is still loaded and its init view rendered,
   * ``4`` ``error``: any exception (traceback in ``meta.json``).

Real-time QA (lite): contacts of the robot and of the held objects with anything but the floor are recorded every
step; ``--collision-policy`` decides whether they fail the run (default ``fail``: the run is completed and evaluated
but reported as ``collision``), abort it at the first contact, or are only recorded.

``--out`` defaults to ``<template_dir>/runs``. Only ``--mode oracle`` exists (replay / policy: TODO F1).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np  # noqa: E402
import torch as th  # noqa: E402

from ispatialgym.oracle.curobo_planner import R1ProCuRobo  # noqa: E402, F401  (import before omnigibson: warp pinning)

import omnigibson as og  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
from omnigibson.utils.usd_utils import RigidContactAPI  # noqa: E402

from ispatialgym.activity.problem import AGENT_INSTANCE  # noqa: E402
from ispatialgym.eval.group_entity import GroupEntity  # noqa: E402
from ispatialgym.eval.tro import TROFile, load_tro  # noqa: E402
from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.oracle.action import ActionState  # noqa: E402
from ispatialgym.oracle.poses import Pose, as_tensor, quat_angle, relative  # noqa: E402
from ispatialgym.oracle.skills import SkillError, Skills  # noqa: E402
from ispatialgym.oracle.tasks import CameraRelocalizationOracle, MentalRotationOracle, ObjectScalingOracle  # noqa: E402
from ispatialgym.oracle.trunk import TrunkState  # noqa: E402
from ispatialgym.tasks import ISpatialGymTask  # noqa: E402, F401  (registers the task type)

log = logging.getLogger("ispatialgym.run_isg")

TRO_SUFFIX = "-tro_state.json"
DEFAULT_IMAGE_SIZE = 1080  # evaluation camera profile (isg-v1)
DEFAULT_HEAD_APERTURE_MM = 40.0  # sampling/sim_utils.HEAD_APERTURE_MM: the camera model the sampler validated visibility with
DEFAULT_ROBOT_NAME = "robot_r1"

#: activities with a v1 oracle -> oracle name
ORACLES = {
    "camera_relocalization": "CameraRelocalizationOracle",
    "object_scaling": "ObjectScalingOracle",
    "mental_rotation": "MentalRotationOracle",
}

EXIT_SUCCESS, EXIT_ORACLE_FAILED, EXIT_NO_ORACLE, EXIT_ERROR = 0, 2, 3, 4

#: SkillError.reason -> QA category (oracle doc 6.4)
QA_REASON = {
    "planning": "plan_failed",
    "timeout": "timeout",
    "collision": "collision",
    "execution": "execution",
    "no_object": "failed_grasp",
    "grasp": "failed_grasp",
    "unknown_object": "plan_failed",
    "bad_frame": "plan_failed",
}

#: contacts with these categories are never collisions (the robot stands / drives on them)
COLLISION_IGNORE_CATEGORIES = {"floors", "carpet", "rug"}

PHASE_ORACLE, PHASE_SETTLE = 0, 1


class OracleFailure(RuntimeError):
    """QA verdict of the run (``reason`` is one of the categories listed in the module docstring)."""

    def __init__(self, reason: str, message: str = "", **details):
        super().__init__(f"[{reason}] {message}")
        self.reason = reason
        self.message = message
        self.details = details


# ==========================================================================================
# small helpers
# ==========================================================================================


def _json_default(o):
    if isinstance(o, th.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def _dump_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _pose7(pos, quat) -> np.ndarray:
    return np.concatenate([as_tensor(pos).numpy(), as_tensor(quat).numpy()]).astype(np.float32)


def _to_uint8_rgb(img) -> np.ndarray:
    arr = img.detach().cpu().numpy() if isinstance(img, th.Tensor) else np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(arr[..., :3])


def _save_png(path: str, img) -> None:
    from PIL import Image

    Image.fromarray(_to_uint8_rgb(img)).save(path)


def _first_key(d: Dict[str, Any], suffix: str) -> Optional[str]:
    for k in d:
        if k.endswith(suffix):
            return k
    return None


def resolve_instance(path: str) -> Tuple[str, str, dict, str, int]:
    """``(instance id, template dir, template_meta, activity, definition id)`` from a ``*-tro_state.json`` path."""
    path = os.path.abspath(path)
    if not path.endswith(TRO_SUFFIX):
        raise ValueError(f"--instance must point at a *{TRO_SUFFIX} file, got {path}")
    iid = os.path.basename(path)[: -len(TRO_SUFFIX)]
    template_dir = os.path.dirname(os.path.dirname(path))
    meta_path = os.path.join(template_dir, "template_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"no template_meta.json next to the instance (expected {meta_path})")
    with open(meta_path) as f:
        meta = json.load(f)
    activity, did = meta["problem"].rsplit("-", 1)
    return iid, template_dir, meta, activity, int(did)


def robot_name_from_tro(tro: TROFile, default: str = DEFAULT_ROBOT_NAME) -> str:
    """The robot name the TRO was sampled with (sensor keys are ``<robot>:<link>:Camera:0``)."""
    go = tro.meta.goal_observation or {}
    key = go.get("sensor_key")
    if not key:
        for tgt in tro.meta.pose_targets.values():
            if tgt.link and ":" in tgt.link:
                key = tgt.link
                break
            if tgt.frame and "::" in tgt.frame:
                key = tgt.frame.split("::", 1)[1]
                break
    return key.split(":", 1)[0] if key and ":" in key else default


def build_env_config(template_dir: str, meta: dict, tro: TROFile, tro_path: str, activity: str, did: int, robot_name: str, image_size: int) -> dict:
    """Evaluator configuration (mirrors ``validate_instances.py``) with the robot spawned at the TRO base pose."""
    robot_kw = {}
    poses = {k.lower(): v for k, v in (tro.robot_poses or {}).items()}
    pose = (poses.get("robot") or poses.get("r1pro") or [None])[0]
    if pose is not None:
        robot_kw = {"position": [float(v) for v in pose["position"]], "orientation": [float(v) for v in pose["orientation"]]}
    return {
        "env": {"action_frequency": 30, "physics_frequency": 120},
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
                sensor_config={"VisionSensor": {"sensor_kwargs": {"image_height": image_size, "image_width": image_size}}},
                **robot_kw,
            )
        ],
        "task": {
            "type": "ISpatialGymTask",
            "activity_name": activity,
            "activity_definition_id": did,
            "activity_definitions_root": os.path.join(template_dir, "activity_definitions"),
            "tro_path": tro_path,
            "termination_config": {"max_steps": 10**6},  # the executor owns the step budget (--max-steps)
        },
    }


def task_bounds(scene, rooms: Sequence[str], margin: float = 1.0):
    """AABB of the task rooms' floors plus ``margin`` (collision-world relevance, as ``SimWorld.task_bounds``)."""
    los, his = [], []
    for o in scene.objects:
        if getattr(o, "category", None) == "floors" and set(o.in_rooms or []) & set(rooms):
            lo, hi = o.aabb
            los.append(lo)
            his.append(hi)
    if not los:
        return None
    lo = th.stack(los).min(dim=0).values - margin
    hi = th.stack(his).max(dim=0).values + margin
    hi[2] = 10.0
    return lo, hi


def make_obstacle_filter(bounds, robot, always: Sequence[str]):
    always = set(always)

    def f(obj):
        if bounds is None or obj is robot or obj.name in always:
            return True
        lo, hi = obj.aabb
        return bool(((hi >= bounds[0]) & (lo <= bounds[1])).all())

    return f


def gripper_command_for(robot, arm: str, finger_q) -> float:
    """``MultiFingerGripperController`` (smooth mode) command holding the fingers at ``finger_q``: [-1, 1] maps to
    [lower, upper] of the mean finger joint limits."""
    idx = robot.gripper_control_idx[arm]
    lo, hi = robot.joint_lower_limits[idx], robot.joint_upper_limits[idx]
    q = as_tensor(finger_q)
    return float(th.clamp(2.0 * (q.mean() - lo.mean()) / (hi.mean() - lo.mean()) - 1.0, -1.0, 1.0))


# ==========================================================================================
# grasp restoration (TRO ``grasps`` -> assisted-grasp joint)
# ==========================================================================================


def restore_grasps(task, robot, tro: TROFile) -> List[dict]:
    """Re-create the assisted-grasp joints recorded in the TRO ``grasps`` block.

    The object and the fingers are already at their TRO poses (``load_tro_instance``), so the joint is created at the
    current relative pose, exactly what ``Robot._maybe_establish_grasp`` does when the grasp window elapses. Returns one
    record per grasp with the deviation of the realised object-in-eef pose from the TRO one.
    """
    out = []
    for g in tro.meta.extras.get("grasps") or []:
        arm, inst = g["arm"], g["object"]
        obj = task.object_scope.get(inst)
        if obj is None or isinstance(obj, GroupEntity):
            raise OracleFailure("failed_grasp", f"grasp record names {inst!r}, which is not a bound physical instance")
        link = obj.root_link_name
        joint_type = robot._get_assisted_grasp_joint_type(obj, link)
        if joint_type is None:
            raise OracleFailure("failed_grasp", f"{obj.name} cannot be assisted-grasped (too heavy?)")
        held = robot._ag_obj_in_hand.get(arm)
        if held is not None and held is not obj:
            robot.release_grasp_immediately(arm)
        if robot._ag_obj_in_hand.get(arm) is not obj:
            contact = robot.eef_links[arm].get_position_orientation()[0]
            robot._establish_grasp(obj, link, arm, contact, joint_type)
        eef = tuple(x.float() for x in robot.eef_links[arm].get_position_orientation())
        rel = relative(tuple(x.float() for x in obj.get_position_orientation()), eef)
        rec = {"arm": arm, "object": inst, "scene_object": obj.name, "link": link, "joint_type": joint_type,
               "grasp_key": g.get("grasp_key"), "grasp_name": g.get("grasp_name"), "finger_q": g.get("finger_q")}
        ref = g.get("object_pose_in_eef_frame")
        if ref:
            rec["object_in_eef_dev_mm"] = 1000.0 * float(th.linalg.norm(rel[0] - as_tensor(ref["position"])))
            rec["object_in_eef_dev_deg"] = math.degrees(quat_angle(rel[1], as_tensor(ref["orientation"])))
        rec["attached"] = robot._ag_obj_in_hand.get(arm) is obj
        out.append(rec)
        log.info("grasp restored: %s in %s hand (%s, dev %.1f mm / %.1f deg)", obj.name, arm, joint_type, rec.get("object_in_eef_dev_mm", 0.0), rec.get("object_in_eef_dev_deg", 0.0))
    return out


# ==========================================================================================
# monitors
# ==========================================================================================


class CollisionMonitor:
    """Contacts of the robot (and of the held objects) with anything but the floor and each other."""

    def __init__(self, env, robot, held: Sequence, ignore_categories=COLLISION_IGNORE_CATEGORIES):
        self.scene = env.scene
        self.scene_idx = env.scene.idx
        self.robot = robot
        self.held = list(held)
        self.query = [robot] + self.held
        self.own_prefixes = [robot.prim_path + "/"] + [o.prim_path + "/" for o in self.held]
        self.ignore_categories = set(ignore_categories)
        self.events: List[dict] = []
        self.by_object: Counter = Counter()
        self._prim_cache: Dict[str, Optional[str]] = {}

    def _object_name(self, body_path: str) -> Optional[str]:
        """Scene object owning ``body_path`` (None for the ground plane / non-object prims / ignored categories)."""
        if body_path in self._prim_cache:
            return self._prim_cache[body_path]
        name = None
        obj = self.scene.object_registry("prim_path", body_path.rsplit("/", 1)[0], None)
        if obj is not None and getattr(obj, "category", None) not in self.ignore_categories:
            name = obj.name
        self._prim_cache[body_path] = name
        return name

    def check(self, step: int) -> List[str]:
        try:
            pairs = RigidContactAPI.get_contact_pairs(self.scene_idx, self.query, None, current_only=False)
        except Exception as e:  # noqa: BLE001  (contact view not available: monitor off)
            log.warning("collision monitor disabled: %r", e)
            self.check = lambda step: []  # type: ignore[assignment]
            return []
        hits = []
        for own, other in pairs:
            if any(other.startswith(p) for p in self.own_prefixes):
                continue
            name = self._object_name(other)
            if name is None:
                continue
            hits.append(name)
            self.by_object[name] += 1
            if len(self.events) < 200:
                self.events.append({"step": step, "own_link": own.rsplit("/", 1)[1], "object": name})
        return hits


class MotionSnapshot:
    """Robot configuration + task-object poses, to tell a no-op run (``no_motion``) from a real one."""

    def __init__(self, robot, objects: Dict[str, Any]):
        self.q = robot.get_joint_positions().clone()
        self.base = tuple(x.clone() for x in robot.get_position_orientation())
        self.objects = {k: tuple(x.clone() for x in o.get_position_orientation()) for k, o in objects.items()}

    def delta(self, other: "MotionSnapshot") -> Dict[str, float]:
        d = {
            "joint_rad": float(th.max(th.abs(other.q - self.q))) if len(self.q) else 0.0,
            "base_m": float(th.linalg.norm(other.base[0] - self.base[0])),
            "base_rad": quat_angle(other.base[1], self.base[1]),
        }
        for k, (p, q) in self.objects.items():
            p1, q1 = other.objects[k]
            d[f"{k}_m"] = float(th.linalg.norm(p1 - p))
            d[f"{k}_rad"] = quat_angle(q1, q)
        return d

    @staticmethod
    def moved(delta: Dict[str, float], pos_tol: float = 0.02, ang_tol: float = 0.03) -> bool:
        """Thresholds above the physics settling jitter of a held object (~3 mm / 0.01 rad over 50 hold steps)."""
        return any((v > pos_tol) if k.endswith("_m") else (v > ang_tol) for k, v in delta.items())


# ==========================================================================================
# recorder (hdf5 + mp4)
# ==========================================================================================


class Recorder:
    def __init__(self, out_dir: str, env, robot, cam_key: str, objects: Dict[str, Any], fps: int, video: bool, video_size: Optional[int]):
        self.out_dir = out_dir
        self.env, self.robot, self.cam_key = env, robot, cam_key
        self.objects = objects
        self.fps = fps
        self.rows: Dict[str, list] = defaultdict(list)
        self.n = 0
        self.video_path = os.path.join(out_dir, "head_rgb.mp4") if video else None
        self.video_size = video_size
        self.writer = None
        self.video_error: Optional[str] = None
        self.last_rgb: Optional[np.ndarray] = None
        if video:
            try:
                import imageio.v2 as imageio

                self.writer = imageio.get_writer(
                    self.video_path, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=8,
                    ffmpeg_params=["-crf", "20", "-preset", "veryfast"],
                )
            except Exception as e:  # noqa: BLE001
                self.video_error = repr(e)
                log.warning("video writer unavailable (%r); recording without video", e)

    # ------------------------------------------------------------------ per step

    def frame_from_obs(self, obs) -> Optional[np.ndarray]:
        try:
            return _to_uint8_rgb(obs[self.robot.name][self.cam_key]["rgb"])
        except Exception:  # noqa: BLE001
            return None

    def record(self, step: int, action: th.Tensor, obs, info: dict, phase: int) -> None:
        r = self.rows
        q = self.robot.get_joint_positions()
        dq = self.robot.get_joint_velocities()
        r["step"].append(step)
        r["sim_time"].append(float(og.sim.current_time))
        r["phase"].append(phase)
        r["action"].append(as_tensor(action).numpy().astype(np.float32))
        r["joint_positions"].append(q.detach().cpu().numpy().astype(np.float32))
        r["joint_velocities"].append(dq.detach().cpu().numpy().astype(np.float32))
        r["base_pose"].append(_pose7(*self.robot.get_position_orientation()))
        for arm in M.ARMS:
            r[f"eef_pose_{arm}"].append(_pose7(*self.robot.eef_links[arm].get_position_orientation()))
        r["head_camera_pose"].append(_pose7(*self.robot.sensors[self.cam_key].get_position_orientation()))
        for inst, obj in self.objects.items():
            r[f"object_pose/{inst}"].append(_pose7(*obj.get_position_orientation()))
        isg = (info.get("done") or {}).get("ispatialgym") or {}
        r["eval/binary_success"].append(bool(isg.get("binary_success", False)))
        r["eval/settled"].append(bool(isg.get("settled", False)))
        errs = isg.get("errors") or {}
        for key in ("max_position_error", "max_orientation_error", "max_layout_position_error", "max_layout_orientation_error"):
            v = errs.get(key)
            r[f"eval/{key}"].append(float("nan") if v is None else float(v))
        self.n += 1
        frame = self.frame_from_obs(obs)
        if frame is not None:
            self.last_rgb = frame
            if self.writer is not None:
                try:
                    if self.video_size and frame.shape[0] != self.video_size:
                        from PIL import Image

                        frame = np.asarray(Image.fromarray(frame).resize((self.video_size, self.video_size), Image.BILINEAR))
                    self.writer.append_data(frame)
                except Exception as e:  # noqa: BLE001
                    self.video_error = repr(e)
                    log.warning("video frame dropped (%r); closing the writer", e)
                    self._close_writer()

    # ------------------------------------------------------------------ finish

    def _close_writer(self):
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:  # noqa: BLE001
                pass
            self.writer = None

    def close(self, meta: dict) -> Optional[str]:
        """Close the video and write ``demo.hdf5``; returns the hdf5 path (None when nothing was recorded)."""
        self._close_writer()
        if self.n == 0:
            return None
        import h5py

        path = os.path.join(self.out_dir, "demo.hdf5")
        with h5py.File(path, "w") as f:
            f.attrs["meta"] = json.dumps(meta, default=_json_default)
            f.attrs["joint_names"] = json.dumps(list(self.robot.joints.keys()))
            f.attrs["controller_order"] = json.dumps(M.CONTROLLER_ORDER)
            f.attrs["controller_dims"] = json.dumps(M.CONTROLLER_DIMS)
            f.attrs["pose_layout"] = "pos xyz, quat xyzw"
            f.attrs["phase_codes"] = json.dumps({"oracle": PHASE_ORACLE, "settle": PHASE_SETTLE})
            f.attrs["fps"] = self.fps
            f.attrs["robot_name"] = self.robot.name
            f.attrs["head_camera_key"] = self.cam_key
            f.attrs["n_steps"] = self.n
            if self.video_path and os.path.exists(self.video_path) and self.video_error is None:
                f.attrs["head_rgb_video"] = os.path.basename(self.video_path)
            for key, rows in self.rows.items():
                arr = np.asarray(rows)
                if arr.dtype == object:
                    arr = np.stack(rows)
                f.create_dataset(key, data=arr, compression="gzip" if arr.ndim > 1 else None)
        return path


# ==========================================================================================
# executor
# ==========================================================================================


class Executor:
    """Steps the environment with the actions an oracle generator yields; records and monitors every step."""

    def __init__(self, env, recorder: Recorder, collision: Optional[CollisionMonitor], max_steps: int, collision_policy: str):
        self.env = env
        self.recorder = recorder
        self.collision = collision
        self.max_steps = int(max_steps)
        self.collision_policy = collision_policy
        self.n = 0
        self.last_info: dict = {}
        self.last_obs = None
        self.phase_steps: Counter = Counter()

    def step(self, action: th.Tensor, phase: int) -> dict:
        obs, _reward, _term, _trunc, info = self.env.step(action)
        self.n += 1
        self.phase_steps[phase] += 1
        self.last_info, self.last_obs = info, obs
        self.recorder.record(self.n, action, obs, info, phase)
        if self.collision is not None:
            hits = self.collision.check(self.n)
            if hits and self.collision_policy == "abort":
                raise OracleFailure("collision", f"contact with {sorted(set(hits))} at step {self.n}", objects=sorted(set(hits)))
        if self.n >= self.max_steps:
            raise OracleFailure("timeout", f"episode exceeded --max-steps {self.max_steps}")
        return info

    def run(self, gen: Iterator[th.Tensor], phase: int = PHASE_ORACLE) -> int:
        n0 = self.n
        for action in gen:
            self.step(action, phase)
        return self.n - n0

    @property
    def isg(self) -> dict:
        return ((self.last_info or {}).get("done") or {}).get("ispatialgym") or {}


# ==========================================================================================
# oracle drivers
# ==========================================================================================


def _target_instance(tro: TROFile) -> str:
    """The (single) non-agent pose target of a held-object instance."""
    insts = [k for k in tro.meta.pose_targets if k != AGENT_INSTANCE]
    if len(insts) != 1:
        raise OracleFailure("plan_failed", f"expected exactly one object pose target, got {insts}")
    return insts[0]


def _held_arm(tro: TROFile, inst: str, default: str = "right") -> str:
    for g in tro.meta.extras.get("grasps") or []:
        if g.get("object") == inst:
            return g.get("arm", default)
    return default


def _camera_depth(sensor, point: th.Tensor) -> Tuple[float, th.Tensor]:
    """Depth of a world point along the sensor's optical axis (USD: -Z) and that axis in world coordinates."""
    cp, cq = sensor.get_position_orientation()
    axis = -T.quat2mat(cq.float())[:, 2]
    return float(th.dot(as_tensor(point) - cp.float(), axis)), axis


def align_trunk_to_curve(sk: Skills, steps: int = 30, tol: float = 0.05) -> Iterator[th.Tensor]:
    """Bring an off-curve torso configuration (e.g. the all-zero torso of a camera-relocalization init) onto the
    JoyLo keyframe curve with a short linear ramp. ``Skills.set_torso`` parameterises the trunk by (translate, tilt)
    and would otherwise snap the torso onto the curve within one control step."""
    idx = sk.robot.trunk_control_idx
    q0 = sk.robot.get_joint_positions()[idx].clone()
    q1 = TrunkState.from_q(q0).q(sk._joint_limits)
    if float(th.max(th.abs(q1 - q0))) < tol:
        return
    log.info("trunk is %.2f rad off the JoyLo curve; ramping onto it over %d steps", float(th.max(th.abs(q1 - q0))), steps)
    sk.state.stop_base()
    for k in range(1, steps + 1):
        sk.state.trunk_target = q0 + (q1 - q0) * (k / steps)
        yield sk._emit()
    yield from sk._wait_converged([], trunk=True, tol=sk.cfg.joint_tol, max_steps=sk.cfg.settle_steps)
    sk.trunk = TrunkState.from_q(sk.state.trunk_target, translate_rate=sk.cfg.trunk_translate_rate, tilt_rate=sk.cfg.trunk_tilt_rate)


def drive_camera_relocalization(sk: Skills, task, tro: TROFile, ex: Executor) -> dict:
    tgt = tro.meta.pose_targets.get(AGENT_INSTANCE)
    if tgt is None or tgt.position is None or tgt.orientation is None:
        raise OracleFailure("plan_failed", "camera relocalization needs a full pose target for the agent")
    if tgt.frame != "world":
        raise OracleFailure("plan_failed", f"agent pose target frame {tgt.frame!r} is not supported (world only)")
    goal: Pose = (as_tensor(tgt.position), as_tensor(tgt.orientation))
    oracle = CameraRelocalizationOracle(sk)
    e0 = oracle.error(goal)
    n = ex.run(oracle.run(goal))
    e1 = oracle.error(goal)
    log.info("camera relocalization: %d steps, camera error %.3f m / %.3f rad -> %.3f m / %.3f rad", n, e0[0], e0[1], e1[0], e1[1])
    return {"camera_error_init": {"position_m": e0[0], "orientation_rad": e0[1]}, "camera_error_final": {"position_m": e1[0], "orientation_rad": e1[1]}}


def drive_object_scaling(sk: Skills, task, tro: TROFile, ex: Executor, cam_key: str, rounds: int = 3, depth_tol: float = 0.005) -> dict:
    inst = _target_instance(tro)
    tgt = tro.meta.pose_targets[inst]
    if tgt.position_metric != "depth" or tgt.position is None:
        raise OracleFailure("plan_failed", f"object scaling target of {inst} is not a depth target")
    if not tgt.frame.endswith(f"::{cam_key}"):
        log.warning("depth target frame %r is not the head camera %r", tgt.frame, cam_key)
    arm = _held_arm(tro, inst)
    obj = task.object_scope[inst]
    sensor = sk.robot.sensors[cam_key]
    goal_depth = -float(tgt.position[2])  # USD camera: optical axis = -Z
    oracle = ObjectScalingOracle(sk, arm=arm, pos_tol=depth_tol)
    d0, _ = _camera_depth(sensor, obj.get_position_orientation()[0])
    history = [d0]
    for k in range(rounds):
        # the evaluator measures the object *root* depth along the current camera axis: re-derive the target for the
        # oracle (which moves the AABB centre) every round, so torso fallbacks that moved the camera are compensated
        d_now, axis = _camera_depth(sensor, obj.get_position_orientation()[0])
        err = goal_depth - d_now
        log.info("object scaling: round %d, depth %.3f m (goal %.3f m, error %+.3f m)", k, d_now, goal_depth, err)
        if abs(err) <= depth_tol:
            break
        target = obj.aabb_center.float() + axis * err
        ex.run(oracle.run(target, max_corrections=1))
        history.append(_camera_depth(sensor, obj.get_position_orientation()[0])[0])
    d1, _ = _camera_depth(sensor, obj.get_position_orientation()[0])
    return {"arm": arm, "target": inst, "goal_depth_m": goal_depth, "depth_init_m": d0, "depth_final_m": d1, "depth_history_m": history}


def drive_mental_rotation(sk: Skills, task, tro: TROFile, ex: Executor, rounds: int = 2) -> dict:
    inst = _target_instance(tro)
    tgt = tro.meta.pose_targets[inst]
    if tgt.orientation is None:
        raise OracleFailure("plan_failed", f"mental rotation target of {inst} has no orientation")
    if tgt.frame != "world":
        raise OracleFailure("plan_failed", f"orientation target frame {tgt.frame!r} is not supported (world only)")
    arm = _held_arm(tro, inst)
    obj = task.object_scope[inst]
    goal = as_tensor(tgt.orientation)
    # visually equivalent goals (object-frame symmetries e: current * e ~ goal  <=>  current ~ goal * inv(e))
    candidates = [goal] + [T.quat_multiply(goal, T.quat_inverse(as_tensor(e))) for e in tgt.orientation_equivalences]
    q_now = obj.get_position_orientation()[1].float()
    goal = min(candidates, key=lambda q: quat_angle(q_now, q))
    a0 = quat_angle(q_now, goal)
    oracle = MentalRotationOracle(sk, arm=arm, ang_tol=min(0.03, 0.5 * float(tgt.orientation_tolerance or 0.15)))
    for k in range(rounds):
        ang = quat_angle(obj.get_position_orientation()[1].float(), goal)
        log.info("mental rotation: round %d, %.1f deg to go", k, math.degrees(ang))
        if ang <= oracle.ang_tol:
            break
        ex.run(oracle.run(goal, max_corrections=2))
    a1 = quat_angle(obj.get_position_orientation()[1].float(), goal)
    return {"arm": arm, "target": inst, "angle_init_deg": math.degrees(a0), "angle_final_deg": math.degrees(a1), "n_equivalent_goals": len(candidates)}


# ==========================================================================================
# main
# ==========================================================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", required=True, help="path to <iid>-tro_state.json of a sampled instance")
    p.add_argument("--mode", default="oracle", choices=["oracle"], help="only 'oracle' exists (replay / policy: TODO F1)")
    p.add_argument("--out", default=None, help="output root; the run goes to <out>/<iid>/ (default: <template_dir>/runs)")
    p.add_argument("--video", action="store_true", help="record the head camera RGB to <out>/<iid>/head_rgb.mp4")
    p.add_argument("--video-size", type=int, default=None, help="downscale video frames to this square size (default: native)")
    p.add_argument("--max-steps", type=int, default=4000, help="episode step budget (30 Hz control steps)")
    p.add_argument("--settle-steps", type=int, default=50, help="no-op window after the oracle: max(this, 20%% of the oracle steps)")
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="robot camera resolution (square)")
    p.add_argument("--head-aperture", type=float, default=DEFAULT_HEAD_APERTURE_MM,
                   help="head camera horizontal aperture in mm (the sampler measures visibility with 40; 0 = keep the robot default)")
    p.add_argument("--collision-policy", default="fail", choices=["fail", "abort", "ignore"],
                   help="contacts with scene objects: 'fail' = finish the run but report QA reason 'collision'; 'abort' = stop at the first contact; 'ignore' = record only")
    p.add_argument("--robot-name", default=None, help="robot name (default: from the TRO sensor keys, else robot_r1)")
    p.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    return p.parse_args(argv)


def _result_line(result: dict) -> None:
    print("RUN_ISG_RESULT " + json.dumps(result, default=_json_default), flush=True)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.getLogger("ispatialgym").setLevel(args.log_level)
    t_start = time.time()
    timings: Dict[str, float] = {}
    meta: Dict[str, Any] = {"status": "error", "mode": args.mode, "argv": sys.argv[1:], "hostname": os.uname().nodename,
                            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))}
    result: Dict[str, Any] = {"status": "error"}
    exit_code = EXIT_ERROR
    out_dir = None
    recorder: Optional[Recorder] = None
    env = None

    try:
        iid, template_dir, tmeta, activity, did = resolve_instance(args.instance)
        tro_path = os.path.abspath(args.instance)
        tro = load_tro(tro_path)
        out_root = args.out or os.path.join(template_dir, "runs")
        out_dir = os.path.join(out_root, iid)
        os.makedirs(out_dir, exist_ok=True)
        robot_name = args.robot_name or robot_name_from_tro(tro)
        oracle_name = ORACLES.get(activity)
        meta.update({
            "instance_id": iid, "template_dir": template_dir, "template_id": tmeta.get("template_id"), "problem": tmeta["problem"],
            "activity": activity, "activity_definition_id": did, "alias": tmeta.get("alias") or tro.meta.task_variant,
            "scene": tmeta.get("scene"), "load_room_instances": tmeta.get("load_room_instances"), "tro_path": tro_path,
            "oracle": oracle_name, "robot_name": robot_name, "out_dir": out_dir,
        })
        result.update({"instance": iid, "activity": activity, "alias": meta["alias"], "oracle": oracle_name, "out": out_dir})
        log.info("run_isg: %s (%s, oracle %s) -> %s", iid, activity, oracle_name, out_dir)

        # ---------------------------------------------------------------- environment
        for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False)):
            if getattr(gm, name) != value:
                with gm.unlocked():
                    setattr(gm, name, value)
        cfg = build_env_config(template_dir, tmeta, tro, tro_path, activity, did, robot_name, args.image_size)
        t0 = time.time()
        env = og.Environment(configs=cfg)
        from ispatialgym.sampling.scene_variants import patch_trav_map_from_meta

        meta["trav_cells_freed"] = patch_trav_map_from_meta(env.scene, tmeta)
        task = env.task
        robot = env.robots[0]
        cam_key = _first_key(robot.sensors, M.HEAD_CAMERA_SENSOR_SUFFIX)
        if cam_key is None:
            raise RuntimeError(f"robot {robot.name} has no head camera sensor ({M.HEAD_CAMERA_SENSOR_SUFFIX})")
        if args.head_aperture:
            robot.sensors[cam_key].horizontal_aperture = float(args.head_aperture)  # the sampler's camera model (visibility checks)
        meta["head_camera_key"] = cam_key
        meta["head_camera_aperture_mm"] = float(robot.sensors[cam_key].horizontal_aperture)

        # TRO -> simulator (no settle yet: the grasps must exist before the scene settles / is snapshotted)
        task.load_tro_instance(env, tro=tro, settle_steps=0)
        physical = {k: v for k, v in task.object_scope.items() if v is not None and not isinstance(v, GroupEntity)}
        grasps = restore_grasps(task, robot, tro)
        meta["grasps"] = grasps
        held = [physical[g["object"]] for g in grasps]
        hold = ActionState(robot)
        hold.hold_current()
        for g in grasps:
            if g.get("finger_q"):
                hold.gripper_cmd[g["arm"]] = gripper_command_for(robot, g["arm"], g["finger_q"])
        robot.apply_action(hold.action())  # a gripper command must exist or assisted grasping releases the object
        for _ in range(25):
            og.sim.step_physics()
            for ent in physical.values():
                ent.keep_still()
        for _ in range(2):  # full steps refresh the per-step caches (AABBs, contacts, kinematic 'changed' flags)
            og.sim.step()
            for ent in physical.values():
                ent.keep_still()
        env.scene.update_initial_file()
        task._reset_tracking()
        obs, _ = env.reset()  # returns to the snapshot, activates pose_settled tracking, renders
        timings["load_s"] = time.time() - t0
        for g in grasps:
            g["attached_after_reset"] = robot._ag_obj_in_hand.get(g["arm"]) is physical[g["object"]]
        rp, rq = robot.get_position_orientation()
        meta["robot_base_init"] = {"position": rp.tolist(), "orientation": rq.tolist(), "virtual_base_joints": robot.get_joint_positions()[robot.base_idx].tolist()}
        try:
            _save_png(os.path.join(out_dir, "init_rgb.png"), obs[robot.name][cam_key]["rgb"])
        except Exception as e:  # noqa: BLE001
            log.warning("init render not saved: %r", e)
        log.info("instance loaded in %.1fs (base %s, %d grasps)", timings["load_s"], [round(float(v), 3) for v in rp], len(grasps))

        tracked = {k: v for k, v in physical.items() if k != AGENT_INSTANCE and getattr(v, "category", None) != "floors"}
        meta["tracked_objects"] = {k: v.name for k, v in tracked.items()}

        if oracle_name is None:
            meta["status"] = "no_oracle"
            meta["reason"] = f"no v1 oracle for activity {activity!r}"
            result.update({"status": "no_oracle", "reason": meta["reason"], "success": False, "n_steps": 0})
            exit_code = EXIT_NO_ORACLE
            return exit_code

        for g in grasps:
            if not g["attached_after_reset"]:
                raise OracleFailure("failed_grasp", f"assisted grasp of {g['scene_object']} did not survive the scene reset")

        # ---------------------------------------------------------------- oracle stack
        t0 = time.time()
        sk = Skills(env, robot)
        rooms = (tro.meta.scene or {}).get("task_rooms") or tmeta.get("task_rooms") or []
        bounds = task_bounds(env.scene, rooms) if rooms else None
        sk.backend.obstacle_filter = make_obstacle_filter(bounds, robot, [o.name for o in tracked.values()])
        # load the collision world *before* attaching the held objects: ``attach_object`` disables an object's world
        # meshes only when they are already known, otherwise the first ``update_world`` (inside the first skill) would
        # load the held object as an obstacle and every start configuration would be "in collision"
        sk.backend.update_world()
        for g in grasps:
            sk.backend.attach_object(g["arm"], physical[g["object"]])
            if g.get("finger_q"):
                sk.state.gripper_cmd[g["arm"]] = gripper_command_for(robot, g["arm"], g["finger_q"])
        timings["planner_build_s"] = time.time() - t0
        meta["collision_world_rooms"] = list(rooms)
        meta["collision_world_meshes"] = len(sk.backend._world_mesh_names)

        recorder = Recorder(out_dir, env, robot, cam_key, tracked, fps=int(cfg["env"]["action_frequency"]), video=args.video, video_size=args.video_size)
        collision = CollisionMonitor(env, robot, held) if args.collision_policy != "ignore" else None
        ex = Executor(env, recorder, collision, args.max_steps, args.collision_policy)
        before = MotionSnapshot(robot, tracked)

        # ---------------------------------------------------------------- run
        qa_reason, qa_message, qa_details = None, None, {}
        t0 = time.time()
        try:
            ex.run(align_trunk_to_curve(sk))
            if activity == "camera_relocalization":
                meta["oracle_report"] = drive_camera_relocalization(sk, task, tro, ex)
            elif activity == "object_scaling":
                meta["oracle_report"] = drive_object_scaling(sk, task, tro, ex, cam_key)
            elif activity == "mental_rotation":
                meta["oracle_report"] = drive_mental_rotation(sk, task, tro, ex)
        except OracleFailure as e:
            qa_reason, qa_message, qa_details = e.reason, e.message, e.details
        except SkillError as e:
            qa_reason, qa_message, qa_details = QA_REASON.get(e.reason, e.reason), str(e), dict(e.details)
        timings["oracle_s"] = time.time() - t0
        n_oracle = ex.n
        meta["n_oracle_steps"] = n_oracle
        if qa_reason:
            log.warning("oracle failed: [%s] %s", qa_reason, qa_message)

        # epilogue: no-op window (settled success needs stillness), then the evaluator's verdict
        t0 = time.time()
        if qa_reason != "timeout":
            window = max(int(args.settle_steps), int(math.ceil(0.2 * n_oracle)))
            try:
                ex.run(sk.hold(window), phase=PHASE_SETTLE)
            except OracleFailure as e:
                if qa_reason is None:
                    qa_reason, qa_message, qa_details = e.reason, e.message, e.details
        timings["settle_s"] = time.time() - t0
        isg = ex.isg
        after = MotionSnapshot(robot, tracked)
        motion = before.delta(after)
        meta["motion_delta"] = motion
        binary_ever = bool(isg.get("binary_success_ever", False))
        success = bool(isg.get("official_success", binary_ever))
        if qa_reason is None:
            if collision is not None and collision.by_object and args.collision_policy == "fail":
                qa_reason = "collision"
                qa_message = f"contacts with {dict(collision.by_object)}"
            elif not success and not MotionSnapshot.moved(motion):
                qa_reason = "no_motion"
                qa_message = "the oracle produced no robot / object motion and the goal was not reached"
            elif not success:
                qa_reason = "eval_false"
                unsatisfied = [c["bddl"] for c in isg.get("conditions", []) if c["kind"] == "binary" and not c["satisfied"]]
                qa_message = f"binary goal not reached; unsatisfied: {unsatisfied}"
        final_errors = isg.get("final_errors") or isg.get("errors") or {}
        meta.update({
            "success": success,
            "binary_success_ever": binary_ever,
            "binary_success_final": bool(isg.get("binary_success", False)),
            "settled_success": bool(isg.get("settled_success", False)),  # at the last step
            "settled_success_ever": int(isg.get("first_settled_success_step", -1)) >= 0,
            "first_success_step": isg.get("first_success_step"),
            "first_settled_success_step": isg.get("first_settled_success_step"),
            "settle_latency_steps": isg.get("settle_latency_steps"),
            "errors": final_errors,
            "errors_final_step": isg.get("errors"),
            "conditions": isg.get("conditions"),
            "measured": isg.get("measured"),
            "n_steps": ex.n,
            "n_settle_steps": ex.phase_steps[PHASE_SETTLE],
            "qa_reason": qa_reason,
            "qa_message": qa_message,
            "qa_details": qa_details,
            "collisions": {"n_events": sum(collision.by_object.values()), "by_object": dict(collision.by_object), "events": collision.events[:50]} if collision is not None else None,
            "status": "success" if qa_reason is None else "oracle_failed",
        })
        exit_code = EXIT_SUCCESS if qa_reason is None else EXIT_ORACLE_FAILED
        if recorder.last_rgb is not None:
            _save_png(os.path.join(out_dir, "final_rgb.png"), recorder.last_rgb)
        last_errors = isg.get("errors") or {}
        result.update({
            "status": meta["status"], "success": success, "reason": qa_reason, "n_steps": ex.n, "n_oracle_steps": n_oracle,
            # errors at the first binary success (the task's representative values) and at the last step
            "max_position_error": final_errors.get("max_position_error"), "max_orientation_error": final_errors.get("max_orientation_error"),
            "final_step_max_position_error": last_errors.get("max_position_error"), "final_step_max_orientation_error": last_errors.get("max_orientation_error"),
            "settled_success": meta["settled_success"], "collisions": sum(collision.by_object.values()) if collision is not None else None,
        })
        return exit_code
    except BaseException as e:  # noqa: BLE001  (KeyboardInterrupt / SystemExit included: the verdict must be written)
        meta["status"] = "error"
        meta["error"] = repr(e)
        meta["traceback"] = traceback.format_exc()
        result.update({"status": "error", "success": False, "reason": "error", "error": repr(e)})
        traceback.print_exc()
        exit_code = EXIT_ERROR
        return exit_code
    finally:
        timings["total_s"] = time.time() - t_start
        meta["timings"] = timings
        meta["wall_time_s"] = timings["total_s"]
        result["wall_time_s"] = round(timings["total_s"], 1)
        result["exit_code"] = exit_code
        if recorder is not None:
            try:
                meta["hdf5"] = recorder.close(meta)
                meta["video"] = recorder.video_path if (recorder.video_path and recorder.video_error is None and os.path.exists(recorder.video_path)) else None
                meta["video_error"] = recorder.video_error
            except Exception as e:  # noqa: BLE001
                meta["recorder_error"] = repr(e)
                traceback.print_exc()
        if out_dir is not None:
            try:
                _dump_json(os.path.join(out_dir, "meta.json"), meta)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                result["meta_error"] = repr(e)
        _result_line(result)


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    if og.app is not None:
        try:
            og.shutdown()
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # Isaac Sim teardown must not change the exit code
