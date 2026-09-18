"""``run_isg`` -- v2 driver: load one iSpatialGym instance, run the oracle program (or a replay / a policy) through
the executor, record, evaluate (``oracle_primitives_design.md`` §3.8, §6.5; TODO F1 / F3 / F4). One process = one
instance = one episode::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.oracle.run_isg --instance <iid>-tro_state.json \\
        [--sidecar <iid>.oracle.json] --mode oracle|replay|policy [--out <dir>] [--video] [--goal-image]

What it does

1. **Loader (spec §3.14).** The template scene (``template.json``, partial rooms) with the R1Pro JoyLo
   configuration and the evaluation camera profile ``isg-v1`` (head camera 1080 x 1080, 40 mm aperture,
   ``n_render_iterations = 1``); an :class:`ISpatialGymTask` from the template's ``activity_definitions``; the TRO
   applied with ``load_tro_instance`` and then everything the task loader does not restore yet, re-applied after
   every ``scene.reset()`` (:meth:`InstanceLoader.restore_after_reset`): ``constraints`` (the COM hinges of the
   alignment tasks, re-created as PhysX revolute joints), ``grasps`` (held-object tasks: the object at
   ``object_pose_in_eef_frame``, the recorded finger joints, the assisted grasp attached before step 0 — an attach
   failure fails the load with ``failed_grasp``), ``attachment_targets`` (the child / parent collision filter, at
   load and therefore also when the policy attaches), ``articulation_targets`` (the target's initial joint values held
   during the settle), plus the trav-map patches of the scene variant (``freed_footprints`` of the template,
   ``blocked_footprints`` of the instance) on the planner map. ``pose_targets`` / ``groups`` /
   ``orientation_equivalences`` / ``evaluation_profile`` are the task's (``configure_targets``).
2. **Sidecar** (``--sidecar``, default ``<iid>.oracle.json`` next to the TRO): loaded, validated, ``tro_sha256``
   verified (``--allow-sha-mismatch`` for hand-built sidecars), ``expand(alias, sidecar)`` → the action list.
3. **Executor** (:class:`~ispatialgym.oracle.executor.Executor`): per action checkpoint → execute under the QA
   monitors → postcondition → one retry → annotate; epilogue: binary success → ``max(50, 20 %)`` no-op → settled →
   goal image (``--goal-image``: ≥ 3 renders, ``goal_observation.camera_pose_world``, profile ``isg-v1``) → TRO goal
   override with the settled values when within tolerance (written back into the TRO, the sidecar's ``tro_sha256``
   updated; ``--no-goal-override`` to disable).
4. **Recording** (``--record full|hdf5|video``): ``demo.hdf5`` (BEHAVIOR-replayable) + the LeRobot v3 dataset under
   ``lerobot/`` (``full``), ``head_rgb.mp4`` with ``--video``, ``init_rgb.png`` / ``final_rgb.png`` / ``goal_rgb.png``,
   ``annotations.json`` (the three §6.3 levels, QA verdicts) and ``meta.json``.
5. Prints one line ``RUN_ISG_RESULT {...}`` and exits with ``0`` success, ``2`` oracle / QA failure (``reason`` one
   of ``qa.CATEGORIES``), ``3`` ``no_oracle`` (no sidecar / no program for the alias; the instance is still loaded
   and its init view rendered), ``4`` ``error`` (traceback in ``meta.json``).

``--mode replay`` feeds the actions of a LeRobot episode (``--replay <dataset root> [--episode N]``) or of a
``demo.hdf5`` through the same executor; ``--mode policy`` calls ``--policy module:callable`` as
``policy(obs, info) -> 23-D action`` every step (``--policy-steps``). Both are thin: no program, no checkpoints, the
same monitors, recorder and epilogue.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import torch as th  # noqa: E402

from ispatialgym.oracle.curobo_planner import R1ProCuRobo  # noqa: E402  (import before omnigibson: warp pinning)

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402

from ispatialgym.activity.problem import AGENT_INSTANCE  # noqa: E402
from ispatialgym.eval.group_entity import GroupEntity  # noqa: E402
from ispatialgym.eval.targets import TRO_KEY  # noqa: E402
from ispatialgym.eval.tro import TROFile, load_tro  # noqa: E402
from ispatialgym.oracle import r1pro_model as M  # noqa: E402
from ispatialgym.oracle.action import ActionState  # noqa: E402
from ispatialgym.oracle.executor import Executor, ExecutorConfig, SimState, apply_goal_override  # noqa: E402
from ispatialgym.oracle.motions import Motions  # noqa: E402
from ispatialgym.oracle.programs import expand  # noqa: E402
from ispatialgym.oracle.qa import Monitors, QAConfig  # noqa: E402
from ispatialgym.oracle.recorder import EpisodeAnnotations, Recorder, RecorderConfig, dump_json, json_default, save_png  # noqa: E402
from ispatialgym.oracle.sidecar import IDDL_ROOT, Sidecar, SidecarError, family_of_subtask, sha256_of_file  # noqa: E402
from ispatialgym.robot import conventions as C  # noqa: E402
from ispatialgym.tasks import ISpatialGymTask  # noqa: E402, F401  (registers the task type)

log = logging.getLogger("ispatialgym.run_isg")

TRO_SUFFIX = "-tro_state.json"
SIDECAR_SUFFIX = ".oracle.json"
DEFAULT_IMAGE_SIZE = 1080  # evaluation camera profile (isg-v1)
DEFAULT_HEAD_APERTURE_MM = 40.0
DEFAULT_ROBOT_NAME = "robot_r1"
EVALUATION_PROFILE = "isg-v1"

EXIT_SUCCESS, EXIT_ORACLE_FAILED, EXIT_NO_ORACLE, EXIT_ERROR = 0, 2, 3, 4

#: §6.3 (b): the family-level description shared by every instance of a task family
FAMILY_DESCRIPTIONS = {
    "camera_relocalization": "Drive and set the torso so that the head camera reproduces the goal view.",
    "object_scaling": "Move the held object toward or away from the camera until it appears at the goal size.",
    "mental_rotation": "Rotate the held object in hand to the goal orientation.",
    "disembedding": "Find the target among the clutter, pick it up and place it on the destination tray.",
    "configuration_matching": "Reproduce the pictured configuration: a block layout, an articulation state, or an attachment.",
    "alignment": "Push the object until it is aligned: upright, level, or along the table axis.",
    "inter_object_relation_satisfaction": "Move the target so that it satisfies the pictured relation with the partner object.",
}


class LoadError(RuntimeError):
    """A loader-contract violation (spec §3.14); ``reason`` is a QA category (``failed_grasp``) or ``error``."""

    def __init__(self, reason: str, message: str):
        super().__init__(f"[{reason}] {message}")
        self.reason = reason
        self.message = message


# ==========================================================================================
# instance resolution / env config (from the v1 driver)
# ==========================================================================================


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


def default_sidecar_path(tro_path: str) -> str:
    return tro_path[: -len(TRO_SUFFIX)] + SIDECAR_SUFFIX


def robot_name_from_tro(tro: TROFile, default: str = DEFAULT_ROBOT_NAME) -> str:
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


def build_env_config(template_dir: str, meta: dict, tro: TROFile, tro_path: str, activity: str, did: int, robot_name: str, image_size: int, proprio: bool = True) -> dict:
    """Evaluator configuration (mirrors ``validate_instances.py``) with the robot spawned at the TRO base pose so the
    holonomic base's virtual joints stay near zero."""
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
                obs_modalities=["rgb", "proprio"] if proprio else ["rgb"],
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
            "termination_config": {"max_steps": 10**6},  # the executor owns the step budget
        },
    }


def task_bounds(scene, rooms: Sequence[str], margin: float = 1.0):
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
    """``MultiFingerGripperController`` (smooth mode) command holding the fingers at ``finger_q``."""
    idx = robot.gripper_control_idx[arm]
    lo, hi = robot.joint_lower_limits[idx], robot.joint_upper_limits[idx]
    q = th.as_tensor(finger_q, dtype=th.float32)
    return float(th.clamp(2.0 * (q.mean() - lo.mean()) / (hi.mean() - lo.mean()) - 1.0, -1.0, 1.0))


def _first_key(d: Dict[str, Any], suffix: str) -> Optional[str]:
    for k in d:
        if k.endswith(suffix):
            return k
    return None


# ==========================================================================================
# loader (spec §3.14)
# ==========================================================================================


class InstanceLoader:
    """Loads one instance and re-applies what the TRO must let ``run_isg`` restore after every reset."""

    def __init__(self, tro_path: str, robot_name: Optional[str] = None, image_size: int = DEFAULT_IMAGE_SIZE, head_aperture: float = DEFAULT_HEAD_APERTURE_MM):
        self.tro_path = os.path.abspath(tro_path)
        self.iid, self.template_dir, self.tmeta, self.activity, self.did = resolve_instance(self.tro_path)
        self.tro = load_tro(self.tro_path)
        self.robot_name = robot_name or robot_name_from_tro(self.tro)
        self.image_size = int(image_size)
        self.head_aperture = float(head_aperture)
        self.env = None
        self.task = None
        self.robot = None
        self.cam_key: Optional[str] = None
        self.physical: Dict[str, Any] = {}
        self.report: Dict[str, Any] = {}
        self.hinges: List[Any] = []
        self.grasps: List[dict] = []
        self.hold: Optional[ActionState] = None
        self._articulation_q: Dict[str, th.Tensor] = {}

    # -- environment --------------------------------------------------------------------------------------------

    def build_env(self):
        for name, value in (("ENABLE_OBJECT_STATES", True), ("USE_GPU_DYNAMICS", False), ("ENABLE_TRANSITION_RULES", False)):
            if getattr(gm, name) != value:
                with gm.unlocked():
                    setattr(gm, name, value)
        self.cfg = build_env_config(self.template_dir, self.tmeta, self.tro, self.tro_path, self.activity, self.did, self.robot_name, self.image_size)
        self.env = og.Environment(configs=self.cfg)
        from ispatialgym.sampling.scene_variants import block_trav_map_from_meta, patch_trav_map_from_meta

        # the scene variant's freed footprints (template) and the instance's blocked footprints on the planner map
        self.report["trav_cells_freed"] = patch_trav_map_from_meta(self.env.scene, self.tmeta)
        self.report["trav_cells_blocked"] = block_trav_map_from_meta(self.env.scene, self.tro.meta.scene or {})
        self.task = self.env.task
        self.robot = self.env.robots[0]
        self.cam_key = _first_key(self.robot.sensors, M.HEAD_CAMERA_SENSOR_SUFFIX)
        if self.cam_key is None:
            raise RuntimeError(f"robot {self.robot.name} has no head camera sensor ({M.HEAD_CAMERA_SENSOR_SUFFIX})")
        if self.head_aperture:
            self.robot.sensors[self.cam_key].horizontal_aperture = self.head_aperture
        self.report["head_camera_key"] = self.cam_key
        self.report["head_camera_aperture_mm"] = float(self.robot.sensors[self.cam_key].horizontal_aperture)
        self.report["evaluation_profile"] = (self.tro.meta.extras.get("evaluation_profile") or EVALUATION_PROFILE)
        return self.env

    # -- the loader contract --------------------------------------------------------------------------------------

    def restore(self, settle_physics_steps: int = 25):
        """TRO → simulator, then the §3.14 restores, a settle with the task entities held, the snapshot, one reset."""
        env, task, robot = self.env, self.task, self.robot
        task.load_tro_instance(env, tro=self.tro, settle_steps=0)
        self.physical = {k: v for k, v in task.object_scope.items() if v is not None and not isinstance(v, GroupEntity)}
        extras = self.tro.meta.extras
        self.report["constraints"] = self._restore_constraints(extras.get("constraints") or [])
        self.report["attachments"] = self._restore_attachment_filters(extras.get("attachment_targets") or [])
        self._articulation_q = {inst: self.physical[inst].get_joint_positions().clone() for inst in self.tro.meta.articulation_targets if inst in self.physical}
        self.grasps = self._restore_grasps(extras.get("grasps") or [])
        self.report["grasps"] = self.grasps
        # a gripper command must exist or assisted grasping releases the object at the first physics step
        self.hold = ActionState(robot)
        self.hold.hold_current()
        for g in self.grasps:
            if g.get("finger_q"):
                self.hold.gripper_cmd[g["arm"]] = gripper_command_for(robot, g["arm"], g["finger_q"])
        robot.apply_action(self.hold.action())
        for _ in range(settle_physics_steps):
            og.sim.step_physics()
            self._hold_entities()
        for _ in range(2):  # full steps refresh the per-step caches (AABBs, contacts, kinematic 'changed' flags)
            og.sim.step()
            self._hold_entities()
        env.scene.update_initial_file()
        task._reset_tracking()
        obs, info = env.reset()  # returns to the snapshot, activates pose_settled tracking, renders
        self.report["after_reset"] = self.restore_after_reset()
        return obs, info

    def _hold_entities(self):
        for inst, ent in self.physical.items():
            ent.keep_still()
            q0 = self._articulation_q.get(inst)
            if q0 is not None:
                ent.set_joint_positions(q0)
                ent.set_joint_velocities(th.zeros_like(q0))

    def restore_after_reset(self) -> Dict[str, Any]:
        """Re-apply the restores a ``scene.reset()`` may have undone (idempotent): hinge prims, assisted grasps, the
        attachment collision filters (persistent USD state) and the articulation joint values."""
        out: Dict[str, Any] = {"hinges_recreated": 0, "grasps": []}
        stage = og.sim.stage
        for i, h in enumerate(list(self.hinges)):
            prim = stage.GetPrimAtPath(h.prim_path)
            if prim is None or not prim.IsValid():
                rec = (self.tro.meta.extras.get("constraints") or [])[i]
                self.hinges[i] = self._create_hinge(rec)
                out["hinges_recreated"] += 1
        for g in self.grasps:
            obj = self.physical[g["object"]]
            attached = self.robot._ag_obj_in_hand.get(g["arm"]) is obj
            if not attached:
                self._establish(g["arm"], obj)
                attached = self.robot._ag_obj_in_hand.get(g["arm"]) is obj
            g["attached_after_reset"] = attached
            out["grasps"].append({"arm": g["arm"], "object": g["object"], "attached": attached})
            if not attached:
                raise LoadError("failed_grasp", f"assisted grasp of {obj.name} could not be restored after the reset")
        for inst, q0 in self._articulation_q.items():
            ent = self.physical[inst]
            q = ent.get_joint_positions()
            if float(th.max(th.abs(q - q0))) > 1e-3:
                ent.set_joint_positions(q0)
                ent.set_joint_velocities(th.zeros_like(q0))
                out.setdefault("articulation_reset", []).append(inst)
        return out

    # -- constraints (hinges) -------------------------------------------------------------------------------------

    def _create_hinge(self, rec: dict):
        from ispatialgym.sampling.constraints import create_hinge

        child = self.physical.get(rec["child"])
        if child is None:
            raise LoadError("error", f"constraint child {rec['child']!r} is not a bound physical instance")
        parent_link = None
        if rec.get("parent"):
            parent = self.physical.get(rec["parent"]) or self.env.scene.object_registry("name", rec["parent"])
            if parent is None:
                raise LoadError("error", f"constraint parent {rec['parent']!r} not found")
            link_name = rec.get("parent_link") or parent.root_link_name
            parent_link = parent.links[link_name]
        # §7 hinge friction / damping (owner 2026-09-16 22:05): the wall hinge record carries a Coulomb torque + viscous
        # damping, applied as the joint's angular drive (sampling.constraints.apply_hinge_drive); absent = frictionless (axial)
        # rush 2026-09-17 (branch rush-20260917-2100): ISG_HINGE_FRICTION_SCALE / ISG_HINGE_DAMPING_SCALE multiply the record's
        # values at load time (painting 0917-211738: push in tolerance at +0.037 rad, then the departing pads shoved it
        # to -0.39 rad - the record's resistance holds a coasting painting, not a shoved one). Default 1 = the record.
        friction, damping = rec.get("friction"), rec.get("damping")
        f_scale, d_scale = float(os.environ.get("ISG_HINGE_FRICTION_SCALE", "1")), float(os.environ.get("ISG_HINGE_DAMPING_SCALE", "1"))
        if friction is not None and f_scale != 1.0:
            friction = friction * f_scale
        if damping is not None and d_scale != 1.0:
            damping = damping * d_scale
        if f_scale != 1.0 or d_scale != 1.0:
            log.warning("hinge %s: friction x%.1f, damping x%.1f (ISG_HINGE_*_SCALE override)", rec["child"], f_scale, d_scale)
        return create_hinge(child, th.tensor(rec["anchor_position"], dtype=th.float32), th.tensor(rec["axis"], dtype=th.float32), parent_link=parent_link,
                            friction=friction, damping=damping)

    def _restore_constraints(self, records: List[dict]) -> List[dict]:
        out = []
        self.hinges = []
        for rec in records:
            if rec.get("type", "revolute") != "revolute":
                raise LoadError("error", f"unsupported constraint type {rec.get('type')!r}")
            h = self._create_hinge(rec)
            self.hinges.append(h)
            out.append({"child": rec["child"], "parent": rec.get("parent"), "prim_path": h.prim_path, "friction": h.friction, "damping": h.damping})
            log.info("hinge restored: %s about %s at %s (friction %s N·m, damping %s N·m·s/rad)", rec["child"], [round(v, 3) for v in rec["axis"]],
                     [round(v, 3) for v in rec["anchor_position"]], None if h.friction is None else round(h.friction, 4), None if h.damping is None else round(h.damping, 4))
        if self.hinges:
            og.sim.update_handles()
        return out

    # -- attachments ----------------------------------------------------------------------------------------------

    def _restore_attachment_filters(self, records: List[dict]) -> List[dict]:
        out = []
        if not records:
            return out
        from omnigibson.object_states import AttachedTo

        for rec in records:
            child, parent = self.physical.get(rec["child"]), self.physical.get(rec["parent"])
            if child is None or parent is None:
                raise LoadError("error", f"attachment_targets names unbound instances {rec}")
            if AttachedTo not in child.states:
                raise LoadError("error", f"{child.name} has no AttachedTo state (not attachable)")
            child.states[AttachedTo]._disable_collision_between_child_and_parent(child, parent)
            out.append({"child": rec["child"], "parent": rec["parent"], "child_link": rec.get("child_link"), "parent_link": rec.get("parent_link"), "collision_filtered": True})
            log.info("attachment collision filter: %s / %s", child.name, parent.name)
        return out

    # -- grasps -----------------------------------------------------------------------------------------------------

    def _establish(self, arm: str, obj) -> None:
        robot = self.robot
        link = obj.root_link_name
        joint_type = robot._get_assisted_grasp_joint_type(obj, link)
        if joint_type is None:
            raise LoadError("failed_grasp", f"{obj.name} cannot be assisted-grasped (too heavy?)")
        held = robot._ag_obj_in_hand.get(arm)
        if held is not None and held is not obj:
            robot.release_grasp_immediately(arm)
        if robot._ag_obj_in_hand.get(arm) is not obj:
            contact = robot.eef_links[arm].get_position_orientation()[0]
            robot._establish_grasp(obj, link, arm, contact, joint_type)

    def _restore_grasps(self, records: List[dict]) -> List[dict]:
        """spec §3.14: the object at ``object_pose_in_eef_frame``, the recorded finger joints, the assisted grasp
        attached before the first policy step; an attach failure fails the load. The held object is re-placed at
        ``eef ∘ object_pose_in_eef_frame`` **unconditionally** before the grasp is established: the §3.5 reload
        check (``validate_instances``) holds every object to 1 mm / 0.005 rad, so a "close enough" gate above that
        (the earlier 2 mm / 0.02 rad) would let a settle drift into the in-hand pose the sampler recorded; the
        deviation found is still reported (``object_pose_dev_before_*``)."""
        robot = self.robot
        out = []
        for g in records:
            arm, inst = g["arm"], g["object"]
            obj = self.physical.get(inst)
            if obj is None:
                raise LoadError("failed_grasp", f"grasp record names {inst!r}, which is not a bound physical instance")
            eef = tuple(x.float() for x in robot.eef_links[arm].get_position_orientation())
            ref = g.get("object_pose_in_eef_frame")
            rec = {"arm": arm, "object": inst, "scene_object": obj.name, "grasp_key": g.get("grasp_key"), "grasp_name": g.get("grasp_name"), "finger_q": g.get("finger_q")}
            if ref:
                want = C.compose(eef, (th.tensor(ref["position"], dtype=th.float32), th.tensor(ref["orientation"], dtype=th.float32)))
                have = tuple(x.float() for x in obj.get_position_orientation())
                dp, da = float(th.linalg.norm(want[0] - have[0])), C.quat_angle(want[1], have[1])
                rec["object_pose_dev_before_mm"] = 1000.0 * dp
                rec["object_pose_dev_before_deg"] = math.degrees(da)
                obj.set_position_orientation(want[0], want[1])
                obj.keep_still()
                rec["object_replaced"] = True
            if g.get("finger_q"):
                q = robot.get_joint_positions()
                q[robot.gripper_control_idx[arm]] = th.tensor(g["finger_q"], dtype=th.float32)
                robot.set_joint_positions(q)
            self._establish(arm, obj)
            rel = C.relative(tuple(x.float() for x in obj.get_position_orientation()), tuple(x.float() for x in robot.eef_links[arm].get_position_orientation()))
            if ref:
                rec["object_in_eef_dev_mm"] = 1000.0 * float(th.linalg.norm(rel[0] - th.tensor(ref["position"])))
                rec["object_in_eef_dev_deg"] = math.degrees(C.quat_angle(rel[1], th.tensor(ref["orientation"])))
            rec["attached"] = robot._ag_obj_in_hand.get(arm) is obj
            if not rec["attached"]:
                raise LoadError("failed_grasp", f"assisted grasp of {obj.name} did not attach at load")
            out.append(rec)
            log.info("grasp restored: %s in %s hand (dev %.1f mm / %.1f deg)", obj.name, arm, rec.get("object_in_eef_dev_mm", 0.0), rec.get("object_in_eef_dev_deg", 0.0))
        return out

    # -- accessors ------------------------------------------------------------------------------------------------

    @property
    def tracked(self) -> Dict[str, Any]:
        return {k: v for k, v in self.physical.items() if k != AGENT_INSTANCE and getattr(v, "category", None) != "floors"}

    @property
    def rooms(self) -> List[str]:
        return list((self.tro.meta.scene or {}).get("task_rooms") or self.tmeta.get("task_rooms") or [])


# ==========================================================================================
# annotations (§6.3 demo level)
# ==========================================================================================


def iddl_header(subtask: str) -> str:
    """The leading ``;`` comment of the subtask's iddl (the template-level description of §6.3)."""
    try:
        family = family_of_subtask(subtask)
        short = subtask.split("-", 1)[1] if "-" in subtask else subtask
        path = os.path.join(IDDL_ROOT, family, f"{short}.iddl")
        lines = []
        with open(path) as f:
            for line in f:
                if line.startswith(";"):
                    lines.append(line[1:].strip())
                elif lines:
                    break
        return " ".join(lines)
    except Exception:  # noqa: BLE001
        return ""


def episode_annotations(sidecar: Sidecar) -> EpisodeAnnotations:
    subtask = sidecar.subtask
    family = sidecar.task_family
    objects = ", ".join(sorted({o.display_name for o in sidecar.language.objects.values()})) or "no manipulated object"
    template = f"{iddl_header(subtask)} Objects: {objects}.".strip()
    numbers = ", ".join(f"{k} = {v}" for k, v in sorted(sidecar.language.goal_numbers.items()))
    instance = f"{template} Goal: {numbers}." if numbers else template
    return EpisodeAnnotations(
        family=family,
        family_description=FAMILY_DESCRIPTIONS.get(family, ""),
        template_description=template,
        instance_description=instance,
        alias=sidecar.alias,
        instance_id=sidecar.instance_id,
    )


# ==========================================================================================
# replay / policy sources
# ==========================================================================================


def replay_actions(path: str, episode: int = 0) -> Iterator[th.Tensor]:
    """Actions of a recorded episode: a LeRobot dataset root (``meta/info.json`` inside) or a ``demo.hdf5``."""
    if os.path.isdir(path):
        from lerobot.datasets import LeRobotDataset

        ds = LeRobotDataset(repo_id=os.path.basename(path.rstrip("/")), root=path)
        idx = th.nonzero(th.as_tensor(ds.hf_dataset["episode_index"]) == episode).reshape(-1).tolist()
        for i in idx:
            yield th.as_tensor(ds[i]["action"], dtype=th.float32)
        return
    import h5py

    with h5py.File(path, "r") as f:
        actions = f["data"][f"demo_{episode}"]["action"][()]
    for a in actions:
        yield th.as_tensor(a, dtype=th.float32)


def load_policy(spec: str):
    """``module:callable`` → ``policy(obs, info) -> action``."""
    mod, _, name = spec.partition(":")
    fn = getattr(importlib.import_module(mod), name or "policy")
    return fn() if isinstance(fn, type) else fn


def policy_actions(policy, executor: Executor, obs0, info0, max_steps: int) -> Iterator[th.Tensor]:
    obs, info = obs0, info0
    for _ in range(int(max_steps)):
        a = policy(obs, info)
        if a is None:
            return
        yield th.as_tensor(a, dtype=th.float32).reshape(-1)
        obs, info = executor.last_obs, executor.last_info


# ==========================================================================================
# main
# ==========================================================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", required=True, help="path to <iid>-tro_state.json of a sampled instance")
    p.add_argument("--sidecar", default=None, help="oracle sidecar (default: <iid>.oracle.json next to the TRO)")
    p.add_argument("--mode", default="oracle", choices=["oracle", "replay", "policy"])
    p.add_argument("--replay", default=None, help="replay mode: LeRobot dataset root or demo.hdf5")
    p.add_argument("--episode", type=int, default=0, help="replay mode: episode index")
    p.add_argument("--policy", default=None, help="policy mode: module:callable -> policy(obs, info) -> action")
    p.add_argument("--policy-steps", type=int, default=3000, help="policy mode: step budget")
    p.add_argument("--out", default=None, help="output root; the run goes to <out>/<iid>/ (default: <template_dir>/runs)")
    p.add_argument("--record", default="full", choices=["full", "hdf5", "video"], help="full = hdf5 + LeRobot; hdf5 = hdf5 only; video = mp4 / meta only")
    p.add_argument("--video", action="store_true", help="record the head camera RGB to <out>/<iid>/head_rgb.mp4")
    p.add_argument("--video-size", type=int, default=None, help="downscale video frames to this square size")
    p.add_argument("--goal-image", action="store_true", help="capture the goal image after the settled check (>= 3 renders) and write goal_observation into the TRO")
    p.add_argument("--no-goal-override", action="store_true", help="do not write the settled values back into the TRO goal poses")
    p.add_argument("--demo-gating", default="auto", choices=["auto", "placed_tilt", "placed_orientation", "none"],
                   help="demo-collection gate after the epilogue: placed_orientation = every placed object within its recorded orientation tolerance (full error; "
                        "an object with a continuous symmetry about its long axis per the ICL survey - marker, pencil, bottles - modulo rotation about that axis) "
                        "(auto = on for --mode oracle, off for replay / policy); placed_tilt = the retracted principal-axis tilt variant (unused by default); none")
    p.add_argument("--carry-leg", default="straight", choices=["straight", "trajopt"],
                   help="the carry leg lift end -> pre-place: straight Cartesian leg through the recorded carry waypoint (FOV rule (b), default) or the former free trajopt transit (regression reference)")
    p.add_argument("--max-steps", type=int, default=6000, help="episode step budget (30 Hz control steps)")
    p.add_argument("--settle-steps", type=int, default=50, help="no-op window after the program: max(this, 20%% of the action steps)")
    p.add_argument("--retries", type=int, default=1, help="retries per action from its checkpoint")
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--head-aperture", type=float, default=DEFAULT_HEAD_APERTURE_MM)
    p.add_argument("--robot-name", default=None)
    p.add_argument("--allow-sha-mismatch", action="store_true", help="run a sidecar whose tro_sha256 does not match the TRO (hand-built sidecars)")
    p.add_argument("--frame-buffer", default="jpeg", choices=["jpeg", "raw"], help="how pending rgb frames are buffered until an action commits")
    p.add_argument("--all-cameras", action="store_true", help="LeRobot: every rgb sensor (default: the head camera only)")
    p.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    return p.parse_args(argv)


def _result_line(result: dict) -> None:
    print("RUN_ISG_RESULT " + json.dumps(result, default=json_default), flush=True)


def write_tro_updates(tro_path: str, sidecar_path: Optional[str], sidecar: Optional[Sidecar], override: Optional[Dict[str, Any]], goal_obs: Optional[Dict[str, Any]], out_dir: str) -> Dict[str, Any]:
    """The epilogue's writes into the TRO (goal override, goal image) and the sidecar's ``tro_sha256`` refresh."""
    with open(tro_path) as f:
        raw = json.load(f)
    changed: List[str] = []
    if override is not None:
        changed += apply_goal_override(raw, override, TRO_KEY)
    if goal_obs is not None and goal_obs.get("rgb") is not None:
        png = os.path.join(out_dir, "goal_rgb.png")
        save_png(png, goal_obs["rgb"])
        block = raw[TRO_KEY].setdefault("goal_observation", {}) or {}
        block.update({
            "sensor_key": goal_obs["sensor_key"],
            "rgb_path": os.path.relpath(png, os.path.dirname(tro_path)),
            "rgb_sha256": sha256_of_file(png),
            "resolution": goal_obs["resolution"],
            "intrinsics": goal_obs.get("intrinsics"),
            "camera_pose_world": goal_obs["camera_pose_world"],
            "evaluation_profile": EVALUATION_PROFILE,
            "n_render_iterations": 1,
            "n_renders": goal_obs.get("n_renders"),
        })
        raw[TRO_KEY]["goal_observation"] = block
        changed.append("goal_observation")
    out = {"changed": changed, "tro_sha256_before": sha256_of_file(tro_path)}
    if changed:
        with open(tro_path, "w") as f:
            json.dump(raw, f, indent=2, default=json_default)
        out["tro_sha256_after"] = sha256_of_file(tro_path)
        if sidecar is not None and sidecar_path and os.path.exists(sidecar_path):
            sidecar.tro_sha256 = out["tro_sha256_after"]
            sidecar.save(sidecar_path)
            out["sidecar_updated"] = sidecar_path
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.getLogger("ispatialgym").setLevel(args.log_level)
    t_start = time.time()
    timings: Dict[str, float] = {}
    meta: Dict[str, Any] = {"status": "error", "mode": args.mode, "argv": sys.argv[1:], "hostname": os.uname().nodename,
                            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))}
    result: Dict[str, Any] = {"status": "error", "mode": args.mode}
    exit_code = EXIT_ERROR
    out_dir = None
    recorder: Optional[Recorder] = None
    executor: Optional[Executor] = None
    sidecar: Optional[Sidecar] = None
    sidecar_path: Optional[str] = None

    try:
        loader = InstanceLoader(args.instance, robot_name=args.robot_name, image_size=args.image_size, head_aperture=args.head_aperture)
        iid, tro_path = loader.iid, loader.tro_path
        out_root = args.out or os.path.join(loader.template_dir, "runs")
        out_dir = os.path.join(out_root, iid)
        os.makedirs(out_dir, exist_ok=True)
        meta.update({
            "instance_id": iid, "template_dir": loader.template_dir, "template_id": loader.tmeta.get("template_id"), "problem": loader.tmeta["problem"],
            "activity": loader.activity, "activity_definition_id": loader.did, "alias": loader.tmeta.get("alias") or loader.tro.meta.task_variant,
            "scene": loader.tmeta.get("scene"), "load_room_instances": loader.tmeta.get("load_room_instances"), "tro_path": tro_path,
            "robot_name": loader.robot_name, "out_dir": out_dir,
        })
        result.update({"instance": iid, "activity": loader.activity, "alias": meta["alias"], "out": out_dir})

        # ---------------------------------------------------------------- sidecar + program (before the sim: cheap failures first)
        actions = None
        program_error = None
        if args.mode == "oracle":
            sidecar_path = os.path.abspath(args.sidecar) if args.sidecar else default_sidecar_path(tro_path)
            if not os.path.exists(sidecar_path):
                program_error = f"no oracle sidecar at {sidecar_path}"
            else:
                sidecar = Sidecar.load(sidecar_path).validate(check_program=True)
                meta["sidecar"] = sidecar_path
                try:
                    sidecar.verify_tro(tro_path)
                    meta["tro_sha256_verified"] = True
                except SidecarError as e:
                    if not args.allow_sha_mismatch:
                        raise
                    meta["tro_sha256_verified"] = False
                    log.warning("%s (--allow-sha-mismatch)", e)
                alias = sidecar.alias
                try:
                    actions = expand(alias, sidecar)
                except SidecarError as e:
                    program_error = f"no program for {alias!r}: {e}"
                meta["subtask"] = sidecar.subtask
                meta["program"] = [a.summary() for a in actions] if actions else None
        log.info("run_isg v2: %s (%s, mode %s, %s actions) -> %s", iid, loader.activity, args.mode, None if actions is None else len(actions), out_dir)

        # ---------------------------------------------------------------- environment + loader contract
        t0 = time.time()
        env = loader.build_env()
        try:
            obs, info = loader.restore()
        except LoadError as e:
            meta.update({"status": "oracle_failed", "reason": e.reason, "load_error": e.message, "loader": loader.report})
            result.update({"status": "oracle_failed", "success": False, "reason": e.reason, "n_steps": 0, "error": e.message})
            exit_code = EXIT_ORACLE_FAILED
            return exit_code
        timings["load_s"] = time.time() - t0
        task, robot, cam_key = loader.task, loader.robot, loader.cam_key
        meta["loader"] = loader.report
        rp, rq = robot.get_position_orientation()
        meta["robot_base_init"] = {"position": rp.tolist(), "orientation": rq.tolist()}
        try:
            save_png(os.path.join(out_dir, "init_rgb.png"), obs[robot.name][cam_key]["rgb"])
        except Exception as e:  # noqa: BLE001
            log.warning("init render not saved: %r", e)
        tracked = loader.tracked
        meta["tracked_objects"] = {k: v.name for k, v in tracked.items()}
        log.info("instance loaded in %.1fs (base %s, %d grasps, %d hinges)", timings["load_s"], [round(float(v), 3) for v in rp], len(loader.grasps), len(loader.hinges))

        if args.mode == "oracle" and actions is None:
            meta.update({"status": "no_oracle", "reason": program_error})
            result.update({"status": "no_oracle", "reason": program_error, "success": False, "n_steps": 0})
            exit_code = EXIT_NO_ORACLE
            return exit_code

        # ---------------------------------------------------------------- oracle stack
        t0 = time.time()
        arms = sorted({arm for a in (actions or []) for arm in a.arms}) or ["right"]
        prebuild = tuple(f"{arm}_arm" for arm in arms)
        backend = R1ProCuRobo(robot, prebuild=prebuild)
        bounds = task_bounds(env.scene, loader.rooms) if loader.rooms else None
        backend.obstacle_filter = make_obstacle_filter(bounds, robot, [o.name for o in tracked.values()])
        # load the collision world *before* attaching the held objects (their meshes must be known to be disabled)
        backend.update_world()
        motions = Motions(env, backend)
        for g in loader.grasps:
            backend.attach_object(g["arm"], loader.physical[g["object"]])
            if g.get("finger_q"):
                motions.state.gripper_cmd[g["arm"]] = gripper_command_for(robot, g["arm"], g["finger_q"])
        timings["planner_build_s"] = time.time() - t0
        meta["collision_world_rooms"] = loader.rooms
        meta["collision_world_meshes"] = len(backend._world_mesh_names)
        meta["prebuilt_embodiments"] = list(prebuild)

        cam_in_zed = C.camera_offset_in_zed(robot, cam_key)
        meta["camera_in_zed"] = {"position": cam_in_zed[0].tolist(), "orientation": cam_in_zed[1].tolist(),
                                 "constant_dev_mm": 1000.0 * float(th.linalg.norm(cam_in_zed[0] - C.CAMERA_IN_ZED[0]))}
        state = SimState(env, task, robot, cam_key, cam_in_zed=cam_in_zed)
        monitors = Monitors(env, robot, QAConfig(max_episode_steps=args.max_steps), cam_key=cam_key, cam_in_zed=cam_in_zed)
        rec_cfg = RecorderConfig(hdf5=args.record in ("full", "hdf5"), lerobot=args.record == "full", video=args.video, video_size=args.video_size,
                                 all_cameras=args.all_cameras, frame_buffer=args.frame_buffer)
        recorder = Recorder(out_dir, env, robot, cam_key, tracked, fps=int(loader.cfg["env"]["action_frequency"]), cfg=rec_cfg,
                            task_name=loader.activity.replace("_", " "), instance_id=iid)
        if sidecar is not None:
            recorder.episode = episode_annotations(sidecar)
        if args.demo_gating == "auto":
            # the full placed_orientation gate (owner 2026-09-16 17:10 item 4: placed_tilt retracted; cylinders are compared
            # modulo rotation about their long axis inside the gate, ExecutorConfig.demo_revolution_axes)
            gating = ("placed_orientation",) if args.mode == "oracle" else ()
        else:
            gating = () if args.demo_gating == "none" else (args.demo_gating,)
        ex_cfg = ExecutorConfig(max_steps=args.max_steps, retries=args.retries, settle_min_steps=args.settle_steps, goal_image=args.goal_image,
                                record_state=rec_cfg.hdf5, demo_gating=gating)
        meta["demo_gating_enabled"] = list(gating)
        from ispatialgym.oracle import execute as _execute

        _execute.CONFIG.carry_leg = args.carry_leg
        meta["carry_leg"] = args.carry_leg
        if os.environ.get("ISG_PUSH_BACKOFF_M") is not None:  # rush 2026-09-17: 0 keeps the push pad on the target until its retreat
            _execute.CONFIG.push_backoff_m = float(os.environ["ISG_PUSH_BACKOFF_M"])
            meta["push_backoff_m"] = _execute.CONFIG.push_backoff_m
        if os.environ.get("ISG_AIM_BIAS_RAD") is not None:  # rush: the braced-push ladder's arc aim (overshoot side, rad)
            ex_cfg.brace_retry_aim_bias_rad = float(os.environ["ISG_AIM_BIAS_RAD"])
        if os.environ.get("ISG_AIM_FRAC") is not None:
            ex_cfg.brace_retry_aim_frac = float(os.environ["ISG_AIM_FRAC"])
        meta["rush_env"] = {k: v for k, v in os.environ.items() if k.startswith("ISG_")}
        executor = Executor(env, task, sidecar, recorder=recorder, motions=motions, monitors=monitors, cfg=ex_cfg, sim=og.sim, state=state,
                            tracked_objects=tracked, cam_key=cam_key)
        executor.last_obs, executor.last_info = obs, info
        if rec_cfg.hdf5:
            executor.enable_recording_filter(True)
        recorder.start(obs, info, state0=executor._dump_state(full=True))

        # ---------------------------------------------------------------- run
        t0 = time.time()
        if args.mode == "oracle":
            episode = executor.run_program(actions)
        elif args.mode == "replay":
            if not args.replay:
                raise ValueError("--mode replay needs --replay <dataset root | demo.hdf5>")
            episode = executor.run_stream(replay_actions(args.replay, args.episode))
        else:
            if not args.policy:
                raise ValueError("--mode policy needs --policy module:callable")
            policy = load_policy(args.policy)
            episode = executor.run_stream(policy_actions(policy, executor, obs, info, args.policy_steps))
        timings["episode_s"] = time.time() - t0
        timings.update({f"executor_{k}": v for k, v in episode.get("timings", {}).items()})

        # ---------------------------------------------------------------- epilogue writes (goal override, goal image)
        goal_obs = episode.get("goal_observation")
        override = None
        if episode["success"] and not args.no_goal_override:
            try:
                override = executor.goal_override(loader.tro.meta, task.compiled_task)
            except Exception as e:  # noqa: BLE001
                log.warning("goal override not computed: %r", e)
                meta["goal_override_error"] = repr(e)
        if episode["success"] and (override is not None or (goal_obs and goal_obs.get("rgb") is not None)):
            meta["tro_updates"] = write_tro_updates(tro_path, sidecar_path, sidecar, override, goal_obs if args.goal_image else None, out_dir)
        if goal_obs is not None:
            goal_obs = {k: v for k, v in goal_obs.items() if k != "rgb"}
        meta["goal_observation"] = goal_obs
        meta["goal_override"] = None if override is None else {k: v for k, v in override.items()}
        if recorder.last_rgb is not None:
            save_png(os.path.join(out_dir, "final_rgb.png"), recorder.last_rgb)

        success = bool(episode["success"])
        reason = episode.get("reason")
        errors = episode.get("errors") or {}
        last_errors = episode.get("errors_final_step") or {}
        meta.update({k: v for k, v in episode.items() if k not in ("goal_observation", "timings")})
        meta["status"] = "success" if success else "oracle_failed"
        meta["qa_reason"] = reason
        exit_code = EXIT_SUCCESS if success else EXIT_ORACLE_FAILED
        result.update({
            "status": meta["status"], "success": success, "reason": reason, "n_steps": episode["n_steps"], "n_action_steps": episode["n_action_steps"],
            "max_position_error": errors.get("max_position_error"), "max_orientation_error": errors.get("max_orientation_error"),
            "final_step_max_position_error": last_errors.get("max_position_error"), "final_step_max_orientation_error": last_errors.get("max_orientation_error"),
            "settled_success": episode.get("settled_success"), "binary_branch": episode.get("binary_branch"), "collisions": (episode.get("qa") or {}).get("collisions", {}).get("n_events"),
            "qa_by_category": (episode.get("qa") or {}).get("stats", {}).get("by_category"), "goal_override": None if override is None else override.get("applied"),
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
                qa = (executor.result.get("qa") or {}).get("stats") if executor is not None and executor.result else None
                paths = recorder.close(meta, qa=qa, success=bool(result.get("success")))
                meta.update({"hdf5": paths.get("hdf5"), "video": paths.get("video"), "lerobot": paths.get("lerobot"), "annotations": paths.get("annotations"),
                             "video_error": paths.get("video_error"), "lerobot_error": paths.get("lerobot_error")})
            except Exception as e:  # noqa: BLE001
                meta["recorder_error"] = repr(e)
                traceback.print_exc()
        if out_dir is not None:
            try:
                dump_json(os.path.join(out_dir, "meta.json"), meta)
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
