"""Simulator-free tests of the v2 oracle executor stack (``oracle_primitives_design.md`` §3.8, §6.4, §6.5; TODO
F1-F4): the Layer-1 motions against a toy R1Pro, the execution table, the executor loop (checkpoint – retry – fail,
QA verdict propagation, epilogue ordering), the checkpoint round-trip fix-up, the recorder schema and the TRO goal
override. ``pytest ispatialgym/tests/test_v2_oracle.py``.

The fakes model exactly what the executor touches: a 28-D R1Pro whose position-controlled joints follow their
targets within one step, a holonomic base integrating the normalised velocity command, a toy arm kinematics (three
joints = base-relative eef position / 0.5, four joints = eef quaternion) the fake cuRobo backend inverts, an
assisted grasp that engages when the closed gripper is near an object, and a simulator whose serialized state
round-trips the AG bookkeeping (or, with ``ag_roundtrip=False``, does not — the executor must then fix it up).
"""

from __future__ import annotations

import json
import glob
import math
from types import SimpleNamespace

import h5py
import os
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch as th

from ispatialgym.eval import metrics as MET
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import qa as QA
from ispatialgym.oracle.actions import VOCABULARY, Navigate, Push, PushClose, BaseFollow, Place, Reorient, Brace as QA_Brace
from ispatialgym.oracle.executor import AGENT_INSTANCE, Executor, ExecutorConfig, SimState, apply_goal_override
from ispatialgym.oracle.motions import Motions, MotionsConfig, SkillError, as_world_pose
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.recorder import PHASE_ORACLE, PHASE_SETTLE, Recorder, RecorderConfig, sensor_name
from ispatialgym.oracle.sidecar import Sidecar
from ispatialgym.robot import conventions as C

SIDECARS = os.path.join(os.path.dirname(__file__), "data", "sidecars")
VARIANTS = os.path.join(os.path.dirname(__file__), "data", "sidecar_variants")
CAM_KEY = "robot_r1:zed_link:Camera:0"
DT = 1.0 / 30.0
ARM_OFFSET = {"left": th.tensor([0.3, 0.2, 0.5]), "right": th.tensor([0.3, -0.2, 0.5])}
ARM_SCALE = 0.5


# ==========================================================================================
# fakes
# ==========================================================================================


def _quat_norm(q: th.Tensor) -> th.Tensor:
    n = float(th.linalg.norm(q))
    return C.IDENTITY_QUAT.clone() if n < 1e-6 else q / n


class FakeLink:
    def __init__(self, prim_path, pose_fn):
        self.prim_path = prim_path
        self._pose_fn = pose_fn
        self.body_name = prim_path.rsplit("/", 1)[1]

    def get_position_orientation(self):
        p, q = self._pose_fn()
        return p.clone(), q.clone()


class FakeObject:
    def __init__(self, name, pos, quat=None, category="apple", joints=None, extent=(0.08, 0.08, 0.08)):
        self.name = name
        self.category = category
        self.prim_path = f"/World/{name}"
        self.pos = th.tensor(pos, dtype=th.float32)
        self.quat = C.IDENTITY_QUAT.clone() if quat is None else th.tensor(quat, dtype=th.float32)
        self.joints = OrderedDict((j, None) for j in (joints or []))
        self.q = th.zeros(len(self.joints))
        self.links = {"base_link": FakeLink(f"{self.prim_path}/base_link", lambda: (self.pos, self.quat))}
        self.root_link_name = "base_link"
        self.aabb_extent = th.tensor(extent)
        self.woken = 0
        self.states = {}
        self.vel = th.zeros(3)

    def get_position_orientation(self):
        return self.pos.clone(), self.quat.clone()

    def set_position_orientation(self, pos, quat):
        self.pos, self.quat = th.as_tensor(pos).float().clone(), th.as_tensor(quat).float().clone()

    def get_linear_velocity(self):
        return self.vel.clone()

    def get_joint_positions(self):
        return self.q.clone()

    def wake(self):
        self.woken += 1


class FakeSensor:
    image_width = 1080
    image_height = 1080
    focal_length = 17.0
    horizontal_aperture = 40.0

    def __init__(self, robot):
        self.robot = robot

    def get_position_orientation(self):
        q = self.robot.q[self.robot.trunk_control_idx]
        p, o = C.camera_pose(q, base_pose=(self.robot.pos, self.robot.quat))
        return p.clone(), o.clone()

    def get_obs(self):
        """The annotator buffer as it stands (``VisionSensor.get_obs``: no render happens on a read)."""
        env = getattr(self.robot, "_env", None)
        return ({"rgb": env._frame.clone()} if env is not None else {}), {}


class FakeRobot:
    name = "robot_r1"
    prim_path = "/World/robot_r1"
    model = "r1pro"
    action_dim = M.ACTION_DIM
    floor_touching_base_link_names: List[str] = []

    def __init__(self, base=(0.0, 0.0, 0.0)):
        self.joints = OrderedDict((n, None) for n in M.ALL_JOINTS)
        idx = 0
        self.controller_action_idx = {}
        for k in M.CONTROLLER_ORDER:
            n = M.CONTROLLER_DIMS[k]
            self.controller_action_idx[k] = th.arange(idx, idx + n)
            idx += n
        j = {n: i for i, n in enumerate(M.ALL_JOINTS)}
        self.base_idx = th.tensor([j[n] for n in M.BASE_VIRTUAL_JOINTS])
        self.trunk_control_idx = th.tensor([j[n] for n in M.TORSO_JOINTS])
        self.arm_control_idx = {a: th.tensor([j[n] for n in M.ARM_JOINTS[a]]) for a in M.ARMS}
        self.gripper_control_idx = {a: th.tensor([j[n] for n in M.FINGER_JOINTS[a]]) for a in M.ARMS}
        lo, hi = th.full((28,), -3.14), th.full((28,), 3.14)
        lo[self.base_idx], hi[self.base_idx] = -100.0, 100.0
        lo[self.trunk_control_idx], hi[self.trunk_control_idx] = C.TORSO_JOINT_LIMITS[0], C.TORSO_JOINT_LIMITS[1]
        for a in M.ARMS:
            lo[self.gripper_control_idx[a]], hi[self.gripper_control_idx[a]] = 0.0, 0.05
        self.joint_lower_limits, self.joint_upper_limits = lo, hi
        self.q = th.tensor(M.JOYLO_RESET_JOINT_POS, dtype=th.float32)
        self.q[self.trunk_control_idx] = C.torso_q(0.5, 0.0)
        for a in M.ARMS:
            self.q[self.arm_control_idx[a]] = th.tensor([0.3, 0.0, 0.6, 0.0, 0.0, 0.0, 1.0])
        self.dq = th.zeros(28)
        x, y, yaw = base
        self.pos = th.tensor([x, y, C.NOMINAL_BASE_Z], dtype=th.float32)
        self.quat = C.quat_from_yaw(yaw)
        self._ag_obj_in_hand = {a: None for a in M.ARMS}
        self._ag_obj_constraint_params = {a: None for a in M.ARMS}
        self._ag_release_counter = {a: None for a in M.ARMS}
        self._ag_grasp_counter = {a: None for a in M.ARMS}
        self._ag_offset: Dict[str, Tuple[th.Tensor, th.Tensor]] = {}
        self.eef_links = {a: FakeLink(f"{self.prim_path}/{M.EEF_LINK[a]}", lambda a=a: self.eef_pose(a)) for a in M.ARMS}
        self.links = {
            "base_link": FakeLink(f"{self.prim_path}/base_link", lambda: (self.pos, self.quat)),
            "zed_link": FakeLink(f"{self.prim_path}/zed_link", lambda: self.sensors[CAM_KEY].get_position_orientation()),
        }
        for a in M.ARMS:
            for l in M.FINGER_LINKS[a] + [M.GRIPPER_LINK[a]]:
                self.links[l] = FakeLink(f"{self.prim_path}/{l}", lambda a=a: self.eef_pose(a))
        self.sensors = {CAM_KEY: FakeSensor(self)}
        self.untucked_default_joint_pos = self.q.clone()
        for a in M.ARMS:
            self.untucked_default_joint_pos[self.arm_control_idx[a]] = th.tensor([0.8, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0])
        self.ag_fail_next = 0  # the fake assisted grasp refuses this many closures
        self.releases = 0

    # -- toy kinematics --------------------------------------------------------------------------------------------

    def eef_pose(self, arm):
        qa = self.q[self.arm_control_idx[arm]]
        rel = ARM_OFFSET[arm] + ARM_SCALE * qa[:3]
        return self.pos + C.quat_rotate(self.quat, rel), C.qmul(self.quat, _quat_norm(qa[3:7]))

    def ik(self, arm, pose) -> th.Tensor:
        p, q = as_world_pose(pose)
        rel = C.quat_rotate(C.qinv(self.quat), p - self.pos)
        qrel = C.qmul(C.qinv(self.quat), q)
        return th.cat([(rel - ARM_OFFSET[arm]) / ARM_SCALE, qrel])

    # -- OmniGibson-like API -----------------------------------------------------------------------------------------

    def _calculate_in_hand_object(self, arm="right"):
        """OmniGibson's in-hand rule as the fake knows it: both pads touch an object when the finger opening is down to the
        object's half width (+2 mm) and the object lies within 15 cm of the eef; returns ``(object, link_name)`` or None."""
        env = getattr(self, "_env", None)
        if env is None:
            return None
        ep, _ = self.eef_pose(arm)
        opening = float(self.q[self.gripper_control_idx[arm]].max())
        for o in env.objects.values():
            half = 0.5 * float(min(o.aabb_extent[0], o.aabb_extent[1])) if getattr(o, "aabb_extent", None) is not None else 0.03
            if getattr(o, "graspable", True) and float(th.linalg.norm(o.pos - ep)) < 0.15 and opening <= half + 0.002:
                return o, "base_link"
        return None

    def get_joint_positions(self):
        return self.q.clone()

    def get_joint_velocities(self):
        return self.dq.clone()

    def get_position_orientation(self):
        return self.pos.clone(), self.quat.clone()

    def release_grasp_immediately(self, arm):
        self._ag_obj_in_hand[arm] = None
        self._ag_obj_constraint_params[arm] = None
        self._ag_release_counter[arm] = None
        self.releases += 1

    def _create_assisted_grasp_joint(self, arm, params):
        self._ag_obj_in_hand[arm] = params["target_obj"]
        self._ag_obj_constraint_params[arm] = params
        op, oq = params["target_obj"].get_position_orientation()
        self._ag_offset[arm] = C.relative((op, oq), self.eef_pose(arm))


@dataclass
class _Res:
    success: bool
    q_traj: Optional[th.Tensor] = None
    status: str = "ok"
    plan_time: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    @property
    def duration(self):
        return 0.0 if self.q_traj is None else (len(self.q_traj) - 1) * DT


class FakeBackend:
    control_dt = DT

    def __init__(self, robot):
        self.robot = robot
        self._jidx = {n: i for i, n in enumerate(M.ALL_JOINTS)}
        self._attached: Dict[str, dict] = {}
        self.fail_plans = 0  # the next N plan_pose / plan_joint calls fail
        self.fail_ik_paths = 0
        self.world_updates = 0
        self.calls: List[str] = []

    def q_full(self):
        return self.robot.get_joint_positions()

    def update_world(self, ignore_objects=()):
        self.world_updates += 1

    def attach_object(self, arm, obj):
        self._attached[arm] = {"obj": obj, "mesh_names": [], "planners": [], "link": M.ATTACHED_OBJECT_LINK[arm]}

    def detach_object(self, arm):
        self._attached.pop(arm, None)

    def _arms_of(self, emb):
        return list(M.ARMS) if emb == "both_arms" else [emb.split("_")[0]]

    def plan_pose(self, emb, targets, q_start=None, max_attempts=4, sync_world=True):
        self.calls.append("plan_pose")
        if self.fail_plans > 0:
            self.fail_plans -= 1
            return _Res(False, status="IK failed: goal pose blocked (fake)")
        q0 = self.q_full() if q_start is None else q_start.clone()
        q1 = q0.clone()
        for frame, pose in targets.items():
            arm = M.arm_of_eef_link(frame)
            q1[self.robot.arm_control_idx[arm]] = self.robot.ik(arm, pose)
        traj = th.stack([q0 + (q1 - q0) * (k / 9) for k in range(10)])
        return _Res(True, traj)

    def plan_joint(self, emb, q_goal, q_start=None, max_attempts=4, sync_world=True):
        self.calls.append("plan_joint")
        if self.fail_plans > 0:
            self.fail_plans -= 1
            return _Res(False, status="trajopt to the joint goal failed (fake)")
        q0 = self.q_full() if q_start is None else q_start.clone()
        q1 = q0.clone()
        for arm in self._arms_of(emb):
            q1[self.robot.arm_control_idx[arm]] = q_goal[self.robot.arm_control_idx[arm]]
        return _Res(True, th.stack([q0 + (q1 - q0) * (k / 9) for k in range(10)]))

    def solve_ik_path(self, emb, waypoints, q_start=None, pos_tol=2e-3, rot_tol=1e-2, check_collisions=True, sync_world=True, **kw):
        self.calls.append("solve_ik_path")
        if self.fail_ik_paths > 0:
            self.fail_ik_paths -= 1
            return _Res(False, status="differential IK did not converge at waypoint 0 (fake)")
        q = self.q_full() if q_start is None else q_start.clone()
        path = [q.clone()]
        stall = getattr(self, "stall_ik_after", None)  # (n converged waypoints, pos_err): the real solver's stall shape
        for i, wp in enumerate(waypoints):
            if stall is not None and i >= stall[0] and len(waypoints) > 1:
                self.stall_ik_after = None
                diag = {"stage": "diff_ik", "fail_index": i, "n_waypoints": len(waypoints), "pos_err": stall[1], "rot_err": 0.001,
                        "at_lower": list(stall[2]) if len(stall) > 2 else [], "at_upper": [], "converged": i}  # (n, pos_err[, joints at their lower limit])
                return _Res(False, th.stack(path) if len(path) > 1 else None, status=f"differential IK did not converge at waypoint {i}/{len(waypoints)}: pos_err={stall[1]*1000:.1f}mm (fake)", diagnostics=diag)
            for frame, pose in wp.items():
                arm = M.arm_of_eef_link(frame)
                q = q.clone()
                q[self.robot.arm_control_idx[arm]] = self.robot.ik(arm, pose)
            path.append(q.clone())
        return _Res(True, th.stack(path))


class FakeRegistry:
    def __init__(self, objects: Dict[str, Any]):
        self.objects = objects
        self.filters = 0

    def __call__(self, key, value, default=None):
        for o in self.objects.values():
            if getattr(o, key, None) == value:
                return o
        return default

    def set_dump_filter(self, fn):
        self.filters += 1


class FakeScene:
    idx = 0

    def __init__(self, objects: Dict[str, Any], robot):
        self.robots = [robot]
        self.objects_by_name = {o.name: o for o in objects.values()}
        self.objects_by_name[robot.name] = robot
        self.object_registry = FakeRegistry(self.objects_by_name)

    def save(self):
        return {"state": {}, "objects_info": {}}


class FakeTask:
    """The task's scoring (``ispatialgym.eval.metrics``) over the fake world's per-step facts; ``metrics_state`` is what a
    rollback restores (``executor.TASK_TRACKING_ATTRS``)."""

    def __init__(self, objects: Dict[str, Any], robot):
        self.object_scope = dict(objects)
        self.object_scope[AGENT_INSTANCE] = robot
        self.compiled_task = None
        self.tracker = MET.SuccessTracker(settle_eval=True, settle_steps=45)

    @property
    def metrics_state(self):
        return self.tracker.state()

    @metrics_state.setter
    def metrics_state(self, state):
        self.tracker.load(state)

    def update_bddl_scope_metadata(self, env):
        pass


class FakeSim:
    """Serialized state: robot q (28) + base (7) + per object pose (7) + per arm held-object index (1)."""

    def __init__(self, env, ag_roundtrip: bool = True):
        self.env = env
        self.ag_roundtrip = ag_roundtrip
        self.renders = 0
        self.current_time = 0.0

    def dump_state(self, serialized=True):
        r = self.env.robot
        parts = [r.q.clone(), r.pos.clone(), r.quat.clone()]
        for o in self.env.objects.values():
            parts += [o.pos.clone(), o.quat.clone()]
        names = list(self.env.objects.values())
        for a in M.ARMS:
            held = r._ag_obj_in_hand[a]
            parts.append(th.tensor([float(names.index(held)) if held is not None else -1.0]))
        return th.cat(parts)

    def load_state(self, state, serialized=True):
        r = self.env.robot
        r.q = state[:28].clone()
        r.pos, r.quat = state[28:31].clone(), state[31:35].clone()
        i = 35
        for o in self.env.objects.values():
            o.pos, o.quat = state[i : i + 3].clone(), state[i + 3 : i + 7].clone()
            i += 7
        names = list(self.env.objects.values())
        for a in M.ARMS:
            k = int(state[i].item())
            i += 1
            if self.ag_roundtrip:
                if k < 0:
                    r._ag_obj_in_hand[a] = None
                    r._ag_obj_constraint_params[a] = None
                else:
                    obj = names[k]
                    if r._ag_obj_in_hand[a] is not obj:
                        r._create_assisted_grasp_joint(a, {"target_obj": obj, "target_link_name": "base_link", "joint_type": "FixedJoint"})
        r.dq = th.zeros(28)

    def render(self):
        self.renders += 1
        self.env._render_once()


class FakeEnv:
    def __init__(self, robot: FakeRobot, objects: Dict[str, FakeObject], evaluate=None, ag_roundtrip=True, render_lag: int = 1):
        self.robot = robot
        self.robots = [robot]
        self.objects = objects
        self.scene = FakeScene(objects, robot)
        self.task = FakeTask(objects, robot)
        self.sim = FakeSim(self, ag_roundtrip=ag_roundtrip)
        self.evaluate = evaluate or (lambda env: {"binary_success": False, "settled": False})
        self._current_step = 0
        self.config = {"env": {"action_frequency": 30}}
        self.steps = 0
        self.last_action = None
        self._closing = {a: False for a in M.ARMS}
        self._refuse = {a: False for a in M.ARMS}
        robot._env = self  # the fake robot's in-hand rule (``_calculate_in_hand_object``) reads the env's objects
        # the fake renderer: like the simulator, a frame shows the scene the renderer synced ``render_lag`` renders ago
        # (a pipeline: the pose -> renderer sync lags one frame, the annotator buffer a read returns trails the last
        # render by a further, load-dependent 1-2 frames; after ``load_state`` the first draws still show the
        # pre-restore scene). ``render_lag=1`` is the light-load simulator, 4 the loaded one (sim_utils.seg_instance).
        self.render_lag = int(render_lag)
        self._pipeline: List[th.Tensor] = [self._render_state() for _ in range(self.render_lag)]
        self._frame = self._image_of(self._pipeline[0])

    def _render_state(self):
        """What the renderer would draw: the robot joints + every object pose (a scalar signature per pixel row)."""
        parts = [self.robot.q.clone(), self.robot.pos.clone()]
        for o in self.objects.values():
            parts += [o.pos.clone(), o.quat.clone()]
        return th.cat(parts)

    @staticmethod
    def _image_of(state):
        """An 8x8x4 uint8 'image' encoding the state (so a frame can be traced back to the state it shows)."""
        vals = th.remainder((state.float() * 1000.0).round(), 251.0).to(th.uint8)
        rgb = th.zeros(8 * 8 * 3, dtype=th.uint8)
        n = min(vals.numel(), rgb.numel())
        rgb[:n] = vals[:n]
        img = th.full((8, 8, 4), 255, dtype=th.uint8)
        img[..., :3] = rgb.view(8, 8, 3)
        return img

    def _render_once(self):
        self._frame = self._image_of(self._pipeline.pop(0))  # draws the oldest synced scene of the pipeline ...
        self._pipeline.append(self._render_state())  # ... then syncs the current physics state for a later draw

    def frame_matches_state(self, rgb, state=None) -> bool:
        """Whether an observed frame (rgb or rgba) shows ``state`` (default: the current physics state)."""
        want = self._image_of(self._render_state() if state is None else state)[..., :3]
        return bool(th.equal(th.as_tensor(np.asarray(rgb))[..., :3], want))

    def _obs(self):
        return {self.robot.name: {CAM_KEY: {"rgb": self._frame.clone()}, "proprio": self.robot.q.clone()}}

    def _info(self):
        """The task's ``info["ispatialgym"]`` of this step from ``evaluate``'s facts: ``goal`` (G; or ``binary_success``),
        ``settled`` (S), ``measure_within`` (M; default every reported target within tolerance) and ``errors`` (each target a
        ``(:measure ...)`` atom), scored by the fake task's tracker once per episode step."""
        d = self.evaluate(self)
        errors = d.get("errors") or {}
        goal = bool(d.get("goal", d.get("binary_success", False)))
        reps = [r for k in ("pose_errors", "layout_errors") for r in (errors.get(k) or {}).values() if isinstance(r, dict)]
        within = bool(d.get("measure_within", all(r.get("within_tolerance", True) for r in reps)))
        settled = bool(d.get("settled", False))
        tr = self.task.tracker
        if self._current_step > 0 and (tr.last is None or tr.last[0] != self._current_step):
            tr.update(self._current_step, goal, within, settled, errors)
        atoms = [("pose_reached", k) for k in (errors.get("pose_errors") or {})] + [("layout_reached", k) for k in (errors.get("layout_errors") or {})]
        try:
            rep = tr.report(errors, atoms)
        except MET.ErrorMetricError:
            rep = tr.report(errors, [])

        def measured(errs):
            return [{"bddl": f"({p} ?{a})", "kind": "measure", "within_tolerance": ((errs.get("pose_errors" if p == "pose_reached" else "layout_errors") or {})
                                                                                   .get(a) or {}).get("within_tolerance")} for p, a in atoms]

        isg = {**d, "goal": goal, "measure_within": within, "settled": settled, "settle_eval": rep["settle_eval"], "settle_steps": rep["settle_steps"],
               "official_success": rep["geometric_success"], "success_final": rep["success_final"],
               "symbolic_success": rep["symbolic_success"], "symbolic_success_step": rep["symbolic_success_step"],
               "symbolic_first_step": rep["symbolic_first_step"], "geometric_first_step": rep["geometric_first_step"],
               "closest_approach": rep["closest_approach"], "closest_goal_approach": rep["closest_goal_approach"],
               "geometric_success": rep["geometric_success"], "geometric_success_step": rep["geometric_success_step"], "goal_first_step": rep["goal_first_step"],
               "other_mode": rep["other_mode"], "evaluation": {"errors": rep["errors"], "error_metric": rep["error_metric"], "measured": measured(rep["errors"]),
                                                     "step": rep["evaluation_step"], "kind": rep["evaluation_kind"], "settled": rep["evaluation_settled"]},
               "error_metric": rep["error_metric"]}
        isg.setdefault("errors", errors)
        isg.setdefault("conditions", [])
        return {"done": {"ispatialgym": isg}}

    def get_obs(self):
        return self._obs(), {}

    def step(self, action, n_render_iterations=1):
        r = self.robot
        idx = r.controller_action_idx
        q_old = r.q.clone()
        base = action[idx["base"]] * th.tensor(M.BASE_VEL_LIMITS)
        yaw = C.yaw_of(r.quat)
        c, s = math.cos(yaw), math.sin(yaw)
        r.pos = r.pos + th.tensor([c * float(base[0]) - s * float(base[1]), s * float(base[0]) + c * float(base[1]), 0.0]) * DT
        r.quat = C.quat_from_yaw(yaw + float(base[2]) * DT)
        r.q[r.trunk_control_idx] = action[idx["trunk"]]
        for a in M.ARMS:
            r.q[r.arm_control_idx[a]] = action[idx[f"arm_{a}"]]
            cmd = float(action[idx[f"gripper_{a}"]])
            fingers = r.q[r.gripper_control_idx[a]].clone()
            target = 0.05 * (max(-1.0, min(1.0, cmd)) + 1.0) / 2.0  # JoyLo smooth mode: the command is the finger target (-1 = closed, +1 = 50 mm)
            r.q[r.gripper_control_idx[a]] = fingers + th.clamp(target - fingers, -0.01, 0.01)
            # assisted grasp (OmniGibson's rule): the gripper "applies a grasp" whenever its position target is below the
            # open limit; closing pads that reach an object's half width (both pads touching) attach it, the exact open
            # command (+1) releases it. ``ag_fail_next`` refuses whole closures (open -> close transitions), like an
            # assisted grasp that never sees both pads on the object.
            closing = target < float(fingers.max()) - 1e-9
            if cmd < 1.0 - 1e-6:
                if closing and not self._closing[a]:
                    self._closing[a] = True
                    self._refuse[a] = r.ag_fail_next > 0
                    if self._refuse[a]:
                        r.ag_fail_next -= 1
                if closing and r._ag_obj_in_hand[a] is None and not self._refuse[a]:
                    hit = r._calculate_in_hand_object(a)
                    if hit is not None:
                        r._create_assisted_grasp_joint(a, {"target_obj": hit[0], "target_link_name": hit[1], "joint_type": "FixedJoint"})
            else:
                self._closing[a] = False
                if r._ag_obj_in_hand[a] is not None:
                    r._ag_obj_in_hand[a] = None
                    r._ag_obj_constraint_params[a] = None
            held = r._ag_obj_in_hand[a]
            if held is not None:
                off = r._ag_offset.get(a)
                if off is not None:
                    p, q = C.compose(r.eef_pose(a), off)
                    held.pos, held.quat = p, q
        r.dq = (r.q - q_old) / DT
        self.sim.current_time += DT
        self._current_step += 1
        self.steps += 1
        self.last_action = action.clone()
        if getattr(self, "after_step", None) is not None:
            self.after_step(action)  # test hook: scripted world reactions (an object following the pad, spinning, ...)
        self._render_once()  # env.step(..., n_render_iterations=1)
        return self._obs(), 0.0, False, False, self._info()


class FakeContacts:
    """Scripted RigidContactAPI: ``pairs(step)`` -> set of (own_body_path, other_body_path)."""

    def __init__(self, script):
        self.script = script
        self.step = 0

    def get_contact_pairs(self, scene_idx, query, with_set, current_only):
        self.step += 1
        return set(self.script(self.step, query))


def drive(motions_env: FakeEnv, gen) -> int:
    n = 0
    for a in gen:
        motions_env.step(a)
        n += 1
        assert n < 20000, "generator did not finish"
    return n


def iors_world(ag_roundtrip=True, evaluate=None, render_lag: int = 1):
    """The IORS ontop small-scale test sidecar as a fake world: robot at the init stance, apple / plate on the table."""
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-ontop_small_scale.json"))
    init = sc.stance("init")
    robot = FakeRobot(base=tuple(init.base_pose2d))
    g = sc.grasps[0]
    apple = FakeObject("apple_omzprq_0", [g.grasp.position[0], g.grasp.position[1], g.grasp.position[2] - 0.03], category="apple")
    plate = FakeObject("plate_pkkgzc_0", [g.place.position[0], g.place.position[1], g.place.position[2] - 0.06], category="plate", extent=(0.2, 0.2, 0.02))
    objects = {"apple.n.01_1": apple, "plate.n.04_1": plate}
    env = FakeEnv(robot, objects, evaluate=evaluate, ag_roundtrip=ag_roundtrip, render_lag=render_lag)
    return sc, env


def make_executor(env, sc, tmp_path=None, recorder_cfg=None, exec_cfg=None, qa_cfg=None, contacts=None, camera=None, qa=False):
    """The executor over the fake world; ``qa``: run the ``--qa`` metrics (``ExecutorConfig.qa``, off here: the fake world's
    scripted motions are no demos, the metrics have tests of their own)."""
    backend = FakeBackend(env.robot)
    motions = Motions(env, backend, MotionsConfig(settle_steps=5, gripper_steps=20, cartesian_step=0.05, cartesian_rot_step=0.3))
    qa_cfg = qa_cfg or QA.QAConfig(stall_steps=100000, fov_min_steps=100000)
    monitors = QA.Monitors(env, env.robot, cfg=qa_cfg, camera=camera)
    monitors._contact_api = contacts if contacts is not None else FakeContacts(lambda step, query: [])
    recorder = None
    if tmp_path is not None:
        recorder = Recorder(str(tmp_path), env, env.robot, CAM_KEY, dict(env.objects), fps=30, cfg=recorder_cfg or RecorderConfig(hdf5=True, lerobot=False), instance_id=sc.instance_id)
        recorder.start(env.get_obs()[0], {}, state0=env.sim.dump_state(serialized=True))
    cfg = exec_cfg or ExecutorConfig(settle_steps=50)
    cfg.qa = qa
    env.task.tracker = MET.SuccessTracker(settle_eval=env.task.tracker.settle_eval, settle_steps=cfg.settle_steps)
    ex = Executor(env, env.task, sc, recorder=recorder, motions=motions, monitors=monitors, cfg=cfg, sim=env.sim, tracked_objects=dict(env.objects), cam_key=CAM_KEY)
    return ex, motions, backend, recorder


#: ``--qa`` limits the fake world's programs pass: its base drives without moving the base joints (a pause on every program)
#: and its scripted arms move faster on average than BEHAVIOR's teleoperators (jerky motion)
QUIET_QA = QA.QAThresholds(pause_steps=10**6, vel_avg=1e9, vel_prop_over_09max=1.0, acc_avg=1e9, jerk_avg=1e9)


def quiet_qa(monkeypatch, thresholds: QA.QAThresholds = QUIET_QA) -> None:
    """The executor's ``--qa`` metrics with ``thresholds`` (default :data:`QUIET_QA`), for a test of one metric."""
    import functools

    from ispatialgym.oracle import executor as EXE

    monkeypatch.setattr(EXE, "QA", functools.partial(QA.QA, thresholds=thresholds))


def success_eval(goal_obj: str, goal_pos: Sequence[float], tol: float = 0.05):
    def evaluate(env):
        o = env.objects[goal_obj]
        d = float(th.linalg.norm(o.pos - th.tensor(goal_pos)))
        ok = d < tol and all(env.robot._ag_obj_in_hand[a] is None for a in M.ARMS)
        still = float(env.robot.dq.abs().max()) < 1e-6
        return {"binary_success": ok, "settled": still, "errors": {"pose_errors": {goal_obj: {"position_error": d, "within_tolerance": d < tol}}}}

    return evaluate


# ==========================================================================================
# execution table
# ==========================================================================================


def test_executors_cover_every_action_class():
    cov = EX.coverage()
    assert all(cov.values()), cov
    assert set(cov) == {cls.__name__ for cls in VOCABULARY}
    for cls in VOCABULARY:
        fn, post = EX.EXECUTORS[cls]
        assert callable(fn) and callable(post)
    assert set(EX.MANIPULATION_ACTIONS) < set(VOCABULARY)
    assert Navigate not in EX.MANIPULATION_ACTIONS and Push in EX.MANIPULATION_ACTIONS


def test_every_test_sidecar_program_dispatches():
    paths = sorted(glob.glob(os.path.join(SIDECARS, "*.json")) + glob.glob(os.path.join(VARIANTS, "*.json")))
    assert len(paths) >= 22
    for path in paths:
        sc = Sidecar.load(path)
        for action in expand(sc.variant, sc):
            ex_fn, post_fn = EX.executor_for(action)
            assert ex_fn is not None and post_fn is not None


def test_postcondition_base_follow_progress():
    sc = Sidecar.load(os.path.join(SIDECARS, "configuration_matching-articulation_open_large_scale.json"))
    follow = [a for a in expand(sc.variant, sc) if isinstance(a, BaseFollow)][0]
    art = follow.articulation

    class St:
        def __init__(self, q, held=True):
            self.q, self.held = q, held
            self.obj = object()

        def held_object(self, arm):
            return self.obj if self.held else None

        def object(self, name):
            return self.obj

        def joint_position(self, name, joint):
            return self.q

    ctx = {}
    assert EX.check_postcondition(follow, St(art.q_init), {}, ctx) == "no_motion"
    assert EX.check_postcondition(follow, St(art.q_init + 0.4 * (art.q_goal - art.q_init)), {}, ctx) == "tolerance"
    assert EX.check_postcondition(follow, St(art.q_at(art.passive_from)), {}, ctx) is None
    assert EX.check_postcondition(follow, St(art.q_goal), {}, ctx) is None
    assert EX.check_postcondition(follow, St(art.q_goal, held=False), {}, ctx) == "failed_grasp"
    assert EX.joint_tolerance(art) == pytest.approx(max(0.05, 0.1 * abs(art.q_goal - art.q_init)))


def test_push_close_leaf_frame_and_postcondition():
    sc = Sidecar.load(os.path.join(VARIANTS, "configuration_matching-articulation_close_large_scale__room_door.json"))
    push = expand(sc.variant, sc)[-1]
    d0, m0, sgn = EX.leaf_frame(push.hinge_xy, push.handle_xy_closed, push.approach_closed)
    assert float(d0.norm()) == pytest.approx(1.0) and float(m0.norm()) == pytest.approx(1.0) and abs(float(d0 @ m0)) < 1e-6
    assert float(m0[0]) * -push.approach_closed[0] + float(m0[1]) * -push.approach_closed[1] > 0.9  # the outer face looks at the robot
    assert sgn in (1.0, -1.0) and EX.leaf_width(push.handle_radius_m) == pytest.approx(math.sqrt(push.handle_radius_m ** 2 - EX.HANDLE_STANDOFF_M ** 2) + 0.035)

    class St:
        def __init__(self, q):
            self.q = q

        def joint_position(self, name, joint):
            assert (name, joint) == (push.object, push.joint)
            return self.q

    tol = max(0.05, 0.10 * push.q_from)
    assert EX.check_postcondition(push, St(0.0), {}, {}) is None
    assert EX.check_postcondition(push, St(0.9 * tol), {}, {}) is None
    assert EX.check_postcondition(push, St(0.5 * push.q_from), {}, {}) == "tolerance"
    assert EX.check_postcondition(push, St(0.95 * push.q_from), {}, {}) == "no_motion"


# the push finish of the office cabinet door (close_large#03, rvunhj_0 / j_link_2): released at 0.148 rad, swing sign -1 (-z hinge),
# the fist 0.248 m from the hinge line on the outer face, 5 cm below the bar
_FINISH_HINGE = th.tensor([8.6734, -6.7254, 0.4969])
_FINISH_Q = 0.148


def _finish_plan(q_from: float = _FINISH_Q, q_to: float = 0.0) -> Dict[str, Any]:
    turn = C.quat_from_axis_angle([0.0, 0.0, 1.0], -q_from)
    n, d = C.quat_rotate(turn, th.tensor([1.0, 0.0, 0.0])), C.quat_rotate(turn, th.tensor([0.0, 1.0, 0.0]))
    point = th.tensor([8.6734, -6.7254, 0.6811]) + 0.248 * d + 0.0086 * n
    contact = C.fingertip_eef(point, -n, d, clearance=0.0)
    return {"arm": "right", "stance": "manipulation", "base_pose2d": [9.265, -7.09, 2.12], "k_release": 0.855, "q_from": q_from, "q_to": q_to, "q_lower": 0.0,
            "hinge_world": _FINISH_HINGE.tolist(), "swing_sign": -1.0, "contact": {"position": contact[0].tolist(), "orientation": contact[1].tolist()},
            "normal": n.tolist(), "pre_m": 0.10, "arm_q_pre": [0.3] * 7, "object_pose": {"position": [8.4084, -6.3264, 0.7969], "orientation": [0.0, 0.0, 0.0, 1.0]}}


def _finish_push(**kw):
    return PushClose(arm="right", object="bottom_cabinet.n.01_1", joint="j_link_2", hinge_xy=(8.6734, -6.7254), handle_xy_closed=(8.6936, -6.3612),
                     approach_closed=(-1.0, 0.0, 0.0), handle_radius_m=0.364, handle_z=0.7311, q_from=_FINISH_Q, q_lower=0.0, tol_span_rad=1.021,
                     plan=[_finish_plan()], drag_m=0.0, **kw)


def test_planned_push_turns_the_recorded_contact_with_the_leaf():
    """A planned push-close chunk (a cabinet door's push finish): at the angle it was planned at, the recorded contact, its pre-contact
    10 cm back along the approach and the arc to the landing (+angle about +z for swing sign -1); with the leaf 0.05 rad further
    shut the same point of the leaf (turned about the hinge line, the approach with it) and 0.05 rad less to push; nothing left at or
    past the landing; a moved object carries the record with it."""
    plan = _finish_plan()
    g = EX.planned_push(plan, _FINISH_Q)
    assert th.allclose(g["contact"][0], th.tensor(plan["contact"]["position"]), atol=1e-6) and th.allclose(g["contact"][1], th.tensor(plan["contact"]["orientation"]), atol=1e-6)
    assert th.allclose(g["pre"][0], g["contact"][0] - 0.10 * g["approach"], atol=1e-6) and th.allclose(g["approach"], -g["normal"], atol=1e-5)
    assert g["remaining"] == pytest.approx(_FINISH_Q) and g["delta"] == pytest.approx(_FINISH_Q) and float(g["centre"][2]) == pytest.approx(float(g["contact"][0][2]))
    g2 = EX.planned_push(plan, _FINISH_Q - 0.05)
    r1, r2 = (g["contact"][0] - _FINISH_HINGE)[:2], (g2["contact"][0] - _FINISH_HINGE)[:2]
    assert float(r2.norm()) == pytest.approx(float(r1.norm()), abs=1e-5)
    assert C.wrap_angle(math.atan2(float(r2[1]), float(r2[0])) - math.atan2(float(r1[1]), float(r1[0]))) == pytest.approx(0.05, abs=1e-5)
    assert C.quat_angle(g2["contact"][1], g["contact"][1]) == pytest.approx(0.05, abs=1e-4) and g2["remaining"] == pytest.approx(_FINISH_Q - 0.05)
    assert th.allclose(g2["normal"], th.tensor([math.cos(_FINISH_Q - 0.05), -math.sin(_FINISH_Q - 0.05), 0.0]), atol=1e-5)
    assert EX.planned_push(plan, -0.01)["remaining"] == 0.0 and EX.planned_push(plan, -0.01)["delta"] == 0.0
    moved = (th.tensor([8.5084, -6.3264, 0.7969]), C.IDENTITY_QUAT.clone())
    g3 = EX.planned_push(plan, _FINISH_Q, live=moved)
    assert th.allclose(g3["contact"][0], g["contact"][0] + th.tensor([0.1, 0.0, 0.0]), atol=1e-5) and th.allclose(g3["centre"], g["centre"] + th.tensor([0.1, 0.0, 0.0]), atol=1e-5)


def test_push_finish_postcondition_uses_the_instance_tolerance():
    """A push finish is judged on the instance's span (``tol_span_rad``: max(0.05, 0.10 x 1.021 rad) = 0.102 rad), not on its own
    0.148 rad push; ``no_motion`` stays relative to the push. Without ``tol_span_rad`` (a room door) the push's own span counts."""
    class St:
        def __init__(self, q):
            self.q = q

        def joint_position(self, name, joint):
            return self.q

    push = _finish_push()
    assert EX.check_postcondition(push, St(0.09), {}, {}) is None
    assert EX.check_postcondition(push, St(0.12), {}, {}) == "tolerance"
    assert EX.check_postcondition(push, St(0.14), {}, {}) == "no_motion"
    assert EX.check_postcondition(replace(push, tol_span_rad=None), St(0.09), {}, {}) == "tolerance"
    lang = SimpleNamespace(language=SimpleNamespace(phrase=lambda o: "the cabinet"))
    assert push.annotate(lang)[0] == "push the cabinet the rest of the way shut with the closed right hand on its outer face"
    assert push.targets(None, None)["leaf"]["planned_chunks"] == 1


class _FinishMotions:
    """Stub motions of a planned push-close on a leaf about ``_FINISH_HINGE``: the guarded approach touches the face after ``gap``
    steps (each further step shuts the leaf 0.01 rad), every arc waypoint shuts it by ``gain`` x the waypoint's turn about the hinge
    (a hard stop at 0), the back-off leaves it."""

    def __init__(self, q, gap=4, gain=1.0, base=(9.265, -7.09, 2.12)):
        self.q, self.gap, self.gain, self.base = q, gap, gain, base
        self.calls: List[Tuple] = []
        self.arc_turns: List[float] = []
        self.steps = 0
        self.backend = SimpleNamespace(update_world=lambda **kw: None, q_full=lambda: th.zeros(28))
        self.robot = SimpleNamespace(arm_control_idx={"right": th.arange(14, 21), "left": th.arange(7, 14)}, get_joint_positions=lambda: th.zeros(28))
        self.state = SimpleNamespace(trunk_error=lambda: 0.0, arm_target={})
        self.cfg = SimpleNamespace(diff_ik_soft_pos_tol_free=0.02)
        self.eef = (th.zeros(3), C.IDENTITY_QUAT.clone())

    def base_pose2d(self):
        return self.base

    def eef_pose(self, arm):
        return self.eef

    def _deflection(self, arm):
        return None

    def gripper(self, arm, close, **kw):
        self.calls.append(("gripper", close))
        yield th.zeros(23)

    def arm_to(self, arm, pose=None, q=None, ignore_objects=(), q_hint=None, accept_other_branch=False, **kw):
        self.calls.append(("arm_to", "witness" if q_hint is not None else "pose", len(ignore_objects)))
        self.eef = as_world_pose(pose)
        yield th.zeros(23)

    def arm_along(self, arm, waypoints, loaded=False, partial=False, soft_pos_tol=None, **kw):
        wps = [as_world_pose(w) for w in waypoints]
        if partial:  # the arc about the hinge (closed by the caller once the leaf is shut)
            self.arc_turns.append(0.0)
            for w in wps:
                a0 = math.atan2(float(self.eef[0][1] - _FINISH_HINGE[1]), float(self.eef[0][0] - _FINISH_HINGE[0]))
                a1 = math.atan2(float(w[0][1] - _FINISH_HINGE[1]), float(w[0][0] - _FINISH_HINGE[0]))
                self.arc_turns[-1] += C.wrap_angle(a1 - a0)
                self.q[0] = max(0.0, self.q[0] - self.gain * abs(C.wrap_angle(a1 - a0)))
                self.eef = w
                yield th.zeros(23)
            return
        if soft_pos_tol is not None:  # the back-off along the fist's approach
            self.calls.append(("back_off",))
            self.steps = 0
            self.eef = wps[-1]
            yield th.zeros(23)
            return
        self.steps += 1  # a guarded-approach step
        self.calls.append(("approach", round(float(th.linalg.norm(wps[-1][0] - self.eef[0])), 3)))
        self.eef = wps[-1]
        if self.steps > self.gap:
            self.q[0] = max(0.0, self.q[0] - 0.01)
        yield th.zeros(23)

    def hold(self, steps=1):
        for _ in range(int(steps)):
            yield th.zeros(23)

    def settle(self, max_steps=None, vel_tol=None):
        yield th.zeros(23)

    def unload(self, arms, steps=None, trunk=True):
        if False:
            yield th.zeros(23)
        return {a: 0.0 for a in arms}


def _run_finish(push, motions, q):
    state = SimpleNamespace(joint_position=lambda o, j: float(q[0]), object=lambda n: SimpleNamespace(name="bottom_cabinet_rvunhj_0"),
                            object_pose=lambda n: (th.tensor([8.4084, -6.3264, 0.7969]), C.IDENTITY_QUAT.clone()))
    ctx: Dict[str, Any] = {"cfg": EX.ExecuteConfig(), "hooks": {}}
    list(EX.execute_push_close(push, motions, state, push.targets(None, None), ctx))
    return ctx["push_close"], EX.check_postcondition(push, state, {}, ctx)


class _OffStanceMotions(_FinishMotions):
    """The base stands 13.5 cm off the planned stance: ``track_path`` drives it there; the arms-ready plan is recorded."""

    def __init__(self, q):
        super().__init__(q, base=(9.40, -7.09, 2.12))
        self.driven: List[List[Tuple[float, ...]]] = []

    def track_path(self, waypoints, **kw):
        self.driven.append([tuple(w) for w in waypoints])
        self.base = tuple(waypoints[-1])
        yield th.zeros(23)

    def arm_to(self, arm, pose=None, q=None, **kw):
        if q is not None:
            self.calls.append(("arm_to", "ready", 0))
            yield th.zeros(23)
            return
        yield from super().arm_to(arm, pose=pose, **kw)


def test_planned_push_close_replays_the_sampler_chunk(monkeypatch):
    """``push_close`` with a plan (a cabinet door's push finish) from the planned stance: no drive (the base stands on it), the closed
    fist planned to the pre-contact by the recorded arm witness with the cabinet present, the guarded approach along the planned
    approach, one arc about the hinge that shuts the leaf (counter-clockwise for swing sign -1) and stops once it is closed, the fist
    back along its approach; a leaf that lags the fist gets the chunk repeated (at most ``plan_repeats`` times, the witness only while
    the leaf is near the planned angle) and is judged on the instance's tolerance; a base off the planned stance drives back first."""
    q = [_FINISH_Q]
    motions = _FinishMotions(q)
    rep, verdict = _run_finish(_finish_push(), motions, q)
    assert verdict is None and rep["closed"] and rep["planned"] == 1 and len(rep["chunks"]) == 1 and q[0] <= EX.CONFIG.push_close_stop_rad
    chunk = rep["chunks"][0]
    assert chunk["pre"] == "the recorded witness" and "stance" not in chunk and chunk["arc"].startswith("stopped: leaf closed")
    assert motions.calls[0] == ("gripper", True) and motions.calls[1] == ("arm_to", "witness", 0)
    assert [c for c in motions.calls if c[0] == "approach"] == [("approach", 0.02)] * 5 and motions.calls[-1] == ("back_off",)
    assert len(motions.arc_turns) == 1 and motions.arc_turns[0] == pytest.approx(_FINISH_Q - 0.01, abs=1e-4)  # +: counter-clockwise
    # the leaf follows the fist only partly: the chunk is repeated while it gains, at most twice, and the result is judged on 0.102 rad
    q = [_FINISH_Q]
    motions = _FinishMotions(q, gain=0.2)
    rep, verdict = _run_finish(_finish_push(), motions, q)
    assert [c["repeat"] for c in rep["chunks"]] == [False, True, True] and not rep["closed"] and verdict is None and 0.05 < q[0] < 0.07
    assert all(a > b for a, b in zip([c["q_before_deg"] for c in rep["chunks"]], [c["q_settled_deg"] for c in rep["chunks"]]))
    assert [c[1] for c in motions.calls if c[0] == "arm_to"] == ["witness", "witness", "pose"]  # 0.04 rad off the planned angle, then 0.07
    # a plan that lands on the band edge (the arc to the limit was not trackable): pushed there once, not again for the last degree
    q = [_FINISH_Q]
    motions = _FinishMotions(q)
    rep, verdict = _run_finish(replace(_finish_push(), plan=[_finish_plan(q_to=0.051)]), motions, q)
    assert len(rep["chunks"]) == 2 and rep["chunks"][0]["arc"] == "complete" and rep["chunks"][1]["result"].startswith("the leaf is at the planned landing")
    assert q[0] == pytest.approx(0.051, abs=2e-3) and verdict is None and len(motions.arc_turns) == 1 and not rep["closed"]
    # the base stands off the planned stance: both arms ready, a drive back to it, then the chunk as planned
    monkeypatch.setattr(EX, "posture_q", lambda robot, name, arm: th.zeros(7))
    q = [_FINISH_Q]
    motions = _OffStanceMotions(q)
    rep, verdict = _run_finish(_finish_push(), motions, q)
    assert len(motions.driven) == 1 and motions.driven[0] == [pytest.approx((9.265, -7.09, 2.12), abs=1e-5)]
    assert rep["chunks"][0]["stance"] == [9.265, -7.09, 2.12] and verdict is None
    assert motions.calls[:2] == [("arm_to", "ready", 0), ("gripper", True)]


@dataclass
class _Tracker:
    v_max: Tuple[float, float, float] = (0.3, 0.3, 0.5)


class _RoomPushMotions:
    """Stub motions of the room-door push-close loop (no plan) on a leaf about ``hinge`` whose joint value q turns it about +z by s·q:
    the guarded approach touches the face after ``gap`` steps (each further step shuts the leaf ``step_close`` rad), an arc waypoint
    turning the fist about the hinge the closing way (−s) shuts the leaf by ``gain`` x the turn (a hard stop at 0) and one turning it the
    other way leaves it, a base drag (``track_path`` with a tracker config) shuts it by ``drag_gain``; ``trace`` records every decision."""

    def __init__(self, q, hinge, s, gap=3, gain=1.0, drag_gain=0.0, step_close=0.01, base=(10.0, -6.5, math.pi)):
        self.q, self.hinge, self.s, self.gap, self.gain, self.drag_gain, self.base = q, hinge, s, gap, gain, drag_gain, base
        self.step_close = step_close
        self.trace: List[Tuple] = []
        self.arc_turns: List[float] = []
        self.steps = 0
        self.follow_phase = "arm"
        self.backend = SimpleNamespace(update_world=lambda **kw: None)
        self.robot = SimpleNamespace(arm_control_idx={"right": th.arange(14, 21), "left": th.arange(7, 14)}, get_joint_positions=lambda: th.zeros(28))
        self.state = SimpleNamespace(trunk_error=lambda: 0.0, arm_target={})
        self.cfg = SimpleNamespace(diff_ik_soft_pos_tol_free=0.02, base_tracker=_Tracker())
        self.eef = (th.zeros(3), C.IDENTITY_QUAT.clone())

    def _angle(self, p) -> float:
        return math.atan2(float(p[1]) - self.hinge[1], float(p[0]) - self.hinge[0])

    def base_pose2d(self):
        return self.base

    def eef_pose(self, arm):
        return self.eef

    def _deflection(self, arm):
        return None

    def gripper(self, arm, close, **kw):
        yield th.zeros(23)

    def arm_to(self, arms, pose=None, q=None, ignore_objects=(), **kw):
        if pose is not None:
            self.eef = as_world_pose(pose)
        yield th.zeros(23)

    def track_path(self, waypoints, tracker_cfg=None, **kw):
        if tracker_cfg is not None:  # the base drag with the fist on the face
            self.trace.append(("drag", round(self.q[0], 4)))
            self.q[0] = max(0.0, self.q[0] - self.drag_gain)
        else:
            self.trace.append(("drive",))
        self.base = tuple(waypoints[-1])
        yield th.zeros(23)

    def arm_along(self, arm, waypoints, loaded=False, partial=False, soft_pos_tol=None, **kw):
        wps = [as_world_pose(w) for w in waypoints]
        if partial:
            self.arc_turns.append(0.0)
            for w in wps:
                turn = C.wrap_angle(self._angle(w[0]) - self._angle(self.eef[0]))
                self.arc_turns[-1] += turn
                if self.s * turn < 0.0:
                    self.q[0] = max(0.0, self.q[0] + self.gain * self.s * turn)
                self.eef = w
                yield th.zeros(23)
            self.trace.append(("arc", round(self.arc_turns[-1], 4), round(self.q[0], 4)))
            return
        if soft_pos_tol is not None:
            self.trace.append(("back_off",))
            self.steps = 0
            self.eef = wps[-1]
            yield th.zeros(23)
            return
        self.steps += 1
        self.eef = wps[-1]
        if self.steps > self.gap:
            self.q[0] = max(0.0, self.q[0] - self.step_close)
        yield th.zeros(23)

    def hold(self, steps=1):
        for _ in range(int(steps)):
            yield th.zeros(23)

    def settle(self, max_steps=None, vel_tol=None):
        yield th.zeros(23)

    def unload(self, arms, steps=None, trunk=True):
        if False:
            yield th.zeros(23)
        return {a: 0.0 for a in arms}


def _run_room_push(push, motions, q, monkeypatch):
    """``execute_push_close`` without a plan: the arms-ready posture stubbed, and the stance search's IK (``pushable_arc_from``) reaching
    the whole arc from the first candidate of each chunk (the front stance straight out from the pre-contact) and nothing from the others."""
    calls = [0]

    def arc_from(motions_, arm, base, pre, contact, arc):
        calls[0] += 1
        return (len(arc), None) if calls[0] % 20 == 1 else (-1, None)

    monkeypatch.setattr(EX, "posture_q", lambda robot, name, arm: th.zeros(7))
    monkeypatch.setattr(EX, "pushable_arc_from", arc_from)
    state = SimpleNamespace(joint_position=lambda o, j: float(q[0]), object=lambda n: SimpleNamespace(name=n, links={}))
    ctx: Dict[str, Any] = {"cfg": EX.ExecuteConfig(), "hooks": {}}
    list(EX.execute_push_close(push, motions, state, push.targets(None, None), ctx))
    return ctx["push_close"], EX.check_postcondition(push, state, {}, ctx)


def _chunk_qs(rep) -> List[Tuple]:
    return [(c.get("q_before_deg"), c.get("q_after_deg"), c.get("q_after_drag_deg"), c.get("q_settled_deg"), "base_drag" in c, c.get("result")) for c in rep["chunks"]]


def test_room_door_push_close_counts_progress_in_the_joint_for_either_swing_sign(monkeypatch):
    """The room-door push-close loop judges a chunk by the joint: q falls toward ``q_lower`` as the leaf shuts, whatever the swing sign s
    (the arc about the hinge turns by −s·Δ in the world). A −1 leaf (the office cabinet door of close_large#03, −z hinge) that the arcs
    shut chunk by chunk is closed without a base drag, exactly like its mirror image, a +1 leaf (the same chunks, the same joint values);
    a +1 leaf takes the decisions it always took: the base drags when an arc gains < 2°, the loop stops when a chunk gains < 0.5°."""
    cab = PushClose(arm="right", object="bottom_cabinet.n.01_1", joint="j_link_2", hinge_xy=(8.6734, -6.7254), handle_xy_closed=(8.6936, -6.3612),
                    approach_closed=(-1.0, 0.0, 0.0), handle_radius_m=0.364, handle_z=0.7311, q_from=1.021, q_lower=0.0)
    mirror = replace(cab, handle_xy_closed=(8.6936, 2 * -6.7254 + 6.3612))  # reflected across the hinge's x line
    assert EX.leaf_frame(cab.hinge_xy, cab.handle_xy_closed, cab.approach_closed)[2] == -1.0
    assert EX.leaf_frame(mirror.hinge_xy, mirror.handle_xy_closed, mirror.approach_closed)[2] == 1.0
    runs = {}
    for name, push, s in (("cabinet", cab, -1.0), ("mirror", mirror, 1.0)):
        q = [1.021]
        motions = _RoomPushMotions(q, push.hinge_xy, s)
        rep, verdict = _run_room_push(push, motions, q, monkeypatch)
        runs[name] = (rep, verdict, motions)
    rep, verdict, motions = runs["cabinet"]
    assert verdict is None and rep["closed"] and len(rep["chunks"]) == 3 and not any("base_drag" in c for c in rep["chunks"])
    assert [t[0] for t in motions.trace].count("drag") == 0 and all(t > 0.0 for t in motions.arc_turns)  # counter-clockwise shuts a -1 leaf
    assert [c["q_before_deg"] for c in rep["chunks"]] == pytest.approx([58.5, 37.93, 17.35], abs=0.02) and rep["chunks"][-1]["arc"].startswith("stopped: leaf closed")
    rep_m, verdict_m, motions_m = runs["mirror"]
    assert _chunk_qs(rep_m) == _chunk_qs(rep) and verdict_m is None and all(t < 0.0 for t in motions_m.arc_turns)
    assert motions_m.arc_turns == pytest.approx([-t for t in motions.arc_turns], abs=1e-4)
    # +1, an arc that gains only 1.4 deg per chunk (2.0 deg with the approach's step): the base drags each chunk (0.5 deg more), the loop
    # goes on while the chunks gain
    q = [1.021]
    motions = _RoomPushMotions(q, mirror.hinge_xy, 1.0, gain=0.07, drag_gain=math.radians(0.5))
    rep, verdict = _run_room_push(replace(mirror, max_chunks=3), motions, q, monkeypatch)
    assert [c["base_drag"].startswith("0.20 m") for c in rep["chunks"]] == [True, True, True] and [t[0] for t in motions.trace].count("drag") == 3
    assert "result" not in rep["chunks"][-1] and verdict == "tolerance" and not rep["closed"]
    # +1, nothing but the approach's touch (0.46 deg) moves the leaf: a drag, then the loop stops at the first chunk
    q = [1.021]
    motions = _RoomPushMotions(q, mirror.hinge_xy, 1.0, gain=0.0, step_close=0.008)
    rep, verdict = _run_room_push(mirror, motions, q, monkeypatch)
    assert len(rep["chunks"]) == 1 and "base_drag" in rep["chunks"][0] and rep["chunks"][0]["result"] == "no progress" and verdict == "no_motion"


def test_room_door_push_close_execution_is_unchanged(monkeypatch):
    """The room door's push-close (the close fixture's door, +1, no cabinet leaf options) runs as it did before the cabinet's room-door
    style close (values of 388fe4a2 on the same stub): the leaf geometry report, each chunk's contact, free edge, stance and landing."""
    sc = Sidecar.load(os.path.join(VARIANTS, "configuration_matching-articulation_close_large_scale__room_door.json"))
    push = expand(sc.variant, sc)[-1]
    q = [push.q_from]
    motions = _RoomPushMotions(q, push.hinge_xy, 1.0)
    rep, verdict = _run_room_push(push, motions, q, monkeypatch)
    assert rep["geometry"] == {"hinge_xy": [-6.861306703398571, 5.211934738096777], "leaf_width_m": 0.829, "thickness_m": 0.04, "q_start_deg": 49.21}
    assert [(c["q_before_deg"], c["q_after_deg"], c["stance"], c["contact"]) for c in rep["chunks"]] == [
        (49.21, 28.63, [-5.95, 5.198, -2.316], [-6.348, 4.766, 0.987]), (28.63, 8.06, [-6.013, 4.879, -2.675], [-6.538, 4.614, 0.987]),
        (8.06, 0.0, [-6.184, 4.602, -3.034], [-6.768, 4.539, 0.987])]
    assert rep["chunks"][0]["free_edge_xy"] == [-6.252, 4.65] and verdict is None and rep["closed"]


def _cabinet_push(**kw) -> PushClose:
    """``push_close`` of the office cabinet door (close_large#03, swing sign -1) closed room-door style, as ``program_articulation_large``
    builds it from the handle waypoints and the sampler's leaf record (``articulation.push_close``)."""
    opts = dict(leaf_width_m=0.3979, face_offset_m=0.0086, handle_standoff_m=0.0206, chunk_deg=45.0, drag_m=0.08, q_closed=0.0, contact_clearance_m=0.0307)
    opts.update(kw)
    return PushClose(arm="right", object="bottom_cabinet.n.01_1", joint="j_link_2", hinge_xy=(8.6734, -6.7254), handle_xy_closed=(8.694, -6.3612),
                     approach_closed=(-1.0, 0.0, 0.0), handle_radius_m=0.3648, handle_z=0.7311, q_from=1.021, q_lower=0.0, **opts)


def test_cabinet_push_close_pushes_the_measured_leaf(monkeypatch):
    """A cabinet door closed room-door style (owner 2026-10-12): with the recorded leaf (width 0.398 m, outer face 8.6 mm and bar contact
    20.6 mm in front of the hinge line) the push pose has its fingertips on the outer face (the pad centre 3.07 cm in front of it),
    0.248 m from the hinge line and 5 cm below the bar grasp, turned with the leaf (swing sign -1) - the room door's estimate on the
    same leaf (a lever 6 cm off the face, half a 4 cm leaf, the pad centre on the face) put the pad centre 1.5 cm inside the leaf.
    45 deg arcs close it from 58.5 deg in two chunks, counter-clockwise; an arc that gains < 2 deg drags the base 8 cm."""
    q0 = 1.021
    d1 = th.tensor([math.sin(q0), math.cos(q0)])  # the leaf at q0: the closed leaf (+y from the hinge) turned about +z by -q0
    m1 = th.tensor([math.cos(q0), -math.sin(q0)])  # its outer-face normal
    hinge = th.tensor([8.6734, -6.7254])
    q = [q0]
    motions = _RoomPushMotions(q, (8.6734, -6.7254), -1.0)
    rep, verdict = _run_room_push(_cabinet_push(), motions, q, monkeypatch)
    assert verdict is None and rep["closed"] and len(rep["chunks"]) == 2 and not any("base_drag" in c for c in rep["chunks"])
    assert all(t > 0.0 for t in motions.arc_turns) and rep["chunks"][0]["delta_deg"] == 45.0 and rep["chunks"][0]["q_after_deg"] == pytest.approx(12.93, abs=0.02)
    assert rep["geometry"]["leaf_width_m"] == 0.398 and rep["geometry"]["face_offset_m"] == 0.0086 and rep["geometry"]["swing_sign"] == -1.0
    c = th.tensor(rep["chunks"][0]["contact"][:2]) - hinge
    tips = float(c @ m1) - (C.FINGERTIP_LENGTH - C.CONTACT_CENTER_Z)  # the fingertips' offset along the outer normal
    assert tips == pytest.approx(0.0086, abs=1.5e-3) and float(c @ d1) == pytest.approx(0.2479, abs=1.5e-3) and rep["chunks"][0]["contact"][2] == 0.681
    q = [q0]
    room_est = _cabinet_push(leaf_width_m=None, face_offset_m=None, handle_standoff_m=None, contact_clearance_m=0.0)
    rep_room, _ = _run_room_push(room_est, _RoomPushMotions(q, (8.6734, -6.7254), -1.0), q, monkeypatch)
    c_room = th.tensor(rep_room["chunks"][0]["contact"][:2]) - hinge
    assert float(c_room @ m1) < 0.0086 - 0.010 and "face_offset_m" not in rep_room["geometry"]
    # the leaf lags the fist (1.35 deg of the 45 deg arc): the base drags 8 cm toward the leaf
    q = [q0]
    motions = _RoomPushMotions(q, (8.6734, -6.7254), -1.0, gain=0.03)
    rep, _ = _run_room_push(_cabinet_push(max_chunks=1), motions, q, monkeypatch)
    assert rep["chunks"][0]["base_drag"] == "0.08 m toward the frame"


def test_postcondition_push_no_motion():
    sc = Sidecar.load(os.path.join(SIDECARS, "alignment-axial.json"))
    push = [a for a in expand(sc.variant, sc) if isinstance(a, Push)][0]
    goal = push.contact.goal.tensors()

    class St:
        def __init__(self, obj_pose, eef=goal):
            self.obj_pose, self.eef = obj_pose, eef

        def eef_pose(self, arm):
            return self.eef

        def object_pose(self, name):
            return self.obj_pose

    p0 = (th.tensor([1.0, 2.0, 0.8]), C.IDENTITY_QUAT.clone())
    targets = {"waypoints": [push.contact.goal]}
    assert EX.check_postcondition(push, St(p0), targets, {"object_pose0": p0}) == "no_motion"
    moved = (th.tensor([1.0, 2.05, 0.8]), C.IDENTITY_QUAT.clone())
    assert EX.check_postcondition(push, St(moved), targets, {"object_pose0": p0}) is None
    far = (goal[0] + th.tensor([0.1, 0.0, 0.0]), goal[1])
    assert EX.check_postcondition(push, St(moved, eef=far), targets, {"object_pose0": p0}) == "execution"


def test_push_closure_geometry_residual_and_extension():
    """The push closure on the pilot's board-game arc: pivot = the object's COM (the sampler's hinge), swept angle
    = the required rotation; a 10 % under-rotation gives a residual in the push's sense that the arc extension
    covers; an overshoot or a residual beyond the recorded arc is skipped."""
    sc = Sidecar.load(os.path.join(SIDECARS, "alignment-axial.json"))
    push = [a for a in expand(sc.variant, sc) if isinstance(a, Push)][0]
    wps = push.contact.waypoints
    geom = EX.push_geometry(wps, push.contact.direction)
    assert geom.kind == "arc" and geom.radius > 0.03 and abs(geom.angle) > 0.2
    # the pad's swept angle about the pivot equals the angle between the init and goal radii
    p0, p1 = wps[0].tensors()[0], wps[-1].tensors()[0]
    a0 = math.atan2(float(p0[1] - geom.pivot[1]), float(p0[0] - geom.pivot[0]))
    a1 = math.atan2(float(p1[1] - geom.pivot[1]), float(p1[0] - geom.pivot[0]))
    assert abs(C.wrap_angle(a1 - a0) - geom.angle) < 1e-6
    sign = math.copysign(1.0, geom.angle)

    class St:
        def __init__(self, yaw, target=None):
            self.q = C.quat_from_yaw(yaw) if hasattr(C, "quat_from_yaw") else th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
            self.target = target

        def object_pose(self, name):
            return th.tensor([1.0, 2.0, 0.8]), self.q

        def pose_target(self, name):
            return self.target

    pose0 = (th.tensor([1.0, 2.0, 0.8]), C.IDENTITY_QUAT.clone())
    # (a) arc-derived: the object rotated 90 % of the arc -> residual = the missing 10 %, in the push's sense
    r, src = EX.push_residual(geom, St(0.9 * geom.angle), push.contact.object, pose0)
    assert src == "arc_angle" and abs(r - 0.1 * geom.angle) < 1e-6 and r * sign > 0
    # (b) goal-derived: a world orientation target 0.08 rad beyond the current yaw wins over the arc
    tgt = SimpleNamespace(frame="world", orientation=th.tensor([0.0, 0.0, math.sin((geom.angle + 0.08 * sign) / 2), math.cos((geom.angle + 0.08 * sign) / 2)]), orientation_equivalences=[], position=None)
    r, src = EX.push_residual(geom, St(geom.angle, tgt), push.contact.object, pose0)
    assert src == "goal_orientation" and abs(abs(r) - 0.08) < 1e-6 and r * sign > 0
    # (c) the extension continues the circle: every pose keeps the radius, the last one is `residual` further round
    ext = EX.extend_push(geom, wps[-1], r)
    for p, q in ext:
        assert abs(math.hypot(float(p[0] - geom.pivot[0]), float(p[1] - geom.pivot[1])) - geom.radius) < 1e-5
    a_end = math.atan2(float(ext[-1][0][1] - geom.pivot[1]), float(ext[-1][0][0] - geom.pivot[0]))
    assert abs(C.wrap_angle(a_end - a1) - r) < 1e-6
    # the pad orientation turns with the arc (the same yaw change as the position)
    assert abs(C.wrap_angle(C.yaw_of(ext[-1][1]) - C.yaw_of(wps[-1].tensors()[1])) - r) < 1e-5
    # (d) the tangent at the goal points along the push sense, perpendicular to the radius
    t = EX.push_tangent(geom, wps[-1])
    radial = th.tensor([float(p1[0] - geom.pivot[0]), float(p1[1] - geom.pivot[1]), 0.0])
    assert abs(float(th.dot(t, radial))) < 1e-6 and abs(float(th.linalg.norm(t)) - 1.0) < 1e-6
    # (e) a straight push: distance along the direction, extension along it
    line = EX.push_geometry([wps[0], wps[-1]], push.contact.direction)
    assert line.kind == "line" and abs(line.distance - float(th.linalg.norm((p1 - p0)[:2]))) < 1e-6
    ext = EX.extend_push(line, wps[-1], 0.02)
    assert len(ext) == 1 and abs(float(th.linalg.norm(ext[0][0] - p1)) - 0.02) < 1e-6


def test_wall_alignment_without_a_brace_is_push_only():
    """Owner 2026-10-11 (round wall clocks): a target without a low corner records no brace; the world-vertical program is
    then transition -> push -> retreat (as axial's push-only program) instead of failing on the missing brace."""
    sc = Sidecar.load(os.path.join(SIDECARS, "alignment-world_vertical.json"))
    with_brace = expand(sc.variant, sc)
    sc.contacts = replace(sc.contacts, brace=None)
    assert not sc.has_block("contacts.brace") and sc.problems() == []  # the brace is optional in REQUIRED_BLOCKS
    acts = expand(sc.variant, sc)
    assert any(isinstance(a, QA_Brace) for a in with_brace) and not any(isinstance(a, QA_Brace) for a in acts)
    pushes = [a for a in acts if isinstance(a, Push)]
    assert len(pushes) == 1 and type(acts[-1]).__name__ == "Retreat" and acts[-1].arm == pushes[0].contact.arm
    assert [type(a).__name__ for a in acts[:-2]] == [type(a).__name__ for a in with_brace if not isinstance(a, (QA_Brace, Push))][:len(acts) - 2]


def test_push_geometry_of_a_hanging_painting_is_an_arc_about_the_hook_axis():
    """The wall-vertical push swings a painting about its horizontal hook axis: the recorded init / mid / goal pad
    poses lie on a circle in a vertical plane. The geometry is an arc about that (horizontal) axis — not the planar
    'line of 8 mm' the xy projection gave (which measured the residual on the object's position, i.e. on the hook
    that never moves) — the residual is the rotation still missing about the hook, an overshoot flips its sign, the
    extension stays on the circle and the back-off tangent points down the arc."""
    sc = Sidecar.load(os.path.join(SIDECARS, "alignment-world_vertical.json"))
    push = [a for a in expand(sc.variant, sc) if isinstance(a, Push)][0]
    wps = push.contact.waypoints
    geom = EX.push_geometry(wps, push.contact.direction)
    assert geom.kind == "arc" and geom.angle > 0.2 and abs(float(geom.axis[2])) < 0.05 and abs(float(geom.axis[0])) > 0.99  # the hook axis is along x
    p0, p1 = wps[0].tensors()[0], wps[-1].tensors()[0]
    for p in (p0, p1):
        radial = p - geom.pivot
        radial = radial - geom.axis * float(th.dot(radial, geom.axis))
        assert abs(float(th.linalg.norm(radial)) - geom.radius) < 1e-4
    q0 = C.quat_from_axis_angle(geom.axis, -0.4)  # the painting hangs 0.4 rad off, to be swung back by the push

    class St:
        def __init__(self, q, target=None):
            self.q, self.target = q, target

        def object_pose(self, name):
            return geom.pivot.clone(), self.q  # the hook does not move

        def pose_target(self, name):
            return self.target

    pose0 = (geom.pivot.clone(), q0)
    # (a) arc-derived: rotated 90 % of the swept angle -> the missing 10 % in the push's sense
    r, src = EX.push_residual(geom, St(C.qmul(C.quat_from_axis_angle(geom.axis, 0.9 * geom.angle), q0)), push.contact.object, pose0)
    assert src == "arc_angle" and abs(r - 0.1 * geom.angle) < 1e-5 and r > 0
    # (b) goal-derived: an orientation target 0.08 rad beyond -> short; swung 0.14 rad past it -> overshoot (negative)
    goal_q = C.quat_from_axis_angle(geom.axis, 0.0)
    tgt = SimpleNamespace(frame="world", orientation=goal_q, orientation_equivalences=[], position=None)
    r, src = EX.push_residual(geom, St(C.quat_from_axis_angle(geom.axis, -0.08), tgt), push.contact.object, pose0)
    assert src == "goal_orientation" and abs(r - 0.08) < 1e-5
    r, _ = EX.push_residual(geom, St(C.quat_from_axis_angle(geom.axis, 0.14), tgt), push.contact.object, pose0)
    assert abs(r + 0.14) < 1e-5
    # (c) the extension stays on the circle, (d) the tangent at the goal is perpendicular to the radius and mostly up
    ext = EX.extend_push(geom, wps[-1], 0.1)
    for p, q in ext:
        radial = p - geom.pivot
        radial = radial - geom.axis * float(th.dot(radial, geom.axis))
        assert abs(float(th.linalg.norm(radial)) - geom.radius) < 1e-4
    t = EX.push_tangent(geom, wps[-1])
    radial = p1 - geom.pivot
    assert abs(float(th.dot(t, radial))) < 1e-5 and float(t[2]) > 0.5 and abs(float(t[0])) < 1e-5


def test_execute_push_holds_measures_repushes_and_backs_off(tmp_path):
    """``execute_push`` on a fake world whose pushed object follows the pad at 90 %: after the arc it holds until the
    object is still, measures the residual, pushes the extension once, backs the pad off along the reversed push
    direction, and the postcondition checks the eef at that back-off pose."""
    sc = Sidecar.load(os.path.join(SIDECARS, "alignment-axial.json"))
    actions = expand(sc.variant, sc)
    push = [a for a in actions if isinstance(a, Push)][0]
    geom = EX.push_geometry(push.contact.waypoints, push.contact.direction)
    init = sc.stance("manipulation")
    robot = FakeRobot(base=tuple(init.base_pose2d))
    obj_name = push.contact.object
    board = FakeObject(obj_name.replace(".", "_"), pos=[float(geom.pivot[0]), float(geom.pivot[1]), 0.8])
    board.graspable = False  # a pad pushing a face never gets an assisted grasp on it
    env = FakeEnv(robot, {obj_name: board})
    backend = FakeBackend(robot)
    motions = Motions(env, backend, MotionsConfig(settle_steps=5, gripper_steps=5, cartesian_step=0.05, cartesian_rot_step=0.3))
    state = SimState(env, env.task, robot)
    # the object rotates about its COM at 90 % of the pad's swept angle (as seen from the pivot)
    a_init = math.atan2(float(push.contact.init.position[1] - geom.pivot[1]), float(push.contact.init.position[0] - geom.pivot[0]))

    z_contact = float(push.contact.init.position[2])
    contact = {"on": False}

    def follow(step_action):
        p = robot.eef_pose("right")[0]
        if not contact["on"]:
            contact["on"] = float(th.linalg.norm(p - push.contact.init.tensors()[0])) < 0.01  # the pad reached the init contact
            return
        on_circle = abs(math.hypot(float(p[0] - geom.pivot[0]), float(p[1] - geom.pivot[1])) - geom.radius) < 0.02 and abs(float(p[2]) - z_contact) < 0.03
        if not on_circle:
            return  # the pad left the object's face: no contact, no rotation
        a = math.atan2(float(p[1] - geom.pivot[1]), float(p[0] - geom.pivot[0]))
        swept = C.wrap_angle(a - a_init)
        if swept * math.copysign(1.0, geom.angle) > 0:
            yaw = 0.9 * swept
            board.quat = th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])

    env.after_step = follow
    ctx = {}
    targets = push.targets(sc, state)
    n = 0
    for a in EX.run_action(push, motions, state, targets, ctx):
        env.step(a)
        n += 1
    res = ctx["push_residual"]
    assert res["kind"] == "arc" and res["source"] == "arc_angle"
    assert abs(res["value"] - 0.1 * geom.angle) < 0.02 and res.get("extended_by") == res["value"]
    assert abs(res["after"]) < abs(res["value"])  # the extension closed most of the residual
    assert ctx["push_pivot_offset_m"] < 1e-6
    # the pad backed off 2 cm along the reversed tangent and the postcondition checks that pose
    end = state.eef_pose("right")
    assert abs(float(th.linalg.norm(end[0] - ctx["push_end_pose"][0]))) < 1e-3
    assert EX.check_postcondition(push, state, targets, ctx) is None
    assert ctx["push_object_drift_m"] < 1e-6


def axial_world(gain=1.0, stop=None, tol=0.10):
    """The braced axial fixture as a fake world: the board at the arc pivot rotates ``gain`` × the pad's swept angle,
    never back, and never beyond the stop ``stop(offset)`` × the arc angle (``offset`` = how far the brace stands
    outward of its recorded contact; None = no stop); the task has a world orientation target on the board."""
    sc = Sidecar.load(os.path.join(SIDECARS, "alignment-axial.json"))
    actions = expand(sc.variant, sc)
    push = [a for a in actions if isinstance(a, Push)][0]
    brace = [a for a in actions if isinstance(a, QA_Brace)][0]
    geom = EX.push_geometry(push.contact.waypoints, push.contact.direction)
    robot = FakeRobot(base=tuple(sc.stance("manipulation").base_pose2d))
    obj = push.contact.object
    board = FakeObject("board_0", pos=[float(geom.pivot[0]), float(geom.pivot[1]), 0.8])
    board.graspable = False
    env = FakeEnv(robot, {obj: board})
    goal_q = th.tensor([0.0, 0.0, math.sin(geom.angle / 2), math.cos(geom.angle / 2)])  # the arc's full swept angle
    env.task._tro = SimpleNamespace(meta=SimpleNamespace(pose_targets={obj: SimpleNamespace(frame="world", orientation=goal_q, orientation_tolerance=tol, position=None,
                                                                                                position_tolerance=None, orientation_equivalences=[])}))

    def evaluate(env_):
        err = C.quat_angle(goal_q, board.quat)
        return {"binary_success": err < tol, "settled": True, "errors": {"pose_errors": {obj: {"orientation_error": err, "within_tolerance": err < tol, "in_goal": True}}}}

    env.evaluate = evaluate
    a_init = math.atan2(float(push.contact.init.position[1] - geom.pivot[1]), float(push.contact.init.position[0] - geom.pivot[0]))
    z_contact = float(push.contact.init.position[2])
    d_brace = th.tensor(brace.contact.direction[:3]).float()
    c_brace = brace.contact.contact.tensors()[0]
    sign = math.copysign(1.0, geom.angle)
    contact = {"on": False, "yaw": 0.0}

    def brace_offset():
        return float(th.dot(c_brace - robot.eef_pose("left")[0], d_brace))

    def follow(step_action):
        p = robot.eef_pose("right")[0]
        if not contact["on"]:
            contact["on"] = float(th.linalg.norm(p - push.contact.init.tensors()[0])) < 0.01
            return
        on_circle = abs(math.hypot(float(p[0] - geom.pivot[0]), float(p[1] - geom.pivot[1])) - geom.radius) < 0.02 and abs(float(p[2]) - z_contact) < 0.03
        if not on_circle:
            return
        swept = C.wrap_angle(math.atan2(float(p[1] - geom.pivot[1]), float(p[0] - geom.pivot[0])) - a_init)
        if swept * sign <= 0:
            return
        target = gain * abs(swept)
        if stop is not None:
            target = min(target, stop(brace_offset()) * abs(geom.angle))
        contact["yaw"] = min(contact["yaw"], abs(C.yaw_of(board.quat)) + 1e-6)  # a checkpoint restore resets the board
        contact["yaw"] = max(contact["yaw"], target)  # a pushed object does not come back
        yaw = sign * contact["yaw"]
        board.quat = th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])

    env.after_step = follow
    return sc, env, actions, board, brace, push, geom


def test_brace_retry_ladder_short_and_overshoot(tmp_path):
    """Owner decisions 2026-09-16 05:26 / 05:39, split 2026-09-26: a braced push ending out of tolerance is re-run from the
    checkpoint before the brace with the brace +5 / +10 ... mm outward when the target touched the brace finger before the goal
    (short, brace hit). A short with NO brace catch is not the brace's fault: out of tolerance the pad aims past the goal by the
    measured sag (``aim_push_past_goal``, the brace unchanged), inside the tolerance it is accepted without improvement rungs
    (world vertical 0925-193257 / 225052 / 235643: every rung 'short' with brace hits 0, the painting sagged back after the pad
    left, the rungs only added plan failures)."""

    def touch_while_blocked(ex, env, board, pushes=1):
        # the brace finger touches the board only while the brace blocks it (the first ``pushes`` pushes: brace at the recorded
        # contact); moved outward, the board no longer reaches it
        n = {"push": 0}
        orig = ex.monitors.begin_action

        def begin_action(action, index, attempt, **kw):
            if getattr(action, "name", "") == "push":
                n["push"] += 1  # every rung starts its push afresh
            return orig(action, index, attempt, **kw)

        ex.monitors.begin_action = begin_action
        return FakeContacts(lambda step, query: [(f"{env.robot.prim_path}/left_gripper_finger_link1", f"{board.prim_path}/base_link")]
                            if ex.monitors.action is not None and ex.monitors.action.name == "push" and n["push"] <= pushes else [])

    # (a) short with a brace hit: the stop blocks the board at 80 % of the arc with the recorded brace; 5 mm further out the board
    # reaches the goal without touching it -> one rung, then accepted in tolerance
    sc, env, actions, board, brace, push, geom = axial_world(gain=1.0, stop=lambda off: 0.8 + 40.0 * off)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    ex.monitors._contact_api = touch_while_blocked(ex, env, board)
    result = ex.run_program(actions)
    assert result["success"], result["verdict"]
    assert len(result["brace_retry"]) == 1
    br = result["brace_retry"][0]
    assert [r["reason"] for r in br["retries"]] == ["short"] and br["retries"][0]["next_offset_m"] == 0.005 and br["final_offset_m"] == 0.005
    assert br["retries"][0]["brace_hits"] > 0 and br["retries"][0]["residual"] > 0.10
    assert br["final"]["in_tolerance"] and br["final"]["reason"] is None and "accepted" in br["final"]
    assert [a["name"] for a in result["actions"]] == ["transition", "brace", "push", "retreat", "retreat"]
    # the re-run brace stood 5 mm outward along -direction of the recorded contact (offset_brace geometry)
    moved = EX.offset_brace(brace, 0.005)
    assert abs(float(th.dot(brace.contact.contact.tensors()[0] - moved.contact.contact.tensors()[0], th.tensor(brace.contact.direction[:3]).float())) - 0.005) < 1e-6
    assert moved.contact.extra["retry_offset_m"] == 0.005 and moved.contact.pre.position != brace.contact.pre.position
    # the record is consistent: committed rows = committed steps, the rolled-back group kept aside
    assert len(rec.rows["step"]) == result["n_steps"] and np.all(np.diff(np.asarray(rec.rows["step"])) == 1)
    assert any(r["attempt"] == 0 and r["action_index"] == 1 for r in rec.rollbacks)
    # (b) overshoot caught by the stop within the aim band (the arc aim is biased to the overshoot side): no rung
    sc, env, actions, board, brace, push, geom = axial_world(gain=1.15, stop=lambda off: 1.15 + 30.0 * off)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path / "b")
    result = ex.run_program(actions)
    assert result["success"], result["verdict"]
    assert result["brace_retry"] == []
    # (c) hopeless: the stop blocks at 60 % whatever the brace does and is touched every time -> every brace rung, then the goal
    # (its pose_reached atom) never holds: not_reached
    sc, env, actions, board, brace, push, geom = axial_world(gain=1.0, stop=lambda off: 0.6)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path / "c")
    ex.monitors._contact_api = touch_while_blocked(ex, env, board, pushes=99)
    result = ex.run_program(actions)
    assert result["reason"] == "not_reached"
    br = result["brace_retry"][0]
    assert [r["next_offset_m"] for r in br["retries"]] == [0.005, 0.010, 0.015, 0.020] and br["final_offset_m"] == 0.020 and br["final"]["reason"] == "short"
    # (d) short without a brace contact, out of tolerance (the board just slips back): the brace stays, the pad aims past the goal
    # by the measured sag and the re-push ends in tolerance
    sc, env, actions, board, brace, push, geom = axial_world(gain=0.6)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path / "d")
    result = ex.run_program(actions)
    assert result["success"], result["verdict"]
    br = result["brace_retry"][0]
    assert [r["reason"] for r in br["retries"]] == ["short_free"] and br["retries"][0]["next_offset_m"] == 0.0 and br["final_offset_m"] == 0.0
    assert br["final"]["in_tolerance"] and br["final"]["reason"] is None


def test_brace_ladder_falls_back_to_the_in_tolerance_rung_when_an_improvement_rung_fails_to_plan(tmp_path):
    """Owner 2026-09-26: an improvement rung that fails outright (wave 3's fourth push: 'trajectory optimization to the joint goal
    failed' twice) goes back to the rung that was already in tolerance - rolled back, re-run once - instead of failing the demo."""
    sc, env, actions, board, brace, push, geom = axial_world(gain=1.0, stop=lambda off: 0.8 + 40.0 * off)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    # the brace is touched on every push: rung 2 (5 mm out) ends IN tolerance but above the aim, so a third rung is tried
    ex.monitors._contact_api = FakeContacts(lambda step, query: [(f"{env.robot.prim_path}/left_gripper_finger_link1", f"{board.prim_path}/base_link")]
                                            if ex.monitors.action is not None and ex.monitors.action.name == "push" else [])
    n = {"brace": 0}
    orig = ex.run_action

    def run_action(index, action, watch=()):
        if action.name == "brace":
            n["brace"] += 1
            backend.fail_plans = 1000 if n["brace"] == 3 else 0  # every plan of the third rung fails
        return orig(index, action, watch=watch)

    ex.run_action = run_action
    result = ex.run_program(actions)
    assert result["success"], result["verdict"]
    br = result["brace_retry"][0]
    reasons = [r["reason"] for r in br["retries"]]
    assert reasons[:2] == ["short", "short"] and reasons[-1].startswith("rung failed: plan_failed"), reasons
    assert br["final_offset_m"] == 0.005 and br["final"]["in_tolerance"] and br["verdict"] is None
    assert n["brace"] == 4  # rung 1, rung 2, the failed rung 3, the re-run of rung 2


def test_arm_along_unwinds_a_loaded_arm_before_the_leg():
    """A loaded arm reports joint positions whose FK lies past the eef link (PhysX wind-up under a contact force; the
    touching push retreat snapped the link 2.9 cm onto the joints' FK and carried the box 27.6 mm). ``arm_along`` then
    prepends the link's own pose to the tracked leg so the joints are re-targeted onto the link before it moves; an
    unloaded arm (FK == link) tracks the leg as given."""
    from ispatialgym.oracle import executor as EXE

    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc)
    arm = "right"
    frame = M.EEF_LINK[arm]
    captured = []
    orig = backend.solve_ik_path

    def spy(emb, waypoints, *a, **k):
        captured.append([tuple(float(v) for v in wp[frame][0]) for wp in waypoints])
        return orig(emb, waypoints, *a, **k)

    backend.solve_ik_path = spy
    p0, q0 = motions.eef_pose(arm)
    up = (p0 + th.tensor([0.0, 0.0, 0.05]), q0)
    # (a) unloaded: the backend's FK agrees with the link -> the leg starts at the first densified waypoint
    backend.fk = lambda frames, q_full=None, emb_name="whole_body": {frame: (p0.clone(), q0.clone())}
    assert ex.run_generator(motions.arm_along(arm, [up]), EXE.PHASE_ORACLE) is None
    first = captured[-1][0]
    assert abs(first[2] - float(p0[2])) > 0.004  # the first tracked waypoint is already above the start
    # (b) loaded: the joints' FK sits 3 cm past the link -> the link's own pose is the first waypoint
    p1, q1 = motions.eef_pose(arm)
    backend.fk = lambda frames, q_full=None, emb_name="whole_body": {frame: (p1 + th.tensor([-0.03, 0.0, 0.0]), q1.clone())}
    assert ex.run_generator(motions.arm_along(arm, [(p1 + th.tensor([0.0, 0.0, 0.05]), q1)]), EXE.PHASE_ORACLE) is None
    first = captured[-1][0]
    assert all(abs(first[i] - float(p1[i])) < 1e-6 for i in range(3)), (first, p1.tolist())


def test_arm_along_finishes_a_stalled_approach_with_arm_to():
    """A straight approach whose differential IK stalls 7 mm short (no joint limit) is finished: the converged prefix
    is executed, the last centimetres are planned by ``arm_to`` (to the witness when there is one), the grasp goes on."""
    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc)
    actions = expand(sc.variant, sc)
    assert ex.run_action(0, actions[0]) is None
    grasp = actions[1]
    backend.stall_ik_after = (2, 0.007)  # the approach's last dense waypoint (3 of 3 at this resolution) stalls 7 mm short
    n0 = len(backend.calls)
    assert ex.run_action(1, grasp) is None
    calls = backend.calls[n0:]
    assert "plan_pose" in calls and calls.count("solve_ik_path") >= 1
    rec = ex.action_records[1]
    assert rec.attempts == [None]
    assert ex.last_ctx["approach_finish"]["by"] == "curobo_ik" and ex.last_ctx["approach_finish"]["converged"] == 2
    assert ex.action_records[1].notes["approach_finish"]["stall_m"] == 0.007
    assert env.robot._ag_obj_in_hand["right"] is env.objects["apple.n.01_1"]
    # a stall beyond the finish bounds (3 cm short) is still plan_failed
    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc)
    assert ex.run_action(0, actions[0]) is None
    backend.stall_ik_after = (2, 0.03)
    assert ex.run_action(1, expand(sc.variant, sc)[1]) is None  # the retry from the checkpoint succeeds (the fake stall is one-shot)
    rec = ex.action_records[1]
    assert rec.attempts[0]["category"] == "plan_failed" and rec.attempts[1] is None and "approach_finish" not in rec.notes


class _CollectingWriter:
    """Stands in for the imageio video writer: keeps the frames the recorder commits."""

    def __init__(self):
        self.frames = []

    def append_data(self, frame):
        self.frames.append(np.array(frame))

    def close(self):
        pass


@pytest.mark.parametrize("render_lag", [1, 4])
def test_failed_attempt_leaves_no_frame_in_the_committed_episode(tmp_path, monkeypatch, render_lag):
    """Owner review 2026-09-16: frames of a rolled-back attempt survived in ``head_rgb.mp4``. The renderer trails the
    physics, so the restore's recorded hold step showed the failed attempt's last arm pose. Every committed row,
    video frame and LeRobot frame must belong to the committed steps, and every committed frame must show the state
    of its own row (or the previous one: the renderer's lag). ``render_lag=1`` is the lightly loaded simulator (the
    2026-09-16 fix's 3 fixed renders were enough); ``render_lag=4`` the loaded one, where the annotator buffer trails
    the last render by further frames and the fixed 3 renders still recorded the failed attempt's last arm pose as the
    hold step's frame (the one-frame flash of the hand in ~half of the retried demos of 09-18 .. 09-22): the restore
    must render until the head camera shows the checkpoint scene again."""
    writer = _CollectingWriter()
    monkeypatch.setattr(Recorder, "_open_views", lambda self, robs: self._writers.__setitem__("head", writer))
    sc, env = iors_world(render_lag=render_lag)
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, recorder_cfg=RecorderConfig(hdf5=True, lerobot=False, views=("head",), frame_buffer="raw"))
    backend.fail_ik_paths = 1  # the grasp approach fails once: attempt 0 has N > 0 recorded steps (gripper + arm_to)
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    grasp = result["actions"][1]
    assert grasp["attempts"][0]["category"] == "plan_failed" and grasp["attempts"][1] is None
    n_failed = grasp["attempts"][0]["step"] - grasp["step_start"]
    assert n_failed > 0
    rows = rec.rows
    steps = np.asarray(rows["step"])
    assert len(steps) == result["n_steps"] and np.all(np.diff(steps) == 1)  # committed steps only, contiguous
    frames = writer.frames
    assert len(frames) == result["n_steps"]
    # every committed frame shows a state of the committed sequence: its own row's or (the renderer's lag) an earlier
    # committed row's — never the failed attempt's (the restore hold step used to show its last pose)
    q = np.asarray(rows["joint_positions"])
    base = np.asarray(rows["base_pose"])
    obj_pose = {name: np.asarray(rows[f"object_pose/{name}"]) for name in env.objects}

    def state_of(i):
        return th.cat([th.tensor(q[i]).float(), th.tensor(base[i][:3]).float()] + [th.cat([th.tensor(obj_pose[n][i][:3]).float(), th.tensor(obj_pose[n][i][3:]).float()]) for n in env.objects])

    states = [state_of(i) for i in range(len(steps))]
    for i in range(render_lag, len(steps)):  # the first frames show the reset state the pipeline was primed with
        ok = any(env.frame_matches_state(frames[i], states[j]) for j in range(i - render_lag, i + 1))
        assert ok, f"frame at committed row {i} (step {steps[i]}) shows a state outside the committed episode"
    # and the boundary frame in particular is the restored state, not the failed attempt's last one
    b = grasp["step_start"]  # row index of the restore hold step (steps are 1-based)
    paths = rec.close({"instance_id": sc.instance_id, "success": True}, qa=result["monitors"]["stats"], success=True)
    with h5py.File(paths["hdf5"]) as f:
        g = f["data/demo_0"]
        assert len(g["isg/step"]) == result["n_steps"]
        rb = g["rollbacks/action_1_attempt_0"]
        assert len(rb["isg/step"]) == n_failed and int(rb["isg/step"][0]) == grasp["step_start"] + 1
        q_fail = th.tensor(rb["isg/joint_positions"][-1]).float()
    failed_state = th.cat([q_fail, states[b][28:]])
    assert not env.frame_matches_state(frames[b], failed_state) and not env.frame_matches_state(frames[b + 1], failed_state)
    assert env.frame_matches_state(frames[b], states[b - 1]) or env.frame_matches_state(frames[b], states[b])
    # the restore rendered until the head camera showed the checkpoint scene: 3 frames with the light lag, more with
    # the deep pipeline (each frame read and compared, meta.checkpoint_reports records it)
    rr = result["checkpoint_reports"][0]["restore_renders"]
    assert rr["checked"] and rr["converged"] and rr["diff"] == 0.0
    # the pipeline holds render_lag pre-restore frames; the next one is the scene - but never fewer than the configured
    # minimum (30 by default, owner 2026-09-24: the accumulation grain of the reset scene fades over ~30 frames)
    assert rr["n"] == max(ex.cfg.restore_renders, render_lag + 1), rr


def test_restore_render_loop_caps_and_falls_back():
    """The restore's render loop is bounded and anchored: a head camera stuck on the rolled-back scene (identical
    reads, never closer to the checkpoint observation than to the pre-restore one) stops at ``restore_renders_max``
    with ``converged False`` (a warning, the run goes on); a fresh frame that keeps changing (the temporal
    accumulation after the state jump) is waited for until it settles; without a readable camera or a checkpoint
    observation the fixed ``restore_renders`` minimum is rendered (``checked False``)."""
    sc, env = iors_world(render_lag=1)
    ex, *_ = make_executor(env, sc, exec_cfg=ExecutorConfig(restore_renders=3, restore_renders_max=8, restore_frame_tol=1.0, restore_frame_gain_tol=0.25))
    ex.last_obs, ex.last_info = env.get_obs()[0], {}
    cp = ex.checkpoint(0)
    unchecked = {"n": 3, "diff": None, "diff_stale": None, "change": None, "gain": None, "fresh": None, "converged": None, "checked": False}
    # the plain case: the scene is unchanged, the first read after the minimum matches the checkpoint observation
    renders0 = env.sim.renders
    rr = ex.render_restored(cp)
    assert env.sim.renders - renders0 == 3 and rr["n"] == 3 and rr["diff"] == 0.0 and rr["converged"] and rr["checked"]
    # a camera stuck on the rolled-back scene (identical reads, never fresh): the cap, reported
    sensor = env.robot.sensors[CAM_KEY]
    stuck = th.full((8, 8, 4), 7, dtype=th.uint8)
    sensor.get_obs = lambda: ({"rgb": stuck.clone()}, {})
    renders0 = env.sim.renders
    rr = ex.render_restored(cp, stale=stuck[..., :3].numpy())
    assert env.sim.renders - renders0 == 8 and rr["n"] == 8 and rr["checked"] and rr["converged"] is False and rr["fresh"] is False
    assert rr["diff"] > 1.0 and rr["diff_stale"] == 0.0 and rr["change"] == 0.0
    # a fresh frame converging over renders toward a steady residual above restore_frame_tol (the grain of a reset
    # temporal accumulation): reads 3 .. 7 = the rolled-back scene (not fresh), ref + 40 (fresh, the ghost fading),
    # ref + 10 (still fading: gain 30), ref + 10 again (an exactly repeated read = a duplicate from the queue: neither
    # settles nor counts), ref + 10 with a quarter of the entries + 1 (a new frame no closer to the checkpoint scene
    # than the previous new one: gain -0.25 <= restore_frame_gain_tol, settled)
    ref = np.asarray(ex.head_rgb_of(cp.obs)).astype(np.float32)
    far = np.full_like(ref, 200.0)
    quarter = np.zeros_like(ref)
    quarter.reshape(-1)[: quarter.size // 4] = 1.0
    frames = iter([far, ref + 40, ref + 10, ref + 10, ref + 10 + quarter, ref + 10 + quarter])

    def converging():
        img = np.concatenate([np.clip(next(frames), 0, 255).astype(np.uint8), np.full((8, 8, 1), 255, dtype=np.uint8)], axis=-1)
        return {"rgb": th.from_numpy(img)}, {}

    sensor.get_obs = converging
    renders0 = env.sim.renders
    rr = ex.render_restored(cp, stale=far.astype(np.uint8))
    assert rr["converged"] and rr["n"] == 7 and 1.0 < rr["diff"] < rr["diff_stale"] and 0.0 < rr["change"] <= 0.25 and -0.25 <= rr["gain"] <= 0.0, rr
    assert env.sim.renders - renders0 == 7
    del sensor.get_obs
    # no camera to read: the minimum, unchecked
    sensors = env.robot.sensors
    env.robot.sensors = {}
    try:
        renders0 = env.sim.renders
        rr = ex.render_restored(cp)
    finally:
        env.robot.sensors = sensors
    assert env.sim.renders - renders0 == 3 and rr == unchecked
    # no checkpoint observation (a checkpoint before the first observation): the minimum, unchecked
    cp.obs = None
    renders0 = env.sim.renders
    rr = ex.render_restored(cp)
    assert env.sim.renders - renders0 == 3 and rr == unchecked


def test_placed_orientation_is_reported_for_an_oracle_demo_and_fails_nothing(tmp_path):
    """Owner decision 2026-09-16 10:15 item 2, as a fact since 2026-10-14 (the gate is dealt with elsewhere): every placed
    object's final orientation error against its recorded target tolerance is in ``result.placed_orientation``; an error above
    the tolerance no longer fails the demo, and the TRO target stays non-gating for the evaluation."""
    goal_q = th.tensor([0.0, 0.0, math.sin(0.25), math.cos(0.25)])  # the apple's recorded orientation target (0.5 rad yaw)

    def world(err_rad):
        sc, env = iors_world()
        goal = _place_goal(sc)
        env.task._tro = SimpleNamespace(meta=SimpleNamespace(pose_targets={"apple.n.01_1": SimpleNamespace(frame="world", orientation=goal_q, orientation_tolerance=0.2,
                                                                                                                 orientation_metric="full", position=None, position_tolerance=None,
                                                                                                                 orientation_equivalences=[])}))

        def evaluate(env_):
            d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
            # the evaluator reports the IORS pose target (in :measure, not in :goal): orientation error as measured; its own
            # (profile) tolerance is not under test here - the measured targets count as within
            d["errors"]["pose_errors"]["apple.n.01_1"].update({"orientation_error": err_rad, "in_goal": False, "within_tolerance": err_rad <= 0.2})
            d["measure_within"] = True
            return d

        env.evaluate = evaluate
        return sc, env

    # (a) the cup fell at release: 0.46 rad is outside the tolerance, reported; the demo still succeeds
    sc, env = world(0.46)
    ex, *_ = make_executor(env, sc, tmp_path)
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    po = result["placed_orientation"]
    assert po["apple.n.01_1"] == {"orientation_error": 0.46, "tolerance": 0.2, "within": False, "metric": "full", "target_in_goal": False}
    assert "plate.n.04_1" not in po  # only the placed objects, not the partner
    # (b) within tolerance
    sc, env = world(0.12)
    ex, *_ = make_executor(env, sc, tmp_path / "b")
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"] and result["placed_orientation"]["apple.n.01_1"]["within"] is True
    # (c) no orientation target on the placed object: nothing to compare
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, *_ = make_executor(env, sc, tmp_path / "c")
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"] and result["placed_orientation"]["apple.n.01_1"]["within"] is None


def test_placed_orientation_compares_cylinders_modulo_their_long_axis(tmp_path, monkeypatch):
    """Owner decision 2026-09-16 17:10 item 4, fallback (the closure investigation ended without its criterion): in the
    ``placed_orientation`` report an object whose model has a continuous symmetry about its long axis in the ICL symmetry
    survey (``assets/icl/revolution_axes.json``: marker, pencil, bottles) is compared modulo rotation about that axis; a
    marker rolled 92 deg about its own axis during finger closure (0916-152503: 0.386 rad full error) is within, the same
    marker standing on its end is not, every untagged object keeps the full error, and the equivalence can be switched off."""
    from ispatialgym.oracle import executor as EXE

    goal_q = th.tensor([0.0, 0.0, math.sin(0.25), math.cos(0.25)])  # the recorded goal: yaw 0.5 rad, the long axis (x) level
    table = {"marker/xakoic": {"axis": "x", "tags": ["xinf"], "sources": ["symmetry_near", "geom_near"]}}
    monkeypatch.setattr(EXE, "revolution_axes", lambda path=EXE.REVOLUTION_AXES_PATH: table)

    def world(err_rad, category="marker", model="xakoic", quat_after=None):
        sc, env = iors_world()
        goal = _place_goal(sc)
        obj = env.objects["apple.n.01_1"]
        obj.category, obj.model = category, model
        obj.aabb_extent = th.tensor([0.1536, 0.0171, 0.0141])
        env.task._tro = SimpleNamespace(meta=SimpleNamespace(pose_targets={"apple.n.01_1": SimpleNamespace(frame="world", orientation=goal_q, orientation_tolerance=0.2,
                                                                                                                 orientation_metric="full", position=None, position_tolerance=None,
                                                                                                                 orientation_equivalences=[])}))

        def evaluate(env_):
            d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
            d["errors"]["pose_errors"]["apple.n.01_1"].update({"orientation_error": err_rad, "in_goal": False, "within_tolerance": err_rad <= 0.2})
            d["measure_within"] = True  # the profile's measured tolerance is not under test here
            return d

        env.evaluate = evaluate
        if quat_after is not None:

            def after(step_action):
                if env.robot._ag_obj_in_hand["right"] is None and float(th.linalg.norm(obj.pos - th.tensor(goal))) < 0.08:
                    obj.quat = quat_after

            env.after_step = after
        return sc, env

    def report(sc, env, path, cfg=None):
        ex, *_ = make_executor(env, sc, path, exec_cfg=cfg)
        result = ex.run_program(expand(sc.variant, sc))
        assert result["success"], result["verdict"]
        return result["placed_orientation"]["apple.n.01_1"]

    rolled = C.qmul(goal_q, th.tensor([math.sin(0.8), 0.0, 0.0, math.cos(0.8)]))  # the goal yaw, then 1.6 rad (92 deg) about the marker's own x
    standing = C.qmul(goal_q, th.tensor([0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]))  # the long axis pitched up 90 deg
    # (a) the rolled marker: the evaluator's full error 1.6 rad, compared modulo the roll -> within
    rep = report(*world(1.6, quat_after=rolled), tmp_path / "a")
    assert rep["within"] is True and rep["orientation_error"] < 0.02 and rep["full_error"] == 1.6 and rep["tolerance"] == 0.2
    assert rep["revolution_axis"] == "x" and rep["model"] == "marker/xakoic" and "rotation about the x axis" in rep["equivalence"]
    # (b) the same marker standing on its end: the axis error is 90 deg
    rep = report(*world(1.6, quat_after=standing), tmp_path / "b")
    assert rep["within"] is False and abs(rep["orientation_error"] - math.pi / 2) < 0.01 and rep["full_error"] == 1.6
    # (c) a target equivalence (a half turn about the marker's y) makes the upside-down marker pass the axis reading too
    sc, env = world(1.6, quat_after=C.qmul(goal_q, th.tensor([0.0, 1.0, 0.0, 0.0])))
    env.task._tro.meta.pose_targets["apple.n.01_1"].orientation_equivalences = [th.tensor([0.0, 1.0, 0.0, 0.0])]
    assert report(sc, env, tmp_path / "c")["orientation_error"] < 0.02
    # (d) an untagged model keeps the full error
    rep = report(*world(1.6, category="apple", model="omzprq", quat_after=rolled), tmp_path / "d")
    assert rep["within"] is False and rep["orientation_error"] == 1.6 and "full_error" not in rep and "equivalence" not in rep
    # (e) the equivalence switched off: the marker's roll keeps the full error
    rep = report(*world(1.6, quat_after=rolled), tmp_path / "e", ExecutorConfig(settle_steps=50, demo_revolution_axes=False))
    assert rep["orientation_error"] == 1.6
    # (f) the category / model fall back to the scene object's name (<category>_<model>_<i>) when the attributes are missing
    sc, env = world(1.6, quat_after=rolled)
    obj = env.objects["apple.n.01_1"]
    del obj.category
    obj.name = "marker_xakoic_0"
    assert report(sc, env, tmp_path / "f")["revolution_axis"] == "x"


def test_revolution_axes_asset_tags_the_pilot_cylinders():
    """The committed table (``scripts/symmetry/revolution_axes.py`` over the ICL symmetry survey) carries the nextto pilot's
    marker and pencil about x, the standing shakers / cans about z, and leaves flat / boxy objects out."""
    from ispatialgym.oracle import executor as EXE

    table = EXE.revolution_axes()
    assert table, EXE.REVOLUTION_AXES_PATH
    assert EXE.revolution_axis("marker", "xakoic")["axis"] == "x" and EXE.revolution_axis("pencil", "hmawzy")["axis"] == "x"
    assert EXE.revolution_axis("colored_pencil", "acumgp")["axis"] == "x" and EXE.revolution_axis("salt_shaker", "iomwtn")["axis"] == "z"
    assert EXE.revolution_axis("can_of_soda", "bfrzvk")["axis"] == "z"
    for key in ("plate/pkkgzc", "apple/omzprq", "box_of_tissues/uglbjc", "box_of_sanitary_napkins/tkrzkb", "spatula/hkvsep"):
        assert key not in table, key
    for k, v in table.items():
        assert v["axis"] in ("x", "y", "z") and f"{v['axis']}inf" in v["tags"] and v["ratio"] >= 1.5, (k, v)


def test_epilogue_is_exactly_settle_steps_and_the_metrics_decide(tmp_path):
    """2026-10-14: a program's epilogue is exactly ``settle_steps`` no-op steps (reserved in the step budget, no second window, no
    motion rule) and the task's metrics decide: success = the run mode's geometric success; the goal held but never through a
    settled window -> ``not_settled``; the goal held, the measured targets never within tolerance -> ``symbolic_only``. Motion over
    the window is a fact (``motion_delta``)."""
    sc, env = iors_world()
    goal = _place_goal(sc)
    env.evaluate = success_eval("apple.n.01_1", goal, tol=0.08)
    apple = env.objects["apple.n.01_1"]

    def spin(step_action):
        if env.robot._ag_obj_in_hand["right"] is None and float(th.linalg.norm(apple.pos - th.tensor(goal))) < 0.08:
            yaw = 0.05 * env._current_step
            apple.quat = th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])

    env.after_step = spin
    ex, *_ = make_executor(env, sc, tmp_path, exec_cfg=ExecutorConfig(settle_steps=40))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    assert result["epilogue_steps"] == 40 and result["n_settle_steps"] == 40 and result["settle_steps"] == 40
    assert result["geometric_success"] and result["geometric_success_step"] <= result["n_steps"] and result["symbolic_success"]
    assert result["motion_delta"]["apple.n.01_1_rad"] > 0.1  # a fact, no verdict
    assert result["other_mode"]["settle_eval"] is False and result["other_mode"]["geometric_success"]
    # never settled: the goal holds, S never does
    sc, env = iors_world()
    env.evaluate = lambda env_: {**success_eval("apple.n.01_1", goal, tol=0.08)(env_), "settled": False}
    ex, *_ = make_executor(env, sc, tmp_path / "b", exec_cfg=ExecutorConfig(settle_steps=40))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["reason"] == "not_settled" and not result["symbolic_success"] and result["goal_first_step"] is not None
    assert result["other_mode"]["symbolic_success"] and result["other_mode"]["geometric_success"]  # settle-eval off would pass
    # symbolic only: the goal holds and settles, the measured target is outside its tolerance
    sc, env = iors_world()
    env.evaluate = lambda env_: {**success_eval("apple.n.01_1", goal, tol=0.08)(env_), "measure_within": False}
    ex, *_ = make_executor(env, sc, tmp_path / "c", exec_cfg=ExecutorConfig(settle_steps=40))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["reason"] == "symbolic_only" and result["symbolic_success"] and not result["geometric_success"]
    # ... which the symbolic success metric accepts
    sc, env = iors_world()
    env.evaluate = lambda env_: {**success_eval("apple.n.01_1", goal, tol=0.08)(env_), "measure_within": False}
    ex, *_ = make_executor(env, sc, tmp_path / "d", exec_cfg=ExecutorConfig(settle_steps=40, success_metric="symbolic"))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"] and result["success_metric"] == "symbolic" and not result["geometric_success"]


# ==========================================================================================
# Layer-1 motions on the toy robot
# ==========================================================================================


def test_track_path_reaches_the_recorded_waypoints():
    robot = FakeRobot()
    env = FakeEnv(robot, {})
    m = Motions(env, FakeBackend(robot))
    n = drive(env, m.track_path([[0.4, 0.1, 0.3], [0.8, 0.5, 1.2]]))
    x, y, yaw = m.base_pose2d()
    assert math.hypot(x - 0.8, y - 0.5) < 0.03 and abs(C.wrap_angle(yaw - 1.2)) < 0.05
    assert n > 30
    # arm / trunk hold targets were not touched by the base motion
    assert th.allclose(robot.q[robot.trunk_control_idx], C.torso_q(0.5, 0.0), atol=1e-5)
    exported = m.export_state()
    assert exported["tracker_index"] is not None and exported["trunk"] == pytest.approx((0.5, 0.0), abs=1e-6)


def test_base_follow_hybrid_drives_base_and_ramps_arm_together():
    """D-011 C hybrid follow: per recorded pose the base drives to ``base_pose2d`` while the arm's hold target ramps to
    ``arm_q`` by the tracker's progress; an arm-only pose (base unchanged) ramps the joints alone; the last pose is met
    by both."""
    robot = FakeRobot(base=(0.0, 0.0, 0.0))
    env = FakeEnv(robot, {})
    m = Motions(env, FakeBackend(robot))
    idx = robot.arm_control_idx["right"]
    q_start = robot.q[idx].clone()
    qa = q_start + th.tensor([0.10, -0.05, 0.0, 0.08, 0.0, 0.05, -0.04])
    qb = qa + th.tensor([0.05, 0.05, -0.05, 0.0, 0.03, 0.0, 0.02])
    qc = qb + th.tensor([0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0])
    path = [
        {"k": 0.0, "q": 0.0, "base_pose2d": [0.0, 0.0, 0.0], "arm_q": q_start.tolist()},
        {"k": 0.25, "q": 0.4, "base_pose2d": [0.15, 0.05, 0.2], "arm_q": qa.tolist()},   # a base move with an arm ramp
        {"k": 0.5, "q": 0.8, "base_pose2d": [0.15, 0.05, 0.2], "arm_q": qb.tolist()},    # arm only
        {"k": 0.75, "q": 1.2, "base_pose2d": [0.30, 0.15, 0.5], "arm_q": qc.tolist()},   # both again
    ]
    seen = []
    orig_emit = m._emit

    def spy(*args, **kwargs):
        seen.append((m.base_pose2d(), m.state.arm_target["right"].clone()))
        return orig_emit(*args, **kwargs)

    m._emit = spy
    n = drive(env, m.base_follow_hybrid(path, "right"))
    x, y, yaw = m.base_pose2d()
    assert math.hypot(x - 0.30, y - 0.15) < 0.03 and abs(C.wrap_angle(yaw - 0.5)) < 0.05
    assert th.allclose(robot.q[idx], qc, atol=1e-3) and n > 20
    # the arm target moved gradually with the base (no jump to the pose's joints at the start of a segment)
    jumps = [float(th.max(th.abs(b[1] - a[1]))) for a, b in zip(seen[:-1], seen[1:])]
    assert max(jumps) < 0.05
    # while the base was still near its start the arm target was still near its start: the ramp follows the progress
    early = [t for (b, t) in seen if math.hypot(b[0], b[1]) < 0.02]
    assert early and float(th.max(th.abs(early[-1] - q_start))) < 0.06
    assert m.export_state()["tracker_index"] is not None
    with pytest.raises(SkillError):
        drive(env, m.base_follow_hybrid([path[0], {"k": 1.0, "q": 1.5, "base_pose2d": [0.3, 0.15, 0.5], "arm_q": [0.0, 0.0]}], "right"))


def test_track_path_stays_in_the_corridor_at_corners():
    """Pilot red flag 2: the constant-speed reference + base-frame smoothing cut 45–63° corners by 5–8 cm and drifted
    25 cm after a 90° turn. The corner-aware profile and the world-frame filter must keep the base within ~2 cm of a
    recorded polyline with sharp corners, with the yaw within a few degrees of the recorded headings."""
    wps = [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, -0.785], [0.8, -0.2, -0.785], [1.0, -0.4, -1.571], [1.0, -0.6, -1.571],
           [1.0, -0.8, -1.571], [1.0, -1.0, 3.142], [0.8, -1.0, 3.142], [0.6, -1.0, 3.142], [0.4, -1.0, 2.356]]
    robot = FakeRobot(base=(0.0, 0.0, 0.0))
    env = FakeEnv(robot, {})
    m = Motions(env, FakeBackend(robot))

    poly = [[0.0, 0.0, 0.0]] + wps  # the tracked polyline starts at the current base pose

    def cross_track(pt):
        best = 1e9
        for a, b in zip(poly[:-1], poly[1:]):
            a, b = th.tensor(a[:2]), th.tensor(b[:2])
            ab = b - a
            t = float(th.clamp(th.dot(pt - a, ab) / max(float(th.dot(ab, ab)), 1e-9), 0.0, 1.0))
            best = min(best, float(th.linalg.norm(pt - (a + ab * t))))
        return best

    errs, yaw_errs, speeds = [], [], []
    for a in m.track_path(wps):
        env.step(a)
        p, q = robot.get_position_orientation()
        errs.append(cross_track(p[:2]))
        r = m.tracker.ref[min(m.tracker.k, len(m.tracker.ref) - 1)]
        yaw_errs.append(abs(C.wrap_angle(C.yaw_of(q) - float(r[2]))))
        speeds.append(float(th.linalg.norm(a[:2] * th.tensor(M.BASE_VEL_LIMITS[:2]))))
    assert max(errs) < 0.02, max(errs)
    assert max(yaw_errs) < math.radians(10.0), math.degrees(max(yaw_errs))
    assert max(speeds) > 0.3  # the straights still run near the path speed
    x, y, yaw = m.base_pose2d()
    assert math.hypot(x - 0.4, y + 1.0) < 0.03 and abs(C.wrap_angle(yaw - 2.356)) < 0.05
    # the reference slows down through the 90° corners: min speed along the reference well below the straights
    ref = m.tracker.ref
    v_ref = th.linalg.norm(ref[1:, :2] - ref[:-1, :2], dim=1) * 30
    assert float(v_ref[len(ref) // 2 :].min()) < 0.2 < float(v_ref.max())


def test_track_path_face_travel_drives_the_staircase_smoothly():
    """Owner 2026-09-22 (camera relocalization): the dense grid record of the pilot (one waypoint per 0.2 m cell, headings
    alternating 0° / -45°, the base slowing at every stair and zig-zagging) is driven as the shared ``face_travel``
    schedule: the base reaches the goal pose, cruises through the stairs without speed dips, turns left / right no more
    often than the real corners, and stays within one map cell + the fillet budget of the recorded polyline."""
    from ispatialgym.robot import base_path as BP

    wps = [[0.2, 0.0, 0.0], [0.4, 0.0, 0.0], [0.6, 0.0, -0.785], [0.8, -0.2, 0.0], [1.0, -0.2, 0.0], [1.2, -0.2, 0.0], [1.4, -0.2, -0.464],
           [1.6, -0.3, 0.0], [1.8, -0.3, 0.0], [2.0, -0.3, 0.0], [2.2, -0.3, 0.0], [2.4, -0.3, -0.785], [2.6, -0.5, 0.0], [2.8, -0.5, 0.0],
           [3.0, -0.5, 0.0], [3.2, -0.5, 0.464], [3.4, -0.4, 0.464], [3.6, -0.3, 0.4]]
    robot = FakeRobot(base=(0.0, 0.0, 0.0))
    env = FakeEnv(robot, {})
    m = Motions(env, FakeBackend(robot))
    poly = [[0.0, 0.0, 0.0]] + wps

    def cross_track(pt):
        best = 1e9
        for a, b in zip(poly[:-1], poly[1:]):
            a, b = th.tensor(a[:2]), th.tensor(b[:2])
            ab = b - a
            t = float(th.clamp(th.dot(pt - a, ab) / max(float(th.dot(ab, ab)), 1e-9), 0.0, 1.0))
            best = min(best, float(th.linalg.norm(pt - (a + ab * t))))
        return best

    errs, speeds, yaw_rates = [], [], []
    for a in m.track_path(wps, style=BP.FACE_TRAVEL):
        env.step(a)
        p, _ = robot.get_position_orientation()
        errs.append(cross_track(p[:2]))
        speeds.append(float(th.linalg.norm(a[:2] * th.tensor(M.BASE_VEL_LIMITS[:2]))))
        yaw_rates.append(float(a[2]) * float(M.BASE_VEL_LIMITS[2]))
    x, y, yaw = m.base_pose2d()
    assert math.hypot(x - 3.6, y + 0.3) < 0.03 and abs(C.wrap_angle(yaw - 0.4)) < 0.05
    cfg = m.cfg.face_travel
    assert max(errs) < cfg.simplify_tol_m + cfg.fillet_deviation_m + 0.02, max(errs)
    # the commanded speed: at cruise most of the drive, no dip below half cruise once up to speed (the stairs are gone)
    v = th.tensor(speeds)
    moving = (v > 0.05).nonzero().flatten()
    inner = v[int(moving[0]) + 40 : int(moving[-1]) - 40]
    assert float(inner.min()) > 0.2, float(inner.min())
    assert float((inner > 0.36).float().mean()) > 0.7
    # the body turns left / right at most as often as the real corners of the simplified path (the record reversed 6 times)
    sign, reversals = None, 0
    for w in yaw_rates:
        if abs(w) < 0.1:
            continue
        s = 1 if w > 0 else -1
        reversals += int(sign is not None and s != sign)
        sign = s
    assert reversals <= 4, reversals
    # the reference the tracker followed is the shared schedule: the same call the sampler validates
    ref = m.tracker.ref
    v_ref = th.linalg.norm(ref[1:, :2] - ref[:-1, :2], dim=1) * 30
    assert float(v_ref[len(ref) // 4 : -len(ref) // 4].min()) > 0.3
    with pytest.raises(SkillError, match="style"):
        drive(env, m.track_path(wps, style="sideways"))


def test_trunk_to_moves_both_scalars_jointly_at_bounded_rates():
    robot = FakeRobot()
    env = FakeEnv(robot, {})
    m = Motions(env, FakeBackend(robot), MotionsConfig(trunk_translate_rate=0.25, trunk_tilt_rate=0.15, settle_steps=5))
    pairs = []
    for a in m.trunk_to(0.75, -0.42):
        env.step(a)
        pairs.append(C.torso_q_to_translate_tilt(robot.q[robot.trunk_control_idx]))
    t, k = pairs[-1]
    assert t == pytest.approx(0.75, abs=1e-3) and k == pytest.approx(-0.42, abs=1e-3)
    # the sweep is the straight line of conventions.trunk_sweep: tilt is an affine function of translate on the way
    on_line = [abs((k - 0.0) - (t - 0.5) / 0.25 * (-0.42)) for t, k in pairs if 0.5 < t < 0.75]
    assert on_line and max(on_line) < 0.02
    # both rates stay within the JoyLo limits and the slower scalar sets the duration
    dts = [pairs[i + 1][0] - pairs[i][0] for i in range(len(pairs) - 1)]
    assert max(dts) <= 0.25 * DT + 1e-6
    assert len(pairs) >= int((0.42 / 0.15) / DT) - 2


def test_trunk_to_ramps_an_off_curve_torso_first():
    robot = FakeRobot()
    robot.q[robot.trunk_control_idx] = th.tensor([1.025, -1.45, -0.47, 0.0]) + th.tensor([0.0, 0.3, 0.0, 0.0])  # off the curve
    env = FakeEnv(robot, {})
    m = Motions(env, FakeBackend(robot), MotionsConfig(settle_steps=5))
    n = drive(env, m.trunk_to(0.5, 0.0))
    assert n > 5
    assert th.allclose(robot.q[robot.trunk_control_idx], C.torso_q(0.5, 0.0), atol=1e-4)


def test_arm_to_and_arm_along_toy_kinematics_and_plan_failed():
    robot = FakeRobot()
    env = FakeEnv(robot, {})
    be = FakeBackend(robot)
    m = Motions(env, be, MotionsConfig(settle_steps=5, cartesian_step=0.02, cartesian_rot_step=0.2))
    goal = (robot.pos + th.tensor([0.6, -0.1, 0.9]), C.quat_from_yaw(0.3))
    drive(env, m.arm_to("right", pose=goal))
    p, q = m.eef_pose("right")
    assert float(th.linalg.norm(p - goal[0])) < 1e-4 and C.quat_angle(q, goal[1]) < 1e-3
    wps = [(goal[0] + th.tensor([0.0, 0.0, -0.05 * k]), goal[1]) for k in (1, 2)]
    n = drive(env, m.arm_along("right", wps))
    p, _ = m.eef_pose("right")
    assert float(th.linalg.norm(p - wps[-1][0])) < 1e-4
    # 10 cm at 0.10 m/s = 1 s = 30 steps of motion plus the convergence hold
    assert 25 <= n <= 60
    be.fail_plans = 1
    with pytest.raises(SkillError) as e:
        drive(env, m.arm_to("right", pose=goal))
    assert e.value.reason == "plan_failed" and e.value.details["stage"] == "trajopt"
    be.fail_ik_paths = 1
    with pytest.raises(SkillError) as e:
        drive(env, m.arm_along("right", wps))
    assert e.value.reason == "plan_failed" and e.value.details["stage"] == "diff_ik"
    # joint-space target: the untucked vector
    drive(env, m.arm_to(("left", "right"), q=robot.untucked_default_joint_pos))
    for a in M.ARMS:
        assert th.allclose(robot.q[robot.arm_control_idx[a]], robot.untucked_default_joint_pos[robot.arm_control_idx[a]], atol=1e-5)


def test_arm_along_without_convergence_wait_and_unwind():
    """The closed-loop articulation follow streams the leg without the joint-convergence hold (``converge=False``: a
    door held by its friction never lets the loaded arm converge) and ``unwind`` re-targets the arm onto its measured
    joints so the drives stop pressing before the release."""
    robot = FakeRobot()
    env = FakeEnv(robot, {})
    be = FakeBackend(robot)
    m = Motions(env, be, MotionsConfig(settle_steps=40, cartesian_step=0.02, cartesian_rot_step=0.2))
    goal = (robot.pos + th.tensor([0.6, -0.1, 0.9]), C.quat_from_yaw(0.3))
    drive(env, m.arm_to("right", pose=goal))
    wps = [(goal[0] + th.tensor([0.0, 0.0, -0.05 * k]), goal[1]) for k in (1, 2)]
    n_conv = drive(env, m.arm_along("right", wps))
    drive(env, m.arm_to("right", pose=goal))
    n_free = drive(env, m.arm_along("right", wps, converge=False))
    # the same motion, minus the convergence hold (the toy joints converge within a few steps)
    assert n_free <= n_conv and n_free >= int((0.10 / 0.10) / DT) - 2
    # unwind: a target displaced from the measured joints (a wound-up arm) snaps back onto them, reporting the deflection
    idx = robot.arm_control_idx["right"]
    m.state.arm_target["right"] = robot.q[idx] + 0.05
    assert m.unwind("right") == pytest.approx(0.05, abs=1e-6)
    assert th.allclose(m.state.arm_target["right"], robot.q[idx])
    assert m.unwind("right") == pytest.approx(0.0, abs=1e-9)


def test_gripper_and_state_roundtrip():
    robot = FakeRobot()
    env = FakeEnv(robot, {})
    m = Motions(env, FakeBackend(robot), MotionsConfig(gripper_steps=30))
    drive(env, m.gripper("right", close=True))
    assert float(robot.q[robot.gripper_control_idx["right"]].max()) < 1e-6 and m.gripper_closed("right")
    st = m.export_state()
    drive(env, m.gripper("right", close=False))
    assert not m.gripper_closed("right")
    m.import_state(st)
    assert m.gripper_closed("right") and m.tracker is None


def test_gripper_closure_ramp_and_preshaped_opening():
    """Owner decision 2026-09-16 17:10 item 4 (closure probe): ``gripper(close, rate)`` ramps the smooth-mode command so the
    finger target moves at most ``rate`` per control step from the current opening to closed (both pads track the same
    target; the stock step command follows), ``gripper_opening`` holds a pre-shaped opening (pinch width / 2 + margin)."""
    robot = FakeRobot()
    env = FakeEnv(robot, {})
    m = Motions(env, FakeBackend(robot), MotionsConfig(gripper_steps=30))
    idx = robot.gripper_control_idx["right"]
    drive(env, m.gripper("right", close=False))
    assert abs(float(robot.q[idx].max()) - 0.05) < 1e-6
    # the pre-shaped opening: the command holds the fingers at 17.9 mm (the marker's 15.9 mm pinch + 10 mm margin)
    cmds = []
    env.after_step = lambda a: cmds.append(float(a[robot.controller_action_idx["gripper_right"]]))
    n = drive(env, m.gripper_opening("right", 0.5 * 0.01585 + 0.010))
    assert n == m.cfg.gripper_preshape_steps == 12 and len(set(cmds)) == 1
    assert abs(cmds[0] - (2.0 * 0.017925 / 0.05 - 1.0)) < 1e-6 and abs(float(robot.q[idx].max()) - 0.017925) < 1e-6
    assert abs(m.gripper_command("right", 0.05) - 1.0) < 1e-6 and abs(m.gripper_command("right", 0.0) + 1.0) < 1e-6 and abs(m.gripper_command("right", 0.9) - 1.0) < 1e-6
    # the ramped closure from the pre-shaped opening: the command's finger target falls by exactly 2.25 mm per step
    # (8 ramp steps for 17.9 mm), then the stock -1 command; without an object the fingers end at 0
    cmds.clear()
    drive(env, m.gripper("right", close=True, rate=0.00225))
    targets = [0.05 * (c + 1.0) / 2.0 for c in cmds]
    ramp = [t for t in targets if t > 1e-9]
    assert len(ramp) == 7 and all(abs((ramp[i] - ramp[i + 1]) - 0.00225) < 1e-6 for i in range(len(ramp) - 1)), ramp
    assert abs(ramp[0] - (0.017925 - 0.00225)) < 1e-6 and cmds[len(ramp)] == -1.0 and cmds[-1] == -1.0
    assert float(robot.q[idx].max()) < 1e-6 and m.gripper_closed("right")
    # from fully open the ramp takes 22 steps (50 mm / 2.25 mm); with an object in reach the assisted grasp attaches during the
    # ramp and the closure finishes with the full command
    drive(env, m.gripper("right", close=False))
    cmds.clear()
    drive(env, m.gripper("right", close=True, rate=0.00225))
    assert len([t for t in (0.05 * (c + 1.0) / 2.0 for c in cmds) if t > 1e-9]) == 22
    drive(env, m.gripper("right", close=False))
    ep, _ = robot.eef_pose("right")
    env.objects["ball"] = FakeObject("ball_0", ep.tolist())
    cmds.clear()
    drive(env, m.gripper("right", close=True, rate=0.00225))
    # symmetric closure: the ramp stops when both pads touch the 8 cm ball (opening 41 mm), the command holds the fingers at
    # the contact width minus the 1 mm squeeze instead of driving to -1 through the object, the assisted grasp attaches there
    assert robot._ag_obj_in_hand["right"] is env.objects["ball"] and cmds[-1] > -1.0
    hold_m = 0.05 * (cmds[-1] + 1.0) / 2.0
    assert 0.038 <= hold_m <= 0.042 and all(c == cmds[-1] for c in cmds[-6:]), cmds[-8:]
    ev = m.release_events[-1]
    assert ev["event"] == "contact_hold" and abs(ev["hold_m"] - hold_m) < 1e-3 and ev["ramp_steps"] == 4 and ev["attached_after_steps"] >= 1 and "fallback" not in ev
    assert m.gripper_closed("right")
    # no squeeze configured: the stock fight (-1) after the ramp, as before
    drive(env, m.gripper("right", close=False))
    m.cfg.gripper_contact_squeeze_m = None
    cmds.clear()
    drive(env, m.gripper("right", close=True, rate=0.00225))
    assert robot._ag_obj_in_hand["right"] is env.objects["ball"] and cmds[-1] == -1.0
    m.cfg.gripper_contact_squeeze_m = 0.001
    # both pads on the object but the assisted grasp never attaches: the stock command takes over after the step budget
    drive(env, m.gripper("right", close=False))
    robot.ag_fail_next = 1
    cmds.clear()
    drive(env, m.gripper("right", close=True, rate=0.00225))
    assert robot._ag_obj_in_hand["right"] is None and cmds[-1] == -1.0 and m.release_events[-1].get("fallback") == "full_close"
    robot.ag_fail_next = 0
    # no rate (opening, or a push / brace closure): the stock step command alone
    drive(env, m.gripper("right", close=False))
    del env.objects["ball"]
    cmds.clear()
    drive(env, m.gripper("right", close=True))
    assert set(cmds) == {-1.0}
    env.after_step = None


def test_support_disturbance_fact_tracks_the_unmanipulated_supports(tmp_path):
    """Owner 2026-09-16 23:00 (fact, no verdict): ``meta.support_disturbance`` carries the pose drift of every tracked object
    no action manipulates (the table under a tabletop task) - max / end displacement and up-axis tilt - and, with a contact
    view, the finger-support contact steps by action and the peak force; without the simulator's contact view those are
    None. A world whose tracked objects are all manipulated reports nothing."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    table = FakeObject("breakfast_table_rhjoby_0", [7.0, 3.1, 0.40], category="breakfast_table", extent=(1.2, 0.8, 0.8))
    table.graspable = False
    table.fixed_base = False
    env.objects["breakfast_table.n.01_1"] = table
    q0 = table.quat.clone()

    def rock(step_action):
        # the table rocks 0.3 mm / 0.02 deg while the arm is at the place height, and settles back afterwards
        held = env.robot._ag_obj_in_hand["right"] is not None
        z = float(env.robot.eef_pose("right")[0][2])
        if held and z < 0.90:
            table.pos = th.tensor([7.0003, 3.1, 0.40])
            table.quat = C.qmul(q0, th.tensor([math.sin(math.radians(0.01)), 0.0, 0.0, math.cos(math.radians(0.01))]))
        else:
            table.pos, table.quat = th.tensor([7.0, 3.1, 0.40]), q0.clone()

    env.after_step = rock
    ex, *_ = make_executor(env, sc, tmp_path)
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    sd = result["support_disturbance"]
    assert sd["supports"] == ["breakfast_table.n.01_1"] and sd["fixed_base"] == {"breakfast_table.n.01_1": False}
    assert abs(sd["max_displacement_m"] - 0.0003) < 5e-5 and abs(sd["max_tilt_deg"] - 0.02) < 0.003, sd
    assert sd["end_displacement_m"] < 1e-6 and sd["end_tilt_deg"] < 1e-4
    assert sd["contact_steps"] is None and sd["peak_force_N"] is None and sd["by_action"] is None  # no contact view in the fake
    assert sd["steps_observed"] == result["n_steps"] and sd["force_sampling"]
    # every tracked object manipulated: nothing to report
    sc2, env2 = iors_world()
    env2.evaluate = success_eval("apple.n.01_1", _place_goal(sc2), tol=0.08)
    ex2, *_ = make_executor(env2, sc2, tmp_path / "b")
    result2 = ex2.run_program(expand(sc2.variant, sc2))
    assert result2["success"] and result2["support_disturbance"] == {}
    # the collector itself with a fake contact view: contact steps by action and the peak force
    class _View:
        def get_contact_force_matrix(self, dt):
            return th.tensor([[[0.0, 0.0, 12.5], [0.0, 0.0, 0.0]], [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]])

    class _API:
        _CONTACT_VIEW = {0: _View()}

        @staticmethod
        def get_contact_pairs(scene_idx, query, with_set, current_only):
            return {("finger1", "table")}

        @staticmethod
        def get_contact_row_indices(scene_idx, links):
            return th.tensor([0, 1])

        @staticmethod
        def get_contact_col_indices(scene_idx, links):
            return th.tensor([0])

    from ispatialgym.oracle.executor import SupportLoad
    import sys as _sys, types as _types
    og_stub = _types.ModuleType("omnigibson")
    og_stub.sim = SimpleNamespace(get_physics_dt=lambda: 1.0 / 120.0)
    robot = env.robot
    robot.finger_links = {a: [SimpleNamespace(prim_path=f"/f_{a}_{i}") for i in range(2)] for a in M.ARMS}
    table.links = {"base_link": SimpleNamespace(prim_path="/table/base_link")}
    sl = SupportLoad(robot, {"breakfast_table.n.01_1": table}, scene_idx=0)
    sl.contact_api = _API()
    saved = _sys.modules.get("omnigibson")
    _sys.modules["omnigibson"] = og_stub
    try:
        for name in ("grasp", "grasp", "place", None):
            sl.observe(name)
    finally:
        if saved is not None:
            _sys.modules["omnigibson"] = saved
        else:
            _sys.modules.pop("omnigibson", None)
    rep = sl.report()
    assert rep["contact_steps"] == 4 and rep["contact_steps_by_action"] == {"grasp": 2, "place": 1, "epilogue": 1}
    assert rep["peak_force_N"] == 12.5 and rep["peak_force_action"] == "grasp" and rep["by_action"]["place"] == {"contact_steps": 1, "peak_force_N": 12.5}


def test_epilogue_reports_the_predicate_branch_of_the_final_binary_evaluation(tmp_path):
    """Owner 2026-09-16 22:05 item 4 (D-007 wrapper, core 5555526): the executor resets the task's predicate branch log
    before each settle window and copies ``predicate_branches()`` of the final evaluation into the result (→
    ``meta.binary_branch``), naming the atoms a fallback branch (rays / geometry) passed; a task without the wrapper reports
    nothing."""
    from ispatialgym.sampling import predicates as PRED

    sc, env = iors_world()
    goal = _place_goal(sc)
    calls = {"reset": 0}

    def evaluate(env_):
        d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
        # the wrapper's log as the task evaluator fills it every step: omnigibson said no, the rays decided under(...)
        PRED._log("ontop", [env.objects["apple.n.01_1"], env.objects["plate.n.04_1"]], "omnigibson", False)
        PRED._log("ontop", [env.objects["apple.n.01_1"], env.objects["plate.n.04_1"]], "rays", True, {"hits": 3})
        return d

    env.evaluate = evaluate
    env.task.predicate_branches = staticmethod(PRED.decided_by)

    def reset():
        calls["reset"] += 1
        PRED.reset_branches()

    env.task.reset_predicate_branches = staticmethod(reset)
    PRED._log("under", [env.objects["plate.n.04_1"], env.objects["apple.n.01_1"]], "geometry", True)  # a stale entry from before the epilogue
    ex, *_ = make_executor(env, sc, tmp_path)
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    assert calls["reset"] >= 1
    assert result["binary_branch"] == {"ontop(apple_omzprq_0, plate_pkkgzc_0)": "rays"}  # the stale under(...) entry was reset away
    assert result["binary_branch_fallback"] == ["ontop(apple_omzprq_0, plate_pkkgzc_0)"]
    assert result["binary_branch_log"] and result["binary_branch_log"][-1]["branch"] == "rays" and result["binary_branch_log"][-1]["detail"] == {"hits": 3}
    PRED.reset_branches()
    # without the wrapper (a task that has no predicate_branches) the fields are empty
    sc2, env2 = iors_world()
    env2.evaluate = success_eval("apple.n.01_1", _place_goal(sc2), tol=0.08)
    ex2, *_ = make_executor(env2, sc2, tmp_path / "b")
    result2 = ex2.run_program(expand(sc2.variant, sc2))
    assert result2["success"] and result2["binary_branch"] == {} and result2["binary_branch_fallback"] == [] and result2["binary_branch_log"] == []


def test_every_release_opens_to_plus_one_at_once():
    """Owner 2026-10-10: every opening is exactly the open command (+1) in one step - no ramp, no staged or partial opening,
    nothing deferred. A held object, free or articulated, is released by that command; an empty gripper opens the same way;
    the checkpoints carry no release staging (an older checkpoint's staging keys are ignored)."""
    robot = FakeRobot()
    env = FakeEnv(robot, {})
    m = Motions(env, FakeBackend(robot), MotionsConfig(gripper_steps=30, settle_steps=5, cartesian_step=0.05, cartesian_rot_step=0.3))
    idx = robot.gripper_control_idx["right"]
    cmds = []
    env.after_step = lambda a: cmds.append(float(a[robot.controller_action_idx["gripper_right"]]))
    # (a) nothing held: from closed, every opening command is +1
    drive(env, m.gripper("right", close=True))
    cmds.clear()
    drive(env, m.gripper("right", close=False))
    assert cmds and set(cmds) == {1.0} and not m.gripper_closed("right") and abs(float(robot.q[idx].max()) - 0.05) < 1e-6
    # (b) a held free object, (c) an articulated flap: +1 from the first step of the opening, released by that command
    for key, name, joints in (("marker.n.03_1", "marker_xakoic_0", None), ("carton.n.02_1", "carton_flap_0", ["flap_joint"])):
        env.objects.clear()
        ep, _ = robot.eef_pose("right")
        obj = FakeObject(name, ep.tolist(), category="marker", extent=(0.1536, 0.0171, 0.0141), joints=joints)
        env.objects[key] = obj
        drive(env, m.gripper_opening("right", 0.5 * 0.01585 + 0.010))
        drive(env, m.gripper("right", close=True, rate=0.00225))
        assert robot._ag_obj_in_hand["right"] is obj and m.gripper_closed("right")
        seen = []
        env.after_step = lambda a: seen.append((float(a[robot.controller_action_idx["gripper_right"]]), robot._ag_obj_in_hand["right"]))
        drive(env, m.gripper("right", close=False))
        assert seen[0] == (1.0, None) and set(c for c, _ in seen) == {1.0}, name  # let go on the first step, at +1
        assert m.release_events[-1] == {"arm": "right", "event": "instant_open", "from_m": m.release_events[-1]["from_m"], "object": name}
        m.backend.detach_object("right")
    assert not any(e["event"] in ("staged", "full_open") for e in m.release_events)
    # (d) checkpoints: no staging state; an older checkpoint's staging keys load and change nothing
    st = m.export_state()
    assert not {"deferred_open", "release_opening", "preshape_opening"} & set(st)
    m.import_state({**st, "deferred_open": {"right": {"stage_m": 0.021, "steps": 0}}, "release_opening": {"right": 0.021}})
    cmds.clear()
    env.after_step = lambda a: cmds.append(float(a[robot.controller_action_idx["gripper_right"]]))
    drive(env, m.hold(5))
    assert set(cmds) == {1.0} and not m.gripper_closed("right")
    env.after_step = None


def test_place_releases_with_the_open_command_at_the_place_pose(tmp_path):
    """Owner 2026-10-03 / 2026-10-10: the place opens the gripper to exactly +1 in one step at the place pose and the
    assisted grasp lets go at that command, before the retreat - from the closure's contact hold straight to +1, no ramp,
    no staged or deferred opening; the release check counts one commanded release and no violation; the demo passes."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    robot = env.robot
    cmds = []
    env.after_step = lambda a: cmds.append(float(a[robot.controller_action_idx["gripper_right"]]))
    ex, motions, *_ = make_executor(env, sc, tmp_path)
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    place = result["actions"][3]
    events = [e["event"] for e in place["notes"].get("release") or []]
    assert place["name"] == "place" and events == ["instant_open"], events
    assert robot._ag_obj_in_hand["right"] is None and float(robot.q[robot.gripper_control_idx["right"]].max()) > 0.049
    i0 = min(range(len(cmds)), key=lambda i: cmds[i])
    assert -1.0 < cmds[i0] < 0.9  # the symmetric closure holds at the apple's contact width, never at -1
    rises = [(a, b) for a, b in zip(cmds[i0:-1], cmds[i0 + 1:]) if b > a]
    assert rises and all(b == 1.0 for _, b in rises), rises  # every opening after the hold is one step to +1
    assert not result["release_check"]["violations"]


def test_grasp_preshapes_from_the_library_pinch_width_and_closes_with_the_ramp(tmp_path):
    """``execute_grasp`` pre-shapes the opening to pinch width / 2 + ``gripper_preshape_margin`` (the apple's library grasp:
    73.4 mm -> 46.7 mm per finger) before the descent and closes with ``gripper_close_rate``; both are recorded in the
    action's notes; a cube / handle grasp or a library entry without a width keeps the full opening."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    cmds = []
    robot = env.robot
    env.after_step = lambda a: cmds.append(float(a[robot.controller_action_idx["gripper_right"]]))
    ex, motions, *_ = make_executor(env, sc, tmp_path)
    assert motions.cfg.gripper_close_rate == 0.00225 and motions.cfg.gripper_preshape_margin == 0.010
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    grasp = result["actions"][1]
    assert grasp["name"] == "grasp"
    assert abs(grasp["notes"]["preshape_opening_m"] - (0.5 * 0.0734 + 0.010)) < 1e-6 and grasp["notes"]["closure_rate_m_per_step"] == 0.00225
    # the command sequence: open (+1), the pre-shaped opening (0.868), the ramp down at <= 2.25 mm per step to the contact
    # hold (the symmetric closure stops at the apple's width: 4 cm + 2 mm - 1 mm squeeze), never -1
    pre = 2.0 * (0.5 * 0.0734 + 0.010) / 0.05 - 1.0
    assert any(abs(c - pre) < 1e-6 for c in cmds)
    i0 = next(i for i, c in enumerate(cmds) if abs(c - pre) < 1e-6)
    i1 = min(range(i0, len(cmds)), key=lambda i: cmds[i])
    hold_m = 0.05 * (cmds[i1] + 1.0) / 2.0
    assert 0.038 <= hold_m <= 0.042 and -1.0 not in cmds[i0:], (hold_m, cmds[i0:i1 + 3])
    steps = [0.05 * (a - b) / 2.0 for a, b in zip(cmds[i0:i1 - 1], cmds[i0 + 1:i1])]
    assert steps and all(-1e-9 <= s <= 0.00225 + 1e-9 for s in steps), steps
    assert grasp["notes"]["release"][0]["event"] == "contact_hold" and abs(grasp["notes"]["release"][0]["hold_m"] - hold_m) < 1e-3
    assert EX.grasp_pinch_width(expand(sc.variant, sc)[1]) == 0.0734
    # no width / not a library grasp -> no pre-shape
    from ispatialgym.oracle.actions import Grasp as GraspAction
    from ispatialgym.oracle.sidecar import GraspRecord
    g = sc.grasps[0]
    assert EX.grasp_pinch_width(GraspAction(kind="library", arm="right", object=g.object, grasp=GraspRecord(**{**{k: getattr(g, k) for k in GraspRecord.KNOWN if hasattr(g, k)}, "grasp_key": "no_such_key"}))) is None
    motions.cfg.gripper_preshape_margin = None
    cmds.clear()
    sc2, env2 = iors_world()
    env2.evaluate = success_eval("apple.n.01_1", _place_goal(sc2), tol=0.08)
    env2.after_step = lambda a: cmds.append(float(a[env2.robot.controller_action_idx["gripper_right"]]))
    ex2, motions2, *_ = make_executor(env2, sc2, tmp_path / "b")
    motions2.cfg.gripper_preshape_margin = None
    result2 = ex2.run_program(expand(sc2.variant, sc2))
    assert result2["success"] and "preshape_opening_m" not in result2["actions"][1]["notes"]
    i_close = next(i for i, c in enumerate(cmds) if c < 1.0 - 1e-6)  # the first command below fully open = the closure's first ramp step
    before_close = cmds[:i_close]  # the descent runs fully open
    assert not any(abs(c - pre) < 1e-6 for c in before_close) and before_close and set(before_close) == {1.0}


# ==========================================================================================
# the executor loop
# ==========================================================================================


def _place_goal(sc):
    g = sc.grasps[0]
    return [g.place.position[0], g.place.position[1], g.place.position[2] - 0.03]


def test_program_runs_end_to_end_with_retry_from_checkpoint(tmp_path):
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    env.robot.ag_fail_next = 1  # the first grasp closure does not attach: failed_grasp -> rollback -> retry
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    actions = expand(sc.variant, sc)
    assert [a.name for a in actions] == ["transition", "grasp", "lift", "place", "retreat"]
    result = ex.run_program(actions)
    assert result["success"], result["verdict"]
    grasp = result["actions"][1]
    assert grasp["name"] == "grasp" and len(grasp["attempts"]) == 2
    assert grasp["attempts"][0]["category"] == "failed_grasp" and grasp["attempts"][1] is None
    assert result["monitors"]["stats"]["by_category"] == {"failed_grasp": 1}
    assert result["monitors"]["stats"]["by_action"] == {"grasp": {"failed_grasp": 1}}
    # the checkpoint restored the simulator and the executor state; OmniGibson's own AG round-trip was fine here
    assert len(result["checkpoint_reports"]) == 1 and result["checkpoint_reports"][0]["ag_roundtrip_ok"] is True
    # planner attachment follows the grasp / release
    assert backend._attached == {}
    assert env.robot._ag_obj_in_hand["right"] is None
    # epilogue: exactly settle_steps no-op steps (50 in make_executor's configuration)
    assert result["epilogue_steps"] == 50 and result["n_settle_steps"] == 50
    # the recorder: the failed attempt was rolled back, the demo holds only committed steps
    assert rec.n == result["n_steps"] and rec.n_pending == 0
    assert len(rec.rollbacks) == 1 and rec.rollbacks[0]["action_index"] == 1 and rec.rollbacks[0]["verdict"]["category"] == "failed_grasp"
    assert len(rec.actions) == 5 and rec.actions[1].attempts == 2 and rec.actions[1].rolled_back[0]["category"] == "failed_grasp"
    assert all(a.fine for a in rec.actions)
    assert rec.actions[-1].step_end == result["n_action_steps"]


def test_grasp_plans_to_the_recorded_ik_witness_and_closes_the_residual(tmp_path):
    """Sidecars carry ``ik_witnesses`` (the program check's joint solutions per checked end pose); ``arm_to`` then plans
    in joint space to the witness — the IK branch the approach was validated from — and closes the residual to the
    pose with a straight segment (the pilot's nextto grasps failed the approach 9 mm short in cuRobo's own branch)."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    actions = expand(sc.variant, sc)
    grasp = actions[1]
    pre = grasp.targets(sc, ex.state)["pre_grasp"]
    q28 = env.robot.q.clone()
    q28[env.robot.arm_control_idx["right"]] = env.robot.ik("right", pre) + th.tensor([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0])  # 1 cm off: the residual is closed
    sc.extra["ik_witnesses"] = [{"action": 1, "name": "grasp", "stance": "manipulation", "arm": "right", "carried": None,
                                 "targets": {"pre_grasp": {"pose": pre.to_dict(), "q": q28.tolist()}}}]
    assert set(ex.witnesses_for(1)) == {"pre_grasp"} and ex.witnesses_for(0) == {}
    assert ex.run_action(0, actions[0]) is None
    n_after_transition = len(backend.calls)
    assert ex.run_action(1, grasp) is None
    calls = backend.calls[n_after_transition:]
    # residual + the approach check from where the transit left the arm (execute._into_witness_posture) + approach
    assert calls[0] == "plan_joint" and "plan_pose" not in calls and calls.count("solve_ik_path") == 3
    assert env.robot._ag_obj_in_hand["right"] is env.objects["apple.n.01_1"]
    # without a witness the pose goal is planned as before
    sc.extra["ik_witnesses"] = []
    sc2, env2 = iors_world()
    ex2, motions2, backend2, _ = make_executor(env2, sc2, tmp_path / "b")
    for i in range(2):
        assert ex2.run_action(i, expand(sc2.variant, sc2)[i]) is None
    assert "plan_pose" in backend2.calls


def test_execute_functions_toggle_the_gripper_in_view_span_around_the_checked_segment(tmp_path, monkeypatch):
    """FOV rule (b): the grasp opens the acting arm's span once the pre-grasp is reached; nothing touches it through the lift
    and the carry; the place closes it after the release (the executor wires ``ctx["hooks"]["fov"]`` to the retry rule's
    ``Monitors.set_fov_enabled`` and the QA log's ``QA.set_fov_enabled``). The carry leg is a straight Cartesian leg
    (``targets["carry"]``), not a trajopt."""
    quiet_qa(monkeypatch)
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, qa=True)
    actions = expand(sc.variant, sc)
    ex._init_qa(oracle=True, actions=actions)
    calls = []
    spans = []
    current = {"index": None}
    orig = ex.qa.set_fov_enabled

    def spy(enabled=True):
        calls.append((current["index"], bool(enabled)))
        orig(enabled)

    ex.qa.set_fov_enabled = spy
    plans_before_place = None
    for i, a in enumerate(actions[:4]):
        if i == 3:
            plans_before_place = sum(1 for c in backend.calls if c in ("plan_pose", "plan_joint"))
        current["index"] = i
        assert ex.run_action(i, a) is None, a.name
        spans.append(dict(ex.qa.fov_span))
        assert ex.monitors.fov_spans() == spans[-1]  # the retry rule's spans follow the same hook
    by_action = {}
    for idx, en in calls:
        by_action.setdefault(idx, []).append(en)
    assert by_action[1] == [False, True]  # grasp: transit to the pre-grasp unwatched, the span opens at the pre-grasp
    assert 2 not in by_action  # lift: the span stays open, no hook
    assert by_action[3] == [False]  # place: no hook before the carry / descent, the span closes after the release
    assert spans[1]["right"] and spans[2]["right"] and not spans[3]["right"]  # open after the grasp and the lift, closed after the place
    assert not any(ex.qa.fov_span.values())  # no span open after the release
    place_notes = ex.action_records[3].notes
    assert place_notes["carry_leg"]["kind"] == "straight" and place_notes["carry_leg"]["waypoints"] >= 3
    if place_notes["carry_leg"]["finish"] is None:  # the carry is differential IK: no trajopt in the place at all
        assert sum(1 for c in backend.calls if c in ("plan_pose", "plan_joint")) == plans_before_place


def test_transition_moves_the_idle_arm_to_its_recorded_posture_before_the_trunk(tmp_path):
    """A6: the arriving stance records the idle arm's joint values (``arms_q``) / posture (``idle_posture``); the
    oracle moves that arm there at the stand-off, before the trunk motion (the sampler swept the trunk with it)."""
    sc, env = iors_world()
    manip = sc.stance("manipulation")
    assert manip.working_arms == ["right"]
    left_q = [0.9, 0.3, -0.7, -1.2, 0.5, 0.2, 0.8]
    manip.extra["arms_q"] = {"left": left_q, "right": env.robot.q[env.robot.arm_control_idx["right"]].tolist()}
    manip.extra["idle_posture"] = "ready"
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    transition = expand(sc.variant, sc)[0]
    assert transition.name == "transition"
    targets = EX.idle_arm_targets(manip, ex.state)
    assert set(targets) == {"left"} and th.allclose(targets["left"], th.tensor(left_q))
    assert ex.run_action(0, transition) is None
    rows = rec.rows
    q = np.asarray(rows["joint_positions"])
    left_idx = env.robot.arm_control_idx["left"].numpy()
    trunk_idx = env.robot.trunk_control_idx.numpy()
    first_arm_move = int(np.argmax(np.abs(q[:, left_idx] - q[0, left_idx]).max(axis=1) > 1e-4))
    first_trunk_move = int(np.argmax(np.abs(q[:, trunk_idx] - q[0, trunk_idx]).max(axis=1) > 1e-4))
    assert 0 < first_arm_move < first_trunk_move
    assert th.allclose(env.robot.q[env.robot.arm_control_idx["left"]], th.tensor(left_q), atol=1e-4)
    assert ex.action_records[0].attempts == [None]
    # the working arm is never moved by the transition, and an arm already at its posture is left alone
    assert th.allclose(env.robot.q[env.robot.arm_control_idx["right"]], th.from_numpy(q[0][env.robot.arm_control_idx["right"].numpy()]), atol=1e-6)
    ctx = {}
    list(EX.move_idle_arms(motions, ex.state, manip, ctx))
    assert ctx["idle_arms_moved"] == [] and ctx["fingers_set"] is None
    # item 5: the stance's recorded finger opening (closed pads for the wall two-arm stance) is applied with arms_q on
    # every arm that holds nothing; an arm carrying an object keeps its grasp
    manip.extra["fingers"] = 0.0
    assert EX.stance_fingers(manip) == 0.0
    assert not motions.gripper_closed("left") and not motions.gripper_closed("right")
    ctx = {}
    for a in EX.move_idle_arms(motions, ex.state, manip, ctx):
        env.step(a)
    assert ctx["fingers_set"] == {"arms": ["left", "right"], "closed": True} and motions.gripper_closed("left") and motions.gripper_closed("right")
    assert float(env.robot.q[env.robot.gripper_control_idx["left"]].max()) < 0.01
    ctx = {}
    list(EX.move_idle_arms(motions, ex.state, manip, ctx))
    assert ctx["fingers_set"] == {"arms": [], "closed": True}  # already closed: nothing to do
    manip.extra["fingers"] = 0.05
    env.robot._ag_obj_in_hand["right"] = env.objects["apple.n.01_1"]  # the right hand carries: its grasp is kept
    ctx = {}
    for a in EX.move_idle_arms(motions, ex.state, manip, ctx):
        env.step(a)
    assert ctx["fingers_set"] == {"arms": ["left"], "closed": False} and not motions.gripper_closed("left") and motions.gripper_closed("right")
    env.robot._ag_obj_in_hand["right"] = None
    del manip.extra["fingers"]
    # owner 10:47: a two-arm stance (both arms working, ``arm_posture: tucked`` + arms_q for both) puts BOTH arms into
    # the recorded posture before the drive-in; an arm carrying an object keeps its grasp and its joints
    manip.arms = ["left", "right"]
    manip.extra["arm_posture"] = "tucked"
    manip.extra["arms_q"] = {"left": [0.0] * 7, "right": [0.0] * 7}
    manip.extra.pop("idle_posture", None)
    assert set(EX.stance_arm_targets(manip, ex.state)) == {"left", "right"} and EX.idle_arm_targets(manip, ex.state) == {}
    ctx = {}
    for a in EX.move_idle_arms(motions, ex.state, manip, ctx):
        env.step(a)
    assert ctx["arms_posture"] == {"moved": ["left", "right"], "kept_holding": [], "source": "arms_q"}
    assert float(env.robot.q[env.robot.arm_control_idx["left"]].abs().max()) < 1e-3 and float(env.robot.q[env.robot.arm_control_idx["right"]].abs().max()) < 1e-3
    env.robot.q[env.robot.arm_control_idx["right"]] = th.tensor([0.5] * 7)
    env.robot._ag_obj_in_hand["right"] = env.objects["apple.n.01_1"]
    ctx = {}
    for a in EX.move_idle_arms(motions, ex.state, manip, ctx):
        env.step(a)
    assert ctx["arms_posture"] == {"moved": [], "kept_holding": ["right"], "source": "arms_q"}
    assert float(env.robot.q[env.robot.arm_control_idx["right"]].min()) > 0.4
    env.robot._ag_obj_in_hand["right"] = None
    # without arms_q the posture name applies to every arm
    del manip.extra["arms_q"]
    ctx = {}
    for a in EX.move_idle_arms(motions, ex.state, manip, ctx):
        env.step(a)
    assert ctx["arms_posture"]["moved"] == ["right"] and ctx["arms_posture"]["source"] == "tucked"
    assert float(env.robot.q[env.robot.arm_control_idx["right"]].abs().max()) < 1e-3
    # a posture name alone works too, and a posture-string stance (init) treats both arms as idle
    init = sc.stance("init")
    init.extra["idle_posture"] = "tucked"
    both = EX.idle_arm_targets(init, ex.state)
    assert set(both) == {"left", "right"} and float(both["right"].abs().max()) == 0.0
    init.extra["idle_posture"] = "ready"
    assert th.allclose(EX.idle_arm_targets(init, ex.state)["left"], th.tensor(M.ARM_READY_Q["left"]))
    del init.extra["idle_posture"]
    assert EX.idle_arm_targets(init, ex.state) == {}


def test_second_failure_fails_the_episode_with_the_action_category(tmp_path):
    sc, env = iors_world()
    env.robot.ag_fail_next = 5
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    result = ex.run_program(expand(sc.variant, sc))
    assert not result["success"] and result["reason"] == "failed_grasp"
    assert result["verdict"]["action"] == "grasp" and result["verdict"]["attempt"] == 1
    assert [a["name"] for a in result["actions"]] == ["transition", "grasp"]
    assert len(result["actions"][1]["attempts"]) == 2
    assert result["monitors"]["stats"]["by_action"]["grasp"]["failed_grasp"] == 2
    # the epilogue still ran its no-op window; the failed attempt stays in the record for diagnosis
    assert result["n_settle_steps"] >= 50
    assert len(rec.rollbacks) == 1 and rec.actions[1].verdict["category"] == "failed_grasp"


def test_plan_failed_is_retried_and_counted_separately(tmp_path):
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    # the pre-grasp transit tries two plans per attempt (2026-09-20: with the supports present, then the former plan with them
    # ignored as the fallback): both trajopts of the first attempt fail, the retry's plan succeeds
    backend.fail_plans = 2
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    assert result["monitors"]["stats"]["by_category"] == {"plan_failed": 1}
    assert result["actions"][1]["attempts"][0]["details"]["stage"] == "trajopt"
    assert result["actions"][1]["notes"]["pre_grasp_transit"] == "supports present"


def test_qa_monitor_verdict_propagates_and_is_retried(tmp_path):
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    table = FakeObject("table_0", [7.2, 3.1, 0.4], category="breakfast_table")
    env.scene.objects_by_name[table.name] = table
    hits = {"n": 0}

    def script(step, query):
        # the wrist touches the table for 3 steps during the first attempt of the transition only
        if 5 <= step <= 7:
            hits["n"] += 1
            return [(f"{env.robot.prim_path}/right_arm_link6", f"{table.prim_path}/base_link")]
        return []

    ex, motions, backend, rec = make_executor(env, sc, tmp_path, contacts=FakeContacts(script))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    assert result["actions"][0]["attempts"][0]["category"] == "collision"
    assert result["actions"][0]["attempts"][0]["details"]["object"] == "table_0"
    assert result["monitors"]["stats"]["by_action"] == {"transition": {"collision": 1}}
    assert result["monitors"]["collisions"]["by_object"] == {"table_0": 3}
    assert hits["n"] == 3


def test_intended_contacts_and_finger_links_are_exempt(tmp_path):
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    apple, plate = env.objects["apple.n.01_1"], env.objects["plate.n.04_1"]
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    seen = {"manip": 0}

    def script(step, query):
        # during every manipulation action: the acting fingers on the apple (intended anyway) and the wrist on the
        # plate (intended for place / retreat only — but the finger rule is what exempts the finger link); while
        # carrying (lift) the held apple touches the plate: held-object contacts are not judged in manipulation
        if ex.monitors.manipulation:
            seen["manip"] += 1
            return [
                (f"{env.robot.prim_path}/right_gripper_finger_link1", f"{apple.prim_path}/base_link"),
                (f"{env.robot.prim_path}/right_gripper_finger_link2", f"{plate.prim_path}/base_link"),
                (f"{apple.prim_path}/base_link", f"{plate.prim_path}/base_link"),
            ]
        return []

    ex.monitors._contact_api = FakeContacts(script)
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    assert result["monitors"]["stats"]["by_category"] == {} and seen["manip"] > 0


def test_grasp_may_touch_objects_must_stay_on_the_source_tray():
    """Owner 2026-10-12 (A2b): disembedding's grasp may touch the parts / distractors of ``grasps[].may_touch`` (intended contacts of
    the Grasp); after the grasp each must still lie over the source tray's footprint - the tray's own yawed box, not its world AABB -
    and after the lift it must not have risen more than 2 cm with the target (``distractor_displaced``); no record, no check."""
    d = json.load(open(os.path.join(SIDECARS, "disembedding.json")))
    d["grasps"][0]["may_touch"] = {"objects": ["box_of_cereal.n.01_1", "box_of_cereal.n.01_2"], "stay_on": "tray.n.01_1"}
    sc = Sidecar.from_dict(d)
    actions = expand(sc.variant, sc)
    grasp, lift = actions[1], actions[2]
    assert (grasp.name, lift.name) == ("grasp", "lift")
    assert set(EX.intended_contacts(grasp)) == {"bottle_of_beer.n.01_1", "box_of_cereal.n.01_1", "box_of_cereal.n.01_2"}
    assert "distractor_displaced" in QA.CATEGORIES

    yaw = math.pi / 4
    R = lambda x, y: (1.0 + x * math.cos(yaw) - y * math.sin(yaw), 2.0 + x * math.sin(yaw) + y * math.cos(yaw))  # tray frame -> world xy

    class Tray:
        aabb = (th.tensor([0.66, 1.66, 0.79]), th.tensor([1.34, 2.34, 0.81]))  # the yawed 0.565 x 0.396 m tray's world AABB

        def get_base_aligned_bbox(self, xy_aligned=False):
            return th.tensor([1.0, 2.0, 0.8]), C.quat_from_yaw(yaw), th.tensor([0.565, 0.396, 0.02]), th.zeros(3)

    def box(x, y, z=0.85):
        return SimpleNamespace(aabb=(th.tensor([x - 0.03, y - 0.03, z - 0.05]), th.tensor([x + 0.03, y + 0.03, z + 0.05])), pos=th.tensor([x, y, z]))

    target = box(*R(0.1, 0.0))
    objects = {"tray.n.01_1": Tray(), "bottle_of_beer.n.01_1": target, "box_of_cereal.n.01_1": box(*R(0.25, 0.0)), "box_of_cereal.n.01_2": box(*R(-0.2, 0.1))}
    state = SimpleNamespace(object=lambda n: objects[n], held_object=lambda arm: target, object_pose=lambda n: (objects[n].pos, C.IDENTITY_QUAT.clone()),
                            init_poses={n: (o.pos.clone(), C.IDENTITY_QUAT.clone()) for n, o in objects.items() if hasattr(o, "pos")})
    ctx: Dict[str, Any] = {}
    assert EX.post_grasp(grasp, state, {}, ctx) is None
    assert ctx["may_touch"]["displaced"] == [] and ctx["may_touch"]["objects"]["box_of_cereal.n.01_1"] == {"over_support": True, "moved_m": 0.0}
    # pushed 25 cm across the tray's short axis: inside its world AABB, off its footprint
    objects["box_of_cereal.n.01_2"] = box(*R(0.0, 0.25))
    assert EX.rests_over(objects["box_of_cereal.n.01_2"], SimpleNamespace(aabb=Tray.aabb)) is True
    ctx = {}
    assert EX.post_grasp(grasp, state, {}, ctx) == "distractor_displaced" and ctx["may_touch"]["displaced"] == ["box_of_cereal.n.01_2"]
    objects["box_of_cereal.n.01_2"] = box(*R(-0.2, 0.1))
    # the lift: one part rose 6 cm with the target (still over the tray), the other 1 cm (jostled, settled back)
    ctx = {"may_touch_z0": {"box_of_cereal.n.01_1": 0.85, "box_of_cereal.n.01_2": 0.85}}
    objects["box_of_cereal.n.01_1"] = box(*R(0.25, 0.0), z=0.91)
    objects["box_of_cereal.n.01_2"] = box(*R(-0.2, 0.1), z=0.86)
    assert EX.may_touch_verdict(lift, state, ctx, "lift") == "distractor_displaced" and ctx["may_touch"]["displaced"] == ["box_of_cereal.n.01_1"]
    assert ctx["may_touch"]["objects"]["box_of_cereal.n.01_1"]["rise_m"] == pytest.approx(0.06)
    # no record (every other task, older sidecars): nothing is checked
    plain = expand(sc.variant, Sidecar.load(os.path.join(SIDECARS, "disembedding.json")))[1]
    assert EX.post_grasp(plain, state, {}, {}) is None and EX.may_touch_objects(state, plain) == {}
    # the pre-grasp's own overlaps (``may_touch.pre_grasp``, the puzzle's fixed-layout parts): supports of the transit to the pre-grasp
    assert EX.may_touch_objects(state, grasp, pre_grasp=True) == {}
    d["grasps"][0]["may_touch"]["pre_grasp"] = ["box_of_cereal.n.01_2"]
    puzzle = expand(sc.variant, Sidecar.from_dict(d))[1]
    assert list(EX.may_touch_objects(state, puzzle, pre_grasp=True)) == ["box_of_cereal.n.01_2"]
    assert list(EX.may_touch_objects(state, puzzle)) == ["box_of_cereal.n.01_1", "box_of_cereal.n.01_2"]


def inside_world():
    """The IORS inside test sidecar as a fake world: robot at the init stance, sprout on the table, bowl at the drop."""
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-inside_small_scale.json"))
    init = sc.stance("init")
    robot = FakeRobot(base=tuple(init.base_pose2d))
    g = sc.grasps[0]
    sprout = FakeObject("brussels_sprouts_mbkrxe_0", [g.grasp.position[0], g.grasp.position[1], g.grasp.position[2] - 0.03], category="brussels_sprouts", extent=(0.04, 0.04, 0.04))
    bowl = FakeObject("bowl_belcml_0", [g.place.position[0], g.place.position[1], g.place.position[2] - 0.12], category="bowl", extent=(0.2, 0.2, 0.08))
    objects = {"brussels_sprouts.n.01_1": sprout, "bowl.n.01_1": bowl}
    env = FakeEnv(robot, objects)
    return sc, env


def test_push_under_contacts_floor_and_target_allowed_partner_forbidden():
    """Item 7: for the push-under push the floor and the target are intended contacts, the partner is forbidden
    (fingers / held object included); the preceding drop on the floor forbids the partner too."""
    sc = Sidecar.load(os.path.join(VARIANTS, "inter_object_relation_satisfaction-under_large_scale__push_under.json"))
    actions = expand(sc.variant, sc)
    place = [a for a in actions if isinstance(a, Place)][0]
    push = [a for a in actions if isinstance(a, Push)][0]
    assert push.kind == "push_under"
    assert set(EX.intended_contacts(push)) == {"briefcase.n.01_1", "floor.n.01_1"} and EX.forbidden_contacts(push) == ("breakfast_table.n.01_1",)
    assert EX.intended_contacts(place) == ("briefcase.n.01_1",) and set(EX.forbidden_contacts(place)) == {"floor.n.01_1", "breakfast_table.n.01_1"}
    # the geometry of the recorded straight floor push: a line, 25 cm along +y
    geom = EX.push_geometry(push.targets(sc, SimpleNamespace(object_pose=lambda n: None))["waypoints"], push.contact.direction)
    assert geom.kind == "line" and abs(geom.distance - 0.25) < 1e-6


def test_drop_place_forbids_any_contact_with_the_container(tmp_path):
    """Owner decision 2026-09-16 (IORS inside): the drop releases at the recorded place pose above the rim; the
    container is not an intended contact — a finger (exempt link during manipulation) or the held object touching
    the bowl is a ``collision`` verdict, never a silent success; a support place keeps the partner intended."""
    sc, env = inside_world()
    actions = expand(sc.variant, sc)
    place = [a for a in actions if isinstance(a, Place)][0]
    assert place.kind == "drop" and place.on == "bowl.n.01_1"
    assert EX.intended_contacts(place) == ("brussels_sprouts.n.01_1",) and EX.forbidden_contacts(place) == ("bowl.n.01_1",)
    support = Place(kind="support", arm="right", object="a.n.01_1", place=place.place, on="b.n.01_1")
    assert EX.intended_contacts(support) == ("a.n.01_1", "b.n.01_1") and EX.forbidden_contacts(support) == ()
    # (a) the opening fingers touch the bowl during the drop -> collision (the finger exemption does not apply)
    sprout, bowl = env.objects["brussels_sprouts.n.01_1"], env.objects["bowl.n.01_1"]
    env.evaluate = success_eval("brussels_sprouts.n.01_1", [float(v) for v in place.place.position], tol=0.15)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, qa_cfg=QA.QAConfig(stall_steps=100000, fov_min_steps=100000, collision_min_steps=2))

    def fingers_on_bowl(step, query):
        if ex.monitors.action is not None and ex.monitors.action.name == "place":
            return [(f"{env.robot.prim_path}/right_gripper_finger_link1", f"{bowl.prim_path}/base_link")]
        return []

    ex.monitors._contact_api = FakeContacts(fingers_on_bowl)
    result = ex.run_program(actions)
    assert result["reason"] == "collision" and result["verdict"]["details"]["object"] == bowl.name and result["verdict"]["action"] == "place"
    # (b) the held sprout brushing the bowl on the way down is a collision too (held-object contacts with the
    # forbidden container are judged even during manipulation)
    sc, env = inside_world()
    sprout, bowl = env.objects["brussels_sprouts.n.01_1"], env.objects["bowl.n.01_1"]
    env.evaluate = success_eval("brussels_sprouts.n.01_1", [float(v) for v in place.place.position], tol=0.15)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path / "b", qa_cfg=QA.QAConfig(stall_steps=100000, fov_min_steps=100000, collision_min_steps=2))

    def held_on_bowl(step, query):
        if ex.monitors.action is not None and ex.monitors.action.name == "place" and env.robot._ag_obj_in_hand["right"] is sprout:
            return [(f"{sprout.prim_path}/base_link", f"{bowl.prim_path}/base_link")]
        return []

    ex.monitors._contact_api = FakeContacts(held_on_bowl)
    result = ex.run_program(actions)
    assert result["reason"] == "collision" and result["verdict"]["details"]["link"] == "held:base_link"
    # (c) without contacts the drop succeeds: release at the recorded pose, retreat along -a, no settle-on-support check
    sc, env = inside_world()
    env.evaluate = success_eval("brussels_sprouts.n.01_1", [float(v) for v in place.place.position], tol=0.15)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path / "c")
    result = ex.run_program(actions)
    assert result["success"], result["verdict"]
    rec_place = result["actions"][[a.name for a in actions].index("place")]
    assert rec_place["attempts"] == [None] and "object_speed" not in rec_place["notes"]


def test_targets_failed_and_error_categories(tmp_path):
    sc, env = iors_world()
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    actions = expand(sc.variant, sc)

    class Broken(type(actions[1])):
        def targets(self, sidecar, state):
            raise RuntimeError("no such grasp")

    broken = Broken(kind="library", arm="right", object="apple.n.01_1", grasp=sc.grasps[0])
    result = ex.run_program([actions[0], broken])
    assert result["reason"] == "targets_failed"
    assert result["actions"][1]["attempts"][0]["category"] == "targets_failed"


def test_epilogue_verdicts_and_goal_image(tmp_path):
    # 1) the goal never holds -> not_reached, no goal image
    sc, env = iors_world()
    ex, *_ = make_executor(env, sc, exec_cfg=ExecutorConfig(goal_image=True))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["reason"] == "not_reached" and result["goal_observation"] is None and env.sim.renders == 0
    # 2) the goal holds but a measured target is outside its tolerance -> symbolic_only (symbolic, no geometric success)
    sc, env = iors_world()
    goal = _place_goal(sc)

    def eval_tol(env_):
        d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
        d["errors"]["pose_errors"]["plate.n.04_1"] = {"position_error": 0.5, "within_tolerance": False}
        return d

    env.evaluate = eval_tol
    ex, *_ = make_executor(env, sc, exec_cfg=ExecutorConfig(goal_image=True))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["reason"] == "symbolic_only" and result["verdict"]["details"]["outside"] == ["(pose_reached ?plate.n.04_1)"]
    assert result["error_metric"] == {"position_error": 0.5, "orientation_error": None, "case": "poses"} and result["goal_observation"] is None
    # 3) the goal holds, the evaluator's pose_settled never does -> not_settled (settle-eval on)
    sc, env = iors_world()

    def eval_unsettled(env_):
        d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
        d["settled"] = False
        return d

    env.evaluate = eval_unsettled
    ex, *_ = make_executor(env, sc, exec_cfg=ExecutorConfig(goal_image=True))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["reason"] == "not_settled" and "never through a settled window of 45 steps" in result["verdict"]["message"]
    # 3b) the goal held a few steps, then not any more -> not_settled
    sc, env = iors_world()
    hits = {"n": 0}

    def eval_transient(env_):
        d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
        if d["binary_success"]:
            hits["n"] += 1
            if hits["n"] > 3:
                d["binary_success"] = False
        return d

    env.evaluate = eval_transient
    ex, *_ = make_executor(env, sc)
    result = ex.run_program(expand(sc.variant, sc))
    assert result["reason"] == "not_settled" and result["goal_first_step"] is not None
    # 4) success: the goal image is captured after >= 3 renders with the physics-synced camera pose
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", goal, tol=0.08)
    ex, *_ = make_executor(env, sc, exec_cfg=ExecutorConfig(goal_image=True, goal_image_renders=10))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"] and env.sim.renders == 10
    go = result["goal_observation"]
    assert go["rgb"].shape == (8, 8, 3) and go["sensor_key"] == CAM_KEY and go["n_renders"] == 10
    cam_p, _ = env.robot.sensors[CAM_KEY].get_position_orientation()
    assert go["camera_pose_world"]["position"] == pytest.approx(cam_p.tolist(), abs=1e-6)
    assert go["intrinsics"]["fx"] == pytest.approx(459.0, abs=1.0)


def test_replay_stream_holds_the_last_streamed_configuration(tmp_path):
    """Replay mode: the demo's action rows and its settle rows are streamed (no epilogue: the settle rows, recorded as settle
    steps, are the window the metrics read), and nothing re-applies the Layer-1 hold targets captured at construction (which
    would drive the robot back to its initial pose)."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    # record the oracle program's actions and its settle window, then replay them through a fresh executor on a fresh world
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"]
    actions = [np.asarray(a) for a in rec.rows["action"][: result["n_steps"]]]
    n_action = result["n_action_steps"]
    sc2, env2 = iors_world()
    env2.evaluate = success_eval("apple.n.01_1", _place_goal(sc2), tol=0.08)
    ex2, motions2, *_ = make_executor(env2, sc2, tmp_path / "replay")
    ex2.stream_phases = [PHASE_ORACLE] * n_action + [PHASE_SETTLE] * (len(actions) - n_action)
    q_init = env2.robot.q.clone()
    result2 = ex2.run_stream(th.as_tensor(a) for a in actions)
    assert result2["success"], result2["verdict"]
    assert result2["n_steps"] == len(actions) and result2["n_settle_steps"] == 50 and result2["epilogue_steps"] == 0
    # the arm stayed where the stream left it (the initial hold targets were not re-applied)
    last = th.as_tensor(actions[-1])
    idx = env2.robot.controller_action_idx
    assert th.allclose(env2.robot.q[env2.robot.arm_control_idx["right"]], last[idx["arm_right"]], atol=1e-5)
    assert not th.allclose(env2.robot.q[env2.robot.arm_control_idx["right"]], q_init[env2.robot.arm_control_idx["right"]], atol=1e-3)
    assert th.equal(env2.last_action[idx["base"]], th.zeros(3))
    assert result2["monitors"]["stats"]["by_category"] == {}


def test_replay_stream_logs_its_qa_metrics_and_the_goal_decides(tmp_path):
    """A replay / policy stream only logs its ``--qa`` metrics (owner 2026-10-14): a robot link on an object during the replay
    is in ``result.qa`` with its steps (consecutive steps one run), the stream goes on and the goal decides."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"]
    actions = [np.asarray(a) for a in rec.rows["action"][: result["n_action_steps"]]]
    sc2, env2 = iors_world()
    env2.evaluate = success_eval("apple.n.01_1", _place_goal(sc2), tol=0.08)
    P = env2.robot.prim_path
    contacts = FakeContacts(lambda step, query: [(f"{P}/base_link", "/World/plate_pkkgzc_0/base_link")] if 5 <= step <= 8 else [])
    ex2, *_ = make_executor(env2, sc2, tmp_path / "log", contacts=contacts, qa=True)
    r2 = ex2.run_stream(th.as_tensor(a) for a in actions)
    assert r2["success"], r2["verdict"]
    qa = r2["qa"]
    assert qa["mode"] == "stream" and qa["stopped_by"] is None and "robot_touching_objects" in qa["hard_triggered"]
    assert qa["metrics"]["robot_touching_objects"]["runs"] == [{"link": "base_link", "object": "plate_pkkgzc_0", "action": None,
                                                                "first_step": 5, "last_step": 8, "n": 4}]
    assert qa["metrics"]["gripper_in_view"] == {"measured": False, "why": "oracle only", "mode": "soft", "triggered": False}


def test_checkpoint_restore_fixes_a_broken_ag_roundtrip(tmp_path):
    sc, env = iors_world(ag_roundtrip=False)
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    actions = expand(sc.variant, sc)
    # run transition + grasp, then checkpoint with the apple in hand, drop it, restore
    for i in range(2):
        assert ex.run_action(i, actions[i]) is None
    apple = env.objects["apple.n.01_1"]
    assert env.robot._ag_obj_in_hand["right"] is apple and "right" in backend._attached
    cp = ex.checkpoint(2)
    env.robot.release_grasp_immediately("right")
    backend.detach_object("right")
    motions.state.gripper_cmd["right"] = 1.0
    report = ex.restore(cp)
    # the fake sim does not carry the AG bookkeeping: the executor noticed, re-created the joint and re-attached
    assert report["ag_roundtrip_ok"] is False and report["ag_fixed"][0] == {"arm": "right", "restored": None, "checkpoint": apple.name}
    assert env.robot._ag_obj_in_hand["right"] is apple
    assert report["attached_fixed"] == [{"arm": "right", "attached": apple.name}] and backend._attached["right"]["obj"] is apple
    # the restored gripper command is the grasp's contact hold (the symmetric closure holds at the apple's width, not -1)
    hold_cmd = motions.state.gripper_cmd["right"]
    assert -1.0 < hold_cmd < 0.9 and motions.gripper_closed("right") and abs(0.05 * (hold_cmd + 1.0) / 2.0 - 0.040) < 0.003
    # the counters are rolled back, then the one recorded hold step re-applies the restored drive targets
    assert ex.n == cp.step + 1 and env._current_step == cp.env_step + 1
    assert th.allclose(env.last_action[env.robot.controller_action_idx["gripper_right"]], th.tensor([hold_cmd]))
    # and with a faithful round-trip the report says so
    sc2, env2 = iors_world(ag_roundtrip=True)
    ex2, motions2, backend2, _ = make_executor(env2, sc2)
    for i in range(2):
        assert ex2.run_action(i, expand(sc2.variant, sc2)[i]) is None
    cp2 = ex2.checkpoint(2)
    env2.robot.release_grasp_immediately("right")
    report2 = ex2.restore(cp2)
    assert report2["ag_roundtrip_ok"] is True and report2["ag_fixed"] == []


# ==========================================================================================
# QA monitors
# ==========================================================================================


def test_camera_model_projection_and_fov_margin():
    cam = QA.CameraModel.from_hfov(1080, 1080, 99.2)
    assert cam.fx == pytest.approx(459.5, abs=0.5)
    pose = (th.zeros(3), C.IDENTITY_QUAT.clone())  # looks along -z
    u, v, d = cam.project(pose, th.tensor([0.0, 0.0, -1.0]))
    assert (u, v, d) == pytest.approx((540.0, 540.0, 1.0))
    assert cam.in_frame(pose, th.tensor([0.0, 0.0, -1.0]))
    assert not cam.in_frame(pose, th.tensor([0.0, 0.0, 1.0]))  # behind the camera
    edge = math.tan(math.radians(99.2) / 2.0) * 1.0
    assert not cam.in_frame(pose, th.tensor([0.97 * edge, 0.0, -1.0]), margin=0.05)  # inside the frame but in the 5 % margin
    assert cam.in_frame(pose, th.tensor([0.8 * edge, 0.0, -1.0]), margin=0.05)


def test_fov_stalled_and_held_object_drift_rules():
    robot = FakeRobot()
    env = FakeEnv(robot, {})
    mon = QA.Monitors(env, robot, cfg=QA.QAConfig(fov_min_steps=3, stall_steps=4), camera=QA.CameraModel.from_hfov())
    mon._contact_api = FakeContacts(lambda s, q: [])
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-ontop_small_scale.json"))
    actions = expand(sc.variant, sc)
    grasp, lift, retreat = actions[1], actions[2], actions[-1]
    # rule (b): the monitor runs during every action; the spans decide the arms (a retreat after a release has none)
    assert EX.fov_monitored(grasp) and EX.fov_monitored(retreat) and EX.is_manipulation(retreat)
    out_q = th.tensor([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # the right eef far behind the robot: outside the frame
    in_q = th.tensor([0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0])  # in front of the robot: in frame
    robot.dq = th.ones(28)
    # (a) the transit to the pre-grasp (no span open yet): the idle posture out of frame is not a verdict, only
    # fov_idle_out events
    robot.q[robot.arm_control_idx["right"]] = out_q
    mon.begin_action(grasp, 1, 0, intended={"apple_omzprq_0"}, held=[])
    mon.set_fov_enabled(False)
    assert [mon.check(i) for i in range(1, 8)] == [None] * 7
    assert mon.stats.events["fov_idle_out"] == 7 and mon.stats.by_category == {}
    # (b) the span opens at the pre-grasp and is armed unconditionally: a leg whose fingertips never enter the frame
    # is a verdict after fov_min_steps, with fov_first_entry_step null and the out-of-frame step count
    mon.set_fov_enabled(True)
    verdicts = [mon.check(i) for i in range(8, 11)]
    assert verdicts[:2] == [None, None] and verdicts[2].category == "gripper_out_of_fov"
    assert verdicts[2].details["fov_first_entry_step"] is None and verdicts[2].details["fov_out_steps"] == 3 and "never in the head camera frame" in verdicts[2].message
    mon.end_action()
    # (b') in frame at the pre-pose, then leaving it: verdict after fov_min_steps with the entry step recorded
    mon.begin_action(grasp, 1, 1, intended=set(), held=[])
    robot.q[robot.arm_control_idx["right"]] = in_q
    assert mon.check(1) is None and mon._fov_first_entry["right"] == 1
    robot.q[robot.arm_control_idx["right"]] = out_q
    verdicts = [mon.check(i) for i in range(2, 5)]
    assert verdicts[:2] == [None, None] and verdicts[2].category == "gripper_out_of_fov" and verdicts[2].details["arm"] == "right"
    assert verdicts[2].details["fov_first_entry_step"] == 1 and verdicts[2].details["fov_out_steps"] == 3
    mon.end_action()
    # (b'') disabling before the retreat leg (after the release) stops the monitoring; a brief excursion while enabled
    # that returns within fov_min_steps is no verdict
    mon.begin_action(grasp, 1, 1, intended=set(), held=[])
    robot.q[robot.arm_control_idx["right"]] = in_q
    mon.check(1)
    robot.q[robot.arm_control_idx["right"]] = out_q
    assert [mon.check(i) for i in range(2, 4)] == [None, None]
    robot.q[robot.arm_control_idx["right"]] = in_q
    assert mon.check(4) is None and mon._fov_streak["right"] == 0
    mon.set_fov_enabled(False)  # e.g. after the release, before the retreat leg
    robot.q[robot.arm_control_idx["right"]] = out_q
    assert [mon.check(i) for i in range(5, 15)] == [None] * 10
    mon.end_action()
    # (c) a retreat after the release has no open span: the hand may leave the frame (fov_idle_out only)
    mon.begin_action(retreat, 4, 0, intended=set(), held=[])
    assert mon.fov_active and not mon.fov_enabled
    robot.q[robot.arm_control_idx["right"]] = in_q
    mon.check(20)
    robot.q[robot.arm_control_idx["right"]] = out_q
    assert [mon.check(i) for i in range(21, 30)] == [None] * 9
    mon.end_action()
    # (d) the span spans actions: opened at the grasp's pre-pose it is still open when the lift begins (no hook), so a
    # lift / carry that leaves the frame is a verdict; a checkpoint carries the spans and a restore puts them back
    mon.begin_action(grasp, 1, 0, intended=set(), held=[])
    robot.q[robot.arm_control_idx["right"]] = in_q
    mon.set_fov_enabled(True)
    assert mon.check(30) is None and mon.fov_spans() == {"left": False, "right": True}
    mon.end_action()
    spans = mon.fov_spans()
    mon.begin_action(lift, 2, 0, intended=set(), held=[])
    assert mon.fov_enabled and mon.fov_span["right"]
    robot.q[robot.arm_control_idx["right"]] = out_q
    verdicts = [mon.check(i) for i in range(31, 34)]
    assert verdicts[:2] == [None, None] and verdicts[2].category == "gripper_out_of_fov"
    mon.end_action()
    mon.set_fov_span("right", False)
    mon.restore_fov_spans(spans)
    assert mon.fov_span["right"] and not mon.fov_span["left"]
    mon.set_fov_span("right", False)
    # in frame: the eef in front of the robot -> the stalled rule fires after 4 stationary steps; a settle step (a hold /
    # settle wait of the motion layer) restarts the count
    robot.q[robot.arm_control_idx["right"]] = in_q
    robot.dq = th.zeros(28)
    mon.begin_action(grasp, 1, 1, intended=set(), held=[], manipulation=True)
    assert [mon.check(i) for i in range(1, 4)] == [None] * 3 and mon.check(4, settle=True) is None
    verdicts = [mon.check(i) for i in range(5, 9)]
    assert verdicts[:3] == [None, None, None] and verdicts[3].category == "stalled" and verdicts[3].details["steps"] == 4
    # held object drift: the held object drifts off the hand
    apple = FakeObject("apple_0", robot.eef_pose("right")[0].tolist())
    robot._ag_obj_in_hand["right"] = apple
    mon.begin_epilogue(held=[apple])
    assert mon.check(10) is None
    apple.pos = apple.pos + th.tensor([0.2, 0.0, 0.0])
    v = mon.check(11)
    assert v is not None and v.category == "held_object_drift" and v.details["object"] == "apple_0"
    stats = mon.stats.to_dict()
    assert stats["by_category"] == {"gripper_out_of_fov": 3, "stalled": 1, "held_object_drift": 1}  # (b), (b') and the lift of (d)
    assert stats["by_action"]["grasp"] == {"gripper_out_of_fov": 2, "stalled": 1} and stats["by_action"]["lift"] == {"gripper_out_of_fov": 1}


def test_category_of_and_verdict_validation():
    assert QA.category_of(SkillError("plan_failed", "x")) == "plan_failed"
    assert QA.category_of(SkillError("weird", "x")) == "execution"
    assert QA.category_of(RuntimeError("boom")) == "error"
    with pytest.raises(ValueError):
        QA.Verdict("nonsense")
    assert set(QA.STEP_CATEGORIES) | set(QA.EPILOGUE_CATEGORIES) < set(QA.CATEGORIES)


# ==========================================================================================
# recorder schema (BEHAVIOR-replayable hdf5 + annotations + LeRobot feature mapping)
# ==========================================================================================


def test_recorder_hdf5_schema_and_state_pairing(tmp_path):
    h5py = pytest.importorskip("h5py")
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    env.robot.ag_fail_next = 1
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    rec.episode.family = sc.task_family
    rec.episode.instance_description = "put the apple on the plate"
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"]
    paths = rec.close({"instance_id": sc.instance_id, "success": True}, qa=result["monitors"]["stats"], success=True)
    assert paths["hdf5"] and os.path.exists(paths["hdf5"]) and paths["annotations"]
    T = result["n_steps"]
    with h5py.File(paths["hdf5"], "r") as f:
        data = f["data"]
        assert data.attrs["n_episodes"] == 1 and data.attrs["n_steps"] == T
        assert json.loads(data.attrs["config"])["env"]["action_frequency"] == 30
        assert "scene_file" in data.attrs and data.attrs["isg_schema"] == "isg-demo-v2"
        demo = data["demo_0"]
        assert demo.attrs["num_samples"] == T and json.loads(demo.attrs["transitions"]) == {}
        assert demo["action"].shape == (T, 23) and demo["state"].shape[0] == T and demo["state_size"].shape == (T,)
        assert demo["reward"].shape == (T,) and demo["terminated"].dtype == bool and "init_metadata" in demo
        # state[i] is the state before action[i]: state[0] is the reset state (the initial trunk), state[1] the
        # state the recorder saw after step 0
        s0 = th.from_numpy(demo["state"][0][: int(demo["state_size"][0])])
        assert th.allclose(s0[: 28][env.robot.trunk_control_idx], C.torso_q(0.5, 0.0), atol=1e-5)
        q_after0 = th.from_numpy(demo["isg/joint_positions"][0])
        s1 = th.from_numpy(demo["state"][1][: int(demo["state_size"][1])])
        assert th.allclose(s1[:28], q_after0, atol=1e-6)
        isg = demo["isg"]
        for key in ("step", "phase", "action_index", "attempt", "joint_positions", "joint_velocities", "base_pose", "eef_pose_left", "eef_pose_right", "head_camera_pose", "proprio", "eval/symbolic", "eval/settled", "eval/geometric_within", "eval/position_error", "eval/orientation_error"):
            assert key in isg, key
        assert isg["object_pose/apple.n.01_1"].shape == (T, 7)
        phases = np.asarray(isg["phase"])
        assert (phases[: result["n_action_steps"]] == PHASE_ORACLE).all() and (phases[result["n_action_steps"] :] == PHASE_SETTLE).all()
        # every dataset in the demo group is numeric (h5py_group_to_torch of the BEHAVIOR playback converts them all)
        def walk(g):
            for k, v in g.items():
                if isinstance(v, h5py.Group):
                    walk(v)
                else:
                    assert v.dtype.kind in "fiub", (k, v.dtype)

        walk(demo)
        assert "rollbacks" in demo and "action_1_attempt_0" in demo["rollbacks"]
        rb = demo["rollbacks/action_1_attempt_0"]
        assert json.loads(rb.attrs["verdict"])["category"] == "failed_grasp"
        assert rb["state"].shape[0] == rb["action"].shape[0]
        ann = json.loads(demo.attrs["isg_annotations"])
    assert ann["episode"]["family"] == sc.task_family and ann["episode"]["instance_description"]
    assert [a["name"] for a in ann["actions"]] == ["transition", "grasp", "lift", "place", "retreat"]
    assert ann["actions"][1]["attempts"] == 2 and ann["actions"][1]["rolled_back"][0]["category"] == "failed_grasp"
    assert all(a["fine"] and a["numbers"] for a in ann["actions"])
    assert ann["actions"][0]["step_start"] == 0 and ann["actions"][-1]["step_end"] == result["n_action_steps"]
    assert ann["qa"]["by_category"] == {"failed_grasp": 1}
    with open(paths["annotations"]) as fh:
        assert json.load(fh)["n_steps"] == T


def test_recorder_lerobot_feature_mapping_and_video_only_mode(tmp_path):
    sc, env = iors_world()
    rec = Recorder(str(tmp_path), env, env.robot, CAM_KEY, dict(env.objects), cfg=RecorderConfig(hdf5=False, lerobot=False))
    obs = env.get_obs()[0]
    rec.start(obs)
    robs = obs[env.robot.name]
    rec._video_keys = rec._rgb_keys(robs)
    feats = rec._lerobot_features(robs)
    assert feats["action"]["shape"] == (23,) and feats["next.reward"]["shape"] == (1,)
    assert feats["observation.images.zed_link_camera_0"]["dtype"] == "video" and feats["observation.images.zed_link_camera_0"]["shape"] == (8, 8, 3)
    assert feats["observation.state"]["shape"] == (28,) and feats["annotation.action_index"]["dtype"] == "int64"
    assert sensor_name(CAM_KEY) == "zed_link_camera_0" and sensor_name("zed_link:Camera:0") == "zed_link_camera_0"
    # a LeRobot frame pairs the observation *before* the action with the action (wrapper convention)
    prev = rec._lerobot_obs(robs)
    frame = rec._lerobot_frame(prev, th.zeros(23), 0.0, False, False, 3, "grasp the apple")
    assert frame["task"] == "grasp the apple" and int(frame["annotation.action_index"]) == 3 and "observation.state" in frame
    # video-only mode: nothing but the in-memory rows
    env.step(th.zeros(23))
    rec.record(1, th.zeros(23), env.get_obs()[0], {}, phase=PHASE_ORACLE)
    rec.commit()
    paths = rec.close({}, qa=None)
    assert paths["hdf5"] is None and paths["lerobot"] is None and os.path.exists(paths["annotations"])


# ==========================================================================================
# TRO goal override
# ==========================================================================================


def test_apply_goal_override_writes_only_applied_targets():
    tro = {
        "ispatialgym": {
            "pose_targets": {"a": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}, "b": {"position": [1, 1, 1], "orientation": [0, 0, 0, 1]}},
            "groups": {"g": {"target_slots": {"s0": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}}, "layout_relations": []}},
            "articulation_targets": {"c": {"joints": {"j": {"position": 0.0, "tolerance": 0.1}}}},
        }
    }
    override = {
        "pose_targets": {"a": {"applied": True, "position": [0.1, 0.2, 0.3], "orientation": [0, 0, 1, 0]}, "b": {"applied": False, "position": [9, 9, 9]}},
        "groups": {"g": {"applied": True, "target_slots": {"s0": {"position": [0.5, 0.5, 0.5], "orientation": [0, 0, 0, 1]}}, "layout_relations": [{"from": "s0", "to": "s0", "position": [0, 0, 0], "orientation": [0, 0, 0, 1]}]}},
        "articulation_targets": {"c": {"applied": True, "joints": {"j": {"position": 1.25}}}},
    }
    changed = apply_goal_override(tro, override)
    assert changed == ["pose_targets.a", "groups.g", "articulation_targets.c"]
    m = tro["ispatialgym"]
    assert m["pose_targets"]["a"]["position"] == [0.1, 0.2, 0.3] and m["pose_targets"]["b"]["position"] == [1, 1, 1]
    assert m["groups"]["g"]["target_slots"]["s0"]["position"] == [0.5, 0.5, 0.5] and len(m["groups"]["g"]["layout_relations"]) == 1
    assert m["articulation_targets"]["c"]["joints"]["j"]["position"] == 1.25


def test_goal_override_uses_each_targets_own_tolerance():
    """One tolerance per target (the ``isg-v2`` profile's, which the loaded TRO carries): the goal override applies a settled
    value within it, whether the target's atom is in :goal or only in :measure."""
    from ispatialgym.eval.targets import PoseTarget

    sc, env = iors_world()
    ex, *_ = make_executor(env, sc)
    apple = env.objects["apple.n.01_1"]
    tgt = PoseTarget.from_dict("apple.n.01_1", {"position": (apple.pos + th.tensor([0.005, 0.0, 0.0])).tolist(), "orientation": [0, 0, 0, 1],
                                                 "position_tolerance": 0.03, "orientation_tolerance": 0.2})

    class Compiled:
        def is_gating(self, pred, inst):
            return False

    class Gating(Compiled):
        def is_gating(self, pred, inst):
            return True

    for compiled in (Compiled(), Gating()):
        apple.pos = tgt.position.float() - th.tensor([0.005, 0.0, 0.0])
        rec = ex._override_pose_target("apple.n.01_1", tgt, compiled)
        assert rec["applied"] and rec["position_tolerance"] == pytest.approx(0.03) and rec["position_error"] == pytest.approx(0.005, abs=1e-6)
        apple.pos = apple.pos + th.tensor([0.05, 0.0, 0.0])  # 5.5 cm off: rejected
        assert ex._override_pose_target("apple.n.01_1", tgt, compiled)["applied"] is False
    depth = PoseTarget.from_dict("apple.n.01_1", {"position": [0, 0, -0.5], "position_metric": "depth", "frame": f"{AGENT_INSTANCE}::{CAM_KEY}", "position_tolerance": 0.02})
    assert ex._override_pose_target("apple.n.01_1", depth, Gating())["applied"] is False


def test_locked_joint_transform_is_exact_under_tf32():
    """cuRobo / OmniGibson switch TF32 matmul on process-wide; a locked virtual base joint carries the 40-60 m from the
    parked articulation root, and the float32 composition of its fixed transform rounded that to the 10-bit mantissa
    (door_crtrpa_0 stance: -13.75 / -5 mm = the TF32 rounding of the base x / y joint values). ``_lock_transform``
    composes in float64: exact to the micrometre wherever the base stands."""
    from ispatialgym.oracle import curobo_planner as CP

    dev = "cuda" if th.cuda.is_available() else "cpu"
    origin = th.eye(4, dtype=th.float32, device=dev)
    origin[:3, 3] = th.tensor([0.05, -0.02, 0.10], device=dev)
    prev = th.backends.cuda.matmul.allow_tf32
    try:
        th.backends.cuda.matmul.allow_tf32 = True
        for jtype, val, axis in ((CP.JointType.X_PRISM, -45.955, 0), (CP.JointType.Y_PRISM, -35.37, 1), (CP.JointType.Z_PRISM, -5.0171, 2)):
            t = CP._lock_transform(origin, jtype, val, dev)
            expect = origin[:3, 3].double().clone()
            expect[axis] += val
            err = float(th.max(th.abs(t[:3, 3].double().cpu() - expect.cpu())))
            assert err < 1e-5, (jtype, val, err)
        # a locked yaw: rotation exact to float32 precision, translation untouched
        t = CP._lock_transform(origin, CP.JointType.Z_ROT, -3.126, dev)
        assert float(th.abs(t[:3, 3].cpu() - origin[:3, 3].cpu()).max()) < 1e-6
        assert abs(float(t[0, 0]) - math.cos(-3.126)) < 1e-6 and abs(float(t[1, 0]) - math.sin(-3.126)) < 1e-6
    finally:
        th.backends.cuda.matmul.allow_tf32 = prev


@pytest.mark.parametrize("kind", ["push_under", "push_touching"])
def test_push_under_retreat_leaves_the_floor_straight_up(kind):
    """2026-09-21 (the wobble at the end of the push in 0921-021333 / 021337): the radial contact escape of a retreat after a
    push combined the partner (the table the object went under is in the contact's support list) and the floor into a
    diagonal escape normal and projected the back-off perpendicular to it - 10 cm back AND 7 cm down, the fingertips into
    the floor, the arm pressing it for ~100 steps while the base rocked. A floor blade (``contacts.push.kind ==
    "push_under"``, or the large-scale touching push ``"push_touching"``) now leaves straight up along its own approach axis
    (the push-under retreat backs out along the push at the hand's height first, owner 2026-10-13 ~04:30): every waypoint of
    the executed leg is at or above the hand, the last one above it, and the escape is recorded as skipped."""
    sc = Sidecar.load(os.path.join(VARIANTS, "inter_object_relation_satisfaction-under_large_scale__push_under.json"))
    actions = expand(sc.variant, sc)
    push = [a for a in actions if isinstance(a, Push)][0]
    push.contact.extra["kind"] = kind
    retreat = [a for a in actions[actions.index(push) + 1:] if a.name == "retreat"][0]
    assert retreat.after is push and push.kind == kind
    goal = th.tensor(push.contact.goal.position, dtype=th.float32)
    here = goal - th.tensor([0.0, 0.0, 0.0])  # the blade sits at the recorded goal after the push
    table = FakeObject("breakfast_table_x_0", [float(goal[0]), float(goal[1]) + 0.6, 0.4], category="breakfast_table", extent=(1.2, 1.2, 0.8))
    floor = FakeObject("floors_x_0", [float(goal[0]), float(goal[1]), -0.05], category="floors", extent=(10.0, 10.0, 0.1))
    target = FakeObject("target_x_0", [float(goal[0]), float(goal[1]) + 0.15, 0.03], category="briefcase", extent=(0.25, 0.1, 0.06))
    objs = {"breakfast_table.n.01_1": table, "floor.n.01_1": floor, retreat.object: target}
    quat = th.tensor(push.contact.goal.orientation, dtype=th.float32)
    state = SimpleNamespace(eef_pose=lambda arm: (here.clone(), quat.clone()), object_pose=lambda n: (objs[n].pos.clone(), objs[n].quat.clone()),
                            object=lambda n: objs[n], held_object=lambda arm: None, robot=None)
    legs: List[Any] = []

    class StubMotions:
        cfg = MotionsConfig()
        press_hold = None

        def arm_along(self, arm, leg, finish=None, soft_pos_tol=None, **kw):
            legs.append([tuple(float(v) for v in (p[0] if isinstance(p, (tuple, list)) else p.position)) for p in leg])
            if False:
                yield None

        def gripper(self, *a, **k):
            if False:
                yield None

        def hold(self, n):
            if False:
                yield None

    ctx: Dict[str, Any] = {"cfg": EX.ExecuteConfig(), "witnesses": {}, "hooks": {}}
    targets = retreat.targets(sc, state)
    list(EX.execute_retreat(retreat, StubMotions(), state, targets, ctx))
    assert ctx.get("retreat_escape", {}).get("skipped", "").startswith(kind)
    assert legs, "the retreat leg was not executed"
    leg = legs[-1]
    z0 = float(here[2])
    assert all(p[2] >= z0 - 1e-6 for p in leg), f"a floor blade must not descend on its retreat: {leg}"
    assert leg[-1][2] > z0 + 0.05, f"the retreat ends above the floor blade: {leg[-1]}"
    # the old radial escape would have gone down: the escape normal of table + floor points back and up, and the radial
    # back-off (from the pushed object to the blade, horizontal) projected perpendicular to it points into the floor
    for o in (table, floor):
        o.aabb = (o.pos - o.aabb_extent / 2, o.pos + o.aabb_extent / 2)
    n = EX.escape_direction(state, (here, quat), [table, floor])
    assert n is not None and n[2] > 0.5 and abs(n[1]) > 0.5
    radial = here - target.pos
    radial = radial - n * float(th.dot(radial, n))
    assert float(radial[2]) < -0.01


# ==========================================================================================
# 2026-09-22: alignment-world_vertical 0918-000750 (the shaking painting / the robot never settled)
# ==========================================================================================


def test_brace_of_a_hinged_target_stands_at_the_goal_edge_with_a_gap():
    """``brace_targets_at_goal``: the brace of a hinged target is moved to where the braced point of the object will be at
    its goal (the pad centre rotated about the hinge at the COM by the rotation from the current to the goal orientation),
    ``brace_at_goal_gap_m`` back along the brace direction; the fist keeps its orientation and the recorded stand-off of
    the pre pose. Not applicable (None) for a non-hinged brace kind, without a world orientation goal, for an object
    already at its goal, or with the gap disabled."""
    sc = Sidecar.load(os.path.join(SIDECARS, "alignment-world_vertical.json"))
    actions = expand(sc.variant, sc)
    push = [a for a in actions if isinstance(a, Push)][0]
    brace = [a for a in actions if isinstance(a, QA_Brace)][0]
    geom = EX.push_geometry(push.contact.waypoints, push.contact.direction)
    assert brace.contact.kind in EX.HINGE_BRACE_KINDS
    a = C.unit(th.tensor(brace.contact.direction, dtype=th.float32))
    # the painting hangs the arc's angle off its goal (the recorded arc swings it back); the hinge is at its COM = the arc pivot
    q_cur = C.quat_from_axis_angle(geom.axis, -geom.angle)
    goal_q = C.IDENTITY_QUAT.clone()
    tgt = SimpleNamespace(frame="world", orientation=goal_q, position=None, orientation_tolerance=0.08, orientation_equivalences=[])
    state = SimpleNamespace(object_pose=lambda n: (geom.pivot.clone(), q_cur.clone()), pose_target=lambda n: tgt)
    cfg = EX.ExecuteConfig()
    out = EX.brace_targets_at_goal(brace.contact, state, cfg)
    assert out is not None and abs(out["goal_rotation_rad"] - geom.angle) < 1e-4
    c_p, c_q = brace.contact.contact.tensors()
    pre_p, _ = brace.contact.pre.tensors()
    new_c, new_q = out["contact"].tensors()
    new_pre, _ = out["pre"].tensors()
    pad = c_p + a * C.CONTACT_CENTER_Z  # the braced point of the painting at its initial pose
    pad_goal = geom.pivot + C.quat_rotate(C.quat_from_axis_angle(geom.axis, geom.angle), pad - geom.pivot)
    assert th.allclose(new_c + a * C.CONTACT_CENTER_Z, pad_goal - a * cfg.brace_at_goal_gap_m, atol=1e-5)
    assert th.allclose(new_q, c_q)
    assert abs(float(th.dot(new_c - new_pre, a)) - float(th.dot(c_p - pre_p, a))) < 1e-5
    assert len(out["approach"]) == 3 and th.allclose(out["approach"][-1].tensors()[0], new_c, atol=1e-6)
    assert abs(out["shift_m"] - float(th.linalg.norm(new_c - c_p))) < 1e-6 and out["shift_m"] > 0.02
    assert EX.brace_targets_at_goal(replace(brace.contact, kind="stop_edge"), state, cfg) is None
    assert EX.brace_targets_at_goal(brace.contact, SimpleNamespace(object_pose=state.object_pose, pose_target=lambda n: None), cfg) is None
    at_goal = SimpleNamespace(object_pose=lambda n: (geom.pivot.clone(), goal_q.clone()), pose_target=lambda n: tgt)
    assert EX.brace_targets_at_goal(brace.contact, at_goal, cfg) is None
    assert EX.brace_targets_at_goal(brace.contact, state, replace(cfg, brace_at_goal_gap_m=-1.0)) is None


def test_touching_brace_follows_a_shoved_partner():
    """Touching (release validation 2026-10-03): the release opens the gripper fully beside the partner and shoves it 5-6 mm
    toward the stop, more than the stop's 5 mm clearance. Its brace follows the partner (``Brace.follow``): the recorded pre /
    contact / approach move with the partner's planar rigid motion since the program start, beyond ``brace_follow_min_m``;
    a brace of another program (axial, block design) keeps the recorded targets."""
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-touching_small_scale.json"))
    actions = expand(sc.variant, sc)
    brace = [a for a in actions if isinstance(a, QA_Brace)][0]
    push = [a for a in actions if isinstance(a, Push)][0]
    assert brace.follow and push.follow == brace.contact.object
    for name in ("alignment-axial.json", "configuration_matching-block_design.json"):
        other = Sidecar.load(os.path.join(SIDECARS, name))
        assert not any(a.follow for a in expand(other.variant, other) if isinstance(a, (QA_Brace, Push)))
    targets = brace.targets(sc, None)
    p0 = th.tensor([1.0, 2.0, 0.85])
    q0 = C.quat_from_yaw(0.3)
    cfg = EX.ExecuteConfig()

    def at(dp, dyaw):
        return SimpleNamespace(init_pose=lambda n: (p0.clone(), q0.clone()), object_pose=lambda n: (p0 + th.tensor(dp), C.quat_from_yaw(0.3 + dyaw)))

    ctx = {}
    assert EX._brace_follow_targets(brace, at([0.0005, 0.0, 0.0], 0.0), targets, ctx, cfg) is targets
    assert ctx["brace_follow"]["applied"] is False
    ctx = {}
    dp = [-0.006, 0.001, 0.0]
    out = EX._brace_follow_targets(brace, at(dp, 0.0), targets, ctx, cfg)
    assert ctx["brace_follow"]["applied"] and abs(ctx["brace_follow"]["shift_m"] - float(th.linalg.norm(th.tensor(dp)))) < 1e-4
    for key in ("pre", "contact"):
        assert th.allclose(out[key].tensors()[0], targets[key].tensors()[0] + th.tensor(dp), atol=1e-6)
        assert th.allclose(out[key].tensors()[1], targets[key].tensors()[1], atol=1e-6)
    assert len(out["approach"]) == len(targets["approach"])
    assert th.allclose(out["approach"][-1].tensors()[0], out["contact"].tensors()[0], atol=1e-6)
    out = EX._brace_follow_targets(brace, at([0.0, 0.0, 0.0], 0.05), targets, {}, cfg)
    c_rec, c_new = targets["contact"].tensors()[0], out["contact"].tensors()[0]
    assert abs(float(th.linalg.norm(c_new - p0)) - float(th.linalg.norm(c_rec - p0))) < 1e-5  # rotated about the partner
    q_want = C.qmul(C.quat_from_yaw(0.05), targets["contact"].tensors()[1])
    assert abs(abs(float(th.dot(out["contact"].tensors()[1], q_want))) - 1.0) < 1e-5
    # the push's goal follows the partner (planar), seen only through the pushed object's pose target
    from ispatialgym.eval.targets import PoseTarget

    goal = PoseTarget(instance=push.contact.object, link=None, frame="world", position=th.tensor([3.0, 1.0, 0.9]), orientation=None,
                      position_metric="full", orientation_metric=None, position_tolerance=0.03, orientation_tolerance=None)
    st = at([-0.006, 0.0005, 0.002], 0.0)
    st.pose_target = lambda n: goal if n == push.contact.object else None
    ctx = {}
    shift = EX._push_goal_follow(push, st, ctx, cfg)
    assert shift is not None and ctx["push_goal_follow"]["applied"] and abs(float(shift[2])) < 1e-9
    moved = EX._GoalShiftedState(st, push.contact.object, shift)
    assert th.allclose(moved.pose_target(push.contact.object).position, th.tensor([2.994, 1.0005, 0.9]), atol=1e-6)
    assert moved.pose_target("other") is None and moved.init_pose("x")[0].allclose(p0)
    assert EX._push_goal_follow(push, at([0.0004, 0.0, 0.0], 0.0), {}, cfg) is None


def _wall_push_retreat_world():
    sc = Sidecar.load(os.path.join(SIDECARS, "alignment-world_vertical.json"))
    actions = expand(sc.variant, sc)
    push = [a for a in actions if isinstance(a, Push)][0]
    retreat = [a for a in actions[actions.index(push) + 1:] if a.name == "retreat" and a.arm == push.contact.arm][0]
    assert retreat.after is push
    geom = EX.push_geometry(push.contact.waypoints, push.contact.direction)
    here = push.contact.goal.tensors()  # the pad at the recorded arc end
    painting = FakeObject("painting_0", [float(v) for v in geom.pivot.tolist()], category="painting", extent=(0.4, 0.03, 0.35))
    state = SimpleNamespace(eef_pose=lambda arm: (here[0].clone(), here[1].clone()), object_pose=lambda n: (painting.pos.clone(), painting.quat.clone()),
                            object=lambda n: painting, held_object=lambda arm: None, robot=None)
    return sc, retreat, state, here, geom


class _RetreatStubMotions:
    """Records the legs ``execute_retreat`` asks for; ``reach`` decides which escape end poses the fake IK reaches."""

    cfg = MotionsConfig()
    press_hold = None

    def __init__(self, reach=None, with_ik=True):
        self.legs: List[Any] = []
        self.kwargs: List[Dict[str, Any]] = []
        self.probed: List[float] = []
        self.robot = SimpleNamespace(get_joint_positions=lambda: th.zeros(28))
        backend = SimpleNamespace(update_world=lambda **kw: None)
        if with_ik:
            def solve_ik(emb, targets, q_seed=None, sync_world=True):
                (pos, _quat), = targets.values()
                self.probed.append(float(th.linalg.norm(pos)))
                return th.zeros(28) if reach(pos) else None

            backend.solve_ik = solve_ik
        self.backend = backend

    def arm_along(self, arm, leg, finish=None, soft_pos_tol=None, **kw):
        self.legs.append([as_world_pose(p)[0].clone() for p in leg])
        self.kwargs.append(dict(kw))
        if False:
            yield None

    def gripper(self, *a, **k):
        if False:
            yield None

    def hold(self, n):
        if False:
            yield None


def test_retreat_escape_is_shortened_to_a_reachable_end_pose_and_collision_checked():
    """The radial escape after a wall push probes its end pose with the collision-aware IK and shortens the radial
    distance by ``retreat_reach_factors`` until it is reachable (0918-000750: the leg toward a 13 mm unreachable end
    pose folded the elbow into the upper arm); the leg is executed collision-checked with the joint-limit margin. Without
    a backend IK the recorded distance stands."""
    sc, retreat, state, here, geom = _wall_push_retreat_world()
    cfg = EX.ExecuteConfig()
    out = cfg.contact_retreat_escape_m
    far = max(retreat.distance, 2 * out)
    radial = here[0] - geom.pivot
    radial = radial / float(th.linalg.norm(radial))  # no supports in this world: the escape is purely radial

    def run(motions):
        ctx: Dict[str, Any] = {"cfg": cfg, "witnesses": {}, "hooks": {}}
        targets = retreat.targets(sc, state)
        list(EX.execute_retreat(retreat, motions, state, targets, ctx))
        return ctx

    # (a) the full escape is out of reach, 75 % of it is not
    m = _RetreatStubMotions(reach=lambda pos: float(th.linalg.norm(pos - here[0])) <= 0.75 * far + 1e-4)
    ctx = run(m)
    esc = ctx["retreat_escape"]
    assert abs(esc["distance_m"] - 0.75 * far) < 1e-6 and abs(esc["recorded_distance_m"] - far) < 1e-9
    assert [t["reachable"] for t in esc["reach_probe"]["tried"]] == [False, True]
    leg = m.legs[-1]
    assert th.allclose(leg[0], here[0] + radial * out, atol=1e-5) and th.allclose(leg[-1], here[0] + radial * 0.75 * far, atol=1e-5)
    assert m.kwargs[-1]["check_collisions"] is True and abs(m.kwargs[-1]["limit_margin"] - cfg.retreat_limit_margin_rad) < 1e-9
    assert any(o is state.object("x") for o in m.kwargs[-1]["ignore_objects"])
    # (b) nothing is reachable: the shortest tried (the escape itself) stands, judged by the leg's finish / partial acceptance
    m = _RetreatStubMotions(reach=lambda pos: False)
    esc = run(m)["retreat_escape"]
    assert abs(esc["distance_m"] - out) < 1e-6 and esc["reach_probe"]["reachable"] is False and len(esc["reach_probe"]["tried"]) == 3
    # (c) no IK on the backend (the test stubs): the recorded distance
    m = _RetreatStubMotions(with_ik=False)
    esc = run(m)["retreat_escape"]
    assert abs(esc["distance_m"] - far) < 1e-9 and "recorded_distance_m" not in esc and "reach_probe" not in esc


def test_free_leg_jam_is_relaxed_onto_the_measured_joints():
    """``Motions._relax_if_jammed``: after a free-space leg converged, joints that keep reporting velocities above
    ``settle_vel_tol`` without moving (a drive fighting an unseen contact: the folded elbow of 0918-000750) get their hold
    targets relaxed onto the measured joints over ``jam_relax_steps`` steps; a still arm costs no step, a moving arm is
    not a jam."""
    sc, env = iors_world()
    robot = env.robot
    motions = Motions(env, FakeBackend(robot), MotionsConfig(settle_steps=5))
    idx = robot.arm_control_idx["left"]
    q_meas = robot.q[idx].clone()
    motions.state.arm_target["left"] = q_meas + th.tensor([0.0, 0.0, 0.0, -0.0166, 0.0, 0.0, 0.0])  # the elbow 16.6 mrad short of its target
    robot.dq[:] = 0.0
    robot.dq[idx[3:]] = th.tensor([0.29, 2.2, 0.49, 0.70])  # what the jammed left arm reported for the whole settle window
    acts = list(motions._relax_if_jammed(["left"]))  # no env.step: the robot stays where it is
    assert len(acts) == motions.cfg.jam_probe_steps + motions.cfg.jam_relax_steps
    assert th.allclose(motions.state.arm_target["left"], q_meas, atol=1e-7)
    assert th.allclose(acts[-1][robot.controller_action_idx["arm_left"]], q_meas, atol=1e-6)
    assert motions.last_jam is not None and motions.last_jam["arms"] == ["left"] and motions.jams == [motions.last_jam]
    assert motions.last_jam["relaxed_rad"]["left"] == pytest.approx(0.0166, abs=1e-4)
    # a still arm: nothing emitted
    robot.dq[:] = 0.0
    assert list(motions._relax_if_jammed(["left"])) == []
    # a moving arm (velocity with motion) is not a jam: the joints advance while the probe holds
    robot.dq[idx] = 0.5
    n = 0
    for _ in motions._relax_if_jammed(["left"]):
        robot.q[idx] += 0.5 * DT
        n += 1
    assert n == motions.cfg.jam_probe_steps and len(motions.jams) == 1


def test_jam_probe_leaves_a_kept_hold_alone():
    """An arm in ``keep_hold_arms`` (it carries its object to the end of the episode) is not relaxed by the jam probe: the
    wrist's steady offset under the payload is not a contact, and relaxing it lets the payload sag."""
    sc, env = iors_world()
    robot = env.robot
    motions = Motions(env, FakeBackend(robot), MotionsConfig(settle_steps=5))
    idx = robot.arm_control_idx["right"]
    target = robot.q[idx].clone() + th.tensor([0.0, 0.0, 0.0, 0.0, 0.0, -0.008, 0.0])  # the wrist 8 mrad below its target
    motions.state.arm_target["right"] = target.clone()
    robot.dq[:] = 0.0
    robot.dq[idx[5]] = 0.2163  # what the loaded wrist reported while static
    motions.keep_hold_arms = {"right"}
    assert list(motions._relax_if_jammed(["right"])) == []
    assert th.allclose(motions.state.arm_target["right"], target) and motions.jams == []
    # the same arm without a kept hold is a jam and relaxed
    motions.keep_hold_arms = set()
    assert len(list(motions._relax_if_jammed(["right"]))) == motions.cfg.jam_probe_steps + motions.cfg.jam_relax_steps
    assert len(motions.jams) == 1


def test_final_reorient_keeps_the_hold_and_the_epilogue_leaves_it(tmp_path):
    """A program whose last action moves an arm holding an object without placing it (the held-object reorient) keeps
    that hold: the arm joins ``keep_hold_arms`` and the epilogue relaxes only the other arm. An empty hand or a place
    keeps nothing."""
    sc, env = iors_world()
    ex, motions, _, _ = make_executor(env, sc, tmp_path)
    pose = ([0.5, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0])
    reorient = Reorient(mode="line", arm="right", object="apple.n.01_1", start_pose=pose, waypoints=[pose])
    ex._keep_final_hold(reorient)
    assert motions.keep_hold_arms == set()  # the hand is empty
    env.robot._ag_obj_in_hand["right"] = env.objects["apple.n.01_1"]
    ex._keep_final_hold(Place(kind="support", arm="right", object="apple.n.01_1", place=pose))
    assert motions.keep_hold_arms == set()  # a place releases the object
    ex._keep_final_hold(reorient)
    assert motions.keep_hold_arms == {"right"}
    info: Dict[str, Any] = {}
    assert len(list(ex._epilogue_hold(10, info))) == 10
    assert set(info) == {"left"}


def test_object_scaling_move_text_names_the_direction():
    """``Reorient`` (line) annotation: the sampler's signed ``depth_change_cm`` (< 0 toward the camera) names the
    direction; a sidecar without it keeps the direction-free text."""
    from types import SimpleNamespace

    from ispatialgym.oracle.sidecar import Pose as SPose

    start, end = SPose([0.60, 0.0, 1.20], [0.0, 0.0, 0.0, 1.0]), SPose([0.48, 0.0, 1.25], [0.0, 0.0, 0.0, 1.0])
    act = Reorient(mode="line", arm="right", object="can.n.01_1", start_pose=start, waypoints=[end])

    def sidecar(numbers):
        return SimpleNamespace(language=SimpleNamespace(phrase=lambda inst: "the can", goal_numbers=numbers))

    fine, numbers = act.annotate(sidecar({"depth_change_cm": -13.0}))
    assert fine == "move the can toward the camera to the goal distance, keeping its orientation" and numbers == "move the can 13 cm toward the camera"
    fine, numbers = act.annotate(sidecar({"depth_change_cm": 13.0}))
    assert "away from the camera" in fine and numbers == "move the can 13 cm away from the camera"
    fine, numbers = act.annotate(sidecar({}))
    assert numbers == "move the can 13 cm along the camera axis"


def test_epilogue_relaxes_the_arm_targets_inside_the_first_window(tmp_path):
    """The epilogue's first no-op window starts with the arms' hold targets relaxed onto their measured joints
    (``epilogue_relax_arms``): a target left pressing against an unseen contact stops pushing before ``pose_settled`` is
    judged. The window keeps its length (the relax steps are its first steps) and the relax is reported."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    relaxed_from: Dict[str, th.Tensor] = {}
    orig_relax = motions.relax_arms

    def relax_spy(arms, steps=None):
        relaxed_from.update({a: motions.state.arm_target[a].clone() for a in arms})
        return (yield from orig_relax(arms, steps=steps))

    motions.relax_arms = relax_spy
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    assert result["n_settle_steps"] == result["epilogue_steps"]  # the relax steps are the window's first steps
    assert set(result["epilogue_relax"]) == {"left", "right"} == set(relaxed_from)
    # the relaxed targets equal the measured joints (the fake joints follow their targets exactly: nothing to relax, 0 rad)
    for arm in M.ARMS:
        assert th.allclose(motions.state.arm_target[arm], env.robot.q[env.robot.arm_control_idx[arm]], atol=1e-6)
        assert result["epilogue_relax"][arm] < 1e-3
    assert result["jams"] == []
    # with the relax off the window is pure no-op and nothing is reported
    sc2, env2 = iors_world()
    env2.evaluate = success_eval("apple.n.01_1", _place_goal(sc2), tol=0.08)
    ex2, _, _, _ = make_executor(env2, sc2, tmp_path, exec_cfg=ExecutorConfig(settle_steps=50, epilogue_relax_arms=False))
    result2 = ex2.run_program(expand(sc2.variant, sc2))
    assert result2["success"] and result2["epilogue_relax"] == {} and result2["n_settle_steps"] == result2["epilogue_steps"] == 50


def test_a_push_on_a_braked_link_streams_at_the_cartesian_speed():
    """``push_speed_arc`` paces an arc push of a free hinged object; a link on its joint (``contacts.push.articulation``, held by the
    loader's Coulomb joint brake) streams at the arm's Cartesian speed (``speed=None``)."""
    seen = []

    class _Motions:
        def arm_along(self, arm, waypoints, loaded=False, partial=False, speed=None, **kw):
            seen.append(speed)
            return iter(())

    cfg = EX.ExecuteConfig()
    geom = SimpleNamespace(kind="arc", angle=0.5)
    for art in (None, {"joint": "j_lid", "q_from": 0.8, "q_to": 0.2, "tolerance": 0.15}):
        list(EX._push_leg(_Motions(), SimpleNamespace(), "right", "blender.n.01_1", [], geom, None, cfg, {"push_articulation": art}, "push"))
    assert seen == [cfg.push_speed_arc, None]


class _FkBackend(FakeBackend):
    """The fake backend with the FK ``arm_arc_closed_loop`` needs (the toy kinematics of the fake robot)."""

    def fk(self, frames, q_full=None, emb_name="whole_body"):
        q = self.robot.q if q_full is None else q_full
        out = {}
        for f in frames:
            arm = M.arm_of_eef_link(f)
            qa = q[self.robot.arm_control_idx[arm]]
            rel = ARM_OFFSET[arm] + ARM_SCALE * qa[:3]
            out[f] = (self.robot.pos + C.quat_rotate(self.robot.quat, rel), C.qmul(self.robot.quat, _quat_norm(qa[3:7])))
        return out


def test_arc_push_of_a_hinged_target_is_tracked_closed_loop_and_stops_at_the_goal_band():
    """``push_arc_tracking = "closed_loop"``: the pad reaches the first contact along a straight leg, then follows the arc
    about the hinge with ``arm_arc_closed_loop`` (one differential-IK step per control step from the measured joints, at
    ``push_speed_arc``) and stops as soon as the object's residual to its TRO goal is within the stop margin; the end pose
    of the push is where the pad actually stopped. The streamed leg is kept for the ``"streamed"`` setting."""
    for tracking in ("closed_loop", "streamed"):
        sc, env, actions, board, brace, push, geom = axial_world(gain=1.0, tol=0.10)
        robot = env.robot
        backend = _FkBackend(robot)
        motions = Motions(env, backend, MotionsConfig(settle_steps=5, gripper_steps=5, cartesian_step=0.05, cartesian_rot_step=0.3))
        state = SimState(env, env.task, robot)
        cfg = EX.ExecuteConfig(push_arc_tracking=tracking, push_backoff_m=0.02)
        ctx: Dict[str, Any] = {"cfg": cfg, "witnesses": {}, "hooks": {}}
        targets = push.targets(sc, state)
        n = 0
        for a in EX.run_action(push, motions, state, targets, ctx):
            env.step(a)
            n += 1
        goal_q = env.task._tro.meta.pose_targets[push.contact.object].orientation
        err = C.quat_angle(goal_q, board.quat)
        assert err < 0.10, (tracking, err)
        stopped = ctx.get("push_stopped") or []
        assert stopped and stopped[-1]["residual"] <= 0.10 * cfg.push_stop_frac_arc + 1e-6, (tracking, stopped)
        if tracking == "closed_loop":
            cl = ctx["push_arc_closed_loop"]
            assert stopped[-1].get("closed_loop") is True and cl["stop"] == "goal_band" and cl["steps"] == stopped[-1]["steps"]
            assert cl["speed_m_s"] == cfg.push_speed_arc
        else:
            assert "push_arc_closed_loop" not in ctx and not stopped[-1].get("closed_loop")
        # the pad backed off from where it actually stopped and the postcondition holds
        end = state.eef_pose(push.contact.arm)
        assert float(th.linalg.norm(end[0] - ctx["push_end_pose"][0])) < 1e-3
        assert EX.check_postcondition(push, state, targets, ctx) is None


# ==========================================================================================
# owner 2026-09-24: a travel swept with the arm where the previous motion left it (``paths[].arms_q`` /
# ``transitions[].arms_q``) - the executor re-poses only a deviating arm
# ==========================================================================================


def _nav_from(robot, arms_q=None, posture="untucked", dx=0.3):
    """A ``Navigate`` over a short recorded path from the robot's base pose, optionally carrying ``arms_q``."""
    from ispatialgym.oracle import sidecar as SC

    x, y = float(robot.pos[0]), float(robot.pos[1])
    yaw = float(C.yaw_from_quat(robot.quat)) if hasattr(C, "yaw_from_quat") else 0.0
    d = {"from": "manipulation_2", "to": "manipulation_3", "mode": "local", "waypoints": [[x, y, yaw], [x + dx, y, yaw]], "length_m": dx,
         "posture": posture, "carried": None}
    if arms_q is not None:
        d["arms_q"] = arms_q
    return Navigate(path=SC.Path.from_dict(d, "paths[0]"))


def _first_move_step(rows, idx):
    q = np.asarray(rows["joint_positions"])
    moved = np.abs(q[:, idx] - q[0, idx]).max(axis=1) > 1e-4
    return int(np.argmax(moved)) if moved.any() else None


def _first_base_move_step(rows):
    p = np.asarray(rows["base_pose2d"]) if "base_pose2d" in rows else None
    if p is None:
        return None
    moved = np.abs(p[:, :2] - p[0, :2]).max(axis=1) > 1e-3
    return int(np.argmax(moved)) if moved.any() else None


def test_navigate_keeps_an_arm_at_the_records_arms_q(tmp_path):
    """The record was swept with the right arm where the retreat left it (``arms_q``): the arm does not move before the
    drive (no re-pose to the untucked default), the note says so."""
    sc, env = iors_world()
    robot = env.robot
    ridx, lidx = robot.arm_control_idx["right"], robot.arm_control_idx["left"]
    retreat_q = th.tensor([0.5, -0.4, 0.7, 0.3, 0.2, -0.1, 0.9])
    robot.q[ridx] = retreat_q.clone()
    assert float(th.max(th.abs(robot.q[ridx] - robot.untucked_default_joint_pos[ridx]))) > EX.IDLE_ARM_TOL
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    nav = _nav_from(robot, arms_q={"right": retreat_q.tolist(), "left": robot.q[lidx].tolist()})
    targets = nav.targets(sc, ex.state)
    assert targets["arms_q"] == {"right": retreat_q.tolist(), "left": robot.q[lidx].tolist()}
    assert ex.run_action(0, nav) is None
    assert th.allclose(robot.q[ridx], retreat_q, atol=1e-5), "the arm swept in the record must stay where it is"
    notes = ex.action_records[0].notes
    assert notes["navigate_arms_reposed"] == [] and notes["navigate_arms_source"] == "record arms_q"


def test_navigate_reposes_only_the_arm_that_deviates_from_the_records_arms_q(tmp_path):
    """A retreat that ended short in another branch: the arm deviates from the swept joints and is brought there first
    (before the base moves); the other arm, already at its swept joints, is left alone."""
    sc, env = iors_world()
    robot = env.robot
    ridx, lidx = robot.arm_control_idx["right"], robot.arm_control_idx["left"]
    swept_right = th.tensor([0.5, -0.4, 0.7, 0.3, 0.2, -0.1, 0.9])
    robot.q[ridx] = swept_right + th.tensor([0.3, 0.0, -0.2, 0.0, 0.0, 0.0, 0.0])
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    nav = _nav_from(robot, arms_q={"right": swept_right.tolist(), "left": robot.q[lidx].tolist()})
    assert ex.run_action(0, nav) is None
    assert th.allclose(robot.q[ridx], swept_right, atol=1e-4)
    notes = ex.action_records[0].notes
    assert notes["navigate_arms_reposed"] == ["right"]
    rows = rec.rows
    arm_step, base_step = _first_move_step(rows, ridx.numpy()), _first_base_move_step(rows)
    assert arm_step is not None and (base_step is None or arm_step < base_step), (arm_step, base_step)
    assert _first_move_step(rows, lidx.numpy()) is None  # the left arm was already at its swept joints


def test_navigate_without_arms_q_keeps_the_untucked_rule(tmp_path):
    """No ``arms_q`` on the record: the 2026-09-19 rule - an untucked, uncarried path re-poses every empty arm to the
    untucked default before the drive."""
    sc, env = iors_world()
    robot = env.robot
    ridx, lidx = robot.arm_control_idx["right"], robot.arm_control_idx["left"]
    robot.q[ridx] = th.tensor([0.5, -0.4, 0.7, 0.3, 0.2, -0.1, 0.9])
    robot.q[lidx] = robot.untucked_default_joint_pos[lidx].clone()  # the fake spawns off the untucked default: only the right arm deviates
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    nav = _nav_from(robot)
    assert nav.targets(sc, ex.state)["arms_q"] is None
    assert ex.run_action(0, nav) is None
    assert th.allclose(robot.q[ridx], robot.untucked_default_joint_pos[ridx], atol=1e-4)
    assert ex.action_records[0].notes["navigate_arms_reposed"] == ["right"]


def test_transition_arms_q_replaces_the_stance_posture_for_move_idle_arms(tmp_path):
    """``transitions[].arms_q`` (the arm as the previous motion left it): ``move_idle_arms`` targets those joints instead of
    the arriving stance's ``arms_q`` / posture - an arm already there does not move; the stance's fingers still apply."""
    sc, env = iors_world()
    robot = env.robot
    manip = sc.stance("manipulation")
    ridx, lidx = robot.arm_control_idx["right"], robot.arm_control_idx["left"]
    manip.extra["arms_q"] = {"left": [0.9, 0.3, -0.7, -1.2, 0.5, 0.2, 0.8], "right": [0.0] * 7}
    manip.extra["idle_posture"] = "ready"
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    here = {"left": robot.q[lidx].tolist(), "right": robot.q[ridx].tolist()}
    ctx = {}
    list(EX.move_idle_arms(motions, ex.state, manip, ctx, arms_q=here))
    assert ctx["arms_posture"]["moved"] == [] and ctx["arms_posture"]["source"].startswith("record arms_q")
    assert th.allclose(robot.q[lidx], th.tensor(here["left"]), atol=1e-6) and th.allclose(robot.q[ridx], th.tensor(here["right"]), atol=1e-6)
    # a deviating arm is brought to the record's joints (not to the stance's)
    want_right = th.tensor(here["right"]) + th.tensor([0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ctx = {}
    for a in EX.move_idle_arms(motions, ex.state, manip, ctx, arms_q={"left": here["left"], "right": want_right.tolist()}):
        env.step(a)
    assert ctx["arms_posture"]["moved"] == ["right"] and th.allclose(robot.q[ridx], want_right, atol=1e-4)
    # the transition action carries the record's arms_q in its targets
    transition = expand(sc.variant, sc)[0]
    assert transition.name == "transition" and transition.targets(sc, ex.state)["arms_q"] is None
    transition.transition.extra["arms_q"] = {"right": want_right.tolist()}
    assert transition.targets(sc, ex.state)["arms_q"] == {"right": want_right.tolist()}
    transition.transition.extra["arms_q"] = {"right": [0.0, 1.0]}  # malformed: ignored
    assert transition.targets(sc, ex.state)["arms_q"] is None


def test_retreat_stalling_at_a_joint_limit_without_a_witness_is_accepted_as_partial(monkeypatch):
    """Owner 2026-09-24 (block design 0925-062801 retreat[15]): a free-space retreat whose differential IK stalls AT a joint
    limit with no recorded witness used to fail the action outright (nothing executed, ``plan_failed``); now the converged
    prefix is executed and ``execute_retreat`` accepts the leg as the partial retreat it is when it ends within
    ``retreat_partial_accept_m`` of its end pose."""
    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc)
    motions.cfg = replace(motions.cfg, cartesian_step=0.02)  # the 5 cm retreat densifies to 3 waypoints
    actions = expand(sc.variant, sc)
    names = [a.name for a in actions]
    assert names[:5] == ["transition", "grasp", "lift", "place", "retreat"]
    for i in range(4):
        assert ex.run_action(i, actions[i]) is None
    backend.stall_ik_after = (2, 0.013, ["right_arm_joint2"])  # stalls 13 mm short at the third waypoint, joint 2 on its limit
    assert ex.run_action(4, actions[4]) is None
    rec = ex.action_records[4]
    assert rec.attempts == [None], rec.attempts
    assert 0.0 < rec.notes["retreat_partial_m"] <= EX.ExecuteConfig().retreat_partial_accept_m
    # a limit stall before the first waypoint (nothing converged, the whole 5 cm leg missing): owner 2026-09-26 (axial 0926-061151
    # retreat[2]) - a planned transit to the leg's end pose from where the hand is, recorded; without it the attempt fails as before
    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc)
    motions.cfg = replace(motions.cfg, cartesian_step=0.02)
    actions = expand(sc.variant, sc)
    for i in range(4):
        assert ex.run_action(i, actions[i]) is None
    backend.stall_ik_after = (0, 0.05, ["right_arm_joint2"])
    backend.calls.clear()
    assert ex.run_action(4, actions[4]) is None
    rec = ex.action_records[4]
    assert rec.attempts == [None] and rec.notes["retreat_planned_fallback"]["short_m"] > EX.ExecuteConfig().retreat_partial_accept_m
    assert "plan_pose" in backend.calls
    monkeypatch.setattr(EX, "CONFIG", replace(EX.CONFIG, retreat_plan_fallback=False))
    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc)
    motions.cfg = replace(motions.cfg, cartesian_step=0.02)
    actions = expand(sc.variant, sc)
    for i in range(4):
        assert ex.run_action(i, actions[i]) is None
    backend.stall_ik_after = (0, 0.05, ["right_arm_joint2"])
    ex.run_action(4, actions[4])
    rec = ex.action_records[4]
    assert rec.attempts[0] is not None and rec.attempts[0]["category"] == "plan_failed"


# ==========================================================================================
# owner 2026-09-24 regressions: (1) the brace hand on the braced object is allowed while the brace holds; (2) a
# collision-checked leg fails on NEW collisions only
# ==========================================================================================


def test_collision_monitor_allows_the_brace_fingers_on_the_braced_object():
    """Item 1: ``allowed`` pairs (the brace arm's finger / gripper links vs the braced object) are neither flagged nor counted
    during the push; every other link / object pair keeps its rule."""
    robot = FakeRobot()
    partner = FakeObject("partner_box_of_tissues_0", [0.6, 0.1, 0.9], category="box_of_tissues")
    target = FakeObject("target_box_of_napkins_0", [0.6, -0.1, 0.9], category="box")
    env = FakeEnv(robot, {"partner": partner, "target": target})
    mon = QA.Monitors(env, robot, cfg=QA.QAConfig(collision_min_steps=2, fov_min_steps=100000, stall_steps=100000))
    left_pad = f"{robot.prim_path}/{M.FINGER_LINKS['left'][0]}"
    right_pad = f"{robot.prim_path}/{M.FINGER_LINKS['right'][0]}"
    torso = f"{robot.prim_path}/torso_link4"
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-touching_small_scale.json"))
    push = next(a for a in expand(sc.variant, sc) if a.name == "push")
    allowed = [(list(M.FINGER_LINKS["left"]) + [M.GRIPPER_LINK["left"]], partner.name)]
    # the brace (left) pad presses the partner throughout the (right-arm) push: allowed
    mon._contact_api = FakeContacts(lambda s, q: [(left_pad, f"{partner.prim_path}/base_link")])
    mon.begin_action(push, 7, 0, intended={target.name}, held=[], manipulation=True, allowed=allowed)
    assert [mon.check(i) for i in range(1, 6)] == [None] * 5
    assert mon.stats.events.get("contact", 0) == 0 and mon.collision_events == []
    mon.end_action()
    # the same contact without the allowance is the old verdict
    mon._contact_api = FakeContacts(lambda s, q: [(left_pad, f"{partner.prim_path}/base_link")])
    mon.begin_action(push, 7, 0, intended={target.name}, held=[], manipulation=True)
    verdicts = [mon.check(i) for i in range(1, 4)]
    assert verdicts[0] is None and verdicts[1] is not None and verdicts[1].category == "collision" and "left_gripper_finger_link1" in verdicts[1].message
    mon.end_action()
    # an allowed pair covers ONLY those links against THAT object: the torso on the partner, or the left pad on a third object, still fail
    mon._contact_api = FakeContacts(lambda s, q: [(torso, f"{partner.prim_path}/base_link")])
    mon.begin_action(push, 7, 0, intended={target.name}, held=[], manipulation=True, allowed=allowed)
    verdicts = [mon.check(i) for i in range(1, 4)]
    assert verdicts[1] is not None and verdicts[1].details["link"] == "torso_link4"
    mon.end_action()
    # the acting arm's own fingers stay exempt as before
    mon._contact_api = FakeContacts(lambda s, q: [(right_pad, f"{partner.prim_path}/base_link")])
    mon.begin_action(push, 7, 0, intended={target.name}, held=[], manipulation=True, allowed=allowed)
    assert [mon.check(i) for i in range(1, 4)] == [None] * 3
    mon.end_action()


def test_brace_allowed_pairs_span_the_brace_to_its_release():
    """Item 1: the executor derives the allowed pairs from the program - the brace arm's fingers vs the braced object for every
    action from the ``Brace`` through the brace arm's releasing ``Retreat`` (the push and the push arm's retreat included)."""
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-touching_small_scale.json"))
    actions = expand(sc.variant, sc)
    names = [a.name for a in actions]
    b = names.index("brace")
    rel = next(i for i, a in enumerate(actions) if a.name == "retreat" and getattr(a, "release", False) and i > b)
    brace = actions[b]

    class _Ex:
        program_actions = actions

        def _scene_names(self, insts):
            return {f"scene:{i}" for i in insts}

    ex = _Ex()
    for i, a in enumerate(actions):
        pairs = Executor.brace_allowed_pairs(ex, i)
        if b <= i <= rel:
            assert pairs and pairs[0][1] == f"scene:{brace.contact.object}" and set(pairs[0][0]) == set(M.FINGER_LINKS[brace.contact.arm]) | {M.GRIPPER_LINK[brace.contact.arm]}, (i, a.name, pairs)
        else:
            assert pairs == [], (i, a.name, pairs)


def test_start_collisions_relaxed_exempts_only_pairs_present_at_the_start():
    """Item 2 (b): the pairs the leg's start already violates against another arm's link / attached object leave the sphere
    model for the block; pairs within the moving arm (the 0918-000750 elbow fold) and against the trunk / base stay."""
    from ispatialgym.oracle.curobo_planner import R1ProCuRobo

    disabled, enabled = [], []

    class _KC:
        link_name_to_idx_map = {"attached_object_right_eef_link": 0, "left_gripper_finger_link1": 1, "right_arm_link3": 2, "right_arm_link5": 3, "torso_link4": 4, "left_arm_link4": 5}

        def disable_link_spheres(self, link):
            disabled.append(link)

        def enable_link_spheres(self, link):
            enabled.append(link)

    class _FK:
        kinematics_config = _KC()

    class _Entry:
        fk = _FK()

    class _Planner:
        _arm_of_link = staticmethod(R1ProCuRobo._arm_of_link)

        def __init__(self, pairs):
            self._pairs = pairs

        def self_collision_pairs(self, q, emb_name):
            return self._pairs

        def kin_entry(self, emb):
            return _Entry()

    # the bracing left hand leaving the carton the right hand holds (0925-103154): the attached carton is exempted for the left arm's leg
    p = _Planner([{"links": ("attached_object_right_eef_link", "left_gripper_finger_link1"), "penetration": 0.05}])
    with R1ProCuRobo.start_collisions_relaxed(p, "left_arm", th.zeros(28), ["left"]) as done:
        assert done == ["attached_object_right_eef_link"]
    assert disabled == ["attached_object_right_eef_link"] and enabled == ["attached_object_right_eef_link"]
    disabled.clear(); enabled.clear()
    # the moving arm against itself (the elbow fold) and against the trunk: genuine, nothing relaxed
    p = _Planner([{"links": ("right_arm_link3", "right_arm_link5"), "penetration": 0.001}, {"links": ("right_arm_link5", "torso_link4"), "penetration": 0.002}])
    with R1ProCuRobo.start_collisions_relaxed(p, "right_arm", th.zeros(28), ["right"]) as done:
        assert done == []
    assert disabled == []
    # two static arms' links (the old static_links_relaxed case) and a static arm against the trunk: the static links go
    p = _Planner([{"links": ("left_arm_link4", "torso_link4"), "penetration": 0.003}])
    with R1ProCuRobo.start_collisions_relaxed(p, "right_arm", th.zeros(28), ["right"]) as done:
        assert done == ["left_arm_link4"]
    assert R1ProCuRobo._arm_of_link("attached_object_left_eef_link") == "left" and R1ProCuRobo._arm_of_link("torso_link4") is None


def test_retreat_ignores_the_objects_the_hand_touches_where_it_starts(tmp_path):
    """Item 2 (a): ``execute_retreat`` leaves every object the retreating hand is in contact with at the start out of the leg's
    collision world (the partner the push pad ended against), recorded in the notes; objects it meets later still count."""
    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc, tmp_path)
    actions = expand(sc.variant, sc)
    for i in range(4):
        assert ex.run_action(i, actions[i]) is None
    plate = env.objects["plate.n.04_1"]
    ex.state.hand_contacts = lambda arm: [plate]  # the pad still presses the plate where the retreat starts
    seen = {}
    real_update = backend.update_world

    def spy_update(*a, **kw):
        seen["ignore"] = list(kw.get("ignore_objects", ()) or ())
        return real_update(*a, **kw) if callable(real_update) else None

    backend.update_world = spy_update
    assert ex.run_action(4, actions[4]) is None
    rec = ex.action_records[4]
    assert rec.notes.get("retreat_start_contacts") == [plate.name]
    assert any(o is plate for o in seen.get("ignore", [])), seen


def test_retreat_ignores_the_planner_start_offenders_too(tmp_path):
    """Item 2 follow-up (owner 2026-09-25, touching 0925-134446): the start-in-collision set also comes from the planner - the
    objects owning the world meshes the start configuration penetrates in cuRobo's padded sphere model (no physical contact
    reported) join the leg's and the finish's ignore_objects and are noted as ``retreat_start_offenders``; the search runs with
    the leg's own ignore set applied and only when the start reads as a world collision."""
    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc, tmp_path)
    actions = expand(sc.variant, sc)
    for i in range(4):
        assert ex.run_action(i, actions[i]) is None
    plate = env.objects["plate.n.04_1"]
    ex.state.hand_contacts = lambda arm: []  # physics: no contact
    mesh = "/World/scene_0/plate/base_link/collisions/plate__base_link__collision__0"
    plate.links = {"base_link": type("L", (), {"collision_meshes": {"c0": type("CM", (), {"prim_path": mesh})()}})()}
    calls = {}

    def start_world_offenders(q, emb_name="whole_body", limit=6):
        calls["emb"] = emb_name
        calls["ignore_at_query"] = list(seen.get("ignore", []))
        return [mesh]

    def objects_of_world_meshes(names):
        return [o for o in env.objects.values()
                if any(str(cm.prim_path) in set(names) for l in getattr(o, "links", {}).values() for cm in getattr(l, "collision_meshes", {}).values())]

    backend.start_world_offenders = start_world_offenders
    backend.objects_of_world_meshes = objects_of_world_meshes
    seen, legs, finishes = {}, [], []
    real_update, real_along = backend.update_world, motions.arm_along

    def spy_update(*a, **kw):
        seen["ignore"] = list(kw.get("ignore_objects", ()) or ())
        return real_update(*a, **kw)

    def spy_along(arm, waypoints, *a, **kw):
        legs.append(list(kw.get("ignore_objects", ()) or ()))
        finishes.append(kw.get("finish"))
        return real_along(arm, waypoints, *a, **kw)

    backend.update_world = spy_update
    motions.arm_along = spy_along
    assert ex.run_action(4, actions[4]) is None
    rec = ex.action_records[4]
    assert rec.notes.get("retreat_start_offenders") == [plate.name] and "retreat_start_contacts" not in rec.notes
    assert calls["emb"] == "right_arm" and not any(o is plate for o in calls["ignore_at_query"])  # queried with the leg's own ignore set
    assert legs and any(o is plate for o in legs[-1]), legs
    assert finishes[-1] is not None and any(o is plate for o in finishes[-1].ignore_objects)


def test_arm_to_replans_a_transit_near_a_protected_object_via_the_point_above(tmp_path):
    """Owner 2026-09-26 (oracle.protection; teacup 0925-222212 / 235559): inside a protect context a planned transit that passes
    too close to a protected object is re-planned via the point above its goal (rise, plan to above, straight down) and the
    firing is recorded; a plan that clears runs unchanged; outside a context nothing is checked."""
    from types import SimpleNamespace

    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc, tmp_path)
    arm = "right"
    p0, q0 = motions.eef_pose(arm)
    target = (p0 + th.tensor([0.08, 0.0, -0.04]), q0)
    backend.solve_ik = None
    n = {"calls": 0}

    def viol(arms, q_traj, emb, P=None):
        n["calls"] += 1
        return [{"object": "partner_saucer_mgbeah_0", "clearance_m": 0.004, "threshold_m": 0.02, "index": 5, "n": 10, "link": "right_gripper_finger_link1"}] if n["calls"] == 1 else []

    motions._protect_violations = viol
    saucer = [("saucer.n.02_1", SimpleNamespace(name="partner_saucer_mgbeah_0"))]
    backend.calls.clear()
    with motions.protecting(saucer, leg="pre_grasp", task="teacup", action="grasp", index=1):
        drive(env, motions.arm_to(arm, pose=target))
    ev = motions.protect_events[-1]
    assert ev["remedy"] == "via-above" and ev["leg"] == "pre_grasp" and ev["objects"][0]["clearance_m"] == 0.004
    assert backend.calls.count("plan_pose") == 2 and backend.calls.count("solve_ik_path") >= 2  # the plan to above; the lift and the descent
    p1, _ = motions.eef_pose(arm)
    assert float(th.linalg.norm(p1 - target[0])) < 5e-3
    # a plan that clears is executed unchanged: no event, one plan
    motions.protect_events.clear()
    backend.calls.clear()
    motions._protect_violations = lambda arms, q_traj, emb, P=None: []
    with motions.protecting(saucer, leg="pre_grasp", task="teacup", action="grasp", index=1):
        drive(env, motions.arm_to(arm, pose=(p1 + th.tensor([0.0, 0.03, 0.0]), q0)))
    assert motions.protect_events == [] and backend.calls.count("plan_pose") == 1
    # no context: the check is never consulted
    motions._protect_violations = lambda *a, **k: (_ for _ in ()).throw(AssertionError("checked outside a protect context"))
    drive(env, motions.arm_to(arm, pose=(p1, q0)))


def test_arm_to_witness_retries_the_joint_plan_and_refuses_another_branch(tmp_path):
    """Owner 2026-09-25 (teacup pod demo 0925-172723): a failed joint-space plan to a recorded witness is retried
    (``witness_plan_retries`` rounds) before any pose-goal fallback, and a fallback that lands in another IK branch (further than
    ``witness_branch_max_jump`` from the witness) fails the attempt instead of continuing there - the chained legs would fail anyway."""
    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc, tmp_path)
    idx = env.robot.arm_control_idx["right"]
    pose = as_world_pose(env.robot.eef_pose("right"))
    q_hint = env.robot.get_joint_positions().clone()
    q_hint[idx[3]] += 2.0  # the witness sits in another branch than where a pose plan would end
    motions.backend.solve_ik = None  # no refinement: the witness is used as recorded
    rounds = 1 + motions.cfg.witness_plan_retries
    # the first round and the first retry fail, the second retry succeeds: no pose-goal fallback
    backend.fail_plans = 2
    backend.calls.clear()
    drive(env, motions.arm_to("right", pose=pose, q_hint=q_hint))
    assert backend.calls.count("plan_joint") == 3 and "plan_pose" not in backend.calls
    # every joint-space round fails, the pose-goal plan ends 2 rad from the witness and no arm posture carries the transit:
    # plan_failed with the excursion recorded
    n_via = len(motions.cfg.witness_via_postures)
    orig_pose = backend.plan_pose

    def pose_then_block(*a, **k):
        r = orig_pose(*a, **k)
        backend.fail_plans = n_via  # the first leg to every posture fails
        return r

    backend.plan_pose = pose_then_block
    backend.fail_plans = rounds
    backend.calls.clear()
    with pytest.raises(SkillError) as ei:
        drive(env, motions.arm_to("right", pose=pose, q_hint=q_hint))
    assert ei.value.reason == "plan_failed" and ei.value.details.get("excursion") is not None and ei.value.details["excursion"] > 0.6
    assert backend.calls.count("plan_joint") == rounds + n_via and backend.calls.count("plan_pose") == 1
    backend.plan_pose = orig_pose
    # first-demo v2.2 (kielbasa 0928-050633): when a posture carries it (start -> posture -> witness, two joint-space plans), the
    # witness is reached through it instead - recorded; nothing else changes
    backend.fail_plans = rounds
    backend.calls.clear()
    n_ev = len(motions.witness_via_posture_events)
    drive(env, motions.arm_to("right", pose=pose, q_hint=q_hint))
    ev = motions.witness_via_posture_events[-1]
    assert len(motions.witness_via_posture_events) == n_ev + 1 and ev["posture"] == motions.cfg.witness_via_postures[0] and "rad from it" in ev["why"]
    assert backend.calls.count("plan_joint") == rounds + 2 and backend.calls.count("plan_pose") == 1
    # owner 2026-09-26 (axial 0926-043254 / 051919): a caller whose legs start from wherever the hand is (a push's pre-contact) keeps
    # the other-branch pose-goal plan instead - recorded; the arm ends where the pose plan ends, not at the witness
    backend.fail_plans = rounds
    backend.calls.clear()
    drive(env, motions.arm_to("right", pose=pose, q_hint=q_hint, accept_other_branch=True))
    ev = motions.witness_branch_accepted_events[-1]
    assert ev["branch_gap_rad"] > 0.6 and ev["rounds"] == rounds
    assert backend.calls.count("plan_joint") == rounds and backend.calls.count("plan_pose") == 1
    assert abs(float(env.robot.get_joint_positions()[idx[3]]) - float(q_hint[idx[3]])) > 0.6


def test_apply_recorded_targets_sets_the_groups_the_record_covers():
    """The loader holds a TRO's recorded drive targets (``robot_state.controller_targets``) instead of the loaded joints for
    every position-controlled group the record covers completely; a partly covered group keeps its hold target."""
    from types import SimpleNamespace

    from ispatialgym.oracle.action import apply_recorded_targets

    names = [f"torso_joint{k}" for k in range(1, 5)] + [f"left_arm_joint{k}" for k in range(1, 8)] + [f"right_arm_joint{k}" for k in range(1, 8)]
    robot = SimpleNamespace(joints={n: None for n in names}, trunk_control_idx=th.arange(0, 4),
                            arm_control_idx={"left": th.arange(4, 11), "right": th.arange(11, 18)})
    hold = SimpleNamespace(trunk_target=th.zeros(4), arm_target={"left": th.zeros(7), "right": th.zeros(7)})
    targets = {n: 0.1 * i for i, n in enumerate(names) if not n.startswith("left")}
    targets["left_arm_joint1"] = 0.5
    assert apply_recorded_targets(hold, robot, targets) == ["trunk", "arm_right"]
    assert th.allclose(hold.trunk_target, th.tensor([0.0, 0.1, 0.2, 0.3]))
    assert th.allclose(hold.arm_target["right"], th.tensor([0.1 * i for i in range(11, 18)]))
    assert th.allclose(hold.arm_target["left"], th.zeros(7))
    assert apply_recorded_targets(hold, robot, {}) == []


# ==========================================================================================
# object_tilt (former root_moved gate) / stage-2 hand-over (owner 2026-09-27, blender 0926-073955)
# ==========================================================================================


def _tipping_plate_world(tip_deg: float):
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    plate = env.objects["plate.n.04_1"]
    q0, p0 = plate.quat.clone(), plate.pos.clone()

    def tip(step_action):
        # the watched body tips while the arm is low with the apple in hand and rests flat again afterwards (position pinned)
        plate.pos = p0.clone()
        if env.robot._ag_obj_in_hand["right"] is not None and float(env.robot.eef_pose("right")[0][2]) < 0.90:
            h = math.radians(tip_deg) / 2
            plate.quat = C.qmul(q0, th.tensor([math.sin(h), 0.0, 0.0, math.cos(h)]))
        else:
            plate.quat = q0.clone()

    env.after_step = tip
    return sc, env, plate


def test_object_tilt_fails_an_oracle_demo_at_the_step_without_a_retry(tmp_path, monkeypatch):
    """``--qa`` object_tilt (owner 2026-09-27 ``root_moved``; its 10 mm / 0.04 rad band since 2026-10-14): the root of a
    free-standing articulated object whose joint the program works is read every step; a 3 deg tip fails the oracle demo at
    that step, the action not retried; a 1 deg rock passes; ``--no-qa`` watches nothing."""
    quiet_qa(monkeypatch)
    for qa, tip, ok in ((True, 3.0, False), (False, 3.0, True), (True, 1.0, True)):
        sc, env, plate = _tipping_plate_world(tip)
        ex, *_ = make_executor(env, sc, tmp_path / f"{qa}_{tip}", qa=qa)
        ex.operated_roots = lambda actions=None, plate=plate: ({"plate.n.04_1": plate}, {})
        result = ex.run_program(expand(sc.variant, sc))
        assert result["success"] is ok, (qa, tip, result["verdict"])
        if not qa:
            assert result["qa"] == {"enabled": False}
            continue
        tilt = result["qa"]["metrics"]["object_tilt"]
        rep = tilt["objects"]["plate.n.04_1"]
        assert abs(math.degrees(rep["peak"]["tilt_rad"]["value"]) - tip) < 0.05 and rep["peak"]["tilt_rad"]["action"] in ("place", "grasp", "lift")
        if ok:
            assert rep["end"]["rotation_rad"] < 1e-5 and rep["end"]["displacement_m"] < 1e-6 and not tilt["triggered"]
            continue
        (ev,) = tilt["events"]
        v = result["verdict"]
        assert result["reason"] == "object_tilt" and result["qa"]["stopped_by"] == "object_tilt" and v["details"]["object"] == "plate.n.04_1"
        assert v["step"] == ev["step"] and len(result["actions"][v["action_index"]]["attempts"]) == 1


def test_operated_roots_watch_the_worked_free_articulated_object_only():
    """The situation off the program: a handle grasp / follow / articulated push names the object whose body must stay put;
    a fixed-base object is skipped (its root cannot move), a placed or plainly pushed object is never watched."""
    sc = Sidecar.load(os.path.join(SIDECARS, "configuration_matching-articulation_open_small_scale.json"))
    box = FakeObject("toolbox_0", [6.0, 3.4, 0.9], category="toolbox")
    table = FakeObject("breakfast_table_0", [7.0, 3.4, 0.4], category="breakfast_table")
    box.fixed_base = table.fixed_base = False
    ex = Executor.__new__(Executor)
    ex.robot, ex.program_actions = object(), expand(sc.variant, sc)
    ex.tracked = {"toolbox.n.01_1": box, "breakfast_table.n.01_1": table}
    assert ex.operated_roots() == ({"toolbox.n.01_1": box}, {})
    box.fixed_base = True
    assert ex.operated_roots() == ({}, {"toolbox.n.01_1": "fixed base"})
    sc2, env2 = iors_world()
    ex2, *_ = make_executor(env2, sc2)
    ex2.program_actions = expand(sc2.variant, sc2)
    assert ex2.operated_roots() == ({}, {})


def test_root_watch_checkpoint_forgets_a_rolled_back_attempt():
    obj = FakeObject("lidded_0", [0.0, 0.0, 0.5])
    w = QA.RootWatch({"x": obj})
    w.observe("grasp", 0)
    snap = w.snapshot()
    obj.pos = th.tensor([0.02, 0.0, 0.5])  # the failed attempt shoves the body 2 cm; the retry starts from the checkpoint
    w.observe("base_follow", 1)
    assert list(w.over_band(0.005, 0.02)) == ["x"]
    w.load(snap)
    obj.pos = th.tensor([0.0, 0.0, 0.5])
    w.observe("base_follow", 1)
    assert w.over_band(0.005, 0.02) == {}


def test_base_follow_stage2_handover_does_not_ride_past_follow_to():
    """A stage-2 record (``articulation.follow_to``: the fist push takes the joint on) ends the hand's part at q(follow_to): a
    follow that reached it does not ride on toward q_goal (blender 0926-073955 rode 0.47 rad past it and tipped the body); one
    short of it rides on only to it; without a hand-over the ride-on to the goal band is unchanged (0921-021042)."""
    sc = Sidecar.load(os.path.join(SIDECARS, "configuration_matching-articulation_open_small_scale.json"))
    extra0 = dict(sc.articulation.extra)

    class _Motions:
        follow_phase = "arm"

        def __init__(self, q, dq):
            self.q, self.dq, self.legs = q, dq, 0

        def arm_along(self, arm, waypoints, converge=True, **kw):
            self.legs += 1
            for _ in range(40 if self.legs == 1 else 20):
                self.q[0] += self.dq if self.legs == 1 else 0.03
                yield th.zeros(23)

        def unwind(self, arm):
            return 0.0

        def hold(self, steps=1):
            for _ in range(int(steps)):
                yield th.zeros(23)

    def run(follow_to, q_end):
        follow = [a for a in expand(sc.variant, sc) if isinstance(a, BaseFollow)][0]
        art = follow.articulation
        art.passive_from = None
        art.extra = {**extra0, "hinge_world": [6.0, 3.35, 1.15], "axis_world": [1.0, 0.0, 0.0],
                     "object_pose": {"position": [6.0, 3.4, 0.9], "orientation": [0.0, 0.0, 0.0, 1.0]}}
        if follow_to is not None:
            art.extra["follow_to"] = follow_to
        q = [0.0]
        motions = _Motions(q, q_end / 40)
        state = SimpleNamespace(joint_position=lambda o, j: float(q[0]),
                                object_pose=lambda o: (th.tensor([6.0, 3.4, 0.9]), th.tensor([0.0, 0.0, 0.0, 1.0])),
                                eef_pose=lambda a: (th.tensor([6.0, 3.32, 1.17]), th.tensor([0.0, 0.0, 0.0, 1.0])))
        ctx = {}
        for _ in EX.execute_base_follow(follow, motions, state, follow.targets(sc, None), ctx):
            pass
        return motions.legs, ctx["follow"], art

    legs, note, art = run(0.75, 0.75 * 1.4)
    assert legs == 1 and "extended" not in note and note["handover"] == {"follow_to": 0.75, "q": pytest.approx(1.05)}
    legs, note, art = run(0.75, 0.9)
    band = EX.CONFIG.follow_stop_frac * EX.joint_tolerance(art)
    assert legs == 2 and 1.05 - band <= note["extended"]["q"] < 1.05 + 0.04
    legs, note, art = run(None, 0.75 * 1.4)
    assert legs == 2 and "handover" not in note and note["extended"]["q"] >= 1.4 - band


def _lever_lift_push(arm="left", obj="carton.n.02_1"):
    from ispatialgym.oracle.sidecar import Pose as SPose, PushContact

    q = [0.851, -0.311, 0.397, 0.145]
    lift = PushContact(arm=arm, object=obj, pre=SPose([6.2804, 3.8030, 0.9745], q), init=SPose([6.2802, 3.7328, 0.9623], q), mid=SPose([6.2802, 3.7328, 0.9952], q),
                       goal=SPose([6.2802, 3.7328, 1.0280], q), direction=[0.0, 0.0, 1.0], distance_m=0.0658,
                       extra={"hold": True, "approach_is_slide": True, "articulation": {"joint": "j_right_lid", "link": "right_lid", "q_from": 0.0, "q_to": 0.35}})
    return Push(contact=lift, base_yaw=0.0)


def test_lever_retreat_slides_out_before_lifting_when_the_other_hand_holds_the_flap():
    """Two-flap lever (0926-073750 retreat[6]): the fist sits under the flap it lifted and the other hand pinches that flap - the
    recorded lift-first leg drove the fingers into the held flap and tipped the box 1.5 deg / 16 mm. With the object in the other
    hand the fist slides out along ``slide_dir`` first and lifts after (same end pose); with nothing held the recorded order stands."""
    from ispatialgym.oracle.actions import Retreat

    retreat = Retreat(arm="left", object="carton.n.02_1", after=_lever_lift_push(), distance=0.08, lift_first_m=0.03, slide_dir=[0.0, 1.0, 0.0])
    here, quat = th.tensor([6.2802, 3.7328, 1.0280]), th.tensor([0.851, -0.311, 0.397, 0.145])
    box = FakeObject("target_carton_0", [6.2883, 3.7464, 0.8684], category="carton", extent=(0.23, 0.31, 0.19))
    for holder in ("right", None):
        state = SimpleNamespace(eef_pose=lambda arm: (here.clone(), quat.clone()), object_pose=lambda n: (box.pos.clone(), box.quat.clone()),
                                object=lambda n: box, held_object=lambda arm, h=holder: box if arm == h else None, robot=None)
        motions = _RetreatStubMotions(with_ik=False)
        motions.press_hold = {"arm": "left", "object": "carton.n.02_1", "link": "right_lid"}
        ctx: Dict[str, Any] = {"cfg": EX.ExecuteConfig(), "witnesses": {}, "hooks": {}}
        list(EX.execute_retreat(retreat, motions, state, retreat.targets(None, state), ctx))
        leg = motions.legs[-1]
        assert motions.press_hold is None and th.allclose(leg[-1], here + th.tensor([0.0, 0.08, 0.03]), atol=1e-5)
        if holder is not None:
            assert th.allclose(leg[0], here + th.tensor([0.0, 0.08, 0.0]), atol=1e-5)
            assert ctx["retreat_slide_first"] == {"held_by": "right", "link": "right_lid", "slide_m": 0.08, "lift_m": 0.03}
        else:
            assert float(leg[0][1]) == pytest.approx(float(here[1]), abs=1e-5) and float(leg[0][2]) > float(here[2]) + 1e-3
            assert "retreat_slide_first" not in ctx


def test_base_follow_unwind_stops_re_reading_when_the_free_body_turns():
    """The unwind re-reads the measured joints every step; a loaded arm then creeps and a free-standing body turns with it
    (carton lever 0926-073750: 0.08 deg per step). It stops re-reading on the first step the object's root is not still; a root
    that stays put (a fixed-base door) unwinds over all ``follow_unwind_steps``."""
    sc = Sidecar.load(os.path.join(SIDECARS, "configuration_matching-articulation_open_small_scale.json"))
    follow = [a for a in expand(sc.variant, sc) if isinstance(a, BaseFollow)][0]
    art = follow.articulation
    art.passive_from = None
    art.extra = {k: v for k, v in (art.extra or {}).items() if k not in ("hinge_world", "follow_to")}
    cfg = EX.ExecuteConfig()

    def run(yaw_step):
        q, yaw = [art.q_init], [0.0]

        class _Motions:
            follow_phase = "arm"
            unwinds = 0

            def arm_along(self, arm, waypoints, converge=True, **kw):
                for _ in range(40):
                    q[0] += (art.q_goal - art.q_init) / 30
                    yield th.zeros(23)

            def unwind(self, arm):
                self.unwinds += 1
                return 0.001

            def hold(self, steps=1):
                for _ in range(int(steps)):
                    yaw[0] += yaw_step if self.unwinds else 0.0
                    yield th.zeros(23)

        motions = _Motions()
        state = SimpleNamespace(joint_position=lambda o, j: float(q[0]),
                                object_pose=lambda o: (th.tensor([6.0, 3.4, 0.9]), C.quat_from_axis_angle(th.tensor([0.0, 0.0, 1.0]), yaw[0])))
        ctx: Dict[str, Any] = {"cfg": cfg}
        list(EX.execute_base_follow(follow, motions, state, follow.targets(sc, None), ctx))
        return motions.unwinds, ctx["follow"]

    unwinds, note = run(math.radians(0.08))
    assert unwinds == 1 and note["unwind_stopped"]["after_steps"] == 1 and note["unwind_stopped"]["root_step_rad"] > cfg.push_still_rot
    unwinds, note = run(0.0)
    assert unwinds == cfg.follow_unwind_steps and "unwind_stopped" not in note


def _held_marker(joints=None):
    robot = FakeRobot()
    env = FakeEnv(robot, {})
    m = Motions(env, FakeBackend(robot), MotionsConfig(gripper_steps=30, settle_steps=5, cartesian_step=0.05, cartesian_rot_step=0.3))
    ep, _ = robot.eef_pose("right")
    marker = FakeObject("marker_xakoic_0", ep.tolist(), category="marker", extent=(0.1536, 0.0171, 0.0141), joints=joints)
    env.objects["marker.n.03_1"] = marker
    drive(env, m.gripper_opening("right", 0.5 * 0.01585 + 0.010))
    drive(env, m.gripper("right", close=True, rate=0.00225))
    assert robot._ag_obj_in_hand["right"] is marker
    return robot, env, m, marker


def test_release_check_records_how_long_the_weld_outlived_the_open_command(tmp_path):
    """Our Behavior-1K fork's pad-clear release logs each delayed release (``Robot.ag_release_log``); the release check copies
    the delay and what ended it ("pads_clear" / "cap") into the release it records at the step the weld went (the command +1
    there, no violation) and consumes the entry."""
    sc, env = iors_world()
    ex, motions, *_ = make_executor(env, sc, tmp_path)
    robot = env.robot
    robot.ag_release_log = [{"arm": "right", "object": "apple_omzprq_0", "by": "pads_clear", "delay_steps": 12, "delay_s": 0.1}]
    action = th.zeros(robot.action_dim)
    action[robot.controller_action_idx["gripper_right"]] = 1.0
    ex._check_releases({"left": None, "right": "apple_omzprq_0"}, {"left": None, "right": None}, action)
    rec = ex.release_check["releases"][-1]
    assert rec["object"] == "apple_omzprq_0" and rec["gripper_command"] == 1.0
    assert rec["weld_delay_s"] == 0.1 and rec["weld_delay_physics_steps"] == 12 and rec["weld_released_by"] == "pads_clear"
    assert robot.ag_release_log == [] and not ex.release_check["violations"]


def test_free_object_is_released_by_the_open_command():
    """Release semantics (owner 2026-10-03 / 2026-10-10): a held free object leaves the hand only through OmniGibson's rule -
    the open command (+1) in one step with the hand still, the assisted grasp letting go at that step; nothing is released
    directly and nothing is staged."""
    robot, env, m, marker = _held_marker()
    seen = []
    cmd_idx = robot.controller_action_idx["gripper_right"]
    env.after_step = lambda a: seen.append((float(a[cmd_idx]), robot._ag_obj_in_hand["right"]))
    robot.release_grasp_immediately = lambda arm: (_ for _ in ()).throw(AssertionError("direct release"))
    drive(env, m.gripper("right", close=False))
    env.after_step = None
    assert seen[0] == (1.0, None), "let go at the first step, whose command is exactly +1"
    assert not any(e.get("event") == "staged" for e in m.release_events)


def test_attach_rule_acts_at_the_release(monkeypatch):
    """Attach tasks (owner 2026-10-06 / 2026-10-10): at the instant the assisted grasp lets go, the release hook hands a registered
    attachment child to the task's attach rule (``released``: the in-hand registry still names it until the release window ends;
    the rule itself decides - it joins only a part already touching its partner); any other released object is left alone."""
    from ispatialgym.sampling import predicates as PRED

    calls = []

    def fake_attach(child, parent, equivalences=None, in_place=True, released=False, trigger="per_step"):
        calls.append((child.name, parent.name, released, trigger))
        return {"ok": True, "attached": True, "joint": "in_place", "fix": "T"}

    monkeypatch.setattr(PRED, "attach_if_aligned", fake_attach)
    reg = {n: SimpleNamespace(name=n) for n in ("cork_0", "wine_bottle_0", "apple_0")}
    scene = SimpleNamespace(object_registry=lambda key, name: reg.get(name))
    for o in reg.values():
        o.scene = scene
    PRED.set_attachment_context([{"child": "cork_0", "parent": "wine_bottle_0", "equivalences": []}])
    try:
        rec = PRED.attach_on_release(None, "right", reg["cork_0"])
        assert calls == [("cork_0", "wine_bottle_0", True, "release_hook")] and rec["hook"] == "release" and rec["parent"] == "wine_bottle_0"
        assert PRED.LAST_RELEASE_ATTACH["attached"] and "fix" not in PRED.LAST_RELEASE_ATTACH
        assert PRED.attach_on_release(None, "right", reg["apple_0"]) is None and len(calls) == 1
    finally:
        PRED.set_attachment_context([])
    # the task installs the hook on every robot of env.robots (one list per scene), once
    robots = [SimpleNamespace(), SimpleNamespace(release_callbacks=[print])]
    assert PRED.install_release_hook([[robots[0]], [robots[1]]]) == 2 and PRED.install_release_hook([robots]) == 2
    assert robots[0].release_callbacks == [PRED.attach_on_release] and robots[1].release_callbacks == [print, PRED.attach_on_release]


def test_articulated_sheet_is_released_by_the_same_one_step_opening():
    """A flap (an articulated part) pinched across its thickness opens like every release (owner 2026-10-10: never a partial
    opening): +1 in one step, which releases it."""
    robot, env, m, marker = _held_marker(joints=["flap_joint"])
    drive(env, m.gripper("right", close=False))
    assert robot._ag_obj_in_hand["right"] is None and m.release_events[-1]["event"] == "instant_open"


def _arching(backend, robot, arm, rise):
    idx = robot.arm_control_idx[arm]
    orig = backend.plan_pose
    n = {"calls": 0}

    def plan_pose(emb, targets, q_start=None, max_attempts=4, sync_world=True):
        res = orig(emb, targets, q_start=q_start, max_attempts=max_attempts, sync_world=sync_world)
        n["calls"] += 1
        if n["calls"] == 1 and res.success:
            q = res.q_traj.clone()
            k = th.linspace(0, 1, len(q))
            q[:, idx[2]] = q[:, idx[2]] + (rise / ARM_SCALE) * th.sin(k * th.pi)  # toy FK: arm joint 3 is the hand height
            res.q_traj = q
        return res

    def fk_batch(frame, q, emb_name="whole_body"):
        q = C.as_tensor(q)
        return th.stack([robot.pos + C.quat_rotate(robot.quat, ARM_OFFSET[arm] + ARM_SCALE * qq[idx][:3]) for qq in q]), None

    backend.plan_pose = plan_pose
    backend.fk_batch = fk_batch
    return n


def test_a7_clearing_pre_grasp_plan_that_detours_takes_the_via_above(tmp_path):
    """First-demo v2 A7 (kielbasa 0926-074242: a graph-seeded plan arched 1.27 m / 19 cm high for a 0.53 m move): a
    clearing pre-grasp plan that detours past the via-above's envelope takes the via-above."""
    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc, tmp_path)
    arm = "right"
    p0, q0 = motions.eef_pose(arm)
    target = (p0 + th.tensor([0.10, 0.0, -0.02]), q0)
    backend.solve_ik = None
    motions._protect_violations = lambda arms, q_traj, emb, P=None: []
    _arching(backend, env.robot, arm, rise=0.40)
    saucer = [("saucer.n.02_1", SimpleNamespace(name="partner_saucer_mgbeah_0"))]
    with motions.protecting(saucer, leg="pre_grasp", task="t", action="grasp", index=1):
        drive(env, motions.arm_to(arm, pose=target))
    ev = motions.protect_events[-1]
    assert ev["remedy"].startswith("via-above (the clearing plan detoured") and ev["objects"][0]["rise_m"] > 0.3, ev
    assert float(th.linalg.norm(motions.eef_pose(arm)[0] - target[0])) < 5e-3


def test_a7_plain_plan_and_other_legs_are_unchanged(tmp_path):
    sc, env = iors_world()
    ex, motions, backend, _ = make_executor(env, sc, tmp_path)
    arm = "right"
    p0, q0 = motions.eef_pose(arm)
    backend.solve_ik = None
    motions._protect_violations = lambda arms, q_traj, emb, P=None: []
    saucer = [("saucer.n.02_1", SimpleNamespace(name="partner_saucer_mgbeah_0"))]
    # a modest hump (8 cm) on a pre-grasp transit: below the via-above envelope, executed as planned
    n = _arching(backend, env.robot, arm, rise=0.08)
    with motions.protecting(saucer, leg="pre_grasp", task="t", action="grasp", index=1):
        drive(env, motions.arm_to(arm, pose=(p0 + th.tensor([0.10, 0.0, 0.0]), q0)))
    assert motions.protect_events == [] and n["calls"] == 1
    # a big hump on another leg (the carry's finish): not judged
    p1, _ = motions.eef_pose(arm)
    backend2 = backend
    backend2.plan_pose = FakeBackend.plan_pose.__get__(backend2)
    n2 = _arching(backend2, env.robot, arm, rise=0.40)
    with motions.protecting(saucer, leg="carry", task="t", action="place", index=3):
        drive(env, motions.arm_to(arm, pose=(p1 + th.tensor([-0.10, 0.0, 0.0]), q0)))
    assert motions.protect_events == [] and n2["calls"] == 1


def test_lever_grasp_transit_takes_the_witness_branch_when_the_detour_branch_stops_short_of_the_grasp():
    """The lever's pinch comes from the brace beside the box by a Cartesian detour, which keeps the brace's IK branch. When that
    branch cannot reach the grasp (0927-023811: right_arm_joint2 at its limit, 11 mm short), the transit plans to the pre-grasp
    witness's branch - where the sampler checked the approach - instead of executing the detour; when it can, the detour runs."""
    here = (th.tensor([6.126, 3.572, 1.026]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    pre = (th.tensor([6.084, 3.687, 1.077]), th.tensor([0.641633, 0.498948, 0.499852, 0.299175]))
    grasp = (th.tensor([6.178, 3.699, 1.045]), th.tensor([0.641633, 0.498948, 0.499852, 0.299175]))
    w_pre = th.zeros(28)

    class _M:
        def __init__(self, reaches):
            self.reaches, self.calls = reaches, []

        def along_reaches(self, arm, waypoints):
            self.calls.append(("probe", len(waypoints)))
            return self.reaches

        def arm_along(self, arm, legs, **kw):
            self.calls.append(("along", len(legs)))
            yield th.zeros(23)

        def arm_to(self, arm, pose=None, q_hint=None, **kw):
            self.calls.append(("arm_to", q_hint is not None))
            yield th.zeros(23)

    action = SimpleNamespace(arm="right", grasp=SimpleNamespace(extra={}))
    state = SimpleNamespace(eef_pose=lambda arm: (here[0].clone(), here[1].clone()))
    for reaches in (False, True):
        m, ctx = _M(reaches), {"cfg": EX.ExecuteConfig(), "witnesses": {"pre_grasp": w_pre}}
        list(EX._grasp_transit(action, m, state, {"pre_grasp": pre, "grasp": grasp}, ctx, "right", [], [("box", "right_lid")]))
        if reaches:
            assert m.calls == [("probe", 4), ("along", 4)] and ctx["pre_grasp_transit"].startswith("cartesian detour")
        else:
            assert m.calls == [("probe", 4), ("arm_to", True)] and ctx["pre_grasp_transit"].startswith("witness branch")


def test_ag_joint_rule_is_the_tasks_not_the_oracles():
    """The assisted-grasp joint of a link behind a movable joint is a rule of the environment, not of the oracle:
    ``ISpatialGymTask`` sets our Behavior-1K fork's ``Robot.ag_moving_link_joint_type`` (fixed) on every robot, so a recorded
    demo and an evaluated policy (``run_isg --mode replay / policy``) meet the same joint. No oracle module overrides
    OmniGibson's joint choice or the setting (the 2026-10 evaluation found the oracle-only fixed joint: replays of the
    blender demos under the stock spherical joint never opened the lid)."""
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task = open(os.path.join(pkg, "tasks", "ispatialgym_task.py")).read()
    assert 'AG_MOVING_LINK_JOINT_TYPE = "FixedJoint"' in task
    assert "robot.ag_moving_link_joint_type = AG_MOVING_LINK_JOINT_TYPE" in task
    for path in glob.glob(os.path.join(pkg, "oracle", "*.py")):
        src = open(path).read()
        assert "_get_assisted_grasp_joint_type =" not in src, path
        assert "ag_moving_link_joint_type" not in src, path


def test_ag_release_rule_is_the_tasks_not_the_oracles():
    """The assisted grasp's pad-clear release (owner 2026-10-10: the weld stays after the open command until a pad has left the
    object, at most 0.2 s) is a rule of the environment: ``ISpatialGymTask`` sets our Behavior-1K fork's
    ``Robot.ag_release_on_pad_clear`` / ``ag_release_max_delay_s`` on every robot, so demo collection, replays and an evaluated
    policy meet the same release; no oracle module sets them (the executor only reads the fork's ``ag_release_log``)."""
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task = open(os.path.join(pkg, "tasks", "ispatialgym_task.py")).read()
    assert "AG_RELEASE_ON_PAD_CLEAR = True" in task and "AG_RELEASE_MAX_DELAY_S = 0.2" in task
    assert "robot.ag_release_on_pad_clear = AG_RELEASE_ON_PAD_CLEAR" in task and "robot.ag_release_max_delay_s = AG_RELEASE_MAX_DELAY_S" in task
    for path in glob.glob(os.path.join(pkg, "oracle", "*.py")):
        src = open(path).read()
        assert "ag_release_on_pad_clear =" not in src and "ag_release_max_delay_s =" not in src, path


def test_settle_eval_off_takes_the_first_step_the_goal_holds(tmp_path):
    """Settle-eval off (run_isg --settle-eval off, the default of --mode policy): symbolic success is the first step the goal
    holds, geometric success the first step it holds with every measured target within tolerance - no pose_settled window;
    no other mode is reported (``other_mode`` None)."""
    sc, env = iors_world()
    goal = _place_goal(sc)

    def eval_unsettled(env_):
        d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
        d["settled"] = False
        return d

    env.evaluate = eval_unsettled
    env.task.tracker = MET.SuccessTracker(settle_eval=False)
    ex, *_ = make_executor(env, sc, tmp_path / "a")
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    assert result["settle_eval"] is False and result["other_mode"] is None
    # a goal that held a few steps and then not any more still counts: its first step decides
    sc, env = iors_world()
    hits = {"n": 0}

    def eval_transient(env_):
        d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
        if d["binary_success"]:
            hits["n"] += 1
            if hits["n"] > 3:
                d["binary_success"] = False
        return d

    env.evaluate = eval_transient
    env.task.tracker = MET.SuccessTracker(settle_eval=False)
    ex, *_ = make_executor(env, sc, tmp_path / "c")
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"] and result["geometric_success_step"] == result["symbolic_success_step"] == result["goal_first_step"]
    assert result["errors"] is not result["errors_final_step"]  # judged at the geometric success, not at the last step


def test_wall_clock_ends_a_stream_and_times_out_a_program(tmp_path):
    """``ExecutorConfig.max_wall_s``: a stream (replay / policy) stops taking actions and the task's metrics judge the episode,
    as at the end of its steps; a program gets a ``timeout`` verdict."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"] and result["wall_clock_exceeded"] is False
    actions = [np.asarray(a) for a in rec.rows["action"][: result["n_action_steps"]]]
    sc2, env2 = iors_world()
    env2.evaluate = success_eval("apple.n.01_1", _place_goal(sc2), tol=0.08)
    ex2, *_ = make_executor(env2, sc2, tmp_path / "stream", exec_cfg=ExecutorConfig(max_wall_s=1e-9))
    result2 = ex2.run_stream(th.as_tensor(a) for a in actions)
    assert result2["wall_clock_exceeded"] and result2["n_steps"] == 0 and result2["epilogue_steps"] == 0
    assert not result2["success"] and result2["reason"] == "not_reached"  # judged by the metrics: the apple was never moved
    sc3, env3 = iors_world()
    env3.evaluate = success_eval("apple.n.01_1", _place_goal(sc3), tol=0.08)
    ex3, *_ = make_executor(env3, sc3, tmp_path / "program", exec_cfg=ExecutorConfig(max_wall_s=1e-9))
    result3 = ex3.run_program(expand(sc3.variant, sc3))
    assert result3["reason"] == "timeout" and "wall clock" in result3["verdict"]["message"] and result3["wall_clock_exceeded"]


def test_program_epilogue_is_reserved_in_the_step_budget(tmp_path):
    """A program's actions get ``max_steps - settle_steps`` steps (``Executor.action_budget``): its epilogue always fits; a program
    that needs more times out before it (no epilogue then). A stream ends at ``max_steps`` without a timeout verdict and the
    metrics judge it."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    result = ex.run_program(expand(sc.variant, sc))
    n_action = result["n_action_steps"]
    # exactly enough: the program and its 50-step epilogue
    sc2, env2 = iors_world()
    env2.evaluate = success_eval("apple.n.01_1", _place_goal(sc2), tol=0.08)
    ex2, *_ = make_executor(env2, sc2, tmp_path / "fit", exec_cfg=ExecutorConfig(max_steps=n_action + 50, settle_steps=50))
    result2 = ex2.run_program(expand(sc2.variant, sc2))
    assert result2["success"], result2["verdict"]
    assert result2["n_steps"] == n_action + 50 and result2["epilogue_steps"] == 50 and ex2.action_budget == n_action
    # one step short: the actions time out at the action budget, the epilogue is not cut into
    sc3, env3 = iors_world()
    env3.evaluate = success_eval("apple.n.01_1", _place_goal(sc3), tol=0.08)
    ex3, *_ = make_executor(env3, sc3, tmp_path / "short", exec_cfg=ExecutorConfig(max_steps=n_action + 49, settle_steps=50))
    result3 = ex3.run_program(expand(sc3.variant, sc3))
    assert result3["reason"] == "timeout" and result3["epilogue_steps"] == 0 and result3["n_steps"] <= n_action
    # a stream: cut at max_steps, judged by the metrics
    actions = [np.asarray(a) for a in rec.rows["action"][: result["n_steps"]]]
    sc4, env4 = iors_world()
    env4.evaluate = success_eval("apple.n.01_1", _place_goal(sc4), tol=0.08)
    ex4, *_ = make_executor(env4, sc4, tmp_path / "stream", exec_cfg=ExecutorConfig(max_steps=20))
    result4 = ex4.run_stream(th.as_tensor(a) for a in actions)
    assert result4["n_steps"] == 20 and result4["reason"] == "not_reached"
