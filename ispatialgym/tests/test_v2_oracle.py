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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import qa as QA
from ispatialgym.oracle.actions import VOCABULARY, Navigate, Push, BaseFollow, Place, Brace as QA_Brace
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
                diag = {"stage": "diff_ik", "fail_index": i, "n_waypoints": len(waypoints), "pos_err": stall[1], "rot_err": 0.001, "at_lower": [], "at_upper": [], "converged": i}
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
        self.objects_by_name = {o.name: o for o in objects.values()}
        self.objects_by_name[robot.name] = robot
        self.object_registry = FakeRegistry(self.objects_by_name)

    def save(self):
        return {"state": {}, "objects_info": {}}


class FakeTask:
    def __init__(self, objects: Dict[str, Any], robot):
        self.object_scope = dict(objects)
        self.object_scope[AGENT_INSTANCE] = robot
        self.compiled_task = None
        self._first_success_step = -1
        self._first_settled_success_step = -1
        self._errors_at_first_success = None
        self._last_goal_eval = None

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
    def __init__(self, robot: FakeRobot, objects: Dict[str, FakeObject], evaluate=None, ag_roundtrip=True):
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
        # the fake renderer: like the simulator, a frame shows the scene the renderer last synced, one step behind
        # the physics (after ``load_state`` the first draw still shows the pre-restore scene)
        self._usd_state = self._render_state()
        self._frame = self._image_of(self._usd_state)

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
        self._frame = self._image_of(self._usd_state)  # draws the last synced scene ...
        self._usd_state = self._render_state()  # ... then syncs the current physics state for the next draw

    def frame_matches_state(self, rgb, state=None) -> bool:
        """Whether an observed frame (rgb or rgba) shows ``state`` (default: the current physics state)."""
        want = self._image_of(self._render_state() if state is None else state)[..., :3]
        return bool(th.equal(th.as_tensor(np.asarray(rgb))[..., :3], want))

    def _obs(self):
        return {self.robot.name: {CAM_KEY: {"rgb": self._frame.clone()}, "proprio": self.robot.q.clone()}}

    def _info(self):
        isg = self.evaluate(self)
        isg.setdefault("binary_success_ever", isg.get("binary_success", False))
        isg.setdefault("settled_success", isg.get("binary_success", False) and isg.get("settled", False))
        isg.setdefault("errors", {})
        isg.setdefault("conditions", [])
        isg.setdefault("first_settled_success_step", 0 if isg["settled_success"] else -1)
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


def iors_world(ag_roundtrip=True, evaluate=None):
    """The IORS ontop small-scale test sidecar as a fake world: robot at the init stance, apple / plate on the table."""
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-ontop_small_scale.json"))
    init = sc.stance("init")
    robot = FakeRobot(base=tuple(init.base_pose2d))
    g = sc.grasps[0]
    apple = FakeObject("apple_omzprq_0", [g.grasp.position[0], g.grasp.position[1], g.grasp.position[2] - 0.03], category="apple")
    plate = FakeObject("plate_pkkgzc_0", [g.place.position[0], g.place.position[1], g.place.position[2] - 0.06], category="plate", extent=(0.2, 0.2, 0.02))
    objects = {"apple.n.01_1": apple, "plate.n.04_1": plate}
    env = FakeEnv(robot, objects, evaluate=evaluate, ag_roundtrip=ag_roundtrip)
    return sc, env


def make_executor(env, sc, tmp_path=None, recorder_cfg=None, exec_cfg=None, qa_cfg=None, contacts=None, camera=None):
    backend = FakeBackend(env.robot)
    motions = Motions(env, backend, MotionsConfig(settle_steps=5, gripper_steps=20, cartesian_step=0.05, cartesian_rot_step=0.3))
    qa_cfg = qa_cfg or QA.QAConfig(pause_steps=100000, fov_min_steps=100000)
    monitors = QA.Monitors(env, env.robot, cfg=qa_cfg, camera=camera)
    monitors._contact_api = contacts if contacts is not None else FakeContacts(lambda step, query: [])
    recorder = None
    if tmp_path is not None:
        recorder = Recorder(str(tmp_path), env, env.robot, CAM_KEY, dict(env.objects), fps=30, cfg=recorder_cfg or RecorderConfig(hdf5=True, lerobot=False, video=False), instance_id=sc.instance_id)
        recorder.start(env.get_obs()[0], {}, state0=env.sim.dump_state(serialized=True))
    cfg = exec_cfg or ExecutorConfig(settle_min_steps=50, settle_fraction=0.2)
    ex = Executor(env, env.task, sc, recorder=recorder, motions=motions, monitors=monitors, cfg=cfg, sim=env.sim, tracked_objects=dict(env.objects), cam_key=CAM_KEY)
    return ex, motions, backend, recorder


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
        for action in expand(sc.alias, sc):
            ex_fn, post_fn = EX.executor_for(action)
            assert ex_fn is not None and post_fn is not None


def test_postcondition_base_follow_progress():
    sc = Sidecar.load(os.path.join(SIDECARS, "configuration_matching-articulation_open_large_scale.json"))
    follow = [a for a in expand(sc.alias, sc) if isinstance(a, BaseFollow)][0]
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


def test_postcondition_push_no_motion():
    sc = Sidecar.load(os.path.join(SIDECARS, "alignment-axial.json"))
    push = [a for a in expand(sc.alias, sc) if isinstance(a, Push)][0]
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
    push = [a for a in expand(sc.alias, sc) if isinstance(a, Push)][0]
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


def test_push_geometry_of_a_hanging_painting_is_an_arc_about_the_hook_axis():
    """The wall-vertical push swings a painting about its horizontal hook axis: the recorded init / mid / goal pad
    poses lie on a circle in a vertical plane. The geometry is an arc about that (horizontal) axis — not the planar
    'line of 8 mm' the xy projection gave (which measured the residual on the object's position, i.e. on the hook
    that never moves) — the residual is the rotation still missing about the hook, an overshoot flips its sign, the
    extension stays on the circle and the back-off tangent points down the arc."""
    sc = Sidecar.load(os.path.join(SIDECARS, "alignment-world_vertical.json"))
    push = [a for a in expand(sc.alias, sc) if isinstance(a, Push)][0]
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
    actions = expand(sc.alias, sc)
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
    actions = expand(sc.alias, sc)
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
        return {"binary_success": err < tol, "settled": True, "errors": {"pose_errors": {obj: {"orientation_error": err, "within_tolerance": err < tol, "gating": True}}}}

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
    """Owner decisions 2026-09-16 05:26 / 05:39: a braced push ending out of tolerance is re-run from the checkpoint
    before the brace with the brace +5 / +10 mm outward when the target touched the brace finger before the goal
    (short), −5 / −10 mm inward when the residual flipped sign (overshoot); two retries, then ``tolerance``."""
    # (a) short: the stop blocks the board at 80 % of the arc with the recorded brace, at 100 % once it stands 5 mm further out
    sc, env, actions, board, brace, push, geom = axial_world(gain=1.0, stop=lambda off: 0.8 + 40.0 * off)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)

    def brace_touch(step, query):
        if ex.monitors.action is not None and ex.monitors.action.name == "push":
            return [(f"{env.robot.prim_path}/left_gripper_finger_link1", f"{board.prim_path}/base_link")]
        return []

    ex.monitors._contact_api = FakeContacts(brace_touch)
    result = ex.run_program(actions)
    assert result["success"], result["verdict"]
    assert len(result["brace_retry"]) == 1
    br = result["brace_retry"][0]
    assert [r["reason"] for r in br["retries"]] == ["short"] and br["retries"][0]["next_offset_m"] == 0.005 and br["final_offset_m"] == 0.005
    assert br["retries"][0]["brace_hits"] > 0 and br["retries"][0]["residual"] > 0.10
    assert br["final"]["in_tolerance"] and [a["name"] for a in result["actions"]] == ["transition", "brace", "push", "retreat", "retreat"]
    # the re-run brace stood 5 mm outward along -direction of the recorded contact (offset_brace geometry)
    moved = EX.offset_brace(brace, 0.005)
    assert abs(float(th.dot(brace.contact.contact.tensors()[0] - moved.contact.contact.tensors()[0], th.tensor(brace.contact.direction[:3]).float())) - 0.005) < 1e-6
    assert moved.contact.extra["retry_offset_m"] == 0.005 and moved.contact.pre.position != brace.contact.pre.position
    # the record is consistent: committed rows = committed steps, the rolled-back group kept aside
    assert len(rec.rows["step"]) == result["n_steps"] and np.all(np.diff(np.asarray(rec.rows["step"])) == 1)
    assert any(r["attempt"] == 0 and r["action_index"] == 1 for r in rec.rollbacks)
    # (b) overshoot: the board over-rotates 15 % and the recorded stop stands too far to catch it; 5 mm inward it does
    sc, env, actions, board, brace, push, geom = axial_world(gain=1.15, stop=lambda off: 1.15 + 30.0 * off)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path / "b")
    result = ex.run_program(actions)
    assert result["success"], result["verdict"]
    br = result["brace_retry"][0]
    assert [r["reason"] for r in br["retries"]] == ["overshoot"] and br["retries"][0]["next_offset_m"] == -0.005 and br["retries"][0]["residual"] < -0.10
    # (c) hopeless: the stop blocks at 60 % whatever the brace does -> two short retries (+5, +10 mm), then tolerance
    sc, env, actions, board, brace, push, geom = axial_world(gain=1.0, stop=lambda off: 0.6)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path / "c")
    ex.monitors._contact_api = FakeContacts(lambda step, query: [(f"{env.robot.prim_path}/left_gripper_finger_link2", f"{board.prim_path}/base_link")] if ex.monitors.action is not None and ex.monitors.action.name == "push" else [])
    result = ex.run_program(actions)
    assert result["reason"] == "tolerance"
    br = result["brace_retry"][0]
    assert [r["next_offset_m"] for r in br["retries"]] == [0.005, 0.010] and br["final_offset_m"] == 0.010 and br["final"]["reason"] == "short"
    # (d) short without a brace contact (the board just slips) is not the brace's fault: no ladder, straight to tolerance
    sc, env, actions, board, brace, push, geom = axial_world(gain=0.6)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path / "d")
    result = ex.run_program(actions)
    assert result["reason"] == "tolerance" and result["brace_retry"] == []


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
    actions = expand(sc.alias, sc)
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
    assert ex.run_action(1, expand(sc.alias, sc)[1]) is None  # the retry from the checkpoint succeeds (the fake stall is one-shot)
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


def test_failed_attempt_leaves_no_frame_in_the_committed_episode(tmp_path, monkeypatch):
    """Owner review 2026-09-16: frames of a rolled-back attempt survived in ``head_rgb.mp4``. The renderer trails the
    physics, so the restore's recorded hold step showed the failed attempt's last arm pose. Every committed row,
    video frame and LeRobot frame must belong to the committed steps, and every committed frame must show the state
    of its own row (the fake renderer lags one frame like the simulator)."""
    monkeypatch.setattr(Recorder, "_open_video", lambda self: setattr(self, "_writer", _CollectingWriter()))
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, recorder_cfg=RecorderConfig(hdf5=True, lerobot=False, video=True, frame_buffer="raw"))
    backend.fail_ik_paths = 1  # the grasp approach fails once: attempt 0 has N > 0 recorded steps (gripper + arm_to)
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"], result["verdict"]
    grasp = result["actions"][1]
    assert grasp["attempts"][0]["category"] == "plan_failed" and grasp["attempts"][1] is None
    n_failed = grasp["attempts"][0]["step"] - grasp["step_start"]
    assert n_failed > 0
    rows = rec.rows
    steps = np.asarray(rows["step"])
    assert len(steps) == result["n_steps"] and np.all(np.diff(steps) == 1)  # committed steps only, contiguous
    frames = rec._writer.frames
    assert len(frames) == result["n_steps"]
    # every committed frame shows a state of the committed sequence: its own row's or (the renderer's one-frame lag)
    # the previous committed row's — never the failed attempt's (the restore hold step used to show its last pose)
    q = np.asarray(rows["joint_positions"])
    base = np.asarray(rows["base_pose"])
    obj_pose = {name: np.asarray(rows[f"object_pose/{name}"]) for name in env.objects}

    def state_of(i):
        return th.cat([th.tensor(q[i]).float(), th.tensor(base[i][:3]).float()] + [th.cat([th.tensor(obj_pose[n][i][:3]).float(), th.tensor(obj_pose[n][i][3:]).float()]) for n in env.objects])

    states = [state_of(i) for i in range(len(steps))]
    for i in range(1, len(steps)):
        ok = env.frame_matches_state(frames[i], states[i]) or env.frame_matches_state(frames[i], states[i - 1])
        assert ok, f"frame at committed row {i} (step {steps[i]}) shows a state outside the committed episode"
    # and the boundary frame in particular is the restored state, not the failed attempt's last one
    b = grasp["step_start"]  # row index of the restore hold step (steps are 1-based)
    paths = rec.close({"instance_id": sc.instance_id, "success": True}, qa=result["qa"]["stats"], success=True)
    with h5py.File(paths["hdf5"]) as f:
        g = f["data/demo_0"]
        assert len(g["isg/step"]) == result["n_steps"]
        rb = g["rollbacks/action_1_attempt_0"]
        assert len(rb["isg/step"]) == n_failed and int(rb["isg/step"][0]) == grasp["step_start"] + 1
        q_fail = th.tensor(rb["isg/joint_positions"][-1]).float()
    failed_state = th.cat([q_fail, states[b][28:]])
    assert not env.frame_matches_state(frames[b], failed_state) and not env.frame_matches_state(frames[b + 1], failed_state)


def test_placed_orientation_gates_the_demo_verdict_only(tmp_path):
    """Owner decision 2026-09-16 10:15 item 2: with ``demo_gating=("placed_orientation",)`` (run_isg oracle default) a
    placed object whose final orientation error exceeds its recorded target tolerance fails the demo with
    ``placed_orientation`` (errors in ``demo_gating``); the TRO target stays non-gating for the evaluation, and
    without the gate the same run is a success."""
    goal_q = th.tensor([0.0, 0.0, math.sin(0.25), math.cos(0.25)])  # the apple's recorded orientation target (0.5 rad yaw)

    def world(err_rad):
        sc, env = iors_world()
        goal = _place_goal(sc)
        apple = env.objects["apple.n.01_1"]
        env.task._tro = SimpleNamespace(meta=SimpleNamespace(pose_targets={"apple.n.01_1": SimpleNamespace(frame="world", orientation=goal_q, orientation_tolerance=0.2,
                                                                                                                 orientation_metric="full", position=None, position_tolerance=None,
                                                                                                                 orientation_equivalences=[])}))

        def evaluate(env_):
            d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
            # the evaluator reports the (non-gating) IORS pose target: orientation error as measured, never gating
            d["errors"]["pose_errors"]["apple.n.01_1"].update({"orientation_error": err_rad, "gating": False, "within_tolerance": err_rad <= 0.2})
            return d

        env.evaluate = evaluate
        return sc, env

    # (a) the cup fell at release: 0.46 rad -> placed_orientation, the position-only success notwithstanding
    sc, env = world(0.46)
    ex, *_ = make_executor(env, sc, tmp_path, exec_cfg=ExecutorConfig(settle_min_steps=50, settle_fraction=0.2, demo_gating=("placed_orientation",)))
    result = ex.run_program(expand(sc.alias, sc))
    assert result["reason"] == "placed_orientation" and result["binary_success_final"] and result["verdict"]["details"]["targets"] == ["apple.n.01_1"]
    dg = result["demo_gating"]
    assert dg["enabled"] == ["placed_orientation"] and dg["placed_orientation"]["apple.n.01_1"] == {"orientation_error": 0.46, "tolerance": 0.2, "within": False, "metric": "full", "target_gating": False}
    assert "plate.n.04_1" not in dg["placed_orientation"]  # only the placed objects, not the partner
    # (b) within tolerance -> success, the report kept
    sc, env = world(0.12)
    ex, *_ = make_executor(env, sc, tmp_path / "b", exec_cfg=ExecutorConfig(settle_min_steps=50, settle_fraction=0.2, demo_gating=("placed_orientation",)))
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"] and result["demo_gating"]["placed_orientation"]["apple.n.01_1"]["within"] is True
    # (c) the gate off (replay / policy modes): the 0.46 rad run passes, the evaluation is unchanged
    sc, env = world(0.46)
    ex, *_ = make_executor(env, sc, tmp_path / "c")
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"] and result["demo_gating"] == {"enabled": []}
    # (d) no orientation target on the placed object: nothing to gate
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, *_ = make_executor(env, sc, tmp_path / "d", exec_cfg=ExecutorConfig(settle_min_steps=50, settle_fraction=0.2, demo_gating=("placed_orientation",)))
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"] and result["demo_gating"]["placed_orientation"]["apple.n.01_1"]["within"] is None
    # (e) placed_tilt (the oracle default): a roll about the object's own up axis passes, a tipped object fails
    tilt_cfg = ExecutorConfig(settle_min_steps=50, settle_fraction=0.2, demo_gating=("placed_tilt",))
    sc, env = world(0.46)
    apple = env.objects["apple.n.01_1"]
    yaw = 0.5 + 0.46  # the apple rolled 0.46 rad about its own (vertical) axis relative to the goal yaw 0.5

    def roll(step_action):
        if env.robot._ag_obj_in_hand["right"] is None and float(th.linalg.norm(apple.pos - th.tensor(_place_goal(sc)))) < 0.08:
            apple.quat = th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])

    env.after_step = roll
    ex, *_ = make_executor(env, sc, tmp_path / "e", exec_cfg=tilt_cfg)
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"], result["verdict"]
    rep = result["demo_gating"]
    assert rep["placed_tilt"]["apple.n.01_1"]["within"] is True and rep["placed_tilt"]["apple.n.01_1"]["tilt_deg"] < 1.0
    assert rep["placed_orientation"]["apple.n.01_1"]["within"] is False  # the strict reading is reported alongside
    sc, env = world(0.46)
    apple = env.objects["apple.n.01_1"]

    def tip(step_action):
        if env.robot._ag_obj_in_hand["right"] is None and float(th.linalg.norm(apple.pos - th.tensor(_place_goal(sc)))) < 0.08:
            apple.quat = th.tensor([math.sin(0.25), 0.0, 0.0, math.cos(0.25)])  # 0.5 rad = 28.6 deg about x: the up axis tips

    env.after_step = tip
    ex, *_ = make_executor(env, sc, tmp_path / "f", exec_cfg=tilt_cfg)
    result = ex.run_program(expand(sc.alias, sc))
    assert result["reason"] == "placed_tilt" and abs(result["demo_gating"]["placed_tilt"]["apple.n.01_1"]["tilt_deg"] - 28.6) < 0.5
    # an equivalence that flips the object (180 deg about x) makes an upside-down placement acceptable for the tilt
    sc, env = world(0.46)
    env.task._tro.meta.pose_targets["apple.n.01_1"].orientation_equivalences = [th.tensor([1.0, 0.0, 0.0, 0.0])]
    apple = env.objects["apple.n.01_1"]

    def flip(step_action):
        if env.robot._ag_obj_in_hand["right"] is None and float(th.linalg.norm(apple.pos - th.tensor(_place_goal(sc)))) < 0.08:
            apple.quat = th.tensor([1.0, 0.0, 0.0, 0.0])

    env.after_step = flip
    ex, *_ = make_executor(env, sc, tmp_path / "g", exec_cfg=tilt_cfg)
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"] and result["demo_gating"]["placed_tilt"]["apple.n.01_1"]["tilt_deg"] < 1.0
    # (f) an elongated object (a marker lying flat, native bbox 15 x 1.7 x 1.4 cm) is judged on its long axis: a 92
    # deg roll about that axis at release (the pilot nextto re-run) is not a tip; standing up on its end is
    sc, env = world(0.46)
    marker = env.objects["apple.n.01_1"]
    marker.aabb_extent = th.tensor([0.1536, 0.0171, 0.0141])

    def roll_about_x(step_action):
        if env.robot._ag_obj_in_hand["right"] is None and float(th.linalg.norm(marker.pos - th.tensor(_place_goal(sc)))) < 0.08:
            marker.quat = C.qmul(th.tensor([0.0, 0.0, math.sin(0.25), math.cos(0.25)]), th.tensor([math.sin(0.8), 0.0, 0.0, math.cos(0.8)]))  # goal yaw, then 1.6 rad about the marker's own x

    env.after_step = roll_about_x
    ex, *_ = make_executor(env, sc, tmp_path / "h", exec_cfg=tilt_cfg)
    result = ex.run_program(expand(sc.alias, sc))
    rep = result["demo_gating"]["placed_tilt"]["apple.n.01_1"]
    assert result["success"] and rep["axis"] == "x" and rep["rule"] == "long" and rep["tilt_deg"] < 1.0, (result["verdict"], rep)
    assert result["demo_gating"]["placed_orientation"]["apple.n.01_1"]["within"] is False
    sc, env = world(0.46)
    marker = env.objects["apple.n.01_1"]
    marker.aabb_extent = th.tensor([0.1536, 0.0171, 0.0141])

    def stand_up(step_action):
        if env.robot._ag_obj_in_hand["right"] is None and float(th.linalg.norm(marker.pos - th.tensor(_place_goal(sc)))) < 0.08:
            marker.quat = C.qmul(th.tensor([0.0, 0.0, math.sin(0.25), math.cos(0.25)]), th.tensor([0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]))  # the long axis pitched up 90 deg

    env.after_step = stand_up
    ex, *_ = make_executor(env, sc, tmp_path / "i", exec_cfg=tilt_cfg)
    result = ex.run_program(expand(sc.alias, sc))
    assert result["reason"] == "placed_tilt" and abs(result["demo_gating"]["placed_tilt"]["apple.n.01_1"]["tilt_deg"] - 90.0) < 0.5


def test_placed_orientation_gate_compares_cylinders_modulo_their_long_axis(tmp_path, monkeypatch):
    """Owner decision 2026-09-16 17:10 item 4, fallback (the closure investigation ended without its criterion): in the
    full ``placed_orientation`` gate an object whose model has a continuous symmetry about its long axis in the ICL
    symmetry survey (``assets/icl/revolution_axes.json``: marker, pencil, bottles) is compared modulo rotation about that
    axis; a marker rolled 92 deg about its own axis during finger closure (0916-152503: 0.386 rad full error) passes,
    the same marker standing on its end fails, every untagged object keeps the full error, and the equivalence can be
    switched off."""
    from ispatialgym.oracle import executor as EXE

    goal_q = th.tensor([0.0, 0.0, math.sin(0.25), math.cos(0.25)])  # the recorded goal: yaw 0.5 rad, the long axis (x) level
    table = {"marker/xakoic": {"axis": "x", "tags": ["xinf"], "sources": ["symmetry_near", "geom_near"]}}
    monkeypatch.setattr(EXE, "revolution_axes", lambda path=EXE.REVOLUTION_AXES_PATH: table)
    gate = ExecutorConfig(settle_min_steps=50, settle_fraction=0.2, demo_gating=("placed_orientation",))

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
            d["errors"]["pose_errors"]["apple.n.01_1"].update({"orientation_error": err_rad, "gating": False, "within_tolerance": err_rad <= 0.2})
            return d

        env.evaluate = evaluate
        if quat_after is not None:

            def after(step_action):
                if env.robot._ag_obj_in_hand["right"] is None and float(th.linalg.norm(obj.pos - th.tensor(goal))) < 0.08:
                    obj.quat = quat_after

            env.after_step = after
        return sc, env

    rolled = C.qmul(goal_q, th.tensor([math.sin(0.8), 0.0, 0.0, math.cos(0.8)]))  # the goal yaw, then 1.6 rad (92 deg) about the marker's own x
    standing = C.qmul(goal_q, th.tensor([0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)]))  # the long axis pitched up 90 deg
    # (a) the rolled marker: the evaluator's full error 1.6 rad fails the strict reading, the gate compares modulo the roll -> success
    sc, env = world(1.6, quat_after=rolled)
    ex, *_ = make_executor(env, sc, tmp_path / "a", exec_cfg=gate)
    result = ex.run_program(expand(sc.alias, sc))
    rep = result["demo_gating"]["placed_orientation"]["apple.n.01_1"]
    assert result["success"], (result["verdict"], rep)
    assert rep["within"] is True and rep["orientation_error"] < 0.02 and rep["full_error"] == 1.6 and rep["tolerance"] == 0.2
    assert rep["revolution_axis"] == "x" and rep["model"] == "marker/xakoic" and "rotation about the x axis" in rep["equivalence"]
    assert result["demo_gating"]["enabled"] == ["placed_orientation"] and "placed_tilt" in result["demo_gating"]  # the tilt reading is still reported alongside
    # (b) the same marker standing on its end: the axis error is 90 deg -> placed_orientation
    sc, env = world(1.6, quat_after=standing)
    ex, *_ = make_executor(env, sc, tmp_path / "b", exec_cfg=gate)
    result = ex.run_program(expand(sc.alias, sc))
    rep = result["demo_gating"]["placed_orientation"]["apple.n.01_1"]
    assert result["reason"] == "placed_orientation" and abs(rep["orientation_error"] - math.pi / 2) < 0.01 and rep["full_error"] == 1.6
    # (c) a target equivalence (a half turn about the marker's y) makes the upside-down marker pass the axis reading too
    sc, env = world(1.6, quat_after=C.qmul(goal_q, th.tensor([0.0, 1.0, 0.0, 0.0])))
    env.task._tro.meta.pose_targets["apple.n.01_1"].orientation_equivalences = [th.tensor([0.0, 1.0, 0.0, 0.0])]
    ex, *_ = make_executor(env, sc, tmp_path / "c", exec_cfg=gate)
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"] and result["demo_gating"]["placed_orientation"]["apple.n.01_1"]["orientation_error"] < 0.02
    # (d) an untagged model keeps the full error: the same roll fails
    sc, env = world(1.6, category="apple", model="omzprq", quat_after=rolled)
    ex, *_ = make_executor(env, sc, tmp_path / "d", exec_cfg=gate)
    result = ex.run_program(expand(sc.alias, sc))
    rep = result["demo_gating"]["placed_orientation"]["apple.n.01_1"]
    assert result["reason"] == "placed_orientation" and rep["orientation_error"] == 1.6 and "full_error" not in rep and "equivalence" not in rep
    # (e) the equivalence switched off: the marker's roll fails with the full error
    sc, env = world(1.6, quat_after=rolled)
    ex, *_ = make_executor(env, sc, tmp_path / "e", exec_cfg=ExecutorConfig(settle_min_steps=50, settle_fraction=0.2, demo_gating=("placed_orientation",), demo_revolution_axes=False))
    result = ex.run_program(expand(sc.alias, sc))
    assert result["reason"] == "placed_orientation" and result["demo_gating"]["placed_orientation"]["apple.n.01_1"]["orientation_error"] == 1.6
    # (f) the category / model fall back to the scene object's name (<category>_<model>_<i>) when the attributes are missing
    sc, env = world(1.6, quat_after=rolled)
    obj = env.objects["apple.n.01_1"]
    del obj.category
    obj.name = "marker_xakoic_0"
    ex, *_ = make_executor(env, sc, tmp_path / "f", exec_cfg=gate)
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"] and result["demo_gating"]["placed_orientation"]["apple.n.01_1"]["revolution_axis"] == "x"


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


def test_epilogue_motion_rule_ignores_rotation_without_an_orientation_target(tmp_path):
    """Owner rule 2026-09-16: in the epilogue motion window an object's rotation counts only when the goal has an
    orientation target for it; a spinning near-spherical object with a stable position passes; a second, 2x
    longer window is the second chance before a not_settled verdict."""
    sc, env = iors_world()
    goal = _place_goal(sc)
    env.evaluate = success_eval("apple.n.01_1", goal, tol=0.08)  # position-only target: orientation_error absent
    apple = env.objects["apple.n.01_1"]

    def spin(step_action):
        if env.robot._ag_obj_in_hand["right"] is None and float(th.linalg.norm(apple.pos - th.tensor(goal))) < 0.08:
            yaw = 0.05 * env._current_step
            apple.quat = th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])

    env.after_step = spin
    ex, *_ = make_executor(env, sc, tmp_path)
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"], result["verdict"]
    assert result["motion_delta"]["apple.n.01_1_rad"] > 0.1 and result["motion_delta"]["apple.n.01_1_m"] < 1e-6
    assert result["settle_windows"] == [result["settle_window"]]
    # with an orientation target on the apple the same spin is a not_settled verdict, after the second window
    sc, env = iors_world()

    def eval_oriented(env_):
        d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
        d["errors"]["pose_errors"]["apple.n.01_1"]["orientation_error"] = 0.0
        return d

    env.evaluate = eval_oriented
    apple = env.objects["apple.n.01_1"]
    env.after_step = spin
    ex, *_ = make_executor(env, sc, tmp_path / "b")
    result = ex.run_program(expand(sc.alias, sc))
    assert result["reason"] == "not_settled" and len(result["settle_windows"]) == 2 and result["settle_windows"][1] == 2 * result["settle_windows"][0]
    assert "first_window" in result["motion_delta"]
    # a transient (the object stops during the second window) is a success
    sc, env = iors_world()
    env.evaluate = eval_oriented
    apple = env.objects["apple.n.01_1"]
    stop_at = {"step": None}

    def spin_then_stop(step_action):
        if env.robot._ag_obj_in_hand["right"] is None and float(th.linalg.norm(apple.pos - th.tensor(goal))) < 0.08:
            if stop_at["step"] is None:
                stop_at["step"] = env._current_step + 120  # ~55 steps into the first no-op window
            if env._current_step < stop_at["step"]:
                yaw = 0.05 * env._current_step
                apple.quat = th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])

    env.after_step = spin_then_stop
    ex, *_ = make_executor(env, sc, tmp_path / "c")
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"] and len(result["settle_windows"]) == 2


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
    result = ex.run_program(expand(sc.alias, sc))
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
    result2 = ex2.run_program(expand(sc2.alias, sc2))
    assert result2["success"] and result2["binary_branch"] == {} and result2["binary_branch_fallback"] == [] and result2["binary_branch_log"] == []


def test_release_is_rate_limited_and_staged_until_the_hand_clears():
    """Owner 2026-09-16 21:18 (nextto 0916-201902 / 0916-201730): opening moves ramp at ``gripper_open_rate``; the release
    of a held object is staged - first to the pre-shape width (pinch / 2 + margin, or the executor's capped value), the
    full opening deferred until the hand has moved ``gripper_release_clearance_m`` from the release pose (completed by
    the retreat leg through ``_emit``, ending on the exact open command); a gripper holding nothing opens fully in one
    ramp; a new gripper command supersedes a pending deferred opening; the state round-trips through the checkpoints."""
    robot = FakeRobot()
    env = FakeEnv(robot, {})
    m = Motions(env, FakeBackend(robot), MotionsConfig(gripper_steps=30, settle_steps=5, cartesian_step=0.05, cartesian_rot_step=0.3))
    idx = robot.gripper_control_idx["right"]
    cmds = []
    env.after_step = lambda a: cmds.append(float(a[robot.controller_action_idx["gripper_right"]]))
    # (a) nothing held: closed -> open is one ramp of 2.25 mm per step (22 steps for 50 mm) ending on exactly +1
    drive(env, m.gripper("right", close=True))
    cmds.clear()
    drive(env, m.gripper("right", close=False))
    targets = [0.05 * (c + 1.0) / 2.0 for c in cmds]
    ramp = [t for t in targets if t < 0.05 - 1e-9]
    assert len(ramp) == 22 and all(abs((ramp[i + 1] - ramp[i]) - 0.00225) < 1e-6 for i in range(len(ramp) - 1)), ramp
    assert cmds[len(ramp)] == 1.0 and cmds[-1] == 1.0 and not m.gripper_closed("right") and abs(float(robot.q[idx].max()) - 0.05) < 1e-6
    # (b) a held object with the grasp's pre-shape remembered: the release stops at the pre-shape width, the object is
    # still attached (the assisted grasp releases only on the full open command), the deferred opening is pending
    ep, _ = robot.eef_pose("right")
    marker = FakeObject("marker_xakoic_0", ep.tolist(), category="marker", extent=(0.1536, 0.0171, 0.0141))
    env.objects["marker.n.03_1"] = marker
    drive(env, m.gripper_opening("right", 0.5 * 0.01585 + 0.010))
    assert m.preshape_opening["right"] == 0.017925 and not m.gripper_closed("right")
    drive(env, m.gripper("right", close=True, rate=0.00225))
    assert robot._ag_obj_in_hand["right"] is marker and m.gripper_closed("right")
    cmds.clear()
    drive(env, m.gripper("right", close=False))
    stage = 0.017925
    assert abs(float(robot.q[idx].max()) - stage) < 1e-6 and robot._ag_obj_in_hand["right"] is marker
    assert not m.gripper_closed("right") and "right" in m.deferred_open and m.deferred_open["right"]["stage_m"] == stage
    opened = [0.05 * (c + 1.0) / 2.0 for c in cmds]
    assert max(opened) <= stage + 1e-6 and all(opened[i + 1] - opened[i] <= 0.00225 + 1e-6 for i in range(len(opened) - 1))
    assert m.release_events[-1]["event"] == "staged" and m.release_events[-1]["source"] == "preshape" and m.release_events[-1]["stage_m"] == 0.0179
    # the explicit release (execute_place) detaches; holding the hand still keeps the fingers at the stage (no clearance)
    robot._ag_obj_in_hand["right"] = None
    m.backend.detach_object("right")
    drive(env, m.hold(40))
    assert abs(float(robot.q[idx].max()) - stage) < 1e-6 and "right" in m.deferred_open and m.deferred_open["right"]["steps"] >= 40  # + the release's own wait steps
    # (c) the retreat leg lifts the hand: past 3 cm the deferred opening ramps to fully open at 2.25 mm/step, exactly +1 at the end
    cmds.clear()
    p0, q0 = m.eef_pose("right")
    drive(env, m.arm_along("right", [(p0 + th.tensor([0.0, 0.0, 0.10]), q0)]))
    assert "right" not in m.deferred_open and cmds[-1] == 1.0 and abs(float(robot.q[idx].max()) - 0.05) < 1e-6
    ev = m.release_events[-1]
    assert ev["event"] == "full_open" and ev["by"] == "clearance" and 0.03 <= ev["started_clearance_m"] < 0.04, ev
    opened = [0.05 * (c + 1.0) / 2.0 for c in cmds]
    first_move = next(i for i, t in enumerate(opened) if t > stage + 1e-9)
    assert first_move > 0 and all(opened[i + 1] - opened[i] <= 0.00225 + 1e-6 for i in range(first_move, len(opened) - 1))
    assert "right" not in m.preshape_opening
    # (d) the executor's sidecar-derived first stage wins over the pre-shape; a closing command supersedes a pending deferral
    robot.ag_fail_next = 0
    drive(env, m.gripper_opening("right", 0.030))
    drive(env, m.gripper("right", close=True, rate=0.00225))
    assert robot._ag_obj_in_hand["right"] is marker
    m.release_opening["right"] = 0.021
    drive(env, m.gripper("right", close=False))
    assert abs(float(robot.q[idx].max()) - 0.021) < 1e-6 and m.release_events[-1]["source"] == "release_opening"
    st = m.export_state()
    assert st["deferred_open"]["right"]["stage_m"] == 0.021 and st["gripper_intent_closed"]["right"] is False
    drive(env, m.gripper("right", close=True))
    assert "right" not in m.deferred_open and m.gripper_closed("right") and float(robot.q[idx].max()) <= 0.021 + 1e-6
    m.import_state(st)
    assert "right" in m.deferred_open and m.deferred_open["right"]["stage_m"] == 0.021 and not m.gripper_closed("right")
    # (e) the deferred opening completes at the latest after gripper_release_deferred_max_steps without any clearance
    m.cfg.gripper_release_deferred_max_steps = 25
    m.deferred_open["right"]["steps"] = 0
    drive(env, m.hold(60))
    assert "right" not in m.deferred_open and m.release_events[-1]["by"] == "max_steps" and abs(float(robot.q[idx].max()) - 0.05) < 1e-6
    # (f) a pre-shape narrower than the current fingers stages nothing (the pads already clear the object)
    drive(env, m.gripper("right", close=False))
    env.after_step = None


def test_place_releases_in_two_stages_and_the_retreat_completes_the_opening(tmp_path):
    """The executor derives the first stage from the sidecar (library pinch 73.4 mm / 2 + 10 mm = 46.7 mm; capped by the
    sampler's ``release_sweep.release_opening_max_m`` / 2 when recorded), ``execute_place`` needs no edit: its
    ``gripper(open)`` stops at the stage and its retreat leg (10 cm along -a) completes the full opening once the hand
    is 3 cm away; both events are in the place's notes; the object is released (the explicit release) and the demo passes."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    robot = env.robot
    cmds = []
    env.after_step = lambda a: cmds.append(float(a[robot.controller_action_idx["gripper_right"]]))
    ex, motions, *_ = make_executor(env, sc, tmp_path)
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"], result["verdict"]
    place = result["actions"][3]
    assert place["name"] == "place" and abs(place["notes"]["release_stage_m"] - (0.5 * 0.0734 + 0.010)) < 1e-6
    events = place["notes"]["release"]
    assert [e["event"] for e in events] == ["staged", "full_open"], events
    assert events[0]["source"] == "release_opening" and abs(events[0]["stage_m"] - round(0.5 * 0.0734 + 0.010, 4)) < 1e-6
    assert events[1]["by"] == "clearance" and events[1]["started_clearance_m"] >= 0.03 and events[1]["clearance_m"] <= 0.10 + 1e-6
    assert robot._ag_obj_in_hand["right"] is None and float(robot.q[robot.gripper_control_idx["right"]].max()) > 0.049
    # the opening never jumps: every command step opens the fingers by at most 2.25 mm (from the closure's contact hold on)
    i0 = min(range(len(cmds)), key=lambda i: cmds[i])
    assert -1.0 < cmds[i0] < 0.9  # the symmetric closure holds at the apple's contact width, never at -1
    steps = [0.05 * (b - a) / 2.0 for a, b in zip(cmds[i0:-1], cmds[i0 + 1:]) if b > a]
    assert steps and max(steps) <= 0.00225 + 1e-6, max(steps)
    # the sampler's release sweep caps the first stage (core 385de1e: the largest symmetric opening clearing the partner)
    sc2, env2 = iors_world()
    env2.evaluate = success_eval("apple.n.01_1", _place_goal(sc2), tol=0.08)
    sc2.grasps[0].extra["release_sweep"] = {"release_opening_max_m": 0.086, "pinch_width_m": 0.0734, "clearance_at_place_m": -0.004}
    ex2, motions2, *_ = make_executor(env2, sc2, tmp_path / "b")
    result2 = ex2.run_program(expand(sc2.alias, sc2))
    assert result2["success"], result2["verdict"]
    ev2 = result2["actions"][3]["notes"]["release"]
    assert abs(result2["actions"][3]["notes"]["release_stage_m"] - 0.043) < 1e-6 and ev2[0]["stage_m"] == 0.043 and ev2[1]["event"] == "full_open"
    # a cap below pinch / 2 + 5 mm is not honoured (never less than 5 mm of clearance per pad)
    assert abs(ex2.release_stage_opening(SimpleNamespace(object="apple.n.01_1", arm="right")) - 0.043) < 1e-6
    sc2.grasps[0].extra["release_sweep"]["release_opening_max_m"] = 0.075
    assert abs(ex2.release_stage_opening(SimpleNamespace(object="apple.n.01_1", arm="right")) - (0.5 * 0.0734 + 0.005)) < 1e-6
    assert ex2.release_stage_opening(SimpleNamespace(object="plate.n.04_1", arm="right")) is None


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
    result = ex.run_program(expand(sc.alias, sc))
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
    assert EX.grasp_pinch_width(expand(sc.alias, sc)[1]) == 0.0734
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
    result2 = ex2.run_program(expand(sc2.alias, sc2))
    assert result2["success"] and "preshape_opening_m" not in result2["actions"][1]["notes"]
    i_close = next(i for i, c in enumerate(cmds) if c < 1.0 - 1e-6)  # the first command below fully open = the closure's first ramp step
    before_close = cmds[:i_close]  # the descent runs fully open (the staged release later may use the pre-shape width)
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
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == ["transition", "grasp", "lift", "place", "retreat"]
    result = ex.run_program(actions)
    assert result["success"], result["verdict"]
    grasp = result["actions"][1]
    assert grasp["name"] == "grasp" and len(grasp["attempts"]) == 2
    assert grasp["attempts"][0]["category"] == "failed_grasp" and grasp["attempts"][1] is None
    assert result["qa"]["stats"]["by_category"] == {"failed_grasp": 1}
    assert result["qa"]["stats"]["by_action"] == {"grasp": {"failed_grasp": 1}}
    # the checkpoint restored the simulator and the executor state; OmniGibson's own AG round-trip was fine here
    assert len(result["checkpoint_reports"]) == 1 and result["checkpoint_reports"][0]["ag_roundtrip_ok"] is True
    # planner attachment follows the grasp / release
    assert backend._attached == {}
    assert env.robot._ag_obj_in_hand["right"] is None
    # epilogue: the no-op window is max(50, 20 % of the action steps)
    assert result["settle_window"] == max(50, math.ceil(0.2 * result["n_action_steps"]))
    assert result["n_settle_steps"] == result["settle_window"]
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
    actions = expand(sc.alias, sc)
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
    assert calls[0] == "plan_joint" and "plan_pose" not in calls and calls.count("solve_ik_path") == 2  # residual + approach
    assert env.robot._ag_obj_in_hand["right"] is env.objects["apple.n.01_1"]
    # without a witness the pose goal is planned as before
    sc.extra["ik_witnesses"] = []
    sc2, env2 = iors_world()
    ex2, motions2, backend2, _ = make_executor(env2, sc2, tmp_path / "b")
    for i in range(2):
        assert ex2.run_action(i, expand(sc2.alias, sc2)[i]) is None
    assert "plan_pose" in backend2.calls


def test_execute_functions_toggle_the_fov_monitor_around_the_checked_segment(tmp_path):
    """FOV rule (b): the grasp opens the acting arm's span once the pre-grasp is reached; nothing touches it through the
    lift and the carry; the place closes it after the release (the executor wires ``ctx["hooks"]["fov"]`` to
    ``Monitors.set_fov_enabled``). The carry leg is a straight Cartesian leg (``targets["carry"]``), not a trajopt."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    calls = []
    spans = []
    orig = ex.monitors.set_fov_enabled

    def spy(enabled=True):
        calls.append((ex.monitors.action_index, bool(enabled)))
        orig(enabled)

    ex.monitors.set_fov_enabled = spy
    actions = expand(sc.alias, sc)
    plans_before_place = None
    for i, a in enumerate(actions[:4]):
        if i == 3:
            plans_before_place = sum(1 for c in backend.calls if c in ("plan_pose", "plan_joint"))
        assert ex.run_action(i, a) is None, a.name
        spans.append(dict(ex.monitors.fov_span))
    by_action = {}
    for idx, en in calls:
        by_action.setdefault(idx, []).append(en)
    assert by_action[1] == [False, True]  # grasp: transit to the pre-grasp unmonitored, the span opens at the pre-grasp
    assert 2 not in by_action  # lift: the span stays open, no hook
    assert by_action[3] == [False]  # place: no hook before the carry / descent, the span closes after the release
    assert spans[1]["right"] and spans[2]["right"] and not spans[3]["right"]  # open after the grasp and the lift, closed after the place
    assert not ex.monitors.fov_enabled  # no span open after the release
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
    transition = expand(sc.alias, sc)[0]
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
    result = ex.run_program(expand(sc.alias, sc))
    assert not result["success"] and result["reason"] == "failed_grasp"
    assert result["verdict"]["action"] == "grasp" and result["verdict"]["attempt"] == 1
    assert [a["name"] for a in result["actions"]] == ["transition", "grasp"]
    assert len(result["actions"][1]["attempts"]) == 2
    assert result["qa"]["stats"]["by_action"]["grasp"]["failed_grasp"] == 2
    # the epilogue still ran its no-op window; the failed attempt stays in the record for diagnosis
    assert result["n_settle_steps"] >= 50
    assert len(rec.rollbacks) == 1 and rec.actions[1].verdict["category"] == "failed_grasp"


def test_plan_failed_is_retried_and_counted_separately(tmp_path):
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    backend.fail_plans = 1  # the first trajopt of the program (grasp pre-grasp) fails once
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"], result["verdict"]
    assert result["qa"]["stats"]["by_category"] == {"plan_failed": 1}
    assert result["actions"][1]["attempts"][0]["details"]["stage"] == "trajopt"


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
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"], result["verdict"]
    assert result["actions"][0]["attempts"][0]["category"] == "collision"
    assert result["actions"][0]["attempts"][0]["details"]["object"] == "table_0"
    assert result["qa"]["stats"]["by_action"] == {"transition": {"collision": 1}}
    assert result["qa"]["collisions"]["by_object"] == {"table_0": 3}
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
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"], result["verdict"]
    assert result["qa"]["stats"]["by_category"] == {} and seen["manip"] > 0


def inside_world():
    """The IORS inside test sidecar as a fake world: robot at the init stance, sprout on the table, bowl at the drop."""
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-inside.json"))
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
    actions = expand(sc.alias, sc)
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
    actions = expand(sc.alias, sc)
    place = [a for a in actions if isinstance(a, Place)][0]
    assert place.kind == "drop" and place.on == "bowl.n.01_1"
    assert EX.intended_contacts(place) == ("brussels_sprouts.n.01_1",) and EX.forbidden_contacts(place) == ("bowl.n.01_1",)
    support = Place(kind="support", arm="right", object="a.n.01_1", place=place.place, on="b.n.01_1")
    assert EX.intended_contacts(support) == ("a.n.01_1", "b.n.01_1") and EX.forbidden_contacts(support) == ()
    # (a) the opening fingers touch the bowl during the drop -> collision (the finger exemption does not apply)
    sprout, bowl = env.objects["brussels_sprouts.n.01_1"], env.objects["bowl.n.01_1"]
    env.evaluate = success_eval("brussels_sprouts.n.01_1", [float(v) for v in place.place.position], tol=0.15)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, qa_cfg=QA.QAConfig(pause_steps=100000, fov_min_steps=100000, collision_min_steps=2))

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
    ex, motions, backend, rec = make_executor(env, sc, tmp_path / "b", qa_cfg=QA.QAConfig(pause_steps=100000, fov_min_steps=100000, collision_min_steps=2))

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
    actions = expand(sc.alias, sc)

    class Broken(type(actions[1])):
        def targets(self, sidecar, state):
            raise RuntimeError("no such grasp")

    broken = Broken(kind="library", arm="right", object="apple.n.01_1", grasp=sc.grasps[0])
    result = ex.run_program([actions[0], broken])
    assert result["reason"] == "targets_failed"
    assert result["actions"][1]["attempts"][0]["category"] == "targets_failed"


def test_epilogue_ordering_not_reached_tolerance_settled_goal_image(tmp_path):
    # 1) the goal never holds -> not_reached, no goal image
    sc, env = iors_world()
    ex, *_ = make_executor(env, sc, exec_cfg=ExecutorConfig(goal_image=True))
    result = ex.run_program(expand(sc.alias, sc))
    assert result["reason"] == "not_reached" and result["goal_observation"] is None and env.sim.renders == 0
    # 2) the goal holds but a measured error is above tolerance -> tolerance
    sc, env = iors_world()
    goal = _place_goal(sc)

    def eval_tol(env_):
        d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
        d["errors"]["pose_errors"]["plate.n.04_1"] = {"position_error": 0.5, "within_tolerance": False}
        return d

    env.evaluate = eval_tol
    ex, *_ = make_executor(env, sc, exec_cfg=ExecutorConfig(goal_image=True))
    result = ex.run_program(expand(sc.alias, sc))
    assert result["reason"] == "tolerance" and result["verdict"]["details"]["targets"] == ["plate.n.04_1"]
    # 3) the goal holds, nothing moves over the window, but the evaluator's pose_settled never fires: a success by
    #    default (the §6.4 "motion after the window" is measured by the executor), not_settled when the evaluator's
    #    verdict is required
    sc, env = iors_world()

    def eval_unsettled(env_):
        d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
        d["settled"] = False
        return d

    env.evaluate = eval_unsettled
    ex, *_ = make_executor(env, sc, exec_cfg=ExecutorConfig(goal_image=False))
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"] and result["settled_success"] is False
    sc, env = iors_world()
    env.evaluate = eval_unsettled
    ex, *_ = make_executor(env, sc, exec_cfg=ExecutorConfig(goal_image=True, require_task_settled=True))
    result = ex.run_program(expand(sc.alias, sc))
    assert result["reason"] == "not_settled" and result["verdict"]["details"]["task_settled_success"] is False
    # 3b) the goal held once but not at the last step -> not_settled
    sc, env = iors_world()
    hits = {"n": 0}

    def eval_transient(env_):
        d = success_eval("apple.n.01_1", goal, tol=0.08)(env_)
        if d["binary_success"]:
            hits["n"] += 1
            if hits["n"] > 3:
                d["binary_success"] = False
                d["binary_success_ever"] = True
        return d

    env.evaluate = eval_transient
    ex, *_ = make_executor(env, sc)
    result = ex.run_program(expand(sc.alias, sc))
    assert result["reason"] == "not_settled" and "no longer holds" in result["verdict"]["message"]
    # 4) success: the goal image is captured after >= 3 renders with the physics-synced camera pose
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", goal, tol=0.08)
    ex, *_ = make_executor(env, sc, exec_cfg=ExecutorConfig(goal_image=True, goal_image_renders=10))
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"] and env.sim.renders == 10
    go = result["goal_observation"]
    assert go["rgb"].shape == (8, 8, 3) and go["sensor_key"] == CAM_KEY and go["n_renders"] == 10
    cam_p, _ = env.robot.sensors[CAM_KEY].get_position_orientation()
    assert go["camera_pose_world"]["position"] == pytest.approx(cam_p.tolist(), abs=1e-6)
    assert go["intrinsics"]["fx"] == pytest.approx(459.0, abs=1.0)


def test_replay_stream_holds_the_last_streamed_configuration(tmp_path):
    """Replay / policy mode: the epilogue's no-op window must repeat the last streamed action (base stopped), not the
    Layer-1 hold targets captured at construction (which would drive the robot back to its initial pose)."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    # record the oracle program's actions, then replay them through a fresh executor on a fresh world
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"]
    actions = [np.asarray(a) for a in rec.rows["action"][: result["n_action_steps"]]]
    sc2, env2 = iors_world()
    env2.evaluate = success_eval("apple.n.01_1", _place_goal(sc2), tol=0.08)
    ex2, motions2, *_ = make_executor(env2, sc2, tmp_path / "replay")
    q_init = env2.robot.q.clone()
    result2 = ex2.run_stream(th.as_tensor(a) for a in actions)
    assert result2["success"], result2["verdict"]
    assert result2["n_action_steps"] == len(actions) and result2["n_settle_steps"] == max(50, math.ceil(0.2 * len(actions)))
    # the arm stayed where the stream left it (the initial hold targets were not re-applied)
    last = th.as_tensor(actions[-1])
    idx = env2.robot.controller_action_idx
    assert th.allclose(env2.robot.q[env2.robot.arm_control_idx["right"]], last[idx["arm_right"]], atol=1e-5)
    assert not th.allclose(env2.robot.q[env2.robot.arm_control_idx["right"]], q_init[env2.robot.arm_control_idx["right"]], atol=1e-3)
    assert th.equal(env2.last_action[idx["base"]], th.zeros(3))
    assert result2["qa"]["stats"]["by_category"] == {}


def test_checkpoint_restore_fixes_a_broken_ag_roundtrip(tmp_path):
    sc, env = iors_world(ag_roundtrip=False)
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    actions = expand(sc.alias, sc)
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
        assert ex2.run_action(i, expand(sc2.alias, sc2)[i]) is None
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


def test_fov_pause_and_ghost_hand_monitors():
    robot = FakeRobot()
    env = FakeEnv(robot, {})
    mon = QA.Monitors(env, robot, cfg=QA.QAConfig(fov_min_steps=3, pause_steps=4), camera=QA.CameraModel.from_hfov())
    mon._contact_api = FakeContacts(lambda s, q: [])
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-ontop_small_scale.json"))
    actions = expand(sc.alias, sc)
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
    # in frame: the eef in front of the robot -> the pause monitor fires after 4 stationary steps
    robot.q[robot.arm_control_idx["right"]] = in_q
    robot.dq = th.zeros(28)
    mon.begin_action(grasp, 1, 1, intended=set(), held=[], manipulation=True)
    verdicts = [mon.check(i) for i in range(1, 5)]
    assert verdicts[:3] == [None, None, None] and verdicts[3].category == "paused"
    # ghost hand: the held object drifts off the hand
    apple = FakeObject("apple_0", robot.eef_pose("right")[0].tolist())
    robot._ag_obj_in_hand["right"] = apple
    mon.begin_epilogue(held=[apple])
    assert mon.check(10) is None
    apple.pos = apple.pos + th.tensor([0.2, 0.0, 0.0])
    v = mon.check(11)
    assert v is not None and v.category == "ghost_hand" and v.details["object"] == "apple_0"
    stats = mon.stats.to_dict()
    assert stats["by_category"] == {"gripper_out_of_fov": 3, "paused": 1, "ghost_hand": 1}  # (b), (b') and the lift of (d)
    assert stats["by_action"]["grasp"] == {"gripper_out_of_fov": 2, "paused": 1} and stats["by_action"]["lift"] == {"gripper_out_of_fov": 1}


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
    result = ex.run_program(expand(sc.alias, sc))
    assert result["success"]
    paths = rec.close({"instance_id": sc.instance_id, "success": True}, qa=result["qa"]["stats"], success=True)
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
        for key in ("step", "phase", "action_index", "attempt", "joint_positions", "joint_velocities", "base_pose", "eef_pose_left", "eef_pose_right", "head_camera_pose", "proprio", "eval/binary_success", "eval/max_position_error"):
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
    rec = Recorder(str(tmp_path), env, env.robot, CAM_KEY, dict(env.objects), cfg=RecorderConfig(hdf5=False, lerobot=False, video=False))
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


def test_goal_override_thresholds_measured_only_vs_gating():
    from ispatialgym.eval.targets import PoseTarget

    sc, env = iors_world()
    ex, *_ = make_executor(env, sc)
    apple = env.objects["apple.n.01_1"]
    tgt = PoseTarget.from_dict("apple.n.01_1", {"position": (apple.pos + th.tensor([0.005, 0.0, 0.0])).tolist(), "orientation": [0, 0, 0, 1],
                                                 "position_tolerance": 0.03, "orientation_tolerance": 0.2})

    class Compiled:
        def is_gating(self, pred, inst):
            return False

    # measured only: threshold max(0.01, 0.25 * 0.08 m extent) = 0.02 m -> a 5 mm settled offset is applied
    rec = ex._override_pose_target("apple.n.01_1", tgt, Compiled())
    assert rec["applied"] and rec["position_tolerance"] == pytest.approx(0.02) and rec["position_error"] == pytest.approx(0.005, abs=1e-6)
    apple.pos = apple.pos + th.tensor([0.05, 0.0, 0.0])
    assert ex._override_pose_target("apple.n.01_1", tgt, Compiled())["applied"] is False

    class Gating(Compiled):
        def is_gating(self, pred, inst):
            return True

    # gating: the target's own tolerance (3 cm) applies -> 5.5 cm still rejected, 2.5 cm accepted
    assert ex._override_pose_target("apple.n.01_1", tgt, Gating())["applied"] is False
    apple.pos = apple.pos - th.tensor([0.03, 0.0, 0.0])
    rec = ex._override_pose_target("apple.n.01_1", tgt, Gating())
    assert rec["applied"] and rec["position_tolerance"] == pytest.approx(0.03)
    depth = PoseTarget.from_dict("apple.n.01_1", {"position": [0, 0, -0.5], "position_metric": "depth", "frame": f"{AGENT_INSTANCE}::{CAM_KEY}", "position_tolerance": 0.02})
    assert ex._override_pose_target("apple.n.01_1", depth, Gating())["applied"] is False
