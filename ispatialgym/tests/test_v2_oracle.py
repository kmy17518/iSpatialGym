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
            fingers = r.q[r.gripper_control_idx[a]]
            target = 0.0 if cmd < 0 else 0.05
            r.q[r.gripper_control_idx[a]] = fingers + th.clamp(target - fingers, -0.01, 0.01)
            # assisted grasp: closing near an object attaches it; opening releases it. ``ag_fail_next`` refuses whole
            # closures (open -> close transitions), like an assisted grasp that never sees both pads on the object.
            if cmd < 0:
                if not self._closing[a]:
                    self._closing[a] = True
                    self._refuse[a] = r.ag_fail_next > 0
                    if self._refuse[a]:
                        r.ag_fail_next -= 1
                if r._ag_obj_in_hand[a] is None and not self._refuse[a] and float(r.q[r.gripper_control_idx[a]].max()) < 0.03:
                    ep, _ = r.eef_pose(a)
                    for o in self.objects.values():
                        if getattr(o, "graspable", True) and float(th.linalg.norm(o.pos - ep)) < 0.15:
                            r._create_assisted_grasp_joint(a, {"target_obj": o, "target_link_name": "base_link", "joint_type": "FixedJoint"})
                            break
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
    paths = sorted(glob.glob(os.path.join(SIDECARS, "*.json")) + glob.glob(os.path.join(SIDECARS, "variants", "*.json")))
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
    """Grasp / Place enable the gripper-in-FOV monitor once the pre-pose is reached and Place disables it again before
    its retreat leg (the executor wires ``ctx["hooks"]["fov"]`` to ``Monitors.set_fov_enabled``)."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path)
    calls = []
    orig = ex.monitors.set_fov_enabled

    def spy(enabled=True):
        calls.append((ex.monitors.action_index, bool(enabled)))
        orig(enabled)

    ex.monitors.set_fov_enabled = spy
    actions = expand(sc.alias, sc)
    for i, a in enumerate(actions[:4]):
        assert ex.run_action(i, a) is None, a.name
    by_action = {}
    for idx, en in calls:
        by_action.setdefault(idx, []).append(en)
    assert by_action[1] == [False, True]  # grasp: transit to the pre-grasp unmonitored, approach on
    assert 2 not in by_action  # lift: monitored from its first step
    assert by_action[3] == [False, True, False]  # place: transit off, descent on, retreat leg off
    assert ex.monitors.fov_enabled  # end_action resets the hook state


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
    assert ctx["idle_arms_moved"] == []
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
    sc = Sidecar.load(os.path.join(SIDECARS, "variants", "inter_object_relation_satisfaction-under_large_scale__push_under.json"))
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
    assert motions.state.gripper_cmd["right"] == -1.0
    # the counters are rolled back, then the one recorded hold step re-applies the restored drive targets
    assert ex.n == cp.step + 1 and env._current_step == cp.env_step + 1
    assert th.equal(env.last_action[env.robot.controller_action_idx["gripper_right"]], th.tensor([-1.0]))
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
    grasp, retreat = actions[1], actions[-1]
    assert EX.fov_monitored(grasp) and not EX.fov_monitored(retreat) and EX.is_manipulation(retreat)
    out_q = th.tensor([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # the right eef far behind the robot: outside the frame
    in_q = th.tensor([0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0])  # in front of the robot: in frame
    robot.dq = th.ones(28)
    # (a) the idle posture is out of frame before the arm moved: no verdict (the sampler does not guarantee it)
    robot.q[robot.arm_control_idx["right"]] = out_q
    mon.begin_action(grasp, 1, 0, intended={"apple_omzprq_0"}, held=[])
    assert [mon.check(i) for i in range(1, 8)] == [None] * 7
    assert mon.stats.events["fov_idle_out"] == 7 and mon.stats.by_category == {}
    # (b) the fingertips enter the frame (the pre-pose), then leave it: armed -> verdict after fov_min_steps
    robot.q[robot.arm_control_idx["right"]] = in_q
    assert mon.check(8) is None and mon._fov_armed["right"]
    robot.q[robot.arm_control_idx["right"]] = out_q
    verdicts = [mon.check(i) for i in range(9, 12)]
    assert verdicts[:2] == [None, None] and verdicts[2].category == "gripper_out_of_fov" and verdicts[2].details["arm"] == "right"
    mon.end_action()
    # (b') the execute hook: disabled while the hand transits to a pre-pose that sits in the margin, enabled from the
    # pre-pose on; disabling disarms (a place's retreat leg after the release)
    mon.begin_action(grasp, 1, 0, intended=set(), held=[])
    mon.set_fov_enabled(False)
    robot.q[robot.arm_control_idx["right"]] = in_q
    mon.check(1)
    robot.q[robot.arm_control_idx["right"]] = out_q
    assert [mon.check(i) for i in range(2, 10)] == [None] * 8 and not mon._fov_armed["right"]
    mon.set_fov_enabled(True)
    assert [mon.check(i) for i in range(10, 14)] == [None] * 4  # still not armed: never in frame since enabling
    robot.q[robot.arm_control_idx["right"]] = in_q
    assert mon.check(14) is None and mon._fov_armed["right"]
    robot.q[robot.arm_control_idx["right"]] = out_q
    assert [mon.check(i) for i in range(15, 17)] == [None, None] and mon.check(17).category == "gripper_out_of_fov"
    mon.begin_action(grasp, 1, 1, intended=set(), held=[])
    robot.q[robot.arm_control_idx["right"]] = in_q
    mon.check(1)
    mon.set_fov_enabled(False)  # e.g. after the release, before the retreat leg
    robot.q[robot.arm_control_idx["right"]] = out_q
    assert [mon.check(i) for i in range(2, 12)] == [None] * 10
    mon.end_action()
    # (c) a retreat is exempt even when the hand leaves the frame
    mon.begin_action(retreat, 4, 0, intended=set(), held=[])
    robot.q[robot.arm_control_idx["right"]] = in_q
    mon.check(20)
    robot.q[robot.arm_control_idx["right"]] = out_q
    assert [mon.check(i) for i in range(21, 30)] == [None] * 9 and not mon.fov_active
    mon.end_action()
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
    assert stats["by_category"] == {"gripper_out_of_fov": 2, "paused": 1, "ghost_hand": 1}  # (b) and (b')
    assert stats["by_action"]["grasp"] == {"gripper_out_of_fov": 2, "paused": 1}


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
