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
import math
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
from ispatialgym.oracle.actions import VOCABULARY, Navigate, Push, BaseFollow
from ispatialgym.oracle.executor import AGENT_INSTANCE, Executor, ExecutorConfig, apply_goal_override
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
        for wp in waypoints:
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

    def _obs(self):
        return {self.robot.name: {CAM_KEY: {"rgb": th.zeros(8, 8, 4, dtype=th.uint8)}, "proprio": self.robot.q.clone()}}

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
                        if float(th.linalg.norm(o.pos - ep)) < 0.15:
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
    for fn in sorted(os.listdir(SIDECARS)):
        sc = Sidecar.load(os.path.join(SIDECARS, fn))
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
    grasp = expand(sc.alias, sc)[1]
    # the right eef far behind the robot: out of the head camera's frame
    robot.q[robot.arm_control_idx["right"]] = th.tensor([-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    robot.dq = th.ones(28)
    mon.begin_action(grasp, 1, 0, intended={"apple_omzprq_0"}, held=[], manipulation=True)
    verdicts = [mon.check(i) for i in range(1, 4)]
    assert verdicts[:2] == [None, None] and verdicts[2].category == "gripper_out_of_fov" and verdicts[2].details["arm"] == "right"
    # in frame: the eef in front of the robot -> the pause monitor fires after 4 stationary steps
    robot.q[robot.arm_control_idx["right"]] = th.tensor([0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0])
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
    assert stats["by_category"] == {"gripper_out_of_fov": 1, "paused": 1, "ghost_hand": 1}
    assert stats["by_action"]["grasp"] == {"gripper_out_of_fov": 1, "paused": 1}


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
