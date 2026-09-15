"""Simulator tests for the iSpatialGym evaluator (headless OmniGibson; run inside the ``ispatialgym`` conda env):

    OMNIGIBSON_HEADLESS=1 python -m pytest ispatialgym/tests/test_sim_evaluator.py -x -q

Covers the ``PoseReached`` metrics (full pose, depth along the head camera axis, orientation-only with symmetry
equivalences), ``LayoutReached`` on an articulated object, group entities (shared assignment, layout vs. absolute
placement), and the two-level task evaluation flow (binary success -> settle latency -> error snapshot) on
``alignment-0`` (the ``test_alignment_*`` tests; ``-k alignment`` selects them).
"""

import json
import math
import os
import tempfile

import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm

gm.HEADLESS = True
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = False

import omnigibson.utils.transform_utils as T  # noqa: E402
from bddl import predicates as bddl_predicates  # noqa: E402
from omnigibson.object_states import LayoutReached, Open, PoseReached, PoseSettled  # noqa: E402
from omnigibson.objects import DatasetObject  # noqa: E402
from omnigibson.robots import Robot  # noqa: E402
from omnigibson.utils.bddl_utils import evaluate_bddl_predicate  # noqa: E402
from omnigibson.utils.config_utils import TorchEncoder  # noqa: E402

import ispatialgym.tasks  # noqa: E402,F401  (registers ISpatialGymTask)
from ispatialgym.eval.episode import isg_info, run_episode  # noqa: E402
from ispatialgym.eval.geometry import relative_pose  # noqa: E402
from ispatialgym.eval.group_entity import GroupEntity  # noqa: E402
from ispatialgym.eval.targets import GroupTarget  # noqa: E402

HEAD_CAMERA = "robot_r1:zed_link:Camera:0"
IDENTITY = th.tensor([0.0, 0.0, 0.0, 1.0])


def _yaw(angle):
    return T.euler2quat(th.tensor([0.0, 0.0, angle]))


# ----------------------------------------------------------------------------------------------------------------
# Object-state tests (one shared env: robot + visual-only props, no gravity effects)
# ----------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def state_env():
    env = og.Environment(configs={"scene": {"type": "Scene"}, "env": {"action_frequency": 30, "physics_frequency": 120}})
    og.sim.stop()
    robot = Robot(name="robot_r1", model="r1pro", obs_modalities=["rgb"], position=[0, 0, 0], orientation=[0, 0, 0, 1])
    env.scene.add_object(robot)
    mug = DatasetObject(name="mug", category="mug", model="ehnmxj", visual_only=True)
    env.scene.add_object(mug)
    cabinet = DatasetObject(name="cabinet", category="bottom_cabinet", model="immwzb")
    env.scene.add_object(cabinet)
    cabinet.set_position_orientation(position=th.tensor([5.0, 0.0, 0.5]))
    dice = []
    for i in range(4):
        d = DatasetObject(name=f"dice{i}", category="toy_dice", model="akguod", visual_only=True)
        env.scene.add_object(d)
        d.set_position_orientation(position=th.tensor([10.0 + 0.2 * i, 0.0, 0.5]))
        dice.append(d)
    og.sim.play()
    for _ in range(3):
        og.sim.step()
    yield {"env": env, "robot": robot, "mug": mug, "cabinet": cabinet, "dice": dice}
    og.clear()


def test_pose_reached_full_depth_and_orientation(state_env):
    robot, mug = state_env["robot"], state_env["mug"]
    assert HEAD_CAMERA in robot.sensors
    state = mug.states[PoseReached]

    # --- full pose in the world frame -------------------------------------------------------------------------
    mug.set_position_orientation(position=th.tensor([0.6, 0.0, 1.2]), orientation=IDENTITY)
    og.sim.step()
    state.set_goal(goal_pos=[0.6, 0.0, 1.2], goal_ori=IDENTITY, pos_tolerance=0.02, ori_tolerance=0.05)
    report = state.get_error_report()
    assert report["position_metric"] == "full" and report["orientation_metric"] == "full"
    assert report["position_error"] < 1e-3 and report["orientation_error"] < 1e-3 and state.get_value()
    state.set_goal(goal_pos=[0.6, 0.0, 1.25], goal_ori=IDENTITY, pos_tolerance=0.02, ori_tolerance=0.05)
    assert not state.get_value() and state.get_error()[0] == pytest.approx(0.05, abs=1e-3)

    # --- depth along the head-camera optical axis (USD camera: -Z forward) ------------------------------------
    cam_pos, cam_quat = PoseReached._lookup_link_pose(robot, HEAD_CAMERA)
    mug_pos, mug_quat = mug.get_position_orientation()
    rel_pos, _ = T.relative_pose_transform(mug_pos, mug_quat, cam_pos, cam_quat)
    assert rel_pos[2] < 0, "an object in front of the camera must have negative z in the USD camera frame"
    forward = T.quat_apply(cam_quat, th.tensor([0.0, 0.0, -1.0]))
    assert forward[0] > 0.9, "the head camera of a robot facing +x must look along +x"
    state.set_goal(goal_pos=rel_pos, pos_axes="depth", frame=f"robot_r1::{HEAD_CAMERA}", pos_tolerance=0.02)
    report = state.get_error_report()
    assert report["position_metric"] == "depth" and report["orientation_error"] is None
    assert report["depth"] == pytest.approx(-rel_pos[2].item(), abs=1e-4) and report["position_error"] < 1e-4
    assert state.get_value()
    # move 10 cm away along the optical axis -> depth error 0.10, lateral moves do not change it
    mug.set_position_orientation(position=mug_pos + 0.10 * forward)
    og.sim.step()
    assert state.get_error()[0] == pytest.approx(0.10, abs=2e-3) and not state.get_value()
    lateral = T.quat_apply(cam_quat, th.tensor([1.0, 0.0, 0.0]))
    mug.set_position_orientation(position=mug_pos + 0.10 * lateral)
    og.sim.step()
    assert state.get_error()[0] == pytest.approx(0.0, abs=2e-3) and state.get_value()
    with pytest.raises(ValueError):
        state.set_goal(goal_pos=rel_pos, pos_axes="depth", frame="world", pos_tolerance=0.02)

    # --- orientation only, with yaw symmetry equivalences -----------------------------------------------------
    mug.set_position_orientation(position=th.tensor([0.6, 0.0, 1.2]), orientation=_yaw(math.pi / 2))
    og.sim.step()
    state.set_goal(goal_ori=IDENTITY, ori_tolerance=0.05)
    report = state.get_error_report()
    assert report["position_error"] is None and report["position_metric"] is None
    assert report["orientation_error"] == pytest.approx(math.pi / 2, abs=1e-3) and not state.get_value()
    state.set_goal(goal_ori=IDENTITY, ori_tolerance=0.05, orientation_equivalences=[_yaw(k * math.pi / 2) for k in range(1, 4)])
    assert state.get_error()[1] == pytest.approx(0.0, abs=1e-3) and state.get_value()
    # a tilt is not covered by the yaw symmetries
    mug.set_position_orientation(orientation=T.euler2quat(th.tensor([0.3, 0.0, math.pi / 2])))
    og.sim.step()
    assert state.get_error()[1] == pytest.approx(0.3, abs=1e-3) and not state.get_value()
    # BDDL dispatch goes through the same state
    assert evaluate_bddl_predicate(bddl_predicates.PoseReached, mug) is False


def test_pose_reached_on_robot_camera_sensor(state_env):
    """Camera relocalization: the agent's pose target evaluates the head-camera sensor pose (link = sensor key)."""
    robot = state_env["robot"]
    state = robot.states[PoseReached]
    for _ in range(2):  # let the render-side camera transform catch up with the physics state
        og.sim.step()
    cam_pos, cam_quat = PoseReached._lookup_link_pose(robot, HEAD_CAMERA)
    render_pos, render_quat = PoseReached._lookup_link_pose(robot, HEAD_CAMERA, use_render_transform=True)
    # The sensor prim pose (physics-synced) and the render camera transform describe the same USD camera frame.
    assert th.allclose(cam_pos, render_pos, atol=1e-3) and T.quat_distance(cam_quat, render_quat)[3].abs() > 0.9999
    base_pos, base_quat = robot.get_position_orientation()
    assert float(th.norm(cam_pos - base_pos)) > 0.5  # the sensor pose is not the base pose
    state.set_goal(goal_pos=cam_pos, goal_ori=cam_quat, link=HEAD_CAMERA, pos_tolerance=0.03, ori_tolerance=0.05)
    assert state.get_value() and state.get_error()[0] < 1e-3
    # drive the base 10 cm forward: the camera moves with it
    robot.set_position_orientation(position=base_pos + th.tensor([0.1, 0.0, 0.0]), orientation=base_quat)
    robot.keep_still()
    og.sim.step()
    pos_err, ori_err = state.get_error()
    assert pos_err == pytest.approx(0.1, abs=0.01) and ori_err < 0.02 and not state.get_value()
    with pytest.raises(ValueError, match="not found"):
        state.set_goal(goal_pos=cam_pos, goal_ori=cam_quat, link="not_a_sensor", pos_tolerance=0.03, ori_tolerance=0.05)
        state.get_error()
    robot.set_position_orientation(position=base_pos, orientation=base_quat)
    robot.keep_still()
    og.sim.step()


def test_layout_reached_articulated_object(state_env):
    cabinet = state_env["cabinet"]
    state = cabinet.states[LayoutReached]
    joints = cabinet.joints
    assert "j_link_0" in joints and joints["j_link_0"].is_revolute
    closed = {name: {"position": float(j.get_state()[0].item()), "tolerance": 0.05} for name, j in joints.items()}
    state.set_goal(closed)
    assert state.get_value() and evaluate_bddl_predicate(bddl_predicates.LayoutReached, cabinet)
    report = state.get_error_report()
    assert report["orientation_error"] == pytest.approx(0.0, abs=1e-6)  # revolute joints -> orientation error
    assert report["position_error"] == pytest.approx(0.0, abs=1e-6)  # prismatic drawer -> position error
    assert not cabinet.states[Open].get_value()

    # open the door fully: layout no longer reached, error = opening angle, Open becomes True
    positions = cabinet.get_joint_positions()
    idx = list(joints).index("j_link_0")
    positions[idx] = joints["j_link_0"].upper_limit
    cabinet.set_joint_positions(positions)
    og.sim.step()
    assert not state.get_value()
    err = state.get_error()["j_link_0"]
    assert err["error"] == pytest.approx(float(joints["j_link_0"].upper_limit) - closed["j_link_0"]["position"], abs=1e-3)
    assert cabinet.states[Open].get_value()
    # a target at the open angle is reached again
    open_targets = dict(closed)
    open_targets["j_link_0"] = {"position": float(joints["j_link_0"].upper_limit), "tolerance": 0.05}
    state.set_goal(open_targets)
    assert state.get_value()

    # strictness: unknown joints and missing movable joints are rejected
    with pytest.raises(ValueError, match="not found"):
        state.set_goal({"nope": {"position": 0.0, "tolerance": 0.1}})
    with pytest.raises(ValueError, match="every non-fixed joint"):
        state.set_goal({"j_link_0": {"position": 0.0, "tolerance": 0.1}})
    state.set_goal({"j_link_0": {"position": 0.0, "tolerance": 0.1}}, require_all_movable=False)


def _slot_layout():
    slots = {f"slot_{i}": {"position": [10.0 + 0.2 * i, 0.0, 0.5], "orientation": [0, 0, 0, 1]} for i in range(4)}
    relations = []
    for i in range(4):
        for j in range(i + 1, 4):
            p, q = relative_pose(
                (th.tensor(slots[f"slot_{i}"]["position"]), IDENTITY), (th.tensor(slots[f"slot_{j}"]["position"]), IDENTITY)
            )
            relations.append({"from": f"slot_{i}", "to": f"slot_{j}", "position": p.tolist(), "orientation": q.tolist()})
    return slots, relations


def test_group_entity_shared_assignment(state_env):
    dice = state_env["dice"]
    names = [f"toy_dice.n.01_{i + 1}" for i in range(4)]
    scope = dict(zip(names, dice))
    group = GroupEntity("blocks_group.n.01_1", names, lambda n: scope.get(n))
    slots, relations = _slot_layout()
    target = GroupTarget.from_dict(
        "blocks_group.n.01_1",
        {
            "members": names,
            "assignment": {"mode": "exchangeable"},
            "target_slots": slots,
            "pose_tolerances": {"position": 0.02, "orientation": 0.1},
            "layout_relations": relations,
            "layout_tolerances": {"position": 0.02, "orientation": 0.1},
        },
    )
    assert target.assignment_cost == "layout"
    group.set_target(target)
    slot_list = list(slots.values())

    def place(perm, offset=(0.0, 0.0, 0.0), yaw=0.0):
        for k, d in enumerate(dice):
            s = slot_list[perm[k]]
            d.set_position_orientation(position=th.tensor(s["position"]) + th.tensor(offset), orientation=_yaw(yaw))
        og.sim.step()
        group.invalidate()

    # at the slots: both predicates hold, identity assignment
    place([0, 1, 2, 3])
    ev = group.evaluate()
    assert ev.pose_reached and ev.layout_reached
    assert ev.assignment == {names[i]: f"slot_{i}" for i in range(4)}
    assert evaluate_bddl_predicate(bddl_predicates.PoseReached, group) and evaluate_bddl_predicate(bddl_predicates.LayoutReached, group)

    # whole arrangement displaced: layout still holds, absolute placement does not
    place([0, 1, 2, 3], offset=(0.5, 0.3, 0.0))
    ev = group.evaluate()
    assert ev.layout_reached and not ev.pose_reached
    assert ev.max_pose_position_error == pytest.approx(math.hypot(0.5, 0.3), abs=1e-3)
    assert ev.max_layout_position_error < 1e-3

    # members permuted: the shared assignment follows the permutation
    place([1, 0, 3, 2])
    ev = group.evaluate()
    assert ev.pose_reached and ev.layout_reached
    assert ev.assignment == {names[0]: "slot_1", names[1]: "slot_0", names[2]: "slot_3", names[3]: "slot_2"}

    # one block rotated: layout broken through its relations, error reported per relation
    place([0, 1, 2, 3])
    dice[2].set_position_orientation(orientation=_yaw(0.5))
    og.sim.step()
    group.invalidate()
    ev = group.evaluate()
    assert not ev.layout_reached and not ev.pose_reached
    assert ev.max_layout_orientation_error == pytest.approx(0.5, abs=1e-3)
    bad = [r for r in ev.relation_errors if not r["within_tolerance"]]
    assert bad and all(names[2] in (r["from_member"], r["to_member"]) for r in bad)

    # ... unless the blocks are yaw-symmetric
    target_sym = GroupTarget.from_dict(
        "blocks_group.n.01_1",
        {
            "members": names,
            "orientation_equivalences": [_yaw(k * math.pi / 2).tolist() for k in range(1, 4)],
            "target_slots": slots,
            "pose_tolerances": {"position": 0.02, "orientation": 0.1},
            "layout_relations": relations,
            "layout_tolerances": {"position": 0.02, "orientation": 0.1},
        },
    )
    dice[2].set_position_orientation(orientation=_yaw(math.pi / 2))
    og.sim.step()
    group.set_target(target_sym)
    ev = group.evaluate()
    assert ev.layout_reached and ev.pose_reached

    # fixed assignment ignores positions
    fixed = GroupTarget.from_dict(
        "blocks_group.n.01_1",
        {
            "members": names,
            "assignment": {"mode": "fixed", "member_to_slot": {names[i]: f"slot_{3 - i}" for i in range(4)}},
            "target_slots": slots,
            "pose_tolerances": {"position": 0.02, "orientation": 0.1},
        },
    )
    place([0, 1, 2, 3])
    group.set_target(fixed)
    ev = group.evaluate()
    assert ev.assignment == fixed.member_to_slot and not ev.pose_reached and ev.layout_reached is None


# ----------------------------------------------------------------------------------------------------------------
# Task-level flow
# ----------------------------------------------------------------------------------------------------------------


def _alignment_cfg(task_extra=None):
    cfg = {
        "env": {"action_frequency": 30, "physics_frequency": 120},
        "scene": {"type": "Scene", "use_floor_plane": False},
        "objects": [
            {"type": "PrimitiveObject", "name": "floor_slab", "primitive_type": "Cube", "scale": [12.0, 12.0, 0.1], "position": [1.0, 0.0, -0.05], "fixed_base": True},
            {"type": "DatasetObject", "name": "table", "category": "breakfast_table", "model": "skczfi", "position": [1.5, 0.0, 0.45], "fixed_base": True},
            {"type": "DatasetObject", "name": "rod", "category": "fishing_rod", "model": "zhmrtc", "position": [1.5, 0.0, 0.65]},
        ],
        "robots": [{"type": "R1Pro", "name": "robot_r1", "obs_modalities": ["rgb"], "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}],
        "task": {
            "type": "ISpatialGymTask",
            "activity_name": "alignment",
            "activity_definition_id": 0,
            "object_bindings": {"fishing_rod.n.01_1": "rod", "table.n.02_1": "table", "floor.n.01_1": "floor_slab"},
            "termination_config": {"max_steps": 80},
        },
    }
    cfg["task"].update(task_extra or {})
    return cfg


@pytest.fixture(scope="module")
def task_env(state_env):
    """alignment-0 (table_surface) on a breakfast table; the fixture depends on ``state_env`` so that the first
    environment is cleared before the second one is created."""
    og.clear()
    env = og.Environment(configs=_alignment_cfg())
    yield env
    og.clear()


@pytest.fixture(scope="module")
def measure_task_env(task_env):
    """The same scene with a TEST-ONLY copy of alignment-0, written to a temporary activity root, whose
    ``pose_reached`` atom is moved from the goal to ``(:measure ...)``. It exercises the non-gating path (success is
    relational only, the rod's orientation error is reported without gating). The shipped alignment problems are
    untouched: there ``pose_reached`` stays in the goal and gates success."""
    og.clear()
    from ispatialgym.activity import ACTIVITY_DEFINITIONS_ROOT

    tmp = tempfile.mkdtemp(prefix="isg_measure_")
    os.makedirs(os.path.join(tmp, "alignment"))
    with open(os.path.join(ACTIVITY_DEFINITIONS_ROOT, "alignment", "problem0.bddl")) as f:
        text = f.read()
    assert "            (pose_reached ?fishing_rod.n.01_1)\n" in text
    text = text.replace("            (pose_reached ?fishing_rod.n.01_1)\n", "")
    text = text.rstrip()[:-1] + "\n    (:measure\n        (pose_reached ?fishing_rod.n.01_1)\n    )\n)\n"
    with open(os.path.join(tmp, "alignment", "problem0.bddl"), "w") as f:
        f.write(text)
    env = og.Environment(configs=_alignment_cfg({"activity_definitions_root": tmp}))
    yield env
    og.clear()


def _write_tro(env, q_goal, path):
    rod = env.scene.object_registry("name", "rod")
    tro = {
        "fishing_rod.n.01_1": rod.dump_state(serialized=False),
        "ispatialgym": {
            "schema_version": "isg-tro-v1",
            "activity": "alignment",
            "activity_definition_id": 0,
            "task_variant": "table_surface",
            "instance_id": "alignment-test",
            "scene": {"model": "Scene"},
            "object_bindings": {
                "fishing_rod.n.01_1": {"scene_object_name": "rod", "category": "fishing_rod", "model": "zhmrtc"},
                "table.n.02_1": {"scene_object_name": "table"},
                "floor.n.01_1": {"scene_object_name": "floor_slab"},
            },
            "pose_targets": {
                "fishing_rod.n.01_1": {
                    "link": None,
                    "frame": "world",
                    "position": None,
                    "orientation": q_goal.tolist(),
                    "position_metric": None,
                    "orientation_metric": "full",
                    "position_tolerance": None,
                    "orientation_tolerance": 0.05,
                    "orientation_equivalences": [],
                }
            },
        },
    }
    with open(path, "w") as f:
        json.dump(tro, f, cls=TorchEncoder)


def test_alignment_task_two_level_evaluation(task_env):
    env = task_env
    task = env.task
    rod = env.scene.object_registry("name", "rod")
    assert task.activity_name == "alignment" and task.problem.name == "alignment-0"
    assert task.task_variant == "table_surface"
    assert task.compiled_task.pose_reached_args == ["fishing_rod.n.01_1"]
    assert not task.measured_pose_instances  # nothing configured yet

    for _ in range(60):  # let the rod come to rest on the table
        og.sim.step()
    _, q0 = rod.get_position_orientation()
    q_goal = T.quat_multiply(_yaw(-0.6), q0)  # goal = settled pose yawed by -0.6 rad
    with tempfile.TemporaryDirectory() as tmp:
        tro_path = os.path.join(tmp, "tro.json")
        _write_tro(env, q_goal, tro_path)
        task.load_tro_instance(env, tro_path)

    assert task.measured_pose_instances == ["fishing_rod.n.01_1"]
    report = rod.states[PoseReached].get_error_report()
    assert report["position_error"] is None and report["orientation_error"] == pytest.approx(0.6, abs=0.02)
    ok, status = task.compiled_task.check_initial_conditions(task._evaluate_predicate)
    assert ok, status

    teleport_step = 5
    trace = []

    def policy(obs, info):
        step = 0 if info is None else isg_info(info)["step"] + 1
        if step == teleport_step:
            p, _ = rod.get_position_orientation()
            rod.set_position_orientation(position=p, orientation=q_goal)
            rod.keep_still()
        return th.zeros(env.robots[0].action_dim)

    def on_step(i, info):
        isg = isg_info(info)
        trace.append((isg["step"], isg["binary_success"], isg["settled"], info["done"]["success"]))

    result = run_episode(env, policy, on_step=on_step)

    # Level 1: binary success at the teleport step, official success follows it immediately
    assert result["binary_success"] and result["first_success_step"] == teleport_step
    assert all(not b for (s, b, _, _) in trace if s < teleport_step)
    assert all(b and succ for (s, b, _, succ) in trace if s >= teleport_step)
    assert result["success"] and result["terminated"] and not result["truncated"]
    # Level 2: settled some steps later; latency measured from the first binary success
    assert result["settled_success"] and result["first_settled_success_step"] > teleport_step
    assert result["settle_latency_steps"] == result["first_settled_success_step"] - teleport_step
    # the teleport perturbs the rod, so a full PoseSettled window must elapse before it counts as settled
    assert result["settle_latency_steps"] >= rod.states[PoseSettled].n_steps - 1
    assert result["settle_latency_seconds"] == pytest.approx(result["settle_latency_steps"] * og.sim.get_sim_step_dt())
    # Errors: snapshot at the first binary success, orientation-only target
    assert result["position_error"] is None
    assert result["orientation_error"] is not None and result["orientation_error"] <= 0.05
    assert result["errors"]["pose_errors"]["fishing_rod.n.01_1"]["orientation_metric"] == "full"
    assert result["layout_position_error"] is None and result["layout_orientation_error"] is None
    kinds = {c["bddl"]: c["kind"] for c in result["conditions"]}
    assert kinds["(pose_reached ?fishing_rod.n.01_1)"] == "binary"
    assert kinds["(pose_settled ?agent.n.01_1)"] == "pose_settled"
    assert all(c["satisfied"] for c in result["conditions"])

    # A reset returns to the snapshotted initial state: goal not reached, tracking cleared
    env.reset()
    _, _, _, _, info = env.step(th.zeros(env.robots[0].action_dim))
    isg = isg_info(info)
    assert not isg["binary_success"] and isg["first_success_step"] == -1 and isg["settle_latency_steps"] == -1
    assert isg["errors"]["max_orientation_error"] == pytest.approx(0.6, abs=0.05)
    assert isg["final_errors"] is isg["errors"] or isg["final_errors"] == isg["errors"]


def test_alignment_task_non_gating_measure_atom(measure_task_env):
    """(:measure (pose_reached ?rod)): the target is configured and reported but never gates success, and the
    representative error is the one at the first binary success even if the pose improves afterwards."""
    env = measure_task_env
    task = env.task
    rod = env.scene.object_registry("name", "rod")
    assert task.compiled_task.pose_reached_args == []  # nothing gates through pose_reached ...
    assert task.compiled_task.measure_pose_reached_args == ["fishing_rod.n.01_1"]  # ... but the rod is measured

    for _ in range(60):
        og.sim.step()
    _, q0 = rod.get_position_orientation()
    q_goal = T.quat_multiply(_yaw(-0.6), q0)
    with tempfile.TemporaryDirectory() as tmp:
        tro_path = os.path.join(tmp, "tro.json")
        _write_tro(env, q_goal, tro_path)
        task.load_tro_instance(env, tro_path)
    assert task.measured_pose_instances == ["fishing_rod.n.01_1"]

    teleport_step = 5

    def policy(obs, info):
        step = 0 if info is None else isg_info(info)["step"] + 1
        if step == teleport_step:
            p, _ = rod.get_position_orientation()
            rod.set_position_orientation(position=p, orientation=q_goal)
            rod.keep_still()
        return th.zeros(env.robots[0].action_dim)

    result = run_episode(env, policy)

    # binary success is immediate (ontop + released), independent of the 0.6 rad orientation error
    assert result["binary_success"] and result["first_success_step"] == 0 and result["success"]
    kinds = {c["bddl"]: c["kind"] for c in result["conditions"]}
    assert "(pose_reached ?fishing_rod.n.01_1)" not in kinds
    assert result["measured"] == [
        {"bddl": "(pose_reached ?fishing_rod.n.01_1)", "kind": "measure", "configured": True, "within_tolerance": True}
    ]
    # representative error = error at the first binary success (0.6 rad), although the rod was rotated to the goal
    # at step 5 and the terminal error is small
    rod_err = result["errors"]["pose_errors"]["fishing_rod.n.01_1"]
    assert rod_err["gating"] is False and rod_err["orientation_metric"] == "full"
    assert result["orientation_error"] == pytest.approx(0.6, abs=0.03) and result["position_error"] is None
    assert result["errors_at_termination"]["max_orientation_error"] < 0.05
    # settle latency is measured from step 0
    assert result["settled_success"] and result["settle_latency_steps"] == result["first_settled_success_step"]
    assert result["settle_latency_steps"] >= teleport_step
