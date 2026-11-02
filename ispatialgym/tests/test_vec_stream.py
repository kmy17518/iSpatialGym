"""Gate 3a of the vectorized evaluator: ``VecStreamExecutor`` on N fake environments (no simulator).

Every environment runs its own scripted stream against the single-env fake world of ``test_v2_oracle`` (toy kinematics,
30 Hz); the per-env goal (the robot base within 5 cm of a point, a ``pose_reached`` target) is scored by the fake task's tracker
(``ispatialgym.eval.metrics``, settle-eval off as a policy run's). Checked:

* per-env geometric / symbolic success and their steps; a stream has no epilogue: it ends at the source's ``None`` (horizon,
  stop on success) or at ``max_steps`` and the metrics judge it (``not_reached`` when the goal never held);
* environments finishing at different steps, one never reaching the goal; a finished env's robot receives its no-op (last
  action, base stopped) and does not move while the others continue;
* parity: every slot's result equals the single-env ``Executor.run_stream`` run of the same scenario alone (same n_steps, reason,
  success steps, errors, final robot pose), so the batch is N independent episodes.
"""

from __future__ import annotations

import math
import types
from typing import Any, Dict, List, Optional

import pytest
import torch as th

from ispatialgym.eval import metrics as MET
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.executor import Executor, ExecutorConfig
from ispatialgym.oracle.vec_stream import Slot, VecStreamExecutor
from ispatialgym.tests.test_v2_oracle import CAM_KEY, FakeEnv, FakeRobot

AGENT = "agent.n.01_1"


# ---------------------------------------------------------------------------------------------------------------- fakes


class GoalTracker:
    """The fake world's per-step facts: the goal is the robot base within 5 cm of ``goal_xy`` (a ``pose_reached`` atom in :goal
    and :measure)."""

    def __init__(self, goal_xy, tol: float = 0.05):
        self.goal = th.tensor(goal_xy, dtype=th.float32)
        self.tol = tol

    def __call__(self, env: FakeEnv) -> Dict[str, Any]:
        pos, _ = env.robot.get_position_orientation()
        err = float(th.linalg.norm(pos[:2] - self.goal))
        within = err <= self.tol
        errors = {"pose_errors": {AGENT: {"position_error": err, "orientation_error": 0.0, "within_tolerance": within, "in_goal": True, "in_measure": True}},
                  "layout_errors": {}}
        return {"binary_success": within, "settled": False, "errors": errors, "group_assignments": {},
                "conditions": [{"index": 0, "bddl": f"(pose_reached ?{AGENT})", "kind": "binary", "satisfied": within}]}


class FakeVecEnv:
    """N single-env fakes stepped together (``og.Environment`` with ``num_envs`` scenes)."""

    def __init__(self, envs: List[FakeEnv]):
        self.envs = envs
        self.num_envs = len(envs)
        self.scenes = [e.scene for e in envs]
        self.action_log: List[th.Tensor] = []

    def step(self, actions, n_render_iterations=1):
        actions = th.as_tensor(actions, dtype=th.float32)
        assert actions.shape == (self.num_envs, M.ACTION_DIM)
        self.action_log.append(actions.clone())
        obs, rew, term, trunc, infos = [], [], [], [], []
        for i, e in enumerate(self.envs):
            o, r, t, tr, info = e.step(actions[i], n_render_iterations=n_render_iterations)
            obs.append(o)
            rew.append(float(r))
            term.append(bool(t))
            trunc.append(bool(tr))
            infos.append(info)
        return obs, th.tensor(rew), th.tensor(term), th.tensor(trunc), infos


def hold_action(robot: FakeRobot, base=(0.0, 0.0, 0.0)) -> th.Tensor:
    a = th.zeros(M.ACTION_DIM)
    idx = robot.controller_action_idx
    a[idx["base"]] = th.tensor(base, dtype=th.float32)
    a[idx["trunk"]] = robot.q[robot.trunk_control_idx]
    for arm in M.ARMS:
        a[idx[f"arm_{arm}"]] = robot.q[robot.arm_control_idx[arm]]
        opening = float(robot.q[robot.gripper_control_idx[arm]].max())
        a[idx[f"gripper_{arm}"]] = 2.0 * opening / 0.05 - 1.0
    return a


class Drive:
    """A scripted stream: drive straight at ``vx`` (normalized) for ``horizon`` steps; ``stop_on_success`` ends it (``None``)
    once the task reports ``official_success`` - the ACT bridge's rule."""

    def __init__(self, robot: FakeRobot, vx: float, horizon: int, stop_on_success: bool = True):
        self.robot, self.vx, self.horizon, self.stop_on_success = robot, vx, horizon, stop_on_success
        self.steps = 0
        self.stop_reason: Optional[str] = None

    def __call__(self, obs, info):
        isg = (info or {}).get("done", {}).get("ispatialgym") or {}
        if self.stop_on_success and isg.get("official_success"):
            self.stop_reason = "success"
            return None
        if self.steps >= self.horizon:
            self.stop_reason = "horizon"
            return None
        self.steps += 1
        return hold_action(self.robot, base=(self.vx, 0.0, 0.0))


SCENARIOS = {
    # name: (goal distance along +x, normalized vx, horizon, stop_on_success, max_steps)
    "reach_then_stop": (0.5, 0.5, 120, True, 20000),  # geometric success, the stream stops there
    "never_reached": (5.0, 0.5, 120, True, 20000),  # the horizon ends the stream: not_reached
    "reach_and_pass": (0.5, 0.5, 80, False, 20000),  # the goal held around step 40, then the robot drove on: still a success (off mode)
    "capped": (5.0, 0.5, 120, True, 30),  # max_steps ends the stream at 30, no timeout verdict: not_reached
}


def make_world(name: str):
    dist, vx, horizon, stop, max_steps = SCENARIOS[name]
    robot = FakeRobot()
    env = FakeEnv(robot, {}, evaluate=GoalTracker((dist, 0.0)))
    env.task.tracker = MET.SuccessTracker(settle_eval=False, settle_steps=45)
    source = Drive(robot, vx, horizon, stop_on_success=stop)
    cfg = ExecutorConfig(max_steps=max_steps)
    return env, source, cfg


class _State:
    held_objects = staticmethod(lambda: [])


def run_single(name: str) -> Dict[str, Any]:
    """The single-env executor on the same scenario."""
    env, source, cfg = make_world(name)
    ex = Executor(env, env.task, None, recorder=None, motions=None, monitors=None, cfg=cfg, sim=env.sim, state=_State(), tracked_objects={}, cam_key=CAM_KEY)
    obs, info = env._obs(), env._info()
    ex.last_obs, ex.last_info = obs, info

    def stream():
        o, i = obs, info
        while True:
            a = source(o, i)
            if a is None:
                return
            yield a
            o, i = ex.last_obs, ex.last_info

    result = ex.run_stream(stream())
    result["_pose"] = env.robot.get_position_orientation()
    result["_steps"] = env.steps
    return result


def run_vec(names: List[str]):
    worlds = [make_world(n) for n in names]
    venv = FakeVecEnv([w[0] for w in worlds])
    slots = [Slot(idx=i, robot=w[0].robot, source=w[1], cfg=w[2], task=w[0].task, tracked={}, held=lambda: []) for i, w in enumerate(worlds)]
    vex = VecStreamExecutor(venv, slots, sim=None)
    results = vex.run([w[0]._obs() for w in worlds], [w[0]._info() for w in worlds])
    return venv, slots, results


# ---------------------------------------------------------------------------------------------------------------- tests


def test_four_environments_finish_independently_with_their_own_verdicts():
    names = ["reach_then_stop", "never_reached", "reach_and_pass", "capped"]
    venv, slots, results = run_vec(names)
    by = dict(zip(names, results))

    r = by["reach_then_stop"]
    assert r["success"] is True and r["reason"] is None and r["geometric_success"] and r["symbolic_success"]
    assert r["epilogue_steps"] == 0 and r["n_steps"] == r["n_action_steps"]
    assert r["n_action_steps"] == r["geometric_success_step"]  # the stream ends at the geometric success (the fake counts steps after stepping)

    r = by["never_reached"]
    assert r["success"] is False and r["reason"] == "not_reached" and r["geometric_success_step"] is None and r["n_steps"] == 120

    r = by["reach_and_pass"]
    # the goal held, then the robot drove on: off mode, the first step decides
    assert r["success"] is True and r["geometric_success_step"] < 80 and r["n_steps"] == 80
    assert r["errors"]["pose_errors"][AGENT]["within_tolerance"] and not r["errors_final_step"]["pose_errors"][AGENT]["within_tolerance"]

    r = by["capped"]
    assert r["success"] is False and r["reason"] == "not_reached" and r["n_steps"] == 30 and r["n_action_steps"] == 30

    # the batch ran until the longest episode; every env was stepped every batch step
    assert len(venv.action_log) == max(r["n_steps"] for r in results) == 120
    for e in venv.envs:
        assert e.steps == 120


def test_finished_environment_receives_its_noop_and_stays_put():
    names = ["capped", "reach_then_stop", "reach_and_pass"]
    venv, slots, results = run_vec(names)
    for i, (name, r) in enumerate(zip(names, results)):
        done_at = r["n_steps"]
        last = slots[i].last_action.clone()
        last[slots[i].robot.controller_action_idx["base"]] = 0.0
        later = [a[i] for a in venv.action_log[done_at:]]
        if done_at == len(venv.action_log):
            continue  # the last episode of the batch: no later step
        assert later, name
        for a in later:
            assert th.equal(a, last), name
    # the timed-out robot (done at step 30) did not move afterwards
    env_t = venv.envs[0]
    pose_mid = None

    env2, src2, cfg2 = make_world("capped")
    # replay the batch actions for env 0 on a fresh fake up to its verdict, then compare with the final pose
    for a in venv.action_log[:30]:
        env2.step(a[0])
    pose_mid = env2.robot.get_position_orientation()
    pose_end = env_t.robot.get_position_orientation()
    assert th.allclose(pose_mid[0], pose_end[0]) and th.allclose(pose_mid[1], pose_end[1])


@pytest.mark.parametrize("names", [
    ["reach_then_stop", "never_reached", "reach_and_pass", "capped"],
    ["never_reached", "never_reached", "never_reached"],  # the same episode in every slot
    ["reach_and_pass"],
])
def test_each_slot_matches_the_single_env_executor(names):
    venv, slots, results = run_vec(names)
    keys = ("success", "reason", "n_steps", "n_action_steps", "n_settle_steps", "epilogue_steps", "symbolic_success", "symbolic_success_step",
            "geometric_success", "geometric_success_step", "goal_first_step", "error_metric", "other_mode")
    for name, r, s in zip(names, results, slots):
        single = run_single(name)
        for k in keys:
            assert r[k] == single[k], (name, k, r[k], single[k])
        if r["verdict"] is not None:
            assert r["verdict"]["category"] == single["verdict"]["category"]
            assert r["verdict"]["message"] == single["verdict"]["message"]
        assert r["errors"] == single["errors"] and r["errors_final_step"] == single["errors_final_step"]
        pos, quat = s.robot.get_position_orientation()
        assert th.equal(pos, single["_pose"][0]) and th.equal(quat, single["_pose"][1]), name
        for k in ("joint_rad", "base_m", "base_rad"):
            assert math.isclose(r["motion_delta"][k], single["motion_delta"][k], abs_tol=1e-9), (name, k)


def test_result_dict_has_the_single_env_keys():
    _, _, (r,) = run_vec(["reach_then_stop"])
    single = run_single("reach_then_stop")
    missing = set(single) - set(r) - {"_pose", "_steps"}
    assert not missing, missing


# ---------------------------------------------------------------------------------------------------------------- clone frames


class _Scene:
    def __init__(self, pose, idx=1):
        self.pose, self.idx = th.as_tensor(pose, dtype=th.float32), idx


def _translated(x, y=0.0, z=0.0):
    p = th.eye(4)
    p[:3, 3] = th.tensor([x, y, z])
    return p


def test_clone_frames_translate_targets_objects_and_robot_but_not_the_base_scene():
    from ispatialgym.eval.clone_frames import offset_object_state, offset_targets_meta, scene_translation, to_base, to_clone
    from ispatialgym.eval.targets import PoseTarget

    base, clone = _Scene(th.eye(4), idx=0), _Scene(_translated(40.0), idx=1)
    assert scene_translation(None) is None and scene_translation(base) is None
    assert th.equal(scene_translation(clone), th.tensor([40.0, 0.0, 0.0]))
    # a world-frame pose target: translated, the orientation untouched; an object-relative frame: untouched
    world = PoseTarget("agent.n.01_1", "zed_link:Camera:0", "world", th.tensor([15.4, 1.94, 1.15]), th.tensor([0.1, 0.2, 0.3, 0.927]), "full", "full", 0.05, 0.1)
    rel = PoseTarget("cup.n.01_1", None, "table.n.01_1", th.tensor([0.1, 0.0, 0.0]), None, "full", None, 0.01, None)
    meta = types.SimpleNamespace(pose_targets={"agent.n.01_1": world, "cup.n.01_1": rel})
    out = offset_targets_meta(meta, clone)
    assert out is not meta and th.allclose(out.pose_targets["agent.n.01_1"].position, th.tensor([55.4, 1.94, 1.15]))
    assert th.equal(out.pose_targets["agent.n.01_1"].orientation, world.orientation)
    assert th.equal(out.pose_targets["cup.n.01_1"].position, rel.position)
    assert offset_targets_meta(meta, base) is meta  # bit-identical path for the base scene
    assert th.equal(world.position, th.tensor([15.4, 1.94, 1.15]))  # the TRO meta itself is untouched
    # a group target: the absolute slot positions move with the clone, the layout relations (member-relative) do not
    from ispatialgym.eval.targets import GroupTarget, LayoutRelation, Slot

    group = GroupTarget("assembly", ["lid.n.01_1", "pot.n.01_1"], "fixed", "slot_position", {"lid.n.01_1": "s0", "pot.n.01_1": "s1"}, [],
                        {"s0": Slot("s0", th.tensor([1.0, 2.0, 0.9]), th.tensor([0.0, 0.0, 0.0, 1.0])),
                         "s1": Slot("s1", th.tensor([1.0, 2.0, 0.8]), th.tensor([0.0, 0.0, 0.0, 1.0]))},
                        (0.05, 0.1), [LayoutRelation("s1", "s0", th.tensor([0.0, 0.0, 0.1]), th.tensor([0.0, 0.0, 0.0, 1.0]))], (0.01, 0.1))
    meta = types.SimpleNamespace(pose_targets={}, groups={"assembly": group})
    out = offset_targets_meta(meta, clone)
    assert th.allclose(out.groups["assembly"].target_slots["s0"].position, th.tensor([41.0, 2.0, 0.9]))
    assert th.allclose(out.groups["assembly"].target_slots["s1"].position, th.tensor([41.0, 2.0, 0.8]))
    assert th.equal(out.groups["assembly"].layout_relations[0].position, th.tensor([0.0, 0.0, 0.1]))
    assert th.equal(group.target_slots["s0"].position, th.tensor([1.0, 2.0, 0.9]))  # the TRO meta itself is untouched
    assert out.groups["assembly"].members == group.members and out.groups["assembly"].pose_tolerances == (0.05, 0.1)
    # an object state: root position translated, orientation / joints untouched
    st = {"root_link": {"pos": th.tensor([1.0, 2.0, 0.5]), "ori": th.tensor([0.0, 0.0, 0.0, 1.0])}, "joint_pos": th.tensor([0.3])}
    out = offset_object_state(st, clone)
    assert th.equal(out["root_link"]["pos"], th.tensor([41.0, 2.0, 0.5])) and th.equal(out["root_link"]["ori"], th.tensor([0.0, 0.0, 0.0, 1.0]))
    # round trip
    assert th.equal(to_base(clone, to_clone(clone, th.tensor([3.0, 4.0, 5.0]))), th.tensor([3.0, 4.0, 5.0]))
    assert th.equal(to_clone(base, th.tensor([3.0, 4.0, 5.0])), th.tensor([3.0, 4.0, 5.0]))


def test_clone_frames_reject_a_rotated_placement():
    from ispatialgym.eval.clone_frames import scene_translation

    p = th.eye(4)
    p[0, 0], p[0, 1], p[1, 0], p[1, 1] = 0.0, -1.0, 1.0, 0.0
    p[:3, 3] = th.tensor([40.0, 0.0, 0.0])
    with pytest.raises(AssertionError):
        scene_translation(_Scene(p))


def test_attachment_context_is_per_scene():
    from ispatialgym.sampling import predicates as PRED

    class Ent:
        def __init__(self, name, idx):
            self.name, self.scene = name, types.SimpleNamespace(idx=idx)

    PRED.set_attachment_context([{"child": "bolt", "parent": "nut", "equivalences": [[0, 0, 1, 0]]}], scene_idx=0)
    PRED.set_attachment_context([{"child": "bolt", "parent": "nut", "equivalences": [[0, 0, 0, 1], [0, 1, 0, 0]]}], scene_idx=1)
    assert len(PRED.attachment_context(Ent("bolt", 0), Ent("nut", 0))["equivalences"]) == 1
    assert len(PRED.attachment_context(Ent("bolt", 1), Ent("nut", 1))["equivalences"]) == 2
    assert PRED.attachment_context(Ent("bolt", 2), Ent("nut", 2)) is None  # no registration for scene 2, no single-env fallback
    PRED.set_attachment_context([{"child": "bolt", "parent": "nut", "equivalences": []}])  # the single-env task (scene None)
    assert PRED.attachment_context(Ent("bolt", 2), Ent("nut", 2))["equivalences"] == []
    assert len(PRED.attachment_context(Ent("bolt", 1), Ent("nut", 1))["equivalences"]) == 2  # scene 1's entry kept
    PRED.set_attachment_context([], scene_idx=1)
    assert PRED.attachment_context(Ent("bolt", 1), Ent("nut", 1))["equivalences"] == []  # falls back to the single-env entry
    PRED.set_attachment_context([])
    PRED.set_attachment_context([], scene_idx=0)
