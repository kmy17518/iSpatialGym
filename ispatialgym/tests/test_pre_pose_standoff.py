"""Owner 2026-10-12 (axial#02, the paperback on desk_ohjotf_0, c3): a contact pre-pose's standoff that does not track from the recorded
witness is not planned first - the pre-pose is planned straight to with the witness, the supports present (the skip branch's call);
the bare standoff only when that plan fails before the robot moved (``reach_pre_pose``). ``contact_route_goal`` predicts the same
transit end. Every other situation (no supports, none near, clear of them, a standoff that tracks, no witness) runs as before.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.motions import MotionsConfig, SkillError

ARM = "right"
IDX = {a: th.tensor([M.ALL_JOINTS.index(n) for n in M.ARM_JOINTS[a]]) for a in M.ARMS}
QUAT = [0.0, 0.0, 0.0, 1.0]
DESK_TOP = 0.75


class _Support:
    def __init__(self, name: str, lo, hi):
        self.name = name
        self.aabb = (th.tensor(lo, dtype=th.float32), th.tensor(hi, dtype=th.float32))


def _desk():
    return _Support("desk_ohjotf_0", [0.0, 0.0, 0.0], [1.0, 1.0, DESK_TOP])


def _pre(height: float):
    """A push pre-contact ``height`` above the desk top (inside its footprint: the escape direction is straight up)."""
    return SimpleNamespace(position=[0.5, 0.5, DESK_TOP + height], orientation=list(QUAT))


def _witness() -> th.Tensor:
    q = th.zeros(len(M.ALL_JOINTS))
    q[IDX[ARM]] = th.tensor([0.9, -2.1, -0.7, -1.6, 1.5, -0.7, 0.4])
    return q


class _Motions:
    """The motions / backend surface ``reach_pre_pose`` and ``contact_route_goal`` touch. ``track``: the differential IK of the
    standoff leg from the witness converges. ``direct``: how ``arm_to(pose=pre, q_hint=...)`` behaves - "ok" (2 actions),
    "fail_before" (``plan_failed`` before any action), "fail_after" (1 action, then ``plan_failed``), "error_before" (an
    ``execution`` error before any action)."""

    def __init__(self, track: bool = True, direct: str = "ok"):
        self.robot = SimpleNamespace(arm_control_idx=IDX)
        self.cfg = MotionsConfig()
        self.calls: List[Dict[str, Any]] = []
        self.last_relaxed_links: List[str] = []
        self.direct = direct
        self.ik_calls = 0

        def solve_ik_path(emb, waypoints, q_start=None, **kw):
            self.ik_calls += 1
            if not track:
                return SimpleNamespace(success=False, q_traj=None, status="differential IK did not converge at waypoint 2/8")
            q_end = q_start.clone()
            q_end[IDX[ARM][3]] += 0.25  # the standoff's configuration in the witness's branch
            return SimpleNamespace(success=True, q_traj=th.stack([q_start.clone(), q_end]), status="ok")

        self.backend = SimpleNamespace(q_full=lambda: th.zeros(len(M.ALL_JOINTS)), solve_ik_path=solve_ik_path)

    def _densify(self, start, waypoints):
        return list(waypoints)

    def arm_to(self, arm, pose=None, q=None, ignore_objects=(), q_hint=None, max_excursion=None, ignore_links=(), accept_other_branch=False):
        call = {"fn": "arm_to", "arm": arm, "pose": None if pose is None else EX.as_world_pose(pose), "q": q, "q_hint": q_hint,
                "ignore_objects": [getattr(o, "name", o) for o in ignore_objects], "ignore_links": list(ignore_links),
                "accept_other_branch": accept_other_branch}
        self.calls.append(call)
        direct = pose is not None and q_hint is not None and self.direct != "ok"
        if direct and self.direct == "fail_before":
            raise SkillError("plan_failed", f"arm_to({arm}, witness q): joint-space plan failed; pose-goal fallback: IK failed", stage="trajopt")
        if direct and self.direct == "error_before":
            raise SkillError("execution", f"arm_to({arm}): something else", stage="trajopt")
        yield th.zeros(3)
        if direct and self.direct == "fail_after":
            raise SkillError("plan_failed", f"arm_to({arm}): protected-object re-plan failed; no plan from the lifted hand", stage="protected")
        yield th.zeros(3)

    def arm_along(self, arm, waypoints, **kw):
        self.calls.append({"fn": "arm_along", "arm": arm, "waypoints": [EX.as_world_pose(w) for w in waypoints], **kw})
        yield th.zeros(3)


def _reach(motions: _Motions, pre, supports, q_hint: Optional[th.Tensor], through=(), accept_other_branch: bool = True, cfg=None):
    ctx: Dict[str, Any] = {}
    n = len(list(EX.reach_pre_pose(motions, SimpleNamespace(), ARM, pre, supports, list(through), q_hint, ctx, cfg or EX.ExecuteConfig(),
                                   accept_other_branch=accept_other_branch)))
    return n, ctx


def _same_pose(a, b) -> bool:
    return a is not None and b is not None and bool(th.allclose(a[0], b[0], atol=1e-6)) and bool(th.allclose(a[1], b[1], atol=1e-6))


def test_a_standoff_that_tracks_from_the_witness_runs_as_before():
    """The calls HEAD made: the standoff's configuration (joint goal, supports present, no hint / branch flag), then the straight
    leg to the pre-pose - for a push (``accept_other_branch``) and for a brace / grasp alike."""
    pre, desk, wit = _pre(0.072), _desk(), _witness()
    for other_branch in (True, False):
        m = _Motions(track=True)
        n, ctx = _reach(m, pre, [desk], wit, through=["book"], accept_other_branch=other_branch)
        assert [c["fn"] for c in m.calls] == ["arm_to", "arm_along"] and n == 3
        go, leg = m.calls
        assert go["pose"] is None and go["q_hint"] is None and go["accept_other_branch"] is False
        assert go["ignore_objects"] == ["book"] and go["ignore_links"] == []
        q_exp = wit.clone()
        q_exp[IDX[ARM][3]] += 0.25
        assert th.allclose(go["q"], q_exp)
        assert _same_pose(leg["waypoints"][0], EX.as_world_pose(pre)) and leg["soft_pos_tol"] == m.cfg.diff_ik_soft_pos_tol_free
        note = ctx["pre_pose_standoff"]
        assert note["branch"] == "witness" and "direct_failed" not in note and note["direction"] == [0.0, 0.0, 1.0] and note["distance_m"] == 0.08


def test_outside_the_untracked_standoff_every_branch_runs_as_before():
    wit, desk = _witness(), _desk()
    # no supports: planned to the pre-pose with the witness, nothing ignored but the pushed object
    m = _Motions(track=False)
    _reach(m, _pre(0.03), [], wit, through=["book"])
    assert [(c["fn"], c["ignore_objects"]) for c in m.calls] == [("arm_to", ["book"])] and m.calls[0]["q_hint"] is wit and m.ik_calls == 0
    # no support near the pre-pose (> 15 cm): planned to it with the supports ignored
    m = _Motions(track=False)
    _, ctx = _reach(m, _pre(0.20), [desk], wit, through=["book"])
    assert [(c["fn"], c["ignore_objects"]) for c in m.calls] == [("arm_to", ["desk_ohjotf_0", "book"])] and "pre_pose_standoff" not in ctx and m.ik_calls == 0
    # clear of the supports (>= pre_pose_standoff_skip_m): the skip branch, supports present
    m = _Motions(track=False)
    _, ctx = _reach(m, _pre(0.09), [desk], wit, through=["book"])
    assert [(c["fn"], c["ignore_objects"]) for c in m.calls] == [("arm_to", ["book"])] and m.calls[0]["q_hint"] is wit
    assert ctx["pre_pose_standoff"] == {"skipped": True, "clearance_m": 0.09} and m.ik_calls == 0
    # no witness (a lowered / moved pre-pose): the bare standoff as before, then the straight leg
    for track in (True, False):
        m = _Motions(track=track)
        _, ctx = _reach(m, _pre(0.05), [desk], None, through=["book"])
        assert [c["fn"] for c in m.calls] == ["arm_to", "arm_along"] and m.ik_calls == 0
        go = m.calls[0]
        assert _same_pose(go["pose"], (th.tensor([0.5, 0.5, DESK_TOP + 0.05 + 0.08]), th.tensor(QUAT))) and go["q_hint"] is None
        assert go["ignore_objects"] == ["book"] and go["accept_other_branch"] is False and ctx["pre_pose_standoff"]["branch"] == "planner"


def test_an_untracked_standoff_plans_straight_to_the_pre_pose_with_the_witness():
    """c3_k0: the pre-contact 7.2 cm above the desk, the standoff not trackable from the witness - the skip branch's call
    (the pre-pose, the witness, the desk in the collision world), no standoff and no straight leg after it."""
    pre, desk, wit = _pre(0.072), _desk(), _witness()
    for other_branch in (True, False):
        m = _Motions(track=False)
        n, ctx = _reach(m, pre, [desk], wit, through=["book"], accept_other_branch=other_branch)
        assert [c["fn"] for c in m.calls] == ["arm_to"] and n == 2 and m.ik_calls == 1
        go = m.calls[0]
        assert _same_pose(go["pose"], EX.as_world_pose(pre)) and go["q_hint"] is wit and go["accept_other_branch"] is other_branch
        assert go["ignore_objects"] == ["book"]  # the desk stays in the collision world
        note = ctx["pre_pose_standoff"]
        assert note["branch"] == "direct" and "direct_failed" not in note and note["clearance_m"] == pytest.approx(0.072, abs=1e-3)


def test_the_bare_standoff_only_when_the_direct_plan_fails_before_the_robot_moved():
    pre, desk, wit = _pre(0.072), _desk(), _witness()
    m = _Motions(track=False, direct="fail_before")
    n, ctx = _reach(m, pre, [desk], wit, through=["book"])
    assert [c["fn"] for c in m.calls] == ["arm_to", "arm_to", "arm_along"] and n == 3
    direct, bare, leg = m.calls
    assert direct["q_hint"] is wit and _same_pose(direct["pose"], EX.as_world_pose(pre))
    assert _same_pose(bare["pose"], (th.tensor([0.5, 0.5, DESK_TOP + 0.072 + 0.08]), th.tensor(QUAT))) and bare["q_hint"] is None
    assert bare["ignore_objects"] == ["book"] and _same_pose(leg["waypoints"][0], EX.as_world_pose(pre))
    note = ctx["pre_pose_standoff"]
    assert note["branch"] == "planner" and "joint-space plan failed" in note["direct_failed"]
    # a direct plan that failed after the robot moved, or with another reason: raised as it is, no standoff from there
    for direct, reason in (("fail_after", "plan_failed"), ("error_before", "execution")):
        m = _Motions(track=False, direct=direct)
        with pytest.raises(SkillError) as e:
            _reach(m, pre, [desk], wit, through=["book"])
        assert e.value.reason == reason and [c["fn"] for c in m.calls] == ["arm_to"] and m.calls[0]["q_hint"] is wit


def test_closing_the_reach_closes_the_direct_plan():
    """The executor closes an action's generator on a QA verdict: the direct plan's generator is closed with it."""
    closed = []
    m = _Motions(track=False)
    arm_to = m.arm_to

    def arm_to_closing(*a, **kw):
        try:
            yield from arm_to(*a, **kw)
        finally:
            closed.append(True)

    m.arm_to = arm_to_closing
    gen = EX.reach_pre_pose(m, SimpleNamespace(), ARM, _pre(0.072), [_desk()], ["book"], _witness(), {}, EX.ExecuteConfig(), accept_other_branch=True)
    next(gen)
    gen.close()
    assert closed == [True]


def test_contact_route_goal_follows_the_same_rule():
    """The transit end ``keep_contact_arms`` checks a free arm's route to: the standoff in the witness's branch when it tracks,
    the witness itself (the pre-pose planned straight to) when it does not; the witness when clear of the supports."""
    desk, wit = _desk(), _witness()
    q_ctx = th.zeros(len(M.ALL_JOINTS))
    state = SimpleNamespace(contact_supports=lambda name: [desk])
    contact = {"name": "push", "object": "paperback_book.n.01_1", "targets": {"pre": _pre(0.072)}, "witnesses": {"pre": wit}}
    goal, label = EX.contact_route_goal(_Motions(track=True), state, ARM, contact, q_ctx, EX.ExecuteConfig())
    assert float(goal[IDX[ARM][3]] - wit[IDX[ARM][3]]) == pytest.approx(0.25) and label == "the push's standoff in the pre witness's branch"
    goal, label = EX.contact_route_goal(_Motions(track=False), state, ARM, contact, q_ctx, EX.ExecuteConfig())
    assert th.allclose(goal[IDX[ARM]], wit[IDX[ARM]]) and "does not track" in label and "planned straight to the pre-pose" in label
    contact["targets"] = {"pre": _pre(0.09)}
    m = _Motions(track=True)
    goal, label = EX.contact_route_goal(m, state, ARM, contact, q_ctx, EX.ExecuteConfig())
    assert th.allclose(goal[IDX[ARM]], wit[IDX[ARM]]) and "clear of its supports" in label and m.ik_calls == 0
