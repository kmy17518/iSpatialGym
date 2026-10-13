"""Owner 2026-10-12 (axial#02, the paperback on desk_ohjotf_0, c3): a contact pre-pose's standoff that does not track from the recorded
witness is not planned first - the pre-pose is planned straight to with the witness, the supports present (the skip branch's call);
the bare standoff only when that plan fails before the robot moved (``reach_pre_pose``). ``contact_route_goal`` predicts the same
transit end. Every other situation (no supports, none near, clear of them, a standoff that tracks, no witness) runs as before.
Item 7c: the program check applies the same decision to a push pre-contact (``program_check.push_pre_standoff``).
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


def _reach(motions: _Motions, pre, supports, q_hint: Optional[th.Tensor], through=(), accept_other_branch: bool = True, cfg=None, skip_above=None):
    ctx: Dict[str, Any] = {}
    n = len(list(EX.reach_pre_pose(motions, SimpleNamespace(), ARM, pre, supports, list(through), q_hint, ctx, cfg or EX.ExecuteConfig(),
                                   accept_other_branch=accept_other_branch, skip_above_m=skip_above)))
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


# ------------------------------------------------------------------------------------------------ 7c: the program check


class _PcSampler:
    """The sampler surface ``push_pre_standoff`` touches. ``track``: the standoff leg tracks from the witness. ``free``: the witness is
    collision-free with the supports present. ``refine``: arm-joint offset of the IK refinement of the pre-contact (None = no IK).
    ``bare``: the standoff pose has an IK solution."""

    def __init__(self, track=False, free=False, refine=None, bare=False):
        self.calls: List[Any] = []
        self.ignored: List[List[str]] = [[]]
        sampler = self

        def solve_ik_path(emb, waypoints, q_start=None, **kw):
            sampler.calls.append(("track", emb, [w[M.EEF_LINK[ARM]] for w in waypoints], q_start.clone(), kw))
            ok = track
            return SimpleNamespace(success=ok, q_traj=th.stack([q_start]) if ok else None, status="ok" if ok else "differential IK did not converge at waypoint 2/8")

        def ik(pose, q_seed, arm="right"):
            standoff = float(th.as_tensor(pose[0])[2]) > DESK_TOP + 0.12
            sampler.calls.append(("ik", "standoff" if standoff else "pre", list(sampler.ignored[-1])))
            if standoff:
                return q_seed.clone() if bare else None
            if refine is None:
                return None
            q = q_seed.clone()
            q[IDX[arm][0]] += refine
            return q

        class _Ign:
            def __init__(self, objs):
                self.objs = [getattr(o, "name", o) for o in objs]

            def __enter__(self):
                sampler.ignored.append(self.objs)

            def __exit__(self, *exc):
                sampler.ignored.pop()

        def collision_free_at(q, pose, arm, ignore=()):
            sampler.calls.append(("free", [getattr(o, "name", o) for o in ignore]))
            return free

        self.kit = SimpleNamespace(robot=SimpleNamespace(arm_control_idx=IDX), backend=SimpleNamespace(solve_ik_path=solve_ik_path), ik=ik,
                                   _ensure_synced=lambda: None)
        self.ikc = SimpleNamespace(collision_free_at=collision_free_at, ignoring=_Ign, _collision_blame=lambda q, ign, margin=0.0: "collides with desk_ohjotf_0")


def _pc(sampler, height=0.072):
    from ispatialgym.sampling.v2 import program_check as PC

    pre = _pre(height)
    return PC.push_pre_standoff(sampler, ARM, _witness(), (th.tensor(pre.position), th.tensor(pre.orientation)), [_desk()], [_Support("paperback_book_0", [0.4, 0.4, 0.75], [0.6, 0.6, 0.95])])


def test_program_check_push_pre_standoff_mirrors_the_oracles_decision():
    from ispatialgym.sampling.v2.ik import IKResult

    # no support near / clear of the supports: nothing is solved (accepted as before)
    s = _PcSampler()
    ok, why, rep = _pc(s, height=0.20)
    assert ok and rep == {} and s.calls == []
    ok, why, rep = _pc(s, height=0.09)
    assert ok and rep["branch"] == "skipped" and s.calls == []
    # the standoff tracks from the witness: the oracle's leg (8 cm straight up in 1 cm steps from the witness, no collision check)
    s = _PcSampler(track=True)
    ok, why, rep = _pc(s)
    assert ok and rep == {"clearance_m": pytest.approx(0.072, abs=1e-3), "branch": "witness"} and [c[0] for c in s.calls] == ["track"]
    _, emb, dense, q_start, kw = s.calls[0]
    assert emb == "right_arm" and len(dense) == 8 and th.allclose(q_start, _witness()) and kw["check_collisions"] is False
    assert th.allclose(th.as_tensor(dense[-1][0]), th.tensor([0.5, 0.5, DESK_TOP + 0.072 + 0.08]), atol=1e-5)
    assert kw["soft_pos_tol"] == MotionsConfig().diff_ik_soft_pos_tol_free
    # it does not track: straight to the pre-contact with the witness, the supports present (only the pushed object removed)
    s = _PcSampler(free=True)
    ok, why, rep = _pc(s)
    assert ok and rep["branch"] == "direct" and s.calls[1] == ("free", ["paperback_book_0"]) and len(s.calls) == 2
    # the witness collides there: its IK refinement within the oracle's witness_refine_max_jump is the goal instead
    s = _PcSampler(refine=0.3)
    ok, why, rep = _pc(s)
    assert ok and rep["branch"] == "direct" and rep["refined_jump_rad"] == pytest.approx(0.3) and s.calls[2] == ("ik", "pre", ["paperback_book_0"])
    # a refinement in another branch does not count; the bare standoff (an IK solution of its pose, supports present) does
    s = _PcSampler(refine=1.5, bare=True)
    ok, why, rep = _pc(s)
    assert ok and rep["branch"] == "planner" and s.calls[-1] == ("ik", "standoff", ["paperback_book_0"])
    # neither: rejected as approach_untrackable (c3_k0: the standoff out of reach, the bare standoff with no IK solution)
    s = _PcSampler()
    ok, why, rep = _pc(s)
    assert not ok and "branch" not in rep and why.startswith("approach_untrackable: push pre-contact 0.072 m from its supports")
    assert "collides with desk_ohjotf_0" in why and "waypoint 2/8" in why and IKResult(False, None, why, 1).code == "approach_untrackable"


def test_program_check_models_the_pre_contact_the_oracle_keeps_its_witness_for():
    """``execute_push`` lowers a straight push's pre-contact to ``push_pre_dist_m`` and drops its witness; an arc push, a pitched blade
    class and a pre-contact already that close keep the recorded one."""
    from ispatialgym.sampling.v2 import program_check as PC

    q = th.tensor(QUAT)
    init = (th.tensor([0.5, 0.5, 0.80]), q)  # the approach axis of the identity quaternion is world +z
    far, near = (th.tensor([0.5, 0.5, 0.70]), q), (th.tensor([0.5, 0.5, 0.76]), q)
    straight = SimpleNamespace(mid=None, extra={})
    assert not PC.push_keeps_recorded_pre(straight, far, init) and PC.push_keeps_recorded_pre(straight, near, init)
    assert PC.push_keeps_recorded_pre(SimpleNamespace(mid=object(), extra={}), far, init)
    assert PC.push_keeps_recorded_pre(SimpleNamespace(mid=None, extra={"pose_class": "push_blade_45"}), far, init)
    assert not PC.push_keeps_recorded_pre(SimpleNamespace(mid=None, extra={"pose_class": "push_blade_15"}), far, init)


# ------------------------------------------------------------------------------------------------ touching#02: the lowered pre-contact

#: touching#02 c4_k0 (2026-10-12): the raised vertical blade (approach -z, closing along the push) beside the box's +x face, 2.3 cm off
#: it, pushing it 7.85 cm along -x
BLADE_Q = [0.7071074843406677, 0.7071061134338379, 0.0, 0.0]
BLADE_INIT = [-9.185616493225098, -4.6257853507995605, 1.1170214414596558]
BLADE_DIST = 0.0785


def _blade_push(pre_up: float = 0.10, mid=None, pose_class: str = "push_blade_0_high"):
    from ispatialgym.oracle.actions import Push
    from ispatialgym.oracle.sidecar import Pose, PushContact

    x, y, z = BLADE_INIT
    contact = PushContact(arm=ARM, object="box__of__cereal.n.01_1", pre=Pose([x, y, z + pre_up], list(BLADE_Q)), init=Pose([x, y, z], list(BLADE_Q)),
                          goal=Pose([x - BLADE_DIST, y, z], list(BLADE_Q)), direction=[-1.0, 0.0, 0.0], distance_m=BLADE_DIST, mid=mid,
                          extra={"pose_class": pose_class, "blade_standoff_m": 0.023, "palm_clear": True})
    return Push(contact=contact)


def test_a_straight_push_is_lowered_over_its_first_contact_pose_not_its_first_waypoint():
    """c2 / c4 / c7 / c12: the lowered pre-contact stood over the first line waypoint - a third of the push (2.6 cm) toward the box,
    3.5 mm inside its face with the fingertips 5 mm under its top. It lies push_pre_dist_m above the recorded init, and the program
    check reads the same decision from the same pose."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.v2 import program_check as PC

    cfg = EX.ExecuteConfig()
    c = _blade_push().contact
    wps = _blade_push().targets(None, None)["waypoints"]
    assert float(wps[0].position[0]) == pytest.approx(BLADE_INIT[0] - BLADE_DIST / 3, abs=1e-6)  # the line leaves the start out
    p, q = EX.as_world_pose(EX.push_lowered_pre(c, c.pre, c.init, cfg))
    assert th.allclose(p, th.tensor([BLADE_INIT[0], BLADE_INIT[1], BLADE_INIT[2] + cfg.push_pre_dist_m]), atol=1e-5) and th.allclose(q, th.tensor(BLADE_Q))
    assert not PC.push_keeps_recorded_pre(c, c.pre, c.init)
    # kept: an arc, a pitched class, a pre-contact within push_pre_dist_m + 5 mm, the lowering off
    assert EX.push_lowered_pre(_blade_push(mid=c.goal).contact, c.pre, c.init, cfg) is None
    assert EX.push_lowered_pre(_blade_push(pose_class="push_blade_45-").contact, c.pre, c.init, cfg) is None
    near = _blade_push(pre_up=0.045).contact
    assert EX.push_lowered_pre(near, near.pre, near.init, cfg) is None and PC.push_keeps_recorded_pre(near, near.pre, near.init)
    assert EX.push_lowered_pre(c, c.pre, c.init, EX.ExecuteConfig(push_pre_dist_m=None)) is None
    # a fingertip-front push (approach along the push) with a 4 cm pre-contact keeps it; measured from the first waypoint it read 6.6 cm
    tip_q = th.tensor([0.0, -0.7071068, 0.0, 0.7071068])
    assert th.allclose(C.approach_axis(tip_q), th.tensor([-1.0, 0.0, 0.0]), atol=1e-6)
    tip = SimpleNamespace(mid=None, extra={"pose_class": "push_tip_h"})
    tip_init = (th.tensor(BLADE_INIT), tip_q)
    tip_pre = (th.tensor(BLADE_INIT) + th.tensor([0.04, 0.0, 0.0]), tip_q)
    tip_wp0 = (th.tensor(BLADE_INIT) - th.tensor([BLADE_DIST / 3, 0.0, 0.0]), tip_q)
    assert EX.push_lowered_pre(tip, tip_pre, tip_init, cfg) is None and EX.push_lowered_pre(tip, tip_pre, tip_wp0, cfg) is not None


class _PushMotions:
    """The motions surface ``execute_push`` touches with no supports, no protect context and no backend: records the transit
    (``arm_to``) and the Cartesian legs (``arm_along``)."""

    def __init__(self):
        self.cfg = MotionsConfig()
        self.press_hold = None
        self.calls: List[Any] = []

    def gripper(self, arm, close=True, **kw):
        yield th.zeros(3)

    def arm_to(self, arm, pose=None, q=None, ignore_objects=(), q_hint=None, max_excursion=None, ignore_links=(), accept_other_branch=False):
        self.calls.append(("arm_to", EX.as_world_pose(pose), q_hint))
        yield th.zeros(3)

    def arm_along(self, arm, waypoints, **kw):
        self.calls.append(("arm_along", [EX.as_world_pose(w) for w in waypoints]))
        yield th.zeros(3)

    def hold(self, n=1):
        for _ in range(n):
            yield th.zeros(3)


def _run_push(push, witness=None):
    box = SimpleNamespace(name="target_box_of_cereal_iucshm_0", category="box_of_cereal", model="iucshm")
    state = SimpleNamespace(object=lambda name: box, object_pose=lambda name: (th.tensor([-9.235, -4.6258, 1.0069]), th.tensor(QUAT)), pose_target=lambda name: None)
    m = _PushMotions()
    ctx: Dict[str, Any] = {"cfg": EX.ExecuteConfig(push_base_nudge_max_m=0.0), "witnesses": {} if witness is None else {"pre": witness}}
    list(EX.execute_push(push, m, state, push.targets(None, None), ctx))
    return m.calls, ctx


def test_from_a_lowered_pre_contact_the_pad_descends_to_the_first_contact_pose_before_it_pushes():
    """The transit ends push_pre_dist_m above the recorded init (no witness), the push leg goes down to the init and then along the
    line; a recorded pre-contact that is kept is reached with its witness and the leg is the line as before."""
    cfg = EX.ExecuteConfig()
    push = _blade_push()
    line = [EX.as_world_pose(w) for w in push.targets(None, None)["waypoints"]]
    init = EX.as_world_pose(push.contact.init)
    calls, ctx = _run_push(push, witness=_witness())
    (go, goal, hint), (leg, wps) = calls[0], calls[1]
    assert go == "arm_to" and hint is None and th.allclose(goal[0], init[0] + th.tensor([0.0, 0.0, cfg.push_pre_dist_m]), atol=1e-5)
    assert leg == "arm_along" and len(wps) == 1 + len(line) and _same_pose(wps[0], init) and all(_same_pose(a, b) for a, b in zip(wps[1:], line))
    assert ctx["push_pre_lowered"]["init"] == pytest.approx(BLADE_INIT, abs=1e-4) and ctx["push_pre_dist_m"] == cfg.push_pre_dist_m
    calls, ctx = _run_push(_blade_push(pre_up=0.045), witness=_witness())
    (go, goal, hint), (leg, wps) = calls[0], calls[1]
    assert th.allclose(goal[0], init[0] + th.tensor([0.0, 0.0, 0.045]), atol=1e-5) and hint is not None and th.allclose(hint, _witness())
    assert len(wps) == len(line) and all(_same_pose(a, b) for a, b in zip(wps, line)) and "push_pre_lowered" not in ctx


# ------------------------------------------------------------------------------------------------ 2b: the skip-above knob (default off)


def test_the_skip_above_knob_is_off_by_default_and_plans_a_pre_contact_above_its_support_directly():
    assert EX.ExecuteConfig().pre_pose_standoff_skip_above_m is None
    pre, desk, wit = _pre(0.072), _desk(), _witness()
    m = _Motions(track=True)  # off: the standoff as before
    _, ctx = _reach(m, pre, [desk], wit, through=["book"], skip_above=None)
    assert [c["fn"] for c in m.calls] == ["arm_to", "arm_along"] and ctx["pre_pose_standoff"]["branch"] == "witness"
    m = _Motions(track=True)  # 0.06: 7.2 cm above the desk -> the skip branch's call, no standoff IK
    _, ctx = _reach(m, pre, [desk], wit, through=["book"], skip_above=0.06)
    assert [c["fn"] for c in m.calls] == ["arm_to"] and m.calls[0]["q_hint"] is wit and m.calls[0]["ignore_objects"] == ["book"] and m.ik_calls == 0
    assert ctx["pre_pose_standoff"] == {"skipped": True, "clearance_m": pytest.approx(0.072, abs=1e-3), "above_m": pytest.approx(0.072, abs=1e-3)}
    m = _Motions(track=True)  # 5 cm above: below the knob
    _, ctx = _reach(m, _pre(0.05), [desk], wit, through=["book"], skip_above=0.06)
    assert ctx["pre_pose_standoff"]["branch"] == "witness"
    # 7 cm before a wall (beside it, not above): the standoff as before; above the desk with a wall 5 cm beside it: too
    wall = _Support("walls_0", [0.57, 0.0, 0.0], [0.7, 1.0, 2.0])
    m = _Motions(track=True)
    _, ctx = _reach(m, SimpleNamespace(position=[0.5, 0.5, 1.4], orientation=list(QUAT)), [wall], wit, skip_above=0.06)
    assert ctx["pre_pose_standoff"]["branch"] == "witness" and ctx["pre_pose_standoff"]["direction"] == [-1.0, 0.0, 0.0]
    m = _Motions(track=True)
    _, ctx = _reach(m, pre, [desk, _Support("walls_1", [0.55, 0.0, 0.0], [0.7, 1.0, 2.0])], wit, skip_above=0.06)
    assert ctx["pre_pose_standoff"]["branch"] == "witness"
    assert EX.height_above_supports(EX.as_world_pose(pre), [desk]) == pytest.approx(0.072, abs=1e-3)
    assert EX.height_above_supports(EX.as_world_pose(_pre(0.20)), [desk]) is None  # none near


def test_contact_route_goal_and_the_program_check_follow_the_knob(monkeypatch):
    from ispatialgym.sampling.v2 import program_check as PC

    desk, wit = _desk(), _witness()
    state = SimpleNamespace(contact_supports=lambda name: [desk])
    contact = {"name": "push", "object": "paperback_book.n.01_1", "targets": {"pre": _pre(0.072)}, "witnesses": {"pre": wit}}
    m = _Motions(track=True)
    goal, label = EX.contact_route_goal(m, state, ARM, contact, th.zeros(len(M.ALL_JOINTS)), EX.ExecuteConfig(pre_pose_standoff_skip_above_m=0.06))
    assert th.allclose(goal[IDX[ARM]], wit[IDX[ARM]]) and label == "the push's pre witness (0.072 m above its supports)" and m.ik_calls == 0
    ok, _, rep = _pc(_PcSampler())  # off: the decision runs (nothing works in this fake -> rejected)
    assert not ok and "above_m" not in rep
    real = EX.ExecuteConfig
    monkeypatch.setattr(EX, "ExecuteConfig", lambda: real(pre_pose_standoff_skip_above_m=0.06))
    s = _PcSampler()
    ok, why, rep = _pc(s)
    assert ok and rep["branch"] == "skipped" and rep["above_m"] == pytest.approx(0.072, abs=1e-3) and s.calls == [] and "above its supports" in why
