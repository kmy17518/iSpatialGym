"""i7-overpath 2026-11-08 (owner decisions 3(a) / 3(b), inside_large_01): the overhang pick's recorded transit over the book
(``ISPATIALGYM_OVERHANG_OVERPATH``, :mod:`ispatialgym.oracle.overpath`) and the spine guard (``generation.overhang_required_side``).
Simulator-free (``pytest ispatialgym/tests/test_overhang_overpath.py``)."""

from __future__ import annotations

import contextlib
import json
import math
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle import overpath as OP
from ispatialgym.oracle import protection as PROT
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar, SidecarError
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import overhang as OH
from ispatialgym.sampling.v2.ik import IKResult

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")
from ispatialgym.tests.test_iors_large_overhang import _sidecar_with_pick_push  # noqa: E402

CUSTOMS = os.path.join(os.path.dirname(__file__), "..", "assets", "task_specs", "isg_customs.json")
ALIAS_VARIANT = "inter_object_relation_satisfaction-inside_large_scale-hardback-bookcase-desk"
#: hardback bqvvvx: its collision box (geometry.json coll_bbox) and the six v2 overhang grasps' T_object_eef positions (entry
#: hardback__bqvvvx__s1__overhang_yn_29mm_6c7148, side -y)
HB_LO, HB_HI = (-0.11592, -0.08614, -0.01524), (0.11592, 0.08312, 0.01524)
HB_GRASPS = {"side_0": (-0.042727, -0.075256, 0.00017), "side_2": (0.040297, -0.074126, 0.000743), "side_6": (0.106792, -0.069593, 0.000623),
             "side_7": (-0.100614, -0.069568, 0.001268), "side_8": (0.090728, -0.069176, 0.001087), "side_9": (-0.058227, -0.074705, 0.000231)}


def _hb_hull() -> th.Tensor:
    return th.tensor([[x, y, z] for x in (HB_LO[0], HB_HI[0]) for y in (HB_LO[1], HB_HI[1]) for z in (HB_LO[2], HB_HI[2])])


# ------------------------------------------------------------------------------------------------ 3(a) the pure path helpers


def test_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv(OP.ENV, raising=False)
    assert OP.on() is False
    monkeypatch.setenv(OP.ENV, "1")
    assert OP.on() is True
    monkeypatch.setenv(OP.ENV, "0")
    assert OP.on() is False


def test_leg_poses_rise_across_descend():
    q0, q1 = th.tensor([0.0, 0.0, 0.0, 1.0]), th.tensor([0.0, 0.7071, 0.0, 0.7071])
    legs = OP.leg_poses(th.tensor([1.0, 0.0, 0.90]), q0, th.tensor([1.3, 0.1, 0.80]), q1, 0.95)
    assert len(legs) == 4
    (p_s, _), (p_up, o_up), (p_over, o_over), (p_pre, o_pre) = legs
    assert p_up.tolist() == pytest.approx([1.0, 0.0, 0.95]) and th.allclose(o_up, q0)  # straight up, the start's orientation
    assert p_over.tolist() == pytest.approx([1.3, 0.1, 0.95]) and th.allclose(o_over, q1)  # across at the height, turned to the pre-grasp's
    assert p_pre.tolist() == pytest.approx([1.3, 0.1, 0.80])
    # a start above the across height: no rise, the across leg stays at the start's height
    legs = OP.leg_poses(th.tensor([1.0, 0.0, 1.00]), q0, th.tensor([1.3, 0.1, 0.80]), q1, 0.95)
    assert len(legs) == 3 and float(legs[1][0][2]) == pytest.approx(1.00)


def test_densify_and_subsample():
    q0 = th.zeros(7)
    wps = [th.tensor([0.3, 0, 0, 0, 0, 0, 0.0]), th.tensor([0.3, 0.4, 0, 0, 0, 0, 0.0])]
    d = OP.densify(q0, wps, step=0.02)
    assert th.allclose(d[0], q0) and th.allclose(d[-1], wps[-1])
    assert float((d[1:] - d[:-1]).abs().max()) <= 0.02 + 1e-6  # the steepest step of the cosine profile
    assert float((d[1:] - d[:-1]).abs().max(dim=1).values[0]) < 0.005  # zero speed at the start
    # the path passes through the corner (piecewise linear, no shortcut)
    assert any(float((r - wps[0]).abs().max()) < 0.02 for r in d)
    traj = th.stack([th.full((7,), 0.01 * k) for k in range(31)])
    keep = OP.subsample(traj, 0.05)
    assert keep[0] == 0 and keep[-1] == 30 and all(b - a >= 5 for a, b in zip(keep[:-2], keep[1:-1]))


def test_record_and_parse():
    rec = OP.record("right", "manipulation", "escape", [th.zeros(7), th.ones(7)], [th.zeros(3), th.ones(3)], clearance_m=0.08)
    json.dumps(rec)
    assert rec["schema"] == OP.SCHEMA and rec["arm"] == "right" and rec["clearance_m"] == 0.08 and len(rec["q"]) == 2
    qs = OP.parse(rec)
    assert len(qs) == 2 and th.allclose(qs[1], th.ones(7))
    for bad in ({}, {**rec, "schema": "x"}, {**rec, "q": [[0.0] * 7]}, {**rec, "q": [[0.0] * 6, [0.0] * 7]}):
        with pytest.raises(ValueError):
            OP.parse(bad)


# ------------------------------------------------------------------------------------------------ 3(a) the sampler's plan on stubs
#
# a toy arm: joints 0..2 ARE the eef position (FK = q[0:3]), one 2 cm sphere on the right gripper at the eef - the executed joint path is
# then the executed eef path, and the book clearance is the sphere's distance to the book's box minus 2 cm


class _ToyKit:
    def __init__(self):
        self.robot = SimpleNamespace(arm_control_idx={"right": th.arange(7), "left": th.arange(7, 14)})
        self.stats = {}
        self.backend = SimpleNamespace(robot_spheres_world=self._spheres)
        self.collide = lambda qs: th.zeros(len(qs), dtype=th.bool)

    @staticmethod
    def _spheres(q, emb="whole_body"):
        q = th.as_tensor(q).reshape(-1, 28)
        sp = th.zeros(len(q), 2, 4)
        sp[:, 0, :3], sp[:, 0, 3] = q[:, 0:3], 0.02
        sp[:, 1, :3], sp[:, 1, 3] = th.tensor([0.0, 0.0, 0.5]), 0.1  # the body: never part of the moving arm
        return sp, ["right_gripper_link", "torso_link4"]

    def fk(self, frames, q):
        q = th.as_tensor(q).reshape(-1)
        return {f: (q[0:3].clone(), th.tensor([0.0, 0.0, 0.0, 1.0])) for f in frames}

    def collision_free(self, qs, margin=0.0):
        qs = th.as_tensor(qs).reshape(-1, 28)
        return ~self.collide(qs)

    def ik(self, pose, q_seed, arm="right"):
        q = th.as_tensor(q_seed).clone().float().reshape(-1)
        q[0:3] = th.as_tensor(pose[0]).float()
        return q


class _ToyIKC:
    ORACLE_SOFT_POS_TOL, ORACLE_SOFT_ROT_TOL, ORACLE_DIFF_IK_MAX_ITERS = 0.01, 0.03, 100

    def __init__(self, shortcut=False):
        self.shortcut = shortcut  # True: "tracks" straight from the start to the pre-grasp (through the book)
        self.calls = []

    @contextlib.contextmanager
    def ignoring(self, objs, **kw):
        yield

    def track_chain(self, q_from, legs, arm, **kw):
        self.calls.append([p.tolist() for p, _ in legs])
        if getattr(self, "stall", False):
            return None, "approach_untrackable: chain tracking (65 dense poses): differential IK did not converge at waypoint 9/65 at_lower=['right_arm_joint4']"
        pts = [legs[0][0], legs[-1][0]] if self.shortcut else [p for p, _ in legs]
        rows = []
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(1, int(math.ceil(float(th.linalg.norm(b - a)) / 0.01)))
            for k in range(n):
                r = th.as_tensor(q_from).clone().float()
                r[0:3] = a + (b - a) * (k / n)
                rows.append(r)
        r = th.as_tensor(q_from).clone().float()
        r[0:3] = pts[-1]
        rows.append(r)
        self.last_track_traj = th.stack(rows)
        return rows[-1].clone(), "ok"

    def reaches(self, q, pose, arm):
        return float(th.linalg.norm(th.as_tensor(q).reshape(-1)[0:3] - th.as_tensor(pose[0]).float())) <= 0.005

    def continue_chain(self, q, poses, arm, **kw):
        return IKResult(True, q, "tracked continuation", len(poses))

    def _collision_blame(self, q, ignored):
        return "toy"


def _toy_sampler(shortcut=False):
    s = SimpleNamespace(kit=_ToyKit(), ikc=_ToyIKC(shortcut), target=SimpleNamespace(name="hardback_0"), gp=SimpleNamespace(variant="toy", gen=lambda k, d=None: d),
                        hull=_hb_hull())
    return s


def _bind(s):
    for name in ("_overpath_configs", "_overpath_plan_one", "_overpath_where", "_overpath_where_track", "_overpath_limit"):
        if not hasattr(s, name):
            setattr(s, name, getattr(IL.IORSLarge, name).__get__(s))


def _plan(s, q_start, book_pose, pre_pos):
    s._overpath_start = lambda st, gst, arm, chain, settled: (q_start, "escape", "")
    _bind(s)
    pre = (th.tensor(pre_pos), th.tensor([0.0, 0.0, 0.0, 1.0]))
    st = SimpleNamespace(name="manipulation")
    return IL.IORSLarge._overpath_plan_cartesian(s, st, st, "right", {}, book_pose, pre, pre)


def test_overpath_clears_the_book_by_its_margin():
    """The book (hardback box, top at z 0.7824) between the escape's end (beyond its far side, 15 cm up) and the pre-grasp in front of
    its near edge: the recorded path rises to the top + 0.08, crosses and descends; the moving spheres keep >= BOOK_MARGIN_M from the box."""
    book = (th.tensor([0.0, 0.0, 0.7671 + 0.0153]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    q0 = th.zeros(28)
    q0[0:3] = th.tensor([0.0, 0.20, 0.93])  # the escape's end: beyond the far (+y) side, 15 cm over the desk
    rec, key, why = _plan(_toy_sampler(), q0, book, [0.0, -0.16, 0.79])  # the pre-grasp 7 cm in front of the -y (spine) face
    assert rec is not None, (key, why)
    assert rec["clearance_m"] == OP.CLEARANCES_M[0] and rec["z_across"] == pytest.approx(0.7824 + 0.0152 + 0.08, abs=2e-3)
    assert rec["min_book_clearance_m"] >= OP.BOOK_MARGIN_M and rec["start"] == "escape" and rec["n_waypoints"] >= 3
    # the path the oracle runs through the recorded waypoints clears the box too
    qs = OP.parse(rec)
    full = []
    for q in qs:
        r = th.zeros(28)
        r[0:7] = q
        full.append(r)
    dense = OP.densify(full[0], full[1:])
    val, _ = PROT.sphere_box_clearance(_ToyKit._spheres(dense)[0][:, :1], OP.box_of(book, _hb_hull()))
    assert float(val[1:-1].min()) >= OP.BOOK_MARGIN_M - 1e-4
    assert th.allclose(qs[-1][0:3], th.tensor([0.0, -0.16, 0.79]), atol=1e-4)  # it ends at the pre-grasp


def test_overpath_falls_back_to_seeded_ik_when_the_tracking_stalls():
    """i7op_b (first run): the oracle-tolerance tracking from the escape's end stalled with the elbow at its limit; the exact IK of every
    ~2 cm waypoint seeded at the previous solution takes over (kit overpath.track_stalled), the checks unchanged."""
    book = (th.tensor([0.0, 0.0, 0.7671 + 0.0153]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    q0 = th.zeros(28)
    q0[0:3] = th.tensor([0.0, 0.20, 0.93])
    s = _toy_sampler()
    s.ikc.stall = True
    rec, key, why = _plan(s, q0, book, [0.0, -0.16, 0.79])
    assert rec is not None, (key, why)
    assert rec["source"] == "seeded IK" and s.kit.stats.get("overpath.track_stalled") == len([1]) and rec["min_book_clearance_m"] >= OP.BOOK_MARGIN_M
    assert th.allclose(OP.parse(rec)[-1][0:3], th.tensor([0.0, -0.16, 0.79]), atol=1e-4)
    # a branch flip of the seeded IK is a rejection
    s2 = _toy_sampler()
    s2.ikc.stall = True
    s2.kit.ik = lambda pose, q_seed, arm="right": th.full((28,), 2.0)
    rec, key, why = _plan(s2, q0, book, [0.0, -0.16, 0.79])
    assert rec is None and key == "jump" and "joint jump" in why


def test_overpath_rejects_a_path_through_the_book():
    book = (th.tensor([0.0, 0.0, 0.7671 + 0.0153]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    q0 = th.zeros(28)
    q0[0:3] = th.tensor([0.0, 0.20, 0.80])
    rec, key, why = _plan(_toy_sampler(shortcut=True), q0, book, [0.0, -0.16, 0.79])
    assert rec is None and key == "book" and "passes the book" in why and len(_toy_sampler().ikc.calls) == 0
    # a collision of the executed path is a rejection too
    s = _toy_sampler()
    s.kit.collide = lambda qs: qs[:, 2] > 0.86
    rec, key, why = _plan(s, q0, book, [0.0, -0.16, 0.79])
    assert rec is None and key == "collision" and len(s.ikc.calls) == len(OP.CLEARANCES_M) * len(OP.ORIENT_SCHEDULES)


def test_switch_off_is_today(monkeypatch):
    """Off: no record (``overpath_record`` is a no-op, nothing planned), the failed-pick memo key as before; on, the key says so."""
    monkeypatch.delenv(OP.ENV, raising=False)
    s = SimpleNamespace(kit=SimpleNamespace(stats={}), _overpath_plan=lambda *a, **k: pytest.fail("planned with the switch off"))
    g = SimpleNamespace(extras={}, arm="right", pre_grasp=None)
    IL.IORSLarge.overpath_record(s, {"overpath": None}, g)
    assert g.extras == {} and s.kit.stats == {}
    from ispatialgym.tests.test_iors_large_overhang import _GP, _manip, Z_TOP

    m = _manip()
    args = (m, {"right": m}, ["right"], SimpleNamespace(name="side_0"), 4.32, th.tensor([1.0, 0.0]), Z_TOP, False)
    sm = SimpleNamespace(gp=_GP(overhang_fail_memo=True))
    k_off = IL.IORSLarge._overhang_memo_key(sm, *args)
    assert k_off[-1] is False
    monkeypatch.setenv(OP.ENV, "1")
    assert IL.IORSLarge._overhang_memo_key(sm, *args) == k_off + ("overpath", OP.MEMO_VERSION)
    # the source: the plan runs only behind the switch, in the pick and in the record
    import inspect

    src = inspect.getsource(IL.IORSLarge._overhang_pick)
    assert "if OPATH.on():" in src and "self._overpath_plan(" in src


def test_overpath_record_keeps_or_replans(monkeypatch):
    monkeypatch.setenv(OP.ENV, "1")
    pre = (th.tensor([1.0, 0.0, 0.8]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    rec = OP.record("right", "manipulation", "escape", [th.zeros(7), th.ones(7)], [th.zeros(3), th.ones(3)],
                    pre_grasp={"position": pre[0].tolist(), "orientation": pre[1].tolist()})
    calls = []
    s = SimpleNamespace(kit=SimpleNamespace(stats={}), _overpath_plan=lambda *a: calls.append(a) or (None, "book", "passes the book"))
    ov = {"overpath": rec, "stance": "B", "grasp_stance": "B", "chain": {}, "post_pose": None}
    g = SimpleNamespace(extras={}, arm="right", pre_grasp=pre, grasp=pre, grasp_name="side_0")
    IL.IORSLarge.overpath_record(s, ov, g)
    assert g.extras["overpath"]["q"] == rec["q"] and calls == [] and s.kit.stats == {"overpath.recorded": 1}
    g2 = SimpleNamespace(extras={}, arm="right", pre_grasp=(pre[0] + 0.05, pre[1]), grasp=pre, grasp_name="side_0~r180")  # the twin
    with pytest.raises(Reject, match="overpath to the recorded pre-grasp"):
        IL.IORSLarge.overpath_record(s, ov, g2)
    assert len(calls) == 1 and s.kit.stats["overpath.rejected.book"] == 1 and "overpath" not in g2.extras


# ------------------------------------------------------------------------------------------------ 3(a) the program, the program check, the oracle


def _sidecar_with_overpath(**rec_over):
    d = _sidecar_with_pick_push()
    g = d["grasps"][0]
    rec = OP.record(g["arm"], "manipulation", "escape", [th.zeros(7), th.full((7,), 0.1), th.full((7,), 0.2)], [th.zeros(3)] * 3, clearance_m=0.08)
    rec.update(rec_over)
    g["overpath"] = rec
    return d


def test_program_expands_with_the_recorded_path():
    sc = Sidecar.from_dict(_sidecar_with_overpath())
    actions = expand(sc.variant, sc)
    assert [a.name for a in actions][:4] == ["transition", "push", "retreat", "grasp"]
    grasp = actions[3]
    assert grasp.overpath is not None and grasp.overpath["clearance_m"] == 0.08 and len(grasp.overpath["q"]) == 3
    # without the record: the planned transit (today)
    sc0 = Sidecar.from_dict(_sidecar_with_pick_push())
    assert next(a for a in expand(sc0.variant, sc0) if a.name == "grasp").overpath is None
    for bad, match in (({"arm": "left"}, "arm"), ({"q": [[0.0] * 7]}, "q"), ({"schema": "x"}, "isg-overpath")):
        with pytest.raises(SidecarError, match=match):
            expand(sc.variant, Sidecar.from_dict(_sidecar_with_overpath(**bad)))


class _PCKit(_ToyKit):
    pass


def _pc_sampler(collide=None):
    kit = _PCKit()
    if collide is not None:
        kit.collide = collide

    class _IKC:
        def q_locked(self, stance, fingers=None, **kw):
            return th.zeros(28)

        def reaches(self, q, pose, arm):
            return float(th.linalg.norm(th.as_tensor(q)[0:3] - pose[0])) <= 0.005

        def _collision_blame(self, q, ign):
            return "toy"

    return SimpleNamespace(kit=kit, ikc=_IKC())


def test_program_check_replays_the_recorded_path():
    from ispatialgym.sampling.v2 import program_check as PC

    a, b, c = th.zeros(7), th.tensor([0.0, 0.0, 0.2, 0, 0, 0, 0]), th.tensor([0.0, -0.3, 0.2, 0, 0, 0, 0])
    rec = OP.record("right", "manipulation", "escape", [a, b, c], [a[:3], b[:3], c[:3]])
    pre = (c[:3].clone(), th.tensor([0.0, 0.0, 0.0, 1.0]))
    q_end, note = PC.check_overpath(_pc_sampler(), SimpleNamespace(name="manipulation"), "right", rec, pre, th.full((28,), 0.01))
    assert th.allclose(q_end[0:3], c[:3], atol=1e-5) and note["from"] == "where the previous action left the arm" and note["waypoints"] == 3
    # the arm farther than START_GAP_MAX_RAD from the record's start: not followed (the oracle plans the transit, checked as before)
    q_none, note = PC.check_overpath(_pc_sampler(), SimpleNamespace(name="m"), "right", rec, pre, th.full((28,), 0.5))
    assert q_none is None and note["followed"] is False and "rad off" in note["why"]
    with pytest.raises(Reject, match="collides"):
        PC.check_overpath(_pc_sampler(collide=lambda qs: qs[:, 2] > 0.1), SimpleNamespace(name="m"), "right", rec, pre, None)
    with pytest.raises(Reject, match="does not end at this grasp's pre-grasp"):
        PC.check_overpath(_pc_sampler(), SimpleNamespace(name="m"), "right", rec, (th.tensor([1.0, 1.0, 1.0]), pre[1]), None)
    # the book in the way: the oracle's protected-object rule (INFLATION_M)
    book = SimpleNamespace(name="hardback_0", get_base_aligned_bbox=lambda visual=False: (th.tensor([0.0, -0.15, 0.1]), th.tensor([0.0, 0.0, 0.0, 1.0]),
                                                                                         th.tensor([0.1, 0.1, 0.3]), None))
    with pytest.raises(Reject, match="passes hardback_0"):
        PC.check_overpath(_pc_sampler(), SimpleNamespace(name="m"), "right", rec, pre, None, obj=book)
    # the hook: the grasp's pre-grasp witness is the replayed path's end
    import inspect

    src = inspect.getsource(PC.check_action_targets)
    assert "check_overpath(" in src and 'IKResult(True, overpath_q.clone(), "overpath end", 1)' in src


class _FollowMotions:
    def __init__(self, traj_ok=True):
        self.robot = SimpleNamespace(arm_control_idx={"right": list(range(7))})
        self.ran, self.ok = [], traj_ok
        self.q = th.zeros(28)

    def recorded_joint_path(self, arm, qs):
        if not self.ok:
            return None, {"why": "passes hardback_0 at 0.004 m", "start_gap_rad": 0.01}
        self.target = qs[-1]
        return th.zeros(5, 28), {"n_steps": 4, "start_gap_rad": 0.01}

    def execute_joint_trajectory(self, traj, arms):
        self.ran.append(len(traj))
        self.q[0:7] = self.target
        yield th.zeros(23)

    def q_full(self):
        return self.q

    def eef_pose(self, arm):
        return th.tensor([1.0, 0.0, 0.8]), th.tensor([0.0, 0.0, 0.0, 1.0])


def _drive(gen):
    out = []
    while True:
        try:
            out.append(next(gen))
        except StopIteration as e:
            return e.value, out


def test_oracle_follows_the_recorded_path_or_falls_back():
    rec = OP.record("right", "manipulation", "escape", [th.zeros(7), th.full((7,), 0.2)], [th.zeros(3)] * 2, clearance_m=0.08, min_book_clearance_m=0.05)
    action = SimpleNamespace(overpath=rec, object="hardback.n.01_1")
    targets = {"pre_grasp": (th.tensor([1.0, 0.0, 0.803]), th.tensor([0.0, 0.0, 0.0, 1.0]))}
    m, ctx = _FollowMotions(), {}
    ok, steps = _drive(EX.follow_overpath(action, m, targets, ctx, "right"))
    assert ok is True and m.ran == [5] and len(steps) == 1
    note = ctx["overpath"]
    assert note["followed"] is True and note["joint_err_rad"] == pytest.approx(0.0) and note["pre_grasp_err_m"] == pytest.approx(0.003, abs=1e-4)
    assert ctx["pre_grasp_transit"] == "recorded overpath"
    m2, ctx2 = _FollowMotions(traj_ok=False), {}
    ok, steps = _drive(EX.follow_overpath(action, m2, targets, ctx2, "right"))
    assert ok is False and steps == [] and m2.ran == [] and "hardback_0" in ctx2["overpath"]["fallback"]
    # the grasp only follows a record (``Grasp.overpath``); the planned transit runs otherwise
    import inspect

    src = inspect.getsource(EX.execute_grasp)
    assert 'if getattr(action, "overpath", None) is not None:' in src and "if not followed:" in src


def test_motions_recorded_joint_path_checks_like_the_straight_ramp():
    from ispatialgym.oracle.motions import Motions

    class _B:
        def __init__(self, hit=False):
            self.hit = hit

        def update_world(self, ignore_objects=()):
            pass

        def q_full(self):
            return th.zeros(28)

        def check_collision(self, traj, emb_name=None, sync_world=False, margin=0.0):
            h = th.zeros(len(traj), dtype=th.bool)
            if self.hit:
                h[len(traj) // 2] = True
            return h

    m = SimpleNamespace(backend=_B(), robot=SimpleNamespace(arm_control_idx={"right": list(range(7))}), cfg=SimpleNamespace(trajectory_max_steps=2000),
                        _protect=None, _emb_for=lambda arms: "right_arm")
    qs = [th.zeros(7), th.full((7,), 0.3)]
    traj, note = Motions.recorded_joint_path(m, "right", qs)
    assert traj is not None and th.allclose(traj[-1][0:7], qs[-1]) and note["start_gap_rad"] == 0.0 and float((traj[1:] - traj[:-1]).abs().max()) <= OP.STEP_RAD + 1e-6
    m.backend = _B(hit=True)
    traj, note = Motions.recorded_joint_path(m, "right", qs)
    assert traj is None and "penetrates" in note["why"]
    m.backend = _B()
    traj, note = Motions.recorded_joint_path(m, "right", [th.full((7,), 0.5), th.ones(7)])
    assert traj is None and "from the recorded start" in note["why"]


# ------------------------------------------------------------------------------------------------ 3(b) the spine guard


def test_spine_guard_pure():
    hull = _hb_hull()
    assert OH.spine_entry_why("-y", "-y") is None
    assert "not the required -y" in OH.spine_entry_why("-y", "+y") and "records no side" in OH.spine_entry_why("-y", None)
    for name, p in HB_GRASPS.items():  # every bqvvvx overhang grasp pinches the spine (1.1 - 1.7 cm inside the -y face)
        assert OH.spine_grasp_why("-y", name, th.tensor(p), hull) is None
        assert 0.010 <= OH.pinch_depth(th.tensor(p), hull, "-y") <= 0.018
    page = (0.04, 0.0729, 0.0)  # a +y pinch: the page block
    assert "inside the -y face" in OH.spine_grasp_why("-y", "page", th.tensor(page), hull)
    assert OH.spine_grasp_why("+y", "page", th.tensor(page), hull) is None
    with pytest.raises(ValueError):
        OH.side_axis("y")


def _guard_sampler(side="-y", **gen):
    g = {"overhang_required_side": "-y", **gen}
    return SimpleNamespace(gp=SimpleNamespace(variant=ALIAS_VARIANT, gen=lambda k, d=None: g.get(k, d)), kit=SimpleNamespace(stats={}), hull=_hb_hull(),
                           v2_overhang={"side": side, "entry": f"hardback__bqvvvx__s1__overhang_{'yn' if side == '-y' else 'yp'}_29mm"})


def test_spine_guard_passes_a_minus_y_entry_with_spine_pinches():
    s = _guard_sampler()
    IL.overhang_spine_entry(s)
    for name, p in HB_GRASPS.items():
        IL.overhang_spine_grasp(s, SimpleNamespace(name=name, pose=(th.tensor(p), th.tensor([0.0, 0.0, 0.0, 1.0]))))
    assert s.kit.stats == {"overhang.side_checked.entry": 1, "overhang.side_checked.grasp": len(HB_GRASPS)}


def test_spine_guard_rejects_a_plus_y_entry_or_a_grasp_off_the_spine():
    s = _guard_sampler(side="+y")
    with pytest.raises(Reject, match="not the required -y") as e:
        IL.overhang_spine_entry(s)
    assert e.value.reason == "overhang_side" and s.kit.stats["overhang.side_rejected"] == 1 and s.kit.stats["overhang.side_rejected.entry"] == 1
    s = _guard_sampler()
    with pytest.raises(Reject, match="inside the -y face"):
        IL.overhang_spine_grasp(s, SimpleNamespace(name="side_4", pose=(th.tensor([0.04, 0.0729, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))))
    assert s.kit.stats["overhang.side_rejected.grasp"] == 1
    # off (no generation.overhang_required_side) or no v2 overhang entry: nothing checked
    off = _guard_sampler(side="+y")
    off.gp.gen = lambda k, d=None: d
    IL.overhang_spine_entry(off)
    IL.overhang_spine_grasp(off, SimpleNamespace(name="side_4", pose=(th.tensor([0.04, 0.0729, 0.0]), None)))
    none = _guard_sampler()
    none.v2_overhang = None
    IL.overhang_spine_grasp(none, SimpleNamespace(name="side_4", pose=(th.tensor([0.04, 0.0729, 0.0]), None)))
    assert off.kit.stats == {} and none.kit.stats == {}


def test_spine_guard_is_on_for_inside_large_01_only():
    d = json.load(open(CUSTOMS))
    g = d[ALIAS_VARIANT]["generation"]
    assert g.get("overhang_required_side") == "-y" and g.get("overhang_pinch_edge_max_m") == 0.03
    others = [k for k, v in d.items() if isinstance(v, dict) and k != ALIAS_VARIANT and (v.get("generation") or {}).get("overhang_required_side") is not None]
    assert others == []


# ------------------------------------------------------------------------------------------------ CORRECTION 2026-11-08: the turn schedules


def test_memo_key_carries_the_transit_version(monkeypatch):
    """i7op2_b skipped 28 picks remembered from the buggy i7op_b under the same key: the transit's version is in the key now (and a
    schedule override, when set); off, the key is unchanged."""
    from ispatialgym.tests.test_iors_large_overhang import _GP, _manip, Z_TOP

    m = _manip()
    args = (m, {"right": m}, ["right"], SimpleNamespace(name="side_0"), 4.32, th.tensor([1.0, 0.0]), Z_TOP, False)
    sm = SimpleNamespace(gp=_GP(overhang_fail_memo=True))
    monkeypatch.delenv(OP.ENV, raising=False)
    monkeypatch.delenv(OP.ORIENT_ENV, raising=False)
    k_off = IL.IORSLarge._overhang_memo_key(sm, *args)
    monkeypatch.setenv(OP.ENV, "1")
    k_on = IL.IORSLarge._overhang_memo_key(sm, *args)
    # P1 (owner 2026-11-08 item 4): "overpath/3" - no failure of the Cartesian designs ("overpath", "overpath/2") applies
    assert k_on == k_off + ("overpath", "overpath/3") and k_on not in (k_off + ("overpath",), k_off + ("overpath", "overpath/2"))
    monkeypatch.setenv(OP.ORIENT_ENV, "slerp")
    assert IL.IORSLarge._overhang_memo_key(sm, *args) == k_on  # the schedules belong to the Cartesian design only
    monkeypatch.setenv(OP.DESIGN_ENV, "cartesian")
    assert IL.IORSLarge._overhang_memo_key(sm, *args) == k_on + ("design=cartesian", "orient=slerp")
    monkeypatch.delenv(OP.DESIGN_ENV)
    assert OP.orient_schedules() == ("slerp",)
    monkeypatch.setenv(OP.ORIENT_ENV, "nonsense")
    assert OP.orient_schedules() == OP.ORIENT_SCHEDULES


def test_leg_plan_turn_schedules():
    q0, q1 = th.tensor([0.0, 0.0, 0.0, 1.0]), th.tensor([0.0, 0.7071, 0.0, 0.7071])
    p0, p1 = th.tensor([1.0, 0.0, 0.90]), th.tensor([1.3, 0.1, 0.80])
    over = OP.leg_plan(p0, q0, p1, q1, 0.95, "over_pre")
    assert [lb for _, _, lb in over] == ["start", "rise", "across", "turn", "down"]
    assert th.allclose(over[2][1], q0) and th.allclose(over[3][1], q1) and th.allclose(over[3][0], over[2][0])  # across unturned, turn in place
    assert over[3][0].tolist() == pytest.approx([1.3, 0.1, 0.95]) and th.allclose(over[4][1], q1)  # the descent in the pre-grasp's orientation
    start = OP.leg_plan(p0, q0, p1, q1, 0.95, "at_start")
    assert [lb for _, _, lb in start] == ["start", "turn", "rise", "across", "down"]
    assert th.allclose(start[1][0], p0) and all(th.allclose(q, q1) for _, q, _ in start[1:])
    sl = OP.leg_plan(p0, q0, p1, q1, 0.95, "slerp")
    assert [lb for _, _, lb in sl] == ["start", "rise", "across", "down"]
    assert all(th.allclose(a[0], b[0]) and th.allclose(a[1], b[1]) for a, b in zip(OP.leg_poses(p0, q0, p1, q1, 0.95), sl))
    # no rise when the start is above the across height
    assert [lb for _, _, lb in OP.leg_plan(th.tensor([1.0, 0.0, 1.0]), q0, p1, q1, 0.95, "over_pre")] == ["start", "across", "turn", "down"]
    with pytest.raises(ValueError):
        OP.leg_plan(p0, q0, p1, q1, 0.95, "sideways")
    assert OP.leg_of([3, 5, 2], ["start", "rise", "across", "down"], 4) == ("across", 0.2)
    assert OP.leg_of([3, 5, 2], ["start", "rise", "across", "down"], 10) == ("down", 1.0)


def test_plan_tries_the_turn_schedules_in_order_and_names_where_the_ik_fails(monkeypatch):
    """The first schedule (turn over the pre-grasp) fails where the seeded IK finds no solution mid-across: the reason names the leg, the
    waypoint, the height and whether any IK branch reaches it; the next schedule plans and the record says which one."""
    monkeypatch.delenv(OP.ORIENT_ENV, raising=False)
    monkeypatch.delenv(OP.TRY_ALL_ENV, raising=False)
    book = (th.tensor([0.0, 0.0, 0.7671 + 0.0153]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    q0 = th.zeros(28)
    q0[0:3] = th.tensor([0.0, 0.20, 0.93])
    s = _toy_sampler()
    s.ikc.stall = True
    calls = {"n": 0}

    def plan_one(*a):
        calls["n"] += 1
        orient = a[11]
        if orient == "over_pre":
            # a seeded IK with no solution on the across leg below y 0.05
            s.kit.ik = lambda pose, q_seed, arm="right": None if float(pose[0][1]) < 0.05 and float(pose[0][2]) > 0.9 else _ToyKit.ik(s.kit, pose, q_seed, arm)
            s.kit.ik_all = lambda pose, q_seed, arm="right": None
        else:
            s.kit.ik = lambda pose, q_seed, arm="right": _ToyKit.ik(s.kit, pose, q_seed, arm)
        return IL.IORSLarge._overpath_plan_one(s, *a)

    s._overpath_plan_one = plan_one
    rec, key, why = _plan(s, q0, book, [0.0, -0.16, 0.79])
    assert rec is not None and rec["orient"] == "at_start" and calls["n"] == 2
    assert rec["orients_failed"] and rec["orients_failed"][0].startswith("over_pre: track")
    assert s.kit.stats.get("overpath.orient.over_pre.rejected") == 1 and s.kit.stats.get("overpath.orient.at_start.planned") == 1
    # every schedule failing: one rejection naming each schedule's failing leg / waypoint
    s2 = _toy_sampler()
    s2.ikc.stall = True
    s2.kit.ik = lambda pose, q_seed, arm="right": None if float(pose[0][1]) < 0.05 and float(pose[0][2]) > 0.9 else _ToyKit.ik(s2.kit, pose, q_seed, arm)
    s2.kit.ik_all = lambda pose, q_seed, arm="right": th.zeros(2, 28)
    rec, key, why = _plan(s2, q0, book, [0.0, -0.16, 0.79])
    assert rec is None and key == "track"
    for orient in OP.ORIENT_SCHEDULES:
        assert f"{orient}: track:" in why
    assert "seeded IK: no solution at waypoint" in why and "across leg at" in why and "z 0.930" in why and "2 IK branches reach it" in why


def test_try_all_plans_every_schedule(monkeypatch):
    monkeypatch.delenv(OP.ORIENT_ENV, raising=False)
    monkeypatch.setenv(OP.TRY_ALL_ENV, "1")
    book = (th.tensor([0.0, 0.0, 0.7671 + 0.0153]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    q0 = th.zeros(28)
    q0[0:3] = th.tensor([0.0, 0.20, 0.93])
    s = _toy_sampler()
    rec, key, why = _plan(s, q0, book, [0.0, -0.16, 0.79])
    assert rec is not None and rec["orient"] == OP.ORIENT_SCHEDULES[0]
    assert all(s.kit.stats.get(f"overpath.orient.{o}.planned") == 1 for o in OP.ORIENT_SCHEDULES)


# ------------------------------------------------------------------------------------------------ P1 (owner 2026-11-08 item 4): the oracle's planner
#
# the toy arm again (joints 0..2 = the eef position, one 2 cm sphere at the eef), now under the oracle's own Motions code: a toy cuRobo
# backend whose first joint-space plan runs straight (low over the book: inside its 3 cm margin, not touching) and whose plans after
# that run straight too - the oracle's protected-object gate then takes its via-above (lift, plan to above the goal, straight down)


@dataclass
class _PlanRes:
    success: bool
    q_traj: Optional[th.Tensor] = None
    status: str = "ok"
    plan_time: float = 0.0
    diagnostics: dict = field(default_factory=dict)

    @property
    def duration(self):
        return 0.0 if self.q_traj is None else (len(self.q_traj) - 1) / 30.0


BOOK_P1 = (th.tensor([0.0, 0.0, 0.7671 + 0.0153]), th.tensor([0.0, 0.0, 0.0, 1.0]))


class _ToyCurobo:
    control_dt = 1.0 / 30.0

    def __init__(self, box):
        self.box = box
        self.calls: List[str] = []

    def _sp(self, q):
        return _ToyKit._spheres(q)

    def robot_spheres_world(self, q, emb="whole_body"):
        return self._sp(q)

    def check_collision(self, q, emb_name="whole_body", sync_world=False, margin=0.0, **kw):
        sp, _ = self._sp(th.as_tensor(q).reshape(-1, 28))
        val, _ = PROT.sphere_box_clearance(sp[:, :1], self.box)
        return val < float(margin)

    def update_world(self, ignore_objects=(), ignore_links=(), **kw):
        self.calls.append("update_world")

    def fk(self, frames, q_full=None, emb_name="whole_body"):
        q = th.as_tensor(q_full).reshape(-1)
        return {f: (q[0:3].clone(), th.tensor([0.0, 0.0, 0.0, 1.0])) for f in frames}

    def fk_batch(self, frame, q, emb_name="whole_body"):
        q = th.as_tensor(q).reshape(-1, 28)
        return q[:, 0:3].clone(), th.tensor([0.0, 0.0, 0.0, 1.0]).repeat(len(q), 1)

    def solve_ik(self, emb, targets, q_seed=None, sync_world=True, **kw):
        q = th.as_tensor(q_seed).clone().float().reshape(-1)
        q[0:3] = th.as_tensor(list(targets.values())[0][0]).float()
        return q

    def solve_ik_path(self, emb, waypoints, q_start=None, check_collisions=True, **kw):
        q = th.as_tensor(q_start).clone().float().reshape(-1)
        rows = [q.clone()]
        for wp in waypoints:
            q = q.clone()
            q[0:3] = th.as_tensor(list(wp.values())[0][0]).float()
            rows.append(q)
        traj = th.stack(rows)
        if check_collisions and bool(self.check_collision(traj[1:]).any()):
            return _PlanRes(False, traj[:1], status="collision (toy)", diagnostics={"stage": "collision", "converged": 0})
        return _PlanRes(True, traj)

    def _line(self, q0, q1, n=40):
        return th.stack([q0 + (q1 - q0) * (k / n) for k in range(n + 1)])

    def plan_joint(self, emb, q_goal, q_start=None, max_attempts=4, sync_world=True):
        self.calls.append("plan_joint")
        return _PlanRes(True, self._line(th.as_tensor(q_start).float(), th.as_tensor(q_goal).float()))

    def plan_pose(self, emb, targets, q_start=None, max_attempts=4, sync_world=True):
        self.calls.append("plan_pose")
        q1 = self.solve_ik(emb, targets, q_seed=q_start)
        return _PlanRes(True, self._line(th.as_tensor(q_start).float(), q1))


class _PosedBook:
    """The book in the sampler's scene (``get/set_position_orientation``) with the protection box at its pose."""

    def __init__(self, pose):
        self.name = "hardback_0"
        self.pose = (th.as_tensor(pose[0]).float(), th.as_tensor(pose[1]).float())
        self.moves = []

    def get_position_orientation(self):
        return self.pose

    def set_position_orientation(self, p, q):
        self.pose = (th.as_tensor(p).float(), th.as_tensor(q).float())
        self.moves.append(self.pose[0].tolist())

    def keep_still(self):
        pass

    def get_base_aligned_bbox(self, visual=False):
        c, R, h = OP.box_of(self.pose, _hb_hull())
        return c, th.tensor([0.0, 0.0, 0.0, 1.0]), 2 * h, None


def _p1_sampler():
    box = OP.box_of(BOOK_P1, _hb_hull())
    kit = _ToyKit()
    kit.backend = _ToyCurobo(box)
    kit.robot = SimpleNamespace(arm_control_idx={"right": th.arange(7), "left": th.arange(7, 14)})
    kit.syncs = 0
    kit.sync_world = lambda **kw: setattr(kit, "syncs", kit.syncs + 1)
    kit.collide = lambda qs: kit.backend.check_collision(qs)
    ikc = _ToyIKC()
    ikc.exact = lambda poses, st, arm, **kw: IKResult(True, kit.ik(poses[-1], th.zeros(28)), "ok", 1)
    s = SimpleNamespace(kit=kit, ikc=ikc, target=_PosedBook((th.tensor([5.0, 5.0, -10.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))), tname="hardback.n.01_1",
                        gp=SimpleNamespace(variant="toy", gen=lambda k, d=None: d), hull=_hb_hull(), support_objects=lambda: [], _check_deadline=lambda stage: None)
    for name in ("_overpath_verify", "_overpath_plan_planner"):
        setattr(s, name, getattr(IL.IORSLarge, name).__get__(s))
    return s


def _p1_plan(s, q_start, pre_pos):
    s._overpath_start = lambda st, gst, arm, chain, settled: (q_start, "escape", "")
    pre = (th.tensor(pre_pos), th.tensor([0.0, 0.0, 0.0, 1.0]))
    st = SimpleNamespace(name="manipulation")
    return IL.IORSLarge._overpath_plan_planner(s, st, st, "right", {}, BOOK_P1, pre, pre)


def _p1_q0():
    q0 = th.zeros(28)
    q0[0:3] = th.tensor([0.0, 0.20, 0.98])  # the escape's end beyond the far side; the straight plan to the pre-grasp passes the book at 7.8 mm (the live 6-9 mm)
    return q0


def test_p1_switch_on_runs_the_oracle_planner_switch_off_is_today(monkeypatch):
    """Switch on, no design given: the planner design (P1); ``cartesian`` only on request; off: nothing planned (test_switch_off_is_today)."""
    monkeypatch.setenv(OP.ENV, "1")
    monkeypatch.delenv(OP.DESIGN_ENV, raising=False)
    assert OP.design() == "planner" and OP.memo_tag() == ("overpath/3",)
    seen = []
    s = SimpleNamespace(_overpath_plan_planner=lambda *a: seen.append("planner") or (None, "plan", ""),
                        _overpath_plan_cartesian=lambda *a: seen.append("cartesian") or (None, "track", ""))
    IL.IORSLarge._overpath_plan(s, *([None] * 7))
    monkeypatch.setenv(OP.DESIGN_ENV, "cartesian")
    IL.IORSLarge._overpath_plan(s, *([None] * 7))
    monkeypatch.setenv(OP.DESIGN_ENV, "nonsense")
    IL.IORSLarge._overpath_plan(s, *([None] * 7))
    assert seen == ["planner", "cartesian", "planner"]


def test_p1_the_oracle_planner_records_a_path_clearing_the_book_by_its_margin(monkeypatch):
    """The oracle's own ladder on the toy backend: the joint-space plan to the witness passes the book inside the 3 cm margin, the
    protected-object gate (inflation BOOK_MARGIN_M) takes the via-above; the recorded waypoints, run as the oracle runs them, keep the
    moving sphere >= 3 cm from the book's box, start at the escape's end and end at the pre-grasp. The book stood at its settled pose for
    the plan and is put back."""
    monkeypatch.setenv("ISPATIALGYM_FAST_STRAIGHT_ARM", "0")  # no straight ramps: the trajopt rung (the toy cannot model the ramp's obstacles)
    s = _p1_sampler()
    q0 = _p1_q0()
    pre = [0.0, -0.16, 0.79]
    # the plan the oracle would stream without the gate: inside the margin
    line = s.kit.backend._line(q0, th.cat([th.tensor(pre), q0[3:]]))
    v0, _ = PROT.sphere_box_clearance(_ToyKit._spheres(line)[0][:, :1], OP.box_of(BOOK_P1, _hb_hull()))
    assert 0.0 < float(v0.min()) < OP.BOOK_MARGIN_M
    parked = s.target.pose[0].clone()
    rec, key, why = _p1_plan(s, q0, pre)
    assert rec is not None, (key, why)
    assert rec["design"] == "planner" and rec["source"] == "oracle planner (via_above)" and rec["planner"]["branch"] == "via_above"
    assert rec["min_book_clearance_m"] >= OP.BOOK_MARGIN_M and rec["margin_m"] == OP.BOOK_MARGIN_M and rec["rounds"] == 1
    assert s.kit.stats["overpath.planner.rounds"] == 1 and s.kit.stats["overpath.planner.branch.via_above"] == 1 and "overpath.planner_s" in s.kit.stats
    assert "plan_joint" in s.kit.backend.calls
    qs = OP.parse(rec)
    assert th.allclose(qs[0][0:3], q0[0:3]) and th.allclose(qs[-1][0:3], th.tensor(pre), atol=1e-4)
    full = []
    for q in qs:
        r = th.zeros(28)
        r[0:7] = q
        full.append(r)
    dense = OP.densify(full[0], full[1:])
    val, _ = PROT.sphere_box_clearance(_ToyKit._spheres(dense)[0][:, :1], OP.box_of(BOOK_P1, _hb_hull()))
    assert float(val[1:-1].min()) >= OP.BOOK_MARGIN_M - 1e-4
    # the book was at its settled pose while planning, then back where it was
    assert s.target.moves[0] == pytest.approx(BOOK_P1[0].tolist()) and th.allclose(s.target.pose[0], parked) and s.kit.syncs == 2
    json.dumps(rec)


def test_p1_rejects_or_replans_a_transit_that_fails_the_recheck(monkeypatch):
    """A round whose transit passes the book inside the margin is planned again (trajopt is stochastic); both rounds failing: rejected
    with the reason (kit overpath.planner.failed.<stage> for a planner failure)."""
    from ispatialgym.sampling.v2 import overpath_planner as OPP

    s = _p1_sampler()
    q0 = _p1_q0()
    pre = [0.0, -0.16, 0.79]
    q_pre = q0.clone()
    q_pre[0:3] = th.tensor(pre)
    low = s.kit.backend._line(q0, q_pre)  # 7.8 mm from the book
    up, over = q0.clone(), q_pre.clone()
    up[2] = over[2] = 1.10
    L = s.kit.backend._line
    high = th.cat([L(q0, up), L(up, over)[1:], L(over, q_pre)[1:]])  # up, across, down: >= 5 cm from the book
    plans = [low, high]
    monkeypatch.setattr(OPP, "plan_transit", lambda *a, **k: (plans.pop(0), {"branch": "trajopt", "plan_s": 1.0}))
    rec, key, why = _p1_plan(s, q0, pre)
    assert rec is not None and rec["rounds"] == 2 and rec["tried"] == ["1: book"] and rec["min_book_clearance_m"] >= OP.BOOK_MARGIN_M
    plans[:] = [low, low]
    rec, key, why = _p1_plan(s, q0, pre)
    assert rec is None and key == "book" and "passes the book" in why
    monkeypatch.setattr(OPP, "plan_transit", lambda *a, **k: (None, {"why": "plan_failed: arm_to(right): protected-object re-plan failed", "stage": "protected"}))
    rec, key, why = _p1_plan(s, q0, pre)
    assert rec is None and key == "plan" and "protected-object" in why and s.kit.stats["overpath.planner.failed.protected"] == OP.PLANNER_ROUNDS
    # the witness: no exact IK of the pre-grasp -> rejected before any plan
    s.ikc.exact = lambda poses, st, arm, **kw: IKResult(False, None, "ik_reach: no solution", 1)
    rec, key, why = _p1_plan(s, q0, pre)
    assert rec is None and key == "witness"


def test_p1_plan_only_motions_records_instead_of_simulating():
    """The plan-only Motions: its state is a joint vector; a streamed trajectory is recorded and moves it; the backend's implicit robot
    reads (q_full, fk without a configuration, the start of a plan / an IK path) come from that vector."""
    from ispatialgym.sampling.v2 import overpath_planner as OPP

    box = OP.box_of(BOOK_P1, _hb_hull())
    b = _ToyCurobo(box)
    robot = SimpleNamespace(arm_control_idx={"right": th.arange(7), "left": th.arange(7, 14)})
    q0 = _p1_q0()
    m = OPP.plan_only_motions(robot, b, q0)
    assert th.allclose(m.backend.q_full(), q0) and th.allclose(m.eef_pose("right")[0], q0[0:3])
    q1 = q0.clone()
    q1[0:3] = th.tensor([0.0, 0.0, 1.0])
    q1[20] = 5.0  # not an arm joint of the motion: never moved
    assert list(m.execute_joint_trajectory(th.stack([q0, q1]), arms=["right"])) == []
    assert th.allclose(m.q_full()[0:3], q1[0:3]) and float(m.q_full()[20]) == 0.0 and len(m.segments) == 1
    assert th.allclose(m.backend.solve_ik_path("right_arm", [], check_collisions=False).q_traj[0], m.q_full())
    cfg = OPP.oracle_motions_config()
    from ispatialgym.fastflags import fast_on

    if fast_on("PLAN_LADDER"):
        assert cfg.plan_attempts == 2 and cfg.witness_plan_retries == 1
