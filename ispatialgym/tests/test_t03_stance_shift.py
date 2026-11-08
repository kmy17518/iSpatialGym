"""i7-t03 (owner request 2026-11-08 item 1, ISPATIALGYM_SHELF_STANCE_SHIFT, default off): when the shelf layout's fixed start fails the
grasp or the transit check at a manipulation candidate, the base is moved along the edge - the computed stance (the book at
SHELF_STANCE_BOOK_LATERAL_M from the robot's centre line, same stand-off / trunk) first, then the scan - with the same checks at every
stance. Simulator-free."""

import math
from types import SimpleNamespace

import numpy as np

import pytest
import torch as th

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks import iors_small as IS

# the touching_03 frame: the robot faces -y (yaw -pi/2), the edge tangent (+ = the robot's left) is +x
EDGE = EdgeFrame("+u", th.tensor([-3.2418, -0.115]), th.tensor([0.0, -1.0]), 1.2)
BOOK = (th.tensor([-3.5176, -0.1222, 0.9484]), th.tensor([0.5761, 0.0, -0.8174, 0.0]))


def _cand(lateral: float, stand_off: float = 0.55) -> Stance:
    xy = EDGE.base_xy(stand_off, lateral)
    st = Stance("manipulation", (float(xy[0]), float(xy[1]), EDGE.yaw), 0.25, 0.0, stand_off, EDGE, None, lateral)
    st.focus = th.tensor([-3.62, -0.155, 0.947])
    return st


def _sampler(free_floor=lambda xy: True, collides=lambda st: False, gen=None):
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    s.shelf_pose = BOOK
    s.gen = lambda k, d=None: (gen or {}).get(k, d)
    s.kit = SimpleNamespace(trav_free=lambda xy: bool(free_floor(xy)), stats={})
    s.ikc = SimpleNamespace(stances_collision_free=lambda sts: th.tensor([not collides(st) for st in sts]))
    s.gp = SimpleNamespace(variant="t03")
    s.travel_dropped = []
    s.drop_travel_records = lambda a, b: s.travel_dropped.append((a, b))
    s._shelf_camera_rule = lambda cand: (math.radians(40.0), lambda st, cam, focus: True)
    s.world = SimpleNamespace(camera_model=SimpleNamespace(angle_off_axis=lambda cam, focus: 0.1))
    return s


@pytest.fixture(autouse=True)
def _cams(monkeypatch):
    monkeypatch.setattr(TT, "stance_cameras", lambda s, sts: [None for _ in sts])


def test_the_measured_book_lateral():
    """The 7 written demos: the book 0.185-0.200 m left of the base (stand-off 0.55, trunk 0.25 / 0) - the fix2 success at lateral -0.465."""
    st = _cand(-0.465)
    depth, lat = st.local(BOOK[0][:2])
    assert lat == pytest.approx(0.189, abs=2e-3) and IS.SHELF_STANCE_BOOK_LATERAL_M == 0.19


@pytest.mark.parametrize("arm,want", [("left", 0.19), ("right", -0.19)])
def test_computed_stance_puts_the_book_at_the_measured_lateral(arm, want):
    s = _sampler()
    cand = _cand(-0.065)  # 0.2 m right of the successes' start (the book 0.40 m to the robot's left)
    out = s.shelf_shift_stances(cand, arm)
    label, st, shift = out[0]
    assert label == "computed" and st.stand_off == cand.stand_off and (st.translate, st.tilt) == (cand.translate, cand.tilt)
    assert st.local(BOOK[0][:2])[1] == pytest.approx(want, abs=1e-4) and st.yaw == pytest.approx(cand.yaw)
    # the scan: shift list x STAND_OFFS (the computed lateral at the other stand-offs first), the computed stance not repeated
    assert all(lab == "scan" for lab, _, _ in out[1:])
    keys = {(round(float(x.lateral), 4), x.stand_off) for _, x, _ in out}
    assert len(keys) == len(out) == len(IS.SHELF_STANCE_SCAN_SHIFTS_M) * len(TT.STAND_OFFS)
    assert out[1][2] == out[0][2] and out[1][1].stand_off != pytest.approx(0.55)


def test_a_candidate_that_passes_is_never_shifted(monkeypatch):
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, "1")
    s = _sampler()
    cand = _cand(-0.465)
    calls = []
    st, g = s.shelf_fixed_start(None, cand, cand, "left", lambda m: calls.append(m) or "grasp")
    assert st is cand and g == "grasp" and calls == [cand] and s.kit.stats == {}


@pytest.mark.parametrize("switch", ["", "0"])
def test_switch_off_is_the_former_behaviour(monkeypatch, switch):
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, switch)
    s = _sampler()
    cand = _cand(-0.065)
    calls = []

    def fail(m):
        calls.append(m)
        raise Reject("grasp_unreachable", "no library grasp reaches the fixed start pose from manipulation")

    with pytest.raises(Reject) as e:
        s.shelf_fixed_start(None, cand, cand, "left", fail)
    assert "no library grasp" in str(e.value) and calls == [cand] and s.kit.stats == {}


def test_shifts_never_land_off_the_floor_or_in_collision(monkeypatch):
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, "1")
    off_floor = lambda xy: float(xy[0]) > -3.70  # noqa: E731 - the computed stance (x -3.7076) is free, every stance further right is not
    collides = lambda st: st.stand_off is not None and st.stand_off < 0.5  # noqa: E731
    s = _sampler(free_floor=off_floor, collides=collides, gen={"shelf_stance_scan_max": 50})
    cand = _cand(-0.065)
    seen = []

    def check(m):
        seen.append(m)
        if m is cand:
            raise Reject("grasp_unreachable", "transit check: the pre-grasp witness lies 3.52 rad from the transit's start posture (> 1.60)")
        raise Reject("grasp_unreachable", "no library grasp reaches the fixed start pose")

    init = SimpleNamespace(name="init")
    with pytest.raises(Reject):
        s.shelf_fixed_start(init, cand, cand, "left", check)
    tried = seen[1:]
    assert tried and all(float(m.base_xy[0]) > -3.70 and m.stand_off >= 0.5 for m in tried)
    st = s.kit.stats
    assert st["stance_shift.need.transit"] == 1 and st["stance_shift.computed_tried"] == 1 and st.get("stance_shift.rejected.floor", 0) > 0
    assert st.get("stance_shift.rejected.collision", 0) > 0 and st["stance_shift.rejected.check"] == len(tried)
    assert ("init", "manipulation") in s.travel_dropped


def test_the_first_passing_shift_is_the_stance(monkeypatch):
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, "1")
    s = _sampler()
    cand = _cand(-0.065)
    grasp = SimpleNamespace(grasp_name="grasp_0~r180", extras={"shelf_transit": {"reach_rad": 1.48}})

    def check(m):
        if m is cand:
            raise Reject("grasp_unreachable", "no library grasp reaches the fixed start pose from manipulation")
        return grasp

    st, g = s.shelf_fixed_start(SimpleNamespace(name="init"), cand, cand, "left", check)
    assert g is grasp and st is not cand and st.local(BOOK[0][:2])[1] == pytest.approx(0.19, abs=1e-4)
    rec = st.extras["stance_shift"]
    assert rec["label"] == "computed" and rec["need"] == "grasp" and rec["branch_after"] == "near" and st.arms == ["left"]
    assert s.kit.stats == {"stance_shift.need.grasp": 1, "stance_shift.computed_tried": 1, "stance_shift.computed_passed": 1,
                           f"stance_shift.offset.{rec['shift_m']:+.2f}": 1}


def test_failures_the_base_cannot_change_are_not_shifted(monkeypatch):
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, "1")
    assert IS.shelf_need_of(Reject("no_transition", "init -> manipulation: no s_T")) is None
    assert IS.shelf_need_of(Reject("no_transition", "shelf back-off: no s_T >= 0.75")) == "transit"
    assert IS.reach_branch(3.52) == "flipped" and IS.reach_branch(1.48) == "near" and IS.reach_branch(None) == "-"


# ----------------------------------------------------------------------------------------------- round 2 (owner 2026-11-08): (a) / (b) / (d)

def test_round2_switches_default_off(monkeypatch):
    for env in (IS.SHELF_FREE_START_ENV, IS.SHELF_JOINT_STANCE_ENV, IS.SHELF_LIFT_STRAIGHTEN_ENV):
        monkeypatch.delenv(env, raising=False)
    assert not IS.shelf_free_start_on() and not IS.shelf_joint_stance_on() and not IS.shelf_lift_straighten_on()


def test_free_start_keeps_an_edge_stance_that_frames_both_books(monkeypatch):
    s = _sampler()
    s.shelf_partner_pose = BOOK
    s.books_in_frame = lambda st, cam=None: True
    init = _cand(-0.465)
    assert s.shelf_free_start(np.random.default_rng(0), init) is init and s.kit.stats == {"free_start.edge_ok": 1}


def test_free_start_falls_back_to_the_free_floor_draw(monkeypatch):
    from ispatialgym.sampling.v2.tasks import iors_large as IL

    s = _sampler()
    s.shelf_partner_pose = BOOK
    s.books_in_frame = lambda st, cam=None: getattr(st, "name", "") == "free"
    seen = {}

    def fake(s_, c_xy, focus, view_ok, rng, cfg=None, what=""):
        st = Stance("free", (0.0, 1.0, -1.57), 0.25, 0.0, 1.2)
        seen["view"] = view_ok(st, None)
        st.extras["free_start"] = True
        return st

    monkeypatch.setattr(IL, "free_floor_stance", fake)
    st = s.shelf_free_start(np.random.default_rng(0), _cand(-0.065))
    assert st.name == "init" and st.extras["free_start"] and seen["view"] and s.kit.stats == {"free_start.used": 1}
    monkeypatch.setattr(IL, "free_floor_stance", lambda *a, **k: (_ for _ in ()).throw(Reject("no_stance", "none")))
    init = _cand(-0.065)
    assert s.shelf_free_start(np.random.default_rng(0), init) is init and s.kit.stats["free_start.failed.no_stance"] == 1


@pytest.mark.parametrize("release_ok,want", [(True, "both_passed"), (False, "grasp_only")])
def test_joint_stance_prefers_grasp_and_release(monkeypatch, release_ok, want):
    monkeypatch.setenv(IS.SHELF_JOINT_STANCE_ENV, "1")
    monkeypatch.delenv(IS.SHELF_STANCE_SHIFT_ENV, raising=False)
    s = _sampler()
    s.shelf_release_why = lambda st, g: None if release_ok else "clear pose: exact IK failed"
    s.sidecar = SimpleNamespace(transitions=[])
    cand = _cand(-0.465)
    g = SimpleNamespace(extras={})
    st, g1 = s.shelf_fixed_start(None, cand, cand, "left", lambda m, again=False: g)
    assert st is cand and g1 is g and s.kit.stats == {f"stance_shift.{want}": 1}


def test_joint_stance_takes_a_shifted_stance_that_does_both(monkeypatch):
    monkeypatch.setenv(IS.SHELF_JOINT_STANCE_ENV, "1")
    monkeypatch.setenv(IS.SHELF_STANCE_SHIFT_ENV, "1")
    s = _sampler()
    cand = _cand(-0.465)
    s.shelf_release_why = lambda st, g: None if st is not cand else "clear pose: exact IK failed"
    s.sidecar = SimpleNamespace(transitions=[])
    st, g1 = s.shelf_fixed_start(SimpleNamespace(name="init"), cand, cand, "left",
                                 lambda m, again=False: SimpleNamespace(grasp_name="g", extras={"shelf_transit": {"reach_rad": 1.4}}))
    assert st is not cand and st.extras["stance_shift"]["release"] == "ok" and s.kit.stats["stance_shift.both_passed"] == 1


def test_shortest_arc_straightens_a_lean():
    lean = 0.35  # 20 deg about the edge's depth axis
    q = th.tensor([math.sin(lean / 2), 0.0, 0.0, math.cos(lean / 2)])
    from ispatialgym.sampling import geometry as G

    up = G.rotate(q, th.tensor([0.0, 0.0, 1.0]))
    t = IS.shortest_arc(up, th.tensor([0.0, 0.0, 1.0]))
    assert float(th.linalg.norm(G.rotate(t, up) - th.tensor([0.0, 0.0, 1.0]))) < 1e-5
    assert 2 * math.acos(float(t[3])) == pytest.approx(lean, abs=1e-5)


def test_straightened_lift_path_stays_inside_the_headroom():
    from ispatialgym.sampling import geometry as G

    hull = th.tensor([[x, y, z] for x in (-0.02, 0.02) for y in (-0.08, 0.08) for z in (0.0, 0.22)])  # a 22 cm book, origin at its bottom
    lean = 0.35
    q_lean = th.tensor([math.sin(lean / 2), 0.0, 0.0, math.cos(lean / 2)])
    eio = (th.tensor([0.0, 0.0, 0.15]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    obj0 = (th.tensor([0.0, 0.0, 0.84]), q_lean)
    grasp_pose = G.compose(obj0, eio)
    turn = IS.shortest_arc(G.rotate(q_lean, th.tensor([0.0, 0.0, 1.0])), th.tensor([0.0, 0.0, 1.0]))
    end = (G.t(grasp_pose[0]) + th.tensor([0.0, 0.0, 0.04]), G.qmul(turn, G.t(grasp_pose[1])))
    poses = IS.lift_object_poses(grasp_pose, end, eio)
    up_end = G.rotate(G.t(poses[-1][1]), th.tensor([0.0, 0.0, 1.0]))
    assert float(up_end[2]) == pytest.approx(1.0, abs=1e-4)  # upright at the end
    top = max(float(G.local_points_world(hull, p)[:, 2].max()) for p in poses)
    assert IS.lift_headroom_m(hull, poses, top + 0.01) == pytest.approx(0.01, abs=1e-6)
    assert IS.lift_headroom_m(hull, poses, top - 0.01) < 0  # a lower board above: the turn is refused (shelf_lift.no_headroom)


def _touch_sampler(monkeypatch, book_x_min, robot_clear=True):
    from ispatialgym.sampling.v2 import support_split as SS

    monkeypatch.setenv(IS.SHELF_LIFT_STRAIGHTEN_ENV, "1")
    s = _sampler()
    sup = SimpleNamespace(name="bookcase_kmuacd_0")
    s.obj = lambda role: sup
    monkeypatch.setattr(SS, "_pieces", lambda o: [("/board", object())])
    hull = th.tensor([[x, y, z] for x in (0.0, 0.03) for y in (-0.08, 0.08) for z in (0.0, 0.22)])
    s.hull = {"target": hull}
    s.support_objects = lambda: []
    calls = {"n": 0, "ign": []}

    @__import__("contextlib").contextmanager
    def ignoring(objs, split=True):
        calls["ign"].append([getattr(o, "name", "") for o in objs])
        yield

    class Plane:
        def sdf(self, p):
            return th.as_tensor(p)[..., 0] - 0.0  # the board is x <= 0: distance = x

    s.ikc = SimpleNamespace(ignoring=ignoring, _pieces_geometry=lambda sp, only=None: Plane(), _non_finger_links=lambda arm: ["left_arm_link5", "attached_object_left_eef_link"])
    s.kit.links_clear = lambda q, links, m: th.tensor([robot_clear])
    eio = (th.tensor([0.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    gr = SimpleNamespace(arm="left", grasp=(th.tensor([book_x_min, 0.0, 0.84]), th.tensor([0.0, 0.0, 0.0, 1.0])), carry_height=0.04, eef_in_object=eio,
                         extras={"lift_leaned_piece": "/board"})
    return s, gr


@pytest.mark.parametrize("x_min,robot_clear,ok,why", [(0.0, True, True, None), (-0.002, True, False, "penetration"), (0.0, False, False, "robot_link")])
def test_leaned_board_touch_at_the_lift(monkeypatch, x_min, robot_clear, ok, why):
    s, gr = _touch_sampler(monkeypatch, x_min, robot_clear)
    fail = SimpleNamespace(ok=False, q=None)
    passing = SimpleNamespace(ok=True, q=th.zeros(28))
    seq = iter([fail, passing])
    r = s.lift_with_touch(gr, lambda: next(seq))
    assert bool(r.ok) is ok
    if ok:
        assert s.kit.stats == {"shelf_lift.leaned_board_touch_allowed": 1}
    else:
        assert s.kit.stats == {f"shelf_lift.leaned_board_touch_refused.{why}": 1}


@pytest.mark.parametrize("cls,why", [([-5e-5, 0.0, 0.01], None), ([-2e-4, 0.0, 0.01], "penetration_start"), ([0.0, -1e-5, 0.01], "penetration"),
                                     ([-5e-5, 0.01, -1e-6], "penetration"), ([0.0, 0.0], None), ([], "penetration")])
def test_lift_touch_clearance_rule(cls, why):
    """(d) touch-rule fix: pose 0 (the resting lean) may read down to -0.1 mm; every later pose keeps >= 0."""
    assert IS.lift_touch_clearance_why(cls) == why


@pytest.mark.parametrize("xs,robot_clear,ok,stats", [
    ([-5e-5, 0.004, 0.01], True, True, {"shelf_lift.leaned_board_touch_allowed": 1, "shelf_lift.leaned_board_start_contact_allowed": 1}),  # 0.05 mm at pose 0
    ([-2e-4, 0.004, 0.01], True, False, {"shelf_lift.leaned_board_touch_refused.penetration_start": 1}),  # 0.2 mm at pose 0
    ([0.0, -1e-5, 0.01], True, False, {"shelf_lift.leaned_board_touch_refused.penetration": 1}),  # below 0 at pose 1
    ([-5e-5, -5e-5, 0.01], True, False, {"shelf_lift.leaned_board_touch_refused.penetration": 1}),  # the resting depth kept at pose 1
    ([0.0, 0.0, 0.01, -1e-6], True, False, {"shelf_lift.leaned_board_touch_refused.penetration": 1}),  # below 0 at a later pose
    ([-5e-5, 0.004, 0.01], False, False, {"shelf_lift.leaned_board_touch_refused.robot_link": 1}),  # robot links: today's rule
])
def test_leaned_board_start_contact_at_the_lift(monkeypatch, xs, robot_clear, ok, stats):
    s, gr = _touch_sampler(monkeypatch, 0.0, robot_clear)
    q = th.tensor([0.0, 0.0, 0.0, 1.0])
    monkeypatch.setattr(IS, "lift_object_poses", lambda g, end, eio, n=10: [(th.tensor([x, 0.0, 0.84 + 0.004 * k]), q) for k, x in enumerate(xs)])
    fail = SimpleNamespace(ok=False, q=None)
    passing = SimpleNamespace(ok=True, q=th.zeros(28))
    seq = iter([fail, passing])
    r = s.lift_with_touch(gr, lambda: next(seq))
    assert bool(r.ok) is ok
    assert s.kit.stats == stats


def test_leaned_board_touch_switch_off_is_today(monkeypatch):
    s, gr = _touch_sampler(monkeypatch, 0.0)
    monkeypatch.setenv(IS.SHELF_LIFT_STRAIGHTEN_ENV, "0")
    fail = SimpleNamespace(ok=False, q=None)
    assert s.lift_with_touch(gr, lambda: fail) is fail and s.kit.stats == {}


def test_oracle_lift_turn_and_the_sampler_lift_end_agree():
    from ispatialgym.oracle import actions as A
    from ispatialgym.oracle.sidecar import Pose
    from ispatialgym.sampling.v2 import program_check as PC

    turn = [math.sin(0.175), 0.0, 0.0, math.cos(0.175)]
    start = Pose([0.1, 0.2, 0.9], [0.0, 0.0, 0.0, 1.0])
    lift = A.Lift(arm="left", object="book", carry_height=0.04, after=None, away=[0.03, 0.0, 0.0], turn=turn)
    _, end = lift._segment(None, SimpleNamespace(eef_pose=lambda arm: start))
    g = SimpleNamespace(grasp=(start.position, start.orientation), carry_height=0.04, extras={"lift_away": [0.03, 0.0, 0.0], "lift_turn": turn})
    p, q = PC.lift_end_pose(g)
    assert end.position == pytest.approx([0.13, 0.2, 0.94]) and list(p) == pytest.approx(end.position, abs=1e-6)
    assert list(q) == pytest.approx(end.orientation, abs=1e-6)
    assert A.Lift(arm="left", object="book", carry_height=0.04, after=None)._segment(None, SimpleNamespace(eef_pose=lambda arm: start))[1].orientation == [0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("gap,n", [(0.13, 4), (0.012, 0)])
def test_release_drives_stay_inside_the_free_gap(gap, n):
    """(c): every release drive moves the base toward the partner by at most the free gap less the carried path margin (1 cm)."""
    s = _sampler()
    s.shelf_start = {"lean_side": 1.0}  # the target leans on the LEFT board: the partner (and the drive) to the robot's right
    s.lifted_book_gap = lambda manip, grasp: gap
    s.shelf_push_arm = lambda: "left"
    manip = _cand(-0.465)
    plan = s.shelf_release_in_gap_plan(manip, None)
    assert len(plan) == n
    for d, st in plan:
        move = float((st.base_xy - manip.base_xy) @ manip.tangent)
        assert move == pytest.approx(-d, abs=1e-5) and 0.0 < d <= gap - 0.01 + 1e-9 and st.stand_off == manip.stand_off and st.name == "push"


# -- (d) fix 2026-11-08: the straightening lift on touching_03 (IORSSmall.up_axis is inside-only; i7t03r3_d / _v crashed on it) ------------

# acbrnv at the touching_03 start (BOOK): the model -x is its standing axis (height 22.3 cm), +-y its width into the shelf, +-z its 3.1 cm
BOOK_HULL = th.tensor([[x, y, z] for x in (-0.1115, 0.1115) for y in (-0.075, 0.075) for z in (-0.0155, 0.0155)])


class _Board:
    """A kept piece as the lift plan reads it: AABB boxes + a signed distance (a horizontal board above at ``z0``)."""

    def __init__(self, lo, hi):
        self.boxes = [(th.tensor(lo), th.tensor(hi))]

    def sdf(self, p):
        lo, hi = self.boxes[0]
        p = th.as_tensor(p, dtype=th.float32)
        return th.maximum(lo - p, p - hi).max(dim=-1).values


def _lift_sampler(monkeypatch, ceiling_gap=0.10, rest=BOOK[1], key="hardback__acbrnv__bu", switch="1"):
    from ispatialgym.sampling import geometry as G

    monkeypatch.setenv(IS.SHELF_LIFT_STRAIGHTEN_ENV, switch)
    s = _sampler()
    s.relation = "touching"
    s.hull = {"target": BOOK_HULL}
    s.rest = {"target": rest}
    s.grasp_key = key
    s.world = SimpleNamespace(hull_at=lambda h, pose: G.local_points_world(h, (G.t(pose[0]).float(), G.t(pose[1]).float())))
    hull_w = G.local_points_world(BOOK_HULL, BOOK)
    top = float(hull_w[:, 2].max())
    lo, hi = hull_w.min(dim=0).values, hull_w.max(dim=0).values
    above = _Board([float(lo[0]) - 0.1, float(lo[1]) - 0.1, top + ceiling_gap], [float(hi[0]) + 0.1, float(hi[1]) + 0.1, top + ceiling_gap + 0.02])
    s._kept_pieces = lambda: [("/World/bookcase/board_above", None, above)]
    eio = (th.tensor([0.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    gr = SimpleNamespace(arm="left", grasp=G.compose(BOOK, eio), carry_height=0.04, eef_in_object=eio, extras={"lift_away": [0.03, 0.0, 0.0]})
    return s, gr


def test_lift_plan_on_touching_03_does_not_raise_and_straightens(monkeypatch):
    """The i7t03r3_d crash: shelf_lift_plan on touching (up_axis() None) - now the book's standing axis from its rest (-x)."""
    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.v2 import program_check as PC

    s, gr = _lift_sampler(monkeypatch)
    assert s.up_axis() is None  # unchanged: inside-only
    facts = s.shelf_lift_plan(gr)
    assert facts["axis"] == [-1, 0, 0] and facts["axis_source"] == "rest" and facts["start_tilt_deg"] == pytest.approx(19.6, abs=0.2)
    assert s.kit.stats == {"shelf_lift.straightened": 1} and "lift_turn" in gr.extras and facts["turn_deg"] == pytest.approx(19.6, abs=0.2)
    end = PC.lift_end_pose(gr)
    poses = IS.lift_object_poses(gr.grasp, end, gr.eef_in_object)
    up_end = G.rotate(G.t(poses[-1][1]).float(), th.tensor([-1.0, 0.0, 0.0]))
    assert float(up_end[2]) == pytest.approx(1.0, abs=1e-4)  # standing upright at the lift's end


def test_lift_plan_reads_the_v2_lean_base(monkeypatch):
    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling import grasp_library as GL

    upright = IS.shortest_arc(G.rotate(BOOK[1], th.tensor([-1.0, 0.0, 0.0])), th.tensor([0.0, 0.0, 1.0]))
    q = G.qmul(upright, BOOK[1])  # xyzw, -x vertical
    monkeypatch.setattr(GL, "v2_entry", lambda key: SimpleNamespace(data={"lean_base_wxyz": [float(q[3]), float(q[0]), float(q[1]), float(q[2])]}))
    s, gr = _lift_sampler(monkeypatch, rest=None, key="hardback__acbrnv__bu@lean_left_20")
    facts = s.shelf_lift_plan(gr)
    assert facts["axis"] == [-1, 0, 0] and facts["axis_source"] == "v2_lean_base" and s.kit.stats == {"shelf_lift.straightened": 1}


@pytest.mark.parametrize("gap", [0.10, 0.03, 0.0])
def test_straightened_touching_03_lift_stays_inside_the_headroom(monkeypatch, gap):
    """The turn is kept only when the upright book's top stays >= LIFT_HEADROOM_MARGIN_M under the board above along the whole lift."""
    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.v2 import program_check as PC

    s, gr = _lift_sampler(monkeypatch, ceiling_gap=gap)
    facts = s.shelf_lift_plan(gr)
    if "lift_turn" in gr.extras:
        poses = IS.lift_object_poses(gr.grasp, PC.lift_end_pose(gr), gr.eef_in_object)
        top = max(float(G.local_points_world(BOOK_HULL, (G.t(p[0]).float(), G.t(p[1]).float()))[:, 2].max()) for p in poses)
        assert facts["ceiling_z"] - top >= IS.LIFT_HEADROOM_MARGIN_M - 1e-4 and s.kit.stats == {"shelf_lift.straightened": 1}
    else:
        assert facts["headroom_m"] < IS.LIFT_HEADROOM_MARGIN_M and s.kit.stats == {"shelf_lift.no_headroom": 1}
    assert ("lift_turn" in gr.extras) is (gap >= 0.10)  # 0.04 lift + ~2.4 cm the upright book gains: 3 cm is too little


@pytest.mark.parametrize("rest,key,why", [(None, None, "no_rest"), (th.tensor([0.0, 0.0, 0.0, 1.0]), "k", "tilted")])
def test_lift_plan_without_an_axis_is_todays_lift(monkeypatch, rest, key, why):
    """No standing axis (no rest / one lying at the start): no turn, no leaned-board touch (today's lift), counted, never raises."""
    s, gr = _lift_sampler(monkeypatch, rest=rest, key=key)
    gr.extras.update(lift_turn=[0.0, 0.0, 0.0, 1.0], lift_leaned_piece="/stale")  # a stale record is cleared too
    facts = s.shelf_lift_plan(gr)
    assert facts["fallback"] == "no_axis" and facts["why"] == why
    assert "lift_turn" not in gr.extras and "lift_leaned_piece" not in gr.extras and s.kit.stats == {f"shelf_lift.fallback_no_axis.{why}": 1}
    assert s.lift_touch_ignores(gr) == []


def test_lift_plan_error_is_todays_lift(monkeypatch):
    s, gr = _lift_sampler(monkeypatch)

    def boom():
        raise RuntimeError("no geometry")

    s._kept_pieces = boom
    facts = s.shelf_lift_plan(gr)
    assert facts["fallback"] == "error" and "lift_turn" not in gr.extras and s.kit.stats == {"shelf_lift.fallback_error": 1}


def test_lift_straighten_off_is_today(monkeypatch):
    """Switch off: after_goal_check never plans (the call sits under shelf_lift_straighten_on), the lift end is today's, no leaned piece."""
    import inspect

    from ispatialgym.sampling.v2 import program_check as PC

    s, gr = _lift_sampler(monkeypatch, switch="0")
    assert not IS.shelf_lift_straighten_on()
    src = inspect.getsource(IS.IORSSmall.after_goal_check)
    assert "if shelf_lift_straighten_on():" in src and src.index("if shelf_lift_straighten_on():") < src.index("self.shelf_lift_plan(grasp)")
    p, q = PC.lift_end_pose(gr)
    assert list(q) == pytest.approx(list(gr.grasp[1]), abs=1e-7)
    gr.extras["lift_leaned_piece"] = "/World/bookcase/board"
    assert s.lift_touch_ignores(gr) == []


# -- owner 5k.1 (2026-11-08): the shelf-width rule computed from the jaw and the carried book (ISPATIALGYM_SHELF_SPAN_RULE, default off) -----

def _span_sampler(monkeypatch, straighten="0", preshape=True):
    from ispatialgym.sampling import geometry as G

    monkeypatch.setenv(IS.SHELF_LIFT_STRAIGHTEN_ENV, straighten)
    s = _sampler(gen={"approach_finger_support_touch": {"preshape": preshape}, "shelf_free_span_min_m": 0.12})
    s.edge = EDGE
    s.hull = {"target": BOOK_HULL}
    s.rest = {"target": BOOK[1]}
    s.grasp_key = "hardback__acbrnv__bu"
    s.grasps = [SimpleNamespace(meta={"width": 0.0312})]
    s.world = SimpleNamespace(hull_at=lambda h, pose: G.local_points_world(h, (G.t(pose[0]).float(), G.t(pose[1]).float())))
    return s


def test_span_rule_default_off_keeps_the_fixed_minimum(monkeypatch):
    monkeypatch.delenv(IS.SHELF_SPAN_RULE_ENV, raising=False)
    s = _span_sampler(monkeypatch)
    assert not IS.shelf_span_rule_on() and s.shelf_span_min(BOOK) == 0.12


@pytest.mark.parametrize("straighten,carried,want", [("0", "lean", 0.104 + 0.044), ("1", "upright", 0.031 + 0.044)])
def test_span_required_is_the_carried_extent_plus_the_open_jaw(monkeypatch, straighten, carried, want):
    """acbrnv at the touching_03 start (19.6 deg): its extent along the edge 0.223 sin + 0.031 cos = 0.104 m leaning, its 3.1 cm thickness
    upright (the straightening lift); the pre-shaped jaw (pinch 31.2 mm + 2 x 10 mm) plus both 12 mm fingers exceeds the pinch by 44 mm."""
    monkeypatch.setenv(IS.SHELF_SPAN_RULE_ENV, "1")
    s = _span_sampler(monkeypatch, straighten)
    req, f = s.shelf_span_required(BOOK)
    assert f["carried"] == carried and f["jaw_excess_m"] == pytest.approx(0.044, abs=1e-4) and req == pytest.approx(want, abs=2e-3)
    assert s.shelf_span_min(BOOK) == pytest.approx(req)


def test_span_required_at_the_drawn_lean(monkeypatch):
    """The requirement follows the lean the attempt drew: 11 deg needs ~3 cm less than 20 deg; without the pre-shape the jaw opens fully."""
    from ispatialgym.sampling import geometry as G

    monkeypatch.setenv(IS.SHELF_SPAN_RULE_ENV, "1")
    s = _span_sampler(monkeypatch)
    up = G.qmul(IS.shortest_arc(G.rotate(BOOK[1], th.tensor([-1.0, 0.0, 0.0])), th.tensor([0.0, 0.0, 1.0])), BOOK[1])

    def leaned(deg):  # the upright book turned about the edge's inward axis (-y)
        a = math.radians(deg)
        return (BOOK[0], G.qmul(th.tensor([0.0, -math.sin(a / 2), 0.0, math.cos(a / 2)]), up))

    r11, r20 = s.shelf_span_required(leaned(11.0))[0], s.shelf_span_required(leaned(20.0))[0]
    assert r20 - r11 == pytest.approx(0.223 * (math.sin(math.radians(20)) - math.sin(math.radians(11))) + 0.031 * (math.cos(math.radians(20)) - math.cos(math.radians(11))), abs=2e-3)
    s2 = _span_sampler(monkeypatch, preshape=False)
    assert s2.shelf_span_required(leaned(20.0))[1]["jaw_excess_m"] == pytest.approx(0.10 + 0.024 - 0.0312, abs=1e-4)


# -- owner 5p.2: the stance shift tries every candidate's trunk (under (b)'s ISPATIALGYM_SHELF_JOINT_STANCE) ---------------------------------

@pytest.mark.parametrize("joint", ["0", "1"])
def test_stance_shift_tries_every_candidates_trunk(monkeypatch, joint):
    monkeypatch.setenv(IS.SHELF_JOINT_STANCE_ENV, joint)
    s = _sampler()
    cand = _cand(-0.065)
    other = _cand(-0.2).with_trunk(0.5, -0.6)
    same = _cand(-0.3)  # the candidate's own trunk again: not repeated
    s.manip_candidates = [cand, other, same]
    out = s.shelf_shift_stances(cand, "left")
    trunks = [(round(st.translate, 3), round(st.tilt, 3)) for _l, st, _sh in out]
    assert out[0][0] == "computed" and trunks[0] == (0.25, 0.0)
    if joint == "1":
        assert trunks[1] == (0.5, -0.6) and out[1][0] == "scan" and out[1][1].extras["stance_shift_trunk"] == [0.5, -0.6]
        assert out[1][1].base_pose2d == out[0][1].base_pose2d and len(out) == 2 * len(set((round(st.base_pose2d[0], 4), round(st.base_pose2d[1], 4)) for _l, st, _sh in out))
    else:
        assert set(trunks) == {(0.25, 0.0)}


# -- (c) 2f815f1b: the free gap the carried drive may use ---------------------------------------------------------------------------------

def test_lifted_book_gap_measures_the_book_at_the_lift_end_to_the_partner(monkeypatch):
    """(c)'s room: the partner's near face minus the lifted book's far face along the push direction, the book at ``lift_end_pose`` (the lift's
    away move included); the drive plan then uses fractions of that less the carried path margin."""
    from ispatialgym.sampling import geometry as G

    s = _sampler()
    s.shelf_start = {"lean_side": 1.0}  # target on the left board: push toward -x (EDGE tangent +x)
    cube = th.tensor([[x, y, z] for x in (-0.02, 0.02) for y in (-0.05, 0.05) for z in (0.0, 0.2)])
    s.hull = {"target": cube, "partner": cube}
    s.world = SimpleNamespace(hull_at=lambda h, pose: G.local_points_world(h, (G.t(pose[0]).float(), G.t(pose[1]).float())))
    s.obj = lambda role: SimpleNamespace(get_position_orientation=lambda: (th.tensor([-0.30, 0.0, 0.84]), th.tensor([0.0, 0.0, 0.0, 1.0])))
    eio = (th.tensor([0.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    gr = SimpleNamespace(grasp=(th.tensor([0.0, 0.0, 0.84]), th.tensor([0.0, 0.0, 0.0, 1.0])), carry_height=0.04, eef_in_object=eio,
                         extras={"lift_away": [-0.03, 0.0, 0.0]})
    gap = s.lifted_book_gap(_cand(-0.465), gr)
    assert gap == pytest.approx(0.30 - 0.04 - 0.03, abs=1e-5)  # centres 0.30 apart, two 2 cm half thicknesses, the 3 cm away move
    s.lifted_book_gap = lambda manip, grasp: gap
    s.shelf_push_arm = lambda: "left"
    plan = s.shelf_release_in_gap_plan(_cand(-0.465), gr)
    assert [round(d, 4) for d, _st in plan] == [round(f * (gap - 0.01), 4) for f in IS.SHELF_RELEASE_DRIVE_FRACTIONS]
