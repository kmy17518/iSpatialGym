"""i7-t03 round 7 (LEAD DECISION #37, owner away; every switch DEFAULT OFF): the arm_link5 / torso_link2 collision-mesh check
(ISPATIALGYM_T03_PAIR_MESH), the shelf stance's pre-grasp frame test (ISPATIALGYM_SHELF_PREGRASP_FRAME), the tracked release retreat
(ISPATIALGYM_SHELF_RELEASE_TRACKED) and the log-only blame of a colliding tracked-chain end (ISPATIALGYM_TRACKED_END_BLAME). Simulator-free."""

import os
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import pair_mesh as PM
from ispatialgym.sampling.v2.tasks import iors_small as IS
from ispatialgym.tests.test_touching_shelf import _clear_sampler, _stance


def _box(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=float)


# ---------------------------------------------------------------- the geometry

@pytest.mark.parametrize("gap", [0.01, 0.002, 0.0005])
def test_convex_distance_of_separated_boxes(gap):
    a = _box((0, 0, 0), (1, 1, 1))
    b = _box((1 + gap, 0.2, 0.2), (2, 0.8, 0.8))
    d = PM.convex_distance(a, b)
    assert gap - 1e-9 <= d <= gap + 5e-5  # an upper bound within tol, never below the true gap


def test_convex_distance_of_overlapping_and_touching_boxes():
    a = _box((0, 0, 0), (1, 1, 1))
    assert PM.convex_distance(a, _box((0.997, 0.2, 0.2), (2, 0.8, 0.8))) == pytest.approx(-0.003, abs=1e-9)  # 3 mm deep
    assert PM.convex_distance(a, _box((1.0, 0.2, 0.2), (2, 0.8, 0.8))) <= 0.0  # face contact = touching
    assert PM.hulls_overlap(a, _box((0.5, 0.5, 0.5), (0.6, 0.6, 0.6)))
    assert not PM.hulls_overlap(a, _box((1.01, 0, 0), (2, 1, 1)))


def test_urdf_chain_and_link_transform(tmp_path):
    p = tmp_path / "r.urdf"
    p.write_text("""<robot name="r"><link name="base_link"/><link name="a"/><link name="b"/>
      <joint name="j1" type="revolute"><origin xyz="0 0 1" rpy="0 0 0"/><parent link="base_link"/><child link="a"/><axis xyz="0 0 1"/></joint>
      <joint name="j2" type="fixed"><origin xyz="1 0 0" rpy="0 0 0"/><parent link="a"/><child link="b"/></joint></robot>""")
    chain = PM.urdf_chain(str(p))
    T = PM.link_transform("b", {"j1": np.pi / 2}, chain)
    assert T[:3, 3] == pytest.approx([0.0, 1.0, 1.0], abs=1e-9)


def test_parse_usda_pieces_applies_the_mesh_xform():
    text = '''    def Xform "lnk" (
    )
    {
        def "collisions"
        {
            def Mesh "lnk_col_0" (
                apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
            )
            {
                point3f[] points = [(1, 0, 0), (0, 1, 0)]
                quatd xformOp:orient = (1, 0, 0, 0)
                double3 xformOp:scale = (2, 3, 1)
                double3 xformOp:translate = (0, 0, 1)
            }
        }
    }

    def Xform "other" (
'''
    (name, P), = PM.parse_usda_pieces(text, "lnk")
    assert name == "lnk_col_0" and P.tolist() == [[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]]


def test_pair_touch_why_on_synthetic_pieces(monkeypatch):
    monkeypatch.setattr(PM, "urdf_chain", lambda *a, **k: {})
    pieces = {"left_arm_link5": (("col_3", _box((0, 0, 0), (0.1, 0.1, 0.1))),), "torso_link2": (("collisions", _box((0.1005, 0, 0), (0.3, 0.1, 0.1))),)}
    monkeypatch.setattr(PM, "link_pieces", lambda link, *a: pieces[link])
    q = [0.0] * 28
    assert PM.pair_touch_why(q, "left", margin=0.0) is None  # 0.5 mm clear at margin 0
    assert "touches torso_link2" in PM.pair_touch_why(q, "left", margin=0.001)
    assert PM.pair_touch_why(q, "left") is not None  # the default margin (5 mm): the executed arm comes closer than the witness


_ASSETS_OK = True
try:
    _ASSETS_OK = os.path.exists(os.path.join(PM._assets_dir(), "usd", "r1pro.usda"))
except Exception:  # noqa: BLE001
    _ASSETS_OK = False

# the program-check retreat witnesses of r6 (oracle.json ik_witnesses): T7 300400 the place's 10 cm retreat (QA: link5 touches torso_link2),
# T7 300401 the 15 cm retreat (success); mesh clearances measured in r6 -15.6 / +56.1 mm
Q_300400 = [-4.13999, -1.35665, 0.0053, 0.0, 0.0, 1.5708, 1.5425, -2.395, -1.2062, 0.0, -0.17491, -0.073, 0.06064, -0.326, -0.88306, 0.951, -1.15544,
            -1.639, -0.19556, -0.632, -0.23759, 0.137, 0.98076, -0.926, 0.05, 0.05, 0.05, 0.05]
Q_300401 = [-3.67766, -1.35665, 0.0053, 0.0, 0.0, 1.5708, 1.5425, -2.395, -0.893, 0.0, -0.02006, 0.0, 0.23747, -1.57, -0.82052, 0.0, -1.07946, -1.57,
            0.05471, -1.57, -0.38114, 0.0, 0.78967, 0.0, 0.05, 0.05, 0.05, 0.05]


@pytest.mark.skipif(not _ASSETS_OK, reason="r1pro robot assets not available")
def test_the_measured_r6_witnesses():
    cl0, which = PM.pair_clearance(Q_300400, "left")
    assert cl0 < -0.010 and "left_arm_link5_col_3" in which
    cl1, _ = PM.pair_clearance(Q_300401, "left")
    assert cl1 > 0.04
    assert PM.pair_touch_why(Q_300400, "left") is not None and PM.pair_touch_why(Q_300401, "left") is None
    assert PM.PAIR_MESH_MARGIN_M == 0.005


# ---------------------------------------------------------------- the sampler hooks

def _bind(s, *names):
    for n in names:
        setattr(s, n, types.MethodType(getattr(IS.IORSSmall, n), s))


def _r7_sampler(monkeypatch, clear_ok=True, end_ok=True, backoff="0"):
    s, st, place, calls = _clear_sampler(monkeypatch, clear_ok=clear_ok, end_ok=end_ok, switch=backoff)
    _bind(s, "release_tracked", "pair_mesh_why")
    tracked = []
    exact0 = s.ikc.exact

    def exact(poses, st_, arm, **kw):
        if len(poses) > 1:  # the tracked chain [place, *backs]
            tracked.append(([round(float(th.linalg.norm(th.as_tensor(p[0]) - place[0])), 3) for p in poses], kw.get("q_first")))
            return exact0(poses[-1:], st_, arm, **kw)
        return exact0(poses, st_, arm, **kw)

    s.ikc.exact = exact
    return s, st, place, calls, tracked


def test_round7_switches_off_are_today(monkeypatch):
    for env in (PM.PAIR_MESH_ENV, IS.SHELF_RELEASE_TRACKED_ENV, IS.SHELF_PREGRASP_FRAME_ENV, IK.TRACKED_END_BLAME_ENV):
        monkeypatch.delenv(env, raising=False)
    assert not (PM.pair_mesh_on() or IS.shelf_release_tracked_on() or IS.shelf_pregrasp_frame_on() or IK.tracked_end_blame_on())
    s, st, place, calls, tracked = _r7_sampler(monkeypatch)
    monkeypatch.setattr(PM, "pair_touch_why", lambda *a, **k: pytest.fail("pair mesh judged with the switch off"))
    assert s.release_clear_why(st, place, "left", q_place=th.zeros(28)) == (None, None)
    assert calls == [0.15] and tracked == [] and s.kit.stats == {}


def test_tracked_release_leaves_from_the_place_configuration(monkeypatch):
    monkeypatch.setenv(IS.SHELF_RELEASE_TRACKED_ENV, "1")
    s, st, place, calls, tracked = _r7_sampler(monkeypatch)
    qp = th.full((28,), 0.3)
    assert s.release_clear_why(st, place, "left", q_place=qp) == (None, None)
    (backs, q_first), = tracked
    assert backs == [0.0, 0.1, 0.15] and q_first is not None and bool(th.equal(q_first, qp))
    assert s.kit.stats == {"shelf_release_tracked.checked": 1, "shelf_release_tracked.passed": 1}


def test_tracked_release_failure_and_the_backoff_place_end(monkeypatch):
    monkeypatch.setenv(IS.SHELF_RELEASE_TRACKED_ENV, "1")
    s, st, place, calls, tracked = _r7_sampler(monkeypatch, clear_ok=False, end_ok=True, backoff="1")
    w, code = s.release_clear_why(st, place, "left", q_place=th.zeros(28))
    assert w is None  # the 15 cm tracked end fails, the 5p.1 back-off's place end (tracked to 10 cm) passes
    assert [b for b, _ in tracked] == [[0.0, 0.1, 0.15], [0.0, 0.1]]
    assert s.kit.stats["shelf_release_tracked.failed"] == 1 and s.kit.stats["shelf_release_tracked.passed"] == 1


@pytest.mark.parametrize("touch", [False, True])
def test_pair_mesh_at_the_clear_pose(monkeypatch, touch):
    monkeypatch.setenv(PM.PAIR_MESH_ENV, "1")
    s, st, place, calls, tracked = _r7_sampler(monkeypatch)
    monkeypatch.setattr(PM, "pair_touch_why", lambda q, arm, margin=PM.PAIR_MESH_MARGIN_M: "left_arm_link5 touches torso_link2 (-0.4 mm)" if touch else None)
    w, code = s.release_clear_why(st, place, "left")
    if touch:
        assert "self mesh contact at the clear pose" in w and s.kit.stats == {"pair_mesh.checked.clear": 1, "pair_mesh.refused.clear": 1}
    else:
        assert w is None and s.kit.stats == {"pair_mesh.checked.clear": 1}


def test_pair_mesh_refusal_falls_back_to_the_backoff_and_checks_the_place_end(monkeypatch):
    monkeypatch.setenv(PM.PAIR_MESH_ENV, "1")
    s, st, place, calls, tracked = _r7_sampler(monkeypatch, backoff="1")
    seen = []

    def why(q, arm, margin=PM.PAIR_MESH_MARGIN_M):
        seen.append(1)
        return "touch" if len(seen) == 1 else None  # the clear pose touches, the place end (10 cm) is clear

    monkeypatch.setattr(PM, "pair_touch_why", why)
    w, code = s.release_clear_why(st, place, "left")
    assert w is None and calls == [0.15, 0.1]
    assert s.kit.stats["pair_mesh.refused.clear"] == 1 and s.kit.stats["pair_mesh.checked.place_end"] == 1
    assert "pair_mesh.refused.place_end" not in s.kit.stats


# ---------------------------------------------------------------- the shelf stance frame

def test_pregrasp_frame_need_and_check(monkeypatch):
    e = Reject("gripper_not_in_frame", f"{IS.SHELF_PREGRASP_FRAME_TAG} pre-grasp fingertip outside the manipulation image (5 %) at stand-off 0.45")
    assert IS.shelf_need_of(e) == "frame"
    assert IS.shelf_need_of(Reject("gripper_not_in_frame", "pre-grasp fingertip outside the manipulation image (5 %)")) is None
    s = SimpleNamespace(kit=SimpleNamespace(stats={}), camera_pose_at=lambda st: "cam")
    _bind(s, "shelf_pregrasp_frame_check")
    g = SimpleNamespace(pre_grasp="pre", grasp="g")
    s.require_in_frame = lambda cam, poses: None
    s.shelf_pregrasp_frame_check(_stance(), g)
    assert s.kit.stats == {"shelf_pregrasp_frame.checked": 1, "shelf_pregrasp_frame.passed": 1}

    def fails(cam, poses):
        assert cam == "cam" and poses == {"pre-grasp": "pre", "grasp": "g"}
        raise Reject("gripper_not_in_frame", "pre-grasp fingertip outside the manipulation image (5 %)")

    s.require_in_frame = fails
    with pytest.raises(Reject) as ei:
        s.shelf_pregrasp_frame_check(_stance(), g)
    assert IS.shelf_need_of(ei.value) == "frame" and "stand-off 0.45" in str(ei.value)
    assert s.kit.stats["shelf_pregrasp_frame.refused"] == 1


# ---------------------------------------------------------------- the blame line

def test_tracked_end_blame_switch(monkeypatch):
    monkeypatch.setenv(IK.TRACKED_END_BLAME_ENV, "1")
    assert IK.tracked_end_blame_on()
    monkeypatch.setenv(IK.TRACKED_END_BLAME_ENV, "0")
    assert not IK.tracked_end_blame_on()

