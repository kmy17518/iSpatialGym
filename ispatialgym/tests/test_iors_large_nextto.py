"""IORS nextto large-scale (owner 2026-10-12): the base push of a floor-standing target - the sampler mode's pure geometry
(``sampling/v2/tasks/iors_large_nextto.py``), its hook, and the program the sidecar expands to (``program_iors_large_nextto``; no
simulator)."""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import actions as A
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2.tasks import iors_large as IL
from ispatialgym.sampling.v2.tasks import iors_large_nextto as NT

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "sidecars", "inter_object_relation_satisfaction-nextto_large_scale.json")
CART = "serving_cart.n.01_1"
TABLE = "breakfast_table.n.01_1"


def _box(x0, x1, y0, y1, z0, z1) -> th.Tensor:
    return th.tensor([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=th.float32)


# ----------------------------------------------------------------------------------------------------------------
# sampler geometry
# ----------------------------------------------------------------------------------------------------------------


def test_nextto_bound_and_gap_follow_omnigibson():
    """NextTo's bound is mean(dims_a + dims_b) / 6 = sum of both AABBs' extents / 18; the gap is the norm of the per-axis gaps."""
    lo_a, hi_a = th.tensor([0.0, 0.0, 0.0]), th.tensor([0.4, 0.44, 0.85])
    lo_b, hi_b = th.tensor([1.0, -0.5, 0.0]), th.tensor([2.5, 0.5, 0.75])
    assert NT.nextto_bound(lo_a, hi_a, lo_b, hi_b) == pytest.approx((0.4 + 0.44 + 0.85 + 1.5 + 1.0 + 0.75) / 18.0, abs=1e-6)
    assert NT.aabb_gap(lo_a, hi_a, lo_b, hi_b) == pytest.approx(0.6, abs=1e-6)
    assert NT.aabb_gap(lo_a + th.tensor([0.0, 1.2, 0.0]), hi_a + th.tensor([0.0, 1.2, 0.0]), lo_b, hi_b) == pytest.approx(math.hypot(0.6, 0.7), abs=1e-6)
    assert NT.aabb_gap(lo_b, hi_b, lo_b, hi_b) == 0.0


def test_face_yaw_turns_the_face_against_the_push():
    """The robot stands outside the pushed face: its outward normal ends up opposite the push direction, whatever the face."""
    for d in ((1.0, 0.0), (0.0, -1.0), (math.cos(0.7), math.sin(0.7))):
        for face, n in NT.FACES.items():
            yaw = NT.face_yaw(face, d)
            nw = NT.rot2(yaw) @ th.tensor(n)
            assert th.allclose(nw, -th.tensor(d), atol=1e-6), (face, d)
    lo, hi = th.tensor([-0.2, -0.22, -0.4]), th.tensor([0.19, 0.218, 0.45])
    assert NT.face_order(lo, hi)[:2] == ["+y", "-y"] and NT.depth_along("+y", lo, hi) == pytest.approx(0.438, abs=1e-6)


def test_goal_footprint_stands_the_gap_off_the_side():
    """The goal root puts the target's footprint AABB ``gap`` outside the partner side (any yaw), centred ``lateral`` along it."""
    plo, phi = th.tensor([1.0, -0.5, 0.0]), th.tensor([2.5, 0.5, 0.75])
    lo, hi = th.tensor([-0.18, -0.24, -0.4]), th.tensor([0.21, 0.20, 0.45])
    for side in IL.aabb_edges(plo, phi):
        for yaw in (0.0, 0.4, -2.0):
            xy = NT.goal_root_xy(side, lo, hi, yaw, 0.12, 0.2)
            l2, h2 = NT.footprint_aabb(lo, hi, xy, yaw)
            wlo, whi = th.tensor([float(l2[0]), float(l2[1]), 0.0]), th.tensor([float(h2[0]), float(h2[1]), 0.85])
            assert NT.aabb_gap(wlo, whi, plo, phi) == pytest.approx(0.12, abs=1e-5), (side.name, yaw)
            ctr = 0.5 * (l2 + h2)
            assert float((ctr - side.mid) @ side.tangent) == pytest.approx(0.2, abs=1e-5)
    side = IL.aabb_edges(plo, phi)[1]  # "-x": the target west of the table, pushed east
    d = NT.push_direction(side, 0.0)
    assert th.allclose(d, th.tensor([1.0, 0.0]), atol=1e-6) and th.allclose(NT.push_direction(side, 90.0), th.tensor([0.0, 1.0]), atol=1e-6)
    xs = NT.sweep_xys(th.tensor([0.0, 0.0]), th.tensor([0.31, 0.0]), 0.1)
    assert len(xs) == 5 and float(xs[-1][0]) == pytest.approx(0.31) and float(xs[0][0]) == 0.0


def test_footprint_blocked_by_obstacle_boxes():
    corners = NT.footprint(th.tensor([-0.2, -0.2, 0.0]), th.tensor([0.2, 0.2, 0.8]), (0.0, 0.0), 0.3)
    boxes = [("chair", th.tensor([0.5, 0.5]), th.tensor([0.9, 0.9])), ("cabinet", th.tensor([0.21, -0.1]), th.tensor([0.6, 0.1]))]
    assert NT.footprint_blocked(corners, boxes) == "cabinet"
    assert NT.footprint_blocked(corners, boxes[:1]) is None
    assert NT.footprint_blocked(corners, boxes[:1], margin=0.5) == "chair"


def test_push_face_probe_on_convex_pieces():
    """A cart-like target (four posts, a bottom and a middle shelf, an open bay between them): the face of +x at the middle shelf's
    height is the shelf's outer edge (flat across the probes), between the shelves only the posts' material is met off the centre
    line - no face; a face recessed beyond ``face_max_inset_m`` would be rejected by the caller."""
    posts = [_box(sx * 0.18 - 0.02, sx * 0.18 + 0.02, sy * 0.2 - 0.02, sy * 0.2 + 0.02, 0.0, 0.85) for sx in (-1, 1) for sy in (-1, 1)]
    bottom = _box(-0.2, 0.2, -0.22, 0.22, 0.10, 0.14)
    middle = _box(-0.19, 0.19, -0.22, 0.22, 0.40, 0.50)
    pieces = posts + [bottom, middle]
    lo, hi = th.tensor([-0.2, -0.22, 0.0]), th.tensor([0.2, 0.22, 0.85])
    f = NT.face_probe(pieces, "+x", lo, hi, 0.45, 0.03, 0.02)
    assert f is not None and f["coord"] == pytest.approx(0.19, abs=1e-5) and f["spread"] == pytest.approx(0.0, abs=1e-5) and f["t0"] == pytest.approx(0.0, abs=1e-6)
    assert NT.face_probe(pieces, "+x", lo, hi, 0.30, 0.03, 0.02) is None  # the open bay: nothing on the centre line
    g = NT.face_probe(pieces, "-y", lo, hi, 0.12, 0.03, 0.01)
    assert g is not None and g["coord"] == pytest.approx(0.22, abs=1e-5)
    p = NT.contact_local("-y", g["coord"], g["t0"], g["z"])
    assert th.allclose(p, th.tensor([0.0, -0.22, 0.12]), atol=1e-5)
    p2 = NT.contact_local("+x", f["coord"], f["t0"], f["z"])
    assert th.allclose(p2, th.tensor([0.19, 0.0, 0.45]), atol=1e-5)


def test_push_hand_poses_put_the_fingertips_on_the_face():
    """The closed hand's contact puts the fingertips on the face point, approach along the push tilted down by the pitch; the
    pre-contact is ``pre_m`` back along the horizontal push (the guarded approach runs along it); the two wrist turns swap the closing axis."""
    point = th.tensor([2.0, 1.0, 0.45])
    for pitch in (0.0, 35.0):
        pre, contact = NT.push_hand_poses(point, (0.0, 1.0), pitch, 1.0, 0.1)
        assert th.allclose(C.fingertip(contact), point, atol=1e-6)
        a = C.approach_axis(contact[1])
        assert th.allclose(a, th.tensor([0.0, math.cos(math.radians(pitch)), -math.sin(math.radians(pitch))]), atol=1e-6)
        assert th.allclose(contact[0] - pre[0], th.tensor([0.0, 0.1, 0.0]), atol=1e-6)
        _, twin = NT.push_hand_poses(point, (0.0, 1.0), pitch, -1.0, 0.1)
        assert th.allclose(C.closing_axis(twin[1]), -C.closing_axis(contact[1]), atol=1e-6)
    x, y, yaw = NT.push_stance_pose(point[:2], (0.0, 1.0), 0.55, 0.15)
    assert (x, y) == pytest.approx((2.0 - 0.15, 1.0 - 0.55), abs=1e-6) and yaw == pytest.approx(math.pi / 2)
    assert NT.arm_lateral(0.15, "left") == -0.15 and NT.arm_lateral(0.15, "right") == 0.15
    assert NT.shifted_pose2d((1.0, 2.0, 0.3), (0.0, -1.0), 1.5) == pytest.approx((1.0, 0.5, 0.3))


def test_knobs_default_unless_the_alias_sets_them():
    s = SimpleNamespace(gp=SimpleNamespace(gen=lambda k: {"drift_tol_m": 0.2}.get(k)))
    assert NT.knob(s, "drift_tol_m") == 0.2 and NT.knob(s, "push_distance_m") == [1.5, 2.5] and NT.knob(None, "push_pre_m") == 0.10


def test_the_recorded_turn_limit_defaults_to_the_oracle_s():
    from ispatialgym.oracle import execute as EX

    assert NT.knob(None, "yaw_tol_rad") == EX.BASE_PUSH_DEFAULTS["yaw_tol_rad"] == 0.61
    s = SimpleNamespace(gp=SimpleNamespace(gen=lambda k: {"yaw_tol_rad": 0.5}.get(k)))
    assert NT.knob(s, "yaw_tol_rad") == 0.5


def test_the_nextto_hook_returns_the_mode():
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    assert s._sample_nextto() is NT.NextToMode
    s.relation, s.template_facts, s.template_meta = "nextto", {}, {}
    s._adopt_mode(NT.NextToMode)
    assert isinstance(s, NT.NextToMode) and isinstance(s, IL.IORSLarge) and s.record_constraints(None) == []
    assert type(s).sample_instance is NT.NextToMode.sample_instance and type(s).goal is NT.NextToMode.goal


def test_goal_pose_follows_the_settled_init():
    d = SimpleNamespace(summary={"nextto": {"direction": [0.0, 1.0, 0.0], "push_distance_m": 1.8}},
                        object_poses={CART: (th.tensor([1.0, 2.0, 0.43]), th.tensor([0.0, 0.0, 0.0, 1.0]))})
    p, q = NT.goal_pose_of(d, CART)
    assert th.allclose(p, th.tensor([1.0, 3.8, 0.43])) and th.allclose(q, th.tensor([0.0, 0.0, 0.0, 1.0]))


# ----------------------------------------------------------------------------------------------------------------
# oracle: program, targets, annotation
# ----------------------------------------------------------------------------------------------------------------


def _sidecar() -> Sidecar:
    return Sidecar.load(FIXTURE)


def test_program_is_travel_base_push_retreat():
    sc = _sidecar()
    assert sc.problems() == []
    acts = expand(sc.alias, sc)
    assert [a.name for a in acts] == ["navigate", "transition", "base_push", "retreat"]
    push, retreat = acts[2], acts[3]
    assert isinstance(push, A.BasePush) and push.partner == TABLE and push.end_stance == "placement" and push.arms == ("right",) and push.objects == (CART,)
    assert retreat.after is push and retreat.distance == A.RETREAT_DIST
    t = push.targets(sc, None)
    assert set(t) == set(push.TARGET_KEYS)
    assert t["approach"][-1].position == pytest.approx(t["contact"].position, abs=1e-6) and t["base_path"][0] == pytest.approx(sc.stance("manipulation").base_pose2d)
    assert t["base_path"][1] == pytest.approx(sc.stance("placement").base_pose2d) and t["distance_m"] == pytest.approx(1.8)
    assert push.end_pose(sc, None) == sc.contacts.push.goal  # no live hand: the recorded end of the push
    fine, numbers = push.annotate(sc)
    assert fine == "push the serving cart forward across the floor next to the dining table by driving the base, the closed right hand on it"
    assert "180 cm" in numbers and "45 cm height" in numbers
    # the retreat leaves from where the hand is when the state can tell (the push may stop before the recorded end)
    here = (th.tensor([5.9, 2.0, 0.5]), contact_q(sc))
    st = SimpleNamespace(eef_pose=lambda arm: here)
    r = retreat.targets(sc, st)["retreat"]
    assert r[-1].position[0] == pytest.approx(5.9 - 0.10 * math.cos(math.radians(20)), abs=1e-5)


def contact_q(sc) -> th.Tensor:
    return th.tensor(sc.contacts.push.init.orientation)


def test_program_refuses_a_push_without_the_base_push_record():
    sc = _sidecar()
    sc.contacts.push.extra["kind"] = "push"
    with pytest.raises(Exception, match="base push"):
        expand(sc.alias, sc)
    sc = _sidecar()
    sc.contacts.push.extra["end_stance"] = "push_end"
    with pytest.raises(Exception, match="push_end"):
        expand(sc.alias, sc)
