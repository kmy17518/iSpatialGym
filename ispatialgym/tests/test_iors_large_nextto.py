"""IORS nextto large-scale (owner 2026-10-12): the base push of a floor-standing target - the sampler mode's pure geometry
(``sampling/v2/tasks/iors_large_nextto.py``), its hook, and the program the sidecar expands to (``program_iors_large_nextto``; no
simulator)."""

from __future__ import annotations

import json
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


def test_knobs_default_unless_the_variant_sets_them():
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
    acts = expand(sc.variant, sc)
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
        expand(sc.variant, sc)
    sc = _sidecar()
    sc.contacts.push.extra["end_stance"] = "push_end"
    with pytest.raises(Exception, match="push_end"):
        expand(sc.variant, sc)


# ----------------------------------------------------------------------------------------------------------------
# two hands (owner 2026-10-12 ~20:17 item C)
# ----------------------------------------------------------------------------------------------------------------


def _cart_pieces(middle=None):
    """The cart of :func:`test_push_face_probe_on_convex_pieces` with slimmer posts (|y| in [0.185, 0.215]): a middle shelf across the
    +x face at 0.40-0.50 m, an open bay below it."""
    posts = [_box(sx * 0.18 - 0.015, sx * 0.18 + 0.015, sy * 0.2 - 0.015, sy * 0.2 + 0.015, 0.0, 0.85) for sx in (-1, 1) for sy in (-1, 1)]
    bottom = _box(-0.2, 0.2, -0.22, 0.22, 0.10, 0.14)
    return posts + [bottom] + (middle if middle is not None else [_box(-0.19, 0.19, -0.22, 0.22, 0.40, 0.50)])


def test_two_hand_spots_on_the_face():
    """The face probe moved along the face's tangent; two spots symmetric about the centre line on the middle shelf's edge (the open bay
    has none, a spacing wider than the face is skipped); spots recessed differently beyond ``push_hands_depth_tol_m`` are no pair."""
    pieces = _cart_pieces()
    lo, hi = th.tensor([-0.2, -0.22, 0.0]), th.tensor([0.2, 0.22, 0.85])
    f = NT.face_probe(pieces, "+x", lo, hi, 0.45, 0.03, 0.02, lateral=0.15)
    assert f is not None and f["t0"] == pytest.approx(0.15) and f["coord"] == pytest.approx(0.19, abs=1e-5) and f["spread"] == pytest.approx(0.0, abs=1e-5)
    assert NT.face_half_width("+x", lo, hi) == pytest.approx(0.22) and NT.face_half_width("-y", lo, hi) == pytest.approx(0.2)
    assert NT.face_outer("+x", lo, hi) == pytest.approx(0.2) and NT.face_outer("-y", lo, hi) == pytest.approx(0.22)
    pairs = NT.spot_pairs(pieces, "+x", lo, hi, [0.30, 0.45], [0.30, 0.26], 0.03, 0.02, 0.03, 0.15, 0.03)
    assert [(p["height"], p["spacing"]) for p in pairs] == [(0.45, 0.30)]
    a, b = pairs[0]["spots"]
    assert (a["t0"], b["t0"]) == pytest.approx((0.15, -0.15)) and a["coord"] == pytest.approx(0.19, abs=1e-5) and b["coord"] == pytest.approx(0.19, abs=1e-5)
    assert a["inset"] == pytest.approx(0.01, abs=1e-5) and a["height"] == 0.45
    assert th.allclose(NT.contact_local("+x", a["coord"], a["t0"], a["z"]), th.tensor([0.19, 0.15, 0.45]), atol=1e-5)
    assert NT.spot_pairs(pieces, "+x", lo, hi, [0.45], [0.42], 0.03, 0.02, 0.03, 0.15, 0.03) == []  # 0.21 + 0.03 off the centre > 0.22
    stepped = _cart_pieces([_box(-0.19, 0.15, 0.0, 0.22, 0.40, 0.50), _box(-0.19, 0.19, -0.22, 0.0, 0.40, 0.50)])
    assert NT.spot_pairs(stepped, "+x", lo, hi, [0.45], [0.26], 0.03, 0.02, 0.03, 0.15, 0.03) == []
    assert [p["spacing"] for p in NT.spot_pairs(stepped, "+x", lo, hi, [0.45], [0.26], 0.03, 0.02, 0.03, 0.15, 0.05)] == [0.26]


def test_each_hand_takes_the_spot_on_its_side_and_two_hands_are_asked_for():
    pts = [th.tensor([1.0, 0.15, 0.4]), th.tensor([1.0, -0.15, 0.4])]
    assert NT.spots_by_arm(pts, (1.0, 0.0)) == {"right": 1, "left": 0}  # facing +x the robot's left is +y
    assert NT.spots_by_arm(pts, (-1.0, 0.0)) == {"right": 0, "left": 1}
    assert NT.knob(None, "push_hands") == 1 and not NT.two_hands(None)
    assert NT.two_hands(SimpleNamespace(gp=SimpleNamespace(gen=lambda k: {"push_hands": 2}.get(k))))


def test_the_pair_chain_puts_the_second_hand_at_the_first_hand_s_pitch(monkeypatch):
    """Each chain of the first hand in turn; the second hand at its pitch, its wrist turn first; the first second-hand chain that passes
    ``program_check.second_hand_chain`` gives ``second`` and ``q_both``."""
    first = [{"pre": "pre_r", "contact": "c_r", "q": th.ones(28), "q_pre": th.zeros(28), "pitch_deg": p, "wrist": w} for p, w in ((0.0, 1.0), (20.0, -1.0))]
    monkeypatch.setattr(NT, "_hand_chains", lambda *a, **k: iter(first))
    tried = []

    def second(s, st, arm1, q_pre1, q_c1, arm2, pre, approach, target):
        a = C.approach_axis(pre[1])
        tried.append((arm2, round(math.degrees(math.asin(-float(a[2])))), round(float(C.closing_axis(pre[1])[1]))))
        ok = len(tried) == 3
        return {"ok": ok, "why": "" if ok else "approach: untrackable", "q_pre": th.full((28,), 2.0), "q": th.full((28,), 3.0)}

    monkeypatch.setattr(NT.PC, "second_hand_chain", second)
    monkeypatch.setenv("ISG_FINGERTIP_FRAME_CHECK", "off")
    s = SimpleNamespace(target=None)
    st = SimpleNamespace(stand_off=0.55, translate=0.75, tilt=-0.3)
    why = []
    out = NT._push_chain_pair(s, st, [("right", th.tensor([1.0, -0.15, 0.45])), ("left", th.tensor([1.0, 0.15, 0.45]))], th.tensor([1.0, 0.0]), 0.10, why)
    assert [(p, w) for _, p, w in tried] == [(0, 1), (0, -1), (20, -1)] and all(a == "left" for a, _, _ in tried)
    assert out["pitch_deg"] == 20.0 and out["second"]["arm"] == "left" and out["second"]["wrist"] == -1.0 and th.equal(out["q_both"], th.full((28,), 3.0))
    assert th.allclose(C.fingertip(out["second"]["contact"]), th.tensor([1.0, 0.15, 0.45]), atol=1e-6) and len(why) == 2


def test_the_two_hand_record_carries_both_hands():
    """``contacts.push`` keeps the first hand's poses and adds ``second_hand`` (the other arm's pre / init / goal = its contact moved the
    distance, its contact point in the world and the object), the spacing, the face centre between the spots, ``push_hands`` 2; the base
    stands centred (lateral offset 0); the summary names both arms."""
    from ispatialgym.sampling.v2.stances import SidecarDraft, Stance

    sd = SidecarDraft("inter_object_relation_satisfaction-nextto_large_scale-serving_cart-breakfast_table-floor", "inter_object_relation_satisfaction")
    gens = {"push_hands": 2}
    s = SimpleNamespace(tname=CART, pname=TABLE, sidecar=sd, gp=SimpleNamespace(gen=lambda k: gens.get(k), variant=sd.variant), stance_q=lambda st: th.zeros(28),
                        partner=SimpleNamespace(get_position_orientation=lambda: (th.tensor([3.0, 0.0, 0.4]), th.tensor([0.0, 0.0, 0.0, 1.0])), category="breakfast_table"),
                        tol=lambda k, v: v, choice=SimpleNamespace(models={"target": SimpleNamespace(category="serving_cart")}))
    d2 = th.tensor([1.0, 0.0])
    st = Stance("manipulation", (0.25, 0.0, 0.0), 0.75, -0.3, 0.55, None, None, 0.0, th.tensor([0.8, 0.0, 0.45]), ["right", "left"])
    end = Stance("placement", (2.05, 0.0, 0.0), 0.75, -0.3, 0.55, None, None, 0.0, None, ["right", "left"])
    init = Stance("init", (-1.0, 0.3, 0.0), 0.0, 0.0, 1.9)
    spot = {"right": {"coord": 0.19, "t0": 0.15, "z": 0.05, "inset": 0.01, "spread": 0.0, "height": 0.45, "relaxed": False},
            "left": {"coord": 0.19, "t0": -0.15, "z": 0.05, "inset": 0.012, "spread": 0.002, "height": 0.45, "relaxed": False}}
    points = {"right": th.tensor([0.81, -0.15, 0.45]), "left": th.tensor([0.81, 0.15, 0.45])}  # face -x of the cart at x 1.0, yaw 0
    poses = {a: NT.push_hand_poses(points[a], d2, 20.0, 1.0, 0.10) for a in points}
    chain = {"pre": poses["right"][0], "contact": poses["right"][1], "q": th.zeros(28), "q_pre": th.zeros(28), "pitch_deg": 20.0, "wrist": 1.0,
             "second": {"arm": "left", "pre": poses["left"][0], "contact": poses["left"][1], "pitch_deg": 20.0, "wrist": 1.0}}
    info = {"pair": {"spacing": 0.30}, "face": spot["right"], "point": points["right"], "arm": "right", "arms": ["right", "left"], "points": points, "spots": spot,
            "reach_margin": 0.2, "reach_margins": {"right": 0.2, "left": 0.22}}
    side = IL.aabb_edges(th.tensor([3.0, -0.5, 0.0]), th.tensor([4.5, 0.5, 0.75]))[1]
    pose = (th.tensor([1.0, 0.0, 0.4]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    d = NT._record(s, init, st, end, info, chain, side, "-x", 0.0, d2, pose, (th.tensor([2.8, 0.0, 0.4]), pose[1]), 1.8, 0.12, 0.3, True, {"order": "x"})
    push = sd.contacts["push"].to_sidecar()
    sec = push["second_hand"]
    assert push["arm"] == "right" and push["push_hands"] == 2 and push["hand_spacing_m"] == 0.3 and push["lateral_offset_m"] == 0.0
    assert sec["arm"] == "left" and sec["contact_point"] == pytest.approx([0.81, 0.15, 0.45]) and sec["contact_in_object"] == pytest.approx([-0.19, 0.15, 0.05], abs=1e-5)
    assert sec["init"]["position"] == pytest.approx(poses["left"][1][0].tolist(), abs=1e-6)
    assert sec["goal"]["position"] == pytest.approx((poses["left"][1][0] + th.tensor([1.8, 0.0, 0.0])).tolist(), abs=1e-5)
    assert push["face_centre_in_object"] == pytest.approx([-0.19, 0.0, 0.05], abs=1e-5) and push["contact_in_object"] == pytest.approx([-0.19, -0.15, 0.05], abs=1e-5)
    assert d.summary["arm"] == {"arm": "right", "arms": ["right", "left"]} and d.summary["nextto"]["push_hands"] == 2 and st.extras["push"]["hands"] == 2
    sc = Sidecar.from_dict({**json.load(open(FIXTURE)), "contacts": {"push": push}})
    assert expand(sc.variant, sc)[2].second.goal.position == pytest.approx(sec["goal"]["position"])


def test_the_two_hand_stance_stands_centred_behind_the_spots(monkeypatch):
    """With ``push_hands`` 2 and two spots on the face, the push stance stands ``push_stand_offs_m`` behind their midpoint facing the push
    (no lateral shift), solved for both arms, each hand on the spot on its side; the chain, the sweep (both arms), the push end, stance A
    and the leg follow; the record has both hands. Without spots the one hand's path runs and says why."""
    from ispatialgym.sampling.v2 import prompt0 as P0
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.v2.stances import SidecarDraft, Stance

    sd = SidecarDraft("inter_object_relation_satisfaction-nextto_large_scale-serving_cart-breakfast_table-floor", "inter_object_relation_satisfaction")
    lo, hi = th.tensor([-0.2, -0.22, 0.0]), th.tensor([0.2, 0.22, 0.85])
    pair = {"height": 0.45, "spacing": 0.30, "relaxed": False,
            "spots": [{"coord": 0.2, "t0": t0, "z": 0.45, "inset": 0.0, "spread": 0.0, "height": 0.45, "relaxed": False} for t0 in (0.15, -0.15)]}
    gens = {"push_hands": 2, "push_stand_offs_m": [0.55], "push_trunks": [[0.75, -0.3]], "push_reach_margin_m": [0.03, 0.45]}
    stances, swept = [], []
    s = SimpleNamespace(place={"local_lo": lo, "local_hi": hi, "pair_cache": {"-x": [pair]}, "face_cache": {"-x": []}}, gp=SimpleNamespace(gen=lambda k: gens.get(k), variant=sd.variant),
                        sidecar=sd, target="cart", tname=CART, pname=TABLE, stance_q=lambda st: th.zeros(28), tol=lambda k, v: v, _check_deadline=lambda msg: None,
                        reach_margin=lambda st, p, arm: 0.2, camera_pose_at=lambda st: None, record_leg_a_b=lambda init, st: {"order": "x"},
                        ikc=SimpleNamespace(stances_collision_free=lambda sts: (stances.extend(sts), th.ones(len(sts), dtype=th.bool))[1]),
                        partner=SimpleNamespace(get_position_orientation=lambda: (th.tensor([3.0, 0.0, 0.4]), th.tensor([0.0, 0.0, 0.0, 1.0])), category="breakfast_table"),
                        choice=SimpleNamespace(models={"target": SimpleNamespace(category="serving_cart")}))
    monkeypatch.setattr(TT, "offered_arms", lambda s: ["right", "left"])
    monkeypatch.setattr(P0, "base_ok", lambda s, xy: True)
    monkeypatch.setattr(P0, "in_frame", lambda s, cam, pts: True)
    monkeypatch.setattr(NT, "_stance_a", lambda s, rng, init_pose, yaw, goal_pose, d2: Stance("init", (-1.0, 0.0, 0.0), 0.0, 0.0, 1.9))
    monkeypatch.setattr(NT, "_sweep_ok", lambda s, st, end2d, arms, q, why: swept.append((list(arms), q)) or True)

    def pair_chain(s_, st, hands, d2, pre_m, why):
        poses = {a: NT.push_hand_poses(p, d2, 20.0, 1.0, pre_m) for a, p in hands}
        return {"pre": poses["right"][0], "contact": poses["right"][1], "q": th.zeros(28), "q_pre": th.zeros(28), "pitch_deg": 20.0, "wrist": 1.0, "q_both": th.ones(28),
                "second": {"arm": "left", "pre": poses["left"][0], "contact": poses["left"][1], "pitch_deg": 20.0, "wrist": 1.0, "hands": hands}}

    monkeypatch.setattr(NT, "_push_chain_pair", pair_chain)
    side = IL.aabb_edges(th.tensor([3.0, -0.5, 0.0]), th.tensor([4.5, 0.5, 0.75]))[1]
    d2 = th.tensor([1.0, 0.0])
    pose = (th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))  # the cart's root at x 1, its -x face (outward -x) toward the robot
    d = NT._push_and_stances(s, None, side, "-x", 0.0, d2, pose, (th.tensor([2.8, 0.0, 0.0]), pose[1]), 1.8, 0.12, 0.3, True, {})
    st = stances[0]
    assert st.arms == ["right", "left"] and st.lateral == 0.0 and st.base_pose2d == pytest.approx((0.8 - 0.55, 0.0, 0.0), abs=1e-6)
    assert len(swept) == 1 and swept[0][0] == ["right", "left"] and th.equal(swept[0][1], th.ones(28))
    assert sd.stance("placement").base_pose2d == pytest.approx((0.25 + 1.8, 0.0, 0.0), abs=1e-6)
    push = sd.contacts["push"].to_sidecar()
    assert push["contact_point"] == pytest.approx([0.8, -0.15, 0.45], abs=1e-6) and push["second_hand"]["contact_point"] == pytest.approx([0.8, 0.15, 0.45], abs=1e-6)
    assert push["push_hands"] == 2 and d.summary["arm"]["arms"] == ["right", "left"] and push["hand_spacing_m"] == 0.3
    s.place["pair_cache"]["-x"] = []
    s.place["face_cache"]["-x"] = [pair["spots"][0]]
    one = []
    monkeypatch.setattr(NT, "_push_chain", lambda s_, st, arm, point, d2, pre_m, why: one.append(st.arms) or None)
    stats: dict = {}
    with pytest.raises(Exception, match="none passes the IK"):
        NT._push_and_stances(s, None, side, "-x", 0.0, d2, pose, (th.tensor([2.8, 0.0, 0.0]), pose[1]), 1.8, 0.12, 0.3, True, stats)
    assert stats["two_hand_no_spots"] == 1 and one and all(len(a) == 1 for a in one)


def _two_hand_sidecar(arm2: str = "left") -> Sidecar:
    """The fixture with a second hand: the left hand 0.30 m left of the right one (the push runs along +x), the same orientation."""
    d = json.load(open(FIXTURE))
    push = d["contacts"]["push"]

    def at_y(p, y):
        return {"position": [p["position"][0], y, p["position"][2]], "orientation": list(p["orientation"])}

    push.update({"push_hands": 2, "hand_spacing_m": 0.30, "lateral_offset_m": 0.0,
                 "second_hand": {"arm": arm2, "pre": at_y(push["pre"], 2.30), "init": at_y(push["init"], 2.30), "goal": at_y(push["goal"], 2.30),
                                 "contact_point": [4.8, 2.30, 0.452], "contact_in_object": [-0.219, -0.30, 0.0], "pitch_deg": 20.0, "wrist_sign": 1.0}})
    return Sidecar.from_dict(d)


def test_two_hand_program_pushes_with_both_arms_and_retreats_each():
    """The base push carries both arms (the second hand's pre-contact / approach / contact in its targets); each hand retreats from where
    the push leaves it - the first hand first, the second from its own recorded end; the annotation names both hands."""
    sc = _two_hand_sidecar()
    assert sc.problems() == []
    acts = expand(sc.variant, sc)
    assert [a.name for a in acts] == ["navigate", "transition", "base_push", "retreat", "retreat"]
    push, r1, r2 = acts[2:]
    assert push.arms == ("right", "left") and push.objects == (CART,) and push.second.arm == "left" and push.second.distance_m == pytest.approx(1.8)
    t = push.targets(sc, None)
    assert set(t) == set(push.TARGET_KEYS) and t["second"]["arm"] == "left" and len(t["second"]["approach"]) == A.N_WAYPOINTS
    assert t["second"]["approach"][-1].position == pytest.approx(t["second"]["contact"].position, abs=1e-6) and t["second"]["contact"].position[1] == pytest.approx(2.30)
    assert (r1.arm, r2.arm) == ("right", "left") and r1.after is push and r2.after.arms == ("left",) and r2.after.second is None
    assert push.end_pose(sc, None) == sc.contacts.push.goal and r2.after.end_pose(sc, None) == push.second.goal
    assert r2.targets(sc, None)["retreat"][-1].position[1] == pytest.approx(2.30, abs=1e-5)
    here = {"right": (th.tensor([5.9, 2.0, 0.5]), contact_q(sc)), "left": (th.tensor([5.9, 2.31, 0.5]), contact_q(sc))}
    st = SimpleNamespace(eef_pose=lambda arm: here[arm])
    assert r2.targets(sc, st)["retreat"][-1].position[1] == pytest.approx(2.31, abs=1e-5)
    fine, numbers = push.annotate(sc)
    assert fine.endswith("by driving the base, both closed hands on it") and "180 cm" in numbers
    with pytest.raises(Exception, match="second must be 'left'"):
        expand(sc.variant, _two_hand_sidecar("right"))
    with pytest.raises(Exception, match="no left hand"):
        _sidecar_one = expand(_sidecar().variant, _sidecar())[2]
        _sidecar_one.hand("left")
