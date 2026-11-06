"""Touching small scale, ``generation.push_height_sweep`` (oracle-fix 2026-11-06, owner on the touching#04 spray-bottle demo: "for small
scale touching, can we make the push push the most reachable center bottom part of the target object so it doesn't wobble so much"):
every push class at contact heights from the lowest that clears the support up to half the target's height, lowest first, on the line
through the target's footprint centre at the hull's cut surface; the hand checked against the hull instead of its box; the raised blade
only for a target that is not tall; the push stance at the lateral whose feasible contact is lowest. ``IORSSmall`` methods run on
stand-ins (a 27 cm spray-bottle hull: body, neck and a trigger head over the pushed side); no simulator is launched."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks import iors_small as IS
from ispatialgym.tests import test_iors_small_lift as LIFT

TOUCHING = "inter_object_relation_satisfaction-touching_small_scale"
#: the bottle stands at the place (0.70, 0.0) on a 0.81 m table; the push goes along +y (the edge tangent), into its -y face
PLACE = (th.tensor([0.70, 0.0, 0.81]), C.IDENTITY_QUAT.clone())
D = th.tensor([0.0, 1.0, 0.0])
H = 0.273


def _circle(r, z, n=32):
    return [[r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n), z] for k in range(n)]


def _bottle() -> th.Tensor:
    """A 27.3 cm spray bottle in its own frame (bottom at z 0): an 8 cm body to 0.20 m, a 3 cm neck to 0.23 m and the trigger head
    0.22-0.273 m reaching 6.5 cm to -y - over the pushed side, where the whole hull's outermost point is."""
    head = [[x, y, z] for x in (-0.012, 0.012) for y in (-0.065, 0.02) for z in (0.22, H)]
    return th.tensor(_circle(0.04, 0.0) + _circle(0.04, 0.20) + _circle(0.015, 0.20) + _circle(0.015, 0.23) + head, dtype=th.float32)


def _sampler(gen=None, hull=None):
    s = LIFT._sampler("touching", gen={"push_height_sweep": True, **(gen or {})})
    s.hull = {"target": _bottle() if hull is None else hull, "partner": LIFT._box(*LIFT.PARTNER)}
    return s


def _candidates(s, pose=PLACE):
    hull_w = G.local_points_world(s.hull["target"], pose)
    face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
    return list(s.touching_push_candidates(LIFT._stance(), IS.push_face_point(hull_w, D), face_z, D, t_hull_w=hull_w)), hull_w


# ---------------------------------------------------------------------------------------------------------------- the knob


def test_the_sweep_is_the_touching_default_and_false_keeps_todays_classes():
    assert GD.defaults_for(TOUCHING)["push_height_sweep"] is True
    assert GD.defaults_for(TOUCHING)["push_height_step_m"] == 0.01 and GD.defaults_for(TOUCHING)["push_max_height_fraction"] == 0.5
    s = LIFT._sampler("touching")
    assert s.push_height_sweep_on() is False  # no subtask, no key: off
    s.subtask = TOUCHING
    assert s.push_height_sweep_on() is True  # a counted alias without the key gets the default
    s.spec.generation["push_height_sweep"] = False
    assert s.push_height_sweep_on() is False
    shelf = _sampler()
    shelf.shelf_start = {"lean_side": 1.0, "lean_deg": 10.0}
    assert shelf.push_height_sweep_on() is False  # the leaning book of the shelf layout keeps today's push
    # today's generator when the caller passes no hull: the mid-height classes in their order
    s = _sampler()
    hull_w = G.local_points_world(s.hull["target"], PLACE)
    old = [k for k, _, _ in s.touching_push_candidates(LIFT._stance(), IS.push_face_point(hull_w, D), (0.81, 0.81 + H), D)]
    assert old[:5] == ["push_tip_h-", "push_tip_h", "push_tip_v-", "push_h", "push_v"] and "push_blade_0_high" in old


# ---------------------------------------------------------------------------------------------------------------- the candidates


def test_candidates_rise_from_the_lowest_height_on_the_footprint_centre_line():
    s = _sampler()
    cands, hull_w = _candidates(s)
    hs = [c[2]["contact_height_m"] for c in cands]
    assert hs == sorted(hs) and hs[0] <= 0.5 * H
    assert max(hs) <= 0.5 * H + 1e-6  # nothing above half the height: the raised blade is not offered to the tall bottle
    assert not any(k.startswith("push_blade_0_high") for k, _, _ in cands)
    # the lowest: the vertical blade's tips 5 mm above the table, its two wrist turns together, then the leaning blades at 2.5 cm and
    # the horizontal fingertip push at 5 cm
    assert [c[0] for c in cands[:2]] == ["push_blade_0-", "push_blade_0"] and hs[0] == pytest.approx(0.005)
    assert min(c[2]["contact_height_m"] for c in cands if c[0].startswith("push_blade_lean")) == pytest.approx(0.025)
    assert min(c[2]["contact_height_m"] for c in cands if c[0].startswith("push_tip_h")) == pytest.approx(0.05)
    assert min(c[2]["contact_height_m"] for c in cands if c[0] == "push_tip_v-") == pytest.approx(0.062)
    assert {c[0].rstrip("-") for c in cands} == {"push_tip_h", "push_tip_v", "push_h", "push_v", "push_blade_lean20", "push_blade_lean12", "push_blade_lean8",
                                                  "push_blade_0"}  # every class
    # ties keep today's class order: at 5 cm the fingertip push before the pad push
    at5 = [c[0] for c in cands if abs(c[2]["contact_height_m"] - 0.05) < 1e-6]
    assert at5.index("push_tip_h-") < at5.index("push_h")
    for kind, (o, q), ex in cands:
        cp = ex["contact_point"]
        # the line through the footprint centre (the body's base, x 0.70), not the hull points' mean pulled toward the head
        assert cp[0] == pytest.approx(0.70, abs=1e-4), kind
        # on the hull's cut surface on the pushed side: the slope from the 4 cm body at the bottom to the head's -6.5 cm at 0.22 m,
        # the outermost within +-1 cm of the height (1 mm), never the head's own -6.5 cm the old mid-height point took
        h = ex["contact_height_m"]
        want = -0.04 - 0.025 * min(h + 0.01, 0.22) / 0.22
        assert cp[1] == pytest.approx(want, abs=1.5e-3), (kind, h)
        assert cp[2] == pytest.approx(0.81 + h, abs=1e-5)
    old = IS.push_face_point(hull_w, D)
    assert float(old[1]) == pytest.approx(-0.065, abs=1e-4)  # the whole hull's outermost point: the trigger head


def test_the_fingertips_stop_short_of_the_contact_and_the_blade_stands_off_the_cut():
    s = _sampler()
    cands, hull_w = _candidates(s)
    tip = next(c for c in cands if c[0] == "push_tip_h-")
    o, q = tip[1]
    ft = C.fingertip((o, q))
    assert th.allclose(C.approach_axis(q), D, atol=1e-5)
    assert float(ft[1]) == pytest.approx(tip[2]["contact_point"][1] - 0.005, abs=1e-4)  # stop_clearance_m before the face
    blade = cands[0]
    o, q = blade[1]
    assert th.allclose(C.approach_axis(q), th.tensor([0.0, 0.0, -1.0]), atol=1e-5)
    assert float(C.fingertip((o, q))[2]) == pytest.approx(0.815, abs=1e-5) and blade[2]["fingertip_z"] == pytest.approx(0.815)
    # the 3-7 cm finger section (half 18 mm) stands clear of the slope it faces (the cut 7 cm up is 8 mm prouder than at the tips)
    face_hi = -0.04 - 0.025 * 0.075 / 0.22
    assert float(o[1]) + 0.018 <= face_hi - 0.005 + 1e-4
    assert blade[2]["palm_clear"] is True  # recorded as checked: the oracle does not raise it


def test_the_hand_is_checked_against_the_hull_not_its_box():
    s = _sampler()
    cands, _ = _candidates(s)
    # the vertical blade beside the 27 cm bottle: its knuckle / gripper body reach into the body above the fingers
    bad = s.hand_overlap(cands[0][1], "target", PLACE, against="hull")
    assert bad is not None and "from the hull" in bad
    # the horizontal fingertip push: the hand trails behind the tips, clear of the hull
    tip = next(c for c in cands if c[0] == "push_tip_h-")
    assert s.hand_overlap(tip[1], "target", PLACE, against="hull") is None
    # the box of a round body stands proud of it on the diagonals (touching#04: the bottle at 48.6 deg yaw)
    sec = IS.HullSections(G.local_points_world(_bottle(), PLACE))
    p = th.tensor([[0.70 + 0.05 * math.cos(math.radians(45)), 0.05 * math.sin(math.radians(45)), 0.81 + 0.10]])
    assert float(sec.distance(p)[0]) == pytest.approx(0.01, abs=1e-3)
    lo, hi = _bottle().min(dim=0).values, _bottle().max(dim=0).values
    q = p[0] - PLACE[0]
    assert float(th.linalg.norm(th.clamp(lo - q, min=0) + th.clamp(q - hi, min=0))) == 0.0  # inside the box


def test_hull_distance_is_exact_off_faces_edges_and_corners():
    cube = th.tensor([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    sec = IS.HullSections(cube)
    pts = th.tensor([[1.5, 0.5, 0.5], [1.5, 1.5, 0.5], [1.5, 1.5, 1.5], [0.5, 0.5, 0.5], [0.5, -0.2, 0.5]])
    assert sec.distance(pts).tolist() == pytest.approx([0.5, math.sqrt(0.5), math.sqrt(0.75), 0.0, 0.2], abs=1e-5)
    cut = sec.cut(0.3)
    assert len(cut) == 4 and float(IS.polygon_centroid(cut)[0]) == pytest.approx(0.5)


def test_the_raised_blade_is_the_last_resort_of_a_squat_target_only():
    squat = LIFT._box(0.12, 0.12, 0.10)  # height / smallest side 0.83
    s = _sampler(hull=squat)
    cands, _ = _candidates(s)
    assert [c[0] for c in cands[-2:]] == ["push_blade_0_high-", "push_blade_0_high"]
    assert cands[-1][2]["push_height_sweep"] == "raised_last" and cands[-1][2]["contact_height_m"] == pytest.approx(0.055)
    assert all(c[2]["contact_height_m"] <= 0.05 + 1e-6 for c in cands[:-2])
    # the contact of a box face: its plane, on the centre line
    assert all(c[2]["contact_point"][1] == pytest.approx(-0.06, abs=1e-5) for c in cands[:-2])
    # pushed lower than half its height, the cap
    s2 = _sampler(gen={"push_max_height_fraction": 0.25}, hull=squat)
    assert max(c[2]["contact_height_m"] for c in _candidates(s2)[0][:-2]) <= 0.025 + 1e-6


def test_the_choice_takes_the_height_range_and_labels_the_height(monkeypatch):
    s = _sampler()
    s.ikc = SimpleNamespace(q_locked=lambda st: None)
    seen = {}

    def choose_first(ikc, arm, stream, **kw):
        cands = list(stream)
        seen.update(heights=[c[2]["contact_height_m"] for c in cands], label=kw["label"](cands[0]))
        return None, None, []

    monkeypatch.setattr(IS.O, "choose_first", choose_first)
    s._touching_push_choice(LIFT._stance(), PLACE, IS.push_face_point(G.local_points_world(s.hull["target"], PLACE), D), (0.81, 0.81 + H), D, 0.08, [],
                            height_range=(None, 0.03))
    assert max(seen["heights"]) <= 0.03 + 1e-6 and seen["label"] == "push_blade_0-@0.5cm"


# ---------------------------------------------------------------------------------------------------------------- the push stance


def test_the_push_stance_takes_the_lateral_with_the_lowest_feasible_contact(monkeypatch):
    s, d, g, calls = LIFT._push_sampler(monkeypatch, lift_ok=True, shift=0.04)
    s.spec.generation["push_height_sweep"] = True
    stances = {lat: LIFT._stance("push", 0.3 + lat) for lat in (-0.12, -0.16, -0.20, -0.08)}
    for lat, st in stances.items():
        st.extras["lateral_shift_m"] = 0.3 + lat
    s.push_stance_for = lambda manip, t_place, lateral=None: stances[lateral]
    lowest = {-0.12: 0.08, -0.16: 0.05, -0.20: 0.05, -0.08: None}  # the feasible contact per lateral (None: no class)
    asked = []

    def feasible(probe, t_place, p_pose, with_brace=False, height_range=None):
        lat = next(k for k, v in stances.items() if v is probe)
        asked.append((lat, with_brace, height_range))
        h = lowest[lat]
        if h is None or (height_range is not None and height_range[1] is not None and h > height_range[1] + 1e-9):
            return None
        s._push_chain_height = h
        return "push_tip_h-" if not with_brace else ("push_tip_h-+brace_tip_h" if lat == -0.20 else None)

    s.push_chain_feasible = feasible
    s._brace_gate = {"ok": True}
    st = s.push_stance(d, LIFT._stance(), g)
    # -0.16 and -0.20 tie at 5 cm; -0.20 also takes a brace there
    assert st is stances[-0.20] and d.summary["touching"]["push_stance_lowest_contact_m"] == pytest.approx(0.05)
    caps = {lat: hr for lat, wb, hr in asked if not wb}
    assert caps[-0.12] is None and caps[-0.16] == (None, 0.08) and caps[-0.20] == (None, 0.05) and caps[-0.08] == (None, 0.05)
    assert all(hr == (lowest[lat], lowest[lat]) for lat, wb, hr in asked if wb)  # the brace joins the push at its height
    assert any("@ 5.0 cm" in t for t in d.summary["touching"]["push_stance_search"])
    # no lateral with a feasible contact: contact_unreachable (another stance is tried)
    s.push_chain_feasible = lambda *a, **k: None
    s._push_chain_fails = []
    with pytest.raises(Reject) as e:
        s.push_stance(d, LIFT._stance(), g)
    assert e.value.reason == "contact_unreachable"
