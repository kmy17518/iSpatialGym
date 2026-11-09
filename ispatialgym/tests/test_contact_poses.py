"""i7-contact 2026-11-07: the touching push / brace pose rules moved to ``sampling/v2/contact_poses.py`` (shared by ``IORSSmall`` and the contact
reach maps). The sampler's record builders keep producing the records they produced at 63925048 (``data/contact_poses_golden.json``, the
scenarios of ``contact_poses_scenarios``), the module functions produce the same lists from a plain generation source (:class:`PoseGen`,
the map builder's), and the executed-contact helpers are the oracle's rules (``execute.push_lowered_pre``). No simulator."""

from __future__ import annotations

import json
import math

import numpy as np

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.v2 import contact_poses as CPZ
from ispatialgym.tests import contact_poses_scenarios as CS

TOL = 2e-5


def _golden():
    with open(CS.GOLDEN) as f:
        return json.load(f)


def _close(a, b, path="") -> None:
    """Recursive equality with a float tolerance (the golden values are rounded to 1e-6)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and sorted(a) == sorted(b), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: length {len(a)} != {len(b) if isinstance(b, list) else b}"
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a is not None and b is not None and abs(float(a) - float(b)) <= TOL, f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


@pytest.mark.parametrize("name", [n for n, _ in CS.push_scenarios()])
def test_the_push_record_builders_produce_the_golden_records(name):
    fn = dict(CS.push_scenarios())[name]
    got = [[k, CS._pose(p), CS._jsonable(e)] for k, p, e in fn()]
    want = _golden()["push"][name]
    assert [g[0] for g in got] == [w[0] for w in want]  # the classes and their order
    _close(want, got, name)


@pytest.mark.parametrize("name", [n for n, _ in CS.brace_scenarios()])
def test_the_brace_record_builders_produce_the_golden_records(name):
    fn = dict(CS.brace_scenarios())[name]
    got = [[k, CS._pose(pre), CS._pose(c), CS._jsonable(e)] for pre, c, k, e in fn()]
    want = _golden()["brace"][name]
    assert [g[0] for g in got] == [w[0] for w in want]
    _close(want, got, name)


def test_the_golden_lists_cover_every_family_of_classes():
    g = _golden()
    kinds = {k for lst in g["push"].values() for k, *_ in lst} | {k for lst in g["brace"].values() for k, *_ in lst}
    for prefix in ("push_tip_h", "push_tip_v-", "push_h", "push_v", "push_blade_0", "push_blade_lean", "push_blade_0_high", "push_open_v-", "push_open_h",
                   "push_blade_30", "brace_wide_h-", "brace_tip_h", "brace_tip_h-_p15", "blade_0", "blade_0_high", "left_v", "left_h", "brace_wide_h"):
        assert any(k.startswith(prefix) for k in kinds), prefix


def test_the_module_rules_on_a_plain_generation_source_are_the_samplers():
    """The map builder's path: :class:`PoseGen` (a dict of knobs + the subtask's generation_defaults) and the map frame give the sampler's
    records for the same knobs - here the swept pushes and the braces of two scenarios."""
    from ispatialgym.tests import test_iors_small_lift as LIFT

    cereal = LIFT._box(0.1567, 0.0427, 0.2182)
    s = CS._push_sampler({"push_height_sweep": True, "push_raised_max_aspect": 1.5}, cereal)
    gs = CPZ.PoseGen({"push_height_sweep": True, "push_raised_max_aspect": 1.5}, subtask=CS.TOUCHING)
    hull_w = G.local_points_world(cereal, CS._yawed(CS.PLACE, 30.0))
    face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
    fp = CPZ.push_face_point(hull_w, CS.D)
    a = [(k, CS._pose(p), CS._jsonable(e)) for k, p, e in s.touching_push_candidates(LIFT._stance(), fp, face_z, CS.D, t_hull_w=hull_w)]
    b = [(k, CS._pose(p), CS._jsonable(e)) for k, p, e in CPZ.touching_push_candidates(gs, CPZ.MAP_FRAME, fp, face_z, CS.D, t_hull_w=hull_w)]
    assert len(a) == len(b) > 20
    _close([list(x) for x in a], [list(x) for x in b], "swept")
    from ispatialgym.tests import test_touching_brace_open_push_speed as BO

    sb = BO._sampler(gen={"brace_open_when_wide": True})
    st, p_init, b_point, p_hull, face_z = BO._brace_inputs(sb)
    a = [(k, CS._pose(c), CS._jsonable(e)) for _p, c, k, e in sb.touching_brace_candidates(st, b_point, face_z, -BO.DIRECTION, partner_hull_w=p_hull, push_point=p_init)]
    b = [(k, CS._pose(c), CS._jsonable(e)) for _p, c, k, e in CPZ.touching_brace_candidates(CPZ.PoseGen({}), st, b_point, face_z, -BO.DIRECTION, partner_hull_w=p_hull,
                                                                                            push_point=p_init, open_when_wide=True)]
    assert [x[0] for x in a] == [x[0] for x in b] and a[0][0].startswith("brace_wide_h")
    _close([list(x) for x in a], [list(x) for x in b], "brace")


def test_touching_points_is_the_samplers_rule():
    from ispatialgym.tests import test_iors_small_lift as LIFT

    s = LIFT._sampler("touching")
    st = LIFT._stance()
    t_pose, p_pose = LIFT._pose(0.70, -0.12), CS._yawed(LIFT._pose(0.70, 0.10), 20.0)
    w, b = s.touching_points(t_pose, p_pose, st)
    w2, b2 = CPZ.touching_points(G.local_points_world(s.hull["target"], t_pose), G.local_points_world(s.hull["partner"], p_pose), st.tangent)
    assert th.allclose(w, w2) and th.allclose(b, b2)
    # the push contact on the target's right face (min along the tangent), the brace point on the partner's far face (max along it), mid heights
    assert float(w[1]) == pytest.approx(float(G.local_points_world(s.hull["target"], t_pose)[:, 1].min()), abs=1e-6)
    assert float(b[1]) == pytest.approx(float(G.local_points_world(s.hull["partner"], p_pose)[:, 1].max()), abs=1e-6)


def test_the_executed_push_pre_is_the_oracles_lowered_pre():
    """The run-time transforms CALL the oracle (``execute.push_lowered_pre``, ``actions.push_is_open`` / ``brace_is_open``, ``ExecuteConfig``):
    a tip push (pre 0.05) and a pad push (0.10) are lowered to 0.04 m before the init along the approach, a pitched 30 / 45 deg blade and an
    arc keep the recorded pre, a pre already within 4.5 cm too."""
    from ispatialgym.oracle import execute as EX

    cfg = EX.ExecuteConfig()
    assert CPZ.brace_stop_rule() == {"leg": "straight pre -> contact", "touch_stop_m": cfg.brace_touch_stop_m, "backoff_m": cfg.brace_touch_backoff_m}
    want = {("push_tip_h-", 0.05, False): 0.04, ("push_h", None, False): 0.04, ("push_blade_30", None, False): 0.10, ("push_blade_45-", None, False): 0.10,
            ("push_blade_lean12", None, False): 0.04, ("push_blade_0_high-", None, False): 0.04, ("push_tip_h", 0.044, False): 0.044, ("push_h", None, True): 0.10}
    q = C.eef_quat(th.tensor([0.0, 1.0, 0.0]), th.tensor([1.0, 0.0, 0.0]))
    init = (th.tensor([0.6, -0.1, 0.9]), q)
    for (kind, pre_d, arc), d in want.items():
        extras = {} if pre_d is None else {"pre_dist_m": pre_d}
        assert CPZ.executed_push_pre_dist(kind, extras, arc=arc) == pytest.approx(d, abs=1e-6), kind
        low = CPZ.executed_push_pre(kind, init, C.pre_pose(init, CPZ.recorded_pre_dist(extras)), arc=arc, extras=extras)
        assert float(th.linalg.norm(low[0] - init[0])) == pytest.approx(d, abs=1e-5) and th.allclose(low[1], q)
        assert th.allclose(low[0] + d * C.approach_axis(q), init[0], atol=1e-5)  # back along the approach
    assert CPZ.pitched_push_class("push_blade_30") and not CPZ.pitched_push_class("push_blade_lean20-") and not CPZ.pitched_push_class(None)
    assert CPZ.contact_fingers({"fingers_opening_m": 0.0315}) == 0.0315 and CPZ.contact_fingers({}) == 0.0
    assert CPZ.contact_fingers({"fingers_opening_m": 0.0215}, "brace") == 0.0215 and CPZ.contact_fingers({"fingers_opening_m": 0.0}, "brace") == 0.0


@pytest.mark.parametrize("name", ["height_briefcase", "height_sandal_knobs", "height_flat_low", "blade_short_box", "blade_palm_contact", "blade_yawed_offset",
                                  "flat_briefcase", "flat_low_box", "pitched_far_briefcase", "pitched_far_short"])
def test_the_floor_push_builders_produce_the_golden_proposals(name):
    """``IORSLarge.push_height`` / ``_push_under_chain`` (blade, palm contact, flat hand, pitched-far hand) on the shared rules
    (``floor_blade_height``, ``floor_push_chain``): every ``orient.choose`` call's proposals as at 63925048."""
    got = dict(CS.large_scenarios())[name]()
    _close(_golden()["large"][name], CS._jsonable(got), name)


@pytest.mark.parametrize("name", ["nextto_push_hand_poses", "wall_contact_pose", "axial_push_height"])
def test_the_base_push_wall_and_axial_rules_produce_the_golden_poses(name):
    got = dict(CS.nextto_wall_axial_scenarios())[name]()
    _close(_golden()["other"][name], CS._jsonable(got), name)


def test_the_axial_chain_is_the_samplers_former_expression():
    """``axial_push_chain`` = the four lines ``AlignmentAxial._sample_with`` built the chain with (pre_pose of the init, the arc)."""
    side_pt, d = th.tensor([0.62, 0.05, 0.832]), C.unit(th.tensor([0.3, -0.9, 0.0]))
    a_vec = TT_pitched(30.0)
    centre = th.tensor([0.60, 0.11, 0.832])
    for c_axis in (d, C.unit(th.cross(th.tensor([0.0, 0.0, 1.0]), d, dim=0))):
        for c_sign in (1.0, -1.0):
            quat = C.eef_quat(a_vec, c_sign * c_axis)
            p_init = (side_pt - d * 0.0123, quat)
            arc = C.arc_waypoints(p_init, centre, th.tensor([0.0, 0.0, 1.0]), 0.7, n=2)
            old = [C.pre_pose(p_init), p_init, arc[0], arc[1]]
            new = CPZ.axial_push_chain(side_pt, d, 0.0123, a_vec, c_axis, c_sign, centre, 0.7)
            for (p0, q0), (p1, q1) in zip(old, new):
                assert th.equal(G.t(p0), G.t(p1)) and th.equal(G.t(q0), G.t(q1))


def TT_pitched(pitch):
    from ispatialgym.sampling.v2 import tabletop as TT

    return TT.pitched_approach(pitch, th.tensor([1.0, 0.0]))


def test_the_samplers_names_are_the_shared_rules():
    from ispatialgym.sampling.v2.tasks import alignment_wall as AW
    from ispatialgym.sampling.v2.tasks import iors_large_nextto as NT

    assert NT.push_hand_poses is CPZ.base_push_hand_poses and NT.pitched_approach is CPZ.base_push_approach
    assert AW.wall_contact_pose is CPZ.wall_contact_pose and AW.wall_fist_axes is CPZ.wall_fist_axes
    assert CPZ.fingertip_eef is C.fingertip_eef and CPZ.contact_eef is C.contact_eef and CPZ.default_closing_axis is C.default_closing_axis


# ---------------------------------------------------------------------------------------------------------------- the schema


def test_a_schema_entry_is_object_frame_and_comes_back_to_the_world():
    """A touching tip push of the cereal box (yawed 30 deg at the place) as a schema entry: object-frame pose, executed pre (0.04 m), the push
    direction and stroke in the object frame, closed fingers, the height rule; ``entry_world`` at ANOTHER object pose gives the same
    object-relative contact, and at the original pose the builder's own world poses."""
    from ispatialgym.tests import test_iors_small_lift as LIFT

    cereal = LIFT._box(0.1567, 0.0427, 0.2182)
    pose = CS._yawed(CS.PLACE, 30.0)
    hull_w = G.local_points_world(cereal, pose)
    face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
    gs = CPZ.PoseGen({"push_height_sweep": True}, subtask=CS.TOUCHING)
    kind, init, extras = next(c for c in CPZ.touching_push_candidates(gs, CPZ.MAP_FRAME, CPZ.push_face_point(hull_w, CS.D), face_z, CS.D, t_hull_w=hull_w)
                              if c[0] == "push_tip_h-")
    pre = C.pre_pose(init, CPZ.recorded_pre_dist(extras))
    e = CPZ.contact_entry("push", kind, "touching_push", "work", pose, init, pre, extras, face_rule="the target's -y face, footprint centre line",
                          direction=CS.D, stroke_m=0.08, support_z=0.81, object_height=0.2182)
    assert e["schema"] == CPZ.CONTACT_SCHEMA and e["kind"] == "push" and e["class"] == kind == "push_tip_h-" and e["fingers"] == {"state": "closed", "opening_m": 0.0}
    assert e["pre"]["recorded_m"] == pytest.approx(0.05, abs=1e-5) and e["pre"]["executed_m"] == pytest.approx(0.04, abs=1e-5)
    assert e["height_rule"]["absolute_m"] == pytest.approx(0.05, abs=1e-4) and e["height_rule"]["rule_height_m"] == 0.05
    assert e["push"]["stroke_m"] == 0.08 and np.allclose(e["push"]["direction"], [math.sin(math.radians(30)), math.cos(math.radians(30)), 0.0], atol=1e-5)
    w = CPZ.entry_world(e, pose)
    assert th.allclose(w["contact"][0], G.t(init[0]), atol=1e-5) and C.quat_angle(w["contact"][1], G.t(init[1])) < 1e-4
    assert th.allclose(w["pre"][0], G.t(pre[0]), atol=1e-5) and th.allclose(w["goal"][0], G.t(init[0]) + 0.08 * CS.D, atol=1e-5)
    other = (th.tensor([0.1, 0.4, 0.75]), C.quat_from_yaw(1.1))
    w2 = CPZ.entry_world(e, other)
    rel = np.linalg.inv(CPZ._mat(other)) @ CPZ._mat(w2["contact"])
    assert np.allclose(rel, np.linalg.inv(CPZ._mat(pose)) @ CPZ._mat(init), atol=1e-5)
    assert e["digest"] == CPZ.digest({k: v for k, v in e.items() if k != "digest"})


def test_a_brace_entry_records_the_oracles_stop_and_the_open_fingers():
    from ispatialgym.tests import test_touching_brace_open_push_speed as BO

    s = BO._sampler(gen={"brace_open_when_wide": True})
    st, p_init, b_point, p_hull, face_z = BO._brace_inputs(s)
    pre, contact, kind, extras = next(iter(s.touching_brace_candidates(st, b_point, face_z, -BO.DIRECTION, partner_hull_w=p_hull, push_point=p_init)))
    e = CPZ.contact_entry("brace", kind, "touching_brace", "brace", BO.P_POSE, contact, pre, extras, support_z=float(p_hull[:, 2].min()))
    assert e["fingers"]["state"] == "open" and e["fingers"]["opening_m"] == pytest.approx(extras["fingers_opening_m"])
    assert e["stop"] == CPZ.brace_stop_rule() and e["push"] is None and e["pre"]["executed"] == e["pre"]["recorded"]
    with pytest.raises(ValueError):
        CPZ.contact_entry("attach", kind, "x", "work", BO.P_POSE, contact, pre)


def test_the_iors_small_names_stay_importable():
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    for n in ("HullSections", "face_run_at", "hull_cut_points", "polygon_centroid", "min_area_rect", "push_face_point", "open_fingers_drop_below",
              "PUSH_SWEEP_MIN_HEIGHT_M", "PUSH_OPEN_OPENING_M", "CLOSED_TIP_SPREAD_M", "PUSH_WRIST_TURNS", "HAND_SPHERES_EEF", "STOP_CLEARANCE"):
        assert getattr(IS, n) is getattr(CPZ, n), n
    assert IS.PUSH_SWEEP_MIN_HEIGHT_M == {"blade": 0.005, "tip_h": 0.05, "tip_v": 0.062, "pad": 0.08} and IS.PUSH_OPEN_OPENING_M == 0.035
    assert math.isinf(IS.PUSH_RAISED_MAX_ASPECT)


#: owner decision 6: the executed pre-pose per push class (family, pose_class, recorded pre m, arc, executor) -> (rule, distance)
PRE_RULES = [
    ("touching tip push", "push_tip_h-", 0.05, False, "push", "lowered_to_push_pre_dist_m", 0.04),
    ("touching open push", "push_open_v-", 0.05, False, "push", "lowered_to_push_pre_dist_m", 0.04),
    ("touching pad push", "push_h", 0.10, False, "push", "lowered_to_push_pre_dist_m", 0.04),
    ("touching vertical / leaning / raised blade", "push_blade_lean12-", 0.10, False, "push", "lowered_to_push_pre_dist_m", 0.04),
    ("touching pitched blade (push_blade_pitches_deg >= 30)", "push_blade_45", 0.10, False, "push", "recorded", 0.10),
    ("block design push", None, 0.10, False, "push", "lowered_to_push_pre_dist_m", 0.04),
    ("axial push (arc)", None, 0.10, True, "push", "recorded", 0.10),
    ("push-under blade (mid)", None, 0.10, True, "push", "recorded", 0.10),
    ("push-under flat hand (mid)", None, 0.04, True, "push", "recorded", 0.04),
    ("world_vertical wall push (arc)", None, 0.10, True, "push", "recorded", 0.10),
    ("base push (BasePush)", None, 0.10, False, "base_push", "recorded", 0.10),
]


@pytest.mark.parametrize("row", PRE_RULES, ids=[r[0] for r in PRE_RULES])
def test_the_pre_pose_rule_per_push_class_is_what_the_oracle_executes(row):
    _name, cls, rec, arc, executor, rule, dist = row
    r = CPZ.pre_pose_rule(cls, {}, arc=arc, executor=executor, recorded_m=rec)
    assert (r["rule"], r["distance_m"]) == (rule, pytest.approx(dist)) and r["recorded_m"] == pytest.approx(rec)
    if executor == "push":  # the oracle's own function agrees
        q = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([1.0, 0.0, 0.0]))
        init = (th.tensor([0.7, 0.0, 0.85]), q)
        low = CPZ.executed_push_pre(cls, init, C.pre_pose(init, rec), arc=arc)
        assert float(th.linalg.norm(low[0] - init[0])) == pytest.approx(dist, abs=1e-5)
    e = CPZ.contact_entry("push", cls or "push", "x", "work", (th.zeros(3), C.IDENTITY_QUAT.clone()), (th.tensor([0.5, 0.0, 0.1]), C.IDENTITY_QUAT.clone()),
                          C.pre_pose((th.tensor([0.5, 0.0, 0.1]), C.IDENTITY_QUAT.clone()), rec), {}, direction=[1.0, 0.0, 0.0], arc=arc, executor=executor)
    assert e["pre_pose_rule"]["rule"] == rule and e["pre"]["executed_m"] == pytest.approx(dist, abs=1e-5)


def test_the_axial_blade_geometry_and_extreme_point_are_the_samplers_former_bodies():
    """``AlignmentAxial._blade_geometry`` / ``_extreme_point`` now call ``axial_blade_geometry`` / ``extreme_point``: the former bodies, inline."""
    from types import SimpleNamespace

    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.v2.tasks import alignment_axial as AX

    hull = th.tensor([[sx * 0.11, sy * 0.03, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, 0.012)] + [[0.05, 0.031, 0.006]], dtype=th.float32)
    s = SimpleNamespace(top=SimpleNamespace(z=0.7351), extents={"height": 0.012}, gen=lambda k, d=None: d, hull=hull, long_axis=th.tensor([1.0, 0.0, 0.0]))
    s._push_height = lambda: AX.AlignmentAxial._push_height(s)
    s._face_z_range = lambda: AX.AlignmentAxial._face_z_range(s)
    for pitch in (0.0, 30.0, 45.0, 60.0):
        for side in (False, True):
            z_push = s._push_height()
            z_tip = z_push - C.FINGERTIP_LENGTH * math.cos(math.radians(pitch))
            so = TT.blade_standoff(s._face_z_range(), z_tip, pitch, 0.0, side=side)
            old = None if so is None else (so[0], so[1], z_tip)
            assert AX.AlignmentAxial._blade_geometry(s, pitch, 0.0, side=side) == old
    pose = (th.tensor([0.6, -0.1, 0.7351]), C.quat_from_yaw(0.4))
    for direction, near in (((0.0, 1.0), None), ((0.3, -0.9), th.tensor([0.68, -0.06, 0.74])), ((-1.0, 0.0), th.tensor([0.5, -0.14, 0.74]))):
        pts = C.transform_points(pose, hull)
        dxy = th.tensor([float(direction[0]), float(direction[1])])
        dxy = dxy / th.linalg.norm(dxy)
        sel_pts = pts
        if near is not None:
            long_w = C.quat_rotate(pose[1], s.long_axis)[:2]
            along = (pts[:, :2] - near[:2]) @ long_w
            sel = along.abs() <= 0.04
            if int(sel.sum()) >= 3:
                sel_pts = pts[sel]
        old = sel_pts[int(th.argmax(sel_pts[:, :2] @ dxy))]
        assert th.equal(AX.AlignmentAxial._extreme_point(s, pose, th.tensor(direction), near=near), old)


@pytest.mark.parametrize("name", ["magazine_thin", "magazine_thin_back", "keyboard_taller", "spoon_short_face"])
def test_the_axial_stop_brace_produces_the_golden_candidates(name):
    """``AlignmentAxial._stop_geometry`` / ``_brace_candidates`` on the shared rules (``axial_stop_geometry``, ``axial_brace_candidates``): the
    goal pose, the stop points, every candidate in its order and the skip counts as at 45b9ba61 (before the move)."""
    got = dict(CS.axial_brace_scenarios())[name]()
    _close(_golden()["axial_brace"][name], CS._jsonable(got), name)


def test_the_full_open_brace_switch(monkeypatch):
    """Owner decision 2026-11-07: ``ISPATIALGYM_BRACE_FULL_OPEN=1`` puts the full-open spread (2 x 0.05 + 0.027 = 0.127 m, 0.05 per finger)
    first (the default since i7-contactbrace 2026-11-08); "0" keeps today's spreads and records."""
    from ispatialgym.tests import test_iors_small_lift as LIFT
    from ispatialgym.tests import test_touching_brace_open_push_speed as BO

    monkeypatch.setenv(CPZ.BRACE_FULL_OPEN_ENV, "0")
    gs = CPZ.PoseGen({})
    assert CPZ.BRACE_FULL_OPEN_SPREAD_M == 0.127 and CPZ.brace_wide_spreads(gs) == [0.09, 0.07, 0.05] and not CPZ.brace_full_open_on()

    def first_wide(partner):
        s = BO._sampler(gen={"brace_open_when_wide": True}, partner=partner)
        st, p_init, b_point, p_hull, face_z = BO._brace_inputs(s)
        _p, _c, k, e = next(iter(s.touching_brace_candidates(st, b_point, face_z, -BO.DIRECTION, partner_hull_w=p_hull, push_point=p_init)))
        return k, e["stop_width_m"], e["fingers_opening_m"]

    wide, narrow = LIFT._box(0.08, 0.20, 0.24), LIFT._box(0.08, 0.12, 0.24)
    assert first_wide(wide) == ("brace_wide_h-", 0.09, 0.0315) and first_wide(None) == ("brace_wide_h-", 0.09, 0.0315)
    monkeypatch.setenv(CPZ.BRACE_FULL_OPEN_ENV, "1")
    assert CPZ.brace_wide_spreads(gs) == [0.127, 0.09, 0.07, 0.05] and CPZ.brace_wide_spreads(CPZ.PoseGen({"brace_wide_spreads_m": [0.127, 0.05]})) == [0.127, 0.05]
    assert first_wide(wide) == ("brace_wide_h-", 0.127, 0.05)  # the 20 cm face takes the full spread
    assert first_wide(narrow) == ("brace_wide_h-", 0.09, 0.0315) and first_wide(None) == ("brace_wide_h-", 0.09, 0.0315)  # narrower faces: the widest that fits


def test_push_line_through_the_mass_centre_shifts_the_contact_across_the_push():
    """Lead decision 9 (2026-11-08, ISPATIALGYM_PUSH_LINE_COM): with ``line_xy`` (the target's mass centre) the swept / open push classes'
    contact line runs through that point instead of the footprint centre - the same heights and classes, every contact shifted across the
    push by the point's offset, the offset recorded (``push_line``); without it the records are unchanged (no ``push_line``)."""
    from ispatialgym.tests import test_iors_small_lift as LIFT

    cereal = LIFT._box(0.1567, 0.0427, 0.2182)
    pose = CS._yawed(CS.PLACE, 0.0)
    hull_w = G.local_points_world(cereal, pose)
    face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
    gs = CPZ.PoseGen({"push_height_sweep": True}, subtask=CS.TOUCHING)
    fp = CPZ.push_face_point(hull_w, CS.D)
    d2 = CS.D[:2] / th.linalg.norm(CS.D[:2])
    side2 = th.tensor([-float(d2[1]), float(d2[0])])
    sec = CPZ.HullSections(hull_w)
    t_fp, facts = CPZ.push_line_t0(sec, hull_w, face_z[0], side2)
    assert facts == {}
    off = 0.006  # the mass centre 6 mm off the footprint centre across the push
    centre_xy = G.t(pose[0])[:2].float()
    line_xy = centre_xy + off * side2 + 0.01 * d2  # a shift along the push changes nothing
    t0, facts = CPZ.push_line_t0(sec, hull_w, face_z[0], side2, line_xy)
    assert t0 - t_fp == pytest.approx(off + float((centre_xy @ side2)) - t_fp, abs=1e-6)
    assert facts["push_line"]["through"] == "com"
    base = list(CPZ.touching_push_candidates(gs, CPZ.MAP_FRAME, fp, face_z, CS.D, t_hull_w=hull_w))
    com = list(CPZ.touching_push_candidates(gs, CPZ.MAP_FRAME, fp, face_z, CS.D, t_hull_w=hull_w, line_xy=line_xy))
    assert base and [k for k, _p, _e in com][:len(base)] == [k for k, _p, _e in base][:len(com)]
    shift = float(facts["push_line"]["offset_from_footprint_m"])
    n = 0
    for (kb, pb, eb), (kc, pc, ec) in zip(base, com):
        assert kb == kc and "push_line" not in eb
        if "contact_point" in eb and eb.get("push_height_sweep") is True:
            n += 1
            assert ec["push_line"]["offset_from_footprint_m"] == pytest.approx(shift, abs=1e-6)
            dcp = th.tensor(ec["contact_point"][:2]) - th.tensor(eb["contact_point"][:2])
            assert float(dcp @ side2) == pytest.approx(shift, abs=2e-4) and abs(float(dcp @ d2)) < 2e-3
            assert ec["contact_height_m"] == eb["contact_height_m"]
    assert n > 0
    # the open-gripper classes take the same line
    ob = list(CPZ.open_push_candidates(gs, CPZ.MAP_FRAME, hull_w, face_z, CS.D))
    oc = list(CPZ.open_push_candidates(gs, CPZ.MAP_FRAME, hull_w, face_z, CS.D, line_xy=line_xy))
    if ob and oc:
        assert "push_line" not in ob[0][2] and oc[0][2]["push_line"]["offset_from_footprint_m"] == pytest.approx(shift, abs=1e-6)


def test_push_line_com_switch_and_the_samplers_line_point(monkeypatch):
    """The sampler's line point: None with the switch off (default); on, the target's mass centre (root frame, cached) at the place pose."""
    from types import SimpleNamespace

    from ispatialgym.sampling.v2.tasks import alignment_axial as AX
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    monkeypatch.delenv(IS.PUSH_LINE_COM_ENV, raising=False)
    assert not IS.push_line_com_on()
    stats, calls = {}, []
    me = SimpleNamespace(obj=lambda role: SimpleNamespace(name="target_x"), kit=SimpleNamespace(stats=stats), gp=SimpleNamespace(variant="t"))
    me.push_line_xy = IS.IORSSmall.push_line_xy.__get__(me)
    t_pose = (th.tensor([1.0, 2.0, 0.9]), C.quat_from_yaw(math.pi / 2))
    assert me.push_line_xy(t_pose) is None
    monkeypatch.setattr(AX, "_com_local", lambda obj: calls.append(obj.name) or th.tensor([0.01, 0.0, 0.05]))
    monkeypatch.setenv(IS.PUSH_LINE_COM_ENV, "1")
    xy = me.push_line_xy(t_pose)
    assert th.allclose(xy, th.tensor([1.0, 2.01]), atol=1e-5)  # +1 cm along the root x, the root turned 90 deg
    me.push_line_xy(t_pose)
    assert calls == ["target_x"] and stats["push_line_com.used"] == 2


def test_push_tip_height_cap_is_the_ratio_times_the_leading_half_depth_with_the_tip_floor():
    """i7-symonly (ISPATIALGYM_PUSH_TIP_CAP): the cap is ``ratio`` x the footprint's leading half-depth along the push (a box: half its
    extent along the push), never below the lowest fingertip push (5 cm); the spray bottle's ~4.25 cm gives 6.4 cm at the default 1.5."""
    from ispatialgym.tests import test_iors_small_lift as LIFT

    d2 = CS.D[:2] / th.linalg.norm(CS.D[:2])
    for box, yaw in ((LIFT._box(0.0905, 0.085, 0.273), 0.0), (LIFT._box(0.0905, 0.085, 0.273), math.pi / 2), (LIFT._box(0.1567, 0.0427, 0.2182), 0.0)):
        hull_w = G.local_points_world(box, CS._yawed(CS.PLACE, yaw))
        z0 = float(hull_w[:, 2].min())
        proj = hull_w[:, :2].float() @ d2
        half = 0.5 * float(proj.max() - proj.min())
        out = CPZ.push_tip_height_cap(hull_w, z0, CS.D)
        assert out["x_lead_m"] == pytest.approx(half, abs=2e-3)
        assert out["cap_m"] == pytest.approx(max(CPZ.PUSH_TIP_HEIGHT_RATIO * half, CPZ.PUSH_SWEEP_MIN_HEIGHT_M["tip_h"]), abs=3e-3)
        assert out["cap_m"] >= CPZ.PUSH_SWEEP_MIN_HEIGHT_M["tip_h"]
        assert CPZ.push_tip_height_cap(hull_w, z0, CS.D, ratio=3.0)["cap_m"] == pytest.approx(max(3.0 * half, 0.05), abs=6e-3)
    # the swept classes of a tall target keep the low contacts only once capped
    gs = CPZ.PoseGen({"push_height_sweep": True}, subtask=CS.TOUCHING)
    hull_w = G.local_points_world(LIFT._box(0.0905, 0.085, 0.273), CS._yawed(CS.PLACE, 0.0))
    face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
    cands = list(CPZ.touching_push_candidates(gs, CPZ.MAP_FRAME, CPZ.push_face_point(hull_w, CS.D), face_z, CS.D, t_hull_w=hull_w))
    cap = CPZ.push_tip_height_cap(hull_w, face_z[0], CS.D)["cap_m"]
    hs = [float(e["contact_height_m"]) for _k, _p, e in cands if e.get("contact_height_m") is not None]
    assert min(hs) <= cap < max(hs)


def test_push_tip_cap_switch_filters_and_counts(monkeypatch):
    """The sampler side: default off; the map keep reads the label's height; the exact stream skips (and counts) the contacts above the cap."""
    from types import SimpleNamespace

    from ispatialgym.sampling.v2.tasks import iors_small as IS

    monkeypatch.delenv(IS.PUSH_TIP_CAP_ENV, raising=False)
    assert IS.PUSH_TIP_CAP_DEFAULT == "0" and IS.push_tip_cap_mode() == "off"
    for v, mode in (("1", "on"), ("shadow", "shadow"), ("0", "off")):
        monkeypatch.setenv(IS.PUSH_TIP_CAP_ENV, v)
        assert IS.push_tip_cap_mode() == mode
    keep = IS._tip_keep(0.064)
    assert keep({"class": "push_tip_h-@5.0cm"}) and keep({"class": "push_tip_v-@6.2cm"}) and not keep({"class": "push_tip_h-@7.0cm"})
    assert keep({"class": "push_blade_0_high"}) and keep(None)
    stats = {}
    me = SimpleNamespace(kit=SimpleNamespace(stats=stats))
    stream = [("a", None, {"contact_height_m": 0.05}), ("b", None, {"contact_height_m": 0.08}), ("c", None, {}), ("d", None, {"contact_height_m": 0.064})]
    kept = [k for k, _p, _e in IS.IORSSmall._tip_capped(me, iter(stream), 0.064)]
    assert kept == ["a", "c", "d"] and stats["push_tip_cap.dropped"] == 1
    # off: no cap computed
    monkeypatch.setenv(IS.PUSH_TIP_CAP_ENV, "0")
    me.push_height_sweep_on = lambda: True
    assert IS.IORSSmall.push_tip_cap(me, (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), 0.0, CS.D) is None
