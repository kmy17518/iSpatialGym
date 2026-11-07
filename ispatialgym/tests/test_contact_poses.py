"""i7-contact 2026-11-07: the touching push / brace pose rules moved to ``sampling/v2/contact_poses.py`` (shared by ``IORSSmall`` and the contact
reach maps). The sampler's record builders keep producing the records they produced at 63925048 (``data/contact_poses_golden.json``, the
scenarios of ``contact_poses_scenarios``), the module functions produce the same lists from a plain generation source (:class:`PoseGen`,
the map builder's), and the executed-contact helpers are the oracle's rules (``execute.push_lowered_pre``). No simulator."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

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
    """``executed_push_pre_dist`` against ``execute.push_lowered_pre`` itself: a tip push (pre 0.05) and a pad push (0.10) are lowered to
    0.04 m before the init along the approach, a pitched 30 / 45 deg blade and an arc keep the recorded pre, a pre already within 4.5 cm too."""
    from ispatialgym.oracle import execute as EX
    from ispatialgym.oracle.sidecar import Pose

    cfg = EX.ExecuteConfig()
    assert CPZ.PUSH_EXEC_PRE_DIST_M == cfg.push_pre_dist_m and CPZ.BRACE_EXEC_TOUCH_STOP_M == cfg.brace_touch_stop_m
    assert CPZ.BRACE_EXEC_BACKOFF_M == cfg.brace_touch_backoff_m
    q = C.eef_quat(th.tensor([0.0, 1.0, 0.0]), th.tensor([1.0, 0.0, 0.0]))
    init = (th.tensor([0.6, -0.1, 0.9]), q)
    for kind, extras, arc in (("push_tip_h-", {"pre_dist_m": 0.05}, False), ("push_h", {}, False), ("push_blade_30", {}, False), ("push_blade_45-", {}, False),
                              ("push_blade_lean12", {}, False), ("push_blade_0_high-", {}, False), ("push_tip_h", {"pre_dist_m": 0.044}, False), ("push_h", {}, True)):
        pre = C.pre_pose(init, CPZ.recorded_pre_dist(extras))
        contact = SimpleNamespace(mid=object() if arc else None, extra={"pose_class": kind})
        low = EX.push_lowered_pre(contact, Pose(pre[0].tolist(), pre[1].tolist()), Pose(init[0].tolist(), init[1].tolist()), cfg)
        oracle = CPZ.recorded_pre_dist(extras) if low is None else float(th.linalg.norm(init[0] - th.tensor(low.position)))
        assert CPZ.executed_push_pre_dist(kind, extras, arc=arc) == pytest.approx(oracle, abs=1e-6), kind
    assert CPZ.pitched_push_class("push_blade_30") and not CPZ.pitched_push_class("push_blade_lean20-") and not CPZ.pitched_push_class(None)
    assert CPZ.contact_fingers({"fingers_opening_m": 0.0315}) == 0.0315 and CPZ.contact_fingers({}) == 0.0


def test_the_iors_small_names_stay_importable():
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    for n in ("HullSections", "face_run_at", "hull_cut_points", "polygon_centroid", "min_area_rect", "push_face_point", "open_fingers_drop_below",
              "PUSH_SWEEP_MIN_HEIGHT_M", "PUSH_OPEN_OPENING_M", "CLOSED_TIP_SPREAD_M", "PUSH_WRIST_TURNS", "HAND_SPHERES_EEF", "STOP_CLEARANCE"):
        assert getattr(IS, n) is getattr(CPZ, n), n
    assert IS.PUSH_SWEEP_MIN_HEIGHT_M == {"blade": 0.005, "tip_h": 0.05, "tip_v": 0.062, "pad": 0.08} and IS.PUSH_OPEN_OPENING_M == 0.035
    assert math.isinf(IS.PUSH_RAISED_MAX_ASPECT)
