"""i7-contact 2026-11-07: the touching push / brace candidate scenarios of the contact-pose regression (``test_contact_poses``): each scenario
builds an ``IORSSmall`` stand-in (the large-scale / nextto / wall / axial builders: their tests' stand-ins) (``test_iors_small_lift._sampler``: the real class without a simulator), its hulls, the stance and the
generation, and lists one record builder's candidates. ``ispatialgym/tests/data/contact_poses_golden.json`` holds the lists the builders
produced at 63925048 (before the rules moved to ``sampling/v2/contact_poses.py``); ``python -m ispatialgym.tests.contact_poses_scenarios``
rewrites it (only ever from a commit whose records are the reference)."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G

GOLDEN = os.path.join(os.path.dirname(__file__), "data", "contact_poses_golden.json")
TOUCHING = "inter_object_relation_satisfaction-touching_small_scale"
D = th.tensor([0.0, 1.0, 0.0])
PLACE = (th.tensor([0.70, 0.0, 0.81]), C.IDENTITY_QUAT.clone())


def _yawed(pose, deg: float):
    return G.t(pose[0]).clone(), C.qmul(C.quat_from_yaw(math.radians(deg)), G.t(pose[1]))


def _jsonable(x):
    if isinstance(x, th.Tensor):
        return [round(float(v), 6) for v in x.reshape(-1).tolist()]
    if isinstance(x, float):
        return round(x, 6)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _pose(p) -> List[float]:
    return _jsonable(th.cat([G.t(p[0]).float().reshape(-1), G.t(p[1]).float().reshape(-1)]))


def _lift():
    from ispatialgym.tests import test_iors_small_lift as LIFT

    return LIFT


def _bottle():
    from ispatialgym.tests import test_touching_push_sweep as SW

    return SW._bottle()


def _push_sampler(gen: Dict[str, Any], target, subtask: bool = True):
    LIFT = _lift()
    s = LIFT._sampler("touching", gen=dict(gen))
    if subtask:
        s.subtask = TOUCHING
    s.hull = {"target": target, "partner": LIFT._box(*LIFT.PARTNER)}
    return s


def push_scenarios():
    """``(name, callable -> [(kind, pose, extras)])`` of the push record builders."""
    LIFT = _lift()
    cereal = LIFT._box(0.1567, 0.0427, 0.2182)
    short = LIFT._box(0.10, 0.05, 0.06)
    tall = LIFT._box(0.06, 0.05, 0.16)
    out = []

    def today(name, gen, target, pose=PLACE, subtask=False):
        def run():
            s = _push_sampler(gen, target, subtask=subtask)
            hull_w = G.local_points_world(target, pose)
            face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
            from ispatialgym.sampling.v2.tasks import iors_small as IS

            return list(s.touching_push_candidates(LIFT._stance(), IS.push_face_point(hull_w, D), face_z, D))
        out.append((name, run))

    def swept(name, gen, target, pose=PLACE):
        def run():
            s = _push_sampler({"push_height_sweep": True, **gen}, target)
            hull_w = G.local_points_world(target, pose)
            face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
            from ispatialgym.sampling.v2.tasks import iors_small as IS

            return list(s.touching_push_candidates(LIFT._stance(), IS.push_face_point(hull_w, D), face_z, D, t_hull_w=hull_w))
        out.append((name, run))

    def opened(name, gen, target, pose=PLACE):
        def run():
            s = _push_sampler({"push_gripper_open": True, **gen}, target)
            hull_w = G.local_points_world(target, pose)
            face_z = (float(hull_w[:, 2].min()), float(hull_w[:, 2].max()))
            return list(s.open_push_candidates(LIFT._stance(), hull_w, face_z, D))
        out.append((name, run))

    today("today_cereal", {}, cereal)
    today("today_cereal_pitched", {"blade_toward_com": True, "push_blade_pitches_deg": [0.0, 30.0, 45.0], "blade_palm_min_pitch_deg": 30.0}, cereal)
    today("today_short_extra", {"contact_height_extra_m": 0.01}, short)
    today("today_tall_noclear", {"blade_palm_clear_first": False, "blade_lean_deg": 12.0, "stop_clearance_m": 0.004}, tall)
    today("today_tall_yaw25", {"push_tip_pre_dist_m": 0.06}, tall, _yawed(PLACE, 25.0))
    swept("swept_bottle", {}, _bottle())
    swept("swept_cereal_raised", {"push_raised_max_aspect": 1.5}, cereal, _yawed(PLACE, 30.0))
    swept("swept_short_step", {"push_height_step_m": 0.005, "push_max_height_fraction": 0.8, "blade_lean_deg": [20.0, 8.0]}, short)
    swept("swept_tall_raised_inf", {"blade_lean_tip_above_m": 0.03}, tall, _yawed(PLACE, -40.0))
    opened("open_cereal", {}, cereal)
    opened("open_cereal_q30", {"push_open_opening_m": 0.03, "push_max_height_fraction": 0.7}, cereal, _yawed(PLACE, 5.0))
    opened("open_wide_box", {"push_height_step_m": 0.02}, LIFT._box(0.14, 0.05, 0.16))
    return out


def brace_scenarios():
    """``(name, callable -> [(pre, contact, kind, extras)])`` of the brace record builders."""
    from ispatialgym.tests import test_touching_brace_open_push_speed as BO

    LIFT = _lift()
    narrow = LIFT._box(0.06, 0.05, 0.20)
    thin = LIFT._box(0.12, 0.10, 0.03)
    out = []

    def bo(name, gen, partner=None, p_pose=None):
        def run():
            s = BO._sampler(gen=gen, partner=partner)
            st, p_init, b_point, p_hull, face_z = BO._brace_inputs(s, p_pose=p_pose or BO.P_POSE)
            return list(s.touching_brace_candidates(st, b_point, face_z, -BO.DIRECTION, partner_hull_w=p_hull, push_point=p_init))
        out.append((name, run))

    def lift(name, gen, partner, pose):
        def run():
            s = LIFT._sampler("touching", gen=dict(gen))
            s.hull = {"target": LIFT._box(*LIFT.TARGET), "partner": partner}
            st = LIFT._stance()
            t_pose = LIFT._pose(0.70, -0.12)
            p_init, b_point = s.touching_points(t_pose, pose, st)
            p_hull = G.local_points_world(partner, pose)
            face_z = (float(p_hull[:, 2].min()), float(p_hull[:, 2].max()))
            return list(s.touching_brace_candidates(st, b_point, face_z, -D, partner_hull_w=p_hull, push_point=p_init))
        out.append((name, run))

    bo("dkhgxn_default", {})
    bo("dkhgxn_open_when_wide", {"brace_open_when_wide": True})
    bo("dkhgxn_spreads", {"brace_wide_spreads_m": [0.07, 0.05], "brace_tip_pitches_deg": [0.0, 15.0, 30.0]})
    bo("narrow_no_tip", {"brace_tip_first": False}, partner=narrow)
    bo("narrow_open_when_wide", {"brace_open_when_wide": True, "brace_tip_min_above_m": 0.06}, partner=narrow)
    lift("thin_extra", {"contact_height_extra_m": 0.01}, thin, _yawed(LIFT._pose(0.70, 0.10), 20.0))
    lift("narrow_nowide", {"brace_wide_stop": False, "blade_palm_clear_m": 0.02}, narrow, _yawed(LIFT._pose(0.70, 0.10), -35.0))
    return out


# ---------------------------------------------------------------------------------------------------------------- the other push / brace builders


def _capture(IL, fn):
    """Every ``orient.choose`` call's proposals during ``fn()`` (each call answered "none chosen", so every form / pitch / rung is visited):
    ``[[stage, [[label, rank, payload, [pose...]], ...]], ...]``."""
    calls = []
    orig = IL.O.choose

    def choose(ikc, arm, props, evaluate, q_start, camera_idx=None, stage="", downstream=None):
        calls.append([stage, [[p.label, p.rank, _jsonable(p.payload) if not isinstance(p.payload, th.Tensor) else _jsonable(p.payload),
                               [_pose(c) for c in p.chain]] for p in props]])
        return None, {"options": []}

    IL.O.choose = choose
    try:
        fn()
    finally:
        IL.O.choose = orig
    return calls


def large_scenarios():
    """``(name, callable -> jsonable)`` of the floor push builders of ``iors_large`` (push height, the blade / flat / pitched-far chains)."""
    from ispatialgym.sampling.v2.tasks import iors_large as IL
    from ispatialgym.tests import test_iors_large_push_pitched as PP
    from ispatialgym.tests import test_push_under_palm as PALM

    out = []

    def heights(name, height, gen):
        def run():
            s = PP._sampler(height=height, **gen)
            return [list(s.push_height(p)) for p in (0.0, 30.0, 45.0, 60.0, 65.0, 70.0)]
        out.append((name, run))

    def chain(name, make, stance, drop, goal, d2, yaw=0.0):
        def run():
            s = make()
            q = C.quat_from_yaw(math.radians(yaw))
            return _capture(IL, lambda: s._push_under_chain(stance(), q, drop, goal, d2))
        out.append((name, run))

    heights("height_briefcase", 0.27, {})
    heights("height_sandal_knobs", 0.072, {"push_tip_frac": 0.4, "push_tip_clearance_m": 0.02})
    heights("height_flat_low", 0.02, {})
    palm_place = lambda: PALM.PLACE  # noqa: E731
    chain("blade_short_box", lambda: PALM._sampler(PALM._box(0.20, 0.12, 0.08)), palm_place, PALM.DROP, PALM.GOAL, (0.0, 1.0))
    chain("blade_palm_contact", lambda: PALM._sampler(PALM._box(0.33, 0.12, 0.27), push_under_palm_contact=True), palm_place, PALM.DROP, PALM.GOAL, (0.0, 1.0))
    chain("blade_yawed_offset", lambda: PALM._sampler(PALM._box(0.24, 0.09, 0.07), push_contact_offset_m=0.11, push_goal_extra_m=0.02), palm_place,
          (0.05, -0.02), (0.05, 0.35), (0.0, 1.0), yaw=30.0)
    from ispatialgym.tests import test_iors_large_push_flat as FL

    chain("flat_briefcase", lambda: FL._sampler(push_under_hand="flat"), FL._stance, FL.DROP, FL.GOAL, (1.0, 0.0))
    chain("flat_low_box", lambda: FL._sampler(height=0.10, com_z=0.04, push_under_hand="flat", push_flat_pre_dist_m=0.06), FL._stance, FL.DROP, FL.GOAL, (1.0, 0.0),
          yaw=20.0)
    chain("pitched_far_briefcase", lambda: PP._sampler(push_under_hand="pitched_far"), PP._stance, (1.0, 0.0), (1.40, 0.0), (1.0, 0.0))
    chain("pitched_far_short", lambda: PP._sampler(height=0.12, com_z=0.05, push_under_hand="pitched_far", push_far_pre_m=[0.08, 0.04]), PP._stance,
          (1.0, 0.1), (1.35, 0.1), (1.0, 0.0), yaw=15.0)
    return out


def nextto_wall_axial_scenarios():
    """``(name, callable -> jsonable)``: the base push's hand poses (``iors_large_nextto.push_hand_poses``), the wall contacts
    (``alignment_wall.wall_contact_pose``) and the axial push's eef height (``AlignmentAxial._push_height``)."""
    from types import SimpleNamespace

    from ispatialgym.sampling.v2.tasks import alignment_axial as AX
    from ispatialgym.sampling.v2.tasks import alignment_wall as AW
    from ispatialgym.sampling.v2.tasks import iors_large_nextto as NT

    out = []

    def nextto():
        rows = []
        for point, d2, pitch, wrist, pre in (((1.2, 0.3, 0.25), (0.0, 1.0), 20.0, 1.0, 0.10), ((0.4, -0.2, 0.4), (0.6, -0.8), 35.0, -1.0, 0.08),
                                             ((0.0, 0.0, 0.1), (-1.0, 0.0), 0.0, 1.0, 0.12)):
            pre_p, contact = NT.push_hand_poses(th.tensor(point), d2, pitch, wrist, pre)
            rows.append([_pose(pre_p), _pose(contact), _jsonable(NT.pitched_approach(d2, pitch))])
        return rows

    def wall():
        rows = []
        for point, n, d, face, gap, tilt, turn in (((0.5, 0.2, 0.9), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), 0.0084, 0.0, 0.0, 1.0),
                                                   ((0.5, 0.2, 0.9), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0.007, 0.005, 0.3, -1.0),
                                                   ((-0.3, 1.1, 1.2), (0.0, -1.0, 0.0), (0.6, 0.0, 0.8), 0.018, 0.002, -0.2, 1.0)):
            rows.append(_pose(AW.wall_contact_pose(th.tensor(point), th.tensor(n), th.tensor(d), face, gap=gap, tilt=tilt, turn=turn)))
            rows.append(_jsonable(list(AW.wall_fist_axes(th.tensor(n), th.tensor(d), tilt))))
        return rows

    def axial():
        rows = []
        for h, top, gen in ((0.012, 0.81, {}), (0.03, 0.7351, {}), (0.05, 0.6854, {}), (0.03, 0.8126, {"contact_height_extra_m": 0.01})):
            s = SimpleNamespace(top=SimpleNamespace(z=top), extents={"height": h}, gen=lambda k, d=None, g=gen: g.get(k, d))
            rows.append(round(float(AX.AlignmentAxial._push_height(s)), 6))
        return rows

    out += [("nextto_push_hand_poses", nextto), ("wall_contact_pose", wall), ("axial_push_height", axial)]
    return out


def axial_brace_scenarios():
    """``(name, callable -> jsonable)``: ``AlignmentAxial._stop_geometry`` + ``_brace_candidates`` (the stop brace of the axial push) on a
    stand-in (the methods of the class on a namespace: hull, centre of mass, extents, the support top) for a thin and a taller target."""
    import types
    from types import SimpleNamespace

    from ispatialgym.sampling.v2.tasks import alignment_axial as AX
    from ispatialgym.tests import test_iors_small_lift as LIFT

    def stand_in(dims, com_z):
        hull = th.tensor([[sx * dims[0] / 2, sy * dims[1] / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, dims[2])]
                         + [[dims[0] / 4, dims[1] / 2, dims[2] / 2], [-dims[0] / 3, -dims[1] / 2, dims[2] / 3]], dtype=th.float32)
        st = SimpleNamespace(hull=hull, com_local=th.tensor([0.0, 0.0, com_z]), extents={"height": dims[2], "long": dims[0]}, top=SimpleNamespace(z=0.7351),
                             gen=lambda k, d=None: d, _blade_skips={}, long_axis=th.tensor([1.0, 0.0, 0.0]))
        for n in ("_push_height", "_face_z_range", "_blade_geometry", "_stop_geometry", "_stop_face_points", "_brace_candidates"):
            setattr(st, n, types.MethodType(getattr(AX.AlignmentAxial, n), st))
        return st

    out = []

    def case(name, dims, com_z, yaw0_deg, yaw_goal_deg, end_sign, passing=None):
        def run():
            s = stand_in(dims, com_z)
            manip = LIFT._stance()
            q0 = C.quat_from_yaw(math.radians(yaw0_deg))
            pose = (th.tensor([0.62, -0.05, 0.7351]), q0)
            com = G.t(pose[0]) + C.quat_rotate(q0, s.com_local)
            long_w = C.quat_rotate(q0, s.long_axis)
            end = com + long_w * end_sign * (0.5 * dims[0] - 0.03)
            d_goal = C.wrap_angle(math.radians(yaw_goal_deg) - math.radians(yaw0_deg))
            spin = 1.0 if d_goal > 0 else -1.0
            z_brace = 0.7351 + 0.75 * dims[2]
            goal_pose, stop_pts, d_stop = s._stop_geometry(pose, end, com, spin, math.radians(yaw_goal_deg), z_brace)
            rows = [[_pose(pre), _pose(c), k, _jsonable(e)] for pre, c, k, e in s._brace_candidates(manip, stop_pts, d_stop, passing, pose, com, d_goal)]
            return {"goal_pose": _pose(goal_pose), "stop_pts": _jsonable(stop_pts), "d_stop": _jsonable(d_stop), "candidates": rows, "skips": dict(s._blade_skips)}
        out.append((name, run))

    case("magazine_thin", (0.28, 0.20, 0.012), 0.006, 40.0, 90.0, 1.0)
    case("magazine_thin_back", (0.28, 0.20, 0.012), 0.006, 140.0, 90.0, -1.0)
    case("keyboard_taller", (0.44, 0.13, 0.045), 0.02, 60.0, 0.0, 1.0, passing=["near_h"])
    case("spoon_short_face", (0.17, 0.035, 0.02), 0.008, -30.0, 20.0, -1.0)
    return out


def run_all() -> Dict[str, Any]:
    out: Dict[str, Any] = {"push": {}, "brace": {}}
    for name, fn in push_scenarios():
        out["push"][name] = [[k, _pose(p), _jsonable(e)] for k, p, e in fn()]
    for name, fn in brace_scenarios():
        out["brace"][name] = [[k, _pose(pre), _pose(c), _jsonable(e)] for pre, c, k, e in fn()]
    out["large"] = {name: fn() for name, fn in large_scenarios()}
    out["other"] = {name: fn() for name, fn in nextto_wall_axial_scenarios()}
    out["axial_brace"] = {name: fn() for name, fn in axial_brace_scenarios()}
    return out


if __name__ == "__main__":
    import sys

    doc = run_all()
    if "--sections" in sys.argv:  # only these sections are rewritten (the others keep the commit they were made at)
        keep = sys.argv[sys.argv.index("--sections") + 1].split(",")
        with open(GOLDEN) as f:
            old = json.load(f)
        doc = {k: (doc[k] if k in keep else old.get(k, doc[k])) for k in doc}
    with open(GOLDEN, "w") as f:
        json.dump(doc, f, indent=0, sort_keys=True)
    print({k: {n: len(v) for n, v in d.items()} for k, d in doc.items()})
