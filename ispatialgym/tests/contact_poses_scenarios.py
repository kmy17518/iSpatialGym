"""i7-contact 2026-11-07: the touching push / brace candidate scenarios of the contact-pose regression (``test_contact_poses``): each scenario
builds an ``IORSSmall`` stand-in (``test_iors_small_lift._sampler``: the real class without a simulator), its hulls, the stance and the
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


def run_all() -> Dict[str, Any]:
    out: Dict[str, Any] = {"push": {}, "brace": {}}
    for name, fn in push_scenarios():
        out["push"][name] = [[k, _pose(p), _jsonable(e)] for k, p, e in fn()]
    for name, fn in brace_scenarios():
        out["brace"][name] = [[k, _pose(pre), _pose(c), _jsonable(e)] for pre, c, k, e in fn()]
    return out


if __name__ == "__main__":
    doc = run_all()
    with open(GOLDEN, "w") as f:
        json.dump(doc, f, indent=0, sort_keys=True)
    print({k: {n: len(v) for n, v in d.items()} for k, d in doc.items()})
