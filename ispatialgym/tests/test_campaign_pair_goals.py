from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.campaign_held_goals import ENV
from ispatialgym.sampling.campaign_pair_goals import PairGoalPlan
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2.tasks import iors_small as I


def plan(relation="ontop"):
    return PairGoalPlan("fresh", "g", "train", "t", "a" * 64, relation, "target", "partner",
                        ((0., 0., .81), (0., 0., 0., 1.)), ((0., 0., .8), (0., 0., 0., 1.)),
                        ((0., 0., .7), (0., 0., 0., 1.)), 1000., 3, "b" * 64)


def test_disabled_pair_plan_never_accesses_runtime_context(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert I.IORSSmall.campaign_pair_plan(SimpleNamespace()) is None


def test_pair_start_keeps_full_yaw_range_independent_of_goal_alignment(monkeypatch):
    calls = []
    monkeypatch.setattr(I.TT, "phase3_target", lambda *a, **kw: (calls.append(kw) or (None, SimpleNamespace(grasp_name="g"))))
    s = SimpleNamespace(_campaign_pair_active=plan(), relation="ontop", grasps_flush=[], grasps=[],
                        neighbor_start_top=lambda: None, pair_goal_align=lambda: {"relative_yaw_deg": 0},
                        pair_target_yaws=lambda manip: pytest.fail("start yaw must not inherit goal yaw"),
                        hull={"target": None}, rest={"target": None}, support_objects=lambda: [], top=None,
                        choice=SimpleNamespace(models={"target": SimpleNamespace(scale=None)}))
    I.IORSSmall.phase3_two_tier(s, None, np.random.default_rng(1))
    assert "yaws" not in calls[0] and "jitter" not in calls[0]


def test_fixed_partner_runs_geometry_checks(monkeypatch):
    monkeypatch.setattr(I.TT, "partner_pose_why", lambda *a, **kw: "partner footprint off the top")
    s = SimpleNamespace(_campaign_pair_active=plan(), world=SimpleNamespace(footprint=lambda *a: None),
                        hull={"target": None, "partner": None}, top=None, margin_support=.02, gap_min=.03)
    with pytest.raises(Reject, match="off the top"):
        I.IORSSmall.place_partner(s, None, None, None, None)


def test_inside_goal_orientation_is_unconstrained_and_metadata_goal_exact():
    p = plan("inside")
    s = I.IORSSmall.__new__(I.IORSSmall)
    s._campaign_pair_active = p
    s.campaign_group_remaining_s = lambda: 100
    s.tol = lambda key, default: default
    s.gen = lambda key, default=None: default
    s.objects = {"partner": SimpleNamespace(get_position_orientation=lambda: tuple(th.tensor(v) for v in p.partner)),
                 "target": SimpleNamespace(get_position_orientation=lambda: (th.tensor([.002, 0., .8]), th.tensor([1., 0., 0., 0.])))}
    d = Draft()
    d.summary["campaign_goal_group"] = p.summary()
    d.pose_targets = {"target": {"orientation_metric": "full"}, "partner": {"orientation_metric": "full"}}
    s.campaign_verify_pair(d, settled=True)
    assert d.pose_targets["target"]["orientation_metric"] is None
    assert d.pose_targets["target"]["position"] == list(p.goal[0])
    assert d.summary["campaign_goal_group"]["settled_goal_residual"]["position_m"] > 0


def test_large_goal_is_pinned_to_supported_cell_before_starts(monkeypatch):
    from dataclasses import replace
    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge
    p = replace(plan(), scale="large", goal_cell=(.1, .2))
    s = SimpleNamespace(campaign_pair_plan=lambda: p, tname="target", pname="partner",
                        partner=SimpleNamespace(get_position_orientation=lambda: p.partner),
                        num=lambda key, default: default, tol=lambda key, default: default,
                        place={"cells": th.tensor([[.1, .2], [.5, .6]])})
    IORSLarge.campaign_pin_goal(s)
    assert s.place["cells"].shape == (1, 2)
    assert s._campaign_latest_proposal["goal_poses"]["target"][0] == list(p.goal[0])
    s.place["cells"] = th.tensor([[2., 2.]])
    with pytest.raises(Reject, match="supported placement region"):
        IORSLarge.campaign_pin_goal(s)
    monkeypatch.delenv(ENV, raising=False)
    assert IORSLarge.campaign_pair_plan(SimpleNamespace()) is None


def test_large_inside_keeps_orientation_and_exact_goal_after_native_settle():
    from dataclasses import replace
    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge
    p = replace(plan("inside"), scale="large", goal_cell=(0., 0.))
    s = SimpleNamespace()
    s.campaign_verify_pair = lambda draft, **kwargs: IORSLarge.campaign_verify_pair(s, draft, **kwargs)
    s._campaign_pair_active = p
    s.campaign_group_remaining_s = lambda: 100
    s.tol = s.num = lambda key, default: default
    s.tname, s.pname = "target", "partner"
    s.objects = {"partner": SimpleNamespace(get_position_orientation=lambda: p.partner),
                 "target": SimpleNamespace(get_position_orientation=lambda: ((.002, 0., .8), p.goal[1]))}
    s.target, s.partner = s.objects["target"], s.objects["partner"]
    d = Draft()
    d.summary["campaign_goal_group"] = p.summary()
    d.pose_targets = {"target": {"orientation_metric": "full"}}
    s.campaign_verify_pair(d, settled=True)
    assert d.pose_targets["target"]["position"] == list(p.goal[0])
    assert d.pose_targets["target"]["orientation"] == list(p.goal[1])
    s.objects["target"].get_position_orientation = lambda: (p.goal[0], (1., 0., 0., 0.))
    with pytest.raises(Reject, match="settled pair target drifted"):
        s.campaign_verify_pair(d, settled=True)


def test_touching_place_stays_fixed_and_rejects_an_incompatible_push_direction():
    from dataclasses import replace
    p = replace(plan("touching"), place=((0., -.08, .8), (0., 0., 0., 1.)))
    s = SimpleNamespace(_campaign_pair_active=p, relation="touching", place_gap=lambda: .08,
        ikc=SimpleNamespace(WITNESS_POS_TOL=.005, WITNESS_ANG_TOL=.05),
        top=SimpleNamespace(polygon_supported=lambda *args, **kwargs: True), margin_support=.02,
        world=SimpleNamespace(footprint=lambda *args: None), hull={"target": None})
    stance = SimpleNamespace(tangent=th.tensor([0., 1.]))
    start = (th.tensor([.5, -.5, .8]), th.tensor([0., 0., 1., 0.]))
    actual = I.IORSSmall.place_pose_for(s, start, p.partner, stance)
    assert th.allclose(actual[0], th.tensor(p.place[0]))
    assert th.allclose(actual[1], th.tensor(p.place[1]))
    stance.tangent = th.tensor([1., 0.])
    with pytest.raises(Reject, match="push direction"):
        I.IORSSmall.place_pose_for(s, start, p.partner, stance)
    stance.tangent = th.tensor([0., 1.])
    s.top.polygon_supported = lambda *args, **kwargs: False
    with pytest.raises(Reject, match="leaves the support"):
        I.IORSSmall.place_pose_for(s, start, p.partner, stance)
