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
