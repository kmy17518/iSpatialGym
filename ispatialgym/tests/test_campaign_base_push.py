"""Fixed large-push goals retain native geometry and reject settling drift."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.campaign_pair_goals import PairGoalPlan
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2.tasks import iors_large as IL, iors_large_nextto as NT


def fixture():
    lo, hi = th.tensor([-.2, -.2, 0.]), th.tensor([.2, .2, .8])
    plo, phi = th.tensor([1., -.5, 0.]), th.tensor([2., .5, .8])
    edges = IL.aabb_edges(plo, phi)
    side = next(x for x in edges if x.name == '-x')
    bound = NT.nextto_bound(lo, hi, plo, phi)
    gap = bound*.5
    goal = ((1-.2-gap, 0., .003), (0., 0., 0., 1.))
    push = dict(side=side.name, side_inward=side.inward.tolist(), face='-x', direction=[1., 0., 0.], goal_gap_m=gap, nextto_bound_m=bound)
    plan = PairGoalPlan('fresh', 'g', 'train', 't', 'a'*64, 'nextto', 'target', 'partner', goal,
                        ((goal[0][0], 0., 0.), goal[1]), ((1.5, 0., .4), goal[1]), 1000., 3, 'b'*64,
                        scale='large', base_push=push)
    s = SimpleNamespace(_campaign_pair_active=plan, relation='nextto', gp=SimpleNamespace(gen=lambda k: None),
        place={'edges': edges, 'local_lo': lo, 'local_hi': hi, 'partner_aabb': (plo, phi)}, floor_z=0.,
        tname='target', pname='partner', rest=th.tensor(goal[1]), hull=None,
        world=SimpleNamespace(bottom_offset=lambda *a: 0.), _check_deadline=lambda *a: None,
        campaign_group_remaining_s=lambda: 100, kit=SimpleNamespace(sync_world=lambda: None),
        template_facts={}, _campaign_latest_proposal={},
        sidecar=SimpleNamespace(contacts={'push': SimpleNamespace(arm='right', extras={'contact_in_object': [-.2, 0., .4]})}))
    return s, plan


def test_fixed_goal_retains_native_floor_and_sweep_checks_with_fresh_starts(monkeypatch):
    s, plan = fixture()
    checked = []
    monkeypatch.setattr(NT, '_face_candidates', lambda *a: [{}])
    monkeypatch.setattr(NT, 'footprint_free', lambda s, xy, yaw, clearance: checked.append(tuple(float(v) for v in xy)) or None)
    monkeypatch.setattr(NT, 'floor_names_at', lambda *a: ['floor'])
    monkeypatch.setattr(NT, 'single_floor', lambda names: False)
    monkeypatch.setattr(NT, '_goal_nextto', lambda *a: True)
    monkeypatch.setattr(NT, 'put', lambda *a: None)
    def make(s, rng, side, face, yaw, d2, init_pose, goal_pose, dist, gap, bound, nt, stats):
        draft = Draft()
        draft.object_poses['target'] = init_pose
        draft.goal_poses['target'] = goal_pose
        draft.summary['nextto'] = {}
        return draft
    monkeypatch.setattr(NT, '_push_and_stances', make)
    starts = []
    for seed in (1, 2, 3):
        d = NT._draw(s, np.random.default_rng(seed))
        assert tuple(d.goal_poses['target'][0].tolist()) == pytest.approx(plan.release[0])
        assert d.summary['campaign_goal_group']['requested_goal'] == plan.summary()['requested_goal']
        starts.append(tuple(d.object_poses['target'][0].tolist()))
    assert len(set(starts)) == 3
    assert len(checked) > 20  # Whole target footprint and intermediate swept corridor still checked.
    monkeypatch.setattr(NT, 'footprint_free', lambda *a: 'blocked: chair')
    with pytest.raises(Reject, match='none admits the push'):
        NT._draw(s, np.random.default_rng(1))


def test_changed_side_angle_or_floor_goal_is_rejected(monkeypatch):
    s, p = fixture()
    assert len(NT.goal_candidates(s, np.random.default_rng(1))) == NT.DEFAULTS['goal_tries']
    s._campaign_pair_active = replace(p, base_push={**p.base_push, 'direction': [0., 1., 0.]})
    with pytest.raises(Reject, match='native angle set'):
        NT.goal_candidates(s, np.random.default_rng(1))
    s._campaign_pair_active = replace(p, release=((p.release[0][0], 0., .1), p.release[1]))
    monkeypatch.setattr(NT, '_face_candidates', lambda *a: [{}])
    with pytest.raises(Reject, match='not floor-resting'):
        NT._draw(s, np.random.default_rng(1))


def test_settled_start_cannot_drag_goal_and_campaign_disabled_keeps_native_behavior():
    s, plan = fixture()
    s.tol = lambda key, default: default
    s.sidecar.stance = lambda name: SimpleNamespace(base_pose2d=[0, 0, 0], translate=.1, tilt=0.)
    s.sidecar.set_goal = lambda *a, **kw: None
    s.campaign_verify_pair = lambda draft: None
    d = Draft()
    d.object_poses['target'] = (th.tensor(plan.release[0])-th.tensor([2., 0., .003]), th.tensor(plan.release[1]))
    d.summary['nextto'] = {'direction': [1., 0., 0.], 'push_distance_m': 2.}
    NT.goal(s, d)
    assert tuple(d.goal_poses['target'][0].tolist()) == pytest.approx(plan.release[0])
    d.object_poses['target'][0][0] += .02
    with pytest.raises(Reject, match='fixed endpoint'):
        NT.goal(s, d)
    s._campaign_pair_active = None
    NT.goal(s, d)
    assert float(d.goal_poses['target'][0][0]) == pytest.approx(plan.release[0][0]+.02)
