from contextlib import nullcontext
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling import campaign_runtime as CR
from ispatialgym.sampling.campaign_inside_goal_grasp import fixed_goal_grasp_predicate, goal_stance_order
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import GraspRecord, SidecarDraft, Stance
from scripts.campaign_20261110.contracts import ContractError


@pytest.mark.parametrize('option', ['inside_goal_grasp_proposal', 'inside_goal_stance_proposal'])
def test_goal_grasp_policy_is_exact_template_scoped(monkeypatch, option):
    objects = [NS(role='target', category='brussels_sprouts', model='mbkrxe', default_scale=(1.,)*3),
               NS(role='partner', category='bowl', model='belcml', default_scale=(1.,)*3)]
    t = NS(class_id='inter_object_relation_satisfaction-inside_01', task='inter_object_relation_satisfaction-inside_small_scale',
           split='eval_standard', objects=objects, scene_setup={'recovery_options': {option: True}})
    monkeypatch.setattr(CR, 'active', lambda: False)
    assert not CR.contact_recovery(option)
    monkeypatch.setattr(CR, 'active', lambda: True)
    monkeypatch.setattr(CR, 'configuration', lambda: (None, t, None, None))
    assert CR.contact_recovery(option)
    objects[1].model = 'other'
    with pytest.raises(ContractError): CR.contact_recovery(option)


def test_goal_stance_ranking_keeps_native_pool_and_fresh_pick_focus():
    anchor = {'base_pose2d': [1., 2., 0.], 'translate': 0., 'tilt': -.6, 'focus': [0., 0., .8]}
    candidates = [NS(base_pose2d=(1., 2., 0.), translate=0., tilt=-.6, focus=[.3, .2, .8]),
                  NS(base_pose2d=(1.5, 2., 0.), translate=0., tilt=-.6, focus=[0., 0., .8]),
                  NS(base_pose2d=(1., 2., 0.), translate=0., tilt=-.6, focus=[.4, .2, .8]),
                  NS(base_pose2d=(1., 2., 0.), translate=0., tilt=0., focus=[0., 0., .8])]
    order = [1, 3, 2, 0]
    ranked = goal_stance_order(order, candidates, anchor)
    assert ranked == [0, 2, 1, 3] and set(ranked) == set(order)
    assert order == [1, 3, 2, 0] and candidates[0].focus == [.3, .2, .8]
    assert goal_stance_order(order, candidates, None) is order
    # A candidate absent from the filtered native pool is never introduced.
    assert goal_stance_order([1, 3], candidates, anchor) == [1, 3]


def test_full_native_place_screen_keeps_partner_and_restores_candidate_state(monkeypatch):
    from ispatialgym.sampling.v2.tasks import iors_small as I
    sidecar = SidecarDraft('inside', 'iors')
    original = Stance('manipulation', (0., 0., 0.), 0.)
    sidecar.add_stance(original); old_grasp = object(); sidecar.grasps.append(old_grasp)
    stance = Stance('manipulation', (1., 0., 0.), .5, arms=['right'])
    identity = th.tensor([0., 0., 0., 1.]); zero = (th.zeros(3), identity)
    calls = []
    def exact(chain, st, arm, **kw):
        calls.append((chain, st, arm, kw))
        assert sidecar.stance('manipulation') is stance and len(sidecar.grasps) == 1
        return NS(ok=float(chain[-1][0][0]) > 0)
    monkeypatch.setattr(I, 'attached', lambda *a, **kw: nullcontext())
    s = NS(_campaign_pair_active=NS(partner=zero, partner_name='bowl'), sidecar=sidecar,
           goal_pose=lambda *a: zero, pre_place_of=lambda p: (p[0]+th.tensor([0.,0.,.1]), p[1]),
           kit=object(), obj=lambda role: role, support_objects=lambda: ['table'],
           choice=NS(models={'target': NS(scale=(1.,)*3)}), ikc=NS(exact=exact))
    check = fixed_goal_grasp_predicate(s)
    for amount, expected in [(0., False), (.01, True), (.01, True)]:
        eio = (th.tensor([amount, 0., 0.]), identity)
        rec = GraspRecord('sprout', 'right', 'key', 'top_7', 0., zero, zero, eef_in_object=eio)
        assert check(stance, zero, rec) is expected
        assert sidecar.stance('manipulation') is original and sidecar.grasps == [old_grasp]
        assert rec.place_on is None
    assert len(calls) == 2  # cache only within this fixed stance / goal proposal search
    for chain, st, arm, kw in calls:
        assert len(chain) == 2 and kw['ignore_for_last'] == ['table']
        assert kw['approached'] == [] and kw['cache'] is False


def test_native_draw_continues_after_goal_incompatible_pick_without_demoting_map(monkeypatch):
    from ispatialgym.sampling.v2 import map_f as MF
    zero = (th.zeros(3), th.tensor([0., 0., 0., 1.]))
    grasps = [NS(name=n, key=n, finger_q=[]) for n in ('bad_goal', 'good_goal')]
    stance = NS(name='manipulation')
    s = NS(world=NS(footprint=lambda *a: None), ikc=NS(exact=lambda *a, **kw: NS(ok=True)), counters={})
    monkeypatch.setattr(MF, 'draw_order', lambda *a: [0, 1])
    monkeypatch.setattr(TT.SS, 'count_protrusion_hit', lambda *a: None)
    monkeypatch.setattr(TT, 'policy_opening_of', lambda g: None)
    monkeypatch.setattr(TT, '_policy_opening', lambda *a: nullcontext())
    monkeypatch.setattr(TT.NB, 'note_grasp_from', lambda *a: None)
    monkeypatch.setattr(TT, '_scene_mask_draw_note', lambda *a: None)
    seen = []; fails = {'support': 0, 'ik': 0}
    def goal_allowed(st, pose, rec):
        seen.append(rec.grasp_name)
        return rec.grasp_name == 'good_goal'
    out = TT._phase3_draw(s, [(0., g, ['right']) for g in grasps], np.random.default_rng(7),
        0., 1, 0., th.zeros(3), lambda xy, yaw: zero, th.zeros((4, 3)),
        NS(polygon_supported=lambda *a, **kw: True), .02, .8, .8, lambda g: zero,
        NS(grasp_pose=lambda *a: zero, pre_pose=lambda p: p), {'right': stance}, {}, False,
        {'right': None}, 'sprout', ['right'], 'right', stance, fails, None, lambda g:g.name, set(),
        grasp_record_allowed=goal_allowed)
    assert out[1].grasp_name == 'good_goal' and seen == ['bad_goal', 'good_goal']
    assert fails == {'support': 0, 'ik': 0, 'caller_grasp_record': 1}
