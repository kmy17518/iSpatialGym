from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling.campaign_held_goals import GoalPlanError
from ispatialgym.sampling.campaign_shelf_goals import checked_context, author_context, same_start_grasp, CLASS
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks import iors_small as I
from ispatialgym.tests.test_campaign_pair_goals import plan


def raw():
    return {'class_id': CLASS, 'anchor_sources': {'summary': {'sha256': 'a'*64}, 'sidecar': {'sha256': 'b'*64}},
            'shelf_tip_context': {'mode': 'native_shelf_tip', 'support_name': 'bookcase',
                'grasp_key': 'book@lean_left', 'edge_name': '+u', 'target_board': 'left', 'arm': 'left',
                'row_z': .45, 'bay_m': 0., 'target_lean_deg': 20., 'partner_lean_deg': 15.,
                'spine_behind_edge_m': .06, 'source_target_start': {'position': [0, 0, .6], 'orientation': [0, 0, 0, 1]},
                'source_summary_sha256': 'a'*64, 'source_sidecar_sha256': 'b'*64}}


def context(value=None):
    p=plan('touching')
    return checked_context(raw() if value is None else value, relation='touching', scale='small', release=p.release, place=p.release)


@pytest.mark.parametrize('mutation', [
    lambda r:r.update(class_id='another'),
    lambda r:r['shelf_tip_context'].update(mode='push'),
    lambda r:r['shelf_tip_context'].update(source_summary_sha256='c'*64),
    lambda r:r['shelf_tip_context'].update(target_lean_deg=float('nan')),
    lambda r:r['shelf_tip_context'].update(partner_lean_deg=60),
])
def test_context_rejects_unbound_or_incompatible_sources(mutation):
    r=raw();mutation(r)
    with pytest.raises(GoalPlanError):context(r)


def test_tip_release_cannot_be_relabelled_tabletop_push():
    p=plan('touching')
    with pytest.raises(GoalPlanError, match='must equal'):
        checked_context(raw(), relation='touching', scale='small', release=p.release, place=p.goal)
    assert context()['source_target_start'] == ((0.,0.,.6),(0.,0.,0.,1.))


def test_source_author_requires_actual_tip_release():
    a={'summary':{'shelf_start':{'board_draw':{}},'touching':{'tip_release':False}}}
    with pytest.raises(GoalPlanError):author_context(a,target_name='target',support_name='bookcase')


def test_diversity_filter_distinguishes_pose_and_grasp():
    p=plan().goal
    assert same_start_grasp(p,'side1',p,'side1')
    assert not same_start_grasp(p,'side1',p,'side2')
    assert not same_start_grasp(p,'side1',((.02,0,.8),p[1]),'side1')


def test_tip_place_uses_release_not_generic_push_gap():
    p=replace(plan('touching'),shelf_tip_context=context(),place=plan().release)
    s=SimpleNamespace(_campaign_pair_active=p,relation='touching')
    value=I.IORSSmall.place_pose_for(s,None,None,None)
    assert value[0].tolist()==pytest.approx(p.release[0])


def test_shelf_partner_stays_exact_and_never_enters_tabletop_brace(monkeypatch):
    p=replace(plan('touching'),shelf_tip_context=context())
    s=SimpleNamespace(_campaign_pair_active=p,shelf_partner_pose=tuple(th.tensor(v)for v in p.partner))
    monkeypatch.setattr(I.TT,'partner_pose_why',lambda *a,**k:pytest.fail('shelf has its native board check'))
    value=I.IORSSmall.place_partner(s,None,None,None,None)
    assert value[0].tolist()==pytest.approx(p.partner[0]);assert s._brace_gate['ok'] is False
    s.shelf_partner_pose=((0,0,.71),(0,0,0,1))
    with pytest.raises(Reject,match='lost its fixed partner'):I.IORSSmall.place_partner(s,None,None,None,None)


def test_failed_visibility_never_reserves_a_shelf_member(monkeypatch):
    p=replace(plan('touching'),shelf_tip_context=context())
    s=I.IORSSmall.__new__(I.IORSSmall);s._campaign_pair_active=p;s.relation='touching'
    d=Draft();d.object_poses={'target':p.release};d.summary={'grasp':{'name':'side1'}}
    def fail(*args):raise Reject('goal_occluded','native screen failed')
    monkeypatch.setattr(I.SamplerV2,'check_goal_occlusion',fail)
    with pytest.raises(Reject):s.check_goal_occlusion(d,'staged')
    assert not hasattr(s,'_campaign_group_written')
    monkeypatch.setattr(I.SamplerV2,'check_goal_occlusion',lambda *a:None)
    s.check_goal_occlusion(d,'screened')
    assert s._campaign_group_written==1
    assert s._campaign_written_shelf_starts[p.sha256]==[(p.release,'side1')]
