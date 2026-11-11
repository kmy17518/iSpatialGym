from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling.campaign_held_goals import GoalPlanError
from ispatialgym.sampling.campaign_shelf_goals import checked_context, author_context, same_start_grasp, positive_rest_arms, CLASS
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


def shelf_anchor():
    ctx = raw()['shelf_tip_context']
    return {'summary': {'shelf_start': {
        **{k: ctx[k] for k in ('target_board','arm','target_lean_deg','partner_lean_deg','bay_m','spine_behind_edge_m')},
        'board_draw': {'grasp_key': ctx['grasp_key'], 'row_z': ctx['row_z']}},
        'touching': {'tip_release': True}, 'manipulation_stance': {'edge': ctx['edge_name']}},
        'planned_start_poses': {'target': ([0, 0, .6], [0, 0, 0, 1])}, 'sources': raw()['anchor_sources']}


def test_refill_preserves_fixed_partner_depth_after_target_start_draw():
    a = shelf_anchor()
    assert author_context(a,target_name='target',support_name='bookcase')['spine_behind_edge_m'] == .06
    start = a['summary']['shelf_start']
    start['spine_behind_edge_m'] = .0443
    start['target_depth_proposal'] = {'source_depth_m': .06, 'target_depth_m': .04427400200423898,
        'partner_depth_unchanged': True, 'goal_unchanged': True}
    parent = raw()['shelf_tip_context']
    release = {'position': [.1, -.122109, .6], 'orientation': [0, 0, 0, 1]}
    a['summary']['campaign_goal_group'] = {'phase': 'settled_goal_verified', 'relation': 'touching',
        'target_name': 'target', 'plan_sha256': 'c'*64, 'shelf_tip_context': parent,
        'requested_release': release, 'requested_place': release}
    a['planned_goal_poses'] = {'target': (release['position'], release['orientation'])}
    a['planned_start_poses']['target'] = ([0, .015726, .6], [0, 0, 0, 1])
    context = author_context(a,target_name='target',support_name='bookcase')
    assert context['spine_behind_edge_m'] == .06
    assert context['source_target_start'] == parent['source_target_start']
    assert context['source_target_start']['position'] != a['planned_start_poses']['target'][0]
    assert context['inherited_group_plan_sha256'] == 'c'*64
    # A geometry reference cannot be inherited for a different release.
    a['planned_goal_poses']['target'] = ([.1, -.10, .6], release['orientation'])
    with pytest.raises(GoalPlanError, match='source release'):
        author_context(a,target_name='target',support_name='bookcase')
    assert context['source_summary_sha256'] == a['sources']['summary']['sha256']
    assert start['spine_behind_edge_m'] == .0443


@pytest.mark.parametrize('mutation', [
    {'partner_depth_unchanged': False}, {'goal_unchanged': False},
    {'source_depth_m': float('nan')}, {'target_depth_m': .08}])
def test_refill_rejects_inconsistent_depth_provenance(mutation):
    a = shelf_anchor(); start = a['summary']['shelf_start']; start['spine_behind_edge_m'] = .0443
    start['target_depth_proposal'] = {'source_depth_m': .06, 'target_depth_m': .044274,
        'partner_depth_unchanged': True, 'goal_unchanged': True, **mutation}
    with pytest.raises(GoalPlanError): author_context(a,target_name='target',support_name='bookcase')


def test_diversity_filter_distinguishes_pose_and_grasp():
    p=plan().goal
    assert same_start_grasp(p,'side1',p,'side1')
    assert not same_start_grasp(p,'side1',p,'side2')
    assert not same_start_grasp(p,'side1',((.02,0,.8),p[1]),'side1')


def test_positive_arms_keep_native_opt_in_and_exact_row():
    c = context()
    source = {'key': c['grasp_key'], 'rows': [.45], 'arms': {.45: ['left', 'right']}}
    different_rest = {'key': 'book@lean15', 'rows': [.45], 'arms': {.45: ['right']}}
    different_row = {'key': 'book@upper', 'rows': [.8], 'arms': {.8: ['left', 'right']}}
    pool = [source, different_rest, different_row]
    assert positive_rest_arms(c, pool, .45, other_arm=False) == [(source, 'left')]
    assert positive_rest_arms(c, pool, .45, other_arm=True) == [
        (source, 'left'), (source, 'right'), (different_rest, 'right')]
    with pytest.raises(GoalPlanError, match='source shelf'):
        positive_rest_arms(c, [different_rest], .45, other_arm=True)


def test_other_arm_never_implies_an_unmapped_grasp():
    c = context()
    source = {'key': c['grasp_key'], 'rows': [.45], 'arms': {.45: ['left']}}
    assert positive_rest_arms(c, [source], .45, other_arm=True) == [(source, 'left')]


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


def test_target_depth_proposal_is_bounded_and_does_not_mutate_goal_context():
    from copy import deepcopy
    from types import SimpleNamespace
    from ispatialgym.sampling.campaign_shelf_goals import target_depth_proposal
    context = {'spine_behind_edge_m': .06, 'partner_pose': [1, 2, 3], 'goal': [4, 5, 6]}
    original = deepcopy(context)
    depth, facts = target_depth_proposal(context, SimpleNamespace(uniform=lambda lo, hi: hi))
    assert depth == .09 and facts['interval_m'] == [.03, .09]
    assert facts['partner_depth_unchanged'] and facts['goal_unchanged']
    assert context == original
    for value in (-.01, float('nan'), .51, True):
        with pytest.raises(GoalPlanError, match='source shelf depth'):
            target_depth_proposal({'spine_behind_edge_m': value}, None)


def test_depth_draw_changes_only_target_start_preserving_partner_check(monkeypatch):
    ctx = context()
    p = replace(plan('touching'), shelf_tip_context=ctx)
    calls = []
    target_hull, partner_hull = object(), object()
    def lean_pose(s, hull, rest, *args, spine_behind=None, **kwargs):
        calls.append(('target' if hull is target_hull else 'partner', spine_behind))
        value = ((0., float(spine_behind), .7), p.partner[1]) if hull is target_hull else p.partner
        return tuple(th.tensor(v) for v in value), {'position':list(value[0])}
    s = SimpleNamespace(_campaign_pair_active=p, _shelf_template={'key':'key', 'lean_deg':20.},
        shelf_start={'lean_side':1.}, _shelf_side_layouts=[(0.,1.)], _shelf_draw={'layout':[0.,'left']},
        edge=SimpleNamespace(name=ctx['edge_name']), support=SimpleNamespace(name='bookcase'), top=None,
        hull={'target':target_hull,'partner':partner_hull}, rest={}, segment_arms=['left'],
        shelf_start_draw=lambda:{'tries':1,'lean_deg':None,'spine_behind_edge_m':None},
        gen=lambda key, default=None:default, tol=lambda key,default:default,
        shelf_rests=lambda *args:(th.tensor([0.,0.,0.,1.]),th.tensor([0.,0.,0.,1.])),
        world=SimpleNamespace(hull_at=lambda *args:None), gp=SimpleNamespace(variant='test'))
    monkeypatch.setattr(I.TT,'shelf_lean_pose',lean_pose)
    monkeypatch.setattr(I.TT,'lean_rest_side',lambda *a:1.)
    monkeypatch.setattr(I,'shelf_free_span',lambda *a:.2)
    monkeypatch.setattr(I,'shelf_span_rule_on',lambda:False)
    monkeypatch.setattr(I,'merge_shelf_pose_facts',lambda *a:None)
    from ispatialgym.sampling import campaign_runtime
    monkeypatch.setattr(campaign_runtime,'contact_recovery',lambda name:name=='shelf_target_depth_draw')
    I.IORSSmall.draw_shelf_layout(s,SimpleNamespace(uniform=lambda lo,hi:hi))
    assert calls == [('target',.09),('partner',.09),('partner',.06)]
    assert s.shelf_pose[0][1] == pytest.approx(.09)
    assert s.shelf_partner_pose[0].tolist() == pytest.approx(p.partner[0])
    assert s._shelf_draw['target_depth_proposal']['source_depth_m'] == .06


def test_drawn_member_cannot_invent_an_ancestor_reference():
    a=shelf_anchor();start=a['summary']['shelf_start']
    start['spine_behind_edge_m']=.0443
    start['target_depth_proposal']={'source_depth_m':.06,'target_depth_m':.044274,
        'partner_depth_unchanged':True,'goal_unchanged':True}
    with pytest.raises(GoalPlanError,match='verified parent geometry'):
        author_context(a,target_name='target',support_name='bookcase')
