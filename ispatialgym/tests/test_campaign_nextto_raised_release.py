"""Keep the settled nextto goal fixed while native grasps choose release height."""
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling.campaign_pair_goals import PairGoalPlan
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2.tasks import iors_small as I


def plan():
    return PairGoalPlan('c','g','eval_standard','t','a'*64,'nextto','target','partner',
        ((.1,.2,.823),(0.,0.,0.,1.)), ((.1,.2,.8),(0.,0.,0.,1.)),
        ((.1,.25,.8),(0.,0.,0.,1.)),1000.,3,'b'*64,release_policy='native_raised_height')


def test_distinct_grasps_keep_fixed_xy_and_native_clearance_and_raise_limit(monkeypatch):
    p=plan()
    grasp=SimpleNamespace(eef_in_object=(th.zeros(3),th.tensor([0.,0.,0.,1.])))
    s=I.IORSSmall.__new__(I.IORSSmall)
    s.relation='nextto'
    s.raised_release=lambda:{'partner_clearance_m':.005,'max_raise_m':.03,'step_m':.001}
    s.hull={'partner':th.zeros((2,3))}
    s.world=SimpleNamespace(hull_at=lambda *a:th.tensor([[.1,.25,.8],[.1,.25,.81]]))
    s.sidecar=SimpleNamespace(grasp_of=lambda name:grasp)
    s.bindings=SimpleNamespace(one=lambda role:SimpleNamespace(name=role))
    s.open_jaw_clearance=lambda place,*a:float(place[0][2])-.818
    monkeypatch.setattr(I,'gripper_corners_world',lambda place,**kw:th.stack([place[0]]))
    first=s.campaign_raised_release(p)
    assert .023 <= s.raised_release_facts['raise_m'] <= .024
    assert s.raised_release_facts['jaw_clearance_m'] >= .005
    grasp.eef_in_object=(th.tensor([0.,0.,.015]),grasp.eef_in_object[1])
    second=s.campaign_raised_release(p)
    assert .008 <= s.raised_release_facts['raise_m'] <= .009
    assert float(first[0][2]-second[0][2]) == pytest.approx(.015,abs=.0011)
    assert second[0][:2].tolist()==pytest.approx(p.release[0][:2])
    assert second[1].tolist()==list(p.release[1])
    # A grasp that cannot clear the partner within the original30mm cap fails.
    grasp.eef_in_object=(th.tensor([0.,0.,-.1]),grasp.eef_in_object[1])
    with pytest.raises(Reject,match='highest allowed raise'):s.campaign_raised_release(p)


def test_raised_release_does_not_change_settled_goal_or_orientation_checks():
    p=plan();s=I.IORSSmall.__new__(I.IORSSmall);s._campaign_pair_active=p
    s.campaign_group_remaining_s=lambda:100
    s.tol=lambda key,default:default;s.gen=lambda key,default=None:default
    target=SimpleNamespace(get_position_orientation=lambda:p.goal)
    s.objects={'target':target,'partner':SimpleNamespace(get_position_orientation=lambda:p.partner)}
    s.raised_release_facts={'raise_m':.008,'jaw_clearance_m':.0057,'partner_clearance_m':.005,'max_raise_m':.03}
    d=Draft();d.summary['campaign_goal_group']=p.summary();d.pose_targets={'target':{'orientation_metric':'full'},'partner':{'orientation_metric':'full'}}
    d.goal_poses={'target':(th.tensor([.1,.2,.808]),th.tensor([0.,0.,0.,1.]))}
    s.campaign_verify_pair(d,settled=True)
    assert d.goal_poses['target'][0][2]==pytest.approx(.808)
    assert d.pose_targets['target']['position']==list(p.goal[0])
    assert d.pose_targets['target']['orientation']==list(p.goal[1])
    target.get_position_orientation=lambda:((.15,.2,.8),p.goal[1])
    with pytest.raises(Reject,match='campaign_goal_drift'):s.campaign_verify_pair(d,settled=True)
    target.get_position_orientation=lambda:p.goal
    s.raised_release_facts=None
    with pytest.raises(Reject,match='clearance evidence'):s.campaign_verify_pair(d)


def test_native_raise_policy_cannot_apply_without_the_native_raised_release():
    s=I.IORSSmall.__new__(I.IORSSmall);s.relation='nextto';s.raised_release=lambda:None
    with pytest.raises(Reject,match='native tabletop'):s.campaign_raised_release(plan())


def test_release_policy_parser_is_restricted_to_small_nextto():
    from ispatialgym.sampling.campaign_pair_goals import SCHEMA,validate_pair_plan
    from ispatialgym.sampling.campaign_held_goals import GoalPlanError
    p=plan();pose=lambda value:{'position':list(value[0]),'orientation':list(value[1])}
    raw={'schema':SCHEMA,'task':'inter_object_relation_satisfaction-nextto_small_scale',
         'template_id':'t','scene':'s','room':'r','campaign_id':'c','group_id':'g',
         'target_name':'target','partner_name':'partner','split':'eval_standard',
         'contract_template_sha256':'a'*64,'release_pose':pose(p.release),'goal_pose':pose(p.goal),
         'partner_pose':pose(p.partner),'created_at':10.,'budget_seconds':900.,
         'release_policy':'native_raised_height'}
    kwargs=dict(template_id='t',scene='s',rooms=['r'],activated_at=10.,now=11.)
    assert validate_pair_plan(raw,relation='nextto',**kwargs).release_policy=='native_raised_height'
    raw['task']='inter_object_relation_satisfaction-inside_small_scale'
    with pytest.raises(GoalPlanError,match='release policy'):validate_pair_plan(raw,relation='inside',**kwargs)
