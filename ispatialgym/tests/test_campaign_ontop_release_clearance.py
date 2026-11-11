from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch as th

from ispatialgym.sampling import campaign_release_clearance as policy
from ispatialgym.sampling.campaign_pair_goals import PairGoalPlan, SCHEMA, validate_pair_plan
from ispatialgym.sampling.campaign_held_goals import GoalPlanError
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2.tasks import iors_small as I


def plan():
    return PairGoalPlan('c','g','eval_standard','t','a'*64,'ontop','target','partner',
        ((.1,.2,.863),(0.,0.,0.,1.)), ((.1,.2,.86),(0.,0.,0.,1.)),
        ((.1,.2,.82),(0.,0.,0.,1.)),1000.,3,'b'*64,release_policy=policy.POLICY)


def test_option_is_boolean_and_scoped_to_exact_class():
    t = NS(class_id=policy.CLASS_ID, task=policy.TASK, scene_setup={'recovery_options': {policy.OPTION: True}})
    assert policy.settings(t) == {'partner_clearance_m':.005,'max_raise_m':.03,'step_m':.001}
    t.class_id = 'inter_object_relation_satisfaction-ontop_small_scale_02'
    with pytest.raises(ValueError, match='audited kielbasa'): policy.settings(t)
    t.scene_setup['recovery_options'][policy.OPTION] = 'true'
    with pytest.raises(ValueError, match='boolean'): policy.settings(t)
    t.scene_setup = {}
    assert policy.settings(t) is None


def test_parser_requires_small_ontop_and_explicit_recovery_identity():
    p=plan();pose=lambda v:{'position':list(v[0]),'orientation':list(v[1])}
    raw={'schema':SCHEMA,'task':policy.TASK,'template_id':'t','scene':'s','room':'r','campaign_id':'c','group_id':'g',
         'target_name':'target','partner_name':'partner','split':'eval_standard','contract_template_sha256':'a'*64,
         'release_pose':pose(p.release),'goal_pose':pose(p.goal),'partner_pose':pose(p.partner),
         'created_at':10.,'budget_seconds':900.,'release_policy':policy.POLICY}
    kw=dict(template_id='t',scene='s',rooms=['r'],relation='ontop',activated_at=10.,now=11.)
    with pytest.raises(GoalPlanError,match='release policy'):validate_pair_plan(raw,**kw)
    raw['clearance_recovery_class']=policy.CLASS_ID
    assert validate_pair_plan(raw,**kw).release_policy==policy.POLICY
    raw['task']='inter_object_relation_satisfaction-ontop_large_scale'
    with pytest.raises(GoalPlanError,match='release policy'):validate_pair_plan(raw,scale='large',**kw)


def test_native_clearance_search_is_bounded_and_keeps_fixed_goal(monkeypatch):
    monkeypatch.setattr(policy,'settings',lambda:dict(partner_clearance_m=.005,max_raise_m=.03,step_m=.001))
    s=I.IORSSmall.__new__(I.IORSSmall);p=plan();s.relation='ontop';s._campaign_pair_active=p
    s.hull={'partner':th.zeros((2,3))};s.world=NS(hull_at=lambda *a:th.tensor([[.1,.2,.82],[.1,.2,.83]]))
    grasp=NS(eef_in_object=(th.zeros(3),th.tensor([0.,0.,0.,1.])))
    s.sidecar=NS(grasp_of=lambda name:grasp);s.bindings=NS(one=lambda role:NS(name=role))
    # Replay measured the current envelope 6.95mm below the partner surface.
    s.open_jaw_clearance=lambda place,*a:float(place[0][2])-(.86+.00695)
    monkeypatch.setattr(I,'gripper_corners_world',lambda place,**kw:th.stack([place[0]]))
    raised=s.campaign_raised_release(p)
    assert .0119<=s.raised_release_facts['raise_m']<=.0131
    assert s.raised_release_facts['jaw_clearance_m']>=.005
    assert raised[0][:2].tolist()==pytest.approx(p.goal[0][:2])
    assert raised[1].tolist()==list(p.release[1])
    s.campaign_group_remaining_s=lambda:100;s.tol=lambda key,default:default;s.gen=lambda key,default=None:default
    target=NS(get_position_orientation=lambda:p.goal)
    s.objects={'target':target,'partner':NS(get_position_orientation=lambda:p.partner)}
    d=Draft();d.summary['campaign_goal_group']=p.summary();d.goal_poses={'target':raised}
    d.pose_targets={'target':{'orientation_metric':'full'},'partner':{'orientation_metric':'full'}}
    s.campaign_verify_pair(d,settled=True)
    assert d.pose_targets['target']['position']==list(p.goal[0])
    assert d.pose_targets['target']['orientation']==list(p.goal[1])
    target.get_position_orientation=lambda:(p.goal[0],(0.,0.,1.,0.))
    with pytest.raises(Reject,match='target drifted'):s.campaign_verify_pair(d,settled=True)
    s.open_jaw_clearance=lambda *a:-.001
    with pytest.raises(Reject,match='highest allowed raise'):s.campaign_raised_release(p)
    s._campaign_pair_active=replace(p,release_policy='fixed')
    with pytest.raises(Reject,match='explicit fixed-goal'):s.raised_release()
