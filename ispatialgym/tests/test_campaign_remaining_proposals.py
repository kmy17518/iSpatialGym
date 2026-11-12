from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling import campaign_runtime as CR
from ispatialgym.sampling import campaign_attachment_goals as AG
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks import iors_small as I
from ispatialgym.tests.test_campaign_attachment_goals import plan
from scripts.campaign_20261110.contracts import ContractError


def bind(monkeypatch, cls, task, option):
    t=NS(class_id=cls,task=task,split='eval_standard',scene_setup={'recovery_options':{option:True}},
         objects=[NS(role='target',category='cap',model='xxqyrt',default_scale=(1.,)*3)])
    monkeypatch.setattr(CR,'active',lambda:True)
    monkeypatch.setattr(CR,'configuration',lambda:(None,t,None,None))
    return t


def test_cap_tries_measured_stance_without_discarding_exploration(monkeypatch):
    hint={'base_pose2d':[9.1918,3.4242,3.139], 'focus':[8.5218,3.4259,.8182], 'translate':.5,'tilt':-.4542}
    p=replace(plan(),anchor_manipulation_stance=hint)
    candidates=[NS(**{**hint,'focus':[8.8,3.1,.8182]})for _ in range(8)]+[NS(**hint)]
    order=list(range(9));monkeypatch.setattr(CR,'active',lambda:False)
    assert AG.order_source_stances(p,'configuration_matching-attachment_02',order,candidates) is order
    t=bind(monkeypatch,'configuration_matching-attachment_02','configuration_matching-attachment','cap_source_stance_proposal')
    got=AG.order_source_stances(p,t.class_id,order,candidates)
    assert got[:4]==order[:4] and got[4]==8 and sorted(got)==order
    t.objects[0].model='other'
    with pytest.raises(ContractError):AG.order_source_stances(p,t.class_id,order,candidates)


@pytest.mark.parametrize('reject_top',[False,True])
def test_inside_top_tier_keeps_previous_checked_fallbacks(monkeypatch,reject_top):
    # Local approach z points down for the top grasp and sideways for the flush one.
    top=NS(name='top_7',pose=(th.zeros(3),th.tensor([1.,0.,0.,0.])))
    side=NS(name='side_1',pose=(th.zeros(3),th.tensor([0.,.70710678,0.,.70710678])))
    s=NS(relation='inside',grasps_flush=[side],grasps=[side,top],rest={'target':th.tensor([0.,0.,0.,1.])},
         neighbor_start_top=lambda:None,pair_goal_align=lambda:None,widest_faces=lambda:False,
         init_align_support=lambda:False,hull={'target':None},support_objects=lambda:[],top=None,
         choice=NS(models={'target':NS(scale=None)}),gp=NS(variant='inside'))
    calls=[]
    def draw(_s,_manip,_rng,grasps,*args,**kw):
        calls.append(grasps)
        if reject_top and grasps==[top]:raise Reject('grasp_unreachable','native IK rejects top')
        return 'pose',NS(grasp_name=grasps[0].name)
    monkeypatch.setattr(I.TT,'phase3_target',draw);monkeypatch.setattr(I.TT,'_count',lambda *a:None)
    monkeypatch.setattr(CR,'active',lambda:False)
    assert I.IORSSmall.phase3_two_tier(s,None,np.random.default_rng(0))[1].grasp_name=='side_1'
    calls.clear();t=bind(monkeypatch,'inter_object_relation_satisfaction-inside_01','inter_object_relation_satisfaction-inside_small_scale','inside_top_grasp_proposal')
    result=I.IORSSmall.phase3_two_tier(s,None,np.random.default_rng(0))
    assert calls[0]==[top] and result[1].grasp_name==('side_1'if reject_top else'top_7')
    if reject_top:assert calls[1]==[side]
    t.class_id='inter_object_relation_satisfaction-inside_04'
    with pytest.raises(ContractError):I.IORSSmall.phase3_two_tier(s,None,np.random.default_rng(0))
