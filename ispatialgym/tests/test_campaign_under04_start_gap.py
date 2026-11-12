from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch as th
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.tasks import iors_small as I
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import campaign_runtime
from ispatialgym.sampling.sim_utils import Reject


def draw(monkeypatch, predicate=None, **kw):
    exact=[]
    class IK:
        def feasible_sets(self,cells,yaws,grasps,*args):
            return {0:[(yaws[0],grasps[0])]}
        def exact(self,poses,*args,**kw):
            exact.append(poses)
            return NS(ok=True,reason='',q=None)
    top=NS(z=.8,polygon_supported=lambda *a,**kw:True)
    s=NS(gp=NS(gen=lambda k,d=None:False if k=='grasp_twins' else d),ikc=IK(),bands=None,kit=NS(stats={}),
         world=NS(bottom_offset=lambda *a:.01,footprint=lambda *a:th.zeros(4,2)),
         bindings=NS(one=lambda role:NS(name='spatula.n.01_1')))
    manip=NS(focus=th.tensor([1.,2.,.8]),translate=.3,tilt=0.,name='manipulation',working_arms=['right'])
    g=NS(name='g0',key='spatula__hkvsep',pose=(th.zeros(3),C.IDENTITY_QUAT.clone()),finger_q=None)
    out=TT.phase3_target(s,manip,np.random.default_rng(0),[g],th.zeros(8,3),C.IDENTITY_QUAT,[],
                        top=top,support_z=.8,pose_allowed=predicate,**kw)
    return out,exact


def test_rejected_start_is_resampled_before_ik_with_original_bound(monkeypatch):
    seen=[]
    def allowed(p):
        seen.append(p)
        return len(seen)>=4
    (pose,rec),calls=draw(monkeypatch,allowed)
    assert len(seen)==4 and len(calls)==1
    assert th.equal(pose[0],seen[-1][0])
    assert rec.grasp_name=='g0'
    rejected=[]
    with pytest.raises(Reject,match='caller_pose'):
        draw(monkeypatch,lambda p:rejected.append(p) or False)
    assert len(rejected)==5


def test_no_predicate_preserves_default_draw(monkeypatch):
    (a,_),_=draw(monkeypatch)
    (b,_),_=draw(monkeypatch,lambda p:True)
    assert all(th.equal(x,y) for x,y in zip(a,b))


@pytest.mark.parametrize('cls,enabled,expected',[
 ('inter_object_relation_satisfaction-under_small_scale_04','1',True),
 ('inter_object_relation_satisfaction-under_small_scale_04','0',False),
 ('inter_object_relation_satisfaction-under_small_scale_03','1',False)])
def test_scope_and_exact_partner_gap(monkeypatch,cls,enabled,expected):
    monkeypatch.setenv('ISPATIALGYM_UNDER04_START_GAP_PROPOSAL',enabled)
    monkeypatch.setattr(campaign_runtime,'configuration',lambda:(None,NS(class_id=cls)))
    seen=[]
    monkeypatch.setattr(TT,'phase3_target',lambda *a,**kw:seen.append(kw) or (None,NS(grasp_name='g')))
    monkeypatch.setattr(TT,'polygon_gap',lambda t,p:float(t))
    s=NS(_campaign_pair_active=NS(partner=([0.,0.,0.],[0.,0.,0.,1.])),relation='under',
         grasps_flush=[],grasps=[],neighbor_start_top=lambda:None,pair_goal_align=lambda:None,
         hull={'target':None,'partner':None},rest={'target':None},support_objects=lambda:[],top=None,
         world=NS(footprint=lambda h,p:p[0]),gap_min=.03,
         choice=NS(models={'target':NS(scale=None)}))
    I.IORSSmall.phase3_two_tier(s,None,np.random.default_rng(0))
    assert ('pose_allowed' in seen[0]) is expected
    if expected:
        assert not seen[0]['pose_allowed']((.029,None))
        assert seen[0]['pose_allowed']((.03,None))
