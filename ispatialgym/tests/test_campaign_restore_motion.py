from types import SimpleNamespace
import pytest
import torch
from ispatialgym.oracle.campaign_restore import restore_serialized_root_motion


def meta():
    return {'campaign_id':'campaign_20261110','split':'train','class_id':'disembedding_07','task':'disembedding',
        'runtime':{'purpose':'training_pipeline_canary'},'scene_setup':{'restoration_options':{'serialized_root_motion_v1':True}}}


class Body:
    kinematic_only=False
    def __init__(self):self.linear=torch.tensor([9.,8.,7.]);self.angular=torch.tensor([6.,5.,4.]);self.pose=(1,2,3)
    def get_linear_velocity(self):return self.linear
    def get_angular_velocity(self):return self.angular
    def set_linear_velocity(self,v):self.linear=v
    def set_angular_velocity(self,v):self.angular=v


def test_default_is_unchanged_and_scope_is_fail_closed():
    body=Body();assert restore_serialized_root_motion({}, {'x':body},{}) is None
    assert body.linear.tolist()==[9,8,7]
    for change in ({'split':'eval_standard'},{'class_id':'disembedding_08'},{'runtime':{'purpose':'production'}}):
        with pytest.raises(ValueError,match='scope'):restore_serialized_root_motion({**meta(),**change},{'x':body},{})


def test_restores_actual_nonzero_serialized_values_without_pose_or_step_changes():
    body=Body();state={'root_link':{'lin_vel':[.1,-.2,.3],'ang_vel':[1.,2.,-3.]}}
    proof=restore_serialized_root_motion(meta(),{'x':body},{'x':state})
    assert body.linear.tolist()==pytest.approx([.1,-.2,.3]) and body.angular.tolist()==[1,2,-3]
    assert body.pose==(1,2,3) and proof['physics_steps_added']==0 and not proof['pose_or_goal_changes']
    assert proof['objects']['x']['serialized']['angular_velocity']==[1,2,-3]


def test_missing_later_record_prevents_partial_mutation():
    body=Body();state={'root_link':{'lin_vel':[0,0,0],'ang_vel':[0,0,0]}}
    with pytest.raises(ValueError,match='missing or invalid'):
        restore_serialized_root_motion(meta(),{'x':body,'y':Body()},{'x':state,'y':{'root_link':{}}})
    assert body.linear.tolist()==[9,8,7]


def test_wrong_native_readback_is_rejected():
    body=Body();body.set_angular_velocity=lambda v:None
    with pytest.raises(ValueError,match='readback'):
        restore_serialized_root_motion(meta(),{'x':body},{'x':{'root_link':{'lin_vel':[0,0,0],'ang_vel':[0,0,0]}}})
