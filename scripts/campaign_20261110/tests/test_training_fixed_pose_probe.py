import pytest
import torch
from scripts.campaign_20261110.training_fixed_pose_probe import idempotent_fixed_pose_scope
from scripts.campaign_20261110.training_mechanism_probe import physical_delta
from scripts.campaign_20261110.contracts import ContractError


class Xform:
    prim_path='/World/fixed'
    kinematic_only=False
    def __init__(self):
        self.pose=(torch.tensor([1.,2.,3.]),torch.tensor([0.,0.,0.,1.]))
        self.calls=[]
    def get_position_orientation(self,frame='world'):return self.pose
    def set_position_orientation(self,position=None,orientation=None,frame='world'):
        self.calls.append((position,orientation,frame))


class Kinematic(Xform):pass


def test_only_exact_fixed_world_pose_is_skipped():
    k=Kinematic();dynamic=Xform();parent=Xform();parent.kinematic_only=True;events=[]
    original=Xform.set_position_orientation
    with idempotent_fixed_pose_scope(Xform,Kinematic,events):
        for obj in [k,parent]:obj.set_position_orientation(*obj.pose)
        dynamic.set_position_orientation(*dynamic.pose)
        k.set_position_orientation(k.pose[0]+1e-6,k.pose[1])
        k.set_position_orientation(*k.pose,frame='scene')
        k.set_position_orientation(position=None,orientation=k.pose[1])
        k.set_position_orientation(k.pose[0],-k.pose[1])
    assert Xform.set_position_orientation is original
    assert len(k.calls)==4 and len(dynamic.calls)==1 and parent.calls==[]
    assert [e['skipped'] for e in events]==[True,True,False,False]


def test_setter_restored_after_exception_and_bad_input_uses_native_validation():
    k=Kinematic();original=Xform.set_position_orientation;events=[]
    with pytest.raises(RuntimeError):
        with idempotent_fixed_pose_scope(Xform,Kinematic,events):
            k.set_position_orientation(torch.tensor([float('nan'),2.,3.]),k.pose[1])
            raise RuntimeError('native failure')
    assert Xform.set_position_orientation is original and len(k.calls)==1 and events==[]


def test_physical_delta_requires_explicit_valid_mode():
    assert 'idempotent_fixed_pose_sets' not in physical_delta({})
    assert 'bit-identical' in physical_delta({'idempotent_fixed_pose_sets':True})['idempotent_fixed_pose_sets']
    with pytest.raises(ContractError):physical_delta({'idempotent_fixed_pose_sets':'enabled'})
