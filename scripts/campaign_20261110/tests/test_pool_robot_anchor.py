from contextlib import nullcontext
from types import SimpleNamespace as N

import pytest
import torch

from ispatialgym.oracle.pool_robot_anchor import author_native_root_joint


class Attribute:
    def __init__(self, value): self.value = value
    def Get(self): return self.value
    def Set(self, value): self.value = value


def fixture():
    calls = []
    incoming = {'position':[1.,2.,3.], 'orientation':[0.,0.,0.70710677,0.70710677]}
    own = (torch.tensor(incoming['position']), torch.tensor([1e-8,0.,0.70710677,0.70710677]))
    attributes = {'physics:localPos0':Attribute((0.,0.,0.)),
                  'physics:localRot0':Attribute((1.,0.,0.,0.))}
    joint = N(GetAttribute=attributes.get,GetPath=lambda:'/robot/rootJoint')
    robot = N(is_holonomic_base=True,_world_base_fixed_joint_prim=joint,_ag_obj_in_hand={},
              root_link=N(get_position_orientation=lambda:own),
              set_position_orientation=lambda **kw:calls.append(kw))
    sim = N(is_stopped=lambda:True,editing_usd=nullcontext)
    return robot,sim,incoming,own,attributes,calls


def test_authors_own_native_root_not_raw_input_or_an_external_reference():
    robot,sim,incoming,own,attrs,calls=fixture()
    result=author_native_root_joint(robot,sim,incoming,quaternion=lambda *v:v)
    assert calls==[incoming]
    assert attrs['physics:localRot0'].Get()==tuple(own[1][[3,0,1,2]].tolist())
    assert attrs['physics:localRot0'].Get()!=tuple(incoming['orientation'][i]for i in(3,0,1,2))
    assert result['reference_state_copied'] is False and result['extra_stop_play_pairs']==0
    assert result['joint_pose_attribute_writes']==2


def test_wrong_lifecycle_or_unclean_grasp_is_rejected_before_pose_write():
    robot,sim,incoming,own,attrs,calls=fixture()
    sim.is_stopped=lambda:False
    with pytest.raises(ValueError,match='existing stopped'):author_native_root_joint(robot,sim,incoming)
    sim.is_stopped=lambda:True;robot._ag_obj_in_hand={'left':object()}
    with pytest.raises(ValueError,match='cleanup'):author_native_root_joint(robot,sim,incoming)
    assert calls==[]


def test_native_readback_must_remain_at_incoming_source_before_joint_write():
    robot,sim,incoming,own,attrs,calls=fixture();own[0][0]+=.01
    before=[a.Get()for a in attrs.values()]
    with pytest.raises(ValueError,match='exact incoming'):author_native_root_joint(robot,sim,incoming,quaternion=lambda *v:v)
    assert [a.Get()for a in attrs.values()]==before
