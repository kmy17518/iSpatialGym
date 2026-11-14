from types import SimpleNamespace as N

import pytest
import torch

from ispatialgym.oracle.pool_robot_anchor import restore_source_cold_spawn


class Attribute:
    def __init__(self, value): self.value = value
    def Get(self): return self.value
    def Set(self, value): raise AssertionError('do not reauthor the native setter result')


class Quaternion:
    def __init__(self, xyzw): self.value = tuple(xyzw)
    def GetImaginary(self): return self.value[:3]
    def GetReal(self): return self.value[3]


def fixture():
    calls = []
    # Native84ec cold B: entity and fixed joint have different quaternions;
    # the native tensor-tuple joint position also differs from Python floats.
    incoming = {'position': [15.306231498718262, -9.287864685058594, 0.005274295806884766],
                'orientation': [-6.322842091321945e-05, 5.155465623829514e-05,
                                0.7071096301078796, 0.707103967666626]}
    own = (torch.tensor(incoming['position']),
           torch.tensor([-6.322843546513468e-05, 5.1554659876273945e-05,
                         0.7071096301078796, 0.7071039080619812]))
    attributes = {'physics:localPos0':Attribute((0.,0.,0.)),
                  'physics:localRot0':Attribute(Quaternion([0.,0.,0.,1.]))}
    joint = N(GetAttribute=attributes.get,GetPath=lambda:'/robot/rootJoint')
    def native_setter(**kwargs):
        calls.append(kwargs)
        attributes['physics:localPos0'].value = (15.3046875, -9.2890625, 0.005275726318359375)
        attributes['physics:localRot0'].value = Quaternion(
            [-6.322842818917707e-05, 5.155465260031633e-05, 0.7071096301078796, 0.7071039080619812])
    robot = N(is_holonomic_base=True,_world_base_fixed_joint_prim=joint,_ag_obj_in_hand={},
              root_link=N(get_position_orientation=lambda:own),
              set_position_orientation=native_setter)
    sim = N(is_stopped=lambda:True)
    return robot,sim,incoming,own,attributes,calls


def test_source_frame_setter_result_is_preserved_without_reauthoring_from_root():
    robot,sim,incoming,own,attrs,calls=fixture()
    result=restore_source_cold_spawn(robot,sim,incoming,pose_frame='scene')
    assert calls==[{**incoming, 'frame':'scene'}]
    assert result['native_joint_pose'][0]==list(attrs['physics:localPos0'].Get())
    assert result['native_joint_pose'][0]!=own[0].tolist()
    assert result['native_joint_pose'][1]==list(attrs['physics:localRot0'].Get().value)
    assert result['native_joint_pose'][1]!=own[1].tolist()
    assert result['native_joint_pose_preserved'] is True
    assert result['native_setter_calls']==1 and result['source_pose_frame']=='scene'
    assert result['reference_state_copied'] is False and result['extra_stop_play_pairs']==0
    assert result['joint_pose_attribute_writes']==0


def test_wrong_lifecycle_or_unclean_grasp_is_rejected_before_pose_write():
    robot,sim,incoming,own,attrs,calls=fixture()
    sim.is_stopped=lambda:False
    with pytest.raises(ValueError,match='existing stopped'):restore_source_cold_spawn(robot,sim,incoming,pose_frame='scene')
    sim.is_stopped=lambda:True
    with pytest.raises(ValueError,match='unsupported'):restore_source_cold_spawn(robot,sim,incoming,pose_frame='parent')
    robot._ag_obj_in_hand={'left':object()}
    with pytest.raises(ValueError,match='cleanup'):restore_source_cold_spawn(robot,sim,incoming,pose_frame='scene')
    assert calls==[]


def test_native_readback_must_remain_at_incoming_source_without_extra_writes():
    robot,sim,incoming,own,attrs,calls=fixture();own[0][0]+=.01
    with pytest.raises(ValueError,match='exact incoming'):
        restore_source_cold_spawn(robot,sim,incoming,pose_frame='scene')
    assert len(calls)==1
    assert attrs['physics:localPos0'].Get()==(15.3046875, -9.2890625, 0.005275726318359375)
