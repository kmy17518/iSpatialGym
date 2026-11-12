import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_native_anchor_probe import anchored_scene_state, preserve_anchor_at_scene_load


def state_and_pose():
    pose = (torch.tensor([13.49996566772461, 1.699981689453125, .005277156829833984]),
            torch.tensor([4.0265727875521407e-05, 7.211606134660542e-05, -.5557019710540771, .8313815593719482]))
    scene_pose = (torch.zeros(3), torch.tensor([0., 0., 0., 1.]))
    root = {'pos': pose[0].clone(), 'ori': pose[1].clone(), 'lin_vel': torch.tensor([1., 2., 3.]),
            'ang_vel': torch.tensor([4., 5., 6.])}
    root['ori'][2] = -.5557020306587219
    state = {'pos': scene_pose[0], 'ori': scene_pose[1], 'registry': {'object_registry': {
        'robot': {'root_link': root, 'joint_pos': [1., 2.], 'joint_vel': [3., 4.],
                  'controller_groups': {'base': {'target': [2., 3., 4.]}}, 'is_asleep': False},
        'bottle': {'root_link': {'pos': [7., 8., 9.]}}}}}
    return state, pose, scene_pose


def test_exact_native_pose_preserves_all_other_incoming_state_and_input():
    state, pose, scene_pose = state_and_pose(); before = copy.deepcopy(state)
    result = anchored_scene_state(state, 'robot', pose, scene_pose)
    actual = result['registry']['object_registry']['robot']; saved = before['registry']['object_registry']['robot']
    assert torch.equal(actual['root_link']['ori'], pose[1])
    assert torch.equal(state['registry']['object_registry']['robot']['root_link']['ori'], saved['root_link']['ori'])
    for key in ('joint_pos', 'joint_vel', 'controller_groups', 'is_asleep'):
        assert actual[key] == saved[key]
    for key in ('lin_vel', 'ang_vel'):
        assert torch.equal(actual['root_link'][key], saved['root_link'][key])
    assert result['registry']['object_registry']['bottle'] == before['registry']['object_registry']['bottle']
    actual['root_link']['ori'][0] = 0
    assert pose[1][0] != 0


def test_rebased_scene_cannot_silently_reuse_world_pose():
    state, pose, scene_pose = state_and_pose()
    with pytest.raises(ContractError, match='rebasing'):
        anchored_scene_state(state, 'robot', pose, (scene_pose[0] + 1., scene_pose[1]))


def test_scene_restore_wrapper_uses_own_current_pose_and_restores_native_method():
    state, pose, scene_pose = state_and_pose(); calls = []
    def load(value, serialized=False):
        calls.append(value)
        root = value['registry']['object_registry']['robot']['root_link']
        pose[0].copy_(root['pos']); pose[1].copy_(root['ori'])
        return 'loaded'
    scene = SimpleNamespace(load_state=load, get_position_orientation=lambda: scene_pose)
    robot = SimpleNamespace(name='robot', root_link=SimpleNamespace(get_position_orientation=lambda: pose))
    sim = SimpleNamespace(is_playing=lambda: True)
    with preserve_anchor_at_scene_load(scene, robot, sim) as evidence:
        assert scene.load_state(state, serialized=False) == 'loaded'
        with pytest.raises(ContractError, match='once'): scene.load_state(state)
    assert scene.load_state is load and len(calls) == 1 and evidence[0]['native_root_bit_identical']


def test_failure_unwinds_wrapper_and_records_native_difference():
    state, pose, scene_pose = state_and_pose()
    def bad_load(*args, **kwargs): pose[1][2] += 1e-5
    scene = SimpleNamespace(load_state=bad_load, get_position_orientation=lambda: scene_pose)
    robot = SimpleNamespace(name='robot', root_link=SimpleNamespace(get_position_orientation=lambda: pose))
    with pytest.raises(ContractError, match='changed preserved'):
        with preserve_anchor_at_scene_load(scene, robot, SimpleNamespace(is_playing=lambda: True)) as evidence:
            scene.load_state(state)
    assert scene.load_state is bad_load and evidence[0]['native_root_bit_identical'] is False
