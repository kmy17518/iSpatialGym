from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
import torch

from ispatialgym.oracle.campaign_batch import (
    SCHEMA, capture_robot_anchor, prepare_robot_anchor, restore_robot_anchor_stopped, verify_robot_anchor,
)
from scripts.campaign_20261110.contracts import ContractError, file_digest


def fixture(tmp_path):
    anchor = torch.tensor([5., 13.488058, .005275])
    robot = SimpleNamespace(name='robot_r1', model='r1pro', is_holonomic_base=True,
        root_link=SimpleNamespace(get_position_orientation=lambda: (anchor.clone(), torch.tensor([0., 0., 0., 1.]))))
    def loader(name, y):
        path=tmp_path/(name+'.json');path.write_text(json.dumps({'source':name}))
        return SimpleNamespace(robot=robot, tro_path=str(path),
            tro=SimpleNamespace(robot_poses={'robot':[{'position':[5.,y,.005275], 'orientation':[0.,0.,0.,1.]}]}),
            env=SimpleNamespace(scene=SimpleNamespace(convert_world_pose_to_scene_relative=lambda p,q:(p,q))))
    a=loader('A',float(anchor[1]));b=loader('B',13.525492)
    original={'registry':{'object_registry':{'robot_r1':{'root_link':{'pos':anchor.clone(),'ori':torch.tensor([0.,0.,0.,1.])},
        'joint_pos':torch.tensor([0.,0.,0.,1.537])},'chair':{'root_link':{'pos':[8.,9.,0.]}}}}}
    shared={'cold_robot_anchor':capture_robot_anchor(a), 'scene_state0':original}
    contract={'schema':SCHEMA,'members_by_tro':{x.tro_path:{'tro':{'path':x.tro_path,'sha256':file_digest(x.tro_path)}}for x in [a,b]}}
    return a,b,shared,contract


def test_translated_B_and_return_A_match_cold_anchor_without_changing_inputs(tmp_path):
    a,b,shared,contract=fixture(tmp_path);source=deepcopy(shared['scene_state0'])
    bs,br=prepare_robot_anchor(b,shared,contract)
    ast,ar=prepare_robot_anchor(a,shared,contract)
    assert br['anchor_position']==pytest.approx([5.,13.525492,.005275],abs=1e-7)
    assert ar['anchor_position']==shared['cold_robot_anchor']['anchor_position']
    assert torch.equal(bs['registry']['object_registry']['robot_r1']['joint_pos'],source['registry']['object_registry']['robot_r1']['joint_pos'])
    assert torch.equal(shared['scene_state0']['registry']['object_registry']['robot_r1']['root_link']['pos'],source['registry']['object_registry']['robot_r1']['root_link']['pos'])
    assert bs['registry']['object_registry']['chair']==source['registry']['object_registry']['chair']
    assert torch.equal(ast['registry']['object_registry']['robot_r1']['root_link']['pos'],source['registry']['object_registry']['robot_r1']['root_link']['pos'])
    assert br['physics_steps_added']==0


def test_anchor_write_requires_stopped_native_lifecycle(tmp_path):
    a,b,shared,contract=fixture(tmp_path);_,record=prepare_robot_anchor(b,shared,contract);calls=[]
    b.robot.set_position_orientation=lambda **kw:calls.append(kw)
    with pytest.raises(ContractError,match='stopped lifecycle'):
        restore_robot_anchor_stopped(b.robot,SimpleNamespace(is_stopped=lambda:False),record)
    assert not calls
    restore_robot_anchor_stopped(b.robot,SimpleNamespace(is_stopped=lambda:True),record)
    assert calls==[{'position':record['anchor_position'],'orientation':record['anchor_orientation']}]


def test_different_orientation_or_mutated_source_cannot_rebase(tmp_path):
    a,b,shared,contract=fixture(tmp_path)
    b.tro.robot_poses['robot'][0]['orientation']=[0.,0.,.1,.994987]
    with pytest.raises(ContractError,match='rotation unvalidated'):prepare_robot_anchor(b,shared,contract)
    b.tro.robot_poses['robot'][0]['orientation']=[0.,0.,0.,1.]
    from pathlib import Path
    Path(b.tro_path).write_text('changed')
    with pytest.raises(ContractError,match='source changed'):prepare_robot_anchor(b,shared,contract)


def test_missing_cold_capture_is_not_reconstructed_from_reused_state(tmp_path):
    a,b,shared,contract=fixture(tmp_path);shared.pop('cold_robot_anchor')
    with pytest.raises(ContractError,match='cold sequential robot anchor missing'):prepare_robot_anchor(b,shared,contract)


def test_original_A_anchor_cannot_pass_B_native_readback(tmp_path):
    a,b,shared,contract=fixture(tmp_path);_,record=prepare_robot_anchor(b,shared,contract)
    with pytest.raises(ContractError,match='readback differs'):verify_robot_anchor(b.robot,record)
    b.robot.root_link.get_position_orientation=lambda:(torch.tensor(record['anchor_position']),torch.tensor(record['anchor_orientation']))
    verify_robot_anchor(b.robot,record)
    assert record['native_readback']['boundary'].endswith('before_task_restore')
