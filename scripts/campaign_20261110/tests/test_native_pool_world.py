import copy
import json
from types import SimpleNamespace as N
import pytest
import torch
from ispatialgym.oracle.pool_world import PoolWorld
from ispatialgym.sampling.model_pool import file_reference
from scripts.campaign_20261110.training_pool_world import scene_contract
from scripts.campaign_20261110.tests.test_training_pool_world import scenes
from scripts.campaign_20261110.tests.test_training_pool_native_anchor_probe import state_and_pose


def bound(tmp_path):
    documents=scenes();contract=scene_contract(documents,['campaign_pool_A','campaign_pool_B','campaign_pool_A'],{'sha256':'pool'})
    rows=[]
    for i,doc in enumerate(documents[:2]):
        scene=tmp_path/('scene'+str(i)+'.json');scene.write_text(json.dumps(doc))
        tro=tmp_path/('tro'+str(i)+'.json');tro.write_text('{}')
        rows.append({'tro':file_reference(tro),'scene':file_reference(scene),'active':('campaign_pool_A','campaign_pool_B')[i]})
    contract['members']=[rows[0],rows[1],copy.deepcopy(rows[0])]
    world=PoolWorld(contract,rows[0]['tro']['path'],tmp_path)
    loader=N(tro_path=rows[0]['tro']['path']);world.bind_loader(loader)
    return world,loader,documents


def test_native_union_configuration_is_private_and_does_not_overwrite_source(tmp_path):
    world,loader,documents=bound(tmp_path)
    original={'scene':{'scene_file':'original'},'task':{'tro_path':loader.tro_path}}
    out=world.configuration(original)
    assert original['scene']['scene_file']=='original'
    doc=json.loads(open(out['scene']['scene_file']).read())
    assert set(doc['objects_info']['init_info'])=={'support','campaign_pool_A','campaign_pool_B'}
    assert json.loads(open(world.member['scene']['path']).read())==documents[0]
    with pytest.raises(ValueError,match='preserve'):world.configuration(original)


def test_shared_pool_requires_exact_complete_contract_and_one_loader(tmp_path):
    world,loader,_=bound(tmp_path)
    world.cold_anchor={'own':'anchor'};shared={};world.share(shared)
    assert world.compatible(shared)
    shared['model_pool_world']['contract_sha256']='different support or pool'
    with pytest.raises(ValueError,match='initialized world'):world.compatible(shared)
    with pytest.raises(ValueError,match='one incoming'):world.bind_loader(loader)


def test_shared_build_report_cannot_overwrite_incoming_reader_or_lifecycle_evidence(tmp_path):
    world, loader, _ = bound(tmp_path)
    world.cold_anchor = {'own': 'anchor'}
    first_states = [{'initial': True, 'object_names': ['campaign_pool_A']}]
    first_events = [{'phase': 'world_initialized_before_first_restore', 'source': 'A'}]
    build = {'base_link_mass_kg': {'value': 10},
             'model_pool_serialized_states': first_states,
             'model_pool_world': first_events,
             'pool_exact_task_binding': {'source': 'A'}}
    shared = {'build_report': dict(build)}
    world.share(shared)
    # Match InstanceLoader.attach: it updates an incoming per-loader report
    # from the first build after PoolConsumers has bound its own reader list.
    incoming_states = [{'initial': True, 'object_names': ['campaign_pool_B']}]
    incoming = {'model_pool_serialized_states': incoming_states}
    incoming.update(shared['build_report'])
    first_states.append({'initial': False, 'source': 'A'})
    first_events.append({'phase': 'after_complete_native_loader', 'source': 'A'})
    build['base_link_mass_kg']['value'] = 11
    assert incoming['model_pool_serialized_states'] is incoming_states
    assert len(incoming_states) == 1
    assert incoming['base_link_mass_kg'] == {'value': 10}
    assert 'model_pool_world' not in incoming and 'pool_exact_task_binding' not in incoming
    assert shared['model_pool_world']['cold_anchor'] == world.cold_anchor


def test_cold_anchor_joins_existing_stop_without_starting_another_lifecycle(tmp_path, monkeypatch):
    world,loader,_=bound(tmp_path);calls=[]
    monkeypatch.setattr(world, 'observe_reset', lambda *args: None)
    pose={'position':[1,2,3],'orientation':[0,0,0,1]};loader.tro=N(robot_poses={'robot':[pose]})
    native=(torch.tensor(pose['position']),torch.tensor(pose['orientation']))
    loader.robot=N(is_holonomic_base=True,model='r1',set_position_orientation=lambda **kw:calls.append(kw),
                   root_link=N(get_position_orientation=lambda:native))
    with pytest.raises(ValueError,match='existing stopped'):world.cold_stopped(N(is_stopped=lambda:False))
    world.cold_stopped(N(is_stopped=lambda:True))
    assert calls==[] and world.events[0]['extra_physics_steps']==0
    assert world.events[0]['extra_native_pose_writes']==0
    assert world.events[0]['native_anchor_before_existing_mass_play']==[v.tolist()for v in native]
    native[0][0]+=1
    with pytest.raises(ValueError,match='declared incoming'):world.cold_stopped(N(is_stopped=lambda:True))
    assert calls==[] and len(world.events)==1


@pytest.mark.parametrize('robot_config, expected_frame', [({}, 'scene'), ({'pose_frame': 'world'}, 'world')])
def test_incoming_native_load_is_once_and_preserves_own_anchor_without_copying_reference(tmp_path,monkeypatch,robot_config,expected_frame):
    from scripts.campaign_20261110 import training_pool_adapter as adapter
    from ispatialgym.sampling import model_pool
    from ispatialgym.oracle import pool_robot_anchor
    world,loader,_=bound(tmp_path);state,pose,frame=state_and_pose();calls=[]
    monkeypatch.setattr(world, 'observe_reset', lambda *args: None)
    scene=N(get_position_orientation=lambda:frame)
    def load(value,serialized=False):
        calls.append(('load',copy.deepcopy(value)));assert serialized is False
        root=value['registry']['object_registry']['robot']['root_link']
        pose[0].copy_(root['pos']);pose[1].copy_(root['ori'])
        return 'loaded'
    scene.load_state=load;loader.env=N(scene=scene)
    loader.cfg = {'robots': [robot_config]}
    loader.robot=N(name='robot',root_link=N(get_position_orientation=lambda:pose),
                   set_position_orientation=lambda **kw:calls.append(('anchor',kw)))
    def restart(sim,callback):calls.append(('stop',));callback();calls.append(('play',));return {'observed_physics_step_calls':[]}
    monkeypatch.setattr(adapter,'restart_without_steps',restart)
    def restore_spawn(robot, sim, anchor, *, pose_frame):
        assert pose_frame == expected_frame
        calls.append(('anchor', anchor))
        return {'mode': 'source_cold_stopped_spawn_v1'}
    monkeypatch.setattr(pool_robot_anchor, 'restore_source_cold_spawn', restore_spawn)
    monkeypatch.setattr(model_pool,'restore_export',lambda *args:calls.append(('pool_properties',)))
    world._incoming_state={'anchor':{'position':[1,2,3],'orientation':[0,0,0,1]},'loaded':False}
    before=copy.deepcopy(state)
    assert world.load_incoming(state,N())=='loaded'
    assert [c[0]for c in calls]==['stop','anchor','play','load','pool_properties']
    actual=calls[3][1]['registry']['object_registry']['robot']
    assert torch.equal(actual['root_link']['ori'],pose[1])
    assert actual['joint_pos']==before['registry']['object_registry']['robot']['joint_pos']
    assert torch.equal(state['registry']['object_registry']['robot']['root_link']['ori'],before['registry']['object_registry']['robot']['root_link']['ori'])
    with pytest.raises(ValueError,match='exactly once'):world.load_incoming(state,N())
    assert len(calls)==5


def test_unbound_combined_augmentation_is_rejected_before_physics(tmp_path):
    world,loader,_=bound(tmp_path)
    other=PoolWorld(world.contract,world.member['tro']['path'],tmp_path)
    with pytest.raises(ValueError,match='combined reviewed'):other.bind_loader(N(tro_path=loader.tro_path,_sequential_augmentation={'other':True}))


def test_reset_observations_are_detached_reads_and_avoid_stopped_handles(tmp_path):
    world, loader, _ = bound(tmp_path)
    state = {'joint_vel': torch.tensor([0.25]), 'is_asleep': False}
    pose = (torch.tensor([1., 2., 3.]), torch.tensor([0., 0., 0., 1.]))
    quaternion = N(GetImaginary=lambda: [0., 0., 0.7071096301078796],
                   GetReal=lambda: 0.7071039080619812)
    prim = N(GetAttribute=lambda name: N(Get=lambda: quaternion if 'Rot' in name or 'orient' in name else [0., 0., 0.]))
    sim = N(current_time=2.5, current_time_step_index=300, stopped=True)
    sim.is_stopped = lambda: sim.stopped
    reads = []
    def dump(serialized=False):
        assert not serialized and not sim.stopped
        reads.append('state')
        return state
    link = N(prim=prim, get_position_orientation=lambda: pose)
    # No setters, step method or lifecycle methods exist on these objects.
    loader.robot = N(prim=prim, root_link=link, base_footprint_link=link,
                     _world_base_fixed_joint_prim=prim, dump_state=dump,
                     get_position_orientation=lambda: pose)
    loader.env = N(scene=N(object_registry=lambda kind, name: N(dump_state=dump)))
    loader.report = {}
    world.observe_reset('stopped', sim)
    assert not reads
    sim.stopped = False
    world.observe_reset('playing', sim)
    rows = loader.report['model_pool_reset_observations']
    assert len(reads) == 1 + len(world.consumers.active)
    assert rows[0]['active_states'] == {} and 'state' not in rows[0]['robot']
    assert set(rows[1]['active_states']) == set(world.consumers.active)
    assert rows[1]['robot']['state']['joint_vel'] == [0.25]
    assert rows[1]['robot']['world_joint_usd']['physics:localRot0'][-1] == 0.7071039080619812
    state['joint_vel'][0] = 1.
    pose[0][0] = 4.
    world.member['tro']['sha256'] = 'later mutation'
    assert rows[1]['robot']['state']['joint_vel'] == [0.25]
    assert rows[0]['robot']['root_world_pose'][0][0] == 1.
    assert rows[1]['source_tro']['sha256'] != 'later mutation'
    assert all(row['sim_step'] == 300 and row['state_writes'] == 0 for row in rows)
    def advances_clock():
        sim.current_time_step_index += 1
        return pose
    loader.robot.get_position_orientation = advances_clock
    with pytest.raises(ValueError, match='advanced'):
        world.observe_reset('bad read', sim)
    assert len(rows) == 2
