import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_mechanism_probe import scope, ORDERS, DELTA
from scripts.campaign_20261110.training_mechanism_probe import REBUILD_DELTA


def request(control):
    return {'schema':'isg-retained-mechanism-replay-request-v1','campaign_id':'campaign_20261110',
        'control':control,'physical_delta':dict(DELTA),'benchmark_credit':0,'production_released':False,
        'max_native_episodes':len(ORDERS[control]),'timeout_seconds':1800}


@pytest.mark.parametrize('control',list(ORDERS))
def test_exact_retained_source_controls_are_finite_and_never_new_draws(control):
    assert scope(request(control)) == ORDERS[control] and set(scope(request(control))) == {0}


@pytest.mark.parametrize('field,value',[('production_released',True),('benchmark_credit',1),
    ('timeout_seconds',36000),('max_native_episodes',3),('control','ABA'),
    ('physical_delta',{**DELTA,'additional_physics_steps':1})])
def test_control_cannot_expand_episodes_release_scope_or_change_physics(field,value):
    with pytest.raises(ContractError):scope({**request('AA'),field:value})


def test_declared_reconstruction_counts_cold_and_reused_episodes_without_input_changes():
    q={**request('AA'),'physics_reconstruction':'late_incoming_v1','physical_delta':REBUILD_DELTA}
    assert scope(q)==[0,0]
    assert REBUILD_DELTA['additional_physics_steps']==0 and not REBUILD_DELTA['reference_state_copied']


@pytest.mark.parametrize('change',[{'physics_reconstruction':'unknown'},
    {'physics_reconstruction':'late_incoming_v1'}, {'physical_delta':REBUILD_DELTA},
    {'physics_reconstruction':'late_incoming_v1','physical_delta':{**REBUILD_DELTA,'additional_stop_play_pairs':0}}])
def test_reconstruction_cannot_be_unreported_or_use_a_different_native_route(change):
    with pytest.raises(ContractError):scope({**request('AA'),**change})


def test_shared_lifecycle_preserves_existing_step_settings_and_exposes_clock_reset():
    from types import SimpleNamespace as N
    from ispatialgym.oracle.campaign_restore import restart_loaded_physics
    calls=[];sim=N(playing=True,current_time=3.,current_time_step_index=60)
    sim.is_playing=lambda:sim.playing;sim.is_stopped=lambda:not sim.playing
    sim.get_physics_dt=lambda:1/120;sim.get_rendering_dt=lambda:1/30;sim.get_sim_step_dt=lambda:1/30
    sim.step_physics=lambda:calls.append('step');sim._sim_context=N(step=lambda:calls.append('context_step'))
    def stop():sim.playing=False;calls.append('stop')
    def play():sim.playing=True;sim.current_time=0.;sim.current_time_step_index=0;calls.append('play')
    sim.stop=stop;sim.play=play
    before=sim.step_physics
    result=restart_loaded_physics(sim,while_stopped=lambda:calls.append('reconstruct'))
    assert calls==['stop','reconstruct','play'] and sim.step_physics is before
    assert result['observed_physics_step_calls']==[] and result['physics_steps_requested']==0
    assert result['time_step_settings_before']==result['time_step_settings_after']
    assert result['clock_before']['current_time']==3. and result['clock_after']['current_time']==0.


def test_legacy_lifecycle_does_not_opt_unrelated_sources_into_reconstruction():
    from ispatialgym.oracle.campaign_restore import restart_reused_world
    assert restart_reused_world({},None) is None
    with pytest.raises(ContractError):
        restart_reused_world({'scene_setup':{'restoration_options':{'reuse_native_lifecycle_v1':True}}},None)


def additional_fixture(tmp_path, mechanism='camera'):
    import json
    from scripts.campaign_20261110.training_mechanism_probe import ref
    def put(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(value if isinstance(value,str) else json.dumps(value));return ref(p)
    cls='camera_relocalization-standard' if mechanism=='camera' else 'configuration_matching-block_design_01'
    row={'instance_id':'original_source','class_id':cls}
    for k in ('source_tro','source_sidecar','collector_environment','collector_meta','collector_demo',
              'own_replay_meta','generation_row','campaign_config'):
        row[k]=put(k,{'identity':k})
    marker=put('code/.snapshot_commit','original_commit\n')
    row['source_generation_and_collection_runtime']={'path':str(tmp_path/'code'),
        'commit':'original_commit','snapshot_sha256':marker['sha256'],'b1k':put('code/.snapshot_b1k','original_b1k')}
    gate={'success':True,'replay_rc':0,'demo':row['collector_demo']['path'],
          'demo_sha256':row['collector_demo']['sha256'],'instance':row['source_tro']['path'],
          'tro_sha256':row['source_tro']['sha256'],'replay_meta_paths':[row['own_replay_meta']['path']]}
    row['own_replay_gate']=put('gate.json',gate)
    member={'instance_id':row['instance_id'],**{k+'_sha256':row[f]['sha256'] for k,f in (
        ('demo','collector_demo'),('tro','source_tro'),('report','collector_meta'),('sidecar','source_sidecar'))}}
    review={'schema':'isg-camera-release-review-v1' if mechanism=='camera' else 'isg-release-review-v1',
        'class_id':cls,'template_validation_passed':True,'gate':row['own_replay_gate'],
        'members':[member]}
    if mechanism!='camera':review['selected_member_ids']=[row['instance_id']]
    row['first_party_strict_review']=put('review.json',review)
    return {'rows':[row]}, {'mechanism':mechanism,'instance_id':row['instance_id']}, put


@pytest.mark.parametrize('mechanism',['camera','grouped_objects'])
def test_additional_sources_preserve_original_runtime_and_real_review_schema(tmp_path,mechanism):
    from scripts.campaign_20261110.training_mechanism_probe import additional_source
    manifest,q,_=additional_fixture(tmp_path,mechanism)
    result=additional_source(tmp_path,manifest,q)
    assert result['generation_runtime']['commit']=='original_commit'
    assert result['collector_runtime']==result['generation_runtime']
    assert 'strict_distinct_group_review' not in result


@pytest.mark.parametrize('field,value',[('success',False),('demo_sha256','other_demo'),
    ('tro_sha256','other_tro'),('replay_meta_paths',[])])
def test_additional_source_gate_cannot_substitute_inputs_or_own_replay(tmp_path,field,value):
    import json
    from pathlib import Path
    from scripts.campaign_20261110.training_mechanism_probe import additional_source
    manifest,q,put=additional_fixture(tmp_path);row=manifest['rows'][0]
    gate=json.loads(Path(row['own_replay_gate']['path']).read_text());gate[field]=value
    row['own_replay_gate']=put('gate.json',gate)
    with pytest.raises(ContractError,match='exact own replay'):additional_source(tmp_path,manifest,q)


@pytest.mark.parametrize('change',['camera_group_schema','other_gate','ambiguous_source','failed_review','original_marker'])
def test_additional_source_rejects_reinterpreted_review_or_original_identity(tmp_path,change):
    import json
    from pathlib import Path
    from scripts.campaign_20261110.training_mechanism_probe import additional_source
    manifest,q,put=additional_fixture(tmp_path);row=manifest['rows'][0]
    review=json.loads(Path(row['first_party_strict_review']['path']).read_text())
    if change=='camera_group_schema':review['schema']='isg-release-review-v1'
    elif change=='other_gate':review['gate']={'path':'different','sha256':'different'}
    elif change=='failed_review':review['template_validation_passed']=False
    elif change=='ambiguous_source':manifest['rows'].append(row)
    else:row['source_generation_and_collection_runtime']['commit']='new_execution_commit'
    row['first_party_strict_review']=put('review.json',review)
    with pytest.raises(ContractError):additional_source(tmp_path,manifest,q)


def snapshot_runner(fail=False, omitted=False):
    from types import SimpleNamespace as N
    class Registry:
        def __init__(self):self._dump_filter=lambda obj:obj.name=='target'
        def set_dump_filter(self,value):self._dump_filter=value
    registry=Registry();actors=[N(name='target'),N(name='background')]
    class Scene:
        objects=actors;object_registry=registry
        def update_initial_file(self):
            names=[o.name for o in actors if registry._dump_filter(o)]
            if omitted:names.remove('background')
            self._initial_file={'state':{'registry':{'object_registry':dict.fromkeys(names,{})}}}
    scene=Scene()
    class Loader:
        def __init__(self):self.env=N(scene=scene);self.report={}
        def restore(self):
            scene.update_initial_file()
            if fail:raise ValueError('native failure')
            scene.update_initial_file();return 'native result'
    return N(InstanceLoader=Loader),scene,registry


def test_native_restore_snapshots_cover_background_and_restore_exact_recording_filter():
    from scripts.campaign_20261110.training_mechanism_probe import install_complete_snapshots
    runner,scene,registry=snapshot_runner();prior=registry._dump_filter
    install_complete_snapshots(runner);loader=runner.InstanceLoader()
    assert loader.restore()=='native result' and registry._dump_filter is prior
    assert len(loader.report['complete_native_snapshot_inventory'])==2
    assert all(e['complete'] and e['physics_steps_added']==0 for e in loader.report['complete_native_snapshot_inventory'])
    assert 'update_initial_file' not in vars(scene)


@pytest.mark.parametrize('fail,omitted',[(True,False),(False,True)])
def test_snapshot_scope_keeps_failure_evidence_and_restores_native_methods(fail,omitted):
    from scripts.campaign_20261110.training_mechanism_probe import install_complete_snapshots
    runner,scene,registry=snapshot_runner(fail,omitted);prior=registry._dump_filter
    install_complete_snapshots(runner);loader=runner.InstanceLoader()
    with pytest.raises((ValueError,ContractError)):loader.restore()
    assert registry._dump_filter is prior and 'update_initial_file'not in vars(scene)
    assert loader.report['complete_native_snapshot_inventory']
    assert loader.report['complete_scene_snapshot']['prior_filter_restored'] is True


def test_complete_snapshot_request_must_declare_its_changed_physical_boundary():
    from scripts.campaign_20261110.training_mechanism_probe import physical_delta
    q={**request('AA'),'complete_native_snapshots':True}
    with pytest.raises(ContractError):scope(q)
    q['physical_delta']=physical_delta(q)
    assert scope(q)==[0,0] and q['physical_delta']['additional_physics_steps']==0
