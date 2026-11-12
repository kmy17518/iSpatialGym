from copy import deepcopy
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_sequential_augmentation import one_world,compare_physics_reference,observed_augmentation_change
from scripts.campaign_20261110.contracts import digest


def observed(sequence):
    state={'objects':{},'lights':{'lamp':1200.},'skybox':{'intensity':1000.,'color':[1.,1.,1.]}}
    context={'physics_steps_added':0,'dynamic_roots':{'target':{'mass':2.,'sleep_threshold':.001,'solver_position_iteration_count':16,'solver_velocity_iteration_count':1}}}
    return {'process_session':'native:1','scene_runtime_id':4,'sim_runtime_id':5,'sequence':sequence,
            'loaded_state':state,'loaded_state_sha256':digest(state),'native_dynamics_context':context,
            'native_dynamics_context_sha256':digest(context)}


def test_constructor_count_and_native_ids_both_required():
    proof={'world_construction':{'method':'omnigibson.Environment','attempted':1,'completed':1}}
    rows=[observed(1),observed(2),observed(3)]
    assert one_world(proof,rows)=='native:1'
    rows[2]['scene_runtime_id']=6
    with pytest.raises(ContractError,match='rebuilt'):one_world(proof,rows)
    rows[2]['scene_runtime_id']=4;proof['world_construction']['completed']=2
    with pytest.raises(ContractError,match='constructor'):one_world(proof,rows)
    proof['world_construction']['completed']=1;rows[2]['sequence']=2
    with pytest.raises(ContractError,match='order'):one_world(proof,rows)


def test_independent_reference_checks_final_lighting_and_reset_mass():
    a={'boundaries':[observed(i)for i in range(3)],'step':{'after':observed(4)}}
    b=deepcopy(a);compare_physics_reference(a,b)
    b['boundaries'][-1]['loaded_state']['lights']['lamp']=1300
    with pytest.raises(ContractError,match='intensity'):compare_physics_reference(a,b)
    b=deepcopy(a);b['boundaries'][0]['native_dynamics_context']['dynamic_roots']['target']['mass']=3.
    b['boundaries'][0]['native_dynamics_context_sha256']=digest(b['boundaries'][0]['native_dynamics_context'])
    with pytest.raises(ContractError,match='mass'):compare_physics_reference(a,b)


def augmentation_fixture(tmp_path, mutation=None):
    import json
    from scripts.campaign_20261110.contracts import file_digest
    def put(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value))
        return {'path':str(path),'sha256':file_digest(path)}
    stable=put('stable.json',{'objects_info':{'init_info':{'chair':{'args':{'fixed_base':True}}}},
        'state':{'registry':{'object_registry':{'chair':{'root_link':{'pos':[1.,2.,3.],'ori':[0.,0.,0.,1.]}}}}}})
    records=[];states=[]
    for index,factor in enumerate((1.2,.9)):
        color_factor=1.02 if index==0 else .98
        lighting={'schema':'isg-lighting-v1','intensity_factor':factor,'skybox_color_factors':[color_factor,1.,1.]}
        position=[20.,30.,-1000.] if index==0 else [1.,2.,3.]
        perturbations={'lighting':lighting,'removed_objects':['chair'] if index==0 else []}
        augmentation={'lighting':{'request':lighting,'fixtures':{'lamp':{'default':1000.,'applied':1000.*factor}},
            'skybox':{'default':{'intensity':500.,'color':[1.,1.,1.]},
                      'applied':{'intensity':500.*factor,'color':[color_factor,1.,1.]}}},
            'moved_objects':{'chair':{'moved_to':position}} if index==0 else {}}
        state={'objects':{'chair':{'fixed_base':True,'position':position.copy(),'orientation':[0.,0.,0.,1.]},
            'target':{'position':[float(index),0.,0.]}},'lights':{'lamp':1000.*factor},
            'skybox':{'intensity':500.*factor,'color':[color_factor,1.,1.]}}
        if mutation:mutation(index,augmentation,state)
        tro=put(f'tro{index}.json',{'ispatialgym':{'campaign':{'perturbations':perturbations,
            'runtime':{'augmentation':augmentation}}}})
        records.append({'template':{'scene_setup':{'sequential_augmentation_base':{'scene':stable}}},
                        'perturbations':perturbations,'artifacts':{'tro':tro}})
        states.append(state)
    return records,states


def test_changed_augmentation_requires_measured_declared_actor_and_lighting(tmp_path):
    records,states=augmentation_fixture(tmp_path)
    result=observed_augmentation_change(tmp_path,records,*states)
    assert result['changed_fixtures']==['lamp'] and result['skybox_changed'] is True
    assert result['parking_displacement_m']['chair']>1000.
    assert result['native_source_positions_and_lighting_checked'] is True


@pytest.mark.parametrize('fault',['unchanged_lighting','unchanged_parking','wrong_skybox',
    'missing_actor','wrong_orientation','no_material_light_change','no_material_parking_change','nonfinite'])
def test_changed_augmentation_rejects_consistently_wrong_cold_and_reused_states(tmp_path,fault):
    def mutate(index,augmentation,state):
        if fault=='no_material_light_change':
            augmentation['lighting']['fixtures']['lamp']['applied']=0.
            augmentation['lighting']['skybox']['applied']={'intensity':0.,'color':[0.,0.,0.]}
            state['lights']['lamp']=0.;state['skybox']={'intensity':0.,'color':[0.,0.,0.]}
        if fault=='no_material_parking_change' and index==0:
            augmentation['moved_objects']['chair']['moved_to']=[1.,2.,3.]
            state['objects']['chair']['position']=[1.,2.,3.]
        if index!=1:return
        if fault=='unchanged_lighting':state['lights']['lamp']=1200.
        if fault=='unchanged_parking':state['objects']['chair']['position']=[20.,30.,-1000.]
        if fault=='wrong_skybox':state['skybox']['color']=[1.2,1.,1.]
        if fault=='missing_actor':state['objects'].pop('chair')
        if fault=='wrong_orientation':state['objects']['chair']['orientation']=[1.,0.,0.,0.]
        if fault=='nonfinite':state['objects']['chair']['position'][0]=float('nan')
    records,states=augmentation_fixture(tmp_path,mutate)
    with pytest.raises(ContractError):observed_augmentation_change(tmp_path,records,*states)


def cleanup_fixture(tmp_path,monkeypatch,mutation=None):
    """Keep hashes/state checks real; replace only expensive upstream audits."""
    import json
    from scripts.campaign_20261110 import training_sequential_augmentation as module
    from scripts.campaign_20261110.contracts import file_digest
    def put(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
        return {'path':str(p),'sha256':file_digest(p)}
    def ref(p):return {'path':str(p),'sha256':file_digest(p)}
    stamp=tmp_path/'runtime/.snapshot_commit';stamp.parent.mkdir();stamp.write_text('frozen')
    reviews=[put('review_A.json',{}),put('review_B.json',{})]
    source={'instance_id':'A','export_record':{'validation_key':{'exact':'A'},'perturbations':{'parking':'A'}}}
    requests={};matrix={'schema':'isg-sequential-render-cleanup-input-v1','controls':{}}
    for name in ['fresh_A','failure','interruption']:
        control={'fresh_A':'FRESH_A','failure':'RENDER_FAILURE_A','interruption':'RENDER_INTERRUPT_A'}[name]
        request={'canaries':reviews,'diagnostic_control':control,'render_cleanup_only':name!='fresh_A','probe_runtime_sha256':file_digest(stamp)}
        request_ref=put(name+'/request.json',request);requests[request_ref['path']]=request
        probe={'request':request_ref,'probe_runtime':ref(stamp),'runtime_compatibility':{'verified':True},'kind':'kinematic_observation_rendering',
               'world_construction':{'attempted':1,'completed':1,'method':'omnigibson.Environment'},'entries':[]}
        for i in range(1 if name=='fresh_A'else 2):
            observation=observed(i);observation.update(process_session=name,validation_key=source['export_record']['validation_key'],
                perturbations_sha256=digest(source['export_record']['perturbations']),scheduler_job_id='123',simulator_module='omnigibson.sim')
            report={'tag':'REPLAY_DEMO','kept':True,'demo':'/native/demo/A','episode_index':i,
                'campaign_restoration':{'scale_verified':True,'native_observation':observation}}
            entry={'canary_review':reviews[0]}
            if name!='fresh_A'and i==0:
                report.update(tag='REPLAY_EPISODE_ABORTED',kept=False,cleanup={
                    'exception_type':'ControlledRenderFailure'if name=='failure'else'ControlledRenderInterrupt',
                    'boundary':'after_native_writer_action'if name=='failure'else'after_existing_render',
                    'frames_observed':2 if name=='failure'else 1,'native_writer_abort_called':True,
                    'trajectory_buffer_empty':True,'dataset_pending_frames':False,'writer_aborted':True,
                    'physics_steps_added':0,'render_ticks_added':0,'expected_controlled_failure':True})
                entry['expected_controlled_failure']=True
            else:
                output=tmp_path/(name+'/video.mp4');output.write_bytes(b'hashed-video')
                entry['outputs']=[ref(output)]
            if mutation:mutation(name,i,report,probe,entry)
            entry['report']=put(name+f'/report{i}.json',report);probe['entries'].append(entry)
        item={'probe':put(name+'/probe.json',probe),'execution':put(name+'/execution.json',{})}
        if name=='fresh_A':matrix['independent_A']=item
        else:matrix['controls'][name]=item
    monkeypatch.setattr(module,'inputs',lambda root,path:(requests[path],[source]))
    monkeypatch.setattr(module,'validated_canary',lambda *args:source)
    monkeypatch.setattr(module,'verify_probe_execution',lambda *args:{'verified':True})
    monkeypatch.setattr(module,'verify_runtime_compatibility',lambda *args:{'verified':True})
    return module.verify_render_cleanup(tmp_path,put('matrix.json',matrix),0)


def test_cleanup_audit_requires_clean_buffers_and_independent_next_episode(tmp_path,monkeypatch):
    result=cleanup_fixture(tmp_path,monkeypatch)
    assert result['status']=='verified'and result['production_released']is False


@pytest.mark.parametrize('fault',['pending_buffer','wrong_exception','added_step','wrong_boundary','two_worlds','wrong_next_light','same_process','missing_native_identity'])
def test_cleanup_audit_rejects_invalid_native_evidence(tmp_path,monkeypatch,fault):
    def mutate(name,index,report,probe,entry):
        if name!='failure':return
        if index==0:
            if fault=='pending_buffer':report['cleanup']['dataset_pending_frames']=True
            if fault=='wrong_exception':report['cleanup']['exception_type']='OSError'
            if fault=='added_step':report['cleanup']['physics_steps_added']=1
            if fault=='wrong_boundary':report['cleanup']['boundary']='after_scene_load'
            if fault=='two_worlds':probe['world_construction']['completed']=2
        observation=report['campaign_restoration']['native_observation']
        if fault=='same_process':observation['process_session']='fresh_A'
        if fault=='missing_native_identity':observation.pop('scheduler_job_id')
        if fault=='wrong_next_light'and index==1:
            observation['loaded_state']['lights']['lamp']=1500
            observation['loaded_state_sha256']=digest(observation['loaded_state'])
    with pytest.raises(ContractError):cleanup_fixture(tmp_path,monkeypatch,mutate)
