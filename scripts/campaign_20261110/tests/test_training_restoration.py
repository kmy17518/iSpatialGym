from copy import deepcopy
import json
import pytest
from scripts.campaign_20261110.contracts import ContractError,file_digest
from scripts.campaign_20261110.training_restoration import same_loaded_state
from scripts.campaign_20261110.training_batch_probe import inputs,verify_runtime_compatibility,shared_world_proof


def test_legacy_canary_policy_projection_keeps_original_unknown_and_exact_physics():
    from scripts.campaign_20261110.training_restoration import same_canary_export
    from ispatialgym.sampling.campaign_bucket_policy import POLICY as BUCKET_POLICY
    saved={'artifacts':{'tro':{'sha256':'original'}},'accepted':False,'model':'original'}
    current={**deepcopy(saved),'generation_bucket_policy':None,'acceptance_bucket_policy':None,
             'release_bucket_policy':BUCKET_POLICY}
    assert same_canary_export(saved,current) and set(saved)=={'artifacts','accepted','model'}
    for key,value in [('generation_bucket_policy',BUCKET_POLICY),('acceptance_bucket_policy',BUCKET_POLICY),
                      ('release_bucket_policy',{}),('accepted',True),('model','other'),
                      ('artifacts',{'tro':{'sha256':'changed'}}),('unknown_extra_field',True)]:
        changed={**current,key:value}
        assert not same_canary_export(saved,changed),key
    known={**saved,'generation_bucket_policy':{'version':1}}
    assert not same_canary_export(known,current)


def state():
    return {'objects':{'o':{'category':'book','model':'abc','fixed_base':False,'kinematic_only':False,'root_prim_type':'RigidDynamicPrim','scale':[1,1,1],
        'position':[1,2,3],'orientation':[0,0,0,1],'joint_positions':[.2],'joint_velocities':[0],
        'linear_velocity':[0,0,0],'angular_velocity':[0,0,0]}},'lights':{'lamp':1000},
        'skybox':{'intensity':2000,'color':[1,1,1]}}


@pytest.mark.parametrize('field,value', [('joint_positions',[.3]),('joint_velocities',[.1]),
    ('position',[1.1,2,3]),('scale',[.9,.9,.9]),('linear_velocity',[.1,0,0])])
def test_loaded_state_rejects_observed_contamination(field,value):
    a=state();b=deepcopy(a);b['objects']['o'][field]=value
    with pytest.raises(ContractError,match='contaminated'):same_loaded_state(a,b)
    same_loaded_state(a,deepcopy(a))


def test_probe_rejects_same_instance_and_configuration_mixing(tmp_path):
    refs=[]
    for i in range(2):
        artifact=tmp_path/f'demo{i}';artifact.write_bytes(b'demo')
        p=tmp_path/f'review{i}.json'
        p.write_text(json.dumps({'validation_passed':True,'accepted':False,'export_record':{
            'instance_id':str(i),'render_compatibility_sha256':'same',
            'artifacts':{'demo':{'path':str(artifact),'sha256':file_digest(artifact)}}}}))
        refs.append({'path':str(p),'sha256':file_digest(p)})
    value={'schema':'isg-training-batch-probe-request-v1','campaign_id':'campaign_20261110','canaries':refs}
    request=tmp_path/'request.json';request.write_text(json.dumps(value))
    assert len(inputs(tmp_path,request)[1])==2
    value['canaries']=[refs[0],refs[0]];request.write_text(json.dumps(value))
    with pytest.raises(ContractError,match='distinct canaries'):inputs(tmp_path,request)
    value['canaries']=refs;request.write_text(json.dumps(value))
    p=tmp_path/'review1.json';row=json.loads(p.read_text());row['export_record']['render_compatibility_sha256']='other'
    p.write_text(json.dumps(row));refs[1]['sha256']=file_digest(p);request.write_text(json.dumps(value))
    with pytest.raises(ContractError,match='configuration isolation'):inputs(tmp_path,request)


def test_observational_runtime_requires_actual_unchanged_native_code(tmp_path):
    base=tmp_path/'code'/('trunk_'+'a'*12);observed=tmp_path/'code'/('trunk_'+'b'*12)
    paths={'ispatialgym/sampling/campaign_runtime.py':('def physical(): return 1\ndef verify_loaded(): return 1\n',
             'def physical(): return 1\ndef verify_loaded(): return 2\ndef restoration_observation(): return {}\n'),
           'ispatialgym/multienv/campaign_rendering.py':('def select(): return 1\n','def select(): return 2\n'),
           'scripts/campaign_20261110/training_batch_probe.py':(None,'def probe(): return 1\n'),
           'ispatialgym/oracle/execute.py':('def physics(): return 3\n','def physics(): return 3\n')}
    for folder,index,commit in ((base,0,'a'*40),(observed,1,'b'*40)):
        folder.mkdir(parents=True);(folder/'.snapshot_commit').write_text(commit+'\n')
        for relative,values in paths.items():
            if values[index] is not None:
                p=folder/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(values[index])
    changed=list(paths)[:3];proof={'schema':'isg-observational-runtime-compatibility-v1',
        'base_commit':'a'*40,'observational_commit':'b'*40,'changed_files':changed,
        'changed_payloads':{p:file_digest(observed/p) for p in changed}}
    p=tmp_path/'proof.json';p.write_text(json.dumps(proof))
    request={'runtime_compatibility':{'path':str(p),'sha256':file_digest(p)}}
    records=[{'validation_key':{'code_sha256':file_digest(base/'.snapshot_commit')}}]
    assert verify_runtime_compatibility(tmp_path,request,records,observed)==request['runtime_compatibility']
    (observed/'ispatialgym/oracle/execute.py').write_text('def physics(): return 4\n')
    with pytest.raises(ContractError,match='outside observation-only'):
        verify_runtime_compatibility(tmp_path,request,records,observed)
    (observed/'ispatialgym/oracle/execute.py').write_text(paths['ispatialgym/oracle/execute.py'][0])
    records[0]['validation_key']['code_sha256']='wrong'
    with pytest.raises(ContractError,match='canaries native runtime'):
        verify_runtime_compatibility(tmp_path,request,records,observed)
    with pytest.raises(ContractError,match='missing exact'):
        verify_runtime_compatibility(tmp_path,{},records,observed)


def test_shared_world_requires_both_native_signature_and_saved_scene_checks():
    from types import SimpleNamespace
    records=[{'artifacts':{'tro':{'path':path}}} for path in ('/copyA/instances/a','/copyB/instances/b')]
    calls=[]
    def scene_check(*args):
        calls.append(args)
        return True,'exact native scene match'
    runner=SimpleNamespace(instance_signature=lambda p:'same',
        resolve_instance=lambda p:(None,str(__import__('pathlib').Path(p).parent.parent)),
        template_compatible=scene_check)
    proof=shared_world_proof(runner,records)
    assert proof['compatible_template_dirs']==['/copyA','/copyB']
    assert calls==[('/copyA','/copyA/instances/a','/copyB','/copyB/instances/b')]
    runner.template_compatible=lambda *args:(False,'changed object pose')
    with pytest.raises(ContractError,match='saved scenes differ'):shared_world_proof(runner,records)
    runner.instance_signature=lambda path:path
    with pytest.raises(ContractError,match='signatures differ'):shared_world_proof(runner,records)


def test_missing_velocity_is_allowed_only_for_explicit_native_kinematic_root():
    a=state();a['objects']['o']['linear_velocity']=None
    with pytest.raises(ContractError,match='dynamic motion observation missing'):
        same_loaded_state(a,deepcopy(a))
    a['objects']['o'].update(kinematic_only=True,root_prim_type='RigidKinematicPrim',angular_velocity=None)
    same_loaded_state(a,deepcopy(a))
    b=deepcopy(a);b['objects']['o']['kinematic_only']=False
    with pytest.raises(ContractError,match='identity changed'):same_loaded_state(a,b)
