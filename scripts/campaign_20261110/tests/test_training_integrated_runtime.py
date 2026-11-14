import json
from pathlib import Path
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_integrated_runtime import (SCHEMA, reference, payloads,
    source_request, verify)


def fixture(tmp_path):
    def put(path,value):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(value if isinstance(value,str) else json.dumps(value));return reference(path)
    old=tmp_path/'code/old';new=tmp_path/'code/common'
    oldstamp=put('code/old/.snapshot_commit','original');newstamp=put('code/common/.snapshot_commit','candidate')
    put('code/old/ispatialgym/native.py','original');put('code/common/ispatialgym/native.py','changed')
    original_proof=put('original_proof.json',{'original':True})
    base={'selected_pool_world':True,'pool_action_replay_v1':True,'load_only':False,
          'production_released':False,'benchmark_credit':0,'diagnostic_control':'ABA',
          'canaries':[{'original':'A'},{'original':'B'}],'probe_runtime_sha256':oldstamp['sha256'],
          'runtime_compatibility':original_proof,'source_environment_delta':{'prebuild':'1'}}
    base_ref=put('base.json',base);before=payloads(old);after=payloads(new)
    proof={'schema':SCHEMA,'production_released':False,'benchmark_credit':0,
           'physical_equivalence_established':False,'source_inputs_changed':False,
           'permitted_execution_kinds':['physics_action_open_loop'],'base_request':base_ref,
           'base_runtime':oldstamp,'candidate_runtime':newstamp,'candidate_python_payloads':after,
           'changed_payloads':{n:{'base':before.get(n),'candidate':after.get(n)}for n in set(before)|set(after)
                               if before.get(n)!=after.get(n)}}
    ref=put('proof.json',proof);q={**base,'integrated_candidate_v1':True,
        'probe_runtime_sha256':newstamp['sha256'],'runtime_compatibility':ref}
    return q,base,proof,new,put


def test_changed_candidate_preserves_exact_request_and_original_validation(tmp_path):
    q,base,proof,new,_=fixture(tmp_path);calls=[];records=[{'immutable':'source'}]
    def validate(root,value,rows,runtime):
        assert value==base and rows is records and runtime==tmp_path/'code/old'
        calls.append(value);return base['runtime_compatibility']
    assert verify(tmp_path,q,records,new,validate)==q['runtime_compatibility']
    assert calls==[base] and source_request(tmp_path,q)[0]==base


@pytest.mark.parametrize('field,value',[('canaries',[{'different':'source'}]),
    ('diagnostic_control','FRESH_A'),('source_environment_delta',{}),('production_released',True),
    ('benchmark_credit',1),('benchmark_credit',False),('extra_budget',10000),('physical_candidate',{})])
def test_integration_cannot_rewrite_inputs_control_scope_or_budget(tmp_path,field,value):
    q,*_=fixture(tmp_path);q[field]=value
    with pytest.raises(ContractError):source_request(tmp_path,q)


@pytest.mark.parametrize('field,value',[('physical_equivalence_established',True),
    ('source_inputs_changed',True),('production_released',True),('permitted_execution_kinds',['render'])])
def test_integration_proof_never_grants_equivalence_or_expands_execution_kind(tmp_path,field,value):
    q,base,proof,new,put=fixture(tmp_path);proof[field]=value;q['runtime_compatibility']=put('proof.json',proof)
    with pytest.raises(ContractError):source_request(tmp_path,q)


@pytest.mark.parametrize('mutation',['candidate','unlisted_file','undeclared_difference','base_runtime','nested'])
def test_runtime_inventory_and_single_original_proof_are_exact(tmp_path,mutation):
    q,base,proof,new,put=fixture(tmp_path)
    if mutation=='candidate':put('code/common/ispatialgym/native.py','unexpected')
    elif mutation=='unlisted_file':put('code/common/scripts/new.py','unlisted')
    elif mutation=='undeclared_difference':proof['changed_payloads']={}
    elif mutation=='base_runtime':put('code/old/.snapshot_commit','different')
    else:
        base['integrated_candidate_v1']=True;proof['base_request']=put('base.json',base)
    q['runtime_compatibility']=put('proof.json',proof)
    with pytest.raises(ContractError):verify(tmp_path,q,[],new,lambda *args:base['runtime_compatibility'])


def test_original_physical_verifier_failure_is_not_bypassed(tmp_path):
    q,base,proof,new,_=fixture(tmp_path)
    def reject(*args):raise ContractError('old native source no longer authorized')
    with pytest.raises(ContractError,match='no longer authorized'):verify(tmp_path,q,[],new,reject)


def renderer_fixture(tmp_path):
    q,old,proof,new,put=fixture(tmp_path)
    base={'schema':'isg-pool-native-render-reuse-request-v1','candidate_runtime':proof['base_runtime'],
          'changed_payloads':{'old_isolated_payload':'hash'},'sequence':[0,1,0],
          'production_released':False,'benchmark_credit':0,'cold_requests':['A','B'],
          'physical_delta':{'added_physics_steps':0},'member_campaign_configs':['exactA','exactB']}
    proof.update(base_request=put('renderer_base.json',base),permitted_execution_kinds=['kinematic_observation_rendering'])
    request={k:v for k,v in base.items()if k!='changed_payloads'}
    request.update(candidate_runtime=proof['candidate_runtime'],integrated_renderer_candidate_v1=True,
                   integration_proof=put('renderer_proof.json',proof))
    return request,base,proof,new,put


def test_renderer_preserves_original_checked_frames_and_cold_sources(tmp_path):
    from scripts.campaign_20261110.training_integrated_runtime import verify_renderer
    q,base,proof,new,_=renderer_fixture(tmp_path);calls=[]
    def validate(root,path,*,runtime):
        assert Path(path)==Path(proof['base_request']['path']) and runtime==tmp_path/'code/old'
        calls.append(path);return (base,)
    assert verify_renderer(tmp_path,q,new,validate)==proof and len(calls)==1


@pytest.mark.parametrize('field,value',[('sequence',[0,0,0]),('cold_requests',['different','B']),
    ('member_campaign_configs',['different','exactB']),('physical_delta',{'added_physics_steps':1}),
    ('renderer_cleanup','failure'),('changed_payloads',{}),('production_released',True)])
def test_renderer_integration_cannot_change_selection_physics_or_cleanup(tmp_path,field,value):
    from scripts.campaign_20261110.training_integrated_runtime import renderer_source_request
    q,base,proof,new,put=renderer_fixture(tmp_path);q[field]=value
    with pytest.raises(ContractError):renderer_source_request(tmp_path,q)


def test_renderer_rechecks_original_payload_proof_and_new_payload_inventory(tmp_path):
    from scripts.campaign_20261110.training_integrated_runtime import verify_renderer
    q,base,proof,new,put=renderer_fixture(tmp_path)
    def fail(*args,**kwargs):raise ContractError('old recorded source changed')
    with pytest.raises(ContractError,match='old recorded'):verify_renderer(tmp_path,q,new,fail)
    put('code/common/ispatialgym/native.py','changed again')
    with pytest.raises(ContractError,match='payload changed'):
        verify_renderer(tmp_path,q,new,lambda *args,**kwargs:(base,))


def current_recording_fixture(tmp_path):
    base,_,oldproof,new,put=renderer_fixture(tmp_path)
    physics_manifest=put('physical_manifest.json',{'records':[]})
    physics=put('physical_verified.json',{'status':'matched','active_recording_scope_verified':True,
        'manifest':physics_manifest})
    cold_driver=put('cold_driver.py','bound driver')
    cold_requests=[put('cold_'+control+'.json',{'control':control,'recording_manifest':physics_manifest,
        'active_recording_verification':physics,'driver':cold_driver,'native_runtime':base['candidate_runtime']})
        for control in ('FRESH_A','FRESH_B')]
    cold_verified=put('cold_verified.json',{'status':'native_cold_render_verified','controls':{
        name:{'all_original_cold_frames_matched':True,'export_verification':{'exact':'export'}}
        for name in ('FRESH_A','FRESH_B')}})
    proof={'schema':'isg-current-recording-renderer-proof-v1','production_released':False,'benchmark_credit':0,
        'permitted_execution_kinds':['kinematic_observation_rendering'],
        'base_request':put('common_renderer_base.json',base),'base_runtime':base['candidate_runtime'],
        'candidate_runtime':base['candidate_runtime'],'source_recording_manifest':physics_manifest,
        'source_recording_verification':physics,'candidate_python_payloads':payloads(new),'changed_payloads':{}}
    request={k:v for k,v in base.items()if k not in ('integrated_renderer_candidate_v1','integration_proof')}
    request.update(current_recording_renderer_v1=True,current_recording_proof=put('current_proof.json',proof),
        parent_runtime=base['candidate_runtime'],cold_driver=cold_driver,cold_requests=cold_requests,
        cold_verification=cold_verified,scene_union_sha256='new verified actual scenes')
    request['runtime_compatibility']=request['current_recording_proof']
    return request,base,proof,new,put


def test_actual_new_recordings_require_their_own_native_chain(tmp_path):
    from scripts.campaign_20261110.training_integrated_runtime import verify_current_recording_renderer
    q,base,proof,new,_=current_recording_fixture(tmp_path);calls=[]
    def validate(root,path,*,runtime):
        assert runtime==new;calls.append(path);return (base,)
    assert verify_current_recording_renderer(tmp_path,q,new,validate)==proof
    assert calls==[proof['base_request']['path']]


@pytest.mark.parametrize('field,value',[('sequence',[0,0,0]),('member_campaign_configs',['new','B']),
    ('physical_delta',{'added_physics_steps':1}),('renderer_cleanup','failure'),('benchmark_credit',1)])
def test_new_recordings_do_not_change_physical_sequence_or_scope(tmp_path,field,value):
    from scripts.campaign_20261110.training_integrated_runtime import current_recording_renderer_source
    q,*_=current_recording_fixture(tmp_path);q[field]=value
    with pytest.raises(ContractError):current_recording_renderer_source(tmp_path,q)


def test_new_recordings_fail_without_all_original_cold_frames(tmp_path):
    from scripts.campaign_20261110.training_integrated_runtime import current_recording_renderer_source
    q,base,proof,new,put=current_recording_fixture(tmp_path)
    q['cold_verification']=put('bad_cold.json',{'status':'native_cold_render_verified','controls':{
        name:{'all_original_cold_frames_matched':name=='FRESH_A','export_verification':{'exact':'export'}}
        for name in ('FRESH_A','FRESH_B')}})
    with pytest.raises(ContractError):current_recording_renderer_source(tmp_path,q)
