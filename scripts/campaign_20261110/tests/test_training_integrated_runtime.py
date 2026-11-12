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
