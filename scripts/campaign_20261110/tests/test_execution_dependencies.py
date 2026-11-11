from datetime import datetime,timezone
import pytest
from scripts.campaign_20261110.contracts import file_digest,write_json_atomic,ContractError
from scripts.campaign_20261110.execution_dependencies import verify_execution_dependencies


def ledger(root,dependency,**row):
    value={'schema':'isg-campaign-execution-ledger-v1','campaign_id':'campaign_20261110',
        'observed_at':datetime.now(timezone.utc).isoformat(),'rows':[{'id':'a','kind':'pipeline_canary',
        'configuration_sha256':'x','start':1,'count':1,'dependencies':[dependency],**row}]}
    write_json_atomic(root/'infra/execution_ledger.json',value)


def test_stage_exact_originals_required_but_ledger_never_authorizes(tmp_path):
    p=tmp_path/'native.json';p.write_text('{}')
    dep={'id':'native','required_for':['collection'],'state':'verified','artifact':{'path':str(p),'sha256':file_digest(p)}}
    ledger(tmp_path,dep)
    result=verify_execution_dependencies(tmp_path,'x','collection',start=1,count=1)
    assert len(result['dependencies'])==1 and not result['authorization_granted']
    p.write_text('{"changed":true}')
    with pytest.raises(ContractError,match='original changed'):verify_execution_dependencies(tmp_path,'x','collection',start=1,count=1)
    assert not verify_execution_dependencies(tmp_path,'x','native_generation',start=1,count=1)['dependencies']


def test_distinct_attempt_and_stage_failures_are_not_cross_contaminated(tmp_path):
    ledger(tmp_path,{'id':'native','required_for':['collection'],'state':'failed'})
    with pytest.raises(ContractError,match='not verified'):verify_execution_dependencies(tmp_path,'x','collection',start=1,count=1)
    assert verify_execution_dependencies(tmp_path,'x','collection',start=2,count=1)['status']=='not_indexed'
    assert verify_execution_dependencies(tmp_path,'x','native_generation',start=1,count=1)['status']=='exact_originals_checked'


def test_missing_index_does_not_grant_release(tmp_path):
    result=verify_execution_dependencies(tmp_path,'x','native_generation',start=1,count=1)
    assert result['original_gate_required'] and not result['authorization_granted']


def test_materialized_config_resolves_hashed_compact_indexes(tmp_path):
    physical=tmp_path/'physical.json';physical.write_text('{}')
    configs=tmp_path/'config_index.json';deps=tmp_path/'dep_index.json'
    write_json_atomic(configs,{'configurations':[{'id':'configuration','configuration_sha256':'x','dependencies':['d']}]})
    write_json_atomic(deps,{'artifact_requests':[{'id':'d','status':'verified','required_for':['native_generation'],
        'artifact':{'path':str(physical),'sha256':file_digest(physical)}}]})
    ledger(tmp_path,{},configuration_sha256='unrelated')
    path=tmp_path/'infra/execution_ledger.json'
    import json
    value=json.loads(path.read_text());value['indexed_sources']={k:{'path':str(p),'sha256':file_digest(p)}for k,p in [('configurations',configs),('dependencies',deps)]};write_json_atomic(path,value)
    result=verify_execution_dependencies(tmp_path,'x','native_generation')
    assert result['rows']==['configuration'] and len(result['dependencies'])==1
    deps.write_text('{}')
    with pytest.raises(ValueError,match='hash changed'):verify_execution_dependencies(tmp_path,'x','native_generation')
