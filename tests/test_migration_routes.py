import json
from pathlib import Path
import pytest
from scripts.campaign_20261110.migration_routes import source_routes, validate_claim
from scripts.campaign_20261110.contracts import file_digest, ContractError


def test_private_original_routes_are_preserved_and_unknown_routes_fail(tmp_path):
    p=tmp_path/'dispatch/ready/source.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'backends':['minyeong_svl','vast_4090']}))
    result=source_routes(tmp_path,{'dispatch_intent':str(p)})
    assert result['backends']==['minyeong_svl','vast_4090']
    assert 'shiyuc_svl' not in result['backends']
    with pytest.raises(ContractError,match='unavailable'):source_routes(tmp_path,{})


def test_claim_rejects_route_widening_changed_source_and_started_proof(tmp_path):
    p=tmp_path/'dispatch/ready/source.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'backends':['minyeong_svl','vast_4090']}))
    proof=tmp_path/'proof.json'
    value={'never_started':True,'after':{'accounting':{'state':'CANCELLED','start':'Unknown'}},
           'final_execution_proof':{'nonempty_execution_artifacts':[]}}
    proof.write_text(json.dumps(value));routing=source_routes(tmp_path,{'dispatch_intent':str(p)})
    routing['proof']={'path':str(proof),'sha256':file_digest(proof)}
    payload={'backends':routing['backends'],'migration_routing':routing}
    validate_claim(tmp_path,payload)
    with pytest.raises(ContractError,match='widened'):
        validate_claim(tmp_path,{**payload,'backends':['shiyuc_svl']})
    value['after']['accounting']['start']='2026-11-11T10:00:00'
    proof.write_text(json.dumps(value));routing['proof']['sha256']=file_digest(proof)
    with pytest.raises(ContractError,match='never-started'):validate_claim(tmp_path,payload)
    p.write_text(p.read_text()+' ')
    with pytest.raises(ContractError,match='transport changed'):validate_claim(tmp_path,payload)


def test_legacy_nfs_journal_restriction_survives_migration(tmp_path):
    code=tmp_path/'code/legacy';runtime=code/'ispatialgym/sampling/campaign_runtime.py'
    runtime.parent.mkdir(parents=True);runtime.write_text('fcntl.flock(lock, mode)\n')
    p=tmp_path/'dispatch/ready/source.json';p.parent.mkdir(parents=True)
    p.write_text(json.dumps({'command':['submit','--code',str(code)],
        'backends':['minyeong_svl','shiyuc_svl','vast_4090']}))
    assert source_routes(tmp_path,{'dispatch_intent':str(p)})['backends']==['vast_4090']
