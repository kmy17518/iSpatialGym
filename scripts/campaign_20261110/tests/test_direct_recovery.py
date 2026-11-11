import json
from pathlib import Path

import pytest
from scripts.campaign_20261110 import controller_direct_recovery as d


def fixture(tmp_path):
    root=tmp_path/'campaign_20261110';group=root/'groups/standard_recovery_test';group.mkdir(parents=True)
    snapshot=root/'code/current';snapshot.mkdir(parents=True);(snapshot/'.snapshot_commit').write_text('revision\n')
    key={'code_sha256':d.file_digest(snapshot/'.snapshot_commit')};spec={'token':'standard_recovery_test','class_id':'test_class','validation_key':key}
    config=group/'campaign_config.json';config.write_text(json.dumps({'validation_key':key,'split':'eval_standard','template':{'class_id':'test_class'}}))
    plan=group/'goal_plan.json';plan.write_text(json.dumps({'split':'eval_standard','group_id':group.name,'source_instance_is_member':False}))
    job=root/'out/test';job.mkdir(parents=True)
    allocation={'key':'test_class','n':'3','code':str(snapshot),'job_dir':str(job),'jobid':'123'}
    receipt={'allocation':allocation,'routed_command':['ISPATIALGYM_CAMPAIGN_CONFIG='+str(config),'ISPATIALGYM_CAMPAIGN_GOAL_PLAN='+str(plan)]}
    directory=root/'dispatch/receipts';directory.mkdir(parents=True);(directory/(spec['token']+'.json')).write_text(json.dumps(receipt))
    return root,spec,job,receipt


def test_running_direct_group_never_claims_even_if_demo_exists(tmp_path,monkeypatch):
    root,spec,job,_=fixture(tmp_path);monkeypatch.setattr(d.held,'job_terminal',lambda *a:False)
    monkeypatch.setattr(d,'audit_job',lambda *a,**kw:pytest.fail('running group audited'))
    record={};d.refresh(root,spec,record,{'started_at':0},{})
    assert record['status']=='collecting'


def test_terminal_empty_group_is_failure_without_acceptance(tmp_path,monkeypatch):
    root,spec,job,_=fixture(tmp_path);monkeypatch.setattr(d.held,'job_terminal',lambda *a:True)
    monkeypatch.setattr(d,'claim_group',lambda *a,**kw:pytest.fail('empty group claimed'))
    record={};d.refresh(root,spec,record,{'started_at':0},{})
    assert record['status']=='failed' and record['native_written']==0 and record['demo_successes']==0


def test_another_class_or_runtime_cannot_enter_scope(tmp_path):
    root,spec,job,receipt=fixture(tmp_path);receipt['allocation']['key']='other'
    (root/'dispatch/receipts'/(spec['token']+'.json')).write_text(json.dumps(receipt))
    with pytest.raises(d.held.ContractError if hasattr(d.held,'ContractError') else ValueError,match='allocation scope differs'):
        d.refresh(root,spec,{}, {'started_at':0},{})


def test_successful_demo_waits_for_own_replay(tmp_path,monkeypatch):
    root,spec,job,_=fixture(tmp_path);report=job/'demo/member';report.mkdir(parents=True);(report/'meta.json').write_text('{"success":true}')
    monkeypatch.setattr(d.held,'job_terminal',lambda *a:True)
    monkeypatch.setattr(d,'claim_group',lambda *a,**kw:pytest.fail('ungated group claimed'))
    record={};d.refresh(root,spec,record,{'started_at':0},{})
    assert record['status']=='awaiting_replay' and record['demo_successes']==1
    gate=root/'validation/open_loop/test_class/123/gate.json';gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps({'source_job':str(root/'out/another_job'),'success':True}))
    with pytest.raises(ValueError,match='own fresh replay gate'):
        d.refresh(root,spec,record,{'started_at':0},{})
