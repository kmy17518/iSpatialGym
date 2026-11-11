import csv
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110 import controller_admission as a
from scripts.campaign_20261110 import controller_queue_migration as migration
from scripts.campaign_20261110.controller_source_allocations import SourceAllocations
from scripts.campaign_20261110.contracts import ContractError, file_digest
from scripts.campaign_20261110.tests.test_controller_saved_allocations import source, write


def test_first_start_bootstrap_pending_before_legacy_scan(tmp_path, monkeypatch):
    record, _, row = source(tmp_path)
    config=write(tmp_path/'groups/g/campaign_config.json', {'template':{}, 'perturbations':{}})
    record.update(prepared={'config':str(config)})
    bootstrap=write(tmp_path/'queues/one/submission.json',record)
    cfg={'root':str(tmp_path),'existing_bootstraps':[str(bootstrap)],'snapshot':'native'}
    state={'records':{'legacy':{'status':'external_submission'}}};events=[]
    monkeypatch.setattr(a.held,'template_from',lambda value:SimpleNamespace())
    monkeypatch.setattr(a,'scoped_template',lambda *args:None)
    monkeypatch.setattr(a,'model_class_pin',lambda *args,**kw:None)
    monkeypatch.setattr(a,'admission_id',lambda *args:'new')
    monkeypatch.setattr(a,'slurm_status',lambda:{'123':['123','PENDING','owner']})
    monkeypatch.setattr(a,'check_admission_pending',lambda rec,*args:events.append('pending') if rec.get('allocation') else None)
    monkeypatch.setattr(a,'allocations',lambda root:events.append('scan')or[row])
    _,_,blocked=a.observe_admission_allocations(tmp_path,cfg,state,lambda:None)
    assert not blocked and events==['pending','scan']
    assert state['records']['new']['allocation']==row
    assert state['early_pending_observation']['before_allocation_and_source_scan']


def test_exact_receipt_does_not_scan_even_while_router_has_not_submitted(tmp_path,monkeypatch):
    record,receipt,row=source(tmp_path)
    monkeypatch.setattr(a,'allocations',lambda root:pytest.fail('unnecessary historical scan'))
    monkeypatch.setattr(a,'slurm_status',lambda:{})
    cfg={};state={'records':{'one':record}}
    receipt.unlink()
    assert a.observe_admission_allocations(tmp_path,cfg,state,lambda:None)[0]==[]
    assert 'allocation' not in record


def test_bad_receipt_blocks_refresh_scope_without_discovery_bypass(tmp_path,monkeypatch):
    record,receipt,_=source(tmp_path);data=json.loads(receipt.read_text());data['allocation']['n']='6';write(receipt,data)
    monkeypatch.setattr(a,'allocations',lambda root:pytest.fail('bad receipt bypass'))
    monkeypatch.setattr(a,'slurm_status',lambda:{})
    rows,_,blocked=a.observe_admission_allocations(tmp_path,{}, {'records':{'one':record}}, lambda:None)
    assert rows==[] and blocked=={'one'}


def test_pending_migration_requires_unchanged_source_and_preserves_failure(tmp_path,monkeypatch):
    record,_,row=source(tmp_path);record['allocation']=row
    config=write(tmp_path/'groups/source/config.json',{'x':1})
    gate=write(tmp_path/'validation/source/gate.json',{'success':True})
    record['source_proof']={k:{'path':str(p),'sha256':file_digest(p)} for k,p in [('config',config),('source_gate',gate)]}
    calls=[];monkeypatch.setattr(migration,'migrate_pending_queue',lambda *args,**kw:calls.append(kw))
    cfg={'migrate_pending_seconds':0,'_migration_checks_remaining':8};q={'123':['123','PENDING','owner']}
    a.check_admission_pending(record,cfg,tmp_path,q,lambda:None)
    assert calls==[{'minimum_age':0}]
    write(config,{'x':2});a.check_admission_pending(record,cfg,tmp_path,q,lambda:None)
    assert len(calls)==1 and 'source changed' in record['migration_error']['reason']
    q['123'][1]='RUNNING';a.check_admission_pending(record,cfg,tmp_path,q,lambda:None)
    assert len(calls)==1


def test_source_discovery_is_lazy_and_saved_row_is_revalidated(tmp_path):
    _,_,row=source(tmp_path);calls=[];state={}
    lookup=SourceAllocations(tmp_path,state,lambda root:calls.append('scan')or[row])
    assert calls==[]
    assert lookup.find(row['job_dir'])==row and calls==['scan']
    lookup=SourceAllocations(tmp_path,state,lambda root:pytest.fail('saved source rescanned campaign'))
    assert lookup.find(row['job_dir'])==row
    ledger=Path(row['ledger']);lines=ledger.read_text().splitlines(True);ledger.write_text(''.join(lines)+lines[1])
    with pytest.raises(ContractError,match='ambiguous'):lookup.find(row['job_dir'])


def test_exact_seed_ledger_never_needs_discovery_and_changed_row_is_rejected(tmp_path):
    _,_,row=source(tmp_path);state={}
    lookup=SourceAllocations(tmp_path,state,lambda root:pytest.fail('known seed needs no scan'))
    assert lookup.find(row['job_dir'],ledger=row['ledger'])==row
    p=Path(row['ledger']);p.write_text(p.read_text().replace('\t123\t','\t124\t'))
    with pytest.raises(ContractError,match='changed'):lookup.find(row['job_dir'])


def test_two_live_cycles_keep_immutable_config_identity(tmp_path,monkeypatch):
    import sys
    cfg={'root':str(tmp_path),'schema':'isg-group-admission-controller-v1','snapshot':str(tmp_path/'code/native')}
    config=write(tmp_path/'controllers/test/config.json',cfg);state=tmp_path/'controllers/test/state.json'
    cycles=iter([True,True,False]);monkeypatch.setattr(a.lifecycle,'continue_running',lambda cfg:next(cycles))
    monkeypatch.setattr(a.lifecycle,'record_policy',lambda *args:None)
    monkeypatch.setattr(a,'slurm_status',lambda:{})
    monkeypatch.setattr(a,'recent_gates',lambda root:[])
    monkeypatch.setattr(a.time,'sleep',lambda _:None)
    monkeypatch.setattr(a,'allocations',lambda root:pytest.fail('empty scope must not scan'))
    monkeypatch.setattr(sys,'argv',['controller','--config',str(config),'--state',str(state)])
    a.main()
    from scripts.campaign_20261110.contracts import digest
    assert json.loads(state.read_text())['config_sha256']==digest(cfg)
