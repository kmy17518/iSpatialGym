from copy import deepcopy
import datetime
import time

import pytest

from scripts.campaign_20261110 import controller_migration as migration
from scripts.campaign_20261110.controller_spool import candidate_budget_used
from scripts.campaign_20261110.contracts import ContractError


def proof(state='PENDING',start='Unknown'):
    return {'job_id':'123','observed_at':time.time(),'scheduler':{
        'UserId':'owner(123)','JobState':state,'SubmitTime':datetime.datetime.fromtimestamp(time.time()-600,datetime.timezone.utc).isoformat()},
        'accounting':{'state':state,'start':start,'end':'Unknown'}}


def record(root):
    return {'allocation':{'jobid':'123','job_dir':str(root/'out/old')},'status':'waiting','count':9,'index':50000000}


def test_migration_holds_then_cancels_only_verified_never_started_job(tmp_path,monkeypatch):
    stages=iter([proof(),proof(),proof('CANCELLED')]);calls=[];saved=[]
    monkeypatch.setattr(migration,'scheduler_proof',lambda jid:next(stages))
    monkeypatch.setattr(migration.subprocess,'run',lambda argv,**kwargs:calls.append(argv))
    row=record(tmp_path)
    result=migration.migrate_record(tmp_path,row,minimum_age=300,owner='owner',save=lambda:saved.append(deepcopy(row)),execute=True)
    assert calls==[['scontrol','hold','123'],['scancel','123']]
    assert saved[0]['migration_intent'] and saved[0]['status']=='waiting'
    assert row['status']=='complete' and result['never_started'] and row['index']==50000000
    assert candidate_budget_used([row,{'count':3}])==3


def test_no_cancel_if_started_during_hold_and_artifact_guard(tmp_path,monkeypatch):
    stages=iter([proof(),proof('RUNNING','2026-11-10T12:00:00')]);calls=[]
    monkeypatch.setattr(migration,'scheduler_proof',lambda jid:next(stages))
    monkeypatch.setattr(migration.subprocess,'run',lambda argv,**kwargs:calls.append(argv))
    row=record(tmp_path)
    with pytest.raises(ContractError,match='no longer pending'):
        migration.migrate_record(tmp_path,row,minimum_age=300,owner='owner',save=lambda:None,execute=True)
    assert calls==[['scontrol','hold','123']] and row['status']=='waiting'
    job=tmp_path/'out/old';job.mkdir(parents=True);(job/'sample.log').write_text('native simulator started')
    monkeypatch.setattr(migration,'scheduler_proof',lambda jid:proof())
    with pytest.raises(ContractError,match='execution artifacts'):
        migration.migrate_record(tmp_path,record(tmp_path),minimum_age=300,owner='owner',save=lambda:None,execute=True)


def test_cancelled_intent_recovers_without_resubmission_or_reusing_indices(tmp_path,monkeypatch):
    row=record(tmp_path);row['migration_intent']={'before':proof(),'original_allocation':row['allocation']}
    monkeypatch.setattr(migration,'scheduler_proof',lambda jid:proof('CANCELLED'))
    monkeypatch.setattr(migration.subprocess,'run',lambda *args,**kwargs:pytest.fail('no repeated cancellation needed'))
    migration.migrate_record(tmp_path,row,minimum_age=300,owner='owner',save=lambda:None,execute=True)
    assert row['status']=='complete' and row['migration']['never_started'] and row['index']==50000000


def test_preview_and_wrong_owner_never_mutate_jobs(tmp_path,monkeypatch):
    monkeypatch.setattr(migration,'scheduler_proof',lambda jid:proof())
    monkeypatch.setattr(migration.subprocess,'run',lambda *args,**kwargs:pytest.fail('preview must not cancel'))
    row=record(tmp_path)
    result=migration.migrate_record(tmp_path,row,minimum_age=300,owner='owner',save=lambda:pytest.fail('preview save'),execute=False)
    assert result['original_allocation']==row['allocation'] and row['status']=='waiting'
    with pytest.raises(ContractError,match='another user'):
        migration.migrate_record(tmp_path,row,minimum_age=300,owner='other',save=lambda:None,execute=True)


def test_controller_owned_pass_bounds_checks_and_prefers_fresh_vast_refill(tmp_path,monkeypatch):
    monkeypatch.setattr(migration.getpass,'getuser',lambda:'owner')
    calls=[];saved=[]
    def perform(root,row,**kwargs):
        calls.append(row['allocation']['jobid'])
        row.update(status='complete',migration={'never_started':True})
        return row['migration']
    monkeypatch.setattr(migration,'migrate_record',perform)
    cfg={'migrate_pending_seconds':300,'_migration_checks_remaining':1};state={}
    row=record(tmp_path);queue={'123':['123','PENDING','owner','svl','gpu:1']}
    migration.controller_check(tmp_path,row,state,cfg,queue,lambda:saved.append(True),family='held')
    assert calls==['123'] and row['status']=='complete' and saved==[True]
    assert state['next_dispatch_backends'][:2]==['vast_5090','vast_4090']
    migration.controller_check(tmp_path,record(tmp_path),state,cfg,queue,lambda:None,family='held')
    assert calls==['123']
    cfg['_migration_checks_remaining']=1;queue['123'][2]='another_owner'
    migration.controller_check(tmp_path,record(tmp_path),state,cfg,queue,lambda:None,family='held')
    assert calls==['123']
