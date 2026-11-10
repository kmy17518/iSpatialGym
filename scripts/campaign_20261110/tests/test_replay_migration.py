import copy
import datetime
import json
from pathlib import Path

import pytest

from scripts.campaign_20261110 import replay_migration as rm
from scripts.campaign_20261110.tests.test_vast_replay import fixture


def proof(root, state='PENDING'):
    return {'job_id':'123','scheduler':{'UserId':'minyeong(1)','JobName':'isg261110_replay',
        'Comment':'campaign_20261110_replay_456','Command':str(root/'tools/infra_open_loop.sh'),
        'JobState':state,'SubmitTime':'2026-11-10T00:00:00'},
        'accounting':{'state':state,'start':'Unknown','elapsed_seconds':0,'allocated_tres':''}}


def entry(source, output):
    return {'source_job_id':'456','replay_job_id':'123','source_job_dir':str(source),
            'output':str(output),'status':'replay_queued'}


def test_pending_requires_exact_campaign_owner_and_no_prior_gpu_time(tmp_path):
    source,native,helper,out=fixture(tmp_path);e=entry(source,out);p=proof(tmp_path)
    now=datetime.datetime(2026,11,10,0,5,tzinfo=datetime.timezone.utc).timestamp()
    assert rm.eligible(p,e,tmp_path,300,now)
    assert not rm.eligible(p,e,tmp_path,300,now-1)
    for field,value in [('start','2026-11-10T00:00:02'),('elapsed_seconds',1),('allocated_tres','gres/gpu=1')]:
        changed=copy.deepcopy(p);changed['accounting'][field]=value
        assert not rm.eligible(changed,e,tmp_path,300,now)
    p['scheduler']['Comment']='other_campaign'
    with pytest.raises(ValueError,match='identity differs'):rm.eligible(p,e,tmp_path,300,now)


def test_never_started_pending_migration_is_durable_and_idempotent(tmp_path,monkeypatch):
    source,native,helper,out=fixture(tmp_path);e=entry(source,out)
    (tmp_path/'vast').mkdir();(tmp_path/'vast/config.json').write_text('{"budget_usd":3000,"dispatch_job_ids":[]}')
    pending=proof(tmp_path);cancelled=proof(tmp_path,'CANCELLED');answers=iter([pending,pending,cancelled])
    monkeypatch.setattr(rm,'scheduler_proof',lambda _:next(answers))
    saved=[];commands=[]
    def command(argv,**kwargs):
        assert saved[-1]['vast_migration']['status']=='cancel_requested'
        assert not json.loads((tmp_path/'vast/config.json').read_text())['dispatch_job_ids']
        commands.append(argv)
    monkeypatch.setattr(rm.subprocess,'run',command)
    assert rm.migrate(e,tmp_path,native,helper,300,lambda:saved.append(copy.deepcopy(e)))
    assert commands==[['scancel','--state=PENDING','123']]
    assert e['replay_backend']=='vast' and e['vast_migration']['status']=='active'
    assert out.is_symlink() and e['replay_job_id'] in json.loads((tmp_path/'vast/config.json').read_text())['dispatch_job_ids']
    assert not rm.migrate(e,tmp_path,native,helper,300,lambda:None)


def test_scheduler_start_race_leaves_vast_inactive_and_original_output_unmodified(tmp_path,monkeypatch):
    source,native,helper,out=fixture(tmp_path);e=entry(source,out)
    pending=proof(tmp_path);running=proof(tmp_path,'RUNNING');running['accounting']['start']='2026-11-10T00:01:00'
    answers=iter([pending,pending,running]);monkeypatch.setattr(rm,'scheduler_proof',lambda _:next(answers))
    monkeypatch.setattr(rm.subprocess,'run',lambda *args,**kwargs:None)
    with pytest.raises(ValueError,match='started during cancellation'):
        rm.migrate(e,tmp_path,native,helper,300,lambda:None)
    assert not out.exists() and e['replay_job_id']=='123'
    assert not (tmp_path/'vast/config.json').exists()


def test_existing_replay_evidence_blocks_any_cancellation(tmp_path,monkeypatch):
    source,native,helper,out=fixture(tmp_path);e=entry(source,out);out.mkdir(parents=True)
    (out/'replay.log').write_text('native startup began')
    monkeypatch.setattr(rm,'scheduler_proof',lambda _:proof(tmp_path))
    with pytest.raises(ValueError,match='execution evidence'):
        rm.migrate(e,tmp_path,native,helper,300,lambda:None)
    assert not (tmp_path/'vast/queue').exists()
