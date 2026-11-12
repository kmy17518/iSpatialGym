import json
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110.coordinator_dispatch import pending_status, submit


def ticket(tmp_path):
    return {'id':'test','command':['sbatch','--parsable','job.sh'],'inputs':{},
            'claim_path':str(tmp_path/'launch.claim'),'claim_kind':'directory',
            'submission_path':str(tmp_path/'submission.json')}


def test_no_duplicate_submission(tmp_path):
    row=ticket(tmp_path);calls=[]
    def run(*a,**kw):
        calls.append(a);return SimpleNamespace(stdout='123\n',stderr='',returncode=0)
    cap={'observed_at':1,'accounts_checked':['minyeong','shiyuc']}
    assert submit(row,cap,run)['job_id']=='123'
    assert submit(row,cap,run) is None and len(calls)==1


def test_ambiguous_submission_never_retries(tmp_path):
    row=ticket(tmp_path)
    def interrupted(*a,**kw):raise OSError('connection interrupted')
    with pytest.raises(OSError):submit(row,{'observed_at':1,'accounts_checked':[]},interrupted)
    assert pending_status(row)=='submission_requires_reconciliation'
    assert submit(row,{},interrupted) is None


def test_existing_owner_claim_is_preserved(tmp_path):
    row=ticket(tmp_path);(tmp_path/'launch.claim').mkdir()
    assert pending_status(row)=='claim_requires_reconciliation'
    assert submit(row,{}) is None


def test_changed_input_fails_before_claim(tmp_path):
    row=ticket(tmp_path);p=tmp_path/'source';p.write_text('changed');row['inputs']={str(p):'0'*64}
    with pytest.raises(ValueError):submit(row,{})
    assert not (tmp_path/'launch.claim').exists()


def test_preparation_has_lower_scheduler_priority_without_payload_change(tmp_path):
    row=ticket(tmp_path);row['workstream']='training preparation';row['command'].insert(2,'--nice=0')
    before=list(row['command']);calls=[]
    def run(command,**kwargs):
        calls.append(command);return SimpleNamespace(stdout='123\n',stderr='',returncode=0)
    receipt=submit(row,{'observed_at':1,'accounts_checked':[]},run)
    assert row['command']==before and calls[0][-1]=='job.sh'
    assert '--nice=1000' in calls[0] and receipt['prepared_command']==before


def test_critical_work_can_queue_at_full_utilization_but_pending_is_bounded(tmp_path,monkeypatch):
    from scripts.campaign_20261110 import coordinator_dispatch as module
    directory=tmp_path/'infra/coordinator_dispatch';directory.mkdir(parents=True)
    rows=[]
    for i in range(3):
        child=tmp_path/str(i);child.mkdir();row=ticket(child)
        row.update(id=str(i),workstream='pipeline',owner='/root/compute_recovery',expected_account='shiyuc')
        rows.append(row)
    manifest=tmp_path/'prepared.json';manifest.write_text(json.dumps({'tickets':rows}))
    (directory/'policy.json').write_text(json.dumps({'owner':'/root','production_released':False,
        'critical_pending_limit':1,'prepared_manifests':[{'path':str(manifest),'sha256':module.sha(manifest)}]}))
    calls=[]
    def run(command,**kwargs):
        calls.append(command);return SimpleNamespace(stdout='123\n',stderr='',returncode=0)
    original=module.submit;monkeypatch.setattr(module,'submit',lambda t,c:original(t,c,run))
    cap={'status':'observed','observed_at':1,'accounts_checked':['minyeong','shiyuc'],
         'compatible_available_slots':0,'account_slots':{'shiyuc':0},'pending_job_ids':[]}
    result=module.cycle(tmp_path,'shiyuc',lambda:cap)
    assert len(calls)==1 and [r['state'] for r in result['requests']]==['submitted','ready','ready']
    cap['pending_job_ids']=['123'];module.cycle(tmp_path,'shiyuc',lambda:cap)
    assert len(calls)==1


def test_one_pending_job_does_not_block_independent_free_slots(monkeypatch):
    from scripts.campaign_20261110 import coordinator_dispatch as module
    observation={'state':'available','observed_at':1000,'jobs':[
        ['minyeong','svl','PENDING','gres/gpu:titanrtx:1','normal','(Priority)','1']],
        'nodes':[{'NodeName':'svl8','Partitions':'svl','State':'MIXED',
            'CfgTRES':'cpu=64,mem=500000M,gres/gpu=8,gres/gpu:titanrtx=8',
            'AllocTRES':'cpu=0,mem=0,gres/gpu=0','CPUEfctv':'64','CPUAlloc':'0',
            'RealMemory':'500000','AllocMem':'0'}]}
    monkeypatch.setattr(module.dispatch_cluster_first,'probe',lambda:observation)
    result=module.probe()
    assert result['account_slots']['minyeong']==7
    assert result['account_slots']['shiyuc']==8
    assert result['compatible_available_slots']==8
