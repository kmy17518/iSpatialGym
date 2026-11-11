import csv
import json
import pytest
from scripts.campaign_20261110 import controller_saved_allocations as m, controller_held as h
from scripts.campaign_20261110.contracts import ContractError,digest


def write(p, value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p


def source(root):
    command=['python','submit_fast.py','submit','class','3','--code',str(root/'code/native'),'--base-index','50000100']
    identity={'key':'class','start':50000100,'n':3,'code':str(root/'code/native'),'tag':'goal_groups','files':{}}
    row={'key':'class','start':'50000100','n':'3','code':str(root/'code/native'),'tag':'goal_groups','jobid':'123','job_dir':str(root/'out/job')}
    ledger=root/'manifests/dispatch_ledgers/one/allocations.tsv';ledger.parent.mkdir(parents=True)
    with ledger.open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(row),delimiter='\t');w.writeheader();w.writerow(row)
    allocation={**row,'ledger':str(ledger)}
    intent=write(root/'dispatch/ready/one.json',{'token':'one','command':command,'command_sha256':digest(command),'identity':identity})
    receipt=write(root/'dispatch/receipts/one.json',{'token':'one','status':'submitted','command_sha256':digest(command),'identity':identity,'allocation':allocation})
    record={'command':command,'directory':str(root/'queues/one'),'queue':str(root/'queues/one/queue.json'),'dispatch_intent':str(intent),'dispatch_receipt':str(receipt),'index':50000100,'count':3,'status':'external_submission'}
    return record,receipt,allocation


def test_exact_receipt_adopts_one_ledger_without_discovery(tmp_path):
    record,_,allocation=source(tmp_path)
    assert m.adopt_receipt(tmp_path,record)
    assert record['allocation']==allocation
    assert record['allocation_receipt_observation']['receipt']['sha256']
    assert not m.adopt_receipt(tmp_path,record)


@pytest.mark.parametrize('mutation',['command','index','count','runtime','ledger_duplicate','jobid'])
def test_mismatched_receipts_never_attach(tmp_path,mutation):
    record,p,a=source(tmp_path);receipt=json.loads(p.read_text())
    if mutation=='command':receipt['command_sha256']='changed'
    if mutation=='index':record['index']+=1
    if mutation=='count':receipt['allocation']['n']='6'
    if mutation=='runtime':receipt['allocation']['code']=str(tmp_path/'code/other')
    if mutation=='jobid':receipt['allocation']['jobid']='different'
    if mutation=='ledger_duplicate':
        from pathlib import Path
        ledger=Path(a['ledger']);lines=ledger.read_text().splitlines(True);ledger.write_text(''.join(lines)+lines[1])
    write(p,receipt)
    with pytest.raises((ContractError,ValueError)):m.adopt_receipt(tmp_path,record)
    assert 'allocation'not in record


def test_first_cycle_pending_check_precedes_legacy_scan(tmp_path,monkeypatch):
    record,_,_=source(tmp_path);events=[];state={'seeds':{'seed':{'queues':[record]}}};cfg={'seeds':[{'id':'seed','class_id':'class'}]}
    monkeypatch.setattr(h,'slurm_status',lambda:{'123':['123','PENDING','owner']})
    def pending(record,*args,**kwargs):
        assert record['allocation']['jobid']=='123';events.append('pending')
    monkeypatch.setattr(h,'check_pending_record',pending)
    # A second old queue lacks a dispatch receipt and still requires discovery.
    state['seeds']['seed']['queues'].append({'queue':'legacy','status':'external_submission'})
    monkeypatch.setattr(h,'allocations',lambda root:events.append('full_scan')or[])
    h.observe_saved_allocations(tmp_path,state,cfg,lambda:None)
    assert events==['pending','full_scan']
    assert state['early_pending_observation']['records'][0]['scheduler_state']=='PENDING'


def test_saved_only_queue_skips_historical_scan_but_keeps_pending_checks(tmp_path,monkeypatch):
    record,_,a=source(tmp_path);state={'seeds':{'seed':{'queues':[record]}}};cfg={'seeds':[{'id':'seed','class_id':'class'}],'retired_classes':['class']}
    monkeypatch.setattr(h,'slurm_status',lambda:{'123':['123','RUNNING','owner']})
    monkeypatch.setattr(h,'check_pending_record',lambda *a,**k:None)
    monkeypatch.setattr(h,'allocations',lambda root:pytest.fail('saved queue must not scan historical ledgers'))
    rows,_,blocked=h.observe_saved_allocations(tmp_path,state,cfg,lambda:None)
    assert rows==[a]and not blocked


def test_missing_receipt_waits_without_claim_or_resubmission(tmp_path):
    record,p,_=source(tmp_path);p.unlink()
    assert not m.adopt_receipt(tmp_path,record)
    assert 'allocation'not in record
    assert m.bounded_rows({'seeds':[{'id':'s','class_id':'class'}],'retired_classes':['class']},{'seeds':{'s':{'queues':[record]}}})==[]


def test_bad_receipt_blocks_seed_before_expensive_work(tmp_path,monkeypatch):
    record,p,_=source(tmp_path);value=json.loads(p.read_text());value['identity']['key']='other';write(p,value)
    state={'seeds':{'s':{'queues':[record]}}};cfg={'seeds':[{'id':'s','class_id':'class'}],'retired_classes':['class']}
    monkeypatch.setattr(h,'slurm_status',lambda:{})
    monkeypatch.setattr(h,'check_pending_record',lambda *a,**k:pytest.fail('bad receipt must not migrate'))
    monkeypatch.setattr(h,'allocations',lambda root:pytest.fail('bad receipt must not bypass through legacy scan'))
    _,_,blocked=h.observe_saved_allocations(tmp_path,state,cfg,lambda:None)
    assert blocked=={'s'}and 'allocation'not in record


def test_unadopted_legacy_queue_requires_scan():
    cfg={'seeds':[{'id':'s','class_id':'class'}],'retired_classes':['class']}
    assert m.bounded_rows(cfg,{'seeds':{'s':{'queues':[{'status':'external_submission','queue':'old'}]}}})is None


def test_first_start_existing_queue_is_joined_before_scan(tmp_path,monkeypatch):
    from pathlib import Path
    from scripts.campaign_20261110.contracts import file_digest
    record,_,allocation=source(tmp_path);key={'code_sha256':'exact'}
    config=write(tmp_path/'groups/g/campaign_config.json',{'validation_key':key,'split':'eval_standard'})
    queue=Path(record['queue']);queue.parent.mkdir(parents=True)
    prepared={'queue':str(queue),'config':str(config),'candidate_instances':3,'group_ids':['g']}
    write(queue.parent/'submission.json',{**record,'prepared':prepared})
    cfg={'seeds':[{'id':'s','class_id':'class','existing_queues':[str(queue)],
                  'seed_config':str(config),'seed_config_sha256':file_digest(config)}]}
    state={'seeds':{}};events=[]
    monkeypatch.setattr(h,'slurm_status',lambda:{'123':['123','PENDING','owner']})
    monkeypatch.setattr(h,'check_pending_record',lambda record,*a,**k:events.append(record['allocation']['jobid']))
    monkeypatch.setattr(h,'allocations',lambda root:pytest.fail('first-start exact receipt must avoid historical scan'))
    rows,_,blocked=h.observe_saved_allocations(tmp_path,state,cfg,lambda:None)
    assert events==['123']and rows==[allocation]and not blocked


def test_early_migration_preserves_source_gate_check(tmp_path,monkeypatch):
    from scripts.campaign_20261110.contracts import file_digest
    record,_,_=source(tmp_path);config=write(tmp_path/'groups/config.json',{})
    gate=write(tmp_path/'gate.json',{'success':True})
    spec={'id':'s','class_id':'class','seed_config':str(config),'seed_config_sha256':file_digest(config),
          'source_gate':str(gate),'source_gate_sha256':file_digest(gate)}
    write(gate,{'success':False})
    cfg={'seeds':[spec],'migrate_pending_seconds':0};state={'seeds':{'s':{'queues':[record]}}}
    monkeypatch.setattr(h,'slurm_status',lambda:{'123':['123','PENDING','owner']})
    monkeypatch.setattr(h,'check_pending_record',lambda *a,**k:pytest.fail('changed source must not migrate'))
    _,_,blocked=h.observe_saved_allocations(tmp_path,state,cfg,lambda:None)
    assert blocked=={'s'}and 'source replay gate' in record['allocation_receipt_error']['reason']
