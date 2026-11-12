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


def test_new_preparation_classes_and_heldouts_are_not_starved_by_large_older_wave():
    from scripts.campaign_20261110.coordinator_dispatch import dispatch_order
    rows=[{'id':str(i),'class_id':'floating','workstream':'training preparation','ready_since':1}for i in range(100)]
    rows += [{'id':'camera','class_id':'camera','workstream':'training preparation','ready_since':2},
             {'id':'heldout','workstream':'heldout eval','ready_since':3},
             {'id':'physics','workstream':'pipeline','ready_since':4}]
    ordered=list(dispatch_order(rows))
    assert [r['id']for r in ordered[:4]]==['physics','heldout','0','camera']
    assert len(ordered)==len(rows) and len({r['id']for r in ordered})==len(rows)


def test_downstream_collectors_keep_bounded_capacity_when_preparation_is_full():
    from scripts.campaign_20261110.coordinator_dispatch import collector_allowance
    cap={'status':'observed','compatible_available_slots':0,'account_slots':{'minyeong':0},
         'pending_job_ids':['1','2','99']}
    assert collector_allowance(cap,['1','2'])==1
    assert collector_allowance(cap,['1','2','99'])==0
    assert collector_allowance({**cap,'status':'unavailable'},[])==0
    cap.update(compatible_available_slots=4,account_slots={'minyeong':4})
    assert collector_allowance(cap,['1','2','99'])==4


@pytest.fixture
def consumed_preparation(tmp_path):
    from scripts.campaign_20261110 import coordinator_dispatch as module
    runtime=tmp_path/'runtime';runtime.write_text('immutable-runtime')
    script=tmp_path/'builder.py';script.write_text('build_scaled_grasp_library()')
    manifest={'path':str(tmp_path/'normalized.json'),'sha256':'a'*64}
    source={'path':str(tmp_path/'source.json'),'sha256':'b'*64}
    row=ticket(tmp_path)
    row.update(workstream='training preparation',expected_account='minyeong',
               physical_dedup_key='physical123',dedup_key='physical123',
               claim_path=str(tmp_path/'claims/physical123.json'),
               submission_path=str(tmp_path/'receipts/physical123.json'),
               configuration_ids=['selected_scaled_object'],input_refs=[{'path':str(runtime),'sha256':module.sha(runtime)}],
               runtime_ref={'path':str(runtime),'sha256':module.sha(runtime)},
               inputs={str(runtime):module.sha(runtime)},command=['sbatch','--parsable',str(script)],
               _prepared_manifest=manifest)
    contract={'schema':'isg-consumed-preparation-admission-v1',
              'allowed_computation_kinds':{'grasp_library':['attachment_free']},
              'forbidden_preparation_entrypoint':'ispatialgym.sampling.sample_isg',
              'approved_manifests':[{'manifest':source,'account':'minyeong','computation_kind':'grasp_library',
                  'consumer_kind':'attachment_free','required_for_readiness':True,
                  'claim_namespace':str(tmp_path/'claims'),'receipt_namespace':str(tmp_path/'receipts')}]}
    path=tmp_path/'scope.json';path.write_text(json.dumps(contract))
    policy={'preparation_scope':{'policy_ref':{'path':str(path),'sha256':module.sha(path)},
            'normalized_manifests':[{'manifest':manifest,'source_manifest':source}]}}
    return row,policy


def test_consumed_scaled_library_uses_reviewed_manifest_and_original_claim(consumed_preparation):
    from scripts.campaign_20261110.coordinator_dispatch import authorize_preparation
    row,policy=consumed_preparation
    assert authorize_preparation(row,policy)['consumer_kind']=='attachment_free'
    assert not __import__('pathlib').Path(row['claim_path']).exists()


@pytest.mark.parametrize('change',['manifest','physical_key','claim_namespace','runtime','consumer'])
def test_preparation_identity_changes_fail_closed(consumed_preparation,change):
    from scripts.campaign_20261110.coordinator_dispatch import authorize_preparation
    row,policy=consumed_preparation
    if change=='manifest':row['_prepared_manifest']={**row['_prepared_manifest'],'sha256':'c'*64}
    elif change=='physical_key':row['physical_dedup_key']='other'
    elif change=='claim_namespace':row['claim_path']=row['claim_path'].replace('/claims/','/new_claims/')
    elif change=='runtime':row['inputs']={}
    elif change=='consumer':row['configuration_ids']=[]
    with pytest.raises(ValueError):authorize_preparation(row,policy)


def test_preparation_rejects_scene_sampler_even_with_approved_manifest(consumed_preparation):
    from pathlib import Path
    from scripts.campaign_20261110.coordinator_dispatch import authorize_preparation
    row,policy=consumed_preparation
    Path(row['command'][-1]).write_text('python -m ispatialgym.sampling.sample_isg')
    with pytest.raises(ValueError,match='Per-scene'):authorize_preparation(row,policy)


def test_missing_scope_cannot_authorize_preparation(consumed_preparation):
    from scripts.campaign_20261110.coordinator_dispatch import authorize_preparation
    row,_=consumed_preparation
    with pytest.raises(ValueError,match='requires'):authorize_preparation(row,{})


def test_preparation_reserves_physical_capacity_without_erasing_small_account_quota():
    from scripts.campaign_20261110.coordinator_dispatch import preparation_capacity
    cap={'status':'observed','compatible_available_slots':7,'account_slots':{'minyeong':10,'shiyuc':2}}
    assert preparation_capacity(cap,'minyeong',4)==3
    assert preparation_capacity(cap,'shiyuc',4)==2
    assert preparation_capacity({**cap,'compatible_available_slots':4},'minyeong',4)==0
    assert preparation_capacity({**cap,'status':'unavailable'},'minyeong',4)==0
