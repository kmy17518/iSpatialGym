import csv
import json
from pathlib import Path
import time
import pytest
from scripts.campaign_20261110 import runtime_suite_acceptance as a
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.tests.test_runtime_suite import fixture, put


def bound(root):
    base,env,cfg=fixture(root); row=a.reference(base/'rows/class0.json')
    job=root/'out/job';job.mkdir(parents=True)
    (job/'job.env').write_text('\n'.join(k+'='+v for k,v in {**env,'START':'1000','N':'3'}.items()))
    allocation={'jobid':'123','job_dir':str(job),'code':env['ISGE_CODE'],'start':'1000','n':'3'}
    ledger=root/'manifests/job.tsv';ledger.parent.mkdir()
    with ledger.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(allocation),delimiter='\t');writer.writeheader();writer.writerow(allocation)
    receipt=put(root/'dispatch/receipts/job.json',{'status':'submitted','allocation':{**allocation,'ledger':str(ledger)}})
    now=time.time()
    raw=put(base/'raw.json',{'schema':'isg-training-ticket-slurm-observation-v1','campaign_id':'campaign_20261110',
        'job_id':'123','observed_at':now,'squeue_returncode':0,'sacct_returncode':0,'squeue_stdout':'',
        'sacct_stdout':'123|COMPLETED|2026-11-12T04:30:00|0:0\n'})
    terminal=put(base/'terminal.json',{'schema':'isg-training-ticket-terminal-observations-v1','campaign_id':'campaign_20261110',
        'observed_at':now,'jobs':[{'job_id':'123','backend':'cluster','stage':'collection','raw':raw}]})
    ar=put(base/'allocations/class0.json',{'row':row,'allocation':allocation,'dispatch_receipt':receipt,'terminal_evidence':terminal})
    gate=put(root/'validation/gate.json',{'source_job':str(job),'snapshot':env['ISGE_CODE'],'success':True,'replay_rc':0,'instance':str(job/'inst/member__1000-tro_state.json')})
    raw_replay=put(base/'raw_replay.json',{'schema':'isg-training-ticket-slurm-observation-v1','campaign_id':'campaign_20261110',
        'job_id':'456','observed_at':now,'squeue_returncode':0,'sacct_returncode':0,'squeue_stdout':'',
        'sacct_stdout':'456|COMPLETED|2026-11-12T04:30:00|0:0\n'})
    terminal_value=json.loads(Path(terminal['path']).read_text());terminal_value['jobs'].append({'job_id':'456','backend':'cluster','stage':'replay','raw':raw_replay})
    wrapper=json.loads(Path(ar['path']).read_text());wrapper['terminal_evidence']=put(Path(terminal['path']),terminal_value)
    owner=put(base/'replay_owner.json',{'replay_job_id':'456','output':str(root/'validation')})
    wrapper['records']=[{'job_id':'456','backend':'cluster','stage':'physics_replay','job_dir':str(root/'validation'),'submission_ref':owner}]
    ar=put(Path(ar['path']),wrapper)
    return base,row,ar,gate


def test_scope_requires_new_own_gate_and_actual_terminal_binding(tmp_path):
    base,row,allocation,gate=bound(tmp_path)
    assert a.scope(tmp_path,row,allocation,gate)[0]['fresh_start']==1000
    value=json.loads(Path(gate['path']).read_text());value['source_job']=str(tmp_path/'out/old')
    wrong=put(Path(gate['path']),value)
    with pytest.raises(ContractError,match='own newly executed'):a.scope(tmp_path,row,allocation,wrong)


def test_scope_rejects_active_scheduler_despite_status_label(tmp_path):
    base,row,allocation,gate=bound(tmp_path)
    raw=json.loads((base/'raw.json').read_text());raw['squeue_stdout']='123|RUNNING'
    rawref=put(base/'raw.json',raw)
    terminal=json.loads((base/'terminal.json').read_text());terminal['jobs'][0]['raw']=rawref
    terminalref=put(base/'terminal.json',terminal)
    wrapper=json.loads(Path(allocation['path']).read_text());wrapper['terminal_evidence']=terminalref
    allocation=put(Path(allocation['path']),wrapper)
    with pytest.raises(ContractError,match='active'):a.scope(tmp_path,row,allocation,gate)


def test_optional_other_replay_does_not_block_selected_terminal_member(tmp_path):
    base,row,allocation,gate=bound(tmp_path)
    wrapper=json.loads(Path(allocation['path']).read_text());terminal=json.loads(Path(wrapper['terminal_evidence']['path']).read_text())
    pending=put(base/'pending.json',{'schema':'isg-training-ticket-slurm-observation-v1','campaign_id':'campaign_20261110',
        'job_id':'789','observed_at':time.time(),'squeue_returncode':0,'sacct_returncode':0,
        'squeue_stdout':'789|RUNNING','sacct_stdout':'789|RUNNING|2026-11-12T04:30:00|0:0\n'})
    terminal['jobs'].append({'job_id':'789','backend':'cluster','stage':'replay','raw':pending})
    wrapper['terminal_evidence']=put(Path(wrapper['terminal_evidence']['path']),terminal)
    allocation=put(Path(allocation['path']),wrapper)
    assert a.scope(tmp_path,row,allocation,gate)[0]['class_id']=='class0'
    # Moving that active job into the selected member's exact owner still fails.
    owner=put(base/'active_owner.json',{'replay_job_id':'789','output':str(tmp_path/'validation')})
    wrapper['records']=[{'job_id':'789','backend':'cluster','stage':'physics_replay','job_dir':str(tmp_path/'validation'),'submission_ref':owner}]
    with pytest.raises(ContractError,match='active'):a.scope(tmp_path,row,put(Path(allocation['path']),wrapper),gate)


def test_later_optional_member_preserves_and_reaudits_committed_group(tmp_path,monkeypatch):
    base,row,allocation,gate=bound(tmp_path)
    prior={'status':'accepted','row':row,'validation_passed':True,'selected_members':['first','second']}
    ref=put(base/'acceptance/class0.json',prior);verified=[]
    monkeypatch.setattr(a,'audit',lambda *args:pytest.fail('must not create a different review for committed group'))
    def verify(root,result_ref):
        verified.append(result_ref);assert result_ref==ref;return prior
    monkeypatch.setattr(a,'verify_result',verify)
    assert a.reconcile(tmp_path,row,allocation,gate)==prior and verified==[ref]


def test_regression_full_quota_has_no_claim_and_is_reaudited(tmp_path,monkeypatch):
    from types import SimpleNamespace
    base,row_ref,allocation_ref,gate_ref=bound(tmp_path)
    row=json.loads(Path(row_ref['path']).read_text())
    members=[{'instance_id':'member__1000','tro':str(tmp_path/'out/job/inst/member__1000-tro_state.json'),'bucket':0},
             {'instance_id':'member__1001','tro':str(tmp_path/'out/job/inst/member__1001-tro_state.json'),'bucket':0}]
    review={'members':members};validation={'validation_passed':True,'quota_consumed':0,'members':members}
    template=SimpleNamespace(task='object_scaling',class_id='class0')
    monkeypatch.setattr(a,'audit',lambda *args:(row,{},template,review,validation))
    monkeypatch.setattr(a,'quota_snapshot',lambda *args:({'eligible':False,'eligible_before':10},[],{}))
    monkeypatch.setattr('scripts.campaign_20261110.group_lane.claim_group',lambda *args,**kwargs:pytest.fail('quota-full suite must not claim'))
    result=a.reconcile(tmp_path,row_ref,allocation_ref,gate_ref)
    assert result['status']=='regression_validation_passed' and result['validation_passed']
    assert result['quota']['consumed']==0 and result['quota']['accepted']=={}
    assert not (tmp_path/'dataset').exists()
    assert result['members'][1]['physics_replay_applicability']=='own_member_replay'
    result_ref=a.reference(base/'acceptance/class0.json')
    assert a.verify_result(tmp_path,result_ref)['validation_passed']
    monkeypatch.setattr(a,'quota_snapshot',lambda *args:({'eligible':True},[],{}))
    with pytest.raises(ContractError,match='became eligible'):a.verify_result(tmp_path,result_ref)


def test_successful_physics_without_eligible_claim_stays_pending(tmp_path,monkeypatch):
    from types import SimpleNamespace
    base,row_ref,allocation_ref,gate_ref=bound(tmp_path);row=json.loads(Path(row_ref['path']).read_text())
    members=[{'instance_id':'member__1000','tro':str(tmp_path/'out/job/inst/member__1000-tro_state.json'),'bucket':0},
             {'instance_id':'member__1001','tro':str(tmp_path/'out/job/inst/member__1001-tro_state.json'),'bucket':0}]
    monkeypatch.setattr(a,'audit',lambda *args:(row,{},SimpleNamespace(),{'members':members},{'members':members,'validation_passed':True}))
    monkeypatch.setattr(a,'quota_snapshot',lambda *args:({'eligible':True},[],{}))
    result=a.reconcile(tmp_path,row_ref,allocation_ref,gate_ref,claim=False)
    assert result['status']=='validated_pending_acceptance' and result['validation_passed'] is False
    assert result['physical_validation_passed'] is True


def test_two_actual_member_replays_required_with_exact_native_and_demo_hashes(tmp_path):
    from scripts.campaign_20261110.tests.test_group_lane import candidate
    f,review=candidate(tmp_path,0)
    gate=json.loads(f.gate.read_text());gate['replay_rc']=0;first=put(f.gate,gate);member=review['members'][1]
    report=json.loads(Path(member['report']).read_text());report['mode']='replay'
    put(f.gate.parent/'replay'/member['instance_id']/'meta.json',report)
    second=put(f.gate.parent/'gate_second.json',{**gate,'instance':member['tro'],'demo':member['demo'],
        'tro_sha256':member['tro_sha256'],'demo_sha256':member['demo_sha256']})
    row={'run_started_at':f.since,'frozen_runtime':{'path':gate['snapshot']}}
    proof=a.member_replays(tmp_path,row,{'member_gates':[first,second]},review)
    assert set(proof)=={m['instance_id'] for m in review['members']}
    with pytest.raises(ContractError,match='two actual'):a.member_replays(tmp_path,row,{'member_gates':[first]},review)
    with pytest.raises(ContractError,match='duplicated'):a.member_replays(tmp_path,row,{'member_gates':[first,first]},review)
    bad=json.loads(Path(second['path']).read_text());bad['demo_sha256']='0'*64
    second=put(Path(second['path']),bad)
    with pytest.raises(ContractError,match='identity differs'):a.member_replays(tmp_path,row,{'member_gates':[first,second]},review)


def test_full_strict_bucket_is_quota_ineligible_while_other_buckets_remain(tmp_path,monkeypatch):
    from types import SimpleNamespace
    rows=[{'instance_id':'old'+str(i),'task':'object_scaling','class_id':'apple','split':'eval_standard','bucket':0} for i in range(3)]
    monkeypatch.setattr('scripts.campaign_20261110.export_release.snapshot_allocations',lambda root:(rows,{},[]))
    result,_,_=a.quota_snapshot(tmp_path,SimpleNamespace(task='object_scaling',class_id='apple'),[{'bucket':0}])
    assert result['eligible_before']==3 and result['eligible'] is False
    assert result['bucket_limits']=={'0':3,'1':4,'2':3}
    result,_,_=a.quota_snapshot(tmp_path,SimpleNamespace(task='object_scaling',class_id='apple'),[{'bucket':1}])
    assert result['eligible'] is True


@pytest.mark.parametrize('starts,passes', [((1,1,2),True), ((1,1,1),False)])
def test_audited_duplicate_replay_is_checked_but_never_counted(tmp_path,starts,passes):
    from scripts.campaign_20261110.tests.test_acceptance import fixture, audit
    f=fixture(tmp_path,starts=starts);review=audit(f)
    original=json.loads(f.gate.read_text());refs=[]
    for member in review['members']+review.get('excluded_duplicate_members',[]):
        report=json.loads(Path(member['report']).read_text());report['mode']='replay'
        put(f.gate.parent/'replay'/member['instance_id']/'meta.json',report)
        refs.append(put(f.gate.parent/(member['instance_id']+'.json'),{**original,
            'replay_rc':0,'instance':member['tro'],'demo':member['demo'],
            'tro_sha256':member['tro_sha256'],'demo_sha256':member['demo_sha256']}))
    row={'run_started_at':f.since,'frozen_runtime':{'path':original['snapshot']}}
    if not passes:
        with pytest.raises(ContractError,match='fewer than two distinct'):
            a.member_replays(tmp_path,row,{'member_gates':refs},review)
        return
    proof=a.member_replays(tmp_path,row,{'member_gates':refs},review)
    assert set(proof)=={'fresh_0','fresh_2'}
    # Even excluded work cannot smuggle unrelated/modified replay evidence in.
    duplicate=refs[-1];bad=json.loads(Path(duplicate['path']).read_text())
    bad['demo_sha256']='0'*64;refs[-1]=put(Path(duplicate['path']),bad)
    with pytest.raises(ContractError,match='identity differs'):
        a.member_replays(tmp_path,row,{'member_gates':refs},review)
