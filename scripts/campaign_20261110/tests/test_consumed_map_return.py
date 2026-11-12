import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110.infra_vast import consumed_map_return as cr


def fixture(root):
    d=root/'infra/coordinator_dispatch';d.mkdir(parents=True);(d/'capacity_allocation.lock').touch()
    key='a'*64;out=root/'prepared/map';out.mkdir(parents=True);script=out/'job.sh';script.write_text('native map command')
    library=root/'library.json';cr.write(library,{'grasps':3})
    claim=root/'claims'/(key+'.json');receipt=root/'receipts'/(key+'.json')
    ticket={'id':'map','physical_dedup_key':key,'dedup_key':key,'expected_account':'minyeong',
            'consumer_kind':'tabletop','computation_kind':'reach_map','production_released':False,
            'claim_path':str(claim),'submission_path':str(receipt),'output':str(out),'hard_time_seconds':1800,
            'configuration_ids':['chosen'],'input_refs':[cr.ref(library)],'runtime_ref':cr.ref(script),
            'command':['sbatch','--parsable',str(script)],'inputs':{str(library):cr.sha(library),str(script):cr.sha(script)},'request':cr.ref(script)}
    source=root/'source.json';cr.write(source,{'jobs':[ticket]});manifest=root/'tickets.json';cr.write(manifest,{'tickets':[ticket]})
    scope=root/'scope.json';cr.write(scope,{'approved_manifests':[{'manifest':cr.ref(source),'account':'minyeong','consumer_kind':'tabletop','claim_namespace':str(claim.parent),'receipt_namespace':str(receipt.parent)}]})
    cr.write(d/'policy.json',{'prepared_manifests':[cr.ref(manifest)],'preparation_scope':{'policy_ref':cr.ref(scope),'normalized_manifests':[{'manifest':cr.ref(manifest),'source_manifest':cr.ref(source)}]}})
    directory=root/'out/vast_map_prebuild/map';native=directory/'native_inputs.json'
    cr.write(native,{'source_consumed_map_ticket':True,'kind':'tabletop_maps','new_native_instances':0,'production_released':False,
                     'source_request':str(source),'entry_index':0,'physical_dedup_key':key,'claim_path':str(claim),'submission_path':str(receipt),
                     'files':{str(source):cr.sha(source)},'library_files':{str(library):cr.sha(library)}})
    job={'id':'9001','status':'queued','tries':0,'history':[],'job_kind':'native_dependency','job_dir':str(directory),
         'script':'job_map_prebuild.sh','native_inputs_sha256':cr.sha(native),'physical_dedup_key':key,'claim_path':str(claim),'submission_path':str(receipt)}
    queue=root/'vast/queue/9001.json';cr.write(queue,job);cr.write(claim,{'physical_dedup_key':key});cr.write(receipt,{'backend':'vast','job_id':'9001','physical_dedup_key':key})
    cap=cr.base(root)/'capacity/one.json';cr.write(cap,{'observed_at':1000,'preparation_slots':1,'critical_reserved_physical_slots':4,'account_slots':{'minyeong':1},'selected_job_ids':['9001'],'selected_accounts':{'9001':'minyeong'}})
    request={'schema':cr.SCHEMA,'campaign_id':root.name,'job_id':'9001','account':'minyeong','observed_at':1000,'expires_at':1120,
             'queue':cr.ref(queue),'native_inputs':cr.ref(native),'manifest':cr.ref(manifest),'ticket_index':0,'prior_receipt':cr.ref(receipt),'capacity':cr.ref(cap)}
    path=cr.base(root)/'requests/9001.json';cr.write(path,request)
    return request,job,path,ticket


@pytest.mark.parametrize('owner',[0,1,2])
def test_any_assignment_owner_blocks_return(tmp_path,owner):
    request,job,_,_=fixture(tmp_path);owners=[set(),set(),set()];owners[owner].add('9001')
    with pytest.raises(ValueError,match='assignment owner'):cr.validate(tmp_path,request,job,*owners,1001)
    assert job['status']=='queued'


@pytest.mark.parametrize('change',[{'tries':1},{'history':['attempt']},{'status':'assigned'},{'instance':'host'}])
def test_previously_executed_work_never_returns(tmp_path,change):
    request,job,_,_=fixture(tmp_path);job.update(change)
    with pytest.raises(ValueError,match='never-assigned'):cr.validate(tmp_path,request,job,set(),set(),set(),1001)


def test_scope_bytes_and_shared_reserve_are_required(tmp_path):
    request,job,_,ticket=fixture(tmp_path)
    assert cr.validate(tmp_path,request,job,set(),set(),set(),1001)['physical_dedup_key']==ticket['physical_dedup_key']
    with pytest.raises(ValueError,match='Aggregate'):cr.validate(tmp_path,request,job,set(),set(),set(),1001,['another'])
    Path(ticket['input_refs'][0]['path']).write_text('changed')
    with pytest.raises(ValueError,match='Pinned cluster input changed'):cr.validate(tmp_path,request,job,set(),set(),set(),1001)


def test_ack_crash_preserves_exclusive_owner_and_recovers_once(tmp_path,monkeypatch):
    _,job,path,_=fixture(tmp_path);saved=[];original=cr.write
    def fail_ack(p,v):
        if Path(p).parent.name=='acknowledgements':
            assert saved[-1]['status']=='cluster_reserved';raise OSError('interrupted ack')
        original(p,v)
    monkeypatch.setattr(cr,'write',fail_ack)
    report=cr.reconcile(tmp_path,{'9001':job},set(),{},set(),lambda j:saved.append(json.loads(json.dumps(j))),now=1001)
    assert report['deferred'] and not report['rejected'] and len(saved)==1
    monkeypatch.setattr(cr,'write',original)
    report=cr.reconcile(tmp_path,{'9001':job},set(),{},set(),lambda j:saved.append(j),now=1300)
    assert report['acknowledged']==['9001'] and len(saved)==1
    assert not cr.has_pending(tmp_path)


def test_submit_needs_coordinator_ack_and_never_retries_intent(tmp_path,monkeypatch):
    _,job,path,ticket=fixture(tmp_path);calls=[]
    def run(*a,**k):calls.append(a);return SimpleNamespace(returncode=0,stdout='1234\n',stderr='')
    cap=lambda:{'status':'observed','observed_at':1001,'account_slots':{'minyeong':1},'compatible_available_slots':5}
    monkeypatch.setattr(cr.time,'time',lambda:1001)
    assert not cr.submit(tmp_path,'minyeong',cap,run)['submitted'] and not calls
    cr.reconcile(tmp_path,{'9001':job},set(),{},set(),lambda j:cr.write(tmp_path/'vast/state/jobs/9001.json',j),now=1001)
    cr.write(cr.base(tmp_path)/'submission_intents'/path.name,{'status':'submitting'})
    assert cr.submit(tmp_path,'minyeong',cap,run)['preserved_intents'] and not calls


def test_acknowledged_return_submits_once_and_preserves_original_receipt(tmp_path,monkeypatch):
    _,job,_,ticket=fixture(tmp_path);calls=[]
    monkeypatch.setattr(cr.time,'time',lambda:1001)
    cr.reconcile(tmp_path,{'9001':job},set(),{},set(),lambda j:cr.write(tmp_path/'vast/state/jobs/9001.json',j),now=1001)
    cap=lambda:{'status':'observed','observed_at':1001,'account_slots':{'minyeong':1},'compatible_available_slots':5}
    def run(*a,**k):calls.append(a);return SimpleNamespace(returncode=0,stdout='1234\n',stderr='')
    first=cr.submit(tmp_path,'minyeong',cap,run);second=cr.submit(tmp_path,'minyeong',cap,run)
    assert len(first['submitted'])==1 and not second['submitted'] and len(calls)==1
    receipt=cr.read(ticket['submission_path'])
    assert receipt['job_id']=='1234' and receipt['previous_vast_receipt']['job_id']=='9001'
    assert cr.read(ticket['claim_path'])['physical_dedup_key']==ticket['physical_dedup_key']
    assert cr.outstanding_claims(tmp_path)=={}


def test_existing_coordinator_hook_uses_same_durable_boundary(tmp_path,monkeypatch):
    from scripts.campaign_20261110.infra_vast import cluster_return as existing
    _,job,_,_=fixture(tmp_path);saved=[]
    monkeypatch.setattr(existing,'consumed_return_reader',lambda:cr)
    assert existing.has_pending(tmp_path)
    result=existing.reconcile(tmp_path,{'9001':job},set(),{},set(),lambda j:saved.append(j),now=1001)
    assert result['consumed_maps']['acknowledged']==['9001']
    assert saved[0]['status']=='cluster_reserved'
    assert not existing.has_pending(tmp_path)
