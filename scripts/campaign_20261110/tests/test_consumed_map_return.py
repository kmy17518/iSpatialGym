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


def test_failed_durable_claim_cannot_publish_or_recover_false_ack(tmp_path):
    _,job,path,_=fixture(tmp_path)
    def fail_before_write(candidate):raise OSError('disk write did not occur')
    result=cr.reconcile(tmp_path,{'9001':job},set(),{},set(),fail_before_write,now=1001)
    assert job['status']=='queued' and 'consumed_cluster_return'not in job
    assert not (cr.base(tmp_path)/'acknowledgements'/path.name).exists()
    assert result['rejected'] and not result['deferred']
    saved=[]
    repeated=cr.reconcile(tmp_path,{'9001':job},set(),{},set(),lambda j:saved.append(j),now=1002)
    assert not repeated['acknowledged'] and not saved


def test_optional_route_helper_is_hash_bound_and_client_only(tmp_path, monkeypatch):
    helper = tmp_path/'routes.py'; helper.write_text("def probe(): return {'status': 'sentinel'}\n")
    monkeypatch.setenv('ISG_CONSUMED_RETURN_ROUTE_HELPER', str(helper))
    monkeypatch.setenv('ISG_CONSUMED_RETURN_ROUTE_HELPER_SHA256', '0'*64)
    with pytest.raises(ValueError, match='Route helper identity'):
        cr.probe()
    monkeypatch.setenv('ISG_CONSUMED_RETURN_ROUTE_HELPER_SHA256', cr.sha(helper))
    assert cr.probe() == {'status': 'sentinel'}


def test_client_routed_submit_keeps_durable_intent_and_prepared_identity(tmp_path, monkeypatch):
    _, job, _, ticket = fixture(tmp_path); calls = []
    monkeypatch.setattr(cr.time, 'time', lambda: 1001)
    cr.reconcile(tmp_path, {'9001': job}, set(), {}, set(), lambda j: cr.write(tmp_path/'vast/state/jobs/9001.json', j), now=1001)
    command = ['sbatch', '--parsable', '--account=viscam', '--partition=viscam', '--gres=gpu:a40:1', ticket['command'][-1]]
    routing = {'command': command, 'route': {'node': 'viscam2'}, 'prepared_command': ticket['command']}
    helper = SimpleNamespace(reserve_unobserved_receipts=lambda r, c: c, route_ticket=lambda t, c, a: routing)
    monkeypatch.setattr(cr, 'client_route_helper', lambda: helper)
    cap = lambda: {'status': 'observed', 'observed_at': 1001, 'account_slots': {'minyeong': 1},
                   'compatible_available_slots': 1, 'preparation_slots': 1, 'probe': {'jobs': []}}
    def run(argv, **kwargs):
        intent = cr.read(next((cr.base(tmp_path)/'submission_intents').glob('*.json')))
        assert intent['command'] == argv == command and intent['prepared_command'] == ticket['command']
        calls.append(argv); return SimpleNamespace(returncode=0, stdout='1234\n', stderr='')
    assert len(cr.submit(tmp_path, 'minyeong', cap, run)['submitted']) == 1
    assert not cr.submit(tmp_path, 'minyeong', cap, run)['submitted']
    receipt = cr.read(ticket['submission_path'])
    assert receipt['prepared_command'] == ticket['command'] and receipt['command'] == command
    assert receipt['previous_vast_receipt']['job_id'] == '9001' and len(calls) == 1


def many_returns(root, count, *, acknowledge=False):
    import copy
    request, oldjob, oldpath, template = fixture(root)
    oldpath.unlink();tickets=[];jobs={}
    for i in range(count):
        key=f'{i:064x}';jid=str(9100+i);row=copy.deepcopy(template)
        row.update(id=jid,physical_dedup_key=key,dedup_key=key,
                   claim_path=str(root/'claims'/(key+'.json')),submission_path=str(root/'receipts'/(key+'.json')),
                   output=str(root/'prepared'/jid),workstream='training preparation')
        Path(row['output']).mkdir(parents=True)
        job={**oldjob,'id':jid,'job_dir':str(root/'out'/jid),'physical_dedup_key':key,
             'claim_path':row['claim_path'],'submission_path':row['submission_path']}
        native=Path(job['job_dir'])/'native_inputs.json';cr.write(native,{'index':i})
        job['native_inputs_sha256']=cr.sha(native)
        cr.write(row['claim_path'],{'physical_dedup_key':key})
        cr.write(row['submission_path'],{'backend':'vast','job_id':jid,'physical_dedup_key':key})
        cr.write(root/'vast/queue'/(jid+'.json'),job);cr.write(root/'vast/state/jobs'/(jid+'.json'),job)
        tickets.append(row);jobs[jid]=job
    source=root/'source.json';manifest=root/'tickets.json';scope=root/'scope.json'
    cr.write(source,{'jobs':tickets});cr.write(manifest,{'tickets':tickets})
    contract=cr.read(scope);contract['approved_manifests'][0]['manifest']=cr.ref(source);cr.write(scope,contract)
    policy=cr.read(root/'infra/coordinator_dispatch/policy.json')
    policy.update(prepared_manifests=[cr.ref(manifest)],preparation_scope={'policy_ref':cr.ref(scope),
                  'normalized_manifests':[{'manifest':cr.ref(manifest),'source_manifest':cr.ref(source)}]})
    cr.write(root/'infra/coordinator_dispatch/policy.json',policy)
    if acknowledge:
        for i,(jid,job) in enumerate(jobs.items()):
            req={**request,'job_id':jid,'manifest':cr.ref(manifest),'ticket_index':i,
                 'queue':cr.ref(root/'vast/queue'/(jid+'.json')),
                 'native_inputs':cr.ref(Path(job['job_dir'])/'native_inputs.json'),
                 'prior_receipt':cr.ref(tickets[i]['submission_path'])}
            path=cr.base(root)/'requests'/(jid+'.json');cr.write(path,req)
            ack={'job_id':jid,'account':'minyeong','manifest':req['manifest'],'ticket_index':i,
                 'submission_path':tickets[i]['submission_path'],'request':cr.ref(path),'prior_receipt':req['prior_receipt']}
            job.update(status='cluster_reserved',consumed_cluster_return=ack)
            cr.write(root/'vast/state/jobs'/(jid+'.json'),job)
            cr.write(cr.base(root)/'acknowledgements'/path.name,ack)
    return tickets,jobs


def test_select_fills_spare_capacity_and_reads_shared_immutable_inputs_once(tmp_path,monkeypatch):
    rows,_=many_returns(tmp_path,20);monkeypatch.setattr(cr.time,'time',lambda:1001)
    reads={};original=Path.read_bytes
    def counted(path):
        reads[str(path)]=reads.get(str(path),0)+1;return original(path)
    monkeypatch.setattr(Path,'read_bytes',counted)
    calls=[]
    def cap():
        calls.append(1);return {'status':'observed','observed_at':1001,'account_slots':{'minyeong':15},'preparation_slots':15}
    result=cr.select(tmp_path,cap)
    assert len(result['requested'])==15 and len(calls)==1 and not result['errors']
    assert reads[str(tmp_path/'tickets.json')]==1
    assert reads[str(tmp_path/'source.json')]==1
    assert reads[rows[0]['input_refs'][0]['path']]==1


def test_selector_cursor_advances_past_slow_completed_history(tmp_path,monkeypatch):
    rows,jobs=many_returns(tmp_path,250);clock=[0.0];original=cr.read
    for jid,job in list(jobs.items())[:200]:
        job['status']='done';cr.write(tmp_path/'vast/state/jobs'/(jid+'.json'),job)
    def slow_read(path):
        if Path(path).parent.name=='jobs':clock[0]+=.1
        return original(path)
    monkeypatch.setattr(cr,'read',slow_read);monkeypatch.setattr(cr.time,'time',lambda:1001)
    monkeypatch.setattr(cr.time,'monotonic',lambda:clock[0])
    cap=lambda:{'status':'observed','observed_at':1001,'account_slots':{'minyeong':20},'preparation_slots':20}
    first=cr.select(tmp_path,cap);second=cr.select(tmp_path,cap)
    assert not first['requested'] and first['scanned']>=120
    assert len(second['requested'])==20 and second['scanned']<first['scanned']


def test_submit_batches_with_one_observation_and_holds_existing_lock(tmp_path,monkeypatch):
    import fcntl
    rows,_=many_returns(tmp_path,12,acknowledge=True);monkeypatch.setattr(cr.time,'time',lambda:1001)
    probes=[];calls=[]
    def cap():
        probes.append(1);return {'status':'observed','observed_at':1001,'account_slots':{'minyeong':7},'preparation_slots':7}
    def run(command,**kwargs):
        with (tmp_path/'infra/coordinator_dispatch/capacity_allocation.lock').open('a') as contender:
            with pytest.raises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
        calls.append(command);return SimpleNamespace(returncode=0,stdout=str(2000+len(calls)),stderr='')
    report=cr.submit(tmp_path,'minyeong',cap,run)
    assert len(report['submitted'])==7 and len(probes)==1 and len(calls)==7
    assert not report['errors'] and report['capacity_probes']==1


def test_submit_cursor_reaches_pending_ack_after_slow_completed_history(tmp_path,monkeypatch):
    rows,jobs=many_returns(tmp_path,250,acknowledge=True);clock=[0.0];original=cr.read
    for row,jid in zip(rows[:200],jobs):
        cr.write(row['submission_path'],{'backend':'cluster','previous_vast_job_id':jid,'job_id':'123'})
    def slow_read(path):
        if Path(path).parent.name=='acknowledgements':clock[0]+=.1
        return original(path)
    monkeypatch.setattr(cr,'read',slow_read);monkeypatch.setattr(cr.time,'time',lambda:1001)
    monkeypatch.setattr(cr.time,'monotonic',lambda:clock[0])
    cap=lambda:{'status':'observed','observed_at':1001,'account_slots':{'minyeong':20},'preparation_slots':20}
    run=lambda *a,**k:SimpleNamespace(returncode=0,stdout='1234',stderr='')
    first=cr.submit(tmp_path,'minyeong',cap,run);second=cr.submit(tmp_path,'minyeong',cap,run)
    assert not first['submitted'] and first['scanned']>=120
    assert len(second['submitted'])==20


def test_expired_batch_refreshes_and_reserves_previous_submissions(tmp_path,monkeypatch):
    rows,_=many_returns(tmp_path,3,acknowledge=True);now=[1001];probes=[];reservations=[]
    monkeypatch.setattr(cr.time,'time',lambda:now[0]);monkeypatch.setattr(cr.time,'monotonic',lambda:0)
    def cap():
        probes.append(1);return {'status':'observed','observed_at':now[0],'account_slots':{'minyeong':2},'preparation_slots':2,'probe':{'jobs':[]}}
    def reserve(root,capacity):
        # A fresh observation sees no scheduler job yet; durable receipts must
        # still reserve the first job before routing a second account/client.
        capacity['_reservations']=[cr.read(p)['preparation_route']['route'] for p in (cr.base(root)/'submissions').glob('*.json')]
    def route(row,capacity,account):
        reservations.append(len(capacity.get('_reservations',[])))
        if len(capacity.get('_reservations',[]))>=2:return None
        return {'command':row['command'],'route':{'node':'viscam2'}}
    helper=SimpleNamespace(reserve_unobserved_receipts=reserve,route_ticket=route)
    monkeypatch.setattr(cr,'client_route_helper',lambda:helper)
    calls=[]
    def run(*a,**k):
        calls.append(1);now[0]+=61;return SimpleNamespace(returncode=0,stdout=str(3000+len(calls)),stderr='')
    result=cr.submit(tmp_path,'minyeong',cap,run)
    assert len(calls)==2 and len(result['submitted'])==2 and reservations==[0,1,2]
    assert len(probes)==3


def test_stale_capacity_never_creates_submission_intent(tmp_path,monkeypatch):
    many_returns(tmp_path,2,acknowledge=True);monkeypatch.setattr(cr.time,'time',lambda:1100)
    cap=lambda:{'status':'observed','observed_at':1000,'account_slots':{'minyeong':20},'preparation_slots':20}
    report=cr.submit(tmp_path,'minyeong',cap,lambda *a,**k:pytest.fail('stale submission'))
    assert report['errors'] and not report['submitted']
    assert not list((cr.base(tmp_path)/'submission_intents').glob('*.json'))


def test_ambiguous_submission_ends_batch_and_preserves_non_retryable_intent(tmp_path,monkeypatch):
    many_returns(tmp_path,3,acknowledge=True);monkeypatch.setattr(cr.time,'time',lambda:1001)
    cap=lambda:{'status':'observed','observed_at':1001,'account_slots':{'minyeong':3},'preparation_slots':3}
    calls=[]
    def run(*a,**k):calls.append(1);return SimpleNamespace(returncode=0,stdout='unknown',stderr='')
    result=cr.submit(tmp_path,'minyeong',cap,run)
    assert len(calls)==1 and result['errors'] and not result['submitted']
    assert len(list((cr.base(tmp_path)/'submission_intents').glob('*.json')))==1


def test_competing_clients_cannot_spend_same_unobserved_physical_slots(tmp_path,monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    many_returns(tmp_path,4,acknowledge=True);monkeypatch.setattr(cr.time,'time',lambda:1001)
    entered=Event();release=Event();calls=[];probes=[]
    def cap():
        probes.append(1);return {'status':'observed','observed_at':1001,'account_slots':{'minyeong':2},'preparation_slots':2}
    def reserve(root,capacity):
        capacity['_reservations']=[cr.read(p)['preparation_route']['route'] for p in (cr.base(root)/'submissions').glob('*.json')]
    def route(row,capacity,account):
        if len(capacity.get('_reservations',[]))>=2:return None
        return {'command':row['command'],'route':{'node':'viscam2'}}
    monkeypatch.setattr(cr,'client_route_helper',lambda:SimpleNamespace(reserve_unobserved_receipts=reserve,route_ticket=route))
    def run(*a,**k):
        calls.append(1)
        if len(calls)==1:entered.set();assert release.wait(2)
        return SimpleNamespace(returncode=0,stdout=str(4000+len(calls)),stderr='')
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(cr.submit,tmp_path,'minyeong',cap,run)
        assert entered.wait(2)
        second=pool.submit(cr.submit,tmp_path,'minyeong',cap,run)
        release.set();reports=[first.result(),second.result()]
    assert len(calls)==2 and sum(len(r['submitted'])for r in reports)==2
    assert len(probes)==2 and all(not r['errors']for r in reports)


def test_select_never_reads_private_unadmitted_manifest(tmp_path,monkeypatch):
    many_returns(tmp_path,2);policy_path=tmp_path/'infra/coordinator_dispatch/policy.json'
    policy=cr.read(policy_path);private=tmp_path/'private/retired.json'
    policy['prepared_manifests'].append({'path':str(private),'sha256':'0'*64});cr.write(policy_path,policy)
    original=Path.read_bytes
    def read_bytes(path):
        assert path!=private,'retired private manifest must not be touched'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',read_bytes);monkeypatch.setattr(cr.time,'time',lambda:1001)
    cap=lambda:{'status':'observed','observed_at':1001,'account_slots':{'minyeong':2},'preparation_slots':2}
    assert len(cr.select(tmp_path,cap)['requested'])==2


def test_immutable_cache_never_accepts_same_path_with_changed_expected_hash(tmp_path):
    fixture(tmp_path);cache=cr.ClientInputs(tmp_path);path=tmp_path/'library.json'
    assert cache.checked(cr.ref(path))=={'grasps':3}
    with pytest.raises(ValueError,match='Immutable input changed'):
        cache.checked({'path':str(path),'sha256':'0'*64})
