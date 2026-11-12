"""Exclusive return of never-assigned consumed maps to existing cluster routes.

The Vast coordinator alone tombstones queued ownership. Root account clients
only submit after that durable acknowledgement; ambiguous intents are retained.
"""
import argparse
import bisect
from contextlib import contextmanager
import fcntl
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import uuid


SCHEMA = 'isg-consumed-map-cluster-return-v1'


def require(value, message):
    if not value: raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {'path': str(path), 'sha256': sha(path)}


def checked(reference):
    require(sha(reference['path']) == reference['sha256'], 'Immutable input changed: '+reference['path'])
    return read(reference['path'])


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.stat().st_uid == os.geteuid(): path.parent.chmod(0o2770)
    temp = path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.chmod(0o660); temp.replace(path)


def base(root):
    return Path(root)/'vast/consumed_cluster_return'


@contextmanager
def allocation_lock(root):
    p = Path(root)/'infra/coordinator_dispatch/capacity_allocation.lock'
    with p.open('a') as stream:
        if os.fstat(stream.fileno()).st_uid == os.geteuid(): os.fchmod(stream.fileno(),0o660)
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def resolve_ticket(root, reference, index, account, *, verify_inputs=True, cache=None):
    policy = cache.policy if cache is not None else read(Path(root)/'infra/coordinator_dispatch/policy.json')
    pinned = cache.checked if cache is not None else checked
    require(reference in policy['prepared_manifests'], 'Route is not admitted')
    ticket = pinned(reference)['tickets'][index]
    require(ticket['expected_account'] == account and ticket['consumer_kind'] == 'tabletop'
            and ticket['computation_kind'] == 'reach_map' and ticket['production_released'] is False,
            'Only admitted consumed tabletop maps may return')
    scope = policy['preparation_scope']; contract = pinned(scope['policy_ref'])
    link = next((v for v in scope['normalized_manifests'] if v['manifest'] == reference), None)
    require(link is not None, 'Missing exact normalized source')
    source = next((v for v in contract['approved_manifests'] if v['manifest'] == link['source_manifest']), None)
    require(source is not None and source['account'] == account and source['consumer_kind'] == 'tabletop',
            'Source account or consumed scope differs')
    original = pinned(link['source_manifest'])
    key = ticket['physical_dedup_key']
    require(key == ticket['dedup_key'] and any(e['physical_dedup_key'] == key for e in original['jobs']), 'Physical source differs')
    for field, namespace in [('claim_path','claim_namespace'),('submission_path','receipt_namespace')]:
        require(Path(ticket[field]).parent == Path(source[namespace]) and Path(ticket[field]).name == key+'.json', 'Shared ownership path differs')
    require(ticket['command'][0] == 'sbatch' and ticket['hard_time_seconds'] <= 1800, 'Unexpected map launch')
    if verify_inputs:
        for path,digest in ticket['inputs'].items():
            if cache is None: require(sha(path) == digest, 'Pinned cluster input changed: '+path)
            else: cache.verify({'path':path,'sha256':digest})
    output = Path(ticket['output'])
    owned = Path(root) if account == 'minyeong' else Path('/vision/u/shiyuc/minyeong')/Path(root).name
    require(output.is_relative_to(owned) and not any((output/n).exists() for n in ('started.json','terminal.json')),
            'Cluster output already started or outside account root')
    return ticket


def validate(root, request, job, staging, heartbeat_jobs, remote_jobs, now, outstanding=()):
    root = Path(root); jid = str(request['job_id'])
    require(request['schema'] == SCHEMA and request['campaign_id'] == root.name, 'Request identity differs')
    require(0 <= now-request['observed_at'] <= 120 and now <= request['expires_at'] <= request['observed_at']+180,
            'Capacity request expired')
    require(str(job['id']) == jid and job.get('status') == 'queued' and not job.get('tries') and not job.get('instance')
            and not job.get('history') and not job.get('consumed_cluster_return'), 'Job is not never-assigned queued work')
    require(jid not in staging and jid not in heartbeat_jobs and jid not in remote_jobs, 'Job retains an assignment owner')
    directory = Path(job['job_dir'])
    require(job['job_kind'] == 'native_dependency' and directory.is_relative_to(root/'out/vast_map_prebuild')
            and Path(job['script']).name == 'job_map_prebuild.sh' and not(directory/'result').exists(), 'Not untouched typed map work')
    require(request['queue'] == ref(root/'vast/queue'/(jid+'.json')), 'Queued execution changed')
    require(request['native_inputs']['path'] == str(directory/'native_inputs.json')
            and request['native_inputs']['sha256'] == job['native_inputs_sha256'], 'Native identity changed')
    native = checked(request['native_inputs'])
    require(native.get('source_consumed_map_ticket') is True and native['kind'] == 'tabletop_maps'
            and native['new_native_instances'] == 0 and native['production_released'] is False, 'Consumed purpose differs')
    ticket = resolve_ticket(root,request['manifest'],request['ticket_index'],request['account'])
    key = ticket['physical_dedup_key']
    for field in ['physical_dedup_key','claim_path','submission_path']:
        require(native[field] == ticket[field] == job[field], 'Native/cluster physical ownership differs')
    primary = read(native['source_request'])['jobs'][native['entry_index']]
    for field in ['physical_dedup_key','input_refs','runtime_ref','configuration_ids']:
        require(primary[field] == ticket[field], 'Alternate route changed exact compute inputs')
    for path,digest in {**native['files'],**native['library_files'],**native.get('seed_map_files',{})}.items():
        require(sha(path) == digest, 'Native payload changed: '+path)
    receipt = checked(request['prior_receipt'])
    require(request['prior_receipt']['path'] == ticket['submission_path'] and receipt['backend'] == 'vast'
            and str(receipt['job_id']) == jid and receipt['physical_dedup_key'] == key, 'Current physical owner differs')
    require(read(ticket['claim_path'])['physical_dedup_key'] == key, 'Shared physical claim differs')
    capacity = checked(request['capacity'])
    require(Path(request['capacity']['path']).is_relative_to(base(root)/'capacity')
            and 0 <= now-capacity['observed_at'] <= 180, 'Capacity proof expired or outside coordinator')
    selected = capacity['selected_job_ids']; accounts = capacity['selected_accounts']
    require(len(set(selected)) == len(selected) and jid in selected and set(accounts) == set(selected)
            and accounts[jid] == request['account'], 'Capacity account selection differs')
    require(capacity['critical_reserved_physical_slots'] >= 4 and len(set(selected)|set(outstanding)) <= capacity['preparation_slots'],
            'Aggregate returns exceed reserved preparation capacity')
    for account in set(accounts.values()):
        require(sum(a == account for a in accounts.values()) <= capacity['account_slots'].get(account,0), 'Account capacity exceeded')
    return {'job_id':jid,'account':request['account'],'physical_dedup_key':key,'manifest':request['manifest'],
            'ticket_index':request['ticket_index'],'prior_receipt':request['prior_receipt'],
            'submission_path':ticket['submission_path'],'capacity':request['capacity']}


def outstanding_claims(root):
    pending = {}
    for p in (base(root)/'acknowledgements').glob('*.json'):
        ack = read(p); current = read(ack['submission_path'])
        if current.get('backend') != 'cluster' or current.get('previous_vast_job_id') != ack['job_id']:
            pending[ack['job_id']] = ack['account']
    return pending


def has_pending(root):
    return any(not(base(root)/'acknowledgements'/p.name).exists() and not(base(root)/'rejected'/p.name).exists()
               for p in (base(root)/'requests').glob('*.json'))


def reconcile(root, states, staging, heartbeats, remote_jobs, save_state, *, now=None):
    now = time.time() if now is None else now; report = {'observed_at':now,'acknowledged':[],'rejected':[],'deferred':[]}
    began = time.monotonic()
    running = {str(j) for hb in heartbeats.values() for j in (hb.get('running') or {})}
    outstanding = outstanding_claims(root)
    for path in sorted((base(root)/'requests').glob('*.json')):
        if time.monotonic()-began > 15: break
        ack_path = base(root)/'acknowledgements'/path.name
        if ack_path.exists() or (base(root)/'rejected'/path.name).exists(): continue
        request = read(path); job = states.get(str(request['job_id']),{}); reference = ref(path)
        try:
            prior = job.get('consumed_cluster_return',{})
            if job.get('status') == 'cluster_reserved' and prior.get('request') == reference:
                claim = prior
            else:
                identity = validate(root,request,job,staging,running,remote_jobs,now,outstanding)
                claim = {**identity,'request':reference,'claimed_at':now,'boundary':'Sole Vast coordinator removed untouched map before cluster submission'}
                candidate={**job,'status':'cluster_reserved','consumed_cluster_return':claim}
                save_state(candidate)
                job.update(candidate)
            write(ack_path,claim);outstanding[claim['job_id']] = claim['account'];report['acknowledged'].append(claim['job_id'])
        except Exception as error:
            row={'request':reference,'error':str(error),'cluster_submission_allowed':False}
            if job.get('status') == 'cluster_reserved' and job.get('consumed_cluster_return',{}).get('request') == reference:
                report['deferred'].append(row)
            else:write(base(root)/'rejected'/path.name,row);report['rejected'].append(row)
    write(base(root)/'coordinator_status.json',report);return report


def client_route_helper():
    """Optional client-only routing; the sole Vast acknowledgement is unchanged."""
    path=os.environ.get('ISG_CONSUMED_RETURN_ROUTE_HELPER')
    if not path:return None
    require(sha(path)==os.environ.get('ISG_CONSUMED_RETURN_ROUTE_HELPER_SHA256'),'Route helper identity differs')
    spec=importlib.util.spec_from_file_location('consumed_return_routes',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def probe():
    routes=client_route_helper()
    if routes is not None:return routes.probe()
    path=os.environ.get('ISG_CONSUMED_RETURN_CAPACITY_HELPER')
    if path:
        require(sha(path)==os.environ.get('ISG_CONSUMED_RETURN_CAPACITY_HELPER_SHA256'),'Capacity helper identity differs')
        spec=importlib.util.spec_from_file_location('consumed_return_capacity',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.probe()
    from scripts.campaign_20261110.coordinator_dispatch import probe as observed_capacity
    return observed_capacity()


class ClientInputs:
    """One-call cache for hash-bound immutable inputs, never mutable receipts."""
    def __init__(self, root):
        self.policy = read(Path(root)/'infra/coordinator_dispatch/policy.json')
        self.payloads = {}; self.values = {}

    def verify(self, reference):
        path = reference['path']; expected = reference['sha256']
        key = (path, expected)
        if key not in self.payloads:
            payload = Path(path).read_bytes()
            require(hashlib.sha256(payload).hexdigest() == expected, 'Immutable input changed: '+path)
            self.payloads[key] = payload
        return self.payloads[key]

    def checked(self, reference):
        key = (reference['path'], reference['sha256'])
        if key not in self.values:self.values[key] = json.loads(self.verify(reference))
        return self.values[key]


def scan_order(keys, cursor):
    keys = sorted(keys); at = bisect.bisect_right(keys, cursor or '')
    return keys[at:]+keys[:at]


def fresh_capacity(root, capacity_probe, helper):
    cap = helper.probe() if capacity_probe is probe and helper is not None else capacity_probe()
    require(cap.get('status') == 'observed' and 0 <= time.time()-cap['observed_at'] <= 60,
            'Fresh cluster capacity unavailable')
    if helper is not None:helper.reserve_unobserved_receipts(root, cap)
    require(0 <= time.time()-cap['observed_at'] <= 60, 'Capacity expired while reserving receipts')
    return cap


def preparation_slots(cap):
    return max(0, cap['preparation_slots'] if 'preparation_slots' in cap else cap['compatible_available_slots']-4)


def select(root, capacity_probe=probe):
    root=Path(root);report={'requested':[],'errors':[],'scanned':0};started=time.monotonic()
    cursor_path=base(root)/'select_cursor.json'
    with allocation_lock(root):
        cache=ClientInputs(root);policy=cache.policy;routes={}
        # Only the admitted consumed-preparation namespace may be traversed.
        for link in policy['preparation_scope']['normalized_manifests']:
            manifest=link['manifest']
            require(manifest in policy['prepared_manifests'],'Normalized route is not admitted')
            for i,t in enumerate(cache.checked(manifest)['tickets']):
                if (t.get('workstream')=='training preparation' and t.get('consumer_kind')=='tabletop'
                        and t.get('computation_kind')=='reach_map'):
                    routes.setdefault(t['physical_dedup_key'],[]).append((manifest,i,t))
        helper=client_route_helper()
        try:
            capacity=fresh_capacity(root,capacity_probe,helper)
            if helper is not None:capacity=helper.build_capacity(capacity['probe'],reservations=capacity.get('_reservations',[]))
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as error:
            report['errors'].append({'owner':'/root','stage':'capacity_probe','error':str(error)})
            report['blocked']='Fresh cluster capacity unavailable'
            report['observed_at']=time.time();write(base(root)/'select_status.json',report);return report
        now=time.time();outstanding=outstanding_claims(root);pending={}
        for p in (base(root)/'requests').glob('*.json'):
            if not(base(root)/'acknowledgements'/p.name).exists() and not(base(root)/'rejected'/p.name).exists():
                r=read(p)
                if r['expires_at']>=now:pending[r['job_id']]=r['account']
        reserved={**outstanding,**pending};slots=preparation_slots(capacity)
        available=max(0,slots-len(reserved));accounts={a:max(0,n-sum(x==a for x in reserved.values()))for a,n in capacity['account_slots'].items()}
        cursor=read(cursor_path).get('after','') if cursor_path.exists() else ''
        chosen=[];seen=set();began=time.monotonic()
        for key in scan_order(routes,cursor):
            if len(chosen)>=min(available,128) or time.monotonic()-began>12:break
            cursor=key;report['scanned']+=1
            for manifest,index,ticket in sorted(routes[key],key=lambda v:-accounts.get(v[2]['expected_account'],0)):
                account=ticket['expected_account'];receipt_path=Path(ticket['submission_path'])
                if accounts.get(account,0)<=0 or not receipt_path.exists():continue
                receipt=read(receipt_path);jid=str(receipt.get('job_id',''))
                if receipt.get('backend')!='vast' or jid in reserved or jid in seen:continue
                state_path=root/'vast/state/jobs'/(jid+'.json')
                if not state_path.exists():continue
                job=read(state_path)
                if job.get('status')!='queued' or job.get('tries') or job.get('instance') or job.get('history'):continue
                try:
                    resolve_ticket(root,manifest,index,account,cache=cache)
                    request={'schema':SCHEMA,'campaign_id':root.name,'job_id':jid,'account':account,'observed_at':now,'expires_at':now+180,
                             'manifest':manifest,'ticket_index':index,'prior_receipt':ref(receipt_path),
                             'queue':ref(root/'vast/queue'/(jid+'.json')),'native_inputs':ref(Path(job['job_dir'])/'native_inputs.json')}
                    chosen.append(request);seen.add(jid);accounts[account]-=1;break
                except (OSError,ValueError,KeyError)as error:report['errors'].append({'job_id':jid,'error':str(error)})
        write(cursor_path,{'after':cursor,'observed_at':time.time()})
        if chosen and not 0 <= time.time()-capacity['observed_at'] <= 60:
            report['blocked']='Capacity expired during selection; no requests published';chosen=[]
        if chosen:
            stamp=str(time.time_ns());p=base(root)/'capacity'/(stamp+'.json')
            write(p,{'observed_at':capacity['observed_at'],'preparation_slots':slots,'critical_reserved_physical_slots':4,
                     'critical_reserve_meaning':'Policy retains four Titan slots whenever physically available; actual held count is separately observed.',
                     'critical_titan_slots_actually_held':capacity.get('critical_titan_slots_actually_held'),
                     'account_slots':capacity['account_slots'],'selected_job_ids':[r['job_id']for r in chosen],
                     'selected_accounts':{r['job_id']:r['account']for r in chosen},'observation':capacity})
            capacity_ref=ref(p)
            for r in chosen:r['capacity']=capacity_ref;write(base(root)/'requests'/(r['job_id']+'_'+stamp+'.json'),r)
            report['requested']=[r['job_id']for r in chosen]
        if not available:report['blocked']='No preparation capacity after four critical GPUs and prior returns'
    report.update(observed_at=time.time(),elapsed_seconds=time.monotonic()-started)
    write(base(root)/'select_status.json',report);return report


def submit(root, account, capacity_probe=probe, runner=subprocess.run):
    root=Path(root);report={'submitted':[],'errors':[],'preserved_intents':[],'scanned':0,'capacity_probes':0}
    started=time.monotonic();cursor_path=base(root)/('submit_'+account+'_cursor.json')
    # The observation and all local reservations stay under the same existing
    # lock. Other account clients cannot spend the same physical observation.
    with allocation_lock(root):
        cache=ClientInputs(root);helper=client_route_helper();cap=None;began=time.monotonic()
        cursor=read(cursor_path).get('after','') if cursor_path.exists() else ''
        paths={p.name:p for p in (base(root)/'acknowledgements').glob('*.json')}
        for name in scan_order(paths,cursor):
            if len(report['submitted'])>=128 or time.monotonic()-began>12:break
            cursor=name;report['scanned']+=1;path=paths[name]
            ack=read(path)
            if ack['account']!=account:continue
            current=read(ack['submission_path'])
            if current.get('backend')=='cluster' and current.get('previous_vast_job_id')==ack['job_id']:continue
            intent=base(root)/'submission_intents'/path.name
            if intent.exists():report['preserved_intents'].append(str(intent));continue
            try:
                ticket=resolve_ticket(root,ack['manifest'],ack['ticket_index'],account,cache=cache)
                cache.checked(ack['request']);cache.checked(ack['prior_receipt'])
                job=read(root/'vast/state/jobs'/(ack['job_id']+'.json'))
                require(job['status']=='cluster_reserved' and job['consumed_cluster_return']['request']==ack['request'],'Sole coordinator acknowledgement differs')
                if cap is None or not 0<=time.time()-cap['observed_at']<=60:
                    cap=fresh_capacity(root,capacity_probe,helper);report['capacity_probes']+=1
                    # Setup/probe/history reconciliation is not the useful-work budget.
                    if report['capacity_probes']==1:began=time.monotonic()
                if min(cap['account_slots'].get(account,0),preparation_slots(cap))<=0:break
                routing=helper.route_ticket(ticket,cap,account)if helper is not None else None
                if helper is not None and routing is None:continue
                command=routing['command']if routing is not None else ticket['command']
                acknowledgement=ref(path)
                value={'at':time.time(),'account':account,'acknowledgement':acknowledgement,'previous_vast_job_id':ack['job_id'],
                       'prepared_command':ticket['command'],'command':command,'preparation_route':routing,'status':'submitting'}
                intent.parent.mkdir(parents=True,exist_ok=True)
                if intent.parent.stat().st_uid==os.geteuid():intent.parent.chmod(0o2770)
                with intent.open('x')as stream:json.dump(value,stream)
                intent.chmod(0o660)
                result=runner(command,capture_output=True,text=True,timeout=30)
                value.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr);write(intent,value)
                require(result.returncode==0,'Cluster submission failed; preserve intent')
                jid=result.stdout.strip().split(';')[0];require(jid.isdigit(),'Unknown scheduler acknowledgement; preserve intent')
                receipt={**current,'backend':'cluster','account':account,'job_id':jid,'status':'submitted','submitted_at':time.time(),
                         'previous_vast_job_id':ack['job_id'],'previous_vast_receipt':current,'return_acknowledgement':acknowledgement,
                         'prepared':ticket['request'],'output':ticket['output'],'prepared_command':ticket['command'],
                         'command':command,'preparation_route':routing}
                write(ack['submission_path'],receipt);write(base(root)/'submissions'/path.name,receipt)
                report['submitted'].append({'job_id':jid,'previous_vast_job_id':ack['job_id'],'account':account})
                if routing is not None:cap.setdefault('_reservations',[]).append(routing['route'])
                else:
                    cap['account_slots'][account]-=1
                    cap['preparation_slots']=max(0,preparation_slots(cap)-1)
            except (OSError,ValueError,KeyError,subprocess.SubprocessError)as error:
                report['errors'].append({'ack':str(path),'error':str(error)})
                # Any uncertain external effect ends the batch, with its durable
                # intent left intact for explicit reconciliation, never retry.
                if intent.exists():break
        write(cursor_path,{'after':cursor,'observed_at':time.time()})
    report.update(observed_at=time.time(),elapsed_seconds=time.monotonic()-started)
    write(base(root)/('submit_'+account+'_status.json'),report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--mode',choices=['select','submit'],required=True);p.add_argument('--account',choices=['minyeong','shiyuc'],required=True);a=p.parse_args()
    require(getpass.getuser()==a.account and (a.mode!='select' or a.account=='minyeong'),'Account execution owner differs')
    print(json.dumps(select(a.root)if a.mode=='select'else submit(a.root,a.account)))


if __name__=='__main__':main()
