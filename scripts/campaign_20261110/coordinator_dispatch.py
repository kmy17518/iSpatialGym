"""Root-owned automatic intake for new bounded recovery/preparation requests.

Existing production/replay/acceptance dispatchers keep their queues. This worker
uses each prepared request's existing claim and receipt, never retries ambiguous
submissions, and never grants production or acceptance credit.
"""
import argparse
import fcntl
import getpass
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
from collections import deque

from scripts.campaign_20261110.infra_vast.cluster_return_capacity import probe as original_probe
from scripts.campaign_20261110.infra_dispatch import backend_capacity
from scripts.campaign_20261110 import dispatch_cluster_first
from scripts.campaign_20261110.infra_dispatch import shared_json



def probe():
    """Count real compatible slots without holding all work behind one pending job."""
    observation=dispatch_cluster_first.probe()
    if observation['state']!='available':
        return {'status':'unavailable','observed_at':observation['observed_at'],'accounts_checked':['minyeong','shiyuc'],'account_slots':{},'compatible_available_slots':0}
    demand={'cpus':8,'memory_mb':48*1024,'excluded_nodes':['svl9']}
    nodes=[n for n in observation['nodes'] if 'gres/gpu:titanrtx=' in n.get('CfgTRES','')]
    physical={};routes=[];accounts={}
    for account in ['minyeong','shiyuc']:
        total=0
        for partition in ['svl','viscam']:
            quota=backend_capacity(account+'_'+partition,observation['jobs'])
            available={n['NodeName']:dispatch_cluster_first.slots(n,partition,demand) for n in nodes}
            pending=sum(dispatch_cluster_first.gpu_count(r[3]) for r in observation['jobs'] if r[0]==account and partition in r[1].split(',') and r[2]=='PENDING')
            # Pending work consumes headroom and prospective physical slots; it
            # does not make every other independently available slot unusable.
            slots=min(quota,max(0,sum(available.values())-pending))
            total+=slots
            for name,count in available.items():physical[name]=max(physical.get(name,0),count)
            routes.append({'backend':account+'_'+partition,'slots':slots,'quota_headroom':quota,'pending_gpus':pending})
        accounts[account]=total
    return {'status':'observed','observed_at':observation['observed_at'],'accounts_checked':['minyeong','shiyuc'],
            'compatible_available_slots':min(sum(physical.values()),sum(accounts.values())),
            'account_slots':accounts,'routes':routes,
            'pending_job_ids':[r[6] for r in observation['jobs'] if r[2]=='PENDING']}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_ref(ref):
    if sha(ref['path']) != ref['sha256']:
        raise ValueError('Immutable input changed: '+ref['path'])


def native_ticket(root, prepared_path, policy):
    prepared_path=Path(prepared_path)
    prepared=read(prepared_path)
    if prepared.get('owner') != '/root/dataset_contract' or prepared.get('observed_at',0) < policy['native_not_before']:
        return None
    if prepared.get('status') != 'ready_unsubmitted':
        return None
    request_ref=prepared.get('request')
    if not request_ref:
        return None
    verify_ref(request_ref); request=read(request_ref['path'])
    if request.get('purpose') != 'canonical_recovery_bounded_canary':
        return None
    if request.get('class_id') not in policy['canonical_classes']:
        raise ValueError('Class outside root canonical authorization')
    if not (request.get('production_released') is False and request.get('benchmark_credit')==0
            and 1<=request['count']<=3 and 1<=request['max_draws']<=6 and 1<=request['timeout_seconds']<=1800):
        raise ValueError('Request exceeds bounded native authorization')
    for key in ('request','job','pinned','verifier'):
        verify_ref(prepared[key])
    checks=dict(request['inputs'])
    checks.update({prepared[key]['path']:prepared[key]['sha256'] for key in ('request','job','pinned','verifier')})
    checks[str(prepared_path)]=sha(prepared_path)
    directory=prepared_path.parent
    command=['sbatch','--parsable','--nodes=1','--ntasks=1','-A','vision','-p','svl,viscam',
             '--exclude=svl9','--gres=gpu:titanrtx:1','-c','8','--mem=48G','--time=00:40:00',
             '--nice=0','-J','isg261110_auto_native','-o',str(directory/'native_%j.log'),prepared['pinned']['path']]
    return {'id':directory.name,'expected_account':'minyeong','workstream':'canonical recovery',
            'owner':'/root/dataset_contract','stage':'bounded native','class_id':request['class_id'],
            'request':request_ref,'inputs':checks,'command':command,'claim_path':str(directory/'launch.claim'),
            'claim_kind':'directory','submission_path':str(directory/'submission.json'),
            'ready_since':prepared['observed_at'],'benchmark_credit':0,'production_released':False}


def pending_status(ticket):
    receipt=Path(ticket['submission_path'])
    if receipt.exists():
        value=read(receipt)
        return 'submitted' if value.get('job_id') else 'submission_requires_reconciliation'
    if Path(ticket['claim_path']).exists():
        return 'claim_requires_reconciliation'
    return 'ready'


def collector_allowance(capacity, existing_jobs, account='minyeong', pending_limit=3):
    """Keep downstream validation queued even while preparation fills the GPUs."""
    if capacity.get('status')!='observed':return 0
    free=min(capacity.get('compatible_available_slots',0),capacity.get('account_slots',{}).get(account,0))
    pending=set(map(str,capacity.get('pending_job_ids',[])))&set(map(str,existing_jobs))
    return max(free,max(0,pending_limit-len(pending)))


def dispatch_order(tickets):
    """Keep critical work first and distribute preparation across task classes."""
    priority={'pipeline':0,'canonical recovery':1,'heldout eval':2,'training preparation':3}
    ordered=sorted(tickets,key=lambda row:(priority.get(row.get('workstream'),4),row.get('ready_since',0)))
    groups={}
    for ticket in ordered:
        if ticket.get('workstream')=='training preparation':
            key=ticket.get('class_id') or ticket.get('task') or ticket.get('stage') or ticket['id']
            groups.setdefault(key,deque()).append(ticket)
        elif priority.get(ticket.get('workstream'),4)<3:
            yield ticket
    active=deque(groups.values())
    while active:
        group=active.popleft();yield group.popleft()
        if group:active.append(group)
    yield from (t for t in ordered if priority.get(t.get('workstream'),4)>3)


def submit(ticket, capacity, runner=subprocess.run):
    if pending_status(ticket) != 'ready':
        return None
    for path,digest in ticket['inputs'].items():
        verify_ref({'path':path,'sha256':digest})
    command=list(ticket['command'])
    if not command or command[0]!='sbatch' or '--parsable' not in command:
        raise ValueError('Only exact sbatch argv is supported')
    if ticket.get('workstream')=='training preparation':
        # Keep ready engineering controls ahead of preparation as slots free.
        # This changes scheduler priority only, never the pinned native payload.
        for i,arg in enumerate(command):
            if arg.startswith('--nice='):command[i]='--nice='+str(max(1000,int(arg.split('=',1)[1])))
        if not any(arg.startswith('--nice=') for arg in command):command.insert(2,'--nice=1000')
    claim=Path(ticket['claim_path']);claim.parent.mkdir(parents=True,exist_ok=True)
    record={**ticket,'status':'submitting','observed_at':time.time(),'dispatch_owner':'/root',
            'prepared_command':ticket['command'],'command':command,
            'capacity_observed_at':capacity['observed_at'],'accounts_checked':capacity['accounts_checked']}
    if ticket.get('claim_kind')=='directory':
        claim.mkdir();shared_json(claim/'owner.json',record)
    else:
        with claim.open('x') as stream:
            json.dump(record,stream)
    receipt=Path(ticket['submission_path']);receipt.parent.mkdir(parents=True,exist_ok=True)
    # A durable intent precedes the external effect. Interrupted/ambiguous
    # submissions are left for reconciliation, never silently submitted again.
    shared_json(receipt,record)
    result=runner(command,capture_output=True,text=True,timeout=60)
    jid=result.stdout.strip().split(';')[0]
    record.update(returncode=result.returncode,stderr=result.stderr,
                  job_id=jid if result.returncode==0 and jid.isdigit() else None,
                  status='submitted' if result.returncode==0 and jid.isdigit() else 'submission_requires_reconciliation',
                  submitted_at=time.time())
    shared_json(receipt,record)
    return record


def cycle(root, account, capacity_probe=probe):
    root=Path(root);directory=root/'infra/coordinator_dispatch'
    policy=read(directory/'policy.json');tickets=[];errors=[]
    if policy.get('owner')!='/root' or policy.get('production_released') is not False:
        raise ValueError('Root scheduling policy missing')
    for reconciler in policy.get('reconcilers',[]):
        if reconciler['account']!=account:continue
        try:
            for ref in reconciler['inputs']:verify_ref(ref)
            result=subprocess.run(reconciler['command'],capture_output=True,text=True,
                                  timeout=min(120,reconciler.get('timeout_seconds',60)))
            shared_json(directory/(reconciler['id']+'_reconciliation.json'),
                        {'observed_at':time.time(),'returncode':result.returncode,
                         'stdout':result.stdout[-4000:],'stderr':result.stderr[-2000:],'owner':reconciler['owner']})
            if result.returncode:errors.append({'source':reconciler['id'],'error':'Reconciler failed','owner':reconciler['owner']})
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            errors.append({'source':reconciler['id'],'error':str(exc),'owner':reconciler['owner']})
    if account=='minyeong':
        # Prepared requests live one or two levels below diagnostics. Never
        # descend through native scenes, asset trees or recorded trajectories.
        paths=itertools.chain((root/'diagnostics').glob('*/prepared.json'),
                              (root/'diagnostics').glob('*/*/prepared.json'))
        for path in paths:
            try:
                value=native_ticket(root,path,policy)
                if value:tickets.append(value)
            except (OSError,ValueError,KeyError,TypeError) as exc:
                errors.append({'source':str(path),'error':str(exc),'owner':'/root'})
    for reference in policy.get('prepared_manifests',[]):
        try:
            verify_ref(reference)
            for ticket in read(reference['path'])['tickets']:
                if ticket['expected_account']==account:tickets.append(ticket)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            errors.append({'source':reference['path'],'error':str(exc),'owner':'/root'})
    capacity=capacity_probe()
    allowance=min(capacity.get('compatible_available_slots',0),capacity.get('account_slots',{}).get(account,0)) if capacity.get('status')=='observed' else 0
    critical={'pipeline','canonical recovery'}
    pending=set(map(str,capacity.get('pending_job_ids',[])));known_pending=set()
    for ticket in tickets:
        if ticket.get('workstream') in critical and Path(ticket['submission_path']).exists():
            jid=str(read(ticket['submission_path']).get('job_id'))
            if jid in pending:known_pending.add(jid)
    queue_allowance=max(0,int(policy.get('critical_pending_limit',6))-len(known_pending)) if capacity.get('status')=='observed' else 0
    rows=[]
    # New camera/layout work must not sit behind hundreds of earlier queries
    # for one model family. Existing claim receipts still prevent duplicate work.
    for ticket in dispatch_order(tickets):
        try:
            state=pending_status(ticket)
            slots=int(ticket.get('gpu_slots',1));is_critical=ticket.get('workstream') in critical
            if state=='ready' and (allowance>=slots or is_critical and queue_allowance>=slots):
                result=submit(ticket,capacity)
                if result:
                    allowance=max(0,allowance-slots);state=result['status']
                    if is_critical:queue_allowance=max(0,queue_allowance-slots)
            rows.append({'id':ticket['id'],'workstream':ticket['workstream'],'owner':ticket['owner'],
                         'class_id':ticket.get('class_id'),'state':state,'ready_since':ticket.get('ready_since'),
                         'submission_path':ticket['submission_path'],'claim_path':ticket['claim_path'],
                         'blocker':'No currently available account/physical slots' if state=='ready' and allowance<=0 else None})
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
            rows.append({'id':ticket['id'],'state':'needs_reconciliation','owner':ticket['owner'],'error':str(exc)})
    report={'schema':'isg-coordinator-dispatch-status-v1','campaign_id':root.name,'owner':'/root',
            'account':account,'pid':os.getpid(),'observed_at':time.time(),'last_reconciliation_at':time.time(),
            'cadence_seconds':30,'capacity':{k:v for k,v in capacity.items() if k!='probe'},'requests':rows,'errors':errors,
            'production_released':False,'automatic_retry_ambiguous_submission':False}
    shared_json(directory/('status_'+account+'.json'),report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--account',choices=['minyeong','shiyuc'],required=True);parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if getpass.getuser()!=args.account:raise ValueError('Account identity differs')
    directory=args.root/'infra/coordinator_dispatch';directory.mkdir(parents=True,exist_ok=True)
    with (directory/(args.account+'.worker.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                result=cycle(args.root,args.account)
                print(json.dumps({'observed_at':result['observed_at'],'requests':len(result['requests']),'errors':len(result['errors'])}),flush=True)
            except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
                shared_json(directory/('error_'+args.account+'.json'),{'observed_at':time.time(),'error':str(exc),'owner':'/root'})
            if args.once:break
            time.sleep(30)


if __name__=='__main__':main()
