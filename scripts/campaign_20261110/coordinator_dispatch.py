"""Root-owned automatic intake for new bounded recovery/preparation requests.

Existing production/replay/acceptance dispatchers keep their queues. This worker
uses each prepared request's existing claim and receipt, never retries ambiguous
submissions, and never grants production or acceptance credit.
"""
import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.campaign_20261110.infra_vast.cluster_return_capacity import probe
from scripts.campaign_20261110.infra_dispatch import shared_json


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


def submit(ticket, capacity, runner=subprocess.run):
    if pending_status(ticket) != 'ready':
        return None
    for path,digest in ticket['inputs'].items():
        verify_ref({'path':path,'sha256':digest})
    command=ticket['command']
    if not command or command[0]!='sbatch' or '--parsable' not in command:
        raise ValueError('Only exact sbatch argv is supported')
    claim=Path(ticket['claim_path']);claim.parent.mkdir(parents=True,exist_ok=True)
    record={**ticket,'status':'submitting','observed_at':time.time(),'dispatch_owner':'/root',
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
        for path in (root/'diagnostics').glob('**/prepared.json'):
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
    rows=[]
    # Oldest ready request first; all streams get access, no final-production gate
    # is used to block these separately authorized preparation and recovery jobs.
    for ticket in sorted(tickets,key=lambda row:row.get('ready_since',0)):
        try:
            state=pending_status(ticket)
            if state=='ready' and allowance>=int(ticket.get('gpu_slots',1)):
                result=submit(ticket,capacity)
                if result:
                    allowance-=int(ticket.get('gpu_slots',1));state=result['status']
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
