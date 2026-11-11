"""Read-only campaign-wide Slurm GPU accounting, including replay and prerequisites."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from scripts.campaign_20261110.infra_gpu_accounting import gpu_count


FIELDS=('job_id','user','name','comment','state','submitted_utc','start_utc','end_utc',
        'elapsed_seconds','allocated_tres','exit_code','submission')


def parse(text, root, explicit=()):
    root=str(Path(root).resolve());secondary='/vision/u/shiyuc/minyeong/campaign_20261110'
    explicit=set(explicit);records={};ambiguous=[];errors=[]
    for line in text.splitlines():
        values=line.split('|',len(FIELDS)-1)
        if len(values)<len(FIELDS):continue
        values[-1]=values[-1].removesuffix('|')
        row=dict(zip(FIELDS,values));jid=row['job_id']
        if row['user'] not in ('minyeong','shiyuc'):continue
        if '.' in jid:continue  # Allocation rows only, never steps plus their parent.
        owned=(row['comment']=='campaign_20261110' or row['comment'].startswith('campaign_20261110_')
               or root+'/' in row['submission'] or secondary+'/' in row['submission']
               or jid in explicit or jid.split('_')[0] in explicit)
        if not owned:
            if row['name'].startswith('isg261110'):
                ambiguous.append({'job_id':jid,'name':row['name'],'reason':'campaign-shaped name lacks owned comment, submission path or explicit allocation proof'})
            continue
        submission=row.pop('submission');row['submission_sha256']=hashlib.sha256(submission.encode()).hexdigest()
        row['identity_basis']='campaign_comment_or_owned_submission_or_explicit_allocation'
        try:
            row['gpu_count']=gpu_count(row['allocated_tres']);row['elapsed_seconds']=int(row['elapsed_seconds'])
            row['allocated_gpu_seconds']=row['gpu_count']*row['elapsed_seconds']
        except (ValueError,TypeError) as error:
            errors.append({'job_id':jid,'error':str(error)});continue
        if jid in records and records[jid]!=row:errors.append({'job_id':jid,'error':'conflicting exact accounting rows'});continue
        records[jid]=row
    return list(records.values()),ambiguous,errors


def snapshot(root):
    root=Path(root).resolve();started=time.time()
    registry=root/'accounting/additional_allocations.json'
    explicit=json.loads(registry.read_text()).get('job_ids',[]) if registry.exists() else []
    command=['sacct','-X','-n','-P','-S','2026-11-10T00:00:00','-u','minyeong,shiyuc',
             '--format=JobIDRaw,User,JobName%180,Comment%250,State,Submit,Start,End,ElapsedRaw,AllocTRES,ExitCode,SubmitLine%1200']
    probe=subprocess.run(command,capture_output=True,text=True,timeout=60,check=True,
                         env=dict(os.environ,TZ='UTC',SLURM_TIME_FORMAT='standard'))
    rows,ambiguous,errors=parse(probe.stdout,root,explicit)
    return {'schema':'isg-cluster-fleet-gpu-accounting-v1','campaign_id':'campaign_20261110',
            'started_at_unix':started,'observed_at_unix':time.time(),'records':rows,
            'verified_allocated_gpu_seconds':sum(row['allocated_gpu_seconds'] for row in rows),
            'complete_campaign_gpu_seconds':None if ambiguous or errors else sum(row['allocated_gpu_seconds'] for row in rows),
            'ambiguous_campaign_named_allocations':ambiguous,'errors':errors,
            'raw_accounting_sha256':hashlib.sha256(probe.stdout.encode()).hexdigest(),
            'scope':'Exact campaign-owned allocation rows for both accounts since campaign date, including collection, failed starts, retries, native prerequisites and physics replay; CPU-only allocations contribute zero GPU time. Ambiguous names are excluded explicitly.',
            'accepted_instance_rate':None,'warning':'Accepted instances require an exact artifact/time-window join; no accepted count or throughput is inferred here.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--interval',type=int,default=60);args=parser.parse_args()
    output=args.root/'accounting/cluster_fleet_gpu_time.json';output.parent.mkdir(exist_ok=True)
    while True:
        try:
            value=snapshot(args.root);temporary=output.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(output)
            print(json.dumps({'observed_at_unix':value['observed_at_unix'],'allocations':len(value['records']),
                              'gpu_seconds':value['verified_allocated_gpu_seconds'],'ambiguous':len(value['ambiguous_campaign_named_allocations'])}),flush=True)
        except (OSError,ValueError,subprocess.SubprocessError) as error:
            print(type(error).__name__+': '+str(error),flush=True)
        if not args.interval:return
        time.sleep(max(60,args.interval))


if __name__=='__main__':main()
