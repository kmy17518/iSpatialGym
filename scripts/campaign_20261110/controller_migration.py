"""Move owned, never-started pending work back to fresh controller authoring."""
from __future__ import annotations

import argparse
import datetime
import getpass
from pathlib import Path
import shlex
import subprocess
import time

from scripts.campaign_20261110.contracts import digest, require, write_json_atomic
from scripts.campaign_20261110.controller_families import read
from scripts.campaign_20261110.storage import atomic_directory_lock
from scripts.campaign_20261110 import overflow_readiness


def command(argv):
    return subprocess.check_output(argv,text=True,timeout=40).strip()


def scheduler_proof(jobid):
    require(jobid.isdigit(),'invalid migration job ID')
    raw=command(['scontrol','show','job',jobid,'-o'])
    info=dict(item.split('=',1) for item in shlex.split(raw) if '=' in item)
    accounting=command(['sacct','-X','-n','-P','-j',jobid,'--format=JobIDRaw,State,Start,End'])
    rows=[line.split('|') for line in accounting.splitlines() if line.split('|')[0]==jobid]
    require(len(rows)==1,'ambiguous scheduler accounting for migration')
    return {'observed_at':time.time(),'job_id':jobid,'scheduler':info,'accounting':{
        'state':rows[0][1].split()[0].rstrip('+'),'start':rows[0][2],'end':rows[0][3]}}


def no_execution_artifacts(job,root):
    job=Path(job).resolve();require(job.is_relative_to(root/'out'),'migration output outside campaign')
    found=[]
    for pattern in ('inst/**/*-tro_state.json','demo/**/demo.hdf5','**/sample.log','**/sampling.log','**/collect.log','**/collection.log'):
        found += [str(path) for path in job.glob(pattern) if path.is_file() and path.stat().st_size]
    require(not found,'pending job already has native execution artifacts')
    return {'job_dir':str(job),'nonempty_execution_artifacts':found,'checked_at':time.time()}


def eligible(proof,*,owner,minimum_age,now):
    overflow_readiness.pending_delay(minimum_age)
    info=proof['scheduler'];accounting=proof['accounting']
    require(info['UserId'].split('(')[0]==owner,'migration job belongs to another user')
    if info['JobState']!='PENDING' or accounting['state']!='PENDING':return False
    require(accounting['start'] in ('Unknown','None','N/A',''),'pending job has already started')
    submitted=datetime.datetime.fromisoformat(info['SubmitTime']).timestamp()
    return now-submitted>=minimum_age


def confirmed_never_started(proof):
    accounting=proof['accounting']
    return accounting['state']=='CANCELLED' and accounting['start'] in ('Unknown','None','N/A','')


def migrate_record(root,record,*,minimum_age,owner,save,execute=False):
    allocation=record.get('allocation')
    if not allocation or record.get('status')=='complete' or record.get('migration'):return None
    jid=str(allocation['jobid'])
    if (root/'vast/state/jobs'/(jid+'.json')).exists():return None
    before=scheduler_proof(jid)
    recovering=record.get('migration_intent') is not None and confirmed_never_started(before)
    if not recovering and not eligible(before,owner=owner,minimum_age=minimum_age,now=time.time()):return None
    files=no_execution_artifacts(allocation['job_dir'],root)
    proposal=record.get('migration_intent') or {'schema':'isg-never-started-migration-v1','original_allocation':allocation,'before':before,
              'execution_proof':files,'minimum_pending_seconds':minimum_age,'record_sha256_before':digest(record)}
    if not execute:return proposal
    if not recovering:proposal['overflow_readiness']=overflow_readiness.check(root,before['scheduler'])
    # Persist the cancellation intent before acting; an interrupted invocation
    # must reconcile this job, never author a replacement on an assumption.
    record['migration_intent']=proposal;save()
    if not recovering:
        # Holding first prevents a pending job from starting between the final
        # eligibility check and cancellation. If it started first, hold fails
        # and the native job is left alone.
        subprocess.run(['scontrol','hold',jid],check=True,timeout=30)
        held=scheduler_proof(jid)
        require(eligible(held,owner=owner,minimum_age=minimum_age,now=time.time()),'job no longer pending after hold')
        subprocess.run(['scancel',jid],check=True,timeout=30)
    deadline=time.monotonic()+30
    while True:
        after=scheduler_proof(jid)
        if confirmed_never_started(after):break
        require(after['accounting']['start'] in ('Unknown','None','N/A',''),'job started during cancellation; fresh replacement blocked')
        require(time.monotonic()<deadline,'cancellation not yet confirmed; fresh replacement blocked')
        time.sleep(1)
    final_files=no_execution_artifacts(allocation['job_dir'],root)
    record.update(status='complete',migration={**proposal,'after':after,'final_execution_proof':final_files,
        'never_started':True,'replacement_policy':'Fresh group IDs and globally reserved indices; original ledger and input bytes remain immutable.'})
    record.pop('migration_intent',None);save()
    return record['migration']


def controller_check(root,record,state,cfg,queue,save,*,family):
    """A bounded check inside the caller's already-owned controller state lock."""
    age=cfg.get('migrate_pending_seconds')
    if age is None or cfg.get('_migration_checks_remaining',0)<=0 or record.get('status')=='complete':return
    allocation=record.get('allocation') or {};jid=allocation.get('jobid')
    queued=queue.get(jid)
    recovering=bool(record.get('migration_intent'))
    if not recovering and (not queued or queued[1]!='PENDING' or queued[2]!=getpass.getuser()):return
    now=time.time()
    if not recovering and now-record.get('migration_last_checked',0)<60:return
    cfg['_migration_checks_remaining']-=1;record['migration_last_checked']=now
    try:
        with atomic_directory_lock(root/'controllers'/(family+'_planning.lockdir')):
            from scripts.campaign_20261110.migration_routes import source_routes
            routes=source_routes(root,record)
            result=migrate_record(root,record,minimum_age=age,owner=getpass.getuser(),save=save,execute=True)
        if result:
            state['next_dispatch_backends']=routes['backends']
            record['migration_routes']=routes
            record.pop('migration_error',None)
    except Exception as exc:
        record['migration_error']={'at':time.time(),'type':type(exc).__name__,'reason':str(exc)}
    save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--state',action='append',required=True)
    parser.add_argument('--minimum-pending-seconds',type=float,default=0)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--interval',type=float,default=0)
    args=parser.parse_args();root=Path(args.root).resolve()
    overflow_readiness.pending_delay(args.minimum_pending_seconds)
    require(0<=args.interval<=60,'invalid pending migration limits')
    while True:
        for value in args.state:
            path=Path(value).resolve();require(path.is_relative_to(root/'controllers'),'migration state outside campaign controllers')
            with atomic_directory_lock(path.with_suffix('.lockdir')):
                state=read(path);save=lambda:write_json_atomic(path,state)
                for seed in state.get('seeds',{}).values():
                    for family,key in (('held','queues'),('camera','production')):
                        with atomic_directory_lock(root/'controllers'/(family+'_planning.lockdir')):
                            for record in seed.get(key,[]):
                                try:
                                    from scripts.campaign_20261110.migration_routes import source_routes
                                    routes=source_routes(root,record)
                                    result=migrate_record(root,record,minimum_age=args.minimum_pending_seconds,
                                        owner=getpass.getuser(),save=save,execute=args.execute)
                                    if result and args.execute:
                                        seed['next_dispatch_backends']=routes['backends']
                                        record['migration_routes']=routes
                                        seed.pop('error',None);save()
                                    if result:print({'job_id':result['before']['job_id'],'migrated':args.execute},flush=True)
                                except Exception as exc:
                                    record['migration_error']={'at':time.time(),'type':type(exc).__name__,'reason':str(exc)}
                                    if args.execute:save()
        if not args.interval:break
        time.sleep(args.interval)


if __name__=='__main__':main()
