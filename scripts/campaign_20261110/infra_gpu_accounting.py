"""Read-only per-job GPU time, native attempts and collection outcomes."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from scripts.campaign_20261110.infra_vast.fleet_accounting import snapshot as fleet_snapshot
from scripts.campaign_20261110.backend_evidence import vast_job_ids


def read(path):
    return json.loads(Path(path).read_text())


def gpu_count(tres):
    values=dict(item.split('=',1) for item in tres.split(',') if '=' in item)
    if 'gres/gpu' in values:return int(values['gres/gpu'])
    counts=[int(value) for key,value in values.items() if key.startswith('gres/gpu:')]
    return sum(counts) if counts else 0


def native_evidence(job):
    result={'job_stats':None,'demo_reports':0,'demo_successes':0,'first_demo_started_unix':None,
            'last_demo_completed_unix':None,'errors':[]}
    stats=job/'job_stats.json'
    if stats.exists():
        try:
            raw=stats.read_bytes();data=json.loads(raw)
            result['job_stats']={'path':str(stats),'sha256':hashlib.sha256(raw).hexdigest(),
                **{key:data.get(key) for key in ('attempts','written','world_load_s','sample_wall_s','rejections')}}
        except (OSError,ValueError) as error:result['errors'].append(str(error))
    for p in (job/'demo').glob('*/meta.json'):
        try:
            data=read(p);result['demo_reports']+=1
            result['demo_successes']+=int(data.get('success') is True and data.get('status')=='success')
            if data.get('started_utc'):
                start=datetime.datetime.fromisoformat(data['started_utc'].replace('Z','+00:00')).timestamp()
                result['first_demo_started_unix']=min(start,result['first_demo_started_unix'] or start)
                end=start+float(data.get('wall_time_s') or 0)
                result['last_demo_completed_unix']=max(end,result['last_demo_completed_unix'] or end)
        except (OSError,ValueError,TypeError) as error:result['errors'].append(str(error))
    return result


def attempt_accounting(root, jid, state):
    attempts=[]
    for path in sorted((root/'vast/state/attempts'/jid).glob('*.json')):
        data=read(path);done=data['done'];start=done.get('fetch_started_unix');end=done.get('upload_finished_unix')
        count=done.get('gpu_count')
        attempts.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'attempt_id':data['attempt_id'],'try_number':data.get('try_number'),'instance':data.get('instance'),
                         'native_attempts':done.get('attempts'),'written':done.get('written'),'demo_successes':done.get('successes'),
                         'rc':done.get('rc'),'fetch_started_unix':start,'upload_finished_unix':end,'gpu_count':count,
                         'complete_worker_gpu_seconds':((end-start)*count if start is not None and end is not None and count is not None else None)})
    complete=(state['status'] in ('synced','failed') and len(attempts)==int(state.get('tries') or 0)
              and bool(attempts) and all(x['complete_worker_gpu_seconds'] is not None for x in attempts))
    return {'retained_attempts':attempts,'all_attempt_intervals_complete':complete,
            'complete_worker_gpu_seconds_all_attempts':sum(x['complete_worker_gpu_seconds'] for x in attempts) if complete else None,
            'retained_native_attempts_all_tries':sum(x['native_attempts'] for x in attempts)
                if complete and all(x['native_attempts'] is not None for x in attempts) else None}


def snapshot(root):
    started=time.time();rows={};ledger_errors=[]
    for path in sorted((root/'manifests').glob('*allocations.tsv')):
        try:
            with path.open() as stream:
                for row in csv.DictReader(stream,delimiter='\t'):
                    jid=row.get('jobid','')
                    if not re.fullmatch(r'\d+(?:_\d+)?',jid):continue
                    if jid in rows and rows[jid]['job_dir']!=row['job_dir']:
                        ledger_errors.append({'ledger':str(path),'job_id':jid,'error':'conflicting job directory'});continue
                    rows[jid]=row
        except (OSError,ValueError) as error:ledger_errors.append({'ledger':str(path),'error':str(error)})
    vast_ids=vast_job_ids(root)
    cluster=[jid for jid in rows if jid not in vast_ids];accounting={};probes=[]
    for first in range(0,len(cluster),150):
        ids=cluster[first:first+150]
        command=['sacct','-X','-n','-P','-j',','.join(ids),
                 '--format=JobIDRaw,User,Account,Partition,State,Submit,Start,End,ElapsedRaw,AllocTRES,ExitCode']
        env=dict(os.environ,TZ='UTC',SLURM_TIME_FORMAT='standard')
        try:
            probe=subprocess.run(command,capture_output=True,text=True,timeout=60,env=env,check=True)
            probes.append({'ids':ids,'completed_at_unix':time.time(),'stdout_sha256':hashlib.sha256(probe.stdout.encode()).hexdigest()})
            for line in probe.stdout.splitlines():
                fields=line.split('|')
                if len(fields)>=11 and fields[0] in ids:
                    accounting[fields[0]]=dict(zip(('job_id','user','account','partition','state','submitted_utc','start_utc','end_utc','elapsed_seconds','allocated_tres','exit_code'),fields))
        except (subprocess.SubprocessError,OSError) as error:probes.append({'ids':ids,'error':str(error),'failed_at_unix':time.time()})
    records=[]
    for jid,row in rows.items():
        job=Path(row['job_dir']);record={'job_id':jid,'job_dir':str(job),'key':row.get('key'),
            'tag':row.get('tag'),'index':int(row['start']),'requested_candidates':int(row['n']),
            'code':row.get('code'),'native':native_evidence(job),'observed_at_unix':time.time()}
        if jid in vast_ids:
            state_path=root/'vast/state/jobs'/f'{jid}.json'
            try:
                state=read(state_path);done=state.get('done') or {};elapsed=done.get('elapsed_s');fetch=done.get('fetch_s')
                record.update(backend='vast',state=state['status'],gpu_count=1,instance=state.get('instance'),
                    gpu=done.get('gpu') or state.get('gpu_actual'),assigned_unix=state.get('assigned'),
                    runtime_start_unix=done.get('start'),runtime_end_unix=done.get('end'),
                    native_elapsed_seconds=elapsed,input_fetch_seconds=fetch,
                    native_gpu_seconds=elapsed,worker_gpu_seconds_including_fetch=(elapsed+fetch if elapsed is not None and fetch is not None else None),
                    assignment_observed_seconds=(max(0,(done.get('end') or time.time())-state['assigned']) if state.get('assigned') else None),
                    attempts_from_done=done.get('attempts'),written_from_done=done.get('written'),demo_successes_from_done=done.get('successes'),
                    tries=state.get('tries'),attempt_history=state.get('history',[]),source_state=str(state_path))
                record['missing_time_components']=['rental_boot_and_idle_not_attributable_to_one_job','post_run_upload_end_not_recorded']
                if int(state.get('tries') or 0)>1:record['missing_time_components'].append('prior_retry_exact_start_end_not_retained')
                if not done:record['missing_time_components'].append('current_worker_native_start_end_not_yet_synced')
                record.update(attempt_accounting(root,jid,state))
                if record['all_attempt_intervals_complete']:
                    record['missing_time_components']=['rental_boot_and_idle_not_attributable_to_one_job']
            except (OSError,ValueError) as error:record.update(backend='vast',error=str(error))
        else:
            data=accounting.get(jid)
            if data:
                count=gpu_count(data['allocated_tres']);elapsed=int(data['elapsed_seconds'])
                record.update(backend='slurm',**{k:v for k,v in data.items() if k!='job_id'},gpu_count=count,
                              allocated_gpu_seconds=count*elapsed,
                              includes_startup_sampling_collection_failures=True)
            else:record.update(backend='slurm',error='sacct returned no exact allocation row')
        records.append(record)
    return {'schema':'isg-job-gpu-accounting-v1','started_at_unix':started,'observed_at_unix':time.time(),
        'duration_seconds':time.time()-started,'records':records,'sacct_probes':probes,'ledger_errors':ledger_errors,
        'scope':'allocation ledgers only; excludes unledgered dependency/bootstrap jobs; accepted counts must be joined by immutable artifact identity',
        'vast_rental_ledger':str(root/'vast/ledger/events.jsonl'), 'vast_fleet':fleet_snapshot(root),
        'warning':'Native successes are not accepted instances. Running elapsed and live artifact counts are snapshots; no estimated attempt denominators.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--interval',type=int,default=0);args=parser.parse_args()
    destination=args.root/'accounting/job_gpu_time.json';destination.parent.mkdir(exist_ok=True)
    while True:
        value=snapshot(args.root);temporary=destination.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(destination)
        print(json.dumps({'records':len(value['records']),'duration_seconds':value['duration_seconds'],'output':str(destination)}),flush=True)
        if not args.interval:return
        time.sleep(max(60,args.interval))


if __name__=='__main__':main()
