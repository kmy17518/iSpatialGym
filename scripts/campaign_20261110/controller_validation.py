#!/usr/bin/env python3
"""Durable watcher: submit one true open-loop gate per successful canary job."""
import argparse,csv,datetime,json,pathlib,subprocess,time

def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.new');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def read(path,default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--snapshot',required=True);ap.add_argument('--interval',type=float,default=30);ap.add_argument('--once',action='store_true');ap.add_argument('--ledger',type=pathlib.Path,action='append');a=ap.parse_args()
    statepath=a.root/'validation/controller.json';state=read(statepath,{'schema':1,'gates':{}})
    while True:
        ledgers=a.ledger or [a.root/'manifests'/name for name in ('canary_allocations.tsv','vast_canary_allocations.tsv','goal_group_allocations.tsv','augmentation_allocations.tsv')]
        rows=[]
        for ledger in ledgers:
            if ledger.exists():
                with ledger.open() as handle:rows.extend(csv.DictReader(handle,delimiter='\t'))
        queue=subprocess.run(['squeue','-u','minyeong,shiyuc','-h','-o','%i|%T'],capture_output=True,text=True,timeout=30)
        if queue.returncode:time.sleep(a.interval);continue
        active=dict(x.split('|') for x in queue.stdout.splitlines() if '|' in x)
        for row in rows:
            jid=row['jobid'];job=pathlib.Path(row['job_dir']);entry=state['gates'].setdefault(jid,{'source_job_id':jid,'key':row['key'],'source_job_dir':str(job),'status':'waiting_for_demo'})
            source_results=[]
            try:
                for line in (job/'results.jsonl').read_text().splitlines():
                    try:source_results.append(json.loads(line))
                    except ValueError:pass
            except OSError:pass
            successes=[x for x in source_results if x.get('success') is True]
            entry['demo_successes']=len(successes)
            entry['demo_failures']=sum(x.get('success') is False for x in source_results)
            if successes and entry['status'] in ('demo_missing','demo_failed'):
                entry['status']='waiting_for_demo'
            if entry['status']=='waiting_for_demo':
                if successes:
                    out=a.root/'validation/open_loop'/row['key']/jid
                    out.mkdir(parents=True,exist_ok=True)
                    entry.update(status='submitting',output=str(out));atomic(statepath,state)
                    cmd=['sbatch','--parsable','-A','vision','-p','svl','--exclude=svl9','--gres=gpu:titanrtx:1','-c','8','--mem=32G','--time=0-00:40:00','-J','isg261110_replay','--comment','campaign_20261110_replay_'+jid,'-o',str(a.root/'slurm/%x_%j.log'),str(a.root/'tools/infra_open_loop.sh'),str(job),row.get('code') or a.snapshot,str(out)]
                    response=subprocess.run(cmd,capture_output=True,text=True,timeout=45)
                    if response.returncode:
                        entry.update(status='submit_failed',error=response.stderr.strip()[:300])
                    else:
                        entry.update(status='replay_queued',replay_job_id=response.stdout.strip().split(';')[0])
                        print('REPLAY_SUBMITTED',jid,entry['replay_job_id'],row['key'],flush=True)
                elif jid not in active:
                    vast_state=read(a.root/'vast/state/jobs'/f'{jid}.json',{})
                    vast_queued=(a.root/'vast/queue'/f'{jid}.json').exists() and vast_state.get('status','queued') in ('queued','assigned')
                    if not vast_queued:entry['status']='demo_failed' if source_results else 'demo_missing'
            if entry.get('replay_job_id'):
                gate=read(pathlib.Path(entry['output'])/'gate.json',None)
                if gate:
                    entry.update(status='validated' if gate.get('success') else 'replay_failed',gate=gate)
                elif entry['replay_job_id'] not in active and entry['status']=='replay_queued':
                    # sbatch can complete between queue snapshot and this point; wait one full tick before concluding absent.
                    if entry.get('seen_queue_absent'):entry['status']='replay_missing'
                    else:entry['seen_queue_absent']=True
                else:entry.pop('seen_queue_absent',None)
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(statepath,state)
        if a.once:break
        time.sleep(max(10,a.interval))
if __name__=='__main__':main()
