#!/usr/bin/env python3
"""Durable watcher: submit one true open-loop gate per successful canary job."""
import argparse,csv,datetime,hashlib,json,os,pathlib,re,subprocess,time

def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.new');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def read(path,default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def apply_retry_request(entry, request, root, active):
    """Explicitly retry a terminal launcher failure without replacing its evidence."""
    if not request or request.get('request_id') in entry.get('applied_retry_requests', []):return False
    token=request.get('request_id','')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',token):raise ValueError('unsafe retry request id')
    if entry['status'] not in ('replay_missing','submit_failed'):return False
    if entry.get('replay_job_id') in active:return False
    if request.get('expected_replay_job_id') != entry.get('replay_job_id'):raise ValueError('retry replay identity differs')
    if not request.get('reason'):raise ValueError('retry requires a recorded diagnosis')
    canonical=root/'validation/open_loop'/entry['key']/entry['source_job_id']
    if (canonical/'gate.json').exists():raise ValueError('retry cannot replace an existing canonical gate')
    output=canonical/('attempt_'+token)
    if output.exists():raise ValueError('retry output already exists; reconcile before another submission')
    history={k:v for k,v in entry.items() if k not in ('attempt_history','applied_retry_requests')}
    entry.setdefault('attempt_history',[]).append(history)
    entry.setdefault('applied_retry_requests',[]).append(token)
    entry.update(status='waiting_for_demo',retry_output=str(output),retry_request=request)
    for key in ('replay_job_id','seen_queue_absent','error','gate'):entry.pop(key,None)
    return True

def publish_recovered_gate(entry, root):
    """Publish only a successful new attempt at an as-yet-unwritten canonical path."""
    if not entry.get('retry_request') or not entry.get('gate',{}).get('success'):return
    source=pathlib.Path(entry['output'])/'gate.json'
    canonical=root/'validation/open_loop'/entry['key']/entry['source_job_id']/'gate.json'
    body=source.read_bytes()
    if canonical.exists():
        if canonical.read_bytes()!=body:raise ValueError('canonical gate already contains different evidence')
        return
    proof={'schema':'isg-recovered-replay-gate-v1','request':entry['retry_request'],
           'attempt_gate':str(source),'attempt_gate_sha256':hashlib.sha256(body).hexdigest(),
           'canonical_gate':str(canonical),'replay_job_id':entry['replay_job_id'],
           'prior_attempts':entry.get('attempt_history',[])}
    evidence=canonical.parent/'recovered_gate_provenance.json'
    if evidence.exists() and read(evidence,None)!=proof:raise ValueError('recovered gate provenance already differs')
    if not evidence.exists():atomic(evidence,proof)
    staged=canonical.with_suffix('.json.recovery_new')
    with staged.open('wb') as handle:
        handle.write(body);handle.flush();os.fsync(handle.fileno())
    os.link(staged,canonical)  # Atomic, and never replaces existing evidence.
    staged.unlink()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--snapshot',required=True);ap.add_argument('--interval',type=float,default=30);ap.add_argument('--once',action='store_true');ap.add_argument('--ledger',type=pathlib.Path,action='append');ap.add_argument('--scene-helper',type=pathlib.Path);a=ap.parse_args()
    if a.scene_helper:
        a.scene_helper=a.scene_helper.resolve()
        if not a.scene_helper.is_relative_to(a.root/'code') or not a.scene_helper.is_file():raise ValueError('replay scene helper must be frozen inside this campaign')
    statepath=a.root/'validation/controller.json';state=read(statepath,{'schema':1,'gates':{}})
    while True:
        ledgers=a.ledger or [a.root/'manifests'/name for name in ('canary_allocations.tsv','vast_canary_allocations.tsv','goal_group_allocations.tsv','augmentation_allocations.tsv','shiyuc_allocations.tsv')]
        rows=[]
        for ledger in ledgers:
            if ledger.exists():
                with ledger.open() as handle:rows.extend(csv.DictReader(handle,delimiter='\t'))
        queue=subprocess.run(['squeue','-u','minyeong,shiyuc','-h','-o','%i|%T'],capture_output=True,text=True,timeout=30)
        if queue.returncode:time.sleep(a.interval);continue
        active=dict(x.split('|') for x in queue.stdout.splitlines() if '|' in x)
        retry_requests=read(a.root/'validation/replay_retry_requests.json',{}).get('requests',{})
        for row in rows:
            jid=row['jobid'];job=pathlib.Path(row['job_dir']);entry=state['gates'].setdefault(jid,{'source_job_id':jid,'key':row['key'],'source_job_dir':str(job),'status':'waiting_for_demo'})
            try:
                if apply_retry_request(entry,retry_requests.get(jid),a.root,active):atomic(statepath,state)
            except ValueError as error:
                entry['retry_request_error']=str(error)
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
                    out=pathlib.Path(entry.pop('retry_output')) if entry.get('retry_output') else a.root/'validation/open_loop'/row['key']/jid
                    out.mkdir(parents=True,exist_ok=True)
                    entry.update(status='submitting',output=str(out));atomic(statepath,state)
                    cmd=['sbatch','--parsable','-A','vision','-p','svl','--exclude=svl9','--gres=gpu:titanrtx:1','-c','8','--mem=32G','--time=0-00:40:00','-J','isg261110_replay','--comment','campaign_20261110_replay_'+jid,'-o',str(a.root/'slurm/%x_%j.log'),str(a.root/'tools/infra_open_loop.sh'),str(job),row.get('code') or a.snapshot,str(out)]
                    if a.scene_helper:cmd[1:1]=['--export=ALL,ISGF_REPLAY_SCENE_HELPER='+str(a.scene_helper)]
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
                    publish_recovered_gate(entry,a.root)
                elif entry['replay_job_id'] not in active and entry['status']=='replay_queued':
                    # sbatch can complete between queue snapshot and this point; wait one full tick before concluding absent.
                    if entry.get('seen_queue_absent'):entry['status']='replay_missing'
                    else:entry['seen_queue_absent']=True
                else:entry.pop('seen_queue_absent',None)
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(statepath,state)
        if a.once:break
        time.sleep(max(10,a.interval))
if __name__=='__main__':main()
