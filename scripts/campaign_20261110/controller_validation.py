#!/usr/bin/env python3
"""Durable watcher: submit one true open-loop gate per successful canary job."""
import argparse,csv,datetime,fcntl,hashlib,json,os,pathlib,re,subprocess,time

def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.new');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def read(path,default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def ledger_paths(root, explicit=None):
    base=explicit or [root/'manifests'/name for name in ('canary_allocations.tsv','vast_canary_allocations.tsv',
        'goal_group_allocations.tsv','augmentation_allocations.tsv','shiyuc_allocations.tsv')]
    candidates=list(base)+sorted((root/'manifests').glob('dispatch_*_allocations.tsv'))
    seen=set();result=[]
    for path in candidates:
        resolved=path.resolve()
        if not resolved.is_relative_to(root.resolve()/'manifests'):
            raise ValueError('validation ledger outside owned campaign manifests')
        if resolved not in seen:
            seen.add(resolved);result.append(path)
    return result

def archive_diagnosed_launcher_gate(entry, request, root, active):
    """Preserve explicitly diagnosed launcher evidence before a new strict replay.

    This never promotes the old successful simulation result: a new complete
    action replay and scene-identity verification must still pass.
    """
    folder=root/'validation/open_loop'/entry['key']/entry['source_job_id']
    canonical=folder/'gate.json'
    archived=folder/('failed_gate_before_'+request['request_id']+'.json')
    source=canonical if canonical.exists() else archived
    body=source.read_bytes()
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if hashlib.sha256(body).hexdigest()!=request.get('expected_failed_gate_sha256'):
        raise ValueError('failed gate hash differs from diagnosed evidence')
    gate=json.loads(body);result=gate.get('result',{})
    if not (gate.get('success') is False and gate.get('replay_rc')==5 and gate.get('replay_mode')=='physics_action_open_loop'):
        raise ValueError('failed gate is not the diagnosed launcher failure')
    if request['failure_kind']=='mutable_source_metadata':
        if not (result.get('success') is True and result.get('qa_hard')==[] and result.get('goal_override') is None):
            raise ValueError('failed gate is not the diagnosed post-replay metadata failure')
        log=folder/'scene_verify.log'
        if sha(log)!=request.get('scene_verify_log_sha256') or 'source template metadata changed' not in log.read_text():
            raise ValueError('metadata race diagnosis changed')
    else:
        log=folder/'replay.log';foreign=request.get('foreign_cache_path','')
        expected='/vision/u/shiyuc/minyeong/campaign_20261110/isaacsim_overlay/apps/omnigibson_5_1_0.kit'
        if not (request['failure_kind']=='foreign_replay_cache' and foreign==expected
                and result.get('status')=='error' and result.get('success') is False and result.get('exit_code')==4
                and 'PermissionError' in result.get('error','') and result.get('wall_time_s',float('inf'))<5
                and not result.get('n_action_steps') and sha(log)==request.get('replay_log_sha256')
                and foreign in log.read_text() and '_launch_app' in log.read_text()):
            raise ValueError('foreign-cache retry is not a proved pre-simulation launcher failure')
    job=pathlib.Path(entry['source_job_dir']).resolve()
    if not job.is_relative_to(root/'out') or pathlib.Path(gate['source_job']).resolve()!=job:
        raise ValueError('metadata retry source job differs')
    vast=read(root/'vast/state/jobs'/(entry['source_job_id']+'.json'),{})
    if entry['source_job_id'] in active or not (job/'sampling.done').is_file() or (vast and vast.get('status')!='synced'):
        raise ValueError('metadata retry requires closed source sampling and collection')
    metadata=pathlib.Path(request['source_metadata']).resolve()
    if not metadata.is_relative_to(job) or sha(metadata)!=request.get('source_metadata_sha256'):
        raise ValueError('closed source metadata changed')
    audit=pathlib.Path(request['preflight_audit']).resolve()
    if not audit.is_relative_to(root) or sha(audit)!=request.get('preflight_audit_sha256'):
        raise ValueError('strict replay preflight audit changed')
    if archived.exists():
        if archived.read_bytes()!=body:raise ValueError('archived failed gate differs')
    else:
        os.link(canonical,archived)
    # The failed bytes retain their immutable archive. After an interruption the
    # same exact request reconciles that archive; it never authorizes two jobs.
    if canonical.exists():canonical.unlink()
    return {'path':str(archived),'sha256':hashlib.sha256(body).hexdigest()}

def apply_retry_request(entry, request, root, active):
    """Explicitly retry a terminal launcher failure without replacing its evidence."""
    if not request or request.get('request_id') in entry.get('applied_retry_requests', []):return False
    token=request.get('request_id','')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',token):raise ValueError('unsafe retry request id')
    diagnosed_retry=entry['status']=='replay_failed' and request.get('failure_kind') in ('mutable_source_metadata','foreign_replay_cache')
    if entry['status'] not in ('replay_missing','submit_failed') and not diagnosed_retry:return False
    if entry.get('replay_job_id') in active:return False
    if request.get('expected_replay_job_id') != entry.get('replay_job_id'):raise ValueError('retry replay identity differs')
    if not request.get('reason'):raise ValueError('retry requires a recorded diagnosis')
    canonical=root/'validation/open_loop'/entry['key']/entry['source_job_id']
    output=canonical/('attempt_'+token)
    if output.exists():raise ValueError('retry output already exists; reconcile before another submission')
    if diagnosed_retry:
        archived=archive_diagnosed_launcher_gate(entry,request,root,active)
    elif (canonical/'gate.json').exists():
        raise ValueError('retry cannot replace an existing canonical gate')
    else:archived=None
    history={k:v for k,v in entry.items() if k not in ('attempt_history','applied_retry_requests')}
    if archived:history['archived_failed_gate']=archived
    entry.setdefault('attempt_history',[]).append(history)
    entry.setdefault('applied_retry_requests',[]).append(token)
    entry.update(status='waiting_for_demo',retry_output=str(output),retry_request=request)
    for key in ('replay_job_id','seen_queue_absent','error','gate','vast_migration','replay_backend','vast_migration_error'):entry.pop(key,None)
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
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=pathlib.Path,required=True);ap.add_argument('--snapshot',required=True);ap.add_argument('--interval',type=float,default=30);ap.add_argument('--once',action='store_true');ap.add_argument('--ledger',type=pathlib.Path,action='append');ap.add_argument('--scene-helper',type=pathlib.Path);ap.add_argument('--dual-partition-replays',action='store_true');ap.add_argument('--vast-pending-seconds',type=float);ap.add_argument('--vast-helper-snapshot',type=pathlib.Path);a=ap.parse_args()
    a.root=a.root.resolve()
    if a.vast_pending_seconds is not None:
        if a.vast_pending_seconds<300 or not a.vast_helper_snapshot:raise ValueError('Vast replay migration requires a frozen helper and >=300 seconds pending')
        a.vast_helper_snapshot=a.vast_helper_snapshot.resolve()
        if not a.vast_helper_snapshot.is_relative_to(a.root/'code'):raise ValueError('Vast replay helper outside campaign')
    lease=(a.root/'validation/controller.service.lock').open('a')
    fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.scene_helper:
        a.scene_helper=a.scene_helper.resolve()
        if not a.scene_helper.is_relative_to(a.root/'code') or not a.scene_helper.is_file():raise ValueError('replay scene helper must be frozen inside this campaign')
    statepath=a.root/'validation/controller.json';state=read(statepath,{'schema':1,'gates':{}})
    while True:
        ledgers=ledger_paths(a.root,a.ledger)
        rows=[]
        for ledger in ledgers:
            if ledger.exists():
                with ledger.open() as handle:rows.extend(csv.DictReader(handle,delimiter='\t'))
        queue=subprocess.run(['squeue','-u','minyeong,shiyuc','-h','-o','%i|%T'],capture_output=True,text=True,timeout=30)
        if queue.returncode:time.sleep(a.interval);continue
        active=dict(x.split('|') for x in queue.stdout.splitlines() if '|' in x)
        replay_active=dict(active)
        for path in (a.root/'vast/state/jobs').glob('9*.json'):
            value=read(path,{})
            if value.get('job_kind')=='physics_replay' and value.get('status') in ('queued','assigned'):
                replay_active[str(value['id'])]=value['status']
        migration_candidates=[]
        for source_id,entry in state['gates'].items():
            phase=entry.get('vast_migration',{}).get('status')
            if entry.get('status')=='replay_queued' and (active.get(entry.get('replay_job_id'))=='PENDING'
                    or phase in ('prepared','cancel_requested','cancelled_never_started','ready_to_activate')):
                migration_candidates.append((entry.get('vast_migration_last_checked',0),source_id))
        migration_selected={source_id for _,source_id in sorted(migration_candidates)[:3]}
        resources=['-A','vision','-p','svl','--exclude=svl9','--gres=gpu:titanrtx:1']
        if a.dual_partition_replays:
            try:
                from scripts.campaign_20261110.replay_scheduling import dual_partition_resources
                inventory=subprocess.check_output(['scontrol','show','nodes','-o'],text=True,timeout=30)
                resources,proof=dual_partition_resources(inventory)
                state['replay_resource_policy']={**proof,'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
                state.pop('replay_resource_probe_error',None)
            except (ValueError,subprocess.SubprocessError) as error:
                resources=None
                state['replay_resource_probe_error']={'reason':str(error),'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        retry_requests=read(a.root/'validation/replay_retry_requests.json',{}).get('requests',{})
        for row in rows:
            jid=row['jobid'];job=pathlib.Path(row['job_dir']);entry=state['gates'].setdefault(jid,{'source_job_id':jid,'key':row['key'],'source_job_dir':str(job),'status':'waiting_for_demo'})
            try:
                if apply_retry_request(entry,retry_requests.get(jid),a.root,replay_active):atomic(statepath,state)
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
                    # Sampling may rewrite template metadata after the first
                    # demo finishes. Replaying before its durable close marker
                    # can invalidate scene identity mid-replay.
                    if not (job/'sampling.done').is_file():
                        entry['waiting_reason']='source_sampling_not_closed'
                        continue
                    if resources is None:
                        entry['waiting_reason']='replay_resource_inventory_unavailable'
                        continue
                    entry.pop('waiting_reason',None)
                    out=pathlib.Path(entry.pop('retry_output')) if entry.get('retry_output') else a.root/'validation/open_loop'/row['key']/jid
                    out.mkdir(parents=True,exist_ok=True)
                    entry.update(status='submitting',output=str(out));atomic(statepath,state)
                    cmd=['sbatch','--parsable',*resources,'-c','8','--mem=32G','--time=0-00:40:00','-J','isg261110_replay','--comment','campaign_20261110_replay_'+jid,'-o',str(a.root/'slurm/%x_%j.log'),str(a.root/'tools/infra_open_loop.sh'),str(job),row.get('code') or a.snapshot,str(out)]
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
                intent=entry.get('vast_migration',{})
                candidate=(active.get(entry['replay_job_id'])=='PENDING' or intent.get('status') in
                           ('prepared','cancel_requested','cancelled_never_started','ready_to_activate'))
                if a.vast_pending_seconds is not None and jid in migration_selected and entry['status']=='replay_queued' and candidate:
                    entry['vast_migration_last_checked']=time.time()
                    try:
                        from scripts.campaign_20261110.replay_migration import migrate
                        moved=migrate(entry,a.root,row.get('code') or a.snapshot,a.vast_helper_snapshot,
                                      a.vast_pending_seconds,lambda:atomic(statepath,state))
                        if moved:
                            replay_active[entry['replay_job_id']]='queued'
                            print('REPLAY_MIGRATED',jid,entry['replay_job_id'],row['key'],flush=True)
                        entry.pop('vast_migration_error',None)
                    except (ValueError,OSError,subprocess.SubprocessError) as error:
                        entry['vast_migration_error']={'at':time.time(),'reason':str(error)}
                gate=read(pathlib.Path(entry['output'])/'gate.json',None)
                if gate:
                    entry.update(status='validated' if gate.get('success') else 'replay_failed',gate=gate)
                    publish_recovered_gate(entry,a.root)
                elif entry['replay_job_id'] not in replay_active and entry['status']=='replay_queued':
                    # sbatch can complete between queue snapshot and this point; wait one full tick before concluding absent.
                    if entry.get('seen_queue_absent'):entry['status']='replay_missing'
                    else:entry['seen_queue_absent']=True
                else:entry.pop('seen_queue_absent',None)
        state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(statepath,state)
        if a.once:break
        time.sleep(max(10,a.interval))
if __name__=='__main__':main()
