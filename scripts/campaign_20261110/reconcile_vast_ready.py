"""Root intake for bounded native prep; the existing Vast controller owns rentals.

Both backends use the source's physical claim. This process prepares only a
bounded queue buffer, never transfers submitted cluster work or retries a claim.
An activated job removed from the allowlist is held, not silently reactivated.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

from scripts.campaign_20261110 import prepare_floating_vast as adapter
from scripts.campaign_20261110 import vast_native_dependency as native
from scripts.campaign_20261110.infra_dispatch import shared_json
from scripts.campaign_20261110.production_gate import check_job
from scripts.campaign_20261110.storage import atomic_directory_lock


def intake_slots(metrics, active_states, *, timestamp, minimum_buffer=128):
    if (not metrics.get('account_listing_complete') or
            not 0 <= timestamp-metrics.get('observed_at_unix',0) <= 180):
        return 0
    live=sum(int(i.get('num_gpus') or 1) for i in metrics.get('instances',[])
             if i.get('actual_status') not in ('exited','stopped'))
    queued=sum(s.get('status')=='queued' for s in active_states)
    # Grow with observed rentals; this is a queue buffer, not a fleet ceiling.
    return max(0,max(minimum_buffer,3*live)-queued)


def activation_candidates(submitted, configured, previously_activated):
    configured=set(map(str,configured));previously_activated=set(map(str,previously_activated))
    return [r for r in submitted if str(r['job_id']) not in configured|previously_activated]


def cycle(root, policy_path, adapter_module=adapter):
    root=Path(root);policy=json.loads(Path(policy_path).read_text())
    assert policy['owner']=='/root' and policy['production_released'] is False
    assert policy['budget_cap_usd']==3000
    directory=root/'infra/coordinator_dispatch';config_path=root/'vast/config.json'
    metrics=json.loads((root/'vast/metrics.json').read_text())
    config=json.loads(config_path.read_text());active=set(map(str,config.get('dispatch_job_ids',[])))
    states=[]
    for jid in active:
        p=root/'vast/state'/(jid+'.json')
        if p.exists():states.append(json.loads(p.read_text()))
        elif (root/'vast/queue'/(jid+'.json')).exists():states.append({'status':'queued'})
    remaining=intake_slots(metrics,states,timestamp=time.time())
    state_path=directory/'vast_ready_intake_state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'activated':[],'cursor':0}
    results=[];errors=[];submitted=[];began=time.time()
    sources=[s for s in policy['sources'] if s.get('enabled') is True]
    if sources:
        offset=state.get('cursor',0)%len(sources);sources=sources[offset:]+sources[:offset]
    for source in sources:
        try:
            if remaining>0 and time.time()-began<70:
                count=min(16,remaining)
                result=adapter_module.run(root,source['manifest']['path'],source['manifest']['sha256'],
                    source['helper'],leave_for_cluster=0,transport_name=source['transport_name'],max_new_jobs=count)
                remaining-=result['newly_submitted'];results.append({'source':source['id'],'added':result['newly_submitted']})
            progress=Path(source['progress'])
            if progress.exists():
                value=json.loads(progress.read_text())
                assert value['source']==source['manifest']
                submitted.extend(value['submitted'])
        except Exception as exc:errors.append({'source':source['id'],'error':type(exc).__name__+': '+str(exc)})
    # Persist activation intent under the same lock as all config writers.
    # Prior successful activations removed by another owner remain held.
    with atomic_directory_lock(root/'vast/config.dispatch.lockdir'):
        config=json.loads(config_path.read_text());assert config['budget_usd']==3000
        configured=set(map(str,config.get('dispatch_job_ids',[])))
        known=set(map(str,state['activated']))|{str(r['job_id'])for r in submitted if str(r['job_id'])in configured}
        candidates=activation_candidates(submitted,configured,known);verified=[]
        for row in candidates:
            try:
                rec=json.loads(Path(row['receipt']).read_text());job=Path(row['job_dir'])
                claim=Path(row['receipt']).parent.parent/'claims'/(rec['physical_dedup_key']+'.json')
                owner=json.loads(claim.read_text())
                assert rec['status']=='submitted' and rec['backend']=='vast' and rec['job_id']==row['job_id']
                assert owner['source']==rec['source'] and owner['job_dir']==rec['job_dir']
                assert native.sha(job/'native_inputs.json')==rec['native_inputs_sha256']
                gate=check_job(root,job,kind='native_dependency');assert gate['allowed'],gate
                queue=json.loads((root/'vast/queue'/(str(row['job_id'])+'.json')).read_text())
                assert queue['native_inputs_sha256']==rec['native_inputs_sha256']
                verified.append(str(row['job_id']))
            except Exception as exc:errors.append({'job_id':row['job_id'],'error':str(exc)})
        if verified:
            shared_json(directory/'vast_ready_activation_intent.json',{'observed_at':time.time(),'job_ids':verified})
            config['dispatch_job_ids']=list(dict.fromkeys(list(config.get('dispatch_job_ids',[]))+verified))
            config['dispatch_priority_job_ids']=list(dict.fromkeys(list(config.get('dispatch_priority_job_ids',[]))+verified))
            shared_json(config_path,config)
        state.update(activated=sorted(known|set(verified)),cursor=state.get('cursor',0)+1,observed_at=time.time())
        shared_json(state_path,state)
    report={'owner':'/root','observed_at':time.time(),'elapsed_seconds':time.time()-began,
            'results':results,'newly_activated':verified,'errors':errors,'remaining_buffer_slots':remaining,
            'rental_owner':'existing Vast controller','benchmark_credit':0,'production_released':False}
    shared_json(directory/'vast_ready_intake.json',report)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--policy',type=Path,required=True);args=parser.parse_args()
    with (args.root/'infra/coordinator_dispatch/vast_ready_intake.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        print(json.dumps(cycle(args.root,args.policy)),flush=True)


if __name__=='__main__':main()
