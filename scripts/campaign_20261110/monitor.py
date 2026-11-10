#!/usr/bin/env python3
"""Read-only campaign monitor. Writes atomic local artifacts, never submits jobs."""
import argparse
import datetime as dt
import json
import pathlib
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent

def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default

def atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.new')
    temporary.write_text(content)
    temporary.replace(path)

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def refresh(output, host):
    infra=output/'infra'
    error=None
    command=['python3','-'] if host=='local' else ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',host,'python3','-']
    try:
        result=subprocess.run(command,input=(HERE/'capacity_probe.py').read_text(),text=True,capture_output=True,timeout=75)
        if result.returncode:
            raise RuntimeError('SSH capacity probe failed (rc=%s)'%result.returncode)
        cluster=json.loads(result.stdout)
        atomic(infra/'cluster_snapshot.json',json.dumps(cluster,indent=2)+'\n')
    except (OSError,ValueError,subprocess.TimeoutExpired,RuntimeError) as exc:
        error=str(exc)
        cluster=read_json(infra/'cluster_snapshot.json',{})
    runtime=read_json(infra/'runtime.json',{})
    validation=cluster.get('validation',{})
    gates=list(validation.get('gates',{}).values())
    campaign_ids={str(j['job_id']) for j in runtime.get('jobs',[]) if j.get('job_id')}
    campaign_ids.update(str(g[k]) for g in gates for k in ('source_job_id','replay_job_id') if g.get(k))
    jobs=cluster.get('jobs',[])
    campaign=[j for j in jobs if j['id'] in campaign_ids]
    unrelated=[j for j in jobs if j['id'] not in campaign_ids]
    vast=read_json(infra/'vast_snapshot.json',{})
    fleet=[i for a in vast.get('accounts',[]) for i in a.get('instances',[])]
    own_vast_ids={str(i) for i in runtime.get('vast_instance_ids',[])}
    own_fleet=[i for i in fleet if str(i.get('id')) in own_vast_ids]
    unknown_accounts=[a.get('account') for a in vast.get('accounts',[]) if a.get('error_type') or 'instances' not in a]
    def gpu_count(rows):
        return sum(j['gpus'] for j in rows if j['state']=='RUNNING')
    def fleet_gpus(rows):
        return sum(int(i.get('num_gpus') or 0) for i in rows if i.get('actual_status')=='running')
    state={
        'updated_at':now(),'cluster_observed_at':cluster.get('checked_at'),
        'cluster_probe_error':error,'vast_observed_at':vast.get('checked_at'),
        'phase':runtime.get('phase','Infrastructure and contract preparation'),
        'campaign_id':'campaign_20261110','baseline':'9949b012',
        'campaign_cluster_running_gpus':gpu_count(campaign),
        'campaign_cluster_pending_jobs':sum(j['state']=='PENDING' for j in campaign),
        'unrelated_cluster_running_gpus':gpu_count(unrelated),
        'cluster_by_user':cluster.get('summary',{}),'cluster_capacity':cluster.get('capacity',{}),
        'campaign_vast_running_gpus':fleet_gpus(own_fleet) if not unknown_accounts else None,
        'all_vast_running_gpus':fleet_gpus(fleet) if not unknown_accounts else None,
        'vast_accounts_unavailable':unknown_accounts,
        'campaign_submitted_jobs':runtime.get('jobs',[]),
        'baseline_validation':{'demo_passed':sum(g.get('demo_successes',0)>0 for g in gates),'open_loop_passed':sum(g.get('status')=='validated' for g in gates),'gates':gates},
        'campaign_results':runtime.get('results',{'templates_demo_validated':0,'templates_replay_validated':0,'fresh_standard_eval_instances':0,'fresh_model_holdout_instances':0,'fresh_category_holdout_instances':0,'fresh_train_instances':0}),
        'pending':runtime.get('pending',['Freeze split/perturbation contracts','Launch fresh template demo and open-loop replay validation','Apply owner updates for five pending aliases']),
        'new_campaign_vast_budget_usd':runtime.get('vast_budget_usd',300),
        'automatically_stop_at_deadline':False,
    }
    atomic(output/'live.json',json.dumps(state,indent=2)+'\n')
    lines=['# Fresh dataset campaign — live state','',f"Updated: {state['updated_at']} (UTC). Cluster observed: {state['cluster_observed_at']}.",'',f"Current phase: **{state['phase']}**.",'','| GPU usage | Count |','|---|---:|',f"| This campaign: cluster running | {state['campaign_cluster_running_gpus']} |",f"| This campaign: cluster pending jobs | {state['campaign_cluster_pending_jobs']} |",f"| Other work: cluster running | {state['unrelated_cluster_running_gpus']} |",f"| This campaign: Vast running | {state['campaign_vast_running_gpus'] if state['campaign_vast_running_gpus'] is not None else 'not verified'} |",f"| All accounts: Vast running | {state['all_vast_running_gpus'] if state['all_vast_running_gpus'] is not None else 'not verified'} |",'',f"Vast observation: {state['vast_observed_at'] or 'not yet checked'}. New campaign cap: ${state['new_campaign_vast_budget_usd']}; prior campaign resources are separate.",'','| Fresh campaign result | Count |','|---|---:|']
    lines.extend([f"| Baseline canaries: demo passed | {state['baseline_validation']['demo_passed']} |",f"| Baseline canaries: open-loop passed | {state['baseline_validation']['open_loop_passed']} |"])
    lines.extend(f"| {k.replace('_',' ')} | {v} |" for k,v in state['campaign_results'].items())
    lines.extend(['','Cluster QOS limits: 48 GPUs per user on svl; 30 per user on viscam. Capacity below is a point-in-time resource estimate, not a reservation.','',f"Observed capacity: `{json.dumps(state['cluster_capacity'],sort_keys=True)}`.",'','Pending work:',''])
    lines.extend('- '+x for x in state['pending'])
    if error:lines.extend(['',f'**Refresh failed:** {error}. GPU numbers retain the prior observation; check its timestamp.'])
    lines.extend(['','No automatic deadline shutdown. Counts include only recorded campaign evidence; earlier datasets are not counted.',''])
    atomic(output/'LIVE.md','\n'.join(lines))
    print(json.dumps({k:state[k] for k in ('updated_at','campaign_cluster_running_gpus','unrelated_cluster_running_gpus','cluster_probe_error')}),flush=True)
    return state

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--host',default='minyeong@svl11')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=float,default=45)
    args=parser.parse_args()
    while True:
        refresh(args.output,args.host)
        if not args.watch:break
        time.sleep(max(15,args.interval))
if __name__=='__main__':main()
