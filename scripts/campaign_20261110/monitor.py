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
    baseline_gates=[gate for gate in gates if '/baseline_canary/' in gate.get('source_job_dir','')]
    contract_gates=[gate for gate in gates if '/baseline_canary/' not in gate.get('source_job_dir','') and '/vast_arch_canary/' not in gate.get('source_job_dir','')]
    vast_gates=[gate for gate in gates if '/vast_arch_canary/' in gate.get('source_job_dir','')]
    campaign_ids={str(j['job_id']) for j in runtime.get('jobs',[]) if j.get('job_id')}
    campaign_ids.update(str(g[k]) for g in gates for k in ('source_job_id','replay_job_id') if g.get(k))
    jobs=cluster.get('jobs',[])
    campaign=[j for j in jobs if j['id'] in campaign_ids or j['name'].startswith('isg261110_')]
    unrelated=[j for j in jobs if j not in campaign]
    vast=read_json(infra/'vast_snapshot.json',{})
    fleet=[i for a in vast.get('accounts',[]) for i in a.get('instances',[])]
    own_vast_ids={str(i) for i in runtime.get('vast_instance_ids',[])}
    own_fleet=[i for i in fleet if str(i.get('id')) in own_vast_ids]
    own_vast=cluster.get('vast_campaign',{})
    campaign_vast_observed_at=(dt.datetime.fromtimestamp(own_vast['observed_at_unix'],dt.timezone.utc).isoformat()
                               if own_vast.get('observed_at_unix') else None)
    unknown_accounts=[a.get('account') for a in vast.get('accounts',[]) if a.get('error_type') or 'instances' not in a]
    try: vast_age=time.time()-dt.datetime.fromisoformat(vast['checked_at']).timestamp()
    except (KeyError,ValueError): vast_age=None
    def gpu_count(rows):
        return sum(j['gpus'] for j in rows if j['state']=='RUNNING')
    def fleet_gpus(rows):
        return sum(int(i.get('num_gpus') or 0) for i in rows if i.get('actual_status')=='running')
    accepted=cluster.get('acceptance',{})
    results=dict(runtime.get('results',{'templates_demo_validated':0,'templates_replay_validated':0,'fresh_standard_eval_instances':0,'fresh_model_holdout_instances':0,'fresh_category_holdout_instances':0,'fresh_train_instances':0}))
    if accepted.get('sources'):
        results.pop('templates_demo_validated',None)
        results.pop('templates_replay_validated',None)
        results['templates_with_accepted_instances']=accepted['templates']
        for split, field in [('eval_standard','fresh_standard_eval_instances'),('eval_model','fresh_model_holdout_instances'),('eval_category','fresh_category_holdout_instances'),('train','fresh_train_instances')]:
            results[field]=accepted.get('counts',{}).get(split,0)
    state={
        'updated_at':now(),'cluster_observed_at':cluster.get('checked_at'),
        'cluster_probe_error':error,'vast_observed_at':campaign_vast_observed_at,
        'all_vast_observed_at':vast.get('checked_at'),
        'phase':runtime.get('phase','Infrastructure and contract preparation'),
        'campaign_id':'campaign_20261110','baseline':'9949b012',
        'campaign_cluster_running_gpus':gpu_count(campaign),
        'campaign_cluster_pending_jobs':sum(j['state']=='PENDING' for j in campaign),
        'campaign_cluster_pending_gpus':sum(j['gpus'] for j in campaign if j['state']=='PENDING'),
        'unrelated_cluster_running_gpus':gpu_count(unrelated),
        'cluster_by_user':cluster.get('summary',{}),'cluster_capacity':cluster.get('capacity',{}),
        'cluster_by_user_partition':cluster.get('by_user_partition',{}),
        'campaign_vast_running_gpus':own_vast.get('running_gpus') if own_vast else (fleet_gpus(own_fleet) if not unknown_accounts else None),
        'campaign_vast_metrics':own_vast,
        'all_vast_running_gpus':fleet_gpus(fleet) if not unknown_accounts and vast_age is not None and vast_age < 120 else None,
        'all_vast_observation_age_seconds':vast_age,
        'scale_prebuild':cluster.get('scale_prebuild',{}),
        'data_releases':cluster.get('data_releases',[]),
        'vast_accounts_unavailable':unknown_accounts,
        'campaign_submitted_jobs':runtime.get('jobs',[]),
        'baseline_validation':{'demo_passed':sum(g.get('demo_successes',0)>0 for g in baseline_gates),'open_loop_passed':sum(g.get('status')=='validated' for g in baseline_gates),'gates':baseline_gates},
        'contract_validation':{'demo_passed':sum(g.get('demo_successes',0)>0 for g in contract_gates),'open_loop_passed':sum(g.get('status')=='validated' for g in contract_gates),'gates':contract_gates},
        'vast_arch_validation':{'demo_passed':sum(g.get('demo_successes',0)>0 for g in vast_gates),'open_loop_passed':sum(g.get('status')=='validated' for g in vast_gates),'gates':vast_gates},
        'campaign_results':results,'acceptance_evidence':accepted,'annotated_coverage':cluster.get('annotated_coverage',{}),
        'family_controller':cluster.get('family_controller',{}),'throughput':cluster.get('throughput',{}),
        'pending':runtime.get('pending',['Freeze split/perturbation contracts','Launch fresh template demo and open-loop replay validation','Apply owner updates for five pending aliases']),
        'new_campaign_vast_budget_usd':runtime.get('vast_budget_usd',3000),
        'automatically_stop_at_deadline':False,
    }
    atomic(output/'live.json',json.dumps(state,indent=2)+'\n')
    lines=['# Fresh dataset campaign — live state','',f"Updated: {state['updated_at']} (UTC). Cluster observed: {state['cluster_observed_at']}.",'',f"Current phase: **{state['phase']}**.",'','| GPU usage | Count |','|---|---:|',f"| This campaign: cluster running | {state['campaign_cluster_running_gpus']} |",f"| This campaign: cluster pending jobs | {state['campaign_cluster_pending_jobs']} |",f"| Other work: cluster running | {state['unrelated_cluster_running_gpus']} |",f"| This campaign: Vast running | {state['campaign_vast_running_gpus'] if state['campaign_vast_running_gpus'] is not None else 'not verified'} |",f"| All accounts: Vast running | {state['all_vast_running_gpus'] if state['all_vast_running_gpus'] is not None else 'not verified'} |",'',f"Vast observation: {state['vast_observed_at'] or 'not yet checked'}. New campaign cap: ${state['new_campaign_vast_budget_usd']}; prior campaign resources are separate.",'','| Fresh campaign result | Count |','|---|---:|']
    lines.extend([f"| Baseline canaries: demo passed | {state['baseline_validation']['demo_passed']} |",f"| Baseline canaries: open-loop passed | {state['baseline_validation']['open_loop_passed']} |"])
    if state['annotated_coverage']:
        lines.extend([f"| Annotated templates: demo passed | {state['annotated_coverage']['demo_passed']} |",f"| Annotated templates: true replay passed | {state['annotated_coverage']['replay_passed']} |"])
    for release in state['data_releases']:
        lines.extend([f"| Released exact-scale grasp libraries ({pathlib.Path(release['path']).name}) | {release.get('new_scale_libraries',0)} |",f"| Released exact-scale arm maps ({pathlib.Path(release['path']).name}) | {release.get('new_scale_maps',0)} |"])
        if release.get('new_floating_libraries'):
            lines.append(f"| Fresh positive floating libraries ({pathlib.Path(release['path']).name}) | {release['new_floating_libraries']} |")
    lines.extend(f"| {k.replace('_',' ')} | {v} |" for k,v in state['campaign_results'].items())
    if state['throughput']:
        forecast=state['throughput']
        lines.extend(['',f"Forecast observed: {forecast.get('observed_at')}. Remaining throughput required: {forecast.get('required_instances_per_hour',0):.1f} accepted instances/hour."])
        for row in forecast.get('collection_jobs',[]):
            lines.append(f"- {row['key']}: {row['successes']} successful demos / {row['attempted_demos']} completed demos; {row['successes_per_gpu_hour_with_startup']:.1f} successes/GPU-hour including startup. These measured lanes do not establish throughput for other families.")
    if own_vast:
        lines.extend(['',f"Campaign Vast ledger estimate: ${own_vast.get('spend_estimate_usd',0):.2f} spent + ${own_vast.get('future_compute_commitment_usd',0):.2f} reserved compute + ${own_vast.get('transfer_reserve_usd',0):.2f} transfer/storage reserve. Provider invoice reconciliation pending. Fleet observed at Unix {own_vast.get('observed_at_unix')}."])
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
