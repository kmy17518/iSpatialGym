"""Read-only observation of the frozen 100-class suite; no execution authority."""
from collections import Counter
from datetime import datetime,timezone
import argparse,hashlib,json,time
from pathlib import Path


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError):return default


def reconciliation_summary(base):
    path=base/'accepted_reconciliation_inventory_r1/summary_r2.json'
    value=read(path)
    if value is None:return None
    # Keep the observation compact; the full inventory retains every exact row.
    result={k:value[k] for k in ('schema','observed_at','accepted_capture_started_at','accepted_capture_finished_at','counts','records','snapshot','no_retrospective_clearance_from_current100','regeneration_launched') if k in value}
    impact=value.get('impact',{})
    result['impact']={k:v for k,v in impact.items() if k in ('observed_at','pending_change_count','summary','snapshot','unchanged_from_start_audit')}
    result['eligibility_observed_at']=value.get('eligibility',{}).get('observed_at')
    result['source']={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def report(root):
    root=Path(root);pointer=read(root/'runtime_validation/current.json',{});release=read(pointer.get('release',{}).get('path',''),{})
    base=Path(pointer.get('release',{}).get('path','')).parent;manifest=read(pointer.get('manifest',{}).get('path',''),{})
    metrics=read(base/'measurements/current.json',{});rows=[]
    measured={r['class_id']:r for r in metrics.get('classes',[])}
    observer=read(base/'acceptance_observer.json',{})
    for entry in manifest.get('classes',[]):
        cls=entry['class_id'];catalog=read(base/'catalog_rows'/(cls+'.json'),{})
        for folder in ['custom_providers_r1','camera_bddl_r1']:
            resolved=read(base/'input_resolutions'/folder/(cls+'.json'))
            if resolved:catalog=resolved
        launched=read(base/'rows'/(cls+'.json'));dispatch=read(base/'dispatch'/(cls+'.json'));allocation=read(base/'allocations'/(cls+'.json'));accepted=read(base/'acceptance'/(cls+'.json'))
        authored=read(base/'authoring'/(cls+'.json'));attempt=read(base/'authoring_attempts_r1'/(cls+'.json'));failure=read(base/'dispatch'/(cls+'.failure.json'))
        for folder in ['authoring_attempts_camera_r1','authoring_attempts_opening_r2']:
            attempt=read(base/folder/(cls+'.json')) or attempt
        state='inputs_resolved' if catalog.get('state')=='inputs_resolved' else 'blocked_named_input'
        next_action=catalog.get('next_action','Resolve exact inputs');blocker=catalog.get('missing_inputs',[]);jobs=[]
        if attempt and attempt.get('returncode') not in (None,0):state='failed_authoring';blocker=[attempt.get('log')];next_action=attempt['next_action']
        if authored:state='ready_unsubmitted';next_action='Complete command preflight and instrumented dispatch'
        if failure and not dispatch:state='failed_preflight';blocker=[failure.get('log')];next_action=failure['next_action']
        if dispatch:state=dispatch.get('status','queued');next_action='Reconcile backend allocation and native outcome'
        exception=read(base/'dispatch'/(cls+'.exception.json'))
        if exception and (not dispatch or dispatch.get('status')=='ready_unsubmitted'):
            state='failed_intake';blocker=[exception.get('error')];next_action=exception.get('next_action','Inspect exact input/dispatch failure before retry.')
        receipt=read(dispatch.get('dispatch_receipt',''),{}) if dispatch else {}
        if receipt.get('status') in ('blocked','failed'):
            state='failed_dispatch';blocker=[receipt.get('error') or receipt.get('reason') or receipt.get('status')]
            next_action='Compute owner must inspect the exact dry/submit evidence and reconcile any allocation before bounded repair.'
        if receipt.get('allocation'):
            a=receipt['allocation'];jobs=[{'job_id':a.get('jobid'),'backend':receipt.get('backend'),'job_dir':a.get('job_dir')}];state='allocated_execution_unverified'
        if allocation:
            jobs=allocation.get('records',jobs)
            state=allocation.get('execution_state',state);next_action=allocation.get('next_action',next_action)
            if state in ('COMPLETED','completed','synced'):
                state='collection_terminal_awaiting_checks'
        audit=observer.get('classes',{}).get(cls,{})
        audit_allocation=read(audit.get('allocation',{}).get('path',''),{})
        same_allocation=bool(allocation and str(audit_allocation.get('allocation',{}).get('jobid'))==str(allocation.get('allocation',{}).get('jobid')))
        if same_allocation and audit.get('status') in ('native_started','awaiting_demo','awaiting_distinct_demos','native_failed','demo_failed','insufficient_successful_demos'):
            state=audit['status']
            if state in ('native_failed','demo_failed','insufficient_successful_demos'):
                next_action='Root diagnoses actual native/demo evidence before a bounded registered retry; no unchanged automatic resubmission.'
                blocker=audit.get('failed_demo_report_paths',[])
        if same_allocation and (audit.get('status') in ('audit_blocked','replay_failed','awaiting_member_replay','awaiting_terminal_evidence','awaiting_acceptance_lock') or (
                audit.get('status')=='awaiting_own_replay' and measured.get(cls,{}).get('distinct_demo_instance_ids'))):
            state=audit['status'];next_action=audit.get('next_action','Dedicated replay and acceptance owners reconcile this exact allocation; inspect failed checks before retry.')
            blocker=[audit.get('error') or audit.get('reason')] if audit.get('error') or audit.get('reason') else []
        if accepted:
            state='validation_passed' if accepted.get('validation_passed')is True else accepted.get('status','awaiting_acceptance')
            next_action=accepted.get('next_action','Retain exact result; reconcile genuine remaining quota separately')
        measurement=measured.get(cls)
        if measurement and not allocation and measurement.get('jobs'):
            jobs=measurement['jobs']
        owner='/root/templates_assets' if state=='blocked_named_input' else '/root/dataset_contract' if state in ('audit_blocked','awaiting_acceptance_lock','awaiting_terminal_evidence') else '/root/compute' if state in ('failed_dispatch','external_submission','allocated_execution_unverified','replay_failed','awaiting_member_replay','awaiting_own_replay') else '/root'
        rows.append({'class_id':cls,'state':state,'owner':owner,
                     'camera_exempt':catalog.get('camera_exempt'), 'runtime':release.get('frozen_runtime',{}).get('commit'),
                     'fresh_indices':None if not launched else [launched['fresh_start'],launched['fresh_count']],
                     'jobs':jobs,'blocker':blocker,'next_action':next_action,'acceptance':accepted,
                     'catalog_observed_at':catalog.get('observed_at'),'launch_created_at':(launched or{}).get('created_at'),
                     'last_execution_observed_at':(allocation or{}).get('observed_at'),
                     'measurements':measurement,'dispatch_receipt':dispatch.get('dispatch_receipt') if dispatch else None,
                     'stage_observation':audit,
                     'row_path':str(base/'rows'/(cls+'.json')) if launched else None})
    states=dict(Counter(r['state']for r in rows));summary=manifest.get('summary',{})
    return {'schema':'isg-current-runtime-suite-observation-v1','campaign_id':'campaign_20261110',
            'observed_at':datetime.now(timezone.utc).isoformat(),'run_id':release.get('run_id'),'release':pointer.get('release'),
            'frozen_runtime':release.get('frozen_runtime'),'historical_accepted_classes':summary.get('historical_validated'),
            'historical_runtime_identities':summary.get('historical_runtime_identity_count'),
            'required_classes':100,'validation_passed':states.get('validation_passed',0),'state_counts':states,'rows':rows,
            'metrics':metrics,'intake_worker':read(base/'dispatch_intake_worker.json'),
            'existing_sample_reconciliation':reconciliation_summary(base),
            'acceptance_observer':observer,
            'impact_observed_at':summary.get('impact_ledger_observed_at'),'impact_summary':summary.get('impact_summary'),
            'regeneration_decision':{'status':'authorized_conditionally_gates_and_measurements_pending','reason':'Full current-runtime suite and measured per-configuration replacement scope are incomplete; no defensible two-hour full-regeneration estimate.'},
            'observer_authority':'Read existing records only; does not dispatch, cancel, accept, or release.'}


def worker(root, value):
    root=Path(root)
    path=root/'runtime_validation/status.json'
    jobs=[job for row in value['rows'] for job in row['jobs'] if isinstance(job,dict)]
    return {'campaign_id':root.name,'observed_at':value['observed_at'],'owner':'/root',
            'status':'current_runtime_validation_in_progress','active_jobs':jobs,
            'engineering_activity':read(root/'runtime_validation/engineering.json',{'state':'unavailable'}),
            'controllers':{'intake':value['intake_worker'],'acceptance':{
                k:v for k,v in value['acceptance_observer'].items() if k!='classes'}},
            'dependencies':[{'required_classes':100,'authored':sum(bool(r['row_path']) for r in value['rows']),
                             'verified':value['validation_passed'],'states':value['state_counts']}],
            'evidence':[value['release'],{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}],
            'next_action':'Advance independent fresh classes through native, demo, each required member replay, and strict acceptance; diagnose failures; keep bulk training held.',
            'next_artifact':'Complete exact frozen-runtime100 acceptance matrix and measured per-class cost table; then assess actual unreconciled scope for two-hour regeneration.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=45);a=p.parse_args()
    while True:
        value=report(a.root);out=a.root/'runtime_validation/status.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(out)
        out=a.root/'runtime_validation/worker.json';tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(worker(a.root,value),indent=2)+'\n');tmp.replace(out)
        if a.once:break
        time.sleep(a.interval)

if __name__=='__main__':main()
