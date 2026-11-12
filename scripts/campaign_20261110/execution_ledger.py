"""Join existing campaign execution records without granting a validation pass.

The ledger is a dated index of exact source observations. Controllers still
verify the referenced original artifacts. HTTP consumers never run this code.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

CAMPAIGN = 'campaign_20261110'
SOURCES = {
    'requirements': 'manifests/training_requirements.json',
    'configurations': 'training_preparation/configuration_index.json',
    'dependencies': 'training_preparation/dependency_index.json',
    'preparation_status': 'training_preparation/status.json',
    'corrected_inventory': 'training_preparation/model_variant_inventory_r2/current.json',
    'canonical': 'infra/standard_recovery/canonical_validation.json',
    'standard': 'infra/standard_recovery/ledger.json',
    'heldout': 'infra/heldout_recovery/ledger.json',
    'pipeline': 'training_pipeline_canaries/pipeline_status.json',
    'queue': 'infra/capacity_recovery/vast_excluded_queue_audit_r3.json',
    'capacity': 'infra/capacity_recovery/ledger.json',
    'eligibility': 'accounting/current_eligibility.json',
    'production_gate': 'controllers/training_production_gate.json',
    'assignments': 'infra/execution_assignments.json',
    'current_runtime': 'runtime_validation/status.json',
}
STAGES = ('implemented', 'submitted', 'native', 'demo', 'physics_replay', 'rendering', 'accepted')


def epoch(value):
    if isinstance(value, (float, int)): return float(value)
    try: return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
    except (AttributeError, ValueError): return None


def read(root, relative, now):
    p = root/relative
    try:
        raw = p.read_bytes(); value = json.loads(raw)
        if value.get('campaign_id') != CAMPAIGN: raise ValueError('campaign identity mismatch')
        observed = value.get('observed_at', value.get('updated_at'))
        age = None if epoch(observed) is None else now-epoch(observed)
        return value, {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(),
            'observed_at': observed, 'age_seconds': age,
            'state': 'timestamp_missing' if age is None else 'stale' if age > 600 else 'current',
            'file_mtime': p.stat().st_mtime}
    except (OSError, ValueError) as exc:
        return {}, {'path': str(p), 'state': 'unavailable', 'error': str(exc), 'observed_at': None, 'sha256': None}


def records(value):
    return list(value.values()) if isinstance(value, dict) else value or []


def stage(value=None):
    return {'status': 'unobserved'} if value is None else {'status': 'source_evidence', 'evidence': value}


def worker_activity(observation, now):
    """A freshly published wrapper cannot refresh an old worker job report."""
    reported = observation.get('active_jobs', [])
    observed = observation.get('worker_observed_at')
    timestamp = epoch(observed)
    age = None if timestamp is None else now - timestamp
    state = ('timestamp_missing' if age is None else 'future_timestamp' if age < -30
             else 'stale' if age > 600 else 'current_worker_report')
    # Recovery workers retain completed source jobs for provenance. In particular,
    # a fresh report of a synced collection artifact is not an active allocation.
    active_states = {'running', 'pending', 'queued', 'submitted', 'configuring',
                     'completing', 'assigned', 'staging', 'native_running',
                     'demo_running', 'replay_running', 'replay_queued'}
    current = [job for job in reported if isinstance(job, dict)
               and str(job.get('status', job.get('scheduler_state', job.get('state', '')))).lower() in active_states]
    return {'active_work': current if state == 'current_worker_report' else [],
            'reported_work': reported,
            'activity_observation': {'observed_at': observed, 'age_seconds': age, 'state': state,
                'nonactive_or_unverified_reports': len(reported) - len(current),
                'semantics': 'Dated worker ownership report with an explicit active phase. Terminal and unknown job states remain reported history; execution requires separate fresh scheduler or worker evidence.'}}


def row(identifier, kind, owner, reference, **fields):
    return {'id': identifier, 'kind': kind, 'owner': owner, 'observed_at': reference.get('observed_at'),
            'evidence_state': reference['state'], 'requirement_ids': [], 'dependencies': [],
            'active_work': [], 'artifacts': [], 'verification': {'status': 'unobserved'},
            'acceptance': {'status': 'not_applicable' if kind in ('requirement', 'dependency', 'pipeline_canary') else 'unobserved'},
            'stages': {s: stage() for s in STAGES}, 'sources': [reference], **fields}


def assignment_observation(root, assignment, reference, now):
    """Read the owner's current report; an old assignment is not a heartbeat."""
    pointer = assignment.get('worker_record')
    if not pointer:
        return assignment, reference, assignment.get('jobs', [])
    path = (root / pointer).resolve()
    if not path.is_relative_to(root):
        return {}, {'path': str(path), 'state': 'unavailable', 'observed_at': None,
                    'sha256': None, 'error': 'worker report outside campaign'}, []
    worker, observed = read(root, str(path.relative_to(root)), now)
    if worker and worker.get('owner') != assignment['owner']:
        return {}, {**observed, 'state': 'unavailable', 'observed_at': None,
                    'error': 'worker owner differs from assignment'}, []
    classes = worker.get('classes', [])
    jobs = worker.get('active_jobs', [job for item in classes for job in item.get('active_jobs', [])])
    # The canonical root worker reports its class inside the existing recovery
    # schema. Preserve the dated outer observation and use its concrete action.
    details = classes[0] if len(classes) == 1 else {}
    return {**details, **worker}, observed, jobs


def build(root, now=None):
    now = time.time() if now is None else now; root = Path(root).resolve()
    observations = {k: read(root, path, now) for k,path in SOURCES.items()}
    values = {k:v[0] for k,v in observations.items()}; refs = {k:v[1] for k,v in observations.items()}
    result = []; requirements = values['requirements'].get('requirements', [])
    def reqs(task):
        return [r['id'] for r in requirements if task in (r.get('tasks') or []) or r.get('scope') == 'all']
    for assignment in values['assignments'].get('workstreams', []):
        worker, observed, jobs = assignment_observation(root, assignment, refs['assignments'], now)
        activity = worker_activity({'worker_observed_at': worker.get('observed_at'),
                                    'active_jobs': jobs}, now)
        result.append(row('workstream:'+assignment['id'], 'workstream', assignment['owner'], observed,
            status=worker.get('status', worker.get('state', assignment.get('status', 'assigned'))), **activity,
            requirement_ids=assignment.get('requirement_ids', []),
            responsibilities=assignment.get('responsibilities', []),
            dependencies=worker.get('dependencies', assignment.get('dependencies', [])),
            artifacts=worker.get('evidence', assignment.get('evidence', [])),
            controller=worker.get('controllers', worker.get('controller', assignment.get('controllers', []))),
            engineering_activity=worker.get('engineering_activity', worker.get('engineering_status', assignment.get('engineering_activity'))),
            next_artifact=worker.get('next_artifact', assignment.get('next_artifact')),
            next_action=worker.get('next_action', assignment.get('next_action')),
            sources=[refs['assignments']] + ([observed] if assignment.get('worker_record') else []),
            exit_criteria=assignment.get('exit_criteria'),
            verification={'status': 'assignment_only', 'verification_requires_original_artifacts': True},
            acceptance={'status': 'not_applicable', 'assignment_does_not_authorize_release': True}))
    for r in requirements:
        chain = r.get('chain') or {}
        result.append(row('requirement:'+r['id'], 'requirement', chain.get('owner', r.get('owners', 'unassigned')),
            refs['requirements'], requirement_ids=[r['id']], requirement=r.get('requirement'), task=r.get('tasks'),
            status=r.get('status', 'untested'), implementation=r.get('implementation'),
            dependencies=chain.get('dependencies', []), active_work=chain.get('active_work', []),
            artifacts=chain.get('artifacts', r.get('evidence', [])), verification=chain.get('verification', {'status':'unobserved'}),
            acceptance=chain.get('acceptance', {'status':'not_applicable'}),
            next_action=r.get('next_action') or 'Requirement owner must attach scoped evidence and a discriminating action.',
            last_progress_at=r.get('last_observed_progress'), exit_criteria=r.get('exit_criteria')))
    manifest = values['configurations']; status = values['preparation_status']
    configs = records(manifest.get('configurations',manifest.get('rows'))); dep_index=values['dependencies']
    deps = records(dep_index.get('artifact_requests',dep_index.get('rows'))) or records(status.get('artifact_requests') or manifest.get('artifact_requests'))
    dependencies = {d['id']:d for d in deps}; served = 0
    for c in configs:
        needed = c.get('dependencies', c.get('dependency_ids',[])); ids = [d.get('id') if isinstance(d,dict) else d for d in needed]
        unresolved = [d for d in ids if d not in dependencies or dependencies[d].get('status') not in ('positively_verified','verified','not_applicable')]
        # A label alone is insufficient: exact physical identity, actual result,
        # and output references must accompany every applicable dependency.
        unresolved += [d for d in ids if d in dependencies and dependencies[d].get('status') in ('verified','positively_verified')
                       and not ((dependencies[d].get('physical_artifact_id') or dependencies[d].get('artifact')) and dependencies[d].get('evidence'))]
        explicit_na = c.get('status') == 'not_applicable' and bool(c.get('evidence'))
        complete = not unresolved and (bool(ids) or explicit_na)
        served += int(complete)
    recovery = {r['class_id']:r for r in values['standard'].get('classes',[])}
    for c in values['canonical'].get('classes',[]):
        r = recovery.get(c['class_id'], {}); observed = r or c
        stages = {s:stage() for s in STAGES}
        for name,key in [('native','native'),('demo','demo'),('physics_replay','replay'),('accepted','acceptance')]:
            stages[name] = stage(observed.get(key))
        result.append(row('standard:'+c['class_id'], 'standard_eval', observed.get('owner','/root'), refs['standard'] if r else refs['canonical'],
            class_id=c['class_id'], task=c.get('task'), requirement_ids=reqs(c.get('task')),
            configuration=observed.get('configuration'), runtime=observed.get('runtime'),
            status=observed.get('state', c.get('status','unobserved')), stages=stages,
            dependencies=observed.get('dependencies',[]), **worker_activity(observed, now),
            controller=observed.get('controller'), acceptance=observed.get('acceptance',c.get('acceptance',{})),
            verification=c, last_progress_at=observed.get('last_progress_at'), blocker=observed.get('blocker'),
            next_action=observed.get('next_action') or 'Reconcile exact accepted quota and current refill owner.'))
    # The current frozen suite is a separate proof obligation from historical
    # canonical acceptance. Index its exact attempts without replacing history
    # or interpreting replay/submission as an accepted benchmark instance.
    current = values['current_runtime']
    scheduler = {str(j.get('id')):j for account in values['capacity'].get('accounts',{}).values()
                 for j in account.get('jobs',[]) if j.get('id') is not None}
    for i,r in enumerate(current.get('rows',[])):
        observed = r.get('stage_observation') or {}
        accepted = r.get('acceptance') or {}
        jobs = [{**j, 'status':scheduler[str(j['job_id'])]['state'],
                 'scheduler_observation':refs['capacity']} if not j.get('never_started') and
                str(j.get('job_id')) in scheduler else j for j in r.get('jobs',[])]
        activity = worker_activity({'worker_observed_at': r.get('last_execution_observed_at'),
                                    'active_jobs': [j for j in jobs if 'scheduler_observation' not in j]}, now)
        scheduled = worker_activity({'worker_observed_at':refs['capacity'].get('observed_at'),
                                     'active_jobs':[j for j in jobs if 'scheduler_observation' in j]}, now)
        activity['active_work'] += scheduled['active_work']
        activity['reported_work'] = jobs
        activity['scheduler_activity_observation'] = scheduled['activity_observation']
        stages = {s:stage() for s in STAGES}
        if r.get('dispatch_receipt'): stages['submitted']=stage(r['dispatch_receipt'])
        if 'native_written_observed' in observed:
            stages['native']=stage({'written_observed':observed['native_written_observed'],
                                    'observed_at':observed.get('observed_at')})
        if observed.get('successful_demo_reports'):
            stages['demo']=stage(observed['successful_demo_reports'])
        own_replays=[m['member_replay'] for m in accepted.get('members',[]) if m.get('member_replay')]
        if own_replays: stages['physics_replay']=stage(own_replays)
        quota=accepted.get('quota',{})
        claimed=accepted.get('claim_result',{}).get('accepted',{})
        acceptance={'status':accepted.get('status','unobserved'),
                    'accepted_instances':claimed,'quota_consumed':quota.get('consumed'),
                    'regression_is_not_benchmark_acceptance':True}
        if accepted: stages['accepted']=stage(acceptance)
        result.append(row(f"runtime:{current.get('run_id')}:{r.get('class_id',i)}",
            'current_runtime_validation',r.get('owner','/root'),refs['current_runtime'],
            class_id=r.get('class_id'),runtime=r.get('runtime'),run_id=current.get('run_id'),
            status=r.get('state'),requirement_ids=['G14','G15'],
            row_path=r.get('row_path'),attempt_number=r.get('attempt_number'),
            dependencies=r.get('attempt_rows',[]),fresh_indices=r.get('fresh_indices'),
            **activity,stages=stages,acceptance=acceptance,
            verification={'status':accepted.get('status','unverified'),
                          'validation_passed':accepted.get('validation_passed') is True,
                          'strict_review':accepted.get('strict_review'),
                          'source_observation_only':True},
            artifacts=[v for v in (accepted.get('strict_review'),accepted.get('regression_audit')) if v],
            blocker=r.get('blocker'),diagnosis=r.get('diagnosis'),
            last_progress_at=r.get('last_execution_observed_at'),
            next_action=r.get('next_action') or 'Reconcile exact current-runtime evidence; preserve historical results separately.'))
    for i,r in enumerate(values['heldout'].get('rows',[])):
        stages = {s:stage() for s in STAGES}
        for name,key in [('native','native'),('demo','demo'),('physics_replay','replay')]: stages[name] = stage(r.get(key))
        stages['accepted'] = stage({'count':r.get('accepted'),'remaining':r.get('remaining'),'last_accepted_at':r.get('last_accepted_at')})
        result.append(row(f"heldout:{r.get('split')}:{r.get('class_id')}:{r.get('bucket',i)}", 'heldout_eval', r.get('owner','/root/templates_assets'), refs['heldout'],
            class_id=r.get('class_id'), task=r.get('task'), split=r.get('split'), bucket=r.get('bucket'), requirement_ids=reqs(r.get('task')),
            status=r.get('state'), configuration=r.get('configuration'), runtime=r.get('runtime'), stages=stages,
            active_work=r.get('active_jobs',[]), controller=r.get('controller'), last_progress_at=r.get('last_progress_at'),
            verification=r.get('flat_progress_investigation'), acceptance=stages['accepted']['evidence'], blocker=r.get('blocker'),
            next_action=r.get('next_action') or 'Reconcile this held-out bucket with current artifacts and assign the next action.'))
    for i,r in enumerate(records(values['pipeline'].get('records'))):
        submission = r.get('native_submission')
        native = r.get('native')
        native_terminal = isinstance(native, dict) and native.get('returncode') is not None
        # A native submission stays in stage history after its terminal gate.
        # Only the separately owned downstream allocation may still be active.
        work = [v for v in (r.get('allocation'), None if native_terminal else submission)
                if isinstance(v, dict) and v]
        result.append(row(f"pipeline:{r.get('candidate_id')}:{r.get('start',i)}", 'pipeline_canary', '/root/dataset_contract', refs['pipeline'],
            configuration_id=r.get('candidate_id'), class_id=r.get('class_id'), status=r.get('status'), runtime=r.get('native_runtime'),
            configuration_sha256=r.get('configuration_sha256'), start=r.get('start'), count=r.get('count'), dependencies=r.get('dependencies',[]), active_work=work,
            artifacts=[v for v in (r.get('request'),r.get('native'),r.get('review')) if v],
            stages={**{s:stage() for s in STAGES}, 'submitted':stage(submission),
                    'native':stage(r.get('native')), 'accepted':{'status':'excluded_zero_credit'}},
            next_action=r.get('next_action') or 'Owner reconciles completed stages into own physics replay, hard QA and batched restoration; no benchmark credit.'))
    counts = values['eligibility'].get('counts',{}).get('current_eligible')
    corrected = values['corrected_inventory'].get('summary')
    gaps = status.get('scope_gaps', manifest.get('scope_gaps', ['Full selected configuration expansion has not been observed.']))
    return {'schema':'isg-campaign-execution-ledger-v1','campaign_id':CAMPAIGN,
        'observed_at':datetime.fromtimestamp(now,timezone.utc).isoformat(),'owner':'/root',
        'sources':refs,'rows':result,'indexed_sources':{'configurations':refs['configurations'],'dependencies':refs['dependencies']},
        'requirements_by_task':{t:reqs(t) for t in sorted({c.get('task') for c in configs if c.get('task')})},'accepted':{'counts':counts,'source':refs['eligibility']},
        'canonical':{k:values['canonical'].get(k) for k in ('validated','total','state_counts')},
        'current_runtime':{k:current.get(k) for k in ('run_id','frozen_runtime','required_classes',
                           'validation_passed','state_counts','observed_at')},
        'preparation':{'required_configurations':corrected.get('configuration_choices') if corrected else len(configs) if manifest else None,
            'verified_configurations':corrected.get('verified_configurations') if corrected else served if manifest else None,
            'corrected_inventory':{'source':refs['corrected_inventory'],'summary':corrected,
                'checkpoint':values['corrected_inventory'].get('checkpoint')},
            'legacy_index_coverage':{'required_configurations':len(configs) if manifest else None,
                'verified_configurations':served if manifest else None,'scope':'superseded selection; historical evidence only'},
            'denominator_complete':(corrected.get('full_dependency_denominator_complete') is True if corrected else status.get('summary',{}).get('denominator_complete') is True) and not gaps,'scope_gaps':gaps,
            'dependency_status_counts':corrected.get('dependency_statuses',{}) if corrected else dict(Counter(d.get('status','unobserved') for d in deps)),
            'source_summary':status.get('summary',manifest.get('summary')), 'per_class':status.get('per_class',[]), 'dispatch':status.get('execution',status.get('dispatch',status.get('active_jobs',[])))},
        'queue':values['queue'],'capacity':values['capacity'],
        'requirement_status_counts':dict(Counter(r.get('status','untested') for r in requirements)),
        'release_decisions':{'source':refs['production_gate'],'decision':values['production_gate'],
            'ledger_alone_authorizes_release':False,'required':'Revalidate exact referenced dependencies, native/runtime inputs, physical proofs and original production gates.'},
        'semantics':'Joined execution evidence; missing and stale sources remain explicit. A submitted job, implementation, demo, replay and acceptance are independent stages.'}


def publish(root):
    value=build(root);target=Path(root)/'infra/execution_ledger.json';target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(target)
    print(json.dumps({'observed_at':value['observed_at'],'rows':len(value['rows']),'accepted':value['accepted']['counts'],
        'preparation':{k:value['preparation'][k] for k in ('required_configurations','verified_configurations','denominator_complete','dependency_status_counts')}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--loop',action='store_true');a=p.parse_args()
    while True:
        publish(a.root)
        if not a.loop:break
        time.sleep(60)
