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
    'canonical': 'infra/standard_recovery/canonical_validation.json',
    'standard': 'infra/standard_recovery/ledger.json',
    'heldout': 'infra/heldout_recovery/ledger.json',
    'pipeline': 'training_pipeline_canaries/pipeline_status.json',
    'queue': 'infra/capacity_recovery/vast_excluded_queue_audit_r2.json',
    'capacity': 'infra/capacity_recovery/ledger.json',
    'eligibility': 'accounting/current_eligibility.json',
    'production_gate': 'controllers/training_production_gate.json',
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


def row(identifier, kind, owner, reference, **fields):
    return {'id': identifier, 'kind': kind, 'owner': owner, 'observed_at': reference.get('observed_at'),
            'evidence_state': reference['state'], 'requirement_ids': [], 'dependencies': [],
            'active_work': [], 'artifacts': [], 'verification': {'status': 'unobserved'},
            'acceptance': {'status': 'not_applicable' if kind in ('requirement', 'dependency', 'pipeline_canary') else 'unobserved'},
            'stages': {s: stage() for s in STAGES}, 'sources': [reference], **fields}


def build(root, now=None):
    now = time.time() if now is None else now; root = Path(root).resolve()
    observations = {k: read(root, path, now) for k,path in SOURCES.items()}
    values = {k:v[0] for k,v in observations.items()}; refs = {k:v[1] for k,v in observations.items()}
    result = []; requirements = values['requirements'].get('requirements', [])
    def reqs(task):
        return [r['id'] for r in requirements if task in (r.get('tasks') or []) or r.get('scope') == 'all']
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
            dependencies=observed.get('dependencies',[]), active_work=observed.get('active_jobs',[]),
            controller=observed.get('controller'), acceptance=observed.get('acceptance',c.get('acceptance',{})),
            verification=c, last_progress_at=observed.get('last_progress_at'), blocker=observed.get('blocker'),
            next_action=observed.get('next_action') or 'Reconcile exact accepted quota and current refill owner.'))
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
        result.append(row(f"pipeline:{r.get('candidate_id')}:{r.get('start',i)}", 'pipeline_canary', '/root/dataset_contract', refs['pipeline'],
            configuration_id=r.get('candidate_id'), class_id=r.get('class_id'), status=r.get('status'), runtime=r.get('native_runtime'),
            configuration_sha256=r.get('configuration_sha256'), dependencies=r.get('dependencies',[]), active_work=[r.get('allocation',{})],
            artifacts=[v for v in (r.get('request'),r.get('native'),r.get('review')) if v],
            stages={**{s:stage() for s in STAGES}, 'native':stage(r.get('native')), 'accepted':{'status':'excluded_zero_credit'}},
            next_action=r.get('next_action') or 'Owner reconciles completed stages into own physics replay, hard QA and batched restoration; no benchmark credit.'))
    counts = values['eligibility'].get('counts',{}).get('current_eligible')
    gaps = status.get('scope_gaps', manifest.get('scope_gaps', ['Full selected configuration expansion has not been observed.']))
    return {'schema':'isg-campaign-execution-ledger-v1','campaign_id':CAMPAIGN,
        'observed_at':datetime.fromtimestamp(now,timezone.utc).isoformat(),'owner':'/root',
        'sources':refs,'rows':result,'indexed_sources':{'configurations':refs['configurations'],'dependencies':refs['dependencies']},
        'requirements_by_task':{t:reqs(t) for t in sorted({c.get('task') for c in configs if c.get('task')})},'accepted':{'counts':counts,'source':refs['eligibility']},
        'canonical':{k:values['canonical'].get(k) for k in ('validated','total','state_counts')},
        'preparation':{'required_configurations':len(configs) if manifest else None,'verified_configurations':served if manifest else None,
            'denominator_complete':status.get('summary',{}).get('denominator_complete') is True and not gaps,'scope_gaps':gaps,
            'dependency_status_counts':dict(Counter(d.get('status','unobserved') for d in deps)),
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
        'preparation':{k:v for k,v in value['preparation'].items() if k!='dispatch'}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--loop',action='store_true');a=p.parse_args()
    while True:
        publish(a.root)
        if not a.loop:break
        time.sleep(60)
