"""Persistent fail-closed gate for this campaign's training production.

This is an authorization check, never a substitute for physical validation.
No path or config label alone authorizes standalone training.
"""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
import shlex
import sqlite3
from scripts.campaign_20261110.contracts import file_digest

SCHEMA = 'isg-training-production-gate-v1'
CAMPAIGN = 'campaign_20261110'
# Latest owner clarification: the other four promised updates are available.
# Availability removes the hold; their exact native/demo/replay gates still apply.
PARKED_CLASSES = set()  # Owner supplied the final under01 update in c570f17d.


def read_policy(root):
    path = Path(root) / 'controllers/training_production_gate.json'
    policy = json.loads(path.read_text())
    if policy.get('schema') != SCHEMA or policy.get('campaign_id') != CAMPAIGN:
        raise ValueError('invalid production gate identity')
    return policy, file_digest(path)


def read_environment(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'): continue
        if line.startswith('export '): line = line[7:]
        if '=' not in line: continue
        key, value = line.split('=', 1)
        words = shlex.split(value) if value else ['']
        if len(words) != 1: raise ValueError('ambiguous job environment: ' + key)
        result[key] = words[0]
    return result


def _owned(root, path):
    value = Path(path).resolve()
    if not value.is_relative_to(Path(root).resolve()):
        raise ValueError('gate input outside campaign')
    return value


def _inputs(root, environment):
    queue_path = environment.get('ISPATIALGYM_CAMPAIGN_GOAL_QUEUE')
    queue = None
    if queue_path:
        queue = json.loads(_owned(root, queue_path).read_text())
        config_path = _owned(root, queue['config_path'])
        if file_digest(config_path) != queue['config_sha256']:
            raise ValueError('queue configuration hash changed')
        for entry in queue['entries']:
            if file_digest(_owned(root, entry['path'])) != entry['sha256']:
                raise ValueError('queue goal plan hash changed')
    else:
        config_path = _owned(root, environment['ISPATIALGYM_CAMPAIGN_CONFIG'])
    return config_path, json.loads(config_path.read_text()), queue


def evaluation_exhausted(root, template):
    """Read atomic published acceptance; pending work never counts as accepted."""
    from scripts.campaign_20261110.contracts import task_quotas
    import shutil
    import sqlite3
    import tempfile
    task, split, cls = template['task'], template['split'], template['class_id']
    quota = next(q for q in task_quotas() if q.task == task)
    limit = {'eval_standard': 10, 'eval_model': quota.model_instances_per_class,
             'eval_category': quota.category_instances}[split]
    if task.startswith('camera_relocalization'):
        name = 'camera_obstructed_quota.json' if task.endswith('obstructed') else ('camera_category_quota.json' if split == 'eval_category' else 'camera_quota.json')
        path = Path(root)/'dataset'/name
        if not path.exists(): return False
        state = json.loads(path.read_text())
        from scripts.campaign_20261110.current_eligibility import filtered_camera
        state['accepted']=filtered_camera(root,state['accepted'])
        return sum(row['split'] == split and (split == 'eval_category' or row.get('class_id', cls) == cls)
                   for row in state['accepted'].values()) >= limit
    source = Path(root)/'dataset/allocations.sqlite'
    if not source.exists(): return False
    with tempfile.TemporaryDirectory(prefix='isg-gate-count-', dir='/tmp') as temporary:
        local = Path(temporary)/'counts.sqlite'
        shutil.copyfile(source, local)
        with sqlite3.connect(f'file:{local}?mode=ro', uri=True) as db:
            from scripts.campaign_20261110.current_eligibility import install_quota_view
            install_quota_view(root,db)
            count = db.execute('SELECT COUNT(*) FROM current_members WHERE split=? AND task=? AND class_id=?',
                               (split, task, task if split == 'eval_category' else cls)).fetchone()[0]
    return count >= limit


def canonical_training_ready(root):
    """Require the real100-class chain, never a99-class or instance-count proxy."""
    root=Path(root)
    path=root/'infra/standard_recovery/canonical_validation.json'
    try:
        value=json.loads(path.read_text())
        if value.get('schema')!='isg-canonical-template-validation-v1' or value.get('total')!=100 or value.get('validated')!=100:
            return False
        classes=value['classes']
        if len(classes)!=100 or len({r['class_id']for r in classes})!=100 or not all(r.get('validated') is True and r.get('status')=='fresh_chain_verified' for r in classes):
            return False
        from scripts.campaign_20261110.dashboard_evidence import standard_catalog
        from scripts.campaign_20261110.canonical_validation import audit
        from scripts.campaign_20261110.export_release import ArtifactAudit
        catalog=standard_catalog(root/'manifests/dashboard_standard_catalog.json')
        if set(catalog)!={r['class_id']for r in classes}:return False
        # A summary alone cannot release anything: repeat the actual native,
        # demo, replay, hard-QA, sibling/camera and accepted-artifact audit.
        fresh=audit(root,catalog,ArtifactAudit(root,1794321600),1794321600)
        return fresh['total']==fresh['validated']==100
    except (OSError,ValueError,KeyError,TypeError):
        return False


def heldout_parallel_ready(root, config, config_sha256, authorization, *, check_freshness=True):
    """Future training tests must advance their related holdouts in parallel.

    An actionable engineering blocker is permitted; this is deliberately not
    an all-heldouts-complete release condition. The100-class hold is separate.
    """
    ref = authorization['heldout_parallel_work']
    path = _owned(root, ref['path'])
    if file_digest(path) != ref['sha256']:
        raise ValueError('held-out parallel-work evidence changed')
    proof = json.loads(path.read_text()); template = config['template']
    if (proof.get('campaign_id'), proof.get('training_config_sha256'), proof.get('training_class_id'), proof.get('task')) != (
            CAMPAIGN, config_sha256, template['class_id'], template['task']):
        raise ValueError('held-out parallel-work scope differs from exact training config')
    age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(proof['observed_at'].replace('Z','+00:00'))).total_seconds()
    if age < 0 or (check_freshness and age > 1800):
        raise ValueError('held-out parallel-work observation is stale or future-dated')
    from scripts.campaign_20261110.contracts import task_quotas
    quota = next(q for q in task_quotas() if q.task == template['task'])
    for key, split, applicable in [('model','eval_model',bool(quota.model_instances_per_class)),
                                   ('category_room','eval_category',bool(quota.category_instances))]:
        row = proof[key]
        if not applicable:
            if row.get('state') != 'not_applicable':
                raise ValueError('inapplicable held-out scope must be explicit')
            continue
        if row.get('task') != template['task'] or (key == 'model' and row.get('class_id') != template['class_id']):
            raise ValueError('related held-out class/task differs')
        if not row.get('owner'):
            raise ValueError('related held-out work lacks a named owner')
        state = row.get('state')
        if state == 'complete':
            if not evaluation_exhausted(root,{**template,'split':split}):
                raise ValueError('related held-out quota is not authoritatively complete')
        elif state in ('queued','running'):
            if not (row.get('controller') or row.get('active_jobs')):
                raise ValueError('related held-out work has no active controller/job')
        elif state == 'engineering_blocked':
            if not row.get('blocker') or not row.get('next_action'):
                raise ValueError('related held-out blocker lacks an actionable correction')
        else:
            raise ValueError('related held-out work has not advanced')
        if not row.get('evidence'):
            raise ValueError('related held-out work lacks source evidence')
        for evidence in row['evidence']:
            if file_digest(_owned(root,evidence['path'])) != evidence['sha256']:
                raise ValueError('related held-out source evidence changed')
    return ref


def check_environment(root, environment, *, kind='collection', start=None, count=None):
    result = {'allowed': False, 'reason': 'unknown collection scope', 'category': 'unverified',
              'gate_sha256': None}
    try:
        policy, result['gate_sha256'] = read_policy(root)
    except (OSError, ValueError, TypeError) as exc:
        policy = None
        result['policy_error'] = str(exc)
    # These paths create validation evidence, not dataset training instances.
    if kind == 'native_dependency':
        return {**result, 'allowed': True, 'category': kind, 'reason': 'independent validation'}
    if kind not in ('collection', 'source_validation'): return result
    try:
        path, config, queue = _inputs(root, environment)
        template = config['template']
        split = template['split']
        result.update(config_path=str(path), config_sha256=file_digest(path), split=split,
                      task=template['task'], class_id=template['class_id'])
        if template['class_id'] in PARKED_CLASSES:
            return {**result, 'reason': 'owner artifact dependency remains parked', 'category': 'owner_blocked'}
        if split in ('eval_standard', 'eval_model', 'eval_category'):
            if kind == 'collection' and split == 'eval_standard' and not template['task'].startswith('camera_relocalization') and count is not None and count > 3:
                from scripts.campaign_20261110.training_eligibility import standard_group_proof
                proof = standard_group_proof(root, template['class_id'], config['validation_key'])
                if proof is None:
                    return {**result, 'reason': 'bulk standard collection requires exact accepted eval group with distinct sibling; use one bounded group'}
                result['standard_group_production_proof'] = proof
            if kind == 'collection' and (queue is not None or environment.get('ISPATIALGYM_CAMPAIGN_GOAL_PLAN') or template['task'].startswith('camera_relocalization')):
                if evaluation_exhausted(root, template):
                    return {**result, 'reason': 'evaluation quota exhausted; cannot produce training by relabeling'}
            return {**result, 'allowed': True, 'category': split, 'reason': 'evaluation purpose; acceptance quota remains mandatory'}
        if split != 'train': raise ValueError('unknown campaign split')
        result.update(category='standalone_training', reason='standalone training production blocked')
        if policy is None: return result
        # Reopening is intentionally unsupported until the required complete
        # pipeline has an implemented evidence verifier. A hand-edited flag
        # must never act as a production release.
        now = dt.datetime.now(dt.timezone.utc)
        for authorization in policy.get('allowed_canaries', []):
            if authorization.get('config_sha256') != result['config_sha256']: continue
            if authorization.get('purpose') != 'training_pipeline_canary': continue
            if start is None or count is None or count <= 0: continue
            if int(authorization['start']) != int(start) or int(authorization['count']) != int(count): continue
            if count > int(authorization['max_instances']) or count > 24: continue
            expiry = dt.datetime.fromisoformat(authorization['expires_at'].replace('Z', '+00:00'))
            # Replaying an already generated bounded canary produces validation
            # evidence only. The audit separately proves native creation was
            # inside this exact authorization window; expiry still bars every
            # collection attempt, retry and new native instance.
            if kind == 'collection' and expiry <= now: continue
            evidence = _owned(root, authorization['authorization_path'])
            if file_digest(evidence) != authorization['authorization_sha256']: continue
            proof = json.loads(evidence.read_text())
            if proof.get('purpose') != 'training_pipeline_canary' or proof.get('config_sha256') != result['config_sha256']: continue
            if (proof.get('start'), proof.get('count')) != (int(start), int(count)): continue
            result['heldout_parallel_work'] = heldout_parallel_ready(root,config,result['config_sha256'],proof,
                check_freshness=kind == 'collection')
            return {**result, 'allowed': True, 'category': 'validation_canary',
                    'reason': 'exact bounded pipeline canary', 'authorization': authorization}
        if not canonical_training_ready(root):
            return {**result,'category':'eval_validation_hold','reason':'bulk standalone training requires all100 exact canonical eval classes; only separately authorized bounded pipeline canaries may run earlier'}
        return result
    except (OSError, ValueError, TypeError, KeyError, StopIteration, sqlite3.Error) as exc:
        return {**result, 'allowed': False, 'reason': 'unverified collection scope: ' + str(exc)}


def check_dispatch(root, command, *, kind='collection'):
    try:
        environment = {}
        for i, word in enumerate(command):
            if word == '--extra-env':
                key, value = command[i+1].split('=', 1)
                if key in environment and environment[key] != value: raise ValueError('conflicting environment')
                environment[key] = value
        start = int(command[command.index('--base-index')+1]) if '--base-index' in command else None
        pos = command.index('submit')
        count = int(command[pos+2])
        return check_environment(root, environment, kind=kind, start=start, count=count)
    except (ValueError, IndexError) as exc:
        return {'allowed': False, 'category': 'unverified', 'reason': 'invalid dispatch command: ' + str(exc)}


def check_job(root, job_dir, *, kind='collection'):
    try:
        if kind == 'native_dependency':
            value = json.loads((Path(job_dir) / 'native_inputs.json').read_text())
            cls = value['original_entry']['class_id']
            if value['campaign_root'] != str(Path(root).resolve()):
                raise ValueError('native validation campaign differs')
            if cls in PARKED_CLASSES:
                return {'allowed': False, 'category': 'owner_blocked', 'reason': 'owner artifact dependency remains parked'}
            return check_environment(root, {}, kind=kind)
        if kind == 'physics_replay':
            value = json.loads((Path(job_dir) / 'replay_inputs.json').read_text())
            source = _owned(root, value['source_job'])
            result = check_job(root, source, kind='source_validation')
            return {**result, 'source_job': str(source), 'replay_source_checked': True}
        environment = read_environment(Path(job_dir) / 'job.env')
        return check_environment(root, environment, kind=kind,
            start=int(environment['START']) if environment.get('START') else None,
            count=int(environment['N']) if environment.get('N') else None)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'allowed': False, 'category': 'unverified', 'reason': 'unreadable job identity: ' + str(exc)}


def require_dispatch(root, command):
    decision = check_dispatch(root, command)
    if not decision['allowed']: raise ValueError('training production gate: ' + decision['reason'])
    return decision
