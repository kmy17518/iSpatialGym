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
PARKED_CLASSES = {'inter_object_relation_satisfaction-touching_03',
    'inter_object_relation_satisfaction-under_large_scale_01',
    'inter_object_relation_satisfaction-under_large_scale_02',
    'inter_object_relation_satisfaction-under_large_scale_03',
    'inter_object_relation_satisfaction-nextto_04'}


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
        return sum(row['split'] == split and (split == 'eval_category' or row.get('class_id', cls) == cls)
                   for row in state['accepted'].values()) >= limit
    source = Path(root)/'dataset/allocations.sqlite'
    if not source.exists(): return False
    with tempfile.TemporaryDirectory(prefix='isg-gate-count-', dir='/tmp') as temporary:
        local = Path(temporary)/'counts.sqlite'
        shutil.copyfile(source, local)
        with sqlite3.connect(f'file:{local}?mode=ro', uri=True) as db:
            count = db.execute('SELECT COUNT(*) FROM members WHERE split=? AND task=? AND class_id=?',
                               (split, task, task if split == 'eval_category' else cls)).fetchone()[0]
    return count >= limit


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
            if expiry <= now: continue
            evidence = _owned(root, authorization['authorization_path'])
            if file_digest(evidence) != authorization['authorization_sha256']: continue
            proof = json.loads(evidence.read_text())
            if proof.get('purpose') != 'training_pipeline_canary' or proof.get('config_sha256') != result['config_sha256']: continue
            if (proof.get('start'), proof.get('count')) != (int(start), int(count)): continue
            return {**result, 'allowed': True, 'category': 'validation_canary',
                    'reason': 'exact bounded pipeline canary', 'authorization': authorization}
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
