"""Observational cost joins; these records never allocate cost or change routing.

Rental intent is the eligible work set at request time, not a promise that a
worker belongs to one cohort. Missing identities and shared costs stay explicit.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import uuid

CAMPAIGN = 'campaign_20261110'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def environment(path):
    result = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            try: result[key] = ' '.join(shlex.split(value))
            except ValueError: result[key] = value
    return result


def scope(job):
    """Read only selected identities, never serialize arbitrary environment data."""
    result = {'campaign_id': CAMPAIGN, 'job_id': str(job['id']),
              'job_kind': job.get('job_kind', 'collection'), 'job_dir': job.get('job_dir'),
              'group_ids': [], 'group_binding': 'unavailable', 'evidence': [], 'errors': []}

    def document(path):
        path = Path(path); data = path.read_bytes()
        result['evidence'].append({'path': str(path), 'sha256': digest(data)})
        return json.loads(data)

    try:
        directory = Path(job['job_dir'])
        env_path = directory / 'job.env'
        env = environment(env_path)
        result['evidence'].append({'path': str(env_path), 'sha256': digest(env_path.read_bytes())})
        result.update(runtime_snapshot=env.get('ISGE_CODE'), class_id=env.get('ALIAS_KEY') or env.get('ALIAS'),
                      task=env.get('FAMILY'), start_index=env.get('START'), instance_count=env.get('N'))
        config_path = env.get('ISPATIALGYM_CAMPAIGN_CONFIG')
        if config_path:
            config = document(config_path)
            if config.get('campaign_id') != CAMPAIGN: raise ValueError('config campaign identity differs')
            template = config.get('template', {})
            result.update(split=config.get('split'), purpose=config.get('purpose'),
                          class_id=template.get('class_id', result['class_id']),
                          task=template.get('task', result['task']), template_id=template.get('template_id'),
                          validation_key=config.get('validation_key'))
        queue_path = env.get('ISPATIALGYM_CAMPAIGN_GOAL_QUEUE')
        if queue_path:
            queue = document(queue_path)
            if queue.get('campaign_id') != CAMPAIGN: raise ValueError('queue campaign identity differs')
            result['group_ids'] = sorted(set(map(str, queue.get('group_ids', []))))
            result['group_binding'] = 'exact_queue_groups' if result['group_ids'] else 'queue_without_group_ids'
            result['queue_schema'] = queue.get('schema')
        elif env.get('ISPATIALGYM_CAMPAIGN_GOAL_PLAN'):
            plan = document(env['ISPATIALGYM_CAMPAIGN_GOAL_PLAN'])
            if plan.get('campaign_id') != CAMPAIGN:
                raise ValueError('goal plan campaign identity differs')
            if not isinstance(plan.get('group_id'), str) or not plan['group_id']:
                raise ValueError('goal plan lacks an explicit group identity')
            if plan.get('task') != result.get('task') or plan.get('split') != result.get('split'):
                raise ValueError('goal plan task/split differs from config')
            if plan.get('contract_template_sha256') != (result.get('validation_key') or {}).get('template_sha256'):
                raise ValueError('goal plan template differs from config')
            result.update(group_ids=[plan['group_id']], group_binding='exact_goal_plan_group',
                          goal_plan_schema=plan.get('schema'))
        elif str(result.get('task', '')).startswith('camera_relocalization'):
            result['group_binding'] = 'camera_exempt'
        native = directory / 'native_inputs.json'
        replay = directory / 'replay_inputs.json'
        if native.exists():
            manifest = document(native); entry = manifest['original_entry']
            result.update(job_kind='native_dependency', runtime_snapshot=manifest.get('snapshot'),
                          class_id=entry.get('class_id'), task=entry.get('task'), split=entry.get('split'),
                          candidate_id=entry.get('candidate_id'), source_manifest=manifest.get('source_manifest'),
                          source_manifest_sha256=manifest.get('files', {}).get(manifest.get('source_manifest')),
                          source_entry_index=manifest.get('entry_index'), start_index=manifest.get('fresh_start_index'),
                          group_binding='prerequisite_candidate_only', group_ids=[])
        elif replay.exists():
            manifest = document(replay)
            result.update(job_kind='physics_replay', source_job_dir=manifest.get('source_job'),
                          source_tro=manifest.get('source_tro'), runtime_snapshot=manifest.get('snapshot'),
                          source_tro_sha256=manifest.get('files', {}).get(manifest.get('source_tro')))
            if manifest.get('source_tro'):
                tro = document(manifest['source_tro'])
                campaign = tro.get('ispatialgym', {}).get('campaign', {})
                group = campaign.get('goal_group_id') or campaign.get('group_id')
                if group:
                    result.update(group_ids=[str(group)], group_binding='exact_replayed_tro_group')
                else:
                    result['group_binding'] = 'source_queue_groups_replay_member_unavailable'
        result['mode'] = {'eval_standard':'standard_eval','train':'train_only','eval_model':'held_out','eval_category':'held_out'}.get(result.get('split'),'unavailable')
        result['status'] = 'observed'
    except Exception as error:
        result['status'] = 'incomplete'
        result['errors'].append(type(error).__name__ + ': ' + str(error))
    return result


def rental_intent(jobs, requested_gpu):
    rows = [scope(job) for job in jobs if job.get('gpu') == requested_gpu]
    return {'campaign_id': CAMPAIGN, 'purpose': 'shared_eligible_work_set' if rows else 'unassigned_shared_campaign_capacity',
            'requested_gpu': requested_gpu, 'candidate_jobs': rows,
            'exclusive_cost_assignment': False,
            'boundary': 'Eligible queue observed before create request; later worker reuse and shared startup/idle are not allocated to these groups.'}


def retain(root, kind, identity, payload):
    """One immutable record per transition identity, idempotent across recovery."""
    directory = Path(root); directory.mkdir(parents=True, exist_ok=True)
    token = digest((kind + '\0' + str(identity)).encode())
    path = directory / (kind + '_' + token + '.json')
    if path.exists():
        data = path.read_bytes()
        return {'path': str(path), 'sha256': digest(data)}
    value = {'schema': 'isg-vast-cost-provenance-v1', 'campaign_id': CAMPAIGN,
             'kind': kind, 'identity': str(identity), 'observed_at_unix': time.time(), **payload}
    data = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    try: os.link(temporary, path)
    except FileExistsError: pass
    finally: temporary.unlink()
    return {'path': str(path), 'sha256': digest(path.read_bytes())}


def assignment_identity(job, attempt_id=None):
    return str(attempt_id or (str(job['id']) + '_' + str(job.get('instance')) + '_' + str(job.get('assigned'))))
