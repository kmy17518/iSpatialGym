"""Coordinator-owned transfer of never-assigned typed maps to cluster owners."""
from pathlib import Path
import hashlib
import json
import time
import uuid
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.campaign_20261110.immutable_file_audit import ImmutableFileAudit


SCHEMA = 'isg-map-cluster-return-request-v1'


def unrepresented_claims(claims, observed_at):
    """Claims still consume capacity until actual starts precede the observation."""
    result = set()
    for claim in claims:
        represented = False
        started = Path(claim.get('unit', {}).get('output', '/nonexistent')) / 'started.json'
        if started.exists():
            row = json.loads(started.read_text())
            represented = (row.get('index') == claim.get('unit_index')
                           and row.get('started_at', float('inf')) <= observed_at
                           and str(row.get('job_id', '')).isdigit())
        if not represented:
            result.add(str(claim['job_id']))
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate(root, request, job, active_staging, heartbeat_jobs, remote_jobs, now, audit=None, claims=()):
    root = Path(root)
    file_hash = (audit or ImmutableFileAudit()).sha256
    jid = str(request['job_id'])
    if request.get('schema') != SCHEMA or request.get('campaign_id') != root.name:
        raise ValueError('cluster-return request identity differs')
    if not 0 <= now - request['observed_at'] <= 120 or not now <= request['expires_at'] <= request['observed_at'] + 180:
        raise ValueError('cluster capacity request expired')
    if str(job['id']) != jid or job.get('status') != 'queued' or job.get('tries', 0) != 0 or job.get('instance') or job.get('history'):
        raise ValueError('job is not never-assigned queued work')
    if jid in active_staging or jid in heartbeat_jobs or jid in remote_jobs:
        raise ValueError('job retains a stage, heartbeat, or remote assignment owner')
    directory = Path(job['job_dir'])
    if job.get('job_kind') != 'native_dependency' or not directory.is_relative_to(root / 'out/vast_map_prebuild') or Path(job['script']).name != 'job_map_prebuild.sh':
        raise ValueError('only proven typed reach-map work can return')
    if (directory / 'result').exists():
        raise ValueError('typed output already exists')
    queue = root / 'vast/queue' / (jid + '.json')
    manifest = directory / 'native_inputs.json'
    if request['queue'] != {'path': str(queue), 'sha256': file_hash(queue)} or request['native_inputs'] != {'path': str(manifest), 'sha256': file_hash(manifest)}:
        raise ValueError('immutable queue or native inputs changed')
    if job.get('native_inputs_sha256') != request['native_inputs']['sha256']:
        raise ValueError('queued native manifest identity differs')
    inputs = json.loads(manifest.read_text())
    if inputs.get('schema') != 'isg-vast-map-prebuild-v1' or inputs.get('new_native_instances') != 0 or inputs.get('production_released') is not False:
        raise ValueError('typed dependency purpose changed')
    for name, expected in {**inputs['files'], **inputs['library_files'], **inputs.get('seed_map_files', {})}.items():
        if not Path(name).resolve().is_relative_to(root) or file_hash(name) != expected:
            raise ValueError('exact native payload changed: ' + name)
    if request['unit_index'] != inputs['unit_index'] or request['source_plan']['path'] != inputs['source_plan'] or file_hash(inputs['source_plan']) != request['source_plan']['sha256']:
        raise ValueError('exact map plan/unit changed')
    account = request['account']
    roots = {'minyeong': root, 'shiyuc': Path('/vision/u/shiyuc/minyeong') / root.name}
    if account not in roots:
        raise ValueError('unknown cluster owner')
    target = roots[account] / 'training_preparation' / Path(inputs['source_plan']).parent.name / 'build_dispatch_r1'
    prepared = request['prepared']
    if Path(prepared['path']) != target / 'prepared.json' or file_hash(prepared['path']) != prepared['sha256']:
        raise ValueError('owned cluster preparation changed')
    rows = json.loads(Path(prepared['path']).read_text())['units']
    unit = next((u for u in rows if u['index'] == request['unit_index']), None)
    if unit is None or request['unit'] != unit or file_hash(unit['script']) != unit['script_sha256']:
        raise ValueError('owned exact unit script changed')
    for field, name in (('wrapper', 'pinned.sh'), ('runner', 'run.py')):
        if request.get(field) != {'path': str(target / name), 'sha256': file_hash(target / name)}:
            raise ValueError('owned cluster launch code changed')
    output = Path(unit['output'])
    if not output.is_relative_to(target) or (output / 'started.json').exists() or (output / 'terminal.json').exists():
        raise ValueError('cluster unit already started or output escaped')
    capacity = request['capacity']
    capacity_path = Path(capacity['path'])
    if not capacity_path.is_relative_to(root / 'vast/cluster_return/capacity') or file_hash(capacity_path) != capacity['sha256']:
        raise ValueError('immutable cluster capacity proof changed')
    observation = json.loads(capacity_path.read_text())
    if not 0 <= now - observation['observed_at'] <= 180:
        raise ValueError('cluster capacity observation expired')
    selected = observation['selected_job_ids']
    if len(set(selected)) != len(selected) or jid not in selected or len(selected) > observation['compatible_available_slots']:
        raise ValueError('requested cluster capacity is overcommitted')
    assignments = observation.get('selected_accounts', {})
    if assignments.get(jid) != account or set(assignments) != set(selected):
        raise ValueError('capacity selection lacks unique account ownership')
    # Capacity is shared across accounts and request cohorts. A previous unsubmitted
    # claim consumes a slot even when it came from a different capacity file.
    outstanding = unrepresented_claims(claims, observation['observed_at'])
    if len(set(selected) | outstanding) > observation['compatible_available_slots']:
        raise ValueError('aggregate cluster claims exceed observed capacity')
    return {'job_id': jid, 'account': account, 'unit_index': request['unit_index'],
            'source_plan': request['source_plan'], 'prepared': prepared, 'unit': unit,
            'capacity': capacity, 'native_inputs': request['native_inputs'],
            'wrapper': request['wrapper'], 'runner': request['runner']}


def reconcile(root, states, active_staging, heartbeats, remote_jobs, save_state, *, now=None):
    """Only this coordinator writes the exclusive claim before acknowledging it."""
    root = Path(root)
    now = time.time() if now is None else now
    base = root / 'vast/cluster_return'
    heartbeat_jobs = {str(j) for hb in heartbeats.values() for j in (hb.get('running') or {})}
    report = {'observed_at': now, 'acknowledged': [], 'rejected': [], 'deferred': []}
    audit = ImmutableFileAudit()
    for path in sorted((base / 'requests').glob('*.json')):
        acknowledgement = base / 'acknowledgements' / path.name
        rejected = base / 'rejected' / path.name
        if acknowledgement.exists() or rejected.exists():
            continue
        try:
            request = json.loads(path.read_text())
            reference = {'path': str(path), 'sha256': digest(path)}
            jid = str(request['job_id'])
            job = states[jid]
            prior = job.get('cluster_return') or {}
            if job.get('status') == 'cluster_reserved' and prior.get('request') == reference:
                # Crash after the durable claim and before acknowledgement.
                claim = prior
            else:
                identity = validate(root, request, job, active_staging, heartbeat_jobs, remote_jobs, now, audit,
                                    [s['cluster_return'] for s in states.values() if s.get('cluster_return', {}).get('request')])
                claim = {**identity, 'request': reference, 'claimed_at': now,
                         'boundary': 'Coordinator removed this never-assigned job before account submission.'}
                job.update(status='cluster_reserved', cluster_return=claim)
                save_state(job)
            write(acknowledgement, {'schema': 'isg-map-cluster-return-ack-v1', 'campaign_id': root.name,
                                   'observed_at': now, **claim})
            report['acknowledged'].append(jid)
        except Exception as error:
            value = {'observed_at': now, 'request': str(path), 'error': str(error), 'cluster_submission_allowed': False}
            # A durable claim is never discarded because writing its acknowledgement failed.
            if any(s.get('status') == 'cluster_reserved' and (s.get('cluster_return') or {}).get('request', {}).get('path') == str(path)
                   for s in states.values()):
                report['deferred'].append(value)
                continue
            write(rejected, value)
            report['rejected'].append(value)
    report['immutable_input_audit'] = audit.summary()
    write(base / 'coordinator_status.json', report)
    return report


def has_pending(root):
    base = Path(root) / 'vast/cluster_return'
    return any(not (base / 'acknowledgements' / p.name).exists() and not (base / 'rejected' / p.name).exists()
               for p in (base / 'requests').glob('*.json'))
