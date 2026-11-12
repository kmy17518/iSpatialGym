"""Observe exact root-owned preparation waves; never allocate or accept work."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/vision/u/minyeong/projects/isg_home/campaign_20261110')
SECONDARY = Path('/vision/u/shiyuc/minyeong/campaign_20261110')
WAVES = ('selected_artic_motion_r1', 'selected_artic_prefix_r1',
         'selected_artic_motion_close_variants_r1', 'selected_artic_motion_open_variants_r1')


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    raw = Path(path).read_bytes()
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def observe_wave(name, now):
    base = ROOT / 'training_preparation' / name
    request = base / 'request.json'
    wave = read(request)
    source = reference(request)
    assert wave['campaign_id'] == ROOT.name and wave['production_authorization'] is False
    secondary = SECONDARY / 'training_preparation' / name
    receipts = []
    for path in [base / 'submission_primary_r1.json',
                 secondary / 'dispatch_secondary_r1/submission.json',
                 secondary / 'dispatch_secondary_r2/submission.json']:
        if path.exists():
            receipts.append({'source': reference(path), 'record': read(path)})
    jobs = []
    for receipt in receipts:
        record = receipt['record']
        job = record.get('job_id')
        if job:
            jobs.append(str(job))
    scheduler = {'observed_at': now, 'jobs': jobs, 'status': 'no_allocation_receipt'}
    active = []
    if jobs:
        command = ['squeue', '-h', '-r', '-j', ','.join(jobs), '-o', '%i|%T|%R']
        result = subprocess.run(command, capture_output=True, text=True, timeout=25)
        scheduler.update(status='observed' if result.returncode == 0 else 'unavailable',
                         returncode=result.returncode, rows=result.stdout.splitlines(), error=result.stderr)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                job, state, reason = line.split('|', 2)
                active.append({'job_id': job, 'status': state.lower(), 'reason_or_node': reason,
                               'observed_at': now, 'wave': name})
    units = []
    for unit in wave['units']:
        primary = Path(unit['output']) / 'result.json'
        alternate = secondary / 'native_secondary_r2' / unit['id'] / 'result.json'
        row = {'unit_id': unit['id'], 'index': unit['index'], 'request': unit['request'],
               'status': 'no_result_observed', 'next_action': 'Compute reconciles exclusive allocation and input access; do not duplicate.'}
        candidates = [p for p in (primary, alternate) if p.exists()]
        if len(candidates) > 1:
            row.update(status='multiple_output_locations_require_diagnosis', outputs=[reference(p) for p in candidates],
                       next_action='Root and compute inspect output relocation and exclusive ownership before any join.')
        elif candidates:
            path = candidates[0]
            try:
                result = read(path)
                assert result['unit_id'] == unit['id'] and result['request'] == source
                assert result['benchmark_credit'] == 0 and result['production_authorization'] is False
                if path == alternate:
                    relocation = read(path.parent / 'output_relocation.json')
                    assert relocation['request'] == source and relocation['unit_id'] == unit['id']
                    assert Path(relocation['actual_output']).resolve() == path.parent.resolve()
                    row['output_relocation'] = reference(path.parent / 'output_relocation.json')
                row.update(status=result['status'], result=reference(path),
                           result_observed_at=result.get('finished_at', result.get('started_at')),
                           native_returncode=result.get('returncode'), elapsed_seconds=result.get('elapsed_seconds'),
                           reported_positive_components=result.get('positive_maps', result.get('positive_libraries')),
                           reported_artifacts=result.get('artifacts', []), error=result.get('error'),
                           next_action=result.get('next_action', 'Observe terminal native result; no output inferred.'))
            except (ValueError, KeyError, AssertionError, OSError) as exc:
                row.update(status='observation_or_identity_failure', error=repr(exc),
                           next_action='Root checks exact request/output identity; no verification credit.')
        units.append(row)
    return {'name': name, 'request': source, 'native_runtime': wave['native_runtime'],
            'observed_at': now, 'scheduler': scheduler, 'active_jobs': active, 'receipts': receipts,
            'physical_units': len(units), 'statuses': dict(Counter(u['status'] for u in units)),
            'units': units, 'benchmark_credit': 0,
            'semantics': 'Reported component audits only. Graph owner rechecks original artifacts and exact consumers; this observer grants no configuration validity or release.'}


def publish():
    now = datetime.now(timezone.utc).isoformat()
    waves = []
    for name in WAVES:
        try:
            waves.append(observe_wave(name, now))
        except (ValueError, KeyError, AssertionError, OSError, subprocess.TimeoutExpired) as exc:
            waves.append({'name': name, 'status': 'observation_unavailable', 'error': repr(exc), 'observed_at': now})
    directory = ROOT / 'infra/preparation_root'
    observation = {'schema': 'isg-root-component-observation-v1', 'campaign_id': ROOT.name,
                   'observed_at': now, 'owner': '/root', 'waves': waves,
                   'read_only_execution': True, 'collection_jobs_changed': False, 'production_authorization': False}
    target = directory / 'component_observation.json'
    atomic(target, observation)
    worker_path = directory / 'worker.json'
    worker = read(worker_path)
    worker.setdefault('last_agent_engineering_observed_at', worker['observed_at'])
    worker.update(observed_at=now, status='persistent_component_observation',
                  active_jobs=[j for w in waves for j in w.get('active_jobs', [])],
                  component_observation=reference(target),
                  engineering_observation_note='Worker timestamp reports persistent observation, not an interactive-agent heartbeat. Last agent engineering timestamp remains separate.',
                  next_action='Graph owner streams each exact positive artifact; root diagnoses failed motion/prefix components; compute dispatches exclusive ready units. Shared release checks remain separate.')
    worker['evidence'] = [e for e in worker.get('evidence', []) if e.get('path') != str(target)] + [reference(target)]
    atomic(worker_path, worker)
    print(json.dumps({'observed_at': now, 'waves': [{k: w.get(k) for k in ('name', 'physical_units', 'statuses', 'status')} for w in waves]}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', action='store_true')
    args = parser.parse_args()
    lock_path = ROOT / 'infra/preparation_root/observer.lock'
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            publish()
            if not args.loop:
                break
            time.sleep(60)
