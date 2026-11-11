"""Move only proven never-started campaign physics replay jobs to private Vast."""
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from scripts.campaign_20261110.storage import atomic_directory_lock
from scripts.campaign_20261110 import vast_replay, overflow_readiness


def require(value, message):
    if not value: raise ValueError(message)


def scheduler_proof(job_id):
    require(str(job_id).isdigit(), 'invalid replay scheduler identity')
    env = dict(os.environ, TZ='UTC', SLURM_TIME_FORMAT='standard')
    raw = subprocess.check_output(['scontrol', 'show', 'job', str(job_id), '-o'], text=True, timeout=30, env=env)
    fields = dict(item.split('=', 1) for item in shlex.split(raw) if '=' in item)
    accounting = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job_id),
        '--format=JobIDRaw,State,Start,ElapsedRaw,AllocTRES'], text=True, timeout=30, env=env)
    rows = [line.split('|') for line in accounting.splitlines() if line.split('|')[0] == str(job_id)]
    require(len(rows) == 1, 'ambiguous exact replay accounting row')
    row = rows[0]
    return {'observed_at_unix': time.time(), 'job_id': str(job_id), 'scheduler': fields,
            'accounting': {'state': row[1].split()[0].rstrip('+'), 'start': row[2],
                           'elapsed_seconds': int(row[3]), 'allocated_tres': row[4]}}


def never_started(proof, state):
    row = proof['accounting']
    return (row['state'] == state and row['start'] in ('Unknown', 'None', 'N/A', '')
            and row['elapsed_seconds'] == 0 and not row['allocated_tres'])


def eligible(proof, entry, root, minimum_age, timestamp):
    overflow_readiness.pending_delay(minimum_age)
    info = proof['scheduler']
    require(proof['job_id'] == str(entry['replay_job_id']), 'replay scheduler identity differs')
    require(info['UserId'].split('(')[0] == 'minyeong', 'replay belongs to another owner')
    require(info['JobName'] == 'isg261110_replay' and
            info.get('Comment') == 'campaign_20261110_replay_' + entry['source_job_id'], 'replay campaign/source identity differs')
    require(Path(info['Command']).name == 'infra_open_loop.sh' and
            Path(info['Command']).is_relative_to(root), 'replay command is outside campaign')
    if info['JobState'] != 'PENDING' or not never_started(proof, 'PENDING'): return False
    submitted = datetime.datetime.fromisoformat(info['SubmitTime']).replace(tzinfo=datetime.timezone.utc).timestamp()
    return timestamp - submitted >= minimum_age


def output_empty(output, root):
    path = Path(output)
    require(path.absolute().is_relative_to(root / 'validation') and
            path.parent.resolve().is_relative_to(root / 'validation'), 'replay output outside campaign')
    require(not path.is_symlink() and (not path.exists() or (path.is_dir() and not any(path.iterdir()))),
            'pending replay already has execution evidence')


def activate(job, root):
    jid = str(vast_replay.enqueue(job, root))
    vast_replay.verify(job, root)
    with atomic_directory_lock(root / 'vast/config.dispatch.lockdir'):
        config = root / 'vast/config.json'; value = json.loads(config.read_text())
        require(value['budget_usd'] <= 3000, 'campaign budget exceeds authorization')
        for key in ('dispatch_job_ids', 'dispatch_priority_job_ids'):
            if jid not in value.setdefault(key, []): value[key].insert(0, jid)
        vast_replay.write(config, value)
    return jid


def migrate(entry, root, native_snapshot, helper_snapshot, minimum_age, save):
    """Called only by the sole validation watcher; durable intent precedes cancel."""
    root = Path(root).resolve()
    intent = entry.get('vast_migration')
    if intent and intent.get('status') == 'active': return False
    if not intent:
        proof = scheduler_proof(entry['replay_job_id'])
        if not eligible(proof, entry, root, minimum_age, time.time()): return False
        output_empty(entry['output'], root)
        readiness = overflow_readiness.check(root, proof['scheduler'])
        token = 'pending_' + entry['replay_job_id'] + '_' + entry['source_job_id']
        job = root / 'out/vast_replay' / token
        if not job.exists():
            vast_replay.prepare(root, entry['source_job_dir'], native_snapshot, helper_snapshot, entry['output'], token)
        manifest = vast_replay.verify(job, root)
        require(manifest['source_job'] == str(Path(entry['source_job_dir']).resolve()) and
                manifest['snapshot'] == str(Path(native_snapshot).resolve()) and
                manifest['canonical_output'] == entry['output'], 'existing replay transport identity differs')
        jid = str(vast_replay.enqueue(job, root))  # Unactivated; no worker may execute it.
        intent = {'status': 'prepared', 'original_replay_job_id': entry['replay_job_id'],
                  'vast_job_id': jid, 'job_dir': str(job), 'before': proof,
                  'minimum_pending_seconds': minimum_age, 'prepared_at_unix': time.time(), 'overflow_readiness': readiness}
        entry['vast_migration'] = intent; save()
    job = Path(intent['job_dir'])
    if intent['status'] in ('prepared', 'cancel_requested'):
        before = scheduler_proof(intent['original_replay_job_id'])
        if not never_started(before, 'CANCELLED'):
            if not eligible(before, entry, root, minimum_age, time.time()):
                intent.update(status='not_migrated_scheduler_advanced', after=before); save(); return False
            output_empty(entry['output'], root)
            intent['overflow_readiness'] = overflow_readiness.check(root, before['scheduler'])
            intent['status'] = 'cancel_requested'; save()
            # Scheduler atomically filters out jobs which became RUNNING.
            subprocess.run(['scancel', '--state=PENDING', intent['original_replay_job_id']], check=True, timeout=30)
            after = scheduler_proof(intent['original_replay_job_id'])
        else: after = before
        if not never_started(after, 'CANCELLED'):
            intent['last_confirmation'] = after; save()
            require(after['accounting']['start'] in ('Unknown', 'None', 'N/A', ''),
                    'replay started during cancellation; replacement remains inactive')
            return False
        output_empty(entry['output'], root)
        intent.update(status='cancelled_never_started', after=after); save()
    if intent['status'] == 'cancelled_never_started':
        vast_replay.verify(job, root); vast_replay.bind_output(job, root)
        entry.setdefault('scheduler_attempt_history', []).append({
            'replay_job_id': intent['original_replay_job_id'], 'status': 'cancelled_never_started',
            'proof': intent['after']})
        entry.update(replay_job_id=intent['vast_job_id'], replay_backend='vast', status='replay_queued')
        entry.pop('seen_queue_absent', None)
        intent['status'] = 'ready_to_activate'; save()
    if intent['status'] == 'ready_to_activate':
        require(activate(job, root) == intent['vast_job_id'], 'replay activation identity differs')
        intent.update(status='active', activated_at_unix=time.time()); save(); return True
    return False
