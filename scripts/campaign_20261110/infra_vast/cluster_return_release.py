"""Return a disproved cluster reservation to its original, exclusive Vast queue.

Account clients publish durable no-submission or cancelled-never-started evidence.
Only the coordinator releases the owner. Ambiguous submissions remain claimed.
"""
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.campaign_20261110 import overflow_readiness
from scripts.campaign_20261110.replay_migration import scheduler_proof as base_scheduler_proof, never_started
from scripts.campaign_20261110.infra_vast import cluster_return as protocol


def require(value, message):
    if not value:
        raise ValueError(message)


def ref(path):
    return {'path': str(path), 'sha256': protocol.digest(path)}


def read_reference(value):
    require(protocol.digest(value['path']) == value['sha256'], 'release reference changed')
    return json.loads(Path(value['path']).read_text())


def empty(unit):
    output = Path(unit['output'])
    require(not any((output/n).exists() for n in ('started.json', 'terminal.json')),
            'map unit already has execution evidence')


def scheduler_proof(job_id):
    proof = base_scheduler_proof(job_id)
    # Slurm may replace ArrayTaskId with its NO_VAL sentinel after cancelling
    # an unexpanded array parent. Accounting retains the exact singleton set.
    raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', str(job_id),
        '--format=JobIDRaw,JobID%120,State,Start,ElapsedRaw,AllocTRES'], text=True, timeout=30,
        env=dict(os.environ, TZ='UTC', SLURM_TIME_FORMAT='standard'))
    rows = [line.split('|') for line in raw.splitlines() if line.split('|')[0] == str(job_id)]
    require(len(rows) == 1 and len(rows[0]) >= 6, 'ambiguous exact array accounting row')
    row = rows[0]
    proof['accounting'] = {'array_job_id': row[1], 'state': row[2].split()[0].rstrip('+'),
        'start': row[3], 'elapsed_seconds': int(row[4]), 'allocated_tres': row[5]}
    return proof


def identity(proof, ack, receipt, prior=None):
    s = proof['scheduler']
    require(proof['job_id'] == receipt['job_id'] == s['ArrayJobId'], 'scheduler reservation differs')
    if s['ArrayTaskId'] != str(ack['unit_index']):
        require(s['ArrayTaskId'] == '4294967294' and prior is not None,
                'scheduler unit differs')
        identity(prior, ack, receipt)  # Must have the exact real index before cancellation.
        require(never_started(prior, 'PENDING') and never_started(proof, 'CANCELLED')
                and s['JobState'] == 'CANCELLED', 'array sentinel lacks never-started boundary')
        expected = receipt['job_id']+'_'+str(ack['unit_index'])
        bracketed = receipt['job_id']+'_['+str(ack['unit_index'])+']'
        require(proof['accounting'].get('array_job_id') in (expected, bracketed),
                'array sentinel lacks exact singleton accounting identity')
    require(s['UserId'].split('(')[0] == ack['account'], 'scheduler account differs')
    require(s['JobName'] == 'isg261110_map_return' and s['Command'] == ack['wrapper']['path'],
            'scheduler command differs from exact acknowledged wrapper')
    require(s.get('Dependency') in ('(null)', 'None', ''), 'map reservation has a dependency')
    require(int(s.get('Restarts', '0')) == 0, 'map reservation has restarted')


def request_path(root, ack_path):
    return root / 'vast/cluster_return/releases' / ack_path.name


def withdraw(root, ack_path, ack, attempt, capacity):
    """Called under the account submit lock, before any scheduler attempt exists."""
    require(not attempt.exists(), 'submission intent already exists')
    empty(ack['unit'])
    tombstone = {'status': 'withdrawn_before_submission', 'observed_at': time.time(),
                 'acknowledgement': ref(ack_path), 'capacity': capacity}
    protocol.write(attempt, tombstone)  # Old and new clients both preserve this intent.
    path = request_path(root, ack_path)
    protocol.write(path, {'kind': 'not_submitted', 'observed_at': time.time(),
                         'acknowledgement': ref(ack_path), 'attempt': ref(attempt)})
    return str(path)


def recover_withdrawal(root, ack_path, attempt):
    value = json.loads(attempt.read_text())
    if value.get('status') != 'withdrawn_before_submission':
        return False
    require(value['acknowledgement'] == ref(ack_path), 'withdrawal acknowledgement differs')
    protocol.write(request_path(root, ack_path), {'kind': 'not_submitted', 'observed_at': time.time(),
                   'acknowledgement': ref(ack_path), 'attempt': ref(attempt)})
    return True


def recover_pending(root, ack_path, ack, receipt_path):
    """Never cancel a running/started job. Scheduler cancellation filters atomically."""
    receipt = json.loads(receipt_path.read_text())
    require(receipt['acknowledgement'] == ref(ack_path), 'submitted acknowledgement differs')
    require(receipt['job_id_vast'] == ack['job_id'] and receipt['unit'] == ack['unit'],
            'submitted map identity differs')
    empty(ack['unit'])
    proposal_path = receipt_path.with_name(ack_path.stem+'_cancel_intent.json')
    before = scheduler_proof(receipt['job_id'])
    if proposal_path.exists():
        proposal = json.loads(proposal_path.read_text())
        require(proposal['receipt'] == ref(receipt_path) and proposal['acknowledgement'] == ref(ack_path),
                'cancellation intent differs')
    else:
        identity(before, ack, receipt)
        if before['scheduler']['JobState'] != 'PENDING' or not never_started(before, 'PENDING'):
            return {'status': 'preserved_execution', 'proof': before}
        # No age threshold. Dependency and unfunded work cannot enter this path.
        funded = overflow_readiness.check(root, before['scheduler'])
        proposal = {'acknowledgement': ref(ack_path), 'receipt': ref(receipt_path),
                    'before': before, 'funded': funded, 'observed_at': time.time()}
        protocol.write(proposal_path, proposal)
    identity(before, ack, receipt, proposal['before'])
    if not never_started(before, 'CANCELLED'):
        if before['scheduler']['JobState'] != 'PENDING' or not never_started(before, 'PENDING'):
            return {'status': 'preserved_execution', 'proof': before}
        empty(ack['unit'])
        subprocess.run(['scancel', '--state=PENDING', receipt['job_id']], check=True, timeout=30)
    after = scheduler_proof(receipt['job_id'])
    identity(after, ack, receipt, proposal['before'])
    if not never_started(after, 'CANCELLED'):
        return {'status': 'awaiting_exact_cancellation', 'proof': after}
    empty(ack['unit'])
    path = request_path(root, ack_path)
    protocol.write(path, {'kind': 'cancelled_never_started', 'observed_at': time.time(),
                         'acknowledgement': ref(ack_path), 'receipt': ref(receipt_path),
                         'cancel_intent': ref(proposal_path), 'after': after})
    return {'status': 'release_requested', 'path': str(path), 'job_id': receipt['job_id']}


def validate(root, path, value, job, active_staging, heartbeat_jobs, remote_jobs):
    ack_path = root/'vast/cluster_return/acknowledgements'/path.name
    require(value['acknowledgement'] == ref(ack_path), 'release acknowledgement differs')
    ack = read_reference(value['acknowledgement'])
    jid = ack['job_id']
    require(str(job['id']) == jid and job['status'] == 'cluster_reserved'
            and job['cluster_return']['request'] == ack['request'], 'exclusive coordinator owner differs')
    require(not job.get('tries') and not job.get('instance') and not job.get('history'),
            'Vast work already assigned')
    require(jid not in active_staging and jid not in heartbeat_jobs and jid not in remote_jobs,
            'another remote owner exists')
    require(read_reference(ack['request'])['unit'] == ack['unit'], 'claimed unit changed')
    empty(ack['unit'])
    owned = root if ack['account'] == 'minyeong' else Path('/vision/u/shiyuc/minyeong')/root.name
    output = owned/'infra/capacity_recovery/cluster_return_client_r1'
    receipt_path = output/(path.stem+'_submitted.json')
    if value['kind'] == 'not_submitted':
        attempt_path = output/(path.stem+'_attempt.json')
        require(value['attempt'] == ref(attempt_path) and not receipt_path.exists(),
                'withdrawal conflicts with a submission receipt')
        attempt = read_reference(value['attempt'])
        require(attempt.get('status') == 'withdrawn_before_submission'
                and attempt['acknowledgement'] == value['acknowledgement'] and 'command' not in attempt,
                'no-submission tombstone missing')
    elif value['kind'] == 'cancelled_never_started':
        require(value['receipt'] == ref(receipt_path), 'cancellation receipt differs')
        receipt = read_reference(value['receipt'])
        require(receipt['acknowledgement'] == value['acknowledgement']
                and receipt['job_id_vast'] == jid and receipt['unit'] == ack['unit'],
                'cancelled submission is for another map')
        intent_path = output/(path.stem+'_cancel_intent.json')
        require(value['cancel_intent'] == ref(intent_path), 'cancellation intent path differs')
        intent = read_reference(value['cancel_intent'])
        require(intent['receipt'] == value['receipt'] and intent['acknowledgement'] == value['acknowledgement'],
                'cancellation intent identity differs')
        for proof in (value['after'], scheduler_proof(receipt['job_id'])):
            identity(proof, ack, receipt, intent['before'])
            require(never_started(proof, 'CANCELLED'), 'cancellation does not prove never started')
    else:
        raise ValueError('unknown release boundary')
    require(not (Path(job['job_dir'])/'result').exists(), 'Vast output already exists')
    return ack


def reconcile(root, states, active_staging, heartbeat_jobs, remote_jobs, save_state):
    base = root/'vast/cluster_return'
    report = {'released': [], 'errors': []}
    for path in sorted((base/'releases').glob('*.json')):
        done = base/'released'/path.name
        if done.exists():
            continue
        try:
            value = json.loads(path.read_text())
            ack = read_reference(value['acknowledgement'])
            job = states[ack['job_id']]
            reference = ref(path)
            prior = next((r for r in job.get('cluster_return_releases', []) if r['release'] == reference), None)
            if prior is None:
                validate(root, path, value, job, active_staging, heartbeat_jobs, remote_jobs)
                prior = {'release': reference, 'claim': job['cluster_return'], 'observed_at': time.time(),
                         'boundary': value['kind'], 'no_further_cluster_return': True}
                job.setdefault('cluster_return_releases', []).append(prior)
                job.pop('cluster_return')
                job['status'] = 'queued'
                save_state(job)  # Durable exclusive release precedes acknowledgement/dispatch.
            protocol.write(done, prior)
            report['released'].append(ack['job_id'])
        except Exception as error:
            report['errors'].append({'release': str(path), 'error': str(error)})
    protocol.write(base/'release_status.json', {**report, 'observed_at': time.time()})
    return report
