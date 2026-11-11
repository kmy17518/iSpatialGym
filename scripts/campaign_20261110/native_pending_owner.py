"""Exclusive pending recovery for registered, already-pinned native dependencies.

This owner accepts a single primary-account allocation and an existing verified
``native_inputs.json``. It does not infer portability from a shell command, or
authorize a new sampler input. Secondary-private relocation is a separate proof.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.campaign_20261110 import overflow_readiness, vast_native_dependency as native
from scripts.campaign_20261110.infra_vast import cluster_return_capacity
from scripts.campaign_20261110.replay_migration import never_started, scheduler_proof
from scripts.campaign_20261110.storage import atomic_directory_lock


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(path):
    return {'path': str(path), 'sha256': native.sha(path)}


def read_reference(ref):
    require(native.sha(ref['path']) == ref['sha256'], 'registered input changed')
    return json.loads(Path(ref['path']).read_text())


def verify_registration(path, root):
    row = json.loads(Path(path).read_text())
    require(row['schema'] == 'isg-native-pending-owner-v1', 'unsupported owner registration')
    require(row['account'] == 'minyeong' and str(row['job_id']).isdigit(), 'unsupported allocation owner')
    require(row['cluster_routes'] == ['minyeong_svl', 'minyeong_viscam'], 'unverified cluster routes')
    require(row.get('secondary_transport_blocker'), 'secondary transport restriction must be explicit')
    restriction = read_reference(row['secondary_transport_proof'])
    require(restriction.get('account') == 'shiyuc' and restriction.get('compatible') is False
            and restriction.get('failed_paths'), 'secondary restriction lacks exact unreadable-path evidence')
    submission = read_reference(row['submission'])
    require(submission['returncode'] == 0 and str(submission['job_id']) == row['job_id'],
            'successful allocation receipt differs')
    wrapper = Path(row['wrapper']['path'])
    require(wrapper.is_relative_to(root) and native.sha(wrapper) == row['wrapper']['sha256'],
            'registered wrapper changed')
    require(submission['command'][-1] == str(wrapper), 'submitted wrapper differs')
    command = submission['command']
    for flag in ('--array', '--dependency', '--nodelist'):
        require(not any(a == flag or a.startswith(flag+'=') for a in command), 'unsupported scheduler restriction')
    require('--gres=gpu:titanrtx:1' in command and '--exclude=svl9' in command,
            'resource probe must match exact GPU/exclusion declaration')
    for flag, value in (('-c', '8'), ('--mem', '48G'), ('-p', 'svl,viscam')):
        require(flag in command and command[command.index(flag)+1] == value,
                'resource probe differs from allocation')
    manifest = read_reference(row['native_inputs'])
    job = Path(manifest['job_dir'])
    require(row['native_inputs']['path'] == str(job/'native_inputs.json'), 'native manifest locator differs')
    checked = native.verify(job, root)
    require(checked == manifest, 'native verification differs')
    return row, job


def identity(proof, row):
    info = proof['scheduler']
    require(proof['job_id'] == row['job_id'] == info.get('JobId'), 'scheduler job identity differs')
    require(info['UserId'].split('(')[0] == row['account'], 'scheduler account differs')
    require(info['Command'] == row['wrapper']['path'], 'scheduler wrapper differs')
    require(info.get('ArrayJobId') in (None, '', '(null)'), 'array owner unsupported')
    require(info.get('Dependency') in (None, '', '(null)', 'None'), 'dependent native allocation')
    require(int(info.get('Restarts', '0')) == 0, 'restarted allocation')


def output_empty(job):
    require(not (job/'result').exists(), 'native runtime already created output')


def funded_scheduler(job, scheduler):
    manifest = json.loads((job/'native_inputs.json').read_text())
    seconds = native.verified_bounds(manifest)['allocation_seconds']
    clock = f'{seconds//3600:02d}:{seconds%3600//60:02d}:{seconds%60:02d}'
    return {**scheduler, 'TimeLimit': clock}


def recover(path, root, *, proof_fn=scheduler_proof, capacity_fn=cluster_return_capacity.probe,
            cancel_fn=None, activate_fn=None):
    """One locked cycle; the durable cancellation proof survives any later error."""
    root = Path(root).resolve()
    path = Path(path).resolve()
    require(path.is_relative_to(root/'infra/capacity_recovery/native_pending_owner'), 'registration outside owner registry')
    with atomic_directory_lock(path.parent/'owner.lockdir'):
        row, job = verify_registration(path, root)
        state_path = path.parent/'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {'status': 'registered'}
        if state['status'] in ('active', 'cluster_started_or_terminal'):
            return state
        queued_receipt = job/'vast_submission.json'
        if state['status'] != 'cancelled_never_started':
            require(not queued_receipt.exists(), 'native dependency already has a different queue owner')
            output_empty(job)
        proof = proof_fn(row['job_id'])
        identity(proof, row)
        if state['status'] != 'cancelled_never_started':
            if not never_started(proof, 'PENDING') or proof['scheduler']['JobState'] != 'PENDING':
                if state.get('cancel_requested') and never_started(proof, 'CANCELLED'):
                    identity(state['before'], row)
                    require(never_started(state['before'], 'PENDING')
                            and state['registration'] == reference(path),
                            'interrupted cancellation source binding differs')
                else:
                    state.update(status='cluster_started_or_terminal', last_proof=proof, observed_at=time.time())
                    native.write(state_path, state)
                    return state
            else:
                capacity = capacity_fn()  # Actual two-account scheduler query, never periodic inventory alone.
                budget = overflow_readiness.check(root, funded_scheduler(job, proof['scheduler']))
                state.update(cancel_requested=True, before=proof, capacity=capacity, budget=budget,
                             registration=reference(path), observed_at=time.time())
                native.write(state_path, state)
                again = proof_fn(row['job_id'])
                identity(again, row)
                if not never_started(again, 'PENDING') or again['scheduler']['JobState'] != 'PENDING':
                    state.update(status='cluster_started_or_terminal', last_proof=again)
                    native.write(state_path, state)
                    return state
                output_empty(job)
                (cancel_fn or (lambda jid: subprocess.run(['scancel', '--state=PENDING', jid],
                    check=True, timeout=30)))(row['job_id'])
                proof = proof_fn(row['job_id'])
                identity(proof, row)
            require(never_started(proof, 'CANCELLED') and proof['scheduler']['JobState'] == 'CANCELLED',
                    'cancellation lacks exact zero-runtime confirmation; replacement stays inactive')
            output_empty(job)
            state.update(status='cancelled_never_started', after=proof, cancelled_at=time.time())
            native.write(state_path, state)
        else:
            identity(state['before'], row)
            identity(state['after'], row)
            require(never_started(state['before'], 'PENDING') and never_started(state['after'], 'CANCELLED')
                    and never_started(proof, 'CANCELLED'), 'saved cancellation proof differs')
            if queued_receipt.exists():
                queued = json.loads(queued_receipt.read_text())
                jid = str(queued['id'])
                require(native.sha(root/'vast/queue'/(jid+'.json')) == queued['record_sha256'],
                        'saved native queue bytes differ')
                queue = json.loads((root/'vast/queue'/(jid+'.json')).read_text())
                require(queue['job_dir'] == str(job) and queue['native_inputs_sha256'] == row['native_inputs']['sha256'],
                        'saved native queue source differs')
                config = json.loads((root/'vast/config.json').read_text())
                if jid in list(map(str, config.get('dispatch_job_ids', []))):
                    state.update(status='active', vast_job_id=jid, recovered_after_activation=True,
                                 activated_at=time.time(), original_job_id=row['job_id'])
                    native.write(state_path, state)
                    return state
            output_empty(job)
        native.verify(job, root)
        jid = native.enqueue(job, root)
        if activate_fn is None:
            sys.path.insert(0, str(root/'tools'))
            from short_funded_activation_r1 import activate
            activate_fn = activate
        activation = activate_fn(root, jid, funded_scheduler(job, state['before']['scheduler']))
        state.update(status='active', vast_job_id=jid, activation=activation, activated_at=time.time(),
                     minimum_pending_seconds=0, original_job_id=row['job_id'], native_inputs=row['native_inputs'])
        native.write(state_path, state)
        return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--every', type=float, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    require(args.every > 0, 'positive observation interval required')
    directory = args.root/'infra/capacity_recovery/native_pending_owner'
    directory.mkdir(parents=True, exist_ok=True)
    while not (directory/'STOP').exists():
        report = {'observed_at': time.time(), 'rows': [], 'errors': [], 'minimum_pending_seconds': 0}
        for path in sorted(directory.glob('*/registration.json')):
            try:
                report['rows'].append({'registration': reference(path), 'state': recover(path, args.root)})
            except Exception as error:
                report['errors'].append({'registration': str(path), 'error': repr(error)})
        report['finished_at'] = time.time()
        native.write(directory/'current.json', report)
        if args.once:
            return
        time.sleep(args.every)


if __name__ == '__main__':
    main()
