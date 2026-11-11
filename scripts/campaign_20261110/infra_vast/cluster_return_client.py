"""Persistent request/ack clients for returning untouched typed maps to clusters.

Selection is serialized across both accounts. Only the coordinator can remove a
Vast owner; account clients submit only after its exact durable acknowledgement.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import time

from scripts.campaign_20261110.storage import atomic_directory_lock
from scripts.campaign_20261110.infra_vast import cluster_return as protocol
from scripts.campaign_20261110.infra_vast import cluster_return_capacity as capacity_probe
from scripts.campaign_20261110.infra_vast import cluster_return_release as release
from scripts.campaign_20261110.infra_vast.campaign_inputs import dispatch_order


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return {'path': str(path), 'sha256': protocol.digest(path)}


def account_root(root, account):
    return root if account == 'minyeong' else Path('/vision/u/shiyuc/minyeong') / root.name


def select(root):
    base = root / 'vast/cluster_return'
    base.mkdir(parents=True, exist_ok=True)
    with atomic_directory_lock(base / 'selector.lockdir'):
        now = time.time()
        capacity = capacity_probe.probe()
        metrics = read(root / 'vast/metrics.json')
        now = time.time()
        if capacity['status'] != 'observed' or not 0 <= now-metrics['observed_at_unix'] <= 120:
            return {'status': 'stale_observation', 'new_requests': 0}
        capacity_path = base/'capacity'/('probe_'+str(time.time_ns())+'.json')
        protocol.write(capacity_path, capacity)
        slots = capacity['compatible_available_slots']
        states = [read(p) for p in (root / 'vast/state/jobs').glob('*.json')]
        claims = [s['cluster_return'] for s in states if s.get('cluster_return', {}).get('request')]
        outstanding = protocol.unrepresented_claims(claims, capacity['observed_at'])
        pending_requests = {}
        for p in (base / 'requests').glob('*.json'):
            if not (base / 'acknowledgements' / p.name).exists() and not (base / 'rejected' / p.name).exists():
                r = read(p)
                if r['expires_at'] >= now:
                    pending_requests[str(r['job_id'])] = r
        available = max(0, slots-len(outstanding | set(pending_requests)))
        queued = dispatch_order(states, read(root / 'vast/config.json'))
        # Paid ready workers keep useful compatible demand while remaining units
        # return to actually available cluster capacity, without a fleet ceiling.
        available = min(available, max(0, len(queued)-int(metrics.get('worker_idle_ready_gpus') or 0)))
        staged = {p.stem for p in (root / 'vast/state/staging').glob('*.json')
                  if read(p).get('status') in ('preparing', 'ready', 'assigning')}
        chosen = []
        errors = []
        for job in queued:
            if len(chosen) >= available:
                break
            jid = str(job['id'])
            if jid in staged or jid in pending_requests or job.get('tries') or job.get('instance') or job.get('history') or job.get('cluster_return_releases'):
                continue
            directory = Path(job['job_dir'])
            if job.get('job_kind') != 'native_dependency' or not directory.is_relative_to(root / 'out/vast_map_prebuild'):
                continue
            try:
                inputs = read(directory / 'native_inputs.json')
                index = inputs['unit_index']
                account = 'shiyuc' if index % 2 else 'minyeong'
                if sum(r['account'] == account for r in chosen) >= capacity['account_slots'].get(account, 0):
                    continue
                target = account_root(root, account) / 'training_preparation' / Path(inputs['source_plan']).parent.name / 'build_dispatch_r1'
                prepared = target / 'prepared.json'
                unit = next(u for u in read(prepared)['units'] if u['index'] == index)
                if any((Path(unit['output']) / name).exists() for name in ('started.json', 'terminal.json')):
                    continue
                if protocol.digest(unit['script']) != unit['script_sha256']:
                    raise ValueError('owned unit script changed')
                chosen.append({'schema': protocol.SCHEMA, 'campaign_id': root.name, 'job_id': jid,
                               'account': account, 'observed_at': now, 'expires_at': now+180,
                               'queue': ref(root / 'vast/queue' / (jid+'.json')),
                               'native_inputs': ref(directory / 'native_inputs.json'),
                               'source_plan': ref(inputs['source_plan']), 'unit_index': index,
                               'prepared': ref(prepared), 'unit': unit,
                               'wrapper': ref(target / 'pinned.sh'), 'runner': ref(target / 'run.py')})
            except Exception as error:
                errors.append({'job_id': jid, 'error': str(error)})
        if chosen:
            name = str(time.time_ns())
            proof_path = base / 'capacity' / (name+'.json')
            proof = {'observed_at': capacity['observed_at'], 'source': ref(capacity_path),
                     'compatible_available_slots': slots,
                     'selected_job_ids': [r['job_id'] for r in chosen],
                     'selected_accounts': {r['job_id']: r['account'] for r in chosen},
                     'prior_unrepresented_claims': sorted(outstanding),
                     'paid_idle_workers_reserved': int(metrics.get('worker_idle_ready_gpus') or 0)}
            protocol.write(proof_path, proof)
            for request in chosen:
                request['capacity'] = ref(proof_path)
                protocol.write(base / 'requests' / (request['job_id']+'_'+name+'.json'), request)
        return {'status': 'observed', 'new_requests': len(chosen), 'errors': errors,
                'compatible_inventory_slots': slots, 'unrepresented_claims': sorted(outstanding),
                'eligible_queued': len(queued), 'requested_ids': [r['job_id'] for r in chosen]}


def submit(root, account):
    owned = account_root(root, account)
    base = root / 'vast/cluster_return'
    output = owned / 'infra/capacity_recovery/cluster_return_client_r1'
    output.mkdir(parents=True, exist_ok=True)
    report = {'submitted': [], 'errors': [], 'preserved_submission_intents': [], 'recoveries': []}
    audit = protocol.ImmutableFileAudit()
    with atomic_directory_lock(output / 'submit.lockdir'):
        for path in (base / 'acknowledgements').glob('*.json'):
            ack = read(path)
            if ack['account'] != account:
                continue
            receipt = output / (path.stem+'_submitted.json')
            attempt = output / (path.stem+'_attempt.json')
            try:
                if release.request_path(root, path).exists():
                    continue
                state = read(root / 'vast/state/jobs' / (ack['job_id']+'.json'))
                if state['status'] != 'cluster_reserved' or state['cluster_return']['request'] != ack['request']:
                    continue
                if receipt.exists():
                    if not any((Path(ack['unit']['output'])/n).exists() for n in ('started.json', 'terminal.json')):
                        report['recoveries'].append(release.recover_pending(root, path, ack, receipt))
                    continue
                if attempt.exists():
                    if not release.recover_withdrawal(root, path, attempt):
                        report['preserved_submission_intents'].append(str(attempt))
                    continue
                request = read(ack['request']['path'])
                if protocol.digest(ack['request']['path']) != ack['request']['sha256']:
                    raise ValueError('acknowledged request changed')
                unit = ack['unit']
                if protocol.digest(ack['prepared']['path']) != ack['prepared']['sha256'] or protocol.digest(unit['script']) != unit['script_sha256']:
                    raise ValueError('exact prepared inputs changed')
                for field in ('wrapper', 'runner'):
                    if audit.sha256(ack[field]['path']) != ack[field]['sha256']:
                        raise ValueError('acknowledged launch code changed')
                preparation = read(ack['prepared']['path'])
                for name, expected in preparation['copied_library_files'].items():
                    if audit.sha256(name) != expected:
                        raise ValueError('private copied native library changed')
                for name, field in (('jobs.json', 'jobs_sha256'), ('anchors.json', 'anchors_sha256')):
                    if audit.sha256(Path(unit['output']) / name) != unit[field]:
                        raise ValueError('exact native map command inputs changed')
                if any((Path(unit['output']) / n).exists() for n in ('started.json', 'terminal.json')):
                    raise ValueError('source unit already executed')
                # Expensive immutable reads stay outside this short cross-account
                # boundary. The second account observes the first sbatch before
                # spending the same last physical slot.
                with atomic_directory_lock(base/'submit_capacity.lockdir'):
                    current = capacity_probe.probe()
                    if current['status'] != 'observed' or current['account_slots'].get(account, 0) <= 0:
                        report['recoveries'].append({'status': 'withdrawn_before_submission',
                            'path': release.withdraw(root, path, ack, attempt, current)})
                        continue
                    wrapper = Path(ack['prepared']['path']).parent / 'pinned.sh'
                    command = ['sbatch', '--parsable', '--nice=1000', '-A', 'vision', '-p', 'svl,viscam',
                               '--exclude=svl9', '--gres=gpu:titanrtx:1', '--array='+str(ack['unit_index']),
                               '-c', '8', '--mem=48G', '--time=02:00:00', '-J', 'isg261110_map_return',
                               '-o', str(output / (path.stem+'_%A_%a.log')), str(wrapper)]
                    value = {'observed_at': time.time(), 'acknowledgement': ref(path), 'job_id_vast': ack['job_id'],
                             'account': account, 'unit': unit, 'command': command, 'wrapper': ref(wrapper),
                             'status': 'submitting', 'new_native_instances': 0, 'fresh_capacity': current}
                    protocol.write(attempt, value)
                    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    value.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
                    if result.returncode:
                        raise RuntimeError('scheduler rejected exact submission: '+result.stderr)
                    jid = result.stdout.strip().split(';')[0]
                    if not jid.isdigit():
                        raise RuntimeError('unrecognized scheduler acknowledgement; preserve intent')
                    value.update(status='submitted', job_id=jid)
                    protocol.write(receipt, value)
                    report['submitted'].append({'job_id': jid, 'vast_job_id': ack['job_id'], 'receipt': str(receipt)})
            except Exception as error:
                report['errors'].append({'acknowledgement': str(path), 'error': str(error)})
    report['immutable_input_audit'] = audit.summary()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('select', 'submit'), required=True)
    parser.add_argument('--account', choices=('minyeong', 'shiyuc'), default='minyeong')
    parser.add_argument('--every', type=int, default=0)
    args = parser.parse_args()
    if os.environ.get('USER') != args.account or (args.mode == 'select' and args.account != 'minyeong'):
        raise SystemExit('account ownership differs')
    owned = account_root(args.root, args.account)
    output = owned / 'infra/capacity_recovery/cluster_return_client_r1'
    while True:
        try:
            value = select(args.root) if args.mode == 'select' else submit(args.root, args.account)
        except Exception as error:
            value = {'error': str(error)}
        value.update(observed_at=time.time(), pid=os.getpid(), mode=args.mode, account=args.account)
        protocol.write(output / (args.mode+'_current.json'), value)
        print(json.dumps(value), flush=True)
        if not args.every:
            break
        time.sleep(args.every)


if __name__ == '__main__':
    main()
