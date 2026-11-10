"""Durable, backend-independent intents for already-authored fresh GPU jobs.

Controllers reserve indices and author immutable inputs before enqueueing. One
dispatcher claims each intent. An interrupted submission stays ambiguous until
its ledger is reconciled; it is never blindly retried.
"""
from __future__ import annotations
import argparse
import csv
import getpass
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from scripts.campaign_20261110.contracts import digest, file_digest, require
from scripts.campaign_20261110.storage import atomic_directory_lock

BACKENDS = ('shiyuc_svl', 'shiyuc_viscam', 'vast_5090', 'vast_4090', 'minyeong_svl', 'minyeong_viscam')
LEDGER_COLS = ['submitted', 'key', 'alias', 'start', 'n', 'job_tag', 'job_dir', 'flow', 'code', 'jobid',
               'acct', 'part', 'gres', 'time_s', 'sample_limit_s', 'max_attempts', 'demo_limit_s', 'kcollect', 'tag']


def prepare_intent_ledger(root, token):
    """An intent's ledger cannot inherit a later intent's next-index floor."""
    require(token and token.replace('_', '').replace('-', '').isalnum(), 'unsafe ledger token')
    directory = root / 'manifests/dispatch_ledgers' / token
    directory.mkdir(parents=True, exist_ok=True)
    for path in (directory.parent, directory):
        if path.stat().st_uid == os.getuid(): os.chmod(path, 0o2770)
    ledger = directory / 'allocations.tsv'
    if not ledger.exists():
        fd, temporary = tempfile.mkstemp(prefix='.ledger.', dir=directory)
        os.fchmod(fd, 0o660)
        with os.fdopen(fd, 'w') as stream:
            stream.write('\t'.join(LEDGER_COLS) + '\n'); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, ledger)
        except FileExistsError:
            pass
        finally:
            os.unlink(temporary)
    with ledger.open() as stream:
        require(stream.readline().rstrip('\n').split('\t') == LEDGER_COLS, 'intent ledger schema differs')
    alias = root / 'manifests' / ('dispatch_' + token + '_allocations.tsv')
    if alias.is_symlink():
        require(alias.resolve() == ledger.resolve(), 'intent ledger alias differs')
    elif alias.exists():
        raise ValueError('intent ledger alias is not a symlink')
    else:
        try:
            alias.symlink_to(ledger)
        except FileExistsError:
            require(alias.is_symlink() and alias.resolve() == ledger.resolve(),
                    'concurrent intent ledger alias differs')
    return ledger


def verify_dry_indices(stdout, identity):
    values = {name: re.findall(r'^\s*' + name + r'=(\d+)\s*$', stdout, re.MULTILINE)
              for name in ('START', 'N')}
    require(values['START'] == [str(identity['start'])] and values['N'] == [str(identity['n'])],
            'native dry-run shifted the globally reserved interval')


def shared_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        os.fchmod(fd, 0o660)
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def submission_log(receipt_path, suffix, content):
    """Retries never replace another account's existing submission evidence."""
    target = receipt_path.with_suffix(suffix)
    if target.exists():
        target = target.with_name(receipt_path.stem + '.attempt_' + uuid.uuid4().hex[:12] + suffix)
    with target.open('x') as stream:
        os.fchmod(stream.fileno(), 0o660)
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    return target


def option(command, name, default=None):
    indices = [i for i, item in enumerate(command) if item == name]
    require(len(indices) <= 1, 'repeated command option: ' + name)
    return command[indices[0] + 1] if indices else default


def set_option(command, name, value):
    if name in command:
        command[command.index(name) + 1] = str(value)
    else:
        command.extend([name, str(value)])


def input_identity(root, command):
    require(isinstance(command, list) and all(isinstance(x, str) for x in command), 'command must be an argv array')
    require(len(command) > 4 and Path(command[1]).name == 'submit_fast.py' and command[2] == 'submit', 'only prepared trainfast submissions are supported')
    require('--dry-run' not in command and '--shiyuc' not in command, 'dispatcher owns dry-run and write-path routing')
    require(Path(option(command, '--root')).resolve() == root, 'command has a different campaign root')
    code = Path(option(command, '--code')).resolve()
    require(code.is_relative_to(root / 'code'), 'runtime is not an owned immutable snapshot')
    start, count = int(option(command, '--base-index')), int(command[4])
    reservations = json.loads((root / 'manifests/family_index_reservations.json').read_text())['reservations']
    require(any(int(r['start']) <= start and start + count <= int(r['start']) + int(r['n']) for r in reservations.values()), 'indices are not globally reserved')
    files = {str(code / '.snapshot_commit'): file_digest(code / '.snapshot_commit')}
    env = dict(command[i + 1].split('=', 1) for i, item in enumerate(command) if item == '--extra-env')
    require('--no-goal-override' in env.get('ISGF_COLLECT_EXTRA', ''), 'fresh production must preserve exact goals')
    for key in ('ISPATIALGYM_CAMPAIGN_CONFIG', 'ISPATIALGYM_CAMPAIGN_GOAL_PLAN', 'ISPATIALGYM_CAMPAIGN_GOAL_QUEUE'):
        if env.get(key):
            path = Path(env[key]).resolve()
            require(path.is_relative_to(root), 'campaign input outside owned root')
            files[str(path)] = file_digest(path)
    require('ISPATIALGYM_CAMPAIGN_CONFIG' in env or 'ISPATIALGYM_CAMPAIGN_GOAL_QUEUE' in env, 'annotated config required')
    if env.get('EVAL_INDEX'):
        path = Path(env['EVAL_INDEX']).resolve()
        require(path.is_relative_to(root) and path.is_file(), 'fresh evaluation index must be owned')
        raw = json.loads(path.read_text())
        require(raw.get('schema') == 'isg-instance-index-v1' and raw.get('instances') == [] and raw.get('n') == 0,
                'campaign dispatcher requires a fresh empty evaluation index')
        files[str(path)] = file_digest(path)
    if env.get('TEMPLATE'):
        template_dir = Path(env['TEMPLATE']).resolve()
        if template_dir.is_relative_to(root):
            for path in template_dir.glob('*.json'):
                files[str(path)] = file_digest(path)
    if env.get('ISPATIALGYM_CAMPAIGN_GOAL_QUEUE'):
        queue = json.loads(Path(env['ISPATIALGYM_CAMPAIGN_GOAL_QUEUE']).read_text())
        for path, expected in [(queue['config_path'], queue['config_sha256'])] + [(entry['path'], entry['sha256']) for entry in queue['entries']]:
            require(Path(path).resolve().is_relative_to(root), 'queue input outside campaign')
            require(file_digest(path) == expected, 'queue immutable input changed')
            files[path] = expected
    return {'key': command[3], 'start': start, 'n': count, 'code': str(code),
            'tag': option(command, '--tag', 'fast'), 'files': files}


def enqueue(root, token, command, priority=50, backends=BACKENDS):
    root = Path(root).resolve()
    require(token and token.replace('_', '').replace('-', '').isalnum(), 'unsafe dispatch token')
    require(backends and set(backends) <= set(BACKENDS), 'unknown dispatch backend')
    identity = input_identity(root, command)
    prepare_intent_ledger(root, token)
    reservation_path = root / 'manifests/family_index_reservations.json'
    if reservation_path.stat().st_uid == os.getuid():
        os.chmod(reservation_path, reservation_path.stat().st_mode | 0o040)
    # These are new campaign inputs, never provider assets or historical data.
    # Grant the already-authorized private lab group read access without changing
    # bytes. The queue's writable event journal is staged inside the fresh job.
    if any(backend.startswith('shiyuc_') for backend in backends):
        inputs = list(identity['files'])
        seen = set()
        while inputs:
            path = Path(inputs.pop()).resolve()
            if path in seen: continue
            seen.add(path)
            require(path.is_relative_to(root), 'shared input outside owned campaign')
            stat = path.stat()
            if stat.st_uid == os.getuid(): os.chmod(path, stat.st_mode | 0o040)
            if path.suffix == '.json':
                value = json.loads(path.read_text())
                def paths(item):
                    if isinstance(item, dict):
                        for child in item.values(): yield from paths(child)
                    elif isinstance(item, list):
                        for child in item: yield from paths(child)
                    elif isinstance(item, str) and item.startswith(str(root) + '/'):
                        candidate = Path(item)
                        if candidate.suffix == '.json' and candidate.is_file(): yield str(candidate)
                inputs.extend(paths(value))
    path = root / 'dispatch/ready' / (token + '.json')
    payload = {'schema': 'isg-ready-command-v1', 'token': token, 'command': command,
               'command_sha256': digest(command), 'identity': identity, 'priority': int(priority), 'backends': list(backends)}
    if path.exists():
        require(json.loads(path.read_text()) == payload, 'dispatch token already binds different work')
    else:
        shared_json(path, payload)
    return path


def routed_command(root, payload, backend):
    require(backend in payload['backends'], 'backend not authorized for this intent')
    command = list(payload['command'])
    require(input_identity(root, command) == payload['identity'], 'immutable command inputs changed')
    ledger = root / 'manifests/dispatch_ledgers' / payload['token'] / 'allocations.tsv'
    alias = root / 'manifests' / ('dispatch_' + payload['token'] + '_allocations.tsv')
    require(ledger.is_file() and alias.is_symlink() and alias.resolve() == ledger.resolve(),
            'intent ledger needs preparation by the primary dispatcher')
    set_option(command, '--ledger', ledger)
    environment = dict(os.environ)
    if backend.startswith('vast_'):
        command = [str(root / 'vast/tool/vsubmit')] + command
        environment.update(VAST_GPU='RTX ' + backend.split('_')[1], VAST_QUEUE_ROOT=str(root / 'vast'))
    else:
        user, partition = backend.split('_')
        require(getpass.getuser() == user, 'dispatcher is running under the wrong cluster user')
        set_option(command, '--acct', 'vision' if partition == 'svl' else 'viscam')
        set_option(command, '--part', partition)
        set_option(command, '--gres', 'titanrtx' if partition == 'svl' else '')
        environment['ISG_DISPATCH_CAMPAIGN_ROOT'] = str(root)
        environment['ISG_DISPATCH_REAL_SBATCH'] = shutil.which('sbatch') or '/usr/bin/sbatch'
        environment['PATH'] = str(Path(__file__).parent / 'infra_dispatch_bin') + os.pathsep + environment['PATH']
        if user == 'shiyuc':
            execution = root / 'out/shiyuc'
            private = Path('/vision/u/shiyuc/minyeong/campaign_20261110')
            require(os.access(execution, os.W_OK), 'shared campaign execution root not writable')
            set_option(command, '--root', execution)
            overrides = {'ISGF_ROOT': str(root), 'ISGF_SAMPLING_CACHE': str(private / 'cache/dispatch' / payload['token']),
                         'ISGF_ASSET_CACHE': str(private / 'asset_cache'), 'ISGF_JB_NFS': str(private / 'jobs'),
                         'KIT_TPL_NFS_ROOT': str(private / 'kit_templates'), 'KIT_TEX_STORE': str(private / 'kit_textures'),
                         'ISGF_EXP_PATH': str(private / 'isaacsim_overlay/apps'), 'ISGF_GIT_SAFE_ALL': '1',
                         'ISGF_PREDEMO_ROOT': str(root), 'ISPATIALGYM_FAST_BAND_SHARE': '0'}
            for key, value in overrides.items():
                command.extend(['--extra-env', key + '=' + value])
    return command, environment


def backend_capacity(backend, rows):
    """Bound queued plus running GPU work to the user's actual partition caps."""
    if backend.startswith('vast_'): return 8
    user, partition = backend.split('_')
    own = [row for row in rows if row[0] == user]
    # These are PARTITION QoS limits; a job's %q is normally "normal".
    # Pending dual-partition jobs count against both until Slurm picks one.
    # CPU services consume svl job slots but never GPU TRES; viscam has no 56-job limit.
    part = [row for row in own if partition in row[1].split(',')]
    gpus = sum(int(row[3].rsplit(':', 1)[1]) for row in part if 'gpu:' in row[3])
    running = sum(row[2] in ('RUNNING','COMPLETING') for row in part)
    limits = [8, (48 if partition == 'svl' else 30) - gpus]
    if partition == 'svl': limits.extend([100 - len(part), 56 - running])
    return max(0, min(limits))


def publish_owned_outputs(root):
    """Give the authorized lab group read access to newly written job outputs."""
    directory = root / 'out/shiyuc/out'
    if not directory.is_dir() or getpass.getuser() != 'shiyuc': return
    for current, dirs, files in os.walk(directory, followlinks=False):
        for path in [Path(current)] + [Path(current) / name for name in files]:
            if path.is_symlink(): continue
            try: stat = path.stat()
            except FileNotFoundError: continue
            if stat.st_uid != os.getuid(): continue
            target = stat.st_mode | (0o050 if path.is_dir() else 0o040)
            if target != stat.st_mode: os.chmod(path, target)


def find_allocation(root, identity):
    matches = []
    for path in (root / 'manifests').glob('*allocations.tsv'):
        with path.open() as stream:
            matches.extend({**r, 'ledger': str(path)} for r in csv.DictReader(stream, delimiter='\t')
                           if r['key'] == identity['key'] and int(r['start']) == identity['start'])
    require(len(matches) <= 1, 'ambiguous allocation; manual reconciliation required')
    if matches:
        row = matches[0]
        require(int(row['n']) == identity['n'] and row['code'] == identity['code'] and row['tag'] == identity['tag'], 'allocated identity differs')
        return row
    return None


def activate_vast(root, payload, row, timeout=25):
    """Reconcile queue activation after submission without submitting again."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            with atomic_directory_lock(root / 'vast/config.dispatch.lockdir'):
                path = root / 'vast/config.json'
                config = json.loads(path.read_text())
                ids = config.setdefault('dispatch_job_ids', [])
                if row['jobid'] not in ids: ids.append(row['jobid'])
                priority = config.setdefault('dispatch_priority_job_ids', [])
                if payload['priority'] < 50 and row['jobid'] not in priority: priority.append(row['jobid'])
                shared_json(path, config)
            return
        except FileExistsError:
            if time.monotonic() >= deadline: raise
            time.sleep(.25)


def execute(root, path, backend):
    root, path = Path(root).resolve(), Path(path)
    payload = json.loads(path.read_text())
    receipt_path = root / 'dispatch/receipts' / (payload['token'] + '.json')
    with atomic_directory_lock(root / 'dispatch/locks' / (payload['token'] + '.lockdir')):
        prior = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        row = find_allocation(root, payload['identity'])
        if row:
            if prior.get('backend', backend).startswith('vast_'):
                activate_vast(root, payload, row)
            if prior.get('error'):
                prior.setdefault('reconciliation_history', []).append({'at': time.time(), 'previous_error': prior.pop('error')})
            shared_json(receipt_path, {**prior, 'status': 'submitted', 'allocation': row, 'reconciled_at': time.time()})
            return
        require(prior.get('status') not in ('submitting', 'blocked', 'submitted'), 'prior outcome needs reconciliation')
        command, environment = routed_command(root, payload, backend)
        receipt = {'schema': 'isg-dispatch-receipt-v1', 'token': payload['token'], 'status': 'submitting',
                   'backend': backend, 'started_at': time.time(), 'command_sha256': payload['command_sha256'],
                   'routed_command': command, 'identity': payload['identity']}
        if prior:
            archived = receipt_path.with_name(receipt_path.stem + '.attempt_' + uuid.uuid4().hex[:12] + '.receipt.json')
            shared_json(archived, prior)
            receipt['previous_attempt'] = {'path':str(archived), 'sha256':file_digest(archived)}
        shared_json(receipt_path, receipt)
        try:
            for dry in (True, False):
                result = subprocess.run(command + (['--dry-run'] if dry else []), env=environment,
                                        capture_output=True, text=True, timeout=300)
                log = submission_log(receipt_path, '.dry.log' if dry else '.submit.log', result.stdout + result.stderr)
                receipt.setdefault('submission_logs', {})['dry' if dry else 'submit'] = str(log)
                require(result.returncode == 0, 'submission failed; inspect ' + str(log))
                if dry: verify_dry_indices(result.stdout, payload['identity'])
            row = find_allocation(root, payload['identity'])
            require(row is not None, 'submit completed without ledger; reconcile before retry')
            receipt.update(status='submitted', allocation=row, completed_at=time.time())
            if backend.startswith('vast_'):
                activate_vast(root, payload, row)
        except Exception as exc:
            receipt.update(status='blocked', error=str(exc), failed_at=time.time())
            raise
        finally:
            shared_json(receipt_path, receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--backend', choices=BACKENDS, required=True)
    p.add_argument('--once', action='store_true')
    p.add_argument('--interval', type=float, default=20)
    a = p.parse_args(); os.umask(0o007)
    while True:
        publish_owned_outputs(a.root)
        rows = []
        if not a.backend.startswith('vast_'):
            query = subprocess.run(['squeue', '-r', '-u', 'minyeong,shiyuc', '-h', '-o', '%u|%P|%T|%b|%q'],
                                   capture_output=True, text=True, timeout=40, check=True)
            rows = [line.split('|') for line in query.stdout.splitlines() if len(line.split('|')) == 5]
        allowance = backend_capacity(a.backend, rows)
        candidates = []
        for path in (a.root / 'dispatch/ready').glob('*.json'):
            payload = json.loads(path.read_text())
            if getpass.getuser() == 'minyeong':
                prepare_intent_ledger(a.root, payload['token'])
            receipt = a.root / 'dispatch/receipts' / path.name
            prior = json.loads(receipt.read_text()) if receipt.exists() else {}
            activation_retry = (prior.get('status') == 'blocked' and prior.get('allocation')
                                and prior.get('backend') == a.backend and a.backend.startswith('vast_'))
            if a.backend in payload['backends'] and (not receipt.exists() or activation_retry):
                candidates.append((payload['priority'], path.name, path))
        for _, _, path in sorted(candidates)[:allowance]:
            try: execute(a.root, path, a.backend)
            except FileExistsError: pass
            except Exception as exc: print(json.dumps({'intent': str(path), 'error': str(exc)}), flush=True)
        if a.once: break
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
