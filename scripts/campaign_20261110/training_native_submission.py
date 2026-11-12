"""Durable, exclusive CPU-preflight and allocation ownership for one canary.

Claims deliberately survive errors and process death. An interrupted submit may
have allocated a job; an operator must reconcile that exact intent rather than
delete the claim and submit again.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid


class AlreadyClaimed(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def identity(root, request, expected_sha256, member):
    root, request = Path(root).resolve(), Path(request).resolve()
    if root.name != 'campaign_20261110' or not request.is_relative_to(root):
        raise ValueError('request outside this campaign')
    if digest(request) != expected_sha256:
        raise ValueError('request bytes changed')
    document = json.loads(request.read_text())
    if document.get('campaign_id') != root.name or type(member) is not int or member < 0:
        raise ValueError('campaign/member identity differs')
    row = document['records'][member]
    for name, hash_name in [('config_path', 'config_sha256'),
                            ('authorization_path', 'authorization_sha256'),
                            ('input_manifest', 'input_manifest_sha256')]:
        path = Path(row[name]).resolve()
        if not path.is_relative_to(root) or digest(path) != row[hash_name]:
            raise ValueError('canary identity changed: ' + name)
    return {'campaign_id': root.name, 'request_sha256': expected_sha256,
            'member': member, 'candidate_id': row['candidate_id'],
            'start': row['start'], 'count': row['count'],
            'config_sha256': row['config_sha256'],
            'authorization_sha256': row['authorization_sha256'],
            'input_manifest_sha256': row['input_manifest_sha256'],
            'native_runtime': row['native_runtime'], 'output': row['output']}


class Claim:
    def __init__(self, directory, token, expected_identity):
        self.directory = directory
        self.token = token
        self.identity = expected_identity

    def transition(self, phase, **evidence):
        path = self.directory / 'state.json'
        state = json.loads(path.read_text())
        if state['token'] != self.token or state['identity'] != self.identity:
            raise ValueError('submission claim owner changed')
        allowed = {'claimed': {'preflight_passed', 'blocked'},
                   'preflight_passed': {'submitting', 'blocked'},
                   'submitting': {'submitted', 'submission_unresolved'}}
        if phase not in allowed.get(state['phase'], set()):
            raise ValueError('invalid submission transition')
        if phase == 'submitting' and not evidence.get('command'):
            raise ValueError('exact submit command required before allocation')
        if phase == 'submitted' and (evidence.get('returncode') != 0 or
                                      not str(evidence.get('job_id', '')).isdigit()):
            raise ValueError('successful allocation receipt required')
        state['history'].append({'phase': state['phase'], 'observed_unix': state['observed_unix']})
        state.update(phase=phase, observed_unix=time.time(), **evidence)
        _write(path, state)
        return state


def acquire(root, request, expected_sha256, member):
    """Acquire before any parser or allocation. Never reclaim by elapsed time."""
    root = Path(root).resolve()
    value = identity(root, request, expected_sha256, member)
    key = hashlib.sha256((expected_sha256 + ':' + str(member)).encode()).hexdigest()
    directory = root / 'infra/training_native_submissions' / key
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError:
        # Even a missing state means the creating process might still own it.
        raise AlreadyClaimed('exact member already owned: ' + str(directory)) from None
    token = uuid.uuid4().hex
    _write(directory / 'state.json', {'schema': 'isg-native-submission-claim-v1',
           'identity': value, 'request_path': str(Path(request).resolve()),
           'token': token, 'pid': os.getpid(), 'phase': 'claimed',
           'observed_unix': time.time(), 'history': [],
           'automatic_release_allowed': False})
    return Claim(directory, token, value)
