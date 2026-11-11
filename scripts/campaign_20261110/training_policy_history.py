"""Retain exact policy bytes before replacing the mutable campaign pointer."""
import hashlib
import os
from pathlib import Path
import tempfile

from .contracts import canonical_json, require


def preserve(root, payload):
    """Content-address an original byte string; never reconstruct its identity."""
    require(isinstance(payload, bytes), 'policy history requires original bytes')
    sha = hashlib.sha256(payload).hexdigest()
    directory = Path(root) / 'infra/training_production_gate/policy_snapshots'
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (sha + '.json')
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            pass
        require(target.read_bytes() == payload, 'immutable policy snapshot differs')
    finally:
        temporary.unlink()
    return {'path': str(target), 'sha256': sha}


def replace_policy(root, policy, *, expected_previous_sha256):
    """Caller owns the policy lock. Archive both byte strings before replacement."""
    target = Path(root) / 'controllers/training_production_gate.json'
    previous = target.read_bytes()
    require(hashlib.sha256(previous).hexdigest() == expected_previous_sha256,
            'policy changed before history-preserving replacement')
    before = preserve(root, previous)
    payload = (canonical_json(policy) + '\n').encode()
    after = preserve(root, payload)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        require(target.read_bytes() == previous, 'policy changed during replacement')
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {'previous': before, 'current': after}
