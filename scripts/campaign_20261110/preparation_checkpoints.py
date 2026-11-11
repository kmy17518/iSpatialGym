"""Publish immutable per-entry preparation evidence before a whole tranche ends.

These records preserve validated inputs; execution still needs the ordinary
ownership, backend, budget and native guards. Batches contain disjoint source
indices, so a later summary must reference them rather than resubmit them.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError('immutable preparation record differs: ' + str(path))
    finally:
        Path(temporary).unlink()
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def entry(directory, *, source, index, identity, native, inputs, audit, hash_file=digest):
    """Persist a single completed check; never manufacture readiness from counts."""
    if not isinstance(index, int) or index < 0 or not identity:
        raise ValueError('invalid source entry identity')
    if hash_file(source['path']) != source['sha256']:
        raise ValueError('source manifest changed')
    if not inputs or not native or audit.get('status') not in ('verified', 'reused', 'blocked'):
        raise ValueError('incomplete native preparation evidence')
    for path, expected in inputs.items():
        if hash_file(path) != expected:
            raise ValueError('entry input changed: ' + path)
    value = {'schema': 'isg-preparation-entry-checkpoint-v1', 'source': source,
             'source_index': index, 'identity': identity, 'native': native,
             'inputs': inputs, 'audit': audit, 'production_authorized': False}
    return immutable_json(Path(directory) / 'entries' / f'{index:06d}.json', value)


def batch(directory, number, entries, *, maximum=12):
    """Publish a bounded contiguous batch; no other batch may own these indices."""
    if not isinstance(number, int) or number < 0 or not 1 <= len(entries) <= maximum:
        raise ValueError('invalid bounded preparation batch')
    rows = []
    for ref in entries:
        if digest(ref['path']) != ref['sha256']:
            raise ValueError('entry checkpoint changed')
        row = json.loads(Path(ref['path']).read_text())
        if row['audit']['status'] != 'verified':
            raise ValueError('batch includes an unverified entry')
        rows.append(row)
    expected = list(range(number * maximum, number * maximum + len(rows)))
    if [r['source_index'] for r in rows] != expected:
        raise ValueError('batch source indices overlap or are noncontiguous')
    if any(r['source'] != rows[0]['source'] or r['native'] != rows[0]['native'] for r in rows):
        raise ValueError('batch mixes source or native identities')
    return immutable_json(Path(directory) / 'batches' / f'{number:06d}.json', {
        'schema': 'isg-preparation-batch-v1', 'source': rows[0]['source'],
        'native': rows[0]['native'], 'source_indices': expected, 'entries': entries,
        'bounded_size': maximum, 'production_authorized': False,
        'execution_requires_existing_ownership_and_backend_guards': True})
