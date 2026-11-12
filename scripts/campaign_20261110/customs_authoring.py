"""Write runtime customs without renumbering same-synset role bindings.

GroundedProblem and Bindings assign instance suffixes in role insertion order.
Canonical sorted JSON is appropriate for hashes, but changes that runtime input.
"""
from pathlib import Path
import json
import os
import tempfile


def validate_role_order(entry, expected, *, alias):
    if list(entry['roles']) != list(expected):
        raise ValueError(f'Native role order differs for {alias}')


def write_customs_atomic(path, value, *, role_orders):
    if set(value) != set(role_orders):
        raise ValueError('Every customs alias needs its expected native role order')
    for alias, entry in value.items():
        validate_role_order(entry, role_orders[alias], alias=alias)
    payload = json.dumps(value, indent=2, allow_nan=False) + '\n'
    restored = json.loads(payload)
    for alias, order in role_orders.items():
        assert list(restored[alias]['roles']) == list(order)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
