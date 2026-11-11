"""Preserve proven transport permissions across a never-started replacement."""
import json
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require


def source_routes(root, record):
    """Use the immutable original intent; never invent a Vast-only exception."""
    from scripts.campaign_20261110.infra_dispatch import BACKENDS
    root = Path(root).resolve()
    source = record.get('dispatch_intent')
    if source:
        path = Path(source).resolve()
        require(path.is_relative_to(root / 'dispatch/ready'), 'migration source intent outside campaign')
        payload = json.loads(path.read_text())
        allowed = payload['backends']
        evidence = {'path': str(path), 'sha256': file_digest(path)}
    else:
        allowed = record.get('dispatch_backends')
        evidence = None
    require(allowed and set(allowed) <= set(BACKENDS), 'original migration transport routes unavailable')
    if record.get('dispatch_backends'):
        allowed = [b for b in allowed if b in record['dispatch_backends']]
    command = payload.get('command', []) if source else record.get('command', [])
    if '--code' in command:
        runtime = Path(command[command.index('--code') + 1]) / 'ispatialgym/sampling/campaign_runtime.py'
        if runtime.is_file() and 'fcntl.flock(' in runtime.read_text():
            allowed = [b for b in allowed if b.startswith('vast_')]
    # Order is descriptive. The dispatcher rechecks actual capacity at claim.
    routes = [b for b in BACKENDS if b in allowed and not b.startswith('vast_')]
    routes += [b for b in BACKENDS if b in allowed and b.startswith('vast_')]
    require(routes, 'original migration route restrictions have no intersection')
    result = {'backends': routes, 'source_intent': evidence,
              'reason': 'Preserved original transport permissions; fresh cluster-first assessment at replacement claim.'}
    if source:
        receipt = root / 'dispatch/receipts' / path.name
        if receipt.is_file():
            allocation = json.loads(receipt.read_text()).get('allocation')
            require(allocation, 'original migration receipt lacks allocation')
            result['source_receipt'] = str(receipt)
            result['source_allocation'] = allocation
    return result


def validate_claim(root, payload):
    """A replacement claim remains bound to its original cancellation proof."""
    routing = payload['migration_routing']
    source = routing['source_intent']
    require(source is not None, 'replacement claim lacks immutable source intent')
    path = Path(source['path']).resolve()
    require(path.is_relative_to(Path(root).resolve() / 'dispatch/ready'), 'source intent outside campaign')
    require(file_digest(path) == source['sha256'], 'source migration transport changed')
    original = json.loads(path.read_text())
    require(set(payload['backends']) <= set(original['backends']), 'migration widened original transport')
    proof = routing['proof']
    require(file_digest(proof['path']) == proof['sha256'], 'migration cancellation proof changed')
    value = json.loads(Path(proof['path']).read_text())
    allocation = value.get('original_allocation', {})
    identity = original.get('identity', {})
    for field in ('key', 'code', 'start', 'n'):
        require(field in allocation and field in identity
                and str(allocation[field]) == str(identity[field]),
                'cancellation proof original allocation differs from source intent: ' + field)
    if routing.get('source_receipt'):
        receipt = Path(routing['source_receipt']).resolve()
        require(receipt.is_relative_to(Path(root).resolve() / 'dispatch/receipts'), 'source receipt outside campaign')
        current = json.loads(receipt.read_text()).get('allocation')
        require(current == routing['source_allocation'] == allocation,
                'cancellation proof differs from immutable source receipt allocation')
    accounting = value.get('after', {}).get('accounting', {})
    require(value.get('never_started') and accounting.get('state') == 'CANCELLED'
            and accounting.get('start') in ('Unknown', 'None', 'N/A', '')
            and value.get('final_execution_proof', {}).get('nonempty_execution_artifacts') == [],
            'replacement lacks complete never-started cancellation proof')
    return routing
