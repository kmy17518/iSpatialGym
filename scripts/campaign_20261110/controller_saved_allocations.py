"""Join a saved queue to its own dispatch receipt before campaign-wide scans.

This only establishes allocation ownership. Migration still uses the normal
fresh scheduler, capacity, funding, input and never-started checks.
"""
import csv
from copy import deepcopy
import json
from pathlib import Path
import time

from scripts.campaign_20261110.contracts import digest, file_digest, require


def adopt_receipt(root, record, *, expected_tag='goal_groups'):
    require(expected_tag in ('goal_groups', 'camera_auto_production'), 'unsupported saved allocation tag')
    if record.get('allocation') or not record.get('dispatch_receipt'):
        return False
    root = Path(root).resolve()
    receipt_path = Path(record['dispatch_receipt']).resolve()
    intent_path = Path(record['dispatch_intent']).resolve()
    require(receipt_path.is_relative_to(root/'dispatch/receipts') and
            intent_path.is_relative_to(root/'dispatch/ready'), 'saved receipt outside dispatch')
    if not receipt_path.exists():
        return False  # The router still owns submission/recovery, never resubmit here.
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('migration') or receipt.get('status') != 'submitted':
        return False
    intent = json.loads(intent_path.read_text())
    command = record['command']
    require(intent['command'] == command and
            intent['command_sha256'] == receipt['command_sha256'] == digest(command),
            'saved receipt command differs')
    require(receipt['token'] == intent['token'] == Path(record['directory']).name and
            receipt['identity'] == intent['identity'], 'saved receipt identity differs')
    identity, allocation = intent['identity'], receipt['allocation']
    expected = {'key': command[3], 'start': int(record['index']), 'n': int(record['count']),
                'code': command[command.index('--code')+1], 'tag': expected_tag}
    require(all(identity[k] == v for k, v in expected.items()), 'saved queue identity differs')
    require(all((int(allocation[k]) if k in ('start', 'n') else allocation[k]) == v
                for k, v in expected.items()), 'saved allocation differs')
    require(str(allocation['jobid']).isdigit() and
            Path(allocation['job_dir']).resolve().is_relative_to(root/'out'),
            'saved allocation job outside campaign')
    ledger = Path(allocation['ledger']).resolve()
    require(ledger.is_relative_to(root/'manifests'), 'saved allocation ledger outside campaign')
    with ledger.open() as stream:
        matches = [r for r in csv.DictReader(stream, delimiter='\t')
                   if r.get('key') == expected['key'] and int(r['start']) == expected['start']]
    require(len(matches) == 1 and matches[0] == {k: v for k, v in allocation.items() if k != 'ledger'},
            'saved allocation ledger is absent or ambiguous')
    record['allocation'] = allocation
    record['allocation_receipt_observation'] = {
        'observed_at': time.time(), 'receipt': {'path': str(receipt_path), 'sha256': file_digest(receipt_path)},
        'intent': {'path': str(intent_path), 'sha256': file_digest(intent_path)},
        'ledger': {'path': str(ledger), 'sha256': file_digest(ledger)},
        'scope': 'Exact saved allocation only; no execution or migration eligibility inferred.'}
    return True


def seed_existing_queues(root, cfg, state):
    """First-start adoption of saved spooled work; legacy inputs still fall back."""
    root = Path(root).resolve()
    for spec in cfg.get('seeds', []):
        current = state.setdefault('seeds', {}).setdefault(spec['id'], {})
        records = current.setdefault('queues', [])
        for queue_name in spec.get('existing_queues', []):
            if any(r['queue'] == queue_name for r in records):
                continue
            queue = Path(queue_name).resolve()
            require(queue.is_relative_to(root/'queues'), 'existing queue outside campaign')
            submission = json.loads((queue.parent/'submission.json').read_text())
            intent = root/'dispatch/ready'/(queue.parent.name+'.json')
            if not submission.get('command') or not intent.exists():
                continue
            payload = json.loads(intent.read_text())
            require(payload['command'] == submission['command'] and
                    payload['command_sha256'] == digest(submission['command']),
                    'existing queue dispatch command differs')
            require(file_digest(spec['seed_config']) == spec['seed_config_sha256'],
                    'existing queue source config changed')
            source = json.loads(Path(spec['seed_config']).read_text())
            prepared = submission['prepared']
            require(Path(prepared['queue']).resolve() == queue and
                    Path(prepared['config']).resolve().is_relative_to(root/'groups'),
                    'existing prepared queue differs')
            key = json.loads(Path(prepared['config']).read_text())['validation_key']
            require(key == source['validation_key'], 'existing queue source key differs')
            record = deepcopy(submission)
            record.update(queue=str(queue), prepared=prepared, validation_key=key,
                          phase=source['split'], count=prepared['candidate_instances'],
                          status='external_submission', dispatch_intent=str(intent),
                          dispatch_receipt=str(root/'dispatch/receipts'/(queue.parent.name+'.json')))
            # A publication error blocks the entire cycle before any migration.
            adopt_receipt(root, record)
            records.append(record)


def bounded_rows(cfg, state):
    """Known saved allocations need no repeated historical ledger discovery."""
    specs = list(cfg.get('seeds', []))
    if cfg.get('seed_catalog'):
        specs += json.loads(Path(cfg['seed_catalog']).read_text()).get('seeds', [])
    if not specs:
        return None
    if set(state.get('seeds', {})) != {s['id'] for s in specs}:
        return None
    rows = []
    for spec in specs:
        records = state['seeds'][spec['id']].get('queues', [])
        if any(path not in {r['queue'] for r in records} for path in spec.get('existing_queues', [])):
            return None
        for record in records:
            if record.get('allocation'):
                rows.append(record['allocation'])
            elif record.get('status') != 'complete' and not (
                    record.get('status') == 'external_submission' and record.get('dispatch_receipt')):
                return None
    return rows
