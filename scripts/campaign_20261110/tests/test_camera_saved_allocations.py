import csv
import json
from pathlib import Path

import pytest

from scripts.campaign_20261110 import controller_camera as camera
from scripts.campaign_20261110 import controller_migration as migration
from scripts.campaign_20261110 import controller_saved_allocations as saved
from scripts.campaign_20261110.contracts import ContractError, digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def fixture(root):
    command = ['python', 'submit_fast.py', 'submit', 'camera', '3', '--code', str(root/'code/native'),
               '--base-index', '50000100', '--tag', 'camera_auto_production']
    identity = {'key': 'camera', 'start': 50000100, 'n': 3, 'code': str(root/'code/native'),
                'tag': 'camera_auto_production'}
    row = {**{k: str(v) for k, v in identity.items()}, 'jobid': '123', 'job_dir': str(root/'out/job')}
    ledger = root/'manifests/dispatch_ledgers/one/allocations.tsv'
    ledger.parent.mkdir(parents=True)
    with ledger.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), delimiter='\t')
        writer.writeheader(); writer.writerow(row)
    allocation = {**row, 'ledger': str(ledger)}
    intent = write(root/'dispatch/ready/one.json', {'token': 'one', 'command': command,
        'command_sha256': digest(command), 'identity': identity})
    receipt = write(root/'dispatch/receipts/one.json', {'token': 'one', 'status': 'submitted',
        'command_sha256': digest(command), 'identity': identity, 'allocation': allocation})
    record = {'directory': str(root/'production_inputs/one'), 'command': command, 'index': 50000100,
              'count': 3, 'status': 'external_submission', 'dispatch_intent': str(intent),
              'dispatch_receipt': str(receipt)}
    spec = {'id': 'seed', 'class_id': 'camera', 'validation_jobid': '122'}
    cfg = {'seeds': [spec], 'migrate_pending_seconds': 0, '_migration_checks_remaining': 8}
    state = {'seeds': {'seed': {'production': [record]}}}
    write(root/'validation/open_loop/camera/122/gate.json', {'success': True})
    return cfg, state, record, allocation


def source_allocation(root, state, allocation):
    source = dict(allocation, jobid='122', start='50000000', job_dir=str(root/'out/source'),
                  tag='camera_validation', ledger=str(root/'manifests/source_allocations.tsv'))
    path = Path(source['ledger'])
    with path.open('w') as stream:
        row = {k: v for k, v in source.items() if k != 'ledger'}
        writer = csv.DictWriter(stream, fieldnames=list(row), delimiter='\t')
        writer.writeheader(); writer.writerow(row)
    state['seeds']['seed']['source_allocation'] = source
    return source


def test_camera_tag_requires_explicit_scope(tmp_path):
    _, _, record, allocation = fixture(tmp_path)
    with pytest.raises(ContractError):
        saved.adopt_receipt(tmp_path, record)
    assert saved.adopt_receipt(tmp_path, record, expected_tag='camera_auto_production')
    assert record['allocation'] == allocation


def test_first_pending_check_precedes_legacy_source_scan(tmp_path, monkeypatch):
    cfg, state, record, allocation = fixture(tmp_path)
    events = []
    monkeypatch.setattr(camera, 'slurm_status', lambda: {'123': ['123', 'PENDING', 'owner']})
    monkeypatch.setattr(camera, 'validated_seed', lambda *a: events.append('source_checks'))
    monkeypatch.setattr(migration, 'controller_check', lambda *a, **k: events.append('pending'))
    monkeypatch.setattr(camera, 'allocations', lambda *a: events.append('legacy_scan') or [allocation])
    _, _, blocked = camera.observe_camera_allocations(tmp_path, state, cfg, lambda: None)
    assert events == ['source_checks', 'pending', 'legacy_scan'] and not blocked
    assert record['allocation'] == allocation
    assert state['early_pending_observation']['records'][0]['scheduler_state'] == 'PENDING'


def test_known_camera_source_and_receipt_avoid_global_scan(tmp_path, monkeypatch):
    cfg, state, _, allocation = fixture(tmp_path)
    source = source_allocation(tmp_path, state, allocation)
    monkeypatch.setattr(camera, 'slurm_status', lambda: {'123': ['123', 'RUNNING', 'owner']})
    monkeypatch.setattr(camera, 'allocations', lambda *a: pytest.fail('unnecessary global allocation scan'))
    monkeypatch.setattr(migration, 'controller_check', lambda *a, **k: pytest.fail('running job must not migrate'))
    for _ in range(2):
        rows, _, blocked = camera.observe_camera_allocations(tmp_path, state, cfg, lambda: None)
        assert rows == [source, allocation] and not blocked


@pytest.mark.parametrize('problem', ['failed_gate', 'missing_gate', 'changed_seed'])
def test_source_validation_still_precedes_camera_migration(tmp_path, monkeypatch, problem):
    cfg, state, _, allocation = fixture(tmp_path)
    source_allocation(tmp_path, state, allocation)
    gate = tmp_path/'validation/open_loop/camera/122/gate.json'
    if problem == 'failed_gate': write(gate, {'success': False})
    if problem == 'missing_gate': gate.unlink()
    def validate(*args):
        if problem == 'changed_seed': raise ContractError('seed changed')
    monkeypatch.setattr(camera, 'validated_seed', validate)
    monkeypatch.setattr(camera, 'slurm_status', lambda: {'123': ['123', 'PENDING', 'owner']})
    monkeypatch.setattr(camera, 'allocations', lambda *a: pytest.fail('known source must not scan'))
    monkeypatch.setattr(migration, 'controller_check', lambda *a, **k: pytest.fail('invalid source must not migrate'))
    _, _, blocked = camera.observe_camera_allocations(tmp_path, state, cfg, lambda: None)
    assert blocked == {'seed'}


def test_changed_cached_source_ledger_blocks_seed(tmp_path, monkeypatch):
    cfg, state, _, allocation = fixture(tmp_path)
    source = source_allocation(tmp_path, state, allocation)
    path = Path(source['ledger']); path.write_text(path.read_text().replace('122', '999'))
    monkeypatch.setattr(camera, 'slurm_status', lambda: {})
    monkeypatch.setattr(camera, 'allocations', lambda *a: pytest.fail('changed source must not fall back'))
    _, _, blocked = camera.observe_camera_allocations(tmp_path, state, cfg, lambda: None)
    assert blocked == {'seed'}


def test_legacy_discovery_preserves_duplicate_evidence(tmp_path, monkeypatch):
    cfg, state, _, allocation = fixture(tmp_path)
    monkeypatch.setattr(camera, 'slurm_status', lambda: {})
    monkeypatch.setattr(camera, 'allocations', lambda *a: [allocation, allocation])
    rows, _, _ = camera.observe_camera_allocations(tmp_path, state, cfg, lambda: None)
    with pytest.raises(ContractError, match='one durable allocation'):
        camera.allocation_for(rows, '123')
