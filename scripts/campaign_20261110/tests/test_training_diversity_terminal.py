import copy
import json
import sqlite3
from pathlib import Path

import pytest
from scripts.campaign_20261110 import training_diversity as d
from scripts.campaign_20261110.contracts import ContractError, digest, file_digest, write_json_atomic
from scripts.campaign_20261110.tests.test_training_diversity import setup, reserve, C, H


def ref(path):
    return {'path': str(path), 'sha256': file_digest(path)}


def fixture(root, monkeypatch, *, eligible=True):
    plan, _, current = setup(root, monkeypatch, capacity=3, future=2)
    ticket_ref = reserve(root, plan, count=3)
    record = {'instance_id': 'member__50000', 'index': 50000, 'task': 'object_scaling',
        'class_id': C, 'split': 'train', 'training_diversity_ticket': ticket_ref,
        'training_config_sha256': H}
    if eligible: current.append(record)
    database = root/'dataset/allocations.sqlite'; database.parent.mkdir()
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE members(record TEXT)')
        db.execute('INSERT INTO members VALUES (?)', (json.dumps(record),))
    outcome = root/'native_result.json'; write_json_atomic(outcome, {'returncode': 1, 'failed_indices': [50001, 50002]})
    job = {'backend': 'cluster', 'job_id': '12345', 'stage': 'collection'}
    controller = {'schema': 'isg-training-ticket-controller-v1', 'campaign_id': d.CAMPAIGN,
        'owner': '/root/dataset_contract', 'ticket': ticket_ref, 'config_sha256': H,
        'start': 50000, 'count': 3, 'phase': 'prepared', 'jobs': [job],
        'pending_dispatch': [], 'pending_acceptance': [], 'next_action': 'Repair the failed exact source before a new range.',
        'observed_at': 1000, 'outcome_artifacts': [ref(outcome)],
        'committed_records': {record['instance_id']: digest(record)},
        'unused_outcomes': {str(i): {'status': 'native_failed', 'artifacts': [ref(outcome)],
            'next_action': 'Changed bounded native correction before retry.'} for i in (50001, 50002)}}
    cp = root/'controller.json'; write_json_atomic(cp, controller)
    d.bind_ticket_controller(root, ticket_ref, cp)
    controller['phase'] = 'finalized'; write_json_atomic(cp, controller)
    raw = root/'scheduler.json'; write_json_atomic(raw, {
        'schema': 'isg-training-ticket-slurm-observation-v1', 'campaign_id': d.CAMPAIGN,
        'job_id': '12345', 'observed_at': 1000, 'squeue_returncode': 0, 'sacct_returncode': 0,
        'squeue_stdout': '', 'sacct_stdout': '12345|COMPLETED|2026-11-11T18:00:00|0:0\n'})
    observations = root/'observations.json'; write_json_atomic(observations, {
        'schema': 'isg-training-ticket-terminal-observations-v1', 'campaign_id': d.CAMPAIGN,
        'observed_at': 1000, 'jobs': [{**job, 'raw': ref(raw)}]})
    return plan, ticket_ref, cp, observations, raw, record


def retire(root, ticket, controller, observations, now=1001):
    return d.retire_ticket(root, ticket, ref(controller), ref(observations), now=now)


def test_only_unused_capacity_returns_and_indices_never_recycle(tmp_path, monkeypatch):
    plan, ticket, cp, obs, _, record = fixture(tmp_path, monkeypatch)
    original = Path(ticket['path']).read_bytes()
    receipt_ref = retire(tmp_path, ticket, cp, obs)
    receipt = json.loads(Path(receipt_ref['path']).read_text())
    assert receipt['released_indices'] == [50001, 50002]
    assert receipt['committed_records'] == [record]
    assert Path(ticket['path']).read_bytes() == original
    assert retire(tmp_path, ticket, cp, obs, now=2000) == receipt_ref
    with pytest.raises(ContractError, match='retired'): d.check_ticket(tmp_path, H, 50000, 3, ticket)
    assert d.check_ticket_identity(tmp_path, H, 50000, 3, ticket)['count'] == 3
    with pytest.raises(ContractError, match='overlaps'): reserve(tmp_path, plan, 'new', 50001, 2)
    with pytest.raises(ContractError, match='retired'): reserve(tmp_path, plan, count=3)
    replacement = reserve(tmp_path, plan, 'new', 50003, 2)
    assert d.check_ticket(tmp_path, H, 50003, 2, replacement)['count'] == 2
    with pytest.raises(ContractError, match='cell capacity'): reserve(tmp_path, plan, 'extra', 50005, 1)


def test_temporarily_excluded_committed_output_remains_reserved(tmp_path, monkeypatch):
    plan, ticket, cp, obs, _, record = fixture(tmp_path, monkeypatch, eligible=False)
    receipt = json.loads(Path(retire(tmp_path, ticket, cp, obs)['path']).read_text())
    assert receipt['committed_records'] == [record] and 50000 not in receipt['released_indices']
    reserve(tmp_path, plan, 'new', 50003, 2)
    with pytest.raises(ContractError, match='cell capacity'): reserve(tmp_path, plan, 'extra', 50005, 1)


@pytest.mark.parametrize('target,mutation', [
    ('controller', lambda x: x.update(phase='prepared')),
    ('controller', lambda x: x.update(pending_dispatch=['late-submit'])),
    ('controller', lambda x: x.update(pending_acceptance=['own-replay'])),
    ('controller', lambda x: x.update(owner='/wrong-owner')),
    ('controller', lambda x: x.update(committed_records={})),
    ('controller', lambda x: x['unused_outcomes'].pop('50002')),
    ('controller', lambda x: x['unused_outcomes']['50002'].update(status='awaiting_acceptance')),
    ('controller', lambda x: x['jobs'].append({'backend': 'cluster', 'job_id': '54321', 'stage': 'replay'})),
    ('controller', lambda x: x.update(jobs=[])),
    ('observations', lambda x: x.update(observed_at=100)),
    ('raw', lambda x: x.update(observed_at=100)),
    ('raw', lambda x: x.update(squeue_stdout='12345|RUNNING|gpu0')),
    ('raw', lambda x: x.update(sacct_returncode=1)),
    ('raw', lambda x: x.update(sacct_stdout='12345|PREEMPTED|2026-11-11T18:00:00|0:0')),
    ('raw', lambda x: x.update(sacct_stdout='12345|COMPLETED|Unknown|0:0')),
    ('raw', lambda x: x.update(sacct_stdout='other|COMPLETED|2026-11-11T18:00:00|0:0')),
])
def test_partial_stale_or_ambiguous_finalization_cannot_release(tmp_path, monkeypatch, target, mutation):
    _, ticket, cp, obs, raw, _ = fixture(tmp_path, monkeypatch)
    path = {'controller': cp, 'observations': obs, 'raw': raw}[target]
    value = json.loads(path.read_text()); mutation(value); write_json_atomic(path, value)
    if target == 'raw':
        value = json.loads(obs.read_text()); value['jobs'][0]['raw'] = ref(raw); write_json_atomic(obs, value)
    with pytest.raises(ContractError): retire(tmp_path, ticket, cp, obs)
    assert not d._read_state(tmp_path).get('retirements')
    assert d.check_ticket(tmp_path, H, 50000, 3, ticket)['count'] == 3


def test_accepted_identity_mismatch_and_lock_contention_are_fail_closed(tmp_path, monkeypatch):
    _, ticket, cp, obs, _, _ = fixture(tmp_path, monkeypatch)
    lock = tmp_path/'dataset/allocations.lockdir'; lock.mkdir()
    with pytest.raises(FileExistsError): retire(tmp_path, ticket, cp, obs)
    lock.rmdir()
    with sqlite3.connect(tmp_path/'dataset/allocations.sqlite') as db:
        value = json.loads(db.execute('SELECT record FROM members').fetchone()[0])
        value['training_config_sha256'] = 'b'*64
        db.execute('UPDATE members SET record=?', (json.dumps(value),))
    with pytest.raises(ContractError, match='another identity'): retire(tmp_path, ticket, cp, obs)


def test_orphan_retirement_rejoins_without_double_release(tmp_path, monkeypatch):
    _, ticket, cp, obs, _, _ = fixture(tmp_path, monkeypatch)
    real = d.write_json_atomic
    def fail_state(path, value):
        if Path(path).name == 'state.json' and value.get('retirements'): raise OSError('simulated publication interruption')
        return real(path, value)
    monkeypatch.setattr(d, 'write_json_atomic', fail_state)
    with pytest.raises(OSError): retire(tmp_path, ticket, cp, obs)
    assert not d._read_state(tmp_path).get('retirements')
    monkeypatch.setattr(d, 'write_json_atomic', real)
    first = next((tmp_path/'controllers/training_diversity/retirements').glob('*.json'))
    assert retire(tmp_path, ticket, cp, obs, now=1002) == ref(first)
    assert len(list(first.parent.glob('*.json'))) == 1


def test_vast_requires_current_synced_state_and_durable_attempt_output(tmp_path, monkeypatch):
    _, ticket, cp, obs, raw, _ = fixture(tmp_path, monkeypatch)
    controller = json.loads(cp.read_text()); controller['jobs'][0]['backend'] = 'vast'; write_json_atomic(cp, controller)
    state = tmp_path/'vast/state/jobs/12345.json'; write_json_atomic(state, {'id': '12345', 'status': 'synced',
        'done': {'rc': 1, 'start': 800, 'end': 990, 'attempt_evidence_uploaded': True, 'upload_error': None}})
    write_json_atomic(raw, {'schema': 'isg-training-ticket-vast-observation-v1', 'campaign_id': d.CAMPAIGN,
        'job_id': '12345', 'observed_at': 1000, 'state': ref(state)})
    observation = json.loads(obs.read_text()); observation['jobs'][0].update(backend='vast', raw=ref(raw)); write_json_atomic(obs, observation)
    changed = json.loads(state.read_text()); changed['status'] = 'running'; write_json_atomic(state, changed)
    with pytest.raises(ContractError, match='missing or changed'): retire(tmp_path, ticket, cp, obs)
    changed['status'] = 'synced'; write_json_atomic(state, changed)
    assert retire(tmp_path, ticket, cp, obs)


def test_observer_uses_raw_scheduler_results_and_does_not_retire(tmp_path, monkeypatch):
    _, ticket, cp, _, _, _ = fixture(tmp_path, monkeypatch)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        result = type('Result', (), {})()
        result.returncode = 0; result.stderr = ''
        result.stdout = '' if command[0] == 'squeue' else '12345|FAILED|2026-11-11T18:00:00|1:0\n'
        return result
    monkeypatch.setattr(d.subprocess, 'run', run); monkeypatch.setattr(d.time, 'time', lambda: 1001)
    observation = d.observe_ticket_jobs(tmp_path, ref(cp), tmp_path/'observed')
    assert len(calls) == 2 and calls[0][0] == 'squeue' and calls[1][0] == 'sacct'
    assert not d._read_state(tmp_path).get('retirements')
    assert d.retire_ticket(tmp_path, ticket, ref(cp), observation, now=1002)


def test_copied_vast_state_cannot_replace_current_authority(tmp_path, monkeypatch):
    _, ticket, cp, obs, raw, _ = fixture(tmp_path, monkeypatch)
    controller = json.loads(cp.read_text()); controller['jobs'][0]['backend'] = 'vast'; write_json_atomic(cp, controller)
    fake = tmp_path/'copied_vast_state.json'; write_json_atomic(fake, {'id': '12345', 'status': 'synced',
        'done': {'rc': 0, 'start': 800, 'end': 990, 'attempt_evidence_uploaded': True}})
    write_json_atomic(raw, {'schema': 'isg-training-ticket-vast-observation-v1', 'campaign_id': d.CAMPAIGN,
        'job_id': '12345', 'observed_at': 1000, 'state': ref(fake)})
    observation = json.loads(obs.read_text()); observation['jobs'][0].update(backend='vast', raw=ref(raw)); write_json_atomic(obs, observation)
    with pytest.raises(ContractError, match='authoritative'): retire(tmp_path, ticket, cp, obs)
