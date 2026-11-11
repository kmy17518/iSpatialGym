import json
from pathlib import Path

import pytest

from scripts.campaign_20261110.infra_vast import cluster_return as cr
from scripts.campaign_20261110.infra_vast import cluster_return_capacity as cap
from scripts.campaign_20261110.infra_vast import cluster_return_release as rel
from scripts.campaign_20261110.infra_vast import cluster_return_client as client
from scripts.campaign_20261110.tests.test_vast_cluster_return import fixture


def claimed(tmp_path):
    root, request, job, _, path = fixture(tmp_path)
    states = {'9001': job}
    save = lambda j: cr.write(root/'vast/state/jobs/9001.json', j)
    cr.reconcile(root, states, {}, {}, set(), save, now=1001)
    ack_path = root/'vast/cluster_return/acknowledgements'/path.name
    ack = json.loads(ack_path.read_text())
    output = root/'infra/capacity_recovery/cluster_return_client_r1'
    return root, job, states, save, ack_path, ack, output


def scheduler(ack, state='PENDING'):
    return {'job_id': '12345', 'scheduler': {'ArrayJobId': '12345', 'ArrayTaskId': '0',
            'UserId': 'minyeong(25282)', 'JobName': 'isg261110_map_return',
            'Command': ack['wrapper']['path'], 'Dependency': '(null)', 'Restarts': '0',
            'JobState': state, 'Reason': 'Priority', 'TimeLimit': '02:00:00'},
            'accounting': {'state': state, 'start': 'Unknown', 'elapsed_seconds': 0, 'allocated_tres': ''}}


def submitted(ack_path, ack, output):
    path = output/(ack_path.stem+'_submitted.json')
    cr.write(path, {'acknowledgement': rel.ref(ack_path), 'job_id_vast': ack['job_id'],
                    'unit': ack['unit'], 'job_id': '12345'})
    return path


def test_withdrawal_is_durable_and_cannot_submit_again(tmp_path, monkeypatch):
    root, job, states, save, path, ack, output = claimed(tmp_path)
    attempt = output/(path.stem+'_attempt.json')
    rel.withdraw(root, path, ack, attempt, {'status': 'unavailable'})
    monkeypatch.setattr(client.subprocess, 'run', lambda *a, **k: pytest.fail('must not submit'))
    assert client.submit(root, 'minyeong')['submitted'] == []
    result = cr.reconcile(root, states, {}, {}, set(), save, now=1002)
    assert result['releases']['released'] == ['9001']
    assert job['status'] == 'queued' and 'cluster_return' not in job
    assert job['cluster_return_releases'][0]['no_further_cluster_return']
    request = json.loads(Path(ack['request']['path']).read_text())
    with pytest.raises(ValueError, match='never-assigned'):
        cr.validate(root, request, job, set(), set(), set(), 1002)


def test_withdrawal_crash_before_request_is_recovered(tmp_path):
    root, job, states, save, path, ack, output = claimed(tmp_path)
    attempt = output/(path.stem+'_attempt.json')
    rel.withdraw(root, path, ack, attempt, {})
    rel.request_path(root, path).unlink()
    assert rel.recover_withdrawal(root, path, attempt)
    assert rel.request_path(root, path).exists()


def test_final_capacity_check_is_serialized_and_withdraws_without_sbatch(tmp_path, monkeypatch):
    root, _, _, _, path, _, _ = claimed(tmp_path)
    def probe():
        assert (root/'vast/cluster_return/submit_capacity.lockdir').is_dir()
        return {'status': 'observed', 'account_slots': {'minyeong': 0}}
    monkeypatch.setattr(client.capacity_probe, 'probe', probe)
    monkeypatch.setattr(client.subprocess, 'run', lambda *a, **k: pytest.fail('no available slot'))
    result = client.submit(root, 'minyeong')
    assert not result['errors'] and not result['submitted']
    assert result['recoveries'][0]['status'] == 'withdrawn_before_submission'
    assert rel.request_path(root, path).exists()


def test_sbatch_and_receipt_remain_inside_shared_capacity_lock(tmp_path, monkeypatch):
    from types import SimpleNamespace
    root, _, _, _, path, _, _ = claimed(tmp_path)
    lock = root/'vast/cluster_return/submit_capacity.lockdir'
    monkeypatch.setattr(client.capacity_probe, 'probe', lambda: {'status': 'observed', 'account_slots': {'minyeong': 1}})
    def run(*a, **k):
        assert lock.is_dir()
        return SimpleNamespace(returncode=0, stdout='12345\n', stderr='')
    original = cr.write
    def write(p, value):
        if str(p).endswith('_submitted.json'): assert lock.is_dir()
        original(p, value)
    monkeypatch.setattr(client.subprocess, 'run', run)
    monkeypatch.setattr(cr, 'write', write)
    result = client.submit(root, 'minyeong')
    assert not result['errors'] and result['submitted'][0]['job_id'] == '12345'
    assert not lock.exists()


def test_unknown_submission_attempt_is_never_withdrawn(tmp_path):
    root, _, _, _, path, ack, output = claimed(tmp_path)
    attempt = output/(path.stem+'_attempt.json')
    cr.write(attempt, {'status': 'submitting'})
    assert not rel.recover_withdrawal(root, path, attempt)
    with pytest.raises(ValueError, match='already exists'):
        rel.withdraw(root, path, ack, attempt, {})


def test_pending_return_cancelled_then_coordinator_releases_exact_owner(tmp_path, monkeypatch):
    root, job, states, save, path, ack, output = claimed(tmp_path)
    receipt = submitted(path, ack, output)
    pending, cancelled = scheduler(ack), scheduler(ack, 'CANCELLED')
    calls = []
    monkeypatch.setattr(rel, 'scheduler_proof', lambda _: cancelled if calls else pending)
    monkeypatch.setattr(rel.overflow_readiness, 'check', lambda *a: {'budget_usd': 3000})
    monkeypatch.setattr(rel.subprocess, 'run', lambda argv, **k: calls.append(argv))
    assert rel.recover_pending(root, path, ack, receipt)['status'] == 'release_requested'
    assert calls == [['scancel', '--state=PENDING', '12345']]
    assert job['status'] == 'cluster_reserved'  # Client cannot release it.
    result = cr.reconcile(root, states, {}, {}, set(), save, now=1002)
    assert result['releases']['released'] == ['9001']
    assert job['status'] == 'queued'


@pytest.mark.parametrize('case', ['already_running', 'starts_during_cancel', 'dependent', 'wrong_unit', 'wrong_wrapper'])
def test_running_or_mismatched_map_cannot_be_requeued(tmp_path, monkeypatch, case):
    root, job, states, save, path, ack, output = claimed(tmp_path)
    receipt = submitted(path, ack, output)
    pending = scheduler(ack)
    running = scheduler(ack, 'RUNNING')
    running['accounting'].update(start='2026-11-11T21:00:00', elapsed_seconds=1, allocated_tres='gres/gpu=1')
    calls = []
    if case == 'dependent': pending['scheduler']['Dependency'] = 'afterok:99'
    if case == 'wrong_unit': pending['scheduler']['ArrayTaskId'] = '1'
    if case == 'wrong_wrapper': pending['scheduler']['Command'] = '/another/campaign.sh'
    monkeypatch.setattr(rel, 'scheduler_proof', lambda _: running if calls or case == 'already_running' else pending)
    monkeypatch.setattr(rel.overflow_readiness, 'check', lambda *a: {})
    monkeypatch.setattr(rel.subprocess, 'run', lambda argv, **k: calls.append(argv))
    if case in ('dependent', 'wrong_unit', 'wrong_wrapper'):
        with pytest.raises(ValueError): rel.recover_pending(root, path, ack, receipt)
    else:
        assert rel.recover_pending(root, path, ack, receipt)['status'] in ('preserved_execution', 'awaiting_exact_cancellation')
    assert not rel.request_path(root, path).exists()
    assert job['status'] == 'cluster_reserved'


@pytest.mark.parametrize('case', ['execution', 'submission', 'changed_tombstone', 'remote'])
def test_coordinator_refuses_conflicting_withdrawal_evidence(tmp_path, case):
    root, job, states, save, path, ack, output = claimed(tmp_path)
    attempt = output/(path.stem+'_attempt.json')
    rel.withdraw(root, path, ack, attempt, {})
    if case == 'execution': cr.write(Path(ack['unit']['output'])/'started.json', {})
    if case == 'submission': submitted(path, ack, output)
    if case == 'changed_tombstone': cr.write(attempt, {'status': 'submitting'})
    remote = {'9001'} if case == 'remote' else set()
    result = cr.reconcile(root, states, {}, {}, remote, save, now=1002)
    assert result['releases']['errors'] and job['status'] == 'cluster_reserved'


def test_release_ack_crash_recovers_without_duplicate_requeue(tmp_path, monkeypatch):
    root, job, states, save, path, ack, output = claimed(tmp_path)
    rel.withdraw(root, path, ack, output/(path.stem+'_attempt.json'), {})
    original = cr.write
    def fail_ack(p, value):
        if Path(p).parent.name == 'released': raise OSError('crash after durable queue state')
        original(p, value)
    monkeypatch.setattr(cr, 'write', fail_ack)
    assert cr.reconcile(root, states, {}, {}, set(), save, now=1002)['releases']['errors']
    assert job['status'] == 'queued'
    job.update(status='assigned', instance='worker', tries=1)
    monkeypatch.setattr(cr, 'write', original)
    assert cr.reconcile(root, states, {}, {}, set(), save, now=1003)['releases']['released'] == ['9001']
    assert job['status'] == 'assigned' and len(job['cluster_return_releases']) == 1


def observation():
    return {'observed_at': 1000, 'state': 'available', 'jobs': [], 'nodes': [{
        'NodeName': 'svl7', 'Partitions': 'svl,viscam', 'State': 'MIXED', 'CPUTot': '64',
        'CPUAlloc': '0', 'RealMemory': '524288', 'AllocMem': '0',
        'CfgTRES': 'gres/gpu=8,gres/gpu:titanrtx=8', 'AllocTRES': 'gres/gpu=4'}]}


def test_live_capacity_never_double_counts_shared_node_or_uses_planned_slots():
    obs = observation()
    assert cap.assess(obs, now=1001)['compatible_available_slots'] == 4
    obs['nodes'][0]['State'] = 'MIXED+PLANNED'
    assert cap.assess(obs, now=1001)['compatible_available_slots'] == 0
    assert cap.assess(observation(), now=1061)['status'] == 'unavailable_or_stale'


def test_cancelled_array_parent_requires_exact_before_and_singleton_accounting(tmp_path):
    _, _, _, _, path, ack, output = claimed(tmp_path)
    receipt = json.loads(submitted(path, ack, output).read_text())
    before = scheduler(ack)
    after = scheduler(ack, 'CANCELLED')
    after['scheduler']['ArrayTaskId'] = '4294967294'
    after['accounting']['array_job_id'] = '12345_[0]'
    rel.identity(after, ack, receipt, before)
    with pytest.raises(ValueError, match='unit differs'):
        rel.identity(after, ack, receipt)
    for wrong in ('12345_[1]', '12345_[0-2]', '12345_[0,1]', '54321_[0]', '12345'):
        after['accounting']['array_job_id'] = wrong
        with pytest.raises(ValueError, match='singleton'):
            rel.identity(after, ack, receipt, before)
    after['accounting']['array_job_id'] = '12345_[0]'
    before['accounting']['elapsed_seconds'] = 1
    with pytest.raises(ValueError, match='never-started'):
        rel.identity(after, ack, receipt, before)


def test_accounting_probe_keeps_exact_array_set_and_resource_evidence(tmp_path, monkeypatch):
    _, _, _, _, _, ack, _ = claimed(tmp_path)
    monkeypatch.setattr(rel, 'base_scheduler_proof', lambda _: scheduler(ack, 'CANCELLED'))
    monkeypatch.setattr(rel.subprocess, 'check_output', lambda *a, **k: '12345|12345_[0]|CANCELLED by 25282|None|0|\n')
    proof = rel.scheduler_proof('12345')
    assert proof['accounting']['array_job_id'] == '12345_[0]'
    assert rel.never_started(proof, 'CANCELLED')


def test_both_accounts_resource_pending_prevents_returns_but_dependencies_do_not():
    obs = observation()
    obs['jobs'] = [[u, 'svl,viscam', 'PENDING', 'gres/gpu:titanrtx:1', 'normal', '(Priority)', '100']
                   for u in ('minyeong', 'shiyuc')]
    assert cap.assess(obs, now=1001)['compatible_available_slots'] == 0
    for row in obs['jobs']: row[5] = '(Dependency)'
    assert cap.assess(obs, now=1001)['compatible_available_slots'] == 4
