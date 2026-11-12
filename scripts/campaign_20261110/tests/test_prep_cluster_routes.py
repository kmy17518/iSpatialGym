import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts.campaign_20261110 import prep_cluster_routes as routes


def node(name, model, count=8, partition='viscam', cpus=64, memory=512*1024):
    return {'NodeName': name, 'Partitions': partition, 'State': 'IDLE',
            'CfgTRES': f'cpu={cpus},gres/gpu={count},gres/gpu:{model}={count}',
            'AllocTRES': '', 'CPUEfctv': str(cpus), 'CPUAlloc': '0',
            'RealMemory': str(memory), 'AllocMem': '0'}


def observation(nodes=None, jobs=()):
    return {'observed_at': 1000, 'finished_at': 1000, 'jobs': list(jobs),
            'nodes': nodes if nodes is not None else [node('svl8', 'titanrtx', partition='svl'), node('viscam2', 'a40')],
            'partitions': [{'PartitionName': 'svl', 'State': 'UP', 'AllowAccounts': 'vision', 'QoS': 'svl'},
                           {'PartitionName': 'viscam', 'State': 'UP', 'AllowAccounts': 'viscam', 'QoS': 'viscam'}],
            'qos': [['svl', 'gres/gpu=48', '56', '100'], ['viscam', 'gres/gpu=30', '', '']],
            'associations': [[user, account, '', '', '', ''] for user in routes.USERS for account in ('vision', 'viscam')],
            'query_sha256': {'jobs': 'a'*64, 'nodes': 'b'*64}}


def ticket(tmp_path, account='minyeong'):
    script = tmp_path/'job.sh'; script.write_text('pinned native payload\n')
    return {'id': 'one', 'expected_account': account, 'workstream': 'training preparation',
            'command': ['sbatch', '--parsable', '-A', 'vision', '-p', 'svl,viscam', '--gres=gpu:titanrtx:1',
                        '-c', '8', '--mem=48G', '--nodes=1', '--ntasks=1', '--time=00:30:00', str(script), 'native_argument'],
            'inputs': {str(script): __import__('hashlib').sha256(script.read_bytes()).hexdigest()},
            'claim_path': str(tmp_path/'launch.claim'), 'claim_kind': 'directory',
            'submission_path': str(tmp_path/'submission.json')}


def capacity(obs=None):
    return routes.build_capacity(obs or observation(), now=1000)


def test_viscam_uses_correct_account_typed_gpu_and_exact_native_suffix(tmp_path):
    row = ticket(tmp_path); before = copy.deepcopy(row)
    result = routes.route_ticket(row, capacity(), 'minyeong', now=1000)
    assert '--account=viscam' in result['command'] and '--partition=viscam' in result['command']
    assert '--gres=gpu:a40:1' in result['command'] and '--nodelist=viscam2' in result['command']
    assert result['command'][-2:] == row['command'][-2:]
    assert '--nice=1000' in result['command'] and row == before
    assert result['route']['cpus'] == 8 and result['route']['memory_mb'] == 48*1024


def test_no_titans_does_not_invent_four_held_slots_or_block_non_titans(tmp_path):
    cap = capacity(observation([node('viscam2', 'a40')]))
    assert cap['critical_titan_slots_actually_held'] == 0
    assert cap['compatible_available_slots'] == cap['preparation_slots'] == 8
    assert routes.route_ticket(ticket(tmp_path), cap, 'minyeong', now=1000)['route']['gpu_model'] == 'a40'


def test_four_titans_and_their_cpu_memory_are_shared_across_users(tmp_path):
    cap = capacity(observation([node('svl8', 'titanrtx', partition='svl')]))
    assert cap['preparation_slots'] == 4 and cap['critical_titan_slots_actually_held'] == 4
    for i in range(4):
        user = routes.USERS[i % 2]
        result = routes.route_ticket(ticket(tmp_path, user), cap, user, now=1000)
        assert result['route']['gpu_model'] == 'titanrtx'
        cap['_reservations'].append(result['route'])
    assert routes.route_ticket(ticket(tmp_path), cap, 'minyeong', now=1000) is None


def test_reserve_cannot_be_spent_as_other_gpu_model_cpu_on_same_node(tmp_path):
    mixed = node('viscam2', 'titanrtx', count=4)
    mixed['CfgTRES'] = 'cpu=64,gres/gpu=12,gres/gpu:titanrtx=4,gres/gpu:a40=8'
    cap = capacity(observation([mixed]))
    assert cap['critical_titan_slots_actually_held'] == 4 and cap['preparation_slots'] == 4
    for _ in range(4):
        result = routes.route_ticket(ticket(tmp_path), cap, 'minyeong', now=1000)
        assert result['route']['gpu_model'] == 'a40'
        cap['_reservations'].append(result['route'])
    assert routes.route_ticket(ticket(tmp_path), cap, 'minyeong', now=1000) is None


def test_pending_critical_jobs_consume_physical_slots_before_reserve(tmp_path):
    jobs = [['minyeong', 'svl', 'PENDING', 'gres/gpu:titanrtx:1', 'normal', '(Priority)', str(i), 'vision', '8', '48G'] for i in range(4)]
    cap = capacity(observation([node('svl8', 'titanrtx', partition='svl')], jobs))
    assert cap['preparation_slots'] == 0 and cap['critical_titan_slots_actually_held'] == 4
    assert routes.route_ticket(ticket(tmp_path), cap, 'minyeong', now=1000) is None


def test_cpu_only_services_still_exhaust_svl_running_job_quota(tmp_path):
    jobs = [['minyeong', 'svl', 'RUNNING', 'N/A', 'normal', 'svl8', str(i), 'vision', '1', '1G'] for i in range(56)]
    cap = capacity(observation([node('svl8', 'titanrtx', partition='svl')], jobs))
    assert cap['account_slots']['minyeong'] == 0 and cap['account_slots']['shiyuc'] == 4
    assert routes.route_ticket(ticket(tmp_path), cap, 'minyeong', now=1000) is None


def test_fresh_partition_association_and_typed_quotas_are_required(tmp_path):
    obs = observation([node('viscam2', 'a40')]); obs['associations'] = [r for r in obs['associations'] if r[:2] != ['minyeong', 'viscam']]
    assert routes.route_ticket(ticket(tmp_path), capacity(obs), 'minyeong', now=1000) is None
    obs = observation([node('viscam2', 'a40')]); obs['qos'][1][1] = 'gres/gpu=30,gres/gpu:a40=0'
    assert routes.route_ticket(ticket(tmp_path), capacity(obs), 'minyeong', now=1000) is None


def test_unsupported_models_excluded_hosts_stale_capacity_and_payload_flags(tmp_path):
    obs = observation([node('viscam3', 'h200'), node('viscam5', 'a40')])
    assert routes.route_ticket(ticket(tmp_path), capacity(obs), 'minyeong', now=1000) is None
    with pytest.raises(ValueError, match='expired'):
        routes.route_ticket(ticket(tmp_path), capacity(), 'minyeong', now=1061)
    row = ticket(tmp_path); row['command'][2:2] = ['--constraint=titanrtx']
    with pytest.raises(ValueError, match='Unsupported'):
        routes.route_ticket(row, capacity(), 'minyeong', now=1000)


def test_node_cpu_memory_and_generic_allocations_bound_typed_slots(tmp_path):
    n = node('viscam2', 'a40'); n.update(CPUAlloc='48', AllocMem=str(464*1024), AllocTRES='gres/gpu=2')
    cap = capacity(observation([n])); assert cap['preparation_slots'] == 1
    result = routes.route_ticket(ticket(tmp_path), cap, 'minyeong', now=1000)
    cap['_reservations'].append(result['route'])
    assert routes.route_ticket(ticket(tmp_path), cap, 'minyeong', now=1000) is None


def test_probe_queries_both_users_and_proves_fresh_associations(monkeypatch):
    raw = {
        'squeue': '', 'nodes': 'NodeName=viscam2 Partitions=viscam State=IDLE CfgTRES=gres/gpu=1,gres/gpu:a40=1 CPUEfctv=8 CPUAlloc=0 RealMemory=49152 AllocMem=0',
        'partition': 'PartitionName=viscam State=UP AllowAccounts=viscam QoS=viscam',
        'qos': 'viscam|gres/gpu=30|||', 'assoc': 'minyeong|viscam||||\nshiyuc|viscam||||'}
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        key = 'squeue' if argv[0] == 'squeue' else argv[2 if argv[0] == 'scontrol' else 4]
        return SimpleNamespace(stdout=raw[key])
    obs = routes.observe(run)
    assert any('minyeong,shiyuc' in argv for argv in calls)
    assert any('user=minyeong,shiyuc' in argv for argv in calls)
    assert set(obs['query_sha256']) == {'jobs', 'nodes', 'partitions', 'qos', 'associations'}


def test_existing_claim_and_durable_intent_prevent_duplicate_routed_submission(tmp_path, monkeypatch):
    from scripts.campaign_20261110 import coordinator_dispatch as coordinator
    monkeypatch.setattr(routes.time, 'time', lambda: 1000)
    d = tmp_path/'infra/coordinator_dispatch'; d.mkdir(parents=True); (d/'capacity_allocation.lock').touch()
    row = ticket(tmp_path); manifest = tmp_path/'tickets.json'; manifest.write_text(json.dumps({'tickets': [row]}))
    (d/'policy.json').write_text(json.dumps({'owner': '/root', 'production_released': False,
        'prepared_manifests': [{'path': str(manifest), 'sha256': coordinator.sha(manifest)}]}))
    monkeypatch.setattr(coordinator, 'authorize_preparation', lambda t, p: {'verified': True})
    calls = []
    def interrupted(argv, **kwargs):
        calls.append(argv)
        intent = json.loads(Path(row['submission_path']).read_text())
        assert intent['status'] == 'submitting' and intent['source_prepared_command'] == row['command']
        assert intent['preparation_route']['command'] == argv
        raise OSError('lost scheduler acknowledgement')
    first = routes.reconcile(tmp_path, 'minyeong', capacity, interrupted)
    second = routes.reconcile(tmp_path, 'minyeong', capacity, interrupted)
    assert first['errors'] and not second['submitted'] and len(calls) == 1


def test_changed_native_input_fails_before_any_claim(tmp_path, monkeypatch):
    from scripts.campaign_20261110 import coordinator_dispatch as coordinator
    monkeypatch.setattr(routes.time, 'time', lambda: 1000)
    d = tmp_path/'infra/coordinator_dispatch'; d.mkdir(parents=True); (d/'capacity_allocation.lock').touch()
    row = ticket(tmp_path); Path(row['command'][-2]).write_text('changed')
    manifest = tmp_path/'tickets.json'; manifest.write_text(json.dumps({'tickets': [row]}))
    (d/'policy.json').write_text(json.dumps({'owner': '/root', 'production_released': False,
        'prepared_manifests': [{'path': str(manifest), 'sha256': coordinator.sha(manifest)}]}))
    monkeypatch.setattr(coordinator, 'authorize_preparation', lambda t, p: {'verified': True})
    result = routes.reconcile(tmp_path, 'minyeong', capacity, lambda *a, **k: pytest.fail('must not submit'))
    assert result['errors'] and not Path(row['claim_path']).exists()


def test_typed_quota_bounds_aggregate_capacity_and_mem_per_cpu(tmp_path):
    obs = observation([node('viscam2', 'a40')]); obs['qos'][1][1] = 'gres/gpu=30,gres/gpu:a40=1'
    cap = capacity(obs)
    assert cap['preparation_slots'] == 2 and cap['account_slots'] == {'minyeong': 1, 'shiyuc': 1}
    row = ticket(tmp_path); row['command'][row['command'].index('--mem=48G')] = '--mem-per-cpu=6G'
    assert routes.sbatch_parts(row['command'])[-1]['memory_mb'] == 48*1024
    row['command'][row['command'].index('--mem-per-cpu=6G')] = '--mem-per-cpu=6Gc'
    with pytest.raises(ValueError, match='option scope'):
        routes.sbatch_parts(row['command'])


def test_unobserved_receipts_reserve_until_terminal_accounting(tmp_path, monkeypatch):
    from scripts.campaign_20261110 import coordinator_dispatch as coordinator
    d = tmp_path/'infra/coordinator_dispatch'; d.mkdir(parents=True)
    row = ticket(tmp_path); manifest = tmp_path/'tickets.json'; manifest.write_text(json.dumps({'tickets': [row]}))
    (d/'policy.json').write_text(json.dumps({'prepared_manifests': [{'path': str(manifest), 'sha256': coordinator.sha(manifest)}]}))
    routing = routes.route_ticket(row, capacity(), 'minyeong', now=1000)
    Path(row['submission_path']).write_text(json.dumps({'job_id': '123', 'preparation_route': routing}))
    monkeypatch.setattr(routes.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout=''))
    cap = routes.reserve_unobserved_receipts(tmp_path, capacity())
    assert cap['_reservations'] == [routing['route']]
    monkeypatch.setattr(routes.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='123|COMPLETED|\n'))
    assert not routes.reserve_unobserved_receipts(tmp_path, capacity())['_reservations']
    Path(row['submission_path']).write_text(json.dumps({'job_id': None, 'preparation_route': routing}))
    assert len(routes.reserve_unobserved_receipts(tmp_path, capacity())['_reservations']) == 1


def test_unavailable_capacity_is_explicit_owned_error_even_without_tickets(tmp_path):
    d = tmp_path/'infra/coordinator_dispatch'; d.mkdir(parents=True); (d/'capacity_allocation.lock').touch()
    (d/'policy.json').write_text(json.dumps({'owner': '/root', 'production_released': False, 'prepared_manifests': []}))
    result = routes.reconcile(tmp_path, 'minyeong', lambda: {'status': 'unavailable', 'error': 'sacctmgr unavailable'})
    assert result['status'] == 'blocked_capacity_unknown'
    assert result['errors'] == [{'owner': '/root', 'stage': 'capacity_probe', 'error': 'sacctmgr unavailable', 'capacity_status': 'unavailable'}]
    assert not result['submitted']
