import copy
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110 import launch_prepared_kit_recoveries as launch
from scripts.campaign_20261110 import prep_cluster_routes as routes
from scripts.campaign_20261110.tests.test_prep_cluster_routes import observation, node, ticket


def pinned(tmp_path):
    value = ticket(tmp_path)
    value['sbatch_argv'] = value.pop('command')
    value['sbatch_argv'].insert(-2, '--nodelist=svl7')
    return value


def test_exact_node_preserves_command_and_four_titans(tmp_path):
    value = pinned(tmp_path); original = copy.deepcopy(value)
    obs = observation([node('svl7', 'titanrtx', partition='svl')])
    cap = routes.build_capacity(obs, now=1000)
    result = launch.fixed_route(value, cap, routes, now=1000)
    assert result['command'] == original['sbatch_argv'] and result['route']['node'] == 'svl7'
    assert result['critical_titan_slots_actually_held'] == 4 and value == original


def test_free_alternate_node_never_substitutes_for_unavailable_svl7(tmp_path):
    obs = observation([node('svl7', 'titanrtx', count=4, partition='svl'), node('svl8', 'titanrtx', partition='svl')])
    cap = routes.build_capacity(obs, now=1000)
    with pytest.raises(ValueError, match='svl7 unavailable'): launch.fixed_route(pinned(tmp_path), cap, routes, now=1000)


def test_unknown_or_expired_capacity_fails_closed(tmp_path):
    with pytest.raises(ValueError, match='capacity unavailable'): launch.fixed_route(pinned(tmp_path), {}, routes, now=1000)
    cap = routes.build_capacity(observation([node('svl7', 'titanrtx', partition='svl')]), now=1000)
    with pytest.raises(ValueError, match='expired'): launch.fixed_route(pinned(tmp_path), cap, routes, now=1100)


def test_existing_attempt_never_resubmits(tmp_path, monkeypatch):
    request = {'ticket': {'path': str(tmp_path / 'ticket.json')}, 'root_attempt_receipt_path': str(tmp_path / 'attempt_submission.json')}
    monkeypatch.setattr(launch, 'verify_request', lambda value: (request, {}, {}, {}))
    (tmp_path / 'attempt_intent.json').write_text('ambiguous attempt')
    def never(*args, **kwargs): raise AssertionError('existing intent must not call scheduler')
    result = launch.submit_once({}, tmp_path, SimpleNamespace(), execute=True, runner=never, writer=never)
    assert result['status'] == 'existing_attempt_requires_reconciliation'


def test_ambiguous_submit_preserves_original_claim_and_receipt_lineage(tmp_path, monkeypatch):
    import json
    import subprocess
    from contextlib import nullcontext
    claim = tmp_path / 'original_claim.json'; claim.write_text('{"physical":"key"}')
    previous = tmp_path / 'previous_submission.json'; previous.write_text('{"job_id":"old"}')
    global_receipt = tmp_path / 'global_receipt.json'; global_receipt.write_bytes(previous.read_bytes())
    terminal = tmp_path / 'terminal.json'; terminal.write_text('{"job_id":"old"}')
    proof = {'previous_claim': launch.ref(claim), 'previous_submission': launch.ref(previous), 'previous_terminal': launch.ref(terminal)}
    request = {'ticket': {'path': str(tmp_path / 'ticket.json')}, 'root_attempt_receipt_path': str(tmp_path / 'attempt_submission.json'),
               'original_claim_path': str(claim), 'original_submission_path': str(global_receipt)}
    value = {'physical_dedup_key': 'key', 'sbatch_argv': ['sbatch', '--parsable', 'exact.sh']}
    monkeypatch.setattr(launch, 'verify_request', lambda _: (request, value, proof, {'job_id': 'old'}))
    monkeypatch.setattr(launch, 'fixed_route', lambda *a, **k: {'route': {'node': 'svl7'}})
    fake_routes = SimpleNamespace(allocation_lock=lambda root: nullcontext(), reserve_unobserved_receipts=lambda *a: None)
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        if command[0] == 'sacct': return SimpleNamespace(stdout='old|FAILED|2:0\n')
        raise subprocess.TimeoutExpired(command, 60)
    def writer(path, value): __import__('pathlib').Path(path).write_text(json.dumps(value))
    result = launch.submit_once({}, tmp_path, fake_routes, execute=True, runner=runner, probe=lambda: {'status': 'observed'}, writer=writer)
    assert result['status'] == 'submission_requires_reconciliation' and launch.sha(claim) == proof['previous_claim']['sha256']
    saved = json.loads(global_receipt.read_text())
    assert saved['previous_submission'] == proof['previous_submission'] and saved['job_id'] is None
    assert (tmp_path / 'attempt_intent.json').exists()
    launch.submit_once({}, tmp_path, fake_routes, execute=True, runner=runner, writer=writer)
    assert len(calls) == 2
