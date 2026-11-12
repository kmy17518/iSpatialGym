import ast
import copy
from contextlib import contextmanager
import glob
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from types import SimpleNamespace

import pytest
from scripts.campaign_20261110.infra_vast import async_results

from scripts.campaign_20261110.infra_vast.async_results import (
    CoordinatorStateCache, ResultFetchReservations, execute_reserved_fetch,
    identity, output_locks, typed_preparation, write,
)


def job(root, jid='9001', **updates):
    directory = root/jid
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({'kind': 'tabletop_maps', 'new_native_instances': 0,
                          'production_released': False, 'scenes': []}).encode()
    (directory/'native_inputs.json').write_bytes(payload)
    return {'id': jid, 'job_kind': 'native_dependency', 'name': 'isg261110_map_prebuild',
            'job_dir': str(directory), 'status': 'assigned', 'instance': 'worker1',
            'log': str(directory/'run.log'), 'assigned': 100, 'tries': 1,
            'history': [], 'native_inputs_sha256': hashlib.sha256(payload).hexdigest(), **updates}


@pytest.fixture
def managers():
    values = []
    def create(root, owner='10', limit=8):
        manager = ResultFetchReservations(root, owner, limit)
        values.append(manager)
        return manager
    yield create
    for manager in values:manager.close()


def await_poll(manager, jid):
    deadline = time.monotonic()+3
    while jid in manager.futures and time.monotonic() < deadline:
        manager.poll(); time.sleep(.005)
    assert jid not in manager.futures


def completion(row, value=3):
    return {'token': row['token'], 'identity': row['identity'], 'value': value}


def recovered_record(root, item, *, owner='9', status='fetching'):
    row = {'id': item['id'], 'owner': owner, 'status': status, 'job': item,
           'done': {'rc': 0}, 'identity': identity(item, {'rc': 0}), 'token': 'token',
           'result': str(root/'completed'/f"{item['id']}_token.json")}
    write(root/(item['id']+'.json'), row)
    return row


def test_only_hash_bound_consumed_preparation_enters_async(tmp_path, monkeypatch):
    item = job(tmp_path)
    assert typed_preparation(item)
    (Path(item['job_dir'])/'native_inputs.json').write_text('{}')
    with pytest.raises(ValueError, match='identity differs'):typed_preparation(item)
    monkeypatch.setattr(Path, 'read_bytes', lambda self: pytest.fail('legacy input opened'))
    assert not typed_preparation({'job_kind': 'physics_replay'})
    assert not typed_preparation({'job_kind': 'native_dependency', 'name': 'isg261110_native_dependency'})


@pytest.mark.parametrize('kind', ['tabletop', 'attachment', 'tabletop_maps', 'contact_maps'])
def test_consumed_native_payload_kinds(tmp_path, kind):
    item = job(tmp_path)
    path = Path(item['job_dir'])/'native_inputs.json'
    doc = json.loads(path.read_text()); doc['kind'] = kind
    path.write_text(json.dumps(doc)); item['native_inputs_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert typed_preparation(item)
    doc['scenes'] = ['per-scene-query']; path.write_text(json.dumps(doc))
    item['native_inputs_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert not typed_preparation(item)


def test_slow_fetch_does_not_block_second_completion_or_release_assignment(tmp_path, managers):
    manager = managers(tmp_path/'reservations', limit=2)
    first, second = job(tmp_path, '9001'), job(tmp_path, '9002')
    originals = copy.deepcopy([first, second]); done = {'rc': 0}
    wait = threading.Event()
    def fetch(row, path):
        assert json.loads(Path(path).read_text())['token'] == row['token']
        if row['id'] == '9001':assert wait.wait(3)
        return completion(row)
    try:
        assert manager.submit(first, done, fetch)
        assert manager.submit(second, done, fetch)
        assert not manager.submit(first, done, fetch)
        assert not manager.submit(job(tmp_path, '9003'), done, fetch)
        await_poll(manager, '9002')
        assert manager.result(second, done) == 3
        assert manager.result(first, done) is None
        assert manager.retains_assignment(first) and manager.retains_assignment(second)
        assert [first, second] == originals
        manager.committed('9002')
        assert not manager.retains_assignment(second)
        assert manager.submit(job(tmp_path, '9003'), done, fetch)
    finally:wait.set()


def test_output_directory_cannot_have_two_fetches(tmp_path, managers):
    manager = managers(tmp_path/'reservations')
    first = job(tmp_path); done = {'rc': 0}
    assert manager.submit(first, done, lambda row, path: completion(row))
    other = {**first, 'id': '9002'}
    assert not manager.submit(other, done, lambda *args: pytest.fail('duplicate output writer'))
    assert manager.metrics()['parallelism'] == 8
    with pytest.raises(ValueError):ResultFetchReservations(tmp_path/'bad', '10', 9)


def test_failure_keeps_partials_and_assignment_without_implicit_retry(tmp_path, managers):
    manager = managers(tmp_path/'reservations'); item = job(tmp_path); done = {'rc': 0}
    partial = Path(item['job_dir'])/'partial.tar'
    def fetch(row, path):
        partial.write_text('preserved')
        raise TimeoutError('bounded owned child timed out')
    assert manager.submit(item, done, fetch)
    await_poll(manager, item['id'])
    assert partial.read_text() == 'preserved'
    assert manager.retains_assignment(item)
    assert item['status'] == 'assigned'
    with pytest.raises(RuntimeError, match='timed out'):manager.result(item, done)
    assert not manager.submit(item, done, fetch)
    assert manager.metrics()['failed_or_interrupted'] == 1


def test_restart_respects_live_former_owner_and_surviving_child_lock(tmp_path, managers):
    root = tmp_path/'reservations'; item = job(tmp_path)
    row = recovered_record(root, item)
    manager = managers(root)
    manager.recover(lambda owner: True)
    assert manager.records[item['id']]['owner'] == '9'
    assert manager.result(item, row['done']) is None
    with output_locks(root, row):
        manager.recover(lambda owner: False)
        assert manager.records[item['id']]['status'] == 'fetching'
    manager.recover(lambda owner: False)
    assert manager.records[item['id']]['status'] == 'interrupted'
    assert manager.retains_assignment(item)
    assert not manager.submit(item, row['done'], lambda *args: pytest.fail('GPU/result retry'))


def test_restart_adopts_verified_completion_and_commit_after_crash(tmp_path, managers):
    root = tmp_path/'reservations'; item = job(tmp_path)
    row = recovered_record(root, item)
    write(row['result'], completion(row, 7))
    manager = managers(root)
    manager.recover(lambda owner: False)
    assert manager.result(item, row['done']) == 7
    saved = (root/(item['id']+'.json')).read_bytes()
    manager.recover(lambda owner: pytest.fail('already adopted owner'))
    assert (root/(item['id']+'.json')).read_bytes() == saved
    # Main persisted the terminal job before its process died.
    terminal = {**item, 'status': 'synced', 'done': row['done']}
    manager.reconcile_committed({item['id']: terminal})
    assert manager.records[item['id']]['status'] == 'committed'
    assert not manager.retains_assignment(terminal)


def test_wrong_completion_is_preserved_and_blocks_only_its_assignment(tmp_path, managers):
    root = tmp_path/'reservations'; item = job(tmp_path)
    row = recovered_record(root, item)
    wrong = {**completion(row), 'token': 'different'}; write(row['result'], wrong)
    manager = managers(root); manager.recover(lambda owner: False)
    assert json.loads(Path(row['result']).read_text()) == wrong
    with pytest.raises(RuntimeError, match='another assignment'):manager.result(item, row['done'])
    assert manager.retains_assignment(item)
    assert manager.submit(job(tmp_path, '9002'), {'rc': 0}, lambda row, path: completion(row))


def test_child_requires_exact_assigned_job_and_exclusive_output(tmp_path):
    root = tmp_path/'reservations'; item = job(tmp_path); original = copy.deepcopy(item)
    row = recovered_record(root, item); path = root/(item['id']+'.json')
    calls = []
    with pytest.raises(ValueError, match='assignment differs'):
        execute_reserved_fetch(path, {**item, 'tries': 2}, lambda value: calls.append(value))
    with pytest.raises(ValueError, match='assigned typed'):
        execute_reserved_fetch(path, {**item, 'status': 'synced'}, lambda value: calls.append(value))
    with output_locks(root, row):
        with pytest.raises(BlockingIOError):execute_reserved_fetch(path, item, lambda value: calls.append(value))
    assert calls == []
    assert execute_reserved_fetch(path, item, lambda value: 2)['value'] == 2
    assert item == original
    assert json.loads(path.read_text())['status'] == 'fetching'  # main owns state
    assert json.loads(Path(row['result']).read_text())['identity'] == row['identity']


def test_delayed_former_child_cannot_write_after_recovery(tmp_path, monkeypatch):
    root = tmp_path/'reservations'; item = job(tmp_path)
    row = recovered_record(root, item); path = root/(item['id']+'.json')
    @contextmanager
    def recovery_race(*args, **kwargs):
        write(path, {**row, 'status': 'interrupted'})
        yield
    monkeypatch.setattr(async_results, 'output_locks', recovery_race)
    with pytest.raises(ValueError, match='changed before child'):
        execute_reserved_fetch(path, item, lambda value: pytest.fail('retired child wrote output'))
    assert not Path(row['result']).exists()


def state_functions(tmp_path):
    """Execute actual I/O functions without loading provider credentials/clients."""
    source = Path(__file__).parents[1]/'infra_vast/vastctl.py'
    parsed = ast.parse(source.read_text())
    names = {'load_states', 'save_state', 'ingest'}
    module = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)
                             and node.name in names], type_ignores=[])
    state = tmp_path/'state'; state.mkdir()
    queue = tmp_path/'queue'; queue.mkdir()
    scope = {'os': os, 'glob': glob, 'json': json, 'uuid': uuid, 'Path': Path,
             'STATE': str(state), 'QUEUE': str(queue), 'log': lambda message: None,
             'JOB_STATE_CACHE': None}
    exec(compile(module, str(source), 'exec'), scope)
    cache = CoordinatorStateCache(scope['load_states']); scope['JOB_STATE_CACHE'] = cache
    return scope, cache, state, queue


def test_sole_owner_cache_ingests_new_queue_and_preserves_callback_visibility(tmp_path, monkeypatch):
    scope, cache, state, queue = state_functions(tmp_path)
    item = job(tmp_path); scope['save_state'](item)
    calls = []; original = cache.loader
    cache.loader = lambda: (calls.append('load'), original())[1]
    states = cache.get()
    (queue/'9001.json').write_text('immutable owned queue is deliberately not reread')
    item2 = job(tmp_path, '9002'); write(queue/'9002.json', item2)
    scope['ingest'](states, {'default_gpu': 'Titan RTX'})
    assert states['9002']['status'] == 'queued'
    assert json.loads((state/'9002.json').read_text()) == states['9002']
    # Return callback replaces a mapping item after its durable save.
    candidate = {**states['9002'], 'status': 'cluster_reserved'}
    scope['save_state'](candidate); states['9002'] = candidate
    assert cache.get()['9002']['status'] == 'cluster_reserved'
    assert calls == ['load']
    restarted = CoordinatorStateCache(scope['load_states'])
    assert restarted.get() == states and restarted.get() is not states


def test_failed_callback_save_invalidates_mutated_cache_before_dispatch(tmp_path, monkeypatch):
    scope, cache, state, queue = state_functions(tmp_path)
    item = job(tmp_path); scope['save_state'](item); states = cache.get()
    states[item['id']]['status'] = 'cluster_reserved'
    def fail(*args):raise OSError('injected replace failure')
    with monkeypatch.context() as local:
        local.setattr(os, 'replace', fail)
        with pytest.raises(OSError):scope['save_state'](states[item['id']])
    with pytest.raises(RuntimeError, match='persistence failed'):cache.require_durable()
    assert cache.get()[item['id']]['status'] == 'assigned'
    cache.require_durable()


def test_new_queue_not_owned_in_memory_until_persisted(tmp_path, monkeypatch):
    scope, cache, state, queue = state_functions(tmp_path)
    item = job(tmp_path); write(queue/'9001.json', item)
    states = cache.get()
    monkeypatch.setitem(scope, 'save_state', lambda item: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):scope['ingest'](states, {})
    assert states == {} and not (state/'9001.json').exists()


def test_interrupted_tick_and_restart_reload_durable_history(tmp_path):
    scope, cache, state, queue = state_functions(tmp_path)
    item = job(tmp_path); scope['save_state'](item)
    owned = cache.get(); owned[item['id']]['history'].append({'uncommitted': True})
    cache.invalidate()
    assert cache.get()[item['id']]['history'] == []
    owned = cache.get(); owned[item['id']]['history'].append({'durable': True})
    scope['save_state'](owned[item['id']]); cache.invalidate()
    assert cache.get()[item['id']]['history'] == [{'durable': True}]


class MetricsBoundary(Exception):
    pass


def coordinator_completion_scope(tmp_path, manager, item, fetch):
    """Run actual tick through result/lost reconciliation; stop before dispatch."""
    source = Path(__file__).parents[1]/'infra_vast/vastctl.py'
    parsed = ast.parse(source.read_text())
    module = ast.Module(body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)
                             and node.name == 'tick'], type_ignores=[])
    writes, legacy, costs, rows, observations = [], [], [], [], []
    done = {'rc': 0, 'elapsed_s': 2}
    def metrics(config, states, live, hbs, assigned, *args):
        observations.append(copy.deepcopy(assigned))
        raise MetricsBoundary()
    stage = SimpleNamespace(recover=lambda *args: None, active=lambda: {}, poll=lambda: [],
                            resize=lambda limit: None)
    scope = {'os': os, 'Path': Path, 'F': str(tmp_path), 'VJ': str(tmp_path), 'BUCKET': 'test',
             'PREFIX': 'test', 'STAGER': stage, 'RESULT_FETCHER': manager,
             'staging_parallelism': lambda config: 8,
             'cfg': lambda: {'retries': 2, 'transfer_reserve_usd': 300},
             'preserved_instances': lambda path: set(), 'ingest': lambda *args: None,
             'now': lambda: 200, 'my_instances': lambda c: ([], False),
             'observe_rentals': lambda *args: None, 'spend': lambda: (0, [], {}),
             'read_heartbeats': lambda *args: {}, 'r2_list': lambda *args: [item['id']+'.json'],
             'r2_json': lambda *args: done, 'typed_preparation': typed_preparation,
             'result_fetch_bounded': fetch, 'save_state': lambda value: writes.append(copy.deepcopy(value)),
             'log': lambda value: None, 'retain_attempt': lambda *args: {'retained': True},
             'cost_record': lambda *args, **kwargs: costs.append((args, kwargs)),
             'assignment_identity': lambda *args: 'assignment', 'cost_scope': lambda item: 'prep',
             'production_decision': lambda *args: {'allowed': False},
             'sync_back': lambda s3, value: (legacy.append(value['id']), 11)[1],
             'jobs_row': lambda value, done: rows.append(copy.deepcopy(value)),
             'publish_metrics': metrics}
    exec(compile(module, str(source), 'exec'), scope)
    scope['evidence'] = SimpleNamespace(writes=writes, legacy=legacy, costs=costs, rows=rows,
                                        observations=observations, done=done)
    return scope


def test_actual_tick_keeps_pending_assignment_then_main_commits_once(tmp_path, managers):
    manager = managers(tmp_path/'reservations'); item = job(tmp_path)
    block = threading.Event()
    def fetch(row, path):
        assert block.wait(3)
        return completion(row, 4)
    scope = coordinator_completion_scope(tmp_path, manager, item, fetch)
    ev = scope['evidence']; states = {item['id']: item}
    try:
        with pytest.raises(MetricsBoundary):scope['tick'](None, None, {}, states)
        assert item['status'] == 'assigned' and not ev.writes and not ev.costs and not ev.rows
        assert ev.observations == [{'worker1': [item['id']]}]
        # Even a missing remote done listing cannot turn an importing job into a retry.
        scope['r2_list'] = lambda *args: []
        scope['my_instances'] = lambda c: ([], True)
        with pytest.raises(MetricsBoundary):scope['tick'](None, None, {}, states)
        assert item['status'] == 'assigned' and not ev.writes
        block.set(); await_poll(manager, item['id'])
        scope['r2_list'] = lambda *args: [item['id']+'.json']
        with pytest.raises(MetricsBoundary):scope['tick'](None, None, {}, states)
        assert item['status'] == 'synced' and item['cache_added'] == 4
        assert len(ev.writes) == len(ev.costs) == len(ev.rows) == 1
        assert not ev.legacy and manager.records[item['id']]['status'] == 'committed'
        with pytest.raises(MetricsBoundary):scope['tick'](None, None, {}, states)
        assert len(ev.rows) == 1
    finally:block.set()


def test_actual_tick_legacy_result_uses_existing_synchronous_path(tmp_path, managers):
    manager = managers(tmp_path/'reservations')
    item = job(tmp_path, job_kind='physics_replay', name='isg261110_replay')
    scope = coordinator_completion_scope(tmp_path, manager, item,
                                         lambda *args: pytest.fail('legacy asynchronous fetch'))
    with pytest.raises(MetricsBoundary):scope['tick'](None, None, {}, {item['id']: item})
    assert item['status'] == 'synced' and item['cache_added'] == 11
    assert scope['evidence'].legacy == [item['id']]
    assert not manager.records
