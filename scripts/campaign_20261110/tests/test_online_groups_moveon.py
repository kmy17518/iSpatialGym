"""Move-on scheduling: parked groups finalize asynchronously while the sampler starts the next group."""
import json
import time
from types import SimpleNamespace

import ispatialgym.sampling.online_groups as og
from ispatialgym.sampling.online_groups import OnlineGroups


def make_worker(tmp_path, monkeypatch, *, groups, outcome, delay=.15, start=100, end=200, state=None):
    """Physics and oracle replaced by deterministic adapters; collectors resolve after `delay`."""
    worker = OnlineGroups.__new__(OnlineGroups)
    worker.root = tmp_path; worker.state_path = tmp_path / 'state.json'; worker.poll_s = .01
    worker.options = {'groups': groups, 'group_limit_s': 30, 'member_limit_s': 20, 'require_qualifying_groups': True}
    worker.target = 3; worker.minimum = 2; worker.member_deadline = None
    worker.state = state or {'owner': 'test', 'next_index': start, 'index_end': end, 'groups': [], 'current': None}
    sampler = SimpleNamespace(world=object(), kit=object(), deadline=None)
    worker.sampler = sampler; worker.template = SimpleNamespace(class_id='test')
    worker.reset_goal = lambda: None; worker.activate = lambda: None
    ready = {}; calls = []
    def sample(n, max_attempts, indices):
        group = worker.state['current']; index = indices[0]
        assert n == 1 and sampler._campaign_group_written < 3
        group.setdefault('plan', {'goal': group['id']})
        iid = str(index)
        group['members'].append({'instance_id': iid, 'status': 'pending', 'group_id': group['id'],
                                 'deadline': worker.member_deadline, 'published_at': time.time()})
        ready[iid] = time.time() + delay
        calls.append((group['id'], iid))
        return [iid]
    sampler.run_instances = sample
    def results():
        now = time.time()
        return {i: {'instance': i, 'success': outcome(i), 'finished_at_unix': t} for i, t in ready.items() if t <= now}
    worker.results = results
    def verify(member, result, accepted):
        if not result['success']: raise ValueError('oracle did not pass')
        return {'result': result}
    monkeypatch.setattr(og, 'verify_result', verify)
    return worker, calls, ready


def events(tmp_path):
    return [json.loads(l) for l in (tmp_path / 'events.jsonl').read_text().splitlines()]


def test_next_group_starts_while_previous_members_are_in_flight(tmp_path, monkeypatch):
    worker, calls, _ = make_worker(tmp_path, monkeypatch, groups=3, outcome=lambda i: True)
    worker.run(3)
    groups = worker.state['groups']
    assert [g['accepted_members'] for g in groups] == [3, 3, 3] and all(g['qualifies'] for g in groups)
    assert worker.state['open'] == [] and worker.state['current'] is None
    assert len(calls) == 9 and len({i for _, i in calls}) == 9
    ev = events(tmp_path); order = [(e['event'], e.get('group_id')) for e in ev]
    first = calls[0][0]
    second = calls[3][0]
    assert order.index(('group_started', second)) < order.index(('group_finalized', first))  # overlap
    assert sum(e['event'] == 'group_parked' for e in ev) == 3
    assert ev[-1]['event'] == 'worker_stopped' and ev[-1]['finalized'] == 3
    # exact attribution: every member sits in the group it was sampled for
    sampled = {i: g for g, i in calls}
    for g in groups:
        assert g['parked_resolution'] == 'members_resolved'
        for m in g['members']:
            assert m['group_id'] == g['id'] == sampled[m['instance_id']] and m['status'] == 'accepted'
        assert set(g) >= {'id', 'started_at', 'deadline', 'members', 'finished_at', 'accepted_members', 'qualifies', 'reason', 'timing'}
        assert len(g['timing']['seconds_to_accepted_members']) == 3
    fin = {e['group_id']: e for e in ev if e['event'] == 'group_finalized'}
    assert set(fin) == {g['id'] for g in groups}
    assert all(set(e) >= {'group_id', 'accepted', 'qualifies', 'reason', 'timing'} for e in fin.values())
    saved = json.loads((tmp_path / 'state.json').read_text())
    assert [g['id'] for g in saved['groups']] == [g['id'] for g in groups] and saved['open'] == []


def test_parked_group_that_finalizes_short_is_replaced_and_total_never_exceeds(tmp_path, monkeypatch):
    # group 1: two of three in-flight members fail -> accepted 1 < minimum -> not qualifying
    bad = {'101', '102'}
    worker, calls, _ = make_worker(tmp_path, monkeypatch, groups=2, outcome=lambda i: i not in bad)
    worker.run(3)
    groups = worker.state['groups']
    assert sum(g['qualifies'] for g in groups) == 2
    assert [g['qualifies'] for g in groups].count(False) == 1
    short = next(g for g in groups if not g['qualifies'])
    assert short['accepted_members'] == 1 and short['reason'] == 'budget_or_indices_exhausted'
    assert len({g for g, _ in calls}) == 3  # one replacement group


def test_one_rejection_keeps_qualifying_parked_group_and_no_extra_group(tmp_path, monkeypatch):
    worker, calls, _ = make_worker(tmp_path, monkeypatch, groups=1, outcome=lambda i: i != '101')
    worker.run(3)
    assert len(calls) == 3 and len(worker.state['groups']) == 1
    g = worker.state['groups'][0]
    assert g['qualifies'] and g['accepted_members'] == 2 and g['reason'] == 'budget_or_indices_exhausted'
    assert not any(e['event'] == 'group_started' and e['group_id'] != g['id'] for e in events(tmp_path))


def test_resume_finalizes_parked_group_from_state_without_sampling_into_it(tmp_path, monkeypatch):
    now = time.time()
    parked = {'id': 'online-parked', 'started_at': now - 5, 'deadline': now + 30, 'parked_at': now - 1, 'plan': {'goal': 1},
              'members': [{'instance_id': str(i), 'status': 'pending', 'group_id': 'online-parked', 'deadline': now + 20}
                          for i in (100, 101, 102)]}
    state = {'owner': 'test', 'next_index': 103, 'index_end': 200, 'groups': [], 'current': None, 'open': [parked]}
    worker, calls, ready = make_worker(tmp_path, monkeypatch, groups=2, outcome=lambda i: True, state=state)
    for i in ('100', '101', '102'): ready[i] = time.time() + .1
    worker.run(3)
    ids = [g['id'] for g in worker.state['groups']]
    assert 'online-parked' in ids and len(ids) == 2 and all(g['qualifies'] for g in worker.state['groups'])
    assert all(g != 'online-parked' for g, _ in calls) and len(calls) == 3


def test_parked_members_expire_at_their_deadline(tmp_path, monkeypatch):
    worker, calls, ready = make_worker(tmp_path, monkeypatch, groups=1, outcome=lambda i: True, delay=1e9, end=103)
    worker.options['member_limit_s'] = .3; worker.options['group_limit_s'] = .5
    t0 = time.time(); worker.run(3)
    g = worker.state['groups'][0]
    assert not g['qualifies'] and all(m['status'] == 'expired' for m in g['members'])
    assert time.time() - t0 < 5


def test_member_from_another_group_is_refused(tmp_path, monkeypatch):
    worker, _, _ = make_worker(tmp_path, monkeypatch, groups=1, outcome=lambda i: True)
    group = {'id': 'a', 'members': [{'instance_id': '1', 'status': 'pending', 'group_id': 'b', 'deadline': time.time() + 9}]}
    try:
        worker.collect_group(group, {'1': {'instance': '1', 'success': True, 'finished_at_unix': time.time()}})
    except ValueError as e:
        assert 'another group' in str(e)
    else:
        raise AssertionError('cross-group member accepted')
