"""Train-only ungated demos (handoff 2026-11-14 section 3a).

Deadlines gate sampling only. A member published before its group's sampling
deadline keeps its demo; the result counts whenever it finishes (bounded by the
per-class demo_safety_cap_s), and the group does not finalize until it resolves.
Eval semantics are unchanged (covered by test_online_groups*.py).
"""
import json
import time
from types import SimpleNamespace

import pytest

import ispatialgym.sampling.online_groups as og
from ispatialgym.sampling.online_groups import OnlineGroups, result_deadline, verify_result


def make_worker(tmp_path, monkeypatch, *, groups, delays, outcome=lambda i: True, cap=5.0, group_limit=.4,
                member_limit=.4, end=200, ungated=True, real_verify=False):
    """Sampling/oracle replaced by deterministic adapters; member i's demo resolves `delays[i]` s after publication."""
    worker = OnlineGroups.__new__(OnlineGroups)
    worker.root = tmp_path; worker.state_path = tmp_path / 'state.json'; worker.poll_s = .01
    worker.options = {'groups': groups, 'group_limit_s': group_limit, 'member_limit_s': member_limit,
                      'require_qualifying_groups': False, 'demo_safety_cap_s': cap}
    worker.target = 3; worker.minimum = 1; worker.member_deadline = None; worker.ungated = ungated
    worker.state = {'owner': 'test', 'next_index': 100, 'index_end': end, 'groups': [], 'current': None}
    sampler = SimpleNamespace(world=object(), kit=object(), deadline=None)
    worker.sampler = sampler; worker.template = SimpleNamespace(class_id='test')
    worker.reset_goal = lambda: None; worker.activate = lambda: None
    ready = {}; calls = []

    def sample(n, max_attempts, indices):
        group = worker.state['current']; iid = str(indices[0])
        group.setdefault('plan', {'goal': group['id']})
        published = time.time()
        member = {'instance_id': iid, 'status': 'pending', 'group_id': group['id'],
                  'deadline': worker.member_deadline, 'published_at': published}
        if worker.ungated:
            member['demo_deadline'] = published + cap
        group['members'].append(member)
        ready[iid] = published + delays.get(iid, 1e9)
        calls.append((group['id'], iid))
        time.sleep(.12)  # one sampling attempt
        return [iid]
    sampler.run_instances = sample

    def results():
        now = time.time()
        return {i: {'instance': i, 'success': outcome(i), 'finished_at_unix': t} for i, t in ready.items() if t <= now}
    worker.results = results
    if not real_verify:
        def verify(member, result, accepted):
            if not result['success']: raise ValueError('oracle did not pass')
            # the train rule: published before the sampling deadline, finished within the cap
            assert member['published_at'] <= member['deadline']
            if result['finished_at_unix'] > result_deadline(member): raise ValueError('demo safety cap exceeded')
            return {'result': result}
        monkeypatch.setattr(og, 'verify_result', verify)
    return worker, calls, ready


def events(tmp_path):
    return [json.loads(l) for l in (tmp_path / 'events.jsonl').read_text().splitlines()]


def test_result_after_sampling_deadline_counts_and_group_waits(tmp_path, monkeypatch):
    # every demo finishes ~1.0 s after publication: after the 0.4 s group (sampling) deadline
    delays = {str(i): 1.0 for i in range(100, 200)}
    # two indices only: accepted + pending never reaches the target, so the group must hit its sampling deadline
    worker, calls, _ = make_worker(tmp_path, monkeypatch, groups=1, delays=delays, end=102)
    worker.run(3)
    (g,) = worker.state['groups']
    accepted = [m for m in g['members'] if m['status'] == 'accepted']
    assert accepted, 'demo finishing after the sampling deadline must count'
    assert all(m['evidence']['result']['finished_at_unix'] > g['deadline'] for m in accepted)
    assert g['finished_at'] >= max(m['evidence']['result']['finished_at_unix'] for m in accepted)  # finalized after resolution
    assert g['qualifies'] and g['accepted_members'] == len(accepted) == len(g['members'])
    ev = events(tmp_path)
    assert any(e['event'] == 'group_parked' and e.get('reason') == 'sampling_deadline_demos_in_flight' for e in ev)
    assert not any(m['status'] == 'expired' for m in g['members'])


def test_no_sampling_after_deadline_and_next_group_starts_while_demos_run(tmp_path, monkeypatch):
    delays = {str(i): 1.0 for i in range(100, 200)}
    worker, calls, _ = make_worker(tmp_path, monkeypatch, groups=2, delays=delays)
    worker.run(3)
    groups = worker.state['groups']
    assert len(groups) == 2
    for g in groups:
        for m in g['members']:
            assert m['published_at'] <= g['deadline']  # deadlines still gate sampling
    ev = events(tmp_path); order = [(e['event'], e.get('group_id')) for e in ev]
    first, second = groups[0]['id'], groups[1]['id']
    assert order.index(('group_started', second)) < order.index(('group_finalized', first))


def test_hung_demo_expires_only_at_safety_cap(tmp_path, monkeypatch):
    worker, calls, _ = make_worker(tmp_path, monkeypatch, groups=1, delays={}, cap=.8, end=101)
    t0 = time.time(); worker.run(3)
    (g,) = worker.state['groups']
    (m,) = g['members']
    assert m['status'] == 'expired' and m['reason'] == 'demo_safety_cap_without_result'
    assert g['finished_at'] >= m['demo_deadline'] and not g['qualifies']
    assert time.time() - t0 < 5


def test_failed_demo_after_deadline_finalizes_short(tmp_path, monkeypatch):
    delays = {str(i): .9 for i in range(100, 200)}
    worker, _, _ = make_worker(tmp_path, monkeypatch, groups=1, delays=delays, outcome=lambda i: False, end=102)
    worker.run(3)
    (g,) = worker.state['groups']
    assert all(m['status'] == 'rejected' for m in g['members']) and not g['qualifies']


def test_eval_semantics_unchanged_member_expires_at_member_deadline(tmp_path, monkeypatch):
    delays = {str(i): 1.0 for i in range(100, 200)}
    worker, _, _ = make_worker(tmp_path, monkeypatch, groups=1, delays=delays, ungated=False, end=101)
    worker.run(3)
    (g,) = worker.state['groups']
    assert all(m['status'] == 'expired' and m['reason'] in ('group_deadline', 'member_deadline_without_success')
               for m in g['members'])
    assert not g['qualifies']


def test_verify_result_train_rule(tmp_path):
    now = time.time()
    base = {'success': True, 'status': 'success', 'rc': 0, 'provisional': False, 'hdf5_ok': True,
            'tro': str(tmp_path / 'missing.json'), 'out': str(tmp_path)}
    cand = {'deadline': now, 'published_at': now - 1, 'demo_deadline': now + 100,
            'tro': {'path': str(tmp_path / 'missing.json'), 'sha256': 'x'}}
    # finished after the sampling deadline but within the cap: passes the deadline checks (fails later on the missing TRO)
    with pytest.raises((OSError, ValueError)) as exc:
        verify_result(cand, {**base, 'finished_at_unix': now + 50}, [])
    assert 'deadline' not in str(exc.value) and 'safety cap' not in str(exc.value)
    with pytest.raises(ValueError, match='demo safety cap exceeded'):
        verify_result(cand, {**base, 'finished_at_unix': now + 101}, [])
    with pytest.raises(ValueError, match='published after its sampling deadline'):
        verify_result({**cand, 'published_at': now + 1}, {**base, 'finished_at_unix': now + 2}, [])
    # eval candidate (no demo_deadline): the member deadline still gates the demo
    eval_cand = {k: v for k, v in cand.items() if k != 'demo_deadline'}
    with pytest.raises(ValueError, match='member deadline exceeded'):
        verify_result(eval_cand, {**base, 'finished_at_unix': now + 50}, [])


def test_first_group_extra_extends_only_the_first_group(tmp_path, monkeypatch):
    worker, _, _ = make_worker(tmp_path, monkeypatch, groups=2, delays={str(i): .05 for i in range(100, 200)}, end=200)
    worker.options['first_group_extra_s'] = .5
    worker.run(3)
    g1, g2 = worker.state['groups'][:2]
    assert abs((g1['deadline'] - g1['started_at']) - .9) < .05 and abs((g2['deadline'] - g2['started_at']) - .4) < .05
