import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ispatialgym.sampling.online_groups import GoalRegistry, OnlineGroups, verify_result
from ispatialgym.sampling import goal_identity as identity
from ispatialgym.sampling.online_registry import mailbox_claim, serve_once


def goal(x=0):
    pose = {'position': [x, 0, .7], 'orientation': [0, 0, 0, 1]}
    return {'class_id': 'inter_object_relation_satisfaction-nextto_04',
            'requested_goal': {'target': pose, 'partner': {**pose, 'position': [0, .2, .7]}}, 'goal_group': {}}


def test_concurrent_workers_cannot_claim_numerically_equivalent_goal(tmp_path):
    registry = GoalRegistry(tmp_path)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda n: registry.claim(goal(n * .00001), str(n), time.time() + 10), range(8)))
    assert sum(x['claimed'] for x in results) == 1
    assert registry.claim(goal(.1), 'fresh', time.time() + 10)['claimed']
    assert registry.claim(goal(.1), 'fresh', time.time() + 10)['resumed']
    with pytest.raises(ValueError, match='changed physical goal'):
        registry.claim(goal(.2), 'fresh', time.time() + 10)


def test_native_goal_includes_partner_held_base_and_block_pattern(tmp_path):
    a = goal(); b = copy.deepcopy(a)
    b['requested_goal']['partner']['position'][0] += .2
    assert not identity.equivalent_goal(a, b)
    a = {'class_id': 'mental_rotation-object', 'requested_goal': goal()['requested_goal']['target'],
         'goal_group': {'base_pose2d': [0, 0, 0]}}
    b = copy.deepcopy(a); b['goal_group']['base_pose2d'][2] = .1
    assert not identity.equivalent_goal(a, b)
    p = goal()['requested_goal']['target']
    a = {'class_id': 'configuration_matching-block_design_01', 'requested_goal': {'board': p, 'slots': [p] * 4},
         'goal_group': {'pattern': ['red'] * 4}}
    b = copy.deepcopy(a); b['goal_group']['pattern'][3] = 'blue'
    assert not identity.equivalent_goal(a, b)


def test_cross_host_mailbox_is_a_claim_inside_worker_not_job_dispatch(tmp_path):
    mailbox, registry = str(tmp_path / 'mailbox'), str(tmp_path / 'registry')
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(mailbox_claim, mailbox, goal(), 'og-isg261110-worker/g1', time.time() + 10)
        for _ in range(50):
            serve_once(mailbox, registry)
            if pending.done(): break
            time.sleep(.05)
        assert pending.result()['claimed']
    assert len(list((tmp_path / 'registry').rglob('*.json'))) == 1


def test_two_groups_one_worker_oracle_failures_do_not_fill_quota(tmp_path, monkeypatch):
    # Test the actual persistent control loop, with physics and oracle replaced
    # by deterministic adapters. GPU proof is a separate required artifact.
    worker = OnlineGroups.__new__(OnlineGroups)
    worker.root = tmp_path; worker.state_path = tmp_path / 'state.json'
    worker.options = {'groups': 2, 'group_limit_s': 60, 'member_limit_s': 30}
    worker.target = 3; worker.minimum = 2; worker.member_deadline = None
    worker.state = {'owner': 'test', 'next_index': 40, 'index_end': 60, 'groups': [], 'current': None}
    sampler = SimpleNamespace(world=object(), kit=object(), deadline=None)
    worker.sampler = sampler
    worker.template = SimpleNamespace(class_id="test")
    calls = []
    def sample(n, max_attempts, indices):
        group = worker.state['current']; index = indices[0]
        assert n == 1 and sampler._campaign_group_written < 3
        if not group.get('plan'): group['plan'] = {'goal': len(worker.state['groups'])}
        group['members'].append({'instance_id': str(index), 'status': 'pending'})
        calls.append((id(sampler.world), group['id'], index))
        return [str(index)]
    sampler.run_instances = sample
    worker.reset_goal = lambda: None
    worker.activate = lambda: None
    def collect():
        g = worker.state['current']
        for m in g['members']:
            if m['status'] == 'pending':
                m['status'] = 'rejected' if m['instance_id'] == '40' else 'accepted'
                m['evidence'] = {'result': {'finished_at_unix': time.time()}}
        return sum(m['status'] == 'accepted' for m in g['members']), 0
    worker.collect = collect
    worker.run(3)
    assert len(calls) == 7
    assert len({world for world, _, _ in calls}) == 1
    assert len({g for _, g, _ in calls}) == 2
    assert [g['accepted_members'] for g in worker.state['groups']] == [3, 3]
    assert all(g['qualifies'] for g in worker.state['groups'])
    assert len({i for _, _, i in calls}) == 7


def test_deadline_success_and_provisional_world_are_not_accepted():
    candidate = {'deadline': 100}
    base = {'success': True, 'status': 'success', 'rc': 0, 'hdf5_ok': True, 'provisional': False,
            'finished_at_unix': 101}
    with pytest.raises(ValueError, match='deadline'):
        verify_result(candidate, base, [])
    with pytest.raises(ValueError, match='provisional'):
        verify_result(candidate, {**base, 'finished_at_unix': 90, 'provisional': True}, [])
