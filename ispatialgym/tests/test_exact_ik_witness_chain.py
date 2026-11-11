"""Exercise real exact() chain bookkeeping with deterministic solver/world doubles."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ispatialgym.sampling import exact_exchange
from ispatialgym.sampling.v2 import ik as IK


class Solver(IK.IKChecker):
    WITNESS_RESIDUAL_AT_DRAW = None

    def __init__(self):
        self.kit = SimpleNamespace(robot=SimpleNamespace(
            arm_control_idx={'right': torch.tensor([0, 1, 2]), 'left': torch.tensor([3, 4])},
            gripper_control_idx={'right': torch.tensor([5])}))
        self.solutions = {}
        self.counts = {}
        self.solver_calls = 0

    def _count(self, name): self.counts[name] = self.counts.get(name, 0) + 1
    def q_locked(self, *args, **kwargs): return torch.zeros(6)
    def _floor_chain_supports(self, *args): return []
    def exact_cache_scope(self): return {}
    def _touch_objects(self, *args): return []
    def ignoring(self, *args): return nullcontext()
    def _seeded(self, q, *args): return q
    def _pose_key(self, p): return tuple(p[0])
    def approached_objects(self, *args): return []
    def joint_limit_margin(self, *args): return (1., 'joint')
    def away_from_limits(self, q, *args): return q, ''
    def _note_policed(self, *args): pass
    def _fresh_draw(self): return False

    def _solve_exact(self, pose, q, arm):
        self.solver_calls += 1
        out = q.clone()
        out[0] = pose[0][0]
        return out

    def track_approach(self, q, previous, pose, arm, **kwargs):
        self._last_track_start = q.clone()
        self._last_track_start[1] = pose[0][0]  # backward tracking revises previous witness
        out = q.clone()
        out[0] = pose[0][0]
        return out, ''

    def _witness_ok(self, q, pose, arm):
        return torch.equal(q, self.solutions[(arm, self._pose_key(pose))])


@pytest.mark.parametrize('approaches', [False, True])
@pytest.mark.parametrize('count', [1, 2, 3])
def test_each_query_pose_has_one_correct_witness_and_can_be_reused(monkeypatch, approaches, count):
    saved = {}
    monkeypatch.setattr(IK, 'cache_key', lambda *args: 'test-key')
    monkeypatch.setattr(IK, 'cache_get', lambda key: saved.get(key))
    monkeypatch.setattr(IK, 'cache_put', lambda key, value: saved.__setitem__(key, value))
    monkeypatch.setattr(IK, 'fast_on', lambda *args, **kwargs: False)
    monkeypatch.setattr(IK, 'support_policy_on', lambda: False)
    monkeypatch.setattr(IK, 'is_approach', lambda *args: approaches)
    monkeypatch.setattr(IK.SS, 'cache_identity', lambda *args: {})
    monkeypatch.setattr(exact_exchange, 'register_query', lambda *args: None)
    solver = Solver()
    stance = SimpleNamespace(base_pose2d=(0., 0., 0.), translate=0., tilt=0.)
    poses = [([float(i + 1), 0., 0.], [0., 0., 0., 1.]) for i in range(count)]
    assert solver.exact(poses, stance).ok
    entry = saved['test-key']
    assert len(entry['qs']) == count
    assert entry['q'] == entry['qs'][-1]
    assert entry['qs'] == [solver.solutions[('right', solver._pose_key(p))].tolist() for p in poses]
    calls = solver.solver_calls
    result = solver.exact(poses, stance)
    assert result.ok and result.reason == 'cache'
    assert solver.solver_calls == calls
    assert solver.counts['exact_verified_hit'] == 1


def test_same_branch_growth_replaces_invalid_old_primary_and_prunes_bad_alternatives():
    solver = Solver()
    q = [0.] * 6
    new = {'ok': True, 'q': q, 'qs': [q, q]}
    old = {**new, 'qs': [q, q, q]}
    assert solver._grown_entry(old, new, 'right') == new
    old = {**new, 'alts': [[q, q, q]]}
    grown = solver._grown_entry(old, new, 'right')
    assert grown['qs'] == new['qs'] and grown['alts'] == []
    assert old['alts'] == [[q, q, q]]  # no mutation of historical evidence
