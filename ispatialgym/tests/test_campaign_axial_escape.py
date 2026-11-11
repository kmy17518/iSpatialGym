"""Fixed-goal axial retreats must check the radial leg actually executed."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import program_check as PC


def test_axial_escape_is_scoped_and_never_changes_legacy_programs():
    push = SimpleNamespace(name='push', contact=SimpleNamespace(mid=object(), extra={}))
    sampler = SimpleNamespace()
    assert not PC._campaign_axial_push(sampler, push)
    sampler._campaign_surface_active = SimpleNamespace(task='alignment-world_vertical')
    assert not PC._campaign_axial_push(sampler, push)
    sampler._campaign_surface_active.task = 'alignment-axial'
    assert PC._campaign_axial_push(sampler, push)
    push.contact.extra = {'articulation': {'joint': 'hinge'}}
    assert not PC._campaign_axial_push(sampler, push)


def test_axial_escape_requires_tracked_push_end_and_rejects_native_failures(monkeypatch):
    import json
    fixture = Path(__file__).parent/'data/sidecars/alignment-axial.json'
    sc = Sidecar.from_dict(json.loads(fixture.read_text()))
    actions = expand(sc.variant, sc)
    retreat = next(a for a in actions if a.name == 'retreat' and getattr(getattr(a, 'after', None), 'name', None) == 'push')
    calls = []
    ikc = SimpleNamespace(q_locked=lambda *a, **kw: th.zeros(28),
        reaches=lambda q, pose, arm: float(q[0]) == .5,
        exact=lambda *a, **kw: pytest.fail('campaign axial must preserve the checked push branch'))
    sampler = SimpleNamespace(_campaign_surface_active=SimpleNamespace(task='alignment-axial'),
        objects={retreat.object: 'book'}, support_objects=lambda: ['desk'], ikc=ikc,
        kit=SimpleNamespace(robot=SimpleNamespace(arm_control_idx={'right': list(range(7)), 'left': list(range(7, 14))})))
    end = th.full((28,), .75)
    def escape(s, arm, q, obj, supports, ignore, dist, **kw):
        calls.append((q, dist, kw))
        return 'replayed', '', {}, end
    monkeypatch.setattr(PC, 'replay_contact_escape', escape)
    args = (sc, sampler, retreat, retreat.targets(sc, None), 'manipulation', None)
    n, witnesses = PC.check_action_targets(*args, q_end_prev=th.full((28,), .5))
    assert n == 1 and witnesses['__arm_end_q__'] is end
    assert calls[0][1] == .1 and 'fixed-goal axial' in calls[0][2]['start']
    with pytest.raises(Reject, match='checked end configuration'):
        PC.check_action_targets(*args, q_end_prev=th.full((28,), .2))
    monkeypatch.setattr(PC, 'replay_contact_escape', lambda *a, **kw: ('failed', 'native joint limit', {}, end))
    with pytest.raises(Reject, match='native joint limit'):
        PC.check_action_targets(*args, q_end_prev=th.full((28,), .5))
