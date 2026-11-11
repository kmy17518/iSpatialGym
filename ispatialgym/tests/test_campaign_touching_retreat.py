"""A tabletop push must not escape downward when its partner becomes a support."""
from types import SimpleNamespace as NS

import pytest
import torch

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.actions import Push, Retreat
from ispatialgym.oracle.motions import MotionsConfig
from ispatialgym.oracle.sidecar import Pose, PushContact
from ispatialgym.sampling import campaign_runtime as CR
from ispatialgym.tests.test_campaign_contact_recovery import bind
from ispatialgym.tests.test_touching_push_sweep import _sampler
from scripts.campaign_20261110.contracts import ContractError


def test_native_contact_policy_is_exact_and_default_preserving(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    option = 'touching_push_straight_retreat'
    assert not CR.contact_recovery(option)
    template = bind(monkeypatch, 'inter_object_relation_satisfaction-touching_04',
                    'inter_object_relation_satisfaction-touching_small_scale',
                    {'touching_push_stroke_m': .03, option: True})
    sampler = _sampler()
    extras = {'stop_clearance_m': .005}
    current = sampler.touch_push_strokes('push_tip_h-', extras, .08)
    assert current['straight_retreat'] is True
    template.scene_setup['recovery_options'][option] = False
    old = sampler.touch_push_strokes('push_tip_h-', extras, .08)
    assert {k: v for k, v in current.items() if k != 'straight_retreat'} == old
    template.scene_setup['recovery_options'][option] = True
    template.class_id = 'inter_object_relation_satisfaction-touching_01'
    with pytest.raises(ContractError):
        CR.contact_recovery(option)
    template.class_id = 'inter_object_relation_satisfaction-touching_04'
    template.scene_setup['recovery_options'][option] = 'true'
    with pytest.raises(ContractError, match='boolean'):
        CR.contact_recovery(option)


class Motions:
    cfg = MotionsConfig()

    def __init__(self):
        self.calls = []

    def arm_along(self, arm, leg, **kw):
        self.calls.append((leg, kw))
        yield from ()


def retreat(straight):
    # The source50270800 geometry, with the measured table/partner escape
    # directions. No simulator or collision verdict is stubbed into a pass.
    quat = [.5, .5, -.5, .5]
    goal = Pose([-9.2774, -3.4103, .918], quat)
    contact = PushContact('right', 'target',
                          Pose([-9.2774, -3.2795, .918], quat),
                          Pose([-9.2774, -3.3295, .918], quat), goal,
                          [0., -1., 0.], .0808,
                          extra={'straight_retreat': True} if straight else {})
    action = Retreat(arm='right', object='target', after=Push(contact=contact))
    here = torch.tensor([-9.2774, -3.3803, .918])
    target = NS(name='target')
    table = NS(name='table', aabb=(torch.tensor([-10., -4., .1]), torch.tensor([-8., -2., .868])))
    partner = NS(name='partner', aabb=(torch.tensor([-9.35, -3.6, .868]), torch.tensor([-9.2, -3.44, 1.1])))
    state = NS(eef_pose=lambda arm: (here.clone(), torch.tensor(quat)),
               object=lambda name: target, held_object=lambda arm: None,
               object_pose=lambda name: (torch.tensor([-9.2792, -3.4423, 1.0036]), torch.tensor(quat)),
               contact_supports=lambda name: [table, partner], robot=None)
    targets = action.targets(None, state)
    ctx = {'cfg': EX.ExecuteConfig(), 'hooks': {}}
    motions = Motions()
    list(EX.execute_retreat(action, motions, state, targets, ctx))
    return motions.calls[0], ctx, here


def test_measured_downward_escape_is_replaced_by_the_checked_horizontal_leg():
    (old_leg, old_checks), old_ctx, here = retreat(False)
    assert old_ctx['retreat_escape']['normal'] == pytest.approx([0., .707, .707])
    assert old_ctx['retreat_escape']['radial'][2] < -.70
    assert float(old_leg[-1][0][2]) == pytest.approx(float(here[2]) - .03535, abs=1e-4)
    (new_leg, checks), ctx, _ = retreat(True)
    assert 'retreat_escape' not in ctx
    positions = [p[0] if isinstance(p, tuple) else torch.tensor(p.position) for p in new_leg]
    assert all(float(p[2]) == pytest.approx(float(here[2])) for p in positions)
    assert all(float(p[1]) > float(here[1]) for p in positions)
    assert checks['check_collisions'] == old_checks['check_collisions'] is True
    assert checks['limit_margin'] == old_checks['limit_margin']
    assert [o.name for o in checks['ignore_objects']] == [o.name for o in old_checks['ignore_objects']]

