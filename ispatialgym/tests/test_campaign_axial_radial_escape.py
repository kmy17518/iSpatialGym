from types import SimpleNamespace as NS

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.campaign_axial_contact import tangent_backoff_m
from ispatialgym.oracle.executor import SimState
from ispatialgym.oracle.motions import Motions, MotionsConfig
from ispatialgym.sampling import campaign_runtime as CR
from ispatialgym.tests.test_v2_oracle import _FkBackend, axial_world
from scripts.campaign_20261110.contracts import ContractError


def test_radial_escape_requires_exact_magazine_template(monkeypatch):
    monkeypatch.setattr(CR, 'active', lambda: False)
    assert not CR.contact_recovery('axial_magazine_radial_escape')
    obj = NS(role='target', category='magazine', model='tfovwy', default_scale=(1.,)*3)
    template = NS(class_id='alignment-axial_04', task='alignment-axial', split='eval_standard',
                  objects=[obj], scene_setup={'recovery_options': {'axial_magazine_radial_escape': True}})
    monkeypatch.setattr(CR, 'active', lambda: True)
    monkeypatch.setattr(CR, 'configuration', lambda: (None, template, None, None))
    assert CR.contact_recovery('axial_magazine_radial_escape')
    for o, key, value in [(obj, 'model', 'other'), (obj, 'default_scale', (.8,)*3),
                           (template, 'split', 'eval_model'), (template, 'class_id', 'alignment-axial_01')]:
        old = getattr(o, key); setattr(o, key, value)
        with pytest.raises(ContractError): CR.contact_recovery('axial_magazine_radial_escape')
        setattr(o, key, old)


def test_invalid_annotation_cannot_change_other_pushes():
    assert tangent_backoff_m(NS(extra={}, mid=None), .02) == .02
    for contact in [NS(extra={'campaign_axial_radial_escape': 1}, mid=object()),
                    NS(extra={'campaign_axial_radial_escape': True}, mid=None),
                    NS(extra={'campaign_axial_radial_escape': True, 'articulation': {'joint': 'lid'}}, mid=object())]:
        with pytest.raises(ValueError): tangent_backoff_m(contact, .02)


def test_closed_loop_push_keeps_its_goal_stop_and_omits_only_tangent_backoff():
    results = []
    for enabled in (False, True):
        sc, env, actions, board, brace, push, geom = axial_world(gain=1., tol=.10)
        push.contact.extra['campaign_axial_radial_escape'] = enabled
        motions = Motions(env, _FkBackend(env.robot), MotionsConfig(
            settle_steps=5, gripper_steps=5, cartesian_step=.05, cartesian_rot_step=.3))
        state = SimState(env, env.task, env.robot)
        ctx = {'cfg': EX.ExecuteConfig(push_arc_tracking='closed_loop', push_backoff_m=.02),
               'witnesses': {}, 'hooks': {}}
        targets = push.targets(sc, state)
        count = 0
        for action in EX.run_action(push, motions, state, targets, ctx):
            env.step(action); count += 1
        assert EX.check_postcondition(push, state, targets, ctx) is None
        assert ctx['push_arc_closed_loop']['stop'] == 'goal_band'
        results.append((ctx, count, state.eef_pose(push.contact.arm)[0].clone()))
    old, new = results
    assert old[0]['push_stopped'] == new[0]['push_stopped']
    assert old[1] > new[1]
    assert float(th.linalg.norm(old[2] - new[2])) == pytest.approx(.02, abs=.001)
    assert th.allclose(new[2], new[0]['push_end_pose'][0], atol=.001)
