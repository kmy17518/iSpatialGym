from dataclasses import replace
from types import SimpleNamespace

import pytest

from ispatialgym.sampling.v2.tasks.articulation_large import ArticulationLarge
from scripts.campaign_20261110.tests.test_joint_goals import plan


def test_requested_bucket_runs_unchanged_native_stability_test_and_restores_scene():
    p = replace(plan(), fraction_range=(.5, 1.))
    state = {'q': 0.}
    probes, steps, restores = [], [], []
    def set_joint(q):
        state['q'] = q
        probes.append(q)
    def step(n, still):
        steps.append(n)
        assert still == ['robot']
        if state['q'] < 1.75:
            state['q'] += .2
    def restore(value):
        restores.append(value)
        state.update(value)
    s = SimpleNamespace(direction='close', lower=0., upper=2., stable=[.36, .42], joint_friction=None,
        gp=SimpleNamespace(variant='dishwasher', gen=lambda key, default=None: {'open_fraction_range': [.18, .21]}.get(key, default)),
        campaign_group_remaining_s=lambda: 100, template_facts={}, probe_tol=.1, robot='robot',
        world=SimpleNamespace(snapshot=lambda: dict(state), restore=restore, step_physics=step),
        target=SimpleNamespace(keep_still=lambda: None), kit=SimpleNamespace(sync_world=lambda: None),
        set_joint=set_joint, joint_value=lambda: state['q'])
    s.holds_passively = lambda value: ArticulationLarge.holds_passively(s, value)
    ArticulationLarge.campaign_complete_stable_bucket(s, p)
    assert len(probes) == len(restores) == 8 and steps == [120]*8
    assert state == {'q': 0.} and s.joint_friction is None
    assert s.stable == pytest.approx([.36, .42, 1.75, 1.875, 2.])
    facts = s.template_facts['campaign_stable_probe']
    assert sum(row['stable'] for row in facts['probes']) == 3
    ArticulationLarge.campaign_complete_stable_bucket(s, p)
    assert len(probes) == 8  # An immutable goal plan is physically probed once.
