import copy
from types import SimpleNamespace

import pytest

from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_source_anchor_probe import source_anchor, observed_no_steps, bind_anchor_function


def test_anchor_uses_source_components_without_composing_reference_roundoff():
    first = {'position': [1., 2., 3.], 'orientation': [0., 0., 0., 1.]}
    baseline = {'spawn_pose': first, 'anchor_position': list(first['position']),
                'anchor_orientation': [0., 0., 5e-8, 1.]}
    incoming = {'position': [13.49996566772461, 1.699981689453125, 0.005277156829833984],
                'orientation': [4.0265731513500214e-05, 7.211606862256303e-05, -0.5557019710540771, 0.831381618976593]}
    saved = copy.deepcopy(incoming)
    result = source_anchor(baseline, incoming)
    assert result == saved and incoming == saved and result is not incoming
    result['orientation'][0] = 0
    assert incoming == saved
    baseline['anchor_position'][0] += 1e-6
    with pytest.raises(ContractError, match='translation'): source_anchor(baseline, incoming)


def test_nonzero_anchor_rotation_is_not_silently_discarded():
    baseline = {'spawn_pose': {'position': [0]*3, 'orientation': [0, 0, 0, 1]},
                'anchor_position': [0]*3, 'anchor_orientation': [0, 0, .01, .99995]}
    with pytest.raises(ContractError, match='rotation'): source_anchor(baseline, baseline['spawn_pose'])


def test_existing_lifecycle_must_not_add_steps_and_wrappers_are_restored():
    class Sim:
        def step_physics(self): return None
        def get_physics_dt(self): return 1/120
        def get_rendering_dt(self): return 1/30
        def get_sim_step_dt(self): return 1/30
    sim = Sim(); sim._sim_context = SimpleNamespace(step=lambda: None)
    native = sim._sim_context.step
    result, evidence = observed_no_steps(sim, lambda: 'existing stop/play finished')
    assert result == 'existing stop/play finished' and not evidence['observed_physics_step_calls']
    with pytest.raises(ContractError, match='physics stepping'): observed_no_steps(sim, lambda: sim.step_physics())
    assert 'step_physics' not in vars(sim) and sim._sim_context.step is native


def test_python_m_parent_namespace_is_bound_instead_of_imported_module():
    namespace = {'__name__': '__main__', 'anchor_for_member': lambda: 'old SE3 calculation'}
    exec('def install():\n    return lambda: anchor_for_member()\n', namespace)
    parent = namespace['install']; native_attach = parent()
    bind_anchor_function(parent, lambda: 'exact source pose')
    assert native_attach() == 'exact source pose'


def test_missing_parent_anchor_route_fails_before_silent_noop():
    namespace = {'__name__': '__main__'}
    exec('def install():\n    return None\n', namespace)
    with pytest.raises(ContractError, match='route is absent'):
        bind_anchor_function(namespace['install'], lambda: None)
