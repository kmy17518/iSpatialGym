from types import SimpleNamespace as NS

import pytest
import torch

from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_contact_reset_probe import (
    MATRICES, clear_observations, around_one_reset_step)


def api():
    return NS(_PENDING_STEPS=0, **{name: {0: torch.tensor([[True, False], [False, True]]),
                                       1: torch.ones((2, 2), dtype=torch.bool)} for name in MATRICES})


def test_reset_preserves_storage_aliases_other_scenes_and_native_views():
    a = api(); a._CONTACT_VIEW = {'native': object()}; a._BODY_TRANSFORMS = object()
    aliases = {name: getattr(a, name)[0].view(torch.uint8) for name in MATRICES}
    native = a._CONTACT_VIEW; transforms = a._BODY_TRANSFORMS; events = []
    evidence = clear_observations(a, 0, synchronize=lambda: events.append('sync'))
    assert events == ['sync', 'sync'] and evidence['pairs_before'] == dict.fromkeys(MATRICES, 2)
    assert evidence['pairs_after'] == dict.fromkeys(MATRICES, 0)
    assert all(not tensor.any() for tensor in aliases.values())
    assert all(getattr(a, name)[1].all() for name in MATRICES)
    assert a._CONTACT_VIEW is native and a._BODY_TRANSFORMS is transforms


@pytest.mark.parametrize('fault', ['pending', 'missing_mirror', 'shape', 'dtype'])
def test_invalid_or_unconsumed_observations_fail_before_any_mutation(fault):
    a = api()
    if fault == 'pending': a._PENDING_STEPS = 1
    elif fault == 'missing_mirror': del a._CURRENT_CONTACT_MATRIX_GPU[0]
    elif fault == 'shape': a._CURRENT_CONTACT_MATRIX_GPU[0] = torch.ones((3, 2), dtype=torch.bool)
    else: a._CURRENT_CONTACT_MATRIX_GPU[0] = torch.ones((2, 2))
    with pytest.raises(ContractError): clear_observations(a, 0)
    assert a._CONTACT_MATRIX[0].count_nonzero() == 2


def test_cache_invalidated_after_state_restore_before_exact_existing_step():
    events = []
    class Sim:
        def step_physics(self, *args, **kwargs): events.append(('step', args, kwargs)); return 'native'
    sim = Sim()
    class Scene:
        def reset(self, hard=True):
            events.append(('state_restore', hard))
            return sim.step_physics('unchanged', native=True)
    scene = Scene()
    result, evidence = around_one_reset_step(scene, sim, lambda: scene.reset(hard=False),
        lambda: events.append('clear') or {'measured': True})
    assert result == 'native' and evidence == {'measured': True}
    assert events == [('state_restore', False), 'clear', ('step', ('unchanged',), {'native': True})]
    assert 'reset' not in vars(scene) and 'step_physics' not in vars(sim)


def test_original_callbacks_restored_on_native_failure_without_another_step():
    events = []
    def step(): events.append('step'); raise RuntimeError('native failure')
    sim = NS(step_physics=step); scene = NS(reset=lambda: sim.step_physics())
    reset = scene.reset
    with pytest.raises(RuntimeError, match='native failure'):
        around_one_reset_step(scene, sim, lambda: scene.reset(), lambda: events.append('clear'))
    assert scene.reset is reset and sim.step_physics is step and events == ['clear', 'step']
