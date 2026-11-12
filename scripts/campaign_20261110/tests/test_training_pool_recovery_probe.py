from types import SimpleNamespace as N
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_pool_recovery_probe import (
    parent_request, exception_type, install, verify_expected_abort)


@pytest.mark.parametrize('control', ['FRESH_A', 'AA', 'A_FAILURE_A', 'A_INTERRUPT_A'])
def test_scope_projection_preserves_inputs_and_requires_explicit_recovery(control):
    request = dict(pool_recovery_v1=True, pool_action_replay_v1=True, load_only=False,
                   diagnostic_control=control, production_released=False, benchmark_credit=0,
                   source_sequence={'sha256': 'exact'})
    projected = parent_request(request)
    assert projected['diagnostic_control'] == 'FRESH_A' and 'pool_recovery_v1' not in projected
    assert projected['source_sequence'] == request['source_sequence']
    assert request['diagnostic_control'] == control and request['pool_recovery_v1']
    with pytest.raises(ContractError): parent_request({**request, 'production_released': True})


@pytest.mark.parametrize('control', ['A_FAILURE_A', 'A_INTERRUPT_A'])
def test_native_action_happens_once_before_exception_and_next_episode_is_unpatched(control):
    calls = []
    class Env:
        def step(self, action): calls.append(action); return 'native'
    class Loader:
        def __init__(self): self.env = Env(); self.report = {}
        def restore(self): calls.append('restore'); return 'loaded'
    runner = N(InstanceLoader=Loader); current = {'ordinal': 0}; cleanups = []
    install(runner, control, current, cleanups); loader = Loader()
    assert loader.restore() == 'loaded' and calls == ['restore']
    kind = exception_type(control, 0)
    with pytest.raises(kind): loader.env.step('exact source action')
    assert calls == ['restore', 'exact source action'] and 'step' not in vars(loader.env)
    while cleanups: cleanups.pop()()
    result = {'success': False, 'error': kind.__name__+"('bounded')"}
    assert verify_expected_abort(control, 0, 1, result, {'loader': loader.report})
    with pytest.raises(ContractError):
        verify_expected_abort(control, 0, 0, result, {'loader': loader.report})
    current['ordinal'] = 1; next_loader = Loader()
    assert next_loader.restore() == 'loaded' and 'step' not in vars(next_loader.env)
    assert not verify_expected_abort(control, 1, 0, {'success': True}, {})


def test_cleanup_restores_native_step_when_episode_fails_before_action():
    class Env:
        def step(self, action): return action
    class Loader:
        def __init__(self): self.env = Env(); self.report = {}
        def restore(self): return None
    runner = N(InstanceLoader=Loader); cleanups = []
    install(runner, 'A_FAILURE_A', {'ordinal': 0}, cleanups); loader = Loader(); loader.restore()
    while cleanups: cleanups.pop()()
    assert loader.env.step('unchanged') == 'unchanged' and not loader.report['pool_controlled_abort']
    with pytest.raises(ContractError):
        verify_expected_abort('A_FAILURE_A', 0, 1, {'success': False, 'error': 'other'}, {'loader': loader.report})


def test_recovery_does_not_import_newer_unshipped_tray_helpers(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, 'scripts.campaign_20261110.training_batch_probe', N())
    assert exception_type('A_FAILURE_A', 0).__name__ == 'ControlledEpisodeFailure'
    assert issubclass(exception_type('A_INTERRUPT_A', 0), KeyboardInterrupt)
