import copy
import functools
import sys
from types import SimpleNamespace as N

import pytest
import torch

from ispatialgym.oracle.pool_reset_trace import contact_snapshot, reset_targets, trace_calls


def decorated(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        return fn(self, *args, **kwargs)
    return wrapper


class Native:
    def __init__(self):
        self.calls = []

    def __setattr__(self, key, value):
        if hasattr(self, 'calls'):
            raise AssertionError('observation replaced a native attribute')
        super().__setattr__(key, value)

    def reset(self, value, *, fail=False):
        self.calls.append(('reset', value, fail))
        self.step(value)
        for _ in range(3):
            self.render()
        if fail:
            raise RuntimeError('native failure')
        return value

    @decorated
    def step(self, value):
        self.calls.append(('step', value))

    def render(self):
        self.calls.append(('render',))


def test_trace_preserves_original_calls_arguments_return_values_and_object_bindings():
    native, other = Native(), Native()
    payload = object()
    assert native.reset(payload) is payload
    expected = list(native.calls)
    native.calls.clear()
    rows = []
    targets = [(name, native, name) for name in ('reset', 'step', 'render')]
    with trace_calls(targets, rows.append) as report:
        assert native.reset(payload) is payload
        other.reset(payload)  # same code, different instance: not observed
    assert native.calls == other.calls == expected
    assert [(r['operation'], r['boundary']) for r in rows] == [
        ('reset', 'before'), ('step', 'before'), ('step', 'after'),
        ('render', 'before'), ('render', 'after'), ('render', 'before'),
        ('render', 'after'), ('render', 'before'), ('render', 'after'), ('reset', 'after')]
    assert rows[0]['parent_call_id'] is None
    assert all(r['parent_call_id'] == 0 for r in rows[1:-1])
    assert report['completed'] and report['profile_hook_restored'] and report['events'] == 10
    assert sys.getprofile() is None


def test_native_exception_propagates_and_cleans_up_profile_hook():
    native = Native(); rows = []
    with pytest.raises(RuntimeError, match='native failure'):
        with trace_calls([('reset', native, 'reset')], rows.append) as report:
            native.reset(7, fail=True)
    assert [r['boundary'] for r in rows] == ['before', 'after']
    assert not report['completed'] and report['profile_hook_restored']
    assert sys.getprofile() is None and len(native.calls) == 5


def test_existing_profiler_is_preserved_without_executing_native_calls():
    native = Native()
    def existing(*args):
        pass
    sys.setprofile(existing)
    try:
        with pytest.raises(ValueError, match='unused Python profile hook'):
            with trace_calls([('reset', native, 'reset')], lambda event: None):
                native.reset(1)
        assert sys.getprofile() is existing and not native.calls
    finally:
        sys.setprofile(None)


@pytest.mark.parametrize('observation_error', [False, True])
def test_failed_observation_or_event_bound_restores_hook(observation_error):
    native = Native()
    def observe(event):
        if observation_error:
            raise ValueError('observer failed')
    with pytest.raises(ValueError, match='observer failed|finite event bound'):
        with trace_calls([('reset', native, 'reset')], observe, max_events=1) as report:
            native.reset(1)
    assert not report['completed'] and report['profile_hook_restored']
    assert sys.getprofile() is None


def test_contacts_copy_live_and_cached_values_without_refresh_or_buffer_mutation():
    matrices = {'current': torch.tensor([[True, False], [False, True]]),
                'interval': torch.tensor([[True, True], [False, True]]),
                'impulses': torch.tensor([[[0., 0., 2.], [0., 0., 0.]], [[0., 0., 0.], [0., 1., 0.]]]),
                'live': torch.tensor([[1., 2., 3., 0., 0., 0., 1.], [4., 5., 6., 0., 0., 0., 1.]]),
                'cached': torch.zeros((2, 7))}
    before = copy.deepcopy(matrices); reads = []
    def impulses(*, dt):
        reads.append(('impulses', dt)); return matrices['impulses']
    def live():
        reads.append(('live',)); return matrices['live']
    def forbidden(*args, **kwargs):
        raise AssertionError('observation refreshed native cache')
    api = N(_CONTACT_VIEW={0: N(get_contact_force_matrix=impulses)},
            _ROW_IDX_TO_PATH={0: ['/active', '/inactive']},
            _COL_IDX_TO_PATH={0: ['/support', '/other']},
            _RIGID_BODY_VIEW={0: N(get_transforms=live, prim_paths=['/active', '/inactive'])},
            _BODY_TRANSFORMS={0: matrices['cached']}, _CURRENT_CONTACT_MATRIX={0: matrices['current']},
            _CONTACT_MATRIX={0: matrices['interval']}, _PENDING_STEPS=1,
            update=forbidden, read_from_physx=forbidden, initialize_view=forbidden)
    out = contact_snapshot(api, 0, {'/active'}, 1 / 120)
    assert out['cached_current_pairs'] == [{'query': '/active', 'other': '/support', 'value': True}]
    assert len(out['cached_interval_pairs']) == 2
    assert out['live_impulses'] == [{'query': '/active', 'other': '/support', 'value': [0., 0., 2.]}]
    assert out['cached_change_detection_transforms']['/active'] == [0.] * 7
    assert out['live_body_transforms']['/active'] == [1., 2., 3., 0., 0., 0., 1.]
    assert reads == [('live',), ('impulses', 1 / 120)]
    assert all(torch.equal(value, before[key]) for key, value in matrices.items())
    matrices['live'][0, 0] = 99
    assert out['live_body_transforms']['/active'][0] == 1
    assert not contact_snapshot(api, 1, {'/active'}, 1 / 120)['available']


def test_native_deep_profiler_conflict_is_rejected_before_reset():
    with pytest.raises(ValueError, match='deep profiling'):
        reset_targets(None, N(_step_profiler=N(_cprofiler=object())))


def test_internal_pool_readback_is_detached_and_separate_from_legacy_boundaries(tmp_path, monkeypatch):
    from scripts.campaign_20261110.tests.test_native_pool_world import bound
    from ispatialgym.oracle import pool_reset_trace
    world, loader, _ = bound(tmp_path)
    state = {'root_link': {'lin_vel': torch.tensor([0.1, 0., 0.])}, 'is_asleep': False}
    pose = (torch.zeros(3), torch.tensor([0., 0., 0., 1.]))
    prim = N(GetAttribute=lambda name: N(Get=lambda: [0., 0., 0.]))
    link = N(prim=prim, prim_path='/active', get_position_orientation=lambda: pose)
    robot = N(prim=prim, root_link=link, base_footprint_link=link,
              _world_base_fixed_joint_prim=prim, get_position_orientation=lambda: pose,
              dump_state=lambda **kw: state, links={'root': link})
    loader.robot = robot
    loader.env = N(scene=N(idx=0, object_registry=lambda *args: robot))
    loader.report = {}
    sim = N(current_time=1., current_time_step_index=120, is_stopped=lambda: False,
            get_physics_dt=lambda: 1 / 120)
    api = object(); contacts = {'available': True, 'live_impulses': [1.]}
    monkeypatch.setitem(sys.modules, 'omnigibson.utils.usd_utils', N(RigidContactAPI=api))
    def read_contact(actual, scene, paths, dt):
        assert actual is api and scene == 0 and paths == {'/active'} and dt == 1 / 120
        return contacts
    monkeypatch.setattr(pool_reset_trace, 'contact_snapshot', read_contact)
    event = {'operation': 'scene.reset', 'boundary': 'after', 'call_id': 0}
    world.observe_reset('native_reset_internal', sim, internal=event)
    assert 'model_pool_reset_observations' not in loader.report
    row = loader.report['model_pool_reset_internal_observations'][0]
    state['root_link']['lin_vel'][0] = 5
    contacts['live_impulses'][0] = 99
    event['operation'] = 'later'
    assert row['robot']['state']['is_asleep'] is False
    assert row['robot']['state']['root_link']['lin_vel'][0] == pytest.approx(.1)
    assert row['native_contacts']['live_impulses'] == [1.]
    assert row['call']['operation'] == 'scene.reset'
