"""Observe existing native reset calls without replacing methods or physics state."""
from contextlib import contextmanager
import inspect
import sys


@contextmanager
def trace_calls(targets, observe, *, max_events=256):
    """Profile only exact bound Python methods on this thread during one reset.

    Observation runs outside Python profiling recursively. Native methods keep
    their original arguments, return values, exception behavior and bindings.
    A return event can also mean exception unwinding; the scope records success
    separately. No profiler is silently displaced.
    """
    previous = sys.getprofile()
    if previous is not None:
        raise ValueError('native reset observation requires an unused Python profile hook')
    methods = {}
    for label, owner, name in targets:
        method = getattr(owner, name)
        fn = inspect.unwrap(getattr(method, '__func__', method))
        if not hasattr(fn, '__code__') or not fn.__code__.co_argcount:
            raise ValueError('native reset method has no inspectable Python body: ' + label)
        methods.setdefault(fn.__code__, []).append((label, owner, fn.__code__.co_varnames[0]))
    calls, stack = {}, []
    summary = {'completed': False, 'events': 0, 'operations': [],
               'native_method_replacements': 0, 'physics_steps_requested': 0,
               'state_writes': 0, 'profile_hook_restored': False}

    def profile(frame, event, arg):
        if event not in ('call', 'return') or frame.f_code not in methods:
            return
        matches = [label for label, owner, key in methods[frame.f_code]
                   if frame.f_locals.get(key) is owner]
        if not matches:
            return
        if len(matches) != 1:
            raise ValueError('ambiguous native reset observation target')
        if summary['events'] >= max_events:
            raise ValueError('native reset observation exceeded its finite event bound')
        label = matches[0]
        if event == 'call':
            call = {'operation': label, 'call_id': len(summary['operations']),
                    'parent_call_id': stack[-1] if stack else None,
                    'python_function': getattr(frame.f_code, 'co_qualname', frame.f_code.co_name)}
            calls[id(frame)] = call
            summary['operations'].append(dict(call))
            stack.append(call['call_id'])
        else:
            call = calls.pop(id(frame))
            if not stack or stack.pop() != call['call_id']:
                raise ValueError('native reset observation lost call nesting')
        observe({**call, 'boundary': 'before' if event == 'call' else 'after',
                 'event_index': summary['events']})
        summary['events'] += 1

    sys.setprofile(profile)
    try:
        yield summary
        if sys.getprofile() is not profile or calls or stack:
            raise ValueError('native reset observation hook or call sequence changed')
        summary['completed'] = True
    finally:
        sys.setprofile(previous)
        summary['profile_hook_restored'] = sys.getprofile() is previous


def reset_targets(loader, sim):
    """Only operations already called by InstanceLoader's final native reset."""
    for name in ('_step_profiler', '_pre_physics_step_profiler',
                 '_post_physics_step_profiler', '_non_physics_step_profiler'):
        if getattr(getattr(sim, name, None), '_cprofiler', None) is not None:
            raise ValueError('native deep profiling would replace the reset observation hook')
    return [(label + '.' + name, owner, name) for label, owner, names in (
        ('scene', loader.env.scene, ('update_initial_file', 'reset', 'restore', 'load_state')),
        ('task', loader.task, ('_reset_tracking', 'reset')),
        ('env', loader.env, ('reset', 'get_obs')),
        ('sim', sim, ('step_physics', 'step', 'render')),
        ('loader', loader, ('restore_after_reset', '_check_tro_poses')),
    ) for name in names]


def contact_snapshot(api, scene_index, paths, physics_dt):
    """Copy native cached contacts and live tensor-view reads; never refresh them.

    Cached pair flags and change-detection transforms may intentionally describe
    an earlier step. Report them separately from live PhysX reads. Do not call
    update/read_from_physx/initialize_view or assign into any native buffer.
    """
    view = api._CONTACT_VIEW.get(scene_index)
    if view is None:
        return {'available': False, 'reason': 'native contact view absent'}
    rows = api._ROW_IDX_TO_PATH[scene_index]
    cols = api._COL_IDX_TO_PATH[scene_index]
    selected = [(i, path) for i, path in enumerate(rows) if path in paths]

    def sparse(matrix):
        out = []
        # Indexing and CPU copies are detached reads of native buffers.
        for i, path in selected:
            row = matrix[i].detach().cpu().tolist()
            for j, value in enumerate(row):
                nonzero = any(v != 0 for v in value) if isinstance(value, list) else value != 0
                if nonzero:
                    out.append({'query': path, 'other': cols[j], 'value': value})
        return out

    rigid = api._RIGID_BODY_VIEW[scene_index]
    live = rigid.get_transforms()
    previous = api._BODY_TRANSFORMS[scene_index]
    body_rows = [(i, path) for i, path in enumerate(rigid.prim_paths) if path in paths]
    out = {'available': True, 'scene_index': scene_index, 'physics_dt': physics_dt,
           'queried_paths': sorted(paths), 'sensor_paths': [p for _, p in selected],
           'pending_steps': int(api._PENDING_STEPS),
           'cached_current_pairs': sparse(api._CURRENT_CONTACT_MATRIX[scene_index]),
           'cached_interval_pairs': sparse(api._CONTACT_MATRIX[scene_index]),
           'live_impulses': sparse(view.get_contact_force_matrix(dt=physics_dt)),
           'cached_change_detection_transforms': {
               p: previous[i].detach().cpu().tolist() for i, p in body_rows},
           'live_body_transforms': {p: live[i].detach().cpu().tolist() for i, p in body_rows},
           'cache_refresh_calls': 0, 'native_buffer_writes': 0}
    return out
