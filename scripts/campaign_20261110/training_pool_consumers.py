"""Explicit pool consumer helpers; not installed in any native runtime by default.

Physical isolation alone does not prove planner exclusion or recorded-state
exclusion. These helpers inspect the native mesh enable mask and restrict only
an explicitly scoped recording operation, leaving internal snapshots intact.
"""
from contextlib import contextmanager
from .contracts import require


def excluding_inactive(previous, initialized, active):
    initialized, active = set(initialized), set(active)
    require(active and active <= initialized, 'consumer active inventory differs')
    require(previous is None or callable(previous), 'consumer filter is invalid')
    inactive = initialized-active
    return lambda obj: obj.name not in inactive and (previous is None or bool(previous(obj)))


def _list(value):
    if hasattr(value, 'detach'): value = value.detach().cpu()
    return value.tolist() if hasattr(value, 'tolist') else value


def planner_observation(backend, objects, active):
    """Read native masks immediately after full sync and before grasp attachment.

    Cached inactive meshes may remain allocated but must be disabled. Every
    active actor collider must be present and enabled at this specific boundary.
    This is not a planner path or collision-clearance success claim.
    """
    require(objects and active in objects and all(k == v.name for k,v in objects.items()),
            'planner pool inventory differs')
    store = backend.checker.data.meshes
    names, enabled, counts = store.names, _list(store.enable), _list(store.count)
    require(len(names) == len(enabled) == len(counts) == 1 and len(names[0]) == len(enabled[0]),
            'one native planner environment and complete mask required')
    count = counts[0]
    require(type(count) is int and 0 <= count <= len(names[0])
            and all(x in (0,1) for x in enabled[0])
            and all(x == 0 for x in enabled[0][count:]), 'invalid native mesh allocation/mask')
    slots = {}
    for index, name in enumerate(names[0][:count]):
        require(isinstance(name,str) and name and name not in slots, 'ambiguous native mesh name')
        slots[name] = bool(enabled[0][index])
    observations = {}
    for name, obj in objects.items():
        paths = [str(mesh.prim_path) for link in obj.links.values() for mesh in link.collision_meshes.values()]
        require(paths and len(paths) == len(set(paths)), 'pool collider inventory absent or duplicated')
        rows = {path:{'allocated':path in slots,'enabled':slots.get(path,False)} for path in paths}
        require(all(r['allocated'] and r['enabled'] for r in rows.values()) if name == active
                else not any(r['enabled'] for r in rows.values()),
                'native pool planner participation differs: '+name)
        observations[name] = {'active':name == active,'colliders':rows}
    return {'boundary':'after_full_world_sync_before_grasp_attachment','objects':observations,
            'native_enable_mask_read':True,'allocated_mesh_count':count,
            'enabled_mesh_count':sum(enabled[0]),'trajectory_collision_clearance_verified':False}


@contextmanager
def recorded_registry_scope(scene, initialized, active, *, initial_state):
    """Restrict one explicit recording call, preserving the exact prior filter.

    Initial recorded state includes all non-pool actors even if asleep or
    unchanged. Later recorded states retain the original changed/awake filter.
    Internal loader/checkpoint snapshots must not enter this recording scope.
    """
    registry = scene.object_registry
    require(hasattr(registry,'_dump_filter') and callable(getattr(registry,'set_dump_filter',None)),
            'native recording registry filter unavailable')
    prior = registry._dump_filter
    callback = excluding_inactive(None if initial_state else prior, initialized, active)
    registry.set_dump_filter(callback)
    try:
        yield
    finally:
        registry.set_dump_filter(prior)
