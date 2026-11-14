"""Native operations for an explicitly checked, diagnostic one-world pool.

This module does not authorize execution or install itself in production.
The probe must bind the immutable source contract and declare its physical
delta before calling these helpers.
"""
import copy
from pathlib import Path

from .contracts import require, file_digest
from .training_pool_world import cold_anchor_transform, isolated_member_scene, verify_pool_isolation, active_actor_names


def source_member(contract, tro_path):
    path = str(Path(tro_path).resolve())
    matches = [row for row in contract['members'] if str(Path(row['tro']['path']).resolve()) == path]
    require(matches and all(row == matches[0] for row in matches), 'pool source member is absent or ambiguous')
    member = matches[0]
    require(file_digest(path) == member['tro']['sha256']
            and file_digest(member['scene']['path']) == member['scene']['sha256'],
            'pool member source changed')
    return member


def merge_member_state(baseline, member_document, robot_name, anchor, scene_to_local):
    """Reset every source actor while retaining the initialized native robot.

    The robot's fixed-joint anchor is separately set while stopped. Its native
    serialized articulation state is retained here until ordinary TRO loading
    restores its exact requested joints, velocities and controller targets.
    """
    result = copy.deepcopy(baseline)
    target = result['registry']['object_registry']
    source = member_document['state']['registry']['object_registry']
    require(set(source) <= set(target) and robot_name in target,
            'pool member does not fit the initialized world inventory')
    for name, state in source.items():
        if name != robot_name:
            target[name] = copy.deepcopy(state)
    position, orientation = scene_to_local(anchor['position'], anchor['orientation'])
    target[robot_name]['root_link']['pos'] = position
    target[robot_name]['root_link']['ori'] = orientation
    return result


def restart_without_steps(sim, while_stopped):
    """Measure a native stop/play transition and reject any explicit step."""
    require(sim.is_playing(), 'pool transition must begin in a playing world')
    settings = lambda: (sim.get_physics_dt(), sim.get_rendering_dt(), sim.get_sim_step_dt())
    before = settings(); calls = []; wrapped = []
    for owner, name in ((sim, 'step_physics'), (sim._sim_context, 'step')):
        native = getattr(owner, name); present = name in vars(owner); value = vars(owner).get(name)
        def observed(*args, _native=native, _name=name, **kwargs):
            calls.append(_name)
            return _native(*args, **kwargs)
        setattr(owner, name, observed); wrapped.append((owner, name, present, value))
    try:
        sim.stop(); require(sim.is_stopped(), 'pool native stop did not finish')
        while_stopped()
        sim.play(); require(sim.is_playing(), 'pool native play did not finish')
    finally:
        for owner, name, present, value in reversed(wrapped):
            if present: setattr(owner, name, value)
            else: delattr(owner, name)
    require(not calls and before == settings(), 'pool transition stepped physics or changed its timestep')
    return {'native_calls': ['stop', 'play'], 'observed_physics_step_calls': calls,
            'physics_steps_requested': 0, 'time_step_settings': list(before)}


def pool_restore_document(member_document, isolation):
    """Private restore input for all initialized actors, never a dataset export."""
    document = copy.deepcopy(member_document)
    meta = document.setdefault('metadata', {}).setdefault('ispatialgym', {}).setdefault('model_pool', {})
    meta['active_names'] = sorted(isolation)
    meta['active_properties'] = {name: copy.deepcopy(row['properties']) for name, row in isolation.items()}
    for name, row in isolation.items():
        document['state']['registry']['object_registry'][name] = copy.deepcopy(row['state'])
    return document


def native_isolation(scene, robot, isolation, task_bindings, planner_names):
    """Read each native actor's state; caller supplies actual planner inventory."""
    from ispatialgym.sampling.model_pool import native_properties
    from omnigibson.object_states import AttachedTo
    observed = {}
    for name in isolation:
        obj = scene.object_registry('name', name)
        require(obj is not None, 'initialized pool actor disappeared: '+name)
        attachment = getattr(obj, 'states', {}).get(AttachedTo)
        observed[name] = {'properties': native_properties(obj),
                          'position': obj.get_position_orientation()[0].tolist(),
                          'linear_velocity': obj.get_linear_velocity().tolist(),
                          'angular_velocity': obj.get_angular_velocity().tolist(),
                          'assisted_held': any(held is obj for held in robot._ag_obj_in_hand.values()),
                          'attached_parent': getattr(getattr(attachment, 'parent', None), 'name', None)}
    checked = verify_pool_isolation(isolation, observed, task_bindings=task_bindings,
                                    planner_names=planner_names)
    return {**checked, 'native': observed}


def active_export(scene_document, pool_names, active_name):
    """Omit only inactive pool actors and reject surviving cross-references."""
    from ispatialgym.sampling.model_pool import active_scene
    document = copy.deepcopy(scene_document)
    meta = document.get('metadata', {}).get('ispatialgym', {}).get('model_pool')
    active = active_actor_names(active_name)
    require(meta is not None and set(active) <= set(meta.get('active_properties', {})),
            'pool export lacks measured active properties')
    meta['active_names'] = list(active)
    meta['active_properties'] = {name: meta['active_properties'][name] for name in active}
    meta['inactive_omitted'] = len(pool_names) - len(active)
    return active_scene(document, pool_names, active)


def anchor_for_member(baseline, incoming_spawn):
    return cold_anchor_transform(baseline['spawn_pose'],
        {'position': baseline['anchor_position'], 'orientation': baseline['anchor_orientation']}, incoming_spawn)


def author_isolation_stopped(scene, sim, isolation, *, gravity_api=None):
    """Stage exact pool flags and root poses before an existing native play.

    This helper is not installed by the current load probe. It supports the next
    declared physical candidate, subject to the pending reset measurements.
    Native property readback after play and ordinary TRO restoration remain
    required. No invalid stopped tensor handle is accessed here.
    """
    require(sim.is_stopped(), 'pool USD authoring requires stopped physics')
    if gravity_api is None:
        import omnigibson.lazy as lazy
        schema = lazy.pxr.PhysxSchema.PhysxRigidBodyAPI
        gravity_api = lambda prim: schema(prim) if prim.HasAPI(schema) else schema.Apply(prim)
    prepared = []
    # Check every identity and every property path before making any change.
    for name, row in isolation.items():
        obj = scene.object_registry('name', name); expected = row['properties']
        require(obj is not None and obj.category == expected['category'] and obj.model == expected['model']
                and obj.scale.tolist() == expected['scale'] and set(obj.links) == set(expected['links']),
                'stopped pool authoring cannot change initialized geometry: '+name)
        require(int(getattr(obj, 'n_joints', 0)) == 0,
                'articulated pool member needs a separate joint-state reconstruction contract')
        links = []
        for link_name, flags in expected['links'].items():
            link = obj.links[link_name]
            collisions = {str(api.GetPrim().GetPath().MakeRelativePath(obj.prim_path)): api
                          for api in getattr(link, '_collision_apis', [])}
            require(set(collisions) == set(flags['collisions']),
                    'stopped pool collision inventory differs: '+name+'/'+link_name)
            links.append((link, flags, collisions))
        root = row['state']['root_link']
        require(set(root) >= {'pos', 'ori', 'lin_vel', 'ang_vel'}, 'pool source root state incomplete')
        prepared.append((name, obj, row, links))
    records = []
    for name, obj, row, links in prepared:
        for link, flags, collisions in links:
            if flags['gravity_disabled'] is not None:
                gravity_api(link.prim).GetDisableGravityAttr().Set(flags['gravity_disabled'])
            for key, enabled in flags['collisions'].items():
                collisions[key].GetCollisionEnabledAttr().Set(enabled)
        root = row['state']['root_link']
        obj.set_position_orientation(position=root['pos'], orientation=root['ori'], frame='scene')
        obj.visible = row['properties']['visible']
        records.append({'name': name, 'active': row['active'], 'source_root': copy.deepcopy(root),
                        'authored_properties': copy.deepcopy(row['properties'])})
    return {'boundary': 'stopped_before_existing_play', 'actors': records,
            'native_velocity_writes': 0, 'physics_steps_requested': 0,
            'static_geometry_changes': 0,
            'scope': 'USD activation and root pose authoring only; native readback and exact source velocities restored after play.'}
