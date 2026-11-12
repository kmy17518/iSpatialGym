"""Native operations for an explicitly checked, diagnostic one-world pool.

This module does not authorize execution or install itself in production.
The probe must bind the immutable source contract and declare its physical
delta before calling these helpers.
"""
import copy
from pathlib import Path

from .contracts import require, file_digest
from .training_pool_world import cold_anchor_transform, isolated_member_scene, verify_pool_isolation


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
    require(meta is not None and active_name in meta.get('active_properties', {}),
            'pool export lacks measured active properties')
    meta['active_names'] = [active_name]
    meta['active_properties'] = {active_name: meta['active_properties'][active_name]}
    meta['inactive_omitted'] = len(pool_names) - 1
    return active_scene(document, pool_names, [active_name])


def anchor_for_member(baseline, incoming_spawn):
    return cold_anchor_transform(baseline['spawn_pose'],
        {'position': baseline['anchor_position'], 'orientation': baseline['anchor_orientation']}, incoming_spawn)
