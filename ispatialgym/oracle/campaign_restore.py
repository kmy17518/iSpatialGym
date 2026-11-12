"""Explicit canary-only restoration of serialized task root motion."""
from __future__ import annotations
import math
from scripts.campaign_20261110.contracts import require, digest


def _purpose_allowed(meta, execution):
    purpose = meta.get('runtime', {}).get('purpose')
    if purpose == 'training_pipeline_canary':
        return True
    return purpose == 'training_production' and isinstance(execution, dict) and \
        execution.get('schema') == 'isg-training-restoration-execution-v1' and \
        execution.get('authorization_checked') is True and \
        execution.get('category') in ('validation_canary', 'released_training') and \
        execution.get('metadata_sha256') == digest(meta)


def pristine_mass_enabled(meta, execution=None):
    options = meta.get('scene_setup', {}).get('restoration_options', {})
    enabled = options.get('pristine_mass_before_reset_v1')
    if enabled is None or enabled is False:
        return False
    require(enabled is True and not options.get('serialized_mass_before_reset_v1')
            and meta.get('campaign_id') == 'campaign_20261110' and meta.get('split') == 'train'
            and meta.get('class_id') == 'disembedding_07' and meta.get('task') == 'disembedding'
            and _purpose_allowed(meta, execution),
            'pristine mass restoration requires its exact bounded canary scope')
    return True


def _mass_state(link):
    view = getattr(link, '_rigid_prim_view', None)
    require(view is not None and view.is_physics_handle_valid(), 'native mass actor unavailable')
    attr = link.prim.GetAttribute('physics:mass')
    authored = bool(attr and attr.HasAuthoredValueOpinion())
    return {'mass_kg': float(link.mass), 'inertia': view.get_inertias().tolist(),
            'usd_mass_authored': authored, 'usd_mass': attr.Get() if authored else None}


def mass_constraint_objects(meta, scope, records, *, execution=None):
    if not pristine_mass_enabled(meta, execution):
        return []
    names = sorted({r['object'] for r in records if r.get('type') == 'mass'})
    require(names and all(scope.get(name) is not None and hasattr(scope[name], 'links') for name in names),
            'pristine mass constraint actor missing at world construction')
    return [scope[name] for name in names]


def capture_pristine_mass(meta, objects, *, read_state=None, execution=None):
    """Capture once in build_env, before any episode constraints can mutate actors."""
    if not pristine_mass_enabled(meta, execution):
        return None
    read_state = read_state or _mass_state
    rows = {}
    for obj in objects:
        if obj is None or getattr(obj, 'kinematic_only', False):
            continue
        for link in obj.links.values():
            state = read_state(link)
            require(math.isfinite(state['mass_kg']) and state['mass_kg'] > 0, 'invalid pristine native mass')
            key = str(link.prim_path)
            require(key not in rows, 'duplicate pristine actor')
            rows[key] = {'object': obj.name, **state}
    require(rows, 'pristine native actors missing at world construction')
    return {'revision': 'pristine_mass_before_reset_v1', 'captured_at': 'build_env_before_first_restore',
            'actors': rows}


def _write_mass_state(link, state):
    import torch
    import omnigibson as og
    from pxr import UsdPhysics
    view = link._rigid_prim_view
    with og.sim.editing_usd():
        attr = link.prim.GetAttribute('physics:mass')
        if state['usd_mass_authored']:
            if not attr:
                attr = UsdPhysics.MassAPI.Apply(link.prim).CreateMassAttr()
            attr.Set(state['usd_mass'])
        elif attr:
            attr.Clear()
    # Restore captured inertia directly. Scaling it back by a mass ratio would
    # accumulate rounding and would not preserve the original actor state.
    view.set_masses(torch.as_tensor([state['mass_kg']], dtype=view.get_masses().dtype,
                                    device=view.get_masses().device))
    existing = view.get_inertias()
    view.set_inertias(torch.as_tensor(state['inertia'], dtype=existing.dtype, device=existing.device))


def restore_pristine_mass(meta, objects, baseline, *, read_state=None, write_state=None, execution=None):
    """Undo previous episode mass overrides; retain the original cold-load order."""
    if not pristine_mass_enabled(meta, execution):
        return None
    require(isinstance(baseline, dict) and baseline.get('captured_at') == 'build_env_before_first_restore'
            and baseline.get('revision') == 'pristine_mass_before_reset_v1',
            'pristine mass baseline missing; cannot recapture a reused world')
    read_state = read_state or _mass_state; write_state = write_state or _write_mass_state
    pending = []; actual_keys = set()
    for obj in objects:
        if obj is None or getattr(obj, 'kinematic_only', False):
            continue
        for link in obj.links.values():
            key = str(link.prim_path); actual_keys.add(key); saved = baseline['actors'].get(key)
            require(saved is not None and saved['object'] == obj.name, 'pristine actor identity differs')
            state = {k: v for k, v in saved.items() if k != 'object'}
            pending.append((link, state, read_state(link)))
    require(actual_keys == set(baseline['actors']), 'pristine actor inventory differs')
    changed = []
    for link, state, before in pending:
        # Cold A is a literal no-op, including its USD mass authorship.
        if before != state:
            write_state(link, state); changed.append(str(link.prim_path))
        require(read_state(link) == state, 'pristine mass/inertia readback differs')
    return {'revision': 'pristine_mass_before_reset_v1', 'captured_at': baseline['captured_at'],
            'actors_checked': len(pending), 'actors_restored': changed,
            'physics_steps_added': 0, 'later_serialized_mass_application_unchanged': True}


def vector(value, label):
    require(isinstance(value,(list,tuple)) and len(value)==3 and
            all(type(v) in (float,int) and math.isfinite(v) for v in value),
            'serialized root motion missing or invalid: '+label)
    return [float(v) for v in value]


def restore_serialized_root_motion(meta, physical, object_states, *, execution=None):
    option=meta.get('scene_setup',{}).get('restoration_options',{}).get('serialized_root_motion_v1')
    if option is None or option is False:
        return None
    require(option is True and meta.get('campaign_id')=='campaign_20261110' and meta.get('split')=='train'
            and meta.get('class_id')=='disembedding_07' and meta.get('task')=='disembedding'
            and _purpose_allowed(meta, execution),
            'serialized root motion is restricted to its exact bounded canary scope')
    import torch
    # Validate every serialized field before mutating any physical object.
    pending=[];skipped=[]
    for instance,state in object_states.items():
        obj=physical.get(instance);require(obj is not None,'serialized task object absent: '+instance)
        if bool(getattr(obj,'kinematic_only',False)):
            skipped.append(instance);continue
        root=state.get('root_link') or {}
        linear=vector(root.get('lin_vel'),instance+'.lin_vel')
        angular=vector(root.get('ang_vel'),instance+'.ang_vel')
        pending.append((instance,obj,linear,angular))
    require(pending,'no serialized dynamic root motion to restore')
    observations={}
    for instance,obj,linear,angular in pending:
        before={'linear_velocity':obj.get_linear_velocity().tolist(),
                'angular_velocity':obj.get_angular_velocity().tolist()}
        obj.set_linear_velocity(torch.tensor(linear,dtype=torch.float32))
        obj.set_angular_velocity(torch.tensor(angular,dtype=torch.float32))
        after={'linear_velocity':obj.get_linear_velocity().tolist(),
               'angular_velocity':obj.get_angular_velocity().tolist()}
        for key,expected in [('linear_velocity',linear),('angular_velocity',angular)]:
            require(len(after[key])==3 and all(abs(a-b)<=1e-7*max(1.,abs(b)) for a,b in zip(after[key],expected)),
                    'serialized root motion readback differs: '+instance+'.'+key)
        observations[instance]={'serialized':{'linear_velocity':linear,'angular_velocity':angular},
                                'before':before,'after':after}
    return {'revision':'serialized_root_motion_v1','boundary':'complete_native_loader_before_actions',
            'objects':observations,'kinematic_objects_unchanged':skipped,'physics_steps_added':0,
            'pose_or_goal_changes':False,'source':'exact immutable TRO root_link lin_vel/ang_vel'}


def restore_serialized_mass_before_reset(meta, physical, bindings, records, *, apply_mass_fn=None, execution=None):
    """Apply only recorded mass constraints before the loader's existing first step.

    On first load the scene has dataset masses; on reuse the previous instance's
    mass constraints persist. The opt-in canary uses its recorded masses at both
    boundaries. It adds no step and changes no serialized pose or velocity.
    """
    option = meta.get('scene_setup', {}).get('restoration_options', {}).get('serialized_mass_before_reset_v1')
    if option is None or option is False:
        return None
    require(option is True and meta.get('campaign_id') == 'campaign_20261110' and meta.get('split') == 'train'
            and meta.get('class_id') == 'disembedding_07' and meta.get('task') == 'disembedding'
            and _purpose_allowed(meta, execution),
            'pre-reset serialized mass is restricted to its exact bounded canary scope')
    pending = []
    seen = set()
    for record in records:
        if record.get('type') != 'mass':
            continue
        instance = record['object']; obj = physical.get(instance)
        binding = bindings.get(instance)
        expected = binding.get('scene_object_name') if isinstance(binding, dict) else binding
        require(obj is not None and expected and obj.name == expected,
                'serialized mass binding differs before reset: '+instance)
        if isinstance(binding, dict):
            for field in ('category', 'model'):
                require(binding.get(field) is None or getattr(obj, field, None) == binding[field],
                        'serialized mass model/category differs before reset: '+instance)
        mass = record.get('mass_kg')
        require(type(mass) in (float, int) and math.isfinite(mass) and mass > 0,
                'invalid serialized mass before reset')
        link = record.get('link'); identity = (instance, link)
        require(identity not in seen, 'duplicate serialized mass constraint')
        seen.add(identity)
        require(not getattr(obj, 'kinematic_only', False) and
                (link is None or link in obj.links), 'serialized mass link unavailable')
        pending.append((instance, obj, float(mass), link))
    require(pending, 'pre-reset mass canary has no serialized mass constraints')
    if apply_mass_fn is None:
        from ispatialgym.sampling.constraints import apply_mass
        apply_mass_fn = apply_mass
    result = []
    # Every binding and value is checked before any mutation.
    for instance, obj, mass, link in pending:
        applied = apply_mass_fn(obj, mass, link)
        require(abs(float(applied['after_kg']) - mass) <= 1e-6 * max(1., mass),
                'serialized mass readback differs before reset: '+instance)
        result.append({'object': instance, 'serialized_mass_kg': mass, **applied})
    return {'revision': 'serialized_mass_before_reset_v1', 'objects': result,
            'boundary': 'before_task_load_tro_instance_and_existing_reset_step',
            'physics_steps_added': 0, 'pose_or_goal_changes': False,
            'source': 'exact immutable TRO extras.constraints mass records',
            'later_constraint_restore_unchanged': True}
