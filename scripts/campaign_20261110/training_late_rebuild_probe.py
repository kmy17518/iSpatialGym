"""Bounded SDK reconstruction after incoming state restoration, before reset physics."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path

from .contracts import require, file_digest, write_json_atomic

SCHEMA = 'isg-late-incoming-physics-rebuild-runtime-v1'
PARENT_COMMIT = 'afb8050c68fc25749c798294a95e0c90fec9caef'
DELTA = {'name': 'late_incoming_physics_rebuild_v1',
         'boundary': 'incoming_state_loaded_immediately_before_existing_first_reset_physics_step',
         'source': 'same_world_current_full_native_state_no_reference_copy',
         'operations': ['capture_full_state_and_static_native_body_properties', 'stop',
                        'native_stopped_object_root_parent_setters', 'IPhysx.release_physics_objects',
                        'IPhysx.force_load_physics_from_usd', 'play', 'restore_own_captured_full_state'],
         'additional_stop_play_pairs_per_episode': 1, 'explicit_sdk_reconstructions_per_episode': 1,
         'additional_explicit_physics_steps': 0, 'world_and_assets_reloaded': False,
         'static_body_properties_required_unchanged': True, 'source_inputs_changed': False,
         'thresholds_changed': False, 'cold_and_reused_changed': True}
ALLOWED = {'scripts/campaign_20261110/training_late_rebuild_probe.py',
           'scripts/campaign_20261110/tests/test_training_late_rebuild_probe.py',
           'scripts/campaign_20261110/training_batch_probe.py'}


def verify_runtime(root, request, records, runtime, verify_parent):
    require(request.get('late_incoming_physics_rebuild_v1') is True and request.get('fixed_mass_baseline') is True
            and request.get('production_released') is False and request.get('benchmark_credit') == 0,
            'late reconstruction requires bounded fixed-mass scope')
    require(not any(request.get(key) for key in (
        'parking_before_play_v1', 'explicit_physics_timeline_rebuild_v1',
        'explicit_physics_sdk_rebuild_v1', 'reset_contact_cache_before_step_v1')),
        'late reconstruction cannot combine independent physical changes')
    reference = request['runtime_compatibility']; path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'], 'late reconstruction proof changed')
    proof = json.loads(path.read_text()); parent_ref = proof['parent_runtime']; parent = Path(parent_ref['path']).resolve().parent
    require(proof.get('schema') == SCHEMA and proof.get('physical_delta') == DELTA
            and proof.get('permitted_execution_kinds') == ['physics_action_open_loop']
            and parent.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and parent_ref == {'path': str(parent/'.snapshot_commit'), 'sha256': file_digest(parent/'.snapshot_commit')}
            and (parent/'.snapshot_commit').read_text().strip() == PARENT_COMMIT
            and proof['candidate_runtime'] == {'path': str(runtime/'.snapshot_commit'), 'sha256': file_digest(runtime/'.snapshot_commit')},
            'late reconstruction runtime or physical scope differs')
    for key in ('installed_api_bindings', 'installed_api_definition', 'installed_simulation_route_test'):
        item = proof[key]; require(Path(item['path']).is_absolute() and file_digest(item['path']) == item['sha256'],
                                   'inspected SDK implementation changed')
    inherited = {**request, 'runtime_compatibility': proof['parent_compatibility'],
                 'probe_runtime_sha256': parent_ref['sha256']}; inherited.pop('late_incoming_physics_rebuild_v1')
    verify_parent(root, inherited, records, parent)
    paths = {str(p.relative_to(tree)) for tree in (parent, runtime) for folder in ('ispatialgym', 'scripts')
             for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {p for p in paths if not (parent/p).is_file() or not (runtime/p).is_file()
               or file_digest(parent/p) != file_digest(runtime/p)}
    require(changed == set(proof['changed_payloads']) and changed <= ALLOWED
            and all((runtime/p).resolve().is_relative_to(runtime) and file_digest(runtime/p) == h
                    for p, h in proof['changed_payloads'].items()), 'unapproved late reconstruction payload')
    return reference


def plain(value):
    if hasattr(value, 'tolist'): return value.tolist()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [plain(v) for v in value]
    return value


def static_bodies(scene):
    result = {}
    for obj in scene.objects:
        for name, link in obj.links.items():
            view = getattr(link, '_rigid_prim_view', None)
            if view is None or not view.is_physics_handle_valid(): continue
            result[obj.name+'/'+name] = {method: plain(getattr(view, method)())
                                        for method in ('get_masses', 'get_inertias', 'get_coms')}
    require(result, 'late reconstruction lacks native static body observations')
    return result


def own_full_state(scene):
    registry = scene.object_registry; prior = registry._dump_filter
    registry.set_dump_filter(lambda obj: True)
    try: return copy.deepcopy(scene.dump_state(serialized=False))
    finally: registry.set_dump_filter(prior)


@contextmanager
def forbid_physics_steps(sim, event):
    saved = []
    event['whole_boundary_physics_step_calls'] = []
    try:
        for target, name in ((sim, 'step_physics'), (sim._sim_context, 'step')):
            had = name in vars(target); prior = vars(target).get(name)
            def forbidden(*args, _name=name, **kwargs):
                event['whole_boundary_physics_step_calls'].append(_name)
                require(False, 'late reconstruction attempted an additional physics step')
            saved.append((target, name, had, prior)); setattr(target, name, forbidden)
        yield
    finally:
        for target, name, had, prior in reversed(saved):
            if had: setattr(target, name, prior)
            else: delattr(target, name)


def reconstruct(loader, sim, restart, sdk, simulation_sdk, event=None):
    event = {} if event is None else event
    with forbid_physics_steps(sim, event):
        return _reconstruct(loader, sim, restart, sdk, simulation_sdk, event)


def _reconstruct(loader, sim, restart, sdk, simulation_sdk, event):
    scene = loader.env.scene
    # The SDK operation reconstructs the complete loaded physics stage. A
    # second live scene would require its own captured state and invariants.
    require(len(sim.scenes) == 1 and sim.scenes[0] is scene,
            'late reconstruction requires exactly the owned single loaded scene')
    event['loaded_scene_count'] = 1
    state = own_full_state(scene); before = static_bodies(scene)
    objects = {obj.name: obj for obj in scene.objects}
    require(set(state['registry']['object_registry']) == set(objects), 'late full state omitted initialized actors')
    poses = {name: tuple(v.clone() for v in obj.get_position_orientation()) for name, obj in objects.items()}
    identities = {name: id(obj) for name, obj in objects.items()}; stage = sim.stage
    event.update(before_static_bodies=before, before_root_poses=plain(poses), sdk={},
                 captured_initialized_objects=sorted(objects), reference_state_copied=False)
    def stopped():
        require(sim.is_stopped() and simulation_sdk.get_attached_stage() == 0,
                'late reconstruction requires original stopped timeline route')
        for name, obj in objects.items():
            obj.set_position_orientation(position=poses[name][0], orientation=poses[name][1])
        sdk.release_physics_objects(); sdk.force_load_physics_from_usd()
        require(simulation_sdk.get_attached_stage() == 0 and sim.stage is stage,
                'late reconstruction changed loaded stage or native stepping route')
        event['sdk'] = {'explicit_sdk_reconstructions': 1, 'release_calls': 1, 'force_load_calls': 1,
                        'manual_attached_stage': 0, 'implicit_native_reconstruction_count': None}
    event['lifecycle'] = restart(loader.tro.meta.raw['campaign'], sim,
        execution=loader._restoration_execution, while_stopped=stopped)
    require(event['sdk'] and event['lifecycle']['observed_physics_step_calls'] == []
            and event['lifecycle']['physics_steps_requested'] == 0, 'late reconstruction stepped physics')
    scene.load_state(state, serialized=False)
    event['after_static_bodies'] = static_bodies(scene)
    require(event['after_static_bodies'] == before, 'late reconstruction changed initialized static body properties')
    require({obj.name: id(obj) for obj in scene.objects} == identities and sim.stage is stage
            and len(sim.scenes) == 1 and sim.scenes[0] is scene,
            'late reconstruction replaced loaded world objects')
    event['after_root_poses'] = {obj.name: plain(obj.get_position_orientation()) for obj in scene.objects}
    event['static_body_properties_bit_identical'] = True
    return event


def install(runner, request, output, *, restart=None):
    from ispatialgym.oracle import campaign_restore as CR
    require(request.get('late_incoming_physics_rebuild_v1') is True, 'late reconstruction not requested')
    proof = json.loads(Path(request['runtime_compatibility']['path']).read_text())
    restart = CR.restart_reused_world if restart is None else restart
    original = runner.InstanceLoader.restore; events = []; output = Path(output)
    def persist():
        write_json_atomic(output/'late_rebuild_audit.json', {'physical_delta': DELTA, 'events': events,
            'explicit_sdk_reconstructions': sum(e.get('sdk', {}).get('explicit_sdk_reconstructions', 0) for e in events),
            'benchmark_credit': 0, 'production_released': False})
    def restored(loader, *args, **kwargs):
        from omni.physx import get_physx_interface, get_physx_simulation_interface
        from omni.physx.bindings import _physx
        require(file_digest(_physx.__file__) == proof['installed_api_bindings']['sha256'], 'loaded SDK bindings changed')
        task = loader.task; load = task.load_tro_instance
        had = 'load_tro_instance' in vars(task); raw = vars(task).get('load_tro_instance'); observed = []
        def loaded(*load_args, **load_kwargs):
            scene = loader.env.scene; reset = scene.reset
            had_reset = 'reset' in vars(scene); raw_reset = vars(scene).get('reset')
            def resetting(*reset_args, **reset_kwargs):
                sim = runner.og.sim; step = sim.step_physics
                had_step = 'step_physics' in vars(sim); raw_step = vars(sim).get('step_physics')
                def stepping(*step_args, **step_kwargs):
                    require(not observed, 'late reconstruction must precede exactly one reset step')
                    observed.append(True)
                    event = {'instance_id': loader.iid, 'status': 'started'}; events.append(event)
                    try:
                        reconstruct(loader, sim, restart,
                            get_physx_interface(), get_physx_simulation_interface(), event)
                        event['status'] = 'completed'
                    except BaseException as error:
                        event.update(status='failed', error=repr(error)); raise
                    finally: persist()
                    return step(*step_args, **step_kwargs)
                sim.step_physics = stepping
                try: return reset(*reset_args, **reset_kwargs)
                finally:
                    if had_step: sim.step_physics = raw_step
                    else: del sim.step_physics
            scene.reset = resetting
            try: return load(*load_args, **load_kwargs)
            finally:
                if had_reset: scene.reset = raw_reset
                else: del scene.reset
        task.load_tro_instance = loaded
        try:
            result = original(loader, *args, **kwargs)
            require(len(observed) == 1, 'late native reset boundary was not observed')
            return result
        finally:
            if had: task.load_tro_instance = raw
            else: del task.load_tro_instance
            persist()
    runner.InstanceLoader.restore = restored
