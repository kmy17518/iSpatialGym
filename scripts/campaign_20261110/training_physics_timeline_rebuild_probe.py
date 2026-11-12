"""Explicit timeline SDK reconstruction within the existing stopped lifecycle.

The USD stage, loaded assets and Environment remain loaded. This is a declared
physical diagnostic, not a view refresh and not production authorization.
"""
import json
import time
from pathlib import Path

from .contracts import file_digest, require, write_json_atomic

SCHEMA = 'isg-physics-timeline-rebuild-runtime-compatibility-v1'
PARENT_COMMIT = 'afb8050c68fc25749c798294a95e0c90fec9caef'
DELTA = {'name': 'timeline_physx_scene_reconstruction_v1',
         'boundary': 'existing_stopped_callback_after_native_authoring_before_existing_play',
         'operations': ['IPhysx.release_physics_objects', 'IPhysx.force_load_physics_from_usd'],
         'world_and_assets_reloaded': False, 'physics_sdk_objects_recreated': True,
         'additional_explicit_physics_steps': 0, 'additional_stop_play_pairs': 0,
         'cold_and_warm_changed': True, 'source_inputs_changed': False,
         'thresholds_changed': False}
ALLOWED = {'scripts/campaign_20261110/training_physics_timeline_rebuild_probe.py',
           'scripts/campaign_20261110/tests/test_training_physics_timeline_rebuild_probe.py',
           'scripts/campaign_20261110/training_batch_probe.py'}


def verify_runtime(root, request, records, runtime, verify_parent):
    require(request.get('explicit_physics_timeline_rebuild_v1') is True
            and request.get('fixed_mass_baseline') is True
            and request.get('production_released') is False and request.get('benchmark_credit') == 0,
            'timeline reconstruction requires the bounded fixed-mass diagnostic')
    reference = request['runtime_compatibility']; path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'], 'timeline reconstruction proof changed')
    proof = json.loads(path.read_text()); parent_ref = proof['parent_runtime']
    parent = Path(parent_ref['path']).resolve().parent
    require(proof.get('schema') == SCHEMA and proof.get('physical_delta') == DELTA
            and proof.get('permitted_execution_kinds') == ['physics_action_open_loop']
            and proof.get('physical_code_changes') == 1,
            'timeline reconstruction delta differs')
    require(parent.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and parent_ref == {'path': str(parent/'.snapshot_commit'), 'sha256': file_digest(parent/'.snapshot_commit')}
            and (parent/'.snapshot_commit').read_text().strip() == PARENT_COMMIT
            and proof['candidate_runtime'] == {'path': str(runtime/'.snapshot_commit'),
                                               'sha256': file_digest(runtime/'.snapshot_commit')},
            'timeline reconstruction runtime identity differs')
    for name in ('installed_api_bindings', 'installed_api_definition', 'installed_simulation_route_test'):
        ref = proof[name]
        require(Path(ref['path']).is_absolute() and file_digest(ref['path']) == ref['sha256'],
                'inspected installed PhysX API changed: '+name)
    inherited = {**request, 'runtime_compatibility': proof['parent_compatibility'],
                 'probe_runtime_sha256': parent_ref['sha256']}
    inherited.pop('explicit_physics_timeline_rebuild_v1')
    verify_parent(root, inherited, records, parent)
    paths = {str(p.relative_to(tree)) for tree in (parent, runtime) for folder in ('ispatialgym', 'scripts')
             for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {n for n in paths if not (parent/n).is_file() or not (runtime/n).is_file()
               or file_digest(parent/n) != file_digest(runtime/n)}
    require(changed == set(proof['changed_payloads']) and changed <= ALLOWED
            and all((runtime/n).resolve().is_relative_to(runtime) and file_digest(runtime/n) == h
                    for n, h in proof['changed_payloads'].items()), 'unapproved timeline reconstruction payload')
    return reference


def reconstruct_stopped(sim, api, simulation_api, stage_id):
    """Recreate SDK objects through the interface used by native timeline steps."""
    require(sim.is_stopped() and not sim.currently_stepping, 'timeline reconstruction requires stopped physics')
    require(simulation_api.get_attached_stage() == 0,
            'manually attached simulation would disable ordinary timeline stepping')
    stage = sim.stage; scenes = tuple(id(s) for s in sim.scenes)
    start = time.monotonic(); running_before = api.is_running()
    api.release_physics_objects()
    api.force_load_physics_from_usd()
    require(simulation_api.get_attached_stage() == 0 and sim.stage is stage
            and tuple(id(s) for s in sim.scenes) == scenes and sim.is_stopped(),
            'timeline reconstruction changed loaded world or simulation route')
    return {'sdk_release_calls': 1, 'sdk_force_load_calls': 1, 'explicit_sdk_reconstructions': 1,
            'manual_attached_stage_before': 0, 'manual_attached_stage_after': 0,
            'native_running_before': running_before, 'native_running_after': api.is_running(),
            'retained_usd_stage_id': stage_id, 'retained_scene_identities': list(scenes),
            'world_loads_added': 0, 'explicit_physics_steps_requested': 0,
            'elapsed_seconds': time.monotonic()-start,
            'scope': 'Explicit release/load API calls only; implicit native lifecycle rebuilds unknown. '
                     'Original timeline stepping remains unchanged.'}


def install(runner, request, output):
    """Install before the fixed-mass helper captures its native restart call."""
    require(request.get('explicit_physics_timeline_rebuild_v1') is True,
            'timeline reconstruction was not requested')
    from ispatialgym.oracle import campaign_restore as CR
    proof = json.loads(Path(request['runtime_compatibility']['path']).read_text())
    restart = CR.restart_reused_world; events = []; output = Path(output)
    def rebuilt(meta, sim, *, execution=None, while_stopped=None):
        # Kit loads the extension while constructing Environment; the installer
        # intentionally runs before that construction and must not import it.
        from omni.physx import get_physx_interface, get_physx_simulation_interface
        from omni.physx.bindings import _physx
        import omnigibson.lazy as lazy
        require(Path(_physx.__file__).resolve() == Path(proof['installed_api_bindings']['path']).resolve()
                and file_digest(_physx.__file__) == proof['installed_api_bindings']['sha256'],
                'loaded timeline API differs from reviewed installed bindings')
        stage_id = lazy.pxr.UsdUtils.StageCache.Get().Insert(sim.stage).ToLongInt()
        require(stage_id > 0, 'retained USD stage is not registered in its native cache')
        event = {'ordinal': len(events), 'stage_id': stage_id, 'status': 'started'}
        events.append(event)
        def persist():
            write_json_atomic(output/'physics_timeline_rebuild_audit.json', {
                'schema': 'isg-explicit-physics-timeline-rebuild-audit-v1', 'physical_delta': DELTA,
                'runtime_compatibility': request['runtime_compatibility'], 'events': events,
                'explicit_sdk_reconstructions': sum(e.get('sdk', {}).get('explicit_sdk_reconstructions', 0) for e in events),
                'benchmark_credit': 0, 'production_released': False})
        def stopped():
            if while_stopped is not None: while_stopped()
            event['sdk'] = reconstruct_stopped(sim, get_physx_interface(), get_physx_simulation_interface(), stage_id)
            persist()
        persist()
        try:
            lifecycle = restart(meta, sim, execution=execution, while_stopped=stopped)
            require(lifecycle is not None and event.get('sdk'), 'native stopped lifecycle did not execute reconstruction')
            event.update(status='completed', lifecycle=lifecycle)
            return {**lifecycle, 'explicit_physics_sdk_reconstruction': event['sdk']}
        except BaseException as exc:
            event.update(status='failed', error=repr(exc))
            raise
        finally:
            persist()
    CR.restart_reused_world = rebuilt
