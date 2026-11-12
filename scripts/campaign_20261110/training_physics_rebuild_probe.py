"""Explicit SDK scene reconstruction within the existing stopped lifecycle.

The USD stage, loaded assets and Environment remain loaded. This is a declared
physical diagnostic, not a view refresh and not production authorization.
"""
import json
import time
from pathlib import Path

from .contracts import file_digest, require, write_json_atomic

SCHEMA = 'isg-physics-sdk-rebuild-runtime-compatibility-v1'
PARENT_COMMIT = 'afb8050c68fc25749c798294a95e0c90fec9caef'
DELTA = {'name': 'explicit_physx_stage_reconstruction_v1',
         'boundary': 'existing_stopped_callback_after_native_authoring_before_existing_play',
         'operations': ['IPhysxSimulation.detach_stage', 'IPhysxSimulation.attach_stage'],
         'world_and_assets_reloaded': False, 'physics_sdk_objects_recreated': True,
         'additional_explicit_physics_steps': 0, 'additional_stop_play_pairs': 0,
         'cold_and_warm_changed': True, 'source_inputs_changed': False,
         'thresholds_changed': False}
ALLOWED = {'scripts/campaign_20261110/training_physics_rebuild_probe.py',
           'scripts/campaign_20261110/tests/test_training_physics_rebuild_probe.py',
           'scripts/campaign_20261110/training_batch_probe.py'}


def verify_runtime(root, request, records, runtime, verify_parent):
    require(request.get('explicit_physics_sdk_rebuild_v1') is True
            and request.get('fixed_mass_baseline') is True
            and request.get('production_released') is False and request.get('benchmark_credit') == 0,
            'SDK reconstruction requires the bounded fixed-mass diagnostic')
    reference = request['runtime_compatibility']; path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'], 'SDK reconstruction proof changed')
    proof = json.loads(path.read_text()); parent_ref = proof['parent_runtime']
    parent = Path(parent_ref['path']).resolve().parent
    require(proof.get('schema') == SCHEMA and proof.get('physical_delta') == DELTA
            and proof.get('permitted_execution_kinds') == ['physics_action_open_loop']
            and proof.get('physical_code_changes') == 1,
            'SDK reconstruction delta differs')
    require(parent.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and parent_ref == {'path': str(parent/'.snapshot_commit'), 'sha256': file_digest(parent/'.snapshot_commit')}
            and (parent/'.snapshot_commit').read_text().strip() == PARENT_COMMIT
            and proof['candidate_runtime'] == {'path': str(runtime/'.snapshot_commit'),
                                               'sha256': file_digest(runtime/'.snapshot_commit')},
            'SDK reconstruction runtime identity differs')
    for name in ('installed_api_bindings', 'installed_api_definition'):
        ref = proof[name]
        require(Path(ref['path']).is_absolute() and file_digest(ref['path']) == ref['sha256'],
                'inspected installed PhysX API changed: '+name)
    inherited = {**request, 'runtime_compatibility': proof['parent_compatibility'],
                 'probe_runtime_sha256': parent_ref['sha256']}
    inherited.pop('explicit_physics_sdk_rebuild_v1')
    verify_parent(root, inherited, records, parent)
    paths = {str(p.relative_to(tree)) for tree in (parent, runtime) for folder in ('ispatialgym', 'scripts')
             for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {n for n in paths if not (parent/n).is_file() or not (runtime/n).is_file()
               or file_digest(parent/n) != file_digest(runtime/n)}
    require(changed == set(proof['changed_payloads']) and changed <= ALLOWED
            and all((runtime/n).resolve().is_relative_to(runtime) and file_digest(runtime/n) == h
                    for n, h in proof['changed_payloads'].items()), 'unapproved SDK reconstruction payload')
    return reference


def reconstruct_stopped(sim, api, stage_id):
    """Recreate SDK objects, never advance physics or reload the USD stage."""
    require(sim.is_stopped() and not sim.currently_stepping, 'SDK reconstruction requires stopped physics')
    stage = sim.stage; scenes = tuple(id(s) for s in sim.scenes)
    before = api.get_attached_stage(); start = time.monotonic()
    api.detach_stage()
    detached = api.get_attached_stage()
    require(detached != stage_id, 'PhysX SDK stage did not detach')
    require(api.attach_stage(stage_id) is True, 'PhysX SDK rejected the retained USD stage')
    attached = api.get_attached_stage()
    require(attached == stage_id and sim.stage is stage and tuple(id(s) for s in sim.scenes) == scenes
            and sim.is_stopped(), 'SDK reconstruction replaced the loaded stage/world or started physics')
    return {'sdk_detach_calls': 1, 'sdk_attach_calls': 1, 'explicit_sdk_reconstructions': 1,
            'attached_stage_before': before, 'attached_stage_after_detach': detached,
            'attached_stage_after_attach': attached, 'retained_usd_stage_id': stage_id,
            'retained_scene_identities': list(scenes), 'world_loads_added': 0,
            'explicit_physics_steps_requested': 0, 'elapsed_seconds': time.monotonic()-start,
            'scope': 'Explicit SDK API calls counted; ordinary native stop/play remains separately recorded.'}


def install(runner, request, output):
    """Install before the fixed-mass helper captures its native restart call."""
    require(request.get('explicit_physics_sdk_rebuild_v1') is True,
            'SDK reconstruction was not requested')
    from ispatialgym.oracle import campaign_restore as CR
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings import _physx
    import omnigibson.lazy as lazy
    proof = json.loads(Path(request['runtime_compatibility']['path']).read_text())
    require(Path(_physx.__file__).resolve() == Path(proof['installed_api_bindings']['path']).resolve()
            and file_digest(_physx.__file__) == proof['installed_api_bindings']['sha256'],
            'loaded SDK API differs from reviewed installed bindings')
    restart = CR.restart_reused_world; events = []; output = Path(output)
    def rebuilt(meta, sim, *, execution=None, while_stopped=None):
        stage_id = lazy.pxr.UsdUtils.StageCache.Get().Insert(sim.stage).ToLongInt()
        require(stage_id > 0, 'retained USD stage is not registered in its native cache')
        event = {'ordinal': len(events), 'stage_id': stage_id, 'status': 'started'}
        events.append(event)
        def persist():
            write_json_atomic(output/'physics_sdk_rebuild_audit.json', {
                'schema': 'isg-explicit-physics-sdk-rebuild-audit-v1', 'physical_delta': DELTA,
                'runtime_compatibility': request['runtime_compatibility'], 'events': events,
                'explicit_sdk_reconstructions': sum(e.get('sdk', {}).get('explicit_sdk_reconstructions', 0) for e in events),
                'benchmark_credit': 0, 'production_released': False})
        def stopped():
            if while_stopped is not None: while_stopped()
            event['sdk'] = reconstruct_stopped(sim, get_physx_simulation_interface(), stage_id)
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
