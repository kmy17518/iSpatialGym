"""Author incoming fixed-actor parking at the existing stopped boundary.

Native EntityPrim moves its parent while stopped to avoid cancellation against
a distant parked parent. Ordinary scene and parking restoration still run.
"""
import copy
import json
import math
from pathlib import Path

from .contracts import require, file_digest, write_json_atomic

SCHEMA = 'isg-fixed-parking-rebuild-runtime-v1'
PARENT_COMMIT = 'afb8050c68fc25749c798294a95e0c90fec9caef'
DELTA = {'name': 'fixed_parking_before_existing_play_v1',
         'boundary': 'existing_warm_stopped_callback_after_robot_anchor_before_play',
         'operation': 'native_stopped_EntityPrim_set_position_orientation_from_exact_incoming_scene',
         'actors': 'declared_fixed_nonarticulated_parking_only',
         'ordinary_scene_and_parking_restore_retained': True,
         'cold_physical_operations_changed': False, 'source_inputs_changed': False,
         'extra_physics_steps': 0, 'extra_stop_play_pairs': 0,
         'world_or_asset_reload': False, 'thresholds_changed': False}
ALLOWED = {'scripts/campaign_20261110/training_parking_rebuild_probe.py',
           'scripts/campaign_20261110/tests/test_training_parking_rebuild_probe.py',
           'scripts/campaign_20261110/training_batch_probe.py'}


def verify_runtime(root, request, records, runtime, verify_parent):
    require(request.get('parking_before_play_v1') is True and request.get('fixed_mass_baseline') is True
            and request.get('production_released') is False and request.get('benchmark_credit') == 0,
            'parking rebuild requires bounded fixed-mass source scope')
    reference = request['runtime_compatibility']; path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'], 'parking proof changed')
    proof = json.loads(path.read_text()); parent_ref = proof['parent_runtime']; parent = Path(parent_ref['path']).resolve().parent
    require(proof.get('schema') == SCHEMA and proof.get('physical_delta') == DELTA
            and proof.get('permitted_execution_kinds') == ['physics_action_open_loop']
            and parent.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and parent_ref == {'path': str(parent/'.snapshot_commit'), 'sha256': file_digest(parent/'.snapshot_commit')}
            and (parent/'.snapshot_commit').read_text().strip() == PARENT_COMMIT
            and proof['candidate_runtime'] == {'path': str(runtime/'.snapshot_commit'), 'sha256': file_digest(runtime/'.snapshot_commit')},
            'parking reconstruction runtime or scope differs')
    native = runtime/'Behavior-1K/OmniGibson/omnigibson/prims/entity_prim.py'
    require(proof['native_entity_setter']['sha256'] == file_digest(native), 'native stopped entity setter differs')
    inherited = {**request, 'runtime_compatibility': proof['parent_compatibility'],
                 'probe_runtime_sha256': parent_ref['sha256']}; inherited.pop('parking_before_play_v1')
    verify_parent(root, inherited, records, parent)
    paths = {str(p.relative_to(tree)) for tree in (parent, runtime) for folder in ('ispatialgym', 'scripts')
             for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {p for p in paths if not (parent/p).is_file() or not (runtime/p).is_file()
               or file_digest(parent/p) != file_digest(runtime/p)}
    require(changed == set(proof['changed_payloads']) and changed <= ALLOWED
            and all((runtime/p).resolve().is_relative_to(runtime) and file_digest(runtime/p) == h
                    for p, h in proof['changed_payloads'].items()), 'unapproved parking payload')
    return reference


def author_fixed_parking(scene, sim, states):
    require(sim.is_stopped() and states, 'fixed parking authoring requires stopped physics and exact actors')
    pending = []
    for name, state in states.items():
        obj = scene.object_registry('name', name); root = state.get('root_link', {})
        require(obj is not None and obj.fixed_base and obj.kinematic_only
                and not obj.articulated and int(obj.n_joints) == 0,
                'parking authoring cannot modify a dynamic or articulated actor')
        require(len(root.get('pos', [])) == 3 and len(root.get('ori', [])) == 4
                and all(math.isfinite(float(v)) for key in ('pos', 'ori') for v in root[key])
                and abs(math.sqrt(sum(float(v)**2 for v in root['ori'])) - 1) <= 1e-6,
                'parking source pose invalid')
        pending.append((obj, copy.deepcopy(root)))
    observations = []
    for obj, root in pending:
        before = [v.tolist() for v in obj.get_position_orientation()]
        obj.set_position_orientation(position=root['pos'], orientation=root['ori'])
        observations.append({'name': obj.name, 'source_root': root, 'before': before,
                             'after_stopped_authoring': [v.tolist() for v in obj.get_position_orientation()]})
    return observations


def install(runner, request, output):
    from ispatialgym.oracle import campaign_restore as CR
    from ispatialgym.oracle.campaign_batch import SCHEMA as AUGMENTATION_SCHEMA
    require(request.get('parking_before_play_v1') is True, 'parking operation not requested')
    attach = runner.InstanceLoader.attach; restart = CR.restart_reused_world
    current = []; events = []; output = Path(output)
    def attached(loader, shared):
        require(not current, 'nested incoming parking context')
        contract = shared['sequential_augmentation']
        require(contract['schema'] == AUGMENTATION_SCHEMA, 'parking source contract differs')
        member = contract['members_by_tro'][str(Path(loader.tro_path).resolve())]
        require(file_digest(loader.tro_path) == member['tro']['sha256']
                and file_digest(member['scene']['path']) == member['scene']['sha256']
                and set(member['parking_states']) == set(contract['parking_objects'])
                and not set(member['parking_states']).intersection(contract['protected_objects']),
                'parking source identity or protected scope differs')
        offset = len(events); current.append((shared['env'].scene, member))
        try:
            result = attach(loader, shared)
            require(len(events) == offset+1 and events[-1]['status'] == 'completed',
                    'incoming parking stopped callback missing')
            events[-1]['after_ordinary_restoration'] = {
                name: [v.tolist() for v in shared['env'].scene.object_registry('name', name).get_position_orientation()]
                for name in member['parking_states']}
            return result
        finally:
            current.pop()
            write_json_atomic(output/'parking_rebuild_audit.json', {'physical_delta': DELTA, 'events': events,
                'benchmark_credit': 0, 'production_released': False})
    def restarted(meta, sim, *, execution=None, while_stopped=None):
        if not current:
            return restart(meta, sim, execution=execution, while_stopped=while_stopped)
        scene, member = current[0]
        event = {'tro': member['tro'], 'source_scene': member['scene'], 'status': 'started'}; events.append(event)
        def stopped():
            if while_stopped is not None: while_stopped()
            event['authored'] = author_fixed_parking(scene, sim, member['parking_states'])
        try:
            result = restart(meta, sim, execution=execution, while_stopped=stopped)
            require(result is not None and result['observed_physics_step_calls'] == []
                    and result['physics_steps_requested'] == 0 and event.get('authored'),
                    'parking authoring did not use existing no-step lifecycle')
            event.update(status='completed', lifecycle=result)
            return result
        except BaseException as exc:
            event.update(status='failed', error=repr(exc)); raise
    runner.InstanceLoader.attach = attached
    CR.restart_reused_world = restarted
