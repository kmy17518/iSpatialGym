"""Preserve the current native anchor across the incoming scene restore.

The existing stopped source-anchor setter has already reconstructed the fixed
joint and its native root. Replacing that result with a precomputed SE3 pose
introduces measured float roundoff. This bounded experiment restores all other
state normally and uses this same world's post-play root pose for that restore.
"""
import copy
from contextlib import contextmanager
import json
from pathlib import Path

from .contracts import require, file_digest, write_json_atomic

SCHEMA = 'isg-pool-native-anchor-restore-runtime-v1'
PARENT_COMMIT = 'f4a4be11fb3d39bbecf26014bb446e68ea7cfb0a'
DELTA = {'name': 'pool_preserve_native_anchor_v1', 'source_inputs_changed': False,
         'boundary': 'existing_scene_load_after_existing_source_anchor_stop_play',
         'root_pose_source': 'same_world_native_root_immediately_before_scene_load',
         'reference_state_copied': False, 'extra_explicit_physics_steps': 0,
         'extra_stop_play_pairs': 0, 'pool_activation_operations_changed': False,
         'robot_joints_velocities_controllers_restored_unchanged': True,
         'thresholds_changed': False, 'task_actions': 0}
ALLOWED = {'scripts/campaign_20261110/training_pool_native_anchor_probe.py',
           'scripts/campaign_20261110/tests/test_training_pool_native_anchor_probe.py',
           'scripts/campaign_20261110/training_pool_probe.py'}


def verify_runtime(root, request, records, runtime, verify_parent):
    require(request.get('pool_preserve_native_anchor_v1') is True
            and request.get('pool_exact_source_anchor_v1') is True
            and request.get('load_only') is True and request.get('production_released') is False
            and request.get('benchmark_credit') == 0, 'native anchor requires bounded source-anchor load scope')
    reference = request['runtime_compatibility']; path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'], 'native anchor proof changed')
    proof = json.loads(path.read_text()); parent_ref = proof['parent_runtime']
    parent = Path(parent_ref['path']).resolve().parent
    require(proof.get('schema') == SCHEMA and proof.get('physical_delta') == DELTA
            and proof.get('permitted_execution_kinds') == ['native_load_only']
            and parent.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and parent_ref == {'path': str(parent/'.snapshot_commit'), 'sha256': file_digest(parent/'.snapshot_commit')}
            and (parent/'.snapshot_commit').read_text().strip() == PARENT_COMMIT
            and proof['candidate_runtime'] == {'path': str(runtime/'.snapshot_commit'),
                                               'sha256': file_digest(runtime/'.snapshot_commit')},
            'native anchor runtime or physical scope differs')
    inherited = {**request, 'runtime_compatibility': proof['parent_compatibility'],
                 'probe_runtime_sha256': parent_ref['sha256']}
    inherited.pop('pool_preserve_native_anchor_v1')
    verify_parent(root, inherited, records, parent)
    paths = {str(p.relative_to(tree)) for tree in (parent, runtime) for folder in ('ispatialgym', 'scripts')
             for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {p for p in paths if not (parent/p).is_file() or not (runtime/p).is_file()
               or file_digest(parent/p) != file_digest(runtime/p)}
    require(changed == set(proof['changed_payloads']) and changed <= ALLOWED
            and all((runtime/p).resolve().is_relative_to(runtime) and file_digest(runtime/p) == h
                    for p, h in proof['changed_payloads'].items()), 'unapproved native anchor payload')
    return reference


def anchored_scene_state(state, robot_name, native_pose, current_scene_pose):
    import torch
    require(isinstance(state, dict) and 'pos' in state and 'ori' in state,
            'native anchor requires explicit scene frame')
    require(all(torch.equal(torch.as_tensor(state[key], dtype=torch.float32), observed)
                for key, observed in zip(('pos', 'ori'), current_scene_pose)),
            'native anchor cannot bypass scene rebasing')
    result = copy.deepcopy(state)
    root = result['registry']['object_registry'][robot_name]['root_link']
    require(set(root) >= {'pos', 'ori', 'lin_vel', 'ang_vel'}, 'native anchor root state incomplete')
    position, orientation = native_pose
    require(tuple(position.shape) == (3,) and tuple(orientation.shape) == (4,)
            and bool(torch.isfinite(position).all()) and bool(torch.isfinite(orientation).all()),
            'native anchor pose invalid')
    root['pos'], root['ori'] = position.clone(), orientation.clone()
    return result


@contextmanager
def preserve_anchor_at_scene_load(scene, robot, sim):
    import torch
    original = scene.load_state; had = 'load_state' in vars(scene); old = vars(scene).get('load_state')
    observations = []
    def loading(state, serialized=False):
        require(not serialized and not observations and sim.is_playing(),
                'native anchor restore must run once after existing play')
        pose = tuple(v.clone() for v in robot.root_link.get_position_orientation())
        prepared = anchored_scene_state(state, robot.name, pose, scene.get_position_orientation())
        before = state['registry']['object_registry'][robot.name]['root_link']
        event = {'original_serialized_root': {k: before[k].tolist() for k in ('pos', 'ori')},
                 'native_root_before_load': [v.tolist() for v in pose], 'reference_state_copied': False}
        observations.append(event)
        result = original(prepared, serialized=False)
        after = robot.root_link.get_position_orientation()
        event.update(native_root_after_load=[v.tolist() for v in after],
                     native_root_bit_identical=all(torch.equal(a, b) for a, b in zip(pose, after)))
        require(event['native_root_bit_identical'], 'native scene restore changed preserved native anchor')
        return result
    scene.load_state = loading
    try:
        yield observations
        require(len(observations) == 1, 'native anchor scene restore not observed')
    finally:
        if had: scene.load_state = old
        else: del scene.load_state


def install(runner, contract, output, parent_install, request):
    from .training_pool_source_anchor_probe import install as install_source_anchor
    proof = json.loads(Path(request['runtime_compatibility']['path']).read_text())
    inherited = {**request, 'runtime_compatibility': proof['parent_compatibility']}
    inherited.pop('pool_preserve_native_anchor_v1')
    install_source_anchor(runner, contract, output, parent_install, inherited)
    native_attach = runner.InstanceLoader.attach; observations = []; output = Path(output)
    def attach(loader, shared):
        with preserve_anchor_at_scene_load(shared['env'].scene, shared['robot'], runner.og.sim) as events:
            try:
                return native_attach(loader, shared)
            finally:
                observations.append({'tro': loader.tro_path, 'events': events})
                write_json_atomic(output/'native_anchor_restore_audit.json', {
                    'physical_delta': DELTA, 'observations': observations,
                    'benchmark_credit': 0, 'production_released': False})
    runner.InstanceLoader.attach = attach
