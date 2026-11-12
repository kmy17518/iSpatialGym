"""Bounded native load-only experiment for fixed model/scale actor reuse.

This diagnostic installs no production defaults. It changes initialization and
the explicitly measured reuse boundary; source TROs, actions, goals and native
restore checks remain exact. No action, planner or export equivalence is claimed.
"""
import argparse
import copy
from contextlib import contextmanager
import json
import os
import sys
from pathlib import Path

from .contracts import require, file_digest, write_json_atomic
from .training_pool_world import prepare, checked, isolated_member_scene
from .training_pool_adapter import (source_member, merge_member_state,
    restart_without_steps, pool_restore_document, native_isolation, anchor_for_member)

SCHEMA = 'isg-selected-pool-load-runtime-compatibility-v1'
DELTA = {'name': 'fixed_actor_pool_load_v1', 'source_inputs_changed': False,
         'initialization': 'union_of_exact_fixed_model_scale_actors',
         'reuse': 'ordinary_cleanup_then_no_step_stop_anchor_play_then_incoming_scene_restore',
         'isolation': 'inactive_hidden_collision_off_gravity_off_unbound_parked',
         'task_action_steps': 0, 'extra_explicit_physics_steps': 0,
         'thresholds_changed': False, 'recording_enabled': False}
ALLOWED = {'scripts/campaign_20261110/training_batch_probe.py',
           'scripts/campaign_20261110/training_pool_world.py',
           'scripts/campaign_20261110/training_pool_adapter.py',
           'scripts/campaign_20261110/training_pool_probe.py',
           'scripts/campaign_20261110/selected_pool_episode_probe.py',
           'ispatialgym/oracle/campaign_batch.py'}


def verify_inputs(root, request, records):
    require(request.get('selected_pool_world') is True and request.get('load_only') is True
            and request.get('production_released') is False and request.get('benchmark_credit') == 0
            and not request.get('physical_candidate') and not request.get('sequential_augmentation')
            and request.get('diagnostic_control') in ('FRESH_A', 'FRESH_B', 'ABA'),
            'pool load probe requires its exact bounded diagnostic scope')
    source = checked(root, request['source_sequence'])
    contract = prepare(root, request['source_sequence']['path'])
    require(contract == checked(root, request['pool_contract'])
            and request['canaries'] == [e['review'] for e in source['episodes'][:2]]
            and [r['artifacts']['tro'] for r in records] == [m['tro'] for m in contract['members'][:2]],
            'pool source canary or physical contract differs')
    require(request.get('source_environment_delta') == {'ISPATIALGYM_FAST_COLLECTOR_PREBUILD': '1'},
            'undeclared pool execution environment delta')
    return source, contract


def verify_runtime(root, request, records, runtime):
    if request.get('pool_preserve_native_anchor_v1') is True:
        from .training_pool_native_anchor_probe import verify_runtime as verify_native_anchor
        return verify_native_anchor(root, request, records, runtime, verify_runtime)
    if request.get('pool_exact_source_anchor_v1') is True:
        from .training_pool_source_anchor_probe import verify_runtime as verify_anchor_runtime
        return verify_anchor_runtime(root, request, records, runtime, verify_runtime)
    source, contract = verify_inputs(root, request, records)
    proof = checked(root, request['runtime_compatibility'])
    base = Path(source['native_runtime']['path']).resolve().parent
    require(proof.get('schema') == SCHEMA and proof.get('physical_delta') == DELTA
            and proof.get('permitted_execution_kinds') == ['native_load_only']
            and proof.get('native_runtime') == source['native_runtime']
            and all(r['validation_key']['code_sha256'] == source['native_runtime']['sha256'] for r in records)
            and base.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and proof.get('candidate_runtime') == {'path': str(runtime/'.snapshot_commit'),
                                                  'sha256': file_digest(runtime/'.snapshot_commit')},
            'pool diagnostic runtime or declared physical operation differs')
    paths = {str(p.relative_to(tree)) for tree in (base, runtime) for folder in ('ispatialgym', 'scripts')
             for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {p for p in paths if not (base/p).is_file() or not (runtime/p).is_file()
               or file_digest(base/p) != file_digest(runtime/p)}
    require(changed == set(proof['changed_payloads']) and changed <= ALLOWED
            and all((runtime/p).resolve().is_relative_to(runtime) and file_digest(runtime/p) == h
                    for p, h in proof['changed_payloads'].items()),
            'pool runtime changed unreviewed native payloads')
    return request['runtime_compatibility']


def install(runner, contract, output):
    """Install only the load experiment's declared construction/reuse delta."""
    from omnigibson.utils.python_utils import recursively_convert_to_torch
    from ispatialgym.sampling.model_pool import restore_export
    from ispatialgym.oracle.campaign_batch import capture_robot_anchor
    import torch
    native_build_config = runner.build_env_config
    native_build = runner.InstanceLoader.build_env
    native_attach = runner.InstanceLoader.attach
    native_restore = runner.InstanceLoader.restore
    observations = []

    def member(loader):
        row = source_member(contract, loader.tro_path)
        document, isolation = isolated_member_scene(contract,
            json.loads(Path(row['scene']['path']).read_text()), row['active'])
        return row, document, isolation

    def save(loader, phase, extra=None):
        row, document, isolation = member(loader)
        bindings = {k: v.name for k, v in loader.task.object_scope.items() if hasattr(v, 'name')}
        measured = native_isolation(loader.env.scene, loader.robot, isolation, bindings, None)
        entry = {'phase': phase, 'tro': row['tro'], 'active': row['active'], 'isolation': measured,
                 'environment_identity': id(loader.env), 'scene_identity': id(loader.env.scene),
                 'task_actions': 0, 'planner': {'constructed': False, 'exclusion_verified': False},
                 **(extra or {})}
        observations.append(entry)
        write_json_atomic(output/'pool_native_audit.json', {'physical_delta': DELTA, 'observations': observations})
        loader.report.setdefault('pool_native_load', []).append(entry)

    def build_config(*args, **kwargs):
        cfg = native_build_config(*args, **kwargs)
        path = kwargs.get('tro_path') or args[3]
        row = source_member(contract, path)
        document, _ = isolated_member_scene(contract, json.loads(Path(row['scene']['path']).read_text()), row['active'])
        destination = output/('private_union_'+row['active']+'.json')
        require(not destination.exists(), 'union initialization artifact already exists')
        write_json_atomic(destination, document)
        cfg['scene']['scene_file'] = str(destination)
        return cfg

    def build(loader):
        result = native_build(loader)
        row, document, isolation = member(loader)
        restore_export(loader.env.scene, runner.og.sim, pool_restore_document(document, isolation))
        loader._scene_state0 = copy.deepcopy(loader.env.scene.dump_state(serialized=False))
        loader._pool_cold_anchor = capture_robot_anchor(loader)
        save(loader, 'world_initialized_before_first_restore')
        return result

    def attach(loader, shared):
        previous = shared['loader']
        baseline = previous._pool_cold_anchor
        row, document, isolation = member(loader)
        anchor = anchor_for_member(baseline, _incoming_spawn(loader, shared['robot']))
        scene = shared['env'].scene
        def to_local(position, orientation):
            return scene.convert_world_pose_to_scene_relative(torch.tensor(position, dtype=torch.float32),
                                                              torch.tensor(orientation, dtype=torch.float32))
        state = merge_member_state(shared['scene_state0'], document, shared['robot'].name, anchor, to_local)
        prepared = {**shared, 'scene_state0': recursively_convert_to_torch(state)}
        original = scene.load_state
        had = 'load_state' in vars(scene); raw = vars(scene).get('load_state')
        transitions = []
        def at_existing_load(*args, **kwargs):
            require(not transitions, 'unexpected repeated scene load in native attach')
            transitions.append(restart_without_steps(runner.og.sim, lambda:
                shared['robot'].set_position_orientation(position=anchor['position'], orientation=anchor['orientation'])))
            result = original(*args, **kwargs)
            restore_export(scene, runner.og.sim, pool_restore_document(document, isolation))
            return result
        scene.load_state = at_existing_load
        try:
            result = native_attach(loader, prepared)
        finally:
            if had: scene.load_state = raw
            else: del scene.load_state
        require(len(transitions) == 1, 'native attach scene restore was not observed')
        loader._pool_cold_anchor = baseline
        loader.task._tro = loader.tro
        loader.task._config_object_bindings = copy.deepcopy(loader.tro.meta.object_bindings)
        ok, feedback = loader.task.initialize_activity(loader.env)
        require(ok, 'incoming pool task binding failed: '+str(feedback))
        save(loader, 'incoming_scene_loaded_before_task_restore', {'anchor': anchor, 'transition': transitions[0]})
        return result

    def restore(loader, *args, **kwargs):
        result = native_restore(loader, *args, **kwargs)
        save(loader, 'after_complete_native_loader')
        return result

    runner.build_env_config = build_config
    runner.InstanceLoader.build_env = build
    runner.InstanceLoader.attach = attach
    runner.InstanceLoader.restore = restore
    return observations


def _incoming_spawn(loader, robot):
    poses = {k.lower(): v for k, v in loader.tro.robot_poses.items()}
    available = poses.get('robot') or poses.get(robot.model.lower())
    require(isinstance(available, list) and len(available) == 1, 'pool incoming spawn ambiguous')
    return available[0]


@contextmanager
def world_construction_counter(og):
    original = og.Environment
    counts = {'attempted': 0, 'completed': 0}
    def counted(*args, **kwargs):
        counts['attempted'] += 1
        env = original(*args, **kwargs)
        counts['completed'] += 1
        return env
    og.Environment = counted
    try:
        yield counts
    finally:
        og.Environment = original


def run(root, request_path, output):
    from .training_batch_probe import inputs
    root = Path(root).resolve(); output = Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'pool output must be new and private')
    value, records = inputs(root, request_path)
    source, contract = verify_inputs(root, value, records)
    runtime = Path(__file__).resolve().parents[2]
    require(file_digest(runtime/'.snapshot_commit') == value['probe_runtime_sha256'], 'pool runtime stamp changed')
    verify_runtime(root, value, records, runtime)
    output.mkdir(parents=True)
    os.environ['ISPATIALGYM_TRAINING_OBSERVATION_REQUEST'] = str(Path(request_path).resolve())
    os.environ['ISPATIALGYM_TRAINING_OBSERVATION_REQUEST_SHA256'] = file_digest(request_path)
    from ispatialgym.oracle import run_isg
    import omnigibson as og
    order = {'FRESH_A': [0], 'FRESH_B': [1], 'ABA': [0, 1, 0]}[value['diagnostic_control']]
    if value.get('pool_preserve_native_anchor_v1') is True:
        from .training_pool_native_anchor_probe import install as install_native_anchor
        install_native_anchor(run_isg, contract, output, install, value)
    elif value.get('pool_exact_source_anchor_v1') is True:
        from .training_pool_source_anchor_probe import install as install_source_anchor
        install_source_anchor(run_isg, contract, output, install, value)
    else:
        install(run_isg, contract, output)
    shared = {'compatible_template_dirs': {os.path.abspath(run_isg.resolve_instance(r['artifacts']['tro']['path'])[1])
                                           for r in records}}
    result = {'schema': 'isg-selected-pool-native-load-v1', 'request_sha256': file_digest(request_path),
              'physical_delta': DELTA, 'benchmark_credit': 0, 'production_released': False,
              'scope': 'Native load and physical isolation only. Action, planner, export and rendering equivalence unverified.',
              'entries': []}
    with world_construction_counter(og) as construction:
        result['world_construction'] = construction
        for ordinal, index in enumerate(order):
            episode = source['episodes'][index]; row = records[index]
            os.environ.update(episode['execution_environment'])
            os.environ.update(value['source_environment_delta'])
            destination = output/('episode_'+str(ordinal))
            argv = ['--instance', row['artifacts']['tro']['path'], '--load-only', '--mode', 'replay',
                    '--replay', row['artifacts']['demo']['path'], '--no-goal-override', '--out', str(destination)]
            code, entry = run_isg.run_instance(run_isg.parse_args(argv), argv[1], shared)
            report = destination/row['instance_id']/'meta.json'
            result['entries'].append({'source_index': index, 'canary_review': value['canaries'][index],
                'returncode': code, 'result': entry, 'report': {'path': str(report), 'sha256': file_digest(report)}})
            write_json_atomic(output/'probe.json', result)
            require(code == 0 and entry.get('status') == 'loaded', 'native pool load failed')
        require(construction['attempted'] == construction['completed'] == 1,
                'pool load did not retain one constructed Environment')
    inputs(root, request_path)
    write_json_atomic(output/'probe.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'request', 'output'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.request, args.output)
    finally:
        native = sys.modules.get('omnigibson')
        if native is not None:
            native.shutdown()
