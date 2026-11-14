"""Bounded native action replay on the strictly checked model/scale pool.

Diagnostic recordings retain the full initialized union. They are not an
active-only export and cannot authorize rendering or dataset publication.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from .contracts import require, file_digest, write_json_atomic
from .training_pool_world import active_actor_names

SCHEMA = 'isg-pool-native-action-replay-runtime-v1'
PARENT_COMMIT = 'a05963e4342e024e8da58dc57011637b8a24b9ef'
DELTA = {'name': 'pool_native_action_replay_v1', 'source_inputs_changed': False,
         'load_restore_operations_changed': False, 'native_actions': 'exact_source_demo_open_loop',
         'planner_filter': 'compose_existing_filter_with_inactive_pool_exclusion',
         'planner_audit': 'actual_mesh_enable_mask_after_first_full_sync_before_grasp_attachment',
         'recorded_state_scope': 'full_initialized_union_diagnostic_only',
         'active_only_export_verified': False, 'renderer_equivalence_verified': False,
         'thresholds_changed': False, 'extra_physics_steps': 0}
ALLOWED = {'scripts/campaign_20261110/training_pool_replay_probe.py',
           'scripts/campaign_20261110/tests/test_training_pool_replay_probe.py',
           'scripts/campaign_20261110/training_pool_probe.py',
           'scripts/campaign_20261110/training_pool_consumers.py',
           'scripts/campaign_20261110/tests/test_training_pool_consumers.py'}


def load_request(request, proof=None):
    require(request.get('pool_action_replay_v1') is True and request.get('load_only') is False
            and request.get('pool_preserve_native_anchor_v1') is True
            and request.get('diagnostic_control') in ('FRESH_A', 'FRESH_B', 'ABA')
            and request.get('production_released') is False and request.get('benchmark_credit') == 0,
            'pool replay requires exact bounded native action scope')
    result = {**request, 'load_only': True}
    result.pop('pool_action_replay_v1')
    if proof is not None:
        result.update(runtime_compatibility=proof['parent_compatibility'],
                      probe_runtime_sha256=proof['parent_runtime']['sha256'])
    return result


def verify_runtime(root, request, records, runtime, verify_parent):
    load_request(request)
    reference = request['runtime_compatibility']; path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'], 'pool action proof changed')
    proof = json.loads(path.read_text()); parent_ref = proof['parent_runtime']
    parent = Path(parent_ref['path']).resolve().parent
    require(proof.get('schema') == SCHEMA and proof.get('physical_delta') == DELTA
            and proof.get('permitted_execution_kinds') == ['physics_action_open_loop']
            and parent.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and parent_ref == {'path': str(parent/'.snapshot_commit'), 'sha256': file_digest(parent/'.snapshot_commit')}
            and (parent/'.snapshot_commit').read_text().strip() == PARENT_COMMIT
            and proof['candidate_runtime'] == {'path': str(runtime/'.snapshot_commit'),
                                               'sha256': file_digest(runtime/'.snapshot_commit')},
            'pool action runtime or scope differs')
    verify_parent(root, load_request(request, proof), records, parent)
    paths = {str(p.relative_to(tree)) for tree in (parent, runtime) for folder in ('ispatialgym', 'scripts')
             for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {p for p in paths if not (parent/p).is_file() or not (runtime/p).is_file()
               or file_digest(parent/p) != file_digest(runtime/p)}
    require(changed == set(proof['changed_payloads']) and changed <= ALLOWED
            and all((runtime/p).resolve().is_relative_to(runtime) and file_digest(runtime/p) == h
                    for p, h in proof['changed_payloads'].items()), 'unapproved pool action payload')
    return reference


def install_planner(runner, initialized, current, observe):
    from .training_pool_consumers import excluding_inactive, planner_observation
    original_filter = runner.make_obstacle_filter
    original_update = runner.R1ProCuRobo.update_world
    def filtered(*args, **kwargs):
        return excluding_inactive(original_filter(*args, **kwargs), initialized, active_actor_names(current['active']))
    def updated(backend, *args, **kwargs):
        result = original_update(backend, *args, **kwargs)
        if current.get('planner_observed') is not True:
            require(not args and not kwargs and not backend._attached,
                    'first pool planner observation must be full sync before grasp attachment')
            objects = {name: backend.robot.scene.object_registry('name', name) for name in initialized}
            evidence = planner_observation(backend, objects, current['active'])
            evidence.update(tro=current['tro'], ordinal=current['ordinal'])
            observe(evidence); current['planner_observed'] = True
        return result
    runner.make_obstacle_filter = filtered
    runner.R1ProCuRobo.update_world = updated


def run(root, request_path, output):
    from .training_batch_probe import inputs
    from . import training_pool_probe as pool
    from .training_pool_native_anchor_probe import install as install_anchor
    from .training_pool_adapter import source_member
    root = Path(root).resolve(); output = Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'pool replay output must be new and private')
    runtime = Path(__file__).resolve().parents[2]
    multi_active = json.loads(Path(request_path).read_text()).get('multi_active_replay_v1') is True
    if multi_active:
        from .training_multi_active_replay import inputs as multi_inputs
        request, records, source, contract = multi_inputs(root, request_path, runtime)
        action_request = request
        from . import training_pool_recording_probe as recording
        recovery = None
    else:
        request, records = inputs(root, request_path)
        source, contract = pool.verify_inputs(root, request, records)
        require(file_digest(runtime/'.snapshot_commit') == request['probe_runtime_sha256'], 'pool replay runtime changed')
        pool.verify_runtime(root, request, records, runtime)
        operation_request = request
        if request.get('integrated_candidate_v1') is True:
            from .training_integrated_runtime import source_request
            operation_request, _ = source_request(root, request)
        action_request = operation_request
        recovery = None
        recording = None
        if request.get('pool_active_recording_v1') is True:
            from . import training_pool_recording_probe as recording
            action_request = recording.parent_request(operation_request,
                json.loads(Path(operation_request['runtime_compatibility']['path']).read_text()))
        if request.get('pool_recovery_v1') is True:
            from . import training_pool_recovery_probe as recovery
            action_request = recovery.parent_request(operation_request,
                json.loads(Path(operation_request['runtime_compatibility']['path']).read_text()))
    proof = json.loads(Path(action_request['runtime_compatibility']['path']).read_text())
    output.mkdir(parents=True)
    os.environ['ISPATIALGYM_TRAINING_OBSERVATION_REQUEST'] = str(Path(request_path).resolve())
    os.environ['ISPATIALGYM_TRAINING_OBSERVATION_REQUEST_SHA256'] = file_digest(request_path)
    from ispatialgym.oracle import run_isg as runner
    import omnigibson as og
    native_build_config = runner.build_env_config
    if multi_active:
        from .training_pool_native_anchor_probe import install_prepared
        install_prepared(runner, contract, output, pool.install, action_request)
    else:
        install_anchor(runner, contract, output, pool.install, load_request(action_request, proof))
    current = {}; planner = []
    flush_recording = recording.install(runner, contract, current, output, native_build_config) if recording else None
    cleanups = []
    if recovery is not None:
        recovery.install(runner, request['diagnostic_control'], current, cleanups)
    def observed(value):
        planner.append(value)
        write_json_atomic(output/'pool_planner_audit.json', {'observations': planner, 'physical_delta': DELTA})
    install_planner(runner, set(contract['initialized_actors']), current, observed)
    orders = recovery.ORDERS if recovery is not None else {'FRESH_A': [0], 'FRESH_B': [1], 'ABA': [0, 1, 0]}
    order = orders[request['diagnostic_control']]
    shared = {'compatible_template_dirs': {os.path.abspath(runner.resolve_instance(r['artifacts']['tro']['path'])[1])
                                           for r in records}}
    result = {'schema': 'isg-selected-pool-native-action-replay-v1',
              'request': {'path': str(request_path), 'sha256': file_digest(request_path)},
              'runtime': {'path': str(runtime/'.snapshot_commit'), 'sha256': file_digest(runtime/'.snapshot_commit')},
              'physical_delta': DELTA, 'benchmark_credit': 0, 'production_released': False,
              'exportable': False, 'active_only_export_verified': False,
              'scope': 'Native source action replay and planner exclusion only. Full-union diagnostic recording; '
                       'active-only export and cold/reused rendering are not verified.', 'entries': []}
    if recovery is not None: result['recovery_delta'] = recovery.DELTA
    if recording is not None:
        result['recording_delta'] = recording.DELTA
        result['scope'] = 'Native action replay with active-only serialized-state and scene export diagnostic. Renderer and dataset release remain unverified.'
    final_isolation = []
    try:
        with pool.world_construction_counter(og) as construction:
            result['world_construction'] = construction
            for ordinal, index in enumerate(order):
                row = records[index]; source_episode = source['episodes'][index]
                if multi_active:
                    from .training_multi_active_replay import authorize_member
                    authorize_member(root, request, source_episode, row)
                member = source_member(contract, row['artifacts']['tro']['path'])
                current.clear(); current.update(active=member['active'], tro=row['artifacts']['tro'], ordinal=ordinal)
                environment_binding = None
                if member.get('task_definition') is not None:
                    from .training_pool_task import select_episode_environment
                    environment_binding = select_episode_environment(os.environ,
                        source_episode['execution_environment'], request['source_environment_delta'])
                else:
                    os.environ.update(source_episode['execution_environment'])
                    os.environ.update(request['source_environment_delta'])
                destination = output/('episode_'+str(ordinal))
                argv = ['--instance', row['artifacts']['tro']['path'], '--mode', 'replay',
                        '--replay', row['artifacts']['demo']['path'], '--record', 'hdf5',
                        '--no-goal-override', '--out', str(destination)]
                driver = None
                if request.get('native_cli_driver_v1') is True:
                    from .training_native_driver import (canary_driver, replay_arguments,
                                                         apply_native_driver, verify_driver_report)
                    driver_source, driver_original, driver_binding = canary_driver(root, request, index)
                    argv = replay_arguments(driver_original, driver_source, destination)
                    if multi_active:
                        from .training_native_driver import apply_source_fast_settings
                        fast_settings = apply_source_fast_settings(driver_original, request['source_environment_delta'])
                args = runner.parse_args(argv)
                if request.get('native_cli_driver_v1') is True:
                    driver = apply_native_driver(runner, args, driver_original,
                        driver_binding['original_own_report'], world_already_loaded=bool(shared.get('env')))
                try:
                    code, entry = runner.run_instance(args, driver_source['source_tro']['path']
                        if driver is not None else argv[1], shared)
                finally:
                    while cleanups: cleanups.pop()()
                report = destination/row['instance_id']/'meta.json'
                if driver is not None:
                    verify_driver_report(driver_original, json.loads(report.read_text()))
                result['entries'].append({'source_index': index, 'canary_review': request['canaries'][index],
                    'returncode': code, 'result': entry,
                    'report': {'path': str(report), 'sha256': file_digest(report)}})
                if driver is not None:
                    result['entries'][-1]['native_cli_driver'] = driver
                if multi_active:
                    result['entries'][-1]['source_fast_settings'] = fast_settings
                if environment_binding is not None:
                    result['entries'][-1]['exact_source_environment'] = environment_binding
                if recording is not None:
                    result['entries'][-1]['active_recording_scene'] = current.get('active_recording_scene')
                    flush_recording()
                write_json_atomic(output/'probe.json', result)
                require(current.get('planner_observed') is True, 'native pool planner boundary missing')
                aborted = recovery is not None and recovery.verify_expected_abort(
                    request['diagnostic_control'], ordinal, code, entry, json.loads(report.read_text()))
                if aborted:
                    result['entries'][-1]['expected_controlled_abort'] = True
                    write_json_atomic(output/'probe.json', result)
                else:
                    require(code == 0 and entry.get('success') is True, 'native pool action replay failed')
                scene = shared['env'].scene
                _, expected = pool.isolated_member_scene(contract,
                    json.loads(Path(member['scene']['path']).read_text()), member['active'])
                bindings = {k: v.name for k, v in shared['task'].object_scope.items() if hasattr(v, 'name')}
                measured = pool.native_isolation(scene, shared['robot'], expected, bindings, None)
                final_isolation.append({'ordinal': ordinal, 'tro': member['tro'],
                                        'phase': 'after_native_action_replay', 'isolation': measured})
                write_json_atomic(output/'pool_final_isolation.json', {'observations': final_isolation})
            require(construction == {'attempted': 1, 'completed': 1}, 'pool replay rebuilt its loaded world')
        if multi_active: multi_inputs(root, request_path, runtime)
        else: inputs(root, request_path)
        return result
    finally:
        if flush_recording is not None: flush_recording()
        write_json_atomic(output/'probe.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'request', 'output'): parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    try:
        run(args.root, args.request, args.output)
    finally:
        native = sys.modules.get('omnigibson')
        if native is not None: native.shutdown()
