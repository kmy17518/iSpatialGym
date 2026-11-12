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
        return excluding_inactive(original_filter(*args, **kwargs), initialized, {current['active']})
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
    request, records = inputs(root, request_path)
    source, contract = pool.verify_inputs(root, request, records)
    runtime = Path(__file__).resolve().parents[2]
    require(file_digest(runtime/'.snapshot_commit') == request['probe_runtime_sha256'], 'pool replay runtime changed')
    verify_runtime(root, request, records, runtime, pool.verify_runtime)
    proof = json.loads(Path(request['runtime_compatibility']['path']).read_text())
    output.mkdir(parents=True)
    os.environ['ISPATIALGYM_TRAINING_OBSERVATION_REQUEST'] = str(Path(request_path).resolve())
    os.environ['ISPATIALGYM_TRAINING_OBSERVATION_REQUEST_SHA256'] = file_digest(request_path)
    from ispatialgym.oracle import run_isg as runner
    import omnigibson as og
    install_anchor(runner, contract, output, pool.install, load_request(request, proof))
    current = {}; planner = []
    def observed(value):
        planner.append(value)
        write_json_atomic(output/'pool_planner_audit.json', {'observations': planner, 'physical_delta': DELTA})
    install_planner(runner, set(contract['initialized_actors']), current, observed)
    order = {'FRESH_A': [0], 'FRESH_B': [1], 'ABA': [0, 1, 0]}[request['diagnostic_control']]
    shared = {'compatible_template_dirs': {os.path.abspath(runner.resolve_instance(r['artifacts']['tro']['path'])[1])
                                           for r in records}}
    result = {'schema': 'isg-selected-pool-native-action-replay-v1',
              'request': {'path': str(request_path), 'sha256': file_digest(request_path)},
              'runtime': {'path': str(runtime/'.snapshot_commit'), 'sha256': file_digest(runtime/'.snapshot_commit')},
              'physical_delta': DELTA, 'benchmark_credit': 0, 'production_released': False,
              'exportable': False, 'active_only_export_verified': False,
              'scope': 'Native source action replay and planner exclusion only. Full-union diagnostic recording; '
                       'active-only export and cold/reused rendering are not verified.', 'entries': []}
    try:
        with pool.world_construction_counter(og) as construction:
            result['world_construction'] = construction
            for ordinal, index in enumerate(order):
                row = records[index]; source_episode = source['episodes'][index]
                member = source_member(contract, row['artifacts']['tro']['path'])
                current.clear(); current.update(active=member['active'], tro=row['artifacts']['tro'], ordinal=ordinal)
                os.environ.update(source_episode['execution_environment'])
                os.environ.update(request['source_environment_delta'])
                destination = output/('episode_'+str(ordinal))
                argv = ['--instance', row['artifacts']['tro']['path'], '--mode', 'replay',
                        '--replay', row['artifacts']['demo']['path'], '--record', 'hdf5',
                        '--no-goal-override', '--out', str(destination)]
                code, entry = runner.run_instance(runner.parse_args(argv), argv[1], shared)
                report = destination/row['instance_id']/'meta.json'
                result['entries'].append({'source_index': index, 'canary_review': request['canaries'][index],
                    'returncode': code, 'result': entry,
                    'report': {'path': str(report), 'sha256': file_digest(report)}})
                write_json_atomic(output/'probe.json', result)
                require(current.get('planner_observed') is True, 'native pool planner boundary missing')
                require(code == 0 and entry.get('success') is True, 'native pool action replay failed')
            require(construction == {'attempted': 1, 'completed': 1}, 'pool replay rebuilt its loaded world')
        inputs(root, request_path)
        return result
    finally:
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
