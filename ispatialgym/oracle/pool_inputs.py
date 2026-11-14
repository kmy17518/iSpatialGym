"""Bounded source-exact CLI and queue route for the native pool consumers.

This route consumes independently authorized source replays. It does not
authorize training production, change source purposes, or bypass native gates.
"""
import json
import os
import re
import time
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from .pool_world import PoolWorld


def enabled(args):
    return (getattr(args, 'pool_world_request', None) is not None
            or getattr(args, 'pool_world_request_sha256', None) is not None)


def bind(args, runtime):
    from scripts.campaign_20261110.training_multi_active_replay import inputs, scope
    path = getattr(args, 'pool_world_request', None)
    sha = getattr(args, 'pool_world_request_sha256', None)
    require(path and isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha),
            'native pool request needs its exact SHA256')
    path = Path(path).resolve()
    require(file_digest(path) == sha, 'native pool request changed')
    value = json.loads(path.read_text())
    root = Path(value['campaign_root']).resolve()
    require(root.name == value['campaign_id'] == 'campaign_20261110'
            and path.is_relative_to(root), 'native pool request escaped campaign')
    q, records, source, contract = inputs(root, path, Path(runtime).resolve())
    require(q.get('native_pool_consumers_v1') is True,
            'native pool CLI needs the reviewed native consumer proof')
    require(not any(getattr(args, key, None) for key in
                    ('task', 'prebuild', 'restoration_policy_manifest', 'restoration_policy_manifest_sha256')),
            'native pool route cannot combine unrelated task/prebuild/restoration selectors')
    order = scope(q)
    queue = getattr(args, 'queue', None)
    if queue:
        require(Path(queue).resolve().is_relative_to(root)
                and args.max_instances == len(order),
                'bounded native pool queue needs its exact episode count')
    else:
        require(len(order) == 1 and str(Path(args.instance).resolve()) == records[order[0]]['artifacts']['tro']['path'],
                'native pool singleton differs from the exact request')
    output = Path(args.out).resolve()
    require(output.is_relative_to(root) and not output.exists(),
            'native pool output must be new and inside the campaign')
    return root, path, q, records, source, contract, order, output


def run(runner, args):
    """Use native run_instance and its per-loader interfaces; no method patches."""
    from scripts.campaign_20261110.training_multi_active_replay import authorize_member
    from scripts.campaign_20261110.training_native_driver import (
        canary_driver, replay_arguments, apply_native_driver,
        apply_source_fast_settings, verify_driver_report)
    from scripts.campaign_20261110.training_pool_task import select_episode_environment
    runtime = Path(runner.__file__).resolve().parents[2]
    root, path, q, records, source, contract, order, output = bind(args, runtime)
    request_ref = {'path': str(path), 'sha256': file_digest(path)}
    claims = Path(args.claims or output.parent/(output.name + '_claims')).resolve()
    require(claims.is_relative_to(root), 'native pool claims escaped campaign')
    claims.mkdir(parents=True, exist_ok=True)
    names = [f"pool_{request_ref['sha256']}_{i:04d}" for i in range(len(order))]
    require(not any((claims/name).exists() for name in names),
            'native pool sequence already has an owner; preserve its claims')
    output.mkdir(parents=True)
    os.environ.update(ISGF_ROOT=str(root), ISPATIALGYM_CAMPAIGN_RESTORATION_PROBE='1',
                      ISPATIALGYM_CAMPAIGN_RESTORATION_CONTEXT='1',
                      ISPATIALGYM_TRAINING_OBSERVATION_REQUEST=str(path),
                      ISPATIALGYM_TRAINING_OBSERVATION_REQUEST_SHA256=request_ref['sha256'])
    shared = {}
    result = {'schema': 'isg-native-pool-cli-queue-observation-v1', 'request': request_ref,
              'native_runtime': q['native_runtime'], 'native_pool_consumer_delta': q['native_pool_consumer_delta'],
              'production_released': False, 'benchmark_credit': 0, 'entries': [],
              'status': 'running', 'started_at': time.time(),
              'world_loads': 0, 'world_reuses': 0}
    destination = output/'native_pool_queue.json'
    write_json_atomic(destination, result)
    try:
        for ordinal, index in enumerate(order):
            require(not args.deadline or time.time() < args.deadline,
                    'native pool queue deadline expired')
            require(runner._claim(str(claims), names[ordinal]),
                    'native pool episode was claimed by another worker')
            episode, record = source['episodes'][index], records[index]
            authorize_member(root, q, episode, record)
            environment = select_episode_environment(os.environ, episode['execution_environment'],
                                                     q['source_environment_delta'])
            original_source, original, binding = canary_driver(root, q, index)
            fast = apply_source_fast_settings(original, q['source_environment_delta'])
            target = output/f'episode_{ordinal:04d}'
            target.mkdir()
            argv = replay_arguments(original, original_source, target)
            native_args = runner.parse_args(argv)
            driver = apply_native_driver(runner, native_args, original,
                                         binding['original_own_report'], world_already_loaded=bool(shared))
            world = PoolWorld(contract, record['artifacts']['tro']['path'], target)
            before = shared.get('env')
            code, native_result = runner.run_instance(native_args, record['artifacts']['tro']['path'],
                                                      shared, pool_world=world)
            report_path = target/record['instance_id']/'meta.json'
            report = json.loads(report_path.read_text())
            entry = {'ordinal': ordinal, 'source_index': index, 'claim': str(claims/names[ordinal]),
                     'source_tro': record['artifacts']['tro'], 'source_review': episode['review'],
                     'returncode': code, 'result': native_result, 'native_cli_driver': driver,
                     'source_fast_settings': fast, 'source_environment': environment,
                     'report': {'path': str(report_path), 'sha256': file_digest(report_path)},
                     'world_identity': id(shared.get('env'))}
            result['entries'].append(entry)
            result['world_reuses' if before is not None else 'world_loads'] += 1
            write_json_atomic(destination, result)
            verify_driver_report(original, report)
            require(code == 0 and native_result.get('success') is True,
                    'native pool source replay failed; claims and evidence retained')
            require(shared.get('env') is not None and (before is None or before is shared['env']),
                    'native pool queue replaced its loaded world')
        require(result['world_loads'] == 1 and result['world_reuses'] == len(order)-1,
                'native pool queue world count differs')
        result['status'] = 'native_completed_unverified'
        return 0
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['finished_at'] = time.time()
        write_json_atomic(destination, result)
