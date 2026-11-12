"""Bounded supervisor for exact retained room transitions and compatible reuse.

Every incompatible transition uses the actual native compatibility decision and
starts a fresh child. The existing mechanism driver owns all physics/actions.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .contracts import require, write_json_atomic
from .training_loading_boundary_contract import inputs, native_boundary, ref, validate_sequence


def run_segment(root, request_path, output, ordinal):
    from .training_mechanism_probe import run as run_mechanism
    q, sources, plan = inputs(root, request_path)
    require(type(ordinal) is int and 0 <= ordinal < len(plan), 'invalid native segment')
    spec = plan[ordinal]
    result = run_mechanism(root, spec['request']['path'], output)
    boundary = None; code = 0
    if ordinal+1 < len(plan):
        from ispatialgym.oracle import run_isg as runner
        current = sources[spec['source']]['row']['source_tro']['path']
        incoming = sources[plan[ordinal+1]['source']]['row']['source_tro']['path']
        boundary = native_boundary(runner, current, incoming)
        require(boundary['reuse_allowed'] is False and runner.EXIT_QUEUE_RESTART == 75,
                'room transition is compatible; refusing an unnecessary fresh-world claim')
        code = runner.EXIT_QUEUE_RESTART
    observation = {'ordinal': ordinal, 'source': spec['source'], 'process_pid': os.getpid(),
                   'scheduler_job_id': os.environ.get('SLURM_JOB_ID'), 'returncode': code,
                   'boundary': boundary, 'probe': ref(Path(output)/'probe.json'),
                   'native_runtime': q['native_runtime'], 'request': ref(request_path)}
    write_json_atomic(Path(output)/'loading_boundary.json', observation)
    return code, observation


def run(root, request_path, output, *, preflight=False, runner=subprocess.run):
    root = Path(root).resolve(); output = Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(), 'private loading-boundary output required')
    q, sources, plan = inputs(root, request_path)
    if preflight:
        return {'passed': True, 'native_execution': False, 'request': ref(request_path),
                'expected_native_episodes': q['max_native_episodes'], 'expected_world_loads': len(plan),
                'rooms': [source['room'] for source in sources], 'native_runtime': q['native_runtime']}
    output.mkdir(parents=True); started = time.time(); began = time.monotonic()
    result = {'schema': 'isg-retained-loading-boundary-observation-v1', 'request': ref(request_path),
              'native_runtime': q['native_runtime'], 'control': q['control'], 'started_at': started,
              'scheduler_job_id': os.environ.get('SLURM_JOB_ID'), 'segments': [], 'world_loads': 0,
              'native_episodes': 0, 'benchmark_credit': 0, 'production_released': False}
    children = []
    try:
        for ordinal, spec in enumerate(plan):
            remaining = q['timeout_seconds']-(time.monotonic()-began)
            require(remaining > 0, 'loading-boundary budget exhausted')
            destination = output/('segment_'+str(ordinal))
            command = [sys.executable, '-m', __name__, '--root', str(root), '--request', str(request_path),
                       '--output', str(destination), '--segment', str(ordinal)]
            # __name__ is __main__ under CLI; always bind the exact module.
            command[2] = 'scripts.campaign_20261110.training_loading_boundary_probe'
            # Each fresh child gets its own retained source settings. Do not
            # carry A-only task flags into B merely because its wrapper ran first.
            environment = {key:value for key,value in os.environ.items()
                           if not key.startswith(('ISPATIALGYM_', 'ISG_'))}
            environment.update(sources[spec['source']]['environment'])
            environment['HDF5_USE_FILE_LOCKING'] = 'FALSE'
            with (output/('segment_'+str(ordinal)+'.log')).open('w') as log:
                completed = runner(command, stdout=log, stderr=subprocess.STDOUT, timeout=remaining,
                                   env=environment)
            exit_path = output/('segment_'+str(ordinal)+'_exit.json')
            write_json_atomic(exit_path,
                              {'ordinal': ordinal, 'returncode': completed.returncode,
                               'request': ref(request_path), 'native_runtime': q['native_runtime']})
            path = destination/'loading_boundary.json'; segment = json.loads(path.read_text())
            require(completed.returncode == segment['returncode']
                    and segment['request'] == ref(request_path)
                    and segment['native_runtime'] == q['native_runtime'], 'native child exit or identity differs')
            segment['boundary_observation'] = ref(path)
            segment['child_exit_observation'] = ref(exit_path)
            result['segments'].append(segment)
            child = json.loads(Path(segment['probe']['path']).read_text()); children.append(child)
            result['world_loads'] += child['world_construction']['completed']
            result['native_episodes'] += len(child['entries'])
            require(completed.returncode == (75 if ordinal+1 < len(plan) else 0),
                    'native child failed instead of its declared loading boundary')
            write_json_atomic(output/'probe.json', result)
        result.update(status='completed', finished_at=time.time())
        validate_sequence(q, result, children)
        inputs(root, request_path)
    except BaseException as error:
        result.update(status='failed', error=repr(error), finished_at=time.time())
        raise
    finally:
        write_json_atomic(output/'probe.json', result)
    return result


def shutdown_and_exit(code, native, *, exit_process=os._exit):
    """Preserve native queue restart status across Isaac's teardown handlers."""
    sys.stdout.flush(); sys.stderr.flush()
    if native is not None:
        try:
            native.shutdown()
        except SystemExit as error:
            if error.code not in (None, 0):
                traceback.print_exc(); code = 1
        except BaseException:
            traceback.print_exc(); code = 1
    sys.stdout.flush(); sys.stderr.flush()
    exit_process(code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--request', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--segment', type=int); parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(); code = 0
    try:
        if args.segment is None: value = run(args.root, args.request, args.output, preflight=args.preflight)
        else: code, value = run_segment(args.root, args.request, args.output, args.segment)
        print(json.dumps(value), flush=True)
    except BaseException:
        traceback.print_exc(); code = 1
    shutdown_and_exit(code, sys.modules.get('omnigibson'))
