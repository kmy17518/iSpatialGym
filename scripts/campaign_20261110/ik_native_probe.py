"""Bounded native sampler prefix for measured exact-IK cache transfer validation.

This diagnostic uses the recorded source's real sampling path. It never writes
an instance, modifies a source, or produces benchmark credit. The unchanged
native IK method performs every current-world check before a hit is recorded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

CAMPAIGN = 'campaign_20261110'
SCHEMA = 'isg-exact-ik-native-probe-v1'
OBSERVATION_ENV_KEYS = ('ISGE_CODE', 'ISPATIALGYM_EXACT_IK_MANIFEST',
    'ISPATIALGYM_EXACT_IK_LOCAL_CACHE', 'ISPATIALGYM_EXACT_IK_METRICS',
    'ISPATIALGYM_EXACT_IK_PRODUCER', 'ISGF_SAMPLING_CACHE',
    'ISPATIALGYM_GRASP_LIBRARY_V2', 'ISPATIALGYM_REACH_MAPS',
    'ISPATIALGYM_MAP_OVERLAY_SHARDS', 'ISGF_ASSET_CACHE',
    'KIT_TPL_NFS_ROOT', 'KIT_TEX_STORE')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_observation_environment(output, snapshot, env):
    values = {k: env[k] for k in OBSERVATION_ENV_KEYS if k in env}
    values['ISGE_CODE'] = str(snapshot)
    path = Path(output) / 'job.env'
    with path.open('x') as stream:
        stream.write(''.join(k + '=' + shlex.quote(v) + '\n' for k, v in sorted(values.items())))
    return {'path': str(path), 'sha256': sha(path), 'keys': sorted(values),
            'kind': 'allowlisted_process_start_environment',
            'runtime_sharing_requires_metrics_verification': True}


def compatibility_difference(expected, actual):
    """Bounded, credential-free differences; normalize JSON tuple/list values."""
    def same(a, b):
        return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    result = {}
    for key in sorted(set(expected) | set(actual)):
        left, right = expected.get(key), actual.get(key)
        if same(left, right):
            continue
        if isinstance(left, dict) and isinstance(right, dict):
            result[key] = [name for name in sorted(set(left) | set(right))
                           if not same(left.get(name), right.get(name))]
        else:
            result[key] = 'differs'
    return result


def descendant(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('probe input/output outside its campaign root')
    return path


def isolate_probe_writes(output, env):
    """Keep writable sampler telemetry out of the preserved source job."""
    output = Path(output).resolve()
    previous = {key: env.get(key) for key in
                ('ISPATIALGYM_SAMPLING_CACHE', 'ISPATIALGYM_MAP_OVERLAY_SHARDS')}
    env['ISPATIALGYM_SAMPLING_CACHE'] = str(output / 'ordinary_sampling_cache')
    env['ISPATIALGYM_MAP_OVERLAY_SHARDS'] = str(output / 'map_overlay')
    return {'previous': previous, 'effective': {key: env[key] for key in previous},
            'source_files_modified': False}


def validate_spec(spec, output):
    if spec.get('schema') != SCHEMA or spec.get('campaign_id') != CAMPAIGN:
        raise ValueError('foreign native probe spec')
    if spec.get('diagnostic_only') is not True or spec.get('benchmark_credit') != 0:
        raise ValueError('probe must remain permanently nonaccepting')
    root = Path(spec['source_root']).resolve()
    out_root = Path(spec.get('output_root', root)).resolve()
    if root.name != CAMPAIGN or out_root.name != CAMPAIGN:
        raise ValueError('wrong campaign root')
    output = descendant(output, out_root / 'ik_exchange/probes')
    if output.exists() and any(output.iterdir()):
        raise ValueError('probe output already has evidence; never overwrite')
    paths = {k: descendant(spec[k], root) for k in ('snapshot', 'config', 'manifest', 'template', 'customs')}
    required = [paths['snapshot'] / '.snapshot_commit', paths['config'], paths['manifest'],
                paths['template'] / 'template.json', paths['template'] / 'template_meta.json', paths['customs']]
    for path in required:
        if spec.get('inputs', {}).get(str(path)) != sha(path):
            raise ValueError('missing or changed pinned input: ' + str(path))
    config = json.loads(paths['config'].read_text())
    if config.get('split') not in ('eval_standard', 'eval_model', 'eval_category'):
        raise ValueError('cache probe must use an authorized eval source')
    if config['validation_key']['code_sha256'] != sha(paths['snapshot'] / '.snapshot_commit'):
        raise ValueError('native runtime differs from source validation key')
    setup = config['template']['scene_setup']
    if Path(setup['source_template']).resolve() != paths['template'] or setup['source_template_sha256'] != sha(paths['template'] / 'template.json'):
        raise ValueError('source scene differs from exact configuration')
    if not all(o.get('ready') is True for o in config['template']['objects']):
        raise ValueError('source includes a non-ready asset')
    if type(spec.get('reference_index')) is not int or spec['reference_index'] < 0:
        raise ValueError('recorded source index required')
    if not 1 <= spec.get('eligible_query_limit', 32) <= 64 or not 1 <= spec.get('total_query_limit', 300) <= 500:
        raise ValueError('query bounds exceed diagnostic allowance')
    if spec.get('bddl_root'):
        descendant(spec['bddl_root'], root)
    return paths, output


class ProbeFinished(BaseException):
    pass


def observe_native_shutdown(before):
    # multiply_isg imports og locally inside main(), so it has no module-level
    # og attribute. Observe the actual module shared by that local import.
    import omnigibson
    original = omnigibson.shutdown

    def shutdown(*args, **kwargs):
        before()
        return original(*args, **kwargs)

    omnigibson.shutdown = shutdown
    return lambda: setattr(omnigibson, 'shutdown', original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    paths, output = validate_spec(spec, args.output)
    output.mkdir(parents=True)
    sys.path.insert(0, str(paths['snapshot']))
    if os.environ.get('ISPATIALGYM_CAMPAIGN_CONFIG') != str(paths['config']):
        raise ValueError('effective configuration differs from probe spec')
    if os.environ.get('ISPATIALGYM_EXACT_IK_MANIFEST') != str(paths['manifest']):
        raise ValueError('effective exact-IK manifest differs from probe spec')
    # Only typed, explicitly imported witnesses may carry across this experiment.
    # Other sampler caches begin empty for each probe and never touch the source.
    write_isolation = isolate_probe_writes(output, os.environ)
    report = {'schema': SCHEMA, 'campaign_id': CAMPAIGN, 'diagnostic_only': True,
              'benchmark_credit': 0, 'eligible_for_admission_or_acceptance': False,
              'spec_sha256': sha(args.spec), 'snapshot': str(paths['snapshot']),
              'reference_index': spec['reference_index'], 'instance_publication_forbidden': True,
              'write_isolation': write_isolation,
              'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'starting', 'queries': []}

    def save():
        report['observed_at'] = datetime.now(timezone.utc).isoformat()
        temporary = output / 'probe_report.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output / 'probe_report.json')

    report['environment_observation'] = write_observation_environment(output, paths['snapshot'], os.environ)
    save()
    # Build with the actual native interpreter and effective worker environment.
    # ast.dump fingerprints differ between Python versions: a manifest prepared
    # by a separate coordinator can otherwise fail only after costly GPU setup.
    from scripts.campaign_20261110 import ik_compatibility
    descendant(ik_compatibility.__file__, paths['snapshot'])
    try:
        manifest = ik_compatibility.validate_manifest(json.loads(paths['manifest'].read_text()))
        actual = ik_compatibility.build_manifest(paths['snapshot'], paths['config'], dict(os.environ),
                                                manifest['provenance']['robot_refs'])
        report['compatibility_preflight'] = {
            'python_version': sys.version.split()[0],
            'declared_sha256': manifest['compatibility_sha256'],
            'actual_sha256': actual['compatibility_sha256'],
            'differences': compatibility_difference(manifest['compatible_identity'], actual['compatible_identity'])}
        if actual['compatibility_sha256'] != manifest['compatibility_sha256']:
            raise ValueError('native interpreter/effective inputs differ from cache manifest')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.update(status='failed_compatibility_preflight', error=str(exc), published_tro_files=0,
                      completed_at=datetime.now(timezone.utc).isoformat())
        save()
        raise
    save()
    # Import through the pinned native snapshot, following normal bootstrap.
    from ispatialgym.sampling import multiply_isg
    from ispatialgym.sampling.v2.ik import IKChecker
    from ispatialgym.sampling.v2.core import SamplerV2
    from ispatialgym.sampling import exact_exchange
    modules = [multiply_isg, importlib.import_module(IKChecker.__module__),
               importlib.import_module(SamplerV2.__module__), exact_exchange]
    for module in modules:
        descendant(module.__file__, paths['snapshot'])
    report['loaded_native_modules'] = {m.__name__: str(Path(m.__file__).resolve()) for m in modules}
    report['ordinary_sampling_cache'] = os.environ['ISPATIALGYM_SAMPLING_CACHE']
    original_exact = IKChecker.exact
    original_register = exact_exchange.register_query
    original_write = SamplerV2.write_instance
    current_keys = []

    def register(key, *values, **kwargs):
        result = original_register(key, *values, **kwargs)
        current_keys.append(key)
        return result

    def exact(self, *values, **kwargs):
        current_keys.clear()
        before = dict(exact_exchange._COUNTERS)
        started = time.monotonic()
        result = original_exact(self, *values, **kwargs)
        after = dict(exact_exchange._COUNTERS)
        report['queries'].append({'keys': list(current_keys), 'ok': bool(result.ok),
            'reason': result.reason, 'n_checked': result.n_checked,
            'seconds': time.monotonic() - started,
            'counter_delta': {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}})
        report['counters'] = after
        eligible = after.get('exact_cache_eligible_queries', 0)
        total = after.get('exact_queries', 0)
        if eligible >= spec.get('eligible_query_limit', 32) or total >= spec.get('total_query_limit', 300):
            report['status'] = 'bounded_prefix_complete'
            report['stop_reason'] = 'eligible_query_bound' if eligible >= spec.get('eligible_query_limit', 32) else 'total_query_bound'
            exact_exchange.record_event('probe_flush', 0, force=True)
            save()
            raise ProbeFinished()
        if len(report['queries']) % 10 == 0:
            save()
        return result

    def no_publication(self, *values, **kwargs):
        report['status'] = 'bounded_prefix_complete'
        report['stop_reason'] = 'before_instance_publication'
        save()
        raise ProbeFinished()

    def before_shutdown():
        # Kit can exit the process from shutdown, bypassing our outer finally.
        # Flush measured evidence before the unchanged native shutdown call.
        if report['status'] == 'running_native_prefix':
            report['status'] = 'one_draw_finished'
        report['counters'] = dict(exact_exchange._COUNTERS)
        report['published_tro_files'] = len(list(output.rglob('*-tro_state.json')))
        if report['published_tro_files']:
            report['status'] = 'failed_publication_guard'
        report['native_shutdown_observed_at'] = datetime.now(timezone.utc).isoformat()
        exact_exchange.record_event('probe_flush', 0, force=True)
        save()

    IKChecker.exact = exact
    exact_exchange.register_query = register
    SamplerV2.write_instance = no_publication
    restore_shutdown = observe_native_shutdown(before_shutdown)
    argv = ['--split', 'train', '--template', str(paths['template']), '--instances', '1',
            '--start-index', str(spec['reference_index']), '--max-attempts', '1',
            '--customs', str(paths['customs']), '--keep-template-scene', '--no-write-back',
            '--out', str(output / 'native_prefix')]
    if spec.get('bddl_root'):
        argv += ['--bddl-root', spec['bddl_root']]
    if spec.get('seed') is not None:
        argv += ['--seed', str(int(spec['seed']))]
    report['sampler_arguments'] = argv
    report['status'] = 'running_native_prefix'
    save()
    try:
        multiply_isg.main(argv)
    except ProbeFinished:
        pass
    except SystemExit as exc:
        report['sampler_exit_code'] = exc.code
        report['status'] = 'one_draw_finished'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = type(exc).__name__ + ': ' + str(exc)[:1000]
        raise
    finally:
        IKChecker.exact = original_exact
        exact_exchange.register_query = original_register
        SamplerV2.write_instance = original_write
        restore_shutdown()
        exact_exchange.record_event('probe_flush', 0, force=True)
        report['counters'] = dict(exact_exchange._COUNTERS)
        report['published_tro_files'] = len(list(output.rglob('*-tro_state.json')))
        if report['published_tro_files']:
            report['status'] = 'failed_publication_guard'
        report['completed_at'] = datetime.now(timezone.utc).isoformat()
        save()
    if report['published_tro_files']:
        raise RuntimeError('diagnostic unexpectedly published a native instance')
    print(json.dumps({'report': str(output / 'probe_report.json'), 'status': report['status'],
                      'benchmark_credit': 0, 'counters': report['counters']}))


if __name__ == '__main__':
    main()
