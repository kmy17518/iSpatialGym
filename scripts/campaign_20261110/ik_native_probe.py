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
import sys
import time

CAMPAIGN = 'campaign_20261110'
SCHEMA = 'isg-exact-ik-native-probe-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def descendant(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError('probe input/output outside its campaign root')
    return path


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
    os.environ['ISPATIALGYM_SAMPLING_CACHE'] = str(output / 'ordinary_sampling_cache')
    report = {'schema': SCHEMA, 'campaign_id': CAMPAIGN, 'diagnostic_only': True,
              'benchmark_credit': 0, 'eligible_for_admission_or_acceptance': False,
              'spec_sha256': sha(args.spec), 'snapshot': str(paths['snapshot']),
              'reference_index': spec['reference_index'], 'instance_publication_forbidden': True,
              'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'starting', 'queries': []}

    def save():
        report['observed_at'] = datetime.now(timezone.utc).isoformat()
        temporary = output / 'probe_report.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output / 'probe_report.json')

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

    IKChecker.exact = exact
    exact_exchange.register_query = register
    SamplerV2.write_instance = no_publication
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
