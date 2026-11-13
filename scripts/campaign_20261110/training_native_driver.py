"""Bind a direct native replay to its own recorded CLI physics initialization.

This is opt-in for newly authored controls. Old direct-call requests retain their
original semantics and are not relabeled as native CLI comparisons.
"""
import json
import os
from pathlib import Path

from .contracts import file_digest, require


DRIVER_DELTA = {
    'setup': 'native_run_isg.apply_driver_env_before_world_load',
    'arguments': 'original_own_replay_argv_with_private_output_and_reviewed_policy_only',
    'preworld_check': 'exact_original_rush_env_and_driver_physics',
    'postrun_check': 'exact_original_rush_env_and_driver_physics',
    'previous_direct_call_omission_corrected': True,
    'thresholds_changed': False,
}


def reference(value):
    return {key: value[key] for key in ('path', 'sha256')}


def original_driver(root, request, source):
    enabled = request.get('native_cli_driver_v1', False)
    require(type(enabled) is bool, 'native CLI driver flag must be boolean')
    if not enabled:
        require('native_driver_reference' not in request, 'unused native driver reference')
        return None
    item = request.get('native_driver_reference')
    require(item == reference(source['own_replay_meta']),
            'native driver must use this exact original own replay report')
    path = Path(item['path']).resolve()
    require(path.is_relative_to(Path(root).resolve()) and file_digest(path) == item['sha256'],
            'original native driver report changed or escaped campaign')
    report = json.loads(path.read_text())
    require(report.get('mode') == 'replay' and report.get('success') is True
            and report.get('instance_id') == source['instance_id']
            and report.get('replay') == source['collector_demo']['path']
            and isinstance(report.get('rush_env'), dict)
            and all(isinstance(k, str) and k.startswith('ISG_') and isinstance(v, str)
                    for k, v in report['rush_env'].items())
            and set(report.get('driver_physics', {})) == {'rigid_overrides', 'sleep_assist', 'lean_step'},
            'original own replay driver evidence is incomplete or belongs to another source')
    # Validate the input/action binding even during CPU-only author preflight.
    replay_arguments(report, source, '/private/preflight/output')
    return report


def canary_driver(root, request, index):
    """Resolve the own replay through the immutable, separately checked canary.

    This adds driver provenance; the normal canary/source authorization and
    original runtime proof still run independently before physical mutation.
    """
    require(request.get('native_cli_driver_v1') is True
            and (request.get('integrated_candidate_v1') is True
                 or request.get('multi_active_replay_v1') is True),
            'canary CLI setup requires the explicit integrated driver proof')
    item = request['canaries'][index]
    path = Path(item['path']).resolve()
    require(path.is_relative_to(Path(root).resolve()) and file_digest(path) == item['sha256'],
            'native driver canary review changed')
    review = json.loads(path.read_text())
    require(review.get('schema') == 'isg-training-canary-review-v1'
            and review.get('validation_passed') is True and review.get('accepted') is False
            and review.get('eligible_for_benchmark') is False,
            'native driver requires an independently reviewed zero-credit canary')
    evidence = [e for e in review['source_proof']['evidence'] if e['kind'] == 'open_loop_replay']
    require(len(evidence) == 1, 'canary original own replay missing or ambiguous')
    own = {'path': evidence[0]['report_path'], 'sha256': evidence[0]['report_sha256']}
    record = review['export_record']
    require(record['instance_id'] == review['instance_id'], 'canary driver instance differs')
    source = {'instance_id': record['instance_id'], 'source_tro': record['artifacts']['tro'],
              'collector_demo': record['artifacts']['demo'], 'own_replay_meta': own}
    original = original_driver(root, {'native_cli_driver_v1': True,
                               'native_driver_reference': own}, source)
    return source, original, {'canary_review': dict(item), 'instance_id': source['instance_id'],
                              'original_own_report': own}


def replay_arguments(original, source, output, policy_arguments=()):
    argv = original.get('argv')
    require(isinstance(argv, list) and all(isinstance(v, str) for v in argv)
            and not any(v.startswith('--restoration-policy-') for v in argv),
            'original replay arguments are missing or contain another restoration policy')
    argv = list(argv)
    for key, expected in (('--instance', source['source_tro']['path']),
                          ('--mode', 'replay'), ('--replay', source['collector_demo']['path']),
                          ('--record', 'hdf5')):
        require(argv.count(key) == 1 and argv.index(key) + 1 < len(argv)
                and argv[argv.index(key) + 1] == expected,
                'original replay arguments changed source, actions, mode or recording')
    require(argv.count('--out') == 1 and argv.index('--out') + 1 < len(argv)
            and argv.count('--no-goal-override') == 1,
            'original replay lacks explicit private-output or goal contract')
    argv[argv.index('--out') + 1] = str(output)
    return argv + list(policy_arguments)


def native_physics(runner, args):
    """Use the same native accessors as run_isg's saved driver_physics."""
    sa = runner.SA
    assist = {'config': os.environ.get('ISG_SLEEP_ASSIST'), 'mode': sa.sleep_assist_mode(),
              'max_mass_kg': sa.sleep_assist_max_mass(), 'wake_radius_m': sa.sleep_assist_wake_radius()}
    if sa.restart_on_wake():
        assist['restart_on_wake'] = True
    return {'rigid_overrides': dict(sa.rigid_overrides(args)),
            'sleep_assist': assist if sa.sleep_assist_config() is not None else None,
            'lean_step': os.environ.get('ISG_VEC_LEAN_STEP') == '1'}


def verify_driver_report(original, actual):
    require(actual.get('rush_env') == original['rush_env']
            and actual.get('driver_physics') == original['driver_physics'],
            'native CLI driver settings differ from original own replay')


def apply_native_driver(runner, args, original, source_reference, *, world_already_loaded=False):
    # This is the actual native CLI setup, not an emulation of its environment.
    runner.apply_driver_env(args)
    observed = {'rush_env': {k: v for k, v in os.environ.items() if k.startswith('ISG_')},
                'driver_physics': native_physics(runner, args)}
    verify_driver_report(original, observed)
    return {'source_report': dict(source_reference), 'setup': DRIVER_DELTA['setup'],
            'checked_before_episode_load': True,
            'checked_before_initial_world_load': not world_already_loaded, **observed}


def apply_source_fast_settings(original, declared_delta):
    """Restore measured source tunables after wrapper setup, with explicit pool deltas."""
    import math
    from ispatialgym.fastflags import fast_on, fast_value
    flags = original.get('fast', {}).get('flags')
    require(isinstance(flags, dict) and flags
            and declared_delta == {'ISPATIALGYM_FAST_COLLECTOR_PREBUILD': '1'},
            'source fast settings need measured flags and explicit pool prebuild delta')
    for key, value in flags.items():
        require(key.startswith('ISPATIALGYM_FAST_') and type(value) in (bool, int, float, str)
                and (type(value) is not float or math.isfinite(value)),
                'invalid measured source fast setting')
        os.environ[key] = ('1' if value else '0') if type(value) is bool else str(value)
    os.environ.update(declared_delta)
    observed = {}
    for key, value in flags.items():
        name = key.removeprefix('ISPATIALGYM_FAST_')
        expected = True if key in declared_delta else value
        actual = fast_on(name, default=expected) if type(expected) is bool else fast_value(name, expected, type(expected))
        require(actual == expected, 'native source fast setting differs: ' + key)
        observed[key] = actual
    return {'source_flags': dict(flags), 'declared_delta': dict(declared_delta),
            'effective_measured_flags': observed, 'checked_before_episode_load': True,
            'unrecorded_historical_settings': 'unknown'}
