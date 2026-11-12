"""Explicit failure/interruption cleanup controls on the checked native pool."""
import json
from pathlib import Path

from .contracts import require, file_digest

SCHEMA = 'isg-pool-native-recovery-runtime-v1'
PARENT_COMMIT = '53b0cc9274b94e29b44ce2cadbc9269a2f7d1339'
DELTA = {'name': 'pool_native_recovery_v1', 'source_inputs_changed': False,
         'load_restore_operations_changed': False, 'thresholds_changed': False,
         'failure_boundary': 'after_one_unchanged_native_control_action',
         'exception_types': ['ControlledEpisodeFailure', 'ControlledEpisodeInterrupt'],
         'cleanup': 'ordinary_run_instance_BaseException_and_finally',
         'extra_physics_steps': 0, 'reference_state_copied': False,
         'recorded_state_scope': 'full_initialized_union_diagnostic_only'}
ALLOWED = {'scripts/campaign_20261110/training_pool_recovery_probe.py',
           'scripts/campaign_20261110/tests/test_training_pool_recovery_probe.py',
           'scripts/campaign_20261110/training_pool_replay_probe.py',
           'scripts/campaign_20261110/training_pool_probe.py'}
ORDERS = {'FRESH_A': [0], 'AA': [0, 0], 'A_FAILURE_A': [0, 0], 'A_INTERRUPT_A': [0, 0]}


class ControlledEpisodeFailure(RuntimeError):
    """An ordinary native error after one unchanged requested control action."""


class ControlledEpisodeInterrupt(KeyboardInterrupt):
    """An interruption through the ordinary native BaseException cleanup."""


def abort_after_native_control_step(env, destination, exception):
    # Self-contained: the pinned pool source predates the newer tray helper's
    # exception types. Do not depend on unshipped current-checkout helpers.
    step = env.step; original = vars(env).get('step'); had_step = 'step' in vars(env)
    def cleanup():
        if vars(env).get('step') is interrupted:
            if had_step: env.step = original
            else: del env.step
    def interrupted(*args, **kwargs):
        cleanup()
        step(*args, **kwargs)
        destination.update(native_control_calls_completed=1, physics_steps_added=0,
                           exception_type=exception.__name__, injected=True)
        raise exception('bounded restoration cleanup control after one native action')
    env.step = interrupted
    return cleanup


def parent_request(request, proof=None):
    require(request.get('pool_recovery_v1') is True and request.get('pool_action_replay_v1') is True
            and request.get('diagnostic_control') in ORDERS and request.get('load_only') is False
            and request.get('production_released') is False and request.get('benchmark_credit') == 0,
            'pool recovery requires exact bounded action scope')
    result = {**request, 'diagnostic_control': 'FRESH_A'}
    result.pop('pool_recovery_v1')
    if proof is not None:
        result.update(runtime_compatibility=proof['parent_compatibility'],
                      probe_runtime_sha256=proof['parent_runtime']['sha256'])
    return result


def verify_runtime(root, request, records, runtime, verify_parent):
    parent_request(request)
    reference = request['runtime_compatibility']; path = Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path) == reference['sha256'], 'recovery proof changed')
    proof = json.loads(path.read_text()); parent_ref = proof['parent_runtime']
    parent = Path(parent_ref['path']).resolve().parent
    require(proof.get('schema') == SCHEMA and proof.get('physical_delta') == DELTA
            and proof.get('permitted_execution_kinds') == ['physics_action_open_loop']
            and parent.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and parent_ref == {'path': str(parent/'.snapshot_commit'), 'sha256': file_digest(parent/'.snapshot_commit')}
            and (parent/'.snapshot_commit').read_text().strip() == PARENT_COMMIT
            and proof['candidate_runtime'] == {'path': str(runtime/'.snapshot_commit'),
                                               'sha256': file_digest(runtime/'.snapshot_commit')},
            'pool recovery runtime or scope differs')
    verify_parent(root, parent_request(request, proof), records, parent)
    paths = {str(p.relative_to(tree)) for tree in (parent, runtime) for folder in ('ispatialgym', 'scripts')
             for p in (tree/folder).rglob('*.py') if p.is_file()}
    changed = {p for p in paths if not (parent/p).is_file() or not (runtime/p).is_file()
               or file_digest(parent/p) != file_digest(runtime/p)}
    require(changed == set(proof['changed_payloads']) and changed <= ALLOWED
            and all((runtime/p).resolve().is_relative_to(runtime) and file_digest(runtime/p) == h
                    for p, h in proof['changed_payloads'].items()), 'unapproved recovery payload')
    return reference


def exception_type(control, ordinal):
    require(control in ORDERS and 0 <= ordinal < len(ORDERS[control]), 'recovery episode differs')
    if ordinal != 0: return None
    return {'A_FAILURE_A': ControlledEpisodeFailure, 'A_INTERRUPT_A': ControlledEpisodeInterrupt}.get(control)


def install(runner, control, current, cleanups):
    original = runner.InstanceLoader.restore
    def restored(loader, *args, **kwargs):
        result = original(loader, *args, **kwargs)
        kind = exception_type(control, current['ordinal'])
        if kind is not None:
            destination = {}
            loader.report['pool_controlled_abort'] = destination
            cleanups.append(abort_after_native_control_step(loader.env, destination, kind))
        return result
    runner.InstanceLoader.restore = restored


def verify_expected_abort(control, ordinal, code, result, report):
    kind = exception_type(control, ordinal)
    if kind is None: return False
    observed = report.get('loader', {}).get('pool_controlled_abort', {})
    require(code != 0 and result.get('success') is False
            and result.get('error', '').startswith(kind.__name__+'(')
            and observed.get('injected') is True and observed.get('exception_type') == kind.__name__
            and observed.get('native_control_calls_completed') == 1 and observed.get('physics_steps_added') == 0,
            'native pool episode did not execute exact controlled cleanup')
    return True
