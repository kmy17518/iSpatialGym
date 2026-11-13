"""Bounded exact-source multi-active pool replay using the shared pool operations.

Original generation/collection identities are immutable. This new source proof
binds both completed canaries to a separately inventoried execution snapshot;
it establishes no physical equivalence, rendering validity or release.
"""
import json
from pathlib import Path

from .contracts import require, file_digest
from .training_pool_world import checked, prepare, active_actor_names
from .training_native_driver import DRIVER_DELTA, canary_driver

SCHEMA = 'isg-multi-active-source-exact-replay-request-v1'
PROOF_SCHEMA = 'isg-multi-active-source-exact-runtime-v1'
ORDERS = {'FRESH_A': [0], 'FRESH_B': [1], 'ABA': [0, 1, 0]}
MODULES = {
    'selected_pool_episode_probe', 'training_multi_active_sources',
    'training_pool_source_compatibility', 'training_pool_world', 'training_pool_adapter',
    'training_pool_task', 'training_pool_consumers', 'training_pool_probe',
    'training_pool_source_anchor_probe', 'training_pool_native_anchor_probe',
    'training_pool_replay_probe', 'training_pool_recording_probe',
    'training_native_driver', 'training_multi_active_replay',
    'training_canary_scope', 'training_pool_load_boundary',
}
ALLOWED = {'scripts/campaign_20261110/' + name + '.py' for name in MODULES}
DELTA = {
    'initialization': 'existing_exact_model_scale_actor_union',
    'reuse': 'existing_source_anchor_then_native_cleanup_and_incoming_task_restore',
    'task_switch': 'exact_reviewed_source_BDDL_roles_and_support_mapping',
    'actions': 'unaltered_own_source_demo_open_loop',
    'driver': DRIVER_DELTA,
    'fast_settings': 'all_measured_original_flags_after_wrapper_with_explicit_pool_prebuild_delta',
    'recording': 'existing_active_only_native_state_scene_and_configuration',
    'physics_operations_implementation': 'shared_existing_pool_installers',
    'extra_physics_steps': 0,
    'thresholds_changed': False,
    'source_files_changed': False,
    'physical_equivalence_established': False,
}


def reference(path):
    return {'path': str(path), 'sha256': file_digest(path)}


def scope(request):
    native = request.get('native_pool_consumers_v1', False)
    require(type(native) is bool, 'native pool consumer selector must be boolean')
    if native:
        from .training_native_pool_contract import DELTA as NATIVE_DELTA
        require(request.get('native_pool_consumer_delta') == NATIVE_DELTA,
                'native pool consumer operation declaration differs')
    else:
        require('native_pool_consumer_delta' not in request, 'unused native pool consumer declaration')
    require(request.get('schema') == SCHEMA and request.get('multi_active_replay_v1') is True
            and request.get('native_cli_driver_v1') is True
            and request.get('pool_active_recording_v1') is True
            and request.get('pool_action_replay_v1') is True
            and request.get('load_only') is False
            and request.get('production_released') is False
            and type(request.get('benchmark_credit')) is int and request['benchmark_credit'] == 0
            and request.get('diagnostic_control') in ORDERS
            and request.get('physical_delta') == DELTA,
            'multi-active replay requires explicit exact-source bounded operations')
    order = ORDERS[request['diagnostic_control']]
    require(request.get('max_native_episodes') == len(order)
            and type(request.get('timeout_seconds')) is int
            and 0 < request['timeout_seconds'] <= (3600 if len(order) == 3 else 1800)
            and request.get('source_environment_delta') == {'ISPATIALGYM_FAST_COLLECTOR_PREBUILD': '1'}
            and not any(request.get(k) for k in ('integrated_candidate_v1', 'physical_candidate',
                                                'sequential_augmentation', 'pool_recovery_v1')),
            'multi-active replay order, budget or undeclared operation differs')
    return order


def authorize_member(root, request, episode, record):
    """Recheck the original exact canary gate before entering its native load."""
    from .production_gate import check_job
    review = checked(root, episode['review'])
    require(review['export_record'] == record and review['validation_passed'] is True
            and review['accepted'] is False and review['eligible_for_benchmark'] is False
            and review['quota_consumed'] == 0, 'multi-active source review differs')
    decision = check_job(root, Path(review['source_job']), kind='source_validation')
    require(decision.get('allowed') is True and decision.get('category') == 'validation_canary'
            and decision.get('config_sha256') == episode['configuration']['sha256'],
            'multi-active original source authorization failed: ' + json.dumps(decision, sort_keys=True))
    raw = checked(root, record['artifacts']['tro'])['ispatialgym']
    config = checked(root, episode['configuration'])
    purpose = raw['campaign']['runtime']['purpose']
    require(purpose == config['purpose'] and purpose in ('training_pipeline_canary', 'training_production'),
            'multi-active original configuration purpose differs')
    from .training_canary_scope import validate_scope
    authorization = decision['authorization']
    require(all(authorization.get(k) == review['authorization'].get(k) for k in
                ('purpose', 'config_sha256', 'start', 'count', 'authorization_path', 'authorization_sha256')),
            'multi-active original bounded authorization differs from source review')
    declared = checked(root, {'path': authorization['authorization_path'],
                              'sha256': authorization['authorization_sha256']})
    validate_scope(config, declared)
    return decision


def inputs(root, request_path, runtime):
    from .selected_pool_episode_probe import inputs as source_inputs
    from .training_integrated_runtime import payloads
    root = Path(root).resolve(); runtime = Path(runtime).resolve()
    path = Path(request_path).resolve()
    require(path.is_relative_to(root), 'multi-active request escaped campaign')
    q = json.loads(path.read_text()); scope(q)
    require(q['campaign_id'] == root.name == 'campaign_20261110', 'wrong multi-active campaign')
    package = checked(root, q['source_package'])
    require(package.get('schema') == 'isg-measured-two-active-source-package-v1'
            and package.get('normal_native_demo_own_qa_chains_verified') is True
            and package.get('source_original_purposes_unchanged') is True
            and package.get('production_released') is False and package.get('benchmark_credit') == 0
            and package['source_sequence'] == q['source_sequence'], 'measured multi-active package differs')
    source = checked(root, q['source_sequence'])
    actual_source, all_records = source_inputs(q['source_sequence']['path'])
    require(actual_source == source and package['native_runtime'] == source['native_runtime']
            and package['active_role_contract'] == source['active_role_contract'],
            'multi-active original source or role contract differs')
    records = all_records[:2]
    require(q['canaries'] == [e['review'] for e in source['episodes'][:2]], 'multi-active canaries changed')
    for i, (member, episode, record) in enumerate(zip(package['members'], source['episodes'], records)):
        require(member['member'] == ('A', 'B')[i] and member['review'] == episode['review']
                and member['artifacts'] == record['artifacts']
                and member['source_own_report'] == episode['original_cli_driver_report'],
                'multi-active member package changed')
        driver_source, original, binding = canary_driver(root, q, i)
        require(binding['original_own_report'] == episode['original_cli_driver_report'],
                'multi-active driver is not its original own replay')
        authorize_member(root, q, episode, record)
    contract = prepare(root, q['source_sequence']['path'])
    require(contract == checked(root, q['pool_contract'])
            and all(len(active_actor_names(m['active'])) > 1 for m in contract['members'])
            and len(contract['initialized_actors']) >= 4, 'multi-active initialized union changed')
    proof = checked(root, q['runtime_compatibility'])
    base = Path(source['native_runtime']['path']).resolve().parent
    require(runtime.is_relative_to(root/'code') and base.is_relative_to(root/'code')
            and reference(base/'.snapshot_commit') == source['native_runtime']
            and reference(runtime/'.snapshot_commit') == q['native_runtime']
            and q['probe_runtime_sha256'] == q['native_runtime']['sha256']
            and proof.get('schema') == PROOF_SCHEMA and proof['source_package'] == q['source_package']
            and proof['source_sequence'] == q['source_sequence'] and proof['pool_contract'] == q['pool_contract']
            and proof['source_runtime'] == source['native_runtime']
            and proof['candidate_runtime'] == q['native_runtime']
            and proof['physical_delta'] == DELTA and proof['physical_equivalence_established'] is False
            and proof['production_released'] is False and proof['benchmark_credit'] == 0,
            'multi-active candidate identity or operation proof differs')
    before, after = payloads(base), payloads(runtime)
    changes = {p: {'source': before.get(p), 'candidate': after.get(p)}
               for p in set(before) | set(after) if before.get(p) != after.get(p)}
    allowed = set(ALLOWED)
    if q.get('native_pool_consumers_v1') is True:
        from .training_native_pool_contract import verify as verify_native_pool
        allowed.update(verify_native_pool(q, proof))
    elif q.get('incoming_pool_load_boundary_v1') is True:
        from .training_pool_load_boundary import verify_runtime as verify_boundary
        allowed.update(verify_boundary(q, proof, base, runtime))
    else:
        require(not q.get('incoming_pool_load_boundary') and not proof.get('incoming_pool_load_boundary'),
                'undeclared incoming pool boundary')
    require(after == proof['candidate_python_payloads'] and changes == proof['changed_payloads']
            and set(changes) <= allowed, 'multi-active runtime contains unreviewed physical changes')
    setter = proof['native_robot_setter']
    require(Path(setter['path']).is_absolute() and file_digest(setter['path']) == setter['sha256'],
            'multi-active original native anchor setter changed')
    return q, records, source, contract
