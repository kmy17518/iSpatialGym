"""Declared native consumer integration, independent of physical equivalence."""
from .contracts import require

DELTA = {
    'route': 'native_run_isg_explicit_pool_request_v1',
    'loader': 'per_instance_PoolWorld_at_named_build_and_incoming_boundaries',
    'cold_robot_anchor': 'preserve_own_loaded_native_anchor_check_exact_incoming_spawn_without_pose_write',
    'warm_robot_anchor': 'own_stopped_native_root_joint_v1_at_existing_transition',
    'warm_joint_pose_attribute_writes': 2,
    'warm_reference_state_copied': False,
    'planner': 'per_episode_native_obstacle_filter_and_enable_mask',
    'recording': 'native_Executor_and_Recorder_active_actor_consumers',
    'queue': 'existing_exclusive_claims_with_exact_request_episode_order',
    'driver': 'original_own_argv_apply_driver_env_and_saved_report_check',
    'source_authorization': 'original_gate_rechecked_before_each_episode',
    'global_class_replacement': False,
    'shared_restoration_policy_hooks': 'present_but_unselected_and_rejected_when_combined',
    'extra_physics_steps': 0,
    'physical_equivalence_established': False,
    'production_released': False,
}
PAYLOADS = {
    'ispatialgym/oracle/run_isg.py', 'ispatialgym/oracle/executor.py',
    'ispatialgym/oracle/recorder.py', 'ispatialgym/oracle/pool_consumers.py',
    'ispatialgym/oracle/pool_world.py', 'ispatialgym/oracle/pool_inputs.py',
    'ispatialgym/oracle/pool_robot_anchor.py',
    'ispatialgym/oracle/restoration_policy.py',
    'ispatialgym/oracle/restoration_policy_inputs.py',
    'scripts/campaign_20261110/training_native_pool_contract.py',
}


def verify(request, proof):
    require(request.get('native_pool_consumers_v1') is True
            and request.get('native_pool_consumer_delta') == proof.get('native_pool_consumer_delta') == DELTA
            and not request.get('incoming_pool_load_boundary_v1')
            and not request.get('incoming_pool_load_boundary')
            and not proof.get('incoming_pool_load_boundary'),
            'native pool integration requires its exact separate operation proof')
    return set(PAYLOADS)
