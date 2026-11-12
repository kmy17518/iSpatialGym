"""Preparation requirements from actual v2 consumers, not scene feasibility.

The owner scope correction of 2026-11-12 supersedes the historical proposal
inventory's per-scene feasibility obligations. This module describes data, not
generation/replay/acceptance gates. Optional maps cannot block asset readiness.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Consumers:
    required: frozenset[str]
    optional: frozenset[str] = frozenset()
    source: str = ''
    notes: str = ''


def consumers(class_id):
    task = class_id
    if task.startswith('camera_relocalization-'):
        return Consumers(frozenset(), source='v2/tasks/camera_relocalization.py', notes='No v2 data.')
    if task.startswith(('mental_rotation-', 'object_scaling-')):
        return Consumers(frozenset({'floating_grasp_library'}), source='v2/tasks/held_object.py:check_v2_floating',
                         notes='Target model/scale/constraint/arm library only; no map or per-scene motion validation.')
    if 'articulation_' in task:
        if '_large_scale_' in task:
            return Consumers(frozenset({'articulated_handle_geometry'}), source='v2/tasks/articulation_large.py',
                             notes='Handle/scene annotation only; no articulation library or prefix map.')
        return Consumers(frozenset({'articulated_handle_geometry', 'articulated_motion_library', 'articulated_prefix_map'}),
                         source='v2/artic_prefix.py', notes='Per scaled model, joint, motion, q_start, handle arm and support anchor.')
    if task.startswith('configuration_matching-attachment_'):
        return Consumers(frozenset({'attachment_grasp_library', 'tabletop_reach_map', 'attachment_mating_geometry', 'attachment_reach_map'}),
                         source='v2/tasks/attachment.py', notes='Male entry; pair/height/rule attach map. Partner has no grasp library.')
    if task.startswith('configuration_matching-block_design_'):
        return Consumers(frozenset({'generic_reach_map'}), source='v2/tasks/block_design.py', notes='Per height/arm, object independent.')
    if task.startswith('alignment-axial_'):
        return Consumers(frozenset({'contact_push_library', 'contact_push_map', 'generic_reach_map'}),
                         frozenset({'contact_brace_library', 'contact_brace_map'}), source='v2/tasks/alignment_axial.py',
                         notes='Required right push_blade; optional left axial_brace; generic per height/arm. No grasp library.')
    if task.startswith('alignment-world_vertical_'):
        return Consumers(frozenset(), frozenset({'contact_brace_library', 'contact_brace_map'}),
                         source='v2/tasks/alignment_wall.py', notes='Optional wall_brace at hinge height/arm; robot reach volume fallback. No push library.')
    if task.startswith('disembedding_'):
        return Consumers(frozenset({'tabletop_grasp_library', 'tabletop_reach_map'}), source='v2/tasks/disembedding.py',
                         notes='Target entry and exact tray-floor anchor; other tray objects need no libraries/maps.')
    if task.startswith('inter_object_relation_satisfaction-'):
        name = task.removeprefix('inter_object_relation_satisfaction-')
        if name.startswith('nextto_large_scale_') or name == 'touching_large_scale_01':
            return Consumers(frozenset({'floor_push_contact_geometry'}), source='v2/tasks/iors_large_nextto.py',
                             notes='Base-push annotation or automatic probe; no library/map.')
        if name == 'touching_03':
            return Consumers(frozenset({'lean_grasp_library', 'interior_reach_map', 'interior_geometry'}),
                             source='v2/tasks/iors_small.py', notes='Lean rest and board/interior identity; no stable tabletop or touch contact maps.')
        if name.startswith('touching_') and 'large_scale' not in name:
            return Consumers(frozenset({'tabletop_grasp_library', 'tabletop_reach_map', 'contact_push_library',
                                       'contact_push_map', 'contact_brace_library', 'contact_brace_map'}),
                             source='v2/tasks/iors_small.py', notes='Target tabletop/right touch_push; partner left touch_brace only.')
        if name.startswith('inside_large_scale_'):
            return Consumers(frozenset({'tabletop_grasp_library', 'tabletop_reach_map', 'interior_reach_map', 'interior_geometry'}),
                             source='v2/tasks/iors_large_inside.py', notes='Resolve exact stable/overhang init and stable PLACE/lean goal rest from customs.')
        if name.startswith('touching_large_scale_'):
            required = {'tabletop_grasp_library', 'floor_reach_map'}
            if name.endswith('_02'):
                required |= {'interior_reach_map', 'interior_geometry'}
            return Consumers(frozenset(required), source='v2/tasks/iors_large_touching.py')
        if name.startswith(('under_large_scale_', 'ontop_large_scale_')):
            return Consumers(frozenset({'tabletop_grasp_library', 'tabletop_reach_map', 'floor_reach_map'}),
                             source='v2/tasks/iors_large.py', notes='Actual init floor/support and applicable goal top; no floor_push map.')
        return Consumers(frozenset({'tabletop_grasp_library', 'tabletop_reach_map'}), source='v2/tasks/iors_small.py',
                         notes='Target pipeline/rest and support anchor; partner has no grasp library.')
    raise ValueError('Unknown canonical family: ' + class_id)


RETIRED_KINDS = frozenset({'native_maximum_layout_fit', 'camera_room_navigation_and_obstacle_geometry',
                          'floating_native_motion_validation', 'full_distractor_tray_pool_geometry',
                          'floor_push_reach_map', 'grasp_and_reach_not_applicable'})

# A kind alone never grants admission. The scheduler also pins an exact reviewed
# manifest, runtime, input hashes, physical claim and consumer configuration.
ADMISSION_POLICY = {
    'schema': 'isg-consumed-preparation-admission-v1',
    'campaign_id': 'campaign_20261110',
    'manifest_sha256_required': True,
    'selected_code_consumer_required': True,
    'physical_dedup_claim_required': True,
    'allowed_computation_kinds': {
        'grasp_library': ['tabletop', 'tabletop_insert', 'flower_stem_top', 'tabletop_level_pinch', 'attachment_free', 'floating', 'lean', 'overhang'],
        'contact_library': ['touch_push', 'touch_brace', 'push_blade', 'axial_brace', 'wall_brace'],
        'reach_map': ['tabletop', 'floor', 'overhang', 'interior', 'interior_place', 'generic', 'attach', 'articulation_prefix'],
        'contact_map': ['touch_push', 'touch_brace', 'push_blade', 'axial_brace', 'wall_brace'],
        'consumed_metadata': ['exact_object_scale', 'support_height_anchor', 'tray_floor_anchor', 'interior_spec',
                              'articulated_handle_annotation', 'articulation_motion_library', 'attachment_pair_spec',
                              'base_push_contact_annotation', 'grasp_geometry_export', 'contact_frame'],
    },
    'forbidden_preparation_computation_kinds': ['per_scene_feasibility', 'camera_native_query', 'floating_native_query',
                                              'layout_feasibility', 'task_stage_validation', 'legacy_reach_band'],
    'forbidden_preparation_entrypoint': 'ispatialgym.sampling.sample_isg',
    'not_authorized_by_this_policy': ['canonical_validation', 'heldout_eval_recovery', 'pipeline_canary', 'production_generation'],
    'optional_data_not_readiness_blocker': ['axial_brace', 'wall_brace'],
    'production_authorized': False,
}


def use_kind(class_id, kind, inputs):
    """Retain only obligations with a code consumer, and label optional ones."""
    contract = consumers(class_id)
    if kind in RETIRED_KINDS:
        return None
    if kind == 'native_asset_default_geometry':
        return 'metadata' if contract.required or contract.optional else None
    if kind == 'native_support_geometry_and_height':
        maps = contract.required | contract.optional
        return 'metadata' if any('map' in k for k in maps) and not class_id.startswith('alignment-world_vertical_') else None
    if kind not in contract.required | contract.optional:
        return None
    arm = inputs.get('arm')
    if kind == 'contact_push_map' and arm not in (None, 'right'):
        return None
    if kind == 'contact_brace_map' and not class_id.startswith('alignment-world_vertical_') and arm not in (None, 'left'):
        return None
    target = inputs.get('target') or {}
    if kind in {'tabletop_grasp_library', 'attachment_grasp_library', 'floating_grasp_library'} and target.get('role', 'target') != 'target':
        return None
    return 'required' if kind in contract.required else 'optional'
