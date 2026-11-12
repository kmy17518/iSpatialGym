from scripts.campaign_20261110.preparation_consumers import consumers, use_kind
from scripts.campaign_20261110.reconcile_preparation_scope import project


def test_camera_and_floating_have_no_scene_computation_dependencies():
    assert not consumers('camera_relocalization-standard').required
    assert consumers('mental_rotation-mug').required == {'floating_grasp_library'}
    assert use_kind('object_scaling-apple', 'floating_native_motion_validation', {}) is None
    assert use_kind('camera_relocalization-obstructed-hamper_1', 'native_asset_default_geometry', {}) is None


def test_optional_alignment_data_and_exact_arms():
    assert use_kind('alignment-axial_01', 'contact_brace_map', {'arm': 'left'}) == 'optional'
    assert use_kind('alignment-axial_01', 'contact_push_map', {'arm': 'left'}) is None
    assert use_kind('alignment-world_vertical_01', 'contact_push_library', {}) is None
    assert use_kind('alignment-world_vertical_01', 'contact_brace_map', {'arm': 'right'}) == 'optional'


def test_special_consumers_do_not_inherit_generic_wrong_data():
    assert consumers('configuration_matching-block_design_01').required == {'generic_reach_map'}
    assert consumers('configuration_matching-articulation_open_large_scale_01').required == {'articulated_handle_geometry'}
    assert use_kind('inter_object_relation_satisfaction-under_large_scale_01', 'floor_push_reach_map', {}) is None
    assert use_kind('inter_object_relation_satisfaction-touching_03', 'tabletop_grasp_library', {}) is None
    assert use_kind('inter_object_relation_satisfaction-nextto_01', 'tabletop_grasp_library', {'target': {'role': 'partner'}}) is None


def test_projection_retires_feasibility_without_promoting_asset_evidence():
    config = {'configuration_id': 'x', 'class_id': 'object_scaling-apple', 'exact_scales_resolved': True}
    graph = {'rows': [{'id': 'lib', 'kind': 'floating_grasp_library', 'inputs': {}, 'consumers': ['x'], 'status': 'failed'},
                      {'id': 'query', 'kind': 'floating_native_motion_validation', 'inputs': {}, 'consumers': ['x'], 'status': 'verified'}]}
    result = project(graph, {'rows': [config]})
    assert [r['id'] for r in result['rows']] == ['lib']
    assert result['rows'][0]['status'] == 'failed'
    assert result['retired_requirements'][0]['historical_status'] == 'verified'
    assert not result['configuration_requirements'][0]['per_scene_feasibility_required']
