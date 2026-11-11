import json
import pytest

from scripts.campaign_20261110.contracts import ContractError, file_digest
from scripts.campaign_20261110.training_scene_policy import validate_removal_scope
from scripts.campaign_20261110.production_gate import check_environment


def test_removal_gate_uses_actual_bound_categories_and_complete_task_bindings(tmp_path):
    scene = tmp_path/'scene.json'
    objects = {name: {'args': {'category': category}} for name, category in
               [('building_part', 'fixed_window'), ('panel', 'door'), ('support', 'table'), ('clutter', 'chair')]}
    objects['robot'] = {'args': {'category': 'agent', 'type': 'TiagoRobot'}}
    scene.write_text(json.dumps({'objects_info': {'init_info': objects}}))
    config = {'template': {'split': 'train', 'task': 'object_scaling', 'class_id': 'example',
        'objects': [{'instance': 'support.n.01_1'}], 'scene_setup': {
            'inst_to_name': {'support.n.01_1': 'support'},
            'augmentation_scene': {'path': str(scene), 'sha256': file_digest(scene)}}},
        'perturbations': {'removed_objects': ['clutter']}}
    assert validate_removal_scope(tmp_path, config)['removed_objects'] == ['clutter']
    path = tmp_path/'config.json'
    for name, message in [('panel', 'structural'), ('building_part', 'structural'),
                          ('support', 'task-relevant'), ('robot', 'robot'), ('missing', 'unknown')]:
        config['perturbations']['removed_objects'] = [name]
        with pytest.raises(ContractError, match=message): validate_removal_scope(tmp_path, config)
        path.write_text(json.dumps(config))
        result = check_environment(tmp_path, {'ISPATIALGYM_CAMPAIGN_CONFIG': str(path)}, start=1, count=1)
        assert not result['allowed'] and message in result['reason']
    config['perturbations']['removed_objects'] = ['clutter']
    config['template']['scene_setup']['inst_to_name'] = {}
    with pytest.raises(ContractError, match='bindings'): validate_removal_scope(tmp_path, config)


def test_removal_scope_fails_closed_on_missing_or_changed_scene(tmp_path):
    config = {'template': {'scene_setup': {}, 'objects': []},
              'perturbations': {'removed_objects': ['chair']}}
    with pytest.raises(ContractError, match='hash-bound'): validate_removal_scope(tmp_path, config)
    scene = tmp_path/'scene.json';scene.write_text('{}')
    config['template']['scene_setup']['augmentation_scene'] = {'path': str(scene), 'sha256': 'wrong'}
    with pytest.raises(ContractError, match='source changed'): validate_removal_scope(tmp_path, config)
    config['perturbations']['removed_objects'] = []
    assert validate_removal_scope(tmp_path, config) is None
