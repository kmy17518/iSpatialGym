import copy
import json
from types import SimpleNamespace as N

import pytest

from ispatialgym.sampling.prepared_articulation import bind_generation, verify_inputs, verify_selected
from ispatialgym.sampling.selected_context import construction_plan
from scripts.campaign_20261110.contracts import digest
from scripts.campaign_20261110.tests import test_prepared_articulation as prepared_tests

case = prepared_tests.case
document = prepared_tests.document


@pytest.fixture
def instance_case(case):
    scene = document(case.tmp / 'house_single_floor_stable.json', {'objects_info': {'init_info': {
        case.obj.name: {'args': {'category': 'door', 'model': 'model', 'scale': [1., 1., 1.]}}}}})
    selection = json.loads(open(case.options['selection_ref']['path']).read())
    target = selection['rows'][0]['roles'][0]
    target.update(scene_matching_instances=[case.obj.name], scene_scale_evidence=scene)
    selection_ref = document(case.tmp / 'instance_selection.json', selection)
    annotation = document(case.tmp / 'instance_handle.json', {'scene': case.sampler.scene,
        'name': case.obj.name, 'category': 'door', 'model': 'model', 'joints': [{
        'joint': 'j_leaf', 'type': 'RevoluteJoint', 'handles': [case.handle],
        'axis_dir_obj': [0., 1., 0.], 'hinge_point_obj': [0., 0., 0.]}]})
    row = copy.deepcopy(case.row)
    row.update(scale_row_sha256=digest(selection['rows'][0]), handle_record=annotation,
        handle_frame='instance', scene_actor_binding={'name': case.obj.name, 'scene': case.sampler.scene,
        'source': scene, 'source_pointer': '/objects_info/init_info/' + case.obj.name + '/args'},
        native_selected_handle={**case.handle, 'source': 'record:knob'})
    row['role_options'].update(handle_file=annotation['path'], handle_row='record:knob')
    metadata = document(case.tmp / 'instance_metadata.json', {
        'schema': 'isg-exact-articulation-handle-binding-v1', 'source': selection_ref, 'rows': [row]})
    case.options.update(metadata_ref=metadata, selection_ref=selection_ref)
    case.entry['scenes'][case.sampler.scene]['bindings'] = {'target': [case.obj.name]}
    case.bound = bind_generation(case.entry, **case.options)
    case.sampler.gp.entry = case.bound
    case.sampler.world = N(stable_path=scene['path'])
    case.row, case.annotation, case.scene_ref = row, annotation, scene
    return case


def test_actual_instance_load_handle_retains_identity_and_frame(instance_case, monkeypatch):
    from ispatialgym.sampling.v2.tasks import articulation_large as al
    c = instance_case

    class Sampler(al.ArticulationLarge):
        @property
        def target(self):
            return c.obj

    sampler = Sampler.__new__(Sampler)
    sampler.gp = N(entry=c.bound, variant='alias_A', gen=lambda key, default=None: default)
    sampler.target_role = 'target'
    sampler.scene = 'house_single_floor'
    sampler.world = N(scene_model=sampler.scene, stable_path=c.scene_ref['path'])
    sampler.template_facts = {}
    c.obj.links = {}
    monkeypatch.setattr(al, 'HANDLES_CSV', str(c.csv))
    sampler.load_handle(c.obj)
    lookup = sampler.template_facts['prepared_handle_lookup']
    assert lookup['actor']['name'] == c.obj.name
    assert lookup['selected_handle']['contact'] == c.handle['contact']
    assert lookup['motion_readiness_proven'] is False


@pytest.mark.parametrize('mutation', ['actor', 'world', 'annotation', 'source'])
def test_instance_handle_cannot_transfer_to_another_actor_or_context(instance_case, mutation):
    c = instance_case
    if mutation == 'actor': c.obj.name = 'same_model_another_actor'
    if mutation == 'world': c.sampler.world.stable_path = document(c.tmp / 'other.json', {})['path']
    if mutation == 'annotation': (c.tmp / 'instance_handle.json').write_text('{}')
    if mutation == 'source': (c.tmp / 'house_single_floor_stable.json').write_text('{}')
    with pytest.raises(ValueError): verify_inputs(c.sampler, c.obj, handles_csv=c.csv)


def test_generation_and_construction_cannot_move_an_instance_annotation(instance_case):
    c = instance_case
    entry = copy.deepcopy(c.entry)
    entry['scenes']['house_single_floor']['bindings']['target'] = ['another']
    with pytest.raises(ValueError, match='names another actor'):
        bind_generation(entry, **c.options)
    with pytest.raises(ValueError, match='cannot be transferred'):
        construction_plan(**c.options, scene_ref=c.scene_ref, numeric_ref={}, prototype_name='x',
                          floor_name='floor', room='garden_0', yaw_radians=[0.])
    prepared = verify_inputs(c.sampler, c.obj, handles_csv=c.csv)
    with pytest.raises(ValueError, match='another annotation or joint'):
        verify_selected(prepared, handle=c.handle, source='record:knob', path=c.annotation['path'],
                        frame='canonical', joint_name='j_leaf', width_range=[.003, .08])
