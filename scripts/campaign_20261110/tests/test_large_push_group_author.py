from dataclasses import replace
import time

import pytest

from scripts.campaign_20261110 import contracts as c, goal_groups as g
from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests


@pytest.mark.parametrize('relation', ['nextto', 'touching'])
def test_large_push_author_uses_native_horizontal_endpoint_and_audited_side(tmp_path, relation):
    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    target = template.objects[0]
    partner = c.ObjectRef('table.n.01_1', 'partner', 'table', 'b', c.digest('partner'), ready=True, readiness_source='ready fixture')
    template = replace(template, task=f'inter_object_relation_satisfaction-{relation}_large_scale', objects=(target, partner))
    anchor['template_meta']['bindings'] = [{'name': target.instance, 'role': 'target'}, {'name': partner.instance, 'role': 'partner'}]
    anchor['template_meta']['template_facts'] = {'nextto': {'sides': [{'name': '-x', 'inward': [1, 0]}]}}
    bindings = anchor['tro']['ispatialgym']['object_bindings']
    bindings[target.instance]['scene_object_name'] = 'apple_a_0'
    bindings[partner.instance] = {'category': partner.category, 'model': partner.model}
    anchor['planned_start_poses'][partner.instance] = [[2, 2, 0], [0, 0, 0, 1]]
    anchor['summary']['goal_check'] = {'settled_poses': {'apple_a_0': [[1, 2, 0], [0, 0, 0, 1]]}}
    anchor['summary']['nextto'] = dict(side='-x', face='-x', direction=[1, 0, 0], goal_gap_m=.08, nextto_bound_m=.2,
        goal_pose=[[1, 2, .003], [0, 0, 0, 1]])
    kwargs = dict(anchor=anchor, template=template, campaign_id='new', code_sha256=c.file_digest(code / '.snapshot_commit'),
                  physics_sha256=c.digest('physics'), data_sha256=c.digest('maps'), magnitude_range=[0, 1])
    g.prepare_group(**kwargs, group_id='push', output=tmp_path / 'groups/push')
    plan = g.read_json(tmp_path / 'groups/push/goal_plan.json')
    assert plan['release_pose']['position'][2] == .003
    assert plan['goal_pose']['position'][2] == 0
    assert 'goal_cell' not in plan and 'place_pose' not in plan
    assert plan['base_push']['direction'] == [1, 0, 0]
    if relation == 'nextto':
        gap = plan['base_push']['goal_gap_m']
        assert .02 <= gap <= .18
        assert plan['goal_pose']['position'][0] == pytest.approx(1+.08-gap)
    else:
        assert plan['goal_pose']['position'] == [1, 2, 0]
    anchor['template_meta']['template_facts']['nextto']['sides'] = []
    with pytest.raises(c.ContractError, match='partner side geometry'):
        g.prepare_group(**kwargs, group_id='bad', output=tmp_path / 'groups/bad')


def test_surface_touching_author_keeps_release_and_settled_goal_separate(tmp_path):
    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    target = template.objects[0]
    partner = c.ObjectRef('hall_tree.n.01_1', 'partner', 'hall_tree', 'b', c.digest('partner'), ready=True, readiness_source='ready fixture')
    template = replace(template, task='inter_object_relation_satisfaction-touching_large_scale', objects=(target, partner))
    anchor['template_meta']['bindings'] = [{'name': target.instance, 'role': 'target'}, {'name': partner.instance, 'role': 'partner'}]
    bindings = anchor['tro']['ispatialgym']['object_bindings']
    bindings[target.instance]['scene_object_name'] = 'apple_a_0'
    bindings[partner.instance] = {'category': partner.category, 'model': partner.model}
    anchor['planned_start_poses'][partner.instance] = [[2, 2, 0], [0, 0, 0, 1]]
    anchor['summary']['goal_check'] = {'settled_poses': {'apple_a_0': [[1, 2, .7], [0, 0, 0, 1]]}}
    anchor['summary'].update(surface_touching={'goal_cell': [1, 2]}, goal_cell=[1, 2])
    source = anchor['sources']['sidecar']
    c.write_json_atomic(source['path'], {'grasps': [{'object': target.instance,
        'place': {'position': [1.1, 2, .78], 'orientation': [0, 0, 0, 1]},
        'eef_in_object': {'position': [.1, 0, 0], 'orientation': [0, 0, 0, 1]}}]})
    source['sha256'] = c.file_digest(source['path'])
    g.prepare_group(anchor=anchor, template=template, campaign_id='new', code_sha256=c.file_digest(code / '.snapshot_commit'),
        physics_sha256=c.digest('physics'), data_sha256=c.digest('maps'), magnitude_range=[0, 1],
        group_id='surface', output=tmp_path / 'groups/surface')
    plan = g.read_json(tmp_path / 'groups/surface/goal_plan.json')
    assert plan['placement_mode'] == 'surface'
    assert plan['release_pose']['position'] == [1, 2, .78]
    assert plan['goal_pose']['position'] == [1, 2, .7]
    assert plan['goal_cell'] == [1, 2] and 'base_push' not in plan


def test_drop_push_author_uses_audited_sidecar_place_and_native_contact_endpoint(tmp_path):
    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    anchor = g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    target = template.objects[0]
    partner = c.ObjectRef('cabinet.n.01_1', 'partner', 'cabinet', 'b', c.digest('partner'), ready=True, readiness_source='ready fixture')
    template = replace(template, task='inter_object_relation_satisfaction-touching_large_scale', objects=(target, partner))
    anchor['template_meta']['bindings'] = [{'name': target.instance, 'role': 'target'}, {'name': partner.instance, 'role': 'partner'}]
    bindings = anchor['tro']['ispatialgym']['object_bindings']
    bindings[target.instance]['scene_object_name'] = 'apple_a_0'
    bindings[partner.instance] = {'category': partner.category, 'model': partner.model}
    anchor['planned_start_poses'][partner.instance] = [[2, 2, 0], [0, 0, 0, 1]]
    anchor['planned_goal_poses'][target.instance] = [[1, 2, .003], [0, 0, 0, 1]]
    anchor['summary']['goal_check'] = {'settled_poses': {'apple_a_0': [[1.002, 2, 0], [0, 0, 0, 1]]}}
    anchor['summary'].update(push_touching={'goal_cell': [1, 2]}, goal_cell=[1, 2])
    source = anchor['sources']['sidecar']
    c.write_json_atomic(source['path'], {'grasps': [{'object': target.instance,
        'place': {'position': [.9, 2, .03], 'orientation': [0, 0, 0, 1]},
        'eef_in_object': {'position': [.1, 0, 0], 'orientation': [0, 0, 0, 1]}}]})
    source['sha256'] = c.file_digest(source['path'])
    g.prepare_group(anchor=anchor, template=template, campaign_id='new', code_sha256=c.file_digest(code / '.snapshot_commit'),
        physics_sha256=c.digest('physics'), data_sha256=c.digest('maps'), magnitude_range=[0, 1],
        group_id='floorpush', output=tmp_path / 'groups/floorpush')
    plan = g.read_json(tmp_path / 'groups/floorpush/goal_plan.json')
    assert plan['placement_mode'] == 'drop_push'
    assert plan['place_pose']['position'] == [.8, 2, .03]
    assert plan['release_pose']['position'] == [1, 2, .003]
    assert plan['goal_pose']['position'] == [1.002, 2, 0]
