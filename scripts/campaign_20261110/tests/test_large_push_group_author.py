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
