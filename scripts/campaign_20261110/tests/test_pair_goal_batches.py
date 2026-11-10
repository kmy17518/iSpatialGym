import json
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ispatialgym.sampling import campaign_goal_queue as q
from scripts.campaign_20261110 import contracts as c, pair_goal_batches as batches
from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests
from scripts.campaign_20261110.tests import test_goal_queue as held


def pair_fixture(tmp_path, monkeypatch):
    queue, config_path, entries = held.fixture(tmp_path, monkeypatch)
    config = json.loads(config_path.read_text())
    config['template']['task'] = 'inter_object_relation_satisfaction-ontop_small_scale'
    c.write_json_atomic(config_path, config)
    for ordinal, row in enumerate(entries):
        plan = json.loads(open(row['path']).read())
        plan.update(schema='isg-pair-goal-plan-v1', task=config['template']['task'], target_name='target', partner_name='partner',
                    release_pose={'position': [ordinal, 0, 1.01], 'orientation': [0, 0, 0, 1]},
                    partner_pose={'position': [ordinal, 0, .8], 'orientation': [0, 0, 0, 1]})
        c.write_json_atomic(row['path'], plan)
        row['sha256'] = c.file_digest(row['path'])
    c.write_json_atomic(queue, dict(schema=q.PAIR_SCHEMA, campaign_id='new', config_sha256=c.file_digest(config_path), entries=entries))
    return queue, config_path, entries


def test_pair_queue_resets_plan_deadline_and_restores_template_cells(tmp_path, monkeypatch):
    manifest, config, _ = pair_fixture(tmp_path, monkeypatch)
    s = SimpleNamespace(place={'cells': [[0, 0], [1, 1]]}, _campaign_group_started_at=1,
                        _campaign_pair_plan_sha='old', _campaign_pair_active='old', native_cache=object())
    cache = s.native_cache
    assert q.activate_initial(s)
    assert not hasattr(s, '_campaign_pair_plan_sha')
    s.place['cells'] = [[0, 0]]
    s._campaign_group_started_at = 5
    s._campaign_pair_active = 'first'
    assert q.advance(s, 'campaign_group_timeout')
    assert s.place['cells'] == [[0, 0], [1, 1]] and s.native_cache is cache
    assert not hasattr(s, '_campaign_group_started_at') and not hasattr(s, '_campaign_pair_active')
    assert q.read_queue(manifest, config)['entries'][0]['goal_sha256'] != q.read_queue(manifest, config)['entries'][1]['goal_sha256']


def test_author_pins_three_distinct_goals_without_changing_relative_pair_geometry(tmp_path):
    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    anchor = batches.g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    target = template.objects[0]
    partner = c.ObjectRef('platter.n.01_1', 'partner', 'platter', 'b', c.digest('partner'), ready=True, readiness_source='ready fixture')
    template = replace(template, task='inter_object_relation_satisfaction-ontop_small_scale', objects=(target, partner))
    anchor['template_meta']['bindings'] = [{'name': target.instance, 'role': 'target'}, {'name': partner.instance, 'role': 'partner'}]
    bindings = anchor['tro']['ispatialgym']['object_bindings']
    bindings[target.instance]['scene_object_name'] = 'apple_a_0'
    bindings[partner.instance] = {'category': partner.category, 'model': partner.model}
    anchor['planned_start_poses'][partner.instance] = [[1, 2, 2.8], [0, 0, 0, 1]]
    anchor['summary']['goal_check'] = {'settled_poses': {'apple_a_0': [[1, 2, 3], [0, 0, 0, 1]]}}
    result = batches.prepare_batch(anchor=anchor, template=template, campaign_root=tmp_path, batch_id='pairs', count=3, seed=42,
        campaign_id='new', code_sha256=c.file_digest(code / '.snapshot_commit'), physics_sha256=c.digest('physics'), data_sha256=c.digest('maps'),goal_mode='native_region')
    state = q.read_queue(result['queue'], result['config'])
    assert len(state['entries']) == 3 and result['candidate_instances'] == 9
    assert len({row['goal_sha256'] for row in state['entries']}) == 3
    for row in state['entries']:
        plan = batches.g.read_json(row['path'])
        assert [a-b for a,b in zip(plan['goal_pose']['position'], plan['partner_pose']['position'])] == pytest.approx([0, 0, .2])
        assert plan['source_instance_is_member'] is False
    command = batches.submission_command(result['queue'], snapshot=code, campaign_root=tmp_path, v2_root=tmp_path/'maps')
    assert command[command.index('submit')+2] == '9'
    assert any(value.startswith(q.ENV+'=') for value in command)
    assert not any(value.startswith(q.PLAN_ENV+'=') for value in command)
    repeated = batches.prepare_batch(anchor=anchor, template=template, campaign_root=tmp_path, batch_id='anchor_pairs', count=3, seed=43,
        campaign_id='new', code_sha256=c.file_digest(code / '.snapshot_commit'), physics_sha256=c.digest('physics'), data_sha256=c.digest('maps'))
    repeated_state=q.read_queue(repeated['queue'],repeated['config'])
    assert len({row['goal_sha256'] for row in repeated_state['entries']}) == 1
    for row in repeated_state['entries']:
        plan=batches.g.read_json(row['path'])
        assert plan['partner_pose']['position'] == [1,2,2.8]
        assert plan['goal_pose']['position'] == [1,2,3]
        assert plan['source_instance_is_member'] is False and plan['max_members'] == 3
