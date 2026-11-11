"""Source-guided ranking must keep native candidates and all physical gates."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from ispatialgym.sampling import campaign_floor_push_proposals as P
from ispatialgym.sampling import campaign_pair_goals as G
from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge, heel_tight_couple_plan
from scripts.campaign_20261110.contracts import ContractError, file_digest
from scripts.campaign_20261110.tests.test_pair_goals import PairPlanTests


def anchor(tmp_path):
    summary = {'manipulation_stance': {'base_pose2d': [17.2801, -11.8758, 1.2535],
        'focus': [17.5877, -11.4199, .03], 'translate': .5, 'tilt': -1.2},
        'arm': {'arm': 'right'}, 'grasp': {'name': 'top_0~r180'}}
    path = tmp_path/'source.summary.json'
    path.write_text(json.dumps(summary))
    return {'summary': summary, 'sources': {'summary': {'path': str(path), 'sha256': file_digest(path)}}}


def raw_plan(a):
    raw = PairPlanTests().fixture('under')
    raw.update(task='inter_object_relation_satisfaction-under_large_scale', class_id=P.CLASS_ID,
               placement_mode='drop_push', goal_cell=[0., 0.],
               place_pose={'position': [-.3, 0., 1.], 'orientation': [0., 0., 0., 1.]},
               floor_push={'edge': '-x', 'direction': [1., 0.]}, anchor_sources=a['sources'],
               floor_push_source_proposal=P.author_hint(a, P.CLASS_ID))
    return raw


def test_author_requires_unchanged_real_source_summary(tmp_path):
    a = anchor(tmp_path)
    assert P.author_hint(a, 'inter_object_relation_satisfaction-under_large_scale_01') is None
    hint = P.author_hint(a, P.CLASS_ID)
    assert hint['arm'] == 'right' and hint['grasp_name'] == 'top_0~r180'
    a['summary']['arm']['arm'] = 'left'
    with pytest.raises(ContractError, match='differs'):
        P.author_hint(a, P.CLASS_ID)
    a['summary']['arm']['arm'] = 'right'
    (tmp_path/'source.summary.json').write_text('{}')
    with pytest.raises(ContractError, match='changed'):
        P.author_hint(a, P.CLASS_ID)


@pytest.mark.parametrize('field,value', [('arm', 'both'), ('translate', float('nan')),
    ('focus', [0., 0.]), ('source_summary_sha256', 'b'*64), ('grasp_name', '')])
def test_malformed_proposal_cannot_enter_native_plan(tmp_path, field, value):
    raw = raw_plan(anchor(tmp_path)); raw[P.FIELD][field] = value
    with pytest.raises(G.GoalPlanError):
        P.checked_hint(raw, relation='under', scale='large')


def test_hint_cannot_change_goal_or_lane_contract(tmp_path):
    raw = raw_plan(anchor(tmp_path))
    kw = dict(template_id='t', scene='s', rooms=['r'], relation='under', scale='large', activated_at=999, now=1000)
    checked = G.validate_pair_plan(raw, **kw)
    assert checked.floor_push_source_proposal['arm'] == 'right'
    old = copy.deepcopy(raw); old.pop(P.FIELD)
    previous = G.validate_pair_plan(old, **kw)
    for field in ['goal', 'release', 'place', 'partner', 'floor_push', 'max_members', 'deadline_at']:
        assert getattr(checked, field) == getattr(previous, field)
    raw['class_id'] = 'inter_object_relation_satisfaction-under_large_scale_03'
    with pytest.raises(G.GoalPlanError, match='briefcase'):
        G.validate_pair_plan(raw, **kw)
    raw['class_id'] = P.CLASS_ID
    raw['place_pose']['position'][1] += .02
    with pytest.raises(G.GoalPlanError, match='forward lane'):
        G.validate_pair_plan(raw, **kw)


def test_source_priority_only_reorders_native_pool_and_keeps_exploration(tmp_path):
    hint = P.author_hint(anchor(tmp_path), P.CLASS_ID)
    def stance(dx):
        return NS(base_pose2d=[hint['base_pose2d'][0]+dx, *hint['base_pose2d'][1:]],
                  focus=hint['focus'], translate=hint['translate'], tilt=hint['tilt'])
    candidates = [stance(.7), stance(.01), stance(.4), stance(.02)]
    sampler = IORSLarge.__new__(IORSLarge)
    sampler._campaign_pair_active = NS(floor_push_source_proposal=hint)
    sampler._campaign_source_b_draws = 0
    rng = NS(integers=lambda n: n-1)
    pool = list(candidates)
    chosen = [sampler._sm_draw_b(None, pool, rng, None) for _ in range(4)]
    assert chosen == [candidates[1], candidates[3], candidates[2], candidates[0]]
    assert set(map(id, chosen)) == set(map(id, candidates)) and not pool
    sampler._campaign_pair_active = None
    assert sampler._sm_draw_b(None, list(candidates), rng, None) is candidates[-1]
    pre = [(1, c, [], {}) for c in candidates]
    ranked = P.ranked_prescreen(pre, hint)
    assert [r[1] for r in ranked] == [candidates[1], candidates[3], candidates[0], candidates[2]]
    assert P.ranked_prescreen(pre, None) is pre


def test_source_arm_priority_never_discards_other_checked_candidates():
    arms = [['left'], ['left', 'right'], ['right'], []]
    order, _ = heel_tight_couple_plan(arms, 'right', lambda i, a: True)
    assert order[:2] == [(1, ['right']), (2, ['right'])]
    assert sorted((i, a) for i, offered in order for a in offered) == sorted(
        (i, a) for i, offered in enumerate(arms) for a in offered)

