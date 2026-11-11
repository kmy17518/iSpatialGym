from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.campaign_held_goals import ENV
from ispatialgym.sampling.campaign_pair_goals import PairGoalPlan
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2.tasks import iors_small as I


def plan(relation="ontop"):
    return PairGoalPlan("fresh", "g", "train", "t", "a" * 64, relation, "target", "partner",
                        ((0., 0., .81), (0., 0., 0., 1.)), ((0., 0., .8), (0., 0., 0., 1.)),
                        ((0., 0., .7), (0., 0., 0., 1.)), 1000., 3, "b" * 64)


def test_disabled_pair_plan_never_accesses_runtime_context(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert I.IORSSmall.campaign_pair_plan(SimpleNamespace()) is None


def test_inside_stance_is_remembered_only_after_native_write_and_scoped_to_goal(monkeypatch):
    s=I.IORSSmall.__new__(I.IORSSmall)
    s._campaign_pair_active=plan('inside');s.relation='inside'
    s.campaign_verify_pair=lambda draft:None
    d=Draft();d.summary={'campaign_goal_group':{},'manipulation_stance':{'base_pose2d':[1.,2.,0.],'translate':.5,'tilt':0.}}
    def failed(*args):raise Reject('physical_check','failed before write')
    monkeypatch.setattr(I.SamplerV2,'write_instance',failed)
    with pytest.raises(Reject):s.write_instance(d,1,2)
    assert not hasattr(s,'_campaign_written_inside_stances')
    monkeypatch.setattr(I.SamplerV2,'write_instance',lambda *args:'written')
    assert s.write_instance(d,1,2)=='written'
    assert s._campaign_written_inside_stances[s._campaign_pair_active.sha256] == [d.summary['manipulation_stance']]
    assert s._campaign_written_inside_stances.get('another_goal',[]) == []


def test_campaign_frame_ranking_uses_fixed_release_instead_of_random_focus():
    p=plan('inside');observed=[];candidate=SimpleNamespace(focus=None)
    s=SimpleNamespace(_campaign_pair_active=p,camera_pose_at=lambda c:'camera',
                      ikc=SimpleNamespace(eef_frame_margin=lambda pose,camera:(observed.append((pose,camera)) or 4.)))
    assert I.IORSSmall.stance_frame_aim(s,candidate)==4.
    assert observed[0][0][0].tolist()==pytest.approx(p.release[0])
    assert observed[0][1]=='camera'


def test_pair_start_keeps_full_yaw_range_independent_of_goal_alignment(monkeypatch):
    calls = []
    monkeypatch.setattr(I.TT, "phase3_target", lambda *a, **kw: (calls.append(kw) or (None, SimpleNamespace(grasp_name="g"))))
    s = SimpleNamespace(_campaign_pair_active=plan(), relation="ontop", grasps_flush=[], grasps=[],
                        neighbor_start_top=lambda: None, pair_goal_align=lambda: {"relative_yaw_deg": 0},
                        pair_target_yaws=lambda manip: pytest.fail("start yaw must not inherit goal yaw"),
                        hull={"target": None}, rest={"target": None}, support_objects=lambda: [], top=None,
                        choice=SimpleNamespace(models={"target": SimpleNamespace(scale=None)}))
    I.IORSSmall.phase3_two_tier(s, None, np.random.default_rng(1))
    assert "yaws" not in calls[0] and "jitter" not in calls[0]


def test_fixed_partner_runs_geometry_checks(monkeypatch):
    monkeypatch.setattr(I.TT, "partner_pose_why", lambda *a, **kw: "partner footprint off the top")
    s = SimpleNamespace(_campaign_pair_active=plan(), world=SimpleNamespace(footprint=lambda *a: None),
                        hull={"target": None, "partner": None}, top=None, margin_support=.02, gap_min=.03)
    with pytest.raises(Reject, match="off the top"):
        I.IORSSmall.place_partner(s, None, None, None, None)


def test_inside_goal_orientation_is_unconstrained_and_metadata_goal_exact():
    p = plan("inside")
    s = I.IORSSmall.__new__(I.IORSSmall)
    s._campaign_pair_active = p
    s.campaign_group_remaining_s = lambda: 100
    s.tol = lambda key, default: default
    s.gen = lambda key, default=None: default
    s.objects = {"partner": SimpleNamespace(get_position_orientation=lambda: tuple(th.tensor(v) for v in p.partner)),
                 "target": SimpleNamespace(get_position_orientation=lambda: (th.tensor([.002, 0., .8]), th.tensor([1., 0., 0., 0.])))}
    d = Draft()
    d.summary["campaign_goal_group"] = p.summary()
    d.pose_targets = {"target": {"orientation_metric": "full"}, "partner": {"orientation_metric": "full"}}
    s.campaign_verify_pair(d, settled=True)
    assert d.pose_targets["target"]["orientation_metric"] is None
    assert d.pose_targets["target"]["position"] == list(p.goal[0])
    assert d.summary["campaign_goal_group"]["settled_goal_residual"]["position_m"] > 0


def test_inside_release_uses_native_open_jaw_clearance_for_each_grasp(monkeypatch):
    p = replace(plan('inside'), release_policy='native_grasp_height',
                release=((0., 0., .77), (0., 0., 0., 1.)))
    hull = th.tensor([[x, y, z] for x in (-.01, .01) for y in (-.01, .01) for z in (-.01, .01)])
    partner_hull = th.tensor([[x, y, z] for x in (-.2, .2) for y in (-.2, .2) for z in (0., .7)])
    grasp = SimpleNamespace(eef_in_object=(th.tensor([0., 0., 0.]), th.tensor([0., 0., 0., 1.])))
    monkeypatch.setattr(I, 'gripper_corners_world', lambda eef, **kw: th.stack([eef[0]-th.tensor([0., 0., .05])]))
    s = SimpleNamespace(_campaign_pair_active=p, relation='inside',
        sidecar=SimpleNamespace(stance=lambda name: None, grasp_of=lambda name: grasp),
        target_goal_quat=lambda pose, *args: I.G.t(pose[1]),
        hull={'target': hull, 'partner': partner_hull},
        world=SimpleNamespace(hull_at=lambda h, pose: I.G.local_points_world(h, pose)),
        bindings=SimpleNamespace(one=lambda role: SimpleNamespace(name=role)),
        obj=lambda name: None, _container_volume=lambda *args: (th.zeros(3), .7),
        gen=lambda key, default=None: default, up_axis=lambda: None, standing=None,
        ikc=SimpleNamespace(WITNESS_POS_TOL=.005, WITNESS_ANG_TOL=.05))
    s.release_pose = lambda *args: I.IORSSmall.release_pose(s, *args)
    first = I.IORSSmall.goal_pose(s, p.goal, p.partner)
    assert float(first[0][2]) == pytest.approx(.77)
    assert s.release['finger_clearance_m'] == pytest.approx(.02)
    grasp.eef_in_object = (th.tensor([0., 0., -.03]), grasp.eef_in_object[1])
    second = I.IORSSmall.goal_pose(s, p.goal, p.partner)
    assert float(second[0][2]) == pytest.approx(.80)
    assert s.release['finger_clearance_m'] == pytest.approx(.02)
    # The previous fixed-release policy rejects this legitimate per-grasp height.
    s._campaign_pair_active = replace(p, release_policy='fixed')
    with pytest.raises(Reject, match='release differs'):
        I.IORSSmall.goal_pose(s, p.goal, p.partner)
    # A different horizontal endpoint is still rejected with the new policy.
    s._campaign_pair_active = p
    s._container_volume = lambda *args: (th.tensor([.02, 0., 0.]), .7)
    with pytest.raises(Reject, match='release differs'):
        I.IORSSmall.goal_pose(s, p.goal, p.partner)


def test_verified_native_inside_release_is_preserved_but_settled_goal_remains_fixed():
    p = replace(plan('inside'), release_policy='native_grasp_height')
    s = I.IORSSmall.__new__(I.IORSSmall)
    s._campaign_pair_active = p
    s.campaign_group_remaining_s = lambda: 100
    s.tol = lambda key, default: default
    s.gen = lambda key, default=None: default
    s.release = {'finger_clearance_m': .02, 'place_kind': 'drop'}
    s.objects = {'partner': SimpleNamespace(get_position_orientation=lambda: p.partner),
                 'target': SimpleNamespace(get_position_orientation=lambda: p.goal)}
    d = Draft()
    d.summary['campaign_goal_group'] = p.summary()
    d.pose_targets = {'target': {}, 'partner': {}}
    d.goal_poses = {'target': (th.tensor([0., 0., .9]), th.tensor([0., 0., 0., 1.]))}
    s.campaign_verify_pair(d, settled=True)
    assert float(d.goal_poses['target'][0][2]) == pytest.approx(.9)
    assert d.pose_targets['target']['position'] == list(p.goal[0])
    assert d.summary['campaign_goal_group']['release_geometry']['finger_clearance_m'] == .02


@pytest.mark.parametrize("relation", ["ontop", "under"])
def test_surface_release_geometry_uses_release_attitude_not_settled_tilt(relation):
    import math
    hull = th.tensor([[x, y, z] for x in (-.1, .1) for y in (-.1, .1) for z in (-.05, .05)])
    partner_hull = th.tensor([[x, y, z] for x in (-.2, .2) for y in (-.2, .2) for z in (-.01, 0.)])
    p = replace(plan(relation),
                release=((0., 0., .753), (0., 0., 0., 1.)),
                goal=((0., 0., .75), (math.sin(.15), 0., 0., math.cos(.15))))
    s = SimpleNamespace(_campaign_pair_active=p, relation=relation,
        sidecar=SimpleNamespace(stance=lambda name: None),
        target_goal_quat=lambda pose, *args: I.G.t(pose[1]),
        hull={'target': hull, 'partner': partner_hull},
        world=SimpleNamespace(hull_at=lambda h, pose: I.G.local_points_world(h, pose)),
        partner_aligned_to_target=lambda: False,
        gen=lambda key, default=None: default,
        release_surface=lambda xy, pose, top: {'z': .7},
        ikc=SimpleNamespace(WITNESS_POS_TOL=.005, WITNESS_ANG_TOL=.05))
    actual = I.IORSSmall.goal_pose(s, p.goal, p.partner)
    assert th.allclose(actual[0], th.tensor(p.release[0]))
    assert th.allclose(actual[1], th.tensor(p.release[1]))
    # A release that violates native surface clearance is still rejected.
    s._campaign_pair_active = replace(p, release=((0., 0., .78), p.release[1]))
    with pytest.raises(Reject, match='release differs'):
        I.IORSSmall.goal_pose(s, p.goal, p.partner)
    # Without a campaign plan the existing caller-provided orientation remains.
    s._campaign_pair_active = None
    legacy = I.IORSSmall.goal_pose(s, p.goal, p.partner)
    assert th.allclose(legacy[1], th.tensor(p.goal[1]))
    assert float(legacy[0][2]) > p.release[0][2] + .02


def test_large_goal_is_pinned_to_supported_cell_before_starts(monkeypatch):
    from dataclasses import replace
    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge
    p = replace(plan(), scale="large", goal_cell=(.1, .2))
    s = SimpleNamespace(campaign_pair_plan=lambda: p, tname="target", pname="partner",
                        partner=SimpleNamespace(get_position_orientation=lambda: p.partner),
                        num=lambda key, default: default, tol=lambda key, default: default,
                        place={"cells": th.tensor([[.1, .2], [.5, .6]])})
    IORSLarge.campaign_pin_goal(s)
    assert s.place["cells"].shape == (1, 2)
    assert s._campaign_latest_proposal["goal_poses"]["target"][0] == list(p.goal[0])
    s.place["cells"] = th.tensor([[2., 2.]])
    with pytest.raises(Reject, match="supported placement region"):
        IORSLarge.campaign_pin_goal(s)
    monkeypatch.delenv(ENV, raising=False)
    assert IORSLarge.campaign_pair_plan(SimpleNamespace()) is None


def test_large_inside_keeps_orientation_and_exact_goal_after_native_settle():
    from dataclasses import replace
    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge
    p = replace(plan("inside"), scale="large", goal_cell=(0., 0.))
    s = SimpleNamespace()
    s.campaign_verify_pair = lambda draft, **kwargs: IORSLarge.campaign_verify_pair(s, draft, **kwargs)
    s._campaign_pair_active = p
    s.campaign_group_remaining_s = lambda: 100
    s.tol = s.num = lambda key, default: default
    s.tname, s.pname = "target", "partner"
    s.objects = {"partner": SimpleNamespace(get_position_orientation=lambda: p.partner),
                 "target": SimpleNamespace(get_position_orientation=lambda: ((.002, 0., .8), p.goal[1]))}
    s.target, s.partner = s.objects["target"], s.objects["partner"]
    d = Draft()
    d.summary["campaign_goal_group"] = p.summary()
    d.pose_targets = {"target": {"orientation_metric": "full"}}
    s.campaign_verify_pair(d, settled=True)
    assert d.pose_targets["target"]["position"] == list(p.goal[0])
    assert d.pose_targets["target"]["orientation"] == list(p.goal[1])
    s.objects["target"].get_position_orientation = lambda: (p.goal[0], (1., 0., 0., 0.))
    with pytest.raises(Reject, match="settled pair target drifted"):
        s.campaign_verify_pair(d, settled=True)


def test_touching_place_stays_fixed_and_rejects_an_incompatible_push_direction():
    from dataclasses import replace
    p = replace(plan("touching"), place=((0., -.08, .8), (0., 0., 0., 1.)))
    s = SimpleNamespace(_campaign_pair_active=p, relation="touching", place_gap=lambda: .08,
        ikc=SimpleNamespace(WITNESS_POS_TOL=.005, WITNESS_ANG_TOL=.05),
        top=SimpleNamespace(polygon_supported=lambda *args, **kwargs: True), margin_support=.02,
        world=SimpleNamespace(footprint=lambda *args: None), hull={"target": None})
    stance = SimpleNamespace(tangent=th.tensor([0., 1.]))
    start = (th.tensor([.5, -.5, .8]), th.tensor([0., 0., 1., 0.]))
    actual = I.IORSSmall.place_pose_for(s, start, p.partner, stance)
    assert th.allclose(actual[0], th.tensor(p.place[0]))
    assert th.allclose(actual[1], th.tensor(p.place[1]))
    stance.tangent = th.tensor([1., 0.])
    with pytest.raises(Reject, match="push direction"):
        I.IORSSmall.place_pose_for(s, start, p.partner, stance)
    stance.tangent = th.tensor([0., 1.])
    s.top.polygon_supported = lambda *args, **kwargs: False
    with pytest.raises(Reject, match="leaves the support"):
        I.IORSSmall.place_pose_for(s, start, p.partner, stance)


def test_floor_touching_fixed_chain_still_runs_native_solver_and_rejects_other_endpoints():
    from dataclasses import replace
    from ispatialgym.sampling.v2.tasks.iors_large_touching import TouchingMode
    calls = []
    class NativeChain:
        def _push_under_chain(self, *args):
            calls.append(args)
            return {'native_checked': True}
    class Sampler(TouchingMode, NativeChain):
        pass
    p = replace(plan('touching'), scale='large', placement_mode='drop_push',
        release=((1., 2., .003), (0., 0., 0., 1.)), goal_cell=(1., 2.), place=((.8, 2., .03), (0., 0., 0., 1.)))
    s = Sampler()
    s._campaign_pair_active = p
    s.touching_goal_at = lambda *args: th.tensor([1., 2.])
    s.blade_span_at = lambda *args: None
    s.gp = SimpleNamespace(gen=lambda key: None)
    s.place = {}
    s.seg_arm = 'right'
    stance = SimpleNamespace(extras={'face_point': [1.2, 2.]}, base_pose2d=(0, 0, 0), translate=.1, tilt=0.)
    result = s._push_under_chain(stance, th.tensor(p.release[1]), th.tensor([.8, 2.]), th.tensor([1., 2.]), th.tensor([1., 0.]))
    assert result['native_checked'] and len(calls) == 1
    assert s._push_under_chain(stance, th.tensor(p.release[1]), th.tensor([.7, 2.]), th.tensor([1., 2.]), th.tensor([1., 0.])) is None
    assert s._push_under_chain(stance, th.tensor([0., 0., 1., 0.]), th.tensor([.8, 2.]), th.tensor([1., 2.]), th.tensor([1., 0.])) is None
    assert len(calls) == 1


def test_large_under_keeps_only_native_lane_and_solves_at_fixed_drop_attitude():
    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge
    p = replace(plan('under'), scale='large', placement_mode='drop_push',
        release=((1., 2., .003), (0., 0., 0., 1.)), goal_cell=(1., 2.),
        place=((.7, 2., .15), (0., 0., 0., 1.)), floor_push={'edge': '-x', 'direction': [1., 0.]})
    s = IORSLarge.__new__(IORSLarge)
    s._campaign_pair_active = p
    calls = []
    s.push_yaw_kept = lambda q, d: True
    s._push_under_chain_solve = lambda *args: calls.append(args) or {'native_checked': True}
    extra = {'edge': '-x', 'drop_cell': [.7, 2.], 'goal_cell': [1., 2.], 'push_dir': [1., 0.]}
    stance = SimpleNamespace(extras=extra)
    assert s.campaign_floor_push_lane(extra)
    for key, value in [('edge', '+x'), ('drop_cell', [.72, 2.]), ('goal_cell', [1., 2.002]), ('push_dir', [-1., 0.])]:
        assert not s.campaign_floor_push_lane({**extra, key: value})
    # The candidate's initial carried yaw is not the final drop attitude; F_goal
    # independently checks the fixed release, and the contact solver must agree.
    assert s._push_under_chain(stance, th.tensor([0., 0., 1., 0.]), [.7, 2.], [1., 2.], [1., 0.])['native_checked']
    assert th.allclose(calls[0][1], th.tensor(p.place[1]))
    assert s._push_under_chain(stance, th.tensor(p.place[1]), [.7, 2.], [1.02, 2.], [1., 0.]) is None
    s.push_yaw_kept = lambda q, d: False
    assert s._push_under_chain(stance, th.tensor(p.place[1]), [.7, 2.], [1., 2.], [1., 0.]) is None
    assert len(calls) == 1


def test_large_under_filters_checked_candidates_without_mutating_native_stances():
    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge
    p = replace(plan('under'), scale='large', placement_mode='drop_push', goal_cell=(1., 2.),
        place=((.7, 2., .15), (0., 0., 0., 1.)), floor_push={'edge': '-x', 'direction': [1., 0.]})
    s = IORSLarge.__new__(IORSLarge)
    s._campaign_pair_active = p
    edge = SimpleNamespace(name='-x', mid=th.tensor([.8, 2.]))
    s.place = {'edges': [edge]}; s.seg_arm = 'right'
    s.gp = SimpleNamespace(gen=lambda name, default=None: False if name == 'placement_pool_edges' else default, variant='fixture')
    kept = SimpleNamespace(extras={'edge': '-x', 'drop_cell': [.7, 2.], 'goal_cell': [1., 2.], 'push_dir': [1., 0.]})
    other = SimpleNamespace(extras={**kept.extras, 'goal_cell': [1., 3.]})
    s._placement_candidates_at = lambda *a: [other, kept]
    result = s.placement_candidates([.2, .3])
    assert result == [kept] and result[0] is kept
    assert other.extras['goal_cell'] == [1., 3.]


def test_nextto_source_hint_survives_plan_validation_with_exact_goal_and_provenance():
    import copy
    from ispatialgym.sampling.campaign_pair_goals import SCHEMA, validate_pair_plan
    from ispatialgym.sampling.campaign_held_goals import GoalPlanError
    pose={'position':[1.,2.,.8], 'orientation':[0.,0.,0.,1.]}
    raw={'schema':SCHEMA,'task':'inter_object_relation_satisfaction-nextto_small_scale',
         'template_id':'t','scene':'s','room':'r','campaign_id':'c','group_id':'g',
         'target_name':'salt','partner_name':'pepper','split':'eval_standard','contract_template_sha256':'a'*64,
         'release_pose':pose,'goal_pose':pose,'partner_pose':pose,'created_at':10.,'budget_seconds':900.,
         'anchor_sources':{'summary':{'sha256':'b'*64}},
         'anchor_manipulation_stance':{'base_pose2d':[5.389566,3.647212,-.002552],
             'focus':[6.219768,3.725094,.890702],'translate':0.,'tilt':-.610867,'source_summary_sha256':'b'*64}}
    def check(value, relation='nextto'):
        return validate_pair_plan(value,template_id='t',scene='s',rooms=['r'],relation=relation,activated_at=10.,now=11.)
    validated=check(raw)
    assert validated.anchor_manipulation_stance['translate']==0.
    assert validated.goal==validated.release==((1.,2.,.8),(0.,0.,0.,1.))
    touching=copy.deepcopy(raw);touching['task']='inter_object_relation_satisfaction-touching_small_scale'
    touching['place_pose']={'position':[.95,2.,.8],'orientation':[0.,0.,0.,1.]}
    assert check(touching,'touching').anchor_manipulation_stance==validated.anchor_manipulation_stance
    assert check(touching,'touching').place[0][0]==.95
    changed=copy.deepcopy(raw);changed['anchor_manipulation_stance']['source_summary_sha256']='c'*64
    with pytest.raises(GoalPlanError,match='provenance'):check(changed)
    changed=copy.deepcopy(raw);changed['task']='inter_object_relation_satisfaction-ontop_small_scale'
    with pytest.raises(GoalPlanError,match='requires small'):check(changed,'ontop')
