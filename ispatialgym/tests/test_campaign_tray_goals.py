from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling.campaign_held_goals import ENV
from ispatialgym.sampling.campaign_tray_goals import TrayGoalPlan
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2.tasks.disembedding import Disembedding


def plan():
    return TrayGoalPlan("fresh", "g1", "eval_standard", "template", "a" * 64, "target", "source", "dest",
                        ((1., 2., .8), (0., 0., 0., 1.)), ((0., 2., .7), (0., 0., 0., 1.)),
                        ((1., 2., .7), (0., 0., 0., 1.)), (.3, .45), 1000., 3, "b" * 64)


def fixture():
    p = plan()
    s = Disembedding.__new__(Disembedding)
    s._campaign_tray_active = p
    s.campaign_group_remaining_s = lambda: 100
    s.tol = lambda key, default: default
    s.target_equivalences = lambda: []
    s.objects = {"target": SimpleNamespace(get_position_orientation=lambda: tuple(th.tensor(x) for x in p.goal))}
    d = Draft()
    d.object_poses = {"dest": tuple(th.tensor(x) for x in p.destination),
                      "target": (th.tensor([0., 2., .8]), th.tensor(p.goal[1]))}
    d.summary["campaign_goal_group"] = p.summary()
    return s, d


def test_disabled_plan_never_accesses_runtime_context(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert Disembedding.campaign_tray_plan(SimpleNamespace()) is None


def test_fixed_side_uses_proven_viewpoint_and_does_not_redraw(monkeypatch):
    import math
    from dataclasses import replace
    from ispatialgym.sampling.campaign_tray_goals import destination_side_at_anchor
    p = plan()
    assert destination_side_at_anchor(p.source, p.destination, (0., 0., -math.pi/2)) == "left"
    assert destination_side_at_anchor(p.source, p.destination, (0., 0., math.pi/2)) == "right"
    assert destination_side_at_anchor(p.source, p.destination, None) is None
    monkeypatch.setenv(ENV, "/immutable/plan.json")
    s = Disembedding.__new__(Disembedding)
    s.campaign_tray_plan = lambda: replace(p, destination_side="right")
    s.g = lambda name: None
    s.template_facts = {}
    rng = SimpleNamespace(random=lambda: pytest.fail("fixed world trays must not redraw their template side"))
    assert s._destination_side(rng) == "right"
    assert s.template_facts["destination_side"]["source"] == "campaign_fixed_tray_anchor"
    s.g = lambda name: "left"
    with pytest.raises(Reject, match="explicit template side"):
        s._destination_side(rng)


def test_ambiguous_fixed_side_is_rejected():
    from ispatialgym.sampling.campaign_tray_goals import destination_side_at_anchor
    from ispatialgym.sampling.campaign_held_goals import GoalPlanError
    p = plan()
    with pytest.raises(GoalPlanError, match="lateral separation"):
        destination_side_at_anchor(p.source, p.destination, (0., 0., 0.))


def test_settled_target_keeps_exact_requested_goal_and_destination_identity():
    s, d = fixture()
    s.campaign_verify_goal(d, settled=True)
    assert d.pose_targets["target"]["position"] == list(plan().goal[0])
    assert d.summary["campaign_goal_group"]["phase"] == "settled_goal_verified"


def test_settled_goal_or_destination_drift_is_rejected():
    s, d = fixture()
    d.object_poses["dest"][0][0] += .04
    with pytest.raises(Reject, match="destination drifted"):
        s.campaign_verify_goal(d)
    s, d = fixture()
    s.objects["target"].get_position_orientation = lambda: (th.tensor([1.04, 2., .8]), th.tensor([0., 0., 0., 1.]))
    with pytest.raises(Reject, match="settled tray goal drifted"):
        s.campaign_verify_goal(d, settled=True)


def test_fixed_goal_must_fit_destination_and_preserve_start_orientation():
    s, d = fixture()
    s.world = SimpleNamespace(footprint=lambda hull, pose: [[0, 0], [1, 0], [1, 1]])
    s.hull = {"target": None}
    s.g = lambda key: .005
    dst = SimpleNamespace(z=.7, polygon_supported=lambda footprint, margin: False)
    with pytest.raises(Reject, match="footprint leaves"):
        s._goal_pose(d.object_poses["target"], dst)
    dst.polygon_supported = lambda footprint, margin: True
    goal = s._goal_pose(d.object_poses["target"], dst)
    assert th.equal(goal[0], th.tensor(plan().goal[0]))
    with pytest.raises(Reject, match="orientations must match"):
        s._goal_pose((th.zeros(3), th.tensor([0., 0., 1., 0.])), dst)


def test_pinned_trays_are_validated_directly_without_random_destination_search():
    s = Disembedding.__new__(Disembedding)
    s.src_role, s.dst_role, s.dest_side = "source", "destination", 1.
    s._place_destination = lambda *args: pytest.fail("fixed goal must not rediscover a random destination")
    hull = th.tensor([[-.1, -.1, 0.], [-.1, .1, 0.], [.1, .1, .1], [.1, -.1, .1]])
    s.hull = {"source": hull, "destination": hull}
    camera = SimpleNamespace(width=640, height=480, project=lambda cam, pts: (th.tensor([[320., 240.]]), th.tensor([1.])))
    s.world = SimpleNamespace(footprint=lambda h, p: h[:, :2] + p[0][:2], camera_model=camera)
    s.top = SimpleNamespace(polygon_supported=lambda fp, margin: True)
    # Different tray meshes have different floor-mask centres. Native placement
    # aligns their object origins, so those offsets must not change eligibility.
    s.src_floor = SimpleNamespace(at=lambda p: SimpleNamespace(centre=p[0][:2] + th.tensor([0., .02]), z=.7))
    s.dst_floor = SimpleNamespace(at=lambda p: SimpleNamespace(centre=p[0][:2] + th.tensor([0., -.03]), z=.7))
    s._tray_yaw = lambda role, manip: 0.
    s.g = lambda key: {"destination_gap_min_m": .04, "destination_visible_width_m": .04}[key]
    manip = SimpleNamespace(local=lambda p: (float(p[1]), float(p[0])), base_xy=th.zeros(2), tangent=th.tensor([1., 0.]),
                            world=lambda depth, lat: th.tensor([lat, depth]))
    source = (th.tensor([0., 0., .7]), th.tensor([0., 0., 0., 1.]))
    destination = (th.tensor([1., 0., .7]), th.tensor([0., 0., 0., 1.]))
    actual, side, facts = s.campaign_tray_geometry(manip, None, source, destination)
    assert actual is destination and side == 1 and facts["source"] == "goal_plan"
    assert facts["alignment_basis"] == "object_origins" and facts["edge_depth_residual_m"] == 0
    displaced = (destination[0] + th.tensor([0., .006, 0.]), destination[1])
    with pytest.raises(Reject, match="tray origins"):
        s.campaign_tray_geometry(manip, None, source, displaced)
    camera.project = lambda cam, pts: (th.tensor([[700., 240.]]), th.tensor([1.]))
    with pytest.raises(Reject, match="required width"):
        s.campaign_tray_geometry(manip, None, source, destination)


def test_stance_search_filters_actual_fixed_trays_before_attempting_grasps():
    s = Disembedding.__new__(Disembedding)
    p = plan()
    s._campaign_tray_active = p
    s._place_destination = lambda *args: pytest.fail('native random destination must not select a fixed-tray start')
    s._tray_pose_centred = lambda *args: pytest.fail('fixed source must not follow a newly drawn focus')
    s.camera_pose_at = lambda init: 'actual_start_camera'
    s.template_facts = {}
    s.dest_side = 1.
    s.gp = SimpleNamespace(variant='fixed_tray_test')
    seen = []
    def geometry(candidate, camera, source, destination):
        seen.append((camera, source, destination))
        assert th.allclose(source[0], th.tensor(p.source[0]))
        assert th.allclose(destination[0], th.tensor(p.destination[0]))
        if not candidate.visible:
            raise Reject('campaign_destination_not_visible', 'required width')
    s.campaign_tray_geometry = geometry
    bad = SimpleNamespace(focus=th.tensor([5., 5., 0.]), visible=False)
    good = SimpleNamespace(focus=th.tensor([6., 5., 0.]), visible=True)
    init = SimpleNamespace(edge=SimpleNamespace(name='+x'))
    assert s._destination_fit(init, [bad, good]) == [good]
    assert len(seen) == 2 and seen[0][0] == 'actual_start_camera'
    assert s.template_facts['destination_fit'][0]['candidates_after'] == 1


def test_margin_screen_requires_exact_witnesses_and_the_existing_safe_threshold(monkeypatch):
    from ispatialgym.sampling import campaign_runtime as CR
    from ispatialgym.sampling.v2 import orient as O
    s = Disembedding.__new__(Disembedding)
    g = SimpleNamespace(arm='left', pre_grasp='pre', grasp='grasp', extras={})
    monkeypatch.setattr(CR, 'contact_recovery', lambda option: False)
    s.campaign_grasp_joint_margin(g)  # No witness access for existing templates.
    monkeypatch.setattr(CR, 'contact_recovery', lambda option: True)
    s.ikc = SimpleNamespace(JOINT_LIMIT_MARGIN=.05, joint_limit_margin=lambda q, arm: (q, 'joint6'))
    monkeypatch.setattr(O, 'chain_qs', lambda *args: [.1, .0058])
    with pytest.raises(Reject, match='native safe margin'):
        s.campaign_grasp_joint_margin(g)
    monkeypatch.setattr(O, 'chain_qs', lambda *args: [.1, .1204])
    s.campaign_grasp_joint_margin(g)
    assert g.extras['campaign_grasp_joint_margin'] == {'margin_rad': .1, 'required_rad': .05}
    monkeypatch.setattr(O, 'chain_qs', lambda *args: None)
    with pytest.raises(Reject, match='exact native witnesses'):
        s.campaign_grasp_joint_margin(g)
