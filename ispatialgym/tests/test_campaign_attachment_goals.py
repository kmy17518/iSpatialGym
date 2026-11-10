from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch as th

from ispatialgym.sampling import campaign_attachment_goals as goals
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2.stances import GraspRecord, SidecarDraft
from ispatialgym.sampling.v2.tasks import attachment as at


class Sampler(at.Attachment):
    tname = "cap"
    pname = "bottle"


def plan():
    raw = {"schema": goals.SCHEMA, "task": "configuration_matching-attachment", "campaign_id": "new",
           "group_id": "group", "split": "eval_standard", "template_id": "template", "scene": "scene", "room": "room",
           "contract_template_sha256": "a"*64, "target_name": "cap", "partner_name": "bottle",
           "goal_pose": {"position": [.6, .1, .95], "orientation": [0, 0, 0, 1]},
           "partner_pose": {"position": [.6, .1, .8], "orientation": [0, 0, 0, 1]},
           "created_at": 100, "budget_seconds": 900, "max_members": 3}
    return goals.validate_attachment_plan(raw, template_id="template", scene="scene", rooms=["room"], activated_at=200, now=201)


def sampler():
    s = object.__new__(Sampler)
    s._campaign_attachment_active = plan()
    s.campaign_group_remaining_s = lambda: 100
    s.gen = lambda key, default: default
    s.tol = lambda key, default: default
    s.ikc = NS(WITNESS_POS_TOL=.005, WITNESS_ANG_TOL=.05)
    s.symmetry = []
    s._group_name = lambda names: "pair"
    s.sidecar = SidecarDraft("configuration_matching-attachment", "configuration_matching")
    return s


def test_native_partner_map_is_required_without_snapping_the_goal(monkeypatch):
    s = sampler()
    s.attach_bands = object()
    stance = object()
    calls = []
    def in_band(bands, st, xy, tol):
        calls.append((bands, st, xy.clone(), tol))
        return True, th.tensor([.61, .11])
    monkeypatch.setattr(at.TT, "in_band", in_band)
    cells = s.campaign_partner_cells(stance)
    assert cells.tolist()[0] == pytest.approx([.6, .1])
    assert calls[0][0] is s.attach_bands and calls[0][1] is stance
    assert calls[0][3] == at.TT.CELL_TOLERANCE
    monkeypatch.setattr(at.TT, "in_band", lambda *args: (False, None))
    assert s.campaign_partner_cells(stance).shape == (0, 2)


def test_attachment_mating_and_settled_drift_cannot_change_group_goal():
    s = sampler()
    p = s._campaign_attachment_active
    s._attached_pose = lambda partner: (partner[0]+th.tensor([0, 0, .15]), partner[1])
    s.campaign_verify_mating(p)
    with pytest.raises(Reject, match="target drifted"):
        s.campaign_verify_mating(replace(p, goal=((.6, .1, .98), p.goal[1])))
    d = Draft()
    d.goal_poses[s.tname] = s.campaign_tensor_pose(p.goal)
    d.object_poses[s.pname] = s.campaign_tensor_pose(p.partner)
    d.object_poses[s.tname] = s.campaign_tensor_pose(((.4, 0, .81), p.goal[1]))
    s.sidecar.add_grasp(GraspRecord(object=s.tname, arm="right", grasp_key="cap__a", grasp_name="pinch",
        yaw=0, pre_grasp=d.object_poses[s.tname], grasp=d.object_poses[s.tname],
        eef_in_object=(th.zeros(3), th.tensor([0., 0, 0, 1]))))
    s.campaign_finish_draft(d)
    assert d.summary["campaign_goal_group"]["requested_start"]["grasps"][0]["grasp_name"] == "pinch"
    moved_target = (d.goal_poses[s.tname][0]+th.tensor([.002, 0, 0]), d.goal_poses[s.tname][1])
    s.campaign_verify_goal(d, settled=(moved_target, d.object_poses[s.pname]))
    slots = d.groups["pair"]["target_slots"]
    assert slots["male"]["position"] == pytest.approx(p.goal[0])
    assert slots["female"]["position"] == pytest.approx(p.partner[0])
    shifted_partner = (d.object_poses[s.pname][0]+th.tensor([.006, 0, 0]), d.object_poses[s.pname][1])
    with pytest.raises(Reject, match="partner drifted"):
        s.campaign_verify_goal(d, settled=(moved_target, shifted_partner))


def test_only_target_start_and_grasp_change_within_fixed_attachment_goal():
    s = sampler()
    p = plan()
    target, partner = p.goal, p.partner
    p.verify(target, partner, position_tol=.01, orientation_tol=.15)
    with pytest.raises(goals.GoalPlanError, match="partner drifted"):
        p.verify(target, (partner[0], [0, 0, 1, 0]), position_tol=.01, orientation_tol=.15)
    assert p.summary()["phase"] == "planned_before_start"
    assert p.deadline_at == 1100  # Activated time, not the earlier authoring time.
