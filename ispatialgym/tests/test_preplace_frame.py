"""i7-plateqtf2 (owner decision 2026-11-08 02:35 item 3, camera-frame rule change): the IORS small draw judges the PRE-PLACE as the program
check does (the fingertip point inside QA's 5 % + 2 cm slack, program_check.fingertip_frame_margin) instead of the eef box; the place (a
contact pose) keeps the box; QA is unchanged. Simulator-free."""

from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import ik as IKm
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.sampling.v2.tasks import iors_small as IS

CM = G.CameraModel(640, 480, 400.0, 400.0, 320.0, 240.0)
CAM = (th.tensor([0.0, 0.0, 1.5]), th.tensor([0.0, 0.0, 0.0, 1.0]))  # looking down -z
DOWN = th.tensor([1.0, 0.0, 0.0, 0.0])  # the eef's approach axis (+z) pointing down: the fingertips below the eef


def _fake(**gen):
    cls = next(v for v in vars(IS).values() if isinstance(v, type) and hasattr(v, "pre_place_frame_ok"))
    ikc = SimpleNamespace(kit=SimpleNamespace(world=SimpleNamespace(camera_model=CM)))
    ikc.eef_in_frame = lambda pose, cam: IKm.IKChecker.eef_in_frame(ikc, pose, cam)
    ikc.eef_frame_margin = lambda pose, cam: IKm.IKChecker.eef_frame_margin(ikc, pose, cam)
    s = SimpleNamespace(ikc=ikc, kit=SimpleNamespace(world=SimpleNamespace(camera_model=CM), stats={}), gen=lambda k, d=None: gen.get(k, d))
    for m in ("pre_place_frame_ok", "pre_place_frame_margin", "require_in_frame"):
        setattr(s, m, getattr(cls, m).__get__(s))
    return s


def _edge_pose():
    """An eef pose near the image's right border: the fingertip well inside, a corner of the +-6 cm eef box outside the 5 % margin."""
    for x in [i * 0.005 for i in range(0, 200)]:
        pose = (th.tensor([x, 0.0, 0.9]), DOWN)
        fm = PC.fingertip_frame_margin(CM, CAM, pose)
        box = IKm.IKChecker.eef_in_frame(SimpleNamespace(kit=SimpleNamespace(world=SimpleNamespace(camera_model=CM))), pose, CAM)
        if fm["inside"] and not box:
            return pose
    raise AssertionError("no pose between the two rules")


def test_fingertip_margin_matches_the_program_check():
    pose = _edge_pose()
    fm = PC.fingertip_frame_margin(CM, CAM, pose)
    assert fm["inside"] and fm["margin_px"] >= 0.0 and fm["slack_px"] > 0
    sampler = SimpleNamespace(kit=SimpleNamespace(world=SimpleNamespace(camera_model=CM)), ikc=SimpleNamespace(camera_pose=lambda st: CAM))
    assert PC.fingertips_in_frame(sampler, SimpleNamespace(), "place", [("pre_place", [pose])]) is None  # the program check passes it too
    far = (th.tensor([2.0, 0.0, 0.9]), DOWN)
    assert not PC.fingertip_frame_margin(CM, CAM, far)["inside"] and PC.fingertip_frame_margin(CM, CAM, far)["margin_px"] < 0
    out = PC.fingertips_in_frame(sampler, SimpleNamespace(), "place", [("pre_place", [far])])
    assert out is not None and out["key"] == "pre_place"
    tip = C.fingertip(pose)
    assert float(tip[2]) < 0.9  # the criterion is the fingertip point, below the eef


def test_pre_place_takes_the_fingertip_rule_and_counts_the_removed_rejection(monkeypatch):
    monkeypatch.delenv(IS.PREPLACE_FRAME_RULE_ENV, raising=False)
    s, pose = _fake(), _edge_pose()
    assert s.pre_place_frame_ok(pose, CAM)
    assert s.kit.stats == {"preplace_frame.checked": 1, "preplace_frame.removed": 1}
    assert s.pre_place_frame_margin(pose, CAM) >= 0.0
    s.require_in_frame(CAM, {"pre-place": pose})  # passes
    with pytest.raises(Reject) as e:  # the place at that pose is a contact pose: the eef box, unchanged
        s.require_in_frame(CAM, {"pre-place": pose, "place": pose})
    assert e.value.reason == "gripper_not_in_frame" and "place eef pose" in str(e.value)
    # the switch back to the box
    monkeypatch.setenv(IS.PREPLACE_FRAME_RULE_ENV, "box")
    assert not s.pre_place_frame_ok(pose, CAM) and s.pre_place_frame_margin(pose, CAM) < 0.0
    with pytest.raises(Reject):
        s.require_in_frame(CAM, {"pre-place": pose})
    monkeypatch.setenv(IS.PREPLACE_FRAME_RULE_ENV, "wide")
    with pytest.raises(ValueError):
        IS.pre_place_frame_rule()


def test_pre_place_outside_both_rules_still_rejects(monkeypatch):
    monkeypatch.delenv(IS.PREPLACE_FRAME_RULE_ENV, raising=False)
    s = _fake()
    far = (th.tensor([2.0, 0.0, 0.9]), DOWN)
    assert not s.pre_place_frame_ok(far, CAM) and s.kit.stats["preplace_frame.rejected"] == 1
    with pytest.raises(Reject) as e:
        s.require_in_frame(CAM, {"pre-place": far})
    assert "fingertip (+2 cm slack)" in str(e.value)
