from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym.sampling import campaign_joint_goals as CJG
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks.articulation_small import ArticulationSmall


def sampler():
    s = ArticulationSmall.__new__(ArticulationSmall)
    s._campaign_joint_active = SimpleNamespace(joints={"lid": .5})
    s.direction, s.joint_name, s.q_init = "open", "lid", 0.
    s.role, s.handle_arm, s.grasp_fingers = "target", "right", .01665
    s.bindings = SimpleNamespace(one=lambda role: SimpleNamespace(name="blender"))
    s.gen = lambda key, default=None: {"brace_search": False}.get(key, default)
    s.handle = {"contact": [0., .05, .28], "approach_dir": [0., .34, -.94],
                "pinch_axis": [-1., 0., 0.], "width": .0133}
    return s


def test_real_assignments_have_distinct_starts_but_viewpoints_do_not():
    s = sampler()
    base, other = s.campaign_handle_options()
    a, b = s.campaign_handle_start(base), s.campaign_handle_start(other)
    assert a["joints"] == b["joints"] == {"lid": 0.}
    assert a["articulation_grasp"]["contact_obj"] == b["articulation_grasp"]["contact_obj"]
    assert CJG.start_key(a) != CJG.start_key(b)
    CJG.remember_written_start(s, a)
    s.init_stance = SimpleNamespace(base_pose2d=(10., 20., 1.))
    assert s.campaign_handle_options() == [other]
    CJG.remember_written_start(s, b)
    assert s.campaign_handle_options() == []


def test_each_assignment_runs_native_checks_and_annotation_is_restored():
    s = sampler()
    base, calls = s.handle, []
    def native(rng, init, manip, k):
        calls.append(s.handle["pinch_axis"])
        if len(calls) == 1:
            raise Reject("exact_ik", "first assignment cannot reach")
        return s.campaign_handle_start(s.handle)
    s._sample_with = native
    start = s._sample_with_campaign_handle(np.random.default_rng(0), None, None, 0)
    assert len(calls) == 2 and calls[0] != calls[1]
    assert s.handle is base
    assert start["articulation_grasp"]["pinch_axis_obj"] == calls[1]
    assert len(s.campaign_handle_options()) == 2  # Proposal alone does not mark a native write.
    s._sample_with = lambda *args: (_ for _ in ()).throw(Reject("exact_ik", "both fail"))
    with pytest.raises(Reject, match="both fail"):
        s._sample_with_campaign_handle(np.random.default_rng(0), None, None, 0)
    assert s.handle is base


@pytest.mark.parametrize("disabled", ["no_plan", "carton", "brace", "stage2", "closing"])
def test_other_native_paths_are_unchanged(disabled):
    s = sampler()
    if disabled == "no_plan": s._campaign_joint_active = None
    if disabled == "carton": s.v2 = True
    if disabled == "closing": s.direction = "close"
    if disabled in ("brace", "stage2"):
        s.gen = lambda key, default=None: {"brace_search": disabled == "brace", "stage2_push": disabled == "stage2"}.get(key, default)
    s._sample_with = lambda *args: "original_native_path"
    assert s.campaign_handle_options() is None
    assert s._sample_with_campaign_handle(None, None, None, 0) == "original_native_path"
