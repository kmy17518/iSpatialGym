"""i7-artic 2026-11-07 (owner decision: handles stay with the ARTICULATION library): the small-articulation alias motions built with the
sampler's own handle code (``grasping.artic_motions``) for articulation_open_small_01 / 02 and articulation_close_small_01 / 02, and
``validate_artic``'s library segment of a trial (``segment_from_trial``: the validated, monotone part of the motion). Simulator-free."""

import itertools
import json
import math
import os

import pytest
import torch as th

from ispatialgym.grasping import artic_motions as AM
from ispatialgym.grasping import validate_artic as VA

HANDLES = os.path.join(os.path.dirname(__file__), "..", "..", "runlogs", "grasp_library_20260905", "artic", "handles")
ALIASES = {"open_01": "configuration_matching-articulation_open_small_scale-blender-breakfast_table",
           "open_02": "configuration_matching-articulation_open_small_scale-carton-desk",
           "close_01": "configuration_matching-articulation_close_small_scale-blender-breakfast_table",
           "close_02": "configuration_matching-articulation_close_small_scale-carton-breakfast_table"}


# ------------------------------------------------------------------------------------------------------------------
# the alias motions (the sampler's own handle code)
# ------------------------------------------------------------------------------------------------------------------


def _hull(doc):
    lo = [min(doc["base_link"]["bbox_min"][i], *[j["link_bbox_min"][i] for j in doc["joints"]]) for i in range(3)]
    hi = [max(doc["base_link"]["bbox_max"][i], *[j["link_bbox_max"][i] for j in doc["joints"]]) for i in range(3)]
    return th.tensor(list(itertools.product(*zip(lo, hi))), dtype=th.float32)


def _motion(tag):
    alias = ALIASES[tag]
    entry = AM.load_alias(alias)
    cat, model, joint, hkey = AM.target_model(entry)
    with open(os.path.join(HANDLES, f"{hkey}.json")) as f:
        doc = json.load(f)
    j = next(x for x in doc["joints"] if x["joint"] == joint)
    return AM.build(alias, entry, doc, _hull(doc), (j["lower"], j["upper"]))


def test_the_four_alias_motions():
    o1, o2, c1, c2 = (_motion(t) for t in ("open_01", "open_02", "close_01", "close_02"))
    # open_01: the bar ring pinched at the lower limit, pulled to the upper; the oracle's pre-shape w / 2 + 1 cm; both finger assignments
    assert (o1["motion"], o1["joint"], o1["q_start"], o1["q_arc_end"]) == ("pull_open", "j_lid", 0.0, o1["upper"])
    assert [v["name"] for v in o1["variants"]] == ["h#0+", "h#0-"] and o1["variants"][0]["preshape_q"] == pytest.approx(0.5 * 0.0078 + 0.010, abs=1e-5)
    assert o1["variants"][1]["handle"]["pinch_axis"] == [-v for v in o1["variants"][0]["handle"]["pinch_axis"]]
    assert o1["arms_offered"] == ["right", "left"] and o1["handle_arm"] == "right"  # arm "auto": the side rule picks, both are offered
    # open_02 (two_flap_v4): the main pinch's 21 sites / yaws / pitches x 2 at q_ajar 0.35, v2's pre-shape margin 2 cm, the right arm only
    assert (o2["motion"], o2["joint"], o2["q_start"]) == ("v2_ajar_pull", "j_right_lid", 0.35) and len(o2["variants"]) == 42
    assert o2["arms_offered"] == ["right"] and o2["variants"][0]["preshape_q"] == pytest.approx(min(0.05, 0.5 * 0.0008 + 0.02), abs=1e-5)
    assert {v["handle"].get("site", "").split("@")[0] for v in o2["variants"]} == {"deep", "near_edge"}
    assert o2["profile_spec"]["joint_drives"]["j_left_lid"]["lower_limit_rad"] == -0.55
    # close_01 / close_02: the closed fist on the tip at close_init_fraction of the range, to the lower limit + the overshoot
    assert (c1["motion"], c1["q_start"], c1["q_goal"], c1["q_arc_end"]) == ("push_close", pytest.approx(0.9 * c1["upper"]), 0.0, pytest.approx(-0.08))
    assert c1["arms_offered"] == ["right"] and c1["variants"][0]["preshape_q"] == 0.0 and c1["variants"][0]["handle"]["kind"] == "tip_push"
    assert (c2["motion"], c2["q_start"], c2["q_arc_end"]) == ("push_close", pytest.approx(0.7 * c2["upper"]), pytest.approx(-0.03)) and c2["arms_offered"] == ["right", "left"]
    for m in (o1, o2, c1, c2):
        f = m["facing_local"]
        assert m["face_yaw_offset"] == pytest.approx(math.atan2(math.sin(math.pi - math.atan2(f[1], f[0])), math.cos(math.pi - math.atan2(f[1], f[0]))), abs=1e-6)
        assert AM.handle_signature(m["base_handles"], m["jinfo"]) == m["handle_sig"]  # the json copies reproduce the signature
    assert len({m["handle_sig"] for m in (o1, o2, c1, c2)}) == 4
    with pytest.raises(ValueError):
        AM.build("x", {"roles": {"target": {"category": "blender", "representative": "cwkvib", "joint": "j_lid"}}, "generation": {"direction": "close"}},
                 {"joints": []}, th.zeros(1, 3), (0.0, 1.0))


# ------------------------------------------------------------------------------------------------------------------
# validate_artic: the library segment of a trial
# ------------------------------------------------------------------------------------------------------------------


def _pose(x, y=0.0, z=0.3):
    return {"position": [x, y, z], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}


def _trial(qs, until, push=False):
    label = "push0" if push else "pull0"
    traj = [{"step": 1, "segment": "close", "q": qs[0], "eef_in_initial_object": _pose(0.0)}]
    traj += [{"step": 2 + i, "segment": label, "q": q, "eef_in_initial_object": _pose(0.001 * i)} for i, q in enumerate(qs)]
    return {"stop": "limit_reached", "validated_until_step": until, "T_object_eef_used": _pose(0.0), "start": {"q_held": qs[0]}, "trajectory": traj,
            "motion_segment": label, "variant": "h#0+", "candidate": "a:h#0+", "arm_validated": "right"}


def test_segment_from_trial_keeps_the_validated_monotone_part():
    m = {"alias": "a", "motion": "pull_open", "joint": "j", "face_yaw_offset": 3.14, "arms_offered": ["right"], "handle_sig": "sig",
         "variants": [{"name": "h#0+", "assignment": "+pinch", "preshape_q": 0.014, "contact_q": 0.005}]}
    qs = [0.0, 0.01, 0.02, 0.035, 0.03, 0.07, 0.08, 0.5]  # a step back (0.03) is never kept; 0.5 lies past the validated step
    tr = _trial(qs, until=8)
    seg = VA.segment_from_trial(tr, m, {"rest_id": "r0"}, VA.MotionConfig(wp_step_m=1.0), {"validator": "v"})
    assert seg["q_list"] == [0.0, 0.035, 0.07, 0.08] and seg["validated_travel_rad"] == pytest.approx(0.08)
    assert seg["waypoints"][0]["eef_in_object"] == _pose(0.0) and seg["fingers"] == {"pre": 0.014, "path": 0.005} and seg["id"]
    assert VA.segment_from_trial(dict(tr, stop="ag_not_formed"), m, {}, VA.MotionConfig(), {}) is None
    assert VA.segment_from_trial(dict(tr, validated_until_step=None), m, {}, VA.MotionConfig(), {}) is None
    assert VA.segment_from_trial(tr, m, {}, VA.MotionConfig(min_travel_rad=0.2, wp_step_m=1.0), {}) is None
    # a push travels toward the lower limit
    pm = dict(m, motion="push_close")
    seg = VA.segment_from_trial(_trial([1.0, 0.99, 0.95, 0.9], until=5, push=True), pm, {}, VA.MotionConfig(wp_step_m=1.0), {"validator": "v"})
    assert seg["q_list"] == [1.0, 0.95, 0.9] and seg["validated_travel_rad"] == pytest.approx(0.1)
    lib = VA.build_motion_library(dict(m, category="c", model="m", scale=[1, 1, 1], q_start=0.0, handle_arm="right"), {"rest_id": "r0"}, {}, [tr], VA.MotionConfig(wp_step_m=1.0), {"validator": "v"})
    assert lib["schema"] == VA.LIBRARY_V2 and lib["n_segments"] == 1 and lib["rests"] == {"r0": {"rest_id": "r0"}}
