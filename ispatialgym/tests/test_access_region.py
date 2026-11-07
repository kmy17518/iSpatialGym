"""i7-overhang 2026-11-07 (owner decision: a grasp-library entry used at an interior goal gets the goal's ACCESSIBLE REGION as a candidate
constraint): the region's identity (job, entry id, library identity), entries without an interior goal unchanged, the region test on
poses (approach from the open front, the pads' reach of the front face, every hand sphere clear of the boards at the placement opening),
the pick-compatibility report. Simulator-free."""

import math

import numpy as np
import pytest

from ispatialgym.grasping import access_region as AR
from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import lean as LN
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import interiors as IN

SPEC = LN.LeanSpec(16.7, "left")
GOAL = "bookcase_jaysra_0@0.9472"


def _base():
    return {"category": "hardback", "model": "bqvvvx", "scale": [1.0, 1.0, 1.0], "dataset": "behavior-1k-assets"}


def test_region_identity_in_the_job_the_entry_id_and_the_library_identity():
    r = AR.region_identity(IN.spec_named(GOAL))
    assert r["interior_digest"] == IN.spec_named(GOAL).digest() and r["open_side"] == "front" and r["clearance_opening_q"] == 0.05
    assert r["front_depth_m"] == pytest.approx(0.0188 + 0.03811, abs=1e-4) and r["approach_min_cos"] == pytest.approx(math.cos(math.radians(45)))
    assert r["tag"] == "ar" + r["digest"][:6] and AR.region_identity(IN.spec_named(GOAL))["digest"] == r["digest"]
    assert AR.region_identity(IN.spec_named("bookcase_kmuacd_0@0.8354"))["digest"] != r["digest"]  # another goal, another region
    j = LN.make_job(_base(), SPEC, [0, 0, 0, 1.0], base_key="hardback__bqvvvx__bu", validate_arm="left", interior_goal=GOAL)
    assert j["interior_goal"] == GOAL and j["access_region"]["digest"] == r["digest"]
    assert j["id"] == f"hardback__bqvvvx__s1__{SPEC.tag()}__valleft__{r['tag']}" and S.job_dir(j, "/x").endswith(j["id"])
    assert AR.region_of_job(j) == j["access_region"] and S.job_access_region(j) == r["tag"]
    # entries without an interior goal: as before (ids, dirs, no region)
    plain = LN.make_job(_base(), SPEC, [0, 0, 0, 1.0], base_key="hardback__bqvvvx__bu")
    assert plain["id"] == f"hardback__bqvvvx__s1__{SPEC.tag()}" and "access_region" not in plain and S.job_access_region(plain) is None
    assert AR.region_of_job({**_base(), "access_region": r}) is None  # a stable job: never a region (lean entries only)
    assert S.entry_id("hardback", "bqvvvx", None) == "hardback__bqvvvx__s1"
    # the library identity carries the region's digest; one without it is unchanged
    from ispatialgym.grasping import validate_tabletop as VT

    rest = {"schema": S.REST, "rest_id": SPEC.tag(), "quat_xyzw": [0, 0, 0, 1.0], "h_root": 0.115, "stable": True, "rest_kind": "lean", "lean": SPEC.to_json()}
    lj = VT.build_library(j, {"candidates": [], "arm": "left"}, {SPEC.tag(): rest}, [], CP.DEFAULT, {}, validate_arm="left")
    lp = VT.build_library(plain, {"candidates": [], "arm": "right"}, {SPEC.tag(): rest}, [], CP.DEFAULT, {})
    assert lj["access_region"]["digest"] == r["digest"] and lj["interior_goal"] == GOAL and "access_region" not in lp
    mj = S.ModelLibrary("j", lj, {SPEC.tag(): S.Rest.from_record(rest)})
    mp = S.ModelLibrary("p", lp, {SPEC.tag(): S.Rest.from_record(rest)})
    assert mj.identity()["access_region"] == r["digest"] and "access_region" not in mp.identity() and "validate_arm" not in mp.identity()


def _pose(pos, approach, closing):
    a = np.asarray(approach, float) / np.linalg.norm(approach)
    c = np.asarray(closing, float) / np.linalg.norm(closing)
    T = np.eye(4)
    T[:3, :3] = np.stack([np.cross(c, a), c, a], 1)
    T[:3, 3] = pos
    return T


def test_the_region_test_on_poses():
    """A book standing 0.17 deep (x from -0.085, the open front), 0.03 thick (y), leaning on a board at y +0.03 (its inner face): a pinch of the
    front edge approached from the front clears the boards when its fingers stay off the board; a pinch from the back, one deep in the
    book and one whose finger is in the board are out."""
    r = AR.region_identity(IN.spec_named(GOAL))
    V = np.array([[x, y, z] for x in (-0.085, 0.085) for y in (-0.015, 0.015) for z in (0.0, 0.23)])
    lean_rest = {"board_in_root": {"centre": [0.0, 0.045, 0.2], "half": [0.2, 0.015, 0.2]}, "board_face_y_m": 0.03}
    boxes = AR.goal_boxes(r, lean_rest, 0.0, 1.0)
    assert [b.name for b in boxes] == ["lean_board", "opposite_wall", "board_above"]
    assert boxes[1].centre[1] + boxes[1].half[1] == pytest.approx(0.03 - r["width_m"])  # the opposite wall's inner face width_m away
    cz = -0.01186
    T = np.stack([_pose([-0.095, -0.12, 0.06], [1, 0, 0], [0, 0, 1]),  # from the front, mid-compartment (y -0.12): in
                  _pose([0.095, -0.12, 0.06], [-1, 0, 0], [0, 0, 1]),   # from the back: out (approach)
                  _pose([0.0, -0.12, 0.06], [1, 0, 0], [0, 0, 1]),      # the contact 0.073 m behind the front face: out (depth)
                  _pose([-0.095, 0.03, 0.06], [1, 0, 0], [0, 1, 0])])  # the open fingers in the board: out (clearance)
    ok, clr, cnt = AR.evaluate(T, np.full(4, cz), V, boxes, r)
    assert ok.tolist() == [True, False, False, False] and clr[0] > 0 and clr[3] < 0
    assert cnt == {"approach": 3, "front_depth": 2, "clear": 1}


def test_pick_compatibility_matches_a_held_pinch_or_its_twin():
    from ispatialgym.grasping import symmetry as SY

    Tp = _pose([0.03, -0.075, 0.0], [0, 1, 0], [0, 0, 1])
    pick = [{"name": "side_0", "T_object_eef": S.mat_pose(Tp)}]
    goal = [{"name": "g_same", "T_object_eef": S.mat_pose(SY.twin_T(Tp))}, {"name": "g_far", "T_object_eef": S.mat_pose(_pose([0.07, -0.05, 0.0], [0, 1, 0], [0, 0, 1]))}]
    out = AR.pick_compatibility(goal, pick)
    assert out[0]["compatible"] and out[0]["pick"] == "side_0~r180" and out[0]["pos_m"] < 1e-6
    assert not out[1]["compatible"] and out[1]["pos_m"] == pytest.approx(math.hypot(0.04, 0.025), abs=1e-4)
