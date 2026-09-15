"""Simulator-free tests of the v2 held-object and camera-relocalization task modules (spec §6.1–§6.3, §3.9, §3.13):
bucket arithmetic, goal-set construction (depth grid with |Δd| >= 0.10, angle range), waypoint counts, the fixed-hold
eef pose, equivalence expansion from symmetry tags through the customs / conventions helpers
(``pytest ispatialgym/tests/test_v2_held_camera.py``). Importing the task modules pulls in the OmniGibson package
(no simulator is launched)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2.tasks import camera_relocalization as CR
from ispatialgym.sampling.v2.tasks import held_object as HO

# ----------------------------------------------------------------------------------------------------------------
# camera relocalization: buckets (§2.3 / §6.1)
# ----------------------------------------------------------------------------------------------------------------

BUCKETS = [{"max_m": 3, "n": 10}, {"max_m": 5, "n": 10}, {"max_m": 10, "n": 5}]


def test_bucket_bounds_are_half_open():
    b = CR.normalise_buckets(BUCKETS)
    assert CR.bucket_bounds(b) == [(0.0, 3.0), (3.0, 5.0), (5.0, 10.0)]
    assert CR.bucket_index(3.0, b) == 0  # (0, 3] includes 3
    assert CR.bucket_index(3.0001, b) == 1
    assert CR.bucket_index(5.0, b) == 1
    assert CR.bucket_index(10.0, b) == 2
    assert CR.bucket_index(10.5, b) == -1
    assert CR.bucket_index(0.0, b) == -1
    assert CR.bucket_of(4.2, b) == {"max_m": 5.0, "n": 10}
    assert CR.bucket_of(12.0, b) is None


def test_buckets_are_sorted_and_validated():
    b = CR.normalise_buckets([{"max_m": 10, "n": 5}, {"max_m": 3, "n": 10}])
    assert [x["max_m"] for x in b] == [3.0, 10.0]
    with pytest.raises(ValueError):
        CR.normalise_buckets([{"max_m": 3, "n": 1}, {"max_m": 3, "n": 1}])
    with pytest.raises(ValueError):
        CR.normalise_buckets([])
    with pytest.raises(ValueError):
        CR.normalise_buckets([{"max_m": 3, "n": -1}])


def test_bucket_deficits_and_choice():
    b = CR.normalise_buckets(BUCKETS)
    counts = {3.0: 10, 5.0: 4, 10.0: 5}
    assert CR.bucket_deficits(counts, b) == [0, 6, 0]
    rng = np.random.default_rng(0)
    # only the (3, 5] bucket has a deficit: it is always chosen
    assert all(CR.choose_bucket(rng, b, counts) == 1 for _ in range(20))
    # a requested bucket wins
    assert CR.choose_bucket(rng, b, counts, requested_max_m=10) == 2
    with pytest.raises(ValueError):
        CR.choose_bucket(rng, b, counts, requested_max_m=4)
    # all full: uniform over all buckets
    full = {3.0: 10, 5.0: 10, 10.0: 5}
    seen = {CR.choose_bucket(rng, b, full) for _ in range(200)}
    assert seen == {0, 1, 2}


def test_detour_failure_rules():
    assert CR.detour_failure(None, 2.0, 5.0, 0.3) == "disconnected"
    assert CR.detour_failure(2.2, 2.0, 5.0, 0.3) == "short_detour"
    assert CR.detour_failure(5.5, 2.0, 5.0, 0.3) == "too_long"
    assert CR.detour_failure(2.4, 2.0, 5.0, 0.3) is None


def test_path_yaws_follow_the_next_segment():
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, 1.0)]
    yaws = CR.path_yaws(pts, final_yaw=0.7)
    assert yaws[0] == pytest.approx(0.0)
    assert yaws[1] == pytest.approx(math.pi / 2)
    assert yaws[2] == pytest.approx(math.pi / 2)  # zero-length step keeps the previous heading
    assert yaws[3] == pytest.approx(0.7)


def test_path_cells_middle_of_the_path():
    rng = np.random.default_rng(1)
    idx = CR.path_cells(rng, 11, 8)
    assert set(idx) <= set(range(3, 8))
    assert CR.path_cells(rng, 2, 8) == []
    assert set(CR.path_cells(rng, 3, 8)) == {1}


def _straight(n, step=0.1, y=0.0):
    return [(i * step, y) for i in range(n)]


def test_divergence_index_common_prefix():
    before = _straight(10)
    # the detour leaves the straight path after waypoint 4 (shares 0..4, then swerves to y = 0.1)
    after = before[:5] + [(0.5, 0.1), (0.6, 0.1), (0.7, 0.1), (0.8, 0.0), (0.9, 0.0)]
    assert CR.divergence_index(before, after, tol=0.05) == 4
    # identical paths share everything: the divergence is the last common waypoint
    assert CR.divergence_index(before, list(before), tol=0.05) == 9
    # paths that part immediately
    assert CR.divergence_index(before, [(0.0, 0.0), (0.1, 0.3)], tol=0.05) == 0
    # the tolerance is half a map cell: a 4 cm wobble still counts as the same waypoint
    wobble = [(x, y + (0.04 if i == 2 else 0.0)) for i, (x, y) in enumerate(before)]
    assert CR.divergence_index(before, wobble, tol=0.05) == 9
    # a shorter path cannot share more than its own length
    assert CR.divergence_index(before, before[:3], tol=0.05) == 2
    # torch paths (what the planner returns) work too
    assert CR.divergence_index(th.tensor(before), th.tensor(after), tol=0.05) == 4


def test_view_candidates_restricted_to_before_divergence_with_unobstructed_heading():
    before = _straight(10)
    after = before[:5] + [(0.5, 0.1), (0.6, 0.2), (0.7, 0.2), (0.8, 0.1), (0.9, 0.0)]
    idx, pts, yaws, k = CR.view_candidates(after, final_yaw=1.0, path_before=before, tol=0.05)
    assert k == 4
    assert idx == [0, 1, 2, 3, 4]
    assert pts == after[:5]
    # headings along the *unobstructed* path: straight ahead everywhere, also at the divergence waypoint (toward where
    # the unobstructed path would have continued), never the detour's or the goal's yaw
    assert yaws == pytest.approx([0.0] * 5)
    # without a reference path every waypoint counts, with the blocked path's own headings and the goal yaw last
    idx, pts, yaws, k = CR.view_candidates(after, final_yaw=1.0)
    assert k is None and idx == list(range(10)) and yaws[-1] == pytest.approx(1.0)
    assert yaws[4] == pytest.approx(math.atan2(0.1, 0.1))  # the detour heading at the divergence waypoint (not admissible as a view heading)


def test_obstacle_seen_only_after_the_detour_is_rejected():
    """An obstacle that enters the frame only at the last waypoint (the base turns to the goal yaw and looks back) is
    not admissible; one seen on the shared prefix is (its first admissible waypoint is recorded)."""
    divergence = 4
    assert CR.first_admissible_view([9], divergence) is None  # seen only after the detour
    assert CR.first_admissible_view([6, 7, 8], divergence) is None  # seen only along the detour
    assert CR.first_admissible_view([], divergence) is None
    assert CR.first_admissible_view([4], divergence) == 4  # the divergence waypoint itself counts
    assert CR.first_admissible_view([9, 3, 7], divergence) == 3  # the first shared waypoint at which it is in frame
    assert CR.first_admissible_view([9, 3], None) == 3  # no reference path: every waypoint counts


def test_obstructed_path_record_carries_view_facts():
    from ispatialgym.sampling.v2.stances import PathRecord

    base = PathRecord("init", "goal", "global", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.3)], 2.0, posture="tucked", carried=None, trunk=(0.5, 0.0))
    view = {"view_waypoint_index": {"o": 1}, "divergence_waypoint_index": {"o": 1}, "view_pose2d": {"o": [1.0, 0.0, 0.0]}}
    rec = CR.ObstructedPathRecord(base, view)
    d = rec.to_sidecar()
    assert PathRecord.schema_ok(d)
    assert d["view_waypoint_index"] == {"o": 1} and d["divergence_waypoint_index"] == {"o": 1} and d["view_pose2d"]["o"] == [1.0, 0.0, 0.0]
    assert d["waypoints"][-1] == [2.0, 0.0, 0.3] and d["length_m"] == 2.0
    # the F6 sidecar dataclass keeps the extra facts
    from ispatialgym.oracle import sidecar as SC

    p = SC.Path.from_dict(d, "paths[0]")
    assert p.extra["divergence_waypoint_index"] == {"o": 1} and p.goal == [2.0, 0.0, 0.3]


def test_structural_categories_never_count():
    for c in ("walls", "floors", "ceilings", "window", "door", "downlight", "room_light", "table_lamp", "curtain"):
        assert CR.is_structural(c)
    for c in ("sofa", "coffee_table", "pot_plant", "footstool"):
        assert not CR.is_structural(c)
    assert CR.is_structural(None)


def test_heading_between_axes():
    assert CR.heading_between([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert CR.heading_between([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(math.pi / 2)


# ----------------------------------------------------------------------------------------------------------------
# held object: goal sets (§6.3)
# ----------------------------------------------------------------------------------------------------------------


def test_depth_goal_set_grid_and_min_change():
    depths = HO.depth_goal_set(0.50, [0.35, 0.85], 0.05, 0.10)
    assert depths == [0.35, 0.40, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    assert all(abs(d - 0.50) >= 0.10 - 1e-9 for d in depths)
    assert 0.45 not in depths and 0.55 not in depths and 0.50 not in depths
    # a root depth off the grid (the object root is not the pad centre)
    depths = HO.depth_goal_set(0.52, [0.35, 0.85], 0.05, 0.10)
    assert all(abs(d - 0.52) >= 0.10 - 1e-9 for d in depths)
    assert 0.40 in depths and 0.65 in depths and 0.45 not in depths and 0.60 not in depths


def test_rotation_axes_one_per_antipodal_pair():
    axes = HO.rotation_axes()
    assert axes.shape == (13, 3)
    assert th.allclose(th.linalg.norm(axes, dim=1), th.ones(13), atol=1e-6)
    for i in range(13):
        for j in range(13):
            if i != j:
                assert float(th.abs(th.dot(axes[i], axes[j]))) < 1.0 - 1e-6, "two axes are (anti)parallel"


def test_rotation_goal_set_angle_range():
    goals = HO.rotation_goal_set([30.0, 120.0], 15.0)
    mags = sorted({abs(a) for _, a in goals})
    assert mags == [30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0]
    assert len(goals) == 13 * 7 * 2
    assert all(30.0 <= abs(a) <= 120.0 for _, a in goals)
    assert any(a < 0 for _, a in goals) and any(a > 0 for _, a in goals)
    with pytest.raises(ValueError):
        HO.rotation_goal_set([0.0, 120.0])


def test_depth_tolerance_floor_and_fraction():
    tol = {"depth_min_m": 0.02, "depth_fraction": 0.20}
    assert HO.depth_tolerance(0.05, tol) == pytest.approx(0.02)  # floor
    assert HO.depth_tolerance(0.30, tol) == pytest.approx(0.06)
    assert HO.depth_tolerance(-0.30, tol) == pytest.approx(0.06)


def test_visible_fraction():
    assert HO.visible_fraction(40, 100) == pytest.approx(0.4)
    assert HO.visible_fraction(0, 0) == 0.0
    assert HO.visible_fraction(120, 100) == 1.0


# ----------------------------------------------------------------------------------------------------------------
# waypoints (§3.9): 3 intermediate + the goal
# ----------------------------------------------------------------------------------------------------------------


def _pose(x, y, z, yaw=0.0):
    return th.tensor([x, y, z]), C.quat_from_yaw(yaw)


def test_line_key_poses_count_and_ends():
    p0, p1 = _pose(0.0, 0.0, 1.0), _pose(0.4, 0.0, 1.0, yaw=0.0)
    wps = HO.line_key_poses(p0, p1, 3)
    assert len(wps) == 4  # 3 intermediate + goal
    assert th.allclose(wps[-1][0], p1[0], atol=1e-6)
    xs = [float(p[0][0]) for p in wps]
    assert xs == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert all(not th.allclose(p[0], p0[0]) for p in wps), "the start pose is not repeated"


def test_orbit_key_poses_rotate_about_the_pivot():
    eef0 = _pose(0.2, 0.0, 1.0)
    pivot = th.tensor([0.0, 0.0, 1.0])
    dq = C.quat_from_axis_angle([0.0, 0.0, 1.0], math.radians(90.0))
    wps = HO.orbit_key_poses(eef0, pivot, dq, 3)
    assert len(wps) == 4
    for p, _ in wps:
        assert float(th.linalg.norm(p - pivot)) == pytest.approx(0.2, abs=1e-5)  # rigid about the pivot
    assert th.allclose(wps[-1][0], th.tensor([0.0, 0.2, 1.0]), atol=1e-5)
    assert C.quat_angle(wps[-1][1], C.qmul(dq, eef0[1])) == pytest.approx(0.0, abs=1e-5)
    # the intermediate rotations are 22.5°, 45°, 67.5° about the pivot (SLERP)
    angles = [math.degrees(C.quat_angle(q, eef0[1])) for _, q in wps]
    assert angles == pytest.approx([22.5, 45.0, 67.5, 90.0], abs=1e-3)


def test_generation_waypoints_default_is_three():
    from ispatialgym.sampling import generation_defaults as GD

    for st in ("object_scaling", "mental_rotation"):
        d = GD.defaults_for(st)
        assert d["waypoints"] == 3
        assert d["hold"] == {"depth_m": 0.50, "trunk": [0.5, 0.0], "roll_deg": 15.0}
        assert d["min_visible_fraction"] == pytest.approx(0.40)


# ----------------------------------------------------------------------------------------------------------------
# the fixed hold (§6.3)
# ----------------------------------------------------------------------------------------------------------------


def test_hold_constants_from_generation():
    hc = HO.HoldConstants.from_generation({"depth_m": 0.50, "trunk": [0.5, 0.0], "roll_deg": 15.0})
    assert (hc.depth_m, hc.translate, hc.tilt, hc.roll_deg) == (0.5, 0.5, 0.0, 15.0)
    with pytest.raises(ValueError):
        HO.HoldConstants.from_generation(None)
    k1 = HO.hold_cache_key("r1pro", hc)
    k2 = HO.hold_cache_key("r1pro", HO.HoldConstants(0.55, 0.5, 0.0, 15.0))
    assert k1 != k2 and len(k1) == 16


def test_hold_eef_pose_geometry():
    """Pad centre on the optical axis at the depth, approach = base +y, closing = world down rolled 15° about it; the
    v1 runs' recorded hold quaternion (runlogs/isg_reps_20260915 template metas) is reproduced."""
    hc = HO.HoldConstants()
    yaw = 0.3
    # a camera 1.2 m up, 0.3 m ahead of the base, looking along base +x (pitched slightly down)
    base_q = C.quat_from_yaw(yaw)
    cam_quat = C.qmul(base_q, C.quat_from_euler(0.0, math.radians(-10.0), 0.0))
    # the USD camera looks along its -z: build a frame whose -z is the viewing direction
    look = C.quat_rotate(base_q, th.tensor([math.cos(math.radians(10.0)), 0.0, -math.sin(math.radians(10.0))]))
    cam_quat = C.quat_with_z_axis(-look, cam_quat)
    cam_pos = C.quat_rotate(base_q, th.tensor([0.3, 0.0, 1.2]))
    (origin, quat), pad = HO.hold_eef_pose((cam_pos, cam_quat), yaw, hc)
    axis = C.quat_rotate(cam_quat, th.tensor([0.0, 0.0, -1.0]))
    assert float(th.dot(pad - cam_pos, axis)) == pytest.approx(0.50, abs=1e-6)
    assert float(th.linalg.norm(th.cross(pad - cam_pos, axis, dim=0))) == pytest.approx(0.0, abs=1e-6)
    a = C.approach_axis(quat)
    assert th.allclose(a, C.quat_rotate(base_q, th.tensor([0.0, 1.0, 0.0])), atol=1e-6)
    assert th.allclose(C.pad_center((origin, quat)), pad, atol=1e-6)
    # closing axis: 15° from world down, perpendicular to the approach
    c = C.closing_axis(quat)
    assert math.degrees(math.acos(float(th.dot(c, th.tensor([0.0, 0.0, -1.0]))))) == pytest.approx(15.0, abs=1e-4)
    assert float(th.dot(c, a)) == pytest.approx(0.0, abs=1e-6)
    # in the base frame the v1 hold quaternion is recovered (yaw 0 base)
    (o0, q0), _ = HO.hold_eef_pose((th.tensor([0.3, 0.0, 1.2]), C.quat_with_z_axis(th.tensor([-1.0, 0.0, 0.0]), C.IDENTITY_QUAT)), 0.0, hc)
    v1 = th.tensor([-0.7010573744773865, 0.09229596704244614, 0.09229596704244614, 0.7010573744773865])
    assert C.quat_angle(q0, v1) == pytest.approx(0.0, abs=1e-3)
    assert float(o0[1]) == pytest.approx(-C.CONTACT_CENTER_Z, abs=1e-6)  # origin = pad − CONTACT_CENTER_Z·a


# ----------------------------------------------------------------------------------------------------------------
# B7 prior
# ----------------------------------------------------------------------------------------------------------------


def test_visibility_prior_roundtrip(tmp_path):
    path = tmp_path / "visibility.json"
    path.write_text(json.dumps({"records": [
        {"grasp_key": "banana__znakxm", "grasp_name": "grasp_0", "visible_fraction": 0.83, "in_frame": True},
        {"grasp_key": "banana__znakxm", "grasp_name": "grasp_1", "visible_fraction": 0.21, "in_frame": True},
        {"grasp_key": "banana__znakxm", "grasp_name": "grasp_2", "visible_fraction": 0.9, "in_frame": False},
        {"grasp_key": "banana__znakxm", "grasp_name": "grasp_3", "visible_fraction": None, "in_frame": None},
    ]}))
    prior = HO.load_visibility_prior(str(path))
    assert HO.prior_admits(prior, "banana__znakxm", "grasp_0", 0.4) is True
    assert HO.prior_admits(prior, "banana__znakxm", "grasp_1", 0.4) is False
    assert HO.prior_admits(prior, "banana__znakxm", "grasp_2", 0.4) is False
    assert HO.prior_admits(prior, "banana__znakxm", "grasp_3", 0.4) is None
    assert HO.prior_admits(prior, "banana__znakxm", "grasp_9", 0.4) is None
    assert HO.load_visibility_prior(str(tmp_path / "missing.json")) == {}


# ----------------------------------------------------------------------------------------------------------------
# equivalences from the symmetry tags (§3.13) through the customs / problem helpers
# ----------------------------------------------------------------------------------------------------------------


def _held_entry(tags):
    return {
        "iddl": "mental_rotation/mental_rotation.iddl",
        "definition_id": 0,
        "roles": {"target": {"category": "toy_figure", "sampleable": True, "require": {"grasps": "floating"}, "representative": "zuntus",
                             "grasp_keys": {"toy_figure/zuntus": "toy_figure__zuntus"}, "orientation_equivalences": {"toy_figure/zuntus": tags}},
                  "floor": {"sampleable": False}},
        "scenes": {"house_single_floor": {"rooms": ["living_room_0"], "bindings": {"floor": {"room": "living_room_0"}},
                                          "whitelist": {"target": {"toy_figure": ["zuntus"]}}, "blacklist": {}}},
        "generation": {"templates_max": 1, "instances_per_template": 1, "tolerances": {"orientation": 0.15}, "groups": {}, "path_buckets": None},
        "constraints": [],
    }


def test_equivalence_expansion_from_tags():
    from ispatialgym.sampling import customs as CU
    from ispatialgym.sampling.v2.problem import GroundedProblem

    z2x2 = CU.equivalence_group(["z2", "x2"])
    assert len(z2x2) == 4 and z2x2[0] == [0.0, 0.0, 0.0, 1.0]
    assert len(CU.equivalence_group(["z4", "x2"])) == 8
    assert len(CU.equivalence_group(["zinf"])) == 72
    assert CU.equivalence_group(["full"]) is None
    assert CU.parse_symmetry_tags("none") == [] and CU.parse_symmetry_tags("zinf x2") == ["zinf", "x2"]

    gp = GroundedProblem("mental_rotation-toy_figure", _held_entry(["z2"]))
    eq = gp.orientation_equivalences("target", "toy_figure", "zuntus")
    assert eq is not None and len(eq) == 2
    assert eq[1] == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-6)  # the half turn about z
    gp = GroundedProblem("mental_rotation-toy_figure", _held_entry(["none"]))
    assert gp.orientation_equivalences("target", "toy_figure", "zuntus") is None
    gp = GroundedProblem("mental_rotation-toy_figure", _held_entry([]))
    assert gp.orientation_equivalences("target", "toy_figure", "zuntus") is None


def test_registry_has_the_three_subtasks():
    from ispatialgym.sampling.v2.core import REGISTRY

    assert REGISTRY["object_scaling"] is HO.ObjectScalingV2
    assert REGISTRY["mental_rotation"] is HO.MentalRotationV2
    assert REGISTRY[CR.SUBTASK_STANDARD] is CR.CameraRelocalizationV2
    assert REGISTRY[CR.SUBTASK_OBSTRUCTED] is CR.CameraRelocalizationV2
