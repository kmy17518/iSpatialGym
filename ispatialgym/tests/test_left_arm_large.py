"""The arm of the IORS large-scale segment (owner 2026-09-27, first-demo v2.2): the right-side laterals mirrored for the left arm,
the other arm's band (its own or the mirror, never written), the per-arm stances B and the couplings' pair roles, no simulator."""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")


class _G:
    def __init__(self, name):
        self.name = name


# ------------------------------------------------------------------------------------------------------------------
# the right-side laterals, mirrored
# ------------------------------------------------------------------------------------------------------------------


def test_floor_offsets_put_the_anchor_on_the_arms_side():
    offsets = (-0.25, -0.15, -0.35, 0.0)
    assert IL.mirrored_offsets(offsets, "right") == list(offsets)
    assert IL.mirrored_offsets(offsets, "left") == [0.25, 0.15, 0.35, 0.0]
    c_t = th.tensor([1.0, 2.0])
    for arm in ("right", "left"):
        for bearing in (0.3, 2.0, -2.5):
            for dist in IL.STAND_OFFS:
                xy = c_t + th.tensor([math.cos(bearing), math.sin(bearing)]) * dist
                yaw0 = math.atan2(float(c_t[1] - xy[1]), float(c_t[0] - xy[0]))
                for y_off in IL.mirrored_offsets(offsets, arm):
                    st = Stance("manipulation", (float(xy[0]), float(xy[1]), IL.offset_yaw(yaw0, y_off, dist)), 0.95, -0.9)
                    ahead, lat = (float(v) for v in st.world_to_base(c_t))
                    assert lat == pytest.approx(y_off, abs=1e-5) and ahead == pytest.approx(math.sqrt(dist ** 2 - y_off ** 2), abs=1e-5)
                    assert lat * IL.arm_sign(arm) <= 1e-6, "c_t on the robot's right for the right arm, on its left for the left arm"


def test_carry_poses_beside_the_body_on_the_arms_side():
    right = IL.carry_poses_for(IL.IORSLarge.CARRY_POSES, "right")
    assert right == [tuple(p) for p in IL.IORSLarge.CARRY_POSES], "the right arm's poses as before"
    left = IL.carry_poses_for(IL.IORSLarge.CARRY_POSES, "left")
    assert [(a, r) for a, _, r in left] == [(a, r) for a, _, r in IL.IORSLarge.CARRY_POSES]
    assert all(lat is None or lat > 0.3 for _, lat, _ in left) and left[-1][1] is None, "positive laterals, the lift's lateral kept"
    assert IL.carry_poses_for([(0.5, -0.4, 0.05)], "left") == [(0.5, 0.4, 0.05)]


def test_frontal_push_lane_on_the_arms_side_of_the_base():
    edge = EdgeFrame.from_edge("+x", th.tensor([2.0, 1.0]), th.tensor([1.0, 0.0]), 2.0)  # inward -x, tangent -y (the robot's left)
    lane_lat, off, so = 0.3, 0.25, 0.7
    assert IL.frontal_lane_lateral(lane_lat, off, "right") == float(lane_lat) + off, "the right arm's base lateral as before"
    for arm, side in (("right", -1.0), ("left", 1.0)):
        lat_b = IL.frontal_lane_lateral(lane_lat, off, arm)
        xy = edge.base_xy(so, lat_b)
        st = Stance("placement", (float(xy[0]), float(xy[1]), edge.yaw), 0.95, -1.2, so, edge, None, lat_b)
        lane = edge.mid + edge.tangent * lane_lat
        ahead, lat = (float(v) for v in st.world_to_base(lane))
        assert ahead == pytest.approx(so, abs=1e-5) and lat == pytest.approx(side * off, abs=1e-5)


def test_side_push_stance_faces_along_the_edge_with_the_partner_on_the_arms_side():
    mid, inward = th.tensor([0.5, -1.0]), th.tensor([0.0, 1.0])
    tangent = th.tensor([-float(inward[1]), float(inward[0])])
    lane_lat, x_ahead, b = 0.2, 0.55, 0.35
    x, y, yaw = IL.side_push_pose(mid, tangent, inward, lane_lat, x_ahead, b, "right")
    xy_old = mid + tangent * (float(lane_lat) - x_ahead) - inward * b
    assert (x, y, yaw) == (float(xy_old[0]), float(xy_old[1]), math.atan2(float(tangent[1]), float(tangent[0]))), "the right arm's pose as before"
    lane = mid + tangent * lane_lat
    for arm, side in (("right", -1.0), ("left", 1.0)):
        x, y, yaw = IL.side_push_pose(mid, tangent, inward, lane_lat, x_ahead, b, arm)
        st = Stance("placement", (x, y, yaw), 0.95, -1.2, b)
        ahead, lat = (float(v) for v in st.world_to_base(lane))
        assert ahead == pytest.approx(x_ahead, abs=1e-5) and lat == pytest.approx(side * b, abs=1e-5)
        assert float(st.forward @ tangent) == pytest.approx(-side, abs=1e-6), "facing +tangent (right arm) / -tangent (left arm)"
        assert float(st.left @ inward) == pytest.approx(side, abs=1e-6), "the partner (inward) on the arm's side"


# ------------------------------------------------------------------------------------------------------------------
# pairs over arms, the couplings' roles
# ------------------------------------------------------------------------------------------------------------------


def test_union_of_the_arms_pairs_on_the_screening_grid():
    step = 2 * math.pi / 16
    g0, g1, g2 = _G("grasp_0"), _G("grasp_1"), _G("grasp_2")
    F_r = [(0.0, g0), (step, g1)]
    F_l = [(0.3927, g1), (2 * step, g2)]  # a cached yaw rounded to 4 decimals lands in the same bucket
    assert IL.pair_key(0.3927, g1, step) == IL.pair_key(step, g1, step) == (1, "grasp_1")
    assert IL.union_feasible({"right": F_r, "left": F_l}, step) == F_r + [F_l[1]]
    assert IL.union_feasible({"right": F_r}, step) == F_r


def test_segment_arms_offer_the_arms_holding_a_pair_and_the_roles_of_a_won_grasp():
    step = 2 * math.pi / 16
    g0, g1, g2 = _G("grasp_0"), _G("grasp_1"), _G("grasp_2")
    st = Stance("manipulation", (0.0, 0.0, 0.0), 0.5, 0.0)
    seg = IL.SegmentArms({"right": st, "left": st}, {"right": [(0.0, g0), (step, g1)], "left": [(0.3927, g1), (2 * step, g2)]}, yaw_step=step)
    assert seg.arms == ["right", "left"]
    assert seg.arms_of(step, g1) == ["right", "left"] and seg.arms_of(0.0, g0) == ["right"] and seg.arms_of(2 * step, g2) == ["left"]
    # the right arm's coupling runs first: its own grasps proceed, the left arm's wait for the left coupling
    assert seg.role("right", ["right", "left"], "right") == "own"
    assert seg.role("left", ["right", "left"], "right") == "defer"
    # the right coupling failed: the grasps it won go to the left arm where the left arm was viable at the grasp
    seg.done.append("right")
    assert seg.role("left", ["left", "right"], "left") == "own"
    assert seg.role("right", ["right", "left"], "left") == "fallback"
    assert seg.role("right", ["right"], "left") == "none"


def test_the_couplings_start_with_the_arms_whose_band_holds_the_target():
    st = Stance("manipulation", (0.0, 0.0, 0.0), 0.5, 0.0)
    F = [(0.0, _G("grasp_0"))]
    order = IL.IORSLarge._segment_order
    assert order(st, {"right": (st, F)}) == ["right"], "one arm: as before"
    st.extras["focus_arms"] = ["right", "left"]
    assert order(st, {"right": (st, F), "left": (st, F)}) == ["right", "left"], "both bands: the default arm first"
    st.extras["focus_arms"] = ["left"]
    assert order(st, {"right": (st, F), "left": (st, F)}) == ["left", "right"], "a B of the left band: the left coupling first"
    assert order(st, {"right": (st, F), "left": (st, [])}) == ["right"] and order(st, {"right": (st, [])}) == []


# ------------------------------------------------------------------------------------------------------------------
# the other arm's band
# ------------------------------------------------------------------------------------------------------------------


def _band_file(d, *, arm=None, name="floor__s_large__floor__apple-x__k", cells=((0.40, -0.24), (0.48, -0.32), (0.56, -0.16)), n_total=None):
    meta = {"scene": "s_large", "support": "floor", "object": "apple/x", "grasp_key": "apple__x__t", "generic_pinch": False, "lattice": 0.08, "mode": "floor",
            "grasp_names": ["grasp_0", "grasp_1"], "stance": {"base_pose2d": [2.0, 1.0, 0.0]}, "union_extent": [-0.32, -0.16], "under_partner": None,
            "trunks": [{"translate": 0.95, "tilt": -0.9, "collides": False, "posture": "untucked"}]}
    if arm is not None:
        meta["arm"] = arm
    if n_total is not None:
        meta["n_reachable_total"] = n_total
    np.savez_compressed(os.path.join(d, name + ".npz"), **{RB.trunk_key(0.95, -0.9): np.asarray(cells, dtype=np.float32).reshape(-1, 2),
                                                           RB.trunk_key(0.95, -0.9) + "__pairs": np.zeros((0, 3), dtype=np.int32)})
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(meta, f)
    return os.path.join(d, name + ".npz")


def _sampler():
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.choice = SimpleNamespace(models={"target": SimpleNamespace(category="apple", model="x", scale=None)})
    s.world = SimpleNamespace(scene_model="s_large")
    s.grasp_key = "apple__x__t"
    s.gp = SimpleNamespace(variant="ispatialgym-test-large")
    s.template_facts = {}
    return s


@pytest.fixture
def band_dir(tmp_path, monkeypatch):
    d = str(tmp_path)
    monkeypatch.setattr(RB, "default_dirs", lambda: [d])
    monkeypatch.setenv("ISG_REACH_BANDS_COMPUTE", "0")
    return d


def test_the_other_arms_band_is_the_mirror_kept_in_memory(band_dir):
    src = _band_file(band_dir)
    before = sorted(os.listdir(band_dir))
    s = _sampler()
    right = s.band("floor")
    assert right.arm == "right" and right.path == src
    assert s.band("floor", arm="right") is right, "the working arm's band as before"
    left = s.band("floor", arm="left")
    assert left.arm == "left" and left.meta["mirrored_from"] == "right" and left.meta["mirror_source"] == src and left.path is None
    assert np.allclose(left.band(0.95, -0.9), right.band(0.95, -0.9) * np.array([1.0, -1.0]))
    assert s.band("floor", arm="left") is left, "memoised per arm"
    assert s.template_facts["arm_bands_by_factor"]["floor:floor"]["left"] == {"arm": "left", "path": None, "mirrored_from": "right", "mirror_source": src}
    assert sorted(os.listdir(band_dir)) == before, "the mirror is never written"


def test_a_cached_band_of_the_other_arm_wins_and_an_empty_one_is_a_miss(band_dir, tmp_path_factory, monkeypatch):
    _band_file(band_dir)
    real = _band_file(band_dir, arm="left", name="floor__s_large__floor__apple-x__left", cells=((0.40, 0.24),))
    left = _sampler().band("floor", arm="left")
    assert left.path == real and "mirrored_from" not in left.meta and np.allclose(left.band(0.95, -0.9), [[0.40, 0.24]])
    d2 = str(tmp_path_factory.mktemp("empty_left"))
    monkeypatch.setattr(RB, "default_dirs", lambda: [d2])
    _band_file(d2)
    _band_file(d2, arm="left", name="floor__s_large__floor__apple-x__left", n_total=0)
    assert _sampler().band("floor", arm="left").meta.get("mirrored_from") == "right"


# ------------------------------------------------------------------------------------------------------------------
# stance B per arm
# ------------------------------------------------------------------------------------------------------------------


def _b_sampler(free=True):
    s = _sampler()
    s.ikc = SimpleNamespace(stances_collision_free=lambda sts: th.tensor([free] * len(sts)))
    return s


def test_stance_b_per_arm_the_default_arm_always_the_other_where_its_band_holds_the_target():
    s = _b_sampler()
    b = Stance("manipulation", (1.0, 2.0, 0.3), 0.95, -0.9, 0.45, focus=th.tensor([1.4, 2.1, 0.03]), arms=["right"], idle_posture="ready")
    b.extras["focus_arms"] = ["right", "left"]
    out = s._b_stances(b, ["right", "left"])
    assert list(out) == ["right", "left"] and out["right"] is b and b.arms == ["right"] and b.idle_posture == "ready"
    assert out["left"] is not b and out["left"].arms == ["left"] and out["left"].idle_posture is None, "the right arm's idle posture re-derived"
    assert out["left"].base_pose2d == b.base_pose2d and (out["left"].translate, out["left"].tilt) == (0.95, -0.9)
    # a B whose band holds the target for the right arm only: the left arm is not offered there
    b2 = Stance("manipulation", (1.0, 2.0, 0.3), 0.95, -0.9, arms="untucked")
    assert list(s._b_stances(b2, ["right", "left"])) == ["right"] and b2.arms == ["right"]
    # a B searched for the left arm (floor class): the right arm's stance re-derived, the left one is B itself
    b3 = Stance("manipulation", (1.0, 2.0, 0.3), 0.95, -0.9, arms=["left"], idle_posture="tucked")
    b3.extras["focus_arms"] = ["left"]
    out3 = s._b_stances(b3, ["right", "left"])
    assert out3["left"] is b3 and out3["right"].arms == ["right"] and out3["right"].idle_posture is None
    # every idle posture collides: the arm is not offered
    assert list(_b_sampler(free=False)._b_stances(b3, ["right", "left"])) == ["left"]
    # one arm offered: B as the default arm's stance, as before
    b4 = Stance("manipulation", (1.0, 2.0, 0.3), 0.95, -0.9, arms="untucked")
    b4.extras["focus_arms"] = ["right", "left"]
    assert list(s._b_stances(b4, ["right"])) == ["right"] and b4.arms == ["right"]
