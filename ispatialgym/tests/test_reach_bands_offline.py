"""Simulator-free checks of ``ispatialgym.sampling.reach_bands``: the generic contact pose set, the pose-class bitmask
round trip through the band files, the height classes and the brace-candidate compatibility tuple."""

import json
import math
import os
import tempfile

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling import reach_bands as RB


def _bc():
    return RB.BandComputer(world=None, kit=None, arms=("left",))


def test_height_class_bins():
    assert RB.height_class(0.025) == 0.02
    assert RB.height_class(0.031) == 0.04
    assert RB.height_class(0.10) == 0.10
    assert RB.height_class(0.005) == 0.02  # never below one bin


def test_contact_candidates_geometry():
    import omnigibson.utils.transform_utils as T

    bc = _bc()
    yaw = 0.3
    fwd = th.tensor([math.cos(yaw), math.sin(yaw)])
    left = th.tensor([-math.sin(yaw), math.cos(yaw)])
    z_top, h = 0.8, 0.10
    cands = bc.contact_candidates_at(th.tensor([1.0, 2.0]), z_top, h, fwd, left)
    assert [k for k, _ in cands] == list(range(len(RB.CONTACT_CLASSES)))
    for k, (pre, contact) in cands:
        cls = RB.CONTACT_CLASSES[k]
        R = T.quat2mat(contact[1])
        a = R[:, 2]
        expect_a = th.tensor([cls["a"][0] * fwd[0] + cls["a"][1] * left[0], cls["a"][0] * fwd[1] + cls["a"][1] * left[1], cls["a"][2]])
        assert th.allclose(a, expect_a, atol=1e-5), cls["name"]
        # closing axis (eef +y) orthogonal to the approach and matching the class
        c = R[:, 1]
        expect_c = th.tensor([cls["c"][0] * fwd[0] + cls["c"][1] * left[0], cls["c"][0] * fwd[1] + cls["c"][1] * left[1], cls["c"][2]])
        assert abs(float(th.dot(c, a))) < 1e-5
        assert th.allclose(c, expect_c, atol=1e-5), cls["name"]
        # pad centre on the contact point, pre-contact 10 cm back along -a
        pad = contact[0] + RB.CONTACT_CENTER_Z * a
        z_expect = z_top + (h if cls["z"] == "top" else RB.CONTACT_HEIGHT_FRACTION * h)
        assert th.allclose(pad, th.tensor([1.0, 2.0, z_expect]), atol=1e-5), cls["name"]
        assert th.allclose(pre[0], contact[0] - a * RB.PRE_GRASP, atol=1e-6)
        assert th.equal(pre[1], contact[1])


def test_generic_classes_order_matches_candidates():
    # GENERIC_CLASSES enumerates the RobotKit.grasp_candidates_at order: pitch-major, yaw-minor
    assert RB.GENERIC_CLASSES == [f"p{int(t):02d}_y{y}" for t in RB.TILTS_GENERIC for y in (0, 90)]
    assert len(RB.GENERIC_CLASSES) == len(RB.TILTS_GENERIC) * RB.N_YAWS_GENERIC


def _write_band(tmp, meta_extra, cells, pairs, classes):
    meta = {"scene": "s", "support": "t", "object": None, "grasp_key": None, "generic_pinch": False, "lattice": 0.08, "stance": {"base_pose2d": [0, 0, 0]},
            "trunks": [{"translate": 0.5, "tilt": 0.0, "collides": False, "posture": "ready", "idle_arm_q": {"left": [0] * 7, "right": [0] * 7}},
                       {"translate": 0.95, "tilt": -0.9, "collides": True}], "yaws": RB.YAWS, **meta_extra}
    base = os.path.join(tmp, "contact__s__t__left_h0.10__abcd")
    arrays = {"t0.50_k+0.0": np.asarray(cells, dtype=np.float32), "t0.50_k+0.0__pairs": np.asarray(pairs, dtype=np.int32).reshape(-1, 3),
              "t0.50_k+0.0__classes": np.asarray(classes, dtype=np.int32)}
    np.savez_compressed(base + ".npz", **arrays)
    with open(base + ".json", "w") as f:
        json.dump(meta, f)
    return base + ".npz"


def test_contact_band_round_trip_and_queries():
    with tempfile.TemporaryDirectory() as tmp:
        cells = [[0.48, -0.16], [0.56, 0.08]]
        # cell 0: near_v (bit 0) + top_y0 (bit 8); cell 1: left_h (bit 5)
        classes = [(1 << 0) | (1 << 8), 1 << 5]
        pairs = [[0, -1, 0], [0, -1, 8], [1, -1, 5]]
        _write_band(tmp, {"arm": "left", "contact_set": RB.CONTACT_SET, "height_class": 0.10, "classes": RB.CONTACT_CLASSES_NAMES}, cells, pairs, classes)
        rb = RB.ReachBands.load("s", "t", arm="left", contact_set=RB.CONTACT_SET, height=0.094, dirs=[tmp])
        assert rb.trunks() == [(0.5, 0.0)]
        assert rb.band(0.5, 0.0).shape == (2, 2)
        assert rb.union_extent() == pytest.approx((-0.16, 0.08))
        assert rb.classes_at(0.5, 0.0, [0.48, -0.16]) == ["near_v", "top_y0"]
        assert rb.classes_at(0.5, -0.3, [0.57, 0.07]) == ["left_h"]  # nearest tilt, nearest cell
        assert rb.classes_at(0.5, 0.0, [0.9, 0.9]) == []
        assert rb.band_for_class(0.5, 0.0, "left_h").tolist() == [pytest.approx([0.56, 0.08], abs=1e-6)]
        assert rb.posture(0.5, 0.0) == {"posture": "ready", "idle_arm_q": {"left": [0] * 7, "right": [0] * 7}}
        # a right-arm lookup must not return the left contact band, nor a class mismatch
        with pytest.raises(FileNotFoundError):
            RB.ReachBands.load("s", "t", None, "generic", dirs=[tmp])
        with pytest.raises(FileNotFoundError):
            RB.ReachBands.load("s", "t", arm="left", contact_set=RB.CONTACT_SET, height=0.16, dirs=[tmp])
        # any height class when height is not given
        assert RB.ReachBands.load("s", "t", arm="left", contact_set=RB.CONTACT_SET, dirs=[tmp]).meta["height_class"] == 0.10


def test_brace_candidate_unpacks_as_pair():
    c = RB.BraceCandidate("opposite_face", [("pre", "q"), ("eef", "q")], 2)
    kind, pair = c
    assert kind == "opposite_face" and pair[1] == ("eef", "q") and c.q_prefix == 2 and len(c) == 2
