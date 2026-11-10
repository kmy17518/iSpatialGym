"""i7-t03both r3 (lead #84c): the arm-mirror transfer of a lean rest's grasps to the other board (``lean.mirror_job``) - the body map is the
reflection that carries the source rest onto the destination rest's mirror, the object must be symmetric under it, the transferred grasps
are the only (given-pose, "legacy") candidates and the provenance is recorded. Simulator-free."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from ispatialgym.grasping import lean as LN

#: hardback acbrnv: the lean_left_20 rest as measured and its upright base (the entry's lean_base_wxyz)
L20_REST = [0.4027625, 0.4028003, -0.5811905, 0.5811644]
BASE_WXYZ = [0.5, 0.5, 0.5, -0.5]
SIDE_9 = {"position": [0.050367, -0.061758, 0.0], "orientation_xyzw": [-0.7047068, 0.0582088, -0.0582088, 0.7047068]}
BOX = [[sx * 0.1114, sy * 0.0858, sz * 0.0156] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def _src():
    return {"arm": "right", "lean": {"side": "left", "deg": 20.0, "tag": "lean_left_20_c5c708"},
            "rests": {"lean_left_20_c5c708": {"quat_xyzw": L20_REST}}, "grasps": [{"name": "side_9", "T_object_eef": SIDE_9}, {"name": "x", "T_object_eef": SIDE_9}]}


def _job(side="right", deg=20.0):
    j = {"category": "hardback", "model": "acbrnv", "scale": [1, 1, 1], "dataset": "behavior-1k-assets"}
    return LN.make_job(j, LN.LeanSpec(deg, side), LN.xyzw(BASE_WXYZ), base_key="hardback__acbrnv__bu")


def test_the_body_map_is_a_reflection_of_the_book_through_its_thickness_plane():
    B = LN.mirror_body_map(L20_REST, LN.lean_quat(LN.xyzw(BASE_WXYZ), LN.LeanSpec(20.0, "right")))
    assert np.linalg.det(B) == pytest.approx(-1.0, abs=1e-6) and np.allclose(B @ B, np.eye(3), atol=1e-9)  # a pure reflection
    assert np.allclose(B, np.diag([1.0, 1.0, -1.0]), atol=1e-6)  # the book's thickness plane (the measured 20.55 vs design 20 deg projected out)
    assert LN.mirror_symmetry_error(BOX, B) < 1e-9


def test_the_mirrored_hand_is_the_mirror_of_the_world_hand():
    dst = LN.lean_quat(LN.xyzw(BASE_WXYZ), LN.LeanSpec(20.0, "right"))
    B = LN.mirror_body_map(L20_REST, dst)
    g = LN.mirror_grasp(SIDE_9, B)
    Rl, Rr = Rot.from_quat(L20_REST).as_matrix(), Rot.from_quat(dst).as_matrix()
    Wl = Rl @ Rot.from_quat(SIDE_9["orientation_xyzw"]).as_matrix()
    Wr = Rr @ Rot.from_quat(g["quat_xyzw"]).as_matrix()
    assert np.allclose(Wr, LN.MIRROR @ Wl @ LN.MIRROR, atol=0.02)  # the right arm's mirrored eef frame (up to the 0.55 deg settle residual)
    assert np.allclose(Rr @ np.array(g["position"]), LN.MIRROR @ (Rl @ np.array(SIDE_9["position"])), atol=1e-3)
    assert np.linalg.det(Rot.from_quat(g["quat_xyzw"]).as_matrix()) == pytest.approx(1.0)


def test_mirror_job_gives_only_the_transferred_poses_and_its_provenance():
    j = LN.mirror_job(_job(), _src(), BOX, ["side_9"])
    co = j["cand_overrides"]
    assert co["modes"] == [] and co["families"] == ["legacy"] and [g["name"] for g in co["legacy_grasps"]] == ["mirror_side_9"]
    assert j["mirror_of"]["grasps"] == ["side_9"] and j["mirror_of"]["source_rest"] == "lean_left_20_c5c708"
    assert j["id"].endswith("__mirror_lean_left_20_c5c708") and j["lean"]["side"] == "right"
    assert len(LN.mirror_job(_job(), _src(), BOX)["cand_overrides"]["legacy_grasps"]) == 2  # default: every source grasp


def test_mirror_job_refuses_the_same_board_and_an_asymmetric_object():
    with pytest.raises(ValueError, match="other board"):
        LN.mirror_job(_job(side="left"), _src(), BOX)
    lopsided = BOX + [[0.0, 0.0, 0.03]]
    with pytest.raises(ValueError, match="not symmetric"):
        LN.mirror_job(_job(), _src(), lopsided)
    with pytest.raises(ValueError, match="no source grasp"):
        LN.mirror_job(_job(), _src(), BOX, ["nope"])


def test_twins_add_the_hand_turned_about_its_approach():
    j = LN.mirror_job(_job(), _src(), BOX, ["side_9"], twins=True)
    g = {x["name"]: x for x in j["cand_overrides"]["legacy_grasps"]}
    assert set(g) == {"mirror_side_9", "mirror_side_9~r180"} and "twins" in j["mirror_of"]["method"]
    A = Rot.from_quat(g["mirror_side_9"]["quat_xyzw"]).as_matrix()
    T = Rot.from_quat(g["mirror_side_9~r180"]["quat_xyzw"]).as_matrix()
    assert np.allclose(A[:, 2], T[:, 2], atol=1e-6) and np.allclose(A[:, 0], -T[:, 0], atol=1e-6)  # same approach, x / y turned
    assert g["mirror_side_9"]["position"] == g["mirror_side_9~r180"]["position"]
