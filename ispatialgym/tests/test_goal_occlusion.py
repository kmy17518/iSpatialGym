"""ispatialgym.eval.goal_occlusion: roles, the box / panel / pixel screens on synthetic frames."""

import math

import numpy as np
import torch as th
import pytest

from ispatialgym.eval import goal_occlusion as GO

W = H = 120
F = 60.0
INTR = (W, H, F, F, W / 2.0, H / 2.0)
# camera at the origin looking along world +x (camera -z -> world +x, camera +y -> world +z): quaternion of the rotation
# whose columns are the camera axes x_c = -y_w, y_c = +z_w, z_c = -x_w
_R = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _quat_of(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    return [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]


CAM = ([0.0, 0.0, 0.0], _quat_of(_R))
IDENT = [0.0, 0.0, 0.0, 1.0]


def test_quat_roundtrip():
    assert np.allclose(GO.quat_xyzw_to_matrix(CAM[1]), _R, atol=1e-9)


def test_roles():
    assert GO.role("target_banana_znakxm_0") == "object"
    assert GO.role("partner_bowl_belcml_0") == "object"
    assert GO.role("targets_toy_dice_isgblk_3") == "object"
    assert GO.role("dishwasher_ynrhuk_0") == "fixture"
    assert GO.role("door_crtrpa_0") == "fixture"
    for name in ("breakfast_table_rhjoby_0", "distractors_bar_soap_feqemg_0", "source_tray_uekqey_0", "walls_urngpl_0", "obstacles_footstool_mysfnz_0"):
        assert GO.role(name) is None
    assert GO.exempt("camera_relocalization-standard") and GO.exempt("camera_relocalization-obstructed-footstool_1")
    assert not GO.exempt("mental_rotation")


def test_enabled_switch():
    assert GO.enabled({}) and GO.enabled({"ISPATIALGYM_GOAL_OCCLUSION": "on"})
    assert not GO.enabled({"ISPATIALGYM_GOAL_OCCLUSION": "off"}) and not GO.enabled({"ISPATIALGYM_GOAL_OCCLUSION": "0"})


def _depth(value):
    return np.full((H, W), float(value))


def test_box_screen_clear_and_occluded():
    box = ([2.0, 0.0, 0.0], IDENT)
    clear = GO.box_occlusion(_depth(np.inf), CAM, box, [0.2, 0.2, 0.2], [0, 0, 0], INTR)
    assert clear["status"] == "ok" and clear["occluded"] == 0.0 and clear["in_frame_px"] > 0
    assert abs(clear["box_depth_m"] - 1.9) < 0.02
    wall = GO.box_occlusion(_depth(1.0), CAM, box, [0.2, 0.2, 0.2], [0, 0, 0], INTR)
    assert wall["occluded"] == 1.0
    # the surface of the box itself (depth = its entry depth) is not an occluder: within the 2 cm tolerance
    own = GO.box_occlusion(_depth(1.9), CAM, box, [0.2, 0.2, 0.2], [0, 0, 0], INTR)
    assert own["occluded"] == 0.0


def test_box_screen_half_covered():
    box = ([1.0, 0.0, 0.0], IDENT)
    d = _depth(np.inf)
    d[:, : W // 2] = 0.5  # the image's left half (world +y side) behind a near surface
    r = GO.box_occlusion(d, CAM, box, [0.4, 0.4, 0.4], [0, 0, 0], INTR)
    assert 0.4 < r["occluded"] < 0.6
    assert GO.report({"target_x": r}, "test")["occluded"] == (r["occluded"] >= 0.5)


def test_box_screen_offset_and_out_of_frame():
    behind = GO.box_occlusion(_depth(np.inf), CAM, ([-2.0, 0.0, 0.0], IDENT), [0.2, 0.2, 0.2], [0, 0, 0], INTR)
    assert behind["status"] == "out_of_frame"
    # base_link_offset moves the box centre in the object frame: offset +4 m along x puts a box placed behind the camera in front
    shifted = GO.box_occlusion(_depth(np.inf), CAM, ([-2.0, 0.0, 0.0], IDENT), [0.2, 0.2, 0.2], [4.0, 0, 0], INTR)
    assert shifted["status"] == "ok"
    with pytest.raises(ValueError):
        GO.box_occlusion(np.ones((10, 10)), CAM, ([2.0, 0, 0], IDENT), [0.2, 0.2, 0.2], [0, 0, 0], INTR)


def test_box_screen_matches_occ_py_formulation():
    """The ray / slab formulation of occ.py (camera-frame rays with z = -1, rotated into the box frame) on a rotated box."""
    ang = 0.6
    q = [0.0, 0.0, math.sin(ang / 2), math.cos(ang / 2)]
    box = ([1.5, 0.2, -0.1], q)
    d = _depth(np.inf)
    d[: H // 3] = 1.2
    r = GO.box_occlusion(d, CAM, box, [0.3, 0.1, 0.2], [0.01, 0.0, 0.02], INTR)
    Rc, Ro = GO.quat_xyzw_to_matrix(CAM[1]), GO.quat_xyzw_to_matrix(q)
    vs, us = np.mgrid[0:H, 0:W]
    rays = np.stack([(us.ravel() - W / 2) / F, -(vs.ravel() - H / 2) / F, -np.ones(H * W)], 1)
    half, centre = np.array([0.15, 0.05, 0.1]), np.array(box[0]) + Ro @ np.array([0.01, 0.0, 0.02])
    d_b, o_b = (rays @ Rc.T) @ Ro, Ro.T @ (np.zeros(3) - centre)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - o_b) / d_b, (half - o_b) / d_b
    tn, tf = np.nanmax(np.minimum(t1, t2), 1), np.nanmin(np.maximum(t1, t2), 1)
    hit = (tn <= tf) & (tf > 0)
    occ = d.ravel()[hit] < np.maximum(tn[hit], 0) - 0.02
    assert r["in_frame_px"] == int(hit.sum()) and r["occluded"] == round(float(occ.mean()), 4)


def test_panel_screen():
    lo, hi = [1.9, -0.3, -0.3], [2.0, 0.3, 0.3]
    without = _depth(np.inf)
    without[:] = 1.95  # the panel fills the view where its box is
    r = GO.panel_occlusion(_depth(1.95), without, CAM, lo, hi, INTR)
    assert r["status"] == "ok" and r["occluded"] == 0.0
    arm = without.copy()
    arm[:, W // 2 :] = 0.8  # the robot's arm covers the right half
    r2 = GO.panel_occlusion(arm, without, CAM, lo, hi, INTR)
    assert 0.4 < r2["occluded"] < 0.6
    assert GO.panel_occlusion(_depth(1.0), _depth(5.0), CAM, lo, hi, INTR)["status"] == "panel_not_in_view"


class _T(list):
    def tolist(self):
        return list(self)


class _Obj:
    def __init__(self, name, pos, size, offset=(0, 0, 0), scale=(1, 1, 1)):
        self.name, self._pos = name, pos
        self.native_bbox, self.base_link_offset, self.scale = _T(size), _T(offset), _T(scale)

    def get_position_orientation(self):
        return _T(self._pos), _T(IDENT)


class _Joint:
    def __init__(self, body1, joint_type):
        self.body1, self.joint_type = body1, joint_type


class _Link:
    def __init__(self, path, lo, hi):
        self.prim_path, self.aabb = path, (_T(lo), _T(hi))


class _Fixture:
    def __init__(self, name):
        self.name = name
        self.joints = {"hinge": _Joint("/door/leaf", "revolute"), "slide": _Joint("/door/rack", "prismatic")}
        self.links = {"leaf": _Link("/door/leaf", [1.9, -0.3, -0.3], [2.0, 0.3, 0.3]), "rack": _Link("/door/rack", [5.0, 5.0, 5.0], [6.0, 6.0, 6.0]),
                      "base": _Link("/door/base", [-9, -9, -9], [9, 9, 9])}


def test_screen_scene_and_report():
    target = _Obj("target_box_0", [1.0, 0.0, 0.0], [0.2, 0.2, 0.2], scale=(2, 2, 2))
    partner = _Obj("partner_box_0", [-3.0, 0.0, 0.0], [0.2, 0.2, 0.2])
    door = _Fixture("door_crtrpa_0")
    other = _Obj("breakfast_table_rhjoby_0", [1.0, 0.0, 0.0], [1, 1, 1])
    depth = _depth(np.inf)
    depth[:, W // 2 :] = 0.5
    without = _depth(1.95)
    objs = GO.screen_scene(depth, CAM, INTR, [target, partner, door, other], depth_without_robot=without, revolute="revolute")
    assert set(objs) == {"target_box_0", "partner_box_0", "door_crtrpa_0"}
    assert objs["partner_box_0"]["status"] == "out_of_frame"
    assert 0.4 < objs["target_box_0"]["occluded"] < 0.6 and abs(objs["target_box_0"]["box_depth_m"] - 0.8) < 0.02
    assert objs["door_crtrpa_0"]["part"] == ["leaf"] and 0.4 < objs["door_crtrpa_0"]["occluded"] < 0.6
    assert GO.screen_scene(depth, CAM, INTR, [door])["door_crtrpa_0"]["status"] == "not_measured"
    rep = GO.report(objs, "test")
    assert rep["worst"] in ("target_box_0", "door_crtrpa_0") and rep["occluded"] == (rep["max_occluded"] >= 0.5)
    assert GO.report({"target_a": {"status": "out_of_frame"}}, "test")["occluded"] is False


def test_inside_goal_judges_the_container_not_the_target():
    """A3 (owner 2026-10-12): the target of an inside goal sits in an opaque container (inside#02's candy, 100 % hidden in every check):
    it is measured but not judged; the partner - the container - is judged with the same threshold."""
    jar = _Obj("partner_hingeless_jar_0", [1.0, 0.0, 0.0], [0.3, 0.3, 0.3])
    candy = _Obj("target_hard_candy_0", [1.1, 0.0, 0.0], [0.05, 0.05, 0.05])
    depth = _depth(np.inf)
    depth[:] = 0.85  # the jar's front wall stands in front of the candy (its own surface is within the tolerance)
    inside = "inter_object_relation_satisfaction-inside_small_scale-hard_candy-hingeless_jar-bar"
    assert GO.contained_target(inside) and GO.contained_target("inter_object_relation_satisfaction-inside_small_scale")
    assert not GO.contained_target("inter_object_relation_satisfaction-ontop_small_scale") and not GO.contained_target(None)
    objs = GO.screen_scene(depth, CAM, INTR, [jar, candy], subtask=inside)
    assert objs["target_hard_candy_0"]["occluded"] == 1.0 and objs["target_hard_candy_0"]["judged"] is False
    assert objs["partner_hingeless_jar_0"]["occluded"] == 0.0 and "judged" not in objs["partner_hingeless_jar_0"]
    rep = GO.report(objs, "test")
    assert rep["occluded"] is False and rep["worst"] == "partner_hingeless_jar_0" and rep["max_occluded"] == 0.0
    # any other goal: the hidden target makes the frame invalid, as before
    other = GO.report(GO.screen_scene(depth, CAM, INTR, [jar, candy], subtask="inter_object_relation_satisfaction-ontop_small_scale"), "test")
    assert other["occluded"] is True and other["worst"] == "target_hard_candy_0"
    # an inside goal whose container is hidden fails on the container, at the same threshold
    hidden = GO.report(GO.screen_scene(_depth(0.5), CAM, INTR, [jar, candy], subtask=inside), "test")
    assert hidden["occluded"] is True and hidden["worst"] == "partner_hingeless_jar_0" and hidden["threshold"] == GO.THRESHOLD


class _Sensor:
    name = "fake:Camera:0"
    image_height, image_width = 2, 3

    def __init__(self, first_frame=6):
        self.frames, self.first_frame = 0, first_frame

    def render(self):
        self.frames += 1

    def read(self, _sensor):
        if self.frames < self.first_frame:
            return th.zeros(0)
        return th.full((6,), 1.5)


def test_settled_depth_waits_for_a_settled_non_empty_buffer():
    from ispatialgym.oracle.head_depth import settled_depth

    s = _Sensor()
    d = settled_depth(s, s.render, read=s.read)
    assert d.shape == (2, 3) and float(d[0, 0]) == 1.5 and s.frames == 7
    dead = _Sensor(first_frame=10**6)
    with pytest.raises(RuntimeError):
        settled_depth(dead, dead.render, max_renders=5, read=dead.read)
