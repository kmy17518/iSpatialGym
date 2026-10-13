"""Simulator-free tests of the push-under's palm contact (owner 2026-10-12 ~19:40, under_large #02 / #03: ``generation.
push_under_palm_contact``): the palm's half extent from the gripper's cuRobo spheres (``tabletop.palm_half_extent``), the vertical
blade's palm stand-off per wrist turn in ``IORSLarge._push_under_chain`` and the record's ``palm_contact`` / ``palm_clear`` flags
(``pytest ispatialgym/tests/test_push_under_palm.py``)."""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import math
import os
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle import sidecar as SC
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import tabletop as TT

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")

DOWN = th.tensor([0.0, 0.0, -1.0])
D3 = th.tensor([0.0, 1.0, 0.0])
Q0 = th.tensor([0.0, 0.0, 0.0, 1.0])


# ----------------------------------------------------------------------------------------------------------------
# the palm's half extent
# ----------------------------------------------------------------------------------------------------------------


def test_palm_half_extent_per_wrist_turn():
    # the vertical blade closing along the push (the chain's two wrist turns): the knuckles lead with the camera turned away from the
    # face (eef +y toward it), the camera leads when it faces it
    assert TT.palm_half_extent(C.eef_quat(DOWN, D3), D3) == pytest.approx(0.042 + 0.015)
    assert TT.palm_half_extent(C.eef_quat(DOWN, -D3), D3) == pytest.approx(0.057 + 0.018)
    # closing across the push: the palm body (eef +x toward the face) / the camera (eef -x)
    across = th.tensor([1.0, 0.0, 0.0])
    assert TT.palm_half_extent(C.eef_quat(DOWN, across), D3) == pytest.approx(0.04)
    assert TT.palm_half_extent(C.eef_quat(DOWN, -across), D3) == pytest.approx(0.056 + 0.018)
    # the push direction's length and the world yaw do not matter
    yawed = th.tensor([math.cos(0.7), math.sin(0.7), 0.0])
    assert TT.palm_half_extent(C.eef_quat(DOWN, yawed), 3.0 * yawed) == pytest.approx(0.057)


def _r1pro_models():
    from ispatialgym import REPO_ROOT

    roots = [os.path.join(REPO_ROOT, "Behavior-1K", "datasets")]
    if os.environ.get("OMNIGIBSON_DATA_PATH"):
        roots.append(os.environ["OMNIGIBSON_DATA_PATH"])
    spec = importlib.util.find_spec("omnigibson")  # the installed package's location; OmniGibson itself is not imported
    if spec is not None and spec.origin:
        roots.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(spec.origin))), "datasets"))
    for root in roots:
        path = os.path.join(root, "omnigibson-robot-assets", "models", "r1pro")
        if os.path.exists(os.path.join(path, "curobo", "r1pro_description_curobo_default.yaml")) and os.path.exists(os.path.join(path, "urdf", "r1pro.urdf")):
            return path
    return None


def _hom(xyz, rpy):
    r, p, y = rpy
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(r), -math.sin(r)], [0.0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0.0, math.sin(p)], [0.0, 1.0, 0.0], [-math.sin(p), 0.0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0.0], [math.sin(y), math.cos(y), 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


def _same_spheres(a, b, tol):
    """Whether the spheres ``a`` and ``b`` (``(centre, radius)``) pair up one to one, centres within ``tol``, the same radii."""
    b = list(b)
    for ca, ra in a:
        j = next((j for j, (cb, rb) in enumerate(b) if abs(ra - rb) < 1e-9 and max(abs(x - y) for x, y in zip(ca, cb)) <= tol), None)
        if j is None:
            return False
        b.pop(j)
    return not b


def test_palm_spheres_are_the_gripper_collision_model():
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    hand = IS.HAND_SPHERES_EEF
    assert _same_spheres(list(TT.PALM_SPHERES_EEF), [(c, hand["gripper body"][1]) for c in hand["gripper body"][0]]
                         + [(c, hand["knuckle"][1]) for c in hand["knuckle"][0]], 0.0)
    root = _r1pro_models()
    if root is None:
        pytest.skip("R1Pro cuRobo description / URDF not available")
    import yaml

    with open(os.path.join(root, "curobo", "r1pro_description_curobo_default.yaml")) as f:
        spheres = yaml.safe_load(f)["robot_cfg"]["kinematics"]["collision_spheres"]
    joints = {j.get("name"): j for j in ET.parse(os.path.join(root, "urdf", "r1pro.urdf")).getroot().findall("joint")}
    eef_from_gripper = np.linalg.inv(_hom(M.EEF_IN_GRIPPER_XYZ, M.EEF_IN_GRIPPER_RPY))
    for arm in ("left", "right"):
        o = joints[f"{arm}_realsense_joint"].find("origin")
        camera = _hom([float(v) for v in o.get("xyz").split()], [float(v) for v in o.get("rpy").split()])

        def mapped(link, T):
            return [(tuple(float(v) for v in (eef_from_gripper @ T @ np.array(list(s["center"]) + [1.0]))[:3]), float(s["radius"])) for s in spheres[link]]

        # measured from the robot's model: the gripper link's spheres and the wrist camera's, within the constants' 0.5 mm rounding
        assert _same_spheres(mapped(f"{arm}_gripper_link", np.eye(4)), list(TT.PALM_SPHERES_EEF), 5e-4), arm
        assert _same_spheres(mapped(f"{arm}_realsense_link", camera), [(c, TT.WRIST_CAMERA_SPHERE_R) for c in TT.WRIST_CAMERA_SPHERES_EEF], 5e-4), arm


# ----------------------------------------------------------------------------------------------------------------
# the push-under chain
# ----------------------------------------------------------------------------------------------------------------


class _GP:
    def __init__(self, **gen):
        self.alias, self.generation = "inter_object_relation_satisfaction-under_large_scale-briefcase-desk-floor", dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


def _box(w: float, d: float, h: float) -> th.Tensor:
    """Hull points of a w (x) x d (y) x h box about its root."""
    return th.tensor([[sx * w / 2, sy * d / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, h)], dtype=th.float32)


def _sampler(hull: th.Tensor, **gen):
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.gp, s.hull, s.floor_z, s.seg_arm = _GP(**gen), hull, 0.0, "right"
    s.place = {"height": float(hull[:, 2].max() - hull[:, 2].min())}
    s.obj = lambda role: role
    s.kit = SimpleNamespace(floor_z=0.0)
    s.ikc = SimpleNamespace(ignoring=lambda objs, ground=True: contextlib.nullcontext(), q_locked=lambda place: None,
                            exact=lambda chain, place, arm, **kw: SimpleNamespace(ok=True, reason=""))
    return s


PLACE = SimpleNamespace(stand_off=0.65, lateral=0.0, translate=0.75, tilt=-1.2, base_pose2d=(0.0, -0.6, math.pi / 2))
DROP, GOAL = (0.0, 0.0), (0.0, 0.30)


@pytest.fixture
def offered(monkeypatch):
    """The proposals of every ``orient.choose`` call; the first in the generator's order is chosen (the stub IK passes all)."""
    calls = []

    def choose(ikc, arm, props, evaluate, q_start, camera_idx=None, stage=""):
        calls.append(list(props))
        assert all(evaluate(p)[0] for p in props)
        return SimpleNamespace(proposal=props[0]), {"options": []}

    monkeypatch.setattr(IL.O, "choose", choose)
    return calls


def _hand_front(pose) -> float:
    """How far the hand above the fingers reaches along the push (+y) at the eef pose."""
    pos, q = pose
    sph = list(TT.PALM_SPHERES_EEF) + [(c, TT.WRIST_CAMERA_SPHERE_R) for c in TT.WRIST_CAMERA_SPHERES_EEF]
    pts = C.transform_points((pos, q), th.tensor([c for c, _ in sph], dtype=th.float32))
    return max(float(p[1]) + r for p, (_, r) in zip(pts, sph))


def test_palm_contact_off_keeps_the_rejection(offered, caplog):
    s = _sampler(_box(0.33, 0.12, 0.27))  # the briefcase standing: 27 cm tall, 12 cm deep along the push
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        assert s._push_under_chain(PLACE, Q0, DROP, GOAL, (0.0, 1.0)) is None
    assert offered == []
    assert "pitch 0: face reaches the palm; pitch 30: face reaches the palm; pitch 45: face reaches the palm; pitch 60: face reaches the palm" in caplog.text


def test_palm_stand_off_per_wrist_turn(offered):
    s = _sampler(_box(0.33, 0.12, 0.27), push_under_palm_contact=True)
    ch = s._push_under_chain(PLACE, Q0, DROP, GOAL, (0.0, 1.0))
    assert ch["palm_contact"] is True and ch["pitch_deg"] == 0.0 and ch["wrist_sign"] == 1.0
    assert ch["half_extent_m"] == pytest.approx(0.06) and ch["blade_standoff_m"] == pytest.approx(0.057 + IL.PUSH_UNDER_CLEARANCE_M)
    assert ch["fingertip_z_m"] == pytest.approx(0.05) and ch["push_height_m"] == pytest.approx(0.05 + C.FINGERTIP_LENGTH)
    back = 0.06 + 0.062
    assert ch["start"][0].tolist() == pytest.approx([0.0, -back, 0.0688], abs=1e-5)
    assert ch["goal"][0].tolist() == pytest.approx([0.0, 0.30 - back + 0.03, 0.0688], abs=1e-5)  # push_goal_extra_m past the goal cell
    assert ch["pre"][0].tolist() == pytest.approx([0.0, -back, 0.1688], abs=1e-5)  # the descent beside the face
    (props,) = offered  # the vertical blade only: the pitched blades still have no stand-off
    assert [p.payload for p in props] == [1.0, -1.0]
    # each wrist turn's hand comes down to the start 5 mm behind the box's back face (y = -0.06): the camera turned toward the face
    # stands 1.8 cm farther back than the knuckles
    for p in props:
        pre, start = p.chain[0], p.chain[1]
        assert _hand_front(start) == pytest.approx(-0.06 - IL.PUSH_UNDER_CLEARANCE_M, abs=1e-5)
        assert _hand_front(pre) == pytest.approx(-0.06 - IL.PUSH_UNDER_CLEARANCE_M, abs=1e-5)
    assert float(props[0].chain[1][0][1] - props[1].chain[1][0][1]) == pytest.approx(0.075 - 0.057, abs=1e-5)
    assert float(props[1].chain[3][0][1]) == pytest.approx(0.30 - (0.06 + 0.080) + 0.03, abs=1e-5)


def test_palm_contact_offset_over_the_whole_face(offered):
    # a 28 cm can whose rim sticks out 2 cm behind the body above 25 cm: the palm comes down beside the rim, not the body
    hull = th.cat([_box(0.24, 0.24, 0.28), th.tensor([[sx * 0.05, -0.14, z] for sx in (-1, 1) for z in (0.25, 0.28)])])
    ch = _sampler(hull, push_under_palm_contact=True)._push_under_chain(PLACE, Q0, DROP, GOAL, (0.0, 1.0))
    assert ch["half_extent_m"] == pytest.approx(0.14) and ch["start"][0].tolist() == pytest.approx([0.0, -(0.14 + 0.062), 0.0688], abs=1e-5)


def test_palm_contact_on_keeps_the_finger_blade_below_the_palm(offered):
    s = _sampler(_box(0.29, 0.31, 0.07), push_under_palm_contact=True)  # a tray: the face stays within the finger section
    ch = s._push_under_chain(PLACE, Q0, DROP, GOAL, (0.0, 1.0))
    assert "palm_contact" not in ch and ch["blade_standoff_m"] == pytest.approx(0.018 + IL.PUSH_UNDER_CLEARANCE_M)
    (props,) = offered
    assert props[0].chain[1][0].tolist() == props[1].chain[1][0].tolist()  # one stand-off for both turns, as before


# ----------------------------------------------------------------------------------------------------------------
# the record
# ----------------------------------------------------------------------------------------------------------------


def _record(**flags):
    q = C.eef_quat(DOWN, D3)
    pose = lambda y, z=0.0688: (th.tensor([0.0, y, z]), q)  # noqa: E731
    return IL.PushUnderRecord("right", pose(-0.122, 0.1688), pose(-0.122), pose(0.043), pose(0.208), D3, 0.30, "briefcase.n.01_1", pitch_deg=0.0,
                              blade_standoff_m=0.062, push_height_m=0.0688, half_extent_m=0.06, fingertip_z_m=0.05, drop_cell=[0.0, 0.0], goal_cell=[0.0, 0.3],
                              partner="desk.n.01_1", **flags)


def test_palm_push_record_runs_as_recorded():
    assert IL.push_under_palm_flags({"palm_contact": True, "pitch_deg": 0.0}) == {"palm_contact": True, "palm_clear": True}
    assert IL.push_under_palm_flags({"pitch_deg": 0.0}) == {}
    d = _record(**IL.push_under_palm_flags({"palm_contact": True})).to_sidecar()
    assert d["kind"] == "push_under" and d["palm_contact"] is True and d["palm_clear"] is True
    pc = SC.PushContact.from_dict(d, "contacts.push")
    # ``execute_push`` raises a top-down blade for palm clearance unless ``extra["palm_clear"]``
    assert pc.kind == "push_under" and pc.extra["palm_clear"] is True and pc.extra["palm_contact"] is True
    d0 = _record(**IL.push_under_palm_flags({})).to_sidecar()
    assert "palm_contact" not in d0 and "palm_clear" not in d0
