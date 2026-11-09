"""Simulator-free tests of the overhang pick (owner 2026-10-13 ~03:15 item 4; ``generation.pick_overhang_push``): the push / grasp geometry
of :mod:`ispatialgym.sampling.v2.overhang` on inside_large#03's plate and its one grasp, the sampler's flow on stubs, the records, the
oracle program, the program check's world advance and the executor's push rules (``pytest ispatialgym/tests/test_iors_large_overhang.py``)."""

from __future__ import annotations

import contextlib
import copy
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.actions import Grasp
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar, SidecarError
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import overhang as OH
from ispatialgym.sampling.v2.ik import IKResult
from ispatialgym.sampling.v2.stances import EdgeFrame, SidecarDraft, Stance

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")


@pytest.fixture(autouse=True)
def _pre_backoff_pipeline(monkeypatch):
    """i7-startdraw2 r3: ISPATIALGYM_OVERHANG_ESCAPE_BACKOFF defaults on; these tests pin the overhang pick's retreat as it was replayed before
    (the back-off's own tests: test_overhang_escape_backoff.py)."""
    monkeypatch.setenv("ISPATIALGYM_OVERHANG_ESCAPE_BACKOFF", "0")

SUBTASK = "inter_object_relation_satisfaction-inside_large_scale"
FIXTURE = os.path.join(os.path.dirname(__file__), "data", "sidecars", f"{SUBTASK}.json")
#: plate__luhkiz__ts grasp_5 (runlogs/grasp_library_20260905/cands): eef in the plate frame
G5_POS = th.tensor([-0.09103299676898666, 0.0669714734802499, 0.01399618023445373])
G5_QUAT = th.tensor([-0.3310660706472979, 0.7898856183402891, -0.365017785049809, 0.365017785049809])
G5_WIDTH = 0.00783
BOTTOM = -0.0108  # plate/luhkiz: bbox 0.241 x 0.241 x 0.026, origin at the bbox centre
Z_TOP = 0.90
#: the support's front edge in the sampler stubs (between two probe rays: 0.150 / 0.155 m from the plate's centre at x = 1.0)
EDGE_X = 1.1525
#: grasp_5's eef 0.113 m from the plate's centre along its radial direction (aligned with the push)
G5_RADIAL = float(th.linalg.norm(G5_POS[:2]))


def _plate_hull(n: int = 72) -> th.Tensor:
    """The plate as a convex hull: the foot ring at the bottom, the rim 1.6 - 2.6 cm up at r 0.1205."""
    pts = []
    for k in range(n):
        a = 2 * math.pi * k / n
        for r, z in ((0.065, BOTTOM), (0.1205, BOTTOM + 0.016), (0.1205, BOTTOM + 0.0262)):
            pts.append([r * math.cos(a), r * math.sin(a), z])
    return th.tensor(pts)


def _aligned_yaw() -> float:
    """The plate yaw that points grasp_5's radial direction along +x (the push direction of the tests)."""
    r = G5_POS[:2] / th.linalg.norm(G5_POS[:2])
    return -math.atan2(float(r[1]), float(r[0]))


def _plate_pose(x: float = 1.0, yaw: float = None):
    yaw = _aligned_yaw() if yaw is None else yaw
    return th.tensor([x, 0.0, Z_TOP - BOTTOM + 0.003]), C.quat_from_yaw(yaw)


# ----------------------------------------------------------------------------------------------------------------
# the key and the pure geometry
# ----------------------------------------------------------------------------------------------------------------


def test_options_key():
    assert OH.options(None) is None and OH.options(False) is None and OH.options("off") is None
    assert OH.options(True) == OH.DEFAULTS and OH.options("on") == OH.DEFAULTS
    o = OH.options({"mode": "only", "fingers": "closed", "pitches_deg": 30, "com_margin_m": 0.06})
    assert o["mode"] == "only" and o["fingers"] == ["closed"] and o["pitches_deg"] == [30.0] and o["com_margin_m"] == 0.06 and o["tip_m"] == OH.DEFAULTS["tip_m"]
    for bad in ({"mode": "always"}, {"fingers": ["fist"]}, {"pitches_deg": [90]}, {"com_margin_m": -0.01}, {"distance_step_m": 0}, {"max_distances": 0},
                {"unknown_key": 1}, 3, "maybe"):
        with pytest.raises(ValueError):
            OH.options(bad)


def test_min_over_hull_cut_by_planes():
    cube = th.tensor([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    assert OH.min_over_hull(cube, (1.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-6)
    # a slanted cut: min x on the cube with x + z >= 1.5 is at z = 1 -> x = 0.5
    assert OH.min_over_hull(cube, (1.0, 0.0, 0.0), [((-1.0, 0.0, -1.0), -1.5)]) == pytest.approx(0.5, abs=1e-6)
    assert OH.min_over_hull(cube, (1.0, 0.0, 0.0), [((0.0, 0.0, 1.0), -0.1)]) is None  # z <= -0.1: empty
    assert OH._crossings(cube.double(), th.tensor([0.0, 0.0, 1.0], dtype=th.float64), 0.5).shape[1] == 3


def test_grasp_5_needs_a_2_5_cm_overhang():
    """grasp_5 aligned with the push: the lower finger's closing sweep crosses the support-top plane 10.6 cm from the plate's centre (the
    fully open finger hangs 0.4-3.9 cm below the counter top outside the rim), so with a 1 cm finger clearance the edge must sit 9.6 cm
    from the centre: an overhang of 2.5 cm, the centre of mass 9.6 cm inside."""
    root, q = _plate_pose()
    eef = C.grasp_pose_from_transform((root, q), (G5_POS, G5_QUAT))
    sweep = OH.sweep_min_along(eef, (1.0, 0.0), Z_TOP, C.GRIPPER_OPEN_M, G5_WIDTH)
    assert sweep is not None and sweep[1] == 1  # the finger on +c (c points down) is the lower one
    rel = sweep[0] - float(root[0])
    assert rel == pytest.approx(0.1056, abs=0.002)
    d_min, d_max = OH.push_distance_range(0.15, 0.15, rel, 0.01, 0.05, 0.30)
    assert d_min == pytest.approx(0.15 + 0.01 - rel, abs=1e-6) and d_max == pytest.approx(0.10, abs=1e-6)
    assert OH.overhang_m(0.15, d_min, 0.1205) == pytest.approx(0.1205 - rel + 0.01, abs=1e-6)  # 2.5 cm
    assert OH.distance_ladder(d_min, d_max, 0.01, 4) == [round(d_min + k * 0.01, 4) for k in range(4)]
    # a top-down pinch above the support needs no overhang
    top_down = (th.tensor([1.0, 0.0, Z_TOP + 0.08]), C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0])))
    assert OH.sweep_min_along(top_down, (1.0, 0.0), Z_TOP, C.GRIPPER_OPEN_M, 0.02) is None
    # a centre already within the margin of the edge, fingers that would need the centre past the margin (a yaw that puts the grasp
    # beside the push line: its sweep only 2 cm ahead of the centre): no rung; a far edge: the push capped
    assert OH.distance_ladder(*OH.push_distance_range(0.045, 0.045, rel, 0.01, 0.05, 0.30), 0.01, 4) == []
    assert OH.distance_ladder(*OH.push_distance_range(0.15, 0.15, 0.02, 0.01, 0.05, 0.30), 0.01, 4) == []
    assert OH.push_distance_range(0.40, 0.40, rel, 0.01, 0.05, 0.30)[1] == 0.30
    assert OH.push_distance_range(0.10, 0.10, rel, 0.01, 0.05, 0.30)[0] == 0.01  # the rim already past the edge: the shortest push


def test_edge_from_hits_and_ladder():
    assert OH.edge_from_hits([True] * 30 + [False] * 5, 0.005) == pytest.approx(0.1475)
    assert OH.edge_from_hits([False, True]) is None and OH.edge_from_hits([True] * 10) is None and OH.edge_from_hits([]) is None
    assert OH.distance_ladder(0.05, 0.075, 0.01, 9) == [0.05, 0.06, 0.07]
    assert OH.distance_ladder(0.05, 0.05, 0.01, 9) == [0.05] and OH.distance_ladder(0.06, 0.05, 0.01, 9) == []


@pytest.mark.parametrize("pitch", [0.0, 20.0, 40.0])
def test_push_hand_leans_over_the_far_rim_and_straddles_it(pitch):
    root, q = _plate_pose()
    pts = C.transform_points((th.zeros(3), q), _plate_hull())
    rel = th.cat([pts[:, :2], (pts[:, 2] + root[2]).reshape(-1, 1)], dim=1)
    tip_z = Z_TOP + 0.006
    z_hi = tip_z + OH.FINGER_SECTION_M * math.cos(math.radians(pitch))
    c_open = OH.contact_along(rel, (1.0, 0.0), OH.lateral_bands("open"), tip_z, z_hi, pitch, tip_z)
    c_closed = OH.contact_along(rel, (1.0, 0.0), OH.lateral_bands("closed"), tip_z, z_hi, pitch, tip_z)
    if pitch == 0.0:
        # the open fingers meet the rim at +-5 cm from the push line (sqrt(0.1205^2 - 0.05^2) behind the centre), the closed blade on it
        assert c_open == pytest.approx(-math.sqrt(0.1205 ** 2 - 0.05 ** 2), abs=2e-3) and c_closed == pytest.approx(-0.1205, abs=1e-3)
    else:
        # leaned over the plate the face meets the rim top first: the contact is farther behind by (rim top - tips) tan(pitch)
        assert c_open < -math.sqrt(0.1205 ** 2 - 0.05 ** 2) - 0.003
    quat = OH.push_quat(pitch, (1.0, 0.0), 1.0)
    a, cl = C.approach_axis(quat), C.closing_axis(quat)
    assert float(a[2]) == pytest.approx(-math.cos(math.radians(pitch)), abs=1e-5) and float(a[0]) == pytest.approx(-math.sin(math.radians(pitch)), abs=1e-5)
    assert abs(float(cl[1])) == pytest.approx(1.0, abs=1e-5)  # the closing axis across the push
    ch = OH.push_chain_poses(root[:2], (1.0, 0.0), c_open, pitch, quat, tip_z, OH.face_half("open"), 0.01, 0.07)
    tip0 = C.fingertip(ch["start"])
    assert float(tip0[2]) == pytest.approx(tip_z, abs=1e-5)
    assert float(tip0[0]) - float(root[0]) == pytest.approx(c_open - OH.FINGER_HALF_WIDTH / math.cos(math.radians(pitch)) - 0.01, abs=1e-5)
    assert float(th.linalg.norm(ch["goal"][0] - ch["start"][0])) == pytest.approx(0.08, abs=1e-5) == ch["hand_travel_m"]
    assert float(th.linalg.norm(C.pre_pose(ch["start"])[0] - ch["pre"][0])) < 1e-6 and float(ch["direction"][0]) == 1.0
    assert float(th.linalg.norm(ch["mid"][0] - 0.5 * (ch["start"][0] + ch["goal"][0]))) < 1e-6
    # the gap from the face to every hull point in the fingers' bands is >= the clearance at the start
    face_along = float(tip0[0]) - float(root[0]) + OH.FINGER_HALF_WIDTH / math.cos(math.radians(pitch))
    band = (rel[:, 2] >= tip_z) & (rel[:, 2] <= z_hi) & (rel[:, 1].abs() >= C.FINGER_OPEN) & (rel[:, 1].abs() <= C.FINGER_OPEN + OH.FINGER_THICKNESS)
    gaps = rel[band, 0] - (face_along + (rel[band, 2] - tip_z) * math.tan(math.radians(pitch)))
    assert len(gaps) == 0 or float(gaps.min()) >= 0.01 - 1e-4


def test_settle_facts_and_verdict():
    p0 = (th.tensor([1.0, 0.0, 0.913]), C.quat_from_yaw(0.3))
    still = (th.tensor([1.002, 0.001, 0.910]), C.quat_from_yaw(0.31))
    f = OH.settle_facts(p0, still, (1.0, 0.0), rest_offset=0.003)
    assert f["along_m"] == pytest.approx(0.002) and f["across_m"] == pytest.approx(0.001) and f["drop_m"] == pytest.approx(0.0, abs=1e-4)
    assert f["yaw_deg"] == pytest.approx(math.degrees(0.01), abs=0.01) and f["tilt_deg"] == pytest.approx(0.0, abs=0.01)
    assert OH.settle_verdict(f, OH.DEFAULTS) is None
    tipped = (th.tensor([1.03, 0.0, 0.85]), C.qmul(C.quat_from_axis_angle(th.tensor([0.0, 1.0, 0.0]), 0.6), C.quat_from_yaw(0.3)))
    assert OH.settle_verdict(OH.settle_facts(p0, tipped, (1.0, 0.0), 0.003), OH.DEFAULTS).startswith("fell")
    turned = (p0[0].clone(), C.quat_from_yaw(0.5))
    assert OH.settle_verdict(OH.settle_facts(p0, turned, (1.0, 0.0), 0.003), OH.DEFAULTS).startswith("turned")
    assert OH.revolution_about_z(["zinf"]) and OH.revolution_about_z("zinf") and not OH.revolution_about_z(["z2"]) and not OH.revolution_about_z(None)
    q = OH.yaw_aligned_quat(C.quat_from_yaw(0.7), C.quat_from_yaw(0.2))
    assert C.quat_angle(q, C.quat_from_yaw(0.2)) < 1e-5


# ----------------------------------------------------------------------------------------------------------------
# the sampler's flow on stubs
# ----------------------------------------------------------------------------------------------------------------


class _GP:
    def __init__(self, **gen):
        self.variant, self.generation = f"{SUBTASK}-plate-drop_in_sink-countertop", dict(gen)
        self.roles = {"target": SimpleNamespace(equivalence_tags=lambda cat, model: ["zinf"])}

    def gen(self, key, default=None):
        return self.generation.get(key, default)


class _Top:
    """A support top whose front edge (toward +x, the robot) lies at ``edge_x``."""

    def __init__(self, edge_x: float):
        self.edge_x = edge_x

    def contains(self, xy, margin: float = 0.0) -> bool:
        return -2.0 < float(xy[0]) < self.edge_x


class _IKC:
    def __init__(self, grasp_ok=lambda pose: True, chain_ok=lambda chain, ignored: True):
        self.grasp_ok, self.chain_ok = grasp_ok, chain_ok
        self.calls, self.ignored, self.solutions = [], [], {}

    @contextlib.contextmanager
    def ignoring(self, objs, **kw):
        self.ignored.append([getattr(o, "name", o) for o in objs])
        yield

    def exact(self, poses, stance, arm, ignore_for_last=(), fingers=None, **kw):
        self.calls.append({"n": len(poses), "ignore_for_last": [getattr(o, "name", o) for o in ignore_for_last], "fingers": fingers, "ignored": list(self.ignored[-1:])})
        if len(poses) == 2:
            ok = self.grasp_ok(poses[-1])
            return IKResult(ok, th.zeros(28) if ok else None, "" if ok else "approach_untrackable: the lower finger in the counter", 2)
        ok = self.chain_ok(poses, self.ignored[-1] if self.ignored else [])
        return IKResult(ok, th.zeros(28) if ok else None, "" if ok else "exact IK failed at pose 1 of 4", len(poses))

    def eef_in_frame(self, pose, cam):
        return True

    def q_locked(self, stance, **kw):
        return th.zeros(28)


class _World:
    def __init__(self, target, settle_shift=(0.0, 0.0, 0.0), settle_turn=0.0):
        self.target, self.shift, self.turn = target, th.tensor(settle_shift), settle_turn
        self.spawned = {target.name: target}
        self.events = []

    def snapshot(self):
        self.events.append("snapshot")
        return {"pose": self.target.pose}

    def restore(self, state):
        self.events.append("restore")
        self.target.pose = state["pose"]

    def place(self, obj, xy, quat, support_z, local_hull=None, drop=0.003):
        obj.pose = (th.tensor([float(xy[0]), float(xy[1]), support_z - BOTTOM + drop]), th.as_tensor(quat).float())
        self.events.append(("place", round(float(xy[0]), 4)))

    def settle(self, ents, steps=60, **kw):
        p, q = self.target.pose
        self.target.pose = (p + self.shift - th.tensor([0.0, 0.0, 0.003]), C.qmul(C.quat_from_yaw(self.turn), q))
        self.events.append("settle")
        return None

    def park(self, obj, index):
        self.events.append("park")
        obj.pose = (th.tensor([40.0, 40.0, 5.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))


class _Target:
    def __init__(self):
        self.name, self.links = "plate_0", {}
        self.pose = (th.tensor([40.0, 40.0, 5.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))

    def get_position_orientation(self):
        return self.pose


class _Sampler:
    """What :meth:`IORSLarge._overhang_pick` and :meth:`overhang_records` read; the raycast probe falls back to the support-top raster."""

    pick_overhang_opts = IL.IORSLarge.pick_overhang_opts
    _overhang_edges = IL.IORSLarge._overhang_edges
    _overhang_geometry = IL.IORSLarge._overhang_geometry
    _overhang_touch_ok = IL.IORSLarge._overhang_touch_ok
    _overhang_push_chain = IL.IORSLarge._overhang_push_chain
    _overhang_push_branch = IL.IORSLarge._overhang_push_branch
    _overhang_drifted = IL.IORSLarge._overhang_drifted
    _overhang_settle = IL.IORSLarge._overhang_settle
    _overhang_pick = IL.IORSLarge._overhang_pick
    _overhang_grasp = IL.IORSLarge._overhang_grasp
    overhang_records = IL.IORSLarge.overhang_records
    num = IL.IORSLarge.num

    def _overhang_grasp_stances(self, st, arm, opts):
        return list(getattr(self, "backed_off", []))  # IORSLarge's needs the traversability map and the body check

    def __init__(self, edge_x=EDGE_X, ikc=None, world_kw=None, **gen):
        self.gp = _GP(pick_overhang_push=gen.pop("pick_overhang_push", True), **gen)
        self.target = _Target()
        self.world = _World(self.target, **(world_kw or {}))
        self.surface = {"kind": "support", "z": Z_TOP, "top": _Top(edge_x), "obj": SimpleNamespace(name="countertop_kelzer_0", category="countertop", links={})}
        self.ikc = ikc or _IKC()
        self.kit = SimpleNamespace(sync_world=lambda: None, stats={}, robot=SimpleNamespace(arm_control_idx={"right": th.arange(7), "left": th.arange(7, 14)}))
        self.robot = SimpleNamespace(links={})
        self.hull = _plate_hull()
        self.rest = C.IDENTITY_QUAT.clone()
        self.choice = SimpleNamespace(models={"target": SimpleNamespace(scale=None, category="plate", model="luhkiz")})
        self.bindings = SimpleNamespace(has=lambda role: role == "support", one=lambda role: SimpleNamespace(name="countertop.n.01_1"))
        self.sidecar = SidecarDraft(f"{SUBTASK}-plate-drop_in_sink-countertop", "inter_object_relation_satisfaction")
        self.tname = "plate.n.04_1"

    def support_objects(self):
        return [self.surface["obj"]]

    def target_com_local(self):
        return th.zeros(3)

    def camera_pose_at(self, st):
        return None

    def pose_on(self, xy, yaw, z):
        return th.tensor([float(xy[0]), float(xy[1]), z - BOTTOM + 0.003]), C.quat_from_yaw(yaw)

    def _check_deadline(self, stage):
        pass


def _first_rung() -> float:
    """The overhang pick's first push distance for the stubs' edge: the fingers' sweep 1 cm beyond it."""
    root, q = _plate_pose()
    sweep = OH.sweep_min_along(C.grasp_pose_from_transform((root, q), (G5_POS, G5_QUAT)), (1.0, 0.0), Z_TOP, C.GRIPPER_OPEN_M, G5_WIDTH)
    return (EDGE_X - 1.0) + 0.01 - (sweep[0] - float(root[0]))


def _grasp_needs(push: float):
    """A grasp IK that passes only with the plate pushed at least ``push`` (its eef far enough along +x)."""
    return lambda pose: float(pose[0][0]) >= 1.0 + G5_RADIAL + push - 1e-4


def _manip(base_x: float = 1.6):
    # B stands at +x of the support, facing -x: its edge's inward normal is -x, the push direction +x (toward the robot)
    edge = EdgeFrame("+u", th.tensor([EDGE_X, 0.0]), th.tensor([-1.0, 0.0]), 1.2)
    return Stance("manipulation", (base_x, 0.0, math.pi), 0.5, -0.3, base_x - EDGE_X, edge, None, 0.0, th.tensor([1.0, 0.0, Z_TOP]), ["right"])


def _first_passing_choose(monkeypatch):
    """``orient.choose`` reduced to the generator's order: the first proposal whose evaluation passes."""

    def choose(ikc, arm, props, evaluate, q_start, **kw):
        options = []
        for p in props:
            ok, res, qs, why = evaluate(p)
            options.append({"label": p.label, "ok": ok, "why": why})
            if ok:
                return SimpleNamespace(proposal=p, qs=None, result=res), {"options": options, "chosen": p.label}
        return None, {"options": options, "chosen": None}

    monkeypatch.setattr(IL.O, "choose", choose)


def test_overhang_pick_pushes_to_the_edge_and_grasps_the_overhang(monkeypatch):
    _first_passing_choose(monkeypatch)
    # the grasp's lower finger is clear of the counter only from the first rung on
    s = _Sampler(ikc=_IKC(grasp_ok=_grasp_needs(_first_rung())))
    pick, why = s._overhang_pick(_manip(), {"right": _manip()}, ["right"], SimpleNamespace(name="grasp_5", pose=(G5_POS, G5_QUAT), meta={"width": G5_WIDTH}),
                                 _aligned_yaw(), th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())
    assert pick is not None, why
    f = pick["facts"]
    # the raster edge 0.1525 m from the plate's centre; the first rung = the fingers' clearance
    assert f["edge_finger_m"] == pytest.approx(0.1525, abs=1e-4) and f["edge_com_m"] == pytest.approx(0.1525, abs=1e-4)
    assert f["push_m"] == pytest.approx(f["ladder_m"][0]) and f["ladder_m"][0] == pytest.approx(_first_rung(), abs=1e-4)
    assert f["overhang_m"] == pytest.approx(0.1205 + f["push_m"] - 0.1525, abs=2e-3) and f["com_inside_m"] >= 0.05
    assert f["chain"]["fingers"] == "open" and f["chain"]["pitch_deg"] == 0.0 and f["chain"]["support_touch_m"] is None
    # the grasp is the library row at the pushed pose, the support kept in the world (nothing ignored at the grasp), the target out
    eef = C.grasp_pose_from_transform(pick["post_pose"], (G5_POS, G5_QUAT))
    assert float(th.linalg.norm(pick["eef"][0] - eef[0])) < 1e-5 and float(pick["post_pose"][0][0]) == pytest.approx(1.0 + f["push_m"], abs=1e-5)
    grasp_calls = [c for c in s.ikc.calls if c["n"] == 2]
    assert all(c["ignore_for_last"] == [] and c["ignored"] == [["plate_0"]] for c in grasp_calls)
    chain_call = next(c for c in s.ikc.calls if c["n"] == 4)
    assert chain_call["fingers"] == pytest.approx(C.FINGER_OPEN) and chain_call["ignored"] == [["plate_0"]]  # the support present
    # the settle put the plate at the predicted pose, then the world back and the plate parked again
    assert s.world.events[:4] == ["snapshot", ("place", round(1.0 + f["push_m"], 4)), "settle", "restore"] and "park" in s.world.events
    assert float(s.target.pose[0][2]) == 5.0
    # the records
    grasp = SimpleNamespace(extras={})
    s.overhang_records(pick, grasp)
    rec = s.sidecar.contacts[OH.CONTACT_SLOT].to_sidecar()
    assert rec["kind"] == "pick_overhang" and rec["support"] == "countertop.n.01_1" and rec["object"] == "plate.n.04_1" and rec["stance"] == "manipulation"
    assert rec["distance_m"] == pytest.approx(f["push_m"]) and rec["fingers"] == "open" and rec["fingers_opening_m"] == pytest.approx(C.FINGER_OPEN)
    assert rec["ignore_every"] == ["plate.n.04_1"] and rec["ignore_last"] == [] and rec["direction"][:2] == [1.0, 0.0]
    assert rec["post_push_pose"]["position"][0] == pytest.approx(1.0 + f["push_m"], abs=1e-4)
    assert "retreat_up_m" not in rec  # oracle-fix 2026-11-05: only with generation.overhang_retreat_up_m
    s.gp.generation["overhang_retreat_up_m"] = 0.15
    s.overhang_records(pick, SimpleNamespace(extras={}))
    assert s.sidecar.contacts[OH.CONTACT_SLOT].to_sidecar()["retreat_up_m"] == pytest.approx(0.15)
    del s.gp.generation["overhang_retreat_up_m"]
    assert "carry_online" not in grasp.extras  # oracle-fix 2026-11-05: only with generation.carry_via_online
    s.gp.generation["carry_via_online"] = True
    g_on = SimpleNamespace(extras={})
    s.overhang_records(pick, g_on)
    assert g_on.extras["carry_online"] is True
    del s.gp.generation["carry_via_online"]
    json.dumps(rec)
    assert s.sidecar.language["objects"]["countertop.n.01_1"]["display_name"] == "countertop"  # the push's annotation names the support
    ov = grasp.extras["pick_overhang"]
    assert ov["contact"] == "push_pick" and ov["yaw_free"] is True and grasp.extras["ignore_last"] == [] and grasp.extras["ignore_every"] == []
    json.dumps(grasp.extras)


def test_overhang_pick_ladder_and_its_failures(monkeypatch):
    _first_passing_choose(monkeypatch)
    g5 = SimpleNamespace(name="grasp_5", pose=(G5_POS, G5_QUAT), meta={"width": G5_WIDTH})
    # the grasp needs 2 more rungs: the third push distance is taken
    s = _Sampler(ikc=_IKC(grasp_ok=_grasp_needs(_first_rung() + 0.02)))
    pick, _ = s._overhang_pick(_manip(), {"right": _manip()}, ["right"], g5, _aligned_yaw(), th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())
    assert pick is not None and pick["push_m"] == pytest.approx(pick["facts"]["ladder_m"][2])
    # no push chain from B: every rung fails the same way, after one chain search
    s = _Sampler(ikc=_IKC(chain_ok=lambda chain, ignored: False), support_touch_m=0.0)
    s.gp.generation["pick_overhang_push"] = {"support_touch_m": 0.0}
    pick, why = s._overhang_pick(_manip(), {"right": _manip()}, ["right"], g5, _aligned_yaw(), th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())
    assert pick is None and "push chain" in why and sum(1 for c in s.ikc.calls if c["n"] == 4) == 2 * len(OH.DEFAULTS["fingers"]) * len(OH.DEFAULTS["pitches_deg"])
    # the plate slides off its pushed pose in the settle: every rung rejected by the settle
    s = _Sampler(world_kw={"settle_shift": (0.03, 0.0, -0.02)})
    pick, why = s._overhang_pick(_manip(), {"right": _manip()}, ["right"], g5, _aligned_yaw(), th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())
    assert pick is None and "settle fell" in why
    # a small settle drift: the grasp is re-solved at the settled pose, which is recorded
    s = _Sampler(world_kw={"settle_shift": (0.004, 0.0, 0.0)})
    pick, _ = s._overhang_pick(_manip(), {"right": _manip()}, ["right"], g5, _aligned_yaw(), th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())
    assert pick is not None and float(pick["post_pose"][0][0]) == pytest.approx(float(pick["predicted_pose"][0][0]) + 0.004, abs=1e-5)
    assert float(th.linalg.norm(pick["eef"][0] - C.grasp_pose_from_transform(pick["post_pose"], (G5_POS, G5_QUAT))[0])) < 1e-5
    # the plate's centre within the centre-of-mass margin of the edge: no rung, no IK at all
    s = _Sampler(edge_x=1.0425)
    pick, why = s._overhang_pick(_manip(), {"right": _manip()}, ["right"], g5, _aligned_yaw(), th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())
    assert pick is None and why.startswith("no push distance") and s.ikc.calls == []
    # a centre of mass 2 cm ahead of the root along the push: its margin is measured from it, the push capped 2 cm earlier
    s = _Sampler()
    s.target_com_local = lambda: C.quat_rotate(C.quat_from_yaw(-_aligned_yaw()), th.tensor([0.02, 0.0, 0.0]))
    pick, _ = s._overhang_pick(_manip(), {"right": _manip()}, ["right"], g5, _aligned_yaw(), th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())
    assert pick["facts"]["edge_com_m"] == pytest.approx(EDGE_X - 1.0 - 0.02, abs=1e-4) and pick["facts"]["com_ahead_m"] == pytest.approx(0.02, abs=1e-4)
    assert pick["facts"]["push_max_m"] == pytest.approx(EDGE_X - 1.0 - 0.02 - 0.05, abs=1e-4)
    assert pick["facts"]["com_inside_m"] == pytest.approx(EDGE_X - 1.0 - 0.02 - pick["push_m"], abs=1e-4)
    # a floor init never pushes
    s = _Sampler()
    s.surface["kind"] = "floor"
    assert s._overhang_pick(_manip(), {}, ["right"], g5, 0.0, th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())[0] is None


def test_f_init_gains_the_overhang_yaws():
    """grasp_5 at B's direct yaws pinches the plate's side (the sweep 2 cm / -3 cm ahead of the centre: no push distance, sim check v1);
    with the key on F_init gains the yaws whose pinch the push brings past the edge - around the yaw aligning its radial direction with
    the push - and keeps the direct pairs (mode "only": drops them)."""
    s = _Sampler()
    s._overhang_pairs = IL.IORSLarge._overhang_pairs.__get__(s)
    g5 = SimpleNamespace(name="grasp_5", pose=(G5_POS, G5_QUAT), meta={"width": G5_WIDTH})
    s.grasps = [g5]
    yaws = [k * 2.0 * math.pi / 16 for k in range(16)]
    direct = [(yaws[4], g5)]
    F = s._overhang_pairs(_manip(), direct, th.tensor([1.0, 0.0]), yaws, Z_TOP, s.pick_overhang_opts())
    added = sorted(round(y, 3) for y, _ in F[1:])
    aligned = _aligned_yaw() % (2.0 * math.pi)
    assert F[0] == direct[0] and added and all(abs(C.wrap_angle(y - aligned)) < 1.0 for y in added)
    assert round(yaws[9], 3) in added and round(yaws[10], 3) in added and round(yaws[0], 3) not in added  # 0.24 / 0.15 rad from aligned in, pi off out
    only = s._overhang_pairs(_manip(), direct, th.tensor([1.0, 0.0]), yaws, Z_TOP, dict(s.pick_overhang_opts(), mode="only"))
    assert sorted(round(y, 3) for y, _ in only) == added


def test_b_candidates_that_suit_the_overhang_pick_rank_first():
    """The pre-screen's B candidates (``(n_F, stance, F_init, by_arm)``) ranked without an IK: a push ladder for one of the pairs, the
    push's start (behind the far rim) within the arm's reach and (with ``pre_min_ahead_m``) the grasp's pre-grasp at the overhang that far
    ahead of the base -> rank 0 (the larger reach margin first); a plate too deep for a push of at most ``max_push_m``, a push start out of
    reach or a pre-grasp nearer the body than the key allows -> rank 1."""
    s = _Sampler()
    s._overhang_b_rank = IL.IORSLarge._overhang_b_rank.__get__(s)
    reach = {"m": 0.1}
    s.reach_margin = lambda st, point, arm: reach["m"] - 0.0 * float(point[0])
    g5 = SimpleNamespace(name="grasp_5", pose=(G5_POS, G5_QUAT), meta={"width": G5_WIDTH})
    near, deep, close = _manip(1.7), _manip(1.7), _manip(1.6)
    deep.focus = th.tensor([0.75, 0.0, Z_TOP])  # 0.40 m from the edge: a 0.33 m push > max_push_m 0.30
    # the pre-grasp at the overhang 1.0 + push + 0.113 + 0.086 = 1.256 m along x: 0.44 m ahead of B at 1.70, 0.34 m ahead of B at 1.60
    e_close = (2, close, [(_aligned_yaw(), g5)], {"right": (close, [(_aligned_yaw(), g5)])})
    e_near = (2, near, [(_aligned_yaw(), g5)], {"right": (near, [(_aligned_yaw(), g5)])})
    e_deep = (2, deep, [(_aligned_yaw(), g5)], {"right": (deep, [(_aligned_yaw(), g5)])})
    opts = s.pick_overhang_opts()
    r_near, r_deep = s._overhang_b_rank(e_near, Z_TOP, opts), s._overhang_b_rank(e_deep, Z_TOP, opts)
    assert r_near[0] == 0 and r_near[1] == pytest.approx(-0.1) and r_near[2] == pytest.approx(_first_rung(), abs=1e-4)
    assert r_deep == (1, 0.0, None)
    assert s._overhang_b_rank(e_close, Z_TOP, opts)[0] == 0  # the pre-grasp's distance is not checked by default
    assert s._overhang_b_rank(e_close, Z_TOP, dict(opts, pre_min_ahead_m=0.40)) == (1, 0.0, None)
    assert s._overhang_b_rank(e_close, Z_TOP, dict(opts, pre_min_ahead_m=0.30))[0] == 0
    reach["m"] = -0.05  # the far rim out of the arm's reach
    assert s._overhang_b_rank(e_near, Z_TOP, opts) == (1, 0.0, None)
    assert sorted([e_deep, e_near], key=lambda t: s._overhang_b_rank(t, Z_TOP, dict(opts, max_push_m=0.5))[:2])[0] is e_deep  # both rank 0: equal margins keep order


def test_overhang_pick_grasps_from_b_backed_off(monkeypatch):
    """The grasp failing at B (its pre-grasp near the body): taken from B backed off along its axis (the stance ``pick``) - recorded as the
    pick's ``grasp_stance``; the push stays B's."""
    _first_passing_choose(monkeypatch)
    m = _manip()
    back = m.shifted(stand_off=float(m.stand_off) + 0.10, name="pick")
    s = _Sampler(ikc=_IKC())
    s.backed_off = [back]
    s.ikc.grasp_ok = lambda pose: True
    calls = []
    orig = IL.IORSLarge._overhang_grasp

    def grasp(self, st, arm, g, obj_pose, kw, twins, cam):
        calls.append(st.name)
        return "exact IK failed at pose 1 of 2" if st.name == "manipulation" else orig(self, st, arm, g, obj_pose, kw, twins, cam)

    s._overhang_grasp = grasp.__get__(s)
    pick, why = s._overhang_pick(m, {"right": m}, ["right"], SimpleNamespace(name="grasp_5", pose=(G5_POS, G5_QUAT), meta={"width": G5_WIDTH}),
                                 _aligned_yaw(), th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())
    assert pick is not None, why
    assert pick["grasp_stance"] is back and pick["stance"] is m and calls[:2] == ["manipulation", "pick"]
    assert pick["facts"]["grasp_stance"] == {"name": "pick", "stand_off": pytest.approx(float(m.stand_off) + 0.10), "trunk": [0.5, -0.3]}
    assert OH.options({"grasp_back_offs_m": 0.15})["grasp_back_offs_m"] == [0.15] and OH.options({"grasp_trunks": [[0.5, -0.3]]})["grasp_trunks"] == [(0.5, -0.3)]
    with pytest.raises(ValueError):
        OH.options({"grasp_back_offs_m": [0.9]})


def test_overhang_grasp_stance_needs_the_pre_grasp_fingertip_in_frame(monkeypatch):
    """oracle-fix 2026-11-05 (hb16_desk_s3: the program check rejected the pre-grasp from B backed off 0.10 m at v 1058.7): the grasp
    stance passes the program check's fingertip gate on the pre-grasp and the approach's end, else the next back-off is tried."""
    from ispatialgym.sampling.v2 import program_check as PCK

    _first_passing_choose(monkeypatch)
    m = _manip()
    near = m.shifted(stand_off=float(m.stand_off) + 0.10, name="pick")
    far = m.shifted(stand_off=float(m.stand_off) + 0.20, name="pick")
    s = _Sampler(ikc=_IKC())
    s.backed_off = [near, far]
    gated = []

    def frame(sampler, stance, name, items, **kw):
        gated.append((stance, name, [k for k, _ in items]))
        if stance is near:
            return {"key": "pre_grasp", "index": 0, "n": 1, "u": 384.6, "v": 1058.7, "depth_m": 0.425}
        return None

    monkeypatch.setattr(PCK, "fingertips_in_frame", frame)
    orig = IL.IORSLarge._overhang_grasp

    def grasp(self, st, arm, g, obj_pose, kw, twins, cam):
        return "exact IK failed at pose 1 of 2" if st.name == "manipulation" else orig(self, st, arm, g, obj_pose, kw, twins, cam)

    s._overhang_grasp = grasp.__get__(s)
    pick, why = s._overhang_pick(m, {"right": m}, ["right"], SimpleNamespace(name="grasp_5", pose=(G5_POS, G5_QUAT), meta={"width": G5_WIDTH}),
                                 _aligned_yaw(), th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())
    assert pick is not None, why
    assert pick["grasp_stance"] is far
    assert gated[0] == (near, "grasp", ["pre_grasp", "approach"]) and gated[1][0] is far
    # every back-off out of frame: the rung fails with the gate's reason
    monkeypatch.setattr(PCK, "fingertips_in_frame", lambda *a, **k: {"key": "pre_grasp", "index": 0, "n": 1, "u": 1.0, "v": 1070.0, "depth_m": 0.4})
    s.__dict__.pop("_overhang_grasp_stance_memo", None)
    pick, why = s._overhang_pick(m, {"right": m}, ["right"], SimpleNamespace(name="grasp_5", pose=(G5_POS, G5_QUAT), meta={"width": G5_WIDTH}),
                                 _aligned_yaw(), th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())
    assert pick is None and "exact IK failed" in why


def test_carry_track_oracle_takes_the_program_checks_finish_bound():
    """oracle-fix 2026-11-05 (hb16_desk_s1: the carry at C stalled at 9/45 and was finished 1.12 rad away): ``generation.carry_track_oracle``
    tracks the carry leg with the program check's continuation acceptance; off, the tracking keyword arguments are as before."""
    from ispatialgym.sampling.v2.ik import IKChecker

    s = SimpleNamespace(gp=_GP(), ikc=IKChecker)
    assert IL.IORSLarge._carry_track_kw(s) == {}
    s.gp = _GP(carry_track_oracle=True)
    assert IL.IORSLarge._carry_track_kw(s) == {"finish_max_excursion": 0.6, "soft_pos_tol": 0.01, "soft_rot_tol": 0.03, "max_iters": IKChecker.ORACLE_DIFF_IK_MAX_ITERS}
    # oracle-fix 2026-11-05 (generation.carry_finish_max_remaining_m): the stall finish bound goes along
    s.gp = _GP(carry_track_oracle=True, carry_finish_max_remaining_m=0.05)
    assert IL.IORSLarge._carry_track_kw(s)["finish_max_remaining_m"] == pytest.approx(0.05)


def test_track_chain_refuses_a_stall_finish_far_from_the_end():
    """oracle-fix 2026-11-05 (hbD7: the carry at C stalled at 8/53 with 0.38 m left, the IK finish passed the check, the oracle's planned
    finish next to the bookcase failed): with ``finish_max_remaining_m`` a stall leaving more than that to the chain's end fails the chain."""
    from ispatialgym.sampling.v2.ik import IKChecker

    rot = th.tensor([0.0, 0.0, 0.0, 1.0])

    class _Kit:
        robot = SimpleNamespace(arm_control_idx={"right": th.arange(7)})
        stats: Dict[str, Any] = {}

        def _ensure_synced(self):
            pass

        def fk(self, frames, q):
            return {f: (th.tensor([float(q[0]), 0.0, 1.0]), rot) for f in frames}

        def ik(self, pose, q, arm="right"):
            out = q.clone()
            out[0] = float(pose[0][0])
            return out

        def collision_free(self, q, margin=0.0):
            return th.ones(1, dtype=th.bool)

    ikc = IKChecker.__new__(IKChecker)
    ikc.kit = _Kit()
    ikc.ignoring = lambda objs: contextlib.nullcontext()
    ikc._solve_ik_path = lambda *a, **k: SimpleNamespace(success=False, q_traj=th.tensor([[0.12] + [0.0] * 6]), status="differential IK did not converge",
                                                         diagnostics={"stage": "diff_ik", "pos_err": 0.012, "fail_index": 8})
    chain = [(th.tensor([0.0, 0.0, 1.0]), rot), (th.tensor([0.5, 0.0, 1.0]), rot)]
    q_end, why = ikc.track_chain(th.zeros(7), chain, "right", soft_stall_m=0.03, finish_max_remaining_m=0.05)
    assert q_end is None and "the stall leaves 0.38 m" in why and ikc.last_track_stall["remaining_m"] == pytest.approx(0.38)


def test_carry_track_at_c_kinds(monkeypatch):
    """``_carry_track_at_c`` (the carry leg at C, per carry option under ``carry_track_oracle``): ok / untrackable (the tracking or the drop
    IK from its end fails) / branch (the pre-place witness more than ``carry_finish_max_rad`` from the tracked end)."""
    monkeypatch.setattr(IL, "attached", lambda *a, **k: contextlib.nullcontext())
    seen = {}

    class _Ikc(_IKC):
        FINISH_MAX_EXCURSION, ORACLE_SOFT_POS_TOL, ORACLE_SOFT_ROT_TOL, ORACLE_DIFF_IK_MAX_ITERS = 0.6, 0.01, 0.03, 100

        def track_chain(self, q, wps, arm, **kw):
            seen.update(kw)
            seen.setdefault("chains", []).append(len(wps))
            fails = self.track_fails if len(wps) == 2 else getattr(self, "via_fails", True)
            return (None, "approach_untrackable: stalled") if fails else (th.zeros(28), "ok")

        def exact(self, poses, stance, arm, q_seed=None, **kw):
            return IKResult(True, th.full((28,), self.swing), "", 2)

    ikc = _Ikc()
    ikc.track_fails, ikc.swing = False, 0.2
    s = SimpleNamespace(gp=_GP(carry_track_oracle=True), ikc=ikc, carry_arm_q={"hb": th.zeros(7)}, _floor_relaxed=contextlib.nullcontext,
                        kit=SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": th.arange(7)}), fk=lambda links, q: {links[0]: (th.zeros(3), C.IDENTITY_QUAT)}),
                        obj=lambda role: role, num=lambda key, default=None: default)
    s._carry_track_kw = lambda: IL.IORSLarge._carry_track_kw(s)
    args = (None, "right", "hb", None, SimpleNamespace(pose=None), (th.zeros(3), C.IDENTITY_QUAT), (th.zeros(3), C.IDENTITY_QUAT), [])
    q_end, swing, why, kind, via = IL.IORSLarge._carry_track_at_c(s, *args)
    assert kind == "ok" and why is None and via is None and swing == pytest.approx(0.2) and seen["finish_max_excursion"] == 0.6
    ikc.swing = 0.7
    assert IL.IORSLarge._carry_track_at_c(s, *args)[3] == "branch"
    ikc.track_fails, ikc.swing = True, 0.2
    q_end, swing, why, kind, via = IL.IORSLarge._carry_track_at_c(s, *args)
    assert kind == "untrackable" and q_end is None and "stalled" in why and via is None and seen["chains"][-1] == 2  # no waypoint search when off


def test_carry_via_search_at_c(monkeypatch):
    """oracle-fix 2026-11-05 (generation.carry_via_search; 4 / 4 hardback place checks stalled on the straight carry at C): a straight carry leg
    that does not track is retried through the carry waypoints of ``CARRY_VIAS_AT_C`` (rotate first at the carry pose, the raised / shifted
    midpoint, translate first), each with every carry pose's fingertip in C's image; the first that tracks is returned for ``grasps[k].carry``."""
    from ispatialgym.sampling.v2 import program_check as PCK

    monkeypatch.setattr(IL, "attached", lambda *a, **k: contextlib.nullcontext())
    framed = []
    monkeypatch.setattr(PCK, "fingertips_in_frame", lambda sampler, st, name, items, **kw: framed.append(len(items[0][1])) or (None if len(framed) > 1 else {"key": "carry"}))
    seen = {}

    class _Ikc(_IKC):
        FINISH_MAX_EXCURSION, ORACLE_SOFT_POS_TOL, ORACLE_SOFT_ROT_TOL, ORACLE_DIFF_IK_MAX_ITERS = 0.6, 0.01, 0.03, 100

        def track_chain(self, q, wps, arm, **kw):
            seen.setdefault("chains", []).append([tuple(round(float(v), 3) for v in w[0]) for w in wps])
            return (None, "approach_untrackable: stalled at 13/40") if len(wps) == 2 or self.via_fails else (th.zeros(28), "ok")

        def exact(self, poses, stance, arm, q_seed=None, **kw):
            return IKResult(True, th.full((28,), 0.1), "", 2)

    ikc = _Ikc()
    ikc.via_fails = False
    h0 = (th.tensor([0.0, 0.0, 1.0]), C.IDENTITY_QUAT.clone())
    pre = (th.tensor([0.6, 0.0, 1.0]), C.quat_from_axis_angle(th.tensor([1.0, 0.0, 0.0]), math.pi / 2))
    s = SimpleNamespace(gp=_GP(carry_track_oracle=True, carry_via_search=True), ikc=ikc, carry_arm_q={"hb": th.zeros(7)}, _floor_relaxed=contextlib.nullcontext,
                        kit=SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": th.arange(7)}), fk=lambda links, q: {links[0]: h0}),
                        obj=lambda role: role, num=lambda key, default=None: default, CARRY_VIAS_AT_C=IL.IORSLarge.CARRY_VIAS_AT_C)
    for m in ("_carry_track_kw", "_carry_vias_at_c", "_carry_leg_poses"):
        setattr(s, m, getattr(IL.IORSLarge, m).__get__(s) if m != "_carry_leg_poses" else IL.IORSLarge._carry_leg_poses)
    args = (None, "right", "hb", None, SimpleNamespace(pose=None), pre, pre, [])
    q_end, swing, why, kind, via = IL.IORSLarge._carry_track_at_c(s, *args)
    # the first waypoint (rotate first at the carry pose) is out of frame; the second (raised 5 cm there) tracks
    assert kind == "ok" and via["label"] == "start raise 0.05 lateral +0.00 turn 1.0"
    assert seen["chains"][1] == [(0.0, 0.0, 1.0), (0.0, 0.0, 1.05), (0.6, 0.0, 1.0)] and C.quat_angle(via["pose"][1], pre[1]) < 1e-5
    # the carry leg's poses as Place._carry_leg builds them: 3 to the waypoint at the start, a pose every 5 cm over the 0.6 m (float32) after it
    assert framed[0] == 3 + math.ceil(float(th.linalg.norm(pre[0] - h0[0])) / 0.05)
    ikc.via_fails = True
    q_end, swing, why, kind, via = IL.IORSLarge._carry_track_at_c(s, *args)
    assert kind == "untrackable" and via is None and "none of 12 carry waypoints passes" in why and "stalled at 13/40" in why


def test_overhang_fail_memo_key():
    """oracle-fix 2026-11-05 (generation.overhang_fail_memo): a failed overhang pick is remembered under B's pose and trunk, the arms' stances,
    the row, the yaw bucket, the focus cell and the support height; off, there is no key (nothing is remembered)."""
    m = _manip()
    g = SimpleNamespace(name="grasp_0")
    s = SimpleNamespace(gp=_GP())
    args = (m, {"right": m}, ["right"], g, 4.32, th.tensor([1.0, 0.0]), Z_TOP, False)
    assert IL.IORSLarge._overhang_memo_key(s, *args) is None
    s.gp = _GP(overhang_fail_memo=True)
    k = IL.IORSLarge._overhang_memo_key(s, *args)
    assert k == IL.IORSLarge._overhang_memo_key(s, *args) and hash(k) is not None
    moved = m.shifted(stand_off=float(m.stand_off) + 0.1, name="manipulation")
    assert IL.IORSLarge._overhang_memo_key(s, moved, {"right": moved}, *args[2:]) != k
    assert IL.IORSLarge._overhang_memo_key(s, *args[:4], 4.71, *args[5:]) != k
    assert IL.IORSLarge._overhang_memo_key(s, *args[:3], SimpleNamespace(name="grasp_1"), *args[4:]) != k


def test_overhang_push_branch_tracks_the_straight_push_from_where_the_oracle_leaves(monkeypatch):
    """oracle-fix 2026-11-05 (generation.overhang_push_branch_check; hb21_desk_s55: the flipped push witness was moved to the stance branch
    and the oracle's straight push stalled there): the chosen push chain is tracked as the oracle's straight push from the witness, or from
    the stance posture's branch when the witness is flipped; a chain that does not track is refused, off nothing is checked."""
    from ispatialgym.sampling.v2 import program_check as PCK

    pre, start, mid, goal = [((0.1 * i, 0.0, 0.9), (0.0, 0.0, 0.0, 1.0)) for i in range(4)]
    q_wit, q_raw = th.zeros(28), th.full((28,), 3.13)
    chosen = SimpleNamespace(qs=[q_wit, None, None, None], proposal=SimpleNamespace(chain=[pre, start, mid, goal]))
    calls, state = [], {"flipped": None, "ok": True}
    monkeypatch.setattr(PCK, "_flipped_push_witness", lambda s, chain, st, arm, q_first, fingers: state["flipped"])

    def track(s, arm, q_from, waypoints):
        calls.append((q_from, list(waypoints)))
        return state["ok"], "straight push tracked" if state["ok"] else "approach_untrackable: straight push replay (28 dense poses): waypoint 21/28", {}

    monkeypatch.setattr(PCK, "track_line_push", track)
    s = SimpleNamespace(gp=_GP(), kit=SimpleNamespace(stats={}))
    run = lambda: IL.IORSLarge._overhang_push_branch(s, _manip(), "left", chosen, 0.08)
    assert run() == "" and calls == []
    s.gp = _GP(overhang_push_branch_check=True)
    assert run() == "" and calls[-1][0] is q_wit and calls[-1][1] == [start, mid, goal]
    state.update(flipped=(3.13, q_raw), ok=False)
    why = run()
    assert calls[-1][0] is q_raw and "stance posture's branch" in why and "3.13 rad" in why and "21/28" in why
    assert s.kit.stats == {"pick_overhang.push_flipped": 1, "pick_overhang.push_line_fails": 1}
    # generation.push_flipped_keep_recorded: the straight push tracked from the witness keeps the chain (as the program check keeps it)
    s.gp = _GP(overhang_push_branch_check=True, push_flipped_keep_recorded=True)
    state["track"] = lambda q: q is q_wit
    monkeypatch.setattr(PCK, "track_line_push", lambda s_, arm, q_from, w: (calls.append((q_from, list(w))) or state["track"](q_from), "tracked or not", {}))
    assert run() == "" and [c[0] for c in calls[-2:]] == [q_raw, q_wit] and s.kit.stats["pick_overhang.push_flipped_kept"] == 1
    state["track"] = lambda q: False
    assert "from the witness too" in run()


def test_overhang_push_chain_light_sweep(monkeypatch):
    """A chain that only solves with the support out of the world is accepted under ``support_touch_m`` by the rule of
    ``approach_finger_support_touch`` - the whole body free with the support out, every non-finger link clear of the world, the fingers
    within the allowance of it - and recorded (the program check then ignores the supports too); deeper fingers are refused."""
    _first_passing_choose(monkeypatch)
    from ispatialgym.oracle import r1pro_model as M

    fingers = set(M.FINGER_LINKS["right"])
    depth = {"v": 0.003}

    def links_clear(q, links, margin):
        # the fingers reach ``depth`` into the support; every other link stands clear of it
        d = depth["v"] if set(links) <= fingers else -1.0
        return th.full((q.shape[0],), -d >= margin - 1e-9, dtype=th.bool)

    def run():
        s = _Sampler(ikc=_IKC(chain_ok=lambda chain, ignored: "countertop_kelzer_0" in ignored))
        s.ikc._non_finger_links = lambda arm: ["right_arm_link6", "right_gripper_link"]
        s.kit.collision_free = lambda q, margin=0.0: th.ones(q.shape[0], dtype=th.bool)
        s.kit.links_clear = links_clear
        pick, why = s._overhang_pick(_manip(), {"right": _manip()}, ["right"], SimpleNamespace(name="grasp_5", pose=(G5_POS, G5_QUAT), meta={"width": G5_WIDTH}),
                                     _aligned_yaw(), th.tensor([1.0, 0.0]), Z_TOP, False, s.pick_overhang_opts())
        return s, pick, why

    monkeypatch.setattr(IL.O, "chain_qs", lambda ikc, arm, chain: [th.zeros(28) for _ in chain])
    s, pick, why = run()
    assert pick is not None, why
    assert pick["chain"]["support_touch_m"] == pytest.approx(0.003, abs=1e-6)
    grasp = SimpleNamespace(extras={})
    s.overhang_records(pick, grasp)
    assert s.sidecar.contacts[OH.CONTACT_SLOT].to_sidecar()["ignore_every"] == ["plate.n.04_1", "supports"]
    depth["v"] = 0.009  # deeper than 5 mm: refused
    s, pick, why = run()
    assert pick is None and "into the support" in why


def test_overhang_key_off_is_today(monkeypatch):
    s = _Sampler(pick_overhang_push=None)
    assert s.pick_overhang_opts() is None
    s.gp.generation["pick_overhang_push"] = {"mode": "sometimes"}
    with pytest.raises(IL.Reject):
        s.pick_overhang_opts()


# ----------------------------------------------------------------------------------------------------------------
# the program, the program check, the oracle
# ----------------------------------------------------------------------------------------------------------------


def _sidecar_with_pick_push(**over):
    d = json.load(open(FIXTURE))
    g = d["grasps"][0]
    post = {"position": [13.9, 0.36, 0.913], "orientation": [0.0, 0.0, 0.19867, 0.98007]}
    g["eef_in_object"] = {"position": G5_POS.tolist(), "orientation": G5_QUAT.tolist()}
    g["pick_overhang"] = {"contact": "push_pick", "post_push_pose": post, "overhang_m": 0.025, "push_m": 0.06, "finger_clearance_m": 0.01, "yaw_free": True}
    g["ignore_last"], g["ignore_every"] = [], []
    push = {"arm": g["arm"], "object": g["object"], "pre": {"position": [13.9, 0.1, 1.02], "orientation": [0.70711, 0.70711, 0.0, 0.0]},
            "init": {"position": [13.9, 0.1, 0.92], "orientation": [0.70711, 0.70711, 0.0, 0.0]},
            "mid": {"position": [13.9, 0.135, 0.92], "orientation": [0.70711, 0.70711, 0.0, 0.0]},
            "goal": {"position": [13.9, 0.17, 0.92], "orientation": [0.70711, 0.70711, 0.0, 0.0]}, "direction": [0.0, 1.0, 0.0], "distance_m": 0.06,
            "stance": "manipulation", "kind": "pick_overhang", "support": "countertop.n.01_1", "fingers": "open", "fingers_opening_m": 0.05, "post_push_pose": post,
            "ignore_every": [g["object"]], "ignore_last": [], "overhang_m": 0.025}
    push.update(over)
    d["contacts"] = {"push_pick": push}
    d["language"]["objects"]["countertop.n.01_1"] = {"display_name": "countertop", "descriptor": "the countertop"}
    return d


def test_program_pushes_retreats_then_grasps():
    sc = Sidecar.from_dict(_sidecar_with_pick_push())
    assert sc.problems() == []
    actions = expand(sc.variant, sc)
    assert [a.name for a in actions] == ["transition", "push", "retreat", "grasp", "lift", "navigate", "transition", "place", "retreat"]
    push, retreat = actions[1], actions[2]
    assert push.kind == "pick_overhang" and push.touches == ("countertop.n.01_1",) and push.contact.distance_m == pytest.approx(0.06)
    assert retreat.after is push and retreat.arm == push.contact.arm and not retreat.release
    fine, numbers = push.annotate(sc)
    assert fine == "push the plate on the counter along the countertop until it overhangs the edge, with the open right hand" and "6.0 cm backward" in numbers and "overhangs the edge by 2.5 cm" in numbers
    for bad, match in (({"arm": "left"}, "arm"), ({"object": "bowl.n.01_1"}, "object"), ({"kind": "push"}, "kind")):
        with pytest.raises(SidecarError, match=match):
            expand(sc.variant, Sidecar.from_dict(_sidecar_with_pick_push(**bad)))
    assert retreat.escape_up_m is None
    # oracle-fix 2026-11-05 (generation.overhang_retreat_up_m): the recorded escape height goes to the retreat
    d = _sidecar_with_pick_push()
    d["contacts"]["push_pick"]["retreat_up_m"] = 0.15
    assert expand(sc.variant, Sidecar.from_dict(d))[2].escape_up_m == pytest.approx(0.15)
    # without the record: today's program
    d = _sidecar_with_pick_push()
    d.pop("contacts")
    assert [a.name for a in expand(sc.variant, Sidecar.from_dict(d))][:3] == ["transition", "grasp", "lift"]


def test_program_of_the_two_stance_overhang_pick():
    """The grasp from the stance ``pick`` (B backed off along its axis): push and retreat at B, the recorded transition B -> pick, the grasp
    and the lift there, the carry from pick to the placement stance."""
    d = _sidecar_with_pick_push()
    man = d["stances"]["manipulation"]
    d["stances"]["pick"] = dict(man, base_pose2d=[man["base_pose2d"][0], man["base_pose2d"][1] + 0.15, man["base_pose2d"][2]], stand_off=man["stand_off"] + 0.15)
    d["grasps"][0]["stance"] = "pick"
    d["transitions"].append({"from": "manipulation", "to": "pick", "stand_off": 0.7, "order": ["drive-back", "trunk", "drive-in"]})
    for rec in d["paths"] + d["transitions"]:
        if rec["from"] == "manipulation" and rec["to"] == "placement":
            rec["from"] = "pick"
    sc = Sidecar.from_dict(d)
    assert sc.problems() == []
    actions = expand(sc.variant, sc)
    assert [a.name for a in actions] == ["transition", "push", "retreat", "transition", "grasp", "lift", "navigate", "transition", "place", "retreat"]
    assert (actions[3].transition.frm, actions[3].transition.to) == ("manipulation", "pick") and actions[6].path.frm == "pick"
    d["paths"][0]["from"] = "manipulation"  # the carry must leave from the grasp stance
    with pytest.raises(SidecarError, match="'pick'"):
        expand(sc.variant, Sidecar.from_dict(d))


def test_grasp_after_the_push_is_taken_at_the_recorded_yaw():
    sc = Sidecar.from_dict(_sidecar_with_pick_push())
    grasp = next(a for a in expand(sc.variant, sc) if a.name == "grasp")

    class _State:
        def __init__(self, yaw):
            self.yaw = yaw

        def object_pose(self, name):
            return th.tensor([13.9, 0.37, 0.913]), C.quat_from_yaw(self.yaw)

    rec_yaw = C.yaw_of(th.tensor([0.0, 0.0, 0.19867, 0.98007]))
    turned = grasp.targets(sc, _State(rec_yaw + 0.3))["grasp"]
    expect = C.grasp_pose_from_transform((th.tensor([13.9, 0.37, 0.913]), C.quat_from_yaw(rec_yaw)), (G5_POS, G5_QUAT))
    assert float(th.linalg.norm(th.tensor(turned.position) - expect[0])) < 1e-5  # the push turned the plate 0.3 rad: the pinch stays put
    d = _sidecar_with_pick_push()
    d["grasps"][0]["pick_overhang"]["yaw_free"] = False
    sc2 = Sidecar.from_dict(d)
    grasp2 = next(a for a in expand(sc2.variant, sc2) if a.name == "grasp")
    plain = C.grasp_pose_from_transform((th.tensor([13.9, 0.37, 0.913]), C.quat_from_yaw(rec_yaw + 0.3)), (G5_POS, G5_QUAT))
    assert float(th.linalg.norm(th.tensor(grasp2.targets(sc2, _State(rec_yaw + 0.3))["grasp"].position) - plain[0])) < 1e-5
    # no support lift for the overhang grasp
    ctx = {}
    targets = grasp.targets(sc, _State(rec_yaw))
    assert EX.lift_grasp_off_support(grasp, None, _State(rec_yaw), targets, ctx, EX.ExecuteConfig()) is targets
    assert ctx["grasp_support_lift"]["applied"] is False and "overhang" in ctx["grasp_support_lift"]["reason"]


def test_program_check_advances_the_plate_to_the_pushed_pose():
    from ispatialgym.sampling.v2 import program_check as PC

    sc = Sidecar.from_dict(_sidecar_with_pick_push())
    push = next(a for a in expand(sc.variant, sc) if a.name == "push")

    class _Obj:
        def __init__(self):
            self.pose = (th.tensor([13.9, 0.30, 0.913]), th.tensor([0.0, 0.0, 0.0, 1.0]))

        def get_position_orientation(self):
            return self.pose

        def set_position_orientation(self, p, q):
            self.pose = (p, q)

        def keep_still(self):
            pass

    obj = _Obj()
    sampler = SimpleNamespace(objects={"plate.n.04_1": obj}, kit=SimpleNamespace(sync_world=lambda: None))
    tracker = PC._Tracker(SidecarDraft(f"{SUBTASK}-plate-drop_in_sink-countertop", "inter_object_relation_satisfaction"))
    tracker.after(push, {}, sampler)
    assert [round(float(v), 4) for v in obj.pose[0]] == [13.9, 0.36, 0.913]
    tracker.restore_moved(sampler)
    assert [round(float(v), 4) for v in obj.pose[0]] == [13.9, 0.30, 0.913]


def test_program_check_retreats_after_the_overhang_push_as_the_contact_escape(monkeypatch):
    """oracle-fix 2026-11-05 (hardback hb14_desk_s3: the recorded -a leg at a joint limit): the retreat after the overhang push is judged as
    the oracle runs it - the contact escape from where the push left the arm (``replay_contact_escape``) - not as the recorded leg."""
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2 import program_check as PC

    sc = Sidecar.from_dict(_sidecar_with_pick_push())
    actions = expand(sc.variant, sc)
    push, retreat = actions[1], actions[2]
    assert PC._pick_overhang_push(push) and not PC._pick_overhang_push(retreat)
    plate, counter = object(), object()
    idx = list(range(7))
    calls = {"exact": 0, "escape": []}

    class _IKC:
        def q_locked(self, stance, fingers=None, **kw):
            return th.zeros(28)

        def reaches(self, q, pose, arm):
            return float(q[0]) == 0.5  # only the push's tracked end is at its goal

        def exact(self, poses, stance, arm, **kw):
            calls["exact"] += 1
            return IKResult(True, th.full((28,), 0.25), "", 1)

    sampler = SimpleNamespace(objects={push.contact.object: plate}, support_objects=lambda: [counter], ikc=_IKC(),
                              kit=SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": idx, "left": idx})))
    q_end = th.full((28,), 0.75)

    def _escape(sampler_, arm, q, obj, supports, ignore, dist, start="", escape_up_m=None):
        calls["escape"].append((arm, float(q[0]), obj, list(supports), list(ignore), dist, start))
        calls["up"] = escape_up_m
        return calls.get("status", "replayed"), "approach_untrackable: push replay - stalls", {"distance_m": dist}, q_end

    monkeypatch.setattr(PC, "replay_contact_escape", _escape)
    targets = retreat.targets(sc, None)
    n, w = PC.check_action_targets(sc, sampler, retreat, targets, "manipulation", None, q_end_prev=th.full((28,), 0.5))
    assert n == 1 and w["__arm_end_q__"] is q_end and calls["exact"] == 0
    arm, q0, obj, sup, ign, dist, start = calls["escape"][-1]
    assert (arm, q0, obj, sup, dist) == (push.contact.arm, 0.5, plate, [counter], 0.10) and ign == [plate, counter] and "overhang push" in start
    assert calls["up"] is None  # no retreat_up_m recorded: the oracle's 5 cm escape
    # no tracked end at the push's goal: a fresh IK of the push's end pose
    PC.check_action_targets(sc, sampler, retreat, targets, "manipulation", None, q_end_prev=th.full((28,), 0.1))
    assert calls["exact"] == 1 and calls["escape"][-1][1] == 0.25
    calls["status"] = "failed"
    with pytest.raises(Reject, match="push replay - stalls"):
        PC.check_action_targets(sc, sampler, retreat, targets, "manipulation", None, q_end_prev=th.full((28,), 0.5))


def test_executor_pick_push_measures_its_own_displacement():
    class _S:
        def pose_target(self, name):
            return SimpleNamespace(position=[0.0, 0.0, 0.0], frame="world", position_tolerance=0.03)

        def object_pose(self, name):
            return th.tensor([0.0, 0.05, 0.9]), th.tensor([0.0, 0.0, 0.0, 1.0])

    st = EX._RecordedPushState(_S(), "plate.n.04_1")
    assert st.pose_target("plate.n.04_1") is None and st.pose_target("bowl") is not None
    wps = [(th.tensor([0.0, 0.0, 0.92]), th.tensor([0.0, 0.0, 0.0, 1.0])), (th.tensor([0.0, 0.035, 0.92]), th.tensor([0.0, 0.0, 0.0, 1.0])),
           (th.tensor([0.0, 0.07, 0.92]), th.tensor([0.0, 0.0, 0.0, 1.0]))]
    geom = EX.pick_push_geometry(EX.push_geometry(wps, [0.0, 1.0, 0.0]), 0.06)
    assert geom.kind == "line" and geom.distance == pytest.approx(0.06)
    pose0 = (th.tensor([0.0, 0.0, 0.9]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    res, src = EX.push_residual(geom, st, "plate.n.04_1", pose0)
    assert src == "line_distance" and res == pytest.approx(0.01)  # 6 cm to push, 5 cm done; never measured against the sink goal
    assert EX._push_stop_margin(st, "plate.n.04_1", geom, EX.ExecuteConfig(), {}) is None  # no stop-at-goal: the pad drives its line


def test_carry_vias_table_is_shared_with_the_oracle():
    """oracle-fix 2026-11-05: the sampler's carry waypoints at C and the oracle's online carry search read the same candidates."""
    from ispatialgym.oracle import actions as A

    assert tuple(IL.IORSLarge.CARRY_VIAS_AT_C) == tuple(A.CARRY_VIAS)
    h0 = (th.tensor([0.0, 0.0, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    pre = (th.tensor([0.4, 0.0, 1.0]), th.tensor([0.0, 0.0, 0.70711, 0.70711]))
    s = SimpleNamespace(CARRY_VIAS_AT_C=IL.IORSLarge.CARRY_VIAS_AT_C)
    mine = IL.IORSLarge._carry_vias_at_c(s, h0, pre)
    theirs = A.carry_via_candidates(h0, pre)
    assert [l for l, _ in mine] == [l for l, _ in theirs]
    assert all(th.allclose(a[1][0], b[1][0]) and th.allclose(a[1][1], b[1][1]) for a, b in zip(mine, theirs))
    legs = A.carry_leg_poses([h0, theirs[4][1], pre])
    assert len(legs) == len(IL.IORSLarge._carry_leg_poses([h0, theirs[4][1], pre])) and th.allclose(th.as_tensor(legs[-1][0]).float(), pre[0])


def test_oracle_carry_online_replaces_an_untrackable_carry_leg():
    """oracle-fix 2026-11-05 (grasps[k].carry_online; diagnostic hbD4: the carry checked from the sampler's lift configuration stalled at
    10/55 from the real one): the recorded leg is dry-run from where the arm is; when it does not track, the straight leg and the carry
    waypoints are tried in order (fingertips in the head camera's frame, collision-checked dry run) and the first that passes replaces
    the leg; without the flag nothing changes."""
    from ispatialgym.oracle import execute as E

    h0 = (th.tensor([0.0, 0.0, 1.0]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    pre = (th.tensor([0.4, 0.0, 1.0]), th.tensor([0.0, 0.0, 0.70711, 0.70711]))
    calls = []

    class _Backend:
        def __init__(self, ok):
            self.ok = ok

        def update_world(self, ignore_objects=()):
            calls.append(("world", len(ignore_objects)))

        def q_full(self):
            return th.zeros(28)

        def solve_ik_path(self, emb, wps, check_collisions=False, **kw):
            last = list(wps[-1].values())[0]
            mid_z = max(float(list(w.values())[0][0][2]) for w in wps)
            calls.append(("solve", len(wps), check_collisions, round(mid_z, 3)))
            return SimpleNamespace(success=self.ok(len(wps), check_collisions, mid_z), status="stall at 10/55")

    def motions(ok):
        m = SimpleNamespace(backend=_Backend(ok), cfg=SimpleNamespace(diff_ik_pos_tol=0.002, diff_ik_rot_tol=0.01, diff_ik_max_iters=30, diff_ik_soft_pos_tol_free=0.01,
                                                                    diff_ik_soft_rot_tol=0.03), eef_pose=lambda arm: h0)
        m._densify = lambda start, wps: list(wps)
        return m

    action = SimpleNamespace(object="hardback.n.01_1", arm="right", carry_waypoints=lambda sc: [])
    state = SimpleNamespace(eef_pose=lambda arm: h0, robot=SimpleNamespace(arm_control_idx={"right": th.arange(11, 25, 2)}))
    targets = {"pre_place": pre}
    recorded = [pre]

    def ctx(flag):
        g = SimpleNamespace(object="hardback.n.01_1", arm="right", extra={"carry_online": True} if flag else {})
        mon = SimpleNamespace(camera=SimpleNamespace(in_frame=lambda cam, p, margin: True), cfg=SimpleNamespace(fov_margin=0.05), head_camera_pose=lambda: None)
        return {"sidecar": SimpleNamespace(grasps=[g]), "monitors": mon}

    # off: nothing is dry-run
    c = ctx(False)
    assert E._carry_online(action, motions(lambda *a: False), state, "right", targets, c, recorded, []) is None and calls == [] and "carry_online" not in c
    # the recorded leg tracks: it stands
    c = ctx(True)
    assert E._carry_online(action, motions(lambda n, coll, z: True), state, "right", targets, c, recorded, []) is None and c["carry_online"] == {"recorded": "tracks"}
    # the recorded leg (1 pose) and the straight leg stall; the first candidate raised 5 cm tracks collision-free -> it replaces the leg
    calls.clear()
    c = ctx(True)
    kind, legs = E._carry_online(action, motions(lambda n, coll, z: z >= 1.049), state, "right", targets, c, recorded, ["bookcase"])
    assert kind == "legs"
    assert c["carry_online"]["chosen"] == "start raise 0.05 lateral +0.00 turn 1.0" and c["carry_online"]["tier"] == "in frame, collision-free"
    tried = c["carry_online"]["tried"]
    assert [t[0] for t in tried[:3]] == ["straight", "start raise 0.00 lateral +0.00 turn 1.0", "start raise 0.05 lateral +0.00 turn 1.0"]
    assert [t[2] for t in tried[:3]] == [False, False, True] and tried[2][3] is True
    assert th.allclose(th.as_tensor(legs[-1][0]).float(), pre[0]) and calls[0] == ("solve", 1, False, 1.0) and ("world", 1) in calls
    # tracked but never collision-free: nothing replaces the leg (hbD6: the colliding straight leg jammed the arm)
    c = ctx(True)
    assert E._carry_online(action, motions(lambda n, coll, z: (not coll) and z >= 1.099), state, "right", targets, c, recorded, []) is None
    assert c["carry_online"]["chosen"] is None and any(t[2] and not t[3] for t in c["carry_online"]["tried"])
    # nothing passes: the recorded leg runs
    c = ctx(True)
    assert E._carry_online(action, motions(lambda *a: False), state, "right", targets, c, recorded, []) is None and c["carry_online"]["chosen"] is None
    from ispatialgym.oracle import actions as A

    assert len(c["carry_online"]["tried"]) == 1 + len(A.CARRY_VIAS)
    # the recorded leg tracks from the program check's lift witness: the arm returns there first ("repose"), nothing else is tried
    c = ctx(True)
    c["sidecar"].extra = {"ik_witnesses": [{"action": 6, "name": "lift", "arm": "right", "targets": {"lift": {"q": [0.1] * 28}}}]}
    m = motions(lambda *a: False)
    m.backend.fk = lambda frames, q: {frames[0]: h0}
    plain = m.backend.solve_ik_path
    m.backend.solve_ik_path = lambda emb, wps, check_collisions=False, q_start=None, **kw: SimpleNamespace(success=q_start is not None, status="ok")
    kind, (q7, vias) = E._carry_online(action, m, state, "right", targets, c, recorded, [])
    assert kind == "repose" and q7.shape == (7,) and float(q7[0]) == pytest.approx(0.1) and vias == [] and c["carry_online"]["repose"]["jump_rad"] == pytest.approx(0.1)
    # repose=False (the return failed): the candidates from the live hand
    c2 = ctx(True)
    c2["sidecar"].extra = c["sidecar"].extra
    assert E._carry_online(action, m, state, "right", targets, c2, recorded, [], repose=False) is None and "repose" not in c2["carry_online"]


def test_overhang_push_drift_shifts_the_predicted_pose_away_from_the_pushing_arm():
    """oracle-fix 2026-11-05 (generation.overhang_push_drift; hbD3 / hbD7: the open-hand push left the hardback 2.4 / 2.65 cm aside, away from
    the pushing arm, turned +0.048 / -0.096 rad): the predicted post-push pose is shifted away from the pushing arm along B's base left axis
    and turned (+ for the left arm, - for the right); off, unchanged."""
    import math as _m

    st = SimpleNamespace(base_pose2d=[0.0, 0.0, 0.0])
    post = (th.tensor([1.0, 0.0, 0.9]), C.quat_from_yaw(0.3).float())
    s = SimpleNamespace(gp=_GP())
    assert IL.IORSLarge._overhang_drifted(s, post, st, "left") is post
    s.gp = _GP(overhang_push_drift={"lateral_m": 0.025, "yaw_rad": 0.07})

    def yaw(q):
        x, y, z, w = (float(v) for v in q)
        return _m.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    pl = IL.IORSLarge._overhang_drifted(s, post, st, "left")
    pr = IL.IORSLarge._overhang_drifted(s, post, st, "right")
    assert th.allclose(pl[0], th.tensor([1.0, -0.025, 0.9]), atol=1e-6) and yaw(pl[1]) == pytest.approx(0.37, abs=1e-5)
    assert th.allclose(pr[0], th.tensor([1.0, 0.025, 0.9]), atol=1e-6) and yaw(pr[1]) == pytest.approx(0.23, abs=1e-5)
    # B turned 90 deg: its left axis is -x
    p90 = IL.IORSLarge._overhang_drifted(s, post, SimpleNamespace(base_pose2d=[0.0, 0.0, _m.pi / 2]), "left")
    assert th.allclose(p90[0], th.tensor([1.025, 0.0, 0.9]), atol=1e-6)
    # snap_max_rad (hb29_desk_s44: yaw -1.963 pushed along -x turned to -1.574, flush with the pad): within the bound the whole offset from the
    # nearest flush orientation; near 45 deg the small turn stands
    s.gp = _GP(overhang_push_drift={"lateral_m": 0.025, "yaw_rad": 0.07, "snap_max_rad": 0.70})
    d = th.tensor([-0.9999969, 0.0024896, 0.0])
    p44 = IL.IORSLarge._overhang_drifted(s, (post[0], C.quat_from_yaw(-1.963).float()), st, "left", d)
    assert yaw(p44[1]) == pytest.approx(-1.963 + 0.3886, abs=2e-3)
    p45 = IL.IORSLarge._overhang_drifted(s, (post[0], C.quat_from_yaw(-0.785).float()), st, "right", d)
    assert yaw(p45[1]) == pytest.approx(-0.785 - 0.07, abs=1e-4)
    assert yaw(IL.IORSLarge._overhang_drifted(s, (post[0], C.quat_from_yaw(-1.963).float()), st, "left")[1]) == pytest.approx(-1.893, abs=1e-4)  # no d: no snap
