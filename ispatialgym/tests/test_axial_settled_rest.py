"""i7-axialrest (owner 2026-11-08, decision 2): the alignment-axial program at the target's SETTLED v2 rest behind
``ISPATIALGYM_AXIAL_SETTLED_REST`` (default off) - simulator-free: switch off = today, an upright rest gives today's placement / goal /
contacts, a tilted rest keeps its tilt at the spawn and the goal, the goal check measures the turn alone."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_library as CL
from ispatialgym.grasping import schema as S
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import contact_poses as CPZ
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.tasks import alignment_axial as AX
from ispatialgym.tests.test_contact_library import _job, _model_dir

ENV = "ISPATIALGYM_AXIAL_SETTLED_REST"
#: tablespoon tqqwys's v2 rest r0_other (roles_all_20261102): tilted 0.0866 rad
SPOON_REST = (-0.0432275, 0.0019804, 0.0001867, 0.9990633)
#: an upright v2 rest (magazine tfovwy: tilt ~1e-5)
UPRIGHT_REST = (-5e-06, 3e-06, 0.0, 1.0)
SPOON_DIMS = (0.177, 0.057, 0.027)


def _unit(q):
    q = th.tensor([float(v) for v in q], dtype=th.float32)
    return q / th.linalg.norm(q)


def _hull(dims=SPOON_DIMS):
    dx, dy, dz = dims
    return th.tensor([[sx * dx / 2, sy * dy / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, dz)], dtype=th.float32)


def _ax(rest=None, h_root=None, front=(1.0, 0.0, 0.0), long=(1.0, 0.0, 0.0)):
    """A stub with the AlignmentAxial methods the settled rest touches."""
    s = SimpleNamespace(rest_quat=None if rest is None else _unit(rest), rest_h_root=h_root, rest_info={}, hull=_hull(), top=SimpleNamespace(z=0.8126, name="t"),
                        front_axis=th.tensor(front), long_axis=th.tensor(long), com_local=th.tensor([0.0, 0.0, 0.01]), kit=SimpleNamespace(stats={}),
                        world=SimpleNamespace(bottom_offset=lambda hull, q: 0.0, floor_z=0.0), choice=SimpleNamespace(models={"target": SimpleNamespace(category="tablespoon", model="tqqwys")}))
    for m in ("_place", "_heading", "settled_goal_quat", "_aligned_yaw", "_map_rest_ok", "_stop_geometry"):
        setattr(s, m, getattr(AX.AlignmentAxial, m).__get__(s))
    return s


def _tilt(q):
    return CPZ.axial_rest_tilt(q)


def _full_err(q, g):
    """The evaluation's full-quaternion orientation error (``eval.geometry.orientation_error`` without equivalences)."""
    from ispatialgym.eval.geometry import orientation_error

    return float(orientation_error(TT._t(q).float(), TT._t(g).float()))


# ---------------------------------------------------------------------------------------------------- switch off = today


def test_the_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert not CPZ.axial_settled_rest_on()
    monkeypatch.setenv(ENV, "1")
    assert CPZ.axial_settled_rest_on()


def test_off_the_placement_is_object_pose_on_at_the_identity(monkeypatch):
    calls = []
    monkeypatch.setattr(TT, "object_pose_on", lambda s, hull, rq, xy, yaw, z: calls.append((rq, yaw, z)) or "old")
    s = _ax(rest=None)
    assert s._place(th.tensor([0.5, 0.1]), 0.3) == "old" and th.equal(calls[0][0], C.IDENTITY_QUAT) and calls[0][1:] == (0.3, 0.8126)


def test_off_the_stop_geometry_and_the_library_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    hull, com_l = _hull(), th.tensor([0.0, 0.0, 0.013])
    pose = (th.tensor([0.6, 0.0, 0.82]), C.quat_from_yaw(0.5))
    end = C.quat_rotate(pose[1], th.tensor([0.06, 0.0, 0.0])) + pose[0]
    com = pose[0] + C.quat_rotate(pose[1], com_l)
    a = CPZ.axial_stop_geometry(hull, com_l, pose, end, com, 1.0, 1.2, 0.83)
    b = CPZ.axial_stop_geometry(hull, com_l, pose, end, com, 1.0, 1.2, 0.83, rest_quat=None)
    assert th.equal(a[0][1], C.quat_from_yaw(1.2)) and th.equal(a[0][1], b[0][1]) and th.equal(a[2], b[2])
    root = str(tmp_path / "glib")
    _model_dir(root, "tablespoon__tqqwys__s1", SPOON_DIMS, rest_q=SPOON_REST, h_root=0.0085)
    job = _job("tablespoon__tqqwys__s1", ["axial_push", "axial_brace"], subtask="alignment-axial",
               params={"axial_push": {"yaws": 1, "long_axis": [1, 0, 0]}, "axial_brace": {"yaws": 1, "long_axis": [1, 0, 0], "clear_turns": True}})
    with pytest.raises(ValueError, match="upright identity rest"):  # off: the tilted rest is refused, as today
        CL.build_job(job, root)
    _model_dir(root, "magazine__tfovwy__s1", (0.20, 0.28, 0.012), rest_q=UPRIGHT_REST, h_root=0.0047)
    jm = _job("magazine__tfovwy__s1", ["axial_push", "axial_brace"], subtask="alignment-axial",
              params={"axial_push": {"yaws": 1, "long_axis": [1, 0, 0]}, "axial_brace": {"yaws": 1, "long_axis": [1, 0, 0], "clear_turns": True}})
    libs = CL.build_job(jm, root)
    assert all("settled_rest" not in f["params"] for lib in libs.values() for f in lib["families"].values())  # off: not in the identity


# ---------------------------------------------------------------------------------------------------- an upright rest: the same


def test_an_upright_rest_gives_the_same_placement_and_goal():
    old, new = _ax(rest=None), _ax(rest=UPRIGHT_REST, h_root=0.0047)
    for yaw in (0.0, 0.7, -2.4):
        q_new = new._place(th.tensor([0.5, 0.1]), yaw)[1]
        assert C.quat_angle(q_new, C.quat_from_yaw(yaw)) < 1e-4
        assert new._place(th.tensor([0.5, 0.1]), yaw)[0][2] == pytest.approx(0.8126 + 0.0047)  # the measured root height
    st = SimpleNamespace(inward=th.tensor([0.6, 0.8]))
    assert new._aligned_yaw(st) == pytest.approx(old._aligned_yaw(st), abs=1e-4)
    for y1, yg in ((0.3, 1.1), (-2.0, -1.2)):
        assert C.quat_angle(new.settled_goal_quat(C.quat_from_yaw(y1), yg), C.quat_from_yaw(yg)) < 1e-4  # today's q_goal = quat_from_yaw(yaw_goal)


def test_an_upright_rest_gives_the_same_contacts(tmp_path, monkeypatch):
    root = str(tmp_path / "glib")
    _model_dir(root, "magazine__tfovwy__s1", (0.20, 0.28, 0.012), rest_q=UPRIGHT_REST, h_root=0.0047)
    jm = _job("magazine__tfovwy__s1", ["axial_push", "axial_brace"], subtask="alignment-axial",
              params={"axial_push": {"yaws": 2, "long_axis": [1, 0, 0]}, "axial_brace": {"yaws": 2, "long_axis": [1, 0, 0], "clear_turns": True}})
    monkeypatch.delenv(ENV, raising=False)
    off = CL.build_job(jm, root)
    monkeypatch.setenv(ENV, "1")
    on = CL.build_job(jm, root)
    for kind in ("push", "brace"):
        # owner 2026-11-08 item 3: an upright rest builds today's identity-rest library under the switch too - not in the identity, the
        # same digest and entries
        assert all("settled_rest" not in f["params"] for f in on[kind]["families"].values())
        assert on[kind]["digest"] == off[kind]["digest"]
        eo, en = off[kind]["entries"], on[kind]["entries"]
        assert len(eo) > 0 and eo == en


def _bottom_offset(hull, q):
    return -float(C.transform_points((th.zeros(3), TT._t(q).float()), hull.float())[:, 2].min())


def _setup_rest(s, monkeypatch, map_rest, h_root):
    """``AlignmentAxial``'s setup of the rest (``self.rest_quat`` ... ; ``_settled_rest`` under the switch) with a push_blade map at ``map_rest``."""
    pm = SimpleNamespace(path="/m/cmap__push_blade__x", meta={"object": {"rest": {"quat_xyzw": list(map_rest), "h_root": h_root}}})
    monkeypatch.setattr(TT, "contact_maps_on", lambda: True)
    monkeypatch.setattr(TT, "contact_map", lambda self, kind, role, top: pm)
    s.rest_quat, s.rest_h_root, s.rest_info = None, None, {}
    if CPZ.axial_settled_rest_on():
        AX.AlignmentAxial._settled_rest(s)
    return s


@pytest.mark.parametrize("rest", [(0.0, 0.0, 0.0, 1.0), UPRIGHT_REST, (0.02, 0.0, 0.0, 0.9998)])
def test_an_upright_rest_places_identically_with_the_switch_on_and_off(monkeypatch, rest):
    """owner 2026-11-08 item 3: an upright rest (tilt <= 0.05 rad) keeps the usual 3 mm drop gap - the placement (pose and orientation),
    the aligned yaw and the stop geometry are bit-identical with the switch on and off (axial_01 / 03 / 04 draw exactly as today)."""
    assert CPZ.axial_rest_tilt(_unit(rest)) <= CPZ.AXIAL_UPRIGHT_TILT_RAD
    poses = {}
    for sw in ("0", "1"):
        monkeypatch.setenv(ENV, sw)
        s = _ax()
        s.world.bottom_offset = _bottom_offset
        s.gp = SimpleNamespace(variant="v")
        _setup_rest(s, monkeypatch, rest, 0.0047)
        assert s.rest_quat is None
        if sw == "1":
            assert s.kit.stats == {"axial_rest.upright_identity": 1} and s.rest_info["settled"] is False
        st = SimpleNamespace(inward=th.tensor([0.6, 0.8]))
        out = []
        for xy, yaw in ((th.tensor([0.5, 0.1]), 0.0), (th.tensor([0.42, -0.2]), 0.7), (th.tensor([0.6, 0.3]), -2.4)):
            out.append(s._place(xy, yaw))
        pose = out[1]
        com = pose[0] + C.quat_rotate(pose[1], s.com_local)
        end = com + C.quat_rotate(pose[1], s.long_axis) * 0.06
        poses[sw] = (out, s._aligned_yaw(st), s._stop_geometry(pose, end, com, 1.0, 1.3, 0.83))
    (o0, a0, g0), (o1, a1, g1) = poses["0"], poses["1"]
    for (p0, q0), (p1, q1) in zip(o0, o1):
        assert th.equal(p0, p1) and th.equal(q0, q1)
        # the usual drop gap: support top + bottom offset + 3 mm (not the measured h_root)
        assert float(p1[2]) == pytest.approx(0.8126 + _bottom_offset(_hull(), q1) + TT.Z_OFFSET, abs=1e-7)
    assert a0 == a1 and th.equal(g0[0][0], g1[0][0]) and th.equal(g0[0][1], g1[0][1]) and th.equal(g0[2], g1[2])
    assert all(th.equal(x, y) for x, y in zip(g0[1], g1[1]))


def test_a_tilted_rest_is_still_settled_without_the_drop_gap(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    s = _ax()
    s.world.bottom_offset = _bottom_offset
    s.gp = SimpleNamespace(variant="v")
    _setup_rest(s, monkeypatch, SPOON_REST, 0.0085)
    assert s.rest_quat is not None and s.rest_info["settled"] is True and "axial_rest.upright_identity" not in s.kit.stats
    p, q = s._place(th.tensor([0.5, 0.1]), 0.9)
    assert float(p[2]) == pytest.approx(0.8126 + 0.0085) and _tilt(q) == pytest.approx(0.0866, abs=5e-4)


def test_upright_extents_are_the_object_frame_extents():
    hull = _hull((0.20, 0.28, 0.012))
    ext = hull.max(dim=0).values - hull.min(dim=0).values
    l, w, h = AX.rest_extents(hull, _unit(UPRIGHT_REST), th.tensor([0.0, 1.0, 0.0]))
    assert (l, w, h) == pytest.approx((float(ext[1]), float(ext[0]), float(ext[2])), abs=1e-5)


# ---------------------------------------------------------------------------------------------------- a tilted rest keeps its tilt


def test_a_tilted_rest_keeps_its_tilt_at_the_spawn_and_the_goal():
    s = _ax(rest=SPOON_REST, h_root=0.0085)
    t0 = _tilt(s.rest_quat)
    assert t0 == pytest.approx(0.0866, abs=5e-4)
    pose = s._place(th.tensor([0.5, 0.1]), 0.9)
    assert _tilt(pose[1]) == pytest.approx(t0, abs=1e-5) and float(pose[0][2]) == pytest.approx(0.8126 + 0.0085)
    q_goal = s.settled_goal_quat(pose[1], 1.6)
    assert _tilt(q_goal) == pytest.approx(t0, abs=1e-5)
    assert C.wrap_angle(s._heading(q_goal) - s._heading(CPZ.axial_rest_pose(s.rest_quat, 1.6))) == pytest.approx(0.0, abs=1e-5)
    # the stop geometry's goal pose: R_z(yaw_goal) * q_rest
    com = pose[0] + C.quat_rotate(pose[1], s.com_local)
    end = com + C.quat_rotate(pose[1], s.long_axis) * 0.06
    g = CPZ.axial_stop_geometry(s.hull, s.com_local, pose, end, com, 1.0, 1.6, 0.83, rest_quat=s.rest_quat)[0]
    assert C.quat_angle(g[1], CPZ.axial_rest_pose(s.rest_quat, 1.6)) < 1e-5 and _tilt(g[1]) == pytest.approx(t0, abs=1e-5)
    assert float(th.linalg.norm((g[0] + C.quat_rotate(g[1], s.com_local)) - com)) < 1e-5  # the centre of mass stays


def test_the_settled_library_is_built_at_the_tilted_rest(tmp_path, monkeypatch):
    root = str(tmp_path / "glib")
    _model_dir(root, "tablespoon__tqqwys__s1", SPOON_DIMS, rest_q=SPOON_REST, h_root=0.0085)
    job = _job("tablespoon__tqqwys__s1", ["axial_push", "axial_brace"], subtask="alignment-axial",
               params={"axial_push": {"yaws": 2, "long_axis": [1, 0, 0]}, "axial_brace": {"yaws": 2, "long_axis": [1, 0, 0], "clear_turns": True}})
    monkeypatch.setenv(ENV, "1")
    libs = CL.build_job(job, root)
    from ispatialgym.sampling import contact_maps as CMAP

    assert all(f["params"].get("settled_rest") is True for lib in libs.values() for f in lib["families"].values())  # tilted: in the identity

    for kind in ("push", "brace"):
        es = libs[kind]["entries"]
        assert es
        for e in es:
            T = S.pose_of(e["T_rule_object"])
            q = S.mat_to_quat(T[:3, :3])
            assert _tilt(q) == pytest.approx(0.0866, abs=5e-4) and T[2, 3] == pytest.approx(0.0085, abs=1e-6)
            # the maps read the entries' rest back as the v2 rest (the sampler's yaw lookup and the rest check)
            assert CPZ.axial_rest_match(CMAP.entry_rest_quat(e), SPOON_REST, tol=1e-3)
    # the validation's start pose of a push is the v2 rest turned to the slot (validate_contacts: quat_yaw(yaw) * rest) = T_rule_object
    e = libs["push"]["entries"][0]
    q0 = S.quat_mul(S.quat_yaw(math.radians(e["yaw_deg"])), np.asarray(SPOON_REST, dtype=float))
    assert np.allclose(S.quat_to_mat(q0), S.pose_of(e["T_rule_object"])[:3, :3], atol=1e-5)


# ---------------------------------------------------------------------------------------------------- the goal check


@pytest.mark.parametrize("rest", [(0.0, 0.0, 0.0, 1.0), SPOON_REST, (0.12, -0.05, 0.0, 0.99)])
def test_the_goal_check_is_invariant_to_the_rest_tilt_and_sensitive_to_the_turn(rest):
    s = _ax(rest=rest)
    q_settled = s._place(th.tensor([0.5, 0.1]), 0.4)[1]
    q_goal = s.settled_goal_quat(q_settled, 1.3)
    for eps in (0.0, 0.03, -0.07, 0.2):
        # the hinge turns the target about the vertical: the tilt stays, the heading moves
        achieved = C.qmul(C.quat_from_yaw(C.wrap_angle(s._heading(q_goal) - s._heading(q_settled)) + eps), q_settled)
        assert _full_err(achieved, q_goal) == pytest.approx(abs(eps), abs=2e-4)
    # today's upright goal: the tilt alone uses the tolerance (tablespoon: 0.087 rad of the 0.05-0.083 isg-v2 tolerance)
    aligned = C.qmul(C.quat_from_yaw(C.wrap_angle(s._heading(q_goal) - s._heading(q_settled))), q_settled)
    assert _full_err(aligned, C.quat_from_yaw(1.3)) == pytest.approx(_tilt(s.rest_quat), abs=2e-3)


# ---------------------------------------------------------------------------------------------------- the map's rest


def test_a_map_serves_only_the_rest_it_was_built_at(monkeypatch):
    reqs = []
    monkeypatch.setattr(TT, "map_request", lambda s, **kw: reqs.append(kw))
    upright_map = SimpleNamespace(meta={"entry_rest_quat_xyzw": [0.0, 0.0, 0.0, 1.0]})
    spoon_map = SimpleNamespace(meta={"entry_rest_quat_xyzw": list(SPOON_REST)})
    off, on, on_up = _ax(rest=None), _ax(rest=SPOON_REST), _ax(rest=UPRIGHT_REST)
    assert off._map_rest_ok(upright_map, "push_blade") and off._map_rest_ok(None, "push_blade")
    assert not off._map_rest_ok(spoon_map, "push_blade") and off.kit.stats == {"axial_rest.map_rest_mismatch.push_blade": 1} and reqs
    assert on._map_rest_ok(spoon_map, "push_blade") and not on._map_rest_ok(upright_map, "axial_brace")
    assert on_up._map_rest_ok(upright_map, "push_blade")  # the upright identity-built maps serve the switch's upright rests
