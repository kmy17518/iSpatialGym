"""i7-brace (owner 2026-11-08, 3.3 / 3.4): simulator-free tests of the validator's kinematic brace-load pusher and the re-validation
of the earlier-validated entries (the per-entry verdict changes of a validator fix)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import torch as th

from ispatialgym.grasping import validate_contacts as VC


def test_the_brace_load_pusher_is_kinematic(monkeypatch):
    """Owner 3.3: OmniGibson honours kinematic_only only with a fixed base - the pusher is built with both (it was dynamic before)."""
    made = []

    class _Prim:
        def __init__(self, **kw):
            made.append(kw)

        def set_position_orientation(self, *a):
            pass

    mod = types.ModuleType("omnigibson.objects.primitive_object")
    mod.PrimitiveObject = _Prim
    monkeypatch.setitem(sys.modules, "omnigibson.objects.primitive_object", mod)
    w = VC.KitContactWorld.__new__(VC.KitContactWorld)
    w.cfg, w._pusher = VC.ContactValidateConfig(), None
    w.tw = SimpleNamespace(th=th, scene=SimpleNamespace(add_object=lambda o: None), og=SimpleNamespace(sim=SimpleNamespace(step=lambda: None)))
    w._pusher_obj()
    assert made and made[0]["fixed_base"] is True and made[0]["kinematic_only"] is True
    assert VC.PUSHER_KINEMATIC == VC.STOP_STANDIN_KINEMATIC


class _World:
    def __init__(self, valid):
        self.valid, self.calls = valid, []

    def settle(self, obj, geom, q0):
        return {"world_position": [0, 0, 0.1], "world_quat_xyzw": [0, 0, 0, 1]}, None

    def plan_entry(self, obj, geom, e, T, arm):
        return {"q": [0.0]}

    def trial_brace(self, obj, e, plan, state, rest, arm):
        self.calls.append(e["class"])
        return {"verdict": "valid" if self.valid(e) else "invalid", "reasons": []}


def test_revalidate_reruns_exactly_the_validated_entries_and_keeps_the_old_record():
    es = [{"class": f"c{i}", "yaw_deg": 0.0, "family": "touching_brace"} for i in range(6)]
    for i, v in enumerate(["valid", "invalid", "unreachable", None, "screened", "valid"]):
        if v:
            es[i]["validation"] = {"verdict": v}
    es[4]["sampler_screen"] = {"hand_overlap": "knuckle"}
    lib = {"kind": "brace", "object": {"rest": {"quat_xyzw": [0, 0, 0, 1]}, "rest_id": "r0"}, "entries": es}
    w = _World(lambda e: e["class"] in ("c1", "c5"))
    tim = VC.validate_library(w, lib, None, None, np.zeros((8, 3)), VC.ContactValidateConfig(revalidate=True, max_entries=1), "left")
    assert sorted(w.calls) == ["c0", "c1", "c2", "c5"]  # uncapped, only the earlier-validated ones
    assert es[0]["validation"]["verdict"] == "invalid" and es[0]["validation_prev"]["verdict"] == "valid"
    assert es[1]["validation"]["verdict"] == "valid" and es[3].get("validation") is None and es[4]["validation"]["verdict"] == "screened"
    assert tim["n_revalidated"] == 4 and tim["n_changed"] == 3  # c0 valid->invalid, c1 invalid->valid, c2 unreachable->invalid


# ------------------------------------------------------------------------------------------------ 3.4 axial: the clear-turn build

import math  # noqa: E402

import pytest  # noqa: E402

from ispatialgym.grasping import contact_library as CL  # noqa: E402
from ispatialgym.sampling import contact_maps as CMAP  # noqa: E402
from ispatialgym.tests.test_contact_library import _job, _model_dir  # noqa: E402


@pytest.fixture
def mag(tmp_path):
    root = str(tmp_path / "glib")
    _model_dir(root, "magazine__flat__s1", (0.20, 0.28, 0.012))
    return root


def test_the_clear_turn_axial_brace_build_records_the_start_range_and_adds_the_opposite_end(mag):
    """Owner 3.4: one build keyed by pushed end x spin; every entry records the largest turn its sweep disc stays clear for (the sampler's
    blade_sweep_clear at every grid turn up to it); the opposite-end candidates come after, loaded at the far end; off = the 45 deg build."""
    base = CL.build_job(_job("magazine__flat__s1", ["axial_brace"], params={"axial_brace": {"yaws": 1, "long_axis": [0, 1, 0]}}), mag)["brace"]["entries"]
    es = CL.build_job(_job("magazine__flat__s1", ["axial_brace"], params={"axial_brace": {"yaws": 1, "long_axis": [0, 1, 0], "clear_turns": True}}),
                      mag)["brace"]["entries"]
    assert all("clear_turn_max_deg" not in e["goal"] for e in base)  # the default build is unchanged
    assert es and all(0.0 < e["goal"]["clear_turn_max_deg"] <= 90.0 for e in es)
    groups = {(e["goal"]["end_sign"], e["goal"]["spin"]) for e in es}
    assert groups <= {(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)} and len(groups) >= 2
    opp = [e for e in es if e["goal"]["opposite_end"]]
    own = [e for e in es if not e["goal"]["opposite_end"]]
    assert opp and own and all("_opp_" in e["class"] for e in opp)
    # the opposite entry is loaded at the far end, opposite the pushed end's arrival
    e_own = next(e for e in own if e["goal"]["end_sign"] == 1.0 and e["goal"]["spin"] == 1.0)
    e_opp = next(e for e in opp if e["goal"]["end_sign"] == 1.0 and e["goal"]["spin"] == 1.0)
    assert np.dot(e_own["load"]["direction"], e_opp["load"]["direction"]) < -0.9
    assert np.dot(e_own["load"]["point_object"][:2], e_opp["load"]["point_object"][:2]) < 0
    # within one (end, spin), the library order keeps the current candidates before the opposite ones
    g = [e for e in es if (e["goal"]["end_sign"], e["goal"]["spin"]) == (1.0, 1.0)]
    first_opp = next(i for i, e in enumerate(g) if e["goal"]["opposite_end"])
    assert all(not e["goal"]["opposite_end"] for e in g[:first_opp])
    # the map's class rows carry the range; the switch keeps the entries clear for the drawn turn only
    row = CMAP.class_row(e_own)
    assert row["goal"]["clear_turn_max_deg"] == e_own["goal"]["clear_turn_max_deg"]


def test_clear_turn_max_is_the_contiguous_clear_range(monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    seen = []

    def fake(hull, pose_init, com_xy, delta, xy, r, c, n=12):
        seen.append(round(math.degrees(abs(delta))))
        return (math.degrees(abs(delta)) <= 40.0 or math.degrees(abs(delta)) >= 80.0), 0.0

    monkeypatch.setattr(TT, "blade_sweep_clear", fake)
    goal = (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0]))
    t = CL.clear_turn_max(th.zeros(8, 3), th.zeros(3), goal, 1.0, {"xy": [0.1, 0.0], "radius": 0.007, "clearance": 0.005})
    assert t == 40.0 and seen[-1] == 45  # stops at the first failing turn (80+ never counts: the range is contiguous)


def test_the_turn_cover_switch_filters_the_brace_classes(monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.v2.tasks import alignment_axial as AX

    seen = []

    def check(cm, st, pose, keep=None, tol=0.06):
        kept = [r["class"] for r in cm.meta["classes"] if r and (keep is None or keep(r))]
        seen.append((cm.meta["kind"], kept))
        return {"ok": bool(kept), "feasible": kept, "cell": [0.6, 0.0]}

    maps = {"push_blade": SimpleNamespace(meta={"kind": "push_blade", "classes": [{"class": "+end+spin_0@t25", "arc_turn_rad": 0.436}]}),
            "axial_brace": SimpleNamespace(meta={"kind": "axial_brace", "classes": [
                {"class": "+end+spin_v@f0", "goal": {"spin": 1.0, "end_sign": 1.0, "clear_turn_max_deg": 20.0}},
                {"class": "+end+spin_opp_v@f0", "goal": {"spin": 1.0, "end_sign": 1.0, "clear_turn_max_deg": 60.0, "opposite_end": True}}]})}
    monkeypatch.setattr(TT, "contact_map", lambda s, kind, role, top, arm=None: maps.get(kind))
    monkeypatch.setattr(TT, "contact_map_check", check)
    ax = SimpleNamespace(top=SimpleNamespace(name="desk_0"), long_axis=th.tensor([0.0, 1.0, 0.0]))
    ax.axial_maps = lambda: AX.AlignmentAxial.axial_maps(ax)
    ax._map_end_sign = lambda cm, e: AX.AlignmentAxial._map_end_sign(ax, cm, e)
    pose = (th.tensor([0.6, 0.0, 0.8]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    monkeypatch.setenv("ISPATIALGYM_AXIAL_BRACE_TURN_COVER", "0")
    AX.AlignmentAxial.map_joint_placement(ax, None, pose, pose, 1.0, math.radians(30.0), end_sign=1.0)
    assert seen[-1] == ("axial_brace", ["+end+spin_v@f0", "+end+spin_opp_v@f0"])  # off: as before
    monkeypatch.setenv("ISPATIALGYM_AXIAL_BRACE_TURN_COVER", "1")
    AX.AlignmentAxial.map_joint_placement(ax, None, pose, pose, 1.0, math.radians(30.0), end_sign=1.0)
    assert seen[-1] == ("axial_brace", ["+end+spin_opp_v@f0"])  # the 20 deg entry does not cover a 30 deg turn


def test_the_axial_brace_switches_are_on_by_default(monkeypatch):
    """owner answers 20261108b item 4: TURN_COVER and OPPOSITE default ON; =0 turns each off."""
    from ispatialgym.sampling.v2.tasks import alignment_axial as AX

    for k, f in (("ISPATIALGYM_AXIAL_BRACE_TURN_COVER", AX.axial_brace_turn_cover_on), ("ISPATIALGYM_AXIAL_BRACE_OPPOSITE", AX.axial_brace_opposite_on)):
        monkeypatch.delenv(k, raising=False)
        assert f()
        monkeypatch.setenv(k, "0")
        assert not f()


def test_the_opposite_end_entries_are_their_own_stratification_group():
    assert VC.axial_group({"class": "+end-spin_v@f0"}) == "+end-spin"
    assert VC.axial_group({"class": "+end-spin_opp_v@f0"}) == "+end-spin_opp"
    assert VC.axial_group({"class": "push_tip_h"}) == ""


# ------------------------------------------------------------------------------------------------ 3.4 world_vertical: wall_brace


def _painting():
    L, W, T = 0.40, 0.60, 0.03  # height (z, up), width (y, side), thickness (x, front)
    hull = th.tensor([[sx * T / 2, sy * W / 2, sz * L / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=th.float32)
    return hull


def test_alignment_wall_keeps_its_geometry_methods_as_one_implementation():
    """i7-brace: AlignmentWall's contact geometry moved verbatim into WallGeometry - the task resolves to the same functions (live behaviour
    unchanged), and a scene-free WallGeometry gives the same braces as an AlignmentWall with the same attributes."""
    from ispatialgym.sampling.v2.tasks import alignment_wall as AW

    for m in ("dims", "depth_extents", "wall_offset", "mid_plane_point", "hang_pose", "hull_at", "contact_section", "contacts_for", "brace_clear"):
        assert getattr(AW.AlignmentWall, m) is getattr(AW.WallGeometry, m)
    f, s_, u = AW.frame_axes("+x", "+z")
    hull = _painting()
    g = AW.WallGeometry.standalone(hull, f, s_, u, th.zeros(3))
    task = AW.AlignmentWall.__new__(AW.AlignmentWall)
    task.hull, task.front, task.side, task.up, task.com_local, task.pieces = hull, f, s_, u, th.zeros(3), []
    task.ctx = SimpleNamespace()
    task.param = lambda key, default=None, v1_key=None: default
    spot = {"point": th.tensor([2.0, 1.0, 0.0]), "normal": th.tensor([0.0, -1.0, 0.0]), "hang_z": 1.4, "wall_protrusion_m": 0.0}
    a, b = task.contacts_for(spot, math.radians(25.0)), g.contacts_for(spot, math.radians(25.0))
    assert [x["kind"] for x in a["braces"]] == [x["kind"] for x in b["braces"]] and a["braces"]
    assert all(bool(th.allclose(x["pose"][0], y["pose"][0])) for x, y in zip(a["braces"], b["braces"]))


def test_the_wall_brace_library_keys_by_tilt_sign_and_records_the_clear_range(monkeypatch):
    from ispatialgym.sampling.v2.tasks import alignment_wall as AW

    hull = _painting()
    es = CL.build_wall_brace(hull.numpy(), None, {}, {"front_axis": "+x", "up_axis": "+z"})
    assert es and {e["goal"]["tilt_sign"] for e in es} == {1.0, -1.0}
    assert all(e["family"] == "wall_brace" and 0.0 < e["goal"]["clear_delta_max_deg"] <= 45.0 for e in es)
    assert all(e["class"].startswith(("+tilt_", "-tilt_")) for e in es)
    kinds = {e["class"].split("_", 1)[1].rsplit("_", 1)[0] for e in es}
    assert kinds <= {"below_high", "above_low", "side_high"}
    # the recorded range is the task's own brace_clear at every grid tilt up to it (and fails just past it, below the grid's end)
    f, s_, u = AW.frame_axes("+x", "+z")
    g = AW.WallGeometry.standalone(hull, f, s_, u, th.zeros(3))
    spot = {k: (th.tensor(v, dtype=th.float32) if isinstance(v, tuple) else v) for k, v in CL.WALL_RULE_SPOT.items()}
    e = es[0]
    sign, t_max = e["goal"]["tilt_sign"], e["goal"]["clear_delta_max_deg"]
    geo = g.contacts_for(spot, sign * math.radians(t_max))
    b = next(x for x in geo["braces"] if f"{'+' if sign > 0 else '-'}tilt_{x['kind']}_{x['wrist']}" == e["class"])
    assert g.brace_clear(geo, b)
    if t_max < 45.0:
        assert not g.brace_clear(g.contacts_for(spot, sign * math.radians(t_max + 5.0)), b)
    # the load turns the braced point on past level: tangential to the hinge, in the wall plane
    d, B = np.asarray(e["load"]["direction"]), np.asarray(e["load"]["point_object"])
    n = np.asarray(e["wall"]["normal_object"])
    assert abs(float(d @ n)) < 1e-5 and abs(float(d @ (B - np.asarray(e["wall"]["hinge_object"])))) < 1e-5
    with pytest.raises(ValueError):
        CL.build_wall_brace(hull.numpy(), None, {}, {})  # no reviewed frame: no default


# ------------------------------------------------------------------------------------------------ 3.4 world_vertical: the wall world, the map, the switch


def _wall_entries():
    hull = _painting().numpy()
    return hull, CL.build_wall_brace(hull, None, {}, {"front_axis": "+x", "up_axis": "+z"}, pieces=[hull])


def test_the_wall_brace_entry_carries_the_brace_arm_the_oracle_record_and_a_push_driven_load():
    from ispatialgym.sampling.v2.tasks import alignment_wall as AW

    hull, es = _wall_entries()
    f, s_, u = AW.frame_axes("+x", "+z")
    g = AW.WallGeometry.standalone(th.as_tensor(hull, dtype=th.float32), f, s_, u, th.zeros(3))
    spot = {k: (th.tensor(v, dtype=th.float32) if isinstance(v, tuple) else v) for k, v in CL.WALL_RULE_SPOT.items()}
    for sign in (1.0, -1.0):
        geo = g.contacts_for(spot, sign * math.radians(20.0))
        mine = [e for e in es if e["goal"]["tilt_sign"] == sign]
        assert mine and {e["wall"]["brace_arm"] for e in mine} == {geo["brace_arm"]} and {e["wall"]["push_arm"] for e in mine} == {geo["push"]["arm"]}
    assert {e["wall"]["brace_arm"] for e in es} == {"left", "right"}
    for e in es:
        ex = e["extras"]
        assert ex["pose_class"] == ex["kind"] and ex["placed_at"] == "goal" and ex["contact_face"] == "finger_side"  # the sampler's brace record
        assert abs(S_pose(e)[2, 3]) < 1e-6  # the hinge (the COM, here the origin) at z 0 of the rule frame
        ld = e["load"]
        a = np.asarray(e["wall"]["approach_object"])
        assert float(a @ np.asarray(ld["brace_direction"])) < 0  # the brace's approach opposes what it stops
        E, d = np.asarray(ld["point_object"]), np.asarray(ld["direction"])
        hinge, n = np.asarray(e["wall"]["hinge_object"]), np.asarray(e["wall"]["normal_object"])
        assert abs(float(d @ n)) < 1e-5 and abs(float(d @ (E - hinge))) < 1e-5  # tangential at the pushed corner, in the wall plane
        assert ld["r_push_m"] > 0.1 and ld["r_brace_m"] > 0.1 and e["wall"]["extents_perp_m"] == [0.6, 0.4]


def S_pose(e):
    from ispatialgym.grasping import schema as S

    return S.pose_of(e["T_rule_object"])


def test_pieces_com_is_the_volume_centroid():
    box = np.array([[x, y, z] for x in (0.0, 0.2) for y in (0.0, 0.1) for z in (0.0, 0.4)])
    tip = box[:4] + np.array([1.0, 0, 0])  # 4 points: too few for a volume -> ignored? (a degenerate tetra counts 0)
    assert np.allclose(CL.pieces_com([box]), [0.1, 0.05, 0.2], atol=1e-6)
    assert np.allclose(CL.pieces_com([box, tip]), [0.1, 0.05, 0.2], atol=1e-6)
    assert CL.pieces_com([]) is None


def test_a_wall_library_validates_its_arm_entries_from_the_hung_pose():
    _hull, es = _wall_entries()
    lib = {"kind": "brace", "object": {"rest": {"quat_xyzw": [0, 0, 0, 1]}, "rest_id": "r0"}, "entries": es, "validate": {"support_z": 1.45}}
    hung = []

    class W(_World):
        def hang(self, obj, geom, T, h, e):
            hung.append((round(float(T[2, 3]), 6), h))
            return {"world_position": [0, 0, h], "world_quat_xyzw": [0, 0, 0, 1]}, None

        def settle(self, *a):
            raise AssertionError("a wall library hangs, it does not settle")

    w = W(lambda e: True)
    VC.validate_library(w, lib, None, None, np.zeros((8, 3)), VC.ContactValidateConfig(max_entries=100), "left")
    left = [e["class"] for e in es if e["wall"]["brace_arm"] == "left"]
    assert sorted(w.calls) == sorted(left) and hung == [(0.0, 1.45)]
    assert all(e.get("validation") is None for e in es if e["wall"]["brace_arm"] == "right") and len(lib["entries"]) == len(es)
    VC.validate_library(w, lib, None, None, np.zeros((8, 3)), VC.ContactValidateConfig(max_entries=100), "right")
    assert all(e["validation"]["verdict"] == "valid" for e in es) and set(lib["validation_timing_by_arm"]) == {"left", "right"}


def test_the_wall_stance_relation_mirrors_for_the_right_arm():
    left = VC.placement_offsets("wall_brace", (0.0, 0.0), (0.0, 0.0), 0.05, 0.3, arm="left")
    right = VC.placement_offsets("wall_brace", (0.0, 0.0), (0.0, 0.0), 0.05, 0.3, arm="right")
    assert [o[0] for o in left] == [o[0] for o in right] and [o[1] for o in left] == [-o[1] for o in right]
    assert VC.placement_offsets("touching_brace", (0, 0), (0, 0), 0.3, 0.3, arm="right") == VC.placement_offsets("touching_brace", (0, 0), (0, 0), 0.3, 0.3)


def _turn_about(n, ang, c):
    n = np.asarray(n, dtype=float) / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(c) - R @ np.asarray(c)
    return T


def test_the_wall_pusher_meets_the_pushed_corner_and_the_judge_reads_the_braced_point():
    hull, es = _wall_entries()
    e = es[0]
    cfg = VC.ContactValidateConfig()
    T0 = S_pose(e).copy()
    T0[2, 3] += 1.45
    path, qs, facts = VC.wall_pusher_path(e, T0, hull, cfg, 1.0 / 30.0)
    d = np.asarray(facts["direction"])
    from ispatialgym.grasping import schema as GS

    R = np.asarray(GS.quat_to_mat(qs[0]))
    n_w0 = T0[:3, :3] @ np.asarray(e["wall"]["normal_object"])
    assert abs(float(R[:, 1] @ n_w0)) < 1e-4  # the 3 cm side across the line in the wall plane, the 8 cm side along the normal
    E = T0[:3, :3] @ np.asarray(e["load"]["point_object"]) + T0[:3, 3]
    H0 = T0[:3, :3] @ np.asarray(e["wall"]["hinge_object"]) + T0[:3, 3]
    # the face starts brace_pusher_gap_m (of arc) before the corner and ends where the braced point is brace_press_m past level
    face = [c + np.asarray(GS.quat_to_mat(q))[:, 2] * 0.5 * cfg.brace_pusher_size_m[2] for c, q in zip(path, qs)]
    r = float(np.linalg.norm((E - H0) - n_w0 * float((E - H0) @ n_w0)))
    assert all(abs(float(np.linalg.norm((f - H0) - n_w0 * float((f - H0) @ n_w0))) - r) < 2e-3 for f in face)  # on the corner's arc
    phi0, phi1 = facts["arc_rad"]
    assert abs(phi0 * facts["r_face_m"] + cfg.brace_pusher_gap_m) < 1e-4 and abs(phi1 * e["load"]["r_brace_m"] - cfg.brace_press_m) < 1e-4
    assert float((face[1] - face[0]) @ d) > 0  # it moves along the corner's tangent first
    # the judge: the target turned past level by an angle about the hinge -> the braced point moved along its load direction
    n_w = T0[:3, :3] @ np.asarray(e["wall"]["normal_object"])
    H = T0[:3, :3] @ np.asarray(e["wall"]["hinge_object"]) + T0[:3, 3]
    vB = T0[:3, :3] @ np.asarray(e["load"]["brace_direction"])
    B = T0[:3, :3] @ np.asarray(e["load"]["brace_point_object"]) + T0[:3, 3]
    still = VC.judge_wall_brace(e, T0, T0, np.eye(4), np.eye(4), cfg, {"completed": True, "postcondition": None})
    assert still["reasons"] == ["not_loaded"]  # no motion at all: the load never acted - not a pass
    T1 = _turn_about(n_w, 0.01 if float(np.cross(n_w, B - H) @ vB) > 0 else -0.01, H) @ T0
    ok = VC.judge_wall_brace(e, T0, T1, np.eye(4), np.eye(4), cfg, {"completed": True, "postcondition": None}, finger_N=12.0)
    past = VC.judge_wall_brace(e, T0, T1, np.eye(4), np.eye(4), cfg, {"completed": True, "postcondition": None}, finger_N=0.0)
    assert ok["verdict"] == "valid" and past["reasons"] == ["not_braced"]  # moved past a fist it never touched: not a brace
    for ang in (0.02, -0.02):
        T1 = _turn_about(n_w, ang, H) @ T0
        mv = float((T1[:3, :3] @ np.asarray(e["load"]["brace_point_object"]) + T1[:3, 3] - B) @ vB)
        out = VC.judge_wall_brace(e, T0, T1, np.eye(4), np.eye(4), cfg, {"completed": True, "postcondition": None})
        assert abs(out["measures"]["brace_point_moved_m"] - round(mv, 5)) < 1e-5 and out["measures"]["hinge_axis_twist_rad"] < 1e-5
        assert (out["measures"]["turn_past_level_rad"] > 0) == (mv > 0)
    big = _turn_about(n_w, 0.06 if float(np.cross(n_w, B - H) @ vB) > 0 else -0.06, H) @ T0
    Te = np.eye(4)
    Te[0, 3] = 0.02
    out = VC.judge_wall_brace(e, T0, big, np.eye(4), Te, cfg, {"completed": True, "postcondition": None}, other_N=1.0)
    assert set(out["reasons"]) == {"not_held", "hand_pushed", "non_finger_contact"}
    assert VC.judge_wall_standing({"completed": True, "postcondition": "x"}, cfg)["reasons"] == ["oracle:x"]
    # a corner whose line meets no hull point raises (no vacuous load); a thick target's faces are found across its half-thickness
    with pytest.raises(ValueError):
        VC.wall_pusher_path(dict(e, load=dict(e["load"], point_object=[0.0, 5.0, 5.0])), T0, hull, cfg, 1.0 / 30.0)
    thick = hull * np.array([4.0, 1.0, 1.0])  # 12 cm thick: the corner's faces 6 cm off its mid-plane line
    _p, _q, f2 = VC.wall_pusher_path(e, T0, thick, cfg, 1.0 / 30.0)
    assert f2["n_line_points"] >= 2 and abs(f2["face_along_m"]) < 0.01  # the corner's own edge, not points far along the line


def test_the_wall_map_cells_are_one_row_per_stand_off_and_the_lookup_stays_in_its_row():
    _hull, es = _wall_entries()
    cfg = CMAP.contact_config("wall_brace", arm="left", lattice_m=0.08, seeds=8)
    assert cfg.edge_distances == CMAP.WALL_EDGE_DISTANCES and cfg.lattice_m == CMAP.WALL_LATTICE_M and cfg.postures == ("tucked",)
    rx = CMAP.wall_root_x(es[0])
    cells, idx = CMAP.wall_cells(cfg, rx)
    assert len(cells) == len(cfg.edge_distances) * 25 and np.allclose(cells[idx == 3][:, 0], cfg.edge_distances[3] + rx)
    left = CMAP.select_entries({"kind": "brace", "entries": es}, "wall_brace", None, arm="left")
    assert left and all(e["wall"]["brace_arm"] == "left" for e in left)
    row = CMAP.class_row(left[0])
    assert row["goal"]["tilt_sign"] == left[0]["goal"]["tilt_sign"] and row["goal"]["clear_delta_max_deg"] > 0 and row["wall"]["brace_arm"] == "left"
    sb = CMAP.wall_support(0.6)
    assert abs(sb.centre[0] - sb.half[0] - 0.6) < 1e-9 and sb.centre[2] - sb.half[2] == 0.0
    # a fake map: one class feasible in the d_e = 0.80 row only; a root half-way between rows reads the stand-off's own row
    classes = sorted({e["class"] for e in left})
    nC, nG = len(cells), len(classes)
    cm = CMAP.ContactMap.__new__(CMAP.ContactMap)
    k = cfg.edge_distances.index(0.8)
    st = np.full((nC, 1, nG), CMAP.STATUS["off_support"], dtype=np.uint8)
    st[idx == k, 0, 0] = CMAP.STATUS["feasible"]
    cm.meta = {"kind": "wall_brace", "edge_distances": list(cfg.edge_distances), "trunks": [{"translate": 0.25, "tilt": 0.0, "key": "t0.25_k+0.0", "eligible": True}],
               "classes": [CMAP.class_row(next(e for e in left if e["class"] == c)) for c in classes]}
    cm.arrays = {"cells": cells.astype(np.float32), "yaws": np.zeros(1, np.float32), "cell_de": idx.astype(np.int16)}
    cm.arrays.update({f"s{d:.2f}_t0.25_k+0.0": (st if d == 0.8 else np.full_like(st, CMAP.STATUS["off_support"])) for d in cfg.edge_distances})
    cm.cells, cm.yaws, cm.classes, cm.path = cm.arrays["cells"], cm.arrays["yaws"], classes, "fake"
    out = cm.lookup(0.8, 0.25, 0.0, [0.8 + rx + 0.024, 0.0], 0.0)
    assert out["ok"] and out["feasible"] == [classes[0]] and abs(out["cell"][0] - round(0.8 + rx, 3)) < 1e-3
    # the sampler's reading of it: the stand-off from the root's position, the tilt sign's classes, then the cover of |delta|
    from ispatialgym.sampling.v2.tasks import alignment_wall as AW

    stance = SimpleNamespace(world_to_base=lambda xy: th.as_tensor(xy, dtype=th.float32), translate=0.25, tilt=0.0)
    c0 = next(e for e in left if e["class"] == classes[0])
    sign = c0["goal"]["tilt_sign"]
    root = th.tensor([0.8 + rx, 0.0, 1.45])
    wf = AW.wall_map_feasible(cm, stance, root, sign * math.radians(20.0))
    assert wf["ok"] and wf["classes"] == [classes[0]] and abs(wf["d_eff"] - 0.8) < 1e-4
    assert not AW.wall_map_feasible(cm, stance, root, -sign * math.radians(20.0))["ok"]  # the other tilt sign's class set
    wf_far = AW.wall_map_feasible(cm, stance, root, sign * math.radians(c0["goal"]["clear_delta_max_deg"] + 1.0))
    assert not wf_far["ok"] and wf_far["cover_dropped"] == 1
    assert AW.wall_brace_class({"kind": c0["extras"]["kind"], "wrist": c0["extras"]["wrist"]}, sign * 0.3) == c0["class"]


def test_the_wall_brace_map_switch_is_off_by_default(monkeypatch):
    from ispatialgym.sampling.v2.tasks import alignment_wall as AW

    monkeypatch.delenv("ISPATIALGYM_WALL_BRACE_MAP", raising=False)
    assert not AW.wall_brace_map_on()
    monkeypatch.setenv("ISPATIALGYM_WALL_BRACE_MAP", "1")
    assert AW.wall_brace_map_on()


def test_a_small_targets_stop_clears_from_some_tilt_on():
    """The stop at the goal edge clears the tilted hull more the farther the target is tilted: a small plate's brace is clear from 10 deg on
    (at 5 deg its high end rises only ~1.3 cm toward the 5 mm stop) - the entry keeps that range instead of being dropped."""
    from ispatialgym.sampling.v2.tasks import alignment_wall as AW

    L, W, T = 0.15, 0.30, 0.01
    hull = np.array([[sx * T / 2, sy * W / 2, sz * L / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    es = CL.build_wall_brace(hull, None, {}, {"front_axis": "+x", "up_axis": "+z"}, pieces=[hull])
    assert es and all(e["goal"]["clear_delta_min_deg"] == 10.0 for e in es)
    f, s_, u = AW.frame_axes("+x", "+z")
    g = AW.WallGeometry.standalone(th.as_tensor(hull, dtype=th.float32), f, s_, u, th.zeros(3))
    spot = {k: (th.tensor(v, dtype=th.float32) if isinstance(v, tuple) else v) for k, v in CL.WALL_RULE_SPOT.items()}
    e = next(x for x in es if "below_high" in x["class"])
    sign = e["goal"]["tilt_sign"]
    for d, clear in ((5.0, False), (10.0, True), (e["goal"]["clear_delta_max_deg"], True)):
        geo = g.contacts_for(spot, sign * math.radians(d))
        b = next(x for x in geo["braces"] if f"{'+' if sign > 0 else '-'}tilt_{x['kind']}_{x['wrist']}" == e["class"])
        assert g.brace_clear(geo, b) == clear
    assert e["goal"]["clear_deltas_deg"][0] == 10.0


def test_find_near_reads_the_map_heights_whatever_the_model_name(tmp_path, monkeypatch):
    d = tmp_path / "hanger__hjdaqn__s1"
    d.mkdir()
    for h in (1.45, 1.2):
        (d / f"cmap__wall_brace__hanger__hjdaqn__h{h:.4f}__0123abcd.json").write_text("{}")
    asked = []
    monkeypatch.setattr(CMAP.ContactMap, "find", staticmethod(lambda root, kind, entry, h, **kw: asked.append(h) or (SimpleNamespace() if h == 1.45 else None)))
    cm = CMAP.ContactMap.find_near(str(tmp_path), "wall_brace", "hanger__hjdaqn__s1", 1.46, 0.03)
    assert cm is not None and asked == [1.45] and cm.lookup_how["map_h"] == 1.45
    assert CMAP.ContactMap.find_near(str(tmp_path), "wall_brace", "hanger__hjdaqn__s1", 1.30, 0.03) is None
