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
