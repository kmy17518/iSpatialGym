"""i7-axialrest 5y fix (lead decision 2026-11-08, owner away): (a) the axial_brace contact-map builder keeps the live 0.05 rad joint-limit
margin behind ``ISPATIALGYM_AXIAL_BRACE_MAP_JOINT_MARGIN`` (axial_brace only; in the map identity only when on); (b) the sampler reads the
axial_brace map between two built tilts as the INTERSECTION of the bracketing columns behind ``ISPATIALGYM_AXIAL_BRACE_TILT_INTERSECT``.
Simulator-free: the map computation on the fake synthetic world of test_contact_maps (the IK stubbed), the lookup on a hand-made map."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import object_geometry as OG
from ispatialgym.sampling import contact_maps as CMAP
from ispatialgym.sampling import reach_maps as RM
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.tasks import alignment_axial as AX
from ispatialgym.tests.test_contact_library import _model_dir
from ispatialgym.tests.test_contact_maps import _FakeHC, _fake_world, _lib

ARM = [f"left_arm_joint{i}" for i in range(1, 7)]


@pytest.fixture
def glib(tmp_path):
    root = str(tmp_path / "glib")
    _model_dir(root, "magazine__flat__s1", (0.20, 0.28, 0.012))
    return root


def test_the_margin_is_the_live_one_and_both_switches_are_off_by_default(monkeypatch):
    from ispatialgym.sampling.v2.ik import IKChecker

    assert CMAP.LIVE_JOINT_LIMIT_MARGIN == IKChecker.JOINT_LIMIT_MARGIN == 0.05
    monkeypatch.delenv("ISPATIALGYM_AXIAL_BRACE_MAP_JOINT_MARGIN", raising=False)
    monkeypatch.delenv("ISPATIALGYM_AXIAL_BRACE_TILT_INTERSECT", raising=False)
    assert not CMAP.joint_margin_on() and not AX.axial_brace_tilt_intersect_on()
    off = CMAP.contact_config("axial_brace")
    assert off.joint_margin_rad is None and "joint_margin_rad" not in off.key()  # the maps built before keep their keys
    monkeypatch.setenv("ISPATIALGYM_AXIAL_BRACE_MAP_JOINT_MARGIN", "1")
    on = CMAP.contact_config("axial_brace")
    assert on.key()["joint_margin_rad"] == 0.05 and {k: v for k, v in on.key().items() if k != "joint_margin_rad"} == off.key()
    for kind in ("touch_push", "touch_brace", "push_blade", "floor_push"):  # axial_brace only
        assert CMAP.contact_config(kind).joint_margin_rad is None


def _margin_world(cfg):
    w = _fake_world(cfg)
    names = ["torso_joint1"] + ARM  # a locked-trunk joint among the active ones must not count (the live check reads the arm joints)
    w.jidx = {n: i for i, n in enumerate(names)}
    lo, hi = -th.ones(28), th.ones(28)
    w.robot = SimpleNamespace(arm_control_idx={"left": th.tensor([w.jidx[n] for n in ARM]), "right": th.tensor([20, 21])},
                              joint_lower_limits=lo, joint_upper_limits=hi)
    w.entry = lambda: SimpleNamespace(rsc=SimpleNamespace(joint_names=names))
    return w


def _brace_map(monkeypatch, glib, on: bool, margin_at_contact: float):
    lib = _lib(glib, "magazine__flat__s1", "axial_brace", "brace", long_axis=[0, 1, 0])
    monkeypatch.setattr(CP, "HandChecker", _FakeHC)
    monkeypatch.setattr(CP, "sphere_support_ok", lambda sph, ln, sb, arm, policy, **k: (th.ones(sph.shape[0], dtype=th.bool), None))

    def solve(world, entry, T, q_lock, seed_active=None):
        # every pose solves; the pre (no seed) 0.2 rad from a limit, the contact `margin_at_contact` from it ahead of x = 0.7 m (the torso
        # joint ON its limit everywhere: not an arm joint)
        n, S_ = T.shape[0], world.cfg.seeds
        q = th.full((n, S_, 7), 0.0)
        q[..., 0] = 1.0
        m = th.full((n,), 0.2) if seed_active is None else th.where(T[:, 0, 3] < 0.7, th.tensor(margin_at_contact), th.tensor(0.2))
        q[..., 3] = (1.0 - m).unsqueeze(1)
        return th.ones(n, S_, dtype=th.bool), q

    monkeypatch.setattr(RM, "_solve", solve)
    monkeypatch.setattr(RM, "robot_digest", lambda: "fakerobot")
    if on:
        monkeypatch.setenv("ISPATIALGYM_AXIAL_BRACE_MAP_JOINT_MARGIN", "1")
    else:
        monkeypatch.delenv("ISPATIALGYM_AXIAL_BRACE_MAP_JOINT_MARGIN", raising=False)
    cfg = CMAP.contact_config("axial_brace", lattice_m=0.16, seeds=2, edge_distances=(0.45,))
    geom = OG.load(os.path.join(glib, "models", "magazine__flat__s1"))
    return CMAP.compute_contact_map(_margin_world(cfg), lib, "axial_brace", 0.75, {"key": "h0.75", "height": 0.75}, geom, cfg=cfg)


def test_the_builder_accepts_an_ik_solution_only_with_the_live_margin(monkeypatch, glib):
    meta_off, arr_off = _brace_map(monkeypatch, glib, False, 0.03)
    meta_on, arr_on = _brace_map(monkeypatch, glib, True, 0.03)
    meta_ok, arr_ok = _brace_map(monkeypatch, glib, True, 0.06)
    k = next(t["key"] for t in meta_on["trunks"] if t.get("eligible"))
    F, M = CMAP.STATUS["feasible"], CMAP.STATUS_IK_MARGIN
    st_off, st_on, st_ok = arr_off[f"s0.45_{k}"], arr_on[f"s0.45_{k}"], arr_ok[f"s0.45_{k}"]
    # 30 mrad at a contact ahead of x = 0.7 m: ik_margin there; only refusals, never a new feasible, every other status as before
    assert (st_off == F).any() and (st_on == M).any() and (st_on == F).sum() < (st_off == F).sum()
    assert ((st_on == st_off) | ((st_off == F) & (st_on == M))).all()
    assert (st_ok == st_off).all()  # 60 mrad >= the live 50: the map as before
    assert meta_on["timing"]["ik_margin_lost"] > 0 and meta_on["status_codes"]["ik_margin"] == M and "ik_margin" not in meta_off["status_codes"]
    assert CMAP.STATUS_NAME[M] == "ik_margin"
    # the identity: the margin enters the key only when on; the pose set is the same library's
    assert meta_on["config"]["joint_margin_rad"] == 0.05 and "joint_margin_rad" not in meta_off["config"]
    assert meta_on["key"] != meta_off["key"] and meta_on["pose_set"]["digest"] == meta_off["pose_set"]["digest"]


def _hand_map(statuses):
    """A 1-cell, 1-yaw, 3-class axial_brace map with eligible trunks at translate 0.5 and the tilts of ``statuses`` (tilt -> 3 statuses)."""
    cm = CMAP.ContactMap.__new__(CMAP.ContactMap)
    cm.path, cm.lookup_how = "hand", None
    cm.classes = ["-end+spin_blade_0@f0", "-end+spin_blade_0-@f0", "-end+spin_blade_30@f0"]
    trunks = [{"translate": 0.5, "tilt": ti, "key": f"t0.50_k{ti:+.1f}", "eligible": v is not None} for ti, v in statuses.items()]
    cm.meta = {"kind": "axial_brace", "edge_distances": [0.45], "trunks": trunks,
               "classes": [{"class": c, "goal": {"end_sign": -1.0, "spin": 1.0}} for c in cm.classes]}
    cm.arrays = {f"s0.45_t0.50_k{ti:+.1f}": np.array([[v]], dtype=np.uint8) for ti, v in statuses.items() if v is not None}
    cm.cells, cm.yaws = np.array([[0.6, 0.1]], dtype=np.float32), np.array([0.0], dtype=np.float32)
    return cm


def test_the_intersection_read_between_two_built_tilts():
    F, P = CMAP.STATUS["feasible"], CMAP.STATUS["ik_pre"]
    cm = _hand_map({0.0: [F, F, P], -0.3: [F, P, F], -0.6: [P, F, P], -0.9: None})
    look = lambda ti, **kw: cm.lookup(0.45, 0.5, ti, [0.6, 0.1], 0.0, **kw)  # noqa: E731
    # the nearest read (the default): -0.2 reads -0.3 alone
    assert look(-0.2)["feasible"] == ["-end+spin_blade_0@f0", "-end+spin_blade_30@f0"] and "tilt_read" not in look(-0.2)
    # between 0 and -0.3: feasible in both columns only
    out = look(-0.2, tilt_read="intersection")
    assert out["ok"] and out["feasible"] == ["-end+spin_blade_0@f0"] and out["trunk"] == [0.5, -0.3]
    assert out["tilt_read"]["how"] == "between" and out["tilt_read"]["trunks"] == [[0.5, -0.3], [0.5, 0.0]]
    assert out["tilt_read"]["dropped"] == ["-end+spin_blade_30@f0"]
    # -0.454 (a Phase 2 tilt_scan stance; nearest -0.6): -0.3 x -0.6 share no class -> nothing
    out = look(-0.454, tilt_read="intersection")
    assert look(-0.454)["feasible"] == ["-end+spin_blade_0-@f0"] and out["trunk"] == [0.5, -0.6]
    assert not out["ok"] and out["feasible"] == [] and out["tilt_read"]["dropped"] == out["tilt_read"]["nearest_feasible"] == ["-end+spin_blade_0-@f0"]
    # within 5 mrad of a built tilt: that column alone; beyond the eligible tilts (-0.9 ineligible): the nearest alone
    assert look(-0.303, tilt_read="intersection")["tilt_read"]["how"] == "built" and look(-0.303, tilt_read="intersection")["feasible"] == look(-0.3)["feasible"]
    assert look(-0.75, tilt_read="intersection")["tilt_read"]["how"] == "outside" and look(-0.75, tilt_read="intersection")["feasible"] == look(-0.75)["feasible"]
    assert look(0.05, tilt_read="intersection")["tilt_read"]["how"] == "outside"
    # the class filter applies to both columns
    out = look(-0.2, tilt_read="intersection", keep=lambda r: r["class"].endswith("30@f0"))
    assert not out["ok"] and out["tilt_read"]["dropped"] == ["-end+spin_blade_30@f0"]
    with pytest.raises(ValueError):
        look(-0.2, tilt_read="mean")


def test_the_sampler_reads_the_brace_map_by_intersection_only_under_its_switch(monkeypatch):
    F, P = CMAP.STATUS["feasible"], CMAP.STATUS["ik_pre"]
    bm = _hand_map({0.0: [F, F, P], -0.3: [F, P, F], -0.6: [P, F, P]})
    pm = SimpleNamespace(meta={"kind": "push_blade", "classes": [{"class": "-end+spin_0@t45", "arc_turn_rad": 0.785}]})
    calls = []

    def check(cm, st, pose, keep=None, tol=0.06, tilt_read="nearest"):
        calls.append((cm.meta["kind"], tilt_read))
        if cm is pm:
            return {"ok": True, "why": None, "cell": [0.6, 0.1], "feasible": ["-end+spin_0@t45"]}
        return cm.lookup(0.45, 0.5, -0.454, [0.6, 0.1], 0.0, keep=keep, tilt_read=tilt_read)

    monkeypatch.setattr(TT, "contact_map_check", check)
    ax = SimpleNamespace(top=SimpleNamespace(name="desk_0"), kit=SimpleNamespace(stats={}), long_axis=th.tensor([0.0, 1.0, 0.0]))
    ax.axial_maps = lambda: (pm, bm)
    pose = (th.tensor([0.6, 0.1, 0.9]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    monkeypatch.delenv("ISPATIALGYM_AXIAL_BRACE_TILT_INTERSECT", raising=False)
    monkeypatch.delenv("ISPATIALGYM_AXIAL_BRACE_TURN_COVER", raising=False)
    jp = AX.AlignmentAxial.map_joint_placement(ax, None, pose, pose, 1.0, 0.7, end_sign=-1.0)
    assert jp["brace_ok"] and calls[-1] == ("axial_brace", "nearest") and ax.kit.stats == {}
    monkeypatch.setenv("ISPATIALGYM_AXIAL_BRACE_TILT_INTERSECT", "1")
    jp = AX.AlignmentAxial.map_joint_placement(ax, None, pose, pose, 1.0, 0.7, end_sign=-1.0)
    assert not jp["brace_ok"] and calls[-1] == ("axial_brace", "intersection") and calls[-2] == ("push_blade", "nearest")
    assert ax.kit.stats == {"axial_brace.tilt_read.between": 1, "axial_brace.tilt_read.dropped": 1,
                            "axial_brace.tilt_read.dropped_classes": 1, "axial_brace.tilt_read.emptied": 1}


def test_the_live_brace_is_counted_against_the_map_classes():
    ax = SimpleNamespace(kit=SimpleNamespace(stats={}))
    jp = {"maps": {"axial_brace": {"feasible": ["-end+spin_blade_0-@f0", "+end+spin_opp_blade_30@f0"]}, "end_sign_map": {"axial_brace": -1.0}}}
    prop = lambda kind, f=0: SimpleNamespace(payload=(None, None, kind, {"stop_face_fraction": f}))  # noqa: E731
    AX.AlignmentAxial._count_brace_vs_map(ax, jp, 1.0, prop("blade_0-"), False)
    AX.AlignmentAxial._count_brace_vs_map(ax, jp, 1.0, prop("blade_30"), True)
    AX.AlignmentAxial._count_brace_vs_map(ax, jp, -1.0, prop("blade_0-"), True)  # the other spin: not the map's class
    jp["maps"]["end_sign_map"]["axial_brace"] = None  # a map on another axis: matched on the spin and kind
    AX.AlignmentAxial._count_brace_vs_map(ax, jp, 1.0, prop("opp_blade_30"), True)
    AX.AlignmentAxial._count_brace_vs_map(ax, {}, 1.0, prop("left_h"), False)  # no map record: map_other
    assert ax.kit.stats == {"axial_brace.live.map_feasible.tried": 2, "axial_brace.live.map_feasible.ok": 1,
                            "axial_brace.live.map_other.tried": 3, "axial_brace.live.map_other.ok": 2}
