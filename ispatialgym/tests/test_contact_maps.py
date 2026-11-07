"""i7-contact 2026-11-07, step 3: contact maps from the PUSH / BRACE libraries (``sampling/contact_maps.py``) and their consumers under v2only
(``tabletop.contact_map`` / ``contact_map_check``; the touching brace gate, the axial joint placement). The map computation runs on a fake
synthetic world (the IK, the spheres and the hand hulls stubbed); no simulator. Every file goes to tmp_path."""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_library as CL
from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import object_geometry as OG
from ispatialgym.grasping import schema as S
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import contact_maps as CMAP
from ispatialgym.sampling import reach_maps as RM
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.tests.test_contact_library import CEREAL, _job, _model_dir


@pytest.fixture
def glib(tmp_path):
    root = str(tmp_path / "glib")
    _model_dir(root, "box_of_cereal__iucshm__s0.8", CEREAL)
    _model_dir(root, "box_of_cereal__dkhgxn__s1", (0.077, 0.155, 0.242))
    _model_dir(root, "magazine__flat__s1", (0.20, 0.28, 0.012))
    return root


def _lib(glib, eid, fam, kind, valid=lambda i, e: True, **params):
    lib = CL.build_job(_job(eid, [fam], params={fam: dict({"yaws": 2}, **params)}, validate={"support_z": 0.8126}), glib)[kind]
    for i, e in enumerate(lib["entries"]):
        e["validation"] = {"verdict": "valid" if valid(i, e) else "invalid"}
    return lib


def test_the_pose_set_is_the_validated_entries_and_their_executed_pose_content(glib):
    lib = _lib(glib, "box_of_cereal__iucshm__s0.8", "touching_push", "push", valid=lambda i, e: i % 3 != 0)
    every, valid = CMAP.select_entries(lib, "touch_push", None), CMAP.select_entries(lib, "touch_push")
    assert len(every) == len(lib["entries"]) and 0 < len(valid) < len(every) and all(e["validation"]["verdict"] == "valid" for e in valid)
    with pytest.raises(ValueError):
        CMAP.select_entries(lib, "touch_brace")  # a brace map reads a brace library
    yaws, classes, grid = CMAP.axes(valid)
    assert yaws == [0.0, 180.0] and len(grid) == len(valid) and all(grid[(y, g)]["class"] == classes[g] for y, g in grid)
    ps = CMAP.pose_set("touch_push", valid, ("valid",))
    assert ps["n_entries"] == len(valid) and ps["verdicts"] == ["valid"] and ps["frame"] == "start"
    # the identity moves with what the oracle executes - the partner gap, the opening, the executed pre - not with the validation's record
    e = valid[0]
    base = ps["digest"]
    for mut in (lambda x: x["push"].__setitem__("stop_gap_m", 0.03), lambda x: x["fingers"].__setitem__("opening_m", 0.035),
                lambda x: x["pre"]["executed"]["position"].__setitem__(2, 0.2), lambda x: x["pre_pose_rule"].__setitem__("distance_m", 0.05)):
        x = json.loads(json.dumps(e))
        mut(x)
        assert CMAP.pose_set("touch_push", [x] + valid[1:], ("valid",))["digest"] != base
    x = json.loads(json.dumps(e))
    x["validation"] = {"verdict": "valid", "trials": [{"measures": {"moved": 1.0}}]}
    assert CMAP.pose_set("touch_push", [x] + valid[1:], ("valid",))["digest"] == base
    row = CMAP.class_row(e)
    assert row["push"]["stop_gap_m"] == e["push"]["stop_gap_m"] and row["pre_pose_rule"] == e["pre_pose_rule"] and row["fingers"] == e["fingers"]
    # the library's own entry digest covers the fields a family adds after contact_entry (the partner gap, the goal, the yaw slot)
    assert e["digest"] == CL._redigest(json.loads(json.dumps(e)))["digest"]


def test_the_chain_is_what_the_oracle_executes_and_the_cell_is_the_object_root(glib):
    push = CMAP.select_entries(_lib(glib, "box_of_cereal__iucshm__s0.8", "touching_push", "push"), "touch_push")
    e = next(x for x in push if x["class"].startswith("push_tip_h-") and x["yaw_deg"] == 0.0)
    ch = dict(CMAP.chain_world(e, (0.7, -0.1), 0.8126))
    To = CMAP.object_pose(e, (0.7, -0.1), 0.8126)
    assert To[0, 3] == pytest.approx(0.7) and To[1, 3] == pytest.approx(-0.1) and To[2, 3] == pytest.approx(0.8126)  # root at the cell, on the support
    assert np.linalg.norm(ch["pre"][:3, 3] - ch["contact"][:3, 3]) == pytest.approx(0.04, abs=1e-5)  # the executed (lowered) pre, decision 6
    assert np.allclose(ch["goal"][:3, 3] - ch["contact"][:3, 3], [0.0, e["push"]["stroke_m"], 0.0], atol=1e-5)  # the stroke along +y
    brace = CMAP.select_entries(_lib(glib, "box_of_cereal__dkhgxn__s1", "touching_brace", "brace"), "touch_brace")
    assert [n for n, _ in CMAP.chain_object(brace[0])] == ["pre", "contact"]
    arc = CMAP.select_entries(_lib(glib, "magazine__flat__s1", "axial_push", "push", long_axis=[0, 1, 0]), "push_blade")
    assert [n for n, _ in CMAP.chain_object(arc[0])] == ["pre", "contact", "mid", "goal"]
    ab = CMAP.select_entries(_lib(glib, "magazine__flat__s1", "axial_brace", "brace", long_axis=[0, 1, 0]), "axial_brace")
    # an axial brace entry stands at the GOAL pose (its root off the rule origin: the COM stays): the map cell is that pose's root
    T = CMAP.object_pose(ab[0], (0.6, 0.05), 0.75)
    assert T[0, 3] == pytest.approx(0.6) and T[1, 3] == pytest.approx(0.05)
    assert CMAP.entry_rest_quat(ab[0]) == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6)  # the axial identity rest
    q = S.mat_to_quat(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert CMAP.yaw_of(q, [0.0, 0.0, 0.0, 1.0]) == pytest.approx(math.pi / 2)


class _FakeHC:
    def __init__(self, *a, **k):
        pass

    def check(self, T, sb, q, floor=False):
        return (th.ones(T.shape[0], dtype=th.bool),)


def _fake_world(cfg):
    be = SimpleNamespace(ground_contact_links=(), check_collision=lambda q, emb_name=None, margin=0.0: [False],
                         robot_spheres_world=lambda q: (th.zeros(1, 2, 4), ["right_arm_link3", "left_arm_link3"]))
    return SimpleNamespace(cfg=cfg, dev="cpu", backend=be, q_config=lambda tr, ti, p="untucked", fingers=0.05: th.zeros(28),
                           shoulder=lambda tr, ti: (np.array([0.0, 0.0, 1.0]), 0.9), entry=lambda: SimpleNamespace(),
                           link_of_sphere=lambda: ["right_arm_link3", "left_arm_link3"], spheres=lambda q: th.zeros(q.shape[0], 2, 4),
                           full_q=lambda entry, q, q_lock: th.zeros(q.shape[0], 28))


def _compute(monkeypatch, glib, tmp_path, drop_one=True):
    lib = _lib(glib, "box_of_cereal__iucshm__s0.8", "touching_push", "push")
    if drop_one:
        lib["entries"] = lib["entries"][1:]
    monkeypatch.setattr(CP, "HandChecker", _FakeHC)
    monkeypatch.setattr(CP, "sphere_support_ok", lambda sph, ln, sb, arm, policy, **k: (th.ones(sph.shape[0], dtype=th.bool), None))

    def solve(world, entry, T, q_lock, seed_active=None):  # IK reaches x < 0.7 m only
        ok = (T[:, 0, 3] < 0.7).unsqueeze(1).repeat(1, world.cfg.seeds)
        return ok, th.zeros(T.shape[0], world.cfg.seeds, 7)

    monkeypatch.setattr(RM, "_solve", solve)
    monkeypatch.setattr(RM, "robot_digest", lambda: "fakerobot")
    cfg = CMAP.contact_config("touch_push", lattice_m=0.16, seeds=2, edge_distances=(0.45,))
    geom = OG.load(os.path.join(glib, "models", "box_of_cereal__iucshm__s0.8"))
    meta, arrays = CMAP.compute_contact_map(_fake_world(cfg), lib, "touch_push", 0.8126, {"key": "h0.8126", "height": 0.8126}, geom, cfg=cfg)
    return lib, meta, arrays


def test_a_contact_map_is_the_chain_checked_per_cell_yaw_and_class(monkeypatch, glib, tmp_path):
    lib, meta, arrays = _compute(monkeypatch, glib, tmp_path)
    nY, nG = len(meta["pose_set"]["yaws_deg"]), len(meta["pose_set"]["classes"])
    k = next(t["key"] for t in meta["trunks"] if t.get("eligible"))
    st = arrays[f"s0.45_{k}"]
    assert st.shape == (len(arrays["cells"]), nY, nG) and arrays["present"].sum() == len(lib["entries"])
    names = {CMAP.STATUS_NAME[int(v)] for v in np.unique(st)}
    assert {"feasible", "no_entry", "off_support", "reach_bound", "ik_pre"} <= names, names
    assert (st[:, ~arrays["present"]] == CMAP.STATUS["no_entry"]).all()
    cells = arrays["cells"]
    feas = (st == CMAP.STATUS["feasible"]).any(axis=(1, 2))
    assert feas.any() and (cells[feas, 0] < 0.7).all() and (cells[feas, 0] >= 0.45 - 0.08).all()
    assert meta["kind"] == "touch_push" and meta["arm"] == "right" and meta["plane"] == "slab" and meta["robot"] == "fakerobot"
    assert meta["pose_set"]["verdicts"] == ["valid"] and meta["entry_rest_quat_xyzw"] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-6)
    assert len(meta["classes"]) == nY * nG and sum(r is not None for r in meta["classes"]) == len(lib["entries"])
    with pytest.raises(ValueError):
        CMAP.compute_contact_map(_fake_world(CMAP.contact_config("touch_brace")), lib, "touch_push", 0.8126, {}, None)  # the wrong arm


def _map_file(monkeypatch, glib, tmp_path):
    lib, meta, arrays = _compute(monkeypatch, glib, tmp_path, drop_one=False)
    p = CMAP.save(str(tmp_path / "maps"), meta, arrays)
    return lib, meta, arrays, p


def test_a_contact_map_is_found_by_its_identity_and_looked_up_at_a_stance(monkeypatch, glib, tmp_path):
    lib, meta, arrays, p = _map_file(monkeypatch, glib, tmp_path)
    root, entry = str(tmp_path / "maps"), "box_of_cereal__iucshm__s0.8"
    assert os.path.basename(os.path.dirname(p)) == entry and os.path.basename(p).startswith("cmap__touch_push__box_of_cereal__iucshm__h0.8126__")
    kw = dict(rest_id="r0", policy_key=meta["policy_key"], robot="fakerobot")
    cm = CMAP.ContactMap.find(root, "touch_push", entry, 0.8126, **kw)
    assert cm is not None and cm.lookup_how["how"] == "height"
    assert CMAP.ContactMap.find(root, "touch_push", entry, 0.7, **kw) is None  # another height
    assert CMAP.ContactMap.find(root, "touch_brace", entry, 0.8126, **kw) is None  # another kind
    assert CMAP.ContactMap.find(root, "touch_push", entry, 0.8126, arm="left", **{k: v for k, v in kw.items()}) is None
    assert CMAP.ContactMap.find(root, "touch_push", entry, 0.8126, rest_id="r1", policy_key=kw["policy_key"]) is None
    assert CMAP.ContactMap.find(root, "touch_push", entry, 0.8126, pose_set_digest="0" * 16, **kw) is None
    # a stance: translate / tilt of an eligible map trunk, stand-off 0.45; the object's root 0.62 m ahead, 0.1 m right, at the yaw slot 0
    tr, ti = cm.trunks()[0]
    i = int(np.argmin(np.linalg.norm(cm.cells - np.array([0.62, -0.1]), axis=1)))
    xy = cm.cells[i].astype(float)
    out = cm.lookup(0.45, tr, ti, xy, 0.0)
    want = (cm.status(0.45, tr, ti)[i, 0] == CMAP.STATUS["feasible"]).any()
    assert out["ok"] == bool(want) and out["yaw_slot_deg"] == 0.0 and out["cell"] == pytest.approx(xy.tolist(), abs=1e-3)
    assert cm.lookup(0.45, tr, ti, xy + 5.0, 0.0)["why"].startswith("root")  # off the lattice
    assert "slot" in cm.lookup(0.45, tr, ti, xy, math.radians(60.0), yaw_tol=math.radians(30.0))["why"]  # 60 deg from the nearest slot
    assert cm.lookup(0.45, tr, ti, xy, math.radians(60.0))["yaw_slot_deg"] == 0.0  # 2 slots: the default tolerance is half their spacing
    assert cm.lookup(0.45, tr + 0.033, ti, xy, 0.0)["why"].startswith("no eligible map trunk")
    if want:
        assert not cm.lookup(0.45, tr, ti, xy, 0.0, keep=lambda r: False)["ok"]  # the class filter


def _stance(yaw=0.0, x=0.0, y=0.0, translate=None, tilt=0.0):
    edge = EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 1.6)
    return Stance("manipulation", (x, y, yaw), translate, tilt, 0.45, edge)


def test_the_sampler_reads_the_contact_map_of_its_object_and_a_miss_asks_for_it(monkeypatch, glib, tmp_path, caplog):
    lib, meta, arrays, p = _map_file(monkeypatch, glib, tmp_path)
    monkeypatch.setenv("ISPATIALGYM_V2_ONLY", "1")
    monkeypatch.setenv("ISPATIALGYM_REACH_MAPS", str(tmp_path / "maps"))
    monkeypatch.delenv("ISPATIALGYM_REACH_MAP_ANCHORS", raising=False)
    monkeypatch.setattr(TT, "_v2_map_entry", lambda s, spec: (SimpleNamespace(path=os.path.join(glib, "models", spec.grasp_key, "library.json")), "r0")
                        if spec.grasp_key == "box_of_cereal__iucshm__s0.8" else None)
    assert meta["policy_key"] == CP.DEFAULT.key("right")  # the map was computed under the shared contact policy
    assert TT.contact_maps_on()
    models = {"target": SimpleNamespace(category="box_of_cereal", model="iucshm", grasp_key="box_of_cereal__iucshm__s0.8", scale=[0.8] * 3),
              "partner": SimpleNamespace(category="box_of_cereal", model="dkhgxn", grasp_key="box_of_cereal__dkhgxn__s1", scale=None)}
    s = SimpleNamespace(choice=SimpleNamespace(models=models), world=SimpleNamespace(floor_z=0.0, scene_model="house"))
    top = SimpleNamespace(name="bar_0", z=0.8126)
    cm = TT.contact_map(s, "touch_push", "target", top)
    assert cm is not None and cm.path == p and TT.contact_map(s, "touch_push", "target", top) is cm  # cached
    # without a v2 grasp entry the model's contact-library entry is found by its model id at the spawn scale (a pushed / braced object)
    monkeypatch.setattr(TT, "_v2_map_entry", lambda s_, spec: None)
    s2 = SimpleNamespace(choice=SimpleNamespace(models=models), world=SimpleNamespace(floor_z=0.0, scene_model="house"))
    assert TT.contact_map(s2, "touch_push", "target", top).path == p
    with caplog.at_level("INFO"):
        assert TT.contact_map(s, "touch_brace", "partner", top) is None
    assert any("MAP_REQUEST" in r.getMessage() and '"kind": "touch_brace"' in r.getMessage() for r in caplog.records)
    # the check at a stance: the object's world root -> base frame, its yaw from the stance
    tr, ti = cm.trunks()[0]
    st = _stance(yaw=math.pi / 2, x=1.0, y=2.0, translate=tr, tilt=ti)
    i = int(np.argmin(np.linalg.norm(cm.cells - np.array([0.62, -0.1]), axis=1)))
    bx, by = (float(v) for v in cm.cells[i])
    wx, wy = 1.0 - by, 2.0 + bx  # base (bx, by) -> world at yaw 90 deg
    pose = (th.tensor([wx, wy, 0.9]), C.quat_from_yaw(math.pi / 2))  # the object turned with the robot: yaw slot 0
    out = TT.contact_map_check(cm, st, pose)
    assert out["cell"] == pytest.approx([bx, by], abs=1e-3) and out["yaw_slot_deg"] == 0.0 and abs(out["yaw_err_deg"]) < 1e-3
    assert out["ok"] == bool((cm.status(0.45, tr, ti)[i, 0] == CMAP.STATUS["feasible"]).any())


def test_the_touching_gate_and_the_axial_placement_read_their_maps(monkeypatch):
    """Under v2only the touching gate tests the TARGET root in touch_push and the PARTNER root in touch_brace (a miss rejects missing_map);
    the axial joint placement tests the init root in push_blade (the spin's classes at the nearest entry turn) and the GOAL root in
    axial_brace (the spin's classes) - no generic band cell, no left contact band, no proxy box."""
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2.tasks import alignment_axial as AX
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    seen = []

    def fake_check(cm, st, pose, keep=None, tol=0.06):
        rows = [r for r in cm.meta["classes"] if r]
        kept = [r["class"] for r in rows if keep is None or keep(r)]
        seen.append((cm.meta["kind"], kept))
        return {"ok": bool(kept) and cm.meta.get("ok", True), "why": None if kept else "no class", "cell": [0.6, -0.1], "feasible": kept}

    maps = {"touch_push": SimpleNamespace(meta={"kind": "touch_push", "classes": [{"class": "push_tip_h-"}]}),
            "touch_brace": SimpleNamespace(meta={"kind": "touch_brace", "classes": [{"class": "brace_wide_h-"}], "ok": False}),
            "push_blade": SimpleNamespace(meta={"kind": "push_blade", "classes": [{"class": "a@t25", "arc_turn_rad": 0.436}, {"class": "b@t45", "arc_turn_rad": 0.785},
                                                                                {"class": "c@t45", "arc_turn_rad": -0.785}, None]}),
            "axial_brace": SimpleNamespace(meta={"kind": "axial_brace", "classes": [{"class": "+end+spin_v", "goal": {"spin": 1.0}},
                                                                                  {"class": "+end-spin_v", "goal": {"spin": -1.0}}]})}
    monkeypatch.setattr(TT, "contact_map", lambda s, kind, role, top, arm=None: maps.get(kind))
    monkeypatch.setattr(TT, "contact_map_check", fake_check)
    st = _stance(translate=0.5)
    pose = (th.tensor([0.6, -0.1, 0.9]), C.IDENTITY_QUAT.clone())
    me = SimpleNamespace(top=SimpleNamespace(name="bar_0"))
    me.contact_maps_of_pair = lambda: IS.IORSSmall.contact_maps_of_pair(me)
    g = IS.IORSSmall.contact_map_gate(me, st, pose, pose)
    assert g["maps"] and not g["ok"] and g["work"]["feasible"] == ["push_tip_h-"] and "brace OFF" in g["why"]
    maps.pop("touch_brace")
    with pytest.raises(Reject) as ex:
        IS.IORSSmall.contact_map_gate(me, st, pose, pose)
    assert ex.value.reason == "missing_map"
    ax = SimpleNamespace(top=SimpleNamespace(name="desk_0"))
    ax.axial_maps = lambda: AX.AlignmentAxial.axial_maps(ax)
    seen.clear()
    jp = AX.AlignmentAxial.map_joint_placement(ax, st, pose, pose, 1.0, math.radians(50.0))
    assert jp["ok"] and jp["work_ok"] and jp["brace_ok"] and jp["left_band"] is False and jp["maps"]["turn_class_rad"] == pytest.approx(0.785)
    assert seen == [("push_blade", ["b@t45"]), ("axial_brace", ["+end+spin_v"])]  # the spin's class at the nearest turn; the spin's brace
    assert jp["classes"] is None and jp["brace_classes"] is None  # the band class orders keep their defaults


def test_a_missing_map_skips_an_optional_step_and_rejects_a_required_one(monkeypatch):
    """Owner decision 2026-11-07 (item 5): under v2only a missing OPTIONAL map (the axial stop brace) skips the step - the placement stands on
    the push map, the brace reason is ``brace_map_missing``, the skip counted once in ``contact_map_skipped.axial_brace``; a missing REQUIRED
    map (the axial push, both sides of the touching brace gate) rejects ``missing_map``."""
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2.tasks import alignment_axial as AX
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    assert set(TT.CONTACT_MAP_OPTIONAL) == {"axial_brace"} and {"touch_push", "touch_brace", "push_blade"} <= set(TT.CONTACT_MAP_REQUIRED)
    maps = {"push_blade": SimpleNamespace(meta={"kind": "push_blade", "classes": [{"class": "b@t45", "arc_turn_rad": 0.785}]}),
            "touch_push": SimpleNamespace(meta={"kind": "touch_push", "classes": [{"class": "push_tip_h-"}]}),
            "touch_brace": SimpleNamespace(meta={"kind": "touch_brace", "classes": [{"class": "brace_wide_h-"}]})}
    monkeypatch.setattr(TT, "contact_map", lambda s, kind, role, top, arm=None: maps.get(kind))
    monkeypatch.setattr(TT, "contact_map_check", lambda cm, st, pose, keep=None, tol=0.06: {"ok": True, "why": None, "cell": [0.6, -0.1], "feasible": ["x"]})
    st = _stance(translate=0.5)
    pose = (th.tensor([0.6, -0.1, 0.9]), C.IDENTITY_QUAT.clone())
    ax = SimpleNamespace(top=SimpleNamespace(name="desk_0"), kit=SimpleNamespace(stats={}))
    ax.axial_maps = lambda: AX.AlignmentAxial.axial_maps(ax)
    for _ in range(3):
        jp = AX.AlignmentAxial.map_joint_placement(ax, st, pose, pose, 1.0, 0.8)
    assert jp["work_ok"] and not jp["brace_ok"] and not jp["ok"] and jp["brace_map_missing"]  # push-only: the brace step skipped
    assert "skipped" in jp["maps"]["axial_brace"]["why"] and ax.kit.stats == {"contact_map_skipped.axial_brace": 1}
    maps.pop("push_blade")
    with pytest.raises(Reject) as ex:
        AX.AlignmentAxial.axial_maps(ax)
    assert ex.value.reason == "missing_map"
    me = SimpleNamespace(top=SimpleNamespace(name="bar_0"), kit=SimpleNamespace(stats={}))
    assert IS.IORSSmall.contact_maps_of_pair(me) == (maps["touch_push"], maps["touch_brace"])
    for kind in ("touch_push", "touch_brace"):
        m = maps.pop(kind)
        with pytest.raises(Reject) as ex:
            IS.IORSSmall.contact_maps_of_pair(me)
        assert ex.value.reason == "missing_map" and kind in str(ex.value)
        maps[kind] = m
    assert me.kit.stats == {}  # a required miss is a rejection, not a skip
