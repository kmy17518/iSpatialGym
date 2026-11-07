"""i7-contactfu (owner decisions 2026-11-08 01:45, part A): simulator-free tests of

- A1: the touching push stance's stand-off search on the touch_push map (forward / backward, collision-limited) and the partner draw's
  push-side test over the push stance candidates;
- A5: the axial brace pairing (the first draw / candidate stance whose push and brace maps fit on the same pushed end, else push-only);
- A6: the stratified contact-entry validation (per (yaw x end x spin) group) and the learned overlay (harvest -> map_tilt merge -> a
  rebuilt map whose identity carries the overlay version; the frozen map untouched).
"""

from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import validate_contacts as VC
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import contact_maps as CMAP
from ispatialgym.sampling import contact_overlay as CO
from ispatialgym.sampling.v2 import map_tilt as MT
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.tests.test_contact_library import CEREAL, _model_dir
from ispatialgym.tests.test_contact_maps import _compute


@pytest.fixture
def glib(tmp_path):
    root = str(tmp_path / "glib")
    _model_dir(root, "box_of_cereal__iucshm__s0.8", CEREAL)
    return root

# ------------------------------------------------------------------------------------------------ A1


def _edge_stance(stand_off=0.35, lateral=0.0):
    # the edge at x = 1.0 facing +x (inward +x); the base stand_off outside it
    edge = EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 2.0)
    st = Stance("manipulation", (0.0, 0.0, 0.0), 0.25, -0.3, None, edge)
    return st.shifted(stand_off=stand_off, lateral=lateral, name="manipulation")


def _box_hull(pose, half=(0.05, 0.08, 0.12)):
    c = th.as_tensor(pose[0], dtype=th.float32)
    pts = th.tensor([[sx * half[0], sy * half[1], sz * half[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=th.float32)
    return pts + c


def _touching_me(maps, free=lambda st: True, gen=None):
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    gen = dict(gen or {})
    me = SimpleNamespace(relation="touching", shelf_start=None, hull={"target": None}, world=SimpleNamespace(hull_at=lambda h, pose: _box_hull(pose)),
                         gen=lambda k, d=None: gen.get(k, d), ikc=SimpleNamespace(stances_collision_free=lambda sts: th.tensor([bool(free(s)) for s in sts])))
    me.contact_maps_of_pair = lambda: (maps["touch_push"], maps.get("touch_brace"))
    for name in ("push_stance_for", "push_stance_stand_offs", "push_stance_candidates", "push_map_any"):
        setattr(me, name, getattr(IS.IORSSmall, name).__get__(me))
    return me


def _reach_check(max_x):
    """A fake map lookup: a push class is feasible when the object's root lies within ``max_x`` ahead of the base."""

    def check(cm, st, pose, keep=None, tol=0.06):
        x = float(st.world_to_base(th.as_tensor(pose[0])[:2])[0])
        lim = max_x[cm] if isinstance(max_x, dict) else max_x
        ok = x <= lim
        return {"ok": ok, "feasible": ["push_tip_h"] if ok else [], "cell": [round(x, 2), 0.0], "why": None if ok else "no feasible class"}
    return check


def test_the_push_stance_moves_toward_and_away_from_the_edge():
    """Owner A1: a stand-off change moves the base along -inward (closer / farther), keeps the lateral rule and the trunk; with neither a
    lateral nor a stand-off change there is no second base pose."""
    me = _touching_me({"touch_push": "pm"})
    manip = _edge_stance(0.45)
    t_place = (th.tensor([1.30, -0.12 + 0.05, 1.1]), C.IDENTITY_QUAT.clone())  # the target's right face at lateral -0.12 - 0.08 + 0.05
    st = me.push_stance_for(manip, t_place, lateral=-0.12, stand_off=0.25)
    assert st is not None and st.stand_off == pytest.approx(0.25) and st.extras["stand_off_shift_m"] == pytest.approx(-0.20)
    assert float(st.base_xy[0]) == pytest.approx(float(manip.base_xy[0]) + 0.20, abs=1e-5)  # 0.20 m closer to the edge
    assert (st.translate, st.tilt) == (manip.translate, manip.tilt) and st.arms == ["right"]
    # the root is 0.20 m nearer in the base frame
    d0 = float(manip.world_to_base(t_place[0][:2])[0])
    assert float(st.world_to_base(t_place[0][:2])[0]) == pytest.approx(d0 - 0.20, abs=1e-5)
    # no shift at all: the manipulation stance itself
    right_face_lat = float(manip.local(th.tensor([1.30, -0.12 + 0.05 - 0.08]))[1])
    assert me.push_stance_for(manip, t_place, lateral=right_face_lat, stand_off=0.45) is None
    moved = me.push_stance_for(manip, t_place, lateral=right_face_lat, stand_off=0.35)
    assert moved is not None and moved.extras["lateral_shift_m"] == 0.0 and moved.stand_off == pytest.approx(0.35)


def test_the_push_stand_off_is_chosen_from_the_map_and_limited_by_the_collision_check(monkeypatch):
    """Owner A1: the target placed 0.83 m ahead of the 0.35 stand-off, the map reaching 0.75 m: only stand-offs <= 0.27 m fit; the
    collision check (the base against the table) cuts the closest; brace-map candidates first, then the stand-off nearest the manipulation
    stance's; push_map_any answers the partner draw."""
    monkeypatch.setattr(TT, "contact_map_check", _reach_check({"pm": 0.75, "bm": 0.70}))
    me = _touching_me({"touch_push": "pm", "touch_brace": "bm"}, free=lambda st: st.stand_off is None or st.stand_off >= 0.20 - 1e-6)
    manip = _edge_stance(0.35)
    t_place = (manip.base_to_world(th.tensor([0.83, -0.10])).tolist() + [1.1], C.IDENTITY_QUAT.clone())
    t_place = (th.tensor(t_place[0]), t_place[1])
    p_pose = (th.tensor(manip.base_to_world(th.tensor([0.80, 0.10])).tolist() + [1.1]), C.IDENTITY_QUAT.clone())
    rows, facts = me.push_stance_candidates(manip, t_place, [-0.12, -0.16], p_pose=p_pose)
    sos = [round(c.stand_off, 2) for _l, c, _f in rows]
    assert facts["map_feasible"] > 0 and set(sos) <= {0.2, 0.25} and 0.15 not in sos  # 0.15 collides (the footprint limit)
    assert rows[0][2]["brace_map"]  # partner 0.80 ahead: the brace map (<= 0.70) fits at 0.20 only -> that candidate first
    assert rows[0][1].stand_off == pytest.approx(0.20)
    # without any brace fit: the nearest stand-off to the manipulation stance's first (0.25 before 0.20), laterals in order
    monkeypatch.setattr(TT, "contact_map_check", _reach_check({"pm": 0.75, "bm": 0.0}))
    rows, _ = me.push_stance_candidates(manip, t_place, [-0.12, -0.16], p_pose=p_pose)
    assert [(lat, round(c.stand_off, 2)) for lat, c, _f in rows][:2] == [(-0.12, 0.25), (-0.12, 0.2)]
    a = me.push_map_any(manip, t_place, p_pose)
    assert a["push_any"] and not a["pair_any"]
    # a target beyond the map from every stand-off: no candidate - the partner draw rejects it (A1 part 2)
    far = (th.tensor(manip.base_to_world(th.tensor([1.05, -0.10])).tolist() + [1.1]), C.IDENTITY_QUAT.clone())
    assert not me.push_map_any(manip, far, p_pose)["push_any"]
    assert me.push_stance_stand_offs(manip) == [0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.55]


# ------------------------------------------------------------------------------------------------ A5


def _jp(work, brace):
    return {"ok": bool(work and brace), "work_ok": bool(work), "brace_ok": bool(brace)}


def _draw(tag, *ends):
    return (tag, 0.0, 0.0, None, [(None, _jp(w, b), None, None, None) for w, b in ends])


def test_the_axial_brace_pairs_the_same_end_first_else_push_only(monkeypatch):
    """Owner A5: the push map and the brace map on the SAME pushed end - the first draw with such a pair wins (one end's push + the other
    end's brace is no pair), else the first push-feasible draw (push-only); the candidate scan puts the stances with a pair first."""
    from ispatialgym.sampling.v2.tasks import alignment_axial as AX

    me = SimpleNamespace()
    me._draw_paired = AX.AlignmentAxial._draw_paired
    crossed = _draw("crossed", (True, False), (False, True))  # +end push only, -end brace only (i7_contact2's 11 / 11)
    paired = _draw("paired", (True, False), (True, True))
    assert not AX.AlignmentAxial._draw_paired(crossed) and AX.AlignmentAxial._draw_paired(paired)
    assert AX.AlignmentAxial._pick_draw(me, [crossed, paired])[0] == "paired"
    assert AX.AlignmentAxial._pick_draw(me, [crossed])[0] == "crossed"  # push-only
    assert AX.AlignmentAxial._pick_draw(me, []) is None
    # the scan over candidate stances: candidate 3 has no pair, 5 has one, 7 is push-infeasible
    per_k = {3: [crossed], 5: [crossed, paired], 7: []}
    cands = {k: SimpleNamespace(translate=0.25, tilt=-0.3, focus=th.zeros(3), with_trunk=lambda tr, ti, n, k=k: SimpleNamespace(k=k, focus=None)) for k in per_k}
    me2 = SimpleNamespace(manip_candidates=cands, top=SimpleNamespace(z=0.8), extents={"height": 0.03, "long": 0.4}, camera_pose_at=lambda st: None,
                          _aligned_yaw=lambda st: 0.0, _blade_geometry=lambda p, c: (0.0, 0.01, 0.0))
    me2._draw_paired = AX.AlignmentAxial._draw_paired
    me2._pick_draw = lambda draws: AX.AlignmentAxial._pick_draw(me2, draws)
    me2._placement_draws = lambda rng, manip, *a, **k: (per_k[manip.k], {"support": 0})
    scan = AX.AlignmentAxial._scan_candidates(me2, None, SimpleNamespace(), [3, 5, 7])
    assert [(k, d[0]) for k, d, _f in scan] == [(5, "paired"), (3, "crossed")]


# ------------------------------------------------------------------------------------------------ A6: stratified validation


def _entries(groups, n):
    out = []
    for y in (0.0, 90.0):
        for g in groups:
            for i in range(n):
                out.append({"class": f"{g}_c{i}", "yaw_deg": y, "family": "axial_push" if g else "touching_push"})
    return out


def test_the_stratified_schedule_stops_each_group_at_three_valid_or_its_cap():
    es = _entries(["+end+spin", "-end-spin"], 6)
    verdict = {e["class"] + str(e["yaw_deg"]): ("valid" if (e["yaw_deg"] == 0.0 or e["class"].endswith(("c0", "c4"))) else "invalid") for e in es}
    q = VC.GroupQuota(es, per_valid=3, per_max=5)
    ran = []
    while True:
        b = q.next_round()
        if not b:
            break
        assert len({VC.entry_group(e) for e in b}) == len(b)  # one entry per group per round
        for e in b:
            ran.append(e)
            q.done(e, verdict[e["class"] + str(e["yaw_deg"])])
    by = {}
    for e in ran:
        by.setdefault(VC.entry_group(e), []).append(e["class"])
    assert by[(0.0, "+end+spin")] == ["+end+spin_c0", "+end+spin_c1", "+end+spin_c2"]  # 3 valid: done
    assert len(by[(90.0, "+end+spin")]) == 5  # c0 and c4 valid only: the cap (5 tried)
    summ = q.summary(2)
    assert summ["n_groups"] == 4 and summ["n_short"] == 0 and summ["valid_hist"] == {"0": 0, "1": 0, "2": 2, "3": 2}
    # kept verdicts count first and are not re-run
    es2 = _entries([""], 4)
    es2[0]["validation"] = {"verdict": "valid"}
    es2[1]["validation"] = {"verdict": "valid"}
    q2 = VC.GroupQuota(es2, per_valid=3, per_max=8, keep_verdicts=True)
    first = q2.next_round()
    assert [e["class"] for e in first] == ["_c2", "_c0"]  # yaw 0: c2 (c0 / c1 kept); yaw 90: c0
    assert q2.state[(0.0, "")]["valid"] == 2 and q2.state[(0.0, "")]["kept"] == 2


class _World:
    def __init__(self, valid):
        self.valid, self.calls = valid, []

    def settle(self, obj, geom, q0):
        return {"world_position": [0, 0, 0.1], "world_quat_xyzw": [0, 0, 0, 1]}, None

    def plan_entry(self, obj, geom, e, T, arm):
        return {"q": [0.0]}

    def trial_push(self, obj, e, plan, state, rest, arm):
        self.calls.append(e["class"])
        return {"verdict": "valid" if self.valid(e) else "invalid", "reasons": [] if self.valid(e) else ["moved_short"]}


def test_validate_library_stratified_fresh_clears_the_old_verdicts_and_enforces_the_screen():
    es = _entries([""], 6)
    es[5]["sampler_screen"] = {"hand_overlap": "knuckle -12 mm"}
    for e in es:
        e["validation"] = {"verdict": "invalid"}  # an earlier run (e.g. the dynamic partner stand-in)
    lib = {"kind": "push", "object": {"rest": {"quat_xyzw": [0, 0, 0, 1]}, "rest_id": "r0"}, "entries": es}
    w = _World(lambda e: e["class"] != "_c1")
    cfg = VC.ContactValidateConfig(per_group_valid=3, per_group_max=8)
    tim = VC.validate_library(w, lib, None, None, np.zeros((8, 3)), cfg, "right")
    verdicts = {(e["yaw_deg"], e["class"]): (e.get("validation") or {}).get("verdict") for e in es}
    assert verdicts[(0.0, "_c0")] == "valid" and verdicts[(0.0, "_c1")] == "invalid" and verdicts[(0.0, "_c3")] == "valid"
    assert verdicts[(0.0, "_c4")] is None and "validation_prev" in es[4]  # not reached: the old verdict is gone (kept as validation_prev)
    assert verdicts[(0.0, "_c5")] == "screened"  # A3: the hand-overlap screen stays enforced
    assert lib["validation_groups"]["n_groups"] == 2 and lib["validation_groups"]["n_short"] == 0 and tim["n_cleared"] >= 2
    assert len(w.calls) == 8  # per yaw: c0 c1 (invalid) c2 c3


# ------------------------------------------------------------------------------------------------ A6: the learned overlay


def test_a_successful_demo_joins_its_entry_through_the_overlay_into_a_new_map_version(monkeypatch, glib, tmp_path):
    """Owner A6: the frozen map (valid entries) is never edited; a successful demo's entry -> an ok event in the job's own shard -> the
    map_tilt merge -> v1 -> the rebuilt map holds the entry, its identity carries (lineage, v1); a failed demo teaches nothing."""
    lib, meta0, _arr = _compute(monkeypatch, glib, tmp_path, drop_one=False)
    lib_e = lib["entries"]
    target = lib_e[0]
    target["validation"] = {"verdict": "invalid"}  # the physics validation rejected it ...
    _lib2, meta_v0, _ = _compute_with(monkeypatch, glib, tmp_path, lib)
    assert all(r is None or r["class"] != target["class"] or r["yaw_deg"] != target["yaw_deg"] for r in meta_v0["classes"])
    assert "overlay" not in meta_v0
    # ... but a real demo executed it and succeeded
    cm = SimpleNamespace(meta=meta_v0, path="/x/cmap.json", yaws=np.radians(np.array(meta_v0["pose_set"]["yaws_deg"])),
                         class_rows=lambda yi: CMAP.ContactMap.class_rows(SimpleNamespace(classes=meta_v0["pose_set"]["classes"], meta=meta_v0), yi))
    job = tmp_path / "job"
    inst_dir = job / "inst" / "t" / "x" / "instances"
    inst_dir.mkdir(parents=True)
    rows = []
    for name, ok in (("inst_ok", True), ("inst_fail", False)):
        summ = {}
        rec = CO.note_used(summ, cm, "touch_push", target["class"], {"yaw_slot_deg": target["yaw_deg"], "yaw_err_deg": 1.0, "feasible": []}, "target")
        assert rec["in_map"] is False and rec["lineage"] == meta_v0["key"]
        (inst_dir / f"{name}.summary.json").write_text(json.dumps(summ))
        rows.append({"instance": name, "success": ok, "status": "success" if ok else "failed", "tro": str(inst_dir / f"{name}-tro_state.json")})
    (job / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    h = CO.harvest(str(job))
    assert h["successes"] == 1 and h["events"] == 1
    assert CO.harvest(str(job))["skipped_existing"] == 1  # immutable per success: a second harvest adds nothing
    shards = [str(p) for p in job.glob("map_overlay/*/*.jsonl")]
    root = str(tmp_path / "overlay")
    res = MT.merge(shards, root)
    assert res[meta_v0["key"]]["version"] == 1
    conf, v = CO.confirmed(root, meta_v0["key"])
    assert v == 1 and (round(target["yaw_deg"], 4), target["class"]) in conf
    # the rebuild: the entry joins, the identity moves with the overlay version, the lineage is the frozen map's key
    _lib3, meta_v1, _ = _compute_with(monkeypatch, glib, tmp_path, lib, overlay={"root": root})
    assert meta_v1["overlay"]["lineage"] == meta_v0["key"] and meta_v1["overlay"]["version"] == 1
    assert [j["class"] for j in meta_v1["overlay"]["joined"]] == [target["class"]] and meta_v1["overlay"]["joined"][0]["verdict_before"] == "invalid"
    assert meta_v1["key"] != meta_v0["key"] and CO.lineage_key(meta_v1) == meta_v0["key"]
    assert meta_v1["pose_set"]["n_entries"] == meta_v0["pose_set"]["n_entries"] + 1
    # the lineage of a learned map is computable before the map (identity_meta) and equals the frozen map's key
    cfg = CMAP.contact_config("touch_push", lattice_m=0.16, seeds=2, edge_distances=(0.45,))
    from ispatialgym.grasping import contact_policy as CP

    assert CMAP.map_key(CMAP.identity_meta(lib, "touch_push", 0.8126, cfg, CP.DEFAULT, CMAP.select_entries(lib, "touch_push"), ("valid",),
                                           robot="fakerobot")) == meta_v0["key"]
    # no merged version: the map is the lineage's frozen map
    _l4, meta_none, _ = _compute_with(monkeypatch, glib, tmp_path, lib, overlay={"root": str(tmp_path / "empty")})
    assert "overlay" not in meta_none and meta_none["key"] == meta_v0["key"]


def _compute_with(monkeypatch, glib, tmp_path, lib, overlay=None):
    from ispatialgym.grasping import object_geometry as OG
    from ispatialgym.tests import test_contact_maps as TCM

    cfg = CMAP.contact_config("touch_push", lattice_m=0.16, seeds=2, edge_distances=(0.45,))
    geom = OG.load(os.path.join(glib, "models", "box_of_cereal__iucshm__s0.8"))
    meta, arrays = CMAP.compute_contact_map(TCM._fake_world(cfg), lib, "touch_push", 0.8126, {"key": "h0.8126", "height": 0.8126}, geom, cfg=cfg,
                                            overlay=overlay)
    return lib, meta, arrays


def test_the_overlay_entry_ids_round_trip():
    assert CO.parse_entry_id(CO.entry_id(90.0, "push_blade_lean20@3.5cm#")) == (90.0, "push_blade_lean20@3.5cm#")
    assert CO.parse_entry_id("0.1,0.2") is None  # a map_tilt cell entry is not a contact entry
    assert math.isclose(CO.parse_entry_id(CO.entry_id(-22.5, "+end-spin_45-_side@t25"))[0], -22.5)
