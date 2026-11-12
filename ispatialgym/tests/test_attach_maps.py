"""i7-attachmap 2026-11-07: the ATTACH reach map (``sampling/attach_maps.py``) - the oracle's hover rules (box: parity with
``execute.attach_hover_targets``; swept: the exact first touch), the chain (pre -> hover -> touchdown -> seat), the map on a fake world
(statuses incl. the chain's branch, the partner check), its identity (``map_key`` / ``AttachMap.find``), the band view the attachment task
reads, the live pair check, and the task's consumer (v2 only: the map or ``missing_map``; otherwise the legacy band, unchanged), plus the
template's insertion-axis rule moved into ``attachment.insertion_axis_rule`` unchanged. Simulator-free."""

from __future__ import annotations

import json
import logging
import math
import os
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import hand_geometry as H
from ispatialgym.grasping import schema as S
from ispatialgym.oracle import execute as EX
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import attach_maps as AM
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM

PARAMS_BOX = {"rule": "box", "clear_m": 0.003, "footprint_margin_m": 0.01, "touchdown_extra_m": 0.004, "axis_z_min": 0.5, "method": "t"}
PARAMS_SWEPT = dict(PARAMS_BOX, rule="swept")


def _box(x0, x1, y0, y1, z0, z1):
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float64)


def _disc(r, z0, z1, n=16, cx=0.0, cy=0.0):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    return np.concatenate([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)]])


def _pot():
    """A pot (root frame, its bottom at z 0): a rim ring of 8 pieces (r 0.090-0.105, top 0.060) and two loop handles OUTSIDE the lid's
    radius (r 0.119-0.145, top 0.085) - the stockpot oshwps' geometry in the attachment_04 diagnosis, simplified."""
    pieces = []
    for k in range(8):
        a0, a1 = 2 * math.pi * k / 8, 2 * math.pi * (k + 1) / 8
        pts = [[r * math.cos(a), r * math.sin(a), z] for r in (0.090, 0.105) for a in (a0, a1) for z in (0.0, 0.060)]
        pieces.append(np.array(pts))
    pieces.append(_box(0.119, 0.145, -0.02, 0.02, 0.050, 0.085))
    pieces.append(_box(-0.145, -0.119, -0.02, 0.02, 0.050, 0.085))
    return pieces


#: the lid seated: its root 0.058 above the pot root, its rim disc (r 0.111) from 0.050 (1 cm into the rim: the meta-link seat lies below
#: the physical rest, as pbmsji's 12.3 mm) to 0.062, a knob to 0.075
T_PM = S.pose_mat([0.0, 0.0, 0.058], [0.0, 0.0, 0.0, 1.0])
LID = [_disc(0.111, -0.008, 0.004), _disc(0.012, 0.004, 0.017)]
#: a top-down pinch of the knob (eef 0.01 above the knob top, approach -z)
T_OE = S.pose_mat([0.0, 0.0, 0.027], [1.0, 0.0, 0.0, 0.0])


def _spec(**kw):
    pieces = _pot()
    data = {"schema": AM.SPEC_SCHEMA, "id": "lid_pot", "male": {"entry": "lid__abc__s1__attachment_free", "category": "lid", "model": "abc", "scale": [1.0] * 3,
                                                               "geometry_digest": AM.pieces_digest(LID)},
            "partner": {"category": "stockpot", "model": "pot", "scale": [1.0] * 3, "geometry_digest": AM.pieces_digest(pieces), "bottom_offset_m": 0.0,
                        "drop_m": 0.003},
            "attachment": {"child_link": "M", "parent_link": "F", "T_partner_male": S.mat_pose(T_PM)},
            "insertion_axis": {"local": [0.0, 0.0, 1.0]}, "seat_axis": {"local": [0.0, 0.0, 1.0]}}
    for k, v in kw.items():
        data[k] = v
    return AM.AttachSpec(data, pieces)


# ---------------------------------------------------------------------------------------------------------------------- the hover rules


def test_first_touch_is_the_exact_translational_contact():
    a = [_box(-0.01, 0.01, -0.01, 0.01, 0.10, 0.12)]
    b = [_box(-0.05, 0.05, -0.05, 0.05, 0.0, 0.04)]
    # coming down the axis from far above, the moving box first touches the fixed one 6 cm below where it is (t: along the axis)
    assert AM.first_touch(a, b, [0, 0, 1])["t"] == pytest.approx(-0.06, abs=1e-12)
    assert AM.first_touch(a, b, [0, 0, -1])["t"] == pytest.approx(0.12, abs=1e-12)  # coming from below: its top meets the fixed bottom
    assert AM.first_touch([_box(0.2, 0.3, 0.0, 0.1, 0.1, 0.2)], b, [0, 0, 1])["t"] is None  # footprints across the axis miss
    tilted = AM._unit([0.3, 0.0, 1.0])  # along a tilted axis the touch is where the boxes' corners meet
    t = AM.first_touch(a, b, tilted)["t"]
    moved = [a[0] + t * tilted]
    assert AM.pair_first_touch(moved[0], b[0], tilted) == pytest.approx(0.0, abs=1e-9)


def test_swept_first_touch_is_invariant_to_a_common_turn_about_z():
    spec, hand = _spec(), H.load("right")
    g = {"name": "g", "T": T_OE, "finger_q": [0.012, 0.012]}
    rows = AM.hover_table(spec, LID, hand, [g], [0.0, 0.7, 2.1], PARAMS_SWEPT)
    ts = [r[0]["first_touch_m"] for r in rows]
    assert ts[0] == pytest.approx(0.010, abs=1e-6) and ts == pytest.approx([ts[0]] * 3, abs=1e-9)  # the rim at 0.060, the lid's bottom 0.050


def test_box_rule_takes_the_handles_and_the_lid_drops_swept_reaches_the_first_touch():
    """attachment_04's diagnosis on the synthetic pot: the box rule's partner top is the handles' (0.085, outside the lid) - the hover 3.8 cm
    above the seat, a 7 mm touchdown that stops 2.1 cm above the first touch; the swept rule hovers 3 mm above the first touch (1 cm above
    the seat) and its touchdown ends 4 mm past it."""
    spec, hand = _spec(), H.load("right")
    g = {"name": "g", "T": T_OE, "finger_q": [0.012, 0.012]}
    box = AM.hover_table(spec, LID, hand, [g], [0.0], PARAMS_BOX)[0][0]
    assert box["source"] == "box" and box["box_top_z"] == pytest.approx(0.085) and box["h_m"] == pytest.approx(0.085 + 0.003 - 0.050, abs=1e-6)
    assert box["part_clearance_m"] == pytest.approx(0.003, abs=1e-6) and box["depth_m"] == pytest.approx(0.007, abs=1e-6)
    assert box["touchdown_m"] == pytest.approx(0.031, abs=1e-6) and box["gap_to_first_touch_m"] == pytest.approx(0.028, abs=1e-6)
    sw = AM.hover_table(spec, LID, hand, [g], [0.0], PARAMS_SWEPT)[0][0]
    assert sw["source"] == "swept" and sw["h_m"] == pytest.approx(0.013, abs=1e-6) and sw["touchdown_m"] == pytest.approx(0.006, abs=1e-6)
    assert sw["box_h_m"] == pytest.approx(box["h_m"]) and sw["lowest"] == "part"


def test_swept_without_a_touch_falls_back_to_the_box_rule():
    bx = {"h_m": 0.02, "part_clearance_m": 0.003, "lowest": "part"}
    out = AM.hover_of_rule("swept", bx, None, None, 0.003, 0.004)
    assert out["source"] == "box" and out["h_m"] == 0.02 and "swept_fallback" in out and out["touchdown_m"] == pytest.approx(0.013)
    with pytest.raises(ValueError):
        AM.hover_of_rule("footprint", bx, None, None, 0.003, 0.004)
    # a part clearance of exactly 0 is the oracle's ``or``: the touchdown depth falls back to the clearance setting
    assert AM.hover_of_rule("box", dict(bx, part_clearance_m=0.0), None, None, 0.003, 0.004)["depth_m"] == pytest.approx(0.007)


def test_box_rule_is_the_oracles_attach_hover_targets(monkeypatch):
    """Parity with the oracle: the same pieces through ``execute.attach_hover_targets`` (the hand 10 cm above its seat, the part / fingertips
    carried to the seat by the oracle) and through ``attach_maps.hover_box`` at the seat - the height, the part's clearance, the top."""
    monkeypatch.setattr(EX, "_seat_axis", lambda state, action: th.tensor([0.0, 0.0, 1.0]))
    spec, hand = _spec(), H.load("right")
    yaw = 0.4
    R = AM._rz(yaw)
    Tp = R.copy()
    Tp[:3, 3] = [0.6, -0.1, 0.80]
    T_seat_eef = Tp @ T_PM @ T_OE
    up = np.eye(4)
    up[2, 3] = 0.10
    T_now_eef = up @ T_seat_eef  # the hand 10 cm above the seat: the oracle carries the geometry from here
    fq = [0.012, 0.012]
    part_now = [AM._tf(up @ Tp @ T_PM, p) for p in LID]
    fing_now = [AM._tf(T_now_eef, p) for p in AM.finger_pieces(hand, fq)]
    partner_w = [AM._tf(Tp, p) for p in spec.partner_pieces]
    tt = lambda P: th.as_tensor(np.concatenate(P), dtype=th.float32)  # noqa: E731
    part = NS(name="lid_0", links={"base": NS(collision_boundary_points_world=tt(part_now))})
    pot = NS(name="pot_0", links={"base": NS(collision_meshes={f"m{k}": NS(points_in_world_frame=th.as_tensor(p, dtype=th.float32)) for k, p in enumerate(partner_w)})})
    finger = NS(collision_boundary_points_world=tt(fing_now))
    q_now = S.mat_to_quat(T_now_eef[:3, :3])
    state = NS(object=lambda n: part if n == "lid_0" else pot, eef_pose=lambda arm: (th.tensor(T_now_eef[:3, 3], dtype=th.float32), th.tensor(q_now, dtype=th.float32)))
    motions = NS(robot=NS(finger_links={"right": [finger]}))
    targets = {"place": EX.Pose(position=T_seat_eef[:3, 3].tolist(), orientation=list(q_now)), "descent": [], "retreat": []}
    for rule, params in (("box", PARAMS_BOX), ("swept", PARAMS_SWEPT)):
        if rule == "swept" and not hasattr(EX, "swept_first_touch"):
            continue  # an oracle without the swept rule
        ctx = {}
        cfg = EX.ExecuteConfig(attach_hover_rule=rule)
        EX.attach_hover_targets(NS(object="lid_0", on="pot_0", arm="right"), motions, state, targets, ctx, cfg)
        rec = ctx["attach"]["hover"]
        mine = AM.hover_table(spec, LID, hand, [{"name": "g", "T": T_OE, "finger_q": fq}], [yaw], params)[0][0]
        assert rec["applied"] and rec.get("source", "box") == mine["source"] == rule
        assert rec["height_above_seat_m"] == pytest.approx(mine["h_m"], abs=2e-4) and rec["part_clearance_m"] == pytest.approx(mine["part_clearance_m"], abs=2e-4)
        if rule == "box":
            assert rec["partner_top_z"] == pytest.approx(mine["box_top_z"] + 0.80, abs=1e-4)
        else:
            assert rec["first_touch_m"] == pytest.approx(mine["first_touch_m"], abs=1e-5)  # the oracle's convex_sweep = the map's Minkowski cast
        assert mine["depth_m"] == pytest.approx(float(rec["part_clearance_m"] or cfg.attach_hover_clear_m) + cfg.attach_touchdown_extra_m, abs=2e-4)


def test_hover_params_come_from_the_oracle_and_follow_its_switch(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_ATTACH_HOVER", raising=False)
    p = AM.oracle_hover_params()
    cfg = EX.ExecuteConfig()
    assert p["rule"] == AM.oracle_hover_rule() == EX.attach_hover_rule(cfg) == "swept"  # owner 2026-11-07: swept is the oracle's default
    monkeypatch.setenv("ISPATIALGYM_ATTACH_HOVER", "box")
    assert AM.oracle_hover_params()["rule"] == "box"  # the map the sampler reads follows the rule the oracle runs
    assert (p["clear_m"], p["footprint_margin_m"], p["touchdown_extra_m"]) == (cfg.attach_hover_clear_m, cfg.attach_hover_footprint_margin_m, cfg.attach_touchdown_extra_m)
    assert AM.oracle_hover_params("swept")["rule"] == "swept"
    with pytest.raises(ValueError):
        AM.oracle_hover_params("footprint")


# ---------------------------------------------------------------------------------------------------------------------- the chain


def test_chain_offsets_follow_the_axes():
    spec = _spec(insertion_axis={"local": [0.0, 0.6, 0.8]}, seat_axis={"local": [0.0, 0.0, 1.0]})
    hover = [[{"h_m": 0.02, "touchdown_m": 0.005}], [{"h_m": 0.03, "touchdown_m": 0.01}]]
    off = AM.chain_offsets(spec, hover, [0.0, math.pi / 2], 0.10)
    assert off.shape == (2, 1, 4, 3)
    assert off[0, 0, 0] == pytest.approx([0.0, 0.06, 0.08]) and off[1, 0, 0] == pytest.approx([-0.06, 0.0, 0.08], abs=1e-9)  # turned with the partner
    assert off[1, 0, 1] == pytest.approx([0.0, 0.0, 0.03]) and off[1, 0, 2] == pytest.approx([0.0, 0.0, 0.01]) and (off[:, :, 3] == 0).all()
    TS = AM.seat_eef(spec, [{"T": T_OE}], [0.0])
    assert TS[0, 0, :3, 3] == pytest.approx([0.0, 0.0, 0.003 + 0.058 + 0.027])  # the drop, the seat, the grasp


# ---------------------------------------------------------------------------------------------------------------------- the map on a fake world


class _FakeWorld:
    """``compute_attach_map``'s SynthWorld surface on CPU: a configuration is the eef pose (flattened 4x4, the IK's "solution" of a pose is the
    pose); one arm sphere (r 0.02) at the eef + ``sphere_offset`` (world); a trunk with tilt < -0.5 collides."""

    def __init__(self, cfg):
        self.cfg, self.dev = cfg, th.device("cpu")
        self.backend = NS(check_collision=lambda q, emb_name, margin: th.tensor([bool(q[0, 1] < -0.5)]),
                          robot_spheres_world=lambda q: (th.tensor([[[0.0, 0.0, 2.0, 0.05]]]), ["torso_link4"]), position_tolerance=0.005, orientation_tolerance=0.05)
        self.sphere_offset = th.tensor([0.0, 0.0, 0.05])

    def q_config(self, translate, tilt, posture="untucked", fingers=0.05):
        return th.tensor([translate, tilt, fingers])

    def shoulder(self, translate, tilt):
        return np.array([0.3, -0.2, 1.0]), 0.9

    def entry(self):
        return None

    def link_of_sphere(self):
        return ["right_arm_link2"]

    def full_q(self, entry, q_active, q_lock):
        return q_active

    def spheres(self, q_full):
        T = q_full.reshape(-1, 4, 4)
        c = T[:, :3, 3] + self.sphere_offset
        return th.cat([c, th.full((len(c), 1), 0.02)], 1)[:, None, :]

    def eef(self, q_full):
        return q_full.reshape(-1, 4, 4)


def _solver(jump_rows=None):
    def solve(world, entry, T, q_lock, seed_active=None):
        n, S_ = T.shape[0], world.cfg.seeds
        ok = (T[:, 0, 3] < 0.95)[:, None].repeat(1, S_)  # a pose beyond 0.95 m ahead "has no IK"
        q = T.reshape(n, 1, 16).repeat(1, S_, 1).clone()
        if jump_rows is not None and seed_active is not None:
            bad = jump_rows(T)
            q[bad] += 5.0  # every seed of these problems lands on another branch
        return ok, q
    return solve


def _lib(tmp_path):
    d = tmp_path / "glib" / "models" / "lid__abc__s1__attachment_free"
    d.mkdir(parents=True)
    rest = {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0.0, 0.0, 0.0, 1.0], "h_root": 0.008, "stable": True}
    data = {"schema": "isg-grasp-library/2", "category": "lid", "model": "abc", "scale": [1.0, 1.0, 1.0], "constraint": "attachment_free", "rests": {"r0": rest},
            "families": {"f1": {"type": "none"}}, "grasps": [{"name": "top_1", "rest_id": "r0", "family_id": "f1", "T_object_eef": S.mat_pose(T_OE),
                                                              "finger_q": [0.013, 0.011], "width_m": 0.024, "preshape_q": 0.022}]}
    (d / "library.json").write_text(json.dumps(data))
    lib = S.ModelLibrary(path=str(d / "library.json"), data=data, rests={"r0": S.Rest.from_record(rest)})
    return lib


def _compute(tmp_path, monkeypatch, params=PARAMS_BOX, jump_rows=None, sphere=(0.0, 0.0, 0.05), spec=None, **kw):
    monkeypatch.setattr(RM, "_solve", _solver(jump_rows))
    monkeypatch.setattr(RB, "trunk_grid", lambda h: [(0.25, -0.3), (0.5, -0.8)])
    cfg = RM.MapConfig(seeds=2, lattice_m=0.16, n_yaws=4)
    w = _FakeWorld(cfg)
    w.sphere_offset = th.tensor(sphere, dtype=th.float32)
    lib = _lib(tmp_path)
    meta, arrays = AM.compute_attach_map(w, _spec() if spec is None else spec, lib, "r0", 0.7853, {"key": "0.7853", "surfaces": ["office/desk/top"]}, male=LID, hand=H.load("right"),
                                         params=params, pre_back_m=0.10, cfg=cfg, **kw)
    return meta, arrays, lib, cfg


def test_compute_attach_map_on_a_fake_world(tmp_path, monkeypatch):
    # the arm sphere 10 cm below the eef: inside the 4 cm slab at the seat (the eef 8.8 cm above the top), over the pot's open centre
    meta, arrays, lib, cfg = _compute(tmp_path, monkeypatch, sphere=(0.0, 0.0, -0.10))
    assert meta["kind"] == "attach" and meta["key"] == AM.map_key(meta) and [g["name"] for g in meta["grasps"]] == ["top_1", "top_1~r180"]
    assert [g["finger_q"] for g in meta["grasps"]] == [[0.013, 0.011], [0.011, 0.013]]  # the twin's fingers swapped
    cells = arrays["cells"]
    st = {d: arrays[f"s{d:.2f}_t0.25_k-0.3"] for d in cfg.edge_distances}
    assert all(a.shape == (len(cells), 4, 2) for a in st.values()) and (arrays["s0.45_t0.50_k-0.8"] == AM.STATUS["trunk_ineligible"]).all()
    ci = lambda x, y: int(np.argmin(np.abs(cells - np.array([x, y])).sum(axis=1)))  # noqa: E731
    # the partner footprint (r 0.145) behind the edge: a root 0.35 m ahead is off the support at every d_e, 0.67 on it
    assert (st[0.35][ci(0.35, 0.0)] == AM.STATUS["off_support"]).all()
    # ... slab_spheres on every reachable cell; nothing else rejects it
    assert (st[0.45][ci(0.67, 0.0)] == AM.STATUS["slab_spheres"]).all()
    meta2, arrays2, _, _ = _compute(tmp_path / "b", monkeypatch)  # a sphere 5 cm above the eef: clear of everything
    s2 = arrays2["s0.45_t0.25_k-0.3"]
    assert (s2[ci(0.67, 0.0)] == AM.STATUS["feasible"]).all() and (s2[ci(0.99, 0.0)] == AM.STATUS["ik_pre"]).all()  # 0.99 m: no IK of the pre
    assert (s2[ci(1.15, 0.8)] == AM.STATUS["reach_bound"]).all()
    assert meta2["trunks"][0]["n_cells"]["0.45"] == int((s2 == AM.STATUS["feasible"]).any(axis=(1, 2)).sum()) > 0
    assert meta2["hover"]["rule"] == "box" and meta2["hover"]["table"][0][0]["h_m"] == pytest.approx(0.038, abs=1e-6)


def test_the_arm_against_the_partner_and_the_chain_branch(tmp_path, monkeypatch):
    # a sphere over the pot's rim ring (9.75 cm off the axis), 5 cm below the eef: it meets the rim at the hover / touchdown, never the slab
    meta, arrays, _, _ = _compute(tmp_path, monkeypatch, sphere=(0.0975, 0.0, -0.05))
    cells = arrays["cells"]
    ci = lambda x, y: int(np.argmin(np.abs(cells - np.array([x, y])).sum(axis=1)))  # noqa: E731
    assert (arrays["s0.45_t0.25_k-0.3"][ci(0.67, 0.0)] == AM.STATUS["partner_arm"]).all()
    # every touchdown pose of the cells left of the robot lands on another IK branch
    meta, arrays, _, _ = _compute(tmp_path / "b", monkeypatch, jump_rows=lambda T: T[:, 1, 3] > 0.05)
    st = arrays["s0.45_t0.25_k-0.3"]
    assert (st[ci(0.67, 0.16)] == AM.STATUS["branch_jump"]).all() and (st[ci(0.67, -0.16)] == AM.STATUS["feasible"]).all()


# ---------------------------------------------------------------------------------------------------------------------- identity, find, view


def _saved(tmp_path, monkeypatch, **kw):
    meta, arrays, lib, cfg = _compute(tmp_path, monkeypatch, **kw)
    root = tmp_path / "maps"
    return AM.save(str(root), meta, arrays), meta, lib, root


def _find(root, lib, **kw):
    args = dict(partner=("stockpot", "pot", None), rest_id="r0", object_key=lib.identity_key(), arm="right", hover=PARAMS_BOX, pre_back_m=0.10,
                policy_key=CP.DEFAULT.key("right"), robot=RM.robot_digest())
    args.update(kw)
    return AM.AttachMap.find(str(root / "lid__abc__s1__attachment_free"), args.pop("height", 0.7853), **args)


def test_find_enforces_every_identity_field(tmp_path, monkeypatch):
    p, meta, lib, root = _saved(tmp_path, monkeypatch)
    assert os.path.basename(p).startswith("amap__attach__stockpot__pot__h0.7853__") and _find(root, lib).path == p
    assert _find(root, lib, partner=("stockpot", "other", None)) is None and _find(root, lib, partner=("stockpot", "pot", [0.65] * 3)) is None
    assert _find(root, lib, hover=PARAMS_SWEPT) is None and _find(root, lib, hover=dict(PARAMS_BOX, clear_m=0.004)) is None
    assert _find(root, lib, pre_back_m=0.12) is None and _find(root, lib, rest_id="r1") is None and _find(root, lib, object_key="x") is None
    assert _find(root, lib, arm="left") is None and _find(root, lib, policy_key="p") is None and _find(root, lib, robot="r") is None
    assert _find(root, lib, height=0.80) is None and _find(root, lib, config_require={"preshape": "max"}) is None
    # a hand-edited identity does not serve; a tabletop lookup never reads an attach map
    m = json.loads(open(p).read())
    m["pre_attach_back_m"] = 0.12
    open(p, "w").write(json.dumps(m))
    assert _find(root, lib, pre_back_m=0.12) is None
    assert RM.ReachMap.find(str(root / "lid__abc__s1__attachment_free"), lib, 0.7853, rest_id="r0") is None


def test_partner_rest_identity_preserves_old_keys_and_rejects_malformed_quaternions():
    base=_spec(); identity=base.digest()
    base.data['partner']['rest_quat']=[0.,0.,0.,-1.]
    assert base.digest()==identity and base.partner_rest==[0.,0.,0.,1.]
    base.data['partner']['rest_quat']=[1.,0.,0.,0.]
    assert base.digest()!=identity
    changed=base.digest();base.data['partner']['rest_quat']=[-1.,0.,0.,0.]
    assert base.digest()==changed
    for invalid in ([0,0,0,0],[2,0,0,0],[0,0,1],[float('nan'),0,0,1]):
        with pytest.raises(ValueError):AM.canonical_partner_rest(invalid)


def test_rest_transform_matches_independently_rotated_geometry_and_map_checks(tmp_path,monkeypatch):
    rotated=_spec();rest=[math.sin(.4),0.,0.,math.cos(.4)]
    rotated.data['partner']['rest_quat']=rest
    transform=S.pose_mat([0.,0.,0.],rest)
    rotated.data['partner']['bottom_offset_m']=-min(AM._tf(transform,p)[:,2].min() for p in rotated.partner_pieces)
    baked=deepcopy(rotated);baked.data['partner'].pop('rest_quat')
    baked.partner_pieces=[AM._tf(transform,p) for p in rotated.partner_pieces]
    baked.data['partner']['geometry_digest']=AM.pieces_digest(baked.partner_pieces)
    baked.data['attachment']['T_partner_male']=S.mat_pose(transform @ rotated.T_partner_male)
    for key in ('insertion_axis','seat_axis'):
        baked.data[key]['local']=(transform[:3,:3] @ np.array(rotated.data[key]['local'])).tolist()
    yaws=[0.,.7,2.1];grasps=[{'name':'g','T':T_OE,'finger_q':[.012,.012]}]
    a=AM.hover_table(rotated,LID,H.load('right'),grasps,yaws,PARAMS_SWEPT)
    b=AM.hover_table(baked,LID,H.load('right'),grasps,yaws,PARAMS_SWEPT)
    for x,y in zip(a,b):
        for key in ('h_m','touchdown_m','depth_m','part_clearance_m'):
            assert x[0][key]==pytest.approx(y[0][key],abs=1e-6)
    np.testing.assert_allclose(AM.chain_offsets(rotated,a,yaws,.10),AM.chain_offsets(baked,b,yaws,.10),atol=1e-6)
    # The independently baked spec passes through the native six-decimal pose serialization.
    np.testing.assert_allclose(AM.seat_eef(rotated,grasps,yaws),AM.seat_eef(baked,grasps,yaws),atol=1e-6)
    # Full filters also exercise inverse rest in arm/partner collisions and the support footprint.
    _,ar,_,_=_compute(tmp_path/'rotated',monkeypatch,spec=rotated,sphere=(.0975,0.,-.05))
    _,br,_,_=_compute(tmp_path/'baked',monkeypatch,spec=baked,sphere=(.0975,0.,-.05))
    assert ar.keys()==br.keys()
    for key in ar:np.testing.assert_array_equal(ar[key],br[key])


def test_rest_is_required_by_cache_lookup_and_live_pair(tmp_path,monkeypatch):
    spec=_spec();spec.data['partner']['rest_quat']=[1.,0.,0.,0.]
    spec.data['partner']['bottom_offset_m']=.085
    path,meta,lib,root=_saved(tmp_path,monkeypatch,spec=spec)
    assert _find(root,lib) is None
    assert _find(root,lib,partner_rest_quat=[-1.,0.,0.,0.]).path==path
    assert AM.live_mismatch(meta,{}) is not None
    assert AM.live_mismatch(meta,{'partner_rest_quat':[1.,0.,0.,0.]}) is None
    original_key=meta['key'];meta['partner']['rest_quat']=[0.,0.,0.,1.]
    assert AM.map_key(meta)!=original_key
    with open(path,'w') as stream:json.dump(meta,stream)
    assert _find(root,lib) is None  # A relabeled old/new-rest cache may never serve.


def test_map_key_changes_with_the_rule_the_spec_and_the_pre_distance(tmp_path, monkeypatch):
    _, meta, _, _ = _saved(tmp_path, monkeypatch)
    k = meta["key"]
    for path, v in ((("hover", "rule"), "swept"), (("spec", "digest"), "x"), (("pre_attach_back_m",), 0.12), (("partner", "geometry_digest"), "g"),
                    (("male", "object_key"), "o"), (("grasp_set_digest",), "d"), (("height", "value"), 0.8)):
        m = json.loads(json.dumps(meta))
        d = m
        for p_ in path[:-1]:
            d = d[p_]
        d[path[-1]] = v
        assert AM.map_key(m) != k, path
    m = json.loads(json.dumps(meta))
    m["hover"]["table"] = []  # the table is a function of the identity, not part of it
    assert AM.map_key(m) == k
    assert _spec().digest() != _spec(seat_axis={"local": [0.0, 0.1, 0.99]}).digest()


def test_band_view_is_what_the_attachment_task_reads(tmp_path, monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

    p, meta, lib, root = _saved(tmp_path, monkeypatch)
    am = _find(root, lib)
    view = AM.as_bands(am)
    assert view.meta["map_kind"] == "attach" and view.meta["stand_offs"] == [0.35, 0.45, 0.55] and view.path is None
    edge = EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 1.6)
    st = Stance("manipulation", (0.55, 0.0, 0.0), 0.25, -0.3, 0.45, edge)
    cw = TT.band_cells_world(view, st)
    want = am.feasible_cells(0.45, 0.25, -0.3)
    assert len(cw) == len(want) > 0 and np.allclose(sorted(map(tuple, cw.numpy() - [0.55, 0.0])), sorted(map(tuple, want)), atol=1e-5)
    ok, cell = TT.in_band(view, st, cw[0])
    assert ok and TT.joint_placement(view, None, st, cw[0], cw[0])["work_ok"] and TT.cell_classes(view, 0.25, -0.3, cell, 0.45) is None
    assert len(TT.band_cells_world(view, Stance("m", (0.55, 0.0, 0.0), 0.5, -0.8, 0.45, edge))) == 0  # an ineligible trunk: no cell


def test_live_pair_mismatch():
    meta = {"spec": {"T_partner_male": S.mat_pose(T_PM), "insertion_axis": {"local": [0.0, 0.0, 1.0]}, "child_link": "M", "parent_link": "F"},
            "partner": {"bottom_offset_m": 0.0}}
    live = {"T_partner_male": (th.tensor([0.0, 0.0, 0.0585]), th.tensor([0.0, 0.0, 0.0, 1.0])), "insertion_axis": th.tensor([0.0, 0.0, 1.0]),
            "partner_bottom_offset_m": 0.001}
    assert AM.live_mismatch(meta, live) is None
    assert "T_partner_male" in AM.live_mismatch(meta, dict(live, T_partner_male=(th.tensor([0.0, 0.0, 0.061]), th.tensor([0.0, 0.0, 0.0, 1.0]))))
    assert "insertion axis" in AM.live_mismatch(meta, dict(live, insertion_axis=th.tensor([0.0, 0.05, 1.0])))
    assert "bottom offset" in AM.live_mismatch(meta, dict(live, partner_bottom_offset_m=0.004))
    assert "parent_link" in AM.live_mismatch(meta, dict(live, parent_link="G"))


# ---------------------------------------------------------------------------------------------------------------------- the sampler side


def test_sampler_view_hit_miss_and_mismatch(tmp_path, monkeypatch, caplog):
    from ispatialgym.sampling import grasp_library as GL

    p, meta, lib, root = _saved(tmp_path, monkeypatch)
    an = tmp_path / "anchors.json"
    an.write_text(json.dumps({"schema": "isg-height-anchors/1", "max_approx_m": 0.01, "anchors": [{"key": "0.7853", "height": 0.7853, "precision": 1e-4, "surfaces": ["d"]}]}))
    monkeypatch.setenv("ISPATIALGYM_REACH_MAPS", str(root))
    monkeypatch.setenv("ISPATIALGYM_REACH_MAP_ANCHORS", str(an))
    monkeypatch.setattr(GL, "v2_entry", lambda key: lib if key == "lid__abc" else None)
    monkeypatch.setattr(AM, "oracle_hover_params", lambda rule=None, cfg=None: dict(PARAMS_BOX))
    s = NS(world=NS(floor_z=0.0, scene_model="office"))
    top = NS(z=0.7880, name="desk_0")  # 2.7 mm above the anchor: within its range
    live = {"T_partner_male": (th.tensor([0.0, 0.0, 0.058]), th.tensor([0.0, 0.0, 0.0, 1.0])), "insertion_axis": th.tensor([0.0, 0.0, 1.0]), "partner_bottom_offset_m": 0.0}
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        view, req, why = AM.sampler_view(s, grasp_key="lid__abc", partner=("stockpot", "pot", None), top=top, live=live, pre_back_m=0.10)
    assert why is None and view.meta["map_path"] == p and view.meta["anchor_lookup"]["how"] == "approx" and "attach map hit" in caplog.text
    assert req["kind"] == "attach" and req["entry"] == "lid__abc__s1__attachment_free" and req["hover_rule"] == "box" and req["h"] == 0.788
    view, req, why = AM.sampler_view(s, grasp_key="lid__abc", partner=("stockpot", "pot", [0.65] * 3), top=top, live=live, pre_back_m=0.10)
    assert view is None and why == "no compatible attach map" and req["partner_scale"] == [0.65] * 3
    view, req, why = AM.sampler_view(s, grasp_key="lid__abc", partner=("stockpot", "pot", None), top=top, live=dict(live, partner_bottom_offset_m=0.01), pre_back_m=0.10)
    assert view is None and "bottom offset" in req["mismatch"]
    view, req, why = AM.sampler_view(s, grasp_key="lid__zzz", partner=("stockpot", "pot", None), top=top, live=live, pre_back_m=0.10)
    assert view is None and "no v2 entry" in why


def _attachment(monkeypatch, view):
    from ispatialgym.sampling.v2.tasks import attachment as AT

    s = object.__new__(AT.Attachment)
    s.t_role, s.p_role = "target", "partner"
    s.choice = NS(models={"partner": NS(category="stockpot", model="pot", scale=None)})
    s.grasp_key, s.top, s.template_facts = "lid__abc", NS(z=0.7853, name="desk_0"), {}
    s.parent_link_in_female = (th.tensor([0.0, 0.0, 0.05]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    s.root_in_child_link = (th.tensor([0.0, 0.0, 0.008]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    s.insertion_axis_local = th.tensor([0.0, 0.0, 1.0])
    s.hull = {"partner": th.tensor(np.concatenate(_pot()), dtype=th.float32)}
    s.rest = {"partner": C.IDENTITY_QUAT.clone()}
    s.world = NS(bottom_offset=lambda hull, quat: -float(hull[:, 2].min()), floor_z=0.0, scene_model="office")
    s.defaults = {"pre_attach_back_m": 0.1}
    s.gen = lambda key, default=None: default
    calls = []

    def sampler_view(s_, **kw):
        calls.append(kw)
        return view, {"kind": "attach", "entry": "lid__abc__s1__attachment_free", "h": 0.7853}, (None if view is not None else "no compatible attach map")

    monkeypatch.setattr(AM, "sampler_view", sampler_view)
    return s, calls


def test_attachment_consumer_reads_the_map_and_rejects_missing_map(monkeypatch, caplog):
    from ispatialgym.sampling.sim_utils import Reject

    s, calls = _attachment(monkeypatch, None)
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        with pytest.raises(Reject) as e:
            s._attach_map_view()
    assert e.value.reason == "missing_map" and "stockpot/pot" in str(e.value)
    assert "MAP_REQUEST" in caplog.text and '"kind": "attach"' in caplog.text and s.template_facts["attach_map"]["hit"] is False
    kw = calls[0]
    assert kw["partner"] == ("stockpot", "pot", None) and kw["pre_back_m"] == 0.1 and kw["arm"] == "right"
    T = kw["live"]["T_partner_male"]
    assert float(T[0][2]) == pytest.approx(0.058) and kw["live"]["partner_bottom_offset_m"] == pytest.approx(0.0)
    view = NS(meta={"map_path": "/m.json"})
    s, _ = _attachment(monkeypatch, view)
    assert s._attach_map_view() is view and s.template_facts["attach_map"] == {"hit": True, "why": None, "path": "/m.json"}


def test_attach_band_off_v2_only_is_the_legacy_band(monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    s, _ = _attachment(monkeypatch, None)
    s.world = NS(scene_model="office")
    spec = TT.BandSpec(object="stockpot/pot", grasp_key="attach:lid__abc", placed=["attach:lid/abc"], scale=None)
    loads, computes = [], []

    def load(*a, **kw):
        loads.append((a, kw))
        raise FileNotFoundError("miss")

    monkeypatch.setattr(TT.RB.ReachBands, "load", staticmethod(load))
    monkeypatch.setattr(TT, "compute_bands", lambda s_, sp, top, edge: computes.append(sp) or "COMPUTED")
    assert s._attach_band(spec, None) == "COMPUTED" and loads[0][0] == ("office", "desk_0", "stockpot/pot", "attach:lid__abc")
    assert loads[0][1] == {"placed": ["attach:lid/abc"], "scale": None} and computes == [spec]
    view = object()
    assert s._attach_band(spec, None, view) is view and len(loads) == 1  # the map view: no legacy load


# ---------------------------------------------------------------------------------------------------------------------- the template's rule, moved unchanged


def _old_rule(parent_link_in_female, hull, min_off=0.02, max_dev=45.0):
    """The template's inline code before i7-attachmap (attachment.py sample_template, of-v2maps a4620a1e), verbatim but for the names."""
    from ispatialgym.sampling.v2 import tabletop as TT

    axis_geo = TT._t(parent_link_in_female[0]) - TT._t(hull).mean(dim=0)
    axis_z = C.unit(C.quat_rotate(parent_link_in_female[1], th.tensor([0.0, 0.0, 1.0])))
    facts = {"source": "meta link +z", "axis": [round(float(v), 4) for v in axis_z]}
    out = axis_z
    if float(th.linalg.norm(axis_geo)) > float(min_off):
        geo = C.unit(axis_geo)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, float(th.dot(geo, axis_z))))))
        facts.update({"centroid_to_female": [round(float(v), 4) for v in geo], "angle_deg": round(angle, 1)})
        if angle > float(max_dev):
            out = geo
            facts["source"] = "centroid->female"
            facts["axis"] = facts["centroid_to_female"]
    return out, facts


def test_insertion_axis_rule_is_the_templates_rule():
    from ispatialgym.sampling.v2.tasks import attachment as AT

    rng = np.random.default_rng(3)
    cork = (th.tensor([0.0, 0.0, 0.2]), th.tensor([0.5, -0.5, 0.5, 0.5]))  # cork / wine bottle: the link's +z along the bottle's -y
    hull = th.tensor(rng.normal(scale=0.03, size=(40, 3)), dtype=th.float32)
    a, f = AT.insertion_axis_rule(cork, hull)
    assert f["source"] == "centroid->female" and float(a[2]) > 0.95
    lid = (th.tensor([0.0, 0.0, 0.1]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    a, f = AT.insertion_axis_rule(lid, hull)
    assert f["source"] == "meta link +z" and a.tolist() == pytest.approx([0.0, 0.0, 1.0])
    for _ in range(50):
        q = th.tensor(rng.normal(size=4), dtype=th.float32)
        pl = (th.tensor(rng.normal(scale=0.05, size=3), dtype=th.float32), q / th.linalg.norm(q))
        hh = th.tensor(rng.normal(scale=0.04, size=(30, 3)), dtype=th.float32)
        a, f = AT.insertion_axis_rule(pl, hh)
        b, g = _old_rule(pl, hh)
        assert th.equal(a, b) and f == g
