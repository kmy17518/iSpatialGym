"""i7-ul02guide (owner 2026-11-10, under_large_02): the desk's leg panel as a guide wall for the standing briefcase's floor push -
``ISPATIALGYM_PUSH_GUIDE_WALL`` (default off): the wall scan's reduction, the lanes beside a wall, the goal-yaw snap, the switch's scope, the
``"endwise"`` yaw lock, the push record and the oracle's lean / aim in the strokes (no simulator)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.tests import test_iors_large_push_pitched as PP
from ispatialgym.tests import test_push_under_strokes as PS

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")

UL02 = IL.PUSH_GUIDE_WALL_VARIANTS[0]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(IL.PUSH_GUIDE_WALL_ENV, raising=False)
    monkeypatch.delenv(IL.FLOORPUSH_BROADSIDE_ENV, raising=False)


# ---------------------------------------------------------------------------------------------------------------- the scan's reduction


def _rows(faces, n_cells=9, taus0=None, slant=None):
    """Synthetic scan rows: walls at ``faces`` = [(tau, free side)], rays from every 5 cm both ways along a 2.8 m edge; a ray stops at the
    first face it meets (its own direction's), ``slant`` (cell -> extra tau) moves a face with the cell."""
    taus0 = taus0 if taus0 is not None else [-1.4 + 0.05 * k for k in range(57)]
    rows = []
    for c in range(n_cells):
        for t0 in taus0:
            for d in (1, -1):
                hit = None
                for tau, free in sorted(faces, key=lambda f: d * f[0]):
                    tau_c = tau + (slant or {}).get(c, 0.0)
                    if (tau_c - t0) * d > 0 and free == -d:
                        hit = tau_c
                        break
                rows.append((c, t0, d, hit))
    return rows


def test_faces_need_every_cell_a_free_corridor_and_a_flat_face():
    # two panels' outer faces (desk_mdhelw_0's +y edge: 15.265 / 16.936 as -0.839 / +0.832 from the edge's middle)
    rows = _rows([(-0.839, -1), (0.832, 1)])
    faces = IL.guide_faces_from_hits(rows, 9, 0.145)
    assert [(f["tau"], f["free"]) for f in faces] == [(-0.839, -1), (0.832, 1)] and all(f["cells"] == 9 for f in faces)
    # a cell without the face (the wall ends short of the target's front, a bottom-hung panel misses the low rays): no wall
    faces = IL.guide_faces_from_hits([r for r in rows if not (r[0] == 4 and r[3] == -0.839)], 9, 0.145)
    assert [f["tau"] for f in faces] == [0.832]
    # a slanted face (the right panel's inner face: 6 cm over the target's length) is no guide; a 5 mm one is, at its most protruding point
    slanted = _rows([(0.6, -1)], slant={c: 0.01 * (c // 3) * 3 for c in range(9)})
    assert IL.guide_faces_from_hits(slanted, 9, 0.145) == []
    mild = _rows([(0.6, -1)], slant={c: 0.0025 * (c // 3) for c in range(9)})
    f = IL.guide_faces_from_hits(mild, 9, 0.145)
    assert len(f) == 1 and f[0]["tau"] == pytest.approx(0.6) and f[0]["spread_m"] == pytest.approx(0.005, abs=1e-4)
    # rays that start within the corridor (or inside the panel: distance 0) do not count
    assert IL.guide_faces_from_hits([(c, 0.80, 1, 0.832) for c in range(9)], 9, 0.145) == []


def test_lanes_beside_the_faces():
    faces = [{"tau": -0.839, "free": -1}, {"tau": 0.832, "free": 1}, {"tau": 1.30, "free": 1}]
    lanes = IL.guide_lanes(faces, 0.06, 0.015, 1.345)
    # the target's centre line 7.5 cm off each face on its free side; the wall on its +tangent side for a -tangent free side; the third out of range
    assert [ln["lat"] for ln in lanes] == [pytest.approx(-0.914), pytest.approx(0.907)] and [ln["wall_left"] for ln in lanes] == [1, -1]


def test_goal_yaw_snap_puts_the_long_axis_on_the_partner_axes():
    q = math.pi / 2
    for y0, d_yaw, phi0, pyaw in ((0.1, 0.0, 0.0, 0.0), (0.3, -2.336, 0.0, -q), (1.2, 0.4, q, 0.05)):
        y = IL.guide_snap_yaw(y0, d_yaw, phi0, pyaw, math.radians(45.0))
        r = (y + d_yaw + phi0 - pyaw) % q
        assert min(r, q - r) < 1e-6 and abs(y - y0) <= math.radians(45.0) + 1e-9
    assert IL.guide_snap_yaw(math.radians(20.0), 0.0, 0.0, 0.0, math.radians(11.25)) is None
    assert IL.guide_snap_yaw(math.radians(10.0), 0.0, 0.0, 0.0, math.radians(11.25)) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------------------------------------------- the sampler


def _s(variant=UL02, hull=None, partner_yaw=-math.pi / 2, **gen):
    s = PP._sampler(push_under_hand="pitched_far", hull=hull if hull is not None else PP._box(0.33, 0.12, 0.27), **gen)
    s.gp.variant = variant
    s.kit = SimpleNamespace(floor_z=0.0, stats={})
    partner = SimpleNamespace(name="desk_mdhelw_0", get_position_orientation=lambda: (th.zeros(3), G.quat_from_yaw(partner_yaw)),
                              links={"base_link": SimpleNamespace(prim_path="/World/desk/base_link")})
    objs = {"floor": SimpleNamespace(name="floors_jluwuz_0"), "target": SimpleNamespace(name="briefcase.n.01_1"), "partner": partner}
    s.obj = lambda role: objs[role]
    return s


def test_switch_scope(monkeypatch):
    assert not _s().guide_wall_active() and _s().push_yaw_lock() is None and _s().init_yaw_jitter() == IL.TARGET_JITTER
    assert _s().end_aligned_init_yaw(0.3, 0.0) is None
    monkeypatch.setenv(IL.PUSH_GUIDE_WALL_ENV, "1")
    assert _s().guide_wall_active() and _s().push_yaw_lock() == "endwise" and _s().init_yaw_jitter() == 0.0
    # under_large_02 only: another variant, the blade hand, the ontop relation and a round footprint keep today's push
    assert not _s(variant=PP.VARIANT).guide_wall_active()
    blade = _s()
    blade.gp.generation["push_under_hand"] = "blade"
    assert not blade.guide_wall_active()
    on = _s()
    on.relation = "ontop"
    assert not on.guide_wall_active()
    assert not _s(hull=PP._box(0.2, 0.2, 0.27)).guide_wall_active()
    fp = _s().guide_footprint()
    assert fp[1] == pytest.approx(2.75, abs=1e-3) and fp[2] == pytest.approx(0.165, abs=1e-4) and fp[3] == pytest.approx(0.06, abs=1e-4)


def test_endwise_lock_and_the_goal_yaw_on_the_wall(monkeypatch):
    monkeypatch.setenv(IL.PUSH_GUIDE_WALL_ENV, "1")
    s = _s()
    q = lambda deg: G.qmul(G.quat_from_yaw(math.radians(deg)), s.rest)  # noqa: E731
    d = (0.0, -1.0)  # the +y edge's push (the long axis at the rest along x)
    assert s.push_yaw_kept(q(90.0), d) and s.push_yaw_kept(q(-90.0), d) and s.push_yaw_kept(q(100.0), d)
    assert not s.push_yaw_kept(q(0.0), d) and not s.push_yaw_kept(q(67.5), d) and not s.push_yaw_kept(q(45.0), d)
    # the carried yaw snapped onto the desk's axes (the desk at -90 deg): a screened yaw 8 deg off moves onto the axis, 20 deg off has none
    y = s.end_aligned_init_yaw(math.radians(98.0), 0.0)
    assert y == pytest.approx(math.radians(90.0), abs=1e-6)
    assert s.end_aligned_init_yaw(math.radians(70.0), 0.0) is False
    assert s.end_aligned_init_yaw(math.radians(80.0), math.radians(18.0)) == pytest.approx(math.radians(72.0), abs=1e-6)


def _panel_world(boxes):
    """A raycast over axis-aligned boxes ``[(lo xyz, hi xyz)]`` of the partner (slab test, the nearest entry; a start inside = distance 0)."""

    def raycast(starts, ends):
        out = []
        for p0, p1 in zip(starts.tolist(), ends.tolist()):
            best = None
            for lo, hi in boxes:
                t0, t1 = 0.0, 1.0
                for k in range(3):
                    dk = p1[k] - p0[k]
                    if abs(dk) < 1e-12:
                        if not (lo[k] <= p0[k] <= hi[k]):
                            t0, t1 = 1.0, 0.0
                        continue
                    a, b = (lo[k] - p0[k]) / dk, (hi[k] - p0[k]) / dk
                    t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
                if t0 <= t1 and (best is None or t0 < best):
                    best = t0
            if best is None:
                out.append({"hit": False})
            else:
                out.append({"hit": True, "rigidBody": "/World/desk/base_link", "position": [p0[k] + best * (p1[k] - p0[k]) for k in range(3)]})
        return out

    return SimpleNamespace(raycast=raycast)


#: desk_mdhelw_0 as the 2026-11-10 probe measured it: the side panels x 15.265-15.483 / 16.70-16.936, y -13.165..-12.279, floor to 0.62;
#: the modesty panel x 15.485-16.674, y -12.547..-12.450 from 0.119 up
DESK = [((15.265, -13.165, 0.0), (15.483, -12.279, 0.62)), ((16.70, -13.165, 0.0), (16.936, -12.279, 0.62)),
        ((15.485, -12.547, 0.119), (16.674, -12.450, 0.62))]


def test_guide_walls_on_the_desk_panels(monkeypatch):
    monkeypatch.setenv(IL.PUSH_GUIDE_WALL_ENV, "1")
    s = _s()
    s.world = _panel_world(DESK)
    back = EdgeFrame("+y", th.tensor([16.1045, -12.2592]), th.tensor([0.0, -1.0]), 2.8113)
    front = EdgeFrame("-y", th.tensor([16.1045, -13.1898]), th.tensor([0.0, 1.0]), 2.8113)
    wb = s.guide_walls(back)
    # from the back: the two outer faces (the knee space is closed by the modesty panel at the target's front)
    assert [(round(16.1045 + f["tau"], 3), f["free"]) for f in wb] == [(15.265, -1), (16.936, 1)]
    wf = s.guide_walls(front)  # the tangent of the front edge points -x
    assert sorted((round(16.1045 - f["tau"], 3), -f["free"]) for f in wf) == [(15.265, -1), (15.483, 1), (16.70, -1), (16.936, 1)]
    assert s.kit.stats["push_guide_wall.scan"] == 2 and s.guide_walls(back) is wb  # memoised per edge
    lanes = IL.guide_lanes(wb, 0.06, IL.PUSH_GUIDE_GAP_M, back.length / 2 - 0.06)
    assert [round(16.1045 + ln["lat"], 3) for ln in lanes] == [15.19, 17.011]


def test_guide_goal_clear_reads_the_clearance_raster():
    s = _s()
    # a 5 cm raster: free 0.64 except under the modesty panel (x 15.49-16.67, y -12.55..-12.45: 0.12)
    xs, ys = [14.7488 + (i + 0.5) * 0.05 for i in range(55)], [-13.1397 + (j + 0.5) * 0.05 for j in range(17)]
    s.place["clearance"] = {"resolution": 0.05, "origin": [14.7488, -13.1397],
                            "grid": [[0.12 if (15.485 <= x <= 16.674 and -12.55 <= y <= -12.45) else 0.64 for y in ys] for x in xs]}
    d2, t = th.tensor([0.0, -1.0]), th.tensor([1.0, 0.0])
    assert s.guide_goal_clear((17.011, -12.524), d2, t, 0.30)  # the right outer lane: beside the modesty panel's rows, not on them
    assert not s.guide_goal_clear((15.558, -12.524), d2, t, 0.30)  # the knee space from the back: its front over the modesty panel


def test_the_push_record_carries_the_lane_and_the_oracle_knobs():
    s = _s()
    st = Stance("placement", (17.0, -11.5, -math.pi / 2), 2.0, 0.0, 0.8, None, None, 0.9065, None, ["left"])
    assert s.guide_wall_record(st) == {}
    st.extras["guide_wall"] = {"edge": "+y", "lane_m": 0.9065, "wall_left": -1, "gap_m": 0.015}
    rec = s.guide_wall_record(st)["guide_wall"]
    assert rec["wall_left"] == -1 and rec["lean_m_per_m"] == IL.PUSH_GUIDE_LEAN_M_PER_M and rec["aim_away_m"] == IL.PUSH_GUIDE_AIM_AWAY_M
    assert s.kit.stats["push_guide_wall.recorded"] == 1
    plan = EX.push_guide_wall_plan({"guide_wall": rec})
    assert plan == {"side": -1.0, "lean_m_per_m": IL.PUSH_GUIDE_LEAN_M_PER_M, "aim_away_m": IL.PUSH_GUIDE_AIM_AWAY_M}
    assert EX.push_guide_wall_plan({}) is None and EX.push_guide_wall_plan({"guide_wall": {"wall_left": 0}}) is None


# ---------------------------------------------------------------------------------------------------------------- the oracle


def test_strokes_lean_into_the_wall_and_aim_away_from_it():
    env, robot, sandal, goal, _ = PS.sandal_world(yaw_deg=0.0, off_m=0.0)
    gw = {"wall_left": 1, "lean_m_per_m": 0.10, "aim_away_m": 0.01}
    ctx, motions = PS.run_push(env, robot, PS.push_action({"strokes": PS.plan(), "guide_wall": gw}))
    rep = ctx["push_strokes"]
    assert rep["guide_wall"] == {"side": 1.0, "lean_m_per_m": 0.10, "aim_away_m": 0.01}
    for s in rep["strokes"]:
        assert s["guide_aim_m"] == pytest.approx(-0.01) and s["target_m"] == pytest.approx(s["root_off_line_m"] - 0.01, abs=1e-4)
        assert s["guide_lean_m"] == pytest.approx(0.10 * (s["end_along_m"] - s["start_along_m"]), abs=1e-4) and s["guide_lean_m"] > 0.0
    # without the key: no guide fields, today's strokes
    env, robot, sandal, goal, _ = PS.sandal_world(yaw_deg=0.0, off_m=0.0)
    ctx, _ = PS.run_push(env, robot, PS.push_action({"strokes": PS.plan()}))
    assert "guide_wall" not in ctx["push_strokes"] and all("guide_lean_m" not in s for s in ctx["push_strokes"]["strokes"])
