"""i7-t03both 2026-11-09 (owner decision A5): ``ISPATIALGYM_SHELF_BOARD_DRAW`` - each attempt of touching_03's shelf layout draws the
compartment AND the board among every layout whose board has a v2 lean rest, then that board's lean rest (its angle); the rest's grasps,
rest, shelf template and band follow the draw, the arm on the target's side works. Default off (not a shelf tip family default); eval keeps
the board opposite ``generation.partner_side``; without start_per_attempt the template's board stays. ``tabletop.shelf_lean_pool`` lists the
library's lean keys by board. Bare ``IORSSmall`` stand-ins from test_shelf_start_draw; no simulator."""

from __future__ import annotations

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.tests.test_shelf_start_draw import _draw, _iors, _sampler

#: every compartment of the stand-in kmuacd row (three cubes: +0.37 / 0 / -0.37) x both boards
ALL6 = [[0.37, "left"], [0.37, "right"], [0.0, "left"], [0.0, "right"], [-0.37, "left"], [-0.37, "right"]]
KL20, KL11, KR15 = "hb__x__bu@lean_left_20_aaaaaa", "hb__x__bu@lean_left_11_bbbbbb", "hb__x__bu@lean_right_15_cccccc"


Z0, Z1 = 0.8354, 0.4486


def _state(key, side, deg, rest, rows=(Z0,)):
    return {"key": key, "grasps": [f"g-{key}"], "grasps_flush": [f"g-{key}"], "rest": rest.clone(), "finger_open": 0.03 if side > 0 else 0.04,
            "template": {"key": key, "lean_side": side, "lean_deg": deg, "context": "v2 lean rest"}, "bands": {}, "rows": list(rows)}


def _board_sampler(monkeypatch, gen=None, spa=True, layouts=ALL6, rows=None):
    """The stand-in with the board draw's template state: three lean rests (left 20 = the template's, left 11, right 15) at the template's
    row; ``rows`` {z: [layouts]} adds rows (the left 20 and right 15 rests have their maps there, left 11 not) with a raster at that z."""
    from ispatialgym.sampling.sim_utils import SupportTop
    from ispatialgym.sampling.v2 import support_split as SS
    from ispatialgym.sampling.v2 import tabletop as TT

    monkeypatch.setattr(TT, "lean_rest_side", lambda k: None if not k or "@" not in k else (1.0 if "@lean_left" in k else -1.0))
    monkeypatch.setattr(SS, "register_top", lambda s_, obj, top, plane=None, what="top": s_.__dict__.setdefault("_registered", []).append(round(float(top.z), 4)))
    g = dict(gen or {})
    if rows:
        g["shelf_start_draw"] = {"layouts": {"bookcase_x_0": layouts}, "lean_deg": [11.0, 20.0], "rows": {"bookcase_x_0": {str(z): v for z, v in rows.items()}}}
    s = _sampler(monkeypatch, gen=g, spa=spa, layouts=layouts)
    rl = s.rest["target"].clone()
    rr = TT.releaned_quat(rl, 1.0, 15.0, -1.0, 15.0)
    extra = tuple(rows or ())
    own = _state(KL20, 1.0, 20.0, rl, (Z0,) + extra)
    pool = {1.0: [own, _state(KL11, 1.0, 11.0, rl)], -1.0: [_state(KR15, -1.0, 15.0, rr, (Z0,) + extra)]}
    s.grasp_key, s.grasps, s.grasps_flush = KL20, list(own["grasps"]), list(own["grasps_flush"])
    s._shelf_template = dict(own["template"])
    s._board_keys, s._board_current = pool, own
    s._shelf_row_z0 = s._shelf_row_z = Z0
    s._shelf_rows = {Z0: {"top": s.top, "surface_z": Z0, "spa_edges": {}, "facts": {}}}
    for z in extra:
        t = s.top
        s._shelf_rows[z] = {"top": SupportTop("bookcase_x_0", t.xs, t.ys, t.mask.clone(), t.occupied.clone(), float(z), 0.0), "surface_z": z, "spa_edges": {},
                            "facts": {}}
    s.kit = type("K", (), {"stats": {}})()
    return s, pool


def test_the_switch_is_off_by_default_even_for_the_shelf_tip_family(monkeypatch):
    from ispatialgym.sampling.v2 import alias_switches as AS

    IS = _iors()
    monkeypatch.delenv(IS.SHELF_BOARD_DRAW_ENV, raising=False)
    tip = type("T", (), {"shelf_tip": lambda self: True})()
    assert IS.SHELF_BOARD_DRAW_ENV not in AS.SHELF_TIP_DEFAULTS
    assert IS.shelf_board_draw_on(tip) is False and IS.shelf_board_draw_on() is False
    monkeypatch.setenv(IS.SHELF_BOARD_DRAW_ENV, "1")
    assert IS.shelf_board_draw_on(tip) is True
    monkeypatch.setenv(IS.SHELF_BOARD_DRAW_ENV, "0")
    assert IS.shelf_board_draw_on(tip) is False


def test_each_attempt_draws_compartment_board_and_that_boards_rest(monkeypatch):
    s, pool = _board_sampler(monkeypatch)
    seen, keys = set(), set()
    for seed in range(120):
        f = _draw(s, seed)
        side = 1.0 if f["target_board"] == "left" else -1.0
        seen.add((f["bay_m"], f["target_board"]))
        bd = f["board_draw"]
        keys.add(bd["grasp_key"])
        assert bd["grasp_key"] in [st["key"] for st in pool[side]] and s.grasp_key == bd["grasp_key"]
        st = next(x for x in pool[side] if x["key"] == s.grasp_key)
        # the drawn rest is the sampler's: grasps, shelf template, the target lean = the rest's own angle, the pre-shape opening
        assert s.grasps == st["grasps"] and s.grasps_flush == st["grasps_flush"] and s._shelf_template["key"] == st["key"]
        assert s.shelf_start["lean_deg"] == st["template"]["lean_deg"] == f["target_lean_deg"] and s._shelf_finger_open == st["finger_open"]
        assert th.allclose(s._shelf_rest0["target"], st["rest"]) and th.allclose(s.rest["target"], st["rest"])
        arm = "left" if side > 0 else "right"
        assert s.segment_arms == [arm] and s.shelf_start["lean_side"] == side and s.arm_bands == {arm: f"band-{arm}"}
        assert "shelf board draw" in f["restricted"]
        g = _draw(s, seed)  # deterministic per seed
        assert g == f
    # the stand-in row's left cube is +0.37: all six compartments x boards are drawn, every rest of both boards
    assert seen == {tuple(lay) for lay in ALL6}
    assert keys == {KL20, KL11, KR15}
    stats = s.kit.stats
    assert stats["shelf_board_draw.drawn.left"] > 0 and stats["shelf_board_draw.drawn.right"] > 0
    assert any(k.startswith("shelf_board_draw.layout.0.8354_+0.37_") for k in stats) and stats["shelf_board_draw.rest.lean_right_15_cccccc"] > 0


def test_the_band_of_each_rest_is_kept_with_its_state_per_row(monkeypatch):
    s, pool = _board_sampler(monkeypatch, rows={Z1: [[0.0, "left"]]})
    s.bands = "band-of-L20"
    s.apply_board_key(pool[-1.0][0])
    assert s.bands is None and pool[1.0][0]["bands"] == {Z0: "band-of-L20"}  # left behind with its rest; the new rest's is read next
    s.bands = "band-of-R15"
    s.apply_board_key(pool[1.0][0])
    assert s.bands == "band-of-L20" and pool[-1.0][0]["bands"] == {Z0: "band-of-R15"}
    s.apply_board_key(pool[1.0][0])  # the same rest again: its band stays
    assert s.bands == "band-of-L20"
    s.apply_board_key(pool[1.0][0], Z1)  # another row: that row's band (none yet), the row's surface
    assert s.bands is None and s.top.z == pytest.approx(Z1) and s._shelf_row_z == Z1 and s._registered[-1] == Z1
    s.bands = "band-of-L20-at-Z1"
    s.apply_board_key(pool[1.0][0], Z0)
    assert s.bands == "band-of-L20" and pool[1.0][0]["bands"][Z1] == "band-of-L20-at-Z1" and s.top.z == pytest.approx(Z0)
    assert s.kit.stats["shelf_board_draw.row_switch"] == 2


def test_eval_keeps_the_board_opposite_the_partner_side_and_draws_its_compartments(monkeypatch):
    s, pool = _board_sampler(monkeypatch, gen={"partner_side": "right"})
    seen = set()
    for seed in range(40):
        f = _draw(s, seed)
        assert f["target_board"] == "left" and s.grasp_key in (KL20, KL11) and s.segment_arms == ["left"]
        seen.add(f["bay_m"])
    assert seen == {0.37, 0.0, -0.37}
    # a partner side whose board has no lean rest: rejected as the template's lean key would be
    s, pool = _board_sampler(monkeypatch, gen={"partner_side": "right"})
    s._board_keys = {-1.0: pool[-1.0]}
    with pytest.raises(Reject) as e:
        s.draw_shelf_side(np.random.default_rng(0))
    assert e.value.reason == "missing_grasps:lean"


def test_without_start_per_attempt_the_templates_rest_keeps_its_board(monkeypatch):
    s, _ = _board_sampler(monkeypatch, spa=False)
    for seed in range(10):
        f = _draw(s, seed)
        assert f["target_board"] == "left" and s.grasp_key == KL20 and "board_draw" not in f
    assert s.kit.stats["shelf_board_draw.off_no_start_per_attempt"] == 10


def test_off_the_rest_pins_the_board_as_today(monkeypatch):
    s, _ = _board_sampler(monkeypatch)
    s._board_keys = None
    for seed in range(20):
        f = _draw(s, seed)
        assert f["target_board"] == "left" and s.grasp_key == KL20 and "board_draw" not in f and "v2 lean rest" in f["restricted"]
    assert not any(k.startswith("shelf_board_draw") for k in s.kit.stats)


def test_shelf_lean_pool_lists_the_librarys_lean_keys_by_board(tmp_path, monkeypatch):
    from ispatialgym.grasping import lean as LN
    from ispatialgym.sampling import grasp_library as GL
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.tests.test_interior_consumer import _sampler as _ics
    from ispatialgym.tests.test_interior_consumer import _write_lean_entry
    from ispatialgym.tests.test_interior_maps import HB_KEY, LEAN

    glib, maps = tmp_path / "glib", tmp_path / "maps"
    monkeypatch.setenv("ISPATIALGYM_GRASP_LIBRARY_V2", str(glib))
    monkeypatch.setenv("ISPATIALGYM_REACH_MAPS", str(maps))
    GL._v2_index.cache_clear()
    GL._v2_lean_index.cache_clear()
    try:
        r15 = LN.LeanSpec(15.0, "right")
        l20 = LN.LeanSpec(20.0, "left")
        for spec in (LEAN, l20, r15):
            _write_lean_entry(str(glib), spec)
        gen = {"shelf_start_draw": {"lean_deg": [11.0, 20.0]}}
        pool = TT.shelf_lean_pool(_ics(gen), HB_KEY)
        assert pool == {"left": [LN.lean_key(HB_KEY, LEAN), LN.lean_key(HB_KEY, l20)], "right": [LN.lean_key(HB_KEY, r15)]}
        assert TT.shelf_lean_pool(_ics({"shelf_start_draw": {"lean_deg": [14.0, 25.0]}}), HB_KEY) == {"left": [LN.lean_key(HB_KEY, l20)],
                                                                                                   "right": [LN.lean_key(HB_KEY, r15)]}
        assert TT.shelf_lean_pool(_ics(gen), "other__key") == {}
        monkeypatch.delenv("ISPATIALGYM_REACH_MAPS")
        assert TT.shelf_lean_pool(_ics(gen), HB_KEY) == {}
    finally:
        GL._v2_index.cache_clear()
        GL._v2_lean_index.cache_clear()


def test_rows_draw_the_compartments_of_other_board_heights(monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    rows = {Z1: [[0.0, "left"], [0.0, "right"], [-0.37, "left"], [-0.37, "right"]]}
    s, pool = _board_sampler(monkeypatch, rows=rows)
    seen = set()
    for seed in range(160):
        f = _draw(s, seed)
        z = f["board_draw"]["row_z"]
        seen.add((z, f["bay_m"], f["target_board"]))
        assert s.top.z == pytest.approx(z) and s._shelf_row_z == z
        # the books stand on the drawn row's board; the rest has that row's map (left 11 has none at Z1)
        for w in (s.world.hull_at(s.hull["target"], s.shelf_pose), s.world.hull_at(s.hull["partner"], s.shelf_partner_pose)):
            assert float(w[:, 2].min()) == pytest.approx(z + TT.Z_OFFSET, abs=1e-4)
        assert z in next(st for st in pool[1.0 if f["target_board"] == "left" else -1.0] if st["key"] == s.grasp_key)["rows"]
        assert all(len(lay) == 2 for lay in s._shelf_side_layouts)
        if z == Z1:
            assert s.grasp_key != KL11
    assert {(z, b, sd) for z, b, sd in seen if z == Z1} == {(Z1, b, sd) for b, sd in rows[Z1]}
    assert {(b, sd) for z, b, sd in seen if z == Z0} == {tuple(lay) for lay in ALL6}
    monkeypatch.setenv(_iors().SHELF_BOARD_DRAW_ENV, "1")
    assert s.shelf_draw_facts()["rows"] == {str(Z1): rows[Z1]}
    monkeypatch.delenv(_iors().SHELF_BOARD_DRAW_ENV)
    assert "rows" not in s.shelf_draw_facts()  # off: the template's facts as before


def test_a_row_at_the_templates_board_adds_compartments_there(monkeypatch):
    # the customs list the template row's extra compartments under rows (today's layouts stay as they are with the switch off)
    s, pool = _board_sampler(monkeypatch, layouts=[[-0.37, "left"]], rows={Z0 + 0.0004: [[0.0, "left"], [0.37, "right"]]})
    seen = set()
    for seed in range(60):
        f = _draw(s, seed)
        assert f["board_draw"]["row_z"] == Z0 and s.top.z == pytest.approx(Z0)
        seen.add((f["bay_m"], f["target_board"]))
    assert seen == {(-0.37, "left"), (0.0, "left"), (0.37, "right")}
    assert "shelf_board_draw.row_switch" not in s.kit.stats


def test_rows_are_checked_and_ignored_without_the_board_draw(monkeypatch):
    s, _ = _board_sampler(monkeypatch, rows={Z1: [[0.0, "left"]]})
    assert s.shelf_start_draw()["rows"] == [(Z1, [(0.0, 1.0)])]
    s._board_keys = None
    for seed in range(10):
        f = _draw(s, seed)
        assert s.top.z == pytest.approx(Z0) and "board_draw" not in f
    for bad in ({"bookcase_x_0": [[0.0, "left"]]}, {"bookcase_x_0": {"high": [[0.0, "left"]]}}, {"bookcase_x_0": {"0.45": [[0.0, "up"]]}}):
        gen = {"shelf_start_draw": {"layouts": {"bookcase_x_0": ALL6}, "rows": bad}}
        with pytest.raises(Reject) as e:
            _sampler(monkeypatch, gen=gen)  # its template capture reads the spec
        assert e.value.reason == "spec"
