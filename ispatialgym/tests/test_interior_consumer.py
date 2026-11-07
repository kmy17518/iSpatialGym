"""i7-lean 2026-11-07 (owner decision; prompt I item 7): touching_03's consumer of the lean rests and the interior maps - the template's
lean key (``tabletop.shelf_lean_key``: the task's side, one library angle per template), the lean key's shelf facts, ``interior_map_bands``
(a hit polices the board, any miss rejects ``missing_map`` with one MAP_REQUEST), the v2 dispatch (``ensure_bands`` / ``v2_map_view`` /
``map_view_arm_band``: a lean rest never takes a legacy band) and ``IORSSmall.shelf_rests`` (the library rest is never re-leaned).
Simulator-free."""

import json
import logging
import math
import os
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.grasping import lean as LN
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling import interiors as IN
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.tests.test_interior_maps import HB_KEY, KMUACD, LEAN, SCENE, SUPPORT, _lean_lib, _meta, _write


def _write_lean_entry(root, spec=LEAN, face=0.06):
    lib = _lean_lib(face, spec)
    eid = S.entry_id("hardback", "acbrnv", [1, 1, 1], lean=spec.to_json())
    d = os.path.join(root, "models", eid)
    os.makedirs(d, exist_ok=True)
    S.save_json(os.path.join(d, "library.json"), lib.data)
    return eid


@pytest.fixture
def v2(tmp_path, monkeypatch):
    glib, maps = tmp_path / "glib", tmp_path / "maps"
    monkeypatch.setenv("ISPATIALGYM_GRASP_LIBRARY_V2", str(glib))
    monkeypatch.setenv("ISPATIALGYM_REACH_MAPS", str(maps))
    for k in ("ISPATIALGYM_REACH_MAP_ANCHORS", "ISPATIALGYM_REACH_MAP_REQUIRE", "ISPATIALGYM_SUPPORT_POLICY", "ISPATIALGYM_V2_ONLY", "ISPATIALGYM_REACH_MAPS_STRICT",
              "ISPATIALGYM_INTERIORS"):
        monkeypatch.delenv(k, raising=False)
    GL._v2_index.cache_clear()
    GL._v2_lean_index.cache_clear()
    e11 = _write_lean_entry(str(glib), LEAN)
    e20 = _write_lean_entry(str(glib), LN.LeanSpec(20.0, "left"), face=0.07)
    yield SimpleNamespace(glib=str(glib), maps=str(maps), e11=e11, e20=e20, k11=LN.lean_key(HB_KEY, LEAN), k20=LN.lean_key(HB_KEY, LN.LeanSpec(20.0, "left")))
    GL._v2_index.cache_clear()
    GL._v2_lean_index.cache_clear()


def _sampler(gen=None, seed=0, surface_z=0.84):
    g = dict(gen or {})
    return SimpleNamespace(world=SimpleNamespace(floor_z=0.0, scene_model=SCENE), ikc=SimpleNamespace(), gen=lambda k, d=None: g.get(k, d),
                           ctx=SimpleNamespace(base_seed=seed), template_facts={}, gp=SimpleNamespace(variant="touching_03"), surface_z=surface_z)


def _top(**kw):
    d = dict(name=SUPPORT, z=0.8354, headroom=th.tensor([[0.3465, 0.3465]]), mask=th.tensor([[True, True]]), open_edges=["+u", "-u"])
    d.update(kw)
    return SimpleNamespace(**d)


def _requests(caplog):
    return [json.loads(r.getMessage().split(" ", 1)[1]) for r in caplog.records if r.getMessage().startswith("MAP_REQUEST")]


def test_shelf_lean_key_takes_the_tasks_side_and_one_angle_per_template(v2, monkeypatch):
    from ispatialgym.sampling.v2 import tabletop as TT

    gen = {"partner_side": "right", "shelf_start_draw": {"lean_deg": [11.0, 20.0]}}
    keys = {TT.shelf_lean_key(_sampler(gen, seed=k), HB_KEY) for k in range(16)}
    assert keys == {v2.k11, v2.k20}  # both library angles, one per template seed
    s = _sampler(gen, seed=3)
    k = TT.shelf_lean_key(s, HB_KEY)
    assert k == TT.shelf_lean_key(_sampler(gen, seed=3), HB_KEY)  # deterministic per template
    f = s.template_facts["shelf_lean"]
    assert f["side"] == "left" and f["lean_key"] == k and "partner_side right" in f["side_why"] and f["pool"] == sorted(f["pool"])
    assert TT.shelf_lean_key(_sampler({**gen, "shelf_start_draw": {"lean_deg": [15.0, 25.0]}}), HB_KEY) == v2.k20
    with pytest.raises(Reject) as e:
        TT.shelf_lean_key(_sampler({"partner_side": "left"}), HB_KEY)  # the right board: no lean rest there
    assert e.value.reason == "missing_grasps:lean"
    assert TT.shelf_lean_key(_sampler(gen), "other__key") is None  # no lean rests of that key: the key's path as before
    monkeypatch.delenv("ISPATIALGYM_REACH_MAPS")
    assert TT.shelf_lean_key(_sampler(gen), HB_KEY) is None  # lean rests need their maps root


def test_upright_shelf_facts_and_side_of_a_lean_key(v2):
    from ispatialgym.sampling.v2 import tabletop as TT

    q = LN.lean_quat(LN.upright(LN.xyzw([0.5609855267969309, 0.4304593345768794, 0.43045933457687935, -0.5609855267969309])), LEAN)
    f = TT.upright_shelf_facts(v2.k11, th.tensor(q, dtype=th.float32))
    assert f["key"] == v2.k11 and f["lean_side"] == 1.0 and abs(f["lean_deg"] - 11.0) < 0.05 and f["lean_rest"] == LEAN.tag()
    assert TT.lean_rest_side(v2.k11) == 1.0 and TT.lean_rest_side(HB_KEY) is None and TT.lean_rest_of_key(v2.k20)["deg"] == 20.0


def test_interior_map_bands_hit_registers_the_board_and_misses_reject(v2, caplog):
    from ispatialgym.sampling.v2 import tabletop as TT

    sp = IN.spec_named(KMUACD)
    lib = GL.v2_entry(v2.k11)
    path = _write(os.path.join(v2.maps, v2.e11), _meta(lib, sp))
    spec = TT.BandSpec(object="hardback/acbrnv", grasp_key=v2.k11, scale=[1, 1, 1])
    s = _sampler()
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        view = TT.interior_map_bands(s, spec, _top(), "right")
        assert view.meta["map_path"] == path and view.meta["interior"] == KMUACD and view.meta["lean_rest"] == LEAN.tag()
        assert s.ikc.support_plane_z == {SUPPORT: 0.8354} and s.ikc.support_policy_rest["rest_id"] == LEAN.tag()
        assert os.environ["ISPATIALGYM_SUPPORT_POLICY"] == "grasp"
        misses = [("left arm", dict(arm="left")), ("the 20 deg rest", dict(spec=TT.BandSpec(object="hardback/acbrnv", grasp_key=v2.k20, scale=[1, 1, 1]))),
                  ("another bookcase", dict(top=_top(name="bookcase_otwukr_1"))), ("a lower headroom", dict(top=_top(headroom=th.tensor([[0.25, 0.25]])))),
                  ("a stable key", dict(spec=TT.BandSpec(object="hardback/acbrnv", grasp_key=HB_KEY, scale=[1, 1, 1])))]
        for what, kw in misses:
            with pytest.raises(Reject) as e:
                TT.interior_map_bands(s, kw.get("spec", spec), kw.get("top", _top()), kw.get("arm", "right"))
            assert e.value.reason == "missing_map", what
    reqs = _requests(caplog)
    assert len(reqs) == len(misses) and all(r["kind"] == "interior" and r["plane"] == "interior" for r in reqs)
    assert reqs[0]["arm"] == "left" and reqs[0]["interior"] == KMUACD and reqs[0]["entry"] == v2.e11 and reqs[1]["entry"] == v2.e20


def test_the_v2_dispatch_sends_a_lean_rest_to_its_interior_map(v2, monkeypatch, caplog):
    from ispatialgym.sampling.v2 import tabletop as TT

    sp = IN.spec_named(KMUACD)
    _write(os.path.join(v2.maps, v2.e11), _meta(GL.v2_entry(v2.k11), sp))

    def legacy(*a, **k):
        raise AssertionError("a legacy band was loaded / computed for a lean rest")

    monkeypatch.setattr(RB.ReachBands, "load", staticmethod(legacy))
    monkeypatch.setattr(TT, "compute_bands", legacy)
    spec = TT.BandSpec(object="hardback/acbrnv", grasp_key=v2.k11, scale=[1, 1, 1])
    s = _sampler()
    s.bands = None
    bands = TT.ensure_bands(s, spec, top=_top())  # not v2only: a lean rest still reads its map
    assert bands.meta["interior"] == KMUACD and s.bands is bands
    assert TT.v2_map_view(s, spec, _top(), arm="right").meta["interior"] == KMUACD
    with pytest.raises(Reject) as e:
        TT.map_view_arm_band(s, "left", spec=spec, top=_top())  # the left arm's map is missing: rejected, never the legacy path
    assert e.value.reason == "missing_map"
    s.surface_z = None  # the top surface: no lean map kind there
    with pytest.raises(Reject) as e:
        TT.v2_map_view(s, spec, _top(), arm="right")
    assert e.value.reason == "missing_map"
    # a plain (stable) key on the interior surface: as before (one MAP_REQUEST, None)
    s.surface_z = 0.84
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling.v2"):
        assert TT.v2_map_view(s, TT.BandSpec(object="hardback/acbrnv", grasp_key=HB_KEY, scale=[1, 1, 1]), _top(), arm="right") is None
    reqs = _requests(caplog)
    assert [r["kind"] for r in reqs] == ["interior"] and reqs[0]["grasp_key"] == HB_KEY and "entry" not in reqs[0]


def test_iors_small_keeps_the_library_lean_rest(v2):
    IS = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_small")
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    rest0 = th.tensor(LN.lean_quat([0, 0, 0, 1.0], LEAN), dtype=th.float32)
    s.grasp_key = v2.k11
    s._shelf_template = {"key": v2.k11, "lean_side": 1.0, "lean_deg": 11.0}
    s._shelf_rest0 = {"target": rest0.clone(), "partner": rest0.clone()}
    s.hull = {"partner": th.tensor([[-0.08, -0.015, -0.11], [0.08, 0.015, 0.11]])}
    t, p = s.shelf_rests(1.0, 17.0, 14.0)  # a drawn target lean is ignored: the library rest as measured
    assert th.allclose(t, rest0) and not th.allclose(p, rest0)
    s.grasp_key = HB_KEY  # a plain key: re-leaned as before
    t2, _ = s.shelf_rests(1.0, 17.0, 14.0)
    assert not th.allclose(t2, rest0)
    assert math.isclose(LN.lean_of(t2.double().numpy(), LN.up_axis_local(rest0.double().numpy()))["deg"], 17.0, abs_tol=0.05)
