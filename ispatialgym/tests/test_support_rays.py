"""oracle-fix 2026-11-06 (owner: a pack of kielbasa beside a platter, its round side over the rim, passed ontop): the rays branch of the
robust under / ontop check needs the centre of the smaller footprint covered and SUPPORT_MIN_HITS contact hits; the old rule behind
ISPATIALGYM_SUPPORT_RAYS=any. Simulator-free: the AABBs, link paths and ray casts are stubbed."""
import sys
import types

import pytest
import torch as th

from ispatialgym.sampling import predicates as PRED
from ispatialgym.sampling import predicates_fast as PF


class Obj:
    def __init__(self, name, lo, hi):
        self.name, self.aabb, self.links = name, (th.tensor(lo), th.tensor(hi)), {}


def _install_rays(monkeypatch, hit_points, hit_z=0.8228):
    """raytest_batch: downward probes hit the platter's surface (z 0.8178 at its rim, 0.8128 table elsewhere - all 'platter'
    here); upward probes hit the kielbasa at hit_z only at the listed footprint points."""
    def raytest_batch(starts, ends, only_closest=True, ignore_bodies=None):
        out = []
        for s, e in zip(starts, ends):
            if e[2] < s[2]:  # down: the supporting surface
                out.append({"hit": True, "rigidBody": "/World/platter/base_link", "position": [s[0], s[1], 0.8178]})
            else:
                on = any(abs(s[0] - x) < 1e-6 and abs(s[1] - y) < 1e-6 for x, y in hit_points)
                out.append({"hit": on, "rigidBody": "/World/kielbasa/base_link", "position": [s[0], s[1], hit_z]} if on else {"hit": False})
        return out

    mod = types.ModuleType("omnigibson.utils.sampling_utils")
    mod.raytest_batch = raytest_batch
    monkeypatch.setitem(sys.modules, "omnigibson.utils.sampling_utils", mod)
    monkeypatch.setattr(PRED, "_link_paths", lambda o: [f"/World/{o.name}"])


@pytest.fixture
def scene():
    platter = Obj("platter", [6.80, 2.94, 0.8128], [7.13, 3.28, 0.8294])  # 0.33 x 0.34 m
    kielbasa = Obj("kielbasa", [6.70, 3.04, 0.8128], [6.84, 3.14, 0.875])  # 0.14 x 0.10 m, beside the platter's rim
    return platter, kielbasa


def test_rays_verdict_rules():
    probes = [(0, 0.80, False, None), (3, 0.80, True, 0.81)] + [(i, 0.80, False, None) for i in (1, 2, 4, 5, 6, 7, 8)]
    assert PRED.rays_verdict(probes, "any")["ok"] is True  # the kielbasa: one rim hit was enough before
    v = PRED.rays_verdict(probes, "centre")
    assert v["ok"] is False and v["centre_hit"] is False and v["n_hits_on_upper"] == 1
    nine = [(i, 0.80, True, 0.802) for i in range(9)]  # a teacup on its saucer: 9 / 9
    assert PRED.rays_verdict(nine, "centre")["ok"] is True
    far = [(i, 0.80, True, 0.80 + PRED.RAY_CONTACT_MAX_M + 0.01) for i in range(9)]  # hovering 3 cm above
    assert PRED.rays_verdict(far, "centre")["ok"] is False and PRED.rays_verdict(far, "centre")["n_hits_far"] == 9
    two = [(0, 0.80, True, 0.80), (1, 0.80, True, 0.80)] + [(i, 0.80, False, None) for i in range(2, 9)]
    assert PRED.rays_verdict(two, "centre")["ok"] is False  # centre covered, but fewer than SUPPORT_MIN_HITS


def test_footprint_owner_is_the_smaller_object(monkeypatch, scene):
    platter, kielbasa = scene
    assert PRED.footprint_owner(platter, kielbasa, "centre") is kielbasa
    sheet, spatula = Obj("sheet", [0, 0, 0.8], [0.3, 0.4, 0.801]), Obj("spatula", [0.1, 0.1, 0.78], [0.15, 0.35, 0.8])
    assert PRED.footprint_owner(spatula, sheet, "centre") is spatula  # under(spatula, sheet): the spatula's centre covered
    assert PRED.footprint_owner(spatula, sheet, "any") is sheet
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_RAYS", "any")
    assert PRED.support_rays_rule() == "any"


@pytest.mark.parametrize("impl", ["reference", "fast"])
def test_the_kielbasa_on_the_rim_is_not_ontop_and_a_resting_one_is(monkeypatch, scene, impl):
    platter, kielbasa = scene
    _install_rays(monkeypatch, [])
    pts = PRED._footprint_points(kielbasa)
    rays = (lambda lo, up: PRED.support_rays(lo, up)) if impl == "reference" else (lambda lo, up: PF.support_rays(PRED, lo, up))
    _install_rays(monkeypatch, [pts[3]])  # one ring point over the rim
    r = rays(platter, kielbasa)
    assert r["ok"] is False and r["n_hits_on_upper"] == 1 and r["centre_hit"] is False
    monkeypatch.setenv("ISPATIALGYM_SUPPORT_RAYS", "any")
    assert rays(platter, kielbasa)["ok"] is True  # the rule before
    monkeypatch.delenv("ISPATIALGYM_SUPPORT_RAYS")
    _install_rays(monkeypatch, pts)  # resting on it: every probe hits 5 mm above the surface
    r = rays(platter, kielbasa)
    assert r["ok"] is True and r["n_hits_on_upper"] == 9 and r["centre_hit"] is True
