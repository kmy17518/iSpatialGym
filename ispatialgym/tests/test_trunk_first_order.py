"""i7-placeorder 2026-11-07 (owner decision, option 1 for inside_large_01): the stance C trunk-first ORDER - the least trunk translate
first, then the nearest stand-off; the order only (the set and its prior order among equal keys kept); opt-in per alias
(``generation.c_order_trunk_first``), set for inside_large_01 only, which no longer caps its place stand-offs. Simulator-free."""

import inspect
import json
import os
from types import SimpleNamespace

from ispatialgym import REPO_ROOT
from ispatialgym.sampling.v2.tasks import iors_large as IL

ALIAS_VARIANT = "inter_object_relation_satisfaction-inside_large_scale-hardback-bookcase-desk"


def _c(translate, stand_off, name):
    return SimpleNamespace(translate=translate, tilt=-0.6, stand_off=stand_off, name=name)


def test_the_least_extended_trunk_first_then_the_nearest_stand_off():
    pool = [_c(0.50, 0.65, "a"), _c(0.25, 0.85, "b"), _c(0.0, 0.85, "c"), _c(0.25, 0.65, "d"), _c(0.50, 0.55, "e"), _c(0.0, 0.55, "f")]
    prior = [0, 1, 2, 3, 4, 5]
    out = IL.trunk_first_order(pool, prior)
    assert [pool[k].name for k in out] == ["f", "c", "d", "b", "e", "a"]
    assert sorted(out) == sorted(prior)  # the order only: no candidate dropped or added


def test_equal_keys_keep_the_prior_order_and_the_subset_given():
    pool = [_c(0.25, 0.85, "x0"), _c(0.25, 0.85, "x1"), _c(0.25, 0.85, "x2"), _c(0.5, 0.55, "y"), _c(0.0, 0.85, "z")]
    # the prior (travel / edge rank / the seeded permutation) decides among equal (translate, stand-off)
    assert IL.trunk_first_order(pool, [2, 3, 0, 1, 4]) == [4, 2, 0, 1, 3]
    assert IL.trunk_first_order(pool, [1, 0, 2]) == [1, 0, 2]
    assert IL.trunk_first_order(pool, []) == []
    # a missing stand-off sorts as 0 (nearest) within its translate
    pool2 = [_c(0.25, 0.85, "p"), _c(0.25, None, "q")]
    assert IL.trunk_first_order(pool2, [0, 1]) == [1, 0]


def test_the_coupling_applies_it_only_when_the_alias_asks():
    src = inspect.getsource(IL.IORSLarge._couple)
    assert 'self.gp.gen("c_order_trunk_first", False)' in src and "order = trunk_first_order(c_pool, order)" in src
    assert '"stance_c.extended_used"' in src and '"stance_c.trunk_first"' in src


def test_inside_large_01_asks_for_it_and_keeps_every_stand_off():
    d = json.load(open(os.path.join(REPO_ROOT, "ispatialgym", "assets", "task_specs", "isg_customs.json")))
    g = d[ALIAS_VARIANT]["generation"]
    assert g.get("c_order_trunk_first") is True
    assert "inside_stand_off_max_m" not in g and max(g["inside_stand_offs_m"]) == 0.85  # the 0.70 m cap is gone
    others = [k for k, v in d.items() if isinstance(v, dict) and k != ALIAS_VARIANT and (v.get("generation") or {}).get("c_order_trunk_first")]
    assert others == []  # this alias only
