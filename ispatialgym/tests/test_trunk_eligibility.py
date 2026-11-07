"""i7-unblock 2026-11-07 (owner decision: the former trunk eligibility was a bug; nextto_04): the reach maps' trunk eligibility per edge
distance - ``reach_maps.trunk_posture_groups`` (pure), ``MapConfig.trunk_eligibility`` in the map identity (old maps read as "all_d_e",
their keys unchanged), ``compute_map`` / ``compute_generic_map`` on fake worlds (the nextto_04 case, an all-clear trunk unchanged, an
all-colliding trunk still ineligible), the lookup preference and the band view. Simulator-free."""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import map_sampling as MS
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM

POSTURES = ("untucked", "ready", "tucked")


def _free_of(table, calls=None):
    def f(p):
        if calls is not None:
            calls.append(p)
        return table.get(p)
    return f


# ------------------------------------------------------------------------------------------------------------------
# the pure rule
# ------------------------------------------------------------------------------------------------------------------


def test_nextto_04_trunk_is_eligible_where_a_posture_clears_the_slab():
    # gym_shoe kmcbym h 0.3471, t0.95 k+0.0 (v2maps_i7): untucked / ready hit the slab at d_e 0.35 only, tucked self-collides
    table = {"untucked": [False, True, True], "ready": [False, True, True], "tucked": None}
    groups, collide = RM.trunk_posture_groups(POSTURES, 3, _free_of(table), "all_d_e")
    assert groups == [] and collide == {"untucked": [False, True, True], "ready": [False, True, True]}  # the former rule: dropped
    groups, collide = RM.trunk_posture_groups(POSTURES, 3, _free_of(table), "per_d_e")
    assert groups == [("untucked", [1, 2])] and collide == {"untucked": [False, True, True], "ready": [False, True, True]}


def test_an_all_clear_trunk_keeps_its_first_posture_under_both_rules():
    for rule in RM.TRUNK_ELIGIBILITY_RULES:
        calls = []
        groups, collide = RM.trunk_posture_groups(POSTURES, 3, _free_of({"untucked": [True] * 3, "ready": [True] * 3}, calls), rule)
        assert groups == [("untucked", [0, 1, 2])] and collide == {} and calls == ["untucked"]  # the later postures are not evaluated
        calls = []
        # the former choice when the first posture collides somewhere and a later one is free everywhere: that later one, for every d_e
        groups, collide = RM.trunk_posture_groups(POSTURES, 3, _free_of({"untucked": [False, True, True], "ready": [True] * 3}, calls), rule)
        assert groups == [("ready", [0, 1, 2])] and collide == {"untucked": [False, True, True]} and calls == ["untucked", "ready"]


def test_an_all_colliding_trunk_stays_ineligible():
    for rule in RM.TRUNK_ELIGIBILITY_RULES:
        groups, _ = RM.trunk_posture_groups(POSTURES, 3, _free_of({"untucked": [False] * 3, "ready": [False] * 3, "tucked": None}), rule)
        assert groups == []
        groups, collide = RM.trunk_posture_groups(POSTURES, 3, _free_of({}), rule)  # every posture self-collides
        assert groups == [] and collide == {}


def test_per_d_e_takes_the_first_posture_free_at_each_edge_distance():
    table = {"untucked": [False, True, False], "ready": [False, True, True], "tucked": [True, False, False]}
    groups, _ = RM.trunk_posture_groups(POSTURES, 3, _free_of(table), "per_d_e")
    assert groups == [("untucked", [1]), ("ready", [2]), ("tucked", [0])]
    with pytest.raises(ValueError):
        RM.trunk_posture_groups(POSTURES, 3, _free_of(table), "some_d_e")
    with pytest.raises(ValueError):
        RM.trunk_posture_groups(POSTURES, 3, _free_of({"untucked": [True, True]}), "per_d_e")


def test_the_interior_obstacle_ranking_of_i7_lean_is_kept():
    # i7-lean b4230177: among the postures free of the board at every d_e, the first free of the compartment's obstacles at the most d_e
    obst = {"untucked": [False, False, True], "ready": [True, True, True], "tucked": [True, False, False]}
    table = {"untucked": [True] * 3, "ready": [True] * 3, "tucked": [True] * 3}
    calls = []
    for rule in RM.TRUNK_ELIGIBILITY_RULES:
        groups, _ = RM.trunk_posture_groups(POSTURES, 3, _free_of(table, calls), rule, obstacle_free_of=lambda p: obst[p])
        assert groups == [("ready", [0, 1, 2])]
    assert calls == ["untucked", "ready"] * 2  # stops at a posture free of the obstacles everywhere
    groups, _ = RM.trunk_posture_groups(POSTURES, 3, _free_of({"untucked": [True] * 3, "ready": [True] * 3}), "per_d_e",
                                        obstacle_free_of=lambda p: {"untucked": [True, False, False], "ready": [False, True, False]}[p])
    assert groups == [("untucked", [0, 1, 2])]  # a tie in the count keeps the earlier posture
    # the per_d_e fallback (no posture free of the board everywhere): per d_e the first board-free posture that is also obstacle-free there
    groups, _ = RM.trunk_posture_groups(POSTURES, 3, _free_of({"untucked": [False, True, True], "ready": [False, True, True]}), "per_d_e",
                                        obstacle_free_of=lambda p: {"untucked": [True, False, True], "ready": [True, True, True]}[p])
    assert groups == [("untucked", [2]), ("ready", [1])]


# ------------------------------------------------------------------------------------------------------------------
# identity
# ------------------------------------------------------------------------------------------------------------------


def test_the_rule_is_in_the_config_and_old_keys_stay():
    assert "trunk_eligibility" not in RM.MapConfig(trunk_eligibility=None).key()  # the former rule: the config (and key) as before
    k = RM.MapConfig(trunk_eligibility="per_d_e").key()
    assert k["trunk_eligibility"] == "per_d_e"
    assert RM.config_value({"config": {}}, "trunk_eligibility") == "all_d_e" and RM.config_value({"config": k}, "trunk_eligibility") == "per_d_e"
    assert RM.trunk_rule(RM.MapConfig(trunk_eligibility=None)) == "all_d_e"
    with pytest.raises(ValueError):
        RM.trunk_rule(RM.MapConfig(trunk_eligibility="any"))
    assert RM._trunk_kw(SimpleNamespace(trunk_eligibility="all_d_e")) == {"trunk_eligibility": None}
    assert RM._trunk_kw(SimpleNamespace(trunk_eligibility="per_d_e")) == {"trunk_eligibility": "per_d_e"}
    assert RM._trunk_kw(SimpleNamespace(trunk_eligibility=None)) == {}


# ------------------------------------------------------------------------------------------------------------------
# compute_map / compute_generic_map on fake worlds
# ------------------------------------------------------------------------------------------------------------------

H = 0.8
#: (translate, tilt) -> the idle arm's sphere x (m ahead; z 2 cm under the slab top) for the self-free postures: 0.40 is inside the slab at
#: d_e 0.35 only (the nextto_04 case), 0.60 at every d_e (all-colliding), None far above (all-clear)
IDLE_X = {(0.25, -0.3): None, (0.95, 0.0): 0.40, (0.5, -0.6): 0.60}


class _World:
    """The SynthWorld surface compute_map / compute_generic_map use, on CPU: a configuration is a pose (flattened 4x4) + the posture code;
    tucked self-collides everywhere; the idle-arm spheres of a trunk follow IDLE_X (``clear`` = every trunk all-clear); every sphere of a
    solved configuration is far above the slab; the FK eef is the target."""

    def __init__(self, cfg, clear=False):
        self.cfg, self.dev, self.clear = cfg, th.device("cpu"), clear

        def check_collision(q, emb_name, margin):
            return th.tensor([bool(q[0, 2] >= 2)])

        def robot_spheres_world(q):
            x = None if self.clear else IDLE_X[(round(float(q[0, 0]), 2), round(float(q[0, 1]), 2))]
            z = 2.0 if x is None else H - 0.02
            return th.tensor([[[0.0 if x is None else x, 0.0, z, 0.02]]]), ["torso_link4"]

        self.backend = SimpleNamespace(check_collision=check_collision, robot_spheres_world=robot_spheres_world, position_tolerance=0.005,
                                       orientation_tolerance=0.05)

    def q_config(self, translate, tilt, posture="untucked", fingers=0.05):
        return th.tensor([translate, tilt, float(POSTURES.index(posture))])

    def shoulder(self, translate, tilt):
        return np.array([0.3, -0.2, 1.1]), 0.9

    def entry(self):
        return None

    def link_of_sphere(self):
        return ["torso_link4"]

    def full_q(self, entry, q_active, q_lock):
        return q_active

    def spheres(self, q_full):
        return th.tensor([0.0, 0.0, 2.0, 0.02]).repeat(q_full.shape[0], 1, 1)

    def eef(self, q_full):
        return q_full[:, :16].reshape(-1, 4, 4)


def _solve(world, entry, T, q_lock, seed_active=None):
    n, S_ = T.shape[0], world.cfg.seeds
    ok = (T[:, 0, 3] < 0.95)[:, None].repeat(1, S_)
    return ok, T.reshape(n, 1, 16).repeat(1, S_, 1)


def _lib(tmp_path):
    d = S.model_dir("apple", "x", [1, 1, 1], str(tmp_path / "glib"))
    os.makedirs(d)
    S.save_json(os.path.join(d, "library.json"), {
        "schema": S.LIBRARY, "category": "apple", "model": "x", "scale": [1, 1, 1], "arm": "right", "legacy_key": "apple__x__t",
        "policy_key": CP.DEFAULT.key("right"), "rests": {"r0": {"schema": S.REST, "rest_id": "r0", "quat_xyzw": [0, 0, 0, 1], "h_root": 0.04, "stable": True}},
        "grasps": [{"name": "g0", "rest_id": "r0", "T_object_eef": {"position": [0, 0, 0.1], "orientation_xyzw": [1, 0, 0, 0]}, "width_m": 0.03}]})
    return S.load_model("apple", "x", [1, 1, 1], str(tmp_path / "glib"))


def _key(tr, ti):
    return f"t{tr:.2f}_k{ti:+.1f}"


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(RM, "_solve", _solve)
    monkeypatch.setattr(RB, "trunk_grid", lambda h: list(IDLE_X))


@pytest.mark.parametrize("generic", [False, True])
def test_compute_maps_per_d_e(tmp_path, fake, generic):
    lib = _lib(tmp_path)

    def run(rule, clear=False):
        cfg = RM.MapConfig(seeds=2, twins=False, lattice_m=0.16, trunk_eligibility=rule)
        if generic:
            return RM.compute_generic_map(_World(cfg, clear), H, {"key": "0.8000"}, cfg=cfg)
        return RM.compute_map(_World(cfg, clear), lib, "r0", H, {"key": "0.8000"}, cfg=cfg)

    m_old, a_old = run(None)
    m_new, a_new = run("per_d_e")
    m_ref, a_ref = run(None, clear=True)  # every trunk all-clear: the cells a trunk has where its idle arm clears the slab
    t_old = {t["key"]: t for t in m_old["trunks"]}
    t_new = {t["key"]: t for t in m_new["trunks"]}
    clear, nextto, colliding = _key(0.25, -0.3), _key(0.95, 0.0), _key(0.5, -0.6)
    # the former rule: only the all-clear trunk
    assert [t_old[k]["eligible"] for k in (clear, nextto, colliding)] == [True, False, False]
    assert "trunk_eligibility" not in m_old["config"] and m_new["config"]["trunk_eligibility"] == "per_d_e" and m_old["key"] != m_new["key"]
    # an all-clear trunk: the same posture, the same arrays and record under both rules
    assert t_new[clear] == t_old[clear] and "posture_by_d_e" not in t_new[clear]
    for d in (0.35, 0.45, 0.55):
        assert np.array_equal(a_new[f"s{d:.2f}_{clear}"], a_old[f"s{d:.2f}_{clear}"])
    # the nextto_04 trunk: ineligible at d_e 0.35 only; the other d_e hold the cells the trunk has with a clear idle arm
    assert t_new[nextto]["eligible"] and t_new[nextto]["posture"] == "untucked" and t_new[nextto]["ineligible_d_e"] == [0.35]
    assert t_new[nextto]["posture_by_d_e"] == {"0.45": "untucked", "0.55": "untucked"}
    assert t_new[nextto]["slab_collide_by_posture"] == {"untucked": [False, True, True], "ready": [False, True, True]}
    assert (a_new[f"s0.35_{nextto}"] == RM.STATUS["trunk_ineligible"]).all()
    assert (a_old[f"s0.45_{nextto}"] == RM.STATUS["trunk_ineligible"]).all()
    for d in (0.45, 0.55):
        assert np.array_equal(a_new[f"s{d:.2f}_{nextto}"], a_ref[f"s{d:.2f}_{nextto}"])
    assert int((a_new[f"s0.45_{nextto}"] == RM.STATUS["feasible"]).sum()) > 0
    assert t_new[nextto]["n_feasible"] == {k: v for k, v in t_ref_feasible(m_ref, nextto).items() if k != "0.35"}
    # an all-colliding trunk: still ineligible everywhere
    assert not t_new[colliding]["eligible"] and all((a_new[f"s{d:.2f}_{colliding}"] == RM.STATUS["trunk_ineligible"]).all() for d in (0.35, 0.45, 0.55))
    # the band view: the nextto trunk is offered, with cells at 0.45 / 0.55 and none at 0.35
    p = RM.save(str(tmp_path / "maps"), dict(m_new, object={"category": "apple", "model": "x"}), a_new) if not generic else None
    if p:
        view = MS.as_bands(RM.ReachMap(p), overhang=0.0)
        assert len(view.band(0.95, 0.0, stand_off=0.35)) == 0 and len(view.band(0.95, 0.0, stand_off=0.45)) > 0


def t_ref_feasible(meta, key):
    return next(t for t in meta["trunks"] if t["key"] == key)["n_feasible"]


# ------------------------------------------------------------------------------------------------------------------
# lookup
# ------------------------------------------------------------------------------------------------------------------


def _write(root, rule, tag):
    cfg = RM.MapConfig(trunk_eligibility=rule).key()
    meta = {"schema": RM.SCHEMA, "kind": "tabletop", "arm": "right", "object_key": "obj", "rest_id": "r0", "height": {"value": 0.3471},
            "edge_distances": [0.35, 0.45, 0.55], "config": cfg, "policy_key": "pk", "robot": "rb", "brace": None, "trunks": [], "grasps": []}
    meta["key"] = tag
    base = os.path.join(root, f"map__gym_shoe__kmcbym__h0.3471__{tag}")
    np.savez_compressed(base + ".npz", cells=np.zeros((1, 2), np.float32), yaws=np.zeros(8, np.float32))
    with open(base + ".json", "w") as f:
        json.dump(meta, f)
    return base + ".json"


def test_find_serves_both_rules_and_prefers_per_d_e(tmp_path):
    lib = SimpleNamespace(identity_key=lambda: "obj")
    kw = dict(rest_id="r0", arm="right", policy_key="pk", robot="rb")
    old = _write(str(tmp_path), None, "0000aaaa")  # sorts first
    assert RM.ReachMap.find(str(tmp_path), lib, 0.3471, **kw).path == old  # a root of old maps only: as before
    new = _write(str(tmp_path), "per_d_e", "ffffbbbb")
    assert RM.ReachMap.find(str(tmp_path), lib, 0.3471, **kw).path == new
    assert RM.ReachMap.find(str(tmp_path), lib, 0.3471, config_require={"trunk_eligibility": "all_d_e"}, **kw).path == old
    assert RM.ReachMap.find(str(tmp_path), lib, 0.3471, config_require={"trunk_eligibility": "per_d_e"}, **kw).path == new
