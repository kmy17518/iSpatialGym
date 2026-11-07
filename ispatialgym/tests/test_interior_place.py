"""i7-interior 2026-11-07: interior PLACE maps (``sampling/interior_place.py``) - the goal-side interior map of a stable rest judged as
stance C's exact place (the working arm only, any idle posture, the lifted / tilted release, the variant's pre-place, every lateral of the
base), the rule in the identity and the lookup, the keys of every other map unchanged. Simulator-free (the fake synthetic world of
``test_interior_maps``)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.grasping import contact_policy as CP
from ispatialgym.grasping import schema as S
from ispatialgym.sampling import interior_place as IP
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling import reach_maps as RM
from ispatialgym.tests.test_interior_fridge import _fridge, _jar_lib
from ispatialgym.tests.test_interior_maps import TRUNK, FakeWorld, _fake_solve


# ------------------------------------------------------------------------------------------------------------------ the rule


def test_the_rule_is_canonical_and_covers_only_its_own_place():
    r = IP.place_rule(pre="outward", drop_m=0.03, tilts_deg=[45, 30, 0, 30.0])
    assert r["rule"] == IP.RULE and r["tilts_deg"] == [30.0, 45.0] and r["laterals_m"] == list(IP.DEFAULT_LATERALS_M) and r["body"] == "arm"
    assert IP.parse_rule(None) is None and IP.parse_rule("none") is None and IP.parse_rule('{"pre": "outward", "drop_m": 0.03, "tilts_deg": [30, 45]}') == r
    assert IP.parse_rule(r) == r  # idempotent (the "rule" field is dropped and re-set)
    with pytest.raises(ValueError):
        IP.place_rule(pre="sideways")
    with pytest.raises(ValueError):
        IP.place_rule(tilts_deg=[95])
    level = IP.place_rule(pre="outward", drop_m=0.03)
    assert IP.covers(r, level) is None and IP.covers(level, r) is not None  # more tilts cover fewer, not the reverse
    assert IP.covers(r, IP.place_rule(pre="approach", drop_m=0.03)) is not None and IP.covers(r, IP.place_rule(pre="outward", drop_m=0.02)) is not None
    assert IP.covers(IP.place_rule(pre="outward", drop_m=0.03, laterals_m=[0.0], lattice_m=0.08), level) is None  # the builder's resolution
    assert IP.covers(IP.place_rule(pre="outward", drop_m=0.03, body="robot"), level) is not None  # a measurement map never serves
    assert IP.covers(None, None) is None and IP.covers(level, None) is not None and IP.covers(None, level) is not None


def test_the_keys_of_every_other_map_stay():
    sp = _fridge()
    for cfg in (RM.MapConfig(), RM.floor_config(), RM.interior_config(sp), RM.interior_config(sp, rest_kind="stable")):
        assert "interior_place" not in cfg.key()
    k0 = RM.interior_config(sp, rest_kind="stable").key()
    assert "x_min_m" not in k0 and k0["lattice_m"] == 0.08  # the former fridge map's config, as before
    cfg = RM.interior_config(sp, rest_kind="stable", place=IP.place_rule(pre="outward", drop_m=0.03))
    k = cfg.key()
    assert k["interior_place"]["pre"] == "outward" and k["lattice_m"] == 0.04 and k["x_min_m"] == round(0.35 + 0.03, 4)
    assert k["y_half_m"] >= 0.5 * sp.width_m + 0.08 and k["interior"] == sp.digest()
    with pytest.raises(ValueError):
        RM.interior_config(sp, rest_kind="lean", place=IP.place_rule())


# ------------------------------------------------------------------------------------------------------------------ the release (pure geometry)


def test_the_tilt_and_the_open_gripper_are_the_samplers_rules():
    IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large_inside")
    IS = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_small")
    level = th.tensor([0.0, 0.0, math.sin(0.3), math.cos(0.3)])  # xyzw: a yaw of 0.6 rad
    p = [0.12, -0.03, 0.01]
    for deg in (0.0, 30.0, 45.0):
        q = IL.tilted_quat(level, th.tensor(p), deg)
        R = IP.tilted_R(S.quat_to_mat(level.numpy()), p, deg)
        assert np.allclose(R, S.quat_to_mat(np.asarray(q, dtype=float)), atol=1e-5)
    pos, quat = th.tensor([0.4, 0.1, 0.8]), th.tensor([0.2, 0.7, -0.1, 0.68])
    quat = quat / th.linalg.norm(quat)
    low = float(IS.gripper_corners_world((pos, quat))[:, 2].min())
    assert abs(IP.open_gripper_low(S.quat_to_mat(quat.numpy()), pos.numpy()) - low) < 1e-5
    assert IP.release_lift([0.70, 0.75], 0.7064, 0.02) == pytest.approx(0.7064 + 0.02 - 0.70) and IP.release_lift([0.8], 0.7064, 0.02) == 0.0
    assert IP.release_lift([0.70], 0.7064, 0.02) == pytest.approx(IL.release_lift([0.70], 0.7064, 0.02))


def test_the_release_table_lifts_every_library_grasp_clear_and_tilts_the_grasp_side_up():
    import trimesh

    V = np.asarray(trimesh.creation.box(extents=[0.24, 0.24, 0.02]).vertices)  # a flat plate
    side = np.eye(4)
    side[:3, :3] = S.quat_to_mat(np.array([0.0, 0.7071068, 0.0, 0.7071068]))  # approach along the object's +x, at its rim
    side[:3, 3] = [-0.13, 0.0, 0.0]
    T_og = side[None]
    h = 0.7751
    rule = IP.place_rule(drop_m=0.02, tilts_deg=[30, 45])
    R, z, lift, tilts = IP.release_table(V, np.eye(3), [0.0], T_og, [0], h, rule)
    assert tilts == [0.0, 30.0, 45.0] and R.shape == (1, 3, 1, 3, 3)
    for k in range(3):
        Rt = R[0, k, 0]
        bottom = -float((V @ Rt.T)[:, 2].min())
        assert z[0, k, 0] == pytest.approx(h + bottom + 0.02 + lift[0, k, 0])
        low = IP.open_gripper_low(Rt @ T_og[0, :3, :3], np.array([0.0, 0.0, z[0, k, 0]]) + Rt @ T_og[0, :3, 3])
        assert low >= h + 0.02 - 1e-6  # the open gripper clears the surface by the clearance
    eef_z = [float((R[0, k, 0] @ T_og[0, :3, 3])[2]) for k in range(3)]
    assert eef_z[0] < eef_z[1] < eef_z[2]  # the grasp side up
    R2, z2, lift2, t2 = IP.release_table(V, np.eye(3), [0.0], T_og, [0], h, IP.place_rule(release="rest", tilts_deg=[30]))
    assert t2 == [0.0] and z2[0, 0, 0] == pytest.approx(h + 0.01) and lift2.max() == 0.0  # the former map: resting on the board


# ------------------------------------------------------------------------------------------------------------------ the map (fake world)


DOOR_PT = (0.45 - 0.10, 0.20, 1.0)  # inside the test fridge's door box at d_e 0.45 (lateral 0)


class IdleArmWorld(FakeWorld):
    """The fake world with one STATIC sphere of the idle left arm standing in the open door (the measured fridge: left_arm_link1 /
    link3 in the door at every posture)."""

    def spheres(self, q_full):
        sp = super().spheres(q_full)
        idle = th.tensor([[DOOR_PT[0], DOOR_PT[1], DOOR_PT[2], 0.05]]).repeat(sp.shape[0], 1, 1)
        return th.cat([sp, idle], 1)

    def link_of_sphere(self):
        return super().link_of_sphere() + ["left_arm_link3"]


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(RM, "_solve", _fake_solve)
    monkeypatch.setattr(RB, "trunk_grid", lambda h: [TRUNK])
    import trimesh

    jar = trimesh.creation.box(extents=[0.08, 0.08, 0.15])
    return SimpleNamespace(geom=SimpleNamespace(union=jar, pieces=[jar]))


def _place_map(fake, rule, world_cls=IdleArmWorld, sp=None):
    sp = sp or _fridge()
    cfg = RM.interior_config(sp, rest_kind="stable", place=rule, seeds=2, batch=64, edge_distances=(0.45,), postures=("untucked",), n_yaws=2)
    meta, arrays = RM.compute_map(world_cls(cfg), _jar_lib(), "r0", sp.height_m, {"key": "x", "height": sp.height_m}, geom=fake.geom, cfg=cfg,
                                  kind="interior", interior=sp)
    return meta, arrays, arrays[f"s0.45_{RB.trunk_key(*TRUNK)}"]


def test_the_static_body_is_stance_Cs_check_not_the_maps(fake):
    rule = IP.place_rule(pre="outward", drop_m=0.03, laterals_m=[0.0])
    meta, arr, st = _place_map(fake, rule)
    assert meta["interior_placement"] == "fixed_place" and meta["config"]["interior_place"] == rule and meta["interior_place"] == rule
    assert (st == RM.STATUS["feasible"]).any()  # the idle arm in the door does not remove the working arm's places
    meta_r, _, st_r = _place_map(fake, IP.place_rule(pre="outward", drop_m=0.03, laterals_m=[0.0], body="robot"))
    assert not (st_r == RM.STATUS["feasible"]).any() and any(k.startswith("obstacle:left_arm_link3") for k in meta_r["interior_sphere_hits"])
    assert meta["key"] != meta_r["key"]
    assert "release_lift" in arr and arr["release_z"].shape == (2, 1, len(meta["grasps"]))


def test_every_lateral_of_the_base_adds_its_cells(fake):
    one = _place_map(fake, IP.place_rule(pre="outward", drop_m=0.03, laterals_m=[0.0]), FakeWorld)[2] == RM.STATUS["feasible"]
    many_meta, arr, st = _place_map(fake, IP.place_rule(pre="outward", drop_m=0.03, laterals_m=[0.0, 0.12]), FakeWorld)
    many = st == RM.STATUS["feasible"]
    # the two maps' lattices are the same here? (y_half grows with the laterals) - compare by cell coordinates
    c_one = {tuple(np.round(c, 3)) for c in _place_map(fake, IP.place_rule(pre="outward", drop_m=0.03, laterals_m=[0.0]), FakeWorld)[1]["cells"][one.any(axis=(1, 2))]}
    c_many = {tuple(np.round(c, 3)) for c in arr["cells"][many.any(axis=(1, 2))]}
    assert c_one and c_one <= c_many and len(c_many) > len(c_one)
    assert min(y for _, y in c_many) < min(y for _, y in c_one)  # the base 0.12 m to the left: the compartment to its right
    assert many_meta["release"]["centres_y_m"] == [0.0, -0.12]


def test_a_place_rule_is_an_interior_stable_rest_field_only(fake):
    cfg = RM.MapConfig(seeds=2, batch=64, edge_distances=(0.45,), interior_place=IP.place_rule())
    with pytest.raises(ValueError):
        RM.compute_map(FakeWorld(cfg), _jar_lib(), "r0", 0.7, {"key": "x", "height": 0.7}, geom=fake.geom, cfg=cfg, kind="tabletop")


def test_find_serves_a_place_lookup_from_a_place_map_only(tmp_path):
    from ispatialgym.tests.test_interior_fridge import _meta, _write

    sp, lib = _fridge(), _jar_lib()
    rule = IP.place_rule(pre="outward", drop_m=0.03)
    former = _write(str(tmp_path), _meta(lib, sp))
    place = _write(str(tmp_path), _meta(lib, sp, place=rule))
    ok = dict(rest_id="r0", arm="right", anchors=None, policy_key=CP.DEFAULT.key("right"), robot=RM.robot_digest(), kind="interior", interior=sp)
    assert RM.ReachMap.find(str(tmp_path), lib, sp.height_m, **ok).path == former  # the former lookup (no rule): the former map only
    assert RM.ReachMap.find(str(tmp_path), lib, sp.height_m, interior_place=rule, **ok).path == place
    assert RM.ReachMap.find(str(tmp_path), lib, sp.height_m, interior_place=IP.place_rule(pre="approach", drop_m=0.03), **ok) is None
    assert RM._map_meta_matches(place, "right", "interior", interior=sp.digest(), interior_place=rule)
    assert not RM._map_meta_matches(former, "right", "interior", interior=sp.digest(), interior_place=rule)
