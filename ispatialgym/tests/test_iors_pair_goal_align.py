"""IORS small, pair goal alignment (owner 2026-10-13 ~22:23, COMPLAINTS.md "Pair goal alignment": touching#02, nextto#01 / #03 / #04 - the
target ends in a set orientation relative to its partner, the way a person would put the pair down): ``generation.pair_goal_align`` - each
model's front (its pointing end), the fronts' relative yaw, where the target's front points, the pair side by side across their long axes,
the touching faces and a low push - drawn at init, kept to the goal, levelled, checked after the goal settle. On the synthetic table of
``test_iors_start_region`` and with the removed instances' poses. No simulator."""

from __future__ import annotations

import json
import math
import os
import types

import numpy as np
import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks import iors_small as IS
from ispatialgym.tests.test_iors_start_region import _sampler

CUSTOMS = os.path.join(os.path.dirname(__file__), "..", "assets", "task_specs", "isg_customs.json")


def _box(dx, dy, dz, c=(0.0, 0.0, 0.0)):
    """A box hull of the given extents centred at ``c`` (object frame)."""
    return th.tensor([[c[0] + sx * dx / 2, c[1] + sy * dy / 2, c[2] + sz * dz / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=th.float32)


#: the eight models as the probe measured them (complaints_1013/pairgoal_work/probe1: collision extents and centres in their own frames, the
#: spawn scale applied) and their collision oriented boxes' long axes (misc/metadata.json; None: it stands up - the footprint decides)
MODELS = {
    "marker/xakoic": (_box(0.1537, 0.0172, 0.0141, (-0.012, -0.001, 0.001)), (1.0, -0.008, 0.013)),
    "colored_pencil/acumgp": (_box(0.1523, 0.0077, 0.0048, (-0.004, 0.0, 0.0)), (-1.0, -0.017, 0.0)),
    "tablespoon/yxxfdj": (_box(0.1673, 0.0382, 0.023, (0.015, 0.0, 0.0)), (-0.999, 0.005, 1.75 * 0.043)),
    "chopstick/bcwuwv": (_box(0.2224, 0.0087, 0.0052, (-0.012, 0.0, 0.0)), (1.0, -0.022, 0.001)),
    "gym_shoe/kmcbym": (_box(0.3081, 0.112, 0.1446, (0.02, -0.001, 0.022)), (0.999, 0.033, 0.0)),
    "gym_shoe/mcfcwq": (_box(0.307, 0.1158, 0.1433, (0.021, 0.006, 0.023)), (-0.997, -0.08, 0.004)),
    "box_of_cereal/iucshm": (_box(0.0427, 0.1567, 0.2182), (0.0, 0.0, -1.0)),
    "box_of_cereal/dkhgxn": (_box(0.0784, 0.1548, 0.2417, (-0.001, 0.001, -0.001)), (-0.02, -0.001, -1.0)),
}
#: the keys of isg_customs.json (DECISIONS_LOG 2026-10-13 22:40: fronts from the probe's cross-sections, confirmed on its renders)
KEYS = {
    "nextto01": {"front": {"marker/xakoic": "+x", "colored_pencil/acumgp": "-x"}, "relative_yaw_deg": 0.0, "facing": "away"},
    "nextto03": {"front": {"tablespoon/yxxfdj": "-x", "chopstick/bcwuwv": "-x"}, "relative_yaw_deg": 0.0, "facing": "away"},
    "nextto04": {"front": {"gym_shoe/kmcbym": "+x", "gym_shoe/mcfcwq": "+x"}, "relative_yaw_deg": 0.0, "facing": "away"},
    "touching02": {"front": {"box_of_cereal/iucshm": "+x", "box_of_cereal/dkhgxn": "-x"}, "relative_yaw_deg": 0.0, "facing": "any", "faces": "widest",
                   "push_max_height_m": 0.035},
}
PAIRS = {"nextto01": ("marker/xakoic", "colored_pencil/acumgp"), "nextto03": ("tablespoon/yxxfdj", "chopstick/bcwuwv"),
         "nextto04": ("gym_shoe/kmcbym", "gym_shoe/mcfcwq"), "touching02": ("box_of_cereal/iucshm", "box_of_cereal/dkhgxn")}


def _model(key):
    cat, model = key.split("/")
    return types.SimpleNamespace(category=cat, model=model, scale=None, grasp_key=None)


def _pair_sampler(monkeypatch, pair="nextto04", gen=None, relation=None, g_nt=0.0628):
    """The synthetic table (top z 0.75, x 1.0-1.6, robot at x 0.55 facing +x, tangent +y) with ``pair``'s two models."""
    relation = relation or ("touching" if pair.startswith("touching") else "nextto")
    s = _sampler(relation, L=3.0, gen={"pair_goal_align": KEYS[pair], **(gen or {})}, pair={"g_NT": g_nt})
    t, p = PAIRS[pair]
    s.hull = {"target": MODELS[t][0], "partner": MODELS[p][0]}
    s.choice = types.SimpleNamespace(models={"target": _model(t), "partner": _model(p)})
    axes = {"target": MODELS[t][1], "partner": MODELS[p][1]}
    monkeypatch.setattr(IS.IORSSmall, "model_long_axis", lambda self, role: th.tensor(axes[role], dtype=th.float32))
    s.meta_spots, s.counters, s.goal_side = None, {}, None
    s.init_stance.extras["margins"] = {"D_p": 0.30}
    return s


def _manip(s, y=-0.24):
    m = Stance("manipulation", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, s.edge)
    m.focus, m.arms = th.tensor([1.31, y, 0.78]), ["right"]
    m.extras[TT.START_REGION] = {"left": [0.03, 0.30], "right": [0.03, 0.30]}
    return m


def _pose(s, role, yaw, xy):
    q = G.qmul(G.quat_from_yaw(yaw), s.rest[role])
    return th.tensor([float(xy[0]), float(xy[1]), 0.75 + s.world.bottom_offset(s.hull[role], q) + 0.003]), q


def _deg(a):
    return math.degrees(G.wrap_angle(a))


# ---------------------------------------------------------------------------------------------------------------- the key


def test_off_by_default_and_spec_errors(monkeypatch):
    plain = _sampler("nextto", pair={"g_NT": 0.05})
    assert plain.pair_goal_align() is None
    s = _pair_sampler(monkeypatch)
    spec = s.pair_goal_align()
    assert spec["facing"] == "away" and spec["relative_yaw_deg"] == 0.0 and spec["front_axes"] == {"target": "gym_shoe/kmcbym +x", "partner": "gym_shoe/mcfcwq +x"}
    assert spec["tol_deg"] == IS.PAIR_TOL_DEG and spec["stagger_max_m"] == IS.PAIR_STAGGER_MAX_M and spec["faces"] is None
    bad = [
        dict(relation="ontop"),
        dict(gen={"pair_goal_align": {"front": {"gym_shoe/kmcbym": "+x"}}}),  # no front for the partner model
        dict(gen={"pair_goal_align": dict(KEYS["nextto04"], front={"gym_shoe/kmcbym": "+x", "gym_shoe/mcfcwq": "+z"})}),
        dict(gen={"pair_goal_align": dict(KEYS["nextto04"], facing="up")}),
        dict(gen={"pair_goal_align": dict(KEYS["nextto04"], faces="widest")}),  # touching only
        dict(gen={"pair_goal_align": dict(KEYS["nextto04"], push_max_height_m=0.03)}),  # touching only
        dict(gen={"partner_aligned_to_target": True}),
        dict(gen={"partner_on_meta_link": "heatsource"}),
    ]
    for kw in bad:
        with pytest.raises(Reject) as e:
            _pair_sampler(monkeypatch, **kw).pair_goal_align()
        assert e.value.reason == "spec" and "pair_goal_align" in str(e.value), kw
    # a box's printed front is across its long side: it can never point away from the robot with the long side across the edge
    away = _pair_sampler(monkeypatch, "touching02", gen={"pair_goal_align": dict(KEYS["touching02"], facing="away")})
    with pytest.raises(Reject) as e:
        away.pair_target_yaws(_manip(away))
    assert e.value.reason == "spec" and "front away" in str(e.value)


def test_front_yaw_is_the_named_end_snapped_onto_the_long_axis(monkeypatch):
    """gym_shoe/mcfcwq's long axis is 4.6 deg off its local x: its toe (+x) points along that axis; a box's printed face (+x) is the normal of
    its footprint's long side (the oriented box's long axis stands up)."""
    s = _pair_sampler(monkeypatch, "nextto04")
    lean = math.degrees(math.atan2(0.08, 0.997))
    for yaw in (0.0, 0.7, -2.0):
        q = G.qmul(G.quat_from_yaw(yaw), s.rest["partner"])
        assert _deg(s.front_yaw("partner", q) - yaw) == pytest.approx(lean, abs=1e-3)
        assert _deg(s.front_yaw("target", G.qmul(G.quat_from_yaw(yaw), s.rest["target"])) - yaw) == pytest.approx(math.degrees(math.atan2(0.033, 0.999)), abs=1e-3)
    b = _pair_sampler(monkeypatch, "touching02")
    for yaw in (0.0, 1.0):
        assert _deg(b.front_yaw("target", G.quat_from_yaw(yaw)) - yaw) == pytest.approx(0.0, abs=1e-3)  # iucshm's front: +x
        assert abs(_deg(b.front_yaw("partner", G.quat_from_yaw(yaw)) - yaw)) == pytest.approx(180.0, abs=1e-3)  # dkhgxn's: -x


@pytest.mark.parametrize("inward", [(1.0, 0.0), (math.cos(0.3), math.sin(0.3)), (0.0, -1.0)])
def test_target_yaws_long_axis_across_the_edge_front_per_facing(monkeypatch, inward):
    def stance(s):
        edge = EdgeFrame("e", th.tensor([1.0, 0.0]), th.tensor(inward, dtype=th.float32), 2.0)
        return Stance("manipulation", (0.55, 0.0, math.atan2(inward[1], inward[0])), 0.5, 0.0, 0.45, edge)

    want = math.atan2(inward[1], inward[0])
    for facing, expect in (("away", [0.0]), ("toward", [180.0]), ("any", [0.0, 180.0])):
        s = _pair_sampler(monkeypatch, "nextto04", gen={"pair_goal_align": dict(KEYS["nextto04"], facing=facing)})
        st = stance(s)
        yaws = s.pair_target_yaws(st)
        fronts = sorted(round(abs(_deg(s.front_yaw("target", G.qmul(G.quat_from_yaw(y), s.rest["target"])) - want))) for y in yaws)
        assert fronts == expect
        for y in yaws:
            q = G.qmul(G.quat_from_yaw(y), s.rest["target"])
            assert IS.axes_angle_deg(s.pair_axis_angle("target", q), want) < 1e-3  # the toe-heel axis across the edge
    # the boxes: the footprint's long side across the edge, the printed fronts along it (to the robot's left or right)
    b = _pair_sampler(monkeypatch, "touching02")
    st = stance(b)
    yaws = b.pair_target_yaws(st)
    assert len(yaws) == 2 and abs(_deg(yaws[1] - yaws[0])) == pytest.approx(180.0, abs=1e-4)
    for y in yaws:
        assert IS.axes_angle_deg(b.front_yaw("target", G.quat_from_yaw(y)), want + math.pi / 2) < 1e-3
        assert b.widest_face_axis(b.hull["target"], G.quat_from_yaw(y), st) == "tangent"


@pytest.mark.parametrize("pair", ["nextto01", "nextto03", "nextto04", "touching02"])
def test_partner_yaw_puts_the_fronts_the_required_turn_apart(monkeypatch, pair):
    s = _pair_sampler(monkeypatch, pair)
    st = _manip(s)
    for y in s.pair_target_yaws(st) + [0.4]:
        t_pose = _pose(s, "target", y, (1.31, -0.24))
        p_pose = _pose(s, "partner", s.pair_partner_yaw(t_pose), (1.31, 0.0))
        f = s.pair_facts(t_pose, p_pose, st)
        assert f["rel_yaw_deg"] == pytest.approx(0.0, abs=1e-3) and f["long_axes_deg"] < 0.01
    s180 = _pair_sampler(monkeypatch, pair, gen={"pair_goal_align": dict(KEYS[pair], relative_yaw_deg=180.0)})
    t_pose = _pose(s180, "target", s180.pair_target_yaws(st)[0], (1.31, -0.24))
    p_pose = _pose(s180, "partner", s180.pair_partner_yaw(t_pose), (1.31, 0.0))
    assert abs(_deg(s180.front_yaw("target", t_pose[1]) - s180.front_yaw("partner", p_pose[1]))) == pytest.approx(180.0, abs=1e-3)


# ---------------------------------------------------------------------------------------------------------------- the draw, the goal


@pytest.mark.parametrize("pair", ["nextto04", "nextto03"])
def test_place_partner_draws_the_partner_beside_level_and_parallel(monkeypatch, pair):
    """Every drawn partner: its front the target's (0 deg), the long axes parallel and across the edge, the footprint rectangles' centres level
    (the shoes' origins sit 2 cm toward the toe, the spoon's 1.5 cm toward the handle), on the attempt's side; without the key the partner's
    yaw is random."""
    s = _pair_sampler(monkeypatch, pair)
    manip = _manip(s)
    s.region_side = 1.0
    t_pose = _pose(s, "target", s.pair_target_yaws(manip)[0], (1.31, -0.24))
    for seed in range(8):
        p_pose = s.place_partner(np.random.default_rng(seed), t_pose, manip, None)
        f = s.pair_facts(t_pose, p_pose, manip)
        assert f["rel_yaw_deg"] == pytest.approx(0.0, abs=1e-3) and f["long_axes_deg"] < 1e-3 and f["facing_deg"] < 1e-3
        assert abs(f["stagger_m"]) < 1e-4 and f["offset_deg"] == pytest.approx(90.0, abs=0.01) and f["side"] == "right" and f["pair_side"] == "right"
        assert s.partner_draw["side"] == "left" and f["gap_m"] >= float(s.gap_range()[0]) - 1e-3
    plain = _sampler("nextto", L=3.0, pair={"g_NT": 0.0628})
    plain.hull, plain.meta_spots = dict(s.hull), None
    plain.init_stance.extras["margins"] = {"D_p": 0.30}
    plain.region_side = 1.0
    yaws = {round(math.degrees(G.yaw_of(plain.place_partner(np.random.default_rng(k), t_pose, manip, None)[1]))) for k in range(6)}
    assert len(yaws) > 2


def _goal_sampler(monkeypatch, pair="nextto04", **kw):
    s = _pair_sampler(monkeypatch, pair, **kw)
    manip = _manip(s)
    s.sidecar = types.SimpleNamespace(stance=lambda name: manip, grasp_of=lambda name: None)
    s.bindings = types.SimpleNamespace(one=lambda role: types.SimpleNamespace(name=role))
    s.region_side = 1.0
    return s, manip


@pytest.mark.parametrize("pair", ["nextto01", "nextto03", "nextto04"])
def test_goal_puts_the_target_beside_the_partner_and_follows_its_settled_front(monkeypatch, pair):
    s, manip = _goal_sampler(monkeypatch, pair)
    t_pose = _pose(s, "target", s.pair_target_yaws(manip)[0], (1.31, -0.24))
    p_pose = s.place_partner(np.random.default_rng(3), t_pose, manip, None)
    goal = s.goal_pose(t_pose, p_pose)
    f = s.pair_facts(goal, p_pose, manip)
    assert s.goal_side == "right" and f["side"] == "right" and f["rel_yaw_deg"] == pytest.approx(0.0, abs=1e-3)
    assert abs(f["stagger_m"]) < 1e-4 and f["offset_deg"] == pytest.approx(90.0, abs=0.01) and f["faces"] == {"target": "widest", "partner": "widest"}
    assert G.quat_angle(goal[1], t_pose[1]) < 1e-4  # no in-hand turn: the init yaw already meets the requirement
    tan = manip.tangent
    lat_gap = float((s.world.footprint(s.hull["partner"], p_pose) @ tan).min() - (s.world.footprint(s.hull["target"], goal) @ tan).max())
    assert lat_gap == pytest.approx(float(s.pair["g_NT"]) - 0.01, abs=1e-4) and f["gap_m"] >= lat_gap - 1e-4 and s.pair_violation(f) == ""
    # the partner settled 2 deg turned and 1 cm deeper: the goal follows its front and its level
    drift = (p_pose[0] + th.tensor([0.01, 0.0, 0.0]), G.qmul(G.quat_from_yaw(math.radians(2.0)), p_pose[1]))
    g2 = s.goal_pose(t_pose, drift)
    f2 = s.pair_facts(g2, drift, manip)
    assert f2["rel_yaw_deg"] == pytest.approx(0.0, abs=1e-3) and abs(f2["stagger_m"]) < 0.01  # levelled along the edge's depth axis
    assert s.pair_violation(f2) == ""
    with pytest.raises(Reject) as e:  # 8 deg: beyond tol_deg, the attempt is rejected
        s.goal_pose(t_pose, (p_pose[0], G.qmul(G.quat_from_yaw(math.radians(8.0)), p_pose[1])))
    assert e.value.reason == "pair_goal_align"


def test_touching_goal_meets_on_the_widest_faces(monkeypatch):
    s, manip = _goal_sampler(monkeypatch, "touching02", g_nt=0.0496)
    t_pose = _pose(s, "target", s.pair_target_yaws(manip)[0], (1.31, -0.24))
    p_pose = _pose(s, "partner", s.pair_partner_yaw(t_pose), (1.31, 0.0))
    p_pose = (th.tensor([float(s.pair_level_depth("partner", p_pose[1], IS.min_area_rect(s.world.footprint(s.hull["target"], t_pose))[3], manip)) + 1.0,
                         0.0, float(p_pose[0][2])]), p_pose[1])
    goal = s.goal_pose(t_pose, p_pose)
    f = s.pair_facts(goal, p_pose, manip)
    assert f["faces"] == {"target": "widest", "partner": "widest"} and f["gap_m"] < 1e-4 and abs(f["stagger_m"]) < 1e-4 and s.pair_violation(f) == ""
    # c8_k2's layout (both widest faces toward the robot, touching on the narrow faces) breaks it
    t_wide = _pose(s, "target", 0.0, (1.31, -0.24))
    p_wide = _pose(s, "partner", math.pi, (1.31, 0.0))
    f_bad = s.pair_facts(s.goal_pose(t_wide, p_wide), p_wide, manip)
    assert f_bad["faces"] == {"target": "narrow", "partner": "narrow"} and "narrow" in s.pair_violation(f_bad)


def test_check_pair_goal_records_and_rejects_the_settled_goal(monkeypatch):
    s, manip = _goal_sampler(monkeypatch, "nextto04")
    t_pose = _pose(s, "target", s.pair_target_yaws(manip)[0], (1.31, -0.24))
    p_pose = s.place_partner(np.random.default_rng(1), t_pose, manip, None)
    goal = s.goal_pose(t_pose, p_pose)
    obj = lambda pose: types.SimpleNamespace(get_position_orientation=lambda: (pose[0].clone(), pose[1].clone()))  # noqa: E731
    draft = types.SimpleNamespace(summary={"pair_goal_align": {}})
    s.objects = {"t": obj(goal), "p": obj(p_pose)}
    s.check_pair_goal(draft, "t", "p")
    assert draft.summary["pair_goal_align"]["goal_settled"]["rel_yaw_deg"] == pytest.approx(0.0, abs=1e-3)
    # toe to heel (c3_k3): the target in line with the partner; crossed (c1_k2's L)
    for pose, words in (((p_pose[0] + th.tensor([0.0, 0.0, 0.0]) + th.tensor([0.32, 0.0, 0.0]), p_pose[1]), "staggered"),
                        ((goal[0], G.qmul(G.quat_from_yaw(math.pi / 2), goal[1])), "long axes")):
        s.objects["t"] = obj(pose)
        with pytest.raises(Reject) as e:
            s.check_pair_goal(draft, "t", "p")
        assert e.value.reason == "pair_goal_align" and words in str(e.value) and s.counters["pair_goal_align"] >= 1


def test_removed_instances_break_the_requirement(monkeypatch):
    """The owner's examples at their goals (tro poses): nextto#01 cmp1013_c1_k2 perpendicular (an L), nextto#03 cmp1013_c1_k4 the spoon across
    the chopstick (a T), nextto#04 cmp1013_c3_k3 toe to heel, touching#02 cmp1013_c8_k2 on the narrow faces."""
    cases = {  # (pair, edge inward, target goal (x, y, yaw deg), partner (x, y, yaw deg))
        "nextto01": ((1.0, -0.0026), (6.343, 3.844, -89.6), (6.343, 3.945, 178.9)),
        "nextto03": ((0.0026, 1.0), (6.418, 3.482, -179.7), (6.507, 3.482, -88.9)),
        "nextto04": ((-1.0, 0.0025), (21.012, 24.28, 93.8), (21.001, 23.917, 85.2)),
        "touching02": ((-1.0, 0.0025), (8.399, 0.485, -178.2), (8.402, 0.328, -1.9)),
    }
    found = {}
    for pair, (inward, t, p) in cases.items():
        s = _pair_sampler(monkeypatch, pair)
        edge = EdgeFrame("e", th.tensor([t[0], t[1]]), th.tensor(inward, dtype=th.float32), 1.2)
        st = Stance("manipulation", (t[0] - inward[0], t[1] - inward[1], math.atan2(inward[1], inward[0])), 0.5, 0.0, 0.45, edge)
        s.goal_side = None
        f = s.pair_facts(_pose(s, "target", math.radians(t[2]), t[:2]), _pose(s, "partner", math.radians(p[2]), p[:2]), st)
        found[pair] = (f, s.pair_violation(f))
    assert found["nextto01"][0]["long_axes_deg"] > 85 and "long axes" in found["nextto01"][1]
    assert found["nextto03"][0]["long_axes_deg"] > 85 and "long axes" in found["nextto03"][1]
    assert found["nextto04"][0]["long_axes_deg"] < 10 and abs(found["nextto04"][0]["stagger_m"]) > 0.3 and "staggered" in found["nextto04"][1]
    assert found["touching02"][0]["faces"] == {"target": "narrow", "partner": "narrow"} and "narrow" in found["touching02"][1]


def test_edge_partner_quats_follow_the_key(monkeypatch):
    s = _pair_sampler(monkeypatch, "nextto04")
    quats = s.edge_partner_quats()
    assert len(quats) == 1 and IS.axes_angle_deg(s.pair_axis_angle("partner", quats[0]), 0.0) < 1e-3  # across the edge (inward +x)
    assert _deg(s.front_yaw("partner", quats[0])) == pytest.approx(0.0, abs=1e-3)  # the left shoe's toe away from the robot, as the target's


# ---------------------------------------------------------------------------------------------------------------- target yaws, push height


def test_phase3_two_tier_screens_the_pair_target_yaws_without_jitter(monkeypatch):
    seen = []
    monkeypatch.setattr(TT, "phase3_target", lambda self, manip, rng, gs, *a, **k: seen.append(k) or ("pose", "grasp"))
    s = _pair_sampler(monkeypatch, "nextto01", gen={"init_align_support": True})
    s.grasps = s.grasps_flush = ["g"]
    s.support_objects = lambda: []
    manip = _manip(s)
    assert s.phase3_two_tier(manip, np.random.default_rng(0)) == ("pose", "grasp")
    assert seen[0]["jitter"] == 0.0 and seen[0]["yaws"] == pytest.approx(s.pair_target_yaws(manip)) and len(seen[0]["yaws"]) == 1


def test_touching_push_candidates_above_the_cap_are_not_tried(monkeypatch):
    """A 4.3 cm thick box pushed across its thin axis tips when pushed above b / mu (~4.3 cm): with ``push_max_height_m`` only the low blades
    are offered (a blade's fingertips, else the eef's height at the face, above the target's bottom)."""
    fake = [("push_tip_h", (th.tensor([1.3, 0.0, 1.16]), C.IDENTITY_QUAT.clone()), {"tip_front": True}),
            ("push_blade_lean20", (th.tensor([1.3, 0.0, 1.20]), C.IDENTITY_QUAT.clone()), {"fingertip_z": 1.075}),
            ("push_blade_0_high", (th.tensor([1.3, 0.0, 1.30]), C.IDENTITY_QUAT.clone()), {"fingertip_z": 1.173})]
    for gen, expect in (({}, ["push_tip_h", "push_blade_lean20", "push_blade_0_high"]), (None, ["push_blade_lean20"])):
        s = _pair_sampler(monkeypatch, "touching02", g_nt=0.0496) if gen is None else _sampler("touching", pair={"g_NT": 0.0496})
        seen = []
        monkeypatch.setattr(IS.IORSSmall, "touching_push_candidates", lambda self, *a, **k: iter(fake))
        monkeypatch.setattr(IS.O, "choose_first", lambda ikc, arm, stream, **k: (seen.extend(c[0] for c in stream), (None, None, []))[1])
        s.ikc = types.SimpleNamespace(q_locked=lambda st: None)
        s._touching_push_choice(_manip(s), None, th.tensor([1.3, 0.0, 1.16]), (1.05, 1.27), th.tensor([0.0, 1.0, 0.0]), 0.08, [])
        assert seen == expect


# ---------------------------------------------------------------------------------------------------------------- the customs


def test_the_four_aliases_carry_the_key_and_every_use_is_valid():
    with open(CUSTOMS) as f:
        customs = json.load(f)
    want = {"inter_object_relation_satisfaction-touching-box_of_cereal_s080-box_of_cereal-bar": "touching02",
            "inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table": "nextto01",
            "inter_object_relation_satisfaction-nextto-tablespoon-chopstick-breakfast_table": "nextto03",
            "inter_object_relation_satisfaction-nextto-gym_shoe-gym_shoe-bench": "nextto04"}
    have = {k for k, e in customs.items() if isinstance(e, dict) and (e.get("generation") or {}).get("pair_goal_align") is not None}
    assert have == set(want)
    for alias, name in want.items():
        g = customs[alias]["generation"]
        assert g["pair_goal_align"] == KEYS[name]
        roles = customs[alias]["roles"]
        for role in ("target", "partner"):
            assert f"{roles[role]['category']}/{roles[role]['representative']}" in g["pair_goal_align"]["front"]
        assert g["pair_goal_align"]["facing"] in IS.PAIR_FACINGS
        assert all(v in IS.PAIR_FRONT_AXES for v in g["pair_goal_align"]["front"].values())
    assert customs["inter_object_relation_satisfaction-touching-box_of_cereal_s080-box_of_cereal-bar"]["generation"]["widest_faces"] == "parallel"
    assert customs["inter_object_relation_satisfaction-touching-box_of_cereal_s080-box_of_cereal-bar"]["generation"]["touching_place_gap_m"] == 0.05
    for alias in list(want)[1:]:
        assert customs[alias]["generation"]["init_align_support"] is True
