"""Owner rules 2026-10-11 of the IORS small-scale pairs (``sampling/pair_rules.py``): the grounding-time target fit scale (ontop /
under: the partner's top; inside: the opening) with its grasp-key route in ``ground_isg``, the nextto handedness that
``ground_isg`` writes and ``IORSSmall`` honours, and the standing inside target (``generation.target_up_axis``, inside#03: measured
and placed standing on that axis). No simulator; the grounder runs on a small sheet of the new representatives' models (the
shipped candidates / require sheets, a copy of ``grasps.csv`` the tests control)."""

import math
import types

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import pair_rules as PR
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import Stance, nearer_side_lateral

IORS = "inter_object_relation_satisfaction"
ONTOP, UNDER, INSIDE, NEXTTO = (f"{IORS}-ontop_small_scale", f"{IORS}-under_small_scale", f"{IORS}-inside", f"{IORS}-nextto")


# ---------------------------------------------------------------------------------------------------------------- pure rules


def test_rest_extents_turns_the_scaled_box():
    assert PR.rest_extents([0.1, 0.2, 0.3]) == [0.1, 0.2, 0.3]
    assert PR.rest_extents([0.1, 0.2, 0.3], [0.5, 0.5, 2.0]) == pytest.approx([0.05, 0.1, 0.6])
    # x_down rest (90 deg about y): x and z swap - the sampler's hull extents at the same rest agree
    c = math.sqrt(0.5)
    ext = PR.rest_extents([0.1, 0.2, 0.3], None, [c, 0.0, c, 0.0])
    assert ext == pytest.approx([0.3, 0.2, 0.1], abs=1e-9)
    box = th.tensor([[sx * 0.05, sy * 0.1, sz * 0.15] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    assert TT.hull_extents(box, th.tensor([0.0, c, 0.0, c])).tolist() == pytest.approx(ext, abs=1e-6)


def test_fit_on_partner_is_the_samplers_tier_check_and_shrinks_to_the_aligned_tier():
    soap, dish = [0.1087, 0.086, 0.0312], [0.0806, 0.1533, 0.0162]  # ontop#03: bar_soap/ofargb on soap_dish/slksbk
    fit = PR.fit_on_partner(soap, dish, 0.01)
    assert fit.bigger and fit.tier is None and fit.limit == "partner_short_side"
    assert PR.floor_scale(fit.scale) == 0.70 and fit.scale == pytest.approx((0.0806 - 0.02) / 0.086)
    s = PR.floor_scale(fit.scale)
    assert PR.on_partner_tier([v * s for v in soap], dish, 0.01) == "aligned"
    # a cup as long as its saucer but within it fits the centred tier as is (under#01 teacup/oxfzfe on saucer/vghfkh)
    assert PR.fit_on_partner([0.1299, 0.0943, 0.0707], [0.1443, 0.1442, 0.0113], 0.01).tier == "centred"
    # kielbasa on its platter: aligned; with fit_tiers = [any_yaw] it is bigger
    kielbasa, platter = [0.2209, 0.0668, 0.0625], [0.2173, 0.3448, 0.0166]
    assert PR.fit_on_partner(kielbasa, platter, 0.01).tier == "aligned"
    assert PR.fit_on_partner(kielbasa, platter, 0.01, ["any_yaw"]).bigger


def test_fit_inside_opening_and_protrusion():
    sunflower, vase = [0.192, 0.0792, 0.0645], [0.0952, 0.0952, 0.163]  # inside#03 representative: lying flower, tube vase
    fit = PR.fit_inside(sunflower, vase, 0.01, 0.5)
    assert fit.bigger and fit.limit == "opening_long_side" and PR.floor_scale(fit.scale) == 0.44
    assert not PR.fit_inside([0.058, 0.043, 0.0411], [0.1676, 0.1676, 0.1025]).bigger  # sprout in a bowl
    # a tall target in a shallow tray: it would protrude more than half its height even on the tray's floor
    tall = PR.fit_inside([0.03, 0.03, 0.20], [0.2, 0.2, 0.04], 0.01, 0.5)
    assert tall.bigger and tall.limit == "protrusion" and tall.scale == pytest.approx(0.04 / 0.10)


SUNFLOWER, TUBE_VASE = [0.192, 0.0792, 0.0645], [0.0952, 0.0952, 0.163]  # inside#03: sunflower/dkajbh (stem end -> head along +x), vase/gopbrh
UP_KEY = {"sunflower/dkajbh": "+x"}


def test_target_up_axis_is_read_per_model_like_the_wall_frames():
    assert PR.target_up_axis(UP_KEY, "sunflower", "dkajbh") == "+x"
    assert PR.target_up_axis(UP_KEY, "sunflower", "mdpoct") is None and PR.target_up_axis(None, "sunflower", "dkajbh") is None
    by_cat = {"sunflower": "x", "sunflower/urpskm": "-y"}
    assert PR.target_up_axis(by_cat, "sunflower", "mdpoct") == "+x" and PR.target_up_axis(by_cat, "sunflower", "urpskm") == "-y"
    assert PR.target_up_axis({"default": "+z"}, "vase", "gopbrh") == "+z" and PR.target_up_axis("+x", "sunflower", "dkajbh") == "+x"
    with pytest.raises(ValueError):
        PR.target_up_axis({"sunflower/dkajbh": "up"}, "sunflower", "dkajbh")
    for name, (i, sign) in PR.UP_AXES.items():
        R = PR._quat_wxyz_matrix(PR.standing_quat_wxyz(name))
        assert [R[r][i] * sign for r in range(3)] == pytest.approx([0.0, 0.0, 1.0], abs=1e-12), f"{name} is not turned up"


def test_fit_inside_measures_a_standing_target_on_its_up_axis():
    lying = PR.fit_inside(PR.rest_extents(SUNFLOWER), TUBE_VASE, 0.01, 0.5)
    assert lying.bigger and PR.floor_scale(lying.scale) == 0.44
    standing = PR.rest_extents(SUNFLOWER, None, PR.standing_quat_wxyz("+x"))
    assert standing == pytest.approx([0.0645, 0.0792, 0.192])
    # footprint 7.9 x 6.5 cm in the 8.5 cm opening, sinks 9.6 of its 19.2 cm in the 16.3 cm vase: fits at scale 1, no shrink
    fit = PR.fit_inside(standing, TUBE_VASE, 0.01, 0.5)
    assert not fit.bigger and fit.scale == 1.0
    assert PR.rest_extents(SUNFLOWER, None, PR.standing_quat_wxyz("-x")) == pytest.approx(standing)
    assert PR.fit_inside(PR.rest_extents(SUNFLOWER, None, PR.standing_quat_wxyz("+y")), TUBE_VASE, 0.01, 0.5).bigger  # the long side across


def test_floor_scale_and_uniform_factor():
    assert PR.floor_scale(0.70465) == 0.70 and PR.floor_scale(0.70) == 0.70 and PR.floor_scale(0.93) == 0.93 and PR.floor_scale(0.4437) == 0.44
    assert PR.uniform_factor([0.7, 0.7, 0.7], None) == pytest.approx(0.7)
    assert PR.uniform_factor([2.607, 2.607, 2.607], [3.724] * 3) == pytest.approx(0.7, abs=1e-3)
    assert PR.uniform_factor([0.7, 0.7, 0.875], [1.0, 1.0, 1.25]) == pytest.approx(0.7)
    assert PR.uniform_factor([1.0, 1.646, 2.306], None) is None


def test_handedness_rules():
    assert PR.handed_target_side("right", "left") == "right" and PR.handed_target_side("left", "right") == "left"
    assert PR.handed_target_side("right", None) is None and PR.handed_target_side(None, "left") is None
    with pytest.raises(ValueError):
        PR.handed_target_side("left", "left")
    # place_partner's side +1 (partner on the robot's left of the target) leaves the target on the partner's right: the goal's
    # nearer side (the side facing the target's start) is then "right"
    assert PR.partner_side_for("right") == 1.0 and PR.partner_side_for("left") == -1.0
    side, _ = nearer_side_lateral(0.0, (0.10, 0.20), (-0.03, 0.03), 0.01)
    assert side == "right"
    side, _ = nearer_side_lateral(0.0, (-0.20, -0.10), (-0.03, 0.03), 0.01)
    assert side == "left"


def _box(ext):
    x, y, z = (float(e) / 2 for e in ext)
    return th.tensor([[sx * x, sy * y, sz * z] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


@pytest.mark.parametrize("relation,target,partner", [
    ("ontop", [0.1087, 0.086, 0.0312], [0.0806, 0.1533, 0.0162]),
    ("ontop", [0.2209, 0.0668, 0.0625], [0.2173, 0.3448, 0.0166]),
    ("under", [0.3512, 0.0758, 0.025], [0.21, 0.29, 0.0005]),
    ("under", [0.1146, 0.0885, 0.0928], [0.1568, 0.1568, 0.0211]),
    ("inside", [0.192, 0.0792, 0.0645], [0.0952, 0.0952, 0.163]),
    ("inside", [0.0499, 0.0328, 0.0201], [0.2594, 0.4157, 0.1992]),
])
def test_grounding_verdict_matches_the_samplers_pair_check(relation, target, partner):
    """The grounder's *bigger* is exactly the sampler's template-time rejection (``partner_too_small`` / ``does_not_fit_inside``),
    and the target at the floored fit scale passes it."""
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    gen = {"partner_footprint_margin_m": 0.01, "fit_tiers": ["any_yaw", "aligned", "centred"], "opening_margin_m": 0.01}

    def check(t):
        s = types.SimpleNamespace(relation=relation, hull={"target": _box(t), "partner": _box(partner)},
                                  rest={"target": C.IDENTITY_QUAT.clone(), "partner": C.IDENTITY_QUAT.clone()}, gen=lambda k, d=None: gen.get(k, d))
        try:
            return IS.IORSSmall.pair_admissibility(s)
        except Reject as e:
            return e.reason

    fit = PR.fit_inside(target, partner, 0.01, 0.5) if relation == "inside" else PR.fit_on_partner(target, partner, 0.01)
    out = check(target)
    assert isinstance(out, str) == fit.bigger
    if fit.bigger:
        assert out == ("does_not_fit_inside" if relation == "inside" else "partner_too_small")
        assert isinstance(check([v * PR.floor_scale(fit.scale) for v in target]), dict)
    elif relation != "inside":
        assert out["tier"] == fit.tier


# ---------------------------------------------------------------------------------------------------------------- grounding

#: (subtask, pick number, slot, target category/model/grasp key, partner category/model) - models of the 2026-10-11 sheet
# ---------------------------------------------------------------------------------------------------------------- sampler


def test_sampler_handedness_pair_and_partner_side():
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    hands = {"gym_shoe/kmcbym": "right", "gym_shoe/mcfcwq": "left", "gym_shoe/picdro": "right"}

    def stub(t, p):
        models = {"target": types.SimpleNamespace(category="gym_shoe", model=t), "partner": types.SimpleNamespace(category="gym_shoe", model=p)}
        return types.SimpleNamespace(gen=lambda k, d=None: {"handedness": hands}.get(k, d), choice=types.SimpleNamespace(models=models))

    assert IS.IORSSmall.handedness_pair(stub("kmcbym", "mcfcwq")) == {"target": "right", "partner": "left", "target_side": "right"}
    assert IS.IORSSmall.handedness_pair(stub("mcfcwq", "kmcbym"))["target_side"] == "left"
    assert IS.IORSSmall.handedness_pair(stub("kmcbym", "ugbxip")) is None  # a hand unknown: no side rule
    with pytest.raises(Reject) as e:
        IS.IORSSmall.handedness_pair(stub("kmcbym", "picdro"))
    assert e.value.reason == "handedness_pair"
    # A1 (owner 2026-10-12): the pair's admissible partner sides (+1 = the robot's left of the target), fixed before the target is placed
    assert IS.partner_sides("nextto", {"target": "right", "partner": "left", "target_side": "right"}) == [1.0]
    assert IS.partner_sides("nextto", {"target": "left", "partner": "right", "target_side": "left"}) == [-1.0]
    assert IS.partner_sides("nextto", None) == [1.0, -1.0]


def _box_at(lo, hi):
    return th.tensor([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


#: A top-down stem pinch at the sunflower's rest (approach -z, closing -y of the object) and the manipulation stance facing +y.
TOP_DOWN = (th.tensor([-0.04, 0.0, 0.05]), C.mat_to_quat(th.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])))
FACING_Y = Stance("manipulation", (0.0, -0.6, math.pi / 2), 0.5, 0.0, 0.45, None)


def _flower_sampler(gen, relation="inside", eef_in_object=TOP_DOWN):
    """An ``IORSSmall`` without a simulator: the sunflower's collision boxes (stem x -0.146..-0.03, head x -0.03..0.048), the tube
    vase, a world that only projects hulls, ``gen`` as the variant's generation."""
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    s = IS.IORSSmall.__new__(IS.IORSSmall)
    head = _box_at((-0.03, -0.0415, -0.0271), (0.048, 0.0391, 0.0388))
    s.relation, s.hull = relation, {"target": th.cat([_box_at((-0.146, -0.005, -0.005), (-0.03, 0.005, 0.005)), head]), "partner": _box(TUBE_VASE)}
    s.rest = {"target": C.IDENTITY_QUAT.clone(), "partner": C.IDENTITY_QUAT.clone()}
    s.spec = types.SimpleNamespace(generation=gen)
    s.choice = types.SimpleNamespace(models={"target": types.SimpleNamespace(category="sunflower", model="dkajbh"),
                                             "partner": types.SimpleNamespace(category="vase", model="gopbrh")})
    s.gp = types.SimpleNamespace(variant=f"{INSIDE}-sunflower-vase-breakfast_table")
    s.world = types.SimpleNamespace(footprint=lambda h, pose: G.convex_hull_2d(G.local_points_world(h, pose)[:, :2]), hull_at=G.local_points_world)
    grasp = None if eef_in_object is None else types.SimpleNamespace(eef_in_object=eef_in_object)
    s.sidecar = types.SimpleNamespace(grasp_of=lambda name: grasp)
    s.bindings = types.SimpleNamespace(one=lambda role: types.SimpleNamespace(name=f"{role}_1"))
    s.counters, s.standing, s.shelf_start = {}, None, None
    return s


def test_sampler_stands_the_target_on_its_up_axis_at_the_goal():
    from ispatialgym.sampling.v2 import ik as IK

    s = _flower_sampler({"target_up_axis": UP_KEY})
    up, approach = th.tensor([1.0, 0.0, 0.0]), C.quat_to_mat(TOP_DOWN[1])[:, 2]
    quats = s.standing_quats()
    assert len(quats) == 4
    for k, q in enumerate(quats):
        assert G.rotate(q, up).tolist() == pytest.approx([0.0, 0.0, 1.0], abs=1e-6), "stem end down, head up"
        ext = TT.hull_extents(s.hull["target"], q)
        assert sorted(ext[:2].tolist()) == pytest.approx([0.0659, 0.0806], abs=1e-4) and float(ext[2]) == pytest.approx(0.194, abs=1e-4), "footprint along the world axes"
        assert math.degrees(G.quat_angle(quats[0], q)) == pytest.approx(90.0 * min(k, 4 - k), abs=1e-3)
    # the opening check of the template: the standing footprint passes, the lying flower does not
    assert s.pair_admissibility()["target_up_axis"] == "+x"
    with pytest.raises(Reject) as e:
        _flower_sampler({}).pair_admissibility()
    assert e.value.reason == "does_not_fit_inside"
    # the start lies with its head ahead of the robot: a quarter turn stands it up, the grasp approaching the place away from the robot
    q_ahead = s.target_goal_quat((th.zeros(3), G.quat_from_yaw(math.pi / 2)), FACING_Y)
    assert s.standing["turn_deg"] == pytest.approx(90.0, abs=0.1) and s.standing["up_axis"] == "+x"
    assert G.rotate(q_ahead, approach).tolist() == pytest.approx([0.0, 1.0, 0.0], abs=1e-5)
    # head toward the robot: the quarter turn would have the hand reach around the flower; the nearest turn approaching sideways
    q_back = s.standing_goal_quat(G.quat_from_yaw(-math.pi / 2), FACING_Y)
    assert s.standing["turn_deg"] == pytest.approx(120.0, abs=0.1) and s.standing["approach_ahead"] >= 0.0
    assert float(G.rotate(q_back, approach)[1]) > -1e-6
    # without a drawn grasp: the smallest turn
    blind = _flower_sampler({"target_up_axis": UP_KEY}, eef_in_object=None)
    blind.standing_goal_quat(G.quat_from_yaw(-math.pi / 2), FACING_Y)
    assert blind.standing["turn_deg"] == pytest.approx(90.0, abs=0.1) and blind.standing["approach_ahead"] is None
    # deterministic: the same start -> the same goal orientation, call after call and sampler after sampler
    for yaw in (0.3, 1.9, -2.6):
        q0 = G.quat_from_yaw(yaw)
        a = s.standing_goal_quat(q0, FACING_Y)
        assert th.equal(a, s.standing_goal_quat(q0, FACING_Y)) and th.equal(a, _flower_sampler({"target_up_axis": UP_KEY}).standing_goal_quat(q0, FACING_Y))
    # the descent into the release: straight down from 10 cm above (the approach is sideways at the standing place)
    place = (th.tensor([0.1, 0.2, 1.0]), q_ahead)
    pre = s.pre_place_of(place)
    assert pre[0].tolist() == pytest.approx([0.1, 0.2, 1.0 + IK.PRE_DIST]) and th.equal(pre[1], q_ahead)


def test_sampler_without_the_key_or_outside_inside_keeps_the_goal_as_before():
    from ispatialgym.sampling.v2 import ik as IK

    t_pose = (th.tensor([0.0, 0.3, 0.85]), G.quat_from_yaw(0.7))
    place = (th.tensor([0.1, 0.2, 1.0]), C.mat_to_quat(th.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])))
    for gen, relation in (({}, "inside"), ({"target_up_axis": {"sunflower/mdpoct": "+x"}}, "inside"), ({"target_up_axis": UP_KEY}, "ontop"),
                          ({"target_up_axis": UP_KEY}, "nextto")):
        s = _flower_sampler(gen, relation)
        assert s.up_axis() is None
        assert th.equal(s.target_goal_quat(t_pose, FACING_Y), G.t(t_pose[1])) and s.standing is None
        pre = s.pre_place_of(place)
        assert all(th.allclose(a, b) for a, b in zip(pre, IK.pre_pose(place)))
    with pytest.raises(Reject) as e:
        _flower_sampler({"target_up_axis": {"sunflower/dkajbh": "up"}}).up_axis()
    assert e.value.reason == "spec"


def test_sampler_release_pose_of_the_standing_target():
    s = _flower_sampler({"target_up_axis": UP_KEY})
    rim_z, centre = 0.973, th.tensor([0.3, 0.2, 0.9])
    s._container_volume = lambda obj, p_pose: (centre, rim_z)
    s.obj = lambda role: None
    p_pose = (th.tensor([0.3, 0.2, 0.8915]), C.IDENTITY_QUAT.clone())
    q = s.standing_goal_quat(G.quat_from_yaw(math.pi / 2), FACING_Y)
    pos, quat = s.release_pose(q, G.local_points_world(s.hull["target"], (th.zeros(3), q)), G.local_points_world(s.hull["partner"], p_pose), p_pose)
    hull_w = G.local_points_world(s.hull["target"], (pos, quat))
    footprint_centre = 0.5 * (hull_w[:, :2].min(dim=0).values + hull_w[:, :2].max(dim=0).values)
    assert th.equal(quat, q) and footprint_centre.tolist() == pytest.approx(centre[:2].tolist(), abs=1e-6), "the head centred on the opening"
    assert float(th.linalg.norm(pos[:2] - centre[:2])) == pytest.approx(math.hypot(0.0012, 0.00585), abs=1e-4), "the origin (stem) off it by the head's offset"
    assert float(hull_w[:, 2].min()) == pytest.approx(rim_z + 0.005, abs=1e-5), "the stem end 5 mm above the rim"
    assert s.release["drop_height_m"] == pytest.approx(0.005, abs=1e-4) and s.release["finger_clearance_m"] > 0.02
    assert s.release["standing"]["up_axis"] == "+x" and s.release["standing"]["turn_deg"] == pytest.approx(90.0, abs=0.1)
    # without the key the origin stays on the container centre (as before), also for a target whose footprint is off its origin
    lying = _flower_sampler({}, eef_in_object=None)
    lying._container_volume, lying.obj = s._container_volume, s.obj
    lying.hull["target"] = _box_at((-0.01, -0.01, 0.0), (0.03, 0.01, 0.02))
    q0 = C.IDENTITY_QUAT.clone()
    pos0, _ = lying.release_pose(q0, G.local_points_world(lying.hull["target"], (th.zeros(3), q0)), G.local_points_world(lying.hull["partner"], p_pose), p_pose)
    assert pos0[:2].tolist() == pytest.approx(centre[:2].tolist()) and lying.standing is None


#: A pinch from the object's -y side at its rest (approach +y): stood up, its approach stays horizontal, a quarter turn from TOP_DOWN's.
SIDE = (th.tensor([-0.04, -0.05, 0.0]), C.mat_to_quat(th.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])))


def test_after_settle_turns_the_standing_goal_to_a_reselected_grasp():
    """The sunflower rolls ~50 deg off its key's rest while settling, so ``after_settle`` re-selects its grasp; the standing goal's yaw
    follows the grasp's approach, so the goal is computed again with the re-selected grasp (here a 90 deg turn instead of 120)."""
    from ispatialgym.sampling.tasks.base import Draft
    from ispatialgym.sampling.v2 import ik as IK

    q_settled = G.quat_from_yaw(-math.pi / 2)
    s = _flower_sampler({"target_up_axis": UP_KEY})
    grasp = types.SimpleNamespace(eef_in_object=TOP_DOWN, grasp_key="sunflower__dkajbh", grasp_name="grasp_2", yaw=0.0, arm="right", extras={})
    p_pose = (th.tensor([0.3, 0.2, 0.8915]), C.IDENTITY_QUAT.clone())
    obj = {"partner": types.SimpleNamespace(get_position_orientation=lambda: p_pose), "target": types.SimpleNamespace(get_position_orientation=lambda: None)}
    s.obj, s.sidecar = obj.get, types.SimpleNamespace(grasp_of=lambda name: grasp, stance=lambda name: FACING_Y)
    s._container_volume = lambda o, pose: (th.tensor([0.3, 0.2, 0.9]), 0.973)
    s._pose_target = lambda role, pose: {"position": [float(v) for v in pose[0]], "orientation_metric": None}
    s.choice.models["target"].scale = None
    s.support_objects = lambda: []
    s.settle_motion = lambda draft, tname: {"rotation_deg": 50.0}
    s.ikc = types.SimpleNamespace(exact=lambda *a, **k: IK.IKResult(False, None, "the drawn grasp faces the table"))

    def reselect(draft, g, manip, tname, **kw):
        g.eef_in_object, g.grasp_name = SIDE, "grasp_side"
        return g.grasp_name

    s.reselect_grasp = reselect
    q_drawn = s.standing_goal_quat(q_settled, FACING_Y)
    assert s.standing["turn_deg"] == pytest.approx(120.0, abs=0.1)
    d = Draft()
    d.object_poses["target_1"] = (th.tensor([0.0, 0.3, 0.85]), q_settled)
    d.pose_targets["target_1"] = {"position": [0.0, 0.3, 0.85], "orientation_metric": None}
    s.after_settle(d)
    assert grasp.grasp_name == "grasp_side" and s.standing["turn_deg"] == pytest.approx(90.0, abs=0.1)
    assert not th.allclose(d.goal_poses["target_1"][1], q_drawn) and th.allclose(d.goal_poses["target_1"][1], s.standing_goal_quat(q_settled, FACING_Y))
    assert d.pose_targets["target_1"]["position"] == pytest.approx(d.goal_poses["target_1"][0].tolist())


def test_sampler_goal_tilt_of_a_standing_target_is_its_up_axis_lean():
    from ispatialgym.sampling.tasks.base import Draft
    from ispatialgym.sampling.v2.core import SamplerV2

    s = _flower_sampler({"target_up_axis": UP_KEY})
    q_stand = s.standing_quats()[0]
    pose = {"q": q_stand}
    obj = types.SimpleNamespace(get_position_orientation=lambda: (th.zeros(3), pose["q"]))
    s.obj, s.objects = (lambda role: obj), {"target_1": obj}

    def tilt(q):
        pose["q"] = q
        d = Draft()
        d.goal_poses["target_1"] = (th.zeros(3), q_stand)
        try:
            s.check_goal_upright(d)
        except Reject as e:
            return e.reason, d.summary
        return None, d.summary

    spun = G.qmul(G.quat_from_yaw(math.radians(60.0)), q_stand)
    assert tilt(spun)[0] is None and tilt(spun)[1]["goal_tilt_deg"]["target_1"] == pytest.approx(0.0, abs=1e-3)
    # the stock check follows the object's z axis, which the spin about the vertical turns 60 deg
    d = Draft()
    d.goal_poses["target_1"] = (th.zeros(3), q_stand)
    with pytest.raises(Reject):
        SamplerV2.check_goal_upright(s, d)
    s.counters.clear()
    lean = G.qmul(G.quat_about_axis([1.0, 0.0, 0.0], math.radians(20.0)), q_stand)  # leaning on the rim: the vase allows 30.3 deg
    reason, summary = tilt(lean)
    assert reason is None and summary["goal_tilt_deg"]["target_1"] == pytest.approx(20.0, abs=0.01) and summary["goal_tilt_max_deg"] == pytest.approx(30.29, abs=0.01)
    assert tilt(C.IDENTITY_QUAT.clone())[0] == "goal_tipped" and s.counters["goal_tipped"] == 1  # fallen over: lying again
