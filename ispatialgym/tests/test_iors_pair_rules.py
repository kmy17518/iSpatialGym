"""Owner rules 2026-10-11 of the IORS small-scale pairs (``sampling/pair_rules.py``): the grounding-time target fit scale (ontop /
under: the partner's top; inside: the opening) with its grasp-key route in ``ground_isg``, the nextto handedness that
``ground_isg`` writes and ``IORSSmall`` honours, and the standing inside target (``generation.target_up_axis``, inside#03: measured
and placed standing on that axis). No simulator; the grounder runs on a small sheet of the new representatives' models (the
shipped candidates / require sheets, a copy of ``grasps.csv`` the tests control)."""

import csv
import math
import os
import types

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import customs as CU
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import ground_isg as GI
from ispatialgym.sampling import iddl as I
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

SHEET_HEADER = ["family", "subtask", "pick", "role", "slot", "category", "model", "scene", "name", "joint", "kind", "grasp_key", "scale",
                "geometry_group", "exact_group", "status", "note", "triplet"]
SUPPORT = {"category": "breakfast_table", "scene": "house_single_floor", "name": "breakfast_table_rhjoby_0", "kind": "scene",
           "note": "init=ontop; top_z=0.81 area=3.38"}
#: (subtask, pick number, slot, target category/model/grasp key, partner category/model) - models of the 2026-10-11 sheet
PICKS = [
    (ONTOP, 1, "representative", ("bar_soap", "ofargb", "bar_soap__ofargb"), ("soap_dish", "slksbk")),
    (ONTOP, 1, "variation_geometry", ("bar_soap", "utgixp", "bar_soap__utgixp"), ("soap_dish", "gnndfy")),
    (ONTOP, 2, "representative", ("pack_of_kielbasa", "dgodnn", "pack_of_kielbasa__dgodnn__t"), ("platter", "csanbr")),
    (INSIDE, 1, "representative", ("sunflower", "dkajbh", "sunflower__dkajbh"), ("vase", "gopbrh")),
    (UNDER, 1, "representative", ("spatula", "hkvsep", "spatula__hkvsep__t"), ("paper_sheet", "uqbipw")),  # paper: asset bbox only
    (NEXTTO, 1, "representative", ("gym_shoe", "kmcbym", "gym_shoe__kmcbym__t"), ("gym_shoe", "mcfcwq")),
]
COMPAT = [(ONTOP, "bar_soap", "soap_dish"), (UNDER, "spatula", "paper_sheet"), (NEXTTO, "gym_shoe", "gym_shoe")]
SHOE_GRASP = {"key": "gym_shoe__kmcbym__t", "category": "gym_shoe", "model": "kmcbym", "require": "grasps:tabletop", "verdict": "ok"}


def _icl(root, grasp_rows=(), picks=None):
    """An icl root: the shipped sheets (symlinked) with ``picks`` (default :data:`PICKS`) as the selection, :data:`COMPAT` added to the
    compat sheet, and ``grasps.csv`` = the shipped rows without uniform-shrink keys of ofargb, plus :data:`SHOE_GRASP` and ``grasp_rows``."""
    for sub in ("roles", "require"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
        for name in os.listdir(os.path.join(GI.DEFAULT_ICL, sub)):
            if (sub, name) not in (("roles", "representatives.csv"), ("roles", "constraints_compatible.csv"), ("require", "grasps.csv")):
                dst = os.path.join(root, sub, name)
                if not os.path.lexists(dst):
                    os.symlink(os.path.join(GI.DEFAULT_ICL, sub, name), dst)
    with open(os.path.join(root, "roles", "representatives.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, SHEET_HEADER)
        w.writeheader()
        for st, n, slot, (tc, tm, tk), (pc, pm) in (PICKS if picks is None else picks):
            base = {"family": IORS, "subtask": st, "pick": f"{st}#{n:02d}", "slot": slot, "status": "untested"}
            w.writerow(dict(base, role="target", category=tc, model=tm, kind="sampleable", grasp_key=tk))
            w.writerow(dict(base, role="partner", category=pc, model=pm, kind="sampleable"))
            if slot == "representative":
                w.writerow(dict(base, role="support", **SUPPORT))
    with open(os.path.join(GI.DEFAULT_ICL, "roles", "constraints_compatible.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    hdr = list(rows[0])
    with open(os.path.join(root, "roles", "constraints_compatible.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, hdr)
        w.writeheader()
        w.writerows(rows)
        for st, a, b in COMPAT:
            w.writerow({"subtask": st, "relation": "partner_of", "role_a": "target", "category_a": a, "role_b": "partner", "category_b": b})
    with open(os.path.join(GI.DEFAULT_ICL, "require", "grasps.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    hdr = list(rows[0])

    def shrink_key(r):
        return r["category"] == "bar_soap" and r["model"] == "ofargb" and r.get("scale")

    with open(os.path.join(root, "require", "grasps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, hdr)
        w.writeheader()
        w.writerows(r for r in rows if not shrink_key(r))
        for r in (SHOE_GRASP, *grasp_rows):
            w.writerow({**dict.fromkeys(hdr, ""), **r})
    return root


def _ground(tmp_path, slots=("representative",), grasp_rows=(), subtasks=(ONTOP, UNDER, INSIDE, NEXTTO), picks=None):
    icl = GI.ICL(_icl(str(tmp_path / "icl"), grasp_rows, picks))
    customs_path = str(tmp_path / "isg_customs.json")
    g = GI.Grounder(I.load_all(), icl, CU.Customs.load(customs_path), bddl_root=str(tmp_path / "bddl"),
                    options=GI.Options(representative_test=False, slots=tuple(slots)))
    g.run(list(subtasks))
    return g


def _entry(g, subtask, target, partner):
    return g.customs.entries[f"{subtask}-{target}-{partner}-breakfast_table"]


def _scale_row(key, scale):
    return {"key": key, "category": "bar_soap", "model": "ofargb", "require": "grasps:tabletop", "verdict": "ok", "scale": json_list(scale)}


def json_list(v):
    return "[" + ", ".join(str(x) for x in v) + "]"


def test_grounding_withholds_a_bigger_target_until_a_key_at_its_scale_exists(tmp_path):
    g = _ground(tmp_path)
    soap = _entry(g, ONTOP, "bar_soap", "soap_dish")
    sc = soap.scenes["house_single_floor"]
    assert not sc.whitelist.get("target")
    assert sc.blacklist["target"]["bar_soap"]["ofargb"].split(": ", 1)[1].startswith("pending_scaled_grasps:tabletop (x0.70 to fit soap_dish/slksbk)")
    rec = soap.generation["target_fit"]["bar_soap/ofargb"]
    assert rec["status"] == "pending_scaled_grasps" and rec["scale"] == 0.70 and rec["partner"] == "soap_dish/slksbk" and rec["rule"] == "on_partner"
    assert rec["needs"] == {"pipeline": "tabletop", "base_key": "bar_soap__ofargb", "scale": [0.7, 0.7, 0.7]}
    assert "bar_soap/ofargb" not in soap.roles["target"].scale  # never spawned scaled with grasps of another scale
    # a target that fits is left alone
    kielbasa = _entry(g, ONTOP, "pack_of_kielbasa", "platter")
    assert kielbasa.scenes["house_single_floor"].whitelist["target"] == {"pack_of_kielbasa": ["dgodnn"]} and "target_fit" not in kielbasa.generation
    # inside: the lying sunflower needs x0.44 for the tube vase - below the floor, withheld with the sampler's reason
    flower = _entry(g, INSIDE, "sunflower", "vase")
    assert flower.generation["target_fit"]["sunflower/dkajbh"]["status"] == "below_floor"
    assert flower.generation["target_fit"]["sunflower/dkajbh"]["scale"] == 0.44
    assert "does_not_fit_inside (needs x0.44 < 0.50" in flower.scenes["house_single_floor"].blacklist["target"]["sunflower"]["dkajbh"]
    # under: the partner's box from the asset metadata (paper_sheet/uqbipw has no candidates row): the spatula needs x0.76
    spatula = _entry(g, UNDER, "spatula", "paper_sheet")
    assert spatula.generation["target_fit"]["spatula/hkvsep"]["scale"] == 0.76
    assert spatula.generation["target_fit"]["spatula/hkvsep"]["partner_footprint_m"] == [0.21, 0.29]


def test_grounding_scales_with_the_largest_key_within_the_fit_and_restores_it(tmp_path):
    rows = [_scale_row("bar_soap__ofargb__t072", [0.72, 0.72, 0.72]), _scale_row("bar_soap__ofargb__t065", [0.65, 0.65, 0.65]),
            _scale_row("bar_soap__ofargb__t040", [0.4, 0.4, 0.4])]
    g = _ground(tmp_path, grasp_rows=rows, subtasks=(ONTOP,))
    soap = _entry(g, ONTOP, "bar_soap", "soap_dish")
    assert soap.scenes["house_single_floor"].whitelist["target"] == {"bar_soap": ["ofargb"]}
    assert soap.roles["target"].grasp_keys["bar_soap/ofargb"] == "bar_soap__ofargb__t065"  # 0.72 does not fit, 0.40 is below the floor
    assert soap.roles["target"].scale["bar_soap/ofargb"] == [0.65, 0.65, 0.65]
    rec = soap.generation["target_fit"]["bar_soap/ofargb"]
    assert rec["status"] == "scaled" and rec["scale"] == 0.65 and rec["grasp_key"] == "bar_soap__ofargb__t065"
    g.customs.save()
    # the keys gone again (a re-grounding without them): the model's own key and scale come back, the model is withheld
    g2 = _ground(tmp_path, subtasks=(ONTOP,))
    soap = _entry(g2, ONTOP, "bar_soap", "soap_dish")
    assert soap.roles["target"].grasp_keys["bar_soap/ofargb"] == "bar_soap__ofargb" and "bar_soap/ofargb" not in soap.roles["target"].scale
    assert soap.generation["target_fit"]["bar_soap/ofargb"]["status"] == "pending_scaled_grasps"
    assert not soap.scenes["house_single_floor"].whitelist.get("target")


def test_grounding_never_shrinks_a_target_that_fits(tmp_path):
    """A target-fit key of a model (here one whose name sorts before the model's own key, so the grounder's default pick lands on
    it) never shrinks the model where it fits as is: the soap on the big held-out dish keeps its own key and scale."""
    picks = [(ONTOP, 1, "representative", ("bar_soap", "ofargb", "bar_soap__ofargb"), ("soap_dish", "ownvqj"))]
    g = _ground(tmp_path, grasp_rows=[_scale_row("bar_soap__aaa_fit060", [0.6, 0.6, 0.6])], subtasks=(ONTOP,), picks=picks)
    soap = _entry(g, ONTOP, "bar_soap", "soap_dish")
    assert soap.scenes["house_single_floor"].whitelist["target"] == {"bar_soap": ["ofargb"]} and "target_fit" not in soap.generation
    assert soap.roles["target"].grasp_keys["bar_soap/ofargb"] == "bar_soap__ofargb" and "bar_soap/ofargb" not in soap.roles["target"].scale


def test_grounding_fits_every_partner_of_the_alias_across_slots(tmp_path):
    """The sampler draws any whitelisted target with any whitelisted partner: with the variation pair grounded into the same
    alias the representative soap must fit the variation dish too (x0.58), and its record survives the later slot."""
    g = _ground(tmp_path, slots=("representative", "variation_geometry"), subtasks=(ONTOP,))
    soap = _entry(g, ONTOP, "bar_soap", "soap_dish")
    sc = soap.scenes["house_single_floor"]
    assert sc.whitelist["target"] == {"bar_soap": ["utgixp"]} and sorted(sc.whitelist["partner"]["soap_dish"]) == ["gnndfy", "slksbk"]
    rec = soap.generation["target_fit"]["bar_soap/ofargb"]
    assert rec["status"] == "pending_scaled_grasps" and rec["scale"] == 0.58 and rec["partner"] == "soap_dish/gnndfy"
    assert sc.blacklist["target"]["bar_soap"]["ofargb"].split(": ", 1)[1].startswith("pending_scaled_grasps")
    assert set(soap.generation["target_fit"]) == {"bar_soap/ofargb"}


def test_grounding_writes_the_hands_of_a_nextto_mirror_pair(tmp_path):
    g = _ground(tmp_path, subtasks=(NEXTTO,))
    shoes = g.customs.entries[f"{NEXTTO}-gym_shoe-gym_shoe-breakfast_table"]
    assert shoes.scenes["house_single_floor"].whitelist == {"target": {"gym_shoe": ["kmcbym"]}, "partner": {"gym_shoe": ["mcfcwq"]}}
    assert shoes.generation["handedness"] == {"gym_shoe/kmcbym": "right", "gym_shoe/mcfcwq": "left"}
    for (cat, m), hand in GI.HANDEDNESS.items():
        assert cat == "gym_shoe" and hand in PR.HANDS
    # every nextto#04 pair of the sheet is a right target with a left partner
    assert {GI.HANDEDNESS[("gym_shoe", m)] for m in ("kmcbym", "picdro", "pblxfp")} == {"right"}
    assert {GI.HANDEDNESS[("gym_shoe", m)] for m in ("mcfcwq", "ugbxip", "rcozpq")} == {"left"}


def test_grounding_measures_a_standing_target_on_its_up_axis(tmp_path):
    """inside#03: without ``generation.target_up_axis`` the lying sunflower is withheld (x0.44); the key written into the grounded
    entry (``generation`` is seeded from it) makes the re-grounding measure it standing - it fits as is, own key, unit scale. The key
    names models one by one, and an ontop alias does not read it."""
    g = _ground(tmp_path, subtasks=(ONTOP, INSIDE))
    flower, soap = _entry(g, INSIDE, "sunflower", "vase"), _entry(g, ONTOP, "bar_soap", "soap_dish")
    rec = flower.generation["target_fit"]["sunflower/dkajbh"]
    assert rec["status"] == "below_floor" and "up_axis" not in rec and "target_up_axis" not in flower.generation
    flower.generation["target_up_axis"] = {"sunflower/mdpoct": "+x"}  # another model's axis: dkajbh stays lying
    soap.generation["target_up_axis"] = {"bar_soap/ofargb": "+x"}
    g.customs.save()
    g = _ground(tmp_path, subtasks=(ONTOP, INSIDE))
    flower, soap = _entry(g, INSIDE, "sunflower", "vase"), _entry(g, ONTOP, "bar_soap", "soap_dish")
    assert flower.generation["target_fit"]["sunflower/dkajbh"]["status"] == "below_floor"
    assert soap.generation["target_fit"]["bar_soap/ofargb"]["status"] == "pending_scaled_grasps" and soap.generation["target_fit"]["bar_soap/ofargb"]["scale"] == 0.70
    flower.generation["target_up_axis"] = dict(UP_KEY)
    g.customs.save()
    g = _ground(tmp_path, subtasks=(INSIDE,))
    flower = _entry(g, INSIDE, "sunflower", "vase")
    sc = flower.scenes["house_single_floor"]
    assert sc.whitelist["target"] == {"sunflower": ["dkajbh"]} and "dkajbh" not in ((sc.blacklist.get("target") or {}).get("sunflower") or {})
    assert "target_fit" not in flower.generation and flower.generation["target_up_axis"] == UP_KEY
    assert flower.roles["target"].grasp_keys["sunflower/dkajbh"] == "sunflower__dkajbh" and "sunflower/dkajbh" not in flower.roles["target"].scale
    assert f"{INSIDE}-sunflower-vase-breakfast_table: target fit sunflower/dkajbh standing on +x (generation.target_up_axis): fits as is" in g.report.notes[INSIDE]
    flower.generation["target_up_axis"] = {"sunflower/dkajbh": "up"}  # no axis name: the combination is skipped with the reason
    g.customs.save()
    g = _ground(tmp_path, subtasks=(INSIDE,))
    assert any("generation.target_up_axis of sunflower/dkajbh: 'up' is not a model axis" in why for _, why in g.report.skipped[INSIDE])


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
    right = types.SimpleNamespace(pair={"handedness": {"target": "right", "partner": "left", "target_side": "right"}})
    assert IS.IORSSmall.handed_ranges(right, {1.0: (0.05, 0.2), -1.0: (0.05, 0.3)}) == {1.0: (0.05, 0.2)}
    left = types.SimpleNamespace(pair={"handedness": {"target": "left", "partner": "right", "target_side": "left"}})
    assert IS.IORSSmall.handed_ranges(left, {1.0: (0.05, 0.2), -1.0: (0.05, 0.3)}) == {-1.0: (0.05, 0.3)}
    with pytest.raises(Reject) as e:
        IS.IORSSmall.handed_ranges(right, {-1.0: (0.05, 0.3)})
    assert e.value.reason == "partner_placement"
    both = {1.0: (0.0, 1.0), -1.0: (0.0, 1.0)}
    assert IS.IORSSmall.handed_ranges(types.SimpleNamespace(pair={"g_NT": 0.05}), both) == both


def _box_at(lo, hi):
    return th.tensor([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


#: A top-down stem pinch at the sunflower's rest (approach -z, closing -y of the object) and the manipulation stance facing +y.
TOP_DOWN = (th.tensor([-0.04, 0.0, 0.05]), C.mat_to_quat(th.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])))
FACING_Y = Stance("manipulation", (0.0, -0.6, math.pi / 2), 0.5, 0.0, 0.45, None)


def _flower_sampler(gen, relation="inside", eef_in_object=TOP_DOWN):
    """An ``IORSSmall`` without a simulator: the sunflower's collision boxes (stem x -0.146..-0.03, head x -0.03..0.048), the tube
    vase, a world that only projects hulls, ``gen`` as the alias's generation."""
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    s = IS.IORSSmall.__new__(IS.IORSSmall)
    head = _box_at((-0.03, -0.0415, -0.0271), (0.048, 0.0391, 0.0388))
    s.relation, s.hull = relation, {"target": th.cat([_box_at((-0.146, -0.005, -0.005), (-0.03, 0.005, 0.005)), head]), "partner": _box(TUBE_VASE)}
    s.rest = {"target": C.IDENTITY_QUAT.clone(), "partner": C.IDENTITY_QUAT.clone()}
    s.spec = types.SimpleNamespace(generation=gen)
    s.choice = types.SimpleNamespace(models={"target": types.SimpleNamespace(category="sunflower", model="dkajbh"),
                                             "partner": types.SimpleNamespace(category="vase", model="gopbrh")})
    s.gp = types.SimpleNamespace(alias=f"{INSIDE}-sunflower-vase-breakfast_table")
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
