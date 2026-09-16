"""Simulator-free tests of the v2 sampler core: the §9 model draw, the §3.8 stand-off enumeration, the §7 path record
schema and the §3.5 summary layout (``pytest ispatialgym/tests/test_v2_core.py``)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.v2 import navigation as NAV
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.sampling.v2.problem import (
    GroundedProblem,
    LookalikeIndex,
    ModelCycler,
    RejectionLog,
    TemplateChoice,
    draw_group,
    iter_template_choices,
    new_run_summary,
    subtask_of_alias,
    template_summary,
)
from ispatialgym.sampling.v2.stances import EdgeFrame, GraspRecord, PathRecord, SidecarDraft, Stance, Transition, path_length


def _entry(target_models, partner_models, groups=None, templates_max=20):
    return {
        "iddl": "inter_object_relation_satisfaction/ontop_small_scale.iddl",
        "definition_id": 0,
        "roles": {
            "target": {"category": "apple", "sampleable": True, "require": {"grasps": "tabletop"}, "representative": target_models[0]},
            "partner": {"category": "plate", "sampleable": True, "representative": partner_models[0]},
            "support": {"category": "breakfast_table", "sampleable": False},
            "floor": {"sampleable": False},
        },
        "scenes": {
            "house_single_floor": {
                "rooms": ["dining_room_0"],
                "bindings": {"support": ["breakfast_table_rhjoby_0"], "floor": {"room": "dining_room_0"}},
                "whitelist": {"target": {"apple": list(target_models)}, "partner": {"plate": list(partner_models)}},
                "blacklist": {},
            }
        },
        "generation": {"templates_max": templates_max, "instances_per_template": 1, "tolerances": {"position": 0.03, "orientation": 0.2},
                       "groups": groups or {}, "path_buckets": None},
        "constraints": [],
    }


def _problem(entry, lookalikes=None):
    return GroundedProblem("inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table", entry, lookalikes=lookalikes)


# ----------------------------------------------------------------------------------------------------------------
# model draw (§9, §3.12)
# ----------------------------------------------------------------------------------------------------------------


def test_subtask_of_alias():
    assert subtask_of_alias("inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table") == "inter_object_relation_satisfaction-ontop_small_scale"
    assert subtask_of_alias("configuration_matching-block_design-block_4-breakfast_table") == "configuration_matching-block_design"
    assert subtask_of_alias("x", "inter_object_relation_satisfaction/nextto.iddl") == "inter_object_relation_satisfaction-nextto"


def test_cycler_iterates_all_models_before_reuse():
    rng = np.random.default_rng(0)
    cy = ModelCycler("target", {"apple": ["a", "b", "c"]}, rng, LookalikeIndex())
    seen = [cy.next().model for _ in range(6)]
    assert sorted(seen[:3]) == ["a", "b", "c"]
    assert sorted(seen[3:]) == ["a", "b", "c"]
    assert cy.n_passes == 2


def test_cycler_counts_exact_duplicates_once_and_variants_separately():
    la = LookalikeIndex(duplicate_of={("apple", "b"): ("apple", "a")}, geometry_group={("apple", "c"): "g1", ("apple", "d"): "g1"})
    cy = ModelCycler("target", {"apple": ["a", "b", "c", "d"]}, np.random.default_rng(0), la)
    assert len(cy) == 3  # a (b is its exact duplicate), c, d (texture variants stay distinct)
    assert {m for _, m in cy.pool} == {"a", "c", "d"}


def test_templates_unique_across_roles_and_no_lookalike_pairs():
    # apple/x and plate/x share a look-alike group: a template must never pair them
    la = LookalikeIndex(groups=[{("apple", "x"), ("plate", "x")}])
    gp = _problem(_entry(["x", "y"], ["x", "z"]), la)
    rng = np.random.default_rng(1)
    templates = list(iter_template_choices(gp, "house_single_floor", rng))
    assert templates, "no templates drawn"
    for t in templates:
        assert not la.lookalikes(("apple", t.models["target"].model), ("plate", t.models["partner"].model))
        canon = {la.canonical(c.category, c.model) for c in t.models.values()}
        assert len(canon) == len(t.models)
    ids = [t.template_id for t in templates]
    assert len(ids) == len(set(ids))
    assert len(templates) <= 4


def test_templates_respect_templates_max_and_exclude():
    gp = _problem(_entry(["a", "b", "c"], ["p", "q"], templates_max=3))
    rng = np.random.default_rng(2)
    templates = list(iter_template_choices(gp, "house_single_floor", rng))
    assert len(templates) == 3
    excl = [{"target": f"apple/{t.models['target'].model}", "partner": f"plate/{t.models['partner'].model}"} for t in templates[:1]]
    more = list(iter_template_choices(gp, "house_single_floor", np.random.default_rng(2), exclude=excl))
    keys = {(t.models["target"].model, t.models["partner"].model) for t in more}
    assert (templates[0].models["target"].model, templates[0].models["partner"].model) not in keys
    assert len(more) == 2


def test_blacklist_prunes_whitelist_and_hard_records():
    gp = _problem(_entry(["a", "b"], ["p"]))
    gp.add_blacklist("house_single_floor", "target", "apple", "a", "sample_isg: grasp_unreachable")
    assert gp.whitelist("house_single_floor", "target") == {"apple": ["b"]}
    assert gp.blacklist("house_single_floor", "target")["apple"]["a"].startswith("sample_isg")
    gp.add_constraint({"type": "blacklist", "scope": {"scene": "house_single_floor", "target": "apple/b"}, "hard": True})
    assert list(iter_template_choices(gp, "house_single_floor", np.random.default_rng(0))) == []


def test_group_draw_same_model_and_unique():
    entry = _entry(["a"], ["p"], groups={"targets": {"size": 4, "draw": "same_model"}, "distractors": {"size": [1, 3], "draw": "unique"}})
    entry["roles"]["targets"] = {"category": "block", "sampleable": True}
    entry["roles"]["distractors"] = {"category": "beaker", "sampleable": True}
    entry["scenes"]["house_single_floor"]["whitelist"]["targets"] = {"block": ["k", "l"]}
    entry["scenes"]["house_single_floor"]["whitelist"]["distractors"] = {"beaker": ["d1", "d2", "d3", "d4"]}
    la = LookalikeIndex(groups=[{("beaker", "d1"), ("beaker", "d2")}])
    gp = _problem(entry, la)
    assert gp.group_roles() == ["targets", "distractors"]
    rng = np.random.default_rng(3)
    blocks = draw_group(gp, "targets", "house_single_floor", rng, set())
    assert len(blocks) == 4 and len({b.model for b in blocks}) == 1
    for _ in range(20):
        dis = draw_group(gp, "distractors", "house_single_floor", rng, set())
        assert dis is not None and 1 <= len(dis) <= 3
        models = [d.model for d in dis]
        assert len(set(models)) == len(models)
        assert not ({"d1", "d2"} <= set(models)), "two members of one look-alike group"
    t = next(iter_template_choices(gp, "house_single_floor", rng))
    assert len(t.groups["targets"]) == 4 and "targets-block_4" in t.template_id
    assert "targets[0]" in t.choices


def test_group_members_per_category_follow_the_alias():
    # D1 mixed-category groups: ``members`` fixes the count per category (the grounded BDDL declares exactly those)
    entry = _entry(["a"], ["p"], groups={"distractors": {"size": 3, "draw": "unique", "members": {"bottle_of_beer": 2, "bottle_of_sake": 1}}})
    entry["roles"]["distractors"] = {"category": ["bottle_of_beer", "bottle_of_sake"], "sampleable": True}
    entry["scenes"]["house_single_floor"]["whitelist"]["distractors"] = {"bottle_of_beer": ["b1", "b2", "b3"], "bottle_of_sake": ["s1", "s2"]}
    gp = _problem(entry)
    assert gp.roles["distractors"].categories == ["bottle_of_beer", "bottle_of_sake"]
    rng = np.random.default_rng(4)
    for _ in range(10):
        members = draw_group(gp, "distractors", "house_single_floor", rng, set())
        assert [m.category for m in members] == ["bottle_of_beer", "bottle_of_beer", "bottle_of_sake"]
        assert len({m.model for m in members}) == 3
    t = next(iter_template_choices(gp, "house_single_floor", rng))
    assert "distractors-bottle_of_beer_2-bottle_of_sake_1" in t.template_id
    assert subtask_of_alias("configuration_matching-articulation_open_large_scale-bottom_cabinet") == "configuration_matching-articulation_open_large_scale"


def test_stance_sidecar_keeps_extras_and_recorded_configuration():
    edge = EdgeFrame.from_edge("+u", [1.0, 0.0], [1.0, 0.0], 1.6)
    st = Stance("init", (1.45, 0.0, math.pi), 0.5, 0.0, 0.45, edge, arms="reset", carried="bowl.n.01_1",
                extras={"goal_cell": [0.3, -0.1], "bearing_deg": 12.0, "_private": 1})
    d = st.to_sidecar()
    assert d["goal_cell"] == [0.3, -0.1] and d["bearing_deg"] == 12.0 and "_private" not in d
    assert d["arms"] == "reset" and d["carried"] == "bowl.n.01_1" and d["edge"] == "+u"
    assert st.posture == "tucked" or st.posture == "reset"  # the reset arms map to the hanging (zero) posture
    st2 = st.with_trunk(0.75, -0.3)
    assert st2.carried == "bowl.n.01_1" and st2.fingers is None and st2.extras["goal_cell"] == [0.3, -0.1]


def test_constraint_records_and_scope_query():
    gp = _problem(_entry(["a"], ["p"]))
    rec = gp.add_constraint({"type": "stance", "scope": {"scene": "house_single_floor", "support": "breakfast_table_rhjoby_0"}, "edges": ["+u"]},
                            source_instance="iid_0001", sampler_version="2.0.0-dev")
    assert rec["source"]["instance"] == "iid_0001" and rec["hard"] is False and "date" in rec["source"]
    assert gp.constraints_for("stance", scene="house_single_floor") == [rec]
    assert gp.constraints_for("stance", scene="Rs_int") == []
    with pytest.raises(Exception):
        gp.add_constraint({"scope": {}})


def test_v1_facade_roles_and_scene_view():
    gp = _problem(_entry(["a"], ["p"]))
    assert gp.activity == "inter_object_relation_satisfaction" and gp.name == gp.alias
    assert {r.name for r in gp.spawned_roles()} == {"target", "partner"}
    assert {r.name for r in gp.fixed_roles()} == {"support", "floor"}
    ss = gp.scenes["house_single_floor"]
    assert ss.rooms == ["dining_room_0"] and ss.fixed["support"] == ["breakfast_table_rhjoby_0"]
    assert gp.gen("instances_per_template") == 1
    tc = TemplateChoice.from_json(json.loads(json.dumps(next(iter_template_choices(gp, "house_single_floor", np.random.default_rng(0))).to_json())))
    assert tc.template_id == "house_single_floor__partner-plate-p__target-apple-a"


# ----------------------------------------------------------------------------------------------------------------
# transitions (§3.8), paths (§3.7)
# ----------------------------------------------------------------------------------------------------------------


def test_transition_standoff_enumeration():
    assert NAV.transition_standoffs(0.45) == [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    assert NAV.transition_standoffs(0.35)[0] == 0.35 and len(NAV.transition_standoffs(0.35)) == 6
    sweep = NAV.trunk_sweep((0.5, 0.0), (0.75, -0.4), 5)
    assert len(sweep) == 5 and sweep[0] == (0.5, 0.0) and sweep[-1] == pytest.approx((0.75, -0.4))
    assert Transition("init", "manipulation", 0.55).to_sidecar() == {"from": "init", "to": "manipulation", "stand_off": 0.55,
                                                                     "order": ["drive-back", "trunk", "drive-in"]}


def test_stride_poses_and_grid_path():
    poses = NAV.stride_poses([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 0.1)
    assert len(poses) == 11 and poses[-1] == (1.0, 0.0, 0.0)
    free = th.ones(10, 10, dtype=th.bool)
    free[:, 5] = False
    free[0, 5] = True
    path = NAV.grid_shortest_path(free, (9, 0), (9, 9))
    assert path is not None and path[0] == (9, 0) and path[-1] == (9, 9) and (0, 5) in path
    assert NAV.same_region(free, (9, 0), (9, 9))
    free[0, 5] = False
    assert NAV.grid_shortest_path(free, (9, 0), (9, 9)) is None
    assert not NAV.same_region(free, (9, 0), (9, 9))
    simple = NAV.simplify_path([(0, 0), (1, 1), (2, 2), (3, 3)], th.ones(4, 4, dtype=th.bool))
    assert simple == [(0, 0), (3, 3)]


def test_path_record_schema():
    p = PathRecord("init", "manipulation", "local", [(0.0, 0.0, 0.0), (0.3, 0.4, 0.1)], path_length([(0, 0), (0.3, 0.4)]), "untucked", None, (0.5, 0.0))
    d = p.to_sidecar()
    assert PathRecord.schema_ok(d) and d["length_m"] == pytest.approx(0.5) and d["carried"] is None and d["mode"] == "local"
    assert set(d) == {"from", "to", "mode", "waypoints", "length_m", "posture", "carried"}
    assert not PathRecord.schema_ok({**d, "mode": "teleport"})
    assert not PathRecord.schema_ok({k: v for k, v in d.items() if k != "carried"})


# ----------------------------------------------------------------------------------------------------------------
# stances / sidecar (§7)
# ----------------------------------------------------------------------------------------------------------------


def test_edge_frame_and_stance_local_world_roundtrip():
    edge = EdgeFrame.from_edge("+u", mid=[1.0, 0.0], outward_normal=[1.0, 0.0], length=1.6, overhang=0.0)
    assert th.allclose(edge.inward, th.tensor([-1.0, 0.0])) and th.allclose(edge.tangent, th.tensor([0.0, -1.0]))
    xy = edge.base_xy(0.45, 0.2)
    st = Stance("init", (float(xy[0]), float(xy[1]), edge.yaw), 0.5, 0.0, 0.45, edge, lateral=0.2)
    assert st.edge_depth == pytest.approx(0.45)
    depth, lat = st.local(edge.mid + edge.inward * 0.3 + edge.tangent * 0.1)
    assert depth == pytest.approx(0.3) and lat == pytest.approx(0.1 - 0.2)
    back = st.world(depth, lat)
    assert th.allclose(back, edge.mid + edge.inward * 0.3 + edge.tangent * 0.1, atol=1e-6)
    manip = st.shifted(stand_off=0.35, name="manipulation").with_trunk(0.75, -0.4, "manipulation")
    assert manip.stand_off == 0.35 and manip.translate == 0.75 and manip.lateral == 0.2
    assert manip.to_sidecar()["edge"] == "+u" and manip.to_sidecar()["tilt"] == -0.4


def test_sidecar_draft_layout(tmp_path):
    sd = SidecarDraft("alias-x", "inter_object_relation_satisfaction")
    edge = EdgeFrame.from_edge("+u", [1.0, 0.0], [1.0, 0.0], 1.6)
    init = Stance("init", (1.45, 0.0, math.pi), 0.5, 0.0, 0.45, edge)
    manip = Stance("manipulation", (1.35, 0.0, math.pi), 0.75, -0.4, 0.35, edge, focus=th.tensor([0.7, 0.0, 0.83]), arms=["right"])
    sd.add_stance(init)
    sd.add_stance(manip)
    sd.add_transition(Transition("init", "manipulation", 0.55))
    pose = (th.tensor([0.7, 0.0, 0.95]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    sd.add_grasp(GraspRecord("apple.n.01_1", "right", "apple__a__t", "grasp_1", 0.3, pose, pose, pose, pose))
    sd.set_goal(manip.base_pose2d, 0.75, -0.4)
    d = sd.to_dict()
    assert d["schema"] == "isg-oracle-v1" and set(d["stances"]) == {"init", "manipulation"}
    assert d["stances"]["manipulation"]["focus"] == pytest.approx([0.7, 0.0, 0.83]) and d["stances"]["init"]["arms"] == "untucked"
    assert d["grasps"][0]["grasp"]["position"] == pytest.approx([0.7, 0.0, 0.95])
    assert d["waypoint_policy"] == "checked" and "contacts" not in d and "paths" not in d
    assert PC.check_blocks("inter_object_relation_satisfaction-ontop_small_scale", d) == []
    problems = PC.check_blocks("inter_object_relation_satisfaction-touching", d)
    assert any("contacts.push" in p for p in problems) and any("contacts.brace" in p for p in problems)
    tro = tmp_path / "x-tro_state.json"
    tro.write_text("{}")
    out = sd.finalize("iid", str(tro))
    assert out["tro_sha256"] == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a" and out["instance_id"] == "iid"
    sd.reset_instance()
    assert sd.stances == {} and sd.grasps == [] and sd.tro_sha256 is None


def test_program_check_target_extraction():
    pose = {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    targets = {"pre_grasp": pose, "approach": [pose, pose, pose], "base_pose2d": [0.0, 0.0, 0.0], "nested": {"place": pose}}
    found = dict(PC.iter_pose_targets(targets))
    assert set(found) == {"pre_grasp", "approach", "nested.place"} and len(found["approach"]) == 3


def test_program_check_ignore_sets_mirror_the_sampler():
    """Large-scale A4/A5/A8: a brace approach runs along the wall -> the supports are removed for every waypoint (as for a
    push), the braced object only at the contact; explicit ``ignore_every`` / ``ignore_last`` on the record win."""

    class Obj:
        def __init__(self, name):
            self.name = name

    wall, target, partner = Obj("wall"), Obj("painting"), Obj("can")

    class Sampler:
        objects = {"painting": target, "can": partner}

        @staticmethod
        def support_objects():
            return [wall]

    class Contact:
        def __init__(self, obj, extra=None):
            self.object = obj
            self.extra = extra or {}

    class Action:
        def __init__(self, name, contact):
            self.name, self.contact = name, contact

    last, every = PC._ignore_for(Action("brace", Contact("painting")), Sampler, None)
    assert every == [wall] and last == [target]
    last, every = PC._ignore_for(Action("push", Contact("painting")), Sampler, None)
    assert target in every and wall in every and last == []
    # a module that recorded what it ignored while solving (names or the token "supports")
    last, every = PC._ignore_for(Action("brace", Contact("painting", {"ignore_every": ["supports", "can"], "ignore_last": []})), Sampler, None)
    assert every == [wall, partner] and last == []


def test_transition_resweep_never_lowers_the_recorded_standoff():
    """The re-sweep with the objects placed starts at the recorded s_T; the path's end gives the extra drive-back start."""
    from ispatialgym.sampling.v2 import navigation as NAV
    from ispatialgym.sampling.v2.core import SamplerV2

    assert NAV.transition_standoffs(0.4) == [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    edge = EdgeFrame("south", th.tensor([1.5, 1.0]), th.tensor([0.0, 1.0]), 2.0)  # inward = +y: the robot stands at y < 1 facing +y
    to = Stance("manipulation", (1.5, 0.5, math.pi / 2), 0.5, 0.0, 0.5, edge=edge)  # base 0.5 m in front of the edge

    class Fake:
        sidecar = SidecarDraft("alias", "alignment")

    # no path recorded: the path ends at the stance -> s
    assert SamplerV2.path_end_standoff(Fake, Stance("init", (0.0, 0.0, 0.0), 0.5), to) == 0.5
    # prompt0.connect planned the path to the drive-back pose 0.2 m behind the stance (along -inward = -y)
    Fake.sidecar.add_path(PathRecord("init", "manipulation", "local", [(0.0, 0.0, 0.0), (1.5, 0.3, math.pi / 2)], 1.0, "untucked", None))
    assert abs(SamplerV2.path_end_standoff(Fake, Stance("init", (0.0, 0.0, 0.0), 0.5), to) - 0.7) < 1e-6


# ----------------------------------------------------------------------------------------------------------------
# CLIs (§9): path buckets, whitelist pruning after a failed template, exclusion of produced templates
# ----------------------------------------------------------------------------------------------------------------


def test_multiply_path_buckets():
    from ispatialgym.sampling.multiply_isg import parse_path_buckets

    assert parse_path_buckets("3:10,5:10,10:5") == [{"max_m": 3.0, "n": 10}, {"max_m": 5.0, "n": 10}, {"max_m": 10.0, "n": 5}]
    assert parse_path_buckets(None) is None


def test_sample_isg_blacklist_after_failure_and_exclusion(tmp_path):
    from ispatialgym.sampling.sample_isg import _existing_templates, blacklist_after_failure

    gp = _problem(_entry(["a", "b"], ["p", "q"]))
    choice = next(iter_template_choices(gp, "house_single_floor", np.random.default_rng(0)))
    # a grasp failure blames the target only; a stance failure blames every sampleable model of the pair
    out = blacklist_after_failure(gp, choice, {"rejections": {"grasp_unreachable": 5, "no_transition": 1}})
    assert out == [f"target=apple/{choice.models['target'].model}"]
    assert gp.blacklist("house_single_floor", "target")["apple"][choice.models["target"].model].endswith("grasp_unreachable")
    out2 = blacklist_after_failure(gp, choice, {"rejections": {}, "reject_reason": "no_free_edge"})
    assert len(out2) == 2 and gp.whitelist("house_single_floor", "partner") == {"plate": [m for m in ["p", "q"] if m != choice.models["partner"].model]}
    # templates already on disk are excluded from the draw
    d = tmp_path / gp.alias / choice.template_id
    d.mkdir(parents=True)
    (d / "template_meta.json").write_text(json.dumps({"scene": "house_single_floor", "choice": choice.to_json()}))
    excl = _existing_templates(str(tmp_path), gp, "house_single_floor")
    assert excl == [{r: f"{c.category}/{c.model}" for r, c in choice.models.items()}]
    gp2 = _problem(_entry(["a", "b"], ["p", "q"]))
    rest = list(iter_template_choices(gp2, "house_single_floor", np.random.default_rng(0), exclude=excl))
    assert choice.template_id not in {t.template_id for t in rest} and len(rest) == 3


# ----------------------------------------------------------------------------------------------------------------
# summary layout (§3.5)
# ----------------------------------------------------------------------------------------------------------------


def test_a1_joint_band_gate_with_a_synthetic_band():
    from ispatialgym.sampling.v2 import stances as IS

    right = np.array([[x, lat] for x in (0.5, 0.58, 0.66) for lat in (-0.3, -0.22, -0.14)])
    left = np.array([[x, lat] for x in (0.5, 0.58, 0.66) for lat in (0.18, 0.26, 0.34)])
    assert IS.in_band((0.6, -0.2), right) and not IS.in_band((0.6, 0.2), right) and not IS.in_band((0.6, -0.2), np.zeros((0, 2)))
    assert IS.brace_placement_admissible((0.6, -0.18), (0.6, 0.3), right, left)
    assert not IS.brace_placement_admissible((0.6, -0.18), (0.6, 0.6), right, left)  # brace point outside the left band
    assert not IS.brace_placement_admissible((0.9, -0.18), (0.6, 0.3), right, left)  # work point outside the right band


def test_nextto_goal_takes_the_partner_side_facing_the_init():
    """Owner review 2026-09-16: jar (target) at lateral 0.00, can (partner) at -0.16 (robot's right) -> the goal is on the
    can's left (facing the jar), not lifted over it to -0.28."""
    from ispatialgym.sampling.v2.stances import nearer_side_lateral

    partner = (-0.16 - 0.04, -0.16 + 0.04)  # 8 cm wide can centred at -0.16
    target = (-0.04, 0.04)  # 8 cm wide jar about its origin
    side, lat = nearer_side_lateral(0.0, partner, target, gap=0.02)
    assert side == "left" and abs(lat - (-0.12 + 0.02 + 0.04)) < 1e-9  # near face 2 cm off the can's left face, at -0.06
    side, lat = nearer_side_lateral(-0.40, partner, target, gap=0.02)
    assert side == "right" and abs(lat - (-0.20 - 0.02 - 0.04)) < 1e-9
    # touching (gap 0): partner on the robot's left of the target -> the target goes on the partner's right side
    side, lat = nearer_side_lateral(0.0, (0.10, 0.18), target, gap=0.0)
    assert side == "right" and abs(lat - (0.10 - 0.04)) < 1e-9


def test_open_gripper_lowest_point_for_top_down_and_side_grasps():
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.v2.tasks.iors_small import FINGER_THICKNESS, gripper_corners_world

    # top-down: approach a = -z -> the fingertips are the lowest points, FINGERTIP_LENGTH below the eef origin
    q_down = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0]))
    low = float(gripper_corners_world((th.tensor([0.0, 0.0, 1.0]), q_down))[:, 2].min())
    assert abs(low - (1.0 - C.FINGERTIP_LENGTH)) < 1e-6
    # side grasp with a vertical closing axis: the lower open finger is (open / 2 + thickness) below the origin
    q_side = C.eef_quat(th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 1.0]))
    low = float(gripper_corners_world((th.tensor([0.0, 0.0, 1.0]), q_side))[:, 2].min())
    assert abs(low - (1.0 - 0.5 * C.GRIPPER_OPEN_M - FINGER_THICKNESS)) < 1e-6
    # grasp record extras reach the sidecar block
    g = GraspRecord("obj", "right", "k", "n", 0.0, grasp=(th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), extras={"place_kind": "drop", "rim_z": 0.8})
    d = g.to_sidecar()
    assert d["place_kind"] == "drop" and d["rim_z"] == 0.8


def test_stance_idle_posture_and_world_key_string():
    from ispatialgym.sampling.sample_isg import world_key_str

    edge = EdgeFrame.from_edge("+u", [1.0, 0.0], [1.0, 0.0], 1.6)
    manip = Stance("manipulation", (1.35, 0.0, math.pi), 0.75, -0.4, 0.35, edge, arms=["right"])
    assert manip.working_arms == ["right"] and manip.idle_arm == "left" and manip.idle_posture is None
    assert manip.to_sidecar()["idle_posture"] == "untucked"  # the default, always written for a one-arm stance
    manip.idle_posture = "ready"
    d = manip.with_trunk(0.5, -0.2).to_sidecar()
    assert d["idle_posture"] == "ready" and d["arms"] == ["right"]
    init = Stance("init", (1.45, 0.0, math.pi), 0.5, 0.0, 0.45, edge)
    assert init.idle_arm is None and "idle_posture" not in init.to_sidecar()
    assert world_key_str(("house_single_floor", ("dining_room_0", "living_room_0"), ("chair_a", "chair_b"))) == \
        "house_single_floor|rooms=dining_room_0,living_room_0|removed=chair_a,chair_b"


def test_rejection_log_and_summary_layout():
    rl = RejectionLog(keep=3)
    for i in range(5):
        rl.add("grasp_unreachable", f"attempt {i}", index=0, attempt=i)
    rl.add("no_transition", "init -> manipulation")
    d = rl.to_json()
    assert d["counts"] == {"grasp_unreachable": 5, "no_transition": 1} and len(d["examples"]["grasp_unreachable"]) == 3
    assert d["examples"]["grasp_unreachable"][0]["attempt"] == 0 and rl.total == 6
    entry = template_summary(2, 123.456, rl, seeds=[7, 8], extra={"counters": {"scene_reset": 12}})
    assert entry["n_written"] == 2 and entry["time_s"] == 123.46 and entry["rejections"]["no_transition"] == 1 and entry["seeds"] == [7, 8]
    assert entry["counters"] == {"scene_reset": 12} and "scene_reset" not in entry["rejections"]
    rl.counts["scene_reset"] = 3  # a legacy log that still counted resets as rejections
    entry2 = template_summary(0, 1.0, rl)
    assert entry2["counters"] == {"scene_reset": 3} and "scene_reset" not in entry2["rejections"]
    run = new_run_summary("2.0.0-dev", "customs.json", "/tmp/out")
    run["problems"].setdefault("alias", {})["scene__target-apple-a"] = entry
    # the layout bench_report.py reads: problems.<problem>.<template>.{n_written, time_s}
    for temps in run["problems"].values():
        for e in temps.values():
            assert {"n_written", "time_s"} <= set(e)
    json.dumps(run)
