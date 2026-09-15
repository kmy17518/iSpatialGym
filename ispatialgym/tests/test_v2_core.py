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


# ----------------------------------------------------------------------------------------------------------------
# summary layout (§3.5)
# ----------------------------------------------------------------------------------------------------------------


def test_rejection_log_and_summary_layout():
    rl = RejectionLog(keep=3)
    for i in range(5):
        rl.add("grasp_unreachable", f"attempt {i}", index=0, attempt=i)
    rl.add("no_transition", "init -> manipulation")
    d = rl.to_json()
    assert d["counts"] == {"grasp_unreachable": 5, "no_transition": 1} and len(d["examples"]["grasp_unreachable"]) == 3
    assert d["examples"]["grasp_unreachable"][0]["attempt"] == 0 and rl.total == 6
    entry = template_summary(2, 123.456, rl, seeds=[7, 8])
    assert entry["n_written"] == 2 and entry["time_s"] == 123.46 and entry["rejections"]["no_transition"] == 1 and entry["seeds"] == [7, 8]
    run = new_run_summary("2.0.0-dev", "customs.json", "/tmp/out")
    run["problems"].setdefault("alias", {})["scene__target-apple-a"] = entry
    # the layout bench_report.py reads: problems.<problem>.<template>.{n_written, time_s}
    for temps in run["problems"].values():
        for e in temps.values():
            assert {"n_written", "time_s"} <= set(e)
    json.dumps(run)
