"""``ground_isg`` v2 (iddl + icl -> BDDL + customs list) on the shipped icl, into a temporary directory (no simulator;
the BEHAVIOR knowledge base is needed for the synsets)."""

import json
import os
import re
import sys
import types

import pytest

from ispatialgym.activity import problem as P
from ispatialgym.sampling import bddl_from_iddl as B
from ispatialgym.sampling import customs as C
from ispatialgym.sampling import generation_defaults as G
from ispatialgym.sampling import ground_isg as GI
from ispatialgym.sampling import iddl as I

IDDLS = I.load_all()

#: one subtask per family (+ the two ``under`` argument orders, a :groups composite, a scene target, a wall support)
SUBTASKS = [
    "camera_relocalization-standard",
    "camera_relocalization-obstructed",
    "object_scaling",
    "mental_rotation",
    "disembedding",
    "configuration_matching-block_design",
    "configuration_matching-attachment",
    "configuration_matching-articulation_open_large_scale",
    "configuration_matching-articulation_open_small_scale",
    "alignment-world_vertical",
    "alignment-axial",
    "inter_object_relation_satisfaction-under_small_scale",
    "inter_object_relation_satisfaction-under_large_scale",
    "inter_object_relation_satisfaction-nextto",
]


@pytest.fixture(scope="module")
def grounded(tmp_path_factory):
    """Ground the representative picks of :data:`SUBTASKS` into a fresh bddl root + customs file."""
    root = tmp_path_factory.mktemp("ground_isg")
    bddl_root = os.path.join(root, "activity_definitions")
    customs_path = os.path.join(root, "isg_customs.json")
    icl = GI.ICL()
    customs = C.Customs.load(customs_path)
    grounder = GI.Grounder(IDDLS, icl, customs, bddl_root=bddl_root, options=GI.Options(representative_test=False))
    report = grounder.run(SUBTASKS)
    return types.SimpleNamespace(root=root, bddl_root=bddl_root, customs_path=customs_path, grounder=grounder, report=report,
                                 customs=C.Customs.load(customs_path), icl=icl)


def _entries(grounded, subtask):
    return [gp for gp in grounded.grounder.grounded if gp.subtask == subtask]


def test_every_subtask_produced_problems(grounded):
    for st in SUBTASKS:
        assert grounded.report.problems[st], f"{st}: nothing grounded ({grounded.report.skipped.get(st)})"
    # the shipped selection: 30 disembedding picks, 6 attachment picks, 1 block design, 17 obstacles
    assert len(grounded.report.problems["disembedding"]) == 30
    assert len(grounded.report.problems["configuration_matching-attachment"]) == 6
    assert len(grounded.report.problems["configuration_matching-block_design"]) == 1
    assert len(grounded.report.problems["camera_relocalization-standard"]) == 1
    assert grounded.report.text().startswith("ground_isg")


def test_bddl_files_parse_compile_and_registries_are_contiguous(grounded):
    for act in P.list_activities(grounded.bddl_root):
        ids = P.list_problem_ids(act, grounded.bddl_root)
        assert ids == list(range(len(ids)))
        registry = P.validate_problem_registry(act, grounded.bddl_root)  # the activity.problem reader accepts our shape
        assert sorted(registry.values()) == ids  # every grounded problem carries an alias and is registered
        for pid in ids:
            problem = P.load_problem(act, pid, grounded.bddl_root)
            compiled = problem.compile()
            assert problem.alias is not None and P.ALIAS_RE.match(problem.alias)
            assert compiled.ground_goal_state_options
    # ids are per alias and stable: re-grounding one subtask keeps the ids and rewrites the same files
    before = {gp.alias: gp.definition_id for gp in grounded.grounder.grounded}
    again = GI.Grounder(IDDLS, grounded.icl, C.Customs.load(grounded.customs_path), bddl_root=grounded.bddl_root,
                        options=GI.Options(representative_test=False))
    again.run(["mental_rotation"])
    for gp in again.grounded:
        assert before[gp.alias] == gp.definition_id


def test_alias_matches_the_template_and_the_registry(grounded):
    for gp in grounded.grounder.grounded:
        iddl = IDDLS[gp.subtask]
        assert gp.alias.startswith(gp.subtask)
        assert gp.problem.alias == gp.bddl_alias
        assert B.bddl_alias(gp.alias) == gp.bddl_alias
        assert len(gp.alias.split("-")) == len(iddl.alias_template.split("-"))
        reg = B.ProblemRegistry.load(os.path.join(grounded.bddl_root, gp.activity))
        assert reg.id_of(gp.alias) == gp.definition_id
        rec = reg.record(gp.alias)
        assert rec["subtask"] == gp.subtask and rec["iddl"] == gp.entry.iddl


def test_customs_entry_has_the_spec_keys(grounded):
    data = json.load(open(grounded.customs_path))
    for gp in grounded.grounder.grounded:
        e = data[gp.alias]
        assert tuple(e) == C.ENTRY_KEYS
        assert e["iddl"] == I.relative_path(IDDLS[gp.subtask].path) and e["definition_id"] == gp.definition_id
        assert "floor" in e["roles"] and e["roles"]["floor"] == {"sampleable": False}
        for role, r in e["roles"].items():
            assert set(r) <= set(C.ROLE_KEYS), (gp.alias, role, sorted(r))
            assert isinstance(r["sampleable"], bool)
            if role != "floor":
                assert r.get("category")
        for scene, s in e["scenes"].items():
            assert scene in C.SCENE_CANDIDATES
            assert tuple(s) == C.SCENE_KEYS
            assert s["bindings"]["floor"] == {"room": s["rooms"][0]} if s["rooms"] else True
            for role, cats in s["whitelist"].items():
                assert e["roles"][role]["sampleable"]
                for cat, models in cats.items():
                    assert models and cat in ([e["roles"][role]["category"]] if isinstance(e["roles"][role]["category"], str) else e["roles"][role]["category"])
            for role, cats in s["blacklist"].items():
                for cat, models in cats.items():
                    for model, reason in models.items():
                        assert reason.startswith("ground_isg ") and model not in s["whitelist"].get(role, {}).get(cat, [])
        gen = e["generation"]
        for k in ("templates_max", "instances_per_template", "tolerances", "groups", "path_buckets"):
            assert k in gen, (gp.alias, k)
        for rec in e["constraints"]:
            c = C.Constraint.from_dict(rec)
            assert c.type in C.CONSTRAINT_TYPES and set(c.scope) <= set(C.SCOPE_KEYS) and "sampler" in c.source
        # round trip through the dataclasses
        assert C.CustomsEntry.from_dict(gp.alias, e).to_dict() == e


def test_require_is_copied_and_grasp_keys_resolved(grounded):
    gp = _entries(grounded, "mental_rotation")[0]
    t = gp.entry.roles["target"]
    assert t.require == {"grasps": "floating"} and t.representative
    key = t.grasp_keys[f"{t.category}/{t.representative}"]
    assert key.startswith(f"{t.category}__{t.representative}")
    assert f"{t.category}/{t.representative}" in t.orientation_equivalences  # mental rotation targets are asymmetric
    assert t.orientation_equivalences[f"{t.category}/{t.representative}"] == []
    gp = _entries(grounded, "alignment-world_vertical")[0]
    assert gp.entry.roles["target"].require == {"contacts": True}
    gp = _entries(grounded, "configuration_matching-articulation_open_small_scale")[0]
    t = gp.entry.roles["target"]
    assert t.require == {"handle": True}
    assert t.joint[f"{t.category}/{t.representative}"] and t.handle_key[f"{t.category}/{t.representative}"]


def test_heldout_models_are_blacklisted_in_every_role(grounded):
    heldout = grounded.icl.heldout_models
    assert ("box_of_shampoo", "tzghev") in heldout and ("pencil", "fnlbpu") in heldout
    n = 0
    for gp in grounded.grounder.grounded:
        for scene in gp.entry.scenes.values():
            for role, r in gp.entry.roles.items():
                if not r.sampleable:
                    continue
                for cat in r.categories:
                    for hc, hm in heldout:
                        if hc != cat:
                            continue
                        assert scene.is_blacklisted(role, cat, hm), (gp.alias, role, cat, hm)
                        assert C.reason_code(scene.blacklist[role][cat][hm]) == "heldout"
                        assert hm not in scene.whitelist.get(role, {}).get(cat, [])
                        n += 1
    assert n > 50
    # the disembedding target's held-out model is blacklisted for the target and, when its category is a distractor
    # category, for the distractors too
    gp = next(g for g in _entries(grounded, "disembedding") if g.entry.roles["target"].category == "box_of_shampoo")
    sc = gp.entry.scenes["house_single_floor"]
    assert sc.is_blacklisted("target", "box_of_shampoo", "tzghev") and sc.is_blacklisted("distractors", "box_of_shampoo", "tzghev")
    assert "tzghev" not in sc.whitelist["distractors"]["box_of_shampoo"]


def test_under_argument_orders(grounded):
    small = _entries(grounded, "inter_object_relation_satisfaction-under_small_scale")[0]
    t, p = small.rendered.instances["target"], small.rendered.instances["partner"]
    assert ["under", "?" + p, "?" + t] in small.problem.goal  # the partner ends under the manipulated target
    assert ["not", ["under", p, t]] in small.problem.init
    large = _entries(grounded, "inter_object_relation_satisfaction-under_large_scale")[0]
    t, p = large.rendered.instances["target"], large.rendered.instances["partner"]
    assert ["under", "?" + t, "?" + p] in large.problem.goal  # the target ends under the scene partner
    assert p.rsplit("_", 1)[0] in large.problem.objects and large.entry.roles["partner"].sampleable is False


def test_large_scale_scene_roles_and_floor_support(grounded):
    for gp in _entries(grounded, "inter_object_relation_satisfaction-under_large_scale"):
        support = gp.entry.roles["support"]
        for scene, sc in gp.entry.scenes.items():
            assert sc.bindings["partner"], gp.alias
            if support.category == GI.FLOOR_CATEGORY:
                assert gp.alias.endswith("-floor")
                assert sc.bindings["support"] == {"room": sc.rooms[0]}
                assert gp.rendered.instances["support"] == gp.rendered.instances["floor"]
            elif sc.bindings["support"] == sc.bindings["partner"]:
                assert gp.rendered.instances["support"] == gp.rendered.instances["partner"]
            else:
                assert sc.bindings["support"]
    art = _entries(grounded, "configuration_matching-articulation_open_large_scale")
    assert art
    for gp in art:
        assert not gp.entry.roles["target"].sampleable
        for scene, sc in gp.entry.scenes.items():
            for name in sc.bindings["target"]:
                assert gp.entry.roles["target"].joint.get(f"{scene}/{name}"), (gp.alias, name)
                assert gp.entry.roles["target"].handle_key.get(f"{scene}/{name}")
            assert not sc.whitelist  # nothing is spawned
    # small-scale articulation: targets taller than 0.6 m stand on the floor (§5.4)
    small = _entries(grounded, "configuration_matching-articulation_open_small_scale")
    tall = [gp for gp in small if float(gp.combination.roles["target"].pick_row["bbox_z"]) > GI.FLOOR_SUPPORT_HEIGHT_M]
    assert tall and all(gp.alias.endswith("-floor") for gp in tall)
    assert all(not gp.alias.endswith("-floor") for gp in small if gp not in tall)


def test_wall_support_bindings(grounded):
    gp = _entries(grounded, "alignment-world_vertical")[0]
    assert gp.alias.endswith("-walls")
    sc = gp.entry.scenes["house_single_floor"]
    assert sc.bindings["support"] and all(w.startswith("walls_") for w in sc.bindings["support"])
    regions = sc.bindings["support_regions"]
    assert regions and all(r["p0"] and r["p1"] and r["normal"] and r["hang_z"] for r in regions)
    assert {r["wall"] for r in regions} == set(sc.bindings["support"])
    # the wall subtasks are bound to living_room_0 of house_single_floor for now (room rows)
    assert list(gp.entry.scenes) == ["house_single_floor"] and sc.rooms == ["living_room_0"]
    # contacts.csv is header-only: the requirement is pending, recorded softly, the model stays whitelisted
    assert sc.whitelist["target"]
    pend = [c for c in gp.entry.constraints if c.fields.get("reason") == GI.PENDING_CONTACTS_REASON]
    assert pend and all(c.type == "blacklist" and not c.hard for c in pend)
    horizontal = next(iter(_entries(grounded, "alignment-axial")))
    assert horizontal.entry.generation["tolerances"] == {"orientation": 0.10}


def test_groups(grounded):
    blocks = _entries(grounded, "configuration_matching-block_design")[0]
    assert blocks.alias == "configuration_matching-block_design-block_4-breakfast_table"
    assert blocks.entry.generation["groups"]["targets"] == {"size": 4, "draw": "same_model", "members": {"toy_dice": 4}}
    assert blocks.entry.roles["targets"].dataset == "ispatialgym-assets"
    assert blocks.problem.groups == {"targets_group.n.01_1": [f"toy_dice.n.01_{i}" for i in range(1, 5)]}
    obstructed = _entries(grounded, "camera_relocalization-obstructed")[0]
    assert re.fullmatch(r"camera_relocalization-obstructed-[a-z_]+_1", obstructed.alias)
    assert obstructed.entry.generation["groups"]["obstacles"]["size"] == 1
    assert obstructed.entry.generation["path_buckets"] == G.DEFAULTS["camera_relocalization-obstructed"]["path_buckets"]
    assert len(obstructed.problem.groups["obstacles_group.n.01_1"]) == 1
    for gp in _entries(grounded, "disembedding"):
        g = gp.entry.generation["groups"]["distractors"]
        assert g["draw"] == "unique" and g["size"] == sum(g["members"].values()) and 1 <= g["size"] <= 3
        token = gp.alias.split("-")[2]
        assert token == "_".join(f"{c}_{n}" for c, n in g["members"].items())
        members = gp.problem.groups["distractors_group.n.01_1"]
        assert len(members) == g["size"]
        sc = gp.entry.scenes["house_single_floor"]
        for cat in g["members"]:
            assert sc.whitelist["distractors"][cat], (gp.alias, cat)
        target = gp.entry.roles["target"]
        assert target.representative not in sc.whitelist["distractors"].get(target.category, [])
        assert ["ontop", "distractors_group.n.01_1", gp.rendered.instances["source"]] in gp.problem.raw_init
    attachment = _entries(grounded, "configuration_matching-attachment")[0]
    assert "assembly_group.n.01_1" in attachment.problem.groups
    assert attachment.entry.generation["groups"] == {}  # the composite is not a drawn group


def test_generation_is_seeded_from_defaults_then_from_the_existing_entry(grounded):
    gp = _entries(grounded, "inter_object_relation_satisfaction-nextto")[0]
    for k, v in G.DEFAULTS["inter_object_relation_satisfaction-nextto"].items():
        assert gp.entry.generation[k] == v
    tuned = G.seed_generation("inter_object_relation_satisfaction-nextto", {"generation": {"templates_max": 3, "tolerances": {"position": 0.01}}})
    assert tuned["templates_max"] == 3 and tuned["tolerances"] == {"position": 0.01} and tuned["instances_per_template"] == 5
    assert set(G.DEFAULTS) == set(IDDLS)
    for st, gen in G.DEFAULTS.items():
        assert gen["templates_max"] == 20 and gen["instances_per_template"] == 5
        assert isinstance(gen["tolerances"], dict) and "groups" in gen and "path_buckets" in gen
        for role, g in gen["groups"].items():
            assert g["draw"] in ("unique", "same_model") and (isinstance(g["size"], int) or len(g["size"]) == 2)


def test_registry_absorbs_legacy_templates_and_flags_duplicate_aliases(tmp_path):
    act = tmp_path / "configuration_matching"
    act.mkdir()
    for i, alias in enumerate(["block_design", "articulation", "articulation", "attachment"]):
        (act / f"problem{i}.bddl").write_text(f"(define (problem configuration_matching-{i}) (:domain ispatialgym) (:alias {alias}))\n")
    (act / "problem4.bddl").write_text("(define (problem configuration_matching-4) (:domain ispatialgym))\n")
    reg = B.ProblemRegistry.load(str(act))
    assert reg.aliases() == {"block_design": 0, "articulation": 1, "attachment": 3}
    assert [(u["id"], u["alias"]) for u in reg.unregistered] == [(2, "articulation"), (4, None)]
    assert reg.next_id() == 5
    assert reg.allocate("configuration_matching-attachment-a-b-c", subtask="x") == 5
    assert reg.allocate("configuration_matching-attachment-a-b-c") == 5  # never a second id for the same alias
    reg.save()
    assert P.load_problem_registry(str(act)) == {"block_design": 0, "articulation": 1, "attachment": 3, "configuration_matching-attachment-a-b-c": 5}
    with pytest.raises(B.GroundingError, match="not contiguous"):
        (act / "problem7.bddl").write_text("(define (problem configuration_matching-7) (:domain ispatialgym))\n")
        B.ProblemRegistry.load(str(act))


def test_bddl_alias_falls_back_to_the_parser_spelling(monkeypatch):
    alias = "inter_object_relation_satisfaction-nextto-spatula-soup_ladle-breakfast_table"
    assert B.bddl_alias(alias) == alias  # the parser accepts the v2 alias
    monkeypatch.setattr(B, "ALIAS_RE", re.compile(r"^[a-z][a-z0-9_]*$"))  # the pre-A4 parser: no '-'
    spelled = B.bddl_alias(alias)
    assert spelled == "inter_object_relation_satisfaction__nextto__spatula__soup_ladle__breakfast_table"
    assert B.full_alias_of(spelled) == alias


def test_render_rejects_missing_or_extra_roles():
    iddl = IDDLS["inter_object_relation_satisfaction-nextto"]
    from ispatialgym.sampling.knowledge import synset_of_category as syn

    g = {"target": B.RoleGrounding("target", "sampleable", "pencil", syn("pencil")), "support": B.RoleGrounding("support", "scene", "desk", syn("desk"))}
    with pytest.raises(B.GroundingError, match="without a grounding"):
        B.render(iddl, g, 0, "x", "inter_object_relation_satisfaction")
    g["partner"] = B.RoleGrounding("partner", "sampleable", "ruler", syn("ruler"))
    g["bogus"] = B.RoleGrounding("bogus", "sampleable", "ruler", syn("ruler"))
    with pytest.raises(B.GroundingError, match="does not have"):
        B.render(iddl, g, 0, "x", "inter_object_relation_satisfaction")
    del g["bogus"]
    r = B.render(iddl, g, 0, "inter_object_relation_satisfaction-nextto-pencil-ruler-desk", "inter_object_relation_satisfaction")
    t, p = f"{syn('pencil')}_1", f"{syn('ruler')}_1"
    assert r.instances == {"target": t, "partner": p, "support": f"{syn('desk')}_1", "floor": "floor.n.01_1", "agent": "agent.n.01_1"}
    assert ["nextto", "?" + t, "?" + p] in r.problem.goal
    assert ["pose_reached", "?" + p] in r.problem.measure


def test_equivalence_groups():
    assert C.equivalence_group([]) == [[0.0, 0.0, 0.0, 1.0]]
    assert len(C.equivalence_group(["z2"])) == 2
    assert len(C.equivalence_group(["z2", "x2"])) == 4
    assert len(C.equivalence_group(["z4", "x2"])) == 8
    assert len(C.equivalence_group(["zinf"])) == 72
    assert len(C.equivalence_group(["zinf", "x2"])) == 144
    assert C.equivalence_group(["full"]) is None
    assert C.parse_symmetry_tags("none") == [] and C.parse_symmetry_tags("zinf x2") == ["zinf", "x2"]
    assert C.half_turn_tag("+x") == "x2"


def test_representative_test_hook_is_skipped_without_the_sampler(grounded, monkeypatch, caplog):
    gp = _entries(grounded, "mental_rotation")[0]
    fake = types.ModuleType("ispatialgym.sampling.v2.core")  # a v2 core without the entry point yet
    monkeypatch.setitem(sys.modules, "ispatialgym.sampling.v2.core", fake)
    GI._SKIP_LOGGED.clear()
    with caplog.at_level("INFO", logger="ispatialgym.ground_isg"):
        assert GI.run_representative_test(gp, grounded.customs, "house_single_floor") is None
    assert any("TODO D2" in rec.message for rec in caplog.records)
    calls = []
    fake.run_representative_test = lambda problem, customs, scene: calls.append((problem.alias, scene)) or "ok"
    assert GI.run_representative_test(gp, grounded.customs, "house_single_floor") == "ok"
    assert calls == [(gp.alias, "house_single_floor")]


def test_cli_dry_run(tmp_path):
    rc = GI.main(["--dry-run", "--subtask", "object_scaling", "--out-bddl", str(tmp_path / "bddl"), "--customs", str(tmp_path / "c.json"),
                  "--no-representative-test"])
    assert rc == 0
    assert not (tmp_path / "c.json").exists() and not (tmp_path / "bddl").exists()
    rc = GI.main(["--subtask", "object_scaling", "--out-bddl", str(tmp_path / "bddl"), "--customs", str(tmp_path / "c.json"),
                  "--no-representative-test", "--legacy-activity-names", "--report", str(tmp_path / "report.md")])
    assert rc == 0 and (tmp_path / "report.md").exists() and (tmp_path / "report.json").exists()
    data = json.load(open(tmp_path / "c.json"))
    assert len([k for k in data if not k.startswith("_")]) == 30
    assert P.list_problem_ids("object_scaling", str(tmp_path / "bddl")) == list(range(30))


def test_legacy_activity_names(tmp_path):
    icl = GI.ICL()
    customs = C.Customs.load(str(tmp_path / "c.json"))
    grounder = GI.Grounder(IDDLS, icl, customs, bddl_root=str(tmp_path / "bddl"), options=GI.Options(representative_test=False, legacy_activity_names=True))
    grounder.run(["alignment-axial"])
    assert os.path.isdir(tmp_path / "bddl" / "world_alignment")
    assert P.list_problem_ids("world_alignment", str(tmp_path / "bddl")) == list(range(10))
    assert all(gp.activity == "world_alignment" for gp in grounder.grounded)
