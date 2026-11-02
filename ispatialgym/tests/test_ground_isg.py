"""``ground_isg`` v2 (iddl + icl -> BDDL + customs list) on the shipped icl, into a temporary directory (no simulator;
the BEHAVIOR knowledge base is needed for the synsets)."""

import atexit
import collections
import functools
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import types

import pytest

from ispatialgym.activity import problem as P
from ispatialgym.sampling import bddl_from_iddl as B
from ispatialgym.sampling import customs as C
from ispatialgym.sampling import generation_defaults as G
from ispatialgym.sampling import ground_isg as GI
from ispatialgym.sampling import iddl as I

IDDLS = I.load_all()

#: the representative sheet these tests were written against: the first-demo selection (``roles/representatives.csv`` until
#: the owner's new sheet replaced it on 2026-10-11), grounded with the shipped icl's other sheets
FIRST_DEMO_SHEET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "representatives_first_demo.csv")


@functools.lru_cache(maxsize=None)
def first_demo_icl() -> str:
    """An icl root: the shipped ``roles/`` and ``require/`` sheets (symlinked) with :data:`FIRST_DEMO_SHEET` as
    ``roles/representatives.csv``."""
    root = tempfile.mkdtemp(prefix="icl_first_demo_")
    atexit.register(shutil.rmtree, root, True)
    for sub in ("roles", "require"):
        os.makedirs(os.path.join(root, sub))
        for name in os.listdir(os.path.join(GI.DEFAULT_ICL, sub)):
            if (sub, name) != ("roles", "representatives.csv"):
                os.symlink(os.path.join(GI.DEFAULT_ICL, sub, name), os.path.join(root, sub, name))
    shutil.copy(FIRST_DEMO_SHEET, os.path.join(root, "roles", "representatives.csv"))
    return root

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
    "inter_object_relation_satisfaction-nextto_small_scale",
]


@pytest.fixture(scope="module")
def grounded(tmp_path_factory):
    """Ground the representative picks of :data:`SUBTASKS` into a fresh bddl root + customs file."""
    root = tmp_path_factory.mktemp("ground_isg")
    bddl_root = os.path.join(root, "activity_definitions")
    customs_path = os.path.join(root, "isg_customs.json")
    icl = GI.ICL(first_demo_icl())
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
    # the shipped selection: 30 disembedding picks, 6 attachment picks, 1 block design, 17 obstacles -- each tabletop
    # pick once per representative support of its support triplets (A3: T1..T3)
    n_sup = len(grounded.icl.picks["disembedding#01"]["support"]["representative"])
    assert n_sup >= 1
    assert len(grounded.report.problems["disembedding"]) == 30 * n_sup
    assert len(grounded.report.problems["configuration_matching-attachment"]) == 6 * n_sup
    assert len(grounded.report.problems["configuration_matching-block_design"]) == n_sup
    assert len(grounded.report.problems["camera_relocalization-standard"]) == 1
    assert len(grounded.report.problems["camera_relocalization-obstructed"]) == 17
    tokens = {gp.variant.split("-")[-1] for gp in _entries(grounded, "disembedding")}
    assert tokens == {r["category"] for r in grounded.icl.picks["disembedding#01"]["support"]["representative"]}
    assert grounded.report.text().startswith("ground_isg")


def test_bddl_files_parse_compile_and_registries_are_contiguous(grounded):
    for act in P.list_activities(grounded.bddl_root):
        ids = P.list_problem_ids(act, grounded.bddl_root)
        assert ids == list(range(len(ids)))
        registry = P.validate_problem_registry(act, grounded.bddl_root)  # the activity.problem reader accepts our shape
        assert sorted(registry.values()) == ids  # every grounded problem carries a variant and is registered
        for pid in ids:
            problem = P.load_problem(act, pid, grounded.bddl_root)
            compiled = problem.compile()
            assert problem.variant is not None and P.VARIANT_RE.match(problem.variant)
            assert compiled.ground_goal_state_options
    # ids are per variant and stable: re-grounding one subtask keeps the ids and rewrites the same files
    before = {gp.variant: gp.definition_id for gp in grounded.grounder.grounded}
    again = GI.Grounder(IDDLS, grounded.icl, C.Customs.load(grounded.customs_path), bddl_root=grounded.bddl_root,
                        options=GI.Options(representative_test=False))
    again.run(["mental_rotation"])
    for gp in again.grounded:
        assert before[gp.variant] == gp.definition_id


def test_variant_matches_the_template_and_the_registry(grounded):
    for gp in grounded.grounder.grounded:
        iddl = IDDLS[gp.subtask]
        assert gp.variant.startswith(gp.subtask)
        assert gp.problem.variant == gp.bddl_variant
        assert B.bddl_variant(gp.variant) == gp.bddl_variant
        assert len(gp.variant.split("-")) == len(iddl.variant_template.split("-"))
        reg = B.ProblemRegistry.load(os.path.join(grounded.bddl_root, gp.activity))
        assert reg.id_of(gp.variant) == gp.definition_id
        rec = reg.record(gp.variant)
        assert rec["subtask"] == gp.subtask and rec["iddl"] == gp.entry.iddl


def test_customs_entry_has_the_spec_keys(grounded):
    data = json.load(open(grounded.customs_path))
    for gp in grounded.grounder.grounded:
        e = data[gp.variant]
        assert set(e) <= set(C.ENTRY_KEYS) and list(e) == [k for k in C.ENTRY_KEYS if k in e]  # spec order, optional fallback keys
        assert {"iddl", "definition_id", "roles", "scenes", "generation", "constraints"} <= set(e)
        assert e["iddl"] == I.relative_path(IDDLS[gp.subtask].path) and e["definition_id"] == gp.definition_id
        assert "floor" in e["roles"] and e["roles"]["floor"] == {"sampleable": False}
        for role, r in e["roles"].items():
            assert set(r) <= set(C.ROLE_KEYS), (gp.variant, role, sorted(r))
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
            assert k in gen, (gp.variant, k)
        for rec in e["constraints"]:
            c = C.Constraint.from_dict(rec)
            assert c.type in C.CONSTRAINT_TYPES and set(c.scope) <= set(C.SCOPE_KEYS) and "sampler" in c.source
        # round trip through the dataclasses
        assert C.CustomsEntry.from_dict(gp.variant, e).to_dict() == e


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
        if gp.entry.fallback_kind == "next_model":
            continue  # the next-model fallback may name the held-out model of its category on purpose
        for scene in gp.entry.scenes.values():
            for role, r in gp.entry.roles.items():
                if not r.sampleable:
                    continue
                for cat in r.categories:
                    for hc, hm in heldout:
                        if hc != cat:
                            continue
                        assert scene.is_blacklisted(role, cat, hm), (gp.variant, role, cat, hm)
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
            assert sc.bindings["partner"], gp.variant
            if support.category == GI.FLOOR_CATEGORY:
                assert gp.variant.endswith("-floor")
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
                assert gp.entry.roles["target"].joint.get(f"{scene}/{name}"), (gp.variant, name)
                assert gp.entry.roles["target"].handle_key.get(f"{scene}/{name}")
            assert not sc.whitelist  # nothing is spawned
    # small-scale articulation: targets taller than 0.6 m stand on the floor (§5.4)
    small = _entries(grounded, "configuration_matching-articulation_open_small_scale")
    tall = [gp for gp in small if float(gp.combination.roles["target"].pick_row["bbox_z"]) > GI.FLOOR_SUPPORT_HEIGHT_M]
    assert tall and all(gp.variant.endswith("-floor") for gp in tall)
    assert all(not gp.variant.endswith("-floor") for gp in small if gp not in tall)


def test_wall_support_bindings(grounded):
    gp = _entries(grounded, "alignment-world_vertical")[0]
    assert gp.variant.endswith("-walls")
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
    axial = next(iter(_entries(grounded, "alignment-axial")))
    assert axial.entry.generation["tolerances"] == {"orientation": 0.10}


def test_wall_frames_are_written_per_model(grounded):
    """Owner 2026‑09‑16 22:55 (spec §5.5): every whitelisted wall target carries its reviewed icl frame in the variant as
    ``generation.front_axis`` / ``up_axis`` in the per-model dict form the sampler's ``target_param`` resolves — the
    sampler never defaults a frame (``Reject("missing_frame")``)."""
    st = "alignment-world_vertical"
    entries = _entries(grounded, st)
    assert entries
    for gp in entries:
        gen = gp.entry.generation
        whitelisted = {f"{cat}/{m}" for sc in gp.entry.scenes.values() for cat, models in sc.whitelist.get("target", {}).items() for m in models}
        assert whitelisted, gp.variant
        for key in whitelisted:
            assert isinstance(gen.get("front_axis"), dict) and gen["front_axis"].get(key), f"{gp.variant}: no front_axis for {key}"
            assert isinstance(gen.get("up_axis"), dict) and gen["up_axis"].get(key), f"{gp.variant}: no up_axis for {key}"
            assert "level_axis" not in gen
            row = grounded.icl.row(st, "target", *key.split("/", 1))
            assert gen["front_axis"][key] == GI.wall_frame_of(row, st)["front_axis"]
    plate = [gp for gp in entries if gp.entry.roles["target"].category == "license_plate"]
    if plate:  # the representative pick of the license plate: reviewed +z / +y (not the painting-shaped +x / +z)
        gen = plate[0].entry.generation
        key = next(k for k in gen["front_axis"] if k.startswith("license_plate/"))
        assert (gen["front_axis"][key], gen["up_axis"][key]) == ("+z", "+y")


def test_groups(grounded):
    blocks = next(gp for gp in _entries(grounded, "configuration_matching-block_design") if gp.variant.endswith("-breakfast_table"))
    assert blocks.variant == "configuration_matching-block_design-block_4-breakfast_table"
    assert blocks.entry.generation["groups"]["targets"] == {"size": 4, "draw": "same_model", "members": {"toy_dice": 4}}
    assert blocks.entry.roles["targets"].dataset == "ispatialgym-assets"
    assert blocks.problem.groups == {"targets_group.n.01_1": [f"toy_dice.n.01_{i}" for i in range(1, 5)]}
    obstructed = _entries(grounded, "camera_relocalization-obstructed")[0]
    assert re.fullmatch(r"camera_relocalization-obstructed-[a-z_]+_1", obstructed.variant)
    assert obstructed.entry.generation["groups"]["obstacles"]["size"] == 1
    assert obstructed.entry.generation["path_buckets"] == G.DEFAULTS["camera_relocalization-obstructed"]["path_buckets"]
    assert len(obstructed.problem.groups["obstacles_group.n.01_1"]) == 1
    budgets = []
    for gp in _entries(grounded, "disembedding"):
        g = gp.entry.generation["groups"]["distractors"]
        assert g["draw"] == "unique" and g["size"] == sum(g["members"].values()) and 1 <= g["size"] <= 3
        # owner decision 2026-09-16 4a: the §6.4 tray budget decided the member count at grounding
        b = gp.entry.generation["distractor_budget"]
        assert {"tray_floor_m2", "used_m2", "n_distractors"} <= set(b) and b["n_distractors"] == g["size"]
        # 2026-09-24: source tray tray/uekqey at its canonical size (build_icl.TRAY_OVERRIDES); its 0.1673 m2 inner floor
        # equals the retired gsxbym x 1.25 (0.107 x 1.25^2), so the member counts of the shipped picks are unchanged
        assert b["tray"] == "tray/uekqey" and b["tray_scale"] == 1.0 and abs(b["tray_floor_m2"] - 0.1673) < 1e-3
        assert b["used_m2"] <= b["budget_fraction"] * b["tray_floor_m2"] + 1e-4  # both recorded rounded to 4 decimals
        assert b["footprint_rule"] in ("square", "rectangle") and set(b["footprints_m2"]) == set(g["members"])
        assert abs(sum(b["footprints_m2"][c] * n for c, n in g["members"].items()) - b["used_m2"]) < 1e-3
        budgets.append(b)
        token = gp.variant.split("-")[2]
        assert token == "_".join(f"{c}_{n}" for c, n in g["members"].items())
        members = gp.problem.groups["distractors_group.n.01_1"]
        assert len(members) == g["size"]
        sc = next(iter(gp.entry.scenes.values()))
        for cat in g["members"]:
            assert sc.whitelist["distractors"][cat], (gp.variant, cat)
        target = gp.entry.roles["target"]
        assert target.representative not in sc.whitelist["distractors"].get(target.category, [])
        assert ["ontop", "distractors_group.n.01_1", gp.rendered.instances["source"]] in gp.problem.raw_init
        # owner decision 2026-09-16 4b (source tray at generation.source_tray_scale, destination at 1.0); since 2026-09-24 the
        # source tray uekqey spawns at its canonical size, so neither role carries a scale
        src, dst = gp.entry.roles["source"], gp.entry.roles["destination"]
        assert src.representative == "uekqey" and dst.representative == "hvlfig"
        assert src.scale == {} and dst.scale == {}
        assert gp.entry.generation["source_tray_scale"] == 1.0
    # the first-demo picks at the 0.85 budget (owner 2026-10-11): 3, or 2 for the lying bottles (at 0.60 they spanned 3 -> 2 -> 1)
    assert G.DEFAULTS["disembedding"]["tray_budget"] == 0.85 and {b["n_distractors"] for b in budgets} == {2, 3}
    assert any(b["dropped"] for b in budgets) and any(b["footprint_rule"] == "rectangle" for b in budgets)
    attachment = _entries(grounded, "configuration_matching-attachment")[0]
    assert "assembly_group.n.01_1" in attachment.problem.groups
    assert attachment.entry.generation["groups"] == {}  # the composite is not a drawn group


def test_generation_is_seeded_from_defaults_then_from_the_existing_entry(grounded):
    gp = _entries(grounded, "inter_object_relation_satisfaction-nextto_small_scale")[0]
    for k, v in G.DEFAULTS["inter_object_relation_satisfaction-nextto_small_scale"].items():
        assert gp.entry.generation[k] == v
    tuned = G.seed_generation("inter_object_relation_satisfaction-nextto_small_scale", {"generation": {"templates_max": 3, "tolerances": {"position": 0.01}}})
    assert tuned["templates_max"] == 3 and tuned["tolerances"] == {"position": 0.01} and tuned["instances_per_template"] == 5
    assert set(G.DEFAULTS) == set(IDDLS)
    for st, gen in G.DEFAULTS.items():
        assert gen["templates_max"] == 20 and gen["instances_per_template"] == 5
        assert isinstance(gen["tolerances"], dict) and "groups" in gen and "path_buckets" in gen
        for role, g in gen["groups"].items():
            assert g["draw"] in ("unique", "same_model") and (isinstance(g["size"], int) or len(g["size"]) == 2)


def test_registry_absorbs_legacy_templates_and_flags_duplicate_variants(tmp_path):
    act = tmp_path / "configuration_matching"
    act.mkdir()
    for i, variant in enumerate(["block_design", "articulation", "articulation", "attachment"]):
        (act / f"problem{i}.bddl").write_text(f"(define (problem configuration_matching-{i}) (:domain ispatialgym) (:variant {variant}))\n")
    (act / "problem4.bddl").write_text("(define (problem configuration_matching-4) (:domain ispatialgym))\n")
    reg = B.ProblemRegistry.load(str(act))
    assert reg.variants() == {"block_design": 0, "articulation": 1, "attachment": 3}
    assert [(u["id"], u["variant"]) for u in reg.unregistered] == [(2, "articulation"), (4, None)]
    assert reg.next_id() == 5
    assert reg.allocate("configuration_matching-attachment-a-b-c", subtask="x") == 5
    assert reg.allocate("configuration_matching-attachment-a-b-c") == 5  # never a second id for the same variant
    reg.save()
    assert P.load_problem_registry(str(act)) == {"block_design": 0, "articulation": 1, "attachment": 3, "configuration_matching-attachment-a-b-c": 5}
    with pytest.raises(B.GroundingError, match="not contiguous"):
        (act / "problem7.bddl").write_text("(define (problem configuration_matching-7) (:domain ispatialgym))\n")
        B.ProblemRegistry.load(str(act))


def test_bddl_variant_falls_back_to_the_parser_spelling(monkeypatch):
    variant = "inter_object_relation_satisfaction-nextto_small_scale-spatula-soup_ladle-breakfast_table"
    assert B.bddl_variant(variant) == variant  # the parser accepts the v2 variant
    monkeypatch.setattr(B, "VARIANT_RE", re.compile(r"^[a-z][a-z0-9_]*$"))  # the pre-A4 parser: no '-'
    spelled = B.bddl_variant(variant)
    assert spelled == "inter_object_relation_satisfaction__nextto_small_scale__spatula__soup_ladle__breakfast_table"
    assert B.full_variant_of(spelled) == variant


def test_render_rejects_missing_or_extra_roles():
    iddl = IDDLS["inter_object_relation_satisfaction-nextto_small_scale"]
    from ispatialgym.sampling.knowledge import synset_of_category as syn

    g = {"target": B.RoleGrounding("target", "sampleable", "pencil", syn("pencil")), "support": B.RoleGrounding("support", "scene", "desk", syn("desk"))}
    with pytest.raises(B.GroundingError, match="without a grounding"):
        B.render(iddl, g, 0, "x", "inter_object_relation_satisfaction")
    g["partner"] = B.RoleGrounding("partner", "sampleable", "ruler", syn("ruler"))
    g["bogus"] = B.RoleGrounding("bogus", "sampleable", "ruler", syn("ruler"))
    with pytest.raises(B.GroundingError, match="does not have"):
        B.render(iddl, g, 0, "x", "inter_object_relation_satisfaction")
    del g["bogus"]
    r = B.render(iddl, g, 0, "inter_object_relation_satisfaction-nextto_small_scale-pencil-ruler-desk", "inter_object_relation_satisfaction")
    t, p = f"{syn('pencil')}_1", f"{syn('ruler')}_1"
    assert r.instances == {"target": t, "partner": p, "support": f"{syn('desk')}_1", "floor": "floor.n.01_1", "agent": "agent.n.01_1"}
    assert ["nextto", "?" + t, "?" + p] in r.problem.goal
    assert ["pose_reached", "?" + p] in r.problem.measure


def test_hold_constraint_of_the_held_object_sampler_round_trips():
    # the record ``sampling/v2/tasks/held_object.py::record_constraints`` writes back (robot-level: empty scope)
    rec = {
        "scope": {}, "type": "hold", "robot": "r1pro", "depth_m": 0.5, "trunk": [0.5, 0.0], "roll_deg": 15.0,
        "approach": "base +y", "closing": "world down rolled 15.0 deg about the approach",
        "eef_in_base": {"position": [0.31, -0.08, 1.21], "orientation": [0.0, 0.0, 0.0, 1.0]},
        "right_arm_q": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], "idle_arms": "reset", "hard": False,
        "source": {"instance": "object_scaling-beaker-0000", "sampler": "2.0.0", "date": "2026-09-15"},
    }
    c = C.Constraint.from_dict(rec)
    assert "hold" in C.CONSTRAINT_TYPES and c.type == "hold" and c.scope == {} and not c.hard
    assert set(C.HOLD_FIELDS) <= set(c.fields) and c.fields["trunk"] == [0.5, 0.0] and c.fields["eef_in_base"]["position"] == [0.31, -0.08, 1.21]
    assert c.to_dict() == rec
    entry = C.CustomsEntry(variant="object_scaling-beaker")
    assert entry.add_constraint(c) and not entry.add_constraint(C.Constraint.from_dict(dict(rec, source={})))  # deduped by type / scope / fields
    assert entry.constraints_of("hold") == [c] and entry.constraints_of("hold", scene="Rs_int") == [c]  # robot-level: any scene
    assert C.CustomsEntry.from_dict(entry.variant, entry.to_dict()).constraints == [c]
    with pytest.raises(C.CustomsError, match="unknown constraint type"):
        C.Constraint.from_dict(dict(rec, type="grip"))


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
    import ispatialgym.sampling.v2 as v2_pkg  # `from ispatialgym.sampling.v2 import core` reads the package attribute once imported

    monkeypatch.setattr(v2_pkg, "core", fake, raising=False)
    GI._SKIP_LOGGED.clear()
    with caplog.at_level("INFO", logger="ispatialgym.ground_isg"):
        assert GI.run_representative_test(gp, grounded.customs, "house_single_floor") is None
    assert any("TODO D2" in rec.message for rec in caplog.records)
    calls = []
    fake.run_representative_test = lambda problem, customs, scene: calls.append((problem.variant, scene)) or "ok"
    assert GI.run_representative_test(gp, grounded.customs, "house_single_floor") == "ok"
    assert calls == [(gp.variant, "house_single_floor")]


def test_cli_dry_run(tmp_path):
    rc = GI.main(["--icl", first_demo_icl(), "--dry-run", "--subtask", "object_scaling", "--out-bddl", str(tmp_path / "bddl"), "--customs", str(tmp_path / "c.json"),
                  "--no-representative-test"])
    assert rc == 0
    assert not (tmp_path / "c.json").exists() and not (tmp_path / "bddl").exists()
    rc = GI.main(["--icl", first_demo_icl(), "--subtask", "object_scaling", "--out-bddl", str(tmp_path / "bddl"), "--customs", str(tmp_path / "c.json"),
                  "--no-representative-test", "--legacy-activity-names", "--report", str(tmp_path / "report.md")])
    assert rc == 0 and (tmp_path / "report.md").exists() and (tmp_path / "report.json").exists()
    data = json.load(open(tmp_path / "c.json"))
    assert len([k for k in data if not k.startswith("_")]) == 30
    assert P.list_problem_ids("object_scaling", str(tmp_path / "bddl")) == list(range(30))


def test_legacy_activity_names(tmp_path):
    icl = GI.ICL(first_demo_icl())
    customs = C.Customs.load(str(tmp_path / "c.json"))
    grounder = GI.Grounder(IDDLS, icl, customs, bddl_root=str(tmp_path / "bddl"),
                           options=GI.Options(representative_test=False, legacy_activity_names=True, fallbacks=False))
    grounder.run(["alignment-axial"])
    assert os.path.isdir(tmp_path / "bddl" / "world_alignment")
    n_sup = len(icl.picks["alignment-axial#01"]["support"]["representative"])
    assert P.list_problem_ids("world_alignment", str(tmp_path / "bddl")) == list(range(10 * n_sup))
    assert all(gp.activity == "world_alignment" for gp in grounder.grounded)
    assert not grounder.report.fallbacks


# ----------------------------------------------------------------------------------------------------------------
# Fallback variants (owner decision 2026-09-16 B)
# ----------------------------------------------------------------------------------------------------------------


def _fallbacks_of(grounded, variant):
    return [gp for gp in grounded.grounder.grounded if gp.entry.fallback_of == variant]


def test_fallback_entries_carry_the_schema_fields(grounded):
    data = json.load(open(grounded.customs_path))
    n = 0
    for variant, e in data.items():
        if variant.startswith("_"):
            continue
        if "fallback_of" in e:
            n += 1
            assert e["fallback_of"] in data and "fallback_of" not in data[e["fallback_of"]]  # a fallback of a primary
            assert e["fallback_rank"] >= 1 and e["fallback_kind"] in C.FALLBACK_KINDS
            assert variant in data[e["fallback_of"]]["fallbacks"]
            assert variant.startswith(e["fallback_of"].split("-")[0])
            # the registry record carries the same fields
            act = IDDLS[next(gp.subtask for gp in grounded.grounder.grounded if gp.variant == variant)].family
            reg = B.ProblemRegistry.load(os.path.join(grounded.bddl_root, act))
            rec = reg.record(variant)
            assert rec["fallback_of"] == e["fallback_of"] and rec["fallback_rank"] == e["fallback_rank"] and rec["fallback_kind"] == e["fallback_kind"]
        else:
            for fb in e.get("fallbacks", []):
                assert data[fb]["fallback_of"] == variant
            ranks = [data[fb]["fallback_rank"] for fb in e.get("fallbacks", [])]
            assert ranks == sorted(ranks)
        assert C.CustomsEntry.from_dict(variant, e).to_dict() == e
    assert n >= 10
    assert set(grounded.report.to_json()["fallbacks"]) <= set(SUBTASKS)


def test_init_surface_fallback_chain(grounded):
    under = _entries(grounded, "inter_object_relation_satisfaction-under_large_scale")
    primaries = [gp for gp in under if not gp.entry.is_fallback]
    assert primaries
    for gp in primaries:
        support = gp.entry.roles["support"]
        fbs = sorted(_fallbacks_of(grounded, gp.variant), key=lambda g: g.entry.fallback_rank)
        partner_cat = gp.entry.roles["partner"].category
        if support.category == GI.FLOOR_CATEGORY:
            assert fbs == [] and gp.entry.fallbacks == []  # floor is the last class
            continue
        if support.category == partner_cat and all(sc.bindings["support"] == sc.bindings["partner"] for sc in gp.entry.scenes.values()):
            expected = ["floor"]
        else:
            expected = ["partner", "floor"]
        assert [fb.entry.fallback_params["init_surface"] for fb in fbs] == expected
        assert [fb.entry.fallback_rank for fb in fbs] == list(range(1, len(expected) + 1))
        assert gp.entry.fallbacks == [fb.variant for fb in fbs]
        for fb in fbs:
            assert fb.entry.fallback_kind == "init_surface"
            assert fb.entry.roles["target"].to_dict() == gp.entry.roles["target"].to_dict()
            assert fb.entry.roles["partner"].to_dict() == gp.entry.roles["partner"].to_dict()
            cls = fb.entry.fallback_params["init_surface"]
            for scene, sc in fb.entry.scenes.items():
                assert sc.bindings["partner"] == gp.entry.scenes[scene].bindings["partner"]
                assert sc.whitelist == gp.entry.scenes[scene].whitelist
                if cls == "partner":
                    assert fb.variant.endswith(f"-{partner_cat}-{partner_cat}")
                    assert sc.bindings["support"] == sc.bindings["partner"]
                    assert fb.entry.roles["support"].category == partner_cat
                    assert fb.rendered.instances["support"] == fb.rendered.instances["partner"]  # declared once
                else:
                    assert fb.variant.endswith("-floor")
                    assert sc.bindings["support"] == {"room": sc.rooms[0]}
                    assert fb.entry.roles["support"].category == GI.FLOOR_CATEGORY
                    assert fb.rendered.instances["support"] == "floor.n.01_1"
    # ontop: support -> floor only (the partner's own top would satisfy the goal at init)
    ontop_root = os.path.join(grounded.root, "ontop")
    g2 = GI.Grounder(IDDLS, grounded.icl, C.Customs.load(os.path.join(ontop_root, "c.json")), bddl_root=os.path.join(ontop_root, "bddl"),
                     options=GI.Options(representative_test=False))
    g2.run(["inter_object_relation_satisfaction-ontop_large_scale"])
    for gp in [g for g in g2.grounded if not g.entry.is_fallback]:
        fbs = [g for g in g2.grounded if g.entry.fallback_of == gp.variant]
        if gp.entry.roles["support"].category == GI.FLOOR_CATEGORY:
            assert fbs == []
        else:
            assert len(fbs) == 1 and fbs[0].entry.fallback_params == {"init_surface": "floor"} and fbs[0].variant.endswith("-floor")


def test_scale_fallbacks(grounded, tmp_path):
    for st in ("alignment-world_vertical", "alignment-axial"):
        primaries = [gp for gp in _entries(grounded, st) if not gp.entry.is_fallback]
        assert primaries
        for gp in primaries:
            fbs = _fallbacks_of(grounded, gp.variant)
            assert len(fbs) == 1 and gp.entry.fallbacks == [fbs[0].variant]
            fb = fbs[0]
            s = fb.entry.fallback_params["scale"]
            assert fb.entry.fallback_kind == "scale" and 0 < s < 1
            cat = gp.entry.roles["target"].category
            height = cat in GI.HEIGHT_SCALE_FALLBACK.get(st, ())  # owner 2026-10-11: the axial paperbacks shrink their height only
            assert fb.variant == gp.variant.replace(f"-{cat}-", f"-{cat}{GI.scale_token(s, height)}-")
            assert P.VARIANT_RE.match(fb.variant)
            scale = fb.entry.roles["target"].scale
            assert ("default" not in scale and fb.entry.fallback_params["scale_axes"] == "height") if height else scale["default"] == s
            for scene_name, scene in fb.entry.scenes.items():
                for c, models in scene.whitelist["target"].items():
                    assert models and models == gp.entry.scenes[scene_name].whitelist["target"][c]  # same models as the primary ...
                    for m in models:
                        if height:  # ... the height axis (frame_long) at the fallback scale, the other axes at the primary's
                            want = [1.0, 1.0, 1.0]
                            want[grounded.grounder._height_axis(st, c, m)] = s
                            assert scale[f"{c}/{m}"] == want and fb.entry.fallback_params["height_axis"][f"{c}/{m}"] in ("x", "y")
                        else:
                            assert scale[f"{c}/{m}"] == s  # ... spawned at the fallback scale
            assert fb.entry.fallback_params["scale_source"] in ("template_meta.scale_hint", "default_step")
            assert fb.problem.objects == gp.problem.objects  # same synsets, other variant
    assert GI.scale_token(0.7) == "_s070" and GI.scale_token(0.6) == "_s060" and GI.scale_token(0.85) == "_s085"
    # the pilot's scale_hint wins over the default step; several metas -> the largest hint (closest to native)
    metas = tmp_path / "pilot" / "pod" / "v2__alignment-axial-letter-breakfast_table" / "variant" / "template"
    metas.mkdir(parents=True)
    (metas / "template_meta.json").write_text(json.dumps({"variant": "alignment-axial-letter-breakfast_table", "scale_hint": 0.8}))
    metas2 = tmp_path / "pilot" / "pod2" / "v2__alignment-axial-letter-breakfast_table" / "variant" / "template"
    metas2.mkdir(parents=True)
    (metas2 / "template_meta.json").write_text(json.dumps({"variant": "alignment-axial-letter-breakfast_table", "scale_hint": 0.6}))
    (metas2.parent / "other.json").write_text("{}")
    hints = GI.load_scale_hints(str(tmp_path / "pilot" / "*" / "v2__*" / "*" / "*" / "template_meta.json"))
    assert hints == {"alignment-axial-letter-breakfast_table": 0.8}
    g3 = GI.Grounder(IDDLS, grounded.icl, C.Customs.load(str(tmp_path / "c.json")), bddl_root=str(tmp_path / "bddl"),
                     options=GI.Options(representative_test=False, scale_hints=hints, fallback_scale=0.75))
    g3.run(["alignment-axial"])
    fbs = {g.entry.fallback_of: g for g in g3.grounded if g.entry.is_fallback}
    letter = fbs["alignment-axial-letter-breakfast_table"]
    assert letter.variant == "alignment-axial-letter_s080-breakfast_table" and letter.entry.fallback_params == {"scale": 0.8, "scale_source": "template_meta.scale_hint"}
    other = next(g for a, g in fbs.items() if a != "alignment-axial-letter-breakfast_table" and g.entry.roles["target"].category != "paperback_book")
    assert other.variant.split("-")[2].endswith("_s075") and other.entry.fallback_params["scale_source"] == "default_step"
    # a --scale-hint override wins, is persisted in the customs (later runs without the flag keep it) and supersedes
    # the earlier fallback of its kind (which keeps entry, id and BDDL but leaves the primary's live list)
    g4 = GI.Grounder(IDDLS, grounded.icl, C.Customs.load(str(tmp_path / "c.json")), bddl_root=str(tmp_path / "bddl"),
                     options=GI.Options(representative_test=False, scale_hints=hints, fallback_scale=0.75,
                                        scale_overrides={"alignment-axial-letter-breakfast_table": 0.9}))
    g4.run(["alignment-axial"])
    primary = g4.customs.entries["alignment-axial-letter-breakfast_table"]
    assert primary.fallbacks == ["alignment-axial-letter_s090-breakfast_table"]
    new = g4.customs.entries["alignment-axial-letter_s090-breakfast_table"]
    assert new.fallback_params == {"scale": 0.9, "scale_source": "override"} and new.roles["target"].scale["default"] == 0.9
    old = g4.customs.entries["alignment-axial-letter_s080-breakfast_table"]
    assert old.superseded_by == "alignment-axial-letter_s090-breakfast_table" and old.definition_id != new.definition_id and not old.is_live
    assert "superseded_by" in old.to_dict() and "superseded_by" not in new.to_dict()
    assert P.list_problem_ids("alignment", str(tmp_path / "bddl"))  # the superseded problem file stays (ids are never reused)
    g5 = GI.Grounder(IDDLS, grounded.icl, C.Customs.load(str(tmp_path / "c.json")), bddl_root=str(tmp_path / "bddl"),
                     options=GI.Options(representative_test=False, scale_hints=hints, fallback_scale=0.75))
    g5.run(["alignment-axial"])
    assert g5.customs.entries["alignment-axial-letter-breakfast_table"].fallbacks == ["alignment-axial-letter_s090-breakfast_table"]
    assert g5.customs.entries["alignment-axial-letter_s090-breakfast_table"].fallback_params["scale_source"] == "override"
    assert g5.scale_for("alignment-axial-letter-breakfast_table") == (0.9, "override")


def test_wall_deviation_defaults_and_reseed():
    # owner decision 2026-09-16 A8: delta = sign * U(15, 25) deg, template filter +-25 deg; A6 tucked two-arm stance posture
    gen = G.defaults_for("alignment-world_vertical")
    assert gen["deviation_range_deg"] == [15.0, 25.0] and gen["max_deviation_deg"] == 25.0
    assert gen["template_filter_deviation_deg"] == 25.0 and gen["stance_arm_posture"] == "tucked"
    assert G.defaults_for("alignment-axial")["deviation_range_deg"] == [20.0, 70.0]  # §6.10 unchanged
    old = {"generation": {"deviation_range_deg": [15.0, 45.0], "templates_max": 3, "legacy_key": 1}}
    kept = G.seed_generation("alignment-world_vertical", old)
    assert kept["deviation_range_deg"] == [15.0, 45.0] and kept["templates_max"] == 3 and kept["legacy_key"] == 1  # copy semantics
    reseeded = G.seed_generation("alignment-world_vertical", old, reseed=["deviation_range_deg", "legacy_key"])
    assert reseeded["deviation_range_deg"] == [15.0, 25.0] and reseeded["templates_max"] == 3 and "legacy_key" not in reseeded
    assert G.seed_generation("alignment-world_vertical", old, reseed=["*"]) == G.defaults_for("alignment-world_vertical")


def test_next_model_fallbacks(grounded):
    small = _entries(grounded, "configuration_matching-articulation_open_small_scale")
    primaries = [gp for gp in small if not gp.entry.is_fallback]
    assert {gp.entry.roles["target"].category for gp in primaries} == {"blender", "hinged_jar", "laptop", "toolbox", "carton"}
    for gp in primaries:
        cat = gp.entry.roles["target"].category
        fbs = list({g.variant: g for g in _fallbacks_of(grounded, gp.variant)}.values())  # one variant per support triplet (the floor merges them)
        slots = grounded.icl.picks[gp.combination.pick.split("~")[0]]["target"]
        others = [r for s in GI.NEXT_MODEL_SLOT_ORDER for r in slots.get(s, []) if r["model"] != gp.entry.roles["target"].representative]
        if not others:
            assert fbs == [] and gp.entry.fallbacks == []
            assert any(gp.variant in n and "no next-model fallback" in n for n in grounded.report.notes[gp.subtask])
            continue
        assert len(fbs) == 1
        fb = fbs[0]
        model = fb.entry.fallback_params["model"]
        assert fb.entry.fallback_kind == "next_model" and model == others[0]["model"]
        assert fb.entry.fallback_params["primary_model"] == gp.entry.roles["target"].representative
        assert fb.variant.split("-")[2] == f"{cat}_{model}"
        assert fb.entry.roles["target"].representative == model
        for sc in fb.entry.scenes.values():
            assert sc.whitelist["target"] == {cat: [model]}
        assert fb.entry.roles["target"].joint[f"{cat}/{model}"] and fb.entry.roles["target"].handle_key[f"{cat}/{model}"]
        # the support follows the §5.4 floor rule of the new model
        tall = float(fb.combination.roles["target"].pick_row["bbox_z"]) > GI.FLOOR_SUPPORT_HEIGHT_M
        assert fb.variant.endswith("-floor") == tall


def test_pilot_blacklist_gives_the_next_model_fallback(tmp_path):
    """Owner decision 2026-09-16 item 5: a pick the pilot rejected (hanger unstable_init) is blacklisted with the pilot
    reason in every scene block, the primary keeps its variant with an empty whitelist and gets the next model of the
    category as its rank-1 fallback; the blacklist and the fallback survive a re-run without the flag."""
    icl = GI.ICL(first_demo_icl())
    bl = {("alignment-world_vertical", "target", "hanger", "agrpio"): "unstable_init"}
    g = GI.Grounder(IDDLS, icl, C.Customs.load(str(tmp_path / "c.json")), bddl_root=str(tmp_path / "bddl"),
                    options=GI.Options(representative_test=False, blacklist=bl))
    g.run(["alignment-world_vertical"])
    for st, model, nxt in (("alignment-world_vertical", "agrpio", "nagryh"),):
        primary = g.customs.entries[f"{st}-hanger-walls"]
        sc = primary.scenes["house_single_floor"]
        assert not sc.whitelist.get("target") and sc.blacklist["target"]["hanger"][model] == f"pilot {g.today}: unstable_init"
        assert primary.fallbacks == [f"{st}-hanger_{nxt}-walls"]
        fb = g.customs.entries[f"{st}-hanger_{nxt}-walls"]
        assert fb.fallback_kind == "next_model" and fb.fallback_rank == 1 and fb.fallback_params["primary_model"] == model
        assert fb.scenes["house_single_floor"].whitelist["target"] == {"hanger": [nxt]} and fb.roles["target"].representative == nxt
        assert f"{st}-hanger_s070-walls" not in g.customs.entries  # no scaled variant of a model without a whitelist
        other = g.customs.entries[f"{st}-picture-walls"]  # other picks keep their scale fallback
        assert other.fallbacks and g.customs.entries[other.fallbacks[0]].fallback_kind == "scale"
    # persisted: a re-run without --blacklist keeps the pilot rejection and the next-model fallback
    g2 = GI.Grounder(IDDLS, icl, C.Customs.load(str(tmp_path / "c.json")), bddl_root=str(tmp_path / "bddl"), options=GI.Options(representative_test=False))
    g2.run(["alignment-world_vertical"])
    primary = g2.customs.entries["alignment-world_vertical-hanger-walls"]
    assert not primary.scenes["house_single_floor"].whitelist.get("target") and primary.fallbacks == ["alignment-world_vertical-hanger_nagryh-walls"]
    assert g2.customs.entries["alignment-world_vertical-hanger-walls"].to_dict() == g.customs.entries["alignment-world_vertical-hanger-walls"].to_dict()


def test_no_fallbacks_option(tmp_path):
    g = GI.Grounder(IDDLS, GI.ICL(first_demo_icl()), C.Customs.load(str(tmp_path / "c.json")), bddl_root=str(tmp_path / "bddl"),
                    options=GI.Options(representative_test=False, fallbacks=False))
    g.run(["inter_object_relation_satisfaction-under_large_scale", "alignment-axial"])
    assert g.grounded and not any(gp.entry.is_fallback for gp in g.grounded) and not g.report.fallbacks
    assert all(not gp.entry.fallbacks for gp in g.grounded)


def test_several_representative_supports_per_pick(tmp_path):
    """A3 support triplets: several support rows in the representative slot of one pick -> one variant per support
    category (scene per support), same-category supports share the variant and union their scene blocks."""
    icl = GI.ICL(first_demo_icl())
    pick = "inter_object_relation_satisfaction-nextto#01"
    base = icl.picks[pick]["support"]["representative"][0]
    extra = [
        ("countertop", "house_single_floor", "countertop_kelker_0"),
        ("bar", "house_single_floor", "bar_udatjt_0"),
        ("commercial_kitchen_table", "restaurant_diner", "commercial_kitchen_table_vxvtec_0"),
        ("conference_table", "office_cubicles_right", "conference_table_fpkmyw_0"),
        ("desk", "office_cubicles_right", "desk_mdhelw_1"),
        ("bar", "restaurant_diner", "bar_xxftww_0"),
    ]
    for cat, scene, name in extra:
        icl.picks[pick]["support"]["representative"].append(dict(base, category=cat, scene=scene, name=name, model=name.rsplit("_", 2)[1]))
    g = GI.Grounder(IDDLS, icl, C.Customs.load(str(tmp_path / "c.json")), bddl_root=str(tmp_path / "bddl"), options=GI.Options(representative_test=False))
    g.run(["inter_object_relation_satisfaction-nextto_small_scale"])
    variants = {gp.variant for gp in g.grounded if gp.combination.pick.startswith(pick)}
    tokens = {a.split("-")[-1] for a in variants}
    assert tokens == {"breakfast_table", "countertop", "bar", "commercial_kitchen_table", "conference_table", "desk"}
    bar = g.customs.entries["inter_object_relation_satisfaction-nextto_small_scale-pencil-ruler-bar"]
    assert set(bar.scenes) == {"house_single_floor", "restaurant_diner"}
    assert bar.scenes["house_single_floor"].bindings["support"] == ["bar_udatjt_0"] and bar.scenes["restaurant_diner"].bindings["support"] == ["bar_xxftww_0"]
    assert bar.scenes["house_single_floor"].rooms == ["kitchen_0"] and set(bar.scenes["restaurant_diner"].rooms) == {"bar_0", "dining_room_0"}
    for sc in bar.scenes.values():
        assert sc.whitelist["target"] == {"pencil": ["hmawzy"]} and sc.bindings["floor"] == {"room": sc.rooms[0]}
    # the desk variant also carries the other picks' representative desk? no: only the supports of the selection
    desk = g.customs.entries["inter_object_relation_satisfaction-nextto_small_scale-pencil-ruler-desk"]
    assert desk.scenes["office_cubicles_right"].bindings["support"] == ["desk_mdhelw_1"]
    ids = P.list_problem_ids("inter_object_relation_satisfaction", str(tmp_path / "bddl"))
    assert len(ids) == len({gp.variant for gp in g.grounded})
    assert P.validate_problem_registry("inter_object_relation_satisfaction", str(tmp_path / "bddl"))


def test_support_surface_from_the_sheet_note():
    row = {"scene": "house_double_floor_upper", "name": "bookcase_otwukr_2", "note": "init=inside; top_z=0.73 (shelf) area=0.20 best_edge=-y"}
    assert GI.support_surface(row, "p#02~T2") == {"name": "bookcase_otwukr_2", "init": "inside", "top_z": 0.73, "row": "p#02~T2"}
    assert GI.support_surface(dict(row, note="init=ontop; top_z=1.07 area=0.23"), "p#02~T1")["init"] == "ontop"
    assert GI.support_surface(dict(row, note="living-room table"), "p") is None
    # owner 2026-10-11 06:27: the sheet heights of these supports match no surface -> their top
    for scene, name in GI.SUPPORT_TOP_SURFACE:
        s = GI.support_surface({"scene": scene, "name": name, "note": "init=ontop; top_z=1.02 area=0.64"}, "p")
        assert s["surface"] == "top" and s["top_z"] == 1.02


def test_inside_init_rewrites_only_the_support_literals():
    assert B.rewrite_init(["ontop", "target", "support"], {("ontop", "support"): "inside"}) == ["inside", "target", "support"]
    assert B.rewrite_init(["ontop", "agent", "floor"], {("ontop", "support"): "inside"}) == ["ontop", "agent", "floor"]
    assert B.rewrite_init(["not", ["touching", "target", "partner"]], {("ontop", "support"): "inside"}) == ["not", ["touching", "target", "partner"]]
    assert B.rewrite_init(["ontop", "target", "support"], None) == ["ontop", "target", "support"]


def test_inside_support_rows_get_their_own_variant_bddl_and_surface(tmp_path):
    """Sheet notes ``init=`` / ``top_z=`` (owner 2026-10-11 item 2): an ``init=inside`` support row grounds a variant of its own
    (support token ``<category>_inside``) whose BDDL starts the objects ``inside`` the support; every row's surface goes to
    ``scenes.<scene>.bindings.support_surfaces`` and a re-grounding replaces its own entry."""
    icl = GI.ICL(first_demo_icl())
    pick = "inter_object_relation_satisfaction-nextto#01"
    rows = icl.picks[pick]["support"]["representative"]
    base = rows[0]
    rows[0] = dict(base, note="init=ontop; top_z=0.81 area=3.38", triplet="1")
    rows.append(dict(base, note="init=inside; top_z=0.40 (shelf)", triplet="2"))
    for _ in range(2):  # the second run re-grounds the same rows into the same customs
        g = GI.Grounder(IDDLS, icl, C.Customs.load(str(tmp_path / "c.json")), bddl_root=str(tmp_path / "bddl"), options=GI.Options(representative_test=False))
        g.run(["inter_object_relation_satisfaction-nextto_small_scale"])
        g.customs.save()
    by_variant = {gp.variant: gp for gp in g.grounded if gp.combination.pick.startswith(pick)}
    cat = base["category"]
    ontop = next(a for a in by_variant if a.endswith(f"-{cat}"))
    inside = next(a for a in by_variant if a.endswith(f"-{cat}_inside"))
    t, p, s = (by_variant[inside].rendered.instances[r] for r in ("target", "partner", "support"))
    assert ["inside", t, s] in by_variant[inside].problem.init and ["inside", p, s] in by_variant[inside].problem.init
    assert not any(lit[0] == "ontop" and lit[-1] == s for lit in by_variant[inside].problem.init)
    t, p, s = (by_variant[ontop].rendered.instances[r] for r in ("target", "partner", "support"))
    assert ["ontop", t, s] in by_variant[ontop].problem.init and not any(lit[0] == "inside" for lit in by_variant[ontop].problem.init)
    scene = base["scene"]
    srf_in = g.customs.entries[inside].scenes[scene].bindings["support_surfaces"]
    srf_on = g.customs.entries[ontop].scenes[scene].bindings["support_surfaces"]
    assert [(e["name"], e["init"], e["top_z"]) for e in srf_in] == [(base["name"], "inside", 0.40)]
    assert [(e["name"], e["init"], e["top_z"]) for e in srf_on] == [(base["name"], "ontop", 0.81)]
    assert P.validate_problem_registry("inter_object_relation_satisfaction", str(tmp_path / "bddl"))


def test_hyphenated_synset_instances_parse():
    """BEHAVIOR synsets may hold a hyphen (``deep-freeze.n.02``; ``mock-up.n.01``, the category of the 2026-10-11
    mental-rotation block): their instances are valid names; empty or doubled hyphen segments are not."""
    for ok in ("mock-up.n.01_1", "deep-freeze.n.02_12", "t-shirt.n.01_1", "mug.n.04_1"):
        assert P.INSTANCE_RE.match(ok), ok
    for bad in ("-mock.n.01_1", "mock-.n.01_1", "mock--up.n.01_1", "mock-up.n.1_1", "Mock-up.n.01_1"):
        assert not P.INSTANCE_RE.match(bad), bad


def test_custom_asset_without_a_candidate_row_gets_its_dataset(tmp_path):
    """A sheet model without a ``candidates.csv`` row that is one of the repo's own assets is grounded with the
    ``ispatialgym-assets`` dataset (the sampler loads the model from the dataset of its customs role)."""
    assert GI._is_custom_asset("toy_dice", "isgblk") and not GI._is_custom_asset("toy_dice", "nosuchmodel") and not GI._is_custom_asset("apple", None)
    g = GI.Grounder(IDDLS, GI.ICL(first_demo_icl()), C.Customs.load(str(tmp_path / "c.json")), bddl_root=str(tmp_path / "bddl"),
                    options=GI.Options(representative_test=False))
    row = {"subtask": "mental_rotation", "role": "target", "kind": "sampleable", "category": "toy_dice", "model": "isgblk"}
    merged = g._resolve_row("mental_rotation", "target", row)
    assert merged["_synthetic"] == "1" and merged["dataset"] == GI.CUSTOM_DATASET
    assert not g._resolve_row("mental_rotation", "target", dict(row, model="nosuchmodel")).get("dataset")


def test_scene_entry_keeps_underscore_comments():
    """A scene block's hand-written ``_``-keys (the sandal variants' ``_note``) survive a load / write round trip; other
    unknown keys are still errors."""
    raw = {"rooms": ["kitchen_0"], "bindings": {}, "whitelist": {}, "blacklist": {}, "_note": "the 0.15 m drop, the deeper trunk first"}
    se = C.SceneEntry.from_dict(raw)
    assert se.notes == {"_note": raw["_note"]} and se.to_dict() == raw
    with pytest.raises(C.CustomsError):
        C.SceneEntry.from_dict(dict(raw, note="no underscore"))


def test_grasp_key_preference_named_then_pick_suffix_then_scaled_floating():
    """The key a model's candidates row names wins when admissible; else the pick's suffix (touching#03 __bu); else a spawn-scaled
    floating key (__sf, owner 2026-10-11: the thickness-scaled teaspoon); else the grounder's default (alphabetical first)."""
    keys = {"teaspoon__eilfij", "teaspoon__eilfij__sf", "hardback__acbrnv__bo", "hardback__acbrnv__bu"}
    fake = types.SimpleNamespace(icl=types.SimpleNamespace(admissible_grasps=lambda c, m, p: [{"key": k} for k in sorted(keys) if k.startswith(f"{c}__{m}")]))
    iddl = types.SimpleNamespace(require_of=lambda role: [types.SimpleNamespace(kind="grasps", pipeline="floating")])
    pref = lambda c, m, pick, row=None: GI.Grounder._preferred_grasp_key(fake, iddl, "target", c, m, types.SimpleNamespace(pick_row={"pick": pick}),
                                                                         sorted(k for k in keys if k.startswith(f"{c}__{m}"))[0], row)
    assert pref("teaspoon", "eilfij", "mental_rotation#09") == "teaspoon__eilfij__sf"
    # the sheet names the unscaled key (mental_rotation#03 row): its spawn-scaled floating key outranks it
    assert pref("teaspoon", "eilfij", "mental_rotation#09", {"grasp_key": "teaspoon__eilfij"}) == "teaspoon__eilfij__sf"
    assert pref("hardback", "acbrnv", "inter_object_relation_satisfaction-touching#01", {"grasp_key": "hardback__acbrnv__bu"}) == "hardback__acbrnv__bu"
    assert pref("hardback", "acbrnv", "inter_object_relation_satisfaction-touching#03") == "hardback__acbrnv__bu"
    assert pref("hardback", "acbrnv", "inter_object_relation_satisfaction-touching#01") == "hardback__acbrnv__bo"
    assert pref("teaspoon", "eilfij", "x", {"grasp_key": "teaspoon__eilfij__missing"}) == "teaspoon__eilfij__sf"  # a named key that is not admissible is ignored


# ----------------------------------------------------------------------------------------------------------------
# the shipped representative sheet (the owner's of 2026-10-11): disembedding pools, poses and the puzzle's layout (handoff
# 2026-10-10) and the axial paperbacks' height-only scale fallback (owner 2026-10-11)
# ----------------------------------------------------------------------------------------------------------------

PUZZLE_LAYOUT = "ispatialgym/assets/objects/embedded_figure/model/assembly.json"


def _ground_shipped(root: str, subtasks, tray_budget=None, **options):
    """Ground ``subtasks`` from the shipped icl into ``root`` (``tray_budget``: the disembedding default for this run only)."""
    with pytest.MonkeyPatch.context() as mp:
        if tray_budget is not None:
            mp.setitem(G.DEFAULTS["disembedding"], "tray_budget", tray_budget)
        g = GI.Grounder(IDDLS, GI.ICL(), C.Customs.load(os.path.join(root, "c.json")), bddl_root=os.path.join(root, "bddl"),
                        options=GI.Options(representative_test=False, **options))
        g.run(list(subtasks))
        g.customs.save()
    return g


def test_sheet_distractor_helpers():
    assert GI.footprint_dims((0.084, 0.123, 0.181)) == (0.123, 0.084)  # canonical: the bbox's x / y
    assert GI.footprint_dims((0.084, 0.123, 0.181), "lying") == (0.181, 0.123)  # long axis horizontal, on the broadest side
    assert GI.footprint_dims((0.154, 0.051, 0.045), "standing") == (0.051, 0.045)  # long axis vertical
    group_of = {("a", "1"): "g1", ("a", "2"): "g1"}
    rule = lambda d: GI.admissible_draw(d, "a", 2, 1, group_of)  # noqa: E731
    assert rule([("a", "1"), ("a", "3"), ("b", "1")])
    assert not rule([("a", "1"), ("a", "2")])  # one model per look-alike group
    assert not rule([("a", "1"), ("a", "3"), ("a", "4")])  # at most 2 of the target's category
    assert not rule([("b", "1"), ("b", "2")])  # at most 1 of each other category
    pool = [("a", "1"), ("a", "2"), ("a", "3"), ("b", "1"), ("c", "1")]
    area = dict.fromkeys(pool, 0.01)
    area[("c", "1")] = 0.05
    n, worst, draws = GI.count_every_draw(pool, area, 0.065, 3, rule)
    assert n == 2 and worst == pytest.approx(0.06) and draws and all(rule(d) and len(d) == 2 for d in draws)  # a draw of 3 with c: 0.07
    assert GI.count_every_draw(pool, area, 0.1, 3, rule)[0] == 3
    assert GI.count_every_draw(pool, area, 0.04, 3, rule)[0] == 0  # c alone does not fit: not every draw of 1 fits
    key = "disembedding#04|bottle_of_milk/mrejrs|countertop"
    assert GI.stable_index(key, 90) == GI.stable_index(key, 90) and 0 <= GI.stable_index(key, 7) < 7


def test_sheet_disembedding_pools_draw_rule_budget_and_poses(tmp_path):
    """Handoff 2026-10-10 (its docs/disembedding_distractors.md): at the owner's budget 0.85 every pool pick of the sheet gets 3
    distractors - every admissible draw fits, the largest takes 81 % of the uekqey floor on #04 (lying milk family), 73 % on #10
    (fork), at most 56 % elsewhere; at the former 0.60 #04 and #10 only get 2. Each variant's member set is one admissible draw of
    the pick's same_category / other_category rows (<= 2 of the target's category, <= 1 of each other category), its whitelist the
    pick's pool of those categories (no target model, held-out model or look-alike of the target), and #04 / #06 carry the target's
    pose; a re-grounding draws the same member sets."""
    icl = GI.ICL()
    g = _ground_shipped(str(tmp_path / "a"), ["disembedding"], 0.85)
    by_pick = collections.defaultdict(list)
    for gp in g.grounded:
        by_pick[gp.combination.pick.split("~")[0]].append(gp)
    assert set(by_pick) >= {f"disembedding#{i:02d}" for i in range(2, 11)}
    for pick, gps in by_pick.items():
        if pick == "disembedding#01":
            continue  # the puzzle: test_sheet_disembedding_puzzle_fixed_layout
        rows = icl.picks[pick]["distractors"]
        pool = {(r["category"], r["model"]) for s in GI.SHEET_POOL_SLOTS for r in rows.get(s, [])}
        tcat, tmodel = gps[0].entry.roles["target"].category, gps[0].entry.roles["target"].representative
        for gp in gps:
            gen = gp.entry.generation
            members = gen["groups"]["distractors"]["members"]
            b = gen["distractor_budget"]
            assert b["n_distractors"] == gen["groups"]["distractors"]["size"] == sum(members.values()) == 3
            assert b["budget_fraction"] == 0.85 and b["count_rule"] == "every admissible draw fits" and b["n_draws"] >= 1
            assert members.get(tcat, 0) <= 2 and all(n == 1 for c, n in members.items() if c != tcat)
            assert collections.Counter(d.split("/")[0] for d in b["draw"]) == collections.Counter(members)
            assert gp.variant.split("-")[2] == "_".join(f"{c}_{n}" for c, n in members.items())
            assert len(gp.problem.groups["distractors_group.n.01_1"]) == 3
            for sc in gp.entry.scenes.values():
                wl = {(c, m) for c, ms in sc.whitelist["distractors"].items() for m in ms}
                assert wl and wl <= pool and set(sc.whitelist["distractors"]) == set(members)
                assert not wl & ({(tcat, tmodel)} | icl.heldout_models | icl.lookalikes(tcat, tmodel))
            assert gen.get("distractor_pose") == {"disembedding#04": "lying", "disembedding#06": "standing"}.get(pick)
            assert "fixed_layout" not in gen
        share = max(gp.entry.generation["distractor_budget"]["worst_draw_m2"] / gp.entry.generation["distractor_budget"]["tray_floor_m2"] for gp in gps)
        expected = {"disembedding#04": 0.81, "disembedding#10": 0.73}.get(pick)
        assert share == pytest.approx(expected, abs=0.006) if expected else share <= 0.565
    again = _ground_shipped(str(tmp_path / "a"), ["disembedding"], 0.85)
    assert sorted(gp.variant for gp in again.grounded) == sorted(gp.variant for gp in g.grounded)
    old = _ground_shipped(str(tmp_path / "b"), ["disembedding"], 0.60)
    n = {gp.combination.pick.split("~")[0]: gp.entry.generation["distractor_budget"]["n_distractors"] for gp in old.grounded
         if not gp.combination.pick.startswith("disembedding#01")}
    assert {p for p, k in n.items() if k == 2} == {"disembedding#04", "disembedding#10"} and set(n.values()) == {2, 3}


def test_sheet_disembedding_puzzle_fixed_layout(tmp_path, monkeypatch):
    """The puzzle pick (disembedding#01, handoff 2026-10-10): its three fixed_layout parts are the variant's distractors - all of them,
    not drawn, outside the budget - and generation.fixed_layout names the layout file of their notes (sha256 of the grounded
    version); a checkout without the file skips the pick (fixed_layout)."""
    if not os.path.exists(os.path.join(GI.REPO_ROOT, PUZZLE_LAYOUT)):
        g0 = _ground_shipped(str(tmp_path / "none"), ["disembedding"])
        assert any(lab.startswith("disembedding#01") and why.startswith("fixed_layout") for lab, why in g0.report.skipped["disembedding"])
    path = tmp_path / "repo" / PUZZLE_LAYOUT
    path.parent.mkdir(parents=True)
    parts = {"isgefa": ("target", [-0.03065, 0.057516, 0.0]), "isgefb": ("context_frame", [0.0, 0.0, 0.0]),
             "isgefc": ("context_inner_A", [-0.065081, 0.037741, 0.0]), "isgefd": ("context_inner_B", [-0.017335, 0.051505, 0.0])}
    path.write_text(json.dumps({"target": "isgefa", "parts": {m: {"category": "jigsaw_puzzle_piece", "role": r, "origin_in_figure_m": o} for m, (r, o) in parts.items()}}))
    monkeypatch.setattr(GI, "REPO_ROOT", str(tmp_path / "repo"))
    g = _ground_shipped(str(tmp_path / "with"), ["disembedding"])
    puzzle = [gp for gp in g.grounded if gp.combination.pick.startswith("disembedding#01")]
    assert puzzle
    for gp in puzzle:
        gen = gp.entry.generation
        assert gp.variant.split("-")[1:3] == ["jigsaw_puzzle_piece", "jigsaw_puzzle_piece_3"]
        assert gen["groups"]["distractors"]["members"] == {"jigsaw_puzzle_piece": 3}
        assert gen["fixed_layout"] == {"file": PUZZLE_LAYOUT, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "target": "jigsaw_puzzle_piece/isgefa",
                                       "parts": ["jigsaw_puzzle_piece/isgefb", "jigsaw_puzzle_piece/isgefc", "jigsaw_puzzle_piece/isgefd"]}
        assert gen["distractor_budget"]["fixed_layout"] == PUZZLE_LAYOUT and "budget_fraction" not in gen["distractor_budget"]
        assert "distractor_pose" not in gen
        for sc in gp.entry.scenes.values():
            assert sc.whitelist["distractors"] == {"jigsaw_puzzle_piece": ["isgefb", "isgefc", "isgefd"]}
        assert len(gp.problem.groups["distractors_group.n.01_1"]) == 3


def test_axial_paperback_height_only_scale_fallback(tmp_path):
    """Owner 2026-10-11: an alignment-axial paperback that fails falls back to its height scaled - the cover's long side, model x
    (frame_long) - never to a uniform scale; the sheet's spawn scale ([1, 1, 2.5]: the 2 cm thickness) stays on the other axes. The
    other axial targets keep the uniform _s070 fallback. A re-grounding, and a variant with several scene blocks, scale once."""
    for _ in range(2):
        g = _ground_shipped(str(tmp_path), ["alignment-axial"])
        entries = g.customs.entries
        papers = [a for a, e in entries.items() if not e.is_fallback and e.roles["target"].category == "paperback_book"]
        assert papers
        for a in papers:
            primary = entries[a]
            assert primary.roles["target"].scale == {"paperback_book/jhaltj": [1.0, 1.0, 2.5]}  # the sheet's scale column
            assert primary.fallbacks == [a.replace("-paperback_book-", "-paperback_book_h070-")]
            fb = entries[primary.fallbacks[0]]
            assert fb.fallback_kind == "scale" and fb.fallback_params == {"scale": 0.7, "scale_source": "default_step", "scale_axes": "height",
                                                                          "height_axis": {"paperback_book/jhaltj": "x"}}
            assert fb.roles["target"].scale == {"paperback_book/jhaltj": [0.7, 1.0, 2.5]}
        others = [e for e in entries.values() if e.is_fallback and e.roles["target"].category != "paperback_book"]
        assert others and all(e.fallback_params == {"scale": 0.7, "scale_source": "default_step"} and e.roles["target"].scale["default"] == 0.7 for e in others)
    assert any(len(entries[a].scenes) > 1 for a in papers)
    assert GI.scale_token(0.7, height_only=True) == "_h070" and GI.scale_token(0.7) == "_s070"
    # a pilot scale_hint is a uniform scale: not the height fallback's (an explicit override still is)
    g2 = GI.Grounder(IDDLS, GI.ICL(), C.Customs.load(str(tmp_path / "c2.json")), bddl_root=str(tmp_path / "bddl2"),
                     options=GI.Options(representative_test=False, scale_hints={papers[0]: 0.8}, scale_overrides={papers[-1]: 0.85}))
    assert g2.scale_for(papers[0], height_only=True) == (0.7, "default_step") and g2.scale_for(papers[0]) == (0.8, "template_meta.scale_hint")
    assert g2.scale_for(papers[-1], height_only=True) == (0.85, "override")
