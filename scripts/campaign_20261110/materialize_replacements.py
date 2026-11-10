#!/usr/bin/env python3
"""Compile isolated replacement customs and BDDL without altering source assets.

The source model and generation settings come from frozen f24ea424 customs.
Outputs are validation candidates, never released collection templates.
"""
from __future__ import annotations

import copy
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
B1K = Path("/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(B1K / "bddl3"))
from ispatialgym.sampling.bddl_from_iddl import RoleGrounding, render  # noqa: E402
from ispatialgym.sampling.iddl import parse_file  # noqa: E402
from ispatialgym.sampling.knowledge import synset_of_category  # noqa: E402
from ispatialgym.sampling.bddl_writer import Bindings  # noqa: E402
from ispatialgym.oracle.sidecar import subtask_of_variant  # noqa: E402
from ispatialgym.sampling.v2.problem import GroundedProblem, SUBTASK_FAMILY  # noqa: E402
from readiness_catalog import ReadinessCatalog  # noqa: E402
from contracts import ObjectRef, Template, digest as record_digest, task_for_alias  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'tmp/campaign_20261110/materialized')
    args=parser.parse_args()
    inv = ROOT / "tmp/campaign_20261110/inventory"
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    customs_path = inv / "frozen_customs_f24ea424.json"
    source_customs = json.loads(customs_path.read_text())
    records = {r["key"]: r for r in json.loads((inv / "cluster_templates_with_scales.json").read_text())["records"]}
    reservations = json.loads((inv / "split_reservations.json").read_text())
    excluded_models = set(reservations["train_exclude_models"])
    excluded_categories = set(reservations["train_exclude_categories"])
    readiness = ReadinessCatalog(B1K / "bddl3/bddl/generated_data")
    definitions, jobs = {}, []
    scene_cache = {}

    def scene_objects(scene):
        if scene not in scene_cache:
            path = B1K / "datasets/behavior-1k-assets/scenes" / scene / "json" / f"{scene}_best.json"
            scene_cache[scene] = (json.loads(path.read_text())["objects_info"]["init_info"], digest(path))
        return scene_cache[scene]

    def canonical_alias(alias):
        for name in ("inside", "nextto", "touching"):
            alias = alias.replace(f"satisfaction-{name}-", f"satisfaction-{name}_small_scale-")
        return alias

    def make(class_id, *, scene=None, room=None, support=None, support_z=None, partner=None, source_alias=None,
             role_category=None, gen_updates=None, remove_gen=(), distractors=None, region=None, blockers=()):
        record = records[class_id]
        meta = record["metadata"]
        old_alias = source_alias or canonical_alias(record["row"]["alias"])
        entry = copy.deepcopy(source_customs[old_alias])
        iddl = parse_file(str(ROOT / "ispatialgym/assets/iddl" / entry["iddl"]))
        original_scene = meta["scene"]
        original_block = entry["scenes"].get(original_scene) or next(iter(entry["scenes"].values()))
        scene = scene or original_scene
        room = room or meta["task_rooms"][0]
        block = copy.deepcopy(original_block)
        block["rooms"] = [room]
        block["bindings"] = copy.deepcopy(meta["choice"]["bindings"])
        block["bindings"]["floor"] = {"room": room}
        block["blacklist"] = {}
        entry["scenes"] = {scene: block}
        entry["constraints"] = []  # source learned geometry belongs to the source binding
        for key in list(entry):
            if key.startswith("fallback") or key == "superseded_by":
                entry.pop(key)
        entry["generation"]["templates_max"] = 1
        entry["generation"]["instances_per_template"] = 1
        for key in list(entry["generation"]):
            if key.startswith("_"):
                entry["generation"].pop(key)
        for key in remove_gen:
            entry["generation"].pop(key, None)
        if scene != original_scene:
            entry["generation"].pop("remove_objects", None)
        if support:
            category = support.rsplit("_", 2)[0]
            entry["roles"]["support"]["category"] = category
            block["bindings"]["support"] = [support]
            block["bindings"].pop("support_regions", None)
            if support_z is not None:
                block["bindings"]["support_surfaces"] = [{"name": support, "init": "ontop", "top_z": support_z,
                    "row": class_id + "~campaign"}]
        if partner:
            block["bindings"]["partner"] = [partner]
        if role_category:
            for role, category in role_category.items():
                entry["roles"][role]["category"] = category
        if region:
            block["bindings"].pop("support_surfaces", None)
            block["bindings"]["support_regions"] = [region]
        entry["generation"].update(gen_updates or {})
        # Freeze every source-spawned role to the actual template models/scales.
        # An explicit alternate source alias (blender cwkvib) keeps that alias's
        # reviewed model, joint and handle settings instead of old eyedvd values.
        role_instances = {}
        for binding in meta["bindings"]:
            role_instances.setdefault(binding["role"], []).append(binding)
        block["whitelist"] = {}
        for role, spec in entry["roles"].items():
            if not spec.get("sampleable") or role == "distractors" and distractors is not None:
                continue
            if source_alias and role == "target":
                category, model = spec["category"], spec["representative"]
                block["whitelist"][role] = {category: [model]}
                continue
            bindings = role_instances.get(role, [])
            if not bindings:
                raise ValueError(f"no source binding for {class_id}/{role}")
            pool = {}
            scale = {}
            for binding in bindings:
                category, model = binding["category"], binding["model"]
                pool.setdefault(category, []).append(model)
                scale[category + "/" + model] = record["object_init_args"][binding["scene_object_name"]]["scale"]
            block["whitelist"][role] = {c: sorted(set(ms)) for c, ms in pool.items()}
            spec["scale"] = scale
            if len(bindings) == 1:
                spec["representative"] = bindings[0]["model"]
        if distractors is not None:
            pool, members = {}, {}
            for category, model in distractors:
                pool.setdefault(category, []).append(model)
                members[category] = members.get(category, 0) + 1
            block["whitelist"]["distractors"] = pool
            entry["generation"]["groups"]["distractors"] = {"size": len(distractors), "draw": "unique", "members": members}
            budget = entry["generation"].get("distractor_budget", {})
            budget["draw"] = [c + "/" + m for c, m in distractors]
            budget["n_distractors"] = len(distractors)
            budget["used_m2"] = sum(budget["model_footprints_m2"][c + "/" + m] for c, m in distractors)
            budget["worst_draw_m2"] = budget["used_m2"]
            budget["n_draws"] = 1
            budget["count_rule"] = "fixed admitted subset of the unchanged category pool; fresh geometry fit required"
        entry["campaign_provenance"] = {"source_class": class_id, "source_alias": old_alias,
            "source_customs_sha256": digest(customs_path), "status": "validation_candidate",
            "required_before_release": ["actual runtime ready/hash audit", "fresh geometry fit", "fresh demo", "fresh open-loop replay"],
            "geometry_probe_blockers": list(blockers)}
        for role, pool in block["whitelist"].items():
            spec = entry["roles"][role]
            spec.setdefault("scale", {})
            for category, models in pool.items():
                for model in models:
                    key = category + "/" + model
                    if key not in spec["scale"]:
                        actual = [r["object_init_args"][b["scene_object_name"]]["scale"]
                                  for r in records.values() for b in r["metadata"]["bindings"]
                                  if b["category"] == category and b["model"] == model and b["role"] == role]
                        if not actual or any(a != actual[0] for a in actual):
                            raise ValueError(f"missing or ambiguous actual default scale for {key}/{role}")
                        spec["scale"][key] = actual[0]
        # Resolve exact ready scene objects, including task room floor, from the
        # actual stable-scene initialization (record the untouched scene scale).
        scene_info, scene_sha = scene_objects(scene)
        expected = []
        def admit(role, category, model, init=None):
            status = readiness.status(category, model)
            if not status["ready"]:
                raise ValueError(f"{class_id}: unready {role} {category}/{model}")
            if category not in {"floors", "walls"} and (category + "/" + model in excluded_models or category in excluded_categories):
                raise ValueError(f"{class_id}: heldout asset {category}/{model}")
            expected.append({"role": role, "category": category, "model": model, "ready": True,
                "readiness_source": status["readiness_source"], "scene_init_args": init,
                "default_scale": init["scale"] if init else entry["roles"][role]["scale"][category + "/" + model],
                "asset_sha256": digest(B1K / "datasets/behavior-1k-assets/objects" / category / model / "usd" / (model + ".encrypted.usd"))})
        for role, pool in block["whitelist"].items():
            for category, models in pool.items():
                for model in models:
                    admit(role, category, model)
        for role, spec in entry["roles"].items():
            if spec.get("sampleable"):
                continue
            binding = block["bindings"].get(role)
            if isinstance(binding, list):
                for name in binding:
                    init = scene_info[name]["args"]
                    if room not in (init.get("in_rooms") or []) and not (region and init["category"] == "walls" and region["wall"] == name and region["room"] == room):
                        raise ValueError(f"{name} not in requested room {room}: {init.get('in_rooms')}")
                    admit(role, init["category"], init["model"], init)
            elif isinstance(binding, dict) and binding.get("room"):
                floors = [v["args"] for v in scene_info.values() if v["args"].get("category") == "floors" and room in (v["args"].get("in_rooms") or [])]
                if not floors:
                    raise ValueError(f"no floor for {scene}/{room}")
                for init in floors:
                    admit(role, "floors", init["model"], init)
        # Native binding counters follow spawn order. In particular the source
        # tray must precede destination, as in the rendered IDDL. JSON key
        # sorting reverses these roles and silently inverts every tray predicate.
        entry["roles"] = {r: entry["roles"][r] for r in iddl.roles if r in entry["roles"]} | {
            r: s for r, s in entry["roles"].items() if r not in iddl.roles}
        tokens = {}
        for role in iddl.variant_slots:
            group = entry["generation"].get("groups", {}).get(role)
            tokens[role] = "_".join(f"{c}_{n}" for c, n in group["members"].items()) if group else entry["roles"][role]["category"]
        # Private file paths provide isolation. Keep native semantic role tokens
        # exact because some task handlers interpret them (e.g. floors).
        alias = iddl.variant_template.format(**tokens)
        if subtask_of_variant(alias) != iddl.subtask:
            raise ValueError(f"invalid runtime variant {alias}")
        activity = SUBTASK_FAMILY[iddl.subtask][0]
        definition_id = 9000 + len(jobs)
        entry["definition_id"] = definition_id
        groundings = {}
        for role in iddl.roles:
            spec = entry["roles"][role]
            category = spec.get("category")
            group = entry["generation"].get("groups", {}).get(role)
            if group:
                members = [(c, synset_of_category(c)) for c, n in group["members"].items() for _ in range(n)]
                groundings[role] = RoleGrounding(role, "group", members=members)
            elif category == "floors":
                groundings[role] = RoleGrounding(role, "scene", same_as="floor")
            else:
                groundings[role] = RoleGrounding(role, "sampleable" if spec.get("sampleable") else "scene",
                    category=category, synset=synset_of_category(category))
        rendered = render(iddl, groundings, definition_id, alias, activity,
                          comment="Isolated campaign candidate; runtime geometry/demo/replay validation pending.")
        bddl_dir = output / "bddl" / activity
        bddl_dir.mkdir(parents=True, exist_ok=True)
        bddl_path = bddl_dir / f"problem{definition_id}.bddl"
        bddl_path.write_text(rendered.text)
        definitions[alias] = entry
        jobs.append({"class_id": class_id, "alias": alias, "scene": scene, "room": room,
                     "status": "geometry_probe_required" if blockers else "ready_for_fresh_sampling_validation",
                     "blockers": list(blockers), "production_released": False, "expected_objects": expected,
                     "scene_source_sha256": scene_sha, "bddl_sha256": digest(bddl_path), "bddl_instances": rendered.instances,
                     "sample_argv": ["python", "-m", "ispatialgym.sampling.sample_isg", "--variant", alias,
                         "--customs", "${MATERIALIZED}/candidate_customs.json", "--bddl-root", "${MATERIALIZED}/bddl",
                         "--scene", scene, "--templates", "1", "--instances-per-template", "1", "--max-attempts", "3",
                         "--seed", "20261110", "--no-write-back", "--out", "${CAMPAIGN_OUT}/validation/" + class_id]})

    make("configuration_matching-articulation_open_small_scale_01",
         source_alias="configuration_matching-articulation_open_small_scale-blender_cwkvib-breakfast_table")
    make("disembedding_04", distractors=[("bottle_of_milk", "debqen"), ("carton_of_milk", "kklgxk")])
    make("disembedding_06", distractors=[("bottle_of_coconut_milk", "idenxg"), ("carton_of_soy_milk", "orgyvw")])
    make("alignment-world_vertical_02", support="walls_lempwq_0", region={
        "id": "living_room_0/walls_lempwq_0/r00", "wall": "walls_lempwq_0", "room": "living_room_0",
        "p0": [-5.807, -0.493], "p1": [-2.932, -0.493], "normal": [0., -1.], "hang_z": 1.45,
        "length_m": 3.12, "max_object_m": 0.6})
    make("configuration_matching-attachment_01", room="dining_room_0", support="breakfast_table_rhjoby_0", support_z=0.8126)
    make("inter_object_relation_satisfaction-touching_02", room="dining_room_0", support="breakfast_table_rhjoby_0", support_z=0.8126)
    make("inter_object_relation_satisfaction-inside_large_scale_01", scene="house_double_floor_upper", room="bedroom_0",
         support="desk_aduafr_0", support_z=0.773, partner="bookcase_kmuacd_0",
         remove_gen=("inside_goal_region",),
         gen_updates={"inside_surface_z_m": 0.8354, "inside_stance_side": "+y"})
    fridge_probe_path=inv/'fridge_interior_probe.json'
    fridge_plane=None
    if fridge_probe_path.is_file():
        fridge_probe=json.loads(fridge_probe_path.read_text())
        if fridge_probe.get('status')!='measured_planes_require_target_geometry_and_maps' or fridge_probe.get('object')!='fridge_xyejdx_0':
            raise ValueError('fridge probe is not a successful exact replacement measurement')
        usable=[p for p in fridge_probe['planes'] if .65<=p['height_above_floor']<=1.1 and p['n_free_cells']>=100 and p['open_edges']]
        if usable:fridge_plane=max(usable,key=lambda p:p['headroom_m']['min'])
    make("inter_object_relation_satisfaction-inside_large_scale_02", scene="Rs_int", room="kitchen_0",
         support="bottom_cabinet_slgzfc_1", support_z=0.874, partner="fridge_xyejdx_0",
         gen_updates={"partner_joint_init": {"j_link_0": 1.55, "j_link_1": 1.55}, "inside_surface_z_m": fridge_plane['height_above_floor'] if fridge_plane else None},
         remove_gen=("inside_stance_side", "inside_goal_region"),
         blockers=() if fridge_plane else ("Measure the chosen open-fridge interior plane, floor-relative height, opening side and headroom; no source-fridge plane may be reused.",))
    if fridge_plane:
        definitions[jobs[-1]['alias']]['campaign_provenance']['interior_probe']={
            'path':str(fridge_probe_path),'sha256':digest(fridge_probe_path),'selected_plane':fridge_plane,
            'selection':'Usable middle shelf with maximum measured minimum headroom; exact task geometry and maps still required.'}
        jobs[-1]['required_map_dependencies']=['Native measured private interior spec','Positive jar_of_spaghetti_sauce/exmsmm tabletop initial map at support0.874m','Positive goal interior place maps for both offered arms']
    make("inter_object_relation_satisfaction-inside_large_scale_03", support="breakfast_table_xftrki_0", support_z=0.6789)
    make("inter_object_relation_satisfaction-ontop_large_scale_01", room="utility_room_0", partner="countertop_ikwqer_0")
    target_customs = output / "candidate_customs.json"
    target_customs.write_text(json.dumps(definitions, indent=2) + "\n")
    for job in jobs:
        gp = GroundedProblem.load(job["alias"], str(target_customs), str(output / "bddl"))
        if gp.subtask != parse_file(str(ROOT / "ispatialgym/assets/iddl" / definitions[job["alias"]]["iddl"])).subtask:
            raise ValueError("compiled task mismatch")
        # Replay the native category counters without starting the simulator.
        # Scene objects bind before spawned singletons, then group members.
        bindings = Bindings()
        objects_by_role = {}
        for obj in job["expected_objects"]:
            objects_by_role.setdefault(obj["role"], []).append(obj)
        allocated = {}
        scene_roles = sorted(gp.scene_roles(), key=lambda r: r != "floor")
        for role in scene_roles + gp.sampleable_roles() + gp.group_roles():
            for index, obj in enumerate(objects_by_role.get(role, [])):
                intended = job["bddl_instances"].get(role + ".members", [job["bddl_instances"].get(role)])[index]
                scene_name = (obj.get("scene_init_args") or {}).get("name")
                if intended in allocated and scene_name and allocated[intended] == scene_name:
                    continue  # native floor/support aliases share one instance
                actual = bindings.add(role, synset_of_category(obj["category"]))
                if actual.name != intended:
                    raise ValueError(f"runtime/BDDL binding mismatch {job['class_id']}/{role}: {actual.name} != {intended}")
                allocated[intended] = scene_name
        job["binding_order_check"] = "native Bindings allocation matches every rendered BDDL instance"
        job["customs_sha256"] = digest(target_customs)
    contracts = []
    for job in jobs:
        refs, bindings, seen, group_counters = [], {}, set(), {}
        for obj in job["expected_objects"]:
            role = obj["role"]
            if role + ".members" in job["bddl_instances"]:
                index = group_counters.get(role, 0)
                instance = job["bddl_instances"][role + ".members"][index]
                group_counters[role] = index + 1
            else:
                instance = job["bddl_instances"][role]
            if instance in seen:
                continue
            seen.add(instance)
            canonical_role = {"target": "target", "partner": "partner", "distractors": "distractor"}.get(role, "support")
            semantic = {"walls": "wall", "floors": "floor", "tray": "tray"}.get(obj["category"], "object")
            refs.append(ObjectRef(instance=instance, role=canonical_role, category=obj["category"], model=obj["model"],
                asset_sha256=obj["asset_sha256"], default_scale=tuple(obj["default_scale"]), semantic_kind=semantic,
                joint=(definitions[job["alias"]]["roles"][role].get("joint") or {}).get(obj["category"] + "/" + obj["model"]),
                ready=True, readiness_source=obj["readiness_source"]))
            if obj["scene_init_args"]:
                bindings[instance] = obj["scene_init_args"]["name"]
        template = Template(template_id="replacement__" + job["class_id"], task=task_for_alias(job["class_id"]),
            class_id=job["class_id"], split="eval_standard", scene=job["scene"], room=job["room"], objects=tuple(refs),
            definition_sha256=record_digest({"entry": definitions[job["alias"]], "bddl_sha256": job["bddl_sha256"]}),
            scene_setup={"task_rooms": [job["room"]], "inst_to_name": bindings, "runtime_bddl_sha256": job["bddl_sha256"],
                "customs_alias": job["alias"], "validation": job["status"], "fresh_demo": "pending", "fresh_replay": "pending",
                "spawned_object_scene_names": "must bind from fresh template_meta before goal-group construction"})
        contracts.append({**asdict(template), "fingerprint": template.fingerprint})
    (output / "replacement_contract_candidates.json").write_text(json.dumps({"templates": contracts}, indent=2, sort_keys=True) + "\n")
    (output / "validation_jobs.json").write_text(json.dumps({"jobs": jobs}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"compiled": len(jobs), "ready_for_sampling": sum(not j["blockers"] for j in jobs), "production_released": 0}))


if __name__ == "__main__":
    main()
