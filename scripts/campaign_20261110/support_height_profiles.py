#!/usr/bin/env python3
"""Persist two train-only height scales and dependency-ordered map requests.

Analytic heights are candidate predictions. A live bottom-anchored geometry and
collision probe must record the actual height before the map request is admitted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
B1K = Path("/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600")
sys.path.insert(0, str(ROOT))
from ispatialgym.sampling.height_anchors import build_surfaces, _floor_levels, _floor_datum  # noqa: E402
from contracts import digest  # noqa: E402
from readiness_catalog import ReadinessCatalog  # noqa: E402


def main():
    output = ROOT / "tmp/campaign_20261110/inventory"
    records = {r["key"]: r for r in json.loads((output / "cluster_templates_with_scales.json").read_text())["records"]}
    templates = json.loads((output / "standard_contract_ready.json").read_text())["templates"]
    jobs = json.loads((output / "train_scale_glib_jobs.json").read_text())
    reservations_path = output / "split_reservations.json"
    reservations = json.loads(reservations_path.read_text())
    excluded = set(reservations["train_exclude_models"])
    excluded_categories = set(reservations["train_exclude_categories"])
    anchors = json.loads((output / "train_candidate_summary.json").read_text())["shared_support_anchors"]
    catalog = ReadinessCatalog(B1K / "bddl3/bddl/generated_data")
    surfaces, _ = build_surfaces([str(ROOT / "ispatialgym/assets/icl/roles/candidates.csv")])
    measured = {(s.scene, s.support): s for s in surfaces if s.surface == "top"}
    inventory_root = ROOT / "runlogs/scene_inventory_20260915"
    profiles, bindings, skipped = {}, {}, []

    def add(scene, name, init, class_id=None, shared=False):
        category, model = init["category"], init["model"]
        if category in {"floors", "walls"}:
            return
        status = catalog.status(category, model)
        if not status["ready"] or category + "/" + model in excluded or category in excluded_categories:
            skipped.append({"scene":scene,"name":name,"class_id":class_id,"reason":"unready or globally reserved"})
            return
        path = inventory_root / scene / "supports" / f"{name}.json"
        if not path.is_file():
            skipped.append({"scene":scene,"name":name,"class_id":class_id,"reason":"no measured support inventory; fresh geometry probe needed"})
            return
        geometry = json.loads(path.read_text())
        default_scale = init["scale"]
        if max(abs(a-b) for a,b in zip(default_scale, geometry["scale"])) > 1e-4:
            skipped.append({"scene":scene,"name":name,"class_id":class_id,"reason":"inventory scale differs from actual default; fresh geometry probe needed"})
            return
        usd = B1K / "datasets/behavior-1k-assets/objects" / category / model / "usd" / f"{model}.encrypted.usd"
        identity = {"category":category,"model":model,"asset_sha256":hashlib.sha256(usd.read_bytes()).hexdigest(),
                    "default_scale":default_scale}
        profile_id = digest(identity)
        if profile_id not in profiles:
            rng = random.Random(int(digest({"seed":20261110,"kind":"support_height","profile":profile_id}), 16))
            factors = [rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)]
            scales = [[default_scale[0], default_scale[1], round(default_scale[2] * f, 8)] for f in factors]
            profiles[profile_id] = {**identity,"mode":"height","split":"train","sampled_uniform_factors":factors,
                "absolute_scales":scales,"applied_z_factors":[s[2]/default_scale[2] for s in scales],
                "ready":True,"readiness_source":status["readiness_source"],
                "rounding":"z scale to1e-8; x/y preserved exactly; persist actual values"}
        surface = measured.get((scene, name))
        if surface:
            floor_z, floor_source = surface.floor_z, surface.floor_source
            height = surface.height
        else:
            floor_z, floor_source = _floor_datum(_floor_levels(scene, str(inventory_root)), geometry["aabb_world"]["min"][2])
            height = round(geometry["top"]["z"] - floor_z, 4)
        height_source = "scene_inventory_top_minus_verified_floor"
        if class_id:
            facts = records[class_id]["metadata"].get("template_facts", {})
            s = facts.get("support_surface", {})
            if s.get("support") == name and s.get("surface") == "interior":
                # A shelf starts at its interior plane, not the whole bookcase top.
                height = float(s["z_above_floor"])
                floor_z = float(s["z_world"]) - height
                height_source = "runtime_template_interior_z_above_floor"
            elif s.get("support") == name and s.get("top_z_above_floor") is not None:
                height = float(s["top_z_above_floor"])
                floor_z = float(facts["support"]["z"]) - height
                height_source = "runtime_template_top_z_above_floor"
        bottom = float(geometry["aabb_world"]["min"][2]) - floor_z
        profile = profiles[profile_id]
        predicted = [bottom + f * (height-bottom) for f in profile["applied_z_factors"]]
        binding_id = digest({"scene":scene,"name":name,"profile":profile_id,"default_height":height,"bottom":bottom})
        if binding_id not in bindings:
            bindings[binding_id] = {"profile_id":profile_id,"scene":scene,"name":name,
                "default_height_above_floor_m":height,"default_bottom_above_floor_m":bottom,
                "floor_z_world_m":floor_z,"floor_source":floor_source,"height_source":height_source,
                "source_geometry_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                "predicted_heights_above_floor_m":predicted,"predicted_height_keys":[f"{h:.4f}" for h in predicted],
                "classes":[],"shared_anchor":shared,"geometry_admitted":False,
                "admission_requirements":["Keep the object bottom at its measured default world Z", "Check finite positive bounds and collisions after scale",
                    "Scale-up scene objects must have no new overlap with any unmoved fixture", "Recheck support free area, contents, task room navigation and maximum layout fit",
                    "Measure actual support plane above actual floor and use that exact1e-4 key in map requests"],
                "generation_hooks":{"scene_setup":"campaign_augmentation","lighting":"parent-owned train light profile"}}
        if class_id and class_id not in bindings[binding_id]["classes"]:
            bindings[binding_id]["classes"].append(class_id)
        bindings[binding_id]["shared_anchor"] |= shared
    for template in templates:
        record = records[template["class_id"]]
        for role in record["metadata"]["bindings"]:
            if role["role"] == "support":
                add(template["scene"], role["scene_object_name"], record["object_init_args"][role["scene_object_name"]], template["class_id"])
    for anchor in anchors:
        scene = anchor["scene"]
        path = B1K / "datasets/behavior-1k-assets/scenes" / scene / "json" / f"{scene}_best.json"
        init = json.loads(path.read_text())["objects_info"]["init_info"][anchor["scene_object_name"]]["args"]
        add(scene, anchor["scene_object_name"], init, shared=True)
    requests = {}
    for job in jobs:
        for binding_id, binding in bindings.items():
            standard = bool(set(binding["classes"]) & set(job["campaign_classes"]))
            if not standard and not binding["shared_anchor"]:
                continue
            for slot, height_key in enumerate(binding["predicted_height_keys"]):
                key = digest({"entry_id":job["id"],"binding_id":binding_id,"height_slot":slot,"height_key":height_key})
                requests[key] = {"entry_id":job["id"],"target_profile":job["campaign_profile"],
                    "target_scale_slot":job["campaign_scale_slot"],"target_absolute_scale":job["scale"],
                    "support_profile":binding["profile_id"],"support_binding":binding_id,"support_height_slot":slot,
                    "support_absolute_scale":profiles[binding["profile_id"]]["absolute_scales"][slot],
                    "predicted_height_key":height_key,"actual_height_key":None,
                    "priority":1 if standard else 2,"arms":["right","left"],
                    "status":"waiting_support_geometry_and_positive_grasp_entry","production_admitted":False}
    common = {"schema_revision":1,"reservation_sha256":hashlib.sha256(reservations_path.read_bytes()).hexdigest(),
              "scope":"train only; eval standard/model/category remain default unperturbed"}
    (output / "support_height_profiles.json").write_text(json.dumps({**common,"profiles":profiles,"bindings":bindings,"skipped":skipped},indent=2,sort_keys=True)+"\n")
    rows = sorted(requests.values(),key=lambda r:(r["priority"],r["entry_id"],r["support_binding"],r["support_height_slot"]))
    summary = {"support_profiles":len(profiles),"surface_bindings":len(bindings),"shared_anchors":sum(b["shared_anchor"] for b in bindings.values()),
               "map_requests":len(rows),"priority1":sum(r["priority"]==1 for r in rows),"priority2":sum(r["priority"]==2 for r in rows),
               "gpu_map_passes_both_arms":len(rows)*2,"admitted_requests":0,"skipped":skipped}
    (output / "support_height_map_requests.json").write_text(json.dumps({**common,"summary":summary,"requests":rows},indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
