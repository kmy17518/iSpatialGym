#!/usr/bin/env python3
"""Persist exact map requests for ready replacement canaries; no GPU submission."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ispatialgym.grasping.manifest import legacy_job
from ispatialgym.grasping.schema import entry_id, canonical_scale
from ispatialgym.sampling.height_anchors import AnchorSet, build_surfaces


def main():
    inv = ROOT / "tmp/campaign_20261110/inventory"
    out = ROOT / "tmp/campaign_20261110/materialized/dependencies"
    out.mkdir(parents=True, exist_ok=True)
    records = {r["key"]: r for r in json.loads((inv / "cluster_templates_with_scales.json").read_text())["records"]}
    materialized_path = out.parent / "validation_jobs.json"
    materialized = {r["class_id"]: r for r in json.loads(materialized_path.read_text())["jobs"]}
    surfaces, _ = build_surfaces([str(ROOT / "ispatialgym/assets/icl/roles/candidates.csv")])
    surfaces = {(s.scene, s.support): s for s in surfaces if s.surface == "top"}
    requirements = [
        ("inter_object_relation_satisfaction-touching_02", "house_single_floor", "breakfast_table_rhjoby_0", ["tabletop"]),
        ("inter_object_relation_satisfaction-inside_large_scale_03", "house_double_floor_lower", "breakfast_table_xftrki_0", ["tabletop"]),
        ("inter_object_relation_satisfaction-ontop_large_scale_01", "house_single_floor", "countertop_ikwqer_0", ["tabletop", "floor"]),
        ("inter_object_relation_satisfaction-inside_large_scale_01", "house_double_floor_upper", "desk_aduafr_0", ["tabletop"]),
    ]
    jobs, heights, requests, used = [], {}, [], {}
    for class_id, scene, support, kinds in requirements:
        key = records[class_id]["metadata"]["choices"]["target"]["grasp_key"]
        job = legacy_job(key, str(ROOT / "runlogs/grasp_library_20260905"))
        target = next(o for o in materialized[class_id]["expected_objects"] if o["role"] == "target")
        if (job["category"], job["model"]) != (target["category"], target["model"]):
            raise ValueError(f"legacy metadata identity mismatch {class_id}")
        job["scale"] = list(canonical_scale(target["default_scale"]))
        job["id"] = entry_id(job["category"], job["model"], job["scale"], job["constraint"], job.get("rest_hint_wxyz"))
        job["campaign_class"] = class_id
        job["asset_sha256"] = target["asset_sha256"]
        job["ready"] = target["ready"]
        job["readiness_source"] = target["readiness_source"]
        jobs.append(job)
        surface = surfaces[scene, support]
        used[surface.id] = surface
        heights[job["id"]] = [surface.key]
        requests.append({"class_id": class_id, "entry_id": job["id"], "legacy_metadata_key": key,
                         "kinds": kinds, "height_key": surface.key, "scene": scene, "support": support,
                         "arms": ["right", "left"], "status": "check exact existing positive library then compute missing maps"})
    pair_class = "configuration_matching-attachment_01"
    pair_objects = materialized[pair_class]["expected_objects"]
    partner = next(o for o in pair_objects if o["role"] == "partner")
    male = next(o for o in pair_objects if o["role"] == "target")
    pair = {"id": "lid_tnwisp__teapot_mvrhya__campaign_default", "male_entry": "lid__tnwisp__s1__attachment_free",
            "partner": {k:partner[k] for k in ("category", "model")}, "heights": ["0.8126"],
            "class_id":pair_class, "arms":["right"], "rules":["box", "swept"]}
    pair["partner"]["scale"] = partner["default_scale"]
    pair["partner"]["dataset"] = "behavior-1k-assets"
    pair["asset_sha256"] = {"male":male["asset_sha256"], "partner":partner["asset_sha256"]}
    articulation = {"class_id":"configuration_matching-articulation_open_small_scale_01",
                    "source_alias":"configuration_matching-articulation_open_small_scale-blender_cwkvib-breakfast_table",
                    "category":"blender", "model":"cwkvib", "joint":"j_lid", "scale":[1.,1.,1.],
                    "direction":"pull_open", "height_keys":["0.8126"], "arms":["right","left"],
                    "requirements":"Only the matching arm's physically validated articulation library may supply its prefix map; validate rest/motions first if absent."}
    anchors = AnchorSet.from_surfaces(list(used.values()))
    save = lambda name, data: (out / name).write_text(json.dumps(data, indent=2, sort_keys=True)+"\n")
    save("tabletop_jobs.json", jobs)
    save("heights_by_id.json", heights)
    save("anchors.json", {**anchors.to_json(), "surfaces":[s.__dict__ | {"id":s.id,"key":s.key} for s in used.values()]})
    save("attach_pairs.json", [pair])
    save("articulation_prefix.json", articulation)
    save("requests.json", {"materialized_jobs_sha256":hashlib.sha256(materialized_path.read_bytes()).hexdigest(),
                           "tabletop_or_floor":requests, "attachment":pair, "articulation":articulation,
                           "no_map_dependency_before_retry":["alignment-world_vertical_02","disembedding_04","disembedding_06"],
                           "remaining_geometry_probe":["inter_object_relation_satisfaction-inside_large_scale_02"]})
    print(json.dumps({"map_entries":len(jobs),"anchors":len(anchors.anchors),"attachment_pairs":1,"articulation":1}))


if __name__ == "__main__":
    main()
