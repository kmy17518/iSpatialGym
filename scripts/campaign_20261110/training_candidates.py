#!/usr/bin/env python3
"""Expand ready training candidates while respecting global holdout reservations.

Only candidate counts are emitted. Geometry, maps, one fresh demo and open-loop
replay are required before any row becomes a production template.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import re


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "tmp/campaign_20261110/inventory"
    reservation_path = out / "split_reservations.json"
    reservations = json.loads(reservation_path.read_text())
    reservation_sha256 = hashlib.sha256(reservation_path.read_bytes()).hexdigest()
    exclude_models, exclude_categories = set(reservations["train_exclude_models"]), set(reservations["train_exclude_categories"])
    reserved_rooms = {(r["scene"], r["room"]) for r in reservations["category_holdout_rooms"]}
    catalog = json.loads((out / "category_inventory.json").read_text())["categories"]
    lookup = {(v["category"], v["model"]):v for values in catalog.values() for v in values}
    project = json.loads((out / "project_readiness.json").read_text())["objects"]
    source = json.loads((out / "standard_templates.json").read_text())["templates"]
    icl = list(csv.DictReader((root / "ispatialgym/assets/icl/roles/candidates.csv").open()))
    def model(category, mid):
        key = category + "/" + mid
        if key in exclude_models or category in exclude_categories:
            return None
        item = lookup.get((category, mid))
        if item and item["ready"] and item["usd_present"]:
            return {"category": category, "model": mid, "dataset": "behavior-1k-assets", "ready": True,
                    "readiness_source": item["readiness_source"]}
        custom = project.get(key)
        if custom and custom["ready"]:
            return {"category": category, "model": mid, "dataset": "ispatialgym-assets", "ready": True,
                    "readiness_source": custom["readiness_source"]}
        return None
    # B4 surface evidence is stronger than whole-object bounding-box extents.
    support_candidates = {}
    for row in icl:
        if row["role"] != "support" or row["kind"] != "scene" or row["category"] == "walls" or row["status"].startswith("geometric_reject"):
            continue
        parts = row["model"].rsplit("_", 2)
        if len(parts) != 3:
            continue
        asset = model(row["category"], parts[-2])
        rooms = row["room"].split(",")
        if not asset or any((row["scene"], room) in reserved_rooms for room in rooms):
            continue
        note = row.get("note", "")
        area_match, short_match = re.search(r"area ([0-9.]+) m2", note), re.search(r"short side ([0-9.]+) m", note)
        if not area_match or not short_match:
            continue
        area, short = float(area_match[1]), float(short_match[1])
        try:
            height = float(row["top_z"])
        except ValueError:
            continue
        if short < 0.75 or area < 0.9 or not 0.55 <= height <= 1.2:
            continue
        name = (row["scene"], row["model"])
        support_candidates[name] = {"scene": row["scene"], "rooms": rooms, "scene_object_name": row["model"], **asset,
                                    "surface_area_m2": area, "surface_short_side_m": short, "top_z_world_m": height,
                                    "surface_evidence": row["source"], "surface_note": note,
                                    "max_layout_fit": "pending exact footprint, free-space and support-height verification"}
    # First diversify scenes, then rooms, then fill by surface area.
    ranked = sorted(support_candidates.values(), key=lambda r: (-r["surface_area_m2"], r["scene"], r["scene_object_name"]))
    anchors, seen_scenes, seen_rooms = [], set(), set()
    for mode in ("scene", "room", "any"):
        for row in ranked:
            if row in anchors or len(anchors) >= 10:
                continue
            roomkey = (row["scene"], tuple(row["rooms"]))
            if mode == "scene" and row["scene"] in seen_scenes or mode == "room" and roomkey in seen_rooms:
                continue
            anchors.append(row); seen_scenes.add(row["scene"]); seen_rooms.add(roomkey)
    room_plan_path = out / "train_room_plan.json"
    room_plan = json.loads(room_plan_path.read_text())
    room_plan_sha256 = hashlib.sha256(room_plan_path.read_bytes()).hexdigest()
    room_anchors = sorted({(r["scene"], r["room"]) for rows in room_plan["tasks"].values() for r in rows})
    assert not set(room_anchors) & reserved_rooms
    rows, omissions = [], []
    class_counts = collections.Counter()
    for standard in source:
        key, task = standard["key"], standard["task"]
        bindings = standard["roles"]
        target = next((b for b in bindings if b["canonical_role"] == "target"), None)
        partner = next((b for b in bindings if b["canonical_role"] == "partner"), None)
        if key == "configuration_matching-articulation_open_small_scale_01":
            target = {**target, "model": "cwkvib"}
        def options(binding, target_role=False):
            if binding is None:
                return [None]
            cat = binding["category"]
            variants = []
            initial = model(cat, binding["model"])
            if initial:
                variants.append(initial)
            for candidate in catalog.get(cat, []):
                item = model(cat, candidate["model"])
                if not item or item in variants:
                    continue
                if target_role and task == "mental_rotation":
                    survey = [r for r in icl if r["category"] == cat and r["model"] == item["model"] and r["subtask"] == "mental_rotation"]
                    if not any(r["symmetry"] == "none" and not r["symmetry_near"] for r in survey):
                        continue
                    item = {**item, "semantic_review_source": "ICL confirmed no rotational symmetry; physical grasp validation pending"}
                variants.append(item)
                if len(variants) >= 10:
                    break
            return variants
        target_options, partner_options = options(target, True), options(partner)
        if not target_options or not partner_options:
            omissions.append({"class_id": key, "reason": "no ready unreserved target/partner candidate"})
            continue
        pair_count = max(len(target_options), len(partner_options))
        pairs = [(target_options[i % len(target_options)], partner_options[i % len(partner_options)]) for i in range(pair_count)]
        # Exact mating ports, not arbitrary cross-products, for attachment candidates.
        if task == "configuration_matching-attachment":
            inv = json.loads((Path("/home/minyeongk/Desktop/projects/Behavior-1K-grasp-bands-6a6194600/bddl3/bddl/generated_data/object_inventory.json")).read_text())["attachment_pairs"]
            pairs = []
            for t in target_options:
                ports = set(inv.get(t["category"] + "-" + t["model"], {}).get("M", []))
                for pkey, pports in inv.items():
                    if not ports & set(pports.get("F", [])):
                        continue
                    cat, mid = pkey.rsplit("-", 1); p = model(cat, mid)
                    if p:
                        pairs.append((t, p))
                if len(pairs) >= 10:
                    break
            pairs = pairs[:10]
        room_family = task.startswith("camera_relocalization") or task in {"mental_rotation", "object_scaling"}
        scene_anchored = task.endswith("_large_scale")
        if room_family:
            locations = [{"scene": r["scene"], "rooms": [r["room"]],
                          "traversable_m2": r["traversable_m2"], "sampling_weight": r["sampling_weight"],
                          "path_extent_lower_bound_m": r["path_extent_lower_bound_m"],
                          "required_for_eval_sibling": r["required_for_eval_sibling"],
                          "navigation_buckets_validated": False, "screening": r["validation"],
                          "room_plan_sha256": room_plan_sha256} for r in room_plan["tasks"][task]]
        elif scene_anchored:
            locations = [{"scene": standard["scene"], "rooms": standard["rooms"], "partner_replacement": "same root, no larger than original; collision and fit pending"}]
        elif task == "alignment-world_vertical":
            walls = []
            for row in icl:
                if row["subtask"] != task or row["role"] != "support" or row["category"] != "walls":
                    continue
                p = row["model"].rsplit("_",2)
                if len(p) != 3 or (row["scene"], row["room"]) in reserved_rooms or not model("walls",p[-2]):
                    continue
                walls.append({"scene":row["scene"],"rooms":[row["room"]],"scene_object_name":row["model"],"region_id":row["region_id"],
                              "support":model("walls",p[-2]),"opaque_wall_review":"pending","source":row["source"]})
            locations = list({(r["scene"],r["region_id"]):r for r in walls}.values())[:10]
        else:
            locations = anchors
        extras = []
        for b in bindings:
            if b["canonical_role"] in {"target", "partner"} or b["role"] == "floor" or (b["role"] == "support" and not scene_anchored):
                continue
            item = model(b["category"], b["model"])
            if not item and b["role"] == "support" and scene_anchored:
                alternatives = [model(b["category"], candidate["model"]) for candidate in catalog.get(b["category"], [])]
                item = next((a for a in alternatives if a), None)
                if item:
                    item = {**item, "scene_object_replacement_required": True, "source_scene_object": b["scene_object_name"],
                            "placement_constraint": "same scene anchor; footprint/height, collision and geodesic checks required"}
            if item:
                extras.append({"role": b["role"], **item})
            elif b["model"] != "mcjlhs":
                omissions.append({"class_id":key,"role":b["role"],"reason":"extra role not ready or reserved"})
        for t, p in pairs:
            for anchor in locations:
                proposal = {"class_id":key,"task":task,"target":t,"partner":p,"anchor":anchor,"extra_roles":extras,
                            "status":"candidate_only","required_next":["semantic/geometry fit","bind exact room structural objects and verify readiness","scale-specific grasps/reach maps","fresh successful demo","fresh successful open-loop replay"]}
                proposal["candidate_id"] = "train_" + hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:20]
                rows.append(proposal); class_counts[key] += 1
    (out / "train_template_candidates.json").write_text(json.dumps({"schema_revision":2,"reservation_sha256":reservation_sha256,"templates":rows}, indent=2, sort_keys=True)+"\n")
    report = {"schema_revision":2,"reservation_sha256":reservation_sha256,"candidate_templates":len(rows),"represented_standard_classes":len(class_counts),"production_released_templates":0,
              "per_class":dict(sorted(class_counts.items())),"per_task":dict(sorted(collections.Counter(r["task"] for r in rows).items())),
              "shared_support_anchors":anchors,"shared_room_anchors":[{"scene":s,"room":r} for s,r in room_anchors], "room_plan_sha256":room_plan_sha256,
              "omissions":omissions,"registered_variants":"Readiness and missing data-layer gates apply; not auto-released.",
              "readiness":"Every enumerated asset role passed authoritative catalog or explicit established-project readiness and global reservations."}
    (out / "train_candidate_summary.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report[k] for k in ("candidate_templates","represented_standard_classes","production_released_templates","omissions")}))


if __name__ == "__main__":
    main()
