#!/usr/bin/env python3
"""Exact measured default support anchors for the first train-scale tranche."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ispatialgym.sampling.height_anchors import AnchorSet, Surface, build_surfaces  # noqa: E402


def main():
    out = ROOT / "tmp/campaign_20261110/inventory"
    metadata_path = out / "cluster_templates_with_scales.json"
    records = json.loads(metadata_path.read_text())["records"]
    record_by_key = {r["key"]: r for r in records}
    jobs = json.loads((out / "train_scale_glib_jobs.json").read_text())
    tables = [str(ROOT / "ispatialgym/assets/icl/roles/candidates.csv")]
    surfaces, _ = build_surfaces(tables)
    surface_by_identity = {(s.scene, s.support): s for s in surfaces if s.surface == "top"}
    # Prefer the exact exported runtime template's measured above-floor value.
    # Other classes may use the same scene/support at exactly the same scale.
    measured = {}
    for r in records:
        m = r["metadata"]
        s = m.get("template_facts", {}).get("support_surface", {})
        if s.get("top_z_above_floor") is not None:
            measured.setdefault((m["scene"], s["support"]), []).append((float(s["top_z_above_floor"]), r["key"]))
    used, by_id, provenance, missing = {}, {}, [], []
    for job in jobs:
        heights = set()
        for class_id in job["campaign_classes"]:
            m = record_by_key[class_id]["metadata"]
            facts = m.get("template_facts", {})
            support = facts.get("support_surface", {}).get("support") or facts.get("support", {}).get("name")
            identity = (m["scene"], support)
            vals = measured.get(identity, [])
            if len({v for v, _ in vals}) > 1:
                raise ValueError(f"conflicting exact runtime heights for {identity}: {vals}")
            inventory = surface_by_identity.get(identity)
            if vals:
                current_scale = record_by_key[class_id]["object_init_args"][support]["scale"]
                for _, reference_class in vals:
                    reference_scale = record_by_key[reference_class]["object_init_args"][support]["scale"]
                    if max(abs(a-b) for a,b in zip(current_scale, reference_scale)) > 1e-4:
                        raise ValueError(f"support scale mismatch between {class_id} and {reference_class}")
                height = vals[0][0]
                source = "frozen_runtime_template:template_facts.support_surface.top_z_above_floor"
                floor_z = float(facts.get("support", {}).get("z", height)) - height
                surface = Surface(m["scene"], support, "top", support.rsplit("_", 2)[0], height, 0.0001,
                                  source, floor_z, "runtime_measured_top_minus_recorded_above_floor", rows=[c for _, c in vals])
            elif inventory:
                surface = inventory
                source = "scene_inventory:measured top.z minus verified floor datum"
            else:
                missing.append({"job": job["id"], "class_id": class_id, "scene": identity[0], "support": identity[1]})
                continue
            used[surface.id] = surface
            heights.add(surface.key)
            provenance.append({"job": job["id"], "class_id": class_id, "surface": surface.id,
                               "height_key": surface.key, "source": source,
                               "runtime_geometry_revalidation": "required before template release"})
        if heights:
            by_id[job["id"]] = sorted(heights, key=float)
    anchors = AnchorSet.from_surfaces(list(used.values()))
    (out / "train_scale_height_anchors.json").write_text(json.dumps({**anchors.to_json(),
        "surfaces": [s.__dict__ | {"id": s.id, "key": s.key} for s in used.values()],
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}, indent=2) + "\n")
    (out / "train_scale_heights_by_id.json").write_text(json.dumps(by_id, indent=2, sort_keys=True) + "\n")
    (out / "train_scale_anchor_provenance.json").write_text(json.dumps({"provenance": provenance, "missing": missing}, indent=2) + "\n")
    print(json.dumps({"jobs":len(jobs),"with_heights":len(by_id),"anchors":len(anchors.anchors),"missing":missing}))


if __name__ == "__main__":
    main()
