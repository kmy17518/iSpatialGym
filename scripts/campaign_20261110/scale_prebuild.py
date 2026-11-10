#!/usr/bin/env python3
"""Prepare train-only scaled grasp jobs from ready, nonreserved baseline targets.

Uses legacy *metadata* for the rest/constraint, never legacy grasps. Every new
scale gets fresh v2 candidate/physics validation and an independent entry key.
"""
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ispatialgym.grasping.manifest import legacy_job  # noqa: E402
from ispatialgym.grasping import schema as S  # noqa: E402
from contracts import digest  # noqa: E402
from inventory import PENDING_KEYS  # noqa: E402


def main():
    out = ROOT / "tmp/campaign_20261110/inventory"
    manifest = json.loads((out / "standard_contract_ready.json").read_text())["templates"]
    source = {r["key"]: r for r in json.loads((out / "cluster_templates_with_scales.json").read_text())["records"]}
    reserved = json.loads((out / "split_reservations.json").read_text())
    excluded_models, excluded_categories = set(reserved["train_exclude_models"]), set(reserved["train_exclude_categories"])
    jobs, profiles, skipped = {}, {}, []
    for template in manifest:
        key = template["class_id"]
        # First useful tranche: tabletop objects with established constraint/rest metadata.
        if key in PENDING_KEYS or not (template["task"] == "disembedding" or template["task"].startswith("inter_object_relation_satisfaction-") and template["task"].endswith("_small_scale")):
            continue
        meta = source[key]["metadata"]
        for obj in template["objects"]:
            if obj["role"] != "target":
                continue
            model_key = obj["category"] + "/" + obj["model"]
            if not obj["ready"] or model_key in excluded_models or obj["category"] in excluded_categories:
                raise ValueError(f"Unready/reserved training candidate {model_key}")
            grasp_key = (meta.get("choices", {}).get("target", {}) or {}).get("grasp_key")
            if not grasp_key:
                skipped.append({"class_id": key, "model": model_key, "reason": "no established rest/constraint key"})
                continue
            base = legacy_job(grasp_key, str(ROOT / "runlogs/grasp_library_20260905"))
            if base is None or base["category"] != obj["category"] or base["model"] != obj["model"]:
                skipped.append({"class_id": key, "model": model_key, "reason": "rest/constraint metadata absent or mismatched"})
                continue
            identity = {"model": model_key, "asset_sha256": obj["asset_sha256"], "default_scale": obj["default_scale"]}
            profile_key = digest(identity)
            if profile_key not in profiles:
                rng = random.Random(int(hashlib.sha256(("campaign_20261110_train_scale:" + profile_key).encode()).hexdigest(), 16))
                factors = [rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)]
                scales = [list(S.canonical_scale([s * factor for s in obj["default_scale"]])) for factor in factors]
                if scales[0] == scales[1]:
                    raise ValueError("Scale quantization unexpectedly merged the two train candidates")
                profiles[profile_key] = {**identity, "split": "train", "sampled_uniform_factors": factors, "absolute_scales": scales,
                                         "precision_rule": "absolute scales rounded to 1e-4 exactly as v2 schema canonical_scale; persist these exact values"}
            for slot, scale in enumerate(profiles[profile_key]["absolute_scales"]):
                job = {**base, "scale": scale, "legacy_key": None, "source": "campaign_train_scale",
                       "source_legacy_metadata_key": grasp_key, "campaign_profile": profile_key, "campaign_scale_slot": slot,
                       "asset_sha256": obj["asset_sha256"], "ready": True, "readiness_source": obj["readiness_source"]}
                job["id"] = S.entry_id(job["category"], job["model"], scale, job["constraint"], job.get("rest_hint_wxyz"))
                job.setdefault("subtasks", []).append(template["task"])
                job["campaign_classes"] = [key]
                height = meta.get("template_facts", {}).get("support_surface", {}).get("top_z_above_floor")
                job["measured_default_support_heights_m"] = [] if height is None else [height]
                if job["id"] in jobs:
                    old = jobs[job["id"]]
                    old["campaign_classes"] = sorted(set(old["campaign_classes"] + [key]))
                    old["subtasks"] = sorted(set(old["subtasks"] + [template["task"]]))
                    old["measured_default_support_heights_m"] = sorted(set(old["measured_default_support_heights_m"] + job["measured_default_support_heights_m"]))
                else:
                    jobs[job["id"]] = job
    jobs_list = [jobs[key] for key in sorted(jobs)]
    (out / "train_scale_profiles.json").write_text(json.dumps({"profiles": profiles}, indent=2, sort_keys=True) + "\n")
    (out / "train_scale_glib_jobs.json").write_text(json.dumps(jobs_list, indent=2, sort_keys=True) + "\n")
    (out / "train_scale_prebuild_notes.json").write_text(json.dumps({"jobs": len(jobs_list), "profiles": len(profiles), "skipped": skipped,
        "run_stages": "glib first; maps follow per successful entry without waiting for entire batch",
        "safety": "No production sampling until model/defaultscale/scale fingerprint matches persisted profile; no new legacy_key overrides; no eval perturbations",
        "map_requirements": "Default and perturbed support heights require campaign anchors. Missing measured height must be probed; never substitute world Z."}, indent=2) + "\n")
    print(json.dumps({"glib_jobs": len(jobs_list), "model_default_scale_profiles": len(profiles), "skipped": skipped}))


if __name__ == "__main__":
    main()
