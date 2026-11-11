"""Account every selected proposal and its unresolved physical dependencies.

Logical requests are not physical artifacts: an artifact identity is assigned only
after all required measurements and generator compatibility inputs are known.
This inventory never authorizes trajectory generation or marks file presence as
positive native preparation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import random

SCHEMA = "isg-full-training-dependency-inventory-v1"
OWNER = "/root/templates_assets"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ref(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def object_profile(asset):
    measured = asset.get("native_default_geometry")
    if not measured:
        return None
    identity = {"model": asset["model_key"], "asset_sha256": asset["asset_sha256"],
                "default_scale": measured["default_scale"]}
    key = digest(identity)
    rng = random.Random(int(hashlib.sha256(("campaign_20261110_train_scale:" + key).encode()).hexdigest(), 16))
    factors = [rng.uniform(.8, 1.2), rng.uniform(.8, 1.2)]
    scales = [[round(float(x) * factor, 4) for x in measured["default_scale"]] for factor in factors]
    if scales[0] == scales[1]:
        raise ValueError("two selected scale draws collapsed after native scale rounding")
    return {"id": key, **identity, "sampled_uniform_factors": factors, "absolute_scales": scales,
            "native_default_evidence": measured["source"]}


def slots(candidate):
    """Finite selected slots; unresolved joints remain an explicit extra scope."""
    task = candidate["task"]
    if task.startswith("camera_relocalization"):
        return [(None, None)]
    target = candidate.get("target") or {}
    objects = [None] if target.get("category") == "door" and task.endswith("_large_scale") else [0, 1]
    support = candidate["anchor"]
    # Structural walls are not resized. Axial tables retain the original user's
    # two support-height requirement, despite the older checklist's omission.
    height = [0, 1] if support.get("category") not in (None, "walls", "floors") and not task.endswith("_large_scale") else [None]
    return list(itertools.product(objects, height))


def joint_candidates(candidates, assets):
    """Count each measured movable joint; reachability remains a native gate."""
    for candidate in candidates:
        if "articulation_" not in candidate["task"]:
            yield candidate
            continue
        target = candidate.get("target") or {}
        asset = assets.get(target.get("category", "") + "/" + target.get("model", ""), {})
        geometry = asset.get("native_default_geometry")
        joints = (geometry or {}).get("joints")
        usable = []
        unresolved = False
        for name, joint in (joints or {}).items():
            if "fixed" in str(joint.get("type", "")).lower():
                continue
            lower, upper = joint.get("lower_limit"), joint.get("upper_limit")
            if not isinstance(lower, (float, int)) or not isinstance(upper, (float, int)) or not all(map(math.isfinite, (lower, upper))):
                unresolved = True
                continue
            if upper > lower:
                usable.append({"name": name, **joint})
        for joint in usable:
            yield {**candidate, "selected_joint": joint,
                   "joint_selection_state": "measured_movable_reachability_unverified"}
        if not usable or unresolved:
            yield {**candidate, "selected_joint": None, "joint_selection_state":
                   "native_joint_inventory_missing_or_nonfinite" if joints is None or unresolved else "no_movable_joint_measured"}


def requirements(task):
    if task.startswith("camera_relocalization"):
        return []
    if task in ("mental_rotation", "object_scaling"):
        return ["floating_grasp_library", "floating_native_motion_validation"]
    if "articulation_" in task:
        return ["articulated_handle_geometry", "articulated_motion_library", "articulated_prefix_map"]
    if task == "configuration_matching-attachment":
        return ["attachment_grasp_library", "tabletop_reach_map", "attachment_mating_geometry", "attachment_reach_map"]
    if task.startswith("alignment-"):
        return ["contact_push_library", "contact_brace_library", "generic_reach_map", "contact_push_map", "contact_brace_map"]
    if task.startswith("inter_object_relation_satisfaction-") and task.endswith("_large_scale"):
        if any(x in task for x in ("nextto", "touching", "under")):
            return ["floor_push_contact_geometry", "floor_push_reach_map"]
        return ["tabletop_grasp_library", "generic_reach_map", "interior_reach_map"]
    return ["tabletop_grasp_library", "tabletop_reach_map"]


def build(candidates, assets, height_profiles, policy, *, sources, generator_inputs):
    observed = datetime.now(timezone.utc).isoformat()
    artifacts, configurations, gaps = {}, [], []
    profiles = {key: object_profile(asset) for key, asset in assets.items()}
    supports = {(b["scene"], b["name"]): (key, b, height_profiles["profiles"][b["profile_id"]])
                for key, b in height_profiles["bindings"].items() if b.get("shared_anchor")}
    held_rooms = {(r["scene"], r["room"]) for r in policy["category_holdout_rooms"]}

    def request(kind, inputs, cid, cls, *, unresolved=(), verified_evidence=(), not_applicable=False):
        identity = {"kind": kind, "inputs": inputs}
        key = "request_" + digest(identity)
        if key not in artifacts:
            missing = sorted(set(unresolved))
            artifacts[key] = {"id": key, "kind": kind, "inputs": inputs,
                "input_resolution": "unresolved" if missing else "resolved",
                "physical_artifact_id": None if missing or not_applicable else "physical_" + digest(identity),
                "status": "not_applicable" if not_applicable else "verified" if verified_evidence else "blocked" if missing else "missing",
                "unresolved_inputs": missing, "consumers": [], "required_by_classes": [],
                "owner": OWNER, "next_action": "No grasp/reach work for camera navigation" if not_applicable else
                    "Resolve exact measured inputs: " + ", ".join(missing) if missing else
                    "Execute the exact native generator and positive reader/identity audit",
                "required_for": ["native_generation", "collection", "physics_replay"],
                "artifact": list(verified_evidence)[0] if verified_evidence else None,
                "active_jobs": [], "evidence": list(verified_evidence), "observed_at": observed}
        artifacts[key]["consumers"].append(cid)
        artifacts[key]["required_by_classes"].append(cls)
        return key

    for candidate in joint_candidates(candidates, assets):
        task, cls = candidate["task"], candidate["class_id"]
        anchor = candidate["anchor"]
        roles = [("target", candidate.get("target")), ("partner", candidate.get("partner"))]
        roles += [(r["role"], r) for r in candidate.get("extra_roles", [])]
        support = anchor if "category" in anchor else anchor.get("support")
        if support:
            roles.append(("support", support))
        role_rows = [(name, assets[obj["category"] + "/" + obj["model"]]) for name, obj in roles if obj]
        candidate_blockers = sorted({why for _, a in role_rows for why in a["blockers"]})
        if {(anchor["scene"], r) for r in anchor["rooms"]} & held_rooms:
            candidate_blockers.append("current_global_room_holdout")
        for scale_slot, height_slot in slots(candidate):
            configuration_identity = {"candidate_id": candidate["candidate_id"], "object_scale_slot": scale_slot,
                                      "support_height_slot": height_slot}
            if "selected_joint" in candidate:
                configuration_identity["selected_joint"] = candidate["selected_joint"]
                configuration_identity["joint_selection_state"] = candidate["joint_selection_state"]
            cid = "configuration_" + digest(configuration_identity)[:24]
            dependencies, selected_roles, blockers = [], [], list(candidate_blockers)
            coupled = task in ("configuration_matching-attachment", "configuration_matching-block_design")
            for role, asset in role_rows:
                geometry = asset.get("native_default_geometry")
                dependencies.append(request("native_asset_default_geometry", {"asset_sha256": asset["asset_sha256"],
                    "model_key": asset["model_key"], "dataset": asset["dataset"], "scale_mode": "provider_default"}, cid, cls,
                    unresolved=asset["blockers"], verified_evidence=[geometry["source"]] if geometry and not asset["blockers"] else []))
                profile = profiles[asset["model_key"]]
                scale = profile["absolute_scales"][scale_slot] if role == "target" and profile and scale_slot is not None else \
                    geometry["default_scale"] if geometry else None
                scale_reason = "target_two_persisted_draws" if role == "target" and scale_slot is not None else "native_default"
                if coupled and role in ("target", "partner", "board", "blocks"):
                    # Pair/group scale identity must include all mating/group models;
                    # independent per-model draws cannot silently stand in for it.
                    scale = None
                    scale_reason = "coupled_pair_or_group_profile_requires_exact_native_defaults"
                if task.endswith("_large_scale") and role in ("partner", "support", "target"):
                    scale = None
                    scale_reason = "exact_scene_initialization_or_bounded_replacement_scale_required"
                if role == "support" and height_slot is not None:
                    h = supports.get((anchor["scene"], anchor.get("scene_object_name")))
                    scale = h[2]["absolute_scales"][height_slot] if h else None
                    scale_reason = "z_only_support_profile"
                selected_roles.append({"role": role, "model_key": asset["model_key"], "category": asset["category"],
                    "model": asset["model"], "dataset": asset["dataset"], "asset_sha256": asset["asset_sha256"],
                    "ready": asset["ready"], "readiness_source": asset["readiness_source"],
                    "original_default_scale": geometry["default_scale"] if geometry else None,
                    "applied_scale": scale, "scale_rule": scale_reason, "scale_profile": profile["id"] if profile else None})
                if role == "target" and "articulation_" in task:
                    selected_roles[-1]["joint"] = candidate.get("selected_joint")
            if task.startswith("camera_relocalization"):
                dependencies.append(request("grasp_and_reach_not_applicable", {"task": task}, cid, cls, not_applicable=True))
                dependencies.append(request("camera_room_navigation_and_obstacle_geometry", {"scene": anchor["scene"],
                    "rooms": anchor["rooms"], "task": task, "roles": selected_roles}, cid, cls,
                    unresolved=["native room path coverage and collision-safe obstacle scale bounds"]))
            else:
                target = next((r for r in selected_roles if r["role"] == "target"), None)
                if support:
                    h = supports.get((anchor["scene"], anchor.get("scene_object_name")))
                    support_input = {"scene": anchor["scene"], "rooms": anchor["rooms"],
                        "object": anchor.get("scene_object_name"), "height_slot": height_slot,
                        "support_profile": h[1]["profile_id"] if h else None,
                        "selected_scale": h[2]["absolute_scales"][height_slot] if h and height_slot is not None else None,
                        "measured_height_above_floor": None, "source_geometry_sha256": h[1]["source_geometry_sha256"] if h else None}
                    dependencies.append(request("native_support_geometry_and_height", support_input, cid, cls,
                                                unresolved=["fresh exact scaled support plane, floor datum and collision audit"]))
                    dependencies.append(request("native_maximum_layout_fit", {"class_id": cls, "candidate_id": candidate["candidate_id"],
                        "support": support_input, "maximum_relative_scale": 1.2, "roles": selected_roles}, cid, cls,
                        unresolved=["native maximum-layout footprint and usable support edge proof"]))
                else:
                    support_input = {"scene": anchor["scene"], "rooms": anchor["rooms"], "mode": "room_or_scene_anchor"}
                for kind in requirements(task):
                    inputs = {"target": target, "generator_compatibility": generator_inputs.get(kind),
                              "rest_requirement": "none" if kind.startswith("floating") else "exact native designated rest/contact frame"}
                    missing = []
                    if not target or target["applied_scale"] is None:
                        missing.append("exact native target default and applied scale")
                    if generator_inputs.get(kind) is None:
                        missing.append("physical generator component fingerprint")
                    if "map" in kind:
                        inputs["support"] = support_input
                        missing.append("exact measured query geometry/height and positive prerequisite library")
                    if "attachment" in kind or "articulated" in kind or "floor_push" in kind:
                        inputs["roles"] = selected_roles
                        missing.append("native mating/contact or articulated joint/state input")
                    arms = ("right", "left") if "map" in kind or "motion" in kind else (None,)
                    for arm in arms:
                        dependencies.append(request(kind, {**inputs, "arm": arm}, cid, cls, unresolved=missing))
                if task == "disembedding":
                    dependencies.append(request("full_distractor_tray_pool_geometry", {"class_id": cls,
                        "candidate_id": candidate["candidate_id"], "scale_slot": scale_slot}, cid, cls,
                        unresolved=["selected full distractor models, tray size and coupled puzzle context closure"]))
                if "articulation_" in task:
                    blockers.append("native joint reachability and handle/contact suitability unverified")
                    if candidate.get("selected_joint") is None:
                        blockers.append(candidate["joint_selection_state"])
            blockers.extend(sorted({x for d in dependencies for x in artifacts[d]["unresolved_inputs"]}))
            configurations.append({"id": cid, "candidate_id": candidate["candidate_id"], "class_id": cls,
                "configuration_sha256": None,
                "task": task, "scene": anchor["scene"], "rooms": anchor["rooms"], "support": anchor,
                "object_scale_slot": scale_slot, "support_height_slot": height_slot, "roles": selected_roles,
                "selected_joint": candidate.get("selected_joint"),
                "joint_selection_state": candidate.get("joint_selection_state"),
                "dependencies": sorted(set(dependencies)), "status": "blocked" if blockers else "missing",
                "blockers": sorted(set(blockers)), "owner": OWNER,
                "next_action": "Resolve the first unmet physical dependency; retain this slot and any replacement deficit",
                "evidence": [], "observed_at": observed})
    for artifact in artifacts.values():
        artifact["consumers"] = sorted(set(artifact["consumers"]))
        artifact["required_by_classes"] = sorted(set(artifact["required_by_classes"]))
    gap_rules = [
        ("joint_expansion", "configuration_matching-articulation_", "Enumerate every movable reachable joint from the newly measured native assets"),
        ("full_pool_and_tray_selection", "disembedding", "Join every selected distractor model and tray-size variant, including coupled puzzle pieces"),
        ("coupled_board_patterns", "configuration_matching-block_design", "Audit all selected Kohs face permutations, board scaling and four-identical-block configurations"),
        ("native_scene_replacements", "_large_scale", "Bind actual scene scales or prove size-constrained replacement placement and floor-safe scaling"),
    ]
    for key, match, action in gap_rules:
        classes = sorted({c["class_id"] for c in candidates if match in c["task"]})
        if classes:
            gaps.append({"id": key, "class_ids": classes, "owner": OWNER, "status": "actively_reconciling", "next_action": action})
    counts = Counter(a["status"] for a in artifacts.values())
    per_class = []
    for cls in sorted({c["class_id"] for c in candidates}):
        configs = [c for c in configurations if c["class_id"] == cls]
        ids = {i for c in configs for i in c["dependencies"]}
        per_class.append({"class_id": cls, "candidate_rows": sum(c["class_id"] == cls for c in candidates),
            "configuration_slots": len(configs), "fully_served": 0, "required_unique_requests": len(ids),
            "dependency_statuses": dict(Counter(artifacts[i]["status"] for i in ids)),
            "owner": OWNER, "next_action": "Stream native asset/scale preparation, then measured support and exact map dependencies"})
    return {"schema": SCHEMA, "campaign_id": "campaign_20261110", "observed_at": observed, "owner": OWNER,
        "production_authorization": False, "sources": sources,
        "summary": {"proposal_rows": len(candidates), "standard_classes": len(per_class),
            "selected_configuration_slots": len(configurations), "exact_materialized_configurations": 0,
            "fully_served_configurations": 0, "required_unique_logical_requests": len(artifacts),
            "resolved_physical_artifact_identities": sum(a["physical_artifact_id"] is not None for a in artifacts.values()),
            "dependency_statuses": dict(counts), "denominator_complete": False,
            "denominator_note": "All original proposal rows retained; listed joint/pool/scene scope expansions remain explicit, so this is a lower-bound configuration denominator."},
        "scope_gaps": gaps, "per_class": per_class, "configurations": configurations,
        "artifact_requests": list(artifacts.values())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("candidates", "assets", "height-profiles", "policy", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    paths = {k: getattr(args, k.replace("-", "_")) for k in ("candidates", "assets", "height-profiles", "policy")}
    docs = {k: json.loads(v.read_text()) for k, v in paths.items()}
    policy = docs["policy"].get("reservations", docs["policy"])
    result = build(docs["candidates"]["templates"], docs["assets"]["assets"], docs["height-profiles"], policy,
                   sources={k: ref(v) for k, v in paths.items()}, generator_inputs={})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
