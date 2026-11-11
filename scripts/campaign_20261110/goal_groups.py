#!/usr/bin/env python3
"""Author and submit fresh held-object goal-group validation candidates.

An anchor is used only to choose a world goal and base. Its instance is never
copied into the group. All group starts have a new allocation in a separate
ledger. Preparation and submission do not promote candidates to final splits.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.campaign_20261110.contracts import (
    ObjectRef, Perturbations, ScaleState, Template, ValidationKey,
    digest, file_digest, require, write_json_atomic,
)
from ispatialgym.sampling.campaign_held_goals import SCHEMA, validate_goal_plan
from ispatialgym.sampling.campaign_surface_goals import SCHEMA as SURFACE_SCHEMA, validate_surface_plan
from ispatialgym.sampling.campaign_tray_goals import SCHEMA as TRAY_SCHEMA, validate_tray_plan
from ispatialgym.sampling.campaign_pair_goals import SCHEMA as PAIR_SCHEMA, RELATIONS as PAIR_RELATIONS, validate_pair_plan, vary_nextto_gap
from ispatialgym.sampling.campaign_joint_goals import SCHEMA as JOINT_SCHEMA, validate_joint_plan
from ispatialgym.sampling.campaign_attachment_goals import SCHEMA as ATTACHMENT_SCHEMA, validate_attachment_plan
from ispatialgym.sampling.campaign_block_goals import SCHEMA as BLOCK_SCHEMA, validate_block_plan


# These archived templates already had an explicit measure but retained
# the pre-migration gating atom. Each exact current definition below passed
# a fresh baseline demo and true action replay. No other explicit-measure source is
# eligible for this compatibility repair; every other semantic section is still
# compared below, and parsed measurement atoms must remain identical.
EXPLICIT_ARTICULATION_MEASURE_REPAIRS = {
    ("configuration_matching-articulation_close_small_scale_02",
     "40c3592afdc39494609881559eea7558fe570678d62f7ea9d7e3695d72e21f09",
     "137524dc9be630524b4df0a38c7674d6eca94fe7bcad006fc8b5eec2f19acb2b"): "fresh_baseline_18420653",
    ("configuration_matching-articulation_close_small_scale_01",
     "ba0f59d069a7dfc89a9718c1c5dfe2fef046ab62c85cb80f9cdbfac201da9ba5",
     "64e4d3c17b913377e021586a25258051317c03882cccfa81ca2a7bafb7ae3f09"): {
        "fresh_baseline": "18423162", "physics_replay": "18423435",
        "gate_sha256": "bd30c1824a19aac1ecb6692e7a1b9495df494a65a977fee72c8c7b9eb2cc7b7d"},
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_env(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values = shlex.split(value)
        require(len(values) <= 1, f"unexpected shell expression in {key}")
        result[key] = values[0] if values else ""
    return result


def fresh_source(path: Path, *, job: Path, started_at: float):
    path = path.resolve()
    require(path.is_file() and path.is_relative_to(job), "anchor artifact must be in its fresh job directory")
    require(path.stat().st_mtime >= started_at, f"anchor artifact predates campaign: {path.name}")
    return {"path": str(path), "sha256": file_digest(path), "mtime": path.stat().st_mtime}


def successful_anchor(job_dir, *, campaign_root, ledger, campaign_started_at: float):
    root, job = Path(campaign_root).resolve(), Path(job_dir).resolve()
    require(job.is_relative_to(root / "out"), "anchor job is outside campaign output")
    ledger = Path(ledger).resolve()
    require(ledger.is_relative_to(root / "manifests"), "anchor ledger is outside campaign manifests")
    with ledger.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    matches = [row for row in rows if Path(row["job_dir"]).resolve() == job]
    require(len(matches) == 1, "anchor job must have exactly one fresh campaign allocation")
    allocation = matches[0]
    env = read_env(job / "job.env")
    require(Path(env["ISGF_ROOT"]).resolve() == root, "anchor job root does not match campaign")
    fresh_source(job / "job.env", job=job, started_at=campaign_started_at)
    rows = []
    for line in (job / "results.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # a concurrent collector may still be writing the final line
    successes = [row for row in rows if row.get("success") is True]
    require(bool(successes), "anchor has no successful fresh demo")
    result = successes[0]
    iid = result["instance"]
    paths = list((job / "inst").rglob(iid + "-tro_state.json"))
    require(len(paths) == 1, "anchor TRO is missing or duplicated")
    tro_path = paths[0]
    sidecar_path = tro_path.with_name(iid + ".oracle.json")
    summary_path = tro_path.with_name(iid + ".summary.json")
    meta_path = job / "demo" / iid / "meta.json"
    demo_path = job / "demo" / iid / "demo.hdf5"
    sources = {name: fresh_source(path, job=job, started_at=campaign_started_at)
               for name, path in (("tro", tro_path), ("sidecar", sidecar_path), ("summary", summary_path),
                                  ("demo_meta", meta_path), ("demo", demo_path))}
    sources["job_env"] = fresh_source(job / "job.env", job=job, started_at=campaign_started_at)
    tro, sidecar, summary, demo = map(read_json, (tro_path, sidecar_path, summary_path, meta_path))
    require(tro["ispatialgym"]["instance_id"] == sidecar["instance_id"] == summary["instance_id"] == demo["instance_id"] == iid,
            "anchor source instance identities differ")
    require(sidecar["tro_sha256"] == sources["tro"]["sha256"], "anchor sidecar does not match TRO")
    require(demo.get("success") is True and demo.get("status") == "success" and demo.get("mode") == "oracle",
            "anchor is not a successful oracle demo")
    qa = demo.get("qa") or {}
    require(qa.get("enabled") is True and not qa.get("hard_triggered"), "anchor QA disabled or failed")
    require(not any(x.get("triggered") for x in (qa.get("metrics") or {}).values() if x.get("mode") == "hard"),
            "anchor has a hard QA failure")
    runtime = ((tro["ispatialgym"].get("campaign") or {}).get("runtime") or {})
    planned_goals = runtime.get("planned_goal_poses") or {}
    require((summary.get("goal_realised") or planned_goals or runtime.get("planned_goal_joints")) and summary.get("base_pose2d"),
            "anchor has no world goal/base proposal")
    tmeta = read_json(tro_path.parent.parent / "template_meta.json")
    return {"instance_id": iid, "allocation": allocation, "sources": sources, "tro": tro,
            "summary": summary, "template_meta": tmeta, "template_dir": str(tro_path.parent.parent),
            "planned_goal_poses": planned_goals,
            "planned_start_poses": runtime.get("planned_start_poses") or {},
            "bound_object_poses": runtime.get("bound_object_poses") or {},
            "planned_goal_joints": runtime.get("planned_goal_joints") or {},
            "joint_names": runtime.get("joint_names") or {},
            "grasp_library_root": env.get("ISPATIALGYM_GRASP_LIBRARY_V2"),
            "goal_proposal_source": "fresh_sampler_goal_realised", "anchor_is_final_member": False}


def load_template_contract(path, class_id):
    raw = read_json(path)
    rows = raw.get("templates", []) if isinstance(raw, dict) else raw
    matches = [row for row in rows if row.get("class_id") == class_id]
    require(len(matches) == 1, f"expected one ready contract for {class_id}")
    row = dict(matches[0])
    require(not row.pop("blockers", []), "template has unresolved blockers")
    expected_hash = row.pop("fingerprint", None)
    row["objects"] = tuple(ObjectRef(**obj) for obj in row["objects"])
    template = Template(**row)
    require(expected_hash is None or expected_hash == template.fingerprint, "template inventory fingerprint mismatch")
    return template


def bind_runtime_definition(template: Template, runtime_template_dir, *, runtime_metadata=None):
    """Pin fresh bytes only after the real parser proves task semantics unchanged.

    Goal and measure are compared separately: a gating atom cannot silently move
    into the measurement-only section even when content_key() considers it equal.
    """
    from ispatialgym.activity.problem import parse_problem_string

    runtime_dir = Path(runtime_template_dir).resolve()
    runtime_meta = runtime_metadata or read_json(runtime_dir / "template_meta.json")
    source_dir = Path(template.scene_setup["source_template"]).resolve()
    source_meta = read_json(source_dir / "template_meta.json")
    snapshots = []
    for directory, metadata, expected in (
        (source_dir, source_meta, template.scene_setup["source_bddl_sha256"]),
        (runtime_dir, runtime_meta, runtime_meta["bddl_sha256"]),
    ):
        path = (directory / metadata["activity_definition"]).resolve()
        require(path.is_file(), "runtime BDDL source absent")
        actual = file_digest(path)
        require(actual == expected, "BDDL bytes differ from pinned metadata")
        match = re.fullmatch(r"(.+)-(\d+)", metadata["problem"])
        require(match is not None, "invalid source BDDL problem identity")
        parsed = parse_problem_string(path.read_text(), match[1], int(match[2]))
        semantic = {"activity": parsed.activity, "domain": parsed.domain, "variant": parsed.variant,
                    "objects": parsed.objects, "groups": parsed.groups, "init": parsed.init,
                    "goal": parsed.goal, "measure": parsed.measure}
        snapshots.append({"path": str(path), "sha256": actual, "semantics": semantic,
                          "has_explicit_measure": bool(re.search(r"\(:measure\b", re.sub(r";[^\n]*", "", path.read_text())))})
    equivalent = snapshots[0]["semantics"] == snapshots[1]["semantics"]
    migration = None
    explicit_repair = EXPLICIT_ARTICULATION_MEASURE_REPAIRS.get((template.class_id, snapshots[0]["sha256"], snapshots[1]["sha256"]))
    if (not equivalent and template.task.startswith("configuration_matching-articulation_")
            and (not snapshots[0]["has_explicit_measure"] or explicit_repair)):
        before, after = snapshots[0]["semantics"], snapshots[1]["semantics"]
        moved = [atom for atom in before["goal"] if isinstance(atom, list) and atom[:1] == ["layout_reached"]]
        remaining = [atom for atom in before["goal"] if atom not in moved]
        same_sections = {k: v for k, v in before.items() if k != "goal"} == {k: v for k, v in after.items() if k != "goal"}
        if (moved and same_sections and remaining == after["goal"]
                and all(atom in before["measure"] and atom in after["measure"] for atom in moved)):
            migration = {"id": "2026-10-14_articulation_measure_migration", "semantic_equivalence": False,
                         "removed_gating_atoms": moved, "measurement_atoms_unchanged": True,
                         "policy_source": "ispatialgym/activity/problem.py:LEGACY_MEASURE_TASK_RULES",
                         "authorization": "campaign owner preserves the working current runtime scoring migration"}
            if explicit_repair:
                migration["exact_explicit_measure_repair"] = {"source_sha256": snapshots[0]["sha256"],
                    "runtime_sha256": snapshots[1]["sha256"], "current_runtime_validation": explicit_repair}
    require(equivalent or migration is not None, "runtime BDDL semantics differ from ready contract")
    proof = {"source": {k: v for k, v in snapshots[0].items() if k != "semantics"},
             "runtime": {k: v for k, v in snapshots[1].items() if k != "semantics"},
             "semantic_sha256": digest(snapshots[0]["semantics"]),
             "runtime_semantic_sha256": digest(snapshots[1]["semantics"]), "semantic_equivalence": equivalent, "migration": migration,
             "comparison": "parsed activity/domain/variant/objects/groups/init/goal/measure; goal and measure remain separate"}
    prior = template.scene_setup.get("runtime_bddl_equivalence")
    if isinstance(prior, dict) and isinstance(prior.get("runtime"), dict):
        # An exact copy in a new fresh job must not invent a new physical key.
        # Both new files were hashed and parsed above; the original proof stays
        # authoritative only while its referenced runtime bytes still match.
        same_location = {**proof, "runtime": {**proof["runtime"], "path": prior["runtime"].get("path")}}
        if same_location == prior and template.scene_setup.get("runtime_bddl_sha256") == snapshots[1]["sha256"]:
            old_runtime = Path(prior["runtime"]["path"])
            require(old_runtime.is_file() and file_digest(old_runtime) == snapshots[1]["sha256"],
                    "previously pinned runtime BDDL bytes changed")
            return template
    return replace(template, scene_setup={**template.scene_setup,
        "runtime_bddl_sha256": snapshots[1]["sha256"], "runtime_bddl_equivalence": proof})


def native_base_push_endpoint(anchor, target):
    """Use the recorded post-settle endpoint, not the stale initial draw summary."""
    from ispatialgym.sampling.campaign_pair_goals import pose

    source = anchor['sources']['tro']
    require(file_digest(source['path']) == source['sha256'], 'base-push source TRO changed after audit')
    runtime = read_json(source['path'])['ispatialgym'].get('campaign', {}).get('runtime', {})
    endpoint = runtime.get('planned_goal_poses', {}).get(target)
    require(endpoint is not None and endpoint == anchor.get('planned_goal_poses', {}).get(target),
            'base-push source lacks its recorded post-settle endpoint')
    require(isinstance(endpoint, (list, tuple)) and len(endpoint) == 2,
            'base-push source endpoint has invalid native pose format')
    pose({'position': endpoint[0], 'orientation': endpoint[1]}, 'recorded base-push endpoint')
    return endpoint, {'method':'recorded_native_post_settle_endpoint',
                      'source_tro':{'path':source['path'],'sha256':source['sha256']}}


def with_alignment_buckets(template: Template, native_range_deg, *, source):
    """Bind equal thirds of a measured/applied native range, without extending it."""
    require(template.task.startswith("alignment-"), "alignment bucket policy requires an alignment template")
    require(isinstance(native_range_deg, (list, tuple)) and len(native_range_deg) == 2,
            "alignment policy requires the applied native range")
    lo, hi = (float(v) for v in native_range_deg)
    require(math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi <= 180, "invalid native alignment range")
    require(isinstance(source, dict) and source.get("field") and source.get("sha256"), "alignment range needs source provenance")
    policy = {"metric": "actual_magnitude_deg", "unit": "degrees", "method": "equal_thirds_of_applied_native_range",
              "edges": [lo, lo + (hi-lo)/3, lo + 2*(hi-lo)/3, hi],
              "native_range_deg": [lo, hi], "intervals": "first_closed_then_lower_open_upper_closed", "source": source}
    existing = template.scene_setup.get("bucket_policy")
    if existing is not None:
        require({key: value for key, value in existing.items() if key != "source"}
                == {key: value for key, value in policy.items() if key != "source"}, "native alignment bucket policy changed")
        require(isinstance(existing.get("source"), dict) and existing["source"].get("field")
                and existing["source"].get("sha256"), "existing alignment policy lacks source provenance")
        # A fresh native observation confirms the frozen range without changing
        # the already validated contract's original evidence identity.
        return template
    return replace(template, scene_setup={**template.scene_setup, "bucket_policy": policy})


def anchor_perturbations(anchor, template, campaign_id):
    """Carry verified absolute training state into the fresh group's contract."""
    source = anchor["tro"]["ispatialgym"].get("campaign")
    if source is None:
        return Perturbations({obj.instance: ScaleState(obj.default_scale, obj.default_scale, "fixed") for obj in template.objects})
    require(source.get("campaign_id") == campaign_id and source.get("task") == template.task
            and source.get("class_id") == template.class_id, "anchor augmentation belongs to another campaign/task/class")
    raw = source["perturbations"]
    result = Perturbations({name: ScaleState(**value) for name, value in raw["scales"].items()},
                           tuple(raw.get("removed_objects", [])), raw.get("lighting", {}))
    require(set(result.scales) == {obj.instance for obj in template.objects}, "anchor augmentation has different object bindings")
    observed = source.get("runtime", {}).get("observed_scales", {})
    for obj in template.objects:
        state = result.scales[obj.instance]
        require(state.default == obj.default_scale, "anchor default scale differs from the ready model contract")
        actual = observed.get(obj.instance, [])
        require(len(actual) == 3 and all(isinstance(a, (int, float)) and math.isfinite(a)
                and abs(a-b) <= 1e-6*max(1, abs(b)) for a, b in zip(actual, state.applied)),
                "anchor applied scale lacks matching live verification")
    require(source["validation_key"]["scale_sha256"] == result.scale_fingerprint, "anchor scale key differs from exact applied state")
    result.validate(template.split, result.scales)
    if result.perturbed:
        require(source["split"] == "train", "perturbed anchor did not originate in training")
    return result


def bind_scaled_runtime_identity(anchor, template, perturbations):
    """Rebind only the native choice ID implied by verified training scales."""
    runtime_id = anchor["template_meta"]["template_id"]
    declared = template.scene_setup.get("runtime_template_id")
    if declared is None or declared == runtime_id:
        return template
    require(template.split == "train" and any(s.default != s.applied for s in perturbations.scales.values()),
            "runtime template ID mismatch without verified training scale change")
    from ispatialgym.sampling.v2.problem import TemplateChoice

    source_path = Path(template.scene_setup["source_template"]) / "template_meta.json"
    original = read_json(source_path)["choice"]
    current = anchor["template_meta"]["choice"]
    require(TemplateChoice.from_json(original).template_id == declared,
            "source choices do not reproduce the declared runtime template ID")
    require(TemplateChoice.from_json(current).template_id == runtime_id,
            "fresh choices do not reproduce the actual runtime template ID")
    expected = copy.deepcopy(original)
    expected["template_id"] = runtime_id
    rows = [(model, current["models"].get(role, {})) for role, model in expected["models"].items()]
    for role, members in expected.get("groups", {}).items():
        actual = current.get("groups", {}).get(role, [])
        require(len(members) == len(actual), "scaled native group changes its members")
        rows.extend(zip(members, actual))
    routes = []
    for model, actual in rows:
        refs = [obj for obj in template.objects if (obj.category, obj.model) == (model["category"], model["model"])]
        require(refs, "scaled native choice has no matching ready object binding")
        states = {perturbations.scales[obj.instance] for obj in refs}
        require(len(states) == 1, "scaled native choice has ambiguous instance scales")
        state = states.pop()
        if state.default != state.applied:
            model["scale"] = list(state.applied)
            if model.get("grasp_key") != actual.get("grasp_key"):
                routes.append(verify_scaled_grasp_route(anchor, model, actual, state))
                model["grasp_key"] = actual["grasp_key"]
    require(expected == current, "runtime choices differ beyond verified training scales")
    proof = {"source_runtime_template_id": declared, "source_metadata": str(source_path),
             "source_choice_sha256": digest(original), "runtime_choice_sha256": digest(current),
             "scale_sha256": perturbations.scale_fingerprint,
             "comparison": "native template choices unchanged except live-verified training scales"}
    if routes:
        proof["grasp_routes"] = routes
    return replace(template, scene_setup={**template.scene_setup, "runtime_template_id": runtime_id,
                                         "scaled_runtime_identity": proof})


def verify_scaled_grasp_route(anchor, original, current, state):
    """Require the actual native alias reader to resolve both physical scales."""
    from ispatialgym.sampling import grasp_library as G
    from ispatialgym.grasping import schema as S

    root = anchor.get("grasp_library_root")
    require(root and Path(root).is_dir(), "scaled grasp route lacks its fresh anchor library root")
    result = {}
    for label, model, scale in (("source", original, state.default), ("scaled", current, state.applied)):
        key = model.get("grasp_key")
        require(isinstance(key, str) and "@" not in key, "scaled grasp route requires a plain native alias")
        path = G._v2_index(root).get(key)
        require(path is not None, "scaled grasp route is absent from the native alias index")
        data = read_json(path)
        require((data["category"], data["model"], data["dataset"]) ==
                (model["category"], model["model"], model.get("dataset", "behavior-1k-assets")),
                "scaled grasp alias resolves to a different model")
        lib = S.load_model(data["category"], data["model"], data.get("scale"), root,
            constraint=data.get("constraint") or "tabletop",
            rest_hint_wxyz=data.get("rest_hint_wxyz") if "rest_hint_wxyz" in data else G._v2_hint(path))
        require(lib is not None and Path(lib.path).resolve() == Path(path).resolve(), "native grasp loader resolved different bytes")
        S.check_compatible(lib, scale=scale)
        G._check_v2_policy(lib, data)
        require(bool(G.grasps_from_v2(lib, key)), "scaled grasp alias has no positive native grasps")
        result[label] = {"key": key, "library": str(Path(path).resolve()), "sha256": file_digest(path),
                         "scale": list(scale), "constraint": lib.data.get("constraint") or "tabletop"}
    require(result["source"]["constraint"] == result["scaled"]["constraint"], "scaled grasp route changes its constraint")
    return result


def prepare_group(*, anchor, template: Template, campaign_id: str, group_id: str, output: Path,
                  code_sha256: str, physics_sha256: str, data_sha256: str,
                  magnitude_range, budget_seconds: float = 900, max_members: int = 3,
                  reuse_anchor_goal: bool = False):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", group_id), "group id must be a safe path component")
    pair_tasks = {f"inter_object_relation_satisfaction-{relation}_{scale}_scale": (relation, scale)
                  for scale in ("small", "large") for relation in PAIR_RELATIONS}
    joint_task = template.task.startswith("configuration_matching-articulation_")
    block_task = template.task == "configuration_matching-block_design"
    require(block_task or joint_task or template.task in ("object_scaling", "mental_rotation", "alignment-world_vertical", "alignment-axial", "disembedding", "configuration_matching-attachment") or template.task in pair_tasks,
            "unsupported goal author task")
    require(template.split in ("train", "eval_standard", "eval_model", "eval_category"), "invalid group split")
    bindings = anchor["tro"]["ispatialgym"]["object_bindings"]
    for ref in template.objects:
        binding = bindings.get(ref.instance)
        require(binding is not None and binding.get("category") == ref.category and binding.get("model") == ref.model,
                f"anchor object identity mismatch: {ref.instance}")
    tm = anchor["template_meta"]
    require(tm["scene"] == template.scene, "anchor scene differs from contract")
    require(template.room in anchor["tro"]["ispatialgym"]["scene"]["task_rooms"], "anchor room differs from contract")
    runtime_id = tm["template_id"]
    perturbations = anchor_perturbations(anchor, template, campaign_id)
    template = bind_scaled_runtime_identity(anchor, template, perturbations)
    if template.scene_setup.get("source_bddl_sha256"):
        template = bind_runtime_definition(template, anchor["template_dir"], runtime_metadata=tm)
    if template.task in ("alignment-world_vertical", "alignment-axial"):
        applied = tm.get("template_facts", {}).get("deviation_range_applied_deg")
        template = with_alignment_buckets(template, applied, source={"path": str(Path(anchor["template_dir"]) / "template_meta.json"),
            "sha256": digest(tm), "digest_kind": "canonical_json", "field": "template_facts.deviation_range_applied_deg"})
    now = time.time()
    source_goal = anchor["summary"].get("goal_realised")
    if source_goal is None:
        targets = [ref.instance for ref in template.objects if ref.role == "target"]
        require(len(targets) == (4 if block_task else 1), "anchor has no unambiguous target goal")
        pose = anchor["planned_goal_poses"].get(targets[0])
        if pose is None and joint_task:
            target = anchor["tro"]["ispatialgym"].get("pose_targets", {}).get(targets[0], {})
            require(target.get("frame", "world") in (None, "world"), "articulation root goal must be in world coordinates")
            if target.get("position") is not None and target.get("orientation") is not None:
                pose = (target["position"], target["orientation"])
            else:
                pose = anchor.get("bound_object_poses", {}).get(targets[0])
        require(pose is not None, "anchor lacks a world goal/root pose")
        source_goal = pose if isinstance(pose, dict) else {"position": pose[0], "orientation": pose[1]}
    plan = {"schema": SCHEMA, "campaign_id": campaign_id, "group_id": group_id, "split": template.split,
            "task": template.task, "template_id": runtime_id, "contract_template_sha256": template.fingerprint,
            "scene": template.scene, "room": template.room,
            "goal_pose": {"position": source_goal["position"], "orientation": source_goal["orientation"]},
            "base_pose2d": anchor["summary"]["base_pose2d"], "magnitude_range": list(magnitude_range),
            "created_at": now, "budget_seconds": budget_seconds, "max_members": max_members,
            "source_instance_id": anchor["instance_id"], "source_instance_is_member": False,
            "anchor_sources": anchor["sources"]}
    if block_task:
        targets = [ref.instance for ref in template.objects if ref.role == "target"]
        boards = [ref.instance for ref in template.objects if ref.semantic_kind == "board"]
        require(len(targets) == 4 and len(boards) == 1, "block anchor requires four blocks and one board")
        require(len({(ref.model_key, ref.asset_sha256, ref.default_scale) for ref in template.objects if ref.role == "target"}) == 1,
                "block contract must use four identical models")
        groups = [value for value in anchor["tro"]["ispatialgym"].get("groups", {}).values() if set(value.get("members", [])) == set(targets)]
        require(len(groups) == 1, "block anchor lacks one four-block goal group")
        slots = groups[0].get("target_slots", {})
        require(set(slots) == {f"slot_{i}" for i in range(4)}, "block anchor lacks four explicit slots")
        summary = anchor["summary"]
        order = summary.get("placement_order", [])
        require(len(order) == 4 and set(order) == set(targets), "block anchor lacks slot-ordered block identities")
        board = anchor["planned_start_poses"].get(boards[0]) or anchor.get("bound_object_poses", {}).get(boards[0])
        require(board is not None and all(name in anchor["planned_start_poses"] for name in targets), "block anchor lacks start/board poses")
        require(all(name in summary.get("up_faces", {}) and name in summary.get("flip_directions", {}) for name in targets),
                "block anchor lacks initial face and flip profile")
        plan.update(schema=BLOCK_SCHEMA, target_names=targets, board_name=boards[0],
                    board_pose={"position": board[0], "orientation": board[1]},
                    goal_slots=[slots[f"slot_{i}"] for i in range(4)], pattern_id=summary.get("pattern"), pattern=summary.get("pattern_faces"),
                    initial_orientations=[anchor["planned_start_poses"][name][1] for name in order],
                    initial_up_faces=[summary["up_faces"][name] for name in order],
                    flip_directions=[summary["flip_directions"][name] for name in order])
        plan.pop("goal_pose")
        plan.pop("magnitude_range")
        plan["anchor_base_pose2d"] = plan.pop("base_pose2d")
        validate_block_plan(plan, template_id=runtime_id, scene=template.scene, rooms=[template.room], activated_at=now, now=now)
    elif template.task == "configuration_matching-attachment":
        roles = {role: [ref.instance for ref in template.objects if ref.role == role] for role in ("target", "partner")}
        require(all(len(names) == 1 for names in roles.values()), "ambiguous attachment target/partner binding")
        plan["target_name"], plan["partner_name"] = roles["target"][0], roles["partner"][0]
        partner = anchor["planned_start_poses"].get(plan["partner_name"]) or anchor.get("bound_object_poses", {}).get(plan["partner_name"])
        require(partner is not None, "attachment anchor lacks recorded partner pose")
        plan["partner_pose"] = {"position": partner[0], "orientation": partner[1]}
        plan["schema"] = ATTACHMENT_SCHEMA
        plan.pop("magnitude_range")
        plan["anchor_base_pose2d"] = plan.pop("base_pose2d")
        if template.class_id in {'configuration_matching-attachment_03', 'configuration_matching-attachment_04'}:
            proof = anchor['sources']['summary']
            require(file_digest(proof['path']) == proof['sha256'], 'attachment source summary changed after audit')
            stance = anchor['summary'].get('stances', {}).get('manipulation')
            require(isinstance(stance, dict), 'attachment source lacks recorded manipulation stance')
            plan['anchor_manipulation_stance'] = {
                **{k: stance[k] for k in ('base_pose2d', 'focus', 'translate', 'tilt')},
                'source_summary_sha256': proof['sha256']}
        validate_attachment_plan(plan, template_id=runtime_id, scene=template.scene, rooms=[template.room], activated_at=now, now=now)
    elif joint_task:
        targets = [ref for ref in template.objects if ref.role == "target"]
        require(len(targets) == 1 and targets[0].joint, "articulation contract lacks one target joint")
        target, joint = targets[0].instance, targets[0].joint
        require(anchor["summary"].get("joint") == joint, "anchor articulation joint differs from contract")
        names = anchor.get("joint_names", {}).get(target)
        values = anchor.get("planned_goal_joints", {}).get(target)
        require(names and values is not None and len(names) == len(values), "anchor lacks complete joint-goal vector")
        joints = dict(zip(names, values))
        require(joint in joints, "target joint absent from recorded joint vector")
        facts = tm.get("template_facts", {})
        if anchor["summary"].get("direction") == "close":
            limits = facts.get("limits")
            require(limits and len(limits) == 2 and limits[0] < limits[1], "closing anchor lacks authoritative joint limits")
            joints[joint] = float(limits[0])
        elif template.task.endswith("_large_scale"):
            limits = facts.get("limits")
            require(limits and len(limits) == 2 and limits[0] < limits[1], "opening anchor lacks authoritative joint limits")
            lo, hi = plan["magnitude_range"]
            require(0 <= lo < hi <= 1, "opening goal bucket is invalid")
            if reuse_anchor_goal:
                fraction = (joints[joint]-float(limits[0]))/(float(limits[1])-float(limits[0]))
                require(lo <= fraction <= hi, "verified opening anchor is outside the requested strict bucket")
                method = "verified_anchor_joint_goal_before_starts"
            else:
                uniform = (int(digest({"group": group_id, "anchor": anchor["instance_id"], "joint": joint})[:16], 16)+.5)/(2**64)
                fraction = lo+(hi-lo)*uniform
                method = "uniform_requested_fraction_before_starts"
            plan["goal_joint_proposal"] = {"method": method,
                "anchor_value": joints[joint], "fraction": fraction, "requires_native_passive_stability": True}
            if not reuse_anchor_goal:
                joints[joint] = float(limits[0])+fraction*(float(limits[1])-float(limits[0]))
        plan.update(schema=JOINT_SCHEMA, target=target, joint=joint, goal_joints=joints, fraction_range=plan.pop("magnitude_range"))
        plan["anchor_base_pose2d"] = plan.pop("base_pose2d")
        if template.class_id == 'configuration_matching-articulation_open_small_scale_02':
            proof = anchor['sources']['summary']
            require(file_digest(proof['path']) == proof['sha256'], 'carton source summary changed after audit')
            source_stance = anchor['summary'].get('stances', {}).get('manipulation')
            require(isinstance(source_stance, dict), 'carton source lacks recorded manipulation stance')
            plan['anchor_manipulation_stance'] = {
                **{k: source_stance[k] for k in ('base_pose2d', 'focus', 'translate', 'tilt')},
                'source_summary_sha256': proof['sha256']}
        validate_joint_plan(plan, template_id=runtime_id, scene=template.scene, rooms=[template.room],
                            target=target, joint=joint, activated_at=now, now=now)
    elif template.task in pair_tasks:
        relation, scale = pair_tasks[template.task]
        roles = {role: [b["name"] for b in tm.get("bindings", []) if b.get("role") == role] for role in ("target", "partner")}
        require(all(len(names) == 1 for names in roles.values()), "ambiguous pair target/partner binding")
        plan["target_name"], plan["partner_name"] = roles["target"][0], roles["partner"][0]
        partner = anchor["planned_start_poses"].get(plan["partner_name"]) or anchor.get("bound_object_poses", {}).get(plan["partner_name"])
        require(partner is not None, "pair anchor lacks recorded partner pose")
        settled = anchor["summary"].get("goal_check", {}).get("settled_poses", {})
        target_scene_name = bindings[plan["target_name"]].get("scene_object_name")
        target = settled.get(target_scene_name)
        require(target is not None, "pair anchor lacks a settled target goal bound to its scene object")
        plan["release_pose"] = plan["goal_pose"]
        plan["goal_pose"] = {"position": target[0], "orientation": target[1]}
        plan["partner_pose"] = {"position": partner[0], "orientation": partner[1]}
        if template.class_id == 'inter_object_relation_satisfaction-inside_large_scale_03' and anchor['summary'].get('release_tilt_deg'):
            proof = anchor['sources']['summary']
            require(file_digest(proof['path']) == proof['sha256'], 'sink release source summary changed after audit')
            require(anchor['summary'].get('release_level_quat') is not None, 'sink source lacks native level attitude')
            plan['class_id'] = template.class_id
            plan['inside_release_attitude'] = {
                'tilt_deg': anchor['summary']['release_tilt_deg'],
                'level_quat': anchor['summary']['release_level_quat'],
                'source_summary_sha256': proof['sha256']}
        from ispatialgym.sampling.campaign_release_clearance import settings as release_clearance
        if release_clearance(template) is not None:
            proof = anchor['sources']['summary']
            require(file_digest(proof['path']) == proof['sha256'], 'clearance source summary changed after audit')
            require(isinstance(anchor['summary'].get('raised_release'), dict),
                    'clearance source lacks measured native raised release')
            plan['release_policy'] = 'native_ontop_clearance'
            plan['clearance_recovery_class'] = template.class_id
        if relation == "inside" and scale == "small":
            plan["release_policy"] = "native_grasp_height"
        if relation == 'nextto' and scale == 'small' and template.class_id in {
                'inter_object_relation_satisfaction-nextto_01', 'inter_object_relation_satisfaction-nextto_03'}:
            proof = anchor['sources']['summary']
            require(file_digest(proof['path']) == proof['sha256'], 'raised-release source summary changed after audit')
            require(anchor['summary'].get('raised_release') is not None, 'nextto source lacks native raised-release evidence')
            plan['release_policy'] = 'native_raised_height'
        from ispatialgym.sampling.campaign_tabletop_starts import source_stance_hint_enabled
        if source_stance_hint_enabled(relation, scale, template.class_id):
            source_stance = anchor['summary'].get('manipulation_stance')
            if source_stance is not None:
                proof = anchor['sources']['summary']
                require(file_digest(proof['path']) == proof['sha256'], 'pair source summary changed after audit')
                plan['anchor_manipulation_stance'] = {
                    **{k: source_stance[k] for k in ('base_pose2d', 'focus', 'translate', 'tilt')},
                    'source_summary_sha256': proof['sha256']}
        if relation == "touching" and scale == "small":
            from ispatialgym.sampling.campaign_pair_goals import object_pose_from_grasp

            require(file_digest(anchor["sources"]["sidecar"]["path"]) == anchor["sources"]["sidecar"]["sha256"],
                    "touching anchor sidecar changed after audit")
            sidecar = read_json(anchor["sources"]["sidecar"]["path"])
            grasps = [row for row in sidecar.get("grasps", []) if row.get("object") == plan["target_name"]]
            require(len(grasps) == 1 and grasps[0].get("place") and grasps[0].get("eef_in_object"),
                    "touching anchor lacks one verified pre-push grasp/place")
            placed = object_pose_from_grasp(grasps[0]["place"], grasps[0]["eef_in_object"])
            plan["place_pose"] = {"position": placed[0], "orientation": placed[1]}
            if template.class_id == 'inter_object_relation_satisfaction-touching_03':
                from ispatialgym.sampling.campaign_shelf_goals import author_context
                proof = anchor['sources']['summary']
                require(file_digest(proof['path']) == proof['sha256'], 'shelf source summary changed after audit')
                require(grasps[0].get('place_kind') == 'shelf_tip', 'shelf source sidecar lacks native tip-release proof')
                supports = [b['scene_object_name'] for b in tm['bindings']
                            if b.get('role') == 'support' and b.get('category') == 'bookcase']
                require(len(supports) == 1, 'shelf source lacks one bookcase support')
                plan['class_id'] = template.class_id
                plan['shelf_tip_context'] = author_context(anchor, target_name=plan['target_name'], support_name=supports[0])
        if relation == "nextto" and scale == "small" and not reuse_anchor_goal:
            summary = anchor["summary"]
            require("nextto_gap_m" in summary and "g_NT" in summary.get("pair", {}), "nextto anchor lacks measured gap range")
            base = summary.get("manipulation_stance", {}).get("base_pose2d")
            require(base and len(base) == 3, "nextto anchor lacks pair lateral frame")
            changed_goal, changed_release, gap = vary_nextto_gap(target, (source_goal["position"], source_goal["orientation"]), partner,
                gap_max=summary["pair"]["g_NT"], observed_gap=summary["nextto_gap_m"], base_yaw=base[2],
                fraction=int(digest({"group": group_id, "anchor": anchor["instance_id"]})[:16], 16) / (2 ** 64 - 1))
            plan["goal_pose"] = {"position": changed_goal[0], "orientation": changed_goal[1]}
            plan["release_pose"] = {"position": changed_release[0], "orientation": changed_release[1]}
            plan["nextto_gap_proposal_m"] = gap
        if scale == "large" and relation == "under":
            from ispatialgym.sampling.campaign_pair_goals import object_pose_from_grasp

            require(file_digest(anchor["sources"]["sidecar"]["path"]) == anchor["sources"]["sidecar"]["sha256"],
                    "push-under anchor sidecar changed after audit")
            sidecar = read_json(anchor["sources"]["sidecar"]["path"])
            grasps = [row for row in sidecar.get("grasps", []) if row.get("object") == plan["target_name"]]
            push = sidecar.get("contacts", {}).get("push", {})
            require(len(grasps) == 1 and grasps[0].get("place") and grasps[0].get("eef_in_object")
                    and push.get("kind") == "push_under" and push.get("object") == plan["target_name"],
                    "push-under anchor lacks its native drop and push")
            released = object_pose_from_grasp(grasps[0]["place"], grasps[0]["eef_in_object"])
            native_goal = anchor["planned_goal_poses"].get(plan["target_name"])
            require(native_goal is not None and len(native_goal) == 2, "push-under anchor lacks its native endpoint")
            direction = push.get("direction")
            require(isinstance(direction, list) and len(direction) == 3 and abs(direction[2]) < 1e-6,
                    "push-under anchor has no horizontal push direction")
            placement = sidecar.get("stances", {}).get("placement", {})
            edge = placement.get("edge") or placement.get("extras", {}).get("edge")
            require(edge, "push-under anchor has no native placement edge")
            plan.update(placement_mode="drop_push", floor_push={"edge":edge,"direction":direction[:2]},
                        goal_cell=push.get("goal_cell"),
                        release_pose={"position":native_goal[0],"orientation":native_goal[1]},
                        place_pose={"position":released[0],"orientation":released[1]})
            require(push.get("drop_cell") is not None and math.dist(push["drop_cell"], released[0][:2]) <= .005,
                    "push-under native contact and release disagree on drop cell")
            from ispatialgym.sampling.campaign_floor_push_proposals import author_hint
            proposal = author_hint(anchor, template.class_id)
            if proposal is not None:
                plan.update(class_id=template.class_id, floor_push_source_proposal=proposal)
        elif scale == "large" and relation == "touching" and (anchor["summary"].get("surface_touching") or anchor["summary"].get("push_touching")):
            from ispatialgym.sampling.campaign_pair_goals import object_pose_from_grasp

            require(file_digest(anchor["sources"]["sidecar"]["path"]) == anchor["sources"]["sidecar"]["sha256"],
                    "surface touching anchor sidecar changed after audit")
            sidecar = read_json(anchor["sources"]["sidecar"]["path"])
            grasps = [row for row in sidecar.get("grasps", []) if row.get("object") == plan["target_name"]]
            require(len(grasps) == 1 and grasps[0].get("place") and grasps[0].get("eef_in_object"),
                    "surface touching anchor lacks its recorded release")
            released = object_pose_from_grasp(grasps[0]["place"], grasps[0]["eef_in_object"])
            if anchor["summary"].get("surface_touching"):
                plan["release_pose"] = {"position": released[0], "orientation": released[1]}
                plan["placement_mode"] = "surface"
            else:
                native_goal = anchor["planned_goal_poses"].get(plan["target_name"])
                require(native_goal is not None and len(native_goal) == 2, "drop-push anchor lacks its native contact endpoint")
                plan["release_pose"] = {"position": native_goal[0], "orientation": native_goal[1]}
                plan["place_pose"] = {"position": released[0], "orientation": released[1]}
                plan["placement_mode"] = "drop_push"
            require(anchor["summary"].get("goal_cell") is not None, "surface touching anchor lacks fixed goal cell")
            plan["goal_cell"] = anchor["summary"]["goal_cell"]
        elif scale == "large" and relation in ("nextto", "touching"):
            from ispatialgym.sampling.campaign_pair_goals import vary_base_push_gap

            push = anchor["summary"].get("nextto", {})
            sides = tm.get("template_facts", {}).get("nextto", {}).get("sides", [])
            matching = [row for row in sides if row["name"] == push.get("side")]
            require(len(matching) == 1 and push.get("direction") and push.get("face"),
                    "large push anchor lacks audited push and partner side geometry")
            endpoint, endpoint_proof = native_base_push_endpoint(anchor, plan['target_name'])
            plan['base_push_source_endpoint'] = endpoint_proof
            plan["release_pose"] = {"position": endpoint[0], "orientation": endpoint[1]}
            plan["base_push"] = {key: push[key] for key in ("side", "face", "direction", "goal_gap_m", "nextto_bound_m")}
            plan["base_push"]["side_inward"] = matching[0]["inward"]
            if relation == "nextto" and not reuse_anchor_goal:
                changed_goal, changed_release, gap = vary_base_push_gap(target,
                    endpoint, inward=matching[0]["inward"],
                    gap_max=push["nextto_bound_m"], observed_gap=push["goal_gap_m"],
                    fraction=int(digest({"group": group_id, "anchor": anchor["instance_id"]})[:16], 16) / (2 ** 64 - 1))
                plan["goal_pose"] = {"position": changed_goal[0], "orientation": changed_goal[1]}
                plan["release_pose"] = {"position": changed_release[0], "orientation": changed_release[1]}
                plan["base_push"]["goal_gap_m"] = gap
        elif scale == "large":
            require(anchor["summary"].get("goal_cell") is not None, "large pair anchor lacks fixed goal cell")
            plan["goal_cell"] = anchor["summary"]["goal_cell"]
        plan["goal_selection"] = "fresh_anchor" if reuse_anchor_goal else "native_region"
        plan["schema"] = PAIR_SCHEMA
        plan.pop("magnitude_range")
        plan["anchor_base_pose2d"] = plan.pop("base_pose2d")
        validate_pair_plan(plan, template_id=runtime_id, scene=template.scene, rooms=[template.room],
                           relation=relation, scale=scale, activated_at=now, now=now)
    elif template.task == "disembedding":
        roles = {}
        for binding in tm.get("bindings", []):
            roles.setdefault(binding.get("role"), []).append(binding["name"])
        for field, aliases in (("target_name", ("target",)), ("source_name", ("source", "tray_source")),
                               ("destination_name", ("destination", "tray_dest"))):
            names = [n for role in aliases for n in roles.get(role, [])]
            require(len(names) == 1, f"ambiguous {field} binding")
            plan[field] = names[0]
        for field, name in (("source_pose", plan["source_name"]), ("destination_pose", plan["destination_name"])):
            value = anchor["planned_start_poses"].get(name)
            require(value is not None, f"fresh anchor has no recorded tray pose: {name}")
            plan[field] = {"position": value[0], "orientation": value[1]}
        if reuse_anchor_goal:
            value = anchor["planned_goal_poses"].get(plan["target_name"])
            require(value is not None, "tray goal reuse requires a recorded target goal")
            plan["goal_pose"] = value if isinstance(value, dict) else {"position": value[0], "orientation": value[1]}
            plan["goal_selection"] = "fresh_anchor"
        else:
            candidates = anchor["summary"].get("campaign_goal_space", {}).get("target_poses", [])
            require(candidates, "disembedding needs a fresh anchor with verified destination-floor goal candidates")
            plan["goal_pose"] = candidates[int(digest({"group": group_id, "anchor": anchor["instance_id"]}), 16) % len(candidates)]
            plan["goal_selection"] = "native_region"
        plan["schema"] = TRAY_SCHEMA
        plan["crowdedness_range"] = plan.pop("magnitude_range")
        plan["anchor_base_pose2d"] = plan.pop("base_pose2d")
        validate_tray_plan(plan, template_id=runtime_id, scene=template.scene, rooms=[template.room], activated_at=now, now=now)
    elif template.task in ("alignment-world_vertical", "alignment-axial"):
        plan["schema"] = SURFACE_SCHEMA
        plan["magnitude_range_deg"] = plan.pop("magnitude_range")
        plan["anchor_base_pose2d"] = plan.pop("base_pose2d")  # Robot starts vary within this goal group.
        validate_surface_plan(plan, template_id=runtime_id, scene=template.scene, rooms=[template.room], activated_at=now, now=now)
    else:
        validate_goal_plan(plan, task=template.task, template_id=runtime_id, scene=template.scene, rooms=[template.room], now=now)
    key = ValidationKey(template.fingerprint, code_sha256, physics_sha256, data_sha256, perturbations.scale_fingerprint)
    config = {"campaign_id": campaign_id, "split": template.split, "purpose": "validation",
              "template": asdict(template), "perturbations": asdict(perturbations), "validation_key": asdict(key)}
    source = {"campaign_id": campaign_id, "group_id": group_id, "anchor_instance_id": anchor["instance_id"],
              "anchor_allocation": anchor["allocation"], "anchor_sources": anchor["sources"],
              "anchor_is_final_member": False, "goal_proposal_source": anchor["goal_proposal_source"],
              "template_dir": anchor["template_dir"], "created_at": now, "status": "validation_candidates_only"}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("goal_plan.json", plan), ("campaign_config.json", config), ("source.json", source),
                       ("empty_eval_index.json", {"schema": "isg-instance-index-v1", "n": 0, "instances": []})):
        write_json_atomic(output / name, data)
    return {"group_id": group_id, "directory": str(output.resolve()), "plan_sha256": digest(plan),
            "validation_key": asdict(key), "status": "ready_for_group_validation"}


def submission_command(group_dir, *, snapshot, campaign_root, v2_root, base_index=10000000,
                       account="vision", partition="svl", gres="titanrtx", extra_env=()):
    group = Path(group_dir).resolve()
    root, code = Path(campaign_root).resolve(), Path(snapshot).resolve()
    require(group.is_relative_to(root), "group state must be inside campaign root")
    plan, config, source = (read_json(group / name) for name in ("goal_plan.json", "campaign_config.json", "source.json"))
    require(config["purpose"] == "validation", "this submitter only launches candidate validation")
    require(base_index >= 10000000, "group indices must use a fresh range separate from baseline canaries")
    require((code / "scripts/trainfast/submit_fast.py").is_file(), "snapshot missing trainfast submitter")
    identity = code / ".snapshot_commit"
    if not identity.exists():
        identity = code / ".snapshot_isg"
    require(identity.is_file() and file_digest(identity) == config["validation_key"]["code_sha256"],
            "group code identity differs from requested frozen snapshot")
    key = source["anchor_allocation"]["key"]
    n = int(plan["max_members"])
    budget = float(plan.get("budget_seconds", 900))
    env = {"ISPATIALGYM_CAMPAIGN_GOAL_PLAN": str(group / "goal_plan.json"),
           "ISPATIALGYM_CAMPAIGN_CONFIG": str(group / "campaign_config.json"),
           "TEMPLATE": source["template_dir"], "EVAL_INDEX": str(group / "empty_eval_index.json"),
           "ISGF_COLLECT_EXTRA": "--no-goal-override" + (" --max-steps 6000" if plan["task"] == "configuration_matching-block_design" else ""),
           "ISGF_SAMPLING_CACHE": str(root / "cache/groups" / plan["group_id"]),
           "ISGF_ASSET_CACHE": str(root / "asset_cache"), "ISGF_JB_NFS": str(root / "jobs"),
           "ISGF_PREDEMO_ROOT": str(root), "KIT_TPL_NFS_ROOT": str(root / "kit_templates"),
           "KIT_TEX_STORE": str(root / "kit_textures"), "ISGF_EXP_PATH": str(root / "isaacsim_overlay/apps"),
           "ISPATIALGYM_FAST_BAND_SHARE": "0", "ISPATIALGYM_FAST_COLLECTOR_PREBUILD": "0"}
    protected = {"ISPATIALGYM_CAMPAIGN_GOAL_PLAN", "ISPATIALGYM_CAMPAIGN_CONFIG", "TEMPLATE", "EVAL_INDEX", "ISGF_COLLECT_EXTRA"}
    from ispatialgym.sampling.campaign_visibility import required as visibility_required
    data = dict(config['template'])
    data['objects'] = tuple(ObjectRef(**obj) for obj in data['objects'])
    if visibility_required(Template(**data)):
        # fast_env defaults this screen off for most task families. A validated
        # source's exact visibility requirement must survive sibling dispatch.
        env['ISGF_GOAL_OCCLUSION'] = 'on'
        env['ISPATIALGYM_GOAL_OCCLUSION'] = 'on'
        # Earlier immutable runtimes can remove sensors before that screen.
        # Retain cameras explicitly, including when reusing their exact source.
        env['ISPATIALGYM_FAST_SAMPLER_CAMERAS'] = '0'
        protected.update(('ISGF_GOAL_OCCLUSION', 'ISPATIALGYM_GOAL_OCCLUSION',
                          'ISPATIALGYM_FAST_SAMPLER_CAMERAS'))
    for item in extra_env:
        name, value = item.split("=", 1)
        require(name not in protected or env.get(name) == value, f"cannot override group invariant {name}")
        env[name] = value
    cmd = [sys.executable, str(code / "scripts/trainfast/submit_fast.py"), "submit", key, str(n), "--flow", "fast", "--code", str(code),
           "--root", str(root), "--ledger", str(root / "manifests/goal_group_allocations.tsv"), "--base-index", str(base_index),
           "--tag", "goal_groups", "--job-prefix", "isg261110_", "--grasps", "v2only", "--v2-root", str(v2_root),
           "--alias-memory", "off", "--no-memory-aggregate", "--kcollect", "2", "--sample-limit-s", str(int(budget + 1800)),
           # job_fast stops collector intake at JOB_END - DEMO_LIMIT_S - 420.
           # Include the sampler's 1800s startup allowance and a full 3600s
           # demo window for every class, not only block design. Otherwise a
           # 900s group can lose its collectors after just 480s (job1409).
           "--time-s", str(int(budget + 7200)),
           "--acct", account, "--part", partition, "--gres", gres]
    for name, value in env.items():
        cmd += ["--extra-env", f"{name}={value}"]
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prepare = subs.add_parser("prepare")
    for name in ("anchor-job", "campaign-root", "anchor-ledger", "contracts", "class-id", "campaign-id", "group-id", "out",
                 "code-identity-file", "physics-spec", "data-manifest"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--campaign-started-at", required=True, type=float)
    prepare.add_argument("--magnitude-range", nargs=2, required=True, type=float)
    prepare.add_argument("--budget-seconds", type=float, default=900)
    prepare.add_argument("--max-members", type=int, default=3)
    submit = subs.add_parser("submit")
    for name in ("group-dir", "snapshot", "campaign-root", "v2-root"):
        submit.add_argument("--" + name, required=True)
    submit.add_argument("--base-index", type=int, default=10000000)
    submit.add_argument("--acct", default="vision")
    submit.add_argument("--part", default="svl")
    submit.add_argument("--gres", default="titanrtx")
    submit.add_argument("--extra-env", action="append", default=[])
    submit.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        anchor = successful_anchor(args.anchor_job, campaign_root=args.campaign_root, ledger=args.anchor_ledger,
                                   campaign_started_at=args.campaign_started_at)
        result = prepare_group(anchor=anchor, template=load_template_contract(args.contracts, args.class_id),
                               campaign_id=args.campaign_id, group_id=args.group_id, output=Path(args.out),
                               code_sha256=file_digest(args.code_identity_file), physics_sha256=file_digest(args.physics_spec),
                               data_sha256=file_digest(args.data_manifest), magnitude_range=args.magnitude_range,
                               budget_seconds=args.budget_seconds, max_members=args.max_members)
        print(json.dumps(result, indent=2))
    else:
        cmd = submission_command(args.group_dir, snapshot=args.snapshot, campaign_root=args.campaign_root,
                                 v2_root=args.v2_root, base_index=args.base_index, account=args.acct,
                                 partition=args.part, gres=args.gres, extra_env=args.extra_env)
        record_path = Path(args.group_dir) / "submission.json"
        require(not record_path.exists(), "group already has a submission record; reconcile before retrying")
        if args.dry_run:
            cmd.append("--dry-run")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if not args.dry_run:
            write_json_atomic(record_path, {"created_at": time.time(), "command": cmd, "returncode": result.returncode,
                                            "stdout": result.stdout, "stderr": result.stderr,
                                            "status": "submitted" if result.returncode == 0 else "submit_failed"})
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="", file=sys.stderr)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
