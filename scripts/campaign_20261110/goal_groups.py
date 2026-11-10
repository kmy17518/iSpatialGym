#!/usr/bin/env python3
"""Author and submit fresh held-object goal-group validation candidates.

An anchor is used only to choose a world goal and base. Its instance is never
copied into the group. All group starts have a new allocation in a separate
ledger. Preparation and submission do not promote candidates to final splits.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    planned_goals = ((tro["ispatialgym"].get("campaign") or {}).get("runtime") or {}).get("planned_goal_poses") or {}
    require((summary.get("goal_realised") or planned_goals) and summary.get("base_pose2d"),
            "anchor has no world goal/base proposal")
    tmeta = read_json(tro_path.parent.parent / "template_meta.json")
    return {"instance_id": iid, "allocation": allocation, "sources": sources, "tro": tro,
            "summary": summary, "template_meta": tmeta, "template_dir": str(tro_path.parent.parent),
            "planned_goal_poses": planned_goals,
            "planned_start_poses": ((tro["ispatialgym"].get("campaign") or {}).get("runtime") or {}).get("planned_start_poses") or {},
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
        snapshots.append({"path": str(path), "sha256": actual, "semantics": semantic})
    require(snapshots[0]["semantics"] == snapshots[1]["semantics"], "runtime BDDL semantics differ from ready contract")
    proof = {"source": {k: v for k, v in snapshots[0].items() if k != "semantics"},
             "runtime": {k: v for k, v in snapshots[1].items() if k != "semantics"},
             "semantic_sha256": digest(snapshots[0]["semantics"]),
             "comparison": "parsed activity/domain/variant/objects/groups/init/goal/measure; goal and measure remain separate"}
    return replace(template, scene_setup={**template.scene_setup,
        "runtime_bddl_sha256": snapshots[1]["sha256"], "runtime_bddl_equivalence": proof})


def prepare_group(*, anchor, template: Template, campaign_id: str, group_id: str, output: Path,
                  code_sha256: str, physics_sha256: str, data_sha256: str,
                  magnitude_range, budget_seconds: float = 900, max_members: int = 3):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", group_id), "group id must be a safe path component")
    pair_tasks = {f"inter_object_relation_satisfaction-{relation}_small_scale": relation for relation in PAIR_RELATIONS}
    require(template.task in ("object_scaling", "mental_rotation", "alignment-world_vertical", "disembedding") or template.task in pair_tasks,
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
    declared_runtime_id = template.scene_setup.get("runtime_template_id")
    require(declared_runtime_id is None or declared_runtime_id == runtime_id, "runtime template ID mismatch")
    if template.scene_setup.get("source_bddl_sha256"):
        template = bind_runtime_definition(template, anchor["template_dir"], runtime_metadata=tm)
    now = time.time()
    source_goal = anchor["summary"].get("goal_realised")
    if source_goal is None:
        targets = [ref.instance for ref in template.objects if ref.role == "target"]
        require(len(targets) == 1 and targets[0] in anchor["planned_goal_poses"], "anchor has no unambiguous target goal")
        pose = anchor["planned_goal_poses"][targets[0]]
        source_goal = pose if isinstance(pose, dict) else {"position": pose[0], "orientation": pose[1]}
    plan = {"schema": SCHEMA, "campaign_id": campaign_id, "group_id": group_id, "split": template.split,
            "task": template.task, "template_id": runtime_id, "contract_template_sha256": template.fingerprint,
            "scene": template.scene, "room": template.room,
            "goal_pose": {"position": source_goal["position"], "orientation": source_goal["orientation"]},
            "base_pose2d": anchor["summary"]["base_pose2d"], "magnitude_range": list(magnitude_range),
            "created_at": now, "budget_seconds": budget_seconds, "max_members": max_members,
            "source_instance_id": anchor["instance_id"], "source_instance_is_member": False,
            "anchor_sources": anchor["sources"]}
    if template.task in pair_tasks:
        roles = {role: [b["name"] for b in tm.get("bindings", []) if b.get("role") == role] for role in ("target", "partner")}
        require(all(len(names) == 1 for names in roles.values()), "ambiguous pair target/partner binding")
        plan["target_name"], plan["partner_name"] = roles["target"][0], roles["partner"][0]
        partner = anchor["planned_start_poses"].get(plan["partner_name"])
        require(partner is not None, "pair anchor lacks recorded partner pose")
        settled = anchor["summary"].get("goal_check", {}).get("settled_poses", {})
        target_scene_name = bindings[plan["target_name"]].get("scene_object_name")
        target = settled.get(target_scene_name)
        require(target is not None, "pair anchor lacks a settled target goal bound to its scene object")
        plan["release_pose"] = plan["goal_pose"]
        plan["goal_pose"] = {"position": target[0], "orientation": target[1]}
        plan["partner_pose"] = {"position": partner[0], "orientation": partner[1]}
        if pair_tasks[template.task] == "nextto":
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
        plan["schema"] = PAIR_SCHEMA
        plan.pop("magnitude_range")
        plan["anchor_base_pose2d"] = plan.pop("base_pose2d")
        validate_pair_plan(plan, template_id=runtime_id, scene=template.scene, rooms=[template.room],
                           relation=pair_tasks[template.task], activated_at=now, now=now)
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
        candidates = anchor["summary"].get("campaign_goal_space", {}).get("target_poses", [])
        require(candidates, "disembedding needs a fresh anchor with verified destination-floor goal candidates")
        plan["goal_pose"] = candidates[int(digest({"group": group_id, "anchor": anchor["instance_id"]}), 16) % len(candidates)]
        plan["schema"] = TRAY_SCHEMA
        plan["crowdedness_range"] = plan.pop("magnitude_range")
        plan["anchor_base_pose2d"] = plan.pop("base_pose2d")
        validate_tray_plan(plan, template_id=runtime_id, scene=template.scene, rooms=[template.room], activated_at=now, now=now)
    elif template.task == "alignment-world_vertical":
        plan["schema"] = SURFACE_SCHEMA
        plan["magnitude_range_deg"] = plan.pop("magnitude_range")
        plan["anchor_base_pose2d"] = plan.pop("base_pose2d")  # Robot starts vary within this goal group.
        validate_surface_plan(plan, template_id=runtime_id, scene=template.scene, rooms=[template.room], activated_at=now, now=now)
    else:
        validate_goal_plan(plan, task=template.task, template_id=runtime_id, scene=template.scene, rooms=[template.room], now=now)
    perturbations = Perturbations({obj.instance: ScaleState(obj.default_scale, obj.default_scale, "fixed") for obj in template.objects})
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
           "ISGF_COLLECT_EXTRA": "--no-goal-override", "ISGF_SAMPLING_CACHE": str(root / "cache/groups" / plan["group_id"]),
           "ISGF_ASSET_CACHE": str(root / "asset_cache"), "ISGF_JB_NFS": str(root / "jobs"),
           "ISGF_PREDEMO_ROOT": str(root), "KIT_TPL_NFS_ROOT": str(root / "kit_templates"),
           "KIT_TEX_STORE": str(root / "kit_textures"), "ISGF_EXP_PATH": str(root / "isaacsim_overlay/apps"),
           "ISPATIALGYM_FAST_BAND_SHARE": "0", "ISPATIALGYM_FAST_COLLECTOR_PREBUILD": "0"}
    protected = {"ISPATIALGYM_CAMPAIGN_GOAL_PLAN", "ISPATIALGYM_CAMPAIGN_CONFIG", "TEMPLATE", "EVAL_INDEX", "ISGF_COLLECT_EXTRA"}
    for item in extra_env:
        name, value = item.split("=", 1)
        require(name not in protected, f"cannot override group invariant {name}")
        env[name] = value
    cmd = [sys.executable, str(code / "scripts/trainfast/submit_fast.py"), "submit", key, str(n), "--flow", "fast", "--code", str(code),
           "--root", str(root), "--ledger", str(root / "manifests/goal_group_allocations.tsv"), "--base-index", str(base_index),
           "--tag", "goal_groups", "--job-prefix", "isg261110_", "--grasps", "v2only", "--v2-root", str(v2_root),
           "--alias-memory", "off", "--no-memory-aggregate", "--kcollect", "2", "--sample-limit-s", str(int(budget + 1800)),
           "--time-s", str(int(budget + 3600)), "--acct", account, "--part", partition, "--gres", gres]
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
