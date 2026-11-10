"""Validate new-runtime training seeds, then admit them to bounded group refill."""
from __future__ import annotations

from scripts.campaign_20261110 import controller_lifecycle as lifecycle

import argparse
from dataclasses import asdict, replace
import datetime
from pathlib import Path
import time

from scripts.campaign_20261110 import controller_groups as groups, controller_held as held
from scripts.campaign_20261110.acceptance import artifact, audit_job, report_passed, replay_report_path
from scripts.campaign_20261110.contracts import (
    ValidationEvidence, ValidationKey, digest, file_digest, require, require_validation,
    validate_campaign_block, write_json_atomic,
)
from scripts.campaign_20261110.controller_families import allocations, read, reconcile, reserve_indices, run_logged, slurm_status
from scripts.campaign_20261110.controller_spool import enqueue_record
from scripts.campaign_20261110.goal_groups import read_env, successful_anchor
from scripts.campaign_20261110.group_lane import check_reservations, physical_template
from scripts.campaign_20261110.reservation_policy import active_policy
from scripts.campaign_20261110.storage import atomic_directory_lock

AUTHOR_RECONCILIATION = "tray_anchor_goal_reuse_v1"
FIXED_AUTHOR_GAPS = {
    "ContractError: disembedding needs a fresh anchor with verified destination-floor goal candidates",
}


def reconcile_author_gaps(state):
    """Re-audit unchanged source bytes only for this explicit fixed author gap."""
    if state.get("author_reconciliation") == AUTHOR_RECONCILIATION:
        return
    for path, observation in state.get("checked_gates", {}).items():
        if observation.get("reason") not in FIXED_AUTHOR_GAPS:
            continue
        state.setdefault("author_retry_history", []).append({"gate": path, "previous_observation": dict(observation),
            "helper_revision": AUTHOR_RECONCILIATION, "helper_sha256": file_digest(__file__), "observed_at": time.time()})
        observation.pop("sha256", None)
        observation["retry_author_revision"] = AUTHOR_RECONCILIATION
    state["author_reconciliation"] = AUTHOR_RECONCILIATION


def source_proof(gate_path, *, root, started_at, campaign_id, permitted_split=None):
    """Audit one annotated anchor without treating it as accepted dataset data."""
    gate_path = artifact(gate_path, root, started_at)
    gate = read(gate_path)
    require(gate.get("success") is True and gate.get("replay_mode") == "physics_action_open_loop", "anchor has no physics replay gate")
    job = Path(gate["source_job"]).resolve()
    require(job.is_relative_to(root / "out"), "anchor outside owned output")
    tro, demo = artifact(gate["instance"], root, started_at), artifact(gate["demo"], root, started_at)
    require(tro.is_relative_to(job / "inst") and demo.is_relative_to(job / "demo"), "anchor gate source paths differ")
    require(file_digest(tro) == gate.get("tro_sha256") and file_digest(demo) == gate.get("demo_sha256"), "anchor gate source bytes changed")
    raw = read(tro)["ispatialgym"]
    block = raw.get("campaign")
    require(block is not None, "unannotated anchor cannot admit a generic training seed")
    validate_campaign_block(block, object_bindings=raw["object_bindings"])
    require(permitted_split in (None, "eval_model"), "unsupported explicit source-proof split")
    allowed_splits = ("eval_model",) if permitted_split == "eval_model" else ("train", "eval_standard")
    require(block["campaign_id"] == campaign_id and block["split"] in allowed_splits, "anchor outside permitted split; heldout anchor cannot train")
    require(groups.supports(block["task"]) or (permitted_split == "eval_model" and block["task"] in held.RANGES),
            "anchor task has no generic queue adapter")
    gid = block.get("goal_group_id")
    require(not gid or not (root / "groups" / gid / "release_block.json").exists(), "source group is release blocked")
    env = read_env(job / "job.env")
    config_path = artifact(env["ISPATIALGYM_CAMPAIGN_CONFIG"], root, started_at)
    config = read(config_path)
    template = held.template_from(config["template"])
    require(config["campaign_id"] == block["campaign_id"] and config["split"] == block["split"]
            and template.task == block["task"] and template.class_id == block["class_id"], "anchor task/class/split differs from config")
    key = ValidationKey(**block["validation_key"])
    require(config["validation_key"] == asdict(key) and template.fingerprint == key.template_sha256,
            "anchor config differs from its actual validation key")
    snapshot = Path(gate["snapshot"]).resolve()
    require(snapshot.is_relative_to(root / "code") and file_digest(snapshot / ".snapshot_commit") == key.code_sha256,
            "anchor gate used another runtime")
    iid = raw["instance_id"]
    reports = [(artifact(demo.parent / "meta.json", root, started_at), "oracle", "demo"),
        (replay_report_path(gate, gate_path, iid, root=root, started_at=started_at), "replay", "open_loop_replay")]
    evidence = []
    for path, mode, kind in reports:
        report = read(path)
        report_passed(report, mode)
        require(report["instance_id"] == iid and Path(report["provenance"]["isg_root"]).resolve() == snapshot, "anchor report identity/code differs")
        created = datetime.datetime.fromisoformat(report["started_utc"].replace("Z", "+00:00")).timestamp()
        evidence.append(ValidationEvidence(key, kind, True, True, created, str(path), file_digest(path), file_digest(demo), iid, True))
    require_validation(key, evidence, campaign_started_at=started_at)
    policy = active_policy(root)
    require(policy is not None, "anchor admission needs active reservations")
    check_reservations(template, policy[1], policy[2])
    if template.task == "disembedding":
        require(template.scene_setup.get("binding_pool"), "disembedding admission requires the immutable full-pool contract")
    return {"schema": "isg-heldout-anchor-admission-v1" if permitted_split else "isg-training-anchor-admission-v1",
        "campaign_id": campaign_id, "source_job": str(job),
        "source_gate": {"path": str(gate_path), "sha256": file_digest(gate_path)}, "instance_id": iid,
        "config": {"path": str(config_path), "sha256": file_digest(config_path)}, "template": asdict(template),
        "validation_key": asdict(key), "perturbations": config["perturbations"],
        "evidence": [asdict(e) for e in evidence], "physical_template_sha256": physical_template(template),
        "active_policy_sha256": policy[0]["policy_sha256"], "source_is_final_member": False}


def admission_id(template, snapshot, perturbations, split="train"):
    require(split in ("train", "eval_standard"), "invalid admission split")
    identity = {"task": template.task, "class_id": template.class_id, "physical": physical_template(template),
                   "definition": template.definition_sha256, "runtime_bddl": template.scene_setup.get("runtime_bddl_sha256"),
                   "binding_pool": template.scene_setup.get("binding_pool"),
                   "code": file_digest(Path(snapshot) / ".snapshot_commit"), "perturbations": perturbations}
    if split != "train":  # Preserve every existing training admission identity.
        identity["admission_split"] = split
    return digest(identity)


def task_allowed(task, cfg):
    tasks = cfg.get("admitted_tasks")
    if tasks is None:
        return True
    require(isinstance(tasks, list) and tasks and len(tasks) == len(set(tasks))
            and all(groups.supports(value) for value in tasks), "invalid admission task scope")
    return task in tasks


def scoped_template(template, cfg):
    require(task_allowed(template.task, cfg), "task outside immutable admission scope")
    classes = cfg.get("admitted_classes")
    if classes is not None:
        require(isinstance(classes, list) and classes and all(isinstance(value, str) and value for value in classes)
                and len(classes) == len(set(classes)), "invalid admission class ownership")
        require(template.class_id in classes, "class owned by another standard lane")
    split = cfg.get("admission_split", "train")
    require(split in ("train", "eval_standard"), "invalid admission split")
    if split == "eval_standard":
        require(template.split == "eval_standard", "standard admission requires a standard evaluation source")
        return template
    require(template.split in ("train", "eval_standard"), "heldout source cannot train")
    return template if template.split == "train" else replace(template, split="train", template_id=template.template_id + "__train")


def admission_decision(template, counts, cfg):
    split = cfg.get("admission_split", "train")
    decision = groups.next_work(template, counts, max_groups=1,
                               mode="eval_then_train" if split == "eval_standard" else "train_only")
    require(decision is not None and decision["phase"] == split, "class has no unreserved admission capacity")
    return decision


def append_seed(catalog, spec, campaign_id):
    catalog = Path(catalog)
    with atomic_directory_lock(catalog.with_suffix(".lockdir")):
        raw = read(catalog, {"schema": "isg-group-seed-catalog-v1", "campaign_id": campaign_id, "seeds": [], "revision": 0})
        require(raw["schema"] == "isg-group-seed-catalog-v1" and raw["campaign_id"] == campaign_id, "wrong admission catalog")
        found = [s for s in raw["seeds"] if s["id"] == spec["id"]]
        if found:
            require(found == [spec], "admitted seed ID changed")
            return
        updated = {**raw, "seeds": raw["seeds"] + [spec], "revision": raw["revision"] + 1, "previous_sha256": digest(raw)}
        archive = catalog.parent / "catalog_revisions" / (digest(updated) + ".json")
        write_json_atomic(archive, updated)
        write_json_atomic(catalog, updated)


def prepare(proof, allocation, cfg, root):
    from scripts.campaign_20261110.goal_batches import prepare_batch, submission_command
    from ispatialgym.sampling.campaign_goal_queue import schema_for_task
    source_template = held.template_from(proof["template"])
    template = scoped_template(source_template, cfg)
    runtime = Path(cfg["snapshot"]) / "ispatialgym/sampling/campaign_goal_queue.py"
    require(runtime.is_file() and schema_for_task(template.task) in runtime.read_text(),
            "target snapshot does not support this queue schema")
    require(all(proof["validation_key"][key] == cfg[key] for key in ("physics_sha256", "data_sha256")),
            "source anchor differs from pinned campaign physics/data")
    identity = admission_id(source_template, cfg["snapshot"], proof["perturbations"], cfg.get("admission_split", "train"))
    name = "group_admit_" + identity[:20]
    anchor = successful_anchor(allocation["job_dir"], campaign_root=root, ledger=allocation["ledger"], campaign_started_at=cfg["started_at"])
    if template.task.startswith("alignment-") and not template.scene_setup.get("bucket_policy"):
        from scripts.campaign_20261110.goal_groups import with_alignment_buckets
        template = with_alignment_buckets(template, anchor["template_meta"].get("template_facts", {}).get("deviation_range_applied_deg"),
            source={"path": str(Path(anchor["template_dir"]) / "template_meta.json"), "sha256": digest(anchor["template_meta"]),
                    "digest_kind": "canonical_json", "field": "template_facts.deviation_range_applied_deg"})
    counts = groups.read_counts(root, template)
    reserved = groups.reserve_pending_counts(counts, template, groups.pending(root))
    decision = admission_decision(template, reserved, cfg)
    prepared = prepare_batch(anchor=anchor, template=template, campaign_root=root, batch_id=name, count=1,
        magnitude_range=groups.ranges(template)[decision["bucket"]], seed=cfg["seed"], campaign_id=cfg["campaign_id"],
        code_sha256=file_digest(Path(cfg["snapshot"]) / ".snapshot_commit"), physics_sha256=proof["validation_key"]["physics_sha256"],
        data_sha256=proof["validation_key"]["data_sha256"], budget_seconds=cfg.get("group_budget_seconds", 900))
    index = reserve_indices(root, name, 3)
    env = read_env(Path(allocation["job_dir"]) / "job.env")
    keys = ("KEEP_SCENE", "START_PER_ATTEMPT", "ISG_CUSTOMS", "ISPATIALGYM_CAMPAIGN_CUSTOMS_SHA256", "ISPATIALGYM_STABLE_SCENE_DIR")
    extras = [k + "=" + env[k] for k in keys if env.get(k)] + ["ISPATIALGYM_MAP_OVERLAY_RELEASE=3"]
    command = submission_command(prepared["queue"], snapshot=cfg["snapshot"], campaign_root=root, v2_root=cfg["v2_root"],
        base_index=index, extra_env=extras)
    if env.get("START_PER_ATTEMPT") == "0":
        command += ["--customs", "template"]
    directory = Path(prepared["queue"]).parent
    run_logged(command + ["--dry-run"], directory / "dryrun.txt")
    record = {"directory": str(directory), "queue": prepared["queue"], "prepared": prepared, "command": command,
        "index": index, "count": 3, "validation_key": read(prepared["config"])["validation_key"], "phase": template.split,
        "decision": decision, "needs_new_gate": True, "status": "ready", "dry_run_passed": True,
        "admission_id": identity, "source_proof": proof}
    write_json_atomic(directory / "submission.json", record)
    return record


def refresh(record, cfg, root, rows, queue):
    template = held.template_from(read(record["prepared"]["config"])["template"])
    allocation = record.get("allocation") or reconcile(rows, key=record["command"][3], index=record["index"], count=record["count"],
                                                       snapshot=cfg["snapshot"], tag="goal_groups")
    if allocation is None:
        return
    record["allocation"] = allocation
    if not held.job_terminal(allocation, root, queue):
        return
    gate = root / "validation/open_loop" / allocation["key"] / allocation["jobid"] / "gate.json"
    if not gate.exists():
        reports = list((Path(allocation["job_dir"]) / "demo").glob("*/meta.json"))
        if not any(read(path).get("success") is True for path in reports):
            record.update(status="failed", reason="new_runtime_admission_has_no_successful_demo")
        return
    if read(gate).get("success") is not True:
        record.update(status="failed", reason="new_runtime_admission_physics_replay_failed",
                      failed_gate={"path": str(gate), "sha256": file_digest(gate)})
        return
    held.claim_finished_groups(record, allocation, str(gate), cfg, root, job_terminal=True)
    require(record.get("groups"), "admission lacks a durable closed goal group")
    gid = record["prepared"]["group_ids"][0]
    review = audit_job(allocation["job_dir"], gate, root=root, started_at=cfg["started_at"], group_closed=True, group_id=gid)
    require(review["validation_key"] == record["validation_key"], "new admission key differs from replay proof")
    proof_path = root / "groups" / gid / "admission_review.json"
    write_json_atomic(proof_path, review)
    spec = {"id": "admitted_" + record["admission_id"][:20], "class_id": template.class_id,
        "seed_config": record["prepared"]["config"], "seed_config_sha256": file_digest(record["prepared"]["config"]),
        "source_gate": str(gate), "source_gate_sha256": file_digest(gate), "anchor_job": allocation["job_dir"],
        "anchor_ledger": allocation["ledger"], "snapshot": cfg["snapshot"], "v2_root": cfg["v2_root"],
        "max_candidates": cfg.get("max_candidates_per_seed", 1500), "groups_per_batch": 6,
        "mode": "eval_then_train" if cfg.get("admission_split") == "eval_standard" else "train_only",
        "seed_offset": int(record["admission_id"][:8], 16), "group_budget_seconds": cfg.get("group_budget_seconds", 900),
        "existing_queues": [record["queue"]], "admission_review": {"path": str(proof_path), "sha256": file_digest(proof_path)}}
    append_seed(cfg["seed_catalog"], spec, cfg["campaign_id"])
    if record.get("reason"):
        record["recovered_failure"] = {"reason": record.pop("reason"), "failed_gate": record.get("failed_gate")}
    record.update(status="admitted", admitted_seed=spec["id"], admitted_at=time.time())


def refreshable(record, root):
    """A repaired replay must supply new successful evidence before re-audit."""
    if record["status"] == "admitted":
        return False
    if record["status"] != "failed":
        return True
    if record.get("reason") not in ("new_runtime_admission_physics_replay_failed", "new_runtime_admission_has_no_successful_demo"):
        return False
    allocation = record.get("allocation")
    if not allocation:
        return False
    gate = root / "validation/open_loop" / allocation["key"] / allocation["jobid"] / "gate.json"
    observed = observe_gate(gate)
    return (observed is not None and observed[0] != record.get("failed_gate", {}).get("sha256")
            and observed[1].get("success") is True)


def observe_gate(path):
    # The replay watcher archives a failed canonical gate before its strict
    # retry. Losing that directory entry while scanning is not new evidence.
    try:
        return file_digest(path), read(path)
    except FileNotFoundError:
        return None


def recent_gates(root):
    rows = []
    for path in (root / "validation/open_loop").glob("*/*/gate.json"):
        try:
            rows.append((path.stat().st_mtime, path))
        except FileNotFoundError:
            continue
    return [path for _, path in sorted(rows, reverse=True)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True); p.add_argument("--state", required=True); p.add_argument("--once", action="store_true")
    args = p.parse_args(); cfg = read(args.config); root = Path(cfg["root"]).resolve(); path = Path(args.state).resolve()
    require(cfg["schema"] == "isg-group-admission-controller-v1" and path.is_relative_to(root / "controllers"), "invalid admission controller")
    require(Path(cfg["snapshot"]).resolve().is_relative_to(root / "code"), "admission runtime outside owned snapshots")
    require(1 <= cfg.get("new_admissions_per_pass", 2) <= 12 and 1 <= cfg.get("maximum_pending_admissions", 12) <= 32,
            "invalid admission concurrency limits")
    while lifecycle.continue_running(cfg):
        with atomic_directory_lock(path.with_suffix(".lockdir")):
            state = read(path, {"config_sha256": digest(cfg), "records": {}, "checked_gates": {}})
            require(state["config_sha256"] == digest(cfg), "admission controller config changed")
            lifecycle.record_policy(cfg, state)
            reconcile_author_gaps(state)
            save = lambda: write_json_atomic(path, state)
            rows, queue = allocations(root), slurm_status()
            for source in cfg.get("existing_bootstraps", []):
                record = read(source); template = held.template_from(read(record["prepared"]["config"])["template"])
                scoped_template(template, cfg)
                identity = admission_id(template, cfg["snapshot"], read(record["prepared"]["config"])["perturbations"], cfg.get("admission_split", "train"))
                if identity not in state["records"]:
                    record["admission_id"] = identity
                    record["status"] = "external_submission"
                    state["records"][identity] = record
            for record in state["records"].values():
                if not refreshable(record, root):
                    continue
                try:
                    if record["status"] == "ready":
                        enqueue_record(root, record, save, priority=30)
                    refresh(record, cfg, root, rows, queue)
                    record.pop("error", None)
                except Exception as exc:
                    record["error"] = {"at": time.time(), "reason": type(exc).__name__ + ": " + str(exc)}
                save()
            pending = sum(r["status"] not in ("admitted", "failed") for r in state["records"].values())
            allowance = min(cfg.get("new_admissions_per_pass", 2), max(0, cfg.get("maximum_pending_admissions", 12) - pending))
            examined = 0
            for gate in recent_gates(root):
                if allowance <= 0 or examined >= 30:
                    break
                observed = observe_gate(gate)
                if observed is None:
                    continue
                sha, gate_data = observed
                if state["checked_gates"].get(str(gate), {}).get("sha256") == sha:
                    continue
                if gate_data.get("success") is not True:
                    continue
                examined += 1
                try:
                    proof = source_proof(gate, root=root, started_at=cfg["started_at"], campaign_id=cfg["campaign_id"])
                    template = held.template_from(proof["template"])
                    scoped_template(template, cfg)
                    identity = admission_id(template, cfg["snapshot"], proof["perturbations"], cfg.get("admission_split", "train"))
                    found = [r for r in rows if Path(r["job_dir"]).resolve() == Path(proof["source_job"])]
                    require(len(found) == 1, "anchor allocation missing or ambiguous")
                    if not held.job_terminal(found[0], root, queue):
                        continue  # Freeze source template metadata only after its job ends.
                    if identity not in state["records"]:
                        with atomic_directory_lock(root / "controllers/held_planning.lockdir"):
                            record = prepare(proof, found[0], cfg, root)
                            state["records"][identity] = record
                            save()
                        enqueue_record(root, record, save, priority=30)
                        allowance -= 1
                    state["checked_gates"][str(gate)] = {"sha256": sha, "admission_id": identity, "observed_at": time.time()}
                except Exception as exc:
                    observation = {"reason": type(exc).__name__ + ": " + str(exc), "observed_at": time.time()}
                    if not isinstance(exc, FileExistsError):
                        observation["sha256"] = sha
                    state["checked_gates"][str(gate)] = observation
                save()
            state["updated_at"] = time.time(); save()
        if args.once:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
