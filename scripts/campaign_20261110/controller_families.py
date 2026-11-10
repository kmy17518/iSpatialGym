#!/usr/bin/env python3
"""Continue reviewed family seeds through fresh anchors, groups, replay and claims.

The replay watcher owns replay submissions. This controller only adopts its real
gate files; the acceptance and quota helpers re-read all immutable evidence.
An ambiguous submission is stopped for reconciliation, never submitted twice.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.campaign_20261110.contracts import digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.storage import atomic_directory_lock


TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def allocations(root):
    rows = []
    for path in sorted((root / "manifests").glob("*allocations.tsv")):
        with path.open() as stream:
            rows.extend({**row, "ledger": str(path)} for row in csv.DictReader(stream, delimiter="\t"))
    return rows


def reconcile(rows, *, key, index, count, snapshot, tag):
    matches = [row for row in rows if row.get("key") == key and int(row["start"]) == index]
    require(len(matches) <= 1, "ambiguous fresh allocation")
    if not matches:
        return None
    row = matches[0]
    require(int(row["n"]) == count and row["code"] == snapshot and row["tag"] == tag,
            "fresh allocation has different runtime/count/tag")
    return row


def reserve_indices(root, token, count):
    """One campaign-wide disjoint namespace, checked against all existing ledgers."""
    require(isinstance(count, int) and count > 0, "reservation count must be positive")
    path = root / "manifests/family_index_reservations.json"
    with atomic_directory_lock(path.with_suffix(".lockdir")):
        state = read(path, {"next": 50000000, "reservations": {}})
        if token in state["reservations"]:
            prior = state["reservations"][token]
            require(prior["n"] == count, "index reservation changed")
            os.chmod(path, path.stat().st_mode | 0o040)
            return prior["start"]
        start = state["next"]
        intervals = [(int(row["start"]), int(row["start"]) + int(row["n"]))
                     for row in [*allocations(root), *state["reservations"].values()]]
        while any(start < hi and lo < start + count for lo, hi in intervals):
            start += 100
        state["reservations"][token] = {"start": start, "n": count, "created_at": time.time()}
        state["next"] = start + max(100, ((count + 99) // 100) * 100)
        write_json_atomic(path, state)
        os.chmod(path, path.stat().st_mode | 0o040)
        return start


def accepted_counts(root, class_id):
    source = root / "dataset/allocations.sqlite"
    if not source.exists():
        return Counter()
    # The allocator atomically replaces this database. Open its copied bytes
    # locally, never SQLite on NFS, and never count a not-yet-committed export.
    with tempfile.TemporaryDirectory(prefix="family-counts-", dir="/tmp") as temporary:
        local = Path(temporary) / "state.sqlite"
        shutil.copyfile(source, local)
        with sqlite3.connect(f"file:{local}?mode=ro", uri=True) as db:
            rows = db.execute("SELECT split,bucket,COUNT(*) FROM members WHERE class_id=? GROUP BY split,bucket", (class_id,)).fetchall()
    return Counter({(split, bucket): count for split, bucket, count in rows})


def next_bucket(counts, quotas):
    deficits = [max(0, target - counts.get(("eval_standard", index), 0)) for index, target in enumerate(quotas)]
    if not any(deficits):
        return None
    return max(range(len(deficits)), key=lambda index: (deficits[index], -index))


def slurm_status():
    result = subprocess.run(["squeue", "-u", "minyeong,shiyuc", "-h", "-o", "%i|%T|%u|%P|%b"],
                            text=True, capture_output=True, timeout=40, check=True)
    return {parts[0]: parts for line in result.stdout.splitlines() if len(parts := line.split("|")) == 5}


def terminal(jobid):
    result = subprocess.run(["sacct", "-j", jobid, "-n", "-P", "--format=JobIDRaw,State"],
                            text=True, capture_output=True, timeout=40, check=True)
    states = [line.split("|")[1].split()[0].rstrip("+") for line in result.stdout.splitlines()
              if line.split("|")[0] == jobid]
    return bool(states) and states[0] in TERMINAL


def submission_capacity(queue, maximum):
    own = [row for row in queue.values() if row[2] == "minyeong"]
    running = [row for row in own if row[1] == "RUNNING"]
    gpu = sum(int(row[4].rsplit(":", 1)[-1]) for row in running
              if row[3] == "svl" and "gpu:" in row[4])
    return min(maximum, max(0, 48 - gpu - 4), max(0, 56 - len(running) - 4), max(0, 100 - len(own) - 4))


def gate_path(root, row):
    return root / "validation/open_loop" / row["key"] / row["jobid"] / "gate.json"


def copy_seed(seed, directory):
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("campaign_config.json", "source.json", "empty_eval_index.json"):
        write_json_atomic(directory / name, read(Path(seed) / name))


def anchor_command(spec, root, directory, index, helper_snapshot):
    cmd = [sys.executable, str(Path(helper_snapshot) / "scripts/campaign_20261110/infra_submit_validation.py"),
           "--root", str(root), "--group", str(directory), "--snapshot", spec["anchor_snapshot"],
           "--key", spec["class_id"], "--tag", "family_anchor", "--v2-root", spec["v2_root"], "--index", str(index),
           "--kcollect", str(spec.get("kcollect", 2))]
    if spec.get("template_customs"):
        cmd.append("--template-customs")
    return cmd


def run_logged(cmd, path):
    result = subprocess.run(cmd, text=True, capture_output=True, timeout=180)
    path.write_text(result.stdout + result.stderr)
    require(result.returncode == 0, "command failed; inspect " + str(path))


def perform_submit(record, cmd, save, rows, spec, root, tag, snapshot, count, *, dispatch_spool=False):
    row = reconcile(rows, key=spec["class_id"], index=record["index"], count=count, snapshot=snapshot, tag=tag)
    if row:
        record.update(allocation=row, status="waiting")
        save()
        return
    require(record["status"] != "submitting", "submission outcome ambiguous; ledger reconciliation required")
    if dispatch_spool:
        receipt = root / 'dispatch/receipts' / (Path(record['directory']).name + '.json')
        # Persist the intent before any external submission. All dispatchers use
        # the same token lock and immutable command identity.
        record.update(command=cmd, count=count, dispatch_receipt=str(receipt), status='external_submission')
        record['dispatch_intent'] = str(root / 'dispatch/ready' / receipt.name)
        save()
        if tag == 'family_anchor':
            run_logged(cmd + ['--dispatch-spool'], Path(record['directory']) / 'controller_submit.log')
        else:
            from scripts.campaign_20261110.controller_spool import enqueue_record
            record['dry_run_passed'] = True  # exact command dry-run completed by advance
            enqueue_record(root, record, save, priority=25)
        return
    record.update(status="submitting", command_sha256=digest(cmd), submission_started_at=time.time())
    save()  # An interrupted sbatch never permits an automatic second submission.
    run_logged(cmd, Path(record["directory"]) / "controller_submit.log")
    row = reconcile(allocations(root), key=spec["class_id"], index=record["index"], count=count,
                    snapshot=snapshot, tag=tag)
    require(row is not None, "submit returned without a durable matching allocation")
    record.update(allocation=row, status="waiting")
    save()


def restore_recovered_cycle(spec, state, root):
    """Reconsider a failed attempt only after its real physics gate recovered."""
    if state.get('current') or spec.get('allow_new_cycles', True) is False:
        return False
    for cycle in reversed(state.get('history', [])):
        stage = cycle.get('group') or cycle.get('anchor') or {}
        if stage.get('status') != 'failed' or cycle.get('late_recovery'):
            continue
        row = stage.get('allocation') or {}
        if not row or row.get('key') != spec['class_id']:
            continue
        gate_file = gate_path(root, row)
        gate = read(gate_file, {})
        if gate.get('success') is not True or gate.get('replay_mode') != 'physics_action_open_loop':
            continue
        job = Path(row['job_dir']).resolve()
        require(job.is_relative_to(root / 'out') and Path(gate.get('source_job', '')).resolve() == job,
                'recovered gate source differs from the owned allocation')
        restored = copy.deepcopy(cycle)
        restored_stage = restored.get('group') or restored['anchor']
        restored_stage['status'] = 'waiting'
        restored['recovery_evidence'] = {'gate_path': str(gate_file), 'gate_sha256': file_digest(gate_file), 'at': time.time()}
        cycle['late_recovery'] = restored['recovery_evidence']
        state.update(current=restored, status='recovered_replay', consecutive_failures=0)
        return True
    return False


def prepare_from_anchor(spec, anchor_record, group_id, output, magnitude_range, root, started_at):
    from scripts.campaign_20261110.goal_groups import load_template_contract, prepare_group, successful_anchor
    row = anchor_record["allocation"]
    anchor = successful_anchor(row["job_dir"], campaign_root=root, ledger=row["ledger"], campaign_started_at=started_at)
    seed_config = read(Path(spec["seed"]) / "campaign_config.json")
    template_path = output.parent / (group_id + "_source_contract.json")
    write_json_atomic(template_path, {"templates": [seed_config["template"]]})
    template = load_template_contract(template_path, spec["class_id"])
    key = seed_config["validation_key"]
    prepare_group(anchor=anchor, template=template, campaign_id="campaign_20261110", group_id=group_id,
                  output=output, code_sha256=file_digest(Path(spec["group_snapshot"]) / ".snapshot_commit"),
                  physics_sha256=key["physics_sha256"], data_sha256=key["data_sha256"], magnitude_range=magnitude_range,
                  budget_seconds=spec.get("group_budget_seconds", 900), max_members=3,
                  reuse_anchor_goal=spec.get("pair_goal_mode", "anchor") == "anchor")
    config = read(output / "campaign_config.json")
    write_json_atomic(output / "contracts.json", {"templates": [{**config["template"], "fingerprint": config["validation_key"]["template_sha256"]}]})


def reusable_anchor_record(spec, state, root):
    """Reuse only a fresh physics-passed anchor, never its collected instance.

    The owner's goal-reuse policy permits repeated world goals across groups;
    prepare_from_anchor still rechecks every source artifact and authors fresh
    starts, allocation indices, and a separate group validation contract.
    """
    if not spec.get('reuse_successful_anchor', False):
        return None
    for cycle in reversed(state.get('history', [])):
        anchor=cycle.get('anchor') or {};row=anchor.get('allocation') or {}
        if row.get('key')!=spec['class_id'] or row.get('code')!=spec['anchor_snapshot']:
            continue
        gate=read(gate_path(root,row),{})
        if gate.get('success') is not True or gate.get('replay_mode')!='physics_action_open_loop':
            continue
        job=Path(row['job_dir']).resolve()
        require(job.is_relative_to(root/'out') and Path(gate.get('source_job','')).resolve()==job,
                'reusable anchor gate names another job')
        record=copy.deepcopy(anchor)
        record.update(status='waiting',reused_from_cycle=cycle['name'],
            reuse_gate={'path':str(gate_path(root,row)),'sha256':file_digest(gate_path(root,row))})
        return record
    return None


def claim(record, spec, cfg, root):
    from scripts.campaign_20261110.acceptance import audit_job
    from scripts.campaign_20261110.goal_groups import load_template_contract
    from scripts.campaign_20261110.group_lane import claim_group
    from scripts.campaign_20261110.reservation_policy import active_policy
    row = record["allocation"]
    review = audit_job(row["job_dir"], gate_path(root, row), root=root, started_at=cfg["started_at"],
                       group_closed=True, group_id=record["group_id"])
    directory = Path(record["directory"])
    write_json_atomic(directory / "release_review.json", review)
    policy = active_policy(root)
    require(policy is not None, "family controller requires active reservations")
    result = claim_group(review, load_template_contract(directory / "contracts.json", spec["class_id"]),
                         policy[1], reservation_hashes=policy[2], root=root,
                         started_at=cfg["started_at"], state_path=root / "dataset/allocations.sqlite")
    record.update(status="claimed", accepted=result["accepted"], closed_at=time.time())


def advance(spec, state, cfg, root, queue, allowance, save):
    from scripts.campaign_20261110.goal_groups import submission_command
    from scripts.campaign_20261110.controller_held import job_terminal
    counts = accepted_counts(root, spec["class_id"])
    state["counts"] = {split + ":" + str(bucket): n for (split, bucket), n in counts.items()}
    bucket = next_bucket(counts, spec["eval_quotas"])
    cycle = state.get("current")
    if cycle is None:
        if spec.get("allow_new_cycles", True) is False:
            state["status"] = "needs_family_diagnosis"
            return 0
        if bucket is None:
            state["status"] = "evaluation_quota_complete"
            return 0
        if state.get("consecutive_failures", 0) >= cfg.get("max_consecutive_failures", 3):
            state["status"] = "needs_diagnosis"
            return 0
        if allowance <= 0:
            return 0
        number = len(state.get("history", [])) + 1
        name = "family_auto_" + spec["class_id"].replace("-", "_") + "_" + str(number).zfill(4)
        anchor=reusable_anchor_record(spec,state,root)
        if anchor is None:
            directory = root / "groups" / (name + "_anchor")
            require(not directory.exists(), "unreconciled anchor directory already exists")
            copy_seed(spec["seed"], directory)
            anchor={"directory":str(directory),"index":reserve_indices(root,name+":anchor",1),"status":"ready"}
        cycle = {"name": name, "bucket": bucket, "created_at": time.time(), "anchor": anchor}
        state.update(current=cycle, status="anchor")
        save()
    group = cycle.get("group")
    stage = group or cycle["anchor"]
    if stage["status"] in ("ready", "submitting"):
        if allowance <= 0:
            return 0
        if group:
            cmd = submission_command(group["directory"], snapshot=spec["group_snapshot"], campaign_root=root,
                                     v2_root=spec["v2_root"], base_index=group["index"])
            cmd[cmd.index("--kcollect") + 1] = str(spec.get("kcollect", 2))
            run_logged(cmd + ["--dry-run"], Path(group["directory"]) / "controller_dry.txt")
            perform_submit(stage, cmd, save, allocations(root), spec, root, "goal_groups", spec["group_snapshot"], 3,
                           dispatch_spool=cfg.get('dispatch_spool', False))
        else:
            cmd = anchor_command(spec, root, stage["directory"], stage["index"], cfg["helper_snapshot"])
            perform_submit(stage, cmd, save, allocations(root), spec, root, "family_anchor", spec["anchor_snapshot"], 1,
                           dispatch_spool=cfg.get('dispatch_spool', False))
        return 1
    if stage['status'] == 'external_submission':
        receipt = read(stage['dispatch_receipt'], {})
        require(receipt.get('status') != 'blocked', 'dispatch blocked: ' + str(receipt.get('error')))
        if not receipt and not Path(stage['dispatch_intent']).exists():
            # A crash before the ready-file publication is safe to reconcile:
            # this path only publishes an immutable intent, never runs sbatch.
            if group:
                from scripts.campaign_20261110.controller_spool import enqueue_record
                stage['dry_run_passed'] = True
                enqueue_record(root, stage, save, priority=25)
            else:
                run_logged(stage['command'] + ['--dispatch-spool'], Path(stage['directory']) / 'controller_submit.log')
        row = reconcile(allocations(root), key=spec['class_id'], index=stage['index'], count=3 if group else 1,
                        snapshot=spec['group_snapshot'] if group else spec['anchor_snapshot'],
                        tag='goal_groups' if group else 'family_anchor')
        if row is None:
            return 0
        stage.update(allocation=row, status='waiting')
        save()
    row = stage["allocation"]
    ended = job_terminal(row, root, queue)
    gate = read(gate_path(root, row))
    if gate and gate.get("success") is True:
        if group:
            if not ended:
                return 0  # Final membership cannot be allocated while collectors are writing.
            claim(group, spec, cfg, root)
            state.setdefault("history", []).append(cycle)
            state.pop("current")
            state["consecutive_failures"] = 0
            state["status"] = "ready_for_next_goal"
        elif ended and allowance > 0:
            directory = root / "groups" / cycle["name"]
            prepare_from_anchor(spec, stage, cycle["name"], directory, spec["magnitude_ranges"][cycle["bucket"]], root, cfg["started_at"])
            cycle["group"] = {"group_id": cycle["name"], "directory": str(directory), "status": "ready",
                              "index": reserve_indices(root, cycle["name"] + ":group", 3)}
            state["status"] = "group"
        save()
        return 0
    if ended:
        watcher = read(root / "validation/controller.json", {}).get("gates", {}).get(row["jobid"], {})
        if (gate and gate.get("success") is False) or watcher.get("status") in ("demo_missing", "demo_failed", "replay_failed", "replay_missing", "submit_failed"):
            stage.update(status="failed", watcher=watcher.get("status"), closed_at=time.time())
            state.setdefault("history", []).append(cycle)
            state.pop("current")
            state["consecutive_failures"] = state.get("consecutive_failures", 0) + 1
            save()
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--once", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--interval", type=float, default=45)
    p.add_argument("--config-transition-from", type=Path)
    args = p.parse_args()
    cfg = read(args.config)
    root = Path(cfg["root"]).resolve()
    require(args.config.resolve().is_relative_to(root), "controller manifest outside campaign")
    require(len({spec["class_id"] for spec in cfg["families"]}) == len(cfg["families"]), "duplicate family owners")
    for spec in cfg["families"]:
        seed = Path(spec["seed"]).resolve()
        require(seed.is_relative_to(root) and not (seed / "release_block.json").exists(), "blocked or external seed")
        config = read(seed / "campaign_config.json")
        require(file_digest(seed / "campaign_config.json") == spec["seed_config_sha256"], "seed configuration changed")
        require(config["purpose"] == "validation" and config["template"]["class_id"] == spec["class_id"], "seed class differs")
        require(file_digest(Path(spec["anchor_snapshot"]) / ".snapshot_commit") == config["validation_key"]["code_sha256"], "seed code identity mismatch")
        require(len(spec["eval_quotas"]) == len(spec["magnitude_ranges"]), "bucket/range counts differ")
    if not args.execute:
        print(json.dumps({"preflight_passed": True, "families": [s["class_id"] for s in cfg["families"]], "submissions": 0}))
        return
    path = root / "controllers/families/state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_directory_lock(path.with_suffix(".lockdir")):
        state = read(path, {"schema": 1, "config_sha256": file_digest(args.config), "families": {}})
        if state["config_sha256"] != file_digest(args.config) and args.config_transition_from:
            previous = read(args.config_transition_from)
            require(file_digest(args.config_transition_from) == state["config_sha256"], "transition does not name active config")
            require(previous["root"] == cfg["root"] and previous["started_at"] == cfg["started_at"], "transition changes campaign identity")
            require({s["class_id"] for s in previous["families"]} <= {s["class_id"] for s in cfg["families"]}, "transition drops an owned family")
            state.setdefault("config_transitions", []).append({"previous": str(args.config_transition_from),
                "previous_sha256": state["config_sha256"], "next": str(args.config), "next_sha256": file_digest(args.config), "at": time.time()})
            state["config_sha256"] = file_digest(args.config)
        require(state["config_sha256"] == file_digest(args.config), "controller config changed; version and reconcile explicitly")
        def interrupted(_signum, _frame):
            raise KeyboardInterrupt("controller stopped; durable allocations preserved")
        signal.signal(signal.SIGTERM, interrupted)
        def save():
            state["updated_at_unix"] = time.time()
            write_json_atomic(path, state)
        while True:
            try:
                queue = slurm_status()
                active = sum(bool(row.get("current")) for row in state["families"].values())
                allowance = (cfg.get('max_submissions_per_tick', 2) if cfg.get('dispatch_spool') else
                             submission_capacity(queue, cfg.get("max_submissions_per_tick", 2)))
                for spec in cfg["families"]:
                    family = state["families"].setdefault(spec["class_id"], {"history": []})
                    if restore_recovered_cycle(spec, family, root):
                        active += 1
                        save()
                    if not family.get("current") and spec.get("adopt_anchor") and not family.get("anchor_adopted"):
                        entry = spec["adopt_anchor"]
                        rows = [row for row in allocations(root) if row["jobid"] == str(entry["jobid"])]
                        require(len(rows) == 1 and rows[0]["key"] == spec["class_id"], "adopted anchor allocation ambiguous")
                        name = "family_auto_" + spec["class_id"].replace("-", "_") + "_0001"
                        bucket = next_bucket(accepted_counts(root, spec["class_id"]), spec["eval_quotas"])
                        require(bucket is not None, "adopted anchor class quota already full")
                        family.update(anchor_adopted=True, current={"name": name, "bucket": bucket, "created_at": time.time(), "anchor": {
                            "directory": spec["seed"], "status": "waiting", "allocation": rows[0]}})
                        active += 1
                        save()
                    if not family.get("current") and spec.get("adopt_group") and not family.get("adopted"):
                        entry = spec["adopt_group"]
                        rows = [row for row in allocations(root) if row["jobid"] == str(entry["jobid"])]
                        require(len(rows) == 1 and rows[0]["key"] == spec["class_id"], "adopted group allocation ambiguous")
                        family.update(adopted=True, current={"name": entry["group_id"], "created_at": time.time(), "group": {
                            "group_id": entry["group_id"], "directory": str(root / "groups" / entry["group_id"]),
                            "status": "waiting", "allocation": rows[0]}})
                        active += 1
                        save()
                    slots = allowance if family.get("current") or active < cfg.get("max_active_families", 8) else 0
                    before = bool(family.get("current"))
                    try:
                        allowance -= advance(spec, family, cfg, root, queue, slots, save)
                        family.pop("error", None)
                    except Exception as exc:
                        family["error"] = {"type": type(exc).__name__, "message": str(exc), "at": time.time()}
                    active += bool(family.get("current")) - before
                    save()
                state.pop("probe_error", None)
            except Exception as exc:
                state["probe_error"] = {"type": type(exc).__name__, "message": str(exc), "at": time.time()}
            save()
            if args.once:
                break
            time.sleep(max(10, args.interval))


if __name__ == "__main__":
    main()
