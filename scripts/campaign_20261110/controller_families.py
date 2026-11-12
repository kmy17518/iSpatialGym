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
from scripts.campaign_20261110.replay_gate_quality import assess as assess_replay_gate


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


def _allocation_signature(root):
    # Resolve the shared directory once. Resolving every ordinary filename walks
    # the same NFS ancestors thousands of times during each reservation check.
    # Keep symlink destinations and fresh inode/timestamp evidence in the key.
    directory = root / "manifests"
    if not directory.exists():
        return ()
    resolved = directory.resolve(strict=True)
    rows = []
    with os.scandir(directory) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            if not entry.name.endswith("allocations.tsv"):
                continue
            path = directory / entry.name
            target = path.resolve(strict=True) if entry.is_symlink() else resolved / entry.name
            stat = target.stat() if entry.is_symlink() else entry.stat()
            rows.append((str(path), str(target), stat.st_dev, stat.st_ino,
                         stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(rows)


def reserve_indices(root, token, count):
    """Reserve a disjoint interval; parse allocation ledgers outside the lock."""
    require(isinstance(count, int) and count > 0, "reservation count must be positive")
    path = root / "manifests/family_index_reservations.json"
    for _ in range(8):
        # An idempotent retry needs no allocation scan. The actual entry and
        # count are still checked under the common reservation lock.
        peek = read(path, {"next": 50000000, "reservations": {}})
        signature, historical = None, []
        if token not in peek["reservations"]:
            signature = _allocation_signature(root)
            historical = [(int(row["start"]), int(row["start"]) + int(row["n"]))
                          for row in allocations(root)]
            if signature != _allocation_signature(root):
                continue
        with atomic_directory_lock(path.with_suffix(".lockdir")):
            state = read(path, {"next": 50000000, "reservations": {}})
            if token in state["reservations"]:
                prior = state["reservations"][token]
                require(not prior.get('quarantined'), 'index reservation quarantined; author a fresh token and preserve original evidence')
                require(prior["n"] == count, "index reservation changed")
                os.chmod(path, path.stat().st_mode | 0o040)
                return prior["start"]
            # New/changed ledgers invalidate the preview; release before
            # re-reading any bytes. Reservations are never committed from it.
            if signature is None or signature != _allocation_signature(root):
                continue
            intervals = historical + [(int(row["start"]), int(row["start"]) + int(row["n"]))
                                      for row in state["reservations"].values()]
            start = state["next"]
            while any(start < hi and lo < start + count for lo, hi in intervals):
                start += 100
            state["reservations"][token] = {"start": start, "n": count, "created_at": time.time()}
            state["next"] = start + max(100, ((count + 99) // 100) * 100)
            write_json_atomic(path, state)
            os.chmod(path, path.stat().st_mode | 0o040)
            return start
    raise FileExistsError("allocation ledgers kept changing during index reservation preparation")


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
            from scripts.campaign_20261110.current_eligibility import install_quota_view
            install_quota_view(root,db)
            rows = db.execute("SELECT split,bucket,COUNT(*) FROM current_members WHERE class_id=? GROUP BY split,bucket", (class_id,)).fetchall()
    return Counter({(split, bucket): count for split, bucket, count in rows})


def next_bucket(counts, quotas, enabled=None):
    enabled=list(range(len(quotas))) if enabled is None else enabled
    require(isinstance(enabled,list) and len(set(enabled))==len(enabled)
            and all(isinstance(i,int) and not isinstance(i,bool) and 0<=i<len(quotas) for i in enabled),
            'invalid enabled evaluation buckets')
    used = sum(n for (split, _), n in counts.items() if split == "eval_standard")
    # An old empty bucket allowlist is descriptive under the new owner policy.
    # Physical holds remain explicit in allow_new_cycles and failed readiness.
    return None if used >= sum(quotas) else (enabled[0] if enabled else 0)


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
        if not assess_replay_gate(gate, gate_file)['passed']:
            continue
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
        if not assess_replay_gate(gate, gate_path(root,row))['passed']:
            continue
        record=copy.deepcopy(anchor)
        record.update(status='waiting',reused_from_cycle=cycle['name'],
            reuse_gate={'path':str(gate_path(root,row)),'sha256':file_digest(gate_path(root,row))})
        return record
    return None


def adopt_pending_replacement(stage, root):
    """Follow the dispatcher's audited never-started migration, without submitting."""
    if not stage.get('dispatch_receipt'):
        return None
    receipt=read(stage['dispatch_receipt'],{})
    migration=receipt.get('migration')
    if not migration:
        return 'pending' if receipt.get('migration_pending') else None
    require(migration.get('never_started') is True,'replacement lacks never-started proof')
    evidence=migration['proof'];path=Path(evidence['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path)==evidence['sha256'],'migration evidence changed')
    proof=read(path);old=stage.get('allocation') or receipt.get('allocation')
    require(old and proof.get('schema')=='isg-never-started-migration-v1' and proof.get('never_started') is True,
            'invalid migration proof schema')
    require(proof['original_allocation']==old,'migration allocation differs from owned stage')
    accounting=proof['after']['accounting']
    require(accounting['state']=='CANCELLED' and accounting['start'] in ('Unknown','None','N/A','')
            and proof['final_execution_proof']['nonempty_execution_artifacts']==[], 'migrated source may have executed')
    replacement=migration['replacement'];directory=Path(replacement['directory']).resolve()
    original=Path(stage['directory']).resolve()
    require(directory!=original and directory.is_relative_to(root/'groups') and original.is_relative_to(root/'groups'),
            'replacement must use a fresh owned group directory')
    require(file_digest(directory/'campaign_config.json')==file_digest(original/'campaign_config.json'),
            'migration changed exact collection configuration')
    intent=Path(replacement['dispatch_intent']).resolve()
    require(intent.is_relative_to(root/'dispatch/ready'),'replacement intent outside campaign')
    payload=read(intent);identity=payload['identity']
    require(payload['command']==replacement['command'] and digest(payload['command'])==payload['command_sha256'],
            'replacement command differs from immutable intent')
    require(identity['start']==replacement['index'] and identity['n']==replacement['count']==int(old['n'])
            and identity['code']==old['code'] and identity['tag']==old['tag'] and identity['key']==old['key'],
            'replacement changes owned job identity')
    receipt_path=Path(replacement['dispatch_receipt']).resolve()
    require(receipt_path==root/'dispatch/receipts'/intent.name,'replacement receipt does not match intent')
    if 'group_id' in stage:
        plan=read(directory/'goal_plan.json')
        require(plan['group_id']==replacement['group_id']==directory.name,'replacement group plan identity differs')
    history=stage.setdefault('migration_history',[])
    history.append({'prior_stage':{k:copy.deepcopy(v) for k,v in stage.items() if k!='migration_history'},
        'proof':evidence,'adopted_at':time.time()})
    stage.update({key:replacement[key] for key in ('directory','index','count','dispatch_intent','dispatch_receipt','command')})
    if 'group_id' in stage:stage['group_id']=replacement['group_id']
    stage.pop('allocation',None)
    stage.update(status='external_submission',dry_run_passed=True)
    return 'adopted'


def reusable_group_gate(record, anchor, cfg, root, *, job_ended):
    """A terminal group may reuse its source's exact key after the full member audit."""
    from scripts.campaign_20261110.acceptance import audit_job
    if not job_ended:
        return None
    row = record["allocation"]
    current_gate = read(gate_path(root, row), {})
    # An observed failed current replay is a diagnosis requirement, not an
    # excuse to select a different successful replay.
    if current_gate.get("success") is False or (current_gate.get("success") is True
            and not assess_replay_gate(current_gate, gate_path(root,row))['passed']):
        return None
    pinned = record.get("release_gate")
    prior_review = read(Path(record["directory"]) / "release_review.json", {})
    if not pinned and prior_review:
        require(prior_review.get("group_id") == record["group_id"] and prior_review.get("source_job") == row["job_dir"],
                "existing family review belongs to another group/job")
        pinned = prior_review["gate"]
    if pinned:
        candidate = Path(pinned["path"])
        require(file_digest(candidate) == pinned["sha256"], "pinned family release gate changed")
    else:
        if current_gate.get("success") is True or not anchor or not anchor.get("allocation"):
            return None
        candidate = gate_path(root, anchor["allocation"])
    gate = read(candidate, {})
    if gate.get("success") is not True or gate.get("replay_mode") != "physics_action_open_loop":
        return None
    if not assess_replay_gate(gate, candidate)['passed']:
        return None
    source = read(gate["instance"])["ispatialgym"].get("campaign", {})
    config = read(Path(record["directory"]) / "campaign_config.json")
    if (source.get("validation_key") != config["validation_key"] or source.get("campaign_id") != config["campaign_id"]
            or source.get("split") != config["split"]):
        return None
    review = audit_job(row["job_dir"], candidate, root=root, started_at=cfg["started_at"],
                       group_closed=True, group_id=record["group_id"])
    require(review["validation_key"] == config["validation_key"], "family source gate differs from actual group key")
    if prior_review:
        require(digest(review) == digest(prior_review), "existing family review changed during idempotent adoption")
    record["release_gate"] = {"path": str(candidate), "sha256": file_digest(candidate),
                              "reuse_reason": "terminal_group_same_exact_validation_key"}
    log = Path(row["job_dir"]) / "sample.log"
    if log.exists():
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 2_000_000))
            tail = stream.read().decode(errors="replace")
        failures = [line for line in tail.splitlines() if "illegal memory access" in line or "Traceback (most recent call last)" in line]
        if failures:
            record["terminal_runtime_failure"] = {"path": str(log), "sha256": file_digest(log),
                "observed_errors": failures[-3:], "successful_members_audited_separately": True}
    return candidate


def claim(record, spec, cfg, root):
    from scripts.campaign_20261110.acceptance import audit_job
    from scripts.campaign_20261110.goal_groups import load_template_contract
    from scripts.campaign_20261110.group_lane import claim_group
    from scripts.campaign_20261110.reservation_policy import active_policy
    row = record["allocation"]
    selected = record.get("release_gate")
    gate = Path(selected["path"]) if selected else gate_path(root, row)
    if selected:
        require(file_digest(gate) == selected["sha256"], "pinned family release gate changed before claim")
    review = audit_job(row["job_dir"], gate, root=root, started_at=cfg["started_at"],
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
    bucket = next_bucket(counts, spec["eval_quotas"], spec.get('enabled_buckets'))
    cycle = state.get("current")
    if cycle is None:
        if spec.get("allow_new_cycles", True) is False:
            state["status"] = "needs_family_diagnosis"
            return 0
        if bucket is None:
            state["status"] = ('evaluation_quota_complete' if next_bucket(counts,spec['eval_quotas']) is None
                               else 'validated_buckets_complete_other_buckets_need_diagnosis')
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
    migration=adopt_pending_replacement(stage,root)
    if migration:
        state['status']='waiting_for_pending_migration' if migration=='pending' else 'migration_replacement_adopted'
        save()
        if migration=='pending':return 0
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
    observed_gate_path = gate_path(root, row)
    gate = read(observed_gate_path)
    if group and ended:
        reused = reusable_group_gate(group, cycle.get("anchor"), cfg, root, job_ended=ended)
        if reused is not None:
            gate = read(reused)
            observed_gate_path = reused
    quality = assess_replay_gate(gate, observed_gate_path) if gate and gate.get('success') is True else None
    if quality is not None and not quality['passed']:
        stage['replay_quality_failure'] = {**quality, 'gate': str(observed_gate_path),
                                         'gate_sha256': file_digest(observed_gate_path)}
    if gate and gate.get("success") is True and quality['passed']:
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
        if (gate and gate.get("success") is False) or (quality is not None and not quality['passed']) or watcher.get("status") in ("demo_missing", "demo_failed", "replay_failed", "replay_missing", "submit_failed"):
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
        next_bucket({},spec['eval_quotas'],spec.get('enabled_buckets'))
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
                        name = ("family_auto_" + spec["class_id"].replace("-", "_") + "_"
                                + str(len(family.get('history',[]))+1).zfill(4))
                        bucket = next_bucket(accepted_counts(root, spec["class_id"]), spec["eval_quotas"], spec.get('enabled_buckets'))
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
