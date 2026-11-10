#!/usr/bin/env python3
"""Claim reviewed camera jobs and continue bounded batches under exact replay keys."""
from __future__ import annotations

from scripts.campaign_20261110 import controller_lifecycle as lifecycle

import argparse
from collections import Counter
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.campaign_20261110.contracts import digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.controller_families import (
    allocations, perform_submit, read, reconcile, reserve_indices, run_logged, slurm_status,
    submission_capacity, terminal,
)
from scripts.campaign_20261110.storage import atomic_directory_lock


def authority(root, template):
    if template.task.endswith("obstructed"):
        return root / "dataset/camera_obstructed_quota.json"
    if template.split == "eval_category":
        return root / "dataset/camera_category_quota.json"
    return root / "dataset/camera_quota.json"


def remaining_capacity(template, accepted, caps, quotas):
    """Remaining applicable slots; held-out overflow never consumes train slots."""
    rows = list(accepted.values())
    split, class_id = template.split, template.class_id
    eligible = [r for r in rows if r["split"] == split
                and (split == "eval_category" or r.get("class_id", class_id) == class_id)]
    counts = Counter(r["bucket"] for r in eligible)
    evaluation = sum(max(0, n - counts[b]) for b, n in quotas.get(split, {}).items())
    if split not in ("train", "eval_standard"):
        return evaluation
    room = template.scene + "/" + template.room
    training = [r for r in rows if r["split"] == "train"]
    room_left = max(0, caps.get(room, 0) - sum(r["room"] == room for r in training))
    class_left = max(0, 1000 - sum(r.get("class_id", class_id) == class_id for r in training))
    return evaluation + min(room_left, class_left)


def pending_camera(root,accepted):
    result=[]
    for path in sorted((root/'controllers').glob('*/state.json')):
        config_path=path.with_name('config.json')
        if not config_path.exists():continue
        specs={row['id']:row for row in read(config_path).get('seeds',[])}
        for seed,current in read(path).get('seeds',{}).items():
            records=[row for row in current.get('production',[]) if row.get('status')!='complete']
            if not records:continue
            require(seed in specs,'pending camera seed has no immutable config')
            template=read(Path(specs[seed]['inputs'])/'campaign_config.json')['template']
            for record in records:
                job=record.get('allocation',{}).get('job_dir')
                committed=sum(row.get('source_job')==job for row in accepted.values()) if job else 0
                result.append({'task':template['task'],'class_id':template['class_id'],
                    'room':template['scene']+'/'+template['room'],'split':template['split'],
                    'count':max(0,record['count']-committed)})
    return result


def reserved_capacity(template, accepted, caps, quotas, pending):
    caps=dict(caps);same_scope=0
    room=template.scene+'/'+template.room
    for row in pending:
        if row['task']!=template.task:continue
        share_train=template.split in ('train','eval_standard') and row['split'] in ('train','eval_standard')
        share_eval=template.split==row['split']
        if (row['class_id']==template.class_id or template.split=='eval_category') and (share_train or share_eval):
            same_scope+=row['count']
        elif share_train and row['room']==room:
            caps[room]=max(0,caps.get(room,0)-row['count'])
    return max(0,remaining_capacity(template,accepted,caps,quotas)-same_scope)


def production_command(seed_command, *, directory, index, count):
    """Preserve all source runtime options, including private BDDL and obstacle flag."""
    cmd = list(seed_command)
    cmd[cmd.index("submit") + 2] = str(count)
    for flag, value in (("--base-index", index), ("--tag", "camera_auto_production"),
                        ("--sample-limit-s", max(2700, count * 180 + 1800)),
                        ("--time-s", max(5400, count * 180 + 5400))):
        cmd[cmd.index(flag) + 1] = str(value)
    replacements = {
        "ISPATIALGYM_CAMPAIGN_CONFIG": str(directory / "campaign_config.json"),
        "EVAL_INDEX": str(directory / "empty_eval_index.json"),
        "ISGF_SAMPLING_CACHE": str(directory / "sampling_cache"),
    }
    for i, value in enumerate(cmd):
        name = value.split("=", 1)[0]
        if name in replacements:
            cmd[i] = name + "=" + replacements[name]
    require("ISGF_COLLECT_EXTRA=--no-goal-override" in cmd, "camera collector must preserve the TRO goal")
    return cmd


def allocation_for(rows, jobid):
    matches = [row for row in rows if row["jobid"] == str(jobid)]
    require(len(matches) == 1, "camera job must have one durable allocation")
    return matches[0]


def review_and_claim(row, gate, template, cfg, root, directory):
    from scripts.campaign_20261110.camera_lane import audit_camera, claim_camera, room_caps
    from scripts.campaign_20261110.camera_classes import claim_camera_classes
    from scripts.campaign_20261110.reservation_policy import active_policy
    review = audit_camera(row["job_dir"], gate, root=root, started_at=cfg["started_at"], template=template)
    write_json_atomic(directory / ("release_review_" + row["jobid"] + ".json"), review)
    task = template.task
    caps = room_caps(read(cfg["room_plan"]), total=9000 if task.endswith("obstructed") else 1000, task=task)
    state_path = authority(root, template)
    if task.endswith("standard") and template.split in ("train", "eval_standard"):
        result = claim_camera(review, state_path, root=root, started_at=cfg["started_at"], training_room_caps=caps)
    else:
        policy = active_policy(root)
        require(policy is not None, "camera controller requires the active reservation policy")
        classes = [r["class_id"] for r in read(cfg["standard_contracts"])["templates"] if r["task"] == task]
        result = claim_camera_classes(review, state_path, root=root, started_at=cfg["started_at"],
                                      standard_classes=classes, training_room_caps=caps,
                                      reservations=policy[1], reservation_hashes=policy[2])
    return result, caps


def report_signature(job):
    return [(p.parent.name, p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted((Path(job) / "demo").glob("*/meta.json"))]


def sampling_infrastructure_failure(job):
    path = Path(job) / "sample.log"
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    if "fcntl.flock(" in text and "OSError: [Errno 5] Input/output error" in text:
        return {"reason": "unsupported_nfs_failure_journal_lock", "log": str(path), "sha256": file_digest(path)}
    return None


def refresh_production(record, gate, template, cfg, root, directory, *, closed):
    """An empty failed job must not be stuck forever in the release auditor."""
    row = record["allocation"]
    reports = report_signature(row["job_dir"])
    signature = digest(reports)
    if signature != record.get("report_signature"):
        successful = any(read(Path(row["job_dir"]) / "demo" / iid / "meta.json").get("success") is True
                         for iid, _, _ in reports)
        if successful:
            result, _ = review_and_claim(row, gate, template, cfg, root, directory)
            record["last_claim"] = result
        record["report_signature"] = signature
        record["has_successful_reports"] = successful
    if closed:
        record["status"] = "complete"
        failure = sampling_infrastructure_failure(row["job_dir"])
        if failure is not None:
            record["requires_new_runtime_gate"] = failure
    return record.get("requires_new_runtime_gate")


def advance(spec, state, cfg, root, rows, queue, allowance, save):
    from scripts.campaign_20261110.goal_groups import load_template_contract
    from scripts.campaign_20261110.camera_lane import room_caps
    from scripts.campaign_20261110.contracts import task_quotas
    from scripts.campaign_20261110.reservation_policy import active_policy
    from scripts.campaign_20261110.group_lane import check_reservations
    directory = Path(spec["inputs"]).resolve()
    require(directory.is_relative_to(root), "camera seed outside campaign")
    require(all(file_digest(directory / name) == spec["seed_files_sha256"][name]
                for name in ("ready_contract.json", "campaign_config.json", "submission.json")),
            "camera seed files differ from the reviewed controller config")
    template = load_template_contract(directory / "ready_contract.json", spec["class_id"])
    config = read(directory / "campaign_config.json")
    require(config["campaign_id"] == cfg["campaign_id"] and config["validation_key"]["template_sha256"] == template.fingerprint,
            "camera seed contract or campaign differs")
    policy = active_policy(root)
    require(policy is not None, "camera controller requires active reservations")
    check_reservations(template, policy[1], policy[2])
    source = allocation_for(rows, spec["validation_jobid"])
    require(source["key"] == template.class_id, "camera allocation class differs")
    gate = root / "validation/open_loop" / source["key"] / source["jobid"] / "gate.json"
    if not gate.exists():
        state["status"] = "waiting_for_replay"
        return 0
    require(read(gate).get("success") is True, "camera replay failed")
    signature = digest(report_signature(source["job_dir"]))
    if state.get("source_signature") != signature:
        result, _ = review_and_claim(source, gate, template, cfg, root, directory)
        state.update(source_signature=signature, source_accepted=result["accepted"])
        save()
    records = state.setdefault("production", [])
    for jobid in spec.get("existing_production_jobids", []):
        if not any(r.get("allocation", {}).get("jobid") == str(jobid) for r in records):
            row = allocation_for(rows, jobid)
            require(row["code"] == source["code"] and row["key"] == source["key"], "adopted production runtime differs")
            records.append({"allocation": row, "status": "waiting", "count": int(row["n"]), "adopted": True})
    for record in records:
        from scripts.campaign_20261110.controller_migration import controller_check
        controller_check(root,record,state,cfg,queue,save,family='camera')
        if record.get('migration',{}).get('never_started'):continue
        if record.get('status') in ('external_submission','submitting'):
            allocation=reconcile(rows,key=spec['class_id'],index=record['index'],count=record['count'],
                                 snapshot=source['code'],tag='camera_auto_production')
            if allocation is not None:record.update(allocation=allocation,status='waiting')
            elif record['status']=='external_submission':continue
        if record["status"] in ("ready", "submitting"):
            if state.get("runtime_failure") is not None:
                state["status"] = "runtime_revalidation_required"
                return 0
            if allowance <= 0:
                state['status']='validated_waiting_for_dispatch_capacity'
                return 0
            if record["status"] == "ready" and not record.get("dry_run_passed"):
                run_logged(record["command"] + ["--dry-run"], Path(record["directory"]) / "dryrun.txt")
                record["dry_run_passed"] = True
                save()
            if cfg.get('dispatch_spool'):
                from scripts.campaign_20261110.controller_spool import enqueue_record
                enqueue_record(root,record,save,backends=cfg.get('dispatch_backends'))
                return 1
            perform_submit(record, record["command"], save, rows, spec, root, "camera_auto_production", source["code"], record["count"])
            return 1
        row = record["allocation"]
        closed = record["status"] == "complete" or (row["jobid"] not in queue and terminal(row["jobid"]))
        failure = refresh_production(record, gate, template, cfg, root, directory, closed=closed)
        save()
        if failure is not None:
            state.update(status="runtime_revalidation_required", runtime_failure=failure)
            continue  # Existing later jobs may still have valid members to claim.
        if record["status"] != "complete":
            state["status"] = "production_running"
            continue
    if state.get("runtime_failure") is not None:
        state["status"] = "runtime_revalidation_required"
        return 0
    inflight=[record for record in records if record['status']!='complete']
    if len(inflight)>=cfg.get('maximum_inflight_per_seed',1):return 0
    caps = room_caps(read(cfg["room_plan"]), total=9000 if template.task.endswith("obstructed") else 1000, task=template.task)
    quota = next(q for q in task_quotas() if q.task == template.task)
    quotas = {s: dict(zip(("3", "5", "10"), values)) for s, values in
              (("eval_standard", quota.standard_buckets), ("eval_model", quota.model_buckets), ("eval_category", quota.category_buckets))}
    with atomic_directory_lock(root/'controllers/camera_planning.lockdir'):
        accepted = read(authority(root, template), {"accepted": {}})["accepted"]
        remaining = reserved_capacity(template, accepted, caps, quotas,pending_camera(root,accepted))
        from scripts.campaign_20261110.controller_spool import candidate_budget_used
        budget = spec["max_candidates"] - candidate_budget_used(records)
        state.update(remaining_capacity=remaining, remaining_candidate_budget=budget)
        if remaining <= 0 or budget <= 0:
            state["status"] = ("pending_work_reserves_remaining_capacity" if inflight else "quota_complete") if remaining <= 0 else "candidate_budget_exhausted"
            return 0
        if allowance <= 0:
            state['status']='validated_waiting_for_dispatch_capacity'
            return 0
        count = min(remaining + min(3, remaining), budget, spec.get("batch_size", 80))
        token = "camera_auto_" + spec["id"] + "_" + str(len(records) + 1).zfill(3)
        require(re.fullmatch(r"[A-Za-z0-9_.-]+", token), "unsafe camera seed ID")
        output = root / "production_inputs" / token
        require(not output.exists(), "unreconciled camera production directory")
        index = reserve_indices(root, token, count)
        output.mkdir()
        write_json_atomic(output / "campaign_config.json", {**config, "purpose": "production",
            "release_gate": {"path": str(gate), "sha256": file_digest(gate)}})
        write_json_atomic(output / "empty_eval_index.json", {"schema": "isg-instance-index-v1", "n": 0, "instances": []})
        command = production_command(read(directory / "submission.json")["command"], directory=output, index=index, count=count)
        require(command[command.index("--code") + 1] == source["code"], "production command changes validated code")
        record = {"directory": str(output), "index": index, "count": count, "command": command, "status": "ready"}
        if state.get('next_dispatch_backends'):record['dispatch_backends']=state.pop('next_dispatch_backends')
        records.append(record)
        save()
        write_json_atomic(output / "submission.json", {"command": command, "seed": str(directory), "validation_jobid": source["jobid"]})
    run_logged(command + ["--dry-run"], output / "dryrun.txt")
    record["dry_run_passed"] = True
    save()
    if cfg.get('dispatch_spool'):
        from scripts.campaign_20261110.controller_spool import enqueue_record
        enqueue_record(root,record,save,backends=cfg.get('dispatch_backends'))
        return 1
    perform_submit(record, command, save, rows, spec, root, "camera_auto_production", source["code"], count)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument('--max-inflight-per-seed',type=int)
    parser.add_argument('--max-new-jobs-per-pass',type=int)
    parser.add_argument('--dispatch-spool',action='store_true')
    parser.add_argument('--dispatch-backends',help='Optional comma-separated authorized routing subset.')
    parser.add_argument('--migrate-pending-after',type=float)
    args = parser.parse_args()
    cfg = read(args.config)
    original_config_hash=digest(cfg)
    if args.migrate_pending_after is not None:
        require(args.migrate_pending_after>=300,'pending migration requires at least five minutes');cfg['migrate_pending_seconds']=args.migrate_pending_after
    if args.dispatch_spool:cfg['dispatch_spool']=True
    if args.dispatch_backends:
        from scripts.campaign_20261110.infra_dispatch import BACKENDS
        cfg['dispatch_backends']=args.dispatch_backends.split(',')
        require(set(cfg['dispatch_backends'])<=set(BACKENDS),'unknown dispatch backend')
    for argument,key in ((args.max_inflight_per_seed,'maximum_inflight_per_seed'),(args.max_new_jobs_per_pass,'maximum_new_jobs_per_pass')):
        if argument is not None:
            require(1<=argument<=32,'invalid operational concurrency limit');cfg[key]=argument
    require(cfg["schema"] == "isg-camera-controller-v1", "invalid camera controller schema")
    require(len({s["id"] for s in cfg["seeds"]}) == len(cfg["seeds"]), "duplicated camera seed ID")
    require(all(isinstance(s["max_candidates"], int) and s["max_candidates"] > 0
                and isinstance(s.get("batch_size", 80), int) and s.get("batch_size", 80) > 0
                for s in cfg["seeds"]), "invalid camera candidate budgets")
    root, path = Path(cfg["root"]).resolve(), Path(args.state).resolve()
    require(path.is_relative_to(root), "controller state outside campaign")
    path.parent.mkdir(parents=True, exist_ok=True)
    while lifecycle.continue_running(cfg):
        with atomic_directory_lock(path.with_suffix(".lockdir")):
            state = read(path, {"config_sha256": original_config_hash, "seeds": {}})
            require(state["config_sha256"] == original_config_hash, "camera controller config changed")
            lifecycle.record_policy(cfg, state)
            state['operational_limits']={key:cfg.get(key,default) for key,default in (('maximum_inflight_per_seed',1),('maximum_new_jobs_per_pass',3))}
            state['operational_limits']['dispatch_spool']=bool(cfg.get('dispatch_spool'))
            state['operational_limits']['dispatch_backends']=cfg.get('dispatch_backends')
            state['operational_limits']['migrate_pending_seconds']=cfg.get('migrate_pending_seconds')
            cfg['_migration_checks_remaining']=8
            save = lambda: write_json_atomic(path, state)
            rows, queue = allocations(root), slurm_status()
            allowance = cfg.get('maximum_new_jobs_per_pass',3) if cfg.get('dispatch_spool') else submission_capacity(queue, cfg.get("maximum_new_jobs_per_pass", 3))
            for spec in cfg["seeds"]:
                record = state["seeds"].setdefault(spec["id"], {})
                try:
                    allowance -= advance(spec, record, cfg, root, rows, queue, allowance, save)
                    record.pop("error", None)
                except Exception as exc:
                    record.update(error=type(exc).__name__ + ": " + str(exc), error_at=time.time())
                save()
            state["updated_at"] = time.time()
            save()
        if args.once:
            break
        time.sleep(lifecycle.sleep_seconds(cfg))


if __name__ == "__main__":
    main()
