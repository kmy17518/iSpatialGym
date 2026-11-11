#!/usr/bin/env python3
"""Continue exact-key held-object queues through strict evaluation and soft training."""
from __future__ import annotations

from scripts.campaign_20261110 import controller_lifecycle as lifecycle

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.campaign_20261110.contracts import ObjectRef, Template, digest, file_digest, require, task_quotas, write_json_atomic
from scripts.campaign_20261110.controller_families import (
    allocations, perform_submit, read, reconcile, reserve_indices, run_logged,
    slurm_status, submission_capacity, terminal,
)
from scripts.campaign_20261110.storage import atomic_directory_lock


RANGES = {"object_scaling": ((.05, .10), (.10, .15), (.15, .30)),
          "mental_rotation": ((30., 45.), (45., 75.), (75., 120.))}


def job_terminal(allocation, root, queue):
    jobid = allocation["jobid"]
    require(jobid.isdigit(), "invalid held allocation job id")
    vast = root / "vast/state/jobs" / (jobid + ".json")
    if vast.exists():
        # Synced means the worker returned its files; acceptance still checks
        # actual successful demo/replay artifacts independently.
        return read(vast).get("status") in ("synced", "failed")
    return jobid not in queue and terminal(jobid)


def template_from(raw):
    raw = dict(raw)
    raw["objects"] = tuple(ObjectRef(**o) for o in raw["objects"])
    return Template(**raw)


def read_counts(root, template, standards, room_plan):
    from scripts.campaign_20261110.group_lane import accepted_room, training_room_policy
    counts, train_buckets, train_class, room_used, reserve = Counter(), Counter(), 0, 0, 0
    path = root / "dataset/allocations.sqlite"
    room = template.scene + "/" + template.room
    scope = template.task if template.split == "eval_category" else template.class_id
    if path.exists():
        with tempfile.TemporaryDirectory(prefix="held-counts-", dir="/tmp") as temporary:
            local = Path(temporary) / "state.sqlite"
            shutil.copyfile(path, local)
            with sqlite3.connect(f"file:{local}?mode=ro", uri=True) as db:
                from scripts.campaign_20261110.current_eligibility import install_quota_view
                install_quota_view(root,db)
                counts = Counter(dict(db.execute("SELECT bucket,COUNT(*) FROM current_members WHERE split=? AND task=? AND class_id=? GROUP BY bucket",
                                                (template.split, template.task, scope))))
                from scripts.campaign_20261110.training_eligibility import eligible_training_records
                records = eligible_training_records(db, template.task, root=root)
                train_class = sum(row["class_id"] == template.class_id for row in records)
                train_buckets = Counter(row.get("bucket") for row in records if row["class_id"] == template.class_id)
                room_used = sum(accepted_room(row) == room for row in records)
                for standard in standards:
                    if standard["task"] == template.task and standard["scene"] + "/" + standard["room"] == room:
                        n = db.execute("SELECT COUNT(*) FROM current_members WHERE split='eval_standard' AND class_id=?", (standard["class_id"],)).fetchone()[0]
                        reserve += max(0, 10 - n)
    else:
        reserve = 10 * sum(t["task"] == template.task and t["scene"] + "/" + t["room"] == room for t in standards)
    cap = training_room_policy(room_plan, template.task)["caps"].get(room, 0)
    # Soft training permits metrics outside the strict evaluation buckets.
    # Preserve their capacity use separately; mixed None/int mapping keys
    # cannot be durably serialized with canonical JSON's sorted keys.
    unbucketed = train_buckets.pop(None, 0)
    evaluation_unbucketed = counts.pop(None, 0)
    return {"evaluation": dict(counts), "train_buckets": dict(train_buckets), "class_remaining": max(0, 1000-train_class),
            "train_unbucketed": unbucketed, "evaluation_unbucketed": evaluation_unbucketed,
            "room_remaining": max(0, cap-room_used), "evaluation_sibling_reserve": reserve}


def next_work(template, counts, *, max_groups=6, training_validated=False, mode="eval_then_train"):
    require(mode in ("eval_only", "eval_then_train"), "unknown held collection mode")
    quota = next(q for q in task_quotas() if q.task == template.task)
    split = template.split
    quotas = {"eval_standard": quota.standard_buckets, "eval_model": quota.model_buckets,
              "eval_category": quota.category_buckets}.get(split, ())
    deficits = [max(0, n-counts["evaluation"].get(i, 0)) for i, n in enumerate(quotas)]
    if any(deficits):
        bucket = max(range(len(deficits)), key=lambda i: (deficits[i], -i))
        groups = min(max_groups, deficits[bucket])
        if split == "eval_standard":
            groups = min(groups, counts["class_remaining"], counts["room_remaining"])
        return {"phase": split, "bucket": bucket, "groups": groups, "needs_new_gate": False} if groups else None
    # Persistent gate: autonomous standalone refill is disabled until release.
    return None


def reserve_pending_counts(counts, template, pending):
    """Conservatively reserve unpublished slots, without treating them as accepted."""
    result={**counts,'evaluation':Counter(counts['evaluation']),'train_buckets':Counter(counts['train_buckets'])}
    for row in pending:
        if row.get("phase") == "train": continue
        if row['task']!=template.task:continue
        groups=row['groups'];phase=row['phase'];same_class=row['class_id']==template.class_id
        if phase==template.split and (same_class or phase=='eval_category'):
            result['evaluation'][row['bucket']]+=groups*(1 if phase=='eval_standard' else 3)
        if phase in ('train','eval_standard'):
            # A strict eval group can fall back to three training members when
            # another accepted group filled its bucket while it was running.
            if same_class:
                result['class_remaining']=max(0,result['class_remaining']-3*groups)
                result['train_buckets'][row['bucket']]+=3*groups
            if row['room']==template.scene+'/'+template.room:
                result['room_remaining']=max(0,result['room_remaining']-3*groups)
    return result


def pending_held(root):
    """Read atomic controller states while the shared planning lock is held."""
    result=[]
    for path in sorted((root/'controllers').glob('*/state.json')):
        for seed in read(path).get('seeds',{}).values():
            for record in seed.get('queues',[]):
                if record.get('status')=='complete':continue
                gids=[gid for gid in record['prepared']['group_ids'] if gid not in record.get('groups',{})]
                if not gids:continue
                config=read(root/'groups'/gids[0]/'campaign_config.json');template=config['template']
                if template['task'] not in RANGES:continue
                decision=record.get('decision')
                if decision is None:
                    plan=read(root/'groups'/gids[0]/'goal_plan.json')
                    span=tuple(plan['magnitude_range'])
                    require(span in RANGES[template['task']],'cannot reserve unknown held magnitude range')
                    bucket=RANGES[template['task']].index(span)
                else:bucket=decision['bucket']
                result.append({'task':template['task'],'class_id':template['class_id'],'room':template['scene']+'/'+template['room'],
                    'phase':record['phase'],'bucket':bucket,'groups':len(gids),'state':str(path)})
    return result


def queue_closures(record, allocation, root):
    """Read the queue actually executed, including the job-local Vast copy."""
    from scripts.campaign_20261110.goal_groups import read_env
    from ispatialgym.sampling.campaign_goal_queue import read_queue
    job = Path(allocation["job_dir"]).resolve()
    env = read_env(job / "job.env")
    staged = job / "campaign_inputs/queue.json"
    path = staged if staged.exists() else Path(env["ISPATIALGYM_CAMPAIGN_GOAL_QUEUE"]).resolve()
    require(path.is_relative_to(root), "executed queue outside campaign")
    raw = read(path)
    actual = read_queue(path, raw["config_path"])
    require([e["group_id"] for e in actual["entries"]] == record["prepared"]["group_ids"], "executed queue changes goal groups")
    require(read(actual["config_path"])["validation_key"] == record["validation_key"], "executed queue changes validation key")
    events = path.with_suffix(".events.jsonl")
    if not events.exists():
        return {}
    closed = {}
    for line in events.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # A concurrent final append may be incomplete.
        require(row.get("queue_sha256") == actual["sha256"], "queue event refers to different bytes")
        if row.get("event") == "closed":
            require(row["group_id"] not in closed, "goal group closed more than once")
            closed[row["group_id"]] = row
    return closed


def claim_finished_groups(record, allocation, gate, cfg, root, *, job_terminal=False):
    from scripts.campaign_20261110.acceptance import audit_job
    from scripts.campaign_20261110.group_lane import claim_group
    from scripts.campaign_20261110.reservation_policy import active_policy
    closed = queue_closures(record, allocation, root)
    done = record.setdefault("groups", {})
    job = Path(allocation["job_dir"])
    for gid, event in closed.items():
        if gid in done:
            continue
        if event["written"] == 0:
            done[gid] = {"accepted": {}, "reason": "closed_without_a_published_member"}
            continue
        members = []
        for path in (job / "inst").rglob("*-tro_state.json"):
            block = read(path)["ispatialgym"].get("campaign", {})
            if block.get("goal_group_id") == gid:
                members.append(path)
        require(len(members) == event["written"], "published TRO count differs from durable group closure")
        reports = [job / "demo" / p.name.removesuffix("-tro_state.json") / "meta.json" for p in members]
        finished = [read(p) for p in reports if p.exists() and isinstance(read(p).get("success"), bool)]
        if not job_terminal and (len(finished) != len(reports) or not all(p["success"] for p in finished)):
            continue
        if not any(p["success"] for p in finished):
            done[gid] = {"accepted": {}, "reason": "all_published_members_failed_demo"}
            continue
        if gate is None or not Path(gate).exists():
            continue
        review = audit_job(job, gate, root=root, started_at=cfg["started_at"], group_closed=True, group_id=gid)
        directory = root / "groups" / gid
        template = template_from(read(directory / "campaign_config.json")["template"])
        write_json_atomic(directory / "release_review.json", review)
        policy = active_policy(root)
        require(policy is not None, "held controller requires active reservations")
        result = claim_group(review, template, policy[1], reservation_hashes=policy[2], root=root,
                             started_at=cfg["started_at"], state_path=root / "dataset/allocations.sqlite")
        done[gid] = {"accepted": result["accepted"], "review_sha256": digest(review)}


def prepare_queue(spec, state, decision, cfg, root, template, save):
    from scripts.campaign_20261110.goal_groups import successful_anchor
    from scripts.campaign_20261110.held_goal_batches import prepare_batch, submission_command
    from scripts.campaign_20261110.base_regions import room_base_region
    from scripts.campaign_20261110.group_lane import check_reservations
    from scripts.campaign_20261110.reservation_policy import active_policy
    phase = decision["phase"]
    if phase == "train" and template.split != "train":
        template = replace(template, split="train", template_id=template.template_id + "__train")
    policy = active_policy(root)
    require(policy is not None, "held queue requires active reservations")
    check_reservations(template, policy[1], policy[2])
    anchor = successful_anchor(spec["anchor_job"], campaign_root=root, ledger=spec["anchor_ledger"], campaign_started_at=cfg["started_at"])
    snapshot = Path(spec["snapshot"])
    region = room_base_region(snapshot / "Behavior-1K/datasets", template.scene, template.room)
    number = len(state["queues"]) + 1
    name = "held_auto_" + spec["id"] + "_" + str(number).zfill(4)
    require(all(c.isalnum() or c in "_.-" for c in name), "unsafe held seed ID")
    seed_config = read(spec["seed_config"])
    key = seed_config["validation_key"]
    require(file_digest(snapshot / ".snapshot_commit") == key["code_sha256"], "held source code identity changed")
    count = decision["groups"]
    index = reserve_indices(root, name, 3*count)
    prepared = prepare_batch(anchor=anchor, template=template, campaign_root=root, batch_id=name, count=count,
        magnitude_range=RANGES[template.task][decision["bucket"]], seed=cfg["seed"]+number*1009+spec["seed_offset"],
        campaign_id=cfg["campaign_id"], code_sha256=key["code_sha256"], physics_sha256=key["physics_sha256"],
        data_sha256=key["data_sha256"], budget_seconds=spec.get("group_budget_seconds", 900), base_region=region)
    actual = read(prepared["config"])["validation_key"]
    if phase != "train" or template.split == seed_config["split"]:
        require(actual == key, "held author changed an already validated source key")
    if phase in state.setdefault("phase_keys", {}):
        require(state["phase_keys"][phase] == actual, "held phase changed its immutable key")
    state["phase_keys"][phase] = actual
    extras, template_customs = source_environment(spec["anchor_job"], root=root)
    command = submission_command(prepared["queue"], snapshot=snapshot, campaign_root=root, v2_root=spec["v2_root"],
                                 base_index=index, extra_env=extras + ["ISPATIALGYM_MAP_OVERLAY_RELEASE=3"])
    if template_customs:
        command += ["--customs", "template"]
    record = {"queue": prepared["queue"], "prepared": prepared, "directory": str(Path(prepared["queue"]).parent),
              "validation_key": actual, "phase": phase, "index": index, "count": 3*count,
              "command": command, "status": "ready", "needs_new_gate": decision["needs_new_gate"], 'decision':dict(decision)}
    if state.get('next_dispatch_backends'):record['dispatch_backends']=state.pop('next_dispatch_backends')
    state["queues"].append(record)
    save()
    write_json_atomic(Path(record["directory"]) / "submission.json", record)
    return record


def source_environment(job, *, root):
    """Preserve the exact private definitions already used by the gated anchor."""
    from scripts.campaign_20261110.goal_groups import read_env
    env = read_env(Path(job) / "job.env")
    if env.get("ISG_CUSTOMS"):
        path = Path(env["ISG_CUSTOMS"]).resolve()
        require(path.is_relative_to(Path(root).resolve()) and path.is_file(), "held private customs outside campaign")
        require(file_digest(path) == env.get("ISPATIALGYM_CAMPAIGN_CUSTOMS_SHA256"), "held native private customs changed")
    keys = ("KEEP_SCENE", "START_PER_ATTEMPT", "ISG_CUSTOMS", "ISPATIALGYM_CAMPAIGN_CUSTOMS_SHA256", "ISPATIALGYM_STABLE_SCENE_DIR")
    classroom_keys = ("ISPATIALGYM_HELD_BOUND_FLOOR_START", "ISGF_TRANSPORT_ASSET_CLOSURE")
    if any(env.get(key) for key in classroom_keys):
        config_path = Path(env.get("ISPATIALGYM_CAMPAIGN_CONFIG", "")).resolve()
        require(config_path.is_relative_to(Path(root).resolve()) and config_path.is_file(),
                "held classroom source config outside campaign")
        config = read(config_path)
        template = config.get("template", {})
        targets = [obj for obj in template.get("objects", []) if obj.get("role") == "target"]
        require(config.get("split") == "eval_model" and template.get("task") == "mental_rotation"
                and template.get("class_id") == "mental_rotation-classroom_mock_up"
                and len(targets) == 1 and targets[0].get("category") == "classroom_mock_up"
                and targets[0].get("model") == "isgm10", "held classroom flags outside exact source scope")
        require(env.get(classroom_keys[0]) == "1", "held classroom bound-floor flag differs")
        provider = Path(root).resolve() / "assets/custom_mental_v1/objects/classroom_mock_up/isgm10"
        require(env.get(classroom_keys[1]) == str(provider) and provider.resolve() == provider
                and (provider / "usd/isgm10.usd").is_file(), "held classroom provider closure differs")
        keys += classroom_keys
    return [key + "=" + env[key] for key in keys if env.get(key)], env.get("START_PER_ATTEMPT") == "0"


def record_runtime_failure(record, state):
    from scripts.campaign_20261110.controller_camera import sampling_infrastructure_failure
    allocation = record.get("allocation")
    if allocation is None:
        return None
    failure = sampling_infrastructure_failure(allocation["job_dir"])
    if failure is not None:
        record["requires_new_runtime_gate"] = failure
        state["runtime_failure"] = failure
    return failure


def completed_empty_tail(records):
    """Count only completed attempts since the most recent accepted queue.

    Queued or running work has no measured yield yet. It must not exhaust the
    empty-attempt safeguard or obscure an earlier successful completed queue.
    """
    count = 0
    for record in reversed(records):
        if record.get('status') != 'complete' or record.get('migration', {}).get('never_started'):
            continue
        if any(value.get('accepted') for value in record.get('groups', {}).values()):
            break
        count += 1
    return count


def retired_class(cfg, class_id):
    values = cfg.get('retired_classes', [])
    require(isinstance(values, list) and all(isinstance(v, str) and v for v in values)
            and len(values) == len(set(values)), 'invalid retired runtime class scope')
    return class_id in values


def advance(spec, state, cfg, root, rows, queue, allowance, save, *, adapter=None):
    require(file_digest(spec["seed_config"]) == spec["seed_config_sha256"], "held seed config changed")
    seed_config = read(spec["seed_config"])
    template = template_from(seed_config["template"])
    require((adapter.supports(template.task) if adapter else template.task in RANGES)
            and template.class_id == spec["class_id"], "unsupported goal controller seed")
    source_gate = Path(spec["source_gate"])
    require(file_digest(source_gate) == spec["source_gate_sha256"] and read(source_gate).get("success") is True,
            "held source replay gate changed or failed")
    records = state.setdefault("queues", [])
    gates = state.setdefault("gates", {template.split: spec["source_gate"]})
    for path in spec.get("existing_queues", []):
        if any(r["queue"] == path for r in records):
            continue
        submission = read(Path(path).parent / "submission.json")
        prepared = submission["prepared"]
        actual = read(prepared["config"])["validation_key"]
        require(actual == seed_config["validation_key"], "adopted queue differs from seed key")
        records.append({"queue": path, "prepared": prepared, "validation_key": actual, "phase": template.split,
                        "index": submission["index"], "count": prepared["candidate_instances"], "status": "external_submission",
                        **({"decision": submission["decision"], "command": submission['command']}
                           if adapter is not None else {})})
    for record in records:
        from scripts.campaign_20261110.controller_migration import controller_check
        controller_check(root,record,state,cfg,queue,save,family='held')
        if record["status"] == "complete":
            record_runtime_failure(record, state)
            continue
        routing_key = record['command'][3] if record.get('command') else spec['class_id']
        allocation = record.get("allocation") or reconcile(rows, key=routing_key, index=record["index"],
            count=record["count"], snapshot=spec["snapshot"], tag="goal_groups")
        if allocation is None:
            if state.get("runtime_failure") is not None:
                state["status"] = "runtime_revalidation_required"
                return 0
            if record["status"] == "external_submission" or allowance <= 0:
                state["status"] = "waiting_for_submission"
                continue
            if record["status"] == "ready" and not record.get("dry_run_passed"):
                run_logged(record["command"] + ["--dry-run"], Path(record["directory"]) / "dryrun.txt")
                record["dry_run_passed"] = True
                save()
            if cfg.get('dispatch_spool'):
                from scripts.campaign_20261110.controller_spool import enqueue_record
                enqueue_record(root,record,save,backends=cfg.get('dispatch_backends'))
                return 1
            perform_submit(record, record["command"], save, rows, spec, root, "goal_groups", spec["snapshot"], record["count"])
            return 1
        record["allocation"] = allocation
        if record.get("needs_new_gate"):
            gate = root / "validation/open_loop" / allocation['key'] / allocation["jobid"] / "gate.json"
            if gate.exists() and read(gate).get("success") is True:
                gates[record["phase"]] = str(gate)
        is_terminal = job_terminal(allocation, root, queue)
        claim_finished_groups(record, allocation, gates.get(record["phase"]), cfg, root, job_terminal=is_terminal)
        if not is_terminal:
            state["status"] = "queue_running"
            continue
        closed = queue_closures(record, allocation, root)
        if any(gid not in record.get("groups", {}) for gid, event in closed.items() if event["written"]):
            state["status"] = "waiting_for_demo_or_replay"
            continue
        record["status"] = "complete"
        record_runtime_failure(record, state)
    if state.get("runtime_failure") is not None:
        state["status"] = "runtime_revalidation_required"
        return 0
    save()
    inflight=[record for record in records if record['status']!='complete']
    if retired_class(cfg, spec['class_id']):
        state['status'] = 'retired_runtime_reconciling' if inflight else 'retired_runtime_no_new_groups'
        save()
        return 0
    maximum_groups = spec.get("groups_per_batch", 6)
    maximum_inflight = cfg.get('maximum_inflight_per_seed', 1)
    if template.split == 'eval_standard':
        from scripts.campaign_20261110.training_eligibility import standard_group_proof
        proof = standard_group_proof(root, template.class_id, seed_config['validation_key'])
        state['standard_group_production_proof'] = proof
        if proof is None:
            maximum_groups = 1
            maximum_inflight = 1
            state['production_limit_reason'] = 'Exact runtime/template needs successful distinct eval group with sibling; bounded validation only'
        else:
            state.pop('production_limit_reason', None)
    if len(inflight) >= maximum_inflight:
        return 0
    if any(record.get('needs_new_gate') and record['phase'] not in gates for record in inflight):
        state['status']='waiting_for_new_phase_gate'
        return 0
    zero_tail = completed_empty_tail(records)
    if zero_tail >= cfg.get("maximum_empty_queues", 3):
        state["status"] = "needs_feasibility_diagnosis"
        return 0
    with atomic_directory_lock(root/'controllers/held_planning.lockdir'):
        if adapter is None:
            counts = read_counts(root, template, read(cfg["standard_contracts"])["templates"], read(cfg["room_plan"]))
            reserved=reserve_pending_counts(counts,template,pending_held(root))
            decision = next_work(template, reserved, max_groups=maximum_groups,
                                 training_validated="train" in gates, mode=spec.get("mode", "eval_then_train"))
        else:
            counts = adapter.read_counts(root, template)
            reserved = adapter.reserve_pending_counts(counts, template, adapter.pending(root))
            decision = adapter.next_work(template, reserved, max_groups=maximum_groups,
                training_validated="train" in gates, mode=spec.get("mode", "eval_then_train"))
        state.update(counts=counts,counts_after_pending_reservations=reserved)
        from scripts.campaign_20261110.controller_spool import candidate_budget_used
        remaining_budget = spec["max_candidates"] - candidate_budget_used(records)
        if decision is None or remaining_budget < 3:
            state["status"] = ("pending_work_reserves_remaining_capacity" if inflight else "quota_or_room_capacity_complete") if decision is None else "candidate_budget_exhausted"
            return 0
        decision["groups"] = lifecycle.bounded_group_count(cfg, min(decision["groups"], remaining_budget//3),
                                                            spec.get("group_budget_seconds", 900))
        if allowance <= 0 or decision["groups"] <= 0:
            return 0
        record = (adapter.prepare_queue if adapter else prepare_queue)(spec, state, decision, cfg, root, template, save)
    run_logged(record["command"] + ["--dry-run"], Path(record["directory"]) / "dryrun.txt")
    record["dry_run_passed"] = True
    save()
    if cfg.get('dispatch_spool'):
        from scripts.campaign_20261110.controller_spool import enqueue_record
        enqueue_record(root,record,save,backends=cfg.get('dispatch_backends'))
        return 1
    perform_submit(record, record["command"], save, rows, spec, root, "goal_groups", spec["snapshot"], record["count"])
    return 1


def main(*, adapter=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--once", action="store_true")
    p.add_argument('--max-inflight-per-seed',type=int)
    p.add_argument('--max-new-jobs-per-pass',type=int)
    p.add_argument('--dispatch-spool',action='store_true')
    p.add_argument('--dispatch-backends',help='Optional comma-separated authorized routing subset.')
    p.add_argument('--migrate-pending-after',type=float)
    args = p.parse_args()
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
    require(cfg["schema"] == ("isg-group-controller-v1" if adapter else "isg-held-controller-v1"), "invalid goal controller config")
    require(len({s["id"] for s in cfg["seeds"]}) == len(cfg["seeds"]), "duplicate held seed ID")
    require(all(isinstance(s["max_candidates"], int) and s["max_candidates"] >= 3
                and 1 <= s.get("groups_per_batch", 6) <= 6 for s in cfg["seeds"]), "invalid held candidate budgets")
    root, path = Path(cfg["root"]).resolve(), Path(args.state).resolve()
    require(path.is_relative_to(root), "held controller state outside campaign")
    path.parent.mkdir(parents=True, exist_ok=True)
    while lifecycle.continue_running(cfg):
        with atomic_directory_lock(path.with_suffix(".lockdir")):
            state = read(path, {"config_sha256": original_config_hash, "seeds": {}})
            require(state["config_sha256"] == original_config_hash, "held controller config changed")
            lifecycle.record_policy(cfg, state)
            state['operational_limits']={key:cfg.get(key,default) for key,default in (('maximum_inflight_per_seed',1),('maximum_new_jobs_per_pass',3))}
            state['operational_limits']['dispatch_spool']=bool(cfg.get('dispatch_spool'))
            state['operational_limits']['dispatch_backends']=cfg.get('dispatch_backends')
            state['operational_limits']['migrate_pending_seconds']=cfg.get('migrate_pending_seconds')
            cfg['_migration_checks_remaining']=8
            save = lambda: write_json_atomic(path, state)
            rows, queue = allocations(root), slurm_status()
            allowance = cfg.get('maximum_new_jobs_per_pass',3) if cfg.get('dispatch_spool') else submission_capacity(queue, cfg.get("maximum_new_jobs_per_pass", 3))
            if cfg.get("seed_catalog"):
                if adapter is None:
                    from scripts.campaign_20261110 import controller_held_sources
                    specs = controller_held_sources.load_seeds(cfg, state)
                else:
                    specs = adapter.load_seeds(cfg, state)
            else:
                specs = cfg["seeds"]
            for spec in specs:
                current = state["seeds"].setdefault(spec["id"], {})
                try:
                    allowance -= advance(spec, current, cfg, root, rows, queue, allowance, save, adapter=adapter)
                    current.pop("error", None)
                except Exception as exc:
                    current.update(error=type(exc).__name__+": "+str(exc), error_at=time.time())
                save()
            state["updated_at"] = time.time()
            save()
        if args.once:
            break
        time.sleep(lifecycle.sleep_seconds(cfg))


if __name__ == "__main__":
    main()
