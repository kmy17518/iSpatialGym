"""Validate new-runtime training seeds, then admit them to bounded group refill."""
from __future__ import annotations

from scripts.campaign_20261110 import controller_lifecycle as lifecycle

import argparse
from contextlib import contextmanager
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
from scripts.campaign_20261110.group_lane import check_reservations, physical_template, conceptual_template
from scripts.campaign_20261110.reservation_policy import active_policy
from scripts.campaign_20261110.storage import atomic_directory_lock

AUTHOR_RECONCILIATION = "tray_anchor_goal_reuse_v1"
FIXED_AUTHOR_GAPS = {
    "ContractError: disembedding needs a fresh anchor with verified destination-floor goal candidates",
}


@contextmanager
def admission_planning_lock(root, *, wait_seconds=30):
    """Give a freshly verified source a bounded chance between refill passes.

    A busy live owner is never removed. Only acquisition contention is retried;
    failures after entering the lock remain ordinary authoring failures.
    """
    deadline = time.monotonic() + wait_seconds
    while True:
        lock = atomic_directory_lock(Path(root) / 'controllers/held_planning.lockdir')
        try:
            lock.__enter__()
            break
        except FileExistsError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(.25, remaining))
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


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
    require(permitted_split in (None, "train", "eval_standard", "eval_model", "eval_category"), "unsupported explicit source-proof split")
    allowed_splits = (permitted_split,) if permitted_split else ("train", "eval_standard")
    require(block["campaign_id"] == campaign_id and block["split"] in allowed_splits, "anchor outside permitted split; heldout anchor cannot train")
    require(groups.supports(block["task"]) or (permitted_split and block["task"] in held.RANGES),
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
    return {"schema": ("isg-heldout-anchor-admission-v1" if permitted_split in ("eval_model", "eval_category")
                       else "isg-standard-anchor-admission-v1" if permitted_split == "eval_standard"
                       else "isg-training-anchor-admission-v1"),
        "campaign_id": campaign_id, "source_job": str(job),
        "source_gate": {"path": str(gate_path), "sha256": file_digest(gate_path)}, "instance_id": iid,
        "config": {"path": str(config_path), "sha256": file_digest(config_path)}, "template": asdict(template),
        "validation_key": asdict(key), "perturbations": config["perturbations"],
        "evidence": [asdict(e) for e in evidence], "physical_template_sha256": physical_template(template),
        "active_policy_sha256": policy[0]["policy_sha256"], "source_is_final_member": False}


def admission_id(template, snapshot, perturbations, split="train", author_revision=None):
    require(split in ("train", "eval_standard", "eval_model", "eval_category"), "invalid admission split")
    identity = {"task": template.task, "class_id": template.class_id, "physical": physical_template(template),
                   "definition": template.definition_sha256, "runtime_bddl": template.scene_setup.get("runtime_bddl_sha256"),
                   "binding_pool": template.scene_setup.get("binding_pool"),
                   "code": file_digest(Path(snapshot) / ".snapshot_commit"), "perturbations": perturbations}
    if split != "train":  # Preserve every existing training admission identity.
        identity["admission_split"] = split
    if author_revision is not None:
        standard = split == 'eval_standard' and author_revision in ('standard_native_coverage_v1', 'standard_native_opening_v1')
        held_opening = split in ('eval_model', 'eval_category') and author_revision == 'held_native_opening_v1'
        held_push = split in ('eval_model', 'eval_category') and author_revision == 'held_native_push_endpoint_v1' and template.task in (
            'inter_object_relation_satisfaction-nextto_large_scale', 'inter_object_relation_satisfaction-touching_large_scale')
        require(standard or held_opening or held_push, 'unsupported admission author revision')
        identity['author_revision'] = author_revision
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
    excluded = cfg.get("excluded_classes", [])
    require(isinstance(excluded, list) and all(isinstance(value, str) and value for value in excluded)
            and len(excluded) == len(set(excluded)), "invalid excluded admission classes")
    require(template.class_id not in excluded, "class excluded from this admission runtime")
    classes = cfg.get("admitted_classes")
    if classes is not None:
        require(isinstance(classes, list) and classes and all(isinstance(value, str) and value for value in classes)
                and len(classes) == len(set(classes)), "invalid admission class ownership")
        require(template.class_id in classes, "class owned by another standard lane")
    split = cfg.get("admission_split", "train")
    require(split in ("train", "eval_standard", "eval_model", "eval_category"), "invalid admission split")
    if split in ("eval_standard", "eval_model", "eval_category"):
        if split == "eval_model":
            require(classes is not None, "model admission requires explicit class ownership")
            require(template.split == split, "model admission requires a model evaluation source")
        elif split == "eval_category":
            require(cfg.get('admitted_tasks') is not None, 'category admission requires explicit task ownership')
            require(template.split == split, 'category admission requires a category evaluation source')
        else:
            require(template.split == split, "standard admission requires a standard evaluation source")
        return template
    require(template.split in ("train", "eval_standard"), "heldout source cannot train")
    return template if template.split == "train" else replace(template, split="train", template_id=template.template_id + "__train")


def admission_decision(template, counts, cfg):
    split = cfg.get("admission_split", "train")
    decision = groups.next_work(template, counts, max_groups=1,
                               mode="eval_then_train" if split.startswith("eval_") else "train_only")
    require(decision is not None and decision["phase"] == split, "class has no unreserved admission capacity")
    return decision


def refill_mode(cfg):
    split = cfg.get("admission_split", "train")
    mode = cfg.get("refill_mode", "eval_then_train" if split.startswith("eval_") else "train_only")
    require(mode in (("eval_only", "eval_then_train") if split == "eval_standard"
                     else ("eval_then_train",) if split.startswith("eval_") else ("train_only",)),
            "refill mode is incompatible with admission split")
    return mode


def model_class_pin(template, state, cfg, *, commit=False):
    """One conceptual model template per class or category template per task."""
    split = cfg.get('admission_split')
    if split not in ('eval_model', 'eval_category'):
        return
    identity = conceptual_template(template)
    pins = state.setdefault("model_class_pins" if split == 'eval_model' else 'category_task_pins', {})
    scope = template.class_id if split == 'eval_model' else template.task
    require(pins.get(scope, identity) == identity, "held-out scope already pinned to another conceptual template")
    if commit:
        pins[scope] = identity


def group_budget(cfg, task):
    value = cfg.get("group_budgets_by_task", {}).get(task, cfg.get("group_budget_seconds", 900))
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 18000, "invalid group budget")
    return value


def source_release(template, env, cfg):
    if template.split not in ("eval_model", "eval_category") and not cfg.get("preserve_source_release", False):
        return cfg["v2_root"]
    require(all(env.get(key) for key in ("ISPATIALGYM_GRASP_LIBRARY_V2", "ISPATIALGYM_REACH_MAPS")),
            "model source has no exact native grasp/map release")
    library, maps = Path(env["ISPATIALGYM_GRASP_LIBRARY_V2"]).resolve(), Path(env["ISPATIALGYM_REACH_MAPS"]).resolve()
    require(library.name == "glib" and maps.name == "maps" and library.parent == maps.parent
            and library.is_dir() and maps.is_dir(), "model source has no exact native grasp/map release")
    return str(library.parent)


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


def bind_missing_runtime_identity(template, anchor):
    """Bind a legacy single-group source to its actual native choice for a new queue."""
    if template.scene_setup.get("runtime_template_id"):
        return template
    from ispatialgym.sampling.v2.problem import TemplateChoice
    metadata = anchor["template_meta"]
    native = TemplateChoice.from_json(metadata["choice"])
    require(native.template_id == metadata["template_id"], "native choices do not reproduce source runtime identity")
    require(native.scene == template.scene and template.room in native.rooms,
            "native runtime identity belongs to another scene/room")
    path = Path(anchor["template_dir"]) / "template_meta.json"
    require(read(path) == metadata, "native identity metadata changed after source audit")
    return replace(template, scene_setup={**template.scene_setup, "runtime_template_id": native.template_id,
        "runtime_identity_binding": {"metadata": str(path), "metadata_sha256": file_digest(path),
            "choice_sha256": digest(metadata["choice"]), "method": "native_TemplateChoice_from_audited_anchor"}})


def prepare_indices(proof, cfg, root):
    from scripts.campaign_20261110.planning_indices import prepare as reserve
    source=held.template_from(proof['template'])
    identity=admission_id(source,cfg['snapshot'],proof['perturbations'],cfg.get('admission_split','train'),cfg.get('author_revision'))
    return reserve(root,'group_admit_'+identity[:20],3)


def prepare(proof, allocation, cfg, root, *, prepared_source=None, prepared_indices=None, dry_run=True):
    from scripts.campaign_20261110.goal_batches import prepare_batch, submission_command
    from ispatialgym.sampling.campaign_goal_queue import schema_for_task
    source_template = held.template_from(proof["template"])
    template = scoped_template(source_template, cfg)
    runtime = Path(cfg["snapshot"]) / "ispatialgym/sampling/campaign_goal_queue.py"
    require(runtime.is_file() and schema_for_task(template.task) in runtime.read_text(),
            "target snapshot does not support this queue schema")
    require(all(proof["validation_key"][key] == cfg[key] for key in ("physics_sha256", "data_sha256")),
            "source anchor differs from pinned campaign physics/data")
    require_source_runtime(proof, cfg, root)
    env = read_env(Path(allocation["job_dir"]) / "job.env")
    v2_root = source_release(template, env, cfg)
    identity = admission_id(source_template, cfg["snapshot"], proof["perturbations"], cfg.get("admission_split", "train"), cfg.get('author_revision'))
    name = "group_admit_" + identity[:20]
    if prepared_source is None:
        anchor = successful_anchor(allocation["job_dir"], campaign_root=root, ledger=allocation["ledger"], campaign_started_at=cfg["started_at"])
    else:
        from scripts.campaign_20261110.planning_inputs import checked_anchor
        anchor = checked_anchor(prepared_source, allocation["job_dir"], campaign_root=root,
            ledger=allocation["ledger"], campaign_started_at=cfg["started_at"])
    template = bind_missing_runtime_identity(template, anchor)
    if cfg.get('author_revision') == 'standard_native_coverage_v1':
        from scripts.campaign_20261110.disembedding_coverage import bind_standard_coverage
        template = bind_standard_coverage(template, anchor)
    if cfg.get('author_revision') == 'standard_native_opening_v1':
        from scripts.campaign_20261110.joint_goal_eligibility import bind_standard_opening
        template = bind_standard_opening(template, anchor)
    if cfg.get('author_revision') == 'held_native_opening_v1':
        from scripts.campaign_20261110.joint_goal_eligibility import bind_opening
        require(template.split in ('eval_model', 'eval_category'), 'held opening author requires held evaluation')
        template = bind_opening(template, anchor)
    if template.task.startswith("alignment-") and not template.scene_setup.get("bucket_policy"):
        from scripts.campaign_20261110.goal_groups import with_alignment_buckets
        template = with_alignment_buckets(template, anchor["template_meta"].get("template_facts", {}).get("deviation_range_applied_deg"),
            source={"path": str(Path(anchor["template_dir"]) / "template_meta.json"), "sha256": digest(anchor["template_meta"]),
                    "digest_kind": "canonical_json", "field": "template_facts.deviation_range_applied_deg"})
    counts = groups.read_counts(root, template)
    reserved = groups.reserve_pending_counts(counts, template, groups.pending(root))
    decision = admission_decision(template, reserved, cfg)
    # Reserve before authoring immutable queue files. A contended global index
    # lock must leave no half-prepared queue that blocks the next controller pass.
    index_proof=None
    if prepared_indices is None:
        index = reserve_indices(root, name, 3)
    else:
        from scripts.campaign_20261110.planning_indices import consume
        index,index_proof=consume(prepared_indices,root,name,3)
    prepared = prepare_batch(anchor=anchor, template=template, campaign_root=root, batch_id=name, count=1,
        magnitude_range=groups.ranges(template)[decision["bucket"]], seed=cfg["seed"], campaign_id=cfg["campaign_id"],
        code_sha256=file_digest(Path(cfg["snapshot"]) / ".snapshot_commit"), physics_sha256=proof["validation_key"]["physics_sha256"],
        data_sha256=proof["validation_key"]["data_sha256"], budget_seconds=group_budget(cfg, template.task))
    keys = ("KEEP_SCENE", "START_PER_ATTEMPT", "ISG_CUSTOMS", "ISPATIALGYM_CAMPAIGN_CUSTOMS_SHA256", "ISPATIALGYM_STABLE_SCENE_DIR",
            "ISPATIALGYM_CAMPAIGN_PROTECTED_CARRY")
    extras = [k + "=" + env[k] for k in keys if env.get(k)] + ["ISPATIALGYM_MAP_OVERLAY_RELEASE=3"]
    command = submission_command(prepared["queue"], snapshot=cfg["snapshot"], campaign_root=root, v2_root=v2_root,
        base_index=index, extra_env=extras)
    if env.get("START_PER_ATTEMPT") == "0":
        command += ["--customs", "template"]
    directory = Path(prepared["queue"]).parent
    if dry_run:
        run_logged(command + ["--dry-run"], directory / "dryrun.txt")
    record = {"directory": str(directory), "queue": prepared["queue"], "prepared": prepared, "command": command,
        "index": index, "count": 3, "validation_key": read(prepared["config"])["validation_key"], "phase": template.split,
        "decision": decision, "needs_new_gate": True, "status": "ready", "dry_run_passed": dry_run,
        "admission_id": identity, "source_proof": proof, "v2_root": v2_root}
    if index_proof is not None:record['index_reservation']=index_proof
    backends = cfg.get('dispatch_backends')
    if backends is None and template.task == 'alignment-world_vertical':
        # Native wall provenance includes provider JSON symlinks. Keep those
        # provider permissions intact by staging private worker inputs on Vast.
        backends = ['vast_5090', 'vast_4090']
    if backends:
        record['dispatch_backends'] = list(backends)
    write_json_atomic(directory / "submission.json", record)
    return record


def seed_bootstraps(cfg, state):
    for source in cfg.get("existing_bootstraps", []):
        record = read(source); template = held.template_from(read(record["prepared"]["config"])["template"])
        scoped_template(template, cfg)
        model_class_pin(template, state, cfg, commit=True)
        identity = admission_id(template, cfg["snapshot"], read(record["prepared"]["config"])["perturbations"],
                                cfg.get("admission_split", "train"), cfg.get('author_revision'))
        if identity not in state["records"]:
            record["admission_id"] = identity
            record["status"] = "external_submission"
            # Older immutable bootstrap submissions predate the spool fields.
            directory = Path(record['directory'])
            intent = Path(cfg['root'])/'dispatch/ready'/(directory.name+'.json')
            if intent.exists():
                record.setdefault('dispatch_intent', str(intent))
                record.setdefault('dispatch_receipt', str(Path(cfg['root'])/'dispatch/receipts'/(directory.name+'.json')))
            state["records"][identity] = record


def check_admission_pending(record, cfg, root, queue, save):
    """Use the existing exact queue continuation; never cancel without its proof."""
    if cfg.get('migrate_pending_seconds') is None or cfg.get('_migration_checks_remaining', 0) <= 0:
        return
    allocation = record.get('allocation') or {}
    scheduled = queue.get(str(allocation.get('jobid'))) or [None, None]
    if scheduled[1] != 'PENDING' and not record.get('migration_replacement'):
        return
    cfg['_migration_checks_remaining'] -= 1
    try:
        proof = record['source_proof']
        for name in ('config', 'source_gate'):
            ref = proof[name]
            require(Path(ref['path']).resolve().is_relative_to(root) and
                    file_digest(ref['path']) == ref['sha256'], 'pending admission source changed')
        require(read(proof['source_gate']['path']).get('success') is True,
                'pending admission source replay failed')
        from scripts.campaign_20261110.controller_queue_migration import migrate_pending_queue
        migrate_pending_queue(root, record, save, queue, minimum_age=cfg['migrate_pending_seconds'])
    except Exception as exc:
        record['migration_error'] = {'at': time.time(), 'type': type(exc).__name__, 'reason': str(exc)}
        save()


def observe_admission_allocations(root, cfg, state, save):
    from scripts.campaign_20261110.controller_saved_allocations import adopt_receipt
    started = time.time()
    seed_bootstraps(cfg, state)
    queue = slurm_status()
    rows, blocked, observed, legacy = [], set(), [], False
    cfg['_migration_checks_remaining'] = 8
    for identity, record in state['records'].items():
        if record.get('status') in ('admitted', 'retired_after_claim', 'failed'):
            continue
        try:
            adopted = adopt_receipt(root, record)
            if adopted: save()
            check_admission_pending(record, cfg, root, queue, save)
            allocation = record.get('allocation')
            if allocation:
                rows.append(allocation)
            elif record.get('status') != 'ready' and not record.get('dispatch_receipt'):
                legacy = True
            observed.append({'admission_id': identity, 'index': record.get('index'), 'adopted': adopted,
                             'job_id': (allocation or {}).get('jobid'),
                             'scheduler_state': (queue.get(str((allocation or {}).get('jobid'))) or [None, None])[1]})
            record.pop('allocation_receipt_error', None)
        except Exception as exc:
            blocked.add(identity)
            record['allocation_receipt_error'] = {'observed_at': time.time(), 'reason': repr(exc)}
            save()
    state['early_pending_observation'] = {'started_at': started, 'completed_at': time.time(),
        'records': observed, 'blocked_admissions': sorted(blocked), 'before_allocation_and_source_scan': True}
    save()
    # Legacy unresolved jobs retain the old discovery path, after known pending checks.
    return (allocations(root) if legacy else rows), queue, blocked


def refresh(record, cfg, root, rows, queue, save=None):
    allocation = record.get("allocation") or reconcile(rows, key=record["command"][3], index=record["index"], count=record["count"],
                                                       snapshot=cfg["snapshot"], tag="goal_groups")
    if allocation is None:
        return
    record["allocation"] = allocation
    if save is not None:
        check_admission_pending(record, cfg, root, queue, save)
        if record.get('allocation') != allocation:
            return
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
    template = held.template_from(read(record["prepared"]["config"])["template"])
    held.claim_finished_groups(record, allocation, str(gate), cfg, root, job_terminal=True)
    require(record.get("groups"), "admission lacks a durable closed goal group")
    if held.retired_class(cfg, template.class_id):
        require(template.class_id in cfg.get('excluded_classes', []), 'retired source class can still be rediscovered')
        record.update(status='retired_after_claim', retired_at=time.time(),
                      retirement={'reason':'Exact native runtime replaced; preserve claim and suppress new refill seed',
                                  'gate':str(gate),'gate_sha256':file_digest(gate)})
        return
    gid = record["prepared"]["group_ids"][0]
    review = audit_job(allocation["job_dir"], gate, root=root, started_at=cfg["started_at"], group_closed=True, group_id=gid)
    require(review["validation_key"] == record["validation_key"], "new admission key differs from replay proof")
    proof_path = root / "groups" / gid / "admission_review.json"
    write_json_atomic(proof_path, review)
    spec = {"id": "admitted_" + record["admission_id"][:20], "class_id": template.class_id,
        "seed_config": record["prepared"]["config"], "seed_config_sha256": file_digest(record["prepared"]["config"]),
        "source_gate": str(gate), "source_gate_sha256": file_digest(gate), "anchor_job": allocation["job_dir"],
        "anchor_ledger": allocation["ledger"], "snapshot": cfg["snapshot"], "v2_root": record.get("v2_root", cfg["v2_root"]),
        "max_candidates": cfg.get("max_candidates_per_seed", 1500), "groups_per_batch": 6,
        "mode": refill_mode(cfg),
        "seed_offset": int(record["admission_id"][:8], 16), "group_budget_seconds": group_budget(cfg, template.task),
        "existing_queues": [record["queue"]], "admission_review": {"path": str(proof_path), "sha256": file_digest(proof_path)}}
    if record.get('dispatch_backends'):
        spec['dispatch_backends'] = record['dispatch_backends']
    append_seed(cfg["seed_catalog"], spec, cfg["campaign_id"])
    if record.get("reason"):
        record["recovered_failure"] = {"reason": record.pop("reason"), "failed_gate": record.get("failed_gate")}
    record.update(status="admitted", admitted_seed=spec["id"], admitted_at=time.time())


def refreshable(record, root):
    """A repaired replay must supply new successful evidence before re-audit."""
    if record["status"] in ("admitted", "retired_after_claim"):
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


def gate_observation_cached(observation, sha, *, now=None):
    """Pending quota reservations may expire; immutable validation failures do not."""
    if observation.get('sha256') != sha:
        return False
    if observation.get('reason') == 'ContractError: class has no unreserved admission capacity':
        now = time.time() if now is None else now
        return now < float(observation.get('observed_at', 0)) + 120
    return True


def recent_gates(root):
    rows = []
    for path in (root / "validation/open_loop").glob("*/*/gate.json"):
        try:
            rows.append((path.stat().st_mtime, path))
        except FileNotFoundError:
            continue
    return [path for _, path in sorted(rows, reverse=True)]


def require_source_runtime(proof, cfg, root):
    expected = cfg.get("required_source_snapshot")
    if expected is None:
        return
    snapshot = Path(expected).resolve()
    require(snapshot.is_relative_to(root / "code"), "required source runtime outside campaign snapshots")
    require(proof["validation_key"]["code_sha256"] == file_digest(snapshot / ".snapshot_commit"),
            "source has not passed the required fresh runtime")


def source_in_scope(gate, root, cfg):
    """Cheap negative filter only; every eligible source still gets the full audit."""
    try:
        if cfg.get("required_source_snapshot") and gate.get("snapshot"):
            if Path(gate["snapshot"]).resolve() != Path(cfg["required_source_snapshot"]).resolve():
                return False
        path = Path(gate['instance']).resolve()
        if not path.is_relative_to(root / 'out'):
            return True  # Let the authoritative audit explain invalid paths.
        block = read(path)['ispatialgym'].get('campaign')
        if block is None:
            return False
        split = cfg.get('admission_split', 'train')
        allowed = (split,) if split in ('eval_model', 'eval_category') else ('train', 'eval_standard')
        if block.get('split') not in allowed:
            return False
        if block.get('class_id') in cfg.get('excluded_classes', []):
            return False
        if cfg.get('admitted_classes') is not None and block.get('class_id') not in cfg['admitted_classes']:
            return False
        if cfg.get('admitted_tasks') is not None and block.get('task') not in cfg['admitted_tasks']:
            return False
        return True
    except (KeyError, TypeError, OSError, ValueError):
        return True  # Malformed or unavailable evidence must still fail the full audit.


def ensure_dry_run(record, save):
    """A saved reservation may outlive a failed dry-run; never enqueue it early."""
    if record.get("dry_run_passed") is not True:
        run_logged(record["command"] + ["--dry-run"], Path(record["directory"]) / "dryrun.txt")
        record["dry_run_passed"] = True
        save()


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
            rows, queue, blocked = observe_admission_allocations(root, cfg, state, save)
            from scripts.campaign_20261110.controller_source_allocations import SourceAllocations
            sources = SourceAllocations(root, state, allocations)
            for identity, record in state["records"].items():
                if identity in blocked or not refreshable(record, root):
                    continue
                try:
                    if record["status"] == "ready":
                        ensure_dry_run(record, save)
                        enqueue_record(root, record, save, priority=30)
                    refresh(record, cfg, root, rows, queue, save)
                    record.pop("error", None)
                except Exception as exc:
                    record["error"] = {"at": time.time(), "reason": type(exc).__name__ + ": " + str(exc)}
                save()
            pending = sum(r["status"] not in ("admitted", "failed", "retired_after_claim") for r in state["records"].values())
            allowance = min(cfg.get("new_admissions_per_pass", 2), max(0, cfg.get("maximum_pending_admissions", 12) - pending))
            examined = 0
            for gate in recent_gates(root):
                if allowance <= 0 or examined >= 30:
                    break
                observed = observe_gate(gate)
                if observed is None:
                    continue
                sha, gate_data = observed
                if gate_observation_cached(state["checked_gates"].get(str(gate), {}), sha):
                    continue
                if gate_data.get("success") is not True:
                    continue
                if not source_in_scope(gate_data, root, cfg):
                    continue
                examined += 1
                try:
                    proof = source_proof(gate, root=root, started_at=cfg["started_at"], campaign_id=cfg["campaign_id"],
                        permitted_split=cfg['admission_split'] if cfg.get("admission_split") in ('eval_model', 'eval_category') else None)
                    template = held.template_from(proof["template"])
                    scoped_template(template, cfg)
                    model_class_pin(template, state, cfg)
                    identity = admission_id(template, cfg["snapshot"], proof["perturbations"], cfg.get("admission_split", "train"), cfg.get('author_revision'))
                    found = [sources.find(proof["source_job"])]
                    if not held.job_terminal(found[0], root, queue):
                        continue  # Freeze source template metadata only after its job ends.
                    if identity not in state["records"]:
                        from scripts.campaign_20261110.planning_inputs import prepare_anchor
                        prepared_source = prepare_anchor(found[0]["job_dir"], campaign_root=root,
                            ledger=found[0]["ledger"], campaign_started_at=cfg["started_at"],
                            extra_paths=[Path(cfg["snapshot"]) / ".snapshot_commit"])
                        prepared_indices = prepare_indices(proof,cfg,root)
                        with admission_planning_lock(root):
                            record = prepare(proof, found[0], cfg, root,
                                             prepared_source=prepared_source, prepared_indices=prepared_indices, dry_run=False)
                            state["records"][identity] = record
                            model_class_pin(template, state, cfg, commit=True)
                            save()
                        ensure_dry_run(record, save)
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
