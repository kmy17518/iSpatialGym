from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from scripts.campaign_20261110 import controller_admission as admission, controller_groups as groups
from scripts.campaign_20261110 import contracts as c
from scripts.campaign_20261110.tests.test_acceptance import fixture


def anchor_fixture(root, monkeypatch, split="eval_standard"):
    f = fixture(root, split=split)
    template = replace(f.template, task="inter_object_relation_satisfaction-ontop_small_scale")
    for path in (f.job / "inst").rglob("*-tro_state.json"):
        tro = json.loads(path.read_text()); block = tro["ispatialgym"]["campaign"]
        block["task"] = template.task
        block["template_sha256"] = template.fingerprint
        block["validation_key"]["template_sha256"] = template.fingerprint
        c.write_json_atomic(path, tro)
        iid = tro["ispatialgym"]["instance_id"]
        c.write_json_atomic(path.with_name(iid+".oracle.json"), {"tro_sha256": c.file_digest(path)})
    gate = json.loads(f.gate.read_text()); gate["tro_sha256"] = c.file_digest(gate["instance"]); c.write_json_atomic(f.gate, gate)
    block = json.loads(Path(gate["instance"]).read_text())["ispatialgym"]["campaign"]
    config = {"campaign_id": "campaign", "split": split, "purpose": "validation", "template": asdict(template),
              "validation_key": block["validation_key"], "perturbations": block["perturbations"]}
    config_path = root / "groups/source/campaign_config.json"; c.write_json_atomic(config_path, config)
    (f.job / "job.env").write_text("ISPATIALGYM_CAMPAIGN_CONFIG="+str(config_path)+"\n")
    monkeypatch.setattr(admission, "active_policy", lambda root: ({"policy_sha256": "a"*64}, {}, {}))
    monkeypatch.setattr(admission, "check_reservations", lambda *args: None)
    return f, template, config_path


def test_anchor_admission_requires_exact_fresh_demo_replay_and_key(tmp_path, monkeypatch):
    f, template, config = anchor_fixture(tmp_path, monkeypatch)
    proof = admission.source_proof(f.gate, root=tmp_path, started_at=f.since, campaign_id="campaign")
    assert proof["validation_key"]["template_sha256"] == template.fingerprint
    assert proof["source_is_final_member"] is False and len(proof["evidence"]) == 2
    value = json.loads(config.read_text()); value["validation_key"]["physics_sha256"] = "b"*64; c.write_json_atomic(config, value)
    with pytest.raises(c.ContractError, match="actual validation key"):
        admission.source_proof(f.gate, root=tmp_path, started_at=f.since, campaign_id="campaign")


def test_heldout_or_goal_overridden_anchor_cannot_enter_training(tmp_path, monkeypatch):
    f, template, _ = anchor_fixture(tmp_path, monkeypatch, split="eval_model")
    with pytest.raises(c.ContractError, match="cannot train"):
        admission.source_proof(f.gate, root=tmp_path, started_at=f.since, campaign_id="campaign")
    proof = admission.source_proof(f.gate, root=tmp_path, started_at=f.since, campaign_id="campaign", permitted_split="eval_model")
    assert proof["template"]["split"] == "eval_model" and proof["schema"] == "isg-heldout-anchor-admission-v1"
    with pytest.raises(c.ContractError, match="heldout source cannot train"):
        admission.scoped_template(template, {})
    second = tmp_path / "another"; f, _, _ = anchor_fixture(second, monkeypatch)
    value = json.loads(f.reports[0].read_text()); value["goal_override"] = "oracle"; c.write_json_atomic(f.reports[0], value)
    with pytest.raises(c.ContractError, match="overwritten"):
        admission.source_proof(f.gate, root=second, started_at=f.since, campaign_id="campaign")


def test_admission_identity_keeps_applied_scale_and_light_separate(tmp_path, monkeypatch):
    f, template, _ = anchor_fixture(tmp_path, monkeypatch)
    snapshot = tmp_path / "code/snapshot"
    base = admission.admission_id(template, snapshot, {"scales": [1,1,1], "lighting": {}})
    assert base != admission.admission_id(template, snapshot, {"scales": [1.1,1.1,1.1], "lighting": {}})
    assert base != admission.admission_id(template, snapshot, {"scales": [1,1,1], "lighting": {"gain": 2}})
    assert base == admission.admission_id(template, snapshot, {"scales": [1,1,1], "lighting": {}}, "train")
    assert base != admission.admission_id(template, snapshot, {"scales": [1,1,1], "lighting": {}}, "eval_standard")
    model = admission.admission_id(template, snapshot, {"scales": [1,1,1], "lighting": {}}, "eval_model")
    assert model != base
    assert model != admission.admission_id(template, snapshot, {"scales": [1,1,1], "lighting": {}}, "eval_standard")
    legacy = {"task": template.task, "class_id": template.class_id, "physical": admission.physical_template(template),
              "definition": template.definition_sha256, "runtime_bddl": template.scene_setup.get("runtime_bddl_sha256"),
              "binding_pool": template.scene_setup.get("binding_pool"), "code": c.file_digest(snapshot / ".snapshot_commit"),
              "perturbations": {"scales": [1,1,1], "lighting": {}}}
    assert base == c.digest(legacy)


def test_standard_admission_preserves_default_contract_ownership_and_sibling_capacity(tmp_path, monkeypatch):
    _, template, _ = anchor_fixture(tmp_path, monkeypatch)
    cfg = {"admission_split": "eval_standard", "admitted_classes": [template.class_id]}
    assert admission.scoped_template(template, cfg) is template
    with pytest.raises(c.ContractError, match="another standard lane"):
        admission.scoped_template(template, {**cfg, "admitted_classes": ["owned_elsewhere"]})
    with pytest.raises(c.ContractError, match="standard evaluation source"):
        admission.scoped_template(replace(template, split="train"), cfg)
    counts = {"evaluation": {}, "train_buckets": {}, "class_remaining": 1, "task_remaining": 10}
    decision = admission.admission_decision(template, counts, cfg)
    assert decision["phase"] == "eval_standard" and decision["groups"] == 1
    with pytest.raises(c.ContractError, match="no unreserved admission capacity"):
        admission.admission_decision(template, {**counts, "class_remaining": 0}, cfg)
    with pytest.raises(c.ContractError, match="no unreserved admission capacity"):
        admission.admission_decision(template, {**counts, "evaluation": {0: 10}}, cfg)


def test_new_attachment_runtime_can_be_scoped_without_rekeying_existing_families():
    scoped = {"admitted_tasks": ["configuration_matching-attachment"]}
    assert admission.task_allowed("configuration_matching-attachment", scoped)
    assert not admission.task_allowed("alignment-axial", scoped)
    assert admission.task_allowed("alignment-axial", {})
    with pytest.raises(c.ContractError, match="task scope"):
        admission.task_allowed("alignment-axial", {"admitted_tasks": ["invented_task"]})


def test_model_admission_never_converts_source_or_falls_back_to_training(tmp_path, monkeypatch):
    _, template, _ = anchor_fixture(tmp_path, monkeypatch, split="eval_model")
    cfg = {"admission_split": "eval_model", "admitted_classes": [template.class_id]}
    assert admission.scoped_template(template, cfg) is template
    with pytest.raises(c.ContractError, match="native grasp/map release"):
        admission.source_release(template, {}, {"v2_root": "/default"})
    release = tmp_path / "exact_native_release"
    (release / "glib").mkdir(parents=True); (release / "maps").mkdir()
    env = {"ISPATIALGYM_GRASP_LIBRARY_V2": str(release / "glib"), "ISPATIALGYM_REACH_MAPS": str(release / "maps")}
    assert admission.source_release(template, env, {"v2_root": "/default"}) == str(release)
    with pytest.raises(c.ContractError, match="native grasp/map release"):
        admission.source_release(template, {**env, "ISPATIALGYM_REACH_MAPS": str(tmp_path / "maps")}, {"v2_root": "/default"})
    for split in ("train", "eval_standard", "eval_category"):
        with pytest.raises(c.ContractError, match="model evaluation source"):
            admission.scoped_template(replace(template, split=split), cfg)
    with pytest.raises(c.ContractError, match="explicit class ownership"):
        admission.scoped_template(template, {"admission_split": "eval_model"})
    counts = {"evaluation": {0: 6}, "train_buckets": {}, "class_remaining": 0, "task_remaining": 0}
    decision = admission.admission_decision(template, counts, cfg)
    assert decision["phase"] == "eval_model" and decision["groups"] == 1
    with pytest.raises(c.ContractError, match="no unreserved admission capacity"):
        admission.admission_decision(template, {**counts, "evaluation": {0: 9}}, cfg)
    assert groups.next_work(template, {**counts, "evaluation": {0: 9}}, training_validated=True) is None
    state = {}
    admission.model_class_pin(template, state, cfg, commit=True)
    changed = replace(template, objects=(replace(template.objects[0], model="another", asset_sha256=c.digest("another")),))
    with pytest.raises(c.ContractError, match="another conceptual template"):
        admission.model_class_pin(changed, state, cfg)
    assert admission.group_budget({"group_budgets_by_task": {"configuration_matching-block_design": 9000}},
                                  "configuration_matching-block_design") == 9000


def test_category_admission_requires_its_own_proof_and_one_template_per_task(tmp_path, monkeypatch):
    f, template, _ = anchor_fixture(tmp_path, monkeypatch, split='eval_category')
    cfg = {'admission_split': 'eval_category', 'admitted_tasks': [template.task]}
    assert admission.scoped_template(template, cfg) is template
    proof = admission.source_proof(f.gate, root=tmp_path, started_at=f.since, campaign_id='campaign', permitted_split='eval_category')
    assert proof['template']['split'] == 'eval_category'
    for split in (None, 'eval_model'):
        with pytest.raises(c.ContractError, match='permitted split'):
            admission.source_proof(f.gate, root=tmp_path, started_at=f.since, campaign_id='campaign', permitted_split=split)
    with pytest.raises(c.ContractError, match='explicit task ownership'):
        admission.scoped_template(template, {'admission_split': 'eval_category'})
    with pytest.raises(c.ContractError, match='category evaluation source'):
        admission.scoped_template(replace(template, split='eval_model'), cfg)
    state = {}; admission.model_class_pin(template, state, cfg, commit=True)
    changed = replace(template, class_id='another_category_class', objects=(replace(template.objects[0], model='different', asset_sha256=c.digest('different')),))
    with pytest.raises(c.ContractError, match='another conceptual template'):
        admission.model_class_pin(changed, state, cfg)
    assert admission.admission_id(template, tmp_path/'code/snapshot', {}, 'eval_category') != admission.admission_id(template, tmp_path/'code/snapshot', {}, 'eval_model')
    counts = {'evaluation': {0: 4}, 'train_buckets': {}, 'class_remaining': 1000, 'task_remaining': 1000}
    assert groups.next_work(template, counts) is None


def test_scope_filter_skips_unowned_history_without_replacing_full_proof(tmp_path, monkeypatch):
    f, template, _ = anchor_fixture(tmp_path, monkeypatch, split='eval_model')
    gate = json.loads(f.gate.read_text())
    cfg = {'admission_split': 'eval_model', 'admitted_classes': [template.class_id]}
    assert admission.source_in_scope(gate, tmp_path, cfg)
    assert not admission.source_in_scope(gate, tmp_path, {**cfg, 'admission_split': 'train'})
    assert not admission.source_in_scope(gate, tmp_path, {**cfg, 'admitted_classes': ['different']})
    assert admission.source_in_scope({**gate, 'instance': '/outside/missing'}, tmp_path, cfg)
    assert admission.source_in_scope({}, tmp_path, cfg)


def test_runtime_exclusion_keeps_other_repaired_classes_eligible(tmp_path, monkeypatch):
    f, template, _ = anchor_fixture(tmp_path, monkeypatch)
    cfg = {'excluded_classes': ['disembedding_02']}
    assert admission.scoped_template(template, cfg).split == 'train'
    assert admission.source_in_scope(json.loads(f.gate.read_text()), tmp_path, cfg)
    repaired = replace(template, class_id='newly_ready_existing_class')
    assert admission.scoped_template(repaired, cfg).class_id == repaired.class_id
    with pytest.raises(c.ContractError, match='excluded from this admission runtime'):
        admission.scoped_template(replace(template, class_id='disembedding_02'), cfg)
    assert not admission.source_in_scope(json.loads(f.gate.read_text()), tmp_path,
                                         {'excluded_classes': [template.class_id]})
    with pytest.raises(c.ContractError, match='invalid excluded admission classes'):
        admission.scoped_template(template, {'excluded_classes': 'disembedding_02'})


def test_author_upgrade_reopens_only_the_fixed_gap_and_retains_previous_evidence():
    previous = {"sha256": "a"*64, "reason": next(iter(admission.FIXED_AUTHOR_GAPS)), "observed_at": 123}
    state = {"checked_gates": {"tray": dict(previous), "other": {"sha256": "b"*64, "reason": "replay QA missing"}}}
    admission.reconcile_author_gaps(state)
    assert "sha256" not in state["checked_gates"]["tray"]
    assert state["checked_gates"]["other"]["sha256"] == "b"*64
    assert state["author_retry_history"][0]["previous_observation"] == previous
    assert state["author_retry_history"][0]["helper_sha256"] == c.file_digest(admission.__file__)
    admission.reconcile_author_gaps(state)
    assert len(state["author_retry_history"]) == 1


def test_catalog_is_additive_and_changed_existing_seed_is_rejected(tmp_path):
    path = tmp_path / "controllers/groups/catalog.json"
    a, b = {"id": "a", "key": "unchanged"}, {"id": "b", "key": "new"}
    admission.append_seed(path, a, "campaign")
    admission.append_seed(path, a, "campaign")
    admission.append_seed(path, b, "campaign")
    raw = json.loads(path.read_text())
    assert raw["seeds"] == [a,b] and raw["revision"] == 2
    assert len(list((path.parent / "catalog_revisions").glob("*.json"))) == 2
    with pytest.raises(c.ContractError, match="seed ID changed"):
        admission.append_seed(path, {**a, "key": "replaced"}, "campaign")


def test_failed_replay_requires_changed_successful_gate_before_full_readmission(tmp_path):
    path = tmp_path / "validation/open_loop/task/123/gate.json"
    c.write_json_atomic(path, {"success": False, "reason": "replay startup permission error"})
    record = {"status": "failed", "reason": "new_runtime_admission_physics_replay_failed",
              "allocation": {"key": "task", "jobid": "123"},
              "failed_gate": {"path": str(path), "sha256": c.file_digest(path)}}
    assert not admission.refreshable(record, tmp_path)
    c.write_json_atomic(path, {"success": False, "reason": "a new failed replay"})
    assert not admission.refreshable(record, tmp_path)
    c.write_json_atomic(path, {"success": True, "reason": "fresh physics replay passed"})
    assert admission.refreshable(record, tmp_path)
    assert record["status"] == "failed"  # The existing full refresh/audit still decides admission.
    record["reason"] = "new_runtime_admission_has_no_successful_demo"
    assert admission.refreshable(record, tmp_path)  # An externally repaired prelaunch job still needs full new proof.
    path.unlink()
    assert not admission.refreshable(record, tmp_path)


def test_archiving_gate_during_admission_scan_does_not_stop_controller(tmp_path, monkeypatch):
    path = tmp_path / "validation/open_loop/task/123/gate.json"
    c.write_json_atomic(path, {"success": False})
    assert admission.recent_gates(tmp_path) == [path]
    def archive_during_hash(value):
        Path(value).rename(Path(value).with_name("failed_gate.json"))
        return c.file_digest(value)
    monkeypatch.setattr(admission, "file_digest", archive_during_hash)
    assert admission.observe_gate(path) is None
    assert admission.recent_gates(tmp_path) == []


def test_pending_admission_canary_reserves_class_capacity_before_catalog_admission(tmp_path):
    record = {"admission_id": "a", "status": "external_submission", "phase": "train", "prepared": {"group_ids": ["new"]},
              "decision": {"bucket": 0}}
    c.write_json_atomic(tmp_path / "controllers/admission/state.json", {"records": {"a": record}})
    c.write_json_atomic(tmp_path / "groups/new/campaign_config.json", {"template": {
        "task": "inter_object_relation_satisfaction-ontop_small_scale", "class_id": "soap"}})
    rows = groups.pending(tmp_path)
    assert len(rows) == 1 and rows[0]["groups"] == 1 and rows[0]["class_id"] == "soap"
    record["status"] = "admitted"
    c.write_json_atomic(tmp_path / "controllers/admission/state.json", {"records": {"a": record}})
    assert groups.pending(tmp_path) == []


@pytest.mark.parametrize("split", ["train", "eval_standard"])
def test_catalog_consumer_reaudits_new_key_and_pins_existing_seed(tmp_path, monkeypatch, split):
    from scripts.campaign_20261110.acceptance import audit_job
    from scripts.campaign_20261110 import group_lane, reservation_policy
    f, template, config = anchor_fixture(tmp_path, monkeypatch, split=split)
    review = audit_job(f.job, f.gate, root=tmp_path, started_at=f.since, group_closed=True, group_id="group")
    proof = tmp_path / "groups/group/admission_review.json"; c.write_json_atomic(proof, review)
    snapshot = tmp_path / "code/snapshot"
    spec = {"id": "admitted", "snapshot": str(snapshot), "class_id": template.class_id,
        "max_candidates": 1500, "groups_per_batch": 6, "seed_config": str(config), "seed_config_sha256": c.file_digest(config),
        "anchor_job": str(f.job), "source_gate": str(f.gate), "admission_review": {"path": str(proof), "sha256": c.file_digest(proof)}}
    catalog = tmp_path / "controllers/groups/catalog.json"
    admission.append_seed(catalog, spec, "campaign")
    cfg = {"root": str(tmp_path), "seed_catalog": str(catalog), "campaign_id": "campaign", "admission_snapshot": str(snapshot),
           "started_at": f.since, "seeds": [], "admission_split": split, "admitted_classes": [template.class_id]}
    monkeypatch.setattr(group_lane, "check_reservations", lambda *args: None)
    monkeypatch.setattr(reservation_policy, "active_policy", lambda root: ({}, {}, {}))
    state = {}
    assert groups.load_seeds(cfg, state) == [spec]
    assert state["admitted_seed_hashes"] == {"admitted": c.digest(spec)}
    raw = json.loads(catalog.read_text()); raw["seeds"][0]["source_gate"] = "different"; c.write_json_atomic(catalog, raw)
    with pytest.raises(c.ContractError, match="removed or changed"):
        groups.load_seeds(cfg, state)
