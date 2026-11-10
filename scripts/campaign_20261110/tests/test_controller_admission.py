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
    f, _, _ = anchor_fixture(tmp_path, monkeypatch, split="eval_model")
    with pytest.raises(c.ContractError, match="cannot train"):
        admission.source_proof(f.gate, root=tmp_path, started_at=f.since, campaign_id="campaign")
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


def test_catalog_consumer_reaudits_new_key_and_pins_existing_seed(tmp_path, monkeypatch):
    from scripts.campaign_20261110.acceptance import audit_job
    from scripts.campaign_20261110 import group_lane, reservation_policy
    f, template, config = anchor_fixture(tmp_path, monkeypatch, split="train")
    review = audit_job(f.job, f.gate, root=tmp_path, started_at=f.since, group_closed=True, group_id="group")
    proof = tmp_path / "groups/group/admission_review.json"; c.write_json_atomic(proof, review)
    snapshot = tmp_path / "code/snapshot"
    spec = {"id": "admitted", "snapshot": str(snapshot), "class_id": template.class_id,
        "max_candidates": 1500, "groups_per_batch": 6, "seed_config": str(config), "seed_config_sha256": c.file_digest(config),
        "anchor_job": str(f.job), "source_gate": str(f.gate), "admission_review": {"path": str(proof), "sha256": c.file_digest(proof)}}
    catalog = tmp_path / "controllers/groups/catalog.json"
    admission.append_seed(catalog, spec, "campaign")
    cfg = {"root": str(tmp_path), "seed_catalog": str(catalog), "campaign_id": "campaign", "admission_snapshot": str(snapshot),
           "started_at": f.since, "seeds": []}
    monkeypatch.setattr(group_lane, "check_reservations", lambda *args: None)
    monkeypatch.setattr(reservation_policy, "active_policy", lambda root: ({}, {}, {}))
    state = {}
    assert groups.load_seeds(cfg, state) == [spec]
    assert state["admitted_seed_hashes"] == {"admitted": c.digest(spec)}
    raw = json.loads(catalog.read_text()); raw["seeds"][0]["source_gate"] = "different"; c.write_json_atomic(catalog, raw)
    with pytest.raises(c.ContractError, match="removed or changed"):
        groups.load_seeds(cfg, state)
