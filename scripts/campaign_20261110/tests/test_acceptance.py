import datetime
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.campaign_20261110 import acceptance as a, contracts as c


def fixture(root, split="eval_standard", starts=(1, 2), prefix=""):
    since = time.time() - 60
    snapshot = root / "code/snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / ".snapshot_commit").write_text("test-frozen-code\n")
    obj = c.ObjectRef("apple.n.01_1", "target", "apple", "abc", c.digest("asset"), ready=True, readiness_source="test")
    template = c.Template("template", "object_scaling", "apple", split, "house_single_floor", "room", (obj,), c.digest("definition"))
    perturb = c.Perturbations({obj.instance: c.ScaleState((1, 1, 1), (1, 1, 1))})
    key = c.ValidationKey(template.fingerprint, c.file_digest(snapshot / ".snapshot_commit"), c.digest("physics"), c.digest("data"), perturb.scale_fingerprint)
    job = root / "out" / (prefix+"group") / "job"
    inst = job / "inst/template/instances"
    inst.mkdir(parents=True)
    gate_dir = root / "validation" / (prefix+"gate")
    gate_dir.mkdir(parents=True)
    reports = []
    for n, start in enumerate(starts):
        iid = f"{prefix}fresh_{n}"
        tro = c.attach_tro_metadata({"ispatialgym": {"instance_id": iid, "object_bindings": {obj.instance: {"category": "apple", "model": "abc"}}}},
            campaign_id="campaign", split=split, template=template, created_at=time.time(), seed=n, perturbations=perturb,
            validation_key=key, goal_group_id=f"{prefix}group")
        tro["ispatialgym"]["campaign"]["runtime"] = {"goal_group": {"requested_goal": [0, 1, 2], "requested_start": [start, 3, 4]}}
        tro_path = inst / f"{iid}-tro_state.json"
        c.write_json_atomic(tro_path, tro)
        c.write_json_atomic(inst / f"{iid}.oracle.json", {"tro_sha256": c.file_digest(tro_path)})
        demo_dir = job / "demo" / iid
        demo_dir.mkdir(parents=True)
        demo = demo_dir / "demo.hdf5"
        demo.write_bytes(b"fixture-hdf5")
        report = {"mode": "oracle", "instance_id": iid, "status": "success", "success": True,
            "qa": {"enabled": True, "hard_triggered": False, "metrics": {}},
            "loader": {"campaign_restoration": {"scale_verified": True}}, "goal_override": None,
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "provenance": {"isg_root": str(snapshot)}}
        report_path = demo_dir / "meta.json"
        c.write_json_atomic(report_path, report)
        reports.append(report_path)
        if n == 0:
            replay = gate_dir / "replay" / iid / "meta.json"
            c.write_json_atomic(replay, {**report, "mode": "replay"})
            gate = {"source_job": str(job), "success": True, "replay_mode": "physics_action_open_loop", "snapshot": str(snapshot),
                "instance": str(tro_path), "demo": str(demo), "tro_sha256": c.file_digest(tro_path), "demo_sha256": c.file_digest(demo)}
    gate_path = gate_dir / "gate.json"
    c.write_json_atomic(gate_path, gate)
    return SimpleNamespace(root=root, job=job, gate=gate_path, since=since, reports=reports, template=template)


def audit(f, closed=True):
    return a.audit_job(f.job, f.gate, root=f.root, started_at=f.since, group_closed=closed)


def test_release_requires_verified_evidence_and_distinct_fresh_sibling(tmp_path):
    f = fixture(tmp_path)
    result = audit(f)
    assert result["template_validation_passed"]
    assert result["proposed_allocation"] == {"fresh_0": "eval_standard", "fresh_1": "train"}
    assert result["accepted"] is False  # Separate quota claim/export remains necessary.
    assert audit(f, closed=False)["proposed_allocation"] is None


def test_explicit_distinct_member_subset_preserves_strict_review(tmp_path):
    f=fixture(tmp_path,starts=(1,2,3))
    value=a.audit_job(f.job,f.gate,root=f.root,started_at=f.since,group_closed=True,member_ids=['fresh_0','fresh_2'])
    assert value['selected_member_ids']==['fresh_0','fresh_2']
    assert value['proposed_allocation']=={'fresh_0':'eval_standard','fresh_2':'train'}
    with pytest.raises(c.ContractError,match='not distinct'):
        a.audit_job(f.job,f.gate,root=f.root,started_at=f.since,group_closed=True,member_ids=['fresh_0','missing'])


@pytest.mark.parametrize("split,expected", [("eval_standard", "train"), ("eval_model", "eval_model"), ("eval_category", "eval_category")])
def test_singletons_do_not_leak_heldout_assets_into_train(tmp_path, split, expected):
    f = fixture(tmp_path, split=split, starts=(1,))
    assert audit(f)["proposed_allocation"] == {"fresh_0": expected}


def test_modified_demo_and_disabled_qa_cannot_pass_release(tmp_path):
    f = fixture(tmp_path)
    report = json.loads(f.reports[1].read_text())
    report["qa"]["enabled"] = False
    c.write_json_atomic(f.reports[1], report)
    with pytest.raises(c.ContractError, match="QA"):
        audit(f)
    report["qa"]["enabled"] = True
    c.write_json_atomic(f.reports[1], report)
    source = Path(json.loads(f.gate.read_text())["demo"])
    source.write_bytes(b"changed")
    with pytest.raises(c.ContractError, match="demo content"):
        audit(f)


def test_same_intended_start_cannot_be_counted_as_two_instances(tmp_path):
    f = fixture(tmp_path, starts=(1, 1))
    result = audit(f)
    assert result["proposed_allocation"] == {"fresh_0": "train"}
    assert [m["instance_id"] for m in result["excluded_duplicate_members"]] == ["fresh_1"]


def test_duplicate_start_cannot_make_a_heldout_training_singleton(tmp_path):
    f = fixture(tmp_path, starts=(1, 1), split="eval_model")
    assert audit(f)["proposed_allocation"] == {"fresh_0": "eval_model"}


def test_reuse_gate_requires_exact_template_key_and_campaign(tmp_path):
    source = fixture(tmp_path, prefix="source_")
    target = fixture(tmp_path, prefix="target_")
    result = a.audit_job(target.job, source.gate, root=tmp_path, started_at=source.since,
                         group_closed=True, group_id="target_group")
    assert result["proposed_allocation"] == {"target_fresh_0": "eval_standard", "target_fresh_1": "train"}
    with pytest.raises(c.ContractError, match="explicit current group"):
        a.audit_job(target.job, source.gate, root=tmp_path, started_at=source.since, group_closed=True)
    path = next((target.job / "inst").rglob("*-tro_state.json"))
    tro = json.loads(path.read_text())
    tro["ispatialgym"]["campaign"]["campaign_id"] = "different"
    c.write_json_atomic(path, tro)
    with pytest.raises(c.ContractError, match="campaign identities"):
        a.audit_job(target.job, source.gate, root=tmp_path, started_at=source.since,
                    group_closed=True, group_id="target_group")
    tro["ispatialgym"]["campaign"]["campaign_id"] = "campaign"
    tro["ispatialgym"]["campaign"]["validation_key"]["physics_sha256"] = c.digest("other-physics")
    c.write_json_atomic(path, tro)
    with pytest.raises(c.ContractError, match="validation keys"):
        a.audit_job(target.job, source.gate, root=tmp_path, started_at=source.since,
                    group_closed=True, group_id="target_group")


def test_recovered_replay_uses_exact_hash_bound_attempt_report(tmp_path):
    f=fixture(tmp_path)
    gate=a.read(f.gate);iid=a.read(gate['instance'])['ispatialgym']['instance_id']
    old=f.gate.parent/'replay'/iid/'meta.json'
    target=f.gate.parent/'attempt_02/replay'/iid/'meta.json';target.parent.mkdir(parents=True)
    old.rename(target)
    gate['replay_meta_paths']=[str(target)]
    gate['scene_identity']={'verified_replay':True,'replay_report':str(target),'replay_report_sha256':c.file_digest(target)}
    c.write_json_atomic(f.gate,gate)
    result=audit(f)
    assert next(e for e in result['evidence'] if e['kind']=='open_loop_replay')['report_path']==str(target)
    target.write_text(target.read_text()+' ')
    with pytest.raises(c.ContractError,match='report hash changed'):
        audit(f)


def test_explicit_replay_locator_rejects_ambiguity_escape_and_wrong_instance(tmp_path):
    f=fixture(tmp_path);gate=a.read(f.gate);iid=a.read(gate['instance'])['ispatialgym']['instance_id']
    original=f.gate.parent/'replay'/iid/'meta.json'
    second=f.gate.parent/'attempt_02/replay'/iid/'meta.json';second.parent.mkdir(parents=True);second.write_bytes(original.read_bytes())
    gate['replay_meta_paths']=[str(original),str(second)]
    with pytest.raises(c.ContractError,match='exactly one'):
        a.replay_report_path(gate,f.gate,iid,root=tmp_path,started_at=f.since)
    gate['replay_meta_paths']=[str(f.job/'demo'/iid/'meta.json')]
    with pytest.raises(c.ContractError,match='outside this gate'):
        a.replay_report_path(gate,f.gate,iid,root=tmp_path,started_at=f.since)
    gate['replay_meta_paths']=[str(second)];report=a.read(second);report['instance_id']='different';c.write_json_atomic(second,report)
    with pytest.raises(c.ContractError,match='exactly one'):
        a.replay_report_path(gate,f.gate,iid,root=tmp_path,started_at=f.since)
