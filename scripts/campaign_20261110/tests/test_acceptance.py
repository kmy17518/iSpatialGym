import datetime
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.campaign_20261110 import acceptance as a, contracts as c


def fixture(root, split="eval_standard", starts=(1, 2)):
    since = time.time() - 60
    snapshot = root / "code/snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / ".snapshot_commit").write_text("test-frozen-code\n")
    obj = c.ObjectRef("apple.n.01_1", "target", "apple", "abc", c.digest("asset"), ready=True, readiness_source="test")
    template = c.Template("template", "object_scaling", "apple", split, "house_single_floor", "room", (obj,), c.digest("definition"))
    perturb = c.Perturbations({obj.instance: c.ScaleState((1, 1, 1), (1, 1, 1))})
    key = c.ValidationKey(template.fingerprint, c.file_digest(snapshot / ".snapshot_commit"), c.digest("physics"), c.digest("data"), perturb.scale_fingerprint)
    job = root / "out/group/job"
    inst = job / "inst/template/instances"
    inst.mkdir(parents=True)
    gate_dir = root / "validation/gate"
    gate_dir.mkdir(parents=True)
    reports = []
    for n, start in enumerate(starts):
        iid = f"fresh_{n}"
        tro = c.attach_tro_metadata({"ispatialgym": {"instance_id": iid, "object_bindings": {obj.instance: {"category": "apple", "model": "abc"}}}},
            campaign_id="campaign", split=split, template=template, created_at=time.time(), seed=n, perturbations=perturb,
            validation_key=key, goal_group_id="group")
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
    return SimpleNamespace(root=root, job=job, gate=gate_path, since=since, reports=reports)


def audit(f, closed=True):
    return a.audit_job(f.job, f.gate, root=f.root, started_at=f.since, group_closed=closed)


def test_release_requires_verified_evidence_and_distinct_fresh_sibling(tmp_path):
    f = fixture(tmp_path)
    result = audit(f)
    assert result["template_validation_passed"]
    assert result["proposed_allocation"] == {"fresh_0": "eval_standard", "fresh_1": "train"}
    assert result["accepted"] is False  # Separate quota claim/export remains necessary.
    assert audit(f, closed=False)["proposed_allocation"] is None


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
    with pytest.raises(c.ContractError, match="duplicates a start"):
        audit(f)
