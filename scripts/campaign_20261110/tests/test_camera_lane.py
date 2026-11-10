import datetime
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.campaign_20261110 import camera_lane as lane, contracts as c


def fixture(root, distances=(1, 2, 3, 3.1, 4, 4.5, 5, 6, 7, 10, 1.5, 5.5)):
    since = time.time() - 60
    snapshot = root / "code/snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / ".snapshot_commit").write_text("camera-code\n")
    obj = c.ObjectRef("floor.n.01_1", "support", "floors", "abc", c.digest("asset"), semantic_kind="floor", ready=True, readiness_source="test")
    template = c.Template("template", lane.TASK, lane.TASK, "eval_standard", "house_single_floor", "room", (obj,), c.digest("definition"))
    perturb = c.Perturbations({obj.instance: c.ScaleState((1, 1, 1), (1, 1, 1))})
    key = c.ValidationKey(template.fingerprint, c.file_digest(snapshot / ".snapshot_commit"), c.digest("physics"), c.digest("data"), perturb.scale_fingerprint)
    job = root / "out/camera/job"
    inst = job / "inst/template/instances"
    inst.mkdir(parents=True)
    gate_dir = root / "validation/gate"
    gate_dir.mkdir(parents=True)
    for n, distance in enumerate(distances):
        iid = f"camera_{n:02d}"
        tro = c.attach_tro_metadata({"ispatialgym": {"instance_id": iid, "object_bindings": {obj.instance: {"category": "floors", "model": "abc"}}}},
            campaign_id="campaign", split="eval_standard", template=template, created_at=time.time(), seed=n, perturbations=perturb, validation_key=key)
        tro_path = inst / f"{iid}-tro_state.json"
        c.write_json_atomic(tro_path, tro)
        c.write_json_atomic(inst / f"{iid}.oracle.json", {"tro_sha256": c.file_digest(tro_path), "path_bucket_m": float(lane.path_bucket(distance))})
        c.write_json_atomic(inst / f"{iid}.summary.json", {"geodesic_m": distance, "path_bucket_m": float(lane.path_bucket(distance))})
        demo_dir = job / "demo" / iid
        demo_dir.mkdir(parents=True)
        demo = demo_dir / "demo.hdf5"
        demo.write_bytes(b"fixture-hdf5")
        report = {"mode": "oracle", "instance_id": iid, "status": "success", "success": True,
            "qa": {"enabled": True, "hard_triggered": False, "metrics": {}}, "goal_override": None,
            "loader": {"campaign_restoration": {"scale_verified": True}},
            "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "provenance": {"isg_root": str(snapshot)}}
        c.write_json_atomic(demo_dir / "meta.json", report)
        if n == 0:
            c.write_json_atomic(gate_dir / "replay" / iid / "meta.json", {**report, "mode": "replay"})
            gate = {"source_job": str(job), "success": True, "replay_mode": "physics_action_open_loop", "snapshot": str(snapshot),
                "instance": str(tro_path), "demo": str(demo), "tro_sha256": c.file_digest(tro_path), "demo_sha256": c.file_digest(demo)}
    gate_path = gate_dir / "gate.json"
    c.write_json_atomic(gate_path, gate)
    caps = {"house_single_floor/room": 2, **{f"s/r{i}": 111 for i in range(8)}, "s/r8": 110}
    return SimpleNamespace(root=root, job=job, gate=gate_path, since=since, template=template, caps=caps, inst=inst)


def audit(f):
    return lane.audit_camera(f.job, f.gate, root=f.root, started_at=f.since, template=f.template)


def claim(f, review):
    return lane.claim_camera(review, f.root / "accepted/camera.json", root=f.root, started_at=f.since, training_room_caps=f.caps)


def test_strict_eval_quotas_training_soft_buckets_and_idempotency(tmp_path):
    f = fixture(tmp_path)
    review = audit(f)
    result = claim(f, review)
    assert result["counts"]["eval_standard"] == {"3": 3, "5": 4, "10": 3}
    assert result["counts"]["train_total"] == 2
    assert claim(f, review)["accepted"] == {}
    assert len(lane.read(f.root / "accepted/camera.json")["accepted"]) == 12


def test_wrong_code_and_bucket_fail_closed(tmp_path):
    f = fixture(tmp_path)
    path = f.inst / "camera_01.summary.json"
    c.write_json_atomic(path, {"geodesic_m": 4, "path_bucket_m": 3})
    with pytest.raises(c.ContractError, match="bucket disagrees"):
        audit(f)
    (f.root / "code/snapshot/.snapshot_commit").write_text("changed")
    with pytest.raises(c.ContractError, match="gate code"):
        audit(f)


def test_mutated_report_after_audit_and_train_room_cap_are_enforced(tmp_path):
    f = fixture(tmp_path, distances=(1, 1, 1, 1, 1, 1, 1))
    review = audit(f)
    result = claim(f, review)
    assert len(result["accepted"]) == 5 and len(result["deferred"]) == 2
    report = Path(review["members"][1]["artifacts"]["report"]["path"])
    report.write_text("{}")
    with pytest.raises(c.ContractError, match="changed after review"):
        claim(f, review)


def test_area_weighted_room_allocation_preserves_total():
    rows = [{"scene": "s", "room": f"r{i}", "traversable_m2": i+1} for i in range(10)]
    caps = lane.room_caps({"rooms_per_task": 10, "tasks": {lane.TASK: rows}})
    assert sum(caps.values()) == 1000 and len(caps) == 10
    assert caps["s/r9"] > caps["s/r0"]
    assert [lane.path_bucket(x) for x in (3, 3.00001, 5, 5.00001, 10)] == ["3", "5", "5", "10", "10"]
    for invalid in (0, -1, float("nan"), 10.01):
        with pytest.raises(c.ContractError):
            lane.path_bucket(invalid)
