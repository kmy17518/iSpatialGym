import datetime
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.campaign_20261110 import camera_lane as lane, contracts as c


def fixture(root, distances=(1, 2, 3, 3.1, 4, 4.5, 5, 6, 7, 10, 1.5, 5.5), *, obstructed=False):
    since = time.time() - 60
    snapshot = root / "code/snapshot"
    snapshot.mkdir(parents=True)
    (snapshot / ".snapshot_commit").write_text("camera-code\n")
    obj = c.ObjectRef("floor.n.01_1", "support", "floors", "abc", c.digest("asset"), semantic_kind="floor", ready=True, readiness_source="test")
    task = "camera_relocalization-obstructed" if obstructed else lane.TASK
    obstacle = c.ObjectRef("hamper.n.02_1", "target", "hamper", "abc", c.digest("obstacle"), ready=True, readiness_source="test")
    objects = (obj, obstacle) if obstructed else (obj,)
    setup = {"runtime_bddl_sha256": c.digest("bddl"), "runtime_bddl_adaptation": {
        "schema": "isg-stationary-obstacle-definition-v1", "measure": [["pose_reached", "?agent.n.01_1"]],
        "obstacle_bindings": [obstacle.instance]}} if obstructed else {}
    template = c.Template("template", task, task+"-hamper_1" if obstructed else task, "eval_standard", "house_single_floor", "room", objects, c.digest("definition"), setup)
    perturb = c.Perturbations({o.instance: c.ScaleState((1, 1, 1), (1, 1, 1)) for o in objects})
    key = c.ValidationKey(template.fingerprint, c.file_digest(snapshot / ".snapshot_commit"), c.digest("physics"), c.digest("data"), perturb.scale_fingerprint)
    job = root / "out/camera/job"
    inst = job / "inst/template/instances"
    inst.mkdir(parents=True)
    gate_dir = root / "validation/gate"
    gate_dir.mkdir(parents=True)
    for n, distance in enumerate(distances):
        iid = f"camera_{n:02d}"
        tro = c.attach_tro_metadata({"ispatialgym": {"instance_id": iid, "object_bindings": {o.instance: {"category": o.category, "model": o.model} for o in objects}}},
            campaign_id="campaign", split="eval_standard", template=template, created_at=time.time(), seed=n, perturbations=perturb, validation_key=key)
        if obstructed:
            pose = [[1, 2, 3], [0, 0, 0, 1]]
            tro["ispatialgym"].update({"bddl_sha256": c.digest("bddl"), "pose_targets": {obstacle.instance: {
                "frame": "world", "position_metric": "full", "orientation_metric": "full", "position": pose[0], "orientation": pose[1]}}})
            tro["ispatialgym"]["campaign"]["runtime"] = {"planned_start_poses": {obstacle.instance: pose}}
        tro_path = inst / f"{iid}-tro_state.json"
        c.write_json_atomic(tro_path, tro)
        c.write_json_atomic(inst / f"{iid}.oracle.json", {"tro_sha256": c.file_digest(tro_path), "path_bucket_m": float(lane.path_bucket(distance))})
        c.write_json_atomic(inst / f"{iid}.summary.json", {"geodesic_m": distance, "geodesic_blocked_m": distance, "path_bucket_m": float(lane.path_bucket(distance))})
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
    accepted=lane.read(f.root / "accepted/camera.json")["accepted"]
    assert len(accepted) == 12
    assert all(row["contract"]["class_id"]==f.template.class_id and row["source_job"]==str(f.job) for row in accepted.values())
    state=lane.read(f.root / "accepted/camera.json")
    state["accepted"]["camera_00"].pop("contract")
    c.write_json_atomic(f.root / "accepted/camera.json",state)
    assert claim(f,review)["accepted"]=={}
    assert lane.read(f.root / "accepted/camera.json")["accepted"]["camera_00"]["contract"]["class_id"]==f.template.class_id


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
