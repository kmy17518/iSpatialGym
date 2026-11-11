import json
from types import SimpleNamespace

from scripts.campaign_20261110 import capacity_probe, monitor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def member(split, template):
    return {"split": split, "validation_key": {"template_sha256": template}}


def fixture(root):
    dataset = root / "dataset"
    write(dataset / "groups/standard.json", {"members": [member("eval_standard", "standard"), member("train", "standard")]})
    write(dataset / "groups/model.json", {"members": [member("eval_model", "model")]})
    write(dataset / "groups/category.json", {"members": [member("eval_category", "category")]})
    write(dataset / "group_counts.json", {"campaign_id": "campaign_20261110", "groups": 3,
        "counts": {"eval_standard": 1, "eval_model": 1, "eval_category": 1, "train": 1}})
    for name, rows in {
        "camera_quota.json": {"s": member("eval_standard", "camera"), "t1": member("train", "camera"), "t2": member("train", "camera")},
        "camera_obstructed_quota.json": {"m": member("eval_model", "obstructed"), "c": member("eval_category", "obstructed"), "t3": member("train", "obstructed")},
        "camera_category_quota.json": {"r1": member("eval_category", "room"), "r2": member("eval_category", "room")},
    }.items():
        write(dataset / name, {"campaign_id": "campaign_20261110", "accepted": rows})
    return {"eval_standard": 2, "eval_model": 2, "eval_category": 4, "train": 4}


def test_live_artifacts_include_all_three_authoritative_camera_ledgers(tmp_path, monkeypatch):
    expected = fixture(tmp_path)
    accepted, templates = capacity_probe.acceptance_observation(tmp_path)
    assert accepted["counts"] == expected and accepted["errors"] == []
    assert accepted["templates"] == len(templates) == 6
    assert {p["path"].split("/")[-1] for p in accepted["sources"]} == {
        "group_counts.json", "camera_quota.json", "camera_obstructed_quota.json", "camera_category_quota.json"}
    observation = {"checked_at": "2026-11-11T01:20:00+00:00", "acceptance": accepted, "jobs": []}
    monkeypatch.setattr(monitor.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=json.dumps(observation)))
    state = monitor.refresh(tmp_path, "local")
    assert state["acceptance_evidence"]["counts"] == expected
    assert state["campaign_results"]["fresh_category_holdout_instances"] == 4
    assert json.loads((tmp_path / "live.json").read_text())["acceptance_evidence"]["counts"] == expected
    assert "fresh category holdout instances | 4" in (tmp_path / "LIVE.md").read_text()


def test_duplicate_camera_source_is_rejected_without_partial_count(tmp_path):
    fixture(tmp_path)
    write(tmp_path / "dataset/camera_category_quota.json", {"campaign_id": "campaign_20261110",
        "accepted": {"new": member("eval_category", "new"), "s": member("eval_category", "duplicate")}})
    accepted, _ = capacity_probe.acceptance_observation(tmp_path)
    assert accepted["counts"]["eval_category"] == 2
    assert accepted["errors"] == ["camera instance appears in two quota states"]
    assert not any(p["path"].endswith("camera_category_quota.json") for p in accepted["sources"])


def test_unverified_group_export_never_contributes_a_partial_total(tmp_path):
    fixture(tmp_path)
    write(tmp_path / "dataset/group_counts.json", {"campaign_id": "campaign_20261110", "groups": 99, "counts": {"train": 900}})
    accepted, _ = capacity_probe.acceptance_observation(tmp_path)
    assert accepted["counts"]["train"] == 3
    assert accepted["errors"] == ["group export/count mismatch"]
