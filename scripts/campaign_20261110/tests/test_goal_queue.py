import json
import time
from types import SimpleNamespace

import pytest

from ispatialgym.sampling import campaign_goal_queue as q
from scripts.campaign_20261110.contracts import ContractError, file_digest, write_json_atomic


def fixture(tmp_path, monkeypatch):
    config = {"campaign_id": "new", "split": "eval_standard", "template": {"task": "object_scaling", "scene": "s", "room": "r",
        "scene_setup": {"runtime_template_id": "t"}}, "validation_key": {"template_sha256": "a"*64},
        "perturbations": {"lighting": {}, "removed_objects": [], "scales": {}}}
    config_path = tmp_path / "config.json"
    write_json_atomic(config_path, config)
    entries = []
    for n in range(2):
        plan = {"schema": "isg-held-goal-plan-v1", "campaign_id": "new", "split": "eval_standard", "task": "object_scaling",
            "template_id": "t", "scene": "s", "room": "r", "contract_template_sha256": "a"*64, "group_id": f"g{n}",
            "goal_pose": {"position": [n, 0, 1], "orientation": [0, 0, 0, 1]}, "base_pose2d": [0, 0, 0],
            "magnitude_range": [.05, .1], "max_members": 3, "created_at": time.time()-5, "budget_seconds": 600}
        path = tmp_path / f"plan{n}.json"
        write_json_atomic(path, plan)
        entries.append({"path": str(path), "sha256": file_digest(path)})
    queue = tmp_path / "queue.json"
    write_json_atomic(queue, {"schema": q.SCHEMA, "campaign_id": "new", "config_sha256": file_digest(config_path), "entries": entries})
    monkeypatch.setenv(q.ENV, str(queue))
    monkeypatch.setenv("ISPATIALGYM_CAMPAIGN_CONFIG", str(config_path))
    monkeypatch.setenv(q.PLAN_ENV, "")
    return queue, config_path, entries


def test_queue_resets_only_group_state_and_closes_on_last_publication(tmp_path, monkeypatch):
    queue, config, entries = fixture(tmp_path, monkeypatch)
    sampler = SimpleNamespace(_campaign_goal_started_at=1, _campaign_goal_plan_sha="old", _campaign_group_written=3, native_cache=object())
    cache = sampler.native_cache
    assert q.activate_initial(sampler)
    assert not hasattr(sampler, "_campaign_goal_started_at") and sampler.native_cache is cache
    sampler._campaign_group_written = 3
    assert q.after_written(sampler) is False
    assert sampler._campaign_goal_queue["index"] == 1
    assert not hasattr(sampler, "_campaign_group_written")
    sampler._campaign_group_written = 3
    assert q.after_written(sampler) is True
    events = [json.loads(line) for line in queue.with_suffix(".events.jsonl").read_text().splitlines()]
    assert [event["group_id"] for event in events if event["event"] == "closed"] == ["g0", "g1"]


def test_changed_lighting_or_plan_cannot_cross_queue_transition(tmp_path, monkeypatch):
    _, config, _ = fixture(tmp_path, monkeypatch)
    sampler = SimpleNamespace()
    q.activate_initial(sampler)
    value = json.loads(config.read_text())
    value["perturbations"]["lighting"] = {"intensity": 2}
    write_json_atomic(config, value)
    with pytest.raises(ContractError, match="config changed"):
        q.advance(sampler, "campaign_group_timeout")


def test_quaternion_sign_duplicate_goal_rejected(tmp_path, monkeypatch):
    queue, config, entries = fixture(tmp_path, monkeypatch)
    first = json.loads(open(entries[0]["path"]).read())
    second = json.loads(open(entries[1]["path"]).read())
    second["goal_pose"] = {"position": first["goal_pose"]["position"], "orientation": [0, 0, 0, -1]}
    write_json_atomic(entries[1]["path"], second)
    entries[1]["sha256"] = file_digest(entries[1]["path"])
    raw = json.loads(queue.read_text())
    raw["entries"] = entries
    write_json_atomic(queue, raw)
    with pytest.raises(ContractError, match="duplicate canonical"):
        q.read_queue(queue, config)


def test_queue_disabled_never_reads_sampler_context(monkeypatch):
    monkeypatch.delenv(q.ENV, raising=False)
    assert q.activate_initial(SimpleNamespace()) is False
    assert q.advance(SimpleNamespace(), "unused") is False


def test_job_end_closes_partial_group_without_claiming_unstarted_groups(tmp_path, monkeypatch):
    queue, _, _ = fixture(tmp_path, monkeypatch)
    sampler = SimpleNamespace()
    q.activate_initial(sampler)
    sampler._campaign_group_written = 1
    q.after_written(sampler)
    q.finalize(sampler)
    q.finalize(sampler)
    closed = [json.loads(line) for line in queue.with_suffix(".events.jsonl").read_text().splitlines() if json.loads(line)["event"] == "closed"]
    assert len(closed) == 1 and closed[0]["group_id"] == "g0"
    assert closed[0]["reason"] == "job_end" and closed[0]["written"] == 1 and closed[0]["unstarted_groups"] == 1
