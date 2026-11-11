"""Index-lock contention must precede immutable goal authoring."""
from types import SimpleNamespace

import pytest

from scripts.campaign_20261110 import controller_admission as admission
from scripts.campaign_20261110 import controller_groups as groups
from scripts.campaign_20261110 import controller_held as held
from scripts.campaign_20261110 import contracts, goal_batches, held_goal_batches
from scripts.campaign_20261110 import base_regions, goal_groups, group_lane, reservation_policy


@pytest.mark.parametrize("kind", ["admission", "generic", "held"])
def test_busy_index_lock_leaves_no_authored_queue(tmp_path, monkeypatch, kind):
    module = {"admission": admission, "generic": groups, "held": held}[kind]
    snapshot = tmp_path / "code"; snapshot.mkdir()
    (snapshot / ".snapshot_commit").write_text("immutable")
    runtime = snapshot / "ispatialgym/sampling/campaign_goal_queue.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("isg-campaign-surface-pair-goal-queue-v1")
    task = "mental_rotation" if kind == "held" else "inter_object_relation_satisfaction-ontop_small_scale"
    template = SimpleNamespace(task=task, split="train", scene="scene", room="room")
    key = {"code_sha256": contracts.file_digest(snapshot / ".snapshot_commit"), "physics_sha256": "p", "data_sha256": "d"}
    cfg = {"snapshot": str(snapshot), "started_at": 1, "seed": 2, "campaign_id": "test", **key}
    monkeypatch.setattr(goal_groups, "successful_anchor", lambda *a, **k: {})
    monkeypatch.setattr(reservation_policy, "active_policy", lambda *a: ({}, {}, {}))
    monkeypatch.setattr(group_lane, "check_reservations", lambda *a: None)
    monkeypatch.setattr(base_regions, "room_base_region", lambda *a: {})
    monkeypatch.setattr(module, "read", lambda *a: {"validation_key": key, "split": "train"})
    calls = []

    def busy(root, token, count):
        calls.append((token, count))
        raise FileExistsError("family_index_reservations.lockdir")

    def forbidden_author(**kwargs):
        (tmp_path / "unrecoverable_authored_queue").write_text("would block retry")
        pytest.fail("authoring ran before index reservation")

    monkeypatch.setattr(module, "reserve_indices", busy)
    monkeypatch.setattr(goal_batches, "prepare_batch", forbidden_author)
    monkeypatch.setattr(held_goal_batches, "prepare_batch", forbidden_author)
    if kind == "admission":
        from ispatialgym.sampling import campaign_goal_queue
        monkeypatch.setattr(campaign_goal_queue, "schema_for_task", lambda task: "isg-campaign-surface-pair-goal-queue-v1")
        monkeypatch.setattr(held, "template_from", lambda value: template)
        monkeypatch.setattr(admission, "scoped_template", lambda *a: template)
        monkeypatch.setattr(admission, "source_release", lambda *a: str(tmp_path))
        monkeypatch.setattr(admission, "admission_id", lambda *a: "identity")
        monkeypatch.setattr(admission, "successful_anchor", lambda *a, **k: {})
        monkeypatch.setattr(admission, "read_env", lambda *a: {})
        monkeypatch.setattr(groups, "read_counts", lambda *a: {})
        monkeypatch.setattr(groups, "reserve_pending_counts", lambda *a: {})
        monkeypatch.setattr(groups, "pending", lambda *a: [])
        monkeypatch.setattr(admission, "admission_decision", lambda *a: {"bucket": 0})
        invoke = lambda: admission.prepare({"template": {}, "validation_key": key, "perturbations": {}},
            {"job_dir": str(tmp_path), "ledger": "ledger"}, cfg, tmp_path)
    else:
        spec = {"anchor_job": "job", "anchor_ledger": "ledger", "snapshot": str(snapshot),
                "id": "seed", "seed_config": "config", "seed_offset": 0}
        invoke = lambda: module.prepare_queue(spec, {"queues": []}, {"phase": "train", "groups": 2, "bucket": 0},
            cfg, tmp_path, template, lambda: pytest.fail("state must not change"))
    with pytest.raises(FileExistsError, match="family_index_reservations"):
        invoke()
    assert calls == [(("group_admit_identity" if kind == "admission" else
                       ("held_auto_" if kind == "held" else "group_auto_") + "seed_0001"),
                      3 if kind == "admission" else 6)]
    assert not (tmp_path / "unrecoverable_authored_queue").exists()
    assert not (tmp_path / "queues").exists()
    assert not (tmp_path / "groups").exists()
