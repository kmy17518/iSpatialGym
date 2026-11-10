import json
from types import SimpleNamespace

import pytest

from ispatialgym.sampling import campaign_goal_queue as q
from scripts.campaign_20261110.contracts import ContractError, file_digest, write_json_atomic
from scripts.campaign_20261110.tests.test_goal_queue import fixture


def generic_fixture(tmp_path, monkeypatch, task):
    manifest, config_path, entries = fixture(tmp_path, monkeypatch)
    config = json.loads(config_path.read_text())
    config["template"]["task"] = task
    config["template"]["objects"] = [{"instance": "target", "role": "target", "joint": "hinge"}]
    if task == "configuration_matching-attachment":
        config["template"]["objects"].append({"instance": "partner", "role": "partner"})
    write_json_atomic(config_path, config)
    for row in entries:
        plan = json.loads(open(row["path"]).read())
        plan["task"] = task
        if task.startswith("alignment-"):
            plan.update(schema="isg-surface-goal-plan-v1", magnitude_range_deg=[15, 20])
        elif task == "disembedding":
            plan.update(schema="isg-tray-goal-plan-v1", target_name="target", source_name="source", destination_name="destination",
                source_pose={"position": [0, 0, .8], "orientation": [0, 0, 0, 1]},
                destination_pose={"position": [1, 0, .8], "orientation": [0, 0, 0, 1]}, crowdedness_range=[.3, .45])
        elif task == "configuration_matching-attachment":
            plan.update(schema="isg-attachment-goal-plan-v1", target_name="target", partner_name="partner",
                partner_pose={"position": [1, 0, .8], "orientation": [0, 0, 0, 1]})
        else:
            plan.update(schema="isg-joint-goal-plan-v1", target="target", joint="hinge", goal_joints={"hinge": 0}, fraction_range=[0, .5])
        write_json_atomic(row["path"], plan)
        row["sha256"] = file_digest(row["path"])
    write_json_atomic(manifest, {"schema": q.schema_for_task(task), "campaign_id": "new",
        "config_sha256": file_digest(config_path), "entries": entries})
    return manifest, config_path, entries


@pytest.mark.parametrize("task", ["alignment-world_vertical", "alignment-axial", "disembedding",
    "configuration_matching-articulation_close_small_scale", "configuration_matching-articulation_open_large_scale",
    "configuration_matching-attachment"])
def test_family_queue_clears_only_group_state_and_preserves_native_caches(tmp_path, monkeypatch, task):
    manifest, config, _ = generic_fixture(tmp_path, monkeypatch, task)
    sampler = SimpleNamespace(native_cache=object(), _pairs=[{"q_init": .4, "q_goal": 0}],
        _campaign_surface_active="old", _campaign_tray_active="old", _campaign_joint_active="old",
        _campaign_joint_written_starts={"old-start"}, _campaign_attachment_active="old", _campaign_attachment_plan_sha="old")
    cache = sampler.native_cache
    assert q.activate_initial(sampler)
    assert not hasattr(sampler, "_campaign_joint_written_starts")
    assert not hasattr(sampler, "_campaign_attachment_active") and not hasattr(sampler, "_campaign_attachment_plan_sha")
    sampler._campaign_group_written = 3
    sampler._campaign_joint_plan_sha = "first"
    sampler._campaign_joint_written_starts = {"fresh-start"}
    sampler._pairs = []
    assert q.after_written(sampler) is False
    assert not hasattr(sampler, "_campaign_joint_plan_sha") and not hasattr(sampler, "_campaign_joint_written_starts")
    assert sampler.native_cache is cache
    if task.endswith("_small_scale"):
        assert sampler._pairs == [{"q_init": .4, "q_goal": 0}] and sampler._pair_drawn is False
    assert len(q.read_queue(manifest, config)["entries"]) == 2


def test_attachment_queue_requires_both_actual_role_bindings(tmp_path, monkeypatch):
    manifest, config, entries = generic_fixture(tmp_path, monkeypatch, "configuration_matching-attachment")
    raw = json.loads(open(entries[0]["path"]).read())
    raw["partner_name"] = "unbound_partner"
    write_json_atomic(entries[0]["path"], raw)
    queue = json.loads(manifest.read_text())
    queue["entries"][0]["sha256"] = file_digest(entries[0]["path"])
    write_json_atomic(manifest, queue)
    with pytest.raises(ContractError, match="declared binding"):
        q.read_queue(manifest, config)


def test_joint_queue_rejects_other_bound_joint_and_task(tmp_path, monkeypatch):
    manifest, config, entries = generic_fixture(tmp_path, monkeypatch, "configuration_matching-articulation_close_small_scale")
    raw = json.loads(open(entries[0]["path"]).read())
    raw["joint"] = "another_hinge"
    write_json_atomic(entries[0]["path"], raw)
    queue = json.loads(manifest.read_text())
    queue["entries"][0]["sha256"] = file_digest(entries[0]["path"])
    write_json_atomic(manifest, queue)
    with pytest.raises(ValueError, match="role/joint mismatch"):
        q.read_queue(manifest, config)


def test_surface_queue_cannot_cross_task_on_shared_schema(tmp_path, monkeypatch):
    manifest, config, entries = generic_fixture(tmp_path, monkeypatch, "alignment-axial")
    raw = json.loads(open(entries[0]["path"]).read())
    raw["task"] = "alignment-world_vertical"
    write_json_atomic(entries[0]["path"], raw)
    queue = json.loads(manifest.read_text())
    queue["entries"][0]["sha256"] = file_digest(entries[0]["path"])
    write_json_atomic(manifest, queue)
    with pytest.raises(ContractError, match="task differs"):
        q.read_queue(manifest, config)


def test_batch_author_keeps_one_exact_contract_and_fresh_three_member_groups(tmp_path):
    import time
    from dataclasses import replace
    from scripts.campaign_20261110 import goal_batches as batches
    from scripts.campaign_20261110.contracts import digest
    from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests
    job, ledger, _, template, code = GoalGroupControllerTests().fixture(tmp_path)
    template = replace(template, task="alignment-axial")
    anchor = batches.g.successful_anchor(job, campaign_root=tmp_path, ledger=ledger, campaign_started_at=time.time()-10)
    anchor["template_meta"]["template_facts"] = {"deviation_range_applied_deg": [20, 70]}
    result = batches.prepare_batch(anchor=anchor, template=template, campaign_root=tmp_path, batch_id="axial_train",
        count=3, magnitude_range=[20, 30], seed=42, campaign_id="new", code_sha256=file_digest(code/".snapshot_commit"),
        physics_sha256=digest("physics"), data_sha256=digest("data"))
    state = q.read_queue(result["queue"], result["config"])
    assert len(state["entries"]) == 3 and result["candidate_instances"] == 9
    assert len({entry["goal_sha256"] for entry in state["entries"]}) == 1
    for entry in state["entries"]:
        plan = json.loads(open(entry["path"]).read())
        assert plan["max_members"] == 3 and plan["source_instance_is_member"] is False
    assert not list((tmp_path/"groups").rglob("*-tro_state.json"))
