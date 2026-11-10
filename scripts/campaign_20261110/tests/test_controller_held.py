from types import SimpleNamespace

import pytest

from scripts.campaign_20261110.controller_held import job_terminal, next_work, queue_closures, record_runtime_failure


def test_pending_queues_do_not_count_as_empty_completed_attempts():
    from scripts.campaign_20261110.controller_held import completed_empty_tail

    accepted = {'status': 'complete', 'groups': {'g': {'accepted': {'member': 'train'}}}}
    empty = {'status': 'complete', 'groups': {}}
    pending = {'status': 'external_submission', 'groups': {}}
    never_started = {'status': 'complete', 'groups': {}, 'migration': {'never_started': True}}
    assert completed_empty_tail([accepted, pending, pending, pending]) == 0
    assert completed_empty_tail([empty, accepted, empty, pending, never_started, empty]) == 2
    assert completed_empty_tail([empty, pending, empty, empty]) == 3
from scripts.campaign_20261110.contracts import ContractError


def counts(**changes):
    return {"evaluation": {}, "train_buckets": {}, "class_remaining": 1000,
            "room_remaining": 191, "evaluation_sibling_reserve": 100, **changes}


def test_strict_bucket_refill_uses_deficits_and_requires_sibling_capacity():
    t = SimpleNamespace(task="mental_rotation", split="eval_standard")
    assert next_work(t, counts(), max_groups=6) == {
        "phase":"eval_standard", "bucket":1, "groups":4, "needs_new_gate":False}
    assert next_work(t, counts(evaluation={0:3,1:4,2:1}), max_groups=6)["groups"] == 2
    assert next_work(t, counts(room_remaining=0)) is None
    assert next_work(t, counts(room_remaining=1))["groups"] == 1


def test_training_reserves_room_capacity_for_other_eval_siblings_and_needs_its_own_gate():
    t = SimpleNamespace(task="object_scaling", split="eval_standard")
    completed = counts(evaluation={0:3,1:4,2:3})
    proposal = next_work(t, completed)
    assert proposal == {"phase":"train", "bucket":0, "groups":1, "needs_new_gate":True}
    assert next_work(t, completed, training_validated=True)["groups"] == 6
    assert next_work(t, counts(evaluation={0:3,1:4,2:3},room_remaining=10,evaluation_sibling_reserve=10)) is None
    assert next_work(t, counts(evaluation={0:3,1:4,2:3},class_remaining=0)) is None


def test_model_and_category_queues_never_become_training():
    model = SimpleNamespace(task="mental_rotation", split="eval_model")
    assert next_work(model, counts(evaluation={0:3,1:3,2:2}))["groups"] == 1
    assert next_work(model, counts(evaluation={0:3,1:3,2:3})) is None
    category = SimpleNamespace(task="mental_rotation", split="eval_category")
    assert next_work(category, counts(evaluation={0:3,1:4,2:3})) is None


def test_private_native_customs_and_definition_are_preserved_and_hash_checked(tmp_path):
    from scripts.campaign_20261110.controller_held import source_environment
    from scripts.campaign_20261110.contracts import file_digest
    customs=tmp_path/'customs.json';customs.write_text('{"native_alias": {}}')
    job=tmp_path/'out/native';job.mkdir(parents=True)
    (job/'job.env').write_text("START_PER_ATTEMPT=0\nKEEP_SCENE='--keep-template-scene --bddl-root /owned/bddl'\n"
        +f"ISG_CUSTOMS={customs}\nISPATIALGYM_CAMPAIGN_CUSTOMS_SHA256={file_digest(customs)}\n")
    values,template_mode=source_environment(job,root=tmp_path)
    assert template_mode and 'KEEP_SCENE=--keep-template-scene --bddl-root /owned/bddl' in values
    assert 'ISG_CUSTOMS='+str(customs) in values
    customs.write_text('{"changed_alias": {}}')
    with pytest.raises(ContractError,match='customs changed'):
        source_environment(job,root=tmp_path)


def test_training_bucket_selection_remains_a_preference():
    t = SimpleNamespace(task="mental_rotation", split="train")
    proposal = next_work(t, counts(train_buckets={0:30,1:0,2:30}), training_validated=True)
    assert proposal["phase"] == "train" and proposal["bucket"] == 1


@pytest.mark.parametrize("family", ["held", "generic"])
def test_unbucketed_soft_training_counts_remain_serializable_and_consume_capacity(tmp_path, monkeypatch, family):
    import json
    import sqlite3
    from scripts.campaign_20261110 import controller_held, controller_groups, group_lane
    from scripts.campaign_20261110.contracts import canonical_json

    path = tmp_path / "dataset/allocations.sqlite"; path.parent.mkdir()
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE members (split, task, class_id, bucket, record)")
        for bucket in (None, 0):
            record = {"class_id": "apple", "bucket": bucket, "room": "scene/room"}
            db.execute("INSERT INTO members VALUES (?,?,?,?,?)", ("train", "object_scaling", "apple", bucket, json.dumps(record)))
    template = SimpleNamespace(task="object_scaling", class_id="apple", split="train", scene="scene", room="room")
    monkeypatch.setattr(group_lane, "training_room_policy", lambda *args: {"caps": {"scene/room": 100}})
    counts = (controller_held.read_counts(tmp_path, template, [], {}) if family == "held"
              else controller_groups.read_counts(tmp_path, template))
    assert counts["train_buckets"] == {0: 1} and counts["train_unbucketed"] == 1
    assert counts["class_remaining"] == 998 and counts["evaluation_unbucketed"] == 1
    canonical_json({"counts": counts})  # Controller state publication must not crash on soft-bucket nulls.


def test_closure_uses_job_local_queue_and_rejects_wrong_event_hash(tmp_path, monkeypatch):
    import json
    from ispatialgym.sampling import campaign_goal_queue as queue
    from scripts.campaign_20261110.contracts import file_digest
    job=tmp_path/"out/job";local=job/"campaign_inputs";local.mkdir(parents=True)
    config=local/"config.json";config.write_text(json.dumps({"validation_key":{"key":"unchanged"}}))
    path=local/"queue.json";path.write_text(json.dumps({"config_path":str(config)}))
    (job/"job.env").write_text("ISPATIALGYM_CAMPAIGN_GOAL_QUEUE='/old/external/queue.json'\n")
    monkeypatch.setattr(queue,"read_queue",lambda p,c:{"entries":[{"group_id":"g"}],"sha256":file_digest(p),"config_path":str(c)})
    record={"prepared":{"group_ids":["g"]},"validation_key":{"key":"unchanged"}}
    event={"queue_sha256":file_digest(path),"event":"closed","group_id":"g","written":3}
    path.with_suffix(".events.jsonl").write_text(json.dumps(event)+"\n")
    assert queue_closures(record,{"job_dir":str(job)},tmp_path)=={"g":event}
    event["queue_sha256"]="wrong"
    path.with_suffix(".events.jsonl").write_text(json.dumps(event)+"\n")
    with pytest.raises(ContractError,match="different bytes"):
        queue_closures(record,{"job_dir":str(job)},tmp_path)


def test_vast_terminal_state_is_not_success_evidence(tmp_path):
    import json
    path=tmp_path/"vast/state/jobs/900000030.json";path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status":"assigned"}))
    assert not job_terminal({"jobid":"900000030"},tmp_path,{})
    path.write_text(json.dumps({"status":"synced","done":{"successes":0}}))
    assert job_terminal({"jobid":"900000030"},tmp_path,{})


def test_old_runtime_lock_failure_cannot_trigger_another_same_key_refill(tmp_path):
    (tmp_path/"sample.log").write_text("fcntl.flock(stream.fileno(), fcntl.LOCK_EX)\nOSError: [Errno 5] Input/output error\n")
    record={"allocation":{"job_dir":str(tmp_path)},"status":"complete","groups":{"good":{"accepted":{"iid":"train"}}}}
    state={}
    failure=record_runtime_failure(record,state)
    assert state["runtime_failure"] == failure == record["requires_new_runtime_gate"]
    assert record["groups"]["good"]["accepted"] == {"iid":"train"}


def test_pending_queues_reserve_distinct_strict_buckets_and_shared_room_capacity():
    from scripts.campaign_20261110.controller_held import reserve_pending_counts
    template=SimpleNamespace(task='object_scaling',class_id='apple',scene='s',room='r',split='eval_standard')
    pending=[{'task':'object_scaling','class_id':'apple','room':'s/r','phase':'eval_standard','bucket':1,'groups':4},
             {'task':'object_scaling','class_id':'tomato','room':'s/r','phase':'train','bucket':0,'groups':6}]
    original=counts()
    reserved=reserve_pending_counts(original,template,pending)
    assert original['evaluation']=={}
    assert reserved['evaluation']=={1:4}
    assert reserved['class_remaining']==988 and reserved['room_remaining']==161
    assert next_work(template,reserved)['bucket']==0
    assert next_work(template,reserve_pending_counts(counts(room_remaining=3),template,pending)) is None


def test_pending_completed_groups_do_not_reserve_again_and_legacy_ranges_are_exact(tmp_path):
    from scripts.campaign_20261110.controller_held import pending_held
    from scripts.campaign_20261110.contracts import write_json_atomic
    gids=['done','pending']
    state={'seeds':{'apple':{'queues':[{'status':'external_submission','phase':'eval_standard',
        'prepared':{'group_ids':gids},'groups':{'done':{'accepted':{'id':'train'}}}}]}}}
    write_json_atomic(tmp_path/'controllers/held/state.json',state)
    write_json_atomic(tmp_path/'groups/pending/campaign_config.json',{'template':{
        'task':'object_scaling','class_id':'apple','scene':'s','room':'r'}})
    write_json_atomic(tmp_path/'groups/pending/goal_plan.json',{'magnitude_range':[.1,.15]})
    row=pending_held(tmp_path)[0]
    assert row['groups']==1 and row['bucket']==1
    write_json_atomic(tmp_path/'groups/pending/goal_plan.json',{'magnitude_range':[.11,.15]})
    with pytest.raises(ContractError,match='unknown held magnitude'):
        pending_held(tmp_path)
