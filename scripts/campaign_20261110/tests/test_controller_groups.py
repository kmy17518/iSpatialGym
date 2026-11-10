from types import SimpleNamespace

from scripts.campaign_20261110.controller_groups import next_work, reserve_pending_counts, ranges


def counts(**kwargs):
    return {"evaluation": {}, "train_buckets": {}, "class_remaining": 1000, "task_remaining": 4000, **kwargs}


def test_unbucketed_pair_eval_continues_to_training_with_new_phase_gate():
    template = SimpleNamespace(task="inter_object_relation_satisfaction-ontop_small_scale", split="eval_standard")
    assert next_work(template, counts())["groups"] == 6
    assert next_work(template, counts(evaluation={0: 10})) == {
        "phase": "train", "bucket": 0, "groups": 1, "needs_new_gate": True}
    assert next_work(template, counts(evaluation={0: 10}), training_validated=True)["groups"] == 6
    assert next_work(template, counts(), mode="train_only")["phase"] == "train"


def test_holdouts_never_spill_into_training_and_count_three_members_per_group():
    model = SimpleNamespace(task="inter_object_relation_satisfaction-under_small_scale", split="eval_model")
    assert next_work(model, counts())["groups"] == 3
    assert next_work(model, counts(evaluation={0: 9})) is None
    assert next_work(model, counts(), mode="train_only") is None
    category = SimpleNamespace(task=model.task, split="eval_category")
    assert next_work(category, counts())["groups"] == 2
    assert next_work(category, counts(evaluation={0: 4})) is None


def test_strict_joint_buckets_and_soft_training_use_their_own_weights():
    template = SimpleNamespace(task="configuration_matching-articulation_close_small_scale", split="eval_standard")
    assert next_work(template, counts(evaluation={0: 5, 1: 2}))["bucket"] == 1
    result = next_work(template, counts(evaluation={0: 5, 1: 5}, train_buckets={0: 100, 1: 0}), training_validated=True)
    assert result["bucket"] == 1 and result["phase"] == "train"
    assert next_work(template, counts(class_remaining=0), mode="train_only") is None
    assert ranges(template) == ((0, .5), (.5, 1))


def test_pending_generic_groups_reserve_global_task_and_class_capacity():
    template = SimpleNamespace(task="disembedding", class_id="class", split="eval_standard")
    pending = [{"task": "disembedding", "class_id": "class", "phase": "eval_standard", "bucket": 1, "groups": 4},
               {"task": "disembedding", "class_id": "another", "phase": "train", "bucket": 0, "groups": 6}]
    result = reserve_pending_counts(counts(), template, pending)
    assert result["evaluation"] == {1: 4}
    assert result["class_remaining"] == 988 and result["task_remaining"] == 3970
    assert next_work(template, result)["bucket"] == 0


def test_training_cannot_consume_unfinished_standard_evaluation_siblings():
    template = SimpleNamespace(task="inter_object_relation_satisfaction-ontop_small_scale", split="eval_standard")
    assert next_work(template, counts(class_remaining=4, evaluation_sibling_reserve=2),
                     training_validated=True, mode="train_only") is None
    result = next_work(template, counts(class_remaining=5, evaluation_sibling_reserve=2),
                       training_validated=True, mode="train_only")
    assert result["groups"] == 1  # Three candidates still leave two required siblings.
    assert next_work(template, counts(class_remaining=1, evaluation_sibling_reserve=0),
                     training_validated=True, mode="train_only")["groups"] == 1


def test_fresh_alignment_anchor_confirms_frozen_policy_without_rekeying(tmp_path):
    from dataclasses import replace
    import pytest
    from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests
    from scripts.campaign_20261110.goal_groups import with_alignment_buckets
    from scripts.campaign_20261110.contracts import ContractError
    _, _, _, original, _ = GoalGroupControllerTests().fixture(tmp_path)
    template = replace(original, task="alignment-world_vertical")
    first = with_alignment_buckets(template, [15, 25], source={"path": "first", "field": "range", "sha256": "a" * 64})
    confirmed = with_alignment_buckets(first, [15, 25], source={"path": "fresh", "field": "range", "sha256": "b" * 64})
    assert confirmed is first and confirmed.fingerprint == first.fingerprint
    with pytest.raises(ContractError, match="policy changed"):
        with_alignment_buckets(first, [10, 25], source={"field": "range", "sha256": "c" * 64})
