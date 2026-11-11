import pytest
from scripts.campaign_20261110.heldout_recovery import targets
from scripts.campaign_20261110.heldout_recovery import quota_bucket
from scripts.campaign_20261110.heldout_recovery import recovery_state
from scripts.campaign_20261110.contracts import task_quotas,ContractError


def catalog():
    return {q.task+'__'+str(i):q.task for q in task_quotas() for i in range(q.standard_classes)}


def test_all_held_targets_include_zero_count_scopes_and_strict_owner_buckets():
    rows=targets(catalog())
    assert sum(r['target']for r in rows if r['split']=='eval_model')==891
    assert sum(r['target']for r in rows if r['split']=='eval_category')==109
    assert len({r['class_id']for r in rows if r['split']=='eval_model'})==99
    assert len({r['task']for r in rows if r['split']=='eval_category'})==23
    assert not any(r['split']=='eval_model'and r['task']=='camera_relocalization-standard'for r in rows)
    assert [r['target']for r in rows if r['split']=='eval_category'and r['task']=='configuration_matching-articulation_open_large_scale']==[2,1]
    assert [r['target']for r in rows if r['split']=='eval_model'and r['task']=='configuration_matching-block_design']==[2,3,2,2]


def test_incomplete_ready_manifest_cannot_silently_remove_seven_missing_classes():
    c=catalog()
    for key in list(c)[-7:]:c.pop(key)
    with pytest.raises(ContractError,match='incomplete canonical'):targets(c)


def test_native_camera_radius_labels_map_exactly_to_quota_positions():
    assert [quota_bucket('camera_relocalization-obstructed',v,True)for v in ('3','5','10')]==[0,1,2]
    assert quota_bucket('configuration_matching-attachment',0,False) is None
    with pytest.raises(ContractError):quota_bucket('camera_relocalization-standard','4',True)


def test_unknown_scheduler_state_is_not_pending_and_new_admission_is_not_old_failure():
    assert recovery_state(3,[{'worker_status':'scheduler_state_not_observed'}],{'status':'external_submission'})=='awaiting_observation'
    assert recovery_state(3,[{'worker_status':'RUNNING'}],{'status':'external_submission'})=='running'
    assert recovery_state(3,[{'worker_status':'PENDING'}],{'status':'external_submission'})=='queued'
    assert recovery_state(6,[],{'status':'admitted'})=='ready_for_scoped_refill'
    assert recovery_state(3,[],{'status':'external_submission'})=='awaiting_acceptance'
    assert recovery_state(0,[],{'status':'failed'})=='complete'
