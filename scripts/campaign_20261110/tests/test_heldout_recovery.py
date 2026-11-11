import pytest
from scripts.campaign_20261110.heldout_recovery import targets
from scripts.campaign_20261110.heldout_recovery import quota_bucket
from scripts.campaign_20261110.heldout_recovery import recovery_state
from scripts.campaign_20261110.heldout_recovery import scheduler_states
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


def test_completed_native_geometry_remains_distinct_from_demo_failure():
    assert recovery_state(9,[],{'status':'native_passed','reason':'Own demo and replay required'})=='awaiting_validation'
    assert recovery_state(9,[],{'status':'native_preparation','reason':'Missing brace candidates'})=='engineering_blocked'
    assert recovery_state(9,[{'worker_status':'RUNNING'}],{'status':'native_passed'})=='running'


def test_array_tasks_keep_authored_identity_when_pending_running_and_completed(monkeypatch):
    from types import SimpleNamespace
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        if command[0]=='squeue':
            assert '--array' in command
            return SimpleNamespace(stdout='18440875_1|RUNNING|svl17\n18440875_2|PENDING|Priority\n')
        assert command[-1]=='JobID,State,Reason,ExitCode'
        return SimpleNamespace(stdout='18440875_0|COMPLETED|None|0:0\n18440875_0.batch|COMPLETED|None|0:0\n')
    monkeypatch.setattr('scripts.campaign_20261110.heldout_recovery.subprocess.run',run)
    states,errors=scheduler_states({'18440875_0','18440875_1','18440875_2'})
    assert errors==[] and len(calls)==2
    assert set(states)=={'18440875_0','18440875_1','18440875_2'}
    assert states['18440875_0']['worker_status']=='COMPLETED'
    assert states['18440875_1']['worker_status']=='RUNNING'
    assert states['18440875_2']['pending_reason']=='Priority'


def test_vast_assignment_requires_fresh_exact_execution_heartbeat():
    from scripts.campaign_20261110.heldout_recovery import vast_execution
    import copy
    state={'status':'assigned','instance':'42','tries':1}
    mirror={'schema':'isg-vast-worker-heartbeat-mirror-v1','campaign_id':'campaign_20261110',
        'instance_id':'42','observed_at_unix':995,'heartbeat':{'time':990,
        'running':{'900001':{'phase':'run','runtime_started_unix':980}}}}
    assert vast_execution('900001',state,now=1000)['worker_status']=='assigned_unverified'
    good=vast_execution('900001',state,mirror,1000)
    assert good['worker_status']=='running'and good['execution_verified']
    assert good['worker_heartbeat_at']!=good['heartbeat_observed_at']
    for key,value in [('time',800),('running',{})]:
        bad=copy.deepcopy(mirror);bad['heartbeat'][key]=value
        assert not vast_execution('900001',state,bad,1000)['execution_verified']
    bad=copy.deepcopy(mirror);bad['instance_id']='43'
    assert not vast_execution('900001',state,bad,1000)['execution_verified']
    assert vast_execution('900002',state,mirror,1000)['worker_status']=='assigned_unverified'
    assert vast_execution('900001',dict(state,status='synced'),mirror,1000)['worker_status']=='synced'


@pytest.mark.parametrize('phase,status',[('fetch','fetching'),('upload','uploading')])
def test_vast_fetch_upload_and_assignment_are_not_running_gpus(phase,status):
    from scripts.campaign_20261110.heldout_recovery import vast_execution
    state={'status':'assigned','instance':'42'}
    mirror={'schema':'isg-vast-worker-heartbeat-mirror-v1','campaign_id':'campaign_20261110',
        'instance_id':'42','observed_at_unix':995,'heartbeat':{'time':990,
        'running':{'1':{'phase':phase,'runtime_started_unix':980}}}}
    row=vast_execution('1',state,mirror,1000)
    assert row['worker_status']==status and not row['execution_verified']
    assert recovery_state(1,[row],{})=='awaiting_observation'
    assert recovery_state(1,[{'worker_status':'assigned'}],{})=='awaiting_observation'
