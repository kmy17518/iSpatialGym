from scripts.campaign_20261110.reconcile_vast_ready import intake_slots, activation_candidates


def metrics(instances=()):
    return {'account_listing_complete':True,'observed_at_unix':100,'instances':list(instances)}


def test_idle_backlog_intake_ignores_stale_ready_counters_and_exited_providers():
    value=metrics([{'actual_status':'exited','num_gpus':10}]);value['worker_ready_gpus']=10
    assert intake_slots(value,[],timestamp=101)==128
    assert intake_slots(value,[{'status':'queued'}]*120,timestamp=101)==8
    assert intake_slots(value,[{'status':'assigned'}]*120,timestamp=101)==128


def test_intake_expands_with_real_fleet_without_fixed_hundred_gpu_ceiling():
    value=metrics([{'actual_status':'running','num_gpus':200}])
    assert intake_slots(value,[{'status':'queued'}]*100,timestamp=101)==500


def test_stale_or_missing_provider_observation_cannot_claim_more_work():
    assert intake_slots(metrics(),[],timestamp=281)==0
    value=metrics();value['account_listing_complete']=False
    assert intake_slots(value,[],timestamp=101)==0


def test_activation_does_not_release_an_explicit_hold_or_duplicate_active_job():
    submitted=[{'job_id':str(i)} for i in range(4)]
    assert activation_candidates(submitted,['1'],['0','1'])==[{'job_id':'2'},{'job_id':'3'}]


def test_authoritative_terminal_state_overrides_historical_queue_record(tmp_path):
    import json
    from scripts.campaign_20261110.reconcile_vast_ready import active_job_states
    (tmp_path/'vast/state/jobs').mkdir(parents=True);(tmp_path/'vast/queue').mkdir()
    for jid in ['1','2','3']:(tmp_path/'vast/queue'/(jid+'.json')).write_text('{}')
    (tmp_path/'vast/state/jobs/1.json').write_text(json.dumps({'id':'1','status':'synced'}))
    (tmp_path/'vast/state/jobs/2.json').write_text(json.dumps({'id':'2','status':'queued'}))
    states=active_job_states(tmp_path,['1','2','3'])
    assert [s['status']for s in states]==['synced','queued','queued']
    assert intake_slots(metrics(),states,timestamp=101)==126
