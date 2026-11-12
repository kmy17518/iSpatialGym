import pytest
from scripts.campaign_20261110 import training_pipeline_status as status
from scripts.campaign_20261110.contracts import write_json_atomic
import copy


def request(root, name, start):
    folder = root/'training_pipeline_canaries'/name
    row = {'candidate_id': name, 'start': start, 'count': 1, 'class_id': 'class',
           'config_sha256': 'a'*64, 'native_runtime': '/native', 'output': str(folder/'native')}
    write_json_atomic(folder/'requests.json', {'campaign_id': 'campaign_20261110', 'records': [row]})
    return folder, row


def test_failed_and_pending_attempts_never_become_credit(tmp_path):
    folder, row = request(tmp_path, 'failed', 1)
    request(tmp_path, 'pending', 2)
    write_json_atomic(folder/'native/sampling_gate.json', {**row, 'returncode': 0, 'tro_count': 0})
    value = status.observe(tmp_path)
    assert value['attempts_observed'] == 2 and value['counts'] == {'native_failed': 1, 'native_not_observed': 1}
    assert value['benchmark_credit'] == value['quota_consumed'] == 0
    assert not (tmp_path/'dataset').exists()


def test_instance_refresh_publishes_before_slow_batch_and_preserves_prior_dates(tmp_path, monkeypatch):
    request(tmp_path, 'fresh', 12)
    folder=tmp_path/'training_pipeline_canaries/batch/batch_restoration_r1'
    write_json_atomic(folder/'request.json', {'canaries': []})
    write_json_atomic(folder/'physics/probe.json', {'entries': [{}, {}, {}]})
    prior={'request': status.ref(folder/'request.json'), 'status': 'failed_validation',
           'tasks': [], 'observed_at': '2026-11-11T00:00:00Z',
           'stages': [{'kind': 'physics_action_open_loop', 'status': 'failed_validation',
                       'verification_observed_at': '2026-11-11T00:01:00Z'}]}
    published=[]
    def verify(*args):
        assert published, 'fresh records must be visible before expensive verification'
        first=published[0]
        assert first['records'][0]['start']==12 and first['cycle_finished_at'] is None
        assert first['batch_stages'][0]['observation_current_cycle'] is False
        assert first['batch_stages'][0]['stages'][0]['verification_observed_at']=='2026-11-11T00:01:00Z'
        return {'status': 'failed_validation', 'error': 'unchanged strict failure'}
    monkeypatch.setattr(status,'observed_batch_verification',verify)
    final=status.observe(tmp_path,publish=lambda value:published.append(copy.deepcopy(value)),previous_batches=[prior],batch_verifier=verify)
    assert final['cycle_phase']=='complete' and final['cycle_finished_at'] is not None
    assert final['batch_stages'][0]['observation_current_cycle'] is True
    assert final['batch_stages'][0]['status']=='failed_validation'
    assert final['benchmark_credit']==final['quota_consumed']==0


def test_review_must_match_original_native_index(tmp_path):
    folder, row = request(tmp_path, 'candidate', 1)
    write_json_atomic(folder/'existing_native_reviews.json', {'campaign_id': 'campaign_20261110',
        'observed_at': '2026-11-11T00:00:00Z', 'records': {'candidate': {
            'allocation': {'start': '2'}, 'status': 'validated_canary_exported'}}})
    value = status.observe(tmp_path)
    assert value['records'][0]['review'] is None
    assert value['records'][0]['status'] == 'native_not_observed'


def test_native_submission_binds_request_index_and_runtime_without_success(tmp_path):
    folder,row=request(tmp_path,'pool',10)
    submitted={'campaign_id':'campaign_20261110','request':status.ref(folder/'requests.json'),
        'start':10,'native_runtime':'/native','job_id':'123','returncode':0}
    write_json_atomic(folder/'submission.json',submitted)
    value=status.observe(tmp_path)
    assert value['records'][0]['status']=='native_submitted'
    assert value['records'][0]['native'] is None
    assert value['records'][0]['native_submission']['scheduler_state']=='unavailable'
    assert value['benchmark_credit']==value['quota_consumed']==0
    write_json_atomic(folder/'submission.json',{**submitted,'start':11})
    invalid=status.observe(tmp_path)
    assert invalid['records'][0]['status']=='evidence_identity_failed'
    assert 'exact request/runtime/index' in invalid['errors'][0]['error']


def test_batch_request_is_not_batch_success(tmp_path):
    path = tmp_path/'training_pipeline_canaries/batch/batch_restoration_r1'
    write_json_atomic(path/'request.json', {'canaries': []})
    write_json_atomic(path/'preflight_failure.json', {'reason': 'incompatible scene'})
    value = status.observe(tmp_path)
    assert value['batch_stages'][0]['status'] == 'preflight_failed'
    assert all(s['status'] == 'not_observed' for s in value['batch_stages'][0]['stages'])


def test_submission_does_not_imply_running_or_success(tmp_path):
    path = tmp_path/'training_pipeline_canaries/batch/batch_restoration_r2'
    write_json_atomic(path/'request.json', {'canaries': []})
    write_json_atomic(path/'submission.json', {'request': status.ref(path/'request.json'),
        'jobs': [{'kind': 'physics_action_open_loop', 'job_id': '123'}]})
    value = status.observe(tmp_path)['batch_stages'][0]['stages']
    assert value[0]['status'] == 'submitted' and value[0]['scheduler_state'] == 'unavailable'
    assert value[1]['status'] == 'not_observed'


def test_authorization_refresh_preserves_one_native_attempt(tmp_path):
    folder, row = request(tmp_path, 'candidate', 1)
    original = status.ref(folder/'requests.json')
    current = folder/'requests_refresh_r2.json'
    write_json_atomic(current, {'campaign_id': 'campaign_20261110',
        'records': [{**row, 'authorization_sha256': 'b'*64}]})
    write_json_atomic(folder/'current_request.json', {'previous': original, 'current': status.ref(current)})
    value = status.observe(tmp_path)
    assert value['attempts_observed'] == 1 and value['records'][0]['original_request'] == original
    changed = status.read(current); changed['records'][0]['start'] = 2
    write_json_atomic(current, changed)
    write_json_atomic(folder/'current_request.json', {'previous': original, 'current': status.ref(current)})
    value = status.observe(tmp_path)
    assert value['attempts_observed'] == 0 and 'changed native attempt identity' in value['errors'][0]['error']


def test_family_evidence_does_not_validate_unrelated_paths():
    rows=[{'task':'disembedding','configuration_sha256':'a','native':{'written':1},
           'validated_evidence':{'review':{'path':'/review','sha256':'x'}}}]
    families={r['task']:r for r in status.family_paths(rows,[])}
    assert len(families)==23
    assert families['disembedding']['exact_validation']['status']=='partial'
    assert families['configuration_matching-attachment']['exact_validation']['status']=='untested'
    assert all(r['model_pool_switch_active_only_export']['status']=='untested' for r in families.values())
    assert sum(r['standard_classes'] for r in families.values())==100


def test_terminal_allocation_does_not_stay_submitted_or_imply_probe_success(tmp_path):
    path=tmp_path/'training_pipeline_canaries/batch/batch_restoration_r2'
    write_json_atomic(path/'request.json',{'canaries':[]})
    write_json_atomic(path/'submission.json',{'request':status.ref(path/'request.json'),
        'jobs':[{'kind':'physics_action_open_loop','job_id':'123'},
                {'kind':'kinematic_observation_rendering','job_id':'124'}]})
    write_json_atomic(path/'execution_observation.json',{
        'schema':'isg-batch-execution-observation-v1','request':status.ref(path/'request.json'),
        'observed_at':'2026-11-11T16:00:00Z','jobs':[
            {'job_id':'123','state':'FAILED','exit_code':'1:0'},
            {'job_id':'124','state':'COMPLETED','exit_code':'0:0'}]})
    value=status.observe(tmp_path)['batch_stages'][0]
    assert value['stages'][0]['status']=='failed_execution'
    assert value['stages'][1]['status']=='completed_without_complete_probe'
    assert value['status']!='batch_restoration_verified'
    assert value['benchmark_credit']==0


def test_reporting_cache_keeps_audit_time_and_invalidates_changed_nested_artifacts(tmp_path,monkeypatch):
    from scripts.campaign_20261110 import training_restoration
    artifact=tmp_path/'demo.hdf5';artifact.write_bytes(b'original')
    report=tmp_path/'report.json';write_json_atomic(report,{'demo':status.ref(artifact)})
    probe=tmp_path/'probe.json';write_json_atomic(probe,{'report':status.ref(report)})
    calls=[]
    def verify(*args):
        calls.append(args)
        if artifact.read_bytes()!=b'original':raise ValueError('changed native artifact')
    monkeypatch.setattr(training_restoration,'verify_batch',verify)
    first=status.observed_batch_verification(tmp_path,status.ref(probe),'physics_action_open_loop')
    cached=status.observed_batch_verification(tmp_path,status.ref(probe),'physics_action_open_loop')
    assert len(calls)==1 and first['status']==cached['status']=='verified'
    assert not first['verification_cached'] and cached['verification_cached']
    assert first['verification_observed_at']==cached['verification_observed_at']
    artifact.write_bytes(b'changed')
    failed=status.observed_batch_verification(tmp_path,status.ref(probe),'physics_action_open_loop')
    assert len(calls)==2 and failed['status']=='failed_validation'
    assert 'changed native artifact' in failed['error']
    assert status.observed_batch_verification(tmp_path,status.ref(probe),'physics_action_open_loop')['status']=='failed_validation'
    assert len(calls)==2


def test_repeated_native_tree_references_are_walked_once(tmp_path,monkeypatch):
    from pathlib import Path
    native=tmp_path/'code/trunk_example'
    for folder in ('ispatialgym','scripts'):
        (native/folder).mkdir(parents=True)
        (native/folder/'runtime.py').write_text('original')
    (native/'.snapshot_commit').write_text('exact')
    probe=tmp_path/'probe.json'
    write_json_atomic(probe,{'repeated':[str(native)]*30})
    calls=[];original=Path.rglob
    def counted(path,pattern):
        calls.append(path)
        return original(path,pattern)
    monkeypatch.setattr(Path,'rglob',counted)
    stamps=status._evidence_stamps(tmp_path,probe)
    assert len(calls)==2
    assert str(native/'ispatialgym/runtime.py') in stamps
    assert str(native/'scripts/runtime.py') in stamps


def test_slow_batch_worker_returns_pending_without_blocking_instance_polling(tmp_path,monkeypatch):
    import threading
    started=threading.Event();finish=threading.Event();calls=[]
    def slow(*args):
        calls.append(args);started.set()
        assert finish.wait(3)
        return {'status':'failed_validation','error':'preserved strict failure',
                'verification_observed_at':'2026-11-11T18:00:00Z'}
    monkeypatch.setattr(status,'observed_batch_verification',slow)
    worker=status.BatchObservationWorker();reference={'path':str(tmp_path/'probe.json'),'sha256':'a'*64}
    try:
        first=worker.observe(tmp_path,reference,'physics_action_open_loop')
        assert first['status']=='verification_pending' and started.wait(1)
        second=worker.observe(tmp_path,reference,'physics_action_open_loop')
        assert second['status']=='verification_pending' and len(calls)==1
        folder,row=request(tmp_path,'independent',44)
        snapshot=status.observe(tmp_path,batch_verifier=worker.observe)
        assert snapshot['records'][0]['start']==44 and snapshot['cycle_finished_at']
        finish.set()
        next(iter(worker.work.values()))['future'].result(timeout=1)
        result=worker.observe(tmp_path,reference,'physics_action_open_loop')
        assert result['status']=='failed_validation' and not result['verification_refresh_pending']
        assert result['verification_observed_at']=='2026-11-11T18:00:00Z'
    finally:
        finish.set();worker.executor.shutdown(wait=True)


def test_sequence_submission_requires_exact_member_config_and_does_not_mean_success(tmp_path):
    folder,row=request(tmp_path,'sequence',10)
    document=status.read(folder/'requests.json');document['sequence']=['A','B','A']
    write_json_atomic(folder/'requests.json',document)
    submitted={'request':status.ref(folder/'requests.json'),'native_runtime':'/native','job_id':'123','returncode':0,
        'members':[{'candidate_id':'sequence','start':10,'count':1,'config_sha256':row['config_sha256']}]}
    write_json_atomic(folder/'submission.json',submitted)
    record=status.observe(tmp_path)['records'][0]
    assert record['native_submission']['job_id']=='123' and record['status']=='native_submitted'
    assert record['sampler_sequence']['release_eligible'] is False and record['native'] is None
    submitted['members'][0]['config_sha256']='b'*64;write_json_atomic(folder/'submission.json',submitted)
    assert status.observe(tmp_path)['errors']


def test_repeated_a_control_is_separate_from_restoration_release_evidence(tmp_path):
    for name in ['restoration_control_aa_r1','batch_restoration_control_step_r2/control_aa',
                 'restoration_pristine_lifecycle_r1','restoration_pristine_mass_r1/aborted_AA',
                 'restoration_snapshot_scope_r1/successful_AA','restoration_integrated_cold_r1/fresh_A']:
        folder=tmp_path/'training_pipeline_canaries/tray'/name
        write_json_atomic(folder/'request.json',{'diagnostic_control':'A_A','diagnostic_only':True,'production_released':False})
        write_json_atomic(folder/'physics/report.json',{'success':True})
        write_json_atomic(folder/'physics/probe.json',{'schema':'isg-training-restoration-control-v1',
            'request':status.ref(folder/'request.json'),'entries':[{'returncode':0,'result':{'success':True},
            'report':status.ref(folder/'physics/report.json')}]*2})
    observed=status.observe(tmp_path)
    assert observed['batch_stages']==[]
    assert len(observed['diagnostic_controls'])==6
    assert all(c['successful_episodes']==2 and c['release_eligible'] is False for c in observed['diagnostic_controls'])
    assert observed['benchmark_credit']==0


def test_native_submission_normalization_preserves_original_and_checks_pointer(tmp_path):
    folder,row=request(tmp_path,'normalized',22)
    original={'request':status.ref(folder/'requests.json'),'start':22,'native_runtime':'/native','job_id':'44'}
    write_json_atomic(folder/'submission.json',original);original_ref=status.ref(folder/'submission.json')
    write_json_atomic(folder/'submission_normalized.json',{**original,'returncode':0,'original_submission':original_ref})
    write_json_atomic(folder/'submission_current.json',status.ref(folder/'submission_normalized.json'))
    record=status.observe(tmp_path)['records'][0]
    assert record['status']=='native_submitted' and record['original_native_submission']==original_ref
    assert record['native_submission']['evidence']==status.ref(folder/'submission_normalized.json')
    write_json_atomic(folder/'submission_current.json',{'path':str(folder/'submission_normalized.json'),'sha256':'b'*64})
    assert 'native submission revision changed' in status.observe(tmp_path)['errors'][0]['error']


def test_independent_native_submissions_do_not_wait_for_other_members(tmp_path):
    folder,row=request(tmp_path,'independent',40)
    document=status.read(folder/'requests.json')
    document['records'].append({**row,'candidate_id':'second','start':41,'output':str(folder/'second_native')})
    write_json_atomic(folder/'requests.json',document)
    first={**row,'request':status.ref(folder/'requests.json'),'job_id':'77','returncode':0}
    write_json_atomic(folder/'submission.json',{'request':status.ref(folder/'requests.json'),'jobs':[first]})
    rows=status.observe(tmp_path)['records']
    assert rows[0]['native_submission']['job_id']=='77'
    assert rows[1]['status']=='native_not_observed' and 'native_submission' not in rows[1]
    first['config_sha256']='changed'
    write_json_atomic(folder/'submission.json',{'request':status.ref(folder/'requests.json'),'jobs':[first]})
    assert 'configuration/count differs' in status.observe(tmp_path)['errors'][0]['error']


def test_stage_reconciliation_must_reproduce_original_recorded_hash(tmp_path):
    import pytest
    native=tmp_path/'native.json';native.write_text('native')
    template=tmp_path/'template.json';template.write_text('template')
    journal=tmp_path/'journal.json';journal.write_text('first')
    original=status.ref(journal)
    stage={'native_gate':status.ref(native),'isolation':{'template_export':status.ref(template),
        'pool_execution':original,'event_range':[0,4]}}
    sequence=tmp_path/'sequence_result.json';write_json_atomic(sequence,{'stages':[stage]})
    snapshot=tmp_path/'immutable_stage_evidence_r1/stage_0.json';snapshot.parent.mkdir();snapshot.write_text('first')
    journal.write_text('first plus later append')
    audit={'original_sequence_result':status.ref(sequence),'status':'all_recorded_stage_hashes_reproduced',
        'stages':[{'original_reference':original,'event_prefix_length':4,'immutable_reference':status.ref(snapshot)}]}
    write_json_atomic(snapshot.parent/'audit.json',audit)
    proof=status.sequence_stage_references(tmp_path,sequence,stage)
    assert proof['pool_execution']==status.ref(snapshot) and proof['original_pool_execution']==original
    snapshot.write_text('wrong')
    with pytest.raises(ValueError,match='evidence changed'):
        status.sequence_stage_references(tmp_path,sequence,stage)


def test_review_keys_are_scoped_to_their_request_batch(tmp_path):
    one,row=request(tmp_path,'one',12)
    two,_=request(tmp_path,'two',12)
    other=status.read(two/'requests.json');other['records'][0]['candidate_id']='one'
    write_json_atomic(two/'requests.json',other)
    write_json_atomic(one/'existing_native_reviews.json',{'campaign_id':'campaign_20261110',
        'records':{'one':{'allocation':{'start':'12'},'status':'awaiting_own_replay'}}})
    value=status.observe(tmp_path)
    assert len(value['records'])==2 and not value['errors']
    by_batch={r['batch']:r for r in value['records']}
    assert by_batch['one']['status']=='awaiting_own_replay'
    assert by_batch['two']['review'] is None


def test_one_bad_record_does_not_hide_later_records_in_same_batch(tmp_path):
    folder,row=request(tmp_path,'bad',1)
    good={**row,'candidate_id':'good','start':2,'output':str(folder/'good')}
    write_json_atomic(folder/'requests.json',{'campaign_id':'campaign_20261110','records':[row,good]})
    write_json_atomic(folder/'native/sampling_gate.json',{'candidate_id':'wrong','config_sha256':'a'*64})
    value=status.observe(tmp_path)
    assert [(r['candidate_id'],r['status'])for r in value['records']]==[
        ('bad','evidence_identity_failed'),('good','native_not_observed')]
    assert value['errors'][0]['candidate_id']=='bad'
    assert value['benchmark_credit']==0


def test_physical_control_retains_observed_success_without_release(tmp_path):
    request=tmp_path/'restoration_pristine/fresh_A/request.json'
    write_json_atomic(request,{'diagnostic_control':'FRESH_A','diagnostic_only':True,'production_released':False,
        'physical_candidate':{'path':'physical-proof','sha256':'p'*64}})
    report=request.parent/'physics/episode/meta.json';write_json_atomic(report,{'success':True})
    write_json_atomic(request.parent/'physics/probe.json',{'schema':'isg-training-physical-diagnostic-v1',
        'request':status.ref(request),'entries':[{'report':status.ref(report),'returncode':0,'result':{'success':True}}]})
    value=status.diagnostic_control_observation(tmp_path,request)
    assert value['status']=='episodes_recorded' and value['successful_episodes']==1
    assert value['release_eligible'] is False and value['benchmark_credit']==0
    report.write_text('{}')
    with pytest.raises(ValueError,match='report changed'):status.diagnostic_control_observation(tmp_path,request)
