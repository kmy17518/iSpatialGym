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
    final=status.observe(tmp_path,publish=lambda value:published.append(copy.deepcopy(value)),previous_batches=[prior])
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
    assert not invalid['records'] and 'exact request/runtime/index' in invalid['errors'][0]['error']


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
