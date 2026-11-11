from scripts.campaign_20261110 import training_pipeline_status as status
from scripts.campaign_20261110.contracts import write_json_atomic


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


def test_review_must_match_original_native_index(tmp_path):
    folder, row = request(tmp_path, 'candidate', 1)
    write_json_atomic(folder/'existing_native_reviews.json', {'campaign_id': 'campaign_20261110',
        'observed_at': '2026-11-11T00:00:00Z', 'records': {'candidate': {
            'allocation': {'start': '2'}, 'status': 'validated_canary_exported'}}})
    value = status.observe(tmp_path)
    assert value['records'][0]['review'] is None
    assert value['records'][0]['status'] == 'native_not_observed'


def test_batch_request_is_not_batch_success(tmp_path):
    path = tmp_path/'training_pipeline_canaries/batch/batch_restoration_r1'
    write_json_atomic(path/'request.json', {'canaries': []})
    write_json_atomic(path/'preflight_failure.json', {'reason': 'incompatible scene'})
    value = status.observe(tmp_path)
    assert value['batch_stages'][0]['status'] == 'preflight_failed'
    assert all(s['status'] == 'not_observed' for s in value['batch_stages'][0]['stages'])
