import json

from scripts.campaign_20261110 import reconcile_consumed_libraries as m


def test_unchanged_terminal_is_audited_once_and_real_times_preserved(tmp_path):
    prepared = tmp_path / 'job/prepared.json'; prepared.parent.mkdir(); prepared.write_text('{}')
    terminal = prepared.parent / 'terminal.json'; terminal.write_text(json.dumps({'finished_at': 100}))
    manifest = tmp_path / 'tickets.json'; manifest.write_text(json.dumps({'jobs': [{'physical_dedup_key': 'key', 'prepared': m.ref(prepared)}]}))
    calls = []
    def audit(*args, **kwargs):
        calls.append(kwargs)
        return {'rows': [{'physical_dedup_key': 'key', 'status': 'verified_positive', 'job_id': 'j'}],
                'native_reader': {'path': 'native'}, 'native_candidate_generator': {'path': 'generator'}}
    out = tmp_path / 'reader'; first = m.cycle(tmp_path, manifest, m.ref(manifest)['sha256'], out, auditor=audit)
    proof = json.loads((out / 'state.json').read_text())['rows']['key']
    second = m.cycle(tmp_path, manifest, m.ref(manifest)['sha256'], out, auditor=audit)
    assert len(calls) == first['verified_this_cycle'] == 1 and second['verified_this_cycle'] == 0
    assert json.loads((out / 'state.json').read_text())['rows']['key']['verified_at'] == proof['verified_at']
    sidecar = json.loads((tmp_path / 'infra/coordinator_dispatch/preparation_verification/key.json').read_text())
    assert sidecar['terminal_at'] == 100 and sidecar['downstream_required']


def test_native_zero_is_not_positive_or_a_downstream_job(tmp_path):
    prepared = tmp_path / 'prepared.json'; prepared.write_text('{}')
    (tmp_path / 'terminal.json').write_text(json.dumps({'finished_at': 100}))
    manifest = tmp_path / 'tickets.json'; manifest.write_text(json.dumps({'jobs': [{'physical_dedup_key': 'zero', 'prepared': m.ref(prepared)}]}))
    def audit(*args, **kwargs):
        return {'rows': [{'physical_dedup_key': 'zero', 'status': 'valid_zero'}], 'native_reader': {}, 'native_candidate_generator': {}}
    result = m.cycle(tmp_path, manifest, m.ref(manifest)['sha256'], tmp_path / 'reader', auditor=audit)
    assert result['summary']['verified_zero'] == 1
    sidecar = json.loads((tmp_path / 'infra/coordinator_dispatch/preparation_verification/zero.json').read_text())
    assert not sidecar['downstream_required']
