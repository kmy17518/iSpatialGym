import json
from scripts.campaign_20261110.infra_vast.heartbeat_mirror import mirror_heartbeats


def test_mirror_keeps_exact_execution_timing_and_excludes_arbitrary_fields(tmp_path):
    h = {'time': 100, 'phase': 'run', 'gpu_used_mb': '7000,100', 'environment': {'TOKEN': 'secret'},
         'running': {'900001748': {'cpus': '1,2', 'phase': 'run', 'runtime_started_unix': 90,
                                   'environment': {'TOKEN': 'secret'}}, '../invalid': {'phase': 'run'}}}
    paths = mirror_heartbeats(tmp_path, {'60218588': h}, 110)
    d = json.loads(open(paths[0]).read())
    assert d['heartbeat']['time'] == 100
    assert d['observed_at_unix'] == 110
    assert d['heartbeat']['running'] == {'900001748': {'cpus': '1,2', 'phase': 'run', 'runtime_started_unix': 90}}
    assert 'secret' not in open(paths[0]).read()
    assert d['source_object'] == 'hb/60218588.json'


def test_older_or_invalid_heartbeat_cannot_replace_newer_evidence(tmp_path):
    mirror_heartbeats(tmp_path, {'1': {'time': 100, 'running': {}}}, 110)
    original = (tmp_path / '1.json').read_bytes()
    assert not mirror_heartbeats(tmp_path, {'1': {'time': 90}, '../2': {'time': 120},
                                            '3': {'time': float('nan')}}, 130)
    assert (tmp_path / '1.json').read_bytes() == original
    assert [p.name for p in tmp_path.iterdir()] == ['1.json']
