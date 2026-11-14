import json
import pytest
from scripts.campaign_20261110 import reconcile_consumed_maps as m


def case(tmp_path):
    maps = tmp_path / 'maps'; (maps / 'logs').mkdir(parents=True)
    path = maps / 'map.json'; path.write_text('{}'); path.with_suffix('.npz').write_bytes(b'grid')
    identity = {'kind': 'push_blade', 'arm': 'right', 'object_key': {'entry': 'exact'}, 'height_above_floor': .7}
    row = {'id': 'exact', 'kind': 'push_blade', 'arm': 'right', 'h': .7, 'ok': True, 'path': str(path)}
    (maps / 'logs/maps_push_blade_0of1.jsonl').write_text(json.dumps(row) + '\n')
    (tmp_path / 'native.log').write_text('GB_CONTACT_MAPS_DONE\nunloading all plugins\nFatal Python error: Segmentation fault\n')
    terminal = {'returncode': -11, 'native_done': True, 'map_files': [{'path': str(p), 'sha256': m.sha(p)} for p in (path, path.with_suffix('.npz'))]}
    return terminal, tmp_path / 'terminal.json', maps, identity


def test_complete_teardown_preserves_failed_process_disposition(tmp_path):
    result = m.completed_contact_teardown(*case(tmp_path))
    assert result['process_exit_success'] is False and result['original_returncode'] == -11
    assert len(result['completion_evidence']) == 2


@pytest.mark.parametrize('mutation', ['early_crash', 'no_unload', 'wrong_identity', 'changed_payload', 'missing_payload', 'failed_receipt', 'incomplete_terminal'])
def test_incomplete_or_mismatched_work_never_salvaged(tmp_path, mutation):
    args = case(tmp_path); terminal, _, maps, identity = args
    if mutation == 'early_crash': (tmp_path / 'native.log').write_text('Fatal Python error: Segmentation fault\nGB_CONTACT_MAPS_DONE\nunloading all plugins')
    if mutation == 'no_unload': (tmp_path / 'native.log').write_text('GB_CONTACT_MAPS_DONE\nFatal Python error: Segmentation fault')
    if mutation == 'wrong_identity': identity['height_above_floor'] = .8
    if mutation == 'changed_payload': (maps / 'map.npz').write_bytes(b'changed')
    if mutation == 'missing_payload': (maps / 'map.npz').unlink()
    if mutation == 'failed_receipt': (maps / 'logs/maps_push_blade_0of1.jsonl').write_text('{"ok": false}')
    if mutation == 'incomplete_terminal': terminal['map_files'].pop()
    with pytest.raises((ValueError, OSError)): m.completed_contact_teardown(*args)
