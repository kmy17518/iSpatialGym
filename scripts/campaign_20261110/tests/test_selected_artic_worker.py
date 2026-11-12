import copy
from dataclasses import asdict
import json

import pytest

from ispatialgym.grasping import schema as S, validate_artic as VA
from scripts.campaign_20261110.selected_artic_worker import audit_library
from scripts.campaign_20261110 import selected_artic_worker as worker


def fixture(tmp_path):
    unit = {'alias': 'selected', 'identity': {'category': 'blender', 'model': 'cwkvib', 'joint': 'j_lid',
            'scale': [.9316] * 3, 'arm': 'right', 'generation': {'direction': 'open'}}}
    trace = {'validation_world': {'arm': 'right'}, 'config': {'artic': {'fixed_base': False}},
             'motion': {'handle_sig': 'exact', 'scale': [.9316] * 3}}
    (tmp_path / 'trace.json').write_text(json.dumps(trace))
    lib = {'schema': VA.LIBRARY_V2, 'alias': 'selected', 'category': 'blender', 'model': 'cwkvib',
           'joint': 'j_lid', 'scale': [.9316] * 3, 'arm_validated': 'right', 'n_segments': 1,
           'motion': 'pull_open', 'config': asdict(VA.MotionConfig()), 'q_start': 0., 'handle_sig': 'exact',
           'rests': {'r0': {'schema': S.REST, 'rest_id': 'r0', 'stable': True, 'quat_xyzw': [0, 0, 0, 1], 'h_root': .1}},
           'segments': [{'arm_validated': 'right', 'joint': 'j_lid', 'trial_stop': 'limit_reached',
                         'validated_travel_rad': .2, 'waypoints': [{}, {}], 'handle_sig': 'exact',
                         'rest_id': 'r0', 'source': {'validation': 'trace.json'}}]}
    return unit, lib


def check(tmp_path, unit, lib):
    path = tmp_path / 'library.json'
    path.write_text(json.dumps(lib))
    return audit_library(path, unit)


def test_positive_motion_is_distinct_from_contact_only(tmp_path):
    unit, lib = fixture(tmp_path)
    assert check(tmp_path, unit, lib)['positive']
    lib['segments'][0].update(validated_travel_rad=0, waypoints=[{}])
    assert not check(tmp_path, unit, lib)['positive']


@pytest.mark.parametrize('preparation', ['two_flap_v2', 'two_flap_v4'])
def test_exact_two_flap_motion_kind_is_not_plain_pull(tmp_path, preparation):
    unit, lib = fixture(tmp_path)
    unit['identity']['generation']['prep_push'] = preparation
    lib['motion'] = 'v2_ajar_pull'
    assert check(tmp_path, unit, lib)['positive']
    lib['motion'] = 'pull_open'
    with pytest.raises(AssertionError):
        check(tmp_path, unit, lib)


def test_free_hinge_close_cannot_claim_push_component(tmp_path):
    unit, lib = fixture(tmp_path)
    unit['identity']['generation']['direction'] = 'close'
    lib['motion'] = 'push_close'
    with pytest.raises(AssertionError):
        check(tmp_path, unit, lib)
    unit['identity']['generation']['close_push'] = True
    assert check(tmp_path, unit, lib)['positive']


@pytest.mark.parametrize('change', [
    {'scale': [1, 1, 1]}, {'arm_validated': 'left'}, {'motion': 'push_close'},
    {'alias': 'other'}, {'model': 'eyedvd'}, {'joint': 'other'}, {'n_segments': 2},
])
def test_rejects_incompatible_library(tmp_path, change):
    unit, lib = fixture(tmp_path)
    lib.update(change)
    with pytest.raises(AssertionError):
        check(tmp_path, unit, lib)


@pytest.mark.parametrize('change', [
    {'arm_validated': 'left'}, {'trial_stop': 'ag_not_formed'},
    {'source': {'validation': '../outside.json'}}, {'handle_sig': 'other'},
])
def test_rejects_incompatible_segment(tmp_path, change):
    unit, lib = fixture(tmp_path)
    lib['segments'][0].update(copy.deepcopy(change))
    with pytest.raises(AssertionError):
        check(tmp_path, unit, lib)


def test_rejects_fixed_base_and_unstable_rest(tmp_path):
    unit, lib = fixture(tmp_path)
    lib['rests']['r0']['stable'] = False
    with pytest.raises(AssertionError):
        check(tmp_path, unit, lib)
    lib['rests']['r0']['stable'] = True
    trace = json.loads((tmp_path / 'trace.json').read_text())
    trace['config']['artic']['fixed_base'] = True
    (tmp_path / 'trace.json').write_text(json.dumps(trace))
    with pytest.raises(AssertionError):
        check(tmp_path, unit, lib)


def relocation_fixture(tmp_path, monkeypatch):
    roots = (tmp_path / 'primary/campaign_20261110', tmp_path / 'secondary/campaign_20261110')
    monkeypatch.setattr(worker, 'CAMPAIGN_ROOTS', roots)
    wave_root = roots[0] / 'training_preparation/selected_artic_motion_r1'
    wave_root.mkdir(parents=True)
    request = wave_root / 'request.json'
    request.write_text('{"immutable": true}\n')
    unit = {'id': 'artic_' + 'a' * 24, 'index': 2, 'output': str(wave_root / 'native/original'),
            'request': worker.ref(request), 'native_runtime': '/pinned/native', 'identity': {'scale': [.9316] * 3}}
    output = roots[1] / 'training_preparation/selected_artic_motion_r1/native_secondary'
    return roots, request, unit, output


def test_output_relocation_preserves_request_and_unit(tmp_path, monkeypatch):
    _, request, unit, output = relocation_fixture(tmp_path, monkeypatch)
    before = copy.deepcopy(unit); source = request.read_bytes()
    out, receipt = worker.resolve_output(request, {'campaign_id': 'campaign_20261110'}, unit, output)
    assert out == output / unit['id']
    assert receipt['request'] == worker.ref(request)
    assert receipt['unit_id'] == unit['id'] and receipt['unit_index'] == 2
    assert receipt['original_output'] == unit['output'] and receipt['actual_output'] == str(out)
    assert receipt['native_inputs_unchanged'] and len(receipt['unit_payload_sha256']) == 64
    assert unit == before and request.read_bytes() == source and not output.exists()


def test_no_relocation_preserves_original_path(tmp_path, monkeypatch):
    _, request, unit, _ = relocation_fixture(tmp_path, monkeypatch)
    out, mapping = worker.resolve_output(request, {}, unit)
    assert str(out) == unit['output'] and mapping is None


@pytest.mark.parametrize('kind', ['outside', 'different_wave', 'relative', 'parent_escape', 'symlink_escape'])
def test_relocation_rejects_foreign_or_escaping_paths(tmp_path, monkeypatch, kind):
    roots, request, unit, output = relocation_fixture(tmp_path, monkeypatch)
    if kind == 'outside': output = tmp_path / 'outside'
    if kind == 'different_wave': output = roots[1] / 'training_preparation/another_wave'
    if kind == 'relative': output = type(output)('relative/output')
    if kind == 'parent_escape': output = output / '..' / '..' / 'foreign'
    if kind == 'symlink_escape':
        output.parent.mkdir(parents=True); target = tmp_path / 'outside'; target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    with pytest.raises(AssertionError):
        worker.resolve_output(request, {'campaign_id': 'campaign_20261110'}, unit, output)


def test_exact_input_copy_binds_original_request(tmp_path, monkeypatch):
    _, request, _, output = relocation_fixture(tmp_path, monkeypatch)
    copied = output.parent / 'inputs/customs.json'; copied.parent.mkdir(parents=True); copied.write_bytes(b'{"exact": true}\n')
    manifest = output.parent / 'mapping.json'
    record = {'schema': 'isg-selected-artic-input-relocations-v1', 'campaign_id': 'campaign_20261110',
              'source_request': worker.ref(request), 'files': {'/private/customs.json': worker.ref(copied)}}
    manifest.write_text(json.dumps(record)); before = request.read_bytes()
    sources, proof = worker.load_input_relocations(request, manifest, worker.ref(manifest)['sha256'])
    assert sources == {'/private/customs.json': copied} and proof == worker.ref(manifest)
    assert request.read_bytes() == before
    copied.write_bytes(b'changed')
    with pytest.raises(AssertionError, match='input copy changed'):
        worker.load_input_relocations(request, manifest, worker.ref(manifest)['sha256'])


@pytest.mark.parametrize('mismatch', ['digest', 'request', 'outside'])
def test_rejects_unbound_input_copy(tmp_path, monkeypatch, mismatch):
    _, request, _, output = relocation_fixture(tmp_path, monkeypatch)
    copied = output.parent / 'inputs/customs.json'; copied.parent.mkdir(parents=True); copied.write_bytes(b'{}')
    record = {'schema': 'isg-selected-artic-input-relocations-v1', 'campaign_id': 'campaign_20261110',
              'source_request': worker.ref(request), 'files': {'/private/customs.json': worker.ref(copied)}}
    if mismatch == 'request': record['source_request']['sha256'] = '0' * 64
    if mismatch == 'outside': record['files']['/private/customs.json']['path'] = str(tmp_path / 'outside')
    manifest = output.parent / 'mapping.json'; manifest.write_text(json.dumps(record))
    expected = '0' * 64 if mismatch == 'digest' else worker.ref(manifest)['sha256']
    with pytest.raises(AssertionError):
        worker.load_input_relocations(request, manifest, expected)
