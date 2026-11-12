import copy
import json
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("prefix_worker_portability", Path(__file__).resolve().parents[1] / "selected_artic_prefix_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

def relocation_fixture(tmp_path, monkeypatch):
    roots = (tmp_path / 'primary/campaign_20261110', tmp_path / 'secondary/campaign_20261110')
    monkeypatch.setattr(worker, 'CAMPAIGN_ROOTS', roots)
    wave_root = roots[0] / 'training_preparation/selected_artic_prefix_r1'
    wave_root.mkdir(parents=True)
    request = wave_root / 'request.json'
    request.write_text('{"immutable": true}\n')
    unit = {'id': 'prefix_' + 'a' * 24, 'index': 2, 'output': str(wave_root / 'native/original'),
            'request': worker.ref(request), 'native_runtime': '/pinned/native', 'identity': {'scale': [.9316] * 3}}
    output = roots[1] / 'training_preparation/selected_artic_prefix_r1/native_secondary'
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
    record = {'schema': 'isg-selected-artic-prefix-input-relocations-v1', 'campaign_id': 'campaign_20261110',
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
    record = {'schema': 'isg-selected-artic-prefix-input-relocations-v1', 'campaign_id': 'campaign_20261110',
              'source_request': worker.ref(request), 'files': {'/private/customs.json': worker.ref(copied)}}
    if mismatch == 'request': record['source_request']['sha256'] = '0' * 64
    if mismatch == 'outside': record['files']['/private/customs.json']['path'] = str(tmp_path / 'outside')
    manifest = output.parent / 'mapping.json'; manifest.write_text(json.dumps(record))
    expected = '0' * 64 if mismatch == 'digest' else worker.ref(manifest)['sha256']
    with pytest.raises(AssertionError):
        worker.load_input_relocations(request, manifest, expected)

from scripts.campaign_20261110.tests.test_selected_artic_prefix_worker import artifact
from ispatialgym.sampling import reach_maps as RM


def test_relocated_motion_reader_preserves_original_identity(artifact, tmp_path):
    path, unit, meta, arrays = artifact
    original = unit['identity']['libraries'][0]
    copied = tmp_path / 'copy/motion.json'; copied.parent.mkdir(); copied.write_bytes(Path(original['path']).read_bytes())
    anchors = tmp_path / 'copy/anchors.json'; anchors.write_bytes(Path(unit['anchors']['path']).read_bytes())
    sources = {original['path']: copied, unit['anchors']['path']: anchors}
    meta['source']['library'] = str(copied)
    meta['key'] = RM.prefix_map_key(meta)
    changed = Path(RM.save_prefix(str(path.parent), meta, arrays))
    result = worker.audit_map(changed, unit, sources)
    assert result['verified'] and result['source_motion'] == original
    assert result['resolved_source_motion']['sha256'] == original['sha256']
    copied.write_text('{}')
    with pytest.raises(AssertionError): worker.audit_map(changed, unit, sources)
