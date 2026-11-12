import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[3] / 'ispatialgym/sampling/support_rules.py'
spec = importlib.util.spec_from_file_location('private_support_rules_test', MODULE)
sr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sr)


def write_ref(path, value):
    path.write_text(json.dumps(value))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    scene = write_ref(tmp_path / 'private_scene_stable.json', {'objects': {'support': 'original'}})
    record = {'scene': 'private_scene', 'name': 'support', 'category': 'table', 'model': 'model', 'scale': [1., 1., 1.],
              'top': {'resolution': .02, 'hull_xy': [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]}}
    inventory = write_ref(tmp_path / 'support.json', record)
    measured = {'native_inventory_record': inventory, 'source_scene': scene, 'object_name': 'support',
                'actual_scale': [1., 1., 1.], 'category': 'table', 'model': 'model'}
    proof = write_ref(tmp_path / 'measurement.json', measured)
    entry = {key: record[key] for key in ['scene', 'name', 'category', 'model', 'scale']}
    entry.update(scene_file=scene, inventory_record=inventory, measurement=proof)
    manifest = {'schema': 'isg-private-support-inventory-v1', 'entries': [entry]}
    path = tmp_path / 'manifest.json'
    def publish():
        reference = write_ref(path, manifest)
        monkeypatch.setenv(sr.INVENTORY_MANIFEST_ENV, reference['path'])
        monkeypatch.setenv(sr.INVENTORY_MANIFEST_SHA_ENV, reference['sha256'])
    publish()
    monkeypatch.setenv('ISPATIALGYM_STABLE_SCENE_DIR', str(tmp_path))
    monkeypatch.setattr(sr, 'DEFAULT_INVENTORY', str(tmp_path / 'empty'))
    return record, manifest, publish


def test_exact_b4_locator_keeps_rectangularity_rule(inputs):
    assert sr.top_rectangularity('private_scene', 'support') == pytest.approx(1.)
    assert sr.exclusion_reason('alignment-axial', 'private_scene', 'support') is None
    assert 'no B4' in sr.exclusion_reason('alignment-axial', 'private_scene', 'other')


def test_nonrectangular_native_record_still_rejected(inputs):
    record, manifest, publish = inputs
    entry = manifest['entries'][0]
    record['top']['hull_xy'] = [[0, 0], [1, 0], [0, 1]]
    entry['inventory_record'] = write_ref(Path(entry['inventory_record']['path']), record)
    measured = json.loads(Path(entry['measurement']['path']).read_text())
    measured['native_inventory_record'] = entry['inventory_record']
    entry['measurement'] = write_ref(Path(entry['measurement']['path']), measured)
    publish()
    assert sr.exclusion_reason('alignment-axial', 'private_scene', 'support').startswith('not_rectangular:')


@pytest.mark.parametrize('field', ['scene_file', 'inventory_record', 'measurement'])
def test_changed_evidence_fails_even_after_previous_read(inputs, field):
    _, manifest, _ = inputs
    assert sr.top_rectangularity('private_scene', 'support') == pytest.approx(1.)
    Path(manifest['entries'][0][field]['path']).write_text('{}')
    with pytest.raises(sr.SupportRuleError, match='digest differs'):
        sr.top_rectangularity('private_scene', 'support')


@pytest.mark.parametrize('change', ['scale', 'duplicate', 'scene', 'manifest_hash', 'missing_hash', 'nonfinite'])
def test_identity_and_binding_fail_closed(inputs, monkeypatch, change):
    record, manifest, publish = inputs
    if change == 'scale': manifest['entries'][0]['scale'] = [2., 2., 2.]
    elif change == 'duplicate': manifest['entries'].append(manifest['entries'][0])
    elif change == 'scene': monkeypatch.setenv('ISPATIALGYM_STABLE_SCENE_DIR', '/wrong/scene')
    elif change == 'nonfinite':
        record['top']['hull_xy'][0][0] = float('nan')
        entry = manifest['entries'][0]
        entry['inventory_record'] = write_ref(Path(entry['inventory_record']['path']), record)
        measured = json.loads(Path(entry['measurement']['path']).read_text()); measured['native_inventory_record'] = entry['inventory_record']
        entry['measurement'] = write_ref(Path(entry['measurement']['path']), measured)
    publish()
    if change == 'manifest_hash': monkeypatch.setenv(sr.INVENTORY_MANIFEST_SHA_ENV, '0' * 64)
    if change == 'missing_hash': monkeypatch.delenv(sr.INVENTORY_MANIFEST_SHA_ENV)
    with pytest.raises(sr.SupportRuleError): sr.top_rectangularity('private_scene', 'support')


def test_default_and_explicit_inventory_behavior_unchanged(inputs, monkeypatch, tmp_path):
    monkeypatch.delenv(sr.INVENTORY_MANIFEST_ENV)
    monkeypatch.delenv(sr.INVENTORY_MANIFEST_SHA_ENV)
    assert sr.top_rectangularity('private_scene', 'support') is None
    directory = tmp_path / 'ordinary/private_scene/supports'; directory.mkdir(parents=True)
    write_ref(directory / 'support.json', {'top': {'hull_xy': [[0, 0], [1, 0], [0, 1]]}})
    assert sr.top_rectangularity('private_scene', 'support', str(tmp_path / 'ordinary')) == pytest.approx(.5)
