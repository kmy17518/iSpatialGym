import json
import pytest
from scripts.campaign_20261110.customs_authoring import write_customs_atomic


def customs():
    return {'disembedding-example': {'roles': {
        'target': {'category': 'video_game'},
        'distractors': {'category': ['board_game']},
        'support': {'category': 'conference_table'},
        'source': {'category': 'tray', 'representative': 'gsxbym'},
        'destination': {'category': 'tray', 'representative': 'mhhoga'},
    }, 'generation': {'arm': 'right', 'other_arm': False}}}


def test_same_synset_roles_keep_their_native_instance_numbering(tmp_path):
    value = customs(); alias = next(iter(value)); order = list(value[alias]['roles'])
    path = tmp_path/'customs.json'
    write_customs_atomic(path, value, role_orders={alias: order})
    loaded = json.loads(path.read_text())
    assert loaded == value
    assert list(loaded[alias]['roles']) == order
    assert [r for r, v in loaded[alias]['roles'].items() if v['category'] == 'tray'] == ['source', 'destination']


def test_canonicalized_input_cannot_overwrite_a_valid_runtime_contract(tmp_path):
    value = customs(); alias = next(iter(value)); order = list(value[alias]['roles'])
    path = tmp_path/'customs.json'
    write_customs_atomic(path, value, role_orders={alias: order}); prior = path.read_bytes()
    reordered = json.loads(json.dumps(value, sort_keys=True))
    with pytest.raises(ValueError, match='Native role order'):
        write_customs_atomic(path, reordered, role_orders={alias: order})
    assert path.read_bytes() == prior


def test_unchecked_alias_is_rejected_before_writing(tmp_path):
    with pytest.raises(ValueError, match='Every customs alias'):
        write_customs_atomic(tmp_path/'customs.json', customs(), role_orders={})
    assert not (tmp_path/'customs.json').exists()
