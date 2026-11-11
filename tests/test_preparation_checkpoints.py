import json
from pathlib import Path

import pytest

from scripts.campaign_20261110 import preparation_checkpoints as p


def checkpoint(tmp_path, index=0, status='verified'):
    source = tmp_path / 'source.json'
    if not source.exists():
        source.write_text('[{"id":"object"}]')
    payload = tmp_path / 'native-input'
    if not payload.exists():
        payload.write_bytes(b'original physical bytes')
    return p.entry(tmp_path, source={'path': str(source), 'sha256': p.digest(source)},
                   index=index, identity={'id': f'object{index}'},
                   native={'snapshot': 'frozen'}, inputs={str(payload): p.digest(payload)},
                   audit={'status': status, 'native_map_config': {'arm': 'left'}})


def test_entry_is_replayable_but_cannot_replace_original(tmp_path):
    ref = checkpoint(tmp_path)
    assert checkpoint(tmp_path) == ref
    (tmp_path / 'native-input').write_bytes(b'changed physical bytes')
    with pytest.raises(ValueError, match='immutable preparation record differs'):
        checkpoint(tmp_path)
    assert p.digest(ref['path']) == ref['sha256']


def test_changed_source_is_rejected_before_publication(tmp_path):
    source = tmp_path / 'source.json'
    source.write_text('original')
    with pytest.raises(ValueError, match='source manifest changed'):
        p.entry(tmp_path, source={'path': str(source), 'sha256': '0' * 64}, index=0,
                identity={'id': 'x'}, native={'snapshot': 'x'}, inputs={str(source): p.digest(source)},
                audit={'status': 'verified'})
    assert not (tmp_path / 'entries').exists()


def test_batch_is_disjoint_and_rejects_failed_or_changed_entries(tmp_path):
    refs = [checkpoint(tmp_path, i) for i in range(2)]
    ref = p.batch(tmp_path, 0, refs, maximum=2)
    assert json.loads(Path(ref['path']).read_text())['source_indices'] == [0, 1]
    with pytest.raises(ValueError, match='overlap'):
        p.batch(tmp_path, 1, refs, maximum=2)
    with pytest.raises(ValueError, match='immutable preparation record differs'):
        p.batch(tmp_path, 1, [refs[1]], maximum=1)
    failed = checkpoint(tmp_path, 2, status='blocked')
    with pytest.raises(ValueError, match='unverified'):
        p.batch(tmp_path, 1, [failed], maximum=2)
    Path(refs[0]['path']).write_text('{}')
    with pytest.raises(ValueError, match='checkpoint changed'):
        p.batch(tmp_path, 0, refs, maximum=2)
