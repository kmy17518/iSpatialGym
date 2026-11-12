import json
from pathlib import Path

import pytest

from ispatialgym.tests.test_artic_prefix_maps import _FakeWorld, _alib, _fake_solve
from ispatialgym.sampling import reach_maps as RM, reach_bands as RB
from scripts.campaign_20261110.selected_artic_prefix_worker import audit_map, ref


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(RM, '_solve', _fake_solve)
    monkeypatch.setattr(RB, 'trunk_grid', lambda h: [(.25, -.3), (.5, -.8)])
    cfg = RM.MapConfig(arm='right', lattice_m=.08, seeds=8, batch=2048, twins=False)
    lib = _alib()
    libpath = tmp_path / 'motion.json'
    libpath.write_text(json.dumps(lib))
    lib['_path'] = str(libpath)
    meta, arrays = RM.compute_prefix_map(_FakeWorld(cfg), lib, .8126, {'key': '0.8126'}, cfg=cfg)
    path = Path(RM.save_prefix(str(tmp_path), meta, arrays))
    anchors = tmp_path / 'anchors.json'
    anchors.write_text(json.dumps({'anchors': [{'height': .8126, 'key': '0.8126', 'precision': .0001, 'range': [.8126, .8126]}]}))
    unit = {'identity': {'libraries': [ref(libpath)], 'arm': 'right', 'height_key': '0.8126'}, 'anchors': ref(anchors)}
    return path, unit, meta, arrays


def test_full_map_native_lookup_and_positive_travel(artifact):
    path, unit, _, _ = artifact
    result = audit_map(path, unit)
    assert result['verified'] and result['positive'] and result['positive_travel_cells'] > 0


@pytest.mark.parametrize('change', ['missing_trunk', 'extra_array', 'wrong_q_start', 'wrong_scale', 'wrong_yaw'])
def test_rejects_incomplete_or_incompatible_map(artifact, change):
    path, unit, meta, arrays = artifact
    if change == 'missing_trunk': meta['trunks'].pop()
    if change == 'extra_array': arrays['unexpected'] = arrays['cells']
    if change == 'wrong_q_start': meta['q_start'] += .1
    if change == 'wrong_scale': meta['object']['scale'] = [.9316] * 3
    if change == 'wrong_yaw': arrays['yaws'] = arrays['yaws'] + .3
    meta['key'] = RM.prefix_map_key(meta)
    changed = Path(RM.save_prefix(str(path.parent), meta, arrays))
    with pytest.raises(AssertionError):
        audit_map(changed, unit)
