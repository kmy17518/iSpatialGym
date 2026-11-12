import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scripts.campaign_20261110 import reconcile_consumed_maps as m
from scripts.campaign_20261110 import vast_prebuild as v


def fixture(tmp_path, monkeypatch, positive=True):
    cpfile = tmp_path / 'contact_policy.py'; cpfile.write_text('pinned policy')
    rmfile = tmp_path / 'reach_maps.py'; rmfile.write_text('pinned native reader')
    identity = {'kind': 'generic', 'height': .7, 'arm': 'right', 'pose_set': {'dz_m': .03},
                'config': {'twins': False}, 'robot': 'robot', 'policy': 'policy'}
    maps = tmp_path / 'maps'; maps.mkdir(); path = maps / 'map__generic.json'
    meta = {**identity, 'height': {'value': .7}, 'key': 'native_key', 'policy_key': 'policy'}
    v.write(path, meta); path.with_suffix('.npz').write_text('pinned arrays')
    class Map:
        def __init__(self, name):
            self.path = name; self.meta = json.loads(Path(name).read_text())
            self.cells = np.array([[0., 0.]]); self.yaws = np.array([0.])
            self.arrays = {'s0': np.array([1 if positive else 2])}
    cp = ModuleType('contact_policy'); cp.__file__ = str(cpfile); cp.DEFAULT = SimpleNamespace(key=lambda arm: 'policy')
    rm = ModuleType('reach_maps'); rm.__file__ = str(rmfile); rm.ReachMap = Map
    rm.generic_map_key = lambda meta: 'native_key'; rm.robot_digest = lambda: 'robot'
    rm.STATUS = {'feasible': 1, 'invalid': 2}; rm.find_generic = lambda *args, **kw: Map(str(path))
    grasping = ModuleType('ispatialgym.grasping'); grasping.contact_policy = cp
    sampling = ModuleType('ispatialgym.sampling'); sampling.reach_maps = rm
    monkeypatch.setitem(sys.modules, 'ispatialgym.grasping', grasping)
    monkeypatch.setitem(sys.modules, 'ispatialgym.sampling', sampling)
    refs = [{'path': str(p), 'sha256': v.sha(p)} for p in (cpfile, rmfile)]
    prepared = tmp_path / 'prepared.json'; v.write(prepared, {'physical_identity': identity})
    ticket = {'consumer_kind': 'generic', 'physical_dedup_key': 'physical', 'prepared': {'path': str(prepared), 'sha256': v.sha(prepared)},
              'input_refs': refs, 'runtime_ref': refs[0]}
    terminal = tmp_path / 'terminal.json'
    v.write(terminal, {'physical_dedup_key': 'physical', 'prepared_sha256': v.sha(prepared), 'returncode': 0, 'native_done': True,
                       'started_at': 1, 'finished_at': 2, 'map_files': [{'path': str(p), 'sha256': v.sha(p)} for p in (path, path.with_suffix('.npz'))]})
    return ticket, terminal, maps, path


@pytest.mark.parametrize('positive', [True, False])
def test_generic_reader_distinguishes_native_positive_and_zero_without_library(tmp_path, monkeypatch, positive):
    ticket, terminal, maps, _ = fixture(tmp_path, monkeypatch, positive)
    result = m.native_read(ticket, terminal, maps, 'cluster', None, {})
    assert result['status'] == ('verified_positive' if positive else 'valid_zero')
    assert result['positive_cells'] == int(positive)
    assert not result['downstream_computation_required']


def test_generic_reader_rejects_wrong_pose_and_changed_grid(tmp_path, monkeypatch):
    ticket, terminal, maps, path = fixture(tmp_path, monkeypatch)
    meta = json.loads(path.read_text()); meta['pose_set']['dz_m'] = .04; v.write(path, meta)
    with pytest.raises(ValueError, match='pose or configuration'): m.native_read(ticket, terminal, maps, 'cluster', None, {})
    meta['pose_set']['dz_m'] = .03; v.write(path, meta); path.with_suffix('.npz').write_text('changed arrays')
    with pytest.raises(ValueError, match='Unbound generic'): m.native_read(ticket, terminal, maps, 'cluster', None, {})
