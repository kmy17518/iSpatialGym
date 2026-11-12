import copy
from dataclasses import asdict
import json

import pytest

from ispatialgym.grasping import schema as S, validate_artic as VA
from scripts.campaign_20261110.selected_artic_worker import audit_library


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
