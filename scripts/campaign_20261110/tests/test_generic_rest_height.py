import math
import sys
from types import ModuleType

import pytest

from scripts.campaign_20261110.prepare_consumed_generic_maps import native_object_height


def test_native_upright_rule_keeps_identity_and_stable_id_order(monkeypatch):
    cp = ModuleType('ispatialgym.sampling.v2.contact_poses'); cp.axial_rest_upright = lambda q: True
    monkeypatch.setitem(sys.modules, cp.__name__, cp)
    geometry = {'scale': [1, 1, 1], 'coll_bbox_min': [0, 0, -.01], 'coll_bbox_max': [1, 1, .03]}
    rest = {'rest_id': 'r0_other', 'stable': True, 'contact': True, 'scale': [1, 1, 1], 'quat_xyzw': [.001, 0, 0, .9999995]}
    height, rid, rule = native_object_height(geometry, {'rests': {'r0_other': rest}})
    assert height == .04 and rid == 'r0_other' and rule == 'upright_identity'


def test_tilted_rest_rotates_actual_hull_and_refuses_articulated_assumption(monkeypatch):
    cp = ModuleType('ispatialgym.sampling.v2.contact_poses'); cp.axial_rest_upright = lambda q: False
    monkeypatch.setitem(sys.modules, cp.__name__, cp)
    geometry = {'n_links': 1, 'scale': [1, 1, 1], 'pieces': [{'vertices': [[x, y, z] for x in [0, 1] for y in [0, 2] for z in [0, .1]]}]}
    rest = {'rest_id': 'r0_other', 'stable': True, 'contact': True, 'scale': [1, 1, 1],
            'quat_xyzw': [math.sqrt(.5), 0, 0, math.sqrt(.5)]}
    height, _, rule = native_object_height(geometry, {'rests': {'r0_other': rest}})
    assert height == pytest.approx(2, abs=1e-6) and rule == 'native_rotated_collision_hull'
    geometry['n_links'] = 2
    with pytest.raises(AssertionError, match='joint configuration'):
        native_object_height(geometry, {'rests': {'r0_other': rest}})
