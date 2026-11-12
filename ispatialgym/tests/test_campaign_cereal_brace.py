from types import SimpleNamespace as NS

import pytest
import torch as th

from ispatialgym.sampling import campaign_runtime as CR
from ispatialgym.sampling.v2 import contact_poses as CPZ
from ispatialgym.tests import test_touching_brace_open_push_speed as B
from scripts.campaign_20261110.contracts import ContractError


def bind(monkeypatch):
    t = NS(class_id='inter_object_relation_satisfaction-touching_02',
           task='inter_object_relation_satisfaction-touching_small_scale', split='eval_standard',
           scene_setup={'recovery_options': {'cereal_brace_gap_1mm': True}},
           objects=[NS(role='target', category='box_of_cereal', model='iucshm', default_scale=(.8,)*3),
                    NS(role='partner', category='box_of_cereal', model='dkhgxn', default_scale=(1.,)*3)])
    monkeypatch.setattr(CR, 'active', lambda: True)
    monkeypatch.setattr(CR, 'configuration', lambda: (None, t, None, None))
    return t


def test_gap_is_exact_template_bound(monkeypatch):
    t = bind(monkeypatch)
    assert CR.contact_recovery('cereal_brace_gap_1mm')
    t.objects[0].default_scale = (.800000011920929,)*3
    assert CR.contact_recovery('cereal_brace_gap_1mm')
    for obj, key, bad in [(t, 'class_id', 'inter_object_relation_satisfaction-touching_01'),
                          (t, 'split', 'eval_model'), (t.objects[0], 'model', 'other'),
                          (t.objects[1], 'default_scale', (.9,)*3)]:
        old = getattr(obj, key); setattr(obj, key, bad)
        with pytest.raises(ContractError): CR.contact_recovery('cereal_brace_gap_1mm')
        setattr(obj, key, old)


def test_only_brace_translates_four_mm_with_orientation_and_shape_preserved(monkeypatch):
    s = B._sampler(gen={'brace_open_when_wide': True})
    st, p_init, b_point, hull, face_z = B._brace_inputs(s)
    monkeypatch.setattr(CR, 'active', lambda: False)
    old = list(s.touching_brace_candidates(st, b_point, face_z, -B.DIRECTION, hull, p_init))
    bind(monkeypatch)
    new = list(s.touching_brace_candidates(st, b_point, face_z, -B.DIRECTION, hull, p_init))
    assert len(old) == len(new) > 0
    for (pre0, c0, k0, e0), (pre1, c1, k1, e1) in zip(old, new):
        assert k0 == k1 and th.equal(c0[1], c1[1]) and th.equal(pre0[1], pre1[1])
        delta = -.004 * B.DIRECTION
        assert th.allclose(c1[0] - c0[0], delta, atol=1e-6)
        assert th.allclose(pre1[0] - pre0[0], delta, atol=1e-6)
        assert e0['stop_clearance_m'] == .005 and e1['stop_clearance_m'] == .001
        assert e0['fingers_opening_m'] == e1['fingers_opening_m']
    # The shared generation setting used by push construction is unchanged.
    assert s.gen('stop_clearance_m', CPZ.STOP_CLEARANCE) == .005
