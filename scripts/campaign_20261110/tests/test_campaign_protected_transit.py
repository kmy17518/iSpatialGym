import numpy as np
import pytest
from ispatialgym.oracle.campaign_protected_transit import above_offset,enabled


def test_only_explicit_campaign_pregrasp_changes(monkeypatch):
    from ispatialgym.sampling import campaign_runtime
    monkeypatch.setattr(campaign_runtime, 'contact_recovery', lambda option: False)
    monkeypatch.setenv('ISPATIALGYM_PROTECT_ABOVE_GEOMETRY','1')
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG',raising=False)
    assert not enabled({'leg':'pre_grasp'})
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','exact.json')
    assert enabled({'leg':'pre_grasp'}) and not enabled({'leg':'carry'})
    monkeypatch.delenv('ISPATIALGYM_PROTECT_ABOVE_GEOMETRY')
    assert not enabled({'leg':'pre_grasp'})


def test_declarative_cap_option_requires_exact_pair_and_pregrasp(monkeypatch):
    from types import SimpleNamespace as NS
    from ispatialgym.sampling import campaign_runtime as CR
    from scripts.campaign_20261110.contracts import ContractError
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG', 'exact.json')
    monkeypatch.delenv('ISPATIALGYM_PROTECT_ABOVE_GEOMETRY', raising=False)
    target = NS(role='target', category='cap', model='xxqyrt', default_scale=(1.,)*3)
    partner = NS(role='partner', category='lemonade_bottle', model='yprkek', default_scale=(1.,)*3)
    template = NS(class_id='configuration_matching-attachment_02', task='configuration_matching-attachment',
                  split='eval_standard', objects=[target, partner],
                  scene_setup={'recovery_options': {'cap_geometry_above_pregrasp': True}})
    monkeypatch.setattr(CR, 'active', lambda: True)
    monkeypatch.setattr(CR, 'configuration', lambda: (None, template, None, None))
    assert enabled({'leg': 'pre_grasp'})
    assert not enabled({'leg': 'carry'})
    for obj, key, value in [(target, 'model', 'other'), (partner, 'model', 'other'),
                            (partner, 'default_scale', (.8,)*3), (template, 'split', 'eval_model'),
                            (template, 'class_id', 'configuration_matching-attachment_04')]:
        old = getattr(obj, key); setattr(obj, key, value)
        with pytest.raises(ContractError): enabled({'leg': 'pre_grasp'})
        setattr(obj, key, old)
    template.scene_setup['recovery_options']['cap_geometry_above_pregrasp'] = False
    assert not enabled({'leg': 'pre_grasp'})


def test_geometry_sets_height_without_weakening_clearance():
    offset,proof=above_offset([0,0,.97],[0,0,0,1],[[0,0,-.03],[0,0,.1]],
        [('pot',[[-.2,-.2,.87],[.2,.2,1.067]])],baseline=.12,clearance=.02)
    assert offset==pytest.approx(.147)
    assert proof['hand_lowest_z_at_goal']+offset==pytest.approx(1.087)
    assert proof['clearance_m']==.02


def test_rotated_hand_and_multiple_obstacles_are_bounded():
    offset,proof=above_offset([0,0,1],[1,0,0,0],[[0,0,-.03],[0,0,.1]],
        [('low',[[-1,-1,0],[1,1,.8]]),('high',[[-1,-1,0],[1,1,1.2]])],baseline=.12,clearance=.02)
    assert offset==pytest.approx(.32)
    assert proof['hand_lowest_z_at_goal']==pytest.approx(.9)


def test_baseline_remains_when_already_clear():
    assert above_offset([0,0,2],[0,0,0,1],[[0,0,0]],
        [('pot',[[0,0,0],[1,1,1]])],baseline=.12,clearance=.02)[0]==.12


@pytest.mark.parametrize('quaternion',[[0,0,0,0],[0,0,0,float('nan')]])
def test_invalid_geometry_cannot_claim_a_proposal(quaternion):
    with pytest.raises(ValueError):above_offset([0,0,1],quaternion,[[0,0,0]],[],baseline=.12,clearance=.02)
