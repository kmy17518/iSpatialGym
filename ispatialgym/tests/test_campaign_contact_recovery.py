from types import SimpleNamespace as NS

import pytest

from ispatialgym.sampling import campaign_runtime as CR
from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.motions import MotionsConfig
from scripts.campaign_20261110.contracts import ContractError


def bind(monkeypatch, class_id, task, options):
    template = NS(class_id=class_id, task=task, scene_setup={'recovery_options': options})
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG', '/unused')
    monkeypatch.setattr(CR, 'configuration', lambda: (None, template, None, None))
    return template


def test_legacy_closure_and_unconfigured_campaign_stay_identical(monkeypatch):
    cfg = MotionsConfig()
    action = NS(kind='library')
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    assert EX.grasp_closure(action, .014, cfg) == (cfg.gripper_close_rate, None, False)
    assert not CR.contact_recovery('touching_push_mass_center')
    bind(monkeypatch, 'any', 'any', {})
    assert EX.grasp_closure(action, .014, cfg) == (cfg.gripper_close_rate, None, False)


def test_marker_contact_closure_is_bounded_and_does_not_change_handles(monkeypatch):
    bind(monkeypatch, 'inter_object_relation_satisfaction-nextto_01',
         'inter_object_relation_satisfaction-nextto_small_scale', {'marker_gentle_pinch': True})
    cfg = MotionsConfig()
    assert EX.grasp_closure(NS(kind='library'), .014, cfg) == (.0005, 0.0, True)
    for width in [None, .021, float('nan'), 0.0, -1.0]:
        assert EX.grasp_closure(NS(kind='library'), width, cfg) == EX.closure_for_width(width, cfg)
    assert EX.grasp_closure(NS(kind='handle'), .014, cfg) == (cfg.gripper_close_rate, None, False)


def test_mass_center_option_uses_existing_physical_constructor(monkeypatch):
    from ispatialgym.sampling.v2.tasks import iors_small as IS
    monkeypatch.delenv(IS.PUSH_LINE_COM_ENV, raising=False)
    template = bind(monkeypatch, 'inter_object_relation_satisfaction-touching_04',
                    'inter_object_relation_satisfaction-touching_small_scale', {'touching_push_mass_center': True})
    assert IS.push_line_com_on()
    template.class_id = 'inter_object_relation_satisfaction-touching_01'
    with pytest.raises(ContractError, match='another audited class'):
        IS.push_line_com_on()
    template.scene_setup['recovery_options']['touching_push_mass_center'] = 'true'
    with pytest.raises(ContractError, match='boolean'):
        IS.push_line_com_on()
