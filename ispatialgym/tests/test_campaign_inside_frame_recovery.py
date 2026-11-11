from types import SimpleNamespace as NS

import pytest

from ispatialgym.sampling import campaign_runtime as CR
from scripts.campaign_20261110.contracts import ContractError


def test_legacy_and_unconfigured_campaign_do_not_change(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG', raising=False)
    assert CR.inside_frame_recovery() is False
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG', '/unused')
    monkeypatch.setattr(CR, 'configuration', lambda: (None, NS(scene_setup={}), None, None))
    assert CR.inside_frame_recovery() is False


def test_recovery_is_pinned_to_the_audited_class(monkeypatch):
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG', '/unused')
    template=NS(class_id='inter_object_relation_satisfaction-inside_04',
        task='inter_object_relation_satisfaction-inside_small_scale',
        scene_setup={'recovery_options': {'inside_frame_rank': True}})
    monkeypatch.setattr(CR, 'configuration', lambda: (None, template, None, None))
    assert CR.inside_frame_recovery() is True
    template.class_id='inter_object_relation_satisfaction-inside_01'
    with pytest.raises(ContractError, match='audited saucepot'):
        CR.inside_frame_recovery()
    template.scene_setup['recovery_options']['inside_frame_rank']='false'
    with pytest.raises(ContractError, match='boolean'):
        CR.inside_frame_recovery()
