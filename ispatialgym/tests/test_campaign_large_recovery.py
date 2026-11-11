from types import SimpleNamespace
import pytest
from ispatialgym.sampling import campaign_runtime as runtime
from ispatialgym.sampling.campaign_large_recovery import pillow_grasp_margin


def test_legacy_and_other_campaigns_keep_original_grasp_selection(monkeypatch):
    monkeypatch.setattr(runtime,'active',lambda:False)
    assert pillow_grasp_margin() is None
    monkeypatch.setattr(runtime,'active',lambda:True)
    monkeypatch.setattr(runtime,'configuration',lambda:(None,SimpleNamespace(scene_setup={}),None,None))
    assert pillow_grasp_margin() is None


def test_option_is_exact_class_scoped_and_strict(monkeypatch):
    monkeypatch.setattr(runtime,'active',lambda:True)
    t=SimpleNamespace(scene_setup={'recovery_options':{'pillow_grasp_margin':True}},
        class_id='inter_object_relation_satisfaction-ontop_large_scale_03',
        task='inter_object_relation_satisfaction-ontop_large_scale')
    monkeypatch.setattr(runtime,'configuration',lambda:(None,t,None,None))
    assert pillow_grasp_margin()==.015
    t.class_id='another_class'
    with pytest.raises(ValueError,match='audited pillow class'):pillow_grasp_margin()
    t.scene_setup['recovery_options']['pillow_grasp_margin']='true'
    with pytest.raises(ValueError,match='must be boolean'):pillow_grasp_margin()
