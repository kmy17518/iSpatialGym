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


def test_soap_early_lift_uses_existing_program_projection(monkeypatch):
    import torch
    from ispatialgym.sampling.campaign_large_recovery import SOAP_LIFT_FRAME_ENV, soap_lift_frame_rejection
    from ispatialgym.sampling.v2.program_check import fingertips_in_frame

    class Camera:
        width, height, fx = 1920, 1080, 1000

        def project(self, camera, points):
            # The grasp is in frame; the lifted pose falls below the lower bound.
            return torch.tensor([[960., 900. + 900. * float(points[0, 2])]]), torch.tensor([.8])

    sampler = SimpleNamespace(
        gp=SimpleNamespace(variant='inter_object_relation_satisfaction-ontop_large_scale-soap_dispenser-countertop-floor'),
        kit=SimpleNamespace(world=SimpleNamespace(camera_model=Camera())),
        ikc=SimpleNamespace(camera_pose=lambda stance: object()))
    stance = object()
    grasp = (torch.tensor([0., 0., 0.]), torch.tensor([0., 0., 0., 1.]))
    lift = (torch.tensor([0., 0., .25]), grasp[1])
    monkeypatch.delenv(SOAP_LIFT_FRAME_ENV, raising=False)
    assert soap_lift_frame_rejection(sampler, stance, grasp, lift) is None
    monkeypatch.setenv(SOAP_LIFT_FRAME_ENV, '1')
    expected = fingertips_in_frame(sampler, stance, 'lift', [('lift', [grasp, lift])])
    assert expected is not None
    assert soap_lift_frame_rejection(sampler, stance, grasp, lift) == expected
    assert soap_lift_frame_rejection(sampler, stance, grasp, grasp) is None
    sampler.gp.variant = 'inter_object_relation_satisfaction-ontop_large_scale-mouse-desk-floor'
    with pytest.raises(ValueError, match='audited soap class'):
        soap_lift_frame_rejection(sampler, stance, grasp, lift)
