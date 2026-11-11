from types import SimpleNamespace
import pytest
from ispatialgym.sampling import campaign_runtime as runtime
from ispatialgym.sampling.campaign_large_recovery import pillow_grasp_margin


def test_candidate_screen_retains_original_sidecar_after_rejection_and_success(monkeypatch):
    from contextlib import nullcontext
    from ispatialgym.sampling.campaign_large_recovery import candidate_program_rejection
    from ispatialgym.sampling.v2 import program_check
    from ispatialgym.sampling.sim_utils import Reject
    monkeypatch.setattr(runtime, 'active', lambda: True)
    template = SimpleNamespace(scene_setup={'recovery_options': {'heel_candidate_program_screen': True}},
        class_id='inter_object_relation_satisfaction-under_large_scale_01',
        task='inter_object_relation_satisfaction-under_large_scale')
    monkeypatch.setattr(runtime, 'configuration', lambda: (None, template, None, None))
    original = SimpleNamespace(ik_witnesses=[{'q': [-1.98]}], paths=['checked_original'])
    sampler = SimpleNamespace(sidecar=original, gp=SimpleNamespace(variant='sandal'), subtask='under',
        _floor_relaxed=nullcontext)
    calls = []

    def check(variant, sidecar, current, subtask):
        assert current.sidecar is sidecar and sidecar is not original
        calls.append((variant, subtask))
        sidecar.paths.append('provisional_transition')
        sidecar.ik_witnesses = [{'q': [-2.0921]}]

    def heel_gate():
        if sampler.sidecar.ik_witnesses[0]['q'][0] < -2.0644:
            raise Reject('program_check_failed', 'retreat8 left-j4 2.3mrad <30mrad')

    monkeypatch.setattr(program_check, 'run', check)
    sampler.heel_tight_left_j4_check = heel_gate
    assert 'retreat8' in candidate_program_rejection(sampler)
    assert sampler.sidecar is original and original.paths == ['checked_original']
    assert original.ik_witnesses == [{'q': [-1.98]}]
    sampler.heel_tight_left_j4_check = lambda: None
    assert candidate_program_rejection(sampler) is None
    assert sampler.sidecar is original and original.paths == ['checked_original']
    assert calls == [('sandal', 'under'), ('sandal', 'under')]
    monkeypatch.setattr(program_check, 'run', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('unexpected failure')))
    with pytest.raises(RuntimeError, match='unexpected failure'):
        candidate_program_rejection(sampler)
    assert sampler.sidecar is original


def test_candidate_screen_is_scoped_to_explicit_audited_config_or_plan(monkeypatch):
    from ispatialgym.sampling.campaign_large_recovery import candidate_program_screen
    monkeypatch.setattr(runtime, 'active', lambda: False)
    assert candidate_program_screen() is False
    monkeypatch.setattr(runtime, 'active', lambda: True)
    template = SimpleNamespace(scene_setup={}, class_id='inter_object_relation_satisfaction-under_large_scale_02',
        task='inter_object_relation_satisfaction-under_large_scale')
    monkeypatch.setattr(runtime, 'configuration', lambda: (None, template, None, None))
    assert candidate_program_screen() is False
    assert candidate_program_screen(SimpleNamespace(floor_push_source_proposal={'arm': 'right'})) is True
    template.class_id = 'unrelated'
    with pytest.raises(ValueError, match='audited briefcase'):
        candidate_program_screen(SimpleNamespace(floor_push_source_proposal={'arm': 'right'}))
    template.scene_setup = {'recovery_options': {'heel_candidate_program_screen': 'true'}}
    with pytest.raises(ValueError, match='must be boolean'):
        candidate_program_screen()
    template.scene_setup['recovery_options']['heel_candidate_program_screen'] = True
    with pytest.raises(ValueError, match='audited sandal'):
        candidate_program_screen()


def test_heel_path_screen_catches_interior_limit_approach(monkeypatch):
    from ispatialgym.sampling.campaign_large_recovery import heel_carry_path_screen, heel_carry_path_failure
    monkeypatch.setattr(runtime, 'active', lambda: True)
    template = SimpleNamespace(scene_setup={'recovery_options': {'heel_carry_path_screen': True}},
        class_id='inter_object_relation_satisfaction-under_large_scale_01',
        task='inter_object_relation_satisfaction-under_large_scale')
    monkeypatch.setattr(runtime, 'configuration', lambda: (None, template, None, None))
    robot = SimpleNamespace(arm_control_idx={'left': [0, 1, 2, 3]}, joint_lower_limits=[-3, -3, -3, -2.0944])
    sampler = SimpleNamespace(kit=SimpleNamespace(robot=robot), push_heel_tight=lambda: 5.,
        ikc=SimpleNamespace(last_track_traj=[[0, 0, 0, -1.98], [0, 0, 0, -2.082752], [0, 0, 0, -1.98]]))
    assert 'configuration 1' in heel_carry_path_failure(sampler, 'left', .03)
    assert heel_carry_path_failure(sampler, 'right', .03) is None
    sampler.ikc.last_track_traj[1][3] = -2.05
    assert heel_carry_path_failure(sampler, 'left', .03) is None
    sampler.ikc.last_track_traj = None
    with pytest.raises(ValueError, match='no tracked configurations'):
        heel_carry_path_failure(sampler, 'left', .03)
    sampler.push_heel_tight = lambda: None
    assert heel_carry_path_failure(sampler, 'left', .03) is None
    template.class_id = 'another_class'
    with pytest.raises(ValueError, match='audited sandal class'):
        heel_carry_path_screen()
    template.scene_setup['recovery_options']['heel_carry_path_screen'] = 'true'
    with pytest.raises(ValueError, match='must be boolean'):
        heel_carry_path_screen()
    monkeypatch.setattr(runtime, 'active', lambda: False)
    assert heel_carry_path_screen() is False


def test_soap_carry_screen_reuses_final_program_bounds(monkeypatch):
    from ispatialgym.sampling.v2.tasks.iors_large import IORSLarge
    from ispatialgym.sampling.campaign_large_recovery import soap_carry_oracle_screen
    monkeypatch.setattr(runtime, 'active', lambda: True)
    template = SimpleNamespace(scene_setup={'recovery_options': {'soap_carry_oracle_screen': True}},
        class_id='inter_object_relation_satisfaction-ontop_large_scale_01',
        task='inter_object_relation_satisfaction-ontop_large_scale')
    monkeypatch.setattr(runtime, 'configuration', lambda: (None, template, None, None))
    ik = SimpleNamespace(FINISH_MAX_EXCURSION=.6, ORACLE_SOFT_POS_TOL=.01,
                         ORACLE_SOFT_ROT_TOL=.02, ORACLE_DIFF_IK_MAX_ITERS=100)
    sampler = SimpleNamespace(gp=SimpleNamespace(gen=lambda key: None), ikc=ik)
    expected = {'finish_max_excursion': .6, 'soft_pos_tol': .01,
                'soft_rot_tol': .02, 'max_iters': 100}
    assert IORSLarge._carry_track_kw(sampler) == expected
    template.scene_setup = {}
    assert IORSLarge._carry_track_kw(sampler) == {}
    sampler.gp.gen = lambda key: key == 'carry_track_oracle' or None
    assert IORSLarge._carry_track_kw(sampler) == expected
    template.scene_setup = {'recovery_options': {'soap_carry_oracle_screen': True}}
    template.class_id = 'another_class'
    with pytest.raises(ValueError, match='audited soap class'):
        soap_carry_oracle_screen()
    template.scene_setup['recovery_options']['soap_carry_oracle_screen'] = 'true'
    with pytest.raises(ValueError, match='must be boolean'):
        soap_carry_oracle_screen()
    monkeypatch.setattr(runtime, 'active', lambda: False)
    assert soap_carry_oracle_screen() is False


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


@pytest.mark.parametrize('floor_alias', ['floor', 'floors'])
def test_soap_early_lift_uses_existing_program_projection(monkeypatch, floor_alias):
    import torch
    from ispatialgym.sampling.campaign_large_recovery import SOAP_LIFT_FRAME_ENV, soap_lift_frame_rejection
    from ispatialgym.sampling.v2.program_check import fingertips_in_frame

    class Camera:
        width, height, fx = 1920, 1080, 1000

        def project(self, camera, points):
            # The grasp is in frame; the lifted pose falls below the lower bound.
            return torch.tensor([[960., 900. + 900. * float(points[0, 2])]]), torch.tensor([.8])

    sampler = SimpleNamespace(
        gp=SimpleNamespace(variant='inter_object_relation_satisfaction-ontop_large_scale-soap_dispenser-countertop-' + floor_alias),
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
