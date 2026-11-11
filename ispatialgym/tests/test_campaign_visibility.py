from types import SimpleNamespace
import pytest
from ispatialgym.sampling import campaign_visibility as C


def template(value=False, class_id=C.CLASS_ID, task=C.TASK):
    return SimpleNamespace(class_id=class_id, task=task, scene_setup={'recovery_options': {C.OPTION: value}})


def test_scope_and_typed_opt_in():
    assert C.required(template(True))
    assert not C.required(template())
    for value in ('true', 1, None):
        with pytest.raises(ValueError): C.required(template(value))
    with pytest.raises(ValueError): C.required(template(True, class_id='another_class'))
    with pytest.raises(ValueError): C.required(template(True, task='another_task'))


@pytest.mark.parametrize('missing', ['enabled','judged','checked_program','camera','exempt'])
def test_missing_native_evidence_fails_closed(missing):
    inputs=dict(enabled=True,exempt=False,judged=True,checked_program=True,camera=True)
    C.preflight(**inputs)
    inputs[missing]=not inputs[missing]
    with pytest.raises(ValueError): C.preflight(**inputs)


def test_measured_occlusion_remains_a_failure_for_original_screen():
    report={'objects': {'target': {'status':'ok','occluded':.6147}}, 'occluded':True}
    C.verify_report(report)
    assert report['occluded'] is True
    report['occluded']=False;report['objects']['target']['occluded']=.4128
    C.verify_report(report)


@pytest.mark.parametrize('report', [{}, {'error':'camera render failed','occluded':False},
    {'occluded':False,'objects':{'target':{'status':'out_of_frame'}}},
    {'occluded':False,'objects':{'target':{'status':'ok','occluded':float('nan')}}},
    {'occluded':False,'objects':{'target':{'status':'ok','occluded':True}}}])
def test_failed_prediction_cannot_pass_as_unoccluded(report):
    with pytest.raises(ValueError): C.verify_report(report)


def test_required_screen_cannot_publish_without_camera(monkeypatch):
    from ispatialgym.sampling.v2 import core
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.eval import goal_occlusion as G
    sampler=object.__new__(core.SamplerV2)
    sampler.subtask=C.TASK
    sampler.objects={'target':SimpleNamespace(name='target_spatula')}
    # dict.fromkeys in the core screen requires hashable simulator objects.
    class Obj:
        name='target_spatula'
    sampler.objects={'target':Obj()}
    sampler.world=SimpleNamespace(cam=None)
    sampler.final_configuration=lambda: ('q','base','source')
    removed=[];withheld=[]
    sampler._count_withheld=lambda:withheld.append(True)
    monkeypatch.setattr(C,'required',lambda:True)
    monkeypatch.setattr(G,'enabled',lambda:True)
    monkeypatch.setattr(core,'remove_instance_files',removed.append)
    with pytest.raises(Reject,match='camera'):
        sampler.check_goal_occlusion(SimpleNamespace(summary={}),'/candidate/tro.json')
    assert removed==['/candidate/tro.json'] and withheld==[True]


def test_under02_requires_measured_goal_without_releasing_other_large_classes():
    task='inter_object_relation_satisfaction-under_large_scale'
    assert C.required(template(True, class_id='inter_object_relation_satisfaction-under_large_scale_02',task=task))
    for suffix in ['01','03']:
        with pytest.raises(ValueError):
            C.required(template(True,class_id='inter_object_relation_satisfaction-under_large_scale_'+suffix,task=task))
    with pytest.raises(ValueError):
        C.required(template(True,class_id='inter_object_relation_satisfaction-under_large_scale_02',task=C.TASK))


def test_inside02_requires_same_measured_partner_visibility_policy():
    from types import SimpleNamespace
    t=SimpleNamespace(class_id='inter_object_relation_satisfaction-inside_02',
        task='inter_object_relation_satisfaction-inside_small_scale',
        scene_setup={'recovery_options':{C.OPTION:True}})
    assert C.required(t)
    C.verify_report({'objects':{'target':{'status':'ok','occluded':1.,'judged':False},
        'partner':{'status':'ok','occluded':.6733}},'occluded':True})
    # Verification does not turn the passing-pose but occluded jar into a pass;
    # the original GO report and unchanged threshold remain authoritative.
    from ispatialgym.eval import goal_occlusion as GO
    result=GO.report({'target':{'status':'ok','occluded':1.,'judged':False},
        'partner':{'status':'ok','occluded':.6733}},method='measured')
    assert result['occluded'] and result['threshold']==.5 and result['worst']=='partner'


def test_mouse02_reference_occlusion_requires_new_measured_native_goal():
    task='inter_object_relation_satisfaction-ontop_large_scale'
    assert C.required(template(True,class_id=task+'_02',task=task))
    for suffix in ['01','03']:
        with pytest.raises(ValueError): C.required(template(True,class_id=task+'_'+suffix,task=task))
    report={'objects':{'mouse':{'status':'ok','occluded':.8992}},'occluded':True}
    C.verify_report(report)
    from ispatialgym.eval import goal_occlusion as GO
    result=GO.report(report['objects'],method='measured')
    assert result['occluded'] and result['threshold']==.5


def test_required_native_screen_retains_real_camera_before_sampling(monkeypatch):
    from ispatialgym.sampling import sample_isg as S
    monkeypatch.setattr(S,'fast_on',lambda name:True)
    monkeypatch.setattr(C,'required',lambda:False)
    assert not S.world_cameras_required('inter_object_relation_satisfaction-inside_small_scale')
    assert S.world_cameras_required('camera_relocalization-standard')
    assert S.world_cameras_required(None)
    monkeypatch.setattr(C,'required',lambda:True)
    assert S.world_cameras_required('inter_object_relation_satisfaction-inside_small_scale')
    assert S.world_cameras_required('inter_object_relation_satisfaction-under_large_scale')
    monkeypatch.setattr(C,'required',lambda:False)
    monkeypatch.setattr(S,'fast_on',lambda name:False)
    assert S.world_cameras_required('inter_object_relation_satisfaction-inside_small_scale')
