import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.campaign_carton_contacts import press_slide_extensions
from ispatialgym.sampling import campaign_runtime as CR
from scripts.campaign_20261110.contracts import ContractError


def test_residual_checks_use_native_tracking_clamp_flap_and_restore_nominal():
    start = (th.tensor([.2,0.,0.]), th.tensor([0.,0.,0.,1.]))
    middle, goal = C.orbit_waypoints(start, th.zeros(3), C.quat_from_axis_angle(th.tensor([0.,0.,1.]), -.5), n=2, frame='world')
    slides=[]; states=[]
    def track(q, old, new, label):
        if label.startswith('slide'):
            slides.append((q.clone(),new,label))
        return q+1, None
    kw=dict(track=track,set_flap=states.append,q_press=th.zeros(7),press_waypoints=[start,middle,goal],
            direction=th.tensor([0.,-1.,0.]),slide_pose=start,q_other=0.,angle=.5,lower=-.55,tolerance=.1)
    error, cap = press_slide_extensions(**kw)
    assert error is None and cap == pytest.approx(.2)
    assert len(slides)==2 and all(row[1] is start for row in slides)
    assert states == [-.55,-.55,-.5]
    # An unreachable slide rejects the candidate and still restores the nominal
    # state; it does not replace the failed native check with a new branch.
    states.clear()
    def fail(q, old, new, label):
        return (None,'native slide joint limit') if label.startswith('slide') else (q,None)
    error, _ = press_slide_extensions(**{**kw,'track':fail})
    assert error=='native slide joint limit' and states[-1]==-.5
    assert th.equal(kw['q_press'],th.zeros(7))


def test_carton_screen_requires_the_exact_bool_class_option(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG',raising=False)
    assert not CR.contact_recovery('carton_press_slide_envelope')
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','/unused')
    template=SimpleNamespace(class_id='configuration_matching-articulation_open_small_scale_02',
        task='configuration_matching-articulation_open_small_scale',scene_setup={'recovery_options':{'carton_press_slide_envelope':True}})
    monkeypatch.setattr(CR,'configuration',lambda:(None,template,None,None))
    assert CR.contact_recovery('carton_press_slide_envelope')
    template.class_id='configuration_matching-articulation_open_small_scale_01'
    with pytest.raises(ContractError,match='another audited class'):
        CR.contact_recovery('carton_press_slide_envelope')
