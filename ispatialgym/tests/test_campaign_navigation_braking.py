"""Reproduce the measured under03 terminal overshoot without a simulator."""
import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle.motions import Motions, MotionsConfig, PathTracker
from ispatialgym.sampling import campaign_navigation as CN, campaign_runtime as CR

START = (18.2733, -9.4573, 1.184)
GOAL = (18.458812713623047, -7.637513160705566, -.26078751996222865)


def drive(deceleration, start=START, goal=GOAL):
    cfg = MotionsConfig()
    fake = SimpleNamespace(cfg=cfg, dt=1/30, base_pose2d=lambda: start)
    tc = cfg.base_tracker
    ref = Motions._base_reference(fake, [goal], cfg.path_speed_fraction*min(tc.v_max[:2]),
        cfg.path_speed_fraction*tc.v_max[2], a_max=min(tc.a_max[:2]), deceleration=deceleration)
    tracker = PathTracker(ref, fake.dt, tc, max_lag=cfg.path_max_lag)
    cur = list(start)
    direction = [goal[i]-start[i] for i in range(2)]
    length = math.hypot(*direction)
    overshoot, peak_acceleration = 0., 0.
    previous = th.zeros(3)
    for step in range(1000):
        command, done = tracker.step(cur)
        c, s = math.cos(cur[2]), math.sin(cur[2])
        world = th.tensor([c*command[0]-s*command[1], s*command[0]+c*command[1], command[2]])
        if not done:
            peak_acceleration = max(peak_acceleration, float(th.max(th.abs(world[:2]-previous[:2])))/fake.dt)
        previous = world
        cur = [cur[i]+float(world[i])*fake.dt for i in range(3)]
        overshoot = max(overshoot, sum((cur[i]-goal[i])*direction[i]/length for i in range(2)))
        if done:
            break
    assert done
    return overshoot, peak_acceleration, step+1, ref


def test_same_measured_line_stops_within_native_margin_without_changing_controller_limits():
    old, _, old_steps, old_ref = drive(None)
    new, acceleration, new_steps, new_ref = drive(.2)
    assert old > .03  # Recorded physical failure crossed the endpoint by32.7mm.
    assert new < .005  # Original whole-body check retains its10mm margin.
    assert acceleration <= .60001
    assert new_steps <= old_steps+10
    assert th.equal(old_ref[-1], new_ref[-1])


def test_braking_requires_exact_audited_class_and_boolean_opt_in(monkeypatch):
    monkeypatch.setattr(CR, 'active', lambda: False)
    assert CN.terminal_deceleration() is None
    monkeypatch.setattr(CR, 'active', lambda: True)
    template=SimpleNamespace(class_id='inter_object_relation_satisfaction-under_large_scale_03',
        task='inter_object_relation_satisfaction-under_large_scale', scene_setup={'recovery_options':{}})
    monkeypatch.setattr(CR, 'configuration', lambda: (None,template,None,None))
    assert CN.terminal_deceleration() is None
    template.scene_setup['recovery_options']['navigation_terminal_braking']=True
    assert CN.terminal_deceleration()==.2
    template.class_id='inter_object_relation_satisfaction-under_large_scale_02'
    with pytest.raises(ValueError, match='audited'):CN.terminal_deceleration()
    template.scene_setup['recovery_options']['navigation_terminal_braking']='true'
    with pytest.raises(ValueError, match='boolean'):CN.terminal_deceleration()


def test_navigate_changes_only_opted_in_empty_recorded_paths(monkeypatch):
    from ispatialgym.oracle import execute as EX
    seen=[]
    monkeypatch.setattr(CN,'terminal_deceleration',lambda:.2)
    def track_path(waypoints,**kw):
        seen.append(kw)
        yield th.zeros(1)
    motions=SimpleNamespace(track_path=track_path)
    for carried,style in [(None,'recorded'),('target','recorded'),(None,'face_travel')]:
        targets={'waypoints':[GOAL],'carried':carried,'style':style}
        ctx={}
        list(EX.execute_navigate(None,motions,None,targets,ctx))
        assert ('deceleration' in seen[-1]) == (carried is None and style=='recorded')
        assert ('navigation_terminal_deceleration_m_s2' in ctx) == (carried is None and style=='recorded')


@pytest.mark.parametrize('bad',[0.,-1.,float('nan'),float('inf')])
def test_invalid_reference_deceleration_fails_closed(bad):
    with pytest.raises(Exception, match='finite and positive'):drive(bad)


def test_reference_override_cannot_silently_change_another_path_style():
    cfg=MotionsConfig()
    fake=SimpleNamespace(cfg=cfg,dt=1/30,base_pose2d=lambda:START)
    with pytest.raises(Exception,match='requires recorded'):
        Motions._base_reference(fake,[GOAL],.4,.64,style='face_travel',deceleration=.2)


def test_lid_same_recorded_line_keeps_original_endpoint_and_controller_limits():
    start=(9.29119706978382,3.1856777118460653,3.139041022485923)
    goal=(9.0909423828125,3.086188316345215,3.139041022485923)
    old,_,_,old_ref=drive(None,start,goal)
    new,acceleration,_,new_ref=drive(.2,start,goal)
    assert old>.005 and new<.005
    assert acceleration<=.60001
    assert th.equal(old_ref[-1],new_ref[-1])


def test_lid_braking_does_not_enable_other_attachment_classes(monkeypatch):
    template=SimpleNamespace(class_id='configuration_matching-attachment_01',task='configuration_matching-attachment',
        scene_setup={'recovery_options':{'navigation_terminal_braking':True}})
    monkeypatch.setattr(CR,'active',lambda:True)
    monkeypatch.setattr(CR,'configuration',lambda:(None,template,None,None))
    assert CN.terminal_deceleration()==.2
    template.class_id='configuration_matching-attachment_02'
    with pytest.raises(ValueError,match='audited'):CN.terminal_deceleration()
