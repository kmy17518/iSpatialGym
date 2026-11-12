from types import SimpleNamespace as NS
import copy
import pytest
from ispatialgym.sampling import campaign_nextto_contact_proposal as P


def template():
    hint = {k: 'a'*64 for k in ('tro_sha256', 'summary_sha256', 'sidecar_sha256', 'demo_sha256', 'own_replay_gate_sha256')}
    hint.update(face='+x', hands=2, height_m=.45, spacing_m=.3, stand_off_m=.55, translate=.75, tilt=-.8)
    return NS(class_id=P.CLASS_ID, split='eval_standard', task='inter_object_relation_satisfaction-nextto_large_scale',
              objects=[NS(role='target', category='swivel_chair', model='iiihwn')], scene_setup={P.FIELD: hint})


def candidate(height, spacing, t, k, so=.55):
    return (NS(stand_off=so, translate=t, tilt=k), {'pair': {'height':height, 'spacing':spacing}, 'arms':['right','left']})


def test_default_and_other_face_leave_native_order_and_objects_unchanged():
    t=template(); t.scene_setup={}; assert P.checked_hint(t) is None
    c=[candidate(.66,.5,.25,-1), candidate(.45,.3,.75,-.8)]
    assert P.prioritize(c,'+x',None) is c
    assert P.prioritize(c,'-x',P.checked_hint(template())) is c


def test_rank_only_two_existing_exact_contacts_and_preserve_every_fallback():
    c=[candidate(.66,.5,.25,-1),candidate(.45,.3,.5,-1),candidate(.45,.3,.75,-.8),candidate(.45,.3,.95,-.3),candidate(.49,.34,.5,-1)]
    before=copy.deepcopy(c); out=P.prioritize(c,'+x',P.checked_hint(template()))
    assert out == [c[2],c[1],c[0],c[3],c[4]]
    assert c == before and sorted(map(id,c)) == sorted(map(id,out))


@pytest.mark.parametrize('key,value',[('class_id','other'),('split','train'),('task','other')])
def test_scope_is_exact(key,value):
    t=template();setattr(t,key,value)
    with pytest.raises(ValueError): P.checked_hint(t)


@pytest.mark.parametrize('key,value',[('height_m',float('nan')),('spacing_m',0),('tilt',True),('own_replay_gate_sha256','missing')])
def test_malformed_or_unproven_hint_rejected(key,value):
    t=template();t.scene_setup[P.FIELD][key]=value
    with pytest.raises(ValueError): P.checked_hint(t)


def test_wrong_model_rejected():
    t=template();t.objects[0].model='other'
    with pytest.raises(ValueError): P.checked_hint(t)


def test_missing_matching_contact_preserves_order():
    c=[candidate(.66,.5,.25,-1)]
    assert P.prioritize(c,'+x',P.checked_hint(template())) is c


def test_native_hook_is_after_body_screen_and_before_unchanged_ik_cap():
    from pathlib import Path
    s=(Path(__file__).parents[1]/'sampling/v2/tasks/iors_large_nextto.py').read_text()
    start=s.index('def _complete(');s=s[start:s.index('def _push_chain(',start)]
    assert s.index('stances_collision_free') < s.index('prioritize(cands') < s.index('cands[: int(knob(s, "push_ik_max"))]')
    for name in ('_sweep_ok','_stance_a','record_leg_a_b'):
        assert s.index(name)>s.index('prioritize(cands')


def test_start_search_keeps_physical_rejections_and_reuses_only_checked_result():
    class Rejected(Exception): pass
    calls=[]; valid=object()
    def sample():
        calls.append(1)
        if len(calls)<12: raise Rejected('camera/floor/body/trunk check')
        return valid
    cached=P.CheckedInitialStance(12)
    assert cached.get(sample,Rejected) is valid
    assert cached.get(sample,Rejected) is valid
    assert len(calls)==12


def test_failed_start_budget_cannot_restart_for_next_contact_or_hide_other_errors():
    class Rejected(Exception): pass
    calls=[]
    def sample(): calls.append(1); raise Rejected('physical rejection')
    cached=P.CheckedInitialStance(12)
    for _ in range(2):
        with pytest.raises(Rejected): cached.get(sample,Rejected)
    assert len(calls)==12
    with pytest.raises(RuntimeError):
        P.CheckedInitialStance(12).get(lambda: (_ for _ in ()).throw(RuntimeError('native failure')),Rejected)


def test_default_starts_are_resampled_and_opt_in_requires_proven_exact_scope():
    cached=P.CheckedInitialStance(1); a=object(); b=object()
    assert cached.get(lambda:a,ValueError) is a
    assert cached.get(lambda:b,ValueError) is b
    t=template(); assert P.initial_stance_batches(t)==1
    t.scene_setup['recovery_options']={'nextto_reuse_initial_stance':True}
    assert P.initial_stance_batches(t)==12
    t.class_id='other'
    with pytest.raises(ValueError): P.initial_stance_batches(t)
