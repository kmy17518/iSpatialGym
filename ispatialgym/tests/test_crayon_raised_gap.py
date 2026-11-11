"""Keep existing raised-release gates while trying the two already legal nextto gaps."""
from types import SimpleNamespace as NS
import pytest
import torch as th
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.tasks.iors_small import IORSSmall


def sampler(enabled=True):
    s=IORSSmall.__new__(IORSSmall);s.gen=lambda key,default=None:enabled
    s.gp=NS(variant='inter_object_relation_satisfaction-nextto_small_scale-crayon-ruler-pedestal_table')
    s.choice=NS(models={'target':NS(category='crayon',model='coapeh'),'partner':NS(category='ruler',model='gajhql')})
    s.relation='nextto';s.pair={'g_NT':.025};s.raised_release_facts=None
    return s


def draw(s):
    return s.raised_nextto_goal(th.tensor([.2,-.02,.75]),th.tensor([0.,0.,0.,1.]),th.zeros((2,3)),
        {'partner_clearance_m':.005,'max_raise_m':.03,'step_m':.001},
        NS(world=lambda depth,lat:th.tensor([depth,lat])),.2,-.1,th.tensor([0.,.01]),th.tensor([-.005,.005]),.015)


def test_clearance_failure_tries_only_existing_legal_gaps_and_retains_limits():
    s=sampler();calls=[]
    def raise_at(pos,quat,hull,settings):
        calls.append((pos.clone(),quat.clone(),dict(settings)))
        c=.0032 if len(calls)<3 else .006
        s.raised_release_facts={'jaw_clearance_m':c,'raise_m':.03}
        if c<settings['partner_clearance_m']:raise Reject('release_sweep','clearance too small')
        return .03
    s.release_raise=raise_at;result=draw(s)
    assert len(calls)==3
    assert [float(x[0][1])for x in calls]==pytest.approx([-.02,-.024,-.027])
    assert result.tolist()==pytest.approx([.2,-.027,.78])
    assert all(x[1].tolist()==[0.,0.,0.,1.]for x in calls)
    assert all(x[2]['max_raise_m']==.03 and x[2]['partner_clearance_m']==.005 for x in calls)
    rec=s.raised_release_facts['gap_recovery'];assert rec['chosen_gap_m']==pytest.approx(.022)
    assert len(rec['prior_failures'])==2


def test_nominal_success_is_unchanged_and_does_not_draw_more():
    s=sampler();calls=[];s.release_raise=lambda *a:calls.append(a)or .012
    assert draw(s).tolist()==pytest.approx([.2,-.02,.762]);assert len(calls)==1


@pytest.mark.parametrize('enabled,reason',[(False,'release_sweep'),(True,'unexpected_failure')])
def test_disabled_or_other_failure_never_tries_gap(enabled,reason):
    s=sampler(enabled);calls=[]
    def fail(*args):calls.append(args);raise Reject(reason,'original')
    s.release_raise=fail
    with pytest.raises(Reject,match='original'):draw(s)
    assert len(calls)==1


def test_exhausted_clearance_preserves_failure_instead_of_waiving_gate():
    s=sampler();calls=[]
    def fail(*args):calls.append(args);s.raised_release_facts={'jaw_clearance_m':.0032};raise Reject('release_sweep','original')
    s.release_raise=fail
    with pytest.raises(Reject,match='original'):draw(s)
    assert len(calls)==3


@pytest.mark.parametrize('bad', ['model','variant','fixed_goal','nonboolean'])
def test_opt_in_rejects_other_models_variants_and_fixed_group_goal(bad):
    s=sampler();s.release_raise=lambda *a:0
    if bad=='model':s.choice.models['target'].model='other'
    if bad=='variant':s.gp.variant='other'
    if bad=='fixed_goal':s._campaign_pair_active=object()
    if bad=='nonboolean':s.gen=lambda *a:'true'
    with pytest.raises(Reject,match='spec'):draw(s)
