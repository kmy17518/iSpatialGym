from dataclasses import dataclass
from ispatialgym.sampling.campaign_axial_approach import choose_push_with_pre_fallback, short_pre_fallback_enabled


def test_unverified_fallback_is_opt_in_and_campaign_scoped(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG',raising=False)
    monkeypatch.delenv('ISPATIALGYM_AXIAL_SHORT_PRE_FALLBACK',raising=False)
    assert not short_pre_fallback_enabled()
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','exact.json')
    assert not short_pre_fallback_enabled()
    monkeypatch.setenv('ISPATIALGYM_AXIAL_SHORT_PRE_FALLBACK','1')
    assert short_pre_fallback_enabled()
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG')
    assert not short_pre_fallback_enabled()


@dataclass
class Proposal:
    label: str
    rank: int
    payload: object
    chain: list
    q_first: object = None
    source: str = 'ik'


def proposals():
    return [Proposal('turn'+str(i), i, i, [(.1,i),(0,i),('mid',i),('goal',i)],
                     q_first='original witness') for i in range(2)]


def test_original_success_and_disabled_path_remain_exact():
    for enabled, winner in ((True, object()), (False, None)):
        rows=proposals(); decision={'options':['original']}; calls=[]
        def choose(value): calls.append(value); return winner,decision
        def forbidden(*args): raise AssertionError('unneeded alternate pose')
        result=choose_push_with_pre_fallback(rows,choose,forbidden,enabled=enabled)
        assert result==(winner,decision) and result[1] is decision
        assert len(calls)==1 and calls[0] is rows


def test_shorter_proposal_rechecks_full_chain_without_contact_or_goal_changes():
    rows=proposals(); calls=[]; winner=object()
    def choose(value):
        calls.append(value)
        return (None,{'failed':'original exact IK'}) if len(calls)==1 else (winner,{'passed':'same evaluator'})
    result,proof=choose_push_with_pre_fallback(rows,choose,
        lambda contact,distance:(distance,contact[1]),enabled=True)
    assert result is winner and len(calls)==2
    for old,new in zip(rows,calls[1]):
        assert new.chain[0]==(.05,old.payload)
        assert all(a is b for a,b in zip(old.chain[1:],new.chain[1:]))
        assert new.rank==old.rank and new.payload==old.payload
        assert new.q_first is None and new.source=='ik'
        assert old.chain[0]==(.1,old.payload) and old.q_first=='original witness'
    assert proof['pre_approach_fallback']['original_decision']=={'failed':'original exact IK'}


def test_failed_shorter_check_remains_a_rejection():
    calls=[]
    def choose(value): calls.append(value); return None,{'why':'collision'}
    chosen,proof=choose_push_with_pre_fallback(proposals(),choose,
        lambda contact,distance:(distance,contact[1]),enabled=True)
    assert chosen is None and len(calls)==2 and proof['why']=='collision'
