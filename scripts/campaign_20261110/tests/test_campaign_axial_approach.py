from dataclasses import dataclass
from ispatialgym.sampling.campaign_axial_approach import choose_push_with_pre_fallback, short_pre_fallback_enabled, short_pre_only_enabled


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


def test_short_only_requires_explicit_campaign_opt_in(monkeypatch):
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG',raising=False)
    monkeypatch.setenv('ISPATIALGYM_AXIAL_SHORT_PRE_ONLY','1')
    assert not short_pre_only_enabled()
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','exact.json')
    assert short_pre_only_enabled()
    monkeypatch.delenv('ISPATIALGYM_AXIAL_SHORT_PRE_ONLY')
    assert not short_pre_only_enabled()


def test_short_only_does_not_pollute_solver_with_original_failures():
    for winner in (object(),None):
        rows=proposals();calls=[]
        def choose(value):
            calls.append(value)
            assert all(p.chain[0][0]==.05 for p in value)
            return winner,{'options':[{'ok':winner is not None}]}
        chosen,proof=choose_push_with_pre_fallback(rows,choose,
            lambda contact,distance:(distance,contact[1]),enabled=True,short_only=True)
        assert chosen is winner and len(calls)==1
        for old,new in zip(rows,calls[0]):
            assert old.label==new.label and old.rank==new.rank and old.payload==new.payload
            assert all(a is b for a,b in zip(old.chain[1:],new.chain[1:]))
            assert new.q_first is None and old.q_first=='original witness'
        assert proof['pre_approach_program']['original_checked_first']is False


def test_program_witness_selection_is_exact_class_opt_in(monkeypatch):
    from types import SimpleNamespace
    from ispatialgym.sampling import campaign_runtime
    from ispatialgym.sampling.campaign_axial_approach import program_witness_selection_enabled
    monkeypatch.setenv('ISPATIALGYM_AXIAL_PROGRAM_WITNESS_SELECTION','1')
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_CONFIG',raising=False)
    assert not program_witness_selection_enabled()
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG','exact.json')
    for cls,expected in [('alignment-axial_04',True),('alignment-axial_01',False)]:
        monkeypatch.setattr(campaign_runtime,'configuration',lambda:(None,SimpleNamespace(class_id=cls)))
        assert program_witness_selection_enabled() is expected


def test_program_endpoint_rejection_keeps_prior_solutions_and_tolerance():
    from types import SimpleNamespace
    import torch
    from ispatialgym.sampling.campaign_axial_approach import polish_program_endpoints
    original={('right','pre'):torch.tensor([1.]),('right','goal'):torch.tensor([2.])}
    calls=[]
    def polish(q,p,arm,ignore):
        calls.append((p,arm,ignore))
        return q+10, .00047 if p=='pre' else .004716873
    checker=SimpleNamespace(solutions=dict(original),_pose_key=lambda p:p,polish_witness=polish,WITNESS_RESIDUAL_MAX=.002)
    ok,why=polish_program_endpoints(checker,['pre','middle','goal'],arm='right',ignored=['support','target'])
    assert not ok and '4.717mm' in why and '2.000mm' in why
    assert all(torch.equal(checker.solutions[k],q)for k,q in original.items())
    assert calls==[('pre','right',['support','target']),('goal','right',['support','target'])]
    checker.polish_witness=lambda q,*a,**kw:(q+10,.00047)
    assert polish_program_endpoints(checker,['pre','goal'],arm='right',ignored=[])[0]
    assert all(torch.equal(checker.solutions[k],q+10)for k,q in original.items())


def test_path_selection_requires_both_exact_scoped_options(monkeypatch):
    import ispatialgym.sampling.campaign_axial_approach as A
    monkeypatch.setenv('ISPATIALGYM_AXIAL_PATH_SELECTION', '1')
    monkeypatch.setattr(A, 'program_witness_selection_enabled', lambda: False)
    assert not A.path_selection_enabled()
    monkeypatch.setattr(A, 'program_witness_selection_enabled', lambda: True)
    assert A.path_selection_enabled()
    monkeypatch.delenv('ISPATIALGYM_AXIAL_PATH_SELECTION')
    assert not A.path_selection_enabled()


def test_wider_stance_search_requires_strict_program_path_scope(monkeypatch):
    import ispatialgym.sampling.campaign_axial_approach as A
    monkeypatch.setenv('ISPATIALGYM_AXIAL_CHECKED_STANCE_SEARCH', '1')
    monkeypatch.setattr(A, 'path_selection_enabled', lambda: False)
    assert A.checked_stance_limits() == (64, 6)
    monkeypatch.setattr(A, 'path_selection_enabled', lambda: True)
    assert A.checked_stance_limits() == (256, 24)
    monkeypatch.delenv('ISPATIALGYM_AXIAL_CHECKED_STANCE_SEARCH')
    assert A.checked_stance_limits() == (64, 6)


def test_program_path_rejects_failed_original_and_actual_branch_without_replacing_witness(monkeypatch):
    from types import SimpleNamespace
    import torch
    from ispatialgym.sampling.v2 import program_check as PC
    from ispatialgym.oracle import execute as E
    from ispatialgym.sampling.campaign_axial_approach import check_program_push_path
    old = torch.tensor([1.]); alternate = torch.tensor([2.])
    monkeypatch.setattr(E, 'push_speed_of', lambda contact: contact.extra['push_speed_m_s'])
    monkeypatch.setattr(PC, '_push_stop_frac_arc', lambda: .2)
    gp = {'tolerances': {'orientation': .1}, 'push_arc_replay_lag_rad': .08, 'push_speed_m_s': .03}
    checker = SimpleNamespace(solutions={('right', 'pre'): old}, _pose_key=lambda p: p)
    sampler = SimpleNamespace(ikc=checker, gp=SimpleNamespace(gen=lambda k: gp.get(k)), support_objects=lambda: ['support'])
    calls = []; branches = []
    def replay(*args, **kw):
        calls.append((args, kw))
        return False, 'retreat joint jump', {'retreat': {'fail_index': 1}}
    monkeypatch.setattr(PC, 'replay_arc_push', replay)
    monkeypatch.setattr(PC, '_rewitness_flipped_push', lambda *a, **kw: branches.append(a))
    ok, why, report = check_program_push_path(sampler, ['pre','init','mid','goal'], 'stance', 'direction', 'target', 'support')
    assert not ok and 'joint jump' in why and not branches
    assert checker.solutions[('right','pre')] is old
    args, kw = calls[0]
    assert args[1:2] == ('right',) and args[3:8] == ('pre', ['init','mid','goal'], 'direction', 'target', ['support'])
    assert args[8] == ['support','target'] and abs(args[9]-.02)<1e-12 and args[10]==.1
    assert kw == {'speed_m_s':.03,'line':False,'lag_allowance_rad':.08}
    def branch(*args):
        assert args[3] == 'right' and args[5] == 0.
        args[8]['pre'] = alternate
        return False, 'stance branch retreat stalls'
    monkeypatch.setattr(PC, 'replay_arc_push', lambda *a, **kw: (True, 'arc and retreat pass', {}))
    monkeypatch.setattr(PC, '_rewitness_flipped_push', branch)
    ok, why, _ = check_program_push_path(sampler, ['pre','init','mid','goal'], 'stance', 'direction', 'target', 'support')
    assert not ok and 'stance branch' in why and checker.solutions[('right','pre')] is old
    def passing_branch(*args):
        args[8]['pre'] = alternate
        return True, 'native checked replacement'
    monkeypatch.setattr(PC, '_rewitness_flipped_push', passing_branch)
    assert check_program_push_path(sampler, ['pre','init','mid','goal'], 'stance', 'direction', 'target', 'support')[0]
    assert torch.equal(checker.solutions[('right','pre')], alternate)
    assert checker.solutions[('right','pre')] is not alternate
