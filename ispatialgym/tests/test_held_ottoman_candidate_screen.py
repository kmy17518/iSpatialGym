from types import SimpleNamespace
import pytest
from ispatialgym.sampling import campaign_large_recovery as recovery


def sampler(flag=False):
    return SimpleNamespace(
        gp=SimpleNamespace(gen=lambda name, default=None: flag,
            variant='inter_object_relation_satisfaction-touching_large_scale-ottoman-sofa-floor'),
        choice=SimpleNamespace(models={'target': SimpleNamespace(category='ottoman', model='xcuyiw')}),
        partner=SimpleNamespace(category='sofa', model='mnfbbh'))


def test_disabled_does_not_require_model_context_or_run_checker(monkeypatch):
    s = SimpleNamespace(gp=SimpleNamespace(gen=lambda *args: False))
    monkeypatch.setattr(recovery, '_candidate_program_rejection', lambda *args: pytest.fail('disabled checker'))
    assert recovery.ottoman_candidate_program_rejection(s, object()) is None


@pytest.mark.parametrize('field,value', [('target', 'other'), ('partner', 'hiphpn'), ('variant', 'another_class')])
def test_wrong_task_or_unreserved_model_fails_closed(field, value):
    s = sampler(True)
    if field == 'target': s.choice.models['target'].model = value
    elif field == 'partner': s.partner.model = value
    else: s.gp.variant = value
    with pytest.raises(ValueError, match='exact held target and reserved partner'):
        recovery.ottoman_candidate_program_screen(s)


@pytest.mark.parametrize('value', ['true', 1, None])
def test_only_explicit_boolean_is_accepted(value):
    with pytest.raises(ValueError, match='must be boolean'):
        recovery.ottoman_candidate_program_screen(sampler(value))


def test_exact_opt_in_preserves_original_checker_result(monkeypatch):
    s, draft = sampler(True), object()
    calls = []
    def check(actual, proposed):
        calls.append((actual, proposed))
        return 'program_check_failed: retreat 2.35mm; polished witness self-collides'
    monkeypatch.setattr(recovery, '_candidate_program_rejection', check)
    assert 'self-collides' in recovery.ottoman_candidate_program_rejection(s, draft)
    assert calls == [(s, draft)]


@pytest.mark.parametrize('enabled,all_rejected', [(False, False), (True, False), (True, True)])
def test_full_program_rejection_keeps_checked_alternative_stances_searchable(monkeypatch, enabled, all_rejected):
    import torch
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2.tasks import iors_large_nextto as nextto

    s = sampler(enabled)
    s.gp.gen = lambda name, default=None: enabled if name == recovery.OTTOMAN_CANDIDATE_SCREEN else default
    attempted, screened, resets = [], [], []
    s._check_deadline = lambda label: None
    s.ikc = SimpleNamespace(stances_collision_free=lambda candidates: torch.ones(len(candidates), dtype=torch.bool))
    s.sidecar = SimpleNamespace(add_stance=lambda stance: None, reset_instance=lambda: resets.append(True))
    s.record_leg_a_b = lambda start, end: 'checked_leg'
    stances = [SimpleNamespace(name=name, base_pose2d=(0., 0., 0.), translate=.5, tilt=0.,
        with_trunk=lambda *args: SimpleNamespace()) for name in ('first', 'second')]
    cands = [(st, {'reach_margin': .2, 'arm': 'right'}) for st in stances]
    monkeypatch.setattr(nextto, 'knob', lambda *args: 2)
    monkeypatch.setattr(nextto, 'shifted_pose2d', lambda *args: (1., 0., 0.))
    monkeypatch.setattr(nextto, '_sweep_ok', lambda *args: True)
    monkeypatch.setattr(nextto, '_stance_a', lambda *args: 'checked_start')
    monkeypatch.setattr(nextto, '_record', lambda sampler, start, stance, *args: SimpleNamespace(name=stance.name, summary={}))

    def chain(stance, info, reasons):
        attempted.append(stance.name)
        return {'q': [0.]}

    def screen(sampler, draft):
        screened.append(draft.name)
        return 'program_check_failed: retreat self-collision' if all_rejected or draft.name == 'first' else None

    monkeypatch.setattr(recovery, '_candidate_program_rejection', screen)
    stats = {}
    args = (s, None, cands, chain, None, '+x', 0., None, None, None, 1., .1, .2, None, stats)
    if all_rejected:
        with pytest.raises(Reject, match='retreat self-collision'):
            nextto._complete(*args)
        assert len(resets) == 2 and stats['candidate_program_rejected'] == 2
    else:
        result = nextto._complete(*args)
        assert result.name == ('second' if enabled else 'first')
        if enabled:
            assert result.summary['campaign_early_program_screen']['final_check_required'] is True
            assert len(resets) == 1 and stats['candidate_program_rejected'] == 1
        else:
            assert not resets and not result.summary
    assert attempted == (['first', 'second'] if enabled else ['first'])
    assert screened == (['first', 'second'] if enabled else [])
