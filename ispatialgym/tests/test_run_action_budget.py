from dataclasses import asdict
from types import SimpleNamespace as NS

import pytest

from ispatialgym.eval.instances import step_budget
from ispatialgym.oracle.qa import Monitors
from ispatialgym.oracle.run_isg import build_parser, qa_config_for_budget


def test_measured_action_budget_changes_only_timeout_and_still_enforces_episode_cap():
    budget = step_budget(None, max_steps=5000)
    old = qa_config_for_budget(budget, 'oracle')
    args = build_parser().parse_args(['--max-steps', '5000', '--max-action-steps', '4000'])
    new = qa_config_for_budget(budget, 'oracle', args.max_action_steps)
    assert {k for k in asdict(old) if asdict(old)[k] != asdict(new)[k]} == {'max_action_steps'}
    monitor = NS(cfg=old, action=NS(name='push_close'), action_steps=3012, episode_steps=3356,
                 _verdict=lambda *args, **kw: args)
    assert Monitors._check_timeout(monitor)[0] == 'timeout'
    monitor.cfg = new
    assert Monitors._check_timeout(monitor) is None
    monitor.action_steps = 4001
    assert '4000 steps' in Monitors._check_timeout(monitor)[1]
    monitor.action_steps = 3012; monitor.episode_steps = 5046
    assert 'episode exceeded 5045' in Monitors._check_timeout(monitor)[1]


@pytest.mark.parametrize('value', [0, -1, 5001, 2.5, True])
def test_explicit_action_budget_must_fit_finite_program_budget(value):
    with pytest.raises(ValueError, match='positive integer no larger'):
        qa_config_for_budget(step_budget(None, max_steps=5000), 'oracle', value)


@pytest.mark.parametrize('mode', ['oracle', 'replay', 'policy'])
def test_omitted_action_budget_preserves_each_mode_default(mode):
    b = step_budget(None, max_steps=5000, mode=mode, demo_settle_steps=45)
    cfg = qa_config_for_budget(b, mode)
    assert cfg.max_action_steps == (3000 if mode == 'oracle' else b['episode_cap'])
    assert cfg.max_episode_steps == b['episode_cap']
