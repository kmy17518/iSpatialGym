from dataclasses import replace
import pytest
from scripts.campaign_20261110 import contracts as c
from scripts.campaign_20261110.bind_standard_disembedding_pool import make_pool
from scripts.campaign_20261110.tests.test_binding_pools import pool


def source():
    p = pool()
    return c.Template('standard', 'disembedding', p.class_id, 'eval_standard', p.scene, p.room,
                      p.fixed_objects + tuple(replace(row.object, instance='obj'+str(i)) for i,row in enumerate(p.candidates)),
                      p.source_definition_sha256)


def test_full_declared_ready_pool_and_exact_active_selection():
    definition = {'roles': {'target': {'category': 'apple'}, 'distractors': {'category': ['shampoo', 'tissue']}}}
    maximum, active = make_pool(source(), definition)
    assert len(maximum.candidates) == 3 and len(active) == 3
    assert maximum.role_limits['distractor'] == (1, 48)
    definition['roles']['distractors']['category'].append('unrepresented')
    with pytest.raises(c.ContractError, match='complete declared distractor'):
        make_pool(source(), definition)


def test_fixed_layout_cannot_gain_copies_or_change_selection():
    definition = {'roles': {'target': {'category': 'apple'}, 'distractors': {'category': ['shampoo', 'tissue']}},
                  'generation': {'fixed_layout': {'enabled': True}}}
    maximum, active = make_pool(source(), definition)
    assert maximum.fixed_layout and maximum.role_limits['distractor'] == (2, 2)
    with pytest.raises(c.ContractError):
        maximum.check_selection({**active, 'extra': active['obj1']})
