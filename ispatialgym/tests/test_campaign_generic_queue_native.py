"""Queue transitions still use native angle pairs and every core validation stage."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym.sampling.v2.tasks import articulation_small as small
from ispatialgym.sampling.v2.core import Reject
from scripts.campaign_20261110.tests.test_joint_goals import plan
from scripts.campaign_20261110.tests.test_generic_goal_queue import generic_fixture
from ispatialgym.tests.test_campaign_goal_queue import test_two_groups_share_one_sampler_and_do_not_consume_attempts_when_full


def test_small_queue_refreshes_goal_and_filters_native_pairs_before_start(monkeypatch):
    monkeypatch.setenv(small.CJG.ENV, "queue-current-plan.json")
    monkeypatch.setattr(small.TT, "_count", lambda *args: None)
    current = [plan()]
    native = [{"q_init": value, "q_goal": 0, "key": str(value), "required": [], "passive_from": 0} for value in (.3, .8)]
    selected = []
    sampler = SimpleNamespace(_pairs=native, campaign_joint_plan=lambda: current[0],
        _apply_pair=selected.append, direction="close", lower=0, upper=1,
        gp=SimpleNamespace(variant="fixture"))
    assert small.ArticulationSmall._draw_pair(sampler, np.random.default_rng(1))["q_init"] == .3
    current[0] = replace(plan(), group_id="next", fraction_range=(.5, 1), sha256="c" * 64)
    assert small.ArticulationSmall._draw_pair(sampler, np.random.default_rng(1))["q_init"] == .8
    assert sampler._campaign_joint_active.group_id == "next" and sampler._pairs == native
    current[0] = replace(current[0], joints={"hinge": .2})
    with pytest.raises(Reject, match="campaign_joint_bucket"):
        small.ArticulationSmall._draw_pair(sampler, np.random.default_rng(1))
    assert len(selected) == 2


@pytest.mark.parametrize("task", ["alignment-world_vertical", "alignment-axial", "disembedding",
    "configuration_matching-articulation_open_large_scale", "configuration_matching-block_design"])
def test_generic_groups_retain_core_checks(tmp_path, monkeypatch, task):
    from ispatialgym.tests import test_campaign_goal_queue as test
    monkeypatch.setattr(test.fixtures, "fixture", lambda path, patch: generic_fixture(path, patch, task))
    test_two_groups_share_one_sampler_and_do_not_consume_attempts_when_full(tmp_path, monkeypatch, "held")
