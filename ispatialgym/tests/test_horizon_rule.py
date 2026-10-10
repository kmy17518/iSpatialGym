"""The lab evaluator's step budget (``lab_tools.horizon``): ``--horizon auto`` = the demo's length for a ground-truth replay,
twice the longest training demo of the alias for a policy; ``--max-steps auto`` = ceil(1.6 x horizon) + 100."""

from __future__ import annotations

import argparse
import math
import os
from fractions import Fraction

import pytest

from lab_tools import horizon as H


def _dataset(root, episodes):
    import pandas as pd

    d = os.path.join(root, "meta", "episodes", "chunk-000")
    os.makedirs(d)
    pd.DataFrame({"episode_index": [e for e, _, _ in episodes], "tasks": [[t] for _, t, _ in episodes],
                  "length": [n for _, _, n in episodes]}).to_parquet(os.path.join(d, "file-000.parquet"))
    return str(root)


def test_replay_horizon_is_the_demo_length(tmp_path):
    root = _dataset(tmp_path, [(100, "blend", 741), (101, "blend", 800)])
    assert H.demo_length(root, 100) == 741
    b = H.budget("auto", None, "replay", demo_len=H.demo_length(root, 100))
    assert b == {"horizon": 741, "horizon_rule": "demo_length", "max_steps": 1286, "max_steps_rule": "ceil(1.6 x horizon) + 100"}


def test_policy_horizon_is_twice_the_longest_training_demo(tmp_path):
    train = _dataset(tmp_path, [(0, "banana", 120), (1, "banana", 141), (2, "can", 500), (3, "banana", 99)])
    assert H.policy_horizon("banana", train) == 282
    b = H.budget("auto", None, "policy", task="banana", train_root=train)
    assert (b["horizon"], b["horizon_rule"], b["max_steps"]) == (282, "2x_longest_training_demo", 552)
    with pytest.raises(KeyError):
        H.policy_horizon("dishwasher", train)


def test_explicit_numbers_keep_the_fixed_protocol():
    assert H.budget(1500, None, "replay", demo_len=741) == {"horizon": 1500, "horizon_rule": "fixed", "max_steps": 6000, "max_steps_rule": "fixed"}
    assert H.budget(1500, "auto", "replay", demo_len=741)["max_steps"] == 2500
    assert H.budget("auto", 4000, "replay", demo_len=741)["max_steps"] == 4000
    assert H.spec("auto") == "auto" and H.spec("1500") == 1500
    for bad in ("0", "-3", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            H.spec(bad)


def test_max_steps_formula():
    for h in range(1, 6001):
        assert H.max_steps_for(h) == math.ceil(Fraction(16, 10) * h) + 100


@pytest.mark.skipif(not os.path.isdir(os.path.join(H.TRAIN_ROOT, "meta", "episodes")), reason="datasets/isg-init not downloaded")
def test_policy_horizons_of_the_battery_tasks():
    expected = {"mental_rotation-banana": 282, "object_scaling-canned_food": 282, "alignment-axial-board_game-breakfast_table": 1746,
                "configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table": 2540,
                "camera_relocalization-standard": 2564, "configuration_matching-articulation_open_large_scale-dishwasher": 3068}
    assert {t: H.policy_horizon(t) for t in expected} == expected
