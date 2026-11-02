"""Step budgets of the lab evaluator's streams: ``--horizon`` / ``--max-steps`` of replay_episodes.py, eval_episodes.py and
vec_eval.py.

* ``--horizon auto`` (the default): a ground-truth replay streams its whole demo (the episode's ``length`` in the dataset's
  episode metadata); a policy rollout gets twice the longest training demo of its variant (``datasets/isg-init``'s episode
  metadata, columns ``tasks`` and ``length``; never the test set). A number is the fixed protocol (the sanity battery
  passes 1500).
* ``--max-steps auto`` = ceil(1.6 x horizon) + 100: the settle windows add up to 60 % of the streamed steps. Without
  ``--max-steps``: auto with ``--horizon auto``, the fixed protocol's 6000 with a number.
"""

from __future__ import annotations

import argparse
import functools
import glob
import os
from typing import Any, Dict, Optional, Union

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_ROOT = os.path.join(WT, "datasets", "isg-init")
FIXED_MAX_STEPS = 6000
Spec = Union[str, int]


def spec(value: str) -> Spec:
    """argparse type of ``--horizon`` / ``--max-steps``: ``auto`` or a positive number of steps."""
    if value == "auto":
        return value
    try:
        n = int(value)
    except ValueError:
        n = 0
    if n <= 0:
        raise argparse.ArgumentTypeError(f"expected 'auto' or a positive number of steps, got {value!r}")
    return n


@functools.lru_cache(maxsize=None)
def _episodes(root: str):
    import pandas as pd

    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no episode metadata under {root}/meta/episodes")
    return pd.concat([pd.read_parquet(f, columns=["episode_index", "tasks", "length"]) for f in files], ignore_index=True)


def demo_length(root: str, episode: int) -> int:
    """The recorded length of ``episode`` in the dataset at ``root``."""
    df = _episodes(os.path.abspath(root))
    hit = df[df["episode_index"] == int(episode)]
    if not len(hit):
        raise KeyError(f"episode {episode} not in {root}")
    return int(hit.iloc[0]["length"])


def policy_horizon(task: str, train_root: str = TRAIN_ROOT) -> int:
    """Twice the longest training demo of the variant ``task``."""
    df = _episodes(os.path.abspath(train_root))
    lengths = [int(n) for tasks, n in zip(df["tasks"], df["length"]) if task in list(tasks)]
    if not lengths:
        raise KeyError(f"no training episode of {task!r} in {train_root}")
    return 2 * max(lengths)


def max_steps_for(horizon: int) -> int:
    """ceil(1.6 x horizon) + 100."""
    return (16 * int(horizon) + 9) // 10 + 100


def budget(horizon: Spec, max_steps: Optional[Spec], kind: str, task: Optional[str] = None, demo_len: Optional[int] = None,
           train_root: str = TRAIN_ROOT) -> Dict[str, Any]:
    """The budget of one stream as written into its record: ``horizon``, ``horizon_rule``, ``max_steps``,
    ``max_steps_rule``. ``kind``: ``replay`` (``demo_len`` = the episode's recorded length) or ``policy`` (``task`` = the
    variant)."""
    if horizon == "auto":
        if kind == "replay":
            h, rule = int(demo_len), "demo_length"
        else:
            h, rule = policy_horizon(task, train_root), "2x_longest_training_demo"
    else:
        h, rule = int(horizon), "fixed"
    if max_steps is None:
        max_steps = "auto" if horizon == "auto" else FIXED_MAX_STEPS
    if max_steps == "auto":
        ms, ms_rule = max_steps_for(h), "ceil(1.6 x horizon) + 100"
    else:
        ms, ms_rule = int(max_steps), "fixed"
    return {"horizon": h, "horizon_rule": rule, "max_steps": ms, "max_steps_rule": ms_rule}


__all__ = ["spec", "demo_length", "policy_horizon", "max_steps_for", "budget", "TRAIN_ROOT", "FIXED_MAX_STEPS"]
