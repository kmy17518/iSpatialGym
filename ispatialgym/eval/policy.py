"""Policies of ``run_isg --policy module:attribute``.

A policy is a callable ``policy(obs, info) -> action`` that returns R1Pro's 23-D action (:mod:`ispatialgym.robot.r1pro`)
every control step, or None to end the episode. ``obs`` is the environment's observation (``obs[robot_name][sensor]``
holds each camera's ``rgb``, ``obs[robot_name]["proprio"]`` the proprioception); ``info`` is the previous step's info.
The attribute is a class (instantiated with ``policy_kwargs``) or a callable. Optional hooks:

* ``bind(executor=, loader=, meta=, out_dir=, tro_path=)``, once before the first step: the
  :class:`~ispatialgym.eval.executor.Executor` (``env``, ``robot``, ``task``), the
  :class:`~ispatialgym.eval.loader.InstanceLoader` (``goal_image_path``), the run's ``meta.json`` dict and output folder;
* ``finalize(result)``, once after the episode, with the result dict before it is written; setting
  ``result["forced_failure"] = True`` scores the episode as a failure whatever the measurements (e.g. a give-up).

The policy's module is imported before the simulator starts, so it may import libraries that must load first.
"""

from __future__ import annotations

import importlib
from typing import Any


def import_policy(spec: str):
    """The attribute ``attr`` of ``module`` named by ``module:attr`` (default attribute: ``policy``)."""
    module, _, attr = spec.partition(":")
    if not module:
        raise ValueError(f"--policy expects module:attribute, got {spec!r}")
    return getattr(importlib.import_module(module), attr or "policy")


def make_policy(target, kwargs: dict[str, Any] | None = None):
    if isinstance(target, type):
        return target(**(kwargs or {}))
    if kwargs:
        raise ValueError("policy_kwargs need a policy class")
    return target


class HoldPolicy:
    """Holds the robot as the instance loaded it: arm and trunk targets, gripper commands, base still.

    Args:
        steps: actions to return before ending the episode (default: until the step budget ends it).
    """

    def __init__(self, steps: int | None = None):
        self.steps = steps
        self.action = None
        self.n = 0

    def bind(self, executor=None, loader=None, **_):
        self.action = loader.hold.action()

    def __call__(self, obs, info):
        if self.action is None or (self.steps is not None and self.n >= self.steps):
            return None
        self.n += 1
        return self.action.clone()
