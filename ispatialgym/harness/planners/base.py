"""Planner interface (RoboProbe's turn protocol): ``start(ctx)`` once per episode; ``decide(observation, last_outcome)``
with every new observation; then the motion the decision requested is either ``accepted(plan)`` (its tool result is
written before it plays) or ``rejected(reason, repairable)`` (nothing moved, no new observation; the planner answers
with another decision)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ispatialgym.harness.core.types import Decision, MotionOutcome, MotionPlan, Observation


@dataclass
class EpisodeContext:
    instruction: str
    goal_rgb: Optional[np.ndarray]
    alias: Optional[str] = None
    goal_source: Optional[str] = None
    mounting: Dict[str, Sequence[float]] = field(default_factory=dict)  # robot-only positions for the embodiment notes


class Planner:
    name = "base"

    def start(self, ctx: EpisodeContext) -> None:
        raise NotImplementedError

    def decide(self, obs: Observation, outcome: Optional[MotionOutcome]) -> Decision:
        raise NotImplementedError

    def accepted(self, plan: MotionPlan) -> None:
        raise NotImplementedError

    def rejected(self, reason: str, repairable: bool) -> Decision:
        raise NotImplementedError

    def summary(self) -> Dict[str, Any]:
        return {}
