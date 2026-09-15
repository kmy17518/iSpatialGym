"""JoyLo-style trunk (torso) parameterisation for R1Pro.

JoyLo drives the four torso joints through two scalars (``joylo/gello/utils/og_teleop_utils.py``):

* ``translate`` in [0, 2]: piecewise-linear interpolation between three keyframes
  ``upright (0) -> downward (1) -> ground (2)``. Increasing it lowers the torso by bending
  ``torso_joint1..3`` in a coordinated, natural way ("raise / lower torso").
* ``tilt``: an additive offset on ``torso_joint3`` (upper-body pitch), clamped to joint limits.

Both scalars are integrated at a bounded rate (button / stick held), which is what the oracle
reproduces in :func:`TrunkState.step_towards`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.robot.conventions import torso_q, torso_q_to_translate_tilt, translate_to_torso_q  # noqa: F401  (moved to robot/conventions, F5)

__all__ = ["TrunkState", "JOYLO_TRANSLATE_RATE", "JOYLO_TILT_RATE", "torso_q", "torso_q_to_translate_tilt", "translate_to_torso_q"]

#: JoyLo defaults (scripts/run_joylo.py): translate rate when the button is held, tilt rate at full stick.
JOYLO_TRANSLATE_RATE = 0.1  # [translate units / s]
JOYLO_TILT_RATE = 0.05  # [rad / s]


@dataclass
class TrunkState:
    """Integrates the JoyLo trunk scalars at bounded rates."""

    translate: float
    tilt: float
    translate_rate: float = JOYLO_TRANSLATE_RATE
    tilt_rate: float = JOYLO_TILT_RATE

    @staticmethod
    def from_q(q: th.Tensor, **rates) -> "TrunkState":
        t, tilt = torso_q_to_translate_tilt(q)
        return TrunkState(translate=t, tilt=tilt, **rates)

    def step_towards(self, translate: Optional[float], tilt: Optional[float], dt: float) -> bool:
        """Move toward the targets by at most ``rate * dt``. Returns True when both are reached."""
        done = True
        if translate is not None:
            lo, hi = M.TRUNK_TRANSLATE_RANGE
            translate = min(max(float(translate), lo), hi)
            d = translate - self.translate
            step = self.translate_rate * dt
            if abs(d) > step:
                self.translate += step if d > 0 else -step
                done = False
            else:
                self.translate = translate
        if tilt is not None:
            d = float(tilt) - self.tilt
            step = self.tilt_rate * dt
            if abs(d) > step:
                self.tilt += step if d > 0 else -step
                done = False
            else:
                self.tilt = float(tilt)
        return done

    def q(self, joint_limits=None) -> th.Tensor:
        return torso_q(self.translate, self.tilt, joint_limits)
