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
from typing import Optional, Tuple

import torch as th

from ispatialgym.oracle import r1pro_model as M

_UPRIGHT = th.tensor(M.TORSO_UPRIGHT, dtype=th.float32)
_DOWNWARD = th.tensor(M.TORSO_DOWNWARD, dtype=th.float32)
_GROUND = th.tensor(M.TORSO_GROUND, dtype=th.float32)

#: JoyLo defaults (scripts/run_joylo.py): translate rate when the button is held, tilt rate at full stick.
JOYLO_TRANSLATE_RATE = 0.1  # [translate units / s]
JOYLO_TILT_RATE = 0.05  # [rad / s]


def translate_to_torso_q(translate: float) -> th.Tensor:
    """Torso joint positions (4,) on the keyframe curve for ``translate`` in [0, 2] (tilt = 0)."""
    lo, hi = M.TRUNK_TRANSLATE_RANGE
    t = min(max(float(translate), lo), hi)
    if t <= 1.0:
        return (1.0 - t) * _UPRIGHT + t * _DOWNWARD
    s = t - 1.0
    return (1.0 - s) * _DOWNWARD + s * _GROUND


def torso_q_to_translate_tilt(q: th.Tensor) -> Tuple[float, float]:
    """Inverse of the keyframe curve using ``torso_joint1`` (monotonic along the curve); the tilt is
    the residual of ``torso_joint3`` w.r.t. the curve value."""
    q = th.as_tensor(q, dtype=th.float32)
    j1 = float(q[0])
    if j1 > float(_DOWNWARD[0]):
        translate = 1.0 + (j1 - float(_DOWNWARD[0])) / (float(_GROUND[0]) - float(_DOWNWARD[0]))
    else:
        translate = (j1 - float(_UPRIGHT[0])) / (float(_DOWNWARD[0]) - float(_UPRIGHT[0]))
    translate = min(max(translate, M.TRUNK_TRANSLATE_RANGE[0]), M.TRUNK_TRANSLATE_RANGE[1])
    tilt = float(q[M.TRUNK_TILT_JOINT_INDEX] - translate_to_torso_q(translate)[M.TRUNK_TILT_JOINT_INDEX])
    return translate, tilt


def torso_q(translate: float, tilt: float, joint_limits: Optional[Tuple[th.Tensor, th.Tensor]] = None) -> th.Tensor:
    """Torso joint positions for (translate, tilt); ``tilt`` is clamped to the joint limits if given."""
    q = translate_to_torso_q(translate).clone()
    j = M.TRUNK_TILT_JOINT_INDEX
    val = float(q[j]) + float(tilt)
    if joint_limits is not None:
        val = min(max(val, float(joint_limits[0][j])), float(joint_limits[1][j]))
    q[j] = val
    return q


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
