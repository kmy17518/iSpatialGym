"""Holonomic base velocity tracker that behaves like a JoyLo joystick.

Given a reference trajectory ``(x, y, yaw)(t)`` in the world frame (typically the base part of a
cuRobo plan), :class:`BaseTrajectoryTracker` produces base-frame velocity commands
``(vx, vy, wz)`` every control step:

* feed-forward from the reference velocity plus proportional feedback on the pose error
  (pure-pursuit style progress along the path, so lag does not accumulate),
* clipped to the JoyLo velocity limits, acceleration limited, and exponentially smoothed
  (JoyLo filters stick readings with ``alpha = 0.2``),
* translation and rotation happen *simultaneously*: no translate-rotate-translate.

The same class also serves as a point-to-point controller when the reference is a single pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch as th

from ispatialgym.oracle.poses import wrap_angle


@dataclass
class BaseTrackerConfig:
    v_max: Tuple[float, float, float] = (0.5, 0.5, 0.8)  # m/s, m/s, rad/s (<= JoyLo limits 0.75/0.75/1.0)
    a_max: Tuple[float, float, float] = (0.6, 0.6, 1.5)  # m/s^2, rad/s^2
    kp_pos: float = 1.5
    kp_yaw: float = 2.0
    alpha: float = 0.35  # exponential smoothing of the command (1 = no smoothing)
    lookahead_steps: int = 2
    pos_tol: float = 0.02
    yaw_tol: float = 0.03
    settle_steps: int = 5  # consecutive in-tolerance steps required at the goal
    timeout_factor: float = 2.0  # max steps = timeout_factor * len(ref) + timeout_extra
    timeout_extra: int = 150


class BaseTrajectoryTracker:
    def __init__(self, ref: th.Tensor, dt: float, cfg: Optional[BaseTrackerConfig] = None):
        """
        Args:
            ref: ``(T, 3)`` reference ``(x, y, yaw)`` in the world frame, sampled every ``dt``.
            dt: control period (s).
        """
        ref = th.as_tensor(ref, dtype=th.float32).reshape(-1, 3)
        # unwrap yaw so that finite differences are meaningful
        yaw = ref[:, 2].clone()
        for i in range(1, len(yaw)):
            yaw[i] = yaw[i - 1] + wrap_angle(float(yaw[i] - yaw[i - 1]))
        self.ref = th.cat([ref[:, :2], yaw.unsqueeze(1)], dim=1)
        self.dt = float(dt)
        self.cfg = cfg or BaseTrackerConfig()
        self.k = 0
        self.steps = 0
        self.settled = 0
        self.last_cmd = th.zeros(3)
        self.max_steps = int(self.cfg.timeout_factor * len(self.ref) + self.cfg.timeout_extra)

    @property
    def goal(self) -> th.Tensor:
        return self.ref[-1]

    def error(self, cur: Tuple[float, float, float]) -> Tuple[float, float]:
        g = self.goal
        d = math.hypot(float(g[0]) - cur[0], float(g[1]) - cur[1])
        dyaw = abs(wrap_angle(float(g[2]) - cur[2]))
        return d, dyaw

    def _advance(self, cur: Tuple[float, float, float]):
        """Pure-pursuit progress: nearest reference index ahead of the current one (bounded window)."""
        T = len(self.ref)
        if self.k >= T - 1:
            return
        window = self.ref[self.k : min(T, self.k + 30), :2]
        d = th.linalg.norm(window - th.tensor(cur[:2]), dim=1)
        k_near = self.k + int(th.argmin(d))
        # never run ahead of time by more than one step per call, but allow catching up if behind
        self.k = min(max(k_near, self.k + 1), T - 1)

    def step(self, cur: Tuple[float, float, float]) -> Tuple[th.Tensor, bool]:
        """Compute the next base-frame velocity command.

        Args:
            cur: current base pose ``(x, y, yaw)`` in the world frame.

        Returns:
            (``(vx, vy, wz)`` tensor in the base frame, done flag).
        """
        self.steps += 1
        self._advance(cur)
        T = len(self.ref)
        k = self.k
        tgt = self.ref[min(k + self.cfg.lookahead_steps, T - 1)]

        # feed-forward velocity along the reference (world frame)
        if k < T - 1:
            v_ff_w = (self.ref[k + 1] - self.ref[k]) / self.dt
        else:
            v_ff_w = th.zeros(3)

        # feedback on the pose error (world frame)
        ex, ey = float(tgt[0]) - cur[0], float(tgt[1]) - cur[1]
        eyaw = wrap_angle(float(tgt[2]) - cur[2])
        v_w = th.tensor(
            [
                float(v_ff_w[0]) + self.cfg.kp_pos * ex,
                float(v_ff_w[1]) + self.cfg.kp_pos * ey,
                float(v_ff_w[2]) + self.cfg.kp_yaw * eyaw,
            ]
        )

        # world -> base frame
        c, s = math.cos(cur[2]), math.sin(cur[2])
        v_b = th.tensor([c * v_w[0] + s * v_w[1], -s * v_w[0] + c * v_w[1], v_w[2]])

        # limits
        v_max = th.tensor(self.cfg.v_max)
        v_b = th.clamp(v_b, -v_max, v_max)
        a_max = th.tensor(self.cfg.a_max) * self.dt
        v_b = self.last_cmd + th.clamp(v_b - self.last_cmd, -a_max, a_max)
        # exponential smoothing like the JoyLo stick filter
        v_b = self.cfg.alpha * v_b + (1.0 - self.cfg.alpha) * self.last_cmd

        # termination
        d_goal, dyaw_goal = self.error(cur)
        at_goal = k >= T - 1 and d_goal < self.cfg.pos_tol and dyaw_goal < self.cfg.yaw_tol
        self.settled = self.settled + 1 if at_goal else 0
        done = self.settled >= self.cfg.settle_steps
        if done:
            v_b = th.zeros(3)
        self.last_cmd = v_b
        return v_b, done

    @property
    def timed_out(self) -> bool:
        return self.steps >= self.max_steps
