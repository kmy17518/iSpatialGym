"""The episode loop of policy mode: step the environment with the policy's actions until the run's geometric success,
the step budget, the wall clock or the policy stops, then judge the episode with the task's metrics."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch as th

from ispatialgym import single_env as SE

log = logging.getLogger(__name__)

N_RENDER_ITERATIONS = 1  # BEHAVIOR's convention: one render per control step


def isg_info(info: dict[str, Any] | None) -> dict[str, Any]:
    """The task's ``ispatialgym`` block of a step info (OmniGibson nests it under ``info["done"]``)."""
    info = info or {}
    done = info.get("done") if isinstance(info.get("done"), dict) else {}
    return done.get("ispatialgym") or info.get("ispatialgym") or {}


def verdict(isg: dict[str, Any], success_metric: str = "geometric") -> dict[str, str] | None:
    """None when the run reached its ``success_metric`` success, else ``{"reason", "message"}``: ``symbolic_only`` (the
    goal held but the measured targets were never within tolerance at the same step), ``not_settled`` (settle eval on:
    the goal held but never through a settled window) or ``not_reached``."""
    geometric, symbolic = bool(isg.get("geometric_success")), bool(isg.get("symbolic_success"))
    if geometric if success_metric == "geometric" else symbolic:
        return None
    if symbolic:
        outside = [
            m["bddl"] for m in (isg.get("evaluation") or {}).get("measured") or [] if not m.get("within_tolerance")
        ]
        return {
            "reason": "symbolic_only",
            "message": "measured targets outside tolerance: " + (", ".join(outside) or "none"),
        }
    if isg.get("settle_eval") and isg.get("goal_first_step") is not None:
        return {
            "reason": "not_settled",
            "message": f"the goal held from step {isg['goal_first_step']} but never through a settled window",
        }
    unsatisfied = [c["bddl"] for c in isg.get("conditions") or [] if not c.get("satisfied")]
    return {"reason": "not_reached", "message": "unsatisfied at the last step: " + (", ".join(unsatisfied) or "none")}


class Executor:
    """Steps one environment and keeps its latest observation (``bind``-able by policies as ``executor``)."""

    def __init__(
        self,
        env,
        task,
        robot,
        cam_key: str,
        max_steps: int,
        wallclock_s: float,
        success_metric: str = "geometric",
        video=None,
    ):
        self.env, self.task, self.robot, self.cam_key = env, task, robot, cam_key
        self.max_steps = int(max_steps)
        self.wallclock_s = float(wallclock_s)
        self.success_metric = success_metric
        self.video = video
        self.n = 0
        self.last_obs = None
        self.last_info: dict[str, Any] = {}
        self.last_rgb = None
        self.wall_clock_exceeded = False

    @property
    def isg(self) -> dict[str, Any]:
        return isg_info(self.last_info)

    def step(self, action) -> None:
        obs, _, _, _, info = SE.step(
            self.env, th.as_tensor(action, dtype=th.float32).reshape(-1), n_render_iterations=N_RENDER_ITERATIONS
        )
        self.n += 1
        self.last_obs, self.last_info = obs, info
        if self.video is not None:
            self.video.record(obs)
        rgb = (obs.get(self.robot.name) or {}).get(self.cam_key, {}).get("rgb") if isinstance(obs, dict) else None
        if rgb is not None:
            self.last_rgb = rgb

    def run(self, policy) -> dict[str, Any]:
        """Ask ``policy(obs, info)`` for one action per step; returns the episode result."""
        t0 = time.time()
        while self.n < self.max_steps:
            action = policy(self.last_obs, self.last_info)
            if action is None:
                log.info("the policy stopped after %d steps", self.n)
                break
            if time.time() - t0 > self.wallclock_s:
                self.wall_clock_exceeded = True
                log.info("wall clock of %.0f s reached after %d steps", self.wallclock_s, self.n)
                break
            self.step(action)
            if self.isg.get("official_success"):
                log.info("geometric success at step %s", self.isg.get("geometric_success_step"))
                break
        return self.result(time.time() - t0)

    def result(self, episode_s: float = 0.0) -> dict[str, Any]:
        isg = self.isg
        evaluation = isg.get("evaluation") or {}
        v = verdict(isg, self.success_metric)
        return {
            "success": v is None,
            "reason": None if v is None else v["reason"],
            "message": None if v is None else v["message"],
            "success_metric": self.success_metric,
            "settle_eval": isg.get("settle_eval"),
            "settle_steps": isg.get("settle_steps"),
            "symbolic_success": bool(isg.get("symbolic_success")),
            "symbolic_success_step": isg.get("symbolic_success_step"),
            "geometric_success": bool(isg.get("geometric_success")),
            "geometric_success_step": isg.get("geometric_success_step"),
            "errors": evaluation.get("errors") or isg.get("errors") or {},
            "errors_final_step": isg.get("errors") or {},
            "error_metric": evaluation.get("error_metric"),
            "measured": evaluation.get("measured"),
            "other_mode": isg.get("other_mode"),
            "goal_first_step": isg.get("goal_first_step"),
            "conditions": isg.get("conditions"),
            "group_assignments": isg.get("group_assignments"),
            "pose_settled_config": isg.get("pose_settled_config"),
            "n_steps": self.n,
            "wall_clock_exceeded": self.wall_clock_exceeded,
            "episode_s": round(episode_s, 2),
        }
