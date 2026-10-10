"""Open-loop replay of a recorded action sequence as a ``run_isg --mode policy`` callable (``--policy lab_tools.replay_policy:ReplayPolicy``).

The positive control of the relocalization phase: the demo's recorded 23-D actions (the LeRobot ``action`` column) are
fed to the same executor path, at the same 30 Hz control rate and with the same stop-on-success rule and epilogue as a
policy stream (:mod:`lab_tools.act_policy`), so a successful replay validates the whole action path (base velocity
semantics, controller configuration, control rate, settle check) independently of any policy.

The driver sets :data:`CONTEXT` before each episode: ``actions`` (T x 23 float32), ``horizon`` (step budget),
``stop_on_success``. While streaming, the callable records the robot's base pose (x, y, yaw) and the head camera's world
pose at every step in ``CONTEXT["base_xyyaw"]`` / ``CONTEXT["camera_pose"]`` (read from the simulator, *before* the step's
action is applied; index t = the pose the demo's ``observation.state[t]`` was recorded at), so the driver can compare the
replayed base trajectory with the demo's own states.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from lab_tools.act_policy import record_step_error, task_info

#: set by the driver before every episode
CONTEXT: Dict[str, Any] = {}


def _robot():
    import omnigibson as og

    return og.sim.scenes[0].robots[0]


def _yaw(quat_xyzw) -> float:
    x, y, z, w = [float(v) for v in quat_xyzw]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class ReplayPolicy:
    """``policy(obs, info) -> action | None``: the recorded action of the current step, None at the end."""

    def __init__(self):
        self.episode_ctx: Optional[dict] = None
        self.steps = 0
        self.robot = None
        self.cam = None

    def begin_episode(self, ctx: dict):
        self.episode_ctx = ctx
        self.steps = 0
        self.robot = _robot()
        self.cam = None
        for k, s in self.robot.sensors.items():
            if k.endswith("zed_link:Camera:0"):
                self.cam = s
        ctx["base_xyyaw"] = []
        ctx["base_qvel"] = []
        ctx["camera_pose"] = []
        ctx["joint_q"] = []  # the robot's 28 joint positions per step (manipulation tasks: the arm trajectory)
        ctx["object_poses"] = []  # per step: the task's tracked objects' (pos, quat), in ctx["object_names"] order
        ctx["stop_reason"] = None
        ctx["actions_streamed"] = 0
        task = (ctx.get("shared") or {}).get("task")  # the driver's shared world (filled by run_isg before the first policy call)
        scope = getattr(task, "object_scope", None) or {}
        self.objects = [(k, v) for k, v in scope.items() if v is not None and hasattr(v, "get_position_orientation") and k != "agent.n.01_1"
                        and getattr(v, "category", None) != "floors" and not type(v).__name__.startswith("Group")]
        ctx["object_names"] = [k for k, _ in self.objects]

    def record_pose(self, ctx: dict, obs) -> None:
        pos, quat = self.robot.get_position_orientation()
        pos, quat = pos.detach().cpu().numpy(), quat.detach().cpu().numpy()
        ctx["base_xyyaw"].append([float(pos[0]), float(pos[1]), _yaw(quat)])
        try:  # the robot-frame base velocity the demo's observation.state[0:3] holds (for the dead-reckoning self-check)
            pr = obs[self.robot.name]["proprio"]
            pr = pr.detach().cpu().numpy() if hasattr(pr, "detach") else np.asarray(pr)
            ctx["base_qvel"].append([float(v) for v in np.asarray(pr).reshape(-1)[0:3]])
        except Exception:  # noqa: BLE001
            pass
        if self.cam is not None:
            cp, cq = self.cam.get_position_orientation()
            ctx["camera_pose"].append([*[float(v) for v in cp.detach().cpu().numpy()], *[float(v) for v in cq.detach().cpu().numpy()]])
        try:
            ctx["joint_q"].append([float(v) for v in self.robot.get_joint_positions().detach().cpu().numpy()])
            row = []
            for _, o in self.objects:
                op, oq = o.get_position_orientation()
                row.append([*[float(v) for v in op.detach().cpu().numpy()], *[float(v) for v in oq.detach().cpu().numpy()]])
            ctx["object_poses"].append(row)
        except Exception:  # noqa: BLE001
            pass

    def end_stream(self, reason: str):
        ctx = self.episode_ctx or {}
        ctx["stop_reason"] = reason
        ctx["steps"] = self.steps
        return None

    def __call__(self, obs, info):
        ctx = CONTEXT
        if not ctx:
            raise RuntimeError("replay_policy.CONTEXT is empty: the driver must set it before the episode")
        if ctx is not self.episode_ctx:
            self.begin_episode(ctx)
        self.record_pose(ctx, obs)
        actions = ctx["actions"]
        if self.steps >= int(ctx["horizon"]):
            return self.end_stream("horizon")
        if self.steps >= len(actions):
            return self.end_stream("demo_end")
        isg = task_info(info)
        record_step_error(ctx, isg)
        if ctx.get("stop_on_success", True) and isg.get("official_success"):
            return self.end_stream("success")
        a = np.asarray(actions[self.steps], dtype=np.float32).reshape(-1)
        if a.shape != (23,) or not np.isfinite(a).all():
            raise RuntimeError(f"bad recorded action at step {self.steps}: shape {a.shape}")
        self.steps += 1
        ctx["steps"] = self.steps
        ctx["actions_streamed"] = self.steps
        if self.steps >= int(ctx["horizon"]) or self.steps >= len(actions):
            ctx["stop_reason"] = "horizon" if self.steps >= int(ctx["horizon"]) else "demo_end"  # no further call comes when the stream's slots are used up
        return a
