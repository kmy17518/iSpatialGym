"""The lab evaluator's patches of ``run_isg`` for policy / replay streams (shared by ``eval_episodes.py`` and ``replay_episodes.py``).

Installed once per process by :func:`install` (before any world is built):

* ``InstanceLoader.behavior_output`` forced on: BEHAVIOR's sensor set (head 720 px / 40 mm, wrists 480 px, rgb + depth + proprio)
  without writing a LeRobot shard;
* no ``depth_linear`` annotators unless ``keep_depth`` (the policy consumes RGB + proprio; rendering depth costs time);
* stream QA: ``Monitors.check`` keeps the step counters and the contact statistics but returns only the timeout verdict
  (collision / pause / FOV verdicts off during policy streams - machine 1, 2026-10-04T18:33:48Z);
* epilogue: when the binary goal never held during the stream, one 10-step no-op window and no second window (the verdict
  is ``not_reached`` / ``tolerance`` either way); a goal that held at some step gets the executor's default windows
  (max(50, 20 % of the streamed steps), then a 2x second window when the robot moved or the goal was lost).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# OmniGibson's per-step tensorized-state change detection only feeds BaseObject.is_active, which only the HDF5 recorder's
# dump filter reads; the evaluator records video only, and the detection costs ~half of an env step (bc7180af).
os.environ.setdefault("OMNIGIBSON_TRACK_STATE_UPDATES", "0")

_INSTALLED = False
_NO_PLANNER = False
#: ``install(fixed_handle_ag=True)``: the oracle's assisted-grasp rule in streams (see :func:`apply_fixed_handle_ag`)
FIXED_HANDLE_AG = False


def apply_fixed_handle_ag(robot) -> None:
    """The demo collector's assisted-grasp semantics for a stream (item 1b, 2026-10-08): ``run_isg --mode oracle`` closes a
    handle grasp (a lid, a door leaf - a link behind a movable joint) with the assisted-grasp joint forced to a FixedJoint
    (``execute._patch_ag_joint_type``, ``handle_grasp_fixed_ag``; OmniGibson's stock rule picks a SphericalJoint there, under
    which the hand turns without turning the link). A replayed or policy stream has no action boundaries to install the patch
    around, so it holds for the whole episode: rigid bodies keep the stock FixedJoint, links behind movable joints get a
    FixedJoint instead of the spherical one. Without it the blender demos (grasp the lid, pull it open by driving the base)
    replay to the lid turning in the hand - the blender body dragged 11 mm, the lid stopping at 0.34 rad."""
    from ispatialgym.oracle.execute import _patch_ag_joint_type

    _patch_ag_joint_type(robot, "FixedJoint")


class NoPlanner:
    """Stand-in for ``R1ProCuRobo`` in policy / replay streams (evaluator fix 2.1, 2026-10-08): a stream never plans, but
    ``run_isg.run_instance`` builds the cuRobo planner and its collision world unconditionally (~27 s and ~5 GB of GPU memory per
    process). Only the attributes ``run_instance`` and the executor touch outside planning are provided; ``Motions`` reads
    ``control_dt`` at construction and nothing else before a motion is planned (streams plan none)."""

    control_dt = 1.0 / 30.0

    def __init__(self, robot, prebuild=(), **kwargs):
        self.robot = robot
        self.prebuild = tuple(prebuild)
        self._world_mesh_names = []
        self._attached = {}
        self.obstacle_filter = None

    def update_world(self, *args, **kwargs):
        return None

    def attach_object(self, arm, obj):
        self._attached[arm] = obj

    def detach_object(self, arm):
        self._attached.pop(arm, None)

    def __getattr__(self, name):  # a planning call is a programming error in stream mode (AttributeError keeps getattr(..., default) working)
        raise AttributeError(f"NoPlanner has no {name!r}: planner operations are not available in policy / replay streams")


def after_episode_cleanup() -> dict:
    """Between episodes of a long-lived evaluator process: drop Python garbage and return torch's cached CUDA blocks to the
    driver (the caching allocator otherwise keeps every peak); returns the allocator counters after the cleanup."""
    import gc

    gc.collect()
    out = {}
    try:
        import torch as th

        th.cuda.synchronize()
        th.cuda.empty_cache()
        out = {"torch_allocated_mib": int(th.cuda.memory_allocated() // (1 << 20)), "torch_reserved_mib": int(th.cuda.memory_reserved() // (1 << 20))}
    except Exception:  # noqa: BLE001
        pass
    return out


def install(keep_depth: bool = False, no_planner: bool = False, fixed_handle_ag: bool = False) -> None:
    global _INSTALLED, _NO_PLANNER, FIXED_HANDLE_AG
    if fixed_handle_ag:
        FIXED_HANDLE_AG = True
    if no_planner and not _NO_PLANNER:
        from ispatialgym.oracle import run_isg as R

        R.R1ProCuRobo = NoPlanner
        _NO_PLANNER = True
    if _INSTALLED:
        return
    _INSTALLED = True
    from ispatialgym.oracle import qa as QA
    from ispatialgym.oracle import run_isg as R
    from ispatialgym.oracle.executor import Executor

    class EvalLoader(R.InstanceLoader):
        """BEHAVIOR's sensor set (head 720 px, wrists 480 px, rgb + depth + proprio) without a LeRobot shard; the oracle's
        handle-grasp assisted-grasp rule on the robot when ``FIXED_HANDLE_AG`` is set."""

        @property
        def behavior_output(self):
            return True

        @behavior_output.setter
        def behavior_output(self, value):
            pass

        def build_env(self):
            env = super().build_env()
            if FIXED_HANDLE_AG:
                apply_fixed_handle_ag(self.robot)
                self.report["fixed_handle_ag"] = True
            return env

        def attach(self, shared):
            env = super().attach(shared)
            if FIXED_HANDLE_AG:
                apply_fixed_handle_ag(self.robot)
                self.report["fixed_handle_ag"] = True
            return env

    R.InstanceLoader = EvalLoader

    if not keep_depth:
        from ispatialgym.oracle import behavior_lerobot as BL

        _orig_overrides = BL.robot_overrides

        def robot_overrides():
            o = _orig_overrides()
            o["obs_modalities"] = [m for m in o["obs_modalities"] if m != "depth_linear"]
            return o

        BL.robot_overrides = robot_overrides

    def check(self, step=None):
        self.episode_steps = self.episode_steps + 1 if step is None else int(step)
        self.action_steps += 1
        try:
            self._check_collision()  # contact statistics only
        except Exception:  # noqa: BLE001
            pass
        return self._check_timeout()

    QA.Monitors.check = check

    _orig_epilogue = Executor.epilogue

    def epilogue(self, verdict, n_action_steps):
        """A goal that never held during the stream fails as not_reached / tolerance whatever the settle windows show: a
        10-step no-op window instead of max(50, 20 %) + 2x (the final errors are read where the policy left the robot).
        A goal that held at some step gets the executor's default windows."""
        if not self.isg.get("binary_success_ever", self.isg.get("binary_success", False)):
            saved = (self.cfg.settle_second_window_factor, self.cfg.settle_fraction, self.cfg.settle_min_steps)
            self.cfg.settle_second_window_factor, self.cfg.settle_fraction, self.cfg.settle_min_steps = 0.0, 0.0, 10
            try:
                return _orig_epilogue(self, verdict, n_action_steps)
            finally:
                self.cfg.settle_second_window_factor, self.cfg.settle_fraction, self.cfg.settle_min_steps = saved
        return _orig_epilogue(self, verdict, n_action_steps)

    Executor.epilogue = epilogue


__all__ = ["install"]
