"""``run_isg`` with wall-clock timers around the pieces of a demo step (a measurement tool, not used by production).

    python -m ispatialgym.multienv.profile_run <run_isg args...>

Wraps (cumulative seconds, calls; nested timers overlap): ``Executor.step`` and inside it ``env.step`` (split into
``og.sim.step`` / ``Environment.get_obs`` / ``ISpatialGymTask.step``), the recorded state dump, ``Recorder.record``,
``Monitors.check``, the support / root watches; checkpoints, restores and their render loops; the cuRobo planner's
``plan_* / solve_ik* / update_world``; ``Recorder.commit`` / ``close``; with ``--behavior-shard``, the BEHAVIOR LeRobot
writer (``ShardWriter`` and the ``LeRobotDataWrapper`` pieces it calls). Prints ``PROFILE_RUN {json}`` at exit and
writes it to ``<run out dir>/profile_run.json``. Behaviour is unchanged: the wrappers only time the calls.
"""

from __future__ import annotations

import functools
import json
import os
import sys
import time
from collections import defaultdict

from ispatialgym.oracle import run_isg  # first: run_isg's import order (curobo before omnigibson)
from ispatialgym import single_env as SE
from ispatialgym.oracle import curobo_planner as CP
from ispatialgym.oracle import executor as EX
from ispatialgym.oracle import qa as QA
from ispatialgym.oracle import recorder as REC

TOTALS = defaultdict(lambda: [0.0, 0])
_OUT = {"dir": None}


def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            rec = TOTALS[name]
            rec[0] += time.perf_counter() - t0
            rec[1] += 1

    return wrapper


def patch(owner, attr, name=None):
    setattr(owner, attr, timed(name or f"{getattr(owner, '__name__', owner)}.{attr}", getattr(owner, attr)))


patch(SE, "step", "env.step")
patch(EX.Executor, "step", "Executor.step")
patch(EX.Executor, "_dump_state")
patch(EX.Executor, "checkpoint")
patch(EX.Executor, "restore")
patch(EX.Executor, "render_restored")
patch(EX.Executor, "epilogue")
patch(EX.SupportLoad, "observe")
patch(EX.RootWatch, "observe")
patch(REC.Recorder, "record")
patch(REC.Recorder, "commit")
patch(REC.Recorder, "close")
patch(QA.Monitors, "check")
for _m in ("plan_pose", "plan_joint", "solve_ik", "solve_ik_batch", "solve_ik_path", "update_world", "check_collision", "attach_object"):
    patch(CP.R1ProCuRobo, _m)


def _patch_env_classes():
    import omnigibson as og
    from omnigibson.envs.env_base import Environment

    patch(Environment, "get_obs")
    try:
        from ispatialgym.tasks.ispatialgym_task import ISpatialGymTask

        patch(ISpatialGymTask, "step")
    except Exception as e:  # noqa: BLE001
        print("PROFILE_RUN task timer unavailable:", repr(e), flush=True)
    return og


def _patch_behavior_shard():
    """``--behavior-shard``: BEHAVIOR's LeRobot writer as the recorder drives it (imported only then: importing it sets up
    lerobot, which a run without a shard never does)."""
    from omnigibson.envs.lerobot_data_wrapper import LeRobotDataWrapper, LeRobotPlaybackWrapper

    from ispatialgym.oracle import behavior_lerobot as BL

    for m in ("__init__", "begin", "step", "end", "close", "abort"):
        patch(BL.ShardWriter, m, f"ShardWriter.{m}")
    patch(BL.FlatObsView, "flat", "FlatObsView.flat")
    for m in ("_process_obs", "_record_step_trajectory", "flush_current_traj"):
        if hasattr(LeRobotDataWrapper, m):
            patch(LeRobotDataWrapper, m, f"LeRobotDataWrapper.{m}")
    patch(LeRobotPlaybackWrapper, "flush_partial_traj", "LeRobotPlaybackWrapper.flush_partial_traj")


_og = _patch_env_classes()
if "--behavior-shard" in sys.argv:
    _patch_behavior_shard()
_build_env = run_isg.InstanceLoader.build_env


def build_env(self, *a, **kw):
    env = _build_env(self, *a, **kw)
    sim = _og.sim
    if not getattr(sim, "_profile_run_patched", False):
        sim.step = timed("og.sim.step", sim.step)
        sim.render = timed("og.sim.render", sim.render)
        sim._profile_run_patched = True
    return env


run_isg.InstanceLoader.build_env = build_env
_run_instance = run_isg.run_instance


def run_instance(args, tro_arg, shared=None):
    t0 = time.perf_counter()
    code, result = _run_instance(args, tro_arg, shared)
    out = {"wall_s": round(time.perf_counter() - t0, 2), "result": {k: result.get(k) for k in ("status", "n_steps", "wall_time_s", "out")},
           "track_state_updates": run_isg.provenance().get("track_state_updates"), "head_camera_only": os.environ.get("ISPATIALGYM_HEAD_CAMERA_ONLY", ""),
           "timers": {k: {"s": round(v[0], 3), "n": v[1], "ms_per_call": round(1000 * v[0] / max(1, v[1]), 2)} for k, v in sorted(TOTALS.items(), key=lambda kv: -kv[1][0])}}
    print("PROFILE_RUN " + json.dumps(out), flush=True)
    if result.get("out"):
        try:
            with open(os.path.join(result["out"], "profile_run.json"), "w") as f:
                json.dump(out, f, indent=1)
        except OSError:
            pass
    TOTALS.clear()
    return code, result


run_isg.run_instance = run_instance

if __name__ == "__main__":
    code = run_isg.main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(code)
