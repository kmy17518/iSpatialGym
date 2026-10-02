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
import gc
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
_GC_T0 = [None]


def _gc_timer(phase, info):
    """``gc.callbacks``: the garbage collector's time per generation (timers ``gc.gen0`` .. ``gc.gen2``)."""
    if phase == "start":
        _GC_T0[0] = time.perf_counter()
    elif _GC_T0[0] is not None:
        tot = TOTALS[f"gc.gen{info.get('generation', 2)}"]
        tot[0] += time.perf_counter() - _GC_T0[0]
        tot[1] += 1
        _GC_T0[0] = None


gc.callbacks.append(_gc_timer)
#: timers whose calls are also logged one by one (name, start s since the first logged call, duration s), in order
EVENT_NAMES = set()
EVENTS = []
_T0 = [None]


def timed(name, fn):
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            dt = time.perf_counter() - t0
            rec = TOTALS[name]
            rec[0] += dt
            rec[1] += 1
            if name in EVENT_NAMES:
                if _T0[0] is None:
                    _T0[0] = t0
                EVENTS.append((name, round(t0 - _T0[0], 3), round(dt, 4)))

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
# inside cuRobo's planner: the trajectory optimizer, the PRM graph planner that seeds retries, and the IK solver
CP.MotionPlanner.plan_cspace = timed("MotionPlanner.plan_cspace", CP.MotionPlanner.plan_cspace)
CP.MotionPlanner.plan_pose = timed("MotionPlanner.plan_pose", CP.MotionPlanner.plan_pose)
for _cls, _ms in ((CP.TrajOptSolver, ("solve_cspace", "solve_pose")), (CP.PRMGraphPlanner, ("find_path",)), (CP.IKSolver, ("solve_pose",))):
    for _m in _ms:
        if hasattr(_cls, _m):
            patch(_cls, _m, f"{_cls.__name__}.{_m}")
            EVENT_NAMES.add(f"{_cls.__name__}.{_m}")
try:  # CUDA graph resets inside cuRobo's solvers (a reset means the next solve captures its graph again)
    from curobo._src.solver.solver_core import SolverCore as _SolverCore

    for _m in ("reset_shape", "reset_cuda_graph"):
        patch(_SolverCore, _m, f"SolverCore.{_m}")
        EVENT_NAMES.add(f"SolverCore.{_m}")
except Exception as _e:  # noqa: BLE001
    print("PROFILE_RUN solver_core timers unavailable:", repr(_e), flush=True)
for _m in ("plan_pose", "plan_joint", "attach_object", "detach_object", "update_world"):
    EVENT_NAMES.add(f"R1ProCuRobo.{_m}")
patch(CP.R1ProCuRobo, "detach_object")


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


def _patch_mesh_cache(value: str):
    """ISPATIALGYM_MESH_CACHE=auto|<n> (experiment): the cuRobo mesh store's capacity. cuRobo 0.8 launches its sphere-vs-mesh
    kernel over spheres x capacity threads, so the default 8192 is paid on every collision evaluation; ``auto`` sizes it to
    the scene's collision meshes plus headroom, as ``reach_bands --mesh-cache auto`` does (same algorithm; the order of the
    kernel's atomic float sums changes with the capacity, so results can differ marginally)."""
    import math

    _init = CP.R1ProCuRobo.__init__

    def __init__(self, robot, *a, **kw):
        n = None
        if value == "auto":
            n = sum(len(link.collision_meshes) for o in robot.scene.objects if o is not robot and not getattr(o, "visual_only", False)
                    for link in o.links.values())
            kw["mesh_cache_size"] = max(256, 64 * math.ceil((n + 256) / 64))
        else:
            kw["mesh_cache_size"] = int(value)
        print(f"PROFILE_RUN mesh cache {kw['mesh_cache_size']} for {n} scene collision meshes", flush=True)
        _init(self, robot, *a, **kw)

    CP.R1ProCuRobo.__init__ = __init__


_og = _patch_env_classes()
if "--behavior-shard" in sys.argv:
    _patch_behavior_shard()
if os.environ.get("ISPATIALGYM_MESH_CACHE"):
    _patch_mesh_cache(os.environ["ISPATIALGYM_MESH_CACHE"].lower())
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

#: the opt-ins' counters (check modes: calls / mismatches; ISPATIALGYM_SHARED_GOAL_EVAL also counts reuses), cumulative
#: over the process's instances
_CHECK_COUNTERS = (("fast_support", "ispatialgym.sampling.predicates_fast", "_COUNTS"),
                   ("fast_assign", "ispatialgym.eval.group_entity", "_ASSIGN_COUNTS"),
                   ("shared_goal", "ispatialgym.tasks.ispatialgym_task", "_SHARED_COUNTS"),
                   ("world_sync", "ispatialgym.oracle.curobo_planner", "_WS_COUNTS"))


def _check_counts():
    out = {}
    for key, mod, attr in _CHECK_COUNTERS:
        counts = getattr(sys.modules.get(mod), attr, None)
        if counts:
            out[key] = dict(counts)
    return out


def run_instance(args, tro_arg, shared=None):
    t0 = time.perf_counter()
    code, result = _run_instance(args, tro_arg, shared)
    out = {"wall_s": round(time.perf_counter() - t0, 2), "result": {k: result.get(k) for k in ("status", "n_steps", "wall_time_s", "out")},
           "track_state_updates": run_isg.provenance().get("track_state_updates"), "head_camera_only": os.environ.get("ISPATIALGYM_HEAD_CAMERA_ONLY", ""),
           "timers": {k: {"s": round(v[0], 3), "n": v[1], "ms_per_call": round(1000 * v[0] / max(1, v[1]), 2)} for k, v in sorted(TOTALS.items(), key=lambda kv: -kv[1][0])},
           "events": list(EVENTS), "check_counts": _check_counts()}
    print("PROFILE_RUN " + json.dumps(out), flush=True)
    if result.get("out"):
        try:
            with open(os.path.join(result["out"], "profile_run.json"), "w") as f:
                json.dump(out, f, indent=1)
        except OSError:
            pass
    TOTALS.clear()
    EVENTS.clear()
    _T0[0] = None
    return code, result


run_isg.run_instance = run_instance

if __name__ == "__main__":
    code = run_isg.main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(code)
