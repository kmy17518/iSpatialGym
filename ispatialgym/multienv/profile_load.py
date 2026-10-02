"""``run_isg`` for one instance with cProfile around the world load and the planner build, stopping before the
first action (a measurement tool, not used by production).

    python -m ispatialgym.multienv.profile_load --prof-dir DIR [--top 40] <run_isg args for one TRO...>

Profiled phases, each written to ``DIR/<phase>.prof``: ``InstanceLoader.build_env`` (``og.Environment``: scene, objects,
robot, sensors, the trav-map patches, the initial state dump), ``InstanceLoader.restore`` (the instance's TRO state),
``R1ProCuRobo.__init__`` (the cuRobo models of the prebuilt embodiments) and the first ``R1ProCuRobo.update_world``
(the collision world). The run stops when ``Motions`` is built, right after the collision world, so nothing is
recorded (run_isg writes an error verdict for the instance). Prints per phase ``PROFILE_LOAD {json}``: wall seconds,
the top functions by cumulative and by own time, and the USD / Fabric synchronization entries (``editing_usd`` blocks,
``SynchronizeToFabric``, ``update_world_xforms``, ``batched_usd_edits``) with their call counts and times, and the
garbage collector's time per generation (``gc.callbacks``). Kit's launch (``og.launch``, inside ``build_env`` on the
first world) runs with the profiler paused and is reported as its own phase ``kit_launch``: Kit installs its own
profile hook while it starts, which would end cProfile's recording.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import os
import pstats
import sys
import time

from ispatialgym.oracle import run_isg  # first: run_isg's import order (curobo before omnigibson)
from ispatialgym.oracle import curobo_planner as CP
import omnigibson as _og

_ACTIVE = [None]
_GC = {"t0": None, "s": [0.0, 0.0, 0.0], "n": [0, 0, 0]}


def _gc_cb(phase, info):
    if phase == "start":
        _GC["t0"] = time.perf_counter()
    elif _GC["t0"] is not None:
        g = int(info.get("generation", 2))
        _GC["s"][g] += time.perf_counter() - _GC["t0"]
        _GC["n"][g] += 1
        _GC["t0"] = None


gc.callbacks.append(_gc_cb)

FOCUS = ("editing_usd", "SynchronizeToFabric", "update_world_xforms", "batched_usd_edits", "load_state", "_load",
         "import_obj", "add_reference", "AddReference", "CreatePrim", "app.update", "'update' of", "render", "play", "stop")


class _Stop(Exception):
    pass


def _label(func):
    path, line, name = func
    if path == "~":
        return name
    parts = path.split("/")
    return f"{'/'.join(parts[-2:])}:{line}({name})"


def _summary(prof: cProfile.Profile, top: int):
    st = pstats.Stats(prof)
    rows = [(_label(f), cc, nc, tt, ct) for f, (cc, nc, tt, ct, _) in st.stats.items()]
    by_cum = sorted(rows, key=lambda r: -r[4])[:top]
    by_tot = sorted(rows, key=lambda r: -r[3])[:top]
    focus = sorted((r for r in rows if any(k in r[0] for k in FOCUS)), key=lambda r: -r[4])[:top]
    fmt = lambda rs: [{"fn": r[0], "calls": r[2], "tottime": round(r[3], 3), "cumtime": round(r[4], 3)} for r in rs]
    return {"by_cumtime": fmt(by_cum), "by_tottime": fmt(by_tot), "usd_fabric": fmt(focus)}


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--prof-dir", required=True)
    ap.add_argument("--top", type=int, default=40)
    args, rest = ap.parse_known_args()
    os.makedirs(args.prof_dir, exist_ok=True)
    phases = {}

    def wrap(owner, name, phase, once=False):
        orig = getattr(owner, name)

        def wrapper(*a, **kw):
            if once and phase in phases:
                return orig(*a, **kw)
            prof = cProfile.Profile()
            gc_s0, gc_n0 = list(_GC["s"]), list(_GC["n"])
            launch0 = phases.get("kit_launch", 0.0)
            t0 = time.perf_counter()
            _ACTIVE[0] = prof
            prof.enable()
            try:
                return orig(*a, **kw)
            finally:
                prof.disable()
                _ACTIVE[0] = None
                wall = time.perf_counter() - t0
                prof.dump_stats(os.path.join(args.prof_dir, f"{phase}.prof"))
                phases[phase] = wall
                gc_s = [round(s - s0, 3) for s, s0 in zip(_GC["s"], gc_s0)]
                gc_n = [n - n0 for n, n0 in zip(_GC["n"], gc_n0)]
                print("PROFILE_LOAD " + json.dumps({"phase": phase, "wall_s": round(wall, 2),
                                                    "kit_launch_s": round(phases.get("kit_launch", 0.0) - launch0, 2),
                                                    "gc_s_by_generation": gc_s, "gc_n_by_generation": gc_n, **_summary(prof, args.top)}), flush=True)

        setattr(owner, name, wrapper)

    orig_launch = _og.launch

    def launch(*a, **kw):
        prof = _ACTIVE[0]
        if prof is not None:
            prof.disable()
        t0 = time.perf_counter()
        try:
            return orig_launch(*a, **kw)
        finally:
            phases["kit_launch"] = phases.get("kit_launch", 0.0) + time.perf_counter() - t0
            if prof is not None:
                prof.enable()

    _og.launch = launch

    wrap(run_isg.InstanceLoader, "build_env", "build_env")
    wrap(run_isg.InstanceLoader, "restore", "restore")
    wrap(CP.R1ProCuRobo, "__init__", "planner_init")
    wrap(CP.R1ProCuRobo, "update_world", "update_world", once=True)

    def stop(*a, **kw):
        print("PROFILE_LOAD " + json.dumps({"phase": "done", "phases_wall_s": {k: round(v, 2) for k, v in phases.items()}}), flush=True)
        raise _Stop("profile_load: stop after the planner build")

    run_isg.Motions = stop
    sys.argv = [sys.argv[0]] + rest
    return run_isg.main(rest)


if __name__ == "__main__":
    rc = main()
    try:
        import omnigibson as og

        og.shutdown()
    except Exception:  # noqa: BLE001
        pass
    sys.exit(0)
