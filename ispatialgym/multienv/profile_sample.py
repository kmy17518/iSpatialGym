"""Timers inside the sampler's ``sample`` phase: ``sample_isg`` under this wrapper (timers only, nothing else changes).

    python -m ispatialgym.multienv.profile_sample <sample_isg arguments>

Per attempt (every ``SamplerV2.sample`` call, i.e. the ``sample`` phase of ``summary.phase_s``) the phase's wall time split
into exclusive buckets:
  - ``sim_step_render`` / ``sim_step_norender``: ``og.sim.step`` (settles, probes; physics and, with render-on-step, the
    render of that app update, which Kit does in one call), ``sim_step_physics``: ``og.sim.step_physics``
  - ``render``: explicit ``og.sim.render`` (segmentation / visibility renders), ``sensor_read``: ``VisionSensor.get_obs``
  - ``state``: simulator / scene state dumps and loads
  - ``curobo.<method>``: every public ``R1ProCuRobo`` method (IK, batched IK, collision / proximity, FK, world sync)
  - ``python``: the rest of the phase
A timed call nested in another is subtracted from its parent. ``regions``: inclusive time of the Phase-2 steps
(``prompt0``) for reference. Prints ``PROFILE_SAMPLE {json}`` per attempt and ``PROFILE_SAMPLE_TOTAL {json}`` at exit.
"""

from __future__ import annotations

import atexit
import functools
import inspect
import json
import os
import sys
import time
from collections import defaultdict

from ispatialgym.oracle import curobo_planner as CP  # noqa: E402  (curobo before omnigibson, as sample_isg)
from ispatialgym.sampling import sample_isg  # noqa: E402
from ispatialgym.sampling.v2 import core as V2  # noqa: E402
from ispatialgym.sampling.v2 import prompt0 as P0  # noqa: E402

STACK: list = []
ACC: dict = defaultdict(lambda: [0.0, 0])
REG: dict = defaultdict(lambda: [0.0, 0])
TOTAL: dict = defaultdict(lambda: [0.0, 0])
TOTAL_REG: dict = defaultdict(lambda: [0.0, 0])
IN = {"sample": False, "attempts": 0, "wall": 0.0, "python": 0.0}


def _timed_call(bucket, fn, *a, **kw):
    frame = [time.perf_counter(), 0.0]
    STACK.append(frame)
    try:
        return fn(*a, **kw)
    finally:
        STACK.pop()
        dt = time.perf_counter() - frame[0]
        ACC[bucket][0] += dt - frame[1]
        ACC[bucket][1] += 1
        if STACK:
            STACK[-1][1] += dt


def timed(bucket, fn):
    @functools.wraps(fn)
    def w(*a, **kw):
        if not IN["sample"]:
            return fn(*a, **kw)
        return _timed_call(bucket, fn, *a, **kw)

    return w


def region(name, fn):
    @functools.wraps(fn)
    def w(*a, **kw):
        if not IN["sample"]:
            return fn(*a, **kw)
        t0 = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            REG[name][0] += time.perf_counter() - t0
            REG[name][1] += 1

    return w


def _patch_sim():
    import omnigibson as og

    sim = og.sim
    if sim is None or getattr(sim, "_profile_sample_patched", False):
        return
    step = sim.step

    def sim_step(*a, **kw):
        if not IN["sample"]:
            return step(*a, **kw)
        return _timed_call("sim_step_render" if getattr(sim, "_render_on_step", True) else "sim_step_norender", step, *a, **kw)

    sim.step = sim_step
    sim.step_physics = timed("sim_step_physics", sim.step_physics)
    sim.render = timed("render", sim.render)
    sim.dump_state = timed("state", sim.dump_state)
    sim.load_state = timed("state", sim.load_state)
    sim._profile_sample_patched = True


def _patch_classes():
    from omnigibson.scenes.scene_base import Scene
    from omnigibson.sensors.vision_sensor import VisionSensor

    VisionSensor.get_obs = timed("sensor_read", VisionSensor.get_obs)
    Scene.dump_state = timed("state", Scene.dump_state)
    Scene.load_state = timed("state", Scene.load_state)
    for name, fn in list(vars(CP.R1ProCuRobo).items()):
        if not name.startswith("_") and inspect.isfunction(fn):
            setattr(CP.R1ProCuRobo, name, timed(f"curobo.{name}", fn))
    for name in ("phase2_step1", "phase2_screen", "phase2_manipulation", "body_free", "draw_manipulation"):
        if hasattr(P0, name):
            setattr(P0, name, region(f"prompt0.{name}", getattr(P0, name)))


def _round(d):
    return {k: {"s": round(v[0], 3), "n": v[1]} for k, v in sorted(d.items(), key=lambda kv: -kv[1][0])}


_sample = V2.SamplerV2.sample


def sample(self, rng):
    if IN["sample"]:
        return _sample(self, rng)
    _patch_sim()
    ACC.clear()
    REG.clear()
    k0 = self.kit.stats_snapshot() if getattr(self, "kit", None) is not None else {}
    IN["sample"] = True
    t0 = time.perf_counter()
    ok = False
    try:
        out = _sample(self, rng)
        ok = True
        return out
    finally:
        wall = time.perf_counter() - t0
        IN["sample"] = False
        py = wall - sum(v[0] for v in ACC.values())
        kit = self.kit.stats_delta(k0, self.kit.stats_snapshot()) if k0 else {}
        rec = {"alias": getattr(getattr(self, "gp", None), "alias", None), "class": type(self).__name__, "ok": ok, "wall_s": round(wall, 3),
               "python_s": round(py, 3), "buckets": _round(ACC), "regions": _round(REG), "kit": kit}
        print("PROFILE_SAMPLE " + json.dumps(rec), flush=True)
        IN["attempts"] += 1
        IN["wall"] += wall
        IN["python"] += py
        for k, v in ACC.items():
            TOTAL[k][0] += v[0]
            TOTAL[k][1] += v[1]
        for k, v in REG.items():
            TOTAL_REG[k][0] += v[0]
            TOTAL_REG[k][1] += v[1]


V2.SamplerV2.sample = sample
_patch_classes()


@atexit.register
def _total():
    if IN["attempts"]:
        print("PROFILE_SAMPLE_TOTAL " + json.dumps({"attempts": IN["attempts"], "wall_s": round(IN["wall"], 2), "python_s": round(IN["python"], 2),
                                                     "buckets": _round(TOTAL), "regions": _round(TOTAL_REG)}), flush=True)


if __name__ == "__main__":
    sample_isg.main(sys.argv[1:])
