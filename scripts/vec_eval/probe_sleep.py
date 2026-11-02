"""Sleep probe (campaign vec_next, 2026-10-30): build an N-clone world of an instance list's first instance, restore the TRO into every slot,
hold the robot for --steps control steps and print, every --every steps and per slot, whether the task's physical objects are asleep, their
linear / angular speed, their sleep and stabilization thresholds, plus the PhysicsScene's PhysX attributes once. Optional knobs:
--sleep-threshold / --stabilization (scene flag + per-body threshold) applied to every slot's physical objects before stepping.

    python scripts/vec_eval/probe_sleep.py --instances <alias.json> --num-envs 2 --scene-offsets "0,64.6,0" --steps 300 --every 30
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import omnigibson as og  # noqa: F401
import torch as th

from ispatialgym.oracle.vec_loader import VecInstanceLoader  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--instances", required=True)
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--scene-offsets", default=None)
    p.add_argument("--layout", default="grid-centered")
    p.add_argument("--scene-margin", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--every", type=int, default=30)
    p.add_argument("--sleep-threshold", type=float, default=None, help="physxRigidBody:sleepThreshold for the task objects (mass-normalised KE)")
    p.add_argument("--stabilization", type=float, default=None, help="enable physxScene:enableStabilization and set this stabilizationThreshold on the task objects")
    args = p.parse_args()
    if args.scene_offsets:
        os.environ["ISG_SCENE_OFFSETS"] = args.scene_offsets
    with open(args.instances) as f:
        spec = json.load(f)
    tro = spec["instances"][0]["tro"]
    pov = {}
    if args.sleep_threshold is not None:
        pov["rigid_sleep_threshold"] = args.sleep_threshold
    loader = VecInstanceLoader(num_envs=args.num_envs, behavior_output=True, tiled_rendering=False, drop_depth=True, external_camera_hw=None,
                               layout=args.layout, scene_margin=args.scene_margin, physics_overrides=pov or None)
    env = loader.build_env(tro)
    # physics scene attributes
    import omnigibson.lazy as lazy
    stage = og.sim.stage
    scene_prim = None
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsScene":
            scene_prim = prim
            break
    if scene_prim is not None:
        attrs = {a.GetName(): a.Get() for a in scene_prim.GetAttributes() if a.GetName().startswith("physxScene:") and a.HasAuthoredValue()}
        print(f"PROBE physics_scene {scene_prim.GetPath()} authored: {json.dumps(attrs, default=str)}", flush=True)
        for name in ("physxScene:enableStabilization", "physxScene:enableEnhancedDeterminism", "physxScene:solverType", "physxScene:timeStepsPerSecond",
                     "physxScene:bounceThreshold", "physxScene:frictionType", "physxScene:enableCCD", "physxScene:minPositionIterationCount", "physxScene:maxPositionIterationCount"):
            a = scene_prim.GetAttribute(name)
            print(f"PROBE scene attr {name} = {a.Get() if a and a.IsValid() else 'n/a'}", flush=True)
    obs, info = loader.restore_batch([tro] * args.num_envs)
    objs = {i: loader.physical_of(i) for i in range(args.num_envs)}
    for i in range(args.num_envs):
        for inst, obj in objs[i].items():
            try:
                inertia = None
                try:
                    inertia = [round(float(v), 5) for v in obj.root_link.get_attribute("physics:diagonalInertia")]
                except Exception:  # noqa: BLE001
                    pass
                print(f"PROBE slot {i} {inst}: mass={float(obj.mass):.4f} inertia={inertia} n_dof={getattr(obj, 'n_dof', None)} sleep_thr={obj.sleep_threshold} stab_thr={obj.stabilization_threshold} fixed_base={obj.fixed_base} kinematic_only={obj.kinematic_only} links={len(obj.links)}", flush=True)
            except Exception as ex:  # noqa: BLE001
                print(f"PROBE slot {i} {inst}: attrs unavailable ({ex!r})", flush=True)
    if args.stabilization is not None:
        og.sim.stop()
        if args.stabilization is not None and scene_prim is not None:
            with og.sim.editing_usd():
                a = scene_prim.GetAttribute("physxScene:enableStabilization") or scene_prim.CreateAttribute("physxScene:enableStabilization", lazy.pxr.Sdf.ValueTypeNames.Bool)
                a.Set(True)
        for i in range(args.num_envs):
            for inst, obj in objs[i].items():
                if obj.fixed_base or obj.kinematic_only:
                    continue
                if args.stabilization is not None:
                    obj.stabilization_threshold = args.stabilization
        og.sim.play()
        og.sim.update_handles()
        print(f"PROBE knobs applied: sleep_threshold={args.sleep_threshold} stabilization={args.stabilization}", flush=True)
    actions = th.stack([loader.hold_action(i) for i in range(args.num_envs)])
    t0 = time.time()
    for step in range(1, args.steps + 1):
        env.step(actions)
        if step % args.every == 0 or step == 1:
            for i in range(args.num_envs):
                parts = []
                for inst, obj in objs[i].items():
                    if obj.fixed_base or obj.kinematic_only:
                        continue
                    try:
                        v = float(th.linalg.norm(obj.get_linear_velocity())) * 1000
                        w = float(th.linalg.norm(obj.get_angular_velocity())) * 1000
                        asleep = bool(obj.is_asleep) if not callable(obj.is_asleep) else bool(obj.is_asleep())
                    except Exception as ex:  # noqa: BLE001
                        v, w, asleep = float("nan"), float("nan"), f"?{ex!r}"[:40]
                    parts.append(f"{inst.split('.')[0]}: asleep={asleep} v={v:.2f}mm/s w={w:.2f}mrad/s")
                print(f"PROBE step {step:4d} slot {i} off={[round(float(x), 1) for x in loader.report['scene_offsets'][i]]} | " + " | ".join(parts), flush=True)
    print(f"PROBE done {args.steps} steps in {time.time() - t0:.1f} s", flush=True)
    og.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
