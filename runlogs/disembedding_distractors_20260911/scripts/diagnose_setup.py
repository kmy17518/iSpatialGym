#!/usr/bin/env python3
"""Why does the disembedding template fail to set up for a target category on the test table?

Runs ``DisembeddingSampler.setup()`` for one target (the category's test pick) with the reach-table internals
instrumented, and prints where the candidates die: reachable cells of the generic top-down reach grid, library grasp
combinations that pass the screening IK on the source tray (per translate), placement feasibility at the destination
centre, and the geometry of the library grasps themselves (eef height above the object base, approach angle from
vertical). Usage::

    OMNIGIBSON_HEADLESS=1 python diagnose_setup.py --categories teaspoon,box_of_cereal
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import validate_distractors as VD  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", required=True)
    ap.add_argument("--work", default="/tmp/isg_dis_distractors/diag")
    args = ap.parse_args()
    from ispatialgym.sampling.spec import load_task_spec

    spec_file = load_task_spec(VD.DEFAULT_SPEC)
    cands = json.load(open(os.path.join(ROOT, "candidates.json")))
    rooms = VD.rooms_of(spec_file)
    variant = VD.scene_variant_of(spec_file, rooms, args.work)
    from ispatialgym.sampling import geometry as G
    from ispatialgym.sampling.robot_utils import RobotKit
    from ispatialgym.sampling.sim_utils import Reject, SimWorld
    from ispatialgym.sampling.tasks import get_sampler_class
    from ispatialgym.sampling.tasks.base import SamplerContext, TaskSampler

    import omnigibson as og
    import torch as th

    world = SimWorld(VD.SCENE, rooms, scene_variant=variant) if variant else SimWorld(VD.SCENE, rooms)
    kit = RobotKit(world, prebuild=("right_arm",))
    base = spec_file[VD.PROBLEM]
    try:
        for cat in args.categories.split(","):
            t = cands["targets"][cat]
            same = [e for e in cands["same"][cat] if e["size_ok"] and e["ready"] and e["model"] != t["pick1"]]
            d = same[0] if same else [e for e in cands["diff"][cat] if e["size_ok"] and e["ready"]][0]
            pspec = VD.problem_for(base, cat, t["pick1"], t["pick1_key"], d["category"], d["model"])
            choices = pspec.template_choices(np.random.default_rng(0))[0]
            ctx = SamplerContext(pspec, pspec.scenes[VD.SCENE], choices, world, kit, os.path.join(args.work, cat), f"diag-{cat}", spec_file.sha256, base_seed=0)
            s = get_sampler_class(VD.PROBLEM)(ctx)
            s.variant_overrides = ["same_category" if d["category"] == cat else "other_category"]
            stats = {"reach_grid": [], "combos": [], "place": []}

            orig_reach = TaskSampler.reachable_cells

            def reach(self, cells, stance, z, *a, **k):
                mask, tr = orig_reach(self, cells, stance, z, *a, **k)
                stats["reach_grid"].append({"cells": len(cells), "reachable": int(mask.sum()), "z": round(float(z), 3), "edge": stance.edge_name})
                return mask, tr

            orig_combo = TaskSampler.grasp_combo_table

            def combo(self, stance, cells, pose_at, grasps, translates, **k):
                table = orig_combo(self, stance, cells, pose_at, grasps, translates, **k)
                n = sum(len(v) for v in table.values())
                stats["combos"].append({"cells": len(cells), "cells_with_combos": len(table), "combos": n, "grasps": len(grasps), "translates": list(translates), "edge": stance.edge_name})
                return table

            orig_pf = kit.pairs_feasible

            def pf(pairs, q_seed, ignore_for_grasp=None, **k):
                feas = orig_pf(pairs, q_seed, ignore_for_grasp=ignore_for_grasp, **k)
                stats["place"].append({"pairs": len(pairs), "feasible": int(feas.sum())})
                return feas

            TaskSampler.reachable_cells = reach
            TaskSampler.grasp_combo_table = combo
            kit.pairs_feasible = pf
            t0 = time.time()
            try:
                s.setup()
                outcome = f"OK ({len(s.cells)} grasp cells, place_ok {len(s.place_ok)})"
            except Reject as e:
                outcome = f"REJECT [{e.reason}] {str(e)[:300]}"
            except Exception as e:
                traceback.print_exc()
                outcome = f"ERROR {e!r}"
            finally:
                TaskSampler.reachable_cells = orig_reach
                TaskSampler.grasp_combo_table = orig_combo
                kit.pairs_feasible = orig_pf
            # geometry of the library grasps (eef in the object base frame)
            grasps = []
            for g in getattr(s, "grasps", []) or []:
                approach = G.rotate(G.t(g.quat), th.tensor([0.0, 0.0, 1.0]))  # eef +z = approach axis
                ang = math.degrees(math.acos(max(-1.0, min(1.0, -float(approach[2])))))  # 0 = straight down (object canonical up)
                grasps.append({"name": g.name, "eef_z_above_base_cm": round(float(G.t(g.position)[2]) * 100, 1), "approach_from_vertical_deg": round(ang, 1)})
            bb = t["pick1_bbox"]
            print(f"\n=== {cat}/{t['pick1']}: bbox {[round(v * 100, 1) for v in bb]} cm; {outcome} in {time.time() - t0:.0f}s")
            print("   library grasps:", grasps)
            print("   reach grid (generic top-down / 30 deg poses):", stats["reach_grid"])
            print("   grasp combination table (library grasps, source tray):", stats["combos"])
            print("   pairs_feasible calls (placement at the destination centre are the last ones):", stats["place"][-6:])
            for o in list(world.spawned.values()):
                world.remove(o)
            kit.sync_world()
    finally:
        og.shutdown()


if __name__ == "__main__":
    main()
