#!/usr/bin/env python3
"""Pipeline validation: replay a test episode's recorded actions open-loop on its mapped instance with the Tier B evaluator
settings (BEHAVIOR sensor set, stream QA = timeout only). A correct instance mapping + executor should reproduce the demo's
success (up to simulator nondeterminism).

    python lab_tools/replay_check.py --episode 100 [--episode 0 ...] --out runlogs/tierB/replay_check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMNIGIBSON_TRACK_STATE_UPDATES", "0")
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=int, action="append", required=True)
    ap.add_argument("--dataset", default=os.path.join(WT, "datasets", "isg-init-eval"))
    ap.add_argument("--instances", default=os.path.join(WT, "lab", "instances.json"))
    ap.add_argument("--out", default=os.path.join(WT, "runlogs", "tierB", "replay_check"))
    args = ap.parse_args(argv)

    from ispatialgym.oracle import qa as QA
    from ispatialgym.oracle import run_isg as R

    class EvalLoader(R.InstanceLoader):
        @property
        def behavior_output(self):
            return True

        @behavior_output.setter
        def behavior_output(self, value):
            pass

    R.InstanceLoader = EvalLoader

    def check(self, step=None):
        self.episode_steps = self.episode_steps + 1 if step is None else int(step)
        self.action_steps += 1
        try:
            self._check_collision()
        except Exception:  # noqa: BLE001
            pass
        return self._check_timeout()

    QA.Monitors.check = check
    inst = json.load(open(args.instances))
    inst = inst.get("episodes", inst)
    results = []
    shared = {}
    last_template = None
    for e in args.episode:
        v = inst[str(e)]
        tro = v["tro"]
        tdir = os.path.dirname(os.path.dirname(os.path.abspath(tro)))
        if shared.get("env") is not None and last_template != v["task"]:
            print(f"REPLAY_CHECK episode {e}: another task needs its own world; run it in a separate process", flush=True)
            continue
        if shared.get("env") is not None:
            shared["template_dir"] = tdir
        out_dir = os.path.join(args.out, v["task"])
        argv_r = ["--instance", tro, "--mode", "replay", "--replay", args.dataset, "--episode", str(e), "--out", out_dir, "--record", "video", "--video",
                  "--video-size", "540", "--demo-gating", "none", "--no-goal-override", "--log-level", "WARNING"]
        rargs = R.parse_args(argv_r)
        t0 = time.time()
        rc, result = R.run_instance(rargs, tro, shared)
        last_template = v["task"]
        rec = {"episode_index": e, "task": v["task"], "instance": v["instance"], "success": result.get("success"), "reason": result.get("reason"),
               "n_steps": result.get("n_steps"), "n_action_steps": result.get("n_action_steps"), "wall_s": round(time.time() - t0, 1),
               "final_step_max_position_error": result.get("final_step_max_position_error"), "final_step_max_orientation_error": result.get("final_step_max_orientation_error"),
               "out": result.get("out")}
        results.append(rec)
        print("REPLAY_CHECK " + json.dumps(rec), flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"replay_check_{int(time.time())}.json"), "w") as f:
        json.dump(results, f, indent=1)
    try:
        import omnigibson as og

        og.shutdown()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
