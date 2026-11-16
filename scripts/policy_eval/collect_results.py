#!/usr/bin/env python3
"""Gather a Vast policy-eval run (its R2 ``done/`` records and ``out/`` unit tars, pulled by ``vast/collect.sh``) into a results dir:

    <results>/<ckpt>/ckpt.json                    the checkpoint's spec, the runs collected into it and the instances each expected
    <results>/<ckpt>/<instance id>/               every artifact of the instance's episode (run_isg --out): meta.json, result.json,
                                                  config.yaml, init_rgb.png, goal_rgb.png (the policy's goal image), final_rgb.png,
                                                  annotations.json, <view>_rgb.mp4 only if the run passed --record-view, ...
                                                  + worker.json (where it ran)
    <results>/<ckpt>/_unit_logs/<unit>.log        the run_isg log of the unit (one world: several instances)

    collect_results.py --run-dir $EV/runs/<run> [--dest <results>]     (default: spec.json results_dir, else <run-dir>/results)

Idempotent: a unit tar is extracted once (again, into cleared instance dirs, when its done record or tar changes: a redone unit). Several runs (other checkpoints, or a rerun of some
instances) can be collected into one results dir; ``summarize.py <results>`` then reports every checkpoint in it.
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dest", default=None)
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    spec = json.load(open(run_dir / "spec.json"))
    dest = Path(a.dest or spec.get("results_dir") or run_dir / "results")
    done = {}
    for p in sorted((run_dir / "done").glob("*.json")):
        d = json.load(open(p))
        done[d["unit"]] = d
    n_new = n_inst = 0
    for tag, ck in spec["ckpts"].items():
        cdir = dest / tag
        (cdir / "_unit_logs").mkdir(parents=True, exist_ok=True)
        expected = [{"index": x["index"], "instance_id": os.path.basename(x["tro"])[: -len("-tro_state.json")], "class_id": x["class_id"],
                     "task_family": x["task_family"], "subtask": x["subtask"], "unit": u["unit"]}
                    for u in spec["units"] if u["ckpt"] == tag for x in (u.get("instances") or [u])]
        cj = cdir / "ckpt.json"
        info = json.load(open(cj)) if cj.exists() else {"ckpt": tag, "runs": {}}
        info["spec"] = ck
        info["runs"][spec["run"]] = {"run_isg_args": spec["run_isg_args"] + list(ck.get("run_isg_args") or []), "code": spec["code"],
                                     "dataset": spec["dataset"], "results_dir": str(dest), "expected": expected}
        json.dump(info, open(cj, "w"), indent=1)
        for unit, d in sorted(done.items()):
            if d["ckpt"] != tag:
                continue
            tar = run_dir / "out" / tag / f"{unit}.tar.zst"
            mark = cdir / "_unit_logs" / f".{unit}.collected"
            if not tar.exists():
                print(f"no outputs for {unit} (done record only)")
                continue
            # a redone unit (full1 2026-11-16: 45 pi_goal_15k instances) overwrites its tar and done record: re-extract when either
            # changed, and clear its instance dirs first so no file of the earlier attempt stays
            stamp = f"{d.get('time')}:{d.get('instance')}:{tar.stat().st_size}"
            if not (mark.exists() and mark.read_text() == stamp):
                for x in d.get("instances") or []:
                    iid = (x.get("result") or {}).get("instance")
                    if iid and "/" not in iid and (cdir / iid).is_dir():
                        shutil.rmtree(cdir / iid)
                r = subprocess.run(f"zstd -dc '{tar}' | tar -x -C '{cdir}' --exclude=./run.log", shell=True, capture_output=True, text=True)
                if r.returncode:
                    print(f"extract {tar} failed: {r.stderr[-300:]}")
                    continue
                subprocess.run(f"zstd -dc '{tar}' | tar -xO ./run.log > '{cdir / '_unit_logs' / (unit + '.log')}'", shell=True,
                               capture_output=True)
                mark.write_text(stamp)
                n_new += 1
            for x in d.get("instances") or []:
                iid = (x.get("result") or {}).get("instance")
                if not iid or not (cdir / iid).is_dir():
                    continue
                n_inst += 1
                worker = {k: d.get(k) for k in ("unit", "ckpt", "instance", "tag", "gpu_name", "driver", "slot", "procs_per_gpu", "servers_per_gpu",
                                                "cpus", "seconds", "warm", "rc", "partial", "time")}
                worker.update({"run": spec["run"], "vast_instance": d.get("instance"), "index": x["index"], "reused_world": x.get("reused_world"),
                               "unit_seconds": d.get("seconds"), "unit_instances": len(d.get("instances") or [])})
                json.dump(worker, open(cdir / iid / "worker.json", "w"), indent=1)
                res = cdir / iid / "result.json"
                if not res.exists() and x.get("result"):  # runs before run_isg wrote result.json: the done record's copy
                    json.dump(x["result"], open(res, "w"), indent=1)
    print(f"collected {spec['run']} -> {dest}: {n_new} unit tars extracted now, {n_inst} instance dirs with worker records")


if __name__ == "__main__":
    main()
