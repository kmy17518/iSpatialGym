#!/usr/bin/env python3
"""Batched replay: R2 -> cluster for one run (read-only on R2). Status from R2 only (heartbeats, done, fail; no Vast API call).

    sync.py --run <name> status            units done / claimed / failed, live workers (heartbeat age, phase, running units)
    sync.py --run <name> pull [--units a,b] download every finished unit's outputs to $RV/runs/<run>/out/<unit>/ (idempotent)
    sync.py --run <name> sweep             the ISG_SWEEP records (processes per GPU vs frames/s)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

RV = Path(os.environ.get("RV", "/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1"))


def s5(*args, check=True):
    r = subprocess.run(os.environ["S5"].split() + list(args), capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(r.stderr[-400:])
    return r.stdout


def ls(url):
    out = s5("ls", url, check=False)
    return [line.split()[-1] for line in out.splitlines() if line.strip()]


def cat_json(url):
    try:
        return json.loads(s5("cat", url))
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("cmd", choices=("status", "pull", "sweep"))
    ap.add_argument("--units", default="")
    a = ap.parse_args()
    base = f"{os.environ['JOB_S3']}/runs/{a.run}"
    spec = json.load(open(RV / "runs" / a.run / "units.json"))
    units = {u["unit"]: u for u in spec["units"]}
    done = {n[:-5] for n in ls(f"{base}/done/*")}
    if a.cmd == "status":
        claims = {n[:-5] for n in ls(f"{base}/claims/*")}
        fails = ls(f"{base}/fail/*")
        print(f"{a.run}: units {len(done & set(units))}/{len(units)} done, {len(claims - done)} running, {len(fails)} fail records")
        for n in ls(f"{base}/hb/*"):
            hb = cat_json(f"{base}/hb/{n}") or {}
            age = time.time() - hb.get("time", 0)
            run = [(r.get("unit"), r.get("phase"), r.get("done"), r.get("n")) for r in hb.get("running") or []]
            print(f"  {n[:-5]} age {age:.0f}s phase {hb.get('phase')} {hb.get('detail', '')} gpus {hb.get('gpus')} procs/gpu "
                  f"{hb.get('procs_per_gpu')} done {hb.get('done')} failed {hb.get('failed')} frames {hb.get('frames_done')} "
                  f"gpu_util {hb.get('gpu_util')} mem {hb.get('gpu_mem_mb')} running {run}")
        for uid in sorted(done & set(units)):
            d = cat_json(f"{base}/done/{uid}.json") or {}
            print(f"  DONE {uid}: {d.get('frames')} frames kept {d.get('kept')} in {d.get('seconds', 0):.0f}s on {d.get('gpu_name')} "
                  f"steady {d.get('steady_fps')} fps, replay {d.get('env_steps_per_s')} fps, load {d.get('startup_s')} s")
    elif a.cmd == "pull":
        want = [u for u in (a.units.split(",") if a.units else sorted(done)) if u in done]
        for uid in want:
            dst = RV / "runs" / a.run / "out" / uid
            if (dst / ".synced").exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            # replayed TROs have ~240-character names: s5cmd's temporary download name would pass 255, so they are streamed
            s5("sync", "--exclude", "*-tro_state.json", f"{base}/out/{uid}/*", f"{dst}/")
            for name in ls(f"{base}/out/{uid}/instances/*"):
                (dst / "instances").mkdir(exist_ok=True)
                r = subprocess.run(os.environ["S5"].split() + ["cat", f"{base}/out/{uid}/instances/{name}"], capture_output=True, check=True)
                (dst / "instances" / name).write_bytes(r.stdout)
            (dst / "done.json").write_text(json.dumps(cat_json(f"{base}/done/{uid}.json"), indent=1))
            (dst / ".synced").touch()
            print(f"pulled {uid} -> {dst}")
    elif a.cmd == "sweep":
        for n in sorted(ls(f"{base}/sweep/*")):
            r = cat_json(f"{base}/sweep/{n}") or {}
            print(json.dumps({k: v for k, v in r.items() if k != "records"}))


if __name__ == "__main__":
    main()
