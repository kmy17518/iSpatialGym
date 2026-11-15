#!/usr/bin/env python3
"""Cluster lanes of a batched-replay run (minyeong side, with the R2 credentials; source $RV/tool/creds.sh).

    cluster_assign.py --run <run> assign --lane name=<lane dir>:<weight> [...] [--vast-share 0.25]
        open units (not done / claimed / dead on R2), longest first, dealt to the Vast queue (left open on R2, --vast-share of the
        frames) and to the lanes by weight; a lane's units are R2-claimed under owner "cluster:<name>" (vastctl's reaper leaves
        them alone) and appended to <lane dir>/units.txt
    cluster_assign.py --run <run> collect --lane name=<lane dir> [...]
        finished lane units -> $RV/runs/<run>/out/<unit>/ (the layout sync.py pulls Vast units into) and R2 done/<unit>.json
    cluster_assign.py --run <run> release --lane name=<lane dir> --units a,b|--unstarted
        give lane units back to the Vast queue (drop them from units.txt, delete their R2 claims)
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

RV = Path(os.environ.get("RV", "/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1"))


def s5(*args, check=True, inp=None):
    r = subprocess.run(os.environ["S5"].split() + list(args), capture_output=True, text=True, input=inp)
    if check and r.returncode:
        raise RuntimeError(r.stderr[-300:])
    return r.stdout


def names(base, sub):
    return {line.split()[-1][:-5] for line in s5("ls", f"{base}/{sub}/*", check=False).splitlines() if line.strip()}


def units_file(run, name):
    p = RV / "cluster_lanes" / run / f"{name}.units.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def lanes_of(specs):
    out = []
    for s in specs:
        name, rest = s.split("=", 1)
        path, _, w = rest.partition(":")
        out.append((name, Path(path), float(w or 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("cmd", choices=("assign", "collect", "release"))
    ap.add_argument("--lane", action="append", default=[])
    ap.add_argument("--vast-share", type=float, default=0.25)
    ap.add_argument("--units", default="")
    ap.add_argument("--unstarted", action="store_true")
    a = ap.parse_args()
    base = f"{os.environ['JOB_S3']}/runs/{a.run}"
    spec = json.load(open(RV / "runs" / a.run / "units.json"))
    units = {u["unit"]: u for u in spec["units"]}
    lanes = lanes_of(a.lane)
    if a.cmd == "assign":
        taken = names(base, "done") | names(base, "claims") | names(base, "dead")
        open_units = sorted((u for uid, u in units.items() if uid not in taken), key=lambda u: -u["frames"])
        total = sum(u["frames"] for u in open_units)
        wsum = sum(w for _, _, w in lanes)
        target = {"vast": a.vast_share * total, **{n: (1 - a.vast_share) * total * w / wsum for n, _, w in lanes}}
        got = {k: 0 for k in target}
        plan = {k: [] for k in target}
        for u in open_units:  # longest first to the share furthest below its target
            k = min(target, key=lambda k: got[k] / max(1.0, target[k]))
            plan[k].append(u["unit"])
            got[k] += u["frames"]
        for name, path, _ in lanes:
            ok = []
            for uid in plan[name]:
                body = json.dumps({"instance": f"cluster:{name}", "lane": str(path), "time": time.time()})
                r = subprocess.run(os.environ["S5"].split() + ["pipe", f"{base}/claims/{uid}.json"], input=body, text=True,
                                   capture_output=True)
                if r.returncode == 0:
                    ok.append(uid)
            uf = units_file(a.run, name)
            with open(uf, "a") as f:
                f.write("".join(uid + "\n" for uid in ok))
            print(f"lane {name}: {len(ok)} units, {sum(units[u]['frames'] for u in ok)} frames -> {uf} (work dir {path})")
        print(f"vast queue: {len(plan['vast'])} units, {got['vast']} frames left open; total open was {len(open_units)} units, {total} frames")
    elif a.cmd == "collect":
        n = 0
        for name, path, _ in lanes:
            for d in sorted((path / "out").glob("*")) if (path / "out").exists() else []:
                uid = d.name
                dst = RV / "runs" / a.run / "out" / uid
                if not (d / "done.json").exists() or (dst / ".synced").exists():
                    continue
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(d, dst)
                (dst / ".synced").touch()
                done = json.load(open(d / "done.json"))
                s5("pipe", f"{base}/done/{uid}.json", inp=json.dumps({**done, "instance": f"cluster:{name}"}))
                n += 1
                print(f"collected {uid} from {name} ({done.get('frames')} frames, {done.get('seconds')} s on {done.get('gpu')})")
            dead = sorted(p.name for p in (path / "dead").glob("*")) if (path / "dead").exists() else []
            if dead:
                print(f"lane {name}: DEAD units {dead}")
        print(f"collected {n}")
    elif a.cmd == "release":
        for name, path, _ in lanes:
            listed = units_file(a.run, name).read_text().split()
            started = {p.name for p in (path / "claims").glob("*")} | {p.name for p in (path / "out").glob("*")}
            give = set(a.units.split(",")) if a.units else {u for u in listed if u not in started} if a.unstarted else set()
            give &= set(listed) - started
            units_file(a.run, name).write_text("".join(u + "\n" for u in listed if u not in give))
            for uid in give:
                s5("rm", f"{base}/claims/{uid}.json", check=False)
            print(f"lane {name}: released {len(give)} units to the Vast queue")


if __name__ == "__main__":
    main()
