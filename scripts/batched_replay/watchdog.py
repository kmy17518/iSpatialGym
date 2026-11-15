#!/usr/bin/env python3
"""Watchdog of a batched-replay run (owner 2026-11-15: stalled / problematic units are reassigned; short ones to a free cluster lane,
long ones, or any when little time is left, to a fast Vast GPU). Runs as minyeong (source $RV/tool/creds.sh):

    watchdog.py --run <run> --lane shiyuc=<lane dir> [--every 120] [--dry-run]

Every tick it reads the R2 heartbeats / claims / done records (no Vast API call) and the cluster lanes' job logs, and acts:

Vast unit (a running entry of an instance's heartbeat):
  stalled   no new demo for --stall-s (first demo: --startup-s since the unit started)
  slow      in the tail (no open unit left): its remaining time at its own rate exceeds --tail-eta-s and a restart elsewhere would
            take under half of it
  -> ctl/stop_<instance> (the worker claims nothing more); once the instance runs nothing but flagged units (or after --drain-s)
     it is destroyed (vastctl, ledger-scoped), the flagged units' claims are released and each unit is placed: a short unit
     (<= --short-frames) goes to a live cluster lane (R2 claim cluster:<lane> + the front of the lane's list), any other stays open
     for Vast, whose workers take the longest open unit first (the loop rents again while units are open).
Cluster lane job (a Slurm job of the lane; its log START / DONE / FAIL lines):
  stalled   its current unit runs longer than max(--lane-stall-s, frames / 2.5 + 600)
            -> the job is cancelled, the unit leaves the lane's list and its R2 claim is released (Vast takes it)
  slow      a finished unit replayed at < --lane-min-fps (steady fps of the DONE line)
            -> the job is cancelled at its next unit start (the lane's other jobs take its claim over)
  shiyuc's jobs are cancelled by its own helper (watchdog_scancel.sh, explicit ids listed in $RV/watchdog/<run>_scancel.txt);
  minyeong's directly by id.
Every decision is appended to $RV/watchdog/<run>_actions.jsonl.
"""
from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vastctl as V  # noqa: E402

RV = V.W
LONG_FRAMES = 4000
LANE_LINE = re.compile(r"^\[(?P<t>[0-9T:\-]+)Z\] (?P<kind>START|DONE|FAIL) (?:\([a-z]+\) )?(?P<uid>u\d{4}_[^ :]+)")


def ts(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%S"))


def act(a, kind, **kw):
    rec = {"time": time.time(), "kind": kind, "dry": a.dry_run, **kw}
    print(f"[{time.strftime('%H:%M:%S')}] {kind} {json.dumps(kw)}", flush=True)
    with open(RV / "watchdog" / f"{a.run}_actions.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def lane_jobs(lane_name, user):
    out = subprocess.run(["squeue", "-u", user, "-h", "-n", "isgrend_lane", "-o", "%i %T"], capture_output=True, text=True).stdout
    return {line.split()[0]: line.split()[1] for line in out.splitlines() if line.strip()}


def lane_state(log_dir: Path, jobs):
    """job id -> {"current": (uid, start_time) | None, "fps": [steady fps of finished units]}"""
    st = {}
    for j in jobs:
        f = log_dir / f"{j}.out"
        cur, fps = None, []
        if f.exists():
            for line in f.read_text(errors="replace").splitlines():
                m = LANE_LINE.match(line)
                if not m:
                    continue
                if m["kind"] == "START":
                    cur = (m["uid"], ts(m["t"]))
                else:
                    cur = None
                    r = re.search(r"DONE \S+: (\d+) frames .*replay ([0-9.]+) fps", line)
                    # only long units measure the GPU: a short unit's replay rate is dominated by its per-demo restores
                    if m["kind"] == "DONE" and r and int(r.group(1)) >= LONG_FRAMES:
                        fps.append(float(r.group(2)))
        st[j] = {"current": cur, "fps": fps}
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--lane", action="append", default=[], help="name=<lane dir> (shiyuc lanes: name shiyuc)")
    ap.add_argument("--every", type=int, default=120)
    ap.add_argument("--stall-s", type=float, default=900)
    ap.add_argument("--startup-s", type=float, default=1800)
    ap.add_argument("--tail-eta-s", type=float, default=1500)
    ap.add_argument("--drain-s", type=float, default=1200)
    ap.add_argument("--drain-max-running", type=int, default=3, help="drain only instances running at most this many units "
                    "(any instance once no unit is open)")
    ap.add_argument("--short-frames", type=int, default=2500)
    ap.add_argument("--fast-fps", type=float, default=12.0, help="per-process frames/s of a fast Vast GPU (restart estimate)")
    ap.add_argument("--lane-stall-s", type=float, default=1800)
    ap.add_argument("--lane-min-fps", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    V.RUN[0] = a.run
    V.PREFIX = f"{V.ROOT}/runs/{a.run}"
    (RV / "watchdog").mkdir(exist_ok=True)
    c, s3 = V.vclients(), V.r2()
    units = {u["unit"]: u for u in json.load(open(RV / "runs" / a.run / "units.json"))["units"]}
    lanes = dict(x.split("=", 1) for x in a.lane)
    sp = RV / "watchdog" / f"{a.run}_state.json"
    state = json.load(open(sp)) if sp.exists() else {"prog": {}, "draining": {}, "slow_jobs": [], "cancelled": []}
    bucket = os.environ["S3_BUCKET"]

    def put(key, body):
        if not a.dry_run:
            s3.put_object(Bucket=bucket, Key=f"{V.PREFIX}/{key}", Body=json.dumps(body).encode())

    def rm(key):
        if not a.dry_run:
            s3.delete_object(Bucket=bucket, Key=f"{V.PREFIX}/{key}")

    def units_file(name):
        return RV / "cluster_lanes" / a.run / f"{name}.units.txt"

    def place(uid, why):
        """A unit whose claim was just released: a short one to a live cluster lane, else the Vast queue."""
        u = units[uid]
        live = {n: [j for j, s in lane_jobs(n, "shiyuc" if n == "shiyuc" else "minyeong").items() if s == "RUNNING"]
                for n in lanes}
        target = next((n for n, js in live.items() if js), None) if u["frames"] <= a.short_frames else None
        if target:
            put(f"claims/{uid}.json", {"instance": f"cluster:{target}", "lane": lanes[target], "time": time.time(),
                                       "watchdog": why})
            uf = units_file(target)
            listed = uf.read_text().split() if uf.exists() else []
            if not a.dry_run:
                uf.write_text("".join(x + "\n" for x in [uid] + [x for x in listed if x != uid]))
            act(a, "place_cluster", unit=uid, lane=target, frames=u["frames"], why=why)
        else:
            act(a, "place_vast", unit=uid, frames=u["frames"], why=why)

    def cancel_job(job, user, why):
        if job in state["cancelled"]:
            return
        state["cancelled"].append(job)
        if user == "shiyuc":
            if not a.dry_run:
                with open(RV / "watchdog" / f"{a.run}_scancel.txt", "a") as f:
                    f.write(job + "\n")
        elif not a.dry_run:
            subprocess.run(["scancel", job])
        act(a, "cancel_lane_job", job=job, user=user, why=why)

    while True:
        t = time.time()
        try:
            done = {k[:-5] for k in V.r2_list(s3, "done/")} & set(units)
            claims = {k[:-5] for k in V.r2_list(s3, "claims/")} & set(units)
            dead = {k[:-5] for k in V.r2_list(s3, "dead/")} & set(units)
            n_open = len(set(units) - done - claims - dead)
            hbs = V.heartbeats(s3)
            # ---------- Vast units ----------
            running = {}
            for iid, hb in hbs.items():
                if not iid.isdigit() or t - hb.get("time", 0) > V.HB_STALE_S or hb.get("phase") in ("destroyed", "done", "failed"):
                    continue
                for r in hb.get("running") or []:
                    if r.get("unit") in units and r["unit"] not in done:
                        running[r["unit"]] = (iid, r)
            flagged = {}
            for uid, (iid, r) in running.items():
                p = state["prog"].get(uid)
                if not p or p["inst"] != iid or p["done"] != r.get("done", 0) or p.get("started") != r.get("started"):
                    p = state["prog"][uid] = {"inst": iid, "done": r.get("done", 0), "since": t, "started": r.get("started")}
                n, d = max(1, int(r.get("n") or 1)), int(r.get("done") or 0)
                if d == 0:
                    if t - float(r.get("started") or t) > a.startup_s:
                        flagged[uid] = f"no demo {(t - r['started']) / 60:.0f} min after start ({r.get('phase')})"
                    continue
                if t - p["since"] > a.stall_s:
                    flagged[uid] = f"stalled: no new demo for {(t - p['since']) / 60:.0f} min ({d}/{n} demos)"
                    continue
                if n_open == 0:
                    rate = r.get("steady_fps") or (d / n * r["frames"]) / max(1.0, t - float(r["started"]))
                    eta = (1 - d / n) * r["frames"] / max(0.1, float(rate))
                    restart = 300 + r["frames"] / a.fast_fps
                    if eta > a.tail_eta_s and restart < 0.5 * eta:
                        flagged[uid] = f"slow in the tail: eta {eta / 60:.0f} min at {rate:.1f} fps, restart ~{restart / 60:.0f} min"
            by_inst = {}
            for uid, (iid, r) in running.items():
                by_inst.setdefault(iid, []).append(uid)
            for uid, why in flagged.items():
                iid = running[uid][0]
                if iid not in state["draining"] and n_open > 0 and len(by_inst.get(iid, [])) > a.drain_max_running:
                    # a big instance: draining would idle all its GPUs; the worker kills a stalled process itself (stall_s)
                    if uid not in state.setdefault("left", []):
                        state["left"].append(uid)
                        act(a, "stall_left_to_worker", instance=iid, unit=uid, why=why, running=len(by_inst.get(iid, [])))
                    continue
                if iid not in state["draining"]:
                    state["draining"][iid] = {"since": t, "units": [], "why": why}
                    put(f"ctl/stop_{iid}", {"by": "watchdog", "why": why, "time": t})
                    act(a, "stop_instance_claims", instance=iid, unit=uid, why=why, running=by_inst.get(iid))
                if uid not in state["draining"][iid]["units"]:
                    state["draining"][iid]["units"].append(uid)
            for iid, dr in list(state["draining"].items()):
                still = by_inst.get(iid, [])
                healthy = [u for u in still if u not in dr["units"] and u not in flagged]
                gone = iid not in hbs or t - hbs[iid].get("time", 0) > V.HB_STALE_S
                if healthy and not gone and t - dr["since"] < a.drain_s:
                    continue
                if not gone:
                    ok = a.dry_run or V.destroy(c, int(iid), f"watchdog: {dr['why']}")
                    act(a, "destroy_instance", instance=iid, ok=bool(ok), units=dr["units"], healthy_lost=healthy)
                rel = [u for u in set(dr["units"]) | set(healthy) if u not in done]
                for uid in rel:
                    rm(f"claims/{uid}.json")
                    state["prog"].pop(uid, None)
                    place(uid, f"{dr['why']} (instance {iid})")
                rm(f"ctl/stop_{iid}")
                del state["draining"][iid]
            # ---------- cluster lanes ----------
            for name, ldir in lanes.items():
                user = "shiyuc" if name == "shiyuc" else "minyeong"
                jobs = lane_jobs(name, user)
                log_dir = Path("/vision/u/shiyuc/minyeong/isgrend/logs") if user == "shiyuc" else RV / "cluster_lanes" / a.run / "jobs"
                for job, s in lane_state(log_dir, [j for j, st in jobs.items() if st == "RUNNING"]).items():
                    if job in state["cancelled"]:
                        continue
                    cur = s["current"]
                    if cur and cur[0] in units and cur[0] not in done:
                        lim = max(a.lane_stall_s, units[cur[0]]["frames"] / 2.5 + 600)
                        if t - cur[1] > lim:
                            uf = units_file(name)
                            if not a.dry_run:
                                uf.write_text("".join(x + "\n" for x in uf.read_text().split() if x != cur[0]))
                            cancel_job(job, user, f"unit {cur[0]} running {(t - cur[1]) / 60:.0f} min (limit {lim / 60:.0f})")
                            rm(f"claims/{cur[0]}.json")
                            act(a, "lane_unit_to_vast", unit=cur[0], frames=units[cur[0]]["frames"], job=job)
                            continue
                    if s["fps"] and min(s["fps"][-2:]) < a.lane_min_fps:
                        if job not in state["slow_jobs"]:
                            state["slow_jobs"].append(job)
                            act(a, "lane_job_slow", job=job, fps=s["fps"][-3:])
                        if not cur or t - cur[1] < 300:
                            cancel_job(job, user, f"slow: {s['fps'][-3:]} fps")
            act_summary = {"open": n_open, "done": len(done), "claimed": len(claims - done), "vast_running": len(running),
                           "flagged": len(flagged), "draining": list(state["draining"])}
            print(f"[{time.strftime('%H:%M:%S')}] tick {json.dumps(act_summary)}", flush=True)
            if len(done) >= len(units):
                print("ALL UNITS DONE")
                return
        except Exception as e:  # noqa: BLE001
            print(f"[{time.strftime('%H:%M:%S')}] watchdog error: {type(e).__name__}: {e}", flush=True)
        sp.write_text(json.dumps(state))
        time.sleep(a.every)


if __name__ == "__main__":
    main()
