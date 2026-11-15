#!/usr/bin/env python3
"""Cluster lane of a batched-replay run (2026-11-15: Vast is shared with another session; cluster GPUs take the rest).

One Slurm GPU job runs this loop under either account (minyeong or shiyuc): take the next unit of the lane's list
(<lane>/units.txt, longest first) by an exclusive create of <lane>/claims/<unit> (content: the Slurm job id), replay it with the
Vast worker's flags (the inputs are on NFS: the eval_production tree, its templates), check the dataset and leave the unit in the
layout the merge reads: <lane>/out/<unit>/{<unit>/ (LeRobot v3), goal_images/, episodes_isg.jsonl, instances/, results.jsonl,
replay.log, done.json}. A claim whose Slurm job is no longer listed by squeue (pre-empted / killed) is taken over. Exits when no
unit is left. Units are R2-claimed by minyeong under owner "cluster" beforehand, so Vast workers never take them.

    cluster_lane.py --run <run> --lane <lane dir> [--max-units N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RV = Path("/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1")
PURPOSE_FLAGS = {  # = vast/worker.py PURPOSE_FLAGS
    "train": ["--scene-pert", "remove", "--scene-pert-plan", "auto", "--lighting-pert", "replay", "--lighting-skybox", "episode",
              "--pert-unit", "demo", "--scene-pert-inview-pick", "random", "--goal-occlusion", "auto"],
    "eval": ["--scene-pert", "off", "--lighting-pert", "off", "--goal-occlusion", "off"],
}
JOB = os.environ.get("SLURM_JOB_ID", f"local{os.getpid()}")
UNITS_FILE = [None]
STARTUP_FAILS = [0]  # consecutive failures of this job with nothing replayed  # the lane's unit list (minyeong-side, read by either account)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}", flush=True)


def live_jobs():
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%A"], capture_output=True, text=True, timeout=60)
        return set(out.stdout.split()) if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def claim(lane: Path, uid: str) -> bool:
    c = lane / "claims" / uid
    try:
        fd = os.open(c, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, JOB.encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def next_unit(lane: Path, units: dict):
    order = [u for u in Path(UNITS_FILE[0]).read_text().split() if u in units]
    done = {p.name for p in (lane / "out").iterdir() if (p / "done.json").exists()} if (lane / "out").exists() else set()
    for uid in order:
        if uid in done or (lane / "dead" / uid).exists():
            continue
        if claim(lane, uid):
            return uid
    # stale claims (their Slurm job is gone): take them over
    jobs = live_jobs()
    if jobs is None:
        return None
    for uid in order:
        c = lane / "claims" / uid
        if uid in done or not c.exists() or (lane / "dead" / uid).exists():
            continue
        owner = c.read_text().strip()
        if owner != JOB and owner not in jobs:
            try:
                c.unlink()
            except FileNotFoundError:
                pass
            if claim(lane, uid):
                log(f"took over {uid} from gone job {owner}")
                return uid
    return None


def validate(ds: Path, kept, image_size):
    import pyarrow.parquet as pq

    info = json.load(open(ds / "meta/info.json"))
    lengths = [int(r["length"]) for r in kept]
    if info["total_episodes"] != len(lengths) or info["total_frames"] != sum(lengths):
        raise ValueError(f"info: {info['total_episodes']} / {info['total_frames']} vs {len(lengths)} / {sum(lengths)}")
    feats = info["features"]
    cams = ("zed_link", "left_realsense_link", "right_realsense_link")
    need = [f"observation.rgb.{c}_camera_0" for c in cams] + [f"observation.depth_linear.{c}_camera_0" for c in cams]
    if any(k not in feats for k in need) or any(list(feats[k]["shape"][:2]) != [image_size, image_size] for k in need):
        raise ValueError("camera streams missing or not at the image size")
    eps = sorted((r for p in sorted((ds / "meta/episodes").glob("*/*.parquet")) for r in pq.read_table(p).to_pylist()),
                 key=lambda r: r["episode_index"])
    if [int(r["length"]) for r in eps] != lengths:
        raise ValueError("episode lengths differ from the replayer's records")
    return {"episodes": len(lengths), "frames": sum(lengths), "bytes": sum(f.stat().st_size for f in ds.rglob("*") if f.is_file())}


TPL_EXCLUDE = ("local/logs", "tmp", "local/data/documents/Kit/shared/screenshots", "global/cache/texturecache")


def copy_new(src: Path, dst: Path, tag: str) -> int:
    """dst/<name> <- src unless it exists: a temporary name, then a hard link into place (never replaces a file others read)."""
    if dst.exists() or dst.is_symlink():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{tag}.tmp"
    shutil.copyfile(src, tmp)
    os.chmod(tmp, 0o444)
    try:
        os.link(tmp, dst)
        n = 1
    except FileExistsError:
        n = 0
    os.unlink(tmp)
    return n


def publish_cache_deltas() -> str:
    """Owner 2026-11-15: warm Kit starts for later jobs, delta only. Textures this job's Kit converted itself (regular files in its
    texture cache; the store's are links) go into ISGREND_TEX_OUT, data files first and .desc last; the app-data files (shader /
    Kit caches; logs, tmp, screenshots and textures left out) missing from ISGREND_TPL_OUT/<KIT_KEY> are added there, and a new key
    is marked .complete once its first copy is in. Never overwrites."""
    app = Path(os.environ.get("OMNIGIBSON_APPDATA_PATH", ""))
    key, tex_out, tpl_out = os.environ.get("KIT_KEY"), os.environ.get("ISGREND_TEX_OUT"), os.environ.get("ISGREND_TPL_OUT")
    if not (app.is_dir() and key and tex_out and tpl_out) or "nob1k" in key:
        return "skipped"
    t0, tag = time.time(), f"{JOB}.{os.getpid()}"
    n_tex = n_tpl = 0
    try:
        tex = app / "global/cache/texturecache"
        store = Path(tex_out)
        have = set(os.listdir(store)) if store.is_dir() else set()
        parts = {}
        for name in os.listdir(tex) if tex.is_dir() else []:
            stem, sep, _ = name.partition(".ovtex.")
            if sep and not (tex / name).is_symlink():
                parts.setdefault(stem, []).append(name)
        for stem, names in parts.items():
            desc = stem + ".ovtex.desc"
            if desc not in names or desc in have:
                continue
            for name in sorted(n for n in names if n != desc) + [desc]:
                copy_new(tex / name, store / name, tag)
            n_tex += 1
        dst = Path(tpl_out) / key
        new_key = not (dst / ".complete").exists()
        subprocess.run(["chmod", "-R", "u+rwX", str(app)], capture_output=True)
        for root, dirs, files in os.walk(app):
            rel = os.path.relpath(root, app)
            dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(rel, d)) not in TPL_EXCLUDE]
            for f in files:
                src = Path(root) / f
                if src.is_symlink() or not src.is_file():
                    continue
                n_tpl += copy_new(src, dst / rel / f, tag)
        if new_key and n_tpl:
            (dst / ".complete").touch()
    except OSError as e:
        return f"failed: {e}"
    return f"textures +{n_tex}, app-data files +{n_tpl} ({key}) in {time.time() - t0:.0f}s"


def run_unit(lane: Path, spec: dict, u: dict, code: Path) -> bool:
    uid = u["unit"]
    tmp = Path(os.environ.get("ISGREND_SCRATCH", f"/tmp/{os.environ.get('USER')}/isgrend_{JOB}")) / uid
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    out = tmp / "out"
    flags = spec.get("replay_flags") or {}
    size = int(flags.get("image_size", 224))
    (tmp / "demos.txt").write_text("\n".join(d["dir"] for d in u["demos"]) + "\n")
    shutil.copy(RV / "runs" / spec["run"] / u["meta"], tmp / "instance_meta.jsonl")
    cmd = [sys.executable, "-u", "-m", "ispatialgym.multienv.replay", "--demos", f"@{tmp / 'demos.txt'}", "--root", str(out),
           "--repo-prefix", uid, "--task", u["task"], "--slots", "1", "--batched-usd-edits", "--async-flush", "--proprio", "loaded",
           "--warmup-s", "15", "--scene", "full", "--image-size", str(size), "--settle-pad", str(flags.get("settle_pad", 0.2)),
           "--settle-pad-min", str(flags.get("settle_pad_min", 50)), "--goal-images", "on",
           "--instance-meta", str(tmp / "instance_meta.jsonl"), "--results", str(out / "results.jsonl"),
           "--scene-pert-seed", str(spec.get("seed", 20261115))] + PURPOSE_FLAGS[u["purpose"]] + list(u.get("extra_args") or [])
    t0 = time.time()
    log(f"START {uid} purpose={u['purpose']} demos={len(u['demos'])} frames={u['frames']}")
    with open(tmp / "replay.log", "wb") as fh:
        rc = subprocess.run(cmd, cwd=str(code), stdout=fh, stderr=subprocess.STDOUT).returncode
    text = (tmp / "replay.log").read_text(errors="replace")
    recs = [json.loads(line.split(" ", 1)[1]) for line in text.splitlines() if line.startswith("REPLAY_DEMO ")]
    summary = next((json.loads(line.split(" ", 1)[1]) for line in text.splitlines() if line.startswith("REPLAY_SUMMARY ")), None)
    slot0 = out / f"{uid}_slot0"
    try:
        # og.shutdown() after REPLAY_SUMMARY can segfault in Kit's temp-dir cleanup; the writer has finalized the dataset by then
        if not (summary and len(recs) == len(u["demos"]) and slot0.is_dir()):
            raise RuntimeError(f"rc={rc} demos {len(recs)}/{len(u['demos'])} summary={bool(summary)}")
        eps = [json.loads(x) for x in (out / "episodes_isg.jsonl").read_text().splitlines() if x.strip()]
        kept = sorted((e for e in eps if e.get("kept")), key=lambda e: e["episode_index"])
        slot0.rename(out / uid)
        stats = validate(out / uid, kept, size)
        missing = [e["episode_index"] for e in kept if not (out / e["goal_image"]["rgb_path"]).is_file()]
        if missing:
            raise ValueError(f"goal images missing {missing[:5]}")
    except Exception as e:  # noqa: BLE001
        (lane / "fail").mkdir(exist_ok=True)
        startup = not recs  # nothing replayed: the job / host, not the unit (does not count toward dead)
        kind = "startup" if startup else "unit"
        n = len(list((lane / "fail").glob(f"{uid}.*.unit.txt")))
        shutil.copy(tmp / "replay.log", lane / "fail" / f"{uid}.{n}.{JOB}.{kind}.log")
        (lane / "fail" / f"{uid}.{n}.{JOB}.{kind}.txt").write_text(f"{type(e).__name__}: {e}\n")
        log(f"FAIL ({kind}) {uid}: {e}")
        if not startup and n + 1 >= 3:
            (lane / "dead").mkdir(exist_ok=True)
            (lane / "dead" / uid).write_text(str(e))
        (lane / "claims" / uid).unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        STARTUP_FAILS[0] = STARTUP_FAILS[0] + 1 if startup else 0
        return False
    STARTUP_FAILS[0] = 0
    shutil.copy(tmp / "replay.log", out / "replay.log")
    dst = lane / "out" / uid
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(out, dst)
    done = {"unit": uid, "lane": str(lane), "job": JOB, "host": os.uname().nodename, "user": os.environ.get("USER"),
            "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True,
                                  text=True).stdout.strip(),
            "seconds": round(time.time() - t0, 1), "purpose": u["purpose"], "task": u["task"], "kept": len(kept),
            "dropped": len(u["demos"]) - len(kept), "summary": summary, "replayer": "ispatialgym.multienv.replay", "code": str(code),
            "rc": rc, **stats, "time": time.time()}
    (dst / "done.json").write_text(json.dumps(done, indent=1))
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"DONE {uid}: {stats['frames']} frames in {done['seconds']:.0f}s, replay {summary.get('env_steps_per_s')} fps, load "
        f"{summary.get('load_s')} s")
    log(f"CACHE_DELTA {publish_cache_deltas()}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--max-units", type=int, default=0)
    ap.add_argument("--units-file", required=True)
    a = ap.parse_args()
    UNITS_FILE[0] = a.units_file
    os.umask(0o022)
    lane = Path(a.lane)
    for d in ("claims", "out", "fail", "dead"):
        (lane / d).mkdir(parents=True, exist_ok=True)
    spec = json.load(open(RV / "runs" / a.run / "units.json"))
    spec["run"] = a.run
    units = {u["unit"]: u for u in spec["units"]}
    code = Path(os.environ["ISGE_CODE"])
    n = 0
    while not (lane / "STOP").exists() and (not a.max_units or n < a.max_units):
        uid = next_unit(lane, units)
        if uid is None:
            log("no unit left")
            break
        run_unit(lane, spec, units[uid], code)
        n += 1
        if STARTUP_FAILS[0] >= 2:
            log("2 consecutive startup failures (nothing replayed): this job / host is unusable; exiting without taking more units")
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
