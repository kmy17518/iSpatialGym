#!/usr/bin/env python3
"""Head-camera videos of successes and near misses for the coordination repo (user, 2026-10-07 18:30Z, item 5).

Candidates: relocalization episode records with success, or a final position error below ``--near-m`` (0.5 m), from runs whose
step is >= ``--min-step`` (100k checkpoints) or from the execution-settings test (``tag`` set). At most ``--max-files`` files in
total under lab/videos (existing exports count), each re-encoded with ffmpeg (libx264, crf 28) when larger than ``--max-mb``.
Destination: lab/videos/<run>/step_XXXXXXXX/episode_XXXXXX.mp4, test runs lab/videos/exec/<tag>/<run>/step_XXXXXXXX/episode_XXXXXX.mp4;
lab/videos/index.json lists every export (record path, success, errors, size).

    python lab_tools/export_videos.py [--max-files 15] [--reserve 5] [--dry-run]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK = "camera_relocalization-standard"
OUT = os.path.join(WT, "lab", "videos")


def candidates(min_step: int, near_m: float):
    paths = sorted(set(glob.glob(os.path.join(WT, "lab", "tierB*", "*", "step_*", TASK, "episode_*.json")) +
                       glob.glob(os.path.join(WT, "lab", "diagnostics", "exec", "*", "*", "step_*", TASK, "episode_*.json"))))
    out = []
    for p in paths:
        r = json.load(open(p))
        if r.get("error") or not r.get("video"):
            continue
        pe = (r.get("score") or {}).get("position_error_m")
        is_test = bool(r.get("tag"))
        if not (is_test or (r.get("step") or 0) >= min_step):
            continue
        if not (r.get("success") or (pe is not None and pe < near_m)):
            continue
        out.append({"record": os.path.relpath(p, WT), "run": r["run"], "step": r["step"], "tag": r.get("tag"), "episode": r["episode_index"], "split": r.get("split"),
                    "success": bool(r.get("success")), "position_error_m": pe, "orientation_error_deg": (r.get("score") or {}).get("orientation_error_deg"),
                    "geodesic_distance_m": (r.get("score") or {}).get("geodesic_distance_m"), "video": r["video"]})
    # successes first, then the closest misses
    out.sort(key=lambda c: (not c["success"], c["position_error_m"] if c["position_error_m"] is not None else 9e9))
    return out


def dest_of(c):
    if c["tag"]:
        return os.path.join(OUT, "exec", c["tag"], c["run"], f"step_{int(c['step']):08d}", f"episode_{int(c['episode']):06d}.mp4")
    return os.path.join(OUT, c["run"], f"step_{int(c['step']):08d}", f"episode_{int(c['episode']):06d}.mp4")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-files", type=int, default=15)
    ap.add_argument("--reserve", type=int, default=0, help="slots kept free for later exports (e.g. the execution-settings test)")
    ap.add_argument("--max-mb", type=float, default=20.0)
    ap.add_argument("--min-step", type=int, default=100000)
    ap.add_argument("--near-m", type=float, default=0.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    index_path = os.path.join(OUT, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {"format": "isg-lab-videos/v1", "files": []}
    have = {f["record"] for f in index["files"]}
    budget = args.max_files - args.reserve - len(index["files"])
    cands = [c for c in candidates(args.min_step, args.near_m) if c["record"] not in have]
    print(f"candidates {len(cands)} (successes {sum(c['success'] for c in cands)}), exported so far {len(index['files'])}, budget now {budget}")
    for c in cands[: max(0, budget)]:
        src = os.path.join(WT, c["video"])
        dst = dest_of(c)
        if not os.path.exists(src):
            print(f"  missing video {src}")
            continue
        mb = os.path.getsize(src) / 1e6
        print(f"  {'OK ' if c['success'] else 'near'} {c['run']} {c['step']} {c['tag'] or ''} ep {c['episode']} pos {c['position_error_m']:.3f} m  {mb:.1f} MB -> {os.path.relpath(dst, WT)}")
        if args.dry_run:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        reencoded = False
        if mb > args.max_mb:
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-crf", "28", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an", dst], capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(dst):
                print(f"    ffmpeg failed: {r.stderr[-300:]}")
                continue
            reencoded = True
            if os.path.getsize(dst) / 1e6 > args.max_mb:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-crf", "34", "-preset", "veryfast", "-vf", "scale=360:360", "-pix_fmt", "yuv420p", "-an", dst], capture_output=True, text=True)
        else:
            shutil.copy2(src, dst)
        index["files"].append({**c, "path": os.path.relpath(dst, WT), "size_mb": round(os.path.getsize(dst) / 1e6, 2), "reencoded": reencoded, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if not args.dry_run:
        os.makedirs(OUT, exist_ok=True)
        index["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(index_path, "w") as f:
            json.dump(index, f, indent=1)
    print(f"exported {len(index['files'])} files in total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
