#!/usr/bin/env python3
"""Gather the reach-band files pulled from the ``rb`` pods (and local runs) into the committed runlog.

    python scripts/reach_bands/collect_bands.py --manifest runlogs/reach_bands_20260915/jobs.json [--manifest ...] \
        --pulled runlogs/runpod/rb --out runlogs/reach_bands_20260915

For every job directory ``<pulled>/<pod>/<job id>/`` (``collect.py --purpose rb --remote-dir /root/samp`` layout, or any
directory given with ``--extra``) the ``.npz`` + ``.json`` band files are copied into ``<out>/bands/`` (file name =
``<mode>__<scene>__<support>__<object>__<key>``), the job outcome (rc / wall time from ``done.txt``, the ``REACH_BANDS``
summary line of ``run.log`` or the exception) goes into ``<out>/results.json`` and ``<out>/results.csv``; the README
table is regenerated from them (``README.md`` keeps its hand-written head above the ``<!-- results -->`` marker).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def _done_lines(pod_dir: str) -> Dict[str, dict]:
    out = {}
    p = os.path.join(pod_dir, "done.txt")
    if not os.path.exists(p):
        return out
    for line in open(p):
        m = re.match(r"(\S+) rc=(-?\d+) wall=(\d+)", line.strip())
        if m:
            out[m.group(1)] = {"rc": int(m.group(2)), "wall_s": int(m.group(3))}
    return out


def _tail_error(log_path: str) -> Optional[str]:
    if not os.path.exists(log_path):
        return "no run.log"
    lines = open(log_path, errors="replace").read().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if "REACH_BANDS_FAILED" in lines[i]:
            # the exception line precedes the marker (traceback printed before it)
            for j in range(i - 1, max(-1, i - 40), -1):
                if lines[j] and not lines[j].startswith(" ") and ("Error" in lines[j] or "Exception" in lines[j] or "Exit" in lines[j]):
                    return lines[j][:300]
            return "REACH_BANDS_FAILED"
    if lines and ("Killed" in lines[-1] or "Segmentation" in " ".join(lines[-5:])):
        return lines[-1][:200]
    return None


def _current_code(rec: dict) -> bool:
    """Whether the band was written by the committed script (meta carries ``schedule``; the pods bootstrapped from the
    GitHub ``survey`` branch ran the original, un-synced script until 02:10 UTC)."""
    for base in rec.get("files", []):
        try:
            return "schedule" in json.load(open(base + ".json"))
        except (OSError, ValueError):
            return False
    return False


def scan_job_dir(jdir: str) -> dict:
    rec: dict = {"dir": jdir, "files": []}
    for js in sorted(glob.glob(os.path.join(jdir, "*.json"))):
        if os.path.exists(js[:-5] + ".npz"):
            rec["files"].append(js[:-5])
    log_path = os.path.join(jdir, "run.log")
    if os.path.exists(log_path):
        for line in open(log_path, errors="replace"):
            if line.startswith("REACH_BANDS {"):
                try:
                    rec["summary"] = json.loads(line[len("REACH_BANDS "):])
                except ValueError:
                    pass
    rec["error"] = _tail_error(log_path) if not rec["files"] else None
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", action="extend", required=True, help="manifests (several values and / or the flag repeated - both accumulate)")
    ap.add_argument("--pulled", default=os.path.join(REPO, "runlogs", "runpod", "rb"))
    ap.add_argument("--extra", nargs="*", default=[], help="extra job directories (local runs): <job id>=<dir>")
    ap.add_argument("--out", default=os.path.join(REPO, "runlogs", "reach_bands_20260915"))
    args = ap.parse_args()
    jobs: Dict[str, dict] = {}
    for m in args.manifest:
        for j in json.load(open(m)):
            jobs[j["id"]] = j
    bands_dir = os.path.join(args.out, "bands")
    os.makedirs(bands_dir, exist_ok=True)
    results: Dict[str, dict] = {}
    # pulled pods
    for pod_dir in sorted(glob.glob(os.path.join(args.pulled, "*"))):
        if not os.path.isdir(pod_dir) or not os.path.exists(os.path.join(pod_dir, "done.txt")):
            continue
        pod = os.path.basename(pod_dir)
        done = _done_lines(pod_dir)
        for jid, d in done.items():
            jdir = os.path.join(pod_dir, jid)
            rec = scan_job_dir(jdir) if os.path.isdir(jdir) else {"dir": jdir, "files": [], "error": "no output dir"}
            rec.update({"pod": pod, **d})
            prev = results.get(jid)
            if prev is None or (not prev["files"] and rec["files"]) or (rec["files"] and prev["files"] and _current_code(rec) and not _current_code(prev)):
                results[jid] = rec
    for spec in args.extra:
        jid, d = spec.split("=", 1)
        rec = scan_job_dir(d)
        rec.update({"pod": "local", "rc": 0 if rec["files"] else 1, "wall_s": None})
        results[jid] = rec
    # copy the band files
    n_copied = 0
    for jid, rec in results.items():
        for base in rec["files"]:
            for ext in (".npz", ".json"):
                dst = os.path.join(bands_dir, os.path.basename(base) + ext)
                shutil.copy2(base + ext, dst)
            n_copied += 1
        rec["band_files"] = [os.path.basename(b) for b in rec["files"]]
        rec["current_code"] = _current_code(rec) if rec["files"] else None
    # one band per factor set: a recomputation (new cache key, e.g. the pitched classes or the posture retry) supersedes
    # the older file of the same factors
    groups: Dict[tuple, List[str]] = defaultdict(list)
    for js in glob.glob(os.path.join(bands_dir, "*.json")):
        try:
            m = json.load(open(js))
        except ValueError:
            continue
        key = (m.get("scene"), m.get("support"), m.get("object"), m.get("grasp_key"), bool(m.get("generic_pinch")), tuple(sorted(m.get("placed_specs") or [])),
               m.get("under_partner"), json.dumps(m.get("scale")), m.get("arm", "right"), m.get("contact_set"), m.get("height_class"), m.get("target"), m.get("joint"),
               m.get("mode"))
        groups[key].append(js)
    n_superseded = 0
    for key, files in groups.items():
        if len(files) < 2:
            continue
        files.sort(key=lambda p: ("postures" in json.load(open(p)), os.path.getmtime(p)))  # newest / posture-aware last
        for old in files[:-1]:
            for ext in (".json", ".npz"):
                if os.path.exists(old[:-5] + ext):
                    os.remove(old[:-5] + ext)
            n_superseded += 1
    if n_superseded:
        print(f"{n_superseded} superseded band files removed from {bands_dir}")
    # table rows
    rows = []
    for jid, job in sorted(jobs.items()):
        rec = results.get(jid)
        row = {"id": jid, "kind": job["kind"], "wave": job.get("wave"), "subtasks": ";".join(job["subtasks"]), "status": "not run", "pod": "", "wall_s": "",
               "trunks_free": "", "trunks_total": "", "cells_total": "", "union_extent": "", "band_file": "", "error": "", "script": ""}
        if rec is not None:
            row.update({"pod": rec.get("pod", ""), "wall_s": rec.get("wall_s", "")})
            s = rec.get("summary")
            if rec["files"]:
                row["status"] = "ok" if (s is None or s.get("n_reachable_total", 0) > 0) else "empty band"
                row["script"] = "current" if rec.get("current_code") else "original (exhaustive rounds, no pair seeds)"
                row["band_file"] = rec["band_files"][0]
                if s:
                    tr = s.get("trunks", [])
                    row["trunks_total"] = len(tr)
                    row["trunks_free"] = sum(1 for t in tr if not t[3])
                    row["cells_total"] = s.get("n_reachable_total")
                    row["union_extent"] = json.dumps(s.get("union_extent"))
            else:
                row["status"] = "failed"
                row["error"] = (rec.get("error") or f"rc={rec.get('rc')}")[:200]
        rows.append(row)
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"jobs": rows, "raw": results}, f, indent=1, default=str)
    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    by_status = defaultdict(int)
    for r in rows:
        by_status[r["status"]] += 1
    print(f"{len(rows)} manifest jobs: {dict(by_status)}; {n_copied} band files copied to {bands_dir}")
    # README table
    readme = os.path.join(args.out, "README.md")
    head = ""
    if os.path.exists(readme):
        txt = open(readme).read()
        head = txt.split("<!-- results -->")[0]
    lines = [head.rstrip() + "\n\n<!-- results -->\n", "## Results per job (regenerated by `scripts/reach_bands/collect_bands.py`)\n",
             f"Status: {dict(by_status)}\n", "| job | kind | status | pod | wall s | trunks free/total | cells | union extent [λmin, λmax] | note |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        note = r["error"] or ("" if r["script"] in ("", "current") else r["script"])
        lines.append(f"| `{r['id']}` | {r['kind']} | {r['status']} | {r['pod']} | {r['wall_s']} | {r['trunks_free']}/{r['trunks_total']} | {r['cells_total']} | "
                     f"{r['union_extent']} | {note} |")
    with open(readme, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
