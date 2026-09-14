#!/usr/bin/env python3
"""Final step of the world-alignment selection run: aggregate the wall survey, write the sheet, and drop the marker
``DONE.json`` (+ ``DONE`` text) in the runlog directory with a timestamp and a summary of everything produced.

Usage:  python3 finalize.py [--wait]      (--wait blocks until no survey process is running)
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
SEL_CSV = os.path.join(REPO, "runlogs", "grasp_library_20260905", "select", "task_object_selection.csv")
BASE_PY = os.path.expanduser("~/miniconda3/bin/python")
SCENES = ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper", "Rs_int", "hotel_suite_large",
          "office_cubicles_right", "restaurant_diner"]


def survey_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "survey_walls.py --scene|scripts/run_survey_all.sh"], capture_output=True, text=True).stdout.split()
    return any(p.strip() and int(p) != os.getpid() for p in out)


def run(cmd, log):
    print(f"[finalize {time.strftime('%H:%M:%S')}] $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    log.append({"cmd": " ".join(cmd), "returncode": r.returncode, "stdout_tail": r.stdout[-1500:], "stderr_tail": r.stderr[-1500:]})
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait", action="store_true", help="wait until no survey process is running")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    if args.wait:
        while survey_running():
            time.sleep(args.poll)
    steps = []
    ok_agg = run([sys.executable, os.path.join(HERE, "aggregate_walls.py")], steps)
    ok_csv = run([BASE_PY if os.path.exists(BASE_PY) else sys.executable, os.path.join(HERE, "update_selection_csv.py")], steps)

    # ---- summary
    summary = {"timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "status": "ok" if ok_agg and ok_csv else "error",
               "runlog": os.path.relpath(ROOT, REPO), "steps": steps}
    try:
        sel = json.load(open(os.path.join(ROOT, "selection.json")))
        summary["objects"] = {t: {"problem": d["problem"], "rounds": {r: [c["category"] for c in cats] for r, cats in d["rounds"].items()},
                                  "n_valid_models": sum(c["n_valid"] for cats in d["rounds"].values() for c in cats),
                                  "picks": {c["category"]: c["picks"] for cats in d["rounds"].values() for c in cats}}
                              for t, d in sel["tasks"].items()}
    except Exception as e:  # noqa: BLE001
        summary["objects"] = {"error": repr(e)}
    try:
        walls = json.load(open(os.path.join(ROOT, "wall_candidates.json")))
        per_scene = {}
        missing = {}
        for scene in SCENES:
            sc = walls["scenes"].get(scene)
            rooms = sc["rooms"] if sc else {}
            bad = sorted(r for r, rr in rooms.items() if rr.get("status") != "ok")
            per_scene[scene] = {**walls["summary"].get(scene, {"rooms_surveyed": 0, "rooms_ok": [], "n_regions": 0, "n_validated": 0, "length_validated_m": 0.0}),
                                "rooms_not_ok": bad}
            if not sc:
                missing[scene] = "not surveyed"
            elif bad:
                missing[scene] = {r: rooms[r].get("status") for r in bad}
        summary["walls"] = {"per_scene": per_scene, "scenes_or_rooms_missing": missing,
                            "n_validated_regions": sum(v["n_validated"] for v in per_scene.values()),
                            "validated_length_m": round(sum(v["length_validated_m"] for v in per_scene.values()), 1),
                            "files": ["wall_candidates.json", "wall_candidates.md"] + sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "maps", "*.png")))}
    except Exception as e:  # noqa: BLE001
        summary["walls"] = {"error": repr(e)}
    try:
        rows = list(csv.DictReader(open(SEL_CSV, newline="")))
        blocks = {}
        i = 0
        while i < len(rows):
            if rows[i]["task"].startswith("world_alignment"):
                j = i + 1
                while j < len(rows) and not rows[j]["task"]:
                    j += 1
                blocks[f"{rows[i]['task']} / {rows[i]['role']}"] = {"category": rows[i]["category"], "model": rows[i]["model"], "sub_rows": j - i - 1}
                i = j
            else:
                i += 1
        summary["sheet"] = {"csv": os.path.relpath(SEL_CSV, REPO), "n_rows": len(rows), "world_alignment_blocks": blocks,
                            "twins": ["task_object_selection.xlsx", "task_object_selection.html", "legend_extra.json"]}
    except Exception as e:  # noqa: BLE001
        summary["sheet"] = {"error": repr(e)}
    # sampler smoke tests (if they were run into /tmp/isg_wa_test)
    tests = {}
    for prob in ("world_alignment-1", "world_alignment-0"):
        d = os.path.join("/tmp/isg_wa_test", prob)
        if os.path.isdir(d):
            tests[prob] = {os.path.basename(t): len(glob.glob(os.path.join(t, "instances", "*-tro_state.json"))) for t in sorted(glob.glob(os.path.join(d, "*"))) if os.path.isdir(t)}
    summary["sampler_1_4_1_smoke_tests"] = tests or "not run"
    summary["code_changes"] = ["ispatialgym/sampling/tasks/world_alignment.py (per-category frames, fixed.wall / wall_regions)",
                               "ispatialgym/sampling/sample_instances.py (rooms_for skips wall_regions)",
                               "ispatialgym/sampling/CHANGELOG.md, ispatialgym/sampling/__init__.py (1.4.1)", "ispatialgym/docs/instance_sampling_spec.md (5.8 items 5-6)",
                               "runlogs/table_selection_20260911/scripts/update_selection_csv.py (render extra legend)"]
    with open(os.path.join(ROOT, "DONE.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(ROOT, "DONE"), "w") as f:
        f.write(f"{summary['timestamp']} status={summary['status']} see DONE.json\n")
    print(f"[finalize {time.strftime('%H:%M:%S')}] wrote DONE.json (status {summary['status']})", flush=True)
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
