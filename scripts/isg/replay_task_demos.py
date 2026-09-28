#!/usr/bin/env python3
"""BEHAVIOR LeRobot episodes for the demos of a task that have none: demos with a mid-episode retry (their collection episode
was dropped) and demos recorded before the BEHAVIOR output. Each demo.hdf5 is played back with replay_obs.py's writer and
settings (replay_obs_isg.py) into a replay shard of the task, ``<task dir>/replay/lerobot/<shard>``; ``<shard>.episodes.json``
lists the demos in episode order.

    python scripts/isg/replay_task_demos.py --task-dir <prod root>/<sweep>/<alias> --shard <name> [--audit <glob>] [--limit N] [--dry-run]

A demo qualifies when its latest results row is a success recorded with change detection on, in a fresh world, a world with
the scene restore, or a no-restore world the scene audit found clean (``--audit`` reports), with a readable demo.hdf5 and no
collection episode, and no replay shard of the task holds it yet. Scene files saved with the recording filter on (before
157ce1ec) lack the objects that were asleep at the start; they get those objects' states from the template's scene file
(template.json, which build_env loads): the playback loads state[0], a full dump, before its first frame, so these states
only place the objects while the scene loads. The original attribute is kept as scene_file_filtered.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def readable(path):
    try:
        with open(path, "rb") as f:
            return f.read(8) == HDF5_SIGNATURE
    except OSError:
        return False


def audit_labels(pattern):
    labels = {}
    for p in sorted(glob.glob(pattern)) if pattern else []:
        for line in open(p):
            if line.strip():
                a = json.loads(line)
                labels[(a["job_dir"], a["instance"])] = "no_hdf5" if a.get("hdf5_error") else ("disturbed" if a.get("disturbed") else "clean")
    return labels


def qualifying(task_dir, audit):
    replayed = set()
    for mp in glob.glob(os.path.join(task_dir, "replay", "lerobot", "*.episodes.json")):
        replayed |= {e["demo"] for e in json.load(open(mp))}
    out, why = [], {}
    for rp in sorted(glob.glob(os.path.join(task_dir, "i*", "results.jsonl"))):
        if ".empty-" in rp:
            continue
        job_dir, rows = os.path.dirname(rp), {}
        for line in open(rp):
            if line.strip():
                r = json.loads(line)
                rows[r["instance"]] = r
        for iid, r in sorted(rows.items()):
            prov = r.get("provenance") or {}
            b = r.get("behavior_lerobot") or {}
            reason = None
            if not r.get("success"):
                reason = "failed"
            elif prov.get("track_state_updates") is False:
                reason = "change_detection_off"
            elif r.get("reused_world") and not prov.get("queue_scene_restore") and audit.get((job_dir, iid)) != "clean":
                reason = "no_restore_not_audited_clean"
            elif not readable(os.path.join(r.get("out") or "", "demo.hdf5")):
                reason = "no_hdf5"
            elif b.get("episode_index") is not None:
                reason = "has_collection_episode"
            elif r.get("out") in replayed:
                reason = "replayed"
            why[reason or "replay"] = why.get(reason or "replay", 0) + 1
            if reason is None:
                out.append({"instance": iid, "demo": r.get("out"), "job_dir": job_dir, "retried": b.get("status") == "rollback"})
    return out, why


def complete_scene_file(demo_dir) -> int:
    """Fill the scene file of demo.hdf5 with the states of the objects it lacks (from template.json); returns how many."""
    import h5py

    meta = json.load(open(os.path.join(demo_dir, "meta.json")))
    with h5py.File(os.path.join(demo_dir, "demo.hdf5"), "r+") as f:
        data = f["data"]
        original = data.attrs.get("scene_file_filtered", data.attrs["scene_file"])
        sf = json.loads(original)
        reg = sf["state"]["registry"]["object_registry"]
        missing = [n for n in sf["objects_info"]["init_info"] if n not in reg]
        if not missing:
            return 0
        template = json.load(open(os.path.join(os.path.dirname(os.path.dirname(meta["tro_path"])), "template.json")))
        treg = template["state"]["registry"]["object_registry"]
        lacking = [n for n in missing if n not in treg]
        if lacking:
            raise RuntimeError(f"template.json lacks {lacking[:5]}")
        for n in missing:
            reg[n] = treg[n]
        data.attrs["scene_file_filtered"] = original
        data.attrs["scene_file"] = json.dumps(sf)
    return len(missing)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--shard", required=True, help="replay shard name under <task dir>/replay/lerobot")
    ap.add_argument("--audit", default=None, help="glob of check_provisional.py reports")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    task_dir = os.path.abspath(args.task_dir)
    task = os.path.basename(task_dir)
    demos, why = qualifying(task_dir, audit_labels(args.audit))
    demos = demos[: args.limit] if args.limit else demos
    print("REPLAY_PLAN " + json.dumps({"task": task, "replay": len(demos), "retried": sum(d["retried"] for d in demos), "by_reason": why}), flush=True)
    if args.dry_run or not demos:
        return 0
    import subprocess

    root = os.path.join(task_dir, "replay", "lerobot")
    os.makedirs(root, exist_ok=True)
    manifest = os.path.join(root, args.shard + ".episodes.json")
    done = json.load(open(manifest)) if os.path.exists(manifest) else []
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_obs_isg.py")
    for d in demos:
        n = complete_scene_file(d["demo"])
        # a process per demo (a world per demo, as replay_obs.py)
        cmd = [sys.executable, script, "--demo", d["demo"], "--root", root, "--repo-id", args.shard, "--task", task]
        if os.path.exists(os.path.join(root, args.shard, "meta", "info.json")):
            cmd.append("--resume")
        rc = subprocess.run(cmd).returncode
        if rc != 0 or not os.path.exists(os.path.join(root, args.shard, "meta", "info.json")):
            print("REPLAY_FAILED " + json.dumps({**d, "rc": rc}), flush=True)
            continue
        done.append({**d, "scene_states_filled": n})
        with open(manifest + ".tmp", "w") as f:
            json.dump(done, f, indent=1)
        os.replace(manifest + ".tmp", manifest)
        print("REPLAYED " + json.dumps(done[-1]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
