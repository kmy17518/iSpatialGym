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
    ap.add_argument("--part", type=int, default=0, help="this job's share of the template groups (parallel replay jobs)")
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--group-size", type=int, default=8, help="demos per playback process (one world per template group)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    task_dir = os.path.abspath(args.task_dir)
    task = os.path.basename(task_dir)
    demos, why = qualifying(task_dir, audit_labels(args.audit))
    demos = demos[: args.limit] if args.limit else demos
    groups = {}
    for d in demos:
        tro = json.load(open(os.path.join(d["demo"], "meta.json")))["tro_path"]
        groups.setdefault(os.path.dirname(os.path.dirname(tro)), []).append(d)
    batches = [ds[i: i + args.group_size] for _, ds in sorted(groups.items()) for i in range(0, len(ds), args.group_size)]
    batches = batches[args.part:: args.parts]
    print("REPLAY_PLAN " + json.dumps({"task": task, "replay": len(demos), "retried": sum(d["retried"] for d in demos), "by_reason": why,
                                       "templates": len(groups), "this_part": sum(len(b) for b in batches)}), flush=True)
    if args.dry_run or not batches:
        return 0
    import subprocess

    root = os.path.join(task_dir, "replay", "lerobot")
    os.makedirs(root, exist_ok=True)
    shard = args.shard if args.parts == 1 else f"{args.shard}_p{args.part}"
    manifest = os.path.join(root, shard + ".episodes.json")
    done = json.load(open(manifest)) if os.path.exists(manifest) else []
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_obs_isg.py")
    for batch in batches:
        for d in batch:
            d["scene_states_filled"] = complete_scene_file(d["demo"])
        cmd = [sys.executable, script, "--demo", ",".join(d["demo"] for d in batch), "--root", root, "--repo-id", shard, "--task", task, "--one-world"]
        info = os.path.join(root, shard, "meta", "info.json")
        before = json.load(open(info))["total_episodes"] if os.path.exists(info) else 0
        if os.path.exists(info):
            cmd.append("--resume")
        p = subprocess.run(cmd, capture_output=True, text=True)
        finished = [ln.split(" ", 1)[1].strip() for ln in p.stdout.splitlines() if ln.startswith("REPLAY_OBS_DONE ")]
        after = json.load(open(info))["total_episodes"] if os.path.exists(info) else 0
        # the process's exit code is not the test: Kit's cleanup at exit can fail after the dataset is finalized
        if len(finished) != len(batch) or after != before + len(batch):
            # the shard is finalized only when the whole process ends; a process that did not end leaves the batch out
            print("REPLAY_FAILED " + json.dumps({"rc": p.returncode, "demos": [d["demo"] for d in batch], "finished": finished,
                                                 "stderr_tail": p.stderr[-2000:]}), flush=True)
            continue
        done.extend(batch)
        with open(manifest + ".tmp", "w") as f:
            json.dump(done, f, indent=1)
        os.replace(manifest + ".tmp", manifest)
        print("REPLAYED " + json.dumps({"demos": len(batch), "shard": shard, "total": len(done)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
