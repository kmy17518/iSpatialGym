#!/usr/bin/env python3
"""Batched replay (og-isgrend-20261115) run builder: rows (demos + purpose + linkage fields) -> units -> R2.

    # rows from the eval-1000 manifest (one unit per class/family as asked, or all of it)
    make_units.py rows-eval1000 --manifest <eval1000_manifest dir> --out rows.jsonl [--per-family 1 | --per-class N] [--classes a,b]
    # rows of training demos: one json per line with demo_dir (+ class_id, receipt, ...), purpose train
    make_units.py build --rows rows.jsonl --run <name> [--frames 6000] [--max-demos 40] [--upload]

A row: {"demo_dir", "purpose": "train"|"eval", "class_id", ... any linkage fields (unit_id, eval_instance_id, receipt, release),
optional "extra_args": [replay flags of this row's unit, e.g. validation's forced perturbation]}. Rows are grouped into units that one
replay process can play in one world: same purpose, same full template scene file (isg_full_scene_sha256 / template.json sha256),
same loaded rooms, same extra args; goal groups are never split (the replay draws lighting / removals per goal group and the skybox per
process). Units are cut at --frames replayed frames (settle-padded estimate) or --max-demos.

Writes $RV/runs/<run>/{rows.jsonl, units.json, inputs/<unit>.tar.zst, meta/<unit>.jsonl} and with --upload copies them to
$JOB_S3/runs/<run>/ (only there). inputs/<unit>.tar.zst holds, at their absolute cluster paths, every file the replay of the unit
reads: the demo dirs (demo.hdf5 + json files), the TROs with their oracle / summary sidecars, the template dirs' template.json /
template_meta.json / template-partial_rooms.json, the full scene file the hdf5 names. The worker unpacks it at /.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import h5py  # noqa: E402

RV = Path(os.environ.get("RV", "/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/batched_replay_r1"))
FAMILIES = ("alignment", "camera_relocalization", "configuration_matching", "disembedding", "inter_object_relation_satisfaction",
            "mental_rotation", "object_scaling")


def family_of(class_id: str) -> str:
    return next(f for f in FAMILIES if class_id.startswith(f))


def sha256_file(p, bs=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------- rows
def rows_eval1000(a):
    m = json.load(open(Path(a.manifest) / "manifest.json"))
    units = sorted(m["units"], key=lambda u: (u["class_id"], u["source"] != "online", u["unit_id"]))
    classes = set(a.classes.split(",")) if a.classes else None
    per_class, per_fam = collections.Counter(), collections.Counter()
    rows = []
    for u in units:
        c = u["class_id"]
        if classes and c not in classes:
            continue
        if a.per_class and per_class[c] >= a.per_class:
            continue
        if a.per_family and per_fam[family_of(c)] >= a.per_family:
            continue
        unit_dir = Path(a.manifest) / "eval1000" / c / u["unit_id"]
        demo_dir = os.path.dirname(os.path.realpath(unit_dir / "demo.hdf5"))
        eid = (unit_dir / "EVAL_INSTANCE_ID").read_text().strip() if (unit_dir / "EVAL_INSTANCE_ID").exists() else None
        rows.append({"demo_dir": demo_dir, "purpose": a.purpose, "class_id": c, "task_family": family_of(c),
                     "eval_manifest": str(Path(a.manifest).resolve()), "eval_unit_id": u["unit_id"], "eval_unit_dir": str(unit_dir),
                     "eval_instance_id": eid, "eval_source": u["source"], "primary_eval_instance_id": u.get("primary_eval_instance_id"),
                     "goal_group_id": u.get("group_id")})
        per_class[c] += 1
        per_fam[family_of(c)] += 1
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows -> {a.out} ({dict(per_fam)})")


# ------------------------------------------------------------------------------------------------------------------- units
def demo_facts(demo_dir: str) -> dict:
    """What grouping and packing need from one demo (read-only)."""
    d = Path(demo_dir)
    meta = json.load(open(d / "meta.json"))
    with h5py.File(d / "demo.hdf5", "r") as f:
        at = f["data"].attrs
        cfg = json.loads(at["config"])
        T = int(f["data/demo_0/action"].shape[0])
        crop = json.loads(at["isg_scene_crop"]) if "isg_scene_crop" in at else None
        full_file = json.loads(at["isg_full_scene_file"]) if "isg_full_scene_file" in at else None
        full_sha = json.loads(at["isg_full_scene_sha256"]) if "isg_full_scene_sha256" in at else None
        rooms = json.loads(at["isg_full_load_room_instances"]) if "isg_full_load_room_instances" in at else None
    tro = (cfg.get("task") or {}).get("tro_path")
    if not tro or not os.path.exists(tro):
        tro = meta.get("tro_path") if meta.get("tro_path") and os.path.exists(meta.get("tro_path")) else tro
    if not tro or not os.path.exists(tro):
        raise ValueError(f"{demo_dir}: TRO {tro} not found")
    tdir = os.path.dirname(os.path.dirname(tro))
    template = full_file if full_file and os.path.exists(full_file) else os.path.join(tdir, "template.json")
    if not os.path.exists(template):
        raise ValueError(f"{demo_dir}: no full template scene file ({full_file}, {template})")
    a = meta.get("n_action_steps")
    if not isinstance(a, int):
        raise ValueError(f"{demo_dir}: meta.json has no n_action_steps")
    tro_json = json.load(open(tro))
    camp = (tro_json.get("ispatialgym") or {}).get("campaign") or {}
    group = camp.get("goal_group_id") or (meta.get("group_assignments") or {}).get("group_id") or d.name
    files = [str(d / n) for n in sorted(os.listdir(d)) if n == "demo.hdf5" or n.endswith(".json")]
    stem = tro[: -len("-tro_state.json")] if tro.endswith("-tro_state.json") else None
    files += [tro] + [p for p in ((stem + ".oracle.json", stem + ".summary.json") if stem else ()) if os.path.exists(p)]
    files += [os.path.join(tdir, n) for n in ("template.json", "template_meta.json", "template-partial_rooms.json")
              if os.path.exists(os.path.join(tdir, n))]
    files.append(template)
    return {"dir": str(d), "name": d.name, "T": T, "action_steps": a, "scene": cfg["scene"]["scene_model"],
            "crop": (crop or {}).get("mode"), "template_key": full_sha or sha256_file(template),
            "rooms": sorted(rooms) if rooms else None, "group": str(group), "tro": tro, "template_dir": tdir,
            "files": sorted(set(os.path.realpath(p) if os.path.islink(p) else p for p in files) | set(files))}


def replay_frames(action_steps: int, frac: float, min_steps: int) -> int:
    return action_steps + max(int(math.ceil(frac * action_steps)), min_steps)


def build(a):
    rows = [json.loads(x) for x in open(a.rows) if x.strip()]
    run = RV / "runs" / a.run
    if run.exists() and any(run.iterdir()):
        sys.exit(f"{run} exists")
    (run / "inputs").mkdir(parents=True)
    (run / "meta").mkdir()
    facts = {}
    for r in rows:
        facts[r["demo_dir"]] = demo_facts(r["demo_dir"])
    groups = collections.defaultdict(list)
    for r in rows:
        f = facts[r["demo_dir"]]
        key = (r["purpose"], f["template_key"], json.dumps(f["rooms"]), json.dumps(r.get("extra_args") or []), r.get("class_id"))
        groups[key].append(r)
    units = []
    for key, rs in sorted(groups.items()):
        # goal groups whole, in order; cut at --frames / --max-demos between goal groups
        by_group = collections.OrderedDict()
        for r in sorted(rs, key=lambda r: (facts[r["demo_dir"]]["group"], facts[r["demo_dir"]]["name"])):
            by_group.setdefault(facts[r["demo_dir"]]["group"], []).append(r)
        cur, frames = [], 0
        chunks = []
        for g, members in by_group.items():
            fr = sum(replay_frames(facts[m["demo_dir"]]["action_steps"], a.settle_pad, a.settle_pad_min) for m in members)
            if cur and (frames + fr > a.frames or len(cur) + len(members) > a.max_demos):
                chunks.append(cur)
                cur, frames = [], 0
            cur += members
            frames += fr
        if cur:
            chunks.append(cur)
        for ch in chunks:
            purpose, tkey = key[0], key[1]
            first = facts[ch[0]["demo_dir"]]
            uid = f"u{len(units):04d}_{purpose}_{(ch[0].get('class_id') or first['scene'])}"[:80].replace("/", "_")
            units.append({"unit": uid, "purpose": purpose, "task": ch[0].get("class_id") or first["scene"], "scene": first["scene"],
                          "template_key": tkey, "rooms": first["rooms"], "crop": sorted({facts[r["demo_dir"]]["crop"] or "none" for r in ch}),
                          "frames": sum(replay_frames(facts[r["demo_dir"]]["action_steps"], a.settle_pad, a.settle_pad_min) for r in ch),
                          "demos": [{"name": facts[r["demo_dir"]]["name"], "dir": r["demo_dir"], "frames": facts[r["demo_dir"]]["T"],
                                     "group": facts[r["demo_dir"]]["group"]} for r in ch],
                          "goal_groups": sorted({facts[r["demo_dir"]]["group"] for r in ch}),
                          "extra_args": ch[0].get("extra_args") or [],
                          "inputs": f"inputs/{uid}.tar.zst", "meta": f"meta/{uid}.jsonl", "_rows": ch})
    # inputs + meta per unit
    for u in units:
        files = sorted({p for r in u["_rows"] for p in facts[r["demo_dir"]]["files"]})
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for p in files:
                tf.add(p, arcname=p.lstrip("/"), recursive=False)
        zst = subprocess.run(["zstd", "-q", "-T4", "-10", "-c"], input=buf.getvalue(), capture_output=True, check=True).stdout
        (run / u["inputs"]).write_bytes(zst)
        with open(run / u["meta"], "w") as f:
            for r in u["_rows"]:
                f.write(json.dumps({**{k: v for k, v in r.items() if k != "extra_args"}, "unit": u["unit"], "purpose": u["purpose"],
                                    "demo_name": facts[r["demo_dir"]]["name"]}) + "\n")
        u["inputs_bytes"] = len(zst)
        u["n_files"] = len(files)
        del u["_rows"]
    spec = {"run": a.run, "created": time.time(), "code": a.code, "seed": a.seed,
            "replay_flags": {"image_size": a.image_size, "settle_pad": a.settle_pad, "settle_pad_min": a.settle_pad_min},
            "units": units}
    (run / "units.json").write_text(json.dumps(spec, indent=1))
    with open(run / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_frames = sum(u["frames"] for u in units)
    print(f"{len(units)} units, {sum(len(u['demos']) for u in units)} demos, {n_frames} replayed frames (est.) -> {run}")
    for u in units:
        print(f"  {u['unit']}: {u['purpose']} {u['scene']} crop={u['crop']} demos={len(u['demos'])} frames={u['frames']} "
              f"groups={len(u['goal_groups'])} inputs={u['inputs_bytes'] / 2**20:.1f}MB extra={u['extra_args']}")
    if a.upload:
        js3 = os.environ["JOB_S3"]
        assert js3.endswith("/jobs/og-isgrend-20261115"), js3
        s5 = os.environ["S5"].split()
        subprocess.run(s5 + ["cp", f"{run}/", f"{js3}/runs/{a.run}/"], check=True, capture_output=True)
        print(f"uploaded -> {js3}/runs/{a.run}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("rows-eval1000")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--purpose", default="eval", choices=("train", "eval"))
    p.add_argument("--per-family", type=int, default=0)
    p.add_argument("--per-class", type=int, default=0)
    p.add_argument("--classes", default="")
    p = sp.add_parser("build")
    p.add_argument("--rows", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--code", required=True, help="code snapshot name on R2 (bundle/code/<name>.tar.zst)")
    p.add_argument("--seed", type=int, default=20261115)
    p.add_argument("--frames", type=int, default=6000)
    p.add_argument("--max-demos", type=int, default=40)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--settle-pad", type=float, default=0.2)
    p.add_argument("--settle-pad-min", type=int, default=50)
    p.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    {"rows-eval1000": rows_eval1000, "build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
