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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readable import readable  # noqa: E402

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


def _sha_readable(p):
    return sha256_file(readable(p))


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


def rows_eval_production(a):
    """Rows of the consolidated eval_production tree (<root>/<category>/<class>/<group>/<instance>/, index.json): standard-eval ->
    purpose eval (no perturbation), standard-eval-sibling and train -> purpose train (replay-time lighting + scene perturbation; camera
    relocalization: lighting only). Inputs are the tree's readable copies (tro.json, oracle.json, _templates/...) packed at the
    original paths the demo names; the tree's group_id keys the replay-time draws."""
    root = Path(a.root)
    rows_idx = json.load(open(root / "index.json"))
    cats = a.categories.split(",")
    classes = set(a.classes.split(",")) if a.classes else None
    rows = []
    for r in rows_idx:
        if r.get("kind") != "demo" or r.get("category") not in cats:
            continue
        if classes and r["class_id"] not in classes:
            continue
        inst = root / os.path.dirname(r["dest"])
        tdir = os.path.normpath(inst / (inst / "TEMPLATE").read_text().strip())
        rows.append({"demo_dir": str(inst), "purpose": "eval" if r["category"] == "standard-eval" else "train",
                     "category": r["category"], "class_id": r["class_id"], "task_family": family_of(r["class_id"]),
                     "goal_group_id": r["group_id"], "goal_key": r.get("goal_key"), "instance_id": r.get("instance_id"),
                     "eval_unit_id": r.get("eval_unit_id"), "production_group_id": r.get("production_group_id"),
                     "sibling_of_unit": r.get("sibling_of_unit"), "relation": r.get("relation"), "origin": r.get("origin"),
                     "demo_sha256": r.get("sha256"), "eval_production": str(root),
                     "tree_tro": str(inst / "tro.json"), "tree_oracle": str(inst / "oracle.json"), "tree_template_dir": tdir,
                     "tro_override": str(inst / "tro.json"), "full_scene_override": os.path.join(tdir, "template.json")})
    if a.limit_per_category:
        per = collections.Counter()
        keep = []
        for r in rows:
            if per[r["category"]] < a.limit_per_category:
                keep.append(r)
                per[r["category"]] += 1
        rows = keep
    with open(a.out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows -> {a.out}: {dict(collections.Counter(r['category'] for r in rows))}, "
          f"{len({r['goal_group_id'] for r in rows})} goal groups")


def rows_receipts(a):
    """Training rows from admission receipts (TG/admission_r1/receipts/<class>/*.json, TG/camera_eval100/admission/receipts/...):
    decision admitted, members allocated to train and eligible; --exclude lists member ids to leave out (e.g. the wrong-PhysX ones)."""
    import glob

    excl = set(x.strip() for x in open(a.exclude)) if a.exclude else set()
    classes = set(a.classes.split(",")) if a.classes else None
    rows, seen, skipped = [], set(), collections.Counter()
    for root in a.receipts.split(","):
        for f in sorted(glob.glob(os.path.join(root, "*", "*.json"))):
            d = json.load(open(readable(f)))
            if d.get("decision") != "admitted":
                skipped["not admitted"] += 1
                continue
            if classes and d.get("class_id") not in classes:
                continue
            elig = set(d.get("eligible_members") or [])
            for m in d.get("members") or []:
                mid = m.get("member_id") or os.path.basename(os.path.dirname(m["artifacts"]["demo"]["path"]))
                if m.get("allocation") != "train" or (elig and mid not in elig) or mid in excl:
                    skipped["member not train/eligible/excluded"] += 1
                    continue
                demo_dir = os.path.dirname(m["artifacts"]["demo"]["path"])
                if demo_dir in seen:
                    skipped["duplicate"] += 1
                    continue
                seen.add(demo_dir)
                rel = d.get("admitting_release") or {}
                rows.append({"demo_dir": demo_dir, "purpose": "train", "class_id": d["class_id"], "task_family": family_of(d["class_id"]),
                             "member_id": mid, "receipt": f, "receipt_sha256": d.get("content_sha256"), "release": rel.get("name"),
                             "release_commit": rel.get("candidate_commit"), "goal_group_id": d.get("group_id"),
                             "gate_bypass": d.get("gate_bypass"), "demo_sha256": m["artifacts"]["demo"].get("sha256"),
                             "tro": (m["artifacts"].get("tro") or {}).get("path")})
    with open(a.out, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    print(f"{len(rows)} train rows -> {a.out} ({dict(collections.Counter(r['task_family'] for r in rows))}); skipped {dict(skipped)}")


# ------------------------------------------------------------------------------------------------------------------- units
def demo_facts(demo_dir: str, row: dict = None) -> dict:
    """What grouping and packing need from one demo (read-only). ``pack``: (content file, archive path) pairs; a row of the
    eval_production tree packs the tree's copies at the original paths."""
    row = row or {}
    if row.get("tree_tro"):
        return tree_facts(demo_dir, row)
    d = Path(demo_dir)
    meta = json.load(open(readable(str(d / "meta.json"))))
    with h5py.File(readable(str(d / "demo.hdf5")), "r") as f:
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
    tro_json = json.load(open(readable(tro)))
    camp = (tro_json.get("ispatialgym") or {}).get("campaign") or {}
    group = camp.get("goal_group_id") or (meta.get("group_assignments") or {}).get("group_id") or d.name
    files = [str(d / n) for n in sorted(os.listdir(d)) if n == "demo.hdf5" or n.endswith(".json")]
    stem = tro[: -len("-tro_state.json")] if tro.endswith("-tro_state.json") else None
    files += [tro] + [p for p in ((stem + ".oracle.json", stem + ".summary.json") if stem else ()) if os.path.exists(p)]
    files += [os.path.join(tdir, n) for n in ("template.json", "template_meta.json", "template-partial_rooms.json")
              if os.path.exists(os.path.join(tdir, n))]
    files.append(template)
    files = sorted(set(os.path.realpath(p) if os.path.islink(p) else p for p in files) | set(files))
    return {"dir": str(d), "name": d.name, "T": T, "action_steps": a, "scene": cfg["scene"]["scene_model"],
            "crop": (crop or {}).get("mode"), "template_key": full_sha or _sha_readable(template),
            "rooms": sorted(rooms) if rooms else None, "group": str(group), "tro": tro, "template_dir": tdir,
            "pack": [(p if os.path.islink(p) else readable(p), p) for p in files]}


def tree_facts(demo_dir: str, row: dict) -> dict:
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
    tro = (cfg.get("task") or {}).get("tro_path") or meta.get("tro_path")
    a = meta.get("n_action_steps")
    if not isinstance(a, int):
        raise ValueError(f"{demo_dir}: meta.json has no n_action_steps")
    T_dir = Path(row["tree_template_dir"])
    tdir = os.path.dirname(os.path.dirname(tro))
    stem = tro[: -len("-tro_state.json")]
    # the tree's own copies at the tree's paths (the replay reads them through --instance-meta tro_override / full_scene_override);
    # the template once more at the path the hdf5 names, for the offline in-view planner (any version of the template: same objects)
    pack = [(str(d / n), str(d / n)) for n in ("demo.hdf5", "meta.json", "oracle.json", "tro.json")]
    pack += [(str(T_dir / n), str(T_dir / n)) for n in ("template.json", "template_meta.json", "template-partial_rooms.json")
             if (T_dir / n).exists()]
    if full_file:
        pack.append((str(T_dir / "template.json"), full_file))
    names = sorted(json.load(open(T_dir / "template.json"))["objects_info"]["init_info"])
    return {"dir": str(d), "name": d.name, "T": T, "action_steps": a, "scene": cfg["scene"]["scene_model"],
            "crop": (crop or {}).get("mode"), "template_key": full_sha or sha256_file(T_dir / "template.json"),
            # the world one replay process builds: the template's object set (versions of a template differ only in object states;
            # each demo restores its own version)
            "world_key": hashlib.sha256(json.dumps(names).encode()).hexdigest(),
            "rooms": sorted(rooms) if rooms else None, "group": row["goal_group_id"], "tro": row["tree_tro"], "template_dir": str(T_dir),
            "pack": pack}


def replay_frames(action_steps: int, frac: float, min_steps: int) -> int:
    return action_steps + max(int(math.ceil(frac * action_steps)), min_steps)


def build(a):
    rows = [json.loads(x) for x in open(a.rows) if x.strip()]
    run = RV / "runs" / a.run
    if run.exists() and any(run.iterdir()) and not a.resume:
        sys.exit(f"{run} exists")
    (run / "inputs").mkdir(parents=True, exist_ok=True)
    (run / "meta").mkdir(exist_ok=True)
    facts = {}
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(16) as ex:
        for r, f in zip(rows, ex.map(lambda r: demo_facts(r["demo_dir"], r), rows)):
            facts[r["demo_dir"]] = f
    groups = collections.defaultdict(list)
    for r in rows:
        f = facts[r["demo_dir"]]
        key = (r["purpose"], f.get("world_key") or f["template_key"], json.dumps(f["rooms"]), json.dumps(r.get("extra_args") or []),
               r.get("class_id"))
        groups[key].append(r)
    units = []
    for key, rs in sorted(groups.items()):
        # goal groups whole, in order; cut at --frames / --max-demos between goal groups
        by_group = collections.OrderedDict()
        for r in sorted(rs, key=lambda r: (facts[r["demo_dir"]]["group"], facts[r["demo_dir"]]["name"])):
            # --split-groups: per-demo draws (--pert-unit demo) need no goal group kept whole; every demo may go its own way
            gk = facts[r["demo_dir"]]["name"] if a.split_groups else facts[r["demo_dir"]]["group"]
            by_group.setdefault(gk, []).append(r)
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
    # inputs + meta per unit (in parallel; --resume keeps the units whose meta file exists: it is written after the inputs)
    def pack_unit(u):
        if a.resume and (run / u["meta"]).exists() and (run / u["inputs"]).exists():
            u["inputs_bytes"] = (run / u["inputs"]).stat().st_size
            u["n_files"] = None
            del u["_rows"]
            return
        by_arc = {}
        for r in u["_rows"]:
            for src, arc in facts[r["demo_dir"]]["pack"]:
                if arc in by_arc and by_arc[arc] != src:
                    a_sha, b_sha = sha256_file(readable(by_arc[arc])), sha256_file(readable(src))
                    if a_sha != b_sha:
                        u.setdefault("pack_conflicts", []).append({"path": arc, "kept": by_arc[arc], "other": src})
                        print(f"WARNING {u['unit']}: {arc}: {by_arc[arc]} and {src} differ; keeping the first", flush=True)
                    continue
                by_arc[arc] = src
        pack = sorted(((src, arc) for arc, src in by_arc.items()), key=lambda x: x[1])
        files = [arc for _, arc in pack]
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for src, arc in pack:
                tf.add(src, arcname=arc.lstrip("/"), recursive=False)
        zst = subprocess.run(["zstd", "-q", "-T4", "-10", "-c"], input=buf.getvalue(), capture_output=True, check=True).stdout
        (run / u["inputs"]).write_bytes(zst)
        with open(run / u["meta"], "w") as f:
            for r in u["_rows"]:
                f.write(json.dumps({**{k: v for k, v in r.items() if k != "extra_args"}, "unit": u["unit"], "purpose": u["purpose"],
                                    "demo_name": facts[r["demo_dir"]]["name"]}) + "\n")
        u["inputs_bytes"] = len(zst)
        u["n_files"] = len(files)
        del u["_rows"]

    with ThreadPoolExecutor(a.pack_workers) as ex:
        for i, _ in enumerate(ex.map(pack_unit, units)):
            if i % 100 == 99:
                print(f"packed {i + 1}/{len(units)}", flush=True)
    if a.demo_repeat > 1 or a.repeat > 1:
        # throughput sweeps: identical workloads (each unit's demos played demo_repeat times, each unit copied repeat times)
        rep_units = []
        for u in units:
            u = dict(u, demos=u["demos"] * a.demo_repeat, frames=u["frames"] * a.demo_repeat)
            for k in range(a.repeat):
                rep_units.append(dict(u, unit=f"{u['unit']}_r{k}"))
        units = rep_units
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
    p = sp.add_parser("rows-eval-production")
    p.add_argument("--root", default="/vision/u/minyeong/ispatialgym/eval_production")
    p.add_argument("--out", required=True)
    p.add_argument("--categories", default="standard-eval,standard-eval-sibling,train")
    p.add_argument("--classes", default="")
    p.add_argument("--limit-per-category", type=int, default=0)
    p = sp.add_parser("rows-receipts")
    p.add_argument("--receipts", required=True, help="receipt roots (comma-separated), each <root>/<class>/*.json")
    p.add_argument("--out", required=True)
    p.add_argument("--classes", default="")
    p.add_argument("--exclude", default="", help="file of member ids to leave out")
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
    p.add_argument("--repeat", type=int, default=1, help="sweep runs: copies of every unit")
    p.add_argument("--split-groups", action="store_true", help="cut units between demos of one goal group (per-demo draws)")
    p.add_argument("--resume", action="store_true", help="keep the units already packed in an existing run dir (same rows)")
    p.add_argument("--pack-workers", type=int, default=24)
    p.add_argument("--demo-repeat", type=int, default=1, help="sweep runs: every unit plays its demos this many times")
    a = ap.parse_args()
    {"rows-eval1000": rows_eval1000, "rows-eval-production": rows_eval_production, "rows-receipts": rows_receipts, "build": build}[a.cmd](a)


if __name__ == "__main__":
    main()
