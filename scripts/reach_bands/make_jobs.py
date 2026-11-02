#!/usr/bin/env python3
"""Build the reach-band job manifest (TODO B3) from the owner's representative picks.

    python scripts/reach_bands/make_jobs.py --slots representative --wave 1 --out runlogs/reach_bands_20260915/jobs.json
    python scripts/reach_bands/make_jobs.py --slots variation_geometry --wave 2 --out runlogs/reach_bands_20260915/jobs_w2.json

Input: ``ispatialgym/assets/icl/roles/representatives.csv`` (one row per pick x role x slot) + the handle annotations of
the grasp library (joint limits of the small-scale articulation targets). One job per distinct cache factor set:

* tabletop picks (disembedding, block design, attachment, axial alignment, the five small-scale IORS relations): one
  job per (scene support of the pick's support slot, target model [+ spawn scale], grasp set) - the grasp set is the
  pipeline of the subtask's iddl ``:require`` (tabletop / insert / attachment, the ``grasp_key`` column); blocks and
  axial targets use ``--generic-pinch``; disembedding adds ``--placed tray/gsxbym`` (the fixed source tray, spec 4.6);
* small-scale articulation (open + close share one job): ``--brace --target C/M --handle-key C__M --joint J
  --joint-values lower,50 %,75 %,95 %,upper`` (the stable angle S of the v1 probe is not known here);
* IORS large-scale: the placement band (ontop: the partner as the support; under: ``--support floor --under PARTNER``)
  and the init band (the pick's support, or ``--support floor --room R`` when the support is the floor).

``cmd`` is the pod-side command of ``scripts/runpod/dispatch_samp.py`` (``$OUT`` = the job's output directory); ``args``
is the same call for a local run. Jobs are deduplicated on the argument list; ``weight`` is the queue priority
(``dispatch_samp.py`` queues heavier jobs first): brace 4, large-scale IORS bands 3, tabletop 2, disembedding 1.

``--group-by-world N`` additionally writes ``<out stem>_groups.json``: worker jobs for the multi-band mode of
``reach_bands`` (``python -m ispatialgym.sampling.reach_bands --jobs <out> --ids ID ... --out $OUT``), at most N manifest
jobs sharing one world key (scene, loaded rooms, removed chairs) each, so that a pod loads the scene once per group
(wave-1 manifest: 95 jobs -> 8 world keys -> 29 worker jobs of <= 4). The member jobs' own entries are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
from collections import defaultdict
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
REPS = os.path.join(REPO, "ispatialgym", "assets", "icl", "roles", "representatives.csv")
HANDLES = os.path.join(REPO, "runlogs", "grasp_library_20260905", "artic", "handles")
STABLE = os.path.join(REPO, "Behavior-1K", "datasets", "2026-challenge-task-instances", "scenes", "{scene}", "json", "{scene}_stable.json")
SOURCE_TRAY = "tray/uekqey"  # 2026-09-24: replaces tray/gsxbym (build_icl.TRAY_OVERRIDES)
JOINT_FRACTIONS = (0.5, 0.75, 0.95)

TABLETOP_SUBTASKS = {
    "disembedding": "tabletop",
    "configuration_matching-block_design": "generic",
    "configuration_matching-attachment": "attachment",
    "alignment-axial": "generic",
    "inter_object_relation_satisfaction-ontop_small_scale": "tabletop",
    "inter_object_relation_satisfaction-under_small_scale": "tabletop",
    "inter_object_relation_satisfaction-inside_small_scale": "insert",
    "inter_object_relation_satisfaction-nextto_small_scale": "tabletop",
    "inter_object_relation_satisfaction-touching_small_scale": "tabletop",
}
ARTIC_SUBTASKS = {"configuration_matching-articulation_open_small_scale", "configuration_matching-articulation_close_small_scale"}
LARGE_SUBTASKS = {"inter_object_relation_satisfaction-ontop_large_scale": "ontop", "inter_object_relation_satisfaction-under_large_scale": "under"}

POD_PREFIX = ("cd /root/ispatialgym && source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym && "
              "OMNIGIBSON_HEADLESS=1 ISPATIALGYM_REACH_BANDS=/root/samp/rb_cache timeout 3000 python -m ispatialgym.sampling.reach_bands ")


def _rooms(scene: str, name: str) -> List[str]:
    with open(STABLE.format(scene=scene)) as f:
        d = json.load(f)
    info = d["objects_info"]["init_info"].get(name)
    if info is None:
        raise SystemExit(f"{name} not in scene {scene}")
    return list(info["args"].get("in_rooms") or [])


def _joint_values(handle_key: str, joint: str) -> Optional[str]:
    path = os.path.join(HANDLES, f"{handle_key}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    j = next((j for j in d["joints"] if j["joint"] == joint), None)
    if j is None:
        return None
    lo, hi = float(j.get("closed_q", j.get("lower", 0.0))), float(j["upper"])
    vals = [lo] + [lo + f * (hi - lo) for f in JOINT_FRACTIONS] + [hi]
    return ",".join(f"{v:.4f}" for v in vals)


def _slug(s: str) -> str:
    return s.replace("/", "-").replace(" ", "_")


class Manifest:
    def __init__(self, slot: str):
        self.slot = slot
        self.jobs: Dict[str, dict] = {}
        self.notes: List[str] = []

    def add(self, args: List[str], subtask: str, pick: str, kind: str, weight: float, scene: str):
        key = json.dumps(args)
        if key in self.jobs:
            j = self.jobs[key]
            j["subtasks"] = sorted(set(j["subtasks"]) | {subtask})
            j["picks"] = sorted(set(j["picks"]) | {pick})
            return
        # readable, deterministic id from the factors
        a = dict(zip(args[::2], args[1::2])) if all(x.startswith("--") for x in args[::2]) else {}
        parts = ["rb", kind, scene]
        if "--support" in args:
            sup = args[args.index("--support") + 1]
            parts.append(sup if sup != "floor" else ("floor-under-" + args[args.index("--under") + 1] if "--under" in args else "floor-" + args[args.index("--room") + 1]))
        if "--object" in args:
            parts.append(_slug(args[args.index("--object") + 1]))
        if "--target" in args:
            parts.append(_slug(args[args.index("--target") + 1]) + "-" + args[args.index("--joint") + 1])
        if "--generic-pinch" in args:
            parts.append("gp")
        if "--placed" in args:
            parts.append("placed-" + _slug(args[args.index("--placed") + 1]))
        if "--scale" in args:
            parts.append("scaled")
        jid = "__".join(parts)
        self.jobs[key] = {"id": jid, "kind": kind, "subtasks": [subtask], "picks": [pick], "slot": self.slot, "scene": scene, "weight": weight,
                          "args": list(args), "cmd": POD_PREFIX + " ".join(shlex.quote(x) for x in args) + " --out $OUT"}


def build(slot: str, families: Optional[List[str]] = None) -> Manifest:
    rows = list(csv.DictReader(open(REPS)))
    picks: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if r["slot"] == slot:
            picks[r["pick"]].append(r)
    # the support / partner rows of a pick only carry the ``representative`` slot: reuse them for the variation slots
    scene_rows: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r["kind"] == "scene" and r["slot"] == "representative":
            scene_rows[r["pick"]][r["role"]] = r
    m = Manifest(slot)
    for pick, rs in sorted(picks.items()):
        subtask = rs[0]["subtask"]
        if families and not any(subtask.startswith(f) for f in families):
            continue
        roles = {r["role"]: r for r in rs}
        for role, r in scene_rows.get(pick, {}).items():
            roles.setdefault(role, r)
        sup = roles.get("support")
        if subtask in TABLETOP_SUBTASKS:
            tgt = roles.get("target") or roles.get("targets")
            if tgt is None or sup is None:
                m.notes.append(f"{pick}: no target / support row, skipped")
                continue
            scene, support = sup["scene"], sup["name"]
            pipeline = TABLETOP_SUBTASKS[subtask]
            args = ["--scene", scene, "--support", support, "--object", f"{tgt['category']}/{tgt['model']}"]
            weight = 2.0  # queue priority (dispatch_samp.py runs heavier jobs first): brace 4 > large-scale IORS 3 > tabletop 2 > disembedding 1
            if pipeline == "generic":
                args.append("--generic-pinch")
            else:
                if not tgt["grasp_key"]:
                    m.notes.append(f"{pick}: no grasp key for {tgt['category']}/{tgt['model']} ({pipeline}); skipped")
                    continue
                args += ["--grasp-key", tgt["grasp_key"]]
            if tgt.get("scale"):
                args += ["--scale", tgt["scale"].replace("[", "").replace("]", "").replace(" ", "")]
            if subtask == "disembedding":
                args += ["--placed", SOURCE_TRAY]
                weight = 1.0
            m.add(args, subtask, pick, "tabletop", weight, scene)
            if subtask == "configuration_matching-attachment" and roles.get("partner"):
                m.notes.append(f"{pick}: attachment partner {roles['partner']['category']}/{roles['partner']['model']} needs the attach-reach band "
                               f"(every attachment pose reachable, spec 4.6 step 2) - not a reach_bands mode; the target's band is computed")
        elif subtask in ARTIC_SUBTASKS:
            tgt = roles.get("target")
            if tgt is None or sup is None:
                m.notes.append(f"{pick}: no target / support row, skipped")
                continue
            hk = f"{tgt['category']}__{tgt['model']}"
            qv = _joint_values(hk, tgt["joint"])
            if qv is None:
                m.notes.append(f"{pick}: no handle annotation {hk}/{tgt['joint']}; skipped")
                continue
            args = ["--scene", sup["scene"], "--support", sup["name"], "--target", f"{tgt['category']}/{tgt['model']}", "--handle-key", hk,
                    "--joint", tgt["joint"], "--joint-values", qv, "--brace"]
            if tgt.get("scale"):
                args += ["--scale", tgt["scale"].replace("[", "").replace("]", "").replace(" ", "")]
            m.add(args, subtask, pick, "brace", 4.0, sup["scene"])
        elif subtask in LARGE_SUBTASKS:
            tgt, partner = roles.get("target"), roles.get("partner")
            if tgt is None or partner is None or sup is None:
                m.notes.append(f"{pick}: incomplete roles, skipped")
                continue
            if not tgt["grasp_key"]:
                m.notes.append(f"{pick}: no grasp key for {tgt['category']}/{tgt['model']}; skipped")
                continue
            scene = partner["scene"]
            obj = ["--object", f"{tgt['category']}/{tgt['model']}", "--grasp-key", tgt["grasp_key"]]
            if tgt.get("scale"):
                obj += ["--scale", tgt["scale"].replace("[", "").replace("]", "").replace(" ", "")]
            rel = LARGE_SUBTASKS[subtask]
            # placement band (stance C)
            if rel == "ontop":
                m.add(["--scene", scene, "--support", partner["name"]] + obj, subtask, pick, "tabletop", 3.0, scene)
            else:
                m.add(["--scene", scene, "--support", "floor", "--under", partner["name"]] + obj, subtask, pick, "floor", 3.0, scene)
            # init band (stance B) at the init surface
            if sup["name"] == "floor":
                room = _rooms(scene, partner["name"])[0]
                m.add(["--scene", scene, "--support", "floor", "--room", room] + obj, subtask, pick, "floor", 3.0, scene)
            else:
                m.add(["--scene", sup["scene"], "--support", sup["name"]] + obj, subtask, pick, "tabletop", 3.0, sup["scene"])
    return m


def group_by_world(jobs: List[dict], manifest_path: str, max_per_group: int, wave: int) -> List[dict]:
    """Worker jobs for ``python -m ispatialgym.sampling.reach_bands --jobs MANIFEST --ids ...``: the manifest's jobs grouped
    by world key (scene, loaded rooms, removed chairs - ``reach_bands.band_world_key``, stable scene json only) in their
    manifest order, at most ``max_per_group`` per worker job (a pod's 50 min timeout ~ 3-4 tabletop bands after one scene
    load). ``id`` = ``rbw__<scene>__<key hash>__<k>``, ``ids`` = the member jobs, ``weight`` = the members' maximum;
    ``args`` / ``cmd`` call the worker with the manifest path as given (repo-relative on the pods). The member jobs keep
    their own entries unchanged."""
    from ispatialgym.sampling.reach_bands import band_world_key, build_parser

    parser = build_parser()
    stables: Dict[str, dict] = {}
    groups: Dict[tuple, List[dict]] = {}
    for j in jobs:
        scene = j["scene"]
        if scene not in stables:
            with open(STABLE.format(scene=scene)) as f:
                stables[scene] = json.load(f)
        try:
            key = band_world_key(parser.parse_args(list(j["args"])), stable=stables[scene])
        except SystemExit as e:
            print(f"note: {j['id']}: cannot compute the world key ({e}); its own worker job", file=sys.stderr)
            key = (scene, ("__single__", j["id"]), ())
        groups.setdefault(key, []).append(j)
    out = []
    for key, members in groups.items():
        tag = hashlib.sha1(json.dumps(key).encode()).hexdigest()[:8]
        for k in range(0, len(members), max_per_group):
            chunk = members[k:k + max_per_group]
            ids = [m["id"] for m in chunk]
            args = ["--jobs", manifest_path, "--ids"] + ids
            out.append({"id": f"rbw__{key[0]}__{tag}__{k // max_per_group}", "kind": "worker", "world_key": [key[0], list(key[1]), list(key[2])], "ids": ids,
                        "subtasks": sorted({s for m in chunk for s in m["subtasks"]}), "picks": sorted({p for m in chunk for p in m["picks"]}), "slot": chunk[0]["slot"],
                        "scene": key[0], "weight": max(m["weight"] for m in chunk), "wave": wave, "args": args,
                        "cmd": POD_PREFIX + " ".join(shlex.quote(x) for x in args) + " --out $OUT"})
    out.sort(key=lambda j: (-j["weight"], j["id"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slots", nargs="+", default=["representative"])
    ap.add_argument("--families", nargs="*", default=None, help="subtask prefixes to include (default all B3 families)")
    ap.add_argument("--wave", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-seconds", type=float, default=None, help="measured seconds per job for the total estimate")
    ap.add_argument("--group-by-world", type=int, default=None, metavar="N",
                    help="also write <out stem>_groups.json: worker jobs (reach_bands --jobs <out> --ids ...) of at most N manifest jobs sharing one world key")
    args = ap.parse_args()
    jobs, notes = [], []
    seen = set()
    for slot in args.slots:
        m = build(slot, args.families)
        for j in m.jobs.values():
            if json.dumps(j["args"]) in seen:
                continue
            seen.add(json.dumps(j["args"]))
            j["wave"] = args.wave
            jobs.append(j)
        notes += m.notes
    jobs.sort(key=lambda j: (-j["weight"], j["id"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(jobs, f, indent=1)
    kinds = defaultdict(int)
    for j in jobs:
        kinds[j["kind"]] += 1
    print(f"{len(jobs)} jobs -> {args.out}  ({dict(kinds)}; slots {args.slots})")
    if args.smoke_seconds:
        print(f"estimated total: {len(jobs) * args.smoke_seconds / 3600:.1f} pod-hours at {args.smoke_seconds:.0f} s/job")
    for n in notes:
        print("note:", n, file=sys.stderr)
    with open(os.path.splitext(args.out)[0] + "_notes.txt", "w") as f:
        f.write("\n".join(notes) + "\n")
    if args.group_by_world:
        groups = group_by_world(jobs, args.out, args.group_by_world, args.wave)
        gpath = os.path.splitext(args.out)[0] + "_groups.json"
        with open(gpath, "w") as f:
            json.dump(groups, f, indent=1)
        print(f"{len(groups)} worker jobs ({len({tuple(json.dumps(g['world_key'])) for g in groups})} world keys, <= {args.group_by_world} bands each) -> {gpath}")


if __name__ == "__main__":
    main()
