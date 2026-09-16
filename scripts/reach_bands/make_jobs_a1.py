#!/usr/bin/env python3
"""Job manifests for the owner decisions 2026-09-16 A1 / A2 / A6 (reach-band side).

    python scripts/reach_bands/make_jobs_a1.py contact --supports house_single_floor/breakfast_table_rhjoby_0 --out <jobs.json>
    python scripts/reach_bands/make_jobs_a1.py generic --out <jobs.json>              # re-run of the generic pinch / push bands (pitched classes)
    python scripts/reach_bands/make_jobs_a1.py support-objects --supports <scene/support> ... --out <jobs.json>   # A3 supports: right-hand bands per object

``contact``: one left-arm generic CONTACT band per (support, object height class) for the brace-task objects (axial
alignment, block design, attachment targets, touching targets of the ``representative`` slot; ``--slots`` widens it);
heights come from the committed right-hand bands' ``object_extent`` (fallback: the grasp library results' bbox).
``generic``: every generic-pinch job of the manifests (axial / block objects) - the pitched approach classes change the
cache key, so the old bands are replaced. ``support-objects``: the right-hand grasp bands of every manifest object on
the given supports (the picks whose support slot names them, or - with ``--all-objects`` - every tabletop pick object),
plus the left contact bands of their height classes.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shlex
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from ispatialgym.sampling.reach_bands import height_class  # noqa: E402
import make_jobs as MJ  # noqa: E402

BANDS = os.path.join(REPO, "runlogs", "reach_bands_20260915", "bands")
BRACE_SUBTASKS = ("alignment-axial", "configuration_matching-block_design", "configuration_matching-attachment", "inter_object_relation_satisfaction-touching")
TIMEOUT = "timeout 5400 "


def object_heights():
    """category/model -> height (m) from the committed band metas (object_extent z) or the grasp library bbox."""
    out = {}
    for f in glob.glob(os.path.join(BANDS, "*.json")):
        m = json.load(open(f))
        if m.get("object") and m.get("object_extent"):
            out.setdefault(m["object"], float(m["object_extent"][2]))
    lib = os.path.join(REPO, "runlogs", "grasp_library_20260905", "results")
    for f in glob.glob(os.path.join(lib, "*.json")):
        try:
            r = json.load(open(f))
        except ValueError:
            continue
        obj = f"{r.get('category')}/{r.get('model')}"
        bb = r.get("bbox_scaled") or r.get("bbox")
        if obj not in out and bb and isinstance(bb, (list, tuple)) and len(bb) == 3:
            out[obj] = float(bb[2])
    return out


def _argmap(args):
    """{flag: value or True} for a CLI argument list with flags that may have no value."""
    out, i = {}, 0
    while i < len(args):
        if args[i].startswith("--"):
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                out[args[i]] = args[i + 1]
                i += 2
            else:
                out[args[i]] = True
                i += 1
        else:
            i += 1
    return out


def job(args, kind, subtasks, picks, scene, weight=2.0):
    a = _argmap(args)
    parts = ["rb", kind, scene, a.get("--support", "floor")]
    if "--object" in a:
        parts.append(MJ._slug(a["--object"]))
    if "--height" in a:
        parts.append(f"h{float(a['--height']):.2f}")
    if "--generic-pinch" in a:
        parts.append("gp")
    if "--placed" in a:
        parts.append("placed-" + MJ._slug(a["--placed"]))
    if "--scale" in a:
        parts.append("scaled")
    if "--arm" in a and a["--arm"] != "right":
        parts.append(a["--arm"])
    return {"id": "__".join(parts), "kind": kind, "subtasks": sorted(set(subtasks)), "picks": sorted(set(picks)), "slot": "a1", "scene": scene, "weight": weight,
            "args": list(args), "cmd": MJ.POD_PREFIX.replace("timeout 3000 ", TIMEOUT) + " ".join(shlex.quote(x) for x in args) + " --out $OUT"}


def brace_objects(slots):
    rows = list(csv.DictReader(open(MJ.REPS)))
    picks = defaultdict(dict)
    for r in rows:
        if r["subtask"] in BRACE_SUBTASKS and r["slot"] in slots and r["role"] in ("target", "targets"):
            picks[(r["pick"], r["slot"])] = {"object": f"{r['category']}/{r['model']}", "subtask": r["subtask"], "scale": r.get("scale") or None}
    return picks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["contact", "generic", "support-objects"])
    ap.add_argument("--supports", nargs="*", default=["house_single_floor/breakfast_table_rhjoby_0"], help="scene/support names")
    ap.add_argument("--slots", nargs="*", default=["representative"])
    ap.add_argument("--all-objects", action="store_true", help="support-objects: every tabletop pick object of the slots, not only the picks bound to the support")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    heights = object_heights()
    jobs = {}
    if args.mode == "contact":
        per_class = defaultdict(list)
        for (pick, slot), o in brace_objects(args.slots).items():
            h = heights.get(o["object"])
            if h is None:
                print("no height for", o["object"], file=sys.stderr)
                continue
            per_class[height_class(h)].append((pick, o["object"], o["subtask"], round(h, 3)))
        for ss in args.supports:
            scene, support = ss.split("/", 1)
            for hc, members in sorted(per_class.items()):
                a = ["--scene", scene, "--support", support, "--arm", "left", "--generic-contact", "--height", f"{hc:.2f}"]
                j = job(a, "contact", [m[2] for m in members], [m[0] for m in members], scene, weight=3.0)
                j["members"] = [{"pick": m[0], "object": m[1], "height": m[3]} for m in members]
                jobs[j["id"]] = j
        print("height classes:", {f"{k:.2f}": [m[1] for m in v] for k, v in sorted(per_class.items())}, file=sys.stderr)
    elif args.mode == "generic":
        for m in ("jobs.json", "jobs_w2.json", "jobs_w3_heldout.json"):
            for j in json.load(open(os.path.join(REPO, "runlogs", "reach_bands_20260915", m))):
                if "--generic-pinch" in j["args"] and any(f"--support {s.split('/', 1)[1]}" in " ".join(j["args"]) for s in args.supports):
                    jj = dict(j, cmd=j["cmd"].replace("timeout 3000 ", TIMEOUT), weight=3.0)
                    jobs[jj["id"]] = jj
    else:
        rows = list(csv.DictReader(open(MJ.REPS)))
        by_pick = defaultdict(dict)
        for r in rows:
            if r["slot"] in args.slots or (r["kind"] == "scene" and r["slot"] == "representative"):
                by_pick[r["pick"]][r["role"] + ("" if r["slot"] in args.slots else ":rep")] = r
        for ss in args.supports:
            scene, support = ss.split("/", 1)
            classes = set()
            for pick, roles in by_pick.items():
                sup = roles.get("support") or roles.get("support:rep")
                tgt = roles.get("target") or roles.get("targets")
                if tgt is None or sup is None:
                    continue
                subtask = tgt["subtask"]
                if subtask not in MJ.TABLETOP_SUBTASKS:
                    continue
                if not args.all_objects and not (sup["scene"] == scene and sup["name"] == support):
                    continue
                pipeline = MJ.TABLETOP_SUBTASKS[subtask]
                obj = f"{tgt['category']}/{tgt['model']}"
                a = ["--scene", scene, "--support", support, "--object", obj]
                if pipeline == "generic":
                    a.append("--generic-pinch")
                elif tgt["grasp_key"]:
                    a += ["--grasp-key", tgt["grasp_key"]]
                else:
                    continue
                if tgt.get("scale"):
                    a += ["--scale", tgt["scale"].replace("[", "").replace("]", "").replace(" ", "")]
                if subtask == "disembedding":
                    a += ["--placed", MJ.SOURCE_TRAY]
                j = job(a, "tabletop", [subtask], [pick], scene, weight=2.0)
                jobs.setdefault(j["id"], j)
                if subtask in BRACE_SUBTASKS and obj in heights:
                    classes.add(height_class(heights[obj]))
            for hc in sorted(classes):
                a = ["--scene", scene, "--support", support, "--arm", "left", "--generic-contact", "--height", f"{hc:.2f}"]
                j = job(a, "contact", ["brace"], [], scene, weight=3.0)
                jobs.setdefault(j["id"], j)
    out = sorted(jobs.values(), key=lambda j: (-j["weight"], j["id"]))
    json.dump(out, open(args.out, "w"), indent=1)
    kinds = defaultdict(int)
    for j in out:
        kinds[j["kind"]] += 1
    print(f"{len(out)} jobs -> {args.out} {dict(kinds)}")


if __name__ == "__main__":
    main()
