#!/usr/bin/env python3
"""A2 coverage check: does a committed right-hand reach band exist for every (support, object, grasp set) the pilot
manifest's aliases need? (owner decision 2026-09-16 A2)

    python scripts/reach_bands/check_coverage.py --manifest runlogs/pilot_20260915/jobs_sample.json \
        [--customs ispatialgym/assets/task_specs/isg_customs.json] [--out runlogs/reach_bands_20260915/coverage.json] [--jobs-out <manifest for the misses>]

For every alias: the representative target model (customs ``roles.<target>.representative`` / whitelist), the support
bindings per scene, the grasp set of the subtask (disembedding: tabletop grasps with the source tray placed; IORS small:
tabletop / insert; attachment: attachment grasps; axial / block design: generic pinch; small articulation: brace band
of (target, joint); IORS large: placement band (partner as support / floor under the partner) + init band). Subtasks
without a tabletop band (camera relocalization, held objects, wall alignment, large articulation) are listed as
``no_band_needed``. Misses are written as ``reach_bands`` jobs (same schema as ``make_jobs.py``) when ``--jobs-out`` is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from ispatialgym.sampling.reach_bands import PLACED_MODE_FOCUS, ReachBands, placed_spec  # noqa: E402

GRASPS_CSV = os.path.join(REPO, "ispatialgym", "assets", "icl", "require", "grasps.csv")
SOURCE_TRAY = "tray/uekqey"  # 2026-09-24: replaces tray/gsxbym x 1.25 (build_icl.TRAY_OVERRIDES: single-slab floor, same inner area)
#: the disembedding source tray spawn scale (``roles.source.scale``); the customs value wins when present, this is the default
#: until D1 writes it (owner decision 2026-09-16 item 4b was gsxbym x 1.25; uekqey spawns at its canonical size)
SOURCE_TRAY_SCALE = 1.0
POD_PREFIX = ("cd /root/ispatialgym && source /root/miniforge3/etc/profile.d/conda.sh && conda activate ispatialgym && "
              "OMNIGIBSON_HEADLESS=1 ISPATIALGYM_REACH_BANDS=/root/samp/rb_cache timeout 5400 python -m ispatialgym.sampling.reach_bands ")

PIPELINE = {"disembedding": "tabletop", "inter_object_relation_satisfaction-ontop_small_scale": "tabletop", "inter_object_relation_satisfaction-under_small_scale": "tabletop",
            "inter_object_relation_satisfaction-nextto": "tabletop", "inter_object_relation_satisfaction-touching": "tabletop",
            "inter_object_relation_satisfaction-inside": "insert", "configuration_matching-attachment": "attachment",
            "inter_object_relation_satisfaction-ontop_large_scale": "tabletop", "inter_object_relation_satisfaction-under_large_scale": "tabletop"}
GENERIC = {"alignment-axial", "configuration_matching-block_design"}
BRACE = {"configuration_matching-articulation_open_small_scale", "configuration_matching-articulation_close_small_scale"}
NO_BAND = ("camera_relocalization", "object_scaling", "mental_rotation", "alignment-world_vertical",
           "configuration_matching-articulation_open_large_scale", "configuration_matching-articulation_close_large_scale")


def grasp_keys():
    out = defaultdict(dict)
    for r in csv.DictReader(open(GRASPS_CSV)):
        pipe = r["require"].split(":")[-1]
        out[(r["category"], r["model"])].setdefault(pipe, r["key"])
    return out


def normalize_scale(scale, obj: str):
    """Customs ``scale`` field -> ``[sx, sy, sz]`` or None: a list, a scalar (uniform), or a per-model dict
    (``{"cat/model": 0.7, "default": 0.7}`` of the D1 fallback aliases) whose entry for ``obj`` (else ``default``, else
    the first value) is taken."""
    if isinstance(scale, dict):
        scale = scale.get(obj, scale.get("default", next(iter(scale.values()), None)))
    if scale is None:
        return None
    if isinstance(scale, (int, float)):
        return [float(scale)] * 3
    vals = [float(v) for v in scale]
    return vals * 3 if len(vals) == 1 else vals


def subtask_of(alias: str, customs: dict) -> str:
    """Subtask id from the customs ``iddl`` path (``alignment/axial.iddl`` -> ``alignment-axial``, ``disembedding/disembedding.iddl``
    -> ``disembedding``)."""
    iddl = customs.get("iddl", "")
    fam, sub = iddl.replace(".iddl", "").split("/") if "/" in iddl else ("", "")
    return fam if fam == sub else f"{fam}-{sub}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--customs", default=os.path.join(REPO, "ispatialgym", "assets", "task_specs", "isg_customs.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "runlogs", "reach_bands_20260915", "coverage.json"))
    ap.add_argument("--jobs-out", default=None)
    ap.add_argument("--tray-mode", default=PLACED_MODE_FOCUS, help="placed mode of the disembedding source tray the sampler asks for (item 9: "
                    "focus_centred_edge_aligned; 'edge' = the legacy fixed-tray bands)")
    ap.add_argument("--tray-scale", type=float, default=None, help=f"source tray spawn scale when the customs give none (default {SOURCE_TRAY_SCALE}, item 4b)")
    args = ap.parse_args()
    customs = json.load(open(args.customs))
    keys = grasp_keys()
    aliases = sorted({j["alias"] for j in json.load(open(args.manifest)) if j.get("alias")})
    rows, jobs, status = [], {}, Counter()
    for alias in aliases:
        c = customs.get(alias)
        if c is None:
            rows.append({"alias": alias, "status": "no_customs_entry"})
            status["no_customs_entry"] += 1
            continue
        sub = subtask_of(alias, c)
        if sub.startswith(NO_BAND):
            rows.append({"alias": alias, "subtask": sub, "status": "no_band_needed"})
            status["no_band_needed"] += 1
            continue
        roles = c["roles"]
        trole = "targets" if "targets" in roles else "target"
        if trole not in roles:
            rows.append({"alias": alias, "subtask": sub, "status": "no_band_needed"})
            status["no_band_needed"] += 1
            continue
        tgt = roles[trole]
        cat, model = tgt["category"], tgt.get("representative")
        scale = normalize_scale(tgt.get("scale"), f"{cat}/{model}")
        scale_arg = ",".join(str(round(float(v), 4)) for v in scale) if scale else None
        for scene, sc in c.get("scenes", {}).items():
            b = sc.get("bindings", {})
            supports = b.get("support") or []
            if isinstance(supports, dict):
                supports = supports.get("names") or []
            partners = b.get("partner") or []
            needs = []  # (label, find kwargs, cli args)
            obj = f"{cat}/{model}"
            if sub in GENERIC:
                for s in supports:
                    needs.append((f"{s}:generic", dict(scene=scene, support=s, object=obj, grasp_key="generic", scale=scale),
                                  ["--scene", scene, "--support", s, "--object", obj, "--generic-pinch"] + (["--scale", scale_arg] if scale_arg else [])))
            elif sub in BRACE:
                joint = tgt.get("joint")
                if isinstance(joint, dict):
                    joint = next(iter(joint.values()), None)
                for s in supports:
                    needs.append((f"{s}:brace:{joint}", dict(scene=scene, support=s, target=obj, joint=joint, mode="brace"), None))
            elif sub in PIPELINE:
                pipe = PIPELINE[sub]
                gk = keys.get((cat, model), {}).get(pipe) or keys.get((cat, model), {}).get("tabletop")
                placed = []
                if sub == "disembedding":
                    src = roles.get("source") or {}
                    tray = f"{src.get('category', 'tray')}/{src.get('representative', SOURCE_TRAY.split('/')[1])}"
                    tray_scale = normalize_scale(src.get("scale"), tray) or [args.tray_scale if args.tray_scale is not None else SOURCE_TRAY_SCALE] * 3
                    placed = [placed_spec(tray, tray_scale, args.tray_mode)]
                base_args = ["--object", obj] + (["--grasp-key", gk] if gk else []) + (["--scale", scale_arg] if scale_arg else []) + (["--placed", placed[0]] if placed else [])
                if sub.endswith("large_scale"):
                    rel = "under" if "under" in sub else "ontop"
                    for p in partners:
                        if rel == "ontop":
                            needs.append((f"{p}:placement", dict(scene=scene, support=p, object=obj, grasp_key=gk, scale=scale), ["--scene", scene, "--support", p] + base_args))
                        else:
                            needs.append((f"floor-under-{p}", dict(scene=scene, support="floor", object=obj, grasp_key=gk, under=p, scale=scale),
                                          ["--scene", scene, "--support", "floor", "--under", p] + base_args))
                    for s in supports:
                        if s == "floor":
                            room = (b.get("floor") or {}).get("room") or (sc.get("rooms") or [None])[0]
                            needs.append((f"floor-{room}:init", dict(scene=scene, support="floor", object=obj, grasp_key=gk, scale=scale),
                                          ["--scene", scene, "--support", "floor", "--room", room] + base_args))
                        else:
                            needs.append((f"{s}:init", dict(scene=scene, support=s, object=obj, grasp_key=gk, scale=scale), ["--scene", scene, "--support", s] + base_args))
                else:
                    for s in supports:
                        needs.append((f"{s}:{pipe}", dict(scene=scene, support=s, object=obj, grasp_key=gk, placed=placed, scale=scale), ["--scene", scene, "--support", s] + base_args))
                if not gk:
                    rows.append({"alias": alias, "subtask": sub, "scene": scene, "status": "no_grasp_key", "object": obj, "pipeline": pipe})
                    status["no_grasp_key"] += 1
            else:
                rows.append({"alias": alias, "subtask": sub, "status": "unknown_subtask"})
                status["unknown_subtask"] += 1
                continue
            for label, kw, cli in needs:
                paths = ReachBands.find(**kw)
                if paths:
                    m = json.load(open(paths[0][:-4] + ".json"))
                    st = "covered" if m.get("n_reachable_total", 0) > 0 else "covered_empty"
                    rows.append({"alias": alias, "subtask": sub, "scene": scene, "need": label, "object": obj, "status": st, "band": os.path.basename(paths[0]),
                                 "cells": m.get("n_reachable_total"), "union_extent": m.get("union_extent"), "classes": bool(m.get("classes"))})
                else:
                    rows.append({"alias": alias, "subtask": sub, "scene": scene, "need": label, "object": obj, "status": "missing", "cli": cli})
                    if cli:
                        jid = "rb__cov__" + "__".join(a.replace("/", "-") for a in cli if not a.startswith("--"))[:150]
                        jobs.setdefault(json.dumps(cli), {"id": jid, "kind": "floor" if "floor" in cli else "tabletop", "subtasks": [sub], "picks": [alias], "slot": "coverage",
                                                          "scene": scene, "weight": 3.0, "args": cli, "cmd": POD_PREFIX + " ".join(shlex.quote(a) for a in cli) + " --out $OUT"})
                status[rows[-1]["status"]] += 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"status": dict(status), "rows": rows}, open(args.out, "w"), indent=1)
    print(f"{len(aliases)} aliases: {dict(status)} -> {args.out}")
    for r in rows:
        if r["status"] in ("missing", "no_grasp_key", "unknown_subtask", "no_customs_entry"):
            print("  ", r["status"], r["alias"], r.get("need", ""), r.get("object", ""))
    if args.jobs_out:
        json.dump(list(jobs.values()), open(args.jobs_out, "w"), indent=1)
        print(f"{len(jobs)} jobs for the misses -> {args.jobs_out}")


if __name__ == "__main__":
    main()
