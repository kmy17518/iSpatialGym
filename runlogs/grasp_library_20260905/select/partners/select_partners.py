#!/usr/bin/env python3
"""Choose the partner object models of the IORS problems (ontop -0, under small-under-small -1, inside -2, nextto -3)
and write them into ``select/task_object_selection.csv`` (+ regenerate the .xlsx / .html twins).

The partner is not grasped, so there is no grasp test; a partner model is selected on

1. **ready**: no unprocessed knowledge-base complaint (``category_table.json`` of the grasp-library run, same flag the
   grasp library used),
2. **table fit**: its footprint can be placed by the instance sampler on every scene support that the table-selection run
   validated for an IORS problem (``runlogs/table_selection_20260911/results``) plus the test-round table. The sampler
   puts the partner centre on a free cell that is >= 4 cm inside the support top and needs every footprint vertex >= 2 cm
   inside (``region_cells`` / ``polygon_supported``, 2 cm raster), so a footprint of short side ``w`` fits a support of
   depth ``D`` at some yaw iff ``w <= D - 0.06``. The long side must also leave room for the longest test-pick target
   beside it along the support (``L <= length - 0.11 - target_long_side``). Heights above 0.35 m, long sides above 0.45 m
   and thicknesses below 5 mm (physics / visibility) are flagged and rank lower; more than 0.45 m tall or 0.5 m long is
   rejected (not a table partner). A viable model fits every validated support; models that fit the test table and >= 75 %
   of the supports (or miss some test picks) are "partial": not selected, only named in the note,
3. **relation geometry** against every target model listed for the task (rest-orientation extents = knowledge-base
   bbox x scale, rotated by the grasp library's ``rest_quat``):
   * ontop / under (small-under-small): the target is set down centred on the partner (``IORSSampler._goal_pose``), and
     ``OnTop`` / ``Under`` need the vertical ray at the target's AABB centre to hit the partner. Tiers: ``any_yaw`` =
     partner short side >= target long side (+1 cm) - no overhang whatever the relative yaw; ``aligned`` = fits without
     overhang when the long axes are aligned; ``centred`` = target overhangs but its contact patch under the CoM is
     supported (partner short side >= target short side + 3 cm); ``no`` otherwise,
   * inside: ``Inside`` is true when the target's AABB centre lies in the container's ``fillable`` / ``openfillable``
     meta-link volume (``partner_geometry.py`` reads it from the USD). The target is dropped at the volume centroid from
     the container top, so its footprint must pass the opening (``any_yaw``: target long side <= opening min width - 1 cm;
     ``aligned``: long side <= opening max width and short side <= opening min width) and, resting on the volume floor,
     its centre must stay below the volume top (``depth >= height / 2 + 1 cm``; flowers stand on their stem end:
     ``depth >= length / 2 + 1 cm`` and the opening only has to pass the stem),
   * nextto: ``NextTo`` requires the AABB gap to be < mean(dims_a + dims_b) / 6; the sampler's goal gap is
     ``generation.nextto_gap`` (0.02 m), so the pair needs mean(dims) / 6 > 0.02 (``gap_ok``), else the note gives the
     gap to use (``gap_tight``),
4. **ranking**: fits all test picks (pick 1 / pick 2 / alternate) of every paired target category > fewer overhang /
   no-fit targets > fewer support misses > pinned in the test spec (already sampler-validated) > fewer flags > larger
   support margin > size closest to the category median. Pick 2 prefers a different geometry (vertex hash) from pick 1.
   Exactly min(viable, 12) models per category are listed (like the 12-model cap of the grasp library); further viable
   models are named in the first row's note.

Exact duplicates (``select/duplicates/verdicts.json`` where the category was reviewed, else same vertex hash and
diffuse-texture md5) are counted once; same geometry with other textures is marked as a texture variant.

Outputs: ``partner_fit.json`` (every ready model with its measurements, tiers per target and verdict) and the rewritten
partner blocks of the CSV (rows between the ``partner`` header of each IORS task and the next header row; fixed scene
partners such as countertop / desk / bed / couch are kept as they were). Idempotent.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import re
import statistics
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.abspath(os.path.join(HERE, ".."))
LIB = os.path.abspath(os.path.join(SEL, ".."))
REPO = os.path.abspath(os.path.join(LIB, "..", ".."))
CSV_PATH = os.path.join(SEL, "task_object_selection.csv")
GEOM = os.path.join(HERE, "partner_geometry.json")
OUT_JSON = os.path.join(HERE, "partner_fit.json")
VERDICTS = os.path.join(SEL, "duplicates", "verdicts.json")
CATEGORY_TABLE = os.path.join(REPO, "runlogs", "grasp_exp_20260904", "category_table.json")
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
TABLE_RUN = os.path.join(REPO, "runlogs", "table_selection_20260911")
RUN_REL = "runlogs/grasp_library_20260905/select/partners"
COLUMNS = ["task", "role", "category", "model", "selection", "pair", "duplicates", "note"]

TASKS = OrderedDict([
    ("inter_object_relation_satisfaction-0", "ontop"),
    ("inter_object_relation_satisfaction-1", "under"),
    ("inter_object_relation_satisfaction-2", "inside"),
    ("inter_object_relation_satisfaction-3", "nextto"),
])
FLOWERS = {"sunflower", "lily", "dahlia_flower"}
#: partner categories that are fixed scene furniture in the sheet (rows kept as they are)
SCENE_PARTNERS = {"countertop", "desk", "bed", "couch"}
TEST_TABLE = "breakfast_table_rhjoby_0"
HEIGHT_CAP = 0.35  # flagged above
LONG_CAP = 0.45  # flagged above
HEIGHT_HARD = 0.45  # rejected above
LONG_HARD = 0.50  # rejected above
THIN = 0.005
NEXTTO_GAP = 0.02
MAX_ROWS = 12
TIER_RANK = {"sim_ok": 0, "any_yaw": 0, "gap_ok": 0, "aligned": 1, "gap_tight": 1, "centred": 2, "no": 3}
PICK_SEL = ("pick 1", "pick 2", "alternate")


# ----------------------------------------------------------------------------------------------------------------------
# inputs
def quat_to_R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_targets(rows, ct):
    """Per IORS task: the target rows of the sheet with their rest-orientation extents (m) and partner text."""
    targets = defaultdict(list)
    cur, cat = None, None
    for r in rows:
        if r["task"]:
            cur, cat = (r["task"].split(" ")[0], r["role"]), None
            continue
        if cur is None or cur[0] not in TASKS or cur[1] != "target":
            continue
        if r["category"] and not re.match(r"^\d+ \(\d+\)$", r["category"]):
            cat = r["category"]
        m = re.search(r"grasp key (\S+?);", r["note"])
        if not m:
            continue
        key = m.group(1)
        sc = re.search(r"scaled x\[([^\]]+)\]", r["note"])
        scale = [float(v) for v in sc.group(1).split(",")] if sc else [1.0, 1.0, 1.0]
        y = yaml.safe_load(open(os.path.join(LIB, "cands", f"{key}.yaml")))
        q = y["ispatialgym"]["rest_quat_wxyz"]
        ext = np.abs(quat_to_R(*q)) @ (np.array(ct[r["model"]]["bbox"]) * np.array(scale))
        targets[cur[0]].append({"category": cat, "model": r["model"], "selection": r["selection"], "pair": r["pair"].replace("-> ", ""),
                                "key": key, "scale": scale, "rest_pose": y["ispatialgym"].get("rest_pose", "canonical"),
                                "ext": [round(float(v), 4) for v in ext]})
    return targets


def pair_to_categories(task, pair):
    """The partner categories named by a pair text ('platter', 'countertop (scene)', 'platter under fries (fries grasped)',
    'coaster / saucer under glass (glass grasped)', 'vase (flower_stem_top)', 'under bed / desk')."""
    p = pair.split("(")[0].strip()
    if TASKS[task] == "under":
        if p.startswith("under "):  # small under big: scene furniture
            return [c.strip() for c in p[6:].split("/")]
        return [c.strip() for c in p.split(" under ")[0].split("/")]
    return [p.split(" ")[0]]


def partner_categories_of(task, targets):
    out = OrderedDict()
    for t in targets[task]:
        for pc in pair_to_categories(task, t["pair"]):
            out.setdefault(pc, [])
            if t["category"] not in out[pc]:
                out[pc].append(t["category"])
    return OrderedDict(sorted(out.items()))


def validated_supports():
    """Scene supports on which the table-selection run generated >= 1 IORS instance (small/small problems), plus the
    test-round table: 'scene:name' -> depth (short side), length (long side), top height, problems."""
    inv = {}
    with open(os.path.join(TABLE_RUN, "inventory.csv"), newline="") as f:
        for r in csv.DictReader(f):
            inv[(r["scene"], r["name"])] = r
    sup = {}
    for p in sorted(glob.glob(os.path.join(TABLE_RUN, "results", "*__inter_object_relation_satisfaction-*.json"))):
        d = json.load(open(p))
        if d.get("status") != "ok" or d.get("partner"):
            continue
        sup.setdefault((d["scene"], d["table"]), set()).add(d["problem"].rsplit("-", 1)[1])
    spec = json.load(open(SPEC))
    for prob in TASKS:
        for scene, cfg in spec[prob]["scenes"].items():
            t = (cfg.get("fixed") or {}).get("table")
            if isinstance(t, list):
                for n in t:
                    sup.setdefault((scene, n), set()).add("spec")
    out = OrderedDict()
    for (scene, name), probs in sorted(sup.items()):
        r = inv.get((scene, name))
        if r is None:
            continue
        bx, by = float(r["world_ext_x"]), float(r["world_ext_y"])
        out[f"{scene}:{name}"] = {"scene": scene, "name": name, "depth": round(min(bx, by), 3), "length": round(max(bx, by), 3),
                                  "top_z": float(r["top_z"]), "problems": sorted(probs)}
    return out


def load_inside_sim():
    """(container 'cat/model', target 'cat/model') -> {mode: record} from check_inside_sim.py, if it has been run."""
    path = os.path.join(HERE, "inside_sim.json")
    out = defaultdict(dict)
    if not os.path.exists(path):
        return out, {}
    d = json.load(open(path))
    for r in d.get("results", []):
        if "error" in r:
            continue
        out[(r["container"], r["target"])][r["mode"]] = r
    return out, d.get("containers", {})


def load_verdict_groups():
    """category -> list of exact groups (sets of models) from the reviewed duplicate check."""
    out = defaultdict(list)
    if not os.path.exists(VERDICTS):
        return out
    for v in json.load(open(VERDICTS)):
        for g in v.get("exact_groups") or []:
            out[v["category"]].append(set(g))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# rules
def fits_support(p_bbox, t_ext, depth, length):
    """Some arrangement of partner and target side by side on the support exists: each object's extent across the
    support depth <= depth - 0.06 (cell margin 4 cm + footprint margin 2 cm) and the two extents along the support length
    plus 3 cm gap and 8 cm margins <= length; both objects may lie along or across the support."""
    w_p, L_p = sorted(p_bbox[:2])
    t_w, t_L = sorted(t_ext[:2])
    usable = depth - 0.06
    for across_p, along_p in ((w_p, L_p), (L_p, w_p)):
        if across_p > usable:
            continue
        for across_t, along_t in ((t_w, t_L), (t_L, t_w)):
            if across_t > usable:
                continue
            if along_p + along_t + 0.11 <= length:
                return True
    return False


def table_fit(bbox, supports, pick_targets):
    w, L, h = min(bbox[0], bbox[1]), max(bbox[0], bbox[1]), bbox[2]
    misses = []
    for s in supports.values():
        if not all(fits_support(bbox, t["ext"], s["depth"], s["length"]) for t in pick_targets):
            misses.append(s["name"])
    flags = []
    if h > HEIGHT_CAP:
        flags.append(f"tall ({h * 100:.0f} cm)")
    if L > LONG_CAP:
        flags.append(f"long side {L * 100:.0f} cm")
    if h < THIN:
        flags.append(f"very thin ({h * 1000:.0f} mm)")
    return {"short_side": round(w, 4), "long_side": round(L, 4), "height": round(h, 4), "misses": misses, "flags": flags,
            "fits_all": not misses, "fits_test_table": TEST_TABLE not in misses}


def support_tier(p_bbox, t_ext):
    w_p, L_p = sorted(p_bbox[:2])
    t_w, t_L = sorted(t_ext[:2])
    if w_p >= t_L + 0.01:
        return "any_yaw", f"margin {(w_p - t_L) * 100:.0f} cm"
    if w_p >= t_w + 0.01 and L_p >= t_L + 0.01:
        return "aligned", f"target {t_L * 100:.0f} cm long vs partner short side {w_p * 100:.0f} cm"
    if w_p >= t_w + 0.03:
        return "centred", f"overhang {(t_L - w_p) * 100:.0f} cm"
    return "no", f"partner short side {w_p * 100:.0f} cm < target {t_w * 100:.0f} x {t_L * 100:.0f} cm"


def inside_tier(fill, t_ext, flower):
    if fill is None:
        return "no", "no fillable meta-link"
    t_w, t_L = sorted(t_ext[:2])
    t_h = t_ext[2]
    if flower:
        need = max(t_ext) / 2 + 0.01
        if fill["depth"] < need:
            return "no", f"volume depth {fill['depth'] * 100:.1f} cm < half flower length {need * 100:.1f} cm"
        if fill["opening_min"] < 0.03:
            return "no", f"opening {fill['opening_min'] * 100:.1f} cm too narrow for the stem"
        return "any_yaw", f"stands: depth {fill['depth'] * 100:.1f} >= {need * 100:.1f} cm, opening {fill['opening_min'] * 100:.1f} cm"
    need = t_h / 2 + 0.01
    if fill["depth"] < need:
        return "no", f"volume depth {fill['depth'] * 100:.1f} cm < half target height {need * 100:.1f} cm (centre would sit above the volume)"
    if t_L <= fill["opening_min"] - 0.01:
        return "any_yaw", f"opening {fill['opening_min'] * 100:.1f} cm >= target {t_L * 100:.1f} cm, depth {fill['depth'] * 100:.1f} cm"
    if t_L <= fill["opening_max"] - 0.01 and t_w <= fill["opening_min"] - 0.01:
        return "aligned", f"opening {fill['opening_min'] * 100:.1f} x {fill['opening_max'] * 100:.1f} cm fits target {t_w * 100:.1f} x {t_L * 100:.1f} cm only aligned"
    return "no", f"opening {fill['opening_min'] * 100:.1f} x {fill['opening_max'] * 100:.1f} cm < target {t_w * 100:.1f} x {t_L * 100:.1f} cm"


def nextto_bound(p_bbox, t_ext):
    return float(sum(p_bbox) + sum(t_ext)) / 3.0 / 6.0


def fmt_cm(v):
    return f"{v * 100:.1f}"


def bbox_txt(b):
    return " x ".join(fmt_cm(v) for v in b) + " cm"


# ----------------------------------------------------------------------------------------------------------------------
def evaluate_category(task, rel, pcat, tlist, geom, supports, pinned, verdict_groups, sim_results):
    """Every ready model of ``pcat``: measurements, per-target tiers, verdict, ranking; returns (recs sorted, not_ready)."""
    models = [g for g in geom.values() if g["category"] == pcat and "error" not in g]
    ready = [g for g in models if g["ready"]]
    not_ready = [g for g in models if not g["ready"]]
    pick_targets = [t for t in tlist if t["selection"] in PICK_SEL]
    tcats = []
    for t in tlist:
        if t["category"] not in tcats:
            tcats.append(t["category"])
    fits = {g["model"]: table_fit(g["bbox"], supports, pick_targets) for g in ready}
    # "representative size" = median long side of the ready models that fit the supports (oversize models do not skew it)
    pool_sizes = [max(g["bbox"][:2]) for g in ready if fits[g["model"]]["fits_all"]] or [max(g["bbox"][:2]) for g in ready]
    median_size = statistics.median(pool_sizes) if pool_sizes else 0.0
    recs = []
    for g in ready:
        tf = fits[g["model"]]
        per_t = []
        for t in tlist:
            sim = None
            if rel in ("ontop", "under"):
                tier, why = support_tier(g["bbox"], t["ext"])
            elif rel == "inside":
                tier, why = inside_tier(g.get("fillable"), t["ext"], t["category"] in FLOWERS)
                # the simulator check (check_inside_sim.py) is authoritative where it was run: the target dropped into
                # the container as the sampler does (flowers: standing on the stem end; the lying drop is reported too)
                runs = sim_results.get((f"{pcat}/{g['model']}", f"{t['category']}/{t['model']}"), {})
                mode = "standing" if t["category"] in FLOWERS else "rest"
                if mode in runs:
                    r_ = runs[mode]
                    sim = {"mode": mode, "inside": bool(r_["inside"]), "centre_below_top": r_.get("target_centre_below_container_top"),
                           "settled": r_.get("settled"), "container_moved": r_.get("container_moved"), "final_lin_vel": r_.get("final_lin_vel"),
                           # clean = inside, container not pushed away, target at rest by the sampler's STABLE_LIN_VEL (0.05 m/s)
                           "clean": bool(r_["inside"]) and (r_.get("container_moved") or 0) < 0.01 and (r_.get("final_lin_vel") or 0) < 0.05}
                    if "lying" in runs:
                        sim["lying_inside"] = bool(runs["lying"]["inside"])
                    geo = tier
                    cb = r_.get("target_centre_below_container_top") or 0.0
                    where = f"centre {abs(cb) * 100:.1f} cm {'below' if cb >= 0 else 'above'} the rim"
                    if sim["inside"]:
                        tier = "sim_ok"
                        why = f"sim Inside=True ({where}); geometry: {geo} - {why}"
                    else:
                        tier = "no"
                        why = f"sim Inside=False ({where}); geometry: {geo} - {why}"
                elif t["category"] in FLOWERS and tier != "no":
                    # the geometric depth rule is not sufficient for flowers (top-heavy, they topple out of narrow vases or
                    # knock the vase over in the simulator), so a flower pick counts only once check_inside_sim.py has run it
                    tier, why = "no", f"not simulated yet (flowers need the simulator check); geometry: {why}"
            else:
                b = nextto_bound(g["bbox"], t["ext"])
                tier, why = ("gap_ok" if b > NEXTTO_GAP + 0.002 else "gap_tight"), f"NextTo bound {b * 100:.1f} cm"
            per_t.append({"category": t["category"], "model": t["model"], "selection": t["selection"], "tier": tier, "why": why, "sim": sim})
        picks = [x for x in per_t if x["selection"] in PICK_SEL]
        # per paired target category: do all of its picks fit (tier != no)?
        cats_ok = {c: all(x["tier"] != "no" for x in picks if x["category"] == c) for c in tcats}
        margin = min((sorted(g["bbox"][:2])[0] - sorted(t["ext"][:2])[0] for t in pick_targets), default=0.0)
        recs.append({"model": g["model"], "bbox": g["bbox"], "table": tf, "fillable": g.get("fillable"), "targets": per_t, "picks": picks,
                     "cats_ok": cats_ok, "margin": round(margin, 4), "vertex_hash": g["vertex_hash"], "texture_md5": g.get("texture_md5", []),
                     "size_dev": abs(max(g["bbox"][:2]) - median_size), "pinned": bool(pinned.get((task, pcat, g["model"])))})
    # a paired target category that NO ready model of this partner category satisfies (e.g. lily -> vase in the simulator)
    # cannot decide viability; it is dropped from the requirement and reported in the category note
    def placeable(r):
        tf = r["table"]
        return tf["height"] <= HEIGHT_HARD and tf["long_side"] <= LONG_HARD and tf["fits_test_table"] and len(tf["misses"]) <= max(1, len(supports) // 4)
    placeable_recs = [r for r in recs if placeable(r)]
    unsatisfiable = [c for c in tcats if placeable_recs and not any(r["cats_ok"][c] for r in placeable_recs)]
    for r in recs:
        tf = r["table"]
        picks = [x for x in r["picks"] if x["category"] not in unsatisfiable]
        r["worst_pick_tier"] = max((TIER_RANK[x["tier"]] for x in picks), default=0)
        r["n_pick_fail"] = sum(x["tier"] == "no" for x in picks)
        r["n_all_fail"] = sum(x["tier"] == "no" for x in r["targets"] if x["category"] not in unsatisfiable)
        r["n_all_overhang"] = sum(TIER_RANK[x["tier"]] >= 2 for x in r["targets"] if x["category"] not in unsatisfiable)
        r["n_sim_unclean"] = sum(1 for x in picks if x.get("sim") and not x["sim"]["clean"])
        reasons = []
        table_ok = tf["fits_all"]
        table_partial = tf["fits_test_table"] and len(tf["misses"]) <= max(1, len(supports) // 4)
        oversize = tf["height"] > HEIGHT_HARD or tf["long_side"] > LONG_HARD
        if not tf["fits_all"]:
            reasons.append("does not fit " + ", ".join(tf["misses"]))
        if r["n_pick_fail"]:
            nosim = [x for x in picks if x["tier"] == "no" and x["why"].startswith("not simulated")]
            fails = [x for x in picks if x["tier"] == "no" and not x["why"].startswith("not simulated")]
            if fails:
                bad = sorted({x["category"] for x in fails})
                simbad = [f"{x['category']}/{x['model']}" for x in fails if x.get("sim")]
                reasons.append(f"{len(fails)}/{len(picks)} test picks do not fit ({', '.join(bad)})" + (f"; sim Inside=False for {', '.join(simbad)}" if simbad else ""))
            if nosim:
                reasons.append(f"{len(nosim)}/{len(picks)} flower test picks not simulated yet (geometry alone is not trusted for flowers)")
        reasons.extend(tf["flags"])
        if oversize:
            reasons.append(f"oversize for a table partner (> {HEIGHT_HARD * 100:.0f} cm tall or > {LONG_HARD * 100:.0f} cm long)")
        if table_ok and r["n_pick_fail"] == 0 and not oversize:
            verdict = "ok"
        elif table_partial and any(v for c, v in r["cats_ok"].items() if c not in unsatisfiable) and not oversize:
            verdict = "partial"
        else:
            verdict = "rejected"
        r["verdict"], r["reasons"] = verdict, reasons
        r["unsatisfiable_targets"] = unsatisfiable
    # duplicates: reviewed exact groups first, then the vertex hash / texture md5 test; models whose bbox agrees within
    # 1 mm on every axis are re-tessellations of one shape (colour variants) - the reviewed check called that tier same_shape
    by_hash = defaultdict(list)
    for r in recs:
        by_hash[r["vertex_hash"]].append(r)
    for r in recs:
        exact = set()
        for grp in verdict_groups.get(pcat, []):
            if r["model"] in grp:
                exact |= grp - {r["model"]}
        same = [x for x in by_hash[r["vertex_hash"]] if x is not r]
        exact |= {x["model"] for x in same if x["texture_md5"] == r["texture_md5"]}
        r["exact_dup"] = sorted(exact)
        variants = {x["model"] for x in same if x["texture_md5"] != r["texture_md5"]}
        variants |= {x["model"] for x in recs if x is not r and all(abs(a - b) <= 0.001 for a, b in zip(x["bbox"], r["bbox"]))}
        r["texture_variants"] = sorted(variants - exact)
        r["shape_key"] = tuple(round(v, 3) for v in r["bbox"])

    def rank_key(r):
        # support margin (partner short side beyond the target short side) counts in 2 cm steps up to 4 cm, then size
        margin_bucket = -min(2, int(max(r["margin"], 0.0) / 0.02)) if rel in ("ontop", "under") else 0
        return ({"ok": 0, "partial": 1, "rejected": 2}[r["verdict"]], r["n_pick_fail"], r["worst_pick_tier"], r["n_all_fail"],
                r["n_all_overhang"], r["n_sim_unclean"], len(r["table"]["misses"]), 0 if r["pinned"] else 1, len(r["table"]["flags"]),
                margin_bucket, r["size_dev"], r["model"])
    recs.sort(key=rank_key)
    for i, r in enumerate(recs):
        r["rank"] = i + 1

    def looks_different(a, b):
        if a["vertex_hash"] == b["vertex_hash"]:
            return False
        dims = max(abs(x - y) for x, y in zip(a["bbox"], b["bbox"])) > 0.03 * max(a["bbox"])
        return dims or set(a["texture_md5"]) != set(b["texture_md5"])

    # exact groups: only the first member of each group is eligible for the picks
    labels = {}
    pool, seen = [], set()
    for r in recs:
        if r["verdict"] != "ok":
            continue
        sig = frozenset([r["model"]] + r["exact_dup"])
        if sig in seen:
            continue
        seen.add(sig)
        pool.append(r)
    if pool:
        p1 = pool[0]
        labels[p1["model"]] = "pick 1"
        rest = pool[1:]
        if rest:
            best = rank_key(rest[0])[:8]
            p2 = next((r for r in rest if looks_different(r, p1) and rank_key(r)[:8] == best), rest[0])
            labels[p2["model"]] = "pick 2"
            alt = next((r for r in rest if r is not p2), None)
            if alt is not None:
                labels[alt["model"]] = "alternate"
    for r in recs:
        r["selection"] = labels.get(r["model"], "candidate" if r["verdict"] == "ok" else r["verdict"])
    return recs, not_ready, tcats, median_size


def category_rows(rel, pcat, recs, not_ready, tcats, supports):
    """Rows for one partner category: the min(viable, MAX_ROWS) best viable models (verdict ok); partial and rejected
    models are only named in the first row's note."""
    viable = [r for r in recs if r["verdict"] == "ok"]
    shown = viable[:MAX_ROWS]
    overflow = viable[MAX_ROWS:]
    partial = [r for r in recs if r["verdict"] == "partial"]
    rejected = [r for r in recs if r["verdict"] == "rejected"]
    n_geom = len({r["shape_key"] for r in shown})
    n_models = len({frozenset([r["model"]] + r["exact_dup"]) for r in shown})
    rows = []
    for i, r in enumerate(shown):
        note = []
        if r["pinned"]:
            note.append("PINNED in task_custom_lists.json (role partner)")
        note.append("ready")
        note.append("bbox " + bbox_txt(r["bbox"]))
        tf = r["table"]
        if tf["fits_all"]:
            note.append(f"table fit: short side {fmt_cm(tf['short_side'])} cm fits all {len(supports)} validated IORS supports")
        else:
            note.append(f"table fit: fits {len(supports) - len(tf['misses'])}/{len(supports)} validated IORS supports, NOT " + ", ".join(tf["misses"]))
        if tf["flags"]:
            note.append("flags: " + ", ".join(tf["flags"]))
        tiers = defaultdict(list)
        for x in r["targets"]:
            tiers[x["tier"]].append(f"{x['category']}/{x['model']}")
        if rel in ("ontop", "under"):
            parts = [f"{t} {len(tiers[t])}" + ("" if t == "any_yaw" else f" ({', '.join(tiers[t])})") for t in ("any_yaw", "aligned", "centred", "no") if tiers[t]]
            note.append(f"supports the {len(r['targets'])} listed targets: " + ", ".join(parts))
            if r.get("fillable"):
                note.append(f"flat floor {fmt_cm(r['fillable']['floor_min'])}-{fmt_cm(r['fillable']['floor_max'])} cm inside the rim")
        elif rel == "inside":
            f_ = r["fillable"]
            rim = f_["top_below_object_top"]
            note.append(f"fillable volume: opening {fmt_cm(f_['opening_min'])}-{fmt_cm(f_['opening_max'])} cm, depth {fmt_cm(f_['depth'])} cm, "
                        f"floor {fmt_cm(f_['floor_above_bottom'])} cm above the base, top {fmt_cm(abs(rim))} cm {'below' if rim >= 0 else 'above'} the rim")
            parts = [f"{t} {len(tiers[t])}" + ("" if t in ("any_yaw", "sim_ok") else f" ({', '.join(tiers[t])})") for t in ("sim_ok", "any_yaw", "aligned", "no") if tiers[t]]
            note.append(f"fits the {len(r['targets'])} listed targets: " + ", ".join(parts))
            simmed = [x for x in r["targets"] if x.get("sim")]
            if simmed:
                ok_ = [x for x in simmed if x["sim"]["inside"]]
                bad_ = [f"{x['category']}/{x['model']}" for x in simmed if not x["sim"]["inside"]]
                unclean = [f"{x['category']}/{x['model']} (moved {(x['sim']['container_moved'] or 0) * 100:.1f} cm, v {x['sim']['final_lin_vel']:.2f} m/s)"
                           for x in ok_ if not x["sim"]["clean"]]
                txt = (f"SIM (check_inside_sim.py: container on the ground, target dropped centred on the fillable volume from 5 mm above the rim "
                       f"as the sampler's inside goal does, settled): Inside=True for {len(ok_)}/{len(simmed)} test picks")
                if bad_:
                    txt += " (False: " + ", ".join(bad_) + ")"
                if unclean:
                    txt += "; container pushed or target still moving for " + ", ".join(unclean)
                lying = [x for x in simmed if "lying_inside" in x["sim"]]
                if lying:
                    txt += (f"; flowers were tested standing on the stem end ({sum(x['sim']['inside'] for x in lying)}/{len(lying)} inside) and lying "
                            f"flat as today's sampler goal would place them ({sum(x['sim']['lying_inside'] for x in lying)}/{len(lying)} inside)")
                note.append(txt)
        else:
            bounds = {x["model"]: float(x["why"].split()[-2]) for x in r["targets"]}
            lo, hi = min(bounds.values()), max(bounds.values())
            if not tiers["gap_tight"]:
                note.append(f"NextTo bound mean(dims)/6 = {lo:.1f}-{hi:.1f} cm > sampler nextto_gap {fmt_cm(NEXTTO_GAP)} cm for every listed target")
            else:
                note.append(f"NextTo bound mean(dims)/6 = {lo:.1f}-{hi:.1f} cm; the sampler's nextto_gap 2 cm is too large for "
                            f"{len(tiers['gap_tight'])}/{len(r['targets'])} targets ({', '.join(tiers['gap_tight'])}) -> use nextto_gap <= {max(0.5, lo - 0.4):.1f} cm")
        if r["exact_dup"]:
            note.append("exact duplicate of " + ", ".join(r["exact_dup"]) + " (counted once)")
        if i == 0:
            note.append(f"{len(viable)} viable ready model(s), {len(shown)} listed (cap {MAX_ROWS})")
            for c in r.get("unsatisfiable_targets", []):
                fails = [x for x in r["targets"] if x["category"] == c and x["selection"] in PICK_SEL]
                why = "; ".join(sorted({x["why"].split(";")[0] for x in fails}))[:300]
                note.append(f"NO VIABLE PARTNER FOR TARGET CATEGORY {c}: none of the table-sized ready {pcat} models passes for its test picks "
                            f"({why}); {c} is excluded from the viability requirement of this block and needs another partner category or a "
                            f"different goal procedure")
            if overflow:
                note.append("also viable, not listed: " + ", ".join(x["model"] for x in overflow))
            if partial:
                note.append("partial fit (not selected): " + "; ".join(f"{x['model']} ({'; '.join(x['reasons'])})" for x in partial))
            if rejected:
                note.append("rejected ready models: " + "; ".join(f"{x['model']} ({'; '.join(x['reasons'])})" for x in rejected))
            if not_ready:
                note.append("not ready (unprocessed complaint): " + ", ".join(f"{g['model']} [{'/'.join(g['complaints'])}]" for g in not_ready))
        cat_cell = pcat if i == 0 else ""
        if i == len(shown) - 1 and len(shown) > 1:
            cat_cell = f"{n_geom} ({n_models})"
        dup = ""
        if r["exact_dup"]:
            dup = "exact duplicate of " + ", ".join(r["exact_dup"])
        elif r["texture_variants"]:
            dup = "same geometry as " + ", ".join(r["texture_variants"]) + " (texture/colour variant)"
        rows.append({"task": "", "role": "", "category": cat_cell, "model": r["model"], "selection": r["selection"],
                     "pair": "<- " + ", ".join(sorted(tcats)), "duplicates": dup, "note": "; ".join(note)})
    return rows, n_geom, n_models


OLD_SENTENCE = ("One row per partner category named in the selection; model pinned only where the test spec fixes it, otherwise any ready "
                "model of the category.")
NEW_SENTENCE = ("Scene partners (countertop / desk / bed / couch) are fixed scene objects; object partners are listed per model below "
                "(selection = pick 1 / pick 2 / alternate / candidate as for the targets; PINNED = pinned in task_custom_lists.json).")
HEADER_METHOD = (" Partner MODELS ({date}): the rows below list, per partner category, the ready models (no unprocessed knowledge-base "
                 "complaint) that (a) the instance sampler can place on the scene supports validated for IORS by the table-selection "
                 "run so far plus the test table ({n_sup} supports: {sup_txt}; partner and target side by side: each extent across the "
                 "support <= depth - 6 cm and the two extents along it + 11 cm <= length, the sampler's cell / footprint margins) and "
                 "(b) {rule}. Ranking: all test picks of the paired targets fit > fewer overhang / no-fit targets > fewer support misses "
                 "> pinned (sampler-validated) > flags (tall / long / very thin) > support margin > size closest to the category median; "
                 "pick 2 prefers a different geometry. Listed = min(viable, {cap}) models per category (viable = ready + fits every support + "
                 "fits every test pick); further viable models, partial fits (some supports or some picks missed; not selected), "
                 "rejected and not-ready models are named in the first row of the category. Evidence: {run}/partner_fit.json, partner_geometry.json "
                 "(bboxes and fillable volumes read from the USDs), select_partners.py.")
RULE_TXT = {
    "ontop": ("is wide enough to support every listed target set down centred on it (tiers: any_yaw = partner short side >= target long "
              "side, aligned = no overhang when the long axes align, centred = target overhangs but its contact patch is supported)"),
    "under": ("(small-under-small) is wide enough to support the grasped target set down centred on it (tiers as for ontop). NOTE: the "
              "sampler's -1 currently implements only small-under-big (goal = floor point under the fixed partner); small-under-small "
              "needs the ontop-style goal with (under partner target)"),
    "inside": ("lets every test pick of the paired targets end up Inside: geometry first (the target passes the fillable-volume opening "
               "and, resting on the volume floor, keeps its AABB centre inside the volume - what Inside evaluates), then VERIFIED IN "
               "OMNIGIBSON (partners/check_inside_sim.py, partners/inside_sim.json): container on the ground, target in its grasp-library "
               "rest orientation dropped centred on the fillable volume from 5 mm above the rim (the sampler's inside goal), settled, "
               "Inside evaluated - every listed model passed for every test pick (flowers tested standing on the stem end; the simulator "
               "result overrides the geometry and is required for flowers: 2 vases that passed the depth rule dropped the flowers out). "
               "NOTE: the sampler's inside goal keeps the target's init orientation (lying flat for flowers, Inside=False in every vase), "
               "so flower -> vase needs the goal to stand the flower up; no ready table-sized vase holds the 42-47 cm lilies even standing"),
    "nextto": ("gives a NextTo bound mean(dims_a + dims_b)/6 above the sampler's 2 cm goal gap (gap_ok; gap_tight rows say which pairs "
               "need a smaller nextto_gap)"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write partner_fit.json only")
    args = ap.parse_args()

    ct = json.load(open(CATEGORY_TABLE))["objects"]
    geom = json.load(open(GEOM))
    spec = json.load(open(SPEC))
    verdict_groups = load_verdict_groups()
    sim_results, sim_containers = load_inside_sim()
    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    targets = load_targets(rows, ct)
    supports = validated_supports()
    pinned = {}
    for prob in TASKS:
        for cat, models in (spec[prob]["roles"].get("partner", {}).get("models") or {}).items():
            for m in models:
                pinned[(prob, cat, m)] = True

    evidence = {"date": datetime.date.today().isoformat(), "supports": supports,
                "rules": {"table": "partner and target side by side: extents across the support <= depth - 0.06 m, extents along it + 0.11 m <= length (any of the 4 orientations)",
                          "height_cap": HEIGHT_CAP, "long_cap": LONG_CAP, "height_hard": HEIGHT_HARD, "long_hard": LONG_HARD, "thin": THIN, "nextto_gap": NEXTTO_GAP, "max_rows": MAX_ROWS},
                "targets": targets, "tasks": OrderedDict()}
    new_blocks, header_counts = {}, {}
    for task, rel in TASKS.items():
        task_ev = OrderedDict()
        sub_rows = []
        n_geoms, n_models = 0, 0
        for pcat, _ in partner_categories_of(task, targets).items():
            if pcat in SCENE_PARTNERS:
                continue
            tlist = [t for t in targets[task] if pcat in pair_to_categories(task, t["pair"])]
            recs, not_ready, tcats, median_size = evaluate_category(task, rel, pcat, tlist, geom, supports, pinned, verdict_groups, sim_results)
            crows, ng, nm = category_rows(rel, pcat, recs, not_ready, tcats, supports)
            sub_rows.extend(crows)
            n_geoms += ng
            n_models += nm
            task_ev[pcat] = {"targets": tcats, "n_ready": len(recs), "n_not_ready": len(not_ready), "median_long_side": median_size,
                             "picks": {r["selection"]: r["model"] for r in recs if r["selection"] in PICK_SEL}, "models": recs}
        evidence["tasks"][task] = task_ev
        new_blocks[task] = sub_rows
        header_counts[task] = (n_geoms, n_models)
    json.dump(evidence, open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_JSON}")
    for task, ev in evidence["tasks"].items():
        print(task + ": " + "; ".join(f"{c}: " + ", ".join(f"{k}={v}" for k, v in d["picks"].items()) for c, d in ev.items()))
    if args.dry_run:
        return

    # ------------------------------------------------------------------------------------------------ rewrite the CSV
    out, i, n_updated = [], 0, 0
    today = datetime.date.today().isoformat()
    while i < len(rows):
        r = rows[i]
        label = r["task"].split(" ")[0]
        if label in TASKS and r["role"] == "partner":
            j = i + 1
            old = []
            while j < len(rows) and not rows[j]["task"]:
                old.append(rows[j])
                j += 1
            keep = [x for x in old if x["category"] in SCENE_PARTNERS]
            blocks, cur = [], None
            for x in keep + new_blocks[label]:
                if x["category"] and not re.match(r"^\d+ \(\d+\)$", x["category"]):
                    cur = [x["category"], [x]]
                    blocks.append(cur)
                else:
                    cur[1].append(x)
            blocks.sort(key=lambda b: b[0])
            hdr = dict(r)
            base_note = re.sub(r" Partner MODELS \(.*$", "", hdr["note"]).replace(OLD_SENTENCE, NEW_SENTENCE)
            sup_txt = ", ".join(f"{s['name']} {s['depth'] * 100:.0f}x{s['length'] * 100:.0f} cm" for s in supports.values())
            hdr["note"] = base_note + HEADER_METHOD.format(date=today, n_sup=len(supports), sup_txt=sup_txt, rule=RULE_TXT[TASKS[label]],
                                                           cap=MAX_ROWS, run=RUN_REL)
            n_geom, n_models = header_counts[label]
            hdr["category"] = str(len(blocks))
            hdr["model"] = f"{n_geom} ({n_models})"
            out.append(hdr)
            out.extend(x for _, rows_ in blocks for x in rows_)
            n_updated += 1
            i = j
            continue
        out.append(r)
        i += 1
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"updated {n_updated} partner blocks; {len(out)} rows -> {args.csv}")
    if not args.no_render:
        sys.path.insert(0, os.path.join(TABLE_RUN, "scripts"))
        from update_selection_csv import render  # same styling as the table rows

        render(out)


if __name__ == "__main__":
    main()
