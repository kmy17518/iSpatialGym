#!/usr/bin/env python3
"""Rotational-symmetry survey (geometry + texture) of every orientation-evaluated icl model.

Which rotations of the canonical object frame leave a model looking identical? Feeds two rules of
``ispatialgym/docs/instance_sampling_spec.md``: mental-rotation targets must not be symmetric under the sampled
rotation, and every orientation-evaluated target gets ``orientation_equivalences`` in its task record. Output columns
``symmetry`` / ``symmetry_near`` of ``ispatialgym/assets/icl/roles/candidates.csv`` (via ``scripts/icl/build_icl.py
--symmetry``).

Env: the ``base`` conda env (usd-core ``pxr``, trimesh, scipy, pyrender / EGL; no OmniGibson). The texture stage needs
the GPU (~0.3 GB); the geometry stage is CPU multiprocessing.

Stages (all idempotent, ``--force`` recomputes; ``OUT`` defaults to ``runlogs/symmetry_20260915``)::

    python scripts/symmetry/survey_symmetry.py models  [--icl ispatialgym/assets/icl/roles/candidates.csv] [--out OUT]
        -> OUT/models.json : the model set (kind sampleable / group, minus the block-design block, the disembedding
           trays and the small-scale articulation targets) with the grasp-tested spawn scale where the icl has one;
           prints the count and the per-(subtask, role) breakdown
    python scripts/symmetry/survey_symmetry.py geom     [--workers 28] [--limit N] [--only cat/model,...]
        -> OUT/geom/<category>__<model>.json : symmetric Chamfer scores of every candidate rotation (20k-point
           area-weighted surface sample vs a 400k dense sample, about the bbox centre and the centroid), for the
           canonical model and for the scaled variant
    python scripts/symmetry/survey_symmetry.py texture  [--shard K/N] [--limit N] [--only ...] [--layout-file ...] [--tags-file ...]
        -> OUT/tex/<key>.json : flat-shaded render comparison (6 cameras, 384 px, 768 px for elongated bodies) of the
           model vs the model rotated by every geometrically exact / near rotation: blurred percentile statistics and
           the local-feature (blob) test of symcore.image_diff; OUT/renders/<key>.jpg contact sheets (<= 100 KB) for
           review (--layout-file keeps the row order / displayed rotations of issued review batches)
    python scripts/symmetry/survey_symmetry.py report
        -> OUT/symmetry.json, OUT/symmetry.csv, OUT/report.md (per-tag counts, sanity table, mental-rotation targets
           with a symmetry, symmetric_180 disagreements, review list, failures)
    python scripts/symmetry/survey_symmetry.py diff [--previous OUT/superseded/v1/symmetry.json]
        -> OUT/report_diff.md : texture-class flips per tag, final-verdict changes, newly borderline models and the
           agreement with the reviewer verdicts in OUT/review/

Tags: ``x2 y2 z2`` half turns, ``x4 y4 z4`` quarter turns, ``x3 y3 z3 x5 y5 z5 x6 y6 z6`` 3-/5-/6-fold, ``xinf yinf zinf``
bodies of revolution, ``full`` spheres, reduced to generators (``symcore.reduce_tags``); ``none`` = surveyed and
asymmetric. Thresholds: ``symcore.GEOM_EXACT`` / ``GEOM_NEAR`` / ``TEX_PRESERVED`` / ``TEX_BORDERLINE_MAD`` (documented
in the runlog README).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import symcore as S  # noqa: E402

DEFAULT_ICL = os.path.join(REPO, "ispatialgym", "assets", "icl", "roles", "candidates.csv")
DEFAULT_OUT = os.path.join(REPO, "runlogs", "symmetry_20260915")
SKIP_SUBTASK_PREFIX = ("configuration_matching-articulation", "configuration_matching-block_design")
SKIP_ROLES = {("disembedding", "source"), ("disembedding", "destination")}
SHEET_TILE = 96
SHEET_MAX_ROWS = 10
SHEET_MAX_BYTES = 100_000


def key_of(category: str, model: str) -> str:
    return f"{category}__{model}"


# ----------------------------------------------------------------------------------------------- models

def select_models(icl_path: str) -> List[dict]:
    rows = list(csv.DictReader(open(icl_path)))
    models: Dict[str, dict] = collections.OrderedDict()
    breakdown = collections.OrderedDict()
    for r in rows:
        if r["kind"] not in ("sampleable", "group"):
            continue
        if r["subtask"].startswith(SKIP_SUBTASK_PREFIX) or (r["subtask"], r["role"]) in SKIP_ROLES:
            continue
        if r.get("dataset"):
            continue
        k = key_of(r["category"], r["model"])
        e = models.setdefault(k, {"key": k, "category": r["category"], "model": r["model"], "scale": None, "uses": []})
        use = f"{r['subtask']}/{r['role']}"
        if use not in e["uses"]:
            e["uses"].append(use)
        if r.get("scale"):
            sc = json.loads(r["scale"])
            if e["scale"] is not None and e["scale"] != sc:
                print(f"WARNING {k}: several spawn scales in the icl ({e['scale']} vs {sc}); keeping the first", file=sys.stderr)
            elif e["scale"] is None:
                e["scale"] = sc
        breakdown.setdefault((r["subtask"], r["role"]), set()).add(k)
    return list(models.values()), breakdown


def stage_models(args):
    models, breakdown = select_models(args.icl)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "models.json"), "w") as f:
        json.dump(models, f, indent=1)
    n_scaled = sum(1 for m in models if m["scale"])
    print(f"{len(models)} distinct (category, model) to survey ({len({m['category'] for m in models})} categories; "
          f"{n_scaled} with a grasp-tested spawn scale -> evaluated at scale too)")
    for (st, role), ks in breakdown.items():
        print(f"  {st:58s} {role:12s} {len(ks):4d} models")


# ----------------------------------------------------------------------------------------------- geometry

def geom_one(entry: dict, out_dir: str, force: bool = False) -> str:
    key = entry["key"]
    path = os.path.join(out_dir, key + ".json")
    if os.path.exists(path) and not force:
        return "skip"
    from meshload import load_model, merged_geometry, scaled

    t0 = time.time()
    rec = {"key": key, "category": entry["category"], "model": entry["model"], "scale": entry.get("scale")}
    try:
        meshes = load_model(entry["category"], entry["model"], textures=False)
        if not meshes:
            raise RuntimeError("no visual mesh")
        geom = merged_geometry(meshes)
        if geom.area <= 0:
            raise RuntimeError("zero surface area")
        rec.update({"n_meshes": len(meshes), "n_glass": sum(1 for m in meshes if m.metadata.get("glass")), "n_faces": int(len(geom.faces)),
                    "load_s": round(time.time() - t0, 3)})
        t1 = time.time()
        rec["base"] = S.geometric_scores(geom)
        rec["base"]["bbox"] = (geom.vertices.max(0) - geom.vertices.min(0)).round(5).tolist()
        if entry.get("scale"):
            gs = merged_geometry(scaled(meshes, entry["scale"]))
            rec["scaled"] = S.geometric_scores(gs)
            rec["scaled"]["bbox"] = (gs.vertices.max(0) - gs.vertices.min(0)).round(5).tolist()
        rec["score_s"] = round(time.time() - t1, 3)
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[:400]
        rec["traceback"] = traceback.format_exc()[-1500:]
    with open(path, "w") as f:
        json.dump(rec, f)
    return "error" if "error" in rec else "ok"


def _geom_worker(args):
    entry, out_dir, force = args
    try:
        return entry["key"], geom_one(entry, out_dir, force)
    except Exception as e:  # noqa: BLE001
        return entry["key"], f"error {e!r}"


def stage_geom(args):
    models = load_models(args)
    out_dir = os.path.join(args.out, "geom")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(m, out_dir, args.force) for m in models]
    t0 = time.time()
    counts = collections.Counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (key, status) in enumerate(pool.imap_unordered(_geom_worker, jobs, chunksize=2)):
            counts[status.split()[0]] += 1
            if status.startswith("error"):
                print(f"FAILED {key}: {status}", flush=True)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"[{i + 1}/{len(jobs)}] {dict(counts)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"geometry stage done: {dict(counts)} in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------------- texture

def candidate_tags(tag_scores: Dict[str, dict]) -> List[str]:
    """Tags worth a texture test: everything within the near mean threshold (frac2 is applied at report time, so a
    later re-tuning of the geometric thresholds does not need new renders)."""
    return [t for t in S.TAG_ORDER if t in tag_scores and tag_scores[t]["mean"] <= S.GEOM_NEAR["mean"] and not tag_scores[t].get("coarse")]


def texture_variant(renderer, meshes, g: dict, tags: List[str],
                    pins: Optional[Dict[str, str]] = None) -> Tuple[Dict[str, dict], list, Dict[str, list]]:
    """Texture scores of ``tags`` for one variant (base / scaled): {tag: {mad, p95, p99, blob_*, ..., worst, shown}},
    the base renders and the renders shown on the contact sheet per tag: the worst rotation of the tag
    (``symcore.tex_severity``: largest blob, then MAD) unless ``pins`` names the rotation a running review batch was
    issued with (``shown`` records which one it is)."""
    cands = S.candidate_rotations()
    pins = pins or {}
    renderer.set_size(renderer.base_size * renderer.size_factor(g.get("bbox", [1, 1, 1])))
    renderer.prepare(meshes, np.asarray(g["bbox_centre"]), float(g["diag"]))
    base = renderer.render(None)
    centres = {"bbox": np.asarray(g["bbox_centre"]), "centroid": np.asarray(g["centroid"])}
    scores, imgs = {}, {}
    for tag in tags:
        c = centres[g["tags"][tag].get("centre", "bbox")]
        worst, worst_imgs, pinned_imgs = None, None, None
        for rname, R in cands[tag]:
            rend = renderer.render(R, centre=c)
            s = S.image_diff(base, rend, m_per_px=renderer.m_per_px, view_names=renderer.view_names, diag=float(g["diag"]),
                             thin_mm=float(min(g.get("bbox", [1, 1, 1]))) * 1000.0)
            if worst is None or S.tex_severity(s) > S.tex_severity(worst):
                worst, worst_imgs = dict(s, worst=rname), rend
            if pins.get(tag) == rname:
                pinned_imgs = rend
            if worst["mad"] > 2 * S.TEX_BORDERLINE_MAD and worst["blob_px"] >= 2 * worst["n6"] and pins.get(tag) is None:
                break  # clearly broken: the remaining angles cannot rescue the tag
        if pinned_imgs is None and pins.get(tag):  # the pinned angle was not reached (early exit) - render it
            R = dict(cands[tag]).get(pins[tag])
            if R is not None:
                pinned_imgs = renderer.render(R, centre=c)
        worst["shown"] = pins[tag] if pinned_imgs is not None else worst["worst"]
        scores[tag] = worst
        imgs[tag] = pinned_imgs if pinned_imgs is not None else worst_imgs
    return scores, base, imgs


def sheet_rows(g: dict, tex: Dict[str, dict], first: List[str] = (), classify=None) -> List[str]:
    """Tags shown on the contact sheet: ``first`` (the rows a review batch was issued with / asks for, all of them),
    then the reduced exact + near sets, every texture-borderline tag (``classify``, default the current
    ``symcore.classify_tex``), then the other rendered tags, up to SHEET_MAX_ROWS (or the length of ``first``)."""
    classify = classify or S.classify_tex
    ex, nr = S.classify_geom(g["tags"])
    show = [t for t in first if t in tex]
    show += [t for t in S.reduce_tags(ex) if t not in show] + [t for t in S.reduce_tags(nr) if t not in ex and t not in show]
    for t in tex:
        if t not in show and classify(tex[t]) == "borderline":
            show.append(t)
    for t in tex:  # everything else that was rendered, until the cap
        if t not in show:
            show.append(t)
    return [t for t in show if t in tex][:max(SHEET_MAX_ROWS, len([t for t in first if t in tex]))]


def write_sheet(path: str, title: str, base_imgs, rows: List[tuple]):
    """``rows`` = [(label lines, renders)]; the top row is the un-rotated model."""
    from PIL import Image, ImageDraw, ImageFont

    def font(sz):
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
            if os.path.exists(p):
                return ImageFont.truetype(p, sz)
        return ImageFont.load_default()

    label_w, tile, hdr = 210, SHEET_TILE, 22
    n_views = len(base_imgs)
    W = label_w + n_views * tile
    H = hdr + (len(rows) + 1) * tile
    out = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(out)
    dr.text((4, 4), title[:140], fill=(0, 0, 0), font=font(12))
    all_rows = [(["original", "cameras +x +y -x -y +z -z"], base_imgs)] + rows
    for ri, (lines, imgs) in enumerate(all_rows):
        y0 = hdr + ri * tile
        for li, line in enumerate(lines[:5]):
            dr.text((4, y0 + 4 + 13 * li), line[:38], fill=(0, 0, 0) if li == 0 else (70, 70, 70), font=font(11 if li == 0 else 10))
        for vi, view in enumerate(imgs):
            im = Image.fromarray(view[0]).resize((tile, tile), Image.BOX)
            out.paste(im, (label_w + vi * tile, y0))
        dr.line([(0, y0), (W, y0)], fill=(200, 200, 200), width=1)
    for q in (82, 70, 58, 45):
        out.save(path, "JPEG", quality=q, optimize=True)
        if os.path.getsize(path) <= SHEET_MAX_BYTES:
            break


def texture_one(entry: dict, geom_rec: dict, renderer, out_dir: str, renders_dir: str, force: bool = False,
                extra_tags: Optional[Dict[str, List[str]]] = None, pins: Optional[Dict[str, Dict[str, str]]] = None) -> str:
    """Texture scores + contact sheet of one model. ``extra_tags`` = {"base" | "scaled": [tags]} a review round asks
    for (or the row order a running batch was issued with): rendered even when they are no geometric candidate, shown
    first on the sheet in that order, and the sheet is written even for a model without a geometric candidate.
    ``pins`` = {"base" | "scaled": {tag: rotation name}} keeps the displayed rotation of a row a running batch judges."""
    key = entry["key"]
    path = os.path.join(out_dir, key + ".json")
    if os.path.exists(path) and not force:
        return "skip"
    extra_tags = extra_tags or {}
    pins = pins or {}
    variants = [(v, geom_rec[v]) for v in ("base", "scaled") if geom_rec.get(v)]
    todo = [(v, g, [t for t in S.TAG_ORDER if t in set(candidate_tags(g["tags"])) | set(extra_tags.get(v, []))]) for v, g in variants]
    todo = [(v, g, tags) for v, g, tags in todo if tags]
    if not todo:
        return "nocand"
    from meshload import load_model, scaled

    t0 = time.time()
    rec = {"key": key, "category": entry["category"], "model": entry["model"], "scale": entry.get("scale")}
    try:
        meshes = load_model(entry["category"], entry["model"], textures=True)
        # OmniGlass parts are transparent in Isaac Sim (the policy sees what is behind them: a clock face under its
        # cover, a label inside a bottle), so they are left out of the texture renders; an all-glass model (wineglass,
        # water bottle) is rendered in a uniform glass tint instead - its visual symmetry is its geometric one.
        opaque = [m for m in meshes if not m.metadata.get("glass")]
        rec["n_glass"], rec["glass_only"] = len(meshes) - len(opaque), not opaque
        if opaque:
            meshes = opaque
        else:
            import trimesh
            from meshload import GLASS_COLOR

            for m in meshes:
                m.visual = trimesh.visual.ColorVisuals(m, face_colors=GLASS_COLOR)
        rec["n_textured"] = sum(1 for m in meshes if m.visual.kind == "texture")
        for v, g, tags in todo:
            ms = meshes if v == "base" else scaled(meshes, entry["scale"])
            scores, base_imgs, imgs = texture_variant(renderer, ms, g, tags, pins=pins.get(v))
            rec[v] = scores
            sheet = os.path.join(renders_dir, key + ("" if v == "base" else "__scaled") + ".jpg")
            forced = [t for t in extra_tags.get(v, []) if t in scores]
            if not any(S.classify_geom(g["tags"])) and not forced:  # scores kept (re-tuning), but no review sheet without a geometric candidate
                if os.path.exists(sheet):
                    os.remove(sheet)
                continue
            rows = []
            shown_rows = sheet_rows(g, scores, first=forced)
            rec.setdefault("sheet_rows", {})[v] = shown_rows
            for t in shown_rows:
                gs, ts = g["tags"][t], scores[t]
                blob = (f"blob {ts['blob_mm']:.0f}mm {ts['blob_view']} ({ts['blob_px']}px/{ts['n6']})" if ts.get("blob_px") else "blob -")
                lines = [f"{t}  ({ts.get('shown', ts['worst'])}) {S.classify_tex(ts)}",
                         f"geom mean {gs['mean'] * 100:.2f}% p95 {gs['p95'] * 100:.2f}% frac2 {gs['frac2'] * 100:.1f}%",
                         f"tex MAD {ts['mad'] * 255:.1f} p95 {ts['p95'] * 255:.1f} p99 {ts['p99'] * 255:.0f} gain {ts.get('gain', 1):.1f}",
                         f"     {blob}",
                         f"     union MAD {ts['mad_union'] * 255:.1f} IoU {ts['iou']:.3f}" +
                         (f" scores {ts['worst']}" if ts.get("shown", ts["worst"]) != ts["worst"] else "")]
                rows.append((lines, imgs[t]))
            ex, nr = S.classify_geom(g["tags"])
            bb = g.get("bbox", [0, 0, 0])
            title = (f"{entry['category']}/{entry['model']}" + (f"  scale {entry['scale']}" if v == "scaled" else "") +
                     f"   bbox {bb[0]:.3f} x {bb[1]:.3f} x {bb[2]:.3f} m   geom exact {' '.join(S.reduce_tags(ex)) or '-'}"
                     f"   near {' '.join(t for t in S.reduce_tags(nr) if t not in ex) or '-'}")
            write_sheet(sheet, title, base_imgs, rows)
        rec["render_s"] = round(time.time() - t0, 3)
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)[:400]
        rec["traceback"] = traceback.format_exc()[-1500:]
    with open(path, "w") as f:
        json.dump(rec, f)
    return "error" if "error" in rec else "ok"


def stage_texture(args):
    models = load_models(args)
    if args.shard:
        k, n = (int(v) for v in args.shard.split("/"))
        models = [m for i, m in enumerate(models) if i % n == k]
    extra: Dict[str, Dict[str, List[str]]] = {}
    pins: Dict[str, Dict[str, Dict[str, str]]] = {}
    if args.tags_file:  # {"<key>": {"base": [tags], "scaled": [tags]}} - render these, sheet them first
        extra = json.load(open(args.tags_file))
        models = [m for m in models if m["key"] in extra]
    if args.layout_file:  # {"<key>": {"base": {"first": [tags], "pin": {tag: rotation}}, "scaled": {...}}} - keep a batch's sheet layout
        for key, lay in json.load(open(args.layout_file)).items():
            for v, d in lay.items():
                if d.get("first"):
                    extra.setdefault(key, {})[v] = list(dict.fromkeys(list(d["first"]) + extra.get(key, {}).get(v, [])))
                if d.get("pin"):
                    pins.setdefault(key, {})[v] = dict(d["pin"])
    geom_dir, out_dir, renders_dir = os.path.join(args.out, "geom"), os.path.join(args.out, "tex"), os.path.join(args.out, "renders")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(renders_dir, exist_ok=True)
    renderer = S.Renderer(size=args.size, ss=2)
    t0 = time.time()
    counts = collections.Counter()
    for i, m in enumerate(models):
        gp = os.path.join(geom_dir, m["key"] + ".json")
        if not os.path.exists(gp):
            counts["nogeom"] += 1
            continue
        g = json.load(open(gp))
        if "error" in g:
            counts["geomerror"] += 1
            continue
        status = texture_one(m, g, renderer, out_dir, renders_dir, args.force or bool(args.tags_file), extra.get(m["key"]), pins.get(m["key"]))
        counts[status] += 1
        if status == "error":
            print(f"FAILED {m['key']}", flush=True)
        if (i + 1) % 25 == 0 or i + 1 == len(models):
            print(f"[{i + 1}/{len(models)}] {dict(counts)} ({time.time() - t0:.0f}s)", flush=True)
    renderer.close()
    print(f"texture stage done: {dict(counts)} in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------------------------------- report

def classify_model(g: dict, tex: Optional[Dict[str, dict]], prev_symmetry: Optional[List[str]] = None) -> dict:
    """Final verdict of one variant: geometry classes x texture classes -> symmetry / symmetry_near (reduced tags).
    ``prev_symmetry`` (the superseded pass's final tags): a tag the current criterion calls preserved that the previous
    pass did not is a *relaxation* and goes to ``symmetry_near`` (review) instead of ``symmetry`` - a re-tuned criterion
    may only tighten the equivalences on its own."""
    ex, nr = S.classify_geom(g["tags"])
    tex = tex or {}
    preserved, sym_raw, near_raw, tex_missing, relaxed = [], [], [], [], []
    prev_ok = set(S.expand_tags([t for t in (prev_symmetry or []) if t != "none"])) if prev_symmetry is not None else None
    for t in ex + nr:
        if t not in tex:
            tex_missing.append(t)
            cls = "unknown"
        else:
            cls = S.classify_tex(tex[t])
        if cls == "preserved":
            preserved.append(t)
        if t in ex and cls == "preserved":
            if prev_ok is not None and t not in prev_ok:
                relaxed.append(t)
                near_raw.append(t)
            else:
                sym_raw.append(t)
        elif cls in ("preserved", "borderline", "unknown"):
            near_raw.append(t)
    sym = S.reduce_tags(sym_raw)
    implied = set(S.expand_tags(sym))
    near = [t for t in S.reduce_tags(near_raw) if t not in implied]
    scores = {}
    for t in S.TAG_ORDER:
        s = g["tags"].get(t)
        if s is None or s.get("coarse"):
            continue
        scores[t] = {"geom": {k: round(s[k], 5) for k in ("mean", "p95", "frac2")}, "centre": s["centre"], "worst": s["worst"]}
        if t in tex:
            scores[t]["tex"] = {k: (round(tex[t][k], 5) if isinstance(tex[t][k], float) else tex[t][k])
                                for k in ("mad", "p95", "p99", "mad_raw", "mad_union", "iou", "depth_frac", "gain", "blob_px", "blob_frac", "blob_view",
                                          "blob_kind", "blob_d", "blob_mm", "n6", "sil_px", "sil_px_weak", "sil_view", "sil_mm")
                                if k in tex[t]}
            scores[t]["tex"]["worst"] = tex[t]["worst"]
            if tex[t].get("shown", tex[t]["worst"]) != tex[t]["worst"]:
                scores[t]["tex"]["shown"] = tex[t]["shown"]
            scores[t]["tex_class"] = S.classify_tex(tex[t])
    return {"geom_exact": S.reduce_tags(ex), "geom_near": [t for t in S.reduce_tags(nr) if t not in ex], "texture_preserved": S.reduce_tags(preserved),
            "symmetry": sym or ["none"], "symmetry_near": near, "geom_exact_raw": ex, "geom_near_raw": nr, "texture_missing": tex_missing,
            "relaxed_awaiting_review": relaxed,
            "scores": scores, "floor": {k: round(v, 5) for k, v in g["floor"].items()}, "diag": round(g["diag"], 5), "bbox": g.get("bbox")}


#: (category, expected tags / description) for the README sanity table
SANITY = [("bowl", "zinf (plain bowls; a decorated rim breaks it)"), ("plate", "zinf"), ("mug", "none (handle)"), ("teacup", "zinf if handle-less, else none"),
          ("teapot", "none"), ("sunflower", "none"), ("pencil", "xinf/yinf/zinf about the long axis if unprinted; a print breaks it"),
          ("bottle_of_medicine", "zinf geometry; the label breaks the texture -> none"), ("can_of_baking_mix", "zinf geometry; label -> none"),
          ("can_of_soda", "zinf geometry; label -> none"), ("board_game", "x2 y2 z2 box geometry; artwork -> none"), ("box_of_cookies", "box geometry; print -> none"),
          ("box_of_cereal", "box geometry; print -> none"), ("apple", "none / near zinf (stem)"), ("orange", "full or zinf geometry; texture decides"),
          ("tennis_ball", "full (plain) or none (seam)"), ("toy_dice", "cube geometry x4 y4 z4; pips -> none"), ("jar_of_grains", "zinf geometry; label -> none"),
          ("cap", "zinf"), ("lid", "zinf"), ("keyboard", "none"), ("painting", "none"), ("beaker", "zinf"), ("bar_soap", "zinf or x2 y2 z2"),
          ("wineglass", "zinf"), ("egg", "zinf or xinf (about the long axis)"), ("wall_clock", "none (numerals)"), ("vase", "zinf unless decorated"),
          ("platter", "zinf or x2 y2 z2 (oval)"), ("tray", "x2 y2 z2 (plain rectangular) or none")]


def most_common_tags(verdicts: List[List[str]]) -> str:
    c = collections.Counter(" ".join(v) for v in verdicts)
    return ", ".join(f"`{k}` x{n}" for k, n in c.most_common(3))


def alignment_half_turns(icl: List[dict], out: Dict[str, dict]) -> List[dict]:
    """Every alignment target row with the half turn that matters for its subtask (the wall normal = ``frame_front`` for
    the wall subtasks, the vertical = ``frame_up`` / canonical z for ``alignment-axial``): ``tag`` (``x2`` ...),
    ``axis_note``, the reviewer's ``symmetric_180`` flag, whether the survey has the half turn in ``symmetry`` (``has``)
    or in ``symmetry_near`` (``near``), and ``disagree`` = flag != has."""
    rows = []
    for r in icl:
        if not (r["subtask"].startswith("alignment") and r["role"] == "target"):
            continue
        rec = out.get(key_of(r["category"], r["model"]))
        if rec is None:
            continue
        if r["subtask"] == "alignment-axial":
            axis_src = r.get("frame_up") or "+z"
            axis_note = f"{axis_src} (up)"
        else:
            axis_src = r.get("frame_front") or "+x"
            axis_note = f"{axis_src} (front)"
        tag = f"{axis_src[-1]}2"
        has = tag in S.expand_tags(rec["symmetry"])
        near = tag in S.expand_tags(rec["symmetry_near"]) or tag in rec["symmetry_near"]
        flag = r.get("symmetric_180") == "1"
        rows.append({"subtask": r["subtask"], "category": r["category"], "model": r["model"], "key": key_of(r["category"], r["model"]),
                     "tag": tag, "axis_note": axis_note, "symmetric_180": r.get("symmetric_180") or "", "flag": flag, "has": has, "near": near,
                     "disagree": has != flag})
    return rows


def stage_report(args):
    models = load_models(args, full=True)
    icl = list(csv.DictReader(open(args.icl)))
    prev_path = os.path.join(args.out, "superseded", "v1", "symmetry.json")
    prev = json.load(open(prev_path)) if os.path.exists(prev_path) else {}
    out = collections.OrderedDict()
    failures, tex_failures = [], []
    for m in models:
        gp = os.path.join(args.out, "geom", m["key"] + ".json")
        if not os.path.exists(gp):
            failures.append((m["key"], "no geometry record"))
            continue
        g = json.load(open(gp))
        if "error" in g:
            failures.append((m["key"], g["error"]))
            continue
        tp = os.path.join(args.out, "tex", m["key"] + ".json")
        tex = json.load(open(tp)) if os.path.exists(tp) else {}
        if "error" in tex:
            tex_failures.append((m["key"], tex["error"]))
        rec = {"category": m["category"], "model": m["model"], "uses": m["uses"]}
        p = prev.get(m["key"])
        rec.update(classify_model(g["base"], tex.get("base"), p["symmetry"] if p else None))
        rec["scale"] = m.get("scale")
        if g.get("scaled"):
            sc = classify_model(g["scaled"], tex.get("scaled"), (p.get("symmetry_scaled") or ["none"]) if p else None)
            rec["symmetry_scaled"], rec["symmetry_near_scaled"] = sc["symmetry"], sc["symmetry_near"]
            rec["scaled"] = {k: sc[k] for k in ("geom_exact", "geom_near", "texture_preserved", "scores", "bbox", "geom_exact_raw", "geom_near_raw",
                                               "relaxed_awaiting_review")}
        else:
            rec["symmetry_scaled"], rec["symmetry_near_scaled"] = None, None
        rec["n_faces"], rec["n_meshes"] = g.get("n_faces"), g.get("n_meshes")
        rec["render"] = os.path.relpath(os.path.join(args.out, "renders", m["key"] + ".jpg"), REPO) if os.path.exists(os.path.join(args.out, "renders", m["key"] + ".jpg")) else None
        out[m["key"]] = rec
    with open(os.path.join(args.out, "symmetry.json"), "w") as f:
        json.dump(out, f, indent=0)
    # csv
    cols = ["category", "model", "symmetry", "symmetry_near", "geom_exact", "geom_near", "texture_preserved", "scale", "symmetry_scaled",
            "symmetry_near_scaled", "floor_mean", "best_tag", "best_geom_mean", "best_geom_frac2", "best_tex_mad", "best_tex_p95", "n_faces", "render", "uses"]
    with open(os.path.join(args.out, "symmetry.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k, r in out.items():
            best = min(r["scores"].items(), key=lambda kv: kv[1]["geom"]["mean"]) if r["scores"] else None
            w.writerow([r["category"], r["model"], ",".join(r["symmetry"]), ",".join(r["symmetry_near"]), ",".join(r["geom_exact"]),
                        ",".join(r["geom_near"]), ",".join(r["texture_preserved"]), json.dumps(r["scale"]) if r["scale"] else "",
                        ",".join(r["symmetry_scaled"]) if r["symmetry_scaled"] else "", ",".join(r["symmetry_near_scaled"]) if r["symmetry_near_scaled"] else "",
                        f"{r['floor']['mean']:.5f}", best[0] if best else "", f"{best[1]['geom']['mean']:.5f}" if best else "",
                        f"{best[1]['geom']['frac2']:.4f}" if best else "", f"{best[1]['tex']['mad'] * 255:.2f}" if best and "tex" in best[1] else "",
                        f"{best[1]['tex']['p95'] * 255:.2f}" if best and "tex" in best[1] else "", r["n_faces"], r["render"] or "", ";".join(r["uses"])])
    # report.md
    L = []
    n = len(out)
    tag_counts = collections.Counter(t for r in out.values() for t in r["symmetry"])
    near_counts = collections.Counter(t for r in out.values() for t in r["symmetry_near"])
    gex_counts = collections.Counter(t for r in out.values() for t in (r["geom_exact"] or ["none"]))
    L.append(f"## Per-tag counts ({n} models surveyed, {len(failures)} loader failures)\n")
    L.append("| tag | `symmetry` (final) | `symmetry_near` (review) | geometry exact (before texture) |\n|---|---|---|---|")
    for t in S.TAG_ORDER + ["none"]:
        if tag_counts[t] or near_counts[t] or gex_counts[t]:
            L.append(f"| `{t}` | {tag_counts[t]} | {near_counts[t]} | {gex_counts[t]} |")
    n_sym = sum(1 for r in out.values() if r["symmetry"] != ["none"])
    n_near = sum(1 for r in out.values() if r["symmetry_near"])
    n_cand = sum(1 for r in out.values() if r["geom_exact"] or r["geom_near"])
    L.append(f"\nModels with a final symmetry: {n_sym}; asymmetric (`none`): {n - n_sym}; with a `symmetry_near` entry to review: {n_near}; "
             f"with at least one geometric candidate (contact sheet rendered): {n_cand}.\n")
    # scaled variants
    sc = [(k, r) for k, r in out.items() if r["scale"]]
    diff = [(k, r) for k, r in sc if r["symmetry_scaled"] != r["symmetry"] or r["symmetry_near_scaled"] != r["symmetry_near"]]
    L.append(f"## Scaled variants\n\n{len(sc)} models have a grasp-tested spawn scale; the symmetry at that scale differs from the canonical model for "
             f"{len(diff)}:\n")
    for k, r in diff:
        L.append(f"- `{k}` scale {r['scale']}: canonical `{' '.join(r['symmetry'])}` / near `{' '.join(r['symmetry_near']) or '-'}` -> scaled "
                 f"`{' '.join(r['symmetry_scaled'])}` / near `{' '.join(r['symmetry_near_scaled']) or '-'}`")
    # sanity table
    L.append("\n## Sanity table\n\n| category | models | most common `symmetry` | most common geometry-exact | expectation | match |\n|---|---|---|---|---|---|")
    by_cat = collections.defaultdict(list)
    for r in out.values():
        by_cat[r["category"]].append(r)
    for cat, expect in SANITY:
        rs = by_cat.get(cat)
        if not rs:
            continue
        L.append(f"| {cat} | {len(rs)} | {most_common_tags([r['symmetry'] for r in rs])} | {most_common_tags([r['geom_exact'] or ['none'] for r in rs])} | {expect} | |")
    # mental rotation
    mr = [(k, r) for k, r in out.items() if any(u.startswith("mental_rotation/") for u in r["uses"]) and r["symmetry"] != ["none"]]
    mr_near = [(k, r) for k, r in out.items() if any(u.startswith("mental_rotation/") for u in r["uses"]) and r["symmetry"] == ["none"] and r["symmetry_near"]]
    L.append(f"\n## `mental_rotation` targets with a symmetry ({len(mr)} of "
             f"{sum(1 for r in out.values() if any(u.startswith('mental_rotation/') for u in r['uses']))}; exclude or give `orientation_equivalences`)\n")
    L.append("| category | model | `symmetry` | `symmetry_near` |\n|---|---|---|---|")
    for k, r in mr:
        L.append(f"| {r['category']} | {r['model']} | `{' '.join(r['symmetry'])}` | `{' '.join(r['symmetry_near']) or '-'}` |")
    L.append(f"\n`mental_rotation` targets with only a `symmetry_near` entry (review before use): {len(mr_near)}: " +
             ", ".join(f"`{r['category']}/{r['model']}` ({' '.join(r['symmetry_near'])})" for k, r in mr_near) + "\n")
    # symmetric_180 disagreements
    L.append("## `symmetric_180` (alignment targets) vs the survey\n")
    L.append("The half turn that matters is about the alignment axis: the wall normal = the frame's front axis for the wall subtasks, the\n"
             "vertical (`frame_up`, else the canonical `z`) for `alignment-axial` (the object is yawed on the table). `symmetric_180` is only\n"
             "set for `alignment-world_horizontal` targets in the selection sheet (empty = the reviewer did not call the object symmetric),\n"
             "so every axial / vertical row below is a symmetry the reviewers did not record rather than a contradiction.\n")
    L.append("| subtask | category | model | half-turn axis | `symmetric_180` | survey half turn | survey `symmetry` | `symmetry_near` |\n|---|---|---|---|---|---|---|---|")
    n_dis = 0
    n_dis_flagged = 0
    for h in alignment_half_turns(icl, out):
        if h["disagree"]:
            rec = out[h["key"]]
            n_dis += 1
            n_dis_flagged += int(h["flag"])
            L.append(f"| {h['subtask']} | {h['category']} | {h['model']} | {h['axis_note']} | {h['symmetric_180'] or '-'} | "
                     f"{'yes' if h['has'] else ('near' if h['near'] else 'no')} | `{' '.join(rec['symmetry'])}` | `{' '.join(rec['symmetry_near']) or '-'}` |")
    L.append(f"\n{n_dis} alignment-target rows disagree ({n_dis_flagged} flagged `symmetric_180 = 1` without a confirmed survey half turn, "
             f"{n_dis - n_dis_flagged} with a survey half turn the sheet does not record).\n")
    # failures / review
    L.append("## Loader / render failures\n")
    L.append(f"Geometry stage: {len(failures)} failures; texture stage: {len(tex_failures)} failures.\n")
    for k, e in failures + tex_failures:
        L.append(f"- `{k}`: {e[:200]}")
    review = [(k, r) for k, r in out.items() if r["symmetry_near"]]
    L.append(f"\n## Review list ({len(review)} models with a `symmetry_near` entry)\n")
    L.append("| model | `symmetry` | `symmetry_near` | why | sheet |\n|---|---|---|---|---|")
    for k, r in review:
        why = []
        for t in r["symmetry_near"]:
            s = r["scores"].get(t, {})
            gcls = "geom exact" if t in r["geom_exact_raw"] else "geom near"
            why.append(f"{t}: {gcls} (mean {s.get('geom', {}).get('mean', 0) * 100:.2f}% frac2 {s.get('geom', {}).get('frac2', 0) * 100:.1f}%), "
                       f"tex {s.get('tex_class', 'missing')}" + (f" (MAD {s['tex']['mad'] * 255:.1f})" if "tex" in s else ""))
        L.append(f"| `{k}` | `{' '.join(r['symmetry'])}` | `{' '.join(r['symmetry_near'])}` | {'; '.join(why)} | {r['render'] or '-'} |")
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:40]))
    print(f"... report written to {os.path.join(args.out, 'report.md')}; symmetry.json / symmetry.csv: {n} models")


# ----------------------------------------------------------------------------------------------- diff

def stage_diff(args):
    """Compare the current ``symmetry.json`` with a previous one (``--previous``, default ``superseded/v1/symmetry.json``):
    texture-class flips per tag, final-verdict changes, models that gained a borderline tag, and the agreement of the
    current texture classes with the reviewer verdicts in ``review/*.json``. Writes ``report_diff.md``."""
    import glob as _glob

    prev_path = args.previous or os.path.join(args.out, "superseded", "v1", "symmetry.json")
    prev = json.load(open(prev_path))
    cur = json.load(open(os.path.join(args.out, "symmetry.json")))
    flips = collections.Counter()  # (tag, old class, new class)
    flip_rows = []
    sym_lost = collections.Counter()
    sym_gained = collections.Counter()
    changed = []
    newly_border, newly_none_to_border = [], []
    for key, r in cur.items():
        p = prev.get(key)
        if p is None:
            continue
        for variant in ("base", "scaled"):
            ps = p["scores"] if variant == "base" else (p.get("scaled") or {}).get("scores", {})
            cs = r["scores"] if variant == "base" else (r.get("scaled") or {}).get("scores", {})
            for t, s in cs.items():
                old, new = (ps.get(t) or {}).get("tex_class"), s.get("tex_class")
                if old and new and old != new:
                    flips[(t, old, new)] += 1
                    if old == "preserved":
                        flip_rows.append((key + ("" if variant == "base" else "__scaled"), t, old, new, s.get("tex", {})))
        if r["symmetry"] != p["symmetry"] or r["symmetry_near"] != p["symmetry_near"]:
            changed.append((key, p["symmetry"], p["symmetry_near"], r["symmetry"], r["symmetry_near"]))
            for t in set(S.expand_tags(p["symmetry"])) - set(S.expand_tags(r["symmetry"])):
                if t in p["symmetry"]:
                    sym_lost[t] += 1
            for t in set(r["symmetry"]) - set(S.expand_tags(p["symmetry"])):
                if t != "none":
                    sym_gained[t] += 1
        new_near = [t for t in r["symmetry_near"] if t not in p["symmetry_near"]]
        if new_near:
            newly_border.append((key, new_near))
            if p["symmetry"] == ["none"] and not p["symmetry_near"]:
                newly_none_to_border.append((key, new_near))
    # agreement with the reviewer verdicts (their tags are judged on the current texture class)
    agree = collections.Counter()
    disagree_rows = []
    for f in sorted(_glob.glob(os.path.join(args.out, "review", "*.json"))):
        bid = os.path.splitext(os.path.basename(f))[0]
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        for key, tags in d.items():
            base_key = key[: -len("__scaled")] if key.endswith("__scaled") else key
            r = cur.get(base_key)
            if r is None or not isinstance(tags, dict):
                continue
            scores = r["scores"] if not key.endswith("__scaled") else (r.get("scaled") or {}).get("scores", {})
            for t, v in tags.items():
                verdict = (v or {}).get("verdict") if isinstance(v, dict) else None
                cls = (scores.get(t) or {}).get("tex_class")
                if verdict not in ("preserved", "broken") or cls is None:
                    agree["n/a (unsure or not scored)"] += 1
                    continue
                if cls == "borderline":
                    agree[f"reviewer {verdict} / survey borderline"] += 1
                elif cls == verdict:
                    agree[f"agree ({verdict})"] += 1
                else:
                    agree[f"DISAGREE reviewer {verdict} / survey {cls}"] += 1
                    disagree_rows.append((bid, key, t, verdict, cls, (v or {}).get("note", "")[:90]))
    L = [f"# Survey re-run diff: {os.path.relpath(prev_path, REPO)} -> current\n",
         "## Texture-class flips per (model, tag) with a texture score in both runs\n",
         "| tag | preserved -> broken | preserved -> borderline | borderline -> broken | borderline -> preserved | broken -> borderline | broken -> preserved |\n|---|---|---|---|---|---|---|"]
    for t in S.TAG_ORDER:
        row = [flips[(t, "preserved", "broken")], flips[(t, "preserved", "borderline")], flips[(t, "borderline", "broken")],
               flips[(t, "borderline", "preserved")], flips[(t, "broken", "borderline")], flips[(t, "broken", "preserved")]]
        if any(row):
            L.append(f"| `{t}` | " + " | ".join(str(v) for v in row) + " |")
    tot = collections.Counter()
    for (t, o, n), c in flips.items():
        tot[(o, n)] += c
    L.append("\nTotals: " + ", ".join(f"{o} -> {n}: {c}" for (o, n), c in sorted(tot.items())) + "\n")
    L.append(f"## Final verdicts\n\n{len(changed)} models changed `symmetry` / `symmetry_near`. Tags removed from `symmetry`: "
             + (", ".join(f"`{t}` {c}" for t, c in sym_lost.most_common()) or "none") + ". Tags added to `symmetry`: "
             + (", ".join(f"`{t}` {c}" for t, c in sym_gained.most_common()) or "none") + ".\n")
    L.append(f"{len(newly_border)} models gained a `symmetry_near` tag ({len(newly_none_to_border)} of them were `none` without a near entry before).\n")
    L.append("| key | before `symmetry` / `symmetry_near` | now `symmetry` / `symmetry_near` |\n|---|---|---|")
    for key, s0, n0, s1, n1 in changed:
        L.append(f"| `{key}` | `{' '.join(s0)}` / `{' '.join(n0) or '-'}` | `{' '.join(s1)}` / `{' '.join(n1) or '-'}` |")
    L.append(f"\n## (model, tag) that were texture-preserved and are not any more ({len(flip_rows)})\n")
    L.append("| key | tag | now | MAD | p95 | p99 | blob (x 6x6 mm) | blob mean /255 | view |\n|---|---|---|---|---|---|---|---|---|")
    for key, t, old, new, tex in flip_rows:
        L.append(f"| `{key}` | `{t}` | {new} | {tex.get('mad', 0) * 255:.1f} | {tex.get('p95', 0) * 255:.1f} | {tex.get('p99', 0) * 255:.0f} | "
                 f"{tex.get('blob_px', 0) / max(tex.get('n6', 1), 1):.1f} | {tex.get('blob_d', 0) * 255:.0f} | {tex.get('blob_view', '')} |")
    L.append("\n## Agreement with the reviewer verdicts (review/*.json, judged on the current texture class)\n")
    for k, c in sorted(agree.items()):
        L.append(f"- {k}: {c}")
    L.append("\n| batch | key | tag | reviewer | survey | note |\n|---|---|---|---|---|---|")
    for bid, key, t, verdict, cls, note in disagree_rows:
        L.append(f"| {bid} | `{key}` | `{t}` | {verdict} | {cls} | {note} |")
    with open(os.path.join(args.out, "report_diff.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:60]))
    print(f"... written to {os.path.join(args.out, 'report_diff.md')}")


# ----------------------------------------------------------------------------------------------- main

def load_models(args, full: bool = False) -> List[dict]:
    with open(os.path.join(args.out, "models.json")) as f:
        models = json.load(f)
    if getattr(args, "only", None):
        keep = set(args.only.split(","))
        models = [m for m in models if f"{m['category']}/{m['model']}" in keep or m["category"] in keep]
    if getattr(args, "limit", None):
        models = models[: args.limit]
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["models", "geom", "texture", "report", "diff"])
    ap.add_argument("--previous", default=None, help="diff stage: the previous symmetry.json (default superseded/v1/symmetry.json)")
    ap.add_argument("--icl", default=DEFAULT_ICL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma separated category or category/model filters")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--shard", default=None, help="texture stage: K/N (this process renders every N-th model starting at K)")
    ap.add_argument("--tags-file", default=None,
                    help="texture stage: json {key: {base: [tags], scaled: [tags]}} of extra rotations to render and to put first on the "
                         "sheet (review round); only these models are (re-)rendered")
    ap.add_argument("--layout-file", default=None,
                    help="texture stage: json {key: {base|scaled: {first: [tags], pin: {tag: rotation}}}} - keep the sheet row order and "
                         "displayed rotations a running review batch was issued with (prepare_review.py --layout writes it)")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    {"models": stage_models, "geom": stage_geom, "texture": stage_texture, "report": stage_report, "diff": stage_diff}[args.stage](args)


if __name__ == "__main__":
    main()
