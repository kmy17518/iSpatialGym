#!/usr/bin/env python3
"""Assemble the distractor pools (20 per target category) from the candidates, the head-camera render checks, the
look-alike review and the sampler validation.

Per target category ``C``:

1. **Same-category pool** - same-category candidates that are ready, fit the tray, rested stably in the render check,
   were not flagged unusable by the reviewers and produced a sampler instance as the distractor of the category's
   test target. Models are grouped into *look-alike groups* (reviewer groups from the head-camera grids, merged with
   the exact-duplicate groups of the geometry check); a group counts as ONE candidate. The groups are ranked (test
   picks first, then selection score, then AG-valid grasps, then untested models) and the first ``SAME_CAP`` (10)
   groups form the pool - every validated member of a selected group is listed, so that whichever member is the
   target, the other members are excluded through the group. Validated members of further groups are ``reserve``.
2. **Other-category pool** - ``20 - min(n_same_groups, SAME_CAP)`` models from the related categories
   (``candidates.json`` order: tier 1 first, categories interleaved), validated the same way, exact duplicates folded,
   at most ``PER_CATEGORY`` (3) models per category. Further validated candidates are ``reserve``.

If a category's target template could not be set up at all (sampler ``setup_failed`` for every candidate), the
render check alone decides (``validation = render_only``) and the category is flagged.

Outputs (in the run directory):
* ``distractor_pools.json`` - everything, with the evidence per model.
* ``distractors_section.json`` - the ``distractors`` section for ``disembedding-0`` in ``task_custom_lists.json``
  (sampler >= 1.4.0: ``pools.<category>.same_category`` / ``other_categories`` / ``other_models``, ``lookalike_groups``).
* ``distractor_pools.md`` - human summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))

POOL_SIZE = 20
SAME_CAP = 10
PER_CATEGORY = 3
MIN_PIXELS = 200


def rank_key(e):
    order = {"pick 1": 0, "pick 2": 1, "alternate": 2, "candidate": 3, "untested": 4, "excluded": 5}
    return (order.get(e.get("selection"), 6), -(e.get("score") or 0), -(e.get("ag_valid") or 0), e["model"])


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def load_json(path, default=None):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--pool-size", type=int, default=POOL_SIZE)
    ap.add_argument("--same-cap", type=int, default=SAME_CAP)
    ap.add_argument("--prelim", action="store_true", help="before the sampler validation: eligibility from the render check + review only; "
                                                          "also writes pairs.json (pool + reserve) for validate_distractors.py --pairs")
    ap.add_argument("--reserve-other", type=int, default=8, help="related-category reserves kept per category (validated / to validate)")
    args = ap.parse_args()
    root = args.root
    cands = load_json(os.path.join(root, "candidates.json"))
    renders = {}
    for p in glob.glob(os.path.join(root, "renders", "meta", "*.json")):
        r = load_json(p)
        renders[(r["category"], r["model"])] = r
    results = defaultdict(dict)  # target category -> (cat, model) -> result
    for p in glob.glob(os.path.join(root, "results", "*.json")):
        r = load_json(p)
        results[r["target_category"]][(r["category"], r["model"])] = r
    reviews = {}
    for p in glob.glob(os.path.join(root, "groups", "*.json")):
        r = load_json(p)
        reviews[r["category"]] = r
    # {"<target category>": {"<cat>/<model>": reason}} from the related-category review (review_contract_other.md, groups_other/)
    diff_excl_all = load_json(os.path.join(root, "diff_exclusions.json"), {})
    # target models of the test spec (look-alike groups in the spec section may name them even when not pooled)
    spec_targets = set()
    spec_path = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
    if os.path.exists(spec_path):
        entry = load_json(spec_path).get("disembedding-0", {})
        for cat_, ms in ((entry.get("roles") or {}).get("target") or {}).get("models", {}).items():
            spec_targets |= {(cat_, m) for m in ms}

    def render_ok(cat, model):
        r = renders.get((cat, model))
        if r is None:
            return False, "not rendered"
        if r.get("status") != "ok":
            return False, f"render {r.get('status')}: {str(r.get('error', ''))[:80]}"
        r0 = r["renders"].get("y0", {})
        if not r.get("stable"):
            return False, f"unstable on the tray (drift {r0.get('drift_m')} m, tilt {r0.get('tilt_deg')} deg, on_tray {r0.get('on_tray')})"
        px = r0.get("seg_pixels")
        if px is not None and int(px) < MIN_PIXELS:
            return False, f"only {px} px in the head camera"
        return True, f"{px if px is not None else 'seg label missing,'} px, tilt {r0.get('tilt_deg')} deg"

    def validation(cat_t, cat, model):
        r = results.get(cat_t, {}).get((cat, model))
        if r is None:
            return "missing", "no sampler result", None
        st = r.get("status")
        if st == "ok":
            inst = (r.get("instances") or [{}])[0]
            txt = (f"instance written with target {r.get('target_model')} ({'occluded' if r.get('occluded_variant') else 'unoccluded'}"
                   + (f", occlusion {r['occlusion_ratio']:.2f}" if r.get("occlusion_ratio") is not None else "") + ")"
                   + (f"; rejections before {r['rejections']}" if r.get("rejections") else "")
                   + ("; unoccluded variant only" if r.get("viable_unoccluded_only") else "")
                   + f"; footprint {r.get('footprint_fraction_with_target')} of the tray")
            return "ok", txt, r
        if st == "is_target":
            # the category's test target: it stood on the tray and was grasped / placed in every other validated pair
            return "ok", "the category's test target itself (target of every other validated pair of the category)", r
        rej = r.get("rejections") or {}
        if st == "no_instances" and (rej.get("goal_unstable", 0) + rej.get("goal_infeasible", 0)) > 0:
            # goal_unstable / goal_infeasible are raised by check_goal_feasible, i.e. AFTER settle_and_validate passed
            # with the distractor on the tray (settled, :init true, goal false, occlusion measured): the distractor is
            # fine, the TARGET's goal placement at the destination centre failed in every attempt
            return "init_ok", (f"init validated (settled with the target, :init true, occlusion ok) but the TARGET's goal check failed in every attempt "
                               f"(target-side: rejections {rej}); footprint {r.get('footprint_fraction_with_target')} of the tray"), r
        return st, f"{st}: rejections {rej} {str(r.get('error', ''))[:160]}", r

    out_cats = OrderedDict()
    section_pools = OrderedDict()
    lookalike_groups_all = []
    summary_rows = []
    for cat, tinfo in cands["targets"].items():
        same = [e for e in cands["same"][cat] if e["size_ok"] and e["ready"]]
        diff = [e for e in cands["diff"][cat] if e["size_ok"] and e["ready"]]
        res = results.get(cat, {})
        # the category's target template never set up on the test table: no pair could be run (the pick-1 model's
        # own ``is_target`` record does not count as a run)
        ran = [r for r in res.values() if r.get("status") not in ("is_target",)]
        setup_failed = bool(ran) and all(r.get("status") == "setup_failed" for r in ran)
        # ---- look-alike groups: reviewer groups + exact duplicates
        uf = UnionFind()
        reasons = {}
        rev = reviews.get(cat)
        for e in same:
            uf.find(e["model"])
            if e.get("exact_group"):
                for m in e["exact_group"]:
                    uf.union(e["model"], m)
        for g in (rev or {}).get("groups", []):
            ms = [m for m in g["members"] if m]
            for m in ms[1:]:
                uf.union(ms[0], m)
            for m in ms:
                reasons[m] = g.get("reason", "")
        unusable = {u["model"]: u["reason"] for u in (rev or {}).get("unusable", [])}
        groups = defaultdict(list)
        for e in same:
            groups[uf.find(e["model"])].append(e)
        ranked_groups = sorted(groups.values(), key=lambda g: min(rank_key(e) for e in g))
        gid_of = {}
        group_records = []
        for gi, g in enumerate(ranked_groups, start=1):
            gid = f"{cat}__g{gi}"
            for e in g:
                gid_of[e["model"]] = gid
            group_records.append({"id": gid, "members": sorted(e["model"] for e in g), "reason": next((reasons[e["model"]] for e in g if e["model"] in reasons), None)})
        # ---- per-model evidence (same)
        same_rows = []
        for e in same:
            rok, rtxt = render_ok(cat, e["model"])
            vst, vtxt, vres = validation(cat, cat, e["model"])
            usable = e["model"] not in unusable
            if args.prelim:
                ok = rok and usable
                vmode = "prelim"
            elif setup_failed:
                ok = rok and usable
                vmode = "render_only"
            else:
                ok = rok and usable and vst in ("ok", "init_ok")
                vmode = "sampler"
            same_rows.append({**{k: e[k] for k in ("category", "model", "bbox", "selection", "score", "ag_valid", "clean", "grasp_key", "exact_group", "geom_cluster", "reviewer")},
                              "group": gid_of[e["model"]], "render_ok": rok, "render": rtxt, "validation_status": vst, "validation": vtxt, "validation_mode": vmode,
                              "unusable": unusable.get(e["model"]), "eligible": ok,
                              "target_model": (vres or {}).get("target_model"), "occluded_variant": (vres or {}).get("occluded_variant"),
                              "occlusion_ratio": (vres or {}).get("occlusion_ratio")})
        # ---- select same-category groups
        eligible_groups = []
        for g in ranked_groups:
            members = [r for r in same_rows if r["group"] == gid_of[g[0]["model"]] and r["eligible"]]
            if members:
                eligible_groups.append((gid_of[g[0]["model"]], members))
        n_same_groups = len(eligible_groups)
        n_same_sel = min(n_same_groups, args.same_cap)
        pool_same, reserve_same = [], []
        for i, (gid, members) in enumerate(eligible_groups):
            for r in members:
                r["pool"] = "pool" if i < n_same_sel else "reserve"
                (pool_same if i < n_same_sel else reserve_same).append(r)
        # ---- other-category pool
        n_other = args.pool_size - n_same_sel
        other_rows, pool_other, reserve_other = [], [], []
        per_cat = defaultdict(int)
        seen_exact = set()
        for e in diff:
            rok, rtxt = render_ok(e["category"], e["model"])
            vst, vtxt, vres = validation(cat, e["category"], e["model"])
            excl = (diff_excl_all.get(cat) or {}).get(f"{e['category']}/{e['model']}")
            if args.prelim:
                ok = rok and not excl
                vmode = "prelim"
            elif setup_failed:
                ok = rok and not excl
                vmode = "render_only"
            else:
                ok = rok and not excl and vst in ("ok", "init_ok")
                vmode = "sampler"
            row = {**{k: e[k] for k in ("category", "model", "bbox", "selection", "score", "ag_valid", "clean", "grasp_key", "exact_group", "geom_cluster", "tier")},
                   "render_ok": rok, "render": rtxt, "validation_status": vst, "validation": vtxt, "validation_mode": vmode, "excluded": excl, "eligible": ok,
                   "target_model": (vres or {}).get("target_model"), "occluded_variant": (vres or {}).get("occluded_variant"), "occlusion_ratio": (vres or {}).get("occlusion_ratio")}
            other_rows.append(row)
            if not ok:
                continue
            ex = (e["category"], tuple(e["exact_group"]) if e.get("exact_group") else (e["model"],))
            if ex in seen_exact:
                row["pool"] = "duplicate"
                continue
            if per_cat[e["category"]] >= PER_CATEGORY:
                if len(reserve_other) < args.reserve_other:
                    row["pool"] = "reserve"
                    reserve_other.append(row)
                continue
            if len(pool_other) < n_other:
                row["pool"] = "pool"
                pool_other.append(row)
                per_cat[e["category"]] += 1
                seen_exact.add(ex)
            elif len(reserve_other) < args.reserve_other:
                row["pool"] = "reserve"
                reserve_other.append(row)
                seen_exact.add(ex)
        n_total = n_same_sel + len(pool_other)
        out_cats[cat] = {
            "target": tinfo, "template_setup_failed": setup_failed,
            "n_same_candidates": len(same), "n_same_groups_eligible": n_same_groups, "n_same_groups_selected": n_same_sel,
            "n_same_models_selected": len(pool_same), "n_other_selected": len(pool_other), "n_candidates_total": n_total,
            "short": n_total < args.pool_size,
            "lookalike_groups": [g for g in group_records if len(g["members"]) > 1],
            "same_category": same_rows, "other_category": other_rows,
        }
        # ---- spec section
        section_pools.setdefault(cat, {"same_category": [], "other_categories": [], "other_models": OrderedDict()})
        section_pools[cat]["same_category"] = [r["model"] for r in pool_same]
        om = OrderedDict()
        for r in pool_other:
            om.setdefault(r["category"], []).append(r["model"])
        section_pools[cat]["other_models"] = om
        for g in group_records:
            # the spec parser accepts look-alike members that are pooled or spec targets only
            pooled = {r["model"] for r in pool_same}
            members = sorted(m for m in g["members"] if m in pooled or (cat, m) in spec_targets)
            if len(members) > 1:
                lookalike_groups_all.append({"members": [f"{cat}/{m}" for m in members], "reason": g.get("reason") or "exact duplicates (geometry check)", "id": g["id"]})
        summary_rows.append((cat, n_same_groups, n_same_sel, len(pool_same), len(pool_other), n_total, setup_failed, len(reserve_same), len(reserve_other)))
    # other_models of non-target categories must have a pool entry too (spec rule: every pooled category has a pool)
    for cat, pool in list(section_pools.items()):
        for oc in pool["other_models"]:
            section_pools.setdefault(oc, {"same_category": [], "other_categories": [], "other_models": OrderedDict()})
    # lookalike members must be pooled somewhere or be targets: other_models members can be look-alikes across
    # categories only through exact-duplicate cross-category pairs, which the geometry check found once
    # (box_of_butter/mfjuil = box_of_cookies/apjokz); kept as separate models (different categories).
    section = {
        "_comment": ("Disembedding distractor pools per target category, built and validated by runlogs/disembedding_distractors_20260911 "
                     "(head-camera look-alike review from the init stance, related-category candidates, one sampler instance per "
                     "(target category, distractor) with validate_distractors.py). same_category = models of the category usable as "
                     "same-category distractors (every validated member of the selected look-alike groups); other_models = the "
                     "other-category pool (explicit models per related category); lookalike_groups = indistinguishable from the init view."),
        "pools": section_pools,
        "lookalike_groups": [{"members": g["members"], "reason": g["reason"]} for g in lookalike_groups_all],
    }
    with open(os.path.join(root, "distractor_pools.json"), "w") as f:
        json.dump({"pool_size": args.pool_size, "same_cap": args.same_cap, "per_category": PER_CATEGORY, "prelim": bool(args.prelim), "categories": out_cats}, f, indent=1)
    if args.prelim:
        # what the sampler validation has to cover: every pool / reserve member per target category
        pairs = {cat: [[r["category"], r["model"]] for r in d["same_category"] + d["other_category"] if r.get("pool") in ("pool", "reserve")]
                 for cat, d in out_cats.items()}
        with open(os.path.join(root, "pairs.json"), "w") as f:
            json.dump(pairs, f, indent=1)
        print(f"pairs.json: {sum(len(v) for v in pairs.values())} (target category, distractor) pairs to validate")
    with open(os.path.join(root, "distractors_section.json"), "w") as f:
        json.dump(section, f, indent=1)
    # ---- markdown
    lines = ["# Disembedding distractor pools (per target category)", "",
             f"Pool = {args.pool_size} candidate models per target category: same-category look-alike groups first (a group counts once, at most "
             f"{args.same_cap} groups), related-category models fill the rest (at most {PER_CATEGORY} per category). Every listed model rested stably on the "
             "source tray in the head-camera render check and produced one sampler instance as the distractor of the category's test target "
             "(`results/`). `reserve` = validated but beyond the pool size.", "",
             "| target category | same groups (eligible) | same groups in pool | same models in pool | other models | total | reserve same / other | note |",
             "|---|---|---|---|---|---|---|---|"]
    for cat, ng, ns, nm, no, nt, sf, rs, ro in summary_rows:
        note = "template setup FAILED: render-only evidence" if sf else ("SHORT" if nt < args.pool_size else "")
        lines.append(f"| {cat} | {ng} | {ns} | {nm} | {no} | {nt} | {rs} / {ro} | {note} |")
    lines.append("")
    for cat, d in out_cats.items():
        lines.append(f"## {cat} (test target {d['target']['pick1']})")
        if d["lookalike_groups"]:
            lines.append("Look-alike groups (init view): " + "; ".join("{" + ", ".join(g["members"]) + "}" + (f" - {g['reason']}" if g.get("reason") else "") for g in d["lookalike_groups"]))
        same_pool = [r for r in d["same_category"] if r.get("pool") == "pool"]
        lines.append("Same-category pool: " + (", ".join(f"{r['model']} [{r['group'].split('__')[1]}]" for r in same_pool) or "-"))
        other_pool = [r for r in d["other_category"] if r.get("pool") == "pool"]
        lines.append("Other-category pool: " + (", ".join(f"{r['category']}/{r['model']}" for r in other_pool) or "-"))
        dropped = [f"{r['category']}/{r['model']} ({r['unusable'] or (r['validation'] if not r['render_ok'] else r['validation'])[:60]})" for r in d["same_category"] if not r["eligible"]]
        if dropped:
            lines.append("Same-category candidates dropped: " + "; ".join(dropped))
        lines.append("")
    with open(os.path.join(root, "distractor_pools.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:5 + len(summary_rows)]))


if __name__ == "__main__":
    main()
