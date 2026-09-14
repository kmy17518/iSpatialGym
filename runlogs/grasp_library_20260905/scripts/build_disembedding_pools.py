#!/usr/bin/env python3
"""Disembedding distractor pools per target category (same-category pool, other-category pool, look-alike groups).

Builds, from the grasp-library selection data of the 39 disembedding categories:

1. ``select/disembedding_distractor_pools.json`` - the full record: per category the pooled models (grasp key, bbox,
   selection score), the categories of its other-category pool with the family they come from, the look-alike
   groups with the reviewer's reason, and the excluded models.
2. The ``distractors`` section of ``disembedding-0`` in ``ispatialgym/assets/task_specs/task_custom_lists.json``
   (spliced textually so the hand formatting of the rest of the file is kept; idempotent).
3. The disembedding block of ``select/task_object_selection.csv`` (+ .xlsx / .html twins): the ``pair`` column of every
   target row summarises the two pools of that model, look-alike members are flagged in the ``duplicates`` column, and
   the ``same_category_pool`` / ``distractor_pool`` role blocks list one row per target category.

Rules (see ispatialgym/docs/instance_sampling_spec.md section 5.3):

* same-category pool of category C = every disembedding model of C with >= 1 AG-valid grasp (the target rows of the
  sheet) except assets judged unusable (visual score V <= 1);
* other-category pool of C = the same-category pools of the categories that share a visual *family* with C
  (FAMILIES below: cartons, packs, bottles, vessels, small_items; a category may belong to several families);
* look-alike groups = models that cannot be told apart from the robot's initial view (~0.9 m, 459 px focal length:
  objects are 20-130 px tall): the exact-duplicate groups of select/duplicates.md plus the groups of LOOKALIKE_REVIEW,
  judged on the duplicate-cluster renders and the grasp strips (V2 / V3 tiles). When the target is a member, the other
  members are never drawn as distractors, from either pool.

Usage: python3 build_disembedding_pools.py [--no-spec] [--no-sheet] [--no-render]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(LIB, "..", ".."))
SEL = os.path.join(LIB, "select")
SPEC = os.path.join(REPO, "ispatialgym", "assets", "task_specs", "task_custom_lists.json")
CSV_PATH = os.path.join(SEL, "task_object_selection.csv")
POOLS_JSON = os.path.join(SEL, "disembedding_distractor_pools.json")
COLUMNS = ["task", "role", "category", "model", "selection", "pair", "duplicates", "note"]

#: models excluded from every pool (category, model) -> reason
EXCLUDED = {
    ("mouse", "oydobn"): "visual score V=1: unscaled 31x20x11 mm black blob, unusable as clutter",
}

#: visual families: the other-category pool of a category is the union of the other members of its families
FAMILIES = OrderedDict([
    ("cartons", ["box_of_cereal", "box_of_rice", "box_of_cane_sugar", "box_of_shampoo", "box_of_oatmeal", "carton_of_milk",
                 "box_of_baking_mix", "box_of_granola_bars", "box_of_butter", "box_of_cookies", "bottle_of_milk"]),
    ("packs", ["pack_of_cigarettes", "video_game", "notebook", "box_of_tissues", "box_of_sanitary_napkins", "box_of_butter",
               "box_of_cookies", "box_of_granola_bars", "pencil_case", "bar_soap"]),
    ("bottles", ["bottle_of_wine", "bottle_of_beer", "bottle_of_pop", "bottle_of_medicine", "bottle_of_milk", "bottle_of_shampoo",
                 "soap_dispenser", "conditioner_dispenser", "can_of_soda", "jar_of_bath_salt"]),
    ("vessels", ["mug", "beaker", "box_of_yogurt", "pencil_holder", "can_of_soda", "jar_of_bath_salt"]),
    ("small_items", ["teaspoon", "tablefork", "cookie_cutter", "tortilla_chips", "chili", "cellulose_tape", "mouse", "bar_soap",
                     "toy_dice", "video_game", "pack_of_cigarettes", "box_of_yogurt", "pencil_case"]),
])
FAMILY_DESC = {
    "cartons": "standing printed boxes and cartons",
    "packs": "flat / small printed packs and cases",
    "bottles": "upright bottles, cans and pump dispensers",
    "vessels": "open cups, tumblers and holders",
    "small_items": "small tabletop items (cutlery, chips, blocks, rolls, mice, ...)",
}

#: look-alike groups judged from the renders (category, models, reason); exact-duplicate groups of
#: select/duplicates/verdicts.json are added automatically. Cross-category groups use "<category>/<model>" entries.
LOOKALIKE_REVIEW = [
    ("beaker", ["jdijek", "rhohgs", "uobdoq", "exzsal"], "plain white / pale grey tapered tumblers 81-102 mm; only the shade differs"),
    ("beaker", ["effbnc", "sfvswx"], "large faceted grey beakers 146 mm"),
    ("bottle_of_beer", ["crfcwo", "meqliv", "zbsxro"], "same stubby green bottle with gold cap; the label (teal vs yellow) is on one side only"),
    ("bottle_of_beer", ["fgzjnb", "gxxbhh", "saslsh"], "dark green long-necks 217-254 mm with gold cap; labels differ on one side only"),
    ("bottle_of_beer", ["kqskmv", "qepxvl", "rdnopv"], "brown long-necks 205-254 mm with tan crest labels"),
    ("bottle_of_medicine", ["kasbsy", "zfbnjh", "qqsukh"], "white 50 mm pill bottles with blue / dark blue / light blue labels"),
    ("bottle_of_medicine", ["egondf", "syfpak"], "white 50 mm pill bottles with maroon / red labels"),
    ("bottle_of_milk", ["ddvsgl", "utfsmp"], "same carton mesh; white with red cap and blurry print vs plain cream"),
    ("bottle_of_pop", ["mhzttm", "nyupqh", "sevoto"], "dark cola bottles with white cap and light grey wrap label"),
    ("bottle_of_pop", ["vfjhav", "xoldze"], "dark cola bottles with red labels"),
    ("bottle_of_shampoo", ["lvvkhx", "stjjjm"], "same white cylinder with silver cap; only the label accent colour differs"),
    ("bottle_of_wine", ["lacaoa", "njoeiz", "yaxufs"], "dark champagne bottles with cream / gold foil; only the back labels differ"),
    ("bottle_of_wine", ["oslewp", "ynfvhg"], "dark 310 mm bottles with a white front label"),
    ("box_of_baking_mix", ["fmewsz", "ltccey", "luwlfd"], "same white carton with a red top band; only the print text differs"),
    ("box_of_baking_mix", ["fnbgwd", "lgusnr"], "same white carton with a tan / orange band"),
    ("box_of_baking_mix", ["uhvkxj", "xcxrek"], "same white carton with a yellow band"),
    ("box_of_cereal", ["sgirte", "wdkcbu"], "black Dorset boxes; only the accent colour of the small print differs"),
    ("box_of_cereal", ["ihedod", "yorray"], "cream / kraft Dorset boxes 190-200 mm"),
    ("box_of_cereal", ["iucshm", "qirvmd"], "dark brown cocoa boxes 220-273 mm with white text"),
    ("box_of_cookies", ["nirfva", "nnfisz", "sjcnoq"], "orange standing boxes 206-224 mm"),
    ("box_of_sanitary_napkins", ["acaivd", "ykpeja", "cmnpuv"], "pink soft packs 100-109 mm"),
    ("box_of_sanitary_napkins", ["mxdefq", "tkrzkb", "skltgp"], "white soft packs with blue / purple print"),
    ("box_of_tissues", ["ntbrtz", "uglbjc"], "same white soft pack; purple vs blue top print"),
    ("can_of_soda", ["opivig", "wbrrad"], "red cans 114-150 mm"),
    ("can_of_soda", ["evcxlr", "xmjfcg"], "beige / gold octagonal cans 130-148 mm"),
    ("can_of_soda", ["bfrzvk", "lugwcz"], "blue cans 125-127 mm"),
    ("carton_of_milk", ["kszoro", "xmugpm"], "white gable cartons with blue / purple print, 181 vs 218 mm"),
    ("cellulose_tape", ["kavsnx", "sklkyc"], "white tape rolls 50-62 mm"),
    ("chili", ["pbbkpz", "xhbpqh"], "tiny red chilies 39-43 mm"),
    ("conditioner_dispenser", ["udeflt", "uncljo"], "white pump bottles 187-190 mm with a ball head"),
    ("cookie_cutter", ["fvxiun", "jpscvj", "lqrfzo"], "black 56 mm cutters; outline (heart / star / flower) and rim colour are ~28 px in the init view"),
    ("mouse", ["cduwda", "fjljxp", "ihpnjx", "pljjtu"], "white / light grey oval mice 106-124 mm (fjljxp = ihpnjx exact)"),
    ("mug", ["ehnmxj", "fapsrj", "lgxhsc", "ntgftr"], "plain white mugs 70-96 mm tall (ehnmxj = fapsrj exact)"),
    ("mug", ["waqrdy", "yxaapv"], "tall plain white latte mugs 124-135 mm"),
    ("notebook", ["dlhxxl", "zkpswz", "zkmmnf"], "black / dark grey notebooks 121-127 mm"),
    ("pack_of_cigarettes", ["ajphvf", "wpmpcf"], "yellow / gold packs"),
    ("pack_of_cigarettes", ["dthkix", "ppbmak"], "red packs with gold detail"),
    ("pack_of_cigarettes", ["ipmejr", "qldhnm"], "white Diamond packs with a blue circle"),
    ("pencil_case", ["cqxkgl", "otcmih"], "blue soft cases 156-197 mm"),
    ("soap_dispenser", ["zwudcz", "ijxawq"], "white / cream pump dispensers 129-139 mm"),
    ("tablefork", ["flexrc", "pqrtkc", "yuihyg"], "plain matte grey / pale metal forks 201-233 mm"),
    ("tablefork", ["hktwlm", "uodzrq"], "chrome forks 183-189 mm"),
    ("teaspoon", ["qdepdj", "nxddfp", "prbhag", "piqabk", "vgbcmo"], "plain grey spoons 123-147 mm (qdepdj / nxddfp / prbhag: same shape)"),
    ("teaspoon", ["bjzgmj", "eilfij"], "chrome serving-size spoons 187-250 mm"),
    ("tortilla_chips", ["aiohfb", "herjym", "mhbnsy"], "yellow chips 74-84 mm"),
    ("toy_dice", ["akguod", "bjrxkj", "cmqtch", "eqwazv", "gcevml", "hbkqqp", "ievnsq", "kmagva", "krpdji", "mqvaaa", "rbzyrv", "xfqqhh"],
     "41 mm pale-wood alphabet blocks: engraving and frame colour are ~20 px in the init view (all blocks look alike)"),
    ("video_game", ["ceaxsd", "yyvkel"], "white Wii cases 85-87 mm"),
    ("video_game", ["ckugyi", "quzjuo"], "dark 3DS cases 59-60 mm"),
    ("video_game", ["coiafo", "ppnzqm"], "light 3DS cases 59-63 mm"),
    ("video_game", ["qsyyql", "rzaxlj", "veodkq"], "miniature cases 18-42 mm"),
    (None, ["box_of_butter/mfjuil", "box_of_cookies/apjokz"], "identical mesh shared by two categories, both blue boxes"),
]

IGNORED_FLAGS = {"one_finger"}


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------


def load_selection():
    groups, scores, picks = {}, {}, {}
    for p in sorted(glob.glob(os.path.join(SEL, "groups", "disembedding__*.json"))):
        with open(p) as f:
            g = json.load(f)
        groups[g["category"]] = g
        gid = os.path.basename(p)[:-5]
        with open(os.path.join(SEL, "out", gid + ".json")) as f:
            scores[g["category"]] = {c["model"]: c for c in json.load(f)["candidates"]}
    with open(os.path.join(SEL, "test_instances.json")) as f:
        for t in json.load(f):
            if t["task"] == "disembedding":
                picks[t["category"]] = t
    with open(os.path.join(SEL, "duplicates", "verdicts.json")) as f:
        verdicts = json.load(f)
    return groups, scores, picks, verdicts


def build(groups, scores, picks, verdicts):
    cats = sorted(groups)
    # 1. pools
    pools = OrderedDict()
    excluded = []
    for cat in cats:
        models = []
        for c in sorted(groups[cat]["candidates"], key=lambda c: c["model"]):
            reason = EXCLUDED.get((cat, c["model"]))
            if reason:
                excluded.append({"category": cat, "model": c["model"], "reason": reason})
                continue
            s = scores[cat].get(c["model"], {})
            models.append({"model": c["model"], "grasp_key": c["key"], "bbox_mm": c["bbox_mm"], "score": s.get("score"), "V": s.get("V"),
                           "n_valid": c["n_valid"], "n_clean": c["n_clean"], "selection": (
                               "pick 1" if picks[cat]["pick1"] == c["model"] else "pick 2" if picks[cat]["pick2"] == c["model"]
                               else "alternate" if picks[cat]["alternate"] == c["model"] else "candidate")})
        fams = [f for f, members in FAMILIES.items() if cat in members]
        if not fams:
            raise SystemExit(f"{cat} belongs to no family")
        others = OrderedDict()
        for f in fams:
            for other in FAMILIES[f]:
                if other != cat and other in groups:
                    others.setdefault(other, []).append(f)
        pools[cat] = {"same_category": models, "families": fams, "other_categories": others}
    for f, members in FAMILIES.items():
        for m in members:
            if m not in groups:
                raise SystemExit(f"family {f}: {m} is not a disembedding category")
    # 2. look-alike groups: reviewed groups + exact-duplicate groups not already covered
    pooled = {(cat, m["model"]) for cat, p in pools.items() for m in p["same_category"]}
    lookalike = []
    for cat, models, reason in LOOKALIKE_REVIEW:
        members = [tuple(m.split("/")) if "/" in m else (cat, m) for m in models]
        for cm in members:
            if cm not in pooled:
                raise SystemExit(f"look-alike member {cm} is not pooled")
        lookalike.append({"members": [f"{c}/{m}" for c, m in members], "reason": reason, "source": "visual review (cluster renders / grasp strips)"})
    for c in verdicts:
        for grp in c["exact_groups"]:
            members = [(c["category"], m) for m in grp if (c["category"], m) in pooled]
            if len(members) < 2:
                continue
            if any(set(f"{a}/{b}" for a, b in members) <= set(g["members"]) for g in lookalike):
                continue
            lookalike.append({"members": [f"{a}/{b}" for a, b in members], "reason": f"exact duplicates (cluster {c['id']})",
                              "source": "select/duplicates.md"})
    return pools, lookalike, excluded


def spec_section(pools, lookalike):
    return OrderedDict([
        ("_comment", "Per-category distractor pools for disembedding targets (built by runlogs/grasp_library_20260905/scripts/"
                     "build_disembedding_pools.py; details in select/disembedding_distractor_pools.json). For a target C/m: "
                     "same-category pool = pools[C].same_category minus m and minus its look-alikes; other-category pool = the "
                     "same_category lists of pools[C].other_categories minus look-alikes. Look-alike groups are models that cannot "
                     "be told apart from the robot's initial view; members are never drawn as each other's distractors."),
        ("families", OrderedDict((f, list(m)) for f, m in FAMILIES.items())),
        ("lookalike_groups", [g["members"] for g in lookalike]),
        ("pools", OrderedDict((cat, OrderedDict([("same_category", [m["model"] for m in p["same_category"]]),
                                                 ("other_categories", list(p["other_categories"]))]))
                              for cat, p in pools.items())),
    ])


def write_pools_json(pools, lookalike, excluded):
    doc = OrderedDict([
        ("_comment", "Disembedding distractor pools per target category. Source: runlogs/grasp_library_20260905/select "
                     "(groups/disembedding__*.json, out/, test_instances.json, duplicates/verdicts.json) + the look-alike review in "
                     "scripts/build_disembedding_pools.py. The compact form is the 'distractors' section of disembedding-0 in "
                     "ispatialgym/assets/task_specs/task_custom_lists.json."),
        ("rules", {
            "same_category_pool": "every disembedding model of the category with >= 1 AG-valid grasp except 'excluded' (V <= 1); "
                                  "the target itself and its look-alikes are removed at sampling time",
            "other_category_pool": "same_category pools of the other members of the category's families (minus look-alikes of the target)",
            "lookalike_groups": "models indistinguishable from the robot's initial view (~0.9 m, objects 20-130 px): exact duplicates and "
                                "same-shape variants whose only difference is a small print / label or a subtle shade; never drawn as "
                                "each other's distractors",
        }),
        ("families", OrderedDict((f, {"description": FAMILY_DESC[f], "categories": list(m)}) for f, m in FAMILIES.items())),
        ("lookalike_groups", lookalike),
        ("excluded", excluded),
        ("pools", pools),
    ])
    with open(POOLS_JSON, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {POOLS_JSON}: {len(pools)} categories, {sum(len(p['same_category']) for p in pools.values())} pooled models, "
          f"{len(lookalike)} look-alike groups, {len(excluded)} excluded")


# ----------------------------------------------------------------------------------------------------------------
# spec splice
# ----------------------------------------------------------------------------------------------------------------


def format_section(section) -> str:
    """Compact hand-style JSON: one line per family / look-alike group / category pool."""
    ind = "    "
    lines = [f'{ind}"distractors": {{']
    lines.append(f'{ind}  "_comment": {json.dumps(section["_comment"])},')
    lines.append(f'{ind}  "families": {{')
    fams = list(section["families"].items())
    for i, (f, cats) in enumerate(fams):
        lines.append(f'{ind}    {json.dumps(f)}: {json.dumps(cats)}' + ("," if i < len(fams) - 1 else ""))
    lines.append(f'{ind}  }},')
    lines.append(f'{ind}  "lookalike_groups": [')
    grps = section["lookalike_groups"]
    for i, g in enumerate(grps):
        lines.append(f'{ind}    {json.dumps(g)}' + ("," if i < len(grps) - 1 else ""))
    lines.append(f'{ind}  ],')
    lines.append(f'{ind}  "pools": {{')
    pools = list(section["pools"].items())
    for i, (cat, body) in enumerate(pools):
        lines.append(f'{ind}    {json.dumps(cat)}: {json.dumps(body)}' + ("," if i < len(pools) - 1 else ""))
    lines.append(f'{ind}  }}')
    lines.append(f"{ind}}},")
    return "\n".join(lines) + "\n"


def splice_spec(section):
    with open(SPEC) as f:
        text = f.read()
    start = text.index('"disembedding-0": {')
    end = text.index('\n  "configuration_matching-0"', start)
    entry = text[start:end]
    block = format_section(section)
    # replace an existing block (from '    "distractors": {' to the line that closes it) or insert before "generation"
    m = re.search(r'\n {4}"distractors": \{\n.*?\n {4}\},\n', entry, flags=re.S)
    if m:
        entry = entry[: m.start() + 1] + block + entry[m.end():]
    else:
        g = entry.index('\n    "generation": {')
        entry = entry[: g + 1] + block + entry[g + 1:]
    text = text[:start] + entry + text[end:]
    json.loads(text)  # must still be valid JSON
    with open(SPEC, "w") as f:
        f.write(text)
    print(f"spliced the distractors section into {SPEC}")


# ----------------------------------------------------------------------------------------------------------------
# sheet
# ----------------------------------------------------------------------------------------------------------------


def dup_maps(verdicts):
    """(category, model) -> other exact-duplicate models; geometry-family id (for the 'M (N)' counts of the sheet)."""
    exact, family = {}, {}
    for c in verdicts:
        for m in c["models"]:
            family[(c["category"], m)] = c["id"]
        for g in c["exact_groups"]:
            for m in g:
                exact[(c["category"], m)] = sorted(x for x in g if x != m)
    return exact, family


def counts(models, exact, family):
    """'M (N)': M distinct geometries (every flagged cluster once), N distinct models (exact duplicates once)."""
    def canon(cm):
        ex = exact.get(cm)
        return (cm[0], min([cm[1]] + ex)) if ex else cm

    unique = {canon(cm) for cm in models}
    geoms = {(cm[0], family.get(cm, canon(cm)[1])) for cm in models}
    return f"{len(geoms)} ({len(unique)})"


def update_sheet(pools, lookalike, excluded, verdicts, render_twins=True):
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    exact, family = dup_maps(verdicts)
    la_of = defaultdict(list)  # (cat, model) -> [(other members, reason)]
    for g in lookalike:
        members = [tuple(m.split("/")) for m in g["members"]]
        for cm in members:
            la_of[cm].append(([f"{c}/{m}" if c != cm[0] else m for c, m in members if (c, m) != cm], g["reason"]))
    pooled = {(cat, m["model"]) for cat, p in pools.items() for m in p["same_category"]}
    excl = {(e["category"], e["model"]): e["reason"] for e in excluded}

    def other_pool(cat):
        return [(o, [m["model"] for m in pools[o]["same_category"]]) for o in pools[cat]["other_categories"]]

    def pair_text(cat, model):
        skip = {cm for others, _ in la_of[(cat, model)] for cm in others}
        same = [m["model"] for m in pools[cat]["same_category"] if m["model"] != model and m["model"] not in skip]
        n_same_all = len(pools[cat]["same_category"]) - 1
        excl_txt = f" ({n_same_all - len(same)} look-alike(s) excluded)" if n_same_all - len(same) else ""
        others = other_pool(cat)
        n_other = sum(len(ms) for _, ms in others) - sum(1 for o, ms in others for m in ms if f"{o}/{m}" in skip)
        fams = "/".join(pools[cat]["families"])
        return (f"distractors: same-category {len(same)}{excl_txt}; other-category {n_other} models from {len(others)} "
                f"categories (families {fams})")

    def dup_text(cat, model, old):
        """'LOOK-ALIKE with ... | <original duplicates text>' (idempotent: a previous flag is replaced)."""
        if " | " in old:
            old = old.split(" | ", 1)[1]
        elif old.startswith("LOOK-ALIKE"):
            old = ""
        flags = [f"LOOK-ALIKE with {', '.join(others)} (never each other's distractor; {reason})" for others, reason in la_of[(cat, model)]]
        if not flags:
            return old
        return "; ".join(flags) + (f" | {old}" if old else "")

    out, i, task, role, cat = [], 0, None, None, None
    n_target_rows = 0
    while i < len(rows):
        r = dict(rows[i])
        if r["task"]:
            task, role = r["task"], r["role"]
            cat = None
            if task == "disembedding-0" and role in ("same_category_pool", "distractor_pool"):
                # drop the old sub-rows and emit the per-category block
                j = i + 1
                while j < len(rows) and not rows[j]["task"]:
                    j += 1
                r["category"] = str(len(pools))
                r["model"] = counts(pooled, exact, family)
                r["duplicates"] = f"{len(lookalike)} look-alike groups ({sum(len(g['members']) for g in lookalike)} models) across the pools"
                if role == "same_category_pool":
                    r["note"] = (
                        "Same-category distractor pool per target category (spec: disembedding-0.distractors.pools.<category>.same_category; "
                        "generation.same_category_prob decides per template whether this pool or the other-category pool is drawn, "
                        "1..max_distractors unique models within the tray footprint budget). Pool = every disembedding model of the category "
                        "with >= 1 AG-valid grasp (the target rows above) except V<=1 assets ('excluded' in select/disembedding_distractor_pools.json). "
                        "At sampling time the target itself and every model in a LOOK-ALIKE group with the target are removed: look-alikes "
                        "(exact duplicates and same-shape variants that differ only by a small print / label or a subtle shade, judged from "
                        "the robot's initial view) are never each other's distractors. A target whose same-category pool is thereby empty "
                        "(single-model categories, toy_dice, cookie_cutter) falls back to the other-category pool. Rows: one per target "
                        "category (model cell = pool size; note = models and look-alike groups).")
                else:
                    r["note"] = (
                        "Other-category distractor pool per target category (spec: disembedding-0.distractors.pools.<category>.other_categories -> "
                        "their same_category lists). Categories are grouped into visual families - "
                        + "; ".join(f"{f}: {FAMILY_DESC[f]}" for f in FAMILIES) +
                        " - and a category's other-category pool is the union of the other members of its families (a category may belong to "
                        "two families). Look-alikes of the target are removed here as well (cross-category group box_of_butter/mfjuil = "
                        "box_of_cookies/apjokz). Rows: one per target category (model cell = pool size; pair = families; note = categories "
                        "with their models). The sampler's footprint budget (distractor_area_fraction of the source tray) drops distractors "
                        "that do not fit next to the target.")
                out.append(r)
                for c in sorted(pools, key=lambda c: (-len(pools[c]["same_category"]), c)):
                    p = pools[c]
                    if role == "same_category_pool":
                        models = [m["model"] for m in p["same_category"]]
                        la = [g for g in lookalike if any(x.startswith(c + "/") for x in g["members"])]
                        la_txt = "; ".join("{" + ", ".join(x.split("/")[1] if x.startswith(c + "/") else x for x in g["members"]) + "}: " + g["reason"] for g in la)
                        ex = [f"{e['model']} ({e['reason']})" for e in excluded if e["category"] == c]
                        note = f"models: {', '.join(models)}"
                        if la:
                            note += f"; look-alike groups: {la_txt}"
                        if ex:
                            note += f"; excluded: {'; '.join(ex)}"
                        out.append({"task": "", "role": "", "category": c, "model": f"{len(models)} model" + ("s" if len(models) != 1 else ""), "selection": "spec",
                                    "pair": f"same-category distractors for {c} targets", "duplicates": f"{len(la)} look-alike group(s)" if la else "",
                                    "note": note})
                    else:
                        others = other_pool(c)
                        n = sum(len(ms) for _, ms in others)
                        note = "; ".join(f"{o} ({len(ms)}: {', '.join(ms)})" for o, ms in others)
                        out.append({"task": "", "role": "", "category": c, "model": f"{n} model" + ("s" if n != 1 else ""), "selection": "spec",
                                    "pair": f"other-category distractors for {c} targets (families {'/'.join(p['families'])})",
                                    "duplicates": "", "note": f"{len(others)} categories: {note}"})
                i = j
                continue
            if task == "disembedding-0" and role == "target":
                r["note"] = re.sub(r" Distractor pools:.*$", "", r["note"])
                r["note"] += (" Distractor pools: see the same_category_pool / distractor_pool rows below and the pair column of every "
                              "target row (same-category pool size after removing the target and its look-alikes; other-category pool "
                              "size and families). LOOK-ALIKE in the duplicates column = models that cannot be told apart from the robot's "
                              "initial view; they are never drawn as distractors of each other.")
            out.append(r)
            i += 1
            continue
        if task == "disembedding-0" and role == "target" and r["model"]:
            if r["category"] and not re.match(r"^\d+ \(\d+\)$", r["category"]):
                cat = r["category"]
            if cat is not None:
                if (cat, r["model"]) in pooled:
                    r["pair"] = pair_text(cat, r["model"])
                elif (cat, r["model"]) in excl:
                    r["pair"] = f"not in the distractor pools: {excl[(cat, r['model'])]}"
                r["duplicates"] = dup_text(cat, r["model"], r["duplicates"])
                n_target_rows += 1
        out.append(r)
        i += 1
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"updated {CSV_PATH}: {n_target_rows} disembedding target rows annotated, {len(out)} rows")
    if render_twins:
        sys.path.insert(0, os.path.join(REPO, "runlogs", "table_selection_20260911", "scripts"))
        from update_selection_csv import render  # xlsx / html twins with the shared styling

        render(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-spec", action="store_true", help="do not touch task_custom_lists.json")
    ap.add_argument("--no-sheet", action="store_true", help="do not touch task_object_selection.csv")
    ap.add_argument("--no-render", action="store_true", help="do not regenerate the xlsx / html twins")
    args = ap.parse_args()
    groups, scores, picks, verdicts = load_selection()
    pools, lookalike, excluded = build(groups, scores, picks, verdicts)
    write_pools_json(pools, lookalike, excluded)
    if not args.no_spec:
        splice_spec(spec_section(pools, lookalike))
    if not args.no_sheet:
        update_sheet(pools, lookalike, excluded, verdicts, render_twins=not args.no_render)


if __name__ == "__main__":
    main()
