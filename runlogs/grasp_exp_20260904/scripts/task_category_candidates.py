"""Rank BEHAVIOR-1K categories per iSpatialGym task (20/19/19/9/9/4x4 categories) from category_table.json."""
import json, re, collections
import numpy as np

OUT = "/tmp/grasp_exp"
D = json.load(open(f"{OUT}/category_table.json")); C = D["categories"]; O = D["objects"]
SOFT = {"cloth", "softBody", "liquid", "rope", "deformable", "physicalSubstance", "visualSubstance", "macroPhysicalSubstance", "microPhysicalSubstance"}
BOX_WORDS = r"(box|pack|book|hardback|notebook|carton|case|card|tile|brick|block|bar_of|tray|plate|remote|phone|tablet|magazine|folder|envelope|paper|sheet|board|coaster|cassette|dvd|cd|game|cartridge|wallet|calculator|keyboard|mousepad|placemat|napkin|towel|bag_of|sack|package|chocolate|candy_bar|soap|sponge|eraser|cigarette|money|bill|ticket|letter|postcard|picture|frame|mat|slab|domino|puzzle|die$|dice)"
ROUND_WORDS = r"(ball|apple|orange|peach|pear|plum|lemon|lime|melon|tomato|onion|potato|egg|coconut|grape|cherry|berry|olive|nut|globe|sphere|bulb|marble|pearl|pumpkin|squash|kiwi|mango|apricot|fig$|dumpling|meatball|bun|roll$|donut|bagel|cookie|muffin|cupcake|cake)"
ROD_WORDS = r"(pencil|pen$|crayon|marker|chopstick|stick|straw|rod|wand|baton|candle|cigar|tube|pipe|nail|screw|bolt|toothbrush|brush|spoon|fork|knife|ladle|spatula|whisk|ruler|wrench|screwdriver|hammer|file|chisel|drill|pole|cane|bat$|club|racket|broom|mop|rake|shovel|hoe|axe|saw|pliers|scissors|shears|tweezers)"


def rigid(r):
    p = set(r["props"]); return "rigidBody" in p and not (p & SOFT)


def stats(cat):
    ms = [o for o in O.values() if o["category"] == cat and o["ready"] and o["bbox"]]
    if not ms:
        return None
    B = np.array([o["bbox"] for o in ms], dtype=float)                    # canonical frame, z up
    srt = np.sort(B, axis=1)
    mn, md, mx = srt[:, 0], srt[:, 1], srt[:, 2]
    round_fp = np.abs(B[:, 0] - B[:, 1]) / np.maximum(B[:, 0], B[:, 1]) <= 0.12
    pairs = np.stack([np.abs(B[:, 0] - B[:, 1]) / np.maximum(B[:, 0], B[:, 1]), np.abs(B[:, 1] - B[:, 2]) / np.maximum(B[:, 1], B[:, 2]), np.abs(B[:, 0] - B[:, 2]) / np.maximum(B[:, 0], B[:, 2])], 1)
    sym_any = (pairs <= 0.12).any(axis=1)
    flat = mn / np.maximum(md, 1e-6) <= 0.5
    rod = mx / np.maximum(md, 1e-6) >= 3.0
    pinch = mn <= 0.085                                                     # thinnest dimension fits the gripper (rim/edge grasp)
    pinch_w = md <= 0.095                                                   # two dimensions fit (top-down pinch of the body)
    return {"n_ready": len(ms), "graspable_pinch": int(pinch.sum()), "graspable_width": int(pinch_w.sum()), "median_bbox": np.median(B, 0).round(3).tolist(),
            "round_fp": float(round_fp.mean()), "sym_any": float(sym_any.mean()), "flat": float(flat.mean()), "rod": float(rod.mean()),
            "max_extent": float(np.median(mx)), "min_extent": float(np.median(mn))}


pool = {}
for cat, r in C.items():
    if not rigid(r) or r["n_ready"] < 1:
        continue
    st = stats(cat)
    if st is None:
        continue
    st.update({"props": r["props"], "scene": "sceneObject" in r["props"], "openable": "openable" in r["props"], "attach_models": len(r["attachment"]),
               "scenes_ready": r["scenes_ready"], "box_name": bool(re.search(BOX_WORDS, cat)), "round_name": bool(re.search(ROUND_WORDS, cat)), "rod_name": bool(re.search(ROD_WORDS, cat))})
    pool[cat] = st

grasp_pool = {c: s for c, s in pool.items() if not s["scene"] and s["graspable_pinch"] >= 1 and s["max_extent"] <= 0.45}
print(len(pool), "rigid categories with ready models;", len(grasp_pool), "non-scene & pinchable (<=45 cm)")


def rank(cands, key=lambda c, s: (s["n_ready"], s["graspable_width"])):
    return sorted(cands, key=lambda c: key(c, grasp_pool[c]), reverse=True)


# ---------------------------------------------------------------- task filters (graspable object categories)
def is_symmetric(s, c):  # object scaling: prefer symmetric
    return (s["round_fp"] >= 0.6 or s["round_name"]) or (s["sym_any"] >= 0.7 and not s["box_name"])


def is_asymmetric(s, c):  # mental rotation: must not be symmetric (no equal extents, not box-like, not round, not rod-like)
    return s["sym_any"] <= 0.2 and not s["box_name"] and not s["round_name"] and not s["rod_name"] and s["round_fp"] <= 0.2


def settles(s, c):  # disembedding: has a flat stable resting face, does not roll
    return (s["flat"] >= 0.5 or s["box_name"]) and not s["round_name"] and s["rod"] <= 0.3 and s["min_extent"] >= 0.004


scaling = rank([c for c, s in grasp_pool.items() if is_symmetric(s, c)])
rotation = rank([c for c, s in grasp_pool.items() if is_asymmetric(s, c)])
disemb = rank([c for c, s in grasp_pool.items() if settles(s, c)])

# articulation: openable, articulated; scene objects need a ready model in one of the 7 scenes
artic = []
for c, r in C.items():
    if "openable" not in r["props"] or r["n_ready"] < 1 or c in ("door", "sliding_door", "gate", "openable_window", "garage_door", "car", "recreational_vehicle"):
        continue
    if "sceneObject" in r["props"] and not r["scenes_ready"]:
        continue
    artic.append((c, r["n_ready"], r["n_articulated"], r["scenes_ready"], "sceneObject" in r["props"]))
artic.sort(key=lambda x: -x[1])

# attachment: graspable categories whose models have attachment pairs with a ready partner
pairs = collections.defaultdict(lambda: {"F": set(), "M": set()})
for mid, o in O.items():
    if not o["ready"]:
        continue
    for g, names in (o["attachment"] or {}).items():
        for n in names:
            pairs[n][g].add(o["category"])
attach = collections.defaultdict(lambda: {"partners": set(), "n_pairs": 0})
for n, v in pairs.items():
    if v["F"] and v["M"]:
        for g, other in (("M", "F"), ("F", "M")):
            for cat in v[g]:
                if cat in grasp_pool:
                    attach[cat]["partners"] |= v[other]; attach[cat]["n_pairs"] += 1
attach_rank = sorted(attach.items(), key=lambda kv: (-kv[1]["n_pairs"], -grasp_pool[kv[0]]["n_ready"]))

# ---------------------------------------------------------------- greedy non-overlapping assignment
used = set()
def take(cands, n, exclude=()):
    out = []
    for c in cands:
        if c in used or c in exclude:
            continue
        out.append(c); used.add(c)
        if len(out) >= n:
            break
    return out

A = {}
A["attachment"] = [c for c, _ in attach_rank][:9]; used |= set(A["attachment"])
A["articulation"] = [c for c, *_ in artic][:9]
A["mental_rotation"] = take(rotation, 19)
A["object_scaling"] = take(scaling, 20)
A["disembedding"] = take(disemb, 19)
rest = rank([c for c in grasp_pool if c not in used])
A["_leftover_pool"] = rest[:40]


def line(c):
    s = grasp_pool.get(c) or pool.get(c) or {}
    flags = []
    if s.get("graspable_width", 0) < s.get("n_ready", 0): flags.append(f"{s.get('graspable_width', 0)} fit by width, rest edge/rim pinch")
    if s.get("round_fp", 0) >= 0.6: flags.append("round footprint")
    if s.get("flat", 0) >= 0.5: flags.append("flat")
    if s.get("rod", 0) >= 0.5: flags.append("rod-like")
    return f"| {c} | {s.get('n_ready')} | {s.get('graspable_pinch')} | {s.get('median_bbox')} | {'; '.join(flags)} |"


md = ["# Category candidates per task (auto-generated from BDDL generated data + asset metadata)", "",
      "ready = model has no unprocessed QA complaint; pinchable = thinnest extent <= 8.5 cm; bbox in metres (canonical frame, z up).", ""]
for task, n in (("object_scaling", 20), ("mental_rotation", 19), ("disembedding", 19)):
    cands = {"object_scaling": scaling, "mental_rotation": rotation, "disembedding": disemb}[task]
    md += [f"## {task} - {n} categories (assigned {len(A[task])}, {sum(1 for c in A[task] if grasp_pool[c]['n_ready'] >= 9)} with >= 9 ready)", "", "| category | ready | pinchable | median bbox | notes |", "|---|---|---|---|---|"]
    md += [line(c) for c in A[task]]
    alts = [c for c in cands if c not in A[task]][:15]
    md += ["", f"Next alternates (not assigned, may overlap other tasks): {', '.join(f'{c} ({grasp_pool[c][chr(110)+chr(95)+chr(114)+chr(101)+chr(97)+chr(100)+chr(121)]})' for c in alts)}", ""]
md += ["## configuration_matching - articulation - 9 categories", "", "| category | ready | articulated models | ready models in the 7 challenge scenes | scene object |", "|---|---|---|---|---|"]
md += [f"| {c} | {n} | {na} | {', '.join(sc) if sc else '-'} | {so} |" for c, n, na, sc, so in artic[:14]]
md += ["", "## configuration_matching - attachment - 9 categories (graspable part -> attaches to partner)", "", "| category (graspable) | ready | distinct ready pairs | partner categories |", "|---|---|---|---|"]
md += [f"| {c} | {grasp_pool[c]['n_ready']} | {v['n_pairs']} | {', '.join(sorted(v['partners']))[:160]} |" for c, v in attach_rank[:14]]
md += ["", "## leftover graspable pool for IORS (not used above), by ready count", "", "| category | ready | pinchable | median bbox | notes |", "|---|---|---|---|---|"]
md += [line(c) for c in A["_leftover_pool"]]
open(f"{OUT}/task_category_candidates.md", "w").write("\n".join(md))
json.dump({"assignment": A, "pool": grasp_pool, "scaling": scaling, "rotation": rotation, "disembedding": disemb, "articulation": artic, "attachment": [(c, v["n_pairs"], sorted(v["partners"])) for c, v in attach_rank]}, open(f"{OUT}/task_category_candidates.json", "w"), indent=1, default=list)
for k, v in A.items():
    print(f"{k:18s} ({len(v)}): {v}")
print("scaling candidates:", len(scaling), "| rotation:", len(rotation), "| disembedding:", len(disemb), "| articulation:", len(artic), "| attachment:", len(attach_rank))
