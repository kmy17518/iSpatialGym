"""Validate and aggregate the per-group test-instance selections -> test_instances.json / .csv / .md."""
import json, glob, os, csv, collections
SEL = "/tmp/glib/select"
groups = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{SEL}/groups/*.json")}
outs = {}
problems = []
for p in sorted(glob.glob(f"{SEL}/out/*.json")):
    gid = os.path.basename(p)[:-5]
    try: o = json.load(open(p))
    except Exception as e: problems.append((gid, f"unreadable: {e}")); continue
    g = groups.get(gid)
    if g is None: problems.append((gid, "no matching group")); continue
    want = {c["model"] for c in g["candidates"]}; got = {c["model"] for c in o.get("candidates", [])}
    if want != got: problems.append((gid, f"candidate set mismatch: missing {sorted(want - got)} extra {sorted(got - want)}"))
    if o.get("n_opened") != o.get("n_candidates") or o.get("n_candidates") != len(want): problems.append((gid, f"n_opened {o.get('n_opened')} / n_candidates {o.get('n_candidates')} vs {len(want)} input"))
    for c in o.get("candidates", []):
        if c.get("score") != 2 * c.get("G", 0) + 2 * c.get("V", 0) + c.get("T", 0) + c.get("R", 0): problems.append((gid, f"score arithmetic {c['model']}"))
    if len(o.get("selected", [])) > 2: problems.append((gid, "more than 2 selected"))
    if len(o.get("selected", [])) < min(2, len(want)) and not o.get("short"): problems.append((gid, f"{len(o.get('selected', []))} selected of {len(want)} without short flag"))
    outs[gid] = o
missing = sorted(set(groups) - set(outs))
rows = []
for gid, g in sorted(groups.items()):
    o = outs.get(gid)
    sc = {c["model"]: c for c in (o or {}).get("candidates", [])}
    sel = (o or {}).get("selected", [])
    def pick(i):
        if i < len(sel):
            m = sel[i]; c = sc.get(m["model"], {}); return m["model"], m.get("key"), c.get("score"), f"G{c.get('G')} T{c.get('T')} V{c.get('V')} R{c.get('R')}", m.get("reason", "")
        return "", "", "", "", ""
    p1, p2 = pick(0), pick(1)
    rows.append({"task": g["task"], "relation": g["relation"] or "", "partner": g["partner"] or "", "category": g["category"], "round": g["round"], "n_candidates": len(g["candidates"]),
                 "pick1": p1[0], "pick1_key": p1[1], "pick1_score": p1[2], "pick1_scores": p1[3], "pick1_reason": p1[4], "pick2": p2[0], "pick2_key": p2[1], "pick2_score": p2[2], "pick2_scores": p2[3], "pick2_reason": p2[4],
                 "alternate": ((o or {}).get("alternate") or {}).get("model", ""), "short": (o or {}).get("short", o is None), "confidence": (o or {}).get("confidence", "" if o else "missing"),
                 "warnings": " | ".join((o or {}).get("warnings", [])) if o else "no output yet",
                 "partners": " ; ".join(f"{s_['model']} -> {', '.join(s_.get('partners', []))}" for s_ in sel if s_.get("partners")) if o else ""})
json.dump(rows, open(f"{SEL}/test_instances.json", "w"), indent=1)
with open(f"{SEL}/test_instances.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# markdown
L = ["# Test instances: two models per (task, category)", "",
     "Selected by parallel reviewer agents under `select_contract.md` (score = 2*G + 2*V + T + R, max 25; G grasp quality, V visual quality, T task fit, R robustness). "
     f"{len(outs)} of {len(groups)} groups have a selection; {sum(1 for r in rows if r['pick2'])} groups have two picks, {sum(1 for r in rows if r['pick1'] and not r['pick2'])} have one, {sum(1 for r in rows if not r['pick1'])} have none.", ""]
for task in ["object_scaling", "mental_rotation", "disembedding", "configuration_matching", "IORS"]:
    extra = task == "configuration_matching"
    L += [f"## {task}", "", "| round | relation | category | cands | pick 1 | score | pick 2 | score | alternate | conf | warnings |" + (" attachment counterpart(s) |" if extra else ""), "|---|---|---|---|---|---|---|---|---|---|---|" + ("---|" if extra else "")]
    for r in sorted([r for r in rows if r["task"] == task], key=lambda r: (r["round"], r["relation"], r["category"])):
        L.append(f"| {r['round']} | {r['relation']} | {r['category']} | {r['n_candidates']} | {r['pick1']} | {r['pick1_score']} | {r['pick2']} | {r['pick2_score']} | {r['alternate']} | {r['confidence']} | {r['warnings'][:120]} |" + (f" {r['partners']} |" if extra else ""))
    L.append("")
dup = collections.defaultdict(list)
for r in rows:
    for k in ("pick1", "pick2"):
        if r[k]: dup[(r["category"], r[k])].append(r["task"])
multi = {k: v for k, v in dup.items() if len(v) > 1}
L += ["## Models selected for more than one task", "", ", ".join(f"{c}/{m} ({', '.join(t)})" for (c, m), t in sorted(multi.items())) or "none", ""]
if problems or missing:
    L += ["## Validation", ""] + [f"* {g}: {m}" for g, m in problems] + ([f"* missing outputs: {', '.join(missing)}"] if missing else []) + [""]
open(f"{SEL}/test_instances.md", "w").write("\n".join(L))
print(f"{len(outs)}/{len(groups)} groups done; problems: {len(problems)}; missing: {len(missing)}")
for g, m in problems[:20]: print("  PROBLEM", g, m)
