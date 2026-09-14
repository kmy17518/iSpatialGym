"""Attachment test instances must carry an attachment meta-link (M for the graspable part, F for the two female cases) and
have a counterpart model with the same attachment type. Round-1 attachment candidates were not filtered for this, so
re-select from the reviewers' scores with the eligibility filter and record each pick's counterpart(s)."""
import json, glob, collections
O = json.load(open("/tmp/grasp_exp/category_table.json"))["objects"]
F = collections.defaultdict(list); M = collections.defaultdict(list)
for o in O.values():
    for t in o["attachment"].get("F", []): F[t].append(f"{o['category']}/{o['model']}{'' if o['ready'] else ' (not ready)'}")
    for t in o["attachment"].get("M", []): M[t].append(f"{o['category']}/{o['model']}{'' if o['ready'] else ' (not ready)'}")
def partners(cat, model):
    o = next(o for o in O.values() if o["category"] == cat and o["model"] == model); out = []
    for t in o["attachment"].get("M", []): out += [f"{p} [type {t}]" for p in F.get(t, [])]
    for t in o["attachment"].get("F", []): out += [f"{p} [type {t}, this model is the female part]" for p in M.get(t, [])]
    return out
changed = []
for p in sorted(glob.glob("/tmp/glib/select/out/configuration_matching__*.json")):
    o = json.load(open(p)); cat = o["category"]
    eligible = {c["model"] for c in o["candidates"] if partners(cat, c["model"])}
    before = [s["model"] for s in o["selected"]]
    ranked = sorted([c for c in o["candidates"] if c["model"] in eligible], key=lambda c: -c["score"])
    selectable = [c for c in ranked if c["G"] >= 1 and c["V"] >= 2] or ranked[:1]
    new = [c["model"] for c in selectable[:2]]
    if any(m not in eligible for m in before):
        reasons = {s["model"]: s["reason"] for s in o["selected"]}
        o["selected"] = [{"model": c["model"], "key": c["key"], "rank": i + 1, "reason": reasons.get(c["model"], f"re-selected: highest-scoring model that carries an attachment meta-link (score {c['score']})")} for i, c in enumerate(selectable[:2])]
        o["alternate"] = ({"model": selectable[2]["model"], "key": selectable[2]["key"]} if len(selectable) > 2 else None)
        o["short"] = len(o["selected"]) < 2
        o["warnings"] = o.get("warnings", []) + [f"re-selected after attachment-link check: {sorted(set(before) - eligible)} have no attachment meta-link and can never attach"]
        changed.append((o["group"], before, new))
    for s in o["selected"]: s["partners"] = partners(cat, s["model"])
    json.dump(o, open(p, "w"), indent=1)
for g, b, n in changed: print(f"{g}: {b} -> {n}")
print(len(changed), "attachment groups re-selected")
