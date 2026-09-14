"""Aggregate the articulated-object opening tests: per (model, joint) reachability / AG / opened, plus the review verdicts.
Writes artic/artic_results.csv and artic/artic_results.md (with the list of handles to exclude)."""
import json, glob, os, csv, collections
A = "/tmp/glib/artic"
handles = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{A}/handles/*.json")}
reviews = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{A}/review/*.json")}
results = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{A}/results/*.json")}
assign = {f"{a['category']}__{a['model']}": a for a in json.load(open(f"{A}/scene_assign.json"))} if os.path.exists(f"{A}/scene_assign.json") else {}
NON_SCENE = {"blender", "carton", "cedar_chest", "cooler", "hinged_jar", "laptop", "recycling_bin", "toolbox", "violin_case", "mailbox", "backpack"}


def err_class(rec):
    e = (rec.get("error") or "")
    if rec.get("grasp_source", "").startswith("manual skip"): return "infeasible_2_reviews" if "round-2" in rec["grasp_source"] else "asset_issue"
    if not rec.get("grasp_source", "").startswith(("review:", "heuristic:", "manual")) or e.startswith("no graspable"): return "no_handle"
    if rec.get("opened"): return "opened"
    if rec.get("ag") and not rec.get("opened"): return "grasped_not_opened"
    if rec.get("reachable") is False or "unreachable" in e or "no stance" in e: return "unreachable"
    if "not in hand" in e: return "grasp_no_ag"
    if "planning" in e or "IK" in e: return "planning"
    if "collision" in e: return "collision"
    return "other" if e else "untested"


rows = []
for key, H in sorted(handles.items()):
    cat, model = H["category"], H["model"]; rv = reviews.get(key, {}); res = results.get(key)
    for j in H["joints"]:
        rj = next((r for r in rv.get("joints", []) if r["joint"] == j["joint"]), {})
        rec = next((r for r in (res or {}).get("joints", []) if r["joint"] == j["joint"]), None) if res else None
        rows.append({"category": cat, "model": model, "joint": j["joint"], "kind": j.get("kind"), "heuristic": j.get("status"), "review_chosen": rj.get("chosen", ""), "review_pinch": rj.get("pinch", ""), "review_motion": rj.get("motion", ""),
                     "review_graspable": rj.get("graspable", ""), "handle_desc": rj.get("handle_description", ""), "height_concern": rj.get("height_concern", ""),
                     "setting": "table/floor" if cat in NON_SCENE else f"scene:{(res or {}).get('scene', assign.get(key, {}).get('scene', ''))}", "host": (res or {}).get("host", ""), "swap": (res or {}).get("swap", ""), "scale": (res or {}).get("scale", ""),
                     "grasp_source": (rec or {}).get("grasp_source", ""), "handle_height": (rec or {}).get("handle_height_above_floor", (rec or {}).get("handle_height", "")), "reachable": (rec or {}).get("reachable", "") if rec else "",
                     "stance_translate": ((rec or {}).get("stance") or {}).get("translate", ""), "ag": (rec or {}).get("ag", "") if rec else "", "dq": (rec or {}).get("dq", ""), "dq_target": (rec or {}).get("dq_target", ""),
                     "opened": (rec or {}).get("opened", "") if rec else "", "open_state": (rec or {}).get("open_state", ""), "base_shifts": (rec or {}).get("base_shifts", ""), "result": err_class(rec) if rec else ("model_error" if res and res.get("error") else "untested"),
                     "error": (rec or {}).get("error", "") or ((res or {}).get("error", "") if res else "")})
with open(f"{A}/artic_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# markdown
L = ["# Articulated objects: handle inference, review and opening tests", "",
     f"{len(handles)} models, {len(rows)} openable joints. Heuristic handle status: " + ", ".join(f"{k} {v}" for k, v in collections.Counter(r['heuristic'] for r in rows).items()) + ".",
     "Review verdicts (chosen marker): " + ", ".join(f"{k or 'n/a'} {v}" for k, v in collections.Counter(r['review_chosen'] for r in rows).items()) + f"; graspable=false: {sum(1 for r in rows if r['review_graspable'] is False)}.", ""]
tested = [r for r in rows if r["result"] != "untested"]
L += ["## Opening tests", "", f"{len(tested)} joints tested so far. Outcome: " + ", ".join(f"{k} {v}" for k, v in sorted(collections.Counter(r['result'] for r in tested).items())) + ".", "",
      "| category | model | joint | kind | setting | handle (review) | height m | reachable | trunk | AG | dq / target | opened | shifts | result / error |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in tested:
    L.append(f"| {r['category']} | {r['model']} | {r['joint']} | {r['kind']} | {r['setting']} | {r['review_chosen']} {r['review_pinch']} | {r['handle_height']} | {r['reachable']} | {r['stance_translate']} | {r['ag']} | {r['dq']} / {r['dq_target']} | {r['opened']} | {r['base_shifts']} | {r['result']}: {str(r['error'])[:90]} |")
viable = sorted({(r["category"], r["model"]) for r in tested if r["result"] == "opened"})
L += ["", f"## Viable models ({len(viable)}; at least one joint opened by the robot)", ""] + [f"* {c} / {m}: " + ", ".join(f"{r['joint']} ({r['dq']} / {r['dq_target']}, {r['setting']})" for r in tested if (r["category"], r["model"]) == (c, m) and r["result"] == "opened") for c, m in viable]
L += ["", "## Handles to exclude (unreachable or ungraspable)", ""]
for r in rows:
    if r["result"] in ("unreachable", "no_handle", "infeasible_2_reviews", "asset_issue") or r["review_graspable"] is False:
        L.append(f"* {r['category']} / {r['model']} / {r['joint']} ({r['kind']}): {r['result']} - {r['handle_desc'] or r['error']}")
open(f"{A}/artic_results.md", "w").write("\n".join(L))
print(f"{len(rows)} joints; tested {len(tested)}; outcomes: {dict(collections.Counter(r['result'] for r in tested))}")
