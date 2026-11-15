"""The eval_production tree is rebuilt in place (2026-11-15 06:08 PT: instance dirs relocated under new goal groups, categories train ->
standard-eval-sibling for 12, a duplicate-start flag list). Merges therefore resolve an episode's tree files, category and group from the
tree's CURRENT index.json by instance id, never from the paths recorded at replay time."""
import json
import os
from functools import lru_cache


@lru_cache(maxsize=4)
def _index(root: str):
    rows = json.load(open(os.path.join(root, "index.json")))
    demos = {r["instance_id"]: r for r in rows if r.get("kind") == "demo"}
    flagged = set()
    fp = os.path.join(root, "FLAGGED_DUPLICATE_STARTS.txt")
    if os.path.exists(fp):
        flagged = {os.path.basename(line.strip().rstrip("/")) for line in open(fp) if line.strip()}
    return demos, flagged


def refresh(inst: dict) -> dict:
    """``inst`` (an episode's instance linkage) with tree_* paths, demo_dir, category, goal_group_id, relation, origin from the current
    tree index; unchanged when it has no eval_production root. ``tree_stale_fields`` lists what moved."""
    root = inst.get("eval_production")
    iid = inst.get("instance_id") or inst.get("demo_name")
    if not root or not iid:
        return inst
    demos, flagged = _index(root)
    r = demos.get(iid)
    if r is None:
        return {**inst, "tree_missing": True}
    d = os.path.join(root, os.path.dirname(r["dest"]))
    tdir = os.path.normpath(os.path.join(d, open(os.path.join(d, "TEMPLATE")).read().strip()))
    new = {"demo_dir": d, "tree_tro": os.path.join(d, "tro.json"), "tree_oracle": os.path.join(d, "oracle.json"), "tree_template_dir": tdir,
           "category": r["category"], "goal_group_id": r["group_id"], "goal_key": r.get("goal_key"), "relation": r.get("relation"),
           "origin": r.get("origin"), "eval_unit_id": r.get("eval_unit_id"), "duplicate_start_flagged": iid in flagged}
    stale = sorted(k for k, v in new.items() if inst.get(k) != v and k != "duplicate_start_flagged")
    return {**inst, **new, "tree_stale_fields": stale, "replayed_category": inst.get("category")}
