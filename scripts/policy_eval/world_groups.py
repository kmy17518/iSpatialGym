#!/usr/bin/env python3
"""How many worlds the eval instances need if a world is reused across instances whose templates are compatible (run_isg's
``template_compatible`` / ``_scene_leaves`` rule, reimplemented without the simulator: the template.json scene minus the entities
the instance's restore rewrites - its TRO-bound objects, the robot -, PhysX velocities / sleep flags / cached AABBs; equal keys,
equal non-numeric values, numbers within ``--tol``), plus the world-build fields (scene, rooms). Greedy clustering per scene.

    world_groups.py --instances <eval instances root> [--tol 0.005] [--json out.json]
"""
import argparse
import collections
import json
import os


def tro_names(tro):
    names = {str(b.get("scene_object_name")) for b in ((tro.get("ispatialgym") or {}).get("object_bindings") or {}).values()
             if isinstance(b, dict) and b.get("scene_object_name")}
    names |= {k for k, v in tro.items() if isinstance(v, dict) and "root_link" in v}
    return names


def leaves(doc, names):
    doc = json.loads(json.dumps(doc))
    for sect in ((((doc.get("state") or {}).get("registry") or {}).get("object_registry") or {}), (((doc.get("objects_info") or {}).get("init_info")) or {})):
        for n in list(sect):
            if n in names or n.startswith("robot"):
                sect.pop(n, None)
    ((doc.get("metadata") or {}).get("task") or {}).pop("robot_poses", None)
    ((doc.get("init_info") or {}).get("args") or {}).pop("scene_file", None)
    out = {}

    def walk(x, p):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{p}/{k}")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{p}[{i}]")
        elif not (p.endswith("/is_asleep") or "/lin_vel[" in p or "/ang_vel[" in p or "/non_kin/AABB/" in p):
            out[p] = x
    walk(doc, "")
    return out


def compatible(A, B, tol):
    if A.keys() != B.keys():
        return False
    for k, va in A.items():
        vb = B[k]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool) and not isinstance(vb, bool):
            if abs(float(va) - float(vb)) > tol:
                return False
        elif va != vb:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(a.instances, "metadata/instances.jsonl"))]
    tcache = {}
    groups = collections.defaultdict(list)  # (scene, rooms) -> [(leaves, [indices])]
    for i, r in enumerate(recs):
        tm = json.load(open(os.path.join(a.instances, r["files"]["template_meta"])))
        tro = json.load(open(os.path.join(a.instances, r["files"]["tro_state"])))
        tpath = os.path.join(a.instances, r["files"]["template"])
        if tpath not in tcache:
            tcache[tpath] = json.load(open(tpath))
        L = leaves(tcache[tpath], tro_names(tro))
        key = (tm.get("scene"), json.dumps(tm.get("load_room_instances")), tm.get("problem"))  # an attached instance keeps the world's task (BDDL problem)
        for g in groups[key]:
            if compatible(g[0], L, a.tol):
                g[1].append(i)
                break
        else:
            groups[key].append((L, [i]))
    worlds = [g[1] for gs in groups.values() for g in gs]
    sizes = collections.Counter(len(w) for w in worlds)
    print(f"{len(recs)} instances -> {len(worlds)} worlds (tol {a.tol}); sizes {sorted(sizes.items())}")
    fam = collections.Counter()
    for w in worlds:
        fam[recs[w[0]]["task_family"]] += 1
    print("worlds per family:", dict(fam))
    if a.json:
        json.dump(worlds, open(a.json, "w"))
    # within each group: the largest numeric deviation of the shared scene entries (0 = templates identical outside the task objects)
    # and whether the simulator-side records run_isg's instance_signature compares (non-revolute constraint records, attachment
    # filters) agree
    worst, rec_mismatch = 0.0, 0
    for w in worlds:
        Ls, sigs = [], set()
        for i in w:
            r = recs[i]
            tro = json.load(open(os.path.join(a.instances, r["files"]["tro_state"])))
            Ls.append(leaves(tcache[os.path.join(a.instances, r["files"]["template"])], tro_names(tro)))
            ex = tro.get("ispatialgym") or {}
            records = [{k: v for k, v in c.items() if k not in ("stock", "stock_friction")} for c in (ex.get("constraints") or [])
                       if c.get("type", "revolute") != "revolute"]
            att = sorted((str(t.get("child")), str(t.get("parent"))) for t in (ex.get("attachment_targets") or []))
            sigs.add(json.dumps({"records": records, "attachments": att}, sort_keys=True, default=str))
        rec_mismatch += len(sigs) > 1
        for L in Ls[1:]:
            for k, v in Ls[0].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    worst = max(worst, abs(float(v) - float(L[k])))
    print(f"max deviation of shared scene entries within a group: {worst:.3g}; groups whose simulator records differ: {rec_mismatch}")


if __name__ == "__main__":
    main()
