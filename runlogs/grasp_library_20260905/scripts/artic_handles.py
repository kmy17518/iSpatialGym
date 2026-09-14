"""Geometric handle inference for openable links (object frame, closed state).

For every (model, openable joint) from artic/inv: find the movable link's outward face, collect the convex collision pieces that
protrude from it (handles / knobs), cluster them into handle candidates and derive a pinch pose (contact point, pinch axis,
approach direction = into the face, pull direction = opening motion at the contact). Without protruding geometry fall back to an
exposed edge (lid front edge, drawer top edge, door free edge) or mark the link as flush / ungraspable.
Output: artic/handles/<cat>__<model>.json"""
import json, glob, os, collections
import numpy as np, trimesh
INV = "/tmp/glib/artic/inv"; OUT = "/tmp/glib/artic/handles"; os.makedirs(OUT, exist_ok=True)
PROTRUDE_MIN = 0.008      # m: pieces sticking out of the face by less than this are not handles
GRASP_DEPTH_MIN = 0.012   # m: a handle must protrude at least this much to be pinched by the fingertips
BIG_FOOTPRINT = 0.20      # fraction of the link's face area: pieces this big are slab, not handle
MAX_PINCH = 0.085         # m: gripper opening usable for a pinch
EDGE_CLEAR = 0.03         # m: free space needed beyond an edge to pinch it


def load_pieces(path):
    sc = trimesh.load(path, force="scene", process=False, split_object=True, group_material=False)
    out = []
    for name, g in sc.geometry.items():
        if len(g.vertices): out.append((name, np.asarray(g.vertices, float)))
    return out


def unit(v): n = np.linalg.norm(v); return v / n if n > 1e-9 else v


def basis_perp(n):
    a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u = unit(np.cross(n, a)); v = unit(np.cross(n, u)); return u, v


def infer(info, j):
    pieces = load_pieces(j["pieces_obj"]); allv = np.concatenate([v for _, v in pieces])
    lo, hi = np.array(j["link_bbox_min"]), np.array(j["link_bbox_max"]); ext = hi - lo; ctr = (lo + hi) / 2
    blo, bhi = np.array(info["base_link"]["bbox_min"]), np.array(info["base_link"]["bbox_max"]); bctr = (blo + bhi) / 2
    axis = np.array(j["axis_dir_obj"]); kind = "drawer" if j["type"] == "PrismaticJoint" else ("door" if abs(axis[2]) > 0.7 else "flap")
    # ---- outward normal n of the face that carries the handle
    if kind == "drawer":
        n = unit(axis)                                                       # drawers open along the joint axis
    else:
        cand = [i for i in range(3) if abs(axis[i]) < 0.7]                  # axes perpendicular to the hinge
        thin = min(cand, key=lambda i: ext[i]); n = np.zeros(3); n[thin] = 1.0
        if np.dot(ctr - bctr, n) < 0: n = -n                                 # outward = away from the body
        if kind == "flap" and thin == 2 and (ctr[2] > bhi[2] - 0.02): n = np.array([0, 0, 1.0])   # top lid: outward = up
    u, v = basis_perp(n)
    proj = allv @ n; s_all = proj.max()
    # ---- face plane s: outer surface of the slab (big pieces)
    face_area = ext[np.argsort(np.abs(n))[:2]].prod() if True else 1.0
    piece_stats = []
    for name, pv in pieces:
        p = pv @ n; fu, fv = pv @ u, pv @ v; foot = max(fu.ptp(), 1e-4) * max(fv.ptp(), 1e-4)
        piece_stats.append({"name": name, "pmax": float(p.max()), "pmin": float(p.min()), "foot": float(foot / max(face_area, 1e-6)), "verts": pv})
    big = [p for p in piece_stats if p["foot"] >= BIG_FOOTPRINT]
    s = max(p["pmax"] for p in big) if big else float(np.percentile(proj, 60))
    handles = [p for p in piece_stats if p["pmax"] > s + PROTRUDE_MIN and p["foot"] < BIG_FOOTPRINT]
    # ---- cluster protruding pieces (2 cm proximity on the face plane)
    clusters = []
    for p in handles:
        pv = p["verts"][p["verts"] @ n > s + 0.003]
        if len(pv) < 3: continue
        c = pv.mean(0); placed = False
        for cl in clusters:
            if np.linalg.norm((c - cl["c"]) - np.dot(c - cl["c"], n) * n) < 0.06 and abs(np.dot(c - cl["c"], n)) < 0.05:
                cl["verts"].append(pv); cl["c"] = np.concatenate(cl["verts"]).mean(0); placed = True; break
        if not placed: clusters.append({"verts": [pv], "c": c})
    cands = []
    for cl in clusters:
        pv = np.concatenate(cl["verts"]); depth = float(pv @ n).max() - s if False else float((pv @ n).max() - s)
        eu, ev = (pv @ u), (pv @ v); len_u, len_v = float(eu.ptp()), float(ev.ptp())
        long_dir, short_dir, L, W = (u, v, len_u, len_v) if len_u >= len_v else (v, u, len_v, len_u)
        if W > MAX_PINCH and L > MAX_PINCH: continue                          # a plate, not a handle
        if depth < GRASP_DEPTH_MIN: cands.append({"kind": "flush_handle", "center": pv.mean(0).round(4).tolist(), "depth": round(depth, 4), "note": "protrudes < 12 mm; fingertips cannot get behind it"}); continue
        centre_face = np.array([np.dot(pv.mean(0), u), np.dot(pv.mean(0), v)])
        contact_depth = s + max(depth / 2, depth - 0.012)                     # pad centre inside the protruding part
        contact = u * centre_face[0] + v * centre_face[1] + n * contact_depth
        pinch = short_dir                                                    # fingers close across the bar's short dimension
        cands.append({"kind": "bar" if L > 2.5 * W else "knob", "contact": contact.round(4).tolist(), "pinch_axis": pinch.round(4).tolist(), "approach_dir": (-n).round(4).tolist(),
                      "width": round(W, 4), "length": round(L, 4), "depth": round(depth, 4), "face_s": round(s, 4), "n_pieces": len(cl["verts"])})
    # separate handle links attached to the leaf (door lever handles): use their bounding boxes directly
    for sl in j.get("sub_links", []):
        slo, shi = np.array(sl["bbox_min"]), np.array(sl["bbox_max"]); sc = (slo + shi) / 2; se = shi - slo
        depth = float(np.dot(shi, n) if np.dot(n, n) else 0.0) - s if True else 0.0
        depth = float(max((np.array([[slo[i], shi[i]][int(n[i] > 0)] for i in range(3)]) @ n) - s, 0.0))
        inplane = [i for i in range(3) if abs(n[i]) < 0.7]; short_i = min(inplane, key=lambda i: se[i]); long_i = max(inplane, key=lambda i: se[i])
        pinch = np.zeros(3); pinch[short_i] = 1.0
        contact = sc.copy(); contact = contact + n * (max(depth / 2, depth - 0.012) - np.dot(sc - (sc - n * np.dot(sc, n) + n * s), n)) if False else sc
        cands.append({"kind": "bar" if se[long_i] > 2.5 * se[short_i] else "knob", "contact": contact.round(4).tolist(), "pinch_axis": pinch.round(4).tolist(), "approach_dir": (-n).round(4).tolist(),
                      "width": round(float(se[short_i]), 4), "length": round(float(se[long_i]), 4), "depth": round(depth, 4), "face_s": round(s, 4), "n_pieces": 0, "source": f"handle link {sl['link']}"})
    usable = [c for c in cands if c["kind"] in ("bar", "knob")]
    fallback = None
    if True:
        # ---- edge fallbacks (always computed so a reviewer can prefer the edge over a protruding part, e.g. a carry handle)
        if kind == "flap":
            hinge = np.array(j["hinge_point_obj"]); e = ctr - hinge; e = unit(e - np.dot(e, axis) * axis)        # hinge -> free edge, in the leaf plane
            far = allv[np.argsort(allv @ e)[-max(3, len(allv) // 20):]]; edge_c = far.mean(0)
            thickness = float(np.ptp(allv[(allv @ e) > (allv @ e).max() - 0.03] @ n))
            base_beyond = float(max(np.dot(np.array([blo[i] if e[i] < 0 else bhi[i] for i in range(3)]), e) - edge_c @ e, 0.0))
            exposed = base_beyond < 0.01 or (n[2] > 0.7 and lo[2] > bhi[2] - 0.02)
            fallback = {"kind": "edge_lid", "contact": (edge_c - e * 0.035).round(4).tolist(), "pinch_axis": n.round(4).tolist(), "approach_dir": (-e).round(4).tolist(),
                        "width": round(min(thickness, ext[np.argmax(np.abs(n))]), 4), "exposed": bool(exposed), "note": "pinch the lid/flap across its thickness at the free edge"}
        elif kind == "drawer":
            top = allv[(allv @ n) > s - 0.03]; top_z = top[:, 2].max(); front_c = np.array([np.dot(ctr, u), np.dot(ctr, v)])
            others_above = any(np.array(o["link_bbox_min"])[2] < top_z + EDGE_CLEAR and np.array(o["link_bbox_max"])[2] > top_z and o["joint"] != j["joint"] for o in info["joints"]) or (bhi[2] > top_z + 0.01)
            thickness = float(np.ptp(top[top[:, 2] > top_z - 0.03] @ n))
            fallback = {"kind": "edge_drawer_top", "contact": [float(x) for x in (u * np.dot(ctr, u) + v * np.dot(ctr, v) + n * (s - thickness / 2) + np.array([0, 0, top_z - 0.015]) - np.array([0, 0, np.dot(ctr, np.array([0, 0, 1.0]))]) * 0)],
                        "pinch_axis": n.round(4).tolist(), "approach_dir": [0, 0, -1.0], "width": round(thickness, 4), "exposed": not others_above, "note": "pinch the drawer front across its thickness from above"}
            fc = np.array(fallback["contact"]); fc[2] = top_z - 0.035; fallback["contact"] = fc.round(4).tolist()
        else:
            hinge = np.array(j["hinge_point_obj"]); e = ctr - hinge; e = unit(e - np.dot(e, axis) * axis)
            far = allv[np.argsort(allv @ e)[-max(3, len(allv) // 20):]]; edge_c = far.mean(0)
            others = [o for o in info["joints"] if o["joint"] != j["joint"]]
            blocked = any(np.all(np.array(o["link_bbox_min"]) - 0.005 < edge_c + e * 0.02) and np.all(np.array(o["link_bbox_max"]) + 0.005 > edge_c + e * 0.02) for o in others) or \
                      (np.all(blo - 0.005 < edge_c + e * 0.02) and np.all(bhi + 0.005 > edge_c + e * 0.02))
            fallback = {"kind": "edge_door", "contact": (edge_c - e * 0.035).round(4).tolist(), "pinch_axis": n.round(4).tolist(), "approach_dir": (-e).round(4).tolist(),
                        "width": round(float(ext[np.argmax(np.abs(n))]), 4), "exposed": not blocked, "note": "pinch the door leaf across its thickness at the free edge"}
    # ---- pull direction at the contact (opening motion)
    for c in usable + ([fallback] if fallback else []):
        p = np.array(c["contact"])
        if kind == "drawer": pull = axis
        else:
            hinge = np.array(j["hinge_point_obj"]); r = p - hinge; r = r - np.dot(r, axis) * axis; pull = unit(np.cross(axis, r))
        c["pull_dir"] = pull.round(4).tolist()
    return {"joint": j["joint"], "type": j["type"], "kind": kind, "child_link": j["child_link"], "outward_normal": n.round(4).tolist(), "face_s": round(float(s), 4), "axis_dir_obj": j["axis_dir_obj"],
            "hinge_point_obj": j.get("hinge_point_obj"), "open_sign": j["open_sign"], "lower": j["lower"], "upper": j["upper"], "closed_q": j["closed_q"], "link_bbox_min": j["link_bbox_min"], "link_bbox_max": j["link_bbox_max"],
            "handles": usable, "flush_handles": [c for c in cands if c["kind"] == "flush_handle"], "fallback": fallback, "sub_links": j.get("sub_links", []),
            "status": "handle" if usable else ("edge" if fallback and fallback.get("exposed") else "ungraspable")}


summary = collections.Counter(); rows = []
for p in sorted(glob.glob(f"{INV}/*.json")):
    info = json.load(open(p)); res = {"category": info["category"], "model": info["model"], "key": info["key"], "aabb_extent": info["aabb_extent"], "base_link": info["base_link"], "joints": []}
    for j in info["joints"]:
        try: r = infer(info, j)
        except Exception as e: r = {"joint": j["joint"], "type": j["type"], "child_link": j["child_link"], "status": "error", "error": repr(e)}
        res["joints"].append(r); summary[(info["category"], r["status"])] += 1
        rows.append((info["category"], info["model"], j["joint"], r["status"], r.get("kind"), len(r.get("handles", [])), (r.get("fallback") or {}).get("kind")))
    json.dump(res, open(f"{OUT}/{info['key']}.json", "w"), indent=1)
print(f"{'category':22s} handle  edge  ungraspable  error")
for c in sorted({k[0] for k in summary}): print(f"{c:22s} {summary[(c, 'handle')]:6d} {summary[(c, 'edge')]:5d} {summary[(c, 'ungraspable')]:12d} {summary[(c, 'error')]:6d}")
print(len(rows), "openable joints in", len(glob.glob(f"{INV}/*.json")), "models")
