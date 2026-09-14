"""Turn the round-2 reviewer proposals (outer-face coordinates) into grasp candidates in the object frame and merge them into
artic/manual_handles.json (only for joints that have no usable entry yet)."""
import json, glob, os, numpy as np
A = "/tmp/glib/artic"
manual = json.load(open(f"{A}/manual_handles.json")) if os.path.exists(f"{A}/manual_handles.json") else {}
handles = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{A}/handles/*.json")}


def unit(v): v = np.asarray(v, float); n = np.linalg.norm(v); return v / n if n > 1e-9 else v


def to_candidate(prop, C, joint):
    n = unit(C["n"]); up = unit(C["up"]); right = unit(C["right_dir"]); c = {k: np.array(v) for k, v in C["corners_obj"].items()}
    a, b = float(prop["a"]), float(prop["b"]); face = prop["face"]; depth = float(prop.get("depth_mm", 0)) / 1000.0
    if face == "top_edge": b = 1.0
    elif face == "bottom_edge": b = 0.0
    elif face == "left_edge": a = 0.0
    elif face == "right_edge": a = 1.0
    P = c["BL"] + a * (c["BR"] - c["BL"]) + b * (c["TL"] - c["BL"])
    if face == "outer": contact = P - n * depth                                                   # depth > 0 = behind the face, < 0 = protruding
    elif face == "top_edge": contact = P - up * max(depth, 0.03) - n * (np.abs(n) @ (np.array(joint["link_bbox_max"]) - np.array(joint["link_bbox_min"])) / 2)
    elif face == "bottom_edge": contact = P + up * max(depth, 0.03) - n * (np.abs(n) @ (np.array(joint["link_bbox_max"]) - np.array(joint["link_bbox_min"])) / 2)
    elif face == "left_edge": contact = P + right * max(depth, 0.03) - n * (np.abs(n) @ (np.array(joint["link_bbox_max"]) - np.array(joint["link_bbox_min"])) / 2)
    elif face == "right_edge": contact = P - right * max(depth, 0.03) - n * (np.abs(n) @ (np.array(joint["link_bbox_max"]) - np.array(joint["link_bbox_min"])) / 2)
    else: return None
    pinch = {"horizontal": right, "vertical": up, "thickness": n}[prop["pinch"]]
    approach = {"front": -n, "front_down45": unit(-n - np.array([0, 0, 1.0])), "above": np.array([0, 0, -1.0]), "below": np.array([0, 0, 1.0]), "left": right, "right": -right}[prop["approach"]]
    if abs(np.dot(pinch, approach)) > 0.7: return None                                             # degenerate frame
    if joint["type"] == "PrismaticJoint": pull = unit(joint["axis_dir_obj"])
    else:
        hinge = np.array(joint["hinge_point_obj"]); ax = unit(joint["axis_dir_obj"]) * joint["open_sign"]; r = contact - hinge; r = r - np.dot(r, ax) * ax; pull = unit(np.cross(ax, r))
    return {"kind": "review2_" + face, "contact": contact.round(4).tolist(), "pinch_axis": pinch.round(4).tolist(), "approach_dir": approach.round(4).tolist(), "pull_dir": pull.round(4).tolist(),
            "width": 0.03, "depth": 0.03, "note": prop.get("note", ""), "expected_motion": prop.get("expected_motion")}


n_ok = n_inf = n_bad = 0
for p in sorted(glob.glob(f"{A}/review2/*.json")):
    r = json.load(open(p)); key, jn = r["key"], r["joint"]
    if not r.get("feasible") or not r.get("proposal"): n_inf += 1; manual.setdefault(key, {})[jn] = {"skip": f"round-2 review: infeasible - {r.get('note', '')[:80]}"}; continue
    cp = f"{A}/render2/{key}__{jn}_corners.json"
    if not os.path.exists(cp): n_bad += 1; continue
    C = json.load(open(cp)); joint = next(j for j in handles[key]["joints"] if j["joint"] == jn)
    cand = to_candidate(r["proposal"], C, joint)
    alt = to_candidate(r["alternative"], C, joint) if r.get("alternative") else None
    if cand is None and alt is None: n_bad += 1; continue
    entry = cand or alt; entry["alternative"] = alt if (cand is not None and alt is not None) else None
    manual.setdefault(key, {})[jn] = entry; n_ok += 1
json.dump(manual, open(f"{A}/manual_handles.json", "w"), indent=1)
print(f"round-2: {n_ok} candidates added, {n_inf} infeasible, {n_bad} unconvertible; manual_handles now covers {sum(len(v) for v in manual.values())} joints")
