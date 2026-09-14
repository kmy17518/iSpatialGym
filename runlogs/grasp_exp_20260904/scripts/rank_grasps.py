"""Heuristic 'best grasp' ranking for stored candidates.

Features per grasp (eef pose T in the object frame, closing axis = eef y, approach = eef z):
  com_perp   perpendicular distance from the object's centre of mass to the finger-contact line (pinch axis)
             -> gravity torque about the pinch axis; 0 means the object hangs level.
  com_along  offset of the CoM along the pinch axis (load imbalance between the two pads)
  antipodal  min over the two contacts of |n . closing_axis| (friction-cone quality, 1 = perfectly opposed faces)
  coverage   fraction of rays offset along the pad width (eef x, +-0.5/1.0 cm) that hit the object at a similar width
             (flat contact patches vs. touching an edge/corner)
  width      object width between the pads; prefer 1.5-7 cm
  clearance  min gripper-sphere distance to the object with open fingers (m)
  tilt       approach angle from vertical (only meaningful for tabletop picks)
Weighted sum -> score. A1 candidates come from a1_<obj>_all.npz (already table/approach/collision filtered);
GraspGen candidates come from a2_franka_panda_<obj>_raw_big.npz (floating-object setting: no table/approach filter,
only gripper-object collision), score = heuristic + confidence bonus."""
import json, sys
import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml

OUT = "/tmp/grasp_exp"
gripper = json.load(open(f"{OUT}/r1pro_right_gripper_eef.json"))
SPH = np.array([s["center"] for s in gripper["spheres_open"]]); RAD = np.array([s["radius"] for s in gripper["spheres_open"]])
CONTACT_Z = gripper["contact_center_z"]; MAX_OPEN = gripper["max_opening"]
PANDA_DEPTH = 0.10527314
T_CONV = tra.rotation_matrix(np.pi / 2, [0, 0, 1]) @ tra.translation_matrix([0, 0, PANDA_DEPTH - CONTACT_Z])
W = {"com": 5.0, "antipodal": 1.0, "coverage": 1.0, "width": 0.5, "clearance": 1.0, "vertical": 3.0}
MAX_TILT_TABLETOP = 40.0   # deg; steeper approaches were the failure mode of the first CoM-ranked run
TOPK, NMS_POS, NMS_ROT = 5, 0.015, 30.0


def load_pieces(path):
    sc = trimesh.load(path, force="scene", process=False, split_object=True, group_material=False)
    pieces = list(sc.geometry.values()); return pieces, trimesh.util.concatenate(pieces)


def hits(union, origins, dirs):
    n = len(origins); d = np.full(n, np.inf); cosang = np.zeros(n)
    loc, idx, tri = union.ray.intersects_location(origins, dirs, multiple_hits=False)
    if len(idx):
        dist = np.linalg.norm(loc - origins[idx], axis=1)
        order = np.argsort(dist)  # keep nearest per ray
        idx, tri, dist = idx[order], tri[order], dist[order]
        first = np.unique(idx, return_index=True)[1]
        idx, tri, dist = idx[first], tri[first], dist[first]
        d[idx] = dist; cosang[idx] = np.abs(np.einsum("ij,ij->i", union.face_normals[tri], dirs[idx]))
    return d, cosang


def contact_features(union, T):
    c = T[:, :3, 3] + T[:, :3, 2] * CONTACT_Z          # pad centre
    y, x = T[:, :3, 1], T[:, :3, 0]
    dp, cp = hits(union, c, y); dm, cm = hits(union, c, -y)
    width = dp + dm
    antipodal = np.minimum(cp, cm)
    # coverage: rays offset along the pad width direction
    cov = np.zeros(len(T))
    for off in (-0.01, -0.005, 0.005, 0.01):
        o = c + x * off
        d1, _ = hits(union, o, y); d2, _ = hits(union, o, -y)
        w2 = d1 + d2
        cov += (np.isfinite(w2) & (np.abs(w2 - width) < 0.01)).astype(float)
    cov /= 4.0
    return c, width, antipodal, cov


def sphere_clearance(pieces, union, T):
    P = np.einsum("nij,kj->nki", T[:, :3, :3], SPH) + T[:, None, :3, 3]
    flat = P.reshape(-1, 3); inside = np.zeros(len(flat), dtype=bool)
    for pc in pieces:
        normals, faces = pc.face_normals, pc.triangles[:, 0]
        offs = np.einsum("ij,ij->i", normals, faces)
        inside |= ((flat @ normals.T - offs[None]) <= 0).all(axis=1)
    _, dist, _ = trimesh.proximity.closest_point(union, flat)
    dist = np.where(inside, -dist, dist)
    return (dist.reshape(len(T), -1) - RAD[None]).min(axis=1), P


def heuristic(T, com, width, antipodal, cov, clear, tabletop=True, size=0.1):
    c = T[:, :3, 3] + T[:, :3, 2] * CONTACT_Z; y = T[:, :3, 1]
    v = com[None] - c
    along = np.einsum("ij,ij->i", v, y)
    perp = np.linalg.norm(v - along[:, None] * y, axis=1)
    scale = max(0.01, 0.25 * size)            # CoM offset relative to the object size (tiny objects cannot be centred)
    f_com = np.exp(-perp / scale) * (0.6 + 0.4 * np.exp(-np.abs(along) / scale))
    f_width = np.clip(1.0 - np.maximum(0.0, width - 0.07) / 0.03, 0, 1) * np.clip((width - 0.008) / 0.007, 0, 1)
    f_clear = np.clip(clear / 0.02, 0, 1)
    f_vert = np.clip(-T[:, :3, 2][:, 2], 0, 1) if tabletop else np.ones(len(T))
    tilt = np.degrees(np.arccos(np.clip(-T[:, :3, 2][:, 2], -1, 1)))
    gate = np.where(tabletop & (tilt > MAX_TILT_TABLETOP), -10.0, 0.0)
    score = gate + W["com"] * f_com + W["antipodal"] * antipodal + W["coverage"] * cov + W["width"] * f_width + W["clearance"] * f_clear + W["vertical"] * f_vert
    return score, {"com_perp": perp, "com_along": along, "f_com": f_com, "antipodal": antipodal, "coverage": cov, "width": width, "clearance": clear, "tilt_deg": np.degrees(np.arccos(np.clip(-T[:, :3, 2][:, 2], -1, 1)))}


def nms(T, scores):
    order = np.argsort(-scores); keep = []
    for i in order:
        ok = True
        for j in keep:
            R = T[i, :3, :3].T @ T[j, :3, :3]; ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            if np.linalg.norm(T[i, :3, 3] - T[j, :3, 3]) < NMS_POS and ang < NMS_ROT:
                ok = False; break
        if ok: keep.append(i)
        if len(keep) >= TOPK: break
    return keep


def write(T, score, feats, obj, source, path, conf=None):
    data = {"format": "isaac_grasp", "format_version": 1.0, "object_frame": "base_link", "gripper_frame": "right_eef_link",
            "ispatialgym": {"object": {"category": obj["category"], "model": obj["model"]}, "gripper": "r1pro_right", "source": source, "weights": W}, "grasps": {}}
    for i, g in enumerate(T):
        q = tra.quaternion_from_matrix(g)
        e = {k: round(float(v[i]), 5) for k, v in feats.items()}
        if conf is not None: e["graspgen_conf"] = round(float(conf[i]), 4)
        data["grasps"][f"grasp_{i}"] = {"confidence": round(float(score[i]), 4), "position": [float(v) for v in g[:3, 3]], "orientation": {"w": float(q[0]), "xyz": [float(v) for v in q[1:]]},
                                        "pregrasp_cspace_position": {"right_gripper_finger_joint1": 0.05, "right_gripper_finger_joint2": 0.05}, "ispatialgym": e}
    yaml.safe_dump(data, open(path, "w"), sort_keys=False)


summary = {}
for obj in json.load(open(f"{OUT}/objects_exported.json")):
    name = obj["name"]; pieces, union = load_pieces(obj["obj"])
    com = union.center_mass if union.is_volume or True else union.centroid
    row = {"com": com.round(4).tolist()}
    # ---- A1 candidates (tabletop), heuristic ranking
    d = np.load(f"{OUT}/a1_{name}_all.npz"); T = d["T"]
    if len(T):
        c, width, antip, cov = contact_features(union, T)
        size = float(np.max(np.array(obj["coll_bbox_max"]) - np.array(obj["coll_bbox_min"])))
        score, feats = heuristic(T, com, width, antip, cov, d["clearance"], tabletop=True, size=size)
        sel = nms(T, score)
        write(T[sel], score[sel], {k: v[sel] for k, v in feats.items()}, obj, "a1_heuristic_com", f"{OUT}/a1best_{name}.yaml")
        row["a1best"] = [{k: round(float(feats[k][i]), 4) for k in ("com_perp", "antipodal", "coverage", "width", "tilt_deg")} for i in sel]
        # what did the old clearance ranking pick?  (first 5 of the old a1 set) -> com_perp for comparison
        old = yaml.safe_load(open(f"{OUT}/a1_{name}.yaml"))["grasps"]
        To = np.array([tra.quaternion_matrix([g["orientation"]["w"], *g["orientation"]["xyz"]]) @ np.eye(4) for g in old.values()])
        for i, g in enumerate(old.values()): To[i, :3, 3] = g["position"]
        _, wo, ao, co = contact_features(union, To); so, fo = heuristic(To, com, wo, ao, co, np.full(len(To), 0.01), True, size)
        row["a1_old_com_perp"] = fo["com_perp"].round(4).tolist()
    else:
        row["a1best"] = []
    # ---- GraspGen candidates for floating-object tasks: no table/approach filter, gripper-object collision only
    d = np.load(f"{OUT}/a2_franka_panda_{name}_raw_big.npz"); Tg = np.einsum("nij,jk->nik", d["T"], T_CONV); conf = d["conf"]
    clear, _ = sphere_clearance(pieces, union, Tg)
    keep = clear >= 0.002
    Tg, conf, clear = Tg[keep], conf[keep], clear[keep]
    c, width, antip, cov = contact_features(union, Tg)
    ok = np.isfinite(width) & (width <= MAX_OPEN - 0.012) & (width >= 0.006)
    Tg, conf, clear, width, antip, cov = Tg[ok], conf[ok], clear[ok], width[ok], antip[ok], cov[ok]
    size = float(np.max(np.array(obj["coll_bbox_max"]) - np.array(obj["coll_bbox_min"])))
    score, feats = heuristic(Tg, com, width, antip, cov, clear, tabletop=False, size=size)
    score = score + 2.0 * conf
    sel = nms(Tg, score) if len(Tg) else []
    write(Tg[sel], score[sel], {k: v[sel] for k, v in feats.items()}, obj, "graspgen_floating_heuristic_com", f"{OUT}/a2best_{name}.yaml", conf=conf[sel])
    row["a2_collision_free"] = int(len(Tg)); row["a2best"] = [{**{k: round(float(feats[k][i]), 4) for k in ("com_perp", "antipodal", "coverage", "width")}, "conf": round(float(conf[i]), 3)} for i in sel]
    summary[name] = row
    print(f"{name:26s} CoM={np.round(com,3).tolist()} | A1 heuristic top-5 com_perp(mm)={[round(r['com_perp']*1000) for r in row['a1best']]} vs old clearance-ranked={[round(v*1000) for v in row.get('a1_old_com_perp', [])]} | GraspGen floating: {row['a2_collision_free']} collision-free, top-5 com_perp(mm)={[round(r['com_perp']*1000) for r in row['a2best']]} conf={[r['conf'] for r in row['a2best']]}")
json.dump(summary, open(f"{OUT}/rank_summary.json", "w"), indent=1)
print("RANK DONE")
