"""Candidate grasps per model with task/object-specific constraints, ranked -> top-K isaac_grasp YAML.

Pipelines
  a1        : Isaac antipodal sampler on the collision mesh -> generic tabletop filter (approach <= 40 deg from vertical,
              width 6 mm..8.5 cm, antipodal normals, fingertips above the support plane with retreat, open-gripper sphere
              clearance >= 2 mm) -> constraint filter -> ranking -> NMS top-K
  graspgen  : GraspGen raw grasps (Panda hand frame) -> R1Pro eef frame -> collision/width filter -> mode ranking -> NMS top-K

Constraints
  tabletop / tabletop_insert : generic (fingertips never below the object's bottom -> the placed face stays free)
  attachment_free            : all gripper spheres >= 2 cm (+radius) from the attachment meta-link point(s) and the contact is
                               not on the mating side; ranking prefers grasps far from the mating point
  flower_stem_top            : contact in the top 45% of the stem axis (head end), pinch across the stem; ranking prefers the
                               region just below the head (12-35% from the head end)
  floating (GraspGen)        : mode H = confidence-first (CoM tie-break within 0.01), V/C = confidence gate (>= max-0.05) then CoM
"""
import importlib.util, json, os, sys, types
import numpy as np, trimesh, trimesh.transformations as tra, yaml

OUT = "/tmp/glib"; EXP = "/tmp/grasp_exp"
SAMPLER = "/home/minyeongk/miniconda3/envs/ispatialgym/lib/python3.11/site-packages/isaacsim/exts/isaacsim.replicator.grasping/isaacsim/replicator/grasping/sampler_utils.py"
class _Any:
    def __getattr__(self, k): return object
pxr = types.ModuleType("pxr"); pxr.Gf = _Any(); pxr.UsdGeom = _Any(); sys.modules["pxr"] = pxr
spec = importlib.util.spec_from_file_location("sampler_utils", SAMPLER); su = importlib.util.module_from_spec(spec); spec.loader.exec_module(su)

gripper = json.load(open(f"{EXP}/r1pro_right_gripper_eef.json"))
SPH = np.array([s["center"] for s in gripper["spheres_open"]]); RAD = np.array([s["radius"] for s in gripper["spheres_open"]])
FINGER = np.array(["finger" in s["link"] for s in gripper["spheres_open"]])
CONTACT_Z = gripper["contact_center_z"]; MAX_OPEN = gripper["max_opening"]
PANDA_DEPTH = 0.10527314; T_CONV = tra.rotation_matrix(np.pi / 2, [0, 0, 1]) @ tra.translation_matrix([0, 0, PANDA_DEPTH - CONTACT_Z])
N_CAND, N_ORIENT, SEED = 6000, 12, 0
APERTURE, MIN_WIDTH, MAX_TILT, OBJ_MARGIN, CONE_DEG = MAX_OPEN - 0.008, 0.003, 40.0, 0.001, 30.0
TABLE_PEN, TIP_MAX = 0.008, 0.017          # fingertip spheres may dip 5 mm into the support plane; contact no further out than the tip surface
RETREATS = (0.005, 0.01, 0.015, 0.02, 0.025)  # extra retreat steps tried when the fingertips hit the object itself (knobs, rims)
TOPK, NMS_POS, NMS_ROT = 12, 0.015, 30.0
LEVER_CAP = 0.40                              # max CoM offset from the pinch axis, fraction of the longest extent
FLOWER_WINDOW = {"lily": (0.55, 0.85, 0.70)}     # lily: blooms and leaves down to ~60% of the axis, bare stem below
FLOWER_WINDOW_DEFAULT = (0.2, 0.7, 0.35)
STEM_MAX_WIDTH = 0.030                        # flowers: stem or the receptacle behind the head (<= 30 mm), not the petals
W = {"com": 5.0, "antipodal": 1.0, "coverage": 1.0, "width": 0.5, "clearance": 1.0, "vertical": 3.0}


def load_pieces(path, R=None):
    sc = trimesh.load(path, force="scene", process=False, split_object=True, group_material=False)
    pieces = []
    for g in sc.geometry.values():
        g = g.copy()
        if R is not None:
            g.apply_transform(R)
        try:
            trimesh.repair.fix_normals(g)          # collision pieces with inverted winding break the inward ray cast
        except Exception:
            pass
        pieces.append(g)
    return pieces, trimesh.util.concatenate(pieces)


def rest_rotations():
    """Candidate resting orientations: identity first, then each of +-x / +-y pointing down (boxes standing on a side)."""
    out = [("canonical", np.eye(4))]
    for axis, sign, name in ((0, 1, "x_down"), (0, -1, "x_up"), (1, 1, "y_down"), (1, -1, "y_up")):
        v = np.zeros(3); v[axis] = sign                       # object axis that should point to -z
        R = trimesh.geometry.align_vectors(v, np.array([0, 0, -1.0]))
        out.append((name, R))
    return out


class _RayProxy:
    """embree returns only the self-hit for rays starting exactly on a face; offset the origin 0.1 mm along the ray."""
    def __init__(self, ray): self.ray = ray
    def intersects_location(self, origins, dirs, multiple_hits=True):
        o = np.asarray(origins, float) + 1e-4 * np.asarray(dirs, float)
        return self.ray.intersects_location(o, dirs, multiple_hits=multiple_hits)


class _MeshProxy:
    def __init__(self, m): self.m = m; self.ray = _RayProxy(m.ray); self.face_normals = m.face_normals
    def sample(self, *a, **k): return self.m.sample(*a, **k)


def hits(union, origins, dirs):
    n = len(origins); d = np.full(n, np.inf); cosang = np.zeros(n)
    loc, idx, tri = union.ray.intersects_location(origins, dirs, multiple_hits=False)
    if len(idx):
        dist = np.linalg.norm(loc - origins[idx], axis=1); order = np.argsort(dist); idx, tri, dist = idx[order], tri[order], dist[order]
        first = np.unique(idx, return_index=True)[1]; idx, tri, dist = idx[first], tri[first], dist[first]
        d[idx] = dist; cosang[idx] = np.abs(np.einsum("ij,ij->i", union.face_normals[tri], dirs[idx]))
    return d, cosang


def contact_features(union, T, contact_z=None):
    cz = CONTACT_Z if contact_z is None else np.asarray(contact_z).reshape(-1, 1)
    c = T[:, :3, 3] + T[:, :3, 2] * cz; y, x = T[:, :3, 1], T[:, :3, 0]
    dp, cp = hits(union, c, y); dm, cm = hits(union, c, -y)
    width = dp + dm; antipodal = np.minimum(cp, cm); cov = np.zeros(len(T))
    for off in (-0.01, -0.005, 0.005, 0.01):
        o = c + x * off; d1, _ = hits(union, o, y); d2, _ = hits(union, o, -y); w2 = d1 + d2
        with np.errstate(invalid="ignore"):
            cov += (np.isfinite(w2) & (np.abs(w2 - width) < 0.01)).astype(float)
    return c, width, antipodal, cov / 4.0


PAD_Z = np.array([-0.035, -0.02, -0.005, 0.01]); PAD_X = np.array([-0.01, 0.0, 0.01]); PAD_OPEN_HALF = 0.0485
SWEEP_MARGIN = 0.008


def pad_sweep(union, T, cz, width):
    """Closing sweep: rays from the open pad faces inward over the pad footprint. Returns the largest half-width at which a
    pad would touch the object; if that is much larger than the intended contact half-width the pads stop on another
    feature first (e.g. a flower head above the stem, a jar rim above the body)."""
    n = len(T); best = np.zeros(n); y, x, z = T[:, :3, 1], T[:, :3, 0], T[:, :3, 2]
    base = T[:, :3, 3]
    for dz in PAD_Z:
        for dx in PAD_X:
            o = base + z * dz + x * dx
            for sgn in (1.0, -1.0):
                start = o + y * (sgn * PAD_OPEN_HALF); d, _ = hits(union, start, -sgn * y)
                touch = np.where(np.isfinite(d), PAD_OPEN_HALF - d, 0.0)
                best = np.maximum(best, touch)
    return best


def sphere_positions(T):
    return np.einsum("nij,kj->nki", T[:, :3, :3], SPH) + T[:, None, :3, 3]


def sphere_clearance(pieces, union, T):
    P = sphere_positions(T); flat = P.reshape(-1, 3); inside = np.zeros(len(flat), dtype=bool)
    for pc in pieces:
        normals, faces = pc.face_normals, pc.triangles[:, 0]; offs = np.einsum("ij,ij->i", normals, faces)
        inside |= ((flat @ normals.T - offs[None]) <= 0).all(axis=1)
    _, dist, _ = trimesh.proximity.closest_point(union, flat); dist = np.where(inside, -dist, dist)
    return (dist.reshape(len(T), -1) - RAD[None]).min(axis=1), P


def nms(T, scores, k=TOPK):
    order = np.argsort(-scores); keep = []
    for i in order:
        ok = True
        for j in keep:
            R = T[i, :3, :3].T @ T[j, :3, :3]; ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            if np.linalg.norm(T[i, :3, 3] - T[j, :3, 3]) < NMS_POS and ang < NMS_ROT: ok = False; break
        if ok: keep.append(i)
        if len(keep) >= k: break
    return np.array(keep, dtype=int)


def tilt_deg(T):
    return np.degrees(np.arccos(np.clip(-T[:, :3, 2][:, 2], -1, 1)))


# ---------------------------------------------------------------- object-specific geometry helpers
def stem_axis(union, info):
    """Longest bbox axis (canonical frame) as the stem axis; head end = end with the larger perpendicular extent."""
    lo, hi = np.array(info["coll_bbox_min"]), np.array(info["coll_bbox_max"]); ext = hi - lo; ax = int(np.argmax(ext))
    v = union.vertices; t = (v[:, ax] - lo[ax]) / ext[ax]
    perp = np.delete(v, ax, axis=1)
    def spread(mask):
        return float(np.ptp(perp[mask], axis=0).max()) if mask.sum() > 3 else 0.0
    lo_end, hi_end = spread(t < 0.3), spread(t > 0.7)
    d = np.zeros(3); d[ax] = 1.0
    head = hi.copy() if hi_end >= lo_end else lo.copy()      # head end point along the axis
    axis_dir = -d if hi_end >= lo_end else d                  # direction from head end towards the stem end
    head_pt = np.array([(lo[i] + hi[i]) / 2 for i in range(3)]); head_pt[ax] = head[ax]
    return head_pt, axis_dir, float(ext[ax])


def attachment_points(info):
    pts = []
    for link, ml in (info.get("meta_links") or {}).items():
        if not isinstance(ml, dict): continue
        for k, v in ml.get("attachment", {}).items():
            entries = v.values() if isinstance(v, dict) else v
            for e in entries:
                if isinstance(e, dict) and "position" in e: pts.append(np.array(e["position"], dtype=float))
    if not pts:  # fallback: mating end = bottom (dip_candle) -> bottom centre
        lo, hi = np.array(info["coll_bbox_min"]), np.array(info["coll_bbox_max"]); c = (lo + hi) / 2; c[2] = lo[2]; pts.append(c)
    return np.array(pts)


def heuristic_tabletop(T, com, width, antip, cov, clear, size, contact_z=None):
    cz = CONTACT_Z if contact_z is None else np.asarray(contact_z).reshape(-1, 1)
    c = T[:, :3, 3] + T[:, :3, 2] * cz; y = T[:, :3, 1]; v = com[None] - c
    along = np.einsum("ij,ij->i", v, y)
    perp = np.abs(np.einsum("ij,ij->i", v, T[:, :3, 0]))      # lever arm: CoM offset along the pad-width axis (not along the approach: the pads sit above a thin object by construction)
    scale = max(0.01, 0.25 * size); f_com = np.exp(-perp / scale) * (0.6 + 0.4 * np.exp(-np.abs(along) / scale))
    f_width = np.clip(1.0 - np.maximum(0.0, width - 0.07) / 0.03, 0, 1) * np.clip((width - 0.008) / 0.007, 0, 1)
    f_clear = np.clip(clear / 0.02, 0, 1); f_vert = np.clip(-T[:, :3, 2][:, 2], 0, 1)
    score = W["com"] * f_com + W["antipodal"] * antip + W["coverage"] * cov + W["width"] * f_width + W["clearance"] * f_clear + W["vertical"] * f_vert
    return score, perp, f_com


def write_yaml(T, score, feats, info, source, path, extra_meta=None):
    data = {"format": "isaac_grasp", "format_version": 1.0, "object_frame": "base_link", "gripper_frame": "right_eef_link",
            "ispatialgym": {"object": {"category": info["category"], "model": info["model"]}, "task": info["task"], "constraint": info["constraint"], "pipeline": info["pipeline"], "gripper": "r1pro_right", "source": source, **(extra_meta or {})}, "grasps": {}}
    for i, g in enumerate(T):
        q = tra.quaternion_from_matrix(g)
        data["grasps"][f"grasp_{i}"] = {"confidence": round(float(score[i]), 4), "position": [float(v) for v in g[:3, 3]], "orientation": {"w": float(q[0]), "xyz": [float(v) for v in q[1:]]},
                                        "pregrasp_cspace_position": {"right_gripper_finger_joint1": 0.05, "right_gripper_finger_joint2": 0.05},
                                        "ispatialgym": {k: (round(float(v[i]), 5) if np.ndim(v) else v) for k, v in feats.items()}}
    yaml.safe_dump(data, open(path, "w"), sort_keys=False)


def run_a1(info, pieces, union, com, size):
    T = np.array(su.sample_antipodal(_MeshProxy(union), num_candidates=N_CAND, num_orientations=N_ORIENT, gripper_maximum_aperture=APERTURE, gripper_standoff_fingertips=CONTACT_Z,
                                     gripper_approach_direction=np.array([0, 0, 1.0]), grasp_align_axis=np.array([0, 1.0, 0]), orientation_sample_axis=np.array([0, 1.0, 0]), random_seed=SEED))
    log = {"n_sampled": int(len(T))}
    if len(T) == 0: return None, log
    T = T[tilt_deg(T) <= MAX_TILT]; log["n_tilt"] = int(len(T))
    if len(T) == 0: return None, log
    c, width, antip, cov = contact_features(union, T)
    keep = np.isfinite(width) & (width >= MIN_WIDTH) & (width <= APERTURE) & (antip >= np.cos(np.radians(CONE_DEG)))
    T = T[keep]; log["n_width_antipodal"] = int(len(T))
    if len(T) == 0: return None, log
    # table clearance with retreat along the approach (contact stays on the pads)
    Pz = sphere_positions(T)[:, :, 2]; table = (Pz - RAD[None]).min(axis=1) - info["z_min"]; a = T[:, :3, 2]
    s = np.maximum(0.0, -table - TABLE_PEN) / np.maximum(-a[:, 2], 1e-3); keep = (CONTACT_Z + s) <= TIP_MAX
    T, s = T[keep], s[keep]; T[:, :3, 3] -= a[keep] * s[:, None]; log["n_table"] = int(len(T))
    if len(T) == 0: return None, log
    clear, P = sphere_clearance(pieces, union, T); table = (P[:, :, 2] - RAD[None]).min(axis=1) - info["z_min"]
    ok = (clear >= OBJ_MARGIN) & (table >= -TABLE_PEN - 1e-4)
    # object-aware retreat: fingertips hitting the object below the contact (knob of a lid, rim of a cap) -> back off along the approach
    for ds in RETREATS:
        bad = np.where(~ok & ((CONTACT_Z + s + ds) <= TIP_MAX))[0]
        if len(bad) == 0: break
        Tb = T[bad].copy(); Tb[:, :3, 3] -= Tb[:, :3, 2] * ds
        cb, Pb = sphere_clearance(pieces, union, Tb); tb = (Pb[:, :, 2] - RAD[None]).min(axis=1) - info["z_min"]
        good = (cb >= OBJ_MARGIN) & (tb >= -TABLE_PEN - 1e-4)
        idx = bad[good]; T[idx] = Tb[good]; clear[idx] = cb[good]; s[idx] += ds; ok[idx] = True
        s[bad[~good]] += ds                                                   # keep retreating the still-colliding ones
    T, clear, s = T[ok], clear[ok], s[ok]; log["n_collision"] = int(len(T))
    if len(T) == 0: return None, log
    cz = CONTACT_Z + s                                                             # actual contact height on the pads after retreat
    c, width, antip, cov = contact_features(union, T, cz)
    touch = pad_sweep(union, T, cz, width)
    keep = np.isfinite(width) & (touch <= width / 2 + SWEEP_MARGIN)               # the pads must meet the intended contact first
    T, clear, s, cz, c, width, antip, cov = T[keep], clear[keep], s[keep], cz[keep], c[keep], width[keep], antip[keep], cov[keep]; log["n_sweep"] = int(len(T))
    if len(T) == 0: return None, log
    score, perp, f_com = heuristic_tabletop(T, com, width, antip, cov, clear, size, cz)
    feats = {"com_perp": perp, "width": width, "antipodal": antip, "coverage": cov, "clearance": clear, "tilt_deg": tilt_deg(T)}
    cons = info["constraint"]
    if cons != "flower_stem_top":                                                    # lever-arm cap: no pinch further than 40% of the object length from the CoM
        keep = perp <= LEVER_CAP * size
        T, score, clear, cz = T[keep], score[keep], clear[keep], cz[keep]; feats = {k: v[keep] for k, v in feats.items()}; log["n_lever"] = int(len(T))
        if len(T) == 0: return None, log
    if cons == "attachment_free":
        pts = attachment_points(info); P = sphere_positions(T); c = T[:, :3, 3] + T[:, :3, 2] * cz.reshape(-1, 1)
        dmin = np.min(np.linalg.norm(P[:, :, None, :] - pts[None, None, :, :], axis=-1) - RAD[None, :, None], axis=(1, 2))   # min sphere-surface distance to any mating point
        off = pts.mean(0) - com
        rel = np.einsum("ij,ij->i", c - com[None], off[None].repeat(len(c), 0)) / max(1e-6, np.linalg.norm(off) ** 2)
        side_ok = (rel <= 0.5) if np.linalg.norm(off) > 0.015 else np.ones(len(c), dtype=bool)   # side test only when the mating point is clearly off-centre
        keep = (dmin >= 0.01) & side_ok
        T, score, clear, dmin, rel = T[keep], score[keep], clear[keep], dmin[keep], rel[keep]; feats = {k: v[keep] for k, v in feats.items()}
        feats["attach_dist"] = dmin; feats["attach_side"] = rel
        score = score + 2.0 * np.clip(dmin / 0.05, 0, 1) - 2.0 * np.clip(rel, 0, 1)
        log["n_attachment_free"] = int(len(T))
    elif cons == "flower_stem_top":
        head_pt, axis_dir, L = stem_axis(union, info); c = T[:, :3, 3] + T[:, :3, 2] * cz.reshape(-1, 1)
        t = np.einsum("ij,j->i", c - head_pt[None], axis_dir) / L                      # 0 at the head end, 1 at the stem end
        across = np.abs(T[:, :3, 1] @ axis_dir)                                         # pinch axis vs stem axis (0 = across the stem)
        w_lo, w_hi, w_c = FLOWER_WINDOW.get(info.get("category") or info["key"].split("__")[0], FLOWER_WINDOW_DEFAULT)
        keep = (t >= w_lo) & (t <= w_hi) & (across <= 0.5) & (feats["width"] <= STEM_MAX_WIDTH)   # stem below the head, not the petals
        T, score, t, across = T[keep], score[keep], t[keep], across[keep]; feats = {k: v[keep] for k, v in feats.items()}
        f_pos = np.exp(-((t - w_c) / 0.15) ** 2)                                        # centre of the upper stem
        score = score - W["com"] * feats.get("com_perp", np.zeros(len(T))) * 0 + 5.0 * f_pos - 1.0 * across   # CoM term is irrelevant for this constraint
        feats["stem_t"] = t; feats["pinch_across"] = across
        log["n_flower"] = int(len(T)); log["stem_len"] = round(L, 3)
    if len(T) == 0: return None, log
    sel = nms(T, score); log["n_selected"] = int(len(sel))
    return (T[sel], score[sel], {k: v[sel] for k, v in feats.items()}), log


def run_graspgen(info, pieces, union, com, size):
    p = f"{OUT}/graspgen/{info['key']}.npz"
    if not os.path.exists(p): return None, {"error": "no graspgen output"}
    d = np.load(p); T = np.einsum("nij,jk->nik", d["T"], T_CONV); conf = d["conf"]; log = {"n_graspgen": int(len(T))}
    if len(T) == 0: return None, log
    clear, _ = sphere_clearance(pieces, union, T); keep = clear >= OBJ_MARGIN; T, conf, clear = T[keep], conf[keep], clear[keep]; log["n_collision"] = int(len(T))
    if len(T) == 0: return None, log
    c, width, antip, cov = contact_features(union, T)
    keep = np.isfinite(width) & (width >= MIN_WIDTH) & (width <= APERTURE); T, conf, clear, width, antip, cov = T[keep], conf[keep], clear[keep], width[keep], antip[keep], cov[keep]; log["n_width"] = int(len(T))
    if len(T) == 0: return None, log
    c = T[:, :3, 3] + T[:, :3, 2] * CONTACT_Z; y = T[:, :3, 1]; v = com[None] - c; along = np.einsum("ij,ij->i", v, y); perp = np.linalg.norm(v - along[:, None] * y, axis=1)
    mode = (info.get("mode") or "C")[0]
    if info.get("constraint") == "handle_half":
        # thickness profile along the long axis: the handle end is the thick end, the blade the thin end
        lo, hi = np.array(info["coll_bbox_min"]), np.array(info["coll_bbox_max"]); ext = hi - lo; ax = int(np.argmax(ext)); L = float(ext[ax])
        v = union.vertices; tt = (v[:, ax] - lo[ax]) / L; perp_v = np.delete(v, ax, axis=1)
        def thick(mask): return float(np.ptp(perp_v[mask], axis=0).min()) if mask.sum() > 3 else 0.0
        handle_hi = thick(tt > 0.65) >= thick(tt < 0.35)
        d = np.zeros(3); d[ax] = -1.0 if handle_hi else 1.0                      # from the handle end towards the blade
        hpt = (lo + hi) / 2; hpt[ax] = hi[ax] if handle_hi else lo[ax]
        th_ = np.einsum("ij,j->i", c - hpt[None], d) / L                          # 0 at the handle end, 1 at the blade tip
        keep = th_ <= 0.5
        T, conf, clear, width, antip, cov, perp, along, th_ = T[keep], conf[keep], clear[keep], width[keep], antip[keep], cov[keep], perp[keep], along[keep], th_[keep]
        log["n_handle"] = int(len(T))
        if len(T) == 0: return None, log
        score = conf * 100.0 + 20.0 * np.exp(-((th_ - 0.3) / 0.15) ** 2)          # confidence-first, prefer the middle of the handle
        feats = {"com_perp": perp, "width": width, "antipodal": antip, "coverage": cov, "clearance": clear, "tilt_deg": tilt_deg(T), "graspgen_conf": conf, "handle_t": th_}
        sel = nms(T, score); log["n_selected"] = int(len(sel)); log["mode"] = "H+handle"
        return (T[sel], score[sel], {k: v[sel] for k, v in feats.items()}), log
    if mode == "H":
        score = conf * 100.0 - np.clip(perp, 0, 0.3)      # confidence-first; CoM breaks ties within ~0.003 conf
        band = conf >= conf.max() - 0.01; score = np.where(band, conf.max() * 100.0 - perp, score)
    else:
        keep = perp <= LEVER_CAP * size                                              # lever-arm cap for the CoM-ranked modes
        if keep.any():
            T, conf, clear, width, antip, cov, perp = T[keep], conf[keep], clear[keep], width[keep], antip[keep], cov[keep], perp[keep]
        log["n_lever"] = int(len(T))
        gate = conf >= conf.max() - 0.05
        scale = max(0.01, 0.25 * size); f_com = np.exp(-perp / scale)
        score = np.where(gate, 10.0 + 5.0 * f_com + antip + cov + np.clip(clear / 0.02, 0, 1) + 2.0 * conf, conf)
    feats = {"com_perp": perp, "width": width, "antipodal": antip, "coverage": cov, "clearance": clear, "tilt_deg": tilt_deg(T), "graspgen_conf": conf}
    sel = nms(T, score); log["n_selected"] = int(len(sel)); log["mode"] = mode
    return (T[sel], score[sel], {k: v[sel] for k, v in feats.items()}), log


if __name__ == "__main__":
    jobs = json.load(open(os.environ.get("JOBS", f"{OUT}/models.json")))
    only = sys.argv[1] if len(sys.argv) > 1 else None     # 'a1' | 'graspgen' | None
    summary = {}
    for j in jobs:
        if only and j["pipeline"] != only: continue
        key = j["key"]; ip = f"{OUT}/meshes/{key}.json"
        if not os.path.exists(ip): summary[key] = {"error": "no mesh"}; continue
        info0 = json.load(open(ip)); info0.update({k: j[k] for k in ("task", "pipeline", "mode", "constraint")}); info0["key"] = key
        res = None; log = {}
        variants = rest_rotations() if j["pipeline"] == "a1" else [("canonical", np.eye(4))]
        for vname, R in variants:
            info = dict(info0); pieces, union = load_pieces(info["obj"], None if vname == "canonical" else R)
            lo, hi = union.vertices.min(0), union.vertices.max(0); info["coll_bbox_min"], info["coll_bbox_max"], info["z_min"] = lo.tolist(), hi.tolist(), float(lo[2])
            if vname != "canonical":
                ext = hi - lo
                if min(ext[0], ext[1]) / max(ext[2], 1e-6) < 0.2: continue        # would not stand stably on that face
                if info["constraint"] in ("attachment_free", "flower_stem_top"): break  # keep those in their canonical pose
                # rotate attachment meta-link points too
                for link, ml in (info.get("meta_links") or {}).items():
                    if isinstance(ml, dict):
                        for k, v in ml.get("attachment", {}).items():
                            for e in (v.values() if isinstance(v, dict) else v):
                                if isinstance(e, dict) and "position" in e: e["position"] = (R[:3, :3] @ np.array(e["position"])).tolist()
            com = union.center_mass if union.volume > 1e-9 else union.centroid
            size = float(np.max(hi - lo))
            res, log = (run_a1 if j["pipeline"] == "a1" else run_graspgen)(info, pieces, union, com, size)
            log["com"] = com.round(4).tolist(); log["rest_pose"] = vname
            if res is not None: break
        if res is not None:
            T, score, feats = res
            if vname != "canonical":                      # back to the canonical object frame; remember how the object must rest
                Rinv = np.linalg.inv(R); T = np.einsum("ij,njk->nik", Rinv, T)
            q = tra.quaternion_from_matrix(R)                # wxyz, rotation applied to the object for this resting pose
            write_yaml(T, score, feats, info, f"{j['pipeline']}_{j['constraint']}", f"{OUT}/cands/{key}.yaml", {"log": log, "rest_pose": vname, "rest_quat_wxyz": [float(x) for x in q]})
        summary[key] = log
        print(f"{key:40s} {j['pipeline']:8s} {j['constraint']:16s} -> {log.get('n_selected', 0):2d} candidates  {log}", flush=True)
    json.dump(summary, open(f"{OUT}/cands_summary{('_' + only) if only else ''}.json", "w"), indent=1)
    n0 = sum(1 for v in summary.values() if not v.get("n_selected")); print(f"CANDS DONE: {len(summary)} models, {n0} with zero candidates")
