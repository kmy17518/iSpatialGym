"""Approach 1: Isaac antipodal sampler -> geometric filter (R1Pro gripper spheres, table plane, width) -> top-5.
Runs without OmniGibson (pure trimesh/numpy). Grasps are right_eef_link poses in the object base_link frame."""
import importlib.util, json, sys, time, types
import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml

OUT = "/tmp/grasp_exp"
SAMPLER = "/home/minyeongk/miniconda3/envs/ispatialgym/lib/python3.11/site-packages/isaacsim/exts/isaacsim.replicator.grasping/isaacsim/replicator/grasping/sampler_utils.py"


class _Any:
    def __getattr__(self, k):
        return object


pxr = types.ModuleType("pxr"); pxr.Gf = _Any(); pxr.UsdGeom = _Any(); sys.modules["pxr"] = pxr  # USD-only imports not needed here
spec = importlib.util.spec_from_file_location("sampler_utils", SAMPLER); sampler_utils = importlib.util.module_from_spec(spec); spec.loader.exec_module(sampler_utils)

gripper = json.load(open(f"{OUT}/r1pro_right_gripper_eef.json"))
SPH = np.array([s["center"] for s in gripper["spheres_open"]]); RAD = np.array([s["radius"] for s in gripper["spheres_open"]])
CONTACT_Z = gripper["contact_center_z"]          # pad centre along the eef approach axis (negative = behind the origin)
MAX_OPEN = gripper["max_opening"]                 # 0.097 m between the pads when open

# --- filter parameters
N_CAND, N_ORIENT, SEED = 4000, 12, 0
APERTURE = MAX_OPEN - 0.012                       # candidates must fit with >= 6 mm per side when open
MIN_WIDTH = 0.006
MAX_TILT_DEG = 60.0                               # approach axis within 60 deg of straight down
TABLE_CLEAR = 0.0                               # finger spheres stay >= 3 mm above the support plane
OBJ_MARGIN = 0.002                                # spheres stay >= 2 mm away from the object (open gripper)
NMS_POS, NMS_ROT_DEG = 0.015, 30.0
TOPK = 5


def load_pieces(path):
    sc = trimesh.load(path, force="scene", process=False, split_object=True, group_material=False)
    pieces = [g for g in sc.geometry.values()]
    union = trimesh.util.concatenate(pieces)
    return pieces, union


def object_width_along(union, centers, axes, cone_deg=30.0):
    """Width of the object around each centre along +-axis (nearest surface hit each side) and whether both contact
    normals lie within ``cone_deg`` of the closing axis (antipodal / friction-cone check)."""
    n = len(centers)
    w = np.full(n, np.inf)
    antipodal = np.ones(n, dtype=bool)
    cos_min = np.cos(np.radians(cone_deg))
    for sgn in (1.0, -1.0):
        loc, idx_ray, idx_tri = union.ray.intersects_location(centers, sgn * axes, multiple_hits=False)
        d = np.full(n, np.inf)
        ok = np.zeros(n, dtype=bool)
        if len(idx_ray):
            dist = np.linalg.norm(loc - centers[idx_ray], axis=1)
            np.minimum.at(d, idx_ray, dist)
            nrm = union.face_normals[idx_tri]
            cosang = np.abs(np.einsum("ij,ij->i", nrm, axes[idx_ray]))
            ok[idx_ray] = cosang >= cos_min
        w = np.where(np.isinf(w), d, w + d)
        antipodal &= ok
    return w, antipodal


def sphere_clearance(pieces, union, T):
    """Min over spheres of (distance to object surface - radius); negative = penetration. T: (N,4,4)."""
    N = len(T)
    P = np.einsum("nij,kj->nki", T[:, :3, :3], SPH) + T[:, None, :3, 3]      # (N,K,3)
    flat = P.reshape(-1, 3)
    # inside test: convex pieces -> half-space test (exact)
    inside = np.zeros(len(flat), dtype=bool)
    for pc in pieces:
        eq = pc.face_normals @ pc.triangles[:, 0].T  # not used
        normals, faces = pc.face_normals, pc.triangles[:, 0]
        offs = np.einsum("ij,ij->i", normals, faces)
        sd = flat @ normals.T - offs[None]           # (M, F) signed plane distances
        inside |= (sd <= 0).all(axis=1)
    # unsigned distance to surface (union)
    closest, dist, _ = trimesh.proximity.closest_point(union, flat)
    dist = np.where(inside, -dist, dist)
    clear = dist.reshape(N, -1) - RAD[None]
    return clear.min(axis=1), P


def nms(T, scores, pos_th, rot_deg):
    order = np.argsort(-scores); keep = []
    for i in order:
        ok = True
        for j in keep:
            dp = np.linalg.norm(T[i, :3, 3] - T[j, :3, 3])
            R = T[i, :3, :3].T @ T[j, :3, :3]
            ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
            if dp < pos_th and ang < rot_deg:
                ok = False; break
        if ok:
            keep.append(i)
        if len(keep) >= TOPK:
            break
    return keep


def to_isaac_grasp(T, extra, obj, out_path):
    data = {"format": "isaac_grasp", "format_version": 1.0, "object_frame": "base_link", "gripper_frame": "right_eef_link",
            "ispatialgym": {"object": {"category": obj["category"], "model": obj["model"]}, "gripper": "r1pro_right", "source": "isaac_antipodal+geometric_filter"},
            "grasps": {}}
    for i, g in enumerate(T):
        q = tra.quaternion_from_matrix(g)  # wxyz
        data["grasps"][f"grasp_{i}"] = {"confidence": float(extra["score"][i]), "position": [float(v) for v in g[:3, 3]],
                                        "orientation": {"w": float(q[0]), "xyz": [float(v) for v in q[1:]]},
                                        "pregrasp_cspace_position": {"right_gripper_finger_joint1": 0.05, "right_gripper_finger_joint2": 0.05},
                                        "ispatialgym": {k: float(v[i]) for k, v in extra.items()}}
    yaml.safe_dump(data, open(out_path, "w"), sort_keys=False)


results = {}
for obj in json.load(open(f"{OUT}/objects_exported.json")):
    name = obj["name"]; t0 = time.time()
    pieces, union = load_pieces(obj["obj"])
    t_load = time.time() - t0
    t1 = time.time()
    grasps = sampler_utils.sample_antipodal(union, num_candidates=N_CAND, num_orientations=N_ORIENT, gripper_maximum_aperture=APERTURE,
                                            gripper_standoff_fingertips=CONTACT_Z, gripper_approach_direction=np.array([0, 0, 1.0]),
                                            grasp_align_axis=np.array([0, 1.0, 0]), orientation_sample_axis=np.array([0, 1.0, 0]), random_seed=SEED)
    T = np.array(grasps); t_sample = time.time() - t1
    n0 = len(T)
    t2 = time.time()
    # (a) approach from above
    a = T[:, :3, 2]
    keep = a[:, 2] <= -np.cos(np.radians(MAX_TILT_DEG))
    T = T[keep]; n_a = len(T)
    # (b) width along the closing axis at the pad centre
    centers = T[:, :3, 3] + T[:, :3, 2] * CONTACT_Z
    width, antipodal = object_width_along(union, centers, T[:, :3, 1])
    keep = (width >= MIN_WIDTH) & (width <= APERTURE) & antipodal
    T, width = T[keep], width[keep]; n_b = len(T)
    # (c) table clearance: if the fingertips would go below the support plane (z_min in the object frame), retreat the
    #     gripper along its approach axis as long as the contact stays on the finger pads (pad tip at z_e = +0.012)
    Pz = np.einsum("nij,kj->nki", T[:, :3, :3], SPH)[:, :, 2] + T[:, None, 2, 3]   # sphere z (N,K)
    table = (Pz - RAD[None]).min(axis=1) - obj["z_min"]
    a = T[:, :3, 2]
    deficit = np.maximum(0.0, TABLE_CLEAR - table)
    s = deficit / np.maximum(-a[:, 2], 1e-3)                # retreat distance along -approach
    contact_z = CONTACT_Z + s                                # where the contact lands on the pad after retreating
    keep = contact_z <= 0.012                                # contact no further out than the tip-sphere centre (tip surface at +0.017)
    T, width, s = T[keep], width[keep], s[keep]
    T[:, :3, 3] -= a[keep] * s[:, None]
    # (d) object clearance with the open gripper (spheres) and re-check the table
    if len(T):
        clear, P = sphere_clearance(pieces, union, T)
        table = (P[:, :, 2] - RAD[None]).min(axis=1) - obj["z_min"]
    else:
        clear = table = np.zeros(0)
    keep = (clear >= OBJ_MARGIN) & (table >= TABLE_CLEAR - 1e-4)
    T, width, clear, table = T[keep], width[keep], clear[keep], table[keep]; n_c = len(T)
    t_filter = time.time() - t2
    # score: object clearance (dominant), verticality, centred width
    vert = -T[:, :3, 2][:, 2]
    score = clear + 0.02 * vert + 0.5 * np.minimum(clear, 0.01)
    sel = nms(T, score, NMS_POS, NMS_ROT_DEG) if len(T) else []
    T5, extra = T[sel], {"score": score[sel], "width": width[sel], "clearance": clear[sel], "table_clearance": table[sel], "tilt_deg": np.degrees(np.arccos(np.clip(vert[sel], -1, 1)))}
    to_isaac_grasp(T5, extra, obj, f"{OUT}/a1_{name}.yaml")
    np.savez(f"{OUT}/a1_{name}_all.npz", T=T, width=width, clearance=clear, score=score)
    total = time.time() - t0
    results[name] = {"n_sampled": n0, "n_after_retreat": int(len(s)), "n_after_approach": n_a, "n_after_width": n_b, "n_after_collision": n_c, "n_selected": len(sel),
                     "t_load_s": round(t_load, 3), "t_sample_s": round(t_sample, 3), "t_filter_s": round(t_filter, 3), "t_total_s": round(total, 3),
                     "faces": int(len(union.faces)), "selected": [{k: round(float(v[i]), 4) for k, v in extra.items()} for i in range(len(sel))]}
    print(f"{name:28s} sampled={n0:5d} above={n_a:5d} width_ok={n_b:5d} coll_free={n_c:5d} -> {len(sel)} | sample {t_sample*1000:6.0f} ms, filter {t_filter*1000:6.0f} ms, total {total:5.2f} s")
json.dump(results, open(f"{OUT}/a1_results.json", "w"), indent=1)
print("A1 DONE")
