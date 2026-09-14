"""Convert GraspGen (Panda) grasps to R1Pro right_eef_link poses and select top-5 (raw, and table-constrained)."""
import json, sys
import numpy as np
import trimesh
import trimesh.transformations as tra
import yaml

OUT = "/tmp/grasp_exp"
PANDA_DEPTH = 0.10527314                       # GraspGen franka_panda.yaml: base link -> TCP along +z
gripper = json.load(open(f"{OUT}/r1pro_right_gripper_eef.json"))
SPH = np.array([s["center"] for s in gripper["spheres_open"]]); RAD = np.array([s["radius"] for s in gripper["spheres_open"]])
CONTACT_Z = gripper["contact_center_z"]
MAX_TILT_DEG, TABLE_CLEAR, TOPK = 60.0, 0.0, 5
# GraspGen convention: approach +z, fingers close along x (verified by ray-cast widths). Ours: approach +z, close along y.
T_CONV = tra.rotation_matrix(np.pi / 2, [0, 0, 1]) @ tra.translation_matrix([0, 0, PANDA_DEPTH - CONTACT_Z])


def to_isaac_grasp(T, conf, extra, obj, source, out_path):
    data = {"format": "isaac_grasp", "format_version": 1.0, "object_frame": "base_link", "gripper_frame": "right_eef_link",
            "ispatialgym": {"object": {"category": obj["category"], "model": obj["model"]}, "gripper": "r1pro_right", "source": source}, "grasps": {}}
    for i, g in enumerate(T):
        q = tra.quaternion_from_matrix(g)
        data["grasps"][f"grasp_{i}"] = {"confidence": float(conf[i]), "position": [float(v) for v in g[:3, 3]], "orientation": {"w": float(q[0]), "xyz": [float(v) for v in q[1:]]},
                                        "pregrasp_cspace_position": {"right_gripper_finger_joint1": 0.05, "right_gripper_finger_joint2": 0.05},
                                        "ispatialgym": {k: float(v[i]) for k, v in extra.items()}}
    yaml.safe_dump(data, open(out_path, "w"), sort_keys=False)


def tilt_deg(T):
    return np.degrees(np.arccos(np.clip(-T[:, :3, 2][:, 2], -1, 1)))


summary = {}
for obj in json.load(open(f"{OUT}/objects_exported.json")):
    name = obj["name"]
    # --- raw: top-5 by confidence (400 samples -> top 100), no constraints
    d = np.load(f"{OUT}/a2_franka_panda_{name}_raw.npz"); T = np.einsum("nij,jk->nik", d["T"], T_CONV); conf = d["conf"]
    sel = np.argsort(-conf)[:TOPK]
    to_isaac_grasp(T[sel], conf[sel], {"tilt_deg": tilt_deg(T[sel])}, obj, "graspgen_franka_panda_raw_top5", f"{OUT}/a2raw_{name}.yaml")
    raw_from_above = int((tilt_deg(T) <= MAX_TILT_DEG).sum())
    # --- table-constrained: 2000 samples -> top 500, approach within 60 deg of vertical, fingers above the support plane
    d = np.load(f"{OUT}/a2_franka_panda_{name}_raw_big.npz"); Tb = np.einsum("nij,jk->nik", d["T"], T_CONV); confb = d["conf"]
    tilt = tilt_deg(Tb); keep = tilt <= MAX_TILT_DEG
    Tb, confb, tilt = Tb[keep], confb[keep], tilt[keep]; n_above = len(Tb)
    if len(Tb):
        Pz = np.einsum("nij,kj->nki", Tb[:, :3, :3], SPH)[:, :, 2] + Tb[:, None, 2, 3]
        table = (Pz - RAD[None]).min(axis=1) - obj["z_min"]
        a = Tb[:, :3, 2]; s = np.maximum(0.0, TABLE_CLEAR - table) / np.maximum(-a[:, 2], 1e-3)
        keep = (CONTACT_Z + s) <= 0.012
        Tb, confb, tilt, s = Tb[keep], confb[keep], tilt[keep], s[keep]
        Tb[:, :3, 3] -= a[keep] * s[:, None]
    n_table = len(Tb)
    sel = np.argsort(-confb)[:TOPK]
    to_isaac_grasp(Tb[sel], confb[sel], {"tilt_deg": tilt[sel], "retreat_m": s[sel] if len(Tb) else np.zeros(0)}, obj, "graspgen_franka_panda_table_top5", f"{OUT}/a2tab_{name}.yaml")
    summary[name] = {"raw_top100_from_above": raw_from_above, "big_top500_from_above": n_above, "big_table_ok": n_table, "raw_top5_tilt_deg": [round(float(x), 1) for x in tilt_deg(T[np.argsort(-conf)[:TOPK]])],
                     "tab_top5_conf": [round(float(x), 3) for x in confb[sel]]}
    print(f"{name:28s} raw top-100 from above: {raw_from_above:3d} | big top-500 from above: {n_above:3d}, table-ok: {n_table:3d} | raw top-5 tilt(deg): {summary[name]['raw_top5_tilt_deg']}")
json.dump(summary, open(f"{OUT}/a2_convert_summary.json", "w"), indent=1)
print("A2 CONVERT DONE")
