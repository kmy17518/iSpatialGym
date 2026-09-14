#!/usr/bin/env python3
"""Geometry of the candidate IORS partner models, read from the BEHAVIOR-1K USDs without a simulator.

For every *ready* model of the partner categories named in ``task_object_selection.csv`` (IORS-0 ontop, -1 under
small-under-small, -2 inside, -3 nextto) the encrypted USD is decrypted in a temporary directory and opened with
``usd-core`` (``pxr``):

* ``bbox``: extents of the visual meshes in the object frame (cross-checked against the knowledge base bbox),
* ``fillable`` (containers only): the ``fillable`` / ``openfillable`` meta-link volume that ``Inside`` evaluates
  (``RigidPrim.check_points_in_volume``: the target's AABB centre must lie in the union of these convex meshes):
  z range (``depth``), horizontal extents of the whole volume and of its top slice (the opening: min / max width over
  36 in-plane rotations), and the height of the volume floor above the object's bottom,
* ``vertex_hash`` / diffuse-texture md5 (same recipe as select/duplicates.md) to fold exact duplicates.

Output: ``partner_geometry.json`` next to this script (idempotent: models already present are skipped).
Run with a python that has ``usd-core``, ``cryptography`` and ``numpy`` (the base conda env on this machine).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile

import numpy as np
from cryptography.fernet import Fernet
from pxr import Usd, UsdGeom

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ASSETS = os.path.join(REPO, "Behavior-1K", "datasets", "behavior-1k-assets", "objects")
KEY = open(os.path.join(REPO, "Behavior-1K", "datasets", "omnigibson.key"), "rb").read()
CATEGORY_TABLE = os.path.join(REPO, "runlogs", "grasp_exp_20260904", "category_table.json")
OUT = os.path.join(HERE, "partner_geometry.json")

#: partner categories per IORS problem (small-under-big partners of -1 are scene furniture, handled by the table run)
PARTNER_CATEGORIES = {
    "inter_object_relation_satisfaction-0": ["cutting_board", "plate", "platter", "saucer"],
    "inter_object_relation_satisfaction-1": ["coaster", "platter", "saucer", "tray"],
    "inter_object_relation_satisfaction-2": ["beer_glass", "bowl", "hingeless_jar", "saucepot", "vase"],
    "inter_object_relation_satisfaction-3": ["canned_food", "carrot", "colored_pencil", "pepper_shaker", "ruler",
                                             "salt_shaker", "soup_ladle", "star_anise"],
}
CONTAINER_TYPES = ("fillable", "openfillable")
FERNET = Fernet(KEY)


def decrypt_usd(category: str, model: str, tmpdir: str) -> str:
    src = f"{ASSETS}/{category}/{model}/usd/{model}.encrypted.usd"
    dst = os.path.join(tmpdir, f"{model}.usd")
    with open(src, "rb") as f:
        data = FERNET.decrypt(f.read())
    with open(dst, "wb") as f:
        f.write(data)
    return dst


def mesh_points(prim, xcache) -> np.ndarray:
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return np.zeros((0, 3))
    m = xcache.GetLocalToWorldTransform(prim)
    M = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
    v = np.asarray(pts, dtype=np.float64)
    return (np.c_[v, np.ones(len(v))] @ M)[:, :3]


def horizontal_widths(xy: np.ndarray):
    """(min width, max width) of the 2-D point set over 36 in-plane rotations (rotating-calipers approximation)."""
    if len(xy) < 2:
        return 0.0, 0.0
    widths = []
    for ang in np.linspace(0, np.pi, 36, endpoint=False):
        c, s = np.cos(ang), np.sin(ang)
        u = xy @ np.array([c, s])
        widths.append(float(u.max() - u.min()))
    return min(widths), max(widths)


def analyse(usd_path: str, model: str) -> dict:
    stage = Usd.Stage.Open(usd_path)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root = f"/{model}/"
    body_pts, body_hash_pts, fill_pts = [], [], []
    n_fill_meshes = 0
    meta_types = set()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root):
            continue
        if prim.HasAttribute("ig:isMetaLink") and prim.GetAttribute("ig:isMetaLink").Get():
            meta_types.add(str(prim.GetAttribute("ig:metaLinkType").Get()))
        if not prim.IsA(UsdGeom.Mesh):
            continue
        is_meta = "/meta__" in path
        if "/visuals/" not in path:
            continue
        pts = mesh_points(prim, xcache)
        if len(pts) == 0:
            continue
        if is_meta:
            link = path[len(root):].split("/")[0]  # meta__<link>_<type>_<id>_<sub>_link
            parts = link.split("_")
            if any(t in parts for t in CONTAINER_TYPES):
                fill_pts.append(pts)
                n_fill_meshes += 1
            continue
        body_pts.append(pts)
    if not body_pts:
        raise RuntimeError("no visual mesh")
    V = np.concatenate(body_pts)
    lo, hi = V.min(0), V.max(0)
    out = {
        "bbox": [round(float(x), 4) for x in (hi - lo)],
        "z_min": round(float(lo[2]), 4),
        "n_vertices": int(len(V)),
        "vertex_hash": hashlib.md5(np.round(np.sort(np.linalg.norm(V - V.mean(0), axis=1)), 6).tobytes()).hexdigest(),
        "meta_link_types": sorted(meta_types),
    }
    if fill_pts:
        P = np.concatenate(fill_pts)
        z0, z1 = float(P[:, 2].min()), float(P[:, 2].max())
        depth = z1 - z0
        top = P[P[:, 2] >= z1 - max(0.01, 0.15 * depth)]
        bottom = P[P[:, 2] <= z0 + max(0.01, 0.15 * depth)]
        wmin, wmax = horizontal_widths(P[:, :2])
        omin, omax = horizontal_widths(top[:, :2])
        fmin, fmax = horizontal_widths(bottom[:, :2])
        out["fillable"] = {
            "n_meshes": n_fill_meshes,
            "depth": round(depth, 4),
            "floor_above_bottom": round(z0 - float(lo[2]), 4),
            "top_below_object_top": round(float(hi[2]) - z1, 4),
            "width_min": round(wmin, 4), "width_max": round(wmax, 4),
            "opening_min": round(omin, 4), "opening_max": round(omax, 4),
            "floor_min": round(fmin, 4), "floor_max": round(fmax, 4),
            "centre_xy": [round(float(x), 4) for x in P[:, :2].mean(0)],
        }
    return out


def texture_md5(category: str, model: str):
    mdir = f"{ASSETS}/{category}/{model}/material"
    out = []
    if not os.path.isdir(mdir):
        return out
    for fn in sorted(os.listdir(mdir)):
        if "diffuse" in fn.lower() or "albedo" in fn.lower():
            out.append(hashlib.md5(open(os.path.join(mdir, fn), "rb").read()).hexdigest())
    return out


def main():
    ct = json.load(open(CATEGORY_TABLE))["objects"]
    wanted = sorted({c for cats in PARTNER_CATEGORIES.values() for c in cats})
    models = [o for o in ct.values() if o["category"] in wanted]
    results = json.load(open(OUT)) if os.path.exists(OUT) else {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, o in enumerate(sorted(models, key=lambda o: (o["category"], o["model"]))):
            key = f"{o['category']}/{o['model']}"
            if key in results and "error" not in results[key]:
                continue
            rec = {"category": o["category"], "model": o["model"], "ready": bool(o["ready"]), "complaints": o["complaints"],
                   "kb_bbox": o["bbox"]}
            try:
                usd = decrypt_usd(o["category"], o["model"], tmp)
                rec.update(analyse(usd, o["model"]))
                rec["texture_md5"] = texture_md5(o["category"], o["model"])
                os.remove(usd)
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)
            results[key] = rec
            if (i + 1) % 25 == 0:
                json.dump(results, open(OUT, "w"), indent=1)
                print(f"{i + 1}/{len(models)}", flush=True)
    json.dump(results, open(OUT, "w"), indent=1)
    n_err = sum("error" in r for r in results.values())
    print(f"done: {len(results)} models ({n_err} errors) -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
