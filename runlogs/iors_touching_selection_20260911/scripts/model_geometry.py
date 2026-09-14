#!/usr/bin/env python3
"""Geometry / duplicate signatures of the candidate IORS-4 (touching) models, read from the BEHAVIOR-1K USDs without a
simulator (same recipe as runlogs/grasp_library_20260905/select/partners/partner_geometry.py and select/duplicates.md):

* ``bbox`` = extents of the visual meshes in the canonical object frame, ``z_min``,
* ``vertex_hash`` = md5 of the sorted vertex distances to the centroid (rounded to 1e-6) -> same geometry,
* ``texture_md5`` = md5 of the diffuse / albedo textures -> exact duplicate when both agree,
* ``base_contact_fraction`` = fraction of the footprint area (convex hull of the xy projection) covered by the convex hull
  of the vertices within 5 mm of the bottom: ~1 for boxes / cans (flat base), small for rounded or footed bottoms
  (an object with a small base tips or rolls when pushed).

Usage: python model_geometry.py --models render_models.json  ->  model_geometry.json (idempotent per model).
Run with the base conda env (usd-core, cryptography, numpy, scipy).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile

import numpy as np
from cryptography.fernet import Fernet
from pxr import Usd, UsdGeom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
ASSETS = os.path.join(REPO, "Behavior-1K", "datasets", "behavior-1k-assets", "objects")
KEY = open(os.path.join(REPO, "Behavior-1K", "datasets", "omnigibson.key"), "rb").read()
FERNET = Fernet(KEY)
OUT = os.path.join(ROOT, "model_geometry.json")


def decrypt_usd(category, model, tmpdir):
    src = f"{ASSETS}/{category}/{model}/usd/{model}.encrypted.usd"
    dst = os.path.join(tmpdir, f"{model}.usd")
    with open(src, "rb") as f:
        data = FERNET.decrypt(f.read())
    with open(dst, "wb") as f:
        f.write(data)
    return dst


def mesh_points(prim, xcache):
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        return np.zeros((0, 3))
    m = xcache.GetLocalToWorldTransform(prim)
    M = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
    v = np.asarray(pts, dtype=np.float64)
    return (np.c_[v, np.ones(len(v))] @ M)[:, :3]


def hull_area(xy):
    from scipy.spatial import ConvexHull
    if len(xy) < 3:
        return 0.0
    try:
        return float(ConvexHull(xy).volume)  # 2-D hull "volume" = area
    except Exception:
        return 0.0


def analyse(usd_path, model):
    stage = Usd.Stage.Open(usd_path)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root = f"/{model}/"
    body, meta_types = [], set()
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(root):
            continue
        if prim.HasAttribute("ig:isMetaLink") and prim.GetAttribute("ig:isMetaLink").Get():
            meta_types.add(str(prim.GetAttribute("ig:metaLinkType").Get()))
        if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in path or "/meta__" in path:
            continue
        pts = mesh_points(prim, xcache)
        if len(pts):
            body.append(pts)
    if not body:
        raise RuntimeError("no visual mesh")
    V = np.concatenate(body)
    lo, hi = V.min(0), V.max(0)
    ext = hi - lo
    bottom = V[V[:, 2] <= lo[2] + 0.005]
    foot_area = hull_area(V[:, :2])
    base_area = hull_area(bottom[:, :2])
    return {
        "bbox": [round(float(x), 4) for x in ext],
        "z_min": round(float(lo[2]), 4),
        "n_vertices": int(len(V)),
        "vertex_hash": hashlib.md5(np.round(np.sort(np.linalg.norm(V - V.mean(0), axis=1)), 6).tobytes()).hexdigest(),
        "meta_link_types": sorted(meta_types),
        "base_contact_fraction": round(base_area / foot_area, 3) if foot_area > 0 else None,
        "base_hull_width": round(float(np.ptp(bottom[:, 0])), 4) if len(bottom) else 0.0,
        "base_hull_depth": round(float(np.ptp(bottom[:, 1])), 4) if len(bottom) else 0.0,
    }


def texture_md5(category, model):
    mdir = f"{ASSETS}/{category}/{model}/material"
    out = []
    if not os.path.isdir(mdir):
        return out
    for fn in sorted(os.listdir(mdir)):
        if "diffuse" in fn.lower() or "albedo" in fn.lower():
            out.append(hashlib.md5(open(os.path.join(mdir, fn), "rb").read()).hexdigest())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    models = json.load(open(args.models))
    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, m in enumerate(models):
            key = f"{m['category']}/{m['model']}"
            if key in results and "error" not in results[key]:
                continue
            rec = {"category": m["category"], "model": m["model"]}
            try:
                usd = decrypt_usd(m["category"], m["model"], tmp)
                rec.update(analyse(usd, m["model"]))
                rec["texture_md5"] = texture_md5(m["category"], m["model"])
                os.remove(usd)
            except Exception as e:  # noqa: BLE001
                rec["error"] = repr(e)
            results[key] = rec
            if (i + 1) % 50 == 0:
                json.dump(results, open(args.out, "w"), indent=1)
                print(f"{i + 1}/{len(models)}", flush=True)
    json.dump(results, open(args.out, "w"), indent=1)
    n_err = sum("error" in r for r in results.values())
    print(f"done: {len(results)} models ({n_err} errors) -> {args.out}")


if __name__ == "__main__":
    main()
