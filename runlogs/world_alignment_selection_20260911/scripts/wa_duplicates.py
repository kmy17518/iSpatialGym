#!/usr/bin/env python3
"""Geometry-duplicate check for the world-alignment models (same method as select/duplicates.md, geometry tiers only).

Signature per model (visual meshes of the decrypted USD, object frame): vertex / face counts, area, volume, sorted
extents, D2 histogram of 1500 surface samples, rotation-invariant vertex hash; plus md5 of every diffuse texture.
Pairs within a category: ``identical`` (same vertex hash), ``same_shape`` (extents <= 2 %, volume <= 1 %, area <= 2 %,
D2 L1 <= 0.06). A pair is an **exact duplicate** when it is identical / same_shape AND every diffuse texture md5
matches; otherwise a **texture variant** (same geometry, different diffuse maps). Clusters = connected components.

Usage (base env, has pxr):  python wa_duplicates.py --models selection.json --out duplicates.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np
from cryptography.fernet import Fernet
from pxr import Usd, UsdGeom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))
ASSETS = os.path.join(REPO, "Behavior-1K", "datasets", "behavior-1k-assets", "objects")
FERNET = Fernet(open(os.path.join(REPO, "Behavior-1K", "datasets", "omnigibson.key"), "rb").read())


def decrypt_usd(category, model, tmpdir):
    src = f"{ASSETS}/{category}/{model}/usd/{model}.encrypted.usd"
    dst = os.path.join(tmpdir, f"{model}.usd")
    with open(dst, "wb") as f:
        f.write(FERNET.decrypt(open(src, "rb").read()))
    return dst


def triangulate(counts, indices):
    tris = []
    i = 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((indices[i], indices[i + k], indices[i + k + 1]))
        i += c
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def visual_meshes(usd_path, model):
    stage = Usd.Stage.Open(usd_path)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if not path.startswith(f"/{model}/") or "/visuals/" not in path or "collision" in path.lower() or "/meta__" in path.lower():
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts, counts, idx = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        m = xcache.GetLocalToWorldTransform(prim)
        M = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
        v = np.asarray(pts, dtype=np.float64)
        v = (np.c_[v, np.ones(len(v))] @ M)[:, :3]
        yield path, v, triangulate(list(counts), list(idx))


def signature(meshes):
    V = np.concatenate([v for _, v, _ in meshes])
    F, off = [], 0
    for _, v, f in meshes:
        F.append(f + off)
        off += len(v)
    F = np.concatenate(F)
    tri = V[F]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    tri_area = 0.5 * np.linalg.norm(cross, axis=1)
    area = float(tri_area.sum())
    volume = float(abs((tri[:, 0] * cross).sum() / 6.0))
    rng = np.random.default_rng(0)
    n_s = 1500
    p = tri_area / tri_area.sum() if tri_area.sum() > 0 else None
    fi = rng.choice(len(tri), size=n_s, p=p)
    r1, r2 = rng.random(n_s), rng.random(n_s)
    s1 = np.sqrt(r1)
    P = (1 - s1)[:, None] * tri[fi, 0] + (s1 * (1 - r2))[:, None] * tri[fi, 1] + (s1 * r2)[:, None] * tri[fi, 2]
    X = P - P.mean(0)
    ext = V.max(0) - V.min(0)
    diag = float(np.linalg.norm(ext)) or 1.0
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)[np.triu_indices(n_s, 1)]
    hist, _ = np.histogram(d / diag, bins=24, range=(0.0, 1.0))
    hist = (hist / max(1, hist.sum())).round(4).tolist()
    return {"n_vertices": int(len(V)), "n_faces": int(len(F)), "area": round(area, 7), "volume": round(volume, 9),
            "sorted_extent": sorted(round(float(e), 5) for e in ext),
            "vertex_hash": hashlib.md5(np.round(np.sort(np.linalg.norm(V - V.mean(0), axis=1)), 6).tobytes()).hexdigest(), "d2_hist": hist}


def texture_md5s(category, model):
    mdir = f"{ASSETS}/{category}/{model}/material"
    out = {}
    if os.path.isdir(mdir):
        for fn in sorted(os.listdir(mdir)):
            if "diffuse" in fn.lower():
                out[fn.split("__", 1)[-1]] = hashlib.md5(open(os.path.join(mdir, fn), "rb").read()).hexdigest()
    return out


def rel(a, b):
    m = max(abs(a), abs(b))
    return abs(a - b) / m if m > 0 else 0.0


def pair_tier(a, b):
    if a["vertex_hash"] == b["vertex_hash"]:
        return "identical"
    ext = max(rel(x, y) for x, y in zip(a["sorted_extent"], b["sorted_extent"]))
    d2 = float(np.abs(np.array(a["d2_hist"]) - np.array(b["d2_hist"])).sum())
    if ext <= 0.02 and rel(a["volume"], b["volume"]) <= 0.01 and rel(a["area"], b["area"]) <= 0.02 and d2 <= 0.06:
        return "same_shape"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=os.path.join(ROOT, "selection.json"), help="selection.json (valid models of the selected categories) or {category: [models]}")
    ap.add_argument("--out", default=os.path.join(ROOT, "duplicates.json"))
    args = ap.parse_args()
    with open(args.models) as f:
        sel = json.load(f)
    models = defaultdict(set)
    if "tasks" in sel:
        for t in sel["tasks"].values():
            for rnd in t["rounds"].values():
                for r in rnd:
                    for m in r["valid_models"]:
                        models[r["category"]].add(m)
    else:
        for c, ms in sel.items():
            models[c] |= set(ms)
    sigs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for c in sorted(models):
            for m in sorted(models[c]):
                try:
                    meshes = list(visual_meshes(decrypt_usd(c, m, tmp), m))
                    if not meshes:
                        raise RuntimeError("no visual mesh")
                    s = signature(meshes)
                    s["textures"] = texture_md5s(c, m)
                    sigs[f"{c}/{m}"] = s
                except Exception as e:
                    sigs[f"{c}/{m}"] = {"error": repr(e)[:200]}
    clusters = []
    flags = {}
    for c in sorted(models):
        ms = sorted(m for m in models[c] if "error" not in sigs.get(f"{c}/{m}", {"error": 1}))
        parent = {m: m for m in ms}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        pairs = {}
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = sigs[f"{c}/{ms[i]}"], sigs[f"{c}/{ms[j]}"]
                tier = pair_tier(a, b)
                if tier is None:
                    continue
                same_tex = a["textures"] == b["textures"] and bool(a["textures"])
                pairs[(ms[i], ms[j])] = {"tier": tier, "same_textures": same_tex}
                parent[find(ms[i])] = find(ms[j])
        groups = defaultdict(list)
        for m in ms:
            groups[find(m)].append(m)
        for k, (root, members) in enumerate(sorted(groups.items())):
            if len(members) < 2:
                continue
            exact_parent = {m: m for m in members}

            def efind(x):
                while exact_parent[x] != x:
                    exact_parent[x] = exact_parent[exact_parent[x]]
                    x = exact_parent[x]
                return x

            for (a, b), info in pairs.items():
                if a in members and info["same_textures"]:
                    exact_parent[efind(a)] = efind(b)
            eg = defaultdict(list)
            for m in members:
                eg[efind(m)].append(m)
            exact_groups = [sorted(v) for v in eg.values() if len(v) > 1]
            cl = {"id": f"{c}__wa{k + 1}", "category": c, "models": sorted(members), "exact_groups": exact_groups,
                  "pairs": [{"a": a, "b": b, **info} for (a, b), info in pairs.items() if a in members]}
            clusters.append(cl)
            for m in members:
                ex = [g for g in exact_groups if m in g]
                others_exact = sorted(x for x in (ex[0] if ex else []) if x != m)
                others_geom = sorted(x for x in members if x != m and x not in others_exact)
                flags[f"{c}/{m}"] = {"cluster": cl["id"], "exact": others_exact, "same_geometry": others_geom}
    with open(args.out, "w") as f:
        json.dump({"clusters": clusters, "flags": flags, "n_models": sum(len(v) for v in models.values()),
                   "errors": {k: v["error"] for k, v in sigs.items() if "error" in v}}, f, indent=1)
    print(f"{sum(len(v) for v in models.values())} models, {len(clusters)} clusters, {sum(1 for f in flags.values() if f['exact'])} models with an exact duplicate, "
          f"{sum(1 for f in flags.values() if f['same_geometry'])} with a texture variant -> {args.out}")


if __name__ == "__main__":
    main()
