#!/usr/bin/env python3
"""Textured renders of BEHAVIOR-1K object models without OmniGibson (usd-core + trimesh + pyrender/EGL, ~0.3 GB of GPU).

Four views per model around the *canonical* object frame (the frame the sampler places objects in): from +x, from -x,
from -y (all 25 deg above the horizon) and top-down (+z, image up = +y). An RGB axis gizmo (x red, y green, z blue) is
drawn at the bbox centre so that ``front_axis`` / ``long_axis`` / the thin axis can be read off the pictures.

Usage (base conda env, which has pxr):
    python render_models.py --models candidates.json --out ../renders [--rows 6] [--size 256]
``candidates.json`` = [{"category": ..., "model": ...}, ...] (or {"category": [models]}). Output: one PNG strip per
model (``<out>/strips/<category>__<model>.png``) and per-category grids (``<out>/grids/<category>_<k>.png``, ``rows``
models per grid) for review.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import trimesh
from cryptography.fernet import Fernet
from PIL import Image, ImageDraw, ImageFont
from pxr import Usd, UsdGeom, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ASSETS = os.path.join(REPO, "Behavior-1K", "datasets", "behavior-1k-assets", "objects")
KEY = open(os.path.join(REPO, "Behavior-1K", "datasets", "omnigibson.key"), "rb").read()
FERNET = Fernet(KEY)

VIEWS = [("from +x", (1.0, 0.0)), ("from -x", (-1.0, 0.0)), ("from -y", (0.0, -1.0)), ("top (+z, up=+y)", None)]


def decrypt_usd(category, model, tmpdir):
    src = f"{ASSETS}/{category}/{model}/usd/{model}.encrypted.usd"
    dst = os.path.join(tmpdir, f"{model}.usd")
    with open(src, "rb") as f:
        data = FERNET.decrypt(f.read())
    with open(dst, "wb") as f:
        f.write(data)
    return dst


def triangulate(counts, indices):
    """Fan triangulation -> (triangles of vertex indices, triangles of face-corner indices)."""
    tv, tc = [], []
    i = 0
    for c in counts:
        for k in range(1, c - 1):
            tv.append((indices[i], indices[i + k], indices[i + k + 1]))
            tc.append((i, i + k, i + k + 1))
        i += c
    return np.asarray(tv, dtype=np.int64).reshape(-1, 3), np.asarray(tc, dtype=np.int64).reshape(-1, 3)


def diffuse_texture(prim, category, model):
    """Diffuse texture path of a mesh prim: the bound material's texture input, else ``material/<model>__<link>__diffuse.png``."""
    mdir = f"{ASSETS}/{category}/{model}/material"
    try:
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat:
            for shader_prim in Usd.PrimRange(mat.GetPrim()):
                for attr in shader_prim.GetAttributes():
                    n = attr.GetName()
                    if n in ("inputs:diffuse_texture", "inputs:albedo_texture", "inputs:file") and attr.GetTypeName() == "asset":
                        v = attr.Get()
                        if v is None:
                            continue
                        p = v.resolvedPath or v.path
                        if p and os.path.exists(p):
                            return p
                        base = os.path.basename(p or "")
                        if base and os.path.exists(os.path.join(mdir, base)):
                            return os.path.join(mdir, base)
    except Exception:
        pass
    parts = str(prim.GetPath()).split("/")
    link = parts[2] if len(parts) > 2 else "base_link"
    cand = os.path.join(mdir, f"{model}__{link}__diffuse.png")
    if os.path.exists(cand):
        return cand
    if os.path.isdir(mdir):
        for fn in sorted(os.listdir(mdir)):
            if "diffuse" in fn.lower():
                return os.path.join(mdir, fn)
    return None


def load_model(category, model, tmpdir):
    """List of trimesh.Trimesh (world frame of the USD = canonical object frame), textured where possible."""
    usd = decrypt_usd(category, model, tmpdir)
    stage = Usd.Stage.Open(usd)
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    meshes = []
    tex_cache = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if not path.startswith(f"/{model}/") or "/visuals/" not in path:
            continue
        low = path.lower()
        if "collision" in low or "/meta__" in low:
            continue
        if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        if not UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.inherited:
            continue
        try:  # glass parts (OmniGlass material) are transparent in Isaac Sim; pyrender would draw them opaque black
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if mat and "glass" in str(mat.GetPath()).lower():
                continue
        except Exception:
            pass
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        m = xcache.GetLocalToWorldTransform(prim)
        M = np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=np.float64)
        V = np.asarray(pts, dtype=np.float64)
        V = (np.c_[V, np.ones(len(V))] @ M)[:, :3]
        counts, idx = list(counts), list(idx)
        tv, tc = triangulate(counts, idx)
        if len(tv) == 0:
            continue
        uv = None
        interp = None
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if pv and pv.HasValue():
            try:
                st = np.asarray(pv.ComputeFlattened(), dtype=np.float64)
                interp = pv.GetInterpolation()
                if interp == "faceVarying" and len(st) == len(idx):
                    uv = st
                elif interp == "vertex" and len(st) == len(V):
                    uv = st
                else:
                    uv = None
            except Exception:
                uv = None
        tex = diffuse_texture(prim, category, model)
        tm = None
        if uv is not None and tex is not None:
            if tex not in tex_cache:
                try:
                    tex_cache[tex] = Image.open(tex).convert("RGB")
                except Exception:
                    tex_cache[tex] = None
            img = tex_cache[tex]
            if img is not None:
                if interp == "faceVarying":
                    Vn = V[tv.reshape(-1)]
                    UVn = uv[tc.reshape(-1)]
                    Fn = np.arange(len(Vn)).reshape(-1, 3)
                else:
                    Vn, UVn, Fn = V, uv, tv
                tm = trimesh.Trimesh(vertices=Vn, faces=Fn, process=False)
                tm.visual = trimesh.visual.TextureVisuals(uv=UVn, image=img)
        if tm is None:
            tm = trimesh.Trimesh(vertices=V, faces=tv, process=False)
            tm.visual.face_colors = np.array([190, 190, 200, 255], dtype=np.uint8)
        # a flipped transform (negative determinant) turns the winding inside out
        if np.linalg.det(M[:3, :3]) < 0:
            tm.invert()
        meshes.append(tm)
    return meshes


def look_at(eye, target, up):
    eye, target, up = np.asarray(eye, float), np.asarray(target, float), np.asarray(up, float)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, [1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, u, -f, eye
    return T


def axis_gizmo(centre, length, radius):
    out = []
    for axis, col in ((0, [230, 40, 40, 255]), (1, [40, 190, 40, 255]), (2, [40, 80, 240, 255])):
        cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
        cyl.apply_translation([0, 0, length / 2])  # from the origin along +z
        if axis == 0:
            cyl.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
        elif axis == 1:
            cyl.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
        cone = trimesh.creation.cone(radius=2.2 * radius, height=4 * radius, sections=12)
        cone.apply_translation([0, 0, length])
        if axis == 0:
            cone.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
        elif axis == 1:
            cone.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
        for g in (cyl, cone):
            g.apply_translation(centre)
            g.visual.face_colors = np.array(col, dtype=np.uint8)
            out.append(g)
    return out


class Renderer:
    def __init__(self, size=256):
        import pyrender

        self.pyrender = pyrender
        self.size = size
        self.r = pyrender.OffscreenRenderer(size, size)

    def render_views(self, meshes, gizmo):
        pyrender = self.pyrender
        allv = np.concatenate([m.vertices for m in meshes])
        lo, hi = allv.min(0), allv.max(0)
        c = (lo + hi) / 2
        ext = hi - lo
        s = float(np.linalg.norm(ext))
        d = 1.55 * s + 0.03
        imgs = []
        for name, horiz in VIEWS:
            scene = pyrender.Scene(ambient_light=[0.45, 0.45, 0.45], bg_color=[1.0, 1.0, 1.0, 1.0])
            for m in meshes:
                try:
                    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
                except Exception:
                    mm = m.copy()
                    mm.visual = trimesh.visual.ColorVisuals(mm, face_colors=[190, 190, 200, 255])
                    scene.add(pyrender.Mesh.from_trimesh(mm, smooth=False))
            for g in gizmo:
                scene.add(pyrender.Mesh.from_trimesh(g, smooth=False))
            if horiz is None:
                eye = c + np.array([0.0, -1e-4, d])
                pose = look_at(eye, c, [0.0, 1.0, 0.0])
            else:
                el = math.radians(25.0)
                eye = c + np.array([horiz[0] * d * math.cos(el), horiz[1] * d * math.cos(el), d * math.sin(el)])
                pose = look_at(eye, c, [0.0, 0.0, 1.0])
            cam = pyrender.PerspectiveCamera(yfov=math.radians(38.0), znear=0.005, zfar=50.0)
            scene.add(cam, pose=pose)
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.6), pose=pose)
            # a second light from above-left so that top faces are not flat
            lp = look_at(c + np.array([-0.5 * d, -0.5 * d, 1.2 * d]), c, [0, 0, 1])
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.4), pose=lp)
            color, _ = self.r.render(scene, flags=pyrender.RenderFlags.FLAT if False else 0)
            imgs.append((name, Image.fromarray(color[..., :3].copy())))
        return imgs, ext, c


def font(size=13):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def strip_image(category, model, imgs, ext, extra=""):
    size = imgs[0][1].size[0]
    W, H = size * len(imgs), size + 38
    out = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(out)
    f = font(13)
    hdr = f"{category}/{model}   bbox x {ext[0]:.3f}  y {ext[1]:.3f}  z {ext[2]:.3f} m   {extra}"
    dr.text((4, 3), hdr, fill=(0, 0, 0), font=f)
    for i, (name, im) in enumerate(imgs):
        out.paste(im, (i * size, 38))
        dr.text((i * size + 4, 21), name, fill=(60, 60, 60), font=font(12))
    dr.line([(0, H - 1), (W, H - 1)], fill=(150, 150, 150), width=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="json: list of {category, model} or {category: [models]}")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "renders"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--rows", type=int, default=6, help="models per grid image")
    ap.add_argument("--only", default=None, help="comma separated categories")
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    with open(args.models) as f:
        spec = json.load(f)
    if isinstance(spec, dict):
        models = [{"category": c, "model": m} for c, ms in spec.items() for m in ms]
    else:
        models = spec
    if args.only:
        keep = set(args.only.split(","))
        models = [m for m in models if m["category"] in keep]
    strips_dir = os.path.join(args.out, "strips")
    grids_dir = os.path.join(args.out, "grids")
    os.makedirs(strips_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)
    R = Renderer(args.size)
    meta = {}
    meta_path = os.path.join(args.out, "render_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    t0 = time.time()
    by_cat = {}
    with tempfile.TemporaryDirectory() as tmp:
        for i, m in enumerate(models):
            cat, mid = m["category"], m["model"]
            key = f"{cat}__{mid}"
            path = os.path.join(strips_dir, key + ".png")
            by_cat.setdefault(cat, []).append((mid, path))
            if args.skip_done and os.path.exists(path) and key in meta:
                continue
            try:
                meshes = load_model(cat, mid, tmp)
                if not meshes:
                    raise RuntimeError("no visual mesh")
                allv = np.concatenate([x.vertices for x in meshes])
                lo, hi = allv.min(0), allv.max(0)
                ext = hi - lo
                gizmo = axis_gizmo((lo + hi) / 2, 0.62 * float(ext.max()), max(0.0015, 0.012 * float(ext.max())))
                imgs, ext, c = R.render_views(meshes, gizmo)
                n_tex = sum(1 for x in meshes if isinstance(x.visual, trimesh.visual.TextureVisuals))
                strip_image(cat, mid, imgs, ext, f"meshes {len(meshes)} (textured {n_tex})").save(path)
                meta[key] = {"category": cat, "model": mid, "bbox": [round(float(v), 4) for v in ext], "n_meshes": len(meshes), "n_textured": n_tex,
                             "thin_axis": "xyz"[int(np.argmin(ext))], "long_axis": "xyz"[int(np.argmax(ext))]}
                print(f"[{i + 1}/{len(models)}] {key} bbox {np.round(ext, 3).tolist()} ({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"FAILED {key}: {e!r}", flush=True)
                traceback.print_exc()
                meta[key] = {"category": cat, "model": mid, "error": repr(e)[:300]}
            if (i + 1) % 20 == 0:
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=1)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    # grids
    for cat, items in by_cat.items():
        items = [(mid, p) for mid, p in items if os.path.exists(p)]
        for k in range(0, len(items), args.rows):
            chunk = items[k:k + args.rows]
            strips = [Image.open(p) for _, p in chunk]
            W = max(s.size[0] for s in strips)
            H = sum(s.size[1] for s in strips)
            grid = Image.new("RGB", (W, H), "white")
            y = 0
            for s in strips:
                grid.paste(s, (0, y))
                y += s.size[1]
            grid.save(os.path.join(grids_dir, f"{cat}_{k // args.rows:02d}.png"))
    print(f"done {len(models)} models in {time.time() - t0:.0f}s; grids in {grids_dir}")


if __name__ == "__main__":
    main()
