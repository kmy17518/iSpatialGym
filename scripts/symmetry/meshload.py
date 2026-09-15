#!/usr/bin/env python3
"""Offline loader for BEHAVIOR-1K object models without OmniGibson (usd-core + Fernet-decrypted USDs + trimesh).

Factored out of ``runlogs/world_alignment_selection_20260911/scripts/render_models.py`` (same decryption, prim
filtering, triangulation and diffuse-texture lookup) so that the symmetry survey and later offline tools share one
loader. Runs in the ``base`` conda env (which has ``pxr``); no GPU needed.

    from meshload import load_model, merged_geometry
    meshes = load_model("bowl", "belcml")           # list of trimesh.Trimesh in the canonical object frame, textured
    geom = merged_geometry(meshes)                   # one plain Trimesh (no visuals) for point sampling

Assets: ``Behavior-1K/datasets/behavior-1k-assets/objects/<category>/<model>/`` (usd/<model>.encrypted.usd,
material/*.png), key ``Behavior-1K/datasets/omnigibson.key``. Visual meshes only (``/visuals/``), collision / meta
links / guides / invisible prims skipped, OmniGlass parts skipped (transparent in Isaac Sim; pyrender would draw them
opaque). The USD world frame is the canonical object frame the sampler and the KB bounding boxes use.
"""
from __future__ import annotations

import os
import tempfile
from typing import List, Optional

import numpy as np
import trimesh
from cryptography.fernet import Fernet
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ASSETS = os.path.join(REPO, "Behavior-1K", "datasets", "behavior-1k-assets", "objects")
KEY_PATH = os.path.join(REPO, "Behavior-1K", "datasets", "omnigibson.key")
_FERNET: Optional[Fernet] = None


def _fernet() -> Fernet:
    global _FERNET
    if _FERNET is None:
        with open(KEY_PATH, "rb") as f:
            _FERNET = Fernet(f.read())
    return _FERNET


def usd_path(category: str, model: str) -> str:
    return f"{ASSETS}/{category}/{model}/usd/{model}.encrypted.usd"


def decrypt_usd(category: str, model: str, tmpdir: str) -> str:
    dst = os.path.join(tmpdir, f"{model}.usd")
    with open(usd_path(category, model), "rb") as f:
        data = _fernet().decrypt(f.read())
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


def diffuse_texture(prim, category: str, model: str) -> Optional[str]:
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


UNTEXTURED_COLOR = np.array([190, 190, 200, 255], dtype=np.uint8)
GLASS_COLOR = np.array([205, 225, 235, 255], dtype=np.uint8)


def load_model(category: str, model: str, tmpdir: Optional[str] = None, textures: bool = True,
               max_texture: Optional[int] = 1024, include_glass: bool = True) -> List[trimesh.Trimesh]:
    """List of trimesh.Trimesh (world frame of the USD = canonical object frame), textured where possible.

    ``max_texture``: textures larger than this (pixels per side) are downsampled on load (a 4k albedo map is not
    needed for 192 px renders and costs ~1 s to decode). ``include_glass``: keep OmniGlass parts (transparent in Isaac
    Sim, drawn opaque by pyrender; ``mesh.metadata["glass"] = True``) - the symmetry survey needs them (a wineglass is
    all glass), the review renders of the selection runlogs dropped them.
    """
    own_tmp = tmpdir is None
    if own_tmp:
        tmpdir = tempfile.mkdtemp(prefix="symmesh_")
    try:
        usd = decrypt_usd(category, model, tmpdir)
        stage = Usd.Stage.Open(usd)
        xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
        meshes: List[trimesh.Trimesh] = []
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
            glass = False
            try:  # glass parts (OmniGlass material) are transparent in Isaac Sim; pyrender draws them opaque
                mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                glass = bool(mat and "glass" in str(mat.GetPath()).lower())
            except Exception:
                pass
            if glass and not include_glass:
                continue
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
            if textures:
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
            tex = diffuse_texture(prim, category, model) if textures else None
            tm = None
            if uv is not None and tex is not None:
                if tex not in tex_cache:
                    try:
                        img = Image.open(tex).convert("RGB")
                        if max_texture and max(img.size) > max_texture:
                            s = max_texture / max(img.size)
                            img = img.resize((max(1, int(img.size[0] * s)), max(1, int(img.size[1] * s))), Image.BILINEAR)
                        tex_cache[tex] = img
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
                    # trimesh's SimpleMaterial defaults to a diffuse colour of (102, 102, 102): pyrender multiplies the
                    # texture by it and every textured render comes out at 40 % brightness (a white cup rendered grey,
                    # dark trays black). The albedo map is the colour: neutralise the factor.
                    tm.visual.material.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)
                    tm.metadata["texture_path"] = tex
            if tm is None:
                tm = trimesh.Trimesh(vertices=V, faces=tv, process=False)
                tm.visual.face_colors = GLASS_COLOR if glass else UNTEXTURED_COLOR
            # a flipped transform (negative determinant) turns the winding inside out
            if np.linalg.det(M[:3, :3]) < 0:
                tm.invert()
            tm.metadata["glass"] = glass
            tm.metadata["prim"] = path
            meshes.append(tm)
        return meshes
    finally:
        if own_tmp:
            try:
                for fn in os.listdir(tmpdir):
                    os.remove(os.path.join(tmpdir, fn))
                os.rmdir(tmpdir)
            except OSError:
                pass


def merged_geometry(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    """One plain Trimesh (vertices + faces of every part, no visuals) for point sampling."""
    vs, fs, off = [], [], 0
    for m in meshes:
        vs.append(np.asarray(m.vertices, dtype=np.float64))
        fs.append(np.asarray(m.faces, dtype=np.int64) + off)
        off += len(m.vertices)
    return trimesh.Trimesh(vertices=np.concatenate(vs), faces=np.concatenate(fs), process=False)


def scaled(meshes: List[trimesh.Trimesh], scale) -> List[trimesh.Trimesh]:
    """Copies of ``meshes`` with a per-axis scale applied in the canonical frame (the sampler's spawn scale)."""
    s = np.asarray(scale, dtype=np.float64).reshape(3)
    out = []
    for m in meshes:
        mm = m.copy()
        mm.vertices = np.asarray(mm.vertices) * s
        out.append(mm)
    return out
