"""An object model's PhysX collision geometry in its ``base_link`` frame (applied scale, canonical orientation), for candidate
generation and symmetry detection without a simulator.

:func:`export_loaded` writes ``geometry.json`` (+ a viewable ``geometry.obj``) into the model's library directory from a loaded
OmniGibson object (the export stage of :mod:`ispatialgym.grasping.pipeline` runs it in one persistent world for many models,
the validation stage re-uses it when the file is missing); :func:`load` reads it back as trimesh pieces. Every collision prim is a
convex piece for PhysX (``convexHull`` / ``convexDecomposition`` pieces), so the pieces are kept apart.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ispatialgym.grasping import schema as S

FORMAT = "isg-object-geometry/1"


@dataclass
class ObjectGeometry:
    info: Dict[str, Any]
    pieces: list  # trimesh.Trimesh per collision prim (object frame)
    union: Any    # trimesh.Trimesh (concatenation)

    @property
    def bbox(self):
        return self.union.vertices.min(0), self.union.vertices.max(0)

    def rotated(self, R: np.ndarray) -> "ObjectGeometry":
        """The geometry in a rest-rotated frame (``R`` 4x4 applied to the object frame)."""
        import trimesh

        pcs = []
        for p in self.pieces:
            q = p.copy()
            q.apply_transform(R)
            pcs.append(q)
        return ObjectGeometry(info=dict(self.info), pieces=pcs, union=trimesh.util.concatenate(pcs))


def path_of(model_dir: str) -> str:
    return os.path.join(model_dir, "geometry.json")


def export_loaded(obj, model_dir: str, job: Dict[str, Any], dataset_root: Optional[str] = None) -> Dict[str, Any]:
    """Collision prims of a loaded object -> ``geometry.json`` / ``geometry.obj`` in its base_link frame."""
    import omnigibson.utils.transform_utils as T
    from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh

    t0 = time.time()
    rp, rq = obj.root_link.get_position_orientation()
    pieces = []
    approx = {}
    for link_name, link in obj.links.items():
        for cm_name, cm in link.collision_meshes.items():
            pos, quat = cm.get_position_orientation()
            pos, quat = T.relative_pose_transform(pos, quat, rp, rq)
            if cm.geom_type == "Mesh":
                verts = cm.points.numpy().astype(np.float64)
                faces = cm.faces.numpy().astype(np.int64)
            else:
                tm = mesh_prim_shape_to_trimesh_mesh(cm.prim)
                verts, faces = np.asarray(tm.vertices, dtype=np.float64), np.asarray(tm.faces, dtype=np.int64)
            scale = cm.get_world_scale().numpy().astype(np.float64)
            R = T.quat2mat(quat).numpy().astype(np.float64)
            try:
                approx[f"{link_name}/{cm_name}"] = str(cm.prim.GetAttribute("physics:approximation").Get())
            except Exception:  # noqa: BLE001
                pass
            pieces.append({"name": f"{link_name}__{cm_name}", "link": link_name, "vertices": ((verts * scale) @ R.T + pos.numpy()).round(6).tolist(),
                           "faces": faces.tolist()})
    allv = np.concatenate([np.asarray(p["vertices"]) for p in pieces]) if pieces else np.zeros((1, 3))
    lo, hi = allv.min(0), allv.max(0)
    md = {}
    try:
        from omnigibson.utils.asset_utils import get_dataset_path

        mp = os.path.join(get_dataset_path(job.get("dataset") or "behavior-1k-assets"), "objects", job["category"], job["model"], "misc", "metadata.json")
        if os.path.exists(mp):
            md = json.load(open(mp))
    except Exception:  # noqa: BLE001
        md = {}
    out = {"format": FORMAT, "category": job["category"], "model": job["model"], "scale": list(S.canonical_scale(job.get("scale"))),
           "dataset": job.get("dataset"), "n_links": len(obj.links), "link_names": list(obj.links.keys()), "mass": float(obj.mass),
           "coll_bbox_min": lo.round(5).tolist(), "coll_bbox_max": hi.round(5).tolist(), "approximations": approx,
           "meta_links": md.get("meta_links", {}), "pieces": pieces, "export_s": round(time.time() - t0, 3), "exported": S.now()}
    S.save_json(path_of(model_dir), out)
    with open(os.path.join(model_dir, "geometry.obj"), "w") as f:
        off = 0
        for p in pieces:
            f.write(f"o {p['name']}\n")
            for x, y, z in p["vertices"]:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            for a, b, c in p["faces"]:
                f.write(f"f {a + 1 + off} {b + 1 + off} {c + 1 + off}\n")
            off += len(p["vertices"])
    return out


def load(model_dir: str) -> Optional[ObjectGeometry]:
    import trimesh

    d = S.load_json(path_of(model_dir))
    if d is None:
        return None
    pcs = []
    for p in d["pieces"]:
        m = trimesh.Trimesh(vertices=np.asarray(p["vertices"], dtype=np.float64), faces=np.asarray(p["faces"], dtype=np.int64), process=False)
        try:
            trimesh.repair.fix_normals(m)  # inverted windings break the inward ray casts
        except Exception:  # noqa: BLE001
            pass
        pcs.append(m)
    info = {k: v for k, v in d.items() if k != "pieces"}
    return ObjectGeometry(info=info, pieces=pcs, union=trimesh.util.concatenate(pcs))
