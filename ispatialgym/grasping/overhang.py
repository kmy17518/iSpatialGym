"""Overhang rests of the grasp library v2 (i7-overhang, owner decision 2026-11-07: hardback bqvvvx, inside_large_01 - the primary path of the
overhang pick, ``sampling/v2/overhang.py`` / ``IORSLarge._overhang_pick``).

A *stable* rest (``validate_tabletop.TabletopWorld.settle``) is the object settled alone in the middle of the slab. An *overhang* rest is
the object settled in the SAME synthetic world on a slab whose EDGE lies under it: one face of the object (``side``, a signed horizontal
axis of the rest frame, e.g. ``"-y"`` = the hardback's spine) reaches ``distance_m`` beyond the slab edge, the edge parallel to that face
(flush), the centre of mass inside. That is where ``pick_overhang_push`` leaves a flat object before its library grasp: the pinch across
its thickness has its lower finger in the free space below the slab top, beyond the edge (the legacy ``hardback__bqvvvx__bo``
``book_overhang`` rows: "spine-edge pinch ~1 cm into the overhang; lower finger clear of the table", proxy table edge 0.03 m inside the
spine face). The grasps are generated at the measured overhang rest with the slab BOX as the support (its edge where it was measured:
beyond it the half-space below the slab top is free) and physics-validated on that slab (``validate_tabletop``: settle, closure, hold,
the slab dropped away).

* the frame is the stable rests' support frame (z = 0 at the slab top, +z up; the rest frame = the object at its rest quaternion, the root
  at the origin); ``side`` names the face by its outward normal in that frame;
* the object starts at its stable rest orientation (the job's ``rest_hint_wxyz``, canonical when None), ``drop_m`` above the slab, the slab
  box placed so its edge is ``distance_m`` inside the face's extreme point (:func:`slab_for`);
* the measured pose IS the rest (``rest_kind: "overhang"``, :func:`overhang_record`): the face's extreme beyond the measured edge
  (``measured_distance_m``), the edge relative to the root (``edge_along_m`` along the face normal ``normal_xy`` in the support frame),
  the centre of mass inside the edge (``com_inside_m``) and the slab in the root frame.

Identity (owner: "overhang in the entry and map identity"): an :class:`OverhangSpec` (side, distance, the flush edge, the slab) has one
:meth:`OverhangSpec.digest`; its :meth:`OverhangSpec.tag` (``overhang_<axis><n|p>_<mm>mm_<digest6>``) is the rest id, the library entry
id's suffix (``schema.entry_id(..., overhang=)``) and the sampler's overhang grasp key suffix (``<base key>@<tag>``, :func:`overhang_key`).
An overhang rest never serves a stable-rest lookup and a stable rest never serves an overhang one (``grasp_library.v2_rest_id`` /
``schema.check_compatible``; reach maps: the ``overhang`` kind only).

    python -m ispatialgym.grasping.overhang manifest --legacy-key hardback__bqvvvx__bo --side -y --distance 0.0294 --out overhang_jobs.json
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.grasping import schema as S

SCHEMA = "isg-overhang/1"
SETTLE_SCHEMA = "isg-overhang-settle/1"
REST_KIND_OVERHANG = "overhang"
#: the faces an overhang rest can name: signed horizontal axes of the rest frame -> (axis index, sign)
SIDES: Dict[str, Tuple[int, float]] = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0), "-y": (1, -1.0)}
#: the sampler's grasp key of an overhang rest: ``<base key><KEY_SEP><tag>`` (``hardback__bqvvvx__bo@overhang_yn_29mm_1a2b3c``)
KEY_SEP = "@"
TAG_PREFIX = "overhang_"
#: the measured overhang may differ this much from the requested one and still be that overhang rest
DISTANCE_TOL_M = 0.004
#: the centre of mass must stay at least this far inside the edge (``sampling/v2/overhang.py`` keeps ``com_margin_m`` 0.05 in the sampler)
COM_MIN_INSIDE_M = 0.01
#: the object may turn this much about z in the settle and still be flush with the edge
FLUSH_TOL_DEG = 3.0
FRAME = ("support frame: z = 0 at the slab top, +z up; the rest frame = the object at its rest quaternion, the root at the origin; the slab "
         "edge parallel to the named face, the slab on the object's side of it")
SETTLE_FILE = "overhang_settle.json"
EDGE = "flush"


@dataclass(frozen=True)
class OverhangSpec:
    """A requested overhang rest: the face ``side`` (a signed rest-frame axis) ``distance_m`` beyond the slab edge, the edge flush with
    the face (``edge`` "flush"), on a slab ``slab_thickness_m`` thick."""

    side: str
    distance_m: float
    edge: str = EDGE
    slab_thickness_m: float = 0.05

    def __post_init__(self):
        if self.side not in SIDES:
            raise ValueError(f"overhang side must be one of {sorted(SIDES)}, got {self.side!r}")
        if not 0.0 < float(self.distance_m) < 0.20:
            raise ValueError(f"overhang distance must be in (0, 0.20) m, got {self.distance_m!r}")
        if self.edge != EDGE:
            raise ValueError(f"overhang edge must be {EDGE!r} (the slab edge parallel to the face), got {self.edge!r}")

    @property
    def axis(self) -> int:
        return SIDES[self.side][0]

    @property
    def sign(self) -> float:
        return SIDES[self.side][1]

    def identity(self) -> Dict[str, Any]:
        """Every result-affecting factor of the overhang settle (the digest's content)."""
        return {"schema": SCHEMA, "kind": REST_KIND_OVERHANG, "side": self.side, "distance_m": round(float(self.distance_m), 4), "edge": self.edge,
                "slab_thickness_m": round(float(self.slab_thickness_m), 4), "frame": FRAME}

    def digest(self) -> str:
        return S.digest(self.identity(), 12)

    def tag(self) -> str:
        """``overhang_<axis><n|p>_<mm>mm_<digest6>``: the rest id, the entry-id suffix and the overhang grasp key suffix."""
        ax = "xy"[self.axis]
        return f"{TAG_PREFIX}{ax}{'n' if self.sign < 0 else 'p'}_{int(round(1000.0 * float(self.distance_m)))}mm_{self.digest()[:6]}"

    def rest_id(self) -> str:
        return self.tag()

    def to_json(self) -> Dict[str, Any]:
        return {**self.identity(), "digest": self.digest(), "tag": self.tag()}

    @staticmethod
    def from_json(d) -> "OverhangSpec":
        if isinstance(d, OverhangSpec):
            return d
        return OverhangSpec(side=str(d["side"]), distance_m=float(d["distance_m"]), edge=str(d.get("edge", EDGE)),
                            slab_thickness_m=float(d.get("slab_thickness_m", 0.05)))


def rest_kind(record: Optional[Dict[str, Any]]) -> str:
    return str((record or {}).get("rest_kind") or "stable")


def overhang_digest_of(record: Optional[Dict[str, Any]]) -> Optional[str]:
    """The overhang identity digest of a rest record / library / job (``overhang.digest``), None for any other."""
    ov = (record or {}).get("overhang") if isinstance(record, dict) else None
    if not isinstance(ov, dict):
        return None
    return ov.get("digest") or OverhangSpec.from_json(ov).digest()


# ------------------------------------------------------------------------------------------------ keys (the sampler's overhang grasp keys)


def overhang_key(base_key: str, spec_or_tag) -> str:
    tag = spec_or_tag.tag() if isinstance(spec_or_tag, OverhangSpec) else str(spec_or_tag)
    return f"{base_key}{KEY_SEP}{tag}"


def split_overhang_key(key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """``(base key, overhang tag or None)``; any other key is ``(key, None)`` (a lean key too)."""
    if not key or KEY_SEP not in str(key):
        return key, None
    base, tag = str(key).rsplit(KEY_SEP, 1)
    return (base, tag) if tag.startswith(TAG_PREFIX) else (key, None)


def is_overhang_key(key: Optional[str]) -> bool:
    return split_overhang_key(key)[1] is not None


# ------------------------------------------------------------------------------------------------ the settle world's geometry


def face_extreme(V_rest: np.ndarray, spec: OverhangSpec) -> float:
    """The named face's extreme coordinate along its axis (rest frame, root at the origin): ``sign * max(sign * V[:, axis])``."""
    v = np.asarray(V_rest, dtype=float)[:, spec.axis] * spec.sign
    return float(spec.sign * v.max())


def edge_coordinate(V_rest: np.ndarray, spec: OverhangSpec) -> float:
    """The slab edge's coordinate along the face's axis (rest frame): ``distance_m`` inside the face's extreme."""
    return face_extreme(V_rest, spec) - spec.sign * float(spec.distance_m)


def slab_for(V_rest: np.ndarray, spec: OverhangSpec, obj_xy=(0.0, 0.0), half: float = 0.35, slab_top_z: float = 0.75) -> Dict[str, Any]:
    """The slab box (world frame of the validation world at the reference yaw: the rest frame's axes) whose edge is
    :func:`edge_coordinate` (+ ``obj_xy``) and which extends ``2 half`` from it toward the object's inside along the face's axis and
    ``2 half`` across, centred on the object; ``{"centre", "dims", "edge_world"}``."""
    ax, sg = spec.axis, spec.sign
    e = float(obj_xy[ax]) + edge_coordinate(V_rest, spec)
    other = 1 - ax
    c = [0.0, 0.0, float(slab_top_z) - 0.5 * float(spec.slab_thickness_m)]
    c[ax] = e - sg * float(half)
    c[other] = float(obj_xy[other])
    return {"centre": c, "dims": [2.0 * float(half), 2.0 * float(half), float(spec.slab_thickness_m)], "edge_world": e}


def normal_world(quat_rest_xyzw, quat_measured_xyzw, spec: OverhangSpec) -> np.ndarray:
    """The face's outward normal at the measured orientation (support frame, horizontal unit): the rest-frame axis carried by the turn
    from the intended rest to the measured one (a small settle yaw turns the face with the object)."""
    e = np.zeros(3)
    e[spec.axis] = spec.sign
    R0 = S.quat_to_mat(quat_rest_xyzw)
    R1 = S.quat_to_mat(quat_measured_xyzw)
    n = R1 @ (R0.T @ e)
    n[2] = 0.0
    return n / max(float(np.linalg.norm(n)), 1e-9)


def overhang_record(spec: OverhangSpec, V_obj: np.ndarray, q_xyzw, p_root, edge_world: float, com_obj, slab_centre, slab_dims, quat_rest_xyzw) -> Dict[str, Any]:
    """The ``overhang`` block of a measured overhang rest record: the spec's identity, the measured face overhang beyond the edge and the
    centre of mass inside it (both along the face normal), the edge relative to the root (``edge_along_m``: the edge's coordinate along
    the normal from the root - positive: the edge lies toward the face), the normal in the support frame (``normal_xy``), the turn from
    flush, the slab relative to the root."""
    p = np.asarray(p_root, dtype=float)
    R = S.quat_to_mat(q_xyzw)
    W = np.asarray(V_obj, dtype=float) @ R.T + p
    n = normal_world(quat_rest_xyzw, q_xyzw, spec)
    e_axis = np.zeros(3)
    e_axis[spec.axis] = spec.sign
    # the slab edge is a line of the validation world: the world axis of the face at the intended rest, at edge_world
    edge_pt = np.array(p)
    edge_pt[spec.axis] = float(edge_world)
    # distances along the slab's own outward normal (e_axis: the edge is parallel to it by construction)
    face = float((W[:, :2] @ e_axis[:2]).max())
    edge_c = float(edge_pt[:2] @ e_axis[:2])
    com_w = R @ np.asarray(com_obj, dtype=float) + p
    turn = math.degrees(math.atan2(float(n[0] * e_axis[1] - n[1] * e_axis[0]), float(n[:2] @ e_axis[:2])))
    c = np.asarray(slab_centre, dtype=float)
    return {**spec.to_json(), "measured_distance_m": round(face - edge_c, 6), "com_inside_m": round(edge_c - float(com_w[:2] @ e_axis[:2]), 6),
            "edge_along_m": round(edge_c - float(p[:2] @ e_axis[:2]), 6), "normal_xy": [round(float(v), 6) for v in e_axis[:2]],
            "face_normal_measured_xy": [round(float(v), 6) for v in n[:2]], "flush_turn_deg": round(turn, 3),
            "slab_in_root": {"centre": [round(float(v), 6) for v in (c - p)], "half": [round(0.5 * float(v), 6) for v in slab_dims]},
            "slab_pose": {"centre": [round(float(v), 6) for v in c], "dims": [round(float(v), 6) for v in slab_dims]},
            "edge_rule": "the slab box's edge is a line parallel to the named face; distances along its outward normal (support frame)"}


def overhang_ok(ov: Dict[str, Any], tol_m: float = DISTANCE_TOL_M) -> Tuple[bool, str]:
    """Whether a measured overhang is its spec's overhang rest: the face beyond the edge within ``tol_m`` of the requested distance, the
    centre of mass at least :data:`COM_MIN_INSIDE_M` inside, still flush (:data:`FLUSH_TOL_DEG`)."""
    d = float(ov["measured_distance_m"]) - float(ov["distance_m"])
    if abs(d) > tol_m:
        return False, f"overhangs {float(ov['measured_distance_m']):.4f} m, requested {float(ov['distance_m']):.4f} (> {tol_m} m)"
    if float(ov["com_inside_m"]) < COM_MIN_INSIDE_M:
        return False, f"centre of mass {float(ov['com_inside_m']):.4f} m inside the edge (< {COM_MIN_INSIDE_M})"
    if abs(float(ov.get("flush_turn_deg", 0.0))) > FLUSH_TOL_DEG:
        return False, f"turned {float(ov['flush_turn_deg']):.2f} deg from flush (> {FLUSH_TOL_DEG})"
    return True, "ok"


def support_in_rest(ov: Dict[str, Any], z_top: float, name: str = "overhang_slab"):
    """The slab as the candidates' ``contact_policy.Support`` box in the REST frame (the root at the origin, the slab top at ``z_top``):
    the measured slab relative to the root, its top moved to ``z_top`` (the rest frame's z_min, as the stable candidates' plane)."""
    from ispatialgym.grasping import contact_policy as CP

    s = ov["slab_in_root"]
    c = [float(v) for v in s["centre"]]
    h = [float(v) for v in s["half"]]
    c[2] = float(z_top) - h[2]
    return CP.Support.box(c, h, name=name)


# ------------------------------------------------------------------------------------------------ jobs


def is_overhang_job(job: Optional[Dict[str, Any]]) -> bool:
    return bool(job) and rest_kind(job) == REST_KIND_OVERHANG and isinstance(job.get("overhang"), dict)


def spec_of(job_or_record: Dict[str, Any]) -> OverhangSpec:
    return OverhangSpec.from_json(job_or_record["overhang"])


def make_job(base_job: Dict[str, Any], spec: OverhangSpec, *, base_key: Optional[str] = None, base_facts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The pipeline job of one overhang rest of ``base_job``'s model (same category / model / scale / dataset / designated rest): its own
    entry id (``schema.entry_id(..., overhang=)``), ``rest_kind: "overhang"``, the spec, the keys it serves (``overhang_keys``: the
    sampler's lookup; ``legacy_key`` stays None, so no stable lookup ever lists the entry)."""
    j = {k: base_job.get(k) for k in ("category", "model", "scale", "dataset", "rest_hint_wxyz") if k in base_job}
    j.update({"constraint": "tabletop", "rest_kind": REST_KIND_OVERHANG, "overhang": spec.to_json(), "overhang_base": dict(base_facts or {}),
              "overhang_keys": [base_key] if base_key else [], "legacy_key": None, "kind": "overhang", "source": "overhang"})
    j["scale"] = list(S.canonical_scale(j.get("scale")))
    j["id"] = S.entry_id(j["category"], j["model"], j["scale"], "tabletop", j.get("rest_hint_wxyz"), overhang=j["overhang"])
    return j


def job_at_rest(job: Dict[str, Any], rest: Dict[str, Any]) -> Dict[str, Any]:
    """The overhang job with its MEASURED rest: ``generation_rest_wxyz`` (the candidates' rest frame) and ``overhang_rest`` (the slab /
    edge relative to the root, the measured overhang, h_root) - what the candidates and the validation read."""
    ov = rest["overhang"]
    x, y, z, w = (float(v) for v in rest["quat_xyzw"])
    return dict(job, generation_rest_wxyz=[round(w, 7), round(x, 7), round(y, 7), round(z, 7)],
                overhang_rest={"rest_id": rest.get("rest_id"), "h_root": rest["h_root"], "quat_xyzw": rest["quat_xyzw"], "stable": rest.get("stable"),
                               "slab_in_root": ov["slab_in_root"], "edge_along_m": ov["edge_along_m"], "normal_xy": ov["normal_xy"],
                               "measured_distance_m": ov["measured_distance_m"], "com_inside_m": ov["com_inside_m"], "overhang_digest": ov["digest"]})


def resolve_job(job: Dict[str, Any], model_dir: str) -> Dict[str, Any]:
    """An overhang job read from a manifest gets its measured rest from ``<model_dir>/overhang_settle.json`` (the ``overhang`` stage's
    record) when the stage ran in another process; any other job (or one without the record) is returned unchanged."""
    if not is_overhang_job(job) or job.get("overhang_rest"):
        return job
    doc = S.load_json(os.path.join(model_dir, SETTLE_FILE))
    if not doc or not doc.get("rest"):
        return job
    return job_at_rest(job, doc["rest"])


def legacy_base(key: str, root: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(base job, facts)`` of a legacy key (its designated rest: ``rest_hint_wxyz``, canonical when None) and the legacy rows' overhang
    (the cands file's ``overhang_m`` / ``side`` per row) for the record."""
    from ispatialgym.grasping import manifest as MF

    base = MF.legacy_job(key, root or MF.LEGACY_ROOT)
    if base is None:
        raise ValueError(f"{key}: not a tabletop legacy key")
    facts: Dict[str, Any] = {"from_key": key, "rest_pose": base.get("rest_pose")}
    try:
        from ispatialgym.sampling import grasp_library as GL

        rows = (GL._candidates(key) or {}).get("grasps") or {}
        facts["legacy_rows"] = {n: {k: (r.get("ispatialgym") or {}).get(k) for k in ("side", "overhang_m", "edge_offset_m", "context")} for n, r in rows.items()}
    except (FileNotFoundError, ValueError):
        pass
    return base, facts


def manifest(keys: Sequence[str], side: str, distances: Sequence[float], slab_thickness_m: float = 0.05) -> List[Dict[str, Any]]:
    out = []
    for k in keys:
        base, facts = legacy_base(k)
        for d in distances:
            out.append(make_job(base, OverhangSpec(side, float(d), slab_thickness_m=slab_thickness_m), base_key=k, base_facts=facts))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="overhang-rest jobs of legacy keys (one per distance)")
    m.add_argument("--legacy-key", nargs="+", required=True)
    m.add_argument("--side", required=True, choices=sorted(SIDES))
    m.add_argument("--distance", nargs="+", type=float, required=True, help="the face's overhang beyond the slab edge (m)")
    m.add_argument("--slab-thickness", type=float, default=0.05)
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    jobs = manifest(a.legacy_key, a.side, a.distance, a.slab_thickness)
    S.save_json(a.out, jobs)
    for j in jobs:
        print(j["id"], j["overhang"]["tag"])
    print(f"{len(jobs)} overhang jobs -> {a.out}")


if __name__ == "__main__":
    main()
