"""Lean rests of the grasp library v2 (i7-lean, owner decision 2026-11-07: touching_03 and any task that starts an object leaning on a board).

A *stable* rest (``validate_tabletop.TabletopWorld.settle``) is the object settled alone on the slab. A *lean* rest is the object settled in
the SAME synthetic world against a fixed VERTICAL BOARD standing on the slab, at a requested lean angle and side:

* the frame is the stable rests' support frame (z = 0 at the slab top, +z up) with +x INTO the shelf (the object's spine / approach face
  toward -x, the robot) and +y the robot's left - the sampler's shelf convention (``tabletop.upright_lean_quat`` / ``releaned_quat``);
* the object starts upright (``job["lean_base_wxyz"]``: its up axis vertical, e.g. a legacy upright-shelf key's rest re-leaned to 0 by
  :func:`upright`), turned ``deg`` about +x so its top leans toward the ``side`` (+y = "left", -y = "right"; :func:`lean_quat`), its bottom
  :data:`DROP_M`-free above the slab and its extreme point toward the board ``gap_m`` from the board's inner face (:func:`initial_pose`);
* gravity closes the gap; the measured pose (quaternion, ``h_root``, the board's inner face relative to the root, the measured lean and
  the loads on the slab and on the board) IS the rest (``rest_kind: "lean"``, :func:`lean_record`).

Identity (owner: "rest kind, lean angle, side and the board geometry go into the entry and into every map / lookup identity"): a
:class:`LeanSpec` (angle, side, :class:`Board` geometry, initial gap) has one :meth:`LeanSpec.digest`; its :meth:`LeanSpec.tag`
(``lean_<side>_<deg>_<digest6>``) is the rest id, the library entry id's suffix (``schema.entry_id(..., lean=)``) and the sampler's
lean grasp key suffix (``<base key>@<tag>``, :func:`lean_key`). A lean rest never serves a stable-rest lookup and a stable rest never
serves a lean one (``grasp_library.v2_rest_id`` / ``schema.check_compatible``).

    python -m ispatialgym.grasping.lean manifest --legacy-key hardback__acbrnv__bu --deg 11 20 --side left --out lean_jobs.json
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym.grasping import schema as S

SCHEMA = "isg-lean/1"
SETTLE_SCHEMA = "isg-lean-settle/1"
REST_KIND_STABLE = "stable"
REST_KIND_LEAN = "lean"
REST_KINDS = (REST_KIND_STABLE, REST_KIND_LEAN)
SIDES = {"left": 1.0, "right": -1.0}
#: the sampler's grasp key of a lean rest: ``<base key><LEAN_KEY_SEP><tag>`` (``hardback__acbrnv__bu@lean_left_11_1a2b3c``)
LEAN_KEY_SEP = "@"
#: the measured lean may differ this much from the requested one and still be that lean rest (the 2 mm gap closes: ~0.5 deg)
LEAN_TOL_DEG = 2.0
#: the board carries the lean when the object's normal load on it exceeds this fraction of its weight (a leaning book: ~tan(lean) / 2)
BOARD_MIN_LOAD_RATIO = 0.02
#: the frame of every lean record
FRAME = "support frame: z = 0 at the slab top, +z up; +x into the shelf (the spine / approach face toward -x), +y the robot's left"
SETTLE_FILE = "lean_settle.json"


@dataclass(frozen=True)
class Board:
    """The vertical board of the lean settle (a fixed primitive cube on the slab, default physics material): ``thickness_m`` along y,
    ``height_m`` up from the slab top, ``depth_m`` along x (centred on the object's x extent)."""

    thickness_m: float = 0.02
    height_m: float = 0.40
    depth_m: float = 0.40
    material: str = "primitive default"

    def to_json(self) -> Dict[str, Any]:
        return {"thickness_m": round(float(self.thickness_m), 4), "height_m": round(float(self.height_m), 4), "depth_m": round(float(self.depth_m), 4),
                "material": self.material}

    @staticmethod
    def from_json(d: Optional[Dict[str, Any]]) -> "Board":
        d = d or {}
        return Board(**{k: d[k] for k in ("thickness_m", "height_m", "depth_m", "material") if k in d})


@dataclass(frozen=True)
class LeanSpec:
    """A requested lean rest: ``deg`` from vertical toward ``side`` ("left" = +y / "right" = -y), against ``board``, started ``gap_m``
    short of it."""

    deg: float
    side: str
    board: Board = field(default_factory=Board)
    gap_m: float = 0.002

    def __post_init__(self):
        if self.side not in SIDES:
            raise ValueError(f"lean side must be one of {sorted(SIDES)}, got {self.side!r}")
        if not 0.0 < float(self.deg) < 45.0:
            raise ValueError(f"lean angle must be in (0, 45) deg, got {self.deg!r}")

    @property
    def sign(self) -> float:
        return SIDES[self.side]

    def identity(self) -> Dict[str, Any]:
        """Every result-affecting factor of the lean settle (the digest's content)."""
        return {"schema": SCHEMA, "kind": REST_KIND_LEAN, "deg": round(float(self.deg), 3), "side": self.side, "board": self.board.to_json(),
                "gap_m": round(float(self.gap_m), 4), "frame": FRAME}

    def digest(self) -> str:
        return S.digest(self.identity(), 12)

    def tag(self) -> str:
        """``lean_<side>_<deg>_<digest6>``: the rest id, the entry-id suffix and the lean grasp key suffix."""
        return f"lean_{self.side}_{S._num(round(float(self.deg), 3))}_{self.digest()[:6]}"

    def rest_id(self) -> str:
        return self.tag()

    def to_json(self) -> Dict[str, Any]:
        return {**self.identity(), "digest": self.digest(), "tag": self.tag()}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "LeanSpec":
        if isinstance(d, LeanSpec):
            return d
        return LeanSpec(deg=float(d["deg"]), side=str(d["side"]), board=Board.from_json(d.get("board")), gap_m=float(d.get("gap_m", 0.002)))


def side_name(sign: float) -> str:
    return "left" if float(sign) > 0 else "right"


def rest_kind(record: Optional[Dict[str, Any]]) -> str:
    """A rest record's kind: ``rest_kind`` (records before the lean kind are stable rests)."""
    return str((record or {}).get("rest_kind") or REST_KIND_STABLE)


def lean_digest_of(record: Optional[Dict[str, Any]]) -> Optional[str]:
    """The lean identity digest of a rest record / library / job (``lean.digest``), None for a stable one."""
    ln = (record or {}).get("lean") if isinstance(record, dict) else None
    if not isinstance(ln, dict):
        return None
    return ln.get("digest") or LeanSpec.from_json(ln).digest()


# ------------------------------------------------------------------------------------------------ keys (the sampler's lean grasp keys)


def lean_key(base_key: str, spec_or_tag) -> str:
    tag = spec_or_tag.tag() if isinstance(spec_or_tag, LeanSpec) else str(spec_or_tag)
    return f"{base_key}{LEAN_KEY_SEP}{tag}"


def split_lean_key(key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """``(base key, lean tag or None)``; a stable key is ``(key, None)``."""
    if not key or LEAN_KEY_SEP not in str(key):
        return key, None
    base, tag = str(key).rsplit(LEAN_KEY_SEP, 1)
    return (base, tag) if tag.startswith("lean_") else (key, None)


def is_lean_key(key: Optional[str]) -> bool:
    return split_lean_key(key)[1] is not None


# ------------------------------------------------------------------------------------------------ orientation math (numpy, xyzw)


def quat_axis_angle(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    return np.array([*(a * math.sin(angle / 2)), math.cos(angle / 2)])


def up_axis_local(q_xyzw) -> np.ndarray:
    """The signed local axis (+-e_i of the object frame) most aligned with world +z at orientation ``q``."""
    R = S.quat_to_mat(q_xyzw)
    k = int(np.argmax(np.abs(R[2, :])))
    e = np.zeros(3)
    e[k] = 1.0 if R[2, k] > 0 else -1.0
    return e


def lean_of(q_xyzw, up_local=None) -> Dict[str, float]:
    """The lean of orientation ``q``: the tilt of the object's up axis (``up_local``, default :func:`up_axis_local`) from vertical -
    ``deg`` (total), ``side`` (+1 toward +y / -1 toward -y), ``toward_x_deg`` (its component about y: 0 for a pure lean about +x)."""
    u = S.quat_to_mat(q_xyzw) @ (up_axis_local(q_xyzw) if up_local is None else np.asarray(up_local, dtype=float))
    deg = math.degrees(math.acos(max(-1.0, min(1.0, float(u[2])))))
    return {"deg": deg, "side": 1.0 if float(u[1]) >= 0 else -1.0, "toward_x_deg": math.degrees(math.atan2(float(u[0]), float(u[2])))}


def upright(q_xyzw, up_local=None) -> np.ndarray:
    """``q`` turned by the smallest rotation that puts its up axis vertical (a leaning rest re-leaned to 0, e.g. a legacy upright-shelf
    key's 15 deg rest), xyzw."""
    q = np.asarray(q_xyzw, dtype=float)
    u = S.quat_to_mat(q) @ (up_axis_local(q) if up_local is None else np.asarray(up_local, dtype=float))
    z = np.array([0.0, 0.0, 1.0])
    ax = np.cross(u, z)
    s = float(np.linalg.norm(ax))
    if s < 1e-12:
        return q / np.linalg.norm(q)
    r = S.quat_mul(quat_axis_angle(ax, math.atan2(s, float(np.dot(u, z)))), q)
    return r / np.linalg.norm(r)


def lean_quat(q_up_xyzw, spec: LeanSpec) -> np.ndarray:
    """The initial orientation: the upright ``q_up`` turned ``spec.deg`` about +x (into the shelf) so its top leans toward ``spec.side``
    (``tabletop.releaned_quat``'s sense: +y = the robot's left)."""
    r = S.quat_mul(quat_axis_angle([1.0, 0.0, 0.0], -spec.sign * math.radians(float(spec.deg))), np.asarray(q_up_xyzw, dtype=float))
    return r / np.linalg.norm(r)


def wxyz(q_xyzw) -> List[float]:
    x, y, z, w = (float(v) for v in q_xyzw)
    return [round(w, 7), round(x, 7), round(y, 7), round(z, 7)]


def xyzw(q_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in q_wxyz)
    return np.array([x, y, z, w])


# ------------------------------------------------------------------------------------------------ the settle world's geometry


def initial_pose(V: np.ndarray, quat0, spec: LeanSpec, slab_top_z: float, obj_xy=(0.0, 0.0), drop_m: float = 0.003) -> Dict[str, Any]:
    """The lean settle's start: the object (collision vertices ``V``, object frame) at ``quat0``, its lowest point ``drop_m`` above the slab
    top, and the board (fixed, on the slab) with its inner face ``spec.gap_m`` beyond the object's extreme point toward ``spec.side``,
    centred on the object's x extent. World frame of the validation world (``slab_top_z`` absolute)."""
    R = S.quat_to_mat(quat0)
    W = np.asarray(V, dtype=float) @ R.T
    sg = spec.sign
    bottom = -float(W[:, 2].min())
    lat_ext = float((sg * W[:, 1]).max())
    face_y = float(obj_xy[1]) + sg * (lat_ext + float(spec.gap_m))
    b = spec.board
    cx = float(obj_xy[0]) + 0.5 * float(W[:, 0].min() + W[:, 0].max())
    centre = [cx, face_y + sg * 0.5 * float(b.thickness_m), float(slab_top_z) + 0.5 * float(b.height_m)]
    dims = [float(b.depth_m), float(b.thickness_m), float(b.height_m)]
    return {"quat_xyzw": np.asarray(quat0, dtype=float), "position": [float(obj_xy[0]), float(obj_xy[1]), float(slab_top_z) + bottom + float(drop_m)],
            "board_centre": centre, "board_dims": dims, "board_face_y": face_y, "lat_ext_m": lat_ext}


def board_face_y(board_centre, board_dims, sign: float) -> float:
    """The board's inner face y (the face toward the object, which stands on the -``sign`` side of it)."""
    return float(board_centre[1]) - float(sign) * 0.5 * float(board_dims[1])


def lean_record(spec: LeanSpec, q_xyzw, p_root, board_centre, board_dims, up_local, board_load_N: float, weight_N: float) -> Dict[str, Any]:
    """The ``lean`` block of a measured lean rest record: the spec's identity, the measured lean, the board (world pose and relative to
    the root in the support frame's axes), the board's inner face relative to the root (``board_face_y_m``, signed along +y: positive for
    a board on the left) and the board's load."""
    lo = lean_of(q_xyzw, up_local)
    p = np.asarray(p_root, dtype=float)
    c = np.asarray(board_centre, dtype=float)
    ratio = float(board_load_N) / max(float(weight_N), 1e-9)
    return {**spec.to_json(), "measured_deg": round(lo["deg"], 3), "measured_side": side_name(lo["side"]), "tilt_toward_x_deg": round(lo["toward_x_deg"], 3),
            "up_local": [round(float(v), 6) for v in up_local],
            "board_pose": {"centre": [round(float(v), 6) for v in c], "dims": [round(float(v), 6) for v in board_dims]},
            "board_in_root": {"centre": [round(float(v), 6) for v in (c - p)], "half": [round(0.5 * float(v), 6) for v in board_dims]},
            "board_face_y_m": round(board_face_y(c, board_dims, spec.sign) - float(p[1]), 6),
            "board_load_N": round(float(board_load_N), 4), "board_load_ratio": round(ratio, 4), "board_contact": bool(ratio >= BOARD_MIN_LOAD_RATIO),
            "board_contact_rule": f"object's normal load on the board >= {BOARD_MIN_LOAD_RATIO} x weight (awake)"}


def lean_ok(lean: Dict[str, Any], tol_deg: float = LEAN_TOL_DEG) -> Tuple[bool, str]:
    """Whether a measured lean is its spec's lean rest: the board carries it, on the requested side, within ``tol_deg`` of the angle."""
    if not lean.get("board_contact"):
        return False, "no board contact"
    if lean.get("measured_side") != lean.get("side"):
        return False, f"leans {lean.get('measured_side')}, not {lean.get('side')}"
    d = abs(float(lean["measured_deg"]) - float(lean["deg"]))
    if d > tol_deg:
        return False, f"leans {float(lean['measured_deg']):.2f} deg, requested {float(lean['deg']):.2f} (> {tol_deg} deg)"
    return True, "ok"


def board_obstacle_in_rest(lean: Dict[str, Any], name: str = "lean_board"):
    """The board as a ``contact_policy.Support`` box in the REST frame of the candidates (the support frame's axes, the object root at the
    origin): a normal obstacle for every hand part."""
    from ispatialgym.grasping import contact_policy as CP

    b = lean["board_in_root"]
    return CP.Support.box(b["centre"], b["half"], name=name)


def board_obstacle_world(lean: Dict[str, Any], name: str = "lean_board"):
    from ispatialgym.grasping import contact_policy as CP

    bp = lean["board_pose"]
    return CP.Support.box(bp["centre"], [0.5 * float(v) for v in bp["dims"]], name=name)


# ------------------------------------------------------------------------------------------------ jobs


def is_lean_job(job: Optional[Dict[str, Any]]) -> bool:
    return bool(job) and rest_kind(job) == REST_KIND_LEAN and isinstance(job.get("lean"), dict)


def spec_of(job_or_record: Dict[str, Any]) -> LeanSpec:
    return LeanSpec.from_json(job_or_record["lean"])


def base_quat(job: Dict[str, Any]) -> np.ndarray:
    """The lean job's upright orientation (xyzw) from ``lean_base_wxyz`` (identity when absent)."""
    q = job.get("lean_base_wxyz")
    return np.array([0.0, 0.0, 0.0, 1.0]) if q is None else xyzw(q)


def make_job(base_job: Dict[str, Any], spec: LeanSpec, q_up_xyzw, *, base_key: Optional[str] = None, base_facts: Optional[Dict[str, Any]] = None,
             validate_arm: str = "right", interior_goal: Optional[str] = None) -> Dict[str, Any]:
    """The pipeline job of one lean rest of ``base_job``'s model (same category / model / scale / dataset): its own entry id
    (``schema.entry_id(..., lean=)``), ``rest_kind: "lean"``, the spec, the upright start orientation, the stable / legacy key it leans
    (``lean_keys``: the sampler's lookup; ``legacy_key`` stays None, so no stable lookup ever lists the entry). ``validate_arm`` (i7-overhang
    2026-11-07, owner decision: a LEFT-arm validator for lean rests): the arm its candidates are generated for and its physics trials run
    with (``pipeline lean --arm``); a left one is recorded in the job / entry and its id (``__valleft``)."""
    j = {k: base_job.get(k) for k in ("category", "model", "scale", "dataset") if k in base_job}
    j.update({"constraint": "tabletop", "rest_hint_wxyz": None, "rest_kind": REST_KIND_LEAN, "lean": spec.to_json(), "lean_base_wxyz": wxyz(q_up_xyzw),
              "lean_base": dict(base_facts or {}), "lean_keys": [base_key] if base_key else [], "legacy_key": None, "kind": "lean", "source": "lean"})
    if validate_arm not in ("right", "left"):
        raise ValueError(f"validate_arm must be right or left, got {validate_arm!r}")
    if validate_arm != "right":
        j["validate_arm"] = validate_arm
    if interior_goal:
        # i7-overhang 2026-11-07 (owner decision): an entry used at an interior goal - its candidates restricted to the goal's accessible region
        from ispatialgym.grasping import access_region as AR
        from ispatialgym.sampling import interiors as IN

        j["interior_goal"] = interior_goal
        j["access_region"] = AR.region_identity(IN.spec_named(interior_goal))
    j["scale"] = list(S.canonical_scale(j.get("scale")))
    j["id"] = S.entry_id(j["category"], j["model"], j["scale"], "tabletop", None, lean=j["lean"], validate_arm=j.get("validate_arm"),
                         access_region=S.job_access_region(j))
    return j


def job_at_rest(job: Dict[str, Any], rest: Dict[str, Any]) -> Dict[str, Any]:
    """The lean job with its MEASURED lean rest: ``generation_rest_wxyz`` (the candidates' rest frame) and ``lean_rest`` (the board relative
    to the root, the measured lean, h_root) - what the candidates and the validation read."""
    ln = rest["lean"]
    return dict(job, generation_rest_wxyz=wxyz(rest["quat_xyzw"]),
                lean_rest={"rest_id": rest.get("rest_id"), "h_root": rest["h_root"], "quat_xyzw": rest["quat_xyzw"], "stable": rest.get("stable"),
                           "board_in_root": ln["board_in_root"], "board_face_y_m": ln["board_face_y_m"], "measured_deg": ln["measured_deg"],
                           "measured_side": ln["measured_side"], "lean_digest": ln["digest"]})


def resolve_job(job: Dict[str, Any], model_dir: str) -> Dict[str, Any]:
    """A lean job read from a manifest gets its measured rest from ``<model_dir>/lean_settle.json`` (the ``lean`` stage's record) when the
    stage ran in another process; any other job (or one without the record) is returned unchanged."""
    if not is_lean_job(job) or job.get("lean_rest"):
        return job
    doc = S.load_json(os.path.join(model_dir, SETTLE_FILE))
    if not doc or not doc.get("rest"):
        return job
    return job_at_rest(job, doc["rest"])


def legacy_upright(key: str, root: Optional[str] = None) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """``(base job, upright quat xyzw, facts)`` of a legacy upright-shelf key: its rest (``rest_quat_wxyz``) re-leaned to 0 (:func:`upright`)."""
    from ispatialgym.grasping import manifest as MF

    base = MF.legacy_job(key, root or MF.LEGACY_ROOT)
    if base is None:
        raise ValueError(f"{key}: not a tabletop legacy key")
    q = np.array([0.0, 0.0, 0.0, 1.0]) if base.get("rest_hint_wxyz") is None else xyzw(base["rest_hint_wxyz"])
    lo = lean_of(q)
    facts = {"from_key": key, "rest_pose": base.get("rest_pose"), "key_lean_deg": round(lo["deg"], 3), "key_lean_side": side_name(lo["side"]),
             "method": "the key's rest turned by the smallest rotation putting its up axis vertical (lean.upright)"}
    return base, upright(q), facts


def manifest(keys: Sequence[str], degs: Sequence[float], side: str, board: Board = Board(), gap_m: float = 0.002, validate_arm: str = "right",
             interior_goal: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for k in keys:
        base, q_up, facts = legacy_upright(k)
        for d in degs:
            out.append(make_job(base, LeanSpec(float(d), side, board, gap_m), q_up, base_key=k, base_facts=facts, validate_arm=validate_arm,
                                interior_goal=interior_goal))
    return out


# ------------------------------------------------------------------------------------------------ the arm-mirror transfer (i7-t03both r3)
#
# Lead #84c (owner A5, touching_03's right board): the robot's two arms are mirror images (r1pro.urdf: the joints' y offsets and the x / z
# axes' limits mirror; a right-arm configuration of the mirrored joint values puts its eef at M R_l M, M = diag(1, -1, 1) in the robot
# frame, and the wrist cameras at the same gripper-frame offset then sit mirrored too). A grasp that works for the LEFT arm at a left-board
# lean rest is transferred to the RIGHT-board rest as the mirror of its hand: world hand M (R_l T) M; in the right rest's object frame
# T' = R_r^T M R_l T M = B T M with B = R_r^T M R_l - a reflection of the OBJECT's body frame, valid only when the object is symmetric
# under it (``mirror_symmetry_error``: the hull's vertices reflected by B onto the hull, max nearest distance). The transferred poses go
# to the candidate generator as given poses (``candidates`` family "legacy": every gate unchanged) and through the physics validation of the
# validator arm - the method adds candidates, it relaxes nothing.

MIRROR = np.diag([1.0, -1.0, 1.0])
#: the largest hull asymmetry (m: a reflected vertex's distance to the nearest hull vertex) the transfer accepts
MIRROR_SYMMETRY_TOL_M = 0.002
#: a reflection plane within this of a body-frame plane is that plane (deg; the settle's residual lean tilts it by half the residual)
MIRROR_AXIS_SNAP_DEG = 3.0


def mirror_body_map(src_rest_xyzw, dst_rest_xyzw) -> np.ndarray:
    """``B = R_dst^T M R_src``: the body-frame map that carries the source rest's object onto the destination rest's mirror, projected onto
    the nearest pure reflection ``I - 2 n n^T`` (n = its eigenvector of eigenvalue -1): the destination's design rest differs from the
    source's measured one by the settle's fraction of a degree (20.55 measured vs 20 requested), which is no part of the mirror; a normal
    within :data:`MIRROR_AXIS_SNAP_DEG` of a body axis is snapped to it."""
    from scipy.spatial.transform import Rotation as Rot

    Rs, Rd = Rot.from_quat(np.asarray(src_rest_xyzw, dtype=float)).as_matrix(), Rot.from_quat(np.asarray(dst_rest_xyzw, dtype=float)).as_matrix()
    B = Rd.T @ MIRROR @ Rs
    w, v = np.linalg.eig(B)
    k = int(np.argmin(np.abs(w - (-1.0))))
    n = np.real(v[:, k])
    n = n / np.linalg.norm(n)
    i = int(np.argmax(np.abs(n)))
    if math.degrees(math.acos(min(1.0, abs(float(n[i]))))) <= MIRROR_AXIS_SNAP_DEG:
        n = np.eye(3)[i]  # the object's own symmetry plane (its body axes are the export's principal frame)
    return np.eye(3) - 2.0 * np.outer(n, n)


def mirror_symmetry_error(vertices, B) -> float:
    """Max over the hull vertices of how far ``B v`` lies OUTSIDE the convex hull of the vertices (m; its face planes): 0 for an object
    symmetric under ``B`` (a vertex-to-vertex distance would count the hull's own vertex sampling on a rounded corner)."""
    from scipy.spatial import ConvexHull

    V = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if len(V) < 4:
        return 0.0
    W = V @ np.asarray(B, dtype=float).T
    eq = ConvexHull(V).equations  # (F, 4): n . x + d <= 0 inside
    out = (W @ eq[:, :3].T + eq[:, 3]).max(axis=1)
    return float(max(0.0, out.max()))


def mirror_grasp(T_object_eef: Dict[str, Any], B) -> Dict[str, Any]:
    """One grasp's ``T_object_eef`` (position, orientation_xyzw) transferred by the body reflection ``B``: position ``B t``, rotation
    ``B R M`` (a proper rotation: two reflections)."""
    from scipy.spatial.transform import Rotation as Rot

    t = np.asarray(T_object_eef["position"], dtype=float)
    R = Rot.from_quat(np.asarray(T_object_eef["orientation_xyzw"], dtype=float)).as_matrix()
    R2 = np.asarray(B) @ R @ MIRROR
    return {"position": [round(float(v), 6) for v in (np.asarray(B) @ t)], "quat_xyzw": [round(float(v), 7) for v in Rot.from_matrix(R2).as_quat()]}


def mirror_job(job: Dict[str, Any], src_library: Dict[str, Any], src_vertices, grasp_names: Optional[Sequence[str]] = None,
               tol_m: float = MIRROR_SYMMETRY_TOL_M, twins: bool = False, expand: bool = False) -> Dict[str, Any]:
    """The lean job ``job`` (the destination board's rest) with the source library's grasps (``grasp_names``, default all) transferred by the
    arm mirror as its ONLY candidates (``cand_overrides``: no sampled mode, family "legacy" = the given poses) and the provenance in
    ``mirror_of``; its entry id gains ``__mirror_<source tag>``. ``ValueError`` when the source and destination lean to the same side or the
    object is not symmetric under the body map (> ``tol_m``)."""
    src_lean = src_library.get("lean") or {}
    if src_lean.get("side") == job["lean"]["side"]:
        raise ValueError(f"the mirror transfer needs the other board: source {src_lean.get('side')} vs {job['lean']['side']}")
    rid = src_lean.get("tag")
    src_rest = ((src_library.get("rests") or {}).get(rid) or {}).get("quat_xyzw")
    if src_rest is None:
        raise ValueError(f"source library has no rest {rid!r}")
    dst_rest = lean_quat(base_quat(job), spec_of(job))
    B = mirror_body_map(src_rest, dst_rest)
    err = mirror_symmetry_error(src_vertices, B)
    if err > tol_m:
        raise ValueError(f"the object is not symmetric under the mirror's body map ({err * 1000:.2f} mm > {tol_m * 1000:.1f} mm)")
    names = set(grasp_names) if grasp_names else None
    grasps = [g for g in src_library.get("grasps") or [] if names is None or g.get("name") in names]
    if not grasps:
        raise ValueError(f"no source grasp among {sorted(names) if names else 'all'}")
    given = [{"name": f"mirror_{g['name']}", **mirror_grasp(g["T_object_eef"], B)} for g in grasps]
    if twins:
        # the wrist cameras sit at the same gripper-frame offset on both arms, so the mirrored hand's camera does NOT land on the mirror of
        # the source camera (acbrnv side_9 at the right board: the camera 10 mm into the board, the source's 27.5 mm clear); the twin (the
        # same pinch, the hand turned 180 deg about its approach) carries it to the other side of the pinch
        from scipy.spatial.transform import Rotation as Rot

        for g in list(given):
            Rg = Rot.from_quat(g["quat_xyzw"]).as_matrix() @ Rot.from_euler("z", 180.0, degrees=True).as_matrix()
            given.append({"name": g["name"] + "~r180", "position": list(g["position"]), "quat_xyzw": [round(float(v), 7) for v in Rot.from_matrix(Rg).as_quat()]})
    out = dict(job)
    out["cand_overrides"] = {"modes": [], "families": ["legacy"], "legacy_grasps": given}
    if expand:
        # i7-t03both r3: each transferred pose also as a proposal of the given-pose family "agent" (its pinch centre, closing axis and
        # approach in the destination's design rest frame), expanded by candidates.AGENT_EXPAND_DEFAULT (tilt / yaw / slide) - the gates and
        # the physics pick among the neighbours of the mirror (the exact mirror's twin closes the book into the table: obj_slab_ratio 3.3 > 2)
        from scipy.spatial.transform import Rotation as Rot

        from ispatialgym.grasping import candidates as CA

        Rd = Rot.from_quat(np.asarray(dst_rest, dtype=float)).as_matrix()
        props = []
        for g in given:
            Rg = Rd @ Rot.from_quat(g["quat_xyzw"]).as_matrix()
            p = Rd @ np.asarray(g["position"], dtype=float)
            c = p + Rg[:, 2] * CA.CONTACT_Z
            props.append({"name": g["name"], "contact_mm": [round(float(v) * 1000.0, 3) for v in c], "closing_axis": [round(float(v), 6) for v in Rg[:, 1]],
                          "approach": [round(float(v), 6) for v in Rg[:, 2]], "pad_offset_mm": 0.0})
        out["cand_overrides"]["families"] = ["legacy", "agent"]
        out["cand_overrides"]["agent_proposals"] = props
    out["mirror_of"] = {"method": "arm mirror (lean.mirror_job)" + (" + the 180 deg twins" if twins else ""), "source_rest": rid, "source_arm": src_library.get("arm"),
                        "grasps": [g["name"] for g in grasps], "body_map": [[round(float(v), 6) for v in row] for row in B],
                        "symmetry_error_m": round(err, 6), "design_rest_xyzw": [round(float(v), 7) for v in dst_rest]}
    out["id"] = f"{job['id']}__mirror_{rid}"
    return out


def job_arm(job: Dict[str, Any]) -> str:
    """The arm a lean job is validated with (``validate_arm``; right when absent)."""
    return str(job.get("validate_arm") or "right")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="lean-rest jobs of legacy upright keys (one per angle)")
    m.add_argument("--legacy-key", nargs="+", required=True)
    m.add_argument("--deg", nargs="+", type=float, required=True)
    m.add_argument("--side", required=True, choices=sorted(SIDES))
    m.add_argument("--board-thickness", type=float, default=Board.thickness_m)
    m.add_argument("--board-height", type=float, default=Board.height_m)
    m.add_argument("--board-depth", type=float, default=Board.depth_m)
    m.add_argument("--gap", type=float, default=0.002)
    m.add_argument("--validate-arm", default="right", choices=["right", "left"], help="the arm the lean rest is validated with (pipeline lean --arm)")
    m.add_argument("--interior-goal", default=None, help="the interior spec (registry name) the entry is used at: its accessible region (access_region.py)")
    m.add_argument("--out", required=True)
    mm = sub.add_parser("mirror", help="i7-t03both: lean jobs of the OTHER board whose candidates are a library's grasps arm-mirrored (mirror_job)")
    mm.add_argument("--legacy-key", required=True)
    mm.add_argument("--deg", nargs="+", type=float, required=True)
    mm.add_argument("--side", required=True, choices=sorted(SIDES))
    mm.add_argument("--src-entry", required=True, help="the source entry dir (library.json + geometry.json of the other board's lean rest)")
    mm.add_argument("--grasp", nargs="*", default=None, help="source grasp names (default: all)")
    mm.add_argument("--validate-arm", default="right", choices=["right", "left"])
    mm.add_argument("--interior-goal", default=None)
    mm.add_argument("--twins", action="store_true", help="also each mirrored pose turned 180 deg about its approach (the wrist camera's other side)")
    mm.add_argument("--expand", action="store_true", help="also each transferred pose as an expanded agent proposal (tilt / yaw / slide neighbours)")
    mm.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "mirror":
        src = S.load_json(os.path.join(a.src_entry, "library.json"))
        geo = S.load_json(os.path.join(a.src_entry, "geometry.json"))
        verts = [v for pc in geo.get("pieces") or [] for v in pc.get("vertices") or []]
        jobs = [mirror_job(j, src, verts, a.grasp, twins=a.twins, expand=a.expand) for j in manifest([a.legacy_key], a.deg, a.side, Board(), 0.002, a.validate_arm, a.interior_goal)]
        S.save_json(a.out, jobs)
        for j in jobs:
            print(j["id"], j["mirror_of"]["grasps"], "symmetry", j["mirror_of"]["symmetry_error_m"], [g["name"] for g in j["cand_overrides"]["legacy_grasps"]])
        print(f"{len(jobs)} mirror lean jobs -> {a.out}")
        return
    jobs = manifest(a.legacy_key, a.deg, a.side, Board(a.board_thickness, a.board_height, a.board_depth), a.gap, a.validate_arm, a.interior_goal)
    S.save_json(a.out, jobs)
    for j in jobs:
        print(j["id"], j["lean"]["tag"])
    print(f"{len(jobs)} lean jobs -> {a.out}")


if __name__ == "__main__":
    main()
