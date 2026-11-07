"""Versioned records of the grasp library v2 and their compatibility rules.

Layout under the library root (``ISPATIALGYM_GRASP_LIBRARY_V2``, default ``runlogs/grasp_library_v2``)::

    models/<category>__<model>__<scale tag>/
        candidates.json    generation: candidates, symmetry families, geometric screening (schema CANDIDATES)
        rests.json         measured resting configurations (schema REST), keyed by rest_id
        validation.json    physics trials of the family representatives (schema TRIAL)
        library.json       what consumers read (schema LIBRARY): validated grasps with their rest_id, family, policy, profile
    profiles/<category>__<model>__<scale tag>__<joint>.json   articulation physics profiles (schema PROFILE)

Quaternions are **xyzw** (OmniGibson's convention) everywhere in these files. Frames:

* object frame = the model's ``base_link`` at the applied scale;
* support frame = z = 0 at the actual support surface, +z up; its x / y are the validation world's (the absolute xy of the rest is
  diagnostic only - a reach-band cell gives the object-root xy);
* ``T_object_eef`` = the eef (``{arm}_eef_link``: +z approach, +y closing) in the object frame.

Placement contract (reach bands, sampling; :func:`place_object`)::

    q_object  = R_z(yaw) * rest.quat_xyzw            # yaw about the support's vertical, applied on the left
    p_object  = [cell_x, cell_y, support_z + rest.h_root]
    T_world_eef = T_world_object @ T_object_eef

``h_root`` is the measured settled height of the object root above the support plane (no 3 mm offset is added and no
``bottom_offset`` is recomputed). ``rest.reference_yaw`` is the yaw the object was dropped at; ``quat_xyzw`` already contains it.

Compatibility: a consumer passes what it requires (scale, rest_id, policy key, schema versions); :func:`check_compatible` raises
:class:`IncompatibleRecord` on any mismatch. A legacy library entry (``runlogs/grasp_library_20260905``) has no measured rest and
is never substituted for a v2 record (:func:`load_model` returns None; ``grasp_library.load_grasps`` keeps serving legacy keys).

Candidate switches (i7-switches 2026-11-07, owner decision: the plate switches adopted after a count): the ``candidates.CandidateConfig``
switches that change which candidates an entry holds and their recorded poses (:data:`CANDIDATE_SWITCHES`). ``library.json`` records their
state (``candidate_switches``, from its candidates' ``config``); an entry written before that record reads its ``candidates.json`` config,
and a config without the fields (every entry before 6e418172) reads all off. The state enters :meth:`ModelLibrary.identity` (the reach maps'
``object_key``) only when a switch is on - every existing key is unchanged, and an entry built with a switch on never matches a map of one
built without; ``check_compatible(candidate_switches=)`` lets a reader require a state. i7-padsweep 2026-11-07: ``sweep_from_preshape``
(the closing sweep from the oracle's pre-shape) is a candidate switch too; it enters the identity only when on, the two plate switches stay
a pair (``_IDENTITY_PAIR``), so no key written before it changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym import REPO_ROOT

CANDIDATES = "isg-grasp-candidates/2"
REST = "isg-rest/1"
TRIAL = "isg-grasp-trial/1"
LIBRARY = "isg-grasp-library/2"
PROFILE = "isg-artic-profile/1"
ARTIC_TRIAL = "isg-artic-trial/1"
FAMILY = "isg-grasp-family/1"

DEFAULT_ROOT = os.path.join(REPO_ROOT, "runlogs", "grasp_library_v2")


class IncompatibleRecord(ValueError):
    """A stored record does not match what the consumer requires (scale / rest / policy / schema)."""


#: i7-switches 2026-11-07: the candidate-generation switches that are part of an entry's identity (module docstring)
CANDIDATE_SWITCHES = ("blade_contact_at_retreat", "retreat_shift_exact", "sweep_from_preshape")
#: the switches of 884279fd: in the identity as a pair whenever either is on (as they were); a later switch enters only when on, so every
#: key written before it existed is unchanged (i7-padsweep 2026-11-07: sweep_from_preshape)
_IDENTITY_PAIR = ("blade_contact_at_retreat", "retreat_shift_exact")


def candidate_switches(config: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """The switch state of a candidates config (``candidates.json`` ``config`` or a ``library.json`` record); a missing field is off."""
    return {k: bool((config or {}).get(k, False)) for k in CANDIDATE_SWITCHES}


def library_root(root: Optional[str] = None) -> str:
    return root or os.environ.get("ISPATIALGYM_GRASP_LIBRARY_V2", DEFAULT_ROOT)


# ------------------------------------------------------------------------------------------------ identity


def canonical_scale(scale) -> Tuple[float, float, float]:
    """Applied scale as 3 floats rounded to 1e-4 (None / scalar / 3-list)."""
    if scale is None:
        return (1.0, 1.0, 1.0)
    if isinstance(scale, (int, float)):
        s = float(scale)
        return (round(s, 4),) * 3
    v = [float(x) for x in scale]
    if len(v) == 1:
        v = v * 3
    return tuple(round(x, 4) for x in v)  # type: ignore[return-value]


def scale_tag(scale) -> str:
    s = canonical_scale(scale)
    if s == (1.0, 1.0, 1.0):
        return "s1"
    if s[0] == s[1] == s[2]:
        return f"s{_num(s[0])}"
    return "s" + "x".join(_num(v) for v in s)


def _num(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def model_id(category: str, model: str, scale=None) -> str:
    return f"{category}__{model}__{scale_tag(scale)}"


def entry_id(category: str, model: str, scale=None, constraint: str = "tabletop", rest_hint_wxyz=None) -> str:
    """Library entry id: ``<category>__<model>__<scale tag>`` + ``__<constraint>`` (non-tabletop) + ``__r<digest>`` (a designated
    non-canonical rest) - one directory per (model, scale, constraint, designated rest)."""
    out = model_id(category, model, scale)
    if constraint and constraint != "tabletop":
        out += f"__{constraint}"
    if rest_hint_wxyz is not None:
        out += "__r" + digest([round(float(v), 5) for v in rest_hint_wxyz], 6)
    return out


def model_dir(category: str, model: str, scale=None, root: Optional[str] = None, constraint: str = "tabletop", rest_hint_wxyz=None) -> str:
    return os.path.join(library_root(root), "models", entry_id(category, model, scale, constraint, rest_hint_wxyz))


def job_dir(job: Dict[str, Any], root: Optional[str] = None) -> str:
    return model_dir(job["category"], job["model"], job.get("scale"), root, job.get("constraint") or "tabletop", job.get("rest_hint_wxyz"))


def digest(obj: Any, n: int = 12) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=_jsonable).encode()).hexdigest()[:n]


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(type(o))


def save_json(path: str, obj: Any) -> None:
    """Atomic write (tmp + rename): a crashed worker never leaves a truncated record."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=_jsonable)
    os.replace(tmp, path)


def load_json(path: str) -> Optional[Any]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ------------------------------------------------------------------------------------------------ math helpers (numpy, xyzw)


def quat_mul(a, b) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def quat_yaw(yaw: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])


def quat_to_mat(q) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def pose_mat(position, quat_xyzw) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(quat_xyzw)
    T[:3, 3] = np.asarray(position, dtype=float)
    return T


def mat_pose(T) -> Dict[str, List[float]]:
    T = np.asarray(T, dtype=float)
    return {"position": [round(float(v), 6) for v in T[:3, 3]], "orientation_xyzw": [round(float(v), 7) for v in mat_to_quat(T[:3, :3])]}


def pose_of(d: Dict) -> np.ndarray:
    return pose_mat(d["position"], d["orientation_xyzw"])


def yaw_of_quat(q) -> float:
    R = quat_to_mat(q)
    return math.atan2(R[1, 0], R[0, 0])


# ------------------------------------------------------------------------------------------------ rest records


@dataclass
class Rest:
    rest_id: str
    quat_xyzw: np.ndarray
    h_root: float
    reference_yaw: float
    record: Dict[str, Any]

    @staticmethod
    def from_record(d: Dict[str, Any]) -> "Rest":
        if d.get("schema") != REST:
            raise IncompatibleRecord(f"rest record schema {d.get('schema')!r}, expected {REST}")
        return Rest(rest_id=d["rest_id"], quat_xyzw=np.asarray(d["quat_xyzw"], dtype=float), h_root=float(d["h_root"]),
                    reference_yaw=float(d.get("reference_yaw", 0.0)), record=d)

    @property
    def stable(self) -> bool:
        return bool(self.record.get("stable"))


def place_object(rest: Rest, yaw: float, cell_xy: Sequence[float], support_z: float) -> np.ndarray:
    """T_world_object of the placement contract (module docstring)."""
    q = quat_mul(quat_yaw(yaw), rest.quat_xyzw)
    return pose_mat([float(cell_xy[0]), float(cell_xy[1]), float(support_z) + rest.h_root], q)


def world_eef(rest: Rest, yaw: float, cell_xy, support_z: float, T_object_eef) -> np.ndarray:
    return place_object(rest, yaw, cell_xy, support_z) @ np.asarray(T_object_eef, dtype=float)


def same_rest(a: Dict[str, Any], b: Dict[str, Any], rot_tol_rad: float = 0.05, h_tol_m: float = 0.003) -> bool:
    """Two measured rests are the same configuration when their tilts (yaw removed) agree within ``rot_tol_rad`` and their root
    heights within ``h_tol_m``."""
    return tilt_distance(a["quat_xyzw"], b["quat_xyzw"]) <= rot_tol_rad and abs(float(a["h_root"]) - float(b["h_root"])) <= h_tol_m


def tilt_distance(qa, qb) -> float:
    """Angle between the two orientations after removing a yaw about the vertical (yaw-invariant rest comparison)."""
    Ra, Rb = quat_to_mat(qa), quat_to_mat(qb)
    za, zb = Ra.T @ np.array([0, 0, 1.0]), Rb.T @ np.array([0, 0, 1.0])  # the world vertical in each object frame
    return float(math.acos(max(-1.0, min(1.0, float(np.dot(za, zb))))))


# ------------------------------------------------------------------------------------------------ the model library


@dataclass
class ModelLibrary:
    """A model's ``library.json`` (schema LIBRARY) with its rests and families."""

    path: str
    data: Dict[str, Any]
    rests: Dict[str, Rest] = field(default_factory=dict)
    _switches: Optional[Dict[str, bool]] = field(default=None, repr=False, compare=False)

    @property
    def category(self) -> str:
        return self.data["category"]

    @property
    def model(self) -> str:
        return self.data["model"]

    @property
    def scale(self) -> Tuple[float, float, float]:
        return canonical_scale(self.data.get("scale"))

    @property
    def policy_key(self) -> str:
        return self.data.get("policy_key", "")

    def grasps(self, rest_id: Optional[str] = None, include_twins: bool = False) -> List[Dict[str, Any]]:
        out = [g for g in self.data.get("grasps", []) if rest_id is None or g.get("rest_id") == rest_id]
        return out

    def families(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.data.get("families", {}))

    def candidate_switches(self) -> Dict[str, bool]:
        """i7-switches: the candidate switches this entry was generated with - the ``library.json`` record, else its ``candidates.json``
        config (an entry validated before the record), else all off."""
        rec = self.data.get("candidate_switches")
        if isinstance(rec, dict):
            return candidate_switches(rec)
        if self._switches is None:
            doc = None
            if self.path and os.path.exists(self.path):
                doc = load_json(os.path.join(os.path.dirname(self.path), "candidates.json"))
            self._switches = candidate_switches((doc or {}).get("config"))
        return dict(self._switches)

    def identity(self) -> Dict[str, Any]:
        """Every result-affecting factor of this library entry (for reach-map cache keys). i7-switches: the candidate switches only when
        one is on (an entry generated without them keeps its key)."""
        out = {"schema": self.data.get("schema"), "category": self.category, "model": self.model, "scale": list(self.scale),
               "dataset": self.data.get("dataset"), "policy_key": self.policy_key,
               "rests": {k: {"quat_xyzw": [round(float(v), 6) for v in r.quat_xyzw], "h_root": round(r.h_root, 6)} for k, r in self.rests.items()},
               "grasps": digest([{k: g.get(k) for k in ("name", "rest_id", "family_id", "T_object_eef", "finger_q", "width_m")} for g in self.data.get("grasps", [])]),
               "families": digest(self.data.get("families", {}))}
        sw = self.candidate_switches()
        idsw = {k: sw[k] for k in _IDENTITY_PAIR} if any(sw[k] for k in _IDENTITY_PAIR) else {}
        idsw.update({k: True for k in CANDIDATE_SWITCHES if k not in _IDENTITY_PAIR and sw.get(k)})  # i7-padsweep: a later switch only when on
        if idsw:
            out["candidate_switches"] = idsw
        return out

    def identity_key(self) -> str:
        return digest(self.identity(), 16)


def check_compatible(lib: ModelLibrary, *, scale=None, rest_id: Optional[str] = None, policy_key: Optional[str] = None,
                     schema: str = LIBRARY, candidate_switches: Optional[Dict[str, bool]] = None) -> None:
    """Raise :class:`IncompatibleRecord` when ``lib`` does not satisfy the consumer's requirements (``candidate_switches``: the state
    of each named switch, i7-switches)."""
    if candidate_switches is not None:
        have = lib.candidate_switches()
        bad = {k: have.get(k) for k, v in candidate_switches.items() if have.get(k) is not bool(v)}
        if bad:
            raise IncompatibleRecord(f"{lib.path}: candidate switches {bad} != required {dict(candidate_switches)}")
    if lib.data.get("schema") != schema:
        raise IncompatibleRecord(f"{lib.path}: schema {lib.data.get('schema')!r}, expected {schema}")
    if scale is not None and canonical_scale(scale) != lib.scale:
        raise IncompatibleRecord(f"{lib.path}: scale {lib.scale} != required {canonical_scale(scale)}")
    if rest_id is not None:
        r = lib.rests.get(rest_id)
        if r is None:
            raise IncompatibleRecord(f"{lib.path}: no rest {rest_id!r} (has {sorted(lib.rests)})")
        if not r.stable:
            raise IncompatibleRecord(f"{lib.path}: rest {rest_id!r} is not a verified stable rest")
    if policy_key is not None and lib.policy_key != policy_key:
        raise IncompatibleRecord(f"{lib.path}: contact policy {lib.policy_key!r} != required {policy_key!r}")
    for g in lib.data.get("grasps", []):
        if g.get("rest_id") not in lib.rests:
            raise IncompatibleRecord(f"{lib.path}: grasp {g.get('name')} refers to unknown rest {g.get('rest_id')!r}")


def load_model(category: str, model: str, scale=None, root: Optional[str] = None, *, constraint: str = "tabletop", rest_hint_wxyz=None,
               require: Optional[Dict[str, Any]] = None) -> Optional[ModelLibrary]:
    """The v2 entry of (category, model, scale, constraint, designated rest), checked against ``require`` (kwargs of
    :func:`check_compatible`); None when there is no v2 entry - a legacy library entry is never substituted (it has no measured rest)."""
    d = model_dir(category, model, scale, root, constraint, rest_hint_wxyz)
    data = load_json(os.path.join(d, "library.json"))
    if data is None:
        return None
    rests_d = data.get("rests") or (load_json(os.path.join(d, "rests.json")) or {}).get("rests", {})
    lib = ModelLibrary(path=os.path.join(d, "library.json"), data=data, rests={k: Rest.from_record(v) for k, v in rests_d.items()})
    check_compatible(lib, scale=scale if scale is not None else None, **(require or {}))
    return lib


def legacy_has_rest(record: Dict[str, Any]) -> bool:
    """Legacy (``grasp_library_20260905``) entries carry a rest *hint* (``rest_quat_wxyz``) but no measured rest: never a rest record."""
    return record.get("schema") == REST and "h_root" in record and "quat_xyzw" in record


# ------------------------------------------------------------------------------------------------ families


def family_member_T(family: Dict[str, Any], T_rep: np.ndarray, param: float) -> np.ndarray:
    """The member of ``family`` at ``param`` (rad for rotational, m for translational, relative to the representative):
    rotation of the representative about the family axis through its centre / translation along its axis, in the family frame
    (the object frame for ``frame == "object"``)."""
    T_rep = np.asarray(T_rep, dtype=float)
    kind = family.get("type")
    if kind == "rotational":
        a = np.asarray(family["axis"], dtype=float)
        a = a / np.linalg.norm(a)
        c = np.asarray(family["center"], dtype=float)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + math.sin(param) * K + (1 - math.cos(param)) * (K @ K)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = c - R @ c
        return M @ T_rep
    if kind == "translational":
        a = np.asarray(family["axis"], dtype=float)
        a = a / np.linalg.norm(a)
        M = np.eye(4)
        M[:3, 3] = a * float(param)
        return M @ T_rep
    return T_rep


def family_samples(family: Dict[str, Any], n: int, include_endpoints: bool = True) -> List[float]:
    """``n`` parameter values over the family's valid interval (endpoints and interior); a full turn excludes the duplicate end."""
    lo, hi = (float(v) for v in family.get("interval", (0.0, 0.0)))
    if hi <= lo or n <= 1:
        return [0.0]
    full_turn = family.get("type") == "rotational" and hi - lo >= 2 * math.pi - 1e-6
    if full_turn:
        return [lo + (hi - lo) * k / n for k in range(n)]
    return [float(v) for v in np.linspace(lo, hi, n)] if include_endpoints else [lo + (hi - lo) * (k + 1) / (n + 1) for k in range(n)]
