"""i7-attachfix 2026-11-08 (owner direction 2026-11-08 02:20 item 3): every recorded IK witness reaches its pose before the instance is
written - an independent numpy forward kinematics of the R1Pro (the planner's derived holonomic URDF,
:func:`ispatialgym.oracle.r1pro_model.generate_holonomic_urdf`; root ``base_footprint_x`` at the world identity) applied to each
``ik_witnesses[].targets[].q`` (canonical: "root at the world origin (identity pose, z = 0): FK(q) puts base_link at base_pose2d on
the measured floor") and compared with the target's ``pose``.

Campaign scan (2026-11-08, 2463 written sidecars / 24399 targets under $OFH/out): FK vs pose p99 1.60 mm, max 3.46 mm / 0.010 rad;
the sampler's recorded ``fk_residual_m`` agrees with this FK to 0.024 mm. The attachment_01 / 04 "77 / 576 mm witness residual"
failures were not witness misses (their witnesses reach within 0.01 / 0.9 mm): the collector world started those instances with the
lid off its TRO pose (``oracle/run_isg.py`` ``detach_scene_attachments`` / ``tro_pose_mismatches``). The check here guards the writer.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: a witness whose FK misses its pose by more than this is not written (the oracle closes residuals <= 5 mm; the program check's own
#: limits are 2 mm, 3.5 mm for pre-poses - the largest recorded today is 3.46 mm)
WITNESS_POS_TOL_M = 0.005
WITNESS_ROT_TOL_RAD = 0.05


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(a: Sequence[float], t: float) -> np.ndarray:
    a = np.asarray(a, float) / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def quat_to_mat(q: Sequence[float]) -> np.ndarray:
    """xyzw -> 3x3."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _T(R: np.ndarray, p: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


class NumpyFK:
    """Forward kinematics of ``<arm>_eef_link`` from a 28-D q (``r1pro_model.ALL_JOINTS`` order), root at the world identity."""

    def __init__(self, urdf_path: Optional[str] = None):
        from ispatialgym.oracle import r1pro_model as M

        path = urdf_path or M.generate_holonomic_urdf()
        root = ET.parse(path).getroot()
        by_child: Dict[str, Dict[str, Any]] = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()]
            r = [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
            ax = j.find("axis")
            by_child[j.find("child").get("link")] = {"name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
                                                     "T": _T(_rpy(*r), xyz), "axis": [float(v) for v in ax.get("xyz").split()] if ax is not None else [1.0, 0, 0]}
        self.idx = {n: i for i, n in enumerate(M.ALL_JOINTS)}
        self.chains: Dict[str, List[Dict[str, Any]]] = {}
        for arm in M.ARMS:
            link, chain = M.EEF_LINK[arm], []
            while link in by_child:
                chain.append(by_child[link])
                link = by_child[link]["parent"]
            if link != M.ROOT_LINK:
                raise ValueError(f"{arm} eef chain ends at {link!r}, not the root {M.ROOT_LINK!r}")
            self.chains[arm] = chain[::-1]

    def eef(self, q: Sequence[float], arm: str) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        T = np.eye(4)
        for j in self.chains[arm]:
            T = T @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                T = T @ _T(_axis_angle(j["axis"], q[self.idx[j["name"]]]), [0.0, 0.0, 0.0])
            elif j["type"] == "prismatic":
                a = np.asarray(j["axis"], float)
                T = T @ _T(np.eye(3), a / np.linalg.norm(a) * q[self.idx[j["name"]]])
        return T


_FK: Dict[str, NumpyFK] = {}


def default_fk() -> NumpyFK:
    if "default" not in _FK:
        _FK["default"] = NumpyFK()
    return _FK["default"]


def witness_errors(ik_witnesses: Sequence[Dict[str, Any]], fk: Any) -> List[Dict[str, Any]]:
    """Per recorded witness target with a ``q`` and a ``pose``: ``{"action", "name", "key", "arm", "pos_m", "rot_rad"}`` of FK(q) vs
    pose (a target may name its own ``arm``, e.g. a base push's second contact). Targets without a pose (``regrasp`` q-only) are skipped."""
    out = []
    for w in ik_witnesses or []:
        for key, t in (w.get("targets") or {}).items():
            if not isinstance(t, dict) or t.get("q") is None or not isinstance(t.get("pose"), dict):
                continue
            arm = t.get("arm") or w.get("arm")
            T = fk.eef(t["q"], arm)
            p = np.asarray(t["pose"]["position"], dtype=float)
            R = quat_to_mat(t["pose"]["orientation"])
            c = (np.trace(T[:3, :3].T @ R) - 1.0) / 2.0
            out.append({"action": w.get("action"), "name": w.get("name"), "key": key, "arm": arm,
                        "pos_m": float(np.linalg.norm(T[:3, 3] - p)), "rot_rad": float(math.acos(max(-1.0, min(1.0, c))))})
    return out


def witness_violations(ik_witnesses: Sequence[Dict[str, Any]], fk: Any = None, pos_tol: float = WITNESS_POS_TOL_M,
                       rot_tol: float = WITNESS_ROT_TOL_RAD) -> List[Dict[str, Any]]:
    """The witness targets whose FK misses the pose by more than ``pos_tol`` / ``rot_tol`` (:func:`witness_errors`)."""
    fk = fk or default_fk()
    return [e for e in witness_errors(ik_witnesses, fk) if e["pos_m"] > pos_tol or e["rot_rad"] > rot_tol]


def enabled() -> bool:
    """ISPATIALGYM_WITNESS_WRITE_CHECK=0 turns the write-time check off (default on)."""
    return os.environ.get("ISPATIALGYM_WITNESS_WRITE_CHECK", "1") != "0"
