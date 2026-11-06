"""The R1Pro hand's collision geometry in the eef frame - what PhysX collides with, for exact contact-policy checks.

The shipped URDF names collision meshes (``meshes/collision/*-col-0.obj``) that are not in the robot assets; PhysX uses the USD's
collision prims. :func:`export` reads those prims from a running OmniGibson robot once and writes
``ispatialgym/assets/robot/r1pro_hand_collision.json``: for each arm and each link rigidly attached to the eef or sliding on it,
the convex-hull vertices of every collision mesh (world points of the prim -> eef frame), the link's role and, for the fingers,
the slide axis (a finger at joint value ``q`` is its ``q = 0`` hull moved by ``q * axis``).

Roles (explicit link membership, never a name prefix - see :mod:`ispatialgym.grasping.contact_policy`):

* ``finger``       ``{arm}_gripper_finger_link1``, ``{arm}_gripper_finger_link2``
* ``palm``         ``{arm}_gripper_link``
* ``wrist_camera`` ``{arm}_realsense_link`` (the camera and its mount)
* ``wrist``        ``{arm}_arm_link7`` (fixed to the gripper)

Without the exported file (tooling, tests) :func:`load` falls back to the URDF *visual* meshes (``source = urdf_visual``) and says so;
policy versions record which geometry they ran on.

    python -m ispatialgym.grasping.hand_geometry export     # needs Isaac Sim; writes the asset json
    python -m ispatialgym.grasping.hand_geometry info       # prints the loaded geometry and its source
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ispatialgym import REPO_ROOT

ASSET_PATH = os.path.join(REPO_ROOT, "ispatialgym", "assets", "robot", "r1pro_hand_collision.json")
FORMAT = "isg-hand-collision/2"  # 2: per-prim convex pieces
ARMS = ("right", "left")
ROLES = ("finger", "palm", "wrist_camera", "wrist")


def role_links(arm: str) -> Dict[str, Tuple[str, ...]]:
    """Explicit link membership per role (no prefix matching: ``{arm}_gripper_link`` is the palm, not a finger)."""
    return {
        "finger": (f"{arm}_gripper_finger_link1", f"{arm}_gripper_finger_link2"),
        "palm": (f"{arm}_gripper_link",),
        "wrist_camera": (f"{arm}_realsense_link",),
        "wrist": (f"{arm}_arm_link7",),
    }


def role_of(link: str, arm: str) -> Optional[str]:
    for role, links in role_links(arm).items():
        if link in links:
            return role
    return None


@dataclass
class LinkHull:
    link: str
    role: str
    #: convex-hull vertices in the eef frame (finger: at joint value 0)
    points: np.ndarray
    #: finger slide axis in the eef frame (unit); None for rigid links
    axis: Optional[np.ndarray] = None
    joint: Optional[str] = None
    #: the convex pieces PhysX collides with (each a hull vertex array, eef frame, finger at 0); ``points`` is their concatenation
    pieces: Optional[List[np.ndarray]] = None

    def at(self, q: float = 0.0) -> np.ndarray:
        return self.points if self.axis is None else self.points + float(q) * self.axis[None]


@dataclass
class HandGeometry:
    arm: str
    links: List[LinkHull]
    source: str
    finger_open: float = 0.05
    finger_closed: float = 0.0
    meta: Dict = field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Content hash of the geometry (part of every contact-policy key)."""
        h = hashlib.sha1()
        for lk in sorted(self.links, key=lambda l: l.link):
            h.update(lk.link.encode())
            h.update(np.round(lk.points, 6).astype(np.float64).tobytes())
            if lk.axis is not None:
                h.update(np.round(lk.axis, 6).astype(np.float64).tobytes())
        return h.hexdigest()[:12]

    def by_role(self, role: str) -> List[LinkHull]:
        return [l for l in self.links if l.role == role]

    def points(self, roles: Sequence[str] = ROLES, finger_q: Tuple[float, float] = (0.05, 0.05)) -> Tuple[np.ndarray, np.ndarray]:
        """``(P (N, 3) eef-frame points, role index (N,))`` of the hulls of ``roles``; fingers at ``finger_q`` (joint1, joint2)."""
        pts, idx = [], []
        for lk in self.links:
            if lk.role not in roles:
                continue
            q = 0.0
            if lk.role == "finger":
                q = finger_q[0] if lk.link.endswith("1") else finger_q[1]
            p = lk.at(q)
            pts.append(p)
            idx.append(np.full(len(p), ROLES.index(lk.role), dtype=np.int8))
        if not pts:
            return np.zeros((0, 3)), np.zeros(0, dtype=np.int8)
        return np.concatenate(pts), np.concatenate(idx)

    def surface(self, spacing: float = 0.003, roles: Sequence[str] = ROLES, finger_q: Tuple[float, float] = (0.05, 0.05)) -> Tuple[np.ndarray, np.ndarray]:
        """Hull vertices plus points on the hull faces at about ``spacing`` (m): against a box support the deepest point of a convex
        hull is a vertex for the faces of the box, but a box EDGE can cut a hull face between vertices; face samples bound that miss
        by the spacing."""
        P, R = [], []
        for lk in self.links:
            if lk.role not in roles:
                continue
            q = finger_q[0] if (lk.role == "finger" and lk.link.endswith("1")) else (finger_q[1] if lk.role == "finger" else 0.0)
            s = np.concatenate([_hull_surface(lk.link, pc.tobytes(), spacing) for pc in (lk.pieces or [lk.points])])
            p = s + (float(q) * lk.axis[None] if lk.axis is not None else 0.0)
            P.append(p)
            R.append(np.full(len(p), ROLES.index(lk.role), dtype=np.int8))
        return np.concatenate(P), np.concatenate(R)


@lru_cache(maxsize=64)
def _hull_surface(link: str, pts_bytes: bytes, spacing: float) -> np.ndarray:
    from scipy.spatial import ConvexHull

    pts = np.frombuffer(pts_bytes, dtype=np.float64).reshape(-1, 3)
    hull = ConvexHull(pts)
    out = [pts]
    rng = np.random.default_rng(0)
    for tri in hull.simplices:
        a, b, c = pts[tri]
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        n = int(math.ceil(area / (spacing * spacing)))
        if n <= 0:
            continue
        u = rng.random((n, 2))
        flip = u.sum(1) > 1
        u[flip] = 1 - u[flip]
        out.append(a + u[:, :1] * (b - a) + u[:, 1:] * (c - a))
    return np.concatenate(out)


def _from_json(d: Dict, arm: str) -> HandGeometry:
    a = d["arms"][arm]
    links = []
    for n, v in a["links"].items():
        pieces = [np.asarray(pc, dtype=np.float64) for pc in v["pieces"]] if v.get("pieces") else None
        pts = np.concatenate(pieces) if pieces else np.asarray(v["points"], dtype=np.float64)
        links.append(LinkHull(link=n, role=v["role"], points=pts, axis=None if v.get("axis") is None else np.asarray(v["axis"], dtype=np.float64),
                              joint=v.get("joint"), pieces=pieces))
    return HandGeometry(arm=arm, links=links, source=d.get("source", "usd_collision"), finger_open=float(d.get("finger_open", 0.05)),
                        finger_closed=float(d.get("finger_closed", 0.0)), meta={k: d[k] for k in ("format", "exported", "usd", "approximations") if k in d})


def _urdf_visual(arm: str) -> HandGeometry:
    """Fallback: convex hulls of the URDF visual meshes (as ``r1pro_model.finger_hulls_eef``), every hand link."""
    import xml.etree.ElementTree as ET

    from scipy.spatial import ConvexHull

    from ispatialgym.oracle import r1pro_model as M

    root = ET.parse(M.source_urdf_path()).getroot()
    joints = {j.find("child").get("link"): j for j in root.iter("joint") if j.find("child") is not None}
    R_ge = M._rpy_matrix(*M.EEF_IN_GRIPPER_RPY)
    t_ge = np.asarray(M.EEF_IN_GRIPPER_XYZ, dtype=float)

    def origin(j):
        o = j.find("origin")
        xyz = np.array([float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()])
        rpy = [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
        return xyz, M._rpy_matrix(*rpy)

    out = []
    gripper = M.GRIPPER_LINK[arm]
    for role, names in role_links(arm).items():
        for name in names:
            lk = [l for l in root.iter("link") if l.get("name") == name][0]
            vis = lk.find("visual")
            mesh = vis.find("geometry").find("mesh")
            path = os.path.join(M.robot_assets_dir(), "urdf", mesh.get("filename"))
            scale = np.array([float(v) for v in (mesh.get("scale") or "1 1 1").split()])
            pts = np.array([[float(v) for v in line.split()[1:4]] for line in open(path) if line.startswith("v ")]) * scale
            vo = vis.find("origin")
            if vo is not None:
                pts = pts @ M._rpy_matrix(*[float(v) for v in (vo.get("rpy") or "0 0 0").split()]).T + np.array([float(v) for v in (vo.get("xyz") or "0 0 0").split()])
            axis = None
            jname = None
            if name == gripper:
                pts_g = pts
            elif joints[name].find("parent").get("link") == gripper:  # fingers, camera: children of the gripper link
                xyz, R = origin(joints[name])
                pts_g = pts @ R.T + xyz
                if joints[name].get("type") == "prismatic":
                    axis_g = R @ np.array([float(v) for v in joints[name].find("axis").get("xyz").split()])
                    axis = R_ge.T @ axis_g
                    jname = joints[name].get("name")
            else:  # arm_link7: the gripper's parent (fixed joint)
                xyz, R = origin(joints[gripper])
                pts_g = (pts - xyz) @ R
            pts_e = (pts_g - t_ge) @ R_ge
            hull = pts_e[ConvexHull(pts_e).vertices]
            out.append(LinkHull(link=name, role=role, points=hull.astype(float), axis=axis, joint=jname))
    return HandGeometry(arm=arm, links=out, source="urdf_visual")


@lru_cache(maxsize=4)
def load(arm: str = "right", path: Optional[str] = None) -> HandGeometry:
    """The exported USD collision hulls (``ASSET_PATH``), else the URDF visual-mesh fallback (``source == "urdf_visual"``)."""
    p = path or os.environ.get("ISPATIALGYM_HAND_COLLISION", ASSET_PATH)
    if os.path.exists(p):
        with open(p) as f:
            return _from_json(json.load(f), arm)
    return _urdf_visual(arm)


# ------------------------------------------------------------------------------------------------ export (Isaac Sim)


def export(out_path: str = ASSET_PATH) -> Dict:
    """Read the collision prims of every hand link from a simulated R1Pro, express them in the eef frame and write ``out_path``."""
    from ispatialgym.oracle.curobo_planner import R1ProCuRobo  # noqa: F401  (warp pinning before omnigibson)
    import omnigibson as og
    import omnigibson.utils.transform_utils as T
    import torch as th
    from scipy.spatial import ConvexHull

    from ispatialgym.oracle import r1pro_model as M

    env = og.Environment(configs={"env": {"action_frequency": 30, "physics_frequency": 120}, "scene": {"type": "Scene"},
                                  "robots": [M.joylo_robot_config("robot", obs_modalities=["proprio"])]})
    robot = env.scene.robots[0]
    out = {"format": FORMAT, "source": "usd_collision", "finger_open": 0.05, "finger_closed": 0.0, "arms": {}, "approximations": {}}
    for arm in ARMS:
        q = robot.get_joint_positions().clone()
        q[robot.gripper_control_idx[arm]] = 0.0  # fingers at joint value 0: the hulls' reference
        robot.set_joint_positions(q)
        robot.keep_still()
        og.sim.step_physics()
        robot.set_joint_positions(q)
        eef = robot.eef_links[arm]
        ep, eq = (x.float() for x in eef.get_position_orientation())
        inv = T.pose_inv(T.pose2mat((ep, eq)))
        links = {}
        for role, names in role_links(arm).items():
            for name in names:
                link = robot.links[name]
                pts = []
                for mname, mesh in link.collision_meshes.items():
                    w = mesh.points_in_world_frame
                    if w is None or len(w) == 0:
                        continue
                    pts.append(w.float())
                    try:
                        out["approximations"][f"{name}/{mname}"] = str(mesh.prim.GetAttribute("physics:approximation").Get())
                    except Exception:  # noqa: BLE001
                        pass
                pieces = []
                for P in pts:  # every collision prim is a convexHull approximation: keep the pieces apart (their union is concave)
                    Pe = ((inv[:3, :3] @ P.T).T + inv[:3, 3]).numpy().astype(np.float64)
                    pieces.append(np.round(Pe[ConvexHull(Pe).vertices], 6).tolist())
                rec = {"role": role, "pieces": pieces, "n_meshes": len(pts)}
                if role == "finger":
                    jn = f"{arm}_gripper_finger_joint{name[-1]}"
                    # measured slide axis: the link origin at q = 0.05 minus at q = 0, in the eef frame
                    p0 = link.get_position_orientation()[0].float()
                    q2 = q.clone()
                    q2[list(robot.joints.keys()).index(jn)] = 0.05
                    robot.set_joint_positions(q2)
                    og.sim.step_physics()
                    robot.set_joint_positions(q2)
                    p1 = link.get_position_orientation()[0].float()
                    robot.set_joint_positions(q)
                    og.sim.step_physics()
                    robot.set_joint_positions(q)
                    d = (inv[:3, :3] @ (p1 - p0)).numpy() / 0.05
                    rec["axis"] = np.round(d / max(1e-9, np.linalg.norm(d)), 6).tolist()
                    rec["axis_measured_norm"] = round(float(np.linalg.norm(d)), 4)
                    rec["joint"] = jn
                links[name] = rec
        out["arms"][arm] = {"eef_link": M.EEF_LINK[arm], "links": links}
    out["exported"] = __import__("time").strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)
    og.shutdown()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["export", "info"])
    ap.add_argument("--out", default=ASSET_PATH)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        export(a.out)
        return
    for arm in ARMS:
        g = load(arm)
        print(f"{arm}: source {g.source} digest {g.digest}")
        for lk in g.links:
            lo, hi = lk.points.min(0), lk.points.max(0)
            print(f"  {lk.role:12s} {lk.link:30s} {len(lk.points):4d} pts  z [{lo[2]:+.4f}, {hi[2]:+.4f}]  y [{lo[1]:+.4f}, {hi[1]:+.4f}]"
                  + (f"  axis {np.round(lk.axis, 3).tolist()}" if lk.axis is not None else ""))


if __name__ == "__main__":
    main()


@lru_cache(maxsize=8)
def _hand_spheres(arm: str, q1: float, q2: float) -> Tuple[Tuple[str, Tuple[float, float, float], float], ...]:
    import xml.etree.ElementTree as ET

    from ispatialgym.oracle import r1pro_model as M

    kin = M.load_v1_kinematics_yaml()
    cs = kin["collision_spheres"]
    root = ET.parse(M.source_urdf_path()).getroot()
    joints = {j.find("child").get("link"): j for j in root.iter("joint") if j.find("child") is not None}
    R_ge = M._rpy_matrix(*M.EEF_IN_GRIPPER_RPY)
    t_ge = np.asarray(M.EEF_IN_GRIPPER_XYZ, dtype=float)
    gripper = M.GRIPPER_LINK[arm]
    out = []
    for role, names in role_links(arm).items():
        for name in names:
            for s in cs.get(name, []):
                p = np.asarray(s["center"], dtype=float)
                if name == gripper:
                    pg = p
                elif joints[name].find("parent").get("link") == gripper:
                    o = joints[name].find("origin")
                    xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
                    R = M._rpy_matrix(*[float(v) for v in (o.get("rpy") or "0 0 0").split()])
                    pg = R @ p + xyz
                    if joints[name].get("type") == "prismatic":
                        ax = R @ np.array([float(v) for v in joints[name].find("axis").get("xyz").split()])
                        pg = pg + (q1 if name.endswith("1") else q2) * ax
                else:
                    o = joints[gripper].find("origin")
                    xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()])
                    R = M._rpy_matrix(*[float(v) for v in (o.get("rpy") or "0 0 0").split()])
                    pg = R.T @ (p - xyz)
                pe = (pg - t_ge) @ R_ge
                out.append((name, tuple(float(v) for v in pe), float(s["radius"])))
    return tuple(out)


def hand_spheres_eef(arm: str = "right", finger_q: Tuple[float, float] = (0.05, 0.05), buffer: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """The cuRobo collision spheres of the hand links (fingers at ``finger_q``, palm, wrist camera, wrist) in the eef frame, from the
    robot's ``r1pro_description_curobo_default.yaml`` through the URDF joints: ``(centres (S, 3), radii (S,), link per sphere)``;
    ``buffer`` is added to the radii (the model's ``collision_sphere_buffer`` is 0.002)."""
    sp = _hand_spheres(arm, round(float(finger_q[0]), 5), round(float(finger_q[1]), 5))
    return (np.array([c for _, c, _ in sp]), np.array([r for _, _, r in sp]) + float(buffer), [n for n, _, _ in sp])
