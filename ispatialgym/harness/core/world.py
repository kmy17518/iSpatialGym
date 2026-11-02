"""The arm planner's world, as RoboDojo's ``CuroboPlanner`` builds it: one slab under the surface the robot is mounted on,
fixed for the episode and never shown to the model. RoboDojo's arms are bolted onto the table top and its slab (5 cm thick)
has its top 2.5 cm under their mounting point; R1Pro stands on the floor, so its slab lies 2.5 cm under the floor beneath
the base. Built from robot geometry alone: nothing of the scene is in it (RoboDojo leaves out the objects on its table, and
the house's furniture is left out here).

A start in collision refuses every motion, so if one of the robot's collision spheres comes within :data:`CLEARANCE_M` of
the slab as the episode starts, the slab is lowered under it. The geometry is plain numpy; :func:`build_world` reads the
robot."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ispatialgym.harness.core import pose

#: RoboDojo's slab: its thickness, and how far its top sits under the mounting surface
SLAB_M = 0.05
SLAB_BELOW_SURFACE_M = 0.025
FLOOR_SPAN_M = 20.0
#: clearance kept between the slab and the robot's collision spheres at the start
CLEARANCE_M = 0.02
_EPS = 1e-6


@dataclass(frozen=True)
class Box:
    """A cuboid turned about the world z axis by ``yaw``; ``dims`` are its full extents along its own axes."""

    name: str
    center: Tuple[float, float, float]
    yaw: float
    dims: Tuple[float, float, float]
    source: str
    notes: Tuple[str, ...] = ()

    @property
    def top(self) -> float:
        return float(self.center[2] + self.dims[2] / 2)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "center": [round(float(v), 4) for v in self.center], "yaw_deg": round(math.degrees(self.yaw), 2),
                "dims": [round(float(v), 4) for v in self.dims], "source": self.source, "notes": list(self.notes)}


def floor_slab(floor_z: float, center_xy: Sequence[float]) -> Box:
    top = float(floor_z) - SLAB_BELOW_SURFACE_M
    return Box("floor", (float(center_xy[0]), float(center_xy[1]), top - SLAB_M / 2), 0.0, (FLOOR_SPAN_M, FLOOR_SPAN_M, SLAB_M),
               f"floor at z {float(floor_z):.3f}")


def overlapping(box: Box, spheres: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Which spheres ``(N, 4)`` (x y z r, world) come within ``margin`` of the box (points inside it included)."""
    spheres = np.asarray(spheres, dtype=float).reshape(-1, 4)
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    d = spheres[:, :3] - np.asarray(box.center)
    p = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
    h = np.asarray(box.dims) / 2
    return (np.linalg.norm(p - np.clip(p, -h, h), axis=1) < spheres[:, 3] + margin) | np.all(np.abs(p) <= h, axis=1)


def lowered(box: Box, spheres: np.ndarray, why: str, margin: float = CLEARANCE_M) -> Box:
    """``box`` moved down until none of ``spheres`` comes within ``margin`` of it."""
    spheres = np.asarray(spheres, dtype=float).reshape(-1, 4)
    hit = overlapping(box, spheres, margin)
    if not hit.any():
        return box
    dz = min(0.0, float(np.min(spheres[hit, 2] - spheres[hit, 3])) - margin - _EPS - box.top)
    x, y, z = box.center
    return replace(box, center=(x, y, z + dz), notes=box.notes + (f"lowered {-1000 * dz:.0f} mm under {why} at the start",))


def scene_model(boxes: Sequence[Box], root_pos: Sequence[float], root_quat_xyzw: Sequence[float]) -> Dict[str, Any]:
    """cuRobo ``scene_model`` of the boxes in the robot root frame (cuRobo poses: x y z qw qx qy qz)."""
    r_root = pose.quat_to_matrix(np.asarray(root_quat_xyzw, dtype=float))
    t_root = np.asarray(root_pos, dtype=float)
    cuboids = {}
    for b in boxes:
        c, s = math.cos(b.yaw), math.sin(b.yaw)
        r = r_root.T @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        p = r_root.T @ (np.asarray(b.center, dtype=float) - t_root)
        q = pose.matrix_to_quat(r)
        cuboids[b.name] = {"dims": [float(v) for v in b.dims], "pose": [float(p[0]), float(p[1]), float(p[2]), float(q[3]), float(q[0]), float(q[1]), float(q[2])]}
    return {"cuboid": cuboids}


def build_world(emb) -> List[Box]:
    """The planner's boxes for the episode (needs the simulator): the floor slab under the base as the episode starts,
    cleared of the planner's collision spheres at the start configuration."""
    base_pos, _ = emb.base_pose_world()
    spheres, _ = emb.backend.robot_spheres_world(emb.q_full(), "whole_body")
    sp = spheres.reshape(-1, 4).detach().cpu().double().numpy()
    floor = floor_slab(float(base_pos[2]), (float(base_pos[0]), float(base_pos[1])))
    return [lowered(floor, sp[sp[:, 3] > 0], "the robot")]
