"""Pure geometry helpers for the sampler (torch, no simulator).

Conventions follow the rest of the repository: positions are ``(3,)`` float tensors, quaternions are xyzw,
image coordinates are ``(u, v)`` pixels with the origin at the top-left corner.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch as th

import omnigibson.utils.transform_utils as T

Pose = Tuple[th.Tensor, th.Tensor]

IDENTITY_QUAT = th.tensor([0.0, 0.0, 0.0, 1.0])


def t(x, dtype=th.float32) -> th.Tensor:
    return x.detach().to(dtype=dtype, device="cpu") if isinstance(x, th.Tensor) else th.as_tensor(x, dtype=dtype)


def quat_from_yaw(yaw: float) -> th.Tensor:
    return T.euler2quat(th.tensor([0.0, 0.0, float(yaw)]))


def yaw_of(quat) -> float:
    return float(T.quat2euler(t(quat))[2])


def quat_about_axis(axis, angle: float) -> th.Tensor:
    a = t(axis)
    a = a / th.linalg.norm(a)
    return T.axisangle2quat(a * float(angle))


def qmul(a, b) -> th.Tensor:
    return T.quat_multiply(t(a), t(b))


def qinv(a) -> th.Tensor:
    return T.quat_inverse(t(a))


def rotate(quat, vec) -> th.Tensor:
    return T.quat2mat(t(quat)) @ t(vec)


def quat_angle(q1, q2) -> float:
    d = abs(float(th.dot(t(q1) / th.linalg.norm(t(q1)), t(q2) / th.linalg.norm(t(q2)))))
    return 2.0 * math.acos(min(1.0, d))


def compose(a: Pose, b: Pose) -> Pose:
    return T.pose_transform(t(a[0]), t(a[1]), t(b[0]), t(b[1]))


def invert(a: Pose) -> Pose:
    return T.mat2pose(T.pose_inv(T.pose2mat((t(a[0]), t(a[1])))))


def relative(target: Pose, frame: Pose) -> Pose:
    """Pose of ``target`` expressed in ``frame``."""
    return T.relative_pose_transform(t(target[0]), t(target[1]), t(frame[0]), t(frame[1]))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def pose2d_to_pose(x: float, y: float, yaw: float, z: float = 0.0) -> Pose:
    return th.tensor([x, y, z], dtype=th.float32), quat_from_yaw(yaw)


# ----------------------------------------------------------------------------------------------------------------
# 2-D footprints
# ----------------------------------------------------------------------------------------------------------------


def local_points_world(local_points: th.Tensor, pose: Pose) -> th.Tensor:
    return t(local_points) @ T.quat2mat(t(pose[1])).T + t(pose[0])


def footprint_rect(points_xy: th.Tensor):
    """Axis-aligned bounds (xmin, ymin, xmax, ymax) of 2-D points."""
    lo = points_xy.min(dim=0).values
    hi = points_xy.max(dim=0).values
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def convex_hull_2d(points_xy: th.Tensor) -> th.Tensor:
    """Andrew's monotone chain; returns hull vertices CCW as an (N, 2) tensor."""
    pts = sorted({(round(float(p[0]), 6), round(float(p[1]), 6)) for p in points_xy})
    if len(pts) <= 2:
        return th.tensor(pts, dtype=th.float32).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return th.tensor(lower[:-1] + upper[:-1], dtype=th.float32)


def polygons_overlap(a: th.Tensor, b: th.Tensor, margin: float = 0.0) -> bool:
    """Separating-axis test for two convex polygons (N, 2), (M, 2); ``margin`` inflates the gap requirement."""

    def axes(poly):
        n = poly.shape[0]
        for i in range(n):
            e = poly[(i + 1) % n] - poly[i]
            yield th.tensor([-float(e[1]), float(e[0])])

    for ax in itertools.chain(axes(a), axes(b)):
        norm = float(th.linalg.norm(ax))
        if norm < 1e-9:
            continue
        ax = ax / norm
        pa = a @ ax
        pb = b @ ax
        if float(pa.max()) + margin < float(pb.min()) or float(pb.max()) + margin < float(pa.min()):
            return False
    return True


def point_in_convex_polygon(p: th.Tensor, poly: th.Tensor, margin: float = 0.0) -> bool:
    """True if ``p`` is inside the CCW convex polygon shrunk by ``margin``."""
    n = poly.shape[0]
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        nrm = th.tensor([-float(e[1]), float(e[0])])
        nrm = nrm / max(float(th.linalg.norm(nrm)), 1e-9)
        if float(th.dot(p - a, nrm)) < margin:
            return False
    return True


def polygon_inside_polygon(inner: th.Tensor, outer: th.Tensor, margin: float = 0.0) -> bool:
    return all(point_in_convex_polygon(p, outer, margin) for p in inner)


def rect_corners(cx: float, cy: float, half_x: float, half_y: float, yaw: float) -> th.Tensor:
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = sx * half_x, sy * half_y
        out.append([cx + c * lx - s * ly, cy + s * lx + c * ly])
    return th.tensor(out, dtype=th.float32)


# ----------------------------------------------------------------------------------------------------------------
# Camera model
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class CameraModel:
    """Pinhole camera; the USD sensor looks along -Z with +Y up."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_sensor(cls, focal_length_mm: float, horizontal_aperture_mm: float, width: int, height: int) -> "CameraModel":
        fx = focal_length_mm / horizontal_aperture_mm * width
        # square pixels: Omniverse derives the vertical aperture from the resolution aspect ratio
        return cls(width, height, fx, fx, width / 2.0, height / 2.0)

    @property
    def hfov(self) -> float:
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def vfov(self) -> float:
        return 2.0 * math.atan(self.height / (2.0 * self.fy))

    def project(self, cam_pose: Pose, points_world: th.Tensor):
        """Return ``(uv (N,2), depth (N,))``; depth is positive in front of the camera."""
        R = T.quat2mat(t(cam_pose[1]))
        p = (t(points_world).reshape(-1, 3) - t(cam_pose[0])) @ R  # world -> camera
        depth = -p[:, 2]
        u = self.cx + self.fx * p[:, 0] / depth.clamp(min=1e-6)
        v = self.cy - self.fy * p[:, 1] / depth.clamp(min=1e-6)
        return th.stack([u, v], dim=1), depth

    def inside(self, cam_pose: Pose, points_world: th.Tensor, margin_frac: float = 0.0) -> th.Tensor:
        uv, depth = self.project(cam_pose, points_world)
        mu, mv = margin_frac * self.width, margin_frac * self.height
        return (depth > 0.05) & (uv[:, 0] >= mu) & (uv[:, 0] <= self.width - mu) & (uv[:, 1] >= mv) & (uv[:, 1] <= self.height - mv)

    def all_inside(self, cam_pose: Pose, points_world: th.Tensor, margin_frac: float = 0.05) -> bool:
        return bool(self.inside(cam_pose, points_world, margin_frac).all())

    def optical_axis(self, cam_pose: Pose) -> th.Tensor:
        return rotate(cam_pose[1], th.tensor([0.0, 0.0, -1.0]))

    def angle_off_axis(self, cam_pose: Pose, point_world) -> float:
        d = t(point_world) - t(cam_pose[0])
        d = d / th.linalg.norm(d)
        return math.acos(max(-1.0, min(1.0, float(th.dot(d, self.optical_axis(cam_pose))))))

    def ray_plane_intersection(self, cam_pose: Pose, u: float, v: float, plane_z: float) -> Optional[th.Tensor]:
        """World point where the pixel ray hits the horizontal plane ``z = plane_z`` (None if parallel / behind)."""
        R = T.quat2mat(t(cam_pose[1]))
        d_cam = th.tensor([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        d = R @ d_cam
        o = t(cam_pose[0])
        if abs(float(d[2])) < 1e-6:
            return None
        s = (plane_z - float(o[2])) / float(d[2])
        if s <= 0:
            return None
        return o + d * s


def aabb_corners(lo, hi) -> th.Tensor:
    lo, hi = t(lo), t(hi)
    return th.tensor([[float(lo[0] if sx == 0 else hi[0]), float(lo[1] if sy == 0 else hi[1]), float(lo[2] if sz == 0 else hi[2])]
                      for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)])


# ----------------------------------------------------------------------------------------------------------------
# Cube orientations (block design)
# ----------------------------------------------------------------------------------------------------------------

FACE_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def opposite_face(face: str) -> str:
    return ("-" if face[0] == "+" else "+") + face[1]


def cube_rotations() -> List[th.Tensor]:
    """The 24 proper rotations of a cube as xyzw quaternions (rotation matrices with entries in {-1, 0, 1})."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = th.zeros(3, 3)
            for i, (p, s) in enumerate(zip(perm, signs)):
                m[i, p] = float(s)
            if abs(float(th.det(m)) - 1.0) < 1e-6:
                mats.append(m)
    return [T.mat2quat(m) for m in mats]


def rotation_with_face_up(face: str, front_face: Optional[str] = None) -> th.Tensor:
    """Quaternion (block frame -> world) that puts ``face`` along +z; if ``front_face`` is given it points along
    -x (toward a robot standing at -x looking at +x), otherwise the yaw is the smallest-angle choice."""
    up = th.tensor(FACE_AXES[face], dtype=th.float32)
    best = None
    for q in cube_rotations():
        R = T.quat2mat(q)
        if float(th.dot(R @ up, th.tensor([0.0, 0.0, 1.0]))) < 0.99:
            continue
        if front_face is not None:
            f = th.tensor(FACE_AXES[front_face], dtype=th.float32)
            if float(th.dot(R @ f, th.tensor([-1.0, 0.0, 0.0]))) < 0.99:
                continue
            return q
        ang = quat_angle(q, IDENTITY_QUAT)
        if best is None or ang < best[0]:
            best = (ang, q)
    if best is None:
        raise ValueError(f"no cube rotation puts {face} up with {front_face} toward -x")
    return best[1]


def face_normal_world(face: str, quat) -> th.Tensor:
    return rotate(quat, th.tensor(FACE_AXES[face], dtype=th.float32))


def up_face(quat, faces: Iterable[str] = FACE_AXES) -> str:
    best, best_dot = None, -2.0
    for f in faces:
        d = float(face_normal_world(f, quat)[2])
        if d > best_dot:
            best, best_dot = f, d
    return best


def sample_unit_vector(rng) -> th.Tensor:
    v = th.tensor(rng.normal(size=3), dtype=th.float32)
    return v / th.linalg.norm(v)


def uniform_in(rng, rng_pair: Sequence[float]) -> float:
    lo, hi = float(rng_pair[0]), float(rng_pair[1])
    return float(rng.uniform(lo, hi)) if hi > lo else lo
