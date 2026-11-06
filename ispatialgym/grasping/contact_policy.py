"""The versioned contact policy between the robot and the grasp's support, shared by candidate generation, physics validation and
reach-map IK (owner prompts 2026-11-02, ISG1020 grasp library / reach bands).

Rules (``ContactPolicy`` defaults, version :data:`POLICY_VERSION`):

1. The support stays in the collision world for EVERY robot part. Nothing is ignored, no whole hand or robot is exempted.
2. Only the named finger collision shapes (``hand_geometry.role_links(arm)["finger"]``: ``{arm}_gripper_finger_link1/2``) may touch
   the support, and penetrate it by at most ``finger_support_max_penetration_m`` = 2 mm: signed clearance >= -0.002 m, measured on
   the fingers' actual collision hulls (vertices + face samples, :mod:`hand_geometry`), never on the inflated cuRobo spheres. This is
   a permitted maximum, not a target depth.
3. Every other part keeps the normal rejection: the palm (``{arm}_gripper_link``), the wrist camera and its mount
   (``{arm}_realsense_link``) and the wrist (``{arm}_arm_link7``) by their exact hulls (clearance >= ``hand_support_margin_m``) AND by
   their cuRobo spheres (with the model's 2 mm sphere buffer, clearance >= ``sphere_support_margin_m``); the arm, torso, base and head
   by their spheres. A side grasp whose camera mount meets the table fails even when its fingers pass.
4. The allowance is specific to the (finger, support) pair. Finger-vs-object, finger-vs-other-world and self-collision checks are
   unchanged; the sphere backends that cannot express pair tolerances run with the finger spheres switched off against the SUPPORT
   only and add the explicit finger check of rule 2 (``reach_maps`` / ``validate_tabletop``).
5. Floor mode (an object on the floor) uses the same finger rule against the floor plane (``floor_finger_max_penetration_m``); the
   robot's own ground-contact links (wheels, base) are scoped separately - they stand on the floor, which is not a grasp contact.
6. The rules apply at the pre-grasp, along the approach (``approach_step_m``) and through the closing sweep, also when a physics trial
   starts at the grasp pose.

Sphere padding vs actual geometry: the cuRobo finger spheres (r 5-10 mm + 2 mm buffer) sit inside the finger bodies and do not reach
their pad faces; a 2 mm allowance on those spheres would allow ~0-9 mm on the real finger depending on where it touches. That is why
rule 2 is evaluated on the hulls; ``tests/test_contact_policy.py`` measures the sphere-vs-hull offsets per finger.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.grasping import hand_geometry as H

POLICY_VERSION = 1
POLICY_NAME = "isg-support-contact"

REASONS = ("ok", "finger_penetration", "palm_contact", "wrist_camera_contact", "wrist_contact", "link_sphere_contact")
ROLE_REASON = {"palm": "palm_contact", "wrist_camera": "wrist_camera_contact", "wrist": "wrist_contact"}


@dataclass(frozen=True)
class ContactPolicy:
    finger_support_max_penetration_m: float = 0.002
    hand_support_margin_m: float = 0.0
    sphere_support_margin_m: float = 0.0
    hand_sphere_check: bool = True
    floor_finger_max_penetration_m: float = 0.002
    approach_step_m: float = 0.01
    sweep_steps: int = 5
    surface_spacing_m: float = 0.003
    version: int = POLICY_VERSION

    def finger_links(self, arm: str) -> Tuple[str, ...]:
        return H.role_links(arm)["finger"]

    def hand_rigid_links(self, arm: str) -> Tuple[str, ...]:
        r = H.role_links(arm)
        return r["palm"] + r["wrist_camera"] + r["wrist"]

    def meta(self, arm: str = "right") -> Dict:
        g = H.load(arm)
        return {"name": POLICY_NAME, **asdict(self), "arm": arm, "finger_links": list(self.finger_links(arm)),
                "hand_rigid_links": list(self.hand_rigid_links(arm)), "geometry_source": g.source, "geometry_digest": g.digest}

    def key(self, arm: str = "right") -> str:
        """Content key: every result-affecting field plus the hand geometry digest (cache identity of bands / validations)."""
        return f"{POLICY_NAME}-v{self.version}-" + hashlib.sha1(json.dumps(self.meta(arm), sort_keys=True).encode()).hexdigest()[:12]


DEFAULT = ContactPolicy()


# ------------------------------------------------------------------------------------------------ support geometry


@dataclass
class Support:
    """A support in the world: ``kind`` "plane" (solid below ``z``; the local support under an object) or "box" (a slab, centre /
    half-extents / yaw about z). Signed distance > 0 outside."""

    kind: str
    z: float = 0.0
    centre: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    half: Tuple[float, float, float] = (1.0, 1.0, 0.025)
    yaw: float = 0.0
    name: str = "support"

    @staticmethod
    def plane(z: float, name: str = "support") -> "Support":
        return Support(kind="plane", z=float(z), name=name)

    @staticmethod
    def box(centre, half, yaw: float = 0.0, name: str = "support") -> "Support":
        return Support(kind="box", centre=tuple(float(v) for v in centre), half=tuple(float(v) for v in half), yaw=float(yaw), name=name)

    @property
    def top_z(self) -> float:
        return self.z if self.kind == "plane" else self.centre[2] + self.half[2]

    def sdf(self, p: th.Tensor) -> th.Tensor:
        """Signed distance of points ``p`` (..., 3)."""
        if self.kind == "plane":
            return p[..., 2] - self.z
        c = th.tensor(self.centre, dtype=p.dtype, device=p.device)
        h = th.tensor(self.half, dtype=p.dtype, device=p.device)
        d = p - c
        if self.yaw:
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            d = th.stack([cy * d[..., 0] + sy * d[..., 1], -sy * d[..., 0] + cy * d[..., 1], d[..., 2]], dim=-1)
        q = d.abs() - h
        outside = th.linalg.norm(q.clamp(min=0.0), dim=-1)
        inside = q.max(dim=-1).values.clamp(max=0.0)
        return outside + inside


# ------------------------------------------------------------------------------------------------ hand checks


def _as_mats(T) -> th.Tensor:
    T = th.as_tensor(T, dtype=th.float32)
    return T.unsqueeze(0) if T.dim() == 2 else T


class HandChecker:
    """Exact hand-vs-support checks for batches of eef poses (world 4x4). Point sets are cached per (finger opening)."""

    def __init__(self, arm: str = "right", policy: ContactPolicy = DEFAULT, device: str = "cpu", spacing: Optional[float] = None):
        self.arm, self.policy, self.device = arm, policy, th.device(device)
        self.geom = H.load(arm)
        self.spacing = float(spacing if spacing is not None else policy.surface_spacing_m)
        self._pts: Dict[Tuple[float, float], Tuple[th.Tensor, th.Tensor]] = {}

    def points(self, finger_q: Tuple[float, float]) -> Tuple[th.Tensor, th.Tensor]:
        k = (round(float(finger_q[0]), 5), round(float(finger_q[1]), 5))
        if k not in self._pts:
            P, R = self.geom.surface(self.spacing, finger_q=k)
            self._pts[k] = (th.as_tensor(P, dtype=th.float32, device=self.device), th.as_tensor(R, dtype=th.long, device=self.device))
        return self._pts[k]

    def clearance(self, T_eef, support: Support, finger_q: Tuple[float, float] = (0.05, 0.05), chunk: Optional[int] = None,
                  points_per_chunk: int = 2_000_000) -> Dict[str, th.Tensor]:
        """Per-role minimum signed distance (N,) of the hand hulls to ``support`` for eef poses ``T_eef`` (N, 4, 4). Poses are processed in
        chunks of ``points_per_chunk`` transformed points (the box SDF keeps a few (chunk, points, 3) temporaries: ~25 MB each)."""
        T = _as_mats(T_eef).to(self.device)
        P, R = self.points(finger_q)
        if chunk is None:
            chunk = max(64, int(points_per_chunk // max(1, P.shape[0])))
        out = {r: th.full((T.shape[0],), float("inf"), device=self.device) for r in H.ROLES}
        for s in range(0, T.shape[0], chunk):
            Tc = T[s:s + chunk]
            W = th.einsum("nij,pj->npi", Tc[:, :3, :3], P) + Tc[:, None, :3, 3]
            d = support.sdf(W)
            for i, role in enumerate(H.ROLES):
                m = R == i
                if bool(m.any()):
                    out[role][s:s + chunk] = d[:, m].min(dim=1).values
        return out

    def check(self, T_eef, support: Support, finger_q: Tuple[float, float] = (0.05, 0.05), floor: bool = False) -> Tuple[th.Tensor, List[str], Dict[str, th.Tensor]]:
        """``(ok (N,), reason per pose, clearances)`` under rules 2-3 (floor: rule 5's finger allowance)."""
        cl = self.clearance(T_eef, support, finger_q)
        pol = self.policy
        allow = pol.floor_finger_max_penetration_m if floor else pol.finger_support_max_penetration_m
        n = next(iter(cl.values())).shape[0]
        ok = th.ones(n, dtype=th.bool, device=self.device)
        reasons = ["ok"] * n
        bad_f = cl["finger"] < -allow - 1e-9
        for role in ("palm", "wrist_camera", "wrist"):
            bad = cl[role] < pol.hand_support_margin_m - 1e-9
            for i in th.nonzero(bad & ok).flatten().tolist():
                reasons[i] = ROLE_REASON[role]
            ok &= ~bad
        for i in th.nonzero(bad_f & ok).flatten().tolist():
            reasons[i] = "finger_penetration"
        ok &= ~bad_f
        return ok, reasons, cl


def sphere_support_ok(spheres_world: th.Tensor, link_of_sphere: Sequence[str], support: Support, arm: str, policy: ContactPolicy = DEFAULT,
                      ground_contact_links: Sequence[str] = (), floor_support: bool = False) -> Tuple[th.Tensor, List[Optional[str]]]:
    """Rule 3 by spheres: every enabled sphere (r > 0) of every link but ``arm``'s fingers clear of ``support`` by
    ``sphere_support_margin_m`` (the radii carry the model's buffer). ``hand_sphere_check=False`` leaves the hand's rigid links to the
    exact hulls. ``floor_support``: the robot's ground-contact links (wheels, base) stand on that support - not checked against it.
    Returns ``(ok (N,), worst offending link per configuration or None)``."""
    S = th.as_tensor(spheres_world, dtype=th.float32)
    if S.dim() == 2:
        S = S.unsqueeze(0)
    fingers = set(policy.finger_links(arm))
    skip = set(fingers)
    if not policy.hand_sphere_check:
        skip |= set(policy.hand_rigid_links(arm))
    if floor_support:
        skip |= set(ground_contact_links)
    mask = th.tensor([(l not in skip) and bool(l) for l in link_of_sphere], dtype=th.bool, device=S.device)
    active = mask[None] & (S[..., 3] > 0)
    d = support.sdf(S[..., :3]) - S[..., 3]
    d = th.where(active, d, th.full_like(d, float("inf")))
    worst = d.min(dim=1)
    ok = worst.values >= policy.sphere_support_margin_m - 1e-9
    links = [None if bool(o) else link_of_sphere[int(i)] for o, i in zip(ok.tolist(), worst.indices.tolist())]
    return ok, links


# ------------------------------------------------------------------------------------------------ paths


def pre_pose_mat(T_eef: th.Tensor, dist: float = None) -> th.Tensor:
    """The demo pre-grasp (``conventions.pre_pose``): ``dist`` (default ``conventions.PRE_DIST`` = 0.10 m) back along the eef +z."""
    from ispatialgym.robot import conventions as C

    d = C.PRE_DIST if dist is None else float(dist)
    T = _as_mats(T_eef).clone()
    T[:, :3, 3] -= d * T[:, :3, 2]
    return T


def approach_path(T_pre: th.Tensor, T_grasp: th.Tensor, step: float) -> th.Tensor:
    """Straight-line eef poses from the pre-grasp to the grasp (same orientation, the demo's ``line_waypoints``): (K, N, 4, 4),
    pre-grasp first, grasp last, at most ``step`` apart."""
    Tp, Tg = _as_mats(T_pre), _as_mats(T_grasp)
    dist = float(th.linalg.norm(Tg[:, :3, 3] - Tp[:, :3, 3], dim=1).max()) if Tp.shape[0] else 0.0
    k = max(2, int(math.ceil(dist / max(step, 1e-4))) + 1)
    out = []
    for i in range(k):
        t = i / (k - 1)
        T = Tg.clone()
        T[:, :3, 3] = (1 - t) * Tp[:, :3, 3] + t * Tg[:, :3, 3]
        out.append(T)
    return th.stack(out)


def sweep_openings(open_q: float, contact_q: float, steps: int) -> List[Tuple[float, float]]:
    """Finger openings of the closing sweep from the pre-shape to the contact (both fingers symmetric)."""
    qs = np.linspace(open_q, max(contact_q, 0.0), max(2, steps))
    return [(float(q), float(q)) for q in qs]


def check_grasp_geometry(checker: HandChecker, T_grasp, support: Support, contact_q: Sequence[float], open_q: float = 0.05,
                         floor: bool = False) -> Dict[str, object]:
    """Rules 2, 3 and 6 for grasp poses (N, 4, 4): the pre-grasp and every approach waypoint at the pre-shape opening, then the closing
    sweep at the grasp pose down to each pose's contact opening. ``{"ok": (N,), "reason": [...], "stage": [...], "finger_min": (N,),
    "hand_min": (N,)}`` - the reason / stage of the first failing check, the deepest finger and non-finger hand clearances seen."""
    pol = checker.policy
    Tg = _as_mats(T_grasp)
    n = Tg.shape[0]
    path = approach_path(pre_pose_mat(Tg), Tg, pol.approach_step_m)
    ok = th.ones(n, dtype=th.bool)
    reason, stage = ["ok"] * n, [""] * n
    fmin = th.full((n,), float("inf"))
    hmin = th.full((n,), float("inf"))

    def fold(o, r, cl, st):
        nonlocal ok
        for i in th.nonzero(~o.cpu() & ok).flatten().tolist():
            reason[i], stage[i] = r[i], st
        ok &= o.cpu()
        fmin.copy_(th.minimum(fmin, cl["finger"].cpu()))
        hmin.copy_(th.minimum(hmin, th.minimum(th.minimum(cl["palm"], cl["wrist_camera"]), cl["wrist"]).cpu()))

    for k in range(path.shape[0]):
        o, r, cl = checker.check(path[k], support, (open_q, open_q), floor=floor)
        fold(o, r, cl, "pre_grasp" if k == 0 else ("grasp_open" if k == path.shape[0] - 1 else f"approach_{k}"))
    cq = th.as_tensor(contact_q, dtype=th.float32).reshape(-1)
    if cq.numel() == 1:
        cq = cq.repeat(n)
    for j in range(pol.sweep_steps):
        t = (j + 1) / pol.sweep_steps
        q = (1 - t) * open_q + t * cq
        groups: Dict[float, List[int]] = {}  # openings differ per pose: one point set per distinct opening
        for i, v in enumerate(q.tolist()):
            groups.setdefault(round(v, 4), []).append(i)
        o_all = th.ones(n, dtype=th.bool)
        r_all = ["ok"] * n
        cl_all = {r: th.full((n,), float("inf")) for r in H.ROLES}
        for v, idx in groups.items():
            o, r, cl = checker.check(Tg[idx], support, (v, v), floor=floor)
            for a, i in enumerate(idx):
                o_all[i] = o[a]
                r_all[i] = r[a]
                for role in H.ROLES:
                    cl_all[role][i] = cl[role][a]
        fold(o_all, r_all, cl_all, f"sweep_{j + 1}")
    return {"ok": ok, "reason": reason, "stage": stage, "finger_min": fmin, "hand_min": hmin}
