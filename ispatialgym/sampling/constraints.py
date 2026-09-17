"""Revolute hinge constraints for the World Alignment tasks (TRO ``constraints`` records).

A hinge is a PhysX ``RevoluteJoint`` between a parent rigid body (support / wall link, or the world when ``parent``
is None) and the object's root link, anchored at a world point with a world axis. The joint frame's X axis is the
rotation axis (PhysX default ``physics:axis = "X"``). Joints are USD prims, not part of ``og.sim.dump_state``; the
sampler creates them for its own checks and the loader recreates them after every reset.

**Hinge friction / damping** (owner decision 2026‑09‑16 22:05, spec §7): a wall hinge sits at the target's COM, so
gravity exerts no restoring torque and a frictionless joint keeps whatever angular velocity the last contact left
(vertical painting demo 0916‑194735: 0.25–0.6 rad/s for the whole 680-step settle window, ``not_settled``). The
record therefore carries ``friction`` (a Coulomb torque, N·m) and ``damping`` (viscous, N·m·s/rad) and the loader
applies them through a ``UsdPhysics.DriveAPI`` angular drive on the joint (stiffness 0, target velocity 0): PhysX's
implicit force-limited damped spring gives torque = clamp(−damping · ω, ±friction) — Coulomb friction above
ω = friction / damping, viscous creep below it. ``physxJoint:jointFriction`` is not used: it applies to articulation
joints only and the hinge is a plain joint (``excludeFromArticulation``). The sampler sets both from the target's
inertia about the hinge axis (:func:`hinge_resistance`); the axial table hinge records neither (unchanged).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import create_joint

from ispatialgym.sampling import geometry as G

#: wall hinge resistance from the target's inertia I_n about the hinge axis (:func:`hinge_resistance`):
#: Coulomb friction torque = I_n · HINGE_STOP_DECEL_RAD_S2. Pod probe 2026-09-16 23:24 on the painting_s070 instance (2.25 kg,
#: I_n 0.040 kg·m²; the 0916-194735 painting kept 0.25–0.6 rad/s for the whole settle window): at 5 rad/s² a 0.5 / 1.0 rad/s kick
#: settled in 0.5 / 0.67 s with 0.040 / 0.155 rad of drift, at 10 rad/s² in 0.43 / 0.53 s with 0.022 / 0.080 rad — the second
#: keeps even a 1 rad/s kick inside the 0.08 rad tolerance; the push arm works against ≈ friction / lever ≈ 1.6 N.
HINGE_STOP_DECEL_RAD_S2 = 10.0
#: viscous damping = I_n / HINGE_VISCOUS_TAU_S: the Coulomb regime reaches down to friction / damping = 0.2 rad/s, below it
#: the exponential creep has a 20 ms time constant (well under the PoseSettled 0.005 rad/s threshold within a step or two)
HINGE_VISCOUS_TAU_S = 0.02
#: the drive model the loader applies (recorded verbatim in the constraints record for the reader)
HINGE_DRIVE_MODEL = "UsdPhysics angular drive, stiffness 0, target velocity 0: torque = clamp(-damping * omega, +-friction)"


@dataclass
class Hinge:
    prim_path: str
    child_name: str
    parent_name: Optional[str]
    parent_link: Optional[str]
    anchor: th.Tensor  # world
    axis: th.Tensor  # world unit
    friction: Optional[float] = None  # Coulomb torque, N·m (None = frictionless)
    damping: Optional[float] = None  # viscous, N·m·s/rad (None = undamped)
    facts: Dict[str, Any] = field(default_factory=dict)  # inertia / mass the resistance was derived from

    def to_tro(self, child_inst: str, parent_inst: Optional[str]) -> dict:
        rec = {
            "type": "revolute",
            "child": child_inst,
            "parent": parent_inst,
            "parent_link": self.parent_link,
            "anchor_position": [float(v) for v in self.anchor],
            "axis": [float(v) for v in self.axis],
            "limits": None,
        }
        if self.friction is not None or self.damping is not None:
            rec.update({"friction": None if self.friction is None else float(self.friction),
                        "damping": None if self.damping is None else float(self.damping),
                        "drive": HINGE_DRIVE_MODEL})
            if self.facts:
                rec["resistance_facts"] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in self.facts.items()}
        return rec


def inertia_about_axis(mass: float, extents_perp: tuple) -> float:
    """Inertia of a uniform plate of ``mass`` with in-plane extents ``(a, b)`` about the axis through its centre normal to
    the plane: m (a² + b²) / 12 — the wall targets are frames, plates, boards and racks hanging with their thickness
    along the hinge axis (a hollow frame is up to 2× this: the friction torque is a scale, not a fit)."""
    a, b = (float(v) for v in extents_perp)
    return float(mass) * (a * a + b * b) / 12.0


def hinge_resistance(mass: float, extents_perp: tuple, decel: float = HINGE_STOP_DECEL_RAD_S2, tau: float = HINGE_VISCOUS_TAU_S) -> Dict[str, float]:
    """(friction N·m, damping N·m·s/rad, inertia kg·m², mass kg) of a wall hinge for a target of ``mass`` whose extents
    perpendicular to the hinge axis are ``extents_perp``: friction = I_n · decel, damping = I_n / tau (module constants)."""
    I_n = inertia_about_axis(mass, extents_perp)
    return {"friction": I_n * float(decel), "damping": I_n / float(tau), "inertia_kgm2": I_n, "mass_kg": float(mass),
            "extents_perp_m": [float(v) for v in extents_perp], "stop_decel_rad_s2": float(decel), "viscous_tau_s": float(tau)}


def _frame_with_x(axis: th.Tensor, ref_up: th.Tensor) -> th.Tensor:
    """Rotation matrix whose X column is ``axis``; Z chosen close to ``ref_up``."""
    x = axis / th.linalg.norm(axis)
    z = ref_up - th.dot(ref_up, x) * x
    if float(th.linalg.norm(z)) < 1e-6:
        z = th.tensor([1.0, 0.0, 0.0]) - th.dot(th.tensor([1.0, 0.0, 0.0]), x) * x
    z = z / th.linalg.norm(z)
    y = th.cross(z, x, dim=0)
    return th.stack([x, y, z], dim=1)


def apply_hinge_drive(prim_path: str, friction: Optional[float], damping: Optional[float]) -> Dict[str, Any]:
    """Put the friction / damping of a hinge record on the joint prim as a ``UsdPhysics.DriveAPI`` angular drive
    (stiffness 0, target velocity 0; USD's angular damping is per degree/s, so damping [N·m·s/rad] is written as
    damping · π / 180; ``maxForce`` = the Coulomb torque in N·m, ``inf`` when frictionless). Returns what was written."""
    from pxr import UsdPhysics

    if friction is None and damping is None:
        return {}
    stage = og.sim.stage
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"hinge prim {prim_path} not found for its drive")
    d_usd = 0.0 if damping is None else float(damping) * math.pi / 180.0
    f_usd = math.inf if friction is None else float(friction)
    with og.sim.editing_usd():
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(d_usd)
        drive.CreateTargetPositionAttr().Set(0.0)
        drive.CreateTargetVelocityAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(f_usd)
    og.sim.update_handles()
    return {"damping_usd_per_deg_s": d_usd, "max_force_nm": f_usd}


def create_hinge(child_obj, anchor_world, axis_world, parent_link=None, name: str = "isg_hinge", friction: Optional[float] = None,
                 damping: Optional[float] = None, facts: Optional[Dict[str, Any]] = None) -> Hinge:
    """Create a revolute joint holding ``child_obj``'s root at ``anchor_world`` and letting it rotate about ``axis_world``.

    ``parent_link``: a link (RigidPrim) of the parent object; None anchors to the world frame. ``friction`` (N·m) /
    ``damping`` (N·m·s/rad): the hinge resistance of the record (:func:`apply_hinge_drive`); both None = the frictionless
    hinge (the axial table hinge).
    """
    anchor = G.t(anchor_world)
    axis = G.t(axis_world)
    axis = axis / th.linalg.norm(axis)
    R_j = _frame_with_x(axis, th.tensor([0.0, 0.0, 1.0]))
    q_j = T.mat2quat(R_j)
    cp, cq = child_obj.root_link.get_position_orientation()
    # joint frame expressed in the child link frame
    p1, q1 = T.relative_pose_transform(anchor, q_j, cp.float(), cq.float())
    if parent_link is not None:
        pp, pq = parent_link.get_position_orientation()
        p0, q0 = T.relative_pose_transform(anchor, q_j, pp.float(), pq.float())
        body0 = parent_link.prim_path
    else:
        p0, q0 = anchor, q_j
        body0 = None
    prim_path = f"{child_obj.prim_path}/{name}"
    create_joint(
        prim_path=prim_path,
        joint_type="RevoluteJoint",
        body0=body0,
        body1=child_obj.root_link.prim_path,
        joint_frame_in_parent_frame_pos=p0,
        joint_frame_in_parent_frame_quat=q0,
        joint_frame_in_child_frame_pos=p1,
        joint_frame_in_child_frame_quat=q1,
        exclude_from_articulation=True,
    )
    apply_hinge_drive(prim_path, friction, damping)
    return Hinge(prim_path, child_obj.name, None if parent_link is None else parent_link.prim_path.split("/")[-2], None if parent_link is None else parent_link.body_name,
                 anchor, axis, friction=None if friction is None else float(friction), damping=None if damping is None else float(damping), facts=dict(facts or {}))


def remove_hinge(hinge: Optional[Hinge]):
    if hinge is None:
        return
    stage = og.sim.stage
    prim = stage.GetPrimAtPath(hinge.prim_path)
    if prim is not None and prim.IsValid():
        with og.sim.editing_usd():
            stage.RemovePrim(hinge.prim_path)
        og.sim.update_handles()
