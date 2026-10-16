"""Physical setup of an instance from its TRO ``constraints`` records.

* ``revolute``: a PhysX revolute joint between a parent link (or the world) and the object's root, anchored at a world
  point about a world axis. Joints are USD prims, not part of the simulator state, so the loader recreates them after
  every reset. An optional ``friction`` (Coulomb torque, N·m) and ``damping`` (N·m·s/rad) become an angular drive with
  stiffness 0 and target velocity 0: ``torque = clamp(-damping * omega, ±friction)``.
* ``joint_friction``: the same drive on an articulated object's own joint, holding a lid at any angle.
* ``mass``: the mass of an object's link, its inertia scaled along.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import omnigibson as og
import omnigibson.utils.transform_utils as T
import torch as th
from omnigibson.utils.usd_utils import create_joint

#: below this angular speed a braked hinge is viscous; damping = friction / omega keeps a residual torque from creeping
HINGE_BRAKE_OMEGA_RAD_S = 0.0025


@dataclass
class Hinge:
    prim_path: str
    friction: float | None = None
    damping: float | None = None


def hinge_brake_damping(friction: float | None, damping: float | None) -> float | None:
    """The damping a hinge drive gets: at least ``friction / HINGE_BRAKE_OMEGA_RAD_S`` when the hinge has friction."""
    if friction is None or friction <= 0:
        return damping
    floor = float(friction) / HINGE_BRAKE_OMEGA_RAD_S
    return floor if damping is None or float(damping) < floor else float(damping)


def apply_drive(
    prim_path: str,
    max_force: float | None,
    damping: float | None,
    stiffness: float | None = None,
    lower_limit_deg: float | None = None,
    upper_limit_deg: float | None = None,
) -> dict[str, Any]:
    """Write an angular force drive (target position and velocity 0) on a joint prim; optionally rewrite its limits.

    USD angular gains are per degree, so ``damping`` [N·m·s/rad] and ``stiffness`` [N·m/rad] are scaled by pi / 180;
    ``max_force`` is the Coulomb torque in N·m (unbounded when None).
    """
    from pxr import UsdPhysics

    if all(v is None for v in (max_force, damping, stiffness, lower_limit_deg, upper_limit_deg)):
        return {}
    prim = og.sim.stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"joint prim {prim_path} not found")
    d_usd = 0.0 if damping is None else float(damping) * math.pi / 180.0
    k_usd = 0.0 if stiffness is None else float(stiffness) * math.pi / 180.0
    f_usd = math.inf if max_force is None else float(max_force)
    with og.sim.editing_usd():
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(k_usd)
        drive.CreateDampingAttr().Set(d_usd)
        drive.CreateTargetPositionAttr().Set(0.0)
        drive.CreateTargetVelocityAttr().Set(0.0)
        drive.CreateMaxForceAttr().Set(f_usd)
        if lower_limit_deg is not None:
            prim.GetAttribute("physics:lowerLimit").Set(float(lower_limit_deg))
        if upper_limit_deg is not None:
            prim.GetAttribute("physics:upperLimit").Set(float(upper_limit_deg))
    og.sim.update_handles()
    return {"damping_usd_per_deg_s": d_usd, "stiffness_usd_per_deg": k_usd, "max_force_nm": f_usd}


def _frame_with_x(axis: th.Tensor, ref_up: th.Tensor) -> th.Tensor:
    """Rotation matrix whose X column is ``axis`` and whose Z column is close to ``ref_up``."""
    x = axis / th.linalg.norm(axis)
    z = ref_up - th.dot(ref_up, x) * x
    if float(th.linalg.norm(z)) < 1e-6:
        z = th.tensor([1.0, 0.0, 0.0]) - th.dot(th.tensor([1.0, 0.0, 0.0]), x) * x
    z = z / th.linalg.norm(z)
    return th.stack([x, th.cross(z, x, dim=0), z], dim=1)


def body_scale(prim_path: str) -> th.Tensor:
    """World scale of the rigid body at ``prim_path`` (the product of its and its ancestors' ``xformOp:scale``)."""
    scale = th.ones(3)
    prim = og.sim.stage.GetPrimAtPath(prim_path)
    while prim and prim.IsValid() and not prim.IsPseudoRoot():
        attr = prim.GetAttribute("xformOp:scale")
        value = attr.Get() if attr and attr.IsValid() else None
        if value is not None:
            scale = scale * th.tensor([float(v) for v in value], dtype=th.float32)
        prim = prim.GetParent()
    return scale


def create_hinge(
    child_obj, anchor_world, axis_world, parent_link=None, friction=None, damping=None, name: str = "isg_hinge"
) -> Hinge:
    """A revolute joint holding ``child_obj``'s root at ``anchor_world`` about ``axis_world`` (the world when
    ``parent_link`` is None)."""
    anchor = th.as_tensor(anchor_world, dtype=th.float32)
    axis = th.as_tensor(axis_world, dtype=th.float32)
    q_j = T.mat2quat(_frame_with_x(axis / th.linalg.norm(axis), th.tensor([0.0, 0.0, 1.0])))
    cp, cq = child_obj.root_link.get_position_orientation()
    p1, q1 = T.relative_pose_transform(anchor, q_j, cp.float(), cq.float())
    if parent_link is not None:
        pp, pq = parent_link.get_position_orientation()
        p0, q0 = T.relative_pose_transform(anchor, q_j, pp.float(), pq.float())
        body0 = parent_link.prim_path
    else:
        p0, q0, body0 = anchor, q_j, None
    # PhysX scales a joint's local positions by each body's USD scale
    p1 = p1 / body_scale(child_obj.root_link.prim_path)
    if body0 is not None:
        p0 = p0 / body_scale(body0)
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
    apply_drive(prim_path, friction, damping)
    return Hinge(prim_path, None if friction is None else float(friction), None if damping is None else float(damping))


def apply_joint_brake(joint, spec: dict[str, Any]) -> dict[str, Any]:
    """Resistance of an articulated object's own joint: ``friction`` (PhysX coefficient), and ``stiffness`` / ``damping``
    / ``max_effort`` / ``lower_limit_rad`` / ``upper_limit_rad`` as a force drive on the joint prim. Idempotent: the USD
    edit restarts the simulation, so an unchanged drive is not rewritten."""
    out: dict[str, Any] = {}
    if spec.get("friction") is not None:
        joint.friction = float(spec["friction"])
        out["friction"] = float(spec["friction"])
    lo, hi = spec.get("lower_limit_rad"), spec.get("upper_limit_rad")
    stiffness, damping, max_effort = spec.get("stiffness"), spec.get("damping"), spec.get("max_effort")
    if damping is None and max_effort is None and not stiffness and lo is None and hi is None:
        return out
    k_usd = 0.0 if stiffness is None else float(stiffness) * math.pi / 180.0
    d_usd = 0.0 if damping is None else float(damping) * math.pi / 180.0
    f_usd = math.inf if max_effort is None else float(max_effort)
    lo_deg = None if lo is None else math.degrees(float(lo))
    hi_deg = None if hi is None else math.degrees(float(hi))

    prim = joint.prim

    def current(name):
        return prim.GetAttribute(name).Get() if prim.HasAttribute(name) else None

    def same(value, want, tol=1e-6):
        return value is not None and abs(float(value) - want) < tol

    cur_f = current("drive:angular:physics:maxForce")
    unchanged = (
        current("drive:angular:physics:type") == "force"
        and same(current("drive:angular:physics:damping"), d_usd)
        and same(current("drive:angular:physics:stiffness"), k_usd)
        and cur_f is not None
        and (float(cur_f) > 1e30 if math.isinf(f_usd) else same(cur_f, f_usd))
        and (lo_deg is None or same(current("physics:lowerLimit"), lo_deg, 1e-3))
        and (hi_deg is None or same(current("physics:upperLimit"), hi_deg, 1e-3))
    )
    if not unchanged:
        apply_drive(joint.prim_path, max_effort, damping, stiffness, lo_deg, hi_deg)
    out.update(
        {"stiffness": float(stiffness or 0.0), "damping": damping, "max_effort": max_effort, "rewritten": not unchanged}
    )
    return out


def apply_mass(target, mass_kg: float, link_name: str | None = None) -> dict[str, Any]:
    """Set the mass of ``target``'s root link (or ``link_name``) on the USD prim and the running actor, its inertia
    scaled along. Returns ``{"link", "before_kg", "after_kg"}``."""
    from pxr import UsdPhysics

    link = target.root_link if link_name is None else target.links[link_name]
    before = float(link.mass)
    out = {
        "link": str(getattr(link, "body_name", link.prim_path.split("/")[-1])),
        "before_kg": before,
        "after_kg": before,
    }
    if mass_kg is None or float(mass_kg) <= 0.0 or abs(before - float(mass_kg)) < 1e-6:
        return out
    mass_kg = float(mass_kg)
    with og.sim.editing_usd():
        prim = link.prim
        api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(prim)
        api.GetMassAttr().Set(mass_kg)
    view = getattr(link, "_rigid_prim_view", None)
    if view is not None and view.is_physics_handle_valid():
        # PhysX applies the USD change lazily, and a mass write alone keeps the old inertia
        inertia = view.get_inertias()
        view.set_masses(th.tensor([mass_kg]))
        if before > 0.0:
            view.set_inertias(inertia * (mass_kg / before))
    out["after_kg"] = float(link.mass)
    return out
