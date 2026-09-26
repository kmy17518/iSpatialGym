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

**Brake** (2026-09-22, alignment-world_vertical 0922-180242): below ω = friction / damping the drive is a plain damper,
so any residual torque - the anchor a millimetre off the true centre of mass under gravity (0.009 N·m on the 2.25 kg
painting) - makes the hinge creep at torque / damping forever: with the former damping I_n / 0.02 s (1.99 N·m·s/rad)
the levelled painting drifted back 0.25 deg/s and left its 0.08 rad tolerance during the settle window. The damping is
therefore ``friction / HINGE_BRAKE_OMEGA_RAD_S`` (the joint-brake model of :func:`joint_brake_spec`): Coulomb friction
down to 0.0025 rad/s, the creep under a residual torque ~1e-5 rad/s. The pad still works only against ``friction``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

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
#: the former viscous damping I_n / HINGE_VISCOUS_TAU_S (the Coulomb regime down to friction / damping = 0.2 rad/s, a 20 ms
#: creep time constant below it) - kept as the lower bound of the damping; the brake below decides for every real target
HINGE_VISCOUS_TAU_S = 0.02
#: brake (2026-09-22, module docstring): damping = friction / HINGE_BRAKE_OMEGA_RAD_S, the Coulomb regime down to this angular
#: speed - the joint-brake speed of the small-articulation lids (JOINT_BRAKE_OMEGA_RAD_S), where the implicit drive is stable
HINGE_BRAKE_OMEGA_RAD_S = 0.0025
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


def hinge_resistance(mass: float, extents_perp: tuple, decel: float = HINGE_STOP_DECEL_RAD_S2, tau: float = HINGE_VISCOUS_TAU_S,
                     omega_brake: float = HINGE_BRAKE_OMEGA_RAD_S) -> Dict[str, float]:
    """(friction N·m, damping N·m·s/rad, inertia kg·m², mass kg) of a wall hinge for a target of ``mass`` whose extents
    perpendicular to the hinge axis are ``extents_perp``: friction = I_n · decel, damping = max(I_n / tau, friction /
    omega_brake) - the brake (module docstring): Coulomb friction down to ``omega_brake``, no creep under a residual torque."""
    I_n = inertia_about_axis(mass, extents_perp)
    friction = I_n * float(decel)
    damping = max(I_n / float(tau), friction / float(omega_brake)) if omega_brake and omega_brake > 0 else I_n / float(tau)
    return {"friction": friction, "damping": damping, "inertia_kgm2": I_n, "mass_kg": float(mass),
            "extents_perp_m": [float(v) for v in extents_perp], "stop_decel_rad_s2": float(decel), "viscous_tau_s": float(tau),
            "brake_omega_rad_s": float(omega_brake)}


def hinge_brake_damping(friction: Optional[float], damping: Optional[float], omega_brake: float = HINGE_BRAKE_OMEGA_RAD_S) -> Optional[float]:
    """The damping a hinge record's drive should have under the brake model: at least ``friction / omega_brake`` (a record
    written before 2026-09-22 carries I_n / tau, ~1 % of it - the loader raises it, ``InstanceLoader._create_hinge``).
    ``damping`` unchanged when the record has no friction (the frictionless axial hinge) or already brakes."""
    if friction is None or friction <= 0 or not omega_brake or omega_brake <= 0:
        return damping
    floor = float(friction) / float(omega_brake)
    return floor if damping is None or float(damping) < floor else float(damping)


def _frame_with_x(axis: th.Tensor, ref_up: th.Tensor) -> th.Tensor:
    """Rotation matrix whose X column is ``axis``; Z chosen close to ``ref_up``."""
    x = axis / th.linalg.norm(axis)
    z = ref_up - th.dot(ref_up, x) * x
    if float(th.linalg.norm(z)) < 1e-6:
        z = th.tensor([1.0, 0.0, 0.0]) - th.dot(th.tensor([1.0, 0.0, 0.0]), x) * x
    z = z / th.linalg.norm(z)
    y = th.cross(z, x, dim=0)
    return th.stack([x, y, z], dim=1)


_OPENABLE_OVERRIDE: Dict[str, List[str]] = {}


def restrict_openable_joints(obj, joint_names: Sequence[str]) -> None:
    """Make OmniGibson's ``Open`` state of ``obj`` look at ``joint_names`` only (2026-09-19, two-flap cartons): its threshold
    is ``lower + 5 % of the range`` per joint, so a flap whose lower limit was extended below its closed position (the
    pressed non-target flap, ``apply_joint_brake`` with ``lower_limit_rad``) counts as *open* while shut and the BDDL
    ``(not (open carton))`` init fails (0919-041120). Patches ``open_state._get_relevant_joints`` once (a per-object
    registry) and resets an already initialised state."""
    from omnigibson.object_states import open_state as OS

    if not getattr(OS, "_isg_openable_patched", False):
        orig = OS._get_relevant_joints

        def patched(o):
            names = _OPENABLE_OVERRIDE.get(getattr(o, "name", None))
            if names:
                return False, [o.joints[n] for n in names], [1] * len(names)
            return orig(o)

        OS._get_relevant_joints = patched
        OS._isg_openable_patched = True
    names = [str(n) for n in joint_names if n in obj.joints]
    if not names:
        return
    _OPENABLE_OVERRIDE[obj.name] = names
    try:
        from omnigibson.object_states import Open

        st = obj.states.get(Open) if hasattr(obj, "states") else None
        if st is not None and getattr(st, "relevant_joints_info", None) is not None:
            st.relevant_joints_info = (False, [obj.joints[n] for n in names], [1] * len(names))
    except Exception:  # noqa: BLE001  (the registry alone covers a state initialised later)
        pass


def apply_hinge_drive(prim_path: str, friction: Optional[float], damping: Optional[float], stiffness: Optional[float] = None,
                      lower_limit_deg: Optional[float] = None, upper_limit_deg: Optional[float] = None) -> Dict[str, Any]:
    """Put the friction / damping of a hinge record on the joint prim as a ``UsdPhysics.DriveAPI`` angular drive
    (stiffness 0, target velocity 0; USD's angular damping is per degree/s, so damping [N·m·s/rad] is written as
    damping · π / 180; ``maxForce`` = the Coulomb torque in N·m, ``inf`` when frictionless). ``stiffness`` [N·m/rad]
    (2026-09-19, carton flaps) adds a spring toward target position 0; ``lower_limit_deg`` / ``upper_limit_deg`` rewrite
    the joint's limits (a flap pressed below its closed position). Returns what was written."""
    from pxr import UsdPhysics

    if friction is None and damping is None and stiffness is None and lower_limit_deg is None and upper_limit_deg is None:
        return {}
    stage = og.sim.stage
    prim = stage.GetPrimAtPath(prim_path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"hinge prim {prim_path} not found for its drive")
    d_usd = 0.0 if damping is None else float(damping) * math.pi / 180.0
    k_usd = 0.0 if stiffness is None else float(stiffness) * math.pi / 180.0
    f_usd = math.inf if friction is None else float(friction)
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
    out = {"damping_usd_per_deg_s": d_usd, "max_force_nm": f_usd}
    if stiffness is not None:
        out["stiffness_usd_per_deg"] = k_usd
    if lower_limit_deg is not None or upper_limit_deg is not None:
        out["limits_deg"] = [lower_limit_deg, upper_limit_deg]
    return out


#: Coulomb regime of a joint brake starts at this angular speed (damping = max_effort / JOINT_BRAKE_OMEGA_RAD_S). Below it the
#: drive is viscous and the lid creeps at gravity torque / damping: the first blender demo (0918-233145, 0.5 N·m / damping 10)
#: held its 47.5 deg within tolerance but drifted 0.014 rad over the 132-step settle window (``settled_success`` false) -
#: 0.0025 rad/s makes the creep of a 0.03 N·m lid ~1e-4 rad/s (damping 200 at 0.5 N·m; the implicit drive stays stable)
JOINT_BRAKE_OMEGA_RAD_S = 0.0025


def joint_brake_spec(max_effort: float, stiffness: float = 0.0) -> Dict[str, Any]:
    """Brake record for :func:`apply_joint_brake`: the joint drive as a Coulomb brake of ``max_effort`` N·m (damping =
    max_effort / :data:`JOINT_BRAKE_OMEGA_RAD_S`, stiffness 0, target velocity 0 → torque = clamp(−damping·ω, ±max_effort))."""
    return {"stiffness": float(stiffness), "damping": float(max_effort) / JOINT_BRAKE_OMEGA_RAD_S, "max_effort": float(max_effort)}


def apply_joint_brake(joint, spec) -> Dict[str, Any]:
    """Resistance of an articulated object's **own** joint (small articulation, owner 2026-09-18 22:38: "add damping / friction
    to the hinges so that they hold at any angle"). ``spec``: a number = the PhysX ``jointFriction`` coefficient, or a dict
    with any of ``friction`` (that coefficient), ``stiffness`` / ``damping`` (N·m/rad, N·m·s/rad) and ``max_effort`` (N·m).

    Pod probe 2026-09-18 23:00 (``/tmp/probe_lid_brake.py``, blender cwkvib lid 74 g): the dataset joints carry an
    ``acceleration``-type drive with damping 0.05 per deg/s and no force cap; OmniGibson's ``JointPrim`` setters (articulation
    view gains / max effort / friction coefficient) and the ``physxJoint:jointFriction`` attribute change nothing - the lid
    falls closed from 50 % of its range in every case, even at friction 5.0 or damping 50 / max effort 2. A
    ``UsdPhysics.DriveAPI`` **force** drive written on the joint prim inside ``og.sim.editing_usd()`` (stiffness 0, target
    velocity 0, damping, ``maxForce`` = the Coulomb torque; :func:`apply_hinge_drive`, the wall hinges' mechanism) holds
    it with zero drift at 50 % and 90 %. So ``damping`` / ``max_effort`` go through :func:`apply_hinge_drive` (torque =
    clamp(−damping·ω, ±max_effort)); ``friction`` still uses the setter (kept for the record, ineffective on these joints).
    Returns what was applied."""
    out: Dict[str, Any] = {}
    if spec is None:
        return out
    if isinstance(spec, (int, float)):
        joint.friction = float(spec)
        return {"friction": float(spec)}
    if not isinstance(spec, dict):
        raise TypeError(f"joint brake spec must be a number or a dict, got {type(spec).__name__}")
    if spec.get("friction") is not None:
        joint.friction = float(spec["friction"])
        out["friction"] = float(spec["friction"])
    lo, hi = spec.get("lower_limit_rad"), spec.get("upper_limit_rad")
    if spec.get("damping") is not None or spec.get("max_effort") is not None or spec.get("stiffness") not in (None, 0, 0.0) or lo is not None or hi is not None:
        # stiffness > 0 (2026-09-19, carton flaps): a spring toward target position 0 - the non-target flap of a two-flap carton is
        # pressed down (below its closed position: lower_limit_rad extends the range) to open the seam and must come back
        # when the fist leaves; the light damping only steadies the return
        k_usd = 0.0 if spec.get("stiffness") is None else float(spec["stiffness"]) * math.pi / 180.0
        d_usd = 0.0 if spec.get("damping") is None else float(spec["damping"]) * math.pi / 180.0
        f_usd = math.inf if spec.get("max_effort") is None else float(spec["max_effort"])
        prim = joint.prim
        get = lambda name: prim.GetAttribute(name).Get() if prim.HasAttribute(name) else None  # noqa: E731
        cur_type, cur_d, cur_f, cur_k = get("drive:angular:physics:type"), get("drive:angular:physics:damping"), get("drive:angular:physics:maxForce"), get("drive:angular:physics:stiffness")
        cur_lo, cur_hi = get("physics:lowerLimit"), get("physics:upperLimit")
        lo_deg = None if lo is None else math.degrees(float(lo))
        hi_deg = None if hi is None else math.degrees(float(hi))
        same = (cur_type == "force" and cur_d is not None and abs(float(cur_d) - d_usd) < 1e-6 and cur_k is not None and abs(float(cur_k) - k_usd) < 1e-6
                and cur_f is not None and (math.isinf(f_usd) and float(cur_f) > 1e30 or abs(float(cur_f) - f_usd) < 1e-6)
                and (lo_deg is None or (cur_lo is not None and abs(float(cur_lo) - lo_deg) < 1e-3)) and (hi_deg is None or (cur_hi is not None and abs(float(cur_hi) - hi_deg) < 1e-3)))
        # idempotent: the USD edit stops / restarts the simulation, so a re-application after a reset only happens when needed
        if same:
            written = {"damping_usd_per_deg_s": d_usd, "stiffness_usd_per_deg": k_usd, "max_force_nm": f_usd, "already_applied": True}
        else:
            written = apply_hinge_drive(joint.prim_path, friction=spec.get("max_effort"), damping=spec.get("damping"), stiffness=spec.get("stiffness"),
                                        lower_limit_deg=lo_deg, upper_limit_deg=hi_deg)
        out.update({"stiffness": 0.0 if spec.get("stiffness") is None else float(spec["stiffness"]), "damping": None if spec.get("damping") is None else float(spec["damping"]),
                    "max_effort": None if spec.get("max_effort") is None else float(spec["max_effort"]), "lower_limit_rad": lo, "upper_limit_rad": hi, "usd": written})
    return out


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


def apply_mass(target, mass_kg: float, link_name: Optional[str] = None) -> Dict[str, Any]:
    """Give ``target``'s root link (or ``link_name``) the mass ``mass_kg`` with its inertia scaled along, both on the USD
    prim (``physics:mass``, which a later ``og.sim.stop()`` / ``play()`` rebuilds PhysX from) and on the running actor.
    Owner 2026-09-25 (disembedding): the dataset gives a tray its category mass (0.39 kg for the tray/uekqey base link) and
    four bottles of milk 1 kg each, a 10:1 inverted mass ratio in the table-tray-bottle stack that PhysX (TGS, 32 position
    iterations) resolves with contact chatter - the bottles hopped 1-7 mm seconds after their release in 8 of 8 attempts
    of the 3-distractor alias (0925-034257). Headless A/B on identical layouts: tray 0.39 kg 3 of 12 layouts moving,
    tray 2 kg 0 of 6 with every body within 0.01 mm per frame (bottles at 0.3 kg, the mass-ratio probe, 0 of 6 too).
    Recorded per instance as the TRO ``constraints`` record ``{"type": "mass", "object", "mass_kg"}`` the loader
    re-applies, so the sampler's stillness gate and the demo run the same physics. ``mass_kg <= 0`` leaves the mass
    alone. Returns ``{"link", "before_kg", "after_kg"}``."""
    link = target.root_link if link_name is None else target.links[link_name]
    before = float(link.mass)
    out = {"link": str(getattr(link, "body_name", link.prim_path.split("/")[-1])), "before_kg": before, "after_kg": before}
    if mass_kg is None or float(mass_kg) <= 0.0 or abs(before - float(mass_kg)) < 1e-6:
        return out
    mass_kg = float(mass_kg)
    from pxr import UsdPhysics

    with og.sim.editing_usd():
        prim = link.prim
        api = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI.Apply(prim)
        api.GetMassAttr().Set(mass_kg)
    view = getattr(link, "_rigid_prim_view", None)
    if view is not None and view.is_physics_handle_valid():
        # the live actor: PhysX applies the USD change lazily, and a mass write alone leaves the inertia of the old mass
        inertia = view.get_inertias()
        view.set_masses(th.tensor([mass_kg]))
        if before > 0.0:
            view.set_inertias(inertia * (mass_kg / before))
    out["after_kg"] = float(link.mass)
    return out


def apply_counterbalance(target, joint, leaf_link_names: Sequence[str], damping: Optional[float]) -> Dict[str, Any]:
    """A spring-balanced appliance door (owner decision 2026-09-19, ``articulation_open_large_scale``): gravity is switched
    off on the leaf assembly (the moving link and every link fixed to it) and the joint gets a viscous damping, so the
    leaf floats where the hand leaves it - as a dishwasher / oven door on its torsion springs does. Why not friction: the
    PhysX joint friction ``|F_resist| <= mu (|F| + |T|)`` leaks under a sustained load (dishwasher ynrhuk 0918-230708: the
    door crept 7 % of its angle after being posed at 22-27 deg, fell from 52 deg even at mu 0.35) and every N·m added
    to the brake is a N·m the 18-25 N·m arm joints must pull through. With no gravity moment nothing creeps, and the arm
    only works against the damping (acceleration-type drive: ``damping`` in 1/s). Recorded per instance as the TRO
    ``constraints`` record ``{"type": "counterbalance", "object", "joint", "links", "damping"}`` the loader re-applies.
    Returns what was applied (``links`` that exist and took the switch, ``damping``)."""
    applied: Dict[str, Any] = {"links": []}
    links = getattr(target, "links", {}) or {}
    for name in leaf_link_names:
        link = links.get(name)
        if link is None or not hasattr(link, "disable_gravity"):
            continue
        link.disable_gravity()
        applied["links"].append(str(name))
    if damping is not None:
        joint.damping = float(damping)
        applied["damping"] = float(damping)
    return applied
