"""Configuration matching — articulation, small-scale open / close (spec §5.4 Prompt 2, §6.7; tabletop Phases 1–2 with
the §5.4 substitutions, everything object-related from §5.4).

Roles: ``target`` (spawned articulated object; ``roles.target.joint`` / ``handle_key``; the v1 facade names it ``artic``
with the options on the role), scene ``support`` (a scene object, or — for objects taller than 0.6 m — the room floor
bound as ``{"room": R}`` by the iddl grounding: accepted at the binding, the grounded BDDL has one floor instance for
both roles; its init is **not** sampled yet: ``floor_support_todo`` rejects cleanly), ``floor``.

Template: the handle annotation (``artic/handles/<key>.json``, scaled to the instance), the facing direction f (moving
face outward normal; lids: from the hinge line to the centre) -> face yaw per edge, handle arm (the arm on the handle's
side of the centre, > 0.05 m; else right) and brace arm, the stable set S (7 fractions of ``open_fraction_range`` + the
upper limit held 120 steps; probed on a top cell), q_init / q_goal per direction, the waypoints q_k and the required
set (``require_goal_handle: auto``: up to the first k from which the link reaches q_goal passively), the **brace band**
(§5.4 reach-band ``--brace`` mode, computed in-process: per trunk and cell, 3 yaws about the face yaw, handle [pre-grasp,
grasp] at every required q screened with the handle arm, the brace candidates ``opposite_face`` / ``top_press``
screened with the brace arm; a cell is in the band when the required waypoints are feasible and >= 1 brace kind is),
Phases 1a / 1b (H_max = height at q_init) / 1c (row ``articulation_small``: w_a) / 2 (band = the brace band; the whole
target at q_init in frame with ``view_margin``, the handle contact and the brace point in frame).

Instance: target at the focus (±5 cm), yaw = face yaw + U(−0.1, 0.1), footprint on the top, joint at q_init; exact
32-seed IK in order: brace arm ([pre-contact, contact], contact with the target ignored; first feasible kind), handle
arm with the brace arm locked ([pre-grasp, grasp] at q_init, support + target ignored for the grasp; the grasp poses
at the required waypoints with the target re-posed), combined whole-body check; settle with the joint held; drift <=
½ tolerance; ``pose_targets`` measured (0.03 / 0.20) and ``articulation_targets``. Sidecar: ``init`` /
``manipulation`` (arms = [handle arm]), the transition, ``contacts.brace`` (kind, object = target), ``articulation``
(object, joint, joint_type, q_init, q_goal, ``waypoints`` h(q_k) at k ∈ {0, 0.25, …, 1}, ``handle_reach``,
``passive_from``, ``follow_mode: arm``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import constraints as CN
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.grasp_library import library_root
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.articulation import scale_handle, scale_joint_info
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import orient as O
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASKS = ("configuration_matching-articulation_open_small_scale", "configuration_matching-articulation_close_small_scale",
            "configuration_matching-articulation_small_scale-open", "configuration_matching-articulation_small_scale-close")


ATTEMPT_BUDGET_S = 300.0  # per instance attempt (brace search over the manipulation + second stances)


class _LinkSubset:
    """A few links of a spawned object as an ignorable 'object' for the collision checks (``ik._Ignoring`` walks ``.links``):
    the flap a fist presses / the flap a hand holds leaves the cuRobo world while the box body stays in it."""

    def __init__(self, obj, names):
        self.name = f"{getattr(obj, 'name', 'target')}:{'+'.join(sorted(n for n in names if n))}"
        self.links = {n: l for n, l in getattr(obj, "links", {}).items() if n in names}
        self.obj = obj


class _MeshSubset:
    """A few collision meshes of one link as an ignorable 'object' (``ik._Ignoring`` walks ``.links[*].collision_meshes``): the
    box wall a braced blade touches leaves the world for the other arm's checks while that arm is locked in contact (the blade's
    padded spheres inside the wall made every configuration read 'in collision', 0921-000434)."""

    class _L:
        def __init__(self, meshes):
            self.collision_meshes = meshes

    def __init__(self, obj, link_name, keep):
        link = getattr(obj, "links", {}).get(link_name)
        meshes = {n: cm for n, cm in getattr(link, "collision_meshes", {}).items() if keep(n, cm)} if link is not None else {}
        self.name = f"{getattr(obj, 'name', 'target')}:{link_name}:{'+'.join(sorted(meshes))}"
        self.links = {link_name: _MeshSubset._L(meshes)} if meshes else {}
        self.obj = obj


@register(*SUBTASKS)
class ArticulationSmall(SamplerV2):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.direction = "close" if "close" in self.subtask else "open"
        if self.gp.bddl is None:
            self.definition_id = 1 if self.direction == "open" else 2  # v1 writer: configuration_matching-1 / -2
        key = "configuration_matching-articulation_open_small_scale" if self.direction == "open" else "configuration_matching-articulation_close_small_scale"
        self.defaults = GD.defaults_for(key)
        self.role = "target" if "target" in self.gp.roles else "artic"
        self.top = None
        self.hull: Optional[th.Tensor] = None
        self.handle: Dict[str, Any] = {}
        self.jinfo: Dict[str, Any] = {}
        self.joint_name: Optional[str] = None
        self.handle_arm = "right"
        self.brace_arm = "left"
        self.facing_local = th.tensor([1.0, 0.0, 0.0])
        self.stable: List[float] = []
        self.q_init: Optional[float] = None
        self.q_goal: Optional[float] = None
        self.required: List[float] = []
        self.passive_from: Optional[float] = None
        self.edge_frame: Optional[EdgeFrame] = None
        self.init_stance: Optional[Stance] = None
        self.manip_candidates: List[Stance] = []
        self.bands = None
        self.left_bands = None  # the brace arm's contact band (owner 2026-09-16 item 2)
        self.handle_only_band = False
        self.brace_missing = False  # the last instance is handle-only (D-008 generation.brace_optional)
        self.face_yaw: Optional[float] = None
        #: §5.4 floor support: the customs bind ``support = {"room": R}`` (targets > 0.6 m stand on the room floor)
        self.floor_support = False
        self.floor_room: Optional[str] = None
        self.floor_obj = None

    def g(self, key: str):
        return self.gen(key, self.defaults.get(key))

    # ------------------------------------------------------------------ roles

    def v1_role_aliases(self) -> Dict[str, str]:
        return {"artic": self.role} if self.role != "artic" else {}

    def bind_scene_roles(self):
        floor = self.bind_floor_v2()
        req = self.choice.bindings.get("support")
        if isinstance(req, dict) and req.get("room"):
            # §5.4 floor support (targets taller than 0.6 m): the iddl grounding binds the support as ``{"room": R}`` and
            # the grounded BDDL has one floor instance for both roles (``(ontop target floor.n.01_1)``) — the target stands
            # on the room's floor patch; no separate support instance is bound
            self.floor_support = True
            self.floor_room = str(req["room"])
            floor_req = self.choice.bindings.get("floor")
            if isinstance(floor_req, dict) and floor_req.get("room") and floor_req["room"] != self.floor_room:
                raise Reject("spec", f"{self.gp.alias}: support room {self.floor_room} differs from the floor room {floor_req['room']}")
            self.floor_obj = self.objects[floor.name]
            return
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if not names:
            raise Reject("spec", f"{self.gp.alias}: scenes.{self.scene}.bindings.support must list scene object names or a room ({{'room': ...}}, floor support)")
        self.bind_scene_object("support", self.world.obj(names[0]))

    def _support(self):
        """The support object: the bound scene object, or the room floor for the §5.4 floor support."""
        return self.floor_obj if self.floor_support else self.obj("support")

    @property
    def tname(self) -> str:
        return self.bindings.one(self.role).name

    # ------------------------------------------------------------------ handle annotation (§5.3 / §5.4)

    def _role_options(self) -> Tuple[Optional[str], Optional[str]]:
        """``(joint, handle_key)`` from the customs role (D1) or the v1 spec role options (facade)."""
        r = self.gp.roles[self.role]
        mc = self.choice.models[self.role]
        joint = r.joint_of(f"{mc.category}/{mc.model}") if r.joint is not None else None
        hkey = r.handle_key.get(f"{mc.category}/{mc.model}") if isinstance(r.handle_key, dict) else r.handle_key
        if joint is None or hkey is None:
            v1 = TT.facade_v1_spec(self)
            if v1 is not None and "artic" in v1.roles:
                opts = v1.roles["artic"].options
                joint = joint or opts.get("joint")
                hkey = hkey or opts.get("handle_key")
        return joint, hkey or f"{mc.category}__{mc.model}"

    def _load_handle(self, obj):
        joint_name, key = self._role_options()
        path = os.path.join(library_root(), "artic", "handles", f"{key}.json")
        if not os.path.exists(path):
            raise Reject("missing_handle", f"no handle annotation {path}")
        with open(path) as f:
            data = json.load(f)
        joints = [j for j in data["joints"] if joint_name is None or j["joint"] == joint_name]
        if not joints:
            raise Reject("missing_handle", f"joint {joint_name!r} not in {path}")
        j = joints[0]
        handles = [h for h in j.get("handles", []) if h.get("kind") in ("bar", "knob")] or ([j["fallback"]] if j.get("fallback") else [])
        if not handles:
            raise Reject("missing_handle", f"no graspable handle for {key}/{j['joint']}")
        scale = th.as_tensor([float(s) for s in obj.scale], dtype=th.float32) if getattr(obj, "scale", None) is not None else th.ones(3)
        self.handle, self.jinfo = scale_handle(handles[0], scale), scale_joint_info(j, scale)
        self.all_jinfo = {jj["joint"]: scale_joint_info(jj, scale) for jj in data["joints"]}  # every annotated joint (the other flap of a two-flap carton)
        bl = data.get("base_link") or {}
        if bl.get("bbox_min") is not None and bl.get("bbox_max") is not None:
            self.base_link_bbox = {"bbox_min": (th.tensor(bl["bbox_min"], dtype=th.float32) * scale).tolist(), "bbox_max": (th.tensor(bl["bbox_max"], dtype=th.float32) * scale).tolist()}
        else:
            self.base_link_bbox = None
        self.joint_name = j["joint"]
        self.jinfo.setdefault("closed_q", self.jinfo.get("lower", 0.0))
        w = self.handle.get("width")
        lo, hi = self.g("pinch_width_range_m")
        if w is not None and not (lo <= float(w) <= hi):
            raise Reject("missing_handle", f"pinch width {1000 * float(w):.1f} mm outside {[1000 * lo, 1000 * hi]} mm after scaling")
        # the finger opening every handle-grasp IK is checked with: the oracle's pre-shape (pinch / 2 + 1 cm, execute_grasp) - the
        # fully open gripper's lower pad stands 5 cm below a flap and sweeps through its neighbour (cdmmwy 0919-022753)
        self.grasp_fingers = float(min(C.FINGER_OPEN, 0.5 * float(w) + 0.01)) if w is not None else None
        self.template_facts["handle"] = {"key": key, "joint": self.joint_name, "kind": self.handle.get("kind"), "contact_obj": self.handle.get("contact"),
                                         "pinch_axis_obj": self.handle.get("pinch_axis"), "approach_dir_obj": self.handle.get("approach_dir"), "width_m": w}

    def _facing_local(self) -> th.Tensor:
        """f: the moving face's outward normal; lids (vertical normal): the direction from the hinge line to the centre."""
        outward = th.tensor(self.jinfo.get("outward_normal", [1.0, 0.0, 0.0]), dtype=th.float32)
        if abs(float(outward[2])) > 0.9:
            axis = C.unit(th.tensor(self.jinfo.get("axis_dir_obj", [0.0, 1.0, 0.0]), dtype=th.float32))
            hinge = th.tensor(self.jinfo.get("hinge_point_obj", [0.0, 0.0, 0.0]), dtype=th.float32)
            perp = th.cross(axis, th.tensor([0.0, 0.0, 1.0]), dim=0)
            centre = self.hull.mean(dim=0)
            if float(th.dot(perp[:2], (centre - hinge)[:2])) < 0:
                perp = -perp
            if self.direction == "close":
                # (push-close too, 0920-062908: with the tip toward the robot the closing tangent at the tip points AT the robot -
                # the fist would have to pull; hinge toward the robot, the fist pushes the tip away and down)
                # close starts from the stable open state; a free hinge holds only past vertical (hinged jar: the upper
                # limit 2.618 rad alone held in the 22:26 / 23:14 runs), where the lid lies flipped over the hinge — with the
                # free edge toward the robot the open handle ends up behind the body (0/225 handle cells, no_workspace x3);
                # the hinge toward the robot puts the open lid's handle on the robot's side
                perp = -perp
            outward = perp
            lid_facing = str(self.gen("lid_facing", "free_edge"))
            if lid_facing in ("hinge_right", "hinge_left"):
                # 2026-09-18 (blender eyedvd first demo): the hinge line runs along the robot's line of sight, so the lid
                # flips sideways and the hand — rigidly attached to the lid — rolls about the robot's forward axis above
                # the hinge line instead of swinging over the body and ending behind it (free edge toward the robot:
                # 0/225 handle cells at every trunk for the 112 deg follow). ``hinge_right`` puts the hinge (and a hinge-side
                # pull ring) on the robot's right for the right arm; ``hinge_left`` mirrors it.
                want = -1.0 if lid_facing == "hinge_right" else 1.0  # sign of (robot's left) . (centre -> hinge)
                for cand in (axis, -axis):
                    f = th.tensor([float(cand[0]), float(cand[1]), 0.0])
                    if float(th.linalg.norm(f)) < 1e-6:
                        continue
                    f = C.unit(f)
                    robot_left = th.cross(th.tensor([0.0, 0.0, 1.0]), -f, dim=0)  # the robot looks along -f
                    if float(th.dot(robot_left[:2], (hinge - centre)[:2])) * want > 0:
                        outward = f
                        break
        outward[2] = 0.0
        return C.unit(outward)

    def _joint_type(self) -> str:
        return "prismatic" if "prismatic" in str(self.jinfo.get("type", "")).lower() else "revolute"

    def _joint_tol(self, dq: float) -> float:
        t = self.g("tolerances") or {}
        if self._joint_type() == "revolute":
            return max(float(t.get("revolute", 0.05)), float(t.get("joint_fraction", 0.10)) * abs(dq))
        return max(float(t.get("prismatic", 0.01)), float(t.get("joint_fraction", 0.10)) * abs(dq))

    # ------------------------------------------------------------------ template

    def sample_template(self):
        target = self.obj(self.role)
        self.hull = self.world.local_hull(target)
        self._load_handle(target)
        self.facing_local = self._facing_local()
        if self.handle.get("front_end_along_hinge"):
            # the pinch sits near the FRONT end of the flap's free edge and is approached from the front (2026-09-19, two-flap
            # cartons): the annotation gives |axial| and the approach's axial magnitude, the robot's side (facing_local)
            # decides the sign - the hinge line runs along the line of sight, the near end is on the robot's side
            axis = C.unit(th.tensor(self.jinfo["axis_dir_obj"], dtype=th.float32))
            side = 1.0 if float(th.dot(axis, self.facing_local)) >= 0 else -1.0  # +1: the robot is on the +axis end
            hinge = th.tensor(self.jinfo["hinge_point_obj"], dtype=th.float32)
            c = th.tensor(self.handle["contact"], dtype=th.float32)
            ax = float(th.dot(c - hinge, axis))
            c = c + axis * (side * abs(ax) - ax)
            a = C.unit(th.tensor(self.handle["approach_dir"], dtype=th.float32))
            aa = float(th.dot(a, axis))
            a = C.unit(a + axis * (-side * abs(aa) - aa))  # pointing away from the robot along the hinge
            self.handle = dict(self.handle, contact=[float(v) for v in c.tolist()], approach_dir=[float(v) for v in a.tolist()])
            log.info("%s: handle front end on the robot's side: contact %s, approach %s", self.gp.alias,
                     [round(v, 4) for v in self.handle["contact"]], [round(v, 4) for v in self.handle["approach_dir"]])
        self.v4 = self.gen("prep_push", None) == "two_flap_v4"  # owner 2026-09-20 15:21: the left fist levers the seam open, the right hand pinches the raised edge
        self.v2 = self.gen("prep_push", None) == "two_flap_v2" or self.v4
        self.q_ajar: Optional[float] = None
        self.handle_first: Optional[Dict[str, Any]] = None
        if self.v2:
            self._build_v2_handles()
        ext = self.hull.max(dim=0).values - self.hull.min(dim=0).values
        if self.floor_support or float(ext[2]) > float(self.g("floor_support_height_m")):
            # §5.4 floor support (owner decision 2026-09-16, item 2) accepted at the binding, not yet sampled: it needs the
            # floor band (``ReachBands.load(scene, "floor", ...)`` with the A6 posture retry), stances per the tabletop
            # phases with the floor as the support (no edge: the target's facing direction as the edge normal, stand-off
            # from the target's footprint) and the brace candidates on a > 0.6 m body — rejected cleanly for now
            why = f"support bound to the floor of {self.floor_room}" if self.floor_support else f"target height {float(ext[2]):.2f} m > {self.g('floor_support_height_m')}"
            raise Reject("floor_support_todo", f"{why}: the §5.4 floor-support init (floor band + virtual edge along the target's facing direction) is not implemented yet")
        # arms: the handle's side of the target centre (> 0.05 m) decides
        hc = th.tensor(self.handle["contact"], dtype=th.float32)
        # the robot's LEFT when f points at the robot: the robot looks along -f, left = z x (-f) = (f_y, -f_x, 0). (Until
        # 2026-09-19 this was (-f_y, f_x, 0) = the robot's right, so a handle on the right was given to the left arm; the blender
        # rings sat within the 5 cm threshold and were unaffected, the carton msfzpz edge 22 cm to the right got the left arm and
        # no IK solution in any band cell, 0919-011906.)
        side_axis = th.tensor([float(self.facing_local[1]), -float(self.facing_local[0]), 0.0])
        lat = float(th.dot(hc - self.hull.mean(dim=0), side_axis))
        arm_cfg = self.g("arm")
        self.handle_arm = arm_cfg if arm_cfg in ("left", "right") else ("left" if lat > float(self.g("handle_arm_side_threshold_m")) else "right")
        if self.v2:
            self.handle_arm = str(self.gen("v2_main_arm", "right"))  # the main opening (handle B, near end along the hinge); the first opening uses the other hand
        self.brace_arm = "right" if self.handle_arm == "left" else "left"
        support = self._support()
        self.top = self.support_top(support)
        if int(self.top.free.sum()) < 50:
            raise Reject("support_small", f"{support.name}: too little free top")
        self.joint = target.joints[self.joint_name]
        self.lower, self.upper = float(self.joint.lower_limit), float(self.joint.upper_limit)
        self.jinfo["lower"], self.jinfo["upper"] = self.lower, self.upper
        self.jinfo.setdefault("closed_q", self.lower)
        # stable set S and the joint states (§5.3 / §5.4)
        # generation.joint_friction (2026-09-18, blender eyedvd first demo; default None = the asset's 0.2): PhysX articulation
        # joint friction of the moved joint, a number or a list of candidates tried in order until the joint holds an open
        # value. The dataset lids fall to their gravitational equilibrium at the stock 0.2 (grasp library ARTIC_RESUME: hinged
        # jar -61 % of range in 3 s), so a lid that does not flip past vertical has no stable open state and the only reachable
        # goal is the upper limit (123 deg for the blender, a 112 deg roll of the hand). Recorded in the TRO as a
        # ``constraints`` record ``{"type": "joint_friction", ...}`` the loader (run_isg) re-applies, in template_facts and summary.
        # owner 2026-09-18 22:38 ("add damping / friction to the hinges so that they hold at any angle"): a candidate is a number
        # (PhysX jointFriction coefficient), a dict (constraints.apply_joint_brake: friction / stiffness / damping / max_effort) or
        # the shorthand "brake:<N·m>" = constraints.joint_brake_spec (the joint drive as a Coulomb brake of that torque)
        self.joint_friction: Optional[Any] = None
        jf = self.gen("joint_friction", None)
        cands = list(jf) if isinstance(jf, (list, tuple)) else ([jf] if jf is not None else [None])
        cands = [CN.joint_brake_spec(float(str(c).split(":", 1)[1])) if isinstance(c, str) and c.startswith("brake:") else c for c in cands]
        link = target.links.get(self.jinfo.get("child_link")) if isinstance(self.jinfo.get("child_link"), str) else None
        lid_mass = float(link.mass) if link is not None else None
        self.template_facts["moving_link_mass_kg"] = lid_mass
        self.stable = []
        applied: Dict[str, Any] = {}
        for cand in cands:
            if cand is not None:
                applied = CN.apply_joint_brake(self.joint, cand)
                self.joint_friction = cand
            self.stable = self._stable_angles(target)
            log.info("%s: stable set of %s (moving link %s kg) at joint resistance %s: %s", self.gp.alias, self.joint_name,
                     None if lid_mass is None else round(lid_mass, 3), "stock" if cand is None else applied, self.stable)
            if self.stable and (cand is None or any(q < self.upper - 1e-6 for q in self.stable)):
                break  # a resistance candidate must buy an open value below the upper limit, else the next one is tried
        self.template_facts["joint_friction"] = applied or None
        # generation.joint_drives (2026-09-19, two-flap cartons): drives of OTHER joints of the target, ``{joint: spec}`` with the
        # apply_joint_brake spec keys (stiffness / damping / max_effort / lower_limit_rad) - the non-target flap gets an extended
        # lower limit and a spring to 0 so that the prep push can press it down and it comes back when the fist leaves
        self.other_joint_drives: Dict[str, Dict[str, Any]] = {}
        for jn, spec in dict(self.gen("joint_drives", None) or {}).items():
            if jn == self.joint_name:
                continue
            if jn not in target.joints:
                raise Reject("spec", f"generation.joint_drives names {jn!r}, not a joint of {target.name} ({list(target.joints)})")
            self.other_joint_drives[jn] = CN.apply_joint_brake(target.joints[jn], spec)
            log.info("%s: drive of %s applied: %s", self.gp.alias, jn, self.other_joint_drives[jn])
        if any((spec or {}).get("lower_limit_rad") is not None for spec in dict(self.gen("joint_drives", None) or {}).values()):
            CN.restrict_openable_joints(target, [self.joint_name])  # OG's Open state must not read the extended-limit flap as open
        self.template_facts["joint_drives"] = self.other_joint_drives or None
        if not self.stable:
            raise Reject("no_stable_state", f"{target.name}/{self.joint_name} holds no open value passively")
        self.close_push = bool(self.direction == "close" and self.gen("close_push", False))
        if self.direction == "open":
            self.q_init, self.q_goal = self.lower, float(np.random.default_rng(self.ctx.base_seed).choice(self.stable))
        elif self.close_push:
            # owner 2026-09-19 22:59 (close, blender + carton): the joint starts open (close_init_rad, or close_init_fraction of
            # the range; the brake holds any value) and is PUSHED closed at the tip - to 0, fixed (the lower limit)
            q0 = self.gen("close_init_rad", None)
            self.q_init = float(q0) if q0 is not None else float(self.lower + float(self.gen("close_init_fraction", 0.6)) * (self.upper - self.lower))
            self.q_init = float(min(max(self.q_init, self.lower + 0.05), self.upper - 1e-3))
            self.q_goal = float(self.lower)
            self._build_close_push_handle()
        else:
            # close: the init is a stable open value (§6.6), the least open one (a free hinge usually holds only at the upper
            # limit: the hinged jar's stable set was {2.618} in the 22:26 / 23:14 runs; the facing rule turns the hinge toward
            # the robot for close so that the open lid's handle is on the robot's side, see _facing_local)
            self.q_init, self.q_goal = float(min(self.stable)), self.lower
        self.required, self.passive_from = self._required_waypoints(target)
        # H_max at q_init, half extent at q_init
        pts_init = self._hull_at_q(target, self.q_init)
        ext_i = pts_init.max(dim=0).values - pts_init.min(dim=0).values
        H_max = float(ext_i[2])
        w_a = 0.5 * float(max(ext_i[0], ext_i[1]))
        margins = {"kind": "articulation_small", "w_a": w_a, "safety": 0.02, "H_max": H_max}
        rng = np.random.default_rng(self.ctx.base_seed)
        errors = []
        for edge, st in TT.edge_candidates(self, self.top, rng):
            try:
                self.face_yaw = C.wrap_angle(math.atan2(float(-edge.inward[1]), float(-edge.inward[0])) - math.atan2(float(self.facing_local[1]), float(self.facing_local[0])))
                self.bands = self._brace_band(edge, st)
                # owner decision 2026-09-16 item 2: the brace arm's contact band (B3's left contact band of the target's height
                # class) replaces the empty §5.4 brace band; the brace is a pad contact on a face of the base link
                if self.brace_arm == "left":
                    self.left_bands = TT.load_left_contact_band(self, H_max, top=self.top, edge=edge, compute=TT.left_band_compute())
                    if self.left_bands is None:
                        log.info("%s: no left contact band for height %.2f: brace by the exact IK alone", self.gp.alias, H_max)
                init = TT.phase1b_init_trunk(self, st, H_max, bands=self.bands, top=self.top, left_bands=self.left_bands)
                init = TT.phase1c_lateral(self, init, margins, rng, top=self.top, bands=self.bands)
                init.name = "init"
                # §6.7 (owner 2026-09-16 17:10): the Phase-2 loose class is 25° / focus-in-frame for small articulation (the same
                # shape as the ontop-large C rule; the hinged jar's best focus lies 24° off the optical axis from the tall trunks
                # where the handle is reachable, 25° admits 42 foci over 3 stand-offs); the tight class stays at 8°
                # generation.focus_at_handle (2026-09-18, default off): the Phase-2 focus at the handle's height instead of the
                # target's mid height - the optical axis then passes the hand at the grasp (blender eyedvd 0918-225951: every
                # stance that reached the ring failed gripper_not_in_frame; the ring sits 0.35 m above the table top)
                focus_z = float(self.top.z) + 0.5 * H_max
                if self.gen("focus_at_handle", False):
                    focus_z = float(self.top.z) - float(self.hull.min(dim=0).values[2]) + float(self.handle["contact"][2])
                cands = TT.phase2_manipulation_stance(self, init, w_a, target_height=H_max, in_frame_fn=self._target_in_frame, bands=self.bands, top=self.top,
                                                      stand_offs=tuple(self.g("stand_offs_m")), focus_z=focus_z,
                                                      angle_classes_deg=(float(self.gen("phase2_tight_class_deg", 8.0)), float(self.gen("phase2_loose_class_deg", 25.0))))
                if not cands:
                    raise Reject("no_manipulation_stance", "Phase 2 left no survivor with the target, handle and brace point in frame")
                self.edge_frame, self.init_stance, self.manip_candidates = edge, init, cands
                break
            except Reject as e:
                errors.append(f"edge {edge.name}: {e}")
                log.info("%s: %s", self.gp.alias, errors[-1][:300])
        if self.init_stance is None:
            raise Reject("no_stance", "; ".join(errors)[:1500])
        self.template_facts.update({"support": self.top.to_json(), "edge": self.edge_frame.to_json(), "init_stance": self.init_stance.to_meta(),
                                    "n_manipulation_candidates": len(self.manip_candidates), "stable_set": self.stable, "q_init": self.q_init, "q_goal": self.q_goal,
                                    "required_waypoints": self.required, "passive_from": self.passive_from, "handle_arm": self.handle_arm, "brace_arm": self.brace_arm,
                                    "face_yaw": self.face_yaw, "H_max": H_max, "w_a": w_a})

    # -- joint states -------------------------------------------------------------------------------------------------

    def _set_joint(self, obj, q: float):
        qs = obj.get_joint_positions().clone()
        qs[list(obj.joints).index(self.joint_name)] = q
        obj.set_joint_positions(qs)
        obj.set_joint_velocities(th.zeros_like(qs))
        obj.keep_still()

    def _joint_value(self, obj) -> float:
        return float(obj.get_joint_positions()[list(obj.joints).index(self.joint_name)])

    def _set_named_joint(self, obj, name: str, q: float):
        """Set another joint of the target (the other flap of a two-flap carton) kinematically and sync the collision world."""
        qs = obj.get_joint_positions().clone()
        qs[list(obj.joints).index(name)] = q
        obj.set_joint_positions(qs)
        obj.set_joint_velocities(th.zeros_like(qs))
        obj.keep_still()
        self.kit.sync_world()

    def _on_top_for_probe(self, obj):
        cells = self.top.cells(free_only=True)
        centre = cells.mean(dim=0)
        pose = TT.object_pose_on(self, self.hull, C.IDENTITY_QUAT, centre, 0.0, self.top.z)
        obj.set_position_orientation(*pose)
        self.world.activate(obj)
        self.world.step_physics(30, still=[self.robot])

    def _stable_angles(self, obj) -> List[float]:
        """§5.3 stable set S: 7 fractions of ``open_fraction_range`` + the upper limit, each held 120 steps, kept within ½ of
        the probe tolerance; probed with the object standing on the top."""
        state = self.world.snapshot()
        lo, hi = self.g("open_fraction_range")
        fracs = [float(f) for f in np.linspace(lo, hi, int(self.g("stable_set_fractions"))).tolist()] + [1.0]
        tol = float((self.g("stable_probe_tolerances") or {}).get(self._joint_type(), 0.1))
        out = []
        try:
            self._on_top_for_probe(obj)
            for f in fracs:
                q = self.lower + f * (self.upper - self.lower)
                self._set_joint(obj, q)
                self.world.step_physics(int(self.g("stable_hold_steps")), still=[self.robot])
                q_after = self._joint_value(obj)
                if abs(q_after - q) <= float(self.g("stable_drift_tolerance_fraction")) * tol:
                    out.append(q)
        finally:
            self.world.restore(state)
            self.world.park(obj, list(self.world.spawned).index(obj.name))
        return sorted(set(round(v, 4) for v in out))

    def _required_waypoints(self, obj) -> Tuple[List[float], Optional[float]]:
        """§5.3 required handle poses: q_init plus every q_k up to and including the first k from which the link reaches
        q_goal passively (set q_k, 120 steps); all when none passes; ``require_goal_handle`` true / false force q_goal in / out."""
        fr = [float(k) for k in self.g("waypoint_fractions")]
        mode = self.g("require_goal_handle")
        tol = self._joint_tol(self.q_goal - self.q_init)
        passive = None
        if mode == "auto":
            state = self.world.snapshot()
            try:
                self._on_top_for_probe(obj)
                for k in fr:
                    self._set_joint(obj, self.q_init + k * (self.q_goal - self.q_init))
                    self.world.step_physics(int(self.g("stable_hold_steps")), still=[self.robot])
                    if abs(self._joint_value(obj) - self.q_goal) <= tol:
                        passive = k
                        break
            finally:
                self.world.restore(state)
                self.world.park(obj, list(self.world.spawned).index(obj.name))
        required = [k for k in fr if passive is None or k <= passive + 1e-9]
        if mode is True and 1.0 not in required:
            required.append(1.0)
        if mode is False:
            required = [k for k in required if k < 1.0]
        return required, passive

    def _hull_at_q(self, obj, q: float) -> th.Tensor:
        """Local hull with the joint at ``q`` (the object parked; the hull is re-measured in the root frame)."""
        q0 = self._joint_value(obj)
        self._set_joint(obj, q)
        try:
            return self.world.local_hull(obj)
        finally:
            self._set_joint(obj, q0)

    # -- brace band (§5.4 reach-band --brace mode, in-process) -----------------------------------------------------------

    def _stage2_enabled(self) -> bool:
        return bool(self.gen("stage2_push", False)) and self.direction == "open"

    def _band_fraction(self) -> float:
        """The handle-pose fraction the workspace band requires besides q_init: the last required one, or with stage 2 (the
        fist push takes over where the arm's reach ends) the first - the band must not demand a follow to q_goal then."""
        return float(min(self.required) if self._stage2_enabled() else max(self.required))

    def _q_main(self) -> float:
        """The joint value of the MAIN handle grasp: q_init, or q_ajar for the two-flap v2 sequence (the main hand pinches the
        flap once the first hand has opened it a little)."""
        return float(self.q_ajar) if (getattr(self, "v2", False) and self.q_ajar is not None) else float(self.q_init)

    def _band_q_values(self) -> List[float]:
        q0 = self._q_main()
        if getattr(self, "v2", False) or getattr(self, "close_push", False):
            # v2: the band screens the main pinch alone (a follow fraction in the band emptied the near cells, 0919-160406); the follow is
            # the exact IK's. push-close: the first tip contact alone - the arc is covered in segments from successive stances
            return [q0]
        return [q0] + ([q0 + self._band_fraction() * (self.q_goal - q0)] if self.required else [])

    def _build_v2_handles(self):
        """Owner 2026-09-19 14:34 (two-flap carton v2): two pinches of the target flap's free edge, both across its thickness
        (pinch axis z, ``edge_lid``), built from the joint geometry and the robot's side (``facing_local``):
        **A** (``handle_first``, the other hand, the first small opening): on the CENTRE LINE of the seam (mid-way along the
        hinge), approached across the seam toward the flap's hinge, pitched down ``v2_first_pitch_deg``;
        **B** (``handle``, the main hand, from q_ajar): near the END of the free edge on the robot's side
        (``v2_second_axial_inset_m`` in), approached along the hinge away from the robot, pitched down ``v2_second_pitch_deg``.
        The contact sits ``v2_edge_inset_m`` inside the free edge at the flap's mid-thickness."""
        ji = self.jinfo
        axis = C.unit(th.tensor(ji["axis_dir_obj"], dtype=th.float32))
        hinge = th.tensor(ji["hinge_point_obj"], dtype=th.float32)
        z = th.tensor([0.0, 0.0, 1.0])
        bmin, bmax = th.tensor(ji["link_bbox_min"], dtype=th.float32), th.tensor(ji["link_bbox_max"], dtype=th.float32)
        corners = th.stack([th.tensor([x, y, zz]) for x in (float(bmin[0]), float(bmax[0])) for y in (float(bmin[1]), float(bmax[1])) for zz in (float(bmin[2]), float(bmax[2]))])
        perp = C.unit(th.cross(axis, z, dim=0))
        if float(th.dot(perp[:2], (corners.mean(dim=0) - hinge)[:2])) < 0:
            perp = -perp  # from the hinge toward the free edge
        r_edge = float(((corners - hinge) @ perp).max())
        r_c = max(0.01, r_edge - float(self.gen("v2_edge_inset_m", 0.015)))
        ax = (corners - hinge) @ axis
        ax_min, ax_max = float(ax.min()), float(ax.max())
        side = 1.0 if float(th.dot(axis, self.facing_local)) >= 0 else -1.0  # +1: the robot stands at the +axis end
        ax_front = (ax_max if side > 0 else ax_min) - side * float(self.gen("v2_second_axial_inset_m", 0.03))
        z_mid = 0.5 * (float(bmin[2]) + float(bmax[2]))

        def contact_at(a_coord: float) -> List[float]:
            c = hinge + perp * r_c + axis * a_coord
            return [float(c[0]), float(c[1]), z_mid]

        pitches_a = [float(v) for v in self.gen("v2_first_pitches_deg", [25.0, 40.0, 10.0])]
        pb = math.radians(float(self.gen("v2_second_pitch_deg", 20.0)))
        app_b = C.unit(-side * axis * math.cos(pb) - z * math.sin(pb))
        base = {k: v for k, v in self.handle.items() if k not in ("contact", "approach_dir", "approach_dir_obj", "pinch_axis", "pull_dir", "contact_detected", "approach_dir_detected", "front_end_along_hinge")}
        base.update({"kind": "edge_lid"})
        self.handle_first_variants = []
        # v2_first_yaws_deg: the approach's horizontal direction, 0 = straight across the seam toward the hinge, 90 = along the
        # hinge away from the robot; a diagonal keeps the left forearm out of the extended torso (0919-163342: the across pinch
        # failed both finger assignments even 0.63 m ahead of the base)
        yaws_a = [float(v) for v in self.gen("v2_first_yaws_deg", [45.0, 0.0, 65.0])]
        for ydeg in yaws_a:
          for pdeg in pitches_a:
            pa, ya = math.radians(pdeg), math.radians(ydeg)
            horiz = C.unit(-perp * math.cos(ya) - side * axis * math.sin(ya))
            app_a = C.unit(horiz * math.cos(pa) - z * math.sin(pa))
            # v2_first_axial_frac: where along the free edge the first pinch sits, 0.5 = the centre line, 0 = the robot's end
            fa = float(self.gen("v2_first_axial_frac", 0.5))
            a_near, a_far = (ax_max, ax_min) if side > 0 else (ax_min, ax_max)
            a_first = a_near + (a_far - a_near) * fa
            self.handle_first_variants.append(dict(base, contact=contact_at(a_first), approach_dir=[float(v) for v in app_a.tolist()], pinch_axis=[0.0, 0.0, 1.0],
                                                   pitch_deg=pdeg, yaw_deg=ydeg, note="v2 first opening: on the seam, approached toward the hinge (yaw_deg from across to along)"))
        self.handle_first = self.handle_first_variants[0]
        self.handle = dict(base, contact=contact_at(ax_front), approach_dir=[float(v) for v in app_b.tolist()], pinch_axis=[0.0, 0.0, 1.0],
                           note="v2 main opening: the free edge near the robot, along the hinge away from the robot")
        # B variants (0920-165517: the right pinch along the hinge solved with a 2 mrad joint margin at every lever stance): the approach
        # turned from along the hinge (yaw 90) toward across the seam / the hinge (yaw 65, 45) and pitched steeper; ``_best_pinch``
        # prefers the lowest yaw / pitch listed first through its ordering, so the along-the-hinge pinch stays the first choice
        self.handle_variants = []
        a_near = ax_max if side > 0 else ax_min
        sites = [str(x) for x in (self.gen("v2_second_sites", ["free_edge"]) or ["free_edge"])]
        for site in sites:
            if site == "deep":
                # owner 2026-09-20 21:20: a real grasp of the raised flap, not a fingertip pinch - the pads straddle the sheet
                # v2_deep_inset_m (4 cm) in from the free edge (the lower pad in the gap the lift opened, the upper on the top face),
                # approached ACROSS the seam toward the flap's hinge from above the raised edge: the approach at q = 0 points from
                # the free edge toward the hinge (-perp) pitched down v2_deep_pitches_deg; ``handle_waypoints`` turns it with the flap,
                # so at q_ajar it runs along the raised sheet's surface and down. Near the box's near end (v2_deep_axial_inset_m),
                # clear of the lever's palm at mid-depth.
                d_deep = float(self.gen("v2_deep_inset_m", 0.04))
                a_deep = a_near - side * float(self.gen("v2_deep_axial_inset_m", 0.035))
                c_o = hinge + perp * max(0.02, r_edge - d_deep) + axis * a_deep
                c_o = [float(c_o[0]), float(c_o[1]), z_mid]
                for ydeg in [float(v) for v in self.gen("v2_deep_yaws_deg", [0.0, 20.0, -20.0])]:
                    for pdeg in [float(v) for v in self.gen("v2_deep_pitches_deg", [0.0, 10.0, 20.0])]:
                        pa2, ya2 = math.radians(pdeg), math.radians(ydeg)
                        horiz = C.unit(-perp * math.cos(ya2) - side * axis * math.sin(ya2))  # toward the hinge, turned toward (+) / away from (-) the robot
                        app = C.unit(horiz * math.cos(pa2) - z * math.sin(pa2))
                        self.handle_variants.append(dict(self.handle, contact=c_o, approach_dir=[float(v) for v in app.tolist()], pinch_axis=[0.0, 0.0, 1.0],
                                                         pitch_deg=pdeg, yaw_deg=ydeg, site=f"deep@{d_deep:g}",
                                                         note="v2 main opening: a deep grasp of the raised flap across the seam toward its hinge (owner 2026-09-20 21:20)"))
                continue
            if site == "near_edge":
                # the flap's NEAR side edge (along the box's near face, from the hinge corner to the seam corner), pinched at
                # v2_near_edge_width_fracs of the flap's width from the hinge: the same along-the-hinge approach, the contact up to
                # 8 cm closer to the hinge (the right arm's side) than the seam - the lever's stance (seam 10-20 cm left) is out of
                # the right hand's reach at the seam (0920-174724 / 180506)
                contacts = [(hinge + perp * (r_edge * float(wf)) + axis * (a_near - side * float(self.gen("v2_side_edge_inset_m", 0.015))), f"near_edge@{float(wf):g}")
                            for wf in (self.gen("v2_near_edge_width_fracs", [0.5, 0.35]) or [0.5])]
            else:
                contacts = [(hinge + perp * r_c + axis * ax_front, "free_edge")]
            for c_o, cname in contacts:
                c_o = [float(c_o[0]), float(c_o[1]), z_mid]
                for ydeg in [float(v) for v in self.gen("v2_second_yaws_deg", [90.0, 65.0, 45.0])]:
                    for pdeg in [float(v) for v in self.gen("v2_second_pitches_deg", [math.degrees(pb), 35.0, 50.0])]:
                        pa2, ya2 = math.radians(pdeg), math.radians(ydeg)
                        horiz = C.unit(-perp * math.cos(ya2) - side * axis * math.sin(ya2))
                        app = C.unit(horiz * math.cos(pa2) - z * math.sin(pa2))
                        self.handle_variants.append(dict(self.handle, contact=c_o, approach_dir=[float(v) for v in app.tolist()], pitch_deg=pdeg, yaw_deg=90.0 - ydeg, site=cname,
                                                         note=f"v2 main opening: {cname} near the robot (yaw 0 = along the hinge away from the robot, positive = turned toward the hinge)"))
        self.q_ajar = float(self.gen("v2_ajar_rad", 0.45))
        log.info("%s: v2 handles: A (first, centre line) contact %s approach %s; B (main, near end) contact %s approach %s; q_ajar %.3f", self.gp.alias,
                 [round(v, 4) for v in self.handle_first["contact"]], [round(v, 3) for v in self.handle_first["approach_dir"]],
                 [round(v, 4) for v in self.handle["contact"]], [round(v, 3) for v in self.handle["approach_dir"]], self.q_ajar)

    def _build_close_push_handle(self):
        """The push-close contact as a pseudo-handle in the CLOSED configuration (``handle_waypoints`` rotates it with the
        joint): the moving link's TIP (the point farthest from the hinge, on the centre line along the hinge, ``close_push_edge_inset_m``
        inside the free edge) on the face the closing motion pushes on (at q = 0 the tip moves down when closing, so the top face);
        approach −z (the closing tangent at q = 0), the closed pads' faces across the hinge (closing axis = the hinge axis).
        Blender: "push down at the tip"; carton: near the centre line of the flap, the tip in the closing direction (owner 22:59)."""
        ji = self.jinfo
        axis = C.unit(th.tensor(ji["axis_dir_obj"], dtype=th.float32))
        hinge = th.tensor(ji["hinge_point_obj"], dtype=th.float32)
        z = th.tensor([0.0, 0.0, 1.0])
        bmin, bmax = th.tensor(ji["link_bbox_min"], dtype=th.float32), th.tensor(ji["link_bbox_max"], dtype=th.float32)
        corners = th.stack([th.tensor([x, y, zz]) for x in (float(bmin[0]), float(bmax[0])) for y in (float(bmin[1]), float(bmax[1])) for zz in (float(bmin[2]), float(bmax[2]))])
        perp = C.unit(th.cross(axis, z, dim=0))
        if float(th.dot(perp[:2], (corners.mean(dim=0) - hinge)[:2])) < 0:
            perp = -perp
        r_edge = float(((corners - hinge) @ perp).max())
        r_c = max(0.02, r_edge - float(self.gen("close_push_edge_inset_m", 0.02)))
        ax = (corners - hinge) @ axis
        a_mid = 0.5 * (float(ax.min()) + float(ax.max()))
        top_z = float(bmax[2]) + float(self.gen("close_push_face_clear_m", 0.002))
        c = hinge + perp * r_c + axis * a_mid
        base = {k: v for k, v in self.handle.items() if k not in ("contact", "approach_dir", "approach_dir_obj", "pinch_axis", "pull_dir", "width", "length", "depth")}
        self.handle = dict(base, kind="tip_push", contact=[float(c[0]), float(c[1]), top_z], approach_dir=[0.0, 0.0, -1.0], pinch_axis=[float(v) for v in axis.tolist()],
                           width=None, note="push-close: the closed fist on the tip's top face, riding the closing arc (owner 2026-09-19 22:59)")
        self._close_push_handle0 = self.handle  # the fist's own turn: every attempt ranks its two turns from here
        self.grasp_fingers = C.FINGER_CLOSED
        log.info("%s: push-close handle: tip contact %s (r %.3f m from the hinge), q_init %.3f -> q_goal %.3f", self.gp.alias,
                 [round(v, 4) for v in self.handle["contact"]], r_c, self.q_init, self.q_goal)

    def _pinch_variants(self, handle: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Owner 2026-09-19 14:34: the gripper's fingers are symmetric, so a pinch may close along +pinch or −pinch (the hand
        rolled 180 deg about the approach); both are candidates and the joint-limit margin decides."""
        return [handle, dict(handle, pinch_axis=[-float(v) for v in handle["pinch_axis"]])]

    def _best_pinch(self, handle, q: float, stance: Stance, arm: str, target, support, pose: TT.Pose, q_seed=None, fingers: Optional[float] = None):
        """The pinch at joint value ``q``: per approach variant of ``handle`` (one dict or a list of dicts, e.g. several approach
        pitches) its two finger assignments (``_pinch_variants``: +pinch / -pinch, the hand rolled 180 deg about the approach) are
        chosen by the orientation rule (``orient.choose``, owner 2026-09-27: joint-limit safe, then easier to reach from where the
        arm starts - ``q_seed`` or the stance - with +pinch within the tolerance; each assignment also in the other's wrist-flipped
        branch), a pinch passing when its [pre-grasp, grasp] passes the exact IK from ``stance`` with at least
        ``v2_accept_margin_rad``; across the variants the owner's order (below). Returns ``(handle, waypoint, IKResult)`` or
        ``(None, None, None)``."""
        whys = []
        passing_all = []  # (h, wp, r, m): per approach variant its passing assignments, the orientation rule's choice first
        handles = list(handle) if isinstance(handle, (list, tuple)) else [handle]
        handle = handles[0]
        fng = self.grasp_fingers if fingers is None else float(fingers)
        accept_m = float(self.gen("v2_accept_margin_rad", 0.03))
        q_start = q_seed if q_seed is not None else self.ikc.q_locked(stance)
        chosen_all = []
        for hh in handles:
            props = []
            for rank, h in enumerate(self._pinch_variants(hh)):
                wp = C.handle_waypoints(h, self.jinfo, [q], pose)[0]
                props.append(O.Proposal(f"pinch {[round(v, 1) for v in h['pinch_axis']]}", rank, (h, wp), [wp.pre_grasp, wp.eef]))
            ok_here: Dict[int, Tuple[Any, ...]] = {}

            def evaluate(p):
                h, wp = p.payload
                r = self.ikc.exact([wp.pre_grasp, wp.eef], stance, arm, ignore_for_last=[support, target], q_seed=q_seed, cache=False, fingers=fng, q_first=p.q_first)
                if not r.ok:
                    whys.append(f"{p.label}: {str(r.reason)[:90]}")
                    return False, r, None, str(r.reason)
                m, _ = self.ikc.joint_limit_margin(r.q, arm)
                if m < accept_m:
                    whys.append(f"{p.label} yaw {h.get('yaw_deg', '-')}: margin {1000 * m:.0f} mrad < {1000 * accept_m:.0f}")
                    return False, r, None, f"margin {1000 * m:.0f} mrad < {1000 * accept_m:.0f}"
                ok_here[id(p)] = (h, wp, r, m)
                return True, r, None, ""

            chosen, _dec = O.choose(self.ikc, arm, props, evaluate, q_start, stage=f"handle pinch q {q:.2f}")
            if chosen is None:
                continue
            h_c, wp_c = chosen.proposal.payload
            m_c, _ = self.ikc.joint_limit_margin(chosen.result.q, arm)
            first = (h_c, wp_c, chosen.result, m_c)
            chosen_all.append(first)
            passing_all.append(first)
            passing_all.extend(v for k, v in ok_here.items() if v[2] is not chosen.result)
        self._pinch_passing = passing_all
        if not chosen_all:
            log.info("%s: %s pinch at q %.3f fails both finger assignments (%s; contact base %s)", self.gp.alias, arm, q, "; ".join(whys),
                     [round(float(v), 3) for v in stance.world_to_base(C.handle_waypoints(handle, self.jinfo, [q], pose)[0].contact[:2]).tolist()])
            return None, None, None
        best = max(chosen_all, key=lambda c: c[3])
        # owner 2026-09-19 16:55: the hand pointing straight across the seam (yaw 0) is the wanted pinch; a diagonal only when the
        # straight one has no solution with a sane margin - among the variants the lowest yaw with margin >= v2_min_margin
        floor_m = float(self.gen("v2_min_margin_rad", 0.05))
        ok = sorted([c for c in chosen_all if c[3] >= floor_m], key=lambda c: (abs(float(c[0].get("yaw_deg", 0.0))), float(c[0].get("pitch_deg", 0.0)), -c[3]))
        if ok:
            best = ok[0]
        log.info("%s: %s pinch at q %.3f: closing along %s, pitch %s / yaw %s deg (joint margin %.0f mrad)", self.gp.alias, arm, q, [round(v, 2) for v in best[0]["pinch_axis"]],
                 best[0].get("pitch_deg", "-"), best[0].get("yaw_deg", "-"), 1000 * best[3])
        return best[0], best[1], best[2]

    def _handle_pairs(self, obj_pose: TT.Pose, q: float) -> List[TT.Pose]:
        """[pre-grasp, grasp] of the handle at joint value ``q`` (conventions ``handle_waypoints``)."""
        wp = C.handle_waypoints(self.handle, self.jinfo, [q], obj_pose)[0]
        return [wp.pre_grasp, wp.eef]

    def _brace_band(self, edge: EdgeFrame, st: Stance) -> RB.ReachBands:
        """Workspace band of the handle arm. B3's brace band (handle + §5.4 brace) when it has cells; else, per trunk of the
        grid: the band-stance lattice cells (8 cm) on the top, 3 yaws about the face yaw, the handle [pre, grasp] at
        q_init and q_goal screened with the handle arm (support ignored for the grasp); a cell is in the band when both
        are feasible for a yaw (``meta.trunks[i].feasible`` like the CLI's brace mode, ``brace_kinds = ["left_band"]``:
        the brace comes from the left contact band at instance time, owner decision 2026-09-16 item 2)."""
        cache_key = {"scene": self.world.scene_model, "support": self.top.name, "target": f"{self.choice.models[self.role].category}/{self.choice.models[self.role].model}",
                     "joint": self.joint_name}
        kit, world = self.kit, self.world
        support = self._support()
        bc = RB.BandComputer(world, kit, arms=(self.handle_arm, self.brace_arm))
        stance = bc.stance_at_edge(self.top, support, RB.STAND_OFF, edge.name)
        x0, y0, yaw0 = stance["base_pose2d"]
        face_off = C.wrap_angle(self.face_yaw - yaw0)
        # the handle pose class the band was computed for (2026-09-18): ReachBands.find matches a handle band by scene / support /
        # target / joint only, so a re-pitched approach, the sideways lid facing or a scaled target would be served the previous
        # variant's band (the 20 deg band was 2 cells) - a cached "handle" band is accepted only with the same signature
        tgt = self.obj(self.role)
        handle_sig = {"contact": [round(float(v), 4) for v in self.handle["contact"]], "approach": [round(float(v), 4) for v in self.handle.get("approach_dir", [])],
                      "pinch": [round(float(v), 4) for v in self.handle["pinch_axis"]], "face_yaw_offset": round(float(face_off), 4), "arm": self.handle_arm,
                      "scale": [round(float(s), 3) for s in (tgt.scale.tolist() if getattr(tgt, "scale", None) is not None else [1.0, 1.0, 1.0])],
                      "q": [round(float(v), 4) for v in self._band_q_values()]}
        # B3's brace band (handle + §5.4 brace) when it has cells; the owner's 2026-09-16 item 2 replaces the empty §5.4 brace
        # rule by the left contact band, so the workspace band is the **handle band** (handle feasible at q_init and q_goal
        # for a yaw; mode "handle", computed in-process and cached) — the brace comes from the left contact band per instance
        for mode in ("brace", "handle"):
            try:
                b = RB.ReachBands.load(self.world.scene_model, self.top.name, target=cache_key["target"], joint=self.joint_name, mode=mode)
            except FileNotFoundError:
                continue
            if mode == "handle" and b.meta.get("handle_sig") != handle_sig:
                log.info("%s: handle band %s was computed for another handle pose class (%s); computing this one's", self.gp.alias, b.path, b.meta.get("handle_sig"))
                continue
            if any(len(c) for c in b.cells.values()):
                log.info("%s: %s band cache hit (%d cells): %s", self.gp.alias, mode, sum(len(c) for c in b.cells.values()), b.path)
                self.handle_only_band = mode == "handle"
                return b
            log.info("%s: B3 %s band %s has no cell (no §5.4 brace is IK-feasible): computing the handle band", self.gp.alias, mode, b.path)
        self.handle_only_band = True
        base_cells, world_cells = bc.lattice_on_cells(stance, self.top.cells(free_only=True))
        # the §5.3 required set exactly as the instance checks it (q_init + every q_k up to the passive fraction); B3's brace
        # bands record n_all_q = 0 for every target at all five fractions, so requiring q_goal itself would empty the band
        q_values = self._band_q_values()
        log.info("%s: handle band over q = %s (q_init + the %s required fraction of %s, passive from %s; one yaw = the face yaw)", self.gp.alias,
                 [round(v, 3) for v in q_values], "first (stage 2)" if self._stage2_enabled() else "last", self.required, self.passive_from)
        trunks = RB.trunk_grid(self.top.z - kit.floor_z)
        gen_trunks = self.gen("band_trunks", None)
        if gen_trunks:
            trunks = [tuple(float(v) for v in t) for t in gen_trunks]
        kit.sync_world()
        per_trunk, cells_out = [], {}
        scope = RB.ObstacleScope(kit, world, (x0, y0)).__enter__()
        try:
            for tr, ti in trunks:
                q = kit.q_at_base_pose(kit.q_config(arms="untucked", translate=tr, tilt=ti), x0, y0, yaw0)
                entry: Dict[str, Any] = {"translate": tr, "tilt": ti, "collides": False, "n_lattice": int(len(base_cells)), "n_reachable": 0, "feasible": []}
                if not bool(kit.collision_free(q, margin=0.01)[0]):
                    entry["collides"] = True
                    per_trunk.append(entry)
                    cells_out[(tr, ti)] = np.zeros((0, 2))
                    continue
                shoulder, arm_len = bc.shoulder_and_reach(q, self.handle_arm)
                pts = th.cat([world_cells, th.full((len(world_cells), 1), self.top.z)], dim=1)
                near = th.linalg.norm(pts - shoulder, dim=1) <= arm_len + float(th.linalg.norm(th.tensor(self.handle["contact"]))) + RB.PRE_GRASP + 0.05
                combos = [(i, dy) for i in range(len(base_cells)) if bool(near[i]) for dy in (face_off,)]  # the face yaw only (the instance jitters ±yaw_jitter)
                obj_poses = {c: TT.object_pose_on(self, self.hull, C.IDENTITY_QUAT, world_cells[c[0]], yaw0 + c[1], self.top.z) for c in combos}
                pairs = {c: [self._handle_pairs(obj_poses[c], qv) for qv in q_values] for c in combos}
                flat = [(c, k) for c in combos for k in range(len(q_values))]
                ok_pre = kit.ik_batch([pairs[c][k][0] for c, k in flat], q, arm=self.handle_arm, batch_size=bc.IK_BATCH) if flat else th.zeros(0, dtype=th.bool)
                pre_ok = [(c, k) for n, (c, k) in enumerate(flat) if bool(ok_pre[n])]
                with scope.ignoring([support]):
                    ok_g = kit.ik_batch([pairs[c][k][1] for c, k in pre_ok], q, arm=self.handle_arm, batch_size=bc.IK_BATCH) if pre_ok else th.zeros(0, dtype=th.bool)
                handle_ok: Dict[Tuple[int, float], List[int]] = {}
                for n, (c, k) in enumerate(pre_ok):
                    if bool(ok_g[n]):
                        handle_ok.setdefault(c, []).append(k)
                # per-q diagnostics (2026-09-18: a 0/225 band says nothing about WHICH pose kills it - the grasp at q_init or
                # the last required waypoint): pre-grasp / grasp IK passes per joint value
                n_pre = [sum(1 for c, k in pre_ok if k == kq) for kq in range(len(q_values))]
                n_g = [sum(1 for n, (c, k) in enumerate(pre_ok) if k == kq and bool(ok_g[n])) for kq in range(len(q_values))]
                log.info("handle trunk (%.2f, %+.1f): %d near cells; pre-grasp ok per q %s = %s, grasp ok per q = %s", tr, ti, len(combos),
                         [round(v, 3) for v in q_values], n_pre, n_g)
                feasible = []
                for i in range(len(base_cells)):
                    for dy in (face_off,):
                        c = (i, dy)
                        if c in handle_ok and len(handle_ok[c]) == len(q_values):
                            feasible.append({"cell": base_cells[i].tolist(), "yaw_offset": dy, "q_feasible": q_values, "brace_kinds": ["left_band"]})
                            break
                entry.update({"n_reachable": len(feasible), "feasible": feasible})
                per_trunk.append(entry)
                cells_out[(tr, ti)] = np.array([f["cell"] for f in feasible], dtype=np.float32).reshape(-1, 2)
                log.info("handle trunk (%.2f, %+.1f): %d/%d cells feasible", tr, ti, len(feasible), len(base_cells))
        finally:
            scope.__exit__(None, None, None)
            kit.sync_world()
        lat_all = np.concatenate([c[:, 1] for c in cells_out.values() if len(c)]) if any(len(c) for c in cells_out.values()) else np.zeros(0)
        meta = {"scene": world.scene_model, "rooms": list(world.rooms or []), "support": self.top.name, "lattice": bc.lattice, "stand_off": RB.STAND_OFF, "arms": "untucked",
                "grid": {"translates": list(RB.TRANSLATES), "tilts": list(RB.TILTS), "low_top_tilt": RB.LOW_TOP_TILT}, "seeds": 12, "pre_grasp": RB.PRE_GRASP, "yaws": list(RB.YAWS),
                "mode": "handle", "smoke": False, "scale": None, "placed_specs": [], "stance": stance, "top": self.top.to_json(), "z_support": self.top.z,
                "target": cache_key["target"], "joint": self.joint_name, "joint_values": q_values, "face_yaw_offset": face_off, "handle_arm": self.handle_arm,
                "brace_arm": self.brace_arm, "trunks": per_trunk, "union_extent": [float(lat_all.min()), float(lat_all.max())] if len(lat_all) else None,
                "n_reachable_total": int(sum(len(c) for c in cells_out.values())), "computed_by": "sampling.v2.tasks.articulation_small", "handle_sig": handle_sig}
        key = RB._key({**cache_key, "q": q_values, "handle": True, "grid": meta["grid"], "lattice": bc.lattice, "stance": stance["edge"], "handle_sig": handle_sig})
        meta["key"] = key
        out_dir = RB.cache_root()
        os.makedirs(out_dir, exist_ok=True)
        name = f"handle__{world.scene_model}__{self.top.name}__{cache_key['target'].replace('/', '-')}__{key}"
        np.savez_compressed(os.path.join(out_dir, name + ".npz"), **{RB.trunk_key(tr, ti): c for (tr, ti), c in cells_out.items()})
        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        cells = {k: v for k, v in cells_out.items() if not TT._trunk_collides(per_trunk, k)}
        return RB.ReachBands(meta, cells, {}, os.path.join(out_dir, name + ".npz"))

    def _target_in_frame(self, s, stance: Stance, cam, focus, info) -> bool:
        """§5.4 Phase 2 visibility: the whole target at q_init (exact hull, ``view_margin``), the handle contact and the brace
        point in frame."""
        pose = TT.object_pose_on(self, self.hull, C.IDENTITY_QUAT, focus[:2], self.face_yaw, self.top.z)
        pts = C.transform_points(pose, self._hull_q_init)
        if not TT.in_frame(self, cam, pts, float(self.g("view_margin_m"))):
            return False
        wp = C.handle_waypoints(self.handle, self.jinfo, [self.q_init], pose)[0]
        pts_b = [wp.contact]
        if self.handle_only_band:
            # the brace is a pad contact on the near face or the brace arm's own-side face of the base link: the handle
            # contact and at least one of those face points in frame (the far / across faces are never braced)
            own = "left" if self.brace_arm == "left" else "right"
            face_pts = [pt for cls, pt, _, _ in self._body_brace_points(pose, stance) if cls.startswith(("near", own))]
            if not TT.in_frame(self, cam, th.stack(pts_b), TT.IN_FRAME_MARGIN):
                return False
            return any(TT.in_frame(self, cam, pt.unsqueeze(0), TT.IN_FRAME_MARGIN) for pt in face_pts)
        else:
            feas = self.bands.brace_feasible(stance.translate, stance.tilt)
            kinds = next((f["brace_kinds"] for f in feas if abs(f["cell"][0] - float(stance.world_to_base(focus[:2])[0])) < 0.05
                          and abs(f["cell"][1] - float(stance.world_to_base(focus[:2])[1])) < 0.05), [])
            for kind, pair in RB.brace_candidates(self.jinfo, {"handle": self.handle, "base_link": self.base_link_bbox or _base_link_bbox(self.hull, self.jinfo)}, pose,
                                                  [self.q_init] + [self.q_init + k * (self.q_goal - self.q_init) for k in self.required], 1.0 if self.brace_arm == "left" else -1.0, _G()):
                if not kinds or kind in kinds:
                    pts_b.append(pair[1][0])
        return TT.in_frame(self, cam, th.stack(pts_b), TT.IN_FRAME_MARGIN)

    def _body_brace_points(self, pose: TT.Pose, stance: Stance) -> List[Tuple[str, th.Tensor, th.Tensor, th.Tensor]]:
        """Pad contacts of the brace arm on the base link (B3's contact classes): ``(class, point, approach, closing)`` for
        the faces toward the robot (``near_*``), on the robot's left / right (``left_*`` / ``right_*``) and away from it
        (``far_*``) at 0.75 of the base link's height, closing axis vertical (``_v``) / horizontal (``_h``); top presses
        are left out (the moving link sweeps the base link's top, §5.4). Order: the classes B3 measured as passing for
        table-top bodies first (near, left)."""
        bl = self.base_link_bbox or _base_link_bbox(self.hull, self.jinfo)
        bmin, bmax = th.tensor(bl["bbox_min"], dtype=th.float32), th.tensor(bl["bbox_max"], dtype=th.float32)
        corners = th.stack([th.tensor([x, y, z]) for x in (float(bmin[0]), float(bmax[0])) for y in (float(bmin[1]), float(bmax[1])) for z in (float(bmin[2]), float(bmax[2]))])
        cw = C.transform_points(pose, corners)
        centre = cw.mean(dim=0)
        z_side = float(cw[:, 2].min()) + TT.CONTACT_HEIGHT_FRACTION * float(cw[:, 2].max() - cw[:, 2].min())
        fwd = th.tensor([float(stance.forward[0]), float(stance.forward[1]), 0.0])
        left = th.tensor([float(stance.left[0]), float(stance.left[1]), 0.0])
        z_axis = th.tensor([0.0, 0.0, 1.0])
        out = []
        for side, u in (("near", -fwd), ("left", left), ("right", -left), ("far", fwd)):  # u = outward normal of the face touched
            proj = cw[:, :2] @ u[:2]
            point = th.tensor([float(centre[0]), float(centre[1]), z_side]) + u * (float(proj.max()) - float(centre[:2] @ u[:2]))
            a = -u
            side_h = C.unit(th.cross(z_axis, u, dim=0))
            for c_axis, suffix in ((z_axis, "v"), (side_h, "h")):
                out.append((f"{side}_{suffix}", point, a, c_axis))
            # pitched press (jar_diag 20:54: every horizontal pre-contact 17 cm above the top fails the IK from the low
            # trunks): the pads on the face's upper part, the approach 45 deg from above (B3's pitched classes), closing horizontal
            point_hi = th.tensor([float(centre[0]), float(centre[1]), float(cw[:, 2].min()) + 0.85 * float(cw[:, 2].max() - cw[:, 2].min())]) \
                + u * (float(proj.max()) - float(centre[:2] @ u[:2]))
            out.append((f"{side}_p45", point_hi, C.unit(-u - z_axis), side_h))
        return out

    # ------------------------------------------------------------------ brace + handle from a stance, second stance

    def _brace_candidates_at(self, stance: Stance, pose: TT.Pose, q_values) -> Tuple[List[Tuple[str, Tuple[TT.Pose, TT.Pose]]], Dict[str, Any]]:
        """Brace pose candidates of the brace arm at ``stance``: the base link's four side faces (near / own side / far /
        other side, vertical and horizontal closing, either sign) ranked by B3's passing classes at the target's cell,
        then in-band, then near / own side first (owner 20:21: the far and across faces are candidates too, the IK
        decides); without the handle-only band, B3's §5.4 brace kinds."""
        if self.handle_only_band:
            body = self._body_brace_points(pose, stance)
            passing = None
            if self.left_bands is not None:
                cell = self.left_bands.to_base(np.asarray([[float(pose[0][0]), float(pose[0][1])]], dtype=np.float32), stance.base_pose2d)[0]
                passing = TT.cell_classes(self.left_bands, stance.translate, stance.tilt, cell) or None
            own_side = "left" if self.brace_arm == "left" else "right"

            def rank(item):
                cls, pt, _, _ = item
                in_band = TT.in_band(self.left_bands, stance, pt[:2], tol=1.0)[0] if self.left_bands is not None else True
                return (0 if (passing and cls in passing) else 1, 0 if in_band else 1, 0 if cls.startswith(("near", own_side)) else 1)

            body.sort(key=rank)
            cands = []
            for cls, pt, a, c_axis in body:
                for c_sign in (1.0, -1.0):
                    contact = C.contact_eef(pt, a, c_sign * c_axis)
                    cands.append((f"{cls}{'' if c_sign > 0 else '-'}", (C.pre_pose(contact), contact)))
            return cands, {"pose_classes_passing": passing, "left_band": self.left_bands is not None}
        hdata = {"handle": self.handle, "base_link": self.base_link_bbox or _base_link_bbox(self.hull, self.jinfo)}
        cands = RB.brace_candidates(self.jinfo, hdata, pose, q_values, 1.0 if self.brace_arm == "left" else -1.0, _G())
        kinds_pref = list(self.g("brace_kinds"))
        cands.sort(key=lambda c: kinds_pref.index(c[0]) if c[0] in kinds_pref else 99)
        return cands, {}

    def _brace_from(self, stance: Stance, pose: TT.Pose, q_values, grasp_pair, target, support):
        """Handle grasp (pre + grasp, the handle arm) FIRST, then the brace arm's pre-contact + contact with the handle arm
        locked at its solution, combined check (``tabletop.brace_second_arm``; the 0916-172359 run locked the brace first
        and the handle arm's pre-grasp failed for every brace: the handle is the tighter constraint, the brace gets the
        freedom). Returns ``(chosen | None, cands, fails, extras)``."""
        cands, extras = self._brace_candidates_at(stance, pose, q_values)
        fails: List[str] = []
        state: Dict[str, Any] = {"handle_ok": None}

        class _HandleFails(Exception):
            pass

        def evaluate(p):
            kind, (b_pre, b_contact) = p.payload
            self._check_deadline()
            res = TT.brace_second_arm(self, stance, self.handle_arm, grasp_pair, self.brace_arm, [b_pre, b_contact], ignore_work_last=[support, target],
                                      ignore_brace_last=[target], q_first_brace=p.q_first)
            if res.ok:
                return True, res, None, ""
            if res.reason.startswith(f"{self.handle_arm}:"):
                raise _HandleFails(f"{kind}: {res.reason}")  # the handle grasp alone fails at this stance: no brace can help
            state["handle_ok"] = True
            # the handle arm's first solution may be the one branch the brace cannot live with (0916-205838: at the one stance
            # where both pass alone, every two-arm pair failed): the other order gives the brace arm its solution first
            res2 = TT.brace_second_arm(self, stance, self.brace_arm, [b_pre, b_contact], self.handle_arm, grasp_pair, ignore_work_last=[target],
                                       ignore_brace_last=[support, target])
            if res2.ok:
                return True, res2, None, ""
            return False, None, None, f"{res.reason} / reversed: {res2.reason}"

        # the orientation rule (owner 2026-09-27) orders each pad contact's two closing signs (``kind`` / ``kind-``)
        try:
            chosen, decision, fails = O.choose_first(self.ikc, self.brace_arm, cands, key=lambda c: c[1][1], label=lambda c: c[0], chain=lambda c: [c[1][0], c[1][1]],
                                                     evaluate=evaluate, q_start=self.ikc.q_locked(stance), stage="articulation brace")
        except _HandleFails as e:
            fails.append(str(e))
            extras["handle_alone"] = False
            return None, cands, fails, extras
        if chosen is not None:
            kind, (b_pre, b_contact) = chosen.proposal.payload
            if sum(1 for o in decision["options"] if not o.get("predicted") and o.get("source") == "ik") > 1:
                extras["orient"] = decision
            return (kind, b_pre, b_contact, chosen.result.q), cands, fails, extras
        extras["handle_alone"] = state["handle_ok"]
        return None, cands, fails, extras

    SECOND_STANCE_LATERALS = (0.10, -0.10, 0.20, -0.20)
    SECOND_STANCE_MAX = 10

    def _check_deadline(self):
        if time.time() > getattr(self, "_deadline", math.inf):
            raise Reject("time_budget", f"attempt budget of {self.gen('attempt_budget_s', ATTEMPT_BUDGET_S):.0f} s spent")

    def _brace_second_stance(self, manip: Stance, pose: TT.Pose, q_values, grasp_pair, target, support, cam_init):
        """Second stance for the brace + handle pair: the manipulation stance shifted along the edge by
        ``SECOND_STANCE_LATERALS`` with the trunk kept or moved to (0.5, 0) / (0.25, -0.3); on free floor, whole-body
        collision-free, the target (hull at q_init, handle contact) in frame from its camera. Returns
        ``(chosen | None, stance, tried_stances)``."""
        tried: List[Stance] = []
        pts = C.transform_points(pose, self._hull_q_init)
        wp0 = C.handle_waypoints(self.handle, self.jinfo, [self.q_init], pose)[0]
        # candidates: lateral shift x stand-off (kept / 10 cm closer) x trunk, ordered by the promise for the two-arm pair
        # (jar_diag 20:54: the handle arm passes with the target at base lateral -0.15 / depth 0.71 and the brace arm fails
        # every pre-contact there; the brace arm passes alone from the other trunk): the target between the shoulders
        # (|lateral| small) and not deeper than 0.62 m first
        for _, dlat, dso, trunk, st in self._second_stance_combos(manip, pose):
            if len(tried) >= self.SECOND_STANCE_MAX:
                break
            self._check_deadline()
            # the handle-only band (when computed) knows where the handle grasp is feasible per trunk: stances whose target
            # cell is not a handle cell cannot pass (0916-205838: the handle grasp alone passed at 1 of 9 second stances)
            if self.handle_only_band and self.bands is not None and not TT.in_band(self.bands, st, pose[0][:2], tol=1.0)[0]:
                continue
            if not self.kit.trav_free(st.base_xy) or not self.ikc.stance_collision_free(st):
                continue
            cam = self.camera_pose_at(st)
            if not TT.in_frame(self, cam, pts, float(self.g("view_margin_m"))) or not TT.in_frame(self, cam, wp0.contact.unsqueeze(0), TT.IN_FRAME_MARGIN):
                continue
            tried.append(st)
            chosen, _, _, extras = self._brace_from(st, pose, q_values, grasp_pair, target, support)
            if chosen is not None:
                st.extras.update({"second_stance": True, "lateral_shift": float(dlat), "stand_off_shift": float(dso), "trunk_from": [float(manip.translate), float(manip.tilt)]})
                log.info("%s: second stance: lateral %+.2f, stand-off %+.2f, trunk (%.2f, %+.2f) reaches the handle grasp + brace %s", self.gp.alias, dlat, dso,
                         trunk[0], trunk[1], chosen[0])
                return chosen, st, tried
        return None, manip, tried

    def _second_stance_combos(self, manip: Stance, pose: TT.Pose) -> List[Tuple[float, float, float, Tuple[float, float], Stance]]:
        """The second-stance candidates of :meth:`_brace_second_stance` (lateral shift x stand-off x trunk, most promising
        first): ``(score, dlat, dso, trunk, stance)``."""
        combos = []
        for dlat in self.SECOND_STANCE_LATERALS + (0.0,):
            for dso in (0.0, -0.10):
                for trunk in ((manip.translate, manip.tilt), (0.5, 0.0), (0.25, -0.3)):
                    if dlat == 0.0 and dso == 0.0 and abs(trunk[0] - manip.translate) < 1e-6 and abs(trunk[1] - manip.tilt) < 1e-6:
                        continue
                    so = max(0.25, float(manip.stand_off or TT.STAND_OFF) + dso)
                    st = manip.shifted(stand_off=so, lateral=manip.lateral + dlat, name="manipulation").with_trunk(trunk[0], trunk[1], "manipulation")
                    st.focus, st.arms = manip.focus, list(manip.arms) if isinstance(manip.arms, list) else manip.arms
                    b = st.world_to_base(pose[0][:2])
                    combos.append((abs(float(b[1])) + max(0.0, float(b[0]) - 0.62) + 0.1 * abs(trunk[1]), dlat, dso, trunk, st))
        combos.sort(key=lambda c: c[0])
        return combos

    def _handle_only_second_stance(self, manip: Stance, pose: TT.Pose, grasp_pair, target, support, cam_init, q_values=()):
        """Handle-only fallback (brace optional, 2026-09-18): the first second stance (free floor, whole-body collision-free,
        target + handle contact in frame, the grasp eef in the manipulation image) from which the handle grasp at q_init
        passes the exact IK alone **and** every required handle pose h(q_k) (``q_values``, the target re-posed) is reachable
        from that solution - the gates ``_sample_with`` applies after the stance choice (0918-225951: the first reachable
        stance failed ``gripper_not_in_frame`` / the k = 0.25 pose three attempts in a row). Returns
        ``(ik result | None, stance, tried_stances)``."""
        tried: List[Stance] = []
        pts = C.transform_points(pose, self._hull_q_init)
        wp0 = C.handle_waypoints(self.handle, self.jinfo, [self._q_main()], pose)[0]
        for _, dlat, dso, trunk, st in self._second_stance_combos(manip, pose):
            if len(tried) >= self.SECOND_STANCE_MAX:
                break
            self._check_deadline()
            if self.handle_only_band and self.bands is not None and not TT.in_band(self.bands, st, pose[0][:2], tol=1.0)[0]:
                continue
            if not self.kit.trav_free(st.base_xy) or not self.ikc.stance_collision_free(st):
                continue
            cam = self.camera_pose_at(st)
            if not TT.in_frame(self, cam, pts, float(self.g("view_margin_m"))) or not TT.in_frame(self, cam, wp0.contact.unsqueeze(0), TT.IN_FRAME_MARGIN):
                continue
            if not self.ikc.eef_in_frame(wp0.eef, cam):
                log.info("%s: handle-only second stance lateral %+.2f, stand-off %+.2f, trunk (%.2f, %+.2f): grasp eef outside the image", self.gp.alias, dlat, dso, trunk[0], trunk[1])
                continue
            tried.append(st)
            r_h = self.ikc.exact(list(grasp_pair), st, self.handle_arm, [support, target], cache=False, fingers=self.grasp_fingers)
            if not r_h.ok:
                log.info("%s: handle-only second stance lateral %+.2f, stand-off %+.2f, trunk (%.2f, %+.2f): %s", self.gp.alias, dlat, dso, trunk[0], trunk[1], str(r_h.reason)[:160])
                continue
            bad = None
            for qk in list(q_values)[1:]:
                self._set_joint(target, qk)
                self.kit.sync_world()
                wp = C.handle_waypoints(self.handle, self.jinfo, [qk], pose)[0]
                r = self.ikc.exact([wp.eef], st, self.handle_arm, ignore_for_last=[support, target], q_seed=r_h.q, cache=False, fingers=self.grasp_fingers)
                if not r.ok:
                    bad = (qk, str(r.reason)[:120])
                    break
            self._set_joint(target, self._q_main())
            self.kit.sync_world()
            if bad is not None:
                log.info("%s: handle-only second stance lateral %+.2f, stand-off %+.2f, trunk (%.2f, %+.2f): grasp ok, handle pose at q = %.3f fails (%s)", self.gp.alias, dlat, dso,
                         trunk[0], trunk[1], bad[0], bad[1])
                continue
            st.extras.update({"second_stance": True, "lateral_shift": float(dlat), "stand_off_shift": float(dso), "trunk_from": [float(manip.translate), float(manip.tilt)], "handle_only": True})
            log.info("%s: handle-only second stance: lateral %+.2f, stand-off %+.2f, trunk (%.2f, %+.2f) reaches the handle grasp and the %d required handle poses", self.gp.alias, dlat, dso,
                     trunk[0], trunk[1], max(0, len(q_values) - 1))
            return r_h, st, tried
        return None, manip, tried

    def _log_brace_failure(self, manip: Stance, pose: TT.Pose, q_values, grasp_pair, target, support, stances: List[Stance]):
        """BRACE_FAILURE diagnostic (owner 20:21: the numbers for a brace-optional decision): per stance the target's base
        coordinates, the handle grasp alone (pre + grasp exact IK), and per brace candidate the point's base coordinates,
        left-band membership, the brace alone (pre + contact, target ignored) and the two-arm reason."""
        try:
            out: Dict[str, Any] = {"target_base": [round(float(v), 3) for v in manip.world_to_base(pose[0][:2]).tolist()], "handle_arm": self.handle_arm, "brace_arm": self.brace_arm,
                                   "stances": []}
            for st in stances[:3]:  # the manipulation stance and the first two second stances (the two-arm verdicts are in the fails)
                r_h = self.ikc.exact(list(grasp_pair), st, self.handle_arm, [support, target], cache=False)
                cands, extras = self._brace_candidates_at(st, pose, q_values)
                rows = []
                for kind, (b_pre, b_contact) in cands[:6]:
                    r_b = self.ikc.exact([b_pre, b_contact], st, self.brace_arm, [target], cache=False)
                    pt = b_contact[0]
                    rows.append(f"{kind}@({float(st.world_to_base(pt[:2])[0]):.2f},{float(st.world_to_base(pt[:2])[1]):+.2f}):{'B' if r_b.ok else 'b'}")  # brace alone ok / fail
                out["stances"].append({"trunk": [round(float(st.translate), 2), round(float(st.tilt), 2)], "stand_off": st.stand_off, "target_base": [round(float(v), 2) for v in st.world_to_base(pose[0][:2]).tolist()],
                                       "shift": [st.extras.get("lateral_shift"), st.extras.get("stand_off_shift")], "handle_alone": bool(r_h.ok),
                                       "left_band": self.left_bands is not None, "braces": rows})
            log.info("%s: BRACE_FAILURE %s", self.gp.alias, json.dumps(out))
        except Exception as e:  # diagnostics only
            log.info("%s: BRACE_FAILURE diagnostic failed: %r", self.gp.alias, e)

    @property
    def _hull_q_init(self) -> th.Tensor:
        if not hasattr(self, "_hull_qi"):
            self._hull_qi = self._hull_at_q(self.obj(self.role), self.q_init)
        return self._hull_qi

    # ------------------------------------------------------------------ instance (§5.4 Phase 3)

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        init = self.init_stance
        errors = []
        # per-attempt budget (0916-205838: one attempt took 2356 s in the second-stance search + diagnostics)
        self._deadline = time.time() + float(self.gen("attempt_budget_s", ATTEMPT_BUDGET_S))
        order = list(rng.permutation(len(self.manip_candidates)))
        if self.gen("prefer_depth_m", None) is not None:
            # a target depth for the focus (owner 21:20 lever: the right hand's brace on the box's far-right corner needs the box near,
            # the left forearm over the box's left wall needs it not too near - 0.33 m into the table was out of reach, the nearest
            # cells put the forearm into the wall)
            d_pref = float(self.gen("prefer_depth_m"))
            order = sorted(range(len(self.manip_candidates)), key=lambda i: abs(float(self.manip_candidates[i].world_to_base(self.manip_candidates[i].focus[:2])[0]) - d_pref))
        elif self.gen("prefer_near", False):
            # owner 2026-09-19 14:34: the box as close to the robot as the survivors allow (the crossing / centre-line pinches of the
            # two-flap v2 sequence) - the Phase-2 survivors ordered by the focus depth in the base frame, nearest first
            order = sorted(range(len(self.manip_candidates)), key=lambda i: float(self.manip_candidates[i].world_to_base(self.manip_candidates[i].focus[:2])[0]))
        for k in order[:int(self.gen("candidates_per_attempt", 6))]:
            if time.time() > self._deadline:
                errors.append(f"attempt budget of {self.gen('attempt_budget_s', ATTEMPT_BUDGET_S):.0f} s spent")
                break
            cand = self.manip_candidates[int(k)]
            manip = cand.with_trunk(cand.translate, cand.tilt, "manipulation")
            manip.focus, manip.arms = cand.focus, [self.handle_arm]
            self.sidecar.reset_instance()
            self.sidecar.add_stance(init)
            for o in self.world.spawned.values():
                self.world.park(o, list(self.world.spawned).index(o.name))
            self.kit.sync_world()
            try:
                return self._sample_with(rng, init, manip, int(k))
            except Reject as e:
                errors.append(f"candidate {int(k)}: {e}")
                log.info("%s: %s", self.gp.alias, errors[-1][:300])
        raise Reject("target_placement", "; ".join(errors[-3:])[:1500])

    def _sample_with(self, rng, init: Stance, manip: Stance, k: int) -> Draft:
        d = Draft()
        tname = self.tname
        target, support = self.obj(self.role), self._support()
        cam_init = self.camera_pose_at(init)
        yaw = self.face_yaw + float(rng.uniform(-float(self.g("yaw_jitter_rad")), float(self.g("yaw_jitter_rad"))))
        pose = None
        for _ in range(10):
            r, a = float(rng.uniform(0.0, TT.FOCUS_RADIUS)), float(rng.uniform(0.0, 2 * math.pi))
            xy = manip.focus[:2] + th.tensor([r * math.cos(a), r * math.sin(a)])
            cand = TT.object_pose_on(self, self.hull, C.IDENTITY_QUAT, xy, yaw, self.top.z)
            if not self.top.polygon_supported(self.world.footprint(self.hull, cand), margin=TT.SUPPORT_MARGIN):
                continue
            if not TT.in_frame(self, cam_init, C.transform_points(cand, self._hull_q_init), float(self.g("view_margin_m"))):
                continue
            pose = cand
            break
        if pose is None:
            raise Reject("target_placement", "no supported, in-frame position within 5 cm of the focus")
        # target in the world at q_init
        target.set_position_orientation(*pose)
        self._set_joint(target, self.q_init)
        target.keep_still()
        self.kit.sync_world()
        if getattr(self, "v4", False):
            return self._sample_with_v4(rng, d, init, manip, k, target, support, pose, cam_init)
        if self.v2:
            return self._sample_with_v2(rng, d, init, manip, k, target, support, pose, cam_init)
        if getattr(self, "close_push", False):
            return self._sample_with_close_push(rng, d, init, manip, k, target, support, pose, cam_init)
        # two-flap carton (prep_push other_flap_down): every grasp check below runs with the OTHER flap pressed down, as it is at
        # demo time while the free hand holds it (0919-020546: with the flap flat the pre-grasp's lower pad passed through it -
        # "exact IK failed at pose 1 of 2" at every stance); the prep push itself re-poses the flap for its own poses
        if self.gen("prep_push", None) == "other_flap_down":
            other = str(self.gen("prep_push_joint", ""))
            if other in target.joints and other != self.joint_name:
                q0o = float(self.all_jinfo.get(other, {}).get("closed_q", 0.0)) if hasattr(self, "all_jinfo") else 0.0
                self._set_named_joint(target, other, q0o - float(self.gen("prep_push_angle_rad", 0.3)))
        q_values = [self.q_init] + [self.q_init + kk * (self.q_goal - self.q_init) for kk in self.required]
        wp0 = C.handle_waypoints(self.handle, self.jinfo, [self.q_init], pose)[0]
        grasp_pair = [wp0.pre_grasp, wp0.eef]
        # brace + handle grasp from the manipulation stance; else (owner 2026-09-16 20:21 item 1, the attachment treatment) from a
        # SECOND STANCE - the manipulation stance shifted along the edge / re-trunked, the target staying where it is - which then
        # becomes the recorded manipulation stance (nothing is carried: the init -> manipulation transition is recorded after)
        # generation.brace_search (2026-09-18, default true): false = no brace at all (handle-only instance; needs brace_optional) -
        # the blender eyedvd pod run 0918-220524 found a two-arm pair at a second stance on 4 attempts and every one failed the
        # program check's two-arm grasp chain ("approach: exact IK failed at pose 1 of 3 ... left arm locked at its contact")
        brace_search = bool(self.gen("brace_search", True))
        if not brace_search and not self.gen("brace_optional", False):
            raise Reject("spec", "generation.brace_search false needs generation.brace_optional true")
        tried_stances = [manip]
        if brace_search:
            chosen, cands, fails, brace_extras = self._brace_from(manip, pose, q_values, grasp_pair, target, support)
            if chosen is None:
                chosen, manip2, tried = self._brace_second_stance(manip, pose, q_values, grasp_pair, target, support, cam_init)
                tried_stances += tried
                if chosen is not None:
                    manip = manip2
        else:
            chosen, cands, fails, brace_extras = None, [], [], {"brace_search": False}
        brace_state = "present"
        if chosen is None:
            if brace_search:
                self._log_brace_failure(manip, pose, q_values, grasp_pair, target, support, tried_stances)
            if not self.gen("brace_optional", False):
                raise Reject("contact_unreachable", f"no brace of {[c[0] for c in cands][:8]} + handle grasp at q_init passes the exact IK from the manipulation stance or "
                                                    f"{len(tried_stances) - 1} second stances ({sorted(set(fails))[:3]})")
            # D-008 (prepared, owner 2026-09-16 21:54; ``generation.brace_optional``): as axial / touching / blocks, the brace is
            # recorded only when a joint cell exists (here: the two-arm IK from the manipulation or a second stance); otherwise
            # the instance is handle-only — contacts.brace absent, summary.brace = "none", manipulation arms = [handle_arm]
            r_h = self.ikc.exact(list(grasp_pair), manip, self.handle_arm, [support, target], cache=False, fingers=self.grasp_fingers)
            if not r_h.ok:
                # the handle alone from a SECOND stance (2026-09-18): the same shifted / re-trunked stances the brace search walks,
                # accepted when the handle grasp at q_init passes there (the blender's ring is reachable from stand-off -0.10 /
                # trunk 0.25 but not from the Phase-2 stance; the two-arm pairs found there fail the program check)
                # with stage 2 (fist push from the reach limit on) the stance only has to reach the grasp and the first waypoint
                need = q_values[:2] if (self.gen("stage2_push", False) and self.direction == "open") else q_values
                r_h, manip2, tried = self._handle_only_second_stance(manip, pose, grasp_pair, target, support, cam_init, q_values=need)
                tried_stances += tried
                if r_h is None:
                    raise Reject("handle_unreachable", f"handle grasp at q_init fails the exact IK from the manipulation stance and {len(tried)} second stances")
                manip = manip2
            chosen, brace_state = (None, None, None, r_h.q), "none"
            log.info("%s: BRACE_OPTIONAL handle-only instance (%s; %d stances tried)", self.gp.alias, "brace search off" if not brace_search else "no brace passed", len(tried_stances))
        self.record_transition(init, manip)
        kind, b_pre, b_contact, q_both = chosen
        if self.handle_only_band and kind is not None:
            brace_extras["pose_class"] = kind
            kind = "body_face"
        # owner 2026-09-19 01:10 (two-flap cartons, generation.prep_push "other_flap_down"): the other flap is pressed down at the
        # seam by the free hand while the handle hand pinches the target flap's edge; both hands together must pass the exact
        # IK (the press held during the grasp), then the free hand moves to a top-rim brace beside the other flap's hinge
        prep_rec = None
        top_brace = None
        if self.gen("prep_push", None) == "other_flap_down":
            prep_rec = self._prep_push_other_flap(manip, pose, target, support)
            p_arm = prep_rec["record"].arm
            if p_arm != self.handle_arm:
                # only the PRESSING pose is simultaneous with the grasp (the fist's pre-contact is over before the other hand moves)
                res = TT.brace_second_arm(self, manip, self.handle_arm, grasp_pair, p_arm, [prep_rec["record"].goal],
                                          ignore_work_last=[support, target], ignore_brace_last=[target], fingers=self.grasp_fingers)
                if not res.ok:
                    raise Reject("handle_unreachable", f"handle grasp with the {p_arm} hand holding the other flap down fails the exact IK: {res.reason}")
                q_both = res.q
                if b_contact is None and self.gen("prep_brace", True):
                    top_brace = self._brace_top_rim(manip, pose, target, support, grasp_pair, p_arm)
        # grasp poses at the required waypoints with the target re-posed for each world sync (brace arm locked)
        reach: Dict[str, bool] = {"0.0": True}
        waypoints: Dict[str, Any] = {"0.0": wp0.to_sidecar()}
        # owner 2026-09-18 22:38 (generation.stage2_push, default off): when a required handle pose is beyond the arm's reach,
        # the follow stops at the last reachable waypoint (the brake holds the lid there), the hand lets go, closes, and
        # pushes the lid's underside from inside the opening along the hinge arc to the goal (contacts.push, a fist on the
        # lid; articulation.follow_to / stage2)
        stage2 = bool(self.gen("stage2_push", False)) and self.direction == "open"
        # generation.stage2_max_follow_k: the hand swings the lid at most to this fraction even when farther handle poses pass
        # the IK (0919-001843: the margin-0 IK reached 98 deg from the shifted stance, so the push never happened); the fist
        # push covers the rest. The fist's pre-contact lies 10 cm below the underside along its normal, so q_mid must leave
        # the lid open enough for it to clear the rim (eyedvd: >= ~60 deg)
        max_k = self.gen("stage2_max_follow_k", None) if stage2 else None
        k_mid: Optional[float] = None
        for kk in self.required:
            if max_k is not None and kk > float(max_k) + 1e-9 and k_mid is not None:
                log.info("%s: follow capped at k = %g (stage2_max_follow_k %s): the fist push takes the lid on from q %.3f", self.gp.alias, k_mid, max_k, self.q_init + k_mid * (self.q_goal - self.q_init))
                break
            qk = self.q_init + kk * (self.q_goal - self.q_init)
            self._set_joint(target, qk)
            self.kit.sync_world()
            wp = C.handle_waypoints(self.handle, self.jinfo, [qk], pose)[0]
            r = self.ikc.exact([wp.eef], manip, self.handle_arm, ignore_for_last=[support, target], q_seed=q_both, cache=False, fingers=self.grasp_fingers)
            key = f"{kk:g}"
            waypoints[key] = wp.to_sidecar()
            reach[key] = bool(r.ok)
            if not r.ok:
                if stage2 and k_mid is not None:
                    log.info("%s: handle pose at k = %g out of reach -> stage 2: follow to k = %g (q %.3f), then push the lid from inside to q_goal %.3f", self.gp.alias, kk, k_mid,
                             self.q_init + k_mid * (self.q_goal - self.q_init), self.q_goal)
                    break
                self._set_joint(target, self.q_init)
                self.kit.sync_world()
                raise Reject("handle_unreachable", f"handle pose at q_k (k = {kk}) fails the exact IK with the brace arm locked")
            k_mid = kk
        stage2_rec = None
        if stage2 and k_mid is not None and not all(reach.get(f"{kk:g}", False) for kk in self.required):
            q_mid = self.q_init + k_mid * (self.q_goal - self.q_init)
            stage2_rec = self._stage2_push(manip, pose, target, support, q_mid, q_both)  # raises Reject when no fist push chain passes
        else:
            k_mid = None
        for kk in [float(x) for x in self.g("waypoint_fractions")]:
            key = f"{kk:g}"
            if key not in waypoints:
                waypoints[key] = C.handle_waypoints(self.handle, self.jinfo, [self.q_init + kk * (self.q_goal - self.q_init)], pose)[0].to_sidecar()
                reach[key] = False
        self._set_joint(target, self.q_init)
        self.kit.sync_world()
        # combined whole-body check of both arms against the scene (target excluded: both hands touch it)
        with self.kit.ignoring([target]):
            if not bool(self.kit.collision_free(q_both, margin=0.01)[0]):
                raise Reject("stance_collision", "combined brace + handle configuration collides with the scene")
        cam_manip = self.camera_pose_at(manip)
        for nm, p in [("handle grasp", wp0.eef)] + ([("brace", b_contact)] if b_contact is not None else []):
            if not self.ikc.eef_in_frame(p, cam_manip):
                raise Reject("gripper_not_in_frame", f"{nm} eef pose outside the manipulation image (5 %)")
        TT.whole_body_check_at(self, [init, manip], {tname: pose}, require=True, reason="stance_collision")
        # record
        manip.arms = [self.handle_arm, self.brace_arm] if b_contact is not None else [self.handle_arm]
        self.sidecar.add_stance(manip)
        if b_contact is not None:
            brace_dir = C.approach_axis(b_contact[1])
            self.sidecar.add_contact(TT.BraceRecord("brace", self.brace_arm, b_pre, b_contact, direction=brace_dir, brace_kind=kind, object=tname,
                                                    extras={"stop_clearance_m": 0.0, **brace_extras}))
            manip.contacts = ["brace"]
        manip.extras["brace"] = brace_state
        self.sidecar.articulation = {"object": tname, "joint": self.joint_name, "joint_type": self._joint_type(), "q_init": float(self.q_init), "q_goal": float(self.q_goal),
                                     "waypoints": waypoints, "handle_reach": {kk: v for kk, v in reach.items() if kk != "0.0"}, "passive_from": self.passive_from,
                                     "follow_mode": "arm", "handle_arm": self.handle_arm, "brace_kind": kind, "direction": self.direction,
                                     "handle_width_m": self.handle.get("width"), "handle_kind": self.handle.get("kind"),
                                     "object_pose": {"position": [round(float(v), 5) for v in pose[0].tolist()], "orientation": [round(float(v), 6) for v in pose[1].tolist()]},
                                     # the target hinge in the world (2026-09-19): the release of an edge pinch slides off the free
                                     # edge radially before the clear leg (oracle Retreat._edge_escape)
                                     "hinge_world": [round(float(v), 5) for v in (pose[0] + C.quat_rotate(pose[1], th.tensor(self.jinfo["hinge_point_obj"], dtype=th.float32))).tolist()],
                                     "axis_world": [round(float(v), 5) for v in C.unit(C.quat_rotate(pose[1], th.tensor(self.jinfo["axis_dir_obj"], dtype=th.float32))).tolist()]}
        if stage2_rec is not None:
            self.sidecar.add_contact(stage2_rec["record"])
            manip.contacts = list(manip.contacts or []) + ["push"]
            if stage2_rec["record"].arm not in manip.arms:
                manip.arms = list(manip.arms) + [stage2_rec["record"].arm]
            stage2_rec["facts"]["k_mid"] = float(k_mid)
            self.sidecar.articulation.update({"follow_to": float(k_mid), "stage2": stage2_rec["facts"]})
        if prep_rec is not None:
            self.sidecar.add_contact(prep_rec["record"])
            manip.contacts = list(manip.contacts or []) + ["push_prep"]
            if prep_rec["record"].arm not in manip.arms:
                manip.arms = list(manip.arms) + [prep_rec["record"].arm]
            self.sidecar.articulation["prep_push"] = prep_rec["facts"]
        if top_brace is not None:
            self.sidecar.add_contact(top_brace)
            manip.contacts = list(manip.contacts or []) + ["brace"]
            manip.extras["brace"] = "present"
            if top_brace.arm not in manip.arms:
                manip.arms = list(manip.arms) + [top_brace.arm]
        return self._finish_draft(d, target, pose, init, manip, k, reach, kind=kind, brace_state=brace_state, stage2_rec=stage2_rec)


    def _finish_draft(self, d: Draft, target, pose: TT.Pose, init: Stance, manip: Stance, k: int, reach: Dict[str, bool], kind=None, brace_state: str = "none",
                      stage2_rec=None, summary_extra: Optional[Dict[str, Any]] = None) -> Draft:
        """The draft's common tail: init / goal joint vectors (every joint closed but the target's), the constraint records
        (joint brake / drives), the robot's init, the articulation and pose targets, the language and the summary."""
        tname = self.tname
        joint_names = list(target.joints)
        # the init / goal vectors: every joint closed (the pressed non-target flap of a two-flap carton goes back to its closed
        # position, NOT its extended lower limit - 0919-043345 "goal_unstable: j_left_lid drifted to -0.002 (target -0.350)")
        q_closed = target.get_joint_positions().clone()
        for n, j in target.joints.items():
            q_closed[joint_names.index(n)] = float(self.all_jinfo[n].get("closed_q", 0.0)) if n in getattr(self, "all_jinfo", {}) else float(j.lower_limit)
        if self.gen("prep_push", None) in ("other_flap_down", "two_flap_v2", "two_flap_v4"):
            other = str(self.gen("prep_push_joint", ""))
            if other in target.joints:
                self._set_named_joint(target, other, float(q_closed[joint_names.index(other)]))
        q_init_vec, q_goal_vec = q_closed.clone(), q_closed.clone()
        q_init_vec[joint_names.index(self.joint_name)] = self.q_init
        q_goal_vec[joint_names.index(self.joint_name)] = self.q_goal
        d.object_poses[tname] = pose
        d.object_joints[tname] = q_init_vec
        d.goal_joints[tname] = q_goal_vec
        recs = []
        if self.joint_friction is not None:
            # the loader re-applies it (run_isg._restore_constraints -> constraints.apply_joint_brake); the stable set / goal
            # settle above were probed with it
            spec = self.joint_friction if isinstance(self.joint_friction, dict) else {"friction": float(self.joint_friction)}
            recs.append({"type": "joint_friction", "object": tname, "joint": self.joint_name, **spec, "stock": {"friction": 0.2, "damping": 2.86},
                         "moving_link_mass_kg": self.template_facts.get("moving_link_mass_kg"),
                         "note": "resistance of the moved joint (generation.joint_friction, constraints.apply_joint_brake): stock, this lid falls to its "
                                 "gravitational equilibrium and no open value below the upper limit is stable; damping + max_effort = the joint drive as a "
                                 "Coulomb brake of max_effort N·m (torque = clamp(-damping*omega, +-max_effort))"})
        for jn, spec in dict(self.gen("joint_drives", None) or {}).items():
            if jn != self.joint_name:
                recs.append({"type": "joint_friction", "object": tname, "joint": jn, **{k: spec.get(k) for k in ("stiffness", "damping", "max_effort", "lower_limit_rad", "upper_limit_rad") if spec.get(k) is not None},
                             "openable_joints": [self.joint_name],
                             "note": "drive of another joint of the target (generation.joint_drives): a spring to its closed position with an extended lower limit - the "
                                     "non-target flap of a two-flap carton is pressed down to open the seam and comes back when the fist leaves"})
        if recs:
            d.extras["constraints"] = recs
        d.robot_q = self.stance_q(init)
        d.base_pose2d = tuple(init.base_pose2d)
        tol = self._joint_tol(self.q_goal - self.q_init)
        if getattr(self, "close_push", False):
            tol = float(self.gen("close_tolerance_rad", 0.05))  # owner 22:59: close is 0 deg, fixed
        d.articulation_targets[tname] = {"joints": {n: {"position": float(q_goal_vec[joint_names.index(n)]), "tolerance": tol} for n in joint_names}}
        d.pose_targets[tname] = full_pose_target(pose[0], pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        mc = self.choice.models[self.role]
        self.sidecar.describe(tname, mc.category.replace("_", " "))
        self.sidecar.language["goal_numbers"]["joint_delta"] = float(self.q_goal - self.q_init)
        d.summary.update({"direction": self.direction, "joint": self.joint_name, "q_init": self.q_init, "q_goal": self.q_goal, "brace_kind": kind, "brace": brace_state,
                          "handle_arm": self.handle_arm, "manipulation_candidate": k, "handle_reach": reach, "second_stance": bool(manip.extras.get("second_stance")),
                          "stage2": None if stage2_rec is None else stage2_rec["facts"], "joint_friction": self.template_facts.get("joint_friction"), **(summary_extra or {})})
        self.brace_missing = brace_state == "none"
        return d

    # ------------------------------------------------------------------ push-close (owner 2026-09-19 22:59)

    def _sample_with_close_push(self, rng, d: Draft, init: Stance, manip: Stance, k: int, target, support, pose: TT.Pose, cam_init) -> Draft:
        """:meth:`_close_push_flow` with the fist's wrist turn chosen by the orientation rule (owner 2026-09-27): the tip handle and
        its twin (the fist turned pi about its downward approach - the wrist camera on the other side) are ranked on their first
        contact at q_init from the Phase-2 stance (joint-limit safe, then easier to reach, the handle's own turn within the
        tolerance; a turn that does not solve there stays in unscored - the flow searches other stances; a turn whose wrist camera
        comes within 5 mm of the world at the contact is out), the flow runs with the preferred turn, with the other one when it
        rejects."""
        base = getattr(self, "_close_push_handle0", None) or self.handle
        arm, q0 = self.handle_arm, float(self.q_init)
        for n in target.joints:  # as the flow checks: every other joint closed, the target's at q_init
            if n != self.joint_name:
                self._set_named_joint(target, n, float(self.all_jinfo.get(n, {}).get("closed_q", 0.0)) if n in getattr(self, "all_jinfo", {}) else float(target.joints[n].lower_limit))
        self._set_joint(target, q0)
        self.kit.sync_world()
        props = []
        for rank, h in enumerate(self._pinch_variants(base)):
            wp = C.handle_waypoints(h, self.jinfo, [q0], pose)[0]
            props.append(O.Proposal(f"fist {[round(v, 2) for v in h['pinch_axis']]}", rank, h, [wp.pre_grasp, wp.eef]))

        def evaluate(p):
            r = self.ikc.exact(list(p.chain), manip, arm, [support, target], cache=False, fingers=C.FINGER_CLOSED)
            return True, r, (None if r.ok else []), ""

        state0 = (None if manip.q is None else manip.q.clone(), manip.name, dict(manip.extras))
        runs: Dict[str, Any] = {"n": 0, "rejects": []}

        def flow(sc):
            if runs["n"]:
                # the other turn after a rejected flow: back to the state the flow starts from
                self.sidecar.reset_instance()
                self.sidecar.add_stance(init)
                manip.q, manip.name = (None if state0[0] is None else state0[0].clone()), state0[1]
                manip.extras.clear()
                manip.extras.update(state0[2])
            runs["n"] += 1
            self.handle = sc.proposal.payload
            try:
                runs["draft"] = self._close_push_flow(rng, d if runs["n"] == 1 else Draft(), init, manip, k, target, support, pose, cam_init)
            except Reject as e:
                runs["rejects"].append(e)
                return False, str(e)[:240]
            return True, ""

        chosen, dec = O.choose(self.ikc, arm, props, evaluate, self.ikc.q_locked(manip, fingers=C.FINGER_CLOSED), stage="close push fist", downstream=flow, map_twins=False)
        if chosen is None:
            self.handle = base
            rej = runs["rejects"]
            if not rej:
                raise Reject("push_unreachable", f"push-close: neither fist turn passes ({'; '.join(str(o.get('why', '')) for o in dec['options'])[:300]})")
            if len(rej) == 1:
                raise rej[0]
            raise Reject(rej[0].reason, f"{str(rej[0])[:700]} | the other fist turn: {str(rej[1])[:400]}")
        draft = runs["draft"]
        if sum(1 for o in dec["options"] if o.get("source") == "ik") > 1:
            draft.summary["close_push_orient"] = dec
            for name, rec in list(self.sidecar.contacts.items()):
                if str(name).startswith("push") and isinstance(getattr(rec, "extras", None), dict):
                    rec.extras["orient"] = dec
        return draft

    def _close_push_flow(self, rng, d: Draft, init: Stance, manip: Stance, k: int, target, support, pose: TT.Pose, cam_init) -> Draft:
        """Close by pushing the moving link's tip along the closing arc with the closed fist (no grasp): from the stance the
        fist rides the arc as far as the arm reaches (the required fractions of the remaining travel, exact IK), the segment
        is recorded as a push (``contacts.push``, ``push2``, ...), the brake holds the link there, the robot re-stances
        (a second stance around the current one, the transition recorded with the link at that value) and the next segment
        continues from there - up to ``close_push_max_segments`` - until q_goal = the closed limit."""
        from ispatialgym.sampling.v2 import program_check as PC
        from ispatialgym.sampling.v2.stances import ContactRecord

        tname, arm = self.tname, self.handle_arm
        idx_arm = self.kit.robot.arm_control_idx[arm]
        q0, qg = float(self.q_init), float(self.q_goal)
        fractions = [float(x) for x in self.g("waypoint_fractions")]
        # every other joint at its closed value for the checks (the instance starts that way, ``_finish_draft``): the other flap of a
        # two-flap carton standing open in the sampler's sim would block the brace on that side (0920-191145: no braced stance)
        for n in target.joints:
            if n != self.joint_name:
                self._set_named_joint(target, n, float(self.all_jinfo.get(n, {}).get("closed_q", 0.0)) if n in getattr(self, "all_jinfo", {}) else float(target.joints[n].lower_limit))
        pts = C.transform_points(pose, self._hull_q_init)
        hinge_w = pose[0] + C.quat_rotate(pose[1], th.tensor(self.jinfo["hinge_point_obj"], dtype=th.float32))
        axis_w = C.unit(C.quat_rotate(pose[1], th.tensor(self.jinfo["axis_dir_obj"], dtype=th.float32)))

        def wp_at(q):
            self._set_joint(target, q)
            self.kit.sync_world()
            return C.handle_waypoints(self.handle, self.jinfo, [q], pose)[0]

        # ---- helpers: the arc coverage of a stance from q_from (the farthest reachable fraction of the remaining travel; the segment
        # end's retreat pose must be reachable too), the wide-stop brace of the other arm on the box face the push shoves it toward
        retreat_m = float(self.gen("close_push_retreat_m", 0.06))
        overshoot = float(self.gen("close_push_overshoot_rad", 0.15))

        class IKResultLite:
            def __init__(self, q):
                self.q, self.ok = q, True

        over_ok: Dict[int, float] = {}  # id(stance) -> the overshoot beyond q_goal its arm reaches (the final segment's arc end)
        ret_q: Dict[int, th.Tensor] = {}  # id(stance) -> the retreat configuration at its coverage end (where that segment's Retreat leaves the arm)

        def coverage(st: Stance, q_from: float, seed):
            """(q_to, q_seed_at_q_to) of the farthest reachable arc pose from ``st`` (None when none). The arc is TRACKED from the
            contact configuration pose to pose (one IK branch, as the oracle rides it - program check 0921-010925: an independently
            solved arc end sat in a self-colliding branch); the retreat pose at every step must solve with a joint-limit margin. When
            the goal itself is reached, the overshoot pose (``q_arc`` of the final segment; 0920-171440) is checked too - the
            largest of overshoot / half / none that passes is kept."""
            best, last_q = None, seed
            wp_prev = wp_at(q_from)
            for kk in fractions:
                q = q_from + kk * (qg - q_from)
                wp = wp_at(q)
                q_t, _why = self.ikc.track_segment(last_q, wp_prev.eef, wp.eef, arm, ignore_mid=[target], ignore_end=[support, target], ignore_start=[support, target])
                if q_t is None:
                    break
                rr = IKResultLite(q_t)
                wp_prev = wp
                r_ret = self.ikc.exact([C.pre_pose(wp.eef, retreat_m)], st, arm, ignore_for_last=[support, target], q_seed=rr.q, cache=False, fingers=C.FINGER_CLOSED)
                if not r_ret.ok:
                    break
                # the program check rejects a retreat pose within JOINT_LIMIT_MARGIN of a limit (0920-213257: right_arm_joint6 at 97 mrad)
                m_ret, _ = self.ikc.joint_limit_margin(r_ret.q, arm)
                if m_ret < float(getattr(self.ikc, "JOINT_LIMIT_MARGIN", 0.0)) + 0.01:
                    break
                best, last_q = q, rr.q
                ret_q[id(st)] = r_ret.q
            if best is not None and abs(best - qg) < 1e-6:
                sense = 1.0 if qg >= q_from else -1.0
                over_ok[id(st)] = 0.0
                for ov in (overshoot, 0.5 * overshoot):
                    wp_o = C.handle_waypoints(self.handle, self.jinfo, [qg + sense * ov], pose)[0]  # the fist pose beyond the limit (the link stays at its limit)
                    q_o, _w = self.ikc.track_segment(last_q, wp_prev.eef, wp_o.eef, arm, ignore_mid=[target], ignore_end=[support, target], ignore_start=[support, target])
                    if q_o is not None:
                        over_ok[id(st)] = ov
                        break
            return best, last_q

        def first_contact(st: Stance, q_from: float, q_arm=None):
            """The [pre, contact] pair at q_from from ``st`` (exact IK), plus the pre pose's own configuration (the stance's recorded arms).
            ``q_arm`` (28-D, where the previous segment's retreat leaves the arm - the travel carries it): the pair's seed; the IK keeps
            the solution nearest its seed, so the next pre-contact lies in the branch the arm is already in."""
            wp = wp_at(q_from)
            seed = None
            if q_arm is not None:
                seed = self.ikc.q_locked(st, fingers=C.FINGER_CLOSED).clone()
                seed[idx_arm] = th.as_tensor(q_arm, dtype=th.float32)[idx_arm]
            r = self.ikc.exact([wp.pre_grasp, wp.eef], st, arm, [support, target], q_seed=seed, cache=False, fingers=C.FINGER_CLOSED)
            if not r.ok:
                return None
            r_pre = self.ikc.exact([wp.pre_grasp], st, arm, [support, target], q_seed=r.q, cache=False, fingers=C.FINGER_CLOSED)
            return r, (r_pre.q if r_pre.ok else None)

        def segment_targets(st: Stance, q_from: float, q_to: float):
            """The segment's key poses as ``Push.targets`` gives them to the program check: the pre-contact, then the arc (the contact at
            q_from, mid, end - past the goal by the overshoot ``st`` reaches when the segment ends there), as the segment loop records them."""
            wpf = C.handle_waypoints(self.handle, self.jinfo, [q_from], pose)[0]
            sense = 1.0 if qg >= q_from else -1.0
            q_arc = q_to + sense * over_ok.get(id(st), 0.0) if abs(q_to - qg) < 1e-6 else q_to
            mid_pose, goal_pose = C.orbit_waypoints(wpf.eef, hinge_w, C.quat_from_axis_angle(axis_w, q_arc - q_from), n=2, frame="world")
            return [("pre", [wpf.pre_grasp]), ("waypoints", [wpf.eef, mid_pose, goal_pose])]

        def fingertip_out(st: Stance, q_from: float, q_to: float):
            """The program check's fingertip test (``program_check.fingertips_in_frame``: QA's 5 % margin + the execution slack) on the
            segment's key poses from ``st``: None when every pose passes, else the first one outside - the check rejects the attempt for it."""
            if os.environ.get("ISG_FINGERTIP_FRAME_CHECK", "on").lower() in ("off", "0", "false"):
                return None
            return PC.fingertips_in_frame(self, st, "push", segment_targets(st, q_from, q_to))

        brace_on = bool(self.gen("close_push_brace", False))
        brace_arm = "left" if arm == "right" else "right"

        def find_brace(st: Stance, pair, q_from: float):
            """Owner 2026-09-20 16:24: the pushes shove the box along the push direction - the other hand braces the face the box would
            slide toward with an OPEN gripper (the touching demo's wide stop, ``fingers_opening_m``): the pad on the face centre at
            ``close_push_brace_height_frac`` of the height, approach into the face, tips side by side along the face."""
            wpf = wp_at(q_from)
            t_h = th.tensor([float(wpf.approach[0]), float(wpf.approach[1]), 0.0])
            if float(th.linalg.norm(t_h)) < 1e-6:
                return None
            t_h = C.unit(t_h)  # the push's horizontal direction: the box slides this way, the brace holds the face on this side
            hull_w = C.transform_points(pose, self.hull)
            proj = hull_w @ t_h
            face_d = float(proj.max())  # the face on the +t_h side
            centre = hull_w.mean(dim=0)
            z_b = float(hull_w[:, 2].min()) + float(self.gen("close_push_brace_height_frac", 0.6)) * (float(hull_w[:, 2].max()) - float(hull_w[:, 2].min()))
            along = C.unit(th.cross(t_h, th.tensor([0.0, 0.0, 1.0]), dim=0))  # horizontal, along the face
            # along the face: the end nearer the robot (+ a corner inset) rather than the face's centre - the box sits 3 cm from the
            # table's near edge and its far-side face centre was beyond the left arm from every stance (0920-191145 / 201653)
            pa = hull_w @ along
            base_xy = th.tensor([float(st.base_pose2d[0]), float(st.base_pose2d[1]), 0.0])
            c_along = float(th.dot(centre, along))
            ends = [float(pa.min()), float(pa.max())]
            end_pts = [centre + along * (e - c_along) for e in ends]
            near_i = 0 if float(th.linalg.norm((end_pts[0] - base_xy)[:2])) <= float(th.linalg.norm((end_pts[1] - base_xy)[:2])) else 1
            corner_inset = float(self.gen("close_push_brace_corner_inset_m", 0.04))
            a_pos = ends[near_i] + (corner_inset if near_i == 0 else -corner_inset)
            a_pos = min(max(a_pos, ends[0] + 0.01), ends[1] - 0.01)
            on_face = centre + along * (a_pos - c_along)
            opening = float(self.gen("close_push_brace_opening_m", 0.04))
            kinds = [str(k) for k in (self.gen("close_push_brace_kinds", ["top_press", "opposite_face"]) or [])]
            whys = []
            for kind in kinds:
                # owner 2026-09-21 23:23 (0921-030848: the stop drove the pads 5 mm into the face, the box rose 5 mm and the base rocked):
                # the pads stop ``close_push_brace_clearance_m`` short of the face; the push shoves the box that far into them
                b_clear = float(self.gen("close_push_brace_clearance_m", 0.006))
                if kind == "top_press":
                    # the open hand presses DOWN on the box's top at the side it slides toward (the closed other flap over the far wall's
                    # rim, close_push_brace_rim_inset_m inside the face): friction holds the box, and a top-down pad is the easy reach
                    inset = float(self.gen("close_push_brace_rim_inset_m", 0.035))
                    p_top = on_face + t_h * (face_d - float(th.dot(on_face, t_h)) - inset)
                    # the closed flaps' top = the hinge height (the hull's top is the standing target flap at q_init)
                    p_top = th.tensor([float(p_top[0]), float(p_top[1]), float(hinge_w[2]) + 0.003 + b_clear])
                    contact = C.contact_eef(p_top, th.tensor([0.0, 0.0, -1.0]), along)
                    direction = th.tensor([0.0, 0.0, -1.0])
                else:
                    p_face = on_face + t_h * (face_d - float(th.dot(on_face, t_h)) + b_clear)
                    p_face = th.tensor([float(p_face[0]), float(p_face[1]), z_b])
                    contact = C.contact_eef(p_face, -t_h, along)
                    direction = -t_h
                pre = C.pre_pose(contact, 0.08)
                touched = _LinkSubset(target, {self.jinfo.get("child_link")})
                res = TT.brace_second_arm(self, st, arm, list(pair), brace_arm, [pre, contact], ignore_work_last=[support, target], ignore_brace_last=[target],
                                          fingers=C.FINGER_CLOSED, ignore_while_locked=[touched] if touched.links else [target])
                if res.ok:
                    # the brace's lift-off (program check 0920-232450: left_arm_joint7 within 96 mrad of its limit at the retreat pose) - the
                    # clear pose 10 cm back along -a, solved with the fist locked at its contact, must keep the joint-limit margin
                    ret = C.pre_pose(contact, float(self.gen("close_push_brace_retreat_m", 0.06)))
                    # the other arm has retreated from the flap by then (its exact configuration is not known here): the brace arm alone
                    r_ret = self.ikc.exact([ret], st, brace_arm, ignore_for_last=[support, target], cache=False, fingers=C.FINGER_CLOSED)
                    m_ret = self.ikc.joint_limit_margin(r_ret.q, brace_arm)[0] if r_ret.ok else -1.0
                    if not r_ret.ok or m_ret < float(getattr(self.ikc, "JOINT_LIMIT_MARGIN", 0.0)) + 0.01:
                        whys.append(f"{kind}: brace ok but its lift-off pose {'has no solution' if not r_ret.ok else f'sits {1000 * m_ret:.0f} mrad from a joint limit'}")
                        continue
                if not res.ok:
                    b = st.world_to_base(pre[0][:2])
                    r_free = self.ikc.exact([pre], st, brace_arm, ignore_for_last=[support, target], cache=False, fingers=C.FINGER_CLOSED)
                    if not r_free.ok:
                        why_d = "unreachable alone (box ignored)"
                    else:
                        r_box = self.ikc.exact([pre], st, brace_arm, ignore_for_last=[support], cache=False, fingers=C.FINGER_CLOSED)
                        if not r_box.ok:
                            q_c = r_free.q
                            with self.ikc.ignoring([support]):
                                why_d = f"the pre-pose meets the box: {self.ikc._collision_blame(q_c, [support])}"
                        else:
                            why_d = "reachable with the box; fails with the fist arm locked at its contact (the arms are in each other's way)"
                    whys.append(f"{kind}: {str(res.reason)[:60]} (pre at base depth {float(b[0]):.2f} lateral {float(b[1]):+.2f} z {float(pre[0][2]):.2f}: {why_d})")
                    continue
                return TT.BraceRecord("brace", brace_arm, pre, contact, direction=direction, brace_kind=kind, object=tname, stance=st.name,
                                      extras={"stop_clearance_m": b_clear, "pose_class": "wide_stop", "fingers_opening_m": opening, "stop_width_m": 2 * opening + 0.02,
                                              "retreat_m": float(self.gen("close_push_brace_retreat_m", 0.06)),
                                              "note": "the box slides along the push; the other hand, open, holds it (owner 2026-09-20 16:24): a top press on the far rim or a stop on the far face"})
            log.info("%s: push-close brace (%s hand, open) with the fist at q %.2f fails: %s", self.gp.alias, brace_arm, q_from, " | ".join(whys)[:700])
            return None

        # ---- stance 1: among the Phase-2 candidate and its second stances, the one covering the most of the arc from q_init (owner
        # 16:24: fewer re-stances - and one stance when it covers everything); the first contact and its retreat validator-checked
        best1 = None
        n_eval = 0
        for _, dlat, dso, trunk, st in [(0.0, 0.0, 0.0, (manip.translate, manip.tilt), manip)] + self._v2_stance_combos(manip, pose):
            if n_eval >= int(self.gen("close_push_stance_eval", 10)):
                break
            self._check_deadline()
            if st is not manip:
                if not self.kit.trav_free(st.base_xy) or not self.ikc.stance_collision_free(st):
                    continue
                cam = self.camera_pose_at(st)
                if not TT.in_frame(self, cam, pts, float(self.g("view_margin_m"))) or not self.ikc.eef_in_frame(wp_at(q0).eef, cam):
                    continue
            fc = first_contact(st, q0)
            if fc is None:
                continue
            q_to, _ = coverage(st, q0, fc[0].q)
            if q_to is None:
                continue
            out = fingertip_out(st, q0, q_to)
            if out is not None:
                log.info("%s: push-close stance candidate lateral %+.2f stand-off %+.2f trunk (%.2f, %+.2f) covers q %.2f -> %.2f but the fingertip leaves the frame at "
                         "push.%s[%d] (u %.0f, v %.0f): skipped", self.gp.alias, dlat, dso, trunk[0], trunk[1], q0, q_to, out["key"], out["index"], out["u"], out["v"])
                continue
            n_eval += 1
            if fc[1] is not None:
                st.q = fc[1].clone()  # the stance's recorded arms = the fist at its pre-contact: the transition brings the arm straight there
            if not self._v2_transition_ok(init, st):
                st.q = None
                if not self._v2_transition_ok(init, st):
                    log.info("%s: push-close stance candidate lateral %+.2f stand-off %+.2f trunk (%.2f, %+.2f) covers q %.2f -> %.2f but no transition from init", self.gp.alias, dlat, dso, trunk[0], trunk[1], q0, q_to)
                    continue
            cov = abs(q_to - q0)
            b_ok = 1.0 if (brace_on and find_brace(st, [wp_at(q0).pre_grasp, wp_at(q0).eef], q0) is not None) else 0.0
            if brace_on and not b_ok:
                # owner 2026-09-20 23:20: the brace is REQUIRED for segment 1 (the flap standing, the tip push lateral - the box slid 5 cm)
                log.info("%s: push-close stance candidate lateral %+.2f stand-off %+.2f trunk (%.2f, %+.2f) covers q %.2f -> %.2f but has no brace: skipped", self.gp.alias, dlat, dso, trunk[0], trunk[1], q0, q_to)
                continue
            score = (b_ok, cov)
            if best1 is None or score > best1[0]:
                best1 = (score, st, fc, q_to, (dlat, dso, trunk))
                log.info("%s: push-close stance candidate lateral %+.2f stand-off %+.2f trunk (%.2f, %+.2f) covers q %.2f -> %.2f%s", self.gp.alias, dlat, dso, trunk[0], trunk[1], q0, q_to,
                         "" if not brace_on else (" with the brace" if b_ok else " (no brace)"))
            if abs(q_to - qg) < 1e-3:
                break  # the whole arc from one stance
        if best1 is None:
            raise Reject("push_unreachable", f"push-close: no stance of {n_eval} evaluated reaches the tip contact at q {q0:.2f} and any arc pose beyond it{' with the brace' if brace_on else ''}")
        _, stance, (r, q_pre), q_to1, shifts = best1
        if stance is not manip:
            stance.extras.update({"second_stance": True, "lateral_shift": float(shifts[0]), "stand_off_shift": float(shifts[1]), "trunk_from": [float(manip.translate), float(manip.tilt)]})
        stance.name = "manipulation"
        wp_at(q0)
        self.record_transition(init, stance)
        stances: List[Stance] = [stance]
        records: List[Any] = []
        braces: List[Any] = []
        segments: List[Dict[str, Any]] = []
        reach: Dict[str, bool] = {"0.0": True}
        q_from, seed, seed_pre = q0, r.q, q_pre
        max_seg = int(self.gen("close_push_max_segments", 3))
        branch_jump = float(self.gen("close_push_branch_jump_rad", 1.5))
        q_to_next = q_to1
        for seg_i in range(max_seg):
            self._check_deadline()
            q_to = q_to_next
            if q_to is None:
                raise Reject("push_unreachable", f"push-close segment {seg_i + 1}: no arc pose beyond q {q_from:.2f} is reachable from {stance.name}")
            wpf = wp_at(q_from)
            contact, pre = wpf.eef, wpf.pre_grasp
            sense = 1.0 if qg >= q_from else -1.0
            q_arc = q_to + sense * over_ok.get(id(stance), 0.0) if abs(q_to - qg) < 1e-6 else q_to
            mid_pose, goal_pose = C.orbit_waypoints(contact, hinge_w, C.quat_from_axis_angle(axis_w, q_arc - q_from), n=2, frame="world")
            radial = wpf.contact - hinge_w
            radial = radial - axis_w * float(th.dot(radial, axis_w))
            r_w = float(th.linalg.norm(radial))
            slot = "push" if seg_i == 0 else f"push{seg_i + 1}"
            rec = ContactRecord(slot, arm, pre, contact, mid_pose, goal_pose, C.unit(wpf.approach), r_w * abs(q_arc - q_from), object=tname, stance=stance.name,
                                extras={"pose_class": "tip_push", "closed_gripper": True, "straight_retreat": True, "push_arm": arm, "retreat_m": retreat_m,
                                        "articulation": {"joint": self.joint_name, "link": self.jinfo.get("child_link"), "q_from": float(q_from), "q_to": float(q_to),
                                                         "holds": True,  # the brake keeps the link at q_to when the fist leaves (program check: no spring-back)
                                                         "q_arc_end": float(q_arc), "repush_cap_rad": float(self.gen("close_push_repush_cap_rad", 0.3)),
                                                         "tolerance": float(self.gen("close_tolerance_rad", 0.05)) if abs(q_to - qg) < 1e-6 else min(0.5 * abs(q_to - q_from), 0.15),
                                                         "hinge_world": [round(float(v), 4) for v in hinge_w.tolist()], "axis_world": [round(float(v), 4) for v in axis_w.tolist()]}})
            records.append(rec)
            seg = {"push": slot, "stance": stance.name, "q_from": float(q_from), "q_to": float(q_to), "arm": arm}
            if brace_on and seg_i == 0:  # owner 23:20: the brace on the first segment only; the later pushes (the flap low, the push downward) run unbraced
                b = find_brace(stance, [pre, contact], q_from)
                if b is not None:
                    b.kind = "brace" if not braces else f"brace{len(braces) + 1}"
                    braces.append(b)
                    seg["brace"] = b.kind
            segments.append(seg)
            # the program check seeds an unbraced push's witnesses from the solutions remembered per world pose; the stance searches solved
            # this tip's pre-contact / contact from every candidate stance, so the chosen stance's own configurations are put back
            for p_, q_ in ((pre, seed_pre), (contact, seed)):
                if q_ is not None:
                    self.ikc.solutions[(arm, self.ikc._pose_key(p_))] = q_.clone()
            log.info("%s: push-close segment %d from %s: q %.3f -> %.3f (of %.3f) at r %.3f m%s", self.gp.alias, seg_i + 1, stance.name, q_from, q_to, qg, r_w,
                     f", braced by the {brace_arm} hand" if seg.get("brace") else "")
            q_from = q_to
            if abs(q_from - qg) < 1e-3:
                break
            if seg_i == max_seg - 1:
                raise Reject("push_unreachable", f"push-close: q {q_from:.2f} still short of {qg:.2f} after {max_seg} segments")
            # ---- the next stance: around the current one, with the link held at q_from (brake): the one covering the most of the rest
            # (``close_push_stance_eval`` candidates WITH coverage are compared; the search runs on until it has them or the combos end)
            best_n, n_eval, n_fc = None, 0, 0
            q_carry = ret_q.get(id(stance))  # where this segment's Retreat leaves the arm: the travel to the next stance carries it
            for _, dlat, dso, trunk, st in self._v2_stance_combos(stance, pose):
                if n_eval >= int(self.gen("close_push_stance_eval", 10)):
                    break
                self._check_deadline()
                if not self.kit.trav_free(st.base_xy) or not self.ikc.stance_collision_free(st):
                    continue
                cam = self.camera_pose_at(st)
                wpn = wp_at(q_from)
                if not TT.in_frame(self, cam, pts, float(self.g("view_margin_m"))) or not self.ikc.eef_in_frame(wpn.eef, cam):
                    continue
                fc = first_contact(st, q_from, q_arm=q_carry)
                if fc is None:
                    n_fc += 1
                    continue
                q_to_c, _ = coverage(st, q_from, fc[0].q)
                if q_to_c is None:
                    log.info("%s: push-close re-stance candidate lateral %+.2f stand-off %+.2f trunk (%.2f, %+.2f): the tip at q %.2f is reachable but no arc pose beyond it (with its retreat)",
                             self.gp.alias, dlat, dso, trunk[0], trunk[1], q_from)
                    continue
                out = fingertip_out(st, q_from, q_to_c)
                if out is not None:
                    log.info("%s: push-close re-stance candidate lateral %+.2f stand-off %+.2f trunk (%.2f, %+.2f) covers q %.2f -> %.2f but the fingertip leaves the frame at "
                             "push.%s[%d] (u %.0f, v %.0f): skipped", self.gp.alias, dlat, dso, trunk[0], trunk[1], q_from, q_to_c, out["key"], out["index"], out["u"], out["v"])
                    continue
                n_eval += 1
                if fc[1] is not None:
                    st.q = fc[1].clone()  # the arm travels to the next pre-contact during the transition
                if not self._v2_transition_ok(stance, st):
                    st.q = None
                    if not self._v2_transition_ok(stance, st):
                        log.info("%s: push-close re-stance candidate lateral %+.2f stand-off %+.2f trunk (%.2f, %+.2f) covers q %.2f -> %.2f but no transition from %s",
                                 self.gp.alias, dlat, dso, trunk[0], trunk[1], q_from, q_to_c, stance.name)
                        continue
                # the pre-contact's joint distance from the carried arm: an elbow flip in between is an out-and-back swing of the hand
                jump = float(th.max(th.abs(fc[1][idx_arm] - q_carry[idx_arm]))) if (fc[1] is not None and q_carry is not None) else 0.0
                direct = 1.0 if jump <= branch_jump else 0.0
                log.info("%s: push-close re-stance candidate lateral %+.2f stand-off %+.2f trunk (%.2f, %+.2f) covers q %.2f -> %.2f, pre-contact %.2f rad from the carried arm",
                         self.gp.alias, dlat, dso, trunk[0], trunk[1], q_from, q_to_c, jump)
                cov = abs(q_to_c - q_from)
                # at equal coverage the stance keeping the departing trunk: the transition then moves the base only
                same_trunk = 1.0 if (abs(float(st.translate) - float(stance.translate)) < 1e-6 and abs(float(st.tilt) - float(stance.tilt)) < 1e-6) else 0.0
                score = (direct, cov, same_trunk)  # the re-stances: the next pre-contact in the carried arm's branch, then coverage; unbraced (owner 23:20)
                if best_n is None or score > best_n[0]:
                    best_n = (score, st, fc, q_to_c, (dlat, dso, trunk))
                if abs(q_to_c - qg) < 1e-3 and direct and same_trunk:
                    break
            if best_n is None:
                raise Reject("push_unreachable", f"push-close: no stance continues the push from q {q_from:.2f} (segment {seg_i + 2}; {n_fc} candidates without the tip contact, {n_eval} with coverage but no transition)")
            _, new_st, (rr, q_pre_n), q_to_next, shifts = best_n
            new_st.name = f"manipulation{len(stances) + 1}"
            new_st.extras.update({"second_stance": True, "lateral_shift": float(shifts[0]), "stand_off_shift": float(shifts[1]), "trunk_from": [float(stance.translate), float(stance.tilt)],
                                  "close_push_segment": seg_i + 2,
                                  # the validator checks this stance (and its arrival transition) with the flap where the brake holds it
                                  "object_joints": {tname: {self.joint_name: float(q_from)}}})
            wp_at(q_from)
            self.record_transition(stance, new_st)  # the link at q_from, held by the brake, hands free; the arm travels to the next pre-contact (new_st.q)
            stances.append(new_st)
            stance, seed, seed_pre = new_st, rr.q, q_pre_n
        # ---- gates + records (each stance's recorded arms hold the fist at ITS segment's pre-contact: checked with the link where that
        # segment starts, not at q_init - the flap at q_init would sweep through a later pre-contact)
        self._set_joint(target, q0)
        self.kit.sync_world()
        TT.whole_body_check_at(self, [init, stances[0]], {tname: pose}, require=True, reason="stance_collision")
        for st in stances[1:]:
            q_at = next((sg["q_from"] for sg in segments if sg["stance"] == st.name), q0)
            self._set_joint(target, float(q_at))
            self.kit.sync_world()
            TT.whole_body_check_at(self, [st], {tname: pose}, require=True, reason="stance_collision")
        self._set_joint(target, q0)
        self.kit.sync_world()
        for st in stances:
            st.arms = [arm] + ([brace_arm] if any(sg.get("brace") for sg in segments if sg["stance"] == st.name) else [])
            st.contacts = [sg["push"] for sg in segments if sg["stance"] == st.name] + [sg["brace"] for sg in segments if sg["stance"] == st.name and sg.get("brace")]
            st.extras["brace"] = "present" if any(sg.get("brace") for sg in segments if sg["stance"] == st.name) else "none"
            self.sidecar.add_stance(st)
        for rec in records + braces:
            self.sidecar.add_contact(rec)
        waypoints = {}
        for kk in [0.0] + fractions:
            wpk = C.handle_waypoints(self.handle, self.jinfo, [q0 + kk * (qg - q0)], pose)[0]
            key = f"{kk:g}"
            waypoints[key] = wpk.to_sidecar()
            reach[key] = True
        self.sidecar.articulation = {"object": tname, "joint": self.joint_name, "joint_type": self._joint_type(), "q_init": q0, "q_goal": qg, "waypoints": waypoints,
                                     "handle_reach": {kk: v for kk, v in reach.items() if kk != "0.0"}, "passive_from": None, "follow_mode": "arm", "handle_arm": arm,
                                     "brace_kind": None, "direction": self.direction, "handle_kind": "tip_push", "handle_width_m": None,
                                     "object_pose": {"position": [round(float(v), 5) for v in pose[0].tolist()], "orientation": [round(float(v), 6) for v in pose[1].tolist()]},
                                     "hinge_world": [round(float(v), 5) for v in hinge_w.tolist()], "axis_world": [round(float(v), 5) for v in axis_w.tolist()],
                                     "close_push": {"segments": segments, "n_stances": len(stances)}}
        log.info("%s: push-close instance: %d segment(s) over %d stance(s): %s", self.gp.alias, len(segments), len(stances),
                 [(sg["stance"], round(sg["q_from"], 2), round(sg["q_to"], 2)) for sg in segments])
        return self._finish_draft(d, target, pose, init, stances[-1], k, reach, kind=None, brace_state="none", summary_extra={"close_push": {"segments": segments}})

    # ------------------------------------------------------------------ two-flap v4: the lever (owner 2026-09-20 15:21)

    def _v4_fist_frame(self, pose: TT.Pose):
        """World frame of the lever: the target flap's hinge axis (along the seam), ``right_h`` = horizontal from the seam toward
        the target's hinge (the fist points that way), ``left_h`` = toward the other flap's hinge, the seam line points."""
        ji = self.jinfo
        op, oq = pose
        axis_w = C.unit(C.quat_rotate(oq, th.tensor(ji["axis_dir_obj"], dtype=th.float32)))
        hinge_w = op + C.quat_rotate(oq, th.tensor(ji["hinge_point_obj"], dtype=th.float32))
        z = th.tensor([0.0, 0.0, 1.0])
        bmin, bmax = th.tensor(ji["link_bbox_min"], dtype=th.float32), th.tensor(ji["link_bbox_max"], dtype=th.float32)
        corners_o = th.stack([th.tensor([x, y, zz]) for x in (float(bmin[0]), float(bmax[0])) for y in (float(bmin[1]), float(bmax[1])) for zz in (float(bmin[2]), float(bmax[2]))])
        corners_w = C.transform_points(pose, corners_o)
        perp_w = C.unit(th.cross(axis_w, z, dim=0))
        if float(th.dot(perp_w[:2], (corners_w.mean(dim=0) - hinge_w)[:2])) < 0:
            perp_w = -perp_w  # from the target's hinge toward its free edge (the seam)
        r_edge = float(((corners_w - hinge_w) @ perp_w).max())
        ax = (corners_w - hinge_w) @ axis_w
        side = 1.0 if float(th.dot(axis_w, C.quat_rotate(oq, self.facing_local))) >= 0 else -1.0  # +1: the robot at the +axis end
        a_near, a_far = (float(ax.max()), float(ax.min())) if side > 0 else (float(ax.min()), float(ax.max()))
        z_mid = 0.5 * (float(corners_w[:, 2].min()) + float(corners_w[:, 2].max()))
        return {"axis": axis_w, "hinge": hinge_w, "left_h": perp_w, "right_h": -perp_w, "r_edge": r_edge, "a_near": a_near, "a_far": a_far, "z_mid": z_mid, "side": side}

    def _sample_with_v4(self, rng, d: Draft, init: Stance, manip: Stance, k: int, target, support, pose: TT.Pose, cam_init) -> Draft:
        """Owner sequence 2026-09-20 15:21 (two-flap carton): the LEFT arm comes over the box with the forearm about level and the
        closed gripper pointing right; (1) it presses the left flap down near the seam (``contacts.push_prep``, held: the seam
        opens); (2) it slides right into the opening under the right flap's free edge and lifts it a little
        (``contacts.push_lift``: the press's end pose is its pre-contact, the slide is the approach leg, the arc orbits the
        right hinge to q_ajar; held); (3) the right hand pinches the raised edge near the robot (handle B, ``articulation``
        from q_ajar); (4) the left arm slides back out to the left and hovers above the box; (5) the right hand opens the flap
        to the goal (a fist push from inside when its reach ends, stage 2). Both flaps' springs / brake as v2."""
        from ispatialgym.sampling.v2.stances import ContactRecord

        tname, main_arm, lever_arm = self.tname, self.handle_arm, self.brace_arm
        other = str(self.gen("prep_push_joint", ""))
        if other not in target.joints or other == self.joint_name:
            raise Reject("spec", f"two_flap_v4 needs generation.prep_push_joint = the other flap ({list(target.joints)})")
        angle = float(self.gen("prep_push_angle_rad", 0.3))
        q0o = float(self.all_jinfo.get(other, {}).get("closed_q", 0.0))
        qa, q_goal = float(self.q_ajar), float(self.q_goal)
        if not (self.q_init < qa < q_goal):
            raise Reject("spec", f"two_flap_v4: q_ajar {qa:.3f} must lie between q_init {self.q_init:.3f} and q_goal {q_goal:.3f}")
        fr = self._v4_fist_frame(pose)
        axis_w, hinge_w, left_h, right_h, z_mid = fr["axis"], fr["hinge"], fr["left_h"], fr["right_h"], fr["z_mid"]
        z = th.tensor([0.0, 0.0, 1.0])
        jo = self.all_jinfo[other]
        hinge_o_w = pose[0] + C.quat_rotate(pose[1], th.tensor(jo["hinge_point_obj"], dtype=th.float32))
        axis_o_w = C.unit(C.quat_rotate(pose[1], th.tensor(jo["axis_dir_obj"], dtype=th.float32)))
        # the lever's axial position along the seam
        a_opts = {"near": fr["a_near"] + fr["side"] * -float(self.gen("v4_axial_inset_m", 0.06)), "far": fr["a_far"] + fr["side"] * float(self.gen("v4_axial_inset_m", 0.06)),
                  "mid": 0.5 * (fr["a_near"] + fr["a_far"])}
        axials = self.gen("v4_axial", "mid")
        axials = [str(a) for a in (axials if isinstance(axials, (list, tuple)) else [axials])]  # several depths along the seam are tried in order (0920-182140: the
        # lever at mid-depth and the right hand at the flap's near edge were in each other's way; the far half keeps the left arm off the near edge)
        seam_pts = []
        for a_name in axials:
            sp = hinge_w + left_h * fr["r_edge"] + axis_w * a_opts[a_name]  # the target flap's free edge at the lever's depth
            seam_pts.append((a_name, th.tensor([float(sp[0]), float(sp[1]), z_mid])))
        d_press = float(self.gen("v4_press_seam_offset_m", 0.05))
        d_in = float(self.gen("v4_lift_insert_m", 0.025))
        t_flap = float(self.gen("v4_flap_thickness_m", 0.0008))
        # fist variants (0920-154026: the level fist with the pads closing along the seam had no IK anywhere): flat (pads closing along
        # the seam, ~2.4 cm tall near the tips) or on edge (pads above / below, the roll of the pinch that passed), pitched down a little
        # 0920-162644: the level fist pointing straight right had no IK at the press pre-pose from any stance, flat or on edge, pitched
        # 0-25 - like the square-on pinch before it; only yawed + pitched hands passed. ``v4_fist_yaws_deg`` turns the approach from
        # "right" toward "forward" (away from the robot along the seam); the FINGERTIP is the anchor of both the press point and the
        # insertion under the edge, so the yaw does not change where the fist meets the flaps
        fwd_w = -fr["side"] * axis_w  # along the seam, away from the robot
        variants = []
        for a_name, seam_pt in seam_pts:
          for yaw_deg in [float(v) for v in self.gen("v4_fist_yaws_deg", [0.0, 30.0, 50.0])]:
            ya = math.radians(yaw_deg)
            a_h = C.unit(right_h * math.cos(ya) + fwd_w * math.sin(ya))
            for pitch_deg in [float(v) for v in self.gen("v4_fist_pitches_deg", [0.0, 15.0, 25.0])]:
                pa = math.radians(pitch_deg)
                a_w = C.unit(a_h * math.cos(pa) - z * math.sin(pa))
                # the wrist camera's protrusion sits on the eef's -x / -c side (``iors_small.HAND_SPHERES_EEF``: x -5.6 cm, c 0 ... -5.7 cm,
                # 7.6 cm behind the tips) and points UP, away from the box (owner: "wrist camera protrusion to the top"): flat (pads closing
                # along the seam) eef x = c x a = -z, on edge c = -z. With eef x = +z / c = +z it hung below the fist and rode on the pressed
                # flap during the slide: the flap hit its stop and the box was dragged 5-12 mm (carton lever 0926-073750, every lever run)
                c_flat = C.unit(th.cross(z, a_h, dim=0))
                h_flat, h_edge = float(self.gen("v4_fist_half_height_m", 0.016)), float(self.gen("v4_fist_half_height_edge_m", 0.022))
                for roll_name, c_w, h_half in (("flat", c_flat, h_flat), ("edge", -z, h_edge)):
                    q_fist = C.eef_quat(a_w, c_w)
                    # (1) the press: the fingertip region on the left flap d_press left of the seam, the fist h_half above the top face
                    tip_press = seam_pt + left_h * d_press + z * (0.5 * t_flap + 0.002 + h_half)
                    p_press = tip_press - a_w * C.HANDLE_EEF_OFFSET
                    press_pose = (p_press, q_fist)
                    press_pre = (p_press + z * float(self.gen("prep_push_pre_dist_m", 0.06)), q_fist)
                    press_mid, press_goal = C.orbit_waypoints(press_pose, hinge_o_w, C.quat_from_axis_angle(axis_o_w, -angle), n=2, frame="world")
                    # (2) the lift: the fingertips d_in under the right flap's edge, the fist's top just under the flap; the arc about the right hinge to q_ajar
                    clear = float(self.gen("v4_lift_clearance_m", 0.012))  # the finger's top this far under the flap plane where it passes the free edge
                    # (0920-203004: at 3-5 mm the fist caught the edge and shoved the box 4 cm). A PITCHED fist rises behind its tips (0920-204518:
                    # flat/pitch40 - the finger body 2-3 cm behind the tips stood above the plane): only the tips go under the edge, the finger's
                    # top at the edge = tip + d_in tan(pitch) + h_half / cos(pitch)
                    cp = max(math.cos(pa), 0.3)
                    tip_lift = seam_pt + right_h * d_in - z * (0.5 * t_flap + clear + h_half / cp + d_in * math.tan(pa))
                    p_lift = tip_lift - a_w * C.HANDLE_EEF_OFFSET
                    # the slide runs v4_slide_lift_m above the press's end height when that is still under the flap plane: a fist pressed hard
                    # into the other flap dragged the box 3-4 cm sideways while sliding (0920-193159 / 201652)
                    z_slide = min(float(p_lift[2]), float(press_goal[0][2]) + float(self.gen("v4_slide_lift_m", 0.01)))
                    p_lift = th.tensor([float(p_lift[0]), float(p_lift[1]), z_slide])
                    lift_init = (p_lift, q_fist)
                    if str(self.gen("v4_lift_path", "vertical")) == "arc":
                        lift_mid, lift_goal = C.orbit_waypoints(lift_init, hinge_w, C.quat_from_axis_angle(axis_w, qa - self.q_init), n=2, frame="world")
                    else:
                        # a straight vertical rise (0920-193159: the arc about the hinge drove the fist's tip along the flap's underside
                        # toward the hinge and shoved the box 3 cm sideways); the flap at the tip's radius rises r sin(q_ajar), plus a little
                        r_tip = fr["r_edge"] - d_in
                        dz = clear + r_tip * math.sin(qa - self.q_init) + float(self.gen("v4_lift_extra_m", 0.008))  # the finger's top meets the flap after `clear`
                        lift_mid = (p_lift + z * (0.5 * dz), q_fist)
                        lift_goal = (p_lift + z * dz, q_fist)
                    variants.append({"name": f"{a_name}/{roll_name}/yaw{yaw_deg:g}/pitch{pitch_deg:g}", "axial": a_name, "q_fist": q_fist, "press_pose": press_pose, "press_pre": press_pre,
                                     "press_mid": press_mid, "press_goal": press_goal, "lift_init": lift_init, "lift_mid": lift_mid, "lift_goal": lift_goal, "p_press": p_press, "p_lift": p_lift})
        pts = C.transform_points(pose, self._hull_q_init)

        def set_flaps(q_t, q_o):
            self._set_named_joint(target, other, q_o)
            self._set_joint(target, q_t)
            self.kit.sync_world()

        class _FlapsOnly:
            """The two flap links of the carton as an ignorable 'object' (``_Ignoring`` walks ``.links``): the lever's legs are
            checked with the BOX BODY in the collision world - 0920-173005: the press ended 0.8-1.4 cm high and the slide climbed
            3 cm because the gripper's palm (6 x 11 x 8 cm behind the fingers, r1pro.urdf) sat on the box's left wall / the pressed
            flap, which the whole-object ignore had hidden from every check."""
            def __init__(self, obj, names):
                self.name = f"{getattr(obj, 'name', 'target')}:flaps"
                self.links = {n: l for n, l in getattr(obj, "links", {}).items() if n in names}

        # the two flap links plus the object's META links (a carton's ``meta__base_link_fillable_*`` volume fills the box interior and is
        # a collision mesh in the cuRobo world: a fist entering the opening, or hovering over it, reads 'collides with the box body')
        meta_links = {n for n in getattr(target, "links", {}) if str(n).startswith("meta__")}
        flaps = _FlapsOnly(target, ({self.jinfo.get("child_link"), jo.get("child_link")} - {None}) | meta_links)
        if not flaps.links:
            flaps = target

        flaps_locked: List[Any] = []  # the braced wall piece while the brace arm is locked (filled per stance)

        def _ign():
            return [flaps] + list(flaps_locked)

        def track(q_from, p_from, p_to, label: str):
            """0920-170545: the lift's init solved 2.6 rad away from the press's end (another IK branch, 7.5 cm apart) - the oracle's
            Cartesian tracker could not follow the slide and the fist ended 3 cm above the flap. Every lever leg is tracked from the
            previous configuration (``track_segment``: one branch, bounded joint steps - what the oracle's ``arm_along`` executes),
            the flaps ignored along the way (the fist is on / under them by design), the table too at the ends."""
            q, why = self.ikc.track_segment(q_from, p_from, p_to, lever_arm, ignore_mid=_ign(), ignore_end=[support] + _ign(), ignore_start=[support] + _ign())
            if q is None:
                return None, f"{label}: {str(why)[:100]}"
            return q, None

        def try_lever(st: Stance, v: Dict[str, Any], seed_posture: Optional[str] = None, q_lock=None):
            """The lever chain from ``st`` for fist variant ``v``; ``seed_posture`` (untucked / ready / tucked) seeds the press pre-pose IK
            so that different arm branches (elbow out vs dropped in front of the chest) are tried - the branch decides whether the
            other hand can still reach its pinch beside the lever (0920-184337: the final two-hand pose was free, the pinch's
            pre-grasp / approach ran into the lever arm)."""
            press_pre, press_pose, press_mid, press_goal = v["press_pre"], v["press_pose"], v["press_mid"], v["press_goal"]
            lift_init, lift_mid, lift_goal = v["lift_init"], v["lift_mid"], v["lift_goal"]
            set_flaps(self.q_init, q0o)
            q_seed = None
            if seed_posture is not None or q_lock is not None:
                # q_lock (owner 21:20): the other arm braced on the box's face - every lever leg is checked with it locked there
                q_seed = (q_lock if q_lock is not None else self.ikc.q_locked(st, fingers=C.FINGER_CLOSED)).clone()
                q_seed[self.kit.robot.arm_control_idx[lever_arm]] = self.kit.arm_q(seed_posture or "untucked", lever_arm)
            with self.ikc.ignoring(list(flaps_locked)):
                r1 = self.ikc.exact([press_pre, press_pose], st, lever_arm, ignore_for_last=[support, flaps], cache=False, fingers=C.FINGER_CLOSED, approached=[flaps], q_seed=q_seed)
            if not r1.ok:
                # 0921-005643: the solver (its own clearance) finds nothing beside the box body, yet the solution found with the target
                # removed is collision-free with the body back in at margin 0 - take that witness and TRACK the approach to the contact
                # (margin-0 checks, one branch), as the oracle executes it
                r_w = self.ikc.exact([press_pre], st, lever_arm, ignore_for_last=[support, target], cache=False, fingers=C.FINGER_CLOSED, q_seed=q_seed)
                if r_w.ok:
                    with self.ikc.ignoring([support, flaps] + list(flaps_locked)):
                        free_w = bool(self.kit.collision_free(r_w.q, margin=0.0)[0])
                    if free_w:
                        q_c, why_c = self.ikc.track_segment(r_w.q, press_pre, press_pose, lever_arm, ignore_mid=_ign(), ignore_end=[support] + _ign(), ignore_start=[support] + _ign())
                        if q_c is not None:
                            from ispatialgym.sampling.v2.ik import IKResult

                            self.ikc.solutions[(lever_arm, self.ikc._pose_key(press_pre))] = r_w.q.clone()
                            self.ikc.solutions[(lever_arm, self.ikc._pose_key(press_pose))] = q_c.clone()
                            r1 = IKResult(True, q_c, "pre from the box-free solve (free with the body in), approach tracked", 2)
            if not r1.ok:
                # is it the box (palm / camera against the body) or the reach? the pre-pose alone with the whole target removed
                r_free = self.ikc.exact([press_pre], st, lever_arm, ignore_for_last=[support, target], cache=False, fingers=C.FINGER_CLOSED, q_seed=q_seed)
                why_free = "unreachable even without the box"
                if not r_free.ok and q_lock is not None:
                    r_nolock = self.ikc.exact([press_pre], st, lever_arm, ignore_for_last=[support, target], cache=False, fingers=C.FINGER_CLOSED)
                    why_free = "unreachable with the other arm braced (reachable with it at rest)" if r_nolock.ok else "unreachable even without the box and the brace"
                if r_free.ok:
                    try:
                        pairs = []
                        with self.ikc.ignoring([support, flaps]):
                            free_here = bool(self.kit.collision_free(r_free.q, margin=0.0)[0])
                        if free_here:
                            why_free = "reachable without the box; its own solution is free with the box body in - another branch / margin issue"
                        else:
                            with self.ikc.ignoring([support, flaps] + list(flaps_locked)):
                                pairs = self.kit.backend.world_collision_offenders(r_free.q, limit=3) if hasattr(self.kit.backend, "world_collision_offenders") else []
                                free_locked = bool(self.kit.collision_free(r_free.q, margin=0.0)[0])
                            why_free = f"reachable without the box: with the body in (braced wall ignored: {bool(flaps_locked)}), {'free - IK branch/margin' if free_locked else 'offenders ' + str(pairs)}"
                    except Exception as e:  # noqa: BLE001
                        why_free = f"reachable without the box (blame raised {e!r})"
                return f"press pre / contact: {str(r1.reason)[:100]} ({why_free})"
            q_prev, p_prev = r1.q, press_pose
            for frac, wp in ((0.5, press_mid), (1.0, press_goal)):
                set_flaps(self.q_init, q0o - frac * angle)
                q_prev, err = track(q_prev, p_prev, wp, f"press arc at {frac:g}")
                if err:
                    return err
                p_prev = wp
            set_flaps(self.q_init, q0o - angle)
            q_prev, err = track(q_prev, p_prev, lift_init, "slide into the gap")
            if err:
                return err
            p_prev = lift_init
            for frac, wp in ((0.5, lift_mid), (1.0, lift_goal)):
                set_flaps(self.q_init + frac * (qa - self.q_init), q0o)  # the other flap springs back once the fist left it
                q_prev, err = track(q_prev, p_prev, wp, f"lift arc at {frac:g}")
                if err:
                    return err
                p_prev = wp
            q_lift_end = q_prev
            # the lever's retreat (owner 21:20): straight UP off the pressed flap first, then to the left away from the seam (a palm
            # dragging the left flap would pull the box against the other hand's grasp)
            up = (lift_goal[0] + z * float(self.gen("v4_retreat_lift_m", 0.03)), lift_goal[1])
            q_up, err = track(q_lift_end, lift_goal, up, "lever retreat (up)")
            if err:
                return err
            back = (up[0] + left_h * float(self.gen("v4_retreat_m", 0.08)), up[1])
            _, err = track(q_up, up, back, "lever retreat (left)")
            if err:
                return err
            return {"q_lift_end": q_lift_end, "q_press": r1.q}

        def pinch_with_lever(st: Stance, q_lift_end):
            """The main hand's pinch B at q_ajar with the lever arm LOCKED where the tracked lever chain leaves it (q_lift_end: the
            configuration the oracle holds), not re-solved from scratch beside it (0920-174724); ``(h_b, wp_b, r_b)`` or None.
            Fallback (0920-191311: the pinch solved beside the resting arm, and its pre / grasp witnesses were collision-free with the
            lever arm put in, yet the IK beside the lever found nothing - the two arms sit inside the solver's own clearance): the
            resting solution is taken when both witnesses are free with the lever arm in (margin ``v4_arms_margin_m``) and the
            approach tracks between them in that context."""
            set_flaps(qa, q0o)
            q_ctx = q_lift_end.clone()
            q_ctx[self.kit.robot.gripper_control_idx[main_arm]] = float(self.grasp_fingers)  # the pinching hand pre-shaped open; the lever's fist stays closed
            h_b, wp_b, r_b = self._best_pinch(getattr(self, "handle_variants", None) or self.handle, qa, st, main_arm, target, support, pose, q_seed=q_ctx)
            if h_b is not None:
                return h_b, wp_b, r_b
            idx_l, idx_r = self.kit.robot.arm_control_idx[lever_arm], self.kit.robot.arm_control_idx[main_arm]
            margin = float(self.gen("v4_arms_margin_m", 0.005))
            h_free, wp_free, r_free = self._best_pinch(getattr(self, "handle_variants", None) or self.handle, qa, st, main_arm, target, support, pose)
            if h_free is None:
                return None
            for h_c, wp_c, r_c, m_c in sorted(getattr(self, "_pinch_passing", []) or [], key=lambda c: (abs(float(c[0].get("yaw_deg", 0.0))), float(c[0].get("pitch_deg", 0.0)))):
                q_pre = self.ikc.solutions.get((main_arm, self.ikc._pose_key(wp_c.pre_grasp)))
                if q_pre is None or r_c.q is None:
                    continue
                q_pre_c, q_g_c = q_pre.clone(), r_c.q.clone()
                q_pre_c[idx_l], q_g_c[idx_l] = q_lift_end[idx_l], q_lift_end[idx_l]
                with self.ikc.ignoring([support, target]):
                    ok_pre = bool(self.kit.collision_free(q_pre_c, margin=margin)[0])
                with self.ikc.ignoring([support, target]):
                    ok_g = bool(self.kit.collision_free(q_g_c, margin=margin)[0])
                if not (ok_pre and ok_g):
                    continue
                q_end, why = self.ikc.track_segment(q_pre_c, wp_c.pre_grasp, wp_c.eef, main_arm, ignore_mid=[target], ignore_end=[support, target])
                if q_end is None:
                    continue
                log.info("%s: B pinch beside the lever taken from the resting-arm solution (site %s yaw %s pitch %s): pre / grasp free with the lever arm in at %.0f mm, approach tracks",
                         self.gp.alias, h_c.get("site"), h_c.get("yaw_deg"), h_c.get("pitch_deg"), 1000 * margin)
                from ispatialgym.sampling.v2.ik import IKResult

                # the witnesses of the pre-grasp / grasp in the two-arm context (the program check / oracle hints)
                self.ikc.solutions[(main_arm, self.ikc._pose_key(wp_c.pre_grasp))] = q_pre_c.clone()
                self.ikc.solutions[(main_arm, self.ikc._pose_key(wp_c.eef))] = q_end.clone()
                return h_c, wp_c, IKResult(True, q_end, "beside the lever (resting-arm solution verified)", 2)
            return None

        brace_on = bool(self.gen("v4_brace", True))

        def find_lever_brace(st: Stance):
            """Owner 2026-09-20 21:20: the main hand's OPEN gripper on the box's face on the target's hinge side (the box slid 3-4 cm
            that way during the slide + lift), pushing toward the seam, at ``v4_brace_height_frac`` of the box height, near the box's
            near end (``v4_brace_corner_inset_m`` from the corner). Returns ``(BraceRecord, q28 with the arm at the contact)`` or
            ``(None, reason)``."""
            hull_w = C.transform_points(pose, self.hull)
            n_face = right_h  # the face's outward normal: away from the seam past the target's hinge
            proj = hull_w @ n_face
            face_d = float(proj.max())
            centre = hull_w.mean(dim=0)
            z_lo, z_hi = float(hull_w[:, 2].min()), float(hinge_w[2])  # the box body: floor to the flap plane (the hull's top is a standing flap elsewhere)
            z_b = z_lo + float(self.gen("v4_brace_height_frac", 0.6)) * (z_hi - z_lo)
            along = axis_w  # along the face = along the seam
            pa_ = hull_w @ along
            base_xy = th.tensor([float(st.base_pose2d[0]), float(st.base_pose2d[1]), 0.0])
            c_along = float(th.dot(centre, along))
            ends = [float(pa_.min()), float(pa_.max())]
            end_pts = [centre + along * (e - c_along) for e in ends]
            near_i = 0 if float(th.linalg.norm((end_pts[0] - base_xy)[:2])) <= float(th.linalg.norm((end_pts[1] - base_xy)[:2])) else 1
            inset = float(self.gen("v4_brace_corner_inset_m", 0.05))
            a_pos = ends[near_i] + (inset if near_i == 0 else -inset)
            a_pos = min(max(a_pos, ends[0] + 0.01), ends[1] - 0.01)
            p_face = centre + along * (a_pos - c_along)
            p_face = p_face + n_face * (face_d - float(th.dot(p_face, n_face)))
            kind = str(self.gen("v4_brace_kind", "blade_down"))
            if kind == "blade_down":
                # owner 2026-09-20 23:20: a TOP-DOWN blade - approach -z, fingers closed, the finger's side flat against the box's
                # hinge-side face just below the rim (fingertips v4_brace_tip_below_rim_m under the rim top), at the box's near corner,
                # pushing toward the seam. A hand outside the hinge line is never swept by the flap (it turns inward about that rim).
                # The horizontal face stop (0920-214239 ... 230533) was a sideways cross-body reach with no IK at 0.73-0.89 m depth.
                rim_z = float(hinge_w[2])  # the target's hinge sits on this wall's top
                z_tip = rim_z - float(self.gen("v4_brace_tip_below_rim_m", 0.025))
                half_pad = float(TT.finger_half_thickness(0.0) or 0.0084)  # the closed fingers' half thickness near the tips (FINGER_PROFILE)
                half_edge = float(self.gen("v4_fist_half_height_edge_m", 0.022))  # the closed fingers' half WIDTH (the plate's long side)
                clear = float(self.gen("v4_brace_clearance_m", 0.002))
                fwd_b = -fr["side"] * axis_w  # along the face, away from the robot
                # owner 2026-09-22 11:27: the blade EDGE-ON first - closing axis along the face, the fingers' narrow side toward the wall -
                # so that the hand goes from the brace to its pinch (approach forward-down, closing vertical) by a pure pitch about the lateral
                # axis; the pad-on blade (closing across the face) needed a 90 deg roll as well, and the planner swung the hand 70 cm out to
                # the right and out of the frame on the way (0921-232806 ... 0922-152127). Either sign of the closing axis = the wrist yawed
                # 180 deg (the fingers are symmetric); the first that solves wins.
                cands = []  # (pose, half extent along the face normal, closing class)
                for pdeg in [float(v) for v in self.gen("v4_brace_pitches_deg", [0.0, 30.0, 45.0, 60.0])]:
                    # the blade pitched forward-down: the fingertips stay v4_brace_tip_below_rim_m under the rim, the hand leans away from the robot
                    a_b = C.unit(-z * math.cos(math.radians(pdeg)) + fwd_b * math.sin(math.radians(pdeg)))
                    for c_b, half_b, cls in ((fwd_b, half_edge, "edge_on"), (-fwd_b, half_edge, "edge_on"), (n_face, half_pad, "pad_on"), (-n_face, half_pad, "pad_on")):
                        if cls == "edge_on" and not bool(self.gen("v4_brace_edge_on", True)):
                            continue
                        if abs(float(a_b @ c_b)) > 1e-6:
                            # the closing axis must be perpendicular to the approach: the along-the-face axis is pitched with the hand
                            c_b = C.unit(c_b - a_b * float(a_b @ c_b))
                        origin = p_face + n_face * (half_b + clear)
                        tip = th.tensor([float(origin[0]), float(origin[1]), float(z_tip)])
                        o_b = tip - a_b * float(C.FINGERTIP_LENGTH)
                        cands.append(((o_b, C.eef_quat(a_b, c_b)), half_b, cls))
                bk = "side_blade"
                set_flaps(self.q_init, q0o)
                r, contact, pre, half, cls_used = None, None, None, half_pad, "pad_on"
                for c_try, half_b, cls in cands:
                    pre_try = C.pre_pose(c_try, float(self.gen("v4_brace_pre_m", 0.08)))  # straight above
                    r_try = self.ikc.exact([pre_try, c_try], st, main_arm, ignore_for_last=[support, target], cache=False, fingers=C.FINGER_CLOSED)
                    r, contact, pre, half, cls_used = r_try, c_try, pre_try, half_b, cls
                    if r_try.ok:
                        break
                extras = {"blade_half_thickness_m": half, "stop_clearance_m": clear, "pose_class": "blade_down", "blade_closing": cls_used, "fingertip_z": float(z_tip),
                          # owner 2026-09-22 11:27: the blade lifts straight off along its own approach (no radial swing out of the frame) and only a little
                          "straight_retreat": True, "retreat_m": float(self.gen("v4_brace_retreat_m", 0.05)),
                          "note": "two-flap v4: the pinching hand's closed blade hangs beside the hinge-side face below the rim and holds the box against the lever's shove (owner 2026-09-20 23:20)"}
                if not r.ok:
                    b = st.world_to_base(pre[0][:2])
                    r_nobox = self.ikc.exact([pre], st, main_arm, ignore_for_last=[support, target], cache=False, fingers=C.FINGER_CLOSED)
                    r_contact_nobox = self.ikc.exact([contact], st, main_arm, ignore_for_last=[support, target], cache=False, fingers=C.FINGER_CLOSED)
                    why = ("pre reachable with the box ignored: the pre-pose meets the box" if r_nobox.ok else
                           ("pre unreachable; the contact alone " + ("solves" if r_contact_nobox.ok else "does not solve either") + " with the box ignored"))
                    return None, f"brace ({main_arm}, {kind}, {'near' if near_i == 0 else 'far'} end of the hinge-side face, base depth {float(b[0]):.2f} lateral {float(b[1]):+.2f} z {float(pre[0][2]):.2f}, both wrist yaws): {str(r.reason)[:60]} [{why}]"
                rec = TT.BraceRecord("brace", main_arm, pre, contact, direction=-n_face, brace_kind=bk, object=tname, stance="manipulation", extras=extras)
                # the PROBE configuration (the blade 1.2 cm into the face; owner 2026-09-22 11:27, run 0922-132940: the blade 2 mm off the wall
                # had no world offender in one IK branch, the wall stayed in the world and every yawed fist variant collided with it)
                probe = (contact[0] - n_face * 0.012, contact[1])
                rp = self.ikc.exact([probe], st, main_arm, ignore_for_last=[support, target], q_seed=r.q, cache=False, fingers=C.FINGER_CLOSED)
                return rec, (r.q.clone(), rp.q.clone() if rp.ok else None)
            else:
                p_face = th.tensor([float(p_face[0]), float(p_face[1]), z_b])
                contact = C.contact_eef(p_face, -n_face, along)
                pre = C.pre_pose(contact, float(self.gen("v4_brace_pre_m", 0.06)))
                opening = float(self.gen("v4_brace_opening_m", 0.04))
                extras = {"stop_clearance_m": 0.0, "pose_class": "wide_stop", "fingers_opening_m": opening, "stop_width_m": 2 * opening + 0.02,
                          "note": "two-flap v4: the pinching hand, open, holds the box against the lever's shove (owner 2026-09-20 21:20)"}
                bk = "opposite_face"
            set_flaps(self.q_init, q0o)
            r = self.ikc.exact([pre, contact], st, main_arm, ignore_for_last=[support, target], cache=False, fingers=C.FINGER_CLOSED)
            if not r.ok:
                b = st.world_to_base(pre[0][:2])
                return None, f"brace ({main_arm}, {kind}, {'near' if near_i == 0 else 'far'} end of the hinge-side face, base depth {float(b[0]):.2f} lateral {float(b[1]):+.2f} z {float(pre[0][2]):.2f}): {str(r.reason)[:80]}"
            rec = TT.BraceRecord("brace", main_arm, pre, contact, direction=-n_face, brace_kind=bk, object=tname, stance="manipulation", extras=extras)
            # a PROBE configuration with the blade pushed 1.2 cm into the face (owner 2026-09-22 11:27, run 0922-132940: the braced blade 2 mm
            # off the wall had no world offender in one IK branch, the wall stayed in the world and every yawed fist variant collided with
            # it) - its offenders name the braced wall piece(s) reliably
            probe = (contact[0] - n_face * 0.012, contact[1])
            rp = self.ikc.exact([probe], st, main_arm, ignore_for_last=[support, target], q_seed=r.q, cache=False, fingers=C.FINGER_CLOSED)
            return rec, (r.q.clone(), rp.q.clone() if rp.ok else None)

        def try_stance(st: Stance):
            lever_out, lever_v, whys, pinch = None, None, [], None
            n_lever_ok, last_lever = 0, None
            brace_rec, q_lock = None, None
            q_probe = None
            if brace_on:
                brace_rec, q_lock = find_lever_brace(st)
                if brace_rec is None:
                    return str(q_lock)
                q_lock, q_probe = q_lock
                # the wall the braced blade touches (the base link's piece farthest along the face normal) leaves the world for the lever
                # legs while the brace arm is locked there: its padded spheres sit inside that wall
                try:
                    root = getattr(getattr(target, "root_link", None), "body_name", None) or "base_link"
                    # the pieces the braced blade's padded spheres sit in: the world meshes whose removal clears the brace configuration
                    # (the collision prims' own positions are in a normalised space and cannot be projected onto the face normal)
                    flaps_locked.clear()
                    # owner 2026-09-22 11:27 (0922-132940 / 140026 / 143026): the collision probe named the flap, not the wall, in three runs
                    # (the blade's padded spheres 2 mm off a thin cardboard wall are free in cuRobo's world in most IK branches) - the wall is
                    # named by GEOMETRY first: the base link's collision piece whose world centroid lies farthest along the face normal
                    # (per-mesh points in the link frame, ``points_in_parent_frame``, through the link's world transform)
                    names: set = set()
                    try:
                        self.world.sync_render_transforms()  # the link's render-side transform follows the placement (sim_utils.local_hull)
                        link = getattr(target, "links", {}).get(root)
                        best_n, best_d = None, None
                        for mn, cm in (getattr(link, "collision_meshes", {}) or {}).items():
                            pts = getattr(cm, "points_in_parent_frame", None)
                            if pts is None or len(pts) == 0:
                                continue
                            w = link.transform_local_points_to_world(pts)
                            d = float((w.mean(dim=0) @ right_h).item())
                            if best_d is None or d > best_d:
                                best_n, best_d = mn, d
                        if best_n is not None:
                            names = {best_n}
                            if not getattr(self, "_v4_pieces_logged", False):
                                # once per instance: where every body piece sits in the lever frame (hinge-normal / along-the-seam / z; m)
                                self._v4_pieces_logged = True
                                rows = []
                                for mn, cm in (getattr(link, "collision_meshes", {}) or {}).items():
                                    pts = getattr(cm, "points_in_parent_frame", None)
                                    if pts is None or len(pts) == 0:
                                        continue
                                    w = link.transform_local_points_to_world(pts) - hinge_w
                                    c = w.mean(dim=0)
                                    rows.append(f"{mn.rsplit('__', 1)[-1]}: n {float(c @ right_h):+.3f} a {float(c @ axis_w):+.3f} z {float(c[2]):+.3f} (z {float(w[:, 2].min()):+.3f}..{float(w[:, 2].max()):+.3f})")
                                log.info("%s: box body pieces in the lever frame: %s", self.gp.alias, "; ".join(rows))
                    except Exception as e:  # noqa: BLE001
                        log.info("%s: the braced wall could not be named by geometry (%r): falling back to the collision probe", self.gp.alias, e)
                    offenders = []
                    if not names:
                        with self.ikc.ignoring([support, flaps]):
                            for qq in ([q_probe] if q_probe is not None else []) + [q_lock]:
                                offenders = list(self.kit.backend.world_collision_offenders(qq, limit=4)) if hasattr(self.kit.backend, "world_collision_offenders") else []
                                if any(f"/{root}/" in str(o) for o in offenders):
                                    break
                        names = {str(o).rsplit("/", 1)[-1] for o in offenders if f"/{root}/" in str(o)}
                    if names:
                        wall = _MeshSubset(target, root, lambda n, cm, ns=names: n in ns)
                        if wall.links:
                            flaps_locked.append(wall)
                            log.info("%s: lever brace sits in %s - those pieces leave the world for the lever legs", self.gp.alias, sorted(names))
                        else:
                            log.info("%s: the braced wall %s matches no collision mesh of %s (%s): the lever legs are checked with the whole box body", self.gp.alias, sorted(names), root,
                                     sorted(getattr(getattr(target, "links", {}).get(root), "collision_meshes", {}).keys()))
                    elif offenders:
                        log.info("%s: the braced blade's offenders are not the box body (%s): the lever legs are checked with the whole box body", self.gp.alias, offenders)
                    else:
                        with self.ikc.ignoring([support, flaps]):
                            hit_lock = self.kit.collision_free(q_lock, margin=0.0)[0] if hasattr(self.kit, "collision_free") else None
                            hit_probe = self.kit.collision_free(q_probe, margin=0.0)[0] if (q_probe is not None and hasattr(self.kit, "collision_free")) else None
                        log.info("%s: the braced blade has no world offender (backend has offenders: %s; probe solved: %s; free at the brace: %s, free at the probe: %s): the lever legs are checked with the whole box body",
                                 self.gp.alias, hasattr(self.kit.backend, "world_collision_offenders"), q_probe is not None, hit_lock, hit_probe)
                except Exception as e:  # noqa: BLE001
                    log.info("%s: the braced wall could not be resolved (%r): the lever legs are checked with the whole box body", self.gp.alias, e)
            for v in variants:
                for seed_posture in [None] + [str(x) for x in (self.gen("v4_lever_seed_postures", ["ready", "tucked"]) or [])]:
                    o = try_lever(st, v, seed_posture, q_lock=q_lock)
                    if not isinstance(o, dict):
                        if seed_posture is None:
                            whys.append(f"{v['name']}: {o}")
                        else:
                            whys.append(f"{v['name']} (seed {seed_posture}): {str(o)[:160]}")
                        # the pre-pose itself fails from this stance: other seeds solve the same pose (they only change the branch) - UNLESS
                        # the pose is reachable and only the found branch meets the box body (owner 2026-09-22 11:27, 0922-144234: yaw30
                        # solved into a branch against collision__2 where an earlier run's branch was free): another seed may find the free one
                        if "reachable without the box" not in str(o):
                            break
                        continue
                    n_lever_ok += 1
                    last_lever = o["q_lift_end"]
                    pinch = pinch_with_lever(st, o["q_lift_end"])
                    if pinch is not None:
                        lever_out, lever_v = o, v
                        if whys:
                            log.info("%s: lever variants before %s: %s", self.gp.alias, v["name"], " | ".join(w.replace("press pre / contact: exact IK failed at pose 1 of 2", "pre") for w in whys)[:1500])
                        break
                    whys.append(f"{v['name']}{'' if seed_posture is None else ' (seed ' + seed_posture + ')'}: lever ok, the pinch beside it fails")
                if lever_out is not None:
                    break
            if lever_out is None:
                if n_lever_ok:
                    # diagnostic: B at rest, then its pre-grasp / grasp witnesses with the LAST passing lever branch's arm put in - what collides?
                    blame = ""
                    try:
                        set_flaps(qa, q0o)
                        h_free, wp_free, r_free = self._best_pinch(getattr(self, "handle_variants", None) or self.handle, qa, st, main_arm, target, support, pose)
                        if h_free is not None and last_lever is not None:
                            idx_l = self.kit.robot.arm_control_idx[lever_arm]
                            parts = []
                            for nm, pz in (("pre", wp_free.pre_grasp), ("grasp", wp_free.eef)):
                                q_w = self.ikc.solutions.get((main_arm, self.ikc._pose_key(pz)))
                                if q_w is None:
                                    parts.append(f"{nm}: no witness")
                                    continue
                                q_c = q_w.clone()
                                q_c[idx_l] = last_lever[idx_l]
                                with self.ikc.ignoring([support, target]):
                                    ok_c = bool(self.kit.collision_free(q_c, margin=0.0)[0])
                                parts.append(f"{nm}: {'free' if ok_c else self.ikc._collision_blame(q_c, [support, target])}")
                            blame = f" [B at rest passes: site {h_free.get('site')} yaw {h_free.get('yaw_deg')} pitch {h_free.get('pitch_deg')}; with the lever arm in: " + "; ".join(parts) + "]"
                        elif h_free is None:
                            blame = " [B fails at rest too: reach]"
                    except Exception as e:  # noqa: BLE001
                        blame = f" [diagnostic raised {e!r}]"
                    return f"main pinch (B, {main_arm}, q_ajar) fails beside every lever branch ({n_lever_ok} lever chains passed){blame}: " + " | ".join(w for w in whys if "lever ok" in w)[:400]
                return "lever: " + " | ".join(w.replace("press pre / contact: exact IK failed at pose 1 of 2", "pre") for w in whys)[:2400]
            press_pre, press_pose, press_mid, press_goal = lever_v["press_pre"], lever_v["press_pose"], lever_v["press_mid"], lever_v["press_goal"]
            lift_init, lift_mid, lift_goal = lever_v["lift_init"], lever_v["lift_mid"], lever_v["lift_goal"]
            q_lift_end = lever_out["q_lift_end"]
            h_b, wp_b, r_b = pinch
            q_both = r_b.q.clone()
            q_both[self.kit.robot.arm_control_idx[lever_arm]] = q_lift_end[self.kit.robot.arm_control_idx[lever_arm]]
            with self.ikc.ignoring([support, target]):
                if not bool(self.kit.collision_free(q_both, margin=0.01)[0]):
                    return "main pinch with the lever holding the flap: the two arms collide"
            # (5) the follow to the goal (the lever gone): the required fractions from q_ajar; stage 2 where the reach ends
            reach: Dict[str, bool] = {"0.0": True}
            waypoints: Dict[str, Any] = {"0.0": wp_b.to_sidecar()}
            k_last = None
            for kk in self.required:
                qk = qa + kk * (q_goal - qa)
                set_flaps(qk, q0o)
                wp = C.handle_waypoints(h_b, self.jinfo, [qk], pose)[0]
                r = self.ikc.exact([wp.eef], st, main_arm, ignore_for_last=[support, target], q_seed=q_both, cache=False, fingers=self.grasp_fingers)
                key = f"{kk:g}"
                waypoints[key] = wp.to_sidecar()
                reach[key] = bool(r.ok)
                if not r.ok:
                    break
                k_last = kk
            if k_last is None:
                return "main follow: not even the first fraction from q_ajar is reachable"
            if k_last is not None and abs(k_last - 1.0) < 1e-9 and float(self.gen("v4_follow_overshoot_rad", 0.0)) > 0:  # (off: the sidecar rejects keys > 1; the oracle rides on along the arc instead)
                # 0921-021042: the follow ended 0.06 rad short of the goal (the hand a few mm short of the last waypoint; the goal
                # tolerance is 0.05) - one more waypoint past the goal when the arm reaches it; the oracle's stop band ends the follow at
                # the goal anyway
                over = float(self.gen("v4_follow_overshoot_rad", 0.0))
                k_over = 1.0 + over / max(1e-6, (q_goal - qa))
                set_flaps(q_goal + over, q0o)
                wp_o = C.handle_waypoints(h_b, self.jinfo, [q_goal + over], pose)[0]
                r_o = self.ikc.exact([wp_o.eef], st, main_arm, ignore_for_last=[support, target], q_seed=q_both, cache=False, fingers=self.grasp_fingers)
                if r_o.ok:
                    waypoints[f"{k_over:g}"] = wp_o.to_sidecar()
                    reach[f"{k_over:g}"] = True
                    k_last = k_over
            for kk in [float(x) for x in self.g("waypoint_fractions")]:
                key = f"{kk:g}"
                if key not in waypoints:
                    waypoints[key] = C.handle_waypoints(h_b, self.jinfo, [qa + kk * (q_goal - qa)], pose)[0].to_sidecar()
                    reach[key] = False
            return {"q_lift_end": q_lift_end, "h_b": h_b, "wp_b": wp_b, "q_both": q_both, "reach": reach, "waypoints": waypoints, "k_last": k_last, "q_press": lever_out["q_press"],
                    "brace": brace_rec,
                    "lever_v": lever_v}

        # the stances: the seam a little on the lever arm's side (the left hand pointing right passed only with the seam 10-20 cm to the
        # robot's left, 0919-201121 / 0920-160128), the trunk extended - the phase-1 combos (wide laterals, lat preference)
        cands = [(0.0, 0.0, 0.0, (manip.translate, manip.tilt), manip)] + self._v2_phase1_combos(manip, pose)
        skipped = {"base": 0, "frame": 0, "transition": 0}
        found, tried, reasons = None, 0, []
        for _, dlat, dso, trunk, st in cands:
            if tried >= int(self.gen("v4_stance_max", 16)):
                break
            self._check_deadline()
            if st is not manip:
                if not self.kit.trav_free(st.base_xy) or not self.ikc.stance_collision_free(st):
                    skipped["base"] += 1
                    continue
                if not TT.in_frame(self, self.camera_pose_at(st), pts, float(self.g("view_margin_m"))):
                    skipped["frame"] += 1
                    continue
            set_flaps(self.q_init, q0o)
            if not self._v2_transition_ok(init, st):
                skipped["transition"] += 1
                continue
            tried += 1
            out = try_stance(st)
            if isinstance(out, dict):
                found = out
                if st is not manip:
                    st.extras.update({"second_stance": True, "lateral_shift": float(dlat), "stand_off_shift": float(dso), "trunk_from": [float(manip.translate), float(manip.tilt)], "v4": True})
                manip = st
                break
            reasons.append(f"lateral {dlat:+.2f} stand-off {dso:+.2f} trunk ({trunk[0]:.2f}, {trunk[1]:+.2f}): {out}")
            log.info("%s: v4 stance %s", self.gp.alias, reasons[-1][:1500])
        set_flaps(self.q_init, q0o)
        log.info("%s: v4 stance search: %d tried, skipped before the checks: %s (of %d candidates)", self.gp.alias, tried, dict(skipped), len(cands))
        if found is None:
            raise Reject("handle_unreachable", f"two-flap v4: no stance of {tried} passes the lever + pinch sequence (skipped {dict(skipped)}); " + " | ".join(reasons[-3:])[:900])
        h_b, wp_b, reach, waypoints, k_last = found["h_b"], found["wp_b"], found["reach"], found["waypoints"], found["k_last"]
        lv = found["lever_v"]
        press_pre, press_pose, press_mid, press_goal = lv["press_pre"], lv["press_pose"], lv["press_mid"], lv["press_goal"]
        lift_init, lift_mid, lift_goal, p_press, p_lift = lv["lift_init"], lv["lift_mid"], lv["lift_goal"], lv["p_press"], lv["p_lift"]
        r_press = float(th.linalg.norm((p_press - hinge_o_w) - axis_o_w * float(th.dot(p_press - hinge_o_w, axis_o_w))))
        r_lift = float(th.linalg.norm((p_lift - hinge_w) - axis_w * float(th.dot(p_lift - hinge_w, axis_w))))
        self.handle = h_b
        # stage 2 when the follow does not reach the goal
        stage2_rec, k_mid = None, None
        if k_last < 1.0 - 1e-9:
            if not (bool(self.gen("stage2_push", False)) and self.direction == "open"):
                raise Reject("handle_unreachable", f"v4 main follow reaches k = {k_last:g} only and stage2_push is off")
            k_mid = k_last
            q_mid = qa + k_mid * (q_goal - qa)
            set_flaps(q_mid, q0o)
            self._set_joint(target, q_mid)
            stage2_rec = self._stage2_push(manip, pose, target, support, q_mid, found["q_both"])
            set_flaps(self.q_init, q0o)
        # gates
        with self.kit.ignoring([target]):
            if not bool(self.kit.collision_free(found["q_both"], margin=0.01)[0]):
                raise Reject("stance_collision", "v4: the pinch + lever configuration collides with the scene")
        cam_m = self.camera_pose_at(manip)
        if not self.ikc.eef_in_frame(wp_b.eef, cam_m):
            raise Reject("gripper_not_in_frame", "v4 main pinch eef pose outside the manipulation image (5 %)")
        manip.name = "manipulation"
        manip.arms = sorted({main_arm, lever_arm})
        self.record_transition(init, manip)
        TT.whole_body_check_at(self, [init, manip], {tname: pose}, require=True, reason="stance_collision")
        # records
        tol_o = min(self._joint_tol(self.q_goal - self.q_init), 0.5 * angle)
        prep = ContactRecord("push_prep", lever_arm, press_pre, press_pose, press_mid, press_goal, -z, r_press * angle, object=tname,
                             extras={"pose_class": "lever_press", "closed_gripper": True, "hold": True, "push_arm": lever_arm, "fist": lv["name"],
                                     "articulation": {"joint": other, "link": jo.get("child_link"), "q_from": q0o, "q_to": q0o - angle, "tolerance": tol_o,
                                                      "hinge_world": [round(float(v), 4) for v in hinge_o_w.tolist()], "axis_world": [round(float(v), 4) for v in axis_o_w.tolist()]}})
        lift_dir = right_h if str(self.gen("v4_lift_path", "vertical")) == "arc" else z
        lift_dist = r_lift * (qa - self.q_init) if str(self.gen("v4_lift_path", "vertical")) == "arc" else float(th.linalg.norm(lift_goal[0] - lift_init[0]))
        lift = ContactRecord("push_lift", lever_arm, press_goal, lift_init, lift_mid, lift_goal, lift_dir, lift_dist, object=tname,
                             extras={"pose_class": "lever_lift", "closed_gripper": True, "hold": True, "push_arm": lever_arm, "approach_is_slide": True,
                                     "articulation": {"joint": self.joint_name, "link": self.jinfo.get("child_link"), "q_from": float(self.q_init), "q_to": qa,
                                                      "tolerance": min(0.5 * (qa - self.q_init), 0.1), "repush_cap_rad": 0.3,
                                                      "hinge_world": [round(float(v), 4) for v in hinge_w.tolist()], "axis_world": [round(float(v), 4) for v in axis_w.tolist()]}})
        brace_rec = found.get("brace")
        manip.contacts = ["push_prep", "push_lift"] + (["brace"] if brace_rec is not None else [])
        manip.extras["brace"] = "present" if brace_rec is not None else "none"
        self.sidecar.add_stance(manip)
        self.sidecar.add_contact(prep)
        self.sidecar.add_contact(lift)
        if brace_rec is not None:
            self.sidecar.add_contact(brace_rec)
        self.sidecar.articulation = {"object": tname, "joint": self.joint_name, "joint_type": self._joint_type(), "q_init": qa, "q_goal": q_goal, "waypoints": waypoints,
                                     "handle_reach": {kk: v for kk, v in reach.items() if kk != "0.0"}, "passive_from": None, "follow_mode": "arm", "handle_arm": main_arm,
                                     "brace_kind": None, "direction": self.direction, "handle_width_m": self.handle.get("width"), "handle_kind": "edge_lid",
                                     "hinge_world": [round(float(v), 5) for v in hinge_w.tolist()], "axis_world": [round(float(v), 5) for v in axis_w.tolist()],
                                     "object_pose": {"position": [round(float(v), 5) for v in pose[0].tolist()], "orientation": [round(float(v), 6) for v in pose[1].tolist()]},
                                     "lever": {"arm": lever_arm, "press_seam_offset_m": d_press, "lift_insert_m": d_in, "retreat_m": float(self.gen("v4_retreat_m", 0.08)),
                                               "retreat_lift_m": float(self.gen("v4_retreat_lift_m", 0.03)), "retreat_dir": [round(float(v), 5) for v in left_h.tolist()],
                                               "brace": brace_rec is not None, "ajar_rad": qa, "axial": str(lv.get("axial", "mid")), "variant": lv.get("name")},
                                     "ajar_rad": qa, "v4": True, "pinch_axis": [float(v) for v in h_b["pinch_axis"]],
                                     # the main pinch's pre-shape: a wider gap than the 1 cm default - the pads meet a raised sheet edge along a pitched approach
                                     "preshape_margin_m": float(self.gen("v2_second_preshape_margin_m", 0.02)), "pinch_site": h_b.get("site")}
        if stage2_rec is not None:
            self.sidecar.add_contact(stage2_rec["record"])
            manip.contacts = list(manip.contacts) + ["push"]
            stage2_rec["facts"]["k_mid"] = float(k_mid)
            self.sidecar.articulation.update({"follow_to": float(k_mid), "stage2": stage2_rec["facts"]})
        log.info("%s: two-flap v4 instance: %s fist presses the %s flap %.0f cm left of the seam, slides under the %s flap and lifts it to %.2f rad; %s pinch from there to %.2f (k %.2g%s)",
                 self.gp.alias, lever_arm, other, 100 * d_press, self.joint_name, qa, main_arm, q_goal, k_last, ", then stage 2" if stage2_rec else "")
        return self._finish_draft(d, target, pose, init, manip, k, reach, kind=None, brace_state="none", stage2_rec=stage2_rec,
                                  summary_extra={"v4": {"ajar_rad": qa, "lever_arm": lever_arm, "k_last": k_last}})

    # ------------------------------------------------------------------ two-flap v2 (owner 2026-09-19 14:34)

    def _sample_with_v2(self, rng, d: Draft, init: Stance, manip: Stance, k: int, target, support, pose: TT.Pose, cam_init) -> Draft:
        """Owner sequence 2026-09-19 14:34 for a two-flap carton (``generation.prep_push: two_flap_v2``), the target already
        placed at q_init: (1) the MAIN hand's fist (right) presses the other flap down at its far end (``contacts.push_prep``,
        held); (2) the OTHER hand (left) pinches the target flap's free edge on the centre line of the seam (handle A,
        ``articulation.first``), opens it to ``q_ajar``, releases and retreats; the fist leaves (the flap springs back);
        (3) the other hand's fist presses that flap down again (``contacts.push_prep2``, held) while the main hand pinches
        the ajar flap's edge near the robot (handle B, ``articulation``) and opens it to q_goal. The stance is chosen for B
        (the band / Phase 2 were computed for it); every pinch tries both finger assignments (``_best_pinch``); the two hands
        of each phase pass the exact two-arm IK together; the follows are checked per required fraction."""
        tname = self.tname
        other = str(self.gen("prep_push_joint", ""))
        if other not in target.joints or other == self.joint_name:
            raise Reject("spec", f"two_flap_v2 needs generation.prep_push_joint = the other flap ({list(target.joints)})")
        angle = float(self.gen("prep_push_angle_rad", 0.3))
        q0o = float(self.all_jinfo.get(other, {}).get("closed_q", 0.0))
        qa, q_goal = float(self.q_ajar), float(self.q_goal)
        if not (self.q_init < qa < q_goal):
            raise Reject("spec", f"two_flap_v2: q_ajar {qa:.3f} must lie between q_init {self.q_init:.3f} and q_goal {q_goal:.3f}")
        self._set_named_joint(target, other, q0o - angle)  # the other flap pressed for every pinch check (as held at demo time)
        # ---- phase 2 stance (M2): the Phase-2 candidate first, then the second stances - B pinch + press 2 together, B's follow
        pts = C.transform_points(pose, self._hull_q_init)
        cands = [(0.0, 0.0, 0.0, (manip.translate, manip.tilt), manip)] + self._v2_stance_combos(manip, pose)
        tried = 0
        found2 = None
        reasons: List[str] = []
        for _, dlat, dso, trunk, st in cands:
            if tried >= int(self.gen("v2_stance_max", 12)):
                break
            self._check_deadline()
            if st is not manip:
                if not self.kit.trav_free(st.base_xy) or not self.ikc.stance_collision_free(st):
                    continue
                cam = self.camera_pose_at(st)
                if not TT.in_frame(self, cam, pts, float(self.g("view_margin_m"))):
                    continue
            if not self._v2_transition_ok(init, st):
                log.info("%s: v2 phase-2 stance lateral %+.2f stand-off %+.2f trunk (%.2f, %+.2f): the init transition would fail the validator", self.gp.alias, dlat, dso, trunk[0], trunk[1])
                continue
            tried += 1
            out = self._v2_try_phase2(st, pose, target, support, qa, q_goal)
            if isinstance(out, dict):
                found2 = out
                if st is not manip:
                    st.extras.update({"second_stance": True, "lateral_shift": float(dlat), "stand_off_shift": float(dso), "trunk_from": [float(manip.translate), float(manip.tilt)], "v2": True})
                manip = st
                break
            reasons.append(f"lateral {dlat:+.2f} stand-off {dso:+.2f} trunk ({trunk[0]:.2f}, {trunk[1]:+.2f}): {out}")
            log.info("%s: v2 phase-2 stance %s", self.gp.alias, reasons[-1][:300])
        if found2 is None:
            self._set_joint(target, self.q_init)
            self.kit.sync_world()
            raise Reject("handle_unreachable", f"two-flap v2 phase 2: no stance of {tried} passes (main pinch + press 2 + follow); " + " | ".join(reasons[-3:])[:900])
        m2 = manip
        h_b, wp_b, q_both_b, prep2, reach, waypoints = found2["h_b"], found2["wp_b"], found2["q_both_b"], found2["prep2"], found2["reach"], found2["waypoints"]
        self.handle = h_b
        # ---- phase 1 stance (M1): the phase-2 stance itself first (no move between the phases), else stances around it - the robot
        # moves between the phases (owner 2026-09-19 16:55); A pinch + press 1 together, A's follow to q_ajar
        found1 = None
        reasons1: List[str] = []
        tried1 = 0
        first_arm = self._v2_first_arm()
        single_press = False
        cands1 = [(0.0, 0.0, 0.0, (m2.translate, m2.tilt), m2)] + self._v2_phase1_combos(m2, pose)
        for _, dlat, dso, trunk, st in cands1:
            if tried1 >= int(self.gen("v2_phase1_stance_max", 14)):
                break
            self._check_deadline()
            if st is not m2:
                if not self.kit.trav_free(st.base_xy) or not self.ikc.stance_collision_free(st):
                    continue
                cam = self.camera_pose_at(st)
                if not TT.in_frame(self, cam, pts, float(self.g("view_margin_m"))):
                    continue
            if st is not m2:
                self._set_named_joint(target, other, q0o)
                self._set_joint(target, self.q_init)
                self.kit.sync_world()
                ok_t = self._v2_transition_ok(init, st)
                if ok_t:
                    self._set_joint(target, qa)
                    self.kit.sync_world()
                    ok_t = self._v2_transition_ok(st, m2)
                self._set_named_joint(target, other, q0o - angle)
                if not ok_t:
                    log.info("%s: v2 phase-1 stance lateral %+.2f stand-off %+.2f trunk (%.2f, %+.2f): a transition (init -> M1 or M1 -> M2) would fail the validator", self.gp.alias, dlat, dso, trunk[0], trunk[1])
                    continue
            tried1 += 1
            # at the phase-2 stance the phase-2 press may serve phase 1 too (the same fist, the same pose: one continuous press)
            out = self._v2_try_phase1(st, pose, target, support, qa, press=prep2 if st is m2 else None)
            if isinstance(out, dict) and st is m2 and out["prep1"] is prep2:
                single_press = True
            if isinstance(out, dict):
                found1 = out
                if st is not m2:
                    st.extras.update({"second_stance": True, "lateral_shift": float(dlat), "stand_off_shift": float(dso), "trunk_from": [float(m2.translate), float(m2.tilt)], "v2_phase1": True})
                m1 = st
                break
            reasons1.append(f"lateral {dlat:+.2f} stand-off {dso:+.2f} trunk ({trunk[0]:.2f}, {trunk[1]:+.2f}): {out}")
            log.info("%s: v2 phase-1 stance %s", self.gp.alias, reasons1[-1][:300])
        self._set_joint(target, self.q_init)
        self.kit.sync_world()
        if found1 is None:
            raise Reject("handle_unreachable", f"two-flap v2 phase 1: no stance of {tried1} passes (first pinch + press 1 + follow to q_ajar); " + " | ".join(reasons1[-3:])[:900])
        h_a, wp_a, wp_a1, q_both_a, prep1 = found1["h_a"], found1["wp_a"], found1["wp_a1"], found1["q_both_a"], found1["prep1"]
        two_stances = m1 is not m2
        # ---- gates
        with self.kit.ignoring([target]):
            for nm, qq in (("first pinch + far press", q_both_a), ("main pinch + press", q_both_b)):
                if not bool(self.kit.collision_free(qq, margin=0.01)[0]):
                    raise Reject("stance_collision", f"v2 {nm}: the combined two-arm configuration collides with the scene")
        for nm, pp, stx in (("first pinch", wp_a.eef, m1), ("main pinch", wp_b.eef, m2)):
            if not self.ikc.eef_in_frame(pp, self.camera_pose_at(stx)):
                raise Reject("gripper_not_in_frame", f"v2 {nm} eef pose outside the manipulation image (5 %)")
        # transitions: init -> M1 (flap closed), M1 -> M2 with the flap ajar (the brake holds it) and nothing in hand
        m1.name = "manipulation"
        # the stances' working arms BEFORE the transitions are recorded: the sweep drives in with the arriving stance's arms untucked
        # (navigation.transition -> stance_arm_vectors), and settle_validate re-sweeps with the recorded arm list (0919-222712)
        arms_all = sorted({self.handle_arm, first_arm, prep1["record"].arm, prep2["record"].arm})
        m1.arms = list(arms_all)
        m2.arms = list(arms_all)
        self._set_named_joint(target, other, q0o)  # the flaps as the robot meets them: the other flap closed (pressed only while a fist is on it)
        tr1 = self.record_transition(init, m1)
        # diagnostic (0919-224439: settle_validate rejects this very transition): the validator's call right here
        from ispatialgym.sampling.v2 import navigation as NAVD
        chk = NAVD.transition(self.kit, init, m1, min_standoff=tr1.stand_off, path_end_standoff=self.path_end_standoff(init, m1))
        log.info("%s: v2 transition init -> %s recorded at s_T %.2f (M1 stand-off %.2f lateral %+.2f trunk (%.2f, %+.2f), init trunk (%.2f, %+.2f)); validator-style re-check: %s; joints %s",
                 self.gp.alias, m1.name, tr1.stand_off, float(m1.stand_off or 0), float(m1.lateral or 0), m1.translate, m1.tilt, init.translate, init.tilt,
                 "ok" if chk is not None else "NONE", [round(float(v), 3) for v in target.get_joint_positions().tolist()])
        if two_stances:
            m2.name = "manipulation2"
            self._set_joint(target, qa)
            self.kit.sync_world()
            self.record_transition(m1, m2)
            self._set_joint(target, self.q_init)
            self.kit.sync_world()
        self._set_named_joint(target, other, q0o - angle)
        TT.whole_body_check_at(self, [init, m1] + ([m2] if two_stances else []), {tname: pose}, require=True, reason="stance_collision")
        manip = m2
        # ---- records
        if two_stances:
            m1.arms, m1.contacts, m1.extras["brace"] = arms_all, ["push_prep"], "none"
            m2.arms, m2.contacts, m2.extras["brace"] = arms_all, ["push_prep2"], "none"
            self.sidecar.add_stance(m1)
            self.sidecar.add_stance(m2)
            prep2["record"].stance = "manipulation2"
        else:
            m1.arms, m1.contacts, m1.extras["brace"] = arms_all, (["push_prep"] if single_press else ["push_prep", "push_prep2"]), "none"
            self.sidecar.add_stance(m1)
        if single_press:
            # one continuous press of the other hand's fist through both openings (owner 2026-09-19 19:52): recorded once as push_prep
            from ispatialgym.sampling.v2.stances import ContactRecord as _CR
            rec2 = prep2["record"]
            prep1 = {"record": _CR("push_prep", rec2.arm, rec2.pre, rec2.contact, rec2.mid, rec2.goal, rec2.direction, rec2.distance_m, object=rec2.object,
                                   extras=dict(rec2.extras or {})), "facts": dict(prep2["facts"], single_press=True)}
            self.sidecar.add_contact(prep1["record"])
        else:
            self.sidecar.add_contact(prep1["record"])
            self.sidecar.add_contact(prep2["record"])
        hinge_w = [round(float(v), 5) for v in (pose[0] + C.quat_rotate(pose[1], th.tensor(self.jinfo["hinge_point_obj"], dtype=th.float32))).tolist()]
        axis_w = [round(float(v), 5) for v in C.unit(C.quat_rotate(pose[1], th.tensor(self.jinfo["axis_dir_obj"], dtype=th.float32))).tolist()]
        common = {"object": tname, "joint": self.joint_name, "joint_type": self._joint_type(), "follow_mode": "arm", "direction": self.direction, "passive_from": None,
                  "handle_width_m": self.handle.get("width"), "handle_kind": "edge_lid", "hinge_world": hinge_w, "axis_world": axis_w, "brace_kind": None,
                  # the root pose the records were computed at: the oracle moves the handle records with the object (actions._handle_delta)
                  "object_pose": {"position": [round(float(v), 5) for v in pose[0].tolist()], "orientation": [round(float(v), 6) for v in pose[1].tolist()]}}
        first = dict(common, q_init=float(self.q_init), q_goal=qa, handle_arm=first_arm, waypoints={"0.0": wp_a.to_sidecar(), "1": wp_a1.to_sidecar()}, handle_reach={"1": True},
                     pinch_axis=[float(v) for v in h_a["pinch_axis"]], contact_obj=[round(float(v), 4) for v in h_a["contact"]], approach_obj=[round(float(v), 4) for v in h_a["approach_dir"]],
                     pitch_deg=h_a.get("pitch_deg"), yaw_deg=h_a.get("yaw_deg"), stance="manipulation", preshape_margin_m=found1.get("preshape_margin_m"),
                     retreat_m=found1.get("retreat_m"))
        self.sidecar.articulation = dict(common, q_init=qa, q_goal=q_goal, handle_arm=self.handle_arm, waypoints=waypoints, handle_reach={kk: v for kk, v in reach.items() if kk != "0.0"},
                                         first=first, prep_push=prep1["facts"], prep_push2=None if single_press else prep2["facts"], single_press=single_press, ajar_rad=qa, v2=True,
                                         stance="manipulation2" if two_stances else "manipulation",
                                         pinch_axis=[float(v) for v in h_b["pinch_axis"]], contact_obj=[round(float(v), 4) for v in h_b["contact"]])
        log.info("%s: two-flap v2 instance (%s%s): %s fist press (%s) -> %s pinch on the seam (yaw %s, pitch %s) to %.2f rad -> %s fist press (%s) -> %s near-end pinch to %.2f rad", self.gp.alias,
                 "two stances" if two_stances else "one stance", ", single press" if single_press else "", prep1["record"].arm, prep1["facts"].get("axial"), first_arm, h_a.get("yaw_deg"),
                 h_a.get("pitch_deg"), qa, prep2["record"].arm, prep2["facts"].get("axial"), self.handle_arm, q_goal)
        return self._finish_draft(d, target, pose, init, manip, k, reach, kind=None, brace_state="none",
                                  summary_extra={"v2": {"ajar_rad": qa, "first_arm": first_arm, "press1": prep1["facts"], "press2": prep2["facts"]}})


    def _v2_transition_ok(self, frm: Stance, to: Stance) -> bool:
        """The §3.8 transition ``frm -> to`` as settle_validate re-sweeps it (``core.check_stances_with_objects``: the recorded s_T
        as the floor, the extra drive-back from the path's end at the departing trunk) - a stance the validator would reject is
        skipped at the search (0919-230158: init trunk (0.40, 0) backing out of a stance 10 cm closer to the table collided)."""
        from ispatialgym.sampling.v2 import navigation as NAVX
        try:
            tr = NAVX.transition(self.kit, frm, to)
            if tr is None:
                return False
            return NAVX.transition(self.kit, frm, to, min_standoff=tr.stand_off, path_end_standoff=self.path_end_standoff(frm, to)) is not None
        except Exception as e:  # noqa: BLE001
            log.info("%s: transition pre-check %s -> %s raised %r", self.gp.alias, frm.name, to.name, e)
            return False

    def _v2_stance_combos(self, manip: Stance, pose: TT.Pose):
        """The second stances of the v2 search: the standard combos plus the extended-and-leaning trunk (0.5, -0.3) - the
        seam's centre line lies ~0.8 m ahead of the base from the Phase-2 stances (0919-152410 / 153540), the trunk is
        the reach the robot has. ``close_push_lat_pref_m`` (owner 15:21: the blender to the robot's right for a right-arm
        push toward the right) re-scores the combos by the target's lateral offset from that preference."""
        combos = list(self._second_stance_combos(manip, pose))
        lat_pref = self.gen("close_push_lat_pref_m", None)
        if lat_pref is not None and getattr(self, "close_push", False):
            rescored = []
            for sc, dlat, dso, trunk, st in combos:
                b = st.world_to_base(pose[0][:2])
                rescored.append((abs(float(b[1]) - float(lat_pref)) + max(0.0, float(b[0]) - 0.62) + 0.1 * abs(trunk[1]), dlat, dso, trunk, st))
            combos = rescored
        extra = []
        for dlat in (0.0, 0.10, -0.10):
            for dso, trunk in ((0.0, (0.5, -0.3)), (-0.10, (0.5, -0.3)), (0.0, (0.75, 0.0)), (0.0, (0.75, -0.3)), (-0.10, (0.75, 0.0))):
                so = max(0.25, float(manip.stand_off or TT.STAND_OFF) + dso)
                st = manip.shifted(stand_off=so, lateral=manip.lateral + dlat, name="manipulation").with_trunk(trunk[0], trunk[1], "manipulation")
                st.focus, st.arms = manip.focus, list(manip.arms) if isinstance(manip.arms, list) else manip.arms
                b = st.world_to_base(pose[0][:2])
                extra.append((abs(float(b[1])) + max(0.0, float(b[0]) - 0.62) + 0.1 * abs(trunk[1]), dlat, dso, trunk, st))
        combos += extra
        combos.sort(key=lambda c: c[0])
        return combos

    def _v2_try_phase2(self, st: Stance, pose: TT.Pose, target, support, qa: float, q_goal: float):
        """Phase 2 from one stance (the other flap pressed in the world): the main hand's near-end pinch B at q_ajar (both
        finger assignments), the other hand's fist press 2 held with it, B's follow to q_goal. Returns the records dict or
        the reason (str) of the first failing step."""
        first_arm, main_arm = self.brace_arm, self.handle_arm
        self._set_joint(target, qa)
        self.kit.sync_world()
        h_b, wp_b, r_b = self._best_pinch(getattr(self, "handle_variants", None) or self.handle, qa, st, main_arm, target, support, pose)
        if h_b is None:
            return f"main pinch (B, {main_arm}, q_ajar) fails"
        try:
            prep2 = self._prep_push_other_flap(st, pose, target, support, arms=(first_arm,), axial=str(self.gen("v2_press2_axial", "mid")), slot="push_prep2")
        except Reject as e:
            return f"press 2 ({first_arm} fist): {str(e)[:200]}"
        self._set_joint(target, qa)
        self.kit.sync_world()
        # the pinch first, the fist with it locked: the fist's branch of that joint solution is recorded as the press's IK seed
        # (contacts.push_prep*.q_hint) so the program check and the oracle reach the press in the branch that leaves room for
        # the pinch (0919-212931: solved fist-first, the fist's own branch blocked the pinch's approach)
        res_b = TT.brace_second_arm(self, st, main_arm, [wp_b.pre_grasp, wp_b.eef], first_arm, [prep2["record"].goal],
                                    ignore_work_last=[support, target], ignore_brace_last=[target], fingers=self.grasp_fingers)
        if not res_b.ok:
            return f"main pinch with the {first_arm} fist pressing: {str(res_b.reason)[:160]}"
        q_both_b = res_b.q
        prep2["record"].extras["q_hint"] = [round(float(v), 5) for v in q_both_b.tolist()]
        reach: Dict[str, bool] = {"0.0": True}
        waypoints: Dict[str, Any] = {"0.0": wp_b.to_sidecar()}
        for kk in self.required:
            qk = qa + kk * (q_goal - qa)
            self._set_joint(target, qk)
            self.kit.sync_world()
            wp = C.handle_waypoints(h_b, self.jinfo, [qk], pose)[0]
            r = self.ikc.exact([wp.eef], st, main_arm, ignore_for_last=[support, target], q_seed=q_both_b, cache=False, fingers=self.grasp_fingers)
            key = f"{kk:g}"
            waypoints[key] = wp.to_sidecar()
            reach[key] = bool(r.ok)
            if not r.ok:
                return f"main follow: handle pose at k = {kk} (q {qk:.3f}) fails ({str(r.reason)[:100]})"
        for kk in [float(x) for x in self.g("waypoint_fractions")]:
            key = f"{kk:g}"
            if key not in waypoints:
                waypoints[key] = C.handle_waypoints(h_b, self.jinfo, [qa + kk * (q_goal - qa)], pose)[0].to_sidecar()
                reach[key] = False
        return {"h_b": h_b, "wp_b": wp_b, "q_both_b": q_both_b, "prep2": prep2, "reach": reach, "waypoints": waypoints}

    def _v2_first_arm(self) -> str:
        """The hand of the first (small) opening: ``v2_first_arm`` (owner 2026-09-19 19:52: the MAIN hand pointing across the
        seam, the other hand's fist pressing on its own side - no crossing), default the other hand."""
        a = str(self.gen("v2_first_arm", self.brace_arm))
        return a if a in ("left", "right") else self.brace_arm

    def _v2_try_phase1(self, st: Stance, pose: TT.Pose, target, support, qa: float, press: Optional[Dict[str, Any]] = None):
        """Phase 1 from one stance: the first hand's pinch A on the seam at q_init (pointing across the seam when it can, both
        finger assignments, several pitches / yaws), the opposite hand's fist press held with it (``press`` = an already
        found press of that hand to reuse - the single continuous press when the first hand is the main hand -, else press 1
        at the far end is searched), A's follow to q_ajar. Returns the records dict or the reason (str)."""
        first_arm = self._v2_first_arm()
        press_arm = "right" if first_arm == "left" else "left"
        self._set_joint(target, self.q_init)
        self.kit.sync_world()
        # the across pinch's pre-shape: pinch / 2 + v2_first_preshape_margin_m (2 cm; the edge meets the pads head-on at the pitch)
        f_a = float(min(C.FINGER_OPEN, 0.5 * float(self.handle.get("width") or 0.0) + float(self.gen("v2_first_preshape_margin_m", 0.02))))
        h_a, wp_a, r_a = self._best_pinch(getattr(self, "handle_first_variants", None) or self.handle_first, self.q_init, st, first_arm, target, support, pose, fingers=f_a)
        if h_a is None:
            return f"first pinch (A, {first_arm}, seam, q_init) fails"
        # the crossing fist presses just beyond the seam (v2_press1_edge_inset_m) at the near end: its forearm then stays on its own side
        # of the seam while the pinching arm comes over the flap further back (0919-201121: at 8 cm from the seam the fist alone and
        # the pinch alone passed at neighbouring stances, together the arms crossed); a steep fist only (a 55 deg tilt's horizontal
        # component slid the 0.5 kg box 5 cm, 0919-233259); both pad orientations, each checked together with the pinch
        variants = [press] if (press is not None and press["record"].arm == press_arm) else []
        if not variants:
            whys = []
            for closing in ("along", "across"):
                try:
                    variants.append(self._prep_push_other_flap(st, pose, target, support, arms=(press_arm,), axial=str(self.gen("v2_press1_axial", "far")), slot="push_prep",
                                                               inset=self.gen("v2_press1_edge_inset_m", None), tilts=self.gen("v2_press1_tilts_deg", [0.0, 25.0, 40.0]), closing=closing))
                except Reject as e:
                    whys.append(f"{closing}: {str(e)[:120]}")
            if not variants:
                return f"press 1 ({press_arm} fist): " + " | ".join(whys)[:300]
        res_a, prep1 = None, None
        why_a = ""
        for cand in variants:
            self._set_joint(target, self.q_init)
            self.kit.sync_world()
            r_try = TT.brace_second_arm(self, st, first_arm, [wp_a.pre_grasp, wp_a.eef], press_arm, [cand["record"].goal],
                                        ignore_work_last=[support, target], ignore_brace_last=[target], fingers=f_a)
            if r_try.ok:
                res_a, prep1 = r_try, cand
                break
            why_a = str(r_try.reason)[:160]
        if res_a is None:
            return f"first pinch with the {press_arm} fist pressing: {why_a}"
        q_both_a = res_a.q
        if prep1 is not press:  # a press of its own: its fist branch compatible with this pinch (a reused press keeps the branch of phase 2)
            prep1["record"].extras["q_hint"] = [round(float(v), 5) for v in q_both_a.tolist()]
        self._set_joint(target, qa)
        self.kit.sync_world()
        wp_a1 = C.handle_waypoints(h_a, self.jinfo, [qa], pose)[0]
        r_a1 = self.ikc.exact([wp_a1.eef], st, first_arm, ignore_for_last=[support, target], q_seed=q_both_a, cache=False, fingers=f_a)
        if not r_a1.ok:
            return f"first follow: the {first_arm} hand cannot follow the flap to q_ajar ({str(r_a1.reason)[:100]})"
        # the release's clear pose (the oracle's Retreat: the edge escape + v2_first_retreat_m back along the approach) must be reachable
        # too - the pinch sits at the arm's reach margin and a retreat further out had no solution (0920-015251)
        hinge_w = pose[0] + C.quat_rotate(pose[1], th.tensor(self.jinfo["hinge_point_obj"], dtype=th.float32))
        axis_w = C.unit(C.quat_rotate(pose[1], th.tensor(self.jinfo["axis_dir_obj"], dtype=th.float32)))
        radial = wp_a1.eef[0] - hinge_w
        radial = C.unit(radial - axis_w * float(th.dot(radial, axis_w)))
        clear = C.pre_pose((wp_a1.eef[0] + radial * 0.04, wp_a1.eef[1]), float(self.gen("v2_first_retreat_m", 0.06)))
        r_clear = self.ikc.exact([clear], st, first_arm, ignore_for_last=[support, target], q_seed=r_a1.q, cache=False, fingers=f_a)
        if not r_clear.ok:
            return f"first release: the {first_arm} hand's clear pose after the small opening has no solution ({str(r_clear.reason)[:100]})"
        return {"h_a": h_a, "wp_a": wp_a, "wp_a1": wp_a1, "q_both_a": q_both_a, "prep1": prep1, "preshape_margin_m": float(self.gen("v2_first_preshape_margin_m", 0.02)),
                "retreat_m": float(self.gen("v2_first_retreat_m", 0.06))}

    def _v2_phase1_combos(self, m2: Stance, pose: TT.Pose):
        """Stances for phase 1 around the phase-2 stance (owner 2026-09-19 16:55: the robot may move between the phases): wider
        lateral shifts (a hand pointing across the seam is an elbow-out reach, the robot beside the box), stand-off shifts
        and the trunks; scored by the first pinch's contact in the base frame (depth near 0.6 m, slightly on the hand's side)."""
        c_first = C.handle_waypoints(self.handle_first, self.jinfo, [self.q_init], pose)[0].contact
        side_sign = 1.0 if self._v2_first_arm() == "left" else -1.0  # a hand pointing across the seam likes the seam a little on its own side
        lat_pref = self.gen("v2_first_lat_pref_m", None)  # owner 13:50 experiment: the right hand across the seam with the box on the robot's LEFT
        lat_pref = side_sign * 0.12 if lat_pref is None else float(lat_pref)
        out = []
        for dlat in (0.0, 0.10, -0.10, 0.20, -0.20, 0.30, -0.30):
            for dso in (0.0, -0.10, -0.20, 0.10):  # -0.20 (owner 21:20 lever brace): the box's hinge-side face was 0.78-0.88 m deep from the Phase-1 stand-off
                for trunk in ((m2.translate, m2.tilt), (0.75, 0.0), (0.5, 0.0), (0.75, -0.3), (0.25, -0.3), (0.0, -0.31)):
                    if dlat == 0.0 and dso == 0.0 and abs(trunk[0] - m2.translate) < 1e-6 and abs(trunk[1] - m2.tilt) < 1e-6:
                        continue
                    so = max(0.15, float(m2.stand_off or TT.STAND_OFF) + dso)
                    st = m2.shifted(stand_off=so, lateral=m2.lateral + dlat, name="manipulation").with_trunk(trunk[0], trunk[1], "manipulation")
                    st.focus, st.arms = m2.focus, list(m2.arms) if isinstance(m2.arms, list) else m2.arms
                    b = st.world_to_base(c_first[:2])
                    out.append((abs(float(b[0]) - 0.60) + abs(float(b[1]) - lat_pref) + 0.05 * abs(trunk[1]), dlat, dso, trunk, st))
        out.sort(key=lambda c: c[0])
        return out

    # ------------------------------------------------------------------ prep push: the other flap pressed down (owner 2026-09-19 01:10)

    def _prep_push_other_flap(self, manip: Stance, pose: TT.Pose, target, support, arms: Optional[Sequence[str]] = None, axial: str = "mid",
                              slot: str = "push_prep", inset: Optional[float] = None, tilts: Optional[Sequence[float]] = None,
                              closing: Optional[str] = None) -> Dict[str, Any]:
        """Two-flap carton, step 1 of the owner's sequence: the closed gripper presses the NON-target flap down at the seam by
        ``prep_push_angle_rad`` (its joint has an extended lower limit and a spring to 0, ``generation.joint_drives``), opening
        the seam so the other hand can pinch the target flap's free edge; the pad stays pressing until the grasp is done
        (``hold``). Contact: the flap's top face ``prep_push_edge_inset_m`` (2 cm) inside its free edge, centred along its
        hinge; approach −z (rotated with the flap), pads closing along the hinge axis; the pad rides the arc about the
        flap's hinge (``orbit_waypoints``). Arms: the brace arm first, then the handle arm. Exact IK with closed fingers of
        pre / contact (flap at 0) and mid / goal (flap re-posed). Returns ``{"record": ContactRecord, "facts": {...}}``."""
        from ispatialgym.sampling.v2.stances import ContactRecord

        other = str(self.gen("prep_push_joint", ""))
        if other not in self.all_jinfo or other == self.joint_name or other not in target.joints:
            raise Reject("spec", f"generation.prep_push_joint {other!r} must name another annotated joint of the target ({list(self.all_jinfo)})")
        ji = self.all_jinfo[other]
        angle = float(self.gen("prep_push_angle_rad", 0.3))
        axis = C.unit(th.tensor(ji["axis_dir_obj"], dtype=th.float32))
        hinge = th.tensor(ji["hinge_point_obj"], dtype=th.float32)
        z = th.tensor([0.0, 0.0, 1.0])
        bmin, bmax = th.tensor(ji["link_bbox_min"], dtype=th.float32), th.tensor(ji["link_bbox_max"], dtype=th.float32)
        corners = th.stack([th.tensor([x, y, zz]) for x in (float(bmin[0]), float(bmax[0])) for y in (float(bmin[1]), float(bmax[1])) for zz in (float(bmin[2]), float(bmax[2]))])
        perp = C.unit(th.cross(axis, z, dim=0))
        if float(th.dot(perp[:2], (corners.mean(dim=0) - hinge)[:2])) < 0:
            perp = -perp  # from the flap's hinge line toward its free edge
        r_edge = float(((corners - hinge) @ perp).max())
        inset = float(self.gen("prep_push_edge_inset_m", 0.02)) if inset is None else float(inset)
        r_c = max(0.02, r_edge - inset)
        ax_o = (corners - hinge) @ axis
        a_mid = float(ax_o.mean())
        # where along the hinge the fist presses (v2: "far" = away from the robot for the crossing right fist, so the left hand's
        # centre-line pinch and the right forearm do not meet; "near" = the robot's end; "mid" = the flap's middle)
        side_o = 1.0 if float(th.dot(axis, self.facing_local)) >= 0 else -1.0
        a_in = float(self.gen("prep_push_axial_inset_m", 0.04))
        a_near = (float(ax_o.max()) if side_o > 0 else float(ax_o.min())) - side_o * a_in
        a_far = (float(ax_o.min()) if side_o > 0 else float(ax_o.max())) + side_o * a_in
        a_coord = {"mid": a_mid, "near": a_near, "far": a_far}.get(str(axial), a_mid)
        p0 = hinge + perp * r_c + axis * a_coord
        p0 = th.tensor([float(p0[0]), float(p0[1]), float(bmax[2]) + float(self.gen("prep_push_top_clear_m", 0.002))])  # just above the flap's top face
        op, oq = pose
        pw = op + C.quat_rotate(oq, p0)
        hinge_w, axis_w = op + C.quat_rotate(oq, hinge), C.unit(C.quat_rotate(oq, axis))
        q0 = float(ji.get("closed_q", ji.get("lower", 0.0)))
        radial = pw - hinge_w
        radial = radial - axis_w * float(th.dot(radial, axis_w))
        r_w = float(th.linalg.norm(radial))
        tname = self.tname
        jnames = list(target.joints)
        # fist orientations, first passing wins: straight down or tilted away from the robot (0919-034901: the vertical fist with
        # the pads closing along the hinge axis failed the IK for both arms at every stance), pads closing along the hinge axis or
        # across it
        fwd = th.tensor([math.cos(manip.yaw), math.sin(manip.yaw), 0.0])
        perp_w = C.unit(th.cross(axis_w, z, dim=0))
        to_seam = C.unit(th.tensor([float(v) for v in (hinge_w - pw).tolist()]) * -1.0)  # from the hinge toward the seam, horizontal
        to_seam = C.unit(th.tensor([float(to_seam[0]), float(to_seam[1]), 0.0]))
        orients = []
        # pad orientation: closing along the hinge puts the pad's 5 cm side ACROSS the seam; a press close to the seam (< 5 cm) takes
        # the other one first, the pad's long side along the hinge (0919-220112: 3 cm from the seam the fist landed on the braked
        # target flap - no motion of the pressed one, the box shoved 7 mm)
        # pads closing along the hinge = the pad's 5 cm side across the seam, 2 cm along the hinge - the fist's bulk stays clear of the
        # near wall (0920-031614: with the long side along the hinge the fist 4 cm from the wall penetrated base_link) and, at
        # >= 4.5 cm from the seam, of the target flap
        closings = {"along": (axis_w,), "across": (perp_w,)}.get(closing or "", (axis_w, perp_w))
        for tilt_deg in [float(v) for v in (tilts if tilts is not None else self.gen("prep_push_tilts_deg", [0.0, 25.0, 40.0, 55.0, 70.0]))]:
            t = math.radians(tilt_deg)
            dirs = [("away", fwd)] if tilt_deg == 0.0 else [("away", fwd), ("toward", -fwd), ("seam", to_seam), ("hinge", -to_seam)]
            for dname, dvec in dirs:
                a_w = C.unit(-z * math.cos(t) + dvec * math.sin(t))
                for c_w in closings:
                    orients.append((tilt_deg if dname == "away" else tilt_deg + {"toward": 100.0, "seam": 200.0, "hinge": 300.0}[dname], a_w, c_w))

        def set_other(q):
            qs = target.get_joint_positions().clone()
            qs[jnames.index(other)] = q
            target.set_joint_positions(qs)
            target.set_joint_velocities(th.zeros_like(qs))
            target.keep_still()
            self.kit.sync_world()

        errors = []
        if arms is None:
            arms = (self.brace_arm,) if self.gen("prep_push_free_arm_only", True) else (self.brace_arm, self.handle_arm)
        for arm, (tilt_deg, aw, cw) in [(arm, o) for arm in arms for o in orients]:
            contact = C.contact_eef(pw, aw, cw)
            pre = C.pre_pose(contact, float(self.gen("prep_push_pre_dist_m", 0.06)))
            mid_pose, goal_pose = C.orbit_waypoints(contact, hinge_w, C.quat_from_axis_angle(axis_w, -angle), n=2, frame="world")
            why = None
            try:
                set_other(q0)
                r_pre = self.ikc.exact([pre, contact], manip, arm, ignore_for_last=[support, target], cache=False, fingers=C.FINGER_CLOSED)
                if not r_pre.ok:
                    why = f"pre / contact: {r_pre.reason}"
                else:
                    q_prev = r_pre.q
                    for frac, wp in ((0.5, mid_pose), (1.0, goal_pose)):
                        set_other(q0 - frac * angle)
                        r = self.ikc.exact([wp], manip, arm, ignore_for_last=[support, target], q_seed=q_prev, cache=False, fingers=C.FINGER_CLOSED)
                        if not r.ok:
                            why = f"arc pose at q {q0 - frac * angle:.3f}: {r.reason}"
                            break
                        q_prev = r.q
            finally:
                set_other(q0 - angle)  # back to the pressed state the grasp checks assume (0919-020546)
            if why is None:
                tol = self._joint_tol(self.q_goal - self.q_init)
                rec = ContactRecord(slot, arm, pre, contact, mid_pose, goal_pose, aw, r_w * angle, object=tname,
                                    extras={"pose_class": "flap_press_down", "closed_gripper": True, "hold": True, "push_arm": arm,
                                            "articulation": {"joint": other, "link": ji.get("child_link"), "q_from": q0, "q_to": q0 - angle, "tolerance": min(tol, 0.5 * angle),
                                                             "hinge_world": [round(float(v), 4) for v in hinge_w.tolist()], "axis_world": [round(float(v), 4) for v in axis_w.tolist()]}})
                log.info("%s: %s recorded (%s arm, fist tilt %.0f deg, pads closing %s the hinge, axial %s): the %s flap pressed down %.0f deg at r %.3f m",
                         self.gp.alias, slot, arm, tilt_deg, "along" if cw is axis_w else "across", axial, other, math.degrees(angle), r_w)
                return {"record": rec, "facts": {"joint": other, "arm": arm, "angle_rad": angle, "radius_m": r_w, "fist_tilt_deg": tilt_deg, "axial": axial,
                                                 "contact_obj": [round(float(v), 4) for v in p0.tolist()]}}
            errors.append(f"{arm}/tilt{tilt_deg:.0f}/{'along' if cw is axis_w else 'across'}: {why}")
        raise Reject("push_unreachable", f"{slot} (the {other} flap pressed down {math.degrees(angle):.0f} deg, axial {axial}, press point base "
                                         f"{[round(float(v), 3) for v in manip.world_to_base(pw[:2]).tolist()]} z {float(pw[2]):.2f}) fails the exact IK: {'; '.join(errors)[:500]}")

    def _brace_top_rim(self, manip: Stance, pose: TT.Pose, target, support, grasp_pair, arm: str):
        """Owner 2026-09-19 (two-flap carton, step 1b): after the prep press the free arm holds the box down - a closed-gripper
        press on the box top at the OTHER flap's hinge line (the wall rim under the flap root, not swept by the target flap),
        approach −z, pads closing along the hinge axis; two-arm exact IK together with the handle grasp. Returns a
        ``TT.BraceRecord`` or ``None`` (the brace is optional: the flap's own weight is tiny, the press is a safety)."""
        other = str(self.gen("prep_push_joint", ""))
        ji = self.all_jinfo.get(other)
        if ji is None:
            return None
        axis = C.unit(th.tensor(ji["axis_dir_obj"], dtype=th.float32))
        hinge = th.tensor(ji["hinge_point_obj"], dtype=th.float32)
        bmax = th.tensor(ji["link_bbox_max"], dtype=th.float32)
        p0 = th.tensor([float(hinge[0]), float(hinge[1]), float(bmax[2]) + 0.002])
        op, oq = pose
        pw, axis_w = op + C.quat_rotate(oq, p0), C.unit(C.quat_rotate(oq, axis))
        z = th.tensor([0.0, 0.0, 1.0])
        fwd = th.tensor([math.cos(manip.yaw), math.sin(manip.yaw), 0.0])
        perp_w = C.unit(th.cross(axis_w, z, dim=0))
        last = ""
        for tilt_deg in (0.0, 25.0, 40.0):  # the fist straight down or tilted away from the robot, pads closing along / across the hinge
            t = math.radians(tilt_deg)
            aw = C.unit(-z * math.cos(t) + fwd * math.sin(t))
            for cw in (axis_w, perp_w):
                contact = C.contact_eef(pw, aw, cw)
                pre = C.pre_pose(contact, 0.06)
                res = TT.brace_second_arm(self, manip, self.handle_arm, grasp_pair, arm, [pre, contact], ignore_work_last=[support, target], ignore_brace_last=[target], fingers=self.grasp_fingers)
                if res.ok:
                    log.info("%s: top-rim brace recorded (%s arm, fist tilt %.0f deg) at the %s hinge line", self.gp.alias, arm, tilt_deg, other)
                    return TT.BraceRecord("brace", arm, pre, contact, direction=aw, brace_kind="top_rim_press", object=self.tname,
                                          extras={"stop_clearance_m": 0.0, "pose_class": "top_rim_press", "hinge_joint": other, "fist_tilt_deg": tilt_deg})
                last = str(res.reason)[:160]
        log.info("%s: top-rim brace (%s arm) with the handle grasp fails the exact IK (%s): handle-only after the prep press", self.gp.alias, arm, last)
        return None

    # ------------------------------------------------------------------ stage 2: fist push of the lid from inside (owner 2026-09-18 22:38)

    def _stage2_push(self, manip: Stance, pose: TT.Pose, target, support, q_mid: float, q_seed) -> Dict[str, Any]:
        """Stage-2 fist push with whichever arm reaches it: the OTHER arm first (hinge on the robot's right, ring grasped by
        the right hand -> the fist stands left of the jar pushing rightward: a left-arm pose; 0919-004613: the right arm
        failed every pre-contact IK), then the handle arm."""
        errors = []
        for arm in (self.brace_arm, self.handle_arm):
            try:
                return self._stage2_push_arm(manip, pose, target, support, q_mid, q_seed if arm == self.handle_arm else None, arm)
            except Reject as e:
                errors.append(f"{arm}: {e}")
        raise Reject("push_unreachable", "; ".join(errors)[:1200])

    def _stage2_push_arm(self, manip: Stance, pose: TT.Pose, target, support, q_mid: float, q_seed, arm: str) -> Dict[str, Any]:
        """The second stage of a large opening: the lid, held by the joint brake at ``q_mid`` (the last handle pose the arm
        reaches), is pushed on to ``q_goal`` by the **closed** gripper placed on the lid's underside near the free edge -
        the fist enters the opening between the lid and the rim and pushes from inside out, the pad riding the hinge arc
        (``orbit_waypoints`` about the joint axis, as the wall push rides the hook). Contact = the underside
        ``stage2_edge_inset_m`` (2 cm) inside the free edge, ``stage2_underside_clear_m`` (2 mm) below it, centred along
        the hinge axis; approach = into the underside (the arc tangent), pads closing along the hinge axis. Exact IK of
        the pre-contact (target present), the contact and the arc's mid / goal poses (target re-posed, ignored), the
        contact in the manipulation image. Returns ``{"record": ContactRecord, "facts": {...}}``; raises ``Reject``."""
        from ispatialgym.sampling.v2.stances import ContactRecord

        axis = C.unit(th.tensor(self.jinfo["axis_dir_obj"], dtype=th.float32))
        hinge = th.tensor(self.jinfo["hinge_point_obj"], dtype=th.float32)
        z = th.tensor([0.0, 0.0, 1.0])
        perp = C.unit(th.cross(axis, z, dim=0))
        centre = self.hull.mean(dim=0)
        if float(th.dot(perp[:2], (centre - hinge)[:2])) < 0:
            perp = -perp
        bmin, bmax = th.tensor(self.jinfo["link_bbox_min"], dtype=th.float32), th.tensor(self.jinfo["link_bbox_max"], dtype=th.float32)
        corners = th.stack([th.tensor([x, y, zz]) for x in (float(bmin[0]), float(bmax[0])) for y in (float(bmin[1]), float(bmax[1])) for zz in (float(bmin[2]), float(bmax[2]))])
        r_edge = float(((corners - hinge) @ perp).max())
        inset = float(self.gen("stage2_edge_inset_m", 0.02))
        r_c = max(0.03, r_edge - inset)
        a_coord = float(((corners - hinge) @ axis).mean())
        p0 = hinge + perp * r_c + axis * a_coord
        p0 = th.tensor([float(p0[0]), float(p0[1]), float(bmin[2]) - float(self.gen("stage2_underside_clear_m", 0.002))])  # just below the underside: the arc leg presses on
        d_mid = float(q_mid) - float(self.jinfo.get("closed_q", self.lower))
        R = C.quat_to_mat(C.quat_from_axis_angle(axis, d_mid))
        p_mid = hinge + R @ (p0 - hinge)
        a_mid = R @ z  # into the underside, rotated with the lid
        op, oq = pose
        pw, aw, cw = op + C.quat_rotate(oq, p_mid), C.unit(C.quat_rotate(oq, a_mid)), C.unit(C.quat_rotate(oq, axis))
        hinge_w, axis_w = op + C.quat_rotate(oq, hinge), C.unit(C.quat_rotate(oq, axis))
        contact = C.contact_eef(pw, aw, cw)
        pre = C.pre_pose(contact, float(self.gen("stage2_pre_dist_m", 0.10)))
        delta = float(self.q_goal) - float(q_mid)
        mid_pose, goal_pose = C.orbit_waypoints(contact, hinge_w, C.quat_from_axis_angle(axis_w, delta), n=2, frame="world")
        radial = pw - hinge_w
        radial = radial - axis_w * float(th.dot(radial, axis_w))
        r_w = float(th.linalg.norm(radial))
        tname = self.tname
        why = None
        try:
            self._set_joint(target, float(q_mid))
            self.kit.sync_world()
            r_pre = self.ikc.exact([pre, contact], manip, arm, ignore_for_last=[support, target], q_seed=q_seed, cache=False, fingers=C.FINGER_CLOSED)
            if not r_pre.ok:
                why = f"pre-contact / contact at q_mid {q_mid:.3f}: {r_pre.reason}"
            else:
                q_prev = r_pre.q
                for frac, wp in ((0.5, mid_pose), (1.0, goal_pose)):
                    self._set_joint(target, float(q_mid) + frac * delta)
                    self.kit.sync_world()
                    r = self.ikc.exact([wp], manip, arm, ignore_for_last=[support, target], q_seed=q_prev, cache=False, fingers=C.FINGER_CLOSED)
                    if not r.ok:
                        why = f"arc pose at q {q_mid + frac * delta:.3f}: {r.reason}"
                        break
                    q_prev = r.q
        finally:
            self._set_joint(target, self.q_init)
            self.kit.sync_world()
        if why is not None:
            raise Reject("push_unreachable", f"stage 2 fist push on the lid underside (r {r_c:.3f} m, {math.degrees(delta):.0f} deg) fails the exact IK: {why}")
        if not self.ikc.eef_in_frame(contact, self.camera_pose_at(manip)):
            raise Reject("gripper_not_in_frame", "stage 2 push contact eef pose outside the manipulation image (5 %)")
        tol = self._joint_tol(self.q_goal - self.q_init)
        facts = {"k_mid": None, "q_mid": float(q_mid), "q_goal": float(self.q_goal), "delta_rad": delta, "push_radius_m": r_w, "edge_inset_m": inset, "joint_tolerance_rad": tol,
                 "hinge_world": [round(float(v), 4) for v in hinge_w.tolist()], "axis_world": [round(float(v), 4) for v in axis_w.tolist()], "contact_obj": [round(float(v), 4) for v in p0.tolist()]}
        rec = ContactRecord("push", arm, pre, contact, mid_pose, goal_pose, aw, r_w * abs(delta), object=tname,
                            extras={"pose_class": "lid_underside_fist", "closed_gripper": True, "push_arm": arm,
                                    "articulation": {"joint": self.joint_name, "link": self.jinfo.get("child_link"), "q_from": float(q_mid), "q_to": float(self.q_goal),
                                                     "tolerance": tol, "hinge_world": facts["hinge_world"], "axis_world": facts["axis_world"]}})
        log.info("%s: stage 2 push recorded (%s arm): fist on the lid underside r %.3f m, %.0f deg about the hinge from q %.3f to %.3f", self.gp.alias, arm, r_w, math.degrees(delta), q_mid, self.q_goal)
        return {"record": rec, "facts": facts}

    # ------------------------------------------------------------------ program check (brace-less small articulation, D-008)

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        """§3.5 program check. A handle-only instance (``generation.brace_optional``, D-008) is not yet accepted by the F6
        ``program_articulation_small`` (REQUIRED_BLOCKS: contacts.brace); until oracle-B makes it optional, with
        ``ISG_PROGRAM_CHECK=stub`` set inside the smoke command a rejection that only asks for ``contacts.brace`` is recorded as
        ``stub-brace-optional``."""
        try:
            return super().program_check(draft)
        except Reject as e:
            msg = str(e)
            if not (getattr(self, "brace_missing", False) and "brace" in msg and TT.program_check_stub()):
                raise
            log.warning("%s: program check asks for contacts.brace on a handle-only instance (%s); recorded as stub-brace-optional", self.gp.alias, msg[:200])
            report = {"mode": "stub-brace-optional", "problems": [], "note": "brace-less small articulation not yet accepted by the oracle program", "error": msg[:300]}
            self.last_program_check = report
            draft.summary["program_check"] = report
            return report

    # ------------------------------------------------------------------ hooks

    def goal(self, draft: Draft):
        manip = self.sidecar.stance("manipulation")
        self.sidecar.set_goal(manip.base_pose2d, manip.translate, manip.tilt, arms="same")

    def apply_draft(self, draft: Draft):
        super().apply_draft(draft)
        self._set_joint(self.obj(self.role), float(self.q_init))

    def after_settle(self, draft: Draft):
        tname = self.tname
        target = self.obj(self.role)
        p, q = target.get_position_orientation()
        draft.pose_targets[tname] = full_pose_target(p, q, self.tol("position", 0.03), self.tol("orientation", 0.20))
        q_now = self._joint_value(target)
        if abs(q_now - self.q_init) > 0.5 * self._joint_tol(self.q_goal - self.q_init):
            raise Reject("articulation_drift", f"init joint drifted from {self.q_init:.3f} to {q_now:.3f}")

    def refine_targets_from_goal(self, draft: Draft):
        tname = self.tname
        target = self.obj(self.role)
        q = target.get_joint_positions()
        names = list(target.joints)
        tgt = draft.articulation_targets[tname]["joints"]
        for n in names:
            if abs(float(q[names.index(n)]) - tgt[n]["position"]) > 0.5 * tgt[n]["tolerance"]:
                raise Reject("goal_unstable", f"joint {n} drifted to {float(q[names.index(n)]):.3f} (target {tgt[n]['position']:.3f})")
        super().refine_targets_from_goal(draft)


def _base_link_bbox(hull: th.Tensor, jinfo: Dict[str, Any]) -> Dict[str, Any]:
    """Base-link bounding box when the annotation lacks one: the hull below the moving link's bbox."""
    lo, hi = hull.min(dim=0).values, hull.max(dim=0).values
    lmin = jinfo.get("link_bbox_min")
    top_z = float(lmin[2]) if lmin is not None else float(hi[2])
    return {"bbox_min": lo.tolist(), "bbox_max": [float(hi[0]), float(hi[1]), min(float(hi[2]), top_z)]}


def _G():
    from ispatialgym.sampling import geometry as G

    return G
