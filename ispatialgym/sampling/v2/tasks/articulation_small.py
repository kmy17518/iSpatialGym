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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.grasp_library import library_root
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.articulation import scale_handle, scale_joint_info
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASKS = ("configuration_matching-articulation_open_small_scale", "configuration_matching-articulation_close_small_scale",
            "configuration_matching-articulation_small_scale-open", "configuration_matching-articulation_small_scale-close")


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
            outward = perp
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
        side_axis = th.tensor([-float(self.facing_local[1]), float(self.facing_local[0]), 0.0])  # robot's left when f points at the robot
        lat = float(th.dot(hc - self.hull.mean(dim=0), side_axis))
        arm_cfg = self.g("arm")
        self.handle_arm = arm_cfg if arm_cfg in ("left", "right") else ("left" if lat > float(self.g("handle_arm_side_threshold_m")) else "right")
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
        self.stable = self._stable_angles(target)
        if not self.stable:
            raise Reject("no_stable_state", f"{target.name}/{self.joint_name} holds no open value passively")
        if self.direction == "open":
            self.q_init, self.q_goal = self.lower, float(np.random.default_rng(self.ctx.base_seed).choice(self.stable))
        else:
            self.q_init, self.q_goal = float(np.random.default_rng(self.ctx.base_seed).choice(self.stable)), self.lower
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
                cands = TT.phase2_manipulation_stance(self, init, w_a, target_height=H_max, in_frame_fn=self._target_in_frame, bands=self.bands, top=self.top,
                                                      stand_offs=tuple(self.g("stand_offs_m")), focus_z=self.top.z + 0.5 * H_max)
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
        # B3's brace band (handle + §5.4 brace) when it has cells; the owner's 2026-09-16 item 2 replaces the empty §5.4 brace
        # rule by the left contact band, so the workspace band is the **handle band** (handle feasible at q_init and q_goal
        # for a yaw; mode "handle", computed in-process and cached) — the brace comes from the left contact band per instance
        for mode in ("brace", "handle"):
            try:
                b = RB.ReachBands.load(self.world.scene_model, self.top.name, target=cache_key["target"], joint=self.joint_name, mode=mode)
            except FileNotFoundError:
                continue
            if any(len(c) for c in b.cells.values()):
                log.info("%s: %s band cache hit (%d cells): %s", self.gp.alias, mode, sum(len(c) for c in b.cells.values()), b.path)
                self.handle_only_band = mode == "handle"
                return b
            log.info("%s: B3 %s band %s has no cell (no §5.4 brace is IK-feasible): computing the handle band", self.gp.alias, mode, b.path)
        self.handle_only_band = True
        kit, world = self.kit, self.world
        support = self._support()
        bc = RB.BandComputer(world, kit, arms=(self.handle_arm, self.brace_arm))
        stance = bc.stance_at_edge(self.top, support, RB.STAND_OFF, edge.name)
        base_cells, world_cells = bc.lattice_on_cells(stance, self.top.cells(free_only=True))
        q_values = [self.q_init, self.q_goal]  # the extremes; the instance checks every required q_k with the brace arm locked
        x0, y0, yaw0 = stance["base_pose2d"]
        face_off = C.wrap_angle(self.face_yaw - yaw0)
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
                combos = [(i, dy) for i in range(len(base_cells)) if bool(near[i]) for dy in (face_off, face_off - 0.1, face_off + 0.1)]
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
                feasible = []
                for i in range(len(base_cells)):
                    for dy in (face_off, face_off - 0.1, face_off + 0.1):
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
                "n_reachable_total": int(sum(len(c) for c in cells_out.values())), "computed_by": "sampling.v2.tasks.articulation_small"}
        key = RB._key({**cache_key, "q": q_values, "handle": True, "grid": meta["grid"], "lattice": bc.lattice, "stance": stance["edge"]})
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
            # the brace is a pad contact on a face of the base link (left contact band classes): every candidate face point in frame
            pts_b += [pt for _, pt, _, _ in self._body_brace_points(pose, stance)]
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
        return out

    @property
    def _hull_q_init(self) -> th.Tensor:
        if not hasattr(self, "_hull_qi"):
            self._hull_qi = self._hull_at_q(self.obj(self.role), self.q_init)
        return self._hull_qi

    # ------------------------------------------------------------------ instance (§5.4 Phase 3)

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        init = self.init_stance
        errors = []
        for k in rng.permutation(len(self.manip_candidates))[:6]:
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
        self.record_transition(init, manip)
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
        q_values = [self.q_init] + [self.q_init + kk * (self.q_goal - self.q_init) for kk in self.required]
        wp0 = C.handle_waypoints(self.handle, self.jinfo, [self.q_init], pose)[0]
        grasp_pair = [wp0.pre_grasp, wp0.eef]
        brace_extras: Dict[str, Any] = {}
        if self.handle_only_band:
            # owner decision 2026-09-16 item 2: the brace is a pad contact on a face of the base link, the classes the left
            # contact band recorded as passing at the target's cell first (A1 joint placement: the brace point within one
            # lattice cell of a left band cell), then the remaining near / left faces; far / right faces never pass (B3)
            body = self._body_brace_points(pose, manip)
            passing = None
            if self.left_bands is not None:
                cell = self.left_bands.to_base(np.asarray([[float(pose[0][0]), float(pose[0][1])]], dtype=np.float32), manip.base_pose2d)[0]
                passing = TT.cell_classes(self.left_bands, manip.translate, manip.tilt, cell) or None

            def rank(item):
                cls, pt, _, _ = item
                in_band = TT.in_band(self.left_bands, manip, pt[:2], tol=1.0)[0] if self.left_bands is not None else True
                return (0 if (passing and cls in passing) else 1, 0 if in_band else 1, 0 if cls.startswith(("near", "left")) else 1)

            body.sort(key=rank)
            cands = []
            for cls, pt, a, c_axis in body:
                if cls.startswith(("far", "right")) and not (passing and cls in passing):
                    continue  # pulling faces / the robot-right face are out of the left arm's reach (B3 measurements)
                for c_sign in (1.0, -1.0):
                    contact = C.contact_eef(pt, a, c_sign * c_axis)
                    cands.append((f"{cls}{'' if c_sign > 0 else '-'}", (C.pre_pose(contact), contact)))
            brace_extras = {"pose_classes_passing": passing, "left_band": self.left_bands is not None}
        else:
            hdata = {"handle": self.handle, "base_link": self.base_link_bbox or _base_link_bbox(self.hull, self.jinfo)}
            cands = RB.brace_candidates(self.jinfo, hdata, pose, q_values, 1.0 if self.brace_arm == "left" else -1.0, _G())
            kinds_pref = list(self.g("brace_kinds"))
            cands.sort(key=lambda c: kinds_pref.index(c[0]) if c[0] in kinds_pref else 99)
        chosen = None
        fails = []
        for kind, (b_pre, b_contact) in cands:
            # brace arm first (contact with the target ignored), then the handle arm with the brace arm locked (support + target
            # ignored for the grasp); the combined check ignores the target both hands touch (``tabletop.brace_second_arm``)
            res = TT.brace_second_arm(self, manip, self.brace_arm, [b_pre, b_contact], self.handle_arm, grasp_pair, ignore_work_last=[target],
                                      ignore_brace_last=[support, target])
            if res.ok:
                chosen = (kind, b_pre, b_contact, res.q)
                break
            fails.append(f"{kind}: {res.reason}")
        if chosen is None:
            raise Reject("contact_unreachable", f"no brace of {[c[0] for c in cands][:8]} + handle grasp at q_init passes the exact IK ({sorted(set(fails))[:3]})")
        kind, b_pre, b_contact, q_both = chosen
        if self.handle_only_band:
            brace_extras["pose_class"] = kind
            kind = "body_face"
        # grasp poses at the required waypoints with the target re-posed for each world sync (brace arm locked)
        reach: Dict[str, bool] = {"0.0": True}
        waypoints: Dict[str, Any] = {"0.0": wp0.to_sidecar()}
        for kk in self.required:
            qk = self.q_init + kk * (self.q_goal - self.q_init)
            self._set_joint(target, qk)
            self.kit.sync_world()
            wp = C.handle_waypoints(self.handle, self.jinfo, [qk], pose)[0]
            r = self.ikc.exact([wp.eef], manip, self.handle_arm, ignore_for_last=[support, target], q_seed=q_both, cache=False)
            key = f"{kk:g}"
            waypoints[key] = wp.to_sidecar()
            reach[key] = bool(r.ok)
            if not r.ok:
                self._set_joint(target, self.q_init)
                self.kit.sync_world()
                raise Reject("handle_unreachable", f"handle pose at q_k (k = {kk}) fails the exact IK with the brace arm locked")
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
        for nm, p in (("handle grasp", wp0.eef), ("brace", b_contact)):
            if not self.ikc.eef_in_frame(p, cam_manip):
                raise Reject("gripper_not_in_frame", f"{nm} eef pose outside the manipulation image (5 %)")
        TT.whole_body_check_at(self, [init, manip], {tname: pose}, require=True, reason="stance_collision")
        # record
        self.sidecar.add_stance(manip)
        brace_dir = C.approach_axis(b_contact[1])
        self.sidecar.add_contact(TT.BraceRecord("brace", self.brace_arm, b_pre, b_contact, direction=brace_dir, brace_kind=kind, object=tname,
                                                extras={"stop_clearance_m": 0.0, **brace_extras}))
        manip.contacts = ["brace"]
        self.sidecar.articulation = {"object": tname, "joint": self.joint_name, "joint_type": self._joint_type(), "q_init": float(self.q_init), "q_goal": float(self.q_goal),
                                     "waypoints": waypoints, "handle_reach": {kk: v for kk, v in reach.items() if kk != "0.0"}, "passive_from": self.passive_from,
                                     "follow_mode": "arm", "handle_arm": self.handle_arm, "brace_kind": kind, "direction": self.direction}
        joint_names = list(target.joints)
        q_closed = target.get_joint_positions().clone()
        for n, j in target.joints.items():
            q_closed[joint_names.index(n)] = float(j.lower_limit)
        q_init_vec, q_goal_vec = q_closed.clone(), q_closed.clone()
        q_init_vec[joint_names.index(self.joint_name)] = self.q_init
        q_goal_vec[joint_names.index(self.joint_name)] = self.q_goal
        d.object_poses[tname] = pose
        d.object_joints[tname] = q_init_vec
        d.goal_joints[tname] = q_goal_vec
        d.robot_q = self.stance_q(init)
        d.base_pose2d = tuple(init.base_pose2d)
        tol = self._joint_tol(self.q_goal - self.q_init)
        d.articulation_targets[tname] = {"joints": {n: {"position": float(q_goal_vec[joint_names.index(n)]), "tolerance": tol} for n in joint_names}}
        d.pose_targets[tname] = full_pose_target(pose[0], pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        mc = self.choice.models[self.role]
        self.sidecar.describe(tname, mc.category.replace("_", " "))
        self.sidecar.language["goal_numbers"]["joint_delta"] = float(self.q_goal - self.q_init)
        d.summary.update({"direction": self.direction, "joint": self.joint_name, "q_init": self.q_init, "q_goal": self.q_goal, "brace_kind": kind,
                          "handle_arm": self.handle_arm, "manipulation_candidate": k, "handle_reach": reach})
        return d

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
