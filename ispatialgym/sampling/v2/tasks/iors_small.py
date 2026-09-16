"""IORS small-scale (spec §6.11): ontop, under, inside, nextto, touching — the reference v2 task module.

Init per §4 through the shared tabletop phases (``v2/tabletop.py``: 1a edge + stand-off, 1b init trunk, 1c lateral with
the IORS margin row, 2 manipulation stance survivors, 3 target from the feasible set F, partner placement); this module
adds the §6.11 rules only: pair admissibility per template, the partner side / gap rule of §4.6 step 2, the goal pose per
relation, the place / pre-place IK from the manipulation stance after the goal settled, the touching push + brace, the
constraints it learned. Every number comes from the customs ``generation`` block (D1 ``generation_defaults``):
``partner_footprint_margin_m``, ``footprint_gap_min_m``, ``partner_gap_extra_m``, ``partner_moved_max_m``, ``goal_drop_m``,
``fit_tiers``, ``opening_margin_m`` / ``drop_height_m`` (floor) / ``release_clearance_m`` / ``max_protrusion_fraction``
(inside), ``tolerances``.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject, SupportTop
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, GraspRecord, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

RELATION_OF_SUBTASK = {
    "inter_object_relation_satisfaction-ontop_small_scale": "ontop",
    "inter_object_relation_satisfaction-under_small_scale": "under",
    "inter_object_relation_satisfaction-inside": "inside",
    "inter_object_relation_satisfaction-nextto": "nextto",
    "inter_object_relation_satisfaction-touching": "touching",
}
#: v1 ``bddl_writer`` definition ids of the relations (facade rendering when the alias has no grounded BDDL)
V1_DEFINITION_ID = {"ontop": 0, "under": 1, "inside": 2, "nextto": 3, "touching": 4}


def _tabletop():
    """The shared tabletop phases module (``v2/tabletop.py``); kept as an accessor because ``iors_large`` imports it."""
    return TT


# ----------------------------------------------------------------------------------------------------------------
# A1 (owner decision 2026-09-16): joint placement of a work point (right arm) and a brace point (left arm)
# ----------------------------------------------------------------------------------------------------------------

from ispatialgym.sampling.v2.stances import BAND_CELL_TOL, brace_placement_admissible, in_band, nearer_side_lateral  # noqa: E402,F401  (pure rules, tested there)


def left_contact_band_cells(s, trunk: Tuple[float, float], height: float):
    """Base-frame cells of the left-arm contact band (closed gripper, brace approach; one band per object height class —
    B3 A1) for ``trunk``. Coded against the documented APIs in this order: ``tabletop.ensure_contact_bands(s, arm="left",
    height=h)`` (tabletop A1 helper) then ``reach_bands.ReachBands.load(scene, support, mode="contact", arm="left",
    height_class=...)``; returns None while neither exists (the caller records the gate as unavailable and falls back to
    the exact two-arm IK alone)."""
    bands = None
    fn = getattr(TT, "ensure_contact_bands", None)
    if fn is not None:
        try:
            bands = fn(s, arm="left", height=float(height))
        except Exception as e:  # noqa: BLE001 - the helper's signature may still move
            log.info("tabletop.ensure_contact_bands unavailable for the left band: %r", e)
            bands = None
    if bands is None:
        try:
            from ispatialgym.sampling import reach_bands as RB

            bands = RB.ReachBands.load(s.world.scene_model, s.top.name, mode="contact", arm="left", height_class=float(height))
        except TypeError:
            bands = None  # the B3 left-band loader is not there yet (no ``arm`` / ``height_class`` factors)
        except Exception:  # FileNotFoundError: no left band computed for this support yet
            bands = None
    if bands is None:
        return None
    return bands.band(trunk[0], trunk[1])


#: §6.11 touching admissibility numbers (pushability / stability; ``generation`` overrides)
PUSH_MAX_PINCH = 0.089
PUSH_MAX_LONG = 0.30
PUSH_MAX_HEIGHT = 0.30
PARTNER_MAX_ASPECT = 2.5
#: inside (owner decision 2026-09-16): the open gripper's lowest point clears the partner's rim by this at the release
RELEASE_CLEARANCE = 0.02
#: touching brace (axial stop-brace rule): the pads' / blade's outer face this far outside the partner's far face, and the
#: partner height below which the top-down blade is tried before the pad contacts
STOP_CLEARANCE = 0.005
THIN_PARTNER_HEIGHT = 0.04
#: R1Pro finger envelope beyond the pad faces (m): pad / finger thickness outward of the open pads along ±c, finger
#: half width along eef x (conservative; only the open-gripper lowest point is derived from them)
FINGER_THICKNESS = 0.012
FINGER_HALF_WIDTH = 0.012


def gripper_corners_world(eef: G.Pose, open_width: float = C.GRIPPER_OPEN_M, thickness: float = FINGER_THICKNESS,
                          half_width: float = FINGER_HALF_WIDTH) -> th.Tensor:
    """World corners (8, 3) of the open gripper's finger envelope for an eef pose: ``PAD_Z_RANGE[0]`` .. ``FINGERTIP_LENGTH``
    along the approach axis a, ±(``open_width`` / 2 + ``thickness``) along the closing axis c, ±``half_width`` along eef x.
    ``min(..., 2)`` is the lowest point of the open gripper (release-height rule of §6.11 inside)."""
    pos, quat = G.t(eef[0]), G.t(eef[1])
    R = C.quat_to_mat(quat)
    a, c, x = R[:, 2], R[:, 1], R[:, 0]
    pts = []
    for za in (float(C.PAD_Z_RANGE[0]), float(C.FINGERTIP_LENGTH)):
        for sc in (-1.0, 1.0):
            for sx in (-1.0, 1.0):
                pts.append(pos + za * a + sc * (0.5 * float(open_width) + float(thickness)) * c + sx * float(half_width) * x)
    return th.stack(pts, dim=0)


@register(*RELATION_OF_SUBTASK)
class IORSSmall(SamplerV2):
    """One class for the five small-scale relations; the relation comes from the subtask id."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.relation = RELATION_OF_SUBTASK[self.subtask]
        if self.gp.bddl is None:  # facade: the v1 writer's definition id of the relation
            self.definition_id = V1_DEFINITION_ID[self.relation]
        self.top: Optional[SupportTop] = None
        self.hull: Dict[str, th.Tensor] = {}
        self.rest: Dict[str, th.Tensor] = {}
        self.grasps: List[Any] = []
        self.grasp_key: Optional[str] = None
        self.edge: Optional[EdgeFrame] = None
        self.init_stance: Optional[Stance] = None
        self.manip_candidates: List[Stance] = []
        self.pair: Dict[str, Any] = {}
        self.bands = None
        self.goal_side: Optional[str] = None  # nextto / touching: partner side the target goes to ("left" / "right", robot frame)
        self.release: Optional[Dict[str, Any]] = None  # inside: release-height facts of :meth:`release_pose`

    # ------------------------------------------------------------------ numbers (customs generation)

    @property
    def margin_support(self) -> float:
        return float(self.gen("support_margin", 0.02))

    @property
    def gap_min(self) -> float:
        return float(self.gen("footprint_gap_min_m", 0.03))

    @property
    def partner_gap_extra(self) -> float:
        return float(self.gen("partner_gap_extra_m", 0.03))

    @property
    def goal_drop(self) -> float:
        return float(self.gen("goal_drop_m", 0.01))

    # ------------------------------------------------------------------ roles

    def v1_role_aliases(self) -> Dict[str, str]:
        return {"partner_support": "support"}

    def bind_scene_roles(self):
        self.bind_floor_v2()
        req = self.choice.bindings.get("support")
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if not names:
            raise Reject("spec", f"{self.gp.alias}: scenes.{self.scene}.bindings.support must list scene object names")
        self.bind_scene_object("support", self.world.obj(names[0]))

    @property
    def support(self):
        return self.obj("support")

    # ------------------------------------------------------------------ template (§6.11 pair admissibility + §4 phases 1a–2)

    def sample_template(self):
        tgt, prt = self.obj("target"), self.obj("partner")
        self.hull["target"], self.hull["partner"] = self.world.local_hull(tgt), self.world.local_hull(prt)
        self.grasps, self.grasp_key, self.rest["target"] = TT.library_grasps(self, "target")
        # grasps whose attached object self-collides with the gripper palm / wrist (sphere fit vs the robot model) can
        # never pass an attached check (lift, carry, place): they leave the grasp set here, once per template
        attachable = [g for g in self.grasps if self.ikc.attachable(tgt, g.pose, "right")]
        self.template_facts["grasps_dropped_not_attachable"] = [g.name for g in self.grasps if g not in attachable]
        if not attachable:
            raise Reject("no_attachable_grasp", f"{self.grasp_key}: every library grasp self-collides once the object is attached")
        self.grasps = attachable
        self.rest["partner"] = self._partner_rest()
        self.pair = self.pair_admissibility()
        self.template_facts["pair"] = self.pair
        self.top = self.support_top(self.support)
        if int(self.top.free.sum()) < 50:
            raise Reject("support_small", f"{self.support.name}: too little free top")
        H_max = max(TT.height_of(self.hull["target"], self.rest["target"]), TT.height_of(self.hull["partner"], self.rest["partner"]))
        margins = self.lateral_margins()
        rng_t = np.random.default_rng(self.ctx.base_seed)
        errors: List[str] = []
        self.init_stance = None
        # Phases 1a / 1b / 1c per edge: §4.2 has no preference among the edges and §4.4 checks floor + collision per
        # lateral position, so every edge with a free-floor run in front of it (a third of its 10 cm lateral raster at
        # s = 0.45 on free floor) is tried; an edge whose margins exceed its length is skipped (§4.4 step 2)
        for edge, init in TT.edge_frames(self, self.top, rng_t):
            lats = th.linspace(-edge.length / 2 + 0.1, edge.length / 2 - 0.1, max(2, int(edge.length / 0.1))).tolist()
            n_free = sum(1 for lat in lats if self.kit.trav_free(edge.base_xy(init.stand_off, lat)))
            if n_free < max(1, len(lats) // 3):
                errors.append(f"{edge.name}: free floor at {n_free}/{len(lats)} lateral positions")
                continue
            try:
                init = TT.phase1b_init_trunk(self, init, H_max, top=self.top)
                self.init_stance = TT.phase1c_lateral(self, init, margins, rng_t, top=self.top)
                self.edge = edge
                break
            except Reject as e:
                if e.reason in ("no_workspace", "edge_too_short", "stance_collision"):
                    errors.append(f"{edge.name}: {e}")
                    continue
                raise
        if self.init_stance is None:
            raise Reject("no_free_edge", "; ".join(errors) or f"{self.support.name}: no edge with a free, collision-free stance")
        self.init_stance.name = "init"
        # Phase 2: manipulation stance survivors of the best angle class (each with its focus)
        self.manip_candidates = TT.phase2_manipulation_stance(self, self.init_stance, TT.half_extent(self.hull["target"], self.rest["target"]),
                                                              target_height=TT.height_of(self.hull["target"], self.rest["target"]), top=self.top)
        if not self.manip_candidates:
            raise Reject("no_manipulation_stance", "Phase 2 left no survivor")
        self.template_facts.update({"support": self.top.to_json(), "edge": self.edge.to_json(), "init_stance": self.init_stance.to_meta(),
                                    "n_manipulation_candidates": len(self.manip_candidates), "grasp_key": self.grasp_key, "n_grasps": len(self.grasps),
                                    "bands": getattr(getattr(self, "bands", None), "path", None), "H_max": H_max})

    def _partner_rest(self) -> th.Tensor:
        choice = self.choice.models["partner"]
        key = choice.grasp_key or self.gp.roles["partner"].grasp_key_of(choice.category, choice.model)
        if key:
            try:
                return C.rest_quat(key)
            except Exception:
                pass
        return C.IDENTITY_QUAT.clone()

    def pair_admissibility(self) -> Dict[str, Any]:
        """§6.11 per (target, partner): ontop / under -> the partner's top admits the target footprint with >= 1 cm
        (tier any_yaw / aligned / centred, else ``partner_too_small``); nextto -> the centre heights overlap
        (``nextto_height_mismatch``); touching -> partner stability / target pushability; inside -> opening >= footprint
        + 1 cm at the standing or lying orientation (the drop test runs in the goal check)."""
        t_ext = TT.hull_extents(self.hull["target"], self.rest["target"])
        p_ext = TT.hull_extents(self.hull["partner"], self.rest["partner"])
        out: Dict[str, Any] = {"target_extent": t_ext.tolist(), "partner_extent": p_ext.tolist(), "relation": self.relation}
        tx, ty = sorted([float(t_ext[0]), float(t_ext[1])])
        px, py = sorted([float(p_ext[0]), float(p_ext[1])])
        m = float(self.gen("partner_footprint_margin_m", 0.01))
        if self.relation in ("ontop", "under"):
            tiers = list(self.gen("fit_tiers", ["any_yaw", "aligned", "centred"]))
            diag = math.hypot(tx, ty)
            tier = None
            if "any_yaw" in tiers and diag + 2 * m <= px:
                tier = "any_yaw"
            elif "aligned" in tiers and tx + 2 * m <= px and ty + 2 * m <= py:
                tier = "aligned"
            elif "centred" in tiers and tx + 2 * m <= px and ty <= py:
                tier = "centred"
            if tier is None:
                raise Reject("partner_too_small", f"target {tx:.2f}x{ty:.2f} does not fit on partner {px:.2f}x{py:.2f} with {m:.2f} margin")
            out["tier"] = tier
        elif self.relation == "nextto":
            t_h, p_h = float(t_ext[2]), float(p_ext[2])
            if not (min(t_h, p_h) > 0):
                raise Reject("nextto_height_mismatch", "degenerate heights")
            out["g_NT"] = float((t_ext + p_ext).mean() / 6)  # OmniGibson NextTo threshold
        elif self.relation == "touching":
            if float(p_ext[2]) > float(self.gen("partner_max_aspect", PARTNER_MAX_ASPECT)) * min(px, py) or px < 0.03:
                raise Reject("partner_unstable", f"partner {px:.2f}x{py:.2f}x{float(p_ext[2]):.2f}")
            if tx > float(self.gen("push_max_pinch_m", PUSH_MAX_PINCH)) or ty > float(self.gen("push_max_long_m", PUSH_MAX_LONG)) or float(t_ext[2]) > float(self.gen("push_max_height_m", PUSH_MAX_HEIGHT)):
                raise Reject("target_not_pushable", f"target {tx:.2f}x{ty:.2f}x{float(t_ext[2]):.2f}")
            out["g_NT"] = float((t_ext + p_ext).mean() / 6)
        elif self.relation == "inside":
            om = float(self.gen("opening_margin_m", 0.01))
            if tx + om > px or ty + om > py:
                raise Reject("does_not_fit_inside", f"target {tx:.2f}x{ty:.2f} vs opening {px:.2f}x{py:.2f}")
        return out

    def lateral_margins(self) -> Dict[str, Any]:
        """The IORS row of §4.4 for ``tabletop.lateral_margins`` (λ, b, R_img are filled by Phase 1b / 1c): w_t, w_p, W_p,
        D_p_max = W_p + 0.03, safety, relation, g_NT."""
        w_t = TT.half_extent(self.hull["target"], self.rest["target"])
        w_p = TT.half_extent(self.hull["partner"], self.rest["partner"])
        W_p = 2 * w_p
        return {"kind": "iors", "w_t": w_t, "w_p": w_p, "W_p": W_p, "D_p_max": W_p + self.partner_gap_extra, "safety": 0.02, "relation": self.relation,
                "g_NT": self.pair.get("g_NT"), "nextto_gap_extra": self.partner_gap_extra}

    # ------------------------------------------------------------------ instance (§4.6 Phase 3 + §6.11)

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        init = self.init_stance
        self.sidecar.add_stance(init)
        cam_init = self.camera_pose_at(init)
        order = rng.permutation(len(self.manip_candidates))
        errors = []
        for k in order[:6]:
            cand = self.manip_candidates[int(k)]
            manip = cand.with_trunk(cand.translate, cand.tilt, "manipulation")
            manip.focus, manip.arms = cand.focus, ["right"]
            manip.extras.update(cand.extras)
            try:
                # §4.5 step 5: transition init -> manipulation
                self.sidecar.transitions.clear()
                self.record_transition(init, manip)
                # §4.6 step 1: target init pose from the feasible set F at the focus (exact IK confirmation)
                t_pose, grasp = TT.phase3_target(self, manip, rng, self.grasps, self.hull["target"], self.rest["target"], self.support_objects(),
                                                 top=self.top, scale_key=self.choice.models["target"].scale)
                # §4.6 step 2: partner on a random side (touching: robot's left) at gap g ∈ [g_lo, D_p]
                p_pose = self.place_partner(rng, t_pose, manip, cam_init)
                # §4.6 step 4: every object fully in frame (5 %) in the init image
                for role, pose in (("target", t_pose), ("partner", p_pose)):
                    if not TT.in_frame(self, cam_init, TT.hull_corners_world(self.hull[role], pose)):
                        raise Reject("not_in_frame", f"{role} not in the init image (5 %)")
                grasp.place_on = pname
                # carry height: the target (bottom at the table at init) clears the partner's top by 3 cm on the way
                grasp.carry_height = max(float(self.gen("carry_clearance_m", 0.03)), TT.height_of(self.hull["partner"], self.rest["partner"]) + float(self.gen("carry_clearance_m", 0.03)))
                self.sidecar.add_stance(manip)
                self.sidecar.add_grasp(grasp)
                d.object_poses[tname] = t_pose
                d.object_poses[pname] = p_pose
                d.robot_q = self.stance_q(init)
                d.base_pose2d = tuple(init.base_pose2d)
                d.summary.update({"relation": self.relation, "manipulation_candidate": int(k), "grasp": {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw},
                                  "manipulation_stance": manip.to_meta(), "pair": self.pair})
                d.pose_targets[pname] = self._pose_target("partner", p_pose)
                d.pose_targets[tname] = self._pose_target("target", t_pose)
                self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "))
                self.sidecar.describe(pname, self.choice.models["partner"].category.replace("_", " "))
                return d
            except Reject as e:
                errors.append(f"candidate {int(k)}: {e}")
                self.sidecar.stances.pop("manipulation", None)
                self.sidecar.grasps.clear()
        raise Reject("target_placement", "; ".join(errors[-3:]))

    def _pose_target(self, role: str, pose: G.Pose) -> Dict[str, Any]:
        tgt = full_pose_target(pose[0], pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        mc = self.choice.models[role]
        eq = self.gp.orientation_equivalences(role, mc.category, mc.model)
        if eq:
            tgt["orientation_equivalences"] = [[float(v) for v in q] for q in eq]
        tols = self.gen("tolerances") or {}
        if role == "target" and "orientation_metric" in tols and tols["orientation_metric"] is None:
            tgt["orientation_metric"] = None
            tgt["orientation_tolerance"] = None
        return tgt

    def gap_range(self) -> Tuple[float, float]:
        """§4.6 step 2: ontop / under / inside / touching g ∈ [0.03, D_p]; nextto g ∈ [g_NT + 0.03, D_p] (D_p from Phase 1c)."""
        info = (self.init_stance.extras.get("margins") or {})
        D_p = float(info.get("D_p", self.lateral_margins()["D_p_max"]))
        g_lo = self.gap_min
        if self.relation == "nextto":
            g_lo = float(self.pair["g_NT"]) + self.partner_gap_extra
            if D_p < g_lo:
                raise Reject("pair_gap", f"D_p {D_p:.2f} < g_NT + {self.partner_gap_extra:.2f} = {g_lo:.2f}")
        return g_lo, max(g_lo, D_p)

    def place_partner(self, rng: np.random.Generator, t_pose: G.Pose, manip: Stance, cam: G.Pose) -> G.Pose:
        """§4.6 step 2 with the gap range per side cut to the reach band of the manipulation trunk (§4.1 / principle 3:
        placements come from the cached reachability sets): the goal position of the target — the partner's centre for
        ontop / under / inside — must be a lateral the band covers at the target's depth, otherwise the place pose of
        §6.11 cannot pass. Sides with an empty admissible gap range are not drawn; touching keeps the robot's left."""
        g_lo, g_hi = self.gap_range()
        sides = [1.0] if self.relation == "touching" else [1.0, -1.0]
        ranges: Dict[float, Tuple[float, float]] = {}
        if self.relation in ("ontop", "under", "inside") and getattr(self, "bands", None) is not None:
            cells = self.bands.band(manip.translate, manip.tilt)
            t_fp = self.world.footprint(self.hull["target"], t_pose)
            t_lat = (t_fp - manip.base_xy) @ manip.tangent
            depth_t = float(manip.world_to_base(t_pose[0][:2])[0])
            w_p = TT.half_extent(self.hull["partner"], self.rest["partner"])
            near = cells[np.abs(cells[:, 0] - depth_t) <= 0.12] if len(cells) else cells
            if len(near):
                lam_min, lam_max = float(near[:, 1].min()), float(near[:, 1].max())
                for sd in sides:
                    if sd > 0:
                        lo, hi = lam_min - float(t_lat.max()) - w_p, lam_max - float(t_lat.max()) - w_p
                    else:
                        lo, hi = float(t_lat.min()) - w_p - lam_max, float(t_lat.min()) - w_p - lam_min
                    lo, hi = max(lo, g_lo), min(hi, g_hi)
                    if lo <= hi:
                        ranges[sd] = (lo, hi)
            if not ranges:
                raise Reject("partner_placement", f"no side admits a partner centre inside the band (λ ∈ [{lam_min:.2f}, {lam_max:.2f}] at depth "
                                                  f"{depth_t:.2f}) with g ∈ [{g_lo:.2f}, {g_hi:.2f}]" if len(near) else "band has no cell at the target's depth")
        else:
            ranges = {sd: (g_lo, g_hi) for sd in sides}
        side = list(ranges)[int(rng.integers(len(ranges)))]
        if self.relation != "touching":
            return TT.place_partner(self, rng, t_pose, self.hull["target"], self.hull["partner"], self.rest["partner"], manip, side, ranges[side],
                                    top=self.top, cam=cam, margin=self.margin_support, min_gap=self.gap_min)
        # A1: the push work point (target's right face) must lie in the right band and the brace point (partner's far /
        # left face) in the left contact band of the same trunk before the two-arm IK (§4.6 step 6) is attempted
        right_cells = self.bands.band(manip.translate, manip.tilt) if getattr(self, "bands", None) is not None else None
        left_cells = left_contact_band_cells(self, (manip.translate, manip.tilt), TT.height_of(self.hull["partner"], self.rest["partner"]))
        self.template_facts["left_contact_band"] = "unavailable" if left_cells is None else int(len(left_cells))
        errors = []
        for _ in range(8):
            p_pose = TT.place_partner(self, rng, t_pose, self.hull["target"], self.hull["partner"], self.rest["partner"], manip, side, ranges[side],
                                      top=self.top, cam=cam, margin=self.margin_support, min_gap=self.gap_min)
            if left_cells is None or right_cells is None:
                return p_pose  # the joint band gate cannot run yet (recorded); the exact two-arm IK remains the gate
            work, brace = self.touching_points(t_pose, p_pose, manip)
            if brace_placement_admissible(manip.world_to_base(work[:2]), manip.world_to_base(brace[:2]), right_cells, left_cells):
                return p_pose
            errors.append(f"work {manip.world_to_base(work[:2]).tolist()} / brace {manip.world_to_base(brace[:2]).tolist()}")
        # distinct reason for the board (owner: touching may get axial's optional-brace rule later; until then the joint
        # band gate rejects as ``brace_off_band``)
        raise Reject("brace_off_band", f"no partner position puts the push point in the right band and the brace point in the left band "
                                       f"at trunk ({manip.translate}, {manip.tilt:.2f}); tried {errors[-2:]}")

    def touching_points(self, t_pose: G.Pose, p_pose: G.Pose, manip: Stance) -> Tuple[th.Tensor, th.Tensor]:
        """§6.11 touching: the push contact on the target's right face (mid height) and the brace point on the partner's far
        (left) face (mid height), world xyz, from the target / partner poses."""
        tangent = manip.tangent
        t_hull = self.world.hull_at(self.hull["target"], t_pose)
        p_hull = self.world.hull_at(self.hull["partner"], p_pose)
        z_t = float(t_hull[:, 2].min()) + 0.5 * float(t_hull[:, 2].max() - t_hull[:, 2].min())
        right_face = float((t_hull[:, :2] @ tangent).min())
        centre_t = t_hull.mean(dim=0)
        work = th.tensor([*(centre_t[:2] + tangent * (right_face - float(centre_t[:2] @ tangent))).tolist(), z_t])
        left_face = float((p_hull[:, :2] @ tangent).max())
        centre_p = p_hull.mean(dim=0)
        z_p = float(p_hull[:, 2].min()) + 0.5 * float(p_hull[:, 2].max() - p_hull[:, 2].min())
        brace = th.tensor([*(centre_p[:2] + tangent * (left_face - float(centre_p[:2] @ tangent))).tolist(), z_p])
        return work, brace

    # ------------------------------------------------------------------ goal (§6.11)

    def goal(self, draft: Draft):
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        goal = self.goal_pose(draft.object_poses[tname], draft.object_poses[pname])
        draft.goal_poses[tname] = goal
        self._set_target_goal(draft, tname, goal)
        if self.goal_side is not None:
            draft.summary["goal_side"] = self.goal_side
        manip = self.sidecar.stance("manipulation")
        self.sidecar.set_goal(manip.base_pose2d, manip.translate, manip.tilt, arms="same")

    def _set_target_goal(self, draft: Draft, tname: str, goal: G.Pose):
        tgt = draft.pose_targets[tname]
        tgt["position"] = [float(v) for v in goal[0]]
        if tgt.get("orientation_metric") is not None:
            tgt["orientation"] = [float(v) for v in goal[1]]

    def goal_pose(self, t_pose: G.Pose, p_pose: G.Pose) -> G.Pose:
        """§6.11 goal: ontop / under(small) -> centred on the partner, z = partner top + bottom offset + goal_drop; inside
        -> the **release pose**: centred on the fillable centroid, high enough above the rim that the open gripper never
        touches the partner (:meth:`release_pose`; the drop test of the goal check settles it, the settled pose is the
        recorded goal); nextto / touching -> the position closest to the init pose that satisfies the predicate: the
        target goes on the partner **side facing its init position** (nextto: AABB gap < g_NT; touching: contact,
        nudged in the goal check). ``self.goal_side`` = ``"left"`` / ``"right"`` (robot frame) records the chosen side."""
        t_quat = t_pose[1]
        t_hull = G.local_points_world(self.hull["target"], (th.zeros(3), t_quat))
        p_hull_w = self.world.hull_at(self.hull["partner"], p_pose)
        st = self.sidecar.stance("manipulation")
        self.goal_side = None
        if self.relation in ("ontop", "under"):
            z = float(p_hull_w[:, 2].max()) - float(t_hull[:, 2].min()) + self.goal_drop
            xy = (p_hull_w[:, :2].max(dim=0).values + p_hull_w[:, :2].min(dim=0).values) / 2
            return th.tensor([float(xy[0]), float(xy[1]), z]), t_quat
        if self.relation == "inside":
            return self.release_pose(t_quat, t_hull, p_hull_w, p_pose)
        gap = 0.0 if self.relation == "touching" else max(0.0, float(self.pair["g_NT"]) - 0.01)
        p_lat = (p_hull_w[:, :2] - st.base_xy) @ st.tangent
        t_lat = t_hull[:, :2] @ st.tangent  # the target's own lateral extent about its origin
        _, lat_init = st.local(t_pose[0])
        # owner review 2026-09-16: the partner side that faces the target's init position (no lift over the partner)
        self.goal_side, lat = nearer_side_lateral(lat_init, (float(p_lat.min()), float(p_lat.max())), (float(t_lat.min()), float(t_lat.max())), gap)
        if self.relation == "touching" and self.goal_side != "right":
            # §4.6 step 2 puts the partner on the robot's left of the target: the push goes toward +tangent into the
            # partner's right face, the brace holds its left face; a target starting left of the partner contradicts that
            raise Reject("touching_side", f"target init lateral {lat_init:+.2f} is left of the partner centre {0.5 * float(p_lat.min() + p_lat.max()):+.2f}")
        depth_t, _ = st.local(t_pose[0])
        xy = st.world(depth_t, lat)
        return th.tensor([float(xy[0]), float(xy[1]), float(t_pose[0][2])]), t_quat

    def release_pose(self, t_quat: th.Tensor, t_hull: th.Tensor, p_hull_w: th.Tensor, p_pose: G.Pose) -> G.Pose:
        """§6.11 inside, owner decision 2026-09-16: release just above the partner's rim so the gripper never contacts it.
        The object is centred on the container and raised until the lowest point of the **open** gripper (finger corners:
        ``PAD_Z_RANGE[0]`` .. ``FINGERTIP_LENGTH`` along a, ±(``GRIPPER_OPEN_M`` / 2 + finger thickness) along c,
        ±finger half width along x, from the drawn grasp's ``eef_in_object``) clears the rim ``p_hull_w[:, 2].max()`` by
        ``release_clearance_m`` (default 0.02); ``drop_height_m`` stays the floor of the object's bottom above the rim.
        The target footprint at the release pose must lie within the opening (outer extents minus ``opening_margin_m``),
        else ``Reject("does_not_fit_inside")``. Facts for the sidecar / summary in ``self.release``."""
        tname = self.bindings.one("target").name
        centre, rim_z = self._container_volume(self.obj("partner"), p_pose)
        t_min = float(t_hull[:, 2].min())
        drop_floor = float(self.gen("drop_height_m", 0.005))
        clearance = float(self.gen("release_clearance_m", RELEASE_CLEARANCE))
        pos = th.tensor([float(centre[0]), float(centre[1]), rim_z + drop_floor - t_min])
        grasp = self.sidecar.grasp_of(tname)
        finger_low = None
        if grasp is not None and grasp.eef_in_object is not None:
            eef = IK.grasp_pose((pos, t_quat), grasp.eef_in_object)
            finger_low = float(gripper_corners_world(eef, open_width=C.GRIPPER_OPEN_M)[:, 2].min())
            lift = max(0.0, rim_z + clearance - finger_low)
            pos = pos + th.tensor([0.0, 0.0, lift])
            finger_low += lift
        # lateral fit of the target at the release pose inside the opening (outer extents minus the margin per side)
        om = float(self.gen("opening_margin_m", 0.01))
        t_xy = self.world.hull_at(self.hull["target"], (pos, t_quat))[:, :2]
        p_lo, p_hi = p_hull_w[:, :2].min(dim=0).values, p_hull_w[:, :2].max(dim=0).values
        if bool((t_xy.min(dim=0).values < p_lo + om / 2).any()) or bool((t_xy.max(dim=0).values > p_hi - om / 2).any()):
            raise Reject("does_not_fit_inside", f"target footprint {[round(float(v), 3) for v in (t_xy.max(dim=0).values - t_xy.min(dim=0).values)]} at the release "
                                                f"pose leaves the opening {[round(float(v), 3) for v in (p_hi - p_lo)]} (margin {om:.3f})")
        self.release = {"place_kind": "drop", "rim_z": round(rim_z, 4), "drop_height_m": round(float(pos[2]) + t_min - rim_z, 4),
                        "finger_clearance_m": None if finger_low is None else round(finger_low - rim_z, 4), "release_clearance_m": clearance}
        return pos, t_quat

    def after_settle(self, draft: Draft):
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        p, q = self.obj("partner").get_position_orientation()
        draft.pose_targets[pname] = self._pose_target("partner", (p.clone(), q.clone()))
        goal = self.goal_pose(draft.object_poses[tname], (p.clone(), q.clone()))
        draft.goal_poses[tname] = goal
        self._set_target_goal(draft, tname, goal)
        # the recorded grasp poses follow the settled target pose (what the oracle re-derives from the loaded instance)
        grasp = self.sidecar.grasp_of(tname)
        if grasp is not None and grasp.eef_in_object is not None:
            grasp.grasp = IK.grasp_pose(draft.object_poses[tname], grasp.eef_in_object)
            grasp.pre_grasp = IK.pre_pose(grasp.grasp)
            manip = self.sidecar.stance("manipulation")
            res = self.ikc.exact([grasp.pre_grasp, grasp.grasp], manip, grasp.arm, ignore_for_last=self.support_objects(),
                                 scale_key=self.choice.models["target"].scale, cache=False, approached=[self.obj("target")])
            if not res.ok:
                raise Reject(res.code or "grasp_unreachable", f"pre-grasp / grasp at the settled target pose fail the exact IK ({res.reason})")

    def refine_targets_from_goal(self, draft: Draft):
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        if self.relation == "touching":
            self._nudge_until_touching(tname, pname)
        super().refine_targets_from_goal(draft)
        if self.relation == "inside":
            self._check_inside_protrusion(tname, pname)
        if self.relation == "nextto":
            # the nearer-side goal must still satisfy the predicate after the settle: AABB gap < g_NT (PhysX poses + hulls;
            # ``obj.aabb`` reads the render-side transform, which physics-only steps leave stale)
            gap = self._aabb_gap(tname, pname)
            draft.summary["nextto_gap_m"] = round(gap, 4)
            if gap >= float(self.pair["g_NT"]):
                raise Reject("goal_infeasible", f"nextto AABB gap {gap:.3f} >= g_NT {float(self.pair['g_NT']):.3f} after the settle")
        p, _ = self.obj("partner").get_position_orientation()
        tgt = draft.pose_targets[pname]
        moved = float(th.linalg.norm(p - th.tensor(tgt["position"])))
        if moved > float(self.gen("partner_moved_max_m", 0.005)):
            raise Reject("goal_partner_moved", f"partner moved {moved:.3f} m at the goal")

    def _aabb_gap(self, tname: str, pname: str) -> float:
        """Norm of the per-axis separation vector between the two objects' hull AABBs at their current PhysX poses (0 when
        they overlap on every axis) — the distance OmniGibson's ``NextTo`` compares with g_NT = mean(dims_a + dims_b) / 6."""
        boxes = []
        for role, name in (("target", tname), ("partner", pname)):
            p, q = self.objects[name].get_position_orientation()
            h = self.world.hull_at(self.hull[role], (p.float(), q.float()))
            boxes.append((h.min(dim=0).values, h.max(dim=0).values))
        (a_lo, a_hi), (b_lo, b_hi) = boxes
        sep = th.clamp(th.maximum(a_lo - b_hi, b_lo - a_hi), min=0.0)
        return float(th.linalg.norm(sep))

    def _check_inside_protrusion(self, tname: str, pname: str):
        """§6.11 inside drop test: after the settle the target protrudes above the partner's rim by at most half its height
        (hull AABBs at the PhysX poses; ``obj.aabb`` reads the render-side transform, stale after physics-only steps)."""
        boxes = {}
        for role, name in (("target", tname), ("partner", pname)):
            p, q = self.objects[name].get_position_orientation()
            h = self.world.hull_at(self.hull[role], (p.float(), q.float()))
            boxes[role] = (h.min(dim=0).values, h.max(dim=0).values)
        (t_lo, t_hi), (_, p_hi) = boxes["target"], boxes["partner"]
        protrusion = float(t_hi[2] - p_hi[2])
        if protrusion > float(self.gen("max_protrusion_fraction", 0.5)) * float(t_hi[2] - t_lo[2]):
            raise Reject("does_not_fit_inside", f"target protrudes {protrusion:.3f} m above the rim")

    def after_goal_check(self, draft: Draft):
        """§6.11: place pose (settled goal pose ∘ T_obj→eef of the drawn grasp) and pre-place IK-checked from the
        manipulation stance with the object attached (support and partner ignored for the place pose only); the grasp
        and place eef poses in the manipulation image (§3.9); touching: push contact + left brace recorded."""
        tname = self.bindings.one("target").name
        grasp = self.sidecar.grasp_of(tname)
        manip = self.sidecar.stance("manipulation")
        goal_pos = th.tensor(draft.pose_targets[tname]["position"])
        goal_quat = draft.goal_poses[tname][1] if draft.pose_targets[tname].get("orientation") is None else th.tensor(draft.pose_targets[tname]["orientation"])
        if self.relation == "inside":
            # the place is the release above the rim (``draft.goal_poses`` = the release pose of :meth:`release_pose`); the
            # recorded goal is what the drop settled to. The gripper never reaches into the partner, so it is not ignored.
            rel_pos, rel_quat = draft.goal_poses[tname]
            place = IK.grasp_pose((G.t(rel_pos), G.t(rel_quat)), grasp.eef_in_object)
            pre_place = IK.pre_pose(place)
            ignore = self.support_objects()
            rim_z = float(self.release["rim_z"]) if getattr(self, "release", None) else None
            low = float(gripper_corners_world(place)[:, 2].min())
            if rim_z is not None and low < rim_z + 0.5 * float(self.gen("release_clearance_m", RELEASE_CLEARANCE)):
                raise Reject("release_too_low", f"open gripper lowest point {low:.3f} vs rim {rim_z:.3f} at the recorded place")
            grasp.extras.update(self.release or {})
            grasp.extras["place_lowest_finger_z"] = round(low, 4)
            draft.summary["release"] = dict(self.release or {}, place_lowest_finger_z=round(low, 4), settled_goal_z=round(float(goal_pos[2]), 4))
        else:
            place = IK.grasp_pose((goal_pos, goal_quat), grasp.eef_in_object)
            pre_place = IK.pre_pose(place)
            ignore = self.support_objects() + [self.obj("partner")]
        if self.goal_side is not None:
            draft.summary["goal_side"] = self.goal_side
        with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
            res = self.ikc.exact([pre_place, place], manip, grasp.arm, ignore_for_last=ignore, scale_key=self.choice.models["target"].scale, cache=False,
                                 approached=[self.obj("partner")] if self.relation != "inside" else [])
        if not res.ok and res.code:
            raise Reject(res.code, f"pre-place -> place descent from the manipulation stance with the object attached: {res.reason}")
        if not res.ok:
            # diagnostics only on failure (which pose, and whether the attached object / the world / the reach is the cause)
            def _ok(r):
                return "ok" if r.ok else "fail"

            r_pre = self.ikc.exact([pre_place], manip, grasp.arm, cache=False)
            r_place = self.ikc.exact([place], manip, grasp.arm, ignore_for_last=ignore, cache=False)
            everything = [o for o in self.world.scene.objects if self.world.obstacle_filter(o)]
            with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
                r_pre_obj = self.ikc.exact([pre_place], manip, grasp.arm, cache=False)
                r_pre_obj_ign = self.ikc.exact([pre_place], manip, grasp.arm, ignore_for_last=ignore, cache=False)
                r_place_obj = self.ikc.exact([place], manip, grasp.arm, ignore_for_last=ignore, cache=False)
                r_place_obj_free = self.ikc.exact([place], manip, grasp.arm, ignore_for_last=everything, cache=False)
            _, lat = manip.local(goal_pos[:2])
            raise Reject("place_unreachable", f"pre-place / place fail the exact IK from the manipulation stance with the object attached "
                                              f"(without the object: pre-place {_ok(r_pre)}, place {_ok(r_place)}; with the object: pre-place {_ok(r_pre_obj)}, "
                                              f"pre-place ignoring support+partner {_ok(r_pre_obj_ign)}, place {_ok(r_place_obj)}, place ignoring every object "
                                              f"{_ok(r_place_obj_free)}; goal lateral {lat:+.2f} m, trunk ({manip.translate}, {manip.tilt:.2f}), stand-off {manip.stand_off})")
        grasp.pre_place, grasp.place = pre_place, place
        # the lift (oracle ``Lift``): grasp pose raised by carry_height with the object attached, support ignored (the
        # object leaves it); the pre-place is where the carry ends, so the lift end must be IK-feasible too
        lift_end = (G.t(grasp.grasp[0]) + th.tensor([0.0, 0.0, float(grasp.carry_height)]), G.t(grasp.grasp[1]))
        with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
            # tracked from the grasp configuration (one IK branch along the lift), support ignored throughout
            r_lift = self.ikc.waypoints_ok([grasp.grasp, lift_end], manip, grasp.arm, ignore_all=self.support_objects(), cache=False, approached=[])
        if not r_lift.ok:
            raise Reject(r_lift.code or "lift_unreachable", f"lift ({grasp.carry_height:.2f} m above the grasp) fails with the object attached: {r_lift.reason}")
        cam = self.camera_pose_at(manip)
        if not self.ikc.eef_in_frame(grasp.grasp, cam):
            raise Reject("gripper_not_in_frame", "grasp eef pose outside the manipulation image (5 %)")
        if not self.ikc.eef_in_frame(place, cam):
            raise Reject("gripper_not_in_frame", "place eef pose outside the manipulation image (5 %)")
        if self.relation == "touching":
            self.touching_contacts(draft, manip)

    def touching_contacts(self, draft: Draft, manip: Stance):
        """§6.11 touching: push contact on the target's right face (right arm) toward the partner, brace pre-contact +
        contact of the closed left gripper on the partner's far (left) side; both IK-checked (§4.6 step 6)."""
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        t_pose = draft.goal_poses[tname]
        p_now = self.obj("partner").get_position_orientation()
        p_pose = (p_now[0].clone().float(), p_now[1].clone().float())
        tangent = manip.tangent
        direction = th.tensor([float(tangent[0]), float(tangent[1]), 0.0])  # toward the partner = robot's left
        p_init, b_point = self.touching_points(t_pose, p_pose, manip)
        dist = float(self.pair.get("g_NT", 0.02)) + 0.01
        push_init = IK.contact_eef(p_init, direction)
        push_goal = (push_init[0] + direction * dist, push_init[1])
        push_pre = IK.pre_pose(push_init)
        p_hull = self.world.hull_at(self.hull["partner"], p_pose)
        face_z = (float(p_hull[:, 2].min()), float(p_hull[:, 2].max()))
        # closed pads for both contacts (§3.6); the push arm first, then the brace with the push arm locked (§4.6 step 6),
        # brace candidates in order (pads on the far face / top-down blade beside it, ``touching_brace_candidates``)
        why: List[str] = []
        chosen = None
        for pre, contact, kind, extras in self.touching_brace_candidates(manip, b_point, face_z, -direction):
            res = self.ikc.exact_both_arms(("right", [push_pre, push_init, push_goal]), ("left", [pre, contact]), manip,
                                           self.support_objects() + [self.obj("target")], self.support_objects() + [self.obj("partner")], fingers=IK.FINGER_CLOSED)
            if res.ok:
                chosen = (pre, contact, kind, extras)
                break
            why.append(f"{kind}: {res.reason}")
            if res.reason.startswith("right"):
                break  # the push chain itself fails: no brace candidate can change that
        if chosen is None:
            raise Reject("contact_unreachable", f"touching push / brace: {'; '.join(why[-3:]) or 'no brace candidate (partner face reaches the palm section)'}")
        pre, contact, kind, extras = chosen
        self.sidecar.add_contact(ContactRecord("push", "right", push_pre, push_init, None, push_goal, direction, dist, object=tname))
        self.sidecar.add_contact(TT.BraceRecord("brace", "left", pre, contact, direction=-direction, brace_kind=kind, object=pname, extras=extras))
        draft.summary["brace"] = {"kind": kind, **{k: v for k, v in extras.items() if v is not None}}
        manip.arms = ["right", "left"]

    def touching_brace_candidates(self, manip: Stance, b_point: th.Tensor, face_z: Tuple[float, float], a_face: th.Tensor):
        """Brace poses of the closed left gripper at the partner's far (robot-left) face, in the order to try — the axial
        stop-brace rule instead of the former 12 mm pad-slab at the face point: (1) the pads on the face, pad centre
        ``STOP_CLEARANCE`` outside it (closing axis vertical then horizontal, either wrist turn); (2) the top-down blade
        beside the face (``tabletop.blade_standoff``: eef origin ``finger half thickness over the face height above the
        tips + STOP_CLEARANCE`` outside the face, approach pitched per ``tabletop.approach_pitch_order``, the pads' faces
        parallel to the object face). Thin partners (< ``THIN_PARTNER_HEIGHT``) try the blade first. Yields ``(pre, contact,
        kind, extras)`` with the brace record fields ``blade_half_thickness_m`` / ``stop_clearance_m`` / ``blade_standoff_m`` /
        ``pose_class``."""
        a = C.unit(G.t(a_face))  # into the face
        outward = -a
        z_axis = th.tensor([0.0, 0.0, 1.0])
        side_h = C.unit(th.cross(z_axis, outward, dim=0))
        z0, z1 = face_z
        clearance = float(self.gen("stop_clearance_m", STOP_CLEARANCE))

        def side_contacts():
            point = G.t(b_point) + outward * clearance
            for c_axis, suffix in ((z_axis, "v"), (side_h, "h")):
                name = TT.contact_class(manip, a, c_axis) or f"side_{suffix}"
                for c_sign in (1.0, -1.0):
                    contact = IK.contact_eef(point, a, c_sign * c_axis)
                    yield IK.pre_pose(contact), contact, f"{name}{'' if c_sign > 0 else '-'}", {"stop_clearance_m": clearance, "pose_class": name, "blade_half_thickness_m": None}

        def blades():
            for pitch in TT.approach_pitch_order(None):
                z_tip = z0 + 0.005  # the fingertips just above the support: the fingers span the face height
                so = TT.blade_standoff((z0, z1), z_tip, pitch, clearance, side=False)
                if so is None:
                    continue  # the face reaches the palm section: no blade contact
                standoff, half = so
                a_vec = TT.pitched_approach(pitch, manip.forward)
                origin = th.tensor([float(b_point[0]), float(b_point[1]), z_tip + C.FINGERTIP_LENGTH * math.cos(math.radians(pitch))]) + outward * standoff
                for c_sign in (1.0, -1.0):
                    contact = (origin, C.eef_quat(a_vec, c_sign * outward))
                    yield IK.pre_pose(contact), contact, f"blade_{pitch:.0f}{'' if c_sign > 0 else '-'}", {
                        "blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": z_tip, "pose_class": f"blade_{pitch:.0f}"}

        order = (blades, side_contacts) if (z1 - z0) < THIN_PARTNER_HEIGHT else (side_contacts, blades)
        for gen in order:
            yield from gen()

    def _nudge_until_touching(self, tname: str, pname: str, step: float = 0.001, max_steps: int = 25):
        from omnigibson.object_states import Touching

        t, p = self.objects[tname], self.objects[pname]
        direction = self.sidecar.stance("manipulation").tangent
        for _ in range(max_steps):
            self.world.wake([t, p])
            if t.states[Touching].get_value(p):
                return
            pos, quat = t.get_position_orientation()
            t.set_position_orientation(pos + th.tensor([float(direction[0]) * step, float(direction[1]) * step, 0.0]), quat)
            t.keep_still()
            self.world.step_physics(3, still=[self.robot, p])
        raise Reject("goal_infeasible", "could not establish contact for touching")

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        tname = self.bindings.one("target").name
        g = self.sidecar.grasp_of(tname)
        manip = self.sidecar.stance("manipulation")
        recs: List[Dict[str, Any]] = []
        mc = self.choice.models["target"]
        if g is not None and manip.focus is not None:
            recs.append({"scope": {"target": f"{mc.category}/{mc.model}", "support_height": round(float(self.top.z - self.world.floor_z), 2)},
                         "type": "grasp_set", "trunk": [manip.translate, round(manip.tilt, 2)],
                         "cell": [round(v, 2) for v in manip.world_to_base(manip.focus[:2]).tolist()],
                         "feasible": [[round(g.yaw, 2), g.grasp_name]], "hard": False})
        pname = self.bindings.one("partner").name
        depth_t, lat_t = manip.local(draft.object_poses[tname][0])
        depth_p, lat_p = manip.local(draft.object_poses[pname][0])
        recs.append({"scope": {"scene": self.scene, "support": self.support.name, "target": f"{mc.category}/{mc.model}",
                               "partner": f"{self.choice.models['partner'].category}/{self.choice.models['partner'].model}"},
                     "type": "placement", "literal": f"({self.relation} target partner)", "depth": [round(depth_t, 2), round(depth_t, 2)],
                     "lateral": [round(lat_t, 2), round(lat_t, 2)], "partner_lateral": [round(lat_p, 2), round(lat_p, 2)], "hard": False})
        if getattr(self, "bands", None) is not None and getattr(self.bands, "path", None):
            recs.append({"scope": {"scene": self.scene, "support": self.support.name, "target": f"{mc.category}/{mc.model}"}, "type": "band_ref",
                         "path": str(self.bands.path), "hard": False})
        return recs

    # ------------------------------------------------------------------ helpers

    def _container_volume(self, obj, p_pose: G.Pose) -> Tuple[th.Tensor, float]:
        """(container centre — the fillable meta link's xy when the model has one, else the hull centre —, rim z = the
        partner hull's highest point at ``p_pose``)."""
        from omnigibson.macros import macros

        p_hull = self.world.hull_at(self.hull["partner"], p_pose)
        centre = (p_hull.max(dim=0).values + p_hull.min(dim=0).values) / 2
        for link in obj.links.values():
            if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
                # the meta link's offset in the root frame from the current sim poses, applied at ``p_pose`` (the partner may
                # still be parked when the goal is first computed)
                root = obj.get_position_orientation()
                p, _ = link.get_position_orientation()
                local = G.invert((G.t(root[0]), G.t(root[1])))
                p_local = C.transform_points(local, G.t(p).unsqueeze(0))[0]
                p_world = C.transform_points((G.t(p_pose[0]), G.t(p_pose[1])), p_local.unsqueeze(0))[0]
                centre = th.tensor([float(p_world[0]), float(p_world[1]), float(centre[2])])
                break
        return centre, float(p_hull[:, 2].max())
