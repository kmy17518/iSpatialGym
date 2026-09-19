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
from contextlib import contextmanager
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
from ispatialgym.sampling.v2.core import SamplerV2, annotate_height_extra, contact_height_extra, register, requested_height_extra
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, GraspRecord, Stance

RETREAT_DIST_M = 0.10  # oracle ``actions.RETREAT_DIST``: a place ends this far back along the approach (release + lift-off)
RETREAT_LITTLE_M = 0.05  # oracle ``actions.RETREAT_LITTLE``: the retreat after a place

#: Wrist turns of a blade push in the order to try (owner 2026-09-18 19:39): the wrist camera (``right_realsense_link``, three
#: r 1.8 cm spheres 4-7 cm to the side of the gripper at the palm-line height) lies on the eef's -y side; ``eef_quat(a, c·outward)``
#: puts eef +y = c·outward, so c = +1 hangs the camera over the pushed object's top and c = -1 turns it away from it. The camera
#: contacting the box top pushed the robot down against the table (demos 0918-190236 / 191811: robot and table shaking).
PUSH_WRIST_TURNS = (-1.0, 1.0)
#: Hand geometry beyond the finger profile, sphere centres in the EEF frame (gripper-link spheres of
#: ``r1pro_description_curobo_default.yaml`` mapped by ``EEF_IN_GRIPPER`` (0, 0, -0.06; rpy 0, pi, 0): p_eef = (-x, y, -(z + 0.06)))
#: — what the IK collision check does not see against the pushed object (ignored at the contact pose, §3.6).
HAND_SPHERES_EEF: Dict[str, Tuple[Tuple[Tuple[float, float, float], ...], float]] = {
    "wrist camera": (((-0.056, -0.002, -0.076), (-0.056, -0.030, -0.076), (-0.056, -0.057, -0.076)), 0.018),
    "knuckle": (((-0.015, -0.042, -0.0675), (0.015, -0.042, -0.0675), (-0.015, 0.042, -0.0675), (0.015, 0.042, -0.0675)), 0.015),
    "gripper body": (((0.0, 0.0, -0.097),), 0.04),
}

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
    """Base-frame cells of the left-arm generic CONTACT band (closed gripper, brace approach; one band per object height
    class — B3 A1) at ``trunk``, through ``tabletop.load_left_contact_band`` (B3's ``ReachBands.load(scene, support,
    arm="left", contact_set="generic_contact", height=h)``, nearest class within ±4 cm, in-process compute as the last
    resort; cached on ``s.bands_left``). None only when no band can be had — then the key looked for and the left bands
    present for the support are logged and the caller records the gate as unavailable."""
    bands = None
    try:
        bands = TT.load_left_contact_band(s, float(height), top=s.top, edge=s.edge, compute=True)
    except Exception as e:  # noqa: BLE001 - loader / compute failure: recorded, the exact two-arm IK stays the gate
        log.warning("left contact band for height %.3f unavailable: %r", float(height), e)
    if bands is None:
        import glob
        import os

        cls = TT.height_class(float(height))
        root = os.environ.get("ISPATIALGYM_REACH_BANDS", "runlogs/reach_bands_20260915/bands")
        present = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, f"contact__{s.world.scene_model}__{s.top.name}__left_*.npz")))
        log.warning("no left contact band: key scene=%s support=%s arm=left contact_set=%s height=%.3f (class %.2f); left bands present: %s",
                    s.world.scene_model, s.top.name, getattr(TT, "CONTACT_SET", "generic_contact"), float(height), cls, present or "none")
        return None
    cells = bands.band(trunk[0], trunk[1])
    log.info("left contact band %s: %d cells at trunk (%.2f, %.2f)", getattr(bands, "path", "?"), len(cells), trunk[0], trunk[1])
    return cells


#: §6.11 touching admissibility numbers (pushability / stability; ``generation`` overrides)
PUSH_MAX_PINCH = 0.089
PUSH_MAX_LONG = 0.30
PUSH_MAX_HEIGHT = 0.30
PARTNER_MAX_ASPECT = 2.5
#: inside (owner decision 2026-09-16): the open gripper's lowest point clears the partner's rim by this at the release
RELEASE_CLEARANCE = 0.02
#: ontop / under (owner item 3): the object is released at most this far above the actual surface under its footprint
#: centre — half the well depth, at least ``RELEASE_ABOVE_SURFACE_MIN`` (a flat plate)
RELEASE_ABOVE_SURFACE = 0.01
RELEASE_ABOVE_SURFACE_MIN = 0.003
#: library grasps whose closing axis is inclined more than this to the horizontal (and the vertical) in the target's rest
#: frame rotate the object in the hand until the pinch is flush (dropped while a flush grasp remains)
CLOSING_TILT_MAX_DEG = 12.0
#: a target that rolled more than this while settling gets its grasp re-selected at the settled pose
REGRASP_AFTER_ROLL_DEG = 10.0
#: the open gripper at the grasp configuration keeps this clear of the support (closure probe: a pad on the table makes
#: the closure asymmetric and knocks the object)
PAD_SUPPORT_CLEARANCE = 0.002
#: the pinch centre may sit this far from the object's centre along the closing axis; farther pinches are re-centred
PINCH_OFFSET_MAX = 0.005
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
        self.release_surface_z: Optional[float] = None  # ontop / under: the raycast surface under the goal footprint centre
        self.release_above_m: Optional[float] = None  # ontop / under: the release height above that surface
        self._touching_place_pos: Optional[th.Tensor] = None  # touching: the nextto position the program places at
        self._brace_gate: Optional[Dict[str, Any]] = None  # touching: the A1 joint band gate verdict of the drawn partner position
        self.grasps_flush: List[Any] = []  # the flush-pinch tier of ``self.grasps`` (tried first at every focus)
        self._reselected_q: Optional[th.Tensor] = None  # grasp configuration of the last re-selected grasp
        self.release_sweep: Optional[Dict[str, Any]] = None  # nextto / touching: open-jaw footprint facts of the place

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
        # pinch centring (oracle-B, run 0916-202403: a pinch off the object's centre along the closing axis lets one pad reach
        # the object first and drag it until the other lands - +15.4 mm at the closure): the pinch centre is moved onto
        # the hull's centre along c when it sits farther than ``pinch_offset_max_m`` from it (the pads close until contact,
        # so the shift along c changes nothing else); the original offset is recorded per grasp
        self.grasps, offsets = self.recentre_pinches(self.grasps, tgt)
        self.template_facts["grasp_pinch_offset_m"] = {k: round(v, 4) for k, v in offsets.items()}
        self.template_facts["grasps_recentred"] = [k for k, v in offsets.items() if abs(v) > float(self.gen("pinch_offset_max_m", PINCH_OFFSET_MAX))]
        # flush pinch (teacup demo 2026-09-16 14:44: grasp_3's closing axis was inclined 28 deg to the horizontal; the flat
        # pads squeezed the vertical cup wall at two heights, the cup rotated in the hand until the pinch was flush - 16-32
        # deg in the hand, 21 deg at the release, at rest 10 deg on the well slope). The target is placed at its rest
        # orientation, so a grasp's closing axis must be horizontal (or vertical) in the rest frame within
        # ``closing_tilt_max_deg``; inclined pinches are dropped while a flush grasp remains. The pinch line's distance to the
        # centroid (``pivot_arm``, gravity torque about the pinch) is recorded per grasp as well.
        tilts = {g.name: self.closing_tilt_deg(g.pose) for g in self.grasps}
        self.template_facts["grasp_closing_tilt_deg"] = {k: round(v, 1) for k, v in tilts.items()}
        self.template_facts["grasp_pivot_arms_m"] = {g.name: round(self.pivot_arm(g.pose), 4) for g in self.grasps}
        self.template_facts["grasp_pinch_vs_long_axis_deg"] = {g.name: round(self.pinch_vs_long_axis_deg(g.pose), 1) for g in self.grasps}
        tilt_max = float(self.gen("closing_tilt_max_deg", CLOSING_TILT_MAX_DEG))
        # two-tier grasp set: the flush pinches are tried first at every focus (``sample_instance``), the inclined ones only
        # when the flush set has no feasible (yaw, grasp) there - a round apple keeps its grasps (the earlier hard filter left
        # F empty), a cup gets a flush pinch whenever one is reachable (the 18:41 teacup drew grasp_3, 28 deg, and only the
        # corrected release kept it upright)
        flush = [g for g in self.grasps if tilts[g.name] <= tilt_max]
        self.grasps_flush = flush if flush else list(self.grasps)
        self.template_facts["grasps_flush_first"] = [g.name for g in self.grasps_flush]
        self.template_facts["grasps_inclined_fallback"] = [g.name for g in self.grasps if g not in self.grasps_flush]
        if not flush:
            log.info("%s: no library grasp pinches flush (closing tilts %s deg > %.0f); single tier", self.grasp_key,
                     {k: round(v, 1) for k, v in tilts.items()}, tilt_max)
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

    def apply_height_extra_to_grasp(self, grasp: GraspRecord, t_pose: G.Pose, manip: Stance) -> None:
        """A/B knob ``generation.contact_height_extra_m`` on the grasp (owner 2026-09-16 23:00): the pinch moves up the object
        by the effective lift — the fingertip only until it is >= x above the support, the pad centre never above the hull
        top minus 5 mm (``core.contact_height_extra(kind="grasp")``) — by shifting ``eef_in_object`` along world up in the
        object frame, so every derived pose (grasp / pre-grasp at the settled pose, place = goal ∘ T_obj→eef) carries it;
        the lifted pre-grasp / grasp must still pass the exact IK, else the library height is kept. Facts on the record."""
        x = requested_height_extra(self)
        if x <= 0.0 or grasp.eef_in_object is None or grasp.grasp is None:
            grasp.extras.update({"height_extra_m": x, "height_extra_effective_m": 0.0, "height_extra_clip_reason": "off" if x <= 0 else "no grasp pose"})
            return
        tip_z = float(C.fingertip((G.t(grasp.grasp[0]), G.t(grasp.grasp[1])))[2])
        pad_z = float(C.pad_center((G.t(grasp.grasp[0]), G.t(grasp.grasp[1])))[2])
        hull_w = self.world.hull_at(self.hull["target"], t_pose)
        eff, why = contact_height_extra(self, "grasp", fingertip_z=tip_z, support_z=float(self.top.z), band_top_z=float(hull_w[:, 2].max()), pad_center_z=pad_z)
        if eff > 0.0:
            R_obj = C.quat_to_mat(G.t(t_pose[1]))
            up_obj = R_obj.T @ th.tensor([0.0, 0.0, 1.0])  # world up in the object frame
            eef_in_object = (G.t(grasp.eef_in_object[0]) + eff * up_obj, G.t(grasp.eef_in_object[1]))
            eef = IK.grasp_pose(t_pose, eef_in_object)
            pre = IK.pre_pose(eef)
            res = self.ikc.exact([pre, eef], manip, grasp.arm, ignore_for_last=self.support_objects(), scale_key=self.choice.models["target"].scale, cache=False,
                                 approached=[self.obj("target")])
            if res.ok:
                grasp.eef_in_object, grasp.grasp, grasp.pre_grasp = eef_in_object, eef, pre
            else:
                eff, why = 0.0, f"IK failed with the lift ({res.reason}); library height kept"
        grasp.extras.update({"height_extra_m": x, "height_extra_effective_m": round(eff, 4), "height_extra_clip_reason": why})

    def phase3_two_tier(self, manip: Stance, rng: np.random.Generator):
        """§4.6 step 1 with the flush pinches first: ``tabletop.phase3_target`` over ``grasps_flush``; when that finds no
        feasible (yaw, grasp) at the focus and inclined grasps exist, the same draw over every attachable grasp."""
        tiers = [self.grasps_flush]
        if len(self.grasps_flush) < len(self.grasps):
            tiers.append(self.grasps)
        last = None
        for i, gs in enumerate(tiers):
            try:
                t_pose, grasp = TT.phase3_target(self, manip, rng, gs, self.hull["target"], self.rest["target"], self.support_objects(),
                                                 top=self.top, scale_key=self.choice.models["target"].scale)
                if i > 0:
                    log.info("%s: no flush pinch feasible at the focus; inclined grasp %s drawn", self.gp.alias, grasp.grasp_name)
                return t_pose, grasp
            except Reject as e:
                last = e
                if e.reason != "grasp_unreachable":
                    raise
        raise last

    def pinch_offset(self, eef_in_object: G.Pose) -> float:
        """Signed offset (m) of the pinch centre (pad centre) from the target hull's centre along the closing axis c, in the
        object frame: the pads close symmetrically about it, so an offset lets the nearer pad touch first and drag the object."""
        pos, quat = G.t(eef_in_object[0]), G.t(eef_in_object[1])
        R = C.quat_to_mat(quat)
        a, c = R[:, 2], R[:, 1]
        pc = pos + C.CONTACT_CENTER_Z * a
        proj = self.hull["target"] @ c
        mid = 0.5 * float(proj.min() + proj.max())
        return float(th.dot(pc, c)) - mid

    def recentre_pinches(self, grasps: Sequence[Any], tgt) -> Tuple[List[Any], Dict[str, float]]:
        """Library grasps with the pinch centre moved onto the hull centre along the closing axis when the offset exceeds
        ``pinch_offset_max_m`` (a copy of the library record with the shifted position; the attachability screen is re-run on
        it, a grasp that stops being attachable keeps its original pose). Returns ``(grasps, {name: original offset})``."""
        import copy

        limit = float(self.gen("pinch_offset_max_m", PINCH_OFFSET_MAX))
        out: List[Any] = []
        offsets: Dict[str, float] = {}
        for g in grasps:
            off = self.pinch_offset(g.pose)
            offsets[g.name] = off
            if abs(off) <= limit:
                out.append(g)
                continue
            c = C.quat_to_mat(G.t(g.pose[1]))[:, 1]
            shifted = copy.copy(g)
            shifted.position = G.t(g.pose[0]) - off * c
            if self.ikc.attachable(tgt, shifted.pose, "right"):
                log.info("%s: grasp %s pinch centre %+.1f mm off the object's centre along the closing axis - re-centred", self.grasp_key, g.name, off * 1000)
                out.append(shifted)
            else:
                log.info("%s: grasp %s pinch %+.1f mm off centre; the re-centred pose is not attachable - kept as is", self.grasp_key, g.name, off * 1000)
                out.append(g)
        return out, offsets

    def pinch_vs_long_axis_deg(self, eef_in_object: G.Pose) -> float:
        """Angle (deg, 0-90) between a library grasp's pinch line (closing axis c) and the target's long axis — the largest
        AABB extent in the rest frame (a lying cylinder's axis). 90 = the pads close across the cylinder, 0 = along it."""
        c_obj = C.quat_to_mat(G.t(eef_in_object[1]))[:, 1]
        c_rest = G.rotate(G.t(self.rest["target"]), c_obj)
        ext = TT.hull_extents(self.hull["target"], self.rest["target"])
        axis = th.zeros(3)
        axis[int(th.argmax(ext))] = 1.0
        d = min(1.0, abs(float(th.dot(c_rest / th.linalg.norm(c_rest).clamp(min=1e-9), axis))))
        return math.degrees(math.acos(d))

    def grasp_facts(self, grasp: GraspRecord) -> Dict[str, Any]:
        """Facts of the recorded grasp for the closure investigation (agent 52b4fb5f): ``pinch_vs_long_axis_deg``,
        ``pad_height_above_support_m`` (pad centre z at the grasp pose - support top z), ``pad_height_above_axis_m`` (the same
        minus the object's radius = half its rest-frame height: a lying cylinder's axis), ``object_radius_m``,
        ``closing_tilt_deg`` and ``pivot_arm_m``. None where a value is not available."""
        out: Dict[str, Any] = {}
        if grasp.eef_in_object is None:
            return out
        out["pinch_vs_long_axis_deg"] = round(self.pinch_vs_long_axis_deg(grasp.eef_in_object), 1)
        out["pinch_offset_m"] = round(self.pinch_offset(grasp.eef_in_object), 4)
        out["pinch_offset_original_m"] = round(float((self.template_facts.get("grasp_pinch_offset_m") or {}).get(grasp.grasp_name, out["pinch_offset_m"])), 4)
        out["closing_tilt_deg"] = round(self.closing_tilt_deg(grasp.eef_in_object), 1)
        out["pivot_arm_m"] = round(self.pivot_arm(grasp.eef_in_object), 4)
        ext = TT.hull_extents(self.hull["target"], self.rest["target"])
        radius = 0.5 * float(ext[2])
        out["object_radius_m"] = round(radius, 4)
        if grasp.grasp is not None and self.top is not None:
            pad_z = float(C.pad_center((G.t(grasp.grasp[0]), G.t(grasp.grasp[1])))[2])
            out["pad_height_above_support_m"] = round(pad_z - float(self.top.z), 4)
            out["pad_height_above_axis_m"] = round(pad_z - float(self.top.z) - radius, 4)
        return out

    def closing_tilt_deg(self, eef_in_object: G.Pose) -> float:
        """Inclination (deg) of a library grasp's closing axis to the horizontal *or* the vertical, whichever is nearer, in the
        target's rest frame: 0 = a flush pinch of a vertical wall (c horizontal) or of the top / bottom faces (c vertical)."""
        c_obj = C.quat_to_mat(G.t(eef_in_object[1]))[:, 1]
        c_world = G.rotate(G.t(self.rest["target"]), c_obj)
        cz = min(1.0, abs(float(c_world[2])))
        from_horizontal = math.degrees(math.asin(cz))
        return min(from_horizontal, 90.0 - from_horizontal)

    def pivot_arm(self, eef_in_object: G.Pose) -> float:
        """Distance (m) from the target hull's centroid to the pinch line of a library grasp: the pad centre
        (``eef origin + CONTACT_CENTER_Z * a``) along the closing axis c, both in the object frame. Small = the pinch carries
        the object without a gravity torque about the closing axis (the object does not pivot in the hand)."""
        pos, quat = G.t(eef_in_object[0]), G.t(eef_in_object[1])
        R = C.quat_to_mat(quat)
        a, c = R[:, 2], R[:, 1]
        pc = pos + C.CONTACT_CENTER_Z * a
        hull = self.hull["target"]
        centroid = 0.5 * (hull.min(dim=0).values + hull.max(dim=0).values)
        d = centroid - pc
        perp = d - th.dot(d, c) * c
        return float(th.linalg.norm(perp))

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
                t_pose, grasp = self.phase3_two_tier(manip, rng)
                self.apply_height_extra_to_grasp(grasp, t_pose, manip)  # A/B knob generation.contact_height_extra_m (0 = no-op)
                # §4.6 step 2: partner on a random side (touching: robot's left) at gap g ∈ [g_lo, D_p]
                p_pose = self.place_partner(rng, t_pose, manip, cam_init)
                if self.edge_aligned():
                    d.summary["edge_alignment"] = {
                        "edge_yaw_deg": round(math.degrees(math.atan2(float(manip.tangent[1]), float(manip.tangent[0]))), 1),
                        "target_inhand_yaw_deg": round(math.degrees(self.edge_yaw_delta(self.hull["target"], t_pose[1], manip)), 1),
                        "partner_yaw_deg": round(math.degrees(G.yaw_of(p_pose[1])), 1),
                        "partner_residual_deg": round(math.degrees(self.edge_yaw_delta(self.hull["partner"], p_pose[1], manip)), 2),
                    }
                # §4.6 step 4: every object fully in frame (5 %) in the init image
                for role, pose in (("target", t_pose), ("partner", p_pose)):
                    if not TT.in_frame(self, cam_init, TT.hull_corners_world(self.hull[role], pose)):
                        raise Reject("not_in_frame", f"{role} not in the init image (5 %)")
                grasp.place_on = pname
                self.sidecar.add_stance(manip)
                self.sidecar.add_grasp(grasp)  # (popped again by the except below when this candidate is rejected)
                # the place / pre-place of this candidate must project into its manipulation image (owner 1b: the FOV
                # monitor arms from the pre-pose): checked here, before the settle, so the next candidate is tried instead of
                # the attempt dying in the goal check (inside: release rim + 4-5 cm and the pre-place 10 cm above it)
                if self.relation != "touching":
                    cam_manip = self.camera_pose_at(manip)
                    goal_guess = self.goal_pose(t_pose, p_pose)
                    place_guess = IK.grasp_pose(goal_guess, grasp.eef_in_object)
                    for what, pose in (("place", place_guess), ("pre-place", IK.pre_pose(place_guess))):
                        if not self.ikc.eef_in_frame(pose, cam_manip):
                            raise Reject("gripper_not_in_frame", f"{what} eef pose outside the manipulation image (5 %) for candidate {int(k)} "
                                                                 f"(trunk {manip.translate}, {manip.tilt:.2f})")
                # (owner 2026-09-18 01:33, (a)) reach screening of the place at the draw: the pre-place -> place descent with the
                # object attached must pass the exact IK (margin + reach quality) from THIS candidate's stance - else the
                # next candidate is tried instead of the attempt dying in the goal check 20-60 s later (touching re-samples
                # 0917-235350 .. 0918-011947: 5/8 attempts lost to ik_joint_limit / witness residual at the place descent)
                if bool(self.gen("reach_screen_place", True)):
                    obj_at_place = self.place_pose_for(t_pose, p_pose, manip) if self.relation == "touching" else self.goal_pose(t_pose, p_pose)
                    place_guess = IK.grasp_pose(obj_at_place, grasp.eef_in_object)
                    ign_place = self.support_objects() + ([self.obj("partner")] if self.relation != "inside" else [])
                    # touching (owner 2026-09-18 16:32): the place is made from the second base pose the box is carried to
                    st_place = (self.push_stance_for(manip, obj_at_place) if self.relation == "touching" else None) or manip
                    with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
                        res_place = self.ikc.exact([IK.pre_pose(place_guess), place_guess], st_place, grasp.arm, ignore_for_last=ign_place,
                                                   scale_key=self.choice.models["target"].scale, cache=False, approached=ign_place)
                    if not res_place.ok:
                        raise Reject(res_place.code or "place_unreachable", f"place screening at the draw (candidate {int(k)}, trunk {manip.translate}, "
                                                                            f"{manip.tilt:.2f}, from the {st_place.name} stance): {res_place.reason}")
                # carry height: the target (bottom at the table at init) clears the partner's top by 3 cm on the way
                grasp.carry_height = max(float(self.gen("carry_clearance_m", 0.03)), TT.height_of(self.hull["partner"], self.rest["partner"]) + float(self.gen("carry_clearance_m", 0.03)))
                d.object_poses[tname] = t_pose
                d.object_poses[pname] = p_pose
                d.robot_q = self.stance_q(init)
                d.base_pose2d = tuple(init.base_pose2d)
                d.summary.update({"relation": self.relation, "manipulation_candidate": int(k), "grasp": {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw},
                                  "manipulation_stance": manip.to_meta(), "pair": self.pair})
                d.pose_targets[pname] = self._pose_target("partner", p_pose)
                d.pose_targets[tname] = self._pose_target("target", t_pose)
                d.summary["target_draft_pose"] = [[float(v) for v in t_pose[0]], [float(v) for v in t_pose[1]]]
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

    def place_gap(self) -> float:
        """AABB gap between target and partner at the *place* position: nextto ``g_NT - 0.01`` (the predicate holds below
        g_NT); touching = the same nextto position — the oracle program places there and the push closes the gap to
        contact (``push_distance``)."""
        return max(0.0, float(self.pair.get("g_NT", 0.03)) - 0.01)

    def push_margin(self) -> float:
        """Touching: how far past the measured gap the recorded push reaches - the place position tolerance (the executed
        place may leave the target that much farther from the partner: oracle-B run 0916-202403, boxes 4.1 cm apart at a
        2.7 cm recorded push) plus 5 mm of pressing. The oracle pushes the residual it measures; the recorded arc is its cap."""
        return float(self.tol("position", 0.03)) + 0.005

    def push_distance(self, gap_at_place: Optional[float] = None) -> float:
        """Touching: the recorded push spans the gap between the target's near face at the place and the partner's face —
        measured from the hulls when given, else the nominal place gap — plus ``push_margin`` (``touching_contacts``:
        ``push_goal = push_init + direction * push_distance``)."""
        gap = self.place_gap() if gap_at_place is None else max(0.0, float(gap_at_place))
        return gap + self.push_margin()

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
        elif self.relation == "touching" and bool(self.gen("push_stance", True)) and bool(self.gen("reach_screen_place", True)):
            # touching with the push stance (owner 2026-09-18 16:32): the place is judged by the exact-IK screening of the
            # draw (``reach_screen_place``) and the push by its own stance - the band cut on the place gap is not applied
            ranges = {sd: (g_lo, g_hi) for sd in sides}
        elif getattr(self, "bands", None) is not None:
            # nextto / touching: the target's *place* position (its origin at the goal gap — touching: the nextto position
            # the push starts from) must be a band lateral at the target's depth, else the place IK of §6.11 fails
            # (touching box_of_sanitary_napkins: every attempt place_unreachable with the goal 4-9 cm left of centre)
            cells = self.bands.band(manip.translate, manip.tilt)
            base_t = manip.world_to_base(t_pose[0][:2])  # band cells are base-frame (x ahead, lateral + left)
            depth_t, lat_t = float(base_t[0]), float(base_t[1])
            near = cells[np.abs(cells[:, 0] - depth_t) <= 0.12] if len(cells) else cells
            gap_goal = self.place_gap()
            if len(near):
                lam_min, lam_max = float(near[:, 1].min()), float(near[:, 1].max())
                for sd in sides:
                    if sd > 0:  # partner on the robot's left: the target moves left by g - gap_goal
                        lo, hi = lam_min - lat_t + gap_goal, lam_max - lat_t + gap_goal
                    else:
                        lo, hi = lat_t + gap_goal - lam_max, lat_t + gap_goal - lam_min
                    lo, hi = max(lo, g_lo), min(hi, g_hi)
                    if lo <= hi:
                        ranges[sd] = (lo, hi)
            if not ranges:
                raise Reject("partner_placement", f"no side keeps the target's place position inside the band (λ ∈ [{lam_min:.2f}, {lam_max:.2f}] at depth "
                                                  f"{depth_t:.2f}, target at {lat_t:+.2f}) with g ∈ [{g_lo:.2f}, {g_hi:.2f}]" if len(near) else "band has no cell at the target's depth")
        else:
            ranges = {sd: (g_lo, g_hi) for sd in sides}
        side = list(ranges)[int(rng.integers(len(ranges)))]
        if self.relation != "touching":
            return TT.place_partner(self, rng, t_pose, self.hull["target"], self.hull["partner"], self.rest["partner"], manip, side, ranges[side],
                                    top=self.top, cam=cam, margin=self.margin_support, min_gap=self.gap_min)
        # A1 with the brace optional (owner 2026-09-16 17:10, the axial rule): a partner position whose push work point (the
        # target's right face at the *place* position) lies in the right band and whose brace point (partner's far / left
        # face) lies in the left contact band of the same trunk is preferred and enables the brace; otherwise the last
        # drawn position is kept and the instance becomes push-only (``self._brace_gate`` tells ``touching_contacts``)
        right_cells = self.bands.band(manip.translate, manip.tilt) if getattr(self, "bands", None) is not None else None
        left_cells = left_contact_band_cells(self, (manip.translate, manip.tilt), TT.height_of(self.hull["partner"], self.rest["partner"]))
        self.template_facts["left_contact_band"] = "unavailable" if left_cells is None else int(len(left_cells))
        errors = []
        p_pose = None
        # edge alignment (owner 2026-09-18 13:50): the partner's footprint axes along the support edge - the two aligned
        # footprints (long side along / across the edge) alternate over the tries; None = the random yaw of the record
        yaw0 = self.edge_yaw_delta(self.hull["partner"], self.rest["partner"], manip) if self.edge_aligned() else None
        for i in range(8):
            yaw_i = None if yaw0 is None else yaw0 + (i % 2) * (math.pi / 2)
            p_pose = TT.place_partner(self, rng, t_pose, self.hull["target"], self.hull["partner"], self.rest["partner"], manip, side, ranges[side],
                                      top=self.top, cam=cam, margin=self.margin_support, min_gap=self.gap_min, yaw=yaw_i)
            if left_cells is None or right_cells is None:
                self._brace_gate = {"ok": None, "why": "left contact band unavailable"}  # unknown: the two-arm IK decides
                return p_pose
            work, brace = self.touching_points(self.place_pose_for(t_pose, p_pose, manip), p_pose, manip)
            w_b, b_b = manip.world_to_base(work[:2]), manip.world_to_base(brace[:2])
            if brace_placement_admissible(w_b, b_b, right_cells, left_cells):
                self._brace_gate = {"ok": True, "work_base": [round(float(v), 3) for v in w_b], "brace_base": [round(float(v), 3) for v in b_b],
                                    "right_cells": int(len(right_cells)), "left_cells": int(len(left_cells))}
                return p_pose
            errors.append(f"work {[round(float(v), 2) for v in w_b]} {'in' if in_band(w_b, right_cells) else 'OFF'} right band / brace "
                          f"{[round(float(v), 2) for v in b_b]} {'in' if in_band(b_b, left_cells) else 'OFF'} left band")
        self._brace_gate = {"ok": False, "why": f"none of 8 partner positions puts the push point in the right band ({len(right_cells)} cells) and the brace point "
                                                f"in the left contact band ({len(left_cells)} cells) at trunk ({manip.translate}, {manip.tilt:.2f}); e.g. {errors[-2:]}"}
        return p_pose

    # ------------------------------------------------------------------ edge alignment (owner 2026-09-18 13:50 / 14:28)

    def edge_aligned(self) -> bool:
        """Touching (owner 2026-09-18 13:50): the partner is spawned with its footprint axes along the support edge and the
        target's goal / place orientation is the edge-aligned yaw nearest its init yaw (an in-hand rotation <= 45 deg) -
        the boxes meet face to face; ``generation.edge_aligned`` (default on for touching)."""
        return self.relation == "touching" and bool(self.gen("edge_aligned", True))

    def footprint_axis_angle(self, hull: th.Tensor, quat) -> float:
        """World angle (rad, modulo pi/2) of the minimum-area bounding rectangle of the hull's footprint at ``quat``."""
        pts = self.world.footprint(hull, (th.zeros(3), G.t(quat)))
        best = None
        n = len(pts)
        for i in range(n):
            e = pts[(i + 1) % n] - pts[i]
            if float(th.linalg.norm(e)) < 1e-6:
                continue
            ang = math.atan2(float(e[1]), float(e[0]))
            c, s = math.cos(ang), math.sin(ang)
            u = pts[:, 0] * c + pts[:, 1] * s
            v = -pts[:, 0] * s + pts[:, 1] * c
            area = float((u.max() - u.min()) * (v.max() - v.min()))
            if best is None or area < best[0]:
                best = (area, ang)
        return 0.0 if best is None else best[1] % (math.pi / 2)

    def edge_yaw_delta(self, hull: th.Tensor, quat, manip: Stance) -> float:
        """Smallest world yaw (rad, in [-pi/4, pi/4]) that turns the footprint at ``quat`` onto the stance's edge frame."""
        phi = math.atan2(float(manip.tangent[1]), float(manip.tangent[0]))
        theta = self.footprint_axis_angle(hull, quat)
        return (phi - theta + math.pi / 4) % (math.pi / 2) - math.pi / 4

    def edge_aligned_quat(self, hull: th.Tensor, quat, manip: Stance) -> th.Tensor:
        """``quat`` turned about world z by :meth:`edge_yaw_delta`."""
        return G.qmul(G.quat_from_yaw(self.edge_yaw_delta(hull, quat, manip)), G.t(quat))

    def target_goal_quat(self, t_pose: G.Pose, manip: Stance) -> th.Tensor:
        return self.edge_aligned_quat(self.hull["target"], t_pose[1], manip) if self.edge_aligned() else G.t(t_pose[1])

    def place_pose_for(self, t_pose: G.Pose, p_pose: G.Pose, manip: Stance) -> G.Pose:
        """Target pose at its *place* position for a drafted partner pose: the nearer-side rule of ``goal_pose`` with the
        place gap (nextto: g_NT - 1 cm; touching: the same, the push closes it)."""
        t_quat = self.target_goal_quat(t_pose, manip)
        t_hull = G.local_points_world(self.hull["target"], (th.zeros(3), t_quat))
        p_hull_w = self.world.hull_at(self.hull["partner"], p_pose)
        p_lat = (p_hull_w[:, :2] - manip.base_xy) @ manip.tangent
        t_lat = t_hull[:, :2] @ manip.tangent
        depth_t, lat_init = manip.local(t_pose[0])
        _, lat = nearer_side_lateral(lat_init, (float(p_lat.min()), float(p_lat.max())), (float(t_lat.min()), float(t_lat.max())), self.place_gap())
        xy = manip.world(depth_t, lat)
        return th.tensor([float(xy[0]), float(xy[1]), float(t_pose[0][2])]), t_quat

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
        nudged in the goal check). ``self.goal_side`` = ``"left"`` / ``"right"`` (robot frame) records the chosen side.
        Touching with :meth:`edge_aligned`: the goal orientation is the edge-aligned yaw nearest the init yaw."""
        st = self.sidecar.stance("manipulation")
        t_quat = self.target_goal_quat(t_pose, st)
        t_hull = G.local_points_world(self.hull["target"], (th.zeros(3), t_quat))
        p_hull_w = self.world.hull_at(self.hull["partner"], p_pose)
        self.goal_side = None
        if self.relation in ("ontop", "under"):
            xy = (p_hull_w[:, :2].max(dim=0).values + p_hull_w[:, :2].min(dim=0).values) / 2
            top = float(p_hull_w[:, 2].max())
            # owner item 3 (2026-09-16 10:15): release 1 cm above the *actual* surface under the footprint centre (saucer
            # well, plate centre) — a downward raycast from above the partner, the target ignored — instead of goal_drop
            # above the partner's hull top (the teacup fell rim-to-well in the demo); usable once the partner is placed
            surface = self.release_surface(xy, p_pose, top)
            self.release_surface_z = None if surface is None else surface["z"]
            base_z = top if surface is None else float(surface["z"])
            # release height: half the well depth (rim top - well floor), clamped to [3 mm, release_above_surface_m]: a
            # shallow saucer well (1.5 cm) gets a 7.5 mm release, a flat plate 3 mm - a tall cup dropped 1 cm onto the
            # curved well landed on the slope and rested 10 deg tilted (demo 2026-09-16 14:44)
            depth = max(0.0, top - base_z)
            above = min(float(self.gen("release_above_surface_m", RELEASE_ABOVE_SURFACE)), max(RELEASE_ABOVE_SURFACE_MIN, 0.5 * depth))
            above += contact_height_extra(self, "release")[0]  # A/B knob: the release height lifted by the constant
            self.release_above_m = above
            z = base_z + above - float(t_hull[:, 2].min())
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
        if self.relation == "nextto":
            lat, t_quat = self.avoid_release_sweep(st, depth_t, lat, t_quat, lat_init, p_lat, p_hull_w, gap)
        xy = st.world(depth_t, lat)
        return th.tensor([float(xy[0]), float(xy[1]), float(t_pose[0][2])]), t_quat

    # -- open-jaw footprint at the release (oracle nextto re-run 0916-181440: the opening finger swept the pencil 9 cm) --

    def open_jaw_clearance(self, place_eef: G.Pose, obstacle_pts: th.Tensor, open_width: float = C.GRIPPER_OPEN_M, retreat: float = C.PRE_DIST) -> float:
        """Signed clearance (m) of ``obstacle_pts`` (world) from the **swept open-jaw footprint** of a place eef pose: the box
        the opened fingers sweep from the release up through the retreat, in the eef frame — along a from
        ``PAD_Z_RANGE[0] - retreat`` to ``FINGERTIP_LENGTH``, along c ±(``open_width`` / 2 + finger thickness + 1 cm), along
        x ±(finger half width + 1 cm). Positive = every point outside by at least that much; negative = a point inside."""
        pos, quat = G.t(place_eef[0]), G.t(place_eef[1])
        R = C.quat_to_mat(quat)
        local = (G.t(obstacle_pts) - pos) @ R  # columns of R = eef axes -> coordinates (x, c, a)
        hx = FINGER_HALF_WIDTH + 0.01
        hc = 0.5 * float(open_width) + FINGER_THICKNESS + 0.01
        a_lo, a_hi = float(C.PAD_Z_RANGE[0]) - float(retreat), float(C.FINGERTIP_LENGTH)
        dx = local[:, 0].abs() - hx
        dc = local[:, 1].abs() - hc
        da = th.maximum(a_lo - local[:, 2], local[:, 2] - a_hi)
        return float(th.stack([dx, dc, da], dim=1).max(dim=1).values.min())

    def open_jaw_opening_max(self, place_eef: G.Pose, obstacle_pts: th.Tensor, retreat: float = C.PRE_DIST) -> float:
        """Largest symmetric release opening (m) whose swept jaw keeps 1 cm from the obstacle points that lie within the
        jaw's x / a extents: ``2 * (min |c| - finger thickness - 1 cm)``; the full ``GRIPPER_OPEN_M`` when none does."""
        pos, quat = G.t(place_eef[0]), G.t(place_eef[1])
        R = C.quat_to_mat(quat)
        local = (G.t(obstacle_pts) - pos) @ R
        hx = FINGER_HALF_WIDTH + 0.01
        a_lo, a_hi = float(C.PAD_Z_RANGE[0]) - float(retreat), float(C.FINGERTIP_LENGTH)
        in_xa = (local[:, 0].abs() <= hx) & (local[:, 2] >= a_lo) & (local[:, 2] <= a_hi)
        if not bool(in_xa.any()):
            return float(C.GRIPPER_OPEN_M)
        c_min = float(local[in_xa, 1].abs().min())
        return max(0.0, min(float(C.GRIPPER_OPEN_M), 2.0 * (c_min - FINGER_THICKNESS - 0.01)))

    def avoid_release_sweep(self, st: Stance, depth_t: float, lat: float, t_quat: th.Tensor, lat_init: float, p_lat: th.Tensor, p_hull_w: th.Tensor,
                            gap: float) -> Tuple[float, th.Tensor]:
        """nextto (owner 2026-09-16 20:29): the partner must lie outside the open-jaw footprint at the place; otherwise the
        place moves along the nextto bound away from the partner (gap up to g_NT - 3 mm), then — when the target's orientation
        is free — the place yaw turns the jaw away (±90°, 180°). ``self.release_sweep`` records the clearance, the remedy and
        the largest release opening that clears the partner by 1 cm (``release_opening_max_m``, for the oracle)."""
        tname = self.bindings.one("target").name
        grasp = self.sidecar.grasp_of(tname)
        self.release_sweep = None
        if grasp is None or grasp.eef_in_object is None:
            return lat, t_quat
        g_nt = float(self.pair.get("g_NT", 0.03))
        z = float(self.top.z) if self.top is not None else 0.0
        t_hull0 = self.hull["target"]

        def place_of(lat_v: float, quat: th.Tensor) -> G.Pose:
            xy = st.world(depth_t, lat_v)
            t_hull = G.local_points_world(t_hull0, (th.zeros(3), quat))
            return IK.grasp_pose((th.tensor([float(xy[0]), float(xy[1]), z - float(t_hull[:, 2].min()) + 0.005]), quat), grasp.eef_in_object)

        def lat_of(quat: th.Tensor, gap_v: float) -> float:
            t_hull = G.local_points_world(t_hull0, (th.zeros(3), quat))
            t_l = t_hull[:, :2] @ st.tangent
            return nearer_side_lateral(lat_init, (float(p_lat.min()), float(p_lat.max())), (float(t_l.min()), float(t_l.max())), gap_v)[1]

        c0 = self.open_jaw_clearance(place_of(lat, t_quat), p_hull_w)
        rec: Dict[str, Any] = {"clearance_m": round(c0, 4), "remedy": None, "open_width_m": C.GRIPPER_OPEN_M}
        best_lat, best_quat, best_c = lat, t_quat, c0
        if c0 < 0:
            # (a) along the nextto bound: the AABB gap may grow to just under g_NT (the predicate: norm(gap) < g_NT + adjacency)
            for gap_v in (g_nt - 0.006, g_nt - 0.003):
                if gap_v <= gap:
                    continue
                lat_v = lat_of(t_quat, gap_v)
                c_v = self.open_jaw_clearance(place_of(lat_v, t_quat), p_hull_w)
                if c_v > best_c:
                    best_lat, best_c = lat_v, c_v
                    rec.update({"remedy": f"gap {gap_v:.3f}", "gap_m": round(gap_v, 4)})
                if best_c >= 0:
                    break
        if best_c < 0 and self.target_orientation_free():
            # (b) the place yaw turns the jaw away from the partner (the target's orientation is not constrained)
            for yaw_deg in (90.0, -90.0, 180.0):
                q_yaw = G.compose((th.zeros(3), G.quat_from_yaw(math.radians(yaw_deg))), (th.zeros(3), t_quat))[1]  # world yaw applied to the rest pose
                for gap_v in (gap, g_nt - 0.003):
                    lat_v = lat_of(q_yaw, gap_v)
                    c_v = self.open_jaw_clearance(place_of(lat_v, q_yaw), p_hull_w)
                    if c_v > best_c:
                        best_lat, best_quat, best_c = lat_v, q_yaw, c_v
                        rec.update({"remedy": f"yaw {yaw_deg:+.0f} deg, gap {gap_v:.3f}", "yaw_deg": yaw_deg, "gap_m": round(gap_v, 4)})
                if best_c >= 0:
                    break
        # the largest symmetric release opening that clears the partner by 1 cm at the chosen place (the oracle may open
        # only this far: closure probe 'pre-shaped opening'); the pinch width + 1 cm is the least that releases the object
        opening_max = self.open_jaw_opening_max(place_of(best_lat, best_quat), p_hull_w) if best_c < 0 else float(C.GRIPPER_OPEN_M)
        ext = TT.hull_extents(t_hull0, self.rest["target"])
        pinch = float(min(ext[0], ext[1]))
        rec.update({"clearance_after_m": round(best_c, 4), "release_opening_max_m": round(max(0.0, opening_max), 4), "pinch_width_m": round(pinch, 4),
                    "partner_inside_jaw": best_c < 0})
        self.release_sweep = rec
        if best_c < 0 and opening_max < pinch + 0.01:
            raise Reject("release_sweep", f"the partner lies {-best_c * 1000:.0f} mm inside the open-jaw footprint at every admissible place (gap up to g_NT - 3 mm"
                                          f"{', yaw turned' if self.target_orientation_free() else ''}); even an opening of pinch + 1 cm ({pinch + 0.01:.3f} m) would sweep it")
        return best_lat, best_quat

    def target_orientation_free(self) -> bool:
        tols = self.gen("tolerances") or {}
        return "orientation_metric" in tols and tols["orientation_metric"] is None

    def release_surface(self, xy: th.Tensor, p_pose: G.Pose, top_z: float) -> Optional[Dict[str, Any]]:
        """``SimWorld.surface_z_under`` at the goal xy (ray from 30 cm above the partner's top, the target ignored) when the
        partner actually stands at ``p_pose`` (it is parked while the first goal is drafted; ``after_settle`` recomputes);
        a hit above the support and within the partner's height range counts, else None (hull-top rule)."""
        p_now = self.obj("partner").get_position_orientation()[0].float()
        if float(th.linalg.norm(p_now[:2] - G.t(p_pose[0])[:2])) > 0.05:
            return None
        # a small cluster of rays (centre + 4 at +-1.5 cm): a saucer's convex-decomposed well can let the centre ray through
        # to the table (teacup run 17:24: surface 0.8127 = the table, 6 mm under the well floor - the cup was set 4 mm into
        # the saucer); only hits on the partner's own links count, the highest of them is the floor under the footprint centre
        partner = self.obj("partner")
        paths = tuple({l.prim_path for l in partner.links.values()} | {partner.prim_path})
        c = G.t(xy)[:2]
        hits = []
        for dx, dy in ((0.0, 0.0), (0.015, 0.0), (-0.015, 0.0), (0.0, 0.015), (0.0, -0.015)):
            h = self.world.surface_z_under(th.tensor([float(c[0]) + dx, float(c[1]) + dy]), top_z + 0.30, ignore=[self.obj("target")])
            if h is not None and str(h.get("body", "")).startswith(paths) and h["z"] <= top_z + 0.005:
                hits.append(h)
        if not hits:
            return None
        best = max(hits, key=lambda h: h["z"])
        best["n_partner_hits"] = len(hits)
        return best

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
        drop_floor = float(self.gen("drop_height_m", 0.005)) + contact_height_extra(self, "release")[0]  # A/B knob lift
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
            settled = self.settle_motion(draft, tname)
            draft.summary["target_settle"] = settled
            res = self.ikc.exact([grasp.pre_grasp, grasp.grasp], manip, grasp.arm, ignore_for_last=self.support_objects(),
                                 scale_key=self.choice.models["target"].scale, cache=False, approached=[self.obj("target")])
            ok = res.ok
            if not ok and settled.get("rotation_deg", 0.0) > float(self.gen("regrasp_after_roll_deg", REGRASP_AFTER_ROLL_DEG)):
                # a round target rolled while settling: its library grasp now faces the wrong way; re-select among the
                # library grasps at the settled pose from the same stance (what the oracle would have to grasp anyway)
                chosen = self.reselect_grasp(draft, grasp, manip, tname)
                if chosen is not None:
                    log.info("%s: target rolled %.0f deg while settling; grasp re-selected at the settled pose: %s", self.gp.alias, settled["rotation_deg"], chosen)
                    draft.summary["grasp"] = {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw, "reselected_after_roll": True}
                    ok = True
            if not ok:
                # diagnostics (inside regression 2026-09-16 14:29: the pre-grasp failed 6/8 at the settled pose): the pre
                # alone, with the partner removed, with every spawned object removed, and how far the target settled
                def _ok(r):
                    return "ok" if r.ok else "fail"

                everything = list(self.world.spawned.values())
                r_pre = self.ikc.exact([grasp.pre_grasp], manip, grasp.arm, cache=False)
                r_pre_np = self.ikc.exact([grasp.pre_grasp], manip, grasp.arm, ignore_for_last=[self.obj("partner")], cache=False)
                r_pre_all = self.ikc.exact([grasp.pre_grasp], manip, grasp.arm, ignore_for_last=everything + self.support_objects(), cache=False)
                moved = float(th.linalg.norm(G.t(draft.object_poses[tname][0]) - G.t(draft.pose_targets[tname]["position"])))
                raise Reject(res.code or "grasp_unreachable", f"pre-grasp / grasp at the settled target pose fail the exact IK ({res.reason}; pre alone {_ok(r_pre)}, "
                                                                f"pre with the partner removed {_ok(r_pre_np)}, pre with every object removed {_ok(r_pre_all)}; target settled "
                                                                f"{settled.get('translation_mm', float('nan')):.1f} mm / {settled.get('rotation_deg', float('nan')):.0f} deg from its draft "
                                                                f"(goal {moved * 1000:.0f} mm away); trunk ({manip.translate}, {manip.tilt:.2f}), grasp {grasp.grasp_name} yaw {grasp.yaw:.2f})")
            # closure probe 2026-09-16 (runlogs/closure_probe_20260916): the lower *open* pad standing on the table makes the
            # closure asymmetric (one finger crawls, the other knocks the object along the table: marker roll 24-77 deg, the
            # sprout knocked out of the pinch). The support is ignored for the grasp pose by §3.6, so this is checked here:
            # the open gripper at the grasp configuration must stay ``pad_support_clearance_m`` clear of the support; a grasp
            # that fails tries the other library grasps at the settled pose, then ``grasp_pad_on_support``
            q_grasp = res.q if res.ok else getattr(self, "_reselected_q", None)
            clear = None if q_grasp is None else self.open_pads_clear_of_support(q_grasp)
            if clear is False:
                # a preference, not a gate: a grasp whose open pads clear the support replaces this one when the library has
                # one at the settled pose; otherwise the grasp stays (the oracle's closure ramp covers the residual) and the
                # fact is recorded (low cylinders: the pads necessarily reach the table)
                chosen = self.reselect_grasp(draft, grasp, manip, tname, require_pad_clearance=True)
                if chosen is not None:
                    log.info("%s: open pad on the support for %s; grasp re-selected at the settled pose: %s", self.gp.alias, draft.summary.get("grasp", {}).get("name"), chosen)
                    draft.summary["grasp"] = {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw, "reselected_pad_clearance": True}
                    clear = True
                else:
                    log.info("%s: open pad on the support for %s and no library grasp clears it at the settled pose; kept", self.gp.alias, grasp.grasp_name)
            grasp.extras["open_pads_clear_of_support"] = clear
            draft.summary["open_pads_clear_of_support"] = clear

    def open_pads_clear_of_support(self, q: th.Tensor) -> bool:
        """Whether the robot at the grasp configuration ``q`` (fingers open as the sampler solves it) keeps its gripper
        ``pad_support_clearance_m`` (5 mm) clear of the support: whole-body proximity check with every spawned object removed,
        so only the support (and the scene) remain — the complement of the grasp pose's §3.6 support exemption."""
        margin = float(self.gen("pad_support_clearance_m", PAD_SUPPORT_CLEARANCE))
        others = list(self.world.spawned.values())
        with self.ikc.ignoring(others):
            return bool(self.kit.collision_free(q, margin=margin)[0])

    def settle_motion(self, draft: Draft, tname: str) -> Dict[str, float]:
        """Translation (mm) and rotation (deg) of the target between its drafted pose (``summary.target_draft_pose``) and
        its settled pose — a round target may roll while settling, which turns its library grasp away."""
        d = draft.summary.get("target_draft_pose")
        if not d:
            return {}
        p0, q0 = th.tensor(d[0]), th.tensor(d[1])
        p1, q1 = G.t(draft.object_poses[tname][0]), G.t(draft.object_poses[tname][1])
        return {"translation_mm": round(float(th.linalg.norm(p1 - p0)) * 1000, 1), "rotation_deg": round(math.degrees(G.quat_angle(q0, q1)), 1)}

    def reselect_grasp(self, draft: Draft, grasp: GraspRecord, manip: Stance, tname: str, require_pad_clearance: bool = False,
                       exclude: Sequence[str] = ()) -> Optional[str]:
        """Try the library grasps (flush tier first, then the rest; all attachable) at the *settled* target pose from the
        manipulation stance — pre-grasp + grasp exactly, support removed for the grasp, the target removed along the
        approach, optionally the open pads clear of the support — and rewrite the grasp record with the first that passes.
        Returns its name or None."""
        t_pose = draft.object_poses[tname]
        order = list(self.grasps_flush) + [g for g in self.grasps if g not in self.grasps_flush]
        for g in order:
            eef_in_object = getattr(g, "pose", None)
            if eef_in_object is None or (getattr(g, "name", None) == grasp.grasp_name and require_pad_clearance) or getattr(g, "name", None) in exclude:
                continue
            eef = IK.grasp_pose(t_pose, eef_in_object)
            pre = IK.pre_pose(eef)
            res = self.ikc.exact([pre, eef], manip, grasp.arm, ignore_for_last=self.support_objects(), scale_key=self.choice.models["target"].scale,
                                 cache=False, approached=[self.obj("target")])
            if not res.ok:
                continue
            if require_pad_clearance and res.q is not None and not self.open_pads_clear_of_support(res.q):
                continue
            self._reselected_q = res.q
            grasp.grasp_key, grasp.grasp_name = getattr(g, "key", grasp.grasp_key), getattr(g, "name", grasp.grasp_name)
            grasp.finger_q = list(getattr(g, "finger_q", []) or []) or None
            grasp.eef_in_object = eef_in_object
            grasp.grasp, grasp.pre_grasp = eef, pre
            self.apply_height_extra_to_grasp(grasp, t_pose, manip)  # the A/B lift applies to the re-selected library grasp too
            return str(grasp.grasp_name)
        return None

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
            place_pos = goal_pos
            if self.relation == "touching":
                # the oracle program places at the *nextto* position and pushes the target into contact from there: the
                # place is the settled contact position moved back (away from the partner, robot's right) by the push distance
                tangent = manip.tangent
                # the place sits the nominal gap (g_NT - 1 cm) short of the settled contact; the push (recorded in
                # ``touching_contacts``) spans the measured gap at that place plus the margin
                place_pos = goal_pos - th.tensor([float(tangent[0]), float(tangent[1]), 0.0]) * self.place_gap()
                self._touching_place_pos = place_pos
                draft.summary["touching"] = {"place_gap_m": round(self.place_gap(), 4)}
            place = IK.grasp_pose((place_pos, goal_quat), grasp.eef_in_object)
            pre_place = IK.pre_pose(place)
            ignore = self.support_objects() + [self.obj("partner")]
            if self.relation in ("nextto", "touching"):
                # open-jaw footprint facts at the recorded place (touching: the nextto position the push starts from)
                p_now = self.obj("partner").get_position_orientation()
                p_hull_now = self.world.hull_at(self.hull["partner"], (p_now[0].float(), p_now[1].float()))
                c_place = self.open_jaw_clearance(place, p_hull_now)
                rec = dict(self.release_sweep or {})
                rec.update({"clearance_at_place_m": round(c_place, 4), "partner_inside_jaw_at_place": c_place < 0,
                            "release_opening_max_m": round(self.open_jaw_opening_max(place, p_hull_now), 4)})
                grasp.extras["release_sweep"] = rec
                draft.summary["release_sweep"] = rec
            if self.relation in ("ontop", "under"):
                grasp.extras["release_surface_z"] = None if self.release_surface_z is None else round(float(self.release_surface_z), 4)
                grasp.extras["release_above_m"] = None if self.release_above_m is None else round(float(self.release_above_m), 4)
                grasp.extras["pivot_arm_m"] = round(self.pivot_arm(grasp.eef_in_object), 4) if grasp.eef_in_object is not None else None
                grasp.extras["closing_tilt_deg"] = round(self.closing_tilt_deg(grasp.eef_in_object), 1) if grasp.eef_in_object is not None else None
                draft.summary["release_surface_z"] = grasp.extras["release_surface_z"]
                draft.summary["release_above_m"] = grasp.extras["release_above_m"]
                draft.summary["pivot_arm_m"] = grasp.extras["pivot_arm_m"]
        if self.goal_side is not None:
            draft.summary["goal_side"] = self.goal_side
        # grasp facts for the finger-closure investigation (facts only, 2026-09-16 18:16): the pinch line's angle to the
        # object's long axis and the pad-centre height above the support at the grasp pose
        grasp.extras.update(self.grasp_facts(grasp))
        draft.summary["grasp_facts"] = {k: grasp.extras.get(k) for k in ("pinch_vs_long_axis_deg", "pad_height_above_support_m", "pad_height_above_axis_m",
                                                                          "closing_tilt_deg", "pivot_arm_m")}
        approached_place = (self.support_objects() + [self.obj("partner")]) if self.relation != "inside" else self.support_objects()
        # touching (owner 2026-09-18 16:32): the box is carried to a second base pose along the edge (recorded path, in hand)
        # and placed / pushed from there - the place descent and the contacts are checked from that stance
        work = self.push_stance(draft, manip, grasp) if self.relation == "touching" else None
        st_place = work if work is not None else manip

        def descent(pre_p: G.Pose, pl: G.Pose):
            with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
                # along the whole descent: the placed-on partner (owner A) and — as the program check's ``place`` rule and the
                # oracle's descent — the supports (the carried object's sphere fit reaches the table 4-6 cm above the place)
                return self.ikc.exact([pre_p, pl], st_place, grasp.arm, ignore_for_last=ignore, scale_key=self.choice.models["target"].scale, cache=False,
                                      approached=approached_place)

        res = descent(pre_place, place)
        if not res.ok and res.code and "self-collision (attached_object" in res.reason:
            # A7 relay (oracle, 2026-09-16 21:07): a collision appearing along the descent between the carried object and the
            # forearm / torso / head is a genuine infeasibility of *this* grasp yaw / approach class, not a hand-fixed pair
            # (those are not checked) — try the other library grasps at the settled init pose before rejecting
            tried = [grasp.grasp_name]
            for _ in range(len(self.grasps)):
                chosen = self.reselect_grasp(draft, grasp, manip, tname, exclude=tried)
                if chosen is None:
                    break
                tried.append(chosen)
                place = IK.grasp_pose((place_pos if self.relation != "inside" else G.t(draft.goal_poses[tname][0]), goal_quat), grasp.eef_in_object)
                pre_place = IK.pre_pose(place)
                res = descent(pre_place, place)
                if res.ok:
                    log.info("%s: the carried object hit the robot on the descent with the drawn grasp; grasp %s carries clear", self.gp.alias, chosen)
                    draft.summary["grasp"] = {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw, "reselected_descent_self_collision": True,
                                              "tried": tried}
                    grasp.extras.update(self.grasp_facts(grasp))
                    break
        if not res.ok and res.code:
            raise Reject(res.code, f"pre-place -> place descent from the {st_place.name} stance with the object attached: {res.reason}")
        if not res.ok:
            # diagnostics only on failure (which pose, and whether the attached object / the world / the reach is the cause)
            def _ok(r):
                return "ok" if r.ok else "fail"

            r_pre = self.ikc.exact([pre_place], st_place, grasp.arm, cache=False)
            r_place = self.ikc.exact([place], st_place, grasp.arm, ignore_for_last=ignore, cache=False)
            everything = [o for o in self.world.scene.objects if self.world.obstacle_filter(o)]
            with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
                r_pre_obj = self.ikc.exact([pre_place], st_place, grasp.arm, cache=False)
                r_pre_obj_ign = self.ikc.exact([pre_place], st_place, grasp.arm, ignore_for_last=ignore, cache=False)
                r_place_obj = self.ikc.exact([place], st_place, grasp.arm, ignore_for_last=ignore, cache=False)
                r_place_obj_free = self.ikc.exact([place], st_place, grasp.arm, ignore_for_last=everything, cache=False)
            _, lat = st_place.local(goal_pos[:2])
            raise Reject("place_unreachable", f"pre-place / place fail the exact IK from the {st_place.name} stance with the object attached "
                                              f"(without the object: pre-place {_ok(r_pre)}, place {_ok(r_place)}; with the object: pre-place {_ok(r_pre_obj)}, "
                                              f"pre-place ignoring support+partner {_ok(r_pre_obj_ign)}, place {_ok(r_place_obj)}, place ignoring every object "
                                              f"{_ok(r_place_obj_free)}; goal lateral {lat:+.2f} m, trunk ({manip.translate}, {manip.tilt:.2f}), stand-off {manip.stand_off})")
        grasp.pre_place, grasp.place = pre_place, place
        if self.relation == "touching":
            # the oracle's place ends 10 cm back along the approach (release + lift-off) and its retreat goes 5 cm further: the
            # pose 15 cm back from the place, fingers open, the released box ignored, must have a configuration (the program
            # check's 'retreat: exact IK failed / joint 6-7 at the limit (stance push)' on 6/16 attempts of 0918-175907)
            clear = IK.pre_pose(place, RETREAT_DIST_M + RETREAT_LITTLE_M)
            r_clear = self.ikc.exact([clear], st_place, grasp.arm, ignore_for_last=self.support_objects() + [self.obj("target")], cache=False)
            if not r_clear.ok:
                raise Reject(r_clear.code or "place_unreachable", f"place retreat clear pose ({(RETREAT_DIST_M + RETREAT_LITTLE_M) * 100:.0f} cm back along the "
                                                                 f"approach) from the {st_place.name} stance: {r_clear.reason}")
        # the lift (oracle ``Lift``): grasp pose raised by carry_height with the object attached, support ignored (the
        # object leaves it); the pre-place is where the carry ends, so the lift end must be IK-feasible too
        lift_end = (G.t(grasp.grasp[0]) + th.tensor([0.0, 0.0, float(grasp.carry_height)]), G.t(grasp.grasp[1]))
        with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
            # tracked from the grasp configuration (one IK branch along the lift), support ignored throughout
            r_lift = self.ikc.waypoints_ok([grasp.grasp, lift_end], manip, grasp.arm, ignore_all=self.support_objects(), cache=False, approached=[])
        if not r_lift.ok:
            raise Reject(r_lift.code or "lift_unreachable", f"lift ({grasp.carry_height:.2f} m above the grasp) fails with the object attached: {r_lift.reason}")
        cam = self.camera_pose_at(manip)
        cam_place = cam if work is None else self.camera_pose_at(work)
        # the pre-poses join the check (owner 2026-09-16 10:15 item 1b): the oracle's FOV monitor arms from the pre-pose
        self.require_in_frame(cam, {"pre-grasp": grasp.pre_grasp, "grasp": grasp.grasp})
        self.require_in_frame(cam_place, {"pre-place": pre_place, "place": place})
        if self.relation == "touching":
            # the push (+ brace) from the stance the box was placed from (the second base pose when recorded)
            cam = cam_place
            self.touching_contacts(draft, st_place)
            c = self.sidecar.contacts
            poses = {"push pre-contact": c["push"].pre, "push contact": c["push"].contact, "push goal": c["push"].goal}
            if "brace" in c:
                # a brace whose poses leave the frame is dropped (push-only) rather than rejecting the instance
                brace_poses = {"brace pre-contact": c["brace"].pre, "brace contact": c["brace"].contact}
                if all(self.ikc.eef_in_frame(p, cam) for p in brace_poses.values() if p is not None):
                    poses.update(brace_poses)
                else:
                    self.sidecar.contacts.pop("brace", None)
                    manip.arms, manip.contacts = ["right"], ["push"]
                    draft.summary["brace"] = {"present": False, "why": "brace poses outside the manipulation image (5 %)", "gate": (self._brace_gate or {})}
            self.require_in_frame(cam, poses)

    @contextmanager
    def posed(self, role: str, pose: G.Pose):
        """The role's object teleported to ``pose`` (collision world synced) for the block, then put back exactly where it was."""
        obj = self.obj(role)
        prev = obj.get_position_orientation()
        obj.set_position_orientation(G.t(pose[0]), G.t(pose[1]))
        obj.keep_still()
        self.kit.sync_world()
        try:
            yield
        finally:
            obj.set_position_orientation(prev[0], prev[1])
            obj.keep_still()
            self.kit.sync_world()

    def brace_gate_at(self, st: Stance, t_place_pose: G.Pose, p_pose: G.Pose) -> Dict[str, Any]:
        """A1 joint band gate at stance ``st``: the push work point (target's right face at the place) in the right band and the
        brace point (partner's far face) in the left contact band of the same trunk (``brace_placement_admissible``)."""
        right_cells = self.bands.band(st.translate, st.tilt) if getattr(self, "bands", None) is not None else None
        left_cells = left_contact_band_cells(self, (st.translate, st.tilt), TT.height_of(self.hull["partner"], self.rest["partner"]))
        if left_cells is None or right_cells is None:
            return {"ok": None, "why": "left contact band unavailable"}
        work, brace = self.touching_points(t_place_pose, p_pose, st)
        w_b, b_b = st.world_to_base(work[:2]), st.world_to_base(brace[:2])
        ok = brace_placement_admissible(w_b, b_b, right_cells, left_cells)
        return {"ok": bool(ok), "stance": st.name, "work_base": [round(float(v), 3) for v in w_b], "brace_base": [round(float(v), 3) for v in b_b],
                "right_cells": int(len(right_cells)), "left_cells": int(len(left_cells)),
                **({} if ok else {"why": f"work {'in' if in_band(w_b, right_cells) else 'OFF'} right band / brace {'in' if in_band(b_b, left_cells) else 'OFF'} left band at {st.name}"})}

    def hand_overlap(self, eef_pose: G.Pose, role: str, obj_pose: G.Pose, margin: float = 0.005) -> Optional[str]:
        """Which part of the hand beyond the finger profile (``HAND_SPHERES_EEF``: wrist camera, knuckles, gripper body) comes
        within ``margin`` of the role's hull box at ``obj_pose`` for the eef pose — None when all clear. The IK check ignores the
        pushed object at its contact pose, so the camera hanging over the box top was invisible to it (owner 2026-09-18 19:39)."""
        hull = G.t(self.hull[role])
        lo, hi = hull.min(dim=0).values, hull.max(dim=0).values
        inv = C.invert((G.t(obj_pose[0]), G.t(obj_pose[1])))
        eef = (G.t(eef_pose[0]), G.t(eef_pose[1]))
        worst = None
        for part, (centres, r) in HAND_SPHERES_EEF.items():
            pts_o = C.transform_points(inv, C.transform_points(eef, th.tensor(centres, dtype=th.float32)))
            d = th.clamp(lo - pts_o, min=0.0) + th.clamp(pts_o - hi, min=0.0)
            dist = float(th.linalg.norm(d, dim=1).min())
            if dist < r + margin and (worst is None or dist - r < worst[1]):
                worst = (part, dist - r)
        if worst is None:
            return None
        return f"{worst[0]} {worst[1] * 1000:+.0f} mm from the box (margin {margin * 1000:.0f} mm)"

    def push_hand_overlap(self, push_init: G.Pose, push_goal: G.Pose, t_place_pose: G.Pose, direction: th.Tensor, dist: float) -> Optional[str]:
        """The hand at the push contact (box at the place) and at the push goal (box pushed ``dist``): the first overlap, or None."""
        bad = self.hand_overlap(push_init, "target", t_place_pose)
        if bad is None:
            pushed = (G.t(t_place_pose[0]) + G.t(direction) * float(dist), G.t(t_place_pose[1]))
            bad = self.hand_overlap(push_goal, "target", pushed)
        return bad

    def push_chain_feasible(self, st: Stance, t_place_pose: G.Pose, p_pose: G.Pose) -> Optional[str]:
        """The first push class whose [pre, init, goal] chain and retreat clear pose pass the exact IK from ``st`` with the box
        posed at the place (the checks ``touching_contacts`` repeats when recording); None when no class passes."""
        tangent = st.tangent
        direction = th.tensor([float(tangent[0]), float(tangent[1]), 0.0])
        p_init, _ = self.touching_points(t_place_pose, p_pose, st)
        t_hull_w = self.world.hull_at(self.hull["target"], t_place_pose)
        t_face_z = (float(t_hull_w[:, 2].min()), float(t_hull_w[:, 2].max()))
        p_hull = self.world.hull_at(self.hull["partner"], p_pose)
        gap = float((p_hull[:, :2] @ tangent).min() - (t_hull_w[:, :2] @ tangent).max())
        dist = self.push_distance(gap)
        ignore_push = self.support_objects() + [self.obj("target")]
        with self.posed("target", t_place_pose):
            for push_kind, push_init, _extras in self.touching_push_candidates(st, p_init, t_face_z, direction):
                push_goal = (push_init[0] + direction * dist, push_init[1])
                if self.push_hand_overlap(push_init, push_goal, t_place_pose, direction, dist) is not None:
                    continue
                r = self.ikc.exact([IK.pre_pose(push_init), push_init, push_goal], st, "right", ignore_for_last=ignore_push, cache=False, fingers=IK.FINGER_CLOSED)
                if not r.ok:
                    continue
                r2 = self.ikc.exact([IK.pre_pose(push_goal)], st, "right", ignore_for_last=ignore_push, cache=False, fingers=IK.FINGER_CLOSED)
                if r2.ok:
                    return push_kind
        return None

    def place_infeasible_why(self, st: Stance, t_place_pose: G.Pose, grasp: GraspRecord) -> Optional[str]:
        """Why the place of the carried box from ``st`` fails (pre-place -> place descent with the box attached, then the
        15 cm clear pose of the release + retreat with the box ignored), or None when both pass."""
        place = IK.grasp_pose(t_place_pose, grasp.eef_in_object)
        ign = self.support_objects() + [self.obj("partner")]
        with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
            r = self.ikc.exact([IK.pre_pose(place), place], st, grasp.arm, ignore_for_last=ign, scale_key=self.choice.models["target"].scale, cache=False, approached=ign)
        if not r.ok:
            return f"descent: {r.reason}"
        r2 = self.ikc.exact([IK.pre_pose(place, RETREAT_DIST_M + RETREAT_LITTLE_M)], st, grasp.arm, ignore_for_last=self.support_objects() + [self.obj("target")], cache=False)
        if not r2.ok:
            return f"clear pose: {r2.reason}"
        return None

    def push_stance_for(self, manip: Stance, t_place_pose: G.Pose, lateral: Optional[float] = None) -> Optional[Stance]:
        """Touching (owner 2026-09-18 16:32): the second base pose ``push`` = the manipulation stance moved along the edge, same
        trunk, so that the push contact (the placed box's right face) lies at ``push_contact_lateral_m`` from the robot's centre
        line (default -0.12: every push that worked had it 7-18 cm right of centre; at -0.05 the right arm crosses the body
        and joints 2 / 3 sit on their limits, 0918-173814). The box is carried there in hand and set down from there.
        Pure geometry (nothing recorded); None when the shift is below ``push_stance_min_shift_m`` or the feature is off
        (``generation.push_stance``)."""
        if self.relation != "touching" or not bool(self.gen("push_stance", True)):
            return None
        tangent = manip.tangent
        t_hull = self.world.hull_at(self.hull["target"], t_place_pose)
        right_face_xy = t_hull[int(th.argmin(t_hull[:, :2] @ tangent)), :2]
        _, lat_work = manip.local(right_face_xy)
        want = float(self.gen("push_contact_lateral_m", -0.12)) if lateral is None else float(lateral)
        dlat = float(lat_work - want)
        if abs(dlat) < float(self.gen("push_stance_min_shift_m", 0.02)):
            return None
        target_xy = G.t(manip.base_xy) + G.t(tangent) * dlat
        cands = [manip.shifted(lateral=manip.lateral + s * dlat, name="push") for s in (1.0, -1.0)]  # the edge frame's lateral sign
        st = min(cands, key=lambda c: float(th.linalg.norm(G.t(c.base_xy) - target_xy)))
        st = st.with_trunk(manip.translate, manip.tilt, "push")
        st.focus, st.arms = G.t(t_place_pose[0]).clone(), ["right"]
        st.extras.update({"second_base_pose": True, "lateral_shift_m": round(dlat, 4), "push_contact_lateral_m": round(float(lat_work - dlat), 4),
                          "rule": "base moved along the edge (box in hand) so the push contact lies at push_contact_lateral_m from the centre line"})
        return st

    def push_stance(self, draft: Draft, manip: Stance, grasp: GraspRecord) -> Optional[Stance]:
        """Record the ``push`` stance of ``push_stance_for``: whole-body check with the objects at the place, the path
        manipulation -> push carried (the box posed in the hand per the grasp record; ``paths[].carried``), the brace gate
        and the goal observation stance re-evaluated there. ``Reject(stance_collision | no_path | path_collision)``."""
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        place_pos = getattr(self, "_touching_place_pos", None)
        if place_pos is None:
            place_pos = th.tensor(draft.pose_targets[tname]["position"])
        goal_quat = G.t(draft.goal_poses[tname][1])
        t_place_pose = (G.t(place_pos), goal_quat)
        p_now = self.obj("partner").get_position_orientation()
        p_pose = (p_now[0].clone().float(), p_now[1].clone().float())
        # the contact lateral is searched (default -0.12 first, then further right / nearer): the first stance from which a push
        # class passes the chain + retreat clear pose (0918-174815: at -0.12 alone joint 4 / the clear pose failed 9/16 attempts)
        laterals = [float(v) for v in (self.gen("push_contact_laterals_m", None) or [self.gen("push_contact_lateral_m", -0.12), -0.16, -0.20, -0.08])]
        st, tried, manip_ok = None, [], False
        for lat in laterals:
            cand = self.push_stance_for(manip, t_place_pose, lateral=lat)  # None = the manipulation stance already puts the contact there
            probe = cand if cand is not None else manip
            kind = self.push_chain_feasible(probe, t_place_pose, p_pose)
            place_why = self.place_infeasible_why(probe, t_place_pose, grasp) if kind is not None else None
            tried.append(f"{lat:+.2f}{' (manipulation)' if cand is None else ''}: {kind or 'no class'}{'; place: ' + place_why if place_why else ''}")
            if kind is not None and place_why is None:
                if cand is None:
                    manip_ok = True
                st = cand
                break
        draft.summary.setdefault("touching", {})["push_stance_search"] = tried
        if st is None:
            if manip_ok:
                return None  # the manipulation stance itself passes: no second base pose
            raise Reject("contact_unreachable", f"touching push: no push class passes from any push stance lateral ({'; '.join(tried)})")
        try:
            TT.whole_body_check_at(self, [st], {tname: t_place_pose, pname: p_pose}, require=True, reason="stance_collision")
        finally:
            # the box goes back to its init pose: the program check and the grasp checks read the live world (0918-170003: the
            # box left at the place, 3-6 cm from the init, made the checker's grasp IK fail on 4/12 attempts)
            self.obj("target").set_position_orientation(*[G.t(v) for v in draft.object_poses[tname]])
            self.obj("target").keep_still()
        self.sidecar.add_stance(st)
        self.kit.sync_world()
        # the drive is swept with the arm at the LIFT configuration (the box carry_height above the grasp, as the oracle drives
        # it), not the stance's untucked arms - the box posed in the untucked hand stands inside the table (0918-165104)
        lift_end = (G.t(grasp.grasp[0]) + th.tensor([0.0, 0.0, float(grasp.carry_height)]), G.t(grasp.grasp[1]))
        with attached(self.kit, grasp.arm, self.obj("target"), eef_in_object=grasp.eef_in_object):
            r_lift = self.ikc.exact([lift_end], manip, grasp.arm, ignore_for_last=self.support_objects(), cache=False)
        arm_q = None
        if r_lift.ok and r_lift.q is not None:
            idx = self.kit.robot.arm_control_idx
            arm_q = {"right": r_lift.q[idx["right"]].clone(), "left": r_lift.q[idx["left"]].clone()}
        path = self.record_path(manip, st, carried=tname, arm=grasp.arm, arm_q=arm_q)  # Reject(no_path | path_collision) fails the attempt
        st.carried = tname
        st.extras["path_length_m"] = getattr(path, "length_m", None)
        self._brace_gate = self.brace_gate_at(st, t_place_pose, p_pose)
        self.sidecar.set_goal(st.base_pose2d, st.translate, st.tilt, arms="same")  # the goal observation is taken from the push stance
        draft.summary["push_stance"] = {"lateral_shift_m": st.extras["lateral_shift_m"], "base_pose2d": [float(v) for v in st.base_pose2d],
                                        "path_length_m": st.extras["path_length_m"], "brace_gate": self._brace_gate}
        log.info("%s: push stance recorded %+.3f m along the edge (path %.2f m, box in hand)", self.gp.alias, st.extras["lateral_shift_m"],
                 float(st.extras["path_length_m"] or 0.0))
        return st

    def require_in_frame(self, cam: G.Pose, poses: Dict[str, Optional[G.Pose]]):
        """Every given eef pose must project inside the manipulation image with the 5 % margin (``IKChecker.eef_in_frame``
        with the shared camera FK of ``camera_pose_at``); ``Reject("gripper_not_in_frame")`` names the first that does not."""
        for what, pose in poses.items():
            if pose is not None and not self.ikc.eef_in_frame(pose, cam):
                raise Reject("gripper_not_in_frame", f"{what} eef pose outside the manipulation image (5 %)")

    def touching_contacts(self, draft: Draft, manip: Stance):
        """§6.11 touching: push contact on the target's right face (right arm) toward the partner, brace pre-contact +
        contact of the closed left gripper on the partner's far (left) side; both IK-checked (§4.6 step 6)."""
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        # the push starts at the *place* (nextto) position the program sets the target down at and ends at the settled
        # contact position (+5 mm of pressing): ``push_distance`` (the old code pushed from the contact position onward)
        goal_quat = draft.goal_poses[tname][1]
        place_pos = getattr(self, "_touching_place_pos", None)
        if place_pos is None:
            place_pos = th.tensor(draft.pose_targets[tname]["position"])
        t_pose = (G.t(place_pos), G.t(goal_quat))
        p_now = self.obj("partner").get_position_orientation()
        p_pose = (p_now[0].clone().float(), p_now[1].clone().float())
        tangent = manip.tangent
        direction = th.tensor([float(tangent[0]), float(tangent[1]), 0.0])  # toward the partner = robot's left
        p_init, b_point = self.touching_points(t_pose, p_pose, manip)
        p_hull = self.world.hull_at(self.hull["partner"], p_pose)
        face_z = (float(p_hull[:, 2].min()), float(p_hull[:, 2].max()))
        # the push spans the *measured* gap at the place (target's left face at the place pose -> partner's right face along
        # the push direction) plus the margin; the recorded goal lies past the contact by the margin
        t_hull_place = self.world.hull_at(self.hull["target"], t_pose)
        gap_at_place = float((p_hull[:, :2] @ tangent).min() - (t_hull_place[:, :2] @ tangent).max())
        dist = self.push_distance(gap_at_place)
        draft.summary.setdefault("touching", {}).update({"gap_at_place_m": round(gap_at_place, 4), "push_distance_m": round(dist, 4), "push_margin_m": round(self.push_margin(), 4)})
        # closed pads for both contacts (§3.6); the push arm first, then the brace with the push arm locked (§4.6 step 6).
        # Push candidates: the pads closing horizontally (default) then vertically (the other wrist roll — the horizontal
        # one put the right pre-contact out of reach on 5/8 attempts of the sanitary-napkins loop); brace candidates in
        # order (pads on the far face / top-down blade beside it, ``touching_brace_candidates``)
        why: List[str] = []
        t_hull_w = self.world.hull_at(self.hull["target"], t_pose)
        t_face_z = (float(t_hull_w[:, 2].min()), float(t_hull_w[:, 2].max()))
        gate = self._brace_gate or {"ok": None, "why": "no gate verdict"}
        try_brace = gate.get("ok") is not False  # False = no joint two-arm placement at this trunk: push-only from the start
        ignore_push = self.support_objects() + [self.obj("target")]
        ignore_brace = self.support_objects() + [self.obj("partner")]
        push_chosen = None  # (kind, pre, init, goal, extras, q of the right arm)
        brace_chosen = None  # (pre, contact, kind, extras)
        # the collision world holds the target at its INIT pose, but the push happens after the place: the box is posed at the
        # place for the chain's checks (the pre-contact of the placed box lies inside the box at its init pose otherwise -
        # 0918-170804: 'exact IK failed at pose 1 of 3' on 6/12 attempts)
        with self.posed("target", t_pose):
            for push_kind, push_init, push_extras in self.touching_push_candidates(manip, p_init, t_face_z, direction):
                push_goal = (push_init[0] + direction * dist, push_init[1])
                push_pre = IK.pre_pose(push_init)
                # (0) the hand beyond the finger profile (wrist camera, knuckles, body) clear of the box at the contact and the goal
                overlap = self.push_hand_overlap(push_init, push_goal, t_pose, direction, dist)
                if overlap is not None:
                    why.append(f"{push_kind}: {overlap}")
                    continue
                # (1) the push chain on its own - the requirement (brace optional, owner 2026-09-16 17:10: the axial rule)
                r_push = self.ikc.exact([push_pre, push_init, push_goal], manip, "right", ignore_for_last=ignore_push, cache=False, fingers=IK.FINGER_CLOSED)
                if not r_push.ok:
                    why.append(f"{push_kind}: right: {r_push.reason}")
                    continue
                # the oracle's retreat after the push ends 10 cm back along the approach axis from the push GOAL (above it for a
                # blade): a class whose clear pose has no configuration dies in the program check (9/16 of 0918-173814)
                r_ret = self.ikc.exact([IK.pre_pose(push_goal)], manip, "right", ignore_for_last=ignore_push, cache=False, fingers=IK.FINGER_CLOSED)
                if not r_ret.ok:
                    why.append(f"{push_kind}: retreat clear pose: {r_ret.reason}")
                    continue
                push_chosen = (push_kind, push_pre, push_init, push_goal, push_extras)
                # (2) a brace when the joint band gate admitted the position: the first two-arm candidate that passes
                if try_brace:
                    for pre, contact, kind, extras in self.touching_brace_candidates(manip, b_point, face_z, -direction):
                        res = self.ikc.exact_both_arms(("right", [push_pre, push_init, push_goal]), ("left", [pre, contact]), manip,
                                                       ignore_push, ignore_brace, fingers=IK.FINGER_CLOSED)
                        if res.ok:
                            brace_chosen = (pre, contact, kind, extras)
                            break
                        why.append(f"{push_kind}/{kind}: {res.reason}")
                break
        if push_chosen is None:
            d_i, l_i = manip.local(G.t(p_init)[:2])
            raise Reject("contact_unreachable", f"touching push: no push candidate passes the exact IK from the {manip.name} stance (push init at depth {d_i:.2f} "
                                                f"lateral {l_i:+.2f} z {float(p_init[2]):.2f}, face z {t_face_z[0]:.2f}-{t_face_z[1]:.2f}, trunk {manip.translate}, "
                                                f"{manip.tilt:.2f}; {'; '.join(why[-3:])})")
        push_kind, push_pre, push_init, push_goal, push_extras = push_chosen
        draft.summary.setdefault("touching", {}).update({"push_kind": push_kind, "classes_failed": why[:6], **{k: v for k, v in push_extras.items() if v is not None}})
        log.info("%s: touching push class %s from the %s stance%s", self.gp.alias, push_kind, manip.name, f" (failed before it: {'; '.join(why[:3])})" if why else "")
        push_rec = ContactRecord("push", "right", push_pre, push_init, None, push_goal, direction, dist, object=tname, extras=dict(push_extras, pose_class=push_kind))
        push_rec.stance = manip.name  # the stance the chain was checked from ("manipulation" or the second base pose "push")
        self.sidecar.add_contact(push_rec)
        if brace_chosen is not None:
            pre, contact, kind, extras = brace_chosen
            brace_rec = TT.BraceRecord("brace", "left", pre, contact, direction=-direction, brace_kind=kind, object=pname, extras=extras)
            brace_rec.stance = manip.name
            self.sidecar.add_contact(brace_rec)
            draft.summary["brace"] = {"present": True, "kind": kind, "gate": gate, **{k: v for k, v in extras.items() if v is not None}}
            manip.arms = ["right", "left"]
            manip.contacts = ["push", "brace"]
        else:
            # push-only touching: ``contacts.brace`` absent, the manipulation stance right-arm only; the program is
            # transition -> push -> retreat (oracle agent 7deabbed); ``summary.brace.present`` = False, why recorded
            draft.summary["brace"] = {"present": False, "gate": gate, "candidates_failed": len([w for w in why if "/" in w]),
                                      "why": "; ".join(why[-3:]) if try_brace else str(gate.get("why"))}
            manip.arms = ["right"]
            manip.contacts = ["push"]
            self.sidecar.contacts.pop("brace", None)

    def touching_push_candidates(self, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], direction: th.Tensor):
        """Push poses of the closed right gripper at the target's right face (``face_point``, mid height), pushing along
        ``direction`` (robot's left, into the partner), in the order to try: (1) the pads on the face — pad centre on the
        point, approach along ``direction``, closing horizontal then vertical; (2) the top-down blade beside the face
        (``tabletop.blade_standoff`` over the target's face height above the tips, pitched per ``approach_pitch_order``,
        pads' faces parallel to the pushed face, either wrist turn) — a sideways pad push 10 cm right of the target was
        out of reach for the right arm on every attempt of the sanitary-napkins loop. Yields ``(kind, eef pose, extras)``."""
        d = C.unit(G.t(direction))
        outward = -d  # the pushed (right) face's outward normal
        z0, z1 = face_z
        clearance = float(self.gen("stop_clearance_m", STOP_CLEARANCE))
        # A/B knob: pad contacts lifted by the constant (pad centre + half the finger height below the face top), blades lifted
        # with the fingertip kept 1 cm below the face top
        pad_lift, pad_why = contact_height_extra(self, "push", pad_center_z=float(face_point[2]), band_top_z=z1)
        fp = G.t(face_point) + th.tensor([0.0, 0.0, pad_lift])
        pad_facts = {"height_extra_m": requested_height_extra(self), "height_extra_effective_m": round(pad_lift, 4), "height_extra_clip_reason": pad_why}
        yield "push_h", IK.contact_eef(fp, d), {"blade_half_thickness_m": None, **pad_facts}
        yield "push_v", IK.contact_eef(fp, d, closing=th.tensor([0.0, 0.0, 1.0])), {"blade_half_thickness_m": None, **pad_facts}
        raised_fallback = None
        # owner 2026-09-18 15:50: a sideways push uses the VERTICAL blade only - ``pitched_approach`` leans the blade about the
        # push axis, so a pitched finger side meets the face obliquely and yaws the box (0918-153503: 45 deg blade, 29 deg yaw,
        # contact lost); ``generation.push_blade_pitches_deg`` lists the admissible pitches
        pitches = [float(p) for p in (self.gen("push_blade_pitches_deg", None) or [0.0])]
        for pitch in [p for p in TT.approach_pitch_order(None) if p in pitches]:
            z_tip = z0 + 0.005
            if bool(self.gen("blade_toward_com", False)):  # off since 2026-09-18 16:00: the 2 cm raise put the recorded retreat's clear pose at the reach limit (0918-154421: 4/8 'retreat: exact IK failed'); the oracle raises the pad at run time for the palm
                # owner 2026-09-17 23:39: push nearer the centre of mass - the finger section (centre ~3.5 cm above the
                # tips) is centred on the face's mid height when the face allows it, else it stays at the face bottom
                z_tip = max(z_tip, min(float(face_point[2]) - 0.035, z1 - 0.03))
            blade_lift, blade_why = contact_height_extra(self, "blade", fingertip_z=z_tip, band_top_z=z1)
            z_tip += blade_lift
            # a push needs the blade on the lower part of the face only (unlike a stop brace, which the whole face may
            # meet): faces taller than the finger section are pushed over their lowest ``PALM_FROM_TIP`` (7 cm)
            # (the finger section is PALM_FROM_TIP long along the finger axis: pitched, it covers cos(pitch) of that in height)
            z_face_hi = min(z1, z_tip + (TT.PALM_FROM_TIP - 0.005) * math.cos(math.radians(pitch)))
            # owner 2026-09-17 23:39 (touching 0917-22*: the palm pressed the sanitary-napkins box and shook the table): a
            # face that reaches the palm section of a vertical blade (z1 above the fingers' 7 cm) is only pushed by a
            # PITCHED blade - the palm leans away from the face by ~ PALM_FROM_TIP·sin(pitch) - never by the 0° one,
            # whose stand-off (finger profile only) puts the wider palm inside the face
            # (default 0 = off since 2026-09-18 15:45: ``pitched_approach`` leans the blade FORWARD, about the push axis, so
            # a pitched blade does not move the palm away from a face lying to the pad's side - the 45 deg blade of
            # 0918-144517 knocked the box over; the oracle raises a vertical blade until its palm clears the face top)
            if z1 > z_tip + TT.PALM_FROM_TIP and pitch < float(self.gen("blade_palm_min_pitch_deg", 0.0)):
                # fallback (after the pitched blades, which come first in the order): the vertical blade RAISED so that its
                # palm clears the face top - the finger section pushes the top PALM_FROM_TIP of the face (re-sample
                # 0918-010325: every pitched blade fails the exact IK from the stance)
                z_hi_tip = z1 - TT.PALM_FROM_TIP + 0.005
                if bool(self.gen("blade_palm_clear_fallback", True)) and z_hi_tip <= z1 - 0.03:
                    raised_fallback = (z_hi_tip,)  # yielded after the pitched blades (below)
                continue
            leans = self.gen("blade_lean_deg", [20.0, 12.0, 8.0])
            leans = [float(v) for v in (leans if isinstance(leans, (list, tuple)) else [leans]) if float(v) > 0]
            if pitch == 0.0 and z1 > z_tip + TT.PALM_FROM_TIP and leans:
                # owner 2026-09-18 19:05: a tall face is pushed LOW by a blade leaning along the push axis (fingertips lead, the
                # palm recedes from the face by h·tan(lean)): the raised blade pushed the top 4 cm of the 11 cm box and it
                # tipped instead of sliding - each base nudge closed 10.9 mm and the same 10.9 mm reopened at the back-off
                # (0918-184752: touching held during the push, lost at the last step). Leans in order: 20 deg has no pre-pose
                # IK at these stances (0918-191811, the wrist roll at its limit); at 12 deg the thick finger section touches at
                # ~ the centre-of-mass height and the palm still recedes 1.5 cm.
                # The tips ``blade_lean_tip_above_m`` above the face bottom: leaning, the fingertip's leading edge dips below the tip
                # height and 5 mm above the table collided with it at the contact pose (0918-190236)
                z_lean_tip = z0 + float(self.gen("blade_lean_tip_above_m", 0.025))
                for lean in leans:
                    yield from self._leaning_blade(manip, face_point, (z0, z1), d, outward, clearance, z_lean_tip, lean)
            if pitch == 0.0 and z1 > z_tip + TT.PALM_FROM_TIP and bool(self.gen("blade_palm_clear_first", True)):
                # owner 2026-09-18 18:45: a face taller than the finger section is pushed by the RAISED vertical blade first (palm
                # ``blade_palm_clear_m`` above the face top, validated by the chain check) - the low blade's palm overlaps the top
                # and the oracle raised it 3.9 cm at run time to an unvalidated pose: 'joints did not converge 0.087 rad'
                # (0918-182409); with 5 mm of clearance the wider palm / knuckle came down ON the box's top edge as the blade
                # descended beside the face (0918-183535, same error) - 2.5 cm by default
                palm_clear = float(self.gen("blade_palm_clear_m", 0.025))
                yield from self._raised_blade(manip, face_point, (z0, z1), d, outward, clearance, z1 - TT.PALM_FROM_TIP + palm_clear)
            so = TT.blade_standoff((z0, z_face_hi), z_tip, pitch, clearance, side=False)
            if so is None:
                continue  # nothing of the face above the tips: no blade push
            standoff, half = so
            a_vec = TT.pitched_approach(pitch, manip.forward)
            origin = th.tensor([float(face_point[0]), float(face_point[1]), z_tip + C.FINGERTIP_LENGTH * math.cos(math.radians(pitch))]) + outward * standoff
            for c_sign in PUSH_WRIST_TURNS:
                yield f"push_blade_{pitch:.0f}{'' if c_sign > 0 else '-'}", (origin, C.eef_quat(a_vec, c_sign * outward)), {
                    "blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": z_tip,
                    "height_extra_m": requested_height_extra(self), "height_extra_effective_m": round(blade_lift, 4), "height_extra_clip_reason": blade_why}
        if raised_fallback is not None:
            yield from self._raised_blade(manip, face_point, (z0, z1), d, outward, clearance, raised_fallback[0])

    def _leaning_blade(self, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], d: th.Tensor, outward: th.Tensor, clearance: float,
                       z_tip: float, lean_deg: float):
        """The closed blade with its fingertips ``z_tip`` (5 mm above the face bottom) and the hand leaning ``lean_deg`` along
        the push direction (approach = down rotated toward ``d``): the fingertips lead, every higher finger section recedes
        from the face by h·tan(lean), so only the tips touch (below the centre of mass: the box slides, no tipping) and the
        palm never reaches the face. Stand-off = the largest protrusion of the finger profile toward the face + clearance."""
        z0, z1 = face_z
        theta = math.radians(float(lean_deg))
        d3 = th.tensor([float(d[0]), float(d[1]), 0.0])
        d3 = d3 / max(float(th.linalg.norm(d3)), 1e-9)
        a_vec = C.unit(th.tensor([0.0, 0.0, -math.cos(theta)]) + d3 * math.sin(theta))
        protrude, h_prev = 0.0, 0.0
        for h, half in TT.FINGER_PROFILE:  # a section spans (h_prev, h]; its lowest point protrudes the most
            protrude = max(protrude, float(half) - h_prev * math.tan(theta))
            h_prev = float(h)
        standoff = protrude + float(clearance)
        # eef origin = fingertip - a·FINGERTIP_LENGTH, the tips ``standoff`` outside the face at ``z_tip``
        tip = th.tensor([float(face_point[0]), float(face_point[1]), float(z_tip)]) + outward * standoff
        origin = tip - a_vec * C.FINGERTIP_LENGTH
        for c_sign in PUSH_WRIST_TURNS:
            yield f"push_blade_lean{lean_deg:.0f}{'' if c_sign > 0 else '-'}", (origin, C.eef_quat(a_vec, c_sign * outward)), {
                "blade_half_thickness_m": float(TT.FINGER_PROFILE[0][1]), "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": float(z_tip),
                "palm_clear": True, "lean_deg": float(lean_deg), "height_extra_m": requested_height_extra(self), "height_extra_effective_m": 0.0,
                "height_extra_clip_reason": "leaning_blade"}

    def _raised_blade(self, manip: Stance, face_point: th.Tensor, face_z: Tuple[float, float], d: th.Tensor, outward: th.Tensor, clearance: float, z_tip: float):
        """The vertical (0 deg) blade with its fingertips at ``z_tip`` = face top − PALM_FROM_TIP: the palm stands above
        the face, the finger section covers the face's top part. Yielded after every pitched blade failed."""
        z0, z1 = face_z
        so = TT.blade_standoff((z0, z1), z_tip, 0.0, clearance, side=False)
        if so is None:
            return
        standoff, half = so
        a_vec = TT.pitched_approach(0.0, manip.forward)
        origin = th.tensor([float(face_point[0]), float(face_point[1]), z_tip + C.FINGERTIP_LENGTH]) + outward * standoff
        for c_sign in PUSH_WRIST_TURNS:
            yield f"push_blade_0_high{'' if c_sign > 0 else '-'}", (origin, C.eef_quat(a_vec, c_sign * outward)), {
                "blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": z_tip,
                "palm_clear": True, "height_extra_m": requested_height_extra(self), "height_extra_effective_m": 0.0, "height_extra_clip_reason": "palm_clear"}

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

        pad_lift, pad_why = contact_height_extra(self, "brace", pad_center_z=float(b_point[2]), band_top_z=z1)  # A/B knob
        pad_facts = {"height_extra_m": requested_height_extra(self), "height_extra_effective_m": round(pad_lift, 4), "height_extra_clip_reason": pad_why}

        def side_contacts():
            point = G.t(b_point) + outward * clearance + th.tensor([0.0, 0.0, pad_lift])
            for c_axis, suffix in ((z_axis, "v"), (side_h, "h")):
                name = TT.contact_class(manip, a, c_axis) or f"side_{suffix}"
                for c_sign in (1.0, -1.0):
                    contact = IK.contact_eef(point, a, c_sign * c_axis)
                    yield IK.pre_pose(contact), contact, f"{name}{'' if c_sign > 0 else '-'}", {"stop_clearance_m": clearance, "pose_class": name, "blade_half_thickness_m": None, **pad_facts}

        def blades():
            for pitch in TT.approach_pitch_order(None):
                z_tip = z0 + 0.005
                blade_lift, blade_why = contact_height_extra(self, "blade", fingertip_z=z_tip, band_top_z=z1)
                z_tip += blade_lift  # the fingertips just above the support: the fingers span the face height
                so = TT.blade_standoff((z0, z1), z_tip, pitch, clearance, side=False)
                if so is None:
                    continue  # the face reaches the palm section: no blade contact
                standoff, half = so
                a_vec = TT.pitched_approach(pitch, manip.forward)
                origin = th.tensor([float(b_point[0]), float(b_point[1]), z_tip + C.FINGERTIP_LENGTH * math.cos(math.radians(pitch))]) + outward * standoff
                for c_sign in (1.0, -1.0):
                    contact = (origin, C.eef_quat(a_vec, c_sign * outward))
                    yield IK.pre_pose(contact), contact, f"blade_{pitch:.0f}{'' if c_sign > 0 else '-'}", {
                        "blade_half_thickness_m": half, "stop_clearance_m": clearance, "blade_standoff_m": standoff, "fingertip_z": z_tip, "pose_class": f"blade_{pitch:.0f}",
                        "height_extra_m": requested_height_extra(self), "height_extra_effective_m": round(blade_lift, 4), "height_extra_clip_reason": blade_why}

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
