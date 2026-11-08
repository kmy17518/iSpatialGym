"""Configuration matching — attachment (spec §6.8; init per §4 with the attachment bullets of §4.4 / §4.6).

Roles: ``target`` (the male part, grasped with the attachment pipeline's library grasps), ``partner`` (the female
part, upright), scene ``support``, ``floor``; the v1 facade names them ``male`` / ``female``.

Template: both parts get the ``attachable`` ability at spawn; the male / female meta-link pair (``AttachedTo``
parent candidates), the collision filter between the two members (applied once, restored by the loader), the
mating-axis symmetry test (5 mm slabs, a rotation is accepted when >= 90 % of the slabs match) -> ``orientation_
equivalences`` for both members; two reach bands — the male's grasp band and the **attach-reach band** of the partner
(the partner posed at a cell, grasp set = the attach eef poses ``T_partner ∘ T_F ∘ inv(T_M) ∘ T_obj→eef`` of the male
grasps, so a cell is in it when one (partner yaw, attach eef pose) pair is reachable ignoring the target - the band marks
a cell with its first feasible pair, like any library-grasp band); Phases 1a / 1b (H_max = the
taller part) / 1c (row ``attachment``: max(w_target, w_partner) + 0.02) / 2 (male half-extent).

Instance: the male at the focus from F (Phase 3 step 1); the partner on a cell of the attach-reach band (gap >= 0.04,
centre distance >= 0.08, on the top, in frame); the attach pose ``T_partner ∘ T_F ∘ inv(T_M)`` and the **pre-attach**
10 cm back along the female meta-link's insertion axis, exact-IK-checked with the male in hand (partner and support
ignored for the attach pose only); the in-hand reorientation from the lifted grasp pose to the pre-attach pose at 3
SLERP waypoints (§3.9; ``generation.carry_over_partner_m``: a vertical rise over the partner first); the **partner brace** (closed left gripper on the partner's face opposite to the insertion
direction; pre-contact + contact IK with the target arm locked at the attach solution, combined whole-body check).
Goal: partner unchanged, target attached (verified with ``AttachedTo.set_value(True)`` at the goal check); group
target with the fixed assignment, tolerances 0.01 / 0.15, equivalences. Sidecar: ``init`` / ``manipulation`` stances,
the transition, one grasp record (pre-grasp, grasp, carry_height, ``waypoints`` = the SLERP key poses, ``pre_place`` =
pre-attach, ``place`` = attach, ``place_on`` = the partner), ``contacts.brace`` (``partner_face``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling import predicates as PRED
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import orient as O
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import PAD_BAND_MARGIN, SamplerV2, register
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance, Transition
from ispatialgym.sampling.v2.tasks.iors_small import HAND_SPHERES_EEF  # the hand beyond the finger profile (wrist camera, knuckles, body)

import omnigibson as og  # noqa: E402
from omnigibson.object_states import AttachedTo  # noqa: E402
import omnigibson.utils.transform_utils as T  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASK = "configuration_matching-attachment"
#: per-attempt budget (owner 2026-09-16 10:47 item 3): the pilot's cap-lemonade attempt spent 1457 s in the per-cell exact
#: IK with the attached male; a candidate now screens every admissible partner cell in one batch and solves exactly at most
#: PARTNER_EXACT_CAP of them, the attempt stops at ATTEMPT_BUDGET_S (``generation.attempt_budget_s`` overrides)
ATTEMPT_BUDGET_S = 360.0  # the second base pose search (owner 17:10) adds up to SECOND_BASE_MAX_TRIES attach evaluations
PARTNER_EXACT_CAP = 6
#: the attach-band cells nearest the focus that a stance offers the partner (``_attach_and_brace``; the template's ``_partner_room``)
PARTNER_CELLS_TRIED = 40
#: partner yaws ``_partner_room`` tries per cell (the instance draws one uniformly)
PARTNER_ROOM_YAWS = 8
#: §6.8 second base pose: lateral shifts of the grasp stance along the edge (m, + = robot's left) tried in this order
SECOND_BASE_LATERALS = (0.10, -0.10, 0.20, -0.20, 0.30, -0.30)
SECOND_BASE_MAX_TRIES = 6
FOCUS_DEPTH_PREFERRED = 0.62  # base-forward depth of the manipulation focus beyond which the lift / SLERP fail (att_diag2 21:04)
GOOD_TRUNKS = ((0.5, -0.45), (0.5, 0.0), (0.25, -0.3))  # (translate, tilt) verified for pick + pre-poses by the block probes
#: score penalty per earlier rejection of a manipulation candidate within the template (owner 2026-09-25, cap 0925-220515): one rejection
#: outweighs any depth / trunk preference (depth excess <= ~0.3 m, trunk penalty <= ~0.5), so the next attempt's window of six moves on
CANDIDATE_REJECT_PENALTY = 1.0
CARRY_HEIGHTS = (0.12, 0.08, 0.05)  # the lift above the grasp: the first whose lift + SLERP pass the IK
#: thin-part pick (owner 2026-09-20 20:41, cap/xxqyrt 1.1 cm): a male shorter than the pads' fingertip section (+ the band margin)
#: is a fingertip pinch whatever the height; its rule is on the support, not on the male's top - the pads' fingertip-side end
#: stays >= PAD_TABLE_CLEARANCE_M above the support surface (the pinch band = the rim's upper half), and the tilted rim pinches
#: (approach >= THIN_PART_TILT_MIN_DEG from vertical) are screened before the vertical ones
THIN_PART_MAX_HEIGHT_M = float(C.PAD_Z_RANGE[1]) + PAD_BAND_MARGIN
PAD_TABLE_CLEARANCE_M = 0.005
THIN_PART_TILT_MIN_DEG = 15.0
#: every attachment grasp: the lowest vertex of the finger meshes (both fingers, any opening) stays this far above the
#: support at the grasp pose (0920-203052: 0.1 mm below the table -> 22 contact steps at 110 N)
FINGER_SUPPORT_CLEARANCE_M = 0.004
#: after the insertion the pads' fingertip-side end (pad pinch) / the fingertip axis point (fingertip pinch) stay this far above
#: the partner's top along the insertion axis (owner 2026-09-20 23:39 item 2b; cork/wine 0920-212757)
PARTNER_CLEARANCE_M = 0.005
#: how far below the female link a partner point under the fingertips can still matter to them (the pads' span + the fingertips
#: beyond a male held at its top): deeper hits are the partner's body, not a top the tips could meet (owner 2026-09-25 cork rule)
TIP_REACH_BELOW_LINK_M = 0.06
#: an over_partner male's fingertips count as standing ON the partner's top only with at least this much partner surface under them
#: (from the pad face outward); less is the rim's outer edge under the finger's body - no partner floor (cap/xxqyrt: 1.5 mm)
TIP_EDGE_OVERLAP_M = 0.004
#: the partner's top under the FINGERTIPS (owner 2026-09-24 item 4): partner hull points within this of a fingertip's radial distance
#: from the insertion axis count - the fingertip section is 6.5 mm across the closing axis (r1pro_model.finger_hulls_eef), so the
#: band is the tip's own radial footprint (+- 3.25 mm) plus 1 mm; the cap's tips (r ~26 mm) miss the lemonade bottle's neck lip, the
#: cork's tips (r ~9 mm) still stand over the wine bottle's rim (inner radius ~9 mm)
TIP_RADIAL_BAND_M = 0.00425
#: rest orientation by stability (owner 2026-09-24 item 5): the male's library rest orientation is flipped 180 deg about a
#: horizontal axis when the other end carries the mass centre with a tipping margin this many times larger; the lowest hull
#: vertices within REST_SUPPORT_TOL_M of the bottom form the support facet
REST_FLIP_MARGIN_RATIO = 1.3
REST_SUPPORT_TOL_M = 0.002
#: a library rest whose tipping margin (support inradius / mass-centre height, the tangent of the toppling tilt) reaches this is
#: SOUND and is never flipped on geometry alone (owner 2026-09-25: the flat cap, margin 3.4, was turned open side up for a 1.6x
#: "better" margin and lost every grasp - 0925-144739 'grasp_unreachable' 8/8); below it the flip is weighed as before
REST_SOUND_MARGIN = 0.5


def carry_over_ladder(value) -> Tuple[float, ...]:
    """``generation.carry_over_partner_m`` as the clearances (m) the carry over the partner tries in order (a number = one); () = off:
    absent, null, a boolean, 0 or an empty list (the lift ladder ``CARRY_HEIGHTS`` and the straight line to the pre-attach)."""
    if value is None or isinstance(value, bool):
        return ()
    return tuple(float(v) for v in (value if isinstance(value, (list, tuple)) else [value]) if v is not None and float(v) > 0.0)


def v2_library_lift_off(grasp_key: Optional[str]) -> bool:
    """i7-attachfix: the male's grasps come from a v2 library entry (validated at their own height): the alias's generation lift
    ``grasp_height_extra_m`` is not applied to them (``_apply_grasp_height_extra``; its floors still are)."""
    if not grasp_key:
        return False
    from ispatialgym.sampling import grasp_library as GL

    try:
        return GL.v2_entry(grasp_key) is not None
    except Exception:  # noqa: BLE001 - no v2 root / unreadable entry: as before
        return False


#: i7-cork 2026-11-08 (owner decision 4.2, attachment_03's cork): the male's alternative ROUTES, each behind its own DEFAULT-OFF switch.
#: ``lying90`` (``ISG_ATTACH_LYING90=1``): the male rests on its side (the v2 entry served by ``<key>__lying90``: its stable lying rest and
#: its grasps), the grasp, the lift and the recorded SLERP carry turn it 90 deg so the male end points along the insertion axis - the
#: attachment program as it is (``grasps[0].attach_route``). ``handover`` (``ISG_ATTACH_HANDOVER=1``): the male rests wide end down
#: (``<key>__handover_pick``), the right hand grasps its narrow (mating) end, brings it to a hand-over pose in front of the robot with the wide
#: end toward the left hand, the left hand grasps the wide end (``<key>__handover_hold``: fingers off the mating side), the right hand lets
#: go, the base moves to the place pose (heading kept, axis legs) and the left hand attaches. A route whose entry is not in the v2 library
#: is not taken (counter ``attach_route.<route>.no_entry``): the alias keeps the current route.
ATTACH_ROUTE_SWITCHES = {"lying90": "ISG_ATTACH_LYING90", "handover": "ISG_ATTACH_HANDOVER"}
ATTACH_ROUTE_KEYS = {"lying90": ("__lying90",), "handover": ("__handover_pick", "__handover_hold")}


#: the hand-over (route i): the male's centre in the grasp stance's base frame (forward, left) and above the support top (m), its turns
#: about its own axis, the candidates an attempt evaluates with the exact IK, the two hands' smallest hull distance, the place stances'
#: stand-off shifts (the lateral ones are SECOND_BASE_LATERALS)
HANDOVER_FORWARD_M = (0.42, 0.50, 0.58)
HANDOVER_LEFT_M = (0.0, -0.06, 0.06)
HANDOVER_RISE_M = (0.12, 0.18, 0.24)
#: the male's axis tilted from the robot's left toward the world's up (+) / down (-) at the hand-over (deg)
HANDOVER_TILTS_DEG = (0.0, -25.0, 25.0)
HANDOVER_ROLLS = 8
HANDOVER_TRIES = 40
HANDOVER_HANDS_MIN_M = 0.003
HANDOVER_STANDOFF_SHIFTS_M = (-0.10, 0.10)


def attach_route_requested() -> Optional[str]:
    """The route a switch asks for (None: the current route). Both switches on is a configuration error."""
    on = [r for r, env in ATTACH_ROUTE_SWITCHES.items() if os.environ.get(env, "").strip().lower() in ("1", "on", "true", "yes")]
    if len(on) > 1:
        raise ValueError(f"attachment routes {on}: at most one of {sorted(ATTACH_ROUTE_SWITCHES.values())} may be on")
    return on[0] if on else None


def attach_route_keys(base_key: str, route: str) -> Tuple[str, ...]:
    """The grasp keys a route reads (``<base key><suffix>``, served by the v2 entries whose ``legacy_keys`` list them)."""
    return tuple(f"{base_key}{sfx}" for sfx in ATTACH_ROUTE_KEYS[route])


@dataclass
class CarryPlan:
    """The carry from the lifted grasp to the pre-attach that :meth:`Attachment._carry_to_pre_attach` settled on (or tried last):
    the tracked chain's result, the lift pose, the recorded SLERP waypoints (pre-attach last), the lift height, the over-the-partner
    facts (None with the carry over the partner off) and the ladder for the log."""

    result: Any
    lift: Optional[TT.Pose]
    waypoints: List[TT.Pose]
    carry: Optional[float]
    over: Optional[Dict[str, Any]]
    ladder: str


def _support_margin(pts: th.Tensor, com: th.Tensor) -> Tuple[float, float, int]:
    """``(tipping margin, support radius, n support vertices)`` of a body resting on its lowest hull vertices: the distance
    from the mass centre's ground projection to the nearest edge of the support facet's convex outline (negative outside),
    over the mass centre's height above the facet - the tangent of the angle the body may be tilted before it topples."""
    z = pts[:, 2]
    low = pts[z <= float(z.min()) + REST_SUPPORT_TOL_M][:, :2]
    h = max(float(com[2]) - float(z.min()), 1e-6)
    if low.shape[0] < 3:
        if low.shape[0] == 2:  # a knife edge: the margin along the edge only
            a, b = low[0], low[1]
            d = b - a
            t = float(th.dot(com[:2] - a, d) / max(float(th.dot(d, d)), 1e-12))
            inside = 0.0 <= t <= 1.0
            return (min(t, 1.0 - t) * float(d.norm()) if inside else -1.0) / h, 0.0, 2
        return -1.0 / h, 0.0, int(low.shape[0])
    try:
        from scipy.spatial import ConvexHull

        poly = low[ConvexHull(low.numpy()).vertices]
    except Exception:  # noqa: BLE001 - degenerate (collinear) support
        return -1.0 / h, 0.0, int(low.shape[0])
    n = poly.shape[0]
    centre = poly.mean(dim=0)
    order = th.argsort(th.atan2(poly[:, 1] - centre[1], poly[:, 0] - centre[0]))
    poly = poly[order]
    margin = float("inf")
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        L = float(e.norm())
        if L < 1e-9:
            continue
        margin = min(margin, float(e[0] * (com[1] - a[1]) - e[1] * (com[0] - a[0])) / L)  # signed: positive inside a CCW polygon
    radius = float((poly - centre).norm(dim=1).max())
    return (margin if margin != float("inf") else -1.0) / h, radius, n


def stable_rest_quat(hull: th.Tensor, rest_quat, com=None, force: bool = False, insertion_axis_local=None, grasps: Sequence[Any] = ()) -> Tuple[th.Tensor, Dict[str, Any]]:
    """The male's rest orientation: the library's ``rest_quat`` by default (the designer's: the grasps are annotated for it and
    the attachment program does no in-hand 180 deg turn). Owner 2026-09-25: it is flipped 180 deg only when the nudge test
    toppled the male on that end in a previous draw (``force``) AND the flip keeps the insertion direction - the turn is about
    the insertion axis's horizontal projection, so a male inserted vertically (cork, cap) is never flipped - AND at least one
    library grasp still approaches from above afterwards (``grasps``: rows with an eef pose in the object frame). Tipping margins
    (:func:`_support_margin`; the mass centre = ``com`` when given, else the hull's vertex mean) are reported as facts.
    Returns ``(quat, facts)``."""
    q0 = TT._t(rest_quat)
    hull_t = TT._t(hull)
    c_loc = hull_t.mean(dim=0) if com is None else TT._t(com)
    a_loc = TT._t(insertion_axis_local) if insertion_axis_local is not None else th.tensor([0.0, 0.0, 1.0])
    a_w = C.quat_rotate(q0, a_loc / max(float(a_loc.norm()), 1e-9))
    horiz = a_w.clone()
    horiz[2] = 0.0
    out: Dict[str, Any] = {"com_source": "vertex mean" if com is None else "link centre of mass", "flipped": False}
    if float(horiz.norm()) >= 0.9:
        flip_axis = horiz / float(horiz.norm())  # a turn about the insertion direction keeps it
    else:
        flip_axis = th.tensor([1.0, 0.0, 0.0])  # vertical insertion: any horizontal turn reverses it (reported, never applied)
    q1 = C.qmul(C.quat_from_axis_angle(flip_axis, math.pi), q0)
    margins = []
    for tag, q in (("canonical", q0), ("flipped", q1)):
        pts = C.transform_points((th.zeros(3), q), hull_t)
        com_w = C.quat_rotate(q, c_loc)
        m, r, n = _support_margin(pts, com_w)
        margins.append(m)
        out.update({f"margin_{tag}": round(m, 4), f"support_radius_{tag}_m": round(r, 4), f"support_vertices_{tag}": n})
    m0, m1 = margins
    keeps_direction = float(th.dot(C.quat_rotate(q1, a_loc), a_w)) > 0.95
    out["flip_keeps_insertion_direction"] = bool(keeps_direction)
    if not force:
        out["reason"] = f"library rest kept (tipping margin {m0:.2f}; flipped {m1:.2f})"
        return q0, out
    if not keeps_direction:
        out["reason"] = "the nudge test toppled the library end, but a flip would reverse the insertion direction (no in-hand turn): kept"
        return q0, out
    # grasps that still approach from above once the male is turned (the eef +z, wrist -> fingertips, pointing down in the world)
    down = 0
    for g in grasps or ():
        try:
            gq = TT._t(g.pose[1]) if hasattr(g, "pose") else TT._t(g[1])
            a_eef = C.quat_rotate(C.qmul(q1, gq), th.tensor([0.0, 0.0, 1.0]))
            down += int(float(a_eef[2]) < -0.2)
        except Exception:  # noqa: BLE001
            continue
    out["grasps_from_above_after_flip"] = down
    if grasps and down == 0:
        out["reason"] = "the nudge test toppled the library end, but no library grasp approaches from above once turned: kept"
        return q0, out
    if m1 <= 0.0:
        out["reason"] = "the nudge test toppled the library end, but the other end has no support: kept"
        return q0, out
    out["flipped"] = True
    out["reason"] = f"the nudge test toppled the library end (previous draw); turned about the insertion direction (margin {m0:.2f} -> {m1:.2f})"
    return q1, out


#: rest-stability nudge test of the male / partner after the init settle (cork/ncxgpe 0920-200850 toppled off its tapered end)
REST_NUDGE_DEG = 2.0
REST_STABILITY_STEPS = 300
#: a stable part relaxes back below the nudge; one that has drifted past it is rocking over (cork probe 0920-212757: 7.5 deg on
#: attempt 1, 4.5 deg on attempt 2 - both toppling, the second slipped under a 5 deg bound)
REST_TIP_MAX_DEG = 3.0


def attachment_links(male, female):
    """``(male child link, female parent link)`` of the pair - the first parent candidate ``AttachedTo`` offers for ``female`` - or None
    (i7-attachmap 2026-11-07: a module function, so the attach map's pair measurement (``attach_maps.measure_pair``) resolves the pair
    exactly as the template does)."""
    cands = male.states[AttachedTo]._get_parent_candidates(female)
    if not cands:
        return None
    for child_name, parent_names in cands.items():
        for parent_name in parent_names:
            return male.states[AttachedTo].links[child_name], female.states[AttachedTo].links[parent_name]
    return None


#: the insertion-axis rule's defaults (``generation.insertion_axis_min_offset_m`` / ``insertion_axis_max_deviation_deg``)
INSERTION_AXIS_MIN_OFFSET_M = 0.02
INSERTION_AXIS_MAX_DEVIATION_DEG = 45.0


def insertion_axis_rule(parent_link_in_female, partner_hull: th.Tensor, min_offset_m: float = INSERTION_AXIS_MIN_OFFSET_M,
                        max_deviation_deg: float = INSERTION_AXIS_MAX_DEVIATION_DEG) -> Tuple[th.Tensor, Dict[str, Any]]:
    """The pair's insertion axis in the female root frame and its template facts (i7-attachmap 2026-11-07: the rule of
    :meth:`Attachment.sample_template`, moved here unchanged so the attach map measures it by the same code): the direction from the
    partner's hull centroid to the female meta link (the side from which the male can approach without passing through the partner: up
    for a bottle neck / socket / pot top, away from the wall for a nail, the underside for a deck) when it is well-defined (offset >
    ``min_offset_m``) and deviates more than ``max_deviation_deg`` from the meta link's +z; the meta link's +z otherwise. BEHAVIOR's
    attachment frames need only match between male and female, not point along the insertion (cork/ncxgpe -> wine_bottle/inkqch: both
    frames (0.5, -0.5, 0.5, 0.5), +z along the bottle's -y, so the +z rule put the pre-attach 10 cm beside the neck and slid the cork in
    sideways)."""
    axis_geo = TT._t(parent_link_in_female[0]) - TT._t(partner_hull).mean(dim=0)  # both in the female root frame
    axis_z = C.unit(C.quat_rotate(parent_link_in_female[1], th.tensor([0.0, 0.0, 1.0])))
    facts: Dict[str, Any] = {"source": "meta link +z", "axis": [round(float(v), 4) for v in axis_z]}
    axis = axis_z
    if float(th.linalg.norm(axis_geo)) > float(min_offset_m):
        geo = C.unit(axis_geo)
        angle = math.degrees(math.acos(max(-1.0, min(1.0, float(th.dot(geo, axis_z))))))
        facts.update({"centroid_to_female": [round(float(v), 4) for v in geo], "angle_deg": round(angle, 1)})
        if angle > float(max_deviation_deg):
            # the annotation's +z is not the approach direction (a teapot's +z is within 8 deg of it and stays exact)
            axis = geo
            facts["source"] = "centroid->female"
            facts["axis"] = facts["centroid_to_female"]
    return axis, facts


@dataclass
class HoldingTransition(Transition):
    """A recorded stance transition driven with an object in hand (§6.8 second base pose): ``holding`` = the carried bddl
    instance, merged into the sidecar dict (the F6 ``Transition.extra`` keeps unknown keys)."""

    holding: Optional[str] = None

    @staticmethod
    def wrap(tr: Transition, holding: str) -> "HoldingTransition":
        return HoldingTransition(tr.frm, tr.to, tr.stand_off, tr.order, tr.trunk_from, tr.trunk_to, tr.n_samples, holding)

    def to_sidecar(self) -> Dict[str, Any]:
        d = super().to_sidecar()
        if self.holding is not None:
            d["holding"] = self.holding
        return d


@register(SUBTASK)
class Attachment(SamplerV2):
    def __init__(self, ctx):
        super().__init__(ctx)
        if self.gp.bddl is None:
            self.definition_id = 3  # v1 writer: configuration_matching-3
        self.defaults = GD.defaults_for(SUBTASK)
        self.t_role = "target" if "target" in self.gp.roles else "male"
        self.p_role = "partner" if "partner" in self.gp.roles else "female"
        self.top = None
        self.hull: Dict[str, th.Tensor] = {}
        self.rest: Dict[str, th.Tensor] = {}
        #: the previous draw's nudge test toppled the male on its library rest end: the next draws stand it on the other end
        self._rest_flip_forced = False
        self.height: Dict[str, float] = {}
        self.grasps: List[Any] = []
        self.grasp_key: Optional[str] = None
        self.pair = None  # (child link of the male, parent link of the female)
        self.root_in_child_link = None
        self.parent_link_in_female = None
        self.insertion_axis_local = th.tensor([0.0, 0.0, 1.0])
        self.symmetry: List[List[float]] = []
        self.edge_frame: Optional[EdgeFrame] = None
        self.init_stance: Optional[Stance] = None
        self.manip_candidates: List[Stance] = []
        self.band_spec: Optional[TT.BandSpec] = None
        self.bands = None
        self.attach_bands = None
        self.left_bands = None  # A1: the left arm's contact band (partner brace), None until B3's bands exist
        self._att_fails: Dict[str, Any] = {}
        self.brace_missing = False  # the last instance has no partner brace (owner 17:10: optional only without a joint cell)
        #: i7-cork: the route taken (None = the current route; :meth:`_select_attach_route`), the left hand's hold grasps (hand-over)
        self.attach_route: Optional[str] = None
        self.hold_grasps: List[Any] = []
        self.hold_key: Optional[str] = None

    def g(self, key: str):
        return self.gen(key, self.defaults.get(key))

    # ------------------------------------------------------------------ roles

    def v1_role_variants(self) -> Dict[str, str]:
        out = {}
        if self.t_role != "male":
            out["male"] = self.t_role
        if self.p_role != "female":
            out["female"] = self.p_role
        return out

    def bind_scene_roles(self):
        self.bind_floor_v2()
        req = self.choice.bindings.get("support")
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if not names:
            raise Reject("spec", f"{self.gp.variant}: scenes.{self.scene}.bindings.support must list scene object names")
        self.bind_scene_object("support", self.world.obj(names[0]))

    def spawn_roles(self):
        # AttachedTo exists only for "attachable" objects: both parts get the ability (kept from v1 §5.6)
        for role in self.gp.sampleable_roles():
            if role in self.choice.models:
                self.spawn_role(role, self.choice.models[role], extra_abilities={"attachable": {}} if role in (self.t_role, self.p_role) else None)
        for role in self.gp.group_roles():
            for member in self.choice.groups.get(role, []):
                self.spawn_role(role, member)

    @property
    def tname(self) -> str:
        return self.bindings.one(self.t_role).name

    @property
    def pname(self) -> str:
        return self.bindings.one(self.p_role).name

    # ------------------------------------------------------------------ template

    def sample_template(self):
        male, female = self.obj(self.t_role), self.obj(self.p_role)
        if AttachedTo not in male.states:
            raise Reject("spec", f"{male.name} has no AttachedTo state (no attachment meta links?)")
        self.pair = self._attachment_links(male, female)
        if self.pair is None:
            raise Reject("spec", f"{male.name} has no male attachment link matching a female link of {female.name}")
        child_link, parent_link = self.pair
        self.grasps, self.grasp_key, self.rest[self.t_role] = TT.library_grasps(self, self.t_role, "attachment", kinds=["attachment", "tabletop"])
        self._select_attach_route()
        self.rest[self.p_role] = C.IDENTITY_QUAT.clone()
        for role, o in ((self.t_role, male), (self.p_role, female)):
            self.hull[role] = self.world.local_hull(o)
        # owner 2026-09-24 item 5: the male stands on its STABLE end - the library's rest orientation flipped 180 deg about a
        # horizontal axis when that end carries the mass centre with a clearly larger tipping margin (cork/ncxgpe stood on its
        # tapered end: 5-6.5 mm contact radius under a mass centre 1.7 cm up, the 2 deg rest nudge toppled 8/8 draws, 0925-093339;
        # the carry turns the male in hand toward the attach orientation anyway, so the rest end is free). Facts on the template.
        if self.attach_route is not None:
            # i7-cork: a route's rest is its entry's measured rest (lying / wide end down) - never flipped
            flip = {"flipped": False, "route": self.attach_route, "reason": f"route {self.attach_route}: the rest of {self.grasp_key}"}
        elif bool(self.gen("male_rest_by_stability", True)):
            com = None
            try:
                com = male.root_link.center_of_mass  # PhysX's, in the link frame (the cork's is at its origin: the vertex mean sat 5 mm lower)
                if com is not None and (not th.isfinite(TT._t(com)).all() or float(TT._t(com).norm()) > 1.0):
                    com = None
            except Exception:  # noqa: BLE001
                com = None
            # owner 2026-09-25: the library rest by default; a flip only after the nudge test toppled that end (check_rest_stability,
            # previous draw), about the insertion direction (never for a vertical insertion) and with library grasps left that still
            # approach from above (the cap 0925-144739: flipped on geometry alone, open side up, 8/8 'grasp_unreachable')
            a_loc = self._meta_link_axis_local(male)
            self.rest[self.t_role], flip = stable_rest_quat(self.hull[self.t_role], self.rest[self.t_role], com=com, force=bool(self._rest_flip_forced),
                                                            insertion_axis_local=a_loc, grasps=self.grasps)
        else:
            flip = {"flipped": False, "disabled": "generation.male_rest_by_stability false"}
        self.template_facts["male_rest_orientation"] = flip
        if flip.get("flipped"):
            log.info("%s: %s turned onto its other end - %s", self.gp.variant, male.name, flip.get("reason"))
        else:
            log.info("%s: %s keeps the library rest orientation (tipping margin %.2f, other end %.2f; %s)", self.gp.variant, male.name,
                     flip.get("margin_canonical", float("nan")), flip.get("margin_flipped", float("nan")), flip.get("reason", flip.get("disabled")))
        for role in (self.t_role, self.p_role):
            self.height[role] = TT.height_of(self.hull[role], self.rest[role])
        self.grasps = self._recentre_grasps(self.grasps, male)
        # the members collide until they are joined (owner 2026-10-10): the goal check filters the pair at its own attach
        # (apply_goal) and restores the collision afterwards (check_goal_feasible)
        og.sim.step_physics()
        self.kit.sync_world()
        cp, cq = child_link.get_position_orientation()
        rp, rq = male.get_position_orientation()
        self.root_in_child_link = C.relative((rp.float(), rq.float()), (cp.float(), cq.float()))
        fp, fq = female.get_position_orientation()
        pp, pq = parent_link.get_position_orientation()
        self.parent_link_in_female = C.relative((pp.float(), pq.float()), (fp.float(), fq.float()))
        # insertion axis in the female root frame (:func:`insertion_axis_rule`: centroid -> female link when the link's +z deviates)
        self.insertion_axis_local, facts = insertion_axis_rule(self.parent_link_in_female, self.hull[self.p_role],
                                                               float(self.gen("insertion_axis_min_offset_m", INSERTION_AXIS_MIN_OFFSET_M)),
                                                               float(self.gen("insertion_axis_max_deviation_deg", INSERTION_AXIS_MAX_DEVIATION_DEG)))
        self.template_facts["insertion_axis"] = facts
        log.info("%s: insertion axis %s", self.gp.variant, json.dumps(facts))
        self.symmetry = self._symmetry_equivalences(male, child_link)
        support = self.obj("support")
        self.top = self.support_top(support)
        if int(self.top.free.sum()) < 50:
            raise Reject("support_small", f"{support.name}: too little free top")
        mc, pc = self.choice.models[self.t_role], self.choice.models[self.p_role]
        self.band_spec = TT.BandSpec(object=f"{mc.category}/{mc.model}", grasp_key=self.grasp_key, grasps=self.grasps, hull=self.hull[self.t_role],
                                     rest_quat=self.rest[self.t_role], scale=list(mc.scale) if mc.scale else None, height=self.height[self.t_role])
        # attach-reach band of the partner: grasp set = the attach eef poses relative to the partner root
        T_attach = C.compose(self.parent_link_in_female, self.root_in_child_link)  # male root in the female root frame when attached
        pseudo = [TT._BandGrasp(f"attach:{g.name}", C.compose(T_attach, g.pose)) for g in self.grasps]
        attach_spec = TT.BandSpec(object=f"{pc.category}/{pc.model}", grasp_key=f"attach:{self.grasp_key}", grasps=pseudo, hull=self.hull[self.p_role],
                                  rest_quat=C.IDENTITY_QUAT.clone(), scale=list(pc.scale) if pc.scale else None, placed=[f"attach:{mc.category}/{mc.model}"],
                                  height=self.height[self.p_role])
        H_max = max(self.height.values())
        margins = {"kind": "attachment", "w_target": TT.half_extent(self.hull[self.t_role], self.rest[self.t_role]), "w_partner": TT.half_extent(self.hull[self.p_role], C.IDENTITY_QUAT),
                   "safety": 0.02, "H_max": H_max}
        rng = np.random.default_rng(self.ctx.base_seed)
        errors = []
        # i7-attachmap 2026-11-07: under v2 only the attach-reach band is the pair's ATTACH map (attach_maps: the held male at its seat on the
        # partner, pre-attach -> hover -> touchdown -> seat by chained IK, the oracle's hover rule); a miss rejects missing_map (+ MAP_REQUEST).
        # Every other mode loads / computes the legacy band per edge as before.
        if self.attach_route == "handover":
            # i7-cork route (i): the LEFT hand attaches with its wide-end grasp - the partner cells come from the hold entry's left-arm attach map
            if not TT.RB.v2_only():
                raise Reject("spec", "the hand-over route reads the left arm's attach map (v2 only)")
            attach_view = self._attach_map_view(self.hold_key, "left")
        else:
            attach_view = self._attach_map_view() if TT.RB.v2_only() else None
        for edge, st in TT.edge_candidates(self, self.top, rng):
            try:
                bands = TT.ensure_bands(self, self.band_spec, top=self.top, edge=edge)
                self.attach_bands = self._attach_band(attach_spec, edge, attach_view)
                # A1: the left contact band of the partner's height class (partner brace face); §4.4 margins over the union
                self.left_bands = TT.load_left_contact_band(self, self.height[self.p_role], top=self.top, edge=edge, compute=TT.left_band_compute())
                init = TT.phase1b_init_trunk(self, st, H_max, bands=bands, top=self.top, left_bands=self.left_bands)
                init = TT.phase1c_lateral(self, init, margins, rng, top=self.top, bands=bands)
                init.name = "init"
                # D-009 (prepared, owner 2026-09-16 21:54; defaults = the tabletop classes, 20 / 30 deg since owner 2026-10-11): A
                # ``generation.phase2_loose_class_deg`` / B ``generation.phase2_class1_when_deep_m`` (class-1 survivors admitted
                # when every class-0 focus is deeper than that, base-forward: att_diag3 21:30 - all 15 class-0 foci of the 8 deg
                # class at 0.77-0.81 m, the lift / SLERP fail there)
                deep = self.gen("phase2_class1_when_deep_m", None)
                cands = TT.phase2_manipulation_stance(self, init, TT.half_extent(self.hull[self.t_role], self.rest[self.t_role]), target_height=self.height[self.t_role],
                                                      bands=bands, top=self.top,
                                                      angle_classes_deg=TT.phase2_angle_classes(self),
                                                      class1_when_deep_m=None if deep is None else float(deep), split_cone=TT.split_cone_opts(self))
                cands = self._with_partner_room(cands, edge)
                self.edge_frame, self.init_stance, self.manip_candidates = edge, init, cands
                break
            except Reject as e:
                errors.append(f"edge {edge.name}: {e}")
                log.info("%s: %s", self.gp.variant, errors[-1][:300])
        if self.init_stance is None:
            raise Reject("no_stance", "; ".join(errors)[:1500])
        self.template_facts.update({"support": self.top.to_json(), "edge": self.edge_frame.to_json(), "init_stance": self.init_stance.to_meta(),
                                    "n_manipulation_candidates": len(self.manip_candidates), "grasp_key": self.grasp_key, "n_grasps": len(self.grasps),
                                    "attachment_links": [child_link.body_name, parent_link.body_name], "n_symmetry_equivalences": len(self.symmetry),
                                    "band": getattr(self.bands, "path", None), "attach_band": getattr(self.attach_bands, "path", None)})

    def _route_count(self, step: str, n: int = 1) -> None:
        """Kit counter ``attach_route.<route>.<step>`` (per instance in ``summary.json["kit"]``); nothing off the routes."""
        if self.attach_route is not None:
            TT._countn(self, f"attach_route.{self.attach_route}.{step}", n)

    def _select_attach_route(self) -> None:
        """i7-cork (owner decision 4.2): the route a switch asks for (:func:`attach_route_requested`) when the v2 library serves its keys
        (:func:`attach_route_keys`): the male's grasps, key and rest become the route's pick entry's (``lying90``: the lying entry;
        ``handover``: the wide-end-down entry of the right hand's narrow-end pinch, ``self.hold_grasps`` / ``self.hold_key`` the left
        hand's wide-end grasps). Otherwise ``self.attach_route`` stays None and nothing changes (``template_facts.attach_route``)."""
        from ispatialgym.sampling import grasp_library as GL

        self.attach_route, self.hold_grasps, self.hold_key = None, [], None
        want = attach_route_requested()
        if want is None:
            return
        keys = attach_route_keys(self.grasp_key, want)
        facts: Dict[str, Any] = {"requested": want, "keys": list(keys), "base_key": self.grasp_key}
        try:
            libs = [GL.v2_entry(k) for k in keys]
        except ValueError as e:  # an entry under another contact policy: not served
            libs, facts["error"] = [None], str(e)[:300]
        if any(lib is None for lib in libs):
            facts["taken"] = False
            facts["why"] = "no v2 entry serves " + ", ".join(k for k, lib in zip(keys, libs) if lib is None) if "error" not in facts else facts["error"]
            self.template_facts["attach_route"] = facts
            self.counters[f"attach_route.{want}.no_entry"] = self.counters.get(f"attach_route.{want}.no_entry", 0) + 1
            log.info("%s: ATTACH_ROUTE %s not taken: %s", self.gp.variant, want, facts["why"])
            return
        grasps = GL.load_grasps(keys[0])
        rest = C.rest_quat(keys[0])
        self.grasps, self.grasp_key, self.rest[self.t_role] = grasps, keys[0], rest
        if want == "handover":
            self.hold_key = keys[1]
            self.hold_grasps = GL.load_grasps(keys[1])
        self.attach_route = want
        facts.update(taken=True, pick_entry=os.path.basename(os.path.dirname(libs[0].path)), n_pick_grasps=len(grasps),
                     rest_quat_xyzw=[round(float(v), 5) for v in rest])
        if want == "handover":
            facts.update(hold_entry=os.path.basename(os.path.dirname(libs[1].path)), n_hold_grasps=len(self.hold_grasps))
        self.template_facts["attach_route"] = facts
        self.counters[f"attach_route.{want}.template"] = self.counters.get(f"attach_route.{want}.template", 0) + 1
        log.info("%s: ATTACH_ROUTE %s %s", self.gp.variant, want, json.dumps(facts, default=str))

    def _attachment_links(self, male, female):
        return attachment_links(male, female)

    def _attach_band(self, attach_spec: TT.BandSpec, edge: EdgeFrame, view=None):
        """The attach-reach band at ``edge``: the attach map's ``view`` (v2 only, :meth:`_attach_map_view`) when given, else the legacy
        band - loaded from the band caches, computed in-process on a miss (unchanged)."""
        if view is not None:
            return view
        try:
            return TT.RB.ReachBands.load(self.world.scene_model, self.top.name, attach_spec.object, attach_spec.grasp_key, placed=attach_spec.placed, scale=attach_spec.scale)
        except FileNotFoundError:
            return TT.compute_bands(self, attach_spec, self.top, edge)

    def _attach_map_view(self, grasp_key: Optional[str] = None, arm: str = "right"):
        """i7-attachmap 2026-11-07 (v2 only): the attach map of the pair (``attach_maps.sampler_view``: the male's v2 entry + stable rest,
        the partner model / scale, the top's height (anchors), the right arm, the oracle's hover rule, ``pre_attach_back_m``, the contact
        policy, the robot; the map's pair spec checked against the live one) as the attach-reach band view; a miss logs MAP_REQUEST and
        rejects ``missing_map`` - no legacy band under v2 only."""
        from ispatialgym.sampling import attach_maps as AM

        pc = self.choice.models[self.p_role]
        live = {"T_partner_male": C.compose(self.parent_link_in_female, self.root_in_child_link), "insertion_axis": self.insertion_axis_local,
                "partner_bottom_offset_m": float(self.world.bottom_offset(self.hull[self.p_role], C.IDENTITY_QUAT))}
        view, req, why = AM.sampler_view(self, grasp_key=grasp_key or self.grasp_key, partner=(pc.category, pc.model, list(pc.scale) if pc.scale else None),
                                         top=self.top, live=live, pre_back_m=float(self.g("pre_attach_back_m")), arm=arm)
        self.template_facts["attach_map"] = {"hit": view is not None, "why": why, "path": None if view is None else view.meta.get("map_path")}
        if view is None:
            TT.map_request(self, **req)
            raise Reject("missing_map", f"no attach map for {req.get('entry') or self.grasp_key} on {pc.category}/{pc.model} at h {req.get('h')}: {why}")
        return view

    def _symmetry_equivalences(self, male, child_link) -> List[List[float]]:
        """§6.8 mating-axis symmetry test: rotate the male hull about the female axis (the child link's z) in 15° steps; a
        rotation is accepted when >= ``symmetry_match_fraction`` of the 5 mm slabs along the axis have the same angular
        radius profile (12 angular bins, radii within 10 %). Explicit ``generation.orientation_equivalences`` win, then the
        icl symmetry tags of the male (``roles.<role>.orientation_equivalences``, §3.13: ``zinf`` = the 5° rotation group
        about the mating axis, as every other tabletop task writes them; owner 2026-09-19 17:10 — the geometric test found
        nothing for the round lid tnwisp and the group target carried ``[]``, so ``layout_reached`` / ``attached`` pinned
        the lid's yaw); the geometric test is the fallback for an untagged model."""
        explicit = self.gen("orientation_equivalences")
        if explicit:
            return [list(map(float, q)) for q in explicit]
        mc = self.choice.models[self.t_role]
        tagged = self.gp.orientation_equivalences(self.t_role, mc.category, mc.model)
        if tagged:
            self.template_facts["symmetry_source"] = "icl tags"
            return [list(map(float, q)) for q in tagged if any(abs(float(v)) > 1e-9 for v in q[:3])]  # the identity is implied
        self.template_facts["symmetry_source"] = "geometric test"
        rp, rq = male.get_position_orientation()
        cp, cq = child_link.get_position_orientation()
        axis = C.quat_rotate(C.qinv(rq.float()), C.quat_rotate(cq.float(), th.tensor([0.0, 0.0, 1.0])))
        centre = C.quat_rotate(C.qinv(rq.float()), cp.float() - rp.float())
        pts = self.hull[self.t_role] - centre
        h = pts @ axis
        radial = pts - h.unsqueeze(1) * axis
        r = radial.norm(dim=1)
        u = C.unit(th.cross(axis, th.tensor([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else th.tensor([0.0, 1.0, 0.0]), dim=0))
        v = th.cross(axis, u, dim=0)
        ang = th.atan2(radial @ v, radial @ u)
        slab = float(self.g("symmetry_slab_m"))
        frac_ok = float(self.g("symmetry_match_fraction"))
        n_bins = 12
        slabs = []
        for lo in th.arange(float(h.min()), float(h.max()) + 1e-6, slab).tolist():
            sel = (h >= lo) & (h < lo + slab)
            if int(sel.sum()) < 4:
                continue
            prof = th.zeros(n_bins)
            b = ((ang[sel] + math.pi) / (2 * math.pi) * n_bins).long().clamp(0, n_bins - 1)
            for k in range(n_bins):
                m = b == k
                prof[k] = float(r[sel][m].max()) if bool(m.any()) else float("nan")
            slabs.append(prof)
        if not slabs:
            return []
        out = []
        for k in range(1, 24):
            step = k * 2 * math.pi / 24
            shift = int(round(step / (2 * math.pi) * n_bins))
            if abs(step / (2 * math.pi) * n_bins - shift) > 1e-6:
                continue  # 15° steps map onto the 12 bins only for multiples of 30°
            n_match = 0
            for prof in slabs:
                rolled = th.roll(prof, shift)
                both = ~(th.isnan(prof) | th.isnan(rolled))
                if int(both.sum()) == 0 or float(((prof[both] - rolled[both]).abs() / prof[both].clamp(min=1e-3)).max()) <= 0.10:
                    n_match += 1
            if n_match >= frac_ok * len(slabs):
                out.append([float(x) for x in C.quat_from_axis_angle(axis, step)])
        return out

    def _with_partner_room(self, cands: List[Stance], edge: EdgeFrame) -> List[Stance]:
        """Phase 2's survivors at ``edge`` that leave the partner a cell (:meth:`_partner_room`; facts in ``template_facts.partner_room``),
        with separate rejections (owner 2026-10-12, 7h) for a Phase 2 that found nothing and for survivors none of which keeps a cell."""
        if not cands:
            raise Reject("no_manipulation_stance", "Phase 2 found no (stance, focus) pair: no focus within the camera cones with the target "
                                                   "in frame (the phase 2 log lines)")
        kept, room = self._partner_room(cands)
        self.template_facts.setdefault("partner_room", {})[edge.name] = room
        if not kept:
            raise Reject("no_partner_cell", f"Phase 2 left {len(cands)} survivors, none with a partner cell on the free top at the required "
                                            f"gap from the focus ({json.dumps(room)})")
        return kept

    def _partner_room(self, cands: List[Stance]) -> Tuple[List[Stance], Dict[str, Any]]:
        """The Phase-2 survivors that leave the partner a cell an attempt can use (owner 2026-10-12, attachment 7h): from the survivor's
        stance or one of its second base poses (:meth:`_second_base_stances`), among the ``PARTNER_CELLS_TRIED`` attach-band cells
        nearest the focus (the cells :meth:`_attach_and_brace` tries), one where the partner's footprint at one of
        ``PARTNER_ROOM_YAWS`` yaws lies on free cells of the top (``SupportTop.polygon_supported``'s rule with ``TT.SUPPORT_MARGIN``)
        at least ``partner_gap_min_m`` from the male's footprint and ``partner_centre_distance_min_m`` from its centre - the male
        anywhere within ``TT.FOCUS_RADIUS`` of the focus (Phase 3 step 1), its footprint the disc of its rest outline's radius. Before,
        a survivor needed one attach-band cell anywhere at its trunk: on a top whose free cells give the partner no room near any
        focus (a drop-in sink, a cooktop) the template passed and every attempt failed (attachment#04 2026-10-12, countertop_kelzer_0
        / countertop_kelker_0: 31 / 21 attempts over ~24 min each). Returns ``(kept, facts)``; ``facts`` counts the dropped survivors
        by the furthest test they reached (``no_cells`` / ``support`` / ``gap``)."""
        from ispatialgym.sampling import geometry as G

        top = self.top
        gap_min, d_min = float(self.g("partner_gap_min_m")), float(self.g("partner_centre_distance_min_m"))
        slack = float(TT.FOCUS_RADIUS)
        rest_xy = C.transform_points((th.zeros(3), TT._t(self.rest[self.t_role])), TT._t(self.hull[self.t_role]))[:, :2]
        r_m = float(th.linalg.norm(rest_xy, dim=1).max())
        polys = []  # the partner's footprint about its root at each yaw (convex, CCW)
        for k in range(PARTNER_ROOM_YAWS):
            q = C.quat_from_yaw(2.0 * math.pi * k / PARTNER_ROOM_YAWS)
            polys.append(G.convex_hull_2d(C.transform_points((th.zeros(3), q), TT._t(self.hull[self.p_role]))[:, :2]).numpy().astype(np.float64))
        n_v = max(len(p) for p in polys)
        fp = np.stack([np.concatenate([p, np.repeat(p[:1], n_v - len(p), axis=0)]) for p in polys])  # (K, V, 2), padded with the first vertex
        centroid = np.stack([p.mean(axis=0) for p in polys])  # (K, 2): polygon_supported's centroid (the unpadded vertices)
        m = float(TT.SUPPORT_MARGIN)
        offsets = np.array([[0.0, 0.0], [m, 0.0], [-m, 0.0], [0.0, m], [0.0, -m], [m, m], [-m, -m], [m, -m], [-m, m]])  # SupportTop.contains
        free = top.free.numpy()
        x0, y0, res = float(top.xs[0]), float(top.ys[0]), float(top.resolution)

        def on_free(p: np.ndarray) -> np.ndarray:
            i, j = np.rint((p[..., 0] - x0) / res).astype(np.int64), np.rint((p[..., 1] - y0) / res).astype(np.int64)
            inside = (i >= 0) & (j >= 0) & (i < free.shape[0]) & (j < free.shape[1])
            out = np.zeros(i.shape, dtype=bool)
            out[inside] = free[i[inside], j[inside]]
            return out

        cells_of: Dict[Tuple[float, ...], np.ndarray] = {}
        supported_at: Dict[Tuple[float, float], np.ndarray] = {}

        def stance_cells(st: Stance) -> np.ndarray:
            key = (round(float(st.base_pose2d[0]), 4), round(float(st.base_pose2d[1]), 4), round(float(st.base_pose2d[2]), 5), float(st.translate), float(st.tilt))
            if key not in cells_of:
                cells_of[key] = TT.band_cells_world(self.attach_bands, st).numpy().astype(np.float64).reshape(-1, 2)
            return cells_of[key]

        def supported(cells: np.ndarray) -> np.ndarray:
            keys = [(round(float(x), 4), round(float(y), 4)) for x, y in cells]
            todo = [i for i, k in enumerate(keys) if k not in supported_at]
            if todo:
                c = cells[todo]
                ok = on_free(c[:, None, None, None, :] + fp[None, :, :, None, :] + offsets[None, None, None, :, :]).all(axis=(2, 3))
                ok &= on_free(c[:, None, :] + centroid[None, :, :])
                for i, row in zip(todo, ok):
                    supported_at[keys[i]] = row
            return np.stack([supported_at[k] for k in keys])  # (n, K)

        def focus_gap(cells: np.ndarray, f: np.ndarray) -> np.ndarray:
            """Distance from the focus to each partner footprint (0 inside), (n, K)."""
            s0 = cells[:, None, None, :] + fp[None, :, :, :]
            e = np.roll(s0, -1, axis=2) - s0
            d = f[None, None, None, :] - s0
            t = np.clip((d * e).sum(-1) / np.maximum((e * e).sum(-1), 1e-12), 0.0, 1.0)
            dist = np.linalg.norm(d - t[..., None] * e, axis=-1).min(axis=2)
            inside = ((e[..., 0] * d[..., 1] - e[..., 1] * d[..., 0]) >= 0.0).all(axis=2)
            return np.where(inside, 0.0, dist)

        rank = {"no_cells": 0, "support": 1, "gap": 2}
        kept: List[Stance] = []
        dropped = {k: 0 for k in rank}
        for c in cands:
            f = c.focus[:2].numpy().astype(np.float64)
            reached = "no_cells"
            for st in [c] + [s for _, _, s in self._second_base_stances(c)]:
                cells = stance_cells(st)
                if not len(cells):
                    continue
                dist = np.linalg.norm(cells - f, axis=1)
                near = np.argsort(dist, kind="stable")[:PARTNER_CELLS_TRIED]
                sup = supported(cells[near])
                if not sup.any():
                    reached = max(reached, "support", key=rank.get)
                    continue
                ok = sup & (focus_gap(cells[near], f) - r_m + slack >= gap_min) & (dist[near] + slack >= d_min)[:, None]
                if ok.any():
                    reached = None
                    break
                reached = "gap"
            if reached is None:
                kept.append(c)
            else:
                dropped[reached] += 1
        facts = {"survivors": len(cands), "kept": len(kept), "dropped": dropped, "male_radius_m": round(r_m, 4), "yaws": PARTNER_ROOM_YAWS,
                 "cells_tried": PARTNER_CELLS_TRIED}
        log.info("%s: partner room: %d of %d Phase-2 survivors leave the partner a cell on the free top at the required gap (dropped %s)",
                 self.gp.variant, len(kept), len(cands), dropped)
        return kept, facts

    # ------------------------------------------------------------------ geometry

    def _attached_pose(self, female_pose: TT.Pose) -> TT.Pose:
        """Male root pose when its attachment link coincides with the female's (female at ``female_pose``)."""
        parent_link_world = C.compose(female_pose, self.parent_link_in_female)
        return C.compose(parent_link_world, self.root_in_child_link)

    def _insertion_axis_world(self, female_pose: TT.Pose) -> th.Tensor:
        return C.unit(C.quat_rotate(TT._t(female_pose[1]), self.insertion_axis_local))

    # ------------------------------------------------------------------ instance

    def _candidate_order(self, rng: np.random.Generator, rejects: Dict[int, int]) -> List[int]:
        """Manipulation candidates in the order an attempt tries them: the focus at an arm-friendly depth first (att_diag2 21:04:
        the male at 0.81 m base-forward grasps but its 12 cm lift fails the IK at every candidate; the block probes pass picks +
        pre-poses up to 0.65-0.70 m), a trunk close to a verified one next, ties by the draw. Owner 2026-09-25 (cap 0925-220515:
        45 phase-2 survivors, the same 6 best-scored stances tried in all 8 attempts - candidates 6 / 21 / 36 rejected 20 times over
        for the attach pose outside the manipulation image, no transition, the lift without IK - and 39 stances never looked at):
        every rejection a candidate collected in an earlier attempt of this template adds ``CANDIDATE_REJECT_PENALTY``, so the next
        attempt's window of six moves on to stances the score alone never admits. The rejections are keyed by the candidate itself
        (:meth:`_candidate_key`), not its index: the split cone hands each attempt one start position's stances (owner 2026-10-12, 7h:
        the memory is kept across positions, so a position drawn again continues where it left off)."""
        order = [int(k) for k in rng.permutation(len(self.manip_candidates))]

        def score(k: int) -> float:
            c = self.manip_candidates[k]
            depth = float(c.world_to_base(c.focus[:2])[0]) if c.focus is not None else FOCUS_DEPTH_PREFERRED
            trunk_pen = min(abs(float(c.translate) - t0) + abs(float(c.tilt) - i0) for t0, i0 in GOOD_TRUNKS)
            return max(0.0, depth - FOCUS_DEPTH_PREFERRED) + 0.3 * trunk_pen + CANDIDATE_REJECT_PENALTY * rejects.get(Attachment._candidate_key(c), 0)

        order.sort(key=score)
        return order

    @staticmethod
    def _candidate_key(c: Stance) -> Tuple[Any, ...]:
        """A manipulation candidate in the reject memory: its focus, stand-off, lateral and trunk (the same after the split cone's
        per-attempt draw replaced ``manip_candidates``)."""
        focus = (round(float(c.focus[0]), 3), round(float(c.focus[1]), 3)) if c.focus is not None else None
        return (focus, round(float(c.stand_off or 0.0), 3), round(float(c.lateral), 3), round(float(c.translate), 3), round(float(c.tilt), 4))

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        init = self.init_stance
        errors = []
        self._deadline = time.time() + float(self.gen("attempt_budget_s", ATTEMPT_BUDGET_S))
        # candidates with the focus at an arm-friendly depth first (att_diag2 21:04: the male at 0.81 m base-forward grasps
        # but its 12 cm lift fails the IK at every candidate; the block probes pass picks + pre-poses up to 0.65-0.70 m),
        # ties by the draw
        rejects = getattr(self, "_candidate_rejects", None)
        if rejects is None:
            rejects = self._candidate_rejects = {}
        order = self._candidate_order(rng, rejects)
        for k in order[:6]:
            if time.time() > self._deadline:
                errors.append(f"attempt budget of {self.gen('attempt_budget_s', ATTEMPT_BUDGET_S):.0f} s spent")
                break
            cand = self.manip_candidates[int(k)]
            manip = cand.with_trunk(cand.translate, cand.tilt, "manipulation")
            manip.focus, manip.arms = cand.focus, ["right"]
            self.sidecar.reset_instance()
            self.sidecar.add_stance(init)
            for o in self.world.spawned.values():
                self.world.park(o, list(self.world.spawned).index(o.name))
            self.kit.sync_world()
            try:
                if self.attach_route == "handover":
                    return self._sample_with_handover(rng, init, manip, int(k))
                return self._sample_with(rng, init, manip, int(k))
            except Reject as e:
                key = self._candidate_key(cand)
                rejects[key] = rejects.get(key, 0) + 1
                errors.append(f"candidate {int(k)}: {e}")
                log.info("%s: %s", self.gp.variant, errors[-1][:300])
        raise Reject("target_placement", "; ".join(errors[-3:])[:1500])

    def _check_deadline(self):
        """Per-attempt time budget (owner 2026-09-16 10:47 item 3: an attempt stays under ~5 min): the current candidate
        stops with ``Reject("time_budget")`` once it is spent."""
        if time.time() > getattr(self, "_deadline", math.inf):
            raise Reject("time_budget", f"attempt budget of {self.gen('attempt_budget_s', ATTEMPT_BUDGET_S):.0f} s spent")

    def _sample_with(self, rng, init: Stance, manip: Stance, k: int) -> Draft:
        d = Draft()
        tname, pname = self.tname, self.pname
        male, support = self.obj(self.t_role), self.obj("support")
        self.record_stance_travel(init, manip)
        cam_init = self.camera_pose_at(init)
        # step 1: the male at the focus from F (a thin male: the tilted rim pinches first, every library grasp as the fallback)
        m_pose, grasp = None, None
        for pool, label in self._grasp_pools():
            try:
                m_pose, grasp = TT.phase3_target(self, manip, rng, pool, self.hull[self.t_role], self.rest[self.t_role], [support], object_name=tname,
                                                 scale_key=self.choice.models[self.t_role].scale, bands=self.bands)
                break
            except Reject as e:
                if label == "all":
                    raise
                log.info("%s: no %s grasp confirmed at the focus (%s); every library grasp is screened next", self.gp.variant, label, e)
        grasp.place_on = pname
        self._route_count("pick")
        male.set_position_orientation(*m_pose)
        male.keep_still()
        self.kit.sync_world()
        self._apply_grasp_height_extra(grasp, m_pose, manip)  # generation.grasp_height_extra_m (owner 2026-09-19 16:49; 0 = no-op)
        self._record_pinch_width(grasp, m_pose)  # the pre-shape width from the male's own geometry at the pinch (pwgbsb 0919-210926)
        # step 2 (+ §6.8 owner 2026-09-16 17:10): the partner on an attach-reach cell with the attach pose and the left brace
        # reachable — from the grasp stance first; else from a recorded SECOND BASE POSE (lateral trunk move + torso adjust,
        # both arms verified there, the male carried: paths[] / transitions[] manipulation -> attach with the male held);
        # only when the joint placement has no cell anywhere is the brace optional (summary.brace = none).
        # ``generation.brace_search`` false (owner 2026-09-19, lid / teapot: "pick up the lid with the right arm -> move to the
        # place stance -> place the lid on the teapot opening"): no brace at all — the one-arm attach from the grasp stance,
        # else from a recorded place stance (the same second-base-pose mechanism, the left arm idle); the two-arm search that
        # burnt the 360 s attempt budget on every candidate (0916-205838 .. 0917-010834) is skipped
        brace_search = bool(self.gen("brace_search", True))
        att, work_st, brace_state = None, manip, "present"
        if brace_search:
            att = self._attach_and_brace(rng, manip, manip, m_pose, grasp, cam_init, require_joint=True)
            if att is None:
                att, work_st = self._second_base_pose(rng, manip, m_pose, grasp, cam_init, tname)
        if att is None:
            self._check_deadline()
            if brace_search:
                log.info("%s: BRACE_FALLBACK no joint placement from the grasp stance or a second base pose: %s", self.gp.variant, json.dumps(self._att_fails, default=str))
            att = self._attach_and_brace(rng, manip, manip, m_pose, grasp, cam_init, require_joint=False, brace=brace_search)
            work_st, brace_state = manip, "none"
            if att is None and not brace_search:
                att, work_st = self._second_base_pose(rng, manip, m_pose, grasp, cam_init, tname, require_joint=False, brace=False)
            if att is None:
                raise Reject("partner_placement", f"no partner cell with a reachable attach pose from the grasp stance or a second base pose ({self._att_fails})")
            if att.get("brace") is not None:
                brace_state = "present"
        self._route_count("attach_found")
        f_pose, attach, pre_attach, wps, carry, lift = att["f_pose"], att["attach"], att["pre_attach"], att["wps"], att["carry"], att["lift"]
        goal = self._attached_pose(f_pose)
        brace_pre, brace, brace_kind = att["brace"] if att.get("brace") is not None else (None, None, None)
        brace_dir = -att["b_dir"]
        if work_st is not manip:
            # the recorded move between the grasp and the attach stance, the male in hand (paths[] carried + transitions[] holding)
            self.sidecar.add_stance(work_st)
            male.set_position_orientation(*m_pose)
            male.keep_still()
            self.kit.sync_world()
            path = self.record_path(manip, work_st, carried=tname)
            tr = self.record_transition(manip, work_st, carried=tname) if (abs(work_st.translate - manip.translate) > 1e-6 or abs(work_st.tilt - manip.tilt) > 1e-6) else None
            self.sidecar.transitions = [HoldingTransition.wrap(t, tname) if t is tr else t for t in self.sidecar.transitions]
            work_st.carried = tname
            work_st.extras.update({"second_base_pose": True, "lateral_shift": float(work_st.lateral - manip.lateral), "path_length_m": getattr(path, "length_m", None)})
        cam_manip = self.camera_pose_at(manip)
        cam_work = cam_manip if work_st is manip else self.camera_pose_at(work_st)
        # the CONTACT poses keep the box check (slack for execution error); the in-between poses - the SLERP carry, the pre-attach 10 cm
        # closer to the lens - are checked on QA's fingertip point by the program check (program_check.fingertips_in_frame, owner
        # 2026-09-25: cap 0925-191101 gripper_out_of_fov at the pre-attach)
        for nm, p, cam in [("grasp", grasp.grasp, cam_manip), ("attach", attach, cam_work)] + ([("brace", brace, cam_work)] if brace is not None else []):
            if not self.ikc.eef_in_frame(p, cam):
                raise Reject("gripper_not_in_frame", f"{nm} eef pose outside the {'manipulation' if cam is cam_manip else 'attach'} image (5 %)")
        objects = {tname: m_pose, pname: f_pose}
        TT.whole_body_check_at(self, [init, manip] + ([work_st] if work_st is not manip else []), objects, require=True, reason="stance_collision")
        # record
        grasp.carry_height = float(carry)
        grasp.waypoints = wps
        grasp.pre_place, grasp.place = pre_attach, attach
        grasp.extras.update({"attach_stance": work_st.name, "lift": [float(v) for v in lift[0]] if lift is not None else None,
                             "slerp_policy": att.get("slerp_policy")})
        if att.get("carry_over") is not None:
            grasp.extras["carry_over"] = att["carry_over"]
        if self.attach_route is not None:
            # i7-cork: the route the oracle program reads (programs.program_attachment); lying90 = the program as it is, the recorded SLERP
            # carry being the 90 deg in-hand turn
            grasp.extras["attach_route"] = self.attach_route
            grasp.extras["attach_route_turn_deg"] = round(math.degrees(C.quat_angle(TT._t(lift[1]) if lift is not None else TT._t(grasp.grasp[1]), TT._t(pre_attach[1]))), 1)
        self._record_approach_standoff(grasp, f_pose, manip)  # over-the-top pre-grasp approach when the partner stands near the male
        # owner item 3 (release height): the attach pose is fixed by the mating links (no release above a surface), but the
        # surface under the male's footprint centre at the goal (the female's mating face) is recorded for the audit
        grasp.extras["release_surface_z"] = None
        if hasattr(self.world, "surface_z_under"):
            m_hull_goal = C.transform_points(goal, self.hull[self.t_role])
            hit = self.world.surface_z_under(m_hull_goal[:, :2].mean(dim=0), float(m_hull_goal[:, 2].max()) + 0.05, ignore=[male])
            if hit is not None:
                grasp.extras["release_surface_z"] = round(float(hit["z"]), 4)
                grasp.extras["release_surface_body"] = hit.get("body")
        manip.arms = ["right"] if work_st is not manip or brace is None else ["right", "left"]
        self.sidecar.add_stance(manip)
        self.sidecar.add_grasp(grasp)
        if brace is not None:
            b_extras = {"pose_class": brace_kind, "stop_clearance_m": 0.0}
            b_orient = att.get("brace_orient") if att is not None else None
            if b_orient and len([o for o in b_orient.get("options", []) if o.get("source") == "ik"]) > 1:
                b_extras["orient"] = b_orient
            self.sidecar.add_contact(TT.BraceRecord("brace", "left", brace_pre, brace, direction=brace_dir, brace_kind="partner_face", object=pname,
                                                    stance=work_st.name, extras=b_extras))
            work_st.contacts = ["brace"]
            work_st.arms = ["right", "left"]
        work_st.extras.update({"brace": brace_state, "brace_class": brace_kind, "left_band": self.left_bands is not None})
        d.object_poses.update(objects)
        d.goal_poses[tname] = goal
        d.robot_q = self.stance_q(init)
        d.base_pose2d = tuple(init.base_pose2d)
        d.groups[self._group_name([tname, pname])] = self._group_target(f_pose, goal, tname, pname)
        child_link, parent_link = self.pair
        d.extras["attachment_targets"] = [{"child": tname, "parent": pname, "child_link": child_link.body_name, "parent_link": parent_link.body_name, "joint_type": "fixed"}]
        mc, pc = self.choice.models[self.t_role], self.choice.models[self.p_role]
        self.sidecar.describe(tname, mc.category.replace("_", " "))
        self.sidecar.describe(pname, pc.category.replace("_", " "))
        d.summary.update({"grasp": {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw}, "manipulation_candidate": k, "partner_fails": self._att_fails,
                          "brace": brace_state, "brace_class": brace_kind, "attach_stance": work_st.name, "second_base_pose": work_st is not manip,
                          "joint_placement": {"left_band": self.left_bands is not None}})
        self.brace_missing = brace_state == "none"
        if self.attach_route is not None:
            d.summary["attach_route"] = {"route": self.attach_route, "grasp_key": self.grasp_key, "turn_deg": grasp.extras.get("attach_route_turn_deg")}
            self._route_count("drafted")
        return d

    # ------------------------------------------------------------------ i7-cork route (i): the hand-over (owner decision 2026-11-08 4.2 (i))

    def _hand_points(self, arm: str, q_f: float) -> np.ndarray:
        """Surface points (eef frame) of every hand hull of ``arm`` (fingers, palm, wrist camera, wrist) with both fingers at ``q_f``."""
        cache = self.__dict__.setdefault("_hand_pts_cache", {})
        key = (arm, round(float(q_f), 4))
        if key not in cache:
            from ispatialgym.grasping import hand_geometry as HG

            P, _ = HG.load(arm).surface(spacing=0.0015, finger_q=(float(q_f), float(q_f)))
            cache[key] = np.asarray(P, dtype=float)
        return cache[key]

    def _hands_apart_m(self, eef_r: TT.Pose, q_r: float, eef_l: TT.Pose, q_l: float) -> float:
        """The smallest distance between the two hands' exact hulls (surface samples 1.5 mm apart) with the right hand at ``eef_r``
        (fingers at ``q_r``) and the left at ``eef_l`` (``q_l``)."""
        from scipy.spatial import cKDTree

        def world(arm, eef, q):
            R = C.quat_to_mat(TT._t(eef[1])).numpy().astype(float)
            return self._hand_points(arm, q) @ R.T + TT._t(eef[0]).numpy().astype(float)

        d, _ = cKDTree(world("left", eef_l, q_l)).query(world("right", eef_r, q_r), k=1)
        return float(np.min(d))

    @staticmethod
    def _finger_q_of(g, default: float = 0.008) -> float:
        """A library pinch's finger joint value at its hold (half its width when it records none)."""
        fq = getattr(g, "finger_q", None)
        if fq:
            return float(max(fq))
        w = (getattr(g, "meta", None) or {}).get("width") or (getattr(g, "extras", None) or {}).get("pinch_width_m")
        return 0.5 * float(w) if w else default

    def _handover_object_poses(self, st: Stance, rng: np.random.Generator) -> List[TT.Pose]:
        """The male's poses at the hand-over, in front of the robot at ``st`` (base-frame ``HANDOVER_FORWARD_M`` x ``HANDOVER_LEFT_M``, its centre
        ``HANDOVER_RISE_M`` above the support top), its +z axis (male end -> wide end) along the robot's left (the wide end toward the left
        hand), turned ``HANDOVER_ROLLS`` ways about that axis; in the draw's order."""
        f = th.tensor([float(st.forward[0]), float(st.forward[1]), 0.0])
        lft = th.tensor([float(st.left[0]), float(st.left[1]), 0.0])
        up = th.tensor([0.0, 0.0, 1.0])
        frames = []
        for t in HANDOVER_TILTS_DEG:
            z = math.cos(math.radians(t)) * lft + math.sin(math.radians(t)) * up
            frames.append(th.stack([f, th.linalg.cross(z, f), z], dim=1))  # columns: object x -> forward, z -> left (tilted), y = z x x
        out = []
        for fx in HANDOVER_FORWARD_M:
            for ly in HANDOVER_LEFT_M:
                for dz in HANDOVER_RISE_M:
                    xy = st.base_to_world([fx, ly])
                    p = th.tensor([float(xy[0]), float(xy[1]), float(self.top.z) + dz])
                    for R0 in frames:
                        for k in range(HANDOVER_ROLLS):
                            Rz = C.quat_to_mat(C.quat_from_axis_angle(th.tensor([0.0, 0.0, 1.0]), 2.0 * math.pi * k / HANDOVER_ROLLS))
                            out.append((p, C.mat_to_quat(R0 @ Rz)))
        return [out[int(i)] for i in rng.permutation(len(out))]

    def _handover_search(self, rng, manip: Stance, m_pose: TT.Pose, grasp, cam_manip) -> Dict[str, Any]:
        """The hand-over from the right hand's lifted pinch (``grasp``, the male at ``m_pose`` on the support) at the grasp stance: per
        candidate male pose M (:meth:`_handover_object_poses`) and left hold grasp (``self.hold_grasps``) - both eef poses in the stance
        camera's image (the box rule, 5 %), the two hands' exact hulls >= ``HANDOVER_HANDS_MIN_M`` apart, the right hand's carry from the lift
        to M tracked with the male attached (the oracle's dense tracking, as the SLERP carry), the left hand's [pre-grasp, grasp] exact IK
        with the right arm locked at M and the male in the world (ignored at the grasp), the combined whole-body check, and the right hand's
        release retreat (``RETREAT_DIST`` back along its approach) exact with the left arm locked at its grasp. The first candidate that
        passes; ``Reject("handover_unreachable")`` with the counts otherwise."""
        from ispatialgym.oracle.actions import RETREAT_DIST

        male, support = self.obj(self.t_role), self.obj("support")
        scale = self.choice.models[self.t_role].scale
        q_fr = self._finger_q_of(grasp)
        fails: Dict[str, int] = {"frame": 0, "right_ik": 0, "hands": 0, "left_ik": 0, "combined": 0, "carry": 0, "retreat": 0}
        why: Dict[str, int] = {}

        def note(kind, reason):
            k = f"{kind}:{str(reason).split(':')[0].split(' (')[0][:40]}"
            why[k] = why.get(k, 0) + 1

        n = int(self.g("reorientation_waypoints")) + 1
        tried = 0
        for M in self._handover_object_poses(manip, rng):
            if tried >= HANDOVER_TRIES:
                break
            self._check_deadline()
            H_r = IK.grasp_pose(M, grasp.eef_in_object)
            if not self.ikc.eef_in_frame(H_r, cam_manip):
                fails["frame"] += 1
                continue
            tried += 1
            # the right hand at the hand-over, the male attached (a cheap exact IK before the tracked carry)
            with TT.posed_in_hand(self, "right", male, grasp.eef_in_object):
                with attached(self.kit, "right", male):
                    r_h = self.ikc.exact([H_r], manip, "right", ignore_for_last=[support], cache=False, scale_key=scale)
            if not r_h.ok:
                fails["right_ik"] += 1
                note("right_ik", r_h.reason)
                continue
            holds = []
            for g_l in self.hold_grasps:
                H_l = IK.grasp_pose(M, (TT._t(g_l.pose[0]), TT._t(g_l.pose[1])))
                if not self.ikc.eef_in_frame(H_l, cam_manip):
                    fails["frame"] += 1
                    continue
                gap = self._hands_apart_m(H_r, q_fr, H_l, self._finger_q_of(g_l))
                if gap < HANDOVER_HANDS_MIN_M:
                    fails["hands"] += 1
                    continue
                holds.append((g_l, H_l, gap))
            if not holds:
                continue
            # the male where the right hand holds it at the hand-over, in the world for the left hand's chain
            male.set_position_orientation(TT._t(M[0]), TT._t(M[1]))
            male.keep_still()
            self.kit.sync_world()
            try:
                found_l = None
                for g_l, H_l, gap in holds:
                    pre_l = IK.pre_pose(H_l)
                    # the male is touched by both hands here: it leaves the world for both hands' chains (the locked right hand's fingers
                    # stand on it); the two hands' distance is the exact-hull check above, the left approach the entry's validated one
                    with self.ikc.ignoring([male]):
                        r_l = self.ikc.exact([pre_l, H_l], manip, "left", q_seed=r_h.q, cache=False, scale_key=scale)
                    if not r_l.ok:
                        fails["left_ik"] += 1
                        note("left_ik", r_l.reason)
                        continue
                    with self.ikc.ignoring([male]):
                        free = bool(self.kit.collision_free(r_l.q, margin=0.005)[0])
                    if not free:
                        fails["combined"] += 1
                        continue
                    found_l = (g_l, H_l, gap, pre_l, r_l)
                    break
            finally:
                male.set_position_orientation(TT._t(m_pose[0]), TT._t(m_pose[1]))
                male.keep_still()
                self.kit.sync_world()
            if found_l is None:
                continue
            g_l, H_l, gap, pre_l, r_l = found_l
            # the right hand's carry lift -> hand-over, tracked (the lift ladder)
            r_c, lift, wps, carry = None, None, None, None
            for carry in CARRY_HEIGHTS:
                lift = (TT._t(grasp.grasp[0]) + th.tensor([0.0, 0.0, carry]), TT._t(grasp.grasp[1]))
                wps = [(TT._t(p[0]), TT._t(p[1])) for p in C.line_waypoints(lift, H_r, n)]
                with TT.posed_in_hand(self, "right", male, grasp.eef_in_object):
                    with attached(self.kit, "right", male):
                        r_c = self.ikc.waypoints_ok([lift] + wps, manip, "right", ignore_all=[support], track_dense=bool(self.gen("slerp_tracked", True)))
                self._route_count("handover_carry_checked")
                if r_c.ok and r_c.q is not None:
                    break
                note("carry", r_c.reason)
            if r_c is None or not r_c.ok or r_c.q is None:
                fails["carry"] += 1
                continue
            # the left hand again from the tracked right configuration, then the right hand's release retreat with the left locked
            male.set_position_orientation(TT._t(M[0]), TT._t(M[1]))
            male.keep_still()
            self.kit.sync_world()
            try:
                with self.ikc.ignoring([male]):
                    r_l = self.ikc.exact([pre_l, H_l], manip, "left", q_seed=r_c.q, cache=False, scale_key=scale)
                    free = bool(r_l.ok) and bool(self.kit.collision_free(r_l.q, margin=0.005)[0])
                if not free:
                    fails["combined"] += 1
                    continue
                retreat_r = IK.pre_pose(H_r, RETREAT_DIST)
                with self.ikc.ignoring([male]):
                    r_x = self.ikc.exact([retreat_r], manip, "right", q_seed=r_l.q, cache=False)
            finally:
                male.set_position_orientation(TT._t(m_pose[0]), TT._t(m_pose[1]))
                male.keep_still()
                self.kit.sync_world()
            if not r_x.ok:
                fails["retreat"] += 1
                note("retreat", r_x.reason)
                continue
            self._route_count("handover_found")
            return {"M": M, "H_r": H_r, "lift": lift, "carry": carry, "wps_r": wps, "q_r": r_c.q, "g_l": g_l, "H_l": H_l, "pre_l": pre_l,
                    "q_l": r_l.q, "retreat_r": retreat_r, "q_x": r_x.q, "hands_apart_m": gap, "tried": tried, "fails": fails}
        for k_, v_ in fails.items():
            if v_:
                self._route_count(f"handover_fail.{k_}", v_)
        self._route_count("handover_unreachable")
        raise Reject("handover_unreachable", f"no hand-over pose passes ({tried} evaluated): {fails}; reasons {dict(sorted(why.items(), key=lambda kv: -kv[1])[:6])}")

    @staticmethod
    def _carried_pose(frm: Stance, to: Stance, pose: TT.Pose) -> TT.Pose:
        """Where an eef pose at ``frm`` is after the base drove to ``to`` with the arm frozen (same trunk; the same pose in the base frame)."""
        b = frm.world_to_base(TT._t(pose[0])[:2])
        xy = to.base_to_world(b)
        dyaw = to.yaw - frm.yaw
        q = C.qmul(C.quat_from_axis_angle(th.tensor([0.0, 0.0, 1.0]), dyaw), TT._t(pose[1])) if abs(dyaw) > 1e-9 else TT._t(pose[1])
        return (th.tensor([float(xy[0]), float(xy[1]), float(TT._t(pose[0])[2])]), q)

    def _handover_place_stances(self, manip: Stance) -> Iterator[Stance]:
        """The place stances the left hand attaches from, in order: the grasp stance, then the grasp stance driven sideways along the edge
        (``SECOND_BASE_LATERALS``) or back / forward (``HANDOVER_STANDOFF_SHIFTS_M``), the heading and the trunk kept (axis legs only)."""
        yield manip
        for dlat in SECOND_BASE_LATERALS:
            yield manip.shifted(lateral=manip.lateral + dlat, name="attach")
        for ds in HANDOVER_STANDOFF_SHIFTS_M:
            if manip.stand_off is not None:
                yield manip.shifted(stand_off=float(manip.stand_off) + ds, name="attach")

    def _handover_attach_from(self, rng, manip: Stance, st: Stance, m_pose: TT.Pose, ho: Dict[str, Any], cam_init) -> Optional[Dict[str, Any]]:
        """The left hand's attach from stance ``st`` (the grasp stance or a place stance): the partner on a cell of the left-arm attach map of
        the hold entry (``self.attach_bands``: distance / support / gap / init-frame screens as :meth:`_attach_and_brace`), the [pre-attach,
        attach] exact IK of the left hand with the male attached, the fingers clear of the partner, and the left hand's carry from the hold
        pose (carried with the base from the grasp stance) to the pre-attach tracked from the hold configuration. Returns the attach facts or
        None (reasons in ``self._att_fails``)."""
        male, female, support = self.obj(self.t_role), self.obj(self.p_role), self.obj("support")
        g_l = ho["g_l"]
        eio_l = (TT._t(g_l.pose[0]), TT._t(g_l.pose[1]))
        fails = {"distance": 0, "support": 0, "gap": 0, "frame": 0, "screen": 0, "ik": 0, "fingers": 0, "carry": 0, "attach_frame": 0}
        self._att_fails[f"handover:{st.name}@lat{st.lateral:+.2f}/s{float(st.stand_off or 0.0):.2f}"] = fails
        cells = TT.band_cells_world(self.attach_bands, st)
        if len(cells) == 0:
            fails["no_cells"] = 1
            return None
        m_fp = self.world.footprint(self.hull[self.t_role], m_pose)
        focus_xy = st.focus[:2] if st.focus is not None else m_pose[0][:2]
        perm = [int(i) for i in rng.permutation(len(cells))]
        perm.sort(key=lambda i: float(th.linalg.norm(cells[i] - focus_xy)))
        geo = []
        for i in perm[:PARTNER_CELLS_TRIED]:
            xy = cells[int(i)]
            if float(th.linalg.norm(xy - m_pose[0][:2])) < float(self.g("partner_centre_distance_min_m")):
                fails["distance"] += 1
                continue
            pose = TT.object_pose_on(self, self.hull[self.p_role], C.IDENTITY_QUAT, xy, float(rng.uniform(-math.pi, math.pi)), self.top.z)
            fp = self.world.footprint(self.hull[self.p_role], pose)
            if not self.top.polygon_supported(fp, margin=TT.SUPPORT_MARGIN):
                fails["support"] += 1
                continue
            if TT._polygons_overlap(m_fp, fp, float(self.g("partner_gap_min_m"))):
                fails["gap"] += 1
                continue
            if not TT.in_frame(self, cam_init, TT.hull_corners_world(self.hull[self.p_role], pose), TT.IN_FRAME_MARGIN):
                fails["frame"] += 1
                continue
            goal = self._attached_pose(pose)
            eef_attach = IK.grasp_pose(goal, eio_l)
            eef_pre = (eef_attach[0] + self._insertion_axis_world(pose) * float(self.g("pre_attach_back_m")), eef_attach[1])
            geo.append((pose, eef_pre, eef_attach))
        if not geo:
            return None
        with TT.posed_in_hand(self, "left", male, eio_l):
            with attached(self.kit, "left", male):
                ok = self.ikc.screen_pairs([[g_[1], g_[2]] for g_ in geo], st, "left", ignore_for_last=[support])
        screened = [g_ for g_, o in zip(geo, ok.tolist()) if o]
        fails["screen"] = len(geo) - len(screened)
        cam_st = self.camera_pose_at(st)
        start = self._carried_pose(manip, st, ho["H_l"])
        idx_l = self.kit.robot.arm_control_idx["left"]
        q_first = self.ikc.q_locked(st, left_arm=TT._t(ho["q_l"])[idx_l])
        n = int(self.g("reorientation_waypoints")) + 1
        for pose, eef_pre, eef_attach in screened[:PARTNER_EXACT_CAP]:
            self._check_deadline()
            if not self.ikc.eef_in_frame(eef_attach, cam_st):
                fails["attach_frame"] += 1
                continue
            female.set_position_orientation(*pose)
            female.keep_still()
            self.kit.sync_world()
            with TT.posed_in_hand(self, "left", male, eio_l):
                with attached(self.kit, "left", male):
                    res = self.ikc.exact([eef_pre, eef_attach], st, "left", ignore_for_last=[support, female], scale_key=self.choice.models[self.t_role].scale)
                if not res.ok:
                    fails["ik"] += 1
                    continue
                if not self.ikc.collision_free_at(res.q, eef_attach, "left", ignore=[support, male]):
                    fails["fingers"] += 1
                    continue
                wps = [(TT._t(p[0]), TT._t(p[1])) for p in C.line_waypoints(start, eef_pre, n)]
                with attached(self.kit, "left", male):
                    r_w = self.ikc.waypoints_ok([start] + wps, st, "left", ignore_all=[support], track_dense=bool(self.gen("slerp_tracked", True)), q_first=q_first)
                self._route_count("carry_checked")
                if not r_w.ok:
                    fails["carry"] += 1
                    self._route_count("carry_failed")
                    continue
            return {"f_pose": pose, "attach": eef_attach, "pre_attach": eef_pre, "wps": wps, "start": start, "q_attach": res.q, "slerp_policy": r_w.reason}
        return None

    def _sample_with_handover(self, rng, init: Stance, manip: Stance, k: int) -> Draft:
        """i7-cork route (i) (owner decision 2026-11-08 4.2 (i)): the male rests wide end down; the right hand pinches its narrow (mating) end
        (the pick entry, ``self.grasps``), lifts it and carries it to a hand-over pose in front of the robot with the wide end toward the left
        hand (:meth:`_handover_search`); the left hand pinches the wide end (the hold entry, fingers off the mating side), the right hand lets
        go and backs off; the base drives to the place stance along the edge or straight back / forward, the heading and the trunk kept
        (:meth:`_handover_place_stances`; the recorded path is an axis leg), the left hand carries the male to the pre-attach and attaches
        (:meth:`_handover_attach_from`). Records: ``grasps[0]`` = the right hand's pick (``waypoints`` = the carry to the hand-over,
        ``handover``), ``grasps[1]`` = the left hand's hold (grasp at the hand-over, ``waypoints`` / pre_place / place = the attach),
        ``grasps[*].attach_route = "handover"``; the program is ``programs.program_attachment_handover``."""
        d = Draft()
        tname, pname = self.tname, self.pname
        male, support = self.obj(self.t_role), self.obj("support")
        self.record_stance_travel(init, manip)
        cam_init, cam_manip = self.camera_pose_at(init), self.camera_pose_at(manip)
        m_pose, grasp = TT.phase3_target(self, manip, rng, list(self.grasps), self.hull[self.t_role], self.rest[self.t_role], [support], object_name=tname,
                                         scale_key=self.choice.models[self.t_role].scale, bands=self.bands)
        grasp.place_on = pname
        self._route_count("pick")
        male.set_position_orientation(*m_pose)
        male.keep_still()
        self.kit.sync_world()
        self._record_pinch_width(grasp, m_pose)
        ho = self._handover_search(rng, manip, m_pose, grasp, cam_manip)
        att, work_st, tried = None, manip, 0
        for st in self._handover_place_stances(manip):
            if tried >= SECOND_BASE_MAX_TRIES + 1:
                break
            self._check_deadline()
            st.focus, st.arms = manip.focus, ["left"]
            if st is not manip:
                if not self.kit.trav_free(st.base_xy):
                    continue
                male.set_position_orientation(*m_pose)
                male.keep_still()
                self.kit.sync_world()
                if not self.ikc.stance_collision_free(st):
                    continue
            tried += 1
            att = self._handover_attach_from(rng, manip, st, m_pose, ho, cam_init)
            if att is not None:
                work_st = st
                break
        if att is None:
            self._route_count("place_unreachable")
            raise Reject("partner_placement", f"hand-over: no partner cell with a reachable left-hand attach from {tried} place stance(s) ({self._att_fails})")
        self._route_count("attach_found")
        f_pose, attach, pre_attach = att["f_pose"], att["attach"], att["pre_attach"]
        goal = self._attached_pose(f_pose)
        g_l = ho["g_l"]
        idx = self.kit.robot.arm_control_idx
        # the records: the right hand's pick and the left hand's hold
        grasp.carry_height = float(ho["carry"])
        grasp.waypoints = list(ho["wps_r"])
        grasp.extras.update({"attach_route": "handover", "handover": {"role": "pick", "object_pose": [[float(v) for v in ho["M"][0]], [float(v) for v in ho["M"][1]]],
                                                                      "hands_apart_m": round(float(ho["hands_apart_m"]), 4), "release_retreat_m": 0.10,
                                                                      "evaluated": ho["tried"], "fails": ho["fails"]}})
        hold = TT.GraspRecord(object=tname, arm="left", grasp_key=self.hold_key, grasp_name=g_l.name, yaw=0.0, pre_grasp=ho["pre_l"], grasp=ho["H_l"],
                              carry_height=0.0, finger_q=[float(v) for v in (g_l.finger_q or [])] or None, eef_in_object=(TT._t(g_l.pose[0]), TT._t(g_l.pose[1])),
                              stance="manipulation", place_on=pname, waypoints=list(att["wps"]))
        hold.pre_place, hold.place = pre_attach, attach
        hold.extras.update({"attach_route": "handover", "handover": {"role": "hold", "object_pose": grasp.extras["handover"]["object_pose"]},
                            "attach_stance": work_st.name, "lift": [float(v) for v in att["start"][0]], "slerp_policy": att.get("slerp_policy"),
                            "pinch_width_m": round(2.0 * self._finger_q_of(g_l), 4)})
        manip.arms = ["right", "left"]
        self.sidecar.add_stance(manip)
        self.sidecar.add_grasp(grasp)
        self.sidecar.add_grasp(hold)
        self.record_carry_arm_q(tname, "left", TT._t(ho["q_l"])[idx["left"]])
        objects = {tname: m_pose, pname: f_pose}
        if work_st is not manip:
            work_st.carried = tname
            self.sidecar.add_stance(work_st)
            male.set_position_orientation(*m_pose)
            male.keep_still()
            self.kit.sync_world()
            path = self.record_path(manip, work_st, carried=tname, arm="left", axis_leg=[0.0],
                                    arm_q={"left": TT._t(ho["q_l"])[idx["left"]], "right": TT._t(ho["q_x"])[idx["right"]]})
            work_st.extras.update({"second_base_pose": True, "lateral_shift": float(work_st.lateral - manip.lateral),
                                   "stand_off_shift": float((work_st.stand_off or 0.0) - (manip.stand_off or 0.0)), "path_length_m": getattr(path, "length_m", None)})
        work_st.arms = ["left"] if work_st is not manip else work_st.arms
        for nm, p in (("grasp", grasp.grasp), ("hand-over (right)", ho["H_r"]), ("hand-over (left)", ho["H_l"])):
            if not self.ikc.eef_in_frame(p, cam_manip):
                raise Reject("gripper_not_in_frame", f"{nm} eef pose outside the manipulation image (5 %)")
        TT.whole_body_check_at(self, [init, manip] + ([work_st] if work_st is not manip else []), objects, require=True, reason="stance_collision")
        work_st.extras.update({"brace": "none", "handover": True})
        d.object_poses.update(objects)
        d.goal_poses[tname] = goal
        d.robot_q = self.stance_q(init)
        d.base_pose2d = tuple(init.base_pose2d)
        d.groups[self._group_name([tname, pname])] = self._group_target(f_pose, goal, tname, pname)
        child_link, parent_link = self.pair
        d.extras["attachment_targets"] = [{"child": tname, "parent": pname, "child_link": child_link.body_name, "parent_link": parent_link.body_name, "joint_type": "fixed"}]
        mc, pc = self.choice.models[self.t_role], self.choice.models[self.p_role]
        self.sidecar.describe(tname, mc.category.replace("_", " "))
        self.sidecar.describe(pname, pc.category.replace("_", " "))
        d.summary.update({"grasp": {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw}, "hold": {"key": hold.grasp_key, "name": hold.grasp_name},
                          "manipulation_candidate": k, "partner_fails": self._att_fails, "brace": "none", "attach_stance": work_st.name,
                          "second_base_pose": work_st is not manip,
                          "attach_route": {"route": "handover", "pick_key": self.grasp_key, "hold_key": self.hold_key, "hands_apart_m": round(float(ho["hands_apart_m"]), 4)}})
        self.brace_missing = True
        self._route_count("drafted")
        return d

    # ------------------------------------------------------------------ asymmetric library grasps re-centred (cap/xxqyrt)

    def _meta_link_axis_local(self, male) -> Optional[th.Tensor]:
        """The male attachment meta link's +z in the object (root) frame - the insertion direction the pair's geometry uses
        (``insertion axis {"source": "meta link +z"}``); ``None`` when the pair is not resolved yet (the caller assumes vertical)."""
        try:
            child_link = self.pair[0] if self.pair is not None else None
            if child_link is None:
                return None
            cp, cq = child_link.get_position_orientation()
            rp, rq = male.get_position_orientation()
            _, q_rel = T.relative_pose_transform(cp, cq, rp, rq)
            return C.quat_rotate(TT._t(q_rel), th.tensor([0.0, 0.0, 1.0]))
        except Exception:  # noqa: BLE001
            return None

    def _recentre_grasps(self, grasps: List[Any], male) -> List[Any]:
        """Library grasps whose pinch centre (pad centre) lies more than ``generation.pinch_offset_max_m`` (15 mm) off the male
        hull's centre along the closing axis are moved onto it (iors_small's ``recentre_pinches`` rule with a wider limit):
        the oracle closes both fingers symmetrically about the eef, but the library's asymmetric AG grasps (cap/xxqyrt: fingers
        50 / 0 mm, eef 22 mm off the cap's centre, 'both fingertips on the rim') would put one pre-shaped pad on top of the
        part. The re-centred pose must stay attachable (self-collision of the attached part); the limit leaves the validated
        lid grasps (<= 10 mm off) and their cached bands untouched. ``template_facts.recentred_grasps``."""
        import copy

        limit = float(self.gen("pinch_offset_max_m", 0.015))
        hull = TT._t(self.hull[self.t_role])
        out, moved = [], {}
        for g in grasps:
            pos, quat = TT._t(g.pose[0]), TT._t(g.pose[1])
            R = C.quat_to_mat(quat)
            a, c = R[:, 2], R[:, 1]
            pc = pos + C.CONTACT_CENTER_Z * a
            proj = hull @ c
            off = float(th.dot(pc, c)) - 0.5 * float(proj.min() + proj.max())
            if abs(off) <= limit:
                out.append(g)
                continue
            shifted = copy.copy(g)
            shifted.position = pos - off * c
            if self.ikc.attachable(male, shifted.pose, "right"):
                moved[g.name] = round(off, 4)
                out.append(shifted)
            else:
                log.info("%s: grasp %s pinch %+.1f mm off the male's centre; the re-centred pose is not attachable - kept", self.gp.variant, g.name, off * 1000)
                out.append(g)
        if moved:
            log.info("%s: %d asymmetric library grasps re-centred along the closing axis (offsets m): %s", self.gp.variant, len(moved), moved)
        self.template_facts["recentred_grasps"] = moved
        return out

    # ------------------------------------------------------------------ pinch width from the male's geometry (0919-210926)

    def _contact_band_centre(self, eef_pose: TT.Pose, m_pose: TT.Pose) -> Optional[float]:
        """Position along the approach axis (eef frame z, fingertips +) of the middle of the part's extent that lies within the
        finger's inner face, [PAD_Z_RANGE[0], FINGERTIP_LENGTH] - where a symmetric closure pinches it. None when no hull
        point lies in that band."""
        pts = C.transform_points((TT._t(m_pose[0]), TT._t(m_pose[1])), TT._t(self.hull[self.t_role]))
        loc = C.transform_points(C.invert((TT._t(eef_pose[0]), TT._t(eef_pose[1]))), pts)
        band = (loc[:, 2] >= C.PAD_Z_RANGE[0]) & (loc[:, 2] <= C.FINGERTIP_LENGTH) & (loc[:, 0].abs() <= 0.03)
        if int(band.sum()) < 2:
            return None
        return 0.5 * (float(loc[band, 2].min()) + float(loc[band, 2].max()))

    def _record_pinch_width(self, grasp, m_pose: TT.Pose) -> None:
        """``grasps[].pinch_width_m``: the male's width across the closing axis at the pads (its hull points inside the pads'
        band along the approach axis, ``PAD_Z_RANGE``, within the pads' lateral reach) at the recorded grasp pose. The
        oracle pre-shapes the fingers to ``width / 2 + margin`` before the descent; it read the grasp library's ``width``,
        which for lid/pwgbsb is 6-12 mm on a 24-30 mm knob (the library's asymmetric one-finger wedges): the pads opened
        to a 26 mm gap, landed on the knob's flare 13 mm above the pinch, and the closure caught nothing (0919-173115 /
        210926, failed_grasp x4). Facts on the record."""
        if grasp.grasp is None:
            return
        pts = C.transform_points((TT._t(m_pose[0]), TT._t(m_pose[1])), TT._t(self.hull[self.t_role]))
        eef = (TT._t(grasp.grasp[0]), TT._t(grasp.grasp[1]))
        loc = C.transform_points(C.invert(eef), pts)  # object hull in the eef frame: z = approach axis (tips +), y = closing axis
        # the band runs from the pads' wrist-side end to the fingertips: a thin part lifted by the support-clearance rule
        # (cap/xxqyrt) is pinched by the fingertip section beyond the pads
        band = (loc[:, 2] >= C.PAD_Z_RANGE[0]) & (loc[:, 2] <= C.FINGERTIP_LENGTH + 0.002) & (loc[:, 0].abs() <= 0.03)
        if int(band.sum()) < 2:
            grasp.extras["pinch_width_reason"] = "no hull points between the pads and the fingertips"
            return
        width = float(loc[band, 1].max() - loc[band, 1].min())
        lib = None
        try:
            lib = float((C.grasp_entry(grasp.grasp_key, grasp.grasp_name).meta or {}).get("width"))
        except Exception:  # noqa: BLE001
            pass
        grasp.extras.update({"pinch_width_m": round(width, 4), "pinch_width_library_m": lib, "pinch_width_band_points": int(band.sum())})
        log.info("%s: pinch width at the pads %.1f mm from the male's hull (library %s mm)", self.gp.variant, width * 1000, None if lib is None else round(lib * 1000, 1))

    # ------------------------------------------------------------------ over-the-top pre-grasp approach (owner 2026-09-19 20:42)

    def _record_approach_standoff(self, grasp, f_pose: TT.Pose, manip: Stance) -> None:
        """When the partner stands within ``generation.partner_near_m`` (0.35 m, hull-to-grasp horizontal distance) of the
        male's grasp, the grasp record carries ``approach_standoff_m`` (``generation.grasp_approach_standoff_m``, 0.12): the
        oracle reaches the pre-grasp through a standoff that far back along the approach axis (above a top-down grasp) and a
        straight leg, instead of the free trajopt that swung the open hand at table height past the partner (pwgbsb /
        jlalfc 0919-173115 / 181512: the wrist camera grazed the 29 cm teapot's spout). Recorded only when the standoff pose
        passes the exact IK at the grasp stance. Facts on the record."""
        near = float(self.gen("partner_near_m", 0.35))
        ladder = [float(v) for v in (self.gen("grasp_approach_standoff_ladder_m", None) or (0.12, 0.08, 0.05)) if float(v) > 0.0]
        if grasp.grasp is None or grasp.pre_grasp is None or not ladder:
            return
        pts = C.transform_points((TT._t(f_pose[0]), TT._t(f_pose[1])), TT._t(self.hull[self.p_role]))
        g_xy = TT._t(grasp.grasp[0])[:2]
        d_xy = float(th.linalg.norm(pts[:, :2] - g_xy, dim=1).min())
        grasp.extras["partner_distance_m"] = round(d_xy, 4)
        if d_xy > near:
            return
        # a ladder of standoff distances (cap/xxqyrt 0920-164504: the 12 cm standoff had no IK at the 0.82 m-deep stance - joint 2 on
        # its limit - so the direct trajopt ran the open hand along the table past the 25 cm bottle and shoved it 46 mm / 32 deg)
        reasons = []
        for dist in ladder:
            standoff = C.pre_pose((TT._t(grasp.pre_grasp[0]), TT._t(grasp.pre_grasp[1])), dist)
            res = self.ikc.exact([standoff], manip, grasp.arm, ignore_for_last=[self.obj("support")], scale_key=self.choice.models[self.t_role].scale, cache=False)
            if res.ok:
                grasp.extras["approach_standoff_m"] = dist
                if reasons:
                    grasp.extras["approach_standoff_ladder"] = reasons
                log.info("%s: partner %.2f m from the grasp -> over-the-top pre-grasp approach recorded (standoff %.2f m%s)", self.gp.variant, d_xy, dist,
                         f"; {len(reasons)} longer standoff(s) had no IK" if reasons else "")
                return
            reasons.append(f"{dist:.2f} m: {res.reason}")
        grasp.extras["approach_standoff_reason"] = "; ".join(reasons)
        log.info("%s: partner %.2f m from the grasp but no standoff of %s has IK (%s): direct approach kept", self.gp.variant, d_xy, ladder, reasons[-1][:120])

    # ------------------------------------------------------------------ hand envelope vs the partner (0919-160750)

    def _hand_clear_of_partner(self, grasp, f_pose: TT.Pose, margin: float = 0.02) -> Optional[str]:
        """The hand beyond the finger profile (``HAND_SPHERES_EEF``: wrist camera, knuckles, gripper body) along the grasp chain
        at the male — pre-grasp, the approach, the grasp, the lift — must stay ``margin`` clear of the partner's hull box at
        ``f_pose``; returns the offending part or None. 0919-160750 (lid pwgbsb / teapot jlalfc): the grasp is solved with
        the partner parked away and the partner cell is admitted on a 4 cm footprint gap, but the wrist camera hangs 5.6 cm
        beside the eef and 7.6 cm above the pads — it clipped the 29 cm teapot's spout at the end of the grasp (QA
        ``collision`` x2). The SLERP / attach poses are IK-checked with the partner in the world already."""
        if grasp.grasp is None or grasp.pre_grasp is None:
            return None
        hull = TT._t(self.hull[self.p_role])
        lo, hi = hull.min(dim=0).values, hull.max(dim=0).values
        inv = C.invert((TT._t(f_pose[0]), TT._t(f_pose[1])))
        g_pos, g_q = TT._t(grasp.grasp[0]), TT._t(grasp.grasp[1])
        p_pos = TT._t(grasp.pre_grasp[0])
        poses = [(p_pos + (g_pos - p_pos) * k / 4.0, g_q) for k in range(5)]  # pre-grasp -> grasp
        poses.append((g_pos + th.tensor([0.0, 0.0, max(CARRY_HEIGHTS)]), g_q))  # the lift
        for eef in poses:
            for part, (centres, r) in HAND_SPHERES_EEF.items():
                pts_o = C.transform_points(inv, C.transform_points(eef, th.tensor(centres, dtype=th.float32)))
                d = th.clamp(lo - pts_o, min=0.0) + th.clamp(pts_o - hi, min=0.0)
                dist = float(th.linalg.norm(d, dim=1).min())
                if dist < r + margin:
                    return f"{part} {(dist - r) * 1000:+.0f} mm from the partner's hull box at the grasp chain (margin {margin * 1000:.0f} mm)"
        return None

    # ------------------------------------------------------------------ grasp height margin (owner 2026-09-19 16:49)

    def _is_thin_part(self) -> bool:
        """A male shorter than the pads' fingertip section + the band margin (cap/xxqyrt 1.1 cm): a fingertip pinch whatever
        the grasp height, so the thin-part rules (support clearance, tilted pinches first) apply."""
        return float(self.height.get(self.t_role, 1.0)) < THIN_PART_MAX_HEIGHT_M

    def _grasp_tilt_deg(self, g) -> float:
        """Angle between the grasp's approach axis (male at rest) and the vertical: 0 = straight down, 30 = the rim pinches."""
        a = C.quat_rotate(TT._t(self.rest[self.t_role]), C.approach_axis(TT._t(g.pose[1])))
        return float(np.degrees(math.acos(max(-1.0, min(1.0, -float(a[2]))))))

    def _grasp_pools(self) -> List[Tuple[List[Any], str]]:
        """The grasp lists step 1 screens in order. A thin male (owner 2026-09-20 20:41): the tilted rim pinches (approach >=
        THIN_PART_TILT_MIN_DEG from vertical; cap/xxqyrt grasp_1..3) before the vertical ones (grasp_4/5, whose fingertips rest
        on the table around an 1.1 cm part); the full library is the fallback. Any other male: the full library only."""
        grasps = list(self.grasps)
        if not self._is_thin_part():
            return [(grasps, "all")]
        tilted = [g for g in grasps if self._grasp_tilt_deg(g) >= THIN_PART_TILT_MIN_DEG]
        if not tilted or len(tilted) == len(grasps):
            return [(grasps, "all")]
        log.info("%s: thin male (%.1f cm): %d tilted rim pinch(es) screened before the %d vertical one(s)", self.gp.variant,
                 self.height[self.t_role] * 100, len(tilted), len(grasps) - len(tilted))
        return [(tilted, "tilted"), (grasps, "all")]

    @staticmethod
    def _finger_lowest_z(eef_pose: TT.Pose) -> Optional[float]:
        """World z of the lowest finger-mesh vertex at ``eef_pose`` over the closure (finger joint FINGER_CLOSED .. FINGER_OPEN;
        z is linear in the opening, both ends cover it); None without the robot assets."""
        from ispatialgym.oracle import r1pro_model as M

        pos, R = TT._t(eef_pose[0]).numpy(), C.quat_to_mat(TT._t(eef_pose[1])).numpy()
        zs = []
        for q in (C.FINGER_CLOSED, C.FINGER_OPEN):
            pts = M.finger_points_eef(q)
            if pts is None:
                return None
            zs.append(float((pts @ R.T + pos)[:, 2].min()))
        return min(zs)

    def _fingers_in_partner(self, q: Optional[th.Tensor], attach_pose: TT.Pose, support, male, female) -> Optional[str]:
        """Item 2a (owner 2026-09-20 23:39): the attach configuration ``q`` (solved with the male attached and the partner
        ignored, as the male sits inside it) re-verified with the male DETACHED and the partner PRESENT - only the support
        and the held male leave the world - so the fingers / hand are checked against the partner, which the male-partner
        collision filter never covers (cork/wine 0920-212757: the fingertips 1 cm below the rim pushed the bottle 7 mm /
        4.3 deg). Returns the reason when they collide, None when clear. Call it OUTSIDE ``attached(...)``."""
        if q is None:
            return None
        if self.ikc.collision_free_at(q, attach_pose, "right", ignore=[support, male]):
            return None
        return f"robot links collide with {getattr(female, 'name', 'the partner')} at the attach pose (male detached, partner present)"

    def _attach_clearance(self, grasp) -> Optional[Dict[str, float]]:
        """Heights of the pads' fingertip-side end and of the fingertip axis point ABOVE the partner's top after the insertion,
        along the insertion axis (owner 2026-09-20 23:39, item 2b; cork/wine 0920-212757: the recorded pinch 8 mm above the
        mating end, the attach seats the cork 1.8 cm into the neck, the fingertips end 1 cm below the rim and push it - the
        male-partner collision filter never covers the fingers). Computed in the partner's frame from the meta-link
        transforms alone (``T_F ∘ inv(T_M) ∘ T_obj→eef``), so it is available at step 1 before the partner is placed:
        ``depth`` = female link -> partner top along the axis (the partner hull points within the male's radius + 5 mm of
        the axis: the rim, not a handle or spout), ``h_pad`` / ``h_tip`` = pad end / fingertip axis point above that top,
        ``lift_sign`` = how a lift of the grasp along the male's rest up axis moves the fingers along the insertion axis
        (+1 for an upright insertion). None when the transforms are missing."""
        need = ("parent_link_in_female", "root_in_child_link", "insertion_axis_local")
        if any(getattr(self, k, None) is None for k in need) or grasp.eef_in_object is None:
            return None
        p_f, q_f = (TT._t(v) for v in self.parent_link_in_female)
        axis = C.unit(TT._t(self.insertion_axis_local))
        root = C.compose((p_f, q_f), tuple(TT._t(v) for v in self.root_in_child_link))  # male root at the attach, partner frame
        eef = C.compose(root, (TT._t(grasp.eef_in_object[0]), TT._t(grasp.eef_in_object[1])))
        a = C.approach_axis(eef[1])
        pts = TT._t(self.hull[self.p_role]) - p_f
        h = pts @ axis
        radial = (pts - h.unsqueeze(1) * axis).norm(dim=1)
        r_male = float((TT._t(self.hull[self.t_role])[:, :2]).norm(dim=1).max())
        # owner 2026-09-24 item 4: the partner's top is what lies UNDER THE FINGERTIPS - the partner hull points within a fingertip
        # band of each tip's radial distance from the insertion axis (the tips pinch the male at +- half the pinch width along the
        # closing axis) - not the highest point within the male's radius: a cap wider than its bottle's neck (xxqyrt 51 mm rim, the
        # neck ~26 mm) put the rim height under fingertips that sit outside the neck, over the shoulder, and the 11 mm cap had no
        # admissible band (0925-092708: 8/8 fingers_in_partner, 'even a fingertip pinch needs a 24.1 mm lift with 17.1 mm of part
        # left'); the cork's tips stand over the rim and keep their floor
        c_axis = C.quat_to_mat(eef[1])[:, 1]
        w = self._pinch_width_at(grasp.eef_in_object)
        tip_axis = eef[0] + C.FINGERTIP_LENGTH * a
        tips = [tip_axis + s * 0.5 * w * c_axis for s in (1.0, -1.0)] if w is not None else [tip_axis]
        r_tips = []
        for t_ in tips:
            d_ = t_ - p_f
            r_tips.append(float((d_ - float(d_ @ axis) * axis).norm()))
        near = th.zeros(len(pts), dtype=th.bool)
        for r_t in r_tips:
            near |= (radial - r_t).abs() <= TIP_RADIAL_BAND_M
        under_tips = bool(near.any())
        depth_tips = float(h[near].max()) if under_tips else None  # may lie below the female link (the shoulder under a wide cap's tips)
        # the 09-20 rim rule: the highest partner point within the male's radius (+ 5 mm) of the axis, never below the female link
        legacy_near = radial <= r_male + 0.005
        depth_rim = max(0.0, float(h[legacy_near].max())) if bool(legacy_near.any()) else max(0.0, float(h.max()))
        # owner 2026-09-25 (cork 0925-145307: the tips' 4.25 mm band met no hull vertex between the rim circle and the shoulder,
        # so the "partner top under the tips" was a point 20 cm down the bottle; the rim floor vanished, the lift shrank from 31 to
        # 14 mm, the fingers reached into the neck and the attach was refused twice at 15-16 mm): the pair is CLASSIFIED -
        #   into_opening: the male enters the partner (its radius within the partner's rim radius at the female link: the cork) -
        #                 the rim rule, fingertips >= PARTNER_CLEARANCE_M above the rim;
        #   over_partner: the male goes over the partner (wider than the rim: the cap) - item 4's relief (the partner top under
        #                 the fingertips), unless that top lies further below the female link than the male's own length (the
        #                 band caught the shoulder / the body): then the rim rule again.
        r_under_tips = float(radial[near & ((h - (depth_tips if depth_tips is not None else 0.0)).abs() <= 0.005)].max()) if under_tips else 0.0  # the partner's extent at the level found under the tips
        near_link = h.abs() <= 0.005
        r_rim = float(radial[near_link].max()) if bool(near_link.any()) else None  # the rim's outer radius at the female link
        r_open = float(radial[near_link].min()) if bool(near_link.any()) else None  # the opening (a closed mesh: ~ the outer radius)
        male_h = TT._t(self.hull[self.t_role]) @ C.unit(TT._t(self.insertion_axis_local))
        male_len = float(male_h.max() - male_h.min())
        if r_rim is None:
            klass, rule, depth = "unknown_rim", "rim rule (no partner hull at the female link height)", depth_rim
        elif r_male <= r_open + 0.002:
            klass, rule, depth = "into_opening", "rim floor (the male enters the partner's opening)", depth_rim
        elif r_male < r_rim - 0.002:
            klass, rule, depth = "on_rim", "rim floor (the male sits on the rim: between the opening and the outer radius)", depth_rim
        elif depth_tips is None:
            klass, rule, depth = "over_partner", "no partner hull under the fingertips: no partner floor", None
        elif depth_tips < -max(male_len, TIP_REACH_BELOW_LINK_M):
            # the band caught the partner's body far below (the cork's 20 cm), not something the fingers can meet; a wide cap's tips
            # DO hang over the shoulder 3-5 cm under the mouth, further than the 11 mm cap is long - that top stays theirs
            klass, rule, depth = "over_partner", "under-tips point deeper than the male's length / the fingers' reach: rim floor", depth_rim
        elif r_under_tips - max(r_tips) < TIP_EDGE_OVERLAP_M:
            # the pads pinch the male's outside and the finger's body extends OUTWARD from the pad face: the partner surface found under
            # the tips must reach at least TIP_EDGE_OVERLAP_M beyond the pad radius to be a top the tip stands on - less is the rim's
            # outer edge (cap/xxqyrt on the lemonade bottle 0925-172312: the neck's collar reaches r 27.0 mm at the mouth, the tips at
            # r 25.5 mm, 1.5 mm of collar under them, 'even a fingertip pinch needs a 22.3 mm lift with 17.2 mm of part left') or a
            # surface inside the tips' radius; the 09-21 success grasped exactly this way (lift 14.4 mm, the fingertips 17 mm below the
            # collar's top, attach aligned to 1.1 mm, 0 QA verdicts)
            klass, rule, depth = "over_partner", f"the partner under the fingertips ends {(r_under_tips - max(r_tips)) * 1000:.1f} mm beyond the pad radius (edge / inside): no partner floor", None
        else:
            klass, rule, depth = "over_partner", "partner top under the fingertips", depth_tips
        h_pad = None if depth is None else float(((eef[0] + C.PAD_Z_RANGE[1] * a) - p_f) @ axis) - depth
        h_tip = None if depth is None else float(tip_axis @ axis - float(p_f @ axis)) - depth
        up_obj = C.quat_rotate(C.qinv(TT._t(self.rest[self.t_role])), th.tensor([0.0, 0.0, 1.0]))  # world up at rest, male frame
        lift_sign = float(C.quat_rotate(root[1], up_obj) @ axis)
        facts = {"pair_class": klass, "floor_rule": rule, "r_male_m": round(r_male, 4), "r_rim_m": None if r_rim is None else round(r_rim, 4),
                 "r_opening_m": None if r_open is None else round(r_open, 4),
                 "male_length_m": round(male_len, 4), "r_tips_m": [round(r, 4) for r in r_tips], "depth_rim_m": round(depth_rim, 4),
                 "depth_tips_m": None if depth_tips is None else round(depth_tips, 4), "depth_used_m": None if depth is None else round(depth, 4),
                 "r_under_tips_m": round(r_under_tips, 4)}
        self.template_facts["attach_pair_class"] = facts
        return {"depth_m": depth, "h_pad_m": h_pad, "h_tip_m": h_tip, "lift_sign": lift_sign, "r_male_m": r_male, "r_tips_m": [round(r, 4) for r in r_tips],
                "pinch_width_m": None if w is None else round(float(w), 4), "partner_points_under_tips": int(near.sum()), "depth_within_male_radius_m": depth_rim,
                "depth_under_tips_m": depth_tips, "pair_class": klass, "floor_rule": rule, "r_rim_m": r_rim, "r_opening_m": r_open, "male_length_m": male_len}

    def _pinch_width_at(self, eef_in_object) -> Optional[float]:
        """The male's chord across the closing axis between the pads at ``eef_in_object`` (the band of :meth:`_record_pinch_width`,
        in the male's frame); None without two hull points in the band."""
        eef = (TT._t(eef_in_object[0]), TT._t(eef_in_object[1]))
        loc = C.transform_points(C.invert(eef), TT._t(self.hull[self.t_role]))
        band = (loc[:, 2] >= C.PAD_Z_RANGE[0]) & (loc[:, 2] <= C.FINGERTIP_LENGTH + 0.002) & (loc[:, 0].abs() <= 0.03)
        if int(band.sum()) < 2:
            return None
        return float(loc[band, 1].max() - loc[band, 1].min())

    def _apply_grasp_height_extra(self, grasp, m_pose: TT.Pose, manip: Stance) -> None:
        """``generation.grasp_height_extra_m``: the knob pinch moves up the male by up to that much (0 = today's pose). Demo
        0919-153953 (lid tnwisp): the pads held the top centimetre of the knob while the fingertips reached 0.7 cm below the
        pads into the lid's dome, so the closure pressed the lid onto the table and the torso took the reaction (head camera
        +3.5 mm at the closure; the table itself moved 0.02 mm). The generic ``contact_height_extra_m`` rule clips such a
        grasp to zero (its pad centre already lies above the pinch band), so the attachment rule is on the pad's
        fingertip-side end instead: it stays >= ``PAD_BAND_MARGIN`` below the male's top, and the lifted pre-grasp / grasp
        must pass the exact IK at the grasp stance (else the library height is kept). Every derived pose (attach = goal
        ∘ T_obj→eef, the pre-attach, the SLERP) carries the lift through ``eef_in_object``. Facts on the record.

        Thin part (owner 2026-09-20 20:41; cap/xxqyrt 1.1 cm, 0920-184902 / 193713: 27-163 table-contact steps at the
        closure): the pads' fingertip-side end must stay >= PAD_TABLE_CLEARANCE_M ABOVE the support surface (the pinch band
        is the rim's upper half) - that lift is required even with the knob at 0 - and the fingertips <= 3 mm below the top."""
        x = max(0.0, float(self.gen("grasp_height_extra_m", 0.0) or 0.0))
        x_v2 = v2_library_lift_off(self.grasp_key)
        if x_v2 and x > 0.0:
            # i7-attachfix 2026-11-08 (cap xxqyrt, attachment_02): the alias's knob lift (isg_customs: grasp_height_extra_m 5 mm, the cap
            # aliases only) was made for the LEGACY grasps; a v2 library grasp was physics-validated at its own height (the cap's diametric
            # fingertip pinches: contact z 0 mm on the 11 mm cap) and the lift moved it 2.5 mm up onto the rim edge (contact z 2.1-2.5 mm,
            # tips 0.5 mm above the cap's bottom): i7af_v 300026 / 300030 closed the fingers to 0 and pushed the cap 5 mm (QA pause).
            # The support / partner floors below still apply.
            grasp.extras["grasp_height_extra_param_m"] = x
            x = 0.0
        if grasp.eef_in_object is None or grasp.grasp is None:
            grasp.extras.update({"grasp_height_extra_m": x, "grasp_height_extra_effective_m": 0.0, "grasp_height_extra_reason": "no grasp pose"})
            return
        pos, quat = TT._t(grasp.grasp[0]), TT._t(grasp.grasp[1])
        a = C.approach_axis(quat)
        pad_tip_end_z = float((pos + C.PAD_Z_RANGE[1] * a)[2])  # the fingertip-side end of the pads
        tip_z = float((pos + C.FINGERTIP_LENGTH * a)[2])
        hull_w = self.world.hull_at(self.hull[self.t_role], m_pose)
        top_z, bottom_z = float(hull_w[:, 2].max()), float(hull_w[:, 2].min())  # the male's bottom rests on the support
        room_pad = (top_z - PAD_BAND_MARGIN) - pad_tip_end_z  # the pads keep overlapping the part
        room_tip = (top_z - 0.003) - tip_z  # fingertip pinch: the tips stay >= 3 mm below the part's top
        floors: Dict[str, float] = {}  # lifts the support asks for (both regimes)
        # the finger ENVELOPE (both finger meshes, every opening of the closure) must clear the support: the fingers are 3.8 cm
        # across the closing plane, so a 30 deg rim pinch hangs a tip corner up to 1.2 cm below the fingertip axis point -
        # 0920-203052 (cap, pad end 5.4 mm above the table) still touched the table for 22 steps at 110 N with the lowest
        # finger vertex 0.1 mm below the surface (recomputed from the meshes: r1pro_model.finger_points_eef)
        low_z = self._finger_lowest_z(grasp.grasp)
        if low_z is not None:
            floors["finger envelope"] = (bottom_z + FINGER_SUPPORT_CLEARANCE_M) - low_z
        if room_pad <= 0.0:
            # a part shorter than the pads' fingertip section (cap/xxqyrt: 1.1 cm tall) is a FINGERTIP pinch whatever the height -
            # the pad rule would forbid any lift and the fingertips rest on the table during the closure (0920-184902: 23 contact
            # steps, 123 N); the pads' fingertip-side end must clear the support by PAD_TABLE_CLEARANCE_M (owner 20:41)
            floors["pad end (thin part)"] = (bottom_z + PAD_TABLE_CLEARANCE_M) - pad_tip_end_z
        # the partner after the insertion (owner 2026-09-20 23:39, 2b): the pads' fingertip-side end (pad regime) / the fingertip
        # axis point (fingertip regime) must stay PARTNER_CLEARANCE_M above the partner's top - cork/wine 0920-212757 seated
        # the cork 1.8 cm into the neck with the fingertips 1 cm below the rim and pushed the bottle 7 mm / 4.3 deg
        clr = self._attach_clearance(grasp)
        floors_pad, floors_tip = dict(floors), dict(floors)
        if clr is not None and clr["lift_sign"] > 0.5 and clr.get("h_tip_m") is not None:
            # the fingertips reach 6.9 mm beyond the pads' end along the approach, so they - not the pad end - are the lowest part
            # of the finger over the rim in BOTH regimes (cork 0920-235939: the pad end 5 mm above the rim left the tips 1 mm below
            # it); the owner's pad-end condition (pinch point >= depth + PAD_Z_RANGE[1] + 5 mm) is implied by the tips' clearance
            floors_pad["partner top (fingertips)"] = PARTNER_CLEARANCE_M - clr["h_tip_m"]
            floors_tip["partner top (fingertips)"] = PARTNER_CLEARANCE_M - clr["h_tip_m"]
            grasp.extras["attach_clearance"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in clr.items()}
        elif clr is not None and clr["lift_sign"] > 0.5:
            # no partner hull under the fingertips (a cap wider than the neck it screws on: the tips hang over the shoulder, the
            # rim is inside them): the partner-top floor does not apply; the finger-envelope / pad floors and the attach-pose
            # collision check (_fingers_in_partner) still guard the fingers
            grasp.extras["attach_clearance"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in clr.items()}
            grasp.extras["attach_clearance"]["partner_floor"] = "none: no partner hull under the fingertips"
            log.info("%s: no partner hull within %.0f mm of the fingertips' radius %s mm (rim within the male's radius at depth %.1f mm): no partner-top floor",
                     self.gp.variant, TIP_RADIAL_BAND_M * 1000, [round(r * 1000, 1) for r in clr["r_tips_m"]], clr["depth_within_male_radius_m"] * 1000)
        elif clr is not None and clr["lift_sign"] < -0.5 and clr.get("h_tip_m") is not None:
            # the male is inserted the other way up than it rests (its rest flipped onto its stable end, item 5: the cork stands on
            # its wide end, the narrow end goes into the neck): a HIGHER grasp at rest brings the fingertips DOWN toward the rim at
            # the attach, so the partner clearance caps the lift instead of flooring it - the same 5 mm above the partner's top
            ceiling = clr["h_tip_m"] - PARTNER_CLEARANCE_M
            grasp.extras["attach_clearance"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in clr.items()}
            grasp.extras["attach_clearance"]["partner_rule"] = f"ceiling {ceiling * 1000:.1f} mm (inserted the other way up than it rests)"
            if ceiling < 0.0:
                self.counters["fingers_in_partner"] = self.counters.get("fingers_in_partner", 0) + 1
                raise Reject("fingers_in_partner", f"the fingertips end {-clr['h_tip_m'] * 1000:.1f} mm below the partner's top at the attach for the unlifted grasp "
                                                   f"(inserted the other way up than it rests; attach clearance {grasp.extras['attach_clearance']})")
            room_pad, room_tip = min(room_pad, ceiling), min(room_tip, ceiling)
            log.info("%s: the male is inserted the other way up than it rests (lift sign %.2f): the partner clearance caps the grasp lift at %.1f mm", self.gp.variant, clr["lift_sign"], ceiling * 1000)
        elif clr is not None:
            log.info("%s: the male is not inserted along its rest up axis (lift sign %.2f): no partner-clearance floor", self.gp.variant, clr["lift_sign"])
        need_pad = max(floors_pad.values()) if floors_pad else 0.0
        need_tip = max(floors_tip.values()) if floors_tip else 0.0
        if room_pad > 0.0 and need_pad <= room_pad:
            room, rule, regime, floors = room_pad, "pad end 5 mm below the male's top", "pad pinch", floors_pad
        elif need_tip <= room_tip:
            # the pads cannot overlap the part while the fingers clear the support / the partner (every cap/xxqyrt library grasp
            # puts the fingertip axis 8.4-8.9 mm below the table; 0920-212214 clipped the 14.4 mm floor to the 9.2 mm pad room and
            # the lowest vertex stayed 1.2 mm under the surface; the cork protrudes 1.6 cm from the neck, less than the pads need):
            # a fingertip pinch of the part's upper band, tips >= 3 mm below the top
            room, rule, regime, floors = room_tip, f"fingertips 3 mm below the male's top (fingertip pinch; pad end >= {PAD_TABLE_CLEARANCE_M * 1000:.0f} mm above the support)", "fingertip pinch", floors_tip
        else:
            facts = {"floors_pad_mm": {k: round(v * 1000, 1) for k, v in floors_pad.items()}, "room_pad_mm": round(room_pad * 1000, 1),
                     "floors_tip_mm": {k: round(v * 1000, 1) for k, v in floors_tip.items()}, "room_tip_mm": round(room_tip * 1000, 1),
                     "attach_clearance": grasp.extras.get("attach_clearance")}
            self.counters["fingers_in_partner"] = self.counters.get("fingers_in_partner", 0) + 1
            raise Reject("fingers_in_partner", f"no grasp height keeps the fingers out of the support / partner: even a fingertip pinch needs a "
                                               f"{need_tip * 1000:.1f} mm lift with {room_tip * 1000:.1f} mm of part left ({facts})")
        needed = max(floors.values()) if floors else 0.0
        if floors and needed > x:
            which = max(floors.items(), key=lambda kv: kv[1])[0]
            log.info("%s: lift raised from %.1f to %.1f mm by the %s floor (%s regime; floors %s mm)", self.gp.variant, x * 1000, needed * 1000, which, regime,
                     {k: round(v * 1000, 1) for k, v in floors.items()})
            x = needed
        grasp.extras["grasp_height_floors_m"] = {k: round(v, 4) for k, v in floors.items()}
        grasp.extras["grasp_height_regime"] = regime
        if x <= 0.0:
            grasp.extras.update({"grasp_height_extra_m": x, "grasp_height_extra_effective_m": 0.0, "grasp_height_extra_reason": "off"})
            return
        eff = max(0.0, min(x, room))
        why = "full" if eff >= x - 1e-9 else (f"clipped: {rule}" if eff > 0 else f"clipped to 0: {rule}")
        if eff > 0.0:
            R_obj = C.quat_to_mat(TT._t(m_pose[1]))
            up_obj = R_obj.T @ th.tensor([0.0, 0.0, 1.0])  # world up in the male's frame
            # the lift slides the part along the fingers: the contact band (the finger's inner face [PAD_Z_RANGE[0], FINGERTIP_LENGTH]
            # over the part's extent) moves toward the fingertips, and with a tilted approach that axial shift is a horizontal
            # offset of the pinch line - the library centres the PAD centre on the male's axis, a fingertip pinch closes 27 mm
            # further ahead = 13-15 mm beside the axis at 30 deg: cork 0921-004359's tips passed beside the 17 mm cork and their
            # corner tipped it over (the cap's 51 mm rim still spanned the off-centre chord: hold 43 mm). The hand moves
            # horizontally so the contact band's centre stays on the vertical line it had at the library height
            s0 = self._contact_band_centre(grasp.grasp, m_pose)
            lifted = (pos + eff * th.tensor([0.0, 0.0, 1.0]), quat)
            s1 = self._contact_band_centre(lifted, m_pose)
            shift_w = th.zeros(3)
            if s0 is not None and s1 is not None:
                shift_w = -(s1 - s0) * th.tensor([float(a[0]), float(a[1]), 0.0])
            eef_in_object = (TT._t(grasp.eef_in_object[0]) + eff * up_obj + R_obj.T @ shift_w, TT._t(grasp.eef_in_object[1]))
            eef = IK.grasp_pose(m_pose, eef_in_object)
            pre = IK.pre_pose(eef)
            support = self.obj("support")
            res = self.ikc.exact([pre, eef], manip, grasp.arm, ignore_for_last=[support], scale_key=self.choice.models[self.t_role].scale, cache=False,
                                 approached=[self.obj(self.t_role)])
            if res.ok:
                grasp.eef_in_object, grasp.grasp, grasp.pre_grasp = eef_in_object, eef, pre
                if float(th.linalg.norm(shift_w)) > 1e-4:
                    grasp.extras["pinch_recentre_m"] = [round(float(v), 4) for v in shift_w.tolist()]
                    log.info("%s: pinch re-centred: the contact band moved %.1f mm along the fingers with the lift -> hand moved %.1f mm horizontally "
                             "so the band's centre stays on the male's axis", self.gp.variant, (s1 - s0) * 1000, float(th.linalg.norm(shift_w)) * 1000)
            else:
                if floors and needed > 0.0:
                    # owner 2026-09-25 (cork 0925-195316 / 205211, replays 213602 / 215043: 'IK failed with the lift (ik_joint_limit:
                    # right_arm_joint5 within 34 mrad of its limit); library height kept' left the fingertips 25.9 mm BELOW the rim
                    # at the attach - the partner-top floor had asked for 30.9 mm - and the demo's insertion legs jammed on the
                    # neck (right_arm_joint3 0.055 rad short of its target, the cork 18 mm above its seat, 4.4 deg tilted, refused
                    # twice). A lift a floor demands is not optional: this candidate (grasp x stance) is rejected and the next one tried
                    which = max(floors.items(), key=lambda kv: kv[1])[0]
                    self.counters["lift_unreachable"] = self.counters.get("lift_unreachable", 0) + 1
                    raise Reject("lift_unreachable", f"the {which} floor needs a {needed * 1000:.1f} mm grasp lift and the lifted grasp has no IK from this stance "
                                                     f"({res.reason}); the library height would leave the fingers in the support / partner "
                                                     f"(floors {grasp.extras['grasp_height_floors_m']}, attach clearance {grasp.extras.get('attach_clearance')})")
                eff, why = 0.0, f"IK failed with the lift ({res.reason}); library height kept (no floor asked for it)"
        log.info("%s: grasp height extra %.1f mm requested -> %.1f mm (%s; pad end %.3f, male top %.3f)", self.gp.variant, x * 1000, eff * 1000, why, pad_tip_end_z, top_z)
        grasp.extras.update({"grasp_height_extra_m": x, "grasp_height_extra_effective_m": round(eff, 4), "grasp_height_extra_reason": why})

    # ------------------------------------------------------------------ program check (brace-less attachment)

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        """§3.5 program check. A brace-less attachment (owner 17:10: the brace is optional only when the joint placement has no
        cell after the second base pose) is not yet accepted by the F6 program; until that lands, with ``ISG_PROGRAM_CHECK=stub``
        set inside the smoke command a rejection that only asks for ``contacts.brace`` is recorded as ``stub-brace-optional``."""
        try:
            return super().program_check(draft)
        except Reject as e:
            msg = str(e)
            if not (getattr(self, "brace_missing", False) and "brace" in msg and TT.program_check_stub()):
                raise
            log.warning("%s: program check asks for contacts.brace on a brace-less attachment (%s); recorded as stub-brace-optional", self.gp.variant, msg[:200])
            report = {"mode": "stub-brace-optional", "problems": [], "note": "brace-less attachment not yet accepted by the oracle program", "error": msg[:300]}
            self.last_program_check = report
            draft.summary["program_check"] = report
            return report

    # ------------------------------------------------------------------ §6.8 attach + brace from a stance, second base pose

    def _attach_and_brace(self, rng, manip: Stance, st: Stance, m_pose: TT.Pose, grasp, cam_init, require_joint: bool, brace: bool = True) -> Optional[Dict[str, Any]]:
        """Partner placement + attach pose + SLERP reorientation + left brace, all from stance ``st`` (the grasp stance
        ``manip`` or a second base pose; the male grasped at ``manip``). ``require_joint``: the partner cell must satisfy
        the A1 joint placement (brace point in the left contact band at ``st``'s trunk) and the brace two-arm IK must pass;
        with it False the brace is optional (recorded when it passes); ``brace`` False = no brace is even tried (the one-arm
        attach of ``generation.brace_search`` false). Returns the attach facts or None (reasons in ``self._att_fails[st.name]``)."""
        male, female, support = self.obj(self.t_role), self.obj(self.p_role), self.obj("support")
        fails = {"distance": 0, "support": 0, "gap": 0, "frame": 0, "brace_off_band": 0, "brace_class_failed": 0, "screen": 0, "ik": 0, "fingers": 0, "slerp": 0, "brace_ik": 0}
        self._att_fails = getattr(self, "_att_fails", {})
        tag = "" if require_joint else ("/brace-optional" if brace else "/no-brace")
        self._att_fails[f"{st.name}@lat{st.lateral:+.2f}/trunk({st.translate:.2f},{st.tilt:+.2f}){tag}"] = fails
        cells = TT.band_cells_world(self.attach_bands, st)
        if len(cells) == 0:
            fails["no_cells"] = 1
            return None
        m_fp = self.world.footprint(self.hull[self.t_role], m_pose)
        focus_xy = st.focus[:2] if st.focus is not None else m_pose[0][:2]
        perm = [int(i) for i in rng.permutation(len(cells))]
        perm.sort(key=lambda i: float(th.linalg.norm(cells[i] - focus_xy)))
        geo: List[Tuple[Any, ...]] = []
        for i in perm[:PARTNER_CELLS_TRIED]:
            xy = cells[int(i)]
            if float(th.linalg.norm(xy - m_pose[0][:2])) < float(self.g("partner_centre_distance_min_m")):
                fails["distance"] += 1
                continue
            pose = TT.object_pose_on(self, self.hull[self.p_role], C.IDENTITY_QUAT, xy, float(rng.uniform(-math.pi, math.pi)), self.top.z)
            fp = self.world.footprint(self.hull[self.p_role], pose)
            if not self.top.polygon_supported(fp, margin=TT.SUPPORT_MARGIN):
                fails["support"] += 1
                continue
            if TT._polygons_overlap(m_fp, fp, float(self.g("partner_gap_min_m"))):
                fails["gap"] += 1
                continue
            if not TT.in_frame(self, cam_init, TT.hull_corners_world(self.hull[self.p_role], pose), TT.IN_FRAME_MARGIN):
                fails["frame"] += 1
                continue
            hand_hit = self._hand_clear_of_partner(grasp, pose)
            if hand_hit is not None:
                fails["hand"] = fails.get("hand", 0) + 1
                fails.setdefault("hand_example", hand_hit)
                continue
            # the first brace face whose point lies in the left contact band at this trunk (class passing), else the first face
            jp = None
            for point, b_dir in self._partner_brace_points(pose, st):
                jp_ = TT.joint_placement(self.attach_bands, self.left_bands, st, xy, point[:2], brace_class=TT.contact_class(st, -b_dir, th.tensor([0.0, 0.0, 1.0])))
                if not jp_["ok"] and jp_["brace_ok"] and not jp_["brace_class_ok"]:
                    jp_ = TT.joint_placement(self.attach_bands, self.left_bands, st, xy, point[:2], brace_class=TT.contact_class(st, -b_dir, C.default_closing_axis(-b_dir)))
                if jp is None or (jp_["ok"] and not jp["ok"]) or (jp_["brace_ok"] and not jp["brace_ok"] and not jp["ok"]):
                    jp, point_best, b_dir_best = jp_, point, b_dir
                if jp["ok"]:
                    break
            point, b_dir = point_best, b_dir_best
            if require_joint and not jp["ok"]:
                fails["brace_off_band" if not jp["brace_ok"] else "brace_class_failed"] += 1
                continue
            goal = self._attached_pose(pose)
            eef_attach = IK.grasp_pose(goal, grasp.eef_in_object)
            axis = self._insertion_axis_world(pose)
            eef_pre = (eef_attach[0] + axis * float(self.g("pre_attach_back_m")), eef_attach[1])
            geo.append((pose, eef_pre, eef_attach, point, b_dir, jp.get("brace_classes"), bool(jp["ok"])))
        if not geo:
            return None
        with TT.posed_in_hand(self, "right", male, grasp.eef_in_object):
            with attached(self.kit, "right", male):
                ok = self.ikc.screen_pairs([[g_[1], g_[2]] for g_ in geo], st, "right", ignore_for_last=[support])
        screened = [g_ for g_, o in zip(geo, ok.tolist()) if o]
        fails["screen"] = len(geo) - len(screened)
        shift = th.tensor([float(st.base_xy[0] - manip.base_xy[0]), float(st.base_xy[1] - manip.base_xy[1]), 0.0])
        for pose, eef_pre, eef_attach, point, b_dir, brace_classes, joint_ok in screened[:PARTNER_EXACT_CAP]:
            self._check_deadline()
            female.set_position_orientation(*pose)
            female.keep_still()
            self.kit.sync_world()
            with TT.posed_in_hand(self, "right", male, grasp.eef_in_object):
                with attached(self.kit, "right", male):
                    res = self.ikc.exact([eef_pre, eef_attach], st, "right", ignore_for_last=[support, female], scale_key=self.choice.models[self.t_role].scale)
                if not res.ok:
                    fails["ik"] += 1
                    continue
                why = self._fingers_in_partner(res.q, eef_attach, support, male, female)
                if why is not None:
                    fails["fingers"] += 1
                    log.info("%s: partner cell rejected: %s", self.gp.variant, why)
                    continue
                with attached(self.kit, "right", male):
                    # the lift clears the table; the SLERP waypoints carry the male up to the pre-attach pose above the partner
                    plan = self._carry_to_pre_attach(grasp, eef_pre, pose, st, shift)
                    r_w, lift, wps, carry = plan.result, plan.lift, plan.waypoints, plan.carry
                    self._route_count("carry_checked")
                    if not r_w.ok:
                        fails["slerp"] += 1
                        self._route_count("carry_failed")
                        seq = [lift] + wps
                        log.info("%s: SLERP_FAILURE at %s (%s): %s | %s", self.gp.variant, st.name, plan.ladder, r_w.reason,
                                 "; ".join(f"wp{i}: z_top+{float(pz[0][2]) - float(self.top.z):.2f} m, base {[round(float(v), 2) for v in st.world_to_base(pz[0][:2]).tolist()]}, "
                                           f"approach z {float(C.approach_axis(pz[1])[2]):+.2f}" for i, pz in enumerate(seq)))
                        continue
                    brace_rec, brace_orient = None, None
                    if joint_ok and brace:
                        def eval_brace(p):
                            self._check_deadline()
                            res_b = TT.brace_second_arm(self, st, "right", [eef_pre, eef_attach], "left", list(p.chain), ignore_work_last=[support, female],
                                                        ignore_brace_last=[support, female], q_first_brace=p.q_first)
                            return bool(res_b.ok), res_b, None, str(res_b.reason)

                        # a class's two wrist turns by the orientation rule (owner 2026-09-27); the classes in their order
                        b_chosen, b_dec, _b_fails = O.choose_first(self.ikc, "left", self._brace_candidates(st, point, b_dir, brace_classes), key=lambda c: c[1],
                                                                   label=lambda c: c[2], chain=lambda c: [c[0], c[1]], evaluate=eval_brace,
                                                                   q_start=self.ikc.q_locked(st), stage="attachment brace")
                        if b_chosen is not None:
                            brace_rec = tuple(b_chosen.proposal.payload)
                            brace_orient = b_dec
                    if joint_ok and brace and brace_rec is None:
                        fails["brace_ik"] += 1
                        if require_joint:
                            continue
            return {"f_pose": pose, "attach": eef_attach, "pre_attach": eef_pre, "wps": wps, "carry": carry, "lift": lift, "brace": brace_rec, "b_dir": b_dir,
                    "point": point, "slerp_policy": r_w.reason, "brace_orient": brace_orient, "carry_over": plan.over}
        return None

    def _carry_to_pre_attach(self, grasp, eef_pre: TT.Pose, f_pose: TT.Pose, st: Stance, shift: th.Tensor) -> CarryPlan:
        """The carry of the attached male from its lifted grasp to the pre-attach, tracked as the oracle runs it (``Lift``, then
        ``Reorient`` through the recorded waypoints): the first carry whose chain passes, else the last one tried.

        Default: the lift ladder ``CARRY_HEIGHTS`` (owner item 2 / att_diag2: a 12 cm lift of a grasp at 0.8 m base-forward has no IK),
        each followed by the straight SLERP line of ``reorientation_waypoints`` + 1 poses to the pre-attach. ``generation.
        carry_over_partner_m`` (owner 2026-10-12 attachment#04 3b, the lid-stockpot variants: the straight line from a 5-12 cm lift
        clipped the stockpot's rim): the 12 cm lift, then straight up until the male's lowest point is that clearance above the
        partner's highest point (:meth:`_carry_rise_z`; a list is a ladder tried in order), then the line to the pre-attach; a carry
        whose fingertip point leaves the stance camera's image (``program_check.fingertips_in_frame``, the program check's rule) is
        not tried. The chain is TRACKED from the lift's solution at the oracle's densification (track_dense, as block design's
        reorients; owner 2026-09-19 20:42): 32-seed IK at every waypoint said "reachable" for a pre-attach 25 cm from the lift while
        the oracle's differential IK stalled 13.7 mm short at waypoint 24/29 (pwgbsb / jlalfc 0919-185350 / 193702) - a chain the
        arm cannot follow in one branch is rejected here, so the place stance (second base pose, brace False) is recorded instead."""
        tracked = bool(self.gen("slerp_tracked", True))
        n = int(self.g("reorientation_waypoints")) + 1
        support = self.obj("support")
        ladder = carry_over_ladder(self.gen("carry_over_partner_m", None))
        tries = [(max(CARRY_HEIGHTS), c) for c in ladder] if ladder else [(c, None) for c in CARRY_HEIGHTS]
        label = f"carry over the partner {list(ladder)} m" if ladder else f"carry ladder {CARRY_HEIGHTS}"
        plan = CarryPlan(None, None, [], None, None, label)
        for carry, clearance in tries:
            lift = (grasp.grasp[0] + th.tensor([0.0, 0.0, carry]) + shift, grasp.grasp[1])  # the carried male moves with the base
            start, over = lift, None
            if clearance is not None:
                rise_z, over = self._carry_rise_z(grasp, lift, f_pose, clearance)
                if rise_z > float(lift[0][2]) + 1e-3:
                    start = (th.tensor([float(lift[0][0]), float(lift[0][1]), rise_z]), TT._t(lift[1]))
            wps = ([start] if start is not lift else []) + [(TT._t(p[0]), TT._t(p[1])) for p in C.line_waypoints(start, eef_pre, n)]
            plan = CarryPlan(None, lift, wps, carry, over, label)
            if clearance is not None:
                out = self._carry_out_of_frame(st, wps)
                if out is not None:
                    plan.result = IK.IKResult(False, None, f"the carry {clearance:.2f} m over the partner leaves the image: {out}", len(wps))
                    continue
            plan.result = self.ikc.waypoints_ok([lift] + wps, st, "right", ignore_all=[support], track_dense=tracked)
            if plan.result.ok:
                break
        return plan

    def _carry_rise_z(self, grasp, lift: TT.Pose, f_pose: TT.Pose, clearance: float) -> Tuple[float, Dict[str, Any]]:
        """``(eef z, facts)``: the height of the eef, held at ``lift``'s orientation, at which the male in the hand
        (``grasp.eef_in_object``) has its lowest hull point ``clearance`` above the highest hull point of the partner at ``f_pose``."""
        m_pose = C.compose(lift, C.invert((TT._t(grasp.eef_in_object[0]), TT._t(grasp.eef_in_object[1]))))
        low = float(C.transform_points(m_pose, TT._t(self.hull[self.t_role]))[:, 2].min())
        top = float(C.transform_points((TT._t(f_pose[0]), TT._t(f_pose[1])), TT._t(self.hull[self.p_role]))[:, 2].max())
        z = float(lift[0][2]) + top + float(clearance) - low
        return z, {"clearance_m": round(float(clearance), 4), "partner_top_z": round(top, 4), "male_low_below_eef_m": round(float(lift[0][2]) - low, 4),
                   "rise_z": round(max(z, float(lift[0][2])), 4), "rise_m": round(max(0.0, z - float(lift[0][2])), 4)}

    def _carry_out_of_frame(self, st: Stance, wps: Sequence[TT.Pose]) -> Optional[Dict[str, Any]]:
        """The program check's frame rule on the carry's waypoints (``program_check.fingertips_in_frame``: the fingertip point inside the
        stance camera's image, 5 % + execution slack; off with ``ISG_FINGERTIP_FRAME_CHECK=off`` as there): the first pose outside, or None."""
        from ispatialgym.sampling.v2 import program_check as PC

        if os.environ.get("ISG_FINGERTIP_FRAME_CHECK", "on").lower() in ("off", "0", "false"):
            return None
        return PC.fingertips_in_frame(self, st, "reorient", [("waypoints", list(wps))])

    def _second_base_pose(self, rng, manip: Stance, m_pose: TT.Pose, grasp, cam_init, tname: str, require_joint: bool = True, brace: bool = True):
        """§6.8 (owner 2026-09-16 17:10): when the attach pose and the left brace are not both reachable from the grasp
        stance, a second base pose — the grasp stance shifted along the edge by ``SECOND_BASE_LATERALS`` with the trunk kept
        or moved to an arm-friendly grid trunk — from which the attach / brace facts are re-derived with the male carried
        (both arms' exact IK there); on free floor and whole-body collision-free. With ``brace`` False it is the one-arm
        PLACE STANCE of ``generation.brace_search`` false (the attach pose alone must be reachable). Returns ``(facts,
        stance)`` or ``(None, manip)``."""
        male = self.obj(self.t_role)
        tried, skipped = 0, {"no_floor": 0, "stance_collision": 0}
        what = "attach + brace" if brace else "the one-arm attach"
        for dlat, trunk, st in self._second_base_stances(manip):
            if tried >= SECOND_BASE_MAX_TRIES:
                log.info("%s: second base pose: none of %d stances reaches %s (%s)", self.gp.variant, tried, what, skipped)
                return None, manip
            self._check_deadline()
            st.focus = manip.focus
            st.arms = ["right"]
            if not self.kit.trav_free(st.base_xy):
                skipped["no_floor"] += 1
                continue
            male.set_position_orientation(*m_pose)
            male.keep_still()
            self.kit.sync_world()
            if not self.ikc.stance_collision_free(st):
                skipped["stance_collision"] += 1
                continue
            tried += 1
            att = self._attach_and_brace(rng, manip, st, m_pose, grasp, cam_init, require_joint=require_joint, brace=brace)
            if att is not None:
                log.info("%s: second base pose: lateral %+.2f, trunk (%.2f, %+.2f) reaches %s", self.gp.variant, dlat, trunk[0], trunk[1], what)
                return att, st
        log.info("%s: second base pose: none of %d stances reaches %s (%s)", self.gp.variant, tried, what, skipped)
        return None, manip

    @staticmethod
    def _second_base_stances(manip: Stance) -> Iterator[Tuple[float, Tuple[float, float], Stance]]:
        """The §6.8 second base poses in the order :meth:`_second_base_pose` tries them: ``(lateral shift, trunk, stance)``, the grasp
        stance shifted along the edge by ``SECOND_BASE_LATERALS`` with its trunk kept or moved to an arm-friendly grid trunk."""
        for dlat in SECOND_BASE_LATERALS:
            for trunk in ((manip.translate, manip.tilt), (0.5, 0.0), (0.25, -0.3)):
                yield dlat, trunk, manip.shifted(lateral=manip.lateral + dlat, name="attach").with_trunk(trunk[0], trunk[1], "attach")

    def _partner_brace_point(self, f_pose: TT.Pose, manip: Stance) -> Tuple[th.Tensor, th.Tensor]:
        """Brace face of the partner: the face opposite to the insertion direction (the face the insertion pushes
        toward) — the horizontal insertion direction when it has one (>= 30 % horizontal), else the partner's face
        toward the robot (the left gripper reaches forward); ``point`` = the partner's centre pushed out to the hull's
        extreme along ``b_dir`` at B3's contact height (0.75 of the partner's height). Returns ``(point, b_dir)``
        (``b_dir`` = outward normal of the braced face; the approach is −b_dir)."""
        return self._partner_brace_points(f_pose, manip)[0]

    def _partner_brace_points(self, f_pose: TT.Pose, manip: Stance) -> List[Tuple[th.Tensor, th.Tensor]]:
        """Brace face candidates in preference order: the face opposite a horizontal insertion (the face the insertion
        pushes toward) alone; for a vertical insertion the partner's face toward the robot, then its robot-left face,
        then its far face (att_diag 20:12: with the near face alone 22 of 40 partner cells had the brace point off the
        left band — the attach-reach cells lie on the robot's right, the left band's lateral range is [-0.16, 0.56])."""
        axis = self._insertion_axis_world(f_pose)
        d_ins = -axis  # the male moves along -axis into the female
        horiz = th.tensor([float(d_ins[0]), float(d_ins[1]), 0.0])
        f = manip.forward
        fwd = th.tensor([float(f[0]), float(f[1]), 0.0])
        left = th.tensor([-float(f[1]), float(f[0]), 0.0])
        dirs = [horiz / th.linalg.norm(horiz)] if float(th.linalg.norm(horiz)) >= 0.3 else [-fwd, left, fwd]
        pts = C.transform_points(f_pose, self.hull[self.p_role])
        z_c = TT.contact_height(float(pts[:, 2].min()), float(pts[:, 2].max() - pts[:, 2].min()))
        centre = pts.mean(dim=0)
        out = []
        for b_dir in dirs:
            proj = pts[:, :2] @ b_dir[:2]
            point = th.tensor([float(centre[0]), float(centre[1]), z_c]) + b_dir * (float(proj.max()) - float(centre[:2] @ b_dir[:2]))
            out.append((point, b_dir))
        return out

    def _brace_candidates(self, manip: Stance, point: th.Tensor, b_dir: th.Tensor, passing: Optional[List[str]]):
        """Brace poses of the closed left gripper on the partner face at ``point``: approach −b_dir into the face
        (pad centre on the point, ``contact_eef``), closing axis vertical (``_v``) then horizontal (``_h``), either wrist
        turn; the contact classes the left band recorded as passing at the cell first. Yields ``(pre, contact, kind)``."""
        z_axis = th.tensor([0.0, 0.0, 1.0])
        a = -b_dir
        side_h = C.unit(th.cross(z_axis, b_dir, dim=0))
        classes = []
        for c_axis, suffix in ((z_axis, "v"), (side_h, "h")):
            name = TT.contact_class(manip, a, c_axis)
            classes.append((name or f"side_{suffix}", c_axis))
        if passing:
            classes.sort(key=lambda nc: 0 if nc[0] in passing else 1)
        for name, c_axis in classes:
            for c_sign in (1.0, -1.0):
                brace = C.contact_eef(point, a, c_sign * c_axis)
                yield C.pre_pose(brace), brace, f"{name}{'' if c_sign > 0 else '-'}"

    def _group_name(self, names: List[str]) -> str:
        if self.problem is not None:
            for g, members in self.problem.groups.items():
                if set(members) >= set(names):
                    return g
        from ispatialgym.sampling.bddl_writer import GROUP_NAMES

        return GROUP_NAMES["attachment"]

    def _group_target(self, f_pose: TT.Pose, m_goal: TT.Pose, tname: str, pname: str) -> Dict[str, Any]:
        rel_p, rel_q = C.relative(m_goal, f_pose)
        return {
            "members": [tname, pname],
            "assignment": {"mode": "fixed", "member_to_slot": {tname: "male", pname: "female"}},
            "orientation_equivalences": self.symmetry,
            "target_slots": {"male": {"position": [float(v) for v in m_goal[0]], "orientation": [float(v) for v in m_goal[1]]},
                             "female": {"position": [float(v) for v in f_pose[0]], "orientation": [float(v) for v in f_pose[1]]}},
            "pose_tolerances": {"position": self.tol("position", 0.01), "orientation": self.tol("orientation", 0.15)},
            "layout_relations": [{"from": "female", "to": "male", "position": [float(v) for v in rel_p], "orientation": [float(v) for v in rel_q]}],
            "layout_tolerances": {"position": self.tol("position", 0.01), "orientation": self.tol("orientation", 0.15)},
        }

    # ------------------------------------------------------------------ goal / hooks (kept from v1 §5.6)

    def goal(self, draft: Draft):
        manip = self.sidecar.stance("manipulation")
        self.sidecar.set_goal(manip.base_pose2d, manip.translate, manip.tilt, arms="same")

    def check_rest_stability(self, draft: Draft) -> None:
        """Nudge test of the parts' rest poses (cork/ncxgpe 0920-200850): the settled male / partner is tilted by
        REST_NUDGE_DEG about a horizontal axis and left to physics for REST_STABILITY_STEPS with the robot held; its up axis
        must come back within REST_TIP_MAX_DEG of the settled one, else ``Reject("rest_unstable")``. The 90-step init settle
        only checks velocities: the cork stood on its tapered mating end (collision profile: radius 5.2-6.5 mm across a 3 mm
        band, a convex end under a 1.7 cm high COM), sat still through the settle, and in the oracle rocked 2 -> 6 -> 12 ->
        95 deg over the 130 steps of the transition - the demo then aimed at a cork lying 26 mm away (the "IK 235 mm off").
        The world is restored afterwards (the recorded poses are the settled ones)."""
        roles = [r for r in (self.t_role, self.p_role) if self.obj(r) is not None]
        up = th.tensor([0.0, 0.0, 1.0])
        state = self.world.snapshot()
        tilts: Dict[str, float] = {}
        try:
            settled = {r: tuple(x.clone() for x in self.obj(r).get_position_orientation()) for r in roles}
            for r in roles:
                p, q = settled[r]
                q_n = C.qmul(C.quat_from_axis_angle(th.tensor([1.0, 0.0, 0.0]), math.radians(REST_NUDGE_DEG)), q)
                self.obj(r).set_position_orientation(p + th.tensor([0.0, 0.0, 0.001]), q_n)
                self.obj(r).keep_still()
            self.world.wake([self.obj(r) for r in roles])
            for _ in range(REST_STABILITY_STEPS):
                og.sim.step_physics()
                self.robot.keep_still()
            for r in roles:
                _, q_now = self.obj(r).get_position_orientation()
                c = float(th.clamp(th.dot(C.quat_rotate(settled[r][1], up), C.quat_rotate(q_now.float(), up)), -1.0, 1.0))
                tilts[self.obj(r).name] = round(math.degrees(math.acos(c)), 2)
        finally:
            self.world.restore(state)
            self.world.keep_still(self.objects.values())
        draft.summary["rest_tilt_after_nudge_deg"] = tilts
        bad = {n: d for n, d in tilts.items() if d > REST_TIP_MAX_DEG}
        self._route_count("rest_unstable" if bad else "rest_stable")
        if bad:
            self.counters["rest_unstable"] = self.counters.get("rest_unstable", 0) + 1
            male = self.obj(self.t_role)
            if male is not None and male.name in bad and not self._rest_flip_forced and not (self.template_facts.get("male_rest_orientation") or {}).get("flipped"):
                # owner 2026-09-25: the male toppled on its library end - the next draws may turn it onto the other end when that keeps
                # the insertion direction and library grasps (stable_rest_quat(force=True)); otherwise the rejection stands as it is
                self._rest_flip_forced = True
                log.info("%s: %s toppled on its library rest end (%.1f deg) - the next draw considers the other end", self.gp.variant, male.name, bad[male.name])
            raise Reject("rest_unstable", f"up axis {bad} deg off the settled pose {REST_STABILITY_STEPS} steps after a {REST_NUDGE_DEG:.0f} deg nudge "
                                          f"(> {REST_TIP_MAX_DEG:.0f}): the part does not stand on that end")
        log.info("%s: rest stability after a %.0f deg nudge: %s deg", self.gp.variant, REST_NUDGE_DEG, tilts)

    def after_settle(self, draft: Draft):
        tname, pname = self.tname, self.pname
        self.check_rest_stability(draft)
        f_pose = draft.object_poses[pname]
        goal = self._attached_pose(f_pose)
        draft.goal_poses[tname] = goal
        draft.groups[self._group_name([tname, pname])] = self._group_target(f_pose, goal, tname, pname)

    def apply_goal(self, draft: Draft):
        male, female = self.obj(self.t_role), self.obj(self.p_role)
        # the seated part overlaps the partner's convex hulls (attach step 0): the pair's collision is filtered before it is put
        # in its seat, else the next physics step pushes it out
        PRED.set_pair_collision_filter(male, female, True)
        super().apply_goal(draft)
        og.sim.step_physics()
        if not male.states[AttachedTo].set_value(female, True):
            raise Reject("goal_infeasible", "AttachedTo.set_value failed at the computed attach pose")

    def refine_targets_from_goal(self, draft: Draft):
        tname, pname = self.tname, self.pname
        m_pose = tuple(x.clone() for x in self.obj(self.t_role).get_position_orientation())
        f_pose = tuple(x.clone() for x in self.obj(self.p_role).get_position_orientation())
        if float(th.linalg.norm(f_pose[0] - draft.object_poses[pname][0])) > float(self.gen("partner_moved_max_m", 0.005)):
            raise Reject("goal_partner_moved", "the partner moved while attaching")
        draft.groups[self._group_name([tname, pname])] = self._group_target(f_pose, m_pose, tname, pname)

    def check_goal_feasible(self, draft: Draft):
        try:
            super().check_goal_feasible(draft)
        finally:
            male, female = self.obj(self.t_role), self.obj(self.p_role)
            if male.states[AttachedTo].get_value(female):
                male.states[AttachedTo].set_value(female, False)
            # the snapshot restore has put the part back at its init pose: the pair collides again
            PRED.set_pair_collision_filter(male, female, False)
            og.sim.step_physics()
            self.world.keep_still(self.objects.values())

    def after_goal_check(self, draft: Draft):
        """The attach / pre-attach at the settled partner pose (male in hand), from the stance the attach was recorded at: the
        second base pose when the attach needed one (``grasp.extras["attach_stance"]``), else the manipulation stance."""
        tname, pname = self.tname, self.pname
        if self.attach_route == "handover":
            return self._after_goal_check_handover(draft)
        grasp = self.sidecar.grasp_of(tname)
        manip = self.sidecar.stance("manipulation")
        manip = self.sidecar.stances.get(str(grasp.extras.get("attach_stance") or "manipulation")) or manip
        f_pose = draft.object_poses[pname]
        goal = self._attached_pose(f_pose)
        attach = IK.grasp_pose(goal, grasp.eef_in_object)
        pre = (attach[0] + self._insertion_axis_world(f_pose) * float(self.g("pre_attach_back_m")), attach[1])
        male, female, support = self.obj(self.t_role), self.obj(self.p_role), self.obj("support")
        with TT.posed_in_hand(self, "right", male, grasp.eef_in_object):
            with attached(self.kit, "right", male):
                res = self.ikc.exact([pre, attach], manip, "right", ignore_for_last=[support, female], scale_key=self.choice.models[self.t_role].scale)
            why = self._fingers_in_partner(res.q, attach, support, male, female) if res.ok else None
        if not res.ok:
            raise Reject("place_unreachable", f"settled partner: pre-attach / attach fail the exact IK: {res.reason}")
        if why is not None:
            self.counters["fingers_in_partner"] = self.counters.get("fingers_in_partner", 0) + 1
            raise Reject("fingers_in_partner", f"settled partner: {why}; attach clearance {grasp.extras.get('attach_clearance')}")
        # the oracle's attach correction regime (owner 2026-09-20 23:39 item 1): ``generation.attach_correction`` = {soft_pos_tol_m,
        # settle_steps, max_passes, stop_when_improvement_below_m, target_pos_m, target_rot_rad} travels in the sidecar; absent = the
        # oracle's own defaults (execute.AttachCorrection)
        regime = self.gen("attach_correction", None)
        if isinstance(regime, dict) and regime:
            self.sidecar.extra["attach_correction"] = {k: float(v) if k != "max_passes" and k != "settle_steps" else int(v) for k, v in regime.items()}
        grasp.pre_place, grasp.place = pre, attach
        if grasp.waypoints:
            grasp.waypoints[-1] = pre

    def _after_goal_check_handover(self, draft: Draft) -> None:
        """:meth:`after_goal_check` of the hand-over: the LEFT hand's pre-attach / attach at the settled partner pose from the attach stance
        (the male attached to the left hand), the fingers clear of the partner; the hold record's pre_place / place / last waypoint."""
        tname, pname = self.tname, self.pname
        hold = self.sidecar.grasp_of(tname, arm="left")
        st = self.sidecar.stances.get(str(hold.extras.get("attach_stance") or "manipulation")) or self.sidecar.stance("manipulation")
        f_pose = draft.object_poses[pname]
        attach = IK.grasp_pose(self._attached_pose(f_pose), hold.eef_in_object)
        pre = (attach[0] + self._insertion_axis_world(f_pose) * float(self.g("pre_attach_back_m")), attach[1])
        male, female, support = self.obj(self.t_role), self.obj(self.p_role), self.obj("support")
        with TT.posed_in_hand(self, "left", male, hold.eef_in_object):
            with attached(self.kit, "left", male):
                res = self.ikc.exact([pre, attach], st, "left", ignore_for_last=[support, female], scale_key=self.choice.models[self.t_role].scale)
            clear = self.ikc.collision_free_at(res.q, attach, "left", ignore=[support, male]) if res.ok else True
        if not res.ok:
            raise Reject("place_unreachable", f"settled partner: the left hand's pre-attach / attach fail the exact IK: {res.reason}")
        if not clear:
            self.counters["fingers_in_partner"] = self.counters.get("fingers_in_partner", 0) + 1
            raise Reject("fingers_in_partner", "settled partner: the left hand's links collide with the partner at the attach (male detached)")
        regime = self.gen("attach_correction", None)
        if isinstance(regime, dict) and regime:
            self.sidecar.extra["attach_correction"] = {k: float(v) if k != "max_passes" and k != "settle_steps" else int(v) for k, v in regime.items()}
        hold.pre_place, hold.place = pre, attach
        if hold.waypoints:
            hold.waypoints[-1] = pre

    def turn_recheck(self, draft: Draft, grasp, q_first: Optional[th.Tensor] = None) -> Optional[str]:
        """``tabletop.held_turns``: this module's checks that depend on the wrist turn, for ``grasp`` just recorded as its other finger
        assignment - the pick as the draw solved it (support ignored at the grasp; from ``q_first``, the recorded pre-grasp witness
        wrist-flipped, else afresh) with the wrist camera clear of the support top and of the world, and the over-the-top standoff
        when one is recorded; a second base pose's carried path / transition / stance swept again with this turn in the hand; from the
        attach stance the lift + SLERP carry tracked as ``_attach_and_brace`` tracks it, the pre-attach -> attach, the left brace
        solved again with the right arm locked at this attach (``tabletop.brace_second_arm``) when one is recorded, and the fingers
        clear of the partner at the attach (``_fingers_in_partner``). None when every check passes, else the first failure."""
        if self.attach_route == "handover":
            # i7-cork: the hand-over's two pinches are solved together (the hands' distance, both arms locked in turn): no wrist turn
            raise TT.TurnNotOffered("the hand-over route: the two hands' pinches are checked together")
        male, female, support = self.obj(self.t_role), self.obj(self.p_role), self.obj("support")
        manip = self.sidecar.stance("manipulation")
        st = self.sidecar.stances.get(str(grasp.extras.get("attach_stance") or "manipulation")) or manip
        scale = self.choice.models[self.t_role].scale
        kw = {"ignore_for_last": [support], "scale_key": scale, "cache": False}
        with TT.record_policy_opening(self, grasp):  # of-v2maps 2026-11-07: the policy at the grasp's pre-shape
            res = self.ikc.exact([grasp.pre_grasp, grasp.grasp], manip, "right", q_first=q_first, **kw)
            if not res.ok and q_first is not None:
                res = self.ikc.exact([grasp.pre_grasp, grasp.grasp], manip, "right", **kw)
        if not res.ok:
            return f"pick: {res.reason}"
        if TT.wrist_camera_clearance(grasp.grasp, float(self.top.z)) < TT.TWIN_CAMERA_CLEARANCE_M or not O.camera_clear(self.ikc, res.q, "right"):
            return "pick: the wrist camera on the support top / within 5 mm of the world"
        if grasp.extras.get("approach_standoff_m"):
            standoff = C.pre_pose((TT._t(grasp.pre_grasp[0]), TT._t(grasp.pre_grasp[1])), float(grasp.extras["approach_standoff_m"]))
            r = self.ikc.exact([standoff], manip, "right", ignore_for_last=[support], scale_key=scale, cache=False)
            if not r.ok:
                return f"approach standoff: {r.reason}"
        if st is not manip:
            why = self.recheck_carried_travel(self.tname)
            if why:
                return f"second base pose: {why}"
        brace = self.sidecar.contacts.get("brace")
        lift = grasp.extras.get("lift")
        with TT.posed_in_hand(self, "right", male, grasp.eef_in_object):
            with attached(self.kit, "right", male):
                if lift is not None and grasp.waypoints:
                    r = self.ikc.waypoints_ok([(th.as_tensor(lift, dtype=th.float32), TT._t(grasp.grasp[1]))] + list(grasp.waypoints), st, "right",
                                              ignore_all=[support], track_dense=bool(self.gen("slerp_tracked", True)))
                    if not r.ok:
                        return f"lift + SLERP: {r.reason}"
                r = self.ikc.exact([grasp.pre_place, grasp.place], st, "right", ignore_for_last=[support, female], scale_key=scale, cache=False)
                if not r.ok:
                    return f"attach: {r.reason}"
                if brace is not None:
                    r_b = TT.brace_second_arm(self, st, "right", [grasp.pre_place, grasp.place], "left", [brace.pre, brace.contact],
                                              ignore_work_last=[support, female], ignore_brace_last=[support, female])
                    if not r_b.ok:
                        return f"brace with the right arm at this attach: {r_b.reason}"
            why = self._fingers_in_partner(r.q, grasp.place, support, male, female)
        return f"attach: {why}" if why else None
