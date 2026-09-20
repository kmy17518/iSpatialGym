"""Configuration matching — attachment (spec §6.8; init per §4 with the attachment bullets of §4.4 / §4.6).

Roles: ``target`` (the male part, grasped with the attachment pipeline's library grasps), ``partner`` (the female
part, upright), scene ``support``, ``floor``; the v1 facade names them ``male`` / ``female``.

Template: both parts get the ``attachable`` ability at spawn; the male / female meta-link pair (``AttachedTo``
parent candidates), the collision filter between the two members (applied once, restored by the loader), the
mating-axis symmetry test (5 mm slabs, a rotation is accepted when >= 90 % of the slabs match) -> ``orientation_
equivalences`` for both members; two reach bands — the male's grasp band and the **attach-reach band** of the partner
(the partner posed at a cell, grasp set = the attach eef poses ``T_partner ∘ T_F ∘ inv(T_M) ∘ T_obj→eef`` of every male
grasp, so a cell is in it when every attachment pose is reachable ignoring the target); Phases 1a / 1b (H_max = the
taller part) / 1c (row ``attachment``: max(w_target, w_partner) + 0.02) / 2 (male half-extent).

Instance: the male at the focus from F (Phase 3 step 1); the partner on a cell of the attach-reach band (gap >= 0.04,
centre distance >= 0.08, on the top, in frame); the attach pose ``T_partner ∘ T_F ∘ inv(T_M)`` and the **pre-attach**
10 cm back along the female meta-link's insertion axis, exact-IK-checked with the male in hand (partner and support
ignored for the attach pose only); the in-hand reorientation from the lifted grasp pose to the pre-attach pose at 3
SLERP waypoints (§3.9); the **partner brace** (closed left gripper on the partner's face opposite to the insertion
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
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import PAD_BAND_MARGIN, SamplerV2, register
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance, Transition

import omnigibson as og  # noqa: E402
from omnigibson.object_states import AttachedTo  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASK = "configuration_matching-attachment"
#: per-attempt budget (owner 2026-09-16 10:47 item 3): the pilot's cap-lemonade attempt spent 1457 s in the per-cell exact
#: IK with the attached male; a candidate now screens every admissible partner cell in one batch and solves exactly at most
#: PARTNER_EXACT_CAP of them, the attempt stops at ATTEMPT_BUDGET_S (``generation.attempt_budget_s`` overrides)
ATTEMPT_BUDGET_S = 360.0  # the second base pose search (owner 17:10) adds up to SECOND_BASE_MAX_TRIES attach evaluations
PARTNER_EXACT_CAP = 6
#: §6.8 second base pose: lateral shifts of the grasp stance along the edge (m, + = robot's left) tried in this order
SECOND_BASE_LATERALS = (0.10, -0.10, 0.20, -0.20, 0.30, -0.30)
SECOND_BASE_MAX_TRIES = 6
FOCUS_DEPTH_PREFERRED = 0.62  # base-forward depth of the manipulation focus beyond which the lift / SLERP fail (att_diag2 21:04)
GOOD_TRUNKS = ((0.5, -0.45), (0.5, 0.0), (0.25, -0.3))  # (translate, tilt) verified for pick + pre-poses by the block probes
CARRY_HEIGHTS = (0.12, 0.08, 0.05)  # the lift above the grasp: the first whose lift + SLERP pass the IK


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

    def g(self, key: str):
        return self.gen(key, self.defaults.get(key))

    # ------------------------------------------------------------------ roles

    def v1_role_aliases(self) -> Dict[str, str]:
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
            raise Reject("spec", f"{self.gp.alias}: scenes.{self.scene}.bindings.support must list scene object names")
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
        self.rest[self.p_role] = C.IDENTITY_QUAT.clone()
        for role, o in ((self.t_role, male), (self.p_role, female)):
            self.hull[role] = self.world.local_hull(o)
            self.height[role] = TT.height_of(self.hull[role], self.rest[role])
        # collision filter between the members (OmniGibson's AttachedTo no longer filters child / parent collisions)
        male.states[AttachedTo]._disable_collision_between_child_and_parent(male, female)
        og.sim.step_physics()
        self.kit.sync_world()
        cp, cq = child_link.get_position_orientation()
        rp, rq = male.get_position_orientation()
        self.root_in_child_link = C.relative((rp.float(), rq.float()), (cp.float(), cq.float()))
        fp, fq = female.get_position_orientation()
        pp, pq = parent_link.get_position_orientation()
        self.parent_link_in_female = C.relative((pp.float(), pq.float()), (fp.float(), fq.float()))
        # insertion axis (female meta link +z) in the female root frame
        self.insertion_axis_local = C.quat_rotate(self.parent_link_in_female[1], th.tensor([0.0, 0.0, 1.0]))
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
        for edge, st in TT.edge_candidates(self, self.top, rng):
            try:
                bands = TT.ensure_bands(self, self.band_spec, top=self.top, edge=edge)
                try:
                    self.attach_bands = TT.RB.ReachBands.load(self.world.scene_model, self.top.name, attach_spec.object, attach_spec.grasp_key, placed=attach_spec.placed, scale=attach_spec.scale)
                except FileNotFoundError:
                    self.attach_bands = TT.compute_bands(self, attach_spec, self.top, edge)
                # A1: the left contact band of the partner's height class (partner brace face); §4.4 margins over the union
                self.left_bands = TT.load_left_contact_band(self, self.height[self.p_role], top=self.top, edge=edge, compute=TT.left_band_compute())
                init = TT.phase1b_init_trunk(self, st, H_max, bands=bands, top=self.top, left_bands=self.left_bands)
                init = TT.phase1c_lateral(self, init, margins, rng, top=self.top, bands=bands)
                init.name = "init"
                # D-009 (prepared, owner 2026-09-16 21:54; defaults = today's behaviour): A ``generation.phase2_loose_class_deg``
                # (loose class 20-25 deg for attachment) / B ``generation.phase2_class1_when_deep_m`` (class-1 survivors admitted
                # when every class-0 focus is deeper than that, base-forward: att_diag3 21:30 - all 15 class-0 foci at
                # 0.77-0.81 m, the lift / SLERP fail there)
                deep = self.gen("phase2_class1_when_deep_m", None)
                cands = TT.phase2_manipulation_stance(self, init, TT.half_extent(self.hull[self.t_role], self.rest[self.t_role]), target_height=self.height[self.t_role],
                                                      bands=bands, top=self.top,
                                                      angle_classes_deg=(float(self.gen("phase2_tight_class_deg", 8.0)), float(self.gen("phase2_loose_class_deg", 20.0))),
                                                      class1_when_deep_m=None if deep is None else float(deep))
                cands = [c for c in cands if len(TT.band_cells_world(self.attach_bands, c)) > 0]
                if not cands:
                    raise Reject("no_manipulation_stance", "Phase 2 left no survivor with attach-reach cells")
                self.edge_frame, self.init_stance, self.manip_candidates = edge, init, cands
                break
            except Reject as e:
                errors.append(f"edge {edge.name}: {e}")
                log.info("%s: %s", self.gp.alias, errors[-1][:300])
        if self.init_stance is None:
            raise Reject("no_stance", "; ".join(errors)[:1500])
        self.template_facts.update({"support": self.top.to_json(), "edge": self.edge_frame.to_json(), "init_stance": self.init_stance.to_meta(),
                                    "n_manipulation_candidates": len(self.manip_candidates), "grasp_key": self.grasp_key, "n_grasps": len(self.grasps),
                                    "attachment_links": [child_link.body_name, parent_link.body_name], "n_symmetry_equivalences": len(self.symmetry),
                                    "band": getattr(self.bands, "path", None), "attach_band": getattr(self.attach_bands, "path", None)})

    def _attachment_links(self, male, female):
        cands = male.states[AttachedTo]._get_parent_candidates(female)
        if not cands:
            return None
        for child_name, parent_names in cands.items():
            for parent_name in parent_names:
                return male.states[AttachedTo].links[child_name], female.states[AttachedTo].links[parent_name]
        return None

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

    # ------------------------------------------------------------------ geometry

    def _attached_pose(self, female_pose: TT.Pose) -> TT.Pose:
        """Male root pose when its attachment link coincides with the female's (female at ``female_pose``)."""
        parent_link_world = C.compose(female_pose, self.parent_link_in_female)
        return C.compose(parent_link_world, self.root_in_child_link)

    def _insertion_axis_world(self, female_pose: TT.Pose) -> th.Tensor:
        return C.unit(C.quat_rotate(TT._t(female_pose[1]), self.insertion_axis_local))

    # ------------------------------------------------------------------ instance

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        init = self.init_stance
        errors = []
        self._deadline = time.time() + float(self.gen("attempt_budget_s", ATTEMPT_BUDGET_S))
        # candidates with the focus at an arm-friendly depth first (att_diag2 21:04: the male at 0.81 m base-forward grasps
        # but its 12 cm lift fails the IK at every candidate; the block probes pass picks + pre-poses up to 0.65-0.70 m),
        # ties by the draw
        order = [int(k) for k in rng.permutation(len(self.manip_candidates))]

        def score(k: int) -> float:
            c = self.manip_candidates[k]
            depth = float(c.world_to_base(c.focus[:2])[0]) if c.focus is not None else FOCUS_DEPTH_PREFERRED
            trunk_pen = min(abs(float(c.translate) - t0) + abs(float(c.tilt) - i0) for t0, i0 in GOOD_TRUNKS)
            return max(0.0, depth - FOCUS_DEPTH_PREFERRED) + 0.3 * trunk_pen

        order.sort(key=score)
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
                return self._sample_with(rng, init, manip, int(k))
            except Reject as e:
                errors.append(f"candidate {int(k)}: {e}")
                log.info("%s: %s", self.gp.alias, errors[-1][:300])
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
        self.record_transition(init, manip)
        cam_init = self.camera_pose_at(init)
        # step 1: the male at the focus from F
        m_pose, grasp = TT.phase3_target(self, manip, rng, self.grasps, self.hull[self.t_role], self.rest[self.t_role], [support], object_name=tname,
                                         scale_key=self.choice.models[self.t_role].scale, bands=self.bands)
        grasp.place_on = pname
        male.set_position_orientation(*m_pose)
        male.keep_still()
        self.kit.sync_world()
        self._apply_grasp_height_extra(grasp, m_pose, manip)  # generation.grasp_height_extra_m (owner 2026-09-19 16:49; 0 = no-op)
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
                log.info("%s: BRACE_FALLBACK no joint placement from the grasp stance or a second base pose: %s", self.gp.alias, json.dumps(self._att_fails, default=str))
            att = self._attach_and_brace(rng, manip, manip, m_pose, grasp, cam_init, require_joint=False, brace=brace_search)
            work_st, brace_state = manip, "none"
            if att is None and not brace_search:
                att, work_st = self._second_base_pose(rng, manip, m_pose, grasp, cam_init, tname, require_joint=False, brace=False)
            if att is None:
                raise Reject("partner_placement", f"no partner cell with a reachable attach pose from the grasp stance or a second base pose ({self._att_fails})")
            if att.get("brace") is not None:
                brace_state = "present"
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
        for nm, p, cam in [("grasp", grasp.grasp, cam_manip), ("attach", attach, cam_work)] + ([("brace", brace, cam_work)] if brace is not None else []):
            if not self.ikc.eef_in_frame(p, cam):
                raise Reject("gripper_not_in_frame", f"{nm} eef pose outside the {'manipulation' if cam is cam_manip else 'attach'} image (5 %)")
        objects = {tname: m_pose, pname: f_pose}
        TT.whole_body_check_at(self, [init, manip] + ([work_st] if work_st is not manip else []), objects, require=True, reason="stance_collision")
        # record
        grasp.carry_height = float(carry)
        grasp.waypoints = wps
        grasp.pre_place, grasp.place = pre_attach, attach
        grasp.extras.update({"attach_stance": work_st.name, "lift": [float(v) for v in lift[0]] if lift is not None else None})
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
            self.sidecar.add_contact(TT.BraceRecord("brace", "left", brace_pre, brace, direction=brace_dir, brace_kind="partner_face", object=pname,
                                                    stance=work_st.name, extras={"pose_class": brace_kind, "stop_clearance_m": 0.0}))
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
        return d

    # ------------------------------------------------------------------ grasp height margin (owner 2026-09-19 16:49)

    def _apply_grasp_height_extra(self, grasp, m_pose: TT.Pose, manip: Stance) -> None:
        """``generation.grasp_height_extra_m``: the knob pinch moves up the male by up to that much (0 = today's pose). Demo
        0919-153953 (lid tnwisp): the pads held the top centimetre of the knob while the fingertips reached 0.7 cm below the
        pads into the lid's dome, so the closure pressed the lid onto the table and the torso took the reaction (head camera
        +3.5 mm at the closure; the table itself moved 0.02 mm). The generic ``contact_height_extra_m`` rule clips such a
        grasp to zero (its pad centre already lies above the pinch band), so the attachment rule is on the pad's
        fingertip-side end instead: it stays >= ``PAD_BAND_MARGIN`` below the male's top, and the lifted pre-grasp / grasp
        must pass the exact IK at the grasp stance (else the library height is kept). Every derived pose (attach = goal
        ∘ T_obj→eef, the pre-attach, the SLERP) carries the lift through ``eef_in_object``. Facts on the record."""
        x = max(0.0, float(self.gen("grasp_height_extra_m", 0.0) or 0.0))
        if x <= 0.0 or grasp.eef_in_object is None or grasp.grasp is None:
            grasp.extras.update({"grasp_height_extra_m": x, "grasp_height_extra_effective_m": 0.0, "grasp_height_extra_reason": "off" if x <= 0 else "no grasp pose"})
            return
        pos, quat = TT._t(grasp.grasp[0]), TT._t(grasp.grasp[1])
        a = C.approach_axis(quat)
        pad_tip_end_z = float((pos + C.PAD_Z_RANGE[1] * a)[2])  # the fingertip-side end of the pads
        top_z = float(self.world.hull_at(self.hull[self.t_role], m_pose)[:, 2].max())
        room = (top_z - PAD_BAND_MARGIN) - pad_tip_end_z
        eff = max(0.0, min(x, room))
        why = "full" if eff >= x - 1e-9 else ("clipped: pad end at the male's top" if eff > 0 else "clipped: pads already at the male's top")
        if eff > 0.0:
            R_obj = C.quat_to_mat(TT._t(m_pose[1]))
            up_obj = R_obj.T @ th.tensor([0.0, 0.0, 1.0])  # world up in the male's frame
            eef_in_object = (TT._t(grasp.eef_in_object[0]) + eff * up_obj, TT._t(grasp.eef_in_object[1]))
            eef = IK.grasp_pose(m_pose, eef_in_object)
            pre = IK.pre_pose(eef)
            support = self.obj("support")
            res = self.ikc.exact([pre, eef], manip, grasp.arm, ignore_for_last=[support], scale_key=self.choice.models[self.t_role].scale, cache=False,
                                 approached=[self.obj(self.t_role)])
            if res.ok:
                grasp.eef_in_object, grasp.grasp, grasp.pre_grasp = eef_in_object, eef, pre
            else:
                eff, why = 0.0, f"IK failed with the lift ({res.reason}); library height kept"
        log.info("%s: grasp height extra %.1f mm requested -> %.1f mm (%s; pad end %.3f, male top %.3f)", self.gp.alias, x * 1000, eff * 1000, why, pad_tip_end_z, top_z)
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
            log.warning("%s: program check asks for contacts.brace on a brace-less attachment (%s); recorded as stub-brace-optional", self.gp.alias, msg[:200])
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
        fails = {"distance": 0, "support": 0, "gap": 0, "frame": 0, "brace_off_band": 0, "brace_class_failed": 0, "screen": 0, "ik": 0, "slerp": 0, "brace_ik": 0}
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
        for i in perm[:40]:
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
                    # the lift clears the table; the SLERP waypoints carry the male up to the pre-attach pose above the partner. The
                    # carry height ladder (owner item 2 / att_diag2: a 12 cm lift of a grasp at 0.8 m base-forward has no IK)
                    r_w, lift, wps = None, None, []
                    for carry in CARRY_HEIGHTS:
                        lift = (grasp.grasp[0] + th.tensor([0.0, 0.0, carry]) + shift, grasp.grasp[1])  # the carried male moves with the base
                        wps = [(TT._t(p[0]), TT._t(p[1])) for p in C.line_waypoints(lift, eef_pre, int(self.g("reorientation_waypoints")) + 1)]
                        r_w = self.ikc.waypoints_ok([lift] + wps, st, "right", ignore_all=[support])
                        if r_w.ok:
                            break
                    if not r_w.ok:
                        fails["slerp"] += 1
                        seq = [lift] + wps
                        log.info("%s: SLERP_FAILURE at %s (carry ladder %s): %s | %s", self.gp.alias, st.name, CARRY_HEIGHTS, r_w.reason,
                                 "; ".join(f"wp{i}: z_top+{float(pz[0][2]) - float(self.top.z):.2f} m, base {[round(float(v), 2) for v in st.world_to_base(pz[0][:2]).tolist()]}, "
                                           f"approach z {float(C.approach_axis(pz[1])[2]):+.2f}" for i, pz in enumerate(seq)))
                        continue
                    brace_rec = None
                    if joint_ok and brace:
                        for pre_c, c_pose, kind in self._brace_candidates(st, point, b_dir, brace_classes):
                            self._check_deadline()
                            res_b = TT.brace_second_arm(self, st, "right", [eef_pre, eef_attach], "left", [pre_c, c_pose], ignore_work_last=[support, female],
                                                        ignore_brace_last=[support, female])
                            if res_b.ok:
                                brace_rec = (pre_c, c_pose, kind)
                                break
                    if joint_ok and brace and brace_rec is None:
                        fails["brace_ik"] += 1
                        if require_joint:
                            continue
            return {"f_pose": pose, "attach": eef_attach, "pre_attach": eef_pre, "wps": wps, "carry": carry, "lift": lift, "brace": brace_rec, "b_dir": b_dir,
                    "point": point}
        return None

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
        for dlat in SECOND_BASE_LATERALS:
            for trunk in ((manip.translate, manip.tilt), (0.5, 0.0), (0.25, -0.3)):
                if tried >= SECOND_BASE_MAX_TRIES:
                    log.info("%s: second base pose: none of %d stances reaches %s (%s)", self.gp.alias, tried, what, skipped)
                    return None, manip
                self._check_deadline()
                st = manip.shifted(lateral=manip.lateral + dlat, name="attach").with_trunk(trunk[0], trunk[1], "attach")
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
                    log.info("%s: second base pose: lateral %+.2f, trunk (%.2f, %+.2f) reaches %s", self.gp.alias, dlat, trunk[0], trunk[1], what)
                    return att, st
        log.info("%s: second base pose: none of %d stances reaches %s (%s)", self.gp.alias, tried, what, skipped)
        return None, manip

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

    def after_settle(self, draft: Draft):
        tname, pname = self.tname, self.pname
        f_pose = draft.object_poses[pname]
        goal = self._attached_pose(f_pose)
        draft.goal_poses[tname] = goal
        draft.groups[self._group_name([tname, pname])] = self._group_target(f_pose, goal, tname, pname)

    def apply_goal(self, draft: Draft):
        super().apply_goal(draft)
        male, female = self.obj(self.t_role), self.obj(self.p_role)
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
            og.sim.step_physics()
            self.world.keep_still(self.objects.values())

    def after_goal_check(self, draft: Draft):
        """The attach / pre-attach at the settled partner pose from the manipulation stance (male in hand)."""
        tname, pname = self.tname, self.pname
        grasp = self.sidecar.grasp_of(tname)
        manip = self.sidecar.stance("manipulation")
        f_pose = draft.object_poses[pname]
        goal = self._attached_pose(f_pose)
        attach = IK.grasp_pose(goal, grasp.eef_in_object)
        pre = (attach[0] + self._insertion_axis_world(f_pose) * float(self.g("pre_attach_back_m")), attach[1])
        male, female, support = self.obj(self.t_role), self.obj(self.p_role), self.obj("support")
        with TT.posed_in_hand(self, "right", male, grasp.eef_in_object):
            with attached(self.kit, "right", male):
                res = self.ikc.exact([pre, attach], manip, "right", ignore_for_last=[support, female], scale_key=self.choice.models[self.t_role].scale)
        if not res.ok:
            raise Reject("place_unreachable", f"settled partner: pre-attach / attach fail the exact IK: {res.reason}")
        grasp.pre_place, grasp.place = pre, attach
        if grasp.waypoints:
            grasp.waypoints[-1] = pre
