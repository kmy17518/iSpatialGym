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

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, Stance

import omnigibson as og  # noqa: E402
from omnigibson.object_states import AttachedTo  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASK = "configuration_matching-attachment"


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
                init = TT.phase1b_init_trunk(self, st, H_max, bands=bands, top=self.top)
                init = TT.phase1c_lateral(self, init, margins, rng, top=self.top, bands=bands)
                init.name = "init"
                cands = TT.phase2_manipulation_stance(self, init, TT.half_extent(self.hull[self.t_role], self.rest[self.t_role]), target_height=self.height[self.t_role],
                                                      bands=bands, top=self.top)
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
        radius profile (12 angular bins, radii within 10 %). Explicit ``generation.orientation_equivalences`` win."""
        explicit = self.gen("orientation_equivalences")
        if explicit:
            return [list(map(float, q)) for q in explicit]
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
        for k in rng.permutation(len(self.manip_candidates))[:6]:
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

    def _sample_with(self, rng, init: Stance, manip: Stance, k: int) -> Draft:
        d = Draft()
        tname, pname = self.tname, self.pname
        male, female, support = self.obj(self.t_role), self.obj(self.p_role), self.obj("support")
        self.record_transition(init, manip)
        cam_init = self.camera_pose_at(init)
        # step 1: the male at the focus from F
        m_pose, grasp = TT.phase3_target(self, manip, rng, self.grasps, self.hull[self.t_role], self.rest[self.t_role], [support], object_name=tname,
                                         scale_key=self.choice.models[self.t_role].scale, bands=self.bands)
        grasp.place_on = pname
        male.set_position_orientation(*m_pose)
        male.keep_still()
        self.kit.sync_world()
        # step 2: the partner on a cell of the attach-reach band, gap >= 0.04, centre distance >= 0.08, on the top, in frame
        cells = TT.band_cells_world(self.attach_bands, manip)
        if len(cells) == 0:
            raise Reject("partner_placement", "no attach-reach cell at this trunk")
        m_fp = self.world.footprint(self.hull[self.t_role], m_pose)
        f_pose = attach = pre_attach = None
        fails = {"distance": 0, "support": 0, "gap": 0, "frame": 0, "ik": 0}
        for i in rng.permutation(len(cells))[:40]:
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
            # attach pose of the drawn grasp; pre-attach 10 cm back along the female insertion axis; male in hand
            goal = self._attached_pose(pose)
            eef_attach = IK.grasp_pose(goal, grasp.eef_in_object)
            axis = self._insertion_axis_world(pose)
            eef_pre = (eef_attach[0] + axis * float(self.g("pre_attach_back_m")), eef_attach[1])
            female.set_position_orientation(*pose)
            female.keep_still()
            self.kit.sync_world()
            with TT.posed_in_hand(self, "right", male, grasp.eef_in_object):
                with attached(self.kit, "right", male):
                    res = self.ikc.exact([eef_pre, eef_attach], manip, "right", ignore_for_last=[support, female], scale_key=self.choice.models[self.t_role].scale)
            if not res.ok:
                fails["ik"] += 1
                continue
            f_pose, attach, pre_attach = pose, eef_attach, eef_pre
            break
        if f_pose is None:
            raise Reject("partner_placement", f"no partner cell with a reachable attach pose ({fails})")
        goal = self._attached_pose(f_pose)
        # in-hand reorientation: lifted grasp -> pre-attach at 3 SLERP waypoints (male attached)
        carry = 0.12  # the lift clears the table; the SLERP waypoints carry the male up to the pre-attach pose above the partner
        lift = (grasp.grasp[0] + th.tensor([0.0, 0.0, carry]), grasp.grasp[1])
        wps = [(TT._t(p[0]), TT._t(p[1])) for p in C.line_waypoints(lift, pre_attach, int(self.g("reorientation_waypoints")) + 1)]
        with TT.posed_in_hand(self, "right", male, grasp.eef_in_object):
            with attached(self.kit, "right", male):
                r_w = self.ikc.waypoints_ok([lift] + wps, manip, "right", ignore_all=[support])
                if not r_w.ok:
                    raise Reject("reorientation_unreachable", f"SLERP waypoints to the pre-attach pose: {r_w.reason}")
                # partner brace: the closed left gripper on the partner's face opposite to the insertion direction
                brace_pre, brace, brace_dir = self._partner_brace(f_pose, manip)
                res_b = TT.brace_second_arm(self, manip, "right", [pre_attach, attach], "left", [brace_pre, brace], ignore_work_last=[support, female],
                                            ignore_brace_last=[support, female])
        if not res_b.ok:
            raise Reject("contact_unreachable", f"partner brace with the attach arm locked: {res_b.reason}")
        cam_manip = self.camera_pose_at(manip)
        for nm, p in (("grasp", grasp.grasp), ("attach", attach), ("brace", brace)):
            if not self.ikc.eef_in_frame(p, cam_manip):
                raise Reject("gripper_not_in_frame", f"{nm} eef pose outside the manipulation image (5 %)")
        objects = {tname: m_pose, pname: f_pose}
        TT.whole_body_check_at(self, [init, manip], objects, require=True, reason="stance_collision")
        # record
        grasp.carry_height = float(carry)
        grasp.waypoints = wps
        grasp.pre_place, grasp.place = pre_attach, attach
        manip.arms = ["right", "left"]
        self.sidecar.add_stance(manip)
        self.sidecar.add_grasp(grasp)
        self.sidecar.add_contact(ContactRecord("brace", "left", brace_pre, brace, direction=brace_dir, brace_kind="partner_face", object=pname))
        manip.contacts = ["brace"]
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
        d.summary.update({"grasp": {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw}, "manipulation_candidate": k, "partner_fails": fails})
        return d

    def _partner_brace(self, f_pose: TT.Pose, manip: Stance) -> Tuple[TT.Pose, TT.Pose, th.Tensor]:
        """Closed left gripper on the partner's face opposite to the insertion direction (the face the insertion pushes
        toward): the horizontal insertion direction when it has one (>= 30 % horizontal), else the partner's side on
        the robot's left; contact at mid-height, approach into the face, pre-contact 10 cm back."""
        axis = self._insertion_axis_world(f_pose)
        d_ins = -axis  # the male moves along -axis into the female
        horiz = th.tensor([float(d_ins[0]), float(d_ins[1]), 0.0])
        if float(th.linalg.norm(horiz)) >= 0.3:
            b_dir = horiz / th.linalg.norm(horiz)
        else:
            # vertical insertion: hold the partner on the face toward the robot (the left gripper reaches forward)
            f = manip.forward
            b_dir = th.tensor([-float(f[0]), -float(f[1]), 0.0])
        pts = C.transform_points(f_pose, self.hull[self.p_role])
        proj = pts[:, :2] @ b_dir[:2]
        z_mid = float(pts[:, 2].min()) + 0.5 * float(pts[:, 2].max() - pts[:, 2].min())
        centre = pts.mean(dim=0)
        # face point: the partner's centre pushed out to the hull's extreme along b_dir, at mid-height
        point = th.tensor([float(centre[0]), float(centre[1]), z_mid]) + b_dir * (float(proj.max()) - float(centre[:2] @ b_dir[:2]))
        brace = C.contact_eef(point, -b_dir)
        return C.pre_pose(brace), brace, -b_dir

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
