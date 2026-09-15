"""IORS small-scale (spec §6.11): ontop, under, inside, nextto, touching — the reference v2 task module.

Init per §4 (tabletop phases 1a / 1b / 1c / 2 / 3, called on ``tabletop.py``); this module adds only the §6.11 rules:
pair admissibility (per template), the partner side / gap rule of §4.6 step 2, the goal pose per relation, the place /
pre-place IK from the manipulation stance (after the goal settled), the touching push + brace (§6.11).

``tabletop.py`` belongs to another agent; the functions called here and their signatures are the README's tabletop
section. Until it lands, ``_TabletopTemp`` (clearly TEMP, deleted then) provides a minimal stand-in: one edge stance at
s = 0.45, init trunk (0.5, 0), a manipulation stance from the (stand-off, translate, tilt) scan with the exact IK on the
target's library grasps at the focus, and Phase 3 draws on the raster cells in front of the robot.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import grasp_library as GL
from ispatialgym.sampling.sim_utils import Reject, SupportTop
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import full_pose_target
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, GraspRecord, Stance

log = logging.getLogger("ispatialgym.sampling.v2")

RELATION_OF_SUBTASK = {
    "inter_object_relation_satisfaction-ontop_small_scale": "ontop",
    "inter_object_relation_satisfaction-under_small_scale": "under",
    "inter_object_relation_satisfaction-inside": "inside",
    "inter_object_relation_satisfaction-nextto": "nextto",
    "inter_object_relation_satisfaction-touching": "touching",
}
#: v1 ``bddl_writer`` definition ids of the relations (facade rendering until D1)
V1_DEFINITION_ID = {"ontop": 0, "under": 1, "inside": 2, "nextto": 3, "touching": 4}

MIN_GAP = 0.03  # §3.2 footprint clearance target–partner
SUPPORT_MARGIN = 0.02  # §3.2 footprint inside the top shrunk by 2 cm
TARGET_JITTER = math.radians(22.5)  # §4.6 yaw jitter around the drawn fixed yaw
FOCUS_RADIUS = 0.05  # §4.6 target within 5 cm of the focus


def _tabletop():
    """The shared tabletop phases (another agent's module) or the TEMP stand-in."""
    try:
        from ispatialgym.sampling.v2 import tabletop  # type: ignore

        return tabletop
    except ImportError:
        return _TabletopTemp


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
        self.grasps: List[GL.Grasp] = []
        self.grasp_key: Optional[str] = None
        self.edge: Optional[EdgeFrame] = None
        self.init_stance: Optional[Stance] = None
        self.manip_candidates: List[Stance] = []
        self.pair: Dict[str, Any] = {}

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
        for role, o in (("target", tgt), ("partner", prt)):
            self.hull[role] = self.world.local_hull(o)
            self.rest[role] = self._rest_quat(role)
        self.grasps, self.grasp_key = self._library_grasps()
        self.pair = self.pair_admissibility()
        self.template_facts["pair"] = self.pair
        self.top = self.support_top(self.support)
        if int(self.top.free.sum()) < 50:
            raise Reject("support_small", f"{self.support.name}: too little free top")
        tt = _tabletop()
        # Phase 1a / 1b / 1c: edge + stand-off, init trunk at tilt 0, lateral position (margins of the IORS row); an edge
        # whose margins exceed its length is skipped (§4.4 step 2) and the next edge tried
        heights = {"H_max": max(self._height("target"), self._height("partner"))}
        margins = self.lateral_margins()
        rng_t = np.random.default_rng(self.ctx.base_seed)
        tried: List[str] = []
        errors: List[str] = []
        while True:
            try:
                self.edge, init = tt.phase1_edge_and_standoff(self, self.top, rng=rng_t, exclude=tried)
            except Reject as e:
                raise Reject("no_free_edge", "; ".join(errors + [str(e)]))
            tried.append(self.edge.name)
            try:
                init = tt.phase1b_init_trunk(self, init, heights["H_max"])
                self.init_stance = tt.phase1c_lateral(self, init, margins, rng=rng_t)
                break
            except Reject as e:
                if e.reason not in ("edge_too_short", "stance_collision"):
                    raise
                errors.append(f"{self.edge.name}: {e}")
        self.init_stance.name = "init"
        # Phase 2: manipulation stance candidates (best angle class), each with its focus
        self.manip_candidates = tt.phase2_manipulation_stance(self, self.init_stance, target_half_extent=self._half_extent("target"))
        if not self.manip_candidates:
            raise Reject("no_manipulation_stance", "Phase 2 left no survivor")
        self.template_facts.update({"support": self.top.to_json(), "edge": self.edge.to_json(), "init_stance": self.init_stance.to_meta(),
                                    "n_manipulation_candidates": len(self.manip_candidates), "grasp_key": self.grasp_key, "n_grasps": len(self.grasps)})

    def pair_admissibility(self) -> Dict[str, Any]:
        """§6.11 per (target, partner): ontop / under -> the partner's top admits the target footprint with >= 1 cm
        (tier any_yaw / aligned / centred, else ``partner_too_small``); nextto -> the centre heights overlap
        (``nextto_height_mismatch``); touching -> partner stability / target pushability; inside -> opening >= footprint
        + 1 cm (the drop test happens in the goal check)."""
        t_pts = G.local_points_world(self.hull["target"], (th.zeros(3), self.rest["target"]))
        p_pts = G.local_points_world(self.hull["partner"], (th.zeros(3), self.rest["partner"]))
        t_ext = (t_pts.max(dim=0).values - t_pts.min(dim=0).values)
        p_ext = (p_pts.max(dim=0).values - p_pts.min(dim=0).values)
        out: Dict[str, Any] = {"target_extent": t_ext.tolist(), "partner_extent": p_ext.tolist(), "relation": self.relation}
        tx, ty = sorted([float(t_ext[0]), float(t_ext[1])])
        px, py = sorted([float(p_ext[0]), float(p_ext[1])])
        if self.relation in ("ontop", "under"):
            diag = math.hypot(tx, ty)
            if diag + 0.02 <= px:
                out["tier"] = "any_yaw"
            elif tx + 0.02 <= px and ty + 0.02 <= py:
                out["tier"] = "aligned"
            elif tx + 0.02 <= px and ty <= py:
                out["tier"] = "centred"
            else:
                raise Reject("partner_too_small", f"target {tx:.2f}x{ty:.2f} does not fit on partner {px:.2f}x{py:.2f} with 1 cm")
        elif self.relation == "nextto":
            t_h, p_h = float(t_ext[2]), float(p_ext[2])
            if not (min(t_h, p_h) > 0):
                raise Reject("nextto_height_mismatch", "degenerate heights")
            out["g_NT"] = float((t_ext + p_ext).mean() / 6)
        elif self.relation == "touching":
            if float(p_ext[2]) > 2.5 * min(px, py) or px < 0.03:
                raise Reject("partner_unstable", f"partner {px:.2f}x{py:.2f}x{float(p_ext[2]):.2f}")
            if tx > 0.089 or ty > 0.30 or float(t_ext[2]) > 0.30:
                raise Reject("target_not_pushable", f"target {tx:.2f}x{ty:.2f}x{float(t_ext[2]):.2f}")
        elif self.relation == "inside":
            if tx + 0.01 > px or ty + 0.01 > py:
                raise Reject("does_not_fit_inside", f"target {tx:.2f}x{ty:.2f} vs opening {px:.2f}x{py:.2f}")
        return out

    def lateral_margins(self) -> Dict[str, float]:
        """§4.4 margin row of ontop / under / inside / nextto / touching: m = |λ| + w_t + D_p + 2·w_p + 0.02 on each side
        (λ from the band union extent, D_p = min(W_p + 0.03, R_img)); the tabletop module fills λ / R_img."""
        w_t = self._half_extent("target")
        w_p = self._half_extent("partner")
        W_p = 2 * w_p
        return {"w_t": w_t, "w_p": w_p, "W_p": W_p, "D_p_max": W_p + 0.03, "safety": 0.02, "relation": self.relation,
                "g_NT": self.pair.get("g_NT")}

    # ------------------------------------------------------------------ instance (§4.6 Phase 3 + §6.11)

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        tt = _tabletop()
        d = Draft()
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        init = self.init_stance
        self.sidecar.add_stance(init)
        order = rng.permutation(len(self.manip_candidates))
        errors = []
        for k in order[:6]:
            manip = self.manip_candidates[int(k)].with_trunk(self.manip_candidates[int(k)].translate, self.manip_candidates[int(k)].tilt, "manipulation")
            manip.focus = self.manip_candidates[int(k)].focus
            manip.arms = ["right"]
            try:
                # §4.5 step 5: transition init -> manipulation
                self.sidecar.transitions.clear()
                self.record_transition(init, manip)
                # Phase 3 step 1: target init pose from the feasible set F at the focus cell (exact IK confirmation)
                t_pose, grasp = tt.phase3_target(self, manip, rng, self.grasps, self.hull["target"], self.rest["target"], self.support_objects())
                # step 2: partner on a random side (touching: robot's left) with gap g
                p_pose = self.place_partner(rng, t_pose, manip)
                # step 4: every object in frame (5 %) in the init image
                cam = self.camera_pose_at(init)
                for role, pose in (("target", t_pose), ("partner", p_pose)):
                    if not self.frustum_ok(cam, self.hull_corners_world(self.hull[role], pose), 0.05):
                        raise Reject("not_in_frame", f"{role} not in the init image (5 %)")
                # step 5: whole-body checks at init and manipulation with the objects placed (objects applied later; here the
                # stances are checked without them, the settled check repeats it with them in after_settle)
                grasp.place_on = pname
                # carry height: the target's bottom clears the partner's top by 3 cm along the lateral carry
                grasp.carry_height = max(0.12, self._height("partner") + 0.03)
                self.sidecar.add_stance(manip)
                self.sidecar.add_grasp(grasp)
                d.object_poses[tname] = t_pose
                d.object_poses[pname] = p_pose
                d.robot_q = self.stance_q(init)
                d.base_pose2d = tuple(init.base_pose2d)
                d.summary.update({"relation": self.relation, "manipulation_candidate": int(k), "grasp": {"key": grasp.grasp_key, "name": grasp.grasp_name, "yaw": grasp.yaw}})
                d.pose_targets[pname] = full_pose_target(p_pose[0], p_pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
                d.pose_targets[tname] = full_pose_target(t_pose[0], t_pose[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
                self.sidecar.describe(tname, self.choice.models["target"].category.replace("_", " "))
                self.sidecar.describe(pname, self.choice.models["partner"].category.replace("_", " "))
                return d
            except Reject as e:
                errors.append(f"candidate {int(k)}: {e}")
                self.sidecar.stances.pop("manipulation", None)
                self.sidecar.grasps.clear()
        raise Reject("target_placement", "; ".join(errors[-3:]))

    def place_partner(self, rng: np.random.Generator, t_pose: G.Pose, manip: Stance) -> G.Pose:
        """§4.6 step 2: partner at edge-to-edge gap g on a random side of the target along the edge (touching: the
        robot's left), footprint on the top (2 cm), >= 3 cm from the target, in the init image."""
        top = self.top
        t_fp = self.world.footprint(self.hull["target"], t_pose)
        st = manip
        D_p = min(self.lateral_margins()["D_p_max"], 0.25)  # TEMP cap until tabletop.py supplies R_img (§4.4)
        if self.relation == "nextto":
            g_lo = float(self.pair["g_NT"]) + 0.03
            if D_p < g_lo:
                raise Reject("pair_gap", f"D_p {D_p:.2f} < g_NT + 0.03 = {g_lo:.2f}")
        else:
            g_lo = MIN_GAP
        sides = [1.0] if self.relation == "touching" else [1.0, -1.0]
        cam = self.camera_pose_at(self.init_stance)
        for _ in range(40):
            side = sides[int(rng.integers(len(sides)))]
            g = float(rng.uniform(g_lo, max(g_lo, D_p)))
            yaw = float(rng.uniform(-math.pi, math.pi))
            p_quat = G.qmul(G.quat_from_yaw(yaw), self.rest["partner"])
            p_pts = G.local_points_world(self.hull["partner"], (th.zeros(3), p_quat))[:, :2]
            # lateral extent of the partner about its origin along the edge tangent
            p_lat = p_pts @ st.tangent
            t_lat = (t_fp - st.base_xy) @ st.tangent
            if side > 0:
                lat = float(t_lat.max()) + g - float(p_lat.min())
            else:
                lat = float(t_lat.min()) - g - float(p_lat.max())
            depth_t, _ = st.local(t_pose[0])
            xy = st.world(depth_t, lat)
            pos = th.tensor([float(xy[0]), float(xy[1]), top.z + self.world.bottom_offset(self.hull["partner"], p_quat) + 0.003])
            p_pose = (pos, p_quat)
            p_fp = self.world.footprint(self.hull["partner"], p_pose)
            if not top.polygon_supported(p_fp, margin=SUPPORT_MARGIN):
                continue
            if G.polygons_overlap(t_fp, p_fp, margin=MIN_GAP):
                continue
            if not self.frustum_ok(cam, self.hull_corners_world(self.hull["partner"], p_pose), 0.05):
                continue
            return p_pose
        raise Reject("partner_placement", "no partner position with the gap rule on the top and in frame")

    # ------------------------------------------------------------------ goal (§6.11)

    def goal(self, draft: Draft):
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        t_pose, p_pose = draft.object_poses[tname], draft.object_poses[pname]
        goal = self.goal_pose(t_pose, p_pose)
        draft.goal_poses[tname] = goal
        draft.pose_targets[tname] = full_pose_target(goal[0], goal[1], self.tol("position", 0.03), self.tol("orientation", 0.20))
        if self.relation == "inside":
            draft.pose_targets[tname]["orientation_metric"] = None
            draft.pose_targets[tname]["orientation_tolerance"] = None
        manip = self.sidecar.stance("manipulation")
        self.sidecar.set_goal(manip.base_pose2d, manip.translate, manip.tilt, arms="same")

    def goal_pose(self, t_pose: G.Pose, p_pose: G.Pose) -> G.Pose:
        """§6.11 goal: ontop / under(small) -> centred on the partner, z = partner top + bottom offset + 1 cm; inside ->
        centred on the fillable centroid, dropped from the rim + 5 mm; nextto / touching -> the position closest to the
        init pose that satisfies the predicate (nextto: AABB gap < g_NT; touching: contact, nudged in the goal check)."""
        t_quat = t_pose[1]
        t_hull = G.local_points_world(self.hull["target"], (th.zeros(3), t_quat))
        p_hull_w = self.world.hull_at(self.hull["partner"], p_pose)
        st = self.sidecar.stance("manipulation")
        if self.relation in ("ontop", "under"):
            z = float(p_hull_w[:, 2].max()) - float(t_hull[:, 2].min()) + 0.01
            xy = (p_hull_w[:, :2].max(dim=0).values + p_hull_w[:, :2].min(dim=0).values) / 2
            return th.tensor([float(xy[0]), float(xy[1]), z]), t_quat
        if self.relation == "inside":
            centre, drop_z = self._container_volume(self.obj("partner"), p_pose)
            return th.tensor([float(centre[0]), float(centre[1]), drop_z - float(t_hull[:, 2].min())]), t_quat
        # nextto / touching: slide the target along the edge tangent toward the partner until the AABB gap is g
        gap = 0.0 if self.relation == "touching" else max(0.0, float(self.pair["g_NT"]) - 0.01)
        p_lat_min = float(((p_hull_w[:, :2] - st.base_xy) @ st.tangent).min())
        t_lat_max = float((t_hull[:, :2] @ st.tangent).max())
        lat = p_lat_min - gap - t_lat_max
        depth_t, lat_t = st.local(t_pose[0])
        if self.relation == "nextto" and lat < lat_t:
            pass  # the target moves left toward the partner
        xy = st.world(depth_t, lat)
        return th.tensor([float(xy[0]), float(xy[1]), float(t_pose[0][2])]), t_quat

    def after_settle(self, draft: Draft):
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        p, q = self.obj("partner").get_position_orientation()
        draft.pose_targets[pname] = full_pose_target(p, q, self.tol("position", 0.03), self.tol("orientation", 0.20))
        goal = self.goal_pose(draft.object_poses[tname], (p.clone(), q.clone()))
        draft.goal_poses[tname] = goal
        tgt = draft.pose_targets[tname]
        tgt["position"] = [float(v) for v in goal[0]]
        tgt["orientation"] = [float(v) for v in goal[1]]

    def refine_targets_from_goal(self, draft: Draft):
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        if self.relation == "touching":
            self._nudge_until_touching(tname, pname)
        super().refine_targets_from_goal(draft)
        p, _ = self.obj("partner").get_position_orientation()
        tgt = draft.pose_targets[pname]
        if float(th.linalg.norm(p - th.tensor(tgt["position"]))) > 0.005:
            raise Reject("goal_partner_moved", f"partner moved {float(th.linalg.norm(p - th.tensor(tgt['position']))):.3f} m at the goal")

    def after_goal_check(self, draft: Draft):
        """§6.11: place pose (settled goal pose ∘ T_obj→eef of the drawn grasp) and pre-place IK-checked from the
        manipulation stance (support ignored for the place pose only); touching: push contact + left brace recorded."""
        tname = self.bindings.one("target").name
        grasp = self.sidecar.grasp_of(tname)
        manip = self.sidecar.stance("manipulation")
        goal_pos = th.tensor(draft.pose_targets[tname]["position"])
        goal_quat = th.tensor(draft.pose_targets[tname]["orientation"]) if draft.pose_targets[tname].get("orientation") else draft.goal_poses[tname][1]
        place = IK.grasp_pose((goal_pos, goal_quat), grasp.eef_in_object)
        pre_place = IK.pre_pose(place)
        ignore = self.support_objects() + [self.obj("partner")]
        with_obj = self.obj("target")
        from ispatialgym.sampling.v2.navigation import attached

        with attached(self.kit, "right", with_obj, eef_in_object=grasp.eef_in_object):
            res = self.ikc.exact([pre_place, place], manip, "right", ignore_for_last=ignore, scale_key=self.choice.models["target"].scale, cache=False)
        if not res.ok:
            # diagnostics only on failure: is it reach (fails even without the object and without collision partners) or the
            # attached object / collision?
            r_pre = self.ikc.exact([pre_place], manip, "right", cache=False)
            r_place = self.ikc.exact([place], manip, "right", ignore_for_last=ignore, cache=False)
            _, lat = manip.local(goal_pos[:2])
            raise Reject("place_unreachable", f"pre-place / place fail the exact IK from the manipulation stance with the object attached "
                                              f"(without the object: pre-place {'ok' if r_pre.ok else 'fail'}, place {'ok' if r_place.ok else 'fail'}; "
                                              f"goal lateral {lat:+.2f} m, trunk ({manip.translate}, {manip.tilt:.2f}), stand-off {manip.stand_off})")
        grasp.pre_place, grasp.place = pre_place, place
        cam = self.camera_pose_at(manip)
        if not self.ikc.eef_in_frame(place, cam) or not self.ikc.eef_in_frame(grasp.grasp, cam):
            raise Reject("gripper_not_in_frame", "grasp or place eef pose outside the manipulation image (5 %)")
        if self.relation == "touching":
            self.touching_contacts(draft, manip)

    def touching_contacts(self, draft: Draft, manip: Stance):
        """§6.11 touching: push contact on the target's right face (right arm) toward the partner, brace pre-contact +
        contact of the closed left gripper on the partner's far (left) side; both IK-checked (§4.6 step 6)."""
        tname, pname = self.bindings.one("target").name, self.bindings.one("partner").name
        t_pose = draft.goal_poses[tname]
        t_hull = self.world.hull_at(self.hull["target"], t_pose)
        p_hull = self.world.hull_at(self.hull["partner"], self.obj("partner").get_position_orientation())
        tangent = manip.tangent
        direction = th.tensor([float(tangent[0]), float(tangent[1]), 0.0])  # toward the partner = robot's left
        z_t = float(t_hull[:, 2].min()) + 0.5 * float(t_hull[:, 2].max() - t_hull[:, 2].min())
        right_face = float((t_hull[:, :2] @ tangent).min())
        centre_t = t_hull.mean(dim=0)
        p_init = th.tensor([*(centre_t[:2] + tangent * (right_face - float(centre_t[:2] @ tangent))).tolist(), z_t])
        dist = float(self.pair.get("g_NT", 0.02)) + 0.01
        push_init = IK.contact_eef(p_init, direction)
        push_goal = (push_init[0] + direction * dist, push_init[1])
        push_pre = IK.pre_pose(push_init)
        left_face = float((p_hull[:, :2] @ tangent).max())
        centre_p = p_hull.mean(dim=0)
        z_p = float(p_hull[:, 2].min()) + 0.5 * float(p_hull[:, 2].max() - p_hull[:, 2].min())
        b_point = th.tensor([*(centre_p[:2] + tangent * (left_face - float(centre_p[:2] @ tangent))).tolist(), z_p])
        brace = IK.contact_eef(b_point, -direction)
        brace_pre = IK.pre_pose(brace)
        ignore_push = self.support_objects() + [self.obj("target")]
        ignore_brace = self.support_objects() + [self.obj("partner")]
        res = self.ikc.exact_both_arms(("right", [push_pre, push_init, push_goal]), ("left", [brace_pre, brace]), manip, ignore_push, ignore_brace)
        if not res.ok:
            raise Reject("contact_unreachable", f"touching push / brace: {res.reason}")
        self.sidecar.add_contact(ContactRecord("push", "right", push_pre, push_init, None, push_goal, direction, dist, object=tname))
        self.sidecar.add_contact(ContactRecord("brace", "left", brace_pre, brace, direction=-direction, brace_kind="side", object=pname))
        manip.arms = ["right", "left"]

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
        recs = []
        if g is not None and manip.focus is not None:
            recs.append({"scope": {"target": f"{self.choice.models['target'].category}/{self.choice.models['target'].model}",
                                   "support_height": round(float(self.top.z - self.world.floor_z), 2)},
                         "type": "grasp_set", "trunk": [manip.translate, round(manip.tilt, 2)],
                         "cell": [round(v, 2) for v in manip.world_to_base(manip.focus[:2]).tolist()],
                         "feasible": [[round(g.yaw, 2), g.grasp_name]], "hard": False})
        return recs

    # ------------------------------------------------------------------ helpers

    def _rest_quat(self, role: str) -> th.Tensor:
        choice = self.choice.models.get(role)
        if choice is not None and choice.grasp_key:
            try:
                rq = GL.rest_quat_xyzw(choice.grasp_key)
                if rq is not None:
                    return rq
            except FileNotFoundError:
                pass
        return G.IDENTITY_QUAT.clone()

    def _library_grasps(self) -> Tuple[List[GL.Grasp], Optional[str]]:
        choice = self.choice.models["target"]
        pipeline = self.gp.roles["target"].grasp_pipeline() or "tabletop"
        kinds = {"tabletop": ["tabletop"], "insert": ["insert"], "floating": ["floating"], "attachment": ["attachment"]}[pipeline]
        try:
            key = GL.resolve_key(choice.category, choice.model, choice.grasp_key, kinds)
            grasps = GL.load_grasps(key)
        except (FileNotFoundError, ValueError) as e:
            raise Reject(f"missing_grasps:{pipeline}", f"{choice.category}/{choice.model}: {e}")
        if not grasps:
            raise Reject(f"missing_grasps:{pipeline}", f"{choice.category}/{choice.model}: no admissible grasp row")
        return grasps, key

    def _height(self, role: str) -> float:
        pts = G.local_points_world(self.hull[role], (th.zeros(3), self.rest[role]))
        return float(pts[:, 2].max() - pts[:, 2].min())

    def _half_extent(self, role: str) -> float:
        pts = G.local_points_world(self.hull[role], (th.zeros(3), self.rest[role]))
        ext = pts[:, :2].max(dim=0).values - pts[:, :2].min(dim=0).values
        return float(ext.max()) / 2

    def pose_on(self, role: str, xy, yaw: float) -> G.Pose:
        quat = G.qmul(G.quat_from_yaw(yaw), self.rest[role])
        pos = th.tensor([float(xy[0]), float(xy[1]), self.top.z + self.world.bottom_offset(self.hull[role], quat) + 0.003])
        return pos, quat

    def _container_volume(self, obj, p_pose: G.Pose) -> Tuple[th.Tensor, float]:
        from omnigibson.macros import macros

        p_hull = self.world.hull_at(self.hull["partner"], p_pose)
        centre = (p_hull.max(dim=0).values + p_hull.min(dim=0).values) / 2
        for link in obj.links.values():
            if link.is_meta_link and link.meta_link_type in macros.object_states.contains.CONTAINER_META_LINK_TYPES:
                p, _ = link.get_position_orientation()
                centre = th.tensor([float(p[0]), float(p[1]), float(centre[2])])
                break
        return centre, float(p_hull[:, 2].max()) + 0.005


# ----------------------------------------------------------------------------------------------------------------
# TEMP stand-in for tabletop.py (deleted when the tabletop module lands). Signatures = README tabletop section.
# ----------------------------------------------------------------------------------------------------------------


class _TabletopTemp:
    """TEMP minimal tabletop phases so the reference task runs end to end before ``v2/tabletop.py`` exists."""

    STAND_OFFS = (0.35, 0.45, 0.55)
    TRANSLATES = (0.0, 0.25, 0.5, 0.75, 0.95)
    TILTS = (0.0, -0.3, -0.6, -0.9)

    @staticmethod
    def phase1_edge_and_standoff(s: IORSSmall, top: SupportTop, rng=None, stand_off: float = 0.45, exclude: Sequence[str] = ()) -> Tuple[EdgeFrame, Stance]:
        """§4.2: any edge (no preference; ``exclude`` = edges already tried) with a free-floor run in front of it at
        s = 0.45 (+ overhang >= 5 cm): at least a third of the lateral positions (10 cm raster) are on free floor.
        Returns (edge, init stance at the edge midpoint, trunk (0.5, 0)); the collision check happens per lateral
        position in Phase 1c."""
        lo, hi = s.support.aabb
        aabb_xy = G.aabb_corners(lo, hi)[:, :2]
        edges = [e for e in top.edges() if e[0] not in exclude]
        if not edges:
            raise Reject("no_free_edge", "every edge tried")
        order = rng.permutation(len(edges)) if rng is not None else range(len(edges))
        errors = []
        for i in order:
            name, mid, normal, length = edges[int(i)]
            overhang = max(0.0, float((aabb_xy @ normal).max()) - float(th.dot(mid, normal)))
            edge = EdgeFrame.from_edge(name, mid, normal, length, overhang if overhang >= 0.05 else 0.0)
            lats = th.linspace(-length / 2 + 0.1, length / 2 - 0.1, max(2, int(length / 0.1))).tolist()
            n_free = sum(1 for lat in lats if s.kit.trav_free(edge.base_xy(stand_off, lat)))
            if n_free < max(1, len(lats) // 3):
                errors.append(f"{name}: free floor at {n_free}/{len(lats)} lateral positions")
                continue
            xy = edge.base_xy(stand_off, 0.0)
            return edge, Stance("init", (float(xy[0]), float(xy[1]), edge.yaw), 0.5, 0.0, stand_off, edge, lateral=0.0)
        raise Reject("no_free_edge", "; ".join(errors))

    @staticmethod
    def phase1b_init_trunk(s: IORSSmall, init: Stance, H_max: float) -> Stance:
        """§4.3 (TEMP: keeps translate 0.5, tilt 0 — the FK-based Δ solve belongs to tabletop.py)."""
        st = init.with_trunk(0.5, 0.0, "init")
        st.extras["H_max"] = H_max
        return st

    @staticmethod
    def phase1c_lateral(s: IORSSmall, init: Stance, margins: Dict[str, float], rng=None) -> Stance:
        """§4.4 (TEMP: lateral uniform within ±(L/2 − m) with m = |λ| + w_t + D_p + 2·w_p + 0.02, |λ| = 0.10 and
        D_p = min(W_p + 0.03, 0.25) standing in for the band extent / R_img the tabletop module derives; base on free
        floor and whole-body collision-free; redraw up to 20 times)."""
        L = init.edge.length
        m = 0.10 + margins["w_t"] + min(margins["D_p_max"], 0.25) + 2 * margins["w_p"] + margins["safety"]
        half = L / 2 - m
        if half < 0:
            raise Reject("edge_too_short", f"m_R + m_L = {2 * m:.2f} > L = {L:.2f}")
        cands = []
        for _ in range(20):
            lat = float(rng.uniform(-half, half)) if rng is not None else 0.0
            st = init.shifted(lateral=lat, name="init")
            if s.kit.trav_free(st.base_xy):
                cands.append(st)
        if not cands:
            raise Reject("stance_collision", f"no lateral position in ±{half:.2f} m on free floor")
        free = s.ikc.stances_collision_free(cands)
        for st, f in zip(cands, free.tolist()):
            if f:
                return st
        # diagnostics for the rejection: does the body without the arms fit (then the untucked hands hit something)?
        n_body = int(s.kit.collision_free(th.stack([s.ikc.q_locked(c, arms="tucked") for c in cands]), margin=0.01).sum())
        raise Reject("stance_collision", f"{len(cands)} lateral positions on free floor, none whole-body collision-free at trunk (0.5, 0) "
                                         f"(arms hanging: {n_body} free; laterals {[round(c.lateral, 2) for c in cands[:6]]})")

    @staticmethod
    def phase2_manipulation_stance(s: IORSSmall, init: Stance, target_half_extent: float = 0.05) -> List[Stance]:
        """§4.5 (TEMP): scan stand-off × translate × tilt (grid tilts), drop colliding candidates, focus = the free top
        cell nearest the point 0.30 m into the top on the centre line; keep candidates whose optical axis is within 8°
        (else 20°) of the focus; sample a candidate list (the caller draws uniformly)."""
        top = s.top
        cands = []
        for so in _TabletopTemp.STAND_OFFS:
            for tr in _TabletopTemp.TRANSLATES:
                for ti in _TabletopTemp.TILTS:
                    st = init.shifted(stand_off=so, name="manipulation").with_trunk(tr, ti, "manipulation")
                    if not s.kit.trav_free(st.base_xy):
                        continue
                    cands.append(st)
        if not cands:
            return []
        free = s.ikc.stances_collision_free(cands)
        cands = [c for c, f in zip(cands, free.tolist()) if f]
        cells = top.cells(free_only=True)
        out_by_class: Dict[int, List[Stance]] = {0: [], 1: []}
        for st in cands:
            want = st.world(0.30, 0.0)
            if len(cells) == 0:
                continue
            d = th.linalg.norm(cells - want, dim=1)
            focus_xy = cells[int(th.argmin(d))]
            if not top.contains(focus_xy, target_half_extent + 0.02):
                continue
            focus = th.tensor([float(focus_xy[0]), float(focus_xy[1]), top.z + 0.03])
            cam = s.kit.camera_pose(s.ikc.q_locked(st))
            ang = s.world.camera_model.angle_off_axis(cam, focus)
            if ang > math.radians(20.0):
                continue
            st.focus = focus
            st.extras["axis_angle"] = ang
            out_by_class[0 if ang <= math.radians(8.0) else 1].append(st)
        return out_by_class[0] or out_by_class[1]

    @staticmethod
    def phase3_target(s: IORSSmall, manip: Stance, rng, grasps: List[GL.Grasp], hull: th.Tensor, rest: th.Tensor, ignore_for_grasp) -> Tuple[G.Pose, GraspRecord]:
        """§4.6 step 1: screen 8 yaws × library grasps at the focus (12 seeds) -> F; draw (yaw, grasp) from F, jitter
        ±22.5°, place within 5 cm of the focus with the footprint on the top (2 cm); confirm pre-grasp + grasp with the
        exact IK (support ignored for the grasp pose only); next element of F on failure; ``Reject`` when F is exhausted."""
        top = s.top
        focus = manip.focus
        yaws = [k * math.pi / 4 for k in range(8)]

        def pose_at(cell, yaw):
            return s.pose_on("target", cell, yaw)

        F = s.ikc.feasible_sets([focus[:2]], yaws, grasps, pose_at, manip, "right", ignore_for_grasp, scale_key=s.choice.models["target"].scale).get(0, [])
        if not F:
            raise Reject("grasp_unreachable", f"empty feasible set F at the focus {focus[:2].tolist()}")
        order = rng.permutation(len(F))
        for i in order:
            yaw0, g = F[int(i)]
            yaw = yaw0 + float(rng.uniform(-TARGET_JITTER, TARGET_JITTER))
            for _ in range(5):
                r, a = float(rng.uniform(0, FOCUS_RADIUS)), float(rng.uniform(0, 2 * math.pi))
                xy = focus[:2] + th.tensor([r * math.cos(a), r * math.sin(a)])
                pose = pose_at(xy, yaw)
                if not top.polygon_supported(s.world.footprint(hull, pose), margin=SUPPORT_MARGIN):
                    continue
                eef = IK.grasp_pose(pose, g.pose)
                pre = IK.pre_pose(eef)
                res = s.ikc.exact([pre, eef], manip, "right", ignore_for_last=ignore_for_grasp, scale_key=s.choice.models["target"].scale)
                if res.ok:
                    rec = GraspRecord(s.bindings.one("target").name, "right", g.key, g.name, yaw, pre, eef, finger_q=list(g.finger_q), eef_in_object=g.pose)
                    return pose, rec
        raise Reject("grasp_unreachable", f"F ({len(F)} pairs) exhausted without an exact-IK confirmation")
