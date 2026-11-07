"""Alignment — axial (spec §6.10; init per §4.2–4.6 with the axial bullets).

Roles: ``target`` (pushed object with a reviewed frame record: ``frame_front`` / ``frame_long``; ``roles.target.scale``
may be configured), scene ``support``, ``floor``.

Template: hull / COM / extents, the generic-pinch reach band (push poses at the object's height), Phases 1a / 1b
(H_max = the object's height) / 1c (row ``axial``: w_diag = ½√(long² + short²)) with the **reference edge visible**
rule (≥ 3 of 9 points of the front edge over [λ − 0.30, λ + 0.30] in frame and unoccluded) from the init stance —
the template retries the next lateral position, then the next edge — and Phase 2 filtered by the same rule from the
manipulation stance.

Instance: yaw = aligned yaw ± U(20°, 70°) (random sign), the object at the focus (footprint on the top), at least one
end of the long axis inside the band with its top-down push pose confirmed by the exact IK; push contact at that end
(closed right gripper beside the end, pushing tangentially), arc waypoints at δ/2 and the goal about the COM (§3.9);
the **stop edge** braced by the closed left gripper **when the joint two-arm placement exists** (push point in the
right band and stop point in the left contact band at the same trunk, the blade clear of the target's sweep, two-arm IK
with the push arm locked; owner decision 2026-09-16 08:10: axial is push-only otherwise — the oracle's push closure is
the stop, ``summary.brace = "present" | "none"``); every right-hand push pose IK-feasible. After settling a vertical hinge through the COM (parent = support;
the target stays at its settled pose, the anchor lies on the COM's vertical axis ``hinge_sink_m`` = 1 mm below the support top) is installed
and recorded as a TRO constraint (drift ≤ 1 mm / 0.005 rad over 60 steps); it is frictionless
unless the variant's generation sets the world-vertical resistance keys (``hinge_resistance``).
Goal: position unchanged, orientation = long axis parallel to the front edge, front toward the robot (tolerance
0.10 rad, equivalences from the customs entry). Sidecar: ``init`` / ``manipulation`` stances (arms right + left),
the transition, ``contacts.push`` (pre, init, mid, goal, direction, distance) and ``contacts.brace`` (``stop_edge``).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import contact_maps as CMAP
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling import support_rules as SR
from ispatialgym.sampling.constraints import HINGE_BRAKE_OMEGA_RAD_S, HINGE_STOP_DECEL_RAD_S2, HINGE_VISCOUS_TAU_S, Hinge, create_hinge, hinge_resistance, remove_hinge
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import pose_target
from ispatialgym.sampling.v2 import contact_poses as CPZ
from ispatialgym.sampling.v2 import orient as O
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, annotate_height_extra, lift_contact_height, register
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, Stance

import omnigibson as og  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASK = "alignment-axial"
AXES = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0), "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}
#: closed fingers: the blade half thickness beside a face comes from the measured finger profile (``tabletop.
#: FINGER_PROFILE``: 8.4 mm over the fingertip section, 18 mm above it, no blade at the palm) over the height the face
#: spans above the fingertips — the former 12 mm slab is gone (owner review of the axial demo, 2026-09-16)
STOP_CLEARANCE = 0.005  # §5.5 style: the blade's outer face 5 mm outside the goal hull
PUSH_TIP_CLEARANCE = 0.003  # (= contact_poses.AXIAL_PUSH_TIP_CLEARANCE_M) the closed fingertips this far above the top for the thinnest targets (push / blade stop)
THIN_TARGET_HEIGHT = 0.04  # below this height the stop brace tries the top-down blade before the side contact classes
#: the stop point may lie within one lattice cell of a left contact band cell (the blade stands beside the cell; the left band
#: of a flat target starts at λ = 0.08 on the pilot support while the right band ends at 0.0 — the exact two-arm IK decides)
BRACE_CELL_TOL = 1.0
END_INSET = 0.03  # the pushed end lies this far inside the object's long-axis extremity
BAND_TOLERANCE = 0.5  # an end is "inside the band" within this fraction of the lattice of a band cell
#: the world-vertical hinge resistance keys (``alignment_wall.hinge_resistance``): an axial variant whose own ``generation`` sets any
#: of them gets the resisted hinge (``AlignmentAxial.hinge_resistance``), every other axial variant keeps the frictionless one
HINGE_RESISTANCE_KEYS = ("hinge_stop_decel_rad_s2", "hinge_viscous_tau_s", "hinge_brake_omega_rad_s")


def _axis(name: str) -> th.Tensor:
    name = str(name)
    name = name if name[0] in "+-" else "+" + name
    return th.tensor(AXES[name], dtype=th.float32)


@register(SUBTASK)
class AlignmentAxial(SamplerV2):
    def __init__(self, ctx):
        super().__init__(ctx)
        if self.gp.bddl is None:
            self.definition_id = 0  # v1 writer: alignment-0 (table_surface)
        self.defaults = GD.defaults_for(SUBTASK)
        self.top = None
        self.hull: Optional[th.Tensor] = None
        self.com_local = th.zeros(3)
        self.edge_frame: Optional[EdgeFrame] = None
        self.init_stance: Optional[Stance] = None
        self.manip_candidates: List[Stance] = []
        self.band_spec: Optional[TT.BandSpec] = None
        self.bands = None
        self.left_bands = None  # A1: the left arm's contact band (stop-edge brace), None until B3's bands exist
        self.push_only = False  # the last instance had no stop brace (owner 2026-09-16 08:10: the brace is optional)
        self.hinge: Optional[Hinge] = None
        self.front_axis = th.tensor([1.0, 0.0, 0.0])
        self.long_axis = th.tensor([0.0, 1.0, 0.0])

    def g(self, key: str):
        return self.gen(key, self.defaults.get(key))

    # ------------------------------------------------------------------ roles

    def bind_scene_roles(self):
        self.bind_floor_v2()
        req = self.choice.bindings.get("support")
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if not names:
            raise Reject("spec", f"{self.gp.variant}: scenes.{self.scene}.bindings.support must list scene object names")
        SR.check_support(SUBTASK, self.scene, names[0])
        self.bind_scene_object("support", self.world.obj(names[0]))

    @property
    def tname(self) -> str:
        return self.bindings.one("target").name

    # ------------------------------------------------------------------ frame record (icl frame axes / v1 generation)

    def _frame_axes(self):
        """``(front, long)`` object-frame axes: the customs ``roles.target.frame`` (D1: ``{"<cat>/<model>": {"front": "+x",
        "long": "y"}}``), else the v1 ``generation.front_axis`` / ``long_axis`` (per model dict or a value)."""
        mc = self.choice.models["target"]
        key = f"{mc.category}/{mc.model}"
        r = self.gp.roles["target"]
        frames = getattr(r, "frame", None) or (self.gp.entry.get("roles", {}).get("target", {}).get("frame") or {})
        rec = frames.get(key) if isinstance(frames, dict) else None
        front = long = None
        if isinstance(rec, dict):
            front, long = rec.get("front", rec.get("frame_front")), rec.get("long", rec.get("frame_long"))

        def per_model(v, default):
            if isinstance(v, dict):
                return v.get(key, v.get(mc.category, v.get("default", default)))
            return v if v is not None else default

        front = front or per_model(self.gen("front_axis"), None)
        long = long or per_model(self.gen("long_axis"), None)
        if front is None or long is None:
            # the reviewed frame record of the icl (``roles/candidates.csv`` frame_front / frame_long) when the customs entry
            # does not carry it yet (§2.2 "alignment targets need a reviewed frame record")
            f2, l2 = _icl_frame(mc.category, mc.model)
            front, long = front or f2, long or l2
        if front is None or long is None:
            raise Reject("missing_frame", f"{key}: no frame record (front / long axes) in the customs entry or the icl")
        return _axis(front), _axis(long)

    # ------------------------------------------------------------------ template

    def sample_template(self):
        target = self.obj("target")
        self.hull = self.world.local_hull(target)
        self.com_local = _com_local(target)
        self.front_axis, self.long_axis = self._frame_axes()
        ext = self.hull.max(dim=0).values - self.hull.min(dim=0).values
        li = int(th.argmax(self.long_axis.abs()))
        long_len = float(ext[li])
        short_len = float(ext[[i for i in range(2) if i != li][0]]) if li < 2 else float(min(ext[0], ext[1]))
        ez = float(ext[2])
        support = self.obj("support")
        self.top = self.support_top(support)
        if int(self.top.free.sum()) < 50:
            raise Reject("support_small", f"{support.name}: too little free top")
        mc = self.choice.models["target"]
        self.band_spec = TT.BandSpec(object=f"{mc.category}/{mc.model}", grasp_key=None, grasps=None, hull=self.hull, rest_quat=C.IDENTITY_QUAT.clone(),
                                     scale=list(mc.scale) if mc.scale else None, generic_pinch=True, height=ez)
        w_diag = 0.5 * math.hypot(long_len, short_len)
        margins = {"kind": "axial", "w_diag": w_diag, "safety": 0.02, "H_max": ez}
        self.extents = {"long": long_len, "short": short_len, "height": ez, "w_diag": w_diag}
        ev = self.g("edge_visibility") or {}
        rng = np.random.default_rng(self.ctx.base_seed)
        errors = []
        for edge, st in TT.edge_candidates(self, self.top, rng):
            try:
                bands = TT.ensure_bands(self, self.band_spec, top=self.top, edge=edge)
                # A1: the left contact band of the target's height class (stop-edge brace); §4.4 margins over the union
                if TT.contact_maps_on():
                    # i7-contact 2026-11-07 (step 3, under v2only): the push_blade / axial_brace contact maps replace the generic band
                    # (work) and the left contact band / reach proxy box (brace) in the joint placement; a miss rejects missing_map
                    self.left_bands = None
                    self.axial_maps()
                else:
                    self.left_bands = TT.load_left_contact_band(self, ez, top=self.top, edge=edge, compute=TT.left_band_compute())
                init0 = TT.phase1b_init_trunk(self, st, ez, bands=bands, top=self.top, left_bands=self.left_bands)
                init = None
                for _ in range(8):  # §6.10: retry the next lateral position before the next edge (8 draws: the rule is sensitive to the lateral)
                    cand = TT.phase1c_lateral(self, init0, margins, rng, top=self.top, bands=bands)
                    cand.name = "init"
                    # the reference edge must be in frame from the init camera: with the recomputed (deeper) bands Δ_aim
                    # puts the camera high enough for the near edge to leave the image bottom, so the init translate is
                    # moved within its admissible range [Δ_min, Δ_max] (§4.3 / §4.4 check (2) style) before the next
                    # lateral draw — the lowest camera (largest translate) first
                    t_lo, t_hi = (float(v) for v in cand.extras.get("t_range", [cand.translate, cand.translate]))
                    trials = [cand] + [cand.with_trunk(t, 0.0, "init") for t in sorted(np.linspace(t_lo, t_hi, 5).tolist(), reverse=True) if abs(t - cand.translate) > 1e-6]
                    n_vis = 0
                    for st_try in trials:
                        n_vis, n_in = self._edge_visible_points(st_try, self.camera_pose_at(st_try), 0.0, None, with_in_frame=True)
                        if n_vis >= int(ev.get("min_visible", 3)) and (st_try is cand or (self.ikc.stance_collision_free(st_try) and TT.trunk_sweep_free(self, st_try))):
                            if st_try is not cand:
                                st_try.extras.update({**cand.extras, "init_translate_moved_for_edge": True})
                                st_try.lateral = cand.lateral
                            init = st_try
                            break
                        log.info("%s: edge %s lateral %.2f t %.2f: reference edge %d/%d points visible from init (%d in frame)", self.gp.variant, edge.name, cand.lateral,
                                 st_try.translate, n_vis, int(ev.get("points", 9)), n_in)
                    if init is not None:
                        break
                if init is None:
                    raise Reject("edge_not_visible", "the front edge is not visible from the init stance at 8 lateral draws")
                cands = TT.phase2_manipulation_stance(self, init, w_diag, target_height=ez, bands=bands, top=self.top, focus_z=self.top.z + 0.5 * ez,
                                                      angle_classes_deg=TT.phase2_angle_classes(self), split_cone=TT.split_cone_opts(self))
                cands = [c for c in cands if self._edge_visible_points(c, self.camera_pose_at(c), c.local(c.focus[:2])[1], None) >= int(ev.get("min_visible", 3))]
                if not cands:
                    raise Reject("edge_not_visible", "no Phase-2 survivor sees the front edge")
                self.edge_frame, self.init_stance, self.manip_candidates = edge, init, cands
                break
            except Reject as e:
                errors.append(f"edge {edge.name}: {e}")
                log.info("%s: %s", self.gp.variant, errors[-1][:300])
        if self.init_stance is None:
            raise Reject("no_stance", "; ".join(errors)[:1500])
        self.template_facts.update({"support": self.top.to_json(), "edge": self.edge_frame.to_json(), "init_stance": self.init_stance.to_meta(),
                                    "n_manipulation_candidates": len(self.manip_candidates), "extents": self.extents, "band": getattr(self.bands, "path", None),
                                    "left_band": getattr(self.left_bands, "path", None) if self.left_bands is not None else None,
                                    "front_axis": self.front_axis.tolist(), "long_axis": self.long_axis.tolist()})

    def _edge_visible_points(self, stance: Stance, cam, target_lat: float, target_pose: Optional[TT.Pose], with_in_frame: bool = False):
        """§6.10 reference edge: of 9 points along the front edge over [λ − 0.30, λ + 0.30], how many are in frame (5 %) and
        unoccluded (the ray from the camera reaches the point; the support itself may be hit within 1 cm of it).
        ``with_in_frame`` also returns the number of points inside the image."""
        ev = self.g("edge_visibility") or {}
        n, win, margin = int(ev.get("points", 9)), float(ev.get("window_m", 0.30)), float(ev.get("in_frame_margin", 0.05))
        pts = th.stack([th.tensor([*stance.world(0.0, target_lat + l).tolist(), float(self.top.z)]) for l in np.linspace(-win, win, n)])
        inside = self.world.camera_model.inside(cam, pts, margin)
        n_in = int(inside.sum())
        if n_in == 0:
            return (0, 0) if with_in_frame else 0
        cam_p = TT._t(cam[0])
        sel = pts[inside]
        # reverse rays: from 3 mm above the edge point toward the camera, stopped 0.25 m short of it (the camera sits inside
        # the robot's head); any hit (the target, another object) hides the point
        dirs = (cam_p.unsqueeze(0) - sel) / th.linalg.norm(cam_p.unsqueeze(0) - sel, dim=1, keepdim=True)
        starts = sel + th.tensor([0.0, 0.0, 0.003])
        ends = cam_p.unsqueeze(0) - dirs * 0.25
        hits = self.world.raycast(starts, ends)
        n_vis = sum(1 for h in hits if not h["hit"])
        return (n_vis, n_in) if with_in_frame else n_vis

    # ------------------------------------------------------------------ geometry

    def _aligned_yaw(self, stance: Stance) -> float:
        """Long axis parallel to the front edge, front toward the robot: the object's front axis maps onto −inward."""
        target_dir = -stance.inward
        f = self.front_axis[:2]
        return C.wrap_angle(math.atan2(float(target_dir[1]), float(target_dir[0])) - math.atan2(float(f[1]), float(f[0])))

    def _com_world(self, pose: TT.Pose) -> th.Tensor:
        return TT._t(pose[0]) + C.quat_rotate(TT._t(pose[1]), self.com_local)

    def _push_height(self) -> float:
        """eef origin height of the generic top-down push / stop poses: the band's generic-pinch height (§4.1: eef at
        0.6·h above the top) with the floor lowered from 3 cm to the fingertip length + 3 mm — at the 3 cm floor the
        closed fingers' tips (1.88 cm beyond the origin) hover 1.1 cm above the top and never touch a letter / magazine
        / catalog (h <= 1 cm); with the lower floor the blade spans the side face of every target."""
        z = CPZ.axial_push_height(float(self.top.z), self.extents["height"], PUSH_TIP_CLEARANCE)  # i7-contact 2026-11-07: the shared rule
        # A/B knob generation.contact_height_extra_m (D2 core helper, 0 = no-op): the fingertip stays >= 1 cm below the target's top
        return lift_contact_height(self, z, "push", face_top_z=float(self.top.z) + float(self.extents["height"]))

    # ------------------------------------------------------------------ instance

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        init = self.init_stance
        errors = []
        order = [int(k) for k in rng.permutation(len(self.manip_candidates))]
        if self.left_bands is not None:
            # A1: the right band ends at λ ≈ +0.08 and the left contact band starts at λ ≈ 0 / −0.16 (B3 measurements on the
            # pilot support): a two-arm placement sits near the centre line — candidates whose focus lies nearest to the
            # overlap of both bands first (random order among ties)
            lo, hi = TT.union_extent(self.bands)[1], TT.union_extent(self.left_bands)[0]
            mid = 0.5 * (min(lo, hi) + max(lo, hi))
            order.sort(key=lambda k: abs(self.manip_candidates[k].local(self.manip_candidates[k].focus[:2])[1] - mid))
        plan: List[Tuple[int, Any]] = [(int(k), None) for k in order[:6]]
        self._a5_scan = None
        if TT.contact_maps_on():
            # owner A5 (2026-11-08, map lookups only): per candidate stance the placement draws until one has a PAIR (the push map and the
            # brace map fit on the SAME pushed end); the candidates with a pair first, then the push-only ones
            for o in self.world.spawned.values():
                self.world.park(o, list(self.world.spawned).index(o.name))
            self.kit.sync_world()
            scan = self._scan_candidates(rng, init, order[:6])
            plan = [(k, (d, f)) for k, d, f in scan]
            self._a5_scan = {"candidates": len(order[:6]), "push_feasible": len(scan), "paired": sum(1 for _k, d, _f in scan if self._draw_paired(d))}
            if not plan:
                raise Reject("target_placement", f"no manipulation candidate of {len(order[:6])} has a push-feasible placement on the push_blade map")
        for k, pre in plan:
            cand = self.manip_candidates[int(k)]
            manip = cand.with_trunk(cand.translate, cand.tilt, "manipulation")
            manip.focus, manip.arms = cand.focus, ["right"]  # the left arm joins only when a stop brace is recorded
            self.sidecar.reset_instance()
            self.sidecar.add_stance(init)
            for o in self.world.spawned.values():
                self.world.park(o, list(self.world.spawned).index(o.name))
            self.kit.sync_world()
            try:
                return self._sample_with(rng, init, manip, int(k), pre=pre)
            except Reject as e:
                errors.append(f"candidate {int(k)}: {e}")
                log.info("%s: %s", self.gp.variant, errors[-1][:300])
        raise Reject("target_placement", "; ".join(errors[-3:])[:1500])

    def _sample_with(self, rng, init: Stance, manip: Stance, k: int, pre=None) -> Draft:
        draft = Draft()
        tname = self.tname
        target, support = self.obj("target"), self.obj("support")
        self.record_stance_travel(init, manip)
        yaw_goal = self._aligned_yaw(manip)
        cam_init = self.camera_pose_at(init)
        ev = self.g("edge_visibility") or {}
        z_push = self._push_height()
        z_brace = TT.contact_height(self.top.z, self.extents["height"])
        half_len = 0.5 * self.extents["long"] - END_INSET
        z_axis = th.tensor([0.0, 0.0, 1.0])
        # closed-finger half thickness beside the target's face for a straight-down blade (the band test uses it; the
        # pitched pushes recompute theirs): targets whose faces reach the palm section cannot be pushed with a blade
        geo0 = self._blade_geometry(0.0, 0.0)
        if geo0 is None:
            raise Reject("target_placement", f"target {self.extents['height']:.2f} m tall: its face reaches the palm section of the closed fingers, no blade push")
        push_half = geo0[1]
        self._blade_skips = {}
        if pre is not None:  # owner A5: the draw the candidate scan chose (``_scan_candidates``)
            chosen, fails = pre[0], dict(pre[1])
        else:
            draws, fails = self._placement_draws(rng, manip, cam_init, yaw_goal, push_half, z_brace, half_len, stop_at="pair" if TT.contact_maps_on() else "work")
            chosen = self._pick_draw(draws)
        if chosen is None:
            raise Reject("target_placement", f"no (yaw, position) with a long-axis end whose push point lies in the right band at this trunk ({fails})")
        pose, yaw0, delta, com, pairs = chosen
        # target in the world for the checks below
        target.set_position_orientation(*pose)
        target.keep_still()
        self.kit.sync_world()
        # reference edge visible with the target placed (from init and from the manipulation stance)
        lat_t = manip.local(pose[0])[1]
        for st, cam in ((init, cam_init), (manip, self.camera_pose_at(manip))):
            n_vis = self._edge_visible_points(st, cam, lat_t, pose)
            if n_vis < int(ev.get("min_visible", 3)):
                raise Reject("edge_not_visible", f"{st.name}: {n_vis} of {ev.get('points', 9)} front-edge points visible with the target placed")
        # push contact at an end inside the band: the closed right gripper beside the end, pushing tangentially toward the goal yaw
        d_goal = C.wrap_angle(yaw_goal - yaw0)  # signed rotation to the goal
        spin = 1.0 if d_goal > 0 else -1.0
        push = brace_found = push_orient = None
        push_fails, brace_fails = [], []
        n_push_ok = 0
        # with the long axis parallel to the edge at the goal, exactly one end moves toward the robot there: its stop face
        # is the near long face (reachable for the left arm); the other end's stop face is the far side (x ≈ 0.9–1.1 m
        # from the base, beyond every band) — try the near-stop end first
        motions = []
        for e, jp, goal_pose, stop_pts, d_stop in pairs:
            r_xy = th.tensor([float(e[0] - com[0]), float(e[1] - com[1]), 0.0])
            d = spin * th.cross(z_axis, r_xy, dim=0)
            motions.append((e, jp, d / th.linalg.norm(d), r_xy, goal_pose, stop_pts, d_stop))
        motions.sort(key=lambda m: float(th.dot(m[6][:2], manip.inward)))
        paired = TT.contact_maps_on() and self._draw_paired(chosen)
        if TT.contact_maps_on():
            # owner A5 (2026-11-08): the end whose push AND stop brace fit on the maps (the same pushed end in both) first
            motions.sort(key=lambda m: 0 if (m[1]["ok"] and m[1]["work_ok"]) else 1)
            TT._count(self, "axial_brace.paired" if paired else "axial_brace.push_only")
        brace_reason = "no_joint_cell"
        for e, jp, d, r_xy, goal_pose, stop_pts, d_stop in motions:
            if not jp["work_ok"]:
                continue
            side_pt = self._extreme_point(pose, -d, near=e)  # the face the end moves away from
            side_pt = th.tensor([float(side_pt[0]), float(side_pt[1]), z_push])
            # generic push poses of §4.1 / §6.10: approach straight down or pitched 30° / 45° / 60° toward the robot's
            # forward (A1: the far cells keep an IK-feasible arc with the pitched classes), the cell's passing classes
            # first (B3's bitmask), closing axis (the blade) along the push direction with either wrist turn
            side_c = C.unit(th.cross(z_axis, d, dim=0))  # the finger's side pushes when the pad-face wrist yaw is infeasible
            for pitch in TT.approach_pitch_order(jp.get("classes")):
                a_vec = TT.pitched_approach(pitch, manip.forward)
                for c_axis in (d, side_c):
                    # the closed fingers' half thickness beside the pushed face at this pitch: the pad face (closing axis
                    # along the push) or the finger's side (closing axis across it), from the measured finger profile
                    geo = self._blade_geometry(pitch, 0.0, side=c_axis is side_c)
                    if geo is None:
                        push_fails.append(f"{pitch:.0f}°: face reaches the palm section")
                        continue
                    push_half_p = geo[1]
                    props = []
                    for rank, c_sign in enumerate((1.0, -1.0)):
                        # i7-contact 2026-11-07: the shared chain (contact_poses.axial_push_chain)
                        chain = CPZ.axial_push_chain(side_pt, d, push_half_p, a_vec, c_axis, c_sign, th.tensor([float(com[0]), float(com[1]), z_push]), d_goal)
                        props.append(O.Proposal(f"{pitch:.0f}{'' if c_sign > 0 else '-'}{'_side' if c_axis is side_c else ''}", rank, c_sign, chain))

                    def evaluate(p):
                        res = self.ikc.waypoints_ok(list(p.chain), manip, "right", ignore_all=[support, target], q_first=p.q_first)
                        if not res.ok:
                            return False, res, None, str(res.reason)
                        q_pre = self.ikc.solutions.get(("right", self.ikc._pose_key(p.chain[0])))
                        q_goal = self.ikc.solutions.get(("right", self.ikc._pose_key(p.chain[-1])))
                        if q_pre is None or q_goal is None:
                            return True, res, None, ""
                        # the contact configuration (the screening does not keep one), where the camera is checked: tracked from the pre
                        # witness as the oracle's straight leg runs; when that misses the tracker's joint-limit margin, the plain IK from the
                        # pre witness - scored, not a rejection (the push passed as before; the margin ranks it)
                        r_c = self.ikc.exact(list(p.chain[:2]), manip, "right", ignore_for_last=[support, target], cache=False, q_first=q_pre)
                        q_c = r_c.q if r_c.ok else None
                        if q_c is None:
                            with self.ikc.ignoring([support, target]):
                                q_c = self.kit.ik(p.chain[1], q_pre, arm="right")
                        return True, res, ([q_pre, q_c, q_goal] if q_c is not None else [q_pre]), ""

                    # the two wrist turns of the class by the orientation rule (owner 2026-09-27)
                    chosen, dec_push = O.choose(self.ikc, "right", props, evaluate, self.ikc.q_locked(manip), camera_idx=[1], stage=f"axial push {pitch:.0f}")
                    if chosen is None:
                        push_fails.extend(f"{pitch:.0f}°: {o.get('why', '')}" for o in dec_push["options"] if not o.get("ok"))
                        continue
                    p_pre, p_init, p_mid, p_goal = chosen.proposal.chain
                    push_label = chosen.proposal.label
                    n_push_ok += 1
                    # the stop brace is optional (owner decision 2026-09-16 08:10: axial = push-only, the oracle's push closure
                    # is the stop): tried only when the joint two-arm placement exists for this end (work point in the right
                    # band AND stop point in the left contact band at this trunk) — A1 / §4.6 step 6 with the push arm locked
                    # at its goal solution, the target teleported to its GOAL pose (not ignored, 3 mm margin) and the blade
                    # kept clear of the target's sweep (``_brace_candidates``); a failing brace leaves a push-only instance
                    if jp["ok"]:
                        target.set_position_orientation(*goal_pose)
                        target.keep_still()
                        self.kit.sync_world()
                        try:
                            def eval_brace(p):
                                res_b = TT.brace_second_arm(self, manip, "right", [p_pre, p_goal], "left", list(p.chain), ignore_work_last=[support, target],
                                                            ignore_brace_last=[support], brace_only_check=([support], TT.BRACE_TARGET_MARGIN), q_first_brace=p.q_first)
                                if not res_b.ok:
                                    brace_fails.append(f"{p.label}: {res_b.reason}")
                                return bool(res_b.ok), res_b, None, str(res_b.reason)

                            b_chosen, dec_brace, _b_fails = O.choose_first(self.ikc, "left", self._brace_candidates(manip, stop_pts, d_stop, jp.get("brace_classes"), pose, com, d_goal),
                                                                          key=lambda c: c[1], label=lambda c: c[2], chain=lambda c: [c[0], c[1]], evaluate=eval_brace,
                                                                          q_start=self.ikc.q_locked(manip), stage="axial brace")
                            if b_chosen is not None:
                                brace_pre, brace, kind, extras = b_chosen.proposal.payload
                                brace_found = (brace_pre, brace, kind, {**extras, "push_half_thickness_m": push_half_p, "target_checked_at": "goal",
                                                                        **({"orient": dec_brace} if len([o for o in dec_brace["options"] if o["source"] == "ik"]) > 1 else {})})
                        finally:
                            target.set_position_orientation(*pose)
                            target.keep_still()
                            self.kit.sync_world()
                        brace_reason = "present" if brace_found is not None else ("brace_ik_failed" if brace_fails else "brace_sweep_blocked")
                    else:
                        brace_reason = "brace_map_missing" if jp.get("brace_map_missing") else ("no_joint_cell" if not jp["brace_ok"] else "brace_class_failed")
                    dist = abs(d_goal) * float(th.linalg.norm(r_xy))
                    push = (p_pre, p_init, p_mid, p_goal, d, dist, e, goal_pose, stop_pts, d_stop, pitch)
                    push_end_sign = 1.0 if float(th.dot(e - com, C.quat_rotate(pose[1], self.long_axis))) > 0 else -1.0
                    push_orient = dec_push
                    break
                if push is not None:
                    break
            if push is not None:
                break
        if push is None:
            raise Reject("grasp_unreachable", f"no long-axis end in the right band has an IK-feasible push arc at any pitch ({sorted(set(push_fails))[:4]})")
        p_pre, p_init, p_mid, p_goal, d_push, dist, end, goal_pose, stop_pts, d_stop, pitch = push
        q_goal = goal_pose[1]
        cam_manip = self.camera_pose_at(manip)
        in_frame_poses = [("push init", p_init), ("push goal", p_goal)] + ([("brace", brace_found[1])] if brace_found is not None else [])
        for nm, p in in_frame_poses:
            if not self.ikc.eef_in_frame(p, cam_manip):
                raise Reject("gripper_not_in_frame", f"{nm} eef pose outside the manipulation image (5 %)")
        TT.whole_body_check_at(self, [init, manip], {tname: pose}, require=True, reason="stance_collision")
        # record: the push always, the brace only when the joint placement produced one (push-only otherwise: the
        # manipulation stance works with the right arm, the left arm idle)
        self.sidecar.add_stance(manip)
        push_rec = ContactRecord("push", "right", p_pre, p_init, p_mid, p_goal, d_push, dist, object=tname)
        if push_orient is not None and len([o for o in push_orient["options"] if o["source"] == "ik"]) > 1:
            push_rec.extras["orient"] = push_orient
        annotate_height_extra(self.sidecar.add_contact(push_rec), self)
        if brace_found is not None:
            brace_pre, brace, brace_kind, brace_extras = brace_found
            annotate_height_extra(self.sidecar.add_contact(TT.BraceRecord("brace", "left", brace_pre, brace, direction=-d_stop, brace_kind="stop_edge", object=tname,
                                                    extras={k: (float(v) if isinstance(v, (int, float)) else v) for k, v in brace_extras.items()})), self)
            manip.contacts = ["push", "brace"]
            manip.arms = ["right", "left"]
        else:
            brace_kind, brace_extras = None, {"reason": brace_reason, "brace_fails": sorted(set(brace_fails))[:3], "blade_skips": dict(self._blade_skips)}
            manip.contacts = ["push"]
            manip.arms = ["right"]
        self.push_only = brace_found is None
        manip.extras.update({"push_pitch_deg": float(pitch), "brace": "present" if brace_found is not None else "none", "brace_class": brace_kind,
                             "left_band": self.left_bands is not None, "brace_geometry": brace_extras})
        draft.object_poses[tname] = pose
        draft.robot_q = self.stance_q(init)
        draft.base_pose2d = tuple(init.base_pose2d)
        mc = self.choice.models["target"]
        equiv = self.gp.orientation_equivalences("target", mc.category, mc.model) or self.gen("orientation_equivalences", []) or []
        draft.pose_targets[tname] = pose_target(orientation=q_goal, orientation_metric="full", orientation_tolerance=self.tol("orientation", 0.10), equivalences=equiv)
        draft.goal_poses[tname] = goal_pose
        draft.summary.update({"yaw_goal": yaw_goal, "yaw_init": yaw0, "deviation_deg": math.degrees(delta), "push_end": end.tolist(), "push_distance_m": dist,
                          "manipulation_candidate": k, "spin": spin, "push_pitch_deg": float(pitch), "brace_class": brace_kind,
                          "brace": "present" if brace_found is not None else "none", "brace_reason": brace_reason,
                          "joint_placement": {"left_band": self.left_bands is not None, "fails": fails,
                                              **({"map_pair": bool(paired), "scan": getattr(self, "_a5_scan", None)} if TT.contact_maps_on() else {})}})
        if TT.contact_maps_on():
            self._note_contacts(draft, manip, pose, goal_pose, spin, abs(d_goal), push_end_sign, push_label, brace_found)
        self.sidecar.describe(tname, mc.category.replace("_", " "), f"the {mc.category.replace('_', ' ')} on the table")
        self.sidecar.language["goal_numbers"]["rotate_deg"] = math.degrees(d_goal)
        return draft

    def _note_contacts(self, draft: Draft, manip: Stance, pose: TT.Pose, goal_pose: TT.Pose, spin: float, turn: float, end_sign: float,
                       push_label: Optional[str], brace_found) -> None:
        """Owner A6 (2026-11-08): the library entries this instance executes, for the contact maps' learned overlay
        (``contact_overlay.note_used``; the harvest after a successful demo): the push as ``<end><spin>spin_<pitch label>@t<turn>`` in the
        push_blade map's library convention, the brace as ``<end><spin>spin_<kind>@f<stop face fraction>``. Advisory: never rejects."""
        from ispatialgym.sampling import contact_overlay as CO

        try:
            pm, bm = self.axial_maps()
            sp = "+" if spin > 0 else "-"
            e_push = AlignmentAxial._map_end_sign(self, pm, end_sign)
            if push_label and e_push is not None:
                turns = sorted({abs(float(r["arc_turn_rad"])) for r in pm.meta["classes"] if r and r.get("arc_turn_rad") is not None})
                t_near = min(turns, key=lambda t: abs(t - turn)) if turns else None
                cls = f"{'+' if e_push > 0 else '-'}end{sp}spin_{push_label}" + (f"@t{math.degrees(t_near):.0f}" if t_near is not None and len(turns) > 1 else "")
                CO.note_used(draft.summary, pm, "push_blade", cls, TT.contact_map_check(pm, manip, pose), "target")
            if brace_found is not None and bm is not None:
                e_b = AlignmentAxial._map_end_sign(self, bm, end_sign)
                kind, extras = brace_found[2], brace_found[3] or {}
                if e_b is not None:
                    cls = f"{'+' if e_b > 0 else '-'}end{sp}spin_{kind}@f{int(extras.get('stop_face_fraction', 0))}"
                    CO.note_used(draft.summary, bm, "axial_brace", cls, TT.contact_map_check(bm, manip, goal_pose), "target")
        except Exception as ex:  # noqa: BLE001 - the overlay is advisory
            log.info("%s: contact overlay note skipped: %r", self.gp.variant, ex)

    def _placement_draws(self, rng, manip: Stance, cam_init, yaw_goal: float, push_half: float, z_brace: float, half_len: float, n: int = 12,
                         stop_at: str = "work") -> Tuple[List[Tuple[Any, ...]], Dict[str, int]]:
        """The target placement draws at ``manip`` (``n`` tries: a start yaw ``aligned +- U(deviation range)``, a position in the focus
        disc; on the support, in the init image) with their per-end A1 joint placement (``pairs``: the contact maps' :meth:`map_joint_placement`
        under v2only, else the bands' ``TT.joint_placement``). Returns ``(draws, fails)``: every draw with a push-feasible end, in draw order,
        stopping at the first push-feasible draw (``stop_at="work"``: the rule before owner A5) or at the first draw with a PAIR - an end
        whose push AND stop brace both fit (``"pair"``, owner A5 2026-11-08; map lookups only)."""
        target_z_axis = th.tensor([0.0, 0.0, 1.0])
        bands, left_bands = self.bands, self.left_bands
        fails = {"support": 0, "frame": 0, "work_off_band": 0, "brace_off_band": 0, "brace_class_failed": 0}
        draws: List[Tuple[Any, ...]] = []
        for _ in range(n):
            yaw0, delta = TT.axial_init_yaw(rng, yaw_goal, self.g("deviation_range_deg"))
            r, a = float(rng.uniform(0.0, TT.FOCUS_RADIUS)), float(rng.uniform(0.0, 2 * math.pi))
            xy = manip.focus[:2] + th.tensor([r * math.cos(a), r * math.sin(a)])
            pose = TT.object_pose_on(self, self.hull, C.IDENTITY_QUAT, xy, yaw0, self.top.z)
            if not self.top.polygon_supported(self.world.footprint(self.hull, pose), margin=TT.SUPPORT_MARGIN):
                fails["support"] += 1
                continue
            if not TT.in_frame(self, cam_init, TT.hull_corners_world(self.hull, pose), TT.IN_FRAME_MARGIN):
                fails["frame"] += 1
                continue
            com = self._com_world(pose)
            long_w = C.quat_rotate(pose[1], self.long_axis)
            ends = [com + long_w * s * half_len for s in (1.0, -1.0)]
            d_goal0 = C.wrap_angle(yaw_goal - yaw0)
            spin0 = 1.0 if d_goal0 > 0 else -1.0
            # A1 joint placement per end: the blade's contact point beside the end (right arm, work point) must lie in
            # a right band cell AND the stop point of that end at the goal (left arm, brace point) in a left contact
            # band cell at the manipulation trunk, with the brace's contact class passing when the band records classes
            pairs = []
            for end_sign, e in zip((1.0, -1.0), ends):
                r_xy = th.tensor([float(e[0] - com[0]), float(e[1] - com[1]), 0.0])
                d = spin0 * th.cross(target_z_axis, r_xy, dim=0)
                d = d / th.linalg.norm(d)
                side_pt = self._extreme_point(pose, -d, near=e)  # the face the end moves away from
                contact_xy = side_pt[:2] - d[:2] * push_half
                goal_pose, stop_pts, d_stop = self._stop_geometry(pose, e, com, spin0, yaw_goal, z_brace)
                stop_pt = stop_pts[0]  # the band test uses the corner-side point (the brace candidates run along the face)
                if TT.contact_maps_on():
                    pairs.append((e, self.map_joint_placement(manip, pose, goal_pose, spin0, abs(d_goal0), end_sign=end_sign), goal_pose, stop_pts, d_stop))
                    continue
                jp = TT.joint_placement(bands, left_bands, manip, contact_xy, stop_pt[:2], brace_class=TT.contact_class(manip, -d_stop, target_z_axis), tol_brace=BRACE_CELL_TOL)
                if not jp["ok"] and left_bands is not None and jp["brace_ok"] and not jp["brace_class_ok"]:
                    jp = TT.joint_placement(bands, left_bands, manip, contact_xy, stop_pt[:2], brace_class=TT.contact_class(manip, -d_stop, th.cross(target_z_axis, d_stop, dim=0)), tol_brace=BRACE_CELL_TOL)
                pairs.append((e, jp, goal_pose, stop_pts, d_stop))
            # owner decision 2026-09-16 08:10 (axial = push-only, the stop brace optional): the placement needs the push work
            # point in the right band; a brace is recorded only when the joint two-arm placement exists for that end
            if not any(jp["work_ok"] for _, jp, _, _, _ in pairs):
                fails["work_off_band"] += 1
                continue
            for _, jp, _, _, _ in pairs:
                if jp["work_ok"] and not jp["ok"]:
                    fails["brace_off_band" if not jp["brace_ok"] else "brace_class_failed"] += 1
            draws.append((pose, yaw0, delta, com, pairs))
            if stop_at == "work" or (stop_at == "pair" and self._draw_paired(draws[-1])):
                break
        return draws, fails

    @staticmethod
    def _draw_paired(draw) -> bool:
        """A draw has a PAIR (owner A5): one end whose push and stop brace both fit (``jp["ok"]``)."""
        return any(jp["ok"] and jp["work_ok"] for _, jp, _, _, _ in draw[4])

    def _pick_draw(self, draws):
        """Owner A5 (2026-11-08): the first draw with a pair, else the first push-feasible draw (push-only)."""
        for d in draws:
            if self._draw_paired(d):
                return d
        return draws[0] if draws else None

    def _scan_candidates(self, rng, init: Stance, order: List[int]) -> List[Tuple[int, Any, Dict[str, int]]]:
        """Owner A5 (2026-11-08, under v2only; map lookups only): every manipulation candidate of ``order`` in turn, its placement draws on the
        push_blade / axial_brace maps until a draw has a pair (an end whose push and stop brace fit). Returns ``(k, draw, fails)`` per
        candidate with a push-feasible draw - the candidates with a pair first (in order), then the push-only ones (their first draw)."""
        paired, push_only = [], []
        cam_init = self.camera_pose_at(init)
        z_brace = TT.contact_height(self.top.z, self.extents["height"])
        half_len = 0.5 * self.extents["long"] - END_INSET
        geo0 = self._blade_geometry(0.0, 0.0)
        if geo0 is None:
            return []
        for k in order:
            cand = self.manip_candidates[int(k)]
            manip = cand.with_trunk(cand.translate, cand.tilt, "manipulation")
            manip.focus = cand.focus
            draws, fails = self._placement_draws(rng, manip, cam_init, self._aligned_yaw(manip), geo0[1], z_brace, half_len, stop_at="pair")
            d = self._pick_draw(draws)
            if d is None:
                continue
            (paired if self._draw_paired(d) else push_only).append((int(k), d, fails))
        return paired + push_only

    def axial_maps(self):
        """i7-contact 2026-11-07 (step 3, under v2only): the target's ``push_blade`` map (right arm: REQUIRED, a miss rejects ``missing_map``)
        and ``axial_brace`` map (left arm: OPTIONAL, owner 2026-11-07 item 5 - a miss returns None, the brace step is skipped and the push
        runs by its own rule) at the top (``tabletop.need_contact_map``; both log the ``MAP_REQUEST`` line)."""
        return TT.need_contact_map(self, "push_blade", "target", self.top), TT.need_contact_map(self, "axial_brace", "target", self.top)

    def map_joint_placement(self, manip: Stance, pose: TT.Pose, goal_pose: TT.Pose, spin: float, turn: float, end_sign: Optional[float] = None) -> Dict[str, Any]:
        """The A1 joint placement on the contact maps (``TT.joint_placement``'s dict) for the long-axis END ``end_sign`` (+1: the end along
        ``self.long_axis``) pushed with turn sense ``spin``: the WORK side = the target's root at its init pose in the ``push_blade`` map (a
        feasible arc class of that end and spin at the entry turn nearest ``turn``: the map's chain holds the executed pre, the init AND the
        arc to the goal), the BRACE side = the target's root at its GOAL pose in the ``axial_brace`` map (a feasible stop-brace class of that
        end and spin: the stop face the pushed end arrives at) - instead of the generic band's work cell and the left contact band / reach
        proxy box. i7-contact 2026-11-07: the classes are matched on the END too (the stop face of one end is not the other's; a map whose
        library axis is not ``self.long_axis`` up to sign matches on the spin only); ``end_sign`` None: the spin only."""
        pm, bm = self.axial_maps()
        turns = sorted({abs(float(r["arc_turn_rad"])) for r in pm.meta["classes"] if r and r.get("arc_turn_rad") is not None})
        t_near = min(turns, key=lambda t: abs(t - turn)) if turns else None
        e_push, e_brace = AlignmentAxial._map_end_sign(self, pm, end_sign), (AlignmentAxial._map_end_sign(self, bm, end_sign) if bm is not None else None)

        def same_end(r, e_map):
            e_r, _s = CMAP.axial_end_spin(r)
            return e_map is None or e_r is None or e_r == e_map

        def keep_push(r):
            a = r.get("arc_turn_rad")
            return (a is not None and math.copysign(1.0, float(a)) == spin and (t_near is None or abs(abs(float(a)) - t_near) < 1e-4)
                    and same_end(r, e_push))

        def keep_brace(r):
            _e, s_r = CMAP.axial_end_spin(r)
            return (s_r is None or s_r == spin) and same_end(r, e_brace)

        w = TT.contact_map_check(pm, manip, pose, keep=keep_push)
        if bm is None:  # no axial_brace map: the optional brace step is skipped (push-only), the push placement stands on its own map
            b = {"ok": False, "why": "no axial_brace map: the optional brace step is skipped (MAP_REQUEST logged)"}
        else:
            b = TT.contact_map_check(bm, manip, goal_pose, keep=keep_brace)
        return {"ok": bool(w["ok"] and b["ok"]), "work_ok": bool(w["ok"]), "brace_ok": bool(b["ok"]), "brace_class_ok": True, "left_band": False,
                "brace_map_missing": bm is None,
                "maps": {"push_blade": {k: w.get(k) for k in ("cell", "yaw_slot_deg", "yaw_err_deg", "feasible", "why")}, "turn_class_rad": t_near,
                         "end_sign": end_sign, "end_sign_map": {"push_blade": e_push, "axial_brace": e_brace},
                         "axial_brace": {k: b.get(k) for k in ("cell", "yaw_slot_deg", "yaw_err_deg", "feasible", "why")}},
                # the map's class labels are the library's (no band class names: the pitch order and the brace's passing classes keep their defaults)
                "work_cell": w.get("cell"), "brace_cell": b.get("cell"), "classes": None, "brace_classes": None, "brace_class": None}

    def _map_end_sign(self, cm, end_sign: Optional[float]) -> Optional[float]:
        """``end_sign`` (along ``self.long_axis``) in the map library's convention (its builders' ``long_axis`` param, map meta
        ``library.params``; a map without it: the same icl frame record, as built): flipped for an opposite axis, None (match on the spin
        only) for another axis or no end."""
        if end_sign is None:
            return None
        axes = [p.get("long_axis") for p in ((cm.meta.get("library") or {}).get("params") or {}).values() if isinstance(p, dict) and p.get("long_axis") is not None]
        if not axes:
            return float(end_sign)
        a = th.tensor([float(v) for v in axes[0]], dtype=th.float32)
        d = float(th.dot(a / th.linalg.norm(a), self.long_axis.float() / th.linalg.norm(self.long_axis.float())))
        if abs(d) < 0.9:
            return None
        return float(end_sign) * (1.0 if d > 0 else -1.0)

    def _stop_geometry(self, pose: TT.Pose, end: th.Tensor, com: th.Tensor, spin: float, yaw_goal: float, z_brace: float):
        """§6.10 stop edge of the pushed ``end``: ``(goal_pose, stop_pts, d_stop)`` (``contact_poses.axial_stop_geometry``, the rule the brace
        library shares): the goal pose (COM fixed, yaw = aligned), the direction the end moves in at the goal and the points of the goal hull's
        face along it at the brace height - only the part of that face near the pushed end's arrival is free of the object before the goal,
        so the candidates run from that corner toward the far end."""
        return CPZ.axial_stop_geometry(self.hull, self.com_local, pose, end, com, spin, yaw_goal, z_brace)

    def _stop_face_points(self, goal_pose: TT.Pose, d_stop: th.Tensor, near: th.Tensor, z: float, fractions=(0.08, 0.2, 0.35, 0.5, 0.65, 0.8)) -> List[th.Tensor]:
        """``contact_poses.axial_stop_face_points``: the goal hull's stop face points at ``fractions`` of its span from the end nearest ``near``."""
        return CPZ.axial_stop_face_points(self.hull, goal_pose, d_stop, near, z, fractions)

    def _face_z_range(self) -> Tuple[float, float]:
        """World z range of the target's side faces (standing on the top)."""
        return float(self.top.z), float(self.top.z) + float(self.extents["height"])

    def _blade_geometry(self, pitch: float, clearance: float, side: bool = False) -> Optional[Tuple[float, float, float]]:
        """``(stand-off, finger half thickness, tip z)`` of a top-down blade beside the target's face at the push
        height for the approach ``pitch`` (deg): the fingertips sit ``FINGERTIP_LENGTH·cos(pitch)`` below the eef
        origin; the half thickness is the finger profile's maximum over the face height above the tips (``tabletop.
        blade_standoff``). None when the face reaches the palm section."""
        return CPZ.axial_blade_geometry(self._push_height(), self._face_z_range(), pitch, clearance, side=side)  # i7-contact: the shared rule

    def _brace_candidates(self, manip: Stance, stop_pts: List[th.Tensor], d_stop: th.Tensor, passing: Optional[List[str]], pose_init: TT.Pose, com: th.Tensor, d_goal: float):
        """The stop brace's poses of the closed left gripper at the stop face (``contact_poses.axial_brace_candidates``, the rule the brace
        library shares): the pads on the face (closing ``_v`` / ``_h``, the left band's passing classes first) and the top-down blade beside it,
        each clear of the target's sweep from the init pose to the goal; thin targets (< 4 cm) get the blade first. Yields ``(pre, contact,
        kind, extras)``; the skips counted in ``self._blade_skips``."""
        if getattr(self, "_blade_skips", None) is None:
            self._blade_skips = {}
        yield from CPZ.axial_brace_candidates(manip, self.hull, stop_pts, d_stop, passing, pose_init, com, d_goal, z_push=self._push_height(),
                                              face_z=self._face_z_range(), height=self.extents["height"], stop_clearance=STOP_CLEARANCE,
                                              thin_height=THIN_TARGET_HEIGHT, skips=self._blade_skips)

    def _extreme_point(self, pose: TT.Pose, direction: th.Tensor, near: Optional[th.Tensor] = None) -> th.Tensor:
        """Hull point (world) with the largest projection on ``direction`` (xy); with ``near`` only points within 4 cm of
        that end along the long axis are considered (the face at the pushed end)."""
        long_w = C.quat_rotate(TT._t(pose[1]), self.long_axis)[:2] if near is not None else None
        return CPZ.extreme_point(C.transform_points(pose, self.hull), direction, near=near, long_w=long_w)  # i7-contact: the shared rule

    # ------------------------------------------------------------------ program check (push-only sidecars)

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        """§3.5 program check. A push-only axial (no ``contacts.brace``, owner decision 2026-09-16 08:10) is not yet accepted
        by the F6 program / ``REQUIRED_BLOCKS`` (the oracle agent is making the brace optional); until that lands, with
        ``ISG_PROGRAM_CHECK=stub`` set inside the smoke command (``tabletop.program_check_stub``) a rejection that only asks
        for the brace block is recorded as ``mode: stub-push-only`` instead of rejecting the instance (the default stays
        strict; no repo markers)."""
        try:
            return super().program_check(draft)
        except Reject as e:
            msg = str(e)
            brace_only = getattr(self, "push_only", False) and "brace" in msg and TT.program_check_stub()
            if not brace_only:
                raise
            log.warning("%s: program check asks for contacts.brace on a push-only axial (%s); recorded as stub-push-only (ISG_PROGRAM_CHECK=stub)", self.gp.variant, msg[:200])
            report = {"mode": "stub-push-only", "problems": [], "note": "brace-less axial not yet accepted by the oracle program; structural check skipped", "error": msg[:300]}
            self.last_program_check = report
            draft.summary["program_check"] = report
            return report

    # ------------------------------------------------------------------ goal / hinge (§6.10)

    def goal(self, draft: Draft):
        manip = self.sidecar.stance("manipulation")
        self.sidecar.set_goal(manip.base_pose2d, manip.translate, manip.tilt, arms="same")

    def _clear_hinge(self):
        remove_hinge(self.hinge)
        self.hinge = None

    def hinge_resistance(self) -> Optional[Dict[str, Any]]:
        """§6.10 hinge friction / damping (owner 2026-10-12, alignment-axial#05: the tablespoon turned 0.022 rad on its
        frictionless hinge while settling). None - the frictionless hinge - unless the variant's own ``generation`` sets one of
        :data:`HINGE_RESISTANCE_KEYS` (``generation_defaults`` has none for the axial; a key the variant leaves out takes the world
        vertical's default). Then the wall's helper :func:`ispatialgym.sampling.constraints.hinge_resistance` with the inertia
        about the vertical hinge axis: the spawned target's mass over its footprint (the template's long × short extents)."""
        keys = {k: self.gen(k) for k in HINGE_RESISTANCE_KEYS}
        if all(v is None for v in keys.values()):
            return None
        mass = float(self.obj("target").mass)
        if not (mass > 0):
            raise Reject("spec", f"{self.tname}: the spawned target reports mass {mass} (the hinge resistance needs its inertia)")
        decel, tau, omega = (float(d if keys[k] is None else keys[k]) for k, d in zip(HINGE_RESISTANCE_KEYS, (HINGE_STOP_DECEL_RAD_S2, HINGE_VISCOUS_TAU_S, HINGE_BRAKE_OMEGA_RAD_S)))
        return hinge_resistance(mass, (self.extents["long"], self.extents["short"]), decel=decel, tau=tau, omega_brake=omega)

    def hinge_anchor(self, pose: TT.Pose) -> th.Tensor:
        """§6.10 hinge anchor of the target at ``pose``: on the vertical axis through its COM, ``hinge_sink_m`` below the support top."""
        com = self._com_world(pose)
        return th.tensor([float(com[0]), float(com[1]), float(self.top.z) - float(self.g("hinge_sink_m"))])

    def after_settle(self, draft: Draft):
        """Install the hinge on the settled target (anchor: :meth:`hinge_anchor`; friction / damping from :meth:`hinge_resistance`),
        require drift <= 1 mm / 0.005 rad over 60 steps, record the TRO constraint and the goal pose (rotation about the COM to
        the aligned yaw). The target is not moved (owner 2026-10-12, alignment-axial#05: lowered ``hinge_sink_m`` into the
        support and hinged at its COM, the tablespoon popped back up tilted 0.015-0.024 rad and failed the drift check)."""
        target, support = self.obj("target"), self.obj("support")
        tname = self.tname
        self._clear_hinge()
        target.keep_still()
        p, q = target.get_position_orientation()
        anchor = self.hinge_anchor((p.float(), q.float()))
        res = self.hinge_resistance()
        self.hinge = create_hinge(target, anchor, th.tensor([0.0, 0.0, 1.0]), parent_link=support.root_link, friction=None if res is None else res["friction"],
                                  damping=None if res is None else res["damping"],
                                  facts=None if res is None else {k: v for k, v in res.items() if k not in ("friction", "damping")})
        target.keep_still()
        og.sim.step_physics()
        drift = self.g("hinge_drift_max") or {}
        p0, q0 = target.get_position_orientation()
        self.world.step_physics(int(drift.get("steps", 60)), still=[self.robot])
        p1, q1 = target.get_position_orientation()
        moved, turned = float(th.linalg.norm(p1 - p0)), C.quat_angle(q1, q0)
        if res is not None:
            log.info("%s: hinge friction %.4g N·m, damping %.4g N·m·s/rad (I_z %.4g kg·m², %.4g kg): drifted %.1f mm / %.4f rad over %d steps", self.gp.variant,
                     res["friction"], res["damping"], res["inertia_kgm2"], res["mass_kg"], moved * 1000, turned, int(drift.get("steps", 60)))
        if moved > float(drift.get("position_m", 0.001)) or turned > float(drift.get("orientation_rad", 0.005)):
            raise Reject("hinge_drift", f"target drifted {moved * 1000:.1f} mm / {turned:.4f} rad with the hinge")
        draft.extras["constraints"] = [self.hinge.to_tro(tname, self.bindings.one("support").name)]
        if res is not None:
            draft.summary["hinge"] = {"friction_nm": res["friction"], "damping_nms_per_rad": res["damping"], "inertia_kgm2": res["inertia_kgm2"], "mass_kg": res["mass_kg"]}
        draft.object_poses[tname] = (p1.clone(), q1.clone())
        yaw_goal = draft.summary["yaw_goal"]
        q_goal = C.quat_from_yaw(yaw_goal)
        com = self._com_world((p1.float(), q1.float()))
        draft.goal_poses[tname] = (com - C.quat_rotate(q_goal, self.com_local), q_goal)

    def refine_targets_from_goal(self, draft: Draft):
        pass  # orientation-only target: the goal orientation is exact by construction

    def on_reject(self, err: Reject):
        self._clear_hinge()
        super().on_reject(err)

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        path = super().write_instance(draft, index, seed)
        self._clear_hinge()
        return path

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        push = self.sidecar.contacts.get("push")
        manip = self.sidecar.stance("manipulation")
        if push is None:
            return []
        mc = self.choice.models["target"]
        return [{"scope": {"target": f"{mc.category}/{mc.model}", "support_height": round(float(self.top.z - self.world.floor_z), 2)}, "type": "contacts",
                 "trunk": [manip.translate, round(manip.tilt, 2)], "push": {"end": "long_axis", "distance_m": round(float(push.distance_m), 3)},
                 "brace": "stop_edge", "hard": False}]


def _icl_frame(category: str, model: str):
    """``(frame_front, frame_long)`` of an axial target from ``icl/roles/candidates.csv`` (None, None when absent)."""
    import csv

    from ispatialgym.sampling.v2.problem import DEFAULT_ICL_ROOT

    path = os.path.join(DEFAULT_ICL_ROOT, "roles", "candidates.csv")
    if not os.path.exists(path):
        return None, None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("subtask") == SUBTASK and row.get("role") == "target" and row.get("category") == category and row.get("model") == model:
                return (row.get("frame_front") or None), (row.get("frame_long") or None)
    return None, None


def _com_local(obj) -> th.Tensor:
    """Centre of mass in the root frame (mass-weighted link COMs from PhysX; the AABB centre when unavailable)."""
    try:
        total, acc = 0.0, th.zeros(3)
        rp, rq = obj.get_position_orientation()
        for link in obj.links.values():
            m = float(link.mass)
            if m <= 0:
                continue
            lp, lq = link.get_position_orientation()
            acc += m * (lp.float() + C.quat_rotate(lq.float(), TT._t(link.center_of_mass)))
            total += m
        if total > 0:
            return C.quat_rotate(C.qinv(rq.float()), acc / total - rp.float())
    except Exception:  # noqa: BLE001
        pass
    lo, hi = obj.aabb
    rp, rq = obj.get_position_orientation()
    return C.quat_rotate(C.qinv(rq.float()), (lo + hi) / 2 - rp.float())
