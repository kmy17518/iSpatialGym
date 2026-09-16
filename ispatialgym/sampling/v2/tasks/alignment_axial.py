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
the stop, ``summary.brace = "present" | "none"``); every right-hand push pose IK-feasible. After settling a vertical hinge through the COM (parent = support,
anchor 1 mm below the top) is installed and recorded as a TRO constraint (drift ≤ 1 mm / 0.005 rad over 60 steps).
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
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling.constraints import Hinge, create_hinge, remove_hinge
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import pose_target
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, Stance

import omnigibson as og  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASK = "alignment-axial"
AXES = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0), "+z": (0.0, 0.0, 1.0), "-z": (0.0, 0.0, -1.0)}
#: closed fingers: the blade half thickness beside a face comes from the measured finger profile (``tabletop.
#: FINGER_PROFILE``: 8.4 mm over the fingertip section, 18 mm above it, no blade at the palm) over the height the face
#: spans above the fingertips — the former 12 mm slab is gone (owner review of the axial demo, 2026-09-16)
STOP_CLEARANCE = 0.005  # §5.5 style: the blade's outer face 5 mm outside the goal hull
PUSH_TIP_CLEARANCE = 0.003  # the closed fingertips this far above the top for the thinnest targets (push / blade stop)
THIN_TARGET_HEIGHT = 0.04  # below this height the stop brace tries the top-down blade before the side contact classes
#: the stop point may lie within one lattice cell of a left contact band cell (the blade stands beside the cell; the left band
#: of a flat target starts at λ = 0.08 on the pilot support while the right band ends at 0.0 — the exact two-arm IK decides)
BRACE_CELL_TOL = 1.0
END_INSET = 0.03  # the pushed end lies this far inside the object's long-axis extremity
BAND_TOLERANCE = 0.5  # an end is "inside the band" within this fraction of the lattice of a band cell


#: smoke-only switch for the brace-less program check (see ``AlignmentAxial.program_check``): ``ISG_PROGRAM_CHECK=stub`` or,
#: for pod runs whose launcher cannot pass an environment, a marker file under the git-ignored, pod-synced
#: ``runlogs/smoke_inputs/`` containing ``alignment-axial``
STUB_MARKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "runlogs", "smoke_inputs", "program_check_stub.txt")


def _program_check_stub() -> bool:
    if os.environ.get("ISG_PROGRAM_CHECK", "").lower() == "stub":
        return True
    try:
        with open(STUB_MARKER) as f:
            return SUBTASK in f.read()
    except OSError:
        return False


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
            raise Reject("spec", f"{self.gp.alias}: scenes.{self.scene}.bindings.support must list scene object names")
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
                        log.info("%s: edge %s lateral %.2f t %.2f: reference edge %d/%d points visible from init (%d in frame)", self.gp.alias, edge.name, cand.lateral,
                                 st_try.translate, n_vis, int(ev.get("points", 9)), n_in)
                    if init is not None:
                        break
                if init is None:
                    raise Reject("edge_not_visible", "the front edge is not visible from the init stance at 8 lateral draws")
                cands = TT.phase2_manipulation_stance(self, init, w_diag, target_height=ez, bands=bands, top=self.top, focus_z=self.top.z + 0.5 * ez)
                cands = [c for c in cands if self._edge_visible_points(c, self.camera_pose_at(c), c.local(c.focus[:2])[1], None) >= int(ev.get("min_visible", 3))]
                if not cands:
                    raise Reject("edge_not_visible", "no Phase-2 survivor sees the front edge")
                self.edge_frame, self.init_stance, self.manip_candidates = edge, init, cands
                break
            except Reject as e:
                errors.append(f"edge {edge.name}: {e}")
                log.info("%s: %s", self.gp.alias, errors[-1][:300])
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
        return float(self.top.z) + max(0.6 * self.extents["height"], C.FINGERTIP_LENGTH + PUSH_TIP_CLEARANCE)

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
        for k in order[:6]:
            cand = self.manip_candidates[int(k)]
            manip = cand.with_trunk(cand.translate, cand.tilt, "manipulation")
            manip.focus, manip.arms = cand.focus, ["right"]  # the left arm joins only when a stop brace is recorded
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
        draft = Draft()
        tname = self.tname
        target, support = self.obj("target"), self.obj("support")
        self.record_transition(init, manip)
        yaw_goal = self._aligned_yaw(manip)
        cam_init = self.camera_pose_at(init)
        ev = self.g("edge_visibility") or {}
        bands, left_bands = self.bands, self.left_bands
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
        chosen = None
        fails = {"support": 0, "frame": 0, "work_off_band": 0, "brace_off_band": 0, "brace_class_failed": 0}
        for _ in range(12):
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
            for e in ends:
                r_xy = th.tensor([float(e[0] - com[0]), float(e[1] - com[1]), 0.0])
                d = spin0 * th.cross(z_axis, r_xy, dim=0)
                d = d / th.linalg.norm(d)
                side_pt = self._extreme_point(pose, -d, near=e)  # the face the end moves away from
                contact_xy = side_pt[:2] - d[:2] * push_half
                goal_pose, stop_pts, d_stop = self._stop_geometry(pose, e, com, spin0, yaw_goal, z_brace)
                stop_pt = stop_pts[0]  # the band test uses the corner-side point (the brace candidates run along the face)
                jp = TT.joint_placement(bands, left_bands, manip, contact_xy, stop_pt[:2], brace_class=TT.contact_class(manip, -d_stop, z_axis), tol_brace=BRACE_CELL_TOL)
                if not jp["ok"] and left_bands is not None and jp["brace_ok"] and not jp["brace_class_ok"]:
                    jp = TT.joint_placement(bands, left_bands, manip, contact_xy, stop_pt[:2], brace_class=TT.contact_class(manip, -d_stop, th.cross(z_axis, d_stop, dim=0)), tol_brace=BRACE_CELL_TOL)
                pairs.append((e, jp, goal_pose, stop_pts, d_stop))
            # owner decision 2026-09-16 08:10 (axial = push-only, the stop brace optional): the placement needs the push work
            # point in the right band; a brace is recorded only when the joint two-arm placement exists for that end
            if not any(jp["work_ok"] for _, jp, _, _, _ in pairs):
                fails["work_off_band"] += 1
                continue
            for _, jp, _, _, _ in pairs:
                if jp["work_ok"] and not jp["ok"]:
                    fails["brace_off_band" if not jp["brace_ok"] else "brace_class_failed"] += 1
            chosen = (pose, yaw0, delta, com, pairs)
            break
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
        push = brace_found = None
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
                for c_axis, c_sign in ((d, 1.0), (d, -1.0), (side_c, 1.0), (side_c, -1.0)):
                    # the closed fingers' half thickness beside the pushed face at this pitch: the pad face (closing axis
                    # along the push) or the finger's side (closing axis across it), from the measured finger profile
                    geo = self._blade_geometry(pitch, 0.0, side=c_axis is side_c)
                    if geo is None:
                        push_fails.append(f"{pitch:.0f}°: face reaches the palm section")
                        continue
                    push_half_p = geo[1]
                    quat = C.eef_quat(a_vec, c_sign * c_axis)
                    p_init = (side_pt - d * push_half_p, quat)
                    arc = C.arc_waypoints(p_init, th.tensor([float(com[0]), float(com[1]), z_push]), z_axis, d_goal, n=2)
                    p_mid, p_goal = arc[0], arc[1]
                    p_pre = C.pre_pose(p_init)
                    res = self.ikc.waypoints_ok([p_pre, p_init, p_mid, p_goal], manip, "right", ignore_all=[support, target])
                    if not res.ok:
                        push_fails.append(f"{pitch:.0f}°: {res.reason}")
                        continue
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
                            for brace_pre, brace, kind, extras in self._brace_candidates(manip, stop_pts, d_stop, jp.get("brace_classes"), pose, com, d_goal):
                                res_b = TT.brace_second_arm(self, manip, "right", [p_pre, p_goal], "left", [brace_pre, brace], ignore_work_last=[support, target],
                                                            ignore_brace_last=[support], brace_only_check=([support], TT.BRACE_TARGET_MARGIN))
                                if res_b.ok:
                                    brace_found = (brace_pre, brace, kind, {**extras, "push_half_thickness_m": push_half_p, "target_checked_at": "goal"})
                                    break
                                brace_fails.append(f"{kind}: {res_b.reason}")
                        finally:
                            target.set_position_orientation(*pose)
                            target.keep_still()
                            self.kit.sync_world()
                        brace_reason = "present" if brace_found is not None else ("brace_ik_failed" if brace_fails else "brace_sweep_blocked")
                    else:
                        brace_reason = "no_joint_cell" if not jp["brace_ok"] else "brace_class_failed"
                    dist = abs(d_goal) * float(th.linalg.norm(r_xy))
                    push = (p_pre, p_init, p_mid, p_goal, d, dist, e, goal_pose, stop_pts, d_stop, pitch)
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
        self.sidecar.add_contact(ContactRecord("push", "right", p_pre, p_init, p_mid, p_goal, d_push, dist, object=tname))
        if brace_found is not None:
            brace_pre, brace, brace_kind, brace_extras = brace_found
            self.sidecar.add_contact(TT.BraceRecord("brace", "left", brace_pre, brace, direction=-d_stop, brace_kind="stop_edge", object=tname,
                                                    extras={k: (float(v) if isinstance(v, (int, float)) else v) for k, v in brace_extras.items()}))
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
                          "joint_placement": {"left_band": self.left_bands is not None, "fails": fails}})
        self.sidecar.describe(tname, mc.category.replace("_", " "), f"the {mc.category.replace('_', ' ')} on the table")
        self.sidecar.language["goal_numbers"]["rotate_deg"] = math.degrees(d_goal)
        return draft

    def _stop_geometry(self, pose: TT.Pose, end: th.Tensor, com: th.Tensor, spin: float, yaw_goal: float, z_brace: float):
        """§6.10 stop edge of the pushed ``end``: the goal pose (COM fixed, yaw = aligned), the direction the end moves in
        at the goal (``d_stop``) and the stop point = the goal hull's extreme point along it + 5 mm clearance at the
        brace height. Returns ``(goal_pose, stop_pt, d_stop)``."""
        z_axis = th.tensor([0.0, 0.0, 1.0])
        q_goal = C.quat_from_yaw(yaw_goal)
        goal_pose = (com - C.quat_rotate(q_goal, self.com_local), q_goal)
        r_goal = C.quat_rotate(q_goal, C.quat_rotate(C.qinv(pose[1]), end - TT._t(pose[0]))) + goal_pose[0] - com
        d_stop = spin * th.cross(z_axis, th.tensor([float(r_goal[0]), float(r_goal[1]), 0.0]), dim=0)
        d_stop = d_stop / th.linalg.norm(d_stop)
        # the stop face = the goal hull's extreme along d_stop (a whole long face for a box). Only the part of that face
        # near the pushed end's arrival is free of the object before the goal (the rest of the face rises through the
        # blade's position while the object turns: ``tabletop.blade_sweep_clear`` — the pilot's blocked demo had its
        # blade on the swept part). Candidates run from 2 cm inside that corner toward the far end; the first one whose
        # blade clears the sweep is used. Points are ON the goal hull's face (no clearance): every brace pose adds its own
        # stand-off along d_stop. Returns ``(goal_pose, stop_pts (nearest the pushed end first), d_stop)``.
        stop_pts = self._stop_face_points(goal_pose, d_stop, near=com + r_goal, z=z_brace)
        return goal_pose, stop_pts, d_stop

    def _stop_face_points(self, goal_pose: TT.Pose, d_stop: th.Tensor, near: th.Tensor, z: float, fractions=(0.08, 0.2, 0.35, 0.5, 0.65, 0.8)) -> List[th.Tensor]:
        """Points of the goal hull's face extreme along ``d_stop`` (projection within 3 mm of the maximum) at the given
        fractions of the face's span from the end nearest ``near`` (the pushed end's arrival) to the far end, at height
        ``z``; a face shorter than 6 cm yields its midpoint only."""
        pts = C.transform_points(goal_pose, self.hull)
        dxy = th.tensor([float(d_stop[0]), float(d_stop[1])])
        dxy = dxy / th.linalg.norm(dxy)
        proj = pts[:, :2] @ dxy
        face = pts[proj >= float(proj.max()) - 0.003]
        f = TT._t(near)[:2]
        dist = th.linalg.norm(face[:, :2] - f, dim=1)
        p_near, p_far = face[int(th.argmin(dist))], face[int(th.argmax(dist))]
        span = float(th.linalg.norm(p_far[:2] - p_near[:2]))
        if span < 0.06:
            mid = 0.5 * (p_near + p_far)
            return [th.tensor([float(mid[0]), float(mid[1]), float(z)])]
        out = []
        for t in fractions:
            p = p_near + (p_far - p_near) * t
            out.append(th.tensor([float(p[0]), float(p[1]), float(z)]))
        return out

    def _face_z_range(self) -> Tuple[float, float]:
        """World z range of the target's side faces (standing on the top)."""
        return float(self.top.z), float(self.top.z) + float(self.extents["height"])

    def _blade_geometry(self, pitch: float, clearance: float, side: bool = False) -> Optional[Tuple[float, float, float]]:
        """``(stand-off, finger half thickness, tip z)`` of a top-down blade beside the target's face at the push
        height for the approach ``pitch`` (deg): the fingertips sit ``FINGERTIP_LENGTH·cos(pitch)`` below the eef
        origin; the half thickness is the finger profile's maximum over the face height above the tips (``tabletop.
        blade_standoff``). None when the face reaches the palm section."""
        z_push = self._push_height()
        z_tip = z_push - C.FINGERTIP_LENGTH * math.cos(math.radians(float(pitch)))
        so = TT.blade_standoff(self._face_z_range(), z_tip, pitch, clearance, side=side)
        if so is None:
            return None
        return so[0], so[1], z_tip

    def _brace_candidates(self, manip: Stance, stop_pts: List[th.Tensor], d_stop: th.Tensor, passing: Optional[List[str]], pose_init: TT.Pose, com: th.Tensor, d_goal: float):
        """Brace poses of the closed left gripper at the stop face (``stop_pts`` on the goal hull, the pushed end's
        corner side first), in the order to try: B3's contact classes with the pads on the face (approach −d_stop into
        the face, pad centre ``STOP_CLEARANCE`` outside it; closing axis vertical ``_v`` then horizontal ``_h``; the
        classes the left band recorded as passing at the cell first), and the top-down blade beside the stop face
        (straight down / pitched, either wrist turn) whose eef origin stands ``finger half thickness (profile over the
        face height above the tips) + STOP_CLEARANCE`` outside the face and which stays clear of the target's footprint
        over the whole rotation from the init pose to the goal (``tabletop.blade_sweep_clear``; a point of the face that
        the turning object sweeps through is skipped). Yields ``(pre, contact, kind, extras)``; thin targets (< 4 cm) get
        the blade first (a side contact at ¾ of their height puts the wrist on the table)."""
        z_axis = th.tensor([0.0, 0.0, 1.0])
        a = -d_stop  # into the stop face
        side_h = C.unit(th.cross(z_axis, d_stop, dim=0))
        contact_classes = []
        for c_axis, suffix in ((z_axis, "v"), (side_h, "h")):
            name = TT.contact_class(manip, a, c_axis)
            contact_classes.append((name or f"side_{suffix}", c_axis))
        if passing:
            contact_classes.sort(key=lambda nc: 0 if nc[0] in passing else 1)
        r_side = TT.finger_half_thickness(0.0, side=True) or 0.007

        def side_contacts():
            for i, stop_pt in enumerate(stop_pts):
                point = stop_pt + d_stop * STOP_CLEARANCE  # pad centre 5 mm outside the goal face
                # the pads' footprint (a finger's width across, a finger's thickness deep) must be clear of the sweep too
                ok, gap = TT.blade_sweep_clear(self.hull, pose_init, com[:2], d_goal, (point + d_stop * r_side)[:2], r_side, STOP_CLEARANCE)
                if not ok:
                    self._blade_skips["sweep"] = self._blade_skips.get("sweep", 0) + 1
                    continue
                for name, c_axis in contact_classes:
                    for c_sign in (1.0, -1.0):  # either wrist turn of the same pad contact
                        brace = C.contact_eef(point, a, c_sign * c_axis)
                        yield C.pre_pose(brace), brace, f"{name}{'' if c_sign > 0 else '-'}", {"stop_clearance_m": STOP_CLEARANCE, "pose_class": name,
                                                                                               "stop_face_fraction": i, "sweep_min_gap_m": gap}

        def blades():
            z_push = self._push_height()
            for pitch in TT.approach_pitch_order(None):
                geo = self._blade_geometry(pitch, STOP_CLEARANCE)
                if geo is None:
                    self._blade_skips["palm"] = self._blade_skips.get("palm", 0) + 1
                    continue
                standoff, half, z_tip = geo
                a_vec = TT.pitched_approach(pitch, manip.forward)
                for i, stop_pt in enumerate(stop_pts):
                    origin = th.tensor([float(stop_pt[0]), float(stop_pt[1]), z_push]) + d_stop * standoff
                    # the blade must not meet the rotating target before the goal (the face rising through it, a corner)
                    ok, gap = TT.blade_sweep_clear(self.hull, pose_init, com[:2], d_goal, origin[:2], max(half, r_side), STOP_CLEARANCE)
                    if not ok:
                        self._blade_skips["sweep"] = self._blade_skips.get("sweep", 0) + 1
                        continue
                    for c_sign in (1.0, -1.0):
                        brace = (origin, C.eef_quat(a_vec, c_sign * d_stop))
                        yield C.pre_pose(brace), brace, f"blade_{pitch:.0f}{'' if c_sign > 0 else '-'}", {
                            "blade_half_thickness_m": half, "stop_clearance_m": STOP_CLEARANCE, "blade_standoff_m": standoff, "fingertip_z": z_tip,
                            "sweep_min_gap_m": gap, "pose_class": f"blade_{pitch:.0f}", "stop_face_fraction": i}

        order = (blades, side_contacts) if self.extents["height"] < THIN_TARGET_HEIGHT else (side_contacts, blades)
        for gen in order:
            yield from gen()

    def _extreme_point(self, pose: TT.Pose, direction: th.Tensor, near: Optional[th.Tensor] = None) -> th.Tensor:
        """Hull point (world) with the largest projection on ``direction`` (xy); with ``near`` only points within 4 cm of
        that end along the long axis are considered (the face at the pushed end)."""
        pts = C.transform_points(pose, self.hull)
        dxy = th.tensor([float(direction[0]), float(direction[1])])
        dxy = dxy / th.linalg.norm(dxy)
        if near is not None:
            long_w = C.quat_rotate(TT._t(pose[1]), self.long_axis)[:2]
            along = (pts[:, :2] - TT._t(near)[:2]) @ long_w
            sel = along.abs() <= 0.04
            if int(sel.sum()) >= 3:
                pts = pts[sel]
        proj = pts[:, :2] @ dxy
        i = int(th.argmax(proj))
        return pts[i]

    # ------------------------------------------------------------------ program check (push-only sidecars)

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        """§3.5 program check. A push-only axial (no ``contacts.brace``, owner decision 2026-09-16 08:10) is not yet accepted
        by the F6 program / ``REQUIRED_BLOCKS`` (the oracle agent is making the brace optional); until that lands, with
        ``ISG_PROGRAM_CHECK=stub`` set by the caller a rejection that only asks for the brace block is recorded as
        ``mode: stub-push-only`` instead of rejecting the instance (smoke runs only — the default stays strict)."""
        try:
            return super().program_check(draft)
        except Reject as e:
            msg = str(e)
            brace_only = getattr(self, "push_only", False) and "brace" in msg and _program_check_stub()
            if not brace_only:
                raise
            log.warning("%s: program check asks for contacts.brace on a push-only axial (%s); recorded as stub-push-only (ISG_PROGRAM_CHECK=stub)", self.gp.alias, msg[:200])
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

    def after_settle(self, draft: Draft):
        """Install the hinge at the settled COM (anchor 1 mm below the top), require drift <= 1 mm / 0.005 rad over 60
        steps, record the TRO constraint and the goal pose (rotation about the COM to the aligned yaw)."""
        target, support = self.obj("target"), self.obj("support")
        tname = self.tname
        self._clear_hinge()
        sink = float(self.g("hinge_sink_m"))
        p, q = target.get_position_orientation()
        target.set_position_orientation(p - th.tensor([0.0, 0.0, sink]), q)
        target.keep_still()
        anchor = self._com_world((target.get_position_orientation()[0].float(), q.float()))
        self.hinge = create_hinge(target, anchor, th.tensor([0.0, 0.0, 1.0]), parent_link=support.root_link)
        target.keep_still()
        og.sim.step_physics()
        drift = self.g("hinge_drift_max") or {}
        p0, q0 = target.get_position_orientation()
        self.world.step_physics(int(drift.get("steps", 60)), still=[self.robot])
        p1, q1 = target.get_position_orientation()
        if float(th.linalg.norm(p1 - p0)) > float(drift.get("position_m", 0.001)) or C.quat_angle(q1, q0) > float(drift.get("orientation_rad", 0.005)):
            raise Reject("hinge_drift", f"target drifted {float(th.linalg.norm(p1 - p0)) * 1000:.1f} mm / {C.quat_angle(q1, q0):.4f} rad with the hinge")
        draft.extras["constraints"] = [self.hinge.to_tro(tname, self.bindings.one("support").name)]
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
