"""Configuration matching — block design (spec §6.5; init per §4 with the block bullets of §4.5 / §4.6).

Roles: ``targets`` (group of 4 identical blocks, ``generation.groups.targets = {size: 4, draw: same_model}``; the v1
facade names the role ``blocks``), scene ``support``, ``floor``.

Template: the block hull / edge / face colours, the generic-pinch reach band (§4.1 "objects without library grasps"),
Phases 1a / 1b (H_max = edge) / 1c (row ``block_design``: |λ_min| + edge/2 + pitch + 0.02 right, max(0, λ_max) + edge/2
+ 0.02 left) and Phase 2 with **regions** as focus candidates (§4.5: >= n_blocks + 2 band cells at spacing edge·√2 +
0.03 around the focus, the 2 x 2 goal grid (pitch = edge + 1 mm) on the top, the region in frame (8 %)).

Instance (§4.6 blocks, §6.5): one block per region cell drawn at random (>= 3 cm gaps); the anchor = the top-left block
in the init image (its position stays); the pattern's faces are assigned to the slots (anchor = slot 0, the others in
image order); per block: a distinct up face that is **never the opposite of its goal face** (flip rule), the visible-face
rule (a block whose goal face is up shows it on top; a flipping block gets the coarse 90° yaw that turns its goal face
toward the robot), yaw jitter, >= 50 % of every goal face visible from the init camera (3 x 3 face points ray-tested
against the other blocks and the scene); the **cube grasp set** (pinch across a face pair ∥ the flip axis, place tilt θ
= 45° [pilot] about the pinch axis, pick tilt θ − 90·sign — the sign with the downward approach; top-down pinch without
a flip), screened (12 seeds) and confirmed (32 seeds) for pick **and** place in placement order with the already-placed
blocks at their slots, the flip waypoints (§3.9) with the block attached; the row-left / column-top **brace points**
(closed left gripper on the placed row / column, pre-contact + contact) with the right-hand pushes they brace for,
IK-checked together (§4.6 step 6). Goal: the 2 x 2 grid at the anchor aligned with the table axes (``layout_reached``
gates, ``pose_reached`` measured; ``assignment.cost = layout``). Sidecar: ``init`` / ``manipulation`` stances, the
transition, ``blocks[]`` (order, face_pair, tilts, yaw, flip axis / sign, pick, place, carry_height, brace / push
records, flip waypoints).
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
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, Stance, pose_to_sidecar

from omnigibson.utils.asset_utils import get_dataset_path  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASK = "configuration_matching-block_design"
DEFAULT_FACES = {"+z": "red", "-z": "blue", "+x": "white", "-x": "yellow", "+y": "blue|yellow", "-y": "red|white"}
#: approach tilt below the horizontal of the push / brace contacts on a block face (the pads stay above the table)
CONTACT_TILT_DEG = 30.0
#: closed fingers: half thickness of the blade beside a block face (eef origin this far outside the touched face)
FINGER_HALF_THICKNESS = 0.012
#: the place puts the block this far right and toward the robot of its slot; the pushes close the margin (§6.2 "small safety margin")
PLACE_MARGIN = 0.005
REGION_RADIUS = 0.20


@register(SUBTASK)
class BlockDesign(SamplerV2):
    def __init__(self, ctx):
        super().__init__(ctx)
        if self.gp.bddl is None:
            self.definition_id = 0  # v1 writer: configuration_matching-0
        self.defaults = GD.defaults_for(SUBTASK)
        self.role = "targets" if "targets" in self.gp.roles else "blocks"
        self.top = None
        self.hull: Optional[th.Tensor] = None
        self.edge = C.BLOCK_EDGE
        self.faces = dict(DEFAULT_FACES)
        self.edge_frame: Optional[EdgeFrame] = None
        self.init_stance: Optional[Stance] = None
        self.manip_candidates: List[Stance] = []
        self.band_spec: Optional[TT.BandSpec] = None
        self.bands = None
        self.left_bands = None  # A1: the left arm's contact band (row-left / column-top braces), None until B3's bands exist
        self._regions_off_left_band = 0

    def g(self, key: str):
        return self.gen(key, self.defaults.get(key))

    # ------------------------------------------------------------------ roles

    def v1_role_aliases(self) -> Dict[str, str]:
        return {"blocks": self.role} if self.role != "blocks" else {}

    def bind_scene_roles(self):
        self.bind_floor_v2()
        req = self.choice.bindings.get("support")
        names = [n for n in (req if isinstance(req, list) else []) if isinstance(n, str)]
        if not names:
            raise Reject("spec", f"{self.gp.alias}: scenes.{self.scene}.bindings.support must list scene object names")
        self.bind_scene_object("support", self.world.obj(names[0]))

    def n_blocks(self) -> int:
        g = self.g("grid") or [2, 2]
        return int(g[0]) * int(g[1])

    def spawn_roles(self):
        if self.role in self.choice.groups:
            super().spawn_roles()
            return
        # v1 facade: the block role is a single sampleable role; spawn the group's size copies of its model
        choice = self.choice.models.get(self.role)
        if choice is None:
            raise Reject("spec", f"{self.gp.alias}: no block model in the template choice")
        for _ in range(self.n_blocks()):
            self.spawn_role(self.role, choice)

    # ------------------------------------------------------------------ template

    def names(self) -> List[str]:
        return [b.name for b in self.bindings.many(self.role)]

    def sample_template(self):
        blocks = self.objs(self.role)
        if len(blocks) != self.n_blocks():
            raise Reject("spec", f"{len(blocks)} blocks for a {self.g('grid')} grid")
        self.hull = self.world.local_hull(blocks[0])
        ext = self.hull.max(dim=0).values - self.hull.min(dim=0).values
        self.edge = float(ext.mean())
        # geometric centre of the block in its root frame (the asset's root origin need not be the cube centre)
        self.c_local = 0.5 * (self.hull.max(dim=0).values + self.hull.min(dim=0).values)
        self.template_facts["block_centre_offset_m"] = [round(float(v), 4) for v in self.c_local.tolist()]
        choice = self.choice.groups[self.role][0] if self.role in self.choice.groups else self.choice.models[self.role]
        faces_path = os.path.join(get_dataset_path(choice.dataset), "objects", choice.category, choice.model, "misc", "isg_block_faces.json")
        if os.path.exists(faces_path):
            with open(faces_path) as f:
                self.faces = dict(json.load(f)["faces"])
        support = self.obj("support")
        self.top = self.support_top(support)
        if int(self.top.free.sum()) < 50:
            raise Reject("support_small", f"{support.name}: too little free top")
        self.band_spec = TT.BandSpec(object=f"{choice.category}/{choice.model}", grasp_key=None, grasps=None, hull=self.hull, rest_quat=C.IDENTITY_QUAT.clone(),
                                     scale=list(choice.scale) if choice.scale else None, generic_pinch=True, height=self.edge)
        margins = {"kind": "block_design", "edge": self.edge, "block_gap": float(self.g("block_gap_m")), "safety": 0.02, "H_max": self.edge}
        rng = np.random.default_rng(self.ctx.base_seed)
        errors = []
        for edge, st in TT.edge_candidates(self, self.top, rng):
            try:
                bands = TT.ensure_bands(self, self.band_spec, top=self.top, edge=edge)
                # A1: the left contact band of the block's height class (row-left / column-top braces at the goal grid);
                # §4.4 margins over the union of both bands
                self.left_bands = TT.load_left_contact_band(self, self.edge, top=self.top, edge=edge, compute=TT.left_band_compute())
                init = TT.phase1b_init_trunk(self, st, self.edge, bands=bands, top=self.top, left_bands=self.left_bands)
                init = TT.phase1c_lateral(self, init, margins, rng, top=self.top, bands=bands)
                init.name = "init"
                cands = TT.phase2_manipulation_stance(self, init, 0.5 * self.edge, target_height=self.edge, focus_candidates=self._regions, in_frame_fn=self._region_in_frame,
                                                      bands=bands, top=self.top, erosion_extra=0.02 + self.edge, focus_z=self.top.z + 0.5 * self.edge)
                if not cands:
                    raise Reject("no_manipulation_stance", f"Phase 2 left no region survivor ({self._regions_off_left_band} regions off the left contact band)")
                self.edge_frame, self.init_stance, self.manip_candidates = edge, init, cands
                break
            except Reject as e:
                errors.append(f"edge {edge.name}: {e}")
                log.info("%s: %s", self.gp.alias, errors[-1][:300])
        if self.init_stance is None:
            raise Reject("no_stance", "; ".join(errors)[:1500])
        self.template_facts.update({"support": self.top.to_json(), "edge": self.edge_frame.to_json(), "init_stance": self.init_stance.to_meta(),
                                    "n_manipulation_candidates": len(self.manip_candidates), "edge_m": self.edge, "faces": self.faces,
                                    "band": getattr(self.bands, "path", None), "left_band": getattr(self.left_bands, "path", None) if self.left_bands is not None else None,
                                    "regions_off_left_band": self._regions_off_left_band})

    # -- Phase 2 hooks (§4.5 blocks) --------------------------------------------------------------------------------

    def _pitch(self) -> float:
        return self.edge + float(self.g("block_gap_m"))

    def _table_axes(self, stance: Stance) -> Tuple[th.Tensor, th.Tensor, float]:
        """``(right, toward, table_yaw)``: the robot's right / toward-the-robot directions snapped onto the table's own axes."""
        ax, ay = self.top.local_axes()

        def snap(v):
            return max([ax, -ax, ay, -ay], key=lambda c: float(th.dot(c, v)))

        right, toward = snap(-stance.tangent), snap(-stance.inward)
        return right, toward, math.atan2(float(right[1]), float(right[0]))

    def _regions(self, s, stance: Stance, cam, cells_world: th.Tensor) -> List[Tuple[th.Tensor, Any]]:
        """§4.5: regions of >= n_blocks + 2 band cells at spacing edge·√2 + 0.03 within ``REGION_RADIUS`` of a centre cell;
        the 2 x 2 goal grid (pitch = edge + 1 mm) at the region centre still on the top."""
        n_need = self.n_blocks() + int(self.g("region_extra_cells"))
        # §4.5 spacing edge·√2 + 0.03 (6.5 cm for the 2.5 cm cube) puts the neighbouring cubes under the open fingers of a
        # top-down pinch (pads ±5 cm from the centre + the finger): the spacing is at least half the opening + a finger
        # + half an edge + 5 mm (7.95 cm → adjacent 8 cm lattice cells) so that every block can be pinched with the
        # others still at their cells
        spacing = max(self.edge * math.sqrt(2) + float(self.g("region_spacing_extra_m")), 0.5 * C.GRIPPER_OPEN_M + FINGER_HALF_THICKNESS + 0.5 * self.edge + 0.005)
        right, toward, _ = self._table_axes(stance)
        pitch = self._pitch()
        out, seen = [], set()
        for c in cells_world:
            d = th.linalg.norm(cells_world - c, dim=1)
            near = cells_world[d <= REGION_RADIUS]
            chosen: List[th.Tensor] = []
            for p in near[th.argsort(th.linalg.norm(near - c, dim=1))]:
                if all(float(th.linalg.norm(p - q)) >= spacing for q in chosen):
                    chosen.append(p)
            if len(chosen) < n_need:
                continue
            region = th.stack(chosen)
            focus = region.mean(dim=0)
            key = (round(float(focus[0]), 2), round(float(focus[1]), 2))
            if key in seen:
                continue
            grid_ok = all(self.top.contains(focus + right * (cc * pitch) + toward * (rr * pitch), 0.5 * self.edge + 0.02) for rr in range(2) for cc in range(2))
            if not grid_ok:
                continue
            # A1 joint placement: the goal grid is anchored at one of the region's cells (the image-top-left block); its
            # row-left / column-top brace points must lie in a left contact band cell at this trunk — the region needs
            # at least one such cell (the right band supplies the pushes / picks)
            if self.left_bands is not None and not any(self._anchor_braces_in_left_band(c, stance, right, toward) for c in region):
                self._regions_off_left_band += 1
                continue
            seen.add(key)
            out.append((focus, {"cells": region}))
        return out

    def _anchor_braces_in_left_band(self, anchor_xy, stance: Stance, right: th.Tensor, toward: th.Tensor) -> bool:
        """Whether the two brace points of a goal grid anchored at ``anchor_xy`` (the closed left gripper's blade on the
        anchor's robot-left face and on its far face, half an edge + a finger outside the faces) lie within 1.25 lattice
        cells (10 cm) of a left contact band cell at the stance's trunk (the blade stands beside the cell, the bands overlap
        only at λ ≈ 0 on the pilot support)."""
        off = 0.5 * self.edge + FINGER_HALF_THICKNESS
        pts = [anchor_xy - right * off, anchor_xy - toward * off]
        return all(TT.in_band(self.left_bands, stance, p, tol=1.25)[0] for p in pts)

    def _region_in_frame(self, s, stance: Stance, cam, focus, info) -> bool:
        cells = info["cells"]
        pts = th.cat([TT.box_corners(c, 0.5 * self.edge, self.top.z, self.top.z + self.edge) for c in cells])
        return TT.in_frame(self, cam, pts, float(self.g("region_in_frame_margin")))

    # ------------------------------------------------------------------ faces / patterns

    def color_to_face(self) -> Dict[str, str]:
        return {c: f for f, c in self.faces.items()}

    def pattern(self, rng) -> Tuple[str, List[str]]:
        patterns = self.g("patterns") or {}
        if not patterns:
            raise Reject("spec", "generation.patterns is empty")
        if self.variant and self.variant in patterns:
            return self.variant, list(patterns[self.variant])
        pid = list(patterns)[int(rng.integers(len(patterns)))]
        return pid, list(patterns[pid])

    # ------------------------------------------------------------------ instance

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        init = self.init_stance
        errors = []
        for k in rng.permutation(len(self.manip_candidates))[:6]:
            cand = self.manip_candidates[int(k)]
            manip = cand.with_trunk(cand.translate, cand.tilt, "manipulation")
            manip.focus, manip.arms = cand.focus, ["right", "left"]
            manip.extras["focus_info"] = cand.extras.get("focus_info")
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
        names = self.names()
        n = len(names)
        n_rows, n_cols = (int(v) for v in (self.g("grid") or [2, 2]))
        pid, pattern = self.pattern(rng)
        c2f = self.color_to_face()
        for c in pattern:
            if c not in c2f:
                raise Reject("spec", f"pattern colour {c!r} not on the block ({sorted(c2f)})")
        d.layout_template_id = pid
        self.record_transition(init, manip)
        cam_init = self.camera_pose_at(init)
        right, toward, table_yaw = self._table_axes(manip)
        pitch = self._pitch()
        region = manip.extras["focus_info"]["cells"]
        gap = float(self.g("scatter_min_gap_m"))
        # --- cells: one block per region cell, >= 3 cm gaps between footprints ------------------------------------
        z_c = self.top.z + 0.5 * self.edge

        def img_score(xy) -> float:
            uv = self.world.camera_model.project(cam_init, th.tensor([[float(xy[0]), float(xy[1]), z_c]]))[0][0]
            return float(uv[0] + uv[1])

        cells = [region[int(i)] for i in rng.permutation(len(region))]
        chosen: List[th.Tensor] = []
        if self.left_bands is not None:
            # A1: the goal grid is anchored at the image-top-left block; its row-left / column-top brace points must lie
            # in a left contact band cell at this trunk — take such a cell as the anchor (the smallest image score among
            # them) and keep the other blocks below / right of it in the image
            feas = [xy for xy in cells if self._anchor_braces_in_left_band(xy, manip, right, toward)]
            if not feas:
                raise Reject("layout", f"no region cell whose row-left / column-top brace points lie in the left contact band at trunk "
                                       f"({manip.translate}, {manip.tilt}) ({len(cells)} cells)")
            anchor_xy = min(feas, key=img_score)
            chosen.append(anchor_xy)
            s_anchor = img_score(anchor_xy)
            cells = [xy for xy in cells if xy is not anchor_xy and img_score(xy) > s_anchor]
        for xy in cells:
            if all(float(th.linalg.norm(xy - c)) >= self.edge * math.sqrt(2) + gap for c in chosen):
                chosen.append(xy)
            if len(chosen) == n:
                break
        if len(chosen) < n:
            raise Reject("layout", f"only {len(chosen)} of {n} region cells keep >= {gap} m gaps{' below / right of the anchor' if self.left_bands is not None else ''}")
        # --- anchor (top-left in the init image) and slot assignment -----------------------------------------------
        score = [img_score(c) for c in chosen]
        order_img = sorted(range(n), key=lambda i: score[i])  # image order (top-left first)
        anchor_i = order_img[0]
        slot_of = {bi: si for si, bi in enumerate(order_img)}  # block index -> slot index (anchor = slot 0)
        goal_face = {bi: c2f[pattern[si]] for bi, si in slot_of.items()}
        # --- up faces (distinct, never the opposite of the goal face) and yaws (visible-face rule) ------------------
        faces = list(self.faces)
        up_of: Dict[int, str] = {}
        for bi in order_img:
            options = [f for f in faces if f != TT.opposite_face(goal_face[bi]) and f not in up_of.values()]
            if not options:
                raise Reject("layout", "no distinct up face left that is not the opposite of the goal face")
            # prefer showing the goal face on top when it is still free (visible-face rule, first clause)
            up_of[bi] = goal_face[bi] if goal_face[bi] in options and rng.random() < 0.5 else options[int(rng.integers(len(options)))]
        poses: Dict[int, TT.Pose] = {}
        coarse_yaw: Dict[int, float] = {}
        for bi in range(n):
            q_up = _rotation_with_up(up_of[bi])
            if up_of[bi] == goal_face[bi]:
                yaw_c = table_yaw + float(rng.integers(4)) * math.pi / 2  # the goal face is up: any table-aligned coarse yaw
            else:
                yaw_c = TT.facing_coarse_yaw(q_up, goal_face[bi], -manip.inward)  # visible-face rule: goal face toward the robot
            coarse_yaw[bi] = yaw_c
            jitter = float(rng.uniform(-math.radians(float(self.g("yaw_align_max_deg")) * 0.5), math.radians(float(self.g("yaw_align_max_deg")) * 0.5)))
            q = C.qmul(C.quat_from_yaw(yaw_c + jitter), q_up)
            poses[bi] = self._root_pose(chosen[bi], q, 0.002)
        # footprints on the top, gaps, in frame (5 %)
        fps = {bi: self.world.footprint(self.hull, poses[bi]) for bi in range(n)}
        for bi in range(n):
            if not self.top.polygon_supported(fps[bi], margin=TT.SUPPORT_MARGIN):
                raise Reject("layout", f"block {bi} footprint off the top")
        for a in range(n):
            for b in range(a + 1, n):
                if TT._polygons_overlap(fps[a], fps[b], gap):
                    ext = [[round(v, 3) for v in (fps[i].max(dim=0).values - fps[i].min(dim=0).values).tolist()] for i in (a, b)]
                    raise Reject("layout", f"block footprints {a} / {b} closer than the gap {gap} (clearance {TT.polygon_gap(fps[a], fps[b]):.3f} m, centres "
                                           f"{float(th.linalg.norm(chosen[a] - chosen[b])):.3f} m apart, footprint extents {ext}, centres {[[round(float(v), 3) for v in fps[i].mean(dim=0).tolist()] for i in (a, b)]}, "
                                           f"cells {[[round(float(v), 3) for v in chosen[i].tolist()] for i in (a, b)]})")
        missing = TT.all_in_frame(self, cam_init, [(names[bi], self.hull, poses[bi]) for bi in range(n)], TT.IN_FRAME_MARGIN)
        if missing:
            raise Reject("not_in_frame", f"{missing} not fully in the init image")
        objects = {names[bi]: poses[bi] for bi in range(n)}
        # goal face visibility: >= 50 % of a 3 x 3 grid of face points visible from the init camera (blocks placed in the sim)
        self._teleport(objects)
        vis = self._goal_face_visibility(cam_init, poses, goal_face)
        bad = {bi: v for bi, v in vis.items() if v < float(self.g("goal_face_visible_fraction"))}
        if bad:
            raise Reject("layout", f"goal faces below {self.g('goal_face_visible_fraction')} visible: {bad}")
        # --- goal layout at the anchor ------------------------------------------------------------------------------
        anchor = self._centre(poses[anchor_i])[0]
        plans: Dict[int, Dict[str, Any]] = {}
        slots: Dict[str, Dict[str, Any]] = {}
        goal_pose: Dict[int, TT.Pose] = {}
        for bi in range(n):
            si = slot_of[bi]
            r, c = divmod(si, n_cols)
            plan = TT.cube_flip_plan(poses[bi][1], goal_face[bi], table_yaw, float(self.g("yaw_align_max_deg")))
            if plan is None:
                raise Reject("layout", f"block {bi}: no smallest-rotation plan to face {goal_face[bi]}")
            plans[bi] = plan
            q_goal = plan["goal_quat"]
            xy = anchor[:2] + right * (c * pitch) + toward * (r * pitch)
            goal_pose[bi] = self._root_pose(xy, q_goal, 0.001)
            slots[f"slot_{si}"] = {"position": [float(v) for v in goal_pose[bi][0]], "orientation": [float(v) for v in q_goal]}
        # §4.6 step 5: whole-body checks at init and manipulation with the blocks placed
        TT.whole_body_check_at(self, [init, manip], objects, require=True, reason="stance_collision")
        # --- cube grasps, flips and brace points per block in placement order ---------------------------------------
        order = sorted(range(n), key=lambda bi: slot_of[bi])  # anchor first, then the slots in order
        records = []
        for step, bi in enumerate(order):
            rec = self._block_program(rng, manip, names, poses, goal_pose, plans, order, step, right, toward, table_yaw)
            records.append(rec)
        # --- draft ---------------------------------------------------------------------------------------------------
        for bi in range(n):
            d.object_poses[names[bi]] = poses[bi]
            d.goal_poses[names[bi]] = goal_pose[bi]
        d.robot_q = self.stance_q(init)
        d.base_pose2d = tuple(init.base_pose2d)
        gname = self._group_name(names)
        sids = [f"slot_{slot_of[bi]}" for bi in range(n)]
        d.groups[gname] = {
            "members": names,
            "assignment": {"mode": "exchangeable", "cost": "layout"},
            "orientation_equivalences": [],
            "target_slots": {sids[bi]: slots[sids[bi]] for bi in range(n)},
            "pose_tolerances": {"position": self.tol("position", 0.01), "orientation": self.tol("orientation", 0.15)},
            "layout_relations": _relations(slots),
            "layout_tolerances": {"position": self.tol("position", 0.01), "orientation": self.tol("orientation", 0.15)},
        }
        self.sidecar.add_stance(manip)
        self.sidecar.blocks = records
        for bi in range(n):
            self.sidecar.describe(names[bi], "block", f"the block with the {self.faces.get(up_of[bi], up_of[bi])} face up" + (" (top-left)" if bi == anchor_i else ""))
        d.summary.update({"pattern": pid, "pattern_faces": pattern, "up_faces": {names[bi]: up_of[bi] for bi in range(n)}, "goal_faces": {names[bi]: goal_face[bi] for bi in range(n)},
                          "anchor_block": names[anchor_i], "table_yaw": table_yaw, "manipulation_candidate": k, "goal_face_visibility": {names[bi]: v for bi, v in vis.items()},
                          "placement_order": [names[bi] for bi in order]})
        return d

    # -- helpers ------------------------------------------------------------------------------------------------------

    def _group_name(self, names: List[str]) -> str:
        if self.problem is not None:
            for g, members in self.problem.groups.items():
                if set(members) >= set(names):
                    return g
        from ispatialgym.sampling.bddl_writer import GROUP_NAMES

        return GROUP_NAMES["blocks"]

    def _teleport(self, objects: Dict[str, TT.Pose]):
        for name, pose in objects.items():
            o = self.objects[name]
            o.set_position_orientation(TT._t(pose[0]), TT._t(pose[1]))
            o.keep_still()
        self.world.step_physics(1, still=[self.robot] + [self.objects[n] for n in objects])
        self.kit.sync_world()

    def _goal_face_visibility(self, cam, poses: Dict[int, TT.Pose], goal_face: Dict[int, str]) -> Dict[int, float]:
        """Fraction of a 3 x 3 grid of points on each block's goal face whose ray from the camera reaches the point (first
        hit = the block itself within 5 mm of the point) — blocks and the scene at their init poses."""
        out = {}
        cam_p = TT._t(cam[0])
        for bi, pose in poses.items():
            pts = TT.face_points(self._centre(pose), goal_face[bi], self.edge)
            normal = TT.face_normal_world(goal_face[bi], pose[1])
            to_cam = cam_p.unsqueeze(0) - pts
            dist = th.linalg.norm(to_cam, dim=1, keepdim=True)
            dirs = to_cam / dist
            facing = (dirs @ normal) > 0.05  # the face must look at the camera
            # reverse rays: from just outside the face toward the camera, stopped 0.25 m short of it (the camera sits inside
            # the robot's head); any hit = another block or the scene occludes the point
            starts = pts + normal * 0.002
            ends = cam_p.unsqueeze(0) - dirs * 0.25
            hits = self.world.raycast(starts, ends)
            n_vis = sum(1 for h, f in zip(hits, facing.tolist()) if f and not h["hit"])
            out[bi] = n_vis / len(pts)
        return out

    def _block_program(self, rng, manip: Stance, names: List[str], poses: Dict[int, TT.Pose], goal_pose: Dict[int, TT.Pose], plans: Dict[int, Dict[str, Any]],
                       order: List[int], step: int, right: th.Tensor, toward: th.Tensor, table_yaw: float) -> Dict[str, Any]:
        """§6.5 for one block: the cube grasp (pick + place IK in placement order with the already-placed blocks at their
        slots), the flip waypoints, the row / column brace + push contacts."""
        bi = order[step]
        block = self.objects[names[bi]]
        support = self.obj("support")
        plan = plans[bi]
        theta = float(self.g("flip_tilt_deg"))
        # world for the pick: blocks placed so far at their slots, the others (incl. this one) at init
        world_poses = {names[o]: (goal_pose[o] if j < step else poses[o]) for j, o in enumerate(order)}
        self._teleport(world_poses)
        pick_pose, place_pose = poses[bi], goal_pose[bi]
        # candidate grasps: face pair (∥ flip axis for a flipping block; either horizontal pair otherwise) x wrist yaw x tilt pair
        cands: List[Tuple[str, float, float, float, int]] = []  # (face_pair, tilt_pick, tilt_place, yaw, flip_sign)
        if plan["flips"]:
            pair = plan["face_pair"]
            for sign in (1, -1):
                tilt_pick, tilt_place = TT.cube_tilt_pair(theta, sign)
                for yaw in (table_yaw, table_yaw + math.pi / 2, table_yaw + math.pi, table_yaw - math.pi / 2):
                    cands.append((pair, tilt_pick, tilt_place, yaw, sign))
        else:
            up = TT.up_face(pick_pose[1])
            pairs = [ax for ax in "xyz" if ax != up[1]]
            # straight down first, then the pitched approaches of the band's generic classes (the same tilt at the pick
            # and the place: no reorientation in hand) — the far cells of the band pass only with a pitched approach
            for tilt in (0.0, 30.0, -30.0, 45.0, -45.0):
                for pair in pairs:
                    for yaw in (table_yaw, table_yaw + math.pi / 2, table_yaw + math.pi, table_yaw - math.pi / 2):
                        cands.append((pair, tilt, tilt, yaw, 0))
        # keep the downward pick approaches only (exactly one tilt sign for a flipping block), then screen pick + place pairs
        screened = []
        for pair, tilt_pick, tilt_place, yaw, sign in cands:
            try:
                g_pick = C.cube_grasp(self._centre(pick_pose), pair, tilt_pick, yaw, edge=self.edge)
            except ValueError:
                continue
            if not g_pick.downward:
                continue
            # place eef: the rigid in-hand transform carried to the goal block pose
            place_eef = C.compose(self._centre(place_pose), g_pick.eef_in_block)
            if float(C.approach_axis(place_eef[1])[2]) > -0.1:
                continue  # the place approach must point down as well
            screened.append((pair, tilt_pick, tilt_place, yaw, sign, g_pick, place_eef))
        if not screened:
            raise Reject("grasp_unreachable", f"block {names[bi]}: no cube grasp with downward pick and place approaches")
        pre_picks = [[IK.pre_pose(c[5].eef), c[5].eef] for c in screened]
        ok_pick = self.ikc.screen_pairs(pre_picks, manip, "right", ignore_for_last=[support, block])
        idx = [i for i, s in enumerate(ok_pick.tolist()) if s]
        if not idx:
            raise Reject("grasp_unreachable", f"block {names[bi]}: no pick passes the screening IK")
        chosen = None
        for i in [idx[j] for j in rng.permutation(len(idx))]:
            pair, tilt_pick, tilt_place, yaw, sign, g_pick, place_eef = screened[i]
            pre_pick = IK.pre_pose(g_pick.eef)
            res = self.ikc.exact([pre_pick, g_pick.eef], manip, "right", ignore_for_last=[support, block])
            if not res.ok:
                continue
            # place (and pre-place) with the block in hand; the world: placed blocks at their slots, this block removed (attached)
            place_margin_pose = (place_eef[0] + _xy3(right) * PLACE_MARGIN + _xy3(toward) * PLACE_MARGIN, place_eef[1])
            pre_place = IK.pre_pose(place_margin_pose)
            with TT.posed_in_hand(self, "right", block, C.relative(g_pick.eef, pick_pose)):
                with attached(self.kit, "right", block):
                    res_p = self.ikc.exact([pre_place, place_margin_pose], manip, "right", ignore_for_last=[support])
                    wps_ok = True
                    wps: List[TT.Pose] = []
                    if res_p.ok and (plan["flips"] or abs(plan["yaw_deg"]) > 1e-6):
                        # §3.9 in-hand reorientation waypoints (pick tilt -> place tilt) with the block attached, from the lifted grasp
                        lift = (g_pick.eef[0] + th.tensor([0.0, 0.0, self._carry_height()]), g_pick.eef[1])
                        pivot = self._centre(pick_pose)[0] + th.tensor([0.0, 0.0, self._carry_height()])
                        if plan["flips"]:
                            # F5 order: (eef_pose, pivot, yaw [rad], flip_axis, flip_sign)
                            wps = list(C.flip_waypoints(lift, pivot, math.radians(plan["yaw_deg"]), plan["flip_axis"], plan["flip_sign"], n_yaw=2, n_flip=3))
                        else:
                            wps = list(C.orbit_waypoints(lift, pivot, C.quat_from_yaw(math.radians(plan["yaw_deg"])), n=2))
                        r_w = self.ikc.waypoints_ok([lift] + wps, manip, "right", ignore_all=[support])
                        wps_ok = r_w.ok
            if not res_p.ok or not wps_ok:
                continue
            chosen = (pair, tilt_pick, tilt_place, yaw, sign, g_pick, place_margin_pose, pre_pick, pre_place, wps)
            break
        if chosen is None:
            raise Reject("place_unreachable", f"block {names[bi]}: no cube grasp passes pick + place (+ flip waypoints) exact IK")
        pair, tilt_pick, tilt_place, yaw, sign, g_pick, place_eef, pre_pick, pre_place, wps = chosen
        # tilts about the *snapped* closing axis of the pick grasp: pick from ``cube_grasp``, place from the carried eef
        q0 = C.qmul(g_pick.eef[1], C.quat_from_axis_angle([0.0, 1.0, 0.0], -math.radians(g_pick.tilt_deg)))
        tilt_place_c = _tilt_about_y(q0, place_eef[1])
        rec: Dict[str, Any] = {"instance": names[bi], "order": step, "face_pair": pair, "tilt_pick_deg": float(g_pick.tilt_deg), "tilt_place_deg": float(tilt_place_c),
                               "yaw_deg": float(math.degrees(g_pick.yaw)), "yaw_align_deg": float(plan["yaw_deg"]),
                               "flip_axis": None if plan["flip_axis"] is None else [float(v) for v in plan["flip_axis"]],
                               "flip_sign": int(plan["flip_sign"]), "pick": pose_to_sidecar(g_pick.eef), "pre_pick": pose_to_sidecar(pre_pick),
                               "place": pose_to_sidecar(place_eef), "pre_place": pose_to_sidecar(pre_place), "carry_height": float(self._carry_height()),
                               "goal_face": TT.up_face(goal_pose[bi][1]), "slot": step,
                               # the cube's geometric centre in the block's root frame (m): the ``isgblk`` root origin is
                               # off-centre; ``cube_grasp`` / the flip pivot take the centre pose = root pose ∘ (centre_offset, I)
                               "centre_offset": [float(v) for v in self.c_local.tolist()]}
        if wps:
            rec["waypoints"] = [pose_to_sidecar(w) for w in wps]
        # brace points (§6.5): row-left / column-top blocks braced by the closed left gripper while the right hand pushes
        n_cols = int((self.g("grid") or [2, 2])[1])
        r, c = divmod(step, n_cols)
        world_after = dict(world_poses)
        world_after[names[bi]] = (goal_pose[bi][0] + _xy3(right) * PLACE_MARGIN + _xy3(toward) * PLACE_MARGIN, goal_pose[bi][1])
        self._teleport(world_after)
        if c > 0:  # push left against the row, brace the row-left block's left face
            left_bi = order[step - 1]
            self._brace_push(rec, "row", names[bi], names[left_bi], world_after, direction=-right, brace_dir=right, manip=manip)
        if r > 0:  # push up (away from the robot) against the column, brace the column-top block's far face
            top_bi = order[step - n_cols]
            self._brace_push(rec, "col", names[bi], names[top_bi], world_after, direction=toward * -1.0, brace_dir=toward, manip=manip)
        return rec

    def _carry_height(self) -> float:
        return 2.0 * self.edge + 0.03

    def _root_pose(self, centre_xy, q: th.Tensor, drop: float) -> TT.Pose:
        """Root pose of a block whose geometric centre stands at ``centre_xy`` on the top with orientation ``q``."""
        off = C.quat_rotate(q, self.c_local)
        z = self.top.z + self.world.bottom_offset(self.hull, q) + drop
        return th.tensor([float(centre_xy[0]) - float(off[0]), float(centre_xy[1]) - float(off[1]), z]), q

    def _centre(self, root_pose: TT.Pose) -> TT.Pose:
        """Centre pose (the cube centre, same orientation) of a block root pose — what ``cube_grasp`` / ``face_points`` take."""
        p, q = TT._t(root_pose[0]), TT._t(root_pose[1])
        return p + C.quat_rotate(q, self.c_local), q

    def _face_contact(self, pose: TT.Pose, direction_xy: th.Tensor, into: bool) -> Tuple[th.Tensor, th.Tensor]:
        """Contact point on the block face whose outward normal is ``-direction_xy`` (push: the pads press the face the
        block moves away from) or ``+direction_xy`` (brace: into the face), at 3/4 of the block height, approach tilted
        ``CONTACT_TILT_DEG`` below the horizontal along the push / brace direction."""
        dxy = th.tensor([float(direction_xy[0]), float(direction_xy[1]), 0.0])
        dxy = dxy / th.linalg.norm(dxy)
        centre = self._centre(pose)[0]
        face_normal = -dxy if into else dxy  # outward normal of the face touched
        point = centre + face_normal * (0.5 * self.edge) + th.tensor([0.0, 0.0, 0.25 * self.edge])
        a = -face_normal * math.cos(math.radians(CONTACT_TILT_DEG)) + th.tensor([0.0, 0.0, -math.sin(math.radians(CONTACT_TILT_DEG))])
        return point, a / th.linalg.norm(a)

    def _blade_height(self) -> float:
        """eef origin height of a top-down blade beside a block: the closed fingertips 3 mm above the top (a 2.5 cm cube
        sits entirely below the wrist of a horizontal contact)."""
        return float(self.top.z) + max(0.6 * self.edge, C.FINGERTIP_LENGTH + 0.003)

    def _blade_poses(self, pose: TT.Pose, direction_xy: th.Tensor, into: bool, manip: Stance, prefix: str):
        """Top-down blade contacts of the closed gripper beside a block face (outward normal ``-direction_xy`` for a
        push, ``+direction_xy`` for a brace): the eef origin half a finger outside the face at the blade height,
        approach straight down / pitched 30° / 45° / 60° toward the robot, closing axis (the blade) along the face
        normal with either wrist turn. Yields ``(pose, approach, name)``."""
        dxy = th.tensor([float(direction_xy[0]), float(direction_xy[1]), 0.0])
        dxy = dxy / th.linalg.norm(dxy)
        face_normal = -dxy if into else dxy  # outward normal of the face touched
        centre = self._centre(pose)[0]
        origin = th.tensor([float(centre[0]), float(centre[1]), self._blade_height()]) + face_normal * (0.5 * self.edge + FINGER_HALF_THICKNESS)
        side = th.cross(th.tensor([0.0, 0.0, 1.0]), face_normal, dim=0)
        # closing axis along the face normal (the pad face pushes) or across it (the finger's side pushes): a top-down
        # pose with the closing axis lateral to the robot is infeasible at many cells (wrist limits), so both are tried
        for pitch in (0.0, 30.0):  # the cubes sit 0.5–0.8 m ahead: the 45° / 60° classes are for the far cells
            a_vec = TT.pitched_approach(pitch, manip.forward)
            for c_axis, cname in ((face_normal, "pad"), (side, "side")):
                for c_sign in (1.0, -1.0):
                    yield (origin, C.eef_quat(a_vec, c_sign * c_axis)), a_vec, f"{prefix}_blade_{cname}_{pitch:.0f}{'' if c_sign > 0 else '-'}"

    def _brace_push(self, rec: Dict[str, Any], kind: str, pushed: str, braced: str, world_poses: Dict[str, TT.Pose], direction: th.Tensor, brace_dir: th.Tensor, manip: Stance):
        """Right-hand push of ``pushed`` along ``direction`` by the place margin while the closed left gripper braces
        ``braced`` on its face opposite to the push; pre-contact + contact IK for both arms (§4.6 step 6). Contacts:
        the top-down blade beside the face (the blocks are 2.5 cm cubes) with the pitched approach classes first, the
        tilted / horizontal side contacts (pads on the face at 3/4 height) as the fallback; the first push pose whose
        pre + init pass the exact IK is paired with the first brace pose that passes the two-arm check."""
        support = self.obj("support")
        dist = PLACE_MARGIN
        d3 = th.tensor([float(direction[0]), float(direction[1]), 0.0])
        d3 = d3 / th.linalg.norm(d3)
        z_axis = th.tensor([0.0, 0.0, 1.0])
        # push candidates: blades, then the tilted side contact
        push_cands = list(self._blade_poses(world_poses[pushed], direction, False, manip, "push"))
        p_pt, p_a = self._face_contact(world_poses[pushed], direction, into=False)
        c_axis = th.tensor([float(-direction[1]), float(direction[0]), 0.0])  # pads close across the push direction (horizontal)
        push_cands.append((C.contact_eef(p_pt, p_a, c_axis), p_a, "push_tilted"))
        # brace candidates: blades, then the tilted side contact (either wrist turn) and B3's horizontal classes
        brace_cands = list(self._blade_poses(world_poses[braced], brace_dir, True, manip, "brace"))
        b_pt, b_a = self._face_contact(world_poses[braced], brace_dir, into=True)
        c_axis_b = th.tensor([float(-brace_dir[1]), float(brace_dir[0]), 0.0])
        a_h = th.tensor([float(brace_dir[0]), float(brace_dir[1]), 0.0])
        a_h = a_h / th.linalg.norm(a_h)
        for a_vec, c_vec, name in ((b_a, c_axis_b, "tilted_h"), (b_a, -c_axis_b, "tilted_h-"), (a_h, c_axis_b, "side_h"), (a_h, z_axis, "side_v")):
            brace_cands.append((C.contact_eef(b_pt, a_vec, c_vec), a_vec, name))
        push_fails, brace_fails = [], []
        chosen = None
        for push_init, push_a, p_name in push_cands:
            push_goal = (push_init[0] + d3 * dist, push_init[1])
            push_pre = C.pre_pose(push_init)
            # the push init / goal touch the block: init is exact-IK-checked with the block ignored, then pre + goal with the brace arm
            r_init = self.ikc.exact([push_pre, push_init], manip, "right", ignore_for_last=[support, self.objects[pushed]], cache=False)
            if not r_init.ok:
                push_fails.append(f"{p_name}: {r_init.reason}")
                continue
            for b, b_a_vec, b_name in brace_cands:
                b_pre = C.pre_pose(b)
                res = TT.brace_second_arm(self, manip, "right", [push_pre, push_goal], "left", [b_pre, b], ignore_work_last=[support, self.objects[pushed]],
                                          ignore_brace_last=[support, self.objects[braced]])
                if res.ok:
                    chosen = (push_pre, push_init, push_goal, b_pre, b, b_a_vec, f"{p_name}+{b_name}")
                    break
                brace_fails.append(f"{p_name}+{b_name}: {res.reason}")
            if chosen is not None:
                break
        if chosen is None:
            if brace_fails:
                raise Reject("contact_unreachable", f"{kind} brace / push of {pushed}: {sorted(set(brace_fails))[:3]}")
            raise Reject("contact_unreachable", f"{kind} push init of {pushed}: {sorted(set(push_fails))[:3]}")
        push_pre, push_init, push_goal, brace_pre, brace, brace_a, used = chosen
        rec[f"push_{kind}"] = ContactRecord("push", "right", push_pre, push_init, None, push_goal, d3, dist, object=pushed).to_sidecar()
        rec[f"brace_{kind}"] = ContactRecord("brace", "left", brace_pre, brace, direction=brace_a, brace_kind="row_left" if kind == "row" else "column_top",
                                              object=braced).to_sidecar()
        rec[f"brace_{kind}"]["pose_class"] = used

    # ------------------------------------------------------------------ goal / hooks

    def goal(self, draft: Draft):
        manip = self.sidecar.stance("manipulation")
        self.sidecar.set_goal(manip.base_pose2d, manip.translate, manip.tilt, arms="same")

    def refine_targets_from_goal(self, draft: Draft):
        """Snap the slots to the settled goal poses (blocks may sink a fraction of a millimetre) and rebuild the relations."""
        names = self.names()
        gname = self._group_name(names)
        group = draft.groups[gname]
        sids = list(group["target_slots"])
        by_name = {n: sid for n, sid in zip(names, sids)}
        for name in names:
            p, q = self.objects[name].get_position_orientation()
            sid = by_name[name]
            ref_q = th.tensor(group["target_slots"][sid]["orientation"])
            if C.quat_angle(q, ref_q) > 0.05 or float(th.linalg.norm(p - th.tensor(group["target_slots"][sid]["position"]))) > 0.005:
                raise Reject("goal_unstable", f"block {name} moved at the goal")
            group["target_slots"][sid] = {"position": p.tolist(), "orientation": q.tolist()}
        group["layout_relations"] = _relations(group["target_slots"])


def _rotation_with_up(face: str) -> th.Tensor:
    """A cube rotation (xyzw) with ``face`` along world +z (the smallest-angle choice about the vertical)."""
    import itertools

    best = None
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = th.zeros(3, 3)
            for i, (p, sg) in enumerate(zip(perm, signs)):
                m[i, p] = float(sg)
            if abs(float(th.det(m)) - 1.0) < 1e-6:
                q = C.mat_to_quat(m)
                if float(TT.face_normal_world(face, q)[2]) > 0.99:
                    ang = C.quat_angle(q, C.IDENTITY_QUAT)
                    if best is None or ang < best[0]:
                        best = (ang, q)
    return best[1]


def _xy3(v) -> th.Tensor:
    """(x, y) -> (x, y, 0) world vector."""
    return th.tensor([float(v[0]), float(v[1]), 0.0])


def _tilt_about_y(q0: th.Tensor, q: th.Tensor) -> float:
    """Signed rotation (deg) of ``q`` relative to ``q0`` about the eef y axis (the closing axis): the tilt of a cube grasp."""
    rel = C.qmul(C.qinv(q0), q)
    ang = 2.0 * math.atan2(float(rel[1]), float(rel[3]))
    return math.degrees(C.wrap_angle(ang))


def _relations(slots: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    sids = list(slots)
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            a, b = slots[sids[i]], slots[sids[j]]
            p_rel, q_rel = C.relative((th.tensor(b["position"]), th.tensor(b["orientation"])), (th.tensor(a["position"]), th.tensor(a["orientation"])))
            out.append({"from": sids[i], "to": sids[j], "position": [float(v) for v in p_rel], "orientation": [float(v) for v in q_rel]})
    return out
