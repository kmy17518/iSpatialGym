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
#: closed fingers beside a cube face: the measured profile's fingertip section (``tabletop.FINGER_PROFILE``; the region
#: spacing and the brace-point band test use it, the blade poses take the profile over the face height)
FINGER_HALF_THICKNESS = TT.FINGER_PROFILE[0][1]
#: the place puts the block this far right and toward the robot of its slot; the pushes close the margin (§6.2 "small safety margin")
PLACE_MARGIN = 0.005
REGION_RADIUS = 0.20
#: owner 2026-09-16 10:47 ladder: block regions 0.5–0.6 m ahead of the base first, the pitched 30° / 45° picks as the primary
#: class for every block (straight down last), the pinch 1 mm above the table so the pads get more than ~1 cm of face
REGION_DEPTH_PREFERRED = 0.55
MAX_REGION_DEPTH = 0.80  # farthest init cell (m, base-forward): the probes pass picks to 0.80 m at (0.5, -0.45)
MAX_GRID_DEPTH = 0.70  # farthest goal slot (m, base-forward): the pre-places fail beyond (blk_diag 19:45)
GRID_LATERAL_PREFERRED = -0.20  # base-lateral of the goal grid centre: in front of the right shoulder (probe_place_depth 17:59)
GOOD_TRUNKS = ((0.5, -0.45), (0.5, 0.0), (0.25, -0.3))  # (translate, tilt) the place-depth probe verified for pick + pre-place
GRID_LATERAL_WINDOW = (-0.40, -0.10)  # base-lateral window of the init cells / slots the right arm's pitched classes reach
RELEASE_ABOVE_SURFACE = 0.01  # owner item 3: release 1 cm above the actual surface under the slot (surface_z_under)
MAX_MANIP_TRANSLATE = 0.5  # manipulation trunks deeper than this put the tilted picks' pre-grasps out of reach (PICK_FAILURE 13:20)
PICK_TILTS_DEG = (30.0, -30.0, 45.0, -45.0, 0.0)
PICK_TIP_CLEARANCE = 0.001
PICK_TIP_CLEARANCES = (0.001, 0.003, 0.006)  # the ladder: deepest pinch first, shallower when the attached cube self-collides


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
        self.braces_missing = False  # the last instance has a block without a brace / push tail (owner 2026-09-16 11:08)
        self.slot_surface_z: Dict[int, float] = {}  # raycast surface under each goal slot (release height, owner item 3)

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
                # D-009 shape for blocks (prepared, default = today's 8 / 20 deg): with the region depth caps (init cells <= 0.80 m,
                # slots <= 0.70 m) the 20 deg cone at the arm-friendly trunks (0.5, -0.3..-0.45) meets the top beyond the caps and
                # only the translate-0.95 trunks survive (blk_diag2 22:28: 2 survivors, both (0.95, 0 / -0.16), every pick
                # pre-grasp out of reach from that height); a wider loose class (generation.phase2_loose_class_deg ~ 30) lets
                # those trunks see regions at 0.55-0.70 m
                cands = TT.phase2_manipulation_stance(self, init, 0.5 * self.edge, target_height=self.edge, focus_candidates=self._regions, in_frame_fn=self._region_in_frame,
                                                      bands=bands, top=self.top, erosion_extra=0.02 + self.edge, focus_z=self.top.z + 0.5 * self.edge,
                                                      angle_classes_deg=(float(self.gen("phase2_tight_class_deg", 8.0)), float(self.gen("phase2_loose_class_deg", 20.0))))
                if not cands:
                    raise Reject("no_manipulation_stance", f"Phase 2 left no region survivor ({getattr(self, '_regions_too_deep', 0)} regions dropped: init cells beyond "
                                                            f"{MAX_REGION_DEPTH:.2f} m or goal slots beyond {MAX_GRID_DEPTH:.2f} m base-forward)")
                # PICK_FAILURE diagnostics (2026-09-16 13:20): every failing pick sat at translate 0.75–0.95 with the pre-grasp of a
                # tilted pick out of reach (the grasp itself feasible), while the probe passes at (0.5, 0) / (0.75, −0.3):
                # keep the shallow trunks when Phase 2 offers them
                shallow = [c for c in cands if float(c.translate) <= MAX_MANIP_TRANSLATE]
                if shallow:
                    cands = shallow
                self.edge_frame, self.init_stance, self.manip_candidates = edge, init, cands
                break
            except Reject as e:
                errors.append(f"edge {edge.name}: {e}")
                log.info("%s: %s", self.gp.alias, errors[-1][:300])
        if self.init_stance is None:
            raise Reject("no_stance", "; ".join(errors)[:1500])
        self.template_facts.update({"support": self.top.to_json(), "edge": self.edge_frame.to_json(), "init_stance": self.init_stance.to_meta(),
                                    "n_manipulation_candidates": len(self.manip_candidates), "edge_m": self.edge, "faces": self.faces,
                                    "band": getattr(self.bands, "path", None), "left_band": getattr(self.left_bands, "path", None) if self.left_bands is not None else None})

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
        # + half an edge + 3 cm (cuRobo's collision activation distance: 9 mm of clearance at adjacent 8 cm cells still
        # failed every pick screening) → 10.1 cm: diagonal lattice neighbours (11.3 cm)
        spacing = max(self.edge * math.sqrt(2) + float(self.g("region_spacing_extra_m")), 0.5 * C.GRIPPER_OPEN_M + FINGER_HALF_THICKNESS + 0.5 * self.edge + 0.03)
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
            # owner decision 2026-09-16 11:08: the region is chosen for the right arm's picks / places (the braces are a
            # bonus where the bands overlap); the left band no longer forces the anchor toward λ ≈ 0
            n_left = sum(1 for c in region if self._anchor_braces_in_left_band(c, stance, right, toward)) if self.left_bands is not None else 0
            seen.add(key)
            # depth caps (blk_diag 19:45-20:00: the far slots' pre-places failed the IK under every condition at 0.74-0.88 m
            # base-forward, the pick pre-grasps failed the screening at 0.72-0.88 m; the probes pass picks + pre-poses to
            # 0.65-0.70 m at (0.5, 0) and to 0.80 m at (0.5, -0.45)): the goal grid (pre-places, the tighter reach) within
            # MAX_GRID_DEPTH of the base, the init cells within MAX_REGION_DEPTH. A 0.70 m cap on every cell dropped all
            # 867 / 1762 regions of the 20:51 first-demo run (a region spans >= 0.2 m from cells starting at ~0.55 m)
            depth_max = max(float(stance.world_to_base(c)[0]) for c in region)
            grid_depth = max(float(stance.world_to_base(focus + right * (cc * pitch) + toward * (rr * pitch))[0]) for rr in range(2) for cc in range(2))
            if depth_max > MAX_REGION_DEPTH or grid_depth > MAX_GRID_DEPTH:
                self._regions_too_deep = getattr(self, "_regions_too_deep", 0) + 1
                continue
            out.append((focus, {"cells": region, "n_left_band_anchor_cells": n_left, "depth": float(stance.world_to_base(focus)[0]), "depth_max": depth_max}))
        # owner 10:47 ladder: regions 0.5–0.6 m ahead of the base first (the near cells keep the picks / places off the
        # arm's reach limit), then the rest by depth
        out.sort(key=lambda fi: abs(fi[1]["depth"] - REGION_DEPTH_PREFERRED))
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
        order = [int(k) for k in rng.permutation(len(self.manip_candidates))]

        def score(k: int) -> float:
            c = self.manip_candidates[k]
            depth = float(c.extras.get("focus_info", {}).get("depth", REGION_DEPTH_PREFERRED))
            lat = float(c.world_to_base(c.focus[:2])[1]) if c.focus is not None else GRID_LATERAL_PREFERRED
            # regions 0.5–0.6 m ahead and in front of the right shoulder first; among them the probe-verified trunks
            # (probe_place_depth 17:59: (0.5, −0.45) passes the x/t−45 pick + pre-place at every depth 0.50–0.80, (0.5, 0)
            # up to 0.65, (0.25, −0.3) best at 0.50–0.55; the tall bent trunk (0, −0.61) fails the tilted pre-places)
            trunk_pen = min(abs(float(c.translate) - t0) + abs(float(c.tilt) - i0) for t0, i0 in GOOD_TRUNKS)
            return abs(depth - REGION_DEPTH_PREFERRED) + 0.5 * abs(lat - GRID_LATERAL_PREFERRED) + 0.5 * trunk_pen

        order.sort(key=score)
        for k in order[:6]:
            cand = self.manip_candidates[k]
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
        region_all = list(cells)
        # 17:24 first-demo run: every pick-screening failure was a block at base-lateral 0.0 / -0.08 with the other blocks
        # to its right (the arm reaches over them), every pick at -0.16 / -0.24 passed; the isolated kinematics do not
        # depend on the lateral (probe_place_lat 18:25), so keep the blocks in front of the right shoulder when the region
        # holds enough such cells, else use the whole region
        lat_of = {id(xy): float(manip.world_to_base(xy)[1]) for xy in cells}
        window = [xy for xy in cells if GRID_LATERAL_WINDOW[0] <= lat_of[id(xy)] <= GRID_LATERAL_WINDOW[1]]
        if len(window) >= n:
            cells = window
        chosen: List[th.Tensor] = []

        def grid_cells(xy):
            return [xy + right * (cc * pitch) + toward * (rr * pitch) for rr in range(n_rows) for cc in range(n_cols)]

        def arm_score(xy) -> Tuple[int, float]:
            # the goal grid for the RIGHT arm first (owner 11:08): every slot on a right-band cell at this trunk, the grid
            # centre in front of the right shoulder (base-lateral GRID_LATERAL_PREFERRED: the arm does not reach over the
            # other blocks) and near REGION_DEPTH_PREFERRED (probe_place_depth 17:59: the x/t-45 pick + pre-place pass at
            # every depth 0.50-0.80 for trunk (0.5, -0.45), up to 0.65 for (0.5, 0)); the left-band brace feasibility is a
            # small bonus, never required (the 17:24 run anchored at lateral 0.0 through the left-band rule's remnant)
            gc = grid_cells(xy)
            b = manip.world_to_base(th.stack(gc).mean(dim=0))
            on_band = all(TT.in_band(self.bands, manip, g_, tol=1.0)[0] for g_ in gc)
            bonus = 0.05 if (self.left_bands is not None and self._anchor_braces_in_left_band(xy, manip, right, toward)) else 0.0
            return (0 if on_band else 1, abs(float(b[1]) - GRID_LATERAL_PREFERRED) + 0.5 * abs(float(b[0]) - REGION_DEPTH_PREFERRED) - bonus)

        anchor_xy = min(cells, key=arm_score)
        chosen.append(anchor_xy)
        s_anchor = img_score(anchor_xy)
        b_anchor = manip.world_to_base(th.stack(grid_cells(anchor_xy)).mean(dim=0))
        log.info("%s: goal grid anchored at base (%.2f, %+.2f) (score %s)", self.gp.alias, float(b_anchor[0]), float(b_anchor[1]), arm_score(anchor_xy))
        # the other blocks: cells right of / nearer than the preferred anchor first (it stays the top-left block in the init
        # image), then the rest of the window, then the whole region (the anchor is re-derived as the top-left of the final
        # set below; 21:37 first-demo run: the window + top-left filter alone left 2 of 4 cells -> 'layout' x8)
        tiers = [[xy for xy in cells if xy is not anchor_xy and img_score(xy) > s_anchor], [xy for xy in cells if xy is not anchor_xy],
                 [xy for xy in region_all if all(xy is not c for c in chosen)]]
        for tier in tiers:
            for xy in tier:
                if len(chosen) == n:
                    break
                if all(float(th.linalg.norm(xy - c)) >= self.edge * math.sqrt(2) + gap for c in chosen):
                    chosen.append(xy)
            if len(chosen) == n:
                break
        if len(chosen) < n:
            raise Reject("layout", f"only {len(chosen)} of {n} region cells keep >= {gap} m gaps")
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
            # prefer showing the goal face on top when it is still free (visible-face rule, first clause). Prepared, off by
            # default (owner 2026-09-16 14:38): ``generation.block_no_flip: true`` makes that preference certain — with
            # a pattern of distinct colours no block flips, every pick / place uses the same pitched tilt (the in-hand
            # flip's place approach is the open decision item)
            p_goal_up = 1.0 if self.gen("block_no_flip", False) else float(self.gen("goal_face_up_prob", 0.5))
            up_of[bi] = goal_face[bi] if goal_face[bi] in options and rng.random() < p_goal_up else options[int(rng.integers(len(options)))]
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
        # owner item 3 (release height, core 2c59893): the actual surface under every slot (a downward raycast with the
        # blocks ignored) — the place eef releases RELEASE_ABOVE_SURFACE above it (the slot target itself stays on the top)
        self.slot_surface_z = {}
        if hasattr(self.world, "surface_z_under"):
            blocks_all = [self.objects[names[bi]] for bi in range(n)]
            for bi in range(n):
                c_xy = self._centre(goal_pose[bi])[0][:2]
                hit = self.world.surface_z_under(c_xy, float(self.top.z) + 0.20, ignore=blocks_all)
                if hit is not None and float(self.top.z) - 0.02 <= float(hit["z"]) <= float(self.top.z) + 0.02:
                    self.slot_surface_z[bi] = float(hit["z"])
        # --- cube grasps, flips and brace points per block in placement order ---------------------------------------
        order = sorted(range(n), key=lambda bi: slot_of[bi])  # anchor first, then the slots in order
        records = []
        for step, bi in enumerate(order):
            rec = self._block_program(rng, manip, names, poses, goal_pose, plans, order, step, right, toward, table_yaw)
            records.append(rec)
        for bi in range(n):  # a table flip may have re-chosen the block's table-aligned goal yaw (goal face unchanged)
            slots[f"slot_{slot_of[bi]}"]["orientation"] = [float(v) for v in goal_pose[bi][1]]
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
        braces = {rec["instance"]: rec.get("brace", "none") for rec in records}
        self.braces_missing = any(v == "none" for v in braces.values())
        # the manipulation stance works with the right arm; the left arm joins only when some block carries a brace
        manip.arms = ["right", "left"] if any(v == "present" for v in braces.values()) else ["right"]
        self.sidecar.add_stance(manip)
        self.sidecar.blocks = records
        for bi in range(n):
            self.sidecar.describe(names[bi], "block", f"the block with the {self.faces.get(up_of[bi], up_of[bi])} face up" + (" (top-left)" if bi == anchor_i else ""))
        d.summary.update({"pattern": pid, "pattern_faces": pattern, "up_faces": {names[bi]: up_of[bi] for bi in range(n)}, "goal_faces": {names[bi]: goal_face[bi] for bi in range(n)},
                          "anchor_block": names[anchor_i], "table_yaw": table_yaw, "manipulation_candidate": k, "goal_face_visibility": {names[bi]: v for bi, v in vis.items()},
                          "placement_order": [names[bi] for bi in order], "brace": braces, "brace_reasons": {rec["instance"]: rec.get("brace_reasons", {}) for rec in records},
                          "region_depth_m": manip.extras.get("focus_info", {}).get("depth")})
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
        # Placed neighbours at the place pose (replay probe 19:00, the 17:24 first-demo run): with the slots edge + 1 mm
        # apart, the attached cube's coarse cuRobo sphere fit (1-4 spheres for a 2.6 cm dice, bulging ~1 cm beyond the
        # faces) collides with a touching placed neighbour at every class — 0/24 place poses pass with the neighbours
        # present, 14/24 with them parked — a model artefact, not geometry. The neighbours already at their slots are
        # therefore ignored for the attached place pose (like the support under a grasp, §3.6) PROVIDED the finger
        # clearance is guaranteed geometrically: the finger beside the block descends into the gap slot_gap + PLACE_MARGIN
        # (the block is released the margin away from every placed neighbour), which must hold the 8.4 mm fingertip
        # section plus 2 mm; otherwise the honest check stays and the block is rejected with the reason
        placed_neighbours = [self.objects[names[o]] for j, o in enumerate(order) if j < step]
        gap_needed = TT.FINGER_PROFILE[0][1] + 0.002  # the pad on the block face, the outer face 8.4 mm out (fingertip section)
        if placed_neighbours and float(self.g("block_gap_m")) + PLACE_MARGIN < gap_needed:
            raise Reject("finger_clearance", f"block {names[bi]}: slot gap {float(self.g('block_gap_m')) * 1000:.0f} mm + place margin {PLACE_MARGIN * 1000:.0f} mm < "
                                             f"{gap_needed * 1000:.1f} mm the closed finger needs beside a placed neighbour (set generation.block_gap_m >= "
                                             f"{(gap_needed - PLACE_MARGIN) * 1000:.0f} mm or record the push tail)")
        place_ignore = [support] + placed_neighbours
        # release pose: the slot pose lifted so the block bottom sits RELEASE_ABOVE_SURFACE above the raycast surface (the
        # goal target keeps the on-top pose; without a surface hit the 1 mm drop of the slot pose stays)
        surface_z = getattr(self, "slot_surface_z", {}).get(bi)
        if surface_z is not None:
            bottom = float(place_pose[0][2]) - self.world.bottom_offset(self.hull, place_pose[1])
            dz = max(0.0, surface_z + RELEASE_ABOVE_SURFACE - bottom)
            place_pose = (place_pose[0] + th.tensor([0.0, 0.0, dz]), place_pose[1])
        # candidate grasps: face pair (∥ flip axis for a flipping block; either horizontal pair otherwise) x wrist yaw x tilt pair
        cands: List[Tuple[str, float, float, float, int]] = []  # (face_pair, tilt_pick, tilt_place, yaw, flip_sign)
        if plan["flips"]:
            pair = plan["face_pair"]
            for th_deg in (theta, 30.0, 60.0):  # §6.5 θ = 45 [pilot] first, the 30 / 60 alternatives widen the pick classes
                for sign in (1, -1):
                    tilt_pick, tilt_place = TT.cube_tilt_pair(th_deg, sign)
                    for yaw in (table_yaw, table_yaw + math.pi / 2, table_yaw + math.pi, table_yaw - math.pi / 2):
                        cands.append((pair, tilt_pick, tilt_place, yaw, sign))
        else:
            up = TT.up_face(pick_pose[1])
            pairs = [ax for ax in "xyz" if ax != up[1]]
            # straight down first, then the pitched approaches of the band's generic classes (the same tilt at the pick
            # and the place: no reorientation in hand) — the far cells of the band pass only with a pitched approach
            for tilt in PICK_TILTS_DEG:  # owner 10:47 ladder: the pitched 30° / 45° picks first, straight down last
                for pair in pairs:
                    for yaw in (table_yaw, table_yaw + math.pi / 2, table_yaw + math.pi, table_yaw - math.pi / 2):
                        cands.append((pair, tilt, tilt, yaw, 0))
        # keep the downward pick approaches only (exactly one tilt sign for a flipping block), then screen pick + place pairs.
        # Tip clearance ladder (17:24 first-demo run: every place with the block attached failed the exact IK while the
        # picks passed — README "Backend facts": an object deep in the gripper self-collides with the non-exempt gripper
        # link once attached, so every attached check fails whatever the pose): the 1 mm pinch first, then 3 / 6 mm, and a
        # class whose attached cube self-collides (``IKChecker.attachable``, once per template per class) is skipped
        screened = []
        for pair, tilt_pick, tilt_place, yaw, sign in cands:
            for clearance in PICK_TIP_CLEARANCES:
                try:
                    g_pick = C.cube_grasp(self._centre(pick_pose), pair, tilt_pick, yaw, edge=self.edge, tip_clearance=clearance)
                except ValueError:
                    continue
                if not g_pick.downward:
                    continue
                if not self._class_attachable(block, pair, tilt_pick, clearance, g_pick.eef_in_block):
                    continue
                break
            else:
                continue
            # place eef: the rigid in-hand transform carried to the goal block pose
            place_eef = C.compose(self._centre(place_pose), g_pick.eef_in_block)
            if float(C.approach_axis(place_eef[1])[2]) > -0.1:
                continue  # the place approach must point down as well
            screened.append((pair, tilt_pick, tilt_place, yaw, sign, g_pick, place_eef))
        if not screened:
            raise Reject("grasp_unreachable", f"block {names[bi]}: no cube grasp with downward pick and place approaches")
        # classes whose wrist stays on the robot's side first (PLACE_FACTORS 19:45: at 0.7 m base-forward the pre-poses whose
        # lean puts the wrist beyond the block, 0.74 m forward and 11 cm up, fail the IK under every condition while the
        # grasp poses pass; the pre-grasp / pre-place reach decides at the far cells)
        screened.sort(key=lambda c: float(manip.world_to_base(IK.pre_pose(c[5].eef)[0][:2])[0]) + float(manip.world_to_base(IK.pre_pose(c[6])[0][:2])[0]))
        pre_picks = [[IK.pre_pose(c[5].eef), c[5].eef] for c in screened]
        ok_pick = self.ikc.screen_pairs(pre_picks, manip, "right", ignore_for_last=[support, block])
        idx = [i for i, s in enumerate(ok_pick.tolist()) if s]
        if not idx:
            self._log_pick_failure("screen", manip, names, bi, world_poses, screened, pre_picks, support, block)
            raise Reject("grasp_unreachable", f"block {names[bi]}: no pick passes the screening IK")
        chosen = None
        place_diag: List[str] = []
        for i in idx:  # sorted: the wrist-on-the-robot's-side classes first
            pair, tilt_pick, tilt_place, yaw, sign, g_pick, place_eef = screened[i]
            pre_pick = IK.pre_pose(g_pick.eef)
            res = self.ikc.exact([pre_pick, g_pick.eef], manip, "right", ignore_for_last=[support, block])
            if not res.ok:
                place_diag.append(f"{pair}/t{tilt_pick:+.0f}/y{math.degrees(yaw) % 360:.0f}: pick exact {res.reason}")
                continue
            # place (and pre-place) with the block in hand; the world: placed blocks at their slots, this block removed (attached)
            place_margin_pose = (place_eef[0] + _xy3(right) * PLACE_MARGIN + _xy3(toward) * PLACE_MARGIN, place_eef[1])
            pre_place = IK.pre_pose(place_margin_pose)
            with TT.posed_in_hand(self, "right", block, C.relative(g_pick.eef, pick_pose)):
                with attached(self.kit, "right", block):
                    res_p = self.ikc.exact([pre_place, place_margin_pose], manip, "right", ignore_for_last=place_ignore)
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
                place_diag.append(f"{pair}/t{tilt_pick:+.0f}/y{math.degrees(yaw) % 360:.0f}: {'place exact ' + res_p.reason if not res_p.ok else 'flip waypoints ' + r_w.reason}")
                if not res_p.ok and len(place_diag) <= 2:
                    self._place_failure_factors(manip, block, pick_pose, g_pick, pre_place, place_margin_pose, support, names, bi, world_poses)
                continue
            chosen = (pair, tilt_pick, tilt_place, yaw, sign, g_pick, place_margin_pose, pre_pick, pre_place, wps)
            break
        if chosen is None and plan["flips"] and self.gen("block_table_flip", False):
            # §6.5 flip-on-the-table (owner 2026-09-16 17:10, prep behind ``generation.block_table_flip``): grasp → place on a
            # free region cell in an intermediate orientation → re-grasp with the other pitch class → place at the slot
            legs = self._table_flip_chain(rng, manip, names, bi, poses, goal_pose, world_poses, right, toward, table_yaw, surface_z, plan)
            if legs is not None:
                return self._table_flip_record(names, bi, step, legs, plan, goal_pose, surface_z, manip, world_poses, order, right, toward)
        if chosen is None:
            self._log_pick_failure("place", manip, names, bi, world_poses, screened, pre_picks, support, block, extra={"per_class": place_diag[:12], "goal_slot_base":
                                   [round(float(v), 3) for v in manip.world_to_base(self._centre(place_pose)[0][:2]).tolist()],
                                   "placed_neighbours_base": {names[o]: [round(float(v), 3) for v in manip.world_to_base(self._centre(goal_pose[o])[0][:2]).tolist()] for j, o in enumerate(order) if j < step}})
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
                               "goal_face": TT.up_face(goal_pose[bi][1]), "slot": step, "release_surface_z": None if surface_z is None else round(float(surface_z), 4),
                               # the cube's geometric centre in the block's root frame (m): the ``isgblk`` root origin is
                               # off-centre; ``cube_grasp`` / the flip pivot take the centre pose = root pose ∘ (centre_offset, I)
                               "centre_offset": [float(v) for v in self.c_local.tolist()]}
        if wps:
            rec["waypoints"] = [pose_to_sidecar(w) for w in wps]
        # brace points (§6.5): row-left / column-top blocks braced by the closed left gripper while the right hand pushes.
        # Owner decision 2026-09-16 11:08 (same rule as axial): the brace and its paired push are recorded only when the
        # joint two-arm placement exists for this block (brace point in the left contact band at the manipulation trunk,
        # blade clear of the placed blocks, two-arm IK passing); otherwise the block is placed without a brace / push tail
        n_cols = int((self.g("grid") or [2, 2])[1])
        r, c = divmod(step, n_cols)
        world_after = dict(world_poses)
        world_after[names[bi]] = (goal_pose[bi][0] + _xy3(right) * PLACE_MARGIN + _xy3(toward) * PLACE_MARGIN, goal_pose[bi][1])
        self._teleport(world_after)
        outcomes: Dict[str, str] = {}
        if c > 0:  # push left against the row, brace the row-left block's left face
            left_bi = order[step - 1]
            outcomes["row"] = self._brace_push(rec, "row", names[bi], names[left_bi], world_after, direction=-right, brace_dir=right, manip=manip)
        if r > 0:  # push up (away from the robot) against the column, brace the column-top block's far face
            top_bi = order[step - n_cols]
            outcomes["col"] = self._brace_push(rec, "col", names[bi], names[top_bi], world_after, direction=toward * -1.0, brace_dir=toward, manip=manip)
        rec["brace"] = "present" if any(v == "present" for v in outcomes.values()) else ("none" if outcomes else "not_needed")
        rec["brace_reasons"] = outcomes
        return rec

    def _place_failure_factors(self, manip: Stance, block, pick_pose: TT.Pose, g_pick, pre_place: TT.Pose, place_eef: TT.Pose, support, names, bi, world_poses):
        """PLACE_FACTORS diagnostic (17:24 first-demo run: every place failed the pre-place exact IK while the replay
        probes pass it): re-solve the pre-place / place from the manipulation stance under isolated conditions — plain
        32-seed IK from the posture seed (no remembered seeds), the other blocks ignored, the block not attached, the
        support ignored — and log which passes."""
        try:
            others = [self.objects[n] for n in names if n != names[bi]]
            q0 = self.ikc.q_locked(manip)
            out: Dict[str, Any] = {"block": names[bi], "class": f"{g_pick.face_pair}/t{g_pick.tilt_deg:+.0f}",
                                   "pre_place_base": [round(float(v), 3) for v in manip.world_to_base(pre_place[0][:2]).tolist()] + [round(float(pre_place[0][2]) - float(self.top.z), 3)],
                                   "approach_fwd": round(float(th.dot(C.approach_axis(pre_place[1])[:2], manip.inward[:2])), 2)}
            with TT.posed_in_hand(self, "right", block, C.relative(g_pick.eef, pick_pose)):
                with attached(self.kit, "right", block):
                    out["attached: plain ik pre"] = self.kit.ik(pre_place, q0, arm="right") is not None
                    out["attached: plain ik place (support ignored)"] = None
                    with self.ikc.ignoring([support]):
                        out["attached: plain ik place (support ignored)"] = self.kit.ik(place_eef, q0, arm="right") is not None
                    with self.ikc.ignoring(others):
                        out["attached: plain ik pre, other blocks ignored"] = self.kit.ik(pre_place, q0, arm="right") is not None
                    with self.ikc.ignoring(others + [support]):
                        out["attached: plain ik pre, blocks + support ignored"] = self.kit.ik(pre_place, q0, arm="right") is not None
            with self.ikc.ignoring([block]):
                out["not attached: plain ik pre (block ignored)"] = self.kit.ik(pre_place, q0, arm="right") is not None
                with self.ikc.ignoring(others):
                    out["not attached: plain ik pre, all blocks ignored"] = self.kit.ik(pre_place, q0, arm="right") is not None
            log.info("%s: PLACE_FACTORS %s", self.gp.alias, json.dumps(out))
        except Exception as e:  # diagnostics only
            log.info("%s: PLACE_FACTORS failed: %r", self.gp.alias, e)

    def _class_attachable(self, block, pair: str, tilt: float, clearance: float, eef_in_block: TT.Pose) -> bool:
        """Once per template per (face pair, tilt, tip clearance): whether the cube attached per this class's rigid
        in-hand transform is free of self-collision with the non-exempt gripper links (``IKChecker.attachable``); the
        transform does not depend on the block's pose, so the answer is cached in ``self.attachable_classes`` and the
        dropped classes are listed in ``template_facts.cube_classes_not_attachable``."""
        key = (pair, round(float(tilt), 1), round(float(clearance), 4))
        cache = self.__dict__.setdefault("attachable_classes", {})
        if key not in cache:
            cache[key] = bool(self.ikc.attachable(block, eef_in_block, "right"))
            if not cache[key]:
                self.template_facts.setdefault("cube_classes_not_attachable", []).append(f"{pair}/t{tilt:+.0f}/c{clearance * 1000:.0f}mm")
                log.info("%s: cube class %s/t%+.0f at %.0f mm tip clearance: attached cube self-collides with the gripper; dropped", self.gp.alias, pair, tilt, clearance * 1000)
        return cache[key]

    # ------------------------------------------------------------------ §6.5 flip-on-the-table (prep, ``generation.block_table_flip``)

    def _leg_ik(self, manip: Stance, block, pose_a: TT.Pose, pose_b: TT.Pose, world_poses: Dict[str, TT.Pose], table_yaw: float,
                place_margin: Optional[th.Tensor] = None) -> Optional[Dict[str, Any]]:
        """One grasp → place leg of a block from world pose ``pose_a`` to ``pose_b`` (the in-hand rotation R = q_b·q_a⁻¹ is
        a yaw or a quarter turn about the pinch axis): pick classes = horizontal face pairs × PICK_TILTS_DEG × 4 wrist yaws,
        pre-pick + pick exact IK (support + block ignored for the pick), pre-place + place exact IK with the block attached,
        the in-hand waypoints (orbit for a yaw, flip for a quarter turn) screened. Returns the leg record or None."""
        support = self.obj("support")
        self._teleport(world_poses)
        q_a, q_b = TT._t(pose_a[1]), TT._t(pose_b[1])
        rot = C.qmul(q_b, C.qinv(q_a))  # world rotation of the block over the leg (xyzw)
        angle = float(C.quat_angle(rot, C.IDENTITY_QUAT))
        v = TT._t(rot)[:3]
        axis = (v / th.linalg.norm(v)).tolist() if float(th.linalg.norm(v)) > 1e-6 else None
        up = TT.up_face(q_a)
        pairs = [ax for ax in "xyz" if ax != up[1]]
        if axis is not None and abs(axis[2]) < 0.5:
            # a quarter turn about a horizontal axis: the pinch must be along that axis (the block turns about the pads)
            pairs = [p_ for p_ in pairs if abs(float(th.dot(C.quat_rotate(q_a, th.tensor(TT.FACE_AXES["+" + p_])), th.tensor(axis)))) > 0.9]
        cands = []
        for pair in pairs:
            for tilt in PICK_TILTS_DEG:
                for k in range(4):
                    yaw = table_yaw + k * math.pi / 2
                    g = None
                    for clearance in PICK_TIP_CLEARANCES:
                        try:
                            g_ = C.cube_grasp(self._centre(pose_a), pair, tilt, yaw, edge=self.edge, tip_clearance=clearance)
                        except ValueError:
                            continue
                        if g_.downward and self._class_attachable(block, pair, tilt, clearance, g_.eef_in_block):
                            g = g_
                            break
                    if g is None:
                        continue
                    place_eef = C.compose(self._centre(pose_b), g.eef_in_block)
                    if float(C.approach_axis(place_eef[1])[2]) > -0.1:
                        continue
                    cands.append((pair, tilt, yaw, g, place_eef))
        if not cands:
            return None
        pre_picks = [[IK.pre_pose(c[3].eef), c[3].eef] for c in cands]
        ok = self.ikc.screen_pairs(pre_picks, manip, "right", ignore_for_last=[support, block])
        for i in [i for i, s_ in enumerate(ok.tolist()) if s_][:6]:
            pair, tilt, yaw, g, place_eef = cands[i]
            pre_pick = IK.pre_pose(g.eef)
            if not self.ikc.exact([pre_pick, g.eef], manip, "right", ignore_for_last=[support, block]).ok:
                continue
            target_eef = place_eef if place_margin is None else (place_eef[0] + place_margin, place_eef[1])
            pre_place = IK.pre_pose(target_eef)
            with TT.posed_in_hand(self, "right", block, C.relative(g.eef, pose_a)):
                with attached(self.kit, "right", block):
                    if not self.ikc.exact([pre_place, target_eef], manip, "right", ignore_for_last=[support]).ok:
                        continue
                    lift = (g.eef[0] + th.tensor([0.0, 0.0, self._carry_height()]), g.eef[1])
                    pivot = self._centre(pose_a)[0] + th.tensor([0.0, 0.0, self._carry_height()])
                    wps = list(C.orbit_waypoints(lift, pivot, rot, n=3)) if angle > 1e-3 else []
                    if wps and not self.ikc.waypoints_ok([lift] + wps, manip, "right", ignore_all=[support]).ok:
                        continue
            q0 = C.qmul(g.eef[1], C.quat_from_axis_angle([0.0, 1.0, 0.0], -math.radians(g.tilt_deg)))
            return {"face_pair": pair, "tilt_pick_deg": float(g.tilt_deg), "tilt_place_deg": float(_tilt_about_y(q0, target_eef[1])), "yaw_deg": float(math.degrees(g.yaw)),
                    "pick": pose_to_sidecar(g.eef), "pre_pick": pose_to_sidecar(pre_pick), "place": pose_to_sidecar(target_eef), "pre_place": pose_to_sidecar(pre_place),
                    "carry_height": float(self._carry_height()), "waypoints": [pose_to_sidecar(w) for w in wps],
                    "rotation_axis": None if axis is None else [float(x) for x in axis], "rotation_deg": float(math.degrees(angle)),
                    "block_pose_after": {"position": [float(v) for v in pose_b[0]], "orientation": [float(v) for v in pose_b[1]]}}
        return None

    def _free_intermediate_cells(self, manip: Stance, names: List[str], bi: int, world_poses: Dict[str, TT.Pose], goal_pose: Dict[int, TT.Pose]) -> List[th.Tensor]:
        """Region cells free for the intermediate placement: not within ``spacing`` of another block (init or slot) nor of
        the goal grid, on the top; nearest the block's own cell first."""
        cells = manip.extras.get("focus_info", {}).get("cells")
        if cells is None:
            return []
        spacing = 0.5 * C.GRIPPER_OPEN_M + FINGER_HALF_THICKNESS + 0.5 * self.edge + 0.03
        occupied = [self._centre(p)[0][:2] for n, p in world_poses.items() if n != names[bi]] + [self._centre(g)[0][:2] for g in goal_pose.values()]
        own = self._centre(world_poses[names[bi]])[0][:2]
        out = [c for c in cells if all(float(th.linalg.norm(c - o)) >= spacing for o in occupied) and float(th.linalg.norm(c - own)) >= spacing
               and self.top.contains(c, 0.5 * self.edge + 0.02)]
        out.sort(key=lambda c: float(th.linalg.norm(c - own)))
        return out

    def _table_flip_chain(self, rng, manip: Stance, names: List[str], bi: int, poses: Dict[int, TT.Pose], goal_pose: Dict[int, TT.Pose], world_poses: Dict[str, TT.Pose],
                          right: th.Tensor, toward: th.Tensor, table_yaw: float, surface_z: Optional[float], plan: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Two-leg chain for a block whose goal face is not up. Two quarter turns about *different* horizontal axes compose
        to a 120° body-diagonal rotation and two about the same axis to 180°, so the only stable two-leg decomposition is
        **yaw first**: leg 1 yaw-aligns the block (ψ) plus k·90° in hand (an orbit, k ∈ {1, 2, 3}: the goal face moves to a
        side or to the back) and places it on a free region cell; leg 2 re-grasps it there with the pinch along the new
        flip axis and turns it by the quarter turn (either sense the IK admits) that brings the goal face up, placing it at
        the slot. The block's goal orientation becomes the resulting table-aligned quaternion (``goal_pose[bi]`` updated:
        the pattern constrains the up face only). Both legs pass the exact IK from the manipulation stance; None otherwise."""
        block = self.objects[names[bi]]
        q_init, gf = TT._t(poses[bi][1]), TT.up_face(goal_pose[bi][1])
        cells = self._free_intermediate_cells(manip, names, bi, world_poses, goal_pose)
        if not cells:
            return None
        psi = math.radians(float(plan["yaw_deg"]))
        for k in (1, 3, 2):
            q_mid = C.qmul(C.quat_from_yaw(psi + k * math.pi / 2), q_init)
            ends = []
            for ax in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
                for sgn in (1.0, -1.0):
                    q_end = C.qmul(C.quat_from_axis_angle(ax, sgn * math.pi / 2), q_mid)
                    if TT.up_face(q_end) == gf:
                        ends.append((ax, sgn, q_end))
            if not ends:
                continue
            for cell in cells[:3]:
                pose_mid = self._root_pose(cell, q_mid, 0.002)
                leg1 = self._leg_ik(manip, block, poses[bi], pose_mid, world_poses, table_yaw)
                if leg1 is None:
                    continue
                world_mid = dict(world_poses)
                world_mid[names[bi]] = pose_mid
                for ax, sgn, q_end in ends:
                    place_pose = (goal_pose[bi][0], q_end)
                    if surface_z is not None:
                        bottom = float(place_pose[0][2]) - self.world.bottom_offset(self.hull, place_pose[1])
                        place_pose = (place_pose[0] + th.tensor([0.0, 0.0, max(0.0, surface_z + RELEASE_ABOVE_SURFACE - bottom)]), place_pose[1])
                    leg2 = self._leg_ik(manip, block, pose_mid, place_pose, world_mid, table_yaw, place_margin=_xy3(right) * PLACE_MARGIN + _xy3(toward) * PLACE_MARGIN)
                    if leg2 is None:
                        continue
                    goal_pose[bi] = (goal_pose[bi][0], q_end)
                    log.info("%s: table flip for %s: yaw %+.0f deg then quarter turn about %s (%+.0f), intermediate cell base %s", self.gp.alias, names[bi],
                             math.degrees(psi) + k * 90.0, ax, sgn * 90.0, [round(float(v), 2) for v in manip.world_to_base(cell).tolist()])
                    leg1["kind"], leg2["kind"] = "intermediate (yaw first)", "slot (quarter turn)"
                    return [leg1, leg2]
        return None

    def _table_flip_record(self, names, bi, step, legs, plan, goal_pose, surface_z, manip, world_poses, order, right, toward) -> Dict[str, Any]:
        """Block record of a table-flipped block: the top-level pick / place fields mirror the LAST leg (re-grasp at the
        intermediate pose → slot) and ``legs`` carries both grasp → place legs in order (record shape proposed to the oracle
        agent: ``program_block_design`` iterates ``legs`` when present; each leg is its own FOV span, §3.9)."""
        last = legs[-1]
        rec: Dict[str, Any] = {"instance": names[bi], "order": step, "face_pair": last["face_pair"], "tilt_pick_deg": last["tilt_pick_deg"], "tilt_place_deg": last["tilt_place_deg"],
                               "yaw_deg": last["yaw_deg"], "yaw_align_deg": float(plan["yaw_deg"]), "flip_axis": None, "flip_sign": 0,
                               "pick": last["pick"], "pre_pick": last["pre_pick"], "place": last["place"], "pre_place": last["pre_place"], "carry_height": last["carry_height"],
                               "goal_face": TT.up_face(goal_pose[bi][1]), "slot": step, "release_surface_z": None if surface_z is None else round(float(surface_z), 4),
                               "centre_offset": [float(v) for v in self.c_local.tolist()], "flip_mode": "table", "legs": legs,
                               "intermediate_pose": legs[0]["block_pose_after"]}
        if last.get("waypoints"):
            rec["waypoints"] = last["waypoints"]
        rec["brace"], rec["brace_reasons"] = "none", {"table_flip": "braces not attempted after a table flip (prep)"}
        return rec

    def _log_pick_failure(self, stage: str, manip: Stance, names: List[str], bi: int, world_poses: Dict[str, TT.Pose], screened, pre_picks, support, block, extra=None):
        """Owner 2026-09-16 12:52 item 1: one INFO line (JSON) per failing block with everything the probe needs to replay
        it — trunk, base-frame centre, up face, yaw (deg, incl. the jitter), the other blocks' base-frame centres and
        centre distances, the pick classes tried and, for a screening failure, the pre / grasp screening masks."""
        try:
            c = self._centre(world_poses[names[bi]])
            yaw_deg = math.degrees(C.yaw_of(c[1])) if hasattr(C, "yaw_of") else float("nan")
            others = {}
            for n, p in world_poses.items():
                if n == names[bi]:
                    continue
                cc = self._centre(p)[0]
                others[n] = {"base": [round(float(v), 3) for v in manip.world_to_base(cc[:2]).tolist()], "dist": round(float(th.linalg.norm(cc[:2] - c[0][:2])), 3)}
            rec = {"stage": stage, "block": names[bi], "trunk": [manip.translate, round(manip.tilt, 2)], "stand_off": manip.stand_off,
                   "base_xy": [round(float(v), 3) for v in manip.world_to_base(c[0][:2]).tolist()], "up": TT.up_face(c[1]), "yaw_deg": round(yaw_deg, 1),
                   "others": others, "classes": [f"{p_}/t{t_:+.0f}/y{math.degrees(y_) % 360:.0f}" for p_, t_, _, y_, _, _, _ in screened]}
            if stage == "screen":
                pres = [pp[0] for pp in pre_picks]
                grasps = [pp[1] for pp in pre_picks]
                ok_pre = self.ikc.screen(pres, manip, "right")
                ok_g = self.ikc.screen(grasps, manip, "right", ignore=[support, block])
                rec["pre_ok"] = [rec["classes"][i] for i, o in enumerate(ok_pre.tolist()) if o]
                rec["grasp_ok_alone"] = [rec["classes"][i] for i, o in enumerate(ok_g.tolist()) if o]
                # the same grasp poses with every other block removed: the neighbours' share of the failure
                with self.ikc.ignoring([self.objects[n] for n in world_poses if n != names[bi]]):
                    ok_pre_nn = self.ikc.screen(pres, manip, "right")
                    ok_g_nn = self.ikc.screen(grasps, manip, "right", ignore=[support, block])
                rec["pre_ok_without_neighbours"] = int(ok_pre_nn.sum())
                rec["grasp_ok_without_neighbours"] = int(ok_g_nn.sum())
            if extra:
                rec.update(extra)
            log.info("%s: PICK_FAILURE %s", self.gp.alias, json.dumps(rec, default=str))
        except Exception as e:  # noqa: BLE001 - diagnostics must never change the sampling outcome
            log.info("%s: PICK_FAILURE logging failed: %r", self.gp.alias, e)

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
        z_blade = self._blade_height()
        side = th.cross(th.tensor([0.0, 0.0, 1.0]), face_normal, dim=0)
        face_z = (float(self.top.z), float(self.top.z) + self.edge)
        # closing axis along the face normal (the pad face pushes) or across it (the finger's side pushes): a top-down
        # pose with the closing axis lateral to the robot is infeasible at many cells (wrist limits), so both are tried.
        # The eef origin stands the closed fingers' half thickness (measured profile over the cube face above the
        # fingertips; pad face or finger side) outside the face — a contact, no clearance
        for pitch in (0.0, 30.0):  # the cubes sit 0.5–0.8 m ahead: the 45° / 60° classes are for the far cells
            a_vec = TT.pitched_approach(pitch, manip.forward)
            z_tip = z_blade - C.FINGERTIP_LENGTH * math.cos(math.radians(pitch))
            for c_axis, cname in ((face_normal, "pad"), (side, "side")):
                so = TT.blade_standoff(face_z, z_tip, pitch, 0.0, side=cname == "side")
                if so is None:
                    continue
                origin = th.tensor([float(centre[0]), float(centre[1]), z_blade]) + face_normal * (0.5 * self.edge + so[1])
                for c_sign in (1.0, -1.0):
                    yield (origin, C.eef_quat(a_vec, c_sign * c_axis)), a_vec, f"{prefix}_blade_{cname}_{pitch:.0f}{'' if c_sign > 0 else '-'}", so[1]

    def _brace_push(self, rec: Dict[str, Any], kind: str, pushed: str, braced: str, world_poses: Dict[str, TT.Pose], direction: th.Tensor, brace_dir: th.Tensor, manip: Stance) -> str:
        """Right-hand push of ``pushed`` along ``direction`` by the place margin while the closed left gripper braces
        ``braced`` on its face opposite to the push; pre-contact + contact IK for both arms (§4.6 step 6). Contacts:
        the top-down blade beside the face (the blocks are 2.5 cm cubes) with the pitched approach classes first, the
        tilted / horizontal side contacts (pads on the face at 3/4 height) as the fallback; the first push pose whose
        pre + init pass the exact IK is paired with the first brace pose that passes the two-arm check. Optional (owner
        2026-09-16 11:08): records ``push_<kind>`` / ``brace_<kind>`` and returns ``"present"`` only when the joint
        placement exists (brace point within a lattice cell of the left contact band at this trunk) and the two-arm IK
        passes; otherwise returns the reason (``no_joint_cell`` / ``push_ik_failed`` / ``brace_ik_failed``)."""
        support = self.obj("support")
        # joint placement gate: the brace point (half an edge + a finger outside the braced face) in the left band cell
        b_centre = self._centre(world_poses[braced])[0]
        b_dir3 = th.tensor([float(brace_dir[0]), float(brace_dir[1]), 0.0])
        b_dir3 = b_dir3 / th.linalg.norm(b_dir3)
        brace_pt_xy = (b_centre - b_dir3 * (0.5 * self.edge + FINGER_HALF_THICKNESS))[:2]
        if self.left_bands is None or not TT.in_band(self.left_bands, manip, brace_pt_xy, tol=1.0)[0]:
            return "no_joint_cell"
        dist = PLACE_MARGIN
        d3 = th.tensor([float(direction[0]), float(direction[1]), 0.0])
        d3 = d3 / th.linalg.norm(d3)
        z_axis = th.tensor([0.0, 0.0, 1.0])
        # push candidates: blades, then the tilted side contact
        push_cands = list(self._blade_poses(world_poses[pushed], direction, False, manip, "push"))
        p_pt, p_a = self._face_contact(world_poses[pushed], direction, into=False)
        c_axis = th.tensor([float(-direction[1]), float(direction[0]), 0.0])  # pads close across the push direction (horizontal)
        push_cands.append((C.contact_eef(p_pt, p_a, c_axis), p_a, "push_tilted", None))
        # brace candidates: blades, then the tilted side contact (either wrist turn) and B3's horizontal classes
        brace_cands = list(self._blade_poses(world_poses[braced], brace_dir, True, manip, "brace"))
        b_pt, b_a = self._face_contact(world_poses[braced], brace_dir, into=True)
        c_axis_b = th.tensor([float(-brace_dir[1]), float(brace_dir[0]), 0.0])
        a_h = th.tensor([float(brace_dir[0]), float(brace_dir[1]), 0.0])
        a_h = a_h / th.linalg.norm(a_h)
        for a_vec, c_vec, name in ((b_a, c_axis_b, "tilted_h"), (b_a, -c_axis_b, "tilted_h-"), (a_h, c_axis_b, "side_h"), (a_h, z_axis, "side_v")):
            brace_cands.append((C.contact_eef(b_pt, a_vec, c_vec), a_vec, name, None))
        push_fails, brace_fails = [], []
        chosen = None
        for push_init, push_a, p_name, p_half in push_cands:
            push_goal = (push_init[0] + d3 * dist, push_init[1])
            push_pre = C.pre_pose(push_init)
            # the push init / goal touch the block: init is exact-IK-checked with the block ignored, then pre + goal with the brace arm
            r_init = self.ikc.exact([push_pre, push_init], manip, "right", ignore_for_last=[support, self.objects[pushed]], cache=False)
            if not r_init.ok:
                push_fails.append(f"{p_name}: {r_init.reason}")
                continue
            for b, b_a_vec, b_name, b_half in brace_cands:
                b_pre = C.pre_pose(b)
                res = TT.brace_second_arm(self, manip, "right", [push_pre, push_goal], "left", [b_pre, b], ignore_work_last=[support, self.objects[pushed]],
                                          ignore_brace_last=[support, self.objects[braced]])
                if res.ok:
                    chosen = (push_pre, push_init, push_goal, b_pre, b, b_a_vec, f"{p_name}+{b_name}", p_half, b_half)
                    break
                brace_fails.append(f"{p_name}+{b_name}: {res.reason}")
            if chosen is not None:
                break
        if chosen is None:
            log.info("%s: %s brace / push of %s skipped: %s", self.gp.alias, kind, pushed, (sorted(set(brace_fails)) or sorted(set(push_fails)))[:2])
            return "brace_ik_failed" if brace_fails else "push_ik_failed"
        push_pre, push_init, push_goal, brace_pre, brace, brace_a, used, p_half, b_half = chosen
        rec[f"push_{kind}"] = ContactRecord("push", "right", push_pre, push_init, None, push_goal, d3, dist, object=pushed).to_sidecar()
        if p_half is not None:
            rec[f"push_{kind}"]["blade_half_thickness_m"] = float(p_half)
        rec[f"brace_{kind}"] = TT.BraceRecord("brace", "left", brace_pre, brace, direction=brace_a, brace_kind="row_left" if kind == "row" else "column_top",
                                              object=braced, extras={"pose_class": used, "blade_half_thickness_m": None if b_half is None else float(b_half),
                                                                     "stop_clearance_m": 0.0}).to_sidecar()
        return "present"

    # ------------------------------------------------------------------ program check (blocks without a brace / push tail)

    def program_check(self, draft: Draft) -> Dict[str, Any]:
        """§3.5 program check. Blocks without ``brace_row`` / ``brace_col`` (+ pushes) are not yet accepted by the F6 program
        (the oracle agent is making the braces optional); until that lands, with ``ISG_PROGRAM_CHECK=stub`` set by the smoke
        command a rejection that only asks for the brace / push records is recorded as ``mode: stub-brace-optional``."""
        try:
            return super().program_check(draft)
        except Reject as e:
            msg = str(e)
            optional = getattr(self, "braces_missing", False) and ("brace" in msg or "push" in msg) and TT.program_check_stub()
            if not optional:
                raise
            log.warning("%s: program check asks for the brace / push tail of a brace-less block (%s); recorded as stub-brace-optional (ISG_PROGRAM_CHECK=stub)",
                        self.gp.alias, msg[:200])
            report = {"mode": "stub-brace-optional", "problems": [], "note": "brace-less blocks not yet accepted by the oracle program; structural check skipped", "error": msg[:300]}
            self.last_program_check = report
            draft.summary["program_check"] = report
            return report

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
