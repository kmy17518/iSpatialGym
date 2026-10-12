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

import copy
import itertools
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import navigation as NAV
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.v2 import ik as IK
from ispatialgym.sampling.v2 import orient as O
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.core import SamplerV2, annotate_height_extra, lift_contact_height, register
from ispatialgym.sampling.v2.navigation import attached
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, Stance, pose_from_sidecar, pose_to_sidecar

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
PLACE_MARGIN = 0.003  # was 5 mm: with a 1 cm slot tolerance the 7 mm diagonal release offset left 3 mm for everything else (0919-183037: 1.20 cm)
#: the oracle's retreat after a release: the ``Place`` action's own 10 cm leg (actions.RETREAT_DIST) followed by the program's
#: ``Retreat(distance=RETREAT_LITTLE)`` 5 cm further from that end (pod runs 0919-171511 / 174309: 'retreat ... the end pose has
#: no solution on its own' at 15 cm above a set-down whose 10 cm retreat the sampler had checked)
RETREAT_DIST_M = 0.10 + 0.05
#: the arm links kept ``generation.arm_link_margin_m`` (default 12 mm) clear of the world at every pick / place configuration - the
#: QA collision monitor fails a grasp on any arm-table contact (demo 0919-164732: ``right_arm_link5`` brushed the table for 2 steps
#: in a 45 deg pitched pinch every margin-0 check had passed); the wrist links 6 / 7 and the hand work close to the table by design
ARM_CLEAR_LINKS = ("right_arm_link1", "right_arm_link2", "right_arm_link3", "right_arm_link4", "right_arm_link5")
#: the re-pose after a between-block base move (``generation.repose_after_base_move``): rounds of cuRobo joint-space planning to the
#: pre-grasp configuration, and the clearance the previous block's end configuration needs at the new stance to be the plan's start
#: (the program check's travel sweeps keep the arm as left at 1 cm, ``program_check.DEPARTURE_SWEEP_MARGIN``)
REPOSE_PLAN_ROUNDS = 3
REPOSE_START_MARGIN = 0.01
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
#: the board variant (owner 2026-09-19 21:47, option 2): the blocks are released above the board's rim and drop into the pocket -
#: ``generation.release_above_surface_m`` raises the release over the pocket floor (rim 12 mm + 4 mm)
MAX_MANIP_TRANSLATE = 0.5  # manipulation trunks deeper than this put the tilted picks' pre-grasps out of reach (PICK_FAILURE 13:20)
PICK_TILTS_DEG = (30.0, -30.0, 45.0, -45.0, 0.0)
PICK_TIP_CLEARANCE = 0.001
PICK_TIP_CLEARANCES = (0.001, 0.003, 0.006)  # the ladder: deepest pinch first, shallower when the attached cube self-collides

# ---------------------------------------------------------------------------------------------------------------------
# In-hand flip (owner 2026-09-19): the flip is one 90 deg wrist roll about the pinch axis between a steep side pick and a
# near-top-down place. Probe /tmp/blkflip (2026-09-19, breakfast table, right arm, single cube): a flip about the LATERAL
# table axis (goal face toward / away from the robot) has no feasible class - the wrist roll the arm accepts at the place
# (camera toward the robot) puts the wrist camera into the table at the pick for leans >= 55 deg, and shallower leans end
# in a 45 deg over-reach at the place that no cell passes (0 / 28 cells at every trunk). A flip about the FORWARD axis
# (goal face toward the robot's RIGHT, pinch across the near / far faces, hand from the block's right leaning left 60-70
# deg, camera up, place leaning right 20-30 deg with the hand left of the slot) passes the whole chain at cells 0.55-0.70 m
# ahead and 0.15-0.35 m right of the base for trunks (0.5, -0.3), (0.75, 0 / -0.3), (0.95, 0), (0.5, -0.45); the mirror
# (goal face toward the LEFT, hand from the block's left) passes <= 2 / 35 cells. Hence: flipping blocks show their goal
# face to the robot's right, they sit LEFT of the init camera so that face is visible (>= 50 %), and the robot moves along
# the edge to a stance from which the block lies 0.15-0.35 m right of its centre line (per-block stances).
#: flip directions (the side the goal face faces at init, table axes in the stance frame) tried per flipping block, in order
FLIP_DIRECTIONS = ("right",)
#: pick leans from the vertical (deg) of the flipping pick, in order of preference (probe: 60 and 70 pass most cells)
FLIP_LEANS_DEG = (70.0, 60.0, 65.0, 55.0, 45.0)
#: (translate, tilt) trunks the flip chain passes from (probe), added to the stance pool next to the Phase 2 survivors
FLIP_TRUNKS = ((0.75, -0.3), (0.5, -0.3), (0.75, 0.0), (0.5, -0.45), (0.95, 0.0), (0.25, -0.6), (0.5, 0.0), (0.65, -0.3), (0.85, -0.3), (0.75, -0.4), (0.75, -0.2))
#: lateral shifts (m, + = the robot's left) of the per-block stances along the edge, tried in this order after the current stance
STANCE_LATERAL_SHIFTS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, -0.05, -0.1, 0.4)
#: base-frame (depth, lateral) sweet spot of a flipping block's cell / slot and of a non-flipping one (probe chain cells)
FLIP_CELL_SWEET = (0.63, -0.18)  # exact-checker probe 2026-09-19: l70 at trunk (0.75, -0.3) passes pick+flip+place(6 cm left) at depth 0.55-0.65, lateral -0.15..-0.25
PICK_CELL_SWEET = (0.55, -0.20)
#: a flipping block's cell must lie this far right of the stance's centre line (probe: no chain at lateral > -0.10)
FLIP_CELL_MAX_LATERAL = -0.10
#: a flipping block's cell must lie at least this far in from the table edge the robot works from (m): pod run 0919-081444, the
#: flip-in-place + re-grasp passed at every try from the cell 29 cm in (manip depth 0.64 at stand-off 0.35) and the re-grasp of
#: the turned block failed at every class and stance from the cells 13 cm in (depth 0.48 at 0.35, 0.58 at 0.45)
FLIP_CELL_MIN_EDGE_DEPTH = 0.10  # 0.22 (0919-090517 / 091857) left every region with a single flip cell; the pool's 0.40 / 0.50 stand-offs put the 13 cm cell at the proven base depth
#: step (m) of the flip-room move of the init along the edge when the minimum move leaves the board off the top (owner 2026-10-12)
FLIP_ROOM_STEP = 0.01
#: the puzzle board's (base depth, lateral) positions of the manipulation stance when the alias has no ``board_base_xy_candidates``
BOARD_BASE_XY = ((0.46, -0.02), (0.44, -0.10), (0.50, 0.04), (0.60, 0.05))
#: head-camera margin (fraction of the image) of the fingertips along the chain (the oracle's FOV monitor uses 5 %)
CHAIN_FOV_MARGIN = 0.06
#: hand collision spheres in the eef frame (``r1pro_description_curobo_default.yaml`` mapped through EEF_IN_GRIPPER, as in
#: ``iors_small.HAND_SPHERES_EEF``) plus the finger spheres approximated along the finger axis (the tip sphere r 1 cm centred
#: 1 cm behind the fingertip plane, then r 8 / 7 / 5 mm every 1.5-2 cm toward the palm): what the IK does NOT see against
#: the table at a grasp / place pose (the support is ignored there), checked explicitly against the top plane
_L = C.FINGERTIP_LENGTH
HAND_SPHERES_EEF: Tuple[Tuple[Tuple[float, float, float], float], ...] = (
    ((-0.056, -0.002, -0.076), 0.018), ((-0.056, -0.030, -0.076), 0.018), ((-0.056, -0.057, -0.076), 0.018),  # wrist camera
    ((-0.015, -0.042, -0.0675), 0.015), ((0.015, -0.042, -0.0675), 0.015), ((-0.015, 0.042, -0.0675), 0.015), ((0.015, 0.042, -0.0675), 0.015),  # knuckles
    ((0.0, 0.0, -0.097), 0.04),  # gripper body
) + tuple(((0.0, sy * 0.02, _L - dz), r) for sy in (-1.0, 1.0) for dz, r in ((0.010, 0.010), (0.030, 0.008), (0.045, 0.007), (0.060, 0.005)))
_HAND_SPHERE_C = th.tensor([c for c, _ in HAND_SPHERES_EEF], dtype=th.float32)
_HAND_SPHERE_R = th.tensor([r for _, r in HAND_SPHERES_EEF], dtype=th.float32)


def _orient_extra(decision: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``{"orient": decision}`` when the contact class had two wrist turns (``orient.choose``), else nothing."""
    if decision is None or len([o for o in decision.get("options", []) if o.get("source") == "ik"]) < 2:
        return {}
    return {"orient": decision}


def hand_table_clearance(eef: TT.Pose, top_z: float) -> float:
    """Smallest clearance (m) of the hand spheres above the support plane ``top_z`` for an eef pose (negative = a sphere
    dips into the table: the wrist camera of a steep camera-down pick, the gripper body of a horizontal one)."""
    pts = C.transform_points((TT._t(eef[0]), TT._t(eef[1])), _HAND_SPHERE_C)
    return float((pts[:, 2] - _HAND_SPHERE_R).min()) - float(top_z)


def flip_tip_clearance(lean_deg: float) -> float:
    """Fingertip clearance above the table for a pick leaning ``lean_deg`` from the vertical: the tip sphere (r 1 cm, 1 cm
    behind the fingertip plane) must clear the table, ``0.01 (1 - cos lean)`` plus 2 mm, at least 3 mm."""
    return max(0.003, 0.002 + 0.01 * (1.0 - math.cos(math.radians(float(lean_deg)))))


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
                init = self._init_with_room_for_flips(init)
                # D-009 shape for blocks (default = the tabletop classes, 20 / 30 deg since owner 2026-10-11): with the region depth
                # caps (init cells <= 0.80 m, slots <= 0.70 m) a 20 deg cone at the arm-friendly trunks (0.5, -0.3..-0.45) meets the
                # top beyond the caps and only the translate-0.95 trunks survive (blk_diag2 22:28: 2 survivors, both (0.95, 0 / -0.16),
                # every pick pre-grasp out of reach from that height); the 30 deg loose class lets those trunks see regions at
                # 0.55-0.70 m
                angle_classes = TT.phase2_angle_classes(self)
                cands = []
                # owner 2026-09-19 (in-hand flip about the forward axis, see FLIP_DIRECTIONS): the manipulation stance is scanned
                # at the init lateral AND shifted along the edge to the robot's LEFT (``generation.manip_lateral_shifts_m``): the
                # flipping blocks lie 0.15-0.35 m right of the manipulation base for the right arm's flip class and, seen from
                # the unshifted init camera, left of its centre line - the only way their right-facing goal face is visible
                # the split cone (generation.phase2_split_cone) searches the shifts along the edge itself, from the unshifted init
                split = TT.split_cone_opts(self)
                for shift in (self._manip_lateral_shifts(init) if split is None else [0.0]):
                    init_k = init if abs(shift) < 1e-9 else self._shifted_along_edge(init, shift, "init")
                    if init_k is None:
                        continue
                    ck = TT.phase2_manipulation_stance(self, init_k, 0.5 * self.edge, target_height=self.edge, focus_candidates=self._regions, in_frame_fn=self._region_in_frame,
                                                       bands=bands, top=self.top, erosion_extra=0.02 + self.edge, focus_z=self.top.z + 0.5 * self.edge,
                                                       angle_classes_deg=angle_classes, split_cone=split)
                    for c in ck:
                        # a split-cone survivor keeps its own shift; the plain Phase 2's stances copy init_k's extras, whose
                        # lateral_shift_m may be the flip-room move of the init itself
                        if "_position" not in c.extras:
                            c.extras["lateral_shift_m"] = float(shift)
                    cands += ck
                if not cands:
                    raise Reject("no_manipulation_stance", f"Phase 2 left no region survivor ({getattr(self, '_regions_too_deep', 0)} regions dropped: init cells beyond "
                                                            f"{MAX_REGION_DEPTH:.2f} m or goal slots beyond {MAX_GRID_DEPTH:.2f} m base-forward)")
                # PICK_FAILURE diagnostics (2026-09-16 13:20): every failing pick sat at translate 0.75–0.95 with the pre-grasp of a
                # tilted pick out of reach (the grasp itself feasible), while the probe passes at (0.5, 0) / (0.75, −0.3):
                # keep the shallow trunks when Phase 2 offers them. The flip probe (2026-09-19) passes the flip chain at
                # translate 0.75 / 0.95 as well, so with flips required the deep trunks stay in the pool
                shallow = [c for c in cands if float(c.translate) <= MAX_MANIP_TRANSLATE]
                if shallow and not self._flips_expected():
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
        if getattr(self, "_flip_room", None) is not None:
            self.template_facts["init_flip_room_move"] = self._flip_room

    # -- Phase 2 hooks (§4.5 blocks) --------------------------------------------------------------------------------

    def _pitch(self) -> float:
        return self.edge + float(self.g("block_gap_m"))

    def _flip_directions(self) -> List[str]:
        return [str(v) for v in (self.gen("flip_directions", None) or FLIP_DIRECTIONS)]

    def _n_flips(self) -> int:
        """Blocks that must flip for the configured pattern(s): the up faces are distinct, so of the blocks sharing a goal
        colour only one shows it up (``block_no_flip`` makes that preference certain) - ``n - distinct colours`` at least."""
        patterns = self.g("patterns") or {}
        if not patterns:
            return 0
        pats = [list(patterns[self.variant])] if self.variant and self.variant in patterns else [list(p) for p in patterns.values()]
        return max(len(p) - len(set(p)) for p in pats)

    def _flips_expected(self) -> bool:
        return self._n_flips() > 0 and (bool(self.gen("block_no_flip", False)) or float(self.gen("goal_face_up_prob", 0.5)) > 0.0)

    def _flip_cell_ok(self, c: th.Tensor, manip: Stance, init: Stance, primary_dir: str) -> bool:
        """Owner 2026-09-19 (flips about the forward axis): a block may flip only where the right arm's flip class AND the
        re-grasp of the turned block reach - its cell at least ``-FLIP_CELL_MAX_LATERAL`` right of the manipulation base's
        centre line and ``FLIP_CELL_MIN_EDGE_DEPTH`` in from the edge - and where its goal face (toward the robot's right) is
        seen from the init camera: left of the camera's centre line by ``generation.flip_face_visibility_lateral_m``."""
        sgn = 1.0 if primary_dir == "right" else -1.0
        bc = manip.world_to_base(c)
        so = float(manip.stand_off if manip.stand_off is not None else 0.35)
        return (float(bc[1]) * sgn <= FLIP_CELL_MAX_LATERAL and float(init.world_to_base(c)[1]) * sgn >= float(self.gen("flip_face_visibility_lateral_m", 0.07))
                and float(bc[0]) - so >= float(self.gen("flip_cell_min_edge_depth_m", FLIP_CELL_MIN_EDGE_DEPTH)))

    def _flip_plan(self, cs: List[th.Tensor], img_score, slot_faces: List[str], manip: Stance, init: Stance, primary_dir: str, flip_mode: bool) -> Optional[set]:
        """Which blocks of the cell set ``cs`` flip: the block of rank ``i`` in the init image (``img_score``, top-left first) gets
        the goal face ``slot_faces[i]``; per goal colour exactly one block shows it up (a non-eligible cell when there is one), every
        other block of that colour flips and must be flip-eligible; None when no such assignment exists (the spec's image-order
        assignment, every block eligible, when no flip direction constrains the cells)."""
        sc = [img_score(c) for c in cs]
        oi = sorted(range(len(cs)), key=lambda i: sc[i])
        gf = {bi: slot_faces[si] for si, bi in enumerate(oi)}
        anchor_bi = oi[0]
        groups: Dict[str, List[int]] = {}
        for bi, f in gf.items():
            groups.setdefault(f, []).append(bi)
        flips: set = set()
        for f, members in groups.items():
            if len(members) == 1:
                continue
            if not flip_mode:
                flips.update(members[1:])  # image order: the first shows the face, the rest flip
                continue
            elig = [bi for bi in members if self._flip_cell_ok(cs[bi], manip, init, primary_dir)]
            non = [bi for bi in members if bi not in elig]
            if len(non) > 1:
                return None  # two blocks of one colour that cannot flip
            if non:
                up = non[0]
            elif anchor_bi in elig and not self._has_board():
                # the anchor's slot IS its cell: its flip is placed where it was picked (the one flip the exact chain passes
                # broadly), so the anchor flips and the other block of its colour shows the face
                up = next(bi for bi in elig if bi != anchor_bi)
            else:
                # the FARTHEST one shows its face (the flip zone lies at depth 0.55-0.65 of the near stand-offs)
                up = max(elig, key=lambda bi: float(manip.world_to_base(cs[bi])[0]))
            flips.update(bi for bi in members if bi != up)
        # the flip class approaches from the block's right (the side the goal face shows), low over the table: no other
        # block may stand on that side within the approach's sweep (probe run 2026-09-19 03:1x: 'collides with
        # targets_toy_dice_isgblk_2' along the approach of a flipping block with a neighbour 10 cm to its right)
        sgn = 1.0 if primary_dir == "right" else -1.0
        clear_lat, clear_depth = float(self.gen("flip_approach_clear_lateral_m", 0.20)), float(self.gen("flip_approach_clear_depth_m", 0.12))
        for bi in flips:
            bf = manip.world_to_base(cs[bi])
            for bj in range(len(cs)):
                if bj == bi:
                    continue
                bo = manip.world_to_base(cs[bj])
                d_lat = (float(bf[1]) - float(bo[1])) * sgn  # > 0: the other block lies on the approach side
                if 0.0 < d_lat < clear_lat and abs(float(bo[0]) - float(bf[0])) < clear_depth:
                    return None
        return flips

    def _has_flip_set(self, cells: List[th.Tensor], n: int, img_score, slot_faces: List[str], manip: Stance, init: Stance, primary_dir: str,
                      flip_mode: bool) -> bool:
        """Whether ``n`` of ``cells`` keep the scatter gaps (``edge·√2 + scatter_min_gap_m``) and have a :meth:`_flip_plan` - the
        test of the instance's exhaustive cell-set search, which every set the layout keeps passes."""
        gap = float(self.g("scatter_min_gap_m"))
        for combo in itertools.combinations(range(len(cells)), n):
            cs = [cells[i] for i in combo]
            if any(float(th.linalg.norm(cs[a] - cs[b])) < self.edge * math.sqrt(2) + gap for a in range(n) for b in range(a + 1, n)):
                continue
            if self._flip_plan(sorted(cs, key=img_score), img_score, slot_faces, manip, init, primary_dir, flip_mode) is not None:
                return True
        return False

    def _n_flip_cells(self, manip: Stance, init: Stance) -> int:
        """Flip-eligible cells of a manipulation candidate's region."""
        cells = manip.extras.get("focus_info", {}).get("cells")
        if cells is None or not self._flips_expected() or self._flip_directions()[0] not in ("right", "left"):
            return 0
        return sum(1 for c in cells if self._flip_cell_ok(c, manip, init, self._flip_directions()[0]))

    def _manip_lateral_shifts(self, init: Optional[Stance] = None) -> List[float]:
        """Lateral shifts (m, + = the robot's left) of the manipulation stance scan: 0 (the init lateral, spec §4.5) and,
        when flipping blocks are expected with the goal face toward the right, the shifts that put the flip cells (0.15-0.35
        m right of the manipulation base) left of the init camera's centre line - those that keep the shifted lateral inside
        the init's admissible ``lateral_bounds`` when ``init`` is given."""
        shifts = [0.0]
        if self._flips_expected() and self._flip_directions()[0] in ("right", "left"):
            sign = 1.0 if self._flip_directions()[0] == "right" else -1.0
            bounds = (init.extras.get("lateral_bounds") if init is not None else None) or None
            for v in (self.gen("manip_lateral_shifts_m", None) or (0.35, 0.45, 0.25)):
                s = sign * abs(float(v))
                if bounds is not None and not (float(bounds[0]) - 1e-6 <= float(init.lateral) + s <= float(bounds[1]) + 1e-6):
                    continue
                if all(abs(s - o) > 1e-6 for o in shifts):
                    shifts.append(s)
        return shifts

    def _init_with_room_for_flips(self, init: Stance) -> Stance:
        """Phase 1c draws the init lateral anywhere in its admissible range; with right-facing flips the manipulation
        stance must stand ``manip_lateral_shifts_m[0]`` further left, so an init drawn too far left is moved right along the
        edge (same trunk / stand-off) until that shift fits the range (the in-frame / visibility / collision checks of the
        instance validate the moved init); unchanged when the shift fits or no flips are expected.

        With a puzzle board (owner 2026-10-12) the move is the smallest one, from that minimum on in ``FLIP_ROOM_STEP`` steps with
        the init kept inside its range, after which the manipulation stance has a ``board_base_xy_candidates`` position on the top
        at every Phase 2 stand-off (:meth:`_board_fit`) - else the smallest with the most such stand-offs: at the end of the range
        the positions that stand-off 0.55 leaves on the top by depth are off its end (s3 c1_k1: every attempt). An init whose
        shift fits the range but whose manipulation stance has a stand-off without a position on the top is searched the same way
        from its drawn lateral (and kept when no move gives more stand-offs). The drawn and the moved lateral, the reason and the
        board fit go to ``template_facts.init_flip_room_move``."""
        self._flip_room = None
        if not (self._flips_expected() and self._flip_directions()[0] in ("right", "left")):
            return init
        bounds = init.extras.get("lateral_bounds")
        if not bounds:
            return init
        sign = 1.0 if self._flip_directions()[0] == "right" else -1.0
        need = sign * abs(float((self.gen("manip_lateral_shifts_m", None) or (0.35, 0.45, 0.25))[0]))
        lo, hi = float(bounds[0]), float(bounds[1])
        target = float(init.lateral) + need
        excess = (target - hi) if target > hi else ((lo - target) if target < lo else 0.0)
        half = 0.5 * float(self._board_geometry()["plate_m"]) if self._has_board() else None
        rec: Dict[str, Any] = {"lateral_drawn": round(float(init.lateral), 4), "lateral_bounds": [round(lo, 4), round(hi, 4)], "manipulation_shift_m": need}
        fit_drawn = self._board_fit(init, need, half) if (half is not None and excess <= 1e-6) else None
        if excess <= 1e-6 and (fit_drawn is None or all(fit_drawn.values())):
            self._flip_room = {**rec, "lateral": rec["lateral_drawn"], "move_m": 0.0, "reason": f"the {need:+.2f} m manipulation shift fits the lateral range"
                               + (" and its manipulation stance has a board position on the top at every Phase 2 stand-off" if fit_drawn else ""),
                               **({"board_fit": fit_drawn} if fit_drawn is not None else {})}
            return init
        # the shift fits but a stand-off has no board position on the top (s3 c1_k0 / bd01_board_rc k5: 0.15 + 0.35 on the 1.16 m
        # edge): the same search, from the drawn lateral on
        excess = max(excess, 0.0)
        # the minimum move first (no board: the only one), then further along the edge while the moved init stays inside its range
        m_max = (float(init.lateral) - lo) if sign > 0 else (hi - float(init.lateral))
        n_moves = 1 if half is None else 1 + max(0, int(math.floor((m_max - excess) / FLIP_ROOM_STEP + 1e-9)))
        best, fit0 = None, None
        for k in range(n_moves):
            m = excess + k * FLIP_ROOM_STEP
            cand = init if m < 1e-9 else self._shifted_along_edge(init, -sign * m, "init")
            if cand is None:
                continue
            fit = self._board_fit(cand, need, half) if half is not None else {}
            if k == 0:
                fit0 = fit
            n_fit = sum(1 for v in fit.values() if v)
            if best is None or n_fit > best[1]:
                best = (m, n_fit, cand, fit)
            if half is None or n_fit == len(TT.STAND_OFFS):
                break
        if best is None:
            log.info("%s: init lateral %.2f leaves no room for the %.2f m manipulation shift and cannot be moved (off the free floor)", self.gp.alias, init.lateral, need)
            self._flip_room = {**rec, "lateral": rec["lateral_drawn"], "move_m": 0.0,
                               "reason": f"the {need:+.2f} m manipulation shift leaves the lateral range and every move puts the init off the free floor"}
            return init
        m, n_fit, moved, fit = best
        if m < 1e-9:
            self._flip_room = {**rec, "lateral": rec["lateral_drawn"], "move_m": 0.0, "board_fit": fit,
                               "reason": f"the {need:+.2f} m manipulation shift fits the lateral range; its manipulation stance has a board position on the top at "
                                         f"{n_fit} of {len(TT.STAND_OFFS)} Phase 2 stand-offs, no move along the range gives more"}
            return init
        moved.extras.update(init.extras)
        moved.extras["flip_room_shift_m"] = float(-sign * m)
        reason = f"the {need:+.2f} m manipulation shift {'leaves' if excess > 1e-9 else 'fits'} the lateral range [{lo:.2f}, {hi:.2f}] at {init.lateral:.2f}"
        if half is not None:
            if abs(m - excess) < 1e-9:
                reason += "; the minimum move puts a board position on the top at every Phase 2 stand-off" if n_fit == len(TT.STAND_OFFS) else \
                    f"; the minimum move leaves a board position on the top at {n_fit} of {len(TT.STAND_OFFS)} Phase 2 stand-offs, no further move gives more"
            else:
                bare = [so for so, v in (fit0 or {}).items() if not v]
                reason += ((f"; the minimum move {-sign * excess:+.3f} m leaves" if excess > 1e-9 else ", but its manipulation stance has")
                           + f" no board position on the top at stand-off {', '.join(bare) or '-'}"
                           + (" (or puts the init off the free floor)" if fit0 is None else "") + f"; moved {-sign * m:+.3f} m: "
                           + ("every Phase 2 stand-off has one" if n_fit == len(TT.STAND_OFFS) else f"{n_fit} of {len(TT.STAND_OFFS)} stand-offs have one, the most any move gives"))
        self._flip_room = {**rec, "lateral": round(float(moved.lateral), 4), "move_m": round(float(moved.lateral - init.lateral), 4),
                           "minimum_move_m": round(-sign * excess, 4) + 0.0, "reason": reason, **({"board_fit": fit, "board_fit_at_minimum_move": fit0} if half is not None else {})}
        log.info("%s: init moved %+.2f m along the edge (lateral %.2f -> %.2f) so the manipulation stance can stand %+.2f m from it within %s%s", self.gp.alias,
                 -sign * m, init.lateral, moved.lateral, need, [round(lo, 2), round(hi, 2)], "" if half is None else f"; board on the top: {fit} ({reason})")
        return moved

    def _board_positions(self) -> List[Tuple[float, float]]:
        """``generation.board_base_xy_candidates``: the board centre's (base depth, lateral) at the manipulation stance, in order."""
        return [(float(d), float(l)) for d, l in (self.gen("board_base_xy_candidates", None) or BOARD_BASE_XY)]

    def _board_on_top(self, manip: Stance, depth: float, lateral: float, half: float, right: th.Tensor, far: th.Tensor) -> bool:
        """:meth:`_place_board`'s off-the-top test: the four plate corners of the table-aligned board centred ``depth`` ahead of
        and ``lateral`` left of ``manip``'s base lie on free cells of the top (1 cm margin)."""
        xy = manip.base_xy + manip.forward * float(depth) + manip.left * float(lateral)
        return all(self.top.contains(xy + right * sx * half + far * sy * half, 0.01) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0))

    def _board_fit(self, init: Stance, shift: float, half: float) -> Dict[str, List[str]]:
        """Per Phase 2 stand-off (``TT.STAND_OFFS``, key ``"0.35"`` ...), the ``board_base_xy_candidates`` positions that pass the
        off-the-top test (geometry only) at the manipulation stance ``shift`` m along the edge from ``init``, built as Phase 2
        builds it; a stand-off whose base is off the free floor (Phase 2 skips it), or a shift off the floor, has none."""
        manip = self._shifted_along_edge(init, shift, "manipulation")
        out: Dict[str, List[str]] = {}
        for so in TT.STAND_OFFS:
            st = manip.shifted(stand_off=float(so)) if manip is not None else None
            if st is None or not self.kit.trav_free(st.base_xy):
                out[f"{float(so):.2f}"] = []
                continue
            right, toward, _ = self._table_axes(st)
            out[f"{float(so):.2f}"] = [f"({d:.2f}, {l:+.2f})" for d, l in self._board_positions() if self._board_on_top(st, d, l, half, right, -toward)]
        return out

    def _shifted_along_edge(self, stance: Stance, shift: float, name: Optional[str] = None) -> Optional[Stance]:
        """``stance`` moved ``shift`` m along the edge toward the robot's LEFT (world ``+tangent``; the edge frame's own
        lateral sign is resolved by trying both), same stand-off / trunk; None when the base leaves the traversable floor."""
        target = stance.base_xy + stance.tangent * float(shift)
        cands = [stance.shifted(lateral=stance.lateral + s * float(shift), name=name or stance.name) for s in (1.0, -1.0)]
        st = min(cands, key=lambda c: float(th.linalg.norm(c.base_xy - target)))
        if float(th.linalg.norm(st.base_xy - target)) > 0.02 or not self.kit.trav_free(st.base_xy):
            return None
        st.extras["lateral_shift_m"] = float(shift)
        return st

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
            good = FLIP_TRUNKS if self._flips_expected() else GOOD_TRUNKS
            trunk_pen = min(abs(float(c.translate) - t0) + abs(float(c.tilt) - i0) for t0, i0 in good)
            # owner 2026-09-19: with right-facing flips the manipulation stance must stand left of the init camera by enough
            # that the flip cells (0.15-0.35 m right of it) lie left of the camera's centre line (visible goal faces)
            shift = float(c.extras.get("lateral_shift_m", 0.0))
            shift_pen = 2.0 if (self._flips_expected() and self._flip_directions()[0] in ("right", "left") and abs(shift) < 0.3) else 0.0
            # pod run 0919-090517: every layout failed for want of flip cells (deep + right of the base): regions short of
            # ``_n_flips`` such cells go last
            elig_pen = 1.0 * max(0, self._n_flips() - self._n_flip_cells(c, init)) if self._flips_expected() else 0.0
            return abs(depth - REGION_DEPTH_PREFERRED) + 0.5 * abs(lat - GRID_LATERAL_PREFERRED) + 0.5 * trunk_pen + shift_pen + elig_pen

        order.sort(key=score)
        if self._flips_expected():
            log.info("%s: %d manipulation candidates; flip cells per candidate (first 12 by score): %s", self.gp.alias, len(order),
                     ", ".join(f"{k}:{self._n_flip_cells(self.manip_candidates[k], init)}/{len(self.manip_candidates[k].extras.get('focus_info', {}).get('cells', []))}"
                               f"@sh{float(self.manip_candidates[k].extras.get('lateral_shift_m', 0.0)):+.2f}/t{float(self.manip_candidates[k].translate):.2f}"
                               for k in order[:12]))
            for k in order[:2]:
                c = self.manip_candidates[k]
                so = float(c.stand_off if c.stand_off is not None else 0.35)
                log.info("%s: candidate %d region cells (base depth|lateral; edge depth = base depth - %.2f): %s", self.gp.alias, k, so,
                         " ".join(f"({float(c.world_to_base(xy)[0]):.2f}|{float(c.world_to_base(xy)[1]):+.2f})" for xy in c.extras.get("focus_info", {}).get("cells", [])))
        for k in order[: int(self.gen("manipulation_candidates_tried", 6))]:
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
        self._travel_between(init, manip)
        cam_init = self.camera_pose_at(init)
        right, toward, table_yaw = self._table_axes(manip)
        pitch = self._pitch()
        region = manip.extras["focus_info"]["cells"]
        gap = float(self.g("scatter_min_gap_m"))
        n_flip = self._n_flips() if self._flips_expected() else 0
        primary_dir = self._flip_directions()[0]
        # --- cells: one block per region cell, >= 3 cm gaps between footprints ------------------------------------
        z_c = self.top.z + 0.5 * self.edge

        def img_score(xy) -> float:
            uv = self.world.camera_model.project(cam_init, th.tensor([[float(xy[0]), float(xy[1]), z_c]]))[0][0]
            return float(uv[0] + uv[1])

        cells = [region[int(i)] for i in rng.permutation(len(region))]
        slot_faces = [c2f[p] for p in pattern]  # the goal face of every slot (slot = the block's rank in the init image)
        # board variant: the board on the table first (table-aligned; its pocket's far-left corner anchors both grids), the init
        # cells under / beside it dropped
        board_name, board_pose = None, None
        if self._has_board():
            _flip_mode0 = n_flip > 0 and primary_dir in ("right", "left")
            board_name, board_pose, cells = self._place_board(manip, right, toward, table_yaw, cells,
                                                              (lambda c: self._flip_cell_ok(c, manip, init, primary_dir)) if _flip_mode0 else None, n, self.obj("support"),
                                                              layout_ok=(lambda kept: self._has_flip_set(kept, n, img_score, slot_faces, manip, init, primary_dir, True))
                                                              if _flip_mode0 else None)
            self._teleport({board_name: board_pose})
        region_all = list(cells)
        # 17:24 first-demo run: every pick-screening failure was a block at base-lateral 0.0 / -0.08 with the other blocks
        # to its right (the arm reaches over them), every pick at -0.16 / -0.24 passed; the isolated kinematics do not
        # depend on the lateral (probe_place_lat 18:25), so keep the blocks in front of the right shoulder when the region
        # holds enough such cells, else use the whole region
        lat_of = {id(xy): float(manip.world_to_base(xy)[1]) for xy in cells}
        window = [xy for xy in cells if GRID_LATERAL_WINDOW[0] <= lat_of[id(xy)] <= GRID_LATERAL_WINDOW[1]]
        flip_mode = n_flip > 0 and primary_dir in ("right", "left")
        if len(window) >= n:
            cells = window
        chosen: List[th.Tensor] = []
        vis_lat = float(self.gen("flip_face_visibility_lateral_m", 0.07))

        def flip_eligible(c: th.Tensor) -> bool:
            return self._flip_cell_ok(c, manip, init, primary_dir)

        def flip_plan_for(cs: List[th.Tensor]) -> Optional[set]:
            return self._flip_plan(cs, img_score, slot_faces, manip, init, primary_dir, flip_mode)

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
            # with flips (owner 2026-09-19): an anchor that can itself flip (flip-eligible cell) is placed where it is picked -
            # the one in-hand flip the exact chain passes broadly - so such anchors come first
            flip_pen = 0.3 if (flip_mode and not flip_eligible(xy)) else 0.0
            return (0 if on_band else 1, abs(float(b[1]) - GRID_LATERAL_PREFERRED) + 0.5 * abs(float(b[0]) - REGION_DEPTH_PREFERRED) - bonus + flip_pen)

        # the anchor must be the top-left block of the final set (its position stays, the goal grid hangs from it): try the
        # cells in arm-score order and keep the first whose n-1 companions all lie right of / nearer than it (img_score) at
        # >= edge*sqrt(2) + gap; the whole region as a last resort (blk_pre5 23:37: filling from farther cells made a far
        # block the top-left anchor and put the goal grid at 0.80 m base-forward, beyond every place)
        ranked = sorted(cells, key=arm_score)
        n_sets = 0
        set_diag: List[str] = []
        for anchor_xy in ranked:
            s_anchor = img_score(anchor_xy)
            chosen = [anchor_xy]
            for pool in ([xy for xy in cells if xy is not anchor_xy and img_score(xy) > s_anchor],
                         [xy for xy in region_all if img_score(xy) > s_anchor and all(xy is not c for c in chosen)]):
                if flip_mode:
                    # the flip-eligible cells first (0919-090517: the random fill left every set one flip cell short)
                    pool = sorted(pool, key=lambda xy: 0 if flip_eligible(xy) else 1)
                for xy in pool:
                    if len(chosen) == n:
                        break
                    if all(float(th.linalg.norm(xy - c)) >= self.edge * math.sqrt(2) + gap for c in chosen):
                        chosen.append(xy)
                if len(chosen) == n:
                    break
            if len(chosen) == n:
                n_sets += 1
                flip_set = flip_plan_for(chosen)
                if flip_set is not None:
                    break
                if len(set_diag) < 3:
                    set_diag.append("[" + ", ".join(f"({float(manip.world_to_base(c)[1]):+.2f}|{float(init.world_to_base(c)[1]):+.2f})" for c in chosen) + "]")
                chosen = []
        if len(chosen) < n and flip_mode and (self._has_board() or bool(self.gen("layout_exhaustive_sets", False))):
            # every 4-subset of the region cells (C(8, 4) = 70): the anchor-first greedy fill above tries one set per anchor, which
            # with the board's footprint out of the region left no set whose repeated colours both had a flip cell (0920-004315).
            # The set's top-left block in the image is its anchor either way (image-order slot assignment).
            import itertools

            pool_cells = [xy for xy in region_all]
            best = None
            for combo in itertools.combinations(range(len(pool_cells)), n):
                cs = [pool_cells[i] for i in combo]
                if any(float(th.linalg.norm(cs[a] - cs[b])) < self.edge * math.sqrt(2) + gap for a in range(n) for b in range(a + 1, n)):
                    continue
                cs = sorted(cs, key=img_score)  # anchor (top-left) first, as the greedy search orders its sets
                n_sets += 1
                if flip_plan_for(cs) is None:
                    continue
                # prefer sets with more flip-eligible cells (slack for the flip / re-grasp chains), then nearer the arm's sweet depth
                key = (-sum(1 for c in cs if flip_eligible(c)), abs(float(manip.world_to_base(th.stack(cs).mean(dim=0))[0]) - REGION_DEPTH_PREFERRED))
                if best is None or key < best[0]:
                    best = (key, cs)
            if best is not None:
                chosen = list(best[1])
                flip_set = flip_plan_for(chosen)
                log.info("%s: exhaustive cell-set search: %d flip-eligible of %d cells in the set", self.gp.alias, -best[0][0], n)
        if len(chosen) < n:
            if n_sets:
                raise Reject("layout", f"{n_sets} cell sets keep the gaps but none lets every repeated-colour block flip from a cell <= {FLIP_CELL_MAX_LATERAL:+.2f} m of the "
                                       f"manipulation base's lateral, >= {float(self.gen('flip_cell_min_edge_depth_m', FLIP_CELL_MIN_EDGE_DEPTH)):.2f} m in from the edge and >= {vis_lat:.2f} m left of the init camera "
                                       f"(manipulation stance shift {float(manip.extras.get('lateral_shift_m', 0.0)):+.2f} m; sets as (manip|init lateral): {' '.join(set_diag)})")
            raise Reject("layout", f"only {len(chosen)} of {n} region cells keep >= {gap} m gaps right of / nearer than an anchor")
        b_anchor = manip.world_to_base(th.stack(grid_cells(chosen[0])).mean(dim=0))
        log.info("%s: goal grid anchored at base (%.2f, %+.2f) (score %s)", self.gp.alias, float(b_anchor[0]), float(b_anchor[1]), arm_score(chosen[0]))
        # --- anchor (top-left in the init image) and slot assignment -----------------------------------------------
        score = [img_score(c) for c in chosen]
        order_img = sorted(range(n), key=lambda i: score[i])  # image order (top-left first)
        anchor_i = order_img[0]
        slot_of = {bi: si for si, bi in enumerate(order_img)}  # block index -> slot index (anchor = slot 0)
        goal_face = {bi: c2f[pattern[si]] for bi, si in slot_of.items()}
        # --- up faces (distinct, never the opposite of the goal face) and yaws (visible-face rule) ------------------
        # Flip direction (owner 2026-09-19; module docstring above FLIP_DIRECTIONS): the goal face of a flipping block faces
        # the side ``flip_dir`` of the table axes - "right" (the only in-hand flip the right arm passes) means the face is
        # visible from the init camera only for blocks LEFT of the camera's centre line, and the flip class reaches only
        # cells right of the manipulation base: ``flip_plan_for`` chose which block of every repeated colour shows it up
        # (a cell that cannot flip when there is one) and which blocks flip (all flip-eligible). "near" keeps the spec's
        # image-order assignment (goal face toward the robot).
        flip_dirs = [str(v) for v in (self.gen("flip_directions", None) or FLIP_DIRECTIONS)]
        dir_vec = {"near": -manip.inward, "far": manip.inward, "left": manip.tangent, "right": -manip.tangent}
        for dname in flip_dirs:
            if dname not in dir_vec:
                raise Reject("spec", f"generation.flip_directions: unknown side {dname!r} (near / far / left / right)")
        faces = list(self.faces)
        up_of: Dict[int, str] = {}
        must_flip = flip_set if flip_set is not None else set()
        # the blocks showing their goal face first (they take the goal colours), the flipping ones after
        assign_order = [bi for bi in order_img if bi not in must_flip] + [bi for bi in order_img if bi in must_flip]
        for bi in assign_order:
            options = [f for f in faces if f != TT.opposite_face(goal_face[bi]) and f not in up_of.values()]
            if not options:
                raise Reject("layout", "no distinct up face left that is not the opposite of the goal face")
            # prefer showing the goal face on top when it is still free (visible-face rule, first clause). Prepared, off by
            # default (owner 2026-09-16 14:38): ``generation.block_no_flip: true`` makes that preference certain — with
            # a pattern of distinct colours no block flips, every pick / place uses the same pitched tilt (the in-hand
            # flip's place approach is the open decision item)
            p_goal_up = 1.0 if self.gen("block_no_flip", False) else float(self.gen("goal_face_up_prob", 0.5))
            if bi in must_flip or goal_face[bi] not in options:
                side_options = [f for f in options if f != goal_face[bi]]
                up_of[bi] = side_options[int(rng.integers(len(side_options)))] if side_options else options[0]
            else:
                up_of[bi] = goal_face[bi] if rng.random() < p_goal_up else options[int(rng.integers(len(options)))]
        poses: Dict[int, TT.Pose] = {}
        coarse_yaw: Dict[int, float] = {}
        flip_dir_of: Dict[int, Optional[str]] = {}
        for bi in range(n):
            q_up = _rotation_with_up(up_of[bi])
            if up_of[bi] == goal_face[bi]:
                yaw_c = table_yaw + float(rng.integers(4)) * math.pi / 2  # the goal face is up: any table-aligned coarse yaw
                flip_dir_of[bi] = None
            else:
                # the goal face toward the drawn side (uniform over generation.flip_directions when several are listed)
                dname = flip_dirs[int(rng.integers(len(flip_dirs)))] if len(flip_dirs) > 1 else flip_dirs[0]
                flip_dir_of[bi] = dname
                yaw_c = TT.facing_coarse_yaw(q_up, goal_face[bi], dir_vec[dname])
            coarse_yaw[bi] = yaw_c
            # a flipping block's yaw jitter is capped (``generation.flip_yaw_jitter_max_deg``, default 8): the yaw-align of the
            # in-hand reorientation turns the 60-70 deg side pick about the vertical before the flip, and the wrist roll the
            # flip class needs sits close to joint 7's limit (owner 2026-09-19 probe)
            half = float(self.g("yaw_align_max_deg")) * 0.5
            if flip_dir_of[bi] is not None:
                half = min(half, float(self.gen("flip_yaw_jitter_max_deg", 8.0)))
            jitter = float(rng.uniform(-math.radians(half), math.radians(half)))
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
        anchor_place = anchor  # where the loose (placement) grid hangs; the packed goal grid hangs at ``anchor``
        board_goals, board_aims = None, None
        if self._has_board():
            # puzzle board (owner 2026-09-20 22:39): the goal slots ARE the pocket centres (the seat), the release aims the bias toward
            # each pocket's outer corner; no loose grid, no packing
            board_goals, board_aims, _floor_z = self._board_slot_targets(board_pose, right, toward)
        plans: Dict[int, Dict[str, Any]] = {}
        slots: Dict[str, Dict[str, Any]] = {}
        goal_pose: Dict[int, TT.Pose] = {}  # the GOAL slots (the packed grid with the pack tail): the TRO targets
        place_goal: Dict[int, TT.Pose] = {}  # where every block is PLACED (the loose grid the pack tail closes; = goal_pose without it)
        # packing tail (owner 2026-09-19 21:47): the blocks are placed on a loose grid (``place_gap_m`` between them, room for the
        # descending fingers beside the placed neighbours) and then pushed together against a wide-braced column / row; the goal
        # grid is the packed one (``block_gap_m``, "almost touching"); the anchor block (slot 0) is the fixed corner of both grids
        pack_on = bool(self.gen("block_pack_tail", False))
        pitch_place = self.edge + float(self.gen("place_gap_m", self.g("block_gap_m"))) if pack_on else pitch
        for bi in range(n):
            si = slot_of[bi]
            r, c = divmod(si, n_cols)
            plan = TT.cube_flip_plan(poses[bi][1], goal_face[bi], table_yaw, float(self.g("yaw_align_max_deg")) + 0.01)
            if plan is None:
                raise Reject("layout", f"block {bi}: no smallest-rotation plan to face {goal_face[bi]}")
            if abs(float(plan["yaw_deg"])) < float(self.gen("yaw_align_snap_deg", 0.5)):
                # a yaw-align below the snap (owner 2026-09-19 first demo: yaw_align_max_deg 1) is no in-hand reorientation: the
                # rigid in-hand transform places the block table-aligned and the carry leg's SLERP absorbs the fraction of a degree
                plan["yaw_deg"] = 0.0
            plans[bi] = plan
            q_goal = plan["goal_quat"]
            xy = anchor[:2] + right * (c * pitch) + toward * (r * pitch)
            goal_pose[bi] = self._root_pose(xy, q_goal, 0.001)
            xy_p = anchor_place[:2] + right * (c * pitch_place) + toward * (r * pitch_place)
            place_goal[bi] = goal_pose[bi] if not pack_on else self._root_pose(xy_p, q_goal, 0.001)
            if board_name is not None:  # the slots stand on the pocket floor, a plate above the table top: goal = the seat, place = the aim
                dz_b = th.tensor([0.0, 0.0, float(self._board_geometry()["plate_thickness_m"])])
                goal_pose[bi] = (self._root_pose(board_goals[si], q_goal, 0.001)[0] + dz_b, q_goal)
                place_goal[bi] = (self._root_pose(board_aims[si], q_goal, 0.001)[0] + dz_b, q_goal)
            slots[f"slot_{si}"] = {"position": [float(v) for v in goal_pose[bi][0]], "orientation": [float(v) for v in q_goal]}
        # §4.6 step 5: whole-body checks at init and manipulation with the blocks placed
        TT.whole_body_check_at(self, [init, manip], objects, require=True, reason="stance_collision")
        # owner item 3 (release height, core 2c59893): the actual surface under every slot (a downward raycast with the
        # blocks ignored) — the place eef releases RELEASE_ABOVE_SURFACE above it (the slot target itself stays on the top)
        self.slot_surface_z = {}
        if hasattr(self.world, "surface_z_under"):
            blocks_all = [self.objects[names[bi]] for bi in range(n)]
            for bi in range(n):
                c_xy = self._centre(place_goal[bi])[0][:2]
                hit = self.world.surface_z_under(c_xy, float(self.top.z) + 0.20, ignore=blocks_all)
                if hit is not None and float(self.top.z) - 0.02 <= float(hit["z"]) <= float(self.top.z) + 0.02:
                    self.slot_surface_z[bi] = float(hit["z"])
        # --- cube grasps, flips and brace points per block in placement order ---------------------------------------
        # Per-block stances (owner 2026-09-19): every block is solved from the first stance of its candidate list that
        # passes its whole chain - the stance the robot already stands at first, then the pool of lateral / stand-off /
        # trunk variations along the edge (``_stance_pool``); a change of stance between two blocks is recorded as a path +
        # transition with empty hands and the blocks placed so far at their slots
        order = sorted(range(n), key=lambda bi: slot_of[bi])  # anchor first, then the slots in order
        pool = self._stance_pool(manip, objects)
        self._pool = pool
        self._proven_stances: List[Stance] = []  # stances that solved a block of this instance (plain picks / places proven there)
        records = []
        used: List[Stance] = []  # the stance the robot stands at after every placed block, in placement order
        current = manip
        self._prev_end_q = None  # where the previous block's last tracked leg left the arm (28-D at its stance; the re-pose plan's start)
        for step, bi in enumerate(order):
            flipping = bool(plans[bi]["flips"])
            cands = self._rank_stances(pool, current, poses[bi], place_goal[bi], flipping)
            rec, st = self._block_program(rng, cands, names, poses, place_goal, plans, order, step, right, toward, table_yaw, current=current)
            records.append(rec)
            st2 = rec.pop("_regrasp_stance_obj", None)  # flip-in-place with the re-grasp from another stance
            used.append((st, st2))
            current = st2 if st2 is not None else st
            self._prev_end_q = getattr(self, "_chain_end_q", None)
            for s_ in (st, st2):
                if s_ is not None and all(s_ is not p_ for p_ in self._proven_stances):
                    self._proven_stances.append(s_)
        names_of = self._register_stances(init, manip, used, records, names, poses, place_goal, order, right, toward)
        for bi in range(n):  # a table flip may have re-chosen the block's table-aligned goal yaw (goal face unchanged)
            goal_pose[bi] = (goal_pose[bi][0], place_goal[bi][1])
            slots[f"slot_{slot_of[bi]}"]["orientation"] = [float(v) for v in goal_pose[bi][1]]
        if pack_on:
            pack, st_pack = self._pack_tail(manip, current, pool, names, records, slot_of, n_cols, right, toward, pitch_place - pitch)
            if st_pack.name not in self.sidecar.stances:
                st_pack.name = "manipulation_pack"
                st_pack.arms = ["right", "left"]
                self.sidecar.add_stance(st_pack)
                self._teleport({rec["instance"]: (th.tensor([float(v) for v in rec["goal_pose"]["position"]]), th.tensor([float(v) for v in rec["goal_pose"]["orientation"]])) for rec in records})
                self._travel_between(current, st_pack)
                log.info("%s: packing from a stance of its own %s (trunk %.2f/%+.2f, stand-off %.2f)", self.gp.alias, st_pack.name, st_pack.translate, st_pack.tilt, float(st_pack.stand_off or 0.0))
            else:
                st_pack.arms = ["right", "left"]
                self.sidecar.stances[st_pack.name] = st_pack
                # the pack stance is one of the block stances, but the robot may stand elsewhere after the last block: the program
                # needs a recorded transition / path between the two (0920-073308: "stances 'manipulation_4' and 'manipulation' differ
                # in trunk but no transition is recorded")
                if st_pack.name != current.name and not any({str(t.frm), str(t.to)} == {current.name, st_pack.name} for t in self.sidecar.transitions):
                    self._teleport({rec["instance"]: (th.tensor([float(v) for v in rec["goal_pose"]["position"]]), th.tensor([float(v) for v in rec["goal_pose"]["orientation"]])) for rec in records})
                    self._travel_between(current, st_pack)
            pack["stance"] = st_pack.name
            self.sidecar.pack = pack
            self._teleport({names[bi]: poses[bi] for bi in range(n)})
        # --- draft ---------------------------------------------------------------------------------------------------
        for bi in range(n):
            d.object_poses[names[bi]] = poses[bi]
            d.goal_poses[names[bi]] = goal_pose[bi]
        if board_name is not None:
            d.object_poses[board_name] = board_pose
            self.sidecar.describe(board_name, "board", "the wooden puzzle board with four pockets")
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
        self.sidecar.blocks = records
        for bi in range(n):
            self.sidecar.describe(names[bi], "block", f"the block with the {self.faces.get(up_of[bi], up_of[bi])} face up" + (" (top-left)" if bi == anchor_i else ""))
        d.summary.update({"pattern": pid, "pattern_faces": pattern, "up_faces": {names[bi]: up_of[bi] for bi in range(n)}, "goal_faces": {names[bi]: goal_face[bi] for bi in range(n)},
                          "anchor_block": names[anchor_i], "table_yaw": table_yaw, "manipulation_candidate": k, "goal_face_visibility": {names[bi]: v for bi, v in vis.items()},
                          "placement_order": [names[bi] for bi in order], "brace": braces, "brace_reasons": {rec["instance"]: rec.get("brace_reasons", {}) for rec in records},
                          "region_depth_m": manip.extras.get("focus_info", {}).get("depth"), "flip_directions": {names[bi]: flip_dir_of[bi] for bi in range(n)},
                          "block_stances": {rec["instance"]: rec.get("stance") for rec in records}, "stances_used": names_of,
                          "manipulation_lateral_shift_m": manip.extras.get("lateral_shift_m", 0.0),
                          "flip_classes": {rec["instance"]: rec.get("flip_class") for rec in records if rec.get("flip_class")}})
        reposes = {rec["instance"]: f"{rec['repose'].get('from_stance')} -> {rec.get('stance')} ({rec['repose'].get('planned_from')})" for rec in records if isinstance(rec.get("repose"), dict)}
        if reposes:
            d.summary["block_reposes"] = reposes
        return d

    # -- per-block stances (owner 2026-09-19) -----------------------------------------------------------------------

    def _travel_between(self, frm: Stance, to: Stance) -> None:
        """Record how the robot gets from ``frm`` to ``to`` with empty hands: the §3.8 transition alone when the two stances
        share the edge lateral (the stand-off / trunk change), else the §3.7 path to the drive-back pose plus the transition
        (``prompt0.connect``: the path is checked at the departing trunk / arms, the sweep and drive-in with the arriving)."""
        from ispatialgym.sampling.v2 import prompt0 as P0

        lateral_move = abs(float(th.dot(to.base_xy - frm.base_xy, frm.tangent)))
        if lateral_move < 0.02:
            self.record_transition(frm, to)
            return
        # a lateral move along the table edge with the trunk tilted forward (the flip stances, 0.75/-0.30) swept the chest along
        # the edge and brushed it (puzzle board demo 0921-001836: torso_link2 x breakfast_table during the navigate to the re-grasp
        # stance, the only contact of the run): straighten FIRST - a departure transition at the departing stance (drive back,
        # sweep to the travel trunk), the path driven upright, the arrival transition at the destination (owner 2026-09-19 21:56
        # order, as IORS large-scale drives its carries)
        tilt_limit = float(self.gen("travel_straighten_tilt", -0.05))
        if float(frm.tilt) <= tilt_limit or float(to.tilt) <= tilt_limit:
            # the travel trunk: the departing translate straightened first (0921-043457: at 0.95 / 0 the arriving stance's untucked hands
            # hang over the table edge and the path collides), the tall 0.95 / 0 as the fallback - in either case after driving back at
            # least ``travel_back_min_m`` from the edge (0921-023009: the departure at the stance's own stand-off left the lateral drive
            # 0.35 m from the edge at 0.75 / 0 and torso_link2 brushed it)
            travels = [(float(frm.translate), 0.0), (float(self.gen("travel_trunk_translate", 0.95)), 0.0)]
            back_min = float(frm.stand_off or 0.0) + float(self.gen("travel_back_min_m", 0.10))
            whys: List[str] = []
            for travel in travels:
                try:
                    from ispatialgym.sampling.v2.ik import stance_arm_vectors

                    dep = self.record_departure_transition(frm, to, travel, min_standoff=back_min)
                    r, l, _ = stance_arm_vectors(self.kit, to)  # the departure sweep puts the arms in the ARRIVING stance's posture (NAV.departure_transition)
                    self.record_path(frm, to, trunk=travel, arm_q={"right": r, "left": l}, start_pose2d=NAV.departure_pose(frm, dep.stand_off),
                                     max_length_m=float(self.gen("stance_path_max_m", 3.0)))
                    self.record_transition(frm, to, trunk_from=travel)
                    log.info("%s: travel %s -> %s straightened first (departure s_T %.2f, drive at trunk %.2f/%+.2f)", self.gp.alias, frm.name, to.name, dep.stand_off,
                             travel[0], travel[1])
                    return
                except Reject as e:
                    whys.append(f"trunk {travel[0]:.2f}/{travel[1]:+.2f}: {e}")
                    self._drop_travel_records(frm.name, to.name)
            log.info("%s: travel %s -> %s: straighten-first impossible (%s); driving at the departing trunk", self.gp.alias, frm.name, to.name, "; ".join(whys)[:600])
        P0.connect(self, frm, to, path_max_m=float(self.gen("stance_path_max_m", 3.0)))

    def _drop_travel_records(self, frm: str, to: str) -> None:
        """Remove the paths / transitions recorded between two stances (a failed straighten-first attempt)."""
        self.sidecar.paths = [p_ for p_ in self.sidecar.paths if not (str(p_.frm) == frm and str(p_.to) == to)]
        self.sidecar.transitions = [t for t in self.sidecar.transitions if not (str(t.frm) == frm and str(t.to) == to)]

    def _stance_pool(self, manip: Stance, objects: Dict[str, TT.Pose]) -> List[Stance]:
        """Candidate stances for the per-block chains: the manipulation stance, the Phase 2 survivors that see the same region,
        and lateral shifts of the manipulation stance along the edge (``generation.stance_lateral_shifts_m``) at every
        stand-off / trunk of those survivors plus the probe's flip trunks - traversable and whole-body collision-free with
        the blocks at their init poses. ``manip`` itself is first."""
        key = manip.extras.get("focus_info", {}).get("depth")
        same_region = [c for c in self.manip_candidates if c is not manip and c.extras.get("focus_info", {}).get("depth") == key
                       and float(th.linalg.norm(c.focus[:2] - manip.focus[:2])) < 1e-3]
        trunks = {(round(float(manip.translate), 3), round(float(manip.tilt), 3))}
        trunks |= {(round(float(c.translate), 3), round(float(c.tilt), 3)) for c in same_region}
        trunks |= {(float(t), float(k)) for t, k in (self.gen("flip_trunks", None) or FLIP_TRUNKS)}
        shifts = [float(v) for v in (self.gen("stance_lateral_shifts_m", None) or STANCE_LATERAL_SHIFTS)]
        # 0.40 / 0.50 (0919-081444): the re-grasp of the turned block passed at base depth 0.59-0.64 only; a cell 13 cm in from
        # the edge sits there from a 0.50 stand-off
        stand_offs = sorted({float(manip.stand_off or TT.STAND_OFF)} | {float(v) for v in (self.gen("stance_stand_offs_m", None) or (0.30, 0.35, 0.40, 0.45, 0.50))})
        raw: List[Stance] = []
        seen = set()

        def add(st: Stance, shift: float):
            k_ = (round(float(st.base_pose2d[0]), 3), round(float(st.base_pose2d[1]), 3), round(float(st.translate), 3), round(float(st.tilt), 3))
            if k_ in seen:
                return
            seen.add(k_)
            st.focus, st.arms = manip.focus, ["right"]
            st.extras["focus_info"] = manip.extras.get("focus_info")
            st.extras["block_stance_shift_m"] = float(shift)
            raw.append(st)

        add(manip, 0.0)
        for c in same_region:
            add(c.with_trunk(c.translate, c.tilt, "manipulation"), 0.0)
        fk = TT.camera_fk(self)
        for shift in shifts:
            for so in stand_offs:
                base = manip if (abs(shift) < 1e-9 and abs(so - float(manip.stand_off or so)) < 1e-9) else None
                if base is None:
                    b0 = manip.shifted(stand_off=so, name="manipulation")
                    base = b0 if abs(shift) < 1e-9 else self._shifted_along_edge(b0, shift, "manipulation")
                    if base is None:
                        continue
                for t, k in sorted(trunks):
                    if k < float(fk.tilt_limit(t)) - 1e-6:
                        continue  # beyond the torso tilt limit at this translate
                    add(base.with_trunk(t, k, "manipulation"), shift)
        # whole-body check with the blocks at their init poses (batched); ``manip`` was checked by Phase 2 already. A stance whose
        # untucked RIGHT arm hangs into a block (the deep trunks put the untucked hands ~ at the block tops over the near cells)
        # is retried with the working arm in the ready / tucked posture: recorded as the stance's configuration (``q`` ->
        # ``arms_q``), the posture the oracle drives in with and the IK checks start from
        self._teleport(objects)
        free = self.ikc.stances_collision_free(raw)
        # replay probe 2026-09-19 (pod): the flip chain that passes with the idle left arm untucked or 'ready' fails its pick with
        # the left arm TUCKED (the A6 fallback posture) - such stances leave the pool
        pool = [st for st, ok in zip(raw, free.tolist()) if (ok and st.idle_posture != "tucked") or st is manip]
        n_reposed = 0
        retry = [st for st, ok in zip(raw, free.tolist()) if not ok and st is not manip]
        for name in ("ready", "tucked"):
            if not retry:
                break
            vec = self.kit.arm_q(name, "right")
            qs = th.stack([self.ikc.q_locked(st, right_arm=vec) for st in retry])
            ok = self.kit.collision_free(qs, margin=0.01)
            still = []
            for st, q, f in zip(retry, qs, ok.tolist()):
                if f:
                    st.q = q.clone()
                    st.extras["arm_posture"] = name
                    pool.append(st)
                    n_reposed += 1
                else:
                    still.append(st)
            retry = still
        by_key: Dict[str, int] = {}
        for st in pool:
            k_ = f"so{float(st.stand_off or 0):.2f}/sh{float(st.extras.get('block_stance_shift_m', 0.0)):+.2f}"
            by_key[k_] = by_key.get(k_, 0) + 1
        log.info("%s: stance pool: %d candidates, %d collision-free (%d with the right arm re-posed) over shifts %s, stand-offs %s, %d trunks; per (stand-off, shift): %s",
                 self.gp.alias, len(raw), len(pool), n_reposed, shifts, stand_offs, len(trunks), by_key)
        if retry:
            # diagnostic: what the failing stances collide with - the blocks (parked away -> free) or the scene / self
            for o in objects:
                self.world.park(self.objects[o], list(self.objects).index(o))
            self.kit.sync_world()
            free_nb = self.ikc.stances_collision_free(retry, try_idle_postures=False)
            n_blocks_only = int(free_nb.sum())
            self._teleport(objects)
            sample = [f"so{float(st.stand_off or 0):.2f}/sh{float(st.extras.get('block_stance_shift_m', 0.0)):+.2f}/t{st.translate:.2f}/{st.tilt:+.2f}" for st in retry[:6]]
            log.info("%s: stance pool: %d stances rejected; %d of them collide only through the blocks (free with the blocks parked), the rest with the scene or "
                     "themselves; e.g. %s", self.gp.alias, len(retry), n_blocks_only, sample)
        return pool

    def _rank_stances(self, pool: List[Stance], current: Stance, pick_pose: TT.Pose, place_pose: TT.Pose, flipping: bool) -> List[Stance]:
        """The pool ordered for one block: the current stance first (no move), then by how well the block's cell and its slot
        sit in the arm's sweet spot of the stance's base frame (probe: flip chains at depth 0.55-0.70, lateral -0.15..-0.35;
        pitched picks around (0.55, -0.20)) and the size of the move; a flipping block's cell must lie right of
        ``FLIP_CELL_MAX_LATERAL``. At most ``generation.max_stances_per_block`` (default 6)."""
        c_pick = self._centre(pick_pose)[0][:2]
        c_place = self._centre(place_pose)[0][:2]
        sweet = FLIP_CELL_SWEET if flipping else PICK_CELL_SWEET

        # trunk preference from the probe (flip chains per trunk at the sweet cells): (0.5, -0.3) best, then (0.75, 0 / -0.3),
        # (0.5, -0.45), (0.95, 0), (0.25, -0.6), (0.5, 0); others last
        trunk_rank = {(0.75, -0.3): 0.0, (0.65, -0.3): 0.02, (0.85, -0.3): 0.02, (0.75, -0.4): 0.02, (0.75, -0.2): 0.02, (0.5, -0.3): 0.03, (0.75, 0.0): 0.06,
                      (0.5, -0.45): 0.08, (0.95, 0.0): 0.1, (0.25, -0.6): 0.12, (0.5, 0.0): 0.15}

        def trunk_pen(st: Stance) -> float:
            if not flipping:
                return 0.0
            return min(v + abs(float(st.translate) - t) + abs(float(st.tilt) - k) for (t, k), v in trunk_rank.items())

        def score(st: Stance) -> Optional[float]:
            bp, bq = st.world_to_base(c_pick), st.world_to_base(c_place)
            if flipping and (float(bp[1]) > FLIP_CELL_MAX_LATERAL or float(bp[0]) - float(st.stand_off if st.stand_off is not None else 0.35) < float(self.gen("flip_cell_min_edge_depth_m", FLIP_CELL_MIN_EDGE_DEPTH))):
                return None
            # the slot enters as a soft term only: with the flip-in-place fallback the final place is a plain pitched place
            # that reaches slots near the centre line, so a slot left of the flip zone must not exclude the stance
            slot_w = 0.7 if not flipping else 0.3
            s = abs(float(bp[0]) - sweet[0]) + abs(float(bp[1]) - sweet[1]) + slot_w * (abs(float(bq[0]) - sweet[0]) + abs(float(bq[1]) - sweet[1]))
            move = float(th.linalg.norm(st.base_xy - current.base_xy)) + 0.2 * (abs(float(st.translate - current.translate)) + abs(float(st.tilt - current.tilt)))
            return s + 0.3 * move + trunk_pen(st)

        scored = [(score(st), st) for st in pool]
        ranked = sorted([(v, st) for v, st in scored if v is not None], key=lambda t: t[0])
        out = [st for _, st in ranked if st is not current]
        out = [current] + out if (score(current) is not None or not out) else out
        return out[: int(self.gen("max_stances_per_block", 9))]

    def _register_stances(self, init: Stance, manip: Stance, used: List[Stance], records: List[Dict[str, Any]], names: List[str], poses: Dict[int, TT.Pose],
                          goal_pose: Dict[int, TT.Pose], order: List[int], right: th.Tensor, toward: th.Tensor) -> List[str]:
        """Name the stances the blocks use (the first ``manipulation``, then ``manipulation_2``, ...), write them to the
        sidecar with the arms they work with, record the travel between consecutive different stances (empty hands, the blocks
        placed so far at their slots) and stamp every block record with its stance name. The init -> manipulation travel
        was recorded first; when the first block uses another stance, that one is renamed ``manipulation`` and the travel
        re-recorded from init. Returns the stance names in order of first use."""
        visits: List[Stance] = []  # every stance in the order the robot visits it (leg-1 stance, then the re-grasp stance when it differs)
        for st1, st2 in used:
            visits.append(st1)
            if st2 is not None and st2 is not st1:
                visits.append(st2)
        distinct: List[Stance] = []
        for st in visits:
            if all(st is not d for d in distinct):
                distinct.append(st)
        if not distinct:
            return []
        first = distinct[0]
        if first is not manip:
            # the first block works from another stance: drop the init -> manip records and connect init to it instead
            self.sidecar.transitions = [t for t in self.sidecar.transitions if not (t.frm == init.name and t.to == manip.name)]
            self.sidecar.paths = [p for p in self.sidecar.paths if not (p.frm == init.name and p.to == manip.name)]
            first.name = "manipulation"
            self._travel_between(init, first)
        names_of: List[str] = []
        for i, st in enumerate(distinct):
            st.name = "manipulation" if i == 0 else f"manipulation_{i + 1}"
            names_of.append(st.name)
        for rec, (st, st2) in zip(records, used):
            rec["stance"] = st.name
            if st2 is not None and isinstance(rec.get("table_flip"), dict):
                rec["table_flip"]["regrasp"]["stance"] = st2.name
            if rec.get("brace") == "present":
                (st2 or st).arms = ["right", "left"]  # the left arm joins at the stances where a block carries a brace
        for st in distinct:
            if not (isinstance(st.arms, list) and "left" in st.arms):
                st.arms = ["right"]
            self.sidecar.add_stance(st)
        # travel between consecutive visits with the world as the program leaves it: after block ``step``'s leg 1 the block sits
        # flipped on its own cell (``table_flip.block_pose_after``), after its final place at its slot
        def world_at(step: int, after_leg1: bool) -> Dict[str, TT.Pose]:
            w = {names[o]: ((goal_pose[o][0] + _xy3(right) * self._place_margin() + _xy3(toward) * self._place_margin(), goal_pose[o][1]) if j < step else poses[o]) for j, o in enumerate(order)}
            if after_leg1:
                tf = records[step].get("table_flip") or {}
                bp = tf.get("block_pose_after")
                if bp:
                    w[names[order[step]]] = (th.tensor([float(v) for v in bp["position"]]), th.tensor([float(v) for v in bp["orientation"]]))
            return w

        prev: Optional[Stance] = None
        prev_world: Optional[Dict[str, TT.Pose]] = None
        for step, (st1, st2) in enumerate(used):
            legs = [(st1, world_at(step, False))] + ([(st2, world_at(step, True))] if (st2 is not None and st2 is not st1) else [])
            rp = records[step].get("repose")
            if isinstance(rp, dict):
                if prev is None or st1 is prev:
                    log.warning("%s: block %d recorded a re-pose but no base move precedes it: dropped", self.gp.alias, step)
                    records[step].pop("repose")
                else:
                    rp["from_stance"] = prev.name
            for st, world_before in legs:
                if prev is not None and st is not prev and not self.sidecar_has_travel(prev.name, st.name):
                    self._teleport(world_before)
                    self._travel_between(prev, st)
                    log.info("%s: block %d moves the base %s -> %s (%.2f m along the edge)", self.gp.alias, step, prev.name, st.name,
                             float(th.dot(st.base_xy - prev.base_xy, prev.tangent)))
                prev = st
        self._teleport({names[bi]: poses[bi] for bi in range(len(names))})
        return names_of

    def sidecar_has_travel(self, frm: str, to: str) -> bool:
        return any(t.frm == frm and t.to == to for t in self.sidecar.transitions) or any(p.frm == frm and p.to == to for p in self.sidecar.paths)

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

    def _block_program(self, rng, cands: List[Stance], names: List[str], poses: Dict[int, TT.Pose], goal_pose: Dict[int, TT.Pose], plans: Dict[int, Dict[str, Any]],
                       order: List[int], step: int, right: th.Tensor, toward: th.Tensor, table_yaw: float, current: Optional[Stance] = None) -> Tuple[Dict[str, Any], Stance]:
        """§6.5 for one block over its candidate stances (owner 2026-09-19, ``_rank_stances``): the first stance whose whole
        chain passes (``_block_chain``) wins; ``Reject`` with every stance's reason otherwise. Returns ``(record, stance)``.

        ``generation.repose_after_base_move`` (owner 2026-10-12 item 2): a stance other than ``current`` (where the previous block left
        the robot) is reached by a between-block base move; there the arm re-poses before the block's chain (``_repose_record``) and
        the block's legs are tracked with the program check's acceptance (``_tracked_from``)."""
        bi = order[step]
        reasons: List[str] = []
        for j, st in enumerate(cands):
            reposed = bool(self.gen("repose_after_base_move", False)) and step > 0 and current is not None and st is not current
            self._oracle_tracking = reposed
            try:
                rec = self._block_chain(rng, st, names, poses, goal_pose, plans, order, step, right, toward, table_yaw)
                if reposed:
                    rec["repose"] = self._repose_record(st, rec, names, poses, goal_pose, order, step)
            except Reject as e:
                reasons.append(f"stance {j} ({st.extras.get('block_stance_shift_m', 0.0):+.2f} m, so {st.stand_off}, trunk {st.translate:.2f}/{st.tilt:+.2f}): [{e.reason}] {str(e)[:160]}")
                continue
            finally:
                self._oracle_tracking = False
            b_cell = st.world_to_base(self._centre(poses[bi])[0][:2])
            b_slot = st.world_to_base(self._centre(goal_pose[bi])[0][:2])
            log.info("%s: block %s (%s) solved from stance %d of %d (shift %+.2f m, stand-off %.2f, trunk %.2f/%+.2f; cell base (%.2f, %+.2f), slot base (%.2f, %+.2f)) with %s",
                     self.gp.alias, names[bi], "flip" if plans[bi]["flips"] else "pick", j, len(cands), float(st.extras.get("block_stance_shift_m", 0.0)), float(st.stand_off or 0.0),
                     st.translate, st.tilt, float(b_cell[0]), float(b_cell[1]), float(b_slot[0]), float(b_slot[1]), rec.get("grasp_class"))
            return rec, st
        kind = "flip" if plans[bi]["flips"] else "pick"
        raise Reject("place_unreachable" if any("place_unreachable" in r for r in reasons) else "grasp_unreachable",
                     f"block {names[bi]} ({kind}): no stance of {len(cands)} passes its chain: " + " | ".join(reasons[:4]))

    def _repose_record(self, st: Stance, rec: Dict[str, Any], names: List[str], poses: Dict[int, TT.Pose], goal_pose: Dict[int, TT.Pose], order: List[int],
                       step: int) -> Dict[str, Any]:
        """The re-pose before a block's chain at a stance reached by a between-block base move (``generation.repose_after_base_move``,
        owner 2026-10-12 item 2): a planned joint-space move of the working arm to the pre-grasp configuration of the block's first
        grasp - the configuration its approach was tracked from, in the IK branch whose legs reach the place (``_chain_pre_q``).
        Checked in the world before the pick (the blocks placed so far at their slots, the others at init, the board and the table
        present): the configuration reaches the recorded ``pre_pick``, is collision-free, and cuRobo plans a joint-space path to it
        from where the previous block's last tracked leg left the arm (put on this stance; the stance posture when that start
        collides here). ``Reject("repose_unplannable")`` otherwise. The program check plans the move again from where its travel
        leaves the arm; the oracle executes it (``Grasp.repose``)."""
        bi = order[step]
        q7 = getattr(self, "_chain_pre_q", None)
        pre = pose_from_sidecar(rec.get("pre_pick")) if isinstance(rec.get("pre_pick"), dict) else None
        if q7 is None or pre is None:
            raise Reject("repose_unplannable", f"block {names[bi]}: the {rec.get('flip_mode') or 'direct'} chain left no pre-grasp configuration to re-pose to")
        self._teleport({names[o]: (goal_pose[o] if j < step else poses[o]) for j, o in enumerate(order)})
        q_goal = self.ikc.q_locked(st, right_arm=th.tensor(q7, dtype=th.float32))
        if not self.ikc.reaches(q_goal, pre, "right"):
            raise Reject("repose_unplannable", f"block {names[bi]}: the pre-grasp configuration misses the recorded pre-grasp pose")
        if not bool(self.kit.collision_free(q_goal, margin=0.0)[0]):
            raise Reject("repose_unplannable", f"block {names[bi]}: the pre-grasp configuration collides with the world at the stance")
        start, label = self.ikc.q_locked(st), "stance posture"
        prev = getattr(self, "_prev_end_q", None)
        if prev is not None:
            q_prev = self.ikc.q_locked(st, right_arm=TT._t(prev)[self.kit.robot.arm_control_idx["right"]])
            if bool(self.kit.collision_free(q_prev, margin=REPOSE_START_MARGIN)[0]):
                start, label = q_prev, "previous block's end"
        res = None
        for _ in range(REPOSE_PLAN_ROUNDS):
            res = self.kit.backend.plan_joint("right_arm", q_goal, q_start=start, max_attempts=4, sync_world=False)
            if res.success:
                break
        if res is None or not res.success:
            raise Reject("repose_unplannable", f"block {names[bi]}: no joint-space plan from the {label} to the pre-grasp configuration "
                                               f"({getattr(res, 'status', '?')})")
        log.info("%s: block %s re-poses after the base move: joint-space plan from the %s to its pre-grasp configuration (%d waypoints)",
                 self.gp.alias, names[bi], label, len(res.q_traj))
        return {"arm": "right", "target": "pre_grasp", "q": [round(float(v), 5) for v in q7], "pose": rec["pre_pick"], "planned_from": label}

    def _flip_classes(self, block, pick_pose: TT.Pose, place_pose: TT.Pose, plan: Dict[str, Any], table_yaw: float) -> List[Tuple[Any, ...]]:
        """Cube grasp classes of a flipping block (owner 2026-09-19): the pinch across the face pair parallel to the flip axis,
        the pick approach leaning ``FLIP_LEANS_DEG`` from the vertical (both lean directions x both wrist rolls: 4 table yaws x
        +-lean), the tip clearance from the lean (``flip_tip_clearance``), kept when the pick approach points down, the place
        approach (the same rigid in-hand transform at the goal pose) points down, the attached cube does not self-collide and
        the hand spheres clear the table at the pick AND the place (``hand_table_clearance``). Order: lean preference, then
        the wrist roll with the camera up at the pick (the roll the arm accepts at the place, probe 2026-09-19)."""
        pair = plan["face_pair"]
        c_pick, c_place = self._centre(pick_pose), self._centre(place_pose)
        top_z = float(self.top.z)
        out: List[Tuple[Any, ...]] = []
        seen = set()
        leans = [float(v) for v in (self.gen("flip_leans_deg", None) or FLIP_LEANS_DEG)]
        for rank, lam in enumerate(leans):
            clearance = flip_tip_clearance(lam)
            for k in range(4):
                for tilt in (lam, -lam):
                    try:
                        g = C.cube_grasp(c_pick, pair, tilt, table_yaw + k * math.pi / 2, edge=self.edge, tip_clearance=clearance)
                    except ValueError:
                        continue
                    if not g.downward:
                        continue
                    key = tuple(round(float(v), 3) for v in g.eef[1].tolist())
                    if key in seen:
                        continue
                    seen.add(key)
                    place_eef = C.compose(c_place, g.eef_in_block)
                    if float(C.approach_axis(place_eef[1])[2]) > -0.1:
                        continue
                    clr_pick, clr_place = hand_table_clearance(g.eef, top_z), hand_table_clearance(place_eef, top_z)
                    if min(clr_pick, clr_place) <= 0.0:
                        continue
                    if not self._class_attachable(block, pair, tilt, clearance, g.eef_in_block):
                        continue
                    cam_up = float(C.quat_to_mat(g.eef[1])[2, 0]) < -0.2  # eef x pointing down: the wrist camera (-x) up
                    if not cam_up and bool(self.gen("flip_camera_up_only", True)):
                        continue  # probe 2026-09-19: the camera-down roll never passed a whole chain (joint 7 at the place)
                    label = f"{pair}/l{lam:.0f}/{'camup' if cam_up else 'camdown'}/y{k}"
                    out.append((rank, 0 if cam_up else 1, pair, tilt, g, place_eef, clearance, label, lam))
        out.sort(key=lambda t: (t[0], t[1]))
        return out

    def _pick_turns(self, screened: List[Tuple[Any, ...]], manip: Stance, support, block, d_pick: float):
        """``screened`` (cube-pick classes, the module's order) with a class and its twin - the same pick turned pi about the
        approach (a straight-down pick at yaw k and k + 2) - next to each other where the first of them stands, the pair in the
        order of the orientation rule (``orient.choose`` on the pick [pre-grasp, grasp], owner 2026-09-27: joint-limit safe,
        then easier to reach from the stance, the module's order within the tolerance); a class whose wrist camera comes within
        5 mm of the world at the pick is dropped. Lazy: a pair is solved when the chain search reaches it (the search's own
        pick IK, cached); the search takes the first class whose whole chain passes, the other turn next."""
        rest = list(screened)
        q0 = self.ikc.q_locked(manip)

        def evaluate(p):
            res = self._exact2(list(p.chain), manip, ignore_for_last=[support, block], approached=[block])
            return bool(res.ok), res, None, str(res.reason)

        while rest:
            c = rest.pop(0)
            j = next((k for k, o in enumerate(rest) if O.is_twin(c[2].eef, o[2].eef)), None)
            if j is None:
                yield c
                continue
            grp = [c, rest.pop(j)]
            props = [O.Proposal(g[0], r, g, [IK.pre_pose(g[2].eef, d_pick), g[2].eef]) for r, g in enumerate(grp)]
            chosen, dec = O.choose(self.ikc, "right", props, evaluate, q0, stage="cube pick", map_twins=False)
            camera = {o["label"] for o in dec["options"] if not o.get("ok") and str(o.get("why", "")).startswith("wrist camera")}
            first = [] if chosen is None else [chosen.proposal.payload]
            for g in first + [g for g in grp if not first or g is not first[0]]:
                if g[0] not in camera:
                    yield g

    def _exact2(self, poses: List[TT.Pose], stance: Stance, **kw):
        """``IKChecker.exact`` with one retry from FRESH seeds when the seeded solve ends in an untrackable branch or a miss:
        the checker seeds every solve from the nearest remembered witness (hundreds accumulate over the blocks of an
        attempt), and the replay probe 2026-09-19 passed the very chain the sampler reported ``approach_untrackable`` for
        once solved without those seeds. The remembered solutions are restored afterwards (the ones the retry found kept)."""
        res = self.ikc.exact(poses, stance, "right", **kw)
        if res.ok or not any(t in str(res.reason) for t in ("approach_untrackable", "exact IK failed", "ik_reach")):
            return res
        saved = dict(self.ikc.solutions)
        self.ikc.solutions.clear()
        try:
            res2 = self.ikc.exact(poses, stance, "right", **{**kw, "cache": False})
        finally:
            fresh = dict(self.ikc.solutions)
            self.ikc.solutions.clear()
            self.ikc.solutions.update(saved)
            if res2.ok:
                self.ikc.solutions.update(fresh)
        return res2 if res2.ok else res

    def _waypoints_ok2(self, wps: List[TT.Pose], stance: Stance, **kw):
        """``IKChecker.waypoints_ok`` with the same fresh-seed retry as :meth:`_exact2`; every chain that is not a straight
        approach (the yaw-align orbit, the in-hand flip, the carry legs) is also tracked at the oracle's densification
        (``track_dense``: demo 0919-140558 failed the anchor's 11 deg yaw-align with 'joint jump 0.23 rad at waypoint 4' where
        the sampler had only screened the two orbit poses)."""
        kw.setdefault("track_dense", True)
        res = self.ikc.waypoints_ok(wps, stance, "right", **kw)
        if res.ok:
            return res
        saved = dict(self.ikc.solutions)
        self.ikc.solutions.clear()
        try:
            res2 = self.ikc.waypoints_ok(wps, stance, "right", **{**kw, "cache": False})
        finally:
            fresh = dict(self.ikc.solutions)
            self.ikc.solutions.clear()
            self.ikc.solutions.update(saved)
            if res2.ok:
                self.ikc.solutions.update(fresh)
        return res2 if res2.ok else res

    def _arm_clear(self, q) -> Optional[str]:
        """None when the arm links (``ARM_CLEAR_LINKS``) stay ``generation.arm_link_margin_m`` clear of the world at ``q``, else the reason."""
        if q is None:
            return None
        margin = float(self.gen("arm_link_margin_m", 0.012))
        if margin <= 0.0:
            return None
        ok = self.kit.links_clear(TT._t(q), ARM_CLEAR_LINKS, margin)
        return None if bool(ok[0]) else f"an arm link (1-5) within {margin * 1000:.0f} mm of the world"

    def _tracked_from(self, q_from, poses: List[TT.Pose], ignore: List[Any], ignore_end: List[Any] = (), hard: bool = False) -> Tuple[Optional[th.Tensor], str]:
        """The oracle's arm motion from the configuration ``q_from`` (which reaches ``poses[0]``) through ``poses`` as ``arm_along``
        tracks it (``IKChecker.track_chain``: the oracle's densification, no joint step above 0.2 rad, collision-free with
        ``ignore`` removed). Pod runs 0919-140558 / 151544: a chain checked by fresh exact IKs passed while the oracle, tracking
        from the branch the GRASP left the arm in, jumped 0.23 rad / stalled 14 mm - so every leg is tracked from the pick's
        configuration onward. Returns ``(q_end, "ok")`` or ``(None, reason)``."""
        if q_from is None:
            return None, "no start configuration"
        # ``hard``: a reorient - the oracle's arm_along has no finish there, a stall fails the action; the free legs (lift, carry,
        # descent, retreat) are finished by arm_to from a stall below ExecuteConfig.free_leg_finish_max_stall (3 cm; the descent
        # 1.5 cm), so a stall there passes here as it does in the oracle
        soft = None if hard else float(self.gen("tracked_soft_stall_m", 0.015))
        kw: Dict[str, Any] = {}
        if getattr(self, "_oracle_tracking", False):
            # a re-posed block (``_block_program``): the program check's continuation acceptance (``IKChecker.continue_chain``) - the
            # oracle's soft waypoint tolerances and iteration budget, a stall finished only within FINISH_MAX_EXCURSION of where it
            # stalled. Without the bound a stall was finished in another IK branch (c1_k4 2026-10-12: the re-grasp's carry at
            # manipulation_4, whose finish the program check found 3.19 rad away)
            kw = {"finish_max_excursion": self.ikc.FINISH_MAX_EXCURSION, "soft_pos_tol": self.ikc.ORACLE_SOFT_POS_TOL,
                  "soft_rot_tol": self.ikc.ORACLE_SOFT_ROT_TOL, "max_iters": self.ikc.ORACLE_DIFF_IK_MAX_ITERS}
        return self.ikc.track_chain(q_from, poses, "right", ignore=list(ignore), ignore_end=list(ignore_end), soft_stall_m=soft,
                                    clear_links=ARM_CLEAR_LINKS, clear_margin=float(self.gen("arm_link_margin_m", 0.012)), **kw)

    def _approach_clear(self, pre: TT.Pose, end: TT.Pose, ignore: List[Any]) -> Optional[str]:
        """The straight approach ``pre -> end`` tracked from the pre pose's remembered solution with the arm links kept clear of the
        world (``_tracked_from``); None when fine, else the reason. The approach is the oracle's ``arm_along`` from the planned
        pre pose (0919-171511: the forearm brushed the table edge on the way down to a 45 deg pitched pinch)."""
        q_pre = self.ikc.solutions.get(("right", self.ikc._pose_key(pre)))
        if q_pre is None:
            return None
        q_end, why = self._tracked_from(q_pre, [pre, end], ignore)
        return None if q_end is not None else why

    def _retreat_ok(self, place: TT.Pose, stance: Stance, ignore: List[Any]):
        """The ``Place`` action's post-release retreat pose (``RETREAT_DIST`` = 10 cm back along -a from the place) as the program
        check verifies it - with the block still attached and the supports ignored - so a 5 cm pre-place (the FOV-fitting one)
        never leaves that pose unchecked (0919-075115 attempt 1: 'retreat: the end configuration collides' at the anchor's place)."""
        back = IK.pre_pose(place, RETREAT_DIST_M)  # 10 cm (Place) + 5 cm (the program's Retreat "away a little")
        res = self._exact2([back], stance, ignore_for_last=list(ignore), cache=False)
        if not res.ok:
            return res
        # the segment itself, tracked from the place as the program check does it (0919-102716 attempt 1: 'retreat:
        # approach_untrackable ... self-collision (right_arm_link6 x zed_link (1 mm))' 8 mm into the retreat of a place whose
        # end pose passed the exact IK - the place configuration sat 1 mm from the head camera)
        return self._waypoints_ok2([place] + list(C.line_waypoints(place, back, 3)), stance, ignore_all=list(ignore))

    def _chain_in_frame(self, stance: Stance, cam: TT.Pose, poses_by_name: Dict[str, TT.Pose]) -> Optional[str]:
        """The oracle's FOV monitor (fingertips in the head camera image, 5 %) applied to the chain's key poses from the
        stance's camera with ``CHAIN_FOV_MARGIN``; the first pose that fails, None when all pass."""
        margin = float(self.gen("chain_fov_margin", CHAIN_FOV_MARGIN))
        for what, pose in poses_by_name.items():
            if pose is None:
                continue
            p, q = TT._t(pose[0]), TT._t(pose[1])
            pts = th.stack([p, p + C.FINGERTIP_LENGTH * C.approach_axis(q)])
            if not TT.in_frame(self, cam, pts, margin):
                return what
        return None

    def _block_chain(self, rng, manip: Stance, names: List[str], poses: Dict[int, TT.Pose], goal_pose: Dict[int, TT.Pose], plans: Dict[int, Dict[str, Any]],
                     order: List[int], step: int, right: th.Tensor, toward: th.Tensor, table_yaw: float) -> Dict[str, Any]:
        """§6.5 for one block from one stance: the cube grasp (pick + place IK in placement order with the already-placed
        blocks at their slots), the flip waypoints, the fingertips in the stance's camera frame along the chain, the row /
        column brace + push contacts."""
        bi = order[step]
        block = self.objects[names[bi]]
        support = self.obj("support")
        plan = plans[bi]
        self._chosen_pick_q = None  # the pick configuration of the chain chosen below (recorded for the program check's grasp seed)
        # the right arm's joints at the chosen chain's first pre-grasp (its approach was tracked from them) and the configuration its
        # last tracked leg ends in: the re-pose target of a block after a base move and the next re-pose plan's start (``_repose_record``)
        self._chain_pre_q = self._chain_end_q = None
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
        place_gap = float(self.gen("place_gap_m", self.g("block_gap_m"))) if bool(self.gen("block_pack_tail", False)) else float(self.g("block_gap_m"))
        if placed_neighbours and not self._has_board() and place_gap + self._place_margin() < gap_needed:
            raise Reject("finger_clearance", f"block {names[bi]}: slot gap {place_gap * 1000:.0f} mm + place margin {self._place_margin() * 1000:.0f} mm < "
                                             f"{gap_needed * 1000:.1f} mm the closed finger needs beside a placed neighbour (set generation.block_gap_m >= "
                                             f"{(gap_needed - self._place_margin()) * 1000:.0f} mm or record the push tail)")
        # puzzle board: the fingers never enter a pocket - the block is released over the rim and drops onto its seat - so a finger
        # beside a seated neighbour (6 mm away) passes ABOVE it: the pinch classes are filtered by ``_board_fingertips_clear`` below
        place_ignore = [support] + placed_neighbours
        if self._has_board() and bool(self.gen("board_ignore_at_place", True)):
            # puzzle board: at the release the cube's bottom is release_above_surface_m (23 mm) over the pocket floor, 5 mm above the rim
            # and 11 mm above the divider apex, the fingertips higher still - only the attached cube's coarse sphere fit (bulging ~1 cm
            # beyond its faces) could touch the board there, so it is removed for the release pose like a placed neighbour
            place_ignore = place_ignore + [self.objects[self.bindings.one("board").name]]
        placed_xy = [self._centre(goal_pose[o])[0][:2] for j, o in enumerate(order) if j < step]  # the placed blocks' slot centres
        n_sweep = 0
        n_finger_low = 0  # puzzle board: pinch classes whose fingertips would pass below a seated neighbour's top
        # release pose: the slot pose lifted so the block bottom sits RELEASE_ABOVE_SURFACE above the raycast surface (the
        # goal target keeps the on-top pose; without a surface hit the 1 mm drop of the slot pose stays)
        surface_z = getattr(self, "slot_surface_z", {}).get(bi)
        if surface_z is not None:
            bottom = float(place_pose[0][2]) - self.world.bottom_offset(self.hull, place_pose[1])
            dz = max(0.0, surface_z + self._release_above() - bottom)
            place_pose = (place_pose[0] + th.tensor([0.0, 0.0, dz]), place_pose[1])
        cam = self.camera_pose_at(manip)
        top_z = float(self.top.z)
        # candidate classes: (label, face_pair, g_pick, place_eef, tip_clearance, lean or None)
        screened: List[Tuple[str, str, Any, TT.Pose, float, Optional[float]]] = []
        if plan["flips"]:
            for _, _, pair, tilt, g, place_eef, clearance, label, lam in self._flip_classes(block, pick_pose, place_pose, plan, table_yaw):
                if self._sweep_clear(place_eef, place_pose, placed_xy) is not None:
                    n_sweep += 1
                    continue
                if self._has_board() and self._board_fingertips_clear(clearance, placed_neighbours) is not None:
                    n_finger_low += 1
                    continue
                screened.append((label, pair, g, place_eef, clearance, lam))
            if not screened and not bool(self.gen("flip_in_place_fallback", True)):
                raise Reject("grasp_unreachable", f"block {names[bi]}: no flip class with a downward pick and place, the hand clear of the table and an attachable cube "
                                                  f"({n_sweep} swept a placed neighbour)")
            screened = screened[: int(self.gen("max_flip_classes", 6))]
        else:
            up = TT.up_face(pick_pose[1])
            pairs = [ax for ax in "xyz" if ax != up[1]]
            # deep cells (owner follow-up 2026-09-19 21:25; board demo 0920-182445 failed its first grasp at a 0.80 m cell): cuRobo's
            # transit to a 45-deg pitched pre-grasp brushes the table edge with the forearm (right_arm_link5), which the sampler's
            # tracked-leg clearance does not cover - ``pitch45_max_depth_m`` drops the 45-deg classes beyond that base depth
            max45 = self.gen("pitch45_max_depth_m", None)
            depth_pick = float(manip.world_to_base(self._centre(pick_pose)[0][:2])[0])
            tilts = tuple(t for t in PICK_TILTS_DEG if not (max45 is not None and abs(t) > 40.0 and depth_pick > float(max45)))
            # straight down first, then the pitched approaches of the band's generic classes (the same tilt at the pick
            # and the place: no reorientation in hand) — the far cells of the band pass only with a pitched approach.
            # Tip clearance ladder (17:24 first-demo run: every place with the block attached failed the exact IK while the
            # picks passed — README "Backend facts": an object deep in the gripper self-collides with the non-exempt gripper
            # link once attached, so every attached check fails whatever the pose): the 1 mm pinch first, then 3 / 6 mm, and a
            # class whose attached cube self-collides (``IKChecker.attachable``, once per template per class) is skipped
            for tilt in tilts:  # owner 10:47 ladder: the pitched 30° / 45° picks first, straight down last
                for pair in pairs:
                    for k in range(4):
                        yaw = table_yaw + k * math.pi / 2
                        g_pick = None
                        for clearance in PICK_TIP_CLEARANCES:
                            if self._has_board() and self._board_fingertips_clear(clearance, placed_neighbours) is not None:
                                n_finger_low += 1
                                continue
                            try:
                                g_ = C.cube_grasp(self._centre(pick_pose), pair, tilt, yaw, edge=self.edge, tip_clearance=clearance)
                            except ValueError:
                                continue
                            if not g_.downward:
                                continue
                            if not self._class_attachable(block, pair, tilt, clearance, g_.eef_in_block):
                                continue
                            g_pick = g_
                            break
                        if g_pick is None:
                            continue
                        place_eef = C.compose(self._centre(place_pose), g_pick.eef_in_block)
                        if float(C.approach_axis(place_eef[1])[2]) > -0.1:
                            continue  # the place approach must point down as well
                        if min(hand_table_clearance(g_pick.eef, top_z), hand_table_clearance(place_eef, top_z)) <= 0.0:
                            continue
                        sweep = self._sweep_clear(place_eef, place_pose, placed_xy)
                        if sweep is not None:
                            n_sweep += 1
                            continue
                        screened.append((f"{pair}/t{tilt:+.0f}/y{k}", pair, g_pick, place_eef, clearance, None))
            if not screened:
                raise Reject("grasp_unreachable", f"block {names[bi]}: no cube grasp with downward pick and place approaches ({n_sweep} swept a placed neighbour"
                                                  + (f", {n_finger_low} would pass a finger below a seated neighbour's top" if n_finger_low else "") + ")")
            # classes whose wrist stays on the robot's side first (PLACE_FACTORS 19:45: at 0.7 m base-forward the pre-poses
            # whose lean puts the wrist beyond the block fail the IK while the grasp poses pass)
            d_pick0 = float(self.gen("pre_pick_dist_m", C.PRE_DIST))
            screened.sort(key=lambda c: float(manip.world_to_base(IK.pre_pose(c[2].eef, d_pick0)[0][:2])[0]) + float(manip.world_to_base(IK.pre_pose(c[3], d_pick0)[0][:2])[0]))
            screened = screened[: int(self.gen("max_pick_classes", 10))]
        # the oracle's Grasp approaches from PRE_DIST (10 cm) whatever the record says; the place follows the recorded pre_place
        d_pick = float(C.PRE_DIST)
        d_places = [float(v) for v in (self.gen("pre_place_dists_m", None) or (C.PRE_DIST, 0.05))]
        chosen = None
        diag: List[str] = []
        if plan["flips"] and bool(self.gen("flip_in_place_fallback", True)):
            # the direct flip's place passes only within a few cm LEFT of the pick cell (exact probe 2026-09-19: 6 cm left at 2-5
            # cells per trunk); a slot further left goes straight to the flip-in-place fallback (saves ~1 min per stance)
            d_left = float(th.dot(self._centre(place_pose)[0][:2] - self._centre(pick_pose)[0][:2], right))
            if -d_left > float(self.gen("flip_direct_max_slot_left_m", 0.04)):
                diag.append(f"direct flip skipped: the slot lies {-d_left * 100:.0f} cm left of the cell")
                screened = []
        for label, pair, g_pick, place_eef, clearance, lam in self._pick_turns(screened, manip, support, block, d_pick):
            pre_pick = IK.pre_pose(g_pick.eef, d_pick)
            lift = (g_pick.eef[0] + th.tensor([0.0, 0.0, self._carry_height()]), g_pick.eef[1])
            pivot = self._centre(pick_pose)[0] + th.tensor([0.0, 0.0, self._carry_height()])
            wps: List[TT.Pose] = []
            if plan["flips"]:
                # F5 order: (eef_pose, pivot, yaw [rad], flip_axis, flip_sign)
                wps = list(C.flip_waypoints(lift, pivot, math.radians(plan["yaw_deg"]), plan["flip_axis"], plan["flip_sign"], n_yaw=2, n_flip=3))
            elif abs(plan["yaw_deg"]) > 1e-6:
                wps = list(C.orbit_waypoints(lift, pivot, C.quat_from_yaw(math.radians(plan["yaw_deg"])), n=2))
            place_margin_pose = (place_eef[0] + _xy3(right) * self._place_margin() + _xy3(toward) * self._place_margin(), place_eef[1])
            # the fingertips in the head camera frame along the whole chain (the oracle's FOV span: pre-grasp -> release)
            frame_fail = self._chain_in_frame(manip, cam, {"pre_pick": pre_pick, "pick": g_pick.eef, "lift": lift, **{f"flip_wp{i}": w for i, w in enumerate(wps)},
                                                           "place": place_margin_pose})
            if frame_fail is not None:
                diag.append(f"{label}: {frame_fail} out of the head camera frame")
                continue
            fov_d = [d_ for d_ in d_places if self._chain_in_frame(manip, cam, {"pre_place": IK.pre_pose(place_margin_pose, d_)}) is None]
            if not fov_d:
                diag.append(f"{label}: pre_place out of the head camera frame at every pre distance {d_places}")
                continue
            # the block itself is removed along the whole approach (a steep side pinch puts the pad centre outside the block's
            # AABB, so the checker's inference would keep the block in the world and the approach 'collides' with it)
            res = self._exact2([pre_pick, g_pick.eef], manip, ignore_for_last=[support, block], approached=[block])
            if not res.ok:
                diag.append(f"{label}: pick exact {res.reason}")
                continue
            q_pre = self.ikc.solutions.get(("right", self.ikc._pose_key(pre_pick)))
            bad = self._arm_clear(res.q) or self._approach_clear(pre_pick, g_pick.eef, [support, block])
            if bad is not None:
                diag.append(f"{label}: pick: {bad}")
                continue
            # place (and pre-place) with the block in hand; the world: placed blocks at their slots, this block removed (attached)
            pre_place = None
            res_p = None
            with TT.posed_in_hand(self, "right", block, C.relative(g_pick.eef, pick_pose)):
                with attached(self.kit, "right", block):
                    r_l = self._exact2([lift], manip, ignore_for_last=[support], cache=False)
                    if not r_l.ok:
                        diag.append(f"{label}: lift {r_l.reason}")
                        continue
                    if wps:
                        r_w = self._waypoints_ok2([lift] + wps, manip, ignore_all=[support])
                        if not r_w.ok:
                            diag.append(f"{label}: flip waypoints {r_w.reason}")
                            continue
                    # the oracle's motion from the grasp: the lift (a free leg, finish allowed), then the reorient chain (no finish in the
                    # oracle: hard), tracked from the pick configuration
                    q_end, why = self._tracked_from(res.q, [g_pick.eef, lift], [support])
                    if q_end is None:
                        diag.append(f"{label}: lift from the pick {why}")
                        continue
                    if wps:
                        q_end, why = self._tracked_from(q_end, [lift] + wps, [support], hard=True)
                        if q_end is None:
                            diag.append(f"{label}: reorient from the lift {why}")
                            continue
                    for d_place in fov_d:
                        pre_place = IK.pre_pose(place_margin_pose, d_place)
                        res_p = self._exact2([pre_place, place_margin_pose], manip, ignore_for_last=place_ignore)
                        if res_p.ok:
                            break
                    if res_p is None or not res_p.ok:
                        diag.append(f"{label}: place exact {res_p.reason if res_p is not None else 'no pre-place distance'}")
                        if len([d_ for d_ in diag if "place exact" in d_]) <= 2:
                            self._place_failure_factors(manip, block, pick_pose, g_pick, pre_place, place_margin_pose, support, names, bi, world_poses)
                        continue
                    bad = self._arm_clear(res_p.q)
                    if bad is not None:
                        diag.append(f"{label}: place: {bad}")
                        continue
                    r_r = self._retreat_ok(place_margin_pose, manip, place_ignore)
                    if not r_r.ok:
                        diag.append(f"{label}: place retreat {r_r.reason}")
                        continue
                    # the carry leg from the reoriented / lifted hand to the pre-place: a straight leg the oracle tracks (FOV rule b)
                    start = wps[-1] if wps else lift
                    leg = list(C.line_waypoints(start, pre_place, 3))
                    r_c = self._waypoints_ok2([start] + leg, manip, ignore_all=[support] + placed_neighbours)
                    if not r_c.ok:
                        diag.append(f"{label}: carry leg {r_c.reason}")
                        continue
                    # ... and the same carry leg + descent + retreat tracked from where the reorient left the arm
                    q_end, why = self._tracked_from(q_end, [start] + leg + [place_margin_pose] + list(C.line_waypoints(place_margin_pose, IK.pre_pose(place_margin_pose, RETREAT_DIST_M), 3)),
                                                    place_ignore)
                    if q_end is None:
                        diag.append(f"{label}: carry + place + retreat from the reorient {why}")
                        continue
            chosen = (label, pair, g_pick, place_margin_pose, pre_pick, pre_place, wps, clearance, lam)
            self._chosen_pick_q = self._arm_q_list(res.q)
            self._chain_pre_q, self._chain_end_q = self._arm_q_list(q_pre), q_end
            break
        if chosen is None and plan["flips"] and bool(self.gen("flip_in_place_fallback", True)):
            # owner 2026-09-19 "flip in place, then move": the in-hand flip's place LEFT of the pick cell (the goal grid hangs at
            # the leftmost block) is the one step the exact chain rarely passes; the block is turned in hand and set back on its
            # own cell in the goal orientation (leg 1), then re-grasped with a plain pitched pinch and placed at its slot (leg 2)
            self._placed_xy = placed_xy
            rec2 = self._flip_in_place_chain(manip, names, bi, block, support, pick_pose, place_pose, plan, table_yaw, right, toward, cam, place_ignore, placed_neighbours,
                                             world_poses, surface_z, step, goal_pose, d_places, diag)
            if rec2 is not None:
                return self._finish_record(rec2, names, bi, step, world_poses, goal_pose, order, right, toward, rec2.get("_regrasp_stance_obj") or manip)
        if chosen is None and plan["flips"] and self.gen("block_table_flip", False):
            # §6.5 flip-on-the-table (owner 2026-09-16 17:10, prep behind ``generation.block_table_flip``): grasp → place on a
            # free region cell in an intermediate orientation → re-grasp with the other pitch class → place at the slot
            legs = self._table_flip_chain(rng, manip, names, bi, poses, goal_pose, world_poses, right, toward, table_yaw, surface_z, plan)
            if legs is not None:
                return self._table_flip_record(names, bi, step, legs, plan, goal_pose, surface_z, manip, world_poses, order, right, toward)
        if chosen is None:
            b_cell = manip.world_to_base(self._centre(pick_pose)[0][:2])
            b_slot = manip.world_to_base(self._centre(place_pose)[0][:2])
            # one line per stage so the tail of the fallback (the re-grasp from other stances) is visible too
            stages: Dict[str, List[str]] = {}
            for d_ in diag:
                key = d_.split(" ")[0] if d_.startswith(("in-place", "re-grasp", "direct")) else "direct"
                stages.setdefault(key, []).append(d_)
            summary = "; ".join(f"{k_}: {len(v_)} failed, e.g. {' | '.join(x[:150] for x in v_[:3])}" for k_, v_ in stages.items())
            log.info("%s: block %s from trunk %.2f/%+.2f so %.2f (shift %+.2f, left arm %s; cell base (%.2f, %+.2f), slot base (%.2f, %+.2f)%s): %d direct classes; %s", self.gp.alias,
                     names[bi], manip.translate, manip.tilt, float(manip.stand_off or 0.0), float(manip.extras.get("block_stance_shift_m", 0.0)), manip.idle_posture or "untucked",
                     float(b_cell[0]), float(b_cell[1]), float(b_slot[0]), float(b_slot[1]), f", yaw-align {plan['yaw_deg']:+.0f} deg" if plan["flips"] else "", len(screened), summary)
            raise Reject("place_unreachable" if any("place" in d_ or "carry" in d_ or "flip" in d_ for d_ in diag) else "grasp_unreachable",
                         f"block {names[bi]}: no cube grasp passes pick + lift + flip + place exact IK from this stance ({'; '.join(diag[:3])})")
        label, pair, g_pick, place_eef, pre_pick, pre_place, wps, clearance, lam = chosen
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
                               "centre_offset": [float(v) for v in self.c_local.tolist()],
                               # the pinch depth the oracle must reproduce (``Grasp[cube]`` reads both; owner 2026-09-19)
                               "tip_clearance_m": float(clearance), "edge_m": float(self.edge), "grasp_class": label,
                               # the blocks already at their slots, ignored for the place + its retreat (finger clearance guaranteed
                               # geometrically, see ``placed_neighbours``); the program check removes them for those poses too
                               "ignore_place": self._bddl_names([o for o in place_ignore if o is not support]),
                               # the arm joints of the sampler's pick configuration: the program check seeds its grasp IK there, so its
                               # witnesses (the branch the oracle runs the chain in) are the branch the tracked legs above were checked in
                               "pick_q": getattr(self, "_chosen_pick_q", None)}
        if lam is not None:
            rec["flip_class"] = label
            rec["flip_lean_deg"] = float(lam)
        if wps:
            rec["waypoints"] = [pose_to_sidecar(w) for w in wps]
        return self._finish_record(rec, names, bi, step, world_poses, goal_pose, order, right, toward, manip)

    def _finish_record(self, rec: Dict[str, Any], names: List[str], bi: int, step: int, world_poses: Dict[str, TT.Pose], goal_pose: Dict[int, TT.Pose], order: List[int],
                       right: th.Tensor, toward: th.Tensor, manip: Stance) -> Dict[str, Any]:
        """The record's tail: the root pose the block has after its final place (``goal_pose``: the program check advances the
        live block there) and the brace / push contacts."""
        world_after = dict(world_poses)
        world_after[names[bi]] = (goal_pose[bi][0] + _xy3(right) * self._place_margin() + _xy3(toward) * self._place_margin(), goal_pose[bi][1])
        rec["goal_pose"] = {"position": [float(v) for v in world_after[names[bi]][0]], "orientation": [float(v) for v in world_after[names[bi]][1]]}
        # brace points (§6.5): row-left / column-top blocks braced by the closed left gripper while the right hand pushes.
        # Owner decision 2026-09-16 11:08 (same rule as axial): the brace and its paired push are recorded only when the
        # joint two-arm placement exists for this block (brace point in the left contact band at the manipulation trunk,
        # blade clear of the placed blocks, two-arm IK passing); otherwise the block is placed without a brace / push tail
        n_cols = int((self.g("grid") or [2, 2])[1])
        r, c = divmod(step, n_cols)
        self._teleport(world_after)
        outcomes: Dict[str, str] = {}
        if not bool(self.gen("block_brace_push", True)):
            # generation.block_brace_push false (owner 2026-09-19, first demo): no brace / push tail at all - the push closes the
            # block onto the row-left / column-top neighbour, which with the 10 mm layout gaps moves it off its slot
            rec["brace"] = "disabled"
            rec["brace_reasons"] = {}
            return rec
        if c > 0:  # push left against the row, brace the row-left block's left face
            left_bi = order[step - 1]
            outcomes["row"] = self._brace_push(rec, "row", names[bi], names[left_bi], world_after, direction=-right, brace_dir=right, manip=manip)
        if r > 0:  # push up (away from the robot) against the column, brace the column-top block's far face
            top_bi = order[step - n_cols]
            outcomes["col"] = self._brace_push(rec, "col", names[bi], names[top_bi], world_after, direction=toward * -1.0, brace_dir=toward, manip=manip)
        rec["brace"] = "present" if any(v == "present" for v in outcomes.values()) else ("none" if outcomes else "not_needed")
        rec["brace_reasons"] = outcomes
        return rec

    def _flip_in_place_chain(self, manip: Stance, names: List[str], bi: int, block, support, pick_pose: TT.Pose, place_pose: TT.Pose, plan: Dict[str, Any], table_yaw: float,
                             right: th.Tensor, toward: th.Tensor, cam: TT.Pose, place_ignore: List[Any], placed_neighbours: List[Any], world_poses: Dict[str, TT.Pose],
                             surface_z: Optional[float], step: int, goal_pose: Dict[int, TT.Pose], d_places: List[float], diag: List[str]) -> Optional[Dict[str, Any]]:
        """Owner 2026-09-19 "flip in place, then move" (two grasp -> release legs from ONE stance): leg 1 = the flip class's side
        pick, lift, the recorded flip waypoints (yaw-align + quarter turn) and the set-down on the block's OWN cell in the
        table-aligned goal orientation; leg 2 = a plain pitched pinch of the turned block, lift, place at the slot (margin) with
        the carry leg. Every pose exact-IK-checked as in :meth:`_block_chain` (fingertips in frame). Returns the block record
        (top-level fields = leg 1's grasp / flip, ``place`` / ``pre_place`` = leg 2's, ``table_flip`` = the set-down + re-grasp
        the oracle program consumes) or None."""
        cell_xy = self._centre(pick_pose)[0][:2]
        top_z = float(self.top.z)
        d_pick = float(C.PRE_DIST)
        carry = self._carry_height()
        self._leg1_pick_q = self._leg2_pick_q = None
        # set-down spots: the block's own cell first, then a few centimetres toward the robot's RIGHT (the flip class ends with
        # the hand left of the set-down point, so a spot further right sits deeper in the arm's zone), then slightly left; a
        # spot must stay on the top and clear of every other block (init or slot) by the scatter gap
        others = [self._centre(p)[0][:2] for n_, p in world_poses.items() if n_ != names[bi]] + [self._centre(g_)[0][:2] for j_, g_ in goal_pose.items() if j_ != bi]
        gap = float(self.g("scatter_min_gap_m")) + self.edge * math.sqrt(2)
        spots: List[th.Tensor] = []
        # (right offset, depth offset away from the robot): the own cell first, then deeper spots - the goal grid hangs from the
        # deep-left anchor, and the re-grasp of a block set down 13 cm in from the edge failed at every stance (0919-081444)
        # while the same re-grasp passed 29 cm in - then a little right (deeper in the flip class's zone)
        for off_r, off_d in ((0.0, 0.0), (0.0, 0.06), (0.0, 0.10), (0.0, 0.14), (-0.04, 0.0), (-0.04, 0.08)):
            xy = cell_xy + right * off_r - toward * off_d
            if not self.top.contains(xy, 0.5 * self.edge + 0.02):
                continue
            if any(float(th.linalg.norm(xy - o)) < gap for o in others):
                continue
            spots.append(xy)
        if not spots:
            return None

        def own_goal_at(xy: th.Tensor) -> TT.Pose:
            og = self._root_pose(xy, plan["goal_quat"], 0.001)
            bottom = float(og[0][2]) - self.world.bottom_offset(self.hull, og[1])
            return (og[0] + th.tensor([0.0, 0.0, max(0.0, top_z + RELEASE_ABOVE_SURFACE - bottom)]), og[1])  # the set-down is on the table, not the board

        # --- leg 1: side pick + flip + set-down on the own cell (or a spot beside it) ------------------------------------
        leg1 = None
        classes1 = []
        for si, xy in enumerate(spots):
            for cls in self._flip_classes(block, pick_pose, own_goal_at(xy), plan, table_yaw)[: int(self.gen("max_flip_classes_in_place", 3))]:
                classes1.append((si, xy) + tuple(cls))
        classes1.sort(key=lambda c: (c[0], c[2], c[3]))  # own cell first, then by lean preference / camera roll
        self._teleport(world_poses)
        leg2, st_leg2 = None, manip
        spots_failed_leg2: set = set()
        for si, own_xy, _, _, pair, tilt, g, place1, clearance, label, lam in classes1:
            if si in spots_failed_leg2:
                continue  # the set-down spot whose re-grasp failed at every stance: another spot, not another flip class
            own_goal = own_goal_at(own_xy)
            d_r, d_d = float(th.dot(own_xy - cell_xy, right)) * 100, -float(th.dot(own_xy - cell_xy, toward)) * 100
            label = f"{label}@{'own' if si == 0 else f'r{d_r:+.0f}d{d_d:+.0f}cm'}"
            pre_pick = IK.pre_pose(g.eef, d_pick)
            lift = (g.eef[0] + th.tensor([0.0, 0.0, carry]), g.eef[1])
            pivot = self._centre(pick_pose)[0] + th.tensor([0.0, 0.0, carry])
            wps = list(C.flip_waypoints(lift, pivot, math.radians(plan["yaw_deg"]), plan["flip_axis"], plan["flip_sign"], n_yaw=2, n_flip=3))
            frame_fail = self._chain_in_frame(manip, cam, {"pre_pick": pre_pick, "pick": g.eef, "lift": lift, **{f"flip_wp{i}": w for i, w in enumerate(wps)}, "place1": place1})
            if frame_fail is not None:
                diag.append(f"in-place {label}: {frame_fail} out of the head camera frame")
                continue
            fov_d1 = [d_ for d_ in d_places if self._chain_in_frame(manip, cam, {"pre_place1": IK.pre_pose(place1, d_)}) is None]
            if not fov_d1:
                diag.append(f"in-place {label}: pre_place1 out of the head camera frame at every pre distance")
                continue
            res = self._exact2([pre_pick, g.eef], manip, ignore_for_last=[support, block], approached=[block])
            if not res.ok:
                diag.append(f"in-place {label}: pick exact {res.reason}")
                continue
            q_pre1 = self.ikc.solutions.get(("right", self.ikc._pose_key(pre_pick)))
            bad = self._arm_clear(res.q) or self._approach_clear(pre_pick, g.eef, [support, block])
            if bad is not None:
                diag.append(f"in-place {label}: pick: {bad}")
                continue
            pre_place1 = None
            with TT.posed_in_hand(self, "right", block, C.relative(g.eef, pick_pose)):
                with attached(self.kit, "right", block):
                    r_l = self._exact2([lift], manip, ignore_for_last=[support], cache=False)
                    if not r_l.ok:
                        diag.append(f"in-place {label}: lift {r_l.reason}")
                        continue
                    r_w = self._waypoints_ok2([lift] + wps, manip, ignore_all=[support])
                    if not r_w.ok:
                        diag.append(f"in-place {label}: flip waypoints {r_w.reason}")
                        continue
                    q_end1, why = self._tracked_from(res.q, [g.eef, lift], [support])
                    if q_end1 is None:
                        diag.append(f"in-place {label}: lift from the pick {why}")
                        continue
                    q_end1, why = self._tracked_from(q_end1, [lift] + wps, [support], hard=True)
                    if q_end1 is None:
                        diag.append(f"in-place {label}: flip from the lift {why}")
                        continue
                    ok_p = None
                    for d_place in fov_d1:
                        pre_place1 = IK.pre_pose(place1, d_place)
                        ok_p = self._exact2([pre_place1, place1], manip, ignore_for_last=[support])
                        if ok_p.ok:
                            break
                    if ok_p is None or not ok_p.ok:
                        diag.append(f"in-place {label}: set-down exact {ok_p.reason if ok_p else 'no pre distance'}")
                        continue
                    bad = self._arm_clear(ok_p.q)
                    if bad is not None:
                        diag.append(f"in-place {label}: set-down: {bad}")
                        continue
                    r_r = self._retreat_ok(place1, manip, [support])
                    if not r_r.ok:
                        diag.append(f"in-place {label}: set-down retreat {r_r.reason}")
                        continue
                    leg = list(C.line_waypoints(wps[-1], pre_place1, 3))
                    r_c = self._waypoints_ok2([wps[-1]] + leg, manip, ignore_all=[support])
                    if not r_c.ok:
                        diag.append(f"in-place {label}: carry to the set-down {r_c.reason}")
                        continue
                    q_end1, why = self._tracked_from(q_end1, [wps[-1]] + leg + [place1] + list(C.line_waypoints(place1, IK.pre_pose(place1, RETREAT_DIST_M), 3)), [support])
                    if q_end1 is None:
                        diag.append(f"in-place {label}: carry + set-down + retreat from the flip {why}")
                        continue
            leg1 = (label, pair, g, place1, pre_place1, wps, clearance, lam, own_xy, own_goal)
            self._leg1_pick_q = self._arm_q_list(res.q)
            self._chain_pre_q = self._arm_q_list(q_pre1)
            # --- leg 2 for this set-down: pitched re-grasp of the turned block, place at the slot -------------------------
            # from the leg-1 stance first, then from the pool stances ranked for this pick / place (the slot near the centre
            # line is easiest from the stance the non-flipping blocks used); the program travels there with empty hands
            leg2, st_leg2 = self._regrasp_leg(manip, names, bi, block, support, own_goal, place_pose, table_yaw, right, toward, cam, place_ignore,
                                              placed_neighbours, world_poses, d_places, diag, label)
            if leg2 is not None:
                break
            spots_failed_leg2.add(si)
            leg1 = None
            self._chain_pre_q = None
            self._teleport(world_poses)
        if leg1 is None or leg2 is None:
            return None
        label1, pair1, g1, place1, pre_place1, wps1, clr1, lam1, own_xy, own_goal = leg1
        label2, pair2, g2, place2m, pre_pick2, pre_place2, clr2 = leg2
        if st_leg2 is not manip:
            log.info("%s: block %s re-grasped from another stance (trunk %.2f/%+.2f, shift %+.2f)", self.gp.alias, names[bi], st_leg2.translate, st_leg2.tilt,
                     float(st_leg2.extras.get("block_stance_shift_m", 0.0)))
        rec = self._flip_in_place_record(manip, st_leg2, names, bi, block, plan, pick_pose, place_pose, own_goal, leg1, leg2, d_pick, carry, surface_z, step, goal_pose)
        rec["ignore_place"] = self._bddl_names([o for o in place_ignore if o is not support])
        return rec

    def _regrasp_leg(self, manip: Stance, names: List[str], bi: int, block, support, own_goal: TT.Pose, place_pose: TT.Pose, table_yaw: float, right: th.Tensor,
                     toward: th.Tensor, cam: TT.Pose, place_ignore: List[Any], placed_neighbours: List[Any], world_poses: Dict[str, TT.Pose], d_places: List[float],
                     diag: List[str], label1: str) -> Tuple[Optional[Tuple[Any, ...]], Stance]:
        """Leg 2 of the flip-in-place fallback for one set-down pose ``own_goal``: the pitched re-grasp of the turned block, lift,
        place at the slot (margin) with the carry leg, from ``manip`` first, then from the proven / ranked pool stances. Returns
        (leg tuple or None, the stance)."""
        top_z = float(self.top.z)
        d_pick = float(C.PRE_DIST)
        carry = self._carry_height()
        world2 = dict(world_poses)
        world2[names[bi]] = own_goal
        self._teleport(world2)
        up = TT.up_face(own_goal[1])
        pairs = [ax for ax in "xyz" if ax != up[1]]
        cands2 = []
        for tilt in PICK_TILTS_DEG:
            for pair in pairs:
                for k in range(4):
                    yaw = table_yaw + k * math.pi / 2
                    g2 = None
                    for clearance in PICK_TIP_CLEARANCES:
                        if self._has_board() and self._board_fingertips_clear(clearance, placed_neighbours) is not None:
                            continue  # puzzle board: a finger beside a seated neighbour passes above its top
                        try:
                            g_ = C.cube_grasp(self._centre(own_goal), pair, tilt, yaw, edge=self.edge, tip_clearance=clearance)
                        except ValueError:
                            continue
                        if g_.downward and self._class_attachable(block, pair, tilt, clearance, g_.eef_in_block):
                            g2 = g_
                            break
                    if g2 is None:
                        continue
                    place2 = C.compose(self._centre(place_pose), g2.eef_in_block)
                    if float(C.approach_axis(place2[1])[2]) > -0.1:
                        continue
                    if min(hand_table_clearance(g2.eef, top_z), hand_table_clearance(place2, top_z)) <= 0.0:
                        continue
                    if self._sweep_clear(place2, place_pose, getattr(self, "_placed_xy", [])) is not None:
                        continue  # the tilted descent would sweep the block through a placed neighbour (demo 0919-121912)
                    cands2.append((f"{pair}/t{tilt:+.0f}/y{k}", pair, g2, place2, clearance))

        probed: Dict[str, bool] = {}

        def try_leg2(st2: Stance):
            cam2 = cam if st2 is manip else self.camera_pose_at(st2)
            # up to 4 classes per tilt (the pre-pose-depth order alone starves the straight-down classes), nearer pre-poses first
            by_depth = sorted(cands2, key=lambda c: float(st2.world_to_base(IK.pre_pose(c[2].eef, d_pick)[0][:2])[0]) + float(st2.world_to_base(IK.pre_pose(c[3], d_pick)[0][:2])[0]))
            # round-robin over the tilts (nearest pre-poses first within a tilt) so the class cap never starves a tilt
            by_tilt: Dict[float, List[Any]] = {}
            for c in by_depth:
                by_tilt.setdefault(round(float(c[2].tilt_deg), 1), []).append(c)
            ordered = []
            for i_ in range(max((len(v) for v in by_tilt.values()), default=0)):
                for t_ in PICK_TILTS_DEG:
                    lst = by_tilt.get(round(float(t_), 1), [])
                    if i_ < len(lst):
                        ordered.append(lst[i_])
            tag = "re-grasp" if st2 is manip else f"re-grasp@{st2.translate:.2f}/{st2.tilt:+.2f}/sh{float(st2.extras.get('block_stance_shift_m', 0.0)):+.2f}"
            for label2, pair2, g2, place2, clr2 in self._pick_turns(ordered[: int(self.gen("max_regrasp_classes", 12))], st2, support, block, d_pick):
                pre_pick2 = IK.pre_pose(g2.eef, d_pick)
                lift2 = (g2.eef[0] + th.tensor([0.0, 0.0, carry]), g2.eef[1])
                place2m = (place2[0] + _xy3(right) * self._place_margin() + _xy3(toward) * self._place_margin(), place2[1])
                frame_fail = self._chain_in_frame(st2, cam2, {"pre_pick2": pre_pick2, "pick2": g2.eef, "lift2": lift2, "place2": place2m})
                if frame_fail is not None:
                    diag.append(f"{tag} {label2}: {frame_fail} out of the head camera frame")
                    continue
                fov_d2 = [d_ for d_ in d_places if self._chain_in_frame(st2, cam2, {"pre_place2": IK.pre_pose(place2m, d_)}) is None]
                if not fov_d2:
                    diag.append(f"{tag} {label2}: pre_place2 out of the head camera frame at every pre distance")
                    continue
                res = self._exact2([pre_pick2, g2.eef], st2, ignore_for_last=[support, block], approached=[block])
                if res.ok:
                    bad = self._arm_clear(res.q) or self._approach_clear(pre_pick2, g2.eef, [support, block])
                    if bad is not None:
                        diag.append(f"{tag} {label2}: pick: {bad}")
                        continue
                if not res.ok:
                    diag.append(f"{tag} {label2}: pick exact {res.reason}")
                    if not probed.get(tag):
                        # once per stance (0919-081444: every re-grasp class of the second flipper failed 'pose 1 of 2' at every
                        # stance while the plain picks passed): the pre pose and the pick alone, the block + support removed
                        probed[tag] = True
                        r_a = self._exact2([pre_pick2], st2, ignore_for_last=[support, block], cache=False)
                        r_b = self._exact2([g2.eef], st2, ignore_for_last=[support, block], cache=False)
                        r_c = self._exact2([pre_pick2], st2, ignore_for_last=[support], cache=False)
                        pb = st2.world_to_base(pre_pick2[0][:2])
                        diag.append(f"{tag} probe {label2}: pre alone w/o block {'ok' if r_a.ok else r_a.reason}; pick alone w/o block {'ok' if r_b.ok else r_b.reason}; "
                                    f"pre alone w/o support {'ok' if r_c.ok else r_c.reason}; pre base ({float(pb[0]):.2f}, {float(pb[1]):.2f}, z {float(pre_pick2[0][2]):.2f}), "
                                    f"hand-table {hand_table_clearance(pre_pick2, top_z) * 100:.1f} cm, block centre z {float(self._centre(own_goal)[0][2]):.3f} top {top_z:.3f}")
                    continue
                pre_place2 = None
                with TT.posed_in_hand(self, "right", block, C.relative(g2.eef, own_goal)):
                    with attached(self.kit, "right", block):
                        r_l = self._exact2([lift2], st2, ignore_for_last=[support], cache=False)
                        if not r_l.ok:
                            diag.append(f"{tag} {label2}: lift {r_l.reason}")
                            continue
                        q_end2, why = self._tracked_from(res.q, [g2.eef, lift2], [support])
                        if q_end2 is None:
                            diag.append(f"{tag} {label2}: lift from the re-grasp {why}")
                            continue
                        ok_p = None
                        for d_place in fov_d2:
                            pre_place2 = IK.pre_pose(place2m, d_place)
                            ok_p = self._exact2([pre_place2, place2m], st2, ignore_for_last=place_ignore)
                            if ok_p.ok:
                                break
                        if ok_p is None or not ok_p.ok:
                            diag.append(f"{tag} {label2}: place exact {ok_p.reason if ok_p else 'no pre distance'}")
                            continue
                        bad = self._arm_clear(ok_p.q)
                        if bad is not None:
                            diag.append(f"{tag} {label2}: place: {bad}")
                            continue
                        r_r = self._retreat_ok(place2m, st2, place_ignore)
                        if not r_r.ok:
                            diag.append(f"{tag} {label2}: place retreat {r_r.reason}")
                            continue
                        leg = list(C.line_waypoints(lift2, pre_place2, 3))
                        r_c = self._waypoints_ok2([lift2] + leg, st2, ignore_all=[support] + placed_neighbours)
                        if not r_c.ok:
                            diag.append(f"{tag} {label2}: carry leg {r_c.reason}")
                            continue
                        q_end2, why = self._tracked_from(q_end2, [lift2] + leg + [place2m] + list(C.line_waypoints(place2m, IK.pre_pose(place2m, RETREAT_DIST_M), 3)), place_ignore)
                        if q_end2 is None:
                            diag.append(f"{tag} {label2}: carry + place + retreat from the lift {why}")
                            continue
                self._leg2_pick_q = self._arm_q_list(res.q)
                self._chain_end_q = q_end2
                return (label2, pair2, g2, place2m, pre_pick2, pre_place2, clr2)
            return None

        leg2 = try_leg2(manip)
        st_leg2 = manip
        if leg2 is None:
            # the stances that already solved plain picks / places of this instance first (the non-flipping blocks' stance
            # reaches the goal grid by construction), then the pool ranked for this pick / place
            proven = [st for st in getattr(self, "_proven_stances", []) if st is not manip]
            ranked = [st for st in self._rank_stances(getattr(self, "_pool", [manip]), manip, own_goal, place_pose, False) if st is not manip and all(st is not p_ for p_ in proven)]
            for st2 in (proven + ranked)[: int(self.gen("max_regrasp_stances", 5))]:
                leg2 = try_leg2(st2)
                if leg2 is not None:
                    st_leg2 = st2
                    break
        self._teleport(world_poses)
        if leg2 is None:
            diag.append(f"in-place {label1}: leg 1 passed, no re-grasp stance for the block set down there")
        return leg2, st_leg2

    def _flip_in_place_record(self, manip: Stance, st_leg2: Stance, names: List[str], bi: int, block, plan: Dict[str, Any], pick_pose: TT.Pose, place_pose: TT.Pose,
                              own_goal: TT.Pose, leg1: Tuple[Any, ...], leg2: Tuple[Any, ...], d_pick: float, carry: float, surface_z: Optional[float], step: int,
                              goal_pose: Dict[int, TT.Pose]) -> Dict[str, Any]:
        """The block record of a flip-in-place chain (top-level fields = leg 1's grasp / flip, ``place`` / ``pre_place`` = leg 2's,
        ``table_flip`` = the set-down + re-grasp the oracle program consumes)."""
        label1, pair1, g1, place1, pre_place1, wps1, clr1, lam1, own_xy, own_goal = leg1
        label2, pair2, g2, place2m, pre_pick2, pre_place2, clr2 = leg2
        q0 = C.qmul(g1.eef[1], C.quat_from_axis_angle([0.0, 1.0, 0.0], -math.radians(g1.tilt_deg)))
        rec: Dict[str, Any] = {"instance": names[bi], "order": step, "face_pair": pair1, "tilt_pick_deg": float(g1.tilt_deg), "tilt_place_deg": float(_tilt_about_y(q0, place1[1])),
                               "yaw_deg": float(math.degrees(g1.yaw)), "yaw_align_deg": float(plan["yaw_deg"]),
                               "flip_axis": None if plan["flip_axis"] is None else [float(v) for v in plan["flip_axis"]], "flip_sign": int(plan["flip_sign"]),
                               "pick": pose_to_sidecar(g1.eef), "pre_pick": pose_to_sidecar(IK.pre_pose(g1.eef, d_pick)),
                               "place": pose_to_sidecar(place2m), "pre_place": pose_to_sidecar(pre_place2), "carry_height": float(carry),
                               "goal_face": TT.up_face(goal_pose[bi][1]), "slot": step, "release_surface_z": None if surface_z is None else round(float(surface_z), 4),
                               "centre_offset": [float(v) for v in self.c_local.tolist()], "tip_clearance_m": float(clr1), "edge_m": float(self.edge),
                               "grasp_class": label1, "flip_class": label1, "flip_lean_deg": float(lam1), "flip_mode": "in_place_then_move",
                               "pick_q": getattr(self, "_leg1_pick_q", None),
                               "waypoints": [pose_to_sidecar(w) for w in wps1],
                               "table_flip": {"place": pose_to_sidecar(place1), "pre_place": pose_to_sidecar(pre_place1), "cell": [float(v) for v in own_xy.tolist()],
                                              "in_hand_flip": True, "relation": "back on its own cell, turned to its goal face, for the re-grasp",
                                              "block_pose_after": {"position": [float(v) for v in own_goal[0]], "orientation": [float(v) for v in own_goal[1]]},
                                              "regrasp": {"face_pair": pair2, "tilt_pick_deg": float(g2.tilt_deg), "yaw_deg": float(math.degrees(g2.yaw)), "carry_height": float(carry),
                                                          "pick": pose_to_sidecar(g2.eef), "pre_pick": pose_to_sidecar(pre_pick2), "tip_clearance_m": float(clr2), "grasp_class": label2,
                                                          "pick_q": getattr(self, "_leg2_pick_q", None)}}}
        if st_leg2 is not manip:
            rec["_regrasp_stance_obj"] = st_leg2  # named + travel recorded by _register_stances, removed before the sidecar is written
        log.info("%s: block %s flips in place with %s, re-grasped with %s (trunk %.2f/%+.2f, shift %+.2f)", self.gp.alias, names[bi], label1, label2, manip.translate, manip.tilt,
                 float(manip.extras.get("block_stance_shift_m", 0.0)))
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
                        place_pose = (place_pose[0] + th.tensor([0.0, 0.0, max(0.0, surface_z + self._release_above() - bottom)]), place_pose[1])
                    leg2 = self._leg_ik(manip, block, pose_mid, place_pose, world_mid, table_yaw, place_margin=_xy3(right) * self._place_margin() + _xy3(toward) * self._place_margin())
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

    def _sweep_clear(self, place_eef: TT.Pose, place_pose: TT.Pose, neighbour_xy: List[th.Tensor]) -> Optional[str]:
        """Whether a tilted place approach sweeps the carried block through a placed neighbour. The descent runs along the
        approach axis ``a`` from ``d`` back: the block is still above the neighbours' top only while its bottom is above the
        edge, i.e. over the first ``(edge - RELEASE_ABOVE_SURFACE) / |a_z|`` of the last centimetres; below that its footprint
        must clear every neighbour's. Demo 0919-121912 (local): the y/t+30 places at slots 1 and 3 came in leaning toward the
        left column and shoved the placed neighbour 4 cm along the table (both left-column blocks 3.6 cm off, 'not_reached').
        Returns None when clear, else the reason."""
        a = C.approach_axis(place_eef[1])
        a_xy = th.as_tensor(a[:2], dtype=th.float32)
        if float(th.linalg.norm(a_xy)) < 0.05 or not neighbour_xy:
            return None
        slot_xy = self._centre(place_pose)[0][:2]
        # the horizontal travel of the block from where its bottom passes the neighbours' top down to the release
        travel = a_xy / max(abs(float(a[2])), 0.2) * max(self.edge - self._release_above(), 0.0)
        start = slot_xy - travel  # where the block is when its bottom reaches the neighbours' top
        clear = self.edge + 0.003  # aligned squares: the footprints overlap when both centre offsets are below one edge
        R = C.quat_to_mat(TT._t(place_pose[1]))
        axes = [R[:2, i] for i in range(3) if abs(float(R[2, i])) < 0.5]  # the block's two horizontal axes (table-aligned grid)
        for o in neighbour_xy:
            for t_ in (0.0, 0.25, 0.5, 0.75):
                xy = start + travel * t_
                d = xy - o[:2]
                if all(abs(float(th.dot(d, ax / th.linalg.norm(ax)))) < clear for ax in axes):
                    side = th.dot(a_xy / th.linalg.norm(a_xy), (o[:2] - slot_xy) / max(float(th.linalg.norm(o[:2] - slot_xy)), 1e-6))
                    return f"the tilted place approach sweeps through the placed neighbour at {float(th.linalg.norm(o[:2] - slot_xy)) * 100:.1f} cm (approach toward it, cos {float(side):.2f})"
        return None

    def _arm_q_list(self, q) -> Optional[List[float]]:
        """The right arm's 7 joint values of a 28-D configuration (frame-free: the program check seeds its grasp IK with them)."""
        if q is None:
            return None
        idx = self.kit.robot.arm_control_idx["right"]
        return [round(float(v), 5) for v in TT._t(q)[idx].tolist()]

    def _release_above(self) -> float:
        return float(self.gen("release_above_surface_m", RELEASE_ABOVE_SURFACE))

    # ------------------------------------------------------------------ board variant (owner 2026-09-19 21:47, option 2)

    def _has_board(self) -> bool:
        return "board" in self.gp.roles

    def _board_geometry(self) -> Dict[str, Any]:
        """The puzzle board's build manifest (``assets/objects/block_board/model/isgboard/build_manifest.json``; owner 2026-09-20
        22:39 / 23:00): ``pocket_pitch_m``, ``seat_inner_m``, ``outer_mouth_m``, ``wall_h_m``, ``chamfer_h_m``, ``outer_chamfer_w_m``,
        ``divider_w_m``, ``divider_h_m``, ``ridge_apex_z_m`` (above the pocket floor), ``rim_top_z_m`` (above the board's bottom),
        ``rim_top_above_floor_m``, ``plate_thickness_m``, ``plate_m``, ``pocket_centres`` (board frame, slot order 0 far-left, 1
        far-right, 2 near-left, 3 near-right) and ``outer_corner_sign`` per pocket. Falls back to the built values (drop test v5)."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "assets", "objects", "block_board", "model", "isgboard",
                            "build_manifest.json")
        h = 0.0157
        geo: Dict[str, Any] = {"pocket_pitch_m": 0.0314, "seat_inner_m": 0.0274, "outer_mouth_m": 0.0474, "wall_h_m": 0.006, "chamfer_h_m": 0.012, "outer_chamfer_w_m": 0.010,
                               "divider_w_m": 0.004, "divider_h_m": 0.010, "ridge_apex_z_m": 0.012, "rim_top_z_m": 0.026, "rim_top_above_floor_m": 0.018,
                               "plate_thickness_m": 0.008, "plate_m": 0.0908,
                               "pocket_centres": [[-h, h], [h, h], [-h, -h], [h, -h]], "outer_corner_sign": [[-1, 1], [1, 1], [-1, -1], [1, -1]]}
        try:
            with open(path) as f:
                raw = json.load(f)
            for k in list(geo):
                if k in raw:
                    geo[k] = raw[k] if isinstance(geo[k], list) else float(raw[k])
        except Exception:  # noqa: BLE001
            pass
        return geo

    def _board_slot_targets(self, board_pose: TT.Pose, right: th.Tensor, toward: th.Tensor) -> Tuple[List[th.Tensor], List[th.Tensor], float]:
        """``(goal xy per slot, release xy per slot, pocket floor z)`` of the board at ``board_pose``: the goal is the pocket
        centre (the seat), the release aim is ``generation.board_release_bias_m`` toward the pocket's outer corner (an inward
        landing error only brings the block back onto its seat, the outer chamfer absorbs the outward one; drop test v5:
        +-6 mm per axis / diagonal and +-2 deg of yaw seat within 1 mm at a 3.5 mm bias)."""
        geo = self._board_geometry()
        far = -toward  # the board's +y
        bias = float(self.gen("board_release_bias_m", 0.0035))
        goals, aims = [], []
        for (cx, cy), (sx, sy) in zip(geo["pocket_centres"], geo["outer_corner_sign"]):
            c_xy = board_pose[0][:2] + right * float(cx) + far * float(cy)
            goals.append(c_xy)
            aims.append(c_xy + right * (bias * float(sx)) + far * (bias * float(sy)))
        floor_z = float(board_pose[0][2]) + float(geo["plate_thickness_m"])
        return goals, aims, floor_z

    def _place_margin(self) -> float:
        """The diagonal release offset away from the placed neighbours (``PLACE_MARGIN``; ``generation.place_margin_m`` - 0 for the
        puzzle board, whose fingers never enter the pocket and whose landing error it would only add to)."""
        return float(self.gen("place_margin_m", PLACE_MARGIN))

    def _board_fingertips_clear(self, clearance: float, placed_neighbours: List[Any]) -> Optional[str]:
        """Puzzle board: the block is released with its bottom ``release_above_surface_m`` over the pocket floor; a finger on the
        side of a seated neighbour (6 mm away) can only pass ABOVE that neighbour's top, so the pinch's fingertips (``tip_clearance``
        above the block's bottom) must clear ``edge`` above the floor by ``board_finger_clear_m``. None when fine, else the reason."""
        if not placed_neighbours:
            return None
        tips = self._release_above() + float(clearance)
        need = self.edge + float(self.gen("board_finger_clear_m", 0.002))
        if tips + 1e-6 < need:
            return f"fingertips {tips * 1000:.1f} mm over the pocket floor, below a seated neighbour's top + margin ({need * 1000:.1f} mm)"
        return None

    def _place_board(self, manip: Stance, right: th.Tensor, toward: th.Tensor, table_yaw: float, cells: Optional[List[th.Tensor]] = None,
                     flip_eligible=None, n_blocks: int = 4, support=None, layout_ok=None) -> Tuple[str, TT.Pose, List[th.Tensor]]:
        """Put the puzzle board on the table, table-aligned (its +x along ``right``, +y away from the robot), at the first of
        ``generation.board_base_xy_candidates`` (base depth, lateral of the manipulation stance) that (a) lies on the top, (b)
        leaves ``n_blocks`` region cells clear of its footprint (``board_cell_margin_m`` beyond half plate + half cube in both
        axes) with at least the expected number of flip cells among them, (c) passes a quick place probe from the manipulation
        stance: a pitched top-down pinch releasing the cube at the biased release aim of the far-left and near-right pockets
        (0920-030951: the board left of the centre line put the slots out of the right arm's reach), and (d) leaves a cell set the
        layout accepts, ``layout_ok(cells kept)`` (owner 2026-10-12: enough flip cells by count is not a flip plan - with
        (0.52, -0.06) the kept cells of s3 c1_k0 had no set whose repeated colours both flip from a cell with a clear approach
        side, and every attempt ended at the layout). Returns ``(bddl name, root pose, the cells kept)``."""
        name = self.bindings.one("board").name
        geo = self._board_geometry()
        half = 0.5 * float(geo["plate_m"])
        q = C.quat_from_yaw(table_yaw)  # +x = right (table_yaw = atan2 of ``right``), +y = -toward
        far = -toward
        half_b = half + 0.5 * self.edge + float(self.gen("board_cell_margin_m", 0.06))
        n_flip_needed = self._n_flips() if self._flips_expected() else 0
        board_obj = self.objects[name]
        why: List[str] = []
        for depth, lateral in self._board_positions():
            tag = f"({float(depth):.2f}, {float(lateral):+.2f})"
            xy = manip.base_xy + manip.forward * float(depth) + manip.left * float(lateral)
            if not self._board_on_top(manip, depth, lateral, half, right, far):
                why.append(f"{tag}: off the top")
                continue
            kept = list(cells or [])
            if cells is not None:
                kept = [c for c in cells if abs(float(th.dot(c - xy, right))) > half_b or abs(float(th.dot(c - xy, toward))) > half_b]
                n_elig = sum(1 for c in kept if flip_eligible is not None and flip_eligible(c))
                if len(kept) < n_blocks or n_elig < n_flip_needed:
                    why.append(f"{tag}: {len(kept)} cells clear ({n_elig} flip-eligible)")
                    continue
            pose = (th.tensor([float(xy[0]), float(xy[1]), float(self.top.z) + 0.001]), q)
            # (c) the place probe: the cube's centre at the release aim, its bottom release_above over the pocket floor
            self._teleport({name: pose})
            _goals, aims, floor_z = self._board_slot_targets(pose, right, toward)
            z_c = floor_z + self._release_above() + 0.5 * self.edge
            probe_ok = True
            for si in (0, len(aims) - 1):
                sl = aims[si]
                c_place = (th.tensor([float(sl[0]), float(sl[1]), z_c]), C.quat_from_yaw(table_yaw))
                found = False
                for pair in ("x", "y"):
                    for tilt in (30.0, 45.0, 0.0):
                        try:
                            g = C.cube_grasp(c_place, pair, tilt, table_yaw, edge=self.edge)
                        except ValueError:
                            continue
                        if not g.downward:
                            continue
                        r = self.ikc.exact([C.pre_pose(g.eef), g.eef], manip, "right", ignore_for_last=[support] if support is not None else [], cache=False)
                        if r.ok:
                            found = True
                            break
                    if found:
                        break
                if not found:
                    probe_ok = False
                    break
            if not probe_ok:
                why.append(f"{tag}: no pitched place reaches the pockets' release aims from the manipulation stance")
                continue
            if layout_ok is not None and cells is not None and not layout_ok(kept):
                why.append(f"{tag}: {len(kept)} cells clear, no cell set of them lets every repeated-colour block flip")
                continue
            log.info("%s: puzzle board %s at base (%.2f, %+.2f): pitch %.1f mm, seat %.1f mm, rim %.0f mm over the floor, release bias %.1f mm; %d cells clear (%s)",
                     self.gp.alias, name, float(depth), float(lateral), float(geo["pocket_pitch_m"]) * 1000, float(geo["seat_inner_m"]) * 1000,
                     float(geo["rim_top_above_floor_m"]) * 1000, float(self.gen("board_release_bias_m", 0.0035)) * 1000, len(kept), "; ".join(why) or "first candidate")
            return name, pose, kept
        self.world.park(board_obj, list(self.objects).index(name))
        raise Reject("board_placement", "no board position of generation.board_base_xy_candidates works: " + "; ".join(why))

    def _bddl_names(self, objs: List[Any]) -> List[str]:
        """bddl instance names of sim objects (``self.objects`` inverted)."""
        inv = {id(o): n for n, o in self.objects.items()}
        return [inv[id(o)] for o in objs if id(o) in inv]

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
        # the face touched has the outward normal -direction in BOTH cases: a push along d presses the face the block moves away
        # from (outward normal -d), a brace pressing along d presses into the face with outward normal -d. (Until 2026-09-19 the
        # push branch used +d - the pad started on the far side of the block and moved away from it; the per-block tails never
        # ran in a demo so it went unnoticed.) ``into`` is kept for the callers' readability.
        face_normal = -dxy  # outward normal of the face touched
        point = centre + face_normal * (0.5 * self.edge) + th.tensor([0.0, 0.0, 0.25 * self.edge])
        a = -face_normal * math.cos(math.radians(CONTACT_TILT_DEG)) + th.tensor([0.0, 0.0, -math.sin(math.radians(CONTACT_TILT_DEG))])
        return point, a / th.linalg.norm(a)

    def _blade_height(self) -> float:
        """eef origin height of a top-down blade beside a block: the closed fingertips 3 mm above the top (a 2.5 cm cube
        sits entirely below the wrist of a horizontal contact)."""
        floor = float(getattr(self, "_push_floor_z", None) or self.top.z)  # the board's pocket floor during its pack pushes
        z = floor + max(0.6 * self.edge, C.FINGERTIP_LENGTH + 0.003)
        # A/B knob generation.contact_height_extra_m (D2 core helper, 0 = no-op): the fingertip stays >= 1 cm below the block's top
        return lift_contact_height(self, z, "blade", face_top_z=floor + float(self.edge))

    def _blade_poses(self, pose: TT.Pose, direction_xy: th.Tensor, into: bool, manip: Stance, prefix: str):
        """Top-down blade contacts of the closed gripper beside a block face (outward normal ``-direction_xy`` for a
        push, ``+direction_xy`` for a brace): the eef origin half a finger outside the face at the blade height,
        approach straight down / pitched 30° / 45° / 60° toward the robot, closing axis (the blade) along the face
        normal with either wrist turn. Yields ``(pose, approach, name)``."""
        dxy = th.tensor([float(direction_xy[0]), float(direction_xy[1]), 0.0])
        dxy = dxy / th.linalg.norm(dxy)
        face_normal = -dxy  # outward normal of the face touched (push: the face the block moves away from; brace: the face pressed into)
        centre = self._centre(pose)[0]
        yield from self._blade_poses_at(centre, face_normal, manip, prefix)

    def _blade_poses_at(self, centre: th.Tensor, face_normal: th.Tensor, manip: Stance, prefix: str, modes: Tuple[str, ...] = ("pad", "side")):
        """The blade contacts of :meth:`_blade_poses` about an arbitrary centre point (the mid-point of two braced cubes for the
        packing tail's wide brace) and a face outward normal; ``modes`` restricts the closing-axis classes."""
        z_blade = self._blade_height()
        side = th.cross(th.tensor([0.0, 0.0, 1.0]), face_normal, dim=0)
        floor = float(getattr(self, "_push_floor_z", None) or self.top.z)
        face_z = (floor, floor + self.edge)
        # closing axis along the face normal (the pad face pushes) or across it (the finger's side pushes): a top-down
        # pose with the closing axis lateral to the robot is infeasible at many cells (wrist limits), so both are tried.
        # The eef origin stands the closed fingers' half thickness (measured profile over the cube face above the
        # fingertips; pad face or finger side) outside the face — a contact, no clearance
        for pitch in (0.0, 30.0):  # the cubes sit 0.5–0.8 m ahead: the 45° / 60° classes are for the far cells
            a_vec = TT.pitched_approach(pitch, manip.forward)
            z_tip = z_blade - C.FINGERTIP_LENGTH * math.cos(math.radians(pitch))
            for c_axis, cname in ((face_normal, "pad"), (side, "side")):
                if cname not in modes:
                    continue
                so = TT.blade_standoff(face_z, z_tip, pitch, 0.0, side=cname == "side")
                if so is None:
                    continue
                origin = th.tensor([float(centre[0]), float(centre[1]), z_blade]) + face_normal * (0.5 * self.edge + so[1])
                for c_sign in (1.0, -1.0):
                    yield (origin, C.eef_quat(a_vec, c_sign * c_axis)), a_vec, f"{prefix}_blade_{cname}_{pitch:.0f}{'' if c_sign > 0 else '-'}", so[1]

    # ------------------------------------------------------------------ packing tail (owner 2026-09-19 21:47)

    def _pack_tail(self, manip: Stance, current: Stance, pool: List[Stance], names: List[str], records: List[Dict[str, Any]], slot_of: Dict[int, int], n_cols: int,
                   right: th.Tensor, toward: th.Tensor, gap_delta: float) -> Tuple[Dict[str, Any], Stance]:
        """After the last place (every block on the loose grid, ``records[].goal_pose``) the blocks are packed: stage 1 braces the
        LEFT column with the left gripper opened so a finger side rests on the left face of each of its cubes and pushes every
        other column's cube against it with the right hand (closing the column gaps, the columns end aligned); stage 2 braces the
        FAR row on its far faces and pushes the near row away from the robot (the near-row brace with a pull toward the robot
        is the fallback - the log says which one passed). Every stage is one wide brace + one push per cube, all from the
        manipulation stance, checked as two-arm exact IK (brace held while the push arm moves). Raises ``Reject`` when no stage
        passes; returns the ``sidecar.pack`` section."""
        support = self.obj("support")
        world0: Dict[str, TT.Pose] = {}
        for rec in records:
            gp = rec["goal_pose"]
            world0[rec["instance"]] = (th.tensor([float(v) for v in gp["position"]]), th.tensor([float(v) for v in gp["orientation"]]))
        by_slot = {si: names[bi] for bi, si in slot_of.items()}
        n_rows = max(1, len(slot_of) // n_cols)
        braced = [by_slot[r * n_cols] for r in range(n_rows) if r * n_cols in by_slot]
        pushes = [(by_slot[r * n_cols + c], c * gap_delta) for r in range(n_rows) for c in range(1, n_cols) if r * n_cols + c in by_slot]
        far = [by_slot[c] for c in range(n_cols) if c in by_slot]
        near_pushes = [(by_slot[r * n_cols + c], r * gap_delta) for r in range(1, n_rows) for c in range(n_cols) if r * n_cols + c in by_slot]
        near = [by_slot[(n_rows - 1) * n_cols + c] for c in range(n_cols) if (n_rows - 1) * n_cols + c in by_slot]
        far_pushes = [(by_slot[r * n_cols + c], (n_rows - 1 - r) * gap_delta) for r in range(n_rows - 1) for c in range(n_cols) if r * n_cols + c in by_slot]
        # the stances tried: where the robot stands after the last block, the manipulation stance, then the pool ranked about the
        # grid centre (pod run 0919-220203: from the tall manipulation stance every top-down push pre-pose 0.48 m ahead failed the IK
        # with the left arm braced across - the grid needs to sit in the arms' sweet zone, 0.55-0.65 m ahead)
        grid_c = th.stack([self._centre(p_)[0] for p_ in world0.values()]).mean(dim=0)
        grid_root = (grid_c, C.IDENTITY_QUAT.clone())
        ranked = self._rank_stances(pool, current, grid_root, grid_root, False)
        stances = [current] + [manip] + [st for st in ranked if st is not current and st is not manip]
        if self._has_board():
            raise Reject("spec", "the puzzle board seats the blocks in its pockets - generation.block_pack_tail must be false for the board alias")
        seen = set()
        tried: List[str] = []
        for st in stances[: int(self.gen("pack_max_stances", 6))]:
            key = (round(float(st.base_pose2d[0]), 3), round(float(st.base_pose2d[1]), 3), round(float(st.translate), 3), round(float(st.tilt), 3))
            if key in seen:
                continue
            seen.add(key)
            world = dict(world0)
            self._teleport(world)
            tag_st = f"{st.name} (trunk {st.translate:.2f}/{st.tilt:+.2f}, so {float(st.stand_off or 0.0):.2f}, shift {float(st.extras.get('block_stance_shift_m', 0.0)):+.2f})"
            # stage 1: the columns. Braced = column 0 (the robot's left) by the left hand, pushed = the cubes of the other columns by
            # the right hand, each by its column's share of the gap (2 x 2: the right column by gap_delta) toward the left; the
            # mirrored assignment (right column braced by the right hand, the left column pushed right by the left hand) is the
            # fallback - the goal is judged on the layout relations, a uniform shift of the packed grid does not count
            right_col = [by_slot[r * n_cols + n_cols - 1] for r in range(n_rows) if r * n_cols + n_cols - 1 in by_slot]
            left_pushes = [(by_slot[r * n_cols + c], (n_cols - 1 - c) * gap_delta) for r in range(n_rows) for c in range(n_cols - 1) if r * n_cols + c in by_slot]
            st1, why1 = self._pack_stage("columns", world, braced, pushes, -right, right, st, support)
            moved1 = [(nm, -_xy3(right) * dist) for nm, dist in pushes]
            if st1 is None and bool(self.gen("pack_mirror_fallback", True)):
                st1, why1b = self._pack_stage("columns_mirrored", world, right_col, left_pushes, right, -right, st, support, brace_arm="right", push_arm="left")
                moved1 = [(nm, _xy3(right) * dist) for nm, dist in left_pushes]
                why1 = f"{why1}; mirrored: {why1b}"
            if st1 is None:
                tried.append(f"{tag_st}: columns: {why1}")
                log.info("%s: pack @ %s: columns failed: %s", self.gp.alias, tag_st, why1[:700])
                continue
            for nm, dxy in moved1:  # the world after stage 1: the pushed cubes moved
                world[nm] = (world[nm][0] + dxy, world[nm][1])
            self._teleport(world)
            # stage 2: the rows. Far row braced on its far faces, the near rows pushed away from the robot; else the near row
            # braced on its near faces and the far rows pulled toward the robot
            st2, why2 = self._pack_stage("rows_far_brace", world, far, near_pushes, -toward, toward, st, support)
            which = "far row braced (left hand), near row pushed away from the robot (right hand)"
            if st2 is None:
                st2, why2b = self._pack_stage("rows_near_brace", world, near, far_pushes, toward, -toward, st, support)
                which = "near row braced (left hand), far row pulled toward the robot (right hand)"
                why2 = f"far brace: {why2}; near brace: {why2b}"
            if st2 is None and bool(self.gen("pack_mirror_fallback", True)):
                st2, why2c = self._pack_stage("rows_far_brace_mirrored", world, far, near_pushes, -toward, toward, st, support, brace_arm="right", push_arm="left")
                which = "far row braced (right hand), near row pushed away (left hand)"
                if st2 is None:
                    st2, why2d = self._pack_stage("rows_near_brace_mirrored", world, near, far_pushes, toward, -toward, st, support, brace_arm="right", push_arm="left")
                    which = "near row braced (right hand), far row pulled (left hand)"
                    why2 = f"{why2}; mirrored far: {why2c}; mirrored near: {why2d}"
            if st2 is None:
                tried.append(f"{tag_st}: rows: {why2}")
                log.info("%s: pack @ %s: columns ok (%s), rows failed: %s", self.gp.alias, tag_st, st1["brace"]["pose_class"] if st1.get("brace") else "push only", why2[:900])
                continue
            log.info("%s: packing tail from %s: columns %s; rows: %s (%s)", self.gp.alias, tag_st, st1["brace"]["pose_class"] if st1.get("brace") else "push only", which,
                     st2["brace"]["pose_class"] if st2.get("brace") else "push only")
            self._teleport(world0)
            return {"stance": st.name, "gap_delta_m": float(gap_delta), "stages": [st1, st2], "rows_brace": "far" if which.startswith("far") else "near"}, st
        if bool(self.gen("pack_allow_push_only", True)):
            # no stance takes the two-arm brace + push: the pushes alone (the push is position-controlled to the packed pose - the
            # cube travels the gap and meets its neighbour with no travel left, so the neighbour is not carried along)
            for st in self._pack_posture_variants(stances[: int(self.gen("pack_max_stances", 6))]):
                world = dict(world0)
                self._teleport(world)
                tag_st = f"{st.name} (trunk {st.translate:.2f}/{st.tilt:+.2f}, so {float(st.stand_off or 0.0):.2f}, shift {float(st.extras.get('block_stance_shift_m', 0.0)):+.2f})"
                st1, why1 = self._pack_stage("columns_push_only", world, [], pushes, -right, right, st, support)
                if st1 is None:
                    log.info("%s: pack (push only) @ %s: columns failed: %s", self.gp.alias, tag_st, why1[:500])
                    continue
                for nm, dist in pushes:
                    world[nm] = (world[nm][0] - _xy3(right) * dist, world[nm][1])
                self._teleport(world)
                st2, why2 = self._pack_stage("rows_push_only", world, [], near_pushes, -toward, toward, st, support)
                which = "near row pushed away from the robot, no brace"
                if st2 is None:
                    st2, why2b = self._pack_stage("rows_push_only_pull", world, [], far_pushes, toward, -toward, st, support)
                    which = "far row pulled toward the robot, no brace"
                    if st2 is None:
                        log.info("%s: pack (push only) @ %s: rows failed: %s; pull: %s", self.gp.alias, tag_st, why2[:400], why2b[:400])
                        continue
                log.info("%s: packing tail (push only) from %s: %s", self.gp.alias, tag_st, which)
                self._teleport(world0)
                return {"stance": st.name, "gap_delta_m": float(gap_delta), "stages": [st1, st2], "rows_brace": "none", "push_only": True}, st
        self._teleport(world0)
        raise Reject("pack_tail", "no stance packs the grid: " + " | ".join(tried)[:1500])

    def _pack_posture_variants(self, stances: List[Stance]) -> List[Stance]:
        """Each candidate pack stance as it is, then clones with the idle LEFT arm in the ``ready`` / ``tucked`` posture (a new
        name, so the block chains' stance records stay untouched and the oracle drives the arm there at the pack transition):
        the pushes that reach across the robot's centre line (the board left of it, 0920-011816) collide with the hanging
        untucked left arm."""
        out: List[Stance] = []
        postures = list(self.gen("pack_left_postures", ["ready", "tucked"]) or [])
        for st in stances:
            out.append(st)
            for name in postures:
                try:
                    vec = self.kit.arm_q(name, "left")
                except Exception:  # noqa: BLE001
                    continue
                st2 = copy.copy(st)
                st2.extras = dict(st.extras, pack_left_posture=name)
                st2.q = self.ikc.q_locked(st, left_arm=vec)
                st2.name = f"{st.name}_pack_{name}"
                if not bool(self.kit.collision_free(st2.q, margin=0.01)[0]):
                    continue
                out.append(st2)
        return out

    def _pack_stage(self, tag: str, world: Dict[str, TT.Pose], braced: List[str], pushes: List[Tuple[str, float]], push_dir: th.Tensor, brace_dir: th.Tensor,
                    manip: Stance, support, brace_arm: Optional[str] = None, push_arm: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
        """One packing stage: the wide brace (left arm; the fingers opened so a finger SIDE rests on the face of each braced cube,
        the closing axis along the braced line) held while the right hand pushes each ``(cube, distance)`` along ``push_dir`` with
        the closed blade. Returns ``(stage dict, "")`` or ``(None, reason)``."""
        if not pushes:
            return None, "nothing to push"
        brace_arm = str(brace_arm or self.gen("pack_brace_arm", "left"))
        push_arm = str(push_arm or self.gen("pack_push_arm", "right"))
        if not braced:
            # push only: the pushes from the stance posture, no second arm
            push_recs, why = self._pack_pushes(world, pushes, push_dir, manip, support, push_arm, self.ikc.q_locked(manip, fingers=0.0).clone(), [], "no brace")
            if push_recs is None:
                return None, why
            return {"tag": tag, "brace": None, "pushes": push_recs}, ""
        centres = [self._centre(world[nm])[0] for nm in braced]
        mid = th.stack(centres).mean(dim=0)
        spread = float(th.linalg.norm(centres[0][:2] - centres[-1][:2])) if len(centres) > 1 else 0.0
        # per-finger opening: the finger centre line (pad face + half the 8.4 mm finger) on each cube's centre
        q_open = max(0.0, min(float(C.FINGER_OPEN), 0.5 * spread - 0.5 * FINGER_HALF_THICKNESS)) if spread > 0 else 0.0
        b_dxy = th.tensor([float(brace_dir[0]), float(brace_dir[1]), 0.0])
        b_dxy = b_dxy / th.linalg.norm(b_dxy)
        braced_objs = [self.objects[nm] for nm in braced]
        brace_fails: List[str] = []
        push_fails: List[str] = []
        idx_b = self.kit.robot.arm_control_idx[brace_arm]
        pushes_of: Dict[int, List[Dict[str, Any]]] = {}

        def eval_brace(p):
            r_b = self.ikc.exact(list(p.chain), manip, brace_arm, ignore_for_last=[support] + braced_objs, cache=False, fingers=q_open, q_first=p.q_first)
            if not r_b.ok:
                brace_fails.append(f"{p.label}: {r_b.reason}")
            return bool(r_b.ok), r_b, None, str(r_b.reason)

        def pushes_with(sc):
            # the pushes, one per cube, each checked with the brace arm locked at its contact
            q_seed = self.ikc.q_locked(manip, fingers=0.0).clone()
            q_seed[idx_b] = sc.result.q[idx_b]
            push_recs, why_p = self._pack_pushes(world, pushes, push_dir, manip, support, push_arm, q_seed, braced_objs, sc.proposal.label)
            if push_recs is None:
                push_fails.append(why_p)
                return False, why_p
            pushes_of[id(sc)] = push_recs
            return True, ""

        # a blade class's two wrist turns by the orientation rule (owner 2026-09-27); the other turn when the pushes fail with the first
        chosen_b, dec_b, _fails_b = O.choose_first(self.ikc, brace_arm, self._blade_poses_at(mid, -b_dxy, manip, "pack_brace", modes=("side",)), key=lambda c: c[0],
                                                   label=lambda c: c[2], chain=lambda c: [C.pre_pose(c[0]), c[0]], evaluate=eval_brace,
                                                   q_start=self.ikc.q_locked(manip, fingers=0.0), stage="pack brace", downstream=pushes_with)
        if chosen_b is not None:
            b_pose, b_a, b_name, b_half = chosen_b.proposal.payload
            b_pre = chosen_b.proposal.chain[0]
            push_recs = pushes_of[id(chosen_b)]
            brace_rec = annotate_height_extra(TT.BraceRecord("brace", brace_arm, b_pre, b_pose, direction=b_a, brace_kind=f"pack_{tag}", object=braced[0],
                                                             extras={"pose_class": b_name, "blade_half_thickness_m": None if b_half is None else float(b_half), "stop_clearance_m": 0.0,
                                                                     "fingers_opening_m": round(float(q_open), 4), "stop_width_m": round(float(spread), 4),
                                                                     "braced": list(braced),
                                                                     # the program check removes BOTH braced cubes at the contact (a finger side on each), the support throughout
                                                                     "ignore_last": list(braced), "ignore_every": ["supports"],
                                                                     **_orient_extra(dec_b)}), self).to_sidecar()
            return {"tag": tag, "brace": brace_rec, "pushes": push_recs}, ""
        why = (sorted(set(push_fails)) or sorted(set(brace_fails)))[:3]
        return None, f"no brace + pushes pass ({len(brace_fails)} brace, {len(push_fails)} push failures; e.g. {why})"

    def _pack_pushes(self, world: Dict[str, TT.Pose], pushes: List[Tuple[str, float]], push_dir: th.Tensor, manip: Stance, support, push_arm: str, q_seed: th.Tensor,
                     braced_objs: List[Any], b_name: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        """The packing pushes of one stage, one per ``(cube, distance)``: the first blade / tilted class whose pre + init (the cube
        removed) and pre + goal pass the exact IK from ``q_seed`` (the other arm where it is - at its brace contact, or the stance
        posture) and whose contact configuration is collision-free together with that arm. Returns ``(records, "")`` or
        ``(None, reason)``."""
        idx_other = self.kit.robot.arm_control_idx["left" if push_arm == "right" else "right"]
        fails: List[str] = []
        recs: List[Dict[str, Any]] = []
        d3 = th.tensor([float(push_dir[0]), float(push_dir[1]), 0.0])
        d3 = d3 / th.linalg.norm(d3)
        standoff = float(self.gen("pack_push_standoff_m", 0.005))
        for nm, dist in pushes:
            found = None
            # the cubes the push drives ahead of the pushed one (in line along the push direction within its travel): they move
            # with it, so they are out of the blade's way at the goal - ignored for the goal / chain checks and recorded as
            # ``driven`` for the program check (the wall pushes of the board variant drive the inner column / row along)
            c_nm = self._centre(world[nm])[0][:2]
            driven: List[str] = []
            for other in world:
                if other == nm:
                    continue
                rel = self._centre(world[other])[0][:2] - c_nm
                along = float(th.dot(rel, d3[:2]))
                across = float(th.linalg.norm(rel - along * d3[:2]))
                if 0.0 < along < float(dist) + self.edge and across < 0.6 * self.edge:
                    driven.append(other)
            driven_objs = [self.objects[o] for o in driven]
            cands = list(self._blade_poses(world[nm], push_dir, False, manip, "pack_push"))
            t_pt, t_a = self._face_contact(world[nm], push_dir, into=False)
            c_axis_t = th.tensor([float(-push_dir[1]), float(push_dir[0]), 0.0])  # pads close across the push direction
            cands.append((C.contact_eef(t_pt, t_a, c_axis_t), t_a, "pack_push_tilted", None))

            def evaluate(p, nm=nm, dist=dist, driven=driven, driven_objs=driven_objs):
                p_face, p_a, p_name, p_half = p.payload
                # the blade starts ``pack_push_standoff_m`` short of the face (a cube placed a few mm toward the blade is met, not
                # landed on) and travels that plus the gap; the neighbour stops the cube at the packed position
                p_init = (p_face[0] - d3 * standoff, p_face[1])
                p_goal = (p_face[0] + d3 * float(dist), p_face[1])
                p_pre = C.pre_pose(p_init)
                r_i = self.ikc.exact([p_pre, p_init], manip, push_arm, ignore_for_last=[support, self.objects[nm]], q_seed=q_seed, cache=False, fingers=0.0,
                                     q_first=p.q_first)
                if not r_i.ok:
                    fails.append(f"{b_name}/{nm}/{p_name}: init {r_i.reason}")
                    return False, None, None, f"init {r_i.reason}"
                qs = O.chain_qs(self.ikc, push_arm, [p_pre, p_init])
                r_g = self.ikc.exact([p_pre, p_goal], manip, push_arm, ignore_for_last=[support, self.objects[nm]] + braced_objs + driven_objs, q_seed=q_seed, cache=False, fingers=0.0)
                if not r_g.ok:
                    fails.append(f"{b_name}/{nm}/{p_name}: goal {r_g.reason}")
                    return False, None, None, f"goal {r_g.reason}"
                # the program check's own verdict on the push (``check_action_targets``: waypoints pre -> init -> goal with the pushed
                # cube and the support removed throughout, closed pads): run it here so the sampler never records a push the check
                # rejects (0920-055948: 'push: waypoints: exact IK failed at pose 1 of 3' on a board push the two exact checks passed)
                # the oracle's motion: arm_to(pre), then arm_along over ``actions._line(init, goal)`` - the start excluded, so the
                # first tracked segment runs from the pre to a third of the push (the program check verifies this very chain)
                line = [(pt, p_goal[1]) for pt in C.line_waypoints(p_init[0], p_goal[0], 3)]
                r_w = self.ikc.waypoints_ok([p_pre] + line, manip, push_arm, ignore_all=[support, self.objects[nm]] + braced_objs + driven_objs, q_seed=q_seed,
                                            cache=False, fingers=0.0, approached=[self.objects[nm]])
                if not r_w.ok:
                    fails.append(f"{b_name}/{nm}/{p_name}: chain {r_w.reason}")
                    return False, None, None, f"chain {r_w.reason}"
                # the oracle's retreat from a push (``execute_retreat``, contact_retreat_escape_m 0.05): the pad leaves RADIALLY -
                # opposite the push - by ``Retreat.escape_far`` (5 cm for the pack retreats; the default 10 cm left the arm 14 mm
                # short after the board's rightward wall push, 0920-192950 action 33) and up by the same, reached by ``arm_to`` at
                # a 5 mm tolerance - the end pose must have a solution polished to that tolerance
                esc_far = float(self.gen("pack_retreat_dist_m", 0.05))  # = Retreat.escape_far of the program's pack retreats
                r_end = (p_goal[0] - d3 * esc_far + th.tensor([0.0, 0.0, min(0.05, esc_far)]), p_goal[1])
                r_r = self.ikc.exact([r_end], manip, push_arm, ignore_for_last=[support, self.objects[nm]] + braced_objs + driven_objs, q_seed=r_w.q, cache=False, fingers=0.0)
                if not r_r.ok or r_r.q is None:
                    fails.append(f"{b_name}/{nm}/{p_name}: retreat end {r_r.reason}")
                    return False, None, None, f"retreat end {r_r.reason}"
                with self.ikc.ignoring([support, self.objects[nm]] + braced_objs + driven_objs):
                    _q_pol, res_r = self.ikc.polish_witness(r_r.q, r_end, push_arm, ignore=[])
                if res_r > 0.005:
                    fails.append(f"{b_name}/{nm}/{p_name}: retreat end {res_r * 1000:.1f} mm beyond the arm's reach")
                    return False, None, None, f"retreat end {res_r * 1000:.1f} mm beyond the arm's reach"
                q_both = r_g.q.clone()
                q_both[idx_other] = q_seed[idx_other]
                # with a brace the two contact configurations are checked together at 5 mm; a single-handed push only needs its own
                # (collision-aware) IK - the 5 mm margin against the idle other arm rejected every row push of 0919-233714
                with self.ikc.ignoring([support, self.objects[nm]] + braced_objs):
                    free = bool(self.kit.collision_free(q_both, margin=0.005 if braced_objs else 0.0)[0])
                if not free:
                    fails.append(f"{b_name}/{nm}/{p_name}: the two arms collide at the contacts")
                    return False, None, None, "the two arms collide at the contacts"
                rec = annotate_height_extra(ContactRecord("push", push_arm, p_pre, p_init, None, p_goal, d3, float(dist) + standoff, object=nm), self).to_sidecar()
                rec["pose_class"] = p_name
                rec["standoff_m"] = standoff
                if driven:
                    rec["driven"] = list(driven)
                # the chain's pre configuration as the program check's / oracle's IK seed (``q_hint``, the press mechanism of
                # articulation_small): the branch the sampler's chain check passed in
                q_hint = self.ikc.solutions.get((push_arm, self.ikc._pose_key(p_pre)))
                if q_hint is not None:
                    rec["q_hint"] = [round(float(v), 5) for v in q_hint.tolist()]
                if p_half is not None:
                    rec["blade_half_thickness_m"] = float(p_half)
                return True, rec, (qs or []) + [r_w.q, r_r.q], ""

            # a class's two wrist turns by the orientation rule (owner 2026-09-27); the classes in their order
            chosen_p, dec_p, _fails_p = O.choose_first(self.ikc, push_arm, cands, key=lambda c: c[0], label=lambda c: c[2],
                                                       chain=lambda c: [C.pre_pose((c[0][0] - d3 * standoff, c[0][1])), (c[0][0] - d3 * standoff, c[0][1])],
                                                       evaluate=evaluate, q_start=q_seed, camera_idx=[1], stage="pack push")
            if chosen_p is not None:
                found = dict(chosen_p.result, **_orient_extra(dec_p))
            if found is None:
                return None, f"{nm}: no push class passes ({len(fails)} failures; e.g. {sorted(set(fails))[:3]})"
            recs.append(found)
        return recs, ""

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
        brace_of: Dict[int, Any] = {}

        def eval_push(p):
            # the push init / goal touch the block: init is exact-IK-checked with the block ignored, then pre + goal with the brace arm
            r_init = self.ikc.exact(list(p.chain), manip, "right", ignore_for_last=[support, self.objects[pushed]], cache=False, q_first=p.q_first)
            if not r_init.ok:
                push_fails.append(f"{p.label}: {r_init.reason}")
            return bool(r_init.ok), r_init, None, str(r_init.reason)

        def brace_with(sc):
            push_pre, push_init = sc.proposal.chain
            push_goal = (push_init[0] + d3 * dist, push_init[1])

            def eval_brace(p):
                res = TT.brace_second_arm(self, manip, "right", [push_pre, push_goal], "left", list(p.chain), ignore_work_last=[support, self.objects[pushed]],
                                          ignore_brace_last=[support, self.objects[braced]], q_first_brace=p.q_first)
                if not res.ok:
                    brace_fails.append(f"{sc.proposal.label}+{p.label}: {res.reason}")
                return bool(res.ok), res, None, str(res.reason)

            b_chosen, b_dec, _b_fails = O.choose_first(self.ikc, "left", brace_cands, key=lambda c: c[0], label=lambda c: c[2], chain=lambda c: [C.pre_pose(c[0]), c[0]],
                                                       evaluate=eval_brace, q_start=self.ikc.q_locked(manip), stage=f"{kind} brace")
            if b_chosen is None:
                return False, "no brace joins it"
            brace_of[id(sc)] = (b_chosen, b_dec, push_goal)
            return True, ""

        # a class's two wrist turns by the orientation rule (owner 2026-09-27), for the push and for the brace joining it; the push's
        # other turn when no brace joins the first
        p_chosen, p_dec, _p_fails = O.choose_first(self.ikc, "right", push_cands, key=lambda c: c[0], label=lambda c: c[2], chain=lambda c: [C.pre_pose(c[0]), c[0]],
                                                   evaluate=eval_push, q_start=self.ikc.q_locked(manip), camera_idx=[1], stage=f"{kind} push", downstream=brace_with)
        if p_chosen is not None:
            b_chosen, b_dec, push_goal = brace_of[id(p_chosen)]
            push_init, _push_a, p_name, p_half = p_chosen.proposal.payload
            b, b_a_vec, b_name, b_half = b_chosen.proposal.payload
            chosen = (p_chosen.proposal.chain[0], push_init, push_goal, b_chosen.proposal.chain[0], b, b_a_vec, f"{p_name}+{b_name}", p_half, b_half)
            p_orient, b_orient = _orient_extra(p_dec), _orient_extra(b_dec)
        if chosen is None:
            log.info("%s: %s brace / push of %s skipped: %s", self.gp.alias, kind, pushed, (sorted(set(brace_fails)) or sorted(set(push_fails)))[:2])
            return "brace_ik_failed" if brace_fails else "push_ik_failed"
        push_pre, push_init, push_goal, brace_pre, brace, brace_a, used, p_half, b_half = chosen
        rec[f"push_{kind}"] = annotate_height_extra(ContactRecord("push", "right", push_pre, push_init, None, push_goal, d3, dist, object=pushed), self).to_sidecar()
        rec[f"push_{kind}"].update(p_orient)
        if p_half is not None:
            rec[f"push_{kind}"]["blade_half_thickness_m"] = float(p_half)
        rec[f"brace_{kind}"] = annotate_height_extra(TT.BraceRecord("brace", "left", brace_pre, brace, direction=brace_a, brace_kind="row_left" if kind == "row" else "column_top",
                                              object=braced, extras={"pose_class": used, "blade_half_thickness_m": None if b_half is None else float(b_half),
                                                                     "stop_clearance_m": 0.0, **b_orient}), self).to_sidecar()
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
        # the goal observation is taken from the stance the robot ends at: the last block's (owner 2026-09-19 per-block stances)
        last = None
        if self.sidecar.blocks:
            b = self.sidecar.blocks[-1]
            last = ((b.get("table_flip") or {}).get("regrasp") or {}).get("stance") or b.get("stance")
        st = self.sidecar.stances.get(last or "manipulation") or self.sidecar.stance("manipulation")
        self.sidecar.set_goal(st.base_pose2d, st.translate, st.tilt, arms="same")

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
