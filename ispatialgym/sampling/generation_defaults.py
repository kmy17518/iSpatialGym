"""Per-subtask ``generation`` seeds of the task customs list (``docs/instance_sampling_spec.md`` §2.3).

``ground_isg`` creates a customs entry per grounded alias; its ``generation`` block is copied from the existing entry
when the alias was grounded before, otherwise from :data:`DEFAULTS` here. **Every number is the one written in the
spec** — the section is cited next to each value — so that no sampler holds a default of its own (§2.3 "There are no
defaults hidden in the samplers"). Numbers the spec marks ``[pilot]`` (§10) are flagged ``# [pilot]``: the pilot
(TODO E1/E2) verifies or replaces them, here.

Shape (§2.3 example)::

    {"templates_max": 20, "instances_per_template": 5,
     "tolerances": {"position": 0.03, "orientation": 0.20},
     "groups": {"<group role>": {"size": 4 | [1, 3], "draw": "same_model" | "unique"}},
     "path_buckets": null | [{"max_m": 3, "n": 10}, ...],
     ... task-level numbers of §5–§6 ...}

``tolerances`` is flat (``position`` m, ``orientation`` rad, joint tolerances ``revolute`` rad / ``prismatic`` m with
the relative ``joint_fraction``, the depth floor ``depth_min_m`` / relative ``depth_fraction``) because the TRO writer
pins one number per metric; the ``settled_override_*`` keys carry the "goal := settled pose" threshold of §8.

Shared rules that are not per-alias parameters (the tabletop stance scan of §4, the Prompt-0 lattice of §5.2, the IK
seed counts of §3.6, the camera model of §3.4) belong to the sampler core / conventions module and are deliberately
not repeated here.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence

#: §2.3 example: template and instance counts (``sample_isg`` N = min(#valid model combinations, templates_max)).
COMMON: Dict[str, Any] = {
    "templates_max": 20,  # §2.3
    "instances_per_template": 5,  # §2.3
}

#: §8 "disembedding, IORS | measured only | 0.03 m / 0.20 rad; override threshold for goal := settled pose =
#: max(0.01 m, 0.25 × smallest horizontal extent) / 0.20 rad".
TOLERANCES_MEASURED: Dict[str, float] = {
    "position": 0.03,  # §8
    "orientation": 0.20,  # §8
    "settled_override_position_min_m": 0.01,  # §8 max(0.01 m, 0.25 × smallest horizontal extent)
    "settled_override_position_extent_fraction": 0.25,  # §8
    "settled_override_orientation": 0.20,  # §8
}

#: §8 articulation: max(0.05 rad, 0.10·|q_goal − q_init|) revolute, max(0.01 m, 0.10·|Δq|) prismatic; the small-scale
#: root pose is measured only at 0.03 / 0.20 (§6.7).
TOLERANCES_ARTICULATION: Dict[str, float] = {
    "revolute": 0.05,  # §8 rad floor
    "prismatic": 0.01,  # §8 m floor
    "joint_fraction": 0.10,  # §8 relative part of both joint tolerances
}

#: §5.3 joint states / handle rules shared by the four articulation subtasks.
ARTICULATION_COMMON: Dict[str, Any] = {
    "open_fraction_range": [0.5, 0.95],  # §5.3 stable set S: fractions of the joint range
    "stable_set_fractions": 7,  # §5.3 "7 fractions of open_fraction_range + the upper limit"
    "stable_hold_steps": 120,  # §5.3 each candidate held 120 physics steps
    "stable_drift_tolerance_fraction": 0.5,  # §5.3 kept when it stays within ½ tolerance
    "stable_probe_tolerances": {"revolute": 0.10, "prismatic": 0.02},  # §5.3 (revolute 0.10 rad, prismatic 0.02 m)
    "waypoint_fractions": [0.25, 0.5, 0.75, 1.0],  # §5.3 q_k = q_init + k·(q_goal − q_init)
    "require_goal_handle": "auto",  # §2.3 / §5.3 auto: q_goal required unless the link reaches it passively; true / false force
    "arm": "auto",  # §2.3 / §5.3 handle side of the face centre (> 0.10 m) decides, else right; overrides
    "arm_side_threshold_m": 0.10,  # §5.3
    "handle_eef_offset_m": 0.012,  # §3.6 / §5.3 eef origin = contact − 0.012·approach
    "pre_grasp_back_m": 0.10,  # §3.6 pre-grasp 0.10 m back along −approach
    "pinch_width_range_m": [0.015, 0.080],  # §5.3 pinch width after scaling
}

#: §5.5 wall alignment (world vertical / horizontal) shared numbers.
ALIGNMENT_WALL_COMMON: Dict[str, Any] = {
    "hang_height_m": 1.45,  # §5.5 centre at height 1.45 m
    "wall_offset_extra_m": 0.01,  # §5.5 thickness/2 + 0.01 off the wall
    "deviation_range_deg": [15.0, 25.0],  # §5.5 δ = sign · U(15°, 25°) (owner decision 2026-09-16 A8; was U(15°, 45°))
    "max_deviation_deg": 25.0,  # §5.5 A8: |δ| > 25° is beyond the one-stroke push (two-stroke push = later, D list)
    "stance_arm_posture": "tucked",  # §5.5 A6: the two-arm manipulation stance is checked / driven in with hanging arms, closed pads
    "hinge_drift_max_rad": 0.03,  # §5.5 hinge drift ≤ 0.03 rad after settling
    "min_visible_objects": 5,  # §5.5 ≥ 5 objects with ≥ 200 px in the init render
    "min_object_pixels": 200,  # §5.5
    "region_seed_step_m": 0.25,  # §5.5 wall spots from support_regions (0.25 m seeding)
    "spot_search_rays": 24,  # §5.5 or the 24-ray search
    "flat_patch_rays": [5, 5],  # §5.5 flat patch 5 × 5 rays
    "flat_patch_tolerance_m": 0.01,  # §5.5 within 1 cm
    "flat_patch_margin_m": 0.10,  # §5.5 over (w + 0.10) × (h + 0.10)
    "front_clearance_m": 0.35,  # §5.5
    "in_room_test_m": 0.8,  # §5.5 in-room test 0.8 m in front
    "prompt0": {"beta_max_deg": 35.0, "d_min_m": 0.5, "d_max_m": 1.5},  # §5.5 Phase 1
    "lateral_range_m": [-0.40, 0.40],  # §5.5 Phase 2 Λ about C
    "contact_extreme_band_m": 0.03,  # §5.5 hull points within 3 cm of the extremes
    "brace_offset_m": 0.005,  # §5.5 fingertip face 5 mm from B
    "brace_clearance_m": 0.02,  # §5.5 brace hand ≥ 2 cm clear of the tilted hull at init
    "brace_kinds": ["below_high", "above_low", "side_high"],  # §5.5 candidates in order, first reachable wins
    "template_filter_deviation_deg": 25.0,  # §5.5 a spot is kept when steps 1–3 pass for δ = ±25° (A8; was ±30°)
    "scale_hint_steps": [0.9, 0.8, 0.7, 0.6, 0.5],  # §5.5 diagnostic pass s ∈ {0.9 … 0.5}, never applied automatically
    "settle_free": [],  # §5.5
}

#: §6.11 pair admissibility numbers shared by the small-scale IORS subtasks (checked by ground_isg's representative
#: test / the sampler; recorded here so the numbers live in one place).
IORS_SMALL_COMMON: Dict[str, Any] = {
    "partner_footprint_margin_m": 0.01,  # §6.11 the partner's top admits the target footprint with ≥ 1 cm margin
    "footprint_gap_min_m": 0.03,  # §4.6 target–partner edge-to-edge gap g ≥ 0.03
    "partner_gap_extra_m": 0.03,  # §4.4 D_p = min(W_p + 0.03, R_img)
    "partner_moved_max_m": 0.005,  # §6.11 partner displacement ≤ 5 mm at the goal
    "goal_drop_m": 0.01,  # §6.11 ontop / under(small) z = partner top + bottom offset + 1 cm drop
}

_HELD_OBJECT_COMMON: Dict[str, Any] = {
    "hold": {"depth_m": 0.50, "trunk": [0.5, 0.0], "roll_deg": 15.0},  # §6.3 fixed hold [pilot] (depth 0.50, roll 15°)
    "min_visible_fraction": 0.40,  # §6.3 ≥ 40 % of the object visible at init and goal
    "in_frame_margin": 0.05,  # §3.4 / §6.3 whole object in frame (5 %)
    "waypoints": 3,  # §6.3 / §3.9 motion checked at 3 intermediate waypoints
    "grasp_finger_advance_m_per_step": 0.001,  # §6.3 finger advance 1 mm/step
    "grasp_press_m": 0.004,  # §6.3 4 mm press
}

_CAMERA_RELOCALIZATION_COMMON: Dict[str, Any] = {
    "tolerances": {"position": 0.05, "orientation": 0.10},  # §8 camera pose 0.05 m / 0.10 rad (fixed)
    "groups": {},
    # §2.3: half-open geodesic ranges (0, 3], (3, 5], (5, 10] m; multiply_isg --path-buckets 3:10,5:10,10:5 overrides
    "path_buckets": [{"max_m": 3, "n": 10}, {"max_m": 5, "n": 10}, {"max_m": 10, "n": 5}],
    "goal_translate_range": [0.0, 1.25],  # §6.1 [pilot]; 0.95 is the deepest grid trunk (§3.3 clamp)
    "goal_tilt_range": [-0.5, 0.5],  # §6.1 [pilot]
    "max_heading_deg": 60.0,  # §6.1 camera forward-axis angle between init and goal ≤ 60°
    "min_goal_objects": 3,  # §6.1 goal render: ≥ 3 non-structural objects
    "min_object_pixels": 200,  # §6.1 an object counts with ≥ 200 segmentation pixels
    "min_shared_visible_fraction": 0.3,  # §6.1 ≥ 30 % of them visible from the init
    "collision_margin_m": 0.02,  # §6.1 both configurations whole-body collision-free (2 cm)
    "init_pose": {"trunk": 0.5, "arms": "zero"},  # §6.1 JoyLo reset pose (trunk 0.5, arms zero)
}

#: One ``generation`` seed per iddl subtask id.
DEFAULTS: Dict[str, Dict[str, Any]] = {
    # ---- camera relocalization (§6.1 / §6.2) -------------------------------------------------------------------
    "camera_relocalization-standard": {**COMMON, **_CAMERA_RELOCALIZATION_COMMON},
    "camera_relocalization-obstructed": {
        **COMMON,
        **_CAMERA_RELOCALIZATION_COMMON,
        "groups": {"obstacles": {"size": 1, "draw": "unique"}},  # §2.3 obstacles {size: 1, draw: unique}
        "obstacle": {
            "footprint_range_m": [0.12, 0.80],  # §6.2 footprint 0.12–0.80 m
            "height_range_m": [0.10, 1.70],  # §6.2 height 0.10–1.70 m
            "preferred_footprint_range_m": [0.20, 0.55],  # §6.2 preferred
            "preferred_height_range_m": [0.20, 1.10],  # §6.2 preferred
            "settle_steps": 60,  # §6.2 stands stably (settled 60 steps)
            "settle_drift_max_m": 0.005,  # §6.2 drift ≤ 5 mm
            "min_detour_m": 0.3,  # §6.2 blocked-map path ≥ 0.3 m longer than the free path
            "min_corridor_width_m": 0.75,  # §6.2 corridor keeps ≥ 0.75 m of free width on one side
            "in_frame_margin": 0.05,  # §6.2 inside the head camera's frame (5 %) at some pose along the new path
        },
    },
    # ---- held object (§6.3) ------------------------------------------------------------------------------------
    "object_scaling": {
        **COMMON,
        "tolerances": {"depth_min_m": 0.02, "depth_fraction": 0.20},  # §8 max(0.02 m, 0.20·|Δd|)
        "groups": {},
        "path_buckets": None,
        **_HELD_OBJECT_COMMON,
        "goal_depth_range_m": [0.35, 0.85],  # §6.3 depth d1 over [0.35, 0.85] m
        "goal_depth_grid_m": 0.05,  # §6.3 on a 5 cm grid
        "min_depth_change_m": 0.10,  # §6.3 |d1 − d0| ≥ 0.10 m
    },
    "mental_rotation": {
        **COMMON,
        "tolerances": {"orientation": 0.15},  # §8 0.15 rad (+ equivalences)
        "groups": {},
        "path_buckets": None,
        **_HELD_OBJECT_COMMON,
        "goal_angle_range_deg": [30.0, 120.0],  # §6.3 rotation about a sampled axis by U(30°, 120°)
        "reject_symmetric_targets": True,  # §3.13 rule 1 (symmetric_target)
    },
    # ---- disembedding (§6.4, §4.7) -----------------------------------------------------------------------------
    "disembedding": {
        **COMMON,
        "tolerances": dict(TOLERANCES_MEASURED),  # §8 measured only
        "groups": {"distractors": {"size": [1, 3], "draw": "unique"}},  # §2.3 distractors {size: [1, 3], draw: unique}
        "path_buckets": None,
        "tray_budget": 0.45,  # §6.4 distractor footprints (grown by pad_clearance_m) cover ≤ 0.45 of the inner floor
        "tray_raster_m": 0.01,  # §6.4 placement raster 1 cm over the tray's inner floor
        "tray_footprint_margin_m": 0.005,  # §3.2 / §6.4 0.5 cm footprint margin on a tray floor
        "distractor_spacing_extra_m": 0.01,  # §4.6 x = pad_clearance + 0.01 along the pinch axis
        "max_occlusion_ratio": 0.30,  # §4.6 / §6.4 distractors may occlude the target up to 30 %
        "min_visible_pixels": 400,  # §6.4
        "occlusion_reload_tolerance": 0.05,  # §6.4 loader re-measures with tolerance 0.05 on the ratio
        "image_margin": 0.05,  # §6.4 target projection inside a 5 % image margin
        "destination_visible_width_m": 0.04,  # §4.6 exactly 4 cm of the destination tray inside the init image
        "destination_gap_min_m": 0.04,  # §4.6 ≥ 4 cm from the source tray
        "destination_floor_visible_fraction": 0.30,  # §6.4 destination floor ≥ 30 % unoccluded in the confirmation render
        "lateral_margin_m": 0.437,  # §4.4 [pilot] the two fixed trays' half lengths along the edge + gap
        "carry_height_extra_m": 0.03,  # §4.7 lift ≥ tallest distractor + tray rim + 3 cm above the source floor
    },
    # ---- configuration matching (§6.5–6.8) ---------------------------------------------------------------------
    "configuration_matching-block_design": {
        **COMMON,
        "tolerances": {"position": 0.010, "position_edge_fraction": 0.4, "orientation": 0.15},  # §8 0.4·edge (≈ 0.010 m) / 0.15 rad
        "groups": {"targets": {"size": 4, "draw": "same_model"}},  # §2.3 targets {size: 4, draw: same_model}
        "path_buckets": None,
        "grid": [2, 2],  # §6.5 touching 2 × 2 grid anchored at the top-left block
        "patterns": {"test_rbbr": ["red", "blue", "blue", "red"]},  # §6.5 generation.patterns (or --variant)
        "block_gap_m": 0.001,  # §4.5 / §6.5 pitch = edge + 1 mm
        "scatter_min_gap_m": 0.03,  # §4.6 ≥ 3 cm gaps between footprints
        "region_spacing_extra_m": 0.03,  # §4.5 cells at spacing edge·√2 + 0.03
        "region_extra_cells": 2,  # §4.5 ≥ n_blocks + 2 cells
        "region_in_frame_margin": 0.08,  # §4.5 region in frame (8 %)
        "goal_face_visible_fraction": 0.5,  # §4.6 each goal face ≥ 50 % visible from the init stance
        "yaw_align_max_deg": 45.0,  # §6.5 yaw-align (|ψ| ≤ 45°)
        "cube_grasp_tilts_deg": [0.0, 30.0, 45.0, 60.0],  # §6.5 tilt about the pinch axis ∈ {0, ±30°, ±45°, ±60°}
        "flip_tilt_deg": 45.0,  # §6.5 [pilot] θ = 45° (30 / 60 are the alternatives)
        "orientation_equivalences": None,  # §3.13 blocks keep no equivalences (six distinct faces)
    },
    "configuration_matching-articulation_open_large_scale": {
        **COMMON,
        "tolerances": dict(TOLERANCES_ARTICULATION),  # §8
        "groups": {},
        "path_buckets": None,
        **ARTICULATION_COMMON,
        "direction": "open",  # §6.6 closed -> open to q_goal ∈ S
        "prompt0": {"beta_max_deg": 20.0, "d_min_m": 0.5, "d_max_m": 2.5, "d_scan_max_m": 1.10},  # §5.3 Phase 1 / Phase 2 d scan {0.35 … 1.10}
        "lateral_range_m": {"right": [-0.10, 0.60], "left": [-0.60, 0.10]},  # §5.3 Phase 2 Λ per handle arm
        "face_patch_half_width_m": 0.10,  # §5.3 moving face patch = contact ± 0.10 m along the pinch axis
        "pinch_axis_vertical_threshold": 0.7,  # §5.3 fwd_pinch_v if |pinch_axis·ẑ| > 0.7 else fwd_pinch_h
        "path_max_m": 3.0,  # §5.2 step 6 path init -> manipulation ≤ 3 m
    },
    "configuration_matching-articulation_close_large_scale": {
        **COMMON,
        "tolerances": dict(TOLERANCES_ARTICULATION),  # §8
        "groups": {},
        "path_buckets": None,
        **ARTICULATION_COMMON,
        "direction": "close",  # §6.6 q_init ∈ S (must satisfy Open) -> closed limit
        "prompt0": {"beta_max_deg": 20.0, "d_min_m": 0.5, "d_max_m": 2.5, "d_scan_max_m": 1.10},  # §5.3
        "lateral_range_m": {"right": [-0.10, 0.60], "left": [-0.60, 0.10]},  # §5.3
        "face_patch_half_width_m": 0.10,  # §5.3
        "pinch_axis_vertical_threshold": 0.7,  # §5.3
        "path_max_m": 3.0,  # §5.2
    },
    "configuration_matching-articulation_open_small_scale": {
        **COMMON,
        "tolerances": {**TOLERANCES_ARTICULATION, "position": 0.03, "orientation": 0.20},  # §8 + §6.7 root pose measured 0.03 / 0.20
        "groups": {},
        "path_buckets": None,
        **ARTICULATION_COMMON,
        "direction": "open",  # §6.7
        "floor_support_height_m": 0.6,  # §5.4 the floor is the support for objects taller than 0.6 m
        "yaw_jitter_rad": 0.1,  # §5.4 yaw = face_yaw(edge) + U(−0.1, 0.1)
        "handle_arm_side_threshold_m": 0.05,  # §5.4 handle arm = the arm on the handle's side (> 0.05 m; else right)
        "push_horizontal_min_fraction": 0.30,  # §5.4 horizontal component < 30 % -> v from the hinge line to the handle
        "brace_kinds": ["opposite_face", "top_press"],  # §5.4 first reachable wins
        "brace_face_shift_fraction": 0.25,  # §5.4 contact shifted ¼ of the face width toward the brace arm's side
        "brace_swept_clearance_m": 0.02,  # §5.4 contact ≥ 2 cm outside the moving link's swept AABB
        "view_margin_m": 0.03,  # §5.4 in-frame margin for the target stays 0.03 on the exact hull
        "stand_offs_m": [0.35, 0.45, 0.55],  # §5.4 Phase 2
    },
    "configuration_matching-articulation_close_small_scale": {
        **COMMON,
        "tolerances": {**TOLERANCES_ARTICULATION, "position": 0.03, "orientation": 0.20},  # §8 + §6.7
        "groups": {},
        "path_buckets": None,
        **ARTICULATION_COMMON,
        "direction": "close",  # §6.7
        "floor_support_height_m": 0.6,  # §5.4
        "yaw_jitter_rad": 0.1,  # §5.4
        "handle_arm_side_threshold_m": 0.05,  # §5.4
        "push_horizontal_min_fraction": 0.30,  # §5.4
        "brace_kinds": ["opposite_face", "top_press"],  # §5.4
        "brace_face_shift_fraction": 0.25,  # §5.4
        "brace_swept_clearance_m": 0.02,  # §5.4
        "view_margin_m": 0.03,  # §5.4
        "stand_offs_m": [0.35, 0.45, 0.55],  # §5.4
    },
    "configuration_matching-attachment": {
        **COMMON,
        "tolerances": {"position": 0.01, "orientation": 0.15},  # §8 layout relation 0.01 m / 0.15 rad
        "groups": {},  # the assembly is a :groups composite of two sampleable roles, not a drawn group
        "path_buckets": None,
        "partner_gap_min_m": 0.04,  # §4.6 attachment: gap ≥ 0.04
        "partner_centre_distance_min_m": 0.08,  # §4.6 centre distance ≥ 0.08
        "pre_attach_back_m": 0.10,  # §6.8 pre-attach pose 10 cm back along the female meta-link's insertion axis
        "reorientation_waypoints": 3,  # §6.8 in-hand reorientation at 3 SLERP waypoints
        "symmetry_slab_m": 0.005,  # §6.8 mating-axis symmetry test in 5 mm slabs
        "symmetry_match_fraction": 0.90,  # §6.8 accept a rotation when ≥ 90 % of the slabs match
        "grasp_metalink_clearance_m": 0.01,  # §6.8 every gripper sphere ≥ 1 cm from the male meta-link
    },
    # ---- alignment (§5.5, §6.9, §6.10) -------------------------------------------------------------------------
    "alignment-world_vertical": {
        **COMMON,
        "tolerances": {"orientation": 0.08},  # §8 alignment vertical 0.08 rad
        "groups": {},
        "path_buckets": None,
        **ALIGNMENT_WALL_COMMON,
        "goal": "upright",  # §6.9 orientation = aligned (upright)
    },
    "alignment-world_horizontal": {
        **COMMON,
        "tolerances": {"orientation": 0.08},  # §8 alignment horizontal 0.08 rad
        "groups": {},
        "path_buckets": None,
        **ALIGNMENT_WALL_COMMON,
        "goal": "level",  # §6.9 orientation = aligned (level)
        "swept_patch": True,  # §5.5 horizontal: swept extents over the max deviation
    },
    "alignment-axial": {
        **COMMON,
        "tolerances": {"orientation": 0.10},  # §8 alignment axial 0.10 rad
        "groups": {},
        "path_buckets": None,
        "deviation_range_deg": [20.0, 70.0],  # §6.10 target yaw = aligned yaw ± U(20°, 70°)
        "hinge_sink_m": 0.001,  # §6.10 hinge anchor sunk 1 mm below the top
        "hinge_drift_max": {"position_m": 0.001, "orientation_rad": 0.005, "steps": 60},  # §6.10 drift ≤ 1 mm / 0.005 rad over 60 steps
        "edge_visibility": {"points": 9, "min_visible": 3, "window_m": 0.30, "in_frame_margin": 0.05},  # §6.10 reference edge visible
        "scale_hint_steps": [0.9, 0.8, 0.7, 0.6, 0.5],  # §5.5 / §6.10 diagnostic scale pass
    },
    # ---- inter-object relation satisfaction (§6.11, §6.12) ----------------------------------------------------
    "inter_object_relation_satisfaction-ontop_small_scale": {
        **COMMON,
        "tolerances": dict(TOLERANCES_MEASURED),  # §8
        "groups": {},
        "path_buckets": None,
        **IORS_SMALL_COMMON,
        "fit_tiers": ["any_yaw", "aligned", "centred"],  # §6.11 partner top admits the footprint: preferred first
    },
    "inter_object_relation_satisfaction-under_small_scale": {
        **COMMON,
        "tolerances": dict(TOLERANCES_MEASURED),  # §8
        "groups": {},
        "path_buckets": None,
        **IORS_SMALL_COMMON,
        "fit_tiers": ["any_yaw", "aligned", "centred"],  # §6.11
        "argument_order": ["partner", "target"],  # §6.11 / iddl: (under ?partner ?target) — the partner ends under the target
    },
    "inter_object_relation_satisfaction-inside": {
        **COMMON,
        "tolerances": {**TOLERANCES_MEASURED, "orientation_metric": None},  # §8; §6.11 inside: orientation not evaluated
        "groups": {},
        "path_buckets": None,
        **IORS_SMALL_COMMON,
        "opening_margin_m": 0.01,  # §6.11 opening ≥ target footprint + 1 cm
        "drop_height_m": 0.005,  # §6.11 released 5 mm above the rim at the fillable centroid
        "drop_settle_steps": 120,  # §6.11 settled 120 steps
        "max_protrusion_fraction": 0.5,  # §6.11 protrudes above the rim by ≤ half its height
    },
    "inter_object_relation_satisfaction-nextto": {
        **COMMON,
        "tolerances": dict(TOLERANCES_MEASURED),  # §8
        "groups": {},
        "path_buckets": None,
        **IORS_SMALL_COMMON,
        "nextto_gap_extra_m": 0.03,  # §4.4 / §4.6 g ∈ [g_NT + 0.03, D_p], g_NT = mean(dims_a + dims_b)/6 (OmniGibson NextTo)
        "nextto_divisor": 6.0,  # §4.4 g_NT = mean(dims_target + dims_partner) / 6
    },
    "inter_object_relation_satisfaction-touching": {
        **COMMON,
        "tolerances": dict(TOLERANCES_MEASURED),  # §8
        "groups": {},
        "path_buckets": None,
        **IORS_SMALL_COMMON,
        "partner_side": "left",  # §4.6 / §6.11 partner on the robot's left (right hand pushes, left hand braces)
        "nudge_m": 0.001,  # §6.11 touching nudges of 1 mm
        "max_nudges": 25,  # §6.11 (≤ 25)
        "partner_stability": {"base_contact_fraction": 0.25, "foot_width_fractions": [0.6, 0.45], "max_height_over_extent": 2.5},  # §6.11
        "target_pushability": {"max_min_extent_m": 0.089, "max_long_side_m": 0.30, "max_height_m": 0.30},  # §6.11
    },
    "inter_object_relation_satisfaction-ontop_large_scale": {
        **COMMON,
        "tolerances": dict(TOLERANCES_MEASURED),  # §8
        "groups": {},
        "path_buckets": None,
        "max_partner_geodesic_m": 3.0,  # §5.6 [pilot] init support within 3.0 m geodesic of the partner
        "path_max_extra_m": 1.0,  # §3.7 / §5.6 B -> C path ≤ max_partner_geodesic + 1 m
        "goal_cell_margin_m": 0.05,  # §5.6 / §6.12 placement face eroded by half-extent + 0.05
        "goal_drop_m": 0.01,  # §5.6 z = partner face + bottom offset + 1 cm
        "partner_corners_in_frame_min": 4,  # §5.6 ≥ 4 of the partner's 8 AABB corners in frame at stance A
        "floor_init": {"distance_range_m": [0.6, 1.5], "cell_m": 0.02, "clearance_m": 0.02, "max_geodesic_m": 3.0},  # §5.6
        "stand_offs_m": [0.35, 0.45, 0.55],  # §5.6 stance C
        "partner_moved_max_m": 0.005,  # §5.6 partner unchanged during the goal settle (≤ 5 mm)
        "target_max_height_m": 0.35,  # A9 filler: ontop targets ≤ 0.35 m (representatives_report.md)
    },
    "inter_object_relation_satisfaction-under_large_scale": {
        **COMMON,
        "tolerances": dict(TOLERANCES_MEASURED),  # §8
        "groups": {},
        "path_buckets": None,
        "max_partner_geodesic_m": 3.0,  # §5.6 [pilot]
        "path_max_extra_m": 1.0,  # §3.7 / §5.6
        "goal_cell_margin_m": 0.05,  # §5.6
        "goal_drop_m": 0.003,  # §5.6 under: z = floor + bottom offset + 3 mm
        "under_ray_extra_m": 0.03,  # §5.6 upward ray hits the partner ≥ target height + 0.03
        "under_near_edge_max_m": 0.45,  # §5.6 within 0.45 m of the partner's near AABB edge
        "under_clearance_extra_m": 0.17,  # §6.12 clearance ≥ target height + 0.12 (gripper below the wrist) + 0.05
        "under_leg_margin_m": 0.05,  # §6.12 footprint fits between the legs with 5 cm margin
        "partner_corners_in_frame_min": 4,  # §5.6
        "floor_init": {"distance_range_m": [0.6, 1.5], "cell_m": 0.02, "clearance_m": 0.02, "max_geodesic_m": 3.0},  # §5.6
        "stand_offs_m": [0.35, 0.45, 0.55],  # §5.6
        "partner_moved_max_m": 0.005,  # §5.6
        "target_max_height_m": 0.20,  # A9 filler: under targets ≤ 0.20 m (representatives_report.md)
        "argument_order": ["target", "partner"],  # iddl: (under ?target ?partner) — the target ends under the partner
    },
}

#: Group roles whose members are drawn (``generation.groups.<role>``), per subtask (kind ``group`` in the icl).
GROUP_ROLES: Dict[str, tuple] = {
    subtask: tuple(sorted(gen.get("groups") or {})) for subtask, gen in DEFAULTS.items()
}


def subtasks() -> list:
    return sorted(DEFAULTS)


def defaults_for(subtask: str) -> Dict[str, Any]:
    """A deep copy of the ``generation`` seed of ``subtask`` (KeyError for an unknown subtask id)."""
    if subtask not in DEFAULTS:
        raise KeyError(f"no generation defaults for subtask {subtask!r}; known: {sorted(DEFAULTS)}")
    return copy.deepcopy(DEFAULTS[subtask])


def seed_generation(subtask: str, existing_entry: Optional[Dict[str, Any]] = None, reseed: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """The ``generation`` block of a customs entry (§2.3 "Seeding").

    Copied from ``existing_entry["generation"]`` when the alias was grounded before (the tuned numbers survive a
    re-grounding), otherwise :func:`defaults_for`. Keys the defaults have and the existing block lacks are added
    (a spec number introduced after the entry was created), existing values are never overwritten -- except the
    keys named in ``reseed`` (``ground_isg --reseed-generation KEY,...``: a spec number the owner changed, e.g.
    ``deviation_range_deg`` after A8), which take the default again; ``reseed = ["*"]`` re-seeds the whole block
    (the group sizes are re-derived from the alias anyway).
    """
    seed = defaults_for(subtask)
    if existing_entry and isinstance(existing_entry.get("generation"), dict):
        if reseed and "*" in reseed:
            return seed
        gen = copy.deepcopy(existing_entry["generation"])
        for k, v in seed.items():
            gen.setdefault(k, v)
        for k in reseed or ():
            if k in seed:
                gen[k] = copy.deepcopy(seed[k])
            else:
                gen.pop(k, None)  # a key the defaults no longer have
        return gen
    return seed


def group_seed(subtask: str, role: str, generation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """``generation.groups.<role>`` (``size``, ``draw``) of ``subtask`` from ``generation`` (else the defaults)."""
    gen = generation if generation is not None else defaults_for(subtask)
    groups = gen.get("groups") or {}
    if role not in groups:
        raise KeyError(f"subtask {subtask!r} has no generation.groups entry for group role {role!r}")
    g = dict(groups[role])
    g.setdefault("draw", "unique")
    return g


__all__ = [
    "ALIGNMENT_WALL_COMMON",
    "ARTICULATION_COMMON",
    "COMMON",
    "DEFAULTS",
    "GROUP_ROLES",
    "IORS_SMALL_COMMON",
    "TOLERANCES_ARTICULATION",
    "TOLERANCES_MEASURED",
    "defaults_for",
    "group_seed",
    "seed_generation",
    "subtasks",
]
