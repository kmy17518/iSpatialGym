# iSpatialGym instance sampling: requirements and design (v1)

Status: specification for `ispatialgym/sampling` (sampler v1), 2026-09-11. Companion to
`task_bddl_formats.md` (what an instance evaluates) and `task_definition_and_evaluation.md`
(artifacts, deterministic loading). Version history of the sampler itself is in
`ispatialgym/sampling/CHANGELOG.md`.

## 0. Purpose

`python -m ispatialgym.sampling.sample_instances` turns a *task spec* (iSpatialGym's version of
BEHAVIOR's task-custom list) into benchmark instances that can be loaded deterministically
(BDDL problem + scene template + TRO) and that describe the goal numerically for the evaluator
(`ISpatialGymTask`). It does **not** produce goal images or demonstrations; those come from the
separate action script (oracle / policy / gt-replay modes), which may also refine
`pose_reached` targets after the oracle has physically reached them.

Design principles, in the order they were traded off:

1. **Correctness of the saved instance**: `:init` true, goal not satisfied, every settled state
   stable (objects and robot), goal configuration physically feasible without the robot, every
   TRO target consistent with the BDDL (validated by `validate_tro_against_problem`).
2. **No trial-and-error where geometry can be solved**: placements are computed from support
   geometry, the camera model and pre-computed reachability sets, then *verified* once. Random
   sampling only chooses among pre-filtered candidates.
3. **Speed**: one simulator process per (problem, scene, model set) generates many instances;
   everything that depends only on the template (table geometry, stance, camera solution,
   reachability grid, grasp library) is computed once and cached; renders are used only for
   measurements that need them (visibility counts, occlusion ratios).
4. **Identical robot configuration across instances of a task** wherever the requirements ask
   for it (all joints except the base pose, sometimes except the trunk), so policies see the same
   proprioception at t=0.

## 1. Artifacts and layout

```text
<out_root>/                                   default: <repo>/datasets/ispatialgym-task-instances
  task_custom_lists.json                      the spec that was used (copied)
  <activity>-<def_id>/
    <template_id>/                            e.g. house_single_floor__bottle_of_beer-gxxbhh
      template.json                           full scene json (scene + spawned task objects + robot)
      template-partial_rooms.json             same, but only the rooms in load_room_instances
      activity_definitions/<activity>/problem<def_id>.bddl   concrete BDDL for this template
      template_meta.json                      bindings, rooms, cached geometry, spec hash, timings
      instances/
        <instance_id>-tro_state.json          TRO (BEHAVIOR object states + robot_poses + ispatialgym)
        <instance_id>.summary.json            what was sampled / measured (debug, not needed to load)
```

- `template_id = <scene_model>__<role>-<category>-<model>[__...]` for the spawned task objects
  (fixed scene objects are recorded in `template_meta.json`, not in the id).
- `instance_id = <activity>-<def_id>__<template_id>__<NNNN>`.
- The BDDL declares **concrete synsets** (the knowledge-base synset of the bound category:
  `bottle__of__beer.n.01_1`, `breakfast_table.n.01_1`, `floor.n.01_1`, ...). The shipped files
  under `ispatialgym/assets/bddl/activity_definitions` are the *structural templates*; the
  sampler regenerates them per template with `ispatialgym.sampling.bddl_writer`, parses the result
  with `ispatialgym.activity.parse_problem_string` and compiles it before anything is written.
  `ISpatialGymTask(activity_definitions_root=<template_dir>/activity_definitions)` loads it.
- The TRO keeps BEHAVIOR's layout (`<bddl instance>: serialized object state`, `robot_poses`) and
  the reserved `ispatialgym` block (`isg-tro-v1`): `scene` (model, floor, `template_path`,
  `template_sha256`, `load_room_instances`), `object_bindings` (with `category`, `model`,
  `dataset_name` for non-BEHAVIOR assets), `robot_state` (all 28 joints by name), `pose_targets`
  / `groups` / `articulation_targets` per `task_bddl_formats.md`, plus the restoration records
  `grasps`, `constraints`, `attachment_targets`, `initial_occlusion`, `bddl_sha256`,
  `task_spec_sha256`, `generation_seed`.
- The scene template is written exactly like BEHAVIOR (`scene.save`, then `merge_scene_files`
  with the stable scene json), so `InteractiveTraversableScene(scene_file=template.json,
  load_room_instances=...)` reproduces it.
- Custom assets (the Kohs block) live in the dataset `ispatialgym-assets` under `gm.DATA_PATH`
  (source and build scripts in `ispatialgym/assets/objects/`); templates reference them through
  `DatasetObject(dataset_name="ispatialgym-assets")`.

## 2. Inputs

### 2.1 Command line

```text
sample_instances --spec <task_custom_lists.json> --problem <activity>-<def_id> [--problem ...]
                 [--scene <scene_model>] [--n-instances N] [--seed S] [--out <out_root>]
                 [--variant <block_design pattern id> ...] [--overwrite] [--max-attempts K]
```

- `--problem` selects the BDDL problem (activity + definition id), i.e. the variant
  (`configuration_matching-0` = block design, `-1/-2` = articulation open/close, `-3` =
  attachment; `world_alignment-0/1/2` = table surface / world vertical / world horizontal;
  `inter_object_relation_satisfaction-0..4`). Sub-variants
  that live inside one problem (block-design pattern, articulation open vs close, occluded
  vs unoccluded) are chosen from the spec's `generation` block per instance or forced with
  `--variant`.
- `--n-instances` is per template (per scene x model set).
- One OmniGibson process handles one problem; several templates of that problem are processed
  sequentially in the same process (scene reload between templates).

### 2.2 Task spec (`task_custom_lists.json`)

Keyed by `<activity>-<def_id>`. Roles (not BDDL instance names) are the unit of configuration;
the BDDL writer maps roles to instance names.

```json
{
  "object_scaling-0": {
    "alias": null,
    "roles": {
      "target": {"models": {"bottle_of_beer": ["gxxbhh"]}, "grasps": "library"}
    },
    "scenes": {
      "house_single_floor": {"rooms": ["living_room_0"], "fixed": {"floor": ["floors_ifmioj_0"]}}
    },
    "generation": {"instances_per_template": 5, "...task specific...": 0}
  }
}
```

- `roles.<role>.models`: `{category: [model, ...]}`; every listed model must be *ready* (no
  unprocessed complaint in the BEHAVIOR knowledge base) and, for grasped roles, have validated
  grasps in the grasp library (`runlogs/grasp_library_20260905`, see 4.7). One (category, model)
  choice per role per template; the sampler iterates over the cartesian product unless
  `roles.<role>.pick = "random"`.
- `roles.<role>.dataset` (optional): dataset name for non-BEHAVIOR assets.
- `scenes.<scene>.fixed.<role>`: scene object names the fixed role may bind to (table, bed,
  wall, floor). `"auto": {"categories": [...]}` lets the sampler pick among scene objects of
  those categories that pass the role's geometric checks. `rooms` adds room instances to load
  beyond those of the bound fixed objects.
- `generation`: task-specific parameters (section 5); every numeric range is
  `[min, max]` and sampled uniformly unless stated otherwise.
- `distractors` (disembedding only): the distractor pools **per target category** plus the look-alike
  groups (`spec.DistractorSpec`):

  ```json
  "distractors": {
    "families": {"cartons": ["box_of_cereal", "box_of_rice", "..."], "bottles": ["..."], "...": []},
    "lookalike_groups": [["mug/ehnmxj", "mug/fapsrj"], ["box_of_butter/mfjuil", "box_of_cookies/apjokz"]],
    "pools": {
      "box_of_cereal": {"same_category": ["hcwbzw", "hosfhj", "..."], "other_categories": ["box_of_rice", "..."]},
      "mug": {"same_category": ["ehnmxj", "fapsrj", "..."], "other_categories": ["beaker", "pencil_holder", "..."]}
    }
  }
  ```

  For a target `C/m` the *same-category pool* is `pools[C].same_category` without `m` and without every
  model that shares a look-alike group with `m`; the *other-category pool* is the union of the
  `same_category` lists of `pools[C].other_categories` plus an optional explicit
  `pools[C].other_models` (`{category: [models]}`), again without look-alikes of `m` (groups may cross
  categories; a group may also be written as `{"members": [...], "reason": "..."}`). A look-alike group
  holds models that cannot be told apart from the robot's initial view
  (exact duplicates, same-shape variants whose only difference is a small print, label or shade); its
  members are never drawn as each other's distractors, so an instance never asks for a target that a
  distractor could be confused with. `families` is informational (how `other_categories` were derived).
  The loader rejects unknown categories, models that are not ready or not of the stated category,
  `other_categories` without a pool, look-alike members that are neither pooled nor targets, and a target
  category without a pool. The section for the 39 disembedding categories is generated by
  `runlogs/grasp_library_20260905/scripts/build_disembedding_pools.py` (full record with grasp keys, bbox,
  scores and the reason of every look-alike group in `select/disembedding_distractor_pools.json`).

## 3. Shared rules

### 3.1 Scenes

Candidates: `house_single_floor`, `house_double_floor_upper`, `house_double_floor_lower`,
`Rs_int`, `hotel_suite_large`, `restaurant_diner`, `office_cubicles_right` (2026 challenge
`<scene>_stable.json`). Rooms: the *task rooms* are `scenes.<scene>.rooms` + the rooms of the
bound fixed objects; `template_meta.load_room_instances` (what evaluation, validation and
rendering load) adds their **neighbours** - room instances whose segmentation-map footprint lies
within 0.6 m of a task room (shared wall or doorway; `SimWorld.adjacent`) - so that walls, doors and furniture seen through doorways and windows exist
instead of sky. Walls, ceilings and doors of the loaded rooms always load (OmniGibson
behaviour). The sampler itself loads the task rooms only (`template_meta.sampled_with_rooms`;
`ISG_LOAD_ADJACENT=1` changes that): its `seg_instance` measurement renders crash Isaac Sim 5.1's
synthetic-data graph on house_single_floor as soon as the kitchen or the garden come into view,
and the neighbours' objects never occlude a tabletop target nor collide with a stance inside the
task room (cuRobo obstacles are limited to the task rooms' extent + 1 m anyway). Category-based
fixed roles bind to task-room objects only. The TRO records the room list.

Scene variants (v1.4, `sampling/scene_variants.py`): table tasks sample in the stable scene
**minus the chairs around the work table** (`CHAIR_CATEGORIES` within `generation.chair_distance`
= 1.0 m of any table the fixed `table` role can bind to; `generation.remove_chairs_near_tables:
false` opts out, `chair_categories` overrides). The variant json (`<out>/scene_variants/`) is the
scene file the sampler loads, so `template.json` is merged from it and has no chairs, the reload
checks compare against it, and `template_meta.scene_variant` lists `removed_objects` plus the
`freed_footprints` for the traversability map (the PNG is baked with the furniture; the sampler
frees those cells in the loaded map where `floor_trav_no_obj_0.png` has floor and re-blocks the
table's footprint). Evaluation must apply the same patch
(`scene_variants.patch_trav_map_from_meta(env.scene, template_meta)`), otherwise the base planner
treats the former chair spots as blocked.

### 3.2 Support and placement

- An object placed "on" a surface has its **whole footprint** (AABB projected on the surface
  plane, after the sampled yaw) inside the support polygon shrunk by a 2 cm margin; no part hangs
  over an edge. The support polygon is the top face of the support's collision geometry
  (largest horizontal face cluster within 1 cm of the AABB top), not the AABB, so table aprons
  and rounded edges do not count.
- Objects rest at `z = support_top + (object_z_min_offset)`, computed from the object's AABB
  in its canonical (or chosen rest) orientation; a 3 mm drop lets physics seat them.
- Objects never interpenetrate: 2D footprints keep >= 1 cm clearance (unless a task asks for
  contact), and nothing is placed inside the robot's or another object's AABB.
- Placement uses stable rest orientations only: canonical up, or the grasp library's
  `rest_quat_wxyz` for models that must lie on a side, times a random yaw. This is narrower than
  BEHAVIOR's kinematic sampler (`object_state_utils.sample_kinematics` →
  `DatasetObject.sample_orientation`), which draws one of the asset's precomputed stable
  orientations weighted by their probability and adds a random yaw. The restriction is
  deliberate: the grasp library's poses are validated for the rest orientation, and the goal
  predicates (`ontop`, `nextto`, ...) do not depend on it. Extending to the other stable
  orientations would mean re-validating grasps per orientation.

### 3.3 Robot

- R1Pro, JoyLo controller config (`r1pro_model.joylo_robot_config`, name `robot_r1`,
  `grasping_mode: assisted`, head camera `robot_r1:zed_link:Camera:0`, 1080x1080, 40 mm
  horizontal aperture, focal length 17 mm -> fx = fy = 459 px, hfov = vfov = 99.2 deg).
- Arm configurations (measured, base at translate 0.5 unless noted; renders in
  `runlogs/instance_renders_v1.1/arm_poses.png`): "tucked" = all arm joints 0 (arms hanging,
  right eef 0.66 m up at torso 0); "untucked" = `untucked_default_joint_pos`
  (`[0, -1.57, 0, -1.57, -1.57, 0, 0]`: arms stretched out horizontally to the sides, eef
  (0.64, -0.56, 1.20) m from the base) - BEHAVIOR's reset pose; "ready" = `ARM_READY_Q`
  (`[-0.073, -0.326, 0.951, -1.639, -0.632, 0.137, -0.926]`: elbows bent, hands in front at
  chest height, eef (0.51, -0.15, 0.78)). The torso follows the JoyLo keyframe curve at the
  chosen (translate, tilt); the oracle's `set_torso` can reach any (translate in [0, 2], tilt
  within `torso_joint3` limits).
- **Init arm pose per task** (v1.4): `generation.init_arms` (default `untucked`) is used by every
  sampler - table stances, the free arm of the held-object tasks, camera relocalization (torso
  joints zero there), the vertical-alignment arc bases. There is no automatic fallback: a stance
  that collides with the chosen arm pose rejects the template (alternatives only if the spec lists
  them in `generation.view_arm_fallbacks`), so all instances of a task start with the same arms.
  Within a template everything but the base pose is identical across instances: trunk
  (translate, tilt) is the template's view solution, arms are `init_arms` (right arm = the hold
  IK solution for held tasks), grippers open (closed on the object for held tasks).
- Robot base pose is sampled **together with** the objects (single joint sample -> settle ->
  validate), not in a second pass, unless a task explicitly needs the objects first
  (disembedding occlusion measurement, which depends on the settled clutter).
- Whole-body collision freedom of the init configuration is checked with cuRobo spheres
  against every collision mesh in the loaded rooms (`R1ProCuRobo.check_collision`, 1 cm margin
  for non-floor links).
- `robot_poses.robot[0]` (BEHAVIOR) and `robot_state.joint_positions` (the 22 real joints; the
  six virtual base joints are excluded) are written; the loader restores both.
- Implementation facts (OmniGibson `HolonomicBaseRobot`): the articulation root of R1Pro is the
  mass-less virtual link `base_footprint_x`, anchored to the world by a fixed joint at the pose the
  robot was *created* at (`PARK_POSITION` = (40, 40, 5) in the sampler and the validator); below it
  the chain `base_footprint_{x,y,z,rx,ry,rz}_joint` leads to `base_footprint_link`, the link every
  "robot pose" refers to. While the simulator plays, `robot.set_position_orientation()` does not move
  the root: it converts the requested pose into the six virtual joint values (measured: root stays at
  (40, 40, 5), base at (13.8, 1.8, 0) ⇒ x/y/z joints = (−26.2, −38.2, −5.0)). Consequently "the base
  pose" is part of the joint vector (`RobotKit.q_at_base_pose` sets x / y / yaw *and* z / roll /
  pitch from the floor height), the TRO stores the base as a world pose plus the 22 real joints, and
  a loading process recomputes the virtual joints relative to *its own* root — the first and the
  n-th instance loaded in a process go through exactly the same path. Gravity acts on the base
  through the prismatic z joint: parked over a region without floor mesh (an unloaded room) the base
  falls, which is why the sampler parks on a free cell of a **loaded** room. cuRobo stores obstacles
  relative to the root at sync time; `RobotKit._ensure_synced` re-syncs if the root itself moved more
  than 1 cm (a guard; with the fixed world joint this is not expected to trigger).

### 3.4 Camera model and visibility

- Frustum tests use the pinhole model above (points project with fx = fy = 459, cx = cy = 540,
  optical axis -Z of the sensor prim). "Well within view" = every AABB corner of the object
  projects inside the image with a 5 % margin. "Barely visible" = a configured sliver
  (default 4 cm) of the object is inside the image.
- Object visibility counts and occlusion ratios use the head camera's `seg_instance` render at
  the exact evaluation resolution, after 3 renders (renderer latency). A render happens only
  when a measurement is needed (camera relocalization, disembedding occlusion, held-object
  self-occlusion, world-vertical object count).
- Structural categories (never counted as objects): `walls`, `floors`, `ceilings`, `roof`,
  `baseboard`, `downlight`, `room_light`, `electric_switch`.
- Light fixtures (categories containing `light`, `lamp`, `chandelier`) are invisible while the
  sampler runs (renderer crash workaround, section 8) and therefore never appear in visibility
  counts; templates are saved with them visible.

### 3.5 Goal not satisfied at init

Every instance checks, after settling, that the binary goal `G` is **false** and `:init` is
true (compiled conditions, `ISpatialGymTask` machinery). Individual goal atoms that are also
init facts may hold (e.g. `ontop tripod table`); the conjunction may not.

### 3.6 Settling and validation (mirrors `sample_b1k_tasks.py` / `multiply_b1k_tasks.py`)

For every instance:

1. Teleport objects and robot to the sampled configuration, `keep_still()` everything.
2. Let physics run `settle_steps` (default 60 physics steps; objects free, robot joints held by
   the position controllers, base kept still) then 10 more; check every task object's
   linear velocity < 1 mm/s-ish (`_validate_object_state_stability(strict=False)` thresholds).
3. `keep_still()` all task entities, dump the scene state, and run the BEHAVIOR checks of
   `validate_task`: (a) every non-task object of the loaded rooms is within 5 cm / 0.05 rad of
   the stable scene json; (b) all velocities ~0; (c) `:init` conditions true; (d) after one
   physics step and after 10 more, poses/joints unchanged within tolerance.
4. Goal feasibility (robot excluded): the robot is parked (tucked, room centre) and the task
   objects are teleported to their goal poses (or the goal joint positions / attachment set),
   settled 120 steps, every binary goal atom must evaluate true and the objects must stay within
   `pose_targets` tolerances; then the init state is restored exactly (`og.sim.load_state`).
   Parking matters: a hand hovering where the goal object lands ejects it (disembedding, v1.5).
5. Task-specific checks (reachability, visibility, occlusion, grasp restoration) as listed in
   section 5.
6. Write TRO + summary; `update_initial_file`/`reset` to the init state for the next instance.

Scene reset (v1.2): before **every** attempt, and again right before `template.json` is written,
`SimWorld.reset_scene_objects` reloads every non-task scene object (pose, velocities, joints)
from the stable scene json. Every template also starts from a clean world (v1.3): scene objects
reset and the robot parked with tucked arms at the room's free centre before the fixed roles are
bound - left at the previous template's stance, its untucked arms hover over the table and the
support raster's downward rays hit them (cells wrongly "occupied"). Instances therefore differ from the scene json only in the task
objects and the robot, whatever physics did meanwhile (a task object nudging furniture, slow
creep of the living-room floor lamp, ...). The 5 cm / 0.05 rad `check_scene_unchanged` guard is
kept as a rejection criterion for gross disturbance during the instance's own settling.

Evidence written per instance (`<iid>.summary.json`): `bddl_at_init` (init conditions all true,
every goal atom with its truth value at init; the binary goal is false) and `goal_check.goal`
(every binary goal atom true at the teleported and settled goal; `pose_settled` atoms are
runtime-only and excluded).

Kinematic predicates are cached per object pair and treat sleeping objects as unchanged, so the
sampler wakes / re-places task objects before every predicate evaluation
(`task_definition_and_evaluation.md`, "Caveat for scripted settling code").

A separate `validate_instances.py` reloads every template + TRO in a fresh process
(`InteractiveTraversableScene(scene_file=template, load_room_instances=...)` - task rooms plus
neighbours, rgb sensors only - with the JoyLo robot from the config +
`ISpatialGymTask.load_tro_instance`, three settling steps so that the robot rests on the floor)
and checks: `:init` true, binary goal false, no drift of any task object over 60 further steps,
robot pose / joints equal to the TRO, and (informational) the binary goal after a plain teleport
to the world-frame targets. Spawned
objects carry `in_rooms` = the loaded rooms in the template, otherwise partial-room loading would
skip them. Status for the test set: 29 / 29 instances pass (held-object `grasped` and hinge
constraints are skipped until the loader restores them). Since v1.2 it also diffs every non-task
object of `template.json` against the stable scene json (1 mm / 0.005 rad).

Visual check: `render_instances.py` loads every instance the same way and writes
`<iid>-init.png` in the BEHAVIOR replay layout (`joylo/scripts/replay_data.py`: left wrist |
head | right wrist on top, the replay's over-the-shoulder external camera on `base_link` below)
plus a per-problem contact sheet in `<out>/renders/`. Grasped / hinged objects, which the loader
cannot restore yet, are posed from the TRO with gravity disabled for the picture.

### 3.7 Reachability without trial and error

For table tasks the base stance is fixed relative to the support's **front edge** (3.8), and
the set of table positions where a top-down (or 30 deg tilted) grasp is IK-feasible is
computed **once per template** on an 8 cm lattice over the placement region, for every candidate
trunk translate in `generation.reach_translates` (default `[0.5, 0.75, 1.0, 1.25]`), with the
pre-grasp (10 cm above) required as well. Objects that must be grasped are then placed only on
lattice cells that are reachable for at least one translate ("the robot may sit"). On top of the
generic grid, tasks with a grasp library build a **grasp combination table** per template: for
every reachable cell x 8 object yaws x every library grasp, whether pre-grasp and grasp are
IK-feasible (screening IK, 12 seeds, batched). Instance sampling then only *draws* from this
table and confirms the draw with the exact 32-seed IK; it never searches.

Measured facts that shaped the defaults (R1Pro right arm, dining table at 0.80 m, stance 0.45 m
from the edge):

- cuRobo IK costs ~1 ms per (seed, problem) whatever the batch size, so batching helps only
  by amortising Python overhead; the win comes from fewer queries (coarser lattice, 12 seeds
  for screening, cached tables), not from larger batches.
- The reachable strip is about 0.30-0.34 m deep from the front edge and lies to the robot's
  **right** of its centre line (lateral -0.44 .. -0.03 m); deeper trunk translates did not add
  cells. Long objects (keyboard) therefore have their *ends* checked, not their centre.
- Reach checks use the **tucked** arm configuration for the locked joints ("sitting" posture):
  the untucked hands sweep through the table top at deep translates. The oracle lowers the arms
  before sitting.

### 3.8 Canonical table stance

Given a support (table / countertop / desk / nightstand):

- Front edge = the support-polygon edge closest to the robot's approach side (chosen per fixed
  object: the edge with the most free floor in front, or `fixed.<role>.front_edge`).
- Stance: base_link at `stand_off` from the front edge (default 0.45 m, whole-body collision
  free), laterally centred on the placement region, yaw facing the edge inward normal.
- Trunk (translate, tilt) is solved so the head camera's optical axis passes through the
  region of interest (tray centre / object centre / arrangement centre): grid over translate in
  [0, 1.5] x tilt in joint range, FK of the camera link, pick the pair with the smallest angle
  between the optical axis and the ray to the target; ties broken toward translate 0.5.
- All instances of one template share (stand_off, translate, tilt, arm configuration) and
  differ only by base pose (which follows the region on the table) and the sampled objects.

### 3.9 Grasp library

Grasps come from `runlogs/grasp_library_20260905/cands/<key>.yaml` (isaac_grasp, eef pose in the
object base_link frame) filtered by `results/<key>.json` (assisted grasp attached and held under
gravity) and the reviewer verdicts in `review/*.json` (only `ok` rows, `one_finger` tolerated).
The spec names the result key per role model (`grasps: {"key": "bottle_of_beer__gxxbhh"}`) or
`"library"` to use the task's default pipeline key.

## 4. Task customs list for the validation run (chosen objects)

| Problem | Roles (category / model) | Scene(s) | Notes |
| --- | --- | --- | --- |
| camera_relocalization-0 | - | house_single_floor living_room_0 | floor = living room floor |
| object_scaling-0 | target bottle_of_beer/gxxbhh (5 valid grasps, reviewer ok rows 3 and 5 two-sided) | house_single_floor | floating grasps |
| mental_rotation-0 | target toy_figure/zuntus (5 clean) | house_single_floor | floating grasps |
| disembedding-0 | target box_of_cereal/iucshm; trays tray/gsxbym (source), tray/hvlfig (dest); table breakfast_table_rhjoby_0 | house_single_floor dining_room_0 | distractors from the per-category pools (`distractors` section, all 39 disembedding categories): for the cereal box the 10 other cereal models that are not its look-alike (qirvmd is), or the `cartons` family (box_of_rice, box_of_cane_sugar, box_of_shampoo, box_of_oatmeal, carton_of_milk, box_of_baking_mix, box_of_granola_bars, box_of_butter, box_of_cookies, bottle_of_milk) |
| configuration_matching-0 | 4 x toy_dice/isgblk (Kohs block, `ispatialgym-assets`) | house_single_floor | pattern `[red, blue, blue, red]` |
| configuration_matching-1/-2 | toolbox/redmtl (`j_lid`, handle = bar on the lid) on the table | house_single_floor | non-scene articulated object; scene-object variant implemented but untested |
| configuration_matching-3 | cap/xxqyrt (male) -> lemonade_bottle/yprkek (female) | house_single_floor | rotational symmetry of the cap -> equivalences |
| world_alignment-0 | keyboard/{hgtbot, injrgc} | house_single_floor | long axis parallel to the front edge, keys toward the robot |
| world_alignment-1 | painting/{dacihj, ljabhq} hinged on a wall | house_single_floor | candidates checked visually for one sensible "up" |
| world_alignment-2 | towel_rack/{kqrmrh, crivoa} hinged on a wall | house_single_floor | elongated wall item with one clear horizontal direction; goal = level; kqrmrh is a plain bar (`symmetric_180`) |
| inter_object_relation_satisfaction-0 (ontop) | pack_of_kielbasa/dgodnn -> platter/csanbr | house_single_floor | |
| inter_object_relation_satisfaction-1 (under) | toy_figure/cobfpg on desk_xhjsub_0 -> under bed_rrcvaq_0 (childs_room_0) | house_single_floor | small under big; bedroom_0's bed_gxfipj_0 was dropped (no table nearby, frame blocks the arm) |
| inter_object_relation_satisfaction-2 (inside) | lemon/ovbcff -> bowl/fedafr | house_single_floor | |
| inter_object_relation_satisfaction-3 (nextto) | spatula/hkvsep -> soup_ladle/cfzneb | house_single_floor | |
| inter_object_relation_satisfaction-4 (touching) | can_of_baking_mix/{blrqqz, orwvfx} -> board_game/{eoywpy, gcfkqa} | house_single_floor | both categories unused by rounds 1-3, >= 2 ready models each |

Rationale for the free choices:

- *World alignment / table*: `keyboard` (6 ready models, 14 x 45 x 2 cm). The key rows make top
  vs bottom obvious, the asymmetric layout (numpad, escape key) makes left vs right obvious, and
  a keyboard naturally aligns with a table edge. Alternatives considered: `hardback` (cover art
  varies), `laptop` (articulated, 1 ready model).
- *World alignment / world vertical*: `painting` (36 ready models; the BDDL template already uses
  `painting.n.01`). Models whose content has one obvious up direction and a frame 0.4-0.6 m wide
  (pushable through one arm arc) are listed; each candidate was rendered and inspected before it
  entered the list.
- *World alignment / world horizontal*: `towel_rack` (6 ready models, all 6 valid for the
  horizontal rubric, 0.41-0.53 m bars on brackets; the BDDL template's synset). A wall bar is the
  archetypal "rod" of the rod-and-frame test: its long axis reads level at a glance and a 15 deg
  tilt is obvious, it fits every validated wall region (swept patch 0.58 m against a smallest
  validated patch of 0.6 m), and it is natural
  in a house, hotel or office. `kqrmrh` (plain bar, identical upside down) exercises the
  `symmetric_180` equivalence, `crivoa` (hook rail, hooks down) the single upright level pose.
  Round-1 categories of the selection (`rolling_pin`, `baseball_bat`, `shelf`, `radiator`,
  `wall_mounted_tv`, ...) are not used by any other task; `towel_rack` is world vertical's
  round-1 category and therefore world horizontal's round 2 (rounds in
  `runlogs/world_horizontal_selection_20260912/selection.md`).
- *IORS touching*: `board_game` (16 ready, flat 13 x 20 x 3 cm box, stable when pushed) and
  `can_of_baking_mix` (6 ready, 9-10 cm cylinder, stable upright). Neither category appears in
  the round 1-3 selections (`runlogs/grasp_library_20260905/rounds_summary.md`).
- Partners (platter, bowl, soup ladle, trays) are ready models with a `fillable` meta link where
  `inside` needs it; the trays are the largest shallow ready trays so a 27 cm cereal box plus
  distractors fit.

## 5. Per-task requirements

Notation: `T` = support top height, `cam` = head camera, `d(.)` = Euclidean distance.

### 5.1 Camera relocalization (`camera_relocalization-0`)

Init: robot tucked at base pose `B0`. Goal: camera pose at base pose `B1`, trunk (translate,
tilt) = `(t1, k1)`.

1. `B0`, `B1` sampled on the traversability map of the loaded floor, eroded with the robot
   (`TraversableMap._erode_trav_map(robot=robot)`: radius = half the chassis diagonal + 0.2 m),
   inside the configured room(s).
2. `d(B0, B1) < 5 m`; geodesic distance (A* on the same eroded map) `< 10 m`, path exists.
3. Camera forward-axis angle between init and goal `< 60 deg` (yaw difference of the base plus
   the pitch difference from the trunk).
4. `(t1, k1)` sampled from `generation.goal_translate_range` x `goal_tilt_range` (defaults
   [0, 1.25] x [-0.5, 0.5] rad) and clamped to joint limits; both init and goal configurations
   are whole-body collision free with 2 cm margin.
5. Renders (seg_instance) at goal and init: `>= 3` non-structural objects visible at the goal
   (each covering >= 200 px); `>= 30 %` of the goal-visible objects are visible from the init.
6. `pose_targets.agent.n.01_1` = goal camera sensor pose (link = sensor key), tolerances from
   `generation.tolerances` (defaults 0.05 m / 0.10 rad); goal not satisfied at init (camera
   error > tolerance). `pose_settled agent`.

Efficiency: candidate pairs are drawn from the map (cheap), filtered by 2-4 analytically, and
only survivors are rendered (two renders per pair).

### 5.2 Object scaling / mental rotation (`object_scaling-0`, `mental_rotation-0`)

Shared init ("held object"):

1. Robot in the scene at a collision-free base pose (tucked-derived: trunk at JoyLo translate
   0.5, tilt 0; left arm `ARM_READY_Q` mirror; right arm from IK). The right end-effector is at
   the **canonical held pose** `H` expressed in the base frame: the pad centre lies on the head
   camera's optical axis at depth `generation.init_depth` (default 0.55 m), approach axis (eef
   +z) horizontal pointing to the robot's left (+y_base), closing axis (eef y) vertical so the
   gripper is parallel to the ground. `H` and its IK solution are solved **once** for the robot
   model; the same 28-D configuration (minus the base virtual joints) is written to every
   instance's `robot_state`. If IK fails for the exact `H`, the closing axis is rolled about the
   approach axis in 15 deg steps until it succeeds (the first solution is then the canonical one
   for all instances). Measured: the head camera sits ~0.3 m ahead of the base and JoyLo's tilt
   is clamped by `torso_joint3`, so a point 0.55 m along the axis is ~0.75 m ahead of the base --
   beyond the right arm at every trunk. The solver therefore also walks `init_depth_fallbacks`
   (0.50, 0.45, 0.40, 0.35 m) and `hold_trunks`; the test objects hold at depth 0.50, trunk
   (0.5, 0), roll 15 deg. The realised depth goes to `template_meta.hold`.
2. Grasp: a library grasp `g` (eef pose in object frame, `T_obj_eef`) is chosen per instance,
   reviewer-clean rows before `one_finger`-flagged ones. The object is teleported to
   `T_eef(H) * inv(T_obj_eef)` with gravity disabled and kept still. The gripper is **not** simply
   commanded to close (v1.0-1.2): the two R1Pro finger drives are unequal, the first pad shoves
   the free object into the other pad and the realised grasp ends 1-3 cm / 9-14 deg from the
   validated pose with one pad off the object (the library's `one_finger` artefact: slanted
   bottle, gap on the toy figure). Since v1.3 each finger is advanced 1 mm per control step
   (`finger_step_m`, teleport + matching drive target) until it touches the object, the contact
   opening is held with a 4 mm press (`finger_press_m`; assisted grasping reads the *current*
   contacts when it creates the joint, both pads must touch for `m.GRASP_WINDOW`), the object is
   re-posed every step until the joint exists, then gravity is re-enabled and the object must
   still be held after 30 steps; the fingers are left at the contact positions. Measured:
   `summary.grasp_realised` = object pose in the eef frame within 0.3-4 mm / ~1 deg of the
   library pose, gravity adds < 1 mm. Grasps whose pads cannot both reach the object are
   rejected (`grasp_failed`). The realised `T_eef^-1 T_obj` is written to
   `grasps[0].object_pose_in_eef_frame`; the finger joint positions go into `robot_state`.
3. Self-occlusion: `seg_instance` renders with and without the robot hidden give
   `visible / reference` pixel ratios of the object; both init and goal configurations need
   `>= 40 %` visible (i.e. `<= 60 %` occluded by arm / gripper) and the whole object inside the
   image.

Goal options are precomputed **per template** (base-frame quantities are base-invariant): for
every library grasp with a fully visible held pose, `generation.goal_candidates` goal poses are
tested for IK (`right_arm`, else `right_arm_torso`; embodiment recorded) and for full visibility
from the **goal-time** camera (torso solutions move the head); the feasible ones are kept with
their 28-D solution. An instance draws a base pose, a grasp (grasps whose assisted grasp attached
before are preferred: AG attachment fails for some library rows in this arm posture) and one
option, re-checks whole-body collision at that base, then establishes the grasp. The goal object
pose and the target are re-derived from the **realised** eef->object transform (mm from the
library pose since v1.3) and the goal configuration's eef FK; the goal check poses arm + object
together, takes one physics step, and verifies visibility, self-occlusion, `PoseReached` and the
full goal expression (a `grasped` atom reading false right after the teleport is logged, not
rejected).

Object scaling goal: depth `d1` on a 5 cm grid over `generation.goal_depth_range` (default
[0.35, 0.85] m) with `|d1 - d0| >= generation.min_depth_change` (default 0.10 m), nearest first;
target = object root position at depth `d1` along the same camera ray, orientation unchanged.
`pose_targets` frame = camera sensor, `position_metric = depth`, tolerance
`generation.depth_tolerance` (default 0.03 m). Goal not satisfied at init follows from
`min_depth_change > tolerance`.

Mental rotation goal: rotation of the object about a sampled axis (uniform on the sphere in the
robot base frame, or `generation.axes`) by an angle in `generation.goal_angle_range` (default
[30, 120] deg), position unchanged (object root stays; the eef orbits). `pose_targets` frame
world, `orientation_metric = full`, tolerance `generation.orientation_tolerance` (default 0.15
rad), equivalences empty unless the spec lists them.

### 5.3 Disembedding (`disembedding-0`)

1. Two trays on the support, long axes parallel to the support's x/y axes, equally distant from
   the front edge (`generation.tray_edge_offset`, default 0.08 m from the front edge to the tray's
   near edge), fully supported. Source tray centred in front of the robot; destination tray to
   the side. Stance search order (v1.3): round 1 keeps the robot **square to the edge** and, for
   every edge (spec's `table_front_edge` first, then by length) and source lateral offset
   (`generation.source_lateral_offsets`), bisects the destination tray's *lateral* position
   between the closest non-overlapping and the furthest supported position until exactly
   `generation.dest_visible_width` (default 0.04 m) of its top face is inside the image. Round 2
   (fallback, logged as a warning, `template_meta.yaw_adjust_deg`) puts the tray at the furthest
   position and turns the **base yaw** away from it (bisection, <= 60 deg) instead. In both
   rounds the source tray stays fully in view (5 % margin), at least a quarter of its floor keeps
   a target-sized box in view, a seg render confirms the destination is visible but small
   (`dest_max_visible_fraction` of the source tray's pixels), and the grasp / placement
   feasibility tables of step 2 must be non-empty - a stance whose placement pose is blocked
   (chairs) falls through to the next option. On the test dining table the chair-free ends are
   the only usable edges and offer no lateral room, so the 10.5 deg yaw fallback is what makes
   "destination barely visible" satisfiable there; a chair-free long edge or counter gives a
   square stance.
2. Robot: canonical table stance (3.8) in front of the source tray, untucked arms (compact
   `ready` arms if the untucked hands collide with the trays), standing trunk preferred
   (`view_prefer_translate` 0; the optical axis may miss the tray centre by up to
   `view_max_angle_deg` = 20 deg because exact centring needs configurations that lean over the
   edge). Reachability: the library-grasp combination table over the source tray's inner floor
   (1 cm raycast raster) **and** placement feasibility at the destination centre from the stance
   shifted sideways in front of the destination tray (facing the edge); only (cell, yaw, grasp)
   triples that pass both are used.
3. Distractor set: chosen **per template** (the BDDL lists the group members, so all instances
   of a template share the set): `n_distractors ~ U{1..max}` with
   `max = min(generation.max_distractors, |pool|)`; with probability
   `generation.same_category_prob` the pool is the target category's **same-category pool**
   (`distractors.pools[C].same_category` minus the target), otherwise its **other-category pool**
   (the pools of `distractors.pools[C].other_categories`, i.e. the visually similar categories of
   the same family: cartons, packs, bottles, vessels, small_items); unique models. Models in a
   **look-alike group** with the target (`distractors.lookalike_groups`: exact duplicates and
   same-shape variants that differ only by a small print / label / shade, judged from the init
   view where objects are 20-130 px tall) are removed from both pools, so no distractor can be
   confused with the target. An empty requested pool (single-model categories; `toy_dice`,
   `cookie_cutter`, `tortilla_chips`, whose members all look alike) falls back to the other pool
   (`template_meta.distractors.fallback`); `--variant same_category` / `other_category` force the
   pool. The draw stops when the canonical footprints (1 cm margin) of target + distractors would
   exceed `generation.distractor_area_fraction` (0.45) of the source tray's footprint
   (`template_meta.distractors.footprint_fraction`) - three large boxes on a 30 x 44 cm tray
   failed every placement. `generation.templates_per_scene`
   replicates the template (`__v1`, `__v2`, ...) so that several sets are drawn.
   Layout per instance: target at a reachable cell with a random yaw (from the grasp combination
   table, also required to be placeable at the destination centre from the laterally shifted
   stance), distractors anywhere on the source tray's inner floor raster (1 cm, from raycasts),
   footprints non-overlapping (>= 1 cm); unoccluded instances keep distractors out of the 40 deg
   cone between the target and the camera.
4. Occlusion: with probability `generation.occlusion.prob` the instance is *occluded*: the
   external occlusion ratio `1 - visible / reference` of the target from the init camera (seg
   render with distractors visible vs hidden) must lie in `generation.occlusion.ratio_range`
   (default [0.30, 0.70]) with `>= generation.occlusion.min_visible_pixels` visible. The first
   distractor is placed on the camera->target ray, in front of the target at the closest
   non-overlapping distance, then its lateral offset is bisected (<= 6 renders) until the ratio
   is in range. Unoccluded instances require ratio `< 0.10`. The realised measurement goes to
   `initial_occlusion`.
5. Goal: target `ontop` the destination tray at the reachable floor cell closest to the tray
   centre (from the shifted placement stance; `template_meta.dest_place_offset_m`), init
   orientation; feasibility by teleport + settle. `pose_targets.target` (measure only) tolerances
   `generation.tolerances` (0.03 m / 0.20 rad).

### 5.4 Configuration matching / block design (`configuration_matching-0`)

Blocks: `toy_dice/isgblk`, a 1 inch (25.4 mm) cube, 6.9 g (ARAT 2.5 cm block density scaled by
volume), faces `+z red, -z blue, +x white, -x yellow, +y blue|yellow diagonal, -y red|white
diagonal`. Groups of 4 (2 x 2) or 9 (3 x 3, `generation.grid`).

1. Robot: canonical table stance, untucked, trunk solved for the arrangement region centre; the
   whole init scatter region and the goal footprint are inside the reachability grid (some
   translate) and inside the image with margin. No navigation needed.
2. Init scatter: blocks at reachable cells with >= 3 cm gaps, random yaw, **distinct up faces**;
   for 2 x 2 the two faces not on top are assigned to blocks where they are adjacent to the up
   face and yawed so they face the robot (side face normal within 35 deg of the camera direction),
   so all six patterns are visible without exploration (verified by frustum + face-normal test).
3. Goal layout: pattern `[f_TL, f_TR, f_BL, f_BR]` (top faces, robot's view: top = far, left =
   robot's left) from `generation.patterns` (or `--variant`). Slot 0 = init position of the block
   that is upper-left in the image, orientation = `R_yaw(table) * R_up(f_TL)` (canonical rotation
   putting that face up, block axes aligned with the table edges). Slots 1-3 touch slot 0 in a 2
   x 2 grid (pitch = edge + 1 mm) toward the robot's right and toward the robot. Blocks are
   `exchangeable`, `assignment.cost = layout`, equivalences empty (all faces differ). Goal
   feasibility by teleport + settle (blocks must not topple or drift).
4. Init must differ from the goal: at least one relation error above tolerance.

### 5.5 Configuration matching / articulation (`configuration_matching-1` open, `-2` close)

- Direction chosen per instance with `generation.open_prob` (problem 1 = closed -> open,
  problem 2 = open -> closed; a run of `-1` only produces "open" instances and vice versa).
- **Non-scene** objects (toolbox, carton, hinged_jar, ...) are placed on the support with the
  opening side toward the robot (lids: the hinge away from the robot), offset to the robot's
  right (`generation.lateral_center`, default -0.15 m: the right arm's reachable strip);
  canonical table stance with the trunk standing (`view_prefer_translate` 0: the camera stays
  high and behind the edge so a 0.6 m wide object fits the image), stand-off 0.45 m then 0.55 m
  (`generation.stand_offs`); the handle contact (`artic/handles/<key>.json` from the
  articulation pass: bar / knob / edge with pinch axis and approach) must be IK-reachable
  together with a 10 cm pre-grasp for some translate from the stance **or after driving straight
  forward** by up to 0.2 m (`generation.approach_forward`, recorded per instance as
  `approach_forward`): viewing a wide object and reaching its handle are incompatible from one
  spot; the whole object inside the image (exact hull, 3 % margin).
- Passive stability of the open angles is probed once per template (`open_fraction_range`
  sampled at 7 values + the upper limit, 120 physics steps each); instances only use angles
  that held.
- **Scene** objects (cabinets, fridges, ovens): robot untucked in front of the object at the
  distance where the handle's pre-grasp and grasp are IK-feasible for the right arm (torso
  translate solved so the eef height equals the handle height; the robot may drive straight
  forward up to 0.3 m, tested by sliding the base); whole object inside the image.
- Open: `articulation_targets.<obj>.joints` = every non-fixed joint at its init value except the
  chosen one at a target `q* ~ U(generation.open_fraction_range)` of its range (default [0.5,
  0.9]); feasibility: set `q*`, step 120 physics steps, it must stay within tolerance (the lid
  must not fall open/closed); otherwise the target is moved to the nearest stable value (joint
  limit) and the instance is kept only if `Open` is true there. Init: all joints closed.
- Close: init has exactly one joint open at `q0 ~ U(open_fraction_range)` (also stability
  checked); target = closed limits for every joint.
- Tolerances: revolute 0.10 rad, prismatic 0.02 m (`generation.tolerances`).

### 5.6 Configuration matching / attachment (`configuration_matching-3`)

1. Female part (bottle) upright on the support, male part (cap) lying on the support within
   reach, both in view; canonical table stance; pre-grasp of the cap **and** the attach pose
   (cap pose when attached, plus 10 cm above it) IK-feasible from the stance for some translate.
2. Goal: `T_cap_goal = T_bottle * T_F(bottle) * inv(T_M(cap))` from the `attachment` meta links
   (`AttachedTo._get_parent_candidates`), verified by teleporting the cap there and calling
   `AttachedTo.set_value(True)`; the resulting joint is the `attached` evidence. Group target:
   fixed assignment, relation `bottle -> cap`, slots = both goal poses; `attachment_targets`
   names the link pair.
3. Symmetry: `generation.orientation_equivalences` when given; otherwise the male part is tested
   for being revolute about its attachment axis (in every 5 mm slab along the axis the collision
   hull's radial distances vary by < 10 %), in which case `orientation_equivalences` lists the
   rotations about that axis in `generation.symmetry_step_deg` steps (default 15 deg, 23
   quaternions); the layout orientation tolerance must exceed half the step. The list applies
   to both group members (the bottle is revolute about the same axis).
4. Both parts are spawned with the `attachable` ability added (the taxonomy does not flag every
   model that carries attachment meta links, and `AttachedTo` is only instantiated for
   attachable objects). OmniGibson's `AttachedTo._attach` no longer filters collisions between
   child and parent, so an attached cap whose collision hull overlaps the neck jitters; the
   sampler filters the pair once per template before the goal check. **Open point for the
   evaluator**: do the same when the robot attaches the cap during an episode.

### 5.7 World alignment / table surface (`world_alignment-0`)

1. Target on the support at the centre of the image; a sufficient run of the support's front
   edge (>= 3 of 9 samples along it) inside the image; canonical stance. Placement region =
   cells whose footprint (after the sampled yaw) stays inside the support. Pushing reach is
   evaluated on the object's **ends** (long-axis extremes), not its centre: the reach raster
   covers the whole near strip, and a (cell, deviation) draw is kept only if one end lies on a
   reachable raster cell and passes the exact IK for a top-down push pose.
2. Constraint: a hinge (`constraints[0]`, type `revolute`, parent = support root link, child =
   target, anchor = target CoM, axis = world z) lets the object yaw in place; it is written as a
   TRO constraint and recreated by the loader after every reset (and by the sampler for its own
   settling checks). The object rests on the support (`ontop` true) with the hinge carrying no
   vertical load: the joint frame is placed at the settled height.
3. Init yaw = aligned yaw + `delta`, `|delta| ~ U(generation.deviation_range)` (default [20, 70]
   deg), sign random. Aligned yaw: object long axis parallel to the front edge, object "front"
   (the face closest to the keys' bottom row = canonical -y of the model, configured per
   category in `generation.front_axis`) toward the robot.
4. Goal: `pose_targets.target` orientation only, tolerance 0.10 rad; equivalences from the spec.

### 5.8 World alignment / world vertical (`world_alignment-1`)

1. Wall point: rays from the room's interior (1.5 m above the floor centroid, 24 yaw directions)
   hit `walls`; a hit is a candidate when a `w x h` patch around it (object size + 10 cm) is flat
   (grid rays hit within 1 cm of the same plane) and no other object's AABB intersects the box
   in front of it (depth 0.3 m). Object hung with its centre at `generation.height` (default
   1.45 m), face normal = wall normal, gravity irrelevant because the hinge anchors the CoM:
   `constraints[0]` revolute, parent = wall link, child = object, axis = wall normal.
2. Robot on an arc: per wall spot a lattice of 4 radii in `generation.arc_radius_range`
   (default [0.6, 0.95] m; the arm reaches the frame's lower corners from <= ~0.95 m) x 7
   bearings in `+-arc_half_angle` (default 35 deg) is filtered **once per template** for
   traversability, the spec room, whole-body collision freedom (untucked) and pushing reach of
   the aligned object's lower corners; spots without a valid base are dropped. Instances draw a
   base from the cached list, solve the trunk so the object centre is at the image centre and
   check the whole object inside the image. `>= 5` visible objects in the seg render (structural
   categories count here).
3. Init in-plane angle `delta`, `|delta| ~ U(generation.deviation_range)` (default [15, 45]
   deg): the pushing contact (the lowest corner of the frame at init) must be IK-reachable
   for the right arm from the chosen base (pre-push 20 cm and standoff 10 cm off the wall,
   approach into the wall).
4. Goal: orientation only (straight = world up along the object's canonical up), tolerance
   0.08 rad.
5. Object frame (v1.4.1): the object is hung with its `generation.front_axis` (default `+x`)
   along the wall normal and its `generation.up_axis` (default `+z`) along world up; both accept a
   dict keyed by `<category>/<model>`, `<category>` or `default`, because the canonical frames
   differ per category (paintings stand upright, thin x; pictures, license plates, guitars lie flat
   with their up along `-x` / `+y`; see `runlogs/world_alignment_selection_20260911/selection.md`
   for the reviewed frames of every selected model). The table-surface problem's `front_axis` /
   `long_axis` accept the same dict form.
6. Wall restriction (v1.4.1): `scenes.<scene>.fixed.wall` (list of wall object names) limits the
   binding, and `scenes.<scene>.fixed.wall_regions` (list of `{room, wall, p0, p1, normal, hang_z}`
   segments, world xy at the hanging height, normal pointing into the room) seeds the spots along
   the segments instead of casting rays from the room centre; every seeded point is still confirmed
   by a ray onto the wall and by the flat-patch / front-clearance tests. Regions surveyed and
   validated for the seven candidate scenes (one full instance per region) are in
   `runlogs/world_alignment_selection_20260911/wall_candidates.json` (`scenes.<scene>.task_spec` is
   the ready-to-paste block) with maps in `maps/<scene>_walls.png`. When several walls of a room
   have usable spots the template binds the wall with the most spots.

### 5.9 World alignment / world horizontal (`world_alignment-2`)

Same wall hinge, arc of base poses, pushing-reach, hinge-drift and visibility rules as 5.8 (the
sampler `WorldHorizontalSampler` subclasses the world-vertical one; the wall rows / regions of
5.8 are shared). The object is an elongated wall item with one clear horizontal direction (towel
bar, hook rail, shelf, wide sign / board / screen, or a rod displayed on wall hooks); it starts
tilted in the wall plane and the goal is the **level** pose. Differences:

1. Object frame: `generation.front_axis` (out of the wall, default `+x`) and `generation.up_axis`
   (world up when the object hangs correctly, default `+z`) as in 5.8, plus the optional
   `generation.level_axis` (`x` / `y` / `z`: the canonical axis that lies horizontal along the wall
   when aligned, the reviewers' reading) which must be the axis derived from the other two
   (`side = up x front`); a mismatch rejects the template (`spec`). All three accept the dict
   form keyed by `<category>/<model>`, `<category>` or `default`. Hung pose: level axis along
   the wall tangent, front axis along the wall normal, up axis along world +z.
2. Goal: orientation only, the level pose (rotation 0 about the wall normal), tolerance 0.08 rad.
   "Horizontal" is undirected, so `generation.symmetric_180` (bool or the same dict form,
   default false) adds the half turn about the object's front axis to the target's
   `orientation_equivalences` for models that look the same upside down (plain bars, blank
   screens, rolling pins); explicit `generation.orientation_equivalences` are appended. Models with
   a readable top / bottom (brackets below a shelf, hooks below a rail, text) keep the single
   upright level pose.
3. Init angle `delta`, `|delta| ~ U(generation.deviation_range_deg)` (default [15, 45] deg), sign
   random - the long axis is tilted from level; the pushing contact at init is the lowest part of
   the hull, i.e. the low end of the tilted object, and the goal contacts are both ends (as 5.8).
4. Wall patch for elongated objects (`generation.swept_patch`, default true): the flat-patch and
   front-clearance tests use the extents **swept** by the object between the largest init tilt
   and level (for the width-along-the-wall `w` and height `h` of the level object and
   `d = max deviation`: swept width = max over `a in [0, d]` of `w cos a + h sin a`, swept height =
   max of `w sin a + h cos a`, i.e. the full diagonal when the tilt reaches it) instead of the
   level extents of 5.8 (a 0.96 x 0.13 m rail at 45 deg needs a 1.0 x 0.8 m flat, clear patch
   rather than 1.06 x 0.23 m). A validated region of `wall_candidates.json` therefore admits a
   horizontal target only when its `max_object_m` covers the swept width and height;
   `template_meta.swept_dims` records them.
5. Object selection: `runlogs/world_horizontal_selection_20260912` (pyrender renders in the
   canonical frame, 8 reviewers, `review_contract.md`: V visual 0-5, L level readability 0-3, H
   hangability 0-2, score `2V + 2L + H`, valid = V >= 3, L >= 2, H >= 1; rounds 1-3 as for 5.8;
   frames `front / level / up / thin` and the `symmetric_180` flag per model in `selection.md`,
   spec-ready dicts in `spec_fragments.json`), recorded in
   `runlogs/grasp_library_20260905/select/task_object_selection.csv` (rows
   `world_alignment-2 (world_horizontal)` / `target`, `wall`, `floor`; the wall rows are the 78
   validated regions of 5.8 annotated with the picks that fit each region).

### 5.10 Inter-object relation satisfaction (`inter_object_relation_satisfaction-0..4`)

Common: canonical table stance; trunk solved for the *target* (the object that moves); both
objects fully inside the image; the target's pre-grasp reachable (or, for `touching` / `nextto`,
pushable: a horizontal push pose at the target's near face reachable). Pose targets for both
objects are measured only (`:measure`), tolerances 0.03 m / 0.20 rad.

- **ontop, small on small** (`-0`): both on the support; goal = target root at the same
  orientation, positioned above the partner's top surface centre (z from the partner's top face
  + target bottom offset), feasibility by drop + settle; `OnTop` true, partner unchanged.
- **under, small under big** (`-1`): partner = scene bed. Receptacle search in order: (a) the
  spec's table categories within 3 m geodesic of the bed's near side such that from its
  canonical stance both the target (on the receptacle) and the bed are inside the image and the
  target is reachable; (b) the top of the bed itself (target `ontop` the bed; the `table` role
  aliases the bed). Stand-offs (`stand_offs`) and, with `try_all_edges`, every edge of the
  support are tried until reachable cells exist; the stand-off is measured from the support's
  AABB when its top raster is inset (bed frames), and the usable surface is the support's
  largest horizontal face (mattress), not its AABB top (headboard). Measured on
  `bed_gxfipj_0` (bedroom_0): the arm reaches only the 2-6 cm strip along the mattress edge
  from a 0.35 m stand-off (too narrow to support the toy) and crouched trunks collide with the
  frame; the test spec therefore uses childs_room_0 (toy on `desk_xhjsub_0`, goal under
  `bed_rrcvaq_0`). The floor-near-the-bed variant (c) of the original plan is not implemented.
  Goal = the nearest floor point under the bed (footprint shrunk by the target's half extent +
  5 cm) where rays cast upward from the floor hit the bed above `target height + 3 cm`; same
  orientation. `Under` verified.
- **inside** (`-2`): goal = target root centred on the partner's `fillable` volume centroid (xy)
  resting on the volume floor (drop + settle), `Inside` verified, partner unchanged.
- **nextto** (`-3`): left object (from the robot) = partner, right object = target. Goal = target
  at the same orientation, translated so the AABB gap to the partner along the table's lateral
  axis is `generation.nextto_gap` (default 0.02 m; must be < mean(dimsA + dimsB) / 6, the
  `NextTo` bound) and its centre height overlaps the partner's AABB height range; `NextTo`
  verified after settling.
- **touching** (`-4`): goal = target translated toward the partner until the AABBs touch, then
  refined in 1 mm steps with a physics step each until `Touching` (rigid contact) is true and
  the partner has not moved > 2 mm; same orientation.

## 6. TRO contents by task (in addition to the common block)

| Problem | `pose_targets` / `groups` / `articulation_targets` | restoration records |
| --- | --- | --- |
| camera_relocalization-0 | agent camera pose (full) | - |
| object_scaling-0 | target depth in camera frame | `grasps` |
| mental_rotation-0 | target orientation (world) | `grasps` |
| disembedding-0 | target full pose (measure) | `initial_occlusion` |
| configuration_matching-0 | group slots + relations (layout gates) | - |
| configuration_matching-1/2 | joints of the articulated object | - |
| configuration_matching-3 | fixed group (bottle, cap) + `attachment_targets` | `attachment_targets` |
| world_alignment-0 | target orientation | `constraints` (revolute, parent table) |
| world_alignment-1 | target orientation | `constraints` (revolute, parent wall) |
| world_alignment-2 | target orientation (+ 180 deg equivalence for symmetric models) | `constraints` (revolute, parent wall) |
| iors-0..4 | both objects full pose (measure) | - |

Generation details that are not part of the evaluation schema (sampled parameters, rejection
counts, measured visibility, timings) go to the `.summary.json` next to the TRO so that the
schema validator (`ISGTroMeta.KNOWN_FIELDS`) stays strict.

## 7. Efficiency plan and what is measured

- Per template: scene load (partial rooms), cuRobo planner build (`right_arm`, kinematics-only
  `whole_body`; the torso embodiment only as a 0.2 s batched IK solver), world load,
  reachability grid, camera solution. Reported in `template_meta.json.timings` together with
  the IK / collision call counters of each phase (`*_kit`).
- Per instance: wall time split into sample / apply / settle+validate / goal-check phases
  (`summary.phase_s`) and the IK / collision counters (`summary.kit`).
- Measured (RTX 4090, house_single_floor, 2 instances per template): scene load 35-45 s per
  scene; template precompute 15-80 s cold, ~5 s with the on-disk cache (`ISPATIALGYM_SAMPLING_CACHE`);
  instances 3-8 s (table tasks), 4 s (held object), 7 s (camera relocalization). The 15 test
  problems take ~20 min cold end to end (v1.0: ~30 min) plus ~15 min for the reload validation.
- Failure handling: an instance attempt that fails a check is discarded with its reason counted
  (`summary.rejections`); `--max-attempts` bounds attempts per instance. Attempts are made
  productive where possible (disembedding falls back to an unoccluded instance when there is no
  room for an occluder; distractors that occlude are re-drawn before the attempt is dropped).

## 8. Assumptions and open points

- "Untucked" is R1Pro's `untucked_default_joint_pos` (hands high and wide). If the action script
  prefers `ARM_READY_Q` as the working pose, only `generation.arm_pose` changes.
- Object scaling / mental rotation: the eef (pad centre) lies on the optical axis; the object
  centre is then within the grasp's CoM offset of the axis (<= 8 cm for the test objects) and is
  checked to be fully inside the image. Init depth is a template constant; per-instance
  variation comes from the grasp and (for scaling) the goal depth.
- The block-design goal orientation of each block is the canonical `R_up(face)` yaw-aligned
  with the table; the goal image is what disambiguates yaws for the policy.
- Attachment symmetry is detected geometrically and expressed with discrete equivalences.
- Articulation targets on horizontal-hinge lids are only accepted where the lid holds the angle
  passively; otherwise the fully open limit is used (the toolbox lid falls fully open from every
  angle in [85, 170] deg, so its only open target is the 170 deg limit).
- The scene-object articulation path is implemented but has no test object in this round.
- Renderer workaround: Isaac Sim 5.1's synthetic-data graph (`seg_instance`) segfaults
  deterministically on the third rendered frame when a light fixture (`room_light` in the dining
  room, `downlight` in the living room) and glass (sliding doors / windows) are both in the head
  camera's view (bisected object by object; independent of the arm configuration and of the
  objects on the table). The sampler hides every object whose category contains `light`, `lamp`
  or `chandelier` while it runs (`ISG_HIDE_CATEGORY_SUBSTRINGS`) and shows them again around
  `scene.save`, so templates are unaffected. Oracle-mode goal renders will need the same guard or
  a renderer fix.
- Loader gaps (not the sampler's): `ISpatialGymTask.load_tro_instance` does not yet restore
  `grasps` (assisted grasp of the held object), `constraints` (COM hinges) or the attachment
  collision filter; `validate_instances.py` reports the dependent checks as skipped.
