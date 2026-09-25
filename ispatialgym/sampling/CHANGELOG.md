# Instance sampler changelog

Version = `ispatialgym.sampling.SAMPLER_VERSION` (also written into every `sampling_report_*.json`).

## 1.5.0 (unreleased) - one arm pose per task, scene variants without chairs, goal check without the robot

- **Camera relocalization bucket = the geodesic** (2026-09-24). The four bucket decisions (goal sampling, the
  per-cell obstacle try, the final blocked-map path, the re-plan after the settle) read the eroded-map shortest-path
  length `kit.geodesic` returns (spec §2.3 / §6.1), not `paths[].length_m`: since the record became the line-of-sight
  shortcut (2026-09-22) its polyline is up to ~8 % shorter than the grid geodesic the pilot's dense records had, so an
  instance near a bound could change bucket. `path_length_m` stays in the summary as information; the goal candidate's
  bucket is now checked before its path is collision-checked.
- **World horizontal retired** (2026-09-23). The `alignment-world_horizontal` subtask (v1 `world_alignment-2`, alias
  `world_horizontal`) is removed everywhere: the v2 sampler `alignment_wall.py` registers world vertical only (the
  horizontal `swept_patch` extents, `level_axis` frame key and the `V1_DEFINITION_ID` 2 are gone), the v1
  `WorldHorizontalSampler` / `bddl_writer` definition 2, the oracle program / sidecar / program-check registries,
  `generation_defaults`, `v2.problem.SUBTASK_FAMILY`, `v1_bridge`, `ground_isg` (`WALL_SUBTASKS`,
  `SCALE_FALLBACK_SUBTASKS`, `frame_level` is no longer read), the iddl file, the 22 alignment BDDL problems
  (legacy `problem2` + 21 grounded aliases; the remaining alignment problems are renumbered contiguously and the
  customs `definition_id`s follow), the 21 customs entries, the `isg_reps` v1-bridge snapshot, the icl candidate /
  representative rows (+ report) and the revolution-axes table entries only that subtask used. 20 subtasks remain.
- **Neighbouring rooms loaded while sampling, nothing hidden for renders** (B10). `SimWorld.load_adjacent` defaults
  to on (`ISG_LOAD_ADJACENT=0` opts out), so the sampler, evaluation and the oracle load the same room set (collision
  and visual world parity). The v1 `seg_instance` crash (light fixture + glass in view; kitchen / garden with the
  neighbours loaded) was bisected to the RTX flag `/rtx/raytracing/showLights` that OmniGibson enables at launch: the
  light-source shapes it draws get instance ids the replicator's `InstanceSegmentationReduction` node cannot map
  (`unlabelled` regions, then an out-of-bounds index). `SimWorld._set_measurement_render_settings` sets the flag to 0;
  the light-fixture hiding (`ISG_HIDE_CATEGORY_SUBSTRINGS`, default now empty) is an emergency override only, and
  label maps no longer contain `unlabelled` regions in the affected views. Measured on house_single_floor with the
  neighbours loaded: 8 living-room views x 4-6 renders crash-free; the v1 guard still crashed on the kitchen view,
  the viewer-camera workaround of BEHAVIOR-1K #2312 and translucency off did not help. Spec 3.1 / 3.4.
- **Settled segmentation reads.** A sensor read returns a buffer one to two rendered frames old (load-dependent), and
  a scene change reaches the renderer one frame late, so v1's "three renders then read" in `SimWorld.seg_instance`
  reported the state of the *previous* measurement when measurements followed each other (the distractor-hidden
  reference silhouette equalled the occluded one; the tray counts read next carried the hidden state).
  `seg_instance` now renders >= 4 frames and continues one frame at a time until two consecutive raw-buffer fetches
  are identical (cap 8), then reads; `rgb()` reads after the same settling. ~0.3 s per measurement instead of
  ~0.17 s. Same-pose measurements are now identical across repeats and across room sets. Spec 3.4.
- **World horizontal (`world_alignment-2`, alias `world_horizontal`)**: new problem (BDDL template
  `world_alignment/problem2.bddl`, `bddl_writer`, `WorldHorizontalSampler` in `tasks/world_alignment.py`, spec entry,
  docs spec §5.9 / `task_bddl_formats.md` §6c). Same wall hinge, arc and pushing rules as world vertical; the target
  is an elongated wall item whose long axis / reference edge must end up **level**. Differences: the optional
  `generation.level_axis` is checked against the frame derived from `front_axis` / `up_axis`;
  `generation.symmetric_180` (bool or per-model dict) adds the half turn about the front axis to the target's
  `orientation_equivalences` (a plain bar is level either way); the flat-patch / front-clearance tests use the
  extents swept by the object between the largest init tilt and level (`generation.swept_patch`, default true;
  `template_meta.swept_dims`), because a tilted 1 m rail needs a 1.0 x 0.8 m clear patch, not 1.06 x 0.23 m.
  `_HingedSampler.axis_param` now builds on `target_param` (same per-model lookup for non-axis values).
- **Data**: `runlogs/world_horizontal_selection_20260912` - object rounds 1-3 for world horizontal (135 candidate
  models of 40 elongated wall / rod categories, 134 rendered, 8 reviewers, `review_contract.md` with V / L / H scores,
  `selection.md`: 18 categories / 69 valid models), recorded in `select/task_object_selection.csv` (+ xlsx / html) as
  `world_alignment-2 (world_horizontal)` target / wall / floor rows; the wall rows reuse the 78 validated regions of
  the 1.4.1 survey, annotated with the picks whose swept extent fits each region. Test spec pins
  `towel_rack/{kqrmrh, crivoa}`; smoke-tested on the Rs_int entryway region (towel bar and hook rail, one instance
  each, renders in the runlog's `smoke/`).
- **Disembedding occluded instances were impossible**: `_tune_occluder` looked the target's hull / radius up under
  its BDDL instance name (they are stored under the role key `target`) and treated the trays as obstacles, so every
  occluded draw raised `KeyError` (all 11 generated disembedding instances so far are unoccluded). Found by the
  distractor validation (`runlogs/disembedding_distractors_20260911`); fixed.
- **`generation.init_arms`** (default `untucked`, BEHAVIOR's reset pose) fixes the init arm configuration per task;
  every sampler routes through `TaskSampler.init_arms()` (camera relocalization used tucked, the held tasks' free arm
  used ready before). `generation.view_arm_fallbacks` is empty by default: no silent fallback to `ready`, a stance that
  collides with the chosen arm pose rejects the template instead, so all instances of a task share the arm pose.
  `template_meta.init_arms` records it. Within a template the whole robot configuration except the base is identical
  across instances (trunk translate / tilt come from the template's view solution).
- **Scene variants** (`sampling/scene_variants.py`). Table tasks (a fixed `table` role, or
  `generation.remove_chairs_near_tables: true`) sample in the stable scene minus the chairs within
  `generation.chair_distance` (1.0 m) of the work table(s) (`CHAIR_CATEGORIES`; `generation.chair_categories`
  overrides). The variant json is written to `<out>/scene_variants/<scene>__<hash>.json` and is the scene file the
  sampler loads, templates are merged from it (`template.json` has no chairs) and the reload checks compare against
  it. The traversability PNG is baked with the furniture, so the removed objects' footprints (canonical bounding box,
  yaw, `base_link_offset`) are freed in the loaded map where `floor_trav_no_obj_0.png` has floor and re-blocked under
  remaining objects (the table); `template_meta.scene_variant.freed_footprints` records them and
  `scene_variants.patch_trav_map_from_meta(scene, meta)` applies the same patch at evaluation time (the validator
  does; the oracle's base planner must, or it treats the chair spots as blocked). On the test dining table 6
  `straight_chair`s are removed (64 traversability cells freed outside the table footprint); the disembedding stance
  is then square to the edge (`yaw_adjust_deg` 0) and the two v1.3 validation failures are gone - the destination
  tray had been resting against the back of `straight_chair_jgoolz_1` (probe: the only non-table object within 2 cm
  of the tray), which only showed as drift with the neighbouring rooms loaded.
- Disembedding places the target on the destination tray at the reachable floor cell closest to the tray centre
  (`template_meta.dest_place_offset_m`) instead of exactly the centre: the arm's reach over a table is a narrow band
  at a fixed depth from the edge and whether it crosses the centre depends on the tray's depth; every long-edge stance
  had failed with `unreachable` because of it.
- The generic goal check parks the robot (tucked, room centre) while the goal objects are teleported and settled:
  the goal must be feasible *excluding* the robot, and a hand hovering where the goal object lands ejects it. Seen
  in disembedding after the chairs were removed: the square stance put the destination tray 0.65 m to the right,
  exactly under the right hand of a 51 deg forward-tilted torso, and 54 of 55 goal checks threw the box on the floor.
  Disembedding records whether a hand hovers in the column above a tray (`template_meta.hands_over_tray`;
  `generation.require_hands_clear` turns it into a rejection - on the 1.16 m table end with 1.2 m of untucked arm span
  it rejects every stance, so it is off by default).
- Neighbouring rooms: gardens are no longer excluded (`ISG_ADJACENT_EXCLUDE` default empty) - rgb rendering with the
  garden loaded is fine, only the sampler's `seg_instance` renders crash with it in view. A "load the neighbours but
  hide their prims" sampler mode was tried and dropped: hiding ~200 prims leaves the instance-segmentation annotator
  with no labels at all (every pixel `unlabelled`), re-creating the annotator does not recover it. The remaining way
  to keep a base off open-plan room boundaries is `RobotKit.trav_free(in_task_rooms=True)` (opt-in).

## 1.4.1 (2026-09-11) - world alignment: per-category object frames, spec wall regions

- **World vertical (`world_alignment-1`)**: the hung object's frame is no longer assumed to be "thin x, upright";
  `generation.front_axis` (axis out of the wall, default `+x`) and the new `generation.up_axis` (default `+z`) define
  it, and both (like the table-surface `front_axis` / `long_axis`) accept a dict keyed by `<category>/<model>`,
  `<category>` or `default`. `_hang_quat` maps (front, side, up) onto (wall normal, wall tangent, world up);
  `_object_dims` measures thickness / width / height along those axes. Behaviour for the default frame (paintings)
  is unchanged. Needed by the round 1-3 world-vertical categories whose canonical frame lies flat (picture,
  license_plate, guitar, hanger, dartboard, fire_extinguisher, ...; frames per model in
  `runlogs/world_alignment_selection_20260911/selection.md`).
- **Spec wall restriction**: `scenes.<scene>.fixed.wall` (allowed wall object names) and
  `scenes.<scene>.fixed.wall_regions` (surveyed segments `{room, wall, p0, p1, normal, hang_z}`) restrict / seed the
  wall spots; without them the 24-ray search from the room centre is used as before. When several walls of a room
  have usable spots, the template now binds the wall with the most spots (was: the first hit).
- **Data**: `runlogs/world_alignment_selection_20260911` - object rounds 1-3 for both world-alignment problems
  (pyrender renders of 459 models in their canonical frame, 8 reviewers, `review_contract.md`, `selection.md`) and
  the wall survey of every indoor room of the 7 scenes (`wall_candidates.{json,md}`, `maps/`), recorded in
  `select/task_object_selection.csv` (+ xlsx / html).

## 1.4.0 (2026-09-12) - disembedding distractor pools per target category, look-alike exclusion

- **Spec: `distractors` section replaces the `same_category_pool` / `distractor_pool` roles** of `disembedding-0`
  (`spec.DistractorSpec`, spec §2.2 / §5.3). `pools.<category>.same_category` lists the models of that category that may
  be drawn as same-category distractors, `pools.<category>.other_categories` names the categories whose pools form the
  other-category pool, `lookalike_groups` lists `<category>/<model>` sets that cannot be told apart from the robot's
  initial view, `families` documents how `other_categories` were derived. The loader checks that every pooled model
  is a ready model of its category, that `other_categories` reference pools, that look-alike members are pooled (or
  targets) and that every target category has a pool. `ProblemSpec.distractor_spec()` parses the section lazily from
  `raw` for scripts that rebuild a `ProblemSpec` field by field (`validate_tables.py`).
- **Sampler**: for a target `C/m` the same-category pool is `pools[C].same_category` minus `m` **and minus every model
  that shares a look-alike group with `m`**; the other-category pool is the union of the pools of
  `pools[C].other_categories`, again minus look-alikes (groups may cross categories). An empty requested pool falls
  back to the other one (logged; `template_meta.distractors.fallback`). `--variant same_category` /
  `--variant other_category` force the pool. `template_meta.distractors` records pool, requested pool, pool size,
  pool categories and the excluded look-alikes next to the drawn models.
- **Data**: `runlogs/grasp_library_20260905/scripts/build_disembedding_pools.py` builds the section for all 39
  disembedding categories (242 pooled models = every AG-valid disembedding model except one V<=1 asset; 5 visual
  families: cartons, packs, bottles, vessels, small_items; 55 look-alike groups = the exact-duplicate groups of
  `select/duplicates.md` plus same-shape variants that differ only by a small print / label / shade, judged on the
  cluster renders and grasp strips) and records everything in `select/disembedding_distractor_pools.json` and in
  the `same_category_pool` / `distractor_pool` rows of `select/task_object_selection.csv` (+ xlsx / html; every
  disembedding target row carries its pool sizes in `pair` and a `LOOK-ALIKE` flag in `duplicates`).
  Same-category pools that are always empty (single-model `jar_of_bath_salt`; `toy_dice`, `cookie_cutter`,
  `tortilla_chips`, whose members all look alike at ~20-30 px) always fall back to the other-category pool.
- `sample_instances.py` replicates templates (`templates_per_scene`) for problems with a `distractors` section (the
  pool roles that used to trigger the replication are gone).

## 1.3.0 (2026-09-12) - realised grasps, neighbouring rooms, square disembedding stances

- **Held-object grasps are realised at the library pose.** Previously the gripper was simply commanded to close on
  the free-floating object: the two R1Pro finger drives are not equally strong, the first pad shoved the object into
  the other pad and the object ended 1-3 cm / 9-14 deg away from the validated grasp with one pad off the object (the
  library's `one_finger` artefact; visible as a slanted bottle / gap on the toy figure). `_establish_grasp` now advances
  each finger by 1 mm per step until it touches the object (object kept still), holds the contact opening with a 4 mm
  press while the assisted-grasp window elapses, and leaves the fingers at the contact positions. Measured: object
  pose in the eef frame within 0.3-4 mm / 1 deg of the library pose (`summary.grasp_realised`), fingers symmetric
  (e.g. [0.021, 0.016]). Reviewer-flagged (`one_finger`) grasps are only used when no clean grasp has an option;
  grasps whose pads cannot both reach the object are rejected (`grasp_failed`) and the per-template statistics steer
  away from them.
- **Neighbouring rooms.** `SimWorld.adjacent` (rooms sharing a wall / doorway with the task rooms, from the scene's
  room-instance map dilated by 0.6 m; outdoor rooms excluded) is recorded as `template_meta.load_room_instances`
  together with the task rooms, so evaluation, validation and rendering load the neighbours (walls and furniture
  behind doorways and windows instead of sky). The sampler itself keeps loading the task rooms only
  (`ISG_LOAD_ADJACENT=1` to change): Isaac Sim 5.1's synthetic-data graph segfaults on house_single_floor as soon as
  the kitchen or garden are in the head camera's `seg_instance` view (and when their prims are hidden); bisected with
  probes, not fixable from our side. `validate_instances.py` renders rgb only for the same reason. Category-based
  fixed roles (tables, paintings) are restricted to the task rooms; cuRobo obstacles are filtered to the task rooms'
  extent + 1 m (`SimWorld.obstacle_filter`, `R1ProCuRobo.obstacle_filter`) - with the neighbours loaded the mesh cache
  (2048) overflowed.
- **Disembedding.** Round 1 of the stance search keeps the robot square to the edge on every edge (preferred first,
  then by length) and bisects the destination tray's *lateral* position until exactly `dest_visible_width` of it is in
  view; the base-yaw fallback is round 2 and is logged as a warning (`template_meta.yaw_adjust_deg`). The grasp /
  placement feasibility tables are now part of the stance acceptance, so a stance whose placement pose is blocked
  (chairs along the long edges of the dining table) falls through instead of failing the template. The distractor set
  is drawn within a footprint budget (`distractor_area_fraction` 0.45 of the source tray) - replicated templates used
  to fail every placement with three large boxes. On the test dining table the ends are the only chair-free edges and
  the 10.5 deg yaw fallback is still what makes "destination barely visible" satisfiable.
- `render_instances.py` defaults: 640 px tiles and 10 render iterations (384 px / 4 iterations looked blurred; the
  RTX real-time denoiser needs a few frames after a teleport).
- Template setup starts from a clean world: non-task scene objects are reset and the robot is parked (tucked, room
  centre) before the fixed roles are bound. Left at the previous template's stance, the robot's untucked arms hovered
  over the table and the support raster's downward rays hit them (32 "occupied" cells on the dining table), so the
  disembedding source tray had no supported position after the attachment template - a template-order dependence
  present since v1.0.
- `validate_instances.py` steps three times before evaluating `:init` (the robot base starts a few mm above the floor
  and `ontop agent floor` needs the contact; with the neighbours loaded the disembedding stance, whose rear wheels
  rest on the living-room floor across the open-plan boundary, read false after one step and true after three).
  `RobotKit.trav_free(in_task_rooms=True)` can additionally require the whole base footprint inside the task rooms
  (off by default: it removed every disembedding stance on the test table).
- Spec: §3.2 notes that BEHAVIOR's `sample_kinematics` orientation sampling reduces to canonical-up + yaw for the
  2026 assets (every `metadata.json` has `orientations: []`), §3.3 robot facts, §5.2 grasp realisation, §5.3 stance
  search order.
- Status: all 15 problems sample (30 / 30 instances, 7 min of template time after the scene loads); reload
  validation with the neighbouring rooms 28 / 30. Open: the two disembedding instances report the destination tray
  drifting 1.2 cm / 2.6 deg over 60 steps when the neighbours are loaded (identical tray poses validated with zero
  drift in v1.1 with the task room only) - to be probed for a neighbouring-room prim touching the tray.

## 1.2.0 (2026-09-11) - scene-reset guarantee, on-disk evidence, init renders

- **Every attempt starts from the stable scene.** `SimWorld.reset_scene_objects` reloads the state of every non-task
  scene object from the scene json before each attempt and right before `template.json` is written, so instances differ
  from the scene json only in the task-relevant objects and the robot. Previously only the 5 cm / 0.05 rad
  `check_scene_unchanged` guard existed; smaller creep did leak into templates (measured on the v1.1 outputs: a living
  room floor lamp 3-5 mm off in four templates, everything else identical). Reset counts are logged
  (`rejections.scene_reset`).
- `validate_instances.py` compares every non-task object of `template.json` with the stable scene json (1 mm /
  0.005 rad) and fails the template otherwise (`template_scene_deviations`).
- Per-instance summaries now carry the BDDL truth tables: `bddl_at_init` (init conditions all true, every goal atom
  with its value; the binary goal is false) and `goal_check.goal` (every binary goal atom true at the teleported and
  settled goal). Held-object tasks evaluate the complete goal expression at the goal configuration as well (previously
  `pose_reached` only); a `grasped` atom that reads false right after the teleport is logged, any other false atom
  rejects the draft.
- New `render_instances.py`: one PNG per instance in the BEHAVIOR replay layout (left wrist | head | right wrist on top,
  the replay's over-the-shoulder external camera below) plus a per-problem contact sheet under `<out>/renders/`.
  Objects the loader cannot restore yet (assisted grasps, hinge constraints) are posed from the TRO with gravity off.
  30 / 30 v1.1 instances rendered and inspected.
- Spec §3.3 corrected: the R1Pro articulation root (`base_footprint_x`) is anchored to the world by a fixed joint at
  the creation pose and never moves while the simulator plays; `set_position_orientation` only rewrites the six
  virtual joints (measured). The base sinks through the prismatic z joint when parked over an unloaded room, hence the
  parking rule.

## 1.1.0 (2026-09-11) - efficiency pass (same outputs, same checks)

Profiling (`template_meta.timings.*_kit`, `summary.phase_s` / `summary.kit`, new in this version) showed three hot
spots; each fix keeps the accuracy checks unchanged:

- **On-disk cache for the IK precomputations** (`sampling/cache.py`, `$ISPATIALGYM_SAMPLING_CACHE`, default
  `~/.cache/ispatialgym/sampling`, `off` to disable). Reach grids and grasp combination tables are keyed by the scene
  (model + loaded rooms), the placed spawned objects, the stance, the lattice, the query poses and the IK settings.
  Re-running a spec (more instances of an existing template) skips the IK entirely: IORS ontop template 80 s -> 17.5 s.
- **Held-object precomputation uses the batched IK solver's solutions** instead of exact single solves, grouped by
  trunk configuration (every new set of locked joints costs seconds). The torso embodiment now only needs its 0.2 s
  batched solver instead of the 50-90 s planner build. Template precompute: object scaling 62 s -> 19 s, mental
  rotation 232 s -> 20 s.
- **Assisted-grasp establishment without the environment pipeline**: `robot.apply_action` + a non-rendering simulator
  step (`SimWorld.control_step`) instead of `env.step`, which rendered and read six 1080^2 sensor buffers per step.
  Held instance 7.2 s -> 4.0 s. Full steps that precede predicate evaluation keep rendering: the physics -> USD sync
  that happens there is what refreshes the AABB-based predicates (a non-rendering variant made `Inside` fail).
- Bodies teleported through the physics API are woken explicitly (`SimWorld.activate`, held-object placement,
  disembedding drafts): a sleeping body keeps its old render transform and is then missing from segmentation renders.
- Default reach translates are `[0.5, 0.75]` (deeper "sitting" translates never added a reachable cell on tables and
  each costs a full IK pass); specs can still list more.
- Disembedding: an occluded draw with no room in front of the target falls back to an unoccluded instance instead of
  failing the attempt, and distractors that still occlude an unoccluded target are re-drawn (up to 3 times) before the
  attempt is given up.

Measured facts added: cuRobo's exact single IK is ~100 ms (arm) / ~750 ms (arm + torso) per call, while the batched
12-seed solver costs a few ms per pose plus a fixed refresh per distinct trunk configuration.

## 1.0.0 (2026-09-11) - first complete version

Everything in `ispatialgym/docs/instance_sampling_spec.md`:

- Task spec loader (`spec.py`), concrete BDDL writer (`bddl_writer.py`), TRO writer (`tro_writer.py`),
  grasp-library access (`grasp_library.py`), knowledge helpers (`knowledge.py`), geometry (`geometry.py`).
- Simulator layer (`sim_utils.py`): partial-room scene loading, gravity-free object parking, support-top rasters from
  raycasts (largest horizontal face, not the AABB top), BEHAVIOR-style settle / stability / scene-unchanged / drift
  checks, template saving (partial + merged).
- Robot layer (`robot_utils.py`): cuRobo whole-body collision checks, camera FK, view solving, single and batched IK
  (`R1ProCuRobo.solve_ik_batch`, 12-seed screening / 32-seed exact), reach grids, traversability, geodesics.
- Samplers for all 15 problems (`tasks/`): camera relocalization, object scaling / mental rotation (held object with
  assisted-grasp establishment and per-template precomputed goal options), disembedding (destination tray made
  "barely visible" by turning the base, occlusion tuning by seg renders), block design (Kohs blocks), articulation
  (open / close, passive-stability probe of joint angles, drive-forward reach), attachment (meta-link alignment,
  revolute symmetry equivalences, collision filter), world alignment table / vertical (COM hinges as TRO
  `constraints`), inter-object relations (ontop / under / inside / nextto / touching).
- `sample_instances.py` CLI (several problems and scenes per process), `validate_instances.py` reload validator.
- Kohs block asset pipeline (`assets/objects/kohs_block/`): Blender build, URDF -> USD conversion into
  `ispatialgym-assets/objects/toy_dice/isgblk`.

Facts established while validating on house_single_floor (all folded into defaults / the spec):

- cuRobo IK costs ~1 ms per (seed, problem) regardless of batching; screening uses 12 seeds, exact checks 32.
- R1Pro right-arm reach over a 0.80 m table from a 0.45 m stance: a strip 0.30-0.34 m deep, lateral -0.44..-0.03 m
  (robot's right); deeper trunk translates add nothing. Reach checks lock the arms tucked ("sitting").
- The head camera sits ~0.3 m ahead of the base and JoyLo's tilt is clamped, so a held object 0.55 m along the optical
  axis is out of arm reach; 0.50 m works at trunk (0.5, 0) with a 15 deg roll of the closing axis.
- Isaac Sim 5.1 `seg_instance` segfaults when a light fixture and glass are both in view; light-like categories are
  hidden during sampling (templates unaffected).
- OmniGibson's `AttachedTo` no longer filters child / parent collisions; an attached cap on a bottle neck jitters unless
  the pair is filtered.
- The toolbox lid (`toolbox/redmtl`) falls fully open from every angle in [85, 170] deg: its only passive open target is
  the 170 deg limit.
- A holonomic R1Pro keeps its articulation root at the spawn pose; every base pose lives in the six virtual joints.

- Partial-room scene loading skips scene-file objects without a room assignment, so spawned objects are saved with
  `in_rooms` = the loaded rooms and evaluation configs must supply the robot themselves (the validator does).
- `pose_settled` is not emitted for fixed scene partners (kinematic prims have no velocity to read).

Validation (`validate_instances.py`, fresh process per template, `ISpatialGymTask.load_tro_instance`): 29 / 29
instances of the 15 test problems reload with the init conditions true, the binary goal false, no drift over 60 extra
steps and the robot pose / joints within 2 mm / 0.02 rad of the TRO; the plain-teleport goal check passes wherever it
applies. Assisted grasps and COM hinges are not restored by the loader yet (reported as skipped).

Efficiency baseline (one template, 2 instances each, RTX 4090): scene load 35-45 s, cuRobo build 13 s (+50 s for the
torso embodiment when a held task needs it), template precompute 20-90 s (IK grids), instances 3-8 s for table tasks,
7 s camera relocalization, ~10 s held-object instances (assisted-grasp establishment through the environment step).
Whole test set (15 problems, 2 instances each, 3 scene loads): ~30 min sampling + ~15 min validation.
