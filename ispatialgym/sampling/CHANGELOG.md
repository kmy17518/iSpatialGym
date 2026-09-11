# Instance sampler changelog

Version = `ispatialgym.sampling.SAMPLER_VERSION` (also written into every `sampling_report_*.json`).

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
