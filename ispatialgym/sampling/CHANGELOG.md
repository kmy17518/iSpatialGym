# Instance sampler changelog

Version = `ispatialgym.sampling.SAMPLER_VERSION` (also written into every `sampling_report_*.json`).

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
