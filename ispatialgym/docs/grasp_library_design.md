# Per-object grasp library for the R1Pro oracle: survey and proposal

Status: proposal, 2026-09-04. Companion to `oracle_primitives_design.md`.

## 0. Problem statement

`Skills.grasp` currently guesses grasps from the object's AABB (vertical / 30 / 45 deg tilted
approaches x wrist yaws) and discovers whether they work only when the demo is already running.
The expensive part is not the grasp physics but the *search*: every failed candidate costs an IK /
planning attempt, and a failed demo costs a whole episode. The fix is to move the search offline:
compute a set of viable grasps **once per object model** (BEHAVIOR assets are fixed meshes with
known collision geometry), store them with the asset, and have `grasp` consume them as a ranked
goal set.

Two facts specific to our setup shape the design:

* **Assisted grasping (AG) is what the benchmark uses** (data collection and evaluation both use
  `grasping_mode: assisted`, confirmed by the OmniGibson maintainers in
  [BEHAVIOR-1K#2245](https://github.com/StanfordVL/BEHAVIOR-1K/issues/2245)). A grasp
  "succeeds" when both fingers touch the object; friction / force closure / shaking robustness
  are irrelevant. So a *viable grasp* for us means: the closed fingers make two-sided contact with
  the object, the pre-grasp / approach is collision-free (object, support surface, clutter) and the
  pose is reachable. This is much cheaper to validate than physical grasping and means we do not
  need a physics label per grasp, only a geometric one plus an AG check.
* **Nothing exists yet.** BEHAVIOR-1K objects ship bounding boxes and functional meta-links
  (`misc/metadata.json`) but no grasp annotations; OmniGibson's own helper
  (`utils/grasping_planning_utils.get_grasp_poses_for_object_sticky`) is the same AABB-top-down
  heuristic we use. Meshes are encrypted USDs, but collision meshes are available at runtime
  (`link.collision_meshes[*].points/faces`, already used by `curobo_planner.update_world`), so we
  can export per-link collision meshes once.

## 1. Prior work and tools (what to reuse)

### 1.1 Storage formats in use

| Format | Where | Content | Verdict |
| --- | --- | --- | --- |
| **`isaac_grasp` YAML v1.0** ([Isaac Sim Grasp Editor docs](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_simulation/grasp_editor.html), [Isaac ROS spec](https://nvidia-isaac-ros.github.io/v/release-3.2/concepts/manipulation/isaac_grasp.html)) | Isaac Sim Grasp Editor (import/export), Isaac ROS cuMotion manipulation, GraspDataGen output, GraspGen `demo_object_mesh.py --output_file` | per gripper/object pair: `object_frame`, `gripper_frame`, then named grasps with `confidence`, `position`, `orientation {w, xyz}` of the gripper frame **in the object frame**, `cspace_position` (gripper joints when grasping) and `pregrasp_cspace_position` (open) | **Adopt as the canonical human-readable format.** It is the lingua franca of the NVIDIA stack we are already on (Isaac Sim 5.1 ships `isaacsim.robot_setup.grasp_editor` with `import_grasps_from_file() -> GraspSpec`, which also computes gripper pose from object pose and vice versa). |
| ACRONYM HDF5 ([Eppner et al. ICRA'21](https://arxiv.org/abs/2011.09584), [site](https://sites.google.com/view/graspdataset)) | 17.7M Panda grasps on 8.8k ShapeNetSem meshes | `grasps/transforms (N,4,4)` gripper base pose in object frame, `grasps/qualities/flex/object_in_gripper (N)` plus motion-during-closing/shaking qualities, `object/file`, `object/scale` | Good model for the *bulk* numeric store (thousands of sampled grasps with labels). |
| GraspGen dataset JSON ([spec](https://github.com/NVlabs/GraspGen/blob/main/docs/GRASP_DATASET_FORMAT.md)) | 57M grasps, 8.5k Objaverse-LVIS objects, Panda / Robotiq-2F-140 / suction | `object.file`, `object.scale`, `grasps.transforms (N,4,4)`, `grasps.object_in_gripper (N bool)` | Same idea as ACRONYM in JSON; what to emit if we ever train GraspGen for the R1Pro gripper. |
| graspnetAPI 17-D array ([docs](https://graspnetapi.readthedocs.io/en/latest/grasp_format.html)) | GraspNet-1Billion, AnyGrasp, GSNet, EconomicGrasp | `[score, width, height, depth, R(9), t(3), object_id]` per grasp, `GraspGroup.save_npy/from_npy`, `nms()`, open3d gripper meshes for visualisation | Useful conventions: store **width** (our fingers open only 10 cm) and **depth** (finger insertion) with the pose. |
| GraspDataGen / Isaac Replicator results YAML | see below | `gripper_location`, `gripper_orientation`, `joint_states` per evaluated grasp (`capture_N.yaml`) | Convertible to `isaac_grasp`. |

### 1.2 Generators (analytic samplers)

| Tool | What it does | Fit for us |
| --- | --- | --- |
| **`isaacsim.replicator.grasping` (installed in our Isaac Sim 5.1)** — [tutorial](https://docs.isaacsim.omniverse.nvidia.com/latest/synthetic_data_generation/tutorial_replicator_grasping_sdg.html) | `sampler_utils.sample_antipodal(trimesh, num_candidates, num_orientations, gripper_maximum_aperture, gripper_standoff_fingertips, gripper_approach_direction, grasp_align_axis, orientation_sample_axis, lateral_sigma, seed)`: ~100 lines of numpy/trimesh (Apache-2.0). Samples surface points uniformly by area, casts a ray along the inward normal, takes the farthest hit within the aperture as the opposing contact, puts the grasp centre at the midpoint, aligns the gripper closing axis with the contact-to-contact line, spins `num_orientations` copies about that line, and backs the origin off by the standoff along -approach. **It does no collision checking, no check of the opposing normal, and no gripper geometry beyond aperture/standoff** - it is a raw candidate generator (on a cup-like mesh ~28% of its poses approach from below, i.e. through the table). Measured: 3600 poses in ~50 ms (268-face mesh, embree). `GraspingManager` then runs multi-phase PhysX evaluation of a gripper prim against the object and writes one YAML per grasp. | **Best first source of candidates**: no model, no dataset, parameterised by our gripper (aperture 0.10 m, standoff = fingertip offset), runs in the OmniGibson process. Must be followed by the geometric filter. Its physics evaluator needs a standalone gripper articulation prim; we can either extract the R1Pro gripper subtree into its own USD, or (simpler, since AG is what matters) validate with our own AG check (Section 3). |
| **GraspDataGen** ([NVlabs/GraspDataGen](https://github.com/NVlabs/GraspDataGen), Isaac Lab) | Reproduces the ACRONYM pipeline: gripper definition from a USD (finger colliders + base frame + bite depth), collision-free "grasp guess" generation on a mesh, then batched PhysX validation with close + tug sequences (1024 envs). Outputs `isaac_grasp`. Predefined grippers: Robotiq 2F-85, OnRobot RG6, Franka Panda; new grippers via a flattened USD. | Best route to **large-scale, physics-labelled** R1Pro grasps (and the training data if we want a GraspGen model for R1Pro). Needs Isaac Lab and a standalone R1Pro gripper USD. |
| Dex-Net `AntipodalGraspSampler` ([BerkeleyAutomation/dex-net](https://github.com/BerkeleyAutomation/dex-net)) | Friction-cone antipodal sampling + robust quality metrics (Ferrari-Canny, robust epsilon). Python 2 era, heavy deps. | Reference only; the Isaac sampler covers the geometric part. |
| EmbodiChain `graspkit.pg_grasp` ([docs](https://dexforce.github.io/EmbodiChain/main/features/toolkits/grasp_generator.html)) | trimesh antipodal sampler + gripper collision checker + interactive viser annotation of grasp regions, cached per mesh. | Nice reference for the interactive "annotate graspable regions" UX. |

### 1.3 Learned 6-DoF grasp predictors

| Model | Input / gripper | Notes | Fit |
| --- | --- | --- | --- |
| **GraspGen** ([Murali et al. 2025](https://arxiv.org/abs/2507.13097), [NVlabs/GraspGen](https://github.com/NVlabs/GraspGen)) | complete or partial point cloud **or a mesh/USD** (`scripts/demo_object_mesh.py --mesh_file ... --num_grasps N --output_file grasps.yml`); Franka Panda (8 cm), Robotiq 2F-140 (14 cm), suction; diffusion generator + on-generator-trained discriminator that scores grasps; 20 Hz; writes `isaac_grasp` YAML and optionally a USD with gripper wireframes | Convention: approach +Z, closing along X, pose of the gripper *base link*; `depth` = base -> TCP. README: for antipodal grippers with similar stroke you may re-target with a Z offset; for best results retrain (GraspDataGen provides the data). Check the NVIDIA source-code licence for our use. | **Best learned proposal generator**: our gripper opens 10 cm, between Panda and Robotiq-140. Use the Panda checkpoint re-targeted (frame offset + width filter) as a second, better-ranked source of candidates; retrain only if the analytic sampler + AG validation proves insufficient. |
| Contact-GraspNet ([Sundermeyer et al. ICRA'21](https://arxiv.org/abs/2103.14127), TF; PyTorch ports [elchun](https://github.com/elchun/contact_graspnet_pytorch), [cgn_pytorch](https://github.com/sebbyjp/cgn_pytorch)) | partial scene point cloud, Panda | Contact-based 4-DoF parameterisation, trained on ACRONYM. Point-cloud oriented; fine but superseded by GraspGen for complete geometry. | Fallback if GraspGen licensing is a problem. |
| M2T2 (NVIDIA 2023) | scene point cloud, Panda; grasps + placements | Transformer, multi-task. | Same as above. |
| AnyGrasp / GSNet ([anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk)) | RGB-D point cloud, generic two-finger | Licensed closed SDK (machine-bound license); baseline GSNet open. Output in graspnetAPI format. | Not worth the licensing friction for an offline, full-geometry setting. |
| AO-Grasp ([stanford-iprl-lab/ao-grasp](https://github.com/stanford-iprl-lab/ao-grasp)) | articulated objects (PartNet-Mobility), per joint state | 78k grasps on handles/lids etc. | Relevant later for Configuration Matching (articulated opening): grasps on movable links must be stored per link and validated per joint state. |
| GraspFactory (2025), DexGraspNet, GraspXL | CAD grasp dataset (Robotiq-85/Panda); dexterous hands | Dataset / dexterous, not directly applicable. | - |

## 2. Proposed storage format

Use **`isaac_grasp` YAML** as the exchange/curation format, extended with an `ispatialgym`
metadata block, and a compact **NPZ** mirror for large sampled sets. One file per
`(object model, link, gripper)`.

```
ispatialgym/assets/grasps/r1pro_gripper/<category>/<model>/<link>.yaml     # curated, human-readable
ispatialgym/assets/grasps/r1pro_gripper/<category>/<model>/<link>.npz      # all sampled + labels
```

```yaml
format: isaac_grasp
format_version: 1.0
object_frame: base_link            # object *link* frame (rigid objects: base_link; articulated: the movable link)
gripper_frame: right_eef_link      # our eef frame: +z = approach, y = finger closing direction
ispatialgym:
  schema: isg-grasp-v1
  object: {category: mug, model: yhjzwg, link: base_link}
  object_scale: [1.0, 1.0, 1.0]     # scale of the mesh the grasps were computed on (BEHAVIOR instances are scaled per instance!)
  gripper: {name: r1pro_gripper, max_width: 0.10, fingertip_offset: <Skills.fingertip_length>}
  source: {generator: isaac_antipodal | graspgen_panda | manual, version: ..., seed: ...}
  validation: {method: og_assisted_grasp, date: ..., n_tested: 200, n_success: 143}
grasps:
  g_000:
    confidence: 0.93               # ranking score (validation success rate x discriminator score / clearance)
    position: [0.012, -0.031, 0.045]
    orientation: {w: 0.71, xyz: [0.0, 0.71, 0.0]}
    cspace_position: {right_gripper_finger_joint1: 0.021, right_gripper_finger_joint2: 0.021}   # closed-on-object
    pregrasp_cspace_position: {right_gripper_finger_joint1: 0.05, right_gripper_finger_joint2: 0.05}
    ispatialgym:
      width: 0.042                 # object width between the finger pads at this grasp
      approach_standoff: 0.10      # collision-free pre-grasp distance along -z
      tags: [top, rim]             # semantic tags for task oracles (top / side / handle / rim / cap ...)
      validated: {ag_success: true, collision_free_pregrasp: true, approach_clearance: 0.02}
```

Design notes:

* **Frame**: gripper pose expressed in the object *link* frame, as in `isaac_grasp` (the
  `GraspSpec` helpers then give the world eef pose from the live object pose). Rigid objects
  use `base_link`; handles / lids of articulated objects store grasps on the moving link.
* **Our gripper frame is `right_eef_link`** (same frame cuRobo plans for), so a stored grasp
  is directly a planning target. Store `width`; anything above 0.10 m (or below the closed gap)
  is unreachable for R1Pro. Left-arm use: mirror the grasp about the object (or store both).
* **Scale**: a `DatasetObject` can be instantiated at a per-instance `scale` or fitted to a
  `bounding_box` (possibly non-uniform), so grasps computed on the canonical mesh are not
  automatically valid for every instance. Store `object_scale`; at load time, if the instance scale is uniform, scale
  positions/widths and re-check width; if non-uniform or different, re-run the geometric
  sampler on the instance mesh (fast, Section 3.1) instead of trusting stale grasps.
* **NPZ mirror**: `T (N,4,4) float32`, `width (N)`, `score (N)`, `success (N) bool`, `tags`, so
  thousands of grasps per object stay cheap; YAML keeps the curated top-K (e.g. 50).
* Tags let task oracles express preferences: Disembedding / block design want `top` grasps that
  do not occlude the object from the head camera; Mental Rotation wants grasps that leave the
  wrist rotation range free; articulated opening wants `handle`.

### 2.1 Visualisation

Three complementary options, all cheap:

1. **Isaac Sim Grasp Editor** (`isaacsim.robot_setup.grasp_editor`, installed): import the
   `isaac_grasp` file with the R1Pro gripper and the object USD, step through grasps. Good for
   curation / manual authoring of tricky objects.
2. **Offline trimesh + viser** (both already in the env; cuRobo 0.8 uses viser): object collision
   mesh + a gripper marker per grasp (two finger boxes + palm, like ACRONYM's
   `create_gripper_marker`, or the actual R1Pro gripper meshes from the URDF), colored by score /
   AG success, with a slider to page through grasps. Also render a PNG montage per object for
   dataset review (headless `trimesh.Scene.save_image` via pyrender or matplotlib).
3. **In OmniGibson**: spawn visual-only `PrimitiveObject` markers (or a visual copy of the
   gripper links) at the world pose of each candidate during debugging, and record the
   head-camera image.

GraspGen's `create_grasp_usd.py` also writes `/world/grasps_visualization` wireframes into a USD
you can open in Isaac Sim, if you go the GraspGen route.

## 3. Proposed pipeline

### 3.1 Phase 1 - analytic candidates + AG validation (recommended first)

1. **Export meshes once**: headless OmniGibson session that loads each object model used by the
   benchmark (task-custom lists), writes per-link collision meshes (`points/faces`, world scale
   = 1) to OBJ, plus `metadata.json` bbox info. ~seconds per model.
2. **Sample candidates** with `isaacsim.replicator.grasping.sampler_utils.sample_antipodal`
   (aperture 0.10, `gripper_standoff_fingertips` = eef-to-fingertip offset, approach +z, align
   axis y, several orientations per antipodal axis; 500-2000 candidates). Add a few
   task-specific families on top: top-down grasps around the vertical axis for small objects,
   rim grasps for open containers (from the `container` meta-link volume), handle grasps for
   articulated links.
3. **Geometric filter** (no simulation): finger sweep collision-free against the object at the
   pre-grasp and along the approach (use cuRobo's collision checker with the gripper spheres
   already in `r1pro_description_curobo_default.yaml`, or trimesh boolean tests), width in
   `[closed_gap + 5 mm, 0.10 - 5 mm]`, both finger pads overlap the object surface (two-sided
   contact is exactly the AG condition), approach clearance from the object's stable support
   plane (fingers must not need to enter the table).
4. **AG validation in OmniGibson** (batched, offline): for each surviving grasp, place the object
   in the air at the eef with the stored relative pose, close the gripper, check
   `robot._ag_obj_in_hand`, then apply a short shake (`move_eef_cartesian` +-3 cm) and check the
   object stays attached and does not intersect the gripper by more than a tolerance. This is the
   ACRONYM protocol reduced to what AG needs; ~1-2 s per grasp, so validating the top 100 per
   object is minutes, once. Store labels and the closed `cspace_position`.
5. **Rank** by clearance margin and validation outcome; write YAML (top-K) + NPZ (all).

### 3.2 Phase 2 - learned proposals (GraspGen)

Run `GraspGen` (`demo_object_mesh.py`) on the exported meshes with the Panda checkpoint,
convert its gripper-base poses to our `right_eef_link` frame (constant transform: Panda base ->
TCP offset `depth`, then our TCP -> eef offset; rotate 90 deg about z because GraspGen closes
along x and we close along y), drop widths > 0.10, and push the survivors through the same
filter + AG validation. Benefits over the sampler: fewer, better-distributed, discriminator-ranked
candidates on complex shapes (mugs, bottles, tools). If it earns its keep, generate an R1Pro
dataset with GraspDataGen (needs a standalone gripper USD; the R1Pro gripper subtree can be
exported from the robot USD or re-imported from a gripper-only URDF cut from `r1pro.urdf`) and
retrain, emitting the GraspGen JSON format.

### 3.3 Phase 3 - consumption in `Skills.grasp`

* `GraspLibrary.load(obj, link, gripper)` -> candidates in world frame (via live link pose +
  scale handling), filtered by tags requested by the oracle.
* `grasp()` uses them as a **goal set**. cuRobo 0.8 supports this natively: build the planner with
  `MotionPlannerCfg.create(..., max_goalset=K)`, pass `GoalToolPose.from_poses(..., num_goalset=K)`,
  and call `MotionPlanner.plan_grasp(grasp_poses, current_state, grasp_approach_axis="z",
  grasp_approach_offset=-0.15, grasp_lift_offset=..., disable_collision_links=<finger links>)`,
  which plans goalset -> approach -> grasp -> lift in one call and reports the chosen
  `goalset_index`. Our current `plan_pose` path uses `num_goalset=1`
  (`curobo_planner.py` L705); switching to goalset planning removes the per-candidate loop.
* `stance_for` evaluates the stored set rather than AABB guesses, so base placement is chosen
  for grasps that actually exist.
* Cache misses (unknown model / non-uniform scale) fall back to on-the-fly sampling (steps 2-3
  take ~1 s) and the existing AABB heuristics last.

### 3.4 Effort estimate

These are judgment estimates from decomposing the work, not measured data. They lean on pieces
that already exist in this repo (collision-mesh extraction in `curobo_planner.update_world`,
sphere-based collision queries, the object-into-hand teleport used in the earlier skill tests).
The main schedule risks are the AG validation harness (spawning the object inside the open
gripper without physics kicks, AG contact semantics) and switching the planner to goalsets
(CUDA-graph shapes change, `plan_grasp` offsets need validating). A one-day spike on a single
object (sampler -> filter -> AG check -> YAML) is the cheapest way to replace these guesses with
evidence.

| Step | Effort |
| --- | --- |
| Mesh export + sampler wrapper + geometric filter + YAML/NPZ IO | 1-2 days |
| AG validation harness (batched, headless) | 1 day |
| `GraspLibrary` + goal-set planning in `grasp` / `stance_for` | 1 day |
| viser / montage visualiser | 0.5 day |
| GraspGen integration (frame conversion, width filter) | 0.5-1 day + licence check |
| GraspDataGen with an R1Pro gripper USD + retraining | 1-2 weeks (Isaac Lab setup, gripper USD, data gen, training) |

## 4. Open questions

* Which object set to cover first: the task-custom lists for Disembedding, Configuration
  Matching and IORS are the ones that grasp; Object Scaling / Mental Rotation start grasped
  (their initial grasp should itself be sampled from the library so instances are consistent).
* Whether grasps should be stored per *scaled instance* (exact) or per canonical model
  (re-sampled on scale mismatch). The latter is proposed above.
* Articulated links: store per link and re-validate per joint state (AO-Grasp does exactly
  this); the oracle for articulated opening needs handle grasps that keep the wrist within its
  range along the whole arc.
