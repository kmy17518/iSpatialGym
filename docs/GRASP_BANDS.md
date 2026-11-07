# Grasp library v2 and scene-independent reach maps (branch `grasp-bands`, 2026-11-02)

Implements the two ISG1020 prompts (`tmp/isg-1020-recovered/grasp_library_and_reach_bands_prompts.txt`) on `train-fast-vecnext`
(88ec8084: train-fast + vec-eval-next + vec_next's demo physics in `run_isg`). Goal: grasp libraries and reach maps for 1000+ object
models, computed the way demos are collected, fast.

| piece | module | what |
|---|---|---|
| contact policy | `ispatialgym/grasping/contact_policy.py` | versioned finger/support allowance (2 mm on the fingers' collision hulls), non-finger rejection |
| hand geometry | `ispatialgym/grasping/hand_geometry.py`, `assets/robot/r1pro_hand_collision.json` | PhysX collision hulls of fingers / palm / wrist camera / wrist in the eef frame (exported from the USD) |
| records | `ispatialgym/grasping/schema.py` | rest, family, candidates, trial, library, profile schemas; placement contract; compatibility checks |
| candidates | `ispatialgym/grasping/candidates.py` | top-down + side grasps, policy screening, symmetry families, dedup |
| symmetry | `ispatialgym/grasping/symmetry.py` | rotational / translational families (local surface invariance), one candidate per family |
| physics validation | `ispatialgym/grasping/validate_tabletop.py` | persistent world, measured rest, demo closure, force filter, support unloading |
| articulation | `ispatialgym/grasping/{artic_profile,validate_artic}.py` | physics profiles (recorded / calibrated, cached), handle pulls, sequential hybrid |
| pipeline | `ispatialgym/grasping/{pipeline,manifest,object_geometry}.py` | export -> candidates -> validate over a manifest, sharded |
| height anchors | `ispatialgym/sampling/height_anchors.py` | exact support heights, lookup ranges, workload |
| reach maps | `ispatialgym/sampling/reach_maps.py` | robot-relative maps in a synthetic world (batched IK, per-cell status) |
| sampler side | `ispatialgym/sampling/map_sampling.py`, `v2/tabletop.py` (`map_bands`), `v2/ik.py` (`support_policy_ok`), `grasp_library.py` (v2) | band view, weighted continuous proposals, exact-check policy, v2 grasps |
| cluster | `scripts/grasp_bands/{env_cluster.sh,job_glib.sh}`, lambda `env_lambda.sh` | per-shard jobs |

## Contact policy (shared, versioned: `contact_policy.ContactPolicy`, key `isg-support-contact-v1-<hash>`)

- The support stays in the world for every robot part.
- Only `{arm}_gripper_finger_link1/2` (explicit link list, `hand_geometry.role_links`) may touch the support, signed clearance >= -0.002 m,
  measured on their PhysX collision hulls (8 convex pieces per link, vertices + 3-6 mm face samples), never on the cuRobo spheres.
- Palm (`{arm}_gripper_link`), wrist camera + mount (`{arm}_realsense_link`), wrist (`{arm}_arm_link7`): exact hulls >= 0 AND their
  cuRobo spheres (radius + the model's 2 mm buffer) >= 0; every other link by its spheres. Floor mode: same finger rule on the floor,
  the robot's ground-contact links (wheels, base) scoped out (they stand on it).
- Applied at the pre-grasp (`conventions.PRE_DIST` = 0.10 m back along +z), every 1-2 cm of the straight approach and through the
  closing sweep (pre-shape -> pads' first touch).
- Sphere padding vs geometry (`tests/test_contact_policy.py`): with the buffer the finger spheres match the hull at the fingertip
  (18.9 vs 18.8 mm) but sit up to 1.6 mm off the outer face and inside the finger elsewhere: a sphere-based 2 mm rule is not a 2 mm rule.
- Sampler: `ISPATIALGYM_SUPPORT_POLICY=1` (set automatically with reach maps) re-checks `IKChecker.exact`'s last pose - solved with the
  support out of the world - with the support present for every non-finger link and the hand hulls against the support's top plane.

## Grasp library v2

Layout (`ISPATIALGYM_GRASP_LIBRARY_V2`, default `runlogs/grasp_library_v2`): `models/<entry id>/{geometry,candidates,rests,validation,
library}.json`, `profiles/`. Entry id = `<category>__<model>__<scale tag>[__<constraint>][__r<rest digest>]`.

### Candidates (`candidates.py`, CPU)

Isaac's antipodal sampler (6000 x 12; densified x4 / x8 when nothing survives - knobs), then per mode `top` (approach <= 40 deg from
vertical) and `side` (70-100 deg): width 3 mm..aperture, antipodal cone 30 deg; the support plane of the designated rest against the
exact hand hulls at the oracle's pre-shape (pinch / 2 + 10 mm): top grasps retreat along the approach (contact at most 17 mm beyond
the eef origin), side grasps lift along the normal (`lift_grasp_off_support`'s move), aiming 1 mm inside every allowance; sphere
clearance from the object (1 mm) with the knob/rim retreat, the approach from the pre-grasp clear of the object; closing sweep (pads
meet the intended contact first); finger blades (exact hulls) outside the object's pieces down to the first touch (pads' inner faces
and the pinched pieces excluded, 0.5 mm tolerance - the lid__pbmsji__s065 dome lesson); lever cap; constraints (`attachment_free`: the
mating meta link's point and -z: contact on the far side, no finger past the mating face; `flower_stem_top`); NMS top-12 per mode; the
full policy geometry; symmetry families and dedup.

### Symmetry (`symmetry.py`)

Local surface invariance around the pads (5 cm, p95 <= 1.5 mm + sample spacing) under rotation about axis candidates (the rest's
vertical, principal axes; centre fitted to the cross-section at the pad height) and translation along the pad-width axis; members must
keep the width (3 mm), the object clearance and the contact policy against the support. Record: `type`, `frame: object`, `axis`,
`center` / `origin`, `interval` (rad / m, relative to the representative; full turns exact), `vertical_yaw_equivalent`,
`representative`, `members` (folded candidates with relation member / twin / duplicate and parameter), provenance, `physics:
representative_only`. Review instructions: `docs/GRASP_REVIEW.md` (symmetry annotations required).

### Physics validation (`validate_tabletop.py`, GPU, one persistent world)

Demo physics: 30 / 120 Hz CPU PhysX, the JoyLo R1Pro (assisted grasping), 250 kg base, the task's AG rules (FixedJoint on moving links,
pad-clear release 0.2 s), sleep threshold 0.005 on the object, the sleep assist 30,1,3. A fixed slab (primitive cube, 5 cm, default
material) under the object, half size = footprint radius + 0.20 m, its near edge >= 0.30 m from the base. Per model: import, export
geometry, **rest** (robot parked, dropped 3 mm, settled until 1 mm / 3 mrad over 30 steps, slow, and pressing on the slab - load
measured awake, PhysX reports no contacts for sleeping bodies), snapshot (`scene.dump_state`); per representative: re-check at the
measured rest (a <= 3 mm lift if only the finger depth fails), whole-robot placement by batched IK (8 base yaws x 4 offsets x 4 trunks,
contact policy on the whole robot, arm spheres vs the object), restore + verify (0.5 mm / 2 mrad), hand set at the grasp with the
pre-shape (start lifted to a touch if inside the allowance), **the oracle closure** (`Motions.gripper` with `execute.closure_for_width`:
ramp 2.25 mm/step, contact hold 1 mm, thin pinch <= 6 mm at 0.5 mm/step), the object kept awake until the AG forms, 10 hold steps,
**support unloading** (slab lowered 10 cm, 30 steps). Valid = AG formed, held, retained within 1 cm / 0.1 rad, no non-finger robot
link on the slab (> 0.5 N), finger/slab force not above 10 N for 10 consecutive steps, object/slab load not above 2 x weight for 10
steps, achieved pose within 1.5 cm / 15 deg of the candidate. Forces: a per-substep contact view (robot links + object vs slab +
object) hooked into `RigidContactAPI.read_from_physx`; thresholds PROVISIONAL (uncalibrated; the per-trial summaries are for
calibrating them).

Measured-rest retry (`pipeline retry`, part of `all`): a model with no valid grasp whose object settled in a rest other than the
designated one (`intended: false`) gets a derived job designated at that MEASURED rest (its quaternion as the rest hint; own entry id,
the exported geometry reused): candidates for that rest, then validation - new models with an unstable canonical pose get grasps on the
rest they really take.

Rest record (`rests.json`, schema `isg-rest/1`): `quat_xyzw` (support frame = slab top, +z up; xyzw), `h_root`, `reference_yaw`,
`intended_quat_xyzw`, `tilt_from_intended_rad`, `pose_support`, `lowest_point_gap_m`, stability (`stable`, `pose_still`, `contact`,
`support_load_N`, `weight_N`, `settle_steps`, velocities, drift window), slab, physics profile, policy key, validator version. A
materially different rest (> 0.05 rad) is its own `rest_id` (`r0_other`; the brussels sprout rolls 0.34 rad).

Placement contract (`schema.place_object`): `q = R_z(yaw) * quat_xyzw`, `p = [x, y, support_z + h_root]`, `T_world_eef = T_world_obj @
T_object_eef` - no 3 mm, no recomputed bottom offset; the validation world's xy is discarded.

Compatibility: `schema.load_model(..., require={"scale", "rest_id", "policy_key"})` raises `IncompatibleRecord`; a legacy entry has no
measured rest and is never substituted. `grasp_library.load_grasps(key)` serves a legacy key from its v2 replacement (`legacy_key`)
only with `ISPATIALGYM_GRASP_LIBRARY_V2` set (sampler and oracle must agree: grasp names differ); unset = the legacy library unchanged.

### Articulation (`artic_profile.py`, `validate_artic.py`)

Handle pinch-width selection range 3-80 mm (was 15-80: `articulation.py`, `articulation_large.py`, `ground_isg.py`,
`generation_defaults.py`, icl README; the protrusion / depth thresholds of `artic_handles.py` are unrelated and unchanged). Profiles:
small flaps -> brake (`joint_brake_spec`), recorded ladder from the model's customs entries, else a bounded calibration (smallest ladder
brake that holds 25/50/75 % within 0.02 rad over 30 steps), cached by (model, scale, joint, limits, axis, ladder, physics version);
large horizontal leaves -> counterbalance (damping 5 s^-1); other-joint drives only where the model's own customs entries configure them.
Trials in the same persistent world: settle, profile, snapshot, hand at the handle (`conventions.handle_eef`), demo closure, the demo's
arm follow (`_follow_handle_poses`: rigid turn about the live hinge, 0.03 rad steps, `Motions.arm_along` chunks of 4), stops (limit,
stall, wobble, grasp lost, tracking failure), sequential hybrid (release, retreat, re-stance by the same IK search, regrasp, pull).

## Reach maps (`reach_maps.py`)

Synthetic world: the R1Pro alone (floor cuboid + self-collision in cuRobo); slab top at the anchor height, near edge at the edge distance
`d_e` (0.35 / 0.45 / 0.55 = the base centre's distance to the support's physical edge = the scene stance's stand-off: an edge stance stands `stand_off + overhang` outside the RASTER edge, the overhang being the AABB lip beyond the raster, `stances.EdgeFrame`; reconciled 2026-11-07, prompt I item 8), 1.2 m deep, 3.2 m long, 4 cm thick. Per trunk (`reach_bands.trunk_grid(h)`; eligible
when the idle posture untucked / ready / tucked is free of itself and of the slab at every `d_e`): lattice 8 cm x 8 yaws x grasps
(validated reps, twins unless stored, 4 samples of non-yaw-equivalent families incl. endpoints + the tested representative). Shoulder
reach bound; the hand hulls vs the slab per `d_e` (pre-grasp, approach every 2 cm, two sweep openings); tensor-native batched IK of the
pre-grasp (8 seeds, all kept), the grasp seeded from it (branch: max joint step <= 1 rad); per `d_e` and seed: whole-robot spheres at the
pre, 3 interpolated and the grasp configurations vs the slab, hand hulls at the FK eef poses, arm spheres vs the object. Output
`map__<cat>__<model>__h<height>__<key>.{json,npz}`: per (`d_e`, trunk) a uint8 status (cells x yaws x grasps): feasible / off_support /
reach_bound / hand_policy / ik_pre / ik_grasp / branch_jump / slab_spheres / object_arm / skipped / trunk_ineligible. `full` (exhaustive,
default) or `accelerated` (early cut; measured: 1-4 % fewer IK problems, up to 6.5 % fewer feasible entries - not worth it).

Brace / contact maps (`--kind brace --arm left`): 66 contact classes - sides near / far / left / right x approach horizontal or pitched 45
deg down x closing vertical / horizontal x height fractions 0.2 / 0.4 / 0.6 / 0.8 of the settled height, plus two top presses; closed
fingers, the fingertip plane 5 mm before the face (`conventions.fingertip_eef`, the oracle's brace stop); the same slab policy (low
contacts keep the 2 mm finger rule, palm / camera / arm rejected); contact-point-to-eef offsets in the meta. Single-arm proposals only.
Napkins box at 0.8126: 810k IK problems, 50 s (4090); feasible (cell, yaw) per fraction 16k / 20k / 23k / 29k, top 4k.

Articulation prefix maps (`reach_maps prefix --artic-lib <artic_library.json>`): the first fixed-stance pull segment of each tested
handle (object-relative eef waypoints + joint value, from `validate_artic`), per (d_e, trunk, cell, yaw) the pre-grasp and the
waypoints in order, each IK seeded from the previous (joint step <= 0.5 rad: no branch jumps), the whole robot under the slab policy;
stored: the contiguous joint travel reached (rad; -1 = not even the grasp), accepted >= `min_open_rad` (0.2). Blender eyedvd at
0.8126: 0.95 rad validated travel, 60-130 accepted (cell, yaw) per low trunk at d_e 0.45; 43k IK problems, 43 s.

Lookup (`ReachMap.find`): by object identity key (scale, rests, grasps, families, policy), rest, arm, policy and the exact anchor of the
query height (`AnchorSet.lookup`: exact / within the clipped range / uncovered -> None). Legacy bands and maps never substitute for each
other.

Heights (`height_anchors.py`): 1846 support rows -> 149 surfaces (142 tops from the inventory at 1e-4 m, 7 interior boards authored at
1e-2 m, `unmeasured_interior`) -> 93 exact anchors (88 from tops); 48 nominal 2 cm bins, 19 of them hold several distinct anchors; 0
conflicts, 0 missing inventory files. Ranges: midpoints (ties to the lower anchor) clipped to +-1 cm. Workload: every target model of the
role tables shares the subtask-level support pool -> `N_heights = 88` table heights per model (+1 with floor mode); finer task/support
compatibility would shrink it.

Sampler: `ISPATIALGYM_REACH_MAPS=<root>` (+ `ISPATIALGYM_REACH_MAP_ANCHORS`, `ISPATIALGYM_GRASP_LIBRARY_V2`): `tabletop.ensure_bands`
returns `map_sampling.as_bands(map)` - a multi-distance band view (`stand_off = d_e`: `map_bands` passes `overhang=0` because both are measured to the physical edge; `mode: map_view`, no path: band
refinement / job-local copies never touch it); a miss keeps the legacy path (`ISPATIALGYM_REACH_MAPS_STRICT=1`: reject; `--grasps v2only` / `ISPATIALGYM_V2_ONLY=1`: no legacy band and no legacy grasps anywhere - `legacy_band_blocked`, `missing_map`, `missing_grasps:<pipeline>`). Proposals
(`map_sampling.proposals`): cells ~ weight^T mixed with 25 % uniform, weights normalised for the family / yaw discretisation, continuous
jitter (cell, yaw, family parameter) with spacing and neighbour feasibility, `AdaptiveWidth` after failed exact checks; release
ranking (`release_candidates`) matches the full eef pose and grasp row.

## Validation (2026-11-02)

isg-1020 tabletop / attachment targets (16 models, `runlogs/grasp_library_v2/manifests/isg1020.json`), lambda RTX 4090:

| | |
|---|---|
| candidates | 15 / 16 models with policy-valid representatives (cap__xxqyrt: none - its hollow-shell pieces fail the sphere clearance) |
| physics | 89 representatives tested, 56 valid; 15 / 16 models with >= 1 valid grasp; rejections: object_pressed_into_table, ag_not_formed, grasp_off_target, palm / camera at the measured rest |
| rests | 16 stable rests; the brussels sprout settles 0.34 rad from canonical (own rest_id) |
| time per model | 4-19 s after an 18 s world setup (lambda); 12-75 s after 132 s (cluster TITAN RTX, NFS assets) |
| reach maps | 18 maps, 4.5-26 s each (15 trunks x 3 stand-offs x 485 cells x 8 yaws x 2-48 grasps; up to 300k IK problems / map) |
| maps vs demos | 1551 isg-1020 episodes (15 aliases): demonstrated object cell + yaw feasible in the v2 map 56 % exactly, 92 % within one cell / yaw step (`scripts/grasp_bands/compare_demos.py`) |

Full role-table run (cluster svl TITAN RTX, `roles_all.json`: 1014 jobs over 962 object models; tag `roles_all_20261102`): run 18263373
(36 shards, 02:20-03:26), follow-up 18264040 (36 shards, 03:29-04:32: the shard lost to a Kit start failure on svl9, the measured-rest
retries, the maps lost to CUDA OOM) and cleanup 18264544 (1 job, 04:39-04:58: the retry maps a sharding race skipped - fixed, see below).
- Grasps: all 1014 jobs validated; 750 with >= 1 physics-valid grasp (legacy keys 292 / 347, floating-key objects 390 / 455, new models
  68 / 212); 3401 of 4791 representatives valid; 198 jobs had no representative (126 of them new models: mostly flat, wide axial /
  wall-task objects - board games, books, pictures - which those tasks push or pinch generically). Retry: 52 jobs settled in another
  rest without a valid grasp at the designated one; regenerated at the measured rest, 22 got valid grasps (legacy 15 / floating 7 / new
  0). Per object model: 727 of 962 have >= 1 valid grasp, 19 of them only through the retry.
- Rejections (all trials): finger depth at the measured rest 413, object pressed into the table 354, palm at the measured rest 228, no
  placement 196, off target 111, AG not formed 88, slipped on unload 48, wrist camera at the measured rest 29, finger pushing the table 7.
- Time: validation median 30 s per job (p90 45 s), 7.8 GPU-h for 1066 jobs; maps 2316 (every job with a valid grasp x 3 probe heights),
  median 32 s and 96k IK problems per map, 24.5 GPU-h. The 290 OOM maps (24 GB) all completed after the point-sized chunking.
- The 12 isg1020 aliases in the role table give the same tested / valid counts per model as the lambda (4090) run.

IK throughput (synthetic world, tensor-native, 8 seeds, batch 2048): 114k problems/s on a 4090 (the per-pose wrapper: 3.7k/s; scene bands:
~13 poses/s, median 830 s per band).

## Commands

```bash
source scripts/grasp_bands/env_lambda.sh 0            # lambda; app-data copy p0
python -m ispatialgym.grasping.manifest --isg1020 --out runlogs/grasp_library_v2/manifests/isg1020.json
python -m ispatialgym.grasping.pipeline all --jobs runlogs/grasp_library_v2/manifests/isg1020.json --root ~/grasp_bands/glib_v2 --workers 8
python -m ispatialgym.sampling.height_anchors --out ~/grasp_bands/maps/anchors.json
python -m ispatialgym.sampling.reach_maps compute --jobs <manifest> --glib ~/grasp_bands/glib_v2 --anchors ~/grasp_bands/maps/anchors.json --heights 0.8126 --out ~/grasp_bands/maps/out
python -m ispatialgym.grasping.validate_artic --keys blender__eyedvd blender__cwkvib --root ~/grasp_bands/glib_v2
python -m pytest ispatialgym/tests/test_contact_policy.py ispatialgym/tests/test_grasp_bands.py
# cluster (svl): GB_MANIFEST=<manifest> GB_NSHARD=N GB_TAG=<tag> [GB_HEIGHTS="..."] sbatch --array=0-$((N-1)) scripts/grasp_bands/job_glib.sh
#   a shard whose log lacks GB_VALIDATE_DONE is failed (a Kit start that died on a node - svl9 2026-11-02: "No module named omni.kit.usd" -
#   exits 0); resubmitting a shard skips its finished models. The retry stage writes job.json (derived_from) in a derived entry; reach
#   maps shard a derived entry with its parent job (a shard's glib stage makes it), so --include-derived never races other shards
python3 scripts/grasp_bands/aggregate.py --run $GBH/out/<tag> --jobs <manifest>
```

## Fast-flow pipeline (branch `of-v2maps`, 2026-11-06)

The oracle-fix fast flow can run on v2 grasps and reach maps. The switch is per job, and nothing else in the pipeline changes:
sampling, collection, replay, customs, registry, ledger, snapshots and validation are the fast flow as before.

**Switch.** `submit_fast.py submit ... --grasps v2` (the default is `legacy`) writes `ISGF_GRASPS=v2` and the campaign's
`ISPATIALGYM_GRASP_LIBRARY_V2`, `ISPATIALGYM_REACH_MAPS` and `ISPATIALGYM_REACH_MAP_ANCHORS` into `job.env` (`GRASPS_V2`).
- The sampler and the collectors source the same `job.env`. A `recollect` copies it, so the instances keep the library they were
  sampled with; grasp names differ between the libraries.
- `fast_env.sh` validates the choice. With `legacy` it unsets the three variables, but a registry row's own library comes back with
  `job.env`: the umbrella's, as before.
- `job.txt` records `GRASPS <legacy|v2> glib= maps= anchors= strict=`. A v2 job whose paths are missing stops before sampling.
- `ISPATIALGYM_REACH_MAPS_STRICT` stays off. It would reject every alias without a v2 entry, not just map misses. A miss logs
  `reach map miss for ...` and keeps the legacy band path.

**What decides the path.** `tabletop.ensure_bands` uses a reach map only when all of these hold:
- it is a library grasp (not a generic pinch);
- the band is not placed and is not an interior surface band;
- the grasp key has a v2 entry with valid grasps, linked by `legacy_key`;
- a map exists at the support's anchor for that arm.

The library's contents decide which aliases change. A v2 job of an alias without an entry in it keeps its legacy grasps and
bands, so the campaign library holds only the entries meant to change.

**Data** (`$OFH/v2maps_20261106`, campaign-owned; the grasp-bands record is only read):
- `glib/` (21 entries):
  - 10 copied from `roles_all_20261102` and 7 from `pilot_isg1020` (the aliases' legacy keys at their spawn scale; roles_all has
    those with `legacy_key: null` or only at scale 1);
  - the umbrella from `$OFH/glib_v2`;
  - every copied file md5-identical to its source;
  - 3 from a glib run (`job_glib.sh` with `GBH=$OFH GB_OUT=...`, 331 s): steak s0.6, carving knife s1×1×1.25, spray bottle s1.
- Excluded: entries with 0 valid grasps and a legacy key (cap, cork, hard candy, cellulose tape, pillow), because `load_grasps`
  would raise `missing_grasps` on every template of their alias.
- `anchors.json`: the 8 live support heights of these aliases at 1e-4 (the sampler's `top.z - world.floor_z`), with no authored
  1e-2 anchors. All 20 aliases resolve to their own anchor exactly.
- `maps/`: 40 maps (20 entries × right and left arm at their aliases' anchors), median 40 s, compute 1,781 s, wall 0.76 GPU-h
  over 4 shards.
- The build scripts are under `$OFH/jobs/v2build/`.

**Fixes the pipeline needed** (none of the 2026-11-02 runs exercised the sampler with an anchor file or two segment arms):
- `ReachMap.find` with an anchor set overwrote its map kind with the lookup's kind, so every map failed the kind test. No sampler
  lookup could hit.
- `AnchorSet.lookup` returned an authored 1e-2 anchor (0.81) before the measured 0.8126. It now prefers the finest exact match.
- `tabletop.arm_band` mirrored a map view, which raises for two-arm aliases. It now uses the other arm's own map, and on a miss that
  arm's legacy band.
- `reach_maps compute --skip-existing` now checks the arm and kind.
- `job_glib.sh` takes the output root, code root, library, per-id heights and arms as variables.
- The support policy: `ISPATIALGYM_SUPPORT_POLICY=grasp`, the value a map hit now sets, polices only the grasp approach (chains whose
  arm holds nothing). `1` keeps the 2026-11-02 scope, which also policed places, carries and releases.

**A/B** (fast flow, 20 aliases × N 4, kcollect 2; `$OFH/out/<tag>`; tables from `$OFH/jobs/v2build/v2_ab.py` and `v2_rej.py`):

| Arm | Tag (snapshot) | Written / attempts | Demos ok |
|---|---|---|---|
| legacy | `v2legacy1` (f5decd42) | 73 / 344 (21%) | 57 / 73 |
| v2, policy on every chain | `v2maps1` (f5decd42) | 6 / 1,248 (0.5%) | 4 / 6 |
| v2, policy off (6 aliases) | `v2nopol1` (f5decd42) | 15 / 127 | 11 / 15 |
| v2, policy on the grasp approach | `v2scope1` (73d3b572; 13 / 20 jobs done) | 42 / 361 (12%) | 31 / 42 |
| v2, policy on supports, measured top | `v2scope2` (5316ca4d; 10 aliases, early) | 13 / 50 | 6 / 7 |

- **Policy on every chain:** places and releases died. touching_02 lost 62 of 66 attempts at the place screen; ontop_small_03 hit
  `palm_contact` on the partner. Grasp names and the eef frame agree with the legacy library (same positions).
- **Scoped policy:** 10 aliases are at par with legacy (written ≥ legacy − 1): attachment_01/04, inside_01, nextto_01/02,
  ontop_small_01/02/03, under_small_01/02. Some sample faster per instance: attachment_04 337 s vs 1,024 s, inside_01 181 s vs 544 s.
- **Still worse on v2:**
  - inside_03, nextto_03, ontop_small_04 (thin objects lying flat): `finger_penetration on <table>` on the grasp, 130–770 rejections
    each, even against the measured top plane. Legacy demos succeed there.
  - touching_01/02/04: 'touching push: no push class passes'. This is not the policy (0 rejections in v2scope2); the cause is unknown.
  - under_small_03: the plate's one v2 grasp has an empty feasible set at the map's cells.
  - nextto_04: the template setup fails (`no_manipulation_stance`; gym shoe, one v2 grasp, a 0.35 m bench).
  - under_small_04: 2 written against legacy's 4.
- **The map views are optimistic.** They are wider than the scene's legacy bands: the salt shaker at trunk (0.25, −0.9), stand-off
  0.45, has 103 map cells against 31 band cells. Map proposals therefore meet more exact-check rejections.

## Round 4 (2026-11-07): thin objects, the plate, settle-first, the scene mask, frame ranking, the weak aliases

The full numbers are in `$OFH/DECISIONS.md`, from "2026-11-07 ... ROUND 4" on.

### Thin flat objects (the support policy's finger opening)
- **Diagnosis** (opt-in `ISPATIALGYM_SUPPORT_POLICY_DIAG`; v2diag1 / v2diag2). Every rejection was `finger_penetration` on the table:
  - sunflower: median −13.4 mm;
  - tablespoon: −2.5 / −4.3 mm;
  - carving knife: −4.4 mm.

  The plane, the object pose (plane + h_root + the 3 mm lift) and the IK residual (1–2 mm) were all right. The policy measured the
  fingers fully open, at 0.05 m per finger. The oracle descends pre-shaped at width / 2 + 10 mm, which is the library's `preshape_q`.
  The offline HandChecker reproduces the scene's depths at 0.05 m, and gives +3.5 mm of clearance at the pre-shape.
- **Fix.** The policy now measures a library grasp at its pre-shape (`tabletop.policy_opening_of`, `IKChecker.policy_opening`) at:
  - the draw (a1117046);
  - the recorded-grasp re-checks: after-settle, held turn, disembedding, attachment (dd25a979, `record_policy_opening`);
  - the program check's grasp approach (29c77f60).

  The opt-in raise / relax fallback (`ISPATIALGYM_SUPPORT_POLICY_FALLBACK`) never fired once the opening was right.
- **v2open3** (N 4):

  | Alias | Attempts / written / demos ok | Legacy |
  |---|---|---|
  | nextto_03 | 9 / 4 / 4 | 9 / 4 / 4 |
  | inside_03 | 52 / 3 / 3 | 44 / 4 / 4 |
  | ontop_small_04 | 6 / 4 / 4 | 24 / 2 / 0 |

  The shared contact-policy defaults are unchanged.

### The plate (under_small_03)
- **Family.** side_9's family was marked rotational and yaw-equivalent in the campaign library copy only. The collision AABB is ±0.1229 m square; the detector missed it because the hull rim is 0.113–0.128 m.
- **Maps.** The recompute is identical in feasibility: 242880 / 242880 entries feasible, 0 flipped.
- **Empty F is kinematic.** 0 pairs pass even with all 38 scene objects removed. side_9 is a side pinch at rim height; the v2 top grasps were all rejected for object clearance (0 object-clear).
- **gripper_not_in_frame at the place** (`ISPATIALGYM_FRAME_DIAG`): the box is truly out of view at the image's left edge, not a margin case (1 of 16 was margin-only). In 14 of 16 cases the place IK fails too.

### Settle first, then generate (`GB_SETTLE_FIRST=1`, adeac28d / fb868b32)
- **Mechanism.**
  - A survey drops the object 12 times (4 yaws × 3 drops), and the outcomes are clustered into rests by up vector + height.
  - Rests that differ by a roll about a near-round principal axis are merged, using the support-function roundness.
  - The job keeps its directory, and the surveyed rest becomes `generation_rest_wxyz`. Other stable rests seen at least 2 times get derived entries.
  - Validation's settle stays as the guard.
- **Survey.** 24 of 26 objects have one rest (12 / 12).
  - The carving knife lands bimodally: 7 / 12 at the designated rest, 5 / 12 at 0.609 rad.
  - The cork lands upright 8 / 12 and on its side (1.467 rad) 4 / 12.
  - Sprouts, spatula and spray bottle each have one rest, 0.338 / 0.188 / 0.063 rad off the designated one.
- **Library.** Valid grasps: current 80, control 80, settle 87.
  - spatula 4 → 6 (measured-rest finger rejections 5 → 0);
  - sprouts 4 → 5;
  - umbrella 1 → 2;
  - the knife's second-rest entry: 6.

  Guard firings 18 → 15. The 5 excluded entries are still at 0: their failures aren't rest mismatches (cap / tape geometry, cork meta-link frame, candy, pillow load). The survey costs 25–47 s per object.
- **A/B.** Campaign copy `$OFH/v2maps_settle_20261107/campaign`, with the 3 changed entries' maps recomputed. Arm `absettle` vs abv2_base on inside_01 / under_small_04 / touching_04.

### The scene mask (`ISPATIALGYM_MAP_SCENE_MASK` / `_DIAG`, scene_mask.py, 9735c17f / 87fbb8fd)
- **What it is.** Categories evaluated per (cell, yaw, grasp) at the phase-3 focus, with no IK: support, height, trav, static, hand (eef-fixed link spheres via `kit.world_sphere_hits`), features, frame.
- **maskdiag2** (212 foci, 5479 pairs, 1067 in F).
  - Only trav (1526) and hand (20, all outside F) fire. Within F, only trav fires.
  - trav is wrong for manipulation stances: 225 / 305 masked F pairs pass the exact IK. So it is dropped.
- **Conclusion.** The map's surplus is reach (80 % of the focus pairs fail the IK screen), not scene geometry. The lever is the map view's reach at the actual stance: map views are never refined and are read at the nearest grid tilt.

### Frame ranking (trunk 20fd38da / 0e2a7632 / b13a368f; merged)
- **E1.** Frame rejections are almost all place / pre-place at the iors_small draw (pre-place 913, place 743). Oracle FOV failures: 0 in 498 demos.
- **Consistency.** The sampler's 8-corner eef box is stricter than the oracle's single fingertip point. Aligning them is the owner's decision.
- **Opt-in switches.**
  - `ISPATIALGYM_FRAME_RANK` keeps the partner pose (of up to 4 valid ones) whose place / pre-place box sits deepest in frame.
  - `ISPATIALGYM_STANCE_FRAME_AIM` tries the manipulation candidates in order of the expected place's frame margin.
- **A/B.** The first arms crashed (a stance-order bug) and were rerun as `*_rank2` / `*_aim2`.

### The weak aliases (prompt F)
- **First divergence on all six: the band source.**
  - The map view's phase-2 pool is 2–9× the scene band's, with deeper foci.
  - It uses a fixed lattice, all-or-nothing trunk eligibility, and no refinement.
- **touching_01 / 02.** The push is out of reach from the map's deep foci. The goal check finds no push class (11 / 16, 66 / 71). Opt-in `ISPATIALGYM_TOUCHING_PUSH_SCREEN` screens the push at the draw (745863d2).
- **nextto_04.** The map lattice rows can't hit the 1.2 cm strip of the eroded 0.36 m bench, so there is no phase-2 survivor. It stays on legacy, an owner decision.
- **under_small_04.** The program-check policy is fixed by 29c77f60. The goal settle is unstable at the v2 rest (r0_other).
- **inside_04 / touching_04.** Weak under legacy too (frame / contact).

### v2 only: the spec gaps closed (prompt I, of-v2maps bc616c4e .. 8aeb9cbd)

`--grasps v2only` (`ISPATIALGYM_V2_ONLY=1`) now refuses every legacy band (`legacy_band_blocked`) AND legacy grasps (`load_grasps`:
`LEGACY_GRASPS_BLOCKED` -> `missing_grasps:<pipeline>`); a map miss rejects (`missing_map`). What changed for v2 (defaults unless noted):

| Item | Change | Switch |
|---|---|---|
| lookup | `_v2_index` deterministic (stable rest, grasps, id, path) + `legacy_keys` links; `v2_rest_id` = the stable rest by id; `load_grasps` returns that rest's grasps only; policy key / robot digest / scale checked (`v2_scale_mismatch`); `ReachMap.find(config_require=)` | `ISPATIALGYM_REACH_MAP_REQUIRE` |
| cache | the support policy's [key, scope] joins the exact-IK cache key | - |
| SPA | the band signature names a map view by map_key / map_path | - |
| placement | z = support + h_root of the object's entry + rest_id (`tabletop.v2_rest_info` / `rest_z`), partner placements too; leaned / rolled / layout rests keep bottom offset + 3 mm; no facet snap and no pinch re-centring of v2 rests / grasps | `ISPATIALGYM_V2_REST_H_ROOT=0`, `ISPATIALGYM_V2_RECENTRE=1` |
| one policy | maps: each grasp's own pre-shape / contact opening (`MapConfig.preshape`, "max" = the old batch rule; maps without the field read "max"); validator: per-candidate opening in the re-check and the start lift; the 3.3 cm approach tail policed like the end pose | `ISPATIALGYM_SUPPORT_POLICY_TAIL=0` |
| map content | Phase 3 F from the map's pairs at the focus cell (continuous world yaw, family members `<row>@<param>`, twins merged, weighted order, `AdaptiveWidth`) | `ISPATIALGYM_MAP_F=seed` / `diag` (opt-in) |
| release | the valid partner poses ranked by `release_candidates` at the place eef (nextto / under) | `ISPATIALGYM_MAP_RELEASE=1` (opt-in) |
| stand-off | `d_e` = the base's distance to the physical edge = the stance's stand-off (`map_bands` overhang 0) - docs reconciled | - |

Not linked on purpose: held-object floating keys (mental_rotation / object_scaling) - `held_object` refuses another pipeline's grasps
(§2.2), and the v2 library has no floating constraint. Map types without a consumer (brace, floor, interior, tray, articulation prefix)
keep their aliases v2-blocked. The numbers are in `$OFH/DECISIONS.md` (PROMPT I) and `tmp/outputs/round4_reports.md` (I).

### Owner decisions 2026-11-07 (after prompt I item 7)

- Deeper trunk: no. The trunk curve stays: tilt moves joint 3 only, and translate is clamped to [0, 2]. Floor reach uses `trunk_grid` and `FLOOR_TRUNKS`.
- A "lean" rest kind in the grasp library, for touching_03 and any task that starts an object leaning on a board.
  - The object is settled against a vertical board at a given angle and side (touching_03: 11 and 20 deg), and grasps are validated with the board present.
  - Rest kind, angle, side and board geometry are part of the entry and of every map and lookup identity, so a lean rest never serves a stable-rest lookup and vice versa.
  - Interior maps are built at the lean rests: 2 angles x 2 arms for the hardback acbrnv. touching_03's shelf path reads them, and a miss rejects `missing_map`.
- FingerTouch uses the spec's 2 mm finger-geometry rule (P1 §2) instead of the 5 mm allowance on the finger spheres.
  - It applies wherever `approach_finger_support_touch` is set: touching_03 and the inside_large plate-in-sink variants.
  - The contact-policy defaults are unchanged.
  - Old vs new are A/B'd first.
- Carton push-close stop under `ISPATIALGYM_BRACE_FULL_OPEN`: 0.05 opening, corner inset 0.04 -> 0.06.
- attachment touchdown: the hover is the swept first-touch height + 3 mm, behind `ISPATIALGYM_ATTACH_HOVER`, A/B'd before it becomes the default.
- The support split is kept ON. A/B, N 4: 97 / 98 demos and 0.182 GPU-h per success, vs 79 / 94 and 0.220 with it off.
- The support policy polices only supports with a registered plane. world_vertical + articulation_large went from 0 to 36 written / 34 ok under v2only.
- Floor maps: the fingers meet the floor only through the policy's floor allowance (`floor_fingers: "policy"`). The umbrella went from 0 to 71-73 cells per reaching trunk.

### Overhang rests (i7-overhang, 2026-11-07; hardback bqvvvx, inside_large_01)

Owner decision 2026-11-07: an "overhang" grasp-library option as the overhang pick's PRIMARY path (`luhkiz` keeps it as its step 4).

- Measured first (`$OFH/jobs/v2build/oh_measure.py` over 67 records of the alias under `$OFH/out` / `$OFH/runs`): the push the demos
  produced is 0.160-0.165 m (every successful demo 0.1602: hb30_desk_s41 and its 51 fast-flow copies); the pinched spine ends
  0.0294 m beyond the desk edge at the pinch line (0.0269-0.0294 over the distinct picks), the centre of mass 0.051-0.057 m inside, the
  book 41-45 deg off flush in every pick (the flush yaw failed the legacy grasps' IK). The legacy `hardback__bqvvvx__bo` rows: the eef
  1.8 mm outside the spine, the pads ~1 cm on it, proxy table edge 0.03 m inside the spine (`overhang_m` 0.03). The overhang rest is
  side -y, 0.0294 m (the successful demos' number), flush.
- `grasping/overhang.py`: `OverhangSpec` (side, distance, flush edge, slab thickness) -> digest / tag `overhang_yn_29mm_6c7148` = rest id,
  entry suffix (`hardback__bqvvvx__s1__overhang_yn_29mm_6c7148`) and grasp key `hardback__bqvvvx__bo@overhang_yn_29mm_6c7148`
  (`overhang_keys`; `legacy_key` None, never in the stable index). validate_tabletop settles the object on a slab whose edge lies the
  distance inside the face (measured 0.0296, centre of mass 0.0555 inside, flush within 0.06 deg); candidates use that slab BOX as the
  support and let a side grasp retreat along its approach before the lift (the legacy rows' pad-tip pinch).
- Library entry: 12 candidates -> 12 geometry-ok -> 7 representatives -> 6 physics-valid (one ag_not_formed / slipped). Every valid grasp
  is a spine-edge pinch across the thickness, the eef 1.1-1.7 cm inside the spine, approach 30 / 60 deg off the spine normal, its exact
  finger hulls within -0.9..+0.6 mm of the slab box (inside the policy's 2 mm); the straight (0 deg) pinch failed physics.
- Map kind `overhang` (reach_maps): the synthetic slab's near edge at d_e under the object, one yaw (the face toward the robot), per
  d_e one row of lateral cells (the root `edge_along_m` behind the edge), d_e 0.35-0.75 (B and B backed off 0.10 / 0.20), lattice 0.04,
  the overhang rest's digest in the config and the key; `find(kind="overhang")` checks it. Maps at the desk height 0.7671 (anchor added
  to the staged root): right 5163 / left 4902 feasible (cell, grasp) entries over trunks x d_e.
- Consumer (IORSLarge, `pick_overhang_push` mode only, a v2 + maps root): the key's overhang rest; the stance search reads the overhang
  map as INIT cells (feasible post-push cells moved back by `v2_init_push_m` 0.12 / 0.16 / 0.20); the pairs at B = the face flush with B's
  edge and the rows the arm's map holds feasible at the post-push cell (the map-hit step); the push = the library rest (the face 0.0296
  beyond the edge). The legacy finger-box sweep rule (1 cm on the conservative open-width box) is replaced by the library's
  hull-validated rest (the box reads 3.6-7.3 mm over the edge for these pinches; with the 5 cm centre-of-mass margin no push satisfies both
  at a flush rest); `v2_finger_clearance_m` re-imposes it. Every exact check after the map stays.
- Goal side: `generation.inside_goal_lean` (hardback__bqvvvx__bu at 16.7 deg: 52 / 52 written instances settled leaning 16.7 deg on the
  bay's left board) -> the interior map of that v2 LEAN entry on the arm's shelf side in `bookcase_jaysra_0@0.9472` (measured: width
  0.30, depth 0.332, headroom 0.364, open front / back). Lean entries: left 1 valid grasp (a pinch at the page block, deep in the bay:
  its maps 0 feasible for both arms; a 24000 x 24 candidate re-run validated 0 - the right-arm validator's placement hits the left board),
  right 1 valid (right arm 128, left arm 107 feasible at d_e 0.65-0.85). So under v2 only the LEFT arm has no stance C (its map is
  empty: a library gap). Owner decision 2026-11-07: no NO_BAND_TRUNKS fallback under v2 only; `ISPATIALGYM_INSIDE_NO_BAND_FALLBACK=1`
  (committed, default off) is never set.
- Validation `--grasps v2only` N 4 on inside_large_01 (v2-root `$OFH/v2maps_i7oh`): official `i7_oh_m` (trunk_b8245422): 7 attempts,
  4 written, 4 demos, 2 successes, 0.234 GPU-h per success, rejections program_check_failed 2 / target_placement 1, missing_map 0,
  MAP_REQUEST 0. Pre-merge `i7_oh` (trunk_b5ed5e75): 10 / 4 / 4 / 2, 0.262 GPU-h per success. Funnel over both: map-hit 46.7 % of the
  pair lookups at B (the arm's map keeps 3 of 6 grasps), exact 19 / 57 overhang picks, 8 written, oracle 4 / 8 (the physical pick 7 / 8;
  failures robot_wobble x 2 at the place - the torso 1.01 / 1.03 deg off its command -, robot_touching_itself x 1 at the place,
  plan_failed x 1 at the grasp). Every pick and place by the right arm.
- inside_large_03 (the plate into drop_in_sink_lkklqs_0, basin 0.775, rim 0.886): the interior kind does not model a basin (no rim /
  front wall, open top; measure's wall rays at mid headroom pass above the rim) - a basin spec is needed, not built; the plate has no
  v2 grasps yet.
#### Owner decisions on inside_large_01 (2026-11-07, after the first v2only validation)

1. The v2 overhang path keeps the library rest as its finger rule; `pick_overhang_push.v2_finger_clearance_m` (default off) restores the
   finger-box rule there (test `test_the_v2_overhang_finger_rule_defaults_to_the_library_rest_and_the_switch_restores_the_box_rule`).
2. A LEFT-arm validator for lean rests (512e3139): `lean manifest --validate-arm left` -> the job's `validate_arm`, entry
   `..._lean_<tag>__valleft`; `pipeline lean --arm left` (job_glib `GB_VAL_ARM=left`): the validation world's left arm, the object
   mirrored to the robot's left (`validate_tabletop.mirrored_offsets`, as `validate_artic motions --arm left`), the left hand's
   candidates; the library records `validate_arm` and the identity carries it (right-arm entries unchanged). Lean rests only.
   lean_left of hardback__bqvvvx__bu at 16.7 deg with it: 6000 x 12 candidates 5 -> geometry-ok 1 -> 1 representative -> 1 left-arm
   valid - the same page-block pinch as the right-arm run, its interior maps 0 / 0 feasible (both arms); 24000 x 24 at seeds 0-3:
   12 -> 4-8 geometry-ok -> 3-5 representatives -> 0-1 valid (seed 1: a page-block pinch again); every spine-side representative fails
   the placement with `no_placement:board:left_gripper_finger_link1` (the right-arm runs: `...:right_gripper_finger_link1`) - not an
   arm effect: the 16 spine- and page-side representatives sit 3.3-4.8 cm below the centre, where the finger meets the board; the valid lean_right spine pinch sits 6.4 cm below.
3. Nearer places: `generation.inside_stand_off_max_m` (240591df), inside_large_01 0.70 - the stand-offs 0.75 / 0.85 are not laid out.
   All 8 placements of the first validations were at 0.85 (4 / 4 at trunk translate 0.25 succeeded, 4 / 4 at 0.50 failed).
   Validation `i7_oh_L` (trunk_512e3139, v2-root `$OFH/v2maps_i7oh2` = v2maps_i7oh with the left-arm lean_left entry and maps): the cap
   removed 238 of 476 stance-C lattice candidates per arm; the 0.55 / 0.65 rows hold no feasible goal-map cell for either arm (lean_right:
   right 0 / 0, its left-arm map 0 / 2; lean_left: 0) -> setup rejects no_placement_stance, 0 attempts, 0 written (0.09 GPU-h). Not widened.

#### The accessible region of an entry used at an interior goal (owner decision 2026-11-07, 0adc84c9)

- Measured first (`$OFH/overhang_20261107/board_clear.jsonl`, 87 lean candidates of hardback__bqvvvx__bu 16.7 deg over 7 runs): the hand
  spheres at the validator's placement opening (fully open 0.05 + the 0.002 buffer) vs the lean board separate the outcomes exactly -
  every physics-valid grasp +0.3..+5.0 mm clear, every `no_placement:board:<arm>_gripper_finger_link1` -5.8..-9.2 mm; the hulls at the
  pre-shape clear by 1.5-2.9 cm in all of them. Clearing spine pinches existed in the dense lean_left runs (+1.2 mm, 7.1 cm below the
  centre) but never became representatives.
- `grasping/access_region.py`: the region from the goal's interior spec - approach into the compartment from the open front (<= 45 deg),
  the contact within the pads' reach of the front face (0.057 m), every hand sphere at the placement opening clear of the goal's boards
  (lean board, opposite wall at width_m, board above at headroom_m) by the obstacle gate's margin; candidates restricted to it before the
  NMS and ranked by that clearance. Identity (digest / tag `ar<d6>`) in the job, the entry id, the library and its identity. No gate changed.
- lean_left 16.7 deg, `--interior-goal bookcase_jaysra_0@0.9472`, left-arm validator (entry `..._lean_left_16.7_2297df__valleft__ar45e7d8`):
  69 side candidates -> 46 approached from the front -> 46 in reach -> 20 clear -> 4 selected -> 2 geometry-ok -> 2 reps -> 1 left-arm valid
  (side_3, a spine pinch 1.9 cm inside the spine; the other `no_placement:left_realsense_link`); interior maps right 188 / 269 / 257 / 231,
  left 173 / 251 / 226 / 203 feasible at d_e 0.55 / 0.65 / 0.75 / 0.85.
- Pick compatibility (rule 4, `access_region.pick_compatibility`): the region grasp is 8.7 mm / 34 deg from the overhang entry's
  side_2~r180 (the overhang pinches approach 30 / 60 deg off the spine normal, this one straight) - outside a 1.5 cm / 30 deg match;
  lean_right's side_3 3.9 cm / 180 deg. The held grasp at the goal stays the overhang entry's (F_goal is exact); the lean entry is the map.
- Validation `i7_oh_R` (trunk_0adc84c9, v2-root `$OFH/v2maps_i7oh3` = v2maps_i7oh2 with the region lean_left, cap 0.70): stance C left 36
  survivors, right none (lean_right's map is empty at <= 0.65); 14 attempts, 4 written, 4 demos, 2 successes, 0.451 GPU-h per success;
  picks 31 / 31 and places 4 / 4 by the LEFT arm (all at stand-off 0.65); demo failures failed_grasp 1 (the push's differential IK stalled,
  the grasp then missed), collision 1 (the carried book's base_link touched the bookcase in the navigate); 0 robot_wobble, 0 self-contact.
- touching_03 acbrnv (report only): lean_left_11's 4 stored candidates are all outside the region (-1.0..-1.6 cm), a rebuild would
  re-select from the pre-NMS set; lean_left_20: 2 of 12 in the region (side_9, already the one valid grasp, and side_11 untested at
  +7.5 mm) - its two invalid representatives would not be tested.

#### Stance C trunk-first order (owner decision 2026-11-07, option 1 for inside_large_01; branch i7-placeorder)

Owner's wording: "Remove the 0.70 m cap: generation.inside_stand_off_max_m back to 0.85, or removed, for inside_large_01, so the right arm
gets its 0.85 m place stances back. Add a stance C ordering option for inside_large_01 only: least trunk translate first, then nearest
stand-off. More extended trunks are tried only after less extended candidates have failed. It changes the order, not the set: no stance
removed; exact IK and the QA gates unchanged. Kit stance_c.extended_used. Evidence: no global link between trunk extension and place
failures. All 3 place failures are inside_large_01 at translate 0.50 / tilt -0.6 / stand-off 0.85; translate 0.25 at 0.85 went 4/4."

- Code: `iors_large.trunk_first_order(c_pool, order)` - a stable sort of the coupling's prior C order (travel / edge rank / the seeded
  permutation) by (trunk translate, stand-off) ascending; the set and the per-coupling budget (`c_candidates_max`) unchanged. Opt-in
  `generation.c_order_trunk_first` (default off: every other alias as before), applied in `IORSLarge._couple` right after the prior order and
  BEFORE the per-alias outcome memory's hook (i7-memory `alias_memory.reorder`, which keeps the order it is given and only defers the
  candidates of the bins it down-weighted). Kit `stance_c.trunk_first` (couplings ordered) and `stance_c.extended_used` (a coupled C with a
  trunk translate above the pool's least); the summary's `c_order` {rank, translate, stand_off, least_translate, extended}.
- inside_large_01 customs: `c_order_trunk_first: true`; `inside_stand_off_max_m` removed (the option itself stays, default none): stance C
  lays out every stand-off 0.55-0.85 again.
- The campaign-wide check behind it (`$OFH/placeorder_20261107/demos.jsonl`, 2340 demos under `$OFH/out`; goal-stance failures
  robot_wobble / robot_touching_itself / plan_failed per demo reaching the goal stance, by its trunk translate): all demos 0.0 19/508,
  0.25 51/711, 0.50 16/520, 0.75 0/90, 0.95 0/94; pick-and-place 0.0 3/98, 0.25 3/134, 0.50 5/70, 0.75 0/11, 0.95 0/40. The place-time
  wobble / self-contact are inside_large_01's 3 at (0.50, -0.6) / 0.85; the campaign's other wobbles are grasp-stage.
- Tests: `test_trunk_first_order.py` (the order, ties kept in the prior order, the set unchanged, the coupling's opt-in, the alias only).
- Validation `i7_po` (trunk_032203c9, v2-root `$OFH/v2maps_i7oh3`, slurm 18372141, outcome memory OFF - i7-memory not merged; its evidence
  resets on the customs change): 7 attempts, 4 written, 4 demos, 2 successes, 0.262 GPU-h per success (i7_oh_R 0.451, i7_oh_m 0.234);
  rejections target_placement 2, approach_untrackable 1; 0 missing_map / MAP_REQUEST. Places 4 / 4 by the RIGHT arm, all at trunk (0.0, -0.6)
  / stand-off 0.75 (C rank 0-1, `stance_c.extended_used` 0, `stance_c.trunk_first` 45 couplings); picks found 10 right / 2 left. No place
  failure: both placements that ran succeeded; the 2 failed demos are plan_failed at the pick's grasp (the right arm's protected-object
  re-plan from the lifted hand: no plan / still passing the book at 9 mm). 0 robot_wobble, 0 self-contact.

### Lean rests and interior maps (branch i7-lean, 2026-11-07)

- **Lean rest kind** (`grasping/lean.py`): the object settled in the synthetic world against a vertical board at an angle / side; the rest id,
  the entry-id suffix and the sampler key `<key>@lean_<side>_<deg>_<d6>` carry the lean identity (angle, side, board, gap); a lean rest never
  serves a stable lookup or the reverse. `pipeline lean` (job_glib `GB_STAGES=lean`), manifests from `python -m ispatialgym.grasping.lean manifest`.
  touching_03: hardback acbrnv at 11 / 20 deg on the left board (partner_side right); 11 deg has no geometry-ok grasp; the pair check measures
  the key's own 15 deg rest (893d7e61).
- **Interior map kind** (`reach_maps --kind interior --interior <name | registry.json::name>`, specs in `sampling/interiors.py` +
  `assets/interiors/interiors.json`): the support board (policed), the board above, side walls, back panel, open sides, extra oriented boxes
  (an open door at the spec's joint values, a cabinet over a basin), the stand-offs and the joints - all in the spec digest, the map key and
  `ReachMap.find`. A lean rest's compartment follows the cell (its board-side wall); a stable rest's is fixed at `centre_y_m`, the footprint
  between the walls, and its trunks keep the idle posture clear of the obstacles (b4230177: the fridge's untucked idle arm stood in the open
  door - 0 feasible). Open-top basins: open when the walls' rim is >= 0.10 m under the headroom (a sink under a cabinet), the near wall and the
  overhead body as boxes. `interiors measure` (a Kit job) writes a spec from the live scene.
- **Consumers:** touching_03's shelf path (`tabletop.interior_map_bands`, a miss = `missing_map`); `iors_large_inside.inside_band` under
  v2only (fridge inside_large_02, sink inside_large_03, bookcase inside_large_01): the arm's own interior map, a miss = `missing_map`.

## Limitations / not done

- Scene articulation (dishwasher, doors: swept-volume neighbourhoods, partial scene loads) is not implemented; the tabletop articulation
  validator runs, but its pulls stop where the demo tracker's differential IK lags the braked lid (`SkillError ... pos_err 12-15 mm`,
  blenders 51-73 % of their range), and the hybrid's base move is a labelled teleport between segments, not the demo base tracker.
  Prefix maps do not pose the moving link for the arm-vs-object check; brace maps are single-arm proposals.
- The in-process map fallback inside the sampler is not built: a miss keeps the legacy band path (or rejects with STRICT).
- Force thresholds are uncalibrated; `hand_sphere_check` (palm / camera spheres in addition to their hulls) is conservative and rejects
  some hull-valid grasps (lid__tnwisp).
- Interior shelf planes are authored values (1e-2 m), not measured; the 88-height pool assumes subtask-level compatibility.
- Twin physics is assumed identical to its row (symmetric fingers); twins get their own geometric checks in maps.
