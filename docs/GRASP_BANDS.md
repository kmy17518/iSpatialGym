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
(§2.2), and the v2 library has no floating constraint (i7-floating 2026-11-07 added it and links them in a staged root: "Floating grasps
from GraspGen" below). Map types without a consumer (brace, floor, interior, tray, articulation prefix)
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
- The support split is kept ON. It acts (`support_split.applied` > 0) only on the 10 disembedding aliases. There, A/B N 4: ON 36 / 36
  demos and 0.173 GPU-h per success, vs OFF 33 / 36 and 0.184. On the other aliases it never acts, so ON and OFF ran the same
  sampler; their gap (61 / 62 vs 46 / 58) is run-to-run variance at N 4. (Corrected 2026-11-07: the first summary, 97 / 98 at 0.182
  vs 79 / 94 at 0.220, mixed the two groups.) An A/B claims an effect only where its switch's counter shows it acted.
- The support policy polices only supports with a registered plane. world_vertical + articulation_large went from 0 to 36 written / 34 ok under v2only.
- Floor maps: the fingers meet the floor only through the policy's floor allowance (`floor_fingers: "policy"`). The umbrella went from 0 to 71-73 cells per reaching trunk.

### Owner decisions 2026-11-07: a grasp ladder for every v2 entry; floating grasps from GraspGen

**The ladder.** Every library entry climbs these steps until it has physics-valid grasps:
1. Round 1: the standard build.
2. Round 2: the retry for missing grasps (the existing retry ladder, plus per-object fixes read from the gate logs).
3. Agent proposal, run automatically for any entry still at 0 physics-valid grasps after round 2. It works like the
   articulation library's handle grasps (geometric proposals, rendered strips, reviewer agents under a contract:
   `runlogs/grasp_library_20260905/artic/handle_review_contract.md`) and reuses that interface.
   - The agent proposes grasps from the object's geometry (collision pieces, extents, meta-links) and its task role.
   - The proposals go through the SAME v2 gates and physics validation; no gate is relaxed for them.
   - The entry records source "agent".
   - When a gate rejects every proposal, the gate and its measured values are logged (gate_dump) and reported.
4. Last resort: the legacy grasps (`runlogs/grasp_library_20260905`), re-validated by the v2 gates and physics. Every
   alias that would end here is reported to the owner before they are used.

**The four objects** (round-2 fixes first, then steps 3-4 if still at 0):
- **Cork ncxgpe (attachment_03).** The attachment rule passes 0 of 28,181. It reads the mating face through the meta-link
  frame [0.5, -0.5, 0.5, 0.5], which turns -z into +y.
  - Fix: derive the mating face from the same transform as the oracle's seat axis.
  - A unit test for every attach object checks that the rule's axis equals the oracle's seat axis.
  - The rule is shared by every attach object: the entries whose counts change are listed for the owner before adoption.
- **Tape kavsnx (disembedding_07).**
  - First, gate_dump shows which n_sweep test rejects it: finite width, min width or pad sweep.
  - Then the plate families (6e418172) are tried through cand_overrides: lip_shallow, and rim_straddle with the inner
    finger in the core hole.
  - If the cause is a fingertip clearance stricter than the spec's 2 mm finger-geometry rule, it is reported first,
    because it is a shared gate.
- **Cap xxqyrt (attachment_02).**
  - Validate every attachment-rule survivor, not just 1 (max_representatives / topk).
  - Densify the top pool and gate_dump the final finger_blade_in_object margins.
  - Check whether the cork's frame issue also affects the cap.
- **Storage box eicujj (touching_large_03).** It never had a v2 run. Add it to a manifest, build and validate it, then
  build the maps its MAP_REQUEST lines ask for.

**Floating grasps.** The legacy floating grasps came from GraspGen. That generator is ported into the v2 library as a
"floating" constraint (no support).
- Physics validation for a held object: in hand, under gravity, plus the task's held motions.
- Floating goes into the entry identity.
- The floating keys are linked: mental_rotation x10 and object_scaling x10, refused today as legacy (held_object §2.2).
- Maps are built only where these aliases' consumers ask for them (their MAP_REQUEST lines under v2only are read first).

**Validation** (only what is needed):
- Per object and per ladder step: candidates -> gates (gate_dump) -> physics-valid.
- Unit tests for every changed rule; existing keys, rests and choices stay unchanged.
- `--grasps v2only` N 4 on attachment_02, attachment_03, disembedding_07 and touching_large_03.
- Floating: `--grasps v2only` N 1 on the 20 aliases, and N 4 on one mental_rotation and one object_scaling alias.
- No wider no-regression run: the 20 map aliases ride along in the next scheduled run.

**Ask the owner before:**
- changing the shared contact-policy defaults;
- any gate shared by all objects (including the attach-rule fix if it changes other entries);
- any oracle program change;
- using legacy grasps (step 4).

### Owner decisions 2026-11-07 (five: plate switches, the "top" label, goal side, articulation maps, optional contact steps)

1. **The two plate switches** (`blade_contact_at_retreat`, `retreat_shift_exact`) are adopted the way the cork's attach-rule fix is
   handled.
   - First confirm the switch state is part of the entry identity (the digest / config the reader checks). If it is not, add it.
   - Count on the campaign library only: the v2maps_i7 glib entries and the entries of the staged roots in use. Per entry, compare
     candidates -> geometry-ok -> reps with the switches off and on, and note whether any recorded pose (retreat shift) changes. A
     candidates-only run is enough.
   - Report every entry whose counts or recorded poses move. The count can go both ways.
   - Rebuild (validate + maps) only those entries, then make both switches the default for new builds.
   - plate luhkiz keeps its current entry (9 physics-valid, switches off).
2. **The 62-67 deg plate family keeps the label "top"**; the question is closed. The label is build-only (retreat rule, blade gate,
   ranking bonus); no sampler, map or oracle code reads it, and relabelling would invalidate the 9 validated grasps. If tilt ever
   matters, it is recorded in the grasp record, not by relabelling.
3. **Goal side under v2only: no NO_BAND_TRUNKS fallback.** Interior maps are built instead with the interior kind
   (interiors.measure -> spec -> GB_KIND=interior maps, as for the fridge):
   - inside_large_03: the sink basin drop_in_sink_lkklqs_0 (surface ~0.77);
   - inside_large_01: the bookcase board (~0.94 m).
   - The consumer is iors_large_inside.inside_band. The fridge's "no stance C" issue lives in the same consumer, so a fix there
     applies to all three.
4. **Articulation maps.**
   - open_02 maps only the 21 pinch variants the oracle executes; handle B is dropped. The yield per variant is checked from two
     sources: validate_artic's physics trials, and a tally of the variant each past successful open_02 demo executed. Variants that
     never succeed are reported.
   - Left arm: right-arm segments are NOT reused. One left-arm physics validation per object where the task offers the left arm
     (open_01, close_02), and the left-arm maps are built from those segments.
5. **Contact maps under v2only.** A missing map for an OPTIONAL step skips that step and still logs the MAP_REQUEST (counted); for
   example, a missing axial brace map means no brace, and the push runs by its own rule. A missing map for a REQUIRED step (e.g. the
   touching brace gate) still rejects missing_map. Every i7-contact consumer is classified as optional or required.

### Owner decision 2026-11-07: reach-map trunk eligibility per stand-off (a bug fix, the new default)

**The bug.** reach_maps.py (~648-669) drops a whole trunk when no idle-arm posture is free of the map slab at EVERY stand-off.
- On nextto_04, the gym_shoe kmcbym map at 0.3471 loses t0.95 entirely: untucked / ready clip the slab at d_e 0.35 only ([False, True, True]).
- The 0.45 / 0.55 columns the sampler uses are therefore lost too: 0 cells, against 70 per arm in the legacy band.

**The fix.**
- Eligibility is judged per stand-off: a trunk is ineligible only at the d_e where every posture collides. The other d_e keep their cells.
- Trunks that were already eligible under the old rule keep their posture choice, so existing cells do not move. If a per-d_e posture
  choice would change cells of such a trunk, that is reported instead of done.
- The rule goes in the map identity (config), so maps built under the old and the new rule cannot be confused. Old maps keep loading;
  their identity reads as the old rule.
- Unit tests: the nextto_04 case; an all-clear trunk unchanged; an all-colliding trunk still ineligible.

**Count before rebuilding**, on the campaign maps only (v2maps_i7 and the staged roots in use):
- per map: the trunks / (trunk, d_e) columns that become eligible, and the feasible cells old -> new;
- rebuild only the maps whose cells change. Then the new rule becomes the default for new map builds.

**Validation:** the count table; `--grasps v2only` N 4 on nextto_04. Aliases with rebuilt maps ride along in the next scheduled run.

### Owner decisions 2026-11-07: inside_large_01 (hardback bqvvvx, the v2 overhang path)

1. **Overhang pick, finger clearance: keep the new rule.**
   - On the v2 overhang path the pick goes to the library's physics-validated rest (spine 0.0294 m past the edge, COM 0.0555 m inside),
     not the old 1 cm finger-box clearance. The box rule plus the 5 cm COM margin leaves no flush push.
   - `v2_finger_clearance_m` stays as the off-by-default fallback that restores the box rule.
   - Unit test: the default on this path is the library rest, and the switch restores the box rule.
   - Scope: this path only; not a gate shared by all objects.
2. **Left-arm goal side: a left-arm validator for lean rests** (no mirroring of the right-lean entry).
   - Today the lean validator runs physics with the right arm only, and the left board blocks it. So lean_left of
     hardback__bqvvvx__bu (16.7 deg, bookcase_jaysra_0@0.9472) has no valid spine pinch and an empty interior map.
   - A left-arm physics validation for lean entries, like the articulation left-arm validation: the arm that validates is recorded in
     the entry and its identity.
   - Rebuild lean_left with it, then its interior maps (both arms).
   - Lean rests only; ordinary tabletop validation is unchanged.
   - Whether touching_03's acbrnv lean entries would gain from it is reported only (not rebuilt).
3. **Far place: keep the place positions nearer.** The shared QA gates stay unchanged.
   - All 3 place failures (robot_wobble 1.01 / 1.03 deg vs the 1.00 gate; right_arm_link5 vs torso_link4) were at the right arm's far
     place, stand-off 0.75-0.85 m.
   - The place stand-off for this goal is preferred, or capped, at about 0.7 m: a task / generation option of inside_large_01, not a
     change to the shared stance or QA rules.
   - If the cap leaves an arm no stance, that is reported, not widened.

### Owner decision 2026-11-07: gym shoe kmcbym (nextto_04) after the switch default

With the two plate switches on (the default), the rebuild of gym_shoe__kmcbym__s1 has 0 physics-valid grasps: its new representative
fails object_pressed_into_table.
- **Meanwhile:** nextto_04 keeps the old switches-off entry (1 valid), staged. The switch state is in the entry identity, so the two
  builds cannot be confused.
- **Ladder round 2, with the switches on:**
  - the existing retry ladder (more representatives validated, a denser pool, a wider cone);
  - per-object fixes from the gate logs: gate_dump on, and for every object_pressed_into_table rejection the measured load
    (x weight, finger peak) and the stage where it is read;
  - a check of whether the old valid grasp (or its family) is still among the switches-on candidates, and why it was not selected or
    did not validate.
- **If round 2 gives >= 1 valid:** it replaces the old entry, with maps at 0.3471 (both arms) built once under the per-stand-off
  trunk-eligibility rule. It is staged in place of the old entry: the old entry and maps go to _superseded_switches_off in the staged
  copy, and legacy_keys are kept.
- **If round 2 gives 0:** ladder step 3 (agent proposal), same gates and physics, source "agent". The old entry stays staged until step 3
  yields >= 1 valid grasp.
- No legacy grasps (step 4) without asking.

### Owner decisions 2026-11-07: the pad sweep at the oracle's width; no grasps for alignment targets

1. **The pad sweep starts from the oracle's pre-shape width, not the fully open gripper.**
   - The oracle pre-shapes each grasp to its own width (on the tape, 16.8 mm inside the 35 mm core hole); the sweep started from fully
     open (48.5 mm), so the inner pad sat inside the opposite wall.
   - This is a shared gate change, approved by the owner. It follows the switches' pattern: the state in the entry identity, a count on
     the campaign library, a rebuild of the entries that move, then the default.
2. **Alignment targets need no grasps.** The axial / world_vertical targets are pushed and braced, never grasped. Their v2 path is the
   contact libraries and maps.
   - The ladder (round 2, step 3, step 4) applies only to entries a consumer grasps.
   - The step-3 selector picked every library entry at 0 valid grasps, including the alignment targets (board game, paperback, magazine,
     keyboard, license plate, paper towel dispenser). That work is void and is not used.
   - The selector now filters on the grasp role.

### Owner decision 2026-11-07: support-first map lookup as an automatic per-support fallback

**Context.** nextto_04 works under v2only only with `ISPATIALGYM_MAP_SUPPORT_FIRST` on (13 attempts, 4 written, 4/4 demos, 0.058 GPU-h
per success). The bench's eroded top is one ~2 cm row, and the 8 cm map lattice never lands on it, so map-first finds no cell.

**The decision:** support-first is a generic, automatic fallback, not a global default.
1. nextto_04 gets it now as an alias option, so it does not depend on the automatic rule.
2. **The generic rule, per support:**
   - Primary trigger, geometric and decided before sampling: the support's usable region (the eroded top raster the sampler uses)
     cannot hold a map lattice cell. Its narrowest usable extent is below the map's lattice spacing, or no lattice point falls inside it
     at the template's stand-offs. The map's own lattice from its config is used, not a constant.
   - Safety net: if map-first yields zero cells on the support for a template, that template is retried support-first once.
   - Everything else keeps map-first; aliases where neither trigger fires are unchanged (same cells, same choices).
   - Exact checks stay exact; support-first changes only where candidate cells come from.
3. **Logging:** kit `map_support_first.geometric` / `.fallback`, with the support, the usable width and the lattice, plus one log line
   per support where it fires.
4. **Override:** a per-alias force on / force off stays available.
5. **Unit tests:** the narrow bench triggers it; a normal tabletop does not and its cells are identical; the zero-cell fallback fires
   once and is counted; the override works.

**Before adopting:** a report of where either trigger would fire, over the 20 map aliases and the 100 aliases (from the existing v2only
sample logs / supports). Anywhere besides nextto_04 is the A/B scope.

### Owner decision 2026-11-07: a goal-accessible grasp region for interior goals

**Context.** inside_large_01's left-lean rest (hardback__bqvvvx__bu, 16.7 deg, bookcase_jaysra_0@0.9472) has no usable grasp, even with
the left-arm validator (512e3139).
- Its candidates pinch 3.3-4.8 cm from the book's centre, where the finger meets the board.
- The only valid one pinches the page block deep in the bay, so its maps are empty.
- The working right-lean grasp is a spine pinch, 6.4 cm from the centre.
- The problem is where grasps are generated, not which arm validates.

**The constraint**, in the grasp library, for entries used at an interior goal (a lean or stable rest in a compartment / shelf / basin;
the interior kind):
1. From the goal geometry the entry is built for (the interior spec: open sides, board / wall / headroom clearances), derive the object's
   accessible region: the surface reachable from the open side with the fingers clearing the boards by the gates' margins. For a
   leaning book this is the spine edge facing the open front.
2. Candidate contacts are restricted to that region and ranked by board clearance. This is a candidate-generation constraint, not a
   relaxed gate.
3. The same v2 gates and physics apply, nothing relaxed. The region definition goes in the entry identity.
4. Pick compatibility: the grasp is held from pick to place, so each entry reports whether its accessible-region grasps also exist at
   the pick rest (for bqvvvx: the overhang pick's spine-edge pinches).
5. Diagnose first: the pinch positions that clear the board are measured on lean_right vs lean_left, so the region comes from measured
   clearances.

**Apply now:** lean_left of hardback__bqvvvx__bu is rebuilt with the constraint (left-arm validator), then its interior maps (both arms).
For touching_03's acbrnv lean entries, only whether they would change is reported.

### Owner decisions 2026-11-07: swept hover, the seat attach axis, the hand-overlap screen, floating grasps

1. **Attach hover rule: SWEPT becomes the default** (an oracle program change, approved).
   - Evidence, box vs swept at N 4: touched 5/9 vs 11/11; drop up to 22.2 mm vs <= 0.1 mm; protected_contact 4 -> 0; demos 4/14
     (0.698 GPU-h per success) vs 5/15 (0.491).
   - It lands together with the attach maps: i7-attachhover (the oracle rule) with i7-attachmap (the maps for both rules, plus the
     consumer that reads the map of the rule the oracle runs). Never one without the other.
   - If the integration takes long, the pair is merged first, the default flipped, and attachment_01-04 validated (v2only N 4).
2. **attach_axis="seat" (the oracle's seat axis) becomes the default** for the attachment rule.
   - Count: gains for the cork, dip candle, light bulb, webcam and trout; losses for the cabinet door and the wreath; the 24 lids /
     caps are unchanged.
   - The axis goes in the entry identity, the default flips in its own commit, and only the moving entries are rebuilt.
   - Reported, not fixed: the oracle's 45 deg rule picks a doubtful centroid -> female axis for the skateboard deck and the wreath /
     holly partners, and "seat" copies it.
3. **The hand-overlap screen is not relaxed globally.** First, why it over-rejects on the napkins: 42/48 screened entries are
   physically valid there, vs 2/9 for the cereal and 1/23 for the spray.
   - If the screen's hand model is too conservative: fix the model, counted per object before / after.
   - If the overlap is real but harmless in physics: the screen becomes a ranking hint for push entries, and physics decides.
   - Either is a shared-gate change: the cause and counts are reported first.
4. **Floating grasps.**
   - Re-seed GraspGen for the mug, statue and teaspoon (low priority). The teaspoon's limit is physical, so a re-seed that does not
     help is reported rather than forced.
   - The statue: mental_rotation-statue at v2only N 4 first. If the stutter repeats, diagnose it and prefer a SAMPLER fix (a program
     check that screens untrackable grasp / orientation pairs); an oracle change only with the owner's OK.
   - The held-motion check matches the task, not stricter: it validates the motions each task's sampler draws, with the robot placed so
     they are reachable, and drift thresholds calibrated on the collected demos. Entries are rebuilt only if their verdicts change.

### Owner decision 2026-11-07: CR_FAST_ASTAR (camera_relocalization obstructed)

**Context.** teddy_bear was slow, not blocked: each BEHAVIOR A* call rebuilt a whole-map score table (2.6 s per call, 8.6 s when the goal
is unreachable). CR_FAST_ASTAR (i7-unblock 72d684a8) runs the same search without the table.
- Identical paths 12/12 on the real map.
- teddy_bear N 4: 8 attempts, 4 written, 4/4 demos, 0.063 GPU-h per success; attempts take 9-61 s, were 131-707 s.

**Decision.** CR_FAST_ASTAR becomes the camera_relocalization default if a same-instance test passes. It lands with the integration.
1. **The test:** the slow obstructed variants (vacuum_1, toy_car_1, trash_can_1, with teddy_bear_1 as the reference), each sampled twice
   from the same seeds, CR_FAST_ASTAR off vs on, with CR_OBSTACLE_HEADROOM off in both.
2. **Compared:** written instances, paths, attempts and rejections per stage, and wall time. Any differing instance is a finding to
   explain before adoption.
3. **If all match and it is faster:** CR_FAST_ASTAR becomes the default in a separate commit; the old search stays reachable by the
   switch, and a unit test checks for the same path on a fixed map.
4. **CR_OBSTACLE_HEADROOM:** only how often it would skip goals is reported; its default is unchanged.

### Owner decisions 2026-11-07: plate qtfzeq (under_small_03), the place stage, an expanding tilt / yaw map

1. **Plate qtfzeq: one merged entry.** The plate families (lip_shallow + rim_straddle) are adopted for this entry and merged with the
   switches-on rebuild into ONE entry: one build, all grasps validated together, one set of maps.
   - The side grasp side_9 is physics-valid in the library but fails in the alias's sampling (empty F, grasp chains exhausted, 10 of them by
     the support policy's "non-finger link touches the support"). It is diagnosed and fixed.
   - Fallback: if a chosen side grasp then still fails for a reason the sampler cannot fix, the sampler may try the overhang approach (as
     for luhkiz / bqvvvx). It is used only after the side grasp failed for that reason, and counted in kit.
2. **The place stage** (gripper out of frame ~13, partner placement ~12): diagnosed first, no rule change.
   - Which grasps, stances and place / pre-place poses are involved, and whether it is grasp choice, stance tilt or camera pitch; compared
     with v2legacy1.
   - The under relation (plate vs place mat) is checked against the task spec / BDDL, and the sampler and oracle must agree.
   - The camera-frame rule is not relaxed.
3. **An expanding, self-correcting reach map for tilts, then yaws** (instead of TILT_SNAP, which stays opt-in).
   - A stance tilt not in the map gets a provisional column: the union of the two bracketing built tilts, marked "inherited".
   - A cell is invalidated for that tilt only after a repeated REACHABILITY failure (IK / reach bound / joint limit; not a scene collision,
     the support policy or the camera frame). A successful exact check marks it confirmed.
   - Corrections live in an overlay file keyed by the map identity: never written into the frozen map, race-free for concurrent jobs, and
     merged offline (the overlay version enters the identity once used).
   - No extra GPU builds of true columns; building one is a separate decision.
   - Kit counters: map_tilt.inherited_columns / .inherited_feasible / .cells_invalidated / .cells_confirmed / .empty_f_on_inherited.
   - The same mechanism covers in-between yaws (the 8 x 45 deg slots).

### Owner direction 2026-11-07: finish the i7-braceopen and i7-fingertouch A/Bs before the integration

**A. i7-braceopen** (cb164782: the sampler checks a wide brace at the record's fingers_opening_m; dc10d4d2: `ISPATIALGYM_BRACE_FULL_OPEN`,
default off, with the full open 0.127 m spread first, the close_push / v4 stop openings at 0.05, and the pack brace capped at full open).
- **The carton stop:** the decided corner inset of 0.06 at the 0.05 opening is implemented as inset = max(corner_inset, opening + 1 cm),
  with `close_push_brace_corner_inset_m` still 0.04.
- **Merge with i7-contact first.** It defines `ISPATIALGYM_BRACE_FULL_OPEN` again (contact_poses.py) and also has
  `ISPATIALGYM_TOUCH_BRACE_AT_OPENING` (default on). These become ONE switch and ONE opening rule, and the contact libraries / maps use
  the opening the oracle executes. Unit tests; existing keys unchanged with the switch off.
- **The A/B:** off vs on for every alias whose brace opening changes (touching_01 / 02 / 04, articulation close_02, the carton variants,
  block_4; listed from the code), run after the current i7-contact fixes land. Only the aliases where the executed opening changed are
  judged.
- **Adoption:** only where no alias regresses (a regression is reported first). The default flips in its own commit.

**B. i7-fingertouch** (d7783b4b, af08d768: `ISPATIALGYM_FINGER_TOUCH_RULE` = spheres (default) | geometry).
- **The A/B is re-run only where the rule acts:** aliases with approach_finger_support_touch (inside_large_03, the plate-in-sink variants,
  under_small_03, and touching_03 once unblocked). It uses enough instances, or the same instances, to beat the ~15-point N 4 variance.
- **Diagnosis:** the geometry arm's extra rejections (3 approach_untrackable on inside_large_03), checked against the spec's 2 mm rule.
- **Adoption:** the default flips (own commit) only if geometry is no worse where it acts.

**Merge order:** both land after i7-contact's current work; i7-braceopen together with i7-contact.

### Owner direction 2026-11-07: start the integration now

- **One branch at a time.** The finished branches are merged into of-v2maps one at a time, and every other branch folds in as soon as
  its own work and validation finish. Owner-decided defaults land with their branch; nothing new is decided at a merge.
- **Ready now** (in order): i7-switches, i7-padsweep, i7-attach, i7-ladder, i7-floating.
- **Later:** i7-unblock, i7-overhang, i7-contact (+ i7-braceopen, one switch), i7-attachmap + i7-attachhover (as a pair; i7-attach-swept),
  i7-lean, i7-artic, i7-plateqtf and i7-fingertouch.
- **Per merge:**
  - Conflicts resolved preserving every owner-decided default; duplicated implementations (cherry-picks, duplicate switches) are kept
    once, and which one is said.
  - pyflakes; the full suite vs the baseline; the golden-key / identity tests pass.
  - The merged tip is snapshotted.
- **One consolidated v2 root** (new, $OFH/v2maps_int1), updated as branches fold in:
  - built from v2maps_i7 plus the merged branches' staged roots, with the anchors added so far and legacy_keys in the staged
    library.json files;
  - superseded entries are moved aside, 0 key mismatches, every link resolving.
- **Validation:** only on the aliases each merge affects (v2only N 4 from the merged snapshot and the consolidated root, batched where
  they overlap); aliases with nothing observable changed are skipped. The full 100-alias pass is a separate decision, after every branch
  is in.

### i7-switches 2026-11-07: the plate switches adopted (identity, count, rebuild, default)

Owner decision 1 above, done on branch `i7-switches` (884279fd identity, 16a0e27b default). Record: `$OFH/switches_20261107`.

**Identity (884279fd).** The switch state was NOT in the entry identity. `library.json` did not carry it, and
`ModelLibrary.identity()` (the reach maps' `object_key`) did not read it. Only the `GB_CANDIDATES` log line named
`blade_contact_at_retreat`, and it did not name `retreat_shift_exact`. So an entry built with the switches and one built without could be
told apart only by their grasp poses.
- `library.json` now records `candidate_switches` (from the candidates' `config`).
- `ModelLibrary.candidate_switches()` reads that record. For an entry validated before the record it reads `candidates.json`'s config.
  A config without the fields (every entry before 6e418172) reads all off.
- `identity()` adds the state only when a switch is on. Every existing key is unchanged (a golden-key test, and every map of
  `v2maps_i7` still matches its entry under the new code).
- `check_compatible(candidate_switches=)` / `load_model(require=)` let a reader require a state.
- Tests: `ispatialgym/tests/test_glib_switches.py`.

**Count** (candidates only, CPU; the switches off / blade only / shift only / both on; each entry's recorded config and densify;
`count/count_table.md`, trunk_884279fd).
- Scope: 44 entries. These are the 38 of `v2maps_i7`, plus the ladder probe's tape kavsnx and storage box eicujj, plus the 4 hardback
  lean / overhang entries of `v2maps_i7lean` / `v2maps_i7oh`. The hardback entries were counted on lambda with the i7-overhang code and a
  scratch port of the two switches. Plate luhkiz is excluded. The other staged roots hold no other entries.
- With the switches off, the count reproduces every recorded `candidates.json`: all candidates, poses, shifts, verdicts and
  representatives.
- 19 entries do not move, including all 4 hardback entries: their only top candidates fail before the switches act.
- 25 entries move (candidates / geometry-ok / representatives, off -> on):
  - Geometry-ok up on 14 entries. Examples: tape 30/3/3 -> 30/20/16 (blade), tablefork 12/5/4 -> 12/11/10 (blade), jigsaw 7/4/3 -> 11/10/8,
    plate qtfzeq 12/3/3 -> 22/7/7 (shift), garlic 12/4/3 -> 12/8/5, briefcase 5/1/1 -> 6/2/2.
  - Geometry-ok down on 2 entries: salt shaker 24/23/8 -> 24/20/5 (the blade gate rejects 3 representatives that passed before) and
    storage box 38/36/11 -> 38/35/11.
  - The other 9 keep their geometry-ok count and move in their candidates' poses or recorded retreat shifts (5 mm). Box of cereal,
    carton of milk and teacup keep their representatives; gym shoe, soap dispenser, napkins, shampoo and lid pbmsji swap 1, pencil
    case swaps 2.

**Rebuild** (validate + maps, switches on, everything else as built: the densify passes and validation overrides of each entry's build,
its maps' kinds / anchor heights / arms).
- Run as shiyuc, because minyeong was at its GPU cap (`sy_switches.sh`, trunk_884279fd). The 25 entries were copied back into
  `$OFH/switches_20261107/{glib,maps}`.
- Physics-valid grasps, old -> new:
  - Up on 9 entries: tape 2 -> 12, tablefork 3 -> 6, plate qtfzeq 1 -> 4, hard candy 3 -> 5, briefcase 1 -> 2, lid tnwisp 1 -> 2,
    lid pbmsji 3 -> 4, jigsaw 2 -> 3, marker 5 -> 6.
  - Equal on 14 entries.
  - Down on 2 entries:
    - pencil case 2 -> 1. Its two valid top grasps are no longer selected, and the new top_0 fails object_pressed_into_table.
    - gym shoe 1 -> 0. Its only representative changed, and the new one fails object_pressed_into_table.
- 52 maps were built ok. Gym shoe has no maps (0 valid). Plate qtfzeq's reviewed rotational family of side_9 (owner prompt A 2a) was
  carried by pose and its 2 maps rebuilt after that, because the families digest is in the identity.
- `legacy_keys` / `_campaign_link` were carried from the old `library.json`.

**Staged** `$OFH/v2maps_i7sw` = `cp -a v2maps_i7`. Each rebuilt v2maps_i7 entry with >= 1 valid grasp and all of its maps replaces the
old one, which moves to `glib/_superseded_switches` / `maps/_superseded_switches` (`STAGED.txt`). 22 entries are staged.
- Not staged:
  - gym shoe kmcbym: 0 valid, so the switches-off entry stays. This needs an owner decision.
  - tape / storage box: they are not `v2maps_i7` entries; their rebuilt entries are only in `$OFH/switches_20261107`.
- The aliases of the rebuilt entries ride along in the next scheduled run.

**Default (16a0e27b).** Both switches are on for new builds. False in `cand_overrides` reproduces the old build. The overhang
branch's side-grasp retreat (i7-overhang, not merged) still looks the blade's pinched pieces up at CONTACT_Z, as the switch does for
every side grasp. Whether the switch should cover that retreat is a question for the merge.

### i7-switches 2026-11-07: gym shoe kmcbym ladder round 2 (switches on)

Owner decision 20:20. Record: `$OFH/switches_20261107/gymshoe_r2`.

**Where the old grasp went.** The old valid grasp (top_0, switches off) still passes every gate with the switches on (`gs_trace.py`).
- Its recorded shift is now the real 10 mm, not 15 mm, and its width is re-measured at 20.8 mm, not 22.3 mm.
- It reaches NMS ranked 2 of 2. A neighbour 4.8 mm / 30 deg away suppresses it: score 8.52 against 7.78. That neighbour passes object
  clearance only with the exact retreat (its 15 mm try).
- The neighbour becomes the only representative and fails object_pressed_into_table.
- The old pose itself is still physics-valid under today's validator (`diag_old`: 1 of 1).
- No round-2 candidate lies within 10 mm / 20 deg of the old pose; the denser pool samples other poses.

**Round 2** (the retry levels of `v2_retry_20261107`, switches on, gate_dump on; candidates / geometry-ok / reps -> physics-valid):
- L2: 15 / 12 / 12 -> 6 valid. L3: 24 / 23 / 23 -> 6 valid. Validation stops at max_valid 6.
- Each level has one object_pressed_into_table, read during the close: object load 29.1x weight (L2) and 30.6x (L3) against the 2.0x
  limit, sustained 11 / 13 steps, finger peak on the slab 0 N. The validator now records these per phase
  (`forces.diagnostics`, f5182aa4).
- The pad sweep is the largest late gate (all `touch`, none `width_lost`): top 97 -> 33 and side 4 -> 1 at L2; top 269 -> 110 and
  side 29 -> 1 at L3; top 14 -> 2 at round 1. So the planned pre-shape sweep could change this entry's candidates. Its physics rejections
  do not involve the sweep.

**Replaced** with L2 (the first level that recovers).
- Maps at 0.3471, right and left, built once under the per-d_e trunk-eligibility rule. i7-unblock 1527d9b8 / 58cd9939 / 722d687f were
  cherry-picked onto i7-switches; snapshot trunk_0b6a7284.
- Feasible cells right 91 -> 127 and left 92 -> 124 of 253, against the old per_d_e-merged maps; grasps 2 -> 12 (with twins).
- Staged `$OFH/v2maps_i7ugs` = `cp -a v2maps_i7unblock` with the old entry and maps in `_superseded_switches_off`. Reader check:
  38 entries, 88 maps, 0 key mismatches.
- nextto_04 `--grasps v2only` N 4 with ISPATIALGYM_MAP_SUPPORT_FIRST=1, tag `i7_sw_gs_sf`: see DECISIONS.

### i7-padsweep 2026-11-07: the closing sweep from the oracle's pre-shape (adopted)

Owner decision (a shared gate, approved after the tape's diagnosis: the sweep started at the fully open pads, so a wall pinch next to a
hole started its inner pad inside the opposite wall). Branch `i7-padsweep` (47b27893 switch + identity, f6dc7183 default). Record:
`$OFH/padsweep_20261107`.
- **Rule.** `candidates.oracle_preshape_q(width)` = `execute.execute_grasp`'s opening, 0.5 x the recorded pinch width +
  `MotionsConfig().gripper_preshape_margin` (read from the oracle's config), capped at fully open as the validator's trial. The sweep starts
  at `preshape_pad_half` (its own open-pad convention: `PAD_OPEN_HALF` at the fully open fingers, less what the pre-shape closed); a ray
  that starts inside a piece meets the object at its start. Off, `pad_sweep` is the code it was.
- **Identity.** `sweep_from_preshape` is a candidate switch (`schema.CANDIDATE_SWITCHES`, recorded in library.json). The identity keeps
  the plate switches as the pair 884279fd wrote and adds the new switch only when on: no existing key changes.
- **Count** (candidates only, plate switches on; `count/count_table.md`, trunk_47b27893). 44 entries: v2maps_i7's 38 + tape + storage box
  (cluster) + the 4 hardback lean / overhang entries (lambda, the i7-overhang code with a scratch port; off reproduces each, on = off).
  6 move (c/g/r): tape 30/20/16 -> 96/19/15, jigsaw 11/10/8 -> 36/31/17, sandal 12/12/10 -> 13/13/10, teacup 13/5/4 -> 14/6/5, gym shoe
  1/1/1 -> 5/4/4, pillow 2/1/1 -> 1/1/1 (a geometry-failed pose gone, representatives the same). 38 do not. The "on" arm differs from the
  switches count's own on arm on bottle of milk, hard candy and sunflower by 1-3 borderline poses (no sweep involved: ps = on there) - a
  run-to-run difference of the same code on another node.
- **Tape without the plate families:** the top grasps now pass the sweep (top 1030 -> sweep 301, was 0) -> 36 selected -> 4 geometry ok
  (32 finger_blade_in_object) -> 4 reps.
- **Rebuild** (validate + maps; gym shoe excluded: ladder round 2 by i7-switches). Valid old -> new: tape 12 -> 12, jigsaw 3 -> 6,
  sandal 6 -> 6, teacup 4 -> 5, pillow 1 -> 1. 14 maps ok (tape tray floor 0.6791 r + l; sandal floor + 0.4923; jigsaw 0.8126 / 0.8186).
- **Staged** `$OFH/v2maps_i7ps` = `cp -a v2maps_i7sw` + the 5 entries (old ones in `glib/_superseded_padsweep` /
  `maps/_superseded_padsweep`); the tape added with the exact tray-floor anchor 0.6791, `legacy_keys` and its link (`STAGED.txt`).
- **Default (f6dc7183).** On for new builds; False in `cand_overrides` reproduces the old build.

### i7-attach (2026-11-07): the attachment rule's seat axis (cork ncxgpe) and the cap xxqyrt, round 2

Branch `i7-attach` (from of-v2maps a4620a1e). Record: `$OFH/attach_20261107/{seats,counts,cork,cap,stage_cork,stage_cap}`.

**Cork ncxgpe: the cause, measured by the pipeline** (gate dump of the attachment rule, new: `side_mm`, `face_gap_mm`, the frames used).
- The rule (`candidates.attachment_frames`) takes the male meta link frame's -z as the mating direction. The cork's frame is
  (0.5, -0.5, 0.5, 0.5) in its metadata, so the direction is (0, 1, 0): the cork's +y side (8.78 mm out) is taken as the mating face.
- No finger may reach past that face by less than 2 mm. Every pinch of the 8.8 mm-radius cork straddles it: top 7087 / 7087 and side
  1747 / 1747 rejected (48000 x 48 pass). The largest face gap is 0.96 mm (top) and -20.2 mm (side).
- The oracle seats the cork along its own -z. `predicates.seat_axis` takes the female link's +z unless it is more than 45 deg from the
  direction partner hull centroid -> female link. On wine_bottle inkqch the angle is 90.4 deg, so the axis is (-0.0091, 0.0076, 0.9999)
  in the bottle frame (the sampler logs the same value for attachment_03).

**The fix** (`CandidateConfig.attach_axis`, an OWNER DECISION switch: default `"meta_z"` = as written; `"seat"` = the oracle's seat axis).
- A male seats with its link frame on the female link frame (`AttachedTo`). So in the male link frame the seat axis is
  `loc` = the oracle's axis in the female link frame, and the mating direction is `-(R_male_link @ loc)`.
- `loc` is recorded per pair in the simulator: `scripts/grasp_bands/attach_seat_facts.py` (job 18368109) loads each male and every
  partner with a matching female type (XM <-> XF, both spawned `attachable` like the attachment task) and calls `predicates.seat_axis`.
  The result is `ispatialgym/assets/grasping/attach_seats.json`: 33 males, 37 female links. Regenerate it if `seat_axis` changes.
- `"seat"` also takes the male link pose as the simulator loads it.
  - The USD frames differ from the metadata quaternions: cork ncxgpe 10.6 deg, webcam fabztm 8.1 deg, the wine bottle's female 3.8 deg.
  - The metadata positions are unscaled: lid pbmsji s0.65 is 4.4 mm off, caps clhquh / yqvild 1.2 mm.
- A meta link without a matched female keeps the frame as written; an entry without a seat record keeps `"meta_z"`.
- Opt in per entry with `cand_overrides {"attach_axis": "seat"}`.
- Tests: `ispatialgym/tests/test_attach_seats.py` (7). For every attach entry of the three libraries (34), the rule's axis equals the
  oracle's seat axis, rebuilt from the recorded live frames. The table equals what `predicates.seat_axis` computes today (stubs). `"meta_z"`
  is bit-identical to a4620a1e. A synthetic cork rejects every pinch under `"meta_z"` and keeps them under `"seat"`.
- Default unchanged, measured: a CPU candidates-only run of all 34 attach entries (round-1 config) on a4620a1e and on i7-attach (default)
  gives identical candidates, families, representatives and gate logs (`counts/base` vs `counts/off`).
- Full suite: 23 failed, 40 errors on both a4620a1e and the i7-attach tip, the same tests. The only difference is 7 more passed (the new
  tests). In this worktree the 3 expected errors read 40.

**Every attach entry, `meta_z` -> `seat`** (CPU, round-1 config 6000 x 12, densify (4,2),(8,4); candidates / geometry ok / representatives;
`$OFH/attach_20261107/counts/counts_off_on.json`). Ten of 34 change, so the fix stays behind the switch until the owner decides.

| entry | meta_z | seat | the oracle's seat axis (partner) |
|---|---|---|---|
| cork ncxgpe | 0 / 0 / 0 | 22 / 22 / 3 | centroid->female 90.4 deg (wine_bottle) |
| dip_candle poexjo | 0 / 0 / 0 | 24 / 22 / 5 | centroid->female 88.7 deg (candle_holder) |
| dip_candle poexjo `__rfecdee` (derived rest) | 0 / 0 / 0 | 12 / 12 / 6 | same; `"meta_z"` also rotates its mating point twice at a non-identity rest |
| light_bulb kfmkwd | 12 / 9 / 6 | 24 / 20 / 10 | centroid->female 90.4 deg (table_lamp) |
| webcam fabztm | 13 / 4 / 4 | 24 / 8 / 8 | centroid->female 89.4 deg (monitor) |
| trout kqpeke | 0 / 0 / 0 | 2 / 1 / 1 | centroid->female 88.1 deg (fishing_rod) |
| skateboard_wheel jljqtt | 24 / 18 / 10 | 24 / 18 / 10 (top 51 -> 25, side 47 -> 33 pass the rule) | centroid->female 95.8 deg: a horizontal axis along the deck |
| skateboard_wheel acmbvb | 24 / 24 / 10 | 23 / 23 / 10 | centroid->female 108.2 deg: the same |
| cabinet_door ertkre | 16 / 16 / 10 | 11 / 11 / 4 | centroid->female 86.6 deg (cabinet_base) |
| wreath rkrxno | 2 / 2 / 2 | 0 / 0 / 0 | wall_nail: +z (as written); recreational_vehicle: centroid->female 115.4 deg - both must hold |

Unchanged (24): every lid and cap, cork uyceta / lseuwf, blackberry, bookcase_baseboard and holly (0 under both), and the six without an
attachment meta link (lid qhlsdq, dip_candle ghctun / gbaons / mxjejs, light_bulb sxkjea / ymomhw: bottom centre). The campaign libraries
(v2maps_i7, v2maps_20261106) hold only the two lids: unchanged. Note: for the wall, deck and bush partners, the oracle's own 45-deg rule
picks the centroid->female direction, which is often not the insertion direction (skateboard deck, recreational_vehicle). `"seat"`
mirrors the oracle there too, so those entries follow the oracle's picks, questionable or not.

**Cork round 2** (`attach_axis` seat opt-in; L3 config 48000 x 48, cone 60, densify 2x1,4x2, reps 30; job 18368231):
- Round 1 (roles_all): top 7087 -> 0 and side 1747 -> 0 at the rule. L3 retry: top 28181 -> 0.
- Now, first pass: top 7087 -> rule 266 (6821 face_gap: the fingertips within 2 mm of the cork's bottom) -> NMS 16 -> geometry 15. Side
  1747 -> 1597 -> 19 -> 19.
- 34 geometry-ok -> 5 representatives (rotational families) -> **2 physics-valid** (side_0, top_15). The other three failed:
  object_pressed_into_table, grasp_off_target, no_placement.
- Reach maps were built at 0.7853 (desk_mdhelw_1), both arms.

**Cap xxqyrt round 2** (cone 60, first pass 192000 x 96, densify 2x1,4x2 up to 768000 x 192, NMS 1 mm / 1 deg so every rule survivor is
selected, topk / reps 60):
- The cork's frame issue does not apply. The male frame is the identity (live = metadata), and all four partners (lemonade, soda water,
  water bottle, tomato paste can) take the female +z branch at 1.1-5.7 deg. So `"seat"` = `"meta_z"`, with identical counts.
- A, default gates (job 18368124), 768k pass: top 66817 -> width/cone 39307 -> support 31361 -> object clearance 604 (the shell) -> sweep
  266 -> lever 100 -> rule 40 -> NMS 33 -> geometry **0**.
  - The rule drops 60: the fingertips sit 1.0 mm below the cap's bottom face, which is the support allowance.
  - All 33 fail finger_blade_in_object: the blades are 1.48-2.74 mm into the shell as called (tol 0.5).
  - At the real (retreated) contact, 9 of 33 are within tolerance: the i7-plate `blade_contact_at_retreat` case. The top grasps retreat
    25-28 mm.
  - Side: 9693 -> support lift -> object clearance 5031 -> sweep 0 (all width_lost: after the lift the pinch line misses the cap).
- B, the same plus `blade_contact_at_retreat` (an i7-plate owner-decision switch, opt-in for this entry only; job 18368125), 384k pass:
  top rule 18 -> NMS 15 -> geometry 4 -> 1 representative -> **1 physics-valid** (top_0: width 18 mm, tilt 30 deg). Maps at 0.8126, both
  arms.
- At default gates the cap stays at 0 physics-valid: ladder step 3 (agent proposals) unless the owner adopts `blade_contact_at_retreat`.

**The aliases under `--grasps v2only`, N 4** (staged roots `stage_cork` / `stage_cap`: `cp -a v2maps_i7`, plus the entry, its maps and
`legacy_keys` in the staged library.json):
- attachment_03 (18368426) and attachment_02 (18368413): 0 attempts each. Template setup rejects `no_stance` on every edge with
  `LEGACY_BAND_BLOCKED ReachBands.load(object=<partner>, grasp_key=attach:<male key>, placed=['attach:<male>'])`.
- The male's grasps and maps now serve: no missing_grasps or missing_map (before: missing_map).
- The attach band is the partner's reach band with the male's grasps carried to the seat. `tasks/attachment.py` loads it directly with
  `ReachBands.load`, not through `ensure_bands` / `v2_map_view`, so no MAP_REQUEST line is written.
- No v2 map kind covers it, so nothing could be built. It is the same blocker as attachment_01 / 04. The partner's left contact band
  (`load_left_contact_band`) comes after it and is untested.

**Owner decision 2026-11-07 item 2: `attach_axis = "seat"` is the default** (807a7314).
- Identity (84fa3b9f, on top of the i7-switches identity 884279fd, cherry-picked as 33e72d92): `library.json` records `attach_axis`
  for attach entries. An entry without the record reads its `candidates.json` config, else "meta_z". `ModelLibrary.identity()` adds it only
  when it is not "meta_z", so every existing key is unchanged. `check_compatible(attach_axis=)` lets a reader require it.
- Rebuilt with the default (`$OFH/attach_20261107/seat_rebuild`; glib + maps at the original heights / arms). Physics-valid old -> new:

| entry | old | new | maps |
|---|---|---|---|
| cork ncxgpe (L3 config, 0.7853 both arms) | 2 (seat opt-in) | 2 | 2 |
| dip_candle poexjo | 0 | 0 (measured-rest palm / finger, pressed) | - |
| dip_candle poexjo `__rfecdee` | 0 | 5 | 3 |
| light_bulb kfmkwd | 4 | 6 | 3 |
| webcam fabztm | 2 | 5 | 3 |
| trout kqpeke | 0 | 0 (1 rep, pressed) | - |
| cabinet_door ertkre | 6 | 4 | 3 |
| wreath rkrxno | 2 | 0 (no representative) | - |
| skateboard_wheel jljqtt | 6 | 6 | 3 |
| skateboard_wheel acmbvb | 6 | 2 | 3 |

  The roles entries use round-1 config with maps at 0.4230 / 0.8126 / 1.0505 (right arm), as roles_all.
- Staged: `stage_seat` (`cp -a v2maps_i7` + these entries and maps). The old versions are in `stage_seat/_superseded_meta_z/`.
- Not changed (an oracle change): for the skateboard deck and the recreational_vehicle partner of wreath / holly, the oracle's 45-deg rule
  picks the centroid->female direction, and "seat" copies it.

### i7-ladder 2026-11-07: the tape, the storage box, ladder steps 3 and 4 (branch i7-ladder)

Record: `$OFH/ladder_20261107/{tape,tape_sw,box,tape_on,box_on}` (builds; `_on` = both owner switches on, the new default),
`$OFH/ladder_agent/r2zero_20261107{,_build2}` (step 3), `$OFH/v2maps_i7ladder{,_probe,2,3,4}` (validation roots: `cp -a v2maps_i7` + the two
entries, their maps and links, `legacy_keys` set in the staged library.json; `ladder_20261107/stage_ladder.py`; 4 = the switches-on
one). Builds ran on the template's knobs (48000 x 48, cone 60, densify 2x1,4x2,
30 reps / 30 trials).

**Tape kavsnx (disembedding_07): climbs at round 2 with the plate families.** gate_dump, default gates, densest pass (192000 x 96):
- top: 33894 -> width / cone 6279 -> support 5876 -> object clear 3388 -> sweep **0**. All 3388 fail the pad sweep proper (`touch`);
  none fail finite width or min width (widths 3.05-13.57 mm: the ring wall). touch 41-48.5 mm vs limit 9.5-14.8 mm.
- The cause: the pad sweep casts its rays from the FULLY OPEN pads (`PAD_OPEN_HALF` 48.5 mm). A wall pinch (contact r 23.8-25 mm about
  the axis) puts the open inner pad at r ~ -24 mm, inside the OPPOSITE wall of the ~35 mm core hole. The oracle pre-shapes to
  width / 2 + 10 mm = 16.8 mm, inside the hole. Same class as a1117046 and the validator guard's widest pre-shape. It is NOT a fingertip
  clearance against the 2 mm rule; it is a shared gate, so it is not changed (owner question below).
- side: 62784 -> 41648 -> support lift 1973 -> 206 -> sweep 0, all `width_lost` (non-finite width): the lift (19-30 mm, driven by
  wrist / wrist camera / palm on the table) puts the pinch line above the 22 mm roll.
- With `families: [lip_shallow, rim_straddle]` (cand_overrides, default gates): lip_shallow 29380 -> 8450 -> 6889 -> 2968 -> sweep 20 ->
  11 selected -> **3 geometry ok** (8 finger_blade_in_object) -> 3 reps -> **2 physics-valid** (1 object_pressed_into_table, load 20x).
  The two: near-vertical (0.2 deg), contact 24 mm toward the tips, 1.9 mm below the roll top, across the 13.6 mm wall.
  rim_straddle 18000 -> 11995 -> 10350 -> 9763 -> 1487 -> 19 selected -> 0 (19 finger_blade_in_object by the call-site lookup: blade
  -1.6..-4.4 mm; at the real contact -0.15..-1.7 mm).
- rim_straddle "inner finger in the core hole" needs no new parameter: `rim_profile` finds the inner face from the free axis (the hole),
  the contact sits midway (r 21.6-22.4 mm), the inner finger at the pre-shape is in the hole.
- Owner switch arm (`blade_contact_at_retreat` alone, `tape_sw`): rim_straddle 15 / 19 geometry ok, lip 5 / 11; 16 reps -> 12
  physics-valid (the same counts as both switches on below: `retreat_shift_exact` does not change the tape).
- Maps: the alias's MAP_REQUEST asks h 0.6791, plane `tray_floor` (the tape rests in the source tray), not the ledger's h_live 0.6737
  (the conference table). A tray floor takes an EXACT anchor only (`v2_map_view`), so 0.6789 does not serve it: the validation root
  `v2maps_i7ladder3` adds the exact anchor 0.6791 (as `prep_tray.py` did for the other trays; 0.6789's range becomes [0.6733, 0.679])
  with tray-floor maps right + left at 0.6791. Roots `v2maps_i7ladder` (+ 0.6737) and `v2maps_i7ladder2` (maps at 0.6789) are superseded.
- Validation, v2only N 4, switches OFF (the a4620a1e default; the 2-grasp entry, `v2maps_i7ladder3`, tag i7_ladder3): 8 attempts,
  4 written, **4 / 4 demos**, 0.21 GPU-h (0.053 per success), no missing_map; grasps lip_shallow_2 / _4 and their twins.
- **Switches ON** (the owner adopted `blade_contact_at_retreat` + `retreat_shift_exact` as the default for new builds, i7-switches
  16a0e27b; my rebuild `ladder_20261107/tape_on` sets both through cand_overrides on a4620a1e): 30 candidates / **20 geometry ok / 16 reps
  -> 12 physics-valid** (rim_straddle 11 of 15 geometry-ok, lip_shallow 1 of 5; ag_not_formed 2, object_pressed_into_table 1,
  grasp_off_target 1). Without the families the switches give nothing (lambda CPU, switches on, families off: top 3619 -> sweep 0 all
  `touch`, side 206 -> 0 `width_lost`): the plate families are what make the tape graspable; the switches turn rim_straddle (the inner
  finger in the core hole) from 0 to 15 geometry-ok. Validation v2only N 4 (`v2maps_i7ladder4` = v2maps_i7 + anchor 0.6791 + the _on
  entries and their maps, tag i7_ladder_on): 9 attempts, 4 written, **4 / 4 demos**, 0.34 GPU-h, no missing_map, no legacy grasp; the
  demos use rim_straddle grasps (_4, _10, _12, _18). The families add everything on top of the switches: without them the tape has 0.

**Storage box eicujj (touching_large_03): round 1 is enough.** Manifest `ladder_20261107/box/manifest.json` (scale [1,1,1] = the
customs' null scale, legacy key storage_box__eicujj__t, support = the floor of house_double_floor_upper living_room_0, h_live 0.0).
top 5314 -> 3686 -> 3606 -> 1938 -> 1927 -> 1825 -> 36 selected -> 36 geometry ok; side 27879 -> ... -> 2 (approach_hits_object);
11 reps -> **11 physics-valid** (rim pinches of the 3.0-3.7 mm wall, tilts 0-37.5 deg). MAP_REQUEST (v2only probe i7_ladder_boxprobe):
`{"kind": "floor", "h": 0.0, "arm": "right", "rest_id": "r0", "support": "floors_rfqizg_0"}` -> floor maps right + left
(`GB_KIND=floor`, 86 grasp members each). Validation, v2only N 4 (`v2maps_i7ladder`, tag i7_ladder): 4 attempts, 4 written,
**3 / 4 demos** (1 `oracle_failed:not_settled`), 0.35 GPU-h, no missing_map, no legacy grasp; grasps top_3~r180, top_4, top_5~r180.
Switches ON (`ladder_20261107/box_on`): 38 / 35 / 11 -> **11 physics-valid**, floor maps rebuilt; v2only N 4 (`v2maps_i7ladder4`, tag
i7_ladder_on): 4 attempts, 4 written, **3 / 4 demos** (1 `not_settled`), 0.32 GPU-h; grasps top_4, top_5~r180.

**Step 3, agent proposals** (`ispatialgym/grasping/ladder.py`, `scripts/grasp_ladder/`; contract `isg-grasp-proposal/1`,
`scripts/grasp_ladder/CONTRACT.md` + `contract.schema.json`, after the artic handle_review2 contract):
- `agent_step.py select` picks every manifest entry built in a round root and valid in none; `prepare --regen` writes per entry
  `inputs/<id>.json` (rest-frame geometry summary, task role, gripper facts, the rounds' gate counts and reasons, markers A-H = the
  rounds' near misses from the gate dump - regenerated from the round's recorded config when it kept none) and `strips/<id>.png`
  (orthographic views, two sections, oblique; the collision export only).
- `propose --runner codex` runs `codex exec --approve-for-me --skip-git-repo-check --ephemeral -i <strip> --output-schema
  contract.schema.json -o proposals/<id>.json` per entry (the Stanford baseline overrides `--yolo`), checks the answer against the
  contract and asks once more with the errors; `--runner manual` lists what is missing. `run_agent_step.sh` chains it unattended
  (cluster CPU job -> lambda codex -> cluster build -> report), JSON / PNG only between the machines.
- `submit` makes each proposal a given-pose pool (`candidates.agent_proposal_poses`: the pinch centre, closing axis, approach, pad
  offsets; expanded by tilts 0 / +-10 deg, turns 0 / +-15 deg, slides 0 / +-3 mm) through every gate of its mode, with the LAST ROUND's
  gates (`ladder.round_gates`: its recorded thresholds and switches, e.g. cone 60), and the usual physics validation. Candidates and
  grasps carry `source: "agent"` and the proposal index; the library gets `source` and the `ladder` record. `report` gives per proposal
  poses -> each gate -> selected -> geometry ok -> reps -> valid, and the killing gate with its measured values.
- Round-2 zeros this round (batch r2zero_20261107; the cap, cork and both hardbacks excluded: other agents' round 2): board game,
  paperback, keyboard, magazine, paper towel dispenser, license plate. I acted as the proposer: the four slabs are infeasible under the
  gates (wider than 89 mm both ways, a thickness pinch needs a finger under the support, every corner chord is 97-164 mm from the CoM
  over the lever caps 87-123 mm). Keyboard: 2 back-corner chords; dispenser: 4 vertical-corner chords. Build (`r2zero_20261107_build2`, the round's cone 60):
  - keyboard p0: 81 poses -> width / cone 0 (antipodal 0.02-0.48 < 0.5: the back corner is bevelled); p1: 81 -> 3 -> 3 -> object clear 0
    (sphere clearance -1.74..-1.37 mm vs +1 mm).
  - dispenser p0-p3: 54 each -> 30 / 30 / 48 / 48 -> support all -> object clear 0 (best sphere clearance -0.12 / +0.71 / -10.8 / -6.5 mm
    vs +1 mm: a flat pad at 30-45 deg to a face digs its edge in) -> **0 physics-valid**.
  - The dispenser's designated upright rest is not a rest: validation settles it at `r0_other`, tilt 1.575 rad (a wedge bottom: a
    wall-mounted dispenser). Its entry needs the lying rest first (round 1/2 business), not more proposals.

**Step 4, legacy grasps (code only, not run):** `legacy_step.py list` (the entries still at 0 after step 3, their legacy grasps and
aliases) and `submit --owner-approved` (refuses without it); the legacy yaml's eef poses as given poses (`legacy_grasp_poses`) through
the same gates (the last round's) and physics. The legacy root is the snapshot's `runlogs/grasp_library_20260905` (the main
checkout's copy lacks some keys, e.g. the hardbacks, the box, the plate).
- Would end at step 4 (owner decides before any use): none of the tape / box / cap / plate (they have grasps); none of the six step-3
  entries (no legacy key: board game, paperback, keyboard, magazine for alignment-axial_01-04; dispenser, license plate for
  alignment-world_vertical_03 / 05 - they end with no grasps at all). Pending other agents' round 2: cork ncxgpe (attachment_03,
  `cork__ncxgpe`: 12 legacy grasps, 5 held in the legacy validation) if `attach_axis="seat"` is not adopted and step 3 finds none;
  hardback bqvvvx (inside_large_01, `hardback__bqvvvx__bo`: 12 / 5) and acbrnv (touching_03, `hardback__acbrnv__bu`: 9 / 9) if their
  overhang / lean entries stay at 0.

**Owner questions:** (1) the pad sweep starts at the fully open pads, not the oracle's pre-shape (the tape's top grasps, any object with
a hole or a nearby wall); (2) the slab entries of alignment-axial_01-04 / world_vertical_05 have no pinch from their flat rest under the lever cap and no
legacy key: push-only programs or an overhang entry (like the hardback) are the options; (3) the dispenser (world_vertical_03) needs its
lying rest; (4) the tray-floor anchor 0.6791 (office_cubicles_right conference_table_fpkmyw_0 source tray) joins the campaign anchors
when the tape entry is merged (0.6789's range shrinks to [0.6733, 0.679]).

**Correction 2026-11-07 (owner decision, after the report above).** The step-3 run on board game / paperback / keyboard / magazine /
paper towel dispenser / license plate was a SELECTION ERROR and is void: alignment targets are pushed and braced, never grasped; their
task-spec roles require `contacts`, not `grasps`, and their v2 path is the contact libraries and maps. The lines above that suggested
"push-only programs or an overhang entry" and a lying rest for the dispenser do not apply - they need no grasps. The run's batch and builds
moved to `$OFH/ladder_agent/_void_not_grasped/` (VOID.txt). Fix: the step-3 / step-4 selectors take an entry only when a task-spec role with
`require.grasps` names its model (`ladder.grasping_roles`: same category, and the model as the role's representative, a `grasp_keys`
"category/model" key or a scene whitelist); `agent_step.py select` reports the others as NOT_GRASPED, `legacy_step.py list` skips them.
Chosen over the v2only missing_grasps / MAP_REQUEST lines because it needs no sampler run and is what the oracle reads. Test:
`test_grasp_ladder.py::test_only_grasped_entries_climb_the_ladder`. With it, no round-2 zero entry in my scope reaches step 3.

### Floating grasps from GraspGen (i7-floating, 2026-11-07)

**How the legacy floating grasps were made** (`runlogs/grasp_library_20260905`, 285 held-object models; read only):
- On lambda, in the `graspgen` conda env: GraspGen 2dd8852, the Franka Panda checkpoint `GraspGenModels/checkpoints/graspgen_franka_panda.yml`.
- Inputs: `export_meshes_batch.py` wrote each model's COLLISION meshes (OmniGibson collision prims, root frame) to OBJ. `graspgen_batch.py`
  sampled 2000 surface points, centred them and ran the diffusion planner (2000 grasps, threshold -1, top 500, no outlier removal).
- Re-targeting (`gen_candidates.run_graspgen`): `T_CONV = Rz(90 deg) @ Tz(PANDA_DEPTH 0.10527 - CONTACT_Z -0.01186)`. The pad centre sits
  on the Panda TCP, and the closing axis goes from Panda x to R1Pro y.
- Filters: hand spheres 1 mm clear at full open, width 3 mm..aperture. Ranking per mode: C / V = confidence gate, then CoM with a soft 40 %
  lever cap; H = confidence-first; knives = `handle_half` (t <= 0.5). Then NMS to the top 12.
- Validation (`ag_validate_render.py`): the object teleported into the open gripper (gravity off), the gripper closed, AG attached and held
  for 30 steps under gravity; stop at 5 valid. Reviewer agents then judged the rendered strips.
- The cluster has the same env and checkpoint (`/vision/u/minyeong/projects/graspgen`, conda env `graspgen`). Its `pointnet2_ops` is built
  for sm_80/86/89 only, but the checkpoint's ptv3 path does not use it. The TITAN RTX smoke test (18367953) passed, at 0.66 s per object
  warm. No new env or install was needed.

**The v2 port** (`grasping/floating.py`, `validate_floating.py`; i7-floating 8c29166e / 89e8edae):
- **Identity.** Constraint `floating`. The entry id is `<cat>__<model>__<scale>__floating`, so it never aliases the tabletop entry
  `<cat>__<model>__<scale>`. A floating entry has no rest: `rests` is empty and every grasp has `rest_id` None.
  - `schema.check_compatible` enforces this, and `ModelLibrary.identity` adds the constraint (for floating entries only).
  - `grasp_library.rest_quat_xyzw` raises for a floating entry, so it never serves a placement.
  - `load_grasps` serves all of its grasps.
- **Proposals.** GraspGen on the v2 export's collision geometry (`geometry.json`), with the legacy settings, seeded per entry
  (`scripts/grasp_bands/graspgen_floating.py`, graspgen env). No visual asset is decoded or written.
- **Gates** (the v2 ones that apply without a support; none relaxed):
  - width / antipodal cone (30 deg);
  - hand spheres clear of the object at the HELD sampler's opening (fully open: `establish_grasp` teleports the object into the open hand);
  - closing sweep;
  - lever cap (40 %);
  - [knives: handle half];
  - the approach from the pre-grasp, and the finger blades through the closure.

  The support gates (finger allowance, palm / camera / wrist vs the support, retreat / lift) do not apply. Then the legacy mode ranking and
  NMS (12), with no symmetry folding (the held sampler draws rows, never family members). `ISG_GLIB_GATE_DUMP=1` writes `gate_dump.json`.
- **Physics** (`validate_floating`, in the tabletop validation world with its slab parked; same physics profile):
  1. The task's fixed hold, solved by IK: mental_rotation roll 15, object_scaling roll 90.
  2. The held sampler's own closure: 1 mm/step advance, 4 mm press, the 90-step AG window, then 30 steps of gravity.
  3. The task's held motions, tracked by the oracle's `Motions.arm_along`, each out and back to the hold:
     - mental_rotation: an orbit about the object root per signed base axis, at the largest of 90 / 60 / 45 deg the arm tracks;
     - object_scaling: lines along the optical axis, -0.25 m and +0.05 m.
  4. Acceptance: AG formed and held; on target (15 mm / 15 deg); after gravity and after every leg the object still attached, with its in-hand
     drift <= 1 cm / 0.1 rad; >= 2 rotation axes / 1 line executed.

  The first build (8c29166e) chained +90 / -180 / +90 about x and y. A failed diff-IK leg does not move the arm, so the next leg started
  180 deg away and 5 of 6 legs were unreachable. 89e8edae replaced the chain with out-and-back motions.
- **Sampler guard** (`held_object.check_v2_floating`, the v2 path only): a v2 entry serving a floating key must be a floating entry, else
  `missing_grasps:floating`. A spawn scale other than the entry's is logged `V2_SCALE_MISMATCH` and rejects `v2_scale_mismatch` under v2only.
- **Pipeline.** Floating jobs dispatch in `candidates` / `validate`, with no settle survey and no rest retry. `manifest --floating-keys`
  builds them (`legacy_key` = the floating key; scale / dataset / mode / task come from the legacy result's metadata, never its grasps).
  `scripts/grasp_bands/job_floating.sh` runs export, graspgen, candidates and validate.

**Entries** (`$OFH/floating_20261107`: manifest `floating20.json`, one job per legacy floating key that the 20 aliases use; build 18368108,
validation 18368389). Columns: GraspGen proposals, then the count after each gate (w/a = width / antipodal, clear = sphere clearance,
sweep, lever, geo = approach + blades), the NMS representatives, and the physics-valid grasps (stop at 6):

| entry (task) | mode | GraspGen | w/a | clear | sweep | lever | geo | reps | valid | invalid trials |
|---|---|---|---|---|---|---|---|---|---|---|
| apple omzprq (OS) | C | 500 | 496 | 496 | 496 | 496 | 496 | 10 | 6 | - |
| beefsteak tomato ogpans (OS) | C | 500 | 478 | 478 | 435 | 435 | 435 | 10 | 6 | - |
| beeswax candle nhdnje (OS) | C | 500 | 499 | 499 | 498 | 489 | 489 | 10 | 6 | - |
| bell pepper ukkycp (OS) | C | 500 | 394 | 394 | 330 | 330 | 195 | 10 | 6 | - |
| bottle of catsup hvxkso (OS) | C | 500 | 485 | 484 | 349 | 314 | 314 | 10 | 6 | - |
| bottle of medicine kqkwoq (OS) | C | 500 | 323 | 323 | 36 | 36 | 28 | 8 | 6 | - |
| bottle of pop pvabxf (OS) | C | 500 | 438 | 438 | 437 | 437 | 422 | 10 | 6 | - |
| can of soda opivig (OS) | C | 500 | 489 | 489 | 489 | 489 | 489 | 10 | 6 | - |
| canned food ycbspm (OS) | C | 500 | 482 | 476 | 469 | 469 | 352 | 10 | 6 | - |
| pack of cigarettes kienyo (OS) | C | 500 | 500 | 497 | 496 | 489 | 489 | 10 | 6 | - |
| carving knife alekva (MR) | H+handle | 500 | 339 | 330 | 330 | 250 (handle 153) | 153 | 10 | 6 | 1 motion unreachable |
| classroom mock-up isgsma (MR) | C | 500 | 466 | 466 | 381 | 363 | 232 | 10 | 6 | 2 motion unreachable |
| crab ohjsdr (MR) | C | 500 | 385 | 382 | 380 | 376 | 365 | 10 | 6 | 2 motion unreachable |
| hook hewhtl (MR) | H | 500 | 263 | 262 | 245 | 245 | 81 | 10 | 6 | 1 ag_not_formed (4-6 mm wire) |
| mug ppzttc (MR) | V | 500 | 203 | 182 | 11 | 11 | 2 | 1 | 1 | - |
| soup ladle hgvzdm (MR) | H | 500 | 271 | 251 | 200 | 148 | 134 | 10 | 6 | 1 motion unreachable |
| statue tuldri (MR) | C | 500 | 62 | 62 | 4 | 4 | 2 | 2 | 2 | - |
| teaspoon eilfij s1x1x2 (MR) | C | 500 | 135 | 134 | 134 | 13 | 12 | 3 | 1 | 2 ag_not_formed (4.9 mm) |
| toy figure zuntus (MR) | C | 500 | 154 | 148 | 77 | 77 | 53 | 10 | 6 | 2 motion unreachable |
| walker glrsok (MR) | C | 500 | 267 | 267 | 86 | 86 | 33 | 10 | 6 | 2 motion unreachable |

- **Held motions.** No grasp dropped or slipped in any executed leg (largest in-hand drift 0.22 mm).
- **Executed motions** (in the valid trials):
  - object_scaling: the 0.25 m line toward the camera and back. The +0.05 m leg away from the camera is out of reach at the hold for every
    entry.
  - mental_rotation: rot -x 45 deg, rot +y 90 deg, rot -z 60 / 90 deg. The +-x rotations at 90 / 60 deg and most of -y / +-z are out of reach
    from the 15 deg hold.
- **The thin pools.** For the mug, the sweep gate drops 171 of 182: the pads meet the handle / rim first at full opening. For the statue, the
  antipodal cone drops 438 of 500 on its relief. For the teaspoon, the lever cap drops 121 of 134: GraspGen pinches the handle end.
  All of these are gates the legacy floating path did not have (it checked width and spheres only).

**Links and validation.**
- **The staged root.** `$OFH/floating_20261107/v2maps_i7float` = `cp -a v2maps_i7` plus the 20 entries (`stage_floating.py`):
  - each STAGED `library.json` gets `legacy_keys` = [its floating key] - that, not `links.json`, is what `_v2_index` reads;
  - `links.json` gets the 20 links, and the 20 keys leave `skipped`.

  All 37 links resolve under v2only to their entry: 20 floating, and the 17 tabletop links unchanged.
- **Maps.** The held-object samplers log no `MAP_REQUEST` under v2only: the hold, starts and goals read no reach map. No map is built.
- **N 1 on the 20 aliases** (tag `i7_float1`, trunk_89e8edae; `v2only_before` on legacy grasps: 20 / 20 ok):
  - 20 / 20 written, 19 / 20 demos ok, 28 attempts, 2.79 GPU-h = 0.147 / success;
  - every instance on a v2 floating row (`gg_*`);
  - 0 `LEGACY_GRASPS_BLOCKED` / `legacy_band_blocked`, 0 `missing_grasps`, 0 `V2_SCALE_MISMATCH` (the teaspoon at s1x1x2).

  The failed demo is `mental_rotation-statue`. It reached the goal (orientation error 0.013 rad, tolerance 0.093), but QA hard
  `jerky_motion` + `arm_stutter` stopped it: the right arm jolted 4 times in steps 15-19 of the reorient, up to 1.08 rad/s. That is the oracle's
  tracking of the rotation, not the grasp, and the oracle was not changed.
- **N 4** (tag `i7_float4`), both 0 `MAP_REQUEST`, 0 blocked, one reject (`goal_occluded`); together 9 attempts, 0.30 GPU-h = 0.038 / success:
  - `mental_rotation-crab`: 4 / 4 demos ok, rows gg_0 / gg_2 / gg_5 / gg_7;
  - `object_scaling-bottle_of_catsup`: 4 / 4 ok, rows gg_1..gg_4. The held sampler's spread order gives each instance its own row.
- **Open for the owner.**
  - The floating entries live only in the staged root; adopting them into the campaign root is the owner's call.
  - Three entries have few rows: mug 1, statue 2, teaspoon 1. None is at 0, so the ladder did not trigger. More GraspGen seeds would be a
    round-2 retry.
  - The statue's QA stutter is in the oracle's reorient tracking.
  - The approach gate is applied although a held task has no approach, as one of the hand / object gates. It never fired on these 20
    entries: the 684 "geometry" drops are all `finger_blade_in_object`.
  - The held-motion thresholds are the tabletop validator's provisional ones.

#### Owner decisions 2026-11-07, item 4 (i7-floating b671313f)

- **The held-motion check matches the task** (`validate_floating` v2). Per (candidate, task), 3 motions are drawn with the sampler's own
  functions and screening IK:
  - mental_rotation: `draw_rotation_starts` / `draw_rotation_goals`, keeping goals whose orbit key poses the batch IK reaches from the start;
  - object_scaling: the depth window, then `draw_start_goal`.

  Each motion closes the gripper at its START configuration with the held sampler's closure and is tracked by `arm_along`. The motions
  tested are recorded per trial.
  - Drift is judged by the task's goal tolerance: mental_rotation 0.15 rad, object_scaling max(0.02 m, 0.2 |dd|).
  - Calibration (`$OFH/floating_20261107/drift_demos.json`): 63 successful collected held-object demos, the largest in-hand drift 0.24 mm /
    4.4 mrad. All of them pass.
  - The achieved-vs-candidate offset is recorded, not judged.
  - Re-validation of the 20 (`val2`):
    - every entry keeps its valid count;
    - 23 candidate verdicts changed (mostly `held_motion_unreachable` -> valid), so 9 mental_rotation entries have other rows;
    - no drop or drift failure: object_scaling 180 / 180 motions ok; mental_rotation 122 ok, 10 unreachable, 12 `ag_not_formed`.
- **Re-seed** (`graspgen_floating.py --n-points 4096 --n-seeds 4 --num-grasps 4000 --topk 1000`):

  | entry | before (geometry-ok / reps / valid) | after |
  |---|---|---|
  | mug | 2 / 1 / 1 | 7 / 5 / 5 |
  | statue | 2 / 2 / 2 | 20 / 6 / 6 |
  | teaspoon | 12 / 3 / 1 | 142 / 10 / 3 |

  The teaspoon's other 7 representatives fail the assisted grasp at every drawn start (4-5 mm pinches): a physical limit, not forced.
- **Statue N 4** (`i7_float4s`): 3 / 3 collected demos ok with no stutter; the 4th errored on a missing `tro_state.json` in the collector.
  The stutter did not repeat, so there is no sampler or oracle change.
- **Staged root.** `$OFH/floating_20261107/v2maps_i7float2` = v2maps_i7 + the `val2` entries (the re-seeded mug / statue / teaspoon), with
  `legacy_keys` links; all 37 links resolve. It rides along in the next scheduled run.

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
