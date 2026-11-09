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
when the idle posture untucked / ready / tucked is free of itself and of the slab at every `d_e`; since i7-unblock 2026-11-07 the default
`per_d_e` rule makes a trunk ineligible only at the `d_e` where every posture collides - see "i7-unblock" below): lattice 8 cm x 8 yaws x grasps
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
  - Old vs new were A/B'd first: `ISPATIALGYM_FINGER_TOUCH_RULE=spheres` (old) vs `geometry` (new; `ik.FingerTouch.rule`,
    `IKChecker._finger_touch_geometry_ok`). A support without a registered plane (legacy-band runs, a bookcase, a sink) is measured
    against its convex collision pieces.
  - A/B part B (2026-11-07, i7-fingertouch): both arms ran on the SAME instance indices (one ledger per arm, one cold sampling cache per
    job). `ISPATIALGYM_FINGER_TOUCH_SHADOW=1` judged every touched configuration under the other rule too (kit counters
    `finger_touch.shadow.*`, `finger_touch.verdict_changed`, `finger_touch.decision_changed`). Tags `ft2_sph` / `ft2_geo`, table
    `$OFH/jobs/ft2/ft_tab.py`.
  - Where the rule acts, the two rules never disagreed: 0 changed verdicts out of 4651 judged configurations (1984 collision groups,
    710 allowed touches). Every attempt had the same outcome in both arms. The finger hulls never touched the support: the deepest
    allowed hull penetration was 0.00 mm, against the 2 mm cap.

    | alias (grasps) | configs judged | changed | written / attempts | demos ok | GPU-h per success |
    |---|---|---|---|---|---|
    | inside_large_03 (legacy, 300014-29) | 492 | 0 | 16 / 25 both | 11 / 16 both | 0.059 sph / 0.060 geo |
    | plate-in-sink lkklqs_0-breakfast_table (legacy, 300014-21) | 231 | 0 | 8 / 11 both | 7 / 8 both | 0.049 / 0.044 |
    | under_small_03 (v2only side_9, 300500-03) | 3928 | 0 | 0 / 60 both | - | 0.33 GPU-h each, 0 written |

  - Where the rule did not act: under_small_03 under legacy grasps never reached it (0 judged configurations). Its 16 instances went
    15 written / 95 attempts in both arms, with demos 14 vs 15 on identical samples. The plate-drop_in_sink-bar and -countertop variants
    got no template at eval:0 in 3600 s. touching_03 is still blocked.
  - The first A/B's 3 extra `approach_untrackable` (ft_geo, inside_large_03 300016 / 300017) were not finger-touch refusals. They are
    program-check joint jumps on the retreat or carry after the place: 0.24 rad at waypoint 2, 0.23 rad at waypoint 15, 0.21 rad at
    waypoint 2. Both arms reproduce them on the same instances. The first A/B differed only because its arms drew different indices
    (300010-13 vs 300014-17).
  - The implementation is not stricter than the spec. Its non-finger stage is the sphere rule's own check (margin 0), so the
    geometry arm's `rejected_non_finger_link` (130 / 32) and `rejected_body` (4) are configurations the sphere rule refuses too.
    It is more lenient in one place: it does not run the spec's exact-hull check of palm / camera / wrist (P1 §3); only their spheres
    are checked.
  - Default: `geometry` since the A/B; `spheres` stays available.
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

### Owner decisions 2026-11-07: contact follow-ups, inside_large_01 place stance, per-alias outcome memory

Recorded in DECISIONS 2026-11-08 01:45 UTC. No shared gate or QA rule changes.

**A. Contact follow-ups** (branch i7-contactfu, from i7-contact 521dc344).
- **A1, touching_02.** The push stance also moves forward and backward.
  - The stand-off is chosen from the touch_push map: the target is 0.66-0.93 m, the map's ~0.75 m.
  - It is limited by the base's minimum table distance. Sampler only.
  - A partner whose push side fails the map everywhere is redrawn or rejected. Validate at N 4.
- **A2, touching_02 tipping.** The legacy demos do not tip.
  - Find what differs between the demo scene and the validator world: bar / box physics materials and box mass, read from a scene load.
  - Make the validator match. Do not change the scene. Report numbers.
- **A3.** The hand-overlap screen stays ENFORCED, and its switches stay off. Valid push entries: napkins 26, cereal 12, spray 28.
- **A4, axial_02 repeats.** With maps, 57 of 60 attempts repeat an eval instance (repeats_eval); before maps it was 7-11 of ~20-26.
  - First compare the push map's feasible cells with the eval instances' start poses.
  - Propose a fix with numbers. Ask before changing the repeat rule.
- **A5, axial brace.** Per stance and allowed goal orientation, the push map and the brace map are taken on the SAME pushed end.
  - The first fitting pair is used, else push-only. Map lookups only.
  - Kit counters axial_brace.paired / .push_only. Validate axial_01 / 03 / 04 at N 4.
- **A6, contact-entry validation is STRATIFIED.**
  - 2-3 physics-valid entries per (yaw x end x spin) group a map uses. Maps are built from validated entries only.
  - Maps then learn from real runs: a success overlay per map identity, written in per-job shards and merged offline into a new versioned
    map.
  - An entry joins only after a real oracle success or a physics validation.
  - Report the cost against today and the counts per library.

**B. inside_large_01 place stance** (branch i7-placeorder, from i7-overhang c53a3a3b).
- The 0.70 m cap is replaced by the trunk-extension ORDERING. The stand-off range is restored.
- Stance C order: least trunk translate first, then nearest stand-off. Kit stance_c.extended_used. No stance is removed; exact IK and QA
  are unchanged.
- First, a campaign-wide table of place-time failures (wobble / self-contact / plan_failed) against the place stance's trunk translate /
  tilt and stand-off, over all collected demos.
  - If trunk extension drives the failures, the ordering becomes the default for all aliases.
  - Otherwise, report and ask.
- Validate inside_large_01 at N 4 with C on.

**C. Per-alias OUTCOME MEMORY, unattended** (branch i7-memory, from of-v2maps b03f1dca). C works on top of B's order.
- **Record.** Per demo, a configuration key (alias, arm, stand-off bin, trunk translate / tilt, grasp family, goal yaw slot) plus the failure
  type and stage, in per-job shards.
  - A failure is attributed to the choice that precedes its stage.
  - Infrastructure failures and known bugs are excluded.
- **Aggregate.** Offline per batch, into a versioned memory per alias. Keys are hierarchical and back off when sparse. Jobs pin the version
  in job.env.
- **Sample.** Per (alias, bin) from a Beta posterior (Thompson-style), with an exploration floor of ~5 %. Kit alias_memory.*.
- **Forget** when the snapshot, entry, map keys or alias customs change.
- **Guardrails.**
  - A per-batch report and alerts; shared-fix candidates are flagged.
  - It never changes shared rules.
  - Provenance per instance.
  - Never used for eval-instance generation.
- **Thresholds** are proposed from the logs by a replay over $OFH/out. It must catch inside_large_01's 0.85 m / 0.50 failures and flag
  nothing healthy.

## Integration (2026-11-08, of-v2maps)

The owner's direction of 2026-11-07: merge the finished branches one at a time, owner-decided defaults land with their branch, nothing new is
decided at a merge. Record: `$OFH/int1_20261108`, `$OFH/DECISIONS.md` ("integration" lines).

**Merges** (one merge commit each, in the given order; snapshots `$OFH/code/trunk_<sha>`):

| # | branch (tip) | merge |
|---|---|---|
| 1 | i7-switches (6ad20b33) | f789995a |
| 2 | i7-padsweep (ff22a5d9) | 061c5565 |
| 3 | i7-attach (bbac7175) | 50bd14c4 |
| 4 | i7-ladder (d0ad2afe) | 4924318a |
| 5 | i7-floating (75b101db) | dafed151 |
| 6 | i7-overhang (c53a3a3b; carries i7-lean da868175 .. a4a0fa1e) | 62a3f082 |
| 7 | i7-fingertouch (f1af78aa; not i7-fingertouch-ab-us03) | 405d0c2f |
| 8 | i7-unblock (145afcd8) | fcb6a54e |
| 9 | i7-attach-swept (d711771f = i7-attachmap + i7-attachhover, swept default) | b03f1dca |
| 10 | i7-plateqtf (0866bfc0; expanding tilt / yaw map default on - the owner's confirmation pending) | 386ab2b4 |
| 11 | i7-memory (c49eacf6; outcome memory default on for train splits; $OFH/alias_memory not seeded, no thresholds.json) | a8baafe3 |
| 12 | i7-placeorder (3d8492fc; trunk-first order before the memory hook) | f313237f |
| 13 | i7-contactbrace (3d5cd9bb = i7-contact + i7-braceopen; one switch ISPATIALGYM_BRACE_FULL_OPEN, default on) | 0f8e95c0 |
| 14 | i7-attachfix (9681dac8; collector attach restore, witness FK check, opt-in outer_pinch / attach_face_retreat) | 6e5c6cba |
| 15 | i7-plateqtf2 (93072af3; hull fallback on, IORSSmall overhang fallback, pre-place framing, pinned map overlay on) | 543d6d1a |
| 16 | i7-artic (603ad557; prefix maps from own-arm segments, v2only articulation_small) | 61d52902 |
| 17 | i7-contactfu (ec81e006; owner part A; map_tilt.py kept once, the single brace switch kept) | 76a4a4e4 |
| 18 | i7-interior (d67158a7; interior PLACE map rule for stable rests, in the identity) | d9dd4565 |

**Kept once.**
- The switch-state identity: 884279fd (i7-switches) and its cherry-pick 33e72d92 (i7-attach) are identical patches. The merged
  `ModelLibrary.identity()` holds every addition, each only when present: the plate switch pair, `sweep_from_preshape` when on, `attach_axis` off
  "meta_z", the floating constraint, lean / overhang rest records, `validate_arm`, `access_region`. Every existing key is unchanged.
- Per-stand-off trunk eligibility: i7-switches carries i7-unblock's 1527d9b8 / 58cd9939 / 722d687f as identical cherry-picks, so i7-unblock merges
  without duplication. i7-lean's fixed-compartment posture rule (b4230177) now runs through `trunk_posture_groups(obstacle_free_of=)`.
- The overhang side retreat merged with i7-plate's families and the exact retreat; its blade lookup stays at CONTACT_Z (the overhang entry was
  counted with the merged defaults and does not move).

- i7-unblock's own trunk-eligibility commits add nothing at merge 8: the cherry-picks are identical, and only i7-overhang's additions
  differed, so HEAD was kept. i7-plateqtf's bases (i7-unblock, i7-padsweep) were each taken once.
- Both attach defaults are kept: attach_axis "seat" (the candidate rule) and the swept hover (the oracle).

**Checks per merge.** pyflakes (no new messages); the full suite with the trunk PYTHONPATH: 6 failed = the baseline set, 0 errors;
test_glib_switches (golden key), test_attach_seats, test_glib_padsweep, test_grasp_ladder, test_floating_grasps, the lean / overhang /
interior tests pass.

**The consolidated root `$OFH/v2maps_int1`** (`jobs/v2build/build_int1.py`, log `INT1.txt`): v2maps_i7 plus one layer per merged branch.
- switches: 22 entries (v2maps_i7sw).
- padsweep: 4 rebuilt entries plus the tape (v2maps_i7ps), with anchor 0.6791.
- gym shoe: the L2 entry, rebuilt.
- attach: 10 entries (5 from stage_seat, 5 rebuilt).
- ladder: the storage box, with its floor maps rebuilt (the ladder4 maps were keyed before the switch identity existed).
- floating: 20 entries (float2).
- overhang: the hardback overhang entry, lean_right and lean_left valleft (v2maps_i7oh3), with anchor 0.7671 and a record copy of the
  bookcase_jaysra_0@0.9472 interior spec.

Replaced entries are in `glib|maps/_superseded_<reason>`. Each staged entry was built with every merged default, or does not move under them.
The integration count (`int1_20261108/count`, `count2`) moved 5 attach entries and the gym shoe L2. Those were rebuilt: cabinet door 4 -> 5,
webcam 5 -> 6, skateboard jljqtt 6 -> 6, acmbvb 2 -> 1, dip candle poexjo 0 -> 0, gym shoe 6 -> 6.

Later layers:
- cap xxqyrt from stage_cap. It does not move under the merged defaults (count3). Its 0.8126 tabletop maps were rebuilt, because the staged
  ones were keyed before the switch identity.
- The 8 attach maps (box + swept: lids tnwisp / pbmsji, cork, cap), rebuilt against the int1 entries. Every attachmap_20261107 map carried
  a pre-identity male key.
- plate qtfzeq = the merged entry of v2maps_i7pq (27 valid; replaced entry in `_superseded_merged`).
- 30 per_d_e merged maps of v2maps_i7unblock, only for entries int1 still holds unchanged (`_superseded_trunkelig`).
- i7-interior:
  - plate luhkiz (bar 0.8874 tabletop maps and sink-basin maps) from v2maps_i7sink2, with anchor 0.8874;
  - the jar's fridge place maps from v2maps_i7fridge4;
  - the fridge spec (door per piece) and the sink spec those maps were built with are now in the code registry
    (`assets/interiors/interiors.json`; lead decision, fcddc495), so no job needs ISPATIALGYM_INTERIORS. test_interior_registry.py pins their
    digests. The earlier overlay moved to `_superseded_registry/`.
- Contact maps (16), from three roots:
  - v2maps_icb_on: the touch_brace maps built with full-open on (ntbrtz, dkhgxn, gkpmii), plus the touch_push maps;
  - v2maps_i7contact2: the axial h maps (qgvsnb, injrgc, tfovwy), with the pushed-end / spin fix;
  - v2maps_i7contact3: the paperback s1x1x2.5 maps.

  The gkpmii v2 export is added to glib. The check also reads the contact maps, including the touch_brace knob at the run's default.

- Articulation prefix maps (12) in `maps/_articulation`: right-arm maps from artic_prefix_20261107b, left-arm maps from
  artic_prefix_20261107c.
- i7-contactfu's validated-entry contact maps: touch_push x3 and the axial maps of qgvsnb / injrgc / tfovwy. The touch_brace maps stay the
  ON builds.

**per_d_e count and merge** (`int1_20261108/trunkelig`):
- The count found 58 maps that gain (trunk, d_e) columns: 52 tabletop, 4 interior and 2 overhang.
- All 58 were rebuilt with per_d_e and merged by i7-unblock's rule: the former arrays are kept and the new columns added. Cells grew by
  +24109 over 116707.
- 60 maps are staged in all, including the cork's 2 and plate luhkiz's 0.8874 pair.
- plate luhkiz's sink non-place maps were rebuilt but gain no cell, so they are not staged.
- i7-unblock deferred 8 maps:
  - the bqvvvx lean_right / overhang maps are merged;
  - the old right-validated bqvvvx lean_left maps are superseded by the valleft maps, which are merged;
  - the acbrnv lean_left_20 maps (touching_03) are not in v2maps_int1.

`jobs/v2build/check_int1.py` checks the root as the sampler reads it. On trunk_d9dd4565: 76 entries, 138 maps, 8 attach, 16 contact, 12 prefix and 12 interior maps (specs match), 0 key mismatches, 51 links,
21 anchors (+0.6791, +0.7671) by the midpoint rule, 0 errors.

**Validation** (`--grasps v2only` N 4, affected aliases only; `tab_i.py -a int1_1 .. int1_5`; DECISIONS "integration RESULTS").
- **int1_1** (merges 1-7, 33 aliases): 658 attempts, 126 written, 105 ok; 0.164 GPU-h per success.
- **Two regressions with a cause.** disembedding_01 and _10 went 0 / 4. Their earlier runs with the v2maps_i7 entries were 4 / 4 + 4 / 4 each.
  The sampler now draws new grasps of the rebuilt entries (jigsaw: switches + pad sweep; tablefork: switches). Those grasps are
  physics-valid in the library but fail in the tray execution. This goes to the owner.
- **Later batches.**
  - Attachment (after i7-attachfix): 01 4 / 4, 02 2 / 4, 04 4 / 4.
  - inside_large_01: 4 / 4 (trunk-first order).
  - axial_01 / 03 / 04: 3 / 4, 4 / 4, 4 / 4.
  - touching_01 3 / 4; touching_02 2 / 2 (its place-depth limit).
  - inside_01 3 / 4.
- **Pre-existing gaps.**
  - axial_02: repeats_eval.
  - axial_05: no push_blade map of tablespoon tqqwys.
  - ontop_large_01: no goal map at h 0.9158.


### i7-unblock (2026-11-07): nextto_04, under_small_03, camera_relocalization teddy_bear

Branch `i7-unblock` (from of-v2maps a4620a1e). Records: `$OFH/unblock_20261107/` (count, plan, builds, compare, staging), DECISIONS.md
"i7-unblock" lines.

**Trunk eligibility per d_e (owner decision: the former rule was a bug; default of new builds).**
- `reach_maps.trunk_posture_groups` (pure). Both rules first take the first idle posture (untucked / ready / tucked) clear of itself and
  of the slab at every `d_e` - the former choice, so a trunk eligible before keeps its posture and its cells. Without one, `per_d_e`
  gives each `d_e` the first posture clear there; a `d_e` that no posture clears stays `trunk_ineligible`. The record gets
  `posture_by_d_e`, `ineligible_d_e`, `eligibility: per_d_e`. `compute_map` / `compute_generic_map` run one IK pass per posture group.
- Identity: `MapConfig.trunk_eligibility` (default `per_d_e`, in config / key). `None` / `--trunk-eligibility all_d_e` = the former rule,
  left out of config and key, so old maps keep their keys and read as `all_d_e` (`CONFIG_LEGACY_DEFAULTS`). Floor maps keep `None`:
  with one edge distance the two rules give the same map. `--skip-existing` matches the rule. `job_glib.sh GB_TRUNK_ELIGIBILITY`.
- Sampler: both rules serve; there is no config_require change. `ReachMap.find` / `find_generic` prefer a `per_d_e` map when a root holds
  both for one lookup; the first former-rule map is the fallback. An explicit `ISPATIALGYM_REACH_MAP_REQUIRE={"trunk_eligibility": ...}`
  selects either one.
- i7-lean b4230177 (interior obstacle ranking) is compatible: `trunk_posture_groups(..., obstacle_free_of=)` ranks the postures that are
  free of the board at every `d_e` by their obstacle clearance, exactly as b4230177 does. The `per_d_e` fallback then takes, per `d_e`,
  the first board-free posture that is also obstacle-free. The i7-lean block should call the helper with `obstacle_free_of` when the
  compartment is fixed.
- Count (campaign roots, each distinct map once, from the stored trunk records): 116 maps, 108 served. 88 served maps gain columns:
  374 trunks, 614 (trunk, d_e) columns. Of these, 74 tabletop + 6 generic were rebuilt (trunk_58cd9939), and every one changes cells.
  The 6 interior + 2 overhang maps wait for the merge: their builders are on i7-lean / i7-oh. `/v2maps_i7unblock` = v2maps_i7 + 78
  merged maps: 338 trunk-maps, 544 columns, feasible cells (column-summed) 166628 -> 195528. The 2 plate luhkiz maps are in
  `unblock_20261107/build_plate/merged`. The merged staging keeps the former map's arrays wherever the former rule held the trunk
  eligible, and takes the rebuild's arrays only at the added columns. A rebuild flips 0.2-0.3 % of the entries in the old columns
  (feasible <-> branch_jump / ik_pre), because the IK seeds are random. No posture choice of an eligible trunk changed.

**nextto_04 (gym_shoe kmcbym on bench_pfjnnj_0, h 0.3471).**
- The bench is not slatted at its raster. The inventory top is one rectangle of 1102 free cells: 19 x 58 at 2 cm, 0.36 x 1.14 m, hull
  area = cell area, no gaps.
- What blocks Phase 2 is the map lattice against the eroded top. The erosion is the shoe's circumscribed half extent 0.154 m + 0.02, so
  on a 0.36 m deep bench it leaves a single 2 cm row of 40 raster cells on the centre line.
- The 8 cm lattice never lands on that row. MAP_LATTICE (g_sf): 0 lattice points kept at 538 + 39 stances. The v2only logs show no
  "per trunk candidate" line, i.e. no focus survived the containment test.
- The per-d_e eligibility adds t0.95 at d_e 0.45 / 0.55 (5 trunks, 10 columns, cells 1158 -> 1483 right), but the same lattice does not
  meet the row either.
- The opt-in support-first lookup (`ISPATIALGYM_MAP_SUPPORT_FIRST=1`, owner prompt G) reads the map at the raster's own cells. Its
  earlier nextto_04 "0 / 208" (g_sf) was a stale SPA cache: a2f5fb06 had no map key / support-first switch in the start-cell key
  (bc616c4e added them), so the per-attempt cells were hits of earlier rejections. The setup's own Phase 2 there had 266 survivors and
  7 start cells.
- v2only N 4 on `/v2maps_i7unblock`: the per_d_e maps alone give 0 attempts (setup no_manipulation_stance). With
  `ISPATIALGYM_MAP_SUPPORT_FIRST=1`: 13 attempts, 4 written, 4 / 4 demos, 0.058 GPU-h per success. Whether support-first becomes the
  default is the owner's decision (it is a shared Phase-2 rule).

**Support-first as an automatic per-support fallback (owner decision 2026-11-07 21:40; 954bc5bf).**
- `tabletop.phase2_manipulation_stance` wraps the former Phase 2 for map views.
- Forced on / off comes first: the job switch `ISPATIALGYM_MAP_SUPPORT_FIRST=1/0`, then the alias's `generation.map_support_first`
  true / false. Otherwise the rule is automatic:
  - geometric, decided before any stance: the usable region (the top raster eroded by the footprint half extent + 0.02) is narrower
    than the map's own `lattice_m`;
  - safety net: a map-first call whose stances held zero lattice cells on a usable support is run once more support-first.
- Counters `map_support_first.geometric` / `.fallback`; one `MAP_SUPPORT_FIRST` line per (support, trigger) with the usable width and
  the lattice.
- Everything else stays map-first, unchanged. Scene bands are never read support-first.
- iors_small's start-cell key carries the state (`support_first_tag`).
- nextto_04's customs set it on explicitly (baf472f8).
- Scope from the v2only logs (v2only_after, i7_floorfix, i7_tg, v2only_before, g_base; `unblock_20261107/sf_scope.json`): of the 32
  aliases with map hits, only nextto_04 fires (bench_pfjnnj_0; usable 0.02 m in the sampler vs lattice 0.08). It fired on both
  triggers: 20 / 20 zero-cell Phase-2 calls, MAP_LATTICE lattice_kept 0 in 8 / 8.
- ontop_large_03's bed has no inventory record. No other alias fires, so no other alias changes.

**under_small_03 (plate qtfzeq s1x1x3, the one v2 grasp side_9).**
- side_9 approaches nearly horizontally (17.6 deg down), 32 deg off radial, with the closing axis 20 deg from vertical: one finger goes
  under the rim. The legacy grasp_0 approaches from above (60 deg) and straddles the rim radially.
- The approach is tracked with the fingers fully open (0.05 / finger, the stance's configuration). The oracle and the map use the
  pre-shape (width / 2 + 10 mm = 0.018). Fully open, the lower pad passes 0.5-1.0 cm above the table in the body of the approach;
  pre-shaped, 3.4-3.9 cm. That gives the 27 / 60 "F exhausted" cases (21 approach_untrackable against breakfast_table, 3.6-5.5 cm from
  the start).
- The fix is in the alias's customs: `approach_finger_support_touch` {0, 0.005, preshape}, the plate-in-sink variants' rule.
- The 33 / 60 empty F all sit at trunk 0.25 / -0.53. The map's cells come from grid tilt -0.6, and the empty F is kinematic (round 4:
  0 pairs with every scene object out).
- The opt-in TILT_SNAP returned `no_band` for every map view. It now reads a map view at the stance's stand-off. It stays off by
  default; it was A/B'd for this alias only.
- v2only N 4 with side_9, customs only: 60 attempts / 0 written. Breakdown: 37 empty F (DIAG 4 / 4: 0 pairs with no scene object) and
  16 F exhausted, mostly the support policy's "non-finger link touches the support"; 478 finger touches were allowed. With TILT_SNAP:
  empty F 0 / 60 (40 snaps), but 0 written - 39 F exhausted and the place stage fails.
- Grasp ladder step 2 (`unblock_20261107/plate_qtfzeq`): the opt-in families lip_shallow + rim_straddle (cand_overrides, default gates)
  give 36 candidates -> 27 geometry-ok -> 10 representatives -> 6 physics-valid rim_straddle top grasps. Maps at 0.8126 for both arms,
  staged in `/v2maps_i7unblock2` (the entry keeps legacy_key plate__qtfzeq__st).
- v2only N 4 on the rim_straddle grasps: 60 / 0. The grasp chain now passes, so what still blocks is downstream: 35 empty F (kinematic,
  DIAG 4 / 4), 11-12 partner_placement ("no partner yaw of 18 admits a gap"), ~13 gripper_not_in_frame at the place / pre-place (the
  camera-frame rule, not relaxed).

**camera_relocalization teddy_bear_1: slow, not blocked.**
- BEHAVIOR's `astar` builds a score table of every map cell on each call: 2.6 s on house_single_floor's 862 x 862 map, 8.6 s for an
  unreachable goal. A goal search makes about 20 calls, and an obstacle cell 1-2.
- `navigation.astar_exact` is the same search, step for step, without the table: 12 / 12 identical paths on the real map, 0.36 s mean.
  It is behind `CR_FAST_ASTAR` (default on).
- `CR_OBSTACLE_HEADROOM` (default on) skips a goal whose free geodesic leaves less than the bucket's minimum detour below the bucket top.
  No obstacle cell can pass `detour_failure` there, which is exact under float subtraction. Example: 8 / 8 too_long at 4.86 m in (3, 5].
- Sampling budget: `SAMPLE_LIMIT_S` = 2 x fixed + N x budget_s (train85 `params`; 2400 s at N 1, 8766 s at N 4). The knob is
  `submit_fast.py --sample-limit-s`.
- The camera-frame rule is unchanged.
- Off / on test (owner 23:10; sampling only, same start indices, headroom off in both; `unblock_20261107/cr_ab.py`):

  | variant | instances | s / attempt, off -> on | sample wall, off -> on |
  |---|---|---|---|
  | vacuum | 4 / 4 equal | 208 -> 28 | 3761 -> 888 s |
  | toy_car | 3 / 4 equal | 130 -> 50 | 997 -> 777 s |
  | trash_can | 1 / 4 equal | 202 -> 45 | 2938 -> 949 s |
  | teddy | 3 / 4 equal | 202 -> 47 | 1683 -> 749 s |

  - The 6 differing instances differ between two runs of the same "on" code too. A repeat on2 differs on exactly the same 6.
  - The cause is run-to-run nondeterminism: trash_can's and teddy's templates hash differently in every job, and toy_car 310001 differs
    in a render visibility check. It is not the search: the paths are identical.
  - CR_FAST_ASTAR is the camera-relocalization default (a fast switch, on unless `=0`).
  - CR_OBSTACLE_HEADROOM (unchanged) would have skipped 14 / 5 / 6 / 0 goals on vacuum / trash_can / teddy / toy_car.
- v2only N 4: 8 attempts, 4 written, 4 / 4 demos, sample wall 822 s, 0.063 GPU-h per success. Before: N 1 timed out at 2400 s
  (attempts 131-707 s); now attempts take 9-61 s. The headroom skip did not act in this run.

### The attach map kind (i7-attachmap, 2026-11-07)

Owner: "No v2 attach map" (attachment_01 / 04 made 0 attempts in every `v2only` arm). Their lids (tnwisp / pbmsji) have v2 grasps
and their PICK maps hit (`lid__tnwisp__s1__attachment_free` h1.0505, `lid__pbmsji__s0.65__attachment_free` h0.7853). What blocked
them is the second band the attachment task reads: the attach-reach band, which is legacy (`LEGACY_BAND_BLOCKED
ReachBands.load(... grasp_key=attach:lid__tnwisp, placed=['attach:lid/tnwisp'])` on every edge, then `setup rejected: [no_stance]`).

**What the legacy attach band models.** It is a per-scene band of the PARTNER (the pot), not of the lid.
- Its "grasps" are the pseudo-grasps `T_partner_male ∘ T_object_eef`: the eef poses that seat the held lid on the partner.
- A cell (the partner root on the top) is in it when one (partner yaw, lid grasp) pair passes two IK checks: the seat eef pose, and a
  pre-grasp 10 cm back along the GRIPPER's approach. The support is ignored at the seat; the partner is not in the scene.
- The task reads only cells from it, at a stance:
  - `band_cells_world` gives the partner cells that `_partner_room` / `_attach_and_brace` try;
  - `joint_placement` gives the work cell (`classes` is None for library grasps).
- Everything after that is exact: partner support / gap / frame, the hand vs the partner, the `[pre-attach, seat]` exact IK with the
  lid in hand, the fingers in the partner, the carry, the brace.

**The attach map** (`sampling/attach_maps.py`, schema `isg-attach-map/1`) is its scene-independent replacement: the held male at its
SEAT on the partner, in the synthetic world of the reach maps.
- **Cell**: the partner ROOT xy (base frame), the partner upright (rest identity, as the task places it). 8 yaw slots. The root is at
  `h + bottom offset + 3 mm` (`object_pose_on`).
- **Grasp axis**: the pick map's (`reach_maps.grasp_set`: the v2 grasps of the stable rest, twins, family members). Each grasp carries
  the fingers' measured hold opening (`finger_q`, reversed for a twin).
- **The pair** (`AttachSpec`, measured in Isaac Sim by the task's own code: `attachment_links`, `insertion_axis_rule`,
  `predicates.seat_axis`): `T_partner_male`, the insertion axis, the seat axis, the partner's collision pieces and its bottom offset.
- **The chain the oracle runs after the carry**, one eef orientation:
  1. `pre`: the recorded pre-attach, `pre_attach_back_m` (0.10) along the insertion axis;
  2. `hover`: the seat raised along the seat axis by the HOVER RULE;
  3. `touchdown`: the end of `_touch_down`'s leg, i.e. the hover lowered by the part's clearance + 4 mm;
  4. `seat`: the pose the sampler's exact check reaches.
- **Checks per (d_e, trunk, cell, yaw, grasp)**:
  - batched IK of the pre, each next pose seeded from the previous one (the smallest joint step <= 1 rad, else `branch_jump`);
  - the hand hulls (fingers closed at the hold) vs the slab at every chain pose;
  - per d_e, every non-finger sphere vs the slab at the chain + 3 interpolated pre -> hover configurations;
  - the arm (not the hand) vs the partner's pieces at all but the seat (the exact check ignores the partner there too).
- **Statuses**: the reach-map codes plus `ik_hover` / `ik_touchdown` / `ik_seat` / `partner_arm`.
- **Hover rules** (part of the identity): `box`, today's `execute.attach_hover_targets` (unit test: same height / clearance / top as
  the oracle's function on the same pieces); and `swept`, i7-attachhover's option B (the first touch by an exact Minkowski-difference
  ray cast; it equals `convex_sweep.first_touch` to 8e-17 m on 300 random cases; no touch -> box). The sampler reads the rule the
  oracle executes (`ExecuteConfig.attach_hover_rule` where the oracle has it, else `box`: this trunk).
- **Identity** (`map_key`): kind, arm, the male (v2 entry, identity key, rest, grasp-set digest), the partner (category / model /
  scale / geometry digest), the pair spec digest (T_partner_male, both axes, bottom offset, drop, both geometry digests), the hover rule
  + clear / footprint margin / touchdown extra / axis-z clamp, `pre_attach_back_m`, the height, the edge distances, the config, the
  contact-policy key, the robot digest. `AttachMap.find` checks every field and the stored key; files `amap__attach__*` (never read by
  `ReachMap.find`).
- **Consumer** (`Attachment._attach_map_view`, v2 only): `attach_maps.sampler_view` (the male's v2 entry, the top's height through
  the anchors, the right arm), then the map's pair spec against the live template's (`live_mismatch`: T_partner_male 1 mm / 0.01 rad,
  insertion axis 0.01 rad, bottom offset 2 mm) -> the band view (`map_sampling.as_bands`). A miss logs `MAP_REQUEST {"kind":
  "attach", ...}` and rejects `missing_map`. Off v2 only the legacy band is loaded / computed as before (`_attach_band`). The exact
  checks are unchanged.
- **Build**: `scripts/grasp_bands/job_attach.sh` (`python -m ispatialgym.sampling.attach_maps build --pairs ...`: spec, then one map
  per height x rule x arm); pairs `scripts/grasp_bands/attach_pairs_20261107.json`.

**Built (2026-11-07, job 18368282, trunk_1c3b2b28, `$OFH/attachmap_20261107`, right arm, ~30 s per map).**
- **Specs**:
  - tnwisp / teapot mvrhya: `T_partner_male` (-0.00185, 0, 0.0533), yaw 0, both axes +z;
  - pbmsji / stockpot oshwps at 0.65: (0, 0.0009, 0.0797), yaw 90 deg, both axes +z.
  - Against the joined lid of the 8 i7_ah demos: <= 0.07 mm / 0.04 deg.
- **Hover table** (map vs the oracle's own logs, i7_ah box arm + swept recollects):

  | pair | box hover | box touchdown end | first touch | swept hover |
  |---|---|---|---|---|
  | pbmsji / oshwps (map) | 41.56 mm | 34.56 mm | 13.27 mm | 16.27 mm |
  | pbmsji / oshwps (oracle) | 41.6 mm | 34.6 mm | 12.9-14.1 mm | 15.9-17.1 mm |
  | tnwisp / mvrhya (map) | 7.95 mm | 0.95 mm | 4.90 mm | 7.90 mm |
  | tnwisp / mvrhya (oracle) | 9.5-9.7 mm | - | 5.0-5.1 mm | 8.1 mm |

  tnwisp's box hover is 1.6-1.8 mm lower in the map: the executed in-hand pose sits lower on the knob than the library row the map
  uses.
- **Feasible partner cells per trunk** (d_e 0.35 / 0.45 / 0.55; box = swept within 1 cell):
  - tnwisp at h 1.0505 (a bar): 6 trunks reach (t0.00 k-0.6: 50 / 42 / 32; t0.50 k0: 28 / 11 / 3); t0.75 / t0.95 none;
  - pbmsji at h 0.7853: 16 trunks, 7-88 cells.

**All four attachment aliases** (scope addition, 2026-11-07). Cap xxqyrt (attachment_02) and cork ncxgpe (attachment_03) got v2 grasps
in i7-attach round 2; under v2only they stopped at the same legacy attach band. The consumer serves them unchanged.
- **Inputs**: the attach agent's staged roots `$OFH/attach_20261107/stage_{cap,cork}` (read-only; the anchors are v2maps_i7's).
- **Builds** (jobs 18368635 / 18368636, about 4 min each):
  - cap / lemonade_bottle yprkek at h 0.8126:
    - `T_partner_male` z 0.140, axes +z;
    - hover box 13.2 mm, first touch 10.2 mm;
    - 15 trunks with 3-84 cells at d_e 0.45.
  - cork / wine_bottle inkqch at h 0.7853:
    - the insertion axis is the centroid->female rule (the link +z is 90.4 deg off), and the seat axis is identical;
    - hover box 21.3 mm, first touch 18.1 mm;
    - 16 trunks with 36-104 cells.
- **The cork result depends on the owner's `attach_axis` "seat" decision.** Its two grasps exist only under that opt-in candidates
  switch.
- **The partner's left contact band** (the brace of attachment_03, `brace_search` on) is not a legacy band under v2only.
  `load_left_contact_band` returns None, so the brace is decided by `joint_placement`'s reach proxy and the exact two-arm IK.
  No brace map is built: the brace has no map consumer yet.

**Validation** (`--grasps v2only` N 4, tag `i7_attmap`, trunk_1c3b2b28). Every alias hits its BOX attach map: the live pair spec matched,
and there are 0 `LEGACY_BAND_BLOCKED` / `MAP_REQUEST` / `missing_map`. In total: 40 attempts, 16 written, 7 / 16 demos ok,
0.34 GPU-h per success.

| alias | attempts | written | demos ok | GPU-h | per success | before (i7_ah: `--grasps v2` + legacy band) |
|---|---|---|---|---|---|---|
| attachment_01 lid / teapot | 5 | 4 | 3 | 0.36 | 0.12 | 5 / 4 / 2, 0.53 |
| attachment_04 lid / stockpot | 8 | 4 | 2 | 0.42 | 0.21 | 16 / 4 / 2, 0.82 |
| attachment_02 cap / lemonade bottle | 6 | 4 | 0 | 0.47 | - | legacy grasps |
| attachment_03 cork / wine bottle (depends on `attach_axis` "seat") | 21 | 4 | 2 | 1.10 | 0.55 | legacy grasps |

What still fails is upstream of the attach map, or in the oracle's touchdown:
- **attachment_02**: the cap's one grasp, top_0, does not hold (`failed_grasp` at the pick, 4 / 4). The attach never runs.
- **attachment_04**:
  - one carry reorient is `plan_failed` (a 3.27 rad joint swing);
  - in one pick, the wrist camera touches the stockpot.
- **attachment_01**: one `failed_grasp`.
- **attachment_03**:
  - the cork's rest is unstable (7 attempts);
  - the pick's feasible set F is empty (8 attempts);
  - in 2 of 4 demos the touchdown found no contact within 7 mm, so the cork was released but not attached.

**Swept landed with the maps** (owner decision 2026-11-07, branch `i7-attach-swept`):
- **Commits**: bd9e1671 merges i7-attachhover; 54439eee makes `execute.ATTACH_HOVER_DEFAULT = "swept"`.
- **Consumer**: it reads the oracle's own resolver, so it now serves the swept maps.
- **Validation** `i7_attswept` (v2only N 4, 0.57 GPU-h per success over all four):

  | alias | demos ok | GPU-h per success | touched | drop |
  |---|---|---|---|---|
  | 01 | 2 / 4 | 0.23 | 3 / 3 | <= 0.1 mm |
  | 04 | 3 / 4 | 0.18 | 3 / 3 | <= 0.5 mm (box: about 22 mm) |
  | 02 | 0 / 4 | - | - | - |
  | 03 (seat) | 1 / 4 | 2.03 GPU-h total | - | - |

- **What still fails**:
  - 02: the cap's grasp (`failed_grasp`);
  - 03: the cork's rest stability (23 rejections) and in-hand slip;
  - the grasp witness residual (`plan_failed`);
  - one lid joined "not aligned".

### i7-plateqtf (2026-11-07): one merged plate qtfzeq entry, side_9, the place stage, the expanding tilt / yaw map

Branch `i7-plateqtf` (i7-unblock e9f79b96 + i7-padsweep ff22a5d9, merge 3556e037). Record: `$OFH/plateqtf_20261107/`, DECISIONS.md
"i7-plateqtf" lines. Owner decisions 2026-11-07 23:40 (items 1-3).

**1. One merged plate qtfzeq s1x1x3 entry.** One build on trunk_3556e037 (the plate switches and the pre-shape sweep are the defaults there)
with the families `lip_shallow` + `rim_straddle` via `cand_overrides` (+ `gate_dump`). All grasps are validated together: the default caps
(`max_representatives` 10, the validator's `max_trials` 10 / `max_valid` 6) cut the merged list at the first 10 representatives (all
rim_straddle; a first try, `merged_capped`), so the build sets `max_representatives` 40 and `ISG_GLIB_VAL_OVERRIDES` max_trials / max_valid
40. Funnel: 46 candidates (top 10, side 12, lip_shallow 12, rim_straddle 12) -> 31 geometry-ok (28 top-type, 3 side) -> 30 representatives
-> 27 physics-valid (rim_straddle 12, lip_shallow 11, top_4 / top_5 / top_6, side_9; lip_shallow_1, side_1, side_8 invalid: no placement
of the realsense / gripper link). The switches-on entry's 4 valid grasps come back with identical poses. side_9's reviewed rotational family (owner
prompt A 2a) is carried by pose (`carry_family.py`: f6 -> f29) before the maps are built (the families digest is in the map identity). Maps
at 0.8126, right + left (one job per arm). Staged `$OFH/v2maps_i7pq` = `cp -a v2maps_i7unblock2` + the merged entry (the rim_straddle-only
entry and its maps in `_superseded_merged`; `legacy_key` plate__qtfzeq__st in library.json, links.json updated).

**side_9 in sampling (diagnosis).** The rejection "non-finger link touches the support" is the support policy's sphere stage
(`ik.py` `_support_policy`: every non-finger link's cuRobo spheres against the support, margin 0). The link is the PALM
(`{arm}_gripper_link`): at side_9's nominal pose its exact hull is +0.99 mm above the table, its lowest sphere (r 15 mm incl. the model's
2 mm buffer) -0.9 mm - the nominal pose itself fails the rule by 0.9 mm. The library validation (8 yaws x 4 offsets x 4 trunks x 12 seeds,
any passing configuration) and the map (first passing seed) keep it because an IK solution within the tolerance (5 mm / 0.05 rad) can lift
the palm; the sampler's exact IK of one witness per pair lands on either side (i7_unblock: 245 end-pose + 241 approach-tail rejections). The
table is not the cause (8 pieces, spread 0.3 mm about 0.8126). Not fixable by the sampler without changing a shared rule; the options are
an owner decision (below). With the merged entry side_9 is one of 27 grasps: F is drawn without replacement, so its failure falls through
to the top grasps.

**Overhang fallback.** Mechanism (1), the sampler-side `pick_overhang_push` (v2/overhang.py, `IORSLarge._overhang_pick`), is the one that
fits qtfzeq (no library entry or map needed; the palm is past the edge by construction). It exists only in IORSLarge; under_small_03 is
IORSSmall, and its oracle program (`programs.program_iors_small`) calls `_pick` without the push, so the fallback needs an oracle program
change (`_pick_actions`) - not wired (owner). Mechanism (2), the i7-overhang library entry + overhang map kind, is not on this branch and its
consumer reads the overhang key only in mode `only` (no fallback). plate luhkiz never used either; hardback bqvvvx uses (1) on the trunk and
(2) on i7-overhang (mode only).

**2. The place stage (diagnosis, no rule change).** The task: `(under place_mat plate)` - the MAT is under the plate (problem44, definition
44); the sampler moves the plate onto the mat (`iors_small.goal_pose`) and the oracle places "on" the mat: they agree (legacy instance 300004:
the plate settled +2.6 mm on the 2.8 mm mat). gripper_not_in_frame is `iors_small._segment_draw`'s `eef_in_frame` box (+-6 cm, -2..+6 cm
along the approach, 5 %) at the place AND the pre-place (10 cm back along the approach) from the manipulation stance's head camera. The cause
is the layout: the mat is 0.46 x 0.57 m, so the plate's goal sits 0.33-0.52 m to the side of the stance; the side of the image facing the
partner binds, and the pre-place (higher, nearer the camera) is the tightest pose. Rejections by trunk (rim_straddle run): 0.0/-0.61 5+5,
0.25/-0.53 11+6, 0.5/-0.45 13+3, 0.25/-0.80 3+0 (place + pre-place); the legacy successes all at 0.25/-0.53. CPU reconstruction (pass rate,
goal 0.86 m ahead, lateral 0.40): rim_straddle 67 %, legacy 74 %, side_9 71 % at 0.25/-0.53; 7 / 25 / 35 % at 0.25/-0.80 - stance tilt and the
goal's lateral dominate, the grasp moves it 5-15 points. Partner placement ("no partner yaw of 18 admits a gap"): the frame mask needs the
whole mat in the INIT image (79 % of the yaw lines fail it) and the ends mask the 1.16 m edges; no direct grasp dependence. Finding (rule
not changed): the draw judges the pre-place with the contact box, while program_check judges in-between poses on the fingertip point (+2 cm
slack) and QA is fingertip-only - the draw is stricter than both at the pre-place (rim_straddle at lateral 0.40: 67 % -> 100 % under the
program-check criterion); 14 / 47 rim_straddle and 6 / 9 legacy frame rejections were pre-place.

**3. The expanding, self-correcting reach map** (`v2/map_tilt.py`, c332d214; default ON 590a6642, `ISPATIALGYM_FAST_MAP_TILT_INHERIT=0` =
the nearest built tilt as before; TILT_SNAP unchanged, opt-in).
- Inherited columns: a map view read at a stance tilt more than 5 mrad from every built tilt and between two built tilts of the same
  translate gets a column keyed by the tilt rounded to 0.01: the UNION of the two bracketing built columns (cells; pairs and generic class
  bits re-indexed). Outside the built range or on a built tilt: unchanged. `ReachBands._col` routes `band` / `cell_index` / `feasible` /
  `classes_at` / `band_for_class` through it (map views only; a mirrored view keeps the nearest tilt). Yaws: `yaw_union` - an object yaw
  between two of the 8 slots inherits the union of the two slots' grasps; Phase 3 records the drawn (cell, relative yaw, grasp) outcomes
  under the same rule. Today no production path reads the map's (yaw, grasp) pairs (F is screened by IK; MAP_F is off), so the yaw entries
  accumulate evidence only.
- Invalidation: only reachability counts. An empty F on an inherited cell is screened again with every scene object out of the collision
  world (as DIAG_EMPTY_F): still empty = `reach_fail`, pairs = the scene (`other_fail`). An exhausted F re-runs its last failing pair's exact
  IK with the scene out and counts it only for an IK / reach / joint-limit reason (`map_tilt.reach_reason`; the support policy, collisions,
  the camera frame never). K = 2 reach failures with no success invalidate the cell (and the yaw pair) for that column
  (`ISPATIALGYM_FAST_MAP_TILT_K`); one exact success confirms it, and a confirmed entry is never invalidated. An invalidated focus is not
  screened again (`skipped_invalidated`); a cached start-cell candidate standing on it leaves iors_small's draws (`candidates_dropped`).
- Overlay, never the frozen map: events are keyed by the map key; each process appends JSON lines to its OWN shard
  `<ISPATIALGYM_MAP_OVERLAY_SHARDS>/<map key>/<host>_<pid>.jsonl` (job_fast.sh: `$JOBDIR/map_overlay`) - one writer per file, no race.
  `python -m ispatialgym.sampling.v2.map_tilt merge --root <overlay root> <shard globs>` folds them offline into a new
  `<root>/<map key>/v<N>.json` (temp file + rename; refuses to overwrite; the shards it folded are listed and skipped next time). A job reads
  a merged version only with `ISPATIALGYM_MAP_OVERLAY=<root>` (latest, or `ISPATIALGYM_MAP_OVERLAY_VERSION`); the version then enters the
  view's identity (`meta.map_tilt` = `inherit_k2+ov<N>`, part of iors_small's start-cell band signature when the switch is on).
- No GPU builds of true columns. Kit counters (per attempt, + `map_tilt.by_map.<key8>.*`): `map_tilt.inherited_columns`,
  `.inherited_feasible` (cells of the new columns), `.cells_invalidated`, `.cells_confirmed`, `.empty_f_on_inherited`, `.reach_fail`,
  `.other_fail`, `.skipped_invalidated`, `.candidates_dropped`; `map_yaw.pairs_confirmed` / `.pairs_invalidated`; one `MAP_TILT` log line per
  recorded failure.
- Tests: `test_map_tilt.py` (union of the neighbours, yaw union, built tilt / yaw / switch off unchanged, repeated reach failure invalidates
  and collision / policy / frame never, the empty-F verification and the dropped candidate, the overlay never touches the map + merge).

**Validation** (`--grasps v2only` N 4, tag `i7_pq_merged`, trunk_590a6642, `/v2maps_i7pq`): 32 attempts, 3 written (the 4th instance hit its
15-attempt cap), demos 3 / 3 ok, 1.34 GPU-h = 0.448 per success (every earlier v2 arm: 60 / 0). Written grasps: lip_shallow_5~r180 (2), top_4.
All 29 rejections are target_placement, now mostly at the place stage (place screening: place_unreachable / approach_untrackable / joint
limit; gripper_not_in_frame at the pre-place; partner_placement), empty F in a minority. Counters (sum): inherited_columns 8,
inherited_feasible 450, empty_f_on_inherited 133, reach_fail 132 (every empty F stayed empty with the scene out), other_fail 3,
cells_invalidated 32, cells_confirmed 57, skipped_invalidated 5, map_yaw.pairs_confirmed 82. Merged overlay v1
(`$OFH/plateqtf_20261107/map_overlay`): right map ce1af987a8df9d69 129 entries / 22 invalidated / 107 confirmed, left b2f36adb26c194ad
45 / 9 / 32 - not read by any job yet.

### Per-alias outcome memory (i7-memory, 2026-11-08; owner decision 01:45 part C)

Down-weights configurations that keep failing for an alias. It runs unattended, on top of the sampler's own order. Code:
`ispatialgym/sampling/alias_memory.py`. Exclusions: `ispatialgym/assets/alias_memory/exclusions.json`. Tests:
`tests/test_alias_memory.py`.

- **Record:** `job_fast.sh` runs `alias_memory record <job dir>` after its collectors. It writes one row per demo into the job's own
  shard, `alias_memory_shard.jsonl` (one writer per job, so no races). A row holds:
  - the PLACE key: arm, place stance (`placement`, else the tabletop `manipulation` stance), stand-off bin (0.05 m), trunk
    translate / tilt (0.01), goal yaw slot (the sampler's pi / 8 pair bucket);
  - the PICK key: arm, `pick` / `manipulation` stance, stand-off, trunk, grasp family (the twin suffix `~r180` folded in), entry key;
  - the outcome, failure type and STAGE;
  - the context: trunk snapshot; alias customs version (the TRO's `task_spec_sha256`); entry / map key fingerprint (the v2 library
    content and the content-hashed map file names of the alias's models, so a staged copy of the same data has the same keys);
  - provenance: the pinned memory version and the placement stance's `alias_memory` extra.
- **Attribute:** stages come from the verdict's action in the program.
  - `pick`: up to and including the lift, with the push-to-pick.
  - `carry`: between the lift and the first place.
  - `place`: the place and everything after it.
  - `outcome`: verdicts with no action (symbolic_only, not_reached, not_settled).
  - `manipulation`: programs with no grasp-and-place.
  - A pick or manipulation failure counts against the PICK key and leaves the place unobserved.
  - A carry, place or outcome failure counts against the PLACE key and counts as a pick pass.
  - A success is a pass for both.
  - Excluded: infrastructure (status error, rc not 0 / 2, timeout / wall clock, OOM, a missing meta or sidecar) and documented
    known bugs scoped by snapshot (attachment_01 / 04 recorded-IK `plan_failed`).
- **Aggregate:** `alias_memory aggregate --root R` merges the shards and back-fills a finished job that has none. It writes a new
  immutable `R/alias_memory/vNNNN` (`aliases/<alias>.json`, `meta.json`, `report.json` / `report.md`) and updates `LATEST`.
  - `submit_fast.py submit` pins the version in job.env (`ISPATIALGYM_ALIAS_MEMORY`, `--alias-memory auto|off|<path>`; auto = LATEST,
    off while none exists).
  - It also queues `job_memory.sh`: a CPU job, `afterany` the GPU job. It builds a version only when no other job of the batch tag
    is queued or running, so the batch's last job aggregates.
- **Sample:** each (alias, bin) gets a Beta posterior with a flat prior.
  - Levels: arm x trunk, then x stand-off, then x yaw (pick: x family). The finest level with >= `min_n` attributed demos decides;
    sparser data backs off to the coarser level.
  - A bin is DOWN-WEIGHTED when P(theta_bin < theta_rest of the alias) >= `p_worse` and its posterior mean <= `max_mean`. An alias
    where everything fails down-weights nothing.
  - The hook sits in `IORSLarge._couple` (stance C for every large-scale place mode). It keeps the prior order: today
    travel / edge-rank, later i7-placeorder's order if adopted.
  - Per coupling, a down-weighted bin keeps its place with probability max(`floor`, theta_bin / theta_rest), one Thompson draw per
    bin. Otherwise its candidates move to the end and are still tried if everything before them fails.
  - With nothing down-weighted the order is unchanged and no random number is drawn. The memory's own generator is derived from the
    sampler's state without advancing it.
  - Kit counters: `alias_memory.loaded`, `.place.no_down`, `.place.down_bins`, `.place.deferred`, `.place.kept_down`,
    `.blocked_not_train`.
  - The yaw key is recorded and reported only. Since i7-memory2 the pick key also steers: the pick hook (below).
- **Forget:** evidence decays by 0.8 per newer trunk snapshot of the alias. It is RESET under another customs version or another
  entry / map key fingerprint (`customs_change` / `keys_change` = 0).
- **Guardrails** are in each version's report, per batch and alias:
  - bins down-weighted and the share of the observed space (bins, demos) they hold;
  - yield and GPU-h per success, batch vs before.
  - Alerts only:
    - space share > 50 %;
    - yield drop > 0.25 (>= 4 demos on both sides);
    - all failed (>= 4 demos: a bug, not a configuration).
  - Shared-fix candidates: the same (arm, trunk) down-weighted in >= 3 aliases. The loop never changes a shared rule.
- **Eval generation:** the memory is consulted only after `enable_for_split("train")`, which only `multiply_isg --split train` calls.
  `sample_isg` (eval templates) switches it off whatever the environment says.
- **Default: ON** with the proposed thresholds. The owner's text says "(unattended)" and "must run without the owner".
  - The thresholds sit in each version's `meta.json`. A `<root>/alias_memory/thresholds.json` (the owner's) wins.
  - Off per job: `--alias-memory off`.

**Proposed thresholds** come from the replay over `$OFH/out`:
- source: `$OFH/memory_20261108/replay_61d4024d`, 2375 demos, 88 batches (tags in finish order), snapshot `trunk_61d4024d`;
- excluded: 3 status error, 1 timeout, 2 attach recorded-IK known bug;
- seed version: `memory_20261108/alias_memory/v0002`;
- check: for every demo, the memory of the batches before it, in its own context.

| setting | inside_large_01 0.50 caught | 0.25 caught | later demos in flagged bins (ok) | (alias, context) findings |
|---|---|---|---|---|
| **proposed: min_n 3, p_worse 0.95, max_mean 0.35, decay 0.8, reset** | yes | no | 10 (1) | 4 |
| min_n 2 / min_n 4 | yes / NO | no | 12 (4) / 12 (3) | 6 / 3 |
| p_worse 0.90 / 0.99 | yes / NO | no | 14 (3) / 0 | 7 / 1 |
| max_mean 0.25 / 0.50 | yes / yes | no | 9 (0) / 19 (5) | 3 / 12 |
| snapshot decay 0.5 / 1.0 | yes / yes | no | 9 (1) / 16 (5) | 4 / 9 |
| customs / keys change 0.5 instead of reset | yes | no | 10 (2) | 6 |
| no forgetting at all | yes | no | 24 (14) | 24 |

- `min_n` 3: the inside_large_01 case has exactly 3 place-stage failures (the 4th, i7_oh_m 300086, is `plan_failed` at the grasp:
  pick stage).
- The owner's example cutoff (95 % upper bound < 10 %) needs 28 straight failures under a flat prior. It catches nothing in
  `$OFH/out`.
- Floor 5 %: the owner's figure.
- Forgetting is necessary. Without it, 14 of 24 later flagged demos succeeded: configurations healthy again after a fix.

**Replay findings:** four (alias, context) pairs are down-weighted; no bin with a healthy rate is caught.

- inside_large_01 place (right, 0.50, -0.60, 0.85 m): 0 / 3 (2 robot_wobble, 1 robot_touching_itself). P(worse) 0.987, mean 0.22,
  vs the alias's rest 3.6 / 0.
  - Context: i7_oh + i7_oh_m (customs 262df5, keys 22e3a0).
  - (right, 0.25, -0.60, 0.85 m) 4 / 4 is NOT caught.
  - In the i7_oh_R / int1_1 context (lean_left customs and libraries) the evidence is reset by rule.
  - Back-off: in its context, 0.50 at an unseen stand-off falls back to the trunk level and is down-weighted too.
- inside_large_03 place (left, 0.0, -0.60, 0.65 m): 17 demos, 9 collision + 3 plan_failed. Rest 25 / 0. Context: fix2 / ft*.
- touching_04 place (right, 0.0, -0.61, 0.35 m): 0 / 4 (2 gripper_out_of_fov, 2 symbolic_only), rest 4.8 / 3.8. Small-scale task:
  recorded, no hook.
- inside_04 pick (left, 0.25, -0.53, 0.35 m, top_3): 0 / 5 (4 plan_failed, 1 collision). Pick: recorded only.
- Later demos that fell in a flagged bin (all pick-level): 9 of 10 failed.
  - ontop_small_02 pick (right, 0.25, -0.53): 5 later robot_wobble / pause. The placeorder scan found the same: 10 grasp-stage
    wobbles at 0.25.
  - close_small_02 pick: 4 object_tilt.
  - The one success: inside_04 after i_fix13.
- In the aliases' LATEST contexts, 2 of 99 aliases have a down-weighted place bin (inside_large_03, touching_04).
- Alerts for the last batch, int1_1:
  - disembedding_01 and _10: all failed;
  - inside_large_01: yield 45 / 57 -> 1 / 4, 3 failed_grasp at the pick;
  - nextto_01: yield 73 / 89 -> 2 / 4.
- Shared-fix candidates: none.

**Limits:**
- Hook coverage: the place hook covers stance C of the large-scale tasks only. Tabletop (single-stance) and disembedding places are
  recorded, not steered.
- Key fingerprint: the keys cover the alias's model libraries and maps (its customs' grasp keys / representatives). They do not cover
  partner-side maps (interiors, attach).
- Snapshot order: snapshots are ordered by their first demo for the alias, not by git ancestry.

### Outcome memory: owner decisions 2.1-2.4, modes and the pick hook (i7-memory2, 2026-11-08)

The owner's answers to `tmp/outputs/open_decisions_20261108.md` section 2 (verbatim in `tmp/outputs/owner_answers_20261108.md`):
- 2.1 "adopt the proposed values";
- 2.2 "reset (safe: a fix starts clean) in actual production - but for the validation runs carry over at reduced weight to maximize
  speed";
- 2.3 "add a pick hook";
- 2.4 "start empty at production. but for validation runs seed from the replay ($OFH/memory_20261108/alias_memory). and at reduced
  weight to maximize speed".

**Thresholds (2.1).** min_n 3, p_worse 0.95, max_mean 0.35, floor 0.05, snapshot decay 0.8 are the `Thresholds` defaults in code and
the content of each store's `thresholds.json`. The aggregator reads that file (it wins over the code) and writes it with the mode's
defaults when it is missing. Every version's `meta.json` carries the thresholds it was built with; the sampler reads them from there.

**Two modes, one switch, two stores (2.2 / 2.4).** `submit_fast.py submit ... --memory-mode production|validation` writes
`ISPATIALGYM_ALIAS_MEMORY_MODE` into job.env. That one switch picks:
- the store the job's sampler reads (pinned, as before, by `--alias-memory auto` = the mode store's LATEST);
- the store its demos are aggregated into (`job_memory.sh <root> <tag> <code> <mode>` -> `alias_memory aggregate --mode <mode>`).

| | production (default) | validation (opt-in) |
|---|---|---|
| store | `<root>/alias_memory_production` | `<root>/alias_memory_validation` |
| demos aggregated | production-mode jobs only: starts EMPTY | the seed + jobs from before the modes (both marked `seed`) + validation-mode jobs |
| customs / keys change | RESET (weight 0) | carried over at w = 1/3, flat |
| seeded demo | (none) | w = 1/3, flat |
| own context, older snapshot | 0.8 per newer snapshot | 0.8 per newer snapshot, never below w |
| bootstrap | the first production job's batch aggregation | `alias_memory aggregate --root R --mode validation --no-gpu` (pins the seed in `seed.json`) |

- `<root>/alias_memory` is the LEGACY store, and neither mode reads it.
  - Snapshots from before the modes (every round-4 branch until it carries this code) aggregate every demo of the root into it after
    each batch: `$OFH/alias_memory` v0001 (i7tray_gs, trunk_fcddc495) and v0002 (i7i2_fridgeB2, trunk_c7fcb37a), 2,672 demos each,
    with no mode.
  - A production store there would not have started empty, and its LATEST would flip with every old-snapshot batch. So production
    has its own directory, and a version without a mode is `legacy`: submit_fast and the sampler refuse it in either mode.
- Separation is enforced three times:
  - each mode aggregates only its own demos;
  - a store's `thresholds.json` carries its mode, and aggregating the other mode into it is refused;
  - `submit_fast` refuses to pin a version of the other mode, and the sampler ignores one (`alias_memory.mode_mismatch`, memory off).
- `--memory-mode validation` with no seeded store stops the submit and prints the bootstrap command.
- Seed: `$OFH/memory_20261108/alias_memory`, LATEST at seeding = v0003 (2,383 demos of 99 aliases, the replay plus the i7_mem_smoke batch).

**The weight (w = 1/3: three carried demos count as one fresh one).** Measured on the 2,375-demo replay and the v0003 seed:
- inside_large_01's (right, 0.50, -0.60, 0.85 m) bin is 0 / 3. To be flagged, its mean 1 / (2 + 3w) must be <= 0.35, so w >= 0.286.
  - At w = 1/3 it is flagged from the seed at a new snapshot under today's customs / keys: mean 0.333, P(worse) 0.991.
  - At w = 0.25 it is not flagged (mean 0.36).
- Carried-over evidence is discounted once, flat, not also by its age. Compounded with the 0.8-per-snapshot decay (the i7-memory
  rule `customs_change x 0.8^k`), the bin is 4-5 snapshots old, and no weight below ~0.9 flags it (w 0.8: mean 0.373).
- Seeded at w = 1/3, a validation store's first attempt defers exactly two bins, both with a failing record:
  - inside_large_01 place above (0 / 3);
  - inside_large_03 place (left, 0.0, -0.60, 0.65 m): 5 / 17, against a rest of 25 / 0.
  - No pick bin is flagged from the seed.
- Chronological replay (every demo judged by the memory of the batches before it, at the hooks' levels):

| | later demos in flagged bins | failed | flagged bins (alias x bin) |
|---|---|---|---|
| production (reset) | 10 (place 0, pick 10) | 9 | 4 |
| **validation w = 1/3** | **10 (place 1, pick 9)** | **8** | 5 |
| validation w = 0.35 / 0.5 / 0.75 / 1.0 | 11 / 14 / 18 / 24 | 8 / 8 / 9 / 10 | |

- At w = 1/3, every bin flagged had a failing record when it was flagged: at most 3 ok of 7 (the carried successes included), mean
  <= 0.32. The rest of its alias stood at 0.65-0.80 (articulation_close_small_02: 0 / 8 vs 0.47).
  - The one place demo flagged and then succeeding: ontop_small_01 (left, 0.25, -0.80, 0.35 m), 0 / 3 when flagged. It is a tabletop
    place, which no hook steers.
  - The pick demo flagged and then succeeding: inside_04 after i_fix13 (1 / 6 when flagged); also flagged in production.
- From w = 0.5 on, bins healthy again after a fix are flagged: 6 of 14 later flagged demos succeeded.

**Pick hook (2.3).** `alias_memory.reorder_pick` is the same `reorder` as the place hook (prior order kept; a flagged bin's candidates
move to the end unless a Thompson draw keeps them, never below the 5 % floor; never removed; off or nothing flagged = the order unchanged,
no random number drawn). It runs over the pick (manipulation) stance candidates at three sites:
- `IORSLarge.sample_instance`: stance B, after the pre-screen order; the arm is the lead stance's.
- `IORSSmall.sample_instance`: the manipulation stance, after the frame-aim / shelf orders; the default segment arm.
- `ArticulationSmall.sample_instance`: the manipulation stance, after the depth preferences; the handle arm.

- Key: arm x trunk (translate, tilt), then x stand-off (`PICK_HOOK_LEVELS` 2; back-off as the place hook).
- Why not the grasp family: on the replay the stance-level findings were followed by failure in 9 of 10 later demos, the
  family-level keys in fewer:
  - (arm, entry, family): 6 of 10;
  - (entry, family): 10 of 15;
  - stance x stand-off x family: 10 of 12.
- The family stays recorded and reported. ontop_small_02's top_11 family (10 ok / 10 failed with the right arm vs top_6 18 / 1) is
  the one family finding; it overlaps the flagged trunk 0.25 bin.
- Provenance: `extras["alias_memory_pick"]` on the candidate stance, so on the written pick / manipulation stance. Rows record it
  as `sampled_with_pick`, and the job's `mode`.
- Kit counters, per hook (`place` and `pick`):
  - `alias_memory.<kind>.consulted`;
  - `.flagged`: candidates in flagged bins;
  - `.reordered`: moved to the end;
  - `.explored`: kept in place by the draw / floor;
  - besides `.no_down`, `.down_bins`, `.deferred`, `.kept_down`, `alias_memory.mode.<mode>` and `alias_memory.mode_mismatch`.
- Extended to every family with a pick-stance choice (lead decision 2026-11-08 on 2.3; bedf8e2e):
  - attachment, block design, alignment axial, disembedding: `reorder_pick` at their manipulation candidate ordering, after the
    family's own score or permutation and before its cut to the first 6;
  - articulation large and alignment wall (world vertical): `prompt0.draw_manipulation(memory_arm=...)`. The uniform Phase-2 draw
    leaves out the survivors this call defers (`pick_draw_pool`). With only flagged survivors left it draws among them as before;
    memory off or nothing flagged gives the same draw.
  - A drawn stance is keyed by its grid trunk: the sidecar's `grid_tilt`, the choice the hook makes. The drawn tilt itself is
    continuous.
  - articulation hybrid: no stance choice of its own. The hybrid follow plans from articulation large's drawn stance.
  - The hook sits at the candidate ordering only, not at the start-base draw (i7-startdraw).
- Replay of the same 2,375 demos with rows recomputed from the job dirs (`$OFH/memory2_20261108/families`, CPU 18379422): the
  new families have pick evidence but no flagged bin in either mode:
  - attachment 87 attributed demos / 8 failed;
  - alignment axial 78 / 32, spread over 22 trunk bins;
  - disembedding 138 / 12;
  - articulation large 52 / 5;
  - world vertical 33 / 0.
  - The totals are unchanged: production 10 later flagged / 9 failed, validation 10 / 8.
- Block design has no demo in `$OFH/out`: every attempt rejects at `target_placement`.
  - GPU check `i7_mem2_bd` (block_design_01, v2only, validation, trunk_bedf8e2e, slurm 18379435): 60 attempts, 0 written (all
    `target_placement`), 0.46 GPU-h.
  - The hook site runs: `alias_memory.no_alias_file` 60, `mode.validation` 60, no error.
  - The memory learns only from written demos, so the block-design hook stays inert until its sampler writes one.

**Replay outputs:** `$OFH/memory2_20261108` (`replay_production`, `replay_validation`), from CPU job 18377645 on trunk_52621543.

**Stores on `$OFH` (2026-11-08):**
- `alias_memory_validation/v0001`: the seed v0003 (2,383 demos) plus 235 pre-mode demos, all weighted as seeded, 99 aliases.
- `alias_memory_validation/v0002`: the batch aggregation after the GPU check.
- `alias_memory_production`: `thresholds.json` only. It is empty, so `--alias-memory auto` pins off until the first production batch.

**GPU check `i7_mem2_val`** (inside_large_01, v2only, `v2maps_int1`, `--memory-mode validation`, trunk_52621543, slurm 18377666):
- 7 attempts, 4 written, 2 ok (1 failed_grasp, 1 robot_wobble), 0.284 GPU-h per success.
- Kit over the 7 attempts:
  - place: consulted 51, down_bins 28, flagged 56 candidates (the seeded trunk 0.50 bins), reordered 38, explored 18;
  - pick: consulted 7, no_down 7. The seed holds no pick finding for this alias.
- All four placements are at (right, 0.0, -0.6, 0.75 m). The shards carry `mode: validation` and `sampled_with_pick`.

## Contact libraries and contact maps (branch `i7-contact`, 2026-11-07)

The PUSH / BRACE contacts the oracle executes get the grasp library's treatment: one shared pose rule module
(`ispatialgym/sampling/v2/contact_poses.py`, the samplers' record builders as pure functions, golden-tested against 63925048), per-object
libraries in the object frame (`ispatialgym/grasping/contact_library.py`, schema `isg-contact/1`: the contact, the EXECUTED pre - a straight
push's lowered 0.04 m pre, owner decision 6 -, the fingers, the stroke / arc / partner gap / task goal, the brace stop), physics validation
on the oracle's own `execute_push` / `execute_brace` (`ispatialgym/grasping/validate_contacts.py`: the task's goal check, a kinematic partner
stand-in at the place gap, the object's TRO pose target, the variant's customs `generation`), and reach maps of the entries
(`ispatialgym/sampling/contact_maps.py`: kinds touch_push / touch_brace / push_blade / axial_brace / floor_push; cell = the object ROOT, an axial
brace's at its GOAL pose; identity incl. the pose set digest). Consumers under v2only (`tabletop.need_contact_map`): the touching brace gate
(required: a miss rejects `missing_map`), the axial joint placement (push_blade required; axial_brace OPTIONAL - a miss skips the brace,
counted `contact_map_skipped.axial_brace`). Pipeline: `scripts/grasp_bands/job_contacts.sh` (GC_STAGES build / validate / replay / maps), roots
under `$OFH` only. Run E (2026-11-07): the touching_02 push replay tips (validator friction / mass, not the criterion), the axial replays
need the task's COM hinge in the validator world (not yet); see `$OFH/DECISIONS.md` 2026-11-07 (i7-contact).

### i7-contact round 2 (2026-11-07 evening: 8e798b51, e99b45a3)

- **Axial brace lookup** (i7_contact: brace present 0 / 11). Not the yaw slot (goal 180.0 deg from the stance, err 0), the trunk (translate
  exact, tilt nearest), the stand-off (0.35 / 0.45 / 0.55 = the map's edge distances) or the cell (goal roots x 0.79-0.94, y -0.29..0.00, all
  within 6 cm of a cell). The maps held only the `+end+spin` classes: the validator's 48-entry cap round-robined per yaw over a library
  ordered yaw -> end -> spin -> class (6 `+end+spin` per yaw), the maps take the validated entries, and the joint placement tested both ends
  against the same classes. At goal yaw 180 the `+end`'s stop face is the far face of the robot-right end: ik_pre / reach_bound for the
  left arm at all 11 roots (the `-end`, by symmetry the `+end` at yaw 0, is feasible at 7 / 11). Fix: `map_joint_placement(end_sign)` keeps
  the classes of the pair's end and spin (`contact_maps.axial_end_spin`, the map library's `long_axis` in meta `library.params`);
  `validate_contacts.entry_order` round-robins (yaw, end-spin group); `--skip-validated` keeps an earlier run's verdicts uncounted.
- **Rerun i7_contact2** (trunk_8e798b51, maps re-validated in the (yaw, end, spin) order): axial_01 / 03 / 04 53 attempts, 11 written, 11 / 11
  demos ok, 0.047 GPU-h per success (i7_contact 0.069); brace present still 0 / 11, now a true outcome: at every goal root the end the
  right arm can push (+end: 11 / 11 feasible) has no reachable stop brace (0 / 11), and the end whose stop face the left arm reaches (-end:
  7 / 11) has no feasible push (0 / 11).
- **axial_02**: paperback jhaltj at its spawn scale s1x1x2.5 - v2 export (`glib_jobs_axial02_20261107.json`, job_glib.sh) and a manifest
  entry in `contact_jobs_axial_20261107.json`; library push 30 / 48 valid (s1: 3), brace 48 / 48. i7_contact2: 60 attempts, 0 written -
  the maps hit, every sampled instance repeats an eval instance (`repeats_eval` 52: start base 0.08 m, target 0.00 m / 1-5 deg).
- **Friction** (touching_02 push replay, `set_slab_friction`): mu 0.50 (= the PhysX default) tips 0.463 rad as unbound; mu <= 0.30 passes
  (tilt 0.000, gap <= 2 mm).
- **touching_02 contact_unreachable** (`TOUCH_PUSH_DIAG`): the placed target's root lies 0.80-0.92 m ahead of the push stance; the
  `touch_push` map has no feasible class there (feasible x <= 0.75 m at trunk (0.25, -0.3), <= 0.83 m at (0.0, -0.6)) and the exact chain
  agrees (the 7 unscreened classes fail the IK at their pre-pose, the 8 others the hand screen). The map gate's push side is not enforced
  (`_brace_gate ok False` and the partner pose is kept), so the attempt dies ~30 s later in goal_check.
- **Hand-overlap screen** (owner item 3): the hand model is the real gripper (base plate +-54 mm vs the knuckle spheres +-57 mm); the
  overlaps are real; physically valid napkins 42 / 48, cereal 2 / 9, spray 1 / 23, the napkins' screened pushes load the box into the slab
  with 143 N median (live 4 N). Default-off switches: `ISPATIALGYM_HAND_OVERLAP_RANK=1` (sampler: the overlap ranks, not rejects) and
  `validate_contacts --screen-as-rank`.

## Brace opening: one switch, one rule (branch `i7-contactbrace`, 2026-11-08)

The owner's BRACE OPENING rule (2026-11-07): an open brace opens fully (0.05 per finger), narrower only when the braced face requires it.
`i7-contactbrace` = i7-contact 521dc344 + i7-braceopen dc10d4d2 (merge 51786e00) + the reconciliation f14f8a20.

- **One switch**: `ISPATIALGYM_BRACE_FULL_OPEN` is defined once, in `tabletop` (`BRACE_FULL_OPEN_ENV`, `brace_full_open()`,
  `BRACE_FULL_OPENING_M`, `brace_opening(today, room)`); `contact_poses.BRACE_FULL_OPEN_ENV` / `brace_full_open_on` are those names (the
  copy i7-contact had is gone). Its sites: the touching wide stop (`contact_poses.brace_wide_spreads`: the full-open spread 0.127 m =
  2 x 0.05 + 0.027 first, then 0.09 / 0.07 / 0.05 where the face is too short), the push-close and carton-v4 stops
  (`articulation_small.wide_stop_layout`: min(0.05, face room), inset max(corner inset, opening + 1 cm)), the pack brace
  (`block_design.pack_brace_opening`), and the touching_brace library identity (`resolved_knobs`: `brace_full_open`, the spreads).
- **One opening rule for the checks**: every sampler check of a brace puts the brace hand at `tabletop.brace_fingers` = the opening
  `execute_brace` holds (`actions.brace_is_open`: the record's positive `fingers_opening_m`, else closed) - the two-arm solves
  (`brace_second_arm(brace_fingers=)`, `IKChecker.exact_both_arms(fingers_other=)`), the probes, the retreat, the program check
  (`brace_open_fingers`, the locked brace hand). cb164782's unconditional rule kept; `ISPATIALGYM_TOUCH_BRACE_AT_OPENING` retired (its
  default was already the rule; "0" checked another opening than the oracle).
- **Maps at the executed opening**: a contact map's meta records its library's knobs (`library.knobs`, not in the map key); the
  touch_brace consumer (`tabletop.contact_map_knobs`, `ContactMap.find(knobs_require=)`) reads a map only when its library's
  `brace_full_open` = the run's switch (a map without the record = built off).
- Switch off: library identities, map keys and grasp keys unchanged (tests; the ntbrtz / dkhgxn libraries rebuilt off on trunk_f14f8a20
  are byte-identical to the parent i7-contact 8e798b51 build).
- Kit counters `brace_opening.{passed,failed,recorded}.<opening>` (touching brace, push-close stop).
- Where the opening can change (from the code): touching_01 / 02 / 04 (the touching wide stop), close_02 (carton push-close stop). Not:
  block_design_01 (= block_4 board: `block_pack_tail` false, the pack brace never runs; the pack-tail block_4 variants have no alias),
  open_02 (carton v4 brace `blade_down`, closed), touching_03 (shelf layout, no brace), the axial stop brace (closed). The other carton close
  customs variants have no alias. Libraries (partner-centred push line): dkhgxn's wide stops 0.0315 -> 0.05 (validated 6 valid / 2
  unreachable, as off); ntbrtz has no wide entry; gkpmii's face takes at most 0.09 (unchanged). gkpmii (touching_04's partner) got its v2
  export (`$OFH/contactbrace_20261108/glib_gkpmii`) - touching_04 rejected `missing_map touch_brace` before.
- A/B (`$OFH/contactbrace_20261108`: `sub_ab.sh`, `stage_ab.sh`, `ab_tab.py`): roots `v2maps_icb_off` / `v2maps_icb_on`, a ledger per arm at
  the same base index, a cold sampling cache per job, `--grasps v2only`, tags `icb_off` / `icb_on`.
  close_02 under v2only is blocked on this branch (`legacy_band_blocked`: its articulation band needs i7-artic's maps), so close_02 ran
  with `--grasps v2` (v2 grasps, the legacy band where no v2 map; no legacy grasps). Results (N 4 x 2 at 310000-310007 /
  close_02 310008-310015; off -> on):

  | alias | executed opening | attempts | written | demos ok | GPU-h / success | brace candidates passed / failed |
  |---|---|---|---|---|---|---|
  | touching_01 | unchanged (wide stop tried at 0.0315 / 0.0215 only; records closed / blade) | 47 / 47 | 8 / 8 | 7 / 8 | 0.166 / 0.142 | identical |
  | touching_02 | 0.0315 -> 0.05 (spread 0.09 -> 0.127, every candidate and record) | 141 / 141 | 2 / 2 | 2 / 2 | 0.696 / 0.769 | 14 / 48 both |
  | touching_04 | unchanged (gkpmii's face takes <= 0.09) | see DECISIONS | 0 / 0 (place stage) | - | - | identical |
  | close_02 [v2] | 0.04 -> 0.05 (every candidate and record) | 10 / 11 | 8 / 8 | 1 / 8 | 0.94 / - | 77 / 446 vs 119 / 760 |

  touching_01 / 04 sample identically in both arms (same rejections, same counters): the switch does not act there. touching_02 (blocked
  by place depth, not judged) changed only the opening: the same attempts, rejections and brace pass / fail counts - the wider hand
  collided nowhere new - and 2 / 2 demos in both arms. close_02: every failure in both arms is the same hard QA during the push (the
  carton displaced 1.0-1.2 cm, tilt 0.003 rad, ~step 550) - present before (i7_artic3 3 / 4 object_tilt at 0.04); 1 / 8 vs 0 / 8 is
  within N 4 noise and has no cause in the opening; on's 2 program_check_failed are the push arm's tip_push waypoint, not the brace hand.
- **Default ON** (separate commit): `tabletop.BRACE_FULL_OPEN_DEFAULT = "1"`; `ISPATIALGYM_BRACE_FULL_OPEN=0` = the openings of before.
  With it on, the touch_brace consumer reads only maps built on: a root needs the on maps of ntbrtz / dkhgxn / gkpmii
  (`$OFH/contactbrace_20261108/libs_on/maps`, staged in `$OFH/v2maps_icb_on`). Not exercised by any alias: the pack brace (no block_4
  pack-tail alias) and the carton-v4 wide stop (every alias `blade_down`).

## touching_03: the base at the manipulation stance, split-off pieces, the narrow sphere buffer (branch `i7-t03`, 2026-11-08)

Owner request 2026-11-08 04:45 (`tmp/outputs/prompt_t03_base_nudge.md`, DECISIONS "i7-t03"); all three behind DEFAULT-OFF switches until the
owner decides (any default change is the owner's). Record `$OFH/t03_20261108` (per-attempt parser `parse_t03.py`, `tab_lat.py`).

**Diagnosis (logs of fix1_t03nr, fix2_t03init, ft_sph, ft_geo, i7_lean3).** Every written touching_03 demo (7, all legacy) had the book
0.185-0.200 m LEFT of the base (left arm, stand-off 0.55, trunk (0.25, 0), 0.687 m forward). The far-branch transit rejects (2.2-3.9 rad)
happen only on the right-board layouts (never a demo, fix2 included); on bay -0.37 left the grasp passes at start laterals -0.465 / -0.415 /
-0.315 and almost never at -0.265 .. -0.065 (2 of 185 draws), so the uniform lateral draw wastes most of the layout. Since the support split,
the attempts that pass the draw die later: the lift with the book attached (the book starts against the side board the split keeps), the
settled-pose approach against a split-kept piece, and the carried drive to the push stance.

**Item 1 - the stance shift (`ISPATIALGYM_SHELF_STANCE_SHIFT=1`, sampler only, `IORSSmall.shelf_fixed_start`).** When the shelf's fixed
start fails the grasp or the transit / back-off check at a manipulation candidate (`shelf_need_of`), the base moves along the edge: first the
COMPUTED stance - the book at `generation.shelf_stance_book_lateral_m` (0.19, `SHELF_STANCE_BOOK_LATERAL_M`, the right arm mirrored) from the
robot's centre line, as `push_stance_for` places the push contact, same stand-off and trunk - then the scan (`shelf_stance_scan_shifts_m`
(0, +-0.05, +-0.10) around it x `tabletop.STAND_OFFS`, the computed lateral first; at most `shelf_stance_scan_max` = 6 scan stances per
attempt, a stance never tried twice). Every stance: free floor, whole-body collision with the working arm and the other arm idle
(`stance_for_arm`), the work camera's Phase-2 rule at the book (`_shelf_camera_rule`), then the same `fixed_start` as the candidate
(travel, `grasp_at_pose` with every grasp and twin, the transit reach <= 1.6 rad from that stance's backed-off pose, the back-off) and
everything downstream. The travel init -> shifted stance is the axis-aligned leg (`core.record_stance_travel`, `NAV.plan_axis_leg`: heading
kept, forward / backward / sideways only, `generation.stance_shift_back_offs_m`). A candidate that passes is never shifted. Kit
`stance_shift.need.<grasp|transit>`, `.computed_tried / _passed`, `.scan_tried / _passed`, `.rejected.<floor|collision|camera|check>`,
`.offset.<m>`; the stance's `extras.stance_shift` (offset, need, reach before / after, near / flipped branch); one INFO line per shift.

**Item 2 - split-off pieces as finger-only contact (`ISPATIALGYM_SPLIT_FINGER_TOUCH` = 1 / shadow / 0).** `IKChecker.contact_rejudge`:
a configuration that collides with the support split's kept pieces (the side board, walls, board above) passes when (a) it is free with them
out of the world, (b) every link but the hand clears at margin 0 (today's rule), (c) the palm / wrist camera / wrist clear at margin 0, and
(d) the fingers clear everything else at margin 0 and their EXACT hulls touch the pieces at clearance >= 0 (`HandChecker.clearance`
against `ConvexPieces` of the kept pieces) - never judged on the spheres. The policed plane (2 mm) and FingerTouch are unchanged. Call sites:
`IKChecker.track_segment` (every approach / chain configuration and its end), `_witness_ok` (exact-IK witnesses, cache verification,
`collision_free_at`; the active ignore sets tracked by `ignoring` while a rule is on), `exact` (a pose the solver misses is solved once
more with those pieces out and kept only when `contact_rejudge` passes: `_solve_rejudged`) - so the grasp chains, approach, settle
validation, lift / place / program check all see it. Map builder: `reach_maps.compute_map`'s interior sphere stage skips the finger spheres
against the normal obstacles (their hulls stay judged by `check_with_obstacles`, >= 0); a map built so carries `split_finger_touch` in its
key. Grasp validators (`candidates.py`, `access_region.py`) already judge the fingers by hulls >= 0 against obstacles: unchanged. Kit
`support_split.finger_touch_allowed` / `_shadow_pass` / `_failed.<why>`.

**Item 3 - the narrow sphere buffer (`ISPATIALGYM_WITNESS_HULL_FALLBACK` = 1 / shadow / 0).** i7-plateqtf2's side_9 rule (3a2ce4ec,
cherry-picked as c75a1096: `MODEL_SPHERE_BUFFER_M`, `support_hull_fallback_min_m` 0.5 mm, `IKChecker._hull_fallback`) reused: (a) in
`_hull_fallback`, a hand that fails only on the policed support's split-kept pieces is judged by the same hulls against them; (b) in
`contact_rejudge`, the hand's rigid links failing narrowly (clear at margin 0 less the 2 mm buffer) against the split-kept pieces or a
registered support standing whole in the world (the pre-grasp) pass when their hulls clear by >= 0.5 mm; arm / torso / base keep their
spheres. Kit `witness.hull_fallback_passed` / `_shadow_pass` / `_failed.<why>`. With either rule on (not shadow) the exact-IK cache scope
carries `contact_rejudge`.

Tests: `tests/test_t03_stance_shift.py`, `tests/test_t03_contact_rules.py`.

Code: c75a1096 (cherry-pick of i7-plateqtf2 3a2ce4ec), 5299165c (item 1), 6440b551 (items 2 / 3), 982cdcdf (cherry-pick of i7-plateqtf2
1f368d64, `contact_policy.hull_fallback`, with item 3a routed through it; NOT 6a6731ca: the side_9 switch stays default 0 here), 6f744f76
(test). Suite vs base 7fbe45cc: the same 23 failed + 3 errors.

**Results (A/B trunk_6440b551, legacy, N 4, instances 300100-300103; v2only trunk_6f744f76 on the staged lean root).** a (all off) 320 att /
0 written; v (v2only, 1 + 2 + 3) 320 / 0: the computed stance passed 169 / 426 and the scan 156 / 794 (offsets -0.40 / -0.35 / -0.15 m;
reach after 1.26-1.40 rad, near), but every attempt it let through then failed - 52 "partner not in the init image" (the INIT view: the
start lateral stays 0.35-0.40 m from the bay; the shift moves only the manipulation stance) and 42 at the goal check (push drive 21, lift
with the book attached 20, place 1; i7_lean3 without the shift: 43). Items 2 / 3 are inert under v2only for the bookcase: its interior
registration's split plan has CLIPPED straddlers, which `contact_rejudge` does not judge. The shadow count over the 13 other split aliases
(i7t03_cnt4, N 4, 128 attempts): 0 passes for either rule. Legacy arms: a 320 / 0 (1.79 GPU-h); b (item 1) 320 / 0 (4.00 GPU-h; shift
passed 31 computed + 37 scan; 11 then 'partner not in the init image'); c (1 + 2 + 3) 320 / 0 (4.26 GPU-h; witness.hull_fallback_passed 1546,
support_split.finger_touch_allowed 0). No arm wrote a demo; the base shift fixes the manipulation stance, not the alias.

**Open (owner).** The start lateral (a bay-centred init draw, the separate switch item 1 names); the attached book against the side board it
leans on (support split) at the lift; the carried drive to the push stance; the computed stance colliding at the candidate's own trunk
(`stance_shift.rejected.collision`); clipped straddlers in `contact_rejudge`; any default change.
### i7-attachfix (2026-11-08): the attachment aliases diagnosed with numbers (owner direction 2026-11-08 02:20)

Branch i7-attachfix from of-v2maps 386ab2b4; record `$OFH/attachfix_20261108`. Run diagnosed: i7_attswept (trunk_54439eee, 62 att / 16
written / 6 ok).

**One class: the held object / partner not where the plan thinks (items 2b, 3, 4).** Every failing instance of that class was the
instance AFTER a joined episode on the same reused collector world (`run_isg --queue`): the live start pose vs the TRO
(meta.start_state vs tro_state) - attachment_01 300114 lid 103.8 mm + teapot 81.6 mm (after 300113 joined), 300115 lid 76.6 mm
(after 300112), attachment_04 300116 lid 549 mm (after 300114), attachment_03 300020 wine bottle 40 mm (after 300019); every other
of the 16 demos 0.0 mm. The off lid sat exactly at the PREBUILD instance's pose (300108 / 300110: the world's `scene_state0`).
`Loader.attach` released the grasps / hinges / planner attachments but not the BEHAVIOR attachment joint the attach rule made;
`scene.load_state(scene_state0)` then detached it inside the load (`AttachedTo._load_state -> _detach -> og.sim.update_handles()`)
and the TRO pose of the pair did not stick. Consequences: (3) the oracle's grasp target = live lid o eef_in_object, the sidecar's
witness = the TRO lid: the "77.1 / 576 mm witness residual" plan_failed is that offset (the witness itself reaches its pose: the hand
at the witness ended 2.8 mm from the recorded pre-grasp); (4) 300114 grasped the displaced lid correctly (in-hand 0.7 mm / 0.002 rad
vs eef_in_object, rigid through the carry), the arm reached the recorded place within 4.8 mm - over where the TRO puts the teapot,
82 mm from where it stood; the release check was right; (2b) 300020: in-hand 0.8-1.9 mm / 0.06 rad through the carry (the AG
attachment rigid, no slip, the executed grasp on the recorded one), the bottle 40 mm off -> the seat pose inside the bottle.
Also: meta.attach_joins listed the PREVIOUS episodes' joins (PRED.JOINS is per process).
Fix (d9de0ff1, `oracle/run_isg.py`, the loader - not an oracle program): `detach_scene_attachments` before the state load; after the
restore every TRO object is compared with its TRO root pose (`tro_pose_mismatches`, 5 mm / 0.05 rad), re-applied once, else
LoadError; `meta.attach_joins` = this episode's joins only. First validation (i7af_v, trunk_b4d8f3d4): the check still met the pair off
after a joined episode (attachment_01: lid 113 mm / 2.11 rad + teapot 240 mm / 3.06 rad; lid 95 mm / 0.64 rad) and re-applied it (the
episodes ran from the TRO). Remaining cause: the joined pair stays collision-FILTERED in USD across the state load and its deferred
un-filter (`predicates.set_pair_collision_filter`: dump, stop, edit, play, load) ran inside the next restore - the vectorized loader's
cleanup already calls `predicates.reset_pair_records`, the single-world `attach` did not. 74fef537: `attach` resets the pair records
(un-filter) before the state load.

**Recorded IK witnesses (item 3).** Independent numpy FK of the planner's holonomic URDF over every written sidecar under `$OFH/out`
(2463 sidecars, 24399 witness targets; 120 q-only regrasp targets skipped): FK vs pose p50 0.43 / p99 1.60 / max 3.46 mm, rotation
max 0.010 rad; 0 above 5 mm or 0.05 rad; 18 instances with a target 2-3.5 mm (push / base_push pre-poses, the sampler's looser pre
limit); the recorded fk_residual_m agrees with this FK to 0.024 mm. Guard (d0a96dda): `sampling/v2/witness_check.py` runs the same FK
on every witness before an instance is written (`witness_unreachable` > 5 mm / 0.05 rad; ISPATIALGYM_WITNESS_WRITE_CHECK=0 off).

**Cap xxqyrt (item 1).** Rest r0 = canonical, open side DOWN (confirmed: top disc piece z 1.20-3.47 mm, seven wall pieces radius
22.6-26.2 mm (3 mm wall) z -7.76..+2.2 mm, nothing on the axis below z 1.2: the cavity opens onto the table); 52.3 mm x 11.2 mm, 16 g;
gripper max opening 96.9 mm. Pieces: the hulls of 20-70 deg wall arcs bridge INWARD (chords into the cavity), none bulges outward.
Causes (generation): (a) the Isaac sampler asks `multiple_hits=True` and keeps the furthest hit within the aperture, but embree here
returns only the FIRST hit, so every antipodal pair spans one 3 mm wall, the 2.3 mm disc or a piece-cut face (local reproduction:
4000 samples, 100 % of the pad centres inside a collision piece, widths p50 2.3 mm, max 18.1 mm = piece 1's cut faces; the gate dump's
top width drops 0.09-12.7 mm): no pinch across the diameter was ever proposed, the 33 "selected" were tangential piece-cut pinches
(top_0 = width 18.07 mm across piece 1's cut faces) - hence finger_blade_in_object by 1.5-2.7 mm and the single physics-valid
top_0 that does not hold in the demo; (b) the width rays started at the pad centre (in the cavity: the inner wall); (c) the top support
retreat stops the fingers 1 mm below the table, the attachment rule then needs them 2 mm above the rim (face_gap): every retreated
grasp of the 11 mm cap dropped at face_gap -1.0 mm (27 / 27). Side pool: every side grasp loses its width after the support lift
(lift 30-106 mm: the wrist camera / palm sink into the table beside an 11 mm object; the pinch line ends 2.6-27.8 mm above the cap
top) - geometry, unchanged under the pre-shape sweep (the sweep never runs: width_lost first). The 4 pick failures: failed_grasp "not
in the right hand after closing" twice each (AG never formed), fingers closed to 0, executed eef-in-cap 4.3-7.3 mm / 0.02-0.12 rad off
the recorded (the cap pushed 2.9-6.1 mm by the closing fingers of the piece-cut pinch).
Fix (b4d8f3d4, opt-in, identity when on): `CandidateConfig.outer_pinch` (every ray hit - repeated first-hit casts; width / antipodal /
the blade check's pinched points measured from outside along the closing axis) + `attach_face_retreat` (an attach top grasp retreats
until the fingers clear the rule's face gap; the rule unchanged). Local candidates (default density): both on 16 selected / 16
geometry-ok, diametric pinches 50.9-51.0 mm, tilts 2-31 deg; outer_pinch alone 1264 -> 0 at face_gap; face retreat alone 0 at
object clearance; off 0.
Rebuild (trunk_b4d8f3d4, `$OFH/attachfix_20261108/cap`, manifest cand_overrides {outer_pinch, attach_face_retreat, gate_dump}; job
18372013 glib + maps 0.8126 right + left; job 18372014 attach maps box + swept right, both ok): 10 representatives -> 6 PHYSICS-VALID
(top_1..top_6; top_0 object_pressed_into_table) - was 1 (top_0, the piece-cut pinch). Staged root `$OFH/attachfix_20261108/v2maps_af`
(cp -a v2maps_int1 + this entry and its maps; the int1 cap entry -> `_superseded_attachfix`; legacy_keys cap__xxqyrt; links.json).
Next blocker (i7af_v, trunk_b4d8f3d4: 300026 / 300030 QA pause in the grasp, fingers closed to 0, the cap pushed 5 mm): the cap
aliases' own custom `generation.grasp_height_extra_m` 5 mm (isg_customs, only the four cap aliases; the knob lift made for the legacy
grasps) lifted every written grasp by 2.5 mm (clipped: "fingertips 3 mm below the male's top") - the validated pinch's contact z 0 mm
-> 2.1-2.5 mm (rim edge), tips 0.5 mm above the cap's bottom. d4f1bdcd (`attachment._apply_grasp_height_extra`): that alias lift is
not applied to a v2 library grasp (validated at its own height); the support / partner floors still apply (cap: 1.4 mm finger
envelope). Lid / cork aliases have no such custom (their lifts are floors) - unchanged.

**Cork ncxgpe (item 2).** (a) 24 of 41 attempts rejected for the rest (rest_unstable 20 + unstable_init 4), mean 189.5 s each
(4549 s, 71 % of the attempt time; the ~180 s sample stage runs before the settle / nudge, attachment.py after_settle :1656-1698):
the entry's only rest r0 = standing on the TAPERED male end (ring radius 6.1-6.5 mm under a CoM ~18 mm up; tipping margin 0.33 vs 0.48
on the wide end); the 2 deg nudge tilts it 4.7-25 deg (12) or tips it over 61-93 deg (8). r0 was called stable from one 36-step settle
(no perturbation; final angular velocity 0.148 rad/s; no settle-first survey, so no other rest exists). The fallback
`_rest_flip_forced` (attachment.py:1694) is never read again. NOT FIXED (owner decision): a nudge test in the rest survey is a shared
validation gate, and a stable rest of this cork that keeps the male end down probably does not exist (the wide end down needs a 180 deg
in-hand turn, lying needs a 90 deg reorient). (b) = the restore bug above. (c) the left-finger touches (300021 failed protected_contact
at step 989, 300019 same at 658, retried ok) are the BRACING left arm: the brace's touch-stop pushed the bottle 4.6-5.1 mm / 2.8-3.4 deg,
the 4 mm back-off left the closed finger where the bottle rocked back upright within 3 steps of the carry; `_check_protected`
(qa.py:648-666) ignores the brace-allowed pairs. Neither (a) nor (b) explains it (bottle 0.0 mm off its TRO in 300019 / 300021).
NOT FIXED (an oracle program change): wait for the braced object to stop rocking before backing off / back off by the measured push +
4 mm, and / or exempt brace-allowed pairs in the run-time protected check.

**Validation (--grasps v2only N 4).** attachment_01 / 03 / 04 from trunk_74fef537 on v2maps_int1 (tag i7af_v2), attachment_02 from
trunk_d4f1bdcd on `$OFH/attachfix_20261108/v2maps_af` (tag i7af_v3): 01 att 6 / wr 4 / ok 3 (one place plan_failed: the pre-place and
place witnesses 3.00 rad apart - another class), 02 att 7 / wr 4 / ok 3 (0.41 GPU-h = 0.136 / success; one symbolic_only; the first cap
successes: hover 13.2 mm, touched, drop <= 0.3 mm, joined), 03 3 written / 3 ok (4th still sampling at the report: 14 target_placement +
5 rest rejections), 04 att 13 / wr 4 / ok 4. Restore on the final code: 17 demos, 6 after a joined episode (1 joint detached + 1 pair
un-filtered each), 0 TRO mismatches, 0 re-applies. Earlier snapshots: i7af_v (b4d8f3d4: the check re-applied 2 of 2 off pairs; 01 4 / 4
ok; 02 cap 2 / 2 QA pause at the grasp - the alias lift, fixed in d4f1bdcd). Suite (8 matched shards, base 386ab2b4 vs tip): 23 failed +
3 errors on both, the same tests; test_attachfix 9 pass.

### Owner decisions 2026-11-08 (DECISIONS 02:35): plate qtfzeq follow-ups (i7-plateqtf2)

Work: branch `i7-plateqtf2` (from the integration tip of-v2maps 386ab2b4). Record: `$OFH/plateqtf2_20261108/`, DECISIONS.md "i7-plateqtf2" lines.

The owner's decisions (2026-11-07, relayed 2026-11-08 02:35):
1. **side_9 / the support policy's palm check.** Keep the padded sphere check as a fast, conservative SCREEN. When it fails NARROWLY (the
   sphere misses by less than its buffer; here 0.9 of 2 mm), judge with the palm's EXACT hull against the support, requiring a small real
   clearance (propose the value). Only non-finger links; the 2 mm finger allowance (FingerTouch) unchanged. First find where the 2 mm buffer
   is set and whether it is a deliberate non-finger margin or a reused constant. Shared gate APPROVED: behind a switch, count before adopting
   (campaign library + the aliases' sampling logs; who gains), then the default in a separate commit; kit `support_policy.hull_fallback_passed`;
   unit tests (narrow miss + clear hull passes, real hull contact fails, clear sphere pass unchanged); validate under_small_03 + the aliases the
   count names.
2. **Overhang fallback** (APPROVED incl. the oracle program change). When a chosen SIDE grasp fails in sampling for a reason the sampler cannot
   fix (count the reasons), switch to the overhang approach (push over the support edge, then pick) as pick_overhang_push does in IORSLarge;
   add the push step to under_small_03's oracle program and generalize to every alias whose pick can fall back this way (list them first);
   fallback only; kit `overhang_fallback.*`; the same exact checks as IORSLarge (reuse pick_overhang_push / v2/overhang.py); unit tests;
   validate under_small_03 N 4 + the other alias with the most side-grasp failures.
3. **Pre-place framing** (camera-frame rule change APPROVED): the draw's pre-place check matches the program check (fingertip point + 2 cm
   slack) instead of the hand box; QA unchanged; list the call sites; count per alias the rejections removed; unit test; validate under_small_03.
4. **The expanding tilt / yaw map** (`ISPATIALGYM_FAST_MAP_TILT_INHERIT`): CONFIRMED as the default for all aliases (no change).
5. **The merged map overlay** (`ISPATIALGYM_MAP_OVERLAY`): jobs START reading it - first confirm the frozen map is never written, the reader
   applies a PINNED version (job.env + the map identity), the in-memory map cache keys on the overlay version; fix what is not so; the
   offline merge after each batch makes a new version, jobs pin the version they read; on by default (separate commit).
Validation: unit tests; under_small_03 `--grasps v2only` N 4 ONCE with 1 + 2 + 3 + 5 (each item's counters) + the extra aliases of items 1 / 2.

**1. The hull fallback** (`ik.IKChecker._hull_fallback`, switch `ISPATIALGYM_SUPPORT_HULL_FALLBACK` = 1 | shadow | 0; 3a2ce4ec, default ON
6a6731ca). Where the 2 mm comes from: the robot model's cuRobo `collision_sphere_buffer` (`r1pro_model.build_curobo_robot_dict(...,
collision_sphere_buffer=0.002)`, the value of OmniGibson's shipped `r1pro_description_curobo_default.yaml`, a49d8f54 "Instance sampler v1.0.0")
- ONE global padding of every sphere of every link in every cuRobo query (planner, IK, the sampler's checks); the contact policy reuses it
(its own sphere margin `sphere_support_margin_m` is 0; the module doc's rule 3 only names "the model's 2 mm sphere buffer"): a reused
constant, not a deliberate non-finger margin. Now `contact_policy.MODEL_SPHERE_BUFFER_M`. The rule: the sphere stage (non-finger links vs
the world minus the approached objects, margin 0) stays first; when it fails, the fallback applies only if (a) every link but the hand's
rigid links (palm `{arm}_gripper_link`, wrist camera `{arm}_realsense_link`, wrist `{arm}_arm_link7` - the links with exact hulls) is
clear at the margin, (b) the hand's rigid links are clear at the margin less the buffer (each sphere < 2 mm deep), (c) they are clear at
the full margin with the policed supports out of the world (the touch is the support's, not another obstacle's), (d) no policed support is
a floor; then their exact hulls (`HandChecker.clearance`: vertices + face samples) must clear every policed support's plane by
**0.5 mm** (`HULL_FALLBACK_MIN_CLEARANCE_M`, `ISPATIALGYM_SUPPORT_HULL_FALLBACK_MIN_M`). Why 0.5 mm: the hull-vs-plane clearance is exact
(a convex hull's minimum over a plane is at a vertex), so the margin only has to cover the support's own unevenness about its measured
plane (breakfast_table rhjoby: 8 pieces within 0.3 mm) - 0.5 mm covers it with margin and still admits side_9 (1.0 mm nominal); 1.0 mm
would reject side_9 itself (its nominal hull is 0.99-1.03 mm). The rest of the policy (the hull stage incl. the fingers' 2 mm, rule 2)
follows unchanged. Kit `support_policy.hull_fallback_passed` / `.hull_fallback_shadow_pass` / `.hull_fallback_failed.<why>` (floor,
no_hand_link, non_hand_link, deeper_than_buffer, other_obstacle, no_plane, hull_clearance). Tests `test_support_hull_fallback.py` (8).
COUNT before adopting (`$OFH/plateqtf2_20261108/`): library (`hull_fallback_count.py` over the 102 int1 entries' 260 valid grasps at their
nominal pose on the rest): 245 clear, **7 narrow with hull >= 0.5 mm** (plate qtfzeq side_9 -0.85 / +1.03 mm, lip_shallow_4 -0.33 / +0.98,
lip_shallow_5 -0.23 / +1.02; cork ncxgpe side_0 -1.65 / +5.22; brussels sprouts mbkrxe side_10 -0.61 / +1.77; webcam fabztm side_2 -0.73 /
+3.05; hardback lean_right side_3 -0.58 / +1.00), 8 deeper than the buffer (hard candy vwegtt side_2 -2.9 mm; the 6 hardback overhang-rest
grasps and the lean_left one, whose support is a slab box, not a plane); at 1.0 mm one of the 7 drops. Sampling logs (`count_sphere_logs.py`,
every v2 run under `$OFH/out`): sphere-stage rejections under_small_03 1434 (side_9 1133 pose + 274 approach tail; lip_shallow 27),
attachment_03 129 (cork side_0), inside_01 100 (brussels sprouts side_8 96 - an earlier library -, side_10 4), inside_02 47 (hard candy
side_2: deeper, no gain), touching_04 9, nextto_02 8, disembedding_05 7. Who gains: under_small_03, attachment_03, inside_01 (and the
entries without logged rejections: webcam fabztm, hardback lean_right).

**2. The overhang fallback** (`v2/overhang_fallback.py`, 1344b250; switch `ISPATIALGYM_OVERHANG_FALLBACK`, default on). Side-grasp failure
reasons per alias (`count_side_reasons.py`, v2 runs): the support policy (the sampler cannot fix it: the grasp pose is fixed relative to the
object on its support) under_small_03 1409, disembedding_05 292 (palm_contact 285), attachment_03 129, inside_01 100, inside_02 47,
touching_04 9, nextto_02 8; the rest are IK / joint limit / tracking / collisions with other objects (another stance or position can fix
them). Which picks can fall back this way (`side_overhang_list.py`: entries whose SIDE grasps' closing sweep reaches below the support top
at the nominal pose, i.e. a flat object whose lower finger needs free space): plate qtfzeq side_9 (under_small_03, `program_iors_small`),
brussels sprouts mbkrxe side_10 (inside_01, `program_iors_small` inside), hard candy vwegtt side_2 (inside_02, idem), cork ncxgpe side_0
(attachment_03: the attachment program / sampler - not wired: the attachment code is i7-attachfix's; owner), hardback bqvvvx overhang rest
(inside_large_01: IORSLarge, already its primary path). Mechanism: in `tabletop._phase3_draw`, when a drawn pair's exact IK fails, the
grasp is library mode `side` and the reason is the support policy (pose or approach tail: `overhang_fallback.reason_class`), IORSSmall (not
touching: its program picks without the push) calls `overhang_fallback.try_fallback`: IORSLarge's `_overhang_pick` and its helpers
(`_overhang_geometry` / `_edges` / `_grasp` / `_push_chain` / `_push_branch` / `_touch_ok` / `_drifted` / `_settle`, `overhang_records`)
bound UNCHANGED on an adapter of the small sampler (`SmallOverhangPick`: per-role target / hull / rest, the support top as `surface`, the
drawn pose as `pose_on`) - the same exact checks; options `generation.pick_overhang_push` or the defaults (mode fallback, no second grasp
stance). A failed pick is memoised (stance, trunk, grasp, yaw, cell). The record is written with the grasp (`contacts.push_pick` +
`grasps[0].pick_overhang`, dropped with a rejected candidate); `after_settle` keeps the overhang grasp (no re-derivation / re-selection at
the settled init pose). Oracle program (approved): `program_iors_small` picks through `_pick_actions` - push -> retreat -> grasp -> lift
when `contacts.push_pick` is recorded, unchanged otherwise. Kit `overhang_fallback.trigger.<class>` / `.tried` / `.found` / `.failed` /
`.memo_hit` + IORSLarge's `pick_overhang.*`. Tests `test_overhang_fallback.py` (6, on test_iors_large_overhang's stubs).

**3. Pre-place framing** (7fce56ee; `ISPATIALGYM_PREPLACE_FRAME_RULE` = fingertip (default) | box). Call sites where a draw checked the
hand box at the pre-place: `iors_small._segment_draw` (the place / pre-place screen before the settle), `IORSSmall.require_in_frame` (the
grasp check's `{"pre-place", "place"}`), the FRAME_RANK score in `_segment_draw` (`frame_score`). They now judge the PRE-PLACE by
`program_check.fingertip_frame_margin` (the criterion `fingertips_in_frame` applies, factored out: the fingertip point inside QA's 5 % plus
2 cm slack at its depth); the PLACE (a contact pose, `CONTACT_POSE_KEYS`) keeps the eef box, as the program check does. No other draw checks
a pre-place with the box (block_design's `_chain_in_frame` checks the eef + fingertip points; iors_large / inside / attachment check
contact poses). Not changed (camera-frame changes beyond item 3 -> owner): the pre-GRASP box in `require_in_frame`, touching's push / brace
pre-contact boxes, the carry waypoint box in `core.py`. Kit `preplace_frame.checked` / `.rejected` / `.removed`. Tests
`test_preplace_frame.py` (3). Count (`count_preplace.py`, every run under `$OFH/out`): pre-place box rejections per alias inside_04 2635,
inside_03 783, ontop_small_02 331, ontop_small_04 318, under_small_04 173, under_small_03 165, ontop_small_01 80, under_small_01 28,
inside_02 27, under_small_02 18, nextto_03 11, inside_01 8, ontop_small_03 6, nextto_02 / 01 / 04 1-2; only under_small_03 logged the box
corners (FRAME_DIAG, 19 pre-place rejections): 13 / 19 pass the fingertip rule (fingertip reconstructed from the corners' projections).

**5. The merged map overlay, pinned** (4f374e8c mechanism, 14634552 default). Confirmed: the frozen map is never written (map_tilt writes
only its shards and merged version files; `test_overlay_never_touches_the_frozen_map`). Not so before, fixed: (a) the reader took the
LATEST version at first use (per process; another process of the same job or a merge mid-job could read another) - now every merge also
writes a RELEASE `<root>/releases/r<M>.json` = {map key: version}, `submit_fast.py --map-overlay <root>` pins the root's latest release
into job.env (`ISPATIALGYM_MAP_OVERLAY` + `ISPATIALGYM_MAP_OVERLAY_RELEASE`; `job_fast.sh` pins at the job's start when job.env has none),
the reader applies exactly that release's versions (a key it does not name: ov0; an unreadable pinned release fails); unpinned reads are
logged and counted (`map_overlay.unpinned`); the version applied stays in the view identity (`inherit_k2+ov<N>`, the start-cell caches) and
in `meta.map_overlay` (kit `map_overlay.read_v<N>`); (b) the process's overlays were keyed by map key only and a view's inherited-column
cache by the overlay's own-event counter - now keyed by (map key, root, release, version) and by the overlay object + version, so a process
never mixes an overlaid and a plain view. Default ON: `--map-overlay` defaults to `$OFH/map_overlay` (seeded with the plateqtf overlay v1,
release r1 = right ce1af987a8df9d69 v1, left b2f36adb26c194ad v1); `--map-overlay off` = none. After each batch: `python -m
ispatialgym.sampling.v2.map_tilt merge --root $OFH/map_overlay $OFH/out/<tag>/*/g*/map_overlay/*/*.jsonl` (new versions + a new release).
Tests: `test_map_tilt.py` +3 (pinned release vs latest, never mixed views, the view's pin record).

**Reusable hull check** (1f368d64, for i7-t03: the owner applies the item-1 rule to touching_03 too): `contact_policy.narrow_sphere_failure`
/ `hull_clearance_min` / `hull_fallback` - input the sphere query and the links that failed, the hand links with hulls, the HandChecker, the
eef pose, the support / obstacle geometry (planes or boxes, e.g. split-off support pieces), the threshold (`HULL_FALLBACK_MIN_CLEARANCE_M`
0.5 mm); `IKChecker._hull_fallback` calls it (behaviour unchanged). Cherry-pick 3a2ce4ec (`MODEL_SPHERE_BUFFER_M`) + 1f368d64.

**Validation** (trunk_14634552, v2-root `$OFH/v2maps_int1`, `--grasps v2only` N 4, overlay `$OFH/map_overlay` release r1; counters summed
over the kit blocks by `$OFH/plateqtf2_20261108/tab_pq2.py`, the same script on i7_pq_merged for "before"):
- under_small_03 `i7pq2_us03` (18372123; items 1+2+3+5): 36 attempts / 4 written / demos 3 of 4 ok, 1.36 GPU-h = 0.455 per success (i7_pq_merged
  32 / 3 / 3 of 3, 0.448). Written grasps lip_shallow_6~r180 x 3, top_4. Item 1: `hull_fallback_passed` 16, failed deeper_than_buffer 31 /
  hull_clearance 9 / non_hand_link 1; side_9 drawn 10 times, never rejected by the support policy (it failed on IK / joint limits: fixable),
  never chosen. Item 2: no trigger (no side grasp failed on the policy). Item 3: `preplace_frame.checked` 62 / `.removed` 24 / `.rejected` 2;
  gripper_not_in_frame log lines at the pre-place 12 -> 1, at the place 4 -> 4. The failed demo (300150): `gripper_out_of_fov` in the place's
  carry (u 122, v 58, depth 0.50 m) after the oracle's PROTECT via-above lifted the carry over the mat; that instance's pre-place passed only
  under the new rule (`removed` 2 in its written attempt). Item 5: `map_overlay.read_v1`; empty_f_on_inherited 133 -> 95, reach_fail 132 -> 95,
  cells_invalidated 32 -> 13, cells_confirmed 57 -> 38, map_yaw.pairs_confirmed 82 -> 73.
- attachment_03 `i7pq2_at03` (18372125; item 1): 39 / 4 / 4 of 4 ok, 0.411 GPU-h per success; `hull_fallback_passed` 9, no sphere-stage
  rejection (historically 129, cork side_0); written side_0 x 1, top_15~r180 x 3.
- inside_01 `i7pq2_in01` (18372124; items 1+2): 15 / 4 / 4 of 4 ok, 0.101 GPU-h per success; no support-policy rejection of a side grasp (the
  int1 entry's side grasps failed on IK / tracking), `hull_fallback_failed.deeper_than_buffer` 1; the overhang fallback did not trigger.
- Overlay merged after the batch: `$OFH/map_overlay` release r2 (ce1af987a8df9d69 v2 212 entries / 29 invalidated / 179 confirmed,
  b2f36adb26c194ad v2 92 / 15 / 71, + 3 new map keys of attachment_03 / inside_01).

### i7-contactfu (owner decisions 2026-11-08 01:45, part A; branch `i7-contactfu` from 521dc344)

- **A1 touching_02 push stance** (`iors_small.push_stance_candidates`, sampler-only): the push stance moves along the edge AND toward /
  away from it - stand-offs `PUSH_STANCE_STAND_OFFS_M` (0.15-0.55) plus the manipulation stance's, chosen from the target's `touch_push`
  map at its place pose (brace-map fits first, then the stand-off nearest the manipulation stance's), limited by the whole-body collision
  check (the base's distance to the table); at most 8 get the exact push chain. The partner draw (`place_partner`, v2only) tests the maps
  over those push stances; a partner whose push fails the map from every push stance at all 8 positions rejects `contact_unreachable` at
  the draw (kit `touch_push.partner_rejected`) instead of ~30 s later. Kit `touch_push.stand_off_moved` / `.gate_at_push_stance`.
- **A2 touching_02 tipping**: a scene load of the demo (fix2 300008) found NO physics material on the bar, the boxes or the grippers and
  no scene default (PhysX default 0.5 / 0.5 / 0, average) - the same as the validator; both boxes 0.5 kg (iucshm s0.8 density 342 kg/m3),
  same COM, solver iterations, sleep settings. The difference was the validator's partner stand-in: built `kinematic_only=True` without a
  fixed base, which OmniGibson ignores (`compute_kinematic_only`: no fixed base -> dynamic) - a 2.4 kg dynamic box that gave way under the
  tilting target (the demo's partner is held by the left brace: it moved 5 mm). Fix: `STOP_STANDIN_KINEMATIC` (fixed_base + kinematic).
  Replay at the default friction 0.5: VALID (tilt 0.000, orientation err 0.0025, touch gap 2.0 mm; was 0.463 rad). The brace-load pusher is
  built the same way (dynamic) - not changed (it would move brace verdicts; owner question).
- **A3 hand-overlap screen**: enforced; `ISPATIALGYM_HAND_OVERLAP_RANK` and `validate_contacts --screen-as-rank` stay default off; the
  stratified validation marks screened entries `screened` and never counts them.
- **A4 axial_02 repeats** (analysis, `$OFH/contactfu_20261108/a4/`): `instance_separation.measured_repeat` (base <= 0.25 m / 15 deg + the
  task tolerances: goal and start orientation within 0.1 rad) against 2 eval instances (start yaw delta +24.3 / +34.4 deg, the same pinned
  init base) -> a draft repeats iff delta in [18.6, 40.1] deg. The jhaltj s1x1x2.5 `push_blade` map admits at the drawn cells (x 0.79-0.91)
  only `+end-spin_0@t25` (delta 22-34 deg: inside the window) - the map gate narrows the start onto the eval instances (6 classes, turns 25 /
  45 only, 30 / 48 valid). Proposals: widen the library (turns 35 / 55 / 65, the -end 45 class, 16 yaw slots), order candidate stances by
  map coverage; drawing delta outside the eval windows touches the repeat path (owner).
- **A5 axial brace pairing** (`alignment_axial._scan_candidates`, v2only, map lookups only): per candidate stance (the first 6), the placement
  draws until one has a PAIR - one pushed end whose `push_blade` class (init root) and `axial_brace` class (goal root) both fit, the same
  end and spin in both maps; the paired stances first, else the first push-feasible draw (push-only); the paired end is tried first. Kit
  `axial_brace.paired` / `.push_only`. axial_01-04 have no orientation equivalences: one goal yaw per stance.
- **A6 stratified validation + learned maps**: `validate_contacts --per-group-valid 3 --per-group-max 8` (`GroupQuota`: per (yaw x end x
  spin) group, library order, round-robin; kept verdicts with `--skip-validated`; a fresh run moves unreached old verdicts to
  `validation_prev`); maps from VALID entries only. Learning (`sampling/contact_overlay.py`, the ONE overlay mechanism of
  `v2/map_tilt.py`): the sampler notes the executed library entry per instance (`summary.contact_entries`: lineage key, class, yaw slot),
  `python -m ispatialgym.sampling.contact_overlay harvest <job dirs>` turns each SUCCESSFUL demo into `ok` events
  (`<job>/map_overlay/<lineage>/harvest__<instance>.jsonl`, immutable), `python -m ispatialgym.sampling.v2.map_tilt merge --root R` folds
  them into `R/<lineage>/v<N>.json`, `contact_maps compute --overlay-root R` (job_contacts.sh `GC_MAP_OVERLAY`) rebuilds with the
  confirmed entries joined; the map identity carries `{lineage, version}`; frozen maps are never edited.
- **A6 run** (trunk_1da26c1c; `$OFH/contactfu_20261108/submit_gpu.sh`): touching push FRESH (the stand-in changed) on
  `contact_libs_20261108s`: valid cereal iucshm 12 (groups short 4 / 8: 4 yaw slots with no valid entry), napkins 14 (short 4), spray 22
  (short 1), 43-44 entries tried each, 578-883 s per library; touching brace kept (dkhgxn 40, ntbrtz 32 valid, 3 / group everywhere, 0 s);
  axial (`contact_libs_axial_20261108s`, kept + stratified) push qgvsnb 101 (short 0), injrgc 62 (short 14), tfovwy 86 (short 8), brace
  104 / 97 (short 3) / 120; jhaltj s1x1x2.5 (`_20261108sj`) push 80 (short 6), brace 96. GPU: touching ~0.6 h, axial ~4.8 h (tfovwy push
  6725 s, jhaltj push 5381 s - the slow re-contact pushes); today's capped run: 48 entries / library, 470-3330 s. Maps (valid only) staged
  in `v2maps_i7contactfu_t` (touching) / `_a` (axial). job_contacts.sh: the maps stage now maps the validate stage's partition (the
  library-id partition let shard 1 map the napkins / cereal push libraries before their validation ended; rebuilt).
- **A1 runs** (touching_02 v2only N 4, v2maps_i7contactfu_t; i7_contact2 before: 150 att / 1 written / 1 ok, 107 contact_unreachable,
  1.21 GPU-h): i7_contactfu (trunk_1da26c1c, map-only partner test) 112 att / 3 wr / 3 ok, 53 contact_unreachable, 0.32 GPU-h per
  success; the A1 log showed the map fits at 14-18 of 28 (lateral x stand-off) candidates but only 2-6 pass the collision check (the base
  cannot go closer than ~0.30-0.35 m at the bar), so the partner test now includes it (89dd63aa): i7_contactfu2 82 att / 3 wr / 3 ok
  (brace present 3 / 3), contact_unreachable 11, 45 partner draws rejected at ~1-6 s, 0.20 GPU-h per success; the recorded push stances
  moved 0.10 / 0.25 / 0.10 m toward the bar (stand-offs 0.45 / 0.30 / 0.35).
- **Overlay trial**: `contact_overlay harvest` over the i7_contactfu jobs: 13 successes -> 16 ok events in 5 lineages (1 entry not in its
  map: it would join), merged to `$OFH/contactfu_20261108/overlay_trial/<lineage>/v1.json`; no map rebuilt.
- **A5 run** (i7_contactfu, v2maps_i7contactfu_a): axial_01 11 att / 4 wr / 4 ok, axial_03 6 / 4 / 3, axial_04 17 / 4 / 3 (12 written,
  10 / 12 demos, 0.125 GPU-h per success); kit `axial_brace.paired` 5 / `.push_only` 35 (scan per attempt); brace present 0 / 12 - axial_04
  3 instances had a map pair but the exact two-arm brace failed (side contacts all sweep-skipped, blades IK at the pre / target at goal):
  the brace library is built at one turn (45 deg, `AXIAL_TURN_DEG`) while the sampler's sweep test uses the drawn turn; axial_01 now draws
  both spins (2 / 2; i7_contact2 all +1).

#### Interior PLACE maps: the goal side as stance C checks it (branch i7-interior, 2026-11-07)

Why the interior maps of a STABLE rest were too thin for stance C under `--grasps v2only` (i7_fridge3, i7_sink: 0 attempts, "no stance C"),
measured against the legacy stance C of the same aliases (fix2 / ft_sph / ft2_sph logs and summaries):

- **inside_large_02 (jar exmsmm, fridge_petcxr_0 @ 0.7064, door 2.3).** Legacy stance C: 36 lattice candidates (9 band trunks x 4
  stand-offs x the ONE lateral -0.035: `lattice_values(-half_run, half_run, 0.10)` starts at -half_run), 8-9 survivors in postures
  ready 6-7 / tucked 1-2 / untucked 0-1, at stand-offs 0.35-0.65 (tried C: d 0.35 trunk (0.50, 0.00), d 0.45 trunks (0.50, -0.3) /
  (0.75, 0 / -0.3) / (0.95, 0 / -0.3)); the 3 written demos at d 0.55, lateral -0.035, trunk (0.95, -0.3), idle `ready`, the goal
  0.871 m ahead / -0.025 m lateral of the base. The former map (fridge_maps_20261107c): 29 feasible entries, all at d_e 0.55 and all in
  the first lattice row (x 0.83, 0.05 m behind the board front); 0 at d_e 0.35 / 0.45 / 0.65; legacy (d, trunk) pairs covered 1 / 7; the
  written goal 0.048 m from the nearest feasible cell (the snap admits 0.51 x 0.08 = 0.041) -> stance C `band_cells` 2 / 5, `goal_in_band`
  0. What removed the cells: (1) the WHOLE robot against the boxes in ONE idle posture per trunk (hits obstacle:left_arm_link1 64906 /
  link3 24113 in the right-arm map - the idle arm in the door; untucked collides at every stand-off; no posture is clear at d_e 0.35 /
  0.45 / 0.55 / 0.65 for 18 / 13 / 8 / 2 of the 20 trunks) - stance C judges that static body exactly in the real scene, per candidate, in
  the first free posture; (2) the pre-place 0.10 m back along the gripper's approach (a side pinch's approach runs across the 0.34 m
  compartment into a wall) - the variant's exact
  pre-place is 0.10 m straight out of the opening (`inside_pre_outward`); (3) the place at the REST pose on the board with the hand at its
  pre-shape opening - the exact place is the release 0.03 m up (+ the open-gripper clearance lift), the hand closed on the object; (4) the
  compartment at lateral 0 (centre_y 0.0049) while stance C's only lateral puts it at +0.040 in the base frame; (5) the 0.08 lattice (first
  row 0.041+ m from the first eroded goal cell); (6) found with (1)-(5) removed (85 entries at lateral 0, the WORKING forearm
  right_arm_link4 / 5 in the walls ~2 M times): the side walls boxed from the static FRONT PLANE, measured only at the board's mid depth -
  probe 18371680 (rays from the compartment centre line): for x < 0.13 m the open left door (0.20-0.30 m to the left) and the closed right
  door (0.222-0.245 m to the right), the cabinet walls (0.172-0.179) only from ~0.15 m; the spec's walls are phantom over the first ~0.13 m
  of the 0.231 m setback, exactly where the forearm enters. Not causes: the stand-off reference (stance C's so + the 0.241 AABB overhang = the map's
  d_e from the static front plane; the board 0.2309 behind it in both) and the yaw (both square to the opening).
- **inside_large_03 (plate luhkiz, drop_in_sink_lkklqs_0 basin @ 0.7751, rim 0.11).** Legacy stance C had NO band (NO_BAND_TRUNKS): 80
  candidates (20 trunks x 4 stand-offs x lateral 0), 29 collision-free (untucked), 27 visible, 9 kept; the 24 written demos at d 0.55 /
  0.65, trunks (0.95, 0) / (0.25, -0.6) / (0.0, -0.6) / (0.5, -0.3), the goal 0.16 m behind the raster edge, the release TILTED 30 / 45 deg
  (`release_tilt_deg`), grasp_5 (a legacy rim pinch). The former map: 1 feasible entry per arm, at d_e 0.35 (where the real counter
  blocks the robot: stance C collision_free 0), 0 legacy pairs covered; its rejections are hand_policy (2.9-5.8 k per d_e) and ik_pre: the
  level plate RESTING on the basin floor with the rim-pinching hand at its pre-shape opening inside the 0.11 m basin - the exact place is
  the release lifted (drop 0.02 + the open-gripper clearance) and tilted 30 / 45 deg (grasp side up, the hand above the rim).

The fix (`sampling/interior_place.py`, `reach_maps --kind interior --interior-place <rule>`, job_glib `GB_INTERIOR_PLACE`): an interior
PLACE map of a stable rest judges the place as the exact check does - the WORKING arm and its hand (closed on the object, each grasp's
contact opening) against the compartment's boxes, along the variant's pre-place (`outward` / `approach`, `pre_m`) -> release line and at
the release; the release = the object's bottom `drop_m` + the open-gripper clearance lift (`clearance_m`; every library grasp, or the
grasp's own with `lift_per_grasp`) above the board, level and at each of `tilts_deg`; feasible under ANY self-collision-free idle posture
(the posture's only remaining effect is the IK's self-collision), at ANY lateral of `laterals_m` (default -0.08..0.08 by 0.04: the
compartment centred at `centre_y_m - lateral`), lattice `lattice_m` 0.04 over the compartment's span only, the side walls only along the
board's depth (`walls` "board": `InteriorSpec.boxes(walls=)`, default "front" = every former box and digest unchanged). Exact checks unchanged
(stance C's body check, goal cells, frame, F_goal, the exact place IK with the object attached). Identity: `MapConfig.interior_place` (the
canonical rule) in the config and so the key; maps without it keep their keys (tabletop / floor / generic / lean-interior / the former
interior maps). Lookup: `iors_large_inside.inside_place_rule` (the variant's pre-place, drop, clearance, tilts, lift rule) ->
`tabletop.interior_map_bands(place=)` -> `ReachMap.find(interior_place=)` serves only a place map whose rule covers it
(`interior_place.covers`: equal pre / drop / clearance / lift / body / release / walls, its tilts contain the variant's); a lean goal rest
(inside_large_01) keeps its lean map. The rule fields `body` "robot" / `release` "rest" / `walls` "front" put back the former assumptions (measurements only;
never served). Tests: `test_interior_place.py` (rule, sampler-identical tilt / gripper / lift geometry, the static body left to stance C,
laterals, find, walls), `test_interior_fridge.py` (inside_band reads only the place map of its rule). Commits 2059ee0c, 77166591
(snapshot trunk_77166591; suite = base: 23 failed + 3 errors, the same set).

Rebuilt maps (record `$OFH/interior_place_20261107`, `ip_count.py`; feasible cells per d_e 0.35 / 0.45 / 0.55 / 0.65; "pairs" = the legacy
stance-C (d, trunk) pairs with a feasible cell in the real goal region):

| map | arm | entries | cells per d_e | pairs | legacy written goal to the nearest feasible cell |
|---|---|---|---|---|---|
| fridge, former (fridge_maps_20261107c) | R / L | 4 / 25 | 0/0/1/0, 0/0/2/0 | 1/7 | 0.048 m (snap 0.041) |
| fridge, place rule (fridge_full2) | R / L | 45563 / 10482 | 49/57/48/54, 25/33/30/35 | 7/7 | 0.018 m (snap 0.020) |
| sink, former (i7lean_inside sink_maps) | R / L | 1 / 1 | 1/0/0/0 | 0/5 | - |
| sink, place rule (sink_full2) | R / L | 65849 / 62670 | 28/35/28/35 | 5/5 | 0.00-0.02 m |

One-element ablations (right arm; each puts back one former assumption): sink - release at rest 187 entries (0/5: THE cause), level only
(no tilts) 1606, whole robot 59316, lateral 0 28480, lattice 0.08 14148; fridge (on the walls-from-the-front-plane base, 7007) - lateral 0
85 (3/7), whole robot 1355 (0 cells at 0.35), pre along the approach 6002, release at rest 5085, lattice 0.08 1963; walls along the board
45563.

Validation `--grasps v2only` N 4 (trunk_77166591): i7_sink2 (v2maps_i7sink2): stance C 26 survivors (band_cells 79 / 80, goal_in_band 78,
collision-free 29 = legacy's, visible 26; F_goal at C: 0 level, 8-41 at 30 / 45 deg); 3 attempts, 0 written, 3.0 GPU-h, all time_budget -
every one of 249 C tries failed at B, the PICK of the v2 plate lip_shallow grasps on bar_egwapq_0 (approach_untrackable, support policy on
the approach tail: finger_penetration) - not the goal side (cancelled). i7_fridge4 (v2maps_i7fridge4): stance C 17 survivors (band_cells
55 / 80, goal_in_band 48, collision-free 18: ready 14 / tucked 4; before 0, legacy 8-9); attempts rejected target_placement by the B -> C
carry leg path_collision and the C exact place IK (joint2 near its limit) - the legacy run's own rejection (fix2 27 / 36).

## The start base per attempt (branch `i7-startdraw`, owner decision 2026-11-08 3.5)

Owner 3.5 (verbatim): "templates should never fix the robot's start base. so alignment-axial, articulation open/close small, attachment,
block_design, disembedding, all IORS large (inside, ontop, under, nextto, touching), IORS small (inside, nextto, ontop, touching, under)
should all be drawn per attempt." It replaces the options (a) turn classes / (b) repeat path for axial_02 (57 of 60 attempts rejected
`repeats_eval`: both eval instances sit at the template's start base; the push map admits start deviations of 22-34 deg, inside the repeat
window 18.6-40.1 deg).

### Audit (diagnosis before the change; file:line at of-v2maps cf1f5d2d)

The robot's start base is the `init` stance. `ISPATIALGYM_START_PER_ATTEMPT=1` (train85, core.py:71) is set by `submit_fast.py:228` for
EVERY registry alias and every counted alias of these families (all but nextto_large_02), and `SamplerV2.sample` (core.py:381) then calls
the `spa_redraw_start` hook each attempt - but on the trunk only IORS small overrides it. train85's spa2 patch (train85/patches/make_spa.py)
had ports for axial / attachment / block design / small articulation / disembedding / IORS large; they lived only in the train85 snapshots
and never reached the trunk. So today:

| family | start source today | constraints the start passes today | file:line |
|---|---|---|---|
| alignment-axial | TEMPLATE: `sample_template` draws edge / lateral from `default_rng(base_seed)`; every attempt reuses it | §4.2 edge stance on free floor + whole-body collision-free (`edge_candidates`); Phase 1b trunk; Phase 1c lateral: floor, collision (translate retry), trunk sweep; reference edge >= 3 / 9 points in frame and unoccluded from init (<= 8 laterals, translate moved in range); Phase 2 survivors see the front edge; per instance: edge visible with the target placed from init AND manipulation, target in the init image | alignment_axial.py:150-221, 171, 173, 197; instance 317, 552 |
| articulation open / close small | TEMPLATE (the target's facing yaw follows the template's edge) | §4.2, 1b, 1c as above; Phase 2 with target / handle / brace point in frame (`_target_in_frame`); a v2-only prefix-map miss rejects; per instance: target in the init image (`view_margin_m`) | articulation_small.py:347, 471-504; instance 1272-1281 |
| attachment | TEMPLATE | §4.2, 1b, 1c; Phase 2 + `_with_partner_room` (a partner cell at the gap); per instance: the partner in the init image; `_candidate_rejects` memory keyed by candidate index | attachment.py:385, 459-485; instance 745, 781, 1416 |
| block_design | TEMPLATE | §4.2, 1b, 1c; `_init_with_room_for_flips` (lateral moved for the flip shift / board fit); Phase 2 region survivors (`_region_in_frame`, depth caps) at the manipulation lateral shifts, shallow trunks kept without flips; per instance: every block in the init image; `_stance_pool` = trunk variations along the edge DURING the instance (not the start) | block_design.py:240-308; instance 709, 906, 969-972 |
| disembedding | TEMPLATE (destination side drawn per template / `generation.destination_side` for eval) | every edge with a free-floor run (1/3 of the 10 cm raster), 1b, 1c with the destination-side margins; segment arms; Phase 2 with the source tray in frame and the destination tray fitting beside it (`_destination_fit` as `keep_pairs`); per instance: the destination 4 cm inside the init image, every other object in frame | disembedding.py:679-723; instance 1038-1057 |
| IORS large ontop / under / inside / touching (push-under, lean, surface) | TEMPLATE: `bind_init_surface` -> `try_surface` -> `try_support_surface` (edge, lateral <= 6 redraws until the goal region is in frame, or `free_start`; Phase 2) or `try_floor_surface` (anchor c_t, stance B facing it, stance A on the eroded map with target + partner in frame, A -> B leg); `generation.init_setup_draws` = more TEMPLATE draws | §4.2, 1b, 1c; the partner's relevant region in frame from A (`goal_region_visible`, `partner_view_min_points`); the stance-C existence test of the template; per instance: target in the init image, A -> B leg, B -> C carry | iors_large.py:1112-1145, 2246-2325, 2286-2292, 2375-2451, 2600; instance 3512, 5350-5352 |
| IORS large nextto, touching base push (`TouchingBasePushMode`) | PER ATTEMPT already: stance A drawn per goal candidate in the attempt's draw (`_stance_a`: free eroded floor, bearing / distance, target + goal in frame, body-free, trunk sweep) | as listed | iors_large_nextto.py:1081, 1213 |
| IORS small (all five) | PER ATTEMPT already under START_PER_ATTEMPT=1 (every alias of the family): `spa_redraw_start` / `_spa_redraw_start_cached` (SPA_CACHE lateral cells), shelf_start_draw layouts per attempt; template start otherwise | edges with a free-floor run, 1b, 1c, Phase 2 cut to the A1 start region (lateral redrawn), the shelf rules (`shelf_width`, leans); SPLIT_CONE_RESTORE / `_spa_drawn` (core.py:392); per instance: target and partner in the init image (§4.6 step 4), partner placement incl. "partner not in the init image" | iors_small.py:1618-1636, 1928-1980, 1982-2058; instance 2277-2280, 2460 |

Shared by all: the eval-repeat check (`instance_separation.find_repeat`: start base <= 0.25 m / 15 deg of an eval instance AND every
compared orientation <= 0.1 rad; early on the draft `check_early_repeat` core.py:1330, final on the written TRO `check_instance_filter`
core.py:1362); the init -> manipulation travel (`record_stance_travel` / `record_edge_travel`, path length / collision) and the back-off /
transit rules in `sample_instance` (they read the attempt's init); the trajectory recorder's start = the TRO's `robot_poses` and the
sidecar `init` stance written per instance (`write_instance`; under start-per-attempt `template_meta.start_per_attempt` = no template-level
start). The collectors read the instance's TRO, so a per-attempt start needs no collector change.

### Change (switch `ISPATIALGYM_START_DRAW`, default off in d80e7e07, ON since the default commit)

- One switch for all families: `core.start_draw_on()`; `SamplerV2.START_DRAW_FAMILY` marks the families (True on AlignmentAxial,
  ArticulationSmall, Attachment, BlockDesign, Disembedding, IORSLarge, IORSSmall; False on the nextto / base-push modes, which draw
  stance A per attempt already). `start_per_attempt(s)` is also True for such a sampler under the switch (so the hook runs without
  train85's variable too).
- No new search: each family's start loop of `sample_template` is moved VERBATIM into `_start_search(rng, ...)` (axial, attachment,
  block design, articulation small, disembedding) - `sample_template` calls it with the template seed as before (same rng stream, same
  results), `spa_redraw_start` with the attempt's rng under the switch. IORS large: `spa_redraw_start` reruns `try_floor_surface` /
  `try_support_surface` on the template's bound init surface (the support's top reused); the template's stance-C existence test is not
  repeated (the attempt draws its stance-C pool itself). IORS small: its existing draw, unchanged (`spa_draw` now times it).
  Every check of the reused code stays: free floor, collision, trunk sweep, the camera-frame rules (never relaxed), the eval-repeat check,
  the travel / back-off rules of the instance. A search that finds no start rejects the attempt (`no_stance` / the family's reason);
  articulation / disembedding / IORS large then keep the previous start's state.
- Kit counters (only under the switch): `start_draw.drawn`, `start_draw.edge.<edge>`, `start_draw.rejected.<reason>`,
  `start_draw.repeat_passed` / `start_draw.repeat_rejected` (the repeat-check outcome of every written / early-checked draft); the log
  line `START_DRAW: start drawn {...}` and `summary.start_per_attempt` carry the drawn base pose / edge / lateral / trunk (the spread).
- Not drawn per attempt (out of the rule's scope): the joint goal value (articulation), the destination side and tray flips
  (disembedding) - train85's patch drew those too.
- Tests: ispatialgym/tests/test_start_draw.py (off = the template's start and the rng untouched; on = a different start per attempt
  per family; every drawn start passes the fake phases' rules AND the family's own rule in the reused code).
- IORS large FLOOR init (d50cb1b9, after the first A/B): only the robot's stance A is redrawn (`floor_stance_a` around the template's
  anchor c_t + the A -> B leg); c_t is the TARGET's start cell. Rerunning `try_floor_surface` redrew c_t too: touching_large_03 sd_on
  attempt 0 no_floor_cell (0 of the 40 drawn anchors within the geodesic), attempts 1 / 2 time_budget 1500 s at C (job cancelled).

Evidence for the audit (int1_1 / 3 / 4 / 6, written instances): 1 distinct start base per alias for axial, attachment, articulation small,
disembedding, IORS large; 2-4 per alias (= written) for IORS small and the nextto / touching small aliases.

### A/B (2026-11-08, $OFH/startdraw_20261108; sd_off vs sd_on = `--extra-env ISPATIALGYM_START_DRAW=1`, trunk_d80e7e07, the floor pair
### sd2_* on trunk_d50cb1b9; --grasps v2only on the staged copy v2maps_startdraw; N 4, base index 300500 in both arms; titanrtx)

| alias | off att / wr / ok, GPU-h | on att / wr / ok, GPU-h | drawn starts (on) | repeats_eval off -> on |
|---|---|---|---|---|
| alignment-axial_02 | 60 / 0 / 0, 0.29 | 25 / 3 / 3, 0.33 (0.11 / success) | 25 draws, edge +u only (the others fail 1a / the edge rule), lateral span 0.37 m | 46 / 60 -> 11 / 25 (start_draw.repeat_rejected 11, passed 3) |
| articulation_open_small_01 | 6 / 4 / 4, 0.18 | 6 / 4 / 4, 0.16 | 6 draws, edges +v 4 / -v 2 (yaw 180 apart), x 5.39-9.19 | 0 -> 0 |
| attachment_01 | 9 / 4 / 4, 0.45 | 11 / 4 / 4, 0.48 | 11 draws, edges -v 7 / +v 3 / +u 1 | 0 -> 0 |
| block_design_01 | 60 / 0 / 0, 0.55 | 45 / 2 / 0 (oracle timeout x2), 2.25 | 45 draws, edge +v only (`init_edges`), lateral 3.55-3.57: `_init_with_room_for_flips` moves every drawn lateral to the board-fit position | 0 -> 0 |
| disembedding_02 | 6 / 4 / 4, 0.37 | 7 / 4 / 4, 0.72 | 7 draws, edges -u 5 / +u 2, 4 distinct written starts | 0 -> 0 |
| ontop_large_03 (support init) | 8 / 4 / 4, 0.28 | 8 / 4 / 4, 0.22 | 6 draws + 2 rejected no_free_edge, edge +v, x span 0.70 m | 0 -> 0 |
| touching_large_03 (floor init, sd2) | 4 / 4 / 3, 0.23 | 4 / 4 / 4, 0.28 | 4 draws, stance A x span 1.2 m, yaw span 64 deg | 0 -> 0 |
| ontop_small_02 (IORS small, unchanged path) | 7 / 4 / 3, 0.43 | 6 / 4 / 4, 0.38 | 6 draws, edges +u 5 / -u 1 (off: 4 distinct starts as well) | 0 -> 0 |

The draw acts in every family (start_draw.drawn on every attempt; distinct written starts 2-4 per alias where off had 1) and no existing
constraint broke: no not_in_frame / collision / path rejection appeared or grew (the new reason is the draw's own no_free_edge 2 on
ontop_large_03; attachment's trembling 2 is the settle). Block design draws but its flip-room rule pins the lateral (2 cm spread) -
varying it would mean changing that rule (owner). axial_02: the repeat share fell 77 % -> 44 %; the remaining repeats are draws on the
same edge within 0.25 m of the eval start (the edge rule leaves +u only). GPU-h on block design rose (2 instances got to the demo).
DEFAULT ON (owner 3.5 decided the rule): separate commit; `ISPATIALGYM_START_DRAW=0` restores the template's start. touching_03 (open
question, i7-t03): IORS small's draw is the same code with or without the switch, so its "partner not in the init image" rejects
(the start lateral 0.35-0.40 m from the bay) are unchanged; a bay-centred lateral would plug into `_spa_redraw_start_cached` /
`spa_redraw_start` (core `spa_draw` counts it).

### 5o: the edge restrictions, the flip-room draw, axial_02 (owner decisions 2026-11-08 c, 5o / 5o.1 / 5o.2)

Owner (verbatim, 5o): "add a switch that ignores both restrictions, so every edge is offered and the existing checks decide (free floor,
collision, framing, repeats). A/B across the families ... The default is my decision after the A/B." 5o.1: "let the flip-room rule
(`_init_with_room_for_flips`) draw from the whole range that keeps flip room, instead of one point. Behind a switch, run together with
the edge switch and i7-blockdesign's board search." 5o.2: "Re-measure them with the edge switch on; diagnose further only if the repeats stay."

**Verification of the two mechanisms (before the change).**
- Support-row notes: 1651 of 1938 support rows of candidates.csv carry `edges=` (127 of 142 per tabletop subtask), but in B4's
  free-run format `edges=+u(1.65/1.65/1.65),-u(...),+v(...),-v(...)` - every free edge with its runs, not a pin. `ground_isg.parse_edges`
  (regex `edges=([+-][uv](?:,[+-][uv])*)\b`) reads only the FIRST edge of such a note (936 rows list more than one) - a latent
  misreading. It never acts: `ground_isg.support_surface` makes a `support_surfaces` entry only from a note with `top_z=`, and none of
  these notes has one (0 entries from the sheet); the customs has exactly one entry with `edges` (the attachment stockpot-bar's bar +u,
  owner 2026-10-12), and no fast-flow template carries any (every `template_meta.json` under `$OFH/out`: 0 with edges; no
  "the support row keeps edge(s)" line in any fast-flow sample.log). So mechanism 1 is dormant in the fast flow today.
- `generation.init_edges: ["+v"]` (block design only, 2026-11-06) is live: "generation.init_edges keeps edge(s) ['+v']" in all 15
  block_design_01 fast-flow job logs.
- axial_02 is not edge-pinned (sd_on, every draw): all four edges are offered; -u and -v stances collide at trunk (0.5, 0) in Phase 1a,
  +v (the 0.56 m end) fails Phase 1c `edge_too_short` (m_R + m_L = 0.66 m > 0.56 m); only +u passes, lateral [0.19, 0.61].

**Change.**
- `ISPATIALGYM_ALL_EDGES=1` (38990f02, default off): `tabletop.edge_frames` ignores `row_edges` and `gen_edges` - every edge of the top
  is offered and the existing checks decide; an interior surface's open sides (`top.open_edges`, geometry) still apply. Kit counters
  `start_edges.row_ignored` / `start_edges.gen_ignored` (only when a restriction was ignored). It is the shared edge selection
  (axial, articulation small, attachment, block design, disembedding, IORS small, IORS large support init all call `edge_frames`), not
  IORSSmall.sample_instance.
- `ISPATIALGYM_BD_FLIP_ROOM_DRAW=1` (7606f6ca, default off): `_init_with_room_for_flips(init, rng)` -> `_flip_room_draw`: the init lateral
  is drawn uniformly (attempt rng) among the laterals (1 cm grid + the drawn one) of the whole range that keeps the flip room
  (`L + need` inside the Phase 1c bounds), on the free floor, with the most Phase 2 stand-offs that keep a board position on the top
  (`_board_fit`, as the move). Empty range / nothing on the floor: the move as before. Counters `bd_flip_room_draw.drawn` / `.no_range`
  / `.no_floor`; `_flip_room["draw"]` records the admissible span. Unit (rhjoby-like top, bounds [0.044, 0.547], need +0.35): the move puts
  every draw at 0.127; the draw spans [0.044, ~0.134] with the base-relative list, [0.044, 0.197] with i7-blockdesign's BD_BOARD_TABLE.
- 7f7560af: test_start_draw's block design stub takes the rng. bd47a6d4: merge of i7-blockdesign f1fc4ac1 (5m.1 6000-step block
  design collectors, 5m.2 BD_BOARD_EARLY / SEARCH default on, BD_BOARD_TABLE default off) for the 5o.1 run.

**Runs** (`$OFH/startdraw_20261108`, loop 18383579 `loop_5o.sh`, trunk_bd47a6d4, `--grasps v2only` on v2maps_startdraw, N 4, base index
300500 (as sd_off / sd_on), alias memory off, svl titanrtx, <= 4 jobs at a time; tab: `jobs/v2build/tab_i.py -a <tag>`,
`startdraw_20261108/tab_5o.py <tag>` for the 5o counters / drawn edges).

The edge switch across the families (se_on = `ISPATIALGYM_ALL_EDGES=1`; the comparison is sd_on, the same code path with the start draw on,
since the switch only acts where a restriction exists - its counters say where):

| alias | sd_on att / wr / ok, GPU-h | se_on att / wr / ok, GPU-h | start_edges.* ignored | drawn edges (se_on) | written distinct starts | rejections that grew |
|---|---|---|---|---|---|---|
| alignment-axial_02 | 25 / 3 / 3, 0.33 | 25 / 3 / 3, 0.30 | 0 | +u 25 (lateral 0.20-0.57) | 2 | none (identical: repeats_eval 11, target_placement 9) |
| articulation_open_small_01 | 6 / 4 / 4, 0.16 | 6 / 4 / 4, 0.17 | 0 | +v 4 / -v 2 | 2 | none (identical) |
| attachment_01 | 11 / 4 / 4, 0.48 | 10 / 4 / 4, 0.45 | 0 | -v 6 / +v 3 / +u 1 | 3 | none |
| disembedding_02 | 7 / 4 / 4, 0.72 | 7 / 4 / 3, 0.70 | 0 | -u 5 / +u 2 | 4 | none (1 demo not_settled) |
| ontop_large_03 (support init) | 8 / 4 / 4, 0.22 | 8 / 4 / 4, 0.21 | 0 | +v 6 (+ no_free_edge 2) | 4 | none (identical) |
| ontop_small_02 (IORS small) | 6 / 4 / 4, 0.38 | 10 / 4 / 4, 0.39 | 0 | +u 6 / -u 4 | 4 | ik_joint_limit 0 -> 2 (N 4 noise: same code path) |
| touching_large_03 (floor init) | 4 / 4 / 4, 0.28 (sd2_on) | 4 / 4 / 3, 0.25 | 0 (no edge list) | floor 4 | 4 | none (1 demo plan_failed) |

`start_edges.row_ignored` / `.gen_ignored` = 0 on all seven: no restriction exists there, so the switch did not act (consistent with the
verification). The differences are N 4 noise on an unchanged path.

block_design_01 (where `generation.init_edges` acts), all four arms on the merge with i7-blockdesign's EARLY / SEARCH defaults and the
6000-step collectors (`max-steps 6000` in job.env):

| arm | env | att / wr / ok, GPU-h (/ success) | edges drawn (lateral) | rejections | 5o counters |
|---|---|---|---|---|---|
| bd5o_base | defaults (init_edges +v, the flip-room move) | 4 / 4 / 3, 0.96 (0.32) | +v 4 (0.067 every one: 1 start) | none (1 demo symbolic_only) | - |
| bd5o_e | ALL_EDGES | 4 / 4 / 3, 1.07 (0.36) | -v 3 (0.127) / +v 1 (0.067): 2 starts | none (1 demo symbolic_only) | gen_ignored 5 |
| bd5o_ef | ALL_EDGES + BD_FLIP_ROOM_DRAW | 6 / 4 / 3, 1.60 (0.53) | -v 4 (0.10-0.124) / +v 2 (0.044-0.064) | program_check_failed 2 (repose misses the pre-grasp, instance 300501 at -v 0.114 / 0.124; the move's -v 0.127 wrote at attempt 0 in bd5o_e) | gen_ignored 7, flip_room_draw.drawn 7 (admissible -v [0.044, 0.124] 9 laterals, +v [0.044, 0.064] 3) |
| bd5o_eft | + BD_BOARD_TABLE (i7-blockdesign's switch) | 17 / 3 / 3, 3.46 (1.15); the sampler hit its time limit, 300503 unwritten | -v 10 (0.06-0.20) / +v 8 (0.05-0.18), 3 starts (admissible [0.044, 0.197] 17-18 laterals) | target_placement (board_placement) 8, settle_validate no_transition 4, repose_unplannable 1, approach_untrackable 1 | - |

Written attempts: base all attempt 0 (390-882 s), e all attempt 0 (420-1111 s), ef attempts 0 / 2 / 0 / 0. ±u (the 2.90 m sides) were
offered but never reached a draw (the first passing edge is kept; +u collided once). Every arm's demo failure (1 symbolic_only) is in the
base arm too. Reading: the edge switch acts only here and adds the -v end (which never wrote under the 3000-step budget / without the
board search; now 3 of 4 at attempt 0) - 2 distinct starts, no rejection grew. The flip-room draw acts on every draw; with the base-relative
board list its admissible range is narrow (-v 8 cm, +v 2 cm), so the spread stays small; with the table-relative board positions it is the
whole flip-room range (15 cm per edge), but that arm's board placement fails more often (the table plan is i7-blockdesign's switch; its
own A/B decides it). N 4: the ef / eft extra rejections are not attributable to the draw from these counts alone.

Suite (exported trees, lambda): 1ba40aae 23 F + 3 E; 7606f6ca + test_start_draw::test_block_design (its stub lacked the rng; fixed in
7f7560af); bd47a6d4 (the merge) = 1ba40aae's ids = f1fc4ac1's ids.

**Needs the owner:** (1) the default of `ISPATIALGYM_ALL_EDGES` - today it acts only on block design (the one live restriction), where it
added the -v end with no rejection growth; the dormant row-note mechanism would pin the FIRST listed edge if a note ever carried
`top_z=` (parse_edges misreads B4's multi-edge notes) - on would make that moot. (2) the default of `ISPATIALGYM_BD_FLIP_ROOM_DRAW`
(acts; small spread with the base-relative board list, the full flip-room range with BD_BOARD_TABLE, whose own default i7-blockdesign's
A/B decides). (3) axial_02's repeats: options (a) / (b) / (c) below.

**Lead decisions while the owner is away (2026-11-08, tmp/outputs/lead_decisions_while_away.md 12-14):** `ISPATIALGYM_ALL_EDGES` DEFAULT ON
(b77691af, own commit; `=0` restores both restrictions; the restriction tests pin it to 0; suite = bd47a6d4's ids, snapshot
trunk_b77691af); `ISPATIALGYM_BD_FLIP_ROOM_DRAW` stays off until i7-blockdesign's BD_BOARD_TABLE A/B is in; axial_02's repeat-aware
draw is left to the owner (not built).

5o.2 (axial_02, re-measured with the switch on): repeats unchanged, 11 of 25 (`repeats_eval`), because nothing pinned the edge. The
diagnosis (`ISPATIALGYM_STANCE_BLAME=1` in se_on, logging only): every draw offers all four edges; -u's stance collides with the scene
object `fixed_window_ucejof_0` (0.07 m from the base: the desk stands against the window wall) 10 / 10, -v's with `armchair_ybcnmu_0`
(0.23 m) 9 / 9, +v (the 0.56 m end) fails Phase 1c `edge_too_short` (m_R 0.45 + m_L 0.21 = 0.66 m > 0.56 m) 8 / 8; +u alone passes,
lateral [0.19, 0.61]. Both eval instances (0000 / 0001) start at lateral ~0.54 of +u, and the measured entries (the book's start and goal)
always match them (0.00 m / 1-2 deg: the push map's admitted deviations sit inside the measured tolerance), so the repeat is decided by
the start base alone: a draw with lateral > ~0.29 is within 0.25 m of an eval start and repeats (rejected after 30-65 s each). Only
[0.19, ~0.29] of +u is free of repeats. Options for the owner (none implemented): (a) a repeat-aware start draw (laterals outside every
eval start's 0.25 m window first; a CPU test at the draw, the rejection after Phase 2 saved); (b) accept the 44 %; (c) the existing
rules stay and nothing else moves the start (the other edges fail free-floor collision and the edge-length margin, both real).

### touching_03 round 2 (owner 2026-11-08, `prompt_t03_round2.md`; branch `i7-t03`, all switches DEFAULT OFF)

Merged of-v2maps cf1f5d2d first (98c2e9b4; the integration's support hull fallback default kept; i7-t03's round-1 switches stay off).

**Diagnostic 0** (copies of trunk_98c2e9b4 with `t03_20261108/patch_drive_diag.py`, log only): the failing carried drives to the release
stance drove 0.14-0.25 m while the free gap between the lifted book and the partner was 0.11-0.12 m; every drive first collided at
s = 0.12-0.14 m, the ATTACHED BOOK + left finger 2 against the PARTNER book. The release's clear pose (15 cm back along the horizontal
approach: 0.38 m in front of the base) fails the IK with the support and the partner removed - a reach limit, not a collision bug.

- **(a) `ISPATIALGYM_SHELF_FREE_START`** (d510403d, 832fad79, fix 67a1ab4a): `IORSSmall.shelf_free_start` - the edge init stance when it
  frames the target AND the partner, else `iors_large.free_floor_stance` (factored out of `IORSLarge.free_start_stance`, behaviour unchanged)
  with that test; the drive to the edge stance = `prompt0.connect` + the shelf back-off arrival (`core.record_stance_travel`). A travel
  failure from a free start no longer drops the candidate (67a1ab4a). Called once at the top of `sample_instance`; `_spa_redraw_start_cached`
  is untouched (i7-startdraw). Kit `free_start.edge_ok / .used / .failed.*`.
- **(b) `ISPATIALGYM_SHELF_JOINT_STANCE`**: `shelf_fixed_start` / `shelf_stance_shift` judge each stance on the fixed-start grasp AND the tip
  release at the drafted goal (`shelf_release_why`: `place_infeasible_why` at `goal_pose` of the start poses); both first, grasp-only the
  fallback. Kit `stance_shift.both_passed / .grasp_only / .rejected.release`.
- **(c) `ISPATIALGYM_SHELF_RELEASE_IN_GAP`** (2f815f1b): the release stances = the manipulation stance, then drives toward the partner of
  0.25-1.0 x (free gap - `carry_path_margin_m`). Kit `shelf_release.drive_in_gap / .no_stance / .no_room`.
- **(d) `ISPATIALGYM_SHELF_LIFT_STRAIGHTEN`** (approved oracle + contact-rule change): `grasps.lift_turn` (the upright turn, refused when the
  upright book would reach the board above: `shelf_lift.straightened / .no_headroom`), oracle `Lift.turn`, `program_check.lift_end_pose`;
  `grasps.lift_leaned_piece`: at the lift only, the attached book may touch the board it leaned on (hull >= 0), every robot link clear of it
  (`lift_with_touch`; the program check's lift ignores that piece). Kit `shelf_lift.leaned_board_touch_allowed / _refused.*`.

A/B (legacy, N 4, 300100-103, kmuacd template only): a (trunk_2f815f1b) 320 att / 0 written, 1.79 GPU-h; b / c / d rerun on
trunk_67a1ab4a after the candidate-drop fix (i7t03r3_*), v2only i7t03r3_v with d's switches - numbers in DECISIONS "i7-t03 ROUND 2".

#### Round 2 results (2026-11-08; DECISIONS "i7-t03 ROUND 2 RESULTS" parts 1 / 2; `t03_20261108/tab_t03.py` + `tab_t03_extra.py`)

| arm | att | written | ok | GPU-h | where the switches acted |
|---|---|---|---|---|---|
| a all off (trunk_2f815f1b) | 320 | 0 | 0 | 1.79 | - |
| b (a) free start | 320 | 0 | 0 | 1.47 | free_start.used 76 / edge_ok 107 |
| c (a)+(b)+(c) | 320 | 0 | 0 | 2.69 | free_start.used 76; stance_shift.both_passed 13 / grasp_only 2; shelf_release.* 0 |
| d c+(d) | 133 | 0 | 0 | 1.49 | CRASH at the first goal_check pass; shelf_lift.* 0 |
| v v2only, d's switches | 9 | 0 | 0 | 0.13 | CRASH at the first goal_check pass |

- Layouts (a / b / c): bay +0.00 right 63, bay -0.37 left 64, bay -0.37 right 56 attempts, 0 written. The per-attempt draw is the same in
  every arm.
- Start stage: no_free_edge 137 in a / b / c (+u stance_collision 69, -u free floor 68). The free start does not change it, because the
  edge draw rejects before the fallback is asked. not_in_frame 0 at attempt level.
- (a): the fixed-start grasp rejections fall from 167 to 124-128, but 40 attempts now fail as no_path init -> manipulation (the free start
  disconnected on the eroded map).
- (b): 13 stances passed both the grasp and the release. Still, every attempt that reached goal_check in c died at the LIFT (4, vs 2 in
  a), before the release. So (c)'s in-gap drive never acted: carried drive 0 (a 2).
- (d) bug (832fad79): `shelf_lift_plan` calls `up_vector()`, and `IORSSmall.up_axis()` returns None for every relation except inside.
  On touching_03 this gives `ValueError: None is not a model axis`. The lift needs the book's standing axis for touching. Until that is
  fixed, d and v have no numbers.
- touching_03 still writes no demo. All switches stay default OFF.

#### (d) fix and the owner's decisions c (2026-11-08; DECISIONS "i7-t03 ROUND 2 (d) FIX" / "i7-t03 DECISIONS C")

- **(d) fix, b6da685d.** `shelf_lift_plan` takes the book's standing axis from the new `IORSSmall.shelf_standing_axis`, which is local
  to the shelf lift; `up_axis()` stays inside-only. The axis is the signed model axis most aligned with world +z (the rule
  `upright_shelf_facts` reads the lean from), applied to the v2 lean entry's upright base (`lean_base_wxyz`), else to the template's
  rest. For acbrnv both give -x. It is used only within 40 deg of vertical at the attempt's start.
  - Without an axis, or on any error in the plan, the lift is today's: no turn and no leaned-board touch. Kit
    `shelf_lift.fallback_no_axis.<why>` / `.fallback_error`.
  - Reruns: i7t03r4_d / _v on trunk_b6da685d (numbers in DECISIONS).
- **Merge** of i7-t03specs c4a6956a (0dcfa115): the opryhp_0 / otwukr_1 interior specs, the measure `--lateral` option and their tests.
- **5k.1, the computed shelf-width rule** (`ISPATIALGYM_SHELF_SPAN_RULE`, default off; 830e7837).
  - Required free span = the carried book's extent along the edge at the drawn lean (its thickness upright when (d) is on) + the
    open jaw's excess over the pinch, both sides: pre-shape opening `w + 2 x 10 mm` (else 0.10 m) + 2 x 12 mm fingers - `w`.
  - acbrnv: 0.148 m leaning 19.6 deg and 0.075 m upright, against otwukr's 0.09-0.10 and kmuacd's 0.128 at 20 / 20.
  - It is checked per attempt at the drawn leans, and at the template stage at the most favourable drawable leans. Kit
    `shelf_span.passed` / `.rejected` / `.template_rejected`.
- **5p.2, every candidate's trunk in the stance shift** (under `ISPATIALGYM_SHELF_JOINT_STANCE`; 830e7837).
  - Each shifted base position is tried at the candidate's trunk, then at the other trunks of the attempt's candidates.
  - The scan budget counts only the stances that reach the exact checks. Kit `stance_shift.trunk.other_tried` / `.other_passed`.
- **(c) tests.** 2f815f1b had one test (the drive fractions). Added `lifted_book_gap` and the plan built from it.
- **5p.1, the retreat by the base** (`ISPATIALGYM_SHELF_RETREAT_BACKOFF`, default off; approved oracle change; 36e67650).
  - Applies to the shelf tip release, when its 15 cm clear pose fails.
  - The sampler checks the place's own end (10 cm back) by exact IK, then a 5 cm drive-back along the stance axis with both arms frozen
    there and the fingers open (the strides of `navigation.path_collision_free`).
  - It records stance `retreat` plus the transition into it. The oracle's shelf tip program then runs place -> that transition, instead
    of place -> the 5 cm arm retreat.
  - Kit `shelf_retreat.backoff_passed` / `.backoff_failed.<place_end|drive|no_edge>` / `.recorded`.

#### Round 5 results (2026-11-08; DECISIONS "i7-t03 r5 ..." lines, "i7-t03 r5 RESULTS")

Runs on trunk_1fda4bd2 (suite = the base: 23 failed + 3 errors, the same ids, 3165 passed), N 4, memory off. `$OFH/t03_20261108`:
`sub_r5.sh` (+ a slot gate and an `otw` case), `loop_r5.sh`, `after_T7.sh`, `mk_registry.py`, `parse_lift_diag.py`.

| arm (tag) | grasps / support | switches | att / wr / ok | GPU-h |
|---|---|---|---|---|
| c (i7t03r3_c) | legacy, kmuacd | FREE_START + JOINT_STANCE + RELEASE_IN_GAP + STANCE_SHIFT | 320 / 0 / 0 | 2.69 |
| d (i7t03r4_d) | legacy, kmuacd | c + LIFT_STRAIGHTEN | 320 / 0 / 0 | 2.88 |
| v (i7t03r4_v) | v2only lean root, kmuacd | d | 320 / 0 / 0 | 1.19 |
| e (i7t03r5_e) | legacy, kmuacd | d + SPAN_RULE + RETREAT_BACKOFF | 320 / 0 / 0 | 3.41 |
| T6 off (i7t03r5_T6off) | v2only, opryhp_0 T6 | none | 320 / 0 / 0 | 1.92 |
| T6 on (i7t03r5_T6on) | v2only, opryhp_0 T6 | LIFT_STRAIGHTEN | 320 / 0 / 0 | 1.96 |
| T7 (i7t03r5_T7) | v2only, otwukr_1 T7 | e's | 36 / 4 / **2** | 0.26 (0.128 per success) |
| T8 (i7t03r5_T8) | v2only, otwukr_1 T8 | e's | 28 / 4 / 0 | 0.30 |

- **d.** The rejections are c's exactly. (d) acts 4 times, and all 4 are refused as penetration: the lift is reached 4 times in 320.
  The blockers come earlier: no_free_edge 137, the fixed-start grasp 123, init -> manipulation 40.
- **e.** The rejections are c's / d's: target_placement 169 (fixed-start grasp 123, init -> manipulation 40, transit 6), no_free_edge
  137, approach_untrackable 10, lift 3, place 1.
  - The new switches act but clear nothing. shelf_span.passed 183 (0 rejected on kmuacd). The 5p.1 back-off was tried 7 times and
    failed at place_end all 7 (never recorded). 5p.2 tried other trunks 234 times with 0 passing. (d) straightened 4, refused 3.
  - On legacy kmuacd, the blockers sit before the lift.
- **T6 (d) off / on.** With (d) on, lift_unreachable goes 152 -> 0 and ik_joint_limit 26 -> 0 (straightened 222, touch allowed
  181, refused 0).
  - But repeats_eval rises 32 -> 158. Every one repeats the template's single eval instance `__0000`: opryhp T6 draws one
    layout, so a goal-passing attempt is the eval instance again.
  - gripper_not_in_frame rises 1 -> 53.
- **T7 / T8 templates.** With SPAN_RULE + (d), both otwukr templates pass the shelf_width setup that stopped i7-t03specs (drawn free
  spans 0.110-0.125 m). T7 wrote 4 in 9 attempts, T8 4 in 5. Registered in `reg_T7` / `reg_T8` + `registry_T7` / `_T8.tsv`, which
  are the trunk touching_03 row with the template fields swapped.
- **T7 / T8 N 4.** The first touching_03 successes are T7 300401 and 300402: grasp, lift + straighten, place, retreat.
  - 5 of the 6 demo failures are the same self-contact, "left_arm_link5 touches torso_link2": T8 x4 at the 5 cm arm retreat after the
    place, and T7 300400 at the place itself (the instance that recorded the 5p.1 back-off).
  - T7 300403 failed on arm_stutter during the grasp.
  - The sampler checks the 15 cm clear pose by exact IK. It is not yet diagnosed whether that check misses this pair or the executed
    path leaves the checked one.
- **Lift refusal diagnosis** (i7t03r5_diag = the v settings on a log-only copy of trunk_b6da685d; it reproduces v exactly, 53 refusals).
  - All 53 are identical. The piece is `kmuacd__base_link__collision__6`, the leaned-on side board. It is convex: hull volume =
    mesh volume = 0.005155 m3 (11 vertices, 18 faces, watertight), so its hull does not swallow the book.
  - One book hull vertex is involved, (-0.1108, -0.0826, -0.0156) in the book frame: the top-end cover corner. It is less than 0.05 mm
    deep, at mesh distance 0, at lift pose 0 only (the grasp pose = the book's resting lean). The mid and last lift poses are 0.01 m clear.
  - So the refusal is the book's own resting contact at the start pose, refused by the rule's `cl < -1e-9`; the lift does not
    enter the board.
  - Exempting the start pose, or a tolerance, is a contact-rule change for the owner (not made).
- **Still blocking.**
  - otwukr: the left_arm_link5 / torso_link2 self-contact at the retreat and place (next: diagnose the check against the execution).
  - kmuacd: before the lift (start edge 137, fixed-start grasp 123, init -> manipulation 40), plus on the v2 path the start-pose
    contact refusals (53).
  - opryhp T6: the single drawn layout (repeats_eval 158) and the camera (53).
- **For the owner.** The touch rule's start-pose contact; whether otwukr T7 / T8 become touching_03's templates; the defaults of
  the t03 switches. All stay off; no code changed in round 5.

#### Round 6 (2026-11-08; LEAD DECISIONS 26 / 27 while the owner is away; DECISIONS "i7-t03 r6 ...")

- **(d) touch-rule fix (d4f6975a, behind `ISPATIALGYM_SHELF_LIFT_STRAIGHTEN`, default off).**
  - `lift_with_touch` now judges the attached book's hull clearance to the leaned-on board per lift pose
    (`lift_touch_clearance_why`). Pose 0 is the grasp pose, i.e. the book's resting lean; there the hull may read down to
    -0.1 mm (`LIFT_START_CONTACT_TOL_M`). Every later pose keeps clearance >= 0, and every robot link keeps today's rule.
  - Kit: `shelf_lift.leaned_board_start_contact_allowed` (allowed although pose 0 reads < 0);
    `.leaned_board_touch_refused.penetration` (a later pose < 0) / `.penetration_start` (pose 0 deeper than 0.1 mm) / `.ik` /
    `.robot_link`. Log line `SHELF_LIFT_TOUCH refused <why> cl0 .. cl_rest ..`.
  - Tests: 0.05 mm at pose 0 passes; 0.2 mm at pose 0 fails; below 0 at pose 1 or later fails; a robot link is refused as today.
  - Rerun (`t03_20261108/sub_r6.sh` = `sub_r4.sh` on trunk_d4f6975a, 300100-103, svl titanrtx).
  - **Arm v** (v2only, lean root `v2maps_t03`, d's switches): 298 att / 1 written / 1 ok, 1.17 GPU-h. That is the first touching_03
    success on kmuacd (300100); r4_v was 320 / 0 / 0, 1.19 GPU-h.
    - Where the fix acted: the lift refusals went from 53 to 0. `straightened` 61, `leaned_board_touch_allowed` 48, and all 48 are
      `leaned_board_start_contact_allowed`: every allowed lift had the resting contact at pose 0, and none read below 0 later.
    - Past the lift, attempts now die at the camera (gripper_not_in_frame 35, "pre-grasp fingertip outside the manipulation image"),
      the release search (place_unreachable 13) and the retreat after the place (approach_untrackable 12, "the tracked chain's end
      configuration collides").
    - Before the lift nothing changed: no_free_edge 128, target_placement 109.
    - Caveat: trunk_d4f6975a also carries r5's 5p.2 stance shift (830e7837), so the r4 / r6 difference is not only the touch rule.
      The lift counters are the touch rule's alone.
  - **Arm d** (legacy kmuacd, d's switches): 255 att / 2 written / 0 ok, 2.88 GPU-h; r4_d was 320 / 0 / 0, 2.88 GPU-h.
    - The lift was reached 4 times, as in r4. All 4 are now allowed (`start_contact_allowed` 4), where r4 refused them as penetration.
      Two of them were written; the other two failed the program check's retreat ("end configuration collides").
    - Both demos (300102, 300103) failed `robot_touching_itself`, "left_arm_link5 touches torso_link2", at the retreat. This is the
      otwukr mechanism, now on kmuacd too. The 15 cm clear-pose witness is -3.8 / -2.0 mm by mesh, while link5's spheres read
      +10.5 / +16.2 mm. Executed, the arm reaches 0.0 mm by mesh the step before contact.
    - Before the lift nothing changed: target_placement 141, no_free_edge 102.
    - So the sphere-model miss now blocks touching_03 on both bookcases: 5 otwukr demos and 2 kmuacd demos. Its fix is the owner's.
- **otwukr self-contact diagnosis (CPU only, no fix; `t03_20261108/selfc_r6/`).**
  - Method: FK of the r1pro URDF; the simulator's collision meshes (the r1pro.usda convex pieces); the sampler's cuRobo spheres.
    Checked configurations are each demo's `ik_witnesses`, executed ones are `demo.hdf5` joint positions.
  - The pair left_arm_link5 / torso_link2 is in no ignore list, so the sampler checks it - on spheres that miss the contact.
    The touching part is `left_arm_link5_col_3`, the wrist end of link5 (z -0.02 .. -0.19 in the link frame). 46 of its 60
    vertices lie outside every link5-7 sphere, by up to 10 mm, and up to 57 mm outside link5's own spheres, which end at z -0.14.
    It touches torso_link2's front face.
  - T8 (all 4): the 15 cm clear pose that `release_clear` checks (= the action-5 retreat target) has mesh clearance -0.4 / +2.2 /
    -0.6 / +0.2 mm, while the spheres read +5.3 / +16.0 / +9.6 / +11.3 mm (link6's; link5's own +21 .. +38 mm). Executed, the
    arm reaches -0.7 .. 0.0 mm at the contact step.
  - T7 300400: the place's 10 cm retreat witness is 15.6 mm into the torso by mesh, +4.7 mm by spheres; executed -0.3 mm at the
    contact step (inside the place action).
  - The successful T7 300401 / 300402 retreat witnesses are +56 / +17 mm by mesh.
  - So the executed path does not leave the checked one: the arm runs toward the checked witness and touches as it nears it. The
    sphere model passes configurations whose real meshes touch. A fix (a sphere for link5's wrist end, or a mesh check of this
    pair on the release witnesses) changes the shared robot collision model or a shared check, so it is for the owner.
- **Arm v: what blocks after the lift** (CPU only, from `i7t03r6_v/sample.log`; no code change). The three blockers split cleanly by the
  manipulation stand-off.
  - **Camera, gripper_not_in_frame 35: all at stand-off 0.45.** That is Phase 2's own stance; the stance shift moved the base only 13
    times, all to 0.55.
    - The failing check is the goal check's `require_in_frame` at the manipulation stance's head camera, under the customs'
      `grasp_frame_rule: fingertip`. It fails on the pre-grasp, 10 cm back along the approach: its fingertip falls outside the image
      with the 5 % margin.
    - The stance gates frame only the book, so these stances pass and die after the lift and the release search.
    - Narrowest fix: run the same pre-grasp fingertip test when the shelf tip's fixed-start grasp and stance are chosen, so a failing
      stance becomes "need grasp" and the existing stance shift tries 0.55. It is shelf-local, the frame rule is unchanged, and it
      is not a shared gate.
  - **Release, place_unreachable 13 (12 at 0.45).** (c) does act: these are its release searches, 5 laterals along the free gap
    each, 65 tries in all.
    - Clear pose (the 15 cm arm retreat, exact IK): 35.
    - Descent: exact IK 17, left_arm_joint2 within 2-46 mrad of its limit 10, untrackable 3.
    - Narrowest fix: arm v plus the existing owner-approved 5p.1 switch `ISPATIALGYM_SHELF_RETREAT_BACKOFF` (off in arm v), aimed at the
      35 clear-pose failures. It is shelf-local. The joint2-limit descents need another stance, not a rule change.
  - **Retreat after the place, approach_untrackable 12: all at stand-off 0.55, released from the manipulation stance.**
    - All are the program check's "the tracked chain's end configuration collides". It ignores the supports and the released book
      along the retreat, as `release_clear` does, so the world is the same. The configuration differs: the program check tracks the
      retreat from the place configuration, while `release_clear` solved the same 15 cm pose afresh and passed.
    - That code branch logs no blame, so the object is unknown. It is a sphere-model collision, so it is not otwukr's mesh-only
      link5 / torso_link2 contact.
    - Next step: a log-only blame on that branch (`ik.py`, the end-configuration check). Fix candidate: `release_clear` checks the
      retreat tracked from the place configuration, as the program check does (shelf tip only, not a shared gate).
- **Left to the owner:** otwukr T7 / T8 as touching_03's templates, the eval index, the t03 switch defaults.

#### Round 7 (2026-11-08; LEAD DECISION #37 while the owner is away; DECISIONS "i7-t03 r7 ..."; every switch default OFF)

- **Code (37bd1888).**
  - `ISPATIALGYM_T03_PAIR_MESH` (`sampling/v2/pair_mesh.py`): the `{arm}_arm_link5` / `torso_link2` clearance on the simulator's own
    collision pieces, i.e. the r1pro.usda convex hulls, posed by the URDF's chain in numpy.
    - Overlap is decided exactly (an LP); the gap is exact too (Wolfe's min-norm point, equal to SLSQP within 4e-13 m on 300 random hull
      pairs).
    - Margin 5 mm. Two T8 witnesses read +2.1 / +0.1 mm, yet their executed retreats touched; the successful retreats read +16.8 mm or
      more. The sphere model is unchanged.
    - Judged at `release_clear`'s 15 cm clear pose, at the 5p.1 place end, and at the program check's retreat witnesses (reject
      `self_mesh_contact`). Kit `pair_mesh.checked.<where>` / `.refused.<where>`.
  - `ISPATIALGYM_SHELF_PREGRASP_FRAME`: when the shelf stance is chosen, the goal check's own `require_in_frame` (pre-grasp + grasp, same
    rule and 5 % margin) runs at that stance's camera. A failure becomes the stance-shift need `frame`. Kit
    `shelf_pregrasp_frame.checked / .passed / .refused`.
  - `ISPATIALGYM_TRACKED_END_BLAME` (log only): the tracked chain's "end configuration collides" verdict logs `TRACKED_END_BLAME`
    with the colliding object or self-collision pair.
  - `ISPATIALGYM_SHELF_RELEASE_TRACKED`: `release_clear` (and the 5p.1 place end) track place -> 10 cm -> 15 cm from the place
    configuration, as the program check and the oracle do, instead of solving the 15 cm pose afresh. Kit
    `shelf_release_tracked.checked / .passed / .failed`.
  - Tests `test_t03_round7.py` (16). pyflakes and the suite match the base (23 failed + 3 errors, same ids).
- **Cross-alias count of the pair (CPU).** Every recorded step of the 141 int2 written demos (28 aliases, both arms): 0 demos come
  within 0 mm and 0 within 5 mm. The closest are nextto_04 at +45.5 mm and ontop_large_scale_03 at +48.1 mm; every other alias stays
  50 mm or more away. So the pair touches only on touching_03's shelf retreat. The default is the owner's.
- **A/B** (`sub_r7.sh`: r6's settings + `SHELF_RETREAT_BACKOFF` + the four switches; 300100-103, svl titanrtx; counters from
  `tab_t03_r7.py`).

  | arm | att / written / ok | GPU-h | per success | r6 (same arm) |
  |---|---|---|---|---|
  | v (v2only, lean root) | 48 / 4 / 4 | 0.26 | 0.064 | 298 / 1 / 1, 1.17 |
  | d (legacy kmuacd) | 256 / 2 / 2 | 2.88 | 1.44 | 255 / 2 / 0, 2.88 |

  - **Arm v.** The camera, release and retreat failures went from 35 / 13 / 12 to 0 / 3 / 1.
    - Frame test: checked 38, passed 21, refused 17; one stance shift for the `frame` need.
    - Tracked release: checked 44, passed 20, failed 24.
    - 5p.1 back-off: passed 14, recorded 4. All 4 written instances retreat with the base, so the arm stops at the 10 cm place end.
    - Pair mesh: 4 refused at the clear pose (+2.4 to +3.0 mm, inside the margin), 0 refused at the place end or the program check.
    - Log-only blame: the one remaining retreat reject is "left_arm_link6 x torso_link2 (5 mm)", a sphere-model refusal.
  - **Arm d.** 2 written and both succeed; r6's two demos both hit the link5 / torso_link2 contact.
    - Pair mesh: 9 of 11 refused at the clear pose (-3.4 to +4.9 mm).
    - Frame test: 42 checked, 14 refused. Tracked release: 38 checked, 18 failed.
    - Before the lift nothing changed: target_placement 140, no_free_edge 108.
  - **link6 check.** link6 never touched in execution (closest +7.4 mm). Its only retreat witnesses within 5 mm already had link5
    below 0, so extending the mesh check to link6 would refuse nothing more. The check stays link5-only.

#### Round 8: touching_03's defaults (2026-11-08; LEAD DECISION #39 under the owner's mandate "fix all the bugs and speed up / make yield better the slow low yield aliases")

- **Mechanism (b4da2675, `sampling/v2/alias_switches.py`).**
  - A switch's environment variable wins when set ("0" = off). Unset, the switch is on only for the shelf tip family
    (`IORSSmall.shelf_tip()`, the shelf layout with `generation.shelf_tip_release`) and only for the names in `SHELF_TIP_DEFAULTS`.
  - Only touching_03's customs entry has `shelf_tip_release`; a test pins this. touching_01 / 02 / 04 and every other IORS alias read
    unset as off, as before.
  - 5ee25e37 makes the test stand-ins pin these switches off unless a test sets them.
- **One commit per flip**, in this order: e9c7acf4 FREE_START, 1d2aec35 JOINT_STANCE (with 5p.2), 3a986bad RELEASE_IN_GAP, 94782615
  STANCE_SHIFT, 86b9700c LIFT_STRAIGHTEN, 370438e0 RETREAT_BACKOFF, ce191a3f PREGRASP_FRAME, 552b08c5 RELEASE_TRACKED, 58ec7ec2
  T03_PAIR_MESH.
  - pyflakes is clean on every commit, and the full suite matches the base on every commit (23 failed + 3 errors, same ids).
  - Not turned on: `TRACKED_END_BLAME` (log only), `SPAN_RULE`, and the round-1 switches.
- **Runs on trunk_58ec7ec2 with no `--extra-env`** (`sub_r8.sh`):
  - **touching_03** (300100-103): 48 / 4 / 4, 0.26 GPU-h = 0.065 per success. It reproduces r7 arm v attempt for attempt, with the
    same rejections and counters.
  - **touching_02 control** (610100-103): 51 / 4 / 3, 0.41 GPU-h (int2_4 on the integration trunk: 89 / 3 / 3, 0.60). No line or
    counter of any t03 switch appears; the one failed demo is plan_failed.

## Owner decisions on the open list (2026-11-08)

Verbatim answers: `tmp/outputs/owner_answers_20261108.md` (main checkout); record: `$OFH/DECISIONS.md` ("OWNER DECISIONS on
open_decisions_20261108" and "LEAD ROUTING"). Each implemented item is documented on its own branch (all from of-v2maps cf1f5d2d):

| item | decision | branch |
|---|---|---|
| 1.1 | disembedding_01/10: diagnose the tray failure first (open-slab validation vs tray walls); gym shoe switches-on entry: ladder round 2 | i7-tray |
| 1.2 | the full 100-alias pass waits until all blockers are solved and integrated | - |
| 2.1-2.4 | outcome memory: proposed thresholds adopted; production starts empty and resets on customs / keys change; validation runs seed from the replay and carry over at reduced weight; add a pick hook | i7-memory2 |
| 3.1 | full-open brace default confirmed | - |
| 3.2 | close_02 carton slide: diagnosis | i7-contact3 |
| 3.3, 3.4 | fix the validator's brace-load pusher (fixed base), rebuild; then the axial / world_vertical brace libraries | i7-brace |
| 3.5 | templates never fix the robot's start base: drawn per attempt in alignment-axial, articulation open/close small, attachment, block_design, disembedding, all IORS large and IORS small | i7-startdraw |
| 3.6 | cap the place depth by the push map's reach, all affected aliases | i7-contact3 |
| 3.7 | remove the pack brace; block design uses the board variant | i7-contact3 |
| 4.1 | outer_pinch + attach_face_retreat become the default (after a count) | i7-attachdef |
| 4.2, 4.3, 4.5 | cork: try wide-end-down with a hand-over and lying down with a 90 deg reorient, compare yield; brace_search false for attachment_03; no overhang fallback for the cork | i7-cork |
| 4.4 | audit past attachment demos for the collector reload offset; re-collect affected instances | i7-attachdef |
| 4.6 | the oracle's 45 deg seat-axis rule stays | - |
| 5.1, 5.2 | framing also at the lifted carry (torso lift with the arm lift where needed); pre-grasp and touching pre-contact framing use the fingertip rule | i7-framing |
| 6.1-6.3 | IORS large pick checks fingers at the pre-shape opening + overhang fallback; stance-C survivor cap with a map; fridge carry / place diagnosis | i7-interior2 |
| 7.1, 7.2 | FingerTouch: sphere screen then exact hull for palm / camera / wrist; rerun the bar / countertop sink templates with interior maps | i7-ft2 |
| 8.1 | the two decrypted B1K USDs in the lead's scratchpad deleted | - |
| 9.x | touching_03 (i7-t03): not yet answered; switches stay off, branch unmerged | - |
| 2026-11-08c item 3 | block_design_01: diagnose each board_placement sub-reason, fix behind a default-off switch with an N 4 A/B, reject infeasible board draws early | i7-blockdesign |

### i7-blockdesign (owner decision 2026-11-08 item 3): block_design_01 diagnosed, fixed, faster

Branch `i7-blockdesign` from of-v2maps 93229922, i7-contact3's 5aeb9da4 (pack tail removed) cherry-picked first (e1b901aa: the fix
touches block_design.py). Record `$OFH/blockdesign_20261108`, DECISIONS "i7-blockdesign" lines. Baseline c3_bd_base / c3_bd_new (and
bd_off on this branch, switches off): 60 attempts / 0 written, every candidate board_placement, 21.6 s and 74 exact IK per rejected
attempt, 0.45 GPU-h.

Diagnosis (ISPATIALGYM_BD_DIAG, logging only: bd_diag2 = the board positions / probes of the 6 candidates tried, bd_diag3 = every
candidate swept on a 8 x 12 grid of board positions with the probe re-run from FLIP_TRUNKS):

* The template never changes between attempts: the start is not drawn (`start drawn: None`), the init is the template seed's +v edge,
  lateral 0.35 moved by the flip-room rule to 0.07 (any drawn lateral >= 0.07 lands there: i7-startdraw's per-attempt start would give
  the same init and candidates), so all 60 attempts repeat the same 6 candidates and the same 18 verdicts.
* Maps: under the v2 generic reach map (h 0.8126) every one of the 27 Phase 2 survivors stands at trunk translate 0.95 (tilt 0 / -0.16);
  the legacy band's pool (eval85 a2_k4, the written instance) also had 0.75 / 0.25 / 0.00. The 6 tried (score order: the shift
  penalty puts the shift-0.25 candidates last) are all stand-off 0.45 at shifts 0.35 / 0.45; the stand-off 0.35 ones are ranks 7-13.
* "Off the top" (0.46, -0.02) / (0.44, -0.10) / (0.48, +0.04): the plate is 0.0908 m (half 0.0454) and the test wants 1 cm margin, so
  the board centre must be >= 0.055 m in from the edge; at stand-off 0.45 those depths are +0.010 / -0.010 / +0.030 m from the edge
  (both near corners off). The list was set for stand-off 0.35 (eval85 a1_seed0 wrote with (0.46, -0.02) at stand-off 0.35). (0.60,
  +0.05) is off the edge's END at shift 0.45 (manipulation lateral 0.52 + 0.05 + half > 0.58).
* "No pitched place reaches the pockets": reach, not collision - the far-left pocket's pre-place has no IK (`exact IK failed at pose 1
  of 2`) with the board and the board + support removed from the world as well. From the 0.95 trunk at stand-off 0.45 the probe passes
  only right of the base near the edge (board lateral <= -0.10, depth <= 0.54); trunk 0.75 / -0.30 at the same base reaches (0.52,
  -0.06) / (0.52, -0.02) (bd_diag3).
* "No cell set lets every repeated-colour block flip": rbbr needs 2 flips, each from a flip-eligible cell (base lateral <= -0.10, >= 0.10
  m in from the edge, >= 0.07 m left of the init camera) with no other block within 0.20 m to its right and 0.12 m in depth. The
  region lattice is diagonal (8 cm offsets), so two eligible cells are usually diagonal neighbours and block each other; the board
  (footprint + 6 cm cell margin = 0.118 m half) removes the same-column partner (candidate 10 at (0.52, -0.06): 4 kept, 2 eligible,
  diagonal). On the sweep candidates 10 / 13 / 14 / 16 have no board position with a flip set at all; 11 / 12 / 15 have 10-28; the
  stand-off 0.35 candidates 17-22 have 38-48, and the probe passes from their own trunk at depth 0.44.
* Not involved: the place / grasp maps (the probe and the layout are geometric + exact IK), the pack brace (block_design_01 is the
  board variant), alias memory (off in every arm).

Switches (default off, sampler-side; no oracle / shared gate change):

* `ISPATIALGYM_BD_BOARD_EARLY=1` (speed, same verdicts): `_board_precheck` runs the board's CPU tests (on the top, cells kept, flip
  set) on the candidate's region before the init -> manipulation travel and before the exact probe; inside `_place_board` the flip-set
  test runs before the probe; a failed probe is kept per (base pose, trunk, board xy, yaw) for the process. Counters
  `bd_board.precheck_pass / precheck_reject / layout_before_probe / probe_run / probe_memo_hit`.
* `ISPATIALGYM_BD_BOARD_SEARCH=1` (the fix): (1) the candidates whose region has a board position passing the CPU tests (list + grid,
  any configured pattern) are tried first, in score order (`_board_feasible`; here 10 of 27, tried 12, 15, 11, 17, 19, 18); (2) after
  the list, a grid of positions (depth 0.45-0.66 x lateral -0.22..+0.06, nearest first to the list's first position on the top, ties
  right first), the CPU tests before each probe, at most `generation.board_grid_max_probes` (12) probes per candidate. Counters
  `bd_board.feasible_candidates / order_changed / grid_searched / grid_placed / grid_failed`.

A/B (cluster svl titanrtx, N 4 v2only, 320100-320103, alias memory off, snapshot trunk_788907d5; `$OFH/blockdesign_20261108/submit_ab.sh`):

| arm | switches | att / written / ok | s per rejected attempt | exact IK / attempt | job wall | GPU-h |
|---|---|---|---|---|---|---|
| bd_off (= c3_bd_base / new) | - | 60 / 0 / 0 | 21.4 | 74.1 | 1618 s | 0.45 |
| bd_early | BD_BOARD_EARLY | 60 / 0 / 0 | 2.5 (attempt 0 16.9, then 2.2) | 0.3 | 496 s | 0.14 |
| bd_fix | BD_BOARD_EARLY + BD_BOARD_SEARCH | 4 / 4 / 0 | - (no rejection; 1.0 attempt per written) | 522 | 2944 s | 0.82 |

bd_fix wrote the first fast-flow block_design_01 instances (attempt 0 each; 10 of 27 candidates board-feasible, tried 12, 15, 11, 17,
19, 18; the board at the grid's (0.51, -0.10) after 3 probes); written attempts 826 / 383 / 441 / 413 s (the first incl. 531 s of
cuRobo attach sphere fitting). Every demo ended `oracle_failed:timeout` at action 33 (the 4th block's place) when the program reached
the oracle's 3000-action-step budget (`eval/instances.DEFAULT_MAX_STEPS`: no demo length, no --max-steps); 320100 / 320102 had three
blocks seated within 1 mm, 320101 / 320103 three blocks 4.0-4.5 cm off their slots (not diagnosed). The eval85 block_design_01
successes took 3520 / 4149 steps. The step budget is a shared gate: the owner decides (measurement: recollect with
`ISGF_COLLECT_EXTRA="--max-steps 6000"`, slurm 18381777: 4 / 4 SUCCESS at 3267 / 3382 / 3457 / 3512 action steps, every block <= 1 mm -
the 4 cm of 320101 / 320103 were mid-program - demo walls 635-904 s, 0.46 GPU-h). GPU-h per success: off / early - (0 ok); fix +
recollect (0.82 + 0.46) / 4 = 0.32, one job with the larger budget ~0.21-0.23 (estimate).

Needs the owner: (a) the demo step budget for block design (`DEFAULT_MAX_STEPS` 3000 without a demo length is a shared gate; e.g.
`--max-steps 6000` for the block_design collectors via `ISGF_COLLECT_EXTRA`, or a per-family budget); (b) the defaults of
`BD_BOARD_EARLY` / `BD_BOARD_SEARCH` (both off; they act only on board variants).

### i7-blockdesign 5m (owner decisions 2026-11-08c): the step budget, the defaults, table-relative board positions

The owner's 5m.1 / 5m.2 / "Board positions" (verbatim in the 2026-11-08c decisions; the early-reject audit of 5m is i7-diag's):

* **5m.1 (de5779b0) a 6000-step oracle budget for block-design collectors only.** Mechanism: `scripts/trainfast/submit_fast.py`
  `COLLECTOR_MAX_STEPS = {"configuration_matching-block_design": 6000}` (alias-key prefix) - `with_collector_max_steps` appends
  `--max-steps 6000` to `ISGF_COLLECT_EXTRA` in the job's `job.env` (fast_env.sh -> `COLLECT_EXTRA` -> every `run_isg --mode oracle`
  collector of the job), on `submit` and on `recollect` (a recollect of an older job without it gets it too); a job that passes its own
  `--max-steps` in `ISGF_COLLECT_EXTRA` (--extra-env) keeps it. The shared `eval/instances.DEFAULT_MAX_STEPS` 3000 is unchanged; the
  SUBMITTED line prints `collect_extra=...`. Not a customs field: the collectors take the budget from the command line, and job.env
  is the job's record.
* **5m.2 (73d54d48) `ISPATIALGYM_BD_BOARD_EARLY` and `ISPATIALGYM_BD_BOARD_SEARCH` default ON** (own commit; `0` restores each; board
  variants only). The tests pin the off arms explicitly.
* **Board positions (f1fc4ac1) `ISPATIALGYM_BD_BOARD_TABLE=1`, default off.** The hand-set `board_base_xy_candidates` are base-relative
  (base depth, lateral) and were tuned at stand-off 0.35 (`BOARD_LIST_STAND_OFF`), so at 0.45 the near ones are off the top. With the
  switch a board position is (inset of the board centre from the working edge, lateral from the stance's centre line): the depth is
  the TABLE's, so it holds at every stand-off; the lateral stays the robot's, because the right arm's reach and the flip approaches are
  set by the centre line (`_board_position_note`: right of the grid it blocks the flips, left of the centre line the arm misses the
  slots). The listed positions are `generation.board_table_xy_candidates`, else the base list at its own stand-off (inset = depth -
  0.35; the same table's edge overhang cancels); with BD_BOARD_SEARCH the table lattice follows (insets 0.06-0.30 step 0.03 x laterals
  -0.22..+0.06 step 0.04, nearest first to the first kept listed position, ties right first). Per candidate stance a position is kept
  only when the plate lies on the top (1 cm margin) and the board centre's base depth is in `generation.board_table_depth_window_m`
  (default 0.40-0.66: the list's nearest at its stand-off .. the grid's farthest). The lattice positions get the grid's treatment (CPU
  tests before each probe, `board_grid_max_probes`). One function, `_board_plan(manip)`, is the position plan of `_place_board`,
  `_board_precheck` (hence BD_BOARD_SEARCH's candidate order); the init's flip-room move (`_board_fit`) keeps the base list either
  way (2774e392, see the first run below): i7-startdraw's 5o.1 flip-room range can call `_board_fit` / `_board_plan` unchanged.
  Example (unit test, rhjoby-like top, stand-off 0.45): the list becomes (0.56, -0.02), (0.54, -0.10), (0.58, +0.04), (0.62, -0.06),
  and (0.70, +0.05) is out of the window; at 0.35 it is the base list. Counters `bd_board.table_plans / table_listed /
  table_listed_off_top / table_listed_out_of_window / table_lattice / table_lattice_off_top / table_lattice_out_of_window /
  table_listed_placed / table_lattice_placed / table_failed`.

Runs (cluster svl titanrtx, N 4 v2only, alias memory off, two replicates per arm at 320100-103 / 320200-203; every job with
`ISGF_COLLECT_EXTRA='--max-steps 6000'` from 5m.1, the collector logs `max_steps: 6000`; `$OFH/blockdesign_20261108/submit_5m.sh`):

| arm | snapshot | att / written / ok | rejections | written attempt (s) | GPU-h | GPU-h / success |
|---|---|---|---|---|---|---|
| bdv_grid (defaults: list + SEARCH grid; = 5m validation) | trunk_f1fc4ac1 | 8 / 8 / 6 | none | 345-878 | 1.93 | 0.32 |
| bdv_table (TABLE=1, flip-room move on the table positions) | trunk_f1fc4ac1 | 120 / 0 / 0 | 120 board_placement | - | 0.48 | - |
| bdv_tableb (TABLE=1, flip-room move on the base list) | trunk_2774e392 | 16 / 5 / 5 (cancelled at 2 h 37) | 8 approach_untrackable + 3 gripper_not_in_frame (program_check, 0.87-1.29 ks each) | 818-1808 | 5.23 | 1.05 |

* **5m validation (bdv_grid, the defaults after 5m.1 / 5m.2):** every instance written on attempt 0 (candidate 12, stand-off 0.45, the
  board at the grid's (0.51, -0.10) after 3 grid probes; 5 probes per attempt); 6 / 8 demos succeed at 3314-3539 action steps (all
  past the old 3000 budget). Failures: 320101 symbolic_only (one die 4.0 cm off slot_2, rotated pi: the same instance was 4.0-4.5 cm
  off at the 3000-step timeout in bd_fix and succeeded in the 6000-step recollect, so it varies between runs), 320201 not_reached
  (layout error 3.9 mm at the closest approach). Neither diagnosed further.
* **bdv_table (first table run):** the table positions also drove the init's flip-room fit; the minimum move (init 0.35 -> 0.20
  instead of 0.07) then "fit" every stand-off, the manipulation stances stood at the lateral range's end, 3 of 24 candidates were
  board-feasible and all 120 attempts ended board_placement (precheck: no flip set, grid none). Fixed in 2774e392 (the flip-room
  move keeps the base list); the result is evidence for i7-startdraw's 5o.1: the move to 0.07 is what gives the workable candidates.
* **bdv_tableb (the A/B arm):** the same 27 candidates and the same order as the grid (12, 15, 11, 17, 19, 18). Candidate 12 (stand-off
  0.45, trunk 0.95) fails board_placement: its listed table positions (0.56, -0.02) / (0.54, -0.10) / (0.58, +0.04) / (0.62, -0.06) are
  0.10 m deeper than the base list and all out of the tall trunk's reach (the probe passes only at depth <= 0.54 and lateral <= -0.10,
  bd_diag2), and the lattice, ordered nearest to the first listed position, spends the 12-probe cap before (0.51, -0.10) - the
  position the grid finds on its 3rd probe because its order starts near the edge. Candidate 15 gets (0.51, -0.10) from the lattice and
  fails path_collision; candidate 11 board_placement; candidate 17 (stand-off 0.35, the list's own stand-off, the board at (0.46,
  -0.02)) writes, but its program check rejects 11 of 16 attempts (approach_untrackable, gripper_not_in_frame at the place descent).
  31 probes per attempt vs 5. Its 5 written demos all succeed (3004-3525 steps).
* **Verdict: the table-relative positions are worse than BD_BOARD_SEARCH's grid** (5 written / 16 attempts and 1.05 GPU-h per success
  vs 8 / 8 and 0.32). Holding the board's inset from the edge constant moves it away from the base as the stand-off grows, while the
  reach that limits the board is base-relative; the hand-set list worked at 0.35 because there inset and reach coincide. The switch
  stays default off (no default change asked); the owner decides whether to keep it.

## i7-diag (2026-11-08): the IORS large floor aliases and touching_04 (diagnosis only)

Owner request 2026-11-08 ("two diagnoses, no behaviour changes"). CPU tabulation with `scripts/oracle_fix/diag_floor_t04.py`
(`ssh minyeong@svl11 python3 - floor|t04|pdc <tags> < scripts/oracle_fix/diag_floor_t04.py`). The only GPU runs are integration-tip runs
of the aliases that had none (tag `i7diag_floor`, trunk_fcddc495, `--grasps v2only`, `$OFH/v2maps_int1`, N 4, memory off, index 330000).
The record is `$OFH/DECISIONS.md` ("i7-diag" lines).

**A. Floor aliases.** All 12 shiyuc floor-fix jobs (i7_floor, i7_floorfix) COMPLETED. The 0-attempt ones are real setup rejections. The 7
long ones (3-6.4 h) finished after the copy-back and had never been copied; they are copied now with `sy_copyback.sh`. `tab_i.py` shows 0
attempts for running jobs because `job_stats.json` is only written at the end.

| alias | latest finished run | att / wr / ok | GPU-h / success | blocker | covering branch |
|---|---|---|---|---|---|
| under_large_01 (sandal, bed) | int1_1 | 5 / 4 / 3 | 0.27 | none (1 robot_wobble) | - |
| under_large_02 (briefcase, desk) | int1_1 | 6 / 4 / 0 | - (3.22 GPU-h) | oracle push-under: not_reached / symbolic_only, briefcase 0.11-0.23 m short and turned | none |
| under_large_03 (trash can, desk) | i7diag_floor (fcddc495) | 21 / 4 / 0 | - (4.00 GPU-h) | oracle push: 2 symbolic_only (ashcan turned 0.38-0.43 rad), 2 protected_contact (right_arm_link7 touches the can in the push pre-contact); i7_floorfix 19 / 3 / 1 | none |
| ontop_large_01 (soap dispenser, countertop) | int1_1 | 0 / 0 / 0 | - | missing_map tabletop h 0.9158 at countertop_fjkase_0 (entry soap_dispenser__zwudcz__s1, plane top): no anchor covers it, no tabletop map | i7-goalmaps |
| ontop_large_02 | i7diag_floor (fcddc495) | 18 / 4 / 4 | 0.84 | none (approach_untrackable 9 of 14 rejections) | - |
| touching_large_02 (umbrella, hall tree) | i7diag_floor | 0 / 0 / 0 | - | missing_map interior h 0.5809 at hall_tree_upkvgr_0 (entry umbrella__ziqzqo__s1): no registry spec of the board (the surface place) | i7-goalmaps if it covers interior goals, else none |

**B. touching_04.** int1_7 had 156 target_placement rejections: touching_04 94, touching_02 25, touching_01 22, axial_04 11, axial_01 and
axial_03 2 each. In touching_04 the candidates fail on the place depth. The start draw puts the spray bottle 0.56 / 0.64 m deep from the
table edge. At commercial_kitchen_table the base is collision-free only from stand-off 0.30 m, and from there the touch_push map reaches
only 0.25-0.44 m deep (at most 0.57 m). The failures appear as the A1 partner-draw gate "push stances fit the map but none collision-free"
(109 candidates in 70 of 94 attempts) and as the init grasp's empty F at 0.91-0.99 m ahead of the base. i7-contact3's cap (owner 3.6;
touching_04 is in its list) was A/B-tested. In the shadow arm, 214 of 369 candidates were over the cap; 139 of them were tried and none
was drafted. With the cap on, target_placement drops from 45 to 7 and each removed candidate costs about 2 s. But 27 of 53 attempts lose
all their candidates, and written stays 4 / 4 (0.56 vs 0.66 GPU-h per success). Proposed fix, within 3.6's scope and for i7-contact3:
limit the drawn start / place depth to the cap, instead of only filtering on it.

What the cap explains (shadow arm, kit counters joined per attempt, `scripts/oracle_fix/diag_t04_pdc_att.py`): 30 of 45 target_placement
attempts tried only over-cap candidates. Of 155 candidates tried, 123 were over the cap (all failed) and 32 under it. Applied to int1_7,
that is about 2/3 of its 94 target_placement attempts. The remainder, as seen in the on arm (7 attempts):

- the place from the push stance fails exact IK, all at trunk (0.25, -0.53);
- the A1 gate finds no collision-free push stance, all at trunk (0.0, -0.61).

Both are stance tilts the maps are not built at, so the inherited columns and the cap's reach are optimistic there. The largest remaining
cost is outside target_placement: lift_unreachable (7-9 per run, 74-303 s each). Each is the 0.31 m carry lift for the drive to the push
stance, which is only checked in goal_check. Proposed, none implemented: screen that lift at the draw; put the cap and the gate on the
same basis, or keep touching trunks on built tilts; draw the depth within the cap.

### i7-diag AUDIT (2026-11-08, owner 5m): where rejected attempts spend their seconds

CPU only, from the logs; no behaviour change, no GPU run. `scripts/oracle_fix/audit_early_reject.py`:
`scan` (on the cluster) writes one JSON row per attempt from each job's `sample.log` (the reject line's stage / reason / string, the
per-attempt `stages:` seconds, the start draw, the kit counters and kit seconds, counts of camera / place-IK lines) and one per job (sacct,
`job_stats.json`); `aliases` and `items` (local) tabulate it. The per-attempt seconds sum exactly to `job_stats.json` `attempts_s` in all
1,333 jobs that have stats. Record: `$OFH/diag_20261108/early_reject/` (`attempts_r4.jsonl` = every job with a `sample.log` newer than
2026-11-06 12:00 in both out roots, `aliases.tsv`, `items.md`, `seg_fridge.jsonl`, `seg_sink.jsonl`).

**Primary set:** every `int1_*` job, plus, per alias, the latest tag whose jobs all completed. That is 147 (tag, alias) pairs: 148 jobs,
99.8 GPU-h, 473 written, 383 ok. The owner's candidate runs (t03specs_T6, the eight fridge runs, the sink runs) are added per item.

- Rejected attempts take 55.3 h, which is 77.0 % of all attempt time (train85: 79 %).
- Split of the rejected seconds:
  - draw (reset + start draw): 12 %;
  - search (the rest of the sample stage): 79 %;
  - post-sample (apply / settle / goal / program / filter): 9 %.
- So moving a late check (goal_check / program_check) into a candidate's acceptance recovers at most that 9 %.
- The large savings need a cheaper predicate before the expensive search (feasible-first), a memo across attempts, or a time cap.

**The ranked list.** Ranked by estimated GPU-s saved per written demo of the item's runs. A job's sampling is its critical path, so
sampling seconds count as GPU seconds. Notes:
- "addressable" is the whole rejected attempts: an upper bound.
- For an alias with 0 written, the denominator is the 4 demos each job asked for (marked >=).
- The items overlap (CAP and FLOOR-memo share under_large_03's attempts), so the seconds do not add up.

| rank | item | runs | rejections | rejected s | est. saved s (rule) | written / ok | est. s / written (addressable) | proposed fix | status |
|---|---|---|---|---|---|---|---|---|---|
| 1 | BD-board: block_design_01 board_placement | sd_on | 43 | 4,976 | 4,868 (measured: BD_BOARD_EARLY 2.5 s per rejected attempt, bd_early) | 2 / 0 | 2,434 (2,488) | BD_BOARD_EARLY + BD_BOARD_SEARCH default (search wrote on attempt 1, bd_fix 4 / 4) | DECIDED 5m.2 |
| 2 | FLOOR-memo: IORS large floor - the same B init->manipulation path collision / C "no push chain" / C "carry leg failed" over one instance's attempts | i7diag_floor under_large_03 (330003 x9), ontop_large_02 (330002 x2, 330000 x1) | 12 | 16,123 | 11,076 (the repeats after an instance's first, each down to the alias's median written attempt, 130 / 319 s) | 8 / 4 | 1,385 (2,015) | the floor flow draws no new start per attempt (start draw 0 s), so attempts re-search the same B / C: memo dead B (same swept collision, ~(18.50, -8.99) in all of 330003's) and dead C (d, lambda) across attempts, as `generation.c_skip_dead_b` does within one; or redraw the floor init after 2 such attempts | new |
| 3 | AT03-rest: attachment_03 rest_unstable at settle_validate (184 s each) | int1_2 / 4 / 5, i7cork_ho_r2 | 62 | 11,419 | 10,778 (nudge ~10 s right after the rest draw) | 13 / 9 | 829 (878) | the existing rest-nudge test right after the rest draw | DECIDED 5n.2 (i7-cork) |
| 4 | T03-lift (owner): touching_03 opryhp_0 goal_check lift_unreachable, the lift with the book attached; 148 verified, 20.8 s each (goal 10.3, settle 6.9, sample 1.6, apply 1.3) | t03specs_T6 | 148 | 3,073 | 2,744 (post-sample) | 0 / 0 | >=686 (>=768) | the attached-lift IK at the candidate's acceptance. The alias writes 0: the lift geometry is the blocker, so 5k.2's straightening-lift (d) A/B comes first | owner candidate |
| 5 | IN02-j2 (owner): inside_large_02 arm joint-2 limit at the C place / pre-place. 268 / 350 C place failures in fridge4 + A / A2 / B / B2 verified; 841 joint-2 limit rejects in 144 attempts | i7_fridge4, i7i2_fridgeA / A2 / B / B2 / _def, i7i2_j2A, j2B | 102 attempts with a joint-2 reject | 25,744 | 8,617 (regression: 10.2 s per joint-2 reject, R2 0.22) | 32 / 26 | 269 (805) | the 50 mrad joint margin in the interior place map's IK (the map has none) + 5t's inside arm side. With ARM_SIDE (j2B) 65 rejects remain = 167 s / written for the map margin | owner candidate; 5t DECIDED |
| 6 | T0x-contact: touching_01 / 02 / 04 goal_check contact_unreachable | primary (10 pairs) | 133 | 8,175 | 6,840 (screen at the draw) | 33 / 25 | 207 (248) | screen at the draw against the push map | DECIDED 5r (switch) |
| 7 | IN03-cam (owner): inside_large_03 plate -> sink (bar), B pick "wrist camera within 5 mm" (2-4 mm clear of bar_egwapq_0) | i7i2_sink, i7i2_sinkdiag2, ch_sh | 191 camera lines in 6 attempts | 4,568 | 1,930 (regression: 10.1 s per line, R2 0.83, 15 attempts) | 12 / 12 | 161 (381) | i7-camhull's ISPATIALGYM_CAMERA_HULL_FALLBACK (ch_sh shadow: 27 / 27 narrow failures pass the hull) | owner question (shared gate) |
| 8 | T0x-lift: touching_01 / 02 / 04 goal_check lift_unreachable | primary | 59 | 4,388 | 2,369 (measured catch 0.61) | 33 / 25 | 72 (133) | ISPATIALGYM_TOUCH_LIFT_SCREEN default | DECIDED 5r.1 |
| 9 | CAP: attempts far past the alias's written ones, incl. the time_budget stalls | primary | 17 | 30,318 | 17,135 (budget max(300 s, 5 x the alias's median written attempt)) | 473 / 383 | 36 (64); under_large_03 6,939 s / 4 written, articulation_open_small_02 6,528 / 1, under_large_01 3,445 / 8 | an adaptive attempt budget from the alias memory's written-attempt times; the fixed 360 / 2400 / 3000 / 3600 s budgets stay as ceilings. Cuts 1 written attempt (under_large_01, 2,158 s) | new |
| 10 | ON02-track (owner): ontop_large_02 approach_untrackable at program_check (carry chain 7, approach 2); 9 of 18 verified | i7diag_floor | 9 | 5,531 | 120 (post-sample: the search to that C is spent either way) | 4 / 4 | 30 (1,383) | the program_check's tracked carry / approach chain in C's acceptance, with a memo. The alias's cost is its C search (104-1,533 s per attempt), so item 2 is the lever | owner candidate |
| 11 | U3-frame (owner): under_large_03 gripper_not_in_frame (goal_check grasp eef 3, program_check pre-place fingertip 3); 6 of 21 verified | i7diag_floor | 6 | 784 | 64 (post-sample) | 4 / 0 | 16 (196) | project the grasp eef and the pre-place fingertip at the B / C acceptance (same 5 % rule) | owner candidate |
| 12 | REPEAT: instance_filter repeats_eval | primary | 163 | 5,216 | 2,511 (post + half the search) | 473 / 383 | 5 (11); axial_02 368 s / written | the repeats-eval test right after the start / goal draw | new (5o.2 for axial_02) |
| 13 | FOCUS-F: target_placement ending on "empty feasible set F at the focus" | primary | 321 | 10,534 | 127 (only 31 / 321 repeat a focus cell + trunk in the job) | 473 / 383 | 0 (22) | no early check helps (new disagreement per draw); a map-side margin | information |

**Checks of the brief's facts.** All were checked against the logs. The 148, 9 of 18 and 6 of 21 hold. Two need correcting, and one
needs context:
- The sink's "over 30 min per attempt" was i7i2_sinkdiag3 300054/0: 1,836 s, 103 camera lines. That attempt **wrote**, and it ran with
  the CAMERA_BLAME probes on. Without them, the longest sink attempt is 1,173 s (i7i2_sink 300035/2, also written, 89 lines).
- inside_large_02's 349 place-IK failures count 350 in my join.
- Context for opryhp's 148: touching_03's latest round-2 arm (i7t03r3_c, 0 written) is dominated by a different rejection,
  target_placement "no library grasp reaches the fixed start pose" (8,199 s).

**Highest rejected seconds per written demo** (`aliases.tsv`):

| alias | rejected s per written | written |
|---|---|---|
| articulation_open_small_02 | 7,920 | 1 |
| under_large_03 | 3,335 | 4 |
| ontop_large_02 | 2,666 | 4 |
| block_design_01 | 2,487 | 2 |
| touching_04 | 1,968 | 8 |
| inside_04 | 1,632 | 6 |
| under_large_02 | 1,573 | 4 |
| attachment_03 | 1,560 | 13 |
| ontop_large_01 | 1,260 | 4 |
| under_large_01 | 1,180 | 8 |

Aliases with 0 written:
- touching_03: 8,930 s;
- touching_large_02: gm_t02_on, 5,876 s, F_init ∩ F_goal (5q.2);
- articulation_open_large_03 / close_large_03: handle_unreachable + repeats_eval, 5,233 / 3,843 s.

Implementing any item is a separate owner decision.

### i7-t03specs (2026-11-08, owner item 4): touching_03's bookcases opryhp_0 and otwukr_1

Branch `i7-t03specs` (from of-v2maps 93229922, not pushed); record `$OFH/t03specs_20261108`, DECISIONS "i7-t03specs" lines.

**Diagnosis.** touching_03's customs bind four support rows: kmuacd_0 at 0.84 (T1), opryhp_0 at 0.37 (T6), otwukr_1 at 0.39 (T7) and
0.72 (T8). Only kmuacd had an interior spec. No log showed the other rows rejecting `missing_map`: every round-4 touching_03 job ran the
kmuacd T1 template, which the registry row and the eval index pin (43 job dirs over 24 tags, ~6062 attempts, 1 `missing_map` line - the
stale kmuacd digest of i7_lean2). opryhp / otwukr ran only in round 3, under the legacy flow and without maps. Template jobs under
v2only on the lean root confirm the gap: each of T6 / T7 / T8 rejects at setup (0 attempts) with "no interior spec of
house_double_floor_upper/bookcase_<x> at <h> m".

**Measure option** (ee781f49). `interiors measure --lateral <m>` (`measure(lateral_m=)`, default None = unchanged) starts the wall,
rim and back rays in the bay at that offset from the raster edge's midpoint. otwukr_1's boards span two 0.30 m bays on either side of a
5 cm divider at the raster's centre. Without the option the rays start inside the divider and find nothing: width 0.66 (both bays) and
an open back, both wrong.

**Specs** (c4a6956a, `assets/interiors/interiors.json`; `test_interior_registry.py` pins the digests and resolves every touching_03
board to its spec). Measured in house_double_floor_upper, stand-offs 0.35 / 0.45 / 0.55 (kmuacd's), thickness 0.02 assumed (jaysra's):

| spec | digest | width | depth | headroom | back |
|---|---|---|---|---|---|
| bookcase_opryhp_0@0.3722 | b8577f909128 | 0.7213 | 0.3521 | 0.3301 | panel |
| bookcase_otwukr_1@0.3888 | 41a3fa72d143 | 0.3032 | 0.2811 | 0.2875 | panel |
| bookcase_otwukr_1@0.7204 | 71e59f596aa4 | 0.3034 | 0.2792 | 0.2948 | panel |

One spec per board (`spec_for_support` keys support + height). For otwukr the narrower bay is registered (`--lateral -0.19`). The
other bay is 0.3106 / 0.3112 wide with the same depth and headroom. A lean rest's compartment follows its lean-side wall, so only the far
wall moves, by 7-8 mm.

**Maps** (job_glib `GB_STAGES=maps GB_KIND=interior GB_INTERIOR=<name>`, both arms, the lean entry
`hardback__acbrnv__s1__lean_left_20_c5c708`, as kmuacd's). Feasible (cell, grasp) entries summed over trunks, d_e 0.35 / 0.45 / 0.55:

| board | right | left |
|---|---|---|
| opryhp 0.3722 | 423 (3 / 168 / 252) | 298 (0 / 85 / 213) |
| otwukr 0.3888 | 365 (0 / 142 / 223) | 412 (0 / 109 / 303) |
| otwukr 0.7204 | 771 (150 / 231 / 390) | 839 (130 / 208 / 501) |
| kmuacd 0.8354 (existing) | 706 | 689 |

The low boards are reachable only from bent-knee trunks and not at the 0.35 stand-off.

**Staged root** `$OFH/t03specs_20261108/v2maps_t03specs` = `cp -a v2maps_i7lean` (identical to i7-t03's `t03_20261108/v2maps_t03`) + the
6 maps. Jobs need a snapshot containing c4a6956a's registry.

**Validation** (trunk_c4a6956a, v2only, i7-t03's switches off, memory off): 
- Before (lean root, trunk_ee781f49), template jobs: T6 / T7 / T8 each reject at setup with missing_map (0 attempts).
- After (staged root), every row hits its interior map (missing_map 0):
  - T7 / T8 (otwukr, template jobs 18381023 / 18381024): then stop at the template's `shelf_width` check. At the v2 lean rest's
    20 deg (the only lean rest with grasps; the legacy key leaned 15 deg), the two books leave 0.09 / 0.10 m in the 0.30 m bay; the
    minimum is 0.12. Not changed (a customs / rule choice for the owner). Without a template these rows cannot run in the fast flow.
  - T6 (opryhp, fast flow N 4 `t03specs_T6` 18381022; validation registry `t03specs_20261108/registry_T6.tsv`, template from
    `runs/t03tip4_opryhp_s0`): interim 240 attempts, 0 written - goal_check lift_unreachable 104 (the 4 cm lift with the book attached:
    exact IK), target_placement 77, repeats_eval 26, ik_joint_limit 20, path_collision 9, gripper_not_in_frame 3, place_unreachable 1. The
    final line goes to DECISIONS (CPU job 18381965).
- Suite at ee781f49: 23 failed + 3 errors = the 93229922 baseline; interior tests at c4a6956a all pass.

### i7-inside04 (2026-11-08): inside_04's plan failures past the saucepot are the partner's side, not the new grasps

Owner request 2026-11-08 ("four blocked aliases", item 1): inside_04 went 4/4 -> 4/1 in i7-attachdef's validation (ad_int1 old entries
vs ad_v with outer_pinch + attach_face_retreat), the 3 failures oracle `plan_failed` (the protected re-plan past the saucepot), and the
outcome memory showed a pick bin at 0 / 5. Diagnose; hand to i7-attachdef if it is their cause, else fix sampler-side behind a
default-off switch. Record `$OFH/inside04_20261108` (scripts `grasps_of.py`, `all_runs.py`, `geom.py`, `transit.py`; tables
`rows.json`, `geom.json`, `transit.json`), DECISIONS "i7-inside04" lines.

**The entry.** `garlic_clove__xotobt__s3.724__tabletop_insert` (legacy key `garlic_clove__xotobt__s`): int1 grasps top_0 / top_1 /
top_3 (width 25.7 mm, pre-shape 22.9 mm per finger) -> rebuilt top_1 / top_5 (25.7 mm) and the new wide pinches top_8 / top_11
(47.8 mm, pre-shape 33.9 mm); maps right e7d2ec24 -> 7aa9789a, left ee514a32 -> a54f625e. The failed ad_v demos used top_8~r180
(right), top_8~r180 (left) and **top_1 (left, the same grasp in both entries)**; the ok one top_8 (right). ad_int1 and ad_v ran
different indices (300137-140 vs 300141-144), so their draws differ.

**Where it fails.** All 3 at the transit to the pre-grasp (grasp[3]): the straight joint ramp hits the table and the saucepot, the
transit is planned with the supports ignored, the protected-object check fires (clearance -0.044 / -0.015 / -0.029 m) and the
via-above re-plan still passes the saucepot.

**The cause: the saucepot on the working arm's side.** The template puts the saucepot beside the clove (same depth, 0.19-0.32 m to the
side); the partner side is drawn uniformly (A1) and the arm is fixed by the candidate (`arm_offer`: "only arm offered" in nearly every
demo). Over every inside_04 v2 demo in `$OFH/out` before ad_v (77 demos, 30 tags, the old entry):

| partner vs working arm | demos | ok | plan_failed | first straight ramp to the pre-grasp clear |
|---|---|---|---|---|
| on the arm's side | 58 | 31 | 22 | 0 / 58 |
| on the far side | 19 | 18 | 0 (1 arm_stutter) | 16 / 19 |

By trunk on the arm's side: 0.0 13 / 18, 0.25 17 / 32 (stand-off 0.35: 9 / 21), 0.50 1 / 8 (stand-off 0.35: 0 / 4); on the far side
trunk 0.50 7 / 7. ad_v drew 4 / 4 arm-side, 3 of them trunk 0.50 / stand-off 0.35 (0 / 3); ad_int1 3 arm-side at trunk 0.0 / 0.25
plus one far-side (4 / 4). P(at most 1 ok of 4 at the old rate 0.64) = 0.14. Legacy grasps: 9 / 10 ok (arm side 8 / 9, mostly
stand-off 0.45-0.55). So the regression is the draw, not outer_pinch's width or attach_face_retreat - not i7-attachdef's.

**The pick bin.** The memory2 replays flag (right, 0.25, -0.53, 0.35); the "0 / 5" is the first replay's family finding (left, 0.25,
-0.53, 0.35, top_3): i_base 300084, i_mfseed 300098 / 300099, i_ctl 300102, i_pol1 300112 - 4 plan_failed + 1 collision at the
pre-grasp transit, all arm-side - then 2 ok (300122 far side, 300130 arm side). In the left top_3 bin arm side 2 / 7, far side 1 / 1;
in the right bin arm side 5 / 10. The bin is a proxy for the arm-side draw (far-side draws in the same trunk / stand-off bins: 0.25 /
0.35 4 / 5, 0.50 / 0.35 7 / 7); it still occurs as such (ad_v 3 / 3).

**The switch (6d9445a9, default off).** `ISPATIALGYM_IORS_FAR_ARM` (`1` = relation `inside`; else a comma list of relations):
`IORSSmall.sample_instance` tries the attempt's candidates that offer the arm on the far side of the drawn partner side first (each
group in its drawn order; none removed), and a candidate offering both arms offers the far arm alone. Kit counters
`far_arm.{consulted, candidates_far, candidates_near, no_far, reordered, restricted, written_far, written_near}`; the draft summary
records `far_arm`. Off is bit-identical (`offered_arms` is the former inline filter). No oracle, planner, gate or map change. Tests
`ispatialgym/tests/test_iors_far_arm.py` (the real candidate loop on a stub).

**A/B** (Vast, both arms; trunk_6d9445a9, v2only, `$OFH/v2maps_int1`, `--alias-memory off --no-memory-aggregate`, per-arm ledgers,
the same indices g332000-3 + g332100-3): arm a `i7in04_a` (off), arm b `i7in04_b` (`ISPATIALGYM_IORS_FAR_ARM=1`).

| arm | attempts | written | ok | GPU-h / success |
|---|---|---|---|---|
| a (off) | 110 | 8 | 7 | 0.173 |
| b (far arm) | 175 | 8 | 6 | 0.219 |

**The switch does not act as intended.** Over arm b's 175 attempts: far-arm candidates 117 (near 536), no far candidate in 95
attempts, 56 both-arm candidates narrowed to the far arm, drafts written with the near arm 55 and with the far arm **0**. Every
far-arm candidate failed in the sampler - the place screening (`place_unreachable`), the place out of the image
(`gripper_not_in_frame`), empty F: the far arm cannot make the cross-body place into the pot. The far side happens only when the far
arm's place passes, which no ordering creates; the preference only adds attempts (+59 %). All 8 arm-b demos are arm-side (arm a 7 +
1 far side). **6d9445a9 is not proposed for adoption (it stays off).**

Arm-side outcomes by trunk including this A/B: 0.0 20 / 26, 0.25 20 / 37, 0.50 2 / 12 (ad_v 0 / 3). The trunk-0.50 arm-side cell is
an arm x trunk bin of the decided outcome-memory pick hook (i7-memory2), so no second rule is added. inside_04 on the current trunk
writes 7 / 8 and 6 / 8 here (ad_int1 4 / 4): nothing for i7-attachdef from inside_04. Suite: 23 failed + 3 errors, the base
93229922's set (3104 vs 3100 passed: the 4 new tests).

## Integration round 2 (2026-11-08, of-v2maps)

The owner's decisions of 2026-11-08 (`owner_answers_20261108f.md`, "Integration"): a new integration round on of-v2maps from 93229922, the
process of round 1. Each finished branch is folded at its last reported commit; a branch whose agent is still running is folded only after
its report. Nothing new is decided at a merge. Record: `$OFH/int2_20261108`, `$OFH/DECISIONS.md` ("integration-2" lines).

**Merges** (one merge commit each; snapshots `$OFH/code/trunk_<sha>`):

| # | branch (commit merged) | merge | notes |
|---|---|---|---|
| 1 | i7-memory2 (01ac6958) | bc8adc96 | outcome memory 2.1-2.4, production / validation stores, the pick hook in every family; clean |
| 2 | i7-vast (135c0cd4) | d1b3e9bc | scripts/vast_fast/ only; clean |
| 3 | i7-diag (d1f06313) | c9a6e88c | diagnosis scripts + docs; docs conflict (sections unioned) |
| 4 | i7-t03specs (3cc6a30f) | 04ee0673 | opryhp_0 / otwukr_1 interior specs in the registry, `interiors measure --lateral`; owner 5k.2; docs conflict |
| 5 | i7-inside04 (1a4d0604), docs only | 24dde33b, 1b5e5525 | CHERRY-PICKS of the two docs commits a0e41b4b / 1a4d0604; 6d9445a9 (ISPATIALGYM_IORS_FAR_ARM, owner 5h: dropped) is NOT in of-v2maps |
| 6 | i7-framing (18d78f83) | 6d9c7c15 | DEFAULTS fingertip pre-contact rule (3ada83b0, owner 5.2), CARRY_BAND with the torso-or-skip middle band (5a9c191e / 40e38218, owner 5s.1); CARRY_FRAME_CHECK / standalone CARRY_TORSO_LIFT off |
| 7 | i7-axialrest (64efe96b; carries i7-brace 9d2ce7e3) | 691086c4 | DEFAULT AXIAL_SETTLED_REST (dee19fcb, owner 5g); BRACE_PROBE / BRACE_PRE_POLICY off; i7-brace's TURN_COVER / OPPOSITE default (59f7967b) comes with i7-brace |
| 8 | i7-attachdef (5f81fc86) | ab50b82b | DEFAULTS outer_pinch + attach_face_retreat (044715ef, owner 4.1), outer_pinch_preshape (7ac55021 after 71b4d1fc / 599fdad9 / dbd8f4bd / d1ad9cb4, owner 5.1), PUBLISH_AFTER_SCREENS (917f7232, owner 5j) |
| 9 | i7-cork (1d92b089) | e1b91b94 | customs attachment_03 brace_search false (22a13e67, owner 4.3); the lying90 / hand-over / fallback / early-rest / min-turn routes all off; conflicts in candidates.py (attachdef's defaults + cork's finger_axis_window_m) and attachment.py (memory2's pick hook kept in cork's route draw) |
| 10 | i7-contact3 (015b815a) | 2698480b | DEFAULTS place-depth cap, pack brace removed, close_02 TOP_PRESS_ACCEPT, PLACE_DEPTH_CAP_DRAW, TOUCH_LIFT_SCREEN, TOUCH_A1_SCREEN, QA_BRACE_PROTECT_ALLOWED (the last two lead decisions while the owner is away); TOUCH_BUILT_TILTS off; not the push-line commit 4822246b |
| 11 | i7-axialrest (7d21c982; code 2de8f35d) | 0973a5f7 | brace pre call refuses on any failure; AXIAL_BRACE_PRE_POLICY DEFAULT (owner 5y.1) |
| 12 | i7-goalmaps (c54620d5) | fb031c15 | DEFAULTS goal-surface anchors, floor-datum fix, INTERIOR_FRONT_SIDE; hall tree spec; carry switches off; interiors.json = union of every spec, registry test knows every digest |
| 13 | i7-blockdesign (a2f89901) | cd38c3d1 | 6000 collector steps for block design, BD_BOARD_EARLY + BD_BOARD_SEARCH DEFAULT; BD_BOARD_TABLE off; submit_fast summary line keeps memory_mode + collect_extra |
| fix | (integration-2, lead decision #18) | 339fc2bd | the PUBLISH_AFTER_SCREENS pending summary in `instances/.pending/<iid>.summary.json`: the old `.summary.json.pending` name broke the 255-character limit for 235-240-character ids (disembedding_07); default unchanged |
| 14 | i7-startdraw (8e2de681) | f2468042 | DEFAULTS START_DRAW (owner 3.5), floor init redraws stance A only, ALL_EDGES (lead #12); BD_FLIP_ROOM_DRAW off; old blockdesign resolved toward a2f89901; the attachment start search factored with cork's hand-over view, `_start_args` per route |
| 15 | i7-ft2 (72ebd8a4; carries i7-interior2 eec4ca60, i7-camhull 9e697ac1) | 9b87ede1 | DEFAULT FINGER_TOUCH_HAND_HULLS (owner 7.1), awvzkn sink spec; interior2's owner 6.1-6.3 defaults; registry = union of every spec |
| 16 | i7-camhull (e939ade6) | 540478e0 | DEFAULT CAMERA_HULL_FALLBACK (lead #5); CAMERA_TOP_HULL_FALLBACK off |
| 17 | i7-contact3 final (ac2a682d) | 22ae887d | adds only PUSH_LINE_COM (4822246b, lead #9, default off) + docs |
| 18 | i7-interior2 (3749f69b) | 02ccd615 | INSIDE_ARM_SIDE + stance-C map cap for relation inside only, INSIDE_ARM_SIDE DEFAULT for inside (owner 5i.1 / 5t), opt-in joint_margin_rad; the margin maps not folded (lead #22) |
| 19 | i7-tray up to dae908a3 | 4054c3d7 | DEFAULTS LAYOUT_PAD_CHECK=all for fixed layouts (48fe31cc), GRASP_TURN_FILTER (owner 5u.1); NOT the release-sweep gate (686c1dcd / 3115595e, lead #23) nor its root |
| 20 | i7-axialrest final (2f25683f; code b4bcb3d0) | 1083d1f6 | AXIAL_BRACE_MAP_JOINT_MARGIN / AXIAL_BRACE_TILT_INTERSECT both off (lead #24); the margin maps not folded |
| 21 | i7-brace (be096dd7) | 68c6ffc2 | DEFAULTS axial TURN_COVER + OPPOSITE (owner 4), WALL_BRACE_MAP (owner 5w.2); load diagnostics off; contact_library / contact_maps conflicts kept both sides |
| 22 | i7-t03 (7ac9945a) | 2f157bbf | touching_03 rounds 1 / 2 / 5, every t03 switch off (verified); ik.py: camhull's camera rule + t03's contact rules both kept |
| 23 | i7-overpath (6eb0712d; code a55eea8a) | 6b7fcd6d | the inside_large_01 spine guard ON via its customs; OVERHANG_OVERPATH off (lead #29) |
| 24 | i7-t03 round 6 (63bde7f9; code d4f6975a) | 2a760c8d | the (d) pose-0 touch-rule fix behind SHELF_LIFT_STRAIGHTEN (off) |
| 25 | i7-stancememo (76bf33b5; lead #38) | 0ab12366 | DEFAULTS ATTEMPT_CAP (lead #33), START_DRAW_A_REDRAWS 8 (lead #36); STANCE_MEMO / FLOOR_REDRAW off |
| 26 | i7-t03 round 7 (37bd1888; lead #37) | 7c40cc28 | T03_PAIR_MESH, SHELF_PREGRASP_FRAME, SHELF_RELEASE_TRACKED, TRACKED_END_BLAME, all off |
| 27 | i7-brace up to 43472364 (lead #40) | 14bfa044 | the axial brace load geometry behind GC_BRACE_LOAD_GEOM (off; validator only); the 9 braceload maps layered |
| 28 | i7-artlarge up to a93d696c (lead #50) | 795f1ece | DEFAULT ARTLARGE_INIT_DRAW (the large articulation start from the whole Phase-1 set) |
| 29 | i7-artlarge up to 31c7c011 (lead #53) | 58431d18 | DEFAULT ARTLARGE_ALT_SKIP (speed, identical results) |
| 30 | i7-t03 up to 54800b01 (lead #56) | 0153ed46 | per-family switch defaults: the 9 t03 switches DEFAULT ON for touching_03 only (lead #39) |
| 31 | i7-artlarge r3 (087586fb; lead #54 / #57) | 3413d60a | DEFAULT DIP_LEG_SELF_MESH (all aliases: the dip gate's replacement leg refused at 5 mm link3 / link5 mesh clearance); DIAG_ARM_TRACE off |
| 32 | i7-floorpush up to dbd8bc13 (lead #59) | f5b89921 | DEFAULT 0.10 m strokes with re-aim for the pitched floor push; the palm test fix; variant switches off |
| 33 | i7-yield2 up to 757659dc (lead #64) | dff59cfc | DEFAULTS LIFT_PAD_SWEEP (the support lift refused when the pads miss the object), WALL_VIS_SCREEN (world_vertical draw screened by the settle visibility rule) |
| 34 | i7-yield2 r2 (d7723c33; lead #65) | 5b4f42b3 | DEFAULT GRIP_WAIT_NO_PAUSE (the gripper close waits are not counted as a pause) |
| 35 | i7-stancememo r3 (01bd4e3f; lead #45 / #63) | 50bb863e | SupportTop vectorized (the start-draw speed fix, identical verdicts); the attempt cap doubles after 2 consecutive cuts |
| 36 | i7-goalmaps up to 4bc9b600 (lead #67) | 470a83ca | touching_large_02 customs: carry_centred / carry_wide_path (lead #42); the root's umbrella entry unchanged (round 1, not umb2) |
| 37 | i7-yield2 r3 up to 183810f4 (lead #66) | 73faa60d | DISEMB_POS_REDRAWS switch (off at this merge) |
| 38 | i7-yield2 r3 default (d94c2394; lead #69) | f1bb8e24 | DEFAULT DISEMB_POS_REDRAWS 8 (disembedding redraws the start position when every work stance fails) |

i7-inside04 was cherry-picked, not merged and reverted: the far-arm switch never enters the trunk's history, so a later merge of a branch
that contains 6d9445a9 would show it as new.

**Checks per merge.** pyflakes on the changed files (no new messages). The full suite runs on a CPU node (`int2_20261108/job_suite.sh`:
the frozen snapshot copied node-local, trunk PYTHONPATH, 8 shards by test file): lambda's load (~170) could not finish it. Baseline
trunk_93229922 = 15 failed + 6 errors on the node (lambda's run of the same tree: 23 failed + 3 errors; the sets differ by environment,
so each merge is compared with the node baseline); merges 1 (bc8adc96) and 4 (04ee0673): the SAME 21 ids. Merges 2, 3, 5 change scripts / docs
only (covered by merge 4's run).

**The consolidated root `$OFH/v2maps_int2`** (`jobs/v2build/build_int2.py`, log `v2maps_int2/INT2.txt`; v2maps_int1 untouched): `init` =
cp -a v2maps_int1, `anchors.json` / `links.json` kept as `*.before_int2`. Whatever a layer replaces moves to
`glib|maps/_superseded_<reason>_before_int2/`. Layers:
- t03specs: the touching_03 lean entry `hardback__acbrnv__s1__lean_left_20_c5c708` (v2maps_int1 never held it) with its 8 interior maps:
  kmuacd 0.8354 r / l, and opryhp_0@0.3722 / otwukr_1@0.3888 / @0.7204 r / l (staged root `t03specs_20261108/v2maps_t03specs`).
- brace34b (with merge 7): the axial brace maps of `v2maps_int1_brace34b` (clear-turn + opposite-end library) for board_game, keyboard,
  magazine, paperback s1 / s1x1x2.5; the old ones superseded. Its touch_brace rebuilds wait for i7-brace's own fold.
- axialrest: the spoon tqqwys's settled-rest push_blade / axial_brace maps (axial_05), shipped with the settled-rest default.
- attachdef (with merge 8): the 25 entries of `attachdef_20261108/v2maps_ad2` that differ from int2 (owner 4.1 and 5.1 rebuilds), whole entry
  directories; the int1 teacup / lid builds stay (owner 5.2). The qtfzeq overlay follows the new keys in `$OFH/map_overlay` r3.
- i7-cork's staged root is not folded. Its extra cork entries serve only the default-off routes. They share the cork's model, so the sampler's
  entry lookup would see four candidates instead of one (lead: agreed, not folded). If the owner turns a route on, the entries are staged
  in `$SY/v2/v2maps_int1_cork`.
- umb2 (lead #67b): the umbrella's ladder round-2 entry (6 valid) + its 4 maps replace the round-1 entry; check_int2 ERRORS 0.
- goalmaps (with merge 12): ontop_large_01's countertop maps (soap dispenser h0.9158 r / l) with the anchor 0.9158, and the umbrella's hall-tree
  interior PLACE maps; new files only. The umbrella's round-2 entries wait for the lead.
- ft2b (with merge 15): its 4 luhkiz maps were keyed on int1's luhkiz entry, which ad2 replaced, so they are REBUILT against int2's entry
  (`int2_20261108/rebuild_ft2b.sh`, same inputs): sink 36,985 feasible cells (= ft2's round b), bar 1.0505 r / l; layers ft2b_sink / ft2b_bar.
- braceload (with merge 27, lead #40): the 6 axial_brace + 3 touch_brace maps re-validated under the load geometry fix, layered into int2
  (not a root swap) after the running touching jobs ended; check_int2 ERRORS 0.
- brace33_touch / brace34w (with merge 21): the touch_brace maps of gkpmii / dkhgxn / ntbrtz re-validated under the pusher fix, and the 8
  world_vertical wall_brace maps.

After merge 7 (trunk_691086c4): 77 entries, 146 maps, 18 contact maps, 20 interior, ERRORS 0. Suites of 6d9c7c15 / 691086c4: the baseline's 21 ids.

`jobs/v2build/check_int2.py` = check_int1 + the wall_brace rule (a wall map has no support anchor: its file-name height must equal the
recorded hinge height). After merge 23 (trunk_6b7fcd6d): 77 entries, 154 maps, 26 contact (8 wall), 24 interior, 51 links, ERRORS 0.

check_int1.py (reader side, trunk_1b5e5525): 77 entries, 146 maps, 8 attach, 16 contact, 12 prefix, 20 interior maps (0 spec mismatches),
51 links, ERRORS 0. v2maps_int1 on the same snapshot: 76 / 138, ERRORS 0.

**Validation** (`--grasps v2only` N 4 on v2maps_int2, overlay default, memory off as the branches ran; `tab_i.py -a <tag>`):
- int2_1 (trunk_1b5e5525): touching_03, the only alias whose v2only behaviour merges 1-5 change (the lean entry is now in the root).
  Memory2's defaults change nothing in a memory-off run (production store empty, the pick hook inert).
  Result: 320 attempts / 0 written, 0.70 GPU-h, as its branches' runs (i7_lean3 320 / 0, 0.69 GPU-h); interior map hits from v2maps_int2;
  rejections target_placement 154, no_free_edge 129, goal_check path_collision 21 (i7_lean3 148 / 129 / 31). No regression.
- int2_2 (trunk_691086c4): axial_05 / 01 / 03 / 04 at i7ar_def's indices, TURN_COVER=1 + OPPOSITE=1 set explicitly until i7-brace's default
  lands; vs i7ar_def 49 / 16 / 11 (0.093 GPU-h per success) and i7ar_pol0 51 / 16 / 13 (0.071).
- int2_2 result: 51 / 16 / 12, 0.083 GPU-h per success (axial_01 11/4/2, _03 6/4/4, _04 14/4/3, _05 20/4/3); i7ar_def 49 / 16 / 11, 0.093. No
  regression.
- int2_3 was cancelled 5 min in: the attachdef layer swapped entries while it started. Its three aliases rerun in int2_4.
- int2_4 (trunk_e1b91b94, own ledger = the branches' indices): under_small_03 / inside_04 / touching_02 (framing: ifr3 21/3/3 0.215, 32/4/4 0.192,
  ifr_t02_B 25/4/4 0.126), nextto_03 / 04, inside_01 / 02, disembedding_07, attachment_03 (ad_v51 + i7cork2_cur_r 25/4/4 0.164), disembedding_09
  (ad_race 11/4/4).
- int2_4 disembedding_07 crashed at its first written instance (OSError Errno 36: `<iid>.summary.json.pending` = 261 characters) - fixed in
  339fc2bd; int2_6 reruns disembedding_07 / _09 on it.
- int2_12 (trunk_0153ed46): touching_03 300100 48 / 4 / 4 on v2maps_int2 (the t03 defaults; before 320 / 0 / 0).
- int2_7 (trunk_540478e0, after int2_5): the startdraw set at 300500, under_large_01 (floor watch), camhull's inside_large_03 / under_small_03
  at 335000, the sinkbreakfast variant at 300700.
- int2_5 (trunk_cd38c3d1, after the goalmaps check): touching_02 / 04 and close_02 at i7-contact3's indices, axial_05 / 01 / 03 / 04 (i7ar_pd
  49/16/12, 0.075), block_design_01 at 6000 steps (bdv_grid 8/8/6), ontop_large_01 (gm_after 14/4/3), touching_large_02 (known 0 written).

### i7-framing (owner decisions 2026-11-08 5.1 / 5.2): the lifted carry, pre-grasp / pre-contact by the fingertip rule

The owner's decisions (verbatim intent, tmp/outputs/owner_answers_20261108.md): **5.1** also check framing at the LIFTED carry pose (after
the oracle's protective lift); if only the arm lift takes the gripper out of the frame, consider lifting the torso (changing the tilt)
with it. **5.2** the pre-grasp and the touching pre-contact framing use the fingertip rule (the pre-place's function and thresholds); count
first (verdicts that change both ways), the default flips in a separate commit; report first if it admits poses whose fingertips are
framed but whose object is not. The torso change in the oracle is approved as the owner's suggestion behind its own switch, A/B before any
default.

**Where the protective lift is computed.** `oracle/execute.py`, the place's straight carry leg: the leg's tracked path (the carried object
included) is plan-checked against the protected goal objects (`oracle.protection`, `motions.protecting`); on a protected verdict
(`stage == "protected"`) the caller re-runs it via above: the hand rises straight to `z_top = max(lift end z, pre-place z) +
MotionsConfig.protect_above_m` (0.12), crosses at `z_top`, descends onto the pre-place. Deterministic from the recorded poses, so the
sampler can predict it exactly (`program_check.via_above_carry_poses`); whether it fires is decided at run time (clearance vs threshold).
**Diagnosis of 300150** (i7pq2_us03, CPU replay): the straight carry is framed (worst fingertip margin 34 px of 1080), the via-above leaves
the image at u 1034, depth 0.50 m (the oracle logged u 122 of 128 at 0.50 m) - the gripper leaves the frame ONLY because of the arm lift.
**Why a torso lift fixes exactly that.** The head camera (`zed_link`) and both arms hang from `torso_link4` (`conventions.torso_fk`), so
the hand's image position depends on the arm joints alone: a rise made by the trunk with the arm held leaves the image unchanged. The JoyLo
translate keeps `torso_link4`'s orientation (<= 0.005 rad over 0.25-0.5 of translate): 300150's trunk 0.75 -> 0.50 lifts the hand 0.14 m
with 1.6 cm sideways; the tilt turns it (0.2 rad per 0.2 of tilt), so it comes second.

**5.2 switch** (`ISPATIALGYM_PRECONTACT_FRAME_RULE` = box | fingertip; addb8606 default box): `IORSSmall.require_in_frame` judges
`PRECONTACT_FRAME_POSES` (pre-grasp, push pre-contact, brace pre-contact) by `program_check.fingertip_frame_margin` (QA's 5 % + 2 cm slack)
under `fingertip`; the brace-drop check too; the contact poses keep the eef box (as the program check). Both verdicts are always computed:
kit `precontact_frame.<pose>.checked` / `.removed` (box fails, fingertip passes) / `.added` (box passes, fingertip fails) /
`.removed_object_out` (removed, and the object - the target at the pre-grasp / push, the partner at the brace - not wholly in that image).
**Count** (`$OFH/framing_20261108`: `count_precontact_logs.py` over every sample.log under `$OFH/out`; `replay_all.py` over the 1610
written IORS small sidecars): box rejections ever logged - pre-grasp 12 (all touching_02, 8 runs), push / brace pre-contact 0, brace drops
0; on accepted poses box-pass -> fingertip-fail 0 of 1610 pre-grasps, 294 push and 210 brace pre-contacts (the box contains the
fingertip's 2 cm ball). The logs hold no poses, so the object-framing question is answered by the runs' shadow counter (below).

**5.1 switches** (addb8606, both default off). `ISPATIALGYM_CARRY_FRAME_CHECK=1`: `IORSSmall.carry_frame_check` after the pre-place
check - the via-above poses (8 per leg) by the fingertip rule from the manipulation stance; out -> `Reject("gripper_not_in_frame",
"lifted carry ...")`; a place made from another base pose (touching's push stance: a carried drive first) is skipped. Kit
`framing.carry.checked` / `.failed` / `.skipped_drive`. `ISPATIALGYM_CARRY_TORSO_LIFT=1` (with the check): where the straight carry is
framed (the arm lift alone leaves the frame), `program_check.carry_torso_choice` (translate alone first, then with tilt; >= 0.12 m rise
at both ends, <= 4 cm sideways, <= 0.1 rad turn, joint limits) and `IORSSmall.carry_torso_lift` (the object attached: the straight leg
moved with torso_link4 as a tracked exact IK from the carry-trunk stance and framed; the two trunk sweeps, arm at the lift-end / pre-place
configuration, whole body with the supports out, 4 steps each) -> `grasps[].carry_torso` {translate, tilt, from, rise_m, shift_m,
rot_rad}; kit `framing.carry.torso_lift_used` / `.torso_lift_failed.<no_trunk|no_lift_q|frame|leg_ik|sweep_collision>`. Oracle (same
switch, only with the record, only on the protective via-above): `execute._carry_torso_lift` - `trunk_to(carry_torso)` arm held, the
recorded straight carry moved by `Mt` = torso_link4 after o before^-1 (finish: arm_to its end with the pre-place witness at the carry
trunk), `trunk_to` back; `ctx["carry_leg"].kind = "straight via torso lift"`. **Count** (CPU replay, 1367 carries without a carried
drive): the via-above leaves the frame in 100 (inside_04 37, nextto_02 17, inside_03 14, ontop_small_04 8, ontop_small_01 6, nextto_03 6,
under_small_03 4, under_small_04 4, inside_01 2, inside_02 2), the straight carry framed in all 100, torso geometry available in 49; the
demos of those 100: 73 succeeded (the via-above did not fire), 1 gripper_out_of_fov (300150) - a hard check costs ~7 % of written
instances, three quarters of them good demos. Tests `test_framing_carry.py` (13; 300150's numbers replayed).

**Default** (3ada83b0, owner 5.2 decided): `ISPATIALGYM_PRECONTACT_FRAME_RULE` defaults to fingertip (`box` = before). The 5.1 switches stay
off (no owner default).

**Validation** (trunk_addb8606, shiyuc, `--grasps v2only` N 4 on a staged copy of `$OFH/v2maps_int1` (identical files), alias memory v0002
pinned in every arm, titanrtx; one ledger per arm, the same base index; `$OFH/framing_20261108/tab_fr.py`):

| arm | att / written / ok | GPU-h / success | counters | out-of-frame demo failures |
|---|---|---|---|---|
| us03 A (box, carry off) 18379235 | 27 / 3 / 2 | 0.622 | precontact pre_grasp checked 5, removed 0, added 0 | 0 (via-above fired 3x) |
| us03 B (fingertip + carry check) 18379237 | 43 / 4 / 3 | 0.497 | carry checked 9, failed 5 (= 5 gripper_not_in_frame rejections); precontact removed 0 / added 0 | 0 (via-above fired 3x) |
| us03 C (B + torso lift) 18379239 | 32 / 3 / 2 | 0.622 | carry checked 5, failed 3: torso_lift_used 2, torso_lift_failed.no_trunk 1 (rejected) | 0; torso carry executed 1x (trunk (0.25, -0.53) -> (0.05, -0.43), hand back 1.4 mm from the pre-place, demo ok) |
| touching_02 A (box) 18379241 | 25 / 4 / 4 | 0.153 | pre_grasp / push / brace pre-contact checked 4 each, removed 0, added 0 | 0 |
| touching_02 B (fingertip) 18379473 | 25 / 4 / 4 | 0.126 | idem (identical sampling to A) | 0 |

Before (i7pq2_us03, trunk_14634552): 36 / 4 / 3, 0.455, 1 gripper_out_of_fov (300150, the via-above carry). The us03 failures in A / B / C
are one plan_failed each (place[5]: the left arm's finishing plan swings a joint 1.08-1.09 rad), not framing. 5.2 never acted in these
runs (0 removed, 0 added, 0 removed_object_out) - consistent with the count; no pose with framed fingertips and an unframed object was seen.

### i7-framing round 2 (owner decisions 2026-11-08 5d.1-5d.3): the torso lift's runs and the middle option

The owner's decisions: **5d.1** `ISPATIALGYM_CARRY_FRAME_CHECK` stays off (not a default). **5d.2** `ISPATIALGYM_CARRY_TORSO_LIFT` stays off;
more runs - N 4, torso on vs off, same indices / backend, on under_small_03 and the aliases where the replay found the lifted carry
leaves the frame (inside_04, nextto_02, inside_03, ontop_small_04); report how often the lift fired, torso lift used / failed by reason,
out-of-frame failures, att / written / ok, GPU-h per success. **5d.3** a middle option behind a new switch (default off): from the
nominal straight carry compute its clearance to the protected objects against the oracle's lift trigger; far below (the lift will
almost surely fire): the lifted-carry framing check; far above: skip; near: check or the torso lift (say which); band edges from the
replay; then an N 4 A/B (middle option vs off) on under_small_03 + the replay's top aliases; kit `framing.carry.band.{below,near,above}`
/ `.rejected` / `.torso_lift_used`; unit tests (switch off = today, far above never rejected, far below out of frame rejected).

**The trigger** (`oracle.protection.check_trajectory`, `Motions._protect_violations`): the tracked straight carry's configurations ->
cuRobo spheres of the moving arm and the attached object -> the smallest signed distance to each protected object's base-aligned box
(the goal objects the task placed, minus what is in the hand: IORS small = the partner); the threshold `min(0.02, start - 0.002, end -
0.002)` (endpoint-relative), the leg fires when its interior dips below it. **Firing history** (`fired_carry.py`, 1715 IORS small
demos' collect logs): under_small_03 28, ontop_small_02 19, touching_01 63, touching_04 12, touching_02 1 - never in inside_04 /
nextto_02 / inside_03 / ontop_small_04, the aliases where the lifted carry would leave the frame.

**The nominal margin** (`v2/carry_margin.py`, f0cc5c48): the same rule on a cheap proxy - the carried object's base-aligned box posed by
the grasp's `eef_in_object` (the hand box left out: every logged carry firing, 42 / 42, was the attached object's), every 1 cm of the
straight leg lift end -> [vias] -> pre-place, against the partner's box, every clearance less `SPHERE_OVERHANG_M` = 9 mm (cuRobo's sphere
fit overhangs the mesh; the offset with the best separation). **Replay** (`margin_replay.py` + `band_table.py`, 1447 carries / 1423
demos / 42 fired; the boxes from B1K metadata + the template scale): AUC 0.893 (0.825 without the offset); the fired carries all lie at
+1.3..+8.5 mm; above +9 mm 0 of 896 fired; below +2 mm 20 of 131 (15 %), +2..+9 mm 22 of 396 (6 %). The nominal margin predicts
"will NOT fire" well and "will fire" never surely (the trigger is a 2 mm endpoint-relative test; the proxy's noise is of that order).
**Edges** `CARRY_BAND_BELOW_M` +2 mm, `CARRY_BAND_ABOVE_M` +9 mm. **Near: the torso lift when possible, else the check** - the
recorded torso lift costs nothing when the lift does not fire (the oracle runs it only on the protective via-above) and keeps the frame
when it does; without it the camera-frame rule decides (reject). **The 100 out-of-frame carries** by band: above 65 skipped (49 good
demos, 16 other failures), below 4 rejected (3 good + 300150, the out-of-frame failure), near 31: 15 torso lift (10 good) + 16 rejected
(11 good) - 20 rejections instead of the hard check's 100.

**Switches** (f0cc5c48): `ISPATIALGYM_CARRY_BAND=1` (`IORSSmall.carry_frame_check` -> `carry_nominal_margin` -> `PC.carry_band`; the
oracle executes a recorded `carry_torso` under it too); `ISPATIALGYM_CARRY_TORSO_LIFT=1` alone = the torso-only mode (records the torso
lift wherever only the arm lift leaves the frame; never rejects: `framing.carry.accepted_out`). Kit `framing.carry.band.{below,near,
above}` / `.band.rejected` / `.band.torso_lift_used`. Tests `test_framing_carry.py` (+8: switch off = today, the edges, far above never
rejected, far below out of frame rejected, near torso else check, torso-only never rejects, the margin geometry and the oracle constants).

**Runs** (round 2; trunk_f0cc5c48, shiyuc, `--grasps v2only` N 4 on the staged int1 copy, titanrtx, alias memory off, one ledger per
tag with the same base index per alias; `$OFH/framing_20261108/round2_tab.txt`, `tab_fr.py`). att / written / ok, GPU-h per success;
lift fired = the oracle's carry protect check (collect logs); torso executed = `execute._carry_torso_lift` ran:

| alias | off | torso (5d.2) | band (5d.3) |
|---|---|---|---|
| under_small_03 | 33 / 3 / 3, 0.412; lift 2 | 21 / 4 / 3, 0.332; lift 4, torso used 2, executed 1 (demo ok) | 8 / 4 / 4, 0.176; lift 3; band below 1 / near 2 / above 1, torso used 1, executed 1 (ok), rejected 0 |
| inside_04 | 50 / 4 / 4, 0.282; lift 0 | 51 / 4 / 2, 0.594; lift 0; carry out 7: torso used 1, no_trunk 6 (accepted) | 42 / 4 / 3, 0.349; lift 0; above 4 / near 3, rejected 2 (no_trunk) |
| nextto_02 | 27 / 4 / 4, 0.216; lift 0 | 27 / 4 / 4, 0.140; lift 0; carry out 0 | 27 / 4 / 4, 0.129; above 2 / near 2, rejected 0 |
| inside_03 | 21 / 4 / 4, 0.128; lift 0 | 21 / 4 / 4, 0.119; carry out 3, torso used 3 | 21 / 4 / 4, 0.103; below 1 / near 1 / above 2, torso used 1 |
| ontop_small_04 | 6 / 4 / 4, 0.108; lift 0 | 6 / 4 / 4, 0.109; carry out 2: torso used 1, no_trunk 1 | 6 / 4 / 4, 0.108; below 1 / near 3 / above 1, torso used 1 |

Out-of-frame demo failures: 0 in all 15 arms. The other demo failures: inside_04 torso 2 protected_contact (the pre-grasp transit,
before any carry: the finger on the saucepot), inside_04 band 1 robot_wobble, under_small_03 torso 1 plan_failed. The protective lift
fired only in under_small_03 (as the replay's history says: 0 in the other four aliases); the torso carry executed twice (both demos
ok, the hand back within 1.4 mm of the pre-place in the first run). Defaults unchanged; the owner decides.

### i7-framing 5s (owner decisions 2026-11-08c): the middle band skips, ISPATIALGYM_CARRY_BAND the default

The owner's decisions: **5s.1** `ISPATIALGYM_CARRY_BAND` becomes the default, with one change to its middle band (+2 to +9 mm): use the
torso lift when possible, otherwise SKIP the check, do not reject (the lift fires in ~5 % of that band's carries, 22 of 410); only
far-below carries (< +2 mm, 15 % fired) are rejected when they leave the frame. Commit the band change, re-run the replay numbers, then
the default commit. **5s.2** the standalone `ISPATIALGYM_CARRY_TORSO_LIFT` stays off (nothing done).

**The band change** (40e38218): `IORSSmall.carry_frame_check`, band `near`, the lifted carry out of frame and no torso lift (`no_trunk`
/ `frame` / `leg_ik` / `sweep_collision` / `no_lift_q`, or the straight carry itself out of frame): accepted, kit
`framing.carry.band.near_skipped` + `.near_skipped.<why>`, summary `carry_frame.near_skipped`. `below` and `above` unchanged.

**The replay** (`$OFH/framing_20261108/band_table_5s1.py` -> `band_table_5s1.out`; the inputs, margin and torso proxy of
`band_table.py`): of the 100 out-of-frame carries **4 rejected** (5d.3: 20) = **3 good demos** (5d.3: 14) + **300150** (under_small_03,
margin +1.5 mm, band below: still caught); near 31 = 15 torso lift + 16 skipped (11 good, 5 other failures; none of the 16 had its lift
fire); above 65 skipped (unchanged).

**The default** (5a9c191e, its own commit): `program_check.carry_band_on` and `execute._carry_torso_record` read
`ISPATIALGYM_CARRY_BAND` with default `1` (`0` = the rule before: the lifted carry not looked at, no recorded torso lift executed). The
torso-only mode (5d.2) is now `CARRY_TORSO_LIFT=1` with `CARRY_BAND=0`. Tests: the band is the default and `0` turns it off; near
without a torso lift skips; the oracle executes the band's torso lift by default. Suite: 23 failed + 3 errors = the base set (the 40e38218
run had 5 more, test_split_cone's FileNotFoundError on the stray `tabletop__s__t__apple-x__k.json`; they pass alone).

**Confirmation** (trunk_5a9c191e, shiyuc, titanrtx, v2only N 4 on the round-2 staged root, alias memory off, no extra env = the default;
the same base indices as round 2; `$OFH/framing_20261108/round3_tab.txt`). att / written / ok, GPU-h per success:

| alias | round 2 band (5d.3) | default (5s.1) |
|---|---|---|
| under_small_03 (620000) | 8 / 4 / 4, 0.176; below 1 / near 2 / above 1, torso used 1 (executed, ok), rejected 0 | 21 / 3 / 3, 0.215; below 1 / near 2 / above 1, torso used 1 (executed, ok), lift fired 2, rejected 1 (below: 620002 attempt 7, cross 8/8 out of frame) |
| inside_04 (620100) | 42 / 4 / 3, 0.349; near 3, rejected 2 (no_trunk) | 32 / 4 / 4, 0.192; above 4 / near 2, **near_skipped 2 (no_trunk), rejected 0**; lift fired 0 |

inside_04's 2 near-band rejections are gone (the same two no_trunk cases now skipped). Out-of-frame demo failures 0 in both.
under_small_03's 620002 wrote nothing (15 attempts: target_placement 17 - grasp / place reach - and the one below-band reject); the
near band did not act there (near_skipped 0), so the 8 -> 21 attempts are N 4 sampling variance, not the change.

### i7-axialrest (owner decision 2, 2026-11-08): the axial program at the target's settled rest

Owner decision, verbatim prompt `prompt_goalmaps_axialrest_overpath.md` section 2 (DECISIONS 2026-11-08 "OWNER DECISIONS"): spawn the axial target
at its v2 settled rest, compose the goal yaw with that rest's tilt, build the axial library at the settled rest, and compare the goal in that
frame. It is approved as a program and goal-definition change. Switch `ISPATIALGYM_AXIAL_SETTLED_REST=1`, default off (4d22896d); the merge of
i7-brace 9d2ce7e3 (6ab54ba7) supplies the clear-turn axial_brace builder.

**Diagnosis.**
- The success check (`pose_reached`, geometric success, QA `not_reached`) compares the full quaternion with the TRO target. Before the
  switch that target was the upright `R_z(yaw_goal)`.
- The evaluation tolerance comes from the isg-v2 profile: max(0.05, 10 % of the change), 0.05-0.117 rad for these demos, not the TRO's
  0.10.
- The oracle's push stops use the yaw about z, so the tilt does not affect them.
- The 19 past axial_05 demos: 2 succeeded, 14 not_reached, 3 plan_failed. The spoon settles 0.087-0.093 rad tilted; the v2 rest r0_other
  is tilted 0.0866.
- Of the 14 not_reached:
  - 7 reached the goal heading within 0.0055 rad and failed on the tilt alone;
  - 5 fell 0.087-0.18 rad short of the turn;
  - 2 lost the target.
- The 2 successes turned more than 0.92 rad, so their tolerance exceeded the tilt.

**What the switch changes.**
- **Placement:** `R_z(yaw0) * q_rest`, with the root at the support top + `h_root`. The rest is the push_blade map's `object.rest`, the
  v2 entry the library was built and validated at.
- **Goal:** `R_z(yaw_goal) * q_rest`. After the settle the goal is the settled orientation turned about the vertical to the aligned
  heading, written into the TRO target. The full check then measures the turn alone, because the hinge keeps the tilt.
- **Sites changed (identity / upright assumptions):**
  - `band_spec.rest_quat`;
  - the placement draw;
  - `after_settle`'s goal;
  - `_aligned_yaw` (the front axis at the rest);
  - the template extents and height;
  - `axial_stop_geometry`'s goal pose, and the stop points flush with a tilted face's outermost projection;
  - the opposite-end brace;
  - `contact_library.axial_rest`: under the switch it returns the v2 rest (family param `settled_rest`, in the library identity only
    when on);
  - the start pose of the 45 deg build, the clear-turn build and `clear_turn_max`.
- **Map rest check:** a map whose entry rest is not the program's rest is a miss. The push rejects `missing_map`; the optional brace is
  skipped.
- **Counters:** `axial_rest.settled_used`, `.no_v2_rest`, `.map_rest_mismatch.<kind>`; `summary.axial_rest` records the tilts and the
  turn.
- The validator already settles at `R_z(yaw) * v2 rest` and judges the turn from the settled start, so the library entries and the
  program now stand on the same rest.

**tqqwys library and maps** (`$OFH/axialrest_20261108`):
- push: 85 valid of the validated entries;
- axial_brace (clear-turn build): 48 valid, 12 per end x spin group;
- both maps are at h0.8126;
- staged in `$OFH/v2maps_int1_axialrest` (= brace34b + the two maps).

**A/B:** tags `i7ar_off` / `i7ar_on`, trunk_4d22896d, svl titanrtx, v2only N 4, the same indices 310000-310003, root
`$OFH/v2maps_int1_axialrest`. Both arms run with TURN_COVER=1 and OPPOSITE=1; the switch is the only difference. Full record: DECISIONS
"i7-axialrest RESULTS" (2026-11-08).

| alias | off att / wr / ok | off GPU-h per success | on att / wr / ok | on GPU-h per success | brace present (off / on) |
|---|---|---|---|---|---|
| axial_05 (tqqwys, tilt 0.0866) | 0 / 0 / 0 (missing_map) | - (0.09 GPU-h) | 30 / 4 / 2 | 0.17 | - / 0 of 4 |
| axial_01 | 11 / 4 / 2 | 0.17 | 12 / 4 / 3 | 0.090 | 2 / 1 of 4 |
| axial_03 | 6 / 4 / 4 | 0.048 | 10 / 4 / 4 | 0.050 | 2 / 3 of 4 |
| axial_04 | 21 / 4 / 4 | 0.055 | 14 / 4 / 2 | 0.090 | 0 / 0 of 4 |
| all | 38 / 12 / 10 | 0.083 | 66 / 16 / 11 | 0.090 | |

- **axial_05 off:** 0 attempts, as expected. The setup is rejected `no_stance` because both edges are `missing_map`: the settled-rest map
  is not the identity program's rest (`axial_rest.map_rest_mismatch.push_blade` 1). tab_i's missing_map column shows 0, because the
  reject happens at setup.
- **axial_05 on:** `axial_rest.settled_used` 32.
  - Rejections: target_placement 19 (mostly no push-feasible placement on the push_blade map), program_check_failed 4, approach_untrackable 3.
  - The brace never landed: brace_ik_failed in all 4 written demos (kit paired 21, push_only 11, opposite_used 0).
- **Turn vs tilt on axial_05 on.** In every demo the goal tilt equals the settled tilt (0.086-0.087); the full error equals the heading
  error within 0.0002 rad.
  - 310002 and 310003 succeeded: turns 1.21 / 1.22 rad, errors 0.016 / 0.020 (tolerance 0.12).
  - 310000 not_settled: turn 0.425, short by 0.088 (tolerance 0.05).
  - 310001 robot_wobble: lost by 1.21.
  - No demo here was decided by the goal frame. The frame matters for turns below ~0.87 rad.
- **Regression on axial_01 / 03 / 04.** The rests are the identity within 5e-6 rad.
  - The only program difference: the pre-settle drop is 3.0 mm lower (top + h_root instead of `object_pose_on`'s + Z_OFFSET 3 mm). The
    settled pose is identical within 0.1 mm.
  - The same draws (6 instances) give identical attempt, yaw, push end, brace and status in both arms.
  - The other draws diverge on the 3 mm. axial_04 on's 2 failures are spin +1 draws the off arm never drew: within N 4 noise, not
    attributed to the switch.
- **For the owner (default unchanged):** the switch works as approved and is inert on upright rests except the 3 mm drop.
  - Recommend default on, together with tqqwys's settled-rest maps in the consolidated root. The switch off with those maps gives
    missing_map.
  - Option: keep the 3 mm drop under the switch, so upright rests stay draw-identical.

#### Owner item 3 (2026-11-08, owner_20261108g): the settled rest is the DEFAULT

Owner decision, verbatim: "Make ISPATIALGYM_AXIAL_SETTLED_REST the default, in its own commit, together with tablespoon tqqwys's
settled-rest push_blade / axial_brace maps in the consolidated root (they must go together: switch off with those maps gives missing_map).
Keep the usual 3 mm drop gap for upright rests (tilt <= 0.05 rad), so axial_01/03/04 draw exactly as today. Unit test that an upright rest
places identically with the switch on and off. Snapshot; confirm with one N 4 v2only run of axial_05 and axial_01 / 03 / 04 on the staged
root, same indices as i7ar_on. axial_05's brace never landing (IK failed in all 4) is a follow-up: diagnose it (CPU first), report, no fix
without asking."

- Checked first: the switch was not the default on this branch (4d22896d read "0").
- **289ca4c3 - upright rests stay today's program.** An upright v2 rest (tilt <= `contact_poses.AXIAL_UPRIGHT_TILT_RAD` = 0.05 rad,
  the same test `contact_library.axial_rest` already applied) is the identity rest under the switch too:
  - `object_pose_on` places it, with the usual 3 mm drop gap, not top + h_root;
  - the goal, aligned yaw, extents and stop geometry are today's; counter `axial_rest.upright_identity`;
  - the library side matches: `axial_rest` returns the identity for an upright rest under the switch, and `build_job` writes
    `settled_rest` into the library identity only for a TILTED rest (one the switch-off build refuses). So no library identity moves with
    the flip: upright libraries carry no param, and tqqwys's settled-rest maps keep their keys.
  - Tests (`test_axial_settled_rest.py`, 16 pass): an upright rest places bit-identically on and off (pose, orientation, aligned yaw,
    stop geometry; z = top + bottom offset + 3 mm) for the identity, magazine tfovwy's rest and a 0.04 rad tilt; a tilted rest still
    settles at top + h_root; an upright library has the same digest and entries on and off.
- **dee19fcb - the default flip** (own commit): `ISPATIALGYM_AXIAL_SETTLED_REST` defaults on; `=0` turns it off.
- **The maps that must go with it** (the flip and these two maps fold in together; with the switch off they are a rest mismatch, so
  axial_05 gets `missing_map`):
  - `maps/tablespoon__tqqwys__s1/cmap__push_blade__tablespoon__tqqwys__h0.8126__cba67061f6a58376` (.json + .npz; library digest
    3318755d5aaa873e, params settled_rest);
  - `maps/tablespoon__tqqwys__s1/cmap__axial_brace__tablespoon__tqqwys__h0.8126__8753ec81a0698bed` (.json + .npz; library digest
    da04050a5da62ad3, clear-turn build + opposite end, settled_rest);
  - staged root for the integration: `$OFH/v2maps_int1_axialrest` (= `v2maps_int1_brace34b` + these two maps; source
    `$OFH/axialrest_20261108/libs_push`, `libs_brace`).
- The axial brace switches: this branch merged i7-brace at 9d2ce7e3, before i7-brace's own default commit 59f7967b. The confirm run
  passes `ISPATIALGYM_AXIAL_BRACE_TURN_COVER=1` and `ISPATIALGYM_AXIAL_BRACE_OPPOSITE=1` explicitly, as i7ar_on did; 59f7967b is not
  merged here (it would bring i7-brace's wall_brace work along).
- Suite on 38ee02e8: 23 failed + 3 errors, the same set as with the base d72185da's files. 38ee02e8 moves test_contact_library's
  lying-rest refusal to the switch-off setting (`=0`).
- **Confirm run** `i7ar_def`: trunk_38ee02e8, svl titanrtx, v2only N 4, indices 310000-310003 as i7ar_on, root
  `$OFH/v2maps_int1_axialrest`, alias memory v0027 pinned, no settled-rest env, both brace switches explicit (DECISIONS "i7-axialrest
  CONFIRM").

| alias | att / wr / ok | GPU-h per success | vs |
|---|---|---|---|
| axial_05 | 20 / 4 / 3 | 0.10 | i7ar_on 30 / 4 / 2 |
| axial_01 | 11 / 4 / 2 | 0.17 | all 4 draws identical to i7ar_off (11 / 4 / 2) |
| axial_03 | 6 / 4 / 4 | 0.048 | all 4 draws identical to i7ar_off (6 / 4 / 4) |
| axial_04 | 12 / 4 / 2 | 0.095 | 2 draws identical to i7ar_off; 2 diverge at a run-to-run program_check IK-branch verdict |
| all | 49 / 16 / 11 | 0.093 | i7ar_on 0.090, i7ar_off 0.083 |

  - The upright placements are back at the off arm's push heights (no 3 mm lower drop).
  - axial_05 310001 succeeded at a 0.445 rad turn: full error 0.0027 against a 0.050 tolerance. The upright goal would have failed it on
    the 0.087 tilt, so this is the first success the goal frame decides.
  - The brace still failed in all 4 axial_05 demos (`brace_ik_failed`), as diagnosed below.

**axial_05's brace never landing: CPU diagnosis (no fix made).** Script `$OFH/axialrest_20261108/brace05.py` rebuilds the sampler's
brace candidates of the 4 written i7ar_on demos and reads the axial_brace map at the goal cell.
- **Arm:** the left (stop brace); the right is locked at its push goal while the brace is solved.
- **Entries:** the map has 16 classes, all blades at face fraction f0, pitch 0 / 30, either wrist turn. The -end groups hold only their
  own-end blades, the +end groups only opposite-end blades. In every draw one side is skipped whole by the sweep (the reconstruction
  reproduces the recorded skips 5 / 5 / 6 / 6).
- **Map vs live:** the map calls 1-2 classes of the drawn group feasible at the goal cell; those exact classes fail live. blade_0- /
  opp_blade_0 fail the exact IK at the pre; blade_30- intersects the target at its goal pose (3 mm) or is untrackable.
- **Pre vs reach:** not reach. Every pre is 0.17-0.33 m inside the left arm's reach sphere at the stance trunk, 7-12 cm above the top.
- **Collisions (CPU evidence):** the map is single-arm, the live check is two-arm. 310001's opposite-end blade sits 7 mm from the right
  push hand's goal pose; the own-end blades sit 8.7-9.0 cm from it. The locked push arm (or the goal hull's 3 mm margin) is the likely
  cause; a GPU replay of `brace_second_arm` with the collision pairs would separate them.
- **The 3 braces that did land** were `left_h-` pads, which have no map class: the thin target's fallback after every blade failed. The
  pad's pre stands 2.2 cm above the table. program_check rejected the whole attempt ("support_policy: non-finger link touches the
  support"), 3 of the 4 program_check_failed. The sampler's brace screening does not apply the support policy at the pre, so a refused
  brace costs the attempt instead of leaving it push-only.
- **Options for the owner:** (1) screen the brace pre with the program check's support policy (a refused brace -> push-only); (2) thin
  targets try only the map's validated classes; (3) the map / joint placement checks the brace with the right arm at its push goal. A GPU
  probe of the collisions comes first.

**5v: axial_05's brace - the GPU probe and the pre-policy switch (owner 2026-11-08 5v.1 / 5v.2).**
Owner decision, verbatim: "5v.1 APPROVED: a GPU probe to separate the locked push arm from the 3 mm target margin as the cause of the brace
IK / collision failures (diagnosis only). 5v.2 Then the first fix, behind a switch: check the brace pre pose against the program check's
support policy at sampling, so a refused optional brace leaves the attempt push-only instead of costing the whole attempt. N 4 v2only A/B on
axial_05 (+ axial_01/03/04 as the regression check). Default after I see it."

- Checked first: neither was on the branch (tip e9d39ac9).
- **384c25de** (snapshot trunk_384c25de), both switches default OFF:
  - `ISPATIALGYM_AXIAL_BRACE_PROBE=1` (diagnosis only): every brace candidate that fails at sampling is re-checked with one cause removed at
    a time - `again` (the same call), the target check at 2 / 1 / 0 mm, no target check, `no_right` (the right arm at the stance posture,
    not at its push goal), both, `empty` (also every spawned object out) - one `AXIAL_BRACE_PROBE {json}` log line, plus the blame of the
    no-right configuration with the right arm put back at its goal. Verdicts unchanged; the solver's remembered solutions are restored.
  - `ISPATIALGYM_AXIAL_BRACE_PRE_POLICY=1`: a candidate that passed `brace_second_arm` gets the program check's own brace `pre` call
    (fresh exact IK of the left arm from the stance configuration, closed fingers, target + supports in the ignore set, target approached;
    `IKChecker.exact` applies the shared support policy). A `support_policy` failure fails the candidate (next one; none left = push-only);
    any other failure is counted (`other_fail`), not refused. Counters `axial_brace.pre_policy.checked / passed / refused[.<label>] /
    push_only / other_brace / other_fail`. No shared gate or policy changed.
  - Tests `test_axial_brace_pre_policy.py` (4; with test_axial_settled_rest 20 pass). Suite at 384c25de: 23 failed + 3 errors, the same 26
    ids as the base e9d39ac9 run side by side on lambda. pyflakes clean.
- Runs: svl titanrtx, v2only N 4, indices 310000-310003 (i7ar_def's), root `$OFH/v2maps_int1_axialrest`, alias memory v0027, overlay r3,
  `TURN_COVER=1` + `OPPOSITE=1` explicit in every arm. Tags `i7ar_probe` (axial_05, PROBE=1), `i7ar_pol` (PRE_POLICY=1), `i7ar_pol0` (same
  snapshot, off). Record `$OFH/axialrest_20261108` (`loop_5v.sh`, `jobids_5v.txt`, `probe_tab.py`, `probe_sum.py`).

**5v.1 probe** (i7ar_probe, 87 failed candidates, 20 att / 4 wr / 3 ok): neither the locked push arm nor the 3 mm margin is the main cause.

| what frees the candidate | count | where |
|---|---|---|
| nothing: fails with the right arm at the stance posture and every spawned object out (the left arm's IK / joint limits at that pose) | 67 | ik at the pre 48, contact untrackable 10, ik_joint_limit 9; all pads, all opposite-end blades, blade_0 / blade_0- 12 of 12 |
| removing the target check entirely (the blade overlaps the target's goal hull: still fails at 2 / 1 / 0 mm) | 11 | blade_30 3, blade_30- 2, blade_45 3, blade_45- 3 |
| removing the locked push arm AND the target check | 7 | blade_30- 2, blade_45- 2, blade_60 2, blade_60- 1 |
| removing the locked push arm alone | 1 | left_v |
| other spawned objects | 1 | blade_45 |
| the 3 mm margin alone (ok at 2, 1 or 0 mm) | 0 | |
| the same call again (solver noise) | 0 | |

- Where the push arm is in the way, the blame is the two hands: left gripper / finger vs the right wrist camera / gripper, 4-40 mm deep.
- The axial_brace map calls blade_0- feasible at 310002 / 310003's goal cell, yet live it has no IK even in the empty world: the map and
  the live check disagree on the brace pose itself (one-arm map vs two-arm live is not the explanation for these).

**5v.2 A/B** (att / written / ok, GPU-h per success):

| alias | i7ar_def (38ee02e8) | i7ar_pol0 (off) | i7ar_pol (on) | where the switch acted |
|---|---|---|---|---|
| axial_05 | 20 / 4 / 3, 0.10 | 20 / 4 / 3, 0.093 | 20 / 4 / 3, 0.087 | 310000 att 10: left_h, left_h-, left_h refused -> push-only |
| axial_01 | 11 / 4 / 2, 0.17 | 9 / 4 / 3, 0.090 | 11 / 4 / 2, 0.17 | 3 checked, 3 passed |
| axial_03 | 6 / 4 / 4, 0.048 | 6 / 4 / 4, 0.045 | 8 / 4 / 3, 0.067 | 3 checked, 0 refused, 1 other_fail |
| axial_04 | 12 / 4 / 2, 0.095 | 16 / 4 / 3, 0.067 | 18 / 4 / 4, 0.050 | 310001 att 6: left_v refused -> push-only |
| all | 49 / 16 / 11, 0.093 | 51 / 16 / 13, 0.071 | 57 / 16 / 12, 0.082 | 4 refusals, 2 attempts made push-only |

- "pre: support_policy" program-check rejections: 1 in each off arm (and the probe run), always axial_05 310000 attempt 10; 0 in the on arm.
- That attempt then still died at the program check on the push itself ("push pre-contact witness 3.16 rad ... another branch"), as
  attempt 11 does in the off arms; axial_04 310001 attempt 6 was written push-only in all arms. So the switch removes the brace rejection
  but changes no success at N 4.
- axial_03 diverged at 310002 attempt 0 before the switch refused anything: live, `opp_blade_45-` passed `brace_second_arm` (the off arms
  drew `opp_blade_60`); the sampling pre call already failed it on `ik_joint_limit` (`other_fail`, not refused, outside the decision), and
  the program check rejected the attempt on the same reason.
- Default unchanged (owner decides). Open for the owner: (a) the flip; (b) whether the sampling pre call should refuse on any failure, not
  only the support policy (it would have caught axial_03 310002/0); (c) the brace itself: 67 of 87 candidates have no IK in the empty
  world, so the map / live brace pose mismatch, not the two-arm check, is what keeps axial_05's brace from landing.

### i7-attachdef (2026-11-08): the cap fixes as the default (owner 4.1) and the past-demo audit (owner 4.4)

Owner decisions 2026-11-08 (tmp/outputs/owner_answers_20261108.md, verbatim): **4.1 "Make the cap fixes the default"**; **4.4 "approve
an audit (CPU, from the demo metadata), and re-collect any affected instances"**. Branch `i7-attachdef` (from of-v2maps cf1f5d2d), record
`$OFH/attachdef_20261108`. Process as for the plate switches (count first, then the default in its own commit, rebuild only what moves,
stage in a copy, validate the aliases that read the changed entries).

**4.1 Count** (candidates only, CPU, trunk_fcddc495; `count/count_table.md`): all 56 non-floating entries of v2maps_int1 (the 20 floating
entries read neither switch: GraspGen poses, the width at the pad centre, no attachment rule), each with its RECORDED config (a candidate
switch it lacks = off, as built) and the two switches per arm: off / outer (outer_pinch) / face (attach_face_retreat, attachment_free only -
elsewhere it is structurally inert) / on. The arm matching the recorded state reproduces every recorded candidates.json (56 / 56: counts,
poses, shifts, verdicts, representatives). 22 move with both on (candidates / geometry-ok / reps off -> on):

| entry | off -> on | | entry | off -> on |
|---|---|---|---|---|
| cap xxqyrt (already on in int1) | 0/0/0 -> 12/12/10 | | garlic xotobt | 12/8/5 -> 12/5/5 |
| beaker jdijek | 13/8/4 -> 24/20/5 | | sandal chgcrm | 13/13/10 -> 13/10/10 |
| beer glass mxsliu | 19/18/3 -> 24/21/3 | | tablespoon yxxfdj | 12/9/6 -> 12/3/2 |
| detergent gkpmii | 18/14/10 -> 19/15/10 | | teacup wopjex | 14/6/5 -> 20/4/4 |
| tape kavsnx | 96/19/15 -> 89/79/30 | | lid tnwisp (face) | 13/4/4 -> 13/3/3 |
| cork ncxgpe (face) | 35/34/5 -> 55/46/6 | | plate luhkiz | 77/18/17 -> 81/18/17 (poses) |
| gym shoe kmcbym | 24/18/18 -> 26/23/20 | | salt shaker fonexa | poses only (4 swapped) |
| jigsaw isgefa | 36/31/17 -> 39/36/30 | | steak lcebsq | 14/11/9 -> 15/11/10 |
| pencil case otcmih | 15/4/3 -> 16/13/8 | | trash can jlawet | poses only (1) |
| plate qtfzeq | 46/31/30 -> 48/33/32 | | skateboard jljqtt / webcam fabztm | poses only |
| skateboard acmbvb | 22/22/10 -> 24/23/10 | | | |

outer_pinch moves every hollow / multi-piece / non-convex entry (the diameter pinch the first-hit ray never proposed; the width and the
antipodal cone measured from outside); attach_face_retreat moves only the cork, the lid tnwisp and the cap. 34 entries do not move.

**Default** 044715ef: `CandidateConfig.outer_pinch = attach_face_retreat = True` (identity: an entry built before reads both off and keeps
its key; a new build records them on). Tests test_attachfix / test_glib_switches.

**Rebuild** (trunk_044715ef; `build/plan.json`, `submit_rebuild.sh`): the 21 moved entries other than the cap, each with its recorded
config + both switches, its densify passes and validation overrides, its maps as it had them in v2maps_int1 (tabletop at its anchor keys and
arms - per_d_e, the current default -, floor, the sink interior + interior PLACE maps of plate luhkiz with the recorded rule, the attach
maps box + swept of the lid tnwisp / cork; the gkpmii touch_brace contact map reads a contact library, not the grasp entry: carried).
Physics-valid old -> new: up - tape 12 -> 28, pencil case 1 -> 5, plate qtfzeq 27 -> 29, garlic 3 -> 4, cork 2 -> 3; DOWN - teacup 5 -> 1,
tablespoon yxxfdj 5 -> 1, beer glass 2 -> 1, plate luhkiz 9 -> 8; equal on 12 (most capped at max_valid 6). No entry lost all its valid
grasps. Staged `attachdef_20261108/v2maps_ad` (cp -a v2maps_int1 + the 21 entries and maps; old -> `_superseded_attachdef`, legacy_keys
carried in each library.json, links.json records the layer, AD.txt); check_int1.py on it: 76 entries, 136 maps, 0 object-key mismatches,
0 errors. NOTE: plate qtfzeq's map overlay release (map_overlay r1: ce1af987a8df9d69 / b2f36adb26c194ad) is pinned to the int1 map keys;
the rebuilt maps have new keys, so the overlay does not apply to them.

**Validation** (v2only N 4, A/B on the same snapshot trunk_044715ef and the same pinned outcome memory v0001, no aggregation between
arms; tags `ad_int1` = v2maps_int1, `ad_v` = v2maps_ad; `submit_val.sh`; where the switch acted = the written instances' grasp on the
rebuilt entry, kept (a pose of the int1 entry's valid grasps) or NEW; `val_acts.py`). Written / ok A -> B (B's grasps):
disembedding_01 jigsaw 4/0 -> 4/2 (1 new); disembedding_05 beaker 4/3 -> 4/4 (3 new); disembedding_07 tape 4/4 -> 4/4 (3 new);
disembedding_08 pencil case 4/4 -> 4/4 (3 new); nextto_03 tablespoon 4/4 -> 4/4 (4 new: its one valid grasp, 18 att vs 13);
nextto_04 gym shoe 4/2 -> 4/4 (2 new); under_small_02 beer glass 4/4 -> 4/4 (2 new); nextto_02 salt shaker 3/3 -> 3/3 (kept);
ontop_small_02 steak 4/2 -> 4/4 (kept); inside_04 garlic 4/4 -> 4/1 (1 new; the 3 failures plan_failed at the grasp's protected-object
re-plan past the saucepot); **under_small_01 teacup 4/3 -> 0 written in 60 attempts** (44 place_unreachable: its one valid grasp is
side_7, an 85 mm diameter pinch; int1 had 5 rim pinches 5-9 mm); **attachment_01 lid tnwisp 4/4 (0.40 GPU-h) -> 0 written after 1.5
GPU-h** (target_placement: path_collision at the attach swept pose, robot links on the teapot at the attach pose; attach_face_retreat moved
both valid grasps 5-6 mm up the lid, 23-24 -> 28-29 mm). Running at the report (`tab_i.py -a ad_v` / `-a ad_int1`): attachment_01 B,
under_small_03 B (2/2 so far), inside_large_03 A and B, under_large_01 A / B, under_large_03 A / B.

**Cause of the teacup / tablespoon / beer-glass losses** (diagnosis, not changed): outer_pinch measures the width from +-aperture/2
(44.45 mm, the FULLY open pads) inward (`outer_hits`), so a rim / wall / handle pinch whose closing line meets another surface within 44 mm
(the cup's far wall, the spoon bowl) is measured across it and dropped - while the oracle pre-shapes to the pinch width / 2 + margin and
never sweeps that far. The cap needed the all-hits sampling (the diameter pinch the first-hit ray never proposed); the measurement from the
fully open pads is what removes the thin pinches (teacup 5 -> 1, tablespoon 5 -> 1, beer glass 2 -> 1 valid). A fix (measure from the
pre-shape opening, as sweep_from_preshape does for the sweep) changes a gate: owner decision.

**4.4 Audit** (CPU, metadata only; `audit/audit.md`): every written demo under $OFH/out (2619 with a meta.json, every tag and family): its
collector process (results.jsonl pid / host / job, ordered by finish time), whether an earlier episode of the same process joined (before
the fix the attach rule's join list was per process: an episode joined iff its meta.attach_joins grew), and the measured start offset of
every TRO object (meta.start_state vs the TRO root pose). 81 demos ran after a joined episode in the same collector; **52 are affected**
(> 5 mm / 0.05 rad at the start; all attachment aliases, all on code before the fix, each the instance after a joined episode, offsets 40-549
mm): attachment_01 27 (3 successes), _02 2, _03 4, _04 19 (4 successes) over 15 tags. The 29 others after a join started at their TRO pose
(<= 0.8 mm): 18 on the fixed code, 11 on the old code - 10 of them the THIRD episode of a process (the offset hits the episode right after
the join: the deferred un-filter runs inside that restore) and one (the cork, 03 / 300021, i7_attswept) right after a join yet at 0.0 mm. 6 demos off their TRO without a join are a different
matter, not this bug (disembedding_04 / 09 i7t03_cnt4 80-443 mm, disembedding_08 pencil 0.6-0.8 mm / 0.16-0.22 rad, two mental_rotation
error episodes): reported, not touched.

**Re-collect** (submit_fast recollect, collector-only, trunk_fcddc495 = the integration tip with the fix, the same instance indices, K 2;
`recollect_affected.py`): 52 / 52 collected, every one at its TRO pose (max start offset 0.01 mm, tro_pose_check 0 mismatches); successes
7 -> 45 (attachment_01 3 -> 27 / 27, _02 0 -> 1 / 2, _03 0 -> 3 / 4, _04 4 -> 14 / 19), 34 jobs, 7.1 GPU-h = 0.16 GPU-h / success. Two lost to
the node (a collector segfault at start, a numba / cffi mismatch) were rerun (`__ad2_`). The 7 affected successes had a GROUP goal override
(prep_recollect keeps no previous slot poses, so it skips them): only their lid was off, the partner at its TRO pose, so they were
re-collected from the source's overridden TRO (goal = the lid seated on the partner where it stood; recorded in `<job>/unskipped/README`).
The 52 affected demos are not deleted: moved to `$OFH/out/_superseded_attachdef_20261108/<tag>/<alias>/<group>/demo/<iid>`, each source group
keeps `SUPERSEDED_attachdef.txt`; ledger `audit/superseded.tsv`.

### i7-attachdef follow-ups (2026-11-08): the pre-shape measure (owner 5.1), the int1 teacup / lid kept (5.2), the start offsets (a sampler race), the qtfzeq overlay

Owner decisions 2026-11-08 (DECISIONS "OWNER DECISIONS (i7-attachdef follow-ups)"): **5.1 APPROVED gate change** - outer_pinch measures
the pinch width at the oracle's PRE-SHAPE opening (pinch width / 2 + margin) instead of the fully open pads (+-44.45 mm), the same "judge
at the pre-shape" rule as the IORS large picks; count, rebuild, validate. **5.2 APPROVED meanwhile** - keep the int1 teacup wopjex and lid
tnwisp entries (switches off) in the staged root, as plate luhkiz, so under_small_01 and attachment_01 keep writing. **NEW** - diagnose the
six start offsets the 4.4 audit found without a join. Lead: the offsets' fix (below) approved as a sampler-only bug fix, default off; the
qtfzeq map overlay re-pinned to the rebuilt maps. **Owner 5j (2026-11-08, later): the race fix is the DEFAULT** ("Make
ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS the default, in a separate commit on i7-attachdef").

**5.2** (done first): the int1 teacup wopjex / lid tnwisp entries and their maps moved back from `glib|maps/_superseded_attachdef` into
`attachdef_20261108/v2maps_ad` (the 4.1 rebuilds -> `glib|maps/_held_attachdef_rebuild`; AD.txt); check_int1 76 entries, 138 maps, ERRORS 0.
v2only N 4 (`ad_v52`, trunk_044715ef, memory v0001, no aggregation): under_small_01 7 att / 4 written / 4 ok, 0.35 GPU-h (ad_v: 60 / 0 / 0);
attachment_01 7 / 4 / 4, 0.48 GPU-h (ad_v: 0 written at 1.88 GPU-h).

**5.1 code** (each switch off reproduces the build before; identity only when on):
- 71b4d1fc `CandidateConfig.outer_pinch_preshape` (candidates.PreshapeOuter): every surface on the closing line within the open aperture
  (all ray hits) classified as an OUTER boundary (material on the pad-centre side, free beyond); a pinch (p, m) holds when the pads at
  h = preshape_pad_half(p + m) (= oracle_preshape_q: width / 2 + gripper_preshape_margin, the pad sweep's definition) start beyond it and
  meet it first; the holding pinch of smallest width is kept. contact_features / finger_blades_clear / the symmetry width read it.
- 599fdad9 a non-convex collision piece is judged by containment (Trimesh.contains): the face planes bound only a convex piece - the first
  count dropped every candidate of the three hardback lean / overhang entries (12/12/7 -> 0). The pad sweep's `_inside_any` has the same
  convex assumption: reported, not changed.
- dbd8f4bd the first-hit proposals are added: the measure alone moved no thin pinch (count_pre: teacup 20/4/4 -> 21/5/5, tablespoon, garlic,
  lid unmoved), because the all-hits Isaac sampler pairs each surface point with the FURTHEST hit, so a rim / wall / handle pinch is never
  PROPOSED under outer_pinch. Proposals only (same seed), no gate.
- d1ad9cb4 each candidate's policy geometry at its own pre-shape: check_grasp_geometry was called with the batch's widest pre-shape, so the
  8-16 mm spoon pinches were judged ~29 mm per finger open (finger_penetration).
- 7ac55021 the default (its own commit, owner 5.1). Tests test_outer_preshape, test_glib_switches.

**5.1 count** (`count_pre4`, candidates only, CPU, trunk_d1ad9cb4; arm "on" = the 4.1 defaults reproduces the 4.1 count 56 / 56): 19 of
the 56 non-floating entries move (candidates / geometry-ok / reps, on -> pre): tablespoon yxxfdj 12/3/2 -> 12/9/7, garlic xotobt 12/5/5 ->
12/12/9, teacup wopjex 20/4/4 -> 21/6/6, hard candy vwegtt 28/9/9 -> 28/28/20, brussels sprouts mbkrxe 23/8/8 -> 23/22/10, tape kavsnx
89/79/30 -> 108/83/30, cork ncxgpe 55/46/6 -> 55/46/12, gym shoe 26/23/20 -> 27/24/20, qtfzeq 48/33/32 -> 48/34/33, storage box 38/35/11 ->
38/35/12, light bulb 24/20/10 -> 24/22/10, jigsaw 39/36/30 -> 39/37/27; poses / per-pose records only: beaker, bottle of milk, pencil case,
plate luhkiz, sandal, skateboard acmbvb, trout. Unmoved: lid tnwisp 13/3/3 (its 4.1 loss is attach_face_retreat's, not the measure's), beer
glass 24/21/3, the hardback entries, 34 others.

**5.1 rebuild** (`build2/`, trunk_7ac55021: glib array 18381544 19 / 19, maps array 18381545 22 / 22; each entry with its recorded config +
the three switches on, its densify / validation overrides and its maps as in v2maps_ad - tabletop per_d_e, floor sandal / storage box, the
luhkiz sink interior + interior PLACE maps, the cork attach box + swept). Physics-valid, v2maps_ad -> build2 (poses kept): brussels 4 -> 6
(1), cork 3 -> 7 (3), garlic 4 -> 6 (3), hard candy 5 -> 6 (1), tablespoon yxxfdj 1 -> 5 (1), tape 28 -> 29 (26), luhkiz 8 -> 9 (8), qtfzeq
29 -> 30 (29), storage box 11 -> 12 (9); same count, other poses: milk 6 (5 kept), gym shoe 6 (3), jigsaw 6 (2), sandal 6 (4); identical
valid poses: beaker 4, light bulb 6, pencil case 5, acmbvb 1, trout 0 (0 before). **Teacup wopjex: the rebuild has 4 physics-valid grasps,
the int1 build (kept by 5.2) 5 -> the int1 teacup stays** (stage_ad2's rule: replace only when at least as many). The lid tnwisp was not
rebuilt (unmoved by the measure): the int1 lid stays. STAGED `attachdef_20261108/v2maps_ad2` (`stage_ad2.py`: cp -a v2maps_ad + the 18
entries and their maps; old -> `glib|maps/_superseded_attachdef2`, legacy_keys / links carried, AD.txt, staged2.json). **check_int1** on it:
76 entries, 138 maps, 0 object-key mismatches, attach 8 / contact 16 / prefix 12 / interior 12 maps without error, 51 links, 0 unresolved,
ERRORS 0.

**qtfzeq's map overlay re-pinned** (lead; `carry_overlay.py`, `overlay_carry.json`): the overlay release r2 named the int1 map keys
(right ce1af987a8df9d69 v2, left b2f36adb26c194ad v2), so it never applied to a rebuilt qtfzeq map (ad_v's under_small_03 33 att / 2 written).
Cell records (stand-off | translate | tilt # cell) are base-relative reach facts of the map view: carried as they are. Yaw pairs name a grasp:
carried under the rebuilt name when the grasp's pose (T_object_eef, 0.1 mm / 1e-3) is in the rebuilt entry, dropped otherwise. Right ->
52df23ebc0086132 v1: 94 / 94 cells, 116 / 118 yaw pairs; left -> dafc858126a2111b v1: 53 / 53 cells, 37 / 39 yaw pairs; 26 of the 27 int1
grasp names mapped (rim_straddle_11 is not in the rebuild: the 4 dropped pairs). New release `$OFH/map_overlay` **r3** = r2 + the two new
keys (every r2 key at its r2 version: nothing changes for a map r2 named).

**5.1 validation** (v2only N 4, tag `ad_v51`, trunk_7ac55021 - the build's snapshot, the runtime of ad_int1 / ad_v / ad_v52, race switch
off -, `attachdef_20261108/v2maps_ad2`, overlay release r3, memory v0001 no aggregation, base index 300000, <= 4 GPUs via `loop_v51.sh`;
where it acted = the written instances' grasp on a rebuilt entry, NEW (only with the pre-shape measure) or kept (a valid pose of the v2maps_ad
entry), `val_acts51.py`). The 13 aliases whose entries' valid grasps changed; before = ad_int1 | ad_v (same snapshot), else the latest int1
run. att / written / ok, GPU-h:

| alias (entry) | ad_v51 | grasps | before |
|---|---|---|---|
| nextto_03 (tablespoon yxxfdj) | 11 / 4 / 4, 0.26 | 4 new | 13 / 4 / 4, 0.39 \| 18 / 4 / 4, 0.33 |
| nextto_04 (gym shoe) | 28 / 4 / 4, 0.36 | 4 new | 17 / 4 / 2, 0.26 \| 20 / 4 / 4, 0.38 |
| inside_01 (brussels sprouts) | 12 / 4 / 4, 0.46 | 3 new | int1_5 7 / 4 / 3, 0.31 |
| inside_02 (hard candy) | 39 / 4 / 4, 0.51 | 3 new | int1_1 26 / 4 / 4, 0.38 |
| disembedding_07 (tape) | 7 / 4 / 4, 0.38 | 1 new | 8 / 4 / 4, 0.36 \| 8 / 4 / 4, 0.38 |
| disembedding_01 (jigsaw) | 15 / 4 / 0, 0.74 | 3 new | 19 / 4 / 0, 0.66 \| 6 / 4 / 2, 0.52 |
| under_small_03 (plate qtfzeq) | 21 / 4 / 3, 1.03 | kept | 11 / 4 / 3, 0.64 \| 33 / 2 / 2, 1.41 |
| disembedding_04 (bottle of milk) | 8 / 4 / 4, 0.40 | kept | i7t03_cnt4 7 / 4 / 4, 0.31 |
| inside_04 (garlic) | 40 / 4 / 2, 0.95 | kept | 43 / 4 / 4, 0.90 \| 39 / 4 / 1, 1.15 |
| attachment_03 (cork) | 17 / 4 / 4, 1.21 | kept | int1_5 39 / 3 / 2, 2.34 |
| touching_large_03 (storage box) | 4 / 4 / 4, 0.32 | kept | int1_1 5 / 4 / 4, 0.26 |
| under_large_01 (sandal) | 6 / 4 / 3, 1.96 | kept | 5 / 4 / 0, 2.17 \| 8 / 4 / 3, 3.54 |
| inside_large_03 (plate luhkiz, sink) | 0 written at 1.86 GPU-h (cancelled: 2 attempts, each the 3000 s time_budget at the F draw) | - | 0 written at 1.9 \| 1.4 GPU-h (cancelled) |

Where the pre-shape measure acted (the thin pinches back: tablespoon, gym shoe, brussels sprouts, hard candy, tape, jigsaw) every alias
wrote 4 / 4 with the new grasps; all but the jigsaw succeeded 4 / 4 (disembedding_01 fails as before it: failed_grasp, symbolic_only x2,
pause - not a grasp-selection loss). Elsewhere the written grasps are kept poses and the differences are N 4 variance. The qtfzeq run reads
the carried overlay (kit `map_tilt.by_map.52df23eb / dafc8581`; cells invalidated in the run 6, vs 37 in ad_v where the rebuilt maps had no
overlay). Not rerun: under_small_01 (the int1 teacup kept = ad_v52's entry: 7 / 4 / 4, 0.35), disembedding_05 beaker / _08 pencil case (the
same valid poses as v2maps_ad), light bulb / acmbvb / trout (no alias reads them). inside_large_03 never wrote on this snapshot in any arm (the sink
stance is i7-interior2's: i7i2_sinkdiag3 wrote 4 / 4 there): the luhkiz rebuild (8 -> 9 valid) is untested by a demo. Total ad_v51: 12
finished jobs, 208 att / 48 written / 40 ok, 10.45 GPU-h = 0.26 GPU-h per success (+ 1.86 GPU-h for the cancelled sink job).

**Start offsets without a join = a SAMPLER RACE** (diagnosed; not a run_isg reload bug): `write_instance` wrote the TRO, the sidecar and
the SUMMARY - the collector queue's ready marker (`run_isg.queue_ready`) - BEFORE the post-write screens (`check_instance_filter`,
`check_goal_occlusion`). A collector polling in that window claimed an attempt the screen then rejected; the sampler re-sampled the same
index over the files, so the collector either ran the rejected layout and wrote its goal override into the NEW TRO (disembedding_04 300213
/ _09 300211, goal-occluded 52-69 %: the 80-443 mm offsets) or found the files gone (FileNotFoundError: the two mental_rotation errors).
disembedding_08 int1_1 300006 / i7_split 600006 ran an instance_filter-rejected layout (a repeat of an eval instance): the pencil's
0.16-0.22 rad "roll" was this, not the restore settle. Count from the logs (`audit/race_count.json`): 1021 instance_filter + 38
goal_occlusion post-write rejections campaign-wide; 7 claims the fix would have prevented (6 episodes ran a rejected attempt + 1
FileNotFoundError, survey1 disembedding_07 300001). The disembedding_04 / 09 / 08 (int1_1) demos moved to
`out/_superseded_attachdef_20261108` (their TROs carry the old episode's goal override: not re-collectable); i7_split 600006 lives in
shiyuc's root (recorded, not moved).

**Fix** 9bbbba7a (lead decision; sampler only, behind a switch): `ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS=1` writes
`<iid>.summary.json.pending` and `os.replace`s it after every screen passed; a rejected attempt's files (the pending summary too) are removed;
kit counters `publish_after_screens.published` / `.withheld`. Off: the summary in place as before. Test test_publish_after_screens (a
rejected attempt is never claimable, an accepted one exactly once, off unchanged). Demonstration `ad_race` (18381927, disembedding_09 N 4 - the alias with the most goal_occlusion rejections -,
trunk_9bbbba7a, v2maps_int1, memory v0001 / no aggregation, switch on): 11 att / 4 written / 4 ok, 0.43 GPU-h = 0.11 per success (i7t03_cnt4
20 / 4 / 3, 0.51; survey1 7 / 4 / 4, 0.29); **publish_after_screens.published 4, .withheld 2** (= instance 300006's two goal-occluded
attempts, target 62 % / 65 % hidden); 4 claims = the 4 published instances, 0 pending files left.

**Default** (owner decision 2026-11-08, 5j): 917f7232, its own commit - `fast_on("PUBLISH_AFTER_SCREENS", default=True)`;
`ISPATIALGYM_FAST_PUBLISH_AFTER_SCREENS=0` (or `ISPATIALGYM_FAST=0`) restores the summary written before the screens. Test: on by default;
the off test sets =0. Suite: 23 failed + 3 errors, the same ids as 9bbbba7a. Snapshot trunk_917f7232. A run submitted on an earlier
snapshot keeps it (the 5.1 validation below ran on trunk_7ac55021, switch off, as its A arms).

### i7-cork (2026-11-08): the cork's two routes (owner decisions 4.2 / 4.3 / 4.5)

Branch i7-cork from of-v2maps cf1f5d2d. Records: `$OFH/cork_20261108` (probes, first builds), shiyuc `$SY/isg/cork_20261108` (builds,
ledgers), staged root `$SY/v2/v2maps_int1_cork` (`cp` of v2maps_int1 + the three entries below, `CORK_STAGE.txt`).

**Owner decisions implemented.** 4.3: attachment_03's customs `brace_search: false` (22a13e67). 4.5: the overhang fallback is not wired
for the cork. 4.2: two routes, each behind its own default-off switch; the program reads the route the sampler recorded
(`grasps[0].attach_route`).

**Diagnosis.**
- Rest survey (`scripts/grasp_bands/cork_rest_probe.py`, the sampler's nudge: 2 deg about world x / y, +1 mm, 300 steps, 3 deg):

  | rest | validator slab | desk mdhelw (B1K, random xy) |
  |---|---|---|
  | male end down (r0, today) | 12 / 12 | 1 / 16 (median 15.7 deg, max 109) |
  | wide end down | 12 / 12 | 16 / 16 (max 0.35 deg) |
  | lying | 11 / 12 | 16 / 16 on x (the sampler's axis), 13 / 16 on y (an 11 deg roll) |

  The tapered end's 6.5 mm ring is stable on a flat box, not on the desk's collision surface.
- No hand-over primitive existed; the attachment program's tracked SLERP carry is route ii's 90 deg turn as it is; route i is built
  from Grasp / Lift / Reorient / Retreat(release) per arm.
- Hand-over geometry (`cork_handover_geom.py`, both hands' exact hulls on the 34.5 mm cork): axial fingertip pinches from both ends
  3.8-17.2 mm apart; a side pinch of either hand meets the other hand.

**Route ii, lying + 90 deg turn (`ISG_ATTACH_LYING90=1`).** The male's grasps / rest come from the v2 entry serving `<key>__lying90`
(`cork__ncxgpe__s1__attachment_free__raf10b6`: designated at the measured lying rest, attachment_free / seat axis, L3 config, 3 of 7
physics-valid; tabletop maps r / l at 0.7853, attach maps box + swept, right arm). Everything else is the attachment sampler and program
as they are. Kit `attach_route.lying90.{pick, attach_found, carry_checked, carry_failed, rest_stable, rest_unstable, drafted}`.

**Route i, hand-over (`ISG_ATTACH_HANDOVER=1`).** Two new entries (opt-in `CandidateConfig.finger_axis_window_m`: every finger vertex
within a window along the object's +z, a top grasp backed off along its approach into it; recorded in the library and the identity):
`cork__ncxgpe__s1__rd4d4f5` (`<key>__handover_pick`: wide end down, tabletop, fingers z <= -3 mm = the narrow end, 6 valid, right map) and
`cork__ncxgpe__s1__attachment_free__r428115` (`<key>__handover_hold`: r0, attachment_free, fingers z >= +3 mm = the wide end, 6 valid,
left-arm attach maps). Sampler (`_sample_with_handover`): the right pick; the hand-over search (male centre 0.52-0.60 m ahead, +-6 cm,
22-30 cm above the top, its axis along the robot's left, 8 turns; both eef poses in the stance image, the two hands' exact hulls >= 3 mm
apart, the right carry tracked with the male attached, the left [pre-grasp, grasp] with the right locked, the combined whole-body check,
the right's release retreat with the left locked); place stances = the grasp stance, then lateral / stand-off shifts (heading and trunk
kept, the path recorded as an axis leg); the left attach from the hold entry's left attach map, its carry tracked from the hold
configuration. Program `program_attachment_handover`: pick -> lift -> carry to the hand-over -> the left grasps -> the right releases
and backs off -> [axis-leg drive] -> the left carries to the pre-attach -> attach -> retreat. Program check: the hand-over phases (the
male posed at its recorded hand-over pose, both hands' chains with it ignored, the other hand locked); `grasp_of(obj, arm=)` picks the
carrying hand's record. Executor: the planner attachment moves to the taking hand; the hand-over grasp is not lifted off a support.
Kit `attach_route.handover.{pick, handover_carry_checked, handover_found, handover_unreachable, carry_checked, carry_failed,
attach_found, place_unreachable, rest_stable, rest_unstable, drafted}`.

**Validation** (`--grasps v2only`, the staged root, alias memory off, N 4; shiyuc root `$SY/isgf_root`, base index 620300, replicate 620400):

| arm (tag) | attempts | rest rejects | written | ok | GPU-h / success | demo failures |
|---|---|---|---|---|---|---|
| current route, brace_search false (`i7cork_base`, `_r2`) | 15 + 47 | 1 + 23 | 4 + 4 | 4 + 4 | 0.155, 0.175 | none |
| lying90 (`i7cork_ly`, `_r2`) | 27 + 18 | 0 + 0 | 4 + 4 | 3 + 1 | 0.282, 0.610 | grasp: arm link5 on the desk; grasp approach joint jump 0.23 rad; attach: witness / plan branch 1.9-3.8 rad after the turn (x2) |
| handover (`i7cork_ho6`, trunk_44a7fc7c) | 16 | 0 | 4 | 4 | 0.314 | none (+ 3 / 3 recollected demos of the earlier hand-over runs ho2 / ho3 / ho4) |
| handover replicate (`i7cork_ho_r2`, trunk_8e2ff5ef: chains re-checked with the partner) | 9 | 0 | 4 | 3 | 0.312 | symbolic_only x1 (joined, the geometric pose outside the tolerance) |

- brace_search false alone makes the current route's demos clean (8 / 8; before 3 / 4 and 2 / 3 with the brace finger's
  protected_contact); its rest rejections stay placement-dependent (1 / 15 and 23 / 47).
- lying90 removes the rest rejections (0 / 45; the route's kit: rest_stable 15 / 15) but its in-hand 90 deg turn costs: carry_failed 44 of
  61 carries checked, and 4 of 8 demos fail in the oracle (the IK branch of the attach after the turn, the low grasp next to the desk).
- The hand-over runs end to end in the oracle: 7 / 7 demos (1008-1232 steps) - right grasp / lift / carry to the hand-over / LEFT grasp
  of the held cork / right release / [axis-leg drive] / left carry / attach / retreat; no rest rejection (wide end down). Its sampler
  rejections are program-check failures of chains searched with the partner parked (the right carry to the hand-over, the pick, the left
  pre-grasp: 10 of 16 attempts); 8e2ff5ef re-checks those chains with the partner at each attach cell (replicate `i7cork_ho_r2`: program-check rejections 10 / 16 -> 1 / 9).
- The hand-over cost ~3x the current route's GPU-h per success here (0.314 vs 0.155 / 0.175), mostly the 300-360 s hand-over search per
  attempt; the current route's own cost is now low because brace_search false removed its demo failures.

### i7-cork round 2 (2026-11-08): fallback routes, the early rest test, the lying carry (owner decisions 5n.1-5n.3)

Branch i7-cork (from 2a25609b). Snapshots trunk_d693e45f, trunk_d0a5bb6f. Runs: shiyuc root `$SY/isgf_root`, staged v2 root
`$SY/v2/v2maps_int1_cork`, v2only, alias memory off, map overlay r3, N 4. Tabulator: `$OFH/cork_20261108/cork2_tab.py <tag>` (sample.log /
sidecars: rejections by phase, EARLY_REST, EARLY_REST_VS_LATE, CARRY_CHECK turns, fallback routes, demos).

**Owner decisions implemented** (each behind its own default-off switch; no default flipped):
- 5n.2 `ISG_ATTACH_EARLY_REST` = `0` (default) / `shadow` / `1` (18eedb22): the existing nudge test (`_nudge_tilts`, factored out of
  `check_rest_stability`, unchanged) run on the male right after `phase3_target` draws its rest: the male placed with gravity on, settled
  as `settle_and_validate` settles it (`settle_steps` + the tremble window), the 2 deg / 300 step / 3 deg nudge, the world restored.
  `shadow` compares it with the late test (kit `early_rest.vs_late.*`) and repeats the late nudge on the same settled state
  (`early_rest.late_repeat.*`, d0a5bb6f); `1` ends the attempt at the draw (`rest_unstable`, counter `rest_unstable_early`). ~12 s per test.
- 5n.1 `ISG_ATTACH_FALLBACK=1` (d693e45f): the current route primary; when its drawn rest fails the nudge test (the early test, run as
  the route decision), the same attempt goes on with lying90, then with the hand-over (also after any sampling-stage rejection of
  lying90); each route's template (grasps, rest, bands / maps, stances) built once in `sample_template`; each route gets the attempt budget
  afresh; kit / counters `attach_fallback.{used.<route>, <route>.rest_unstable, <route>.rejected.<reason>, exhausted}`,
  `summary.attach_fallback`.
- 5n.3 `ISG_LYING90_MIN_TURN=1` (5fdf3ad8, ladder d0a5bb6f): on the lying route each partner cell is offered the yaw whose attach makes the
  carry's in-hand turn smallest (`min_turn_yaw`, closed form, +-15 deg jitter) and 30 deg offsets around it; the batched screen keeps per cell
  the smallest-turn yaw it reaches, the exact checks take cells by that turn. Carry diagnostics on every route (no decision): `CARRY_CHECK`
  (turn, outcome), kit `carry_turn.<route>.<ok|failed>.<lt100|lt140|ge140>`, `ATTACH_BRANCH_GAP` (carry end vs the attach witness).

**5n.2 check: the early test is not faithful to the late one on this cork - not enabled.**

| shadow run | compared | agree stable | agree unstable | early unstable / late STABLE | early stable / late unstable |
|---|---|---|---|---|---|
| `i7cork2_cur` (base 620500) | 34 | 1 | 25 | 3 | 5 |
| `i7cork2_cur_r` (base 620600) | 15 | 2 | 9 | 3 | 1 |

- 6 of 49 attempts the late test passes would be rejected early, and 5 of the 7 written instances (all 7 demos ok) had an early
  "unstable" (tilts 3.3-84 deg) - the condition "rejects nothing that passes today" fails.
- The cause is the test, not the timing: the late nudge repeated on the very same settled state disagrees with itself 2 / 15
  (0.00 -> 85.3 deg, 0.00 -> 21.0 deg); early-vs-late agreement 37 / 49 (0.76) is barely above two independent draws at the ~80 %
  unstable rate (0.68). The male-end-down cork on the desk is a coin flip per nudge; no early threshold separates the classes (agreeing
  unstables at 3.4-6.5 deg, disagreeing at 8.3-84 deg).
- On the current route the late rest rejections now cost 52-61 s per attempt (brace_search false), not 189 s; the early test costs ~12 s.

**5n.1 A/B (N 4):**

| arm | tag | att | written | ok | GPU-h | GPU-h / success | rejections |
|---|---|---|---|---|---|---|---|
| current route alone (+ early shadow) | `i7cork2_cur` (620500) | 59 | 3 | 3 | 0.82 | 0.272 | rest_unstable 30, target_placement 23 (instance 620503 out of its 32 attempts) |
| current route alone (+ early shadow) | `i7cork2_cur_r` (620600) | 25 | 4 | 4 | 0.65 | 0.164 | rest_unstable 10, target_placement 9 |
| fallback (+ min-turn ladder) | `i7cork2_fb2` (620500) | 20 | 4 | 3 | 1.30 | 0.433 | target_placement 12, rest_unstable 1 (late), program check 3 |

- fb2 counters (per attempt): the current route's rest failed the early test in 8 of 20 attempts; lying90 then drafted 2 and was
  rejected 6x (target_placement); the hand-over drafted 3 and was rejected 3x; 3 attempts exhausted all routes; 3 current-route drafts.
  Written: current 1, lying90 1, hand-over 2; demos ok 3 / 4 (the hand-over 620501 symbolic_only).
- More written per attempt (0.20 vs 0.05 / 0.16) but each fallback attempt runs lying90's and the hand-over's searches (sample wall
  3779 s vs 2896 s; rejected-attempt seconds 1968 vs 1705 for a third of the attempts): the GPU-h per success is worse than the current
  route alone. Its trigger is the coin-flip nudge above (it switches away from current-route draws the late test would pass).
  Recommendation: keep the switch off; the current route alone stays the production route.

**5n.3 diagnosis: why the lying carry fails.** The carry check is `waypoints_ok([lift] + SLERP to the pre-attach, track_dense)`. Its
failures (61 checks in i7cork_ly / _r2, 44 failed; 35 / 30 in i7cork2_ly) fall in two classes:
1. **The lift itself has no IK** ("first waypoint fails the exact IK"): 20 of 37 in i7cork_ly, 20 of 30 in i7cork2_ly, 12 of 23 in
   lymt2 - every one at a second base pose (stance `attach`; 3 at manipulation in lymt2), the SAME lift (eef 0.81-0.86 m base-forward,
   z_top + 0.08) failed once per partner cell in one attempt. At a second base pose the lift is `grasp + carry + base shift` (the eef at the
   same base-frame pose) while the stance's trunk differs; for a lying cork grasped at the reach limit that pose is out of reach at the
   new trunk. It does not depend on the partner cell (only the support is ignored; the partner sits >= 8 cm away), so one attempt spends
   ~20 identical failures. Not the in-hand turn.
2. **The turn** (screening IK of the mid waypoints, `approach_untrackable` of the tracked chain): the lying route's goal turn about the
   vertical insertion axis follows the partner's yaw, drawn uniformly, so the recorded turns were 124.9-179.0 deg (8 instances i7cork_ly /
   _r2; 127-150 in i7cork2_ly) where the route's own turn is the 90 deg tilt. i7cork2_ly carry outcomes by turn: 100-140 deg 2 ok / 9
   failed, >= 140 deg 3 ok / 21 failed. The demo failures of i7cork_ly_r2 sat at the largest turns (162.8-179.0: 3 plan_failed).

**5n.3 fix A/B** (`i7cork2_ly` LYING90 vs `i7cork2_lymt2` LYING90 + MIN_TURN ladder, base 620500): turns moved down (lymt2 drafted
89.6-119.3 deg; checks < 100 deg: 2 ok / 4 failed, 100-140: 5 / 12, >= 140: 0 / 7) - the min-turn attach itself is mostly out of reach
from the stance (the first, single-yaw version `i7cork2_lymt`: 0 written in 23 attempts, the pairs failing the screen; cancelled). Excluding
the lift class, carry checks ok 7 / 18 (lymt2) vs 5 / 15 (ly). Yield: ly 24 att / 4 written / 4 ok, 0.160 GPU-h per success; lymt2
32 / 4 / 2 ok (2 symbolic_only: the cork joined but the bottle carried 34-42 cm off its place after the attach), 0.487. **The turn fix does
not help on its own**: the lying route's carry failures are dominated by the second-base lift (class 1) and the min-turn attach
orientations are hard to reach; the switch stays off. Note i7cork2_ly's own demos were 4 / 4 (i7cork_ly / _r2: 4 / 8).

**5n.3 attach configuration after the turn (diagnose only, shared code; owner to decide):**
- i7cork_ly_r2 620402 / 620403 (turns 172 / 163 deg): after the reorient the arm stood with right_arm_joint5 AT its lower limit (0.004-0.006
  rad) and joint2 at its upper limit; the in-hand slip (2.1-2.5 deg) made the oracle re-aim the pre-place, which the differential IK could
  not reach from the limits; the insertion stalled (waypoint 6/8 or 8/12) and the finish toward the recorded attach witness was refused
  (3.14-3.81 rad away: "pre-pose and end-pose witnesses in different IK branches").
- In the sampler the attach witness is the exact IK of [pre-attach, attach] solved fresh (`ikc.exact`), and the carry is tracked from a fresh
  IK solution of the lift (`waypoints_ok` without `q_first`): neither is continued from the other, and no joint-limit margin is asked at the
  carry's end. `ATTACH_BRANCH_GAP` (carry end vs attach witness, max arm joint): current route drafts 0.19-3.25 rad (11 of 26 > 0.6 in
  i7cork2_cur, all its demos ok), lying 0.49-2.76 - the gap alone does not predict the failure; the limit at the carry's end does.
- The low grasp on the desk (i7cork_ly 620301): the oracle's straight joint ramp to the lying cork's pre-grasp (trajopt failed with the
  supports present) cleared the planner world at 10 mm but right_arm_link5 touched the desk for 2 steps; the sampler checks the grasp
  configurations (non-finger links vs the support with the policy's sphere margin), not the swept arm on the way in.
- Fix proposals (each touches every attachment alias - not done): (a) solve [pre-attach, attach] from the carry's tracked end (`q_first`)
  and reject when it leaves that branch; (b) a joint-limit margin on the carry's end configuration; (c) at a second base pose, the lift
  check once per stance (or the lift from the frozen arm's FK at the new trunk, as the oracle executes it) instead of once per partner cell.

**Defaults / decisions needed:** none flipped. EARLY_REST, FALLBACK and MIN_TURN stay off. The full suite (23 failed) fails identically with
the base attachment module (all 23 in unrelated modules: ground_isg, support_candidates, ag_release_hook, v2_core ...).

## i7-contact3 (owner answers 2026-11-08 3.1 / 3.2 / 3.6 / 3.7; branch `i7-contact3` from of-v2maps cf1f5d2d)

Owner decisions implemented here (verbatim in `tmp/outputs/owner_answers_20261108.md`): 3.1 full-open brace CONFIRMED as the default
(unchanged); 3.2 the close_02 slide diagnosis approved; 3.6 "cap the place depth by the push map's reach for all affected aliases";
3.7 "remove the pack brace. use the board variant for the block design".

**3.2 close_02 (diagnosis; the fix needs the owner).** The ~1 cm "slide" is not friction, push force or the brace opening. Rung 0 of every
`top_press` braced push already ends inside the lid's 0.1 rad tolerance (residual +0.002-0.016 rad, carton drift ~5e-6 m), but
`Executor.push_outcome` aims at -0.05 +- 0.04 rad (`brace_retry_aim_bias_rad` 0.05, `brace_retry_aim_frac` 0.4 x the 0.1 tolerance), so a
small positive residual is "not aimed"; the top press touches the carton all along (brace hits 117-186) -> "short" -> the retry ladder
re-runs with `offset_brace` +5 / +10 mm "outward" along -direction, which for a top press (direction (0, 0, -1)) LIFTS the hand; at
+10 mm the press no longer holds the carton and it slides with the push until the hard object_tilt QA (10 mm from the first pose)
aborts (hence 10.0-12.1 mm). All 20 top_press object_tilt failures (i7_artic3, icb_off / on, int1_6) follow this rung pattern; the
opposite_face braces climb to +20 mm and slide 5.0-10.2 mm. The September demos passed because the carton tolerance was 0.2 rad (aim
-0.05 +- 0.08 admitted +0.006). Re-run of int1_6 300006-300009 (trunk_fcddc495) with the existing knob `ISG_AIM_FRAC=1.0`: 3 / 4 ok, no
rungs, peak carton 3.3-6.2 mm (300006 stays protected_contact: the left finger touches the carton in the push's pre-contact). Record
`$OFH/contact3_20261108/close02_diag` (scan.py / scan_out.txt / scan_recollect.txt).

**close_02 fix, option (a) (owner answers b item 2: an approved oracle change limited to this).** `Executor.push_outcome(push, ctx,
brace=)`: with `ISPATIALGYM_TOP_PRESS_ACCEPT` (default "1" since d555c53d; "0" = the ladder for every brace) an in-tolerance push braced by a
`top_press` is accepted as it is - no improvement rung, so `offset_brace` never raises the press; every other brace kind and every push out
of tolerance keep the ladder. Counter `result.counters["push_outcome.top_press_in_tolerance_accepted"]`; tests `test_top_press_accept.py`.
A/B (trunk_2844d8c2, cluster, close_02 300006-300009): fresh v2only 0 / 4 -> 3 / 4 (0.167 GPU-h per success), the int1_6 instances
re-collected 0 / 4 -> 3 / 4; object_tilt 3 -> 0, the carton's peak 10.0-11.7 mm -> 3.3-6.2 mm. The remaining failure (300006,
protected_contact; diagnosis only): the QA protection run-check exempts only the contacts present when the push's pre_contact span opened
and a `press_hold` hand, not an active Brace, although `brace_allowed` lists the pair - the left top-press finger's contact with the carton
starts 19 steps into the right arm's pre_contact (step 409) and reads as protected_contact (the carton moved 3.2 mm). Candidate fix (oracle
QA rule, owner): the active brace's allowed pairs as the transit's hold exemption.

**3.6 place depth cap (`ISPATIALGYM_PLACE_DEPTH_CAP`; "0" off, "shadow" counts only, "1" on).** `IORSSmall.place_depth_caps`: the touching
target is placed at its start depth (level with the partner) and pushed from an A1 push stance on the same edge and trunk; per
manipulation candidate the push stances at A1's stand-offs are whole-body collision-checked (the base's real minimum distance to the
support), and from each free one the target's touch_push map reaches roots up to the deepest feasible cell (a present class, any yaw /
lateral) + half the lattice; cap = max over the free stand-offs of that reach - the stance's edge depth. `place_depth_cap_filter` (before
the candidate draw in `sample_instance`): on - candidates whose focus depth exceeds the cap leave the draw; shadow - kept, the would-be
removals' outcomes counted per candidate / attempt. Affected aliases: touching_01, touching_02, touching_04 (iors_small touching, mapped
push after the place). Not affected: touching_03 (shelf layout: no drawn depth), touching_large_01-03 and the under_large push-under (floor
pushes: no floor_push map), axial (no place: the start placement is already map-driven), close / open push (no place).
A/B (trunk_5aeb9da4, v2only N 4, indices 320100-320103, memory off; record `$OFH/contact3_20261108`): shadow (= off sampling) - the cap
would remove 196 / 508 candidate draws on touching_02 and 214 / 369 on touching_04; of the would-be removals tried, every one that
reached a push test was unreachable (touching_02 85 / 85: 83 A1 partner-draw contact_unreachable + 2 drafted, both contact_unreachable in
goal_check; touching_04 58 / 58), the rest died earlier (grasp_unreachable 60 / 77, 4 other), none was written. On: removed 108 / 297 and
188 / 315 (all_removed attempts 30 / 27, rejected in ~1 s). Results shadow -> on: touching_02 118 att / 2 wr / 2 ok, 0.40 GPU-h per success
-> 78 / 3 / 3, 0.18; touching_04 61 / 4 / 2, 0.66 -> 53 / 4 / 2, 0.56; total 0.53 -> 0.33. DEFAULT ON (the owner decided the rule) in a
separate commit; "0" restores the draw of before. (The attempt-level shadow counters are counted after the attempt's kit line; the drafted
attempts' outcomes were read from the rejection lines.)
block_design_01 base (trunk_fcddc495) vs pack removed (trunk_5aeb9da4), v2only N 4 at 320100-320103: identical - 60 / 60 attempts, the same
rejection sequence, 0 written in both: every attempt target_placement / board_placement ("off the top" at board positions (0.46, -0.02),
(0.44, -0.10), (0.48, 0.04); "no pitched place reaches the pockets' release aims" at (0.52, -0.06), (0.60, 0.05)) - a pre-existing blocker
of block_design_01 under v2only on the integration trunk, unrelated to the pack removal.

**Touching follow-ups (lead 2026-11-08, from i7-diag's touching_04 diagnosis; all default OFF, the owner decides).**
(a) `ISPATIALGYM_TOUCH_LIFT_SCREEN` (shadow / 1): the goal check's carried-drive lift test (box over the partner's top) at the draw, only where
the goal check is certain to make it (carry rule partner, or auto with every A1 push stance a real second base pose whose drive passes
within `carry_clearance_m` of the partner). (b) `ISPATIALGYM_PLACE_DEPTH_CAP_BASIS=a1`: the cap's verdict = A1's own push stance test at the
focus. (c) `ISPATIALGYM_PLACE_DEPTH_CAP_REDRAW=1` (start re-drawn while all candidates are beyond the cap) and `ISPATIALGYM_PLACE_DEPTH_CAP_DRAW=1`
(Phase 2's pairs cut to the cap, `keep_pairs`; in the SPA cell key when on). A/B (cluster, v2only N 4, 320200-320203, cap on): (a) goal_check
lift_unreachable touching_04 6 -> 2 (118 s each), touching_02 12 -> 5 (~30 s), 46 candidates screened at the draw; in shadow all 18 would-be
rejections failed at or before goal_check (0 passed it). (b) verdicts equal to the map basis in 805 / 808 candidates; the A1 failures at
trunk (0.0, -0.61) fail at the partner-drawn place, the (0.25, -0.53) place-screen failures are exact IK: no gain, not recommended. (c)
redraw: no effect (no start cell under the cap). DRAW: all-removed attempts 47 -> 0, written per attempt touching_04 0.044 -> 0.11,
touching_02 0.045 -> 0.16; GPU-h per success 1.22 -> 0.61 / 0.15 -> 0.20 (demo success N 4 noise).

**Owner decisions 2026-11-08 c, 5r (i7-contact3).**
- 5r.1 defaults (each its own commit): `ISPATIALGYM_PLACE_DEPTH_CAP_DRAW` default "1" (e60b4dc5; "0" = the draw of before; the SPA cell key
  keeps its entry when on) and `ISPATIALGYM_TOUCH_LIFT_SCREEN` default "1" (fcb75869; "0" off, "shadow" counts only).
- 5r.2 (an oracle QA rule, approved for exactly this): `ISPATIALGYM_QA_BRACE_PROTECT_ALLOWED=1` (913e8692, default off). While a brace is
  active, `Monitors._check_protected` skips a new contact whose (link, object) pair is one of the running action's `allowed` pairs
  (`Executor.brace_allowed_pairs`: the brace hand's finger / gripper links on the object it braces, set only while a brace's span contains the
  action), limited to the links of an arm that does not move in the action (the other arm's move). Every other pair keeps its rule; the
  1 cm movement check is unchanged; no active brace = nothing changes. Stats event `protect_brace_allowed` (once per pair and span) + a log
  line. Tests (test_protection.py): the listed pair is allowed; unlisted pairs (moving arm's finger, brace-arm wrist, brace fingers on another
  protected object, the brace arm's own move) fail; no brace / switch off unchanged.
- touching_04 follow-ups (b4b4e9aa, default off): `ISPATIALGYM_TOUCH_A1_SCREEN=shadow|1` - the goal check's push stance search at the draw as
  far as it is certain: A1's map-feasible, collision-free push stances (the same `push_stance_candidates` call, first `push_stance_max_exact`)
  must hold one from which the carried box's place passes (`place_infeasible_why`, the goal check's own "place:" verdict); none =
  contact_unreachable at the draw. No A1 stance (the goal check then falls back to laterals), the shelf layout, the tip release: not screened.
  Kit `touch_a1_screen.*`. In c2 the 5 goal-check contact_unreachable (~129 s each) were 2 x "every A1 stance's place fails" (what the screen
  catches) and 3 x "the only A1 stance (-0.08, s0.30) at trunk (0.25, -0.533) is map-feasible but has no exact class" (hand overlap
  -15 mm, IK). `ISPATIALGYM_TOUCH_BUILT_TILTS=1` - `tabletop.phase2_tilts`' hook: the touching Phase 2 (plain and split cone) scans the
  touch_push map's eligible trunk tilts at each translate within the torso limit instead of `tilt_scan`'s 7 continuous samples (at translate
  0.25: 0, -0.266, -0.533, ...; at 0: 0, -0.305, -0.611, ...), so the manipulation and push stances read built columns; in the SPA cell key
  when on; kit `touch_built_tilts.*`.
- RUNS (trunk_b4b4e9aa, cluster svl titanrtx, v2only on v2maps_int1, memory off; `contact3_20261108/submit_r1.sh`, tabulate `tab_r1.py`).
  Suite: 23 failed + 3 errors = base 1697372a's exact set, +5 new tests pass. 5r.2: 300006 re-collected (c3_tp1's sampled instance) with
  the rule on = SUCCESS (carton 0.9 mm, qa {}; the rule acted once, left_gripper_finger_link1 on the carton at step 409 of the right push's
  pre_contact), =0 control = protected_contact as before; close_02 fresh N 4 at 300006-300009 (c3_qa_on) 4 att / 4 wr / 4 ok, 0.55 GPU-h
  (0.138 per success; c3_tp1 before: 4 / 4 / 3, 0.167); the rule acted only in 300006. 5r.1 rerun with both defaults on (c3_r1_base,
  320300-320303): touching_02 56 att / 4 wr / 3 ok, 0.58 GPU-h (0.19 per success; lift screen rejected 47 candidates at the draw, goal-check
  lift_unreachable 3); touching_04 46 / 4 / 1, 1.33 GPU-h (lift screen 7 at the draw, goal-check lift_unreachable 1; the largest late cost
  is now goal-check contact_unreachable, 8 x 139 s). touching_04 follow-ups vs that base (same snapshot / indices): built tilts
  (c3_r1_tilts) 55 / 4 / 1, 1.34 GPU-h - the switch acted (4725 scans; candidates at (0.25, -0.6) / (0.0, -0.6) / (0.5, -0.3)) but the
  same failures moved with it: place screening from the push stance 23 at (0.25, -0.60) (base 22 at (0.25, -0.53)), goal-check
  contact_unreachable 11 (base 8) - the cause is the exact place IK from the push stance at that translate / deep tilt, not the unbuilt
  column: NOT recommended. A1 screen (c3_r1_a1s) 46 / 4 / 2, 0.99 GPU-h (0.50 per success): checked 22, passed 16, rejected 6-7 at the
  draw (48 place checks, ~1-2 s each); goal-check contact_unreachable 8 -> 5 (the 5 left: 3 "no class" at the one A1 stance (-0.08,
  s0.30) - the map says feasible, the exact push fails on hand overlap -15 mm / IK; 2 where a placing stance exists but not the one whose
  push passes). Note: the a1s / a1sh arms read the SPA cells base cached (same key), so their sampling is cheaper; judge on the counters.
  A1 screen shadow (c3_r1_a1sh) 46 / 4 / 2, 1.08 GPU-h: 3 would-be rejections, all 3 attempts died contact_unreachable in the goal check
  (0 passed it); goal-check contact_unreachable 9 (on: 5). Recommendation (owner decides): the A1 screen default on (moves ~130 s
  rejections to the draw, rejects nothing that passes); the built tilts stay off. Demo failures (all arms): gripper_out_of_fov,
  plan_failed, symbolic_only (the push turn below), not switch-related.
- symbolic_only diagnosis (diagnosis only; `$OFH/contact3_20261108/sym_diag`: sym_scan.py, rows.json, SUMMARY.txt): over 87 touching_04
  demos with a push (40 tags) 38 failed symbolic_only, every one on the spray bottle's ORIENTATION, turned during the push: median |dyaw|
  8.8 deg (failures) vs 0.8 (successes), limit 5.7. It turns while it slides (0 -> 45 mm in 0.4-0.6 s), before it reaches the detergent
  (c2 320203, no brace: the detergent moved 0.0 mm, the bottle -16.5 deg). The push line runs through the FOOTPRINT centroid
  (`swept_push_candidates`), +3.7..4.2 mm (median) off the object's root; in 37 of 46 turned demos the turn's sign is the one an off-centre
  push gives; |offset| 6-9 mm: 13 / 16 turned > 5.7 deg, 0-3 mm: 7 / 23. Higher contacts tip the bottle more (max tilt median 5.8 / 11.4 /
  17.3 deg for height fraction < 0.22 / 0.22-0.33 / > 0.33); the 5 cm pushes (0.18) with offset < 3 mm: 5 / 5 ok. The brace is always on the
  partner (detergent) and does not matter (none 25 / 54 failed, brace_tip_h-_p15 9 / 18). c2's two failures: 8.2 / 8.0 cm (0.30 / 0.29),
  offset +7.9 / +5.5 mm, -15.0 / -16.5 deg, both at trunk (0.25, -0.533); lower classes failed the hand-overlap screen. Fix candidates for
  the owner (not implemented): the push line through the mass centre / root; lower contacts first; a yaw guard in the push; a guide on
  the target instead of the partner brace.

**Lead decisions 7-10 (2026-11-08, owner away; tmp/outputs/lead_decisions_while_away.md).** bba2a9bd `ISPATIALGYM_TOUCH_A1_SCREEN`
DEFAULT ON; 015b815a `ISPATIALGYM_QA_BRACE_PROTECT_ALLOWED` DEFAULT ON (each its own commit; snapshot trunk_015b815a; suite = base's set);
`ISPATIALGYM_TOUCH_BUILT_TILTS` stays off. Decision 9 (the spray bottle's turn): 4822246b `ISPATIALGYM_PUSH_LINE_COM=1` (default off) - the
swept and open-gripper push classes' contact line runs through the target's mass centre (`alignment_axial._com_local`: PhysX link COMs,
mass-weighted; the AABB centre without masses) at its place instead of its footprint centre (`contact_poses.push_line_t0`,
`IORSSmall.push_line_xy`); the extras record `push_line.offset_from_footprint_m`; kit `push_line_com.used`. The touch_push maps / library keep
the footprint line (the exact chain decides). Tests (test_contact_poses.py): the contact shifts across the push by the point's offset with the
heights and classes unchanged; off, the golden records are unchanged; the sampler's point (off None, on the cached COM at the pose). The
default and the other options (lower contacts, a yaw guard, a target guide) stay the owner's.
A/B (trunk_4822246b, both new defaults on in both arms, cluster svl titanrtx, touching_04 v2only, two N 4 blocks per arm 320400-03 / 320410-13,
`contact3_20261108/submit_pl.sh`; turns from `sym_diag/sym_scan.py` -> rows_pl.json): off (c3_pl0) 73 att / 8 wr / 4 ok, 2.03 GPU-h = 0.51 per
success, symbolic_only 2; on (c3_pl1) 80 / 8 / 3, 2.03 = 0.68 per success, symbolic_only 2. The bottle's turn during the push (the pushes that
moved it; 2 + 1 plan_failed demos never pushed): off 7 demos |dyaw| median 2.1 deg, max 18.4, 2 above 5.7; on 6 demos median 5.2, max 15.2, 2
above 5.7. The switch acted (the COM is 4 mm off the root in the root frame; the line moved 0.4-2.8 mm, the contact's offset from the root
3-5 mm -> 1.7-2.9 mm) but the turn did not change: the same instance turns -18.4 (off) / -15.2 deg (on) at push_tip_v- 8.2 cm with a 17 mm
side slip and a 14 deg tilt. So the line offset is not the main cause at these magnitudes; the tipping and sliding at mid-height contacts
remain (the 5 cm pushes turn < 2 deg in both arms). Not recommended as a default; the other options are the owner's.

**3.7 pack brace removed.** The block design packing tail (wide pack brace + pushes: `block_design._pack_tail / _pack_stage /
_pack_pushes / _pack_face_half / _pack_posture_variants`, `pack_brace_opening`, `PACK_BRACE_END_MARGIN_M`), the oracle's
`_pack_stage_actions` (a sidecar that still carries `pack` is refused: SidecarError), the program check's pack push handling
(`advance_push`, `_is_pack_push`) and `SidecarDraft.pack` are gone. Customs: the four pack-tail variants (block_4 breakfast_table /
coffee_table / conference_table / desk; BDDL problems 32 / 7 / 76 / 6 keep their ids) carry `superseded_by` the board variant
(`block_4-breakfast_table_board`), `place_gap_m` / `_pack_note` dropped, `block_pack_tail: true` kept as the marker the sampler refuses
(Reject spec); the board variants lose their `block_pack_tail: false`. No library / map existed for the pack brace. block_design_01 =
the board variant (never ran the pack).

**5y: the pre policy refuses on any failure and is the default; the map / live brace disagreement (owner 2026-11-08 5y).**
Owner decision: "(1) the sampling brace pre call refuses on ANY failure (not only the support policy), then
ISPATIALGYM_AXIAL_BRACE_PRE_POLICY -> DEFAULT (own commit), with an N 4 confirm on axial_05 / 01 / 03 / 04; (2) diagnose (CPU first) why the
axial_brace map and the live check disagree on the brace pose ... Do not rebuild maps or change the builder without asking."

- Checked first: neither was on the branch (384c25de refused on `support_policy` only, default off).
- **12a1a49a**: `_brace_pre_policy` refuses a brace candidate on ANY failure of the program check's pre call (the support policy,
  `ik_joint_limit`, no IK, ...); a refused optional brace still leaves the attempt push-only. Counters `axial_brace.pre_policy.refused`,
  `refused.<label>`, `refused_reason.<key>` (the reason's head before its first `:`), `push_only`, `other_brace`; `other_fail` is gone.
  Unit test: a joint-limit pre failure is refused (and any other IK failure).
- **2de8f35d**: `ISPATIALGYM_AXIAL_BRACE_PRE_POLICY` DEFAULT ON (its own commit; `=0` turns it off).
- Tests `test_axial_brace_pre_policy.py` 5 + `test_axial_settled_rest.py` 16 pass; pyflakes clean; suite at 2de8f35d 23 failed + 3 errors,
  the same 26 ids as 384c25de. Snapshot trunk_2de8f35d.
- Confirm run `i7ar_pd` (no pre-policy env = the default): see the table below.

**5y.2 diagnosis (CPU): the brace POSE is the same; the map is read at another trunk.**

- Same pose. Rebuilt the live candidates of the 4 written i7ar_probe draws (`contact_poses.axial_brace_candidates` from the recorded
  stance / target pose / spin / goal yaw) and the map's own chain for the same class (`contact_maps.chain_world`: the library entry's
  `T_rule_object` with the root at the live goal root xy, the live support top). In the base frame the pre and the contact agree to
  < 0.1 mm and 0.01 deg for every class (blade_0 / blade_0- / blade_30 / blade_30-), including the settled rest's 0.087 rad tilt (the
  entry's `T_rule_object` carries q_rest; the blade's approach is straight down in both). Script `$OFH/axialrest_20261108/diag5y/cmp_pose.py`.
- Different trunk. `ContactMap.lookup` reads the NEAREST built trunk tilt (0 / -0.3 / -0.6 / -0.9), but Phase 2's stances take continuous
  tilts (`tilt_scan`: 7 samples from 0 to the torso limit). The axial_05 draws stand at translate 0.5 / tilt -0.454 (3 of 4) - 0.146 rad
  from -0.6 and 0.154 from -0.3 - and at 0.0 / -0.916. The reach maps got the inherited / self-correcting tilt columns for this
  (`map_tilt.py`, i7-plateqtf); the contact maps did not.
- At 310002's goal cell (0.83, 0.00), translate 0.5, every d_e: blade_0- is `feasible` at tilt -0.6 only - `ik_pre` at -0.3, `reach_bound`
  at 0, `branch_jump` at -0.9 (blade_30-: feasible at -0.6, `ik_grasp` at -0.3). The stance at -0.454 lies half way: the map's "feasible"
  is the -0.6 column's, and the live IK (fresh, 32 seeds) fails there as the -0.3 column does.
- How often neighbouring columns agree (whole axial_brace map, d_e 0.45): P(feasible at the next built tilt | feasible) = 0.36-0.62 at
  translate 0.5 (of 1348 feasible entries at -0.6, 36 % are feasible at -0.3), 0.22-0.74 over all translates. Neighbouring 8 cm cells agree
  0.62-0.67 (the live root is up to 6 cm from its cell: 15-44 mm for these draws). These are boundary poses (left_arm_joint2 near its
  limit), so a column 0.15 rad away is close to a coin flip. Scripts `trunk_status.py`, `tilt_agree.py`, `cell_agree.py` (same dir).
- Second difference: the live exact IK keeps a 0.05 rad joint-limit margin (`IKChecker.JOINT_LIMIT_MARGIN`, `away_from_limits`), and the map's
  cuRobo IK keeps none. In the probe, 2 of the 6 blade_0- empty-world failures were `ik_joint_limit` (left_arm_joint2 within 3-22 mrad).
- GPU decomposition with the MAP'S OWN IK (diagnosis only, no map written; `diag5y/brace_ik_decomp.py`, job 18385226, `decomp.out`): the
  map's problem (A: lattice cell x map trunk), the live root xy at the map trunk (B), the live root xy at the live trunk (C), 64 seeds,
  best solution's joint-limit margin (left_arm_joint2 every time):

| draw / class (map status) | A map's problem | B live xy | C live xy + live trunk |
|---|---|---|---|
| 310002 blade_0- (feasible) | 26 / 64, 0.062 rad | 18 / 64, 0.061 | **0 / 64 (no IK)** |
| 310003 blade_0- (feasible) | 26 / 64, 0.063 | 26 / 64, 0.018 | **0 / 64** |
| 310000 blade_30- (feasible) | 28 / 64, 0.017 | 28 / 64, 0.029 | pre only, no chain |
| 310001 blade_30 / 30- (feasible at -0.9) | 15 / 19 of 64, 0.003 / 0.022 | pre only, no chain | pre only, no chain |
| 310002 / 310003 blade_30- (feasible) | 0.003 / 0.003 | 0.003 / 0.003 | 0.17 / 0.13 (live: fails the 3 mm target check) |
| i7ar_on 310001 opp_blade_0 (feasible) | 0.044 | 0.056 | 0.060 (live: "exact IK at pose 1" in the full world) |

- So: of 8 map-feasible (draw, class) pairs, 6 sit below the live 0.05 rad margin already at the map's own problem, and only 3 are
  solvable with the margin at the live xy and trunk; the owner's example (blade_0-) is lost to the trunk alone (0.146 rad).
- Proposed (owner; nothing changed, no map rebuilt): (a) builder - accept a map IK solution only with the live 0.05 rad joint-limit margin
  (rebuild); (b) sampler, no rebuild - read a contact map at a tilt more than 5 mrad from a built tilt as the INTERSECTION of the two
  bracketing columns, or snap the axial manipulation stance to the brace map's built tilt (camera rules re-checked, as TILT_SNAP did for
  grasps); (c) rebuild with Phase 2's `tilt_scan` tilts as columns and a 4 cm lattice for these boundary-pose classes. (a) + (b) together
  make the map's "feasible" mean what the live check accepts.

**5y.1 confirm** (`i7ar_pd`, trunk_2de8f35d, the default; same indices / root / memory / overlay / switches as 5v; jobs 18384731-34, all
completed first run). att / written / ok, GPU-h per success:

| alias | i7ar_pol0 (off) | i7ar_pol (support policy only) | i7ar_pd (any failure, default) | where it acted (i7ar_pd) |
|---|---|---|---|---|
| axial_05 | 20 / 4 / 3, 0.093 | 20 / 4 / 3, 0.087 | 20 / 4 / 3, 0.10 | 310000 att 10: left_h x2, left_h- (support_policy) -> push-only |
| axial_01 | 9 / 4 / 3, 0.090 | 11 / 4 / 2, 0.17 | 9 / 4 / 3, 0.083 | 310003 att 1: opp_blade_45, opp_blade_60- (ik_joint_limit) -> another brace |
| axial_03 | 6 / 4 / 4, 0.045 | 8 / 4 / 3, 0.067 | 4 / 4 / 3, 0.057 | 310002 att 0: opp_blade_45- (ik_joint_limit) -> another brace, written, success |
| axial_04 | 16 / 4 / 3, 0.067 | 18 / 4 / 4, 0.050 | 16 / 4 / 3, 0.060 | 0 checked |
| all | 51 / 16 / 13, 0.071 | 57 / 16 / 12, 0.082 | 49 / 16 / 12, 0.075 | 13 checked, 7 passed, 6 refused (support_policy 3, ik_joint_limit 3) |

- Push-only fallbacks 1 (axial_05 310000 att 10, which then dies at the program check on the push - "push pre-contact witness 3.16 rad" - as
  in every arm); other_brace 2. Program-check brace ("pre:", left arm) rejections: i7ar_pd 0, i7ar_pol 1 (03 ik_joint_limit), i7ar_pol0 2
  (05 support_policy, 01 ik_joint_limit). axial_01 310003 att 1 then fails at the program check on the push; 310003 is written at attempt 3
  in both arms. The default removes every brace rejection at the program check; success at N 4 is within noise (12 vs 13; the failed demos
  - 03 310003 plan_failed, 01 310003 not_reached - are not the attempts where it acted).

## Goal-surface reach maps for the IORS large predicates (branch i7-goalmaps, 2026-11-08)

**Owner decision** (2026-11-08, verbatim prompt `tmp/outputs/prompt_goalmaps_axialrest_overpath.md` section 1; `$OFH/DECISIONS.md`
"OWNER DECISIONS (verbatim prompt ...)"): change the height-anchor rule so the reach maps cover every goal surface an IORS large predicate
needs - the partner (goal-surface) rows become surfaces of their target model with the same measurement rules (inventory top, floor datum,
canonical key, 10 mm window); the existing anchors and keys stay; only new anchors; build the missing goal maps, stage them in a COPY of
`$OFH/v2maps_int1`, validate N 4 v2only. No height union (a map at one height serving another).

### Audit (before the code change)

**What the goal side reads, per predicate** (the sampler, `--grasps v2only`; `iors_large*.py`):

| predicate (large) | goal side | map kind it reads | surface (height) | role row |
|---|---|---|---|---|
| ontop | carry + place on the partner's top (stance C, `IORSLarge._placement_candidates_at`: `band("tabletop", partner)` -> `v2_band` -> `tabletop.map_bands` at `band_top(partner)` = the live place top) | **tabletop** (anchor height) | the partner's top, live `top.z - world.floor_z` (1e-4) | `partner` rows (candidates.csv ontop_large_scale; representatives.csv ontop_large_scale#NN) |
| inside | carry + place into the partner's compartment (`InsideMode.inside_band` -> `tabletop.interior_map_bands`; PLACE rule `inside_place_rule`, or a lean goal rest) | **interior** (registry spec, no anchor axis) | the registry spec's board / basin (`assets/interiors/interiors.json`, measured 1e-4) | `partner` rows (representatives.csv inside_large_scale#NN) |
| touching, `touching_place` "surface" | as inside, on the partner's board (`SurfaceTouchingMode` = InsideMode) | **interior** | the spec at the variant's `inside_surface_z_m` | `partner` rows (representatives.csv touching_large_scale#NN) |
| touching (floor push, default) | drop on the floor + floor push to the face (`TouchingMode`, push-under machinery: exact IK from C) | none (exact) | the floor | - |
| under (push-under, `under_mode` "push") | drop beside the partner + floor push under it (`push_under_candidates`: exact IK from C; no `band()` call) | none (exact) | the floor under the partner | - |
| nextto | the base pushes the target (no grasp) | none (exact) | the floor | - |

The init side (stances A / B) reads the init support's tabletop map or the floor map (h 0) - not part of this decision.

**Per alias** (the 15 IORS large aliases; MAP_REQUEST lines over `$OFH/out/*/inter_object_relation_satisfaction-*large*/*/sample.log`
+ the dry pass below; "int1" = `$OFH/v2maps_int1`):

| alias | goal kind | goal surface (height) | role row | int1 today |
|---|---|---|---|---|
| ontop_large_01 | tabletop | house_single_floor/countertop_fjkase_0 top, live 0.9158 (inventory top.z 0.9157) | candidates.csv:2719 partner; representatives ontop#01 partner representative | **MISSING** (soap_dispenser__zwudcz__s1 has floor maps only; anchors 0.8874 [0.8801, 0.8974] / 0.9275 [0.9175, 0.93055]; MAP_REQUEST tabletop h 0.9158 in i7_floor, i7_floorfix, int1_1) |
| ontop_large_02 | tabletop | house_double_floor_upper/desk_iotfzl_0 top 0.7351 | candidates.csv:2735 / 2765; reps ontop#02 representative | covered (mouse__pljjtu__s1 h0.7351 r / l) |
| ontop_large_03 | tabletop | hotel_suite_large/bed_kddswt_0 top, live 0.5145 (inventory 0.5083) | candidates.csv:2742 / 2770; reps ontop#03 representative | covered (pillow__mltefs__s1__tabletop_level_pinch h0.5145 r / l; the anchor came from the init bed kddswt_1) |
| inside_large_01 | interior (lean goal rest) | bookcase_jaysra_0@0.9472 | reps inside#01 representative | covered (hand-built lean maps, both arms) |
| inside_large_02 | interior (PLACE) | fridge_petcxr_0@0.7064/j_left_door2.3 | reps inside#02 representative | covered (hand-built, `interior_place.py`) |
| inside_large_03 | interior (PLACE) | drop_in_sink_lkklqs_0@basin 0.7751 | reps inside#03 heldout | covered (hand-built) |
| inside_large_03 sinkcounter / sinkbar (template variants) | interior + tabletop | drop_in_sink_awvzkn_0@basin 0.7316; bar_udatjt_0 top 1.0505 (plate luhkiz) | reps inside#03 representative / support | in progress (i7-ft2) |
| touching_large_02 | interior (PLACE, `touching_place` surface) | house_double_floor_lower/hall_tree_upkvgr_0 board 0.5809 (customs `inside_surface_z_m` 0.58: collision hull c13 top 0.581) | reps touching#02 heldout | **MISSING** - no interior spec (MAP_REQUEST interior h 0.5809 "no interior spec ... in assets/interiors/interiors.json", i7diag_floor) |
| touching_large_01 / 03 | none (floor push) | floor | - | n/a |
| under_large_01 / 02 / 03 | none (push-under) | floor under bed_rrcvaq_0 / desk_mdhelw_0 / desk_mdhelw_1 | - | n/a |
| nextto_large_01 / 02 / 03 | none (base push) | floor | - | n/a |

**Facts from the forked session, checked:**

- `height_anchors.py:105` reads only `role == "support"` rows, `model_surfaces` (:240) the surfaces of the target's subtasks - correct.
- "Its height never became an anchor for the target" - not for the reason given. In `candidates.csv` ontop_large_scale lists 149 support
  rows, countertop_fjkase_0 among them, so the rule already puts that top among soap_dispenser zwudcz's surfaces - but at **0.4180**, not
  0.9157: the floor datum takes countertop_kelzer_0's `under_clearance.floor_z` 0.4977 (the cabinet top under the counter, not a floor)
  as a floor level of house_single_floor, and every support top there whose AABB bottom lies above 0.4777 (15 tops: the 5 countertops and
  10 shelves) is measured from it (`$GBH/anchors.json` has 0.4180 / 0.4298 / 0.3898 for the countertops). Changing the datum changes those 15 existing
  surface heights (owner question, not done).
- The campaign's anchors are not the rule's output: `v2maps_int1/anchors.json` holds 22 `measured_live` anchors (the sampler's live
  `top.z - world.floor_z`: `jobs/v2build/build_anchors.py`, `anchors_i7.py`, then one anchor per MAP_REQUEST via `build_int1.py anchor` /
  `stage_plate.py`). They list the INIT supports of the aliases (and the hand-added goal heights 0.7671, 0.8874); the goal tops were
  never listed - the actual blind spot for ontop_large_01.
- `representatives.csv` (the per-alias table) names the instance in `name`, the model id in `model`: the rule reads `model` as the
  instance, so all 270 of its support rows are "missing inventory" and its partner rows were never readable.
- The 10 mm window: 0.9158 is 11.7 mm below 0.9275 and outside its range [0.9175, 0.93055] - uncovered by design (correct).
- The expanding map (map_tilt) fills tilts / yaws inside a map at one height only - correct (`reach_maps` / `map_tilt` have no height axis).

**Dry pass over the role tables** (every ontop_large_scale partner row of both tables, inventory `top.z` with a floor 0, looked up in the
int1 anchors): 35 partner surfaces; covered exactly / approx: the beds rrcvaq / gxfipj / ivdnny / kddswt, desks bhkhxo / iotfzl / mdhelw /
xhjsub / aduafr / pfsjbc, countertops ikwqer / sjxber / kelker / kelzer / zkhelh; uncovered: countertop_fjkase_0 0.9157 (ontop_large_01),
and the heldout / candidate-only partners of no alias (desk_nbpuns 0.5494, desk_uqcmzf 0.7168, the sofas 0.3991-0.7079, beds klcjcy /
siuxgz / tjfzsb 0.448-0.473, bed_zrumze 0.6156, countertop_tpuwys 1.0792 (Rs_int) / 1.3830). Interior goals have no inventory: their
heights come from the registry (the hall tree has none).

**Missing goal maps of the aliases:** (1) ontop_large_01 - soap_dispenser__zwudcz__s1, countertop_fjkase_0 top, h 0.9158, tabletop, r + l;
(2) touching_large_02 - umbrella__ziqzqo__s1, hall_tree_upkvgr_0 board, h 0.5809, interior PLACE (rule drop 0.005, pre approach) - needs
an interior spec first (no inventory / registry measurement). In progress elsewhere: inside_large_03's awvzkn / bar variants (i7-ft2).

### The anchor rule change (912c12d1)

`height_anchors.GOAL_PLANES` names, per IORS large subtask, the plane its goal side reads a map at (ontop "top", inside "interior",
under / nextto / touching None). For a "top" goal the `partner` rows of both role tables (`representatives.csv` by its `name` column)
become surfaces, after the support rows, with the support rows' rules: the inventory `top.z` (`supports/<name>.json`, else
`partners/<name>.json`), the floor datum, the `%.4f` key, the 10 mm window. A goal row on a surface a support row already measured adds
its row and `Surface.goal_subtasks` only (no new anchor, key unchanged); the support surfaces' height / key / status / `subtasks` are
unchanged. `model_surfaces` (one table or several) returns the supports and the goal surfaces of the target's subtasks; `workload` lists
the goal surfaces. Default on (the decision is the rule change; it is an offline tool - nothing at run time reads it);
`goal_surfaces=False` / `--no-goal-surfaces` = the former set. On the repo tables (candidates + representatives): 44 partner rows, 16 new
surfaces, 15 new anchors (93 -> 108), 0 without an inventory top; the 93 former anchors keep their keys and resolve exactly to themselves.
Tests `ispatialgym/tests/test_goal_anchors.py` (3): a partner goal surface becomes an anchor of its target (both schemas, off = former
set); the real tables' former surfaces / anchors / keys are identical; an uncovered height stays uncovered (no union; the int1 live set
+ 0.9158: 0.9158 exact, 0.9050 uncovered, every former live anchor exact).

Not changed (owner question): the floor datum (countertop_kelzer_0's `under_clearance.floor_z` 0.4977 measures 15 house_single_floor
tops - countertop_fjkase_0 0.4180 instead of 0.9157 - so the rule alone does not give ontop_large_01's height; the campaign anchors
are the live heights anyway), and the representatives.csv SUPPORT rows (still read by `model`, i.e. skipped: reading them adds
surfaces beyond this decision).

**Staged root** `$OFH/v2maps_goalmaps` (`$OFH/goalmaps_20261108/stage_anchors.py`): `cp -a v2maps_int1` + anchor **0.9158**
(house_single_floor/countertop_fjkase_0/top, `measured_live` = the MAP_REQUEST h, as every int1 anchor), midpoint rule: 0.9158
[0.9058, 0.92165], 0.9275 [0.9175, 0.93055] -> [0.92165, 0.93055] (no live query in (0.9175, 0.92165] in `$OFH/out`; the inventory has
no top there); the old file `anchors.json.4_before_goalmaps`; log `STAGED_goalmaps.txt`. v2maps_int1 untouched.

**touching_large_02 (the hall tree) - spec registered, maps not built: owner decision.** The existing measure CLI (`interiors measure`,
slurm 18379979, trunk_fcddc495) measures hall_tree_upkvgr_0's board at 0.5809 (1342 board cells, depth 0.799, board setback 0.3047 =
the canopy's static-AABB overhang 0.3147 as the front plane, headroom 1.505, back ray 0.474 -> a back panel, one side wall found
(0.705) the other not, open edges ['+u', '-v'], digest d6947c534b88); registered as `hall_tree_upkvgr_0@0.5809` in
`assets/interiors/interiors.json` (19186e48, the digest test extended; lead request). The live lookup then misses one step later:
probe gm_t02_before (trunk_19186e48, v2maps_goalmaps, slurm 18380485): "the live interior is not hall_tree_upkvgr_0@0.5809: open sides
['+u', '-v'] vs spec ['front']" - `interiors.live_mismatch` reads two open raster edges as an open front AND back (`back_thickness_m`
None), and this board has a back and an open SIDE. It runs before the map lookup, so a map built on this spec would never be read: no
map built. Options: (a) the open-sides rule tells an open side from an open back (a rule every interior lookup shares), (b) register the
board without its back panel (misstates the geometry).

**Validation, ontop_large_01** (N 4 v2only, trunk_912c12d1, alias memory off, svl titanrtx; missing_map 1 -> 0):

| arm | root | att / written / ok | GPU-h (per success) | notes |
|---|---|---|---|---|
| gm_before | v2maps_int1 | 0 / 0 / 0 | 0.11 | setup no_init_surface: MAP_REQUEST tabletop h 0.9158 |
| gm_after | v2maps_goalmaps | 14 / 4 / 3 | 1.85 (0.62) | goal map h0.9158 hit (r / l); rejections target_placement 7, program_check_failed 1, approach_untrackable 1, place_unreachable 1; 1 demo robot_touching_itself |

### Owner decisions 2026-11-08 (5c.1 / 5c.2, `owner_20261108e_goal_framing.md` items 1 and 2)

**1. The floor datum: fixed (ce7b75d4, on by default as the rule's bug fix).** A partner's `under_clearance.floor_z` is a floor level
only when the partner stands on it (within `FLOOR_STAND_TOL_M` 0.05 m of its AABB bottom); countertop_kelzer_0's 0.4977 (its AABB
bottom 0.8514: the raycast stopped on the cabinet inside the counter) is no longer one - the only floor level dropped in the
inventory (the office levels 0.0045 / -0.0032 stay: their partners stand on them). `real_floors=False` = the former rule. Change list
(repo tables, before anything was rebuilt): 15 house_single_floor tops get their true heights - countertop_fjkase_0 0.4180 -> 0.9157,
ikwqer / sjxber 0.4298 -> 0.9275, kelker / kelzer 0.3898 -> 0.8875, shelf_amyper_2 / 7 0.3149 -> 0.8126, amyper_3 / 8 0.7248 ->
1.2225, amyper_4 / 9 1.1346 -> 1.6323, amyper_5 / 10 1.5445 -> 2.0422, shelf_fjozhc_0 0.5657 -> 1.0634, shelf_pfusrd_0 1.3314 ->
1.8291; the 7 customs interior bindings of those supports (0.81 / 0.89 / 0.92 / 0.93 / 1.22) now lie within 3 cm of their true top and
name it (no interior surface left in candidates.csv: `test_grasp_bands` updated); every other surface height unchanged. Anchors 108 ->
102 (dropped 0.3149 0.3898 0.4180 0.4298 0.5657 0.7248 0.81 0.89 0.92 0.93 1.1346 1.22 1.3314 1.5445; new 0.8875 0.9157 0.9275 1.0634
1.2225 1.6323 1.8291 2.0422). Who reads them: no staged or integrated root - none of the 43 `$OFH/v2maps_*` / 11 shiyuc roots' anchors.json
holds a dropped key, and no map anywhere (`$OFH/v2maps_*/maps`, `$GBH/out/*/maps`) is keyed at one; only `$GBH/anchors.json` (the
read-only grasp-bands record, no map at those keys) lists them. So nothing was rebuilt; the campaign anchors are live heights (the
staged 0.9158 agrees with the rule's 0.9157 within 1e-4, served exactly / within the window). Tests (test_goal_anchors.py): kelzer's
cabinet top is not a floor level, a real raised floor still is; exactly the 15 tops change, to their inventory top above floor 0;
every other height unchanged.

**2. The hall tree: "front + side" as its own open-sides case (a083123a, `ISPATIALGYM_INTERIOR_FRONT_SIDE`, default off).**
`interiors.open_sides_case`: one open raster edge = "front" (a back panel); with the switch two ADJACENT open edges (+u / -v) =
"front_side" (the back closed), any OPPOSITE pair = "front_back"; off = any two edges are front + back (today). `live_mismatch` uses
it; kit counter `interior_open_sides.front_side` at the lookup it served. A lookup-side switch: no digest / map key / map changes.
Digest check (switch off and on, identical): fridge 0b539d8a83ab, sink lkklqs bdb7f8c6f92e, kmuacd b58d4e5c790e, jaysra 9d713ac87564,
hall tree d6947c534b88, and the other branches' specs from their registries: awvzkn (i7-ft2) 4571693d44c4, opryhp_0@0.3722 (i7-t03specs)
b8577f909128, otwukr_1@0.3888 41a3fa72d143, otwukr_1@0.7204 71e59f596aa4. Of the measured rasters only the hall tree's is two
adjacent edges (fridge / t03 bookcases one edge, jaysra +u/-u, lkklqs four, awvzkn none): the switch changes no other verdict (kmuacd
is hand-authored, no measured edges). test_interior_registry.py: the digests on / off, the case table, the hall tree served only
with the switch. Maps: umbrella__ziqzqo__s1 interior PLACE maps in hall_tree_upkvgr_0@0.5809 (the variant's rule: pre approach, drop
0.005, no tilts; trunk_a083123a, 18381078 / 18381079), staged in v2maps_goalmaps; touching_large_02 N 4 v2only switch on / off
(chain_t02.sh 18381085). Maps 5130b76f (r) / 9c3a3a02 (l), ~440 s each, check_int1 ERRORS 0.

| arm (touching_large_02, trunk_a083123a, v2maps_goalmaps) | att / written / ok | GPU-h | notes |
|---|---|---|---|
| gm_t02_off (switch off) | 0 / 0 / 0 | 0.10 | setup missing_map: "open sides ['+u', '-v'] vs spec ['front']" |
| gm_t02_on (switch on) | 60 / 0 / 0 | 1.73 | map hit (`interior_open_sides.front_side` 1); 60 target_placement: 59 "F_init (4-12) ∩ F_goal (0)" at the board's front goal cells (x -0.9), 25 also a failed carry leg, 9 B no_transition |

missing_map is gone with the switch; the next blocker is the exact place at stance C with the umbrella's single library grasp (top_2)
at the board's goal yaw - not in this decision.

### Owner decisions 2026-11-08 (5q.1 / 5q.2, `prompt_decisions_20261108c`)

**5q.1 - the front-side switch is the default (1af21ad3, its own commit).** `interiors.front_side_on()` reads
`ISPATIALGYM_INTERIOR_FRONT_SIDE` as on unless it is `0` / `off` / `false` / `no`. Digest check over EVERY spec of the repo registry, the
switch unset / 0 / 1: fridge 0b539d8a83ab, sink lkklqs bdb7f8c6f92e, jaysra 9d713ac87564, kmuacd b58d4e5c790e, hall tree d6947c534b88 -
identical. test_interior_registry.py now requires a recorded digest for every registry spec (a spec added without one fails), checks the
digests unset / 0 / 1 and each spec's own open-sides verdict on / off, and the default on. Snapshot trunk_1af21ad3.

**5q.2 - why F_goal was empty in gm_t02_on (diagnosis from its log and the staged maps, no new code).** `SurfaceTouchingMode.release_feasible`
screens only the F_init yaws whose CARRIED yaw (init yaw + the base-yaw change B -> C) lies within half a screening step (11.25 deg) of
one of the board's axes, and whose footprint then fits the board (90 / 270 deg only: `fit_per_yaw {0: 0, 90: 268, 180: 0, 270: 279}`).

| per (C, yaw) entry, 3392 C tries / 28111 entries | entries | share |
|---|---|---|
| not alignable (carried yaw > 11.25 deg from every axis) | 21201 | 75.4 % |
| aligned ACROSS the board (0 / 180 deg): `footprint_off_board` | 6069 | 21.6 % |
| aligned along the board -> screening IK | 841 | 3.0 % (573 pass; -90 deg 240 / 384 fail, -270 deg 28 / 457) |

Per C: F_goal empty at 2871 of 3392 (85 %): 2608 (77 %) had no yaw to screen at all (the release-yaw constraint), 263 (8 %) lost their
screened yaw in the IK screen (IK + collision together). The camera is not part of F_goal (stance C's frame filter comes first). F_init's
size decides: <= 10 yaws -> 56 of 2402 C's with an F_goal, >= 11 -> 465 of 990. **Map vs exact: no disagreement** - the interior PLACE
maps hold the object yaw (8 base-frame yaws x 10 grasp members): only 90 / 270 (along the board) are feasible (r 96 / 164 cells, l 106 /
149; all other yaws off_support), as the exact rule. The map is yaw-blind only in the stance-C lookup (any feasible (yaw, grasp) admits a
cell); the exact check needs the carried yaw, i.e. a pick at B with the shaft across B's heading, which the one-grasp F_init rarely holds.
The 521 C's with an F_goal then failed later: 288 B -> C carry path (the 0.82 m umbrella collides near (-0.65, 1.35) / (-0.85, 1.20)),
186 B exact IK, 14 the A -> B drive passes the target, ~30 C attached place IK (joint limits), 5 branch swing. Round 1's entry
(roles_all_20261102): 12 top candidates within 6 cm of the shaft centre, 10 reps -> 1 valid (top_2); 6 of 7 top reps `no_placement` in
the validator (the square slab = footprint radius 0.41 + 0.20 m, the base 0.30 m from its edge: the shaft centre 0.91 m ahead).

The IK part was split by a STANCE_BLAME run (gm_t02_blame, N 1, the round-1 root, trunk_1af21ad3 with the front-side DEFAULT and no env switch:
interior map hit, missing_map 0): every blamed release (release yaws -90 / -270, top_2) is "not reached with collisions off", 32-333 mm short
or joint 2 at its limit - reach, not collision.

**5q.2 - ladder round 2 for umbrella__ziqzqo__s1** (as the gym shoe's; record `$OFH/goalmaps_20261108/umbrella_r2`: prep.py, plan.json,
submit_glib.sh, submit_maps.sh, stage.py, chain.sh). trunk_1af21ad3 (candidate switches on by default) + gate_dump, geometry copied, the job
of manifest_settle25 + legacy_key umbrella__ziqzqo__t. L2v = L2 with one per-entry validator change, `slab_half_m` 0.20 -> 0.05 (the slab
still covers the footprint radius + 5 cm; every physics gate unchanged) - an arm added because round 1 lost its top reps to no_placement.

| level | candidates / geometry-ok | trials | physics-valid | losses |
|---|---|---|---|---|
| round 1 (roles_all_20261102) | 15 / 10 reps | 10 | 1 (top_2) | 6 no_placement, 3 finger_penetration |
| L2 (24000 x 24, 20 reps, cone 45) | 40 / 33 | 20 | 2 (top_2, top_23) | 17 no_placement, 1 palm_contact |
| L3 (48000 x 48, 30 reps, cone 60) | 56 / 41 | 25 | 1 (top_25) | 20 no_placement, 3 palm_contact, 1 grasp_off_target |
| L2v (L2 + slab_half_m 0.05) | 40 / 33 | 6 | 6 (max_valid: top_0 / 2 / 9 / 10 / 14 / 19) | none |

The L2v grasps tilt 14-33 deg toward +-x (along the shaft) and +-y (across). Maps (trunk_1af21ad3, floor h 0 and interior PLACE in
hall_tree_upkvgr_0@0.5809 with the round-1 rule): floor cells per yaw 31-39 (round 1) -> 47-57 (L2) -> 61-67 (L2v) of 72-73; interior
cells (any yaw) r / l 174 / 166 -> 203 / 202 (L2) -> 234 / 227 (L2v), still only at yaws 90 / 270. Staged roots (copies of
v2maps_goalmaps, round-1 entry and maps in `_superseded_ladder_r1`, check_int1 ERRORS 0): **`$OFH/v2maps_goalmaps_umb2`** (L2v, the
primary) and `$OFH/v2maps_goalmaps_umb2s` (L2, the standard validator). L2v's validator change is per entry; the owner confirms it.

**touching_large_02 on the new roots** (N 4 v2only, trunk_1af21ad3, memory off: gm_t02_umb2 18383997 on umb2, gm_t02_umb2s 18383999 on
umb2s; the final numbers are appended to DECISIONS by `umbrella_r2/final.sh`). Interim: the F_goal blocker is gone - F_goal > 0 at 294 / 308
C's (umb2) and 432 / 474 (umb2s), against 521 / 3392 in gm_t02_on. 0 written so far: the C tries end at the B -> C carry path (145 of 227 /
225 of 332) and the B exact IK (89 / 91). The carry blame (gm_t02_umb2_pathblame, `ISPATIALGYM_PATH_BLAME=1`): of 30 blamed collisions 18
are the carried umbrella against standing_mirror_wwkuuf_0 (removing the mirror alone frees them), 7 walls_oqrbxo_0, 5 walls_owxeej_0, around
(-0.87, 1.15..1.26) beside the hall tree - the next blocker, outside 5q.2.

**Lead decisions (owner away), 2026-11-08.** (1) L2v's per-entry `slab_half_m` 0.05 is accepted provisionally for umbrella__ziqzqo__s1 only;
umb2 stays the primary root unless umb2s does as well. (2) The carry past the mirror is this branch's.

**The carry past standing_mirror_wwkuuf_0 (diagnosis).** Per C the planner tries lifts 0.12 / 0.25 m (0.25: no IK from B in ~100 tries),
per lift the CARRY_POSES that pass B's camera frame and IK (the hand 0.30-0.50 m to the holding arm's side), then the lift configuration;
per option the travel trunks (0.5, 0) and (2.0, 0) with the local then the eroded global path, then the drive at B's trunk. PATH_BLAME
(34 blamed): the global path just north of B, base at (-0.82..-0.94, 1.10..1.26) heading 1.0-1.2 rad and (-0.67..-0.70, 1.50..1.54);
the mirror 25 (the umbrella 11, the holding gripper 13, 1 right), walls_oqrbxo_0 8 (the umbrella), walls_owxeej_0 2 (init -> B leg). The
release needs the shaft along the board, i.e. ACROSS the robot's heading at C, and the carried yaw is fixed relative to the base: the 0.82
m umbrella is carried across the heading, and with the hand 0.3-0.5 m to the left it reaches ~0.85 m left of the base.

| hand lateral at the carry pose (umb2 + umb2s + pathblame) | carries ok | failed |
|---|---|---|
| >= 0.3 m (CARRY_POSES beside the body) | 0 | ~730 (path_collision) |
| 0.2 m | 0 | 6 |
| lift configuration | 10 | 126 |
| within 0.1 m of the centre line | 8 | 7 |

**The fix (6ac7931a, `ISPATIALGYM_CARRY_CENTRED`, default off; sampler-side, no oracle or gate change).** `carry_poses_at_b` first tries
`CARRY_CENTRED_POSES` (0.45-0.55 m ahead, raise 0.05-0.20 m) with the hand's lateral chosen so that the carried object's hull centre lies
on the base's centre line (`centred_hand_lateral`); the existing checks judge them (B's camera frame, exact IK attached, the in-hand line,
the carried path). Kit counters `carry_pose.centred_offered` / `centred_used`; test_carry_centred.py. A/B on touching_large_02 N 4 v2only,
root v2maps_goalmaps_umb2, trunk_6ac7931a, svl titanrtx: gm_t02_cc_on (18384743); the OFF arm is gm_t02_umb2 (same root, both switches off).

**Second switch (0c24c2c6, `ISPATIALGYM_CARRY_SIDEWAYS`, default off).** gm_t02_cc_on's first C's offered the centred poses (hand lateral
-0.01) and still collided at the same spot, so centring alone does not clear it. With the switch, a carried B -> C path that collides in
every variant of today (straight arrival / plain, local then global) is planned again with the holonomic base driving sideways:
`NAV.plan_path(heading_offset=+-pi/2)` (`NAV.fill_yaws`: each intermediate waypoint's yaw = its segment's heading + the offset, the end poses
keep their stances' yaws), the offset nearer C's yaw first (`sideways_offsets`); `record_path` passes it through (default 0 = unchanged).
The object held across the heading then lies along the way. The same swept check judges it (object attached, carry arms, yaw-lag
envelope); the record is the recorded style, whose waypoint yaws the oracle tracks blended per segment as `stride_poses` models (no oracle
change). Kit `carry_path.sideways_tried` / `sideways_used`; tests in test_carry_centred.py. Arm gm_t02_side_on (18384880, trunk_0c24c2c6).
**Sideways result: no help.** gm_t02_side_on (sideways only) used it 0 times in 492 tries; the PATH_BLAME run with centred + sideways
(gm_t02_side_blame) shows why: the collisions sit at the path start, where the yaw blends from B's heading toward the sideways one (base
(-0.9, 1.2), yaw 1.0-1.3): the umbrella against walls_oqrbxo_0 10, the mirror 3; the base itself against the mirror 2. B stands in the tight
spot between the wall, hall_tree_daxzaw_0 and the mirror, and the turn needs room there. The switch stays off and is not proposed. Both
runs were cancelled early; I also cancelled the first centred arm (gm_t02_cc_on) too early - it had 12 carries in 3 attempts, 9 at centred
poses, against 5 carries in gm_t02_umb2's 12 attempts - so it was re-run as gm_t02_cc_on2.

**Third switch (c2071b3e, `ISPATIALGYM_CARRY_WIDE_PATH`, default off).** The centred carries' remaining collisions sit 20-30 % along the
path at (-0.6..-0.8, 1.4..1.5), heading north past the mirror: the shortest global path hugs it. With the switch, after every variant of
today collides the global path is planned again on the eroded map eroded 0.15 / 0.30 m more (`NAV.plan_path(global_extra_erosion_m=)`,
`NAV.erode_more`: square kernel at the map's resolution; the ends snapped within 20 + the extra cells). The same swept check judges it. Kit
`carry_path.wide_tried` / `wide_used`. The extra variants never replace the plain variant's error (`carried_path` restructured; off = unchanged).

| arm (root umb2, N 4 v2only, svl titanrtx) | tag / job | snapshot |
|---|---|---|
| OFF | gm_t02_umb2 / 18383997 | trunk_1af21ad3 |
| CENTRED | gm_t02_cc_on2 / 18384993 | trunk_0c24c2c6 |
| CENTRED + WIDE | gm_t02_wide_on / 18385113 | trunk_c2071b3e |

The table is appended to DECISIONS by `umbrella_r2/final_cc.sh` (job 18385122).

Interim (runs still going): attempts / successful B -> C carries / carry legs failed per attempt - OFF 23 / 5 / 14.6, CENTRED 3 / 1 / 14.3
(the cancelled first centred run 3 / 12 / 12.7), CENTRED + WIDE 4 / 24 / 2.5 (21 of the 24 on a WIDE path, mostly at 0.30 m; 20 at a centred
pose). 0 written in every arm so far: after a carry the C tries end at B's exact pick IK, `C: pre-place branch off the tracked carry` and C's
attached place IK - the next blockers, outside the carry. umb2 vs umb2s: 23 att / 5 carries vs 30 / 15, both 0 written - umb2 stays primary
until the final numbers.

### i7-ft2 (2026-11-08): owner decisions 7.1 (FingerTouch hand hulls) and 7.2 (the bar / countertop plate-in-sink templates)

**7.1, the decision:** apply the two-step to the FingerTouch path. Before this, the geometry rule judged the palm, wrist camera + mount
and wrist (the hand's rigid links) by their padded spheres only. That is more lenient than the spec's P1 §3, which asks for hulls AND
spheres.

**7.1, the implementation:** `ISPATIALGYM_FINGER_TOUCH_HAND_HULLS` (6cc62a73; default on since e9efae38; `0` restores the former stage,
`shadow` only counts). It lives in `IKChecker._finger_touch_geometry_ok`'s non-finger stage, against the touched support's geometry
(its registered plane, else its convex pieces):
- A narrow sphere failure is judged by i7-plateqtf2's reusable `contact_policy.hull_fallback`, with the same 0.5 mm threshold (side_9's).
  Narrow means only hand links touch, each sphere less than the 2 mm buffer deep, and only into the touched support.
- A sphere pass also needs the hulls clear by `hand_support_margin_m` (`contact_policy.hull_clearance_min`).
- Counters: `finger_touch.hand_hulls.*` (`fallback_*`, `spec_*`, `verdict_to_pass` / `verdict_to_fail`, `decision_to_*`).

**7.1, the A/B:** `ft2hh_sh` (shadow: today's verdicts plus the count) vs `ft2hh_on`, run on the same indices with a cold cache per job
and the alias memory off.

| alias (grasps) | written / attempts | demos ok | GPU-h per success (sh / on) | fallback judged -> passed | spec hull rejected |
|---|---|---|---|---|---|
| under_small_03 (v2only) | 4 / 22 both | 4 / 4 both | 0.217 / 0.218 | 30 -> 6 (hull 0.53-1.44 mm) | 0 of 327 / 333 |
| inside_large_03 (legacy A/B arm) | 4 / 9 both | 4 / 4 both | 0.080 / 0.077 | 38 -> 13 / 12 | 0 of 131 / 143 |
| plate-in-sink lkklqs_0-breakfast_table (legacy) | 4 / 4 both | 4 / 4 both | 0.051 / 0.051 | 4 -> 2 | 0 of 51 |

- The refused narrow candidates were non_hand_link 23, deeper_than_buffer 27-28 and hull_clearance 1.
- Groups flipped to pass: 6, 1 and 0. None flipped to fail.
- Attempt outcomes were identical in 33 of 35 attempts. In the other two (inside_large_03 300702), both arms rejected the attempt, for
  different reasons after the trajectories diverged.
- touching_03 is still blocked.

**7.2, the sink spec and maps:**
- `drop_in_sink_awvzkn_0@basin` was measured (`interiors.measure`, house_single_floor) and committed to the code registry (3030ddfd).
  It is an open-top basin at 0.7316 m: width 0.2466, depth 0.4063, rim 0.16, digest 4571693d44c4.
- v2only template probes asked for the interior map of that sink and for a plate luhkiz tabletop map at bar_udatjt_0 h 1.0505. The
  countertop pick reads the existing 0.8874 maps.
- Built in `$OFH/ft2_20261108`:
  - the bar maps (right / left), about 600-1900 cells per trunk and d_e;
  - the interior PLACE maps, with the lkklqs sink's rule (pre approach, drop 0.02, tilts 30 / 45).
- The first sink build was EMPTY. The plate's level footprint (0.241 m) must lie between the 0.2466 m walls, a 2.8 mm margin. The
  default laterals put the compartment centre 6.6 mm off every 0.04 lattice row, so every problem was off_support.
- Rebuilt with the laterals shifted by the spec's centre_y (-0.0866 .. 0.0734; the builder's resolution, not an identity-equal field):
  36,985 feasible cells over both arms. Staged as `$OFH/v2maps_ft2b` (a copy of v2maps_int1 plus these maps).

**7.2, the templates:** at eval:0, v2only on v2maps_ft2b, both variants now pass setup (stance C at the sink). The first attempt then
spent its 3000 s budget at "F draw at C (exact IK at B / C)", and the job hit its 4500 s limit with no instance:
- bar: 70 of 108 B tries failed "support policy on the approach tail (finger_penetration on bar_udatjt_0)";
- countertop: 65 of 95 on countertop_kelzer_0.

This is the IORS-large pick's missing pre-shape (owner item 6.1, assigned to i7-interior2), the same symptom as i7_sink2 on lkklqs. No
template exists yet. Rerun once 6.1 lands: `$OFH/ft2_20261108/chain72b.sh`'s template step on v2maps_ft2b.

**7.2, after the merges** (eval:0, v2only, v2maps_ft2b; neither variant has a template yet, so no N 4):
- Round c, after merging i7-interior2 eec4ca60 (8ebca4dd): with the pick at the pre-shape, finger_penetration went to 0. The pick then
  failed on the wrist camera (bar 28 of 35 tries) and on joint limits (countertop 7 of 10).
- Round d, after merging i7-camhull 9e697ac1 (1640bc16), with `ISPATIALGYM_CAMERA_HULL_FALLBACK=1`: the camera fallback passed 72 of 74
  (bar) and 29 of 33 (countertop), and failed pick tries fell from 35 to 3 (bar) and 10 to 11 (countertop, 9 of them joint limits).
- Both runs still spent the 3000 s first attempt and wrote nothing. The block is now stance C, the attached place into the 0.2466 m basin:
  - C tries exhausted: 46 (bar) / 37 (countertop);
  - right-arm joint limits: 67 / 67;
  - approach collisions with drop_in_sink_awvzkn_0: 85 / 69;
  - exact IK failures: 22 / 31.
- The 0.241 m plate leaves 2.8 mm on each side. The shoulder (joint 2) reaches its limit with the base at lateral 0. i7-interior2's
  `ISPATIALGYM_INSIDE_ARM_SIDE`, or a stance choice off lateral 0, could reduce the joint-limit share but not the collisions.
- Not fixed: the lead decides. Record: DECISIONS "i7-ft2" round c / d and the countertop diagnosis.

#### Owner decisions 6.1-6.3 on the interior goals (branch i7-interior2, 2026-11-08)

- **6.1 sink pick (inside_large_03).** The oracle pre-shapes to pinch width / 2 + the margin before the descent; since 11-06 the support
  policy measures the fingers there (`tabletop.policy_opening_of`) - but only in `tabletop.grasp_at_pose`. IORS large's picks judged the
  fingers fully open (0.05 m): the plate luhkiz lip pinches (mode "top", widths 4-8 mm: pre-shape ~0.012-0.014 m per finger) failed
  `support policy on the approach tail` (finger_penetration on bar_egwapq_0) at every one of 249 C tries in i7_sink2. a7f2acb0 wraps the
  main pick at B, the segment-fallback regrasp, the overhang grasp and their twins (`pick_twin`) in `_policy_opening` (the existing switch
  `ISPATIALGYM_SUPPORT_POLICY_OPENING=q` turns it off; counters `pick_preshape.<B | segment_fallback | overhang>[.q]`). Plus the overhang
  FALLBACK of this pick (IORSSmall's, `v2/overhang_fallback.py`): without an alias `pick_overhang_push`, a direct pick failing for a
  support-policy class (`overhang_fallback.reason_class`) takes IORSLarge's own `_overhang_pick` with the overhang defaults in fallback
  mode; any grasp mode (the plate's grasps are "top": a side-only filter would never trigger); switch `ISPATIALGYM_LARGE_OVERHANG_FALLBACK`,
  default ON since c7fcb37a (owner 6.1); counters `large_overhang_fallback.trigger.<class>` / `.mode.<mode>` / `.found` / `.failed`.
- **6.2 stance-C cap with a map.** `ISPATIALGYM_INSIDE_MAP_CAP` (default ON since c7fcb37a): the survivors of a map-view band are capped per
  base pose by reach margin as without a band (`generation.inside_map_trunks`, default `NO_BAND_TRUNKS` 3; every base pose keeps its best;
  `placement_stats.map_cap` before / after / bases, counters `inside_map_cap.kept` / `.dropped`). Legacy bands are not capped.
- **6.3 fridge carry / place (inside_large_02).** i7_fridge4 finished 16 att / 4 written / 2 ok (0.96 GPU-h per success). C-try rejections
  (i7_fridge4 / legacy fix2): B -> C path / transition 158 / 529, the place IK's joint2 near its limit 58 / 189 - the legacy mix. The path
  collision is the BASE turning in place beside the init table (fix1 PATH_BLAME: base_link + wheels vs breakfast_table_xftrki_0, ~0.08 m
  from B), not the carried jar. The demo failures (2 / 2): navigate[4], the idle right gripper (ready) against fridge_petcxr_0. Every earlier
  successful fridge demo (fix1 / fix2, 5 / 5) carried STRAIGHTENED (straighten_then_navigate, travel tilt 0; right arm, C so 0.55, trunk
  (0.95, -0.3), goal 0.871 m ahead); i7_fridge4's demos driven at the leaning grasp trunk (tilt -0.16 / -0.19) 1 / 3 ok, the straightened one 1 / 1.
  No demo failed at the place (release 0.03 + lift, finger clearance 0.080 m, no tilt). Switch `ISPATIALGYM_INSIDE_CARRY_STRAIGHT` (default off):
  the inside carry only straightened (counters `inside_carry_straight.rejected` / `.straight`); A/B below.
- **Commits** a7f2acb0 (the code, switches off), c7fcb37a (defaults on: 6.1 overhang fallback, 6.2 cap); snapshot trunk_c7fcb37a; suite =
  base (cf1f5d2d on this host: 23 failed + 3 errors, the same set).
- **Validation** (`--grasps v2only` N 4, `$OFH/v2maps_int1`):

| run | alias | switch | att | written | ok | GPU-h / success | s / attempt |
|---|---|---|---|---|---|---|---|
| i7_sink2 (before) | inside_large_03 | - | 3 | 0 | - | - (3.0 GPU-h) | 3000 (budget) |
| i7i2_sink | inside_large_03 | defaults | 6 | 4 | 4 | 0.32 | 688 |
| i7_fridge4 (before, uncapped) | inside_large_02 | - | 16 | 4 | 2 | 0.96 | 406 |
| i7i2_fridgeA + A2 | inside_large_02 | defaults | 27 + 13 | 4 + 4 | 3 + 3 | 0.62 / 0.31 | 229 / 229 |
| i7i2_fridgeB + B2 | inside_large_02 | + CARRY_STRAIGHT | 19 + 6 | 4 + 4 | 4 + 4 | 0.34 / 0.135 | 232 / 239 |

  Counters: sink `pick_preshape.B` 163 (fingers 0.0141 m, pre-shaped), the overhang fallback 0 triggers on the sink (3 on the fridge,
  non_finger_link side grasps, 0 found); map cap fridge 17 -> 10, sink 26 -> 9 survivors, base poses 4 -> 4 / 3 -> 3. Where 6.3 acted: the
  leaning grasp-trunk drives 1 / 4 ok over the v2 runs (each failure navigate[4], the idle right gripper on the fridge), straight carries
  15 / 16 (one robot_wobble); B rejected the leaning drive (`inside_carry_straight.rejected` 224 / 47) and wrote straight carries only.
  Default: `ISPATIALGYM_INSIDE_CARRY_STRAIGHT` ON since b165883c (owner 2026-11-08b item 1; "0" off). It is read only by
  `IORSLarge.record_carry_leg` with relation "inside" - the 15 inside_large_scale variants (fridge jar, sink plate, bookcase hardback;
  counted inside_large_01 / 02 / 03); no other alias or family. Validation on trunk_b165883c: i7i2_fridge_def 19 att / 4 written /
  3 of 4 ok (0.35 GPU-h per success), every carry straightened; the failure was the pick (right_arm_link6 vs the breakfast table).
- **Owner item 2 (2026-11-08d), the fridge's place joint 2.** 268 of 349 C place-IK diagnostics (i7_fridge4 + i7i2_fridgeA/A2/B/B2) were
  arm joint 2 at its limit (right 201 / left 67), median 3 mrad from the limit WITH COLLISIONS OFF: shoulder adduction (URDF right upper
  +0.1745 rad) - the goal ~0.025 m from the base's centre line (stance C's single lateral -0.035, the nearest goal cell); the place map's
  IK has no 50 mrad joint margin. Legacy successes (5 / 5): right arm, so 0.55, trunk (0.95, -0.3), goal 0.871 ahead / -0.025, no tilt.
  f288f376 `ISPATIALGYM_INSIDE_ARM_SIDE` (default off): the side's other lateral and the goal cell 0.17 m toward the working arm's
  shoulder (counters `inside_arm_side.*`). A/B: A i7i2_j2A 31 / 4 / 3 ok (0.45 GPU-h per success, joint-2 rejections 5.7 per attempt), B
  i7i2_j2B 13 / 4 / 4 ok (0.31, 5.0 per attempt); every B demo's goal 0.117 m toward its arm.
- **Owner item 3, the sink pick.** i7i2_sink: 41 B-failure lines, 123 reasons - the wrist camera within 5 mm of the world 63, exact IK
  34, joint limit 25. Opt-in `ISPATIALGYM_CAMERA_BLAME` (f288f376 / 19c00c0f, `orient.camera_blame`, no verdict change): 73 / 73 blamed
  cases are the camera vs bar_egwapq_0 (the init support; nothing else frees it), clear at 3 mm (66) / 4 (6) / 2 (1): the buffered spheres'
  narrow gap, not a collision. The sink is not blocked (i7i2_sink + i7i2_sinkdiag2: 11 att, 8 written, 8 / 8 ok); the fix (the hull fallback
  extended to the wrist camera at the policed support) is a shared gate - asked, not implemented.
- **Owner 5t (2026-11-08c), the arm side for "inside" only.** SurfaceTouchingMode (touching_large_02, the umbrella on the hall tree's board,
  relation "touching") subclasses InsideMode and so inherited stance C's two switches. 4f973d68: `ISPATIALGYM_INSIDE_ARM_SIDE` and the
  stance-C map cap (`ISPATIALGYM_INSIDE_MAP_CAP`) act only with relation "inside" (unit test: the surface place's stance C with both on equals
  both off, a map-view band included; with relation "inside" they act). 440d3c89: `ISPATIALGYM_INSIDE_ARM_SIDE` DEFAULT ON ("0" off).
  Readers now: the 15 inside_large_scale variants only (counted inside_large_01 - its `inside_goal_lateral_m` 0.3 keeps its aim, the
  other-end lateral is added -, 02, 03). Confirmation i7i2_armdef (trunk_440d3c89, v2maps_int1, N 4 v2only): 12 att / 4 written / 3 of 4 ok,
  0.28 GPU-h per success (j2B 0.31, j2A 0.45), `inside_arm_side.aimed` 12; stance-C place-IK joint-2 failures 1.7 per attempt (j2A 2.9,
  j2B 1.3; every joint-2 IK line 7.3 / 8.1 / 6.6). touching_large_02 is tested with both switches once the umbrella has grasps (5q.2).
- **Owner 5x.2, the place map's joint margin.** 4b3ca4f1: the place rule's opt-in field `joint_margin_rad` (in the rule / key only when set):
  a pre-place or release IK solution counts only with every working-arm joint at least that far from its limits (0.05 rad = the exact
  check's `JOINT_LIMIT_MARGIN`); a margin map serves the same lookups (the field is not compared); `meta.joint_margin` counts the dropped
  seeds / problems. Fridge maps rebuilt with 0.05 (record `$OFH/interior2_jm_20261108`), staged in `v2maps_int2_jm`; A/B i7i2_jmA
  (v2maps_int2) vs i7i2_jmB (margin maps), same code / backend, arm side on (B's base index advanced to 300062: not A's indices).
  Maps: feasible entries right 45563 -> 33267 (-27 %), left 10482 -> 6293 (-40 %). A 10 att / 4 written / 3 ok, 0.21 GPU-h per success,
  sample wall 2084 s; B 11 / 4 / 4 ok, 0.14 GPU-h per success, 1818 s (-13 %). Stance-C place-IK joint-2 failures per attempt A 0.8, B 1.2
  (every joint-2 IK line 5.1 / 5.5): the margin did NOT remove them (B's still 0-3 mrad from the limit). Hypothesis (unverified): the map
  folds 5 laterals and every trunk / grasp / posture into one "feasible" status; the exact place is one lateral and one grasp. Not folded,
  no default proposed.

#### The wrist-camera hull fallback (branch i7-camhull, 2026-11-08; owner_20261108g item 1)

The owner approved this shared-gate change behind a switch. The default stays the owner's decision after the A/B.

- **Rule (5e4fc9c4, `ISPATIALGYM_CAMERA_HULL_FALLBACK` = `0` default / `shadow` / `1`).** `orient.camera_clear` stays the screen: the
  camera's padded spheres must keep 5 mm from the world at a grasp configuration. A NARROW failure is judged by the camera's exact hull:
  - narrow means the spheres clear at 5 mm less the model's 2 mm buffer, and clear at 5 mm with the policed supports out of the world;
  - the hull (role `wrist_camera`) must then clear the policed supports' registered planes by 0.5 mm.

  The check is i7-plateqtf2's `contact_policy.hull_fallback`, reached through `IKChecker._hull_fallback`. That method gains margin / roles /
  counter-prefix / mode parameters; its defaults are unchanged. The policed supports are the ones the support policy polices at that
  exact-IK result, recorded by `exact` only when the switch is on or shadow. A configuration with no policed support, or a floor, keeps
  today's verdict.
- **Not double-applied.** The support policy's sphere stage (`ISPATIALGYM_SUPPORT_HULL_FALLBACK`) and FingerTouch (i7-ft2
  `ISPATIALGYM_FINGER_TOUCH_HAND_HULLS`) are untouched; this is the separate 5 mm camera rule. The support-top screen
  (`tabletop.wrist_camera_clearance` < 5 mm) is counted only (`camera_policy.top_screen.*`, `tabletop.camera_top_clear`); its verdict never
  changes.
- **Identity, counters, tests.** Under `1` the cache scope gains `camera_hull_fallback`, but not the exact-IK key. Counters are
  `camera_policy.hull_fallback_{checked,passed,shadow_pass,failed.<why>,error}`, with log lines `CAMERA_HULL_FALLBACK` / `CAMERA_TOP_SCREEN`.
  `test_camera_hull_fallback.py` has 13 tests. The suite matches the base eec4ca60: the same 23 failed + 3 errors from the env.
- **Count (record `$OFH/camhull_20261108`).**
  - Logs: every v2 log, including i7-ft2's sink templates. Of 103 aliases, the camera rule refused anything in only 11; 92 never hit it.
  - Shadow runs `ch_sh`: N 4, judged -> would pass, with the hull clearance:

    | alias | judged -> would pass | hull clearance |
    |---|---|---|
    | inside_large_03 | 27 -> 27 | 3.55-5.40 mm |
    | under_small_03 | 34 -> 34 | 4.76-6.11 mm |
    | under_small_01 | 2 -> 2 | 4.01 mm |
    | nextto_02 | 0 (its 59 are the support-top screen, all deeper than the buffer) | - |
    | touching_04, touching_01 | 0 | - |

- **A/B** (v2only N 4, the same indices 335000-3, svl):

  | alias | arm | att / written / ok | GPU-h (per success) | camera judged |
  |---|---|---|---|---|
  | inside_large_03 | shadow | 4 / 4 / 4 | 0.64 (0.16) | 27 (exact IK 109 per attempt) |
  | inside_large_03 | on | 4 / 4 / 4 | 0.54 (0.135) | 5 passed (exact IK 48 per attempt) |
  | under_small_03 | shadow | 27 / 3 / 2 | 1.14 (0.57) | 34 |
  | under_small_03 | on | 29 / 4 / 4 | 1.07 (0.27) | 12 passed |

  The arm with the switch on had 0 refusals and 0 demo failures. The success gain is within N 4 noise. The counters show the rule acts only
  on narrow camera-vs-support gaps whose hulls clear by 3.5 mm or more.
- **Default ON (548fbbfb).** The lead decided this on 2026-11-08 (#5, with the owner away and the decision delegated); the commit is its
  own. `0` restores the former rule. Under `1` the cache identity stays, so verdicts cached under the old rule are not read as the new
  rule's.
- **Support-top screen (178c9df6, `ISPATIALGYM_CAMERA_TOP_HULL_FALLBACK`, separate, default OFF; lead decision #6).** The same two-step
  for `tabletop.wrist_camera_clearance` < 5 mm: the modelled spheres must still be at least 3 mm above the top, and the camera hull at
  least 0.5 mm above it. It applies in `pick_twin` and in `camera_top_clear` (attachment / iors_small / disembedding). Counters are
  `camera_policy.top_screen.*`. Count: under_small_03 24 of 41 would pass (hull 2.33-2.98 mm). nextto_02's 59 are all deeper than the
  buffer (1107 times +2 mm in the logs); touching_04 / touching_01 have a few narrow ones in old logs.
- **Support-top A/B** (world rule on in both arms, N 4, indices 335000-3):

  | alias | shadow | on | GPU-h shadow / on |
  |---|---|---|---|
  | under_small_03 | 29 / 4 / 4 | 29 / 4 / 4 | 1.08 / 1.10 |
  | touching_04 | 48 / 4 / 0 | 48 / 4 / 1 | 1.37 / 1.39 |

  - under_small_03: the rule passed 12 refusals, and every one was then refused by the world camera rule (deeper than the buffer against
    the same table). Nothing downstream changed.
  - touching_04: 0 support-top refusals; its demo failures are oracle ones.
  - Recommendation: keep it off. The default is the owner's.

### i7-tray (2026-11-08, owner 1.1): disembedding_01 / _10 diagnosed; the layout pad check; gym shoe round 2

Owner 1.1 (verbatim intent): diagnose the tray failure first - the library validates each grasp on an open slab with no tray walls, so the
new grasps the defaults admit may be ones the walls block; the gym shoe switches-on rebuild (0 valid) runs its ladder round 2.

**The walls are not the cause.** Hand clearance (exported PhysX hand hulls; conservative wall boxes = the ring between uekqey's bbox and its
inner floor, floor -> rim top +15 mm) over every executed demo (i7_split / i7_split0 old builds, i7t03_cnt4 / int1_1 new builds), min over
pre-grasp -> grasp at the pre-shape opening: fingers >= 67 mm, palm >= 54 mm, wrist / camera >= 57 mm - new and old alike. The sampler's
exact IK has kept the tray walls as obstacles since i7-split (source tray: 10 protrusion pieces), and `support_split.protrusion_hits` = 0 on
every attempt of both aliases in all four runs. No failed demo touched a tray.

**disembedding_10 (tablefork pqrtkc): in-hand rotation at the close.** The rebuilt entry (switches) draws top_5 / top_7; in all 8 executed
demos (int1_1, i7t03_cnt4) the fork turns 12.1-14.4 deg in the hand during the close and is placed 12.2-14.7 deg off (tolerance 0.1 rad =
5.7 deg) -> symbolic_only. The SLAB validator measured the same rotation (`achieved.off_target_rad`: top_5 0.212, top_7 0.233, top_1 0.216,
top_4 0.140; top_2 0.005, top_9 0.015) and admits it: `on_target_rot_rad` = 0.2618 (15 deg). The old entry's rows (top_9 / 10 / 11:
0.002-0.029 rad) turned 0.1-1.7 deg in the tray. Not the tray, a validator gate looser than the task's pose tolerance - owner decision
(asked through the lead): over v2maps_int1, 10 of 280 measured valid rows (7 entries) exceed 0.1 rad.

**disembedding_01 (jigsaw isgefa, the embedded figure): the fingers stand on the figure's other parts.** The pick ignores the fixed-layout
parts for every pose (A2b `pick_world`) and the library validates the piece alone, so a row whose pre-shaped fingers descend onto the frame
(isgefb, 0.3 mm from the target) passes the draw and the oracle's approach stops on it (`approach_support_stop` on isgefb + isgefd, 13-22 mm
short: int1_1 3 of 4). Finger clearance to the parts along the approach at the pre-shape (16 mm per finger), per row (the layout is rigid):
rebuilt entry top_5 / top_10 -6.0 mm (frame + isgefd), top_1 / top_4 -6.0 mm (isgefd only, a 0.5 g inner part the pads push aside),
top_8 +1.3 / top_12 +2.3 mm (clear); the old rows top_2 / top_3 (the 4/4 builds) +3.6 / +0.5 mm. Twins equal their rows. Secondary: the
top_4~r180 / top_5~r180 releases turn the piece 10-20 deg (old top_3~r180: 1-7 deg, one of 8 failed).

**The layout pad check** (`ISPATIALGYM_LAYOUT_PAD_CHECK` = `all` (1) | `heavy`, default OFF; 2a4e64d7, 147070fe, 0ca22d83): the library's
finger rule (`HandChecker` hulls at the oracle's pre-shape opening on the pre-grasp -> grasp line, at most `finger_support_max_penetration_m`
= 2 mm into an obstacle) applied to the layout parts' convex collision pieces; rows that stand on a part (`all`), or on a part at least as
heavy as the target (`heavy`: the frame only), leave the pick (layout screen, `phase3_target`, segment fallback). Counters
`layout_pad.rows / .stand_on[.<model>] / .stand_on_heavy / .dropped / .unavailable`, measured with the switch off. Jigsaw: `all` keeps
top_8 / top_12, `heavy` keeps top_1 / top_4 / top_8 / top_12. Tests: `test_layout_pad_check.py`; suite = base.

A/B disembedding_01 (v2only, v2maps_int1, 2 x N 4 per arm, same base index):

| arm | att / written / ok | GPU-h (per success) | stand-on events | rows drawn |
|---|---|---|---|---|
| off (i7tray_off) | 41 / 7 / 2 | 1.58 (0.79) | 5 | top_10~r180 x2 (failed_grasp), top_5~r180, top_4 x3, top_8~r180 |
| all (i7tray_on) | 96 / 4 / 3 | 1.34 (0.45) | 0 | top_12 x4 (1 arm_stutter) |
| heavy (i7tray_heavy) | 47 / 7 / 1 | 1.23 (1.23) | 0 | top_4 x7 (5 symbolic_only) |

Second finding from the A/B: top_4 (both twins) RELEASES push the piece 23-29 mm and turn it 6.7-11.5 deg (6 of 8 top_4 places fail); top_8 /
top_12 places land 0.7-3.3 deg / 1.5-4 mm off. `all` pays in reach (most attempts `grasp_unreachable`, reach alone). No default flipped:
owner questions - flip `all` for fixed layouts (narrows A2b), or rebuild the jigsaw entry with the figure in the validator; and for the fork
(a) `on_target_rot_rad` 0.2618 -> 0.1 rad (10 of 280 measured valid rows in 7 entries flip), (b) a sampler filter by the recorded
`off_target_rad` where a pose tolerance applies, or (c) an oracle place corrected by the measured in-hand rotation.

**Gym shoe kmcbym (owner 1.1 part 2): round 2 is already the integration entry.** Switches-on round 1 = 0 valid (i7-switches); round 2 (L2
retry levels, switches on) was rebuilt with the pad sweep at the integration (count 24/18/18) = 6 valid (top_1/2/5/6/7/8), maps 0.3471 r + l,
`legacy_key` gym_shoe__kmcbym__t, staged in v2maps_int1. Its consumer nextto_04: int1_1 18 att / 4 / 4 (top_6, top_7); re-run on trunk_fcddc495
(i7tray_gs) 22 / 4 / 3 (1 robot_touching_itself, left arm link3 / link5), 0.27 GPU-h.

### i7-tray round 2 (owner decisions 2026-11-08 items 3 and 4)

**Item 3, the in-hand turn filter** (`ISPATIALGYM_GRASP_TURN_FILTER`, default OFF; 609eb39f). `tabletop.library_grasps` (disembedding,
attachment, IORS small / large) skips a v2 row whose recorded turn at the close (the slab validator's `achieved.off_target_rad`,
`validation.json`) exceeds the orientation tolerance of the grasped target in the task using it: the isg-v2 profile, `small` 0.10 /
`large` 0.15 rad, the customs `evaluation.tolerances.pose.orientation` override first (nextto_01: 0.95); `micro` (instance-dependent) and
targets without a profile rule are unchanged. No library gate change, no rebuild. Counters `grasp.off_target_filtered[.<entry>.<task>]`
(on) and `grasp.off_target_would_filter` (off). Every row filtered -> `Reject("grasps_off_target")`. Count over the 83 eval aliases
(`tray_20261108/turn_count.txt`): 8 rows in 6 aliases - disembedding_10 tablefork 4 of 6 (top_1 0.216, top_4 0.140, top_5 0.212, top_7 0.233
rad; top_2 / top_9 kept), disembedding_07 tape rim_straddle_21, inside_01 brussels sprouts top_8, nextto_02 salt shaker side_2, nextto_03
tablespoon top_2; none emptied (under_large_01's sandal top_1 0.140 < its large 0.15). Tests `test_grasp_turn_filter.py`.
A/B disembedding_10 (v2only, v2maps_int1, N 4 each, svl, trunk_48fe31cc, same base index; template counters off
`grasp.off_target_would_filter` 4, on `grasp.off_target_filtered.tablefork__pqrtkc__s1.disembedding` 4): OFF i7tray_turn_off 12 att / 4 written
/ 0 ok (top_7 x4, placed 11.1-16.3 deg off, symbolic_only), 0.36 GPU-h; ON i7tray_turn_on 14 / 4 / 4 (top_9 x4, placed 1.1-1.8 deg off),
0.36 GPU-h = 0.09 GPU-h per success. The default is the owner's decision.

**Item 4a: `ISPATIALGYM_LAYOUT_PAD_CHECK=all` is the DEFAULT for fixed layouts** (48fe31cc; `0` restores the former pick).

**Item 4b: top_4's release, diagnosed from the demos** (`tray_20261108/release.py`, `release_summary.txt`, `open_sweep.py`; 31 jigsaw places).
- Controller / AG: the R1Pro grippers are `MultiFingerGripperController` in smooth mode, the release command is +1 (fully open, 0.05 m per
  finger; `release_check.gripper_command` 1.0 on every place). The AG weld does NOT go at +1: the fork's pad-clear release
  (`ag_release_on_pad_clear`, cap 0.2 s) kept it 8 physics steps (0.067 s, 2 control steps; 4 steps on 2 places) on every place, released
  `pads_clear` with the pads at 31-34 mm per finger.
- The push: the fingers open 6 -> 14 -> 23 -> 32 -> 41 -> 49 -> 50 mm per control step; nothing moves while welded; from the detach step
  ONE pad stays at 0.0 mm from the piece and drives it 22-28 mm along the closing axis (turning it -12.2..+9.8 deg) until the fingers are
  fully open, 3 control steps, all before the retreat (the retreat moves nothing). top_4's two pads open evenly (<= 2.4 mm apart at the detach).
- Not the goal tray: the piece is released 0.0 +- 0.1 mm above the goal floor; the fingers stay 3.8-5.9 mm above it; goal-wall clearance
  (conservative boxes) >= 31 mm on every top_4 place (the clean top_12 / top_8 places come within -5 mm of the conservative rim band).
- Cause = top_4's geometry: it pinches bar side_G with one pad inside the figure's own slot; swept open, that pad meets the parallel bar
  side_E at 24.3 mm per finger (prediction: 25.7 mm of push; measured 22-28). Same for top_1 (side_E at 22.2), top_5 (side_G at 20.1;
  measured 34-35 mm) and top_10 (side_G at 23.8). top_8 pinches side_C with no other bar in reach (0.3 mm push); top_12 meets side_E only at
  42.5 mm (1-5 mm push). The library checks the closing sweep from the pre-shape, never the release opening.
- Fix proposal (oracle change - asked): release to the descent's pre-shape opening (pinch width / 2 + 10 mm = 16 mm per finger, the
  opening the pad sweep validated) and retreat, fully open only once clear - top_4's pad stops 8 mm short of side_E, top_5's 4 mm, the old
  top_2 / top_3 1.4-1.8 mm. Alternative (library gate): a release sweep in the validator. With the pad-check default, the jigsaw no longer
  draws these rows.

**5y fix: the live joint margin in the axial_brace map builder, the intersection read between built tilts (lead decision 2026-11-08,
owner away).** The lead's decision on the 5y.2 diagnosis: "(a) In the contact-map builder, accept an IK solution only with the live
0.05 rad joint-limit margin ... behind a switch that enters the map identity only when on ... axial_brace only. Rebuild the axial_brace maps
the axial aliases read ... (b) In the sampler, when the live stance's trunk tilt lies between two built tilts, read the contact map as the
INTERSECTION of the two columns ... its own default-off switch. ... Never relax the live check."

- Checked first: neither was on the branch (7d21c982).
- **b4bcb3d0** (both default OFF; the live check is untouched):
  - (a) `ISPATIALGYM_AXIAL_BRACE_MAP_JOINT_MARGIN=1`: `contact_maps.compute_contact_map` counts a chain IK seed only when every left-arm
    joint (the robot's `arm_control_idx`, as `IKChecker.joint_limit_margin` reads it) is at least 0.05 rad (`LIVE_JOINT_LIMIT_MARGIN` =
    `IKChecker.JOINT_LIMIT_MARGIN`) from its limit, at the pre and at the contact (the interpolated approach configurations lie between
    them). A problem with IK only nearer a limit gets the status `ik_margin` (13; in a map's `status_codes` only when on); timing
    `ik_margin_lost`. Set through `contact_config` for the axial_brace kind only; `MapConfig.joint_margin_rad` enters the map key only when
    set, so every earlier map keeps its key.
  - (b) `ISPATIALGYM_AXIAL_BRACE_TILT_INTERSECT=1`: `AlignmentAxial.map_joint_placement` reads the axial_brace map with
    `ContactMap.lookup(tilt_read="intersection")`: a stance tilt more than 5 mrad (`TILT_SNAP_RAD`) from every built tilt and between two
    eligible built tilts calls a class feasible only when BOTH bracketing columns do; within 5 mrad of a built tilt, or outside the eligible
    tilts (e.g. -0.916 beyond -0.9), the nearest column alone. Counters `axial_brace.tilt_read.<built|between|outside|none>`, `.dropped`,
    `.dropped_classes`, `.emptied`. The push_blade read is unchanged.
  - Diagnosis counters, every run, no verdict change: `axial_brace.live.map_feasible.tried / .ok` (a live brace candidate whose class the
    map called feasible at the draw) and `axial_brace.live.map_other.tried / .ok` (the rest).
  - Tests `test_axial_brace_map_fix.py` (5): the margin is the live one, both switches off by default, axial_brace only; the builder refuses
    a 30 mrad contact (`ik_margin`), keeps a 60 mrad one, the key moves only when on; the intersection / built / outside reads; the sampler
    uses it only under its switch, with the counters; the live-vs-map counters. Suite at b4bcb3d0: 23 failed + 3 errors, the same 26 ids as
    2de8f35d (3132 passed = +5). Snapshot trunk_b4bcb3d0.
- **Maps rebuilt** (`$OFH/axialrest_20261108/maps5y`: `job_maps.sh`, `cmp5y.py` -> `cmp5y.out`, `stage5y.sh`; viscam titanrtx jobs
  18385353 on / 18385354 off): the 6 axial_brace maps the axial aliases read, from the libraries v2maps_int2's maps were built from (matched
  by library digest), with this snapshot's builder (i7-brace's, merged 6ab54ba7), switch on and off. The off rebuild reproduces every int2
  key (status agreement 0.997-1.000); the margin keeps 75-80 % of the feasible entries:

| entry | library root | int2 key (off rebuild) | margin key | feasible on / off |
|---|---|---|---|---|
| board_game qgvsnb s1 | brace_20261108/libs_axial_ct2 | 3a8443377a1578e7 (same) | e1bda3a8e1bb118e | 16896 / 21111 = 0.80 |
| keyboard injrgc s1 | libs_axial_ct2 | a8839f1f1b62c385 (same) | cc20e10390e1e91a | 40158 / 50699 = 0.79 |
| magazine tfovwy s1 | libs_axial_ct2 | 0050d164c30ef2fc (same) | 945104cb8ab7c56a | 51474 / 67556 = 0.76 |
| paperback jhaltj s1 | libs_axial_e | 7391a59f4cad4df0 (same) | 97f6d0d61ae81f0f | 20823 / 27142 = 0.77 |
| paperback jhaltj s1x1x2.5 | libs_axial_sj | 43ff4775fb3bc8a5 (same) | 5450eb863c3579c9 | 38909 / 52036 = 0.75 |
| tablespoon tqqwys s1 (settled rest) | axialrest_20261108/libs_brace | 8753ec81a0698bed (same) | 4957182e0672ff07 | 34262 / 43514 = 0.79 |

- Not a pure subset: 4-9 % of the margin map's feasible entries were not feasible off. The map keeps ONE seed per pose (the first that
  passes); with the margin it keeps a different pre seed, whose contact solve can then pass where the first seed's branch jumped or hit the
  slab. Lost to `ik_margin` where off was feasible: 4206 / 11133 / 14645 / 5775 / 11845 / 8103.
- **Staged root** `$OFH/v2maps_int2_axialmargin` = `cp -a v2maps_int2` (int2 holds all 6 maps; INT2.txt 37 lines) + the 6 margin maps; the
  replaced maps under `maps/_superseded_axialmargin/<entry>`; log `AXIALMARGIN.txt`. v2maps_int2 is not touched.
- **The 5y.2 draws on the maps** (`maps5y/pairs5y.py` -> `pairs5y.out`; tqqwys, the draw's (end, spin) group): map-feasible (draw, class)
  pairs - old map, nearest read 8 (the diagnosis's 8; live-solvable with the margin 3, live-solved 0); old map + intersection 3 (the three at
  -0.916, outside the built tilts); (a) nearest 2 (310002 / 310003 blade_0- at -0.454 - the pairs the diagnosis found lost to the TRUNK:
  0 / 64 seeds live); (a) + intersection 0. So (a) alone does not make (b) moot: the trunk losses survive the margin.

**5y fix A/B** (trunk_b4bcb3d0; viscam titanrtx for every arm - minyeong's svl QoS was full; v2only N 4, indices 310000-310003, alias
memory OFF, TURN_COVER=1 + OPPOSITE=1, map overlay r3; `i7ar_5yf_off` root v2maps_int2, `i7ar_5yf_a` root v2maps_int2_axialmargin,
`i7ar_5yf_ab` the same + `ISPATIALGYM_AXIAL_BRACE_TILT_INTERSECT=1`; 12 jobs, all completed first run; `axialrest_20261108/ab5y.py` ->
`ab5y_5yfix.out`). att / written / ok, GPU-h per success:

| alias | off | (a) margin maps | (a) + (b) intersection |
|---|---|---|---|
| axial_05 | 20 / 4 / 3, 0.108 | 12 / 4 / 4, 0.057 | 18 / 4 / 3, 0.091 |
| axial_01 | 11 / 4 / 2, 0.157 | 12 / 4 / 3, 0.088 | 9 / 4 / 4, 0.049 |
| axial_03 | 12 / 4 / 4, 0.048 | 7 / 4 / 3, 0.059 | 8 / 4 / 4, 0.036 |
| axial_04 | 18 / 4 / 4, 0.045 | 12 / 4 / 2, 0.081 | 12 / 4 / 2, 0.081 |
| all | 61 / 16 / 13, 1.01 GPU-h = 0.077 | 43 / 16 / 12, 0.83 = 0.069 | 47 / 16 / 13, 0.78 = 0.060 |
| draws paired on the maps / push-only | 41 / 27 | 20 / 31 | 4 / 58 |
| live brace candidates of a MAP-FEASIBLE class: ok / tried | 0 / 41 | 0 / 32 | 0 / 8 |
| live brace candidates of other classes: ok / tried | 15 / 1149 | 14 / 556 | 2 / 189 |
| brace present in the written demos (executed in the demo) | 4 (4) | 3 (3) | 1 (1) |
| target met the brace (push brace_hits > 0) | 0 | 0 | 0 |
| intersection read: between / dropped / emptied lookups | - | - | 3470 / 368 / 351 (outside 930, built 24) |

- Both switches act where they should: the margin maps halve the paired draws (41 -> 20) and the map-feasible brace classes tried live
  (41 -> 32); the intersection read empties 351 brace lookups and leaves 4 paired draws and 8 map-feasible candidates.
- But no map-feasible brace class solved live in ANY arm (0 / 41, 0 / 32, 0 / 8): the failures stay "exact IK failed at pose 1 of 2" (the
  pre) and "intersects the target at its goal pose (margin 3 mm)". Every brace in a written demo came from a class the map did not call
  feasible at the draw (opposite-end blades at another pitch / face index - `opp_blade_60`, `opp_blade_30@f2` - tried because the draw was
  paired). So making the map stricter removes pairs, and with them the incidental braces: brace present 4 -> 3 -> 1; none of them met the
  target (brace_hits 0) in any arm.
- Success 13 / 12 / 13 of 16 is within N 4 noise; GPU-h per success falls (fewer attempts: the draws stop at a push-feasible draw when no
  pair exists).
- On the diagnosis's 5 draws (`pairs5y.out`) (a) + (b) leave 0 of the 8 false map-feasible pairs; in the runs the remaining
  map-feasible candidates still fail live, so the 8 cm lattice / full-world collisions ((c), not built) are what is left.
- **Defaults: neither flipped.** Both act, but the result is not "no worse" for what they are for: the brace present drops (4 -> 3 -> 1) and
  the map's live precision stays 0. Both stay default OFF; the margin maps are staged in v2maps_int2_axialmargin, not in int2.

### i7-brace (owner 2026-11-08 3.3 / 3.4; branch `i7-brace` from of-v2maps cf1f5d2d)

- **3.3 the brace-load pusher** (a1ef1c8d): built `kinematic_only` without a fixed base (dynamic in OmniGibson, the touching stand-in's
  bug of 021aa89c) - now `PUSHER_KINEMATIC` (fixed_base + kinematic). `validate_contacts --revalidate` re-runs exactly the entries an
  earlier run validated (old record -> `validation_prev`, uncapped). Every brace library the consolidated root reads was re-validated
  (`$OFH/brace_20261108/libs_touch` / `libs_axial` / `libs_axial_sj` / `libs_axial_e`, per-entry changes: `verdict_changes.py`), maps
  valid-only (owner A6), staged in `$OFH/v2maps_int1_brace33` (cp -a v2maps_int1; old maps in `_superseded_brace33/`).
- **3.4 axial** (63da6197; default off): library param `clear_turns` builds the axial_brace entries once at the goal, keyed by pushed end x
  spin, each with `goal.clear_turn_max_deg` (the contiguous range over 5..90 deg of the sampler's own `blade_sweep_clear`), plus the
  OPPOSITE-END candidates (`axial_opposite_brace_candidates`: the far end's face the far end arrives at, resisting the same turn; classes
  `<end><spin>spin_opp_*`, tried after the current ones; no `side_high` analog). Sampler switches `ISPATIALGYM_AXIAL_BRACE_TURN_COVER=1`
  (the map keeps the entries clear for the drawn turn) and `ISPATIALGYM_AXIAL_BRACE_OPPOSITE=1` (the opposite candidates after the current
  ones, the record's direction from the candidate); kit `axial_brace.opposite_tried` / `.opposite_used`. The exact two-arm check unchanged.
- **3.4 the 12 braceless axial instances** (i7_contactfu): 9 `no_joint_cell` (turns 21-64 deg: the axial_brace map has no class of the
  pushed end / spin at the goal root - reach, not the sweep), 3 `brace_ik_failed` (axial_04 magazine, 21.7-21.9 deg: the blades fail the
  left arm's IK at the pre or meet the goal-posed target within 3 mm; all side pads sweep-skipped). The sweep test is monotone in the turn
  on synthetic boxes (a pad clear at 45 deg is clear below it), so the failures at <= 45 deg are reach / goal-pose collisions.
- **3.4 world_vertical `wall_brace`**: `alignment_wall.WallGeometry` = the task's hang / contact geometry moved verbatim out of
  `AlignmentWall` (which inherits it: the live brace choice unchanged, test).
  - *Library* (`contact_library.build_wall_brace`, family `wall_brace`): the task's own braces on the level (goal) pose at the synthetic
    spot `WALL_RULE_SPOT` (wall plane x = 0 facing -x, the HINGE at z 0: a map / the validator add the hinge height), keyed by the tilt
    sign; the hinge at the collision pieces' volume centroid (`pieces_com`); per entry the task's brace arm / push arm (`wall.brace_arm`:
    + tilt and - tilt brace with opposite arms), the oracle record's extras (`pose_class` = the kind, `placed_at: goal`, `contact_face`);
    `goal.clear_delta_min_deg .. clear_delta_max_deg` = the longest contiguous run of grid tilts (5..45 deg) where the task's `brace_clear`
    holds (148f9fb5: the stop at the goal edge clears the tilted hull more the farther the target is tilted - a small target's brace is
    clear from 10-20 deg on; the run from 5 deg dropped every license_plate entry); `load` = the leveling push carried on past level (the
    pushed corner of the 20 deg tilt, its tangent about the COM hinge, 3-D in the wall plane) with the braced point's own direction.
  - *Validator* (`validate_contacts`): `KitContactWorld`'s wall = a KINEMATIC box (`PUSHER_KINEMATIC`, as the pusher fix; category `walls`
    so the oracle's `SimState.contact_supports` gives it to the planner as the target's support, as a scene wall); `hang()` replaces the
    settle: the slab out of reach, the target at its level pose with the hinge at `validate.support_z` (1.45 m), on the task's hinge
    (`create_hinge` at the PhysX COM about the wall normal, parented to the wall, `AlignmentWall.hinge_resistance` of the variant); the
    probe and the whole-robot contact policy against the wall. Each entry is validated by ITS brace arm (`wall_arm_entries`; job_contacts
    `GC_VALIDATE_ARMS="left right"`); the stance relation `wall_brace` from 26 braced demos (contact 0.73-0.86 m ahead, 0.05-0.30 m to the
    brace arm's side, mirrored for the right arm). The oracle's own Brace runs (closed fist, touch stop, back-off), then the LOAD (the
    leveling push carried on past level): the pusher's 3 cm face meets the pushed corner square to its tangent and is carried along the
    corner's ARC about the hinge; the sweep is stopped `brace_press_m` (at the braced point) after the target first turns 2 mrad past level
    (`wall_load_stop`, the contact measured on the SIGNED turn) - the touching / axial calibration (the object driven 1 cm past first
    contact); the target is woken every step (the sleep assist sleeps it while the brace stands). `judge_wall_standing` = valid BEFORE the
    load (the oracle reached its stop, no non-finger wall load); `judge_wall_brace` = after it: clean oracle run, the braced point along its
    load direction <= `brace_hold_max_m` (`not_held`) and >= 1 mm (`not_loaded`), the brace fingers pressed the target >= 1 N
    (`not_braced`), it travelled at most the stop's gap + the hand's own displacement + 4 mm (`slipped_past`), the hinge axis not twisted
    (`topple`), the hand within `brace_eef_drift_m` (`hand_pushed`), no non-finger link loading the wall. The load took seven runs to make
    exact (DECISIONS "WALL VALIDATION RUN 1..7"): the material by distance to the corner line missed thick targets (vacuous passes), the
    8 cm face pushed the edge, the sleeping target ignored the pusher, a straight path and the contact offsets over-turned it, an unsigned
    turn mistook a backward settle for the contact.
  - *Validation* (run 7, `$OFH/brace_20261108/wall/libs`, trunk_45612c00; valid BEFORE the load / AFTER it): painting dacihj 34 / 31 of
    36, signpost vvjtkv 23 / 15 of 24 (the s+25/+ and s-25/- fists let the target slide past), license_plate naclac 34 / 10 of 36,
    paper_towel_dispenser jacqxv 36 / 5 of 36 (its braced point goes past the fist: not_braced / slipped_past), wall_clock 0 entries
    (push-only) - 132 entries, 127 / 61. The maps (01154b97, one arm per process: the second arm in one process ran out of GPU memory)
    from the 61 valid entries: 8 maps (4 objects x 2 arms), staged in `$OFH/v2maps_int1_brace34w` (= v2maps_int1_brace34b + the
    wall_brace maps; found by the model-id directory, no glib entry / legacy key needed).
  - *Maps* (`contact_maps` kind `wall_brace`): per stand-off d_e 0.35-0.90 (prompt0's Phase-2 d lattice) the wall box at x = d_e and the
    hung root's row in front of it (`cell_de`), laterals every 0.05 m over +-0.60, the tucked idle arm, trunk eligibility per d_e, one map
    per brace arm, valid entries only; `ContactMap.find_near` (the hinge height, no support anchor); the lookup stays in the stand-off's row.
  - *Sampler* (`ISPATIALGYM_WALL_BRACE_MAP=1`, default OFF): at each stance with push reach the map (brace arm of the drawn tilt sign, at
    the hinge height) gives the feasible classes (tilt sign + kind + wrist) whose clear range covers |delta| - they replace the reach
    volume's brace candidates; the stand-off of the row is read from the hung root's position (the patch protrusion and the COM included);
    the exact two-arm check is unchanged; a missing map keeps the reach-volume rule. Kit `wall_brace_map.hit / miss / stances /
    stance_feasible / stance_none / reach_volume_kept / reach_volume_none / map_only / reach_only / cover_dropped / written_brace /
    written_push_only`.
  - *Runs* (v2only N 4, trunk_01154b97, `v2maps_int1_brace34w`, base index 310000 both arms, svl titanrtx): `i7_wall_off` 97 att /
    18 wr / 18 ok, 0.106 GPU-h per success, 115.5 exact IK per attempt; `i7_wall_on` (`ISPATIALGYM_WALL_BRACE_MAP=1`) 104 / 18 / 18, 0.107,
    62.6. Per alias off vs on: 01 6/4/4 vs 6/4/4; 02 (wall clock, push-only) identical; 03 7/4/4 vs 8/4/4 (the dispenser's braces become
    side_high: its below_high entries fail the load); 04 51/2/2 vs 51/2/2 (too_few_visible, not the brace); 05 27/4/4 vs 33/4/4. Brace
    present 14 / 14 in both; where it acted, the map's feasible stances vs the reach volume's: 01 1047 vs 485, 03 137 vs 211, 04 7450 vs
    7155, 05 2953 vs 2709 (of 1886 / 873 / 17420 / 8904 screened). The switch stays default OFF (owner).
- **Owner 2026-11-08 (decisions 20261108h 5w.1): the wall load test APPROVED** for the `wall_brace` family only: the corner's arc load
  stopped `brace_press_m` at the braced point after the measured (signed-turn) contact, and its rejection reasons `not_loaded` /
  `not_braced` / `slipped_past` (4 mm). It already is the code of run 7 (45612c00, `validate_contacts.judge_wall_brace` /
  `wall_load_stop`, read only for a `wall_brace` library); no shared gate or contact policy changed, no code change.
- **Owner 2026-11-08 (5w.2): `ISPATIALGYM_WALL_BRACE_MAP` DEFAULT ON** (separate commit; `=0` turns it off), with the wall maps staged in
  **`$OFH/v2maps_int1_brace34w`** (= v2maps_int1_brace34b + the 8 run-7 wall_brace maps: the root to fold into the integration).
  Confirmation run: world_vertical 01-05, N 4 v2only on that snapshot and root, no switch in the environment: `i7_wall_def`
  (trunk_b1a07ba6) 104 att / 18 wr / 18 ok, 0.105 GPU-h per success, 62.6 exact IK per attempt - per alias identical to `i7_wall_on`.
- **Owner 2026-11-08 (5w.3): the touching / axial brace validator, DIAGNOSIS AND COUNT ONLY** (no rebuild, no default; the switches below
  are diagnostic, default off, unset = today's load bit-identical): `ISPATIALGYM_GC_BRACE_LOAD_DIAG=1` records the load
  (`debug.load_diag`: asleep at the load, the assist's sleeps / wakes, the LOAD POINT's travel along the load - the arriving end on the
  axial COM hinge, whose root never translates -, the measured first contact vs the computed face); `_WAKE=1` wakes the braced object at
  every pusher step (the wall's 220ff7db); `_MEASURED=1` stops the pusher `brace_press_m` after the measured contact (the wall's
  fae1e4d8). Re-validation arms D / W / WM of the live libraries (libs_touch, libs_axial_ct2, libs_axial_e jhaltj s1, libs_axial_sj) under
  `$OFH/brace_20261108/w53/<arm>`; counts in DECISIONS ("i7-brace 5w.3"). `validate_contacts probe` (5w.3(b)): the real arm pressed into
  a sleeping partner at 0.1-3.3 mm/step under each sleep condition (`KitContactWorld.probe_wake`).
  - *The sleeping object*: the restored partner falls asleep by PhysX's OWN sleep (threshold 0.005) ~12 steps after the trial's wake,
    before the assist's 30-step window (the assist slept none of them), and the teleported kinematic pusher never wakes it: in the record
    414 of the 462 valid touching / axial brace verdicts were judged with the load never reaching the object (touch 110 / 110, magazine
    96 / 96, keyboard 79 / 94, jhaltj 44 / 44 and 85 / 90; board_game 0 / 28 - its blades press it awake). Woken at every pusher step
    (W) 84 verdicts change (83 hand_pushed, 1 not_held): touch 1 / 110, keyboard 12 / 94, magazine 18 / 96, jhaltj s1x1x2.5 53 / 90.
  - *The load start*: the measured contact is at the computed face on the touch partners but 1.1-2.8 cm EARLY (medians) on the axial
    keyboard / jhaltj s1x1x2.5 / board_game, so the computed load over-presses them; stopped 1 cm after the measured contact (WM) 104
    verdicts change both ways (60 invalid -> valid, 44 valid -> invalid at the 1 cm hand_pushed threshold). Valid D -> W -> WM: 462 -> 378 ->
    394. Still vacuous with both (awake, the load point < 1 mm): magazine 48 / 96, jhaltj s1 34 / 44 trials - the axial load path misses or
    pushes through the COM hinge, and `not_held` cannot fire on a hinged target (the judge reads its pinned root). Owner: rebuild / fix.
  - *Collection / replay / eval* (the same `SleepAssist` in run_isg and run_isg_vec; 3246 written demos mined, the probe): a hand slower
    than 0.3 mm/step is not "moving" for the assist - it neither wakes the body proactively nor keeps it awake; PhysX itself wakes a
    sleeping body when the hand is 1.8-2.2 cm from its box, at every speed (21 / 21); but under the assist a 0.1-0.2 mm/step approach is
    RE-SLEPT at once (the window is already full of still poses) and stays asleep while pressed (12 / 12 probes, 73-124 steps, up to 15 N)
    - it still moves as far as an awake body and no kick was measured (object speeds within the awake control's range). In the demos 2325 first motions after an assist
    sleep came without the assist's wake (1514 with a slow hand); kicks 2.2 % vs 1.7 % after a wake. Teleports (`set_position_orientation`,
    the scene state load) wake the teleported body; Executor.restore wakes every tracked object.
- **Lead decision 2026-11-08 (owner away) on 5w.3: the FIX** - `ISPATIALGYM_GC_BRACE_LOAD_GEOM=1` (default OFF, next to the 5w.3
  diagnostic switches; 8ec97f10 + 1fb0fe77 + 6dcb94f8, merged with of-v2maps 68c6ffc2 in daf44408):
  - *Axial load* (`axial_pusher_path`, `KitContactWorld._axial_load`): on the COM hinge the pusher meets the PUSHED end - the library end
    point's distance along the long axis THROUGH THE HINGE (the library's point sits on its builder's hull-vertex-mean centre, 6.8 cm off
    jhaltj's centre line), the far end's mirror for an opposite-end entry - square to that end's arc tangent (= +-`load.direction`), and is
    carried along the arc about the hinge, stopped `brace_press_m` at the braced end after the MEASURED turn. The box as the wall load's:
    8 cm upright, 3 cm across, 5 cm along the travel. The old straight load used an UNTURNED box (thickness along world y): a load along x
    met the target with its 8 cm side (contacts 11-25 mm early), thin targets started inside it and were lifted, not turned.
  - *Judge* (`judge_axial_brace`): the TURN, not the hinge-pinned root - `not_held` = the braced end's arc travel > `brace_hold_max_m`,
    `not_loaded` = the pushed end never travelled 1 mm (never a pass), `not_braced` = the brace fingers never pressed 1 N during the load
    (the wall judge's rule). Touching braces (load +y, unchanged box): `not_loaded` from the measured load point (`brace_load_reached`).
  - *Tests* (test_brace_lib.py, 30 pass): switch off by default; box along the load; the arc / face / opposite / yaw 90 / on-axis pushed end;
    the face never spans a short lever; no material raises; the judge never passes a missed load and reads the turn on a pinned root.
    Suite trunk_1fb0fe77 = the of-v2maps baseline's same 21 ids.
  - *Re-validation* (arm F = WAKE + MEASURED + GEOM, --revalidate of cp -a copies, `$OFH/brace_20261108/w53fix/F`, axial on trunk_6dcb94f8,
    touch on trunk_1fb0fe77; table `w53fix/tab_fix.py`): valid int2-map -> F, the load reached the target (touched) in 648 / 648 trials
    (the record: 48 / 462 valid): gkpmii 40 -> 40, dkhgxn 38 -> 37 (1 not_held), ntbrtz 32 -> 32; board_game 28 -> 107 (90 invalid ->
    valid: the old box over-pressed them, hand_pushed), keyboard 94 -> 68 (26 not_braced), magazine 96 -> 92, jhaltj s1 44 -> 12 (32
    not_braced: the blades beside the thin book never touch it during a 1 cm turn), jhaltj s1x1x2.5 90 -> 96, tablespoon tqqwys 48 -> 40;
    510 -> 524. Earlier runs v1 (box unturned, pushed point off-axis: jhaltj 44 / 44 never turned) and v2 (8 cm across face spanning the
    spoon's hinge: 29 / 48 not loaded) are in `w53fix/v1_cancelled`, `w53fix/v2_axial`. Note: the kinematic load drives the braced end
    ~1 cm in every trial (held or not); a valid verdict means the brace hand stood (drift < 1 cm) and its fingers took the load.
  - *Maps*: the 9 brace maps rebuilt from the F verdicts with trunk_6dcb94f8's builder (control: the recorded libraries reproduce int2's 6
    axial keys exactly; the touch maps in int2 are int1's older builds, 889c0dbb / 8e3aee3b / b87cb752) and staged in
    **`$OFH/v2maps_int2_braceload`** = cp -a v2maps_int2 at INT2.txt line 46 + the 9 maps (old -> maps/_superseded_braceload; log
    BRACELOAD.txt); check_int2 ERRORS 0. v2maps_int2 untouched.
  - *A/B* (N 4 v2only, trunk_6dcb94f8 both arms, base 350000 per-tag ledgers, alias memory off, svl titanrtx; `w53fix/ab_bl.py`):
    int2 `i7b_bl_int2` vs staged `i7b_bl_load` (att / wr / ok, GPU-h per success): touching_01 9/4/4 0.116 vs 10/4/4 0.102; touching_02
    46/4/4 0.127 vs 46/4/4 0.096; touching_04 26/4/1 1.021 vs 26/4/2 0.407; axial_01 14/4/4 0.076 vs 12/4/4 0.077; axial_03 4/4/3 0.067
    both; axial_04 12/4/4 0.042 vs 26/3/3 0.070 - its replicate at 350100 is IDENTICAL in both arms (26/4/4 0.068, the same brace
    counters; no brace in either: brace_ik_failed / no_joint_cell), so run noise. Total 111/24/20 0.133 vs 124/23/20 0.117; written brace
    9 vs 9, landed 8 vs 9. No alias worse, touching_04 better -> ready to fold under lead #40: the three commits (switch default OFF) +
    `$OFH/v2maps_int2_braceload`. The woken validator does not strengthen the case for the shared sleep-assist change (1 / 110 touch
    verdicts; the axial changes are the geometry).
- **Opposite-end validation**: the stratified quota counted the opposite entries in the current-end group (validated first), so none was
  validated; 9d2ce7e3 makes them their own group (`libs_axial_ct2`: injrgc 47, qgvsnb 14, tfovwy 48 valid opposite entries).
- **Runs** (v2only N 4): i7_brace33 (the 3.3 maps, switches off) touching_01 27 att / 4 wr / 4 ok (brace 3 / 4), touching_02 79 / 4 / 3 (4 / 4),
  axial_01 / 03 / 04 56 / 12 / 12 (brace 0 / 12); i7_brace34 (clear-turn maps, opposite not validated) 42 / 12 / 11, brace 0 / 12;
  i7_brace34b (+ the validated opposite entries, `v2maps_int1_brace34b`, TURN_COVER=1 OPPOSITE=1) 53 / 11 / 10, 0.070 GPU-h per success,
  BRACE PRESENT 4 / 11 (axial_03 3 / 4, axial_01 1 / 4, axial_04 0 / 3), every brace an opposite-end blade (`opp_blade_30`), braced demos
  3 / 4 ok; kit axial_brace.paired 25 / opposite_tried 19 / opposite_used 10.
- **Owner 2026-11-08 (answers 20261108b item 4): the axial brace switches DEFAULT ON** (separate commit): `ISPATIALGYM_AXIAL_BRACE_TURN_COVER`
  and `ISPATIALGYM_AXIAL_BRACE_OPPOSITE` read `1` when unset (`=0` turns either off). The axial brace maps the integration root reads are
  the brace34b maps - the clear-turn library plus the opposite-end entries validated as their own group, staged in
  **`$OFH/v2maps_int1_brace34b`: the root to fold into the integration**. Confirmation run: axial_01 / 03 / 04, N 4 v2only on that
  snapshot and root, no switch in the environment: `i7_brace_def` (trunk_59f7967b) 40 att / 12 wr / 10 ok, 0.067 GPU-h per
  success, brace present 4 / 12 (i7_brace34b with the env switches: 53 / 11 / 10, 4 / 11).
- **Owner 2026-11-08 (item 5): board_game qgvsnb** - the 14 of 104 blade braces that hold the real load are accepted; board_game mostly
  runs push-only; no other brace type is searched. Coverage from the staged map (`brace_20261108/bgcov.py`; % of (cell, yaw slot, d_e,
  trunk) problems with a feasible covering class): the 14 current-end entries 3-4 classes / 5.0 % (spin +) and 3.7 % (spin -) for turns
  20-45 deg, 1-2 classes / 1.5-2.8 % for 50-70 deg (only the yaw 0 / 180 slots' entries clear past 45); with the 14 opposite-end ones
  7.7 / 6.2 % and 4.4 / 3.1 %.

## inside_large_01: the recorded transit over the book and the spine guard (branch `i7-overpath`, 2026-11-08)

Owner decisions 3(a) / 3(b) of `prompt_goalmaps_axialrest_overpath.md`; record `$OFH/DECISIONS.md` ("i7-overpath" lines). Branch from of-v2maps
93229922, not pushed. **No default flipped.**

| commit | what |
|---|---|
| e4443d14 | (a) `ISPATIALGYM_OVERHANG_OVERPATH` (default off): `IORSLarge._overpath_plan` plans the transit escape end -> rise -> across -> pre-grasp, records `grasps[0].overpath` (isg-overpath/1); `Grasp.overpath`; the oracle follows it (`execute.follow_overpath` -> `Motions.recorded_joint_path`), the program check replays it; diagnostic `transit_book_clearance` on every overhang grasp (both arms). (b) the spine guard `generation.overhang_required_side` / `overhang_pinch_edge_max_m` (kit `overhang.side_rejected[.entry/.grasp]`) |
| 06de48d9 | the guard ON in inside_large_01's customs (-y, 0.03 m); replay of the int1_1 / int1_3 / i7_po draws: 3 entries, 2064 drawn grasps, 0 rejections |
| b911cd3d | tracking at the oracle's soft tolerances, else the seeded exact IK of every ~2 cm / 0.1 rad waypoint (jump <= 0.3 rad) |
| 09f97815 | (CORRECTION) memo key + `overpath.MEMO_VERSION` "overpath/2"; the turn push -> pre-grasp placed by schedule (`over_pre`, `at_start`, `slerp`, tried in order; kit `overpath.orient.<s>.planned/.rejected`); every rejection names the leg, waypoint, z, distance from the base, turn done, joint nearest its limit and whether any IK branch reaches the waypoint; `ISPATIALGYM_OVERPATH_TRY_ALL=1` logs `OVERPATH_DIAG` |

**Arm b is not valid yet.** i7op2_b (trunk_b911cd3d) was cancelled: the persisted overhang fail memo replayed 28 failures of the buggy
first run under the same key (fixed by the memo version), and the fresh picks planned 0 paths.

**Diagnosis (i7op3_diag, trunk_09f97815, i7_po indices 300101-4, every schedule tried):** 9 picks reached the transit, 0 of 27
pick x schedule planned. The turn is 126-139 deg; the escape leaves the hand at z 0.941 (above book top 0.797 + 0.08, so the across leg runs
at the escape height), 0.59-0.80 m from the base, with joint4 or joint7 0.000-0.009 rad from a limit. Every failing waypoint is reached by
NO IK branch: `over_pre` loses the push orientation after 8-47 % of the across leg; `at_start` loses the turn after 17-46 deg, and where it
completes (3 picks) the pre-grasp orientation is on another branch (jump 2.9 rad, joint1 / joint3); `slerp` after 11-69 deg. The push and
the pre-grasp configurations lie on different IK branches, so no Cartesian transit exists; a joint-space move is needed (lead / owner
question: record the oracle's own planner's transit with the book protected, or a checked joint interpolation via above the pre-grasp).

**Arm a (guard on, path off; i7op2_a, trunk_b911cd3d, v2only, v2maps_int1, memory off, svl titanrtx, 331000-3 + 331100-3):** 14 attempts,
8 written, 6 / 8 demos ok, 1.42 GPU-h = 0.237 GPU-h per success. Grasp failures: 331100 plan_failed at grasp[5] (no lift: link6-zed self
collision; no plan to the point above), 331102 execution at grasp[4] (joints off 0.622 rad); 331000 recovered on its 2nd attempt after the
via-above "still passes the book at 9.6 mm". Executed transit's minimum clearance to the book (`transit_book_clearance`): 6.1 mm (331000),
others 27.7-67.5 mm. Place: all 6 grasped books placed. `overhang.side_rejected` 0 (checked: 2 entries, 126 grasps). 21 picks came from the
persisted memo of the cancelled i7op_a (same customs, same key: deterministic failures of the same code).

**Suite** at 09f97815 vs 93229922: base 23 failed + 3 errors; head the same 26 plus 5 `test_split_cone` tests, which fail only in the full
run on the missing untracked stray `runlogs/.../tabletop__s__t__apple-x__k.json` (moved out per the rules) and pass alone (11 / 11).

### P1: the oracle's own planner records the transit (owner 2026-11-08 item 4, commit a55eea8a)

The owner's decision on the open question above: the sampler runs the oracle's own motion planner from the escape's end to the pre-grasp
and records that joint trajectory. It **replaces the Cartesian attempt under the same switch** `ISPATIALGYM_OVERHANG_OVERPATH` (default off;
`overpath.design()` "planner"; `ISPATIALGYM_OVERPATH_DESIGN=cartesian` keeps the 0 / 27 legs as a diagnostic). Fail-memo key version
`overpath/3`: no Cartesian failure applies.

- `sampling/v2/overpath_planner.py`: an oracle `Motions` on the sampler's cuRobo backend whose state is a joint vector
  (`execute_joint_trajectory` records instead of simulating). The transit is the oracle's live code path for this grasp:
  `Motions.protecting` (the book, inflated by `BOOK_MARGIN_M` 0.03) around `transit_dip_gate` around `execute.reach_pre_pose` ->
  `Motions.arm_to(pose=pre-grasp, q_hint=witness)`. That covers the straight ramp, the ramp via a posture, the joint plans and their
  retries, the pose fallback, the plan via a posture, and the protect gate's via-above and re-plans. `MotionsConfig` is set as `run_isg`
  sets it (PLAN_LADDER). Not replicated: the supports-ignored fallbacks, and the oracle backend's 2 trajopt seeds (the sampler's backend
  keeps 4).
- `IORSLarge._overpath_plan_planner`: puts the book at its settled post-push pose in the scene, with the desk and the scene as obstacles.
  The witness is the exact IK of the pre-grasp at the grasp stance, retried with the book ignored when that fails. It plans up to 2
  rounds. `_overpath_verify` re-checks the recorded waypoints as the oracle runs them: margin-0 collisions with the book present, the
  arm's spheres >= 3 cm from the book box, the end at the pre-grasp, and the approach tracked. The record (isg-overpath/1, design
  "planner") is followed by the existing `follow_overpath` / `recorded_joint_path`; a run-time refusal falls back to today's live plan.
  `check_overpath` replays it in the program check.
- Kit: `overpath.planned`, `.rejected.<reason>`, `.planner.rounds`, `.planner.branch.<b>`, `.planner.failed.<stage>`, `.planner_s`.
  Oracle notes: `overpath.{followed, fallback, joint_err_rad, pre_grasp_err_m}` and `transit_book_clearance`.
- Tests (`test_overhang_overpath.py`, 27): switch off = today; a recorded path clears the book by the margin (the real Motions ladder on a
  toy backend: the 7.8 mm plan -> via-above); a failed re-check is re-planned or rejected; the program expands with the record; the
  program check replays it. Suite a55eea8a vs 442c7eaf: base 23 failed + 3 errors, head the same plus 2 `inspect.getsource` tests
  disturbed by an edit during the run (they pass alone).

**A/B** (inside_large_01, svl titanrtx, v2only, v2maps_int1, memory off, overlay r2, 331000-3 + 331100-3; `$OFH/jobs/i7op_p1/p1_tab.py`):

| arm | tag / trunk | att / written / ok | GPU-h (per success) | plan_failed at the grasp | executed transit's min book clearance |
|---|---|---|---|---|---|
| a: guard on, path off | i7op2_a / b911cd3d | 14 / 8 / 6 | 1.42 (0.237) | 2 attempts in 2 demos, 1 demo failed | 0.6 cm (331000), others 2.8-6.8 cm |
| b: guard on, P1 on | i7op4_b / a55eea8a | 33 / 8 / 8 | 9.04 (1.130) | 0 | 3.3-6.7 cm (followed: 3.3-5.2) |

Arm b counters:

- Sampler: planned 83, rejected 54 (plan 39, witness 8, book 6, collision 1), planner rounds 190 (failed trajopt 57, protected 20).
  Branches: straight_via 63, straight 20, via_above 8, via_posture 2, protect_replan 1, trajopt 1. Recorded 12, re-planned 20.
  23969 s of planner time; sample walls 8527 / 23715 s vs 1475 / 2804.
- Oracle: **followed 4 / 8** (joint error 0.000 rad; the pre-grasp 2.2-3.4 cm off, because the live book drifted, and the approach closed
  it). **Refused 4 / 8**, all ok on the live plan: start gap 0.45 / 2.62 / 0.36 rad (> 0.35), and one record that passed the live book at
  11.5 mm.

**Default: the owner's call** (P1 removed the grasp's plan_failed and the sub-3 cm transits in this N 8, at ~4.8x GPU-h per success; the
live fallback plans in arm b also succeeded, so N 8 cannot separate the two).

## Stance memo, floor redraw and the attempt cap (branch `i7-stancememo`, owner decisions 2026-11-08 5x.1 / 5x.3 / 5x.4)

Owner decisions (verbatim in the lead's `owner_20261108i.md`; record `$OFH/DECISIONS.md`, "i7-stancememo" lines, runs in
`$OFH/stancememo_20261108`). Branch from of-v2maps 93229922 with i7-startdraw merged at its last default commit 1ba40aae (db50e839), so
every arm draws the robot's start per attempt (IORS large floor: stance A per attempt around the template's c_t and stance-B pool).

### 5x.1 verification (i7-diag item 2)

On i7diag_floor (fcddc495, before the start draw) under_large_03 instance 330003 took 15 attempts, 9 rejected in the B / C search
(1,142-2,221 s each). Every B try of attempts 3 / 5 / 8 / 9 / 11 failed on the A -> B leg (`init -> manipulation` path_collision at swept
pose 10-35 of ~400, 17 cm from the template's fixed stance A): that repeat depends on stance A, which the start draw now redraws per
attempt. The repeats that do not depend on A remain: the C tries of a B (in attempts 6 / 8 / 9 / 12 / 13, 11-34 C keys each, nearly all
exhausted in an earlier attempt, mostly by pairs dead at B's exact IK or "no push chain"), and ontop_large_02's C skipped for a failed
carry leg. The audit's 1,385 s per written demo is therefore an upper bound for the code before the start draw.

### The switches (memo / redraw default off; the cap default on since decision #33)

| switch | commit | what |
|---|---|---|
| `ISPATIALGYM_STANCE_MEMO=1` (a) | c280fab6 | per-INSTANCE memo of failing stance B / C of an IORS large floor init (`ispatialgym/sampling/v2/stance_memo.py`). A B that failed is dropped from the pre-screen's random draw; a C that failed at that B is skipped and does not use up one of the `n_try` tries. A failure that depends on stance A (the A -> B leg, the A -> B drive past the placed target) is keyed with A's pose; the others with B (and C) and A's trunk translate (the carry's travel trunk). Cleared when the sampler moves to the next instance and on a redraw. Every 4th hit (`ISPATIALGYM_STANCE_MEMO_RECHECK`) is searched fresh. |
| `ISPATIALGYM_FLOOR_REDRAW=1` (b) | c280fab6 | after 2 attempts on one floor start failed in the B / C search, the next attempt redraws the floor start with `try_surface("floor")` and all its checks (free floor anchor within the geodesic, stances B collision-free and seeing c_t, stance A framing the target and the partner region, the A -> B leg, the stance-C existence test; the attempt's eval-repeat checks as always). A redraw that finds none keeps the start (`redraw.failed.<reason>`); the next instance starts on the template's start again. |
| `ISPATIALGYM_ATTEMPT_CAP` (DEFAULT ON since the lead's decision #33, own commit; `=0` = no cap) | 1a68082b | the per-attempt cap max(300 s, 5 x the alias's median written attempt) from `ispatialgym/sampling/v2/attempt_cap_medians.tsv` (the audit's primary set, `scripts/oracle_fix/time_cap_count.py table`); armed while the attempt samples (the start draw / redraw exempt), checked at the kit's IK and collision queries and the samplers' deadline checks; a cut attempt is `Reject("time_cap")`. The fixed budgets stay as ceilings. |

Kit counters: `memo.recorded.<b|c>`, `memo.skipped.<b|c>.<reason>`, `memo.recheck.<b|c>`, `memo.would_have_passed.<b|c>` (must be ~0),
`memo.recheck_failed.<b|c>`, `memo.cleared.<instance|redraw>`; `redraw.search_fail`, `redraw.fired`, `redraw.failed.<reason>`,
`redraw.template_restored`, with (b) off `redraw.would_fire` and the log-only `redraw.third_attempt_would_have_succeeded.<written|search>` /
`redraw.third_attempt_failed.<search|start>`; `time_cap.armed`, `time_cap.cut`, `time_cap.cut_at.<where>`. Tests:
`test_stance_memo.py` (per instance, a redraw clears it, off = today), `test_attempt_cap.py`.

### 5x.3 count on the logs (CPU, before any run)

`time_cap_count.py count` on the audit's `attempts_r4.jsonl`, primary set (2,571 attempts): the cap would cut 18 attempts (their search
passes the cap), 1 of them written (under_large_01 300019/3, 2,158 s against a cap of 1,858 s), 17,168 s past the cap; all 17 cut
rejected attempts were on instances a later attempt wrote. under_large_03 (cap 650 s) 8 / 6,939 s; articulation_open_small_02 (cap 341 s)
3 / 6,528 s; under_large_01 3 / 3,445 s; four aliases 1 each. With every round-4 run as the median pool the cap would cut 172 attempts, 20
written (12 in inside_large_03, whose camera-hull / blame runs are slower), so the table uses the primary set.

### Runs

trunk_1a68082b, v2maps_int2, v2only N 4, alias memory off, base index 347000 (one ledger per arm), svl titanrtx, at most 4 at a time.
5x.1: sm_off / sm_a / sm_ab on under_large_03 and ontop_large_02. 5x.3: cap_on on under_large_03 (off arm = sm_off), cap_off / cap_on on
articulation_open_small_02 and inside_large_01 (control). Suite at 1a68082b: 23 F + 3 E, the base set at db50e839.

### Results (attempts / written / ok, GPU-h)

| alias | off | (a) memo | (a)+(b) memo + redraw | where the switches acted |
|---|---|---|---|---|
| under_large_03 | 25 / 4 / 0, 2.12 | 39 / 2 / 0, 2.67 | 25 / 4 / 0, 1.88 | 0 / 2 / 0 search failures; 0 memo skips; no redraw |
| ontop_large_02 | 24 / 3 / 2, 2.61 | 11 / 4 / 4, 1.43 | 11 / 4 / 4, 1.37 | 5 / 1 / 1 search failures; 5 B skips, 0 C skips, 0 re-checks; no redraw |

- With the start draw on, the repeated B / C search that the audit measured is mostly gone. under_large_03's off arm had no search failure
  at all (9 of 21 on i7diag_floor).
- The memo acted only on ontop_large_02, in 5 B draws. A (B, C) key never repeats, because B is drawn at random from 256. The ontop
  difference comes from the off arm's instance 347002: 15 attempts, all post-sample failures. That is N 4 noise, not the memo.
- No re-check fell due, because no instance had 4 hits: `memo.would_have_passed` is unmeasured.
- The redraw never fired, so the (a) and (b) arms are identical on ontop_large_02. In the off arm one instance reached 2 search
  failures, and its 3rd attempt failed the search too.
- no_floor_cell 0 and time_budget 0 in every arm.
- For i7-startdraw: the per-attempt stance A draw fails `no_stance` in under 1 s on most under_large_03 attempts (17-24 of 25-39). These
  rejections use up the 15 attempts per instance.
- Bug fixed in 42392e51: a B failing "target not in the init image" depends on stance A's camera. It is now keyed with A.

| alias (cap) | cap off | cap on | cut |
|---|---|---|---|
| articulation_open_small_02 (341 s) | 14 / 3 / 1, 2.74 GPU-h, sampling 8,279 s | 24 / 4 / 2, 1.49 GPU-h (0.75 / success), 4,907 s | 8 (time_cap) |
| under_large_03 (650 s) | 25 / 4 / 0, 2.12, 7,080 s | 30 / 4 / 0, 0.98, 2,992 s | 3 |
| inside_large_01, control (666 s) | 5 / 4 / 2, 0.59 | 10 / 4 / 3, 0.89 (0.30 / success in both) | 1 |

Every instance was written in both arms of every alias.

**5x.4**, re-measured on the (a)+(b) arm:
- ontop_large_02 approach_untrackable: 2 of 11 attempts (off: 8 of 24), 364 s per written demo. In both, the carry chain at C goes into
  another IK branch.
- under_large_03 gripper_not_in_frame: 3 of 25 attempts (the same 3 as off), 549 s per written demo. Two of them are the pre-place
  fingertip at the right image edge; one of them spent 2,022 s, mostly in the search.
- Both still warrant an early check, at the point where the search can move on to the next C. Proposed, not implemented:
  - ontop_large_02: an A/B of the existing opt-in `generation.carry_track_oracle`;
  - under_large_03: the 5 % projection of the pre-place fingertip at C's acceptance, and of the grasp eef at B's.


### r2: the cap default, and the start draw's stance A redrawn within the attempt (lead decisions #33-36, 2026-11-08)

- #33: `ISPATIALGYM_ATTEMPT_CAP` is the default (69d70a41, in its own commit); `=0` = no cap.
- #34: `ISPATIALGYM_STANCE_MEMO` / `ISPATIALGYM_FLOOR_REDRAW` stay off, in the code.
- #35: 5x.4 is deferred to the owner.
- #36: the start draw's `no_stance` (a bug of the i7-startdraw default, merged here at 1ba40aae).

**Diagnosis.**
- `run_instances` charges every start-draw rejection (0.3-0.9 s) as one of the instance's 15 attempts. sm_a lost two under_large_03
  instances this way: each ran out of its 15 attempts after 8-9 `no_stance`.
- Why stance A fails: new per-bearing counters (r2_sd1, under_large_03) split the failed bearings as follows:
  - 78 % off the eroded free floor or the task rooms;
  - 19 % with no straight trunk seeing the target and the partner region;
  - 1 % colliding.
- The 0.6-1.5 m ring around the template's c_t (beside a cubicle desk) is mostly not free floor, so one draw of 30 bearings finds A with
  p ~ 1/3.

**Fix (b2a51679).**
- `ISPATIALGYM_START_DRAW_A_REDRAWS` (default 8; `0` = the single draw): up to 8 more draws of stance A and its A -> B leg inside the
  attempt's start draw.
- Kit counters `start_draw.redraws`, `start_draw.redraw.<reason>`, `start_draw.no_stance_final`, `stance_a.reject.<floor|frame|body|sweep>`.

**A/B** (trunk_b2a51679, cap on in both arms, v2maps_int2, v2only N 4, svl titanrtx):

| alias | single draw (r2_sd1) | + 8 redraws (r2_sd9) | counters |
|---|---|---|---|
| under_large_03 | 30 / 4 / 1, 0.99 GPU-h, 19 no_stance attempts | 17 / 4 / 0, 1.30 GPU-h, 1 no_stance | 27 redraws (24 no_stance, 3 no_transition), 1 no_stance_final |
| ontop_large_02 (control) | 26 / 3 / 3, 2.10 GPU-h, 5 start rejections, 1 instance lost | 11 / 4 / 4, 0.96 GPU-h, 0 | 1 redraw |

- Start-draw attempts charged: 24 -> 1. Instances lost: 1 -> 0.
- GPU-h per success over both aliases: 0.77 -> 0.57, inside the ~15-point N 4 noise. under_large_03's ok count is held back by the
  oracle push.

## articulation_open / close_large_03: the init (start) stance from the whole Phase-1 set (branch `i7-artlarge`, lead #44, 2026-11-08)

The two cabinet aliases (bottom_cabinet_rvunhj_0, door j_link_2, bar handle j_link_2#C, in the corridor between two cubicle rows of
office_cubicles_right; left arm, hybrid / sequential follow) wrote 0 instances in every fast run (survey1, fix2, i_nopol, i_legwv,
v2only_*, i7_pol: 60 / 0 / 0 open, 60 / 0 / 0 close; one close instance in fix2).

**Diagnosis (CPU, from the i7_pol / fix2 / survey1 logs).**
- `repeats_eval` was EVERY attempt that passed the sample (open 31 / 31, close 38 / 38 in i7_pol). The goals cannot separate an instance
  from the eval pair: open draws q_goal from S = [0.283, 0.393] and the eval goals are 0.388 / 0.280 (tolerance 0.05); close ends at
  lower + 0.03 (eval 0.043 / 0.034); the articulation record measures no start state. So only the start base can, and Phase 1
  (`prompt0.phase1_init_pose`) takes the NEAREST feasible d on a random bearing: every passing start sat 0.02-0.19 m and < 15 deg from the
  eval starts (9.53, -6.58, 176 deg) / (9.57, -6.66, 172 deg), which were drawn the same way.
- `handle_unreachable` is the attempt-level label of the stance loop. Pinch -1 never has an exact IK (25 / 15); pinch +1 failed at
  step 6, the init -> manipulation path (open: path_collision 26, no_transition 4; close: path_collision 16), plus
  hybrid_follow_infeasible 7 / 11. The path collisions were at the init itself (the far starts at x 10.62-10.66: the nearest-first d lands
  right behind an obstacle edge) or a few strides out. Phase 2 (1 survivor per bar offset), IK limits and the camera are not the cause.

**Fix 23733307, `ISPATIALGYM_ARTLARGE_INIT_DRAW` (default off in that commit, on since d867724e).**
- The init is drawn from the whole Phase-1 feasible set: 5 deg bearings in +-beta_max x the d scan, each judged by Phase 1's own checks
  in its order (free floor in the task rooms, the whole target in frame, body collision-free, trunk sweep), memoized per q_init.
  The camera-frame rule is unchanged.
- Starts within 0.25 m / 15 deg of an eval start go last (the repeat rule's start half); the rest in random order.
- Step 6 tries up to 12 further candidates of that order when the drawn start has no path / transition (the start enters neither
  steps 1-5 nor the follow); the sidecar's init is the one that connects.
- Kit counters `artlarge_init.{drawn, drawn_away, drawn_near_eval, empty_set, connect_first, alt_tried, alt_connected,
  alt_exhausted}`; one "ARTLARGE_INIT_DRAW: init set ..." line per q_init with the per-stage counts.
- Measured set (al_on logs): open 110 feasible of 697 lattice cells (24 near an eval start), x 9.52-11.18 along the corridor; close
  98-100 (19-21 near).

**A/B** (trunk_23733307, N 4, svl titanrtx, base index 352000 / control 352100 in both arms, v2only on v2maps_int2,
`ISPATIALGYM_PATH_BLAME=1` in both arms, log only; tags al_off / al_on; GPU-h = sacct elapsed):

| alias | off (`=0`) att / wr / ok, GPU-h | on (`=1`) att / wr / ok, GPU-h | where the switch acted |
|---|---|---|---|
| open_large_03 | 60 / 0 / 0, 1.97 (repeats_eval 23, handle_unreachable 35) | 12 / 4 / 2, 0.70 (handle_unreachable 8) | drawn 12 (all away from the eval starts), alt_tried 164, alt_connected 4 (2 written), alt_exhausted 11 |
| close_large_03 | 45 / 2 / 0, 1.26 (repeats_eval 25, time_cap 9, no_manipulation_stance 5, no_stance 4; demos: collision 2) | 4 / 4 / 4, 0.18 | drawn 4 (away), connect_first 4 |
| close_large_02 (control, dishwasher) | 12 / 4 / 4, 0.22 (repeats_eval 8) | 7 / 4 / 4, 0.28 (program_check 1, ik_joint_limit 1, handle_unreachable 1) | drawn 7 (away), connect_first 6 |

- The integration tip's own run int2_10 (open, l40s, switch absent) repeated the old result: 60 / 0 / 0 (repeats_eval 31,
  handle_unreachable 27).
- Written starts with the switch: open 0.4-0.5 m from the eval starts (x 9.86-10.01), close 0.5-1.4 m (d 1.35-2.3). The control's
  repeats_eval went 8 -> 0; its ok count is unchanged.
- **DEFAULT ON** in d867724e (own commit; lead #44 under the owner's mandate). `=0` restores the nearest-d draw. The path is reached by
  every ArticulationLarge variant (door / dishwasher / bottom_cabinet open and close, bottom_cabinet_push): articulation_open / close
  _large_01..03. The full suite has the base failure set (23 F + 3 E in the lambda env) at 23733307 and d867724e, +7 passes.

**Still open.**
- open_large_03 demos: 2 of 4 fail `robot_touching_itself` (left_arm_link3 touches left_arm_link5) in the grasp, after the
  STRAIGHT_ARM ramp to the recorded witness. Both have a near-upright manipulation tilt (-0.007 / -0.010); the two successes have
  -0.29 / -0.08, and the cabgeo eval demos -0.08..-0.39. The step-5 grasp witness has no self-clearance (the follow's IK keeps 20 mm).
  A fix there is a shared gate (owner).
- open's remaining step-6 rejections are mostly no_transition (the trunk sweep at s_T with the arriving arms), which does not depend
  on the start: the alternatives cannot help and cost tries (164). A refinement would skip them when the drive-in or sweep fails.
- hybrid_follow_infeasible (sequential follow) is unchanged.


### i7-artlarge r2 (lead #51): the open_large_03 self-contact, and the no_transition retries

**The self-clearance margin on the grasp IK was not built: the witnesses refute it.**
- The executor reports `left_arm_link3 touches left_arm_link5`. Measured on the simulator's collision meshes (`pair_mesh.pair_clearance`,
  the r1pro.usda convex pieces + URDF FK, CPU) at the al_on witnesses, link3-link5 clearance at pre_grasp / approach / grasp:
  - the two FAILED demos (352001 / 352003): 22.7 / 23.2 / 23.2 mm;
  - the two SUCCESSFUL demos (352000 / 352002): 17.0 / 22.8 mm and 20.0 / 22.9 mm.
  - No margin refuses the failures without refusing the successes first; 20 mm refuses exactly the two successes.
- Cross-alias count (CPU, every int2_* written demo with a grasp witness, 144 demos, link3-link5 of the grasping arm, the minimum over its
  pre_grasp / approach / grasp witnesses): a 20 mm margin refuses 5 demos, 4 of them SUCCESSFUL (nextto_04 8.8 mm, touching_04 11.3,
  under_small_03 17.8, disembedding_02 19.3) and 1 failed for another reason (under_large_01, not_reached, 13.0 mm). 10 mm refuses 1
  success; 5 mm none. The one int2 robot_touching_itself failure (inside_large_01) reads 52.9 mm at its witness.
- Where the contact happens: the two failures' executed arm (demo.hdf5) leaves untucked at the manipulation stance and the link3-link5
  clearance falls 21.7 -> -0.2 mm over ~70 steps on the way to the pre-grasp. The commanded arm path is not the joint-linear ramp the
  log reports (in the successes it is exactly linear; in the failures j4 is commanded to -2.09 rad against -1.53 at the witness). The hand
  descends 0.94 -> 0.89 m where the plan descends to 0.67 m.
- A re-demo of the same four instances with `ISPATIALGYM_FAST_STRAIGHT_ARM=0` (trajopt; recollect al_rc_sa0) fails the same two the same
  way. al_rc_sa1 (defaults) reproduces 2 / 4. So neither the straight-ramp shortcut nor the witness is the cause.
- Upright trunk alone is not the cause either: cabff_open_a 0001 (tilt 0.0) and cabff_open_b 0001 (-0.016) succeeded on the 11-04
  oracle.
- Open: why the executed / commanded transit departs from the planned path in these two. Next step: a replay with per-step contact
  logging. This is an oracle question for the owner.

**Speed fix b20703b9, `ISPATIALGYM_ARTLARGE_ALT_SKIP` (default on since 397048b2).**
- `prompt0.connect` raises `no_transition` only when every transition stand-off fails the drive-back floor test, the trunk sweep or
  the drive-in. All three are checked at the manipulation stance with the arriving arms. The start enters only through `same_trunk`.
- So after a `no_transition` from a start whose trunk differs from the manipulation stance's, step 6's alternatives skip every other
  such start. Counter `artlarge_init.alt_skipped_no_transition`.

**A/B** (trunk_b20703b9, N 4, svl titanrtx, same seeds in both arms; open 352200, control 352300):

| alias | skip off (al2_skip0) | skip on (al2_skip1) |
|---|---|---|
| open_large_03 | 9 / 4 / 3, 1122 s (attempts 675 s); alt_tried 94 | 9 / 4 / 3, 1043 s (attempts 618 s); alt_tried 82, skipped 12 |
| close_large_02 (control) | 5 / 4 / 4, 601 s | 5 / 4 / 4, 601 s (not reached) |

- The instances and demos are identical. The skip only removes connect tries; it is the default since 397048b2. The suite has the base
  failure set at b20703b9 and 397048b2.
- open_large_03 on the int2 defaults: 9 / 4 / 3 (one robot_touching_itself), against 60 / 0 / 0 before the init draw.

### i7-artlarge r3 (lead #54): the open_large_03 self-contact, found

**Replay** (log-only `ISPATIALGYM_DIAG_ARM_TRACE=1`, 867681d8; re-demo al_rc_trace of the four al_on instances; same 2 / 4).
- The witness transit's straight ramp is planned (128 waypoints, the log line the r2 analysis read).
- The ramp's hand path dips 16-17 cm below its lower end, so the TRANSIT DIP GATE (first-demo v3.0 F5, `Motions._dip_gate`, inside every
  grasp transit) replaces it with its straight Cartesian leg. The leg's differential-IK path (175 waypoints) folds the elbow (j4 to
  -2.09) and the link3 / link5 meshes touch: -0.2 / +0.1 mm.
- cuRobo's sphere self-check passes the leg. Commanded and measured agree (no controller lag, no contact before the self-touch).
- The gate logged its decision only AFTER the leg streamed, so a failure inside the leg left the log reading as if the plan's straight
  ramp had run: the "plan is not what executes" of r2.
- In the two successes the gate found no qualifying leg and the straight ramp ran.

**Fix 4a9e5038 (shared execution path, behind a default-OFF switch).**
- `ISPATIALGYM_DIP_LEG_SELF_MESH=1`: a Cartesian dip leg whose `{arm}_arm_link3` / `link5` mesh clearance (pair_mesh) falls to 5 mm is
  refused (the plan then streams as before). Executor stat `dip_leg_self_mesh_refused`.
- Unconditional, log-only: the gate logs "streaming the ... Cartesian leg" before the leg runs.
- The suite has the base failure set.

**Count** (int2_* demos' meta `transit_dip`, CPU).
- 8 demos ran a Cartesian dip leg: 6 succeeded (close_large_02 x2, inside_01, nextto_03 x2, ontop_large_03) and 2 failed
  robot_touching_itself (open_large_03 352001 / 352003, int2_11).
- The executed link3-link5 minimum over the grasp action: failures -0.2 / +0.1 mm; successes 10.9-21.7 mm. A 5 mm margin refuses
  exactly the two failures.

**Validation re-demo** (trunk_4a9e5038, switch on, same instances):

| alias | before | switch on | what the switch did |
|---|---|---|---|
| open_large_03 | 2 / 4 | 4 / 4 | legs refused at +3.5 / +3.6 mm |
| close_large_02 | 4 / 4 | 4 / 4 | legs streamed |
| nextto_03 | 4 / 4 | 4 / 4 | legs streamed |

- DEFAULT stays OFF: the path is shared by every grasp transit (lead #54: stop for the count).
- **DEFAULT ON in c421a6c0, for all aliases** (lead decision #57, under the owner's mandate, on the count and the re-demo above); `=0`
  restores. Snapshot trunk_c421a6c0. Fresh confirmation: open_large_03 N 4 on the defaults, no env, tag al3_def (base index 352400).
  Next validation: inside_01, nextto_03 and ontop_large_03, the other aliases whose demos ran a dip leg.

## Floor pushes under the desk: under_large_02 / 03 (branch `i7-floorpush`, 2026-11-08; owner decision item 5)

Owner decision 2026-11-08 (verbatim in the lead's scratchpad `owner_20261108f.md`, item 5; DECISIONS "OWNER DECISIONS on
open_decisions_20261108b"): diagnose the pitched floor push first (CPU, from the existing demos), then the approved oracle push-program
variants (a) open gripper, (b) both grippers, (c) strokes of at most 0.05 / 0.10 m with re-aim, (d) a lower contact height - each behind
its own switch, limited to these floor-push aliases, an N 4 v2only A/B per variant vs today on the same indices and backend; defaults
after the owner sees it. Record: `$OFH/DECISIONS.md` "(i7-floorpush)" lines, `$OFH/floorpush_20261108/`.

### Diagnosis (the 15 demos with a push: int1_1, i7_floor, i7_floorfix, i7diag_floor)

- **The hand**: `push_under_hand` pitched_far, pitch 60, wrist turn "lateral" (closing axis across the push, the two closed fingers side by
  side, ~17 mm wide), finger joints 0 through the push, fingertips 0.080 m up, contact (band top) 0.086 m. Centre of mass 0.114 m (the
  briefcase lwbctq: 0.27 m tall on its 0.12 x 0.33 m edge, pushed along its long axis onto the 0.12 m end face) / 0.139 m (the round
  ashcan jlawet): the contact is already below the centre of mass.
- **Push line vs centre of mass**: at the first object motion the hand is -10.5..+4.0 mm (briefcase) / +0.3..+6.9 mm (can) lateral of
  the object's centre (the record has a `mid`, so the oracle's `push_recentre` never runs).
- **Where the rotation starts**: at contact, in the first 1-2 cm, growing with the travel (an unstable point push). Briefcase failures:
  yaw 2-9 deg after 2 cm, 4-11 after 5 cm, 8-19 after 10 cm (the ok demo 0.6 / 1.2 / 2.1); the hand slides along the end face to its corner
  and passes the briefcase (the hand ends 0.115-0.141 m farther along), the residual 0.108-0.132 m exceeds the re-push cap (0.10 m): the 5
  not_reached. One briefcase toppled at the corner. Ashcan: the position is reached (<= 3 mm) but the yaw grows to 19-41 deg (ok 6):
  symbolic_only under `pose_reached`'s full orientation (0.15 rad) - while `placed_orientation` already treats the push-under yaw as free
  and the can is round (an owner question; not changed).
- **protected_contact** (under_03, 2 of 4 on the integration tip): the transit from the drop's retreat round the can to the push's
  pre-contact, planned with the can left out (its witness within 10 mm of the can's box), via-above failed, `right_arm_link7` touches the
  can at its side. Not the push; none of (a)-(d) changes it.
- **(b) kinematic pre-check** (CPU, URDF FK with joint limits, no collisions; 13 right-arm stances C): the other hand's pitched pose
  0.05-0.12 m beside the push line is missed by 0.27-0.54 m - the push lane sits 0.25-0.31 m to the push arm's side.

### The switches (b46be8a3; default off, acting only on the pitched hand of the under relation)

| switch | variant | what it changes | fall-back | counters / record |
|---|---|---|---|---|
| `ISPATIALGYM_FLOORPUSH_OPEN=1` | (a) | the lateral turns with the gripper open: both fingertips across the back (`pitched_open_contact`: gap 60 % of the band's width, <= 8 cm); IK, pinned check, goal check, program check and oracle at that opening (`push_contact_form` two_point, `fingers_opening_m`); one stroke | today's closed chain | `floorpush.open.chain / unsolved`; `contacts.push.floorpush.variant` |
| `ISPATIALGYM_FLOORPUSH_BOTH=1` | (b) | probes the other arm's chain 0.06 / 0.09 / 0.12 m beside (exact IK from stance C); no two-hand oracle push is built | always one hand | `floorpush.both.probe / feasible / infeasible`; `floorpush.both` |
| `ISPATIALGYM_FLOORPUSH_STROKE_M=<m>` | (c) | the existing strokes (`execute._push_strokes`) for the pitched hand: hold, measure, back off 3 cm, re-aim <= 3 cm at the root line + 0.15 m/rad x yaw beyond 2 deg (`ISPATIALGYM_FLOORPUSH_YAW_GAIN`), <= 8 strokes (`_STROKES_MAX`); the hand's `face_half_m` / `contact_strips_m` | - | `floorpush.strokes.recorded`; `strokes` |
| `ISPATIALGYM_FLOORPUSH_TIP_M=<m>` | (d) | the fingertips this high (0.05: contact ~0.056 m) | today's 0.08 m | `floorpush.low_tip.chain / unsolved` |

### A/B (fp_today / fp_open / fp_s05 / fp_s10 / fp_tip / fp_both; trunk_b46be8a3, N 4 v2only, index 350000, svl titanrtx)

| arm | under_02 att / wr / ok, GPU-h/ok | under_03 att / wr / ok, GPU-h/ok | pushed demos' yaw (median) |
|---|---|---|---|
| today | 9 / 4 / 1, 5.32 | 28 / 3 / 0, - | 9.3 / 18.4 deg |
| (c) s05 | 9 / 4 / 0, - | 28 / 3 / 2, 1.67 | 3.9 / 2.8 |
| (c) s10 | 9 / 4 / 1, 5.18 | 28 / 3 / 2, 1.57 | 3.0 / 2.7 |
| (a) open (55d28864) | 12 / 4 / 0, - | 20 / 1 / 0, - | 6.1 / - |
| (d) tip 0.05 | 17 / 4 / 0 (low chain 48 / 1107 solved, never written) | 26 / 4 / 1, 6.11 (0 / 784) | = today |
| (b) both | 11 / 4 / 0 (probe 0 feasible / 1166) | 26 / 4 / 1, 2.73 (0 / 516) | = today |

**Default (82c96a3b, lead decision #41)**: the pitched push in 0.10 m strokes (`ISPATIALGYM_FLOORPUSH_STROKE_M=off` restores one stroke);
(a), (b), (d) stay off. On the default (fp_def, index 360000): under_01 11 / 4 / 3, 0.95 GPU-h / ok (the blade, not affected); under_02
12 / 4 / 1, 4.50; under_03 11 / 4 / 3, 0.18 (every push ok). Remaining under_02 failure: the briefcase topples (tilt 84-93 deg) once its yaw
reaches 8-22 deg within a stroke and the hand meets the end face's corner; next fix (decision needed): a lying rest or a broadside push.

## Slow / low-yield aliases: the fork's lifted grasp and the bare-wall init view (branch `i7-yield2`, lead decision #47, 2026-11-08)

Owner mandate: "Fix all the bugs and speed up / make yield better the slow low yield aliases". Two items from int2_7 (trunk_2f157bbf).

### A. disembedding_10: grasp top_2 [pause] - the executor's support lift took the pads off the fork

- Symptom (int2_7 300019 / 300020, drawn starts at base x 6.77 / 6.98 picked top_2): both demos `oracle_failed [pause]` in the grasp,
  "no joint moved faster than 0.001 for 51 steps". 300018 (top_9) succeeded.
- What the demo shows (hdf5): the arm reaches the grasp pose (eef 1.3 mm off); the ramped closure goes 15.5 -> 6.6 mm per finger
  symmetrically, then finger 2 drives to 0 and shoves the fork 4.4-6.3 mm across the pinch (it turns 3.6-6 deg), pushing finger 1 back out
  to 11.1-11.3 mm; no `contact_hold`, no assisted grasp. `Motions.gripper` then waits 60 steps + holds 5 with nothing moving - longer than the
  pause limit (50), so the hard QA `pause` ends the demo before the executor's `failed_grasp` (whose retry runs the recorded, unlifted grasp).
- Cause: not the sampler (the IK margins are fine) and not the entry (validated with zero finger-slab force): the executor's
  `lift_grasp_off_support`. The finger envelope reaches 1.8 mm below the fork's bottom, so the grasp is lifted 5.8 mm (+ 4 mm clearance).
  The fork's collision hulls at top_2's x (14 mm from its origin) span z 0.5-6.0 mm; the lifted pads start at 9.2 mm - **above the neck**.
  The pad-margin rule read the AABB top 0.840 (the raised handle end), the face-slope probe at the lifted fingertip end missed (None =
  "not judged"). top_9 the same: every lifted top_9 attempt (i7tray_turn_on x4, int2_7 300018) closed on nothing (`failed_grasp`) and
  succeeded on the unlifted retry.
- Survey of all 344 recorded lifts under `$OFH/out` (scratch survey_lib.py, the fork / spatula / steak / tape / dice / mouse / jigsaw /
  spoon hulls from `v2maps_int2/glib`): the library's own pad sweep (`candidates.pad_sweep`, rays across the pinch over PAD_Z x PAD_X) at the
  lifted pose meets no face only for the fork (top_2 x2 pause, top_9 x5, top_10 x1, top_11 x1 - all failed lifted) and the spatula of
  under_small_04 (top_2 x1, top_3 x2 - succeeded lifted); the other 332 lifts are unchanged.
- Fix d5ce72ef, `ISPATIALGYM_LIFT_PAD_SWEEP=1` (`ExecuteConfig.grasp_support_pad_sweep`): before a lift is applied, 24 rays (PAD_Z x PAD_X x
  both sides, from the open pad faces across the pinch, `_pad_sweep_hits`) at the lifted pose; none on the object -> the lift is refused,
  the recorded grasp runs. Record: `grasp_support_lift.pad_sweep {hits, rays}`. No scene queries (stubs) -> not judged.

### B. world_vertical_04: too_few_visible at the init view

- Symptom (int2_7 = i7_wall_def, deterministic): 51 att / 2 written / 2 ok; 49 rejections `[too_few_visible]` in settle_validate.
- The check (`AlignmentWall.extra_checks`): >= `min_visible_objects` 5 objects (structure counted, the robot not) with >= 200 px in the
  INIT camera view. 45 of 49: only floor + wall + target; 3: + another wall; 1: no floor. The drawn (spot, delta, init stance at d 0.5-1.5 m,
  |beta| <= 35 deg) mostly faces a bare stretch of wall gkewle; the passing draws (spots 34, 2) see a door / an elevator door / other walls.
  Not the thresholds and not occlusion. world_vertical_05: 21 / 33 the same.
- Fix c0884d5f, `ISPATIALGYM_WALL_VIS_SCREEN=1` (`_DRAWS`, default 40): each draw is screened at once with the settle check's own rule
  (`visible_at`: the robot teleported to the init stance as `RobotKit.apply` places it, one segmentation, put back) and redrawn within the
  attempt (spot, delta and init together - the accepted distribution is the conditional one the settle check already selected; the first
  draw equals the switch-off draw); the settle check stays. Kit: `vis_screen.passed / redrawn / draws_k / exhausted / settle_rejected`.

### Runs (trunk_c0884d5f, svl titanrtx, v2maps_int2, v2only N 4, memory off; loop `$OFH/yield2_20261108/loop.sh`)

| arm | job | att / written / ok | GPU-h | per success | the switch's counters |
|---|---|---|---|---|---|
| A recollect OFF (int2_7 disembedding_10 g300018-300021, the same 3 instances) | 18391303 | - / 3 / 2 | 0.17 | - | 300018 top_9 failed_grasp -> retry ok; 300019 top_2 [pause]; 300020 top_2 failed_grasp -> retry ok |
| A recollect ON `LIFT_PAD_SWEEP=1` | 18391304 | - / 3 / 3 | 0.14 | - | pad_sweep hits 0 / 24 at all 3 lifts -> refused; every grasp first attempt, contact_hold |
| B world_vertical_04 OFF 310000 | 18391305 | 51 / 2 / 2 | 0.55 | 0.277 | too_few_visible 49 (= int2_7, i7_wall_def) |
| B world_vertical_04 ON `WALL_VIS_SCREEN=1` | 18391307 | 5 / 4 / 4 | 0.32 | 0.080 | passed 5, redrawn 29, settle_rejected 0, exhausted 0 |
| B world_vertical_05 OFF 310000 | 18391503 | 33 / 4 / 4 | 0.36 | 0.089 | too_few_visible 21 |
| B world_vertical_05 ON | 18391399 | 10 / 4 / 4 | 0.28 | 0.071 | passed 10, redrawn 20, settle_rejected 3 |
| B control world_vertical_01 ON | 18391524 | 8 / 4 / 4 | 0.61 | 0.153 | passed 5 at draw 1 (= the off draws); 2 time_cap = the trunk's ATTEMPT_CAP |
| A control under_small_04 ON 300122 (spatula) | 18391613 | 8 / 4 / 4 | 0.38 | 0.096 | sweep ran on every lift, hits 2 / 24, no refusal (as predicted) |

- B: `ISPATIALGYM_WALL_VIS_SCREEN` DEFAULT ON (73e626b2, own commit; reach = world_vertical_01-05, the only subtask AlignmentWall registers).
- The disembedding_10 N 4 arms on this trunk first ran with the ATTEMPT_CAP default: every START_DRAW attempt ended [time_cap] (cap 338 s =
  5 x a median 67.6 s from before the start draw; the drawn start costs ~390 s per attempt) - 0 written in 4 attempts per arm, cancelled; the
  A/B reruns with `ISPATIALGYM_ATTEMPT_CAP=0` in both arms (int2_7's conditions). The cap x start-draw interaction belongs to #45 (i7-stancememo).

| A N 4 OFF disembedding_10 300018, `ATTEMPT_CAP=0` both arms | 18391614 | 22 / 1 / 1 | 2.49 | 2.49 | 300018 top_9 failed_grasp (lifted) -> unlifted retry ok |
| A N 4 ON `LIFT_PAD_SWEEP=1` | 18391653 | 28 / 3 / 3 | 2.49 | 0.83 | refused 3 / 3 (hits 0 / 24): top_9, top_9~r180, top_2 - all first attempt ok |

- The A arms sample identically (same rejections in the same order; the switch is executor-only); ON wrote 3 because its node ran the
  same attempts faster (28 vs 22 in the 8930 s sample limit) - the sampling difference is backend speed, not the switch. Judged on the
  switch's counter: it acted in 6 / 6 fork demos (recollect + N 4), every one a first-attempt success; OFF: 4 lifted fork demos = 1 [pause]
  + 3 failed_grasp retries. 300019 diverged from int2_7 (goal_check place_unreachable at attempt 2 on the newer trunk), so top_2 came up once.
- A: `ISPATIALGYM_LIFT_PAD_SWEEP` DEFAULT ON (13427662, own commit; reach: the fork and the spatula's top_2 / top_3 by the survey).
- Still open: disembedding_10's sampling cost (START_DRAW ~390 s per attempt, 2.5 GPU-h for 1-3 instances) and the ATTEMPT_CAP x START_DRAW
  interaction (every drawn-start attempt [time_cap] with both defaults on) belong to #45 (i7-stancememo); the pause-before-failed_grasp
  ordering in `Motions.gripper` (60 + 5 still steps > the QA pause limit 50) is shared QA / executor behaviour, left as it is (owner's call).

### r2: the gripper close wait vs the pause limit (lead decision #65, `ISPATIALGYM_GRIP_WAIT_NO_PAUSE`)

- Two shared constants disagreed: `Motions.gripper` waits up to `gripper_steps` 60 (+ a 5-step hold) after a closing command (and up to 60
  in the contact-hold wait), the QA pause rule fails a demo after 50 steps without motion. A closure that grips nothing therefore ended in a
  hard [pause] - no retry - instead of the executor's `failed_grasp`, whose retry runs the recorded / unlifted grasp; whether it paused
  depended only on whether any joint moved 3e-5 rad in the wait.
- Count (CPU, scratch `classify_pause.py` on the demo hdf5: the gripper command reached its closing value <= 75 steps before the pause and
  the arm commands stood still since): int2_* 5 pauses in 288 demos - 3 in the close wait (disembedding_10 x2, disembedding_01 x1 in the
  contact-hold wait), 2 elsewhere (touching_04 pre-shape opening wait, under_large_01 push). All `$OFH/out`: 34 of 68 in the close wait
  (under_small_01 13, under_small_04 5, disembedding_07 4, disembedding_10 4, disembedding_01 3, attachment_02 2, attachment_03 / inside_03 /
  inside_04 1), 34 elsewhere (ontop_small_02 12, touching_04 8, touching_01 7, under_large_01 5, ...).
- Fix 46ed7e5b: the close waits are emitted as settle steps (`_emit(settling=True)`, the path the hold / settle waits already take), which
  neither the QA pause rule nor the monitors' stalled rule counts; the ramp, the opening waits and every other step count as before.
  Recorded as `close_wait_exempt_steps` on the closure's `contact_hold` release event (84404e6f), else a `close_wait_exempt {steps, held}`
  release event.

Runs (trunk_46ed7e5b = A / B defaults on, svl titanrtx):

| run | result | the switch |
|---|---|---|
| re-demo int2_7 disembedding_10 300019 / 300020 (`LIFT_PAD_SWEEP=0` to keep the lifted closure), 18393804 | 2 / 2 success | each lifted top_2 closure -> failed_grasp (no pause) -> unlifted retry ok |
| re-demo int2_7 disembedding_01 300021, 18393806 | 0 / 1 | pause -> failed_grasp, the retry failed_grasp too (no pause) |
| control re-demo int2_10 touching_04 320302 (pause was in the pre-shape opening wait), 18393807 | 0 / 1 plan_failed at place | grasp held; no pause |
| control re-demo int2_7 under_large_01 300022 (pause was in the push), 18393808 | 1 / 1 | the push stall did not recur; fist close wait exempt 12 steps |
| N 4 under_small_01 321000 OFF, 18393872 | 9 / 4 / 4, 0.19 GPU-h | - |
| N 4 under_small_01 321000 ON, 18393886 | 9 / 4 / 3, 0.21 | 3 contact-hold waits exempt (all held); the failure: robot_wobble 1.01 deg (> 1.00) at step 300 of the transit, gripper still open - not in a wait |
| N 4 control touching_01 300150 ON, 18393887 | 4 / 4 / 2, 0.27 | grasp waits held 4 / 4; failures at place (gripper_out_of_fov, plan_failed - both in its history), none in a wait |

- `ISPATIALGYM_GRIP_WAIT_NO_PAUSE` DEFAULT ON (1762edd4, own commit): every close-wait pause re-demoed became a retried failed_grasp (2 / 3 then
  succeeded); no failure in the N 4 / control runs falls inside an exempted wait. Handed to the integration agent.

### r3: the start draw's speed and a robust cap (lead decisions #45, #63, 2026-11-09)

**Profile** (cProfile, disembedding_10, trunk_7dbb853e): one attempt took 331 s, of which the start draw took 285 s.
- 220 s of that went to `SupportTop.polygon_supported` -> `contains`: 343k scalar-tensor calls, with `free` recomputed 2.8M times.
- The callers: the destination fit's 2 cm lateral support scan (190 s) and Phase 2's source-tray frame check (68 s).
- IK took about 29 s.
- The stance-A redraws of r2 run in the IORS large floor path only, so they cost disembedding nothing.

**Fix (da01d92d).** Both methods are vectorized with the same float operations.
- Their verdicts are bit-identical to the former loops: `test_support_top_vectorized` checks them against a verbatim copy.
- So the drawn starts and the candidates are unchanged, and no switch is needed. Each call is ~13x faster.

**#63 (a1b48793).** After two consecutive `time_cap` cuts, a job's cap doubles.
- `ISPATIALGYM_ATTEMPT_CAP_RAISE_AFTER` defaults to 2; `0` restores the old rule. The rule is on whenever the cap is.
- The per-alias medians were measured before the start draw: disembedding_10's 338 s cap cut every ~390 s drawn-start attempt.
- Count over the `int2_*` logs: no alias had >= 50 % of its attempts cut. The highest were under_large_03 (32 %) and
  articulation_open_small_02 (27 %); only disembedding_10 crossed 50 %, and only on the old code.

**A/B** (svl titanrtx, cap on; att / wr / ok, seconds per attempt, GPU-h):

| alias | old (trunk_7dbb853e) | new (da01d92d) |
|---|---|---|
| disembedding_10 | 25 / 0 / 0, 333 s, 16 time_cap, 2.49 | 43 / 2 / 2, 85 s, 0 time_cap, 1.13 (0.57 / ok) |
| disembedding_02 | 13 / 4 / 4, 226 s, 1.00 (0.25 / ok) | 13 / 4 / 4, 124 s, 0.58 (0.15 / ok) |
| ontop_large_02 (control) | 28 / 3 / 3, 528 s, 4.24 | 20 / 3 / 3, 524 s, 3.00 (noise) |

**Raise-rule A/B** (disembedding_10 at 300018, a1b48793):
- `=0`: 40 / 4 / 3, 0.99 GPU-h. Default: 40 / 4 / 3, 1.14 GPU-h.
- The attempts are identical: with the speed fix nothing is cut, so the rule never acted.

disembedding_10 now loses its attempts to `target_placement` (empty F at the drawn position), not to the start draw.

**Final A/B (root umb2, N 4, svl titanrtx):** OFF 60 att / 0 written / 0 ok, 5.07 GPU-h (carry legs failed 828); CENTRED 60 / 0 / 0, 5.88
(782; 96 centred carries); CENTRED + WIDE 46 / 1 / 1, 3.39 GPU-h per success (140; 354 wide paths, 331 centred carries). umb2s (L2) 60 / 0 / 0.

### Lead decision #42 (2026-11-08, owner away)

**(1) f9f0829f - CARRY_CENTRED + CARRY_WIDE_PATH on for touching_large_02 only.** `generation.carry_centred` / `carry_wide_path` true in its
customs entry; `iors_large._switch` reads the env var when it is set (so `=0` turns a switch off), else the alias's generation key, else
off. A test pins touching_large_02 as the only customs entry setting them; other IORS large aliases reach the code with the keys absent.
umb2 stays the umbrella root.

**(2) The B blocker, `B: [pre-grasp, grasp] exact IK`** (gm_t02_wide_on, 106 failure lines, first try's reason): approach_untrackable 67,
every one a self-collision of the robot's base_link with a gripper finger (2-4 mm) at every intermediate approach configuration - the tilted
top grasps top_0 / top_10 (27-33 deg) at the deep-crouch B trunks (0.95, -0.9 / -1.2) and the stretched (1.75 / 2.0, 0): the floor umbrella
close to the chassis, not the hall tree or the mirror; ik_joint_limit 33 (joint 2: reach at the floor); no solution 6. The floor map's
approach check has no base_link x finger term. Fix: the ladder's next round - L3v (the gym shoe's L3 levels + the accepted slab_half_m 0.05,
max_valid 12): 12 of 12 trials valid, tilts 14-35 deg incl. three low-tilt grasps (top_0 14, top_33 15, top_32 19); maps (floor r + l,
interior r / l) on trunk_f9f0829f; staged `$OFH/v2maps_goalmaps_umb3` (copy of umb2, the L2v entry in `_superseded_ladder_r2`) by
`umbrella_r3/chain3.sh`, which then runs touching_large_02 N 4 on the new defaults (gm_t02_r3, no env switch).

**(3) Result.** The carry switches act from the customs (centred / wide used from attempt 0).

| run (N 4, svl titanrtx) | root | att / written / ok | GPU-h per success | B exact-IK fails / att | goal passes |
|---|---|---|---|---|---|
| gm_t02_wide_on (switches by env, c2071b3e) | umb2 (6 grasps) | 46 / 1 / 1 | 3.39 | 2.4 | 2 |
| gm_t02_r3 (new defaults, f9f0829f) | umb3 (12 grasps) | 58 / 1 / 1 | 6.38 | 3.5 | 4 |

More grasps did not beat the B blocker: each extra tilted grasp adds failing (B, grasp) pairs (base_link x finger self-collisions
62 -> 135) and raises the cost per attempt (exact IK 1260 / attempt). Three of r3's four goal passes died later: the lift's
approach_untrackable (joint jump 0.22 rad at waypoint 0), the retreat's witness FK 7.8 mm > 2 mm (program check), an approach joint
jump 0.21 rad. umb3 is not adopted; umb2 stays the umbrella root. Next: the narrowest B-stance fix (a base_link x finger term in the floor
map's approach check, or no tilted top grasps at the deep-crouch trunks), not done here.

### r3: disembedding_10's target_placement after the start-draw speed fix (lead decision #66, `ISPATIALGYM_DISEMB_POS_REDRAWS`)

- After i7-stancememo's vectorized SupportTop (333 -> 85 s per attempt) disembedding_10 still wrote 2-4 of 4 in 40-43 attempts: 30-38
  target_placement rejections per job (r3_new / r3c_rule0 / r3c_rule2).
- Sub-reason (CPU, sample logs): ~90 % `[grasp_unreachable] empty feasible set F at the focus (8 yaws x 2 grasps)` - Phase 3's (yaw, grasp)
  screen at the target's start position, the 2 grasps being the fork's top_2 / top_9 left by GRASP_TURN_FILTER; place_unreachable /
  place_leg_collision / no_placement_stance a handful. Not the init -> manipulation leg, not the transit, not the place map.
- Dependence on the drawn start: per work stance the drawn starts pass 3-9 % (r3_new 5 / 163, r3c 10 / 140, int2_7 6 / 69) vs 11-22 % from the
  fixed start (i7tray_turn_on 4 / 36, turn_off 5 / 26, int1_1 5 / 23); per attempt the share lost is alike (75-88 % vs 71 %). The drawn start
  costs ~70 s of the ~85 s attempt (fixed: 25 s median), and the split cone (generation.phase2_split_cone, default on) draws ONE start position
  per attempt and tries only its 0-6 work stances - a position with an empty F throws the whole drawn start away.
- Fix 7642b66f / 183810f4: `ISPATIALGYM_DISEMB_POS_REDRAWS=N` (default 0) - when every work stance of the drawn position fails
  (target_placement), the position is redrawn without replacement from the drawn start's pool, up to N times, keeping the start; other
  rejections pass through; no split cone = unchanged. Kit `pos_redraw.redrawn / passed / passed_after_k / exhausted`. (The accepted start
  is then drawn uniformly rather than weighted by how many feasible positions it has.)

A/B (trunk_183810f4, svl titanrtx, v2maps_int2, v2only N 4, memory off; =8 for on):

| arm | job | att / written / ok | GPU-h | per success | counters |
|---|---|---|---|---|---|
| disembedding_10 300018 OFF | 18394779 | 40 / 4 / 3 | 0.92 | 0.307 | target_placement 30, no_stance 2, repeats_eval 2, place_unreachable 2; 1 demo plan_failed |
| disembedding_10 300018 ON | 18394780 | 7 / 4 / 4 | 0.46 | 0.116 | pos_redraw redrawn 19, passed 3 (after 1 / 2 / 5+), exhausted 1; target_placement 1 |
| control disembedding_02 349100 OFF | 18394782 | 13 / 4 / 4 | 0.45 | 0.113 | approach_untrackable 6, target_placement 3 |
| control disembedding_02 349100 ON | 18395101 | 5 / 4 / 4 | 0.39 | 0.098 | redrawn 6, passed 3; approach_untrackable 1 (the first on job 18394781 died at Kit startup on svl10, infra) |

- `ISPATIALGYM_DISEMB_POS_REDRAWS` DEFAULT 8 (f1d75450, own commit; reach: every disembedding alias with the split cone). Handed to the
  integration agent.

### r4: under_large_03's search, the chain memos, and the bounded cap raise (lead decisions #70, #70b, #70c, #74, 2026-11-09)

**Profile** (under_large_03 360000/0 on trunk_f1bb8e24, an attempt cut at 650 s):
- 594 s went to 86 push-under chain solves, 6.9 s each: `orient.choose` -> exact IK + `track_approach` -> cuRobo `solve_ik_path`, which
  makes 93k per-iteration FK calls.
- No scalar Python hot loop is left. The cost is the breadth of the push-chain search.

**gripper_not_in_frame does not depend on the drawn start.** It rejects 51 % of past-search attempts with fixed starts and 49 % with drawn
ones, so there is no framing screen at the draw.

**Chain memos** (both off by default):
- Exact-input memo (`ISPATIALGYM_PUSH_CHAIN_MEMO=1`): results are identical, but it hit once in 2,052 solves. It is not folded (#70b).
- Quantized memo (`ISPATIALGYM_CHAIN_MEMO` = off / shadow / on, key at 1e-4):
  - In the logs, solves whose result matched an earlier one to the mm were 46 % of sampling: an upper bound.
  - The r6 shadow run measured the real hit rate: 1.3 % (34 / 2,564), with one mismatch (another wrist turn).
  - The on arm: 47 / 3 / 1, 5.53 GPU-h per ok. Left off.
- Overlap with i7-ul02's `ISPATIALGYM_PUSH_CHAIN_LIVE_B`: 0 of the would-hits.

**Cap raise.**
- On the int2_18 / r4 logs, the attempts that ran past the un-raised cap were all on under_large_03: 25 rejected and 4 written, 0 ok.
- `ISPATIALGYM_ATTEMPT_CAP_RAISE_MAX` bounds the raise (cba474c4).
- r5 A/B (under_large_03 360000, per-arm empty caches):
  - unbounded: 44 / 3 / 1, 5.19 GPU-h per ok;
  - 1.5x: 39 / 3 / 1, 4.37 GPU-h per ok (the bound acted 3 times: cap 975 s);
  - control disembedding_10: 7 / 4 / 4.
- Default 1.5 since 4f14b3f7; `=0` = the unbounded doubling.

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
