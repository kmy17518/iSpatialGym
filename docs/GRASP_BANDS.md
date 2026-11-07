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
- v2only N 4: 8 attempts, 4 written, 4 / 4 demos, sample wall 822 s, 0.063 GPU-h per success. Before: N 1 timed out at 2400 s
  (attempts 131-707 s); now attempts take 9-61 s. The headroom skip did not act in this run.

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
