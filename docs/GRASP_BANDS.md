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
  - Pick and yaw keys are recorded and reported only. A pick hook would be the same `reorder` call at the F draw.
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
