# train-fast: the fast instance + demo flow (branch `train-fast`, 2026-10-31)

Goal: 10-20x the isg-1020 training set (54k-108k demos) in about one day on the ~88 TITAN RTX of the `svl` partition, i.e.
26-51 successful demos per GPU-hour against production's 4.6 (train85: 7,183 successes in ~1,551 GPU-hours, sampler then rendered
540 px demos one after the other on one GPU). Rendering is not in that budget: demos are collected physics-only (hdf5) and rendered
later by the kinematic replay (the Vast replay of isg-1020-b1k, or the vectorized `run_isg_vec --mode replay`).

Base: `merge/v3.0-reconciled+lab-vec-eval-next` at 43ad206a (contains all 21 snapshot commits that produced isg-1020) + the train85
spa2still patch (start per attempt, oracle still exit). Cluster worktree `/vision/u/minyeong/projects/ispatialgym-train-fast`, record
`/vision/u/minyeong/projects/isg_home/trainfast/` (DECISIONS.md, scripts/, out/<tag>/<key>/g<start>-<end>/), lambda mirror
`~/Desktop/projects/ispatialgym-train-fast`. The commits, the experiments and how to re-run them:
[TRAIN_FAST_README.md](TRAIN_FAST_README.md).

## Where production's time went (train85, 487 jobs)

| bucket | share | what it is |
|---|---|---|
| rejected sampler attempts | 79 % of sampler time (1,174 of 1,493 h) | 6 attempts per written instance, 109 s each |
| padded batched IK | ~75 % of the `sample` phase | a 64-slot x 12-seed cuRobo call costs ~10 s at the default 8,192-slot mesh store (the sphere-vs-mesh kernel runs over the store's capacity); `feasible_sets` makes 2 per arm per candidate, `screen` one per missed pose |
| start-per-attempt draw | 193 h | Phase 1c + split-cone Phase 2 + start region recomputed every attempt (19 s), nothing of it depends on the attempt |
| repeats_eval | 122 h | the train-vs-eval repeat test ran only after a full attempt and the write |
| demos | 349 s each | 561 action steps at 479 ms with three 1080 px cameras rendered per step, 129 settle steps, 25 s load, 11 s texture wait, trajopt 37 s per plan (1.17 per demo) |
| world load | ~340 s per job | one Kit start + scene load per job of 1-4 instances |

## The flow

One job = one (alias, scene, room) template on one GPU: the sampler (`multiply_isg --split train`) and `KCOLLECT` physics-only
collectors (`run_isg --queue ... --record hdf5`) run **concurrently**; collectors claim instances as they land (O_EXCL claim files),
each in its own Kit app-data copy. `scripts/job_fast.sh` (campaign root) does this; `submit_fast.py submit <key> <n> --flow fast
--code <worktree> --kcollect 2`. Jobs are long (n >= 12) so the 5-8 min world load and the per-template caches amortise.

Environment of the fast flow (`scripts/fast_env.sh`): existing opt-ins production kept off for bit-identity -
`ISPATIALGYM_SAMPLER_MESH_CACHE=auto`, `ISPATIALGYM_EXECUTOR_MESH_CACHE=room`, `ISPATIALGYM_SAMPLER_STEP1_ONCE=1`,
`ISPATIALGYM_SEEDED_SPHERE_FIT=cache`, `ISPATIALGYM_GOAL_OCCLUSION=off`, `ISPATIALGYM_HEAD_CAMERA_ONLY=1`, collectors
`ISPATIALGYM_PHYSICS_ONLY=1` - plus the branch's own switches below. A shared sampling cache (`ISPATIALGYM_SAMPLING_CACHE`) on NFS
holds the start-draw cells and IK witnesses across jobs of the same template.

## Switches of this branch (`ispatialgym/fastflags.py`)

All `ISPATIALGYM_FAST_<NAME>`, **on by default**, `=0` turns one off, `ISPATIALGYM_FAST=0` all of them (A side of an A/B on the
same code). Each logs once when first consulted. Off = the base code's behaviour.

| switch | where | what |
|---|---|---|
| EARLY_REPEAT_FILTER | sampling/v2/core.py, instance_separation.py | the train-vs-eval repeat test on the draft right after `sample()`, before settle / IK / write (the post-write check stays as a guard) |
| PARK_ONCE, PARK_SPAWNED | core.py, tasks/base.py | park cell cached per attempt loop; spawned task objects parked at every attempt start |
| SAMPLER_CAMERAS | sample_isg.py, sim_utils.py | camera-free sampler world for the families that use no pixels (iors, axial, attachment, block design, articulation); `visibility: pending` in the summary; pixel families keep their cameras |
| SPA_CACHE | v2/tasks/iors_small.py, v2/tabletop.py, cache.py | the start-per-attempt draw memoised per (edge, 2 cm lateral cell) in-process and on disk; the attempt draws edge order, lateral and side from its rng as before |
| BATCH_SIZE_FIT, SINGLE_RETRY_EXACT, LIMIT_RETRY_RETURN_ALL, REMEMBERED_WITH_SEED, BATCH_IK_SEEDS | oracle/curobo_planner.py, sampling/robot_utils.py, v2/ik.py | batched IK solver sized to the query (8/16/32/64) instead of always 64 padded slots; single 32-seed re-solves instead of padded batches for one pose; `return_all` solve picks the max-margin branch so `away_from_limits` rarely re-solves; remembered witnesses also accepted with a seed. SINGLE_RETRY_EXACT, LIMIT_RETRY_RETURN_ALL and REMEMBERED_WITH_SEED are **off** since 2026-11-01 (they change the recorded witness branch: robot stutters on touching_large_03) |
| SCENE_CROP (`full` / `rooms` / `region`, region radius 2.5 m) | sampling/scene_crop.py, sample_isg.py, sim_utils.py, v2/core.py | the sampler's world loads a cropped copy of the scene: `full` for camera relocalization and alignment-world_vertical, `rooms` for large-scale IORS / large articulation / object scaling / mental rotation, `region` (support + 2.5 m) for small IORS, small articulation, attachment, block design, alignment-axial and disembedding, `full` for any other family; the uncropped scene stays the template.json merge source; the crop is part of the world pool key (2026-11-01) |
| CROP_NEIGHBOUR_FLOORS | sampling/scene_crop.py | OmniGibson loads floors only with their room: the floors of the neighbour rooms are crop candidates and their rooms join the loaded rooms (without it bases beyond the task room's floor stood 1.5 cm low) (2026-11-01) |
| PARTIAL_SCENE | oracle/run_isg.py, recorder.py, multienv/replay.py | the collector of a cropped template builds the cropped world and writes the crop into the hdf5 header; `multienv.replay --scene full` replays the demo in the full template merged with its recorded objects (`auto`: full for a cropped demo) (2026-11-01) |
| EARLY_FRAME_GATE, SYNC_ONLY_OBJECTS, ATTACH_REUSE | v2/program_check.py | the fingertip-in-frame gate for every static target before any IK; cuRobo world syncs of only the moved objects |
| PHYSICS_ONLY_TRIM | oracle/run_isg.py, executor.py | physics-only collectors skip the texture wait / init image / reset re-renders, small camera buffers |
| STRAIGHT_ARM | oracle/motions.py | a straight joint-space ramp to the sidecar's IK witness, collision-checked in one batched call, before trajopt (trajopt stays the fallback) |
| PLAN_LADDER (TRAJOPT_SEEDS, PLAN_ATTEMPTS, WITNESS_PLAN_RETRIES) | run_isg.py, motions.py | 2 trajopt seeds, 2 attempts, 1 witness retry |
| COLLECTOR_PREBUILD (TEMPLATE_TOL_M 0.005) | oracle/run_isg.py, scripts/job_fast.sh | collectors build their world from a job-local copy of the counted eval template while the sampler loads (`run_isg --queue --prebuild <tro>`); the first instance reuses it when its template identity matches and its scene agrees within 5 mm (robot, bound objects, sleep flags and velocities excluded), else `prebuild.mismatch` and a restart without prebuild |
| STRAIGHT_ARM_VIA | oracle/motions.py | when the direct ramp collides: two checked straight legs through the ready / untucked posture before trajopt (round 2) |
| BAND_FEASIBLE_REUSE (**off** since 2026-11-01) | v2/tabletop.py | the feasible set a sampler stored in the band for the same (trunk, focus cell) instead of the two padded IK screens (round 2); superseded by BAND_EVIDENCE (a reused set that ran dry was never re-screened) |
| BAND_EVIDENCE (BAND_PAIR_FAIL_K 3, BAND_CELL_EMPTY_M 3, BAND_RESCREEN_P 0.25) | v2/band_evidence.py, v2/tabletop.py | counts per (world, trunk, cell, yaw, grasp) instead of verdicts: F = union of every pair a screen passed minus pairs the exact IK demoted (3 failures and > 2x its confirmations); a cell dies only after 3 empty screens with no pair ever; 1 attempt in 4 re-screens a known cell (F grows); a reused F that runs dry is re-screened once; the band itself is no longer rewritten (2026-11-01) |
| BAND_JOB_LOCAL | sampling/reach_bands.py | a band found outside `ISPATIALGYM_REACH_BANDS` is copied into the job's band directory and the copy loaded, and the job's copy wins over the same file elsewhere (the newest-first order let one committed band, rewritten by a job, become every job's shared file) (2026-11-01) |
| BAND_SHARE (**off**) | scripts/job_fast.sh, v2/band_evidence.py | opt-in sharing between the campaign's jobs: the campaign's published bands copied in, the band evidence fetched at job start / published per job at its end (`$F/band_evidence/<alias>/<band>.evidence.<job>.json`, merged by summing; no file written by two jobs) (2026-11-01) |
| IK_FRESH_P (0.2), IK_CACHE_GROW (IK_BRANCH_MIN_RAD 0.2, IK_CACHE_MAX_CHAINS 6) | v2/ik.py | one exact-IK call in 5 with a cache entry solves afresh; an entry keeps every distinct branch found (verified hits tried in a random order) (2026-11-01) |
| BAND_REFINE (BAND_REFINE_K 2, BAND_REFINE_ADD_P 0.1, BAND_REFINE_INIT nearest since 2026-11-01, was bracket) | v2/band_refine.py, reach_bands.py, v2/tabletop.py | bands refined per ACTUAL stance trunk (0.01 rad key) from a copy of the nearest grid trunk (`bracket`: the bracketing grid tilts' common cells, which halved ontop_small_01's yield in the refine A/B); only verified reachability failures (re-checked with the task-spawned objects out of the collision world, K = 2) blacklist cells / pairs; lattice neighbours found reachable are added; world-independent sidecar `<band>.refine.json` (supersedes BAND_EVIDENCE's counting while on) (2026-11-01) |
| BAND_VERIFY_STORE | v2/tabletop.py | the base band write-back deletes a cell on an empty screen only when the screen is still empty with the task objects out (2026-11-01) |
| TILT_SNAP (**off** since 2026-11-01: it lowered inside_02's yield) | v2/tabletop.py | an empty F at a stance's continuous tilt: snap to the band's grid tilt when collision-free and Phase 2's camera rules pass, else keep the tilt and keep the cell only if both bracketing grid tilts hold it (2026-11-01) |
| IK_CACHE_SHARE_CROP | v2/ik.py, v2/core.py | the exact-IK cache key of a cropped world uses the full world's rooms and no crop: crop and full share verified entries (2026-11-01) |
| REPEAT_MEASURED | sampling/instance_separation.py | the train-repeat rule compares what the task measures (TRO pose / articulation / group targets at their own tolerances) + the robot start base; eval indexes rebuilt (`$F/eval_index_v2`) (2026-11-01) |
| RECORD_CHANGED_ONLY (RECORD_CHANGE_TOL 1e-5) | oracle/executor.py | per-step recorded states keep only awake objects whose pose / joints changed since their last recorded state (static-but-awake scenery was re-posed every replay step: 2x slower replay, disturbed temporal history) (2026-11-01) |
| DIAG_EMPTY_F (**off**) | v2/tabletop.py | diagnostic: on an empty F, the screen with every scene object out and with each nearby object out (2026-11-01) |
| COLLECTOR_GPU_DYNAMICS (**off**) | oracle/run_isg.py | GPU PhysX for the single-scene collector (gm.USE_GPU_DYNAMICS, sim device cpu); measured 1.2-2.1x slower per env step than CPU PhysX alone on a TITAN RTX and another physics realisation (follow-up F1, 2026-11-01) |
| SPLIT_CONE_RESTORE | v2/core.py | start-per-attempt: the template's full candidate list is restored before the split-cone pools are rebuilt (they were rebuilt from the previous attempt's one position, so every attempt after the first drew that position; only iors_small rebuilds its list) - a bug of the train85 spa2still patch (2026-11-01) |
| ATTACH_FAST | oracle/curobo_planner.py | collision-mesh geometry cached per prim; the sphere-fit cache key rounds the eef-frame pose to 1 mm so repeated attaches of one grasp hit (round 2) |
| CR_GEODESIC_FIELD (CR_FIELD_TOL_M 0.15) | v2/tasks/camera_relocalization.py | one 8-connected Dijkstra from the init cell on the eroded map (BEHAVIOR's A* metric); goal points drawn among the room's cells whose geodesic lies in the bucket; the A* bucket check stays (2026-11-01) |
| CR_OBSTACLE_PREFILTER | v2/tasks/camera_relocalization.py | obstructed: skip a goal whose path has no middle cell in the room and clear of both bases (the near_robot rule) before its path check and renders; obstacle cells drawn only among those (2026-11-01) |
| CR_OBSTACLE_RETRY (CR_OBSTACLE_GOAL_TRIES 3) | v2/tasks/camera_relocalization.py | obstructed: a failed obstacle placement drops that goal and tries the next candidate instead of rejecting the attempt (2026-11-01) |
| tunables SPA_CELL_M (0.02; round 2 jobs: 0.05), BATCH_IK_SEEDS (12; round 2: 8), SPA_CELL_TRIES (6), BATCH_SIZES (8,16,32,64), STRAIGHT_ARM_STEP_RAD (0.02), STRAIGHT_ARM_MARGIN_M (0.01) | fastflags | set per job in scripts/fast_env.sh |

Instrumentation (always on): per-attempt stage times and cuRobo counters (`ik`, `ik_batch`, `ik_path`, `collision`, `sync`, `attach`)
in the sample log and the instance summary; `program_check.last_stats`; meta.json `fast` block with the switches seen.

## Not done here (deliberately)

- Instance families (one validated anchor, K children re-drawing start and goal): changes what an instance is; measure the levers
  above first.
- Rendering: the replay stage is separate (Vast, or the cluster with the vectorized driver); its cost is ~2,600 process-hours for 10x.
- A vectorized sampler or oracle: the sampler is cuRobo-bound and the oracle main-thread-bound (multienv report, 2026-10-03).

## Follow-ups requested (2026-10-31, after the current tests)

1. GPU PhysX for a single-scene collector: one-instance A/B against CPU PhysX on the same TRO (per-step time, GPU/CPU use, grasp / settle /
   still-exit verdicts; the checks and the vec_next sleep-threshold fix were tuned on CPU dynamics).
2. Vectorized oracle collector: N scenes per process with GPU physics on the vec_next world build. Needs the oracle restructured to step N
   independent skill programs and the 48 synchronous planner calls made non-blocking; bounded by ~31 ms of Python per scene per step on the
   main thread, so ~3-4x per GPU at best.

## Measuring

`scripts/fleet_summary.py --tag base --tag fast` (campaign root): successes per GPU-hour, s per attempt, attempts per written instance,
s per demo, per alias and overall. The A side is `out/base/` (train85's job flow on the frozen `code_base_43ad206a_spa2still`).
