# train-fast: what changed and how to re-run the experiments

The record of the train-fast campaign (2026-10-31 .. 11-02): the code changes on this branch, the experiments run on the
Stanford cluster, their results, and the commands to repeat them. The design and the full switch table are in
[FAST_FLOW.md](FAST_FLOW.md); the 10-31 interim speed report is [SPEED_REPORT.md](SPEED_REPORT.md). The chronological log with
every job id is `$F/DECISIONS.md` on the cluster (append-only); this file is the summary and the replication guide.

Status at the end of the campaign: the code works and is committed; no production data was kept (the owner stopped every
production fleet on 11-01 15:00: "we only need the code to work"). The instances and demos under `$F/out/` are test outputs, not
curated training data.

## 1. Where everything is

| what | where |
|---|---|
| code (commits) | branch `train-fast` on lambda, `~/Desktop/projects/ispatialgym-train-fast`; tip `0bc722ac` and later (section 2); not pushed |
| cluster worktree `$W` | `/vision/u/minyeong/projects/ispatialgym-train-fast`: git HEAD 43ad206a + the branch's changes as uncommitted files (synced from lambda, md5-checked; never commit on the cluster) |
| Behavior-1K | `6a6194600` (pad-clear release), a nested worktree `$W/Behavior-1K`; lambda: `~/Desktop/projects/Behavior-1K-train-fast-6a6194600` |
| campaign record `$F` | `/vision/u/minyeong/projects/isg_home/trainfast/`: `DECISIONS.md`, `scripts/`, `out/<tag>/<alias key>/g<start>-<end>/`, `slurm/` |
| frozen baseline code | `$F/code_base_43ad206a_spa2still` (43ad206a + the train85 spa2still patch, no train-fast change; never edited) |
| eval indexes (measured repeat rule) | `$F/eval_index_v2/per_alias` (83 aliases), built by `scripts/build_eval_index_v2.py` from train85's |
| shared sampling cache | `$F/cache/sampling` (start-draw cells, exact-IK witnesses; `ISGF_SAMPLING_CACHE` overrides) |
| read-only inputs | `$E=/vision/u/minyeong/projects/isg_home/eval85` (counted eval templates, `snapshot.sh`), train85's `bands_train` and templates |

A snapshot of `$F/scripts` (11-01 20:10 PST, refreshed at 22:30 with the E15 scripts and on 11-02 with the E16 / E17 scripts) is in
[scripts/trainfast/](../scripts/trainfast/), without `__pycache__`. The scripts hard-code the cluster paths (`$F`, `$W`, train85,
eval85) and run from `$F/scripts`; the cluster copy is the live one. The `*.bak_*` files are `job_fast.sh` / `submit_fast.py` before
their 10-31 21:50, 11-01 09:27 and 11-02 01:46 edits, which is what earlier tags ran. Section 6 lists the scripts each experiment
needs.

## 2. Code changes (commits on 43ad206a)

| commit | what |
|---|---|
| `cbebf6d4` | the train85 spa2still patch exactly as train85 ran it (start per attempt, oracle still exit) - the starting point |
| `77d1353c` | `ispatialgym/fastflags.py` and the job-flow switches of 10-31 .. 11-01 (collector trim, plan ladder, straight arm, batched-IK sizing, park, stage timing, attach / sync, start-draw cache) |
| `d794d0ad` | camera relocalization: CR_GEODESIC_FIELD, CR_OBSTACLE_PREFILTER, CR_OBSTACLE_RETRY |
| `3c9b86ea` | world loading: SCENE_CROP + CROP_NEIGHBOUR_FLOORS, PARTIAL_SCENE record / replay, SAMPLER_CAMERAS, COLLECTOR_PREBUILD (one commit because they share lines) |
| `a7854b84` | SPLIT_CONE_RESTORE |
| `636a71d0` | reach bands: BAND_JOB_LOCAL, BAND_EVIDENCE, BAND_REFINE, BAND_VERIFY_STORE, TILT_SNAP (+ BAND_FEASIBLE_REUSE and DIAG_EMPTY_F, both off) |
| `ec4c0ac9` | exact-IK cache: IK_FRESH_P, IK_CACHE_GROW, IK_CACHE_SHARE_CROP |
| `7ebae548` | train-repeat rule: EARLY_REPEAT_FILTER + REPEAT_MEASURED |
| `433b1c14` | RECORD_CHANGED_ONLY and the per-step cost split (`Executor.step_costs`) |
| `6124ed00` | unit tests of the switches (stubs, no simulator) |
| `0bc722ac` | FAST_FLOW.md, SPEED_REPORT.md |
| `2616ca71` | this README |
| later | the owner's 11-01 defaults (TILT_SNAP off, BAND_REFINE_INIT nearest) and the `scripts/trainfast/` snapshot: `git log --oneline 2616ca71..train-fast` |
| `cb7bc1dd`, `3eb83ac9` | COLLECTOR_GPU_DYNAMICS (off, E17), COLLECTOR_PHYSX_THREADS (E16) |
| `0b716acf` and later | `job_fast.sh`: the crop-matched prebuild source and COLLECTOR_PIN (default on since 11-02 04:41), `job_pair.sh` halves (E16) |

Every switch is `ISPATIALGYM_FAST_<NAME>`, read by `fastflags.fast_on` / `fast_value`. `=0` turns one off and `ISPATIALGYM_FAST=0`
turns all of them off (the base code's behaviour, an A side on the same code). They are **on by default** except:
BAND_FEASIBLE_REUSE, DIAG_EMPTY_F, the three IK branch switches LIMIT_RETRY_RETURN_ALL / REMEMBERED_WITH_SEED / SINGLE_RETRY_EXACT
(off since 11-01: they caused robot stutters, experiment E6), TILT_SNAP (off since 11-01 20:10, E11), and BAND_SHARE (job-script
level, off). SCENE_CROP takes a mode (`full` / `rooms` / `region`; the default depends on the family, see FAST_FLOW.md).
BAND_REFINE_INIT defaults to `nearest` since 11-01 20:10 (was `bracket`, E14).

**Owner decisions of 11-01 evening:**
- TILT_SNAP off (it lowered inside_02's yield, E11).
- BAND_REFINE_INIT `nearest` (the `bracket` start halved ontop's yield, E14). BAND_REFINE itself stays on: with `nearest` it was
  not worse than off on either alias.
- RECORD_CHANGED_ONLY stays as it is (on). Collections use the cropped scene path; its effect on full-scene physics (E13) matters
  only if full-scene demos are needed.

**Still to decide:** CR_OBSTACLE_RETRY was not shown to help or hurt (E12).

## 3. Setting up a replication

1. **Code.** `git worktree add <dir> train-fast` (or check out a commit of section 2). On the cluster the worktree also needs
   Behavior-1K `6a6194600` as `<dir>/Behavior-1K` and its `datasets/` links (behavior-1k-assets, omnigibson-robot-assets, the
   OmniGibson key, 2026-challenge-task-instances: the main checkout's; ispatialgym-assets: the worktree's own
   `ispatialgym/assets/datasets/ispatialgym-assets`), as `$E/snapshot.sh` builds them. Point `$W` at it in the job scripts.
2. **Sync** lambda -> cluster only with
   `rsync -a --exclude __pycache__ --exclude '*.pyc' <lambda worktree>/ispatialgym/ minyeong@svl11:$W/ispatialgym/`
   (one source, the same subdirectory on both sides, never `--delete`, never the worktree root). Then compare per-file md5 of the
   changed files. A malformed sync on 10-31 deleted most of `$W`.
3. **Unit tests** (lambda, `ispatialgym` conda env): `python -m pytest -q -p no:cacheprovider ispatialgym/tests`. Expected at
   `0bc722ac`: 11 failed (the base set, `base_fails.txt`: ag_release_hook x5, multienv x1, v2_core x2, v2_large_scale x2,
   vec_stream x1), 2178 passed, 4 skipped, and 3 setup errors in test_sim_evaluator's two alignment tests (present before the campaign's
   changes too).
4. **Reach bands:** the committed bands (`runlogs/reach_bands_20260915/bands`) must stay unmodified. After any fleet, run
   `git status --short runlogs/` in `$W`; it must print nothing (BAND_JOB_LOCAL keeps write-backs in the job's own directory).

## 4. Running jobs

One job = one alias template on one TITAN RTX: the sampler and K physics-only collectors run concurrently (`scripts/job_fast.sh`),
demos are hdf5 only, images come later from the kinematic replay. Submit from `$F`:

```bash
cd /vision/u/minyeong/projects/isg_home/trainfast
W=/vision/u/minyeong/projects/ispatialgym-train-fast
COMMON="--flow fast --code $W --kcollect 2 --cpus 12 --mem 32G --sample-limit-s 16000 --time-s 21600"
python3 scripts/submit_fast.py submit <alias key> 24 $COMMON --tag <tag> [--extra-env ISPATIALGYM_FAST_<NAME>=<v> ...]
```

- `<alias key>` is the full key, e.g. `inter_object_relation_satisfaction-ontop_small_scale_01`,
  `camera_relocalization-obstructed-toy_car_1`. Each call is one job (n instances).
  - Train instance indices come from the campaign ledger (`manifests/allocations.tsv`, from 100000 per alias), so a new job never
    reuses an index.
- An A/B arm = a tag plus `--extra-env` overrides. These land in `job.env` and `job_fast.sh` exports them.
- `--flow base --code $F/code_base_43ad206a_spa2still` = train85's job flow (sampler, then rendered demos one after the other).
- **Resources:**
  - 12 CPUs / 32 GB per job. 24 CPUs filled a 72-CPU / 8-GPU TITAN node at 3 jobs; jobs use a median of 8 cores.
  - COLLECTOR_PIN (on by default since 11-02) gives each collector 3 CPUs of its own and the sampler the rest; it needs
    (K + 1) x 3 CPUs, so k=2 runs on 12 and k=3 on 16 (48 GB). With fewer CPUs it stays off and `job.txt` says `PIN off`.
  - Caps are 48 GPUs / 56 jobs per user on svl and 30 GPUs on viscam (`--part viscam --acct viscam --gres l40s`; gres = the type only).
  - viscam12 is excluded (two CUDA illegal-memory-access crashes in cuRobo).
  - Submit with `sbatch` only (srun from svl11 hangs).
- Other job scripts:
  - `job_collect.sh <job dir> <code> [K]`: collectors on an existing queue.
  - `job_replay_multi.sh <job dir> <code> <P> <out dir>`: P replay processes on one GPU; `REPLAY_EXTRA` passes replay flags, e.g.
    `--scene full`, `--delta-load`.
  - `job_base.sh`: the baseline flow.
- Each job writes `out/<tag>/<key>/g<start>-<end>/`: `job.env`, `job.txt` (process walls), `sample.log` (one "rejected in <stage> after
  <s>s: [reason]" or "written (attempt k, <s>s)" line per attempt, plus per-attempt `stages:` / `kit:` counter lines), `inst/`
  (templates, instances with `.oracle.json` / `.summary.json` sidecars), `demo/<instance>/` (`demo.hdf5`, `meta.json` with
  `fast.*`), `results.jsonl`.
- Analysis (`$F/scripts`):
  - `fleet_summary.py --tag A --tag B`: successes per GPU-hour, whole job and steady, s per attempt, attempts per written, s per demo.
  - `stage_summary.py`: where the sampler time goes.
  - `demo_timings.py`: one line per finished demo.
  - `fast_check.py <job>`: per-job statistics.
  - The experiment-specific scripts are named in section 6.

To repeat an old arm on the current code, pin off the switches that did not exist yet when it ran (section 5) and use the eval index
it used. Jobs submitted before 11-01 10:40 read train85's eval index (the old repeat rule); later jobs read `$F/eval_index_v2`.

## 5. When each switch arrived (the code a tag ran on)

The cluster worktree was edited during the campaign, so a tag ran on the code of its submission time:

| time (PST/PDT, as logged) | change |
|---|---|
| 10-31 19:36 | round 1: EARLY_REPEAT_FILTER, PARK_ONCE / PARK_SPAWNED, SAMPLER_CAMERAS, SPA_CACHE, the batched-IK switches, EARLY_FRAME_GATE, SYNC_ONLY_OBJECTS, ATTACH_REUSE, PHYSICS_ONLY_TRIM, STRAIGHT_ARM, PLAN_LADDER (tag `fast`) |
| 10-31 20:03 | round 2 (tag `fast2`): STRAIGHT_ARM_VIA, BAND_FEASIBLE_REUSE, ATTACH_FAST; tunables SPA_CELL_M 0.05, BATCH_IK_SEEDS 8 set per job |
| 10-31 21:14 | touching push-stance search: partial syncs of the moved object (`tabletop.sync_moved`) |
| 10-31 22:08 | COLLECTOR_PREBUILD (job_fast.sh prebuilds from an earlier job's template copy) |
| 11-01 05:45 | the three IK branch switches default off |
| 11-01 06:55 | SCENE_CROP, PARTIAL_SCENE; collector step-cost / world-sync timers |
| 11-01 07:40 | CR_GEODESIC_FIELD, CR_OBSTACLE_PREFILTER, CR_OBSTACLE_RETRY |
| 11-01 08:25 | the crop floor fix (CROP_NEIGHBOUR_FLOORS) - `crop` instances before it are invalid |
| 11-01 09:40 | SPLIT_CONE_RESTORE, BAND_JOB_LOCAL, BAND_EVIDENCE (BAND_FEASIBLE_REUSE off), IK_FRESH_P, IK_CACHE_GROW, BAND_SHARE off |
| 11-01 10:40 | REPEAT_MEASURED; `submit_fast.py` reads `eval_index_v2` |
| 11-01 11:00 | IK_CACHE_SHARE_CROP, DIAG_EMPTY_F |
| 11-01 12:00 | BAND_EVIDENCE keyed by the stance's own trunk |
| 11-01 15:40 | TILT_SNAP |
| 11-01 16:20 | BAND_REFINE, BAND_VERIFY_STORE, RECORD_CHANGED_ONLY (DECISIONS logs them at 17:00; the first `refine_on` jobs started 16:20, `full3` at 16:33) |
| 11-01 20:10 | defaults: TILT_SNAP off, BAND_REFINE_INIT `nearest`. Tags submitted from 15:40 ran TILT_SNAP on, and tags from 16:20 the `bracket` start: to repeat them add `ISPATIALGYM_FAST_TILT_SNAP=1` / `ISPATIALGYM_FAST_BAND_REFINE_INIT=bracket` |
| 11-01 22:51 | COLLECTOR_GPU_DYNAMICS (off) |
| 11-02 01:39 | COLLECTOR_PHYSX_THREADS (0 = PhysX's own count) |
| 11-02 01:46 | `job_fast.sh`: COLLECTOR_PREBUILD takes only a template copy with the job's scene crop request (`prebuild_crop.py`); the `pk_*` tags (E16) ran before it |
| 11-02 02:34 | `job_fast.sh`: COLLECTOR_PIN (off; on per job with `ISPATIALGYM_FAST_COLLECTOR_PIN=1`), `cpu_split.py` |
| 11-02 04:41 | COLLECTOR_PIN on by default (`ISPATIALGYM_FAST_COLLECTOR_PIN=0` turns it off); `job_pair.sh` runs each member on half of the job's physical cores (`cpu_split.py halves`). Tags before it ran unpinned unless they set the switch; the `pk_pair*` members shared all the CPUs |

## 6. Experiments

Aliases are written short: `ontop_small_01` = `inter_object_relation_satisfaction-ontop_small_scale_01`, and so on. n = instances per
job, k = collectors per job. "Steady" successes per GPU-hour exclude the world load at the job start; "whole" includes it.
Production (train85) made 7,183 successful demos in 2,404 GPU-hours of job wall, 2.99 per GPU-hour. The 10-31 notes quoted 4.6,
which the job walls do not reproduce. Speed-ups are against train85 on the same alias keys (`speed_compare.py`; see SPEED_REPORT.md).

### E1. Speed of the fast flow (10-31)

- Tags: `base` (train85 flow, frozen code, n=4), `flags` (fast job flow on the frozen code, n=12, k=2), `fast` (the branch, n=12,
  k=2), `fast2` (round 2), `smoke` / `smoke2` (end-to-end checks), `pilot` (10 aliases x 2 jobs, n=40, k=2, 24 CPUs).
- 14 aliases: ontop_small_01, inside_02, nextto_02, under_small_01, touching_01, axial_01, world_vertical_02, attachment_02,
  artic_open_small_01, disembedding_03, cr-obstructed-toy_car_1, mr-mug, os-bell_pepper, block_design_01.
- Results:
  - Final whole / steady successes per GPU-hour on the 14 keys (train85 on them: 3.2): base 2.1 / 7.4, flags 6.8 / 16.7, fast
    8.8 / 21.9 (x2.8), fast2 10.7 / 22.7 (x3.4). Cancelled jobs count to their slurm end (the 10-31 interim 7.5 for round 1 was
    taken while jobs ran).
  - pilot: 782 written, 714 successes (91 %), 53.0 GPU-h, i.e. 13.5 whole / 15.9 steady per GPU-hour, x3.4 against train85 on the
    same 10 keys (4.0 per GPU-hour there).
  - Demo success by tag: base 84 %, flags 95, fast 97, fast2 91, pilot 91.
  - axial_01, attachment_02, block_design_01 and inside_02 wrote nothing in any tag (yield problems, see E8, E10, E11).
- Caveat: house tabletop numbers from round 2 until 11-01 09:40 ran with the band contamination (E9) and count only as speed
  indications.

### E2. Packing and isolated runs (10-31)

- Tags: `fast_k1` (one collector), `fast_split` (sampler-only and collector-only jobs sharing a queue over NFS, `job_collect.sh`),
  `fast_c24` (24 CPUs), `profile` (one nextto collector alone, base vs fast code).
- Results:
  - Collectors are CPU-bound: 3-12 % GPU use even alone.
  - An isolated nextto demo: base 473 s load / 235 s episode / 56 s plan; fast 431 / 179 / 33.
  - An uncontended sampler takes 62 s per attempt (ontop), against 108-137 s co-located with two collectors.
  - 24 vs 9 CPUs was inconclusive at n=12. The CPU request was later set to 12 from the jobs' measured use (sacct).
  - Per key against fast2 (SPEED_REPORT.md): one collector 7.8 / 10.6 vs 9.3 / 12.7 successes per GPU-hour; the split layout 2.5 / 2.0
    vs 12.2 / 6.0 (its collector GPU idles); 24 CPUs 8.0 / 10.1 / 13.3 vs 9.3 / 12.2 / 12.7. Sampler + 2 collectors stays.

### E3. COLLECTOR_PREBUILD (10-31 22:40)

- Tag `prebuild`: ontop, nextto, mug, bell_pepper, disembedding_03; n=12, k=2.
- Result: every collector accepted the prebuilt world. The first instance loaded in 16-17 s instead of 430-520 s. Job start to first
  demo: ontop 17 min (23-24 without prebuild), nextto 20.6 (27-30), mug 8.2, bell_pepper 7.2.
- Two early refusals were bugs, since fixed: a cached AABB compared as geometry, and prebuilding from the counted template instead of
  the sampler's output.

### E4. Replay throughput (10-31 / 11-01)

- Tag `replay_test` (`job_replay_multi.sh` on the 12 fast2 ontop demos, house scene; flags `--slots 1 --batched-usd-edits
  --async-flush --proprio loaded --warmup-s 15`).
- Results:
  - Per GPU: 1 process 3.65 env-steps/s, 2 processes 6.6 total, 3 processes ~8.0 total (21.7 GB).
  - `--delta-load` 5.71 env-steps/s; `--delta-load --batched-usd-edits` 6.5. Delta-load renders moving parts one state late with
    batched edits.
  - A one-room scene replays at 6.3-6.5 env-steps/s: the per-tick state load scales with the scene's object count.

### E5. Large-scale IORS on the snug mesh store (10-31)

- Tags: `large`, `large_base` on touching_large_03, inside_large_03, nextto_large_02.
- Result: no mesh-store overflow anywhere. touching_large_03 showed new robot stutters, which E6 traces.

### E6. Stutter A/B (11-01 05:45)

- touching_large_03, n=12 per arm. Arms: all switches (tag `large`); `large_ab_noik` with
  `LIMIT_RETRY_RETURN_ALL=0 REMEMBERED_WITH_SEED=0 SINGLE_RETRY_EXACT=0`; `large_ab_noramp` with `STRAIGHT_ARM=0`.
- Successes / stutters / descent joint jumps: all on 6 / 4 / 12; IK branch switches off 10 / 1 / 2; STRAIGHT_ARM off 8 / 2 / 8.
- Decision: the IK branch switches default off (they change which witness branch the sampler records). STRAIGHT_ARM stays on.
- To repeat the "all on" arm today, set the three switches to `=1`.

### E7. Scene crop (11-01)

- Tags:
  - `crop` (defaults, 11 aliases) vs `crop_off` (`SCENE_CROP=full`, the 6 region aliases).
  - `recollect` and `recollect2`: failed instances re-collected partial vs full with `mk_recollect.sh <key> <reason regex> <dest
    tag> [source tag]`.
  - `crop2`: after the floor fix.
  - `crop3` / `full3`; then `crop4` / `full4` (crop default vs `SCENE_CROP=full`) on 10 aliases, then 9 more, n=24, k=2.
- Bug found:
  - The first crop dropped the neighbour rooms' floors. OmniGibson loads floors only with their room, so bases stood 1.5 cm low and
    the arm met walls at the first transit (6 collision failures in `crop`).
  - The fix makes neighbour floors crop candidates (CROP_NEIGHBOUR_FLOORS). Instances of tag `crop` are invalid.
- Failure attribution: the full scene reproduced every failure of the crop2 demos with the same reason. They are failures of the
  instances, not of the cropped world.
- crop4 vs full4 (same warm exact-IK cache, measured repeat rule):

  | | crop4 | full4 |
  |---|---|---|
  | s per attempt | 32 | 44 |
  | attempts per written | 4.8 | 5.4 |
  | demo success | 85 % | 84 % |
  | s per demo | 160 | 304 |
  | succ/GPU-h whole / steady | 14.6 / 20.9 | 8.2 / 13.8 |

  The crop costs no success: about 1.5x steady, 1.8x whole-job.

### E8. Disembedding zero yield: the split-cone collapse (11-01 09:40)

- Tags: `nf_on`, `nf_off`, `cold_full`, `cold_nf_on`, `nofeas_nf_on` (disembedding_03).
- crop2 wrote 0/85 and nf_on 0/98; nf_off 5/34; cold_full (empty cache) 2 in its first 3 attempts.
- Ruled out: the neighbour floors and the cold IK cache.
- Root cause, in the train85 spa2still patch:
  - `split_cone_draw` had overwritten `manip_candidates` with the previous attempt's stances, so every attempt after the first drew
    the same start position. A job's fate was its first draw.
  - Fix: SPLIT_CONE_RESTORE; `tests/test_split_cone_restore.py` reproduces the collapse with the switch off.
- isg-1020 impact: train85 ran start-per-attempt on 22 split-cone aliases (axial 4, articulation small 4, attachment 4,
  block_design 1, disembedding 9). Their instances of one job share one start after the first attempt.

### E9. Band evidence and the IK cache (11-01 12:50)

- Tag `crop3` (BAND_EVIDENCE, IK_FRESH_P, IK_CACHE_GROW on) vs `ev_off` (`BAND_EVIDENCE=0 IK_FRESH_P=0 IK_CACHE_GROW=0`), one job per
  alias per side.
- Attempts per written, on / off: nextto_02 5.6 / 9.9, ontop_small_01 8.0 / 17.0, touching_01 6.8 / 12.2, under_small_01 4.8 / 3.8,
  artic_open_small_01 1.3 / 1.5, disembedding_03 2.4 / 2.1. Kept on.
- Context: the band contamination. `ReachBands.find` sorts newest-first across directories, so once a job wrote back into a committed
  band, every job loaded and rewrote that one file. BAND_JOB_LOCAL fixed it, the shipped bands were restored with `git checkout`, and
  `$F/bands_train` was quarantined (`bands_train.quarantine_20261101`).

### E10. Repeat rule (11-01 10:40)

- Offline: `rule_compare.py` judges every written train instance against its alias's eval instance under both rules. The measured
  rule flags axial_01 4/6 and world_vertical_02 9/144 that the old rule passed; every other alias 0 under both.
- Tag `rule2` (measured rule, `eval_index_v2`): axial_01 24/24 in 123 attempts and attachment_02 24/24 in 54. Under the old rule
  (crop3) they were 5/191 and 0/101, mostly `repeats_eval`.
- block_design_01 fails every attempt at board placement in every tag, the frozen train85 code included. This is a separate,
  pre-existing problem.

### E11. inside_02: the crop divergence and TILT_SNAP (11-01)

- Tags `diag_inside_crop` / `diag_inside_full` (same seeds, START 900000, `DIAG_EMPTY_F=1`): identical results attempt for attempt.
  - The foci 0.96 m ahead are out of reach at trunk 0.25/-1.07 even with every scene object removed.
  - The band's grid tilts are -0.3/-0.6/-0.9/-1.2 and Phase 2 stances take any tilt within ±0.15 of them; -1.07 read the cells of
    -1.2. Not the crop.
- Tag `tiltsnap` (inside_02, TILT_SNAP on) vs crop4's inside_02 (before TILT_SNAP existed):
  - Written / attempts: 8/680 (1.18 %, Wilson 95 % 0.60-2.30) vs 15/493 (3.04 %, 1.85-4.96); Fisher two-sided p = 0.031.
  - `tilt_snap.adopted` 163 times. Rejections shifted to `gripper_not_in_frame` at pre-place (385 vs 258).
  - inside_02 is limited by the pre-place frame check, not the tilt.
  - TILT_SNAP is off by default since 11-01 20:10, so the baseline arm is now the default; the `tiltsnap` arm needs
    `ISPATIALGYM_FAST_TILT_SNAP=1`.

### E12. Camera relocalization switches (11-01)

- Tags `cr_fast` (defaults) vs `cr_off` (`CR_GEODESIC_FIELD=0 CR_OBSTACLE_PREFILTER=0 CR_OBSTACLE_RETRY=0`) on
  obstructed-toy_car_1, obstructed-straight_chair_1 and a standard alias. Then `cr_noretry` (`CR_OBSTACLE_RETRY=0`) on the two
  obstructed aliases. Script: `cr_ab.py [tag ...]`.
- cr_fast vs cr_off:
  - 2.3 vs 3.9 attempts per written, but 215 vs 104 s per attempt; whole succ/GPU-h 6.6 vs 8.0.
  - The standard alias was better with the field (20.5 vs 18.1), the obstructed ones worse.
- First 9 written instances per run, attempts per written / s per attempt / sampler s per written:
  - toy_car: no-retry 2.7 / 124 / 332, fast 1.8 / 194 / 345, off 2.3 / 104 / 244.
  - straight_chair: no-retry 3.1 / 159 / 495, fast 2.0 / 194 / 387, off 7.8 / 102 / 790.
- One job per arm: not enough to turn CR_OBSTACLE_RETRY off.

### E13. Partial-scene validation and the recording fix (11-01)

- Setup:
  - `mk_pv.sh <key> <source tag> [N]` copies N successful crop4 instances into `out/pv/<key>/{partial,full}` queues, then
    collectors run them in the cropped world and in the full scene (`ISPATIALGYM_FAST_PARTIAL_SCENE=0`), each twice (`partial2`,
    `full2`).
  - 6 aliases: nextto_02, ontop_small_01, touching_large_03, artic_open_small_01, disembedding_03, mr-mug.
  - `submit_pv_replays.sh` replays each side: `rp_full` = partial demos `--scene full`, `rf_rec` = full demos `--scene recorded`,
    and so on.
  - Scripts:
    - `compare_pv.py`: trajectories.
    - `first_div.py <pv root> <tag A> <tag B>`: the first divergent step.
    - `div_order.py`: physics or plan first.
    - `compare_frames.py <lerobot A> <lerobot B> <png dir>`: head-camera mean abs grey levels / PSNR per episode.
    - `cmp_prefix.py <A> <B> <episode> <N>`: the same over the first N frames.
- Results:
  - Repeats in one world are bit-identical (cropped 11/12, full 8/9).
  - Cropped and full collections differ reproducibly for some instances: the joints diverge first (1e-6 rad, during contact), never
    the plan, so PhysX numerics depend on the bodies the scene holds.
  - A cropped demo replayed with `--scene full` matches the full demo to the render noise floor (mug 1.7-2.1 / 38-40 dB).
- The soft replay:
  - The full house scene keeps ~50 static objects awake, and BEHAVIOR's recording filter wrote them every step (53 objects per step
    vs 4 cropped). The replay re-posed them every step: half the speed and softer frames.
  - Fix: RECORD_CHANGED_ONLY. Validation: `full3` (the nextto_02 full collection with the filter), replayed as `rf3_rec`.

  | nextto_02 head camera, mean abs / median PSNR | ep 1 (100322) | ep 1, frames 0-223 | ep 0 (100320) |
  |---|---|---|---|
  | rf3_rec (fix) vs rp_full (cropped path) | 2.33 / 34.8 dB | 3.16 / 29.8 dB | 4.82 / 28.3 dB* |
  | rf_rec (before) vs rp_full | 3.23 / 28.8 dB | 5.43 / 23.8 dB | 5.38 / 23.2 dB* |
  | noise rp_full vs rp_full2 | 1.88 / 38.6 dB | 2.96 / 31.8 dB | 1.90 / 38.9 dB |

  \*Episode 0's cropped and full trajectories differ (989 vs 1017 steps).
- Recording-fix results:
  - Replay speed rose from 3.71 to 6.96 env-steps/s (cropped path 7.18); state floats per step dropped from a median of 1,298 to 170.
  - Over the frames with identical trajectories the fix closes 92 % of the gap. The rest sits 0.2 grey levels / 2 dB above noise.
  - **The filter is not physics-neutral.**
    - full3's 100322 diverges from `full` at step 224 (0.016 rad by +200 steps, 747 steps vs 744); 100320 is unaffected.
    - Repeats: `full4` (filter on) is bit-identical to full3, and `full5` (filter off, the same code) to `full` / `full2`. So turning
      the filter on changes that instance's physics; each setting is deterministic.
    - Checkpoints still dump every object (`Executor._dump_state(full=True)` lifts the filter), so restores are not the cause. The
      likely cause is the filter's own per-step reads of every awake object's pose and joints (`Executor._changed_filter`).
    - The demo is still a valid physics run (it succeeded) - another realisation, as with crop vs full. A neutral version would decide
      "changed" from the dumped per-object state instead of extra simulator reads, then pass the same full vs full5 check.

### E14. Band refinement A/B (11-01)

- Arms on ontop_small_01 and touching_01, 4 jobs of n=24 per arm and alias, submitted 16:20-17:01 (TILT_SNAP was on in all of them):
  - `refine_on`: the defaults then (BAND_REFINE, INIT bracket). On today's code add `BAND_REFINE_INIT=bracket TILT_SNAP=1`.
  - `refine_nearest`: `BAND_REFINE_INIT=nearest`, the default since 20:10. On today's code add `TILT_SNAP=1`.
  - `refine_off`: `BAND_REFINE=0`, i.e. BAND_EVIDENCE as before. On today's code add `TILT_SNAP=1`.
- Script: `refine_ab.py`. It reports written / attempts with Wilson intervals, Fisher p against off, the `refine.*` counters, the
  rejection mix, and the focus-ahead distance of written instances.
- Final results: every job ran to completion (the last at 19:04), except job 18248578 (touching, nearest), which was killed for
  running out of memory at 32 GB after 2 attempts (1 written); it is in the pooled numbers but not the job-level test. Yield =
  written / attempts. The 18:12 interim reading gave the same verdict.

  | alias | arm | yield (95 % Wilson) | Fisher p vs off | written per sampler-h | per-job yield % | job-level p vs off |
  |---|---|---|---|---|---|---|
  | ontop_small_01 | refine_on (bracket) | 51/972 = 5.2 % (4.0-6.8) | 1.6e-5 | 7.8 | 8.4, 3.2, 7.8, 3.2 | 0.11 |
  | | refine_nearest | 81/717 = 11.3 % (9.2-13.8) | 0.93 | 12.6 | 6.0, 13.5, 14.9, 13.4 | 0.89 |
  | | refine_off | 78/707 = 11.0 % (8.9-13.6) | - | 12.1 | 10.3, 19.1, 6.6, 12.4 | - |
  | touching_01 | refine_on (bracket) | 76/959 = 7.9 % (6.4-9.8) | 0.55 | 12.3 | 8.8, 7.7, 14.0, 4.4 | 0.49 |
  | | refine_nearest | 64/644 = 9.9 % (7.9-12.5) | 0.045 | 14.5 | 10.1, 13.9, 7.1 (+ the OOM job, 1/2) | 0.23 |
  | | refine_off | 76/1064 = 7.1 % (5.7-8.8) | - | 11.4 | 7.3, 6.6, 6.2, 8.6 | - |

  Job-level p = exact two-sided Mann-Whitney on the per-job yields; with 4 against 4 jobs the smallest possible p is 0.029. The Fisher
  test treats attempts as independent; attempts within a job share a template and the bands it learns, and the jobs of one arm
  spread widely (refine_off on ontop: 6.6-19.1 %).
- Counters (ontop / touching):
  - refine_on: `refine.new_key` 71 / 40, `verify_reach_fail` 223 / 210, `blacklisted` 69 / 33, `added` 15 / 32 of 23 / 69 screened,
    `verify_other` 2 / 2.
  - refine_nearest: `new_key` 72 / 33, `verify_reach_fail` 203 / 170, `blacklisted` 59 / 22, `added` 9 / 9 of 34 / 40 screened,
    `verify_other` 1 / 5.
  - Task objects alone almost never caused an empty screen (`verify_other` 1-5 per arm).
  - `tilt_snap.adopted` 67 / 69 / 87 on ontop and 33 / 22 / 32 on touching (on / nearest / off).
- Rejections on ontop: `path_collision` 40 % under bracket, 22 % under nearest, 27 % under off; `path_too_long` 14 / 41 / 29 %.
- Focus distance ahead of the base, written instances (median, range): ontop 0.80 m (0.72-0.96) for bracket and nearest, 0.88 m
  (0.72-0.96) for off; touching 0.72 m (0.64-0.88) for all three. At 18:12 nearest's ontop median was 0.88, so the shorter focus is
  not specific to the bracket start.
- Verdict:
  - On ontop the bracket start yields half of off: 5.2 % vs 11.0 %, p = 1.6e-5 per attempt. Per job it is p = 0.11 with 4 jobs, and
    every bracket job is below three of the four off jobs. Its rejections shift to `path_collision`.
  - nearest is indistinguishable from off on ontop (11.3 % vs 11.0 %). On touching it is higher (9.9 % vs 7.1 %, p = 0.045 per
    attempt, 0.23 per job with 3 jobs): suggestive, within noise at the job level.
  - bracket on touching: no difference from off.
  - Taken up: BAND_REFINE_INIT `nearest` is the default since 11-01 20:10. The owner's combination (nearest with TILT_SNAP off) was
    not run as an arm.

### E15. Collector costs, job balance and the final accounting (11-01 evening)

- No new jobs; scripts over the existing outputs (all in `$F/scripts`):
  - `step_costs.py [tags]`: the collectors' always-on timers (`meta.json` `fast.step_costs` / `fast.world_sync`), per family.
  - `job_balance.py [tags]`: collector busy share = demo time ÷ collector wall, and the drain after the sampler ends.
  - `gpu_use.py [tags]`: utilisation and peak memory from each job's `gpu.log`.
  - `speed_compare.py <tags>`: each tag against train85 on the same keys.
  - `projection.py <label> <tags> ...`: speed-up at train85's alias mix.
  - `fleet_summary.py` now charges a job cancelled before `ISGF_END` its sacct Elapsed instead of "until now".
- Results (2,021 demos of pilot2-4 / crop2 / crop4; 78 jobs):
  - Collector episode time: env step 59 %, QA 13 %, planning and the rest 15 %, world syncs 7 %, state dump 3 %, recorder 3 %.
  - Collectors are busy 29 % of their wall in the median job; only large IORS and disembedding keep them busy.
  - GPU utilisation is 8-39 % (median per tag); peak memory 17-20 GB per job.
  - At train85's mix: TITAN x2.9-4.3 (9.7-10.9 successes per GPU-hour), L40S x7.4 (21.7).
- Decision: no narrower per-step dump / sync set (it cannot move throughput while the sampler bounds the job). Details and the
  projection: SPEED_REPORT.md.

### E16. More processes per GPU, the CPU contention and COLLECTOR_PIN (11-01 night .. 11-02)

- **Fleet** (11-01 22:36, `pk_submit.sh singles | pairs`, n=24 per sampler). Keys: ontop_small_01 (sampler-bound), disembedding_03
  (collector-bound), mental_rotation-mug (fast sampler, short demos). Arms:
  - `pk_k2`: sampler + 2 collectors, 12 CPUs / 32G (the default);
  - `pk_k3`: + 3 collectors, 16 CPUs / 48G;
  - `pk_pair1`: 2 x (sampler + 1) on one GPU (`job_pair.sh`), 24 CPUs / 64G;
  - `pk_pair2`: 2 x (sampler + 2), 24 CPUs / 96G, L40S only.

  Successes per GPU-hour (`pack_ab.py`; per job `pack_detail.py`):

  | key | TITAN k2 | TITAN k3 | TITAN pair1 | L40S k2 | L40S k3 | L40S pair1 | L40S pair2 |
  |---|---|---|---|---|---|---|---|
  | ontop_small_01 | 12.2 | 12.8 | 14.9 | 7.4 | 11.4 | 25.9 | 19.2 |
  | disembedding_03 | 12.8 | 9.4 | 4.5 | 14.7 | 13.0 | 46.5 | 32.2 |
  | mental_rotation-mug | 28.6 | 24.8 | 39.1 | 50.6 | 55.8 | 66.3 | 82.1 |

  - One unit per cell (two for TITAN k2).
  - ontop's yield is decided by its slice of train indices (written per attempt 5-25 %).
  - The TITAN de03 / mug k2 and k3 jobs prebuilt from SCENE_CROP=full copies and rebuilt their world (the prebuild fix below).
  - TITAN de03 pair1: both samplers ran out of CUDA memory in cuRobo after ~12 min (24 GB) and wrote 1 demo. L40S pair2 peaked at 43
    of 48 GB.
  - A third collector made de03 slower (TITAN 12.8 -> 9.4): its collectors stepped at 724 ms, against 340 with two and 95 alone.
- **The contention.** One de03 instance (crop4 g100336-100359), run by N collectors (`run_isg --instance`) started 45 s apart in one
  job, each with its own Kit app data (`job_ct.sh`, `ct_submit.sh threads | env | pinsize | copy`, table `ct_ab.py`). Every run
  recorded the same demo (929 steps, first success at 714).

  | arm | env ms/step per collector |
  |---|---|
  | 1 collector, 12 CPUs | 95 |
  | 2 collectors, 12 CPUs | 334-350 |
  | 2, PhysX 2 / 4 threads (COLLECTOR_PHYSX_THREADS; PhysX's default is 8) | 322 / 344 |
  | 2, 24 CPUs | 367 (759-808 % CPU each) |
  | 3, 16 CPUs (with PhysX 2) | 516-586 (560-642) |
  | 2, each with its own HOME / isg caches / SLURM_JOB_ID / a pair member's whole environment | 349 / 322 / 341 / 283-298 |
  | 2, each with its own copy of the template files (plus the pair environment) | 307-309 (271-297) |
  | 2, pinned (taskset) to 6 CPUs each | 86-87 (144-149 % CPU each) |
  | 2, pinned to 2 / 3 / 4 CPUs each | 76 / 63-65 / 83 |
  | 3, pinned to 3 CPUs each | 75-97 |

  - Two unpinned Kit processes (~340 threads each) on one job's CPUs slow each other about 4x and burn about 4 cores each. QA goes
    from 3.7 to 22-41 ms per step.
  - More CPUs, fewer PhysX threads, and separate environments, caches or files do not remove it. Disjoint CPU sets do.
  - In the fleet's metas (per-demo time windows), a demo overlapped by another demo of its own `job_fast.sh` steps at 309-342 ms. A
    pair1 L40S de03 demo overlapped by the other member's demo steps at 55 ms. Why the two members of a pair escape it is open (it
    was not reproduced on TITAN; pair1 ran on L40S).
  - E2's sampler (62 s per attempt alone, 108-137 s next to two collectors) is likely the same contention.
- **Prebuild fix (11-02 01:46).**
  - The fleet's TITAN mug / de03 jobs prebuilt from full4's SCENE_CROP=full copies. Each collector then rebuilt its world after its
    first instance (`prebuild.mismatch`, ~8 min per collector).
  - `job_fast.sh` now takes only a copy whose scene crop request is the job's (`prebuild_crop.py`).
  - Smoke `pb_smoke` (mr-mug, n=4): the default took the newest cropped copy, 0 mismatches. `pb_smoke_full` (SCENE_CROP=full)
    skipped 11 cropped copies and took full4's.
- **COLLECTOR_PIN job-level A/B** (11-02 02:34, `pin_submit.sh`, n=24). `job_fast.sh` puts each collector on 3 CPUs of its own and
  the sampler on the rest (`cpu_split.py`). Per job `pack_detail.py`; per demo the medians of meta `fast.step_costs`:

  | arm | succ/GPU-h | collector env ms/step | sampler s/attempt | demo wall s |
  |---|---|---|---|---|
  | de03 TITAN k2 pinned (2 jobs) | 32.0 / 29.2 | 89 | 35-39 | 114 |
  | de03 TITAN k3 pinned, 16 CPUs | 32.6 | 92 | 32 | 127 |
  | de03 TITAN k2 unpinned (fresh control) | 12.9 | 374 | 77 | 423 |
  | de03 TITAN k2 / k3 unpinned (fleet) | 12.8 / 9.4 | 340 / 724 | 54-69 / 82 | 431 / 882 |
  | de03 L40S k2 / k3 pinned | 52.7 / 64.0 | 46 / 47 | 17 / 17 | 61 / 66 |
  | de03 L40S k2 / k3 unpinned (fleet) | 14.7 / 13.0 | 307 / 669 | 45 / 74 | 370 / 763 |
  | ontop TITAN k2 pinned / fleet | 4.0 / 12.2 | 49 / 113 | 24 / 27-49 | 120 / 203 |
  | mr-mug TITAN k2 pinned / fleet | 31.1 / 28.6 | 74 / 77 | 42 / 42-43 | 16 / 20 |

  - Pinned collectors step at about their alone speed, and three pinned collectors step as fast as two.
  - The de03 sampler gets faster too (35-39 against 77 s per attempt; acceptance 36-50 % in every arm), since its collectors are busy.
    mr-mug's collectors are idle (10 % busy) and nothing changes.
  - ontop pinned: its slice (g100921-100944) wrote 8 of 289 attempts. 275 of its 281 rejections are target_placement (geometry), as
    in the fleet's ontop jobs (68-228 of 72-233). The low count is the slice's; its collectors stepped 2.3x faster.
  - mr-mug's 31.1 against 28.6 comes from the prebuild fix (0 mismatches), not from the pinning.
  - **Decision (11-02 04:41): COLLECTOR_PIN on by default.** `job_pair.sh` now runs each member on half of the job's physical cores
    (`cpu_split.py halves`), so each member pins inside its own half.
- **Pinned pair** (11-02 04:42, `pin_submit.sh pair`, tag `pinpair2_l40s`, 18264576 on viscam10): 2 x (sampler + 2 collectors) on
  one L40S, each member pinned inside its half of the cores (a: 14-19 + 78-83, b: 20-23, 26, 27 + 84-87, 90, 91).
  - 104.6 successes per GPU-hour (42 in 24 min), against 32.2 for the unpinned `pk_pair2_l40s` and 64.0 for `pin_k3_l40s`.
  - Collectors 52 ms per step (308 unpinned), samplers 19-20 s per attempt (45-47), drain 65-89 s (1,893-1,983).
  - Peak GPU memory 43.2 of 48 GB; every process exited 0.

### E17. GPU PhysX and the vectorized collector (owner follow-ups, 11-01 evening)

- **F1, GPU PhysX for one scene.** Switch `ISPATIALGYM_FAST_COLLECTOR_GPU_DYNAMICS=1` (`run_isg.InstanceLoader.build_env` sets
  `gm.USE_GPU_DYNAMICS`; the simulator device stays `cpu`). One instance alone per TITAN RTX, two runs per arm, the median-length
  successful demo of a finished job (`job_gpu_dyn.sh`, `gd_submit.sh`, table `gpudyn_ab.py`; out `out/gpudyn/<instance>/<arm>_r<k>/`).

  | instance | env ms/step CPU (r1 / r2) | GPU | episode s CPU | GPU | steps, first success CPU -> GPU |
  |---|---|---|---|---|---|
  | ontop_small_01 | 52.6 / 52.4 | 95.7 / 87.1 | 145 / 144 | 192 / 182 | 896, 688 -> 926, 713 |
  | mental_rotation-mug | 35.5 / 42.8 | 82.2 / 82.3 | 18.6 / 18.7 | 26.0 / 25.6 | 94, 37 (both) |
  | disembedding_03 | 119.3 / 92.0 | 148.0 / 111.6 | 177 / 139 | 203 / 155 | 929, 714 -> 960, 740 |

  - All 12 runs succeed. Each arm repeats its own demo exactly; GPU dynamics is another physics realisation. GPU memory +1.1-1.2 GB per
    process.
  - Decision: keep CPU PhysX (the switch stays off).
- **F2, a vectorized oracle collector** (N scenes per process with GPU physics): not built.
  - It needs GPU PhysX, which is slower here for one scene, and the vec_next world build (merged by another session).
  - The multienv report (10-03) measured several environments in one process at 15.5 env-steps/s against 16.4 for separate processes.
  - Collectors are idle most of the time in the sampler-bound families (E15), and the same "more scenes per GPU" comes from more
    collector processes, which run at the alone speed when pinned to their own CPUs (E16).
  - Revisit if pinned collectors become the bottleneck of the collector-bound families.

### Production-configuration fills (stopped)

- Tags `pilot2` (27 aliases on viscam L40S, 93 % demo success by 12:34), `pilot3`, `pilot4` (`refill.py`, stop file `refill.stop`).
- All cancelled on 11-01 15:00. Not curated as training data.

## 7. Open items

- BAND_REFINE with `nearest` vs off: a job-level answer needs more jobs, not longer ones (E14). The new defaults together
  (nearest, TILT_SNAP off) were not run as an arm.
- Memory: one of 18 jobs at 32 GB was killed for running out of memory (sampler + 2 collectors).
- inside_02's pre-place gripper-in-frame limit (E11).
- RECORD_CHANGED_ONLY changes the physics of some full-scene instances (E13). Left as is: collections use the cropped path. If
  full-scene demos are needed, rewrite it to decide "changed" from the dumped state.
- block_design_01 board placement; inside_large_scale_03 `no_init_surface`.
- disembedding_04 / 07-10 have no train85 template copy, so they were never submitted.
- Low-yield aliases on both crop and full: nextto_01, touching_02, inside_01, axial_04.
- isg-1020 implications:
  - the split-cone collapse (E8);
  - bands written back into the eval85 / train85 snapshots;
  - static-object recording in full-scene demos (E13).
- Why the collectors of two unpinned pair members did not slow each other (pair1 on L40S) while two of one job did (E16).
- `cpu_split.py` takes the job's CPUs in ascending order, which on the svl / viscam nodes puts physical cores first; on a node that
  numbers hyperthread siblings next to each other a collector would get fewer cores. Check the `PIN` line of `job.txt`.
- Two samplers per TITAN RTX do not fit disembedding (cuRobo out of memory, E16); per-family packing on TITAN is not measured
  beyond the three fleet keys.
- Pinned pairs were run once, for disembedding_03 on L40S. The sampler-bound families (ontop, mug) ran in pairs only unpinned.

## 8. Pitfalls met

- **Duplicate sessions.** After a connection loss the editor resumed the session twice and both copies edited and submitted. On
  resume, check `pgrep -af "claude.*--resume"`.
- **NFS.** `flock` returns EIO there (the ledger lock is best-effort). Three or more parallel ssh sessions drop.
- **Profiling.** `py-spy record` segfaults on the Isaac Sim process. Use the demo `meta.json` timings.
- **Long-running work on svl11.** Background processes on the login node die (4 parallel video decodes were killed). Run analyses that
  decode video as small CPU `sbatch` jobs.
- **Lambda GPU.** The lambda 4090 is shared with another session. Test on the cluster.
- **`job.env` names.** A harness that sources a job's `job.env` inherits its variables: it sets `N` (the instance count), so a test
  script that read its own `N` before sourcing started 24 collectors per job (E16's first contention run, cancelled). Read arguments
  into names `job.env` does not use, after sourcing it.
