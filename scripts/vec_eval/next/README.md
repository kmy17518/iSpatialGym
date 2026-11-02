# vec_next: what changed, what was run, how to replicate it

Campaign vec_next (2026-10-29 .. 2026-11-01) built one vectorized evaluator (N scenes in one Isaac Sim world) carrying every fix of the
earlier campaigns, verified it alias by alias at N=8 against N=1, made the world build faster, and changed how an unsettled success is
judged. This file is the index for replicating it: the code (by commit), the setup, the verification protocol, every round and experiment
with its script, command, output and result. Design detail per change: [docs/VEC_NEXT.md](../../../docs/VEC_NEXT.md) (§1 table, up to
round D). The chronological log with every decision is the campaign's `DECISIONS.md` on the cluster (see §2).

Final state: **49 of 53 aliases PASS strict, 51 relaxed** at N=8 (rounds D, E, F); slot 0 bit-identical to N=1 for the whole episode in
24 of 53; N=8 world build 17 % faster than round E (round F), up to 2.6x more throughput per GPU with 4 processes x 2 scenes.

## 1. Code

Branch `vec-eval-next` of the iSpatialGym-dev repo, on f4c0d23c. Commits on lambda (not pushed):

| commit | change | switch (default) |
|---|---|---|
| cdaf196b | The integrated evaluator: driver `ispatialgym/oracle/run_isg_vec.py`, `vec_loader.py`, `tasks/vec_task.py`, `vec_stream.py`, rooms override (`vec_rooms.py`, `run_isg.rooms_for_template`), `vec_video.py`; Behavior-1K overlay `scripts/vec_eval/b1k_overlay/` (prototype-once asset preparation, lazy particle systems, batched Fabric sync for clones, collision-mesh cache, assisted-grasp shared contact view, build timers); physics: rigid sleep threshold 0.005 on free non-articulated objects + the driver's sleep assist; verification tools (`compare_slots.py`, `verify_rules.py`, `summarize.py`, `verification_table.py`, `rooms/`); probes and profilers. = the round D code. | see docs/VEC_NEXT.md §1; `--rigid-sleep-threshold 0.005`, `--sleep-assist 30,1,3` |
| a54d117a | Batch restore without `Scene.reset`'s world physics step (it stepped every scene once per slot restore); the sleep assist's rest window restarts while an object touches the robot. | `ISG_BATCHED_SCENE_RESET=1`; probe `ISG_RESTORE_PROBE=<jsonl>` |
| 23ac6392 | World-build speed: clone caches (local volume / COM / oblong, world volume, collision mesh CPU data) and lazy clone emitters in the overlay; PhysX replicator switch (not adopted); play sub-timers; `scripts/vec_eval/next/mp_launch.py` (K processes x N scenes on one GPU). | `ISG_VEC_MESH_CACHE=1`, `ISG_VEC_LAZY_EMITTERS=1`, `ISG_PHYSX_REPLICATE=0` |
| 109087e4 | Persistent content-keyed asset geometry cache on disk (overlay `utils/isg_asset_cache.py`). | off; `ISG_VEC_ASSET_CACHE_DIR=<dir>` (`ISG_VEC_ASSET_CACHE_MODE`) |
| 1979fee5 | `scripts/vec_eval/next/step_timing.py`: where a control step goes. | - |
| e8862eb1 | Evaluated success step: an UNSETTLED success is judged at the lowest-error step of the W steps after its first step; a settled success keeps its first settled step; `*_first_step`, `evaluation_*`, `success_final`; stop rules wait for `success_final`. | always |
| 5cce13e4 | `closest_approach` / `closest_goal_approach` reported, deciding nothing. | always |
| (this commit) | This README; `verify.sbatch` as run in rounds E / F (node cache on `/scr`, `mp` phase); `jobs/`, `inputs/`, `experiments/`. | - |

The Behavior-1K changes live only in the overlay; the checkouts that ran them are stock 6a6194600 plus the overlay copied in, uncommitted.
The pre-change copies of every file of the last five changes are on the cluster in `$V/campaigns/vec_next/patches/<date>_<change>/{before,after}`.

## 2. Where everything is

| what | cluster (ssh `minyeong@svl11`) | lambda (RTX 4090, Ryzen 7950X) |
|---|---|---|
| evaluator worktree | `/vision/u/minyeong/projects/ispatialgym-eval-vec-next` (git worktree of `ispatialgym-eval`, branch at f4c0d23c, the same files uncommitted) | `~/Desktop/projects/ispatialgym-eval-vec-next` (this repo; commits above) |
| Behavior-1K | `Behavior-1K-vec-next-6a6194600` (round A), `Behavior-1K-vec-next-b-6a6194600` (rounds B-F) under `/vision/u/minyeong/projects/` | `~/Desktop/projects/Behavior-1K-vec-next-6a6194600` (`Behavior-1K` symlink in the worktree) |
| record | `$V=/vision/u/minyeong/projects/isg_home/vec_eval85`; campaign `$V/campaigns/vec_next/`: `DECISIONS.md`, `runs_<round>/<alias>/{n1,n8same}`, `verify_<round>/<alias>.json` + `SUMMARY`, `experiments/`, `patches/`, `logs/`, `jobs/`, `inputs/` | outputs `~/vec_eval85/next/<experiment>/` (§6) |
| instance lists | `$V/instances/eval/<alias>.json` | `~/vec_eval85/instances/eval/<alias>.json` |
| caches | `/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/` (Kit app-data templates, textures, `asset_geometry_cache/`) | warm Kit app-data `~/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata` |

Sync rule: edit on lambda, copy changed files to the cluster worktree file by file. Never rsync this folder wholesale: its `.git` directory
would replace the cluster worktree's `.git` file.

## 3. Setup

1. Python env `ispatialgym-vectorized-envs` (conda). On lambda `source scripts/vec_eval/env_lambda.sh` (set `ISG_EVAL_ROOT` to this
   worktree first: it sets `PYTHONPATH`, `OMNIGIBSON_DATA_PATH`, headless Kit, `PYTHONHASHSEED=0`, `ISG_VEC_ROOT`).
2. Behavior-1K at 6a6194600 (a git worktree), datasets linked as in the stock checkout, then
   `scripts/vec_eval/apply_b1k_overlay.sh <Behavior-1K dir>`; link it as `<worktree>/Behavior-1K`.
3. Instance lists: `python scripts/vec_eval/build_instance_lists.py --out <root>` writes `<root>/instances/eval/<alias>.json` (the eval
   dataset's instance + demo per alias). The 53 aliases: [inputs/aliases.txt](inputs/aliases.txt).
4. Rooms override: `ISG_ROOMS_OVERRIDE=scripts/vec_eval/rooms/rooms_override.json` (14 aliases; same file as the campaign's
   `inputs/rooms_override.json`).
5. Kit app-data (shader / texture cache) is per Behavior-1K checkout. A cold one adds minutes to the first build of every scene: point
   `OMNIGIBSON_APPDATA_PATH` at a warm copy before timing anything (the 2026-10-30 00:05 confound).
6. Cluster jobs need the R2 credentials file that `verify.sbatch` sources (not in the repo, never printed) for the asset staging, and
   a GPU driver 570-580 (the job exits otherwise).

One replay (the unit of everything below):

```bash
python -u -m ispatialgym.oracle.run_isg_vec --mode replay --instances <alias.json> --num-envs 8 --same-instance-all-slots \
  --record-view tiled --grid-video --tile-px 240 --external-size 240 --sleep-assist 30,1,3 --out <dir>
# N=1 reference: --num-envs 1 (no --same-instance-all-slots). Defaults: --layout grid-centered --scene-margin 5, --rigid-sleep-threshold 0.005
```

## 4. Verification protocol

Per alias, on one GPU, in one job: `n1` (the eval instance at N=1, the reference) -> `n8same` (the same demo in all 8 slots) ->
`compare_slots.py <n8same> --ref <n1>` + `verify_rules.py` -> `verify_record.py` composes `verify_<round>/<alias>.json`.

- **Strict pass** (the 10-23 rule): every slot has the reference's verdicts and success steps within 5 steps, trajectories within 2 cm /
  0.05 rad, joints angle-wrapped. **Relaxed**: the rotation rule counts only above 0.3 rad (a real tip-over). `comparator_only` = every slot
  reproduces the verdicts and steps and only the strict rotation rule fails.
- Success steps compared are the FIRST steps (`*_first_step`) since e8862eb1.
- **N=1 parity** (`compare_n1` phase): this round's n1 vs the previous round's n1, same rule; host-confounded for chaotic stacks (CPU PhysX
  differs between CPUs), so the same-job N=8 vs N=1 comparison is the clean gate.

Cluster: [verify.sbatch](verify.sbatch) (one alias per job; `ISG_TASK` or `ISG_TASK_LIST` + array index; `ISG_PHASES=n1+compare_n1+n8+compare`,
`+` not `,`: sbatch `--export` splits on commas; `ISG_B1K_DIR`, `ISG_RUNS_DIR`, `ISG_VERIFY_DIR`, `ISG_REF_RUNS_DIR`, `ISG_EXTRA_ARGS`,
phase `mp` with `ISG_MP_LAYOUTS=1x8+2x4+4x2`). Round scripts: [jobs/roundD.sh](jobs/roundD.sh), [jobs/roundE.sh](jobs/roundE.sh),
[jobs/roundF.sh](jobs/roundF.sh). Lambda: [experiments/lambda/run_roundD_lambda.sh](experiments/lambda/run_roundD_lambda.sh) `<alias>...`.
Summaries: `experiments/cluster/summary_{d,e,f}.py <verify dir>`.

Cluster notes: QOS `svl` caps 48 GPUs per user (`scontrol update JobId=<id> ArrayTaskThrottle=53`, overflow on `-A viscam -p viscam
--constraint=48G`); svl jobs have a private /tmp, so the asset cache lives on `/scr/minyeong` (one ~30 GB copy per node); viscam
preemption requeues by itself; submit with sbatch (srun from svl11 hangs).

## 5. Rounds (53 aliases at N=8 unless noted)

| round | code | where | strict / relaxed | notes |
|---|---|---|---|---|
| A | change-1 overlay (prepare cache, lazy systems) | viscam | 28 of 35 | interrupted for change 2 |
| B | + batched Fabric sync, mesh cache; stock sleep threshold 5e-5 | viscam + lambda | 47 / - | FAIL: teapot lid, cap, sprouts, sunflower, kielbasa (trembling), tablespoon (comparator) |
| C | + rigid sleep threshold 0.005 (scene objects only, by mistake) | viscam + lambda | 47 / 50 | |
| D | + threshold on task objects + sleep assist | viscam + lambda | **49 / 51** | parity vs B 50/53. FAIL sunflower (lean set at the placement), kielbasa (the demo's own settle margin); comparator-only cap, tablespoon. The first two attempts are `{runs,verify}_D_sleepmode` |
| E | + restore fix (a54d117a) | svl Titan RTX + viscam | 49 / 51 | parity vs D 51/53; slot 0 = N=1 all episode 24/53 (D: 0/53) |
| F | + option 3 clone caches (23ac6392) | svl + viscam | 49 / 51 | parity vs E 53/53; N=8 build -17 % (median F/E 0.84) |

Not run as a round: the persistent asset cache (109087e4) and the metric change (e8862eb1, 5cce13e4); both were verified by replays (§6.6, §6.8).

## 6. Experiments

Scripts are archived as run in [experiments/lambda/](experiments/lambda/) (lambda drivers; paths under `~/vec_eval85`) and
[experiments/cluster/](experiments/cluster/) (analysis scripts that read `runs_*`). Some were edited between runs; DECISIONS.md has the
exact sequence. Outputs: lambda `~/vec_eval85/next/<dir>`, cluster `$V/campaigns/vec_next/experiments/<dir>`.

### 6.1 World build (2026-10-29/30)
- Baseline vs change 1 vs change 2, apple / hamper N=1 / N=8 (`run_chain*.sh`, `run_timers.sh`, `run_validate*.sh`, `run_control.sh`,
  `table.py`; `baseline*/`, `after*/`, `timers/`, `validate/`). Apple N=8 595 -> 512 -> **355 s**, hamper N=8 1041 -> 955 -> **622 s**;
  per clone ~55 s -> ~30 s. Profile (py-spy `record -- python`; cProfile is blind past the Kit launch): 42 % full-stage Fabric syncs, 20 %
  trimesh. Change 2 leaves scene 0 bit-identical; clones differ <= 1e-5 m (control: the build itself is deterministic).

### 6.2 Small-object trembling (2026-10-30)
- Placement offsets (`run_offsets.sh`, `offset_drift.py`; kielbasa N=2 with `--scene-offsets`): not float32 distance from the origin.
- Physics knobs (`run_knobs.sh`, `run_sleep_sweep.sh`, `run_trace.sh`; `knobs/`, `sleep_sweep/`, `trace/`): sleep threshold 0.005 freezes
  the stack; 128 position / 8 velocity iterations do not; stabilization segfaults.
- Teapot stack probes (`scripts/vec_eval/probe_stack.py`, `probe_sleep.py`, `run_probe_stack.sh`; `stack/`, `probe/`): convex-hull chatter
  never sleeps at any threshold; iterations only move the odds; a forced sleep sticks. Attachment edits / filtered pairs mid-play
  invalidate the PhysX views.
- Sleep assist validation (`run_assist_test.sh`, `run_assist_alias.sh`; `assist*/`): sleep mode + moving-gripper wake at 5 cm from the box,
  5 kg cap, directional supporter rule. Teapot 3/8 -> 8/8, sprouts 8/8, kielbasa marginal by its own demo. Analysis:
  `assist_window.py`, `rel_probe.py`, `sleep_check.py`, `roll_probe.py`, `diverge_probe.py`, `ulp_check*.py`.

### 6.3 What perturbs the clones: the restore (2026-10-31)
- `run_restore_probe*.sh` + `restore_probe_cmp.py` (`ISG_RESTORE_PROBE`; `restore_probe/`, `restore_probe_fixed/`): every slot started
  3e-5..3e-4 off N=1 because `Scene.reset` steps the whole world per slot restore. After the fix the sim clock equals N=1 and slot 0 ends the
  restore bit-identical. Analysis: `step0*.py`, `first_div*.py`, `why_fail.py`, `fail_detail*.py`, `spoon_rel.py`, `kiel_bc.py`.

### 6.4 Residual cross-scene effect (open)
- Sunflower probes (`run_restore_probe_var.sh`, `run_restore_probe_idle*.sh`): N=2 stays bit-identical to N=1; with >= 3-4 scenes the first
  world step moves slot 0 (2e-5) even when the other slots are idle. Broadphase SAP is not a fix (`run_sap_test.sh`; `sap/`: SAP changes N=1
  itself). Slots 1-7 also start 1-13 um off from float32 rounding of the clone offsets. Unexplained; contact events amplify it.

### 6.5 Build options 1-3 (2026-10-31; `run_options_queue.sh`, `run_after_options.sh`, `run_profile2.sh`)
- **Option 1, K processes x N scenes** (`run_mp.sh` + `make_appdata_copies.sh` on lambda, `mp/`; cluster `verify.sbatch` phase `mp`,
  `mp_cluster_{titan,l40s}`). Wall time for 8 scenes of apple:

  | GPU | 1x8 | 2x4 | 4x2 | 8x1 |
  |---|---|---|---|---|
  | lambda RTX 4090 24 GB | 466 s | 249 s | **178 s (2.6x)** | out of memory |
  | L40S 48 GB | 1143 s | 679 s | **549 s (2.1x)** | 598 s |
  | Titan RTX 24 GB | 2195 s | 1124 s | **840 s (2.6x)** | - |
- **Option 2, PhysX replication** (`ISG_PHYSX_REPLICATE=1`): build 357.5 vs 360.0 s, state changes up to 4e-6 in every scene: dropped.
- **Option 3, clone caches + lazy emitters** (`run_opt3_test.sh`, `run_opt3_replay.sh`; `options/`, `opt3/`): apple N=8 360 -> 292 s; N=1
  bit-identical; caches off = old code bit for bit; round F.

### 6.6 Persistent asset cache (`run_assetcache_test.sh`; `asset_cache_test/`, `options/ac_*`)
- Cache off / fill / read bit-identical (apple N=1, N=8, hamper N=1). Apple N=1 65.5 -> 53.7 s, hamper N=1 106.0 -> 84.9 s, apple N=8
  292 -> 272 s. On the cluster: synced, off, never exercised.

### 6.7 Stepping (`step_timing.py`; `step_profile/`)
- Apple, lambda, per control step (4 physics substeps):

  | N | physics | render | sim step | env step |
  |---|---|---|---|---|
  | 1 | 11.8 ms | 25.5 ms | 52.9 ms | 59.6 ms |
  | 8 | 47.5 ms | 190.6 ms | 189.3 ms | 243.9 ms |

  Rendering dominates (RealTimePathTracing, 1080² head camera + the external camera; GPU at 99-100 %); task checks ~7 ms per scene. Running
  the checks less often is not parity-exact (late detection, attachment predicates create joints when evaluated, per-step settle history).

### 6.8 Evaluated success step and closest approach (2026-11-01)
- Spec: [ispatialgym/docs/task_definition_and_evaluation.md](../../../ispatialgym/docs/task_definition_and_evaluation.md). Settle-eval off:
  lowest max(error / tolerance) over the `:measure` atoms among [first, first + W - 1] where the kind's condition holds (W = settle_steps,
  ties earlier); settle-eval on: unchanged. Policy runs with settle-eval off run W - 1 steps past the first success.
- Checks: unit tests (`test_scoring_metrics.py`, `test_vec_stream.py`, `test_sim_evaluator.py`, `test_v2_oracle.py`); full suite 1822
  passed / 6 failed (the same 6 fail on the unchanged code); apple N=1 + N=8 replays before / after (`run_metric_replay.sh`,
  `metric_compare.py`; `metric_change/`): trajectories bit-identical, verdicts and settled step 73 unchanged; closest goal approach step 24
  (1.534 mm) vs the judged 1.566 mm.
- Safety check, stock single-scene evaluator (`stock_check/`): the production worktree `ispatialgym-eval-vec`, `python -m
  ispatialgym.oracle.run_isg --instance <tro_state.json> --mode replay` on stock Behavior-1K 6a6194600 gives the same 75 steps bit for bit,
  error 1.5659 mm at step 73. The error is measured against the sampled goal pose, not the demo, so a replay is not 0.

## 7. Open items
- The >= 3-scene PhysX effect (§6.4).
- Sunflower (lean at the placement) and kielbasa (the gripper brushes the pack at step 749; the settle window barely fits) fail at N=8 by
  their own physics; cap and tablespoon pass relaxed only.
- Not yet run as a round: the asset cache on the cluster, option 1 inside the verification jobs.
- Stepping: rendering is the cost; the exact task-check speed-ups (incremental window drift, skipping unused per-step reports) are not done.

## 8. Pitfalls met
- `creds.sh` defines a shell variable `W`: never name a job variable `W`.
- cProfile is blind past the Kit launch; py-spy must launch the process (`ptrace_scope` 1).
- `pkill -f <pattern>` kills the calling shell when the pattern is in its command line: kill by PID.
- The co-location `robots/robot.py` overlay needs `utils/scene_query_utils.py`.
- 8 processes x 1 scene does not fit a 24 GB GPU.
- `run_metric_replay.sh` and `run_opt3_replay.sh` kept the SAP script's header comment; their bodies are the metric and option 3 replays.
