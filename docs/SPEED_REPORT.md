# train-fast speed test (2026-10-31, interim - numbers updated as the fleets finish)

Written for: the owner of the isg-1020 scale-up (who knows the sampler / oracle / replay pipeline).

## What was asked

10-20x isg-1020 (54k-108k demos) in about one day on ~80 TITAN RTX. Production (train85) made 4.6 successful demos per GPU-hour;
10x in 24 h on 80 GPUs needs 28 per GPU-hour, 20x needs 56. Rendering is budgeted separately (the replay stage).

## What was built (branch `train-fast`, worktree `ispatialgym-train-fast`; design in `docs/FAST_FLOW.md`)

1. **Job flow**: one GPU runs the sampler and K physics-only collectors concurrently (`scripts/job_fast.sh` in the campaign root
   `isg_home/trainfast`), collectors claim instances as they land; demos are hdf5 only; images come from the kinematic replay
   (`ispatialgym.multienv.replay --watch`, measured 6.3-6.5 env-steps/s per TITAN RTX slot).
2. **Existing opt-ins turned on** (production left them off for bit-identity): snug cuRobo mesh stores for sampler and executor,
   step-1-once, seeded sphere-fit cache, goal occlusion moved out of the sampler/oracle, head camera only, physics-only collection.
3. **Branch switches** (`ISPATIALGYM_FAST_*`, all default on): early train-vs-eval repeat filter, park-cell cache and partial cuRobo
   syncs, camera-free sampler worlds for the pixel-free families, the start-per-attempt draw memoised per lateral cell (in-process and on a
   shared NFS cache), batched-IK solver sized to the query, single 32-seed re-solves instead of padded batches, return-all IK picks the
   joint-limit-clear branch, remembered witnesses accepted with a seed, program-check early frame gate and partial syncs, physics-only
   trim (no texture wait / init image / reset renders, 128 px cameras), straight joint-space ramps to the IK witness (direct or through
   the ready / untucked posture) checked in one batched collision call before trajopt, a shorter plan ladder (2 trajopt seeds, 2 attempts,
   1 retry), band feasible-set reuse, attach geometry cache. Instrumentation: per-attempt stage times and cuRobo counters (IK, batched
   IK, differential IK, collision, world sync, attach) in the sample log and summaries; straight-arm counters in meta.json.

## Fleets (same 14 aliases, TITAN RTX, `-A vision -p svl`)

| tag | code | flow | n per job | notes |
|---|---|---|---|---|
| base | frozen 43ad206a + train patch | train85: sampler, then rendered 540 px demos, sequential | 4 | the A side |
| flags | same frozen code | fast flow (sampler + 2 physics-only collectors) + the existing opt-ins | 12 | isolates flow + opt-ins |
| fast | train-fast branch | fast flow, all switches | 12 | round 1 |
| fast2 | train-fast branch | round 2: 5 cm start-draw cells, 8 screening seeds, band reuse, via ramps | 12 | |
| fast_k1 / fast_split | train-fast branch | one collector per GPU / sampler and collector on separate GPUs | 12 | packing tests |

Four aliases (alignment-axial_01, attachment_02, block_design_01, iors-inside_02) produced no instance in any tag (50-140 rejections
each): a sampler-yield problem, not a speed one; they were dropped from the later rounds.

## Results (interim, `scripts/fleet_summary.py`, `scripts/stage_summary.py`)

TBD - filled from the final summaries. Interim at 20:30 PDT: steady-state successes per GPU-hour (after each job's world load and
collector start) fast 23.6 over the 10 productive aliases (IORS 10-23, mug 25, wall clock 59, bell pepper 246); whole-job rates at
n = 12 are about a third of that because a 12-instance job spends its first 10-20 minutes loading worlds (sampler 400-500 s, each
collector another 420-580 s, started only when the first instance lands).

Per attempt (sampler): base 90 s -> fast 48-59 s over all aliases; IORS nextto 110 -> 78-100, ontop 114 -> 95-138, under 136 -> 103-175
(the fast samplers share their GPU with two collectors; the flags samplers on the same hardware were 1.3-1.5x slower than base).

Per demo (collector): rendered base alone 0.45-0.5 s/step; physics-only under 3-process contention 0.3-0.4 s/step; alignment-world_vertical
demos on the fast code 0.16 s/step (straight ramps replace every trajopt plan there). Trajopt with 2 seeds: 18-30 s per plan under
contention (base alone 30-47 s with 4 seeds). Straight ramps: clear on about a quarter of IORS pre-grasp transits in round 1; round 2 adds
two-leg ramps through a posture.

## What still costs (fast code, IORS, per attempt)

- the start-per-attempt draw while its cell cache warms (21-60 s mean per attempt in 12-instance jobs; 5 cm cells in round 2; the cache
  is shared across jobs of a template on NFS, so a long campaign amortises it)
- the two feasible-set IK screens per candidate (9-26 s per call under contention; band reuse in round 2 removes them for cells seen
  before)
- 3 full cuRobo world syncs per attempt (5-18 s), attach sphere fits (2-16 s; geometry cache + rounded fit key in round 2)
- GPU contention: three Kit processes on one TITAN RTX slow every cuRobo call 1.5-2x (see the packing tests)

## Replay stage

6.3-6.5 env-steps/s per TITAN RTX slot (one slot, batched USD edits), load 220-230 s + 60 s warm-up; ~80 GPU-s per 500-step demo
-> ~1,200 TITAN GPU-hours for 54k demos. The physics-only-trim hdf5 (128 px live cameras, header at 1080) replays identically.

## Cluster constraints found

The svl QOS caps one user at 48 GPUs / 56 jobs; viscam at 30 GPUs (L40S 48 GB, A5000/A6000, H200, rtxpro6000, one TITAN RTX node).
At most 78 GPUs at once for this account. py-spy segfaults on the Isaac Sim process on these nodes.

## Projection and recommendation

TBD after the packing tests: whether sampler + 2 collectors per GPU (contended), sampler + 1 collector, or sampler and collector GPUs
split over a shared NFS queue gives the most successes per GPU-hour; then jobs of 50+ instances per (alias, scene, room) so the
7-8 min loads and the per-template caches amortise.
