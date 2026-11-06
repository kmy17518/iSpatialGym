# train-fast speed report (final, 2026-11-02)

Written for: the owner of the isg-1020 scale-up (who knows the sampler / oracle / replay pipeline). The experiments behind each number
are in [TRAIN_FAST_README.md](TRAIN_FAST_README.md) (E1-E17); the switches in [FAST_FLOW.md](FAST_FLOW.md).

## Bottom line

- On TITAN RTX the train-fast code collects **2.9-4.3x** more successful demos per GPU-hour than production (train85) at train85's own
  alias mix: 9.7-10.9 against 2.5-3.3 on the same aliases. On L40S it collects **7.4x** (21.7 against 2.9), but that includes the
  faster GPU (no alias ran on both GPU types, so hardware and code are not separated).
- 10x isg-1020 (54k demos) is **not reachable in one day** on this account's 78 GPUs at these rates: about 1,120-1,170 successes per
  hour on 48 TITAN + 30 L40S, so ~27-28k per day and 54k in about two days (about 1.6 days if jobs run long enough that the world load
  stops mattering).
- **The sampler is the bottleneck.** The collectors are busy 29 % of their time in the median job and the GPUs are 8-39 % used. A
  sampler attempt costs 28-40 s instead of 109 s, but 61-80 % of sampler time is still rejected attempts, mostly `target_placement`.
- The collector's own costs therefore barely matter for throughput. The per-step state dump is 1-3 % of collector time in cropped
  scenes; a narrower dump or sync set was not built.
- **Collectors of one job slowed each other ~4x** (11-02). Two unpinned Kit processes on one job's CPUs step at 334-374 ms against 95
  alone, whatever the CPU count. Pinning each collector to 3 CPUs of its own (COLLECTOR_PIN, now on by default) brings them back to
  the alone speed. In the collector-bound disembedding_03 it raised successes per GPU-hour **2.4x on TITAN (12.9 -> 30.6-32.6) and
  3.6-4.9x on L40S (13.0-14.7 -> 52.7-64.0)**, and made its sampler 2x faster. Two pinned disembedding jobs on one L40S made
  **104.6**. The rates above were measured unpinned; the sampler-bound families gain less (see Packing).
- GPU PhysX (follow-up F1) is 1.2-2.1x slower per step than CPU PhysX for one scene, so it stays off, and the vectorized collector
  (F2) was not built.
- Next steps, by expected gain: two pinned jobs per L40S for every family (measured only for disembedding), cheaper
  `target_placement` rejections, and longer jobs (50+ instances).

## What was asked

10-20x isg-1020 (54k-108k demos) in about one day on ~80 GPUs. Rendering is a separate stage (the kinematic replay) in both pipelines:
production's 540 px collector videos were not the training images either.

**Production rate, corrected.** train85 made 7,183 successful demos in 2,404 GPU-hours of job wall (487 jobs, 1 GPU each), i.e.
**2.99 per GPU-hour**. The 4.6 quoted on 10-31 used 1,551 GPU-hours, and how that figure was derived is not recorded. Every speed-up
below is against train85 on the same alias keys, so it does not depend on either overall figure.

## What was built

Branch `train-fast`; the commit table is in TRAIN_FAST_README.md section 2. In short:

1. **Job flow:** one GPU runs the sampler and K physics-only collectors concurrently (`scripts/trainfast/job_fast.sh`); the collectors
   claim instances as they land, prebuild their world while the sampler loads, and write hdf5 only. Images come from the replay.
2. **Existing opt-ins turned on** (production left them off for bit-identity): snug cuRobo mesh stores, step-1-once, the seeded
   sphere-fit cache, goal occlusion out of the sampler / oracle, head camera only, physics-only collection.
3. **Branch switches** (`ISPATIALGYM_FAST_*`):
   - Sampler:
     - the early train-vs-eval repeat filter on the measured rule;
     - park and start-draw caches;
     - batched IK sized to the query;
     - band evidence and refinement;
     - partial cuRobo syncs;
     - camera-free worlds for the pixel-free families;
     - the split-cone restore.
   - Collector: straight joint-space ramps, a shorter plan ladder, the attach geometry cache, the physics-only trim.
   - World: scene crop (rooms / region) with neighbour floors; partial-scene record and replay.
   - Camera relocalization: the geodesic field and the obstacle prefilter.

## How it was measured

- Rate = successes per GPU-hour over **whole jobs** (job start to end, world loads included). The job wall comes from `ISGF_END`, or
  slurm's Elapsed for a job cancelled before it. Successes come from `results.jsonl`.
  - Until 11-01 20:40, `fleet_summary.py` charged a cancelled job until "now", which inflated GPU-hours for the cancelled tags (base,
    flags, fast, fast2, pilot3, pilot4). It now uses sacct.
- Speed-up per tag = its pooled rate on the alias keys it ran ÷ train85's pooled rate on the same keys (`speed_compare.py`).
- Production-mix speed-up = the GPU-hours train85 spent on a set of keys ÷ the GPU-hours the fast code would need for the same per-key
  successes (`projection.py`). It weights each key by what production actually made of it.
- "Steady" rates (first to last finished demo of a job) leave out the world load; they show what longer jobs would approach.
- All scripts are in `scripts/trainfast/` (snapshot) and `$F/scripts` (live).

## Results

### Per tag, against train85 on the same keys

| tag | code / layout | GPU | keys | succ / GPU-h | train85 same keys | speed-up | steady |
|---|---|---|---|---|---|---|---|
| base | frozen 43ad206a, train85 flow, n=4 | TITAN | 14 | 2.1 | 3.2 | x0.7 | 7.4 |
| flags | frozen code, fast job flow + opt-ins, n=12 | TITAN | 14 | 6.8 | 3.2 | x2.1 | 16.7 |
| fast | branch round 1, n=12 | TITAN | 14 | 8.8 | 3.2 | x2.8 | 21.9 |
| fast2 | round 2, n=12 | TITAN | 14 | 10.7 | 3.2 | x3.4 | 22.7 |
| pilot | 10 productive keys, n=40, 24 CPUs | TITAN | 10 | 13.5 | 4.0 | x3.4 | 15.9 |
| crop2 | scene crop (fixed floors), n=24 | TITAN | 9 | 16.5 | 3.7 | x4.4 | 25.3 |
| crop4 | scene crop, 21 keys, n=24 | TITAN | 21 | 11.2 | 2.5 | x4.5 | 13.8 |
| pilot3 | 26 keys, n=24 (half cancelled at 15:00) | TITAN | 26 | 10.5 | 3.9 | x2.7 | 15.7 |
| refine_off / refine_nearest | latest code, ontop_small_01 + touching_01, n=24 | TITAN | 2 | 8.4 / 9.9 | 1.7 | x4.9 / x5.8 | 9.6 / 11.2 |
| pilot2 | 27 keys, n=24, 12 CPUs | L40S | 27 | 24.1 | 2.9 | x8.3 | 29.5 |

- `base` ran short n=4 jobs that pay a full world load per 4 instances, which is why it is below production.
- Per-key speed-ups vary widely: x2.0-x31.8 in crop4 (touching_large_03 gains the most). No success at all on the fast code:
  inside_large_03 (TITAN) and ontop_small_scale_04 (L40S). disembedding_03 made none in crop2 but did in other tags.

### At production's alias mix

| set | keys | train85 | fast code | speed-up |
|---|---|---|---|---|
| TITAN, crop2 + crop4 + refine | 19 | 2.54 | 10.9 | x4.3 |
| TITAN, those + pilot3 | 43 | 3.32 | 9.7 | x2.9 |
| TITAN, pilot3 alone | 24 | 4.36 | 8.9 | x2.1 |
| L40S, pilot2 | 24 | 2.93 | 21.7 | x7.4 |

- pilot3's keys are ones production already did comparatively well (4.36 per GPU-hour), so the gain there is smaller.
- Largest GPU-hour shares at the TITAN mix: camera relocalization obstructed (3.2-4.2 per GPU-hour), large articulation close,
  disembedding_01, under_small_04.
- Excluded: inside_large_03 made no fast success (train85: 100 successes in 85 GPU-hours) and is not in these rates.

### Projection for 10x

- 54k demos in 24 h on 78 GPUs needs 28.8 successes per GPU-hour at this mix.
- Measured: TITAN 9.7-10.9, L40S 21.7. 48 TITAN + 30 L40S give 1,117-1,174 successes per hour, i.e. 26.8k-28.2k per day: **~5x
  isg-1020 per day, 10x in ~2 days**.
- With long jobs (steady rates ~1.2x whole at n=24) about 32-34k per day, 10x in ~1.6 days.
- Caveats:
  - These rates cover the 43 (TITAN) and 24 (L40S) keys that were run, not all 78.
  - Three keys produced nothing on the fast code (inside_large_03, ontop_small_04, block_design_01) and need their own fixes.

## Where the time goes now

### Sampler (the bottleneck)

| | train85 | crop4 (TITAN) | refine_nearest (TITAN) | pilot2 (L40S) |
|---|---|---|---|---|
| s per attempt | 109 | 40 | 29 | 28 |
| attempts per written | 6.0 | 5.8 | 9.4 | 4.1 |
| rejected attempts' share of sampler time | 79 % | 66 % | 80 % | 61 % |
| `target_placement`: share of rejections (of attempts) | - | 75 % (62 %) | 93 % (83 %) | 69 % (52 %) |
| world load | - | 253 s | 231 s | 188 s |

Per attempt (crop4, means): sample stage 31 s, settle 10 s, goal check 10 s, program check 8 s. The start-per-attempt draw now
costs 4-8 s (19 s in production) and on L40S 0.8 s.

### Collectors (mostly idle)

- **Busy share** (`job_balance.py`, 78 pilot / crop jobs with k=2): demo time ÷ collector wall, median **29 %** (p10-p90 14-78 %).
  - By family: camera relocalization 10 %, mental rotation 16 %, object scaling 28 %, IORS 34 %, configuration matching 53 %,
    alignment 69 %.
  - Only large IORS (86 %) and disembedding (78-88 %) keep their collectors busy. Large IORS and disembedding_03 jobs run
    2,300-2,700 s after their sampler ends; the other families finish 40-160 s after it.
- **Per-step costs** (`step_costs.py`, 2,021 demos with the always-on timers), share of collector episode time:

  | env step (physics + obs + task) | QA monitors | planning + rest | world syncs | state dump | recorder | other monitors |
  |---|---|---|---|---|---|---|
  | 59 % (102 ms/step) | 13 % (22 ms) | 15 % | 7 % (2.5 s per sync, 3.7 per demo) | 3 % (5 ms) | 3 % (5 ms) | 0.5 % |

  - The exceptions are full-scene families:
    - Camera relocalization: dump 14 %, 1,200 floats per step.
    - Alignment: world syncs 24 % (5.5 s per sync).
    - Large configuration matching: dump 7 %, syncs 12 %.
  - Disembedding's env step is 483 ms (83 % of its time). That is the contention between the job's two unpinned collectors, not the
    physics: one collector alone steps at 95 ms, pinned ones at 63-92 (Packing).

### GPUs (mostly idle)

- `gpu.log` (one nvidia-smi sample per minute): mean utilisation, median over jobs:
  - IORS tags 29-39 %;
  - the mixed pilot tags 8-14 %, with GPU use ≥ 50 % in 9-11 % of samples.
- Peak memory: median 17-20 GB per job, at most 35 GB (L40S pilot2).

## Packing tests

### 10-31 (n=12, same keys as fast2; noisy)

| layout | keys | succ / GPU-h | fast2 (sampler + 2 collectors, 9 CPUs) |
|---|---|---|---|
| sampler + 1 collector (`fast_k1`) | nextto_02, under_small_01 | 7.8, 10.6 | 9.3, 12.7 |
| sampler GPU + collector GPU over NFS (`fast_split`) | ontop_small_01, touching_01 | 2.5, 2.0 | 12.2, 6.0 |
| sampler + 2 collectors, 24 CPUs (`fast_c24`) | nextto_02, ontop_small_01, under_small_01 | 8.0, 10.1, 13.3 | 9.3, 12.2, 12.7 |

- sampler + 2 collectors on one GPU is the best of the tested layouts.
- The split layout halves the rate: the collector GPU is mostly idle.
- 24 CPUs did not beat 9. A sampler alone runs an attempt in 62 s against 108-137 s next to two collectors (ontop, E2), so
  contention is real; but more CPUs did not remove it, so it is not only a CPU shortage. The 11-02 tests below found what it is.

### 11-01 / 11-02 fleet: more processes per GPU (n=24 per sampler, E16)

Successes per GPU-hour, unpinned (one job or pair per cell; a pair is two sampler + collector jobs on one GPU, `job_pair.sh`):

| key | TITAN k2 | TITAN k3 | TITAN pair1 | L40S k2 | L40S k3 | L40S pair1 | L40S pair2 |
|---|---|---|---|---|---|---|---|
| ontop_small_01 (sampler-bound) | 12.2 | 12.8 | 14.9 | 7.4 | 11.4 | 25.9 | 19.2 |
| disembedding_03 (collector-bound) | 12.8 | 9.4 | 4.5 | 14.7 | 13.0 | 46.5 | 32.2 |
| mental_rotation-mug (fast sampler) | 28.6 | 24.8 | 39.1 | 50.6 | 55.8 | 66.3 | 82.1 |

- k = collectors per sampler; pair1 / pair2 = two samplers with 1 / 2 collectors each.
- ontop's yield is decided by the slice of train indices each job draws (5-25 % of attempts written), so its column-to-column
  differences are not the layout's.
- Two disembedding samplers ran out of CUDA memory on the 24 GB TITAN RTX (cuRobo); L40S pair2 peaked at 43 of 48 GB.
- A third unpinned collector made disembedding slower, not faster (TITAN 12.8 -> 9.4; its collectors stepped at 724 ms).

**The contention.** One disembedding instance run by two collectors in one job steps at 334-350 ms each against 95 alone, and they
burn ~4 cores each. 24 CPUs, 2 or 4 PhysX threads, and separate HOME, caches, job ids or template files leave it at 271-367 ms.
Disjoint CPU sets (`taskset`) bring each back to 63-87 ms at 1.1-1.5 cores each, and record the same demo. Collectors of the two
members of an unpinned L40S pair did not slow each other (55 ms); why is open.

**COLLECTOR_PIN** (`job_fast.sh`: each collector on 3 CPUs of its own, the sampler on the rest; job-level A/B, n=24):

| arm | succ / GPU-h | collector ms/step | sampler s/attempt | demo wall s |
|---|---|---|---|---|
| disembedding TITAN k2 pinned (2 jobs) | 32.0 / 29.2 | 89 | 35-39 | 114 |
| disembedding TITAN k3 pinned (16 CPUs) | 32.6 | 92 | 32 | 127 |
| disembedding TITAN k2 unpinned | 12.9 (fleet 12.8) | 340-374 | 54-77 | 423-431 |
| disembedding L40S k2 / k3 pinned | 52.7 / 64.0 | 46 / 47 | 17 | 61 / 66 |
| disembedding L40S k2 / k3 unpinned | 14.7 / 13.0 | 307 / 669 | 45 / 74 | 370 / 763 |
| ontop TITAN k2 pinned / unpinned | 4.0 / 12.2 (slice) | 49 / 113 | 24 / 27-49 | 120 / 203 |
| mental_rotation-mug TITAN k2 pinned / unpinned | 31.1 / 28.6 | 74 / 77 | 42 / 42-43 | 16 / 20 |

- Pinned collectors step at their alone speed, and three pinned collectors step as fast as two, so k=3 now pays for collector-bound
  families (L40S 64.0 against 52.7).
- The sampler gains where its collectors are busy (disembedding 2x); mental rotation's collectors are idle and nothing changes. Its
  31.1 against 28.6 is the prebuild fix (the fleet's TITAN mug jobs prebuilt from full-scene copies and rebuilt their world).
- The pinned ontop job drew a slice where 275 of 281 rejections were `target_placement` (8 written of 289 attempts); its collectors
  stepped 2.3x faster.
- **A pinned pair on one L40S** (two disembedding jobs with 2 collectors each, each member pinned inside half of the cores) made
  **104.6** successes per GPU-hour, against 32.2 for the same pair unpinned and 64.0 for one pinned k=3 job. Its four collectors
  stepped at 52 ms, its samplers took 19-20 s per attempt, and peak memory was 43 of 48 GB.

## Collector state dump and syncs: why no narrower set was built

- The 10-31 question was whether a narrower per-step dump / sync set ("could change" objects only) would pay. The timers answer it:
  - In cropped scenes the dump is 1-3 % of collector time (2 ms per step with RECORD_CHANGED_ONLY on, refine tags).
  - In full scenes it is 14 % (camera relocalization), but those collectors are busy 10 % of the time.
- The cost is the per-object Python pass over the registry, not the data:
  - RECORD_CHANGED_ONLY cuts the dumped floats 9x (1,330 → 150 per step on full-scene nextto_02) but the dump time only from ~26 to
    ~22 ms.
  - So a narrower set would need a cached object list, not a filter, and it would change the step's simulator reads, which E13 showed
    can alter full-scene physics.
- World syncs: 7 % of collector time (24 % in alignment, whose collectors are 69 % busy and drain 124 s after the sampler).
- With the sampler bounding the job, neither change would move successes per GPU-hour measurably. Only large IORS and disembedding
  are collector-bound. Their env step (62-83 % of collector time) and QA (19 % on large IORS) were inflated by the two unpinned
  collectors slowing each other; for disembedding, pinning removed most of it (Packing). Large IORS was not re-measured pinned.

## Follow-ups: GPU PhysX and a vectorized collector (E17)

- **F1, GPU PhysX for one scene** (`ISPATIALGYM_FAST_COLLECTOR_GPU_DYNAMICS=1`; one instance alone per TITAN RTX, two runs per arm):

  | instance | env ms/step CPU | GPU | episode s CPU | GPU |
  |---|---|---|---|---|
  | ontop_small_01 | 52.6 / 52.4 | 95.7 / 87.1 | 145 / 144 | 192 / 182 |
  | mental_rotation-mug | 35.5 / 42.8 | 82.2 / 82.3 | 18.6 / 18.7 | 26.0 / 25.6 |
  | disembedding_03 | 119.3 / 92.0 | 148.0 / 111.6 | 177 / 139 | 203 / 155 |

  - All 12 runs succeed. GPU dynamics is another physics realisation (ontop 896 -> 926 steps, disembedding 929 -> 960; mug
    unchanged), and it takes 1.1-1.2 GB more GPU memory.
  - Kept off: one scene is too small for the GPU solver to pay.
- **F2, a vectorized oracle collector** (N scenes per process on GPU physics): not built.
  - It depends on GPU PhysX, which is slower here, and on the vec_next world build (being merged by another session).
  - The multienv report (10-03) measured 15.5 env-steps/s for several environments in one process against 16.4 for separate
    processes.
  - The collectors are idle most of the time in the sampler-bound families, and "more scenes per GPU" is reached by more pinned
    collector processes, which run at the alone speed.
  - Worth revisiting only if pinned collectors become the bottleneck of the collector-bound families.

## Replay stage (separate budget)

- Full house scene, one process per TITAN RTX: 3.65-3.71 env-steps/s on the original recordings and 6.96 with RECORD_CHANGED_ONLY.
  Cropped-path recordings: 7.18 (E13).
- Packing on one TITAN RTX (full house, original recordings): 2 processes 6.6 total, 3 processes ~8.0 total (21.7 GB, GPU 100 %).
- 54k demos x 654 frames (isg-1020's mean) at 3 processes per TITAN: ~1,000-1,500 TITAN GPU-hours including per-unit loads. Cropped
  recordings should be faster, but they were not measured with packing. Vast measured 171 process-s per episode for isg-1020.

## Cluster constraints found

- The svl QOS caps one user at 48 GPUs / 56 jobs; viscam at 30 GPUs (L40S 48 GB, A5000/A6000, H200, rtxpro6000, one TITAN RTX node).
  At most 78 GPUs at once for this account.
- A 72-CPU / 8-GPU TITAN node fills at 3 jobs of 24 CPUs; the jobs use a median of 8 cores, so 12 CPUs / 32 GB per job.
  - One of 18 such jobs ran out of memory at 32 GB.
- py-spy segfaults on the Isaac Sim process; svl11 kills heavy background processes (run analyses as CPU sbatch jobs).

## Recommendation

1. **Pin, then pack.** COLLECTOR_PIN is on by default (it needs (K + 1) x 3 CPUs).
   - On L40S, run two jobs per GPU (`job_pair.sh`, 24 CPUs / 96 GB). A pinned disembedding pair made 104.6 per GPU-hour against 52.7
     for one pinned job, and the unpinned mental-rotation pair 82.1 against 50.6. Sampler-bound families were not run as pinned pairs.
   - As single jobs, collector-bound families take k=3 on 16 CPUs (TITAN 32.6, L40S 64.0); the others keep k=2.
   - On TITAN keep one job per GPU: two disembedding samplers ran out of the 24 GB, and pinned pairs were not tried there.
2. **Run long jobs.** 50+ instances per (alias, scene, room), so the 3-4 min world load and the per-template caches amortise; steady
   rates are ~1.2x the whole-job rates at n=24.
3. **Cut `target_placement` rejections.** They are 69-93 % of the rejections (52-83 % of all attempts) and the bulk of sampler time; a
   cheaper early check there is the largest remaining sampler lever.
4. **Prefer L40S where available.** It gives 2.2x TITAN's production-mix rate in these runs (to be confirmed on shared keys).
