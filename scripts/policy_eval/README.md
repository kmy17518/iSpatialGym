# Policy eval of the isg-train224 checkpoints on the v2 eval instances (2026-11-16)

Evaluates checkpoints of the isg-train224 runs (pi05 / ACT / DP, `tmp/TRAIN_RUN.md` section 8) on
`kmy17518/isg-standard-eval-224-instances-v2` (1000 instances, 100 classes) with `python -m ispatialgym.run_isg --mode policy`,
settled evaluation (`--settle-eval on --settle-steps 45`), 224 px cameras, the step budget alone ending an episode
(`--wallclock 1000000`). Branch `campaign-20261115-eval`; record dir `EV=/vision/u/minyeong/projects/isg_home/campaign_20261110/eval_policy_r1`.

## Input consistency (eval time vs the batched replay that rendered the training data and the goal images)

Fixed in this branch (each was wrong before):

| what | before | now |
|---|---|---|
| TRO field `goal_observation_collection` (written by export_eval.py) | every v2 instance failed to load (TROValidationError) | accepted, informational |
| template | `template_files` loaded `<subtask>_0_0_template.json` for every instance (876 distinct templates: most instances in another instance's world) | the index record's own template + meta |
| BDDL problem | the release tree's numbering: 600 problem files missing, 300 another variant (problem ids of the v2 export are each template's own) | per-instance overlay of the exact collection BDDL (sha256 = TRO `bddl_sha256`), `ISPATIALGYM_EVAL_BDDL_OVERLAY`, built by `build_bddl_overlay.py`; `check_dataset.py` verifies all 1000 |
| world | template meta `scene_crop` made the evaluator load the collection's cropped world (absent from the dataset) | eval dataset templates load the full template (= the replay's `--scene full`) |
| lighting | 74 promoted train demos' TROs carry a collection lighting draw, re-applied at eval; the replay (goal image, training frames) ignored it | `ISPATIALGYM_EVAL_IGNORE_COLLECTION_LIGHTING=1`: stock lights, as rendered |
| proprioception | the eval robot emitted OmniGibson's default 48-D proprio; the training data / servers read BEHAVIOR's 61-D layout | policy mode carries the 61-D `PROPRIOCEPTION_INDICES` keys (observation only) |
| text-only prompt | openpi's server prompts with one fixed task | per-request `__task_name__` -> the checkpoint's recorded training prompt (`patches/openpi_task_prompts.py`, serving copy only) |

Measured with GT actions through the policy path (`gt_policy.py`, obs dumps) against the eval LeRobot episode of the same instance
(`compare_obs.py`): the 23-D training state is identical (max |diff| 0) at steps 0-30, all 61 proprio values within 1e-3; images
match up to the training data's HEVC compression (head 25-29 dB, wrists 29-37 dB PSNR; the remaining differences are compression and
ray-traced mirror reflections), the lit instance included. Goal PNG vs the episode's decoded last frame (what goal runs trained on,
`GOAL_SOURCE=episode_last`): 33.7-37.2 dB (compression only).

Per run (the client sends the same observation to every server; the server does the training-time preprocessing):

| run | server | client | goal | prompt |
|---|---|---|---|---|
| pi05 goal (8a, 8c) | openpi `serve_b1k.py --action-horizon 16` (regime image: prompt = goal scaffold) | chunk of 16 executed open loop | the instance's goal PNG | scaffold only |
| pi05 shuffle-goal (8e) | as goal | as goal | own goal PNG, or `run_isg --cross-goal` (the dataset's cross-episode goal image) | scaffold only |
| pi05 text (8f) | openpi, regime language | chunk 16, `send_task_name` | none sent | exact training prompt of the instance's class |
| ACT (8d) | `serve_b1k.py --temporal-agg` | one action per step | goal PNG | none |
| DP (8b) | `serve_b1k.py` (8-step plan, own 2-frame history) | one action per step | goal PNG | none |

Task ids: the instance's `source.class_id` = the training set's `task_name` -> `task_index` of `meta/tasks.jsonl`.

## Tools

- `job_env.sh`, `job_list.sh`: cluster jobs (`LIST_MODE=1`: one process, one world).
- `gt_policy.py` (`ispatialgym/eval/`): demo actions through the policy path; `ws_run_policy.py`: the websocket client of run_isg.
- `compare_obs.py`, `compare_frames_png.py`, `compare_runs.py`: input audit; world-reuse equivalence.
- `bench_server.py`, `ws_overhead.py`: server throughput; websocket overhead.
- `world_groups.py`: instances whose templates are compatible (run_isg's reuse rule) -> 100 worlds of 10 (one per class).
- `make_eval_run.py`, `vast/{build_bundle.sh,boot.sh,eval_worker.py,vastctl_eval.py,watch_run.sh}`, `summarize.py`: the Vast fleet.

## Speed (measured 2026-11-16)

Per instance: Kit start ~45 s + world build (scene import 100-430 s) + episode. One world per instance would spend more time
loading than evaluating; the 1000 instances need only **100 worlds** (each class's 10 instances: templates equal outside their task
objects within 0.75 mm, same simulator records, same BDDL problem; `world_groups.py`), so units are chunks of a world group run with
`run_isg --instance-list` (first instance builds the world, the others attach: 5-14 s instead of 100-430 s). Reused vs fresh worlds
with GT actions (6 instances, 2 classes): same success, steps and evaluated step, final object poses within 10 um, errors within 2e-4;
a reused start differs only in velocities (base <= 4e-3 m/s, joints <= 0.06 rad/s), positions within 1e-3.

| setup (RTX 5090 unless noted) | steps/s per process | steps/s per GPU |
|---|---|---|
| pi goal, 1 process (28 CPUs) | 16.4 | 16.4 |
| pi goal, 2 processes (9 CPUs each) | 14.8 | 29.7 |
| ACT --temporal-agg, 3 processes, 1 server | 5.9 | 17.8 |
| ACT --temporal-agg, 2 processes (RTX 4090) | 6.5 | 13.0 |
| DP, 2 processes, 4 servers (RTX 3090) | ~5 | ~10 |

- Websocket vs local policy: the websocket costs 1.45 ms per request (1.6 % of a pi request, ~0.1 ms per control step); a local policy
  would load the model into every simulator process (pi: ~9 GB VRAM each; JAX 0.11 needs Python >= 3.12, the simulator env is 3.10).
  One server per GPU shared by the GPU's processes.
- Processes per GPU: VRAM-bound (a 224 px eval process ~8.5 GB, a pi server ~10 GB): 2 per 32 GB GPU with pi, 3 with ACT / DP; more
  processes per GPU scale nearly linearly while CPUs last (~4+ per process).
- ACT --temporal-agg queries every step (~10 ms solo; 43 ms with 3 clients on one server): give ACT one server per process.
- DP: 100 DDPM steps per 8-step plan (~350 ms on a 4090), server-bound; several servers per GPU (4: 2.2x of one). Its sampling draws
  fresh noise every denoising step, so a faster sampler would not be output-identical; not changed.

## Running an eval (Vast fleet)

```bash
# on svl11 as minyeong; EV=/vision/u/minyeong/projects/isg_home/campaign_20261110/eval_policy_r1; code snapshot $EV/code/eval_<sha>
cd $EV
# 1. checkpoints: {tag: {kind, repo, step, task_alias (pi), server_args, policy_kwargs, run_isg_args, families, servers_per_gpu, procs_per_gpu}}
#    e.g. inputs/ckpts_full1.json (pi goal / shuffle --cross-goal / iors / text-iors send_task_name, ACT --temporal-agg, DP 4 servers)
python3 code/eval_<sha>/scripts/policy_eval/make_eval_run.py --run <run> --ckpts inputs/<ckpts>.json \
    --worlds inputs/world_groups_v2.json --chunk-steps 13000          # one unit = one class's world (<= 13000 budget steps)
env -i HOME=$HOME PATH=/usr/bin:/bin bash -c 'source tool/creds.sh; $S5 cp runs/<run>/spec.json $JOB_S3/runs/<run>/spec.json'
# 2. fleet: keeps N GPUs working while units are open (og-isgeval- labels, own ledger; driver >= 580 for pi; >= 8 CPUs / GPU)
bash code/eval_<sha>/scripts/policy_eval/vast/start_loop.sh <run> 150 1.0     # stop: kill $(cat logs/loop_<run>.pid)
# 3. results (any time): per instance / class / task / family / total + QA violations
env -i HOME=$HOME PATH=/usr/bin:/bin bash -c 'source tool/creds.sh; bash code/eval_<sha>/scripts/policy_eval/vast/collect.sh <run>'
#    -> runs/<run>/summary.md, summary_<ckpt>.json, instances.csv
```

Bundle changes (code / worker / dataset / policy envs): `vast/build_bundle.sh <code dir> [boot code data worker policy]`.
Cluster (one GPU, one world): `job_policy.sh` (server + `run_isg --instance-list`), `job_list.sh` (GT / any policy, `LIST_MODE=1`).

## Full run `full1` (2026-11-16, Vast, $EV/runs/full1/summary.md)

Six checkpoints at once: pi goal 15k, pi shuffle 10k (`--cross-goal`), ACT 130k, DP 80k (1000 instances each), pi goal-iors 15k and
pi text-iors 15k (the 350 iors instances): 4700 episodes, 742 world-chunk units (<= 13000 budget steps), fleet loop at up to 250 GPUs
(~90 hosts, mostly RTX 3090 / 4090, >= 8 CPUs per GPU). Boot (bundle + policy env + checkpoint) took ~25 min; 739/742 units done at
2 h 45 min; spend ~$200. Cost per 1000 instances on that fleet: 35-40 GPU-h (pi goal 34.8, ACT 35.9, DP 40.2): one checkpoint in 1.5 h
needs ~35 such GPUs (or ~20 RTX 5090s at 2 processes each); world loading is 11-17 % of process time.
Lessons: give ACT / DP smaller chunks (~6000 steps; they run 4-5 steps/s, a 13000-step chunk is ~1 h) and pi larger ones; a few slow
hosts (RTX 6000 Ada with DP: 2 steps/s) and retried units set the tail; keep the fleet warm across checkpoints (boot is per host).
