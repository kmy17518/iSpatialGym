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
