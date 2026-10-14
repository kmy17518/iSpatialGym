# iSpatialGym LLM harness

Frontier language models driving R1Pro on the iSpatialGym eval instances, with no learned policy in the action path.
The planner is RoboProbe's L3 Inspect EEF harness ([RoboProbe](https://github.com/RoboProbe/RoboProbe),
`policy/RoboDojo_Agent_L3_Inspect_EEF`) ported to R1Pro: the same turn loop, prompt structure, tool semantics, motion
pipeline, budgets, failure accounting and probes, plus the base and torso controls a mobile robot needs. A CaP-X-style
**code agent** (the model writes Python against the same robot API) is meant to plug into the same core later.

## Disclosure (RoboProbe's checklist)

- **Level:** L3 LLM-as-Policy.
- **Models:** any `provider/model` of `llm/` (OpenAI Responses, Anthropic Messages, OpenAI-compatible chat); GPT-6 Astra
  is the default, reasoning effort `medium` (RoboProbe's published setting). Endpoint and key are runtime configuration.
- **Prompt:** system message in `planners/l3_tools/prompts.py` with the R1Pro embodiment notes of
  `planners/l3_tools/docs.py`; the Goal turn carries the task statement (`core/instructions.py`), the goal image and the
  subtask's `TASK RECIPE` (`planners/l3_tools/recipes/`).
- **Tools:** `move_eef` (named absolute grasp-point targets of both arms, gripper included), `move_base`, `move_torso`,
  `give_up` (reason + hindsight); probe-only `done` behind `allow_done`.
- **Motion stack:** grasp-point targets -> eef poses -> cuRobo `plan_pose` with RoboDojo's settings (32 IK seeds, 4
  trajectory seeds, 5 attempts, no graph planner) against self-collision and the planner world -> path resampled so no
  joint exceeds its per-step budget (the URDF velocity limit x 0.25, the oracle's arm planning scale) -> native joint
  playback; jaw after arrival; base and torso through the oracle's robot-level controllers.
- **Planner world** (`core/world.py`, recorded in `meta.json` as `harness.planner_world`): RoboDojo's one slab under the
  surface the robot is mounted on (5 cm thick, its top 2.5 cm under the mounting point). RoboDojo's arms are bolted onto
  the table top; R1Pro stands on the floor, so its slab lies under the floor, built from robot geometry alone. Nothing of
  the scene is in it (RoboDojo leaves out the objects on its table; the house's furniture is left out here too), and it
  is lowered if a robot collision sphere would touch it at the start.
- **Memory:** camera images of the newest 2 observation turns (`image_horizon`), the goal image always kept; the full text
  history and the model's encrypted reasoning are replayed.
- **Budgets:** 170 LLM calls (every response counted) and 18,000 control steps (10 min at 30 Hz), the latter reported to
  the model every turn as env steps remaining.

## The turn loop (as RoboProbe)

1. System message, then the **Goal turn**: `Goal: <instruction>`, `TASK RECIPE:` and the pinned goal image.
2. Every **observation** is its own user turn: `Instruction: ...`, the state named exactly as the tools take it
   (`left_x=...`, `base_x=...`, `torso_translate=...`, arm joints as read-only context), `Env steps remaining before the
   episode ends: N`, the previous motion's `Arrival check ...`, and the labelled camera images.
3. The model is called until it returns one valid motion call. Its tool result is written at acceptance, before the
   motion plays: `Accepted: playing N waypoints (X s). ... The next observation reports how close the grasp point landed.`
4. A refused motion gets a text-only result and the model answers again on the same observation (no images re-sent).
   An unreachable pose is free; malformed calls, out-of-bound targets and motions over the 10 s playout cap count as
   repairs, and 3 on one observation end the episode as a capability failure. So do a refusal and several tool calls that
   do not fold into one (parallel `move_eef` calls naming disjoint dimensions are merged into one bimanual call).
   No tool choice is forced.

## Episode ends and scoring

- The environment ends a successful episode: the adapter stops at the task's `official_success`, then run_isg's epilogue
  decides `success` as for every policy. The model is not told that success ends the episode (neither is RoboProbe's).
- The step budget is the environment's own limit (the epilogue's verdict stands).
- A planner stop (`give_up`, `done`, call budget, capability failure) is scored as a failure whatever the scene looks like
  (`meta.json` keeps the epilogue's own verdict as `harness.epilogue_success`).
- A provider failure (12 retries on RoboProbe's jittered 1-30 s ladder, a 90 s hard deadline per request) or a harness bug
  raises `InfrastructureFailure`: run_isg records `status: error`, `summarize` does not score the episode, and
  `run --rerun-unscored` resubmits it.

## Model-facing conventions

- **Frame:** the robot base frame (x forward, y left, z up, origin on the floor at the centre of the base); it moves with
  the base. RoboDojo's arms are fixed, so RoboProbe uses the world frame.
- **Position:** the grasp point, the centre of the finger pads (`core/pose.py`), as RoboProbe's grasp point.
- **Orientation:** three angles measured from a reference grasp, RoboProbe's scheme (extrinsic pitch about the lateral
  axis, then roll about the forward axis, then yaw about the vertical). RoboProbe's reference is a straight-down grasp, its
  rig's modal orientation; R1Pro's is a level grasp pointing straight ahead with the jaws closing along base y, where its
  grippers spend most of their time (both arms start there). The triple is singular only with the jaws closing vertically.
- **Bounds:** loose per-arm boxes that contain every grasp point of the eval demos with 0.25 m to spare; the planner, not
  the bounds, decides reachability.

## Differences from RoboProbe Inspect EEF

R1Pro-specific, kept on purpose: `move_base` and `move_torso`; base-frame coordinates and base odometry in the state; the
goal image in the Goal turn (iSpatialGym tasks are defined by it); the recipes describe iSpatialGym subtasks and their
success checks; the orientation reference and the bounds above; the planner's slab is under the floor R1Pro stands on
rather than under a table top, so the work surfaces (house furniture) are not in it, while RoboDojo's table is both its
mounting and its work surface; both arms named in one call are planned as one unit (R1Pro's arms share a torso;
RoboDojo plans each ARX arm on its own); the embodiment notes give no work-surface height (RoboProbe's give its fixed
table's; here it would be scene geometry); the jaw phase uses the oracle's assisted-grasp closing routine; base and
torso acceptances report `about N waypoints` (closed-loop controllers); arrival checks also cover base and torso moves.

Not ported (off in RoboProbe's published run): in-context demonstrations, prior-attempt replay, the delayed first call of
`imitate_sorting_sequence`, and API-key rotation on rate limits.

## Layers

```
run_isg --mode policy                      environment side: instance load, executor, official success, epilogue,
   |  policy(obs, info) -> 23-D action     video / meta.json, Kit app data - unchanged; the harness is the policy
   v
policy.HarnessPolicy                       run_isg adapter: bind() / __call__() / finalize(); accept / reject protocol,
   |                                       planner stops scored as failures, infrastructure failures unscored
   v
planners/l3_tools                          turn loop (agent.py), prompts, embodiment notes, recipes, tool surface
   v
core/                                      shared by every planner
   pose.py         grasp point, reference-based angles, named dimensions and bounds (simulator-free)
   embodiment.py   R1Pro: robot state in the base frame, camera frames, mounting (robot-only reads)
   planning.py     cuRobo arm planner as RoboDojo's plan_path (left_arm / right_arm / both_arms)
   world.py        the planner world: RoboDojo's slab under the mounting surface, here the floor (geometry simulator-free)
   motion.py       requests -> plans -> joint chunks; jaw, base and torso via the oracle's robot-level controllers
   approver.py     target bounds (repairable rejections) and per-action checks
   session.py      streams motions one control step at a time; RobotAPI = blocking calls for code agents
   observation.py  the observation turn: state block, steps remaining, arrival check, labelled images
   instructions.py task statement per family; goal.py: the goal image; perturb.py: RoboProbe's probes
   trace.py        transcript.jsonl + frames/ + summary.json per episode, source revision
llm/                                       provider clients over plain HTTP with RoboProbe's retry policy, scripted client
run.py / summarize.py                      episode preparation + SLURM submission, aggregation (unscored episodes apart)
```

## Probes (`--perturb`)

`flip_vision_ud`, `flip_vision_lr`, `mask_cameras` (black image, label `MASKED`), `negate_xyz` (mirrored position frame:
reported and commanded positions negated together, base odometry and `move_base` included), and pose jitter:
`jitter_dx/dy/dz` (fixed offset), `jitter_sphere_r` or `jitter_sphere_mean_r` (random direction, fixed or exponential
length), `jitter_sphere_per_move`. The arrival check measures against the un-jittered request, so a bias shows.
`--set disable_give_up=true` and `--set allow_done=true` are the other RoboProbe probe switches.

## Running

```bash
# one episode per eval instance (the 84 counted eval instances), GPT-6 Astra, submitted with SLURM
python -m ispatialgym.harness.run --instances eval85-counted --out /path/to/harness_runs/astra_v1 \
    --model openai/gpt-6-astra --submit
python -m ispatialgym.harness.summarize /path/to/harness_runs/astra_v1
python -m ispatialgym.harness.run --instances eval85-counted --out /path/to/harness_runs/astra_v1 --rerun-unscored
```

API keys come from the environment of the submitting shell (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or the variable
named by `--model-option api_key_env=...`); nothing writes them to disk. Images go to the model at the cameras' own
resolution (1080 x 1080 in the eval profile) as JPEG quality 95, as RoboProbe sends its cameras; `--set image_size=512`
shrinks them. The job script (`scripts/harness/job_episode.sh`) follows the cluster's Kit app-data rule and links the
datasets into a per-job directory, so the checkout itself never holds dataset symlinks.
