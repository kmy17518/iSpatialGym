# iSpatialGym: evaluating an LLM harness on 100 eval instances

This branch (`llm-harness-100`) packages the iSpatialGym policy evaluation so an external policy (an LLM harness) can be
evaluated exactly the way the trained policies (pi0.5, ACT, diffusion policy) were: same simulator code, same eval
instances, same evaluation settings, same metrics. You write a small Python adapter class around your harness; the
scripts here run it on the instances and summarize the results.

Contents:

1. [What you will do](#1-what-you-will-do)
2. [Install](#2-install)
3. [Download the eval instances](#3-download-the-eval-instances)
4. [Check the setup](#4-check-the-setup)
5. [The 100 instances](#5-the-100-instances)
6. [Write the policy adapter](#6-write-the-policy-adapter)
7. [What the system prompt must say](#7-what-the-system-prompt-must-say)
8. [Run the evals (in this order)](#8-run-the-evals-in-this-order)
9. [Summarize and send the results](#9-summarize-and-send-the-results)
10. [Reference: settings, metrics, files](#10-reference-settings-metrics-files)

---

## 1. What you will do

| # | eval | instances | conditioning | results folder |
|---|---|---|---|---|
| 1 | IORS with goal image | 35 (`--subset iors`) | the instance's goal image | `results/1_iors_goal` |
| 2 | full set with goal image | 100 (`--subset full`) | the instance's goal image | `results/2_full_goal` |
| 3 | IORS with text | 35 (`--subset iors`) | a text instruction, no image | `results/3_iors_text` |
| 4 | full set (or IORS only, if short on time) with cross-episode goal image | 100 or 35 | another episode's goal image | `results/4_full_cross_goal` |

Do them in this order of priority. After each one, zip its summary (`bash scripts/llm_harness/summarize.sh <results folder>`)
and send the zip.

IORS = inter-object relation satisfaction: put / push one object into a relation with another (on top of, under, inside, next to,
touching), small-scale (tabletop) or large-scale (furniture, floor). The full set covers all 7 task families (section 5).

## 2. Install

Needs: Linux x86_64, an NVIDIA RTX-class GPU with >= 16 GB memory (one simulator process uses ~9 GB) and a driver that
supports Isaac Sim 5.1 (>= 535; we ran 570-580), ~80 GB of disk, `conda` (Miniconda / Miniforge) and `git`.

```bash
git clone --branch llm-harness-100 https://github.com/kmy17518/iSpatialGym.git ispatialgym
cd ispatialgym
# Creates the conda env `ispatialgym` (Python 3.11, torch 2.7.0 cu12x, Isaac Sim 5.1.0, OmniGibson, BDDL, cuRobo 0.8.0),
# clones Behavior-1K (our fork, branch isg-llm-harness-100: the exact simulator version the evals ran with) into ./Behavior-1K,
# downloads the BEHAVIOR-1K assets (~35 GB; you accept their license with the flag below) and installs this repo (editable).
./setup.sh --new-env ispatialgym --omnigibson --bddl --dataset --primitives \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ispatialgym
```

Check:

```bash
git -C Behavior-1K log -1 --format='%h %s'      # 5f2a620b iSpatialGym policy-eval snapshot: 6a6194600 + ...
python -c "import ispatialgym, omnigibson, bddl; print(omnigibson.__file__)"   # .../ispatialgym/Behavior-1K/OmniGibson/...
ls Behavior-1K/datasets          # behavior-1k-assets, omnigibson-robot-assets, ispatialgym-assets, omnigibson.key, ...
```

`./setup.sh --help` lists the options; `ISPATIALGYM_DRY_RUN=1 ./setup.sh ...` prints the plan without installing. If a
`Behavior-1K/` folder already exists, `setup.sh` uses it as is: make sure it is on branch `isg-llm-harness-100` of
`https://github.com/kmy17518/BEHAVIOR-1K`.

## 3. Download the eval instances

The instances are the Hugging Face dataset `kmy17518/isg-standard-eval-224-instances-v2` (private: ask for access / a read
token). Download it at revision `cf94275c` (what the evals ran on; ~1.2 GB):

```bash
huggingface-cli login            # a token that can read the dataset
huggingface-cli download kmy17518/isg-standard-eval-224-instances-v2 --repo-type dataset \
    --revision cf94275ca188039bd64f877b8212d503b33d5a08 --local-dir ~/isg-eval-data
export ISPATIALGYM_INSTANCES_PATH=~/isg-eval-data/ispatialgym-eval-instances   # put this in your shell profile
```

It holds `ispatialgym-eval-instances/` (1000 instances: scene templates, instance files, goal images, BDDL problems, index
`metadata/instances.jsonl`) and `ispatialgym-eval-demos/` (the demonstration of each instance; only the setup check uses them).

## 4. Check the setup

Replay the recorded demonstrations of two instances through the policy path (`GTPolicy` feeds the demo's actions as if a
policy produced them). Both should end with `success=True`:

```bash
python scripts/llm_harness/run_eval.py --policy ispatialgym.eval.gt_policy:GTPolicy \
    --instances 481 7 --results results/0_setup_check --name gt
```

Each instance starts its own simulator: expect ~1 min of Isaac Sim start-up (much longer the very first time: shader cache),
1-7 min to build the scene, then the episode. Logs: `results/0_setup_check/gt/_logs/<instance id>.log`.
`reference_gt_100.csv` lists what the demo replay scored on all 100 instances on our machines (a demo replay does not succeed
everywhere: physics differs slightly from the recording), so you can compare instance by instance.

Then try the example adapter (it holds still, so it fails every task, but it exercises the whole harness path, including
the text and cross-episode conditions):

```bash
python scripts/llm_harness/run_eval.py --policy ispatialgym.eval.llm_policy:HoldStillPolicy \
    --instances 481 --condition text --results results/0_setup_check --name hold_text
```

## 5. The 100 instances

`instances_100.csv` / `instances_100.json`: one instance per task (the dataset's 100 tasks = `task` column, e.g.
`alignment-axial_01`, 10 instances each), drawn reproducibly by `select_instances.py` (it prefers instances that have a
cross-episode goal image). Columns: dataset `index`, `instance_id`, `task`, `task_class` (the 23 task classes), `task_family`,
`scene`, `step_budget` (the episode length, see below), `iors`, `cross_episode_goal` (hard / easy / none), `text_prompt`
(IORS only), and the files. `subsets/full_100.txt` and `subsets/iors_35.txt` are the instance-id lists of the two subsets.

| task family | tasks | what the robot must do |
|---|---|---|
| camera_relocalization | 10 | drive (base + torso) until the head camera sees the goal image's view; *obstructed*: an obstacle stands in the way and must stay in place (it may not be pushed more than 0.15 m / 0.45 rad) |
| object_scaling | 10 | the object starts in the right hand: move it nearer / farther along the head camera's viewing direction until its apparent size matches the goal; keep holding it |
| mental_rotation | 10 | the object starts in the right hand: rotate it to the goal orientation without moving it; keep holding it |
| disembedding | 10 | pick the target out from among similar distractors on one tray and put it on the other tray (no distractor may end there); release it |
| alignment | 10 | *axial*: push an object lying on a table (it pivots about its centre) until it is aligned as in the goal; *world vertical*: push a tilted object hanging on a wall upright |
| configuration_matching | 15 | *block design*: arrange four blocks in a board's pocket; *attachment*: attach a part (cap, cork, ...) to its partner; *articulation open / close*: open or close a lid / door / drawer, small (tabletop) or large (furniture) |
| inter_object_relation_satisfaction | 35 | put / push the target into a relation with the partner: ontop, under, inside, nextto, touching; small or large scale |

In every task except object scaling and mental rotation the robot must have let go of the objects at the end, and in every
task the robot and the objects must be at rest at the end (see section 7).

## 6. Write the policy adapter

Subclass `ispatialgym.eval.llm_policy.HarnessPolicy` (`ispatialgym/eval/llm_policy.py`) in your own module and implement
`act`. The eval calls it every control step (30 Hz) and executes the returned action:

```python
import numpy as np
from ispatialgym.eval.llm_policy import HarnessPolicy, ACTION_SLICES, GRIPPER_OPEN, GRIPPER_CLOSED

class MyHarnessPolicy(HarnessPolicy):
    def __init__(self, model="...", replan_every=15, **kwargs):     # --policy-kwargs '{"model": "...", "replan_every": 15}'
        super().__init__(model=model, replan_every=replan_every, **kwargs)
        self.model, self.replan_every = model, replan_every

    def reset(self):
        # once per episode, conditioning already set:
        #   self.condition      "goal" | "cross_goal" | "text"
        #   self.goal_image     (224, 224, 3) uint8 head-camera image of the goal (None for "text")
        #   self.prompt         the text instruction ("text" only, else None)
        #   self.task_family, self.task, self.step_budget (steps until the episode ends), self.out_dir (this instance's folder)
        self.plan = []

    def act(self, obs):
        # obs["head_rgb"], obs["left_wrist_rgb"], obs["right_wrist_rgb"]: (224, 224, 3) uint8
        # obs["proprio"]: (61,) float32;  obs["proprio_named"]: dict of named slices (below);  obs["step"]: 0-based step
        if not self.plan:
            self.plan = self.query_harness(obs)          # e.g. a list of 23-D actions for the next replan_every steps
        return self.plan.pop(0)                           # one 23-D action per call (np.ndarray or list)
```

Run it with `--policy my_module:MyHarnessPolicy` (the module must be importable: on `PYTHONPATH` or installed).

**Action** (23 values, R1Pro robot, BEHAVIOR's JoyLo layout, `ACTION_SLICES`):

| slice | part | meaning |
|---|---|---|
| `[0:3]` | base | velocity (vx forward, vy left, wz yaw rate) normalised to [-1, 1] = 0.75 m/s, 0.75 m/s, 1.0 rad/s |
| `[3:7]` | trunk | 4 absolute joint positions (rad) |
| `[7:14]` | left arm | 7 absolute joint positions (rad) |
| `[14]` | left gripper | -1 = closed, +1 = open (smooth in between) |
| `[15:22]` | right arm | 7 absolute joint positions (rad) |
| `[22]` | right gripper | -1 = closed, +1 = open |

**Proprioception** (61 values, `obs["proprio_named"]`): `base_qvel` (3), `arm_left_qpos` / `arm_right_qpos` (7 each),
`arm_*_qvel`, `eef_left_pos` / `eef_right_pos` (3, robot frame), `eef_*_quat` (4, xyzw), `gripper_*_qpos` (2 finger
joints, ~0 closed .. 0.05 open), `gripper_*_qvel`, `trunk_qpos` (4), `trunk_qvel`.

Rules of the adapter:

- Use only `obs`, the conditioning attributes and your own state. Do not read the simulator (`executor`, `loader`) or `info`:
  they contain the success signal.
- Never return `None`: the episode always runs for its full `step_budget`; keep returning actions (e.g. `self.hold_action()`:
  the last command with the base stopped) when you are done.
- There is no wall-clock limit: the simulator waits for `act`. Calling the model every step is allowed but slow; returning a
  chunk of actions per model call (as above) is the usual approach.
- `on_episode_end(result)` is called after the episode; files you write to `self.out_dir` (e.g. the model's transcript) stay
  with the instance's results.

## 7. What the system prompt must say

Put these into the harness's system prompt (in your words), together with the action / observation format above.

**1. Finish by holding still.** An episode always runs for its full step budget. When the task looks complete, keep commanding
the same final configuration (same arm / trunk joint targets, base velocity 0, same gripper commands) until the episode
ends; do not keep adjusting. The main score is the *settled* success: the goal must hold, with the robot and the objects at
rest, for 45 consecutive steps (1.5 s). Reaching the goal briefly (*unsettled* success) is reported too, but counts less.
Except in object scaling and mental rotation (where the object must stay in the hand), the goal requires the object to be
released (not grasped) before the hold.

**2. Release with a single +1.** To release a grasped object, set that gripper's command to exactly +1.0 in one step, with no
ramp (not 0.2, 0.4, ... 1.0). Any other release is a `bad_release` violation. Closing may be gradual (ramp toward -1 as
slowly as you like).

**3. Keep the motion clean (QA metrics).** Every step is also checked against quality metrics. At eval time they are all
*soft*: logged and reported, they never fail an episode. They are still part of the results, so avoid them:

| metric | what counts as a violation |
|---|---|
| `robot_touching_objects` | a robot link other than the arms / grippers (base, torso, head) touches an object (the floor excepted) |
| `robot_touching_itself` | two robot links touch each other (the two fingers of one gripper excepted) |
| `arm_hand_contacts` | an arm or gripper touches an object it is not manipulating (any non-floor object; task objects marked) |
| `pause` | more than 50 consecutive steps (1.7 s) with no joint moving faster than 0.001 (the final hold counts too, see below) |
| `jerky_motion` | arm motion too fast / jerky over the episode (mean joint speed > 1.2 rad/s, > 0.5 % of steps near a joint's speed limit, mean acceleration > 3, mean jerk > 150) |
| `arm_stutter` | an arm's commanded motion jolts back and forth (>= 3 velocity changes of >= 0.6 rad/s within 6 steps while moving) |
| `ghost_hand` | an arm is more than 0.1 rad away from its commanded joints when its gripper starts opening (opening while the arm is still being pushed / lagging) |
| `bad_release` | a release that does not come from a gripper command of exactly +1 |
| `held_object_drift` | the held object slips in the hand |
| `stalled` | every joint nearly still for too long (the final hold counts too, see below) |
| `head_camera_upright` | the head camera tilted more than 0.4 rad for more than 30 steps while driving |
| `robot_wobble` | a load the hands bear (pressing, lifting) bends the torso more than 1 degree off its commanded position |
| `object_tilt` | an articulated object being opened / closed (box, laptop) is shifted more than 10 mm or tipped more than 0.04 rad |
| `goal_occlusion` | at the end, a target / partner object is at least half hidden from the head camera |

Practical consequences: move the arms smoothly (small joint steps per control step, no back-and-forth), keep the head level
while driving, do not bump furniture with the base or torso, do not brush other objects with the hands, keep the objects
visible, and do not stand still for long stretches mid-task.

Holding vs. `pause` / `stalled`: the evaluator does not know when the policy considers itself done, so the final hold of rule 1
also registers as `pause` / `stalled` once it lasts more than 50 steps. That is expected and fine: rule 1 (hold) wins, because
the settled success depends on it. The trained policies were scored with the same metric definitions, so the numbers stay
comparable; when reading `pause` / `stalled`, look at the successful rollouts separately (the summary reports them).

Conditioning, to explain per eval:

- **goal image** (evals 1, 2): a 224 x 224 picture from the robot's head camera at the end of a successful demonstration of
  this very instance: it shows the goal state from where the robot stood. For camera relocalization the view itself is the
  goal (bring the head camera to that view); for the other families the goal is the object state shown (object positions /
  orientations / relations / joint states), not the exact robot pose.
- **text** (eval 3): an instruction such as "Put the brussels sprouts inside the bowl." (`text_prompts_iors.json`: the exact
  instructions the text-only trained policy was given). No goal image.
- **cross-episode goal image** (eval 4): the goal image of a *different* demonstration of the same goal (the same goal
  configuration of the same objects, but another episode: the robot started elsewhere, so the view differs; that episode may
  also have had other lighting / textures). It tests whether the harness understands the goal state rather than matching
  pixels. About half are *hard* (the episode whose view differs most), half *easy* (a random one); recorded per instance in
  the results (`goal_kind`, `cross_episode_goal`). Camera relocalization has none (its goal is the view itself); those 10
  instances fall back to their own goal image.

## 8. Run the evals (in this order)

```bash
P=my_module:MyHarnessPolicy        # your adapter
K='{"model": "..."}'               # its keyword arguments (JSON), optional

# 1. IORS, goal image
python scripts/llm_harness/run_eval.py --policy $P --policy-kwargs "$K" --subset iors --condition goal --results results/1_iors_goal --name mymodel
bash scripts/llm_harness/summarize.sh results/1_iors_goal                  # -> send results/1_iors_goal/summary_*.zip

# 2. all 100, goal image (reuse the 35 IORS episodes of eval 1: copy them in, the runner skips finished instances)
mkdir -p results/2_full_goal && cp -r results/1_iors_goal/mymodel results/2_full_goal/
python scripts/llm_harness/run_eval.py --policy $P --policy-kwargs "$K" --subset full --condition goal --results results/2_full_goal --name mymodel
bash scripts/llm_harness/summarize.sh results/2_full_goal

# 3. IORS, text instead of the goal image
python scripts/llm_harness/run_eval.py --policy $P --policy-kwargs "$K" --subset iors --condition text --results results/3_iors_text --name mymodel
bash scripts/llm_harness/summarize.sh results/3_iors_text

# 4. all 100 (or --subset iors if short on time), cross-episode goal image
python scripts/llm_harness/run_eval.py --policy $P --policy-kwargs "$K" --subset full --condition cross_goal --results results/4_full_cross_goal --name mymodel
bash scripts/llm_harness/summarize.sh results/4_full_cross_goal
```

Choosing instances: `--subset iors|full`, or `--instances <id or index> [<id or index> ...]` (one or many instance ids or
dataset indices from `instances_100.csv`), or `--instance-file <file>` (one per line). Every other setting is fixed by the
script (section 10).

- Resume: rerun the same command; instances that already have results are skipped (`--rerun` redoes them).
- Parallel: `--jobs N` runs N instances at once, `--gpus 0,1,...` spreads them over GPUs (each instance: ~9 GB GPU memory,
  ~10 GB RAM, ~4 CPU cores, plus whatever your harness needs).
- Video: `--extra-args "--record-view head"` also writes each episode's head-camera video (`head_rgb.mp4`). Off by default.
- Debug one instance: `--instances <id> --results results/debug --name try1`, then look at `results/debug/try1/<id>/`
  (`final_rgb.png` vs `goal_rgb.png`, `result.json`) and the log in `_logs/`.

## 9. Summarize and send the results

```bash
bash scripts/llm_harness/summarize.sh results/1_iors_goal
```

writes `results/1_iors_goal/summary/` (`summary.md`, `summary_<name>.json`, `instances.csv`, `groups.csv`, `qa.csv`) and
zips it as `results/1_iors_goal/summary_1_iors_goal_<date>.zip`: **send that zip** after each eval. (The full per-instance
folders are larger, ~1 MB per instance without video; keep them, they may be asked for.)

`summary.md` reports, in total, per task family, per task class (23) and per task (100):

- success rates (with counts) for the four verdicts: **settled geometric** (the main score), settled symbolic, unsettled
  geometric, unsettled symbolic;
- the position / orientation error at the evaluated step (all instances, and the successful ones);
- the QA metrics over all rollouts and over the successful rollouts of each verdict (mean violations / mean separate runs /
  share of rollouts where the metric fired), with the number of successful rollouts.

## 10. Reference: settings, metrics, files

**Eval settings** (the same as the trained policies' evals; `run_eval.py` sets them, do not change them):
`python -m ispatialgym.run_isg --mode policy --image-size 224 --settle-eval on --settle-steps 45 --wallclock 1000000 --record video`,
one process per instance, with the simulator switches of those runs (`EVAL_ENV` in `run_eval.py`). This means 224 x 224
cameras (the training resolution), settled evaluation with a 45-step settle window, the policy-mode tolerances, the step
budget `max(300, round(1.5 x the instance's demonstration length))` as the only end of an episode (no wall-clock limit), and the
QA metrics logged without failing episodes.

**Verdicts**: *symbolic* = the BDDL goal predicates hold (e.g. `ontop`, `inside`, `open`, `not grasped`, objects at rest);
*geometric* = the symbolic goal and the measured pose / layout errors within tolerance. *Settled* = held for 45 consecutive
steps; *unsettled* = reached at any step (judged at its lowest-error step). The run's own verdict (`success` in `result.json`)
is settled geometric.

**Per-instance results** (`results/<eval>/<name>/<instance id>/`): `result.json` (verdicts, steps, reason), `meta.json`
(everything: verdicts in both settle modes, errors, QA metrics with the steps they fired at, conditioning in
`policy_client`), `init_rgb.png`, `goal_rgb.png` (the image the policy was given), `final_rgb.png`, `prompt.txt` (text
condition), `annotations.json`, `config.yaml`, `campaign_execution_failure.json` (failed episodes); the run's log in
`_logs/<instance id>.log`; `ckpt.json` = what was asked for (the summary lists anything missing).

**Files of this folder**: `run_eval.py` (runner), `summarize.sh` (summary + zip; uses `scripts/policy_eval/summarize.py`),
`instances_100.{csv,json}` and `subsets/` (the selection), `select_instances.py` (how it was drawn), `text_prompts_iors.json`
(the text instructions), `reference_gt_100.csv` (demo-replay results on our machines), and the adapter base class
`ispatialgym/eval/llm_policy.py`.
