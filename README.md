# iSpatialGym — eval release

iSpatialGym builds on [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
(our fork [kmy17518/BEHAVIOR-1K](https://github.com/kmy17518/BEHAVIOR-1K), branch
`ispatialgym`, checked out under `Behavior-1K/`) and upgrades the motion-planning
backend from the StanfordVL cuRobo fork (v1 API) to upstream **NVlabs cuRobo
v0.8.0 (cuRoboV2)**.

This checkout is the eval release: the code and the 83 counted eval instances (one per subtask; 7 task families,
23 tasks), with only the data that loading, replaying and the oracle need for them.

## Eval instances

The instances live next to BEHAVIOR's challenge instances in OmniGibson's data folder,
`Behavior-1K/datasets/ispatialgym-eval-instances/` (`$OMNIGIBSON_DATA_PATH/ispatialgym-eval-instances`, or
`$ISPATIALGYM_INSTANCES_PATH`), laid out like `2026-challenge-task-instances`; the demos replays use are the sibling dataset
`ispatialgym-eval-demos/` (`$ISPATIALGYM_DEMOS_PATH`). The dataset's `README.md` lists the tasks and describes the files.
Names go task family → task → subtask → instance, e.g. `alignment` → `alignment_axial` → `alignment_axial_board_game` → 0
(a subtask is a task with its objects by category, the support left out; `ispatialgym.eval.names`). `isg_customs.json` is
keyed by subtask. As in BEHAVIOR, the BDDL problems are not in the dataset: an instance names its problem
(`<activity>-<id>`), loaded from this repository's `ispatialgym/assets/bddl/activity_definitions` (numbered from 0 per
activity).

```bash
python -m ispatialgym.eval.instances list --level task          # tasks and their instance counts
python -m ispatialgym.eval.instances show alignment_axial 0 1   # what the indices resolve to (files, goal image, demo)

# GPU job, submitted from the repository root (scripts/eval_instances/env.sh sets up the environment); each instance
# replays its own demo (--record video: no demo.hdf5 / LeRobot copy of the replay)
sbatch scripts/eval_instances/job_run_isg.sh --task alignment_axial --instance-indices 0 1 --mode replay --record video --out runs/replay

# the goal image at another size: replay at that size, then make the run's final frame the goal image
sbatch scripts/eval_instances/job_run_isg.sh --task alignment_axial --instance-indices 0 --mode replay \
    --image-size 720 --record video --out runs/goal_720
python -m ispatialgym.eval.instances set-goal alignment_axial 0 --run runs/goal_720
```

`run_isg --task NAME --instance-indices I [J ...]` loads instances by name and index, as BEHAVIOR's `eval.py` does with a
task name and instance ids; a task's instances are its subtasks' instances in index order. For an eval dataset instance
`--max-steps` (the actions: policy steps, a replay, the oracle program) defaults to `max(300, 2 x` its demo's length from
`metadata/episodes.jsonl` `)` and `--wallclock` to 1 s per step. Success is **geometric** by default (the BDDL goal plus
every `(:measure ...)` atom within its tolerance at the same step(s); `--success-metric symbolic`: the goal alone), with the
`isg-v2` tolerances computed at load (`--tolerance-scale`, default 3 for policy runs; `ispatialgym.eval.tolerance_profile`).
`--settle-eval` (default: off for `--mode policy`, on for oracle / replay) decides whether success needs a settled window:
off, the first step it holds; on, the first step every `pose_settled` atom holds with the goal (and the measured targets)
held through its `--settle-steps` window. `--settle-steps` is that window in every mode: an oracle program's epilogue
(exactly that long, reserved in the step budget; default 45), the first that many of a replayed demo's settle rows (default
all of them), a policy run's window (default 45). The episode ends at the geometric success, else at `--max-steps` (an
oracle program: plus its epilogue). A settle-eval-on run also reports the off mode (`meta.other_mode`, null with off). `--qa` (on by default, `--no-qa`
turns it off) measures BEHAVIOR's demo QA metrics plus the former demo gates and logs them in `meta.qa` with the steps each
fired at. The HARD metrics are robot touching objects or itself, pause, jerky motion, ghost hand, object tilt, goal
occlusion, robot wobble and bad release. The SOFT ones are held object drift and stalled, gripper in view and failed grasp
(oracle only), head camera upright, and arm / hand contacts. A HARD metric ends an oracle demo as failed; in replay and
policy mode every metric is only logged and the goal decides. Held object drift, stalled, gripper in view and failed grasp
are also the oracle's own retry rules: the oracle retries the action and fails the demo once the retries are used up,
`--qa` or not, and QA logs every verdict. Settle steps (the settle windows, a replayed demo's settle rows, the oracle's hold
and settle waits) never count as a pause or a stall. `--record-view` picks the videos, each written to `<view>_rgb.mp4`: `head`, `external`
(BEHAVIOR's camera over the robot's shoulder), `left_wrist`, `right_wrist` and `tiled` (BEHAVIOR's JoyLo replay video:
the wrist and head cameras over the external one). Each instance's goal image is the
head camera's last frame of a successful replay of its demo (1080 x 1080), named by `files.goal_rgb` in the index and
registered in the instance file (`ispatialgym.goal_observation`: `rgb_path`, size, camera; what the policy harness reads);
which replay it came from is an internal record, `datasets/release_records/goal_images/` (untracked, not part of the
dataset). `scripts/eval_instances/export_counted.py` rebuilds both datasets from the eval runs listed in
`scripts/eval_instances/release_runs.tsv`.

## Quick start on a new machine (the `ispatialgym` conda env)

Requirements: Linux x86_64, an NVIDIA GPU with a driver that supports Isaac Sim
5.1 (RTX-class, driver >= 535), `conda` (Miniconda/Miniforge) and `git`. Budget
roughly **60 GB of disk**: ~15 GB for the env (Isaac Sim + torch), ~33 GB for the
BEHAVIOR-1K assets, ~3 GB robot assets, ~1 GB for the two git checkouts.

```bash
git clone git@github.com:kmy17518/iSpatialGym.git ispatialgym
cd ispatialgym

# Clones Behavior-1K/ (fork, branch `ispatialgym`) if missing, creates the conda env,
# installs Isaac Sim / OmniGibson / bddl / cuRoboV2, downloads the datasets,
# pip-installs this repo (editable) and installs the custom `ispatialgym-assets` dataset.
./setup.sh --new-env ispatialgym --omnigibson --bddl --dataset --dev --primitives \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos

source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ispatialgym

# Sanity checks: no-simulator unit tests and the instance index
pytest -q ispatialgym/tests/test_eval_instances.py ispatialgym/tests/test_geometry_and_targets.py
python -m ispatialgym.eval.instances list
```

Notes:
- `Behavior-1K/` is a separate git checkout (gitignored here). To use a different
  clone URL/branch set `ISPATIALGYM_B1K_URL` / `ISPATIALGYM_B1K_BRANCH` before
  running `setup.sh`; if SSH to GitHub is not set up the clone falls back to HTTPS.
- Datasets go to `Behavior-1K/datasets/` unless `OMNIGIBSON_DATA_PATH` is set.
  Put them on a persistent volume on cloud machines; they are ~37 GB. The
  BEHAVIOR-1K assets come from the Hugging Face Hub: export `HF_TOKEN=<your token>`
  before `setup.sh --dataset` for authenticated (faster, rate-limit-free) downloads.
- The oracle reads the grasp library entries of the eval instances (43 keys, plus 7 that
  the oracle unit tests use) from `runlogs/grasp_library_20260905/{cands,results,review}`
  (committed; override with `ISPATIALGYM_GRASP_LIBRARY`). Sampling new instances needs the data this release
  leaves out (the full grasp library, reach bands, scene inventory, candidate lists):
  use the main `ispatialgym` checkout for that.
- `environment/ispatialgym.pip-freeze.txt` is the package list of the dev
  workstation's env, for debugging version drift (do not install from it).

## Setup

`setup.sh` is a wrapper around `Behavior-1K/setup.sh`. It accepts **all** of the
upstream flags and behaves like it, with these differences: it clones
`Behavior-1K/` (our fork) when missing, `--primitives` installs NVlabs cuRobo
`v0.8.0` instead of the fork, it repairs upstream dependency gaps so the install
completes end-to-end, and `--omnigibson` also installs this repo (`pip install -e .`)
plus the `ispatialgym-assets` dataset.

```bash
# Full install (with motion planning via cuRoboV2)
./setup.sh --new-env --omnigibson --bddl --dataset --joylo --eval --primitives

# Core components only (no motion planning)
./setup.sh --new-env --omnigibson --bddl --dataset

# Non-interactive (auto-accept Conda TOS / NVIDIA EULA / dataset license)
./setup.sh --new-env --omnigibson --bddl --dataset \
           --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos

# See all options (forwards to the upstream help)
./setup.sh --help

# Preview the resolved plan without installing anything
ISPATIALGYM_DRY_RUN=1 ./setup.sh --new-env --omnigibson --bddl --primitives
```

## How the cuRobo upgrade works

- The wrapper **intercepts `--primitives`** so the upstream script never installs
  the fork and never runs its `nvcc` pre-check. It runs the upstream installer
  for everything else, then installs cuRoboV2 separately into the same env:

  ```bash
  pip install "nvidia-curobo[cu12] @ git+https://github.com/NVlabs/curobo@v0.8.0"
  ```

  (`cu13` is used automatically when `--cuda-version 13.x` is passed.)
- cuRoboV2 no longer compiles CUDA extensions at install time — it JIT-compiles
  at runtime via `cuda.core` — so `--primitives` no longer requires a system CUDA
  toolkit just to install.
- `Behavior-1K/` is left untouched, so it stays easy to sync with upstream.

## Dependency remediation (automatic)

Upstream OmniGibson imports `av` (PyAV) and `lerobot.datasets` at import time, but
neither `av` nor HuggingFace `datasets` is in its `install_requires` — so a plain
`--omnigibson` install can't even `import omnigibson` (which also aborts the
dataset download). The wrapper intercepts `--dataset`, installs the missing core
deps with Isaac-Sim-safe pins, then runs the dataset download itself. Pins
applied (validated against Isaac Sim 5.1 + OmniGibson 3.8 + cuRoboV2):

| package | pin | why |
| --- | --- | --- |
| `av` | `>=15,<16` | av 16+ removed the `av.option` API lerobot uses |
| `datasets` | `>=4,<5` | range required by `lerobot@release/b1k` |
| `pillow` | `==11.3.0` | `datasets` pulls pillow 12; Isaac Sim/OmniGibson need 11 |
| `pandas` | `<3` | `datasets` pulls pandas 3.x |
| `numpy` | `<2` | Isaac Sim / OmniGibson requirement |
| `packaging` | `<26` | cuRoboV2 pulls packaging 26, violating lerobot's `<26` |

## Tests

All non-primitives OmniGibson test files in the CI matrix pass in this env
(`test_controllers`, `test_object_states`, `test_transition_rules`,
`test_symbolic_primitives`, `test_sensors`, `test_envs`, `test_multiple_envs`,
`test_robot_states_flatcache`, `test_snapshots`, `test_transform_utils`, etc.).
Primitives/curobo tests (`test_primitives`, `test_curobo`, the `curobo_example`)
are excluded — they require the v1 cuRobo API. Two non-CI files
(`test_behavior_task`, `test_data_collection`) fail on pre-existing upstream
issues unrelated to this work (`test_data_collection` is the one upstream CI
disables, tracked in StanfordVL/BEHAVIOR-1K#2016).

## ⚠️ Scope / known limitation

cuRoboV2 is a **ground-up rewrite with a breaking public API**. OmniGibson's
motion-planning integration (`omnigibson/action_primitives/curobo.py`, the robot
`*_description_curobo_*.yaml` configs, and their consumers) still targets the
cuRobo **v1** API.

This wrapper only guarantees that cuRobo `v0.8.0` **installs cleanly**. Running
primitives (motion planning) at runtime requires a separate v1 → v2 port of the
OmniGibson integration and is **not** handled here.
