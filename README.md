# iSpatialGym

iSpatialGym builds on [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
(our fork [kmy17518/BEHAVIOR-1K](https://github.com/kmy17518/BEHAVIOR-1K), branch
`ispatialgym`, checked out under `Behavior-1K/`) and upgrades the motion-planning
backend from the StanfordVL cuRobo fork (v1 API) to upstream **NVlabs cuRobo
v0.8.0 (cuRoboV2)**.

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

# Sanity checks: no-simulator unit tests, then one sampled task instance (launches Isaac Sim headless)
pytest -q ispatialgym/tests/test_problem_parsing.py ispatialgym/tests/test_geometry_and_targets.py
OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.sample_instances \
    --problem inter_object_relation_satisfaction-0 --n-instances 1 --max-templates 1 --seed 0 \
    --out /tmp/isg-smoke
```

Notes:
- `Behavior-1K/` is a separate git checkout (gitignored here). To use a different
  clone URL/branch set `ISPATIALGYM_B1K_URL` / `ISPATIALGYM_B1K_BRANCH` before
  running `setup.sh`; if SSH to GitHub is not set up the clone falls back to HTTPS.
- Datasets go to `Behavior-1K/datasets/` unless `OMNIGIBSON_DATA_PATH` is set.
  Put them on a persistent volume on cloud machines; they are ~37 GB. The
  BEHAVIOR-1K assets come from the Hugging Face Hub: export `HF_TOKEN=<your token>`
  before `setup.sh --dataset` for authenticated (faster, rate-limit-free) downloads.
- The sampler reads the validated grasp library from
  `runlogs/grasp_library_20260905/{cands,results,review,artic/handles}` (committed;
  override with `ISPATIALGYM_GRASP_LIBRARY`) and the object selections under
  `runlogs/*_selection_*/` (committed, images excluded).
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
