# AGENTS.md — working in `ispatialgym-vectorized-envs` without disturbing `ispatialgym`

**Scope check first.** These rules are for the checkout at
`/home/minyeongk/Desktop/projects/ispatialgym-vectorized-envs` on branch `vectorized-envs`, on the
host that also runs the pilot described below. If `pwd` or `git branch --show-current` disagrees,
this file is stale for where you are — stop and ask before relying on it.

This directory is a git worktree (branch `vectorized-envs`) for vectorized-environment work. On this
same machine, the primary checkout `/home/minyeongk/Desktop/projects/ispatialgym` (branch `survey`)
is running a **live pilot**: local Isaac Sim sampling jobs plus a fleet of RunPod pods driven from
that checkout. Nothing you do may affect them.

Rule of thumb: if a path starts with `/home/minyeongk/Desktop/projects/ispatialgym/` (no
`-vectorized-envs`) or `/home/minyeongk/miniconda3/envs/ispatialgym/`, it is off-limits. Read it if
you must; never write, run, or install into it.

The nested `Behavior-1K/AGENTS.md` comes from the upstream BEHAVIOR-1K fork. Where it says to use
the `behavior` conda env (or `conda run`), or documents `setup.sh` / `pip install -e`, **this file
overrides it**: the only environment here is `ispatialgym-vectorized-envs`, and neither repo's
`setup.sh` may be run.

## Layout: what is yours, what is shared

| Yours (isolated)                                                            | Shared with the main checkout (read-only for you)                                  |
|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| Worktree `…/ispatialgym-vectorized-envs`, branch `vectorized-envs`          | The `.git` object store, stash list, `origin` remote                                |
| Conda env `ispatialgym-vectorized-envs` (editable `ispatialgym`, `omnigibson`, `bddl` all point into this worktree) | Conda env `ispatialgym` — never activate it, never `pip install` into it |
| `Behavior-1K/` = a worktree of the Behavior-1K fork, branch `vectorized-envs` (its own OmniGibson/bddl source). The outer repo gitignores it, so commit its changes from inside it. | The Behavior-1K repo itself (primary worktree `…/BEHAVIOR-1K-ORIGIN`, several other worktrees) |
| `Behavior-1K/OmniGibson/appdata` (Kit portable root: caches, logs, settings) | `Behavior-1K/datasets/*` — every entry except `ispatialgym-assets` is a symlink to the main clone's data (~100 GB, one physical copy) |
| `Behavior-1K/datasets/ispatialgym-assets` → this worktree's `ispatialgym/assets/datasets/ispatialgym-assets` | `.env` — a **symlink to the main checkout's file** (RunPod/HF/W&B/GitHub tokens) |
| `<worktree>/datasets/`, `<worktree>/runlogs/`, `/tmp/vecenv-*`              | `~/.cache/ispatialgym/*` (sampler caches, unless redirected — see preflight), `~/.cache/warp`, torch/triton/HF caches |
|                                                                             | The single RTX 4090 (24 GB) and system RAM                                          |

## Preflight — run at the start of every shell, and re-verify before any Isaac Sim launch

```bash
cd /home/minyeongk/Desktop/projects/ispatialgym-vectorized-envs
source ~/miniconda3/etc/profile.d/conda.sh          # non-interactive shells (agent jobs) have no `conda` on PATH without this
conda activate ispatialgym-vectorized-envs          # conda only auto-activates base; never use `ispatialgym`
export OMNIGIBSON_HEADLESS=1 OMNI_KIT_ACCEPT_EULA=YES
# Keep sampler/oracle caches out of the shared ~/.cache/ispatialgym (a poisoned cache hit silently changes the pilot's results).
# All five are read by the code (some at import time), so export them before starting Python:
export ISPATIALGYM_CACHE_DIR=$HOME/.cache/ispatialgym-vecenv
export ISPATIALGYM_SAMPLING_CACHE=$HOME/.cache/ispatialgym-vecenv/sampling
export ISPATIALGYM_REACH_BANDS=$HOME/.cache/ispatialgym-vecenv/reach_bands
export ISPATIALGYM_REACH_VOLUME=$HOME/.cache/ispatialgym-vecenv/reach_volume
export ISPATIALGYM_HELD_HOLD_CACHE=$HOME/.cache/ispatialgym-vecenv/held_hold.json

python -c "import ispatialgym, omnigibson, bddl; print(ispatialgym.REPO_ROOT); print(omnigibson.__file__); print(bddl.__file__)"
#   -> all three lines must contain 'ispatialgym-vectorized-envs'. If any doesn't, STOP: wrong env or wrong cwd.
git branch --show-current                  # must print: vectorized-envs
git -C Behavior-1K branch --show-current   # must print: vectorized-envs
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
```

## Hard rules

**Processes.** Do not kill, signal, renice, or wait on any process you did not start. The pilot's
processes include: `scripts/runpod/remote_run.py`, `scripts/runpod/rp.py … watchdog`,
`scripts/reach_bands/pull_loop.sh`, `scripts/reach_bands/run_local.sh` chains, `/tmp/band_collect_loop.sh`,
`python -m ispatialgym.sampling.sample_isg` / `reach_bands` from the main checkout, `ssh … root@<pod>`
sessions, and `torch/_inductor/compile_worker` processes. Never run `pkill -f python`,
`pkill -f isaac|kit|omni`, `fuser -k /dev/nvidia*`, `nvidia-smi --gpu-reset`. Only kill PIDs you
launched yourself, by exact PID.

**GPU / RAM.** The 4090 is shared with local Isaac Sim jobs from the main checkout (each ~8–10 GB
GPU, ~9 GB RSS; the pilot assumes at most two per card). Before every launch run `nvidia-smi`; if
less than ~10 GB is free, wait rather than launch. Run at most one Isaac Sim process at a time from
this worktree, and size vectorized runs conservatively (small `n_envs`, scale up only after
measuring). If the card runs out of memory, it is often the *other* process that dies, not yours.

**RunPod / credentials.** Never run anything under `scripts/runpod/` (`rp.py`, `remote_run.py`,
`dispatch_samp.py`, …) or `scripts/reach_bands/pull_loop.sh`. They authenticate with the same tokens
as the pilot (through the symlinked `.env`); `remote_run.py` unconditionally rsyncs the *local
working tree* onto the pods, which would overwrite the code running there, and `rp.py terminate`
kills pods. Treat `.env` as read-only: editing it edits the main checkout's file. Do not upload to
HuggingFace, push to GitHub, or log to W&B unless explicitly asked.

**Datasets.** Everything under `Behavior-1K/datasets/` (except `ispatialgym-assets`) is the shared
data. Never write into, delete from, or re-download it: no `download_*` from
`omnigibson.utils.asset_utils`, no `setup.sh` (either repo's), no scripts that generate into the
dataset tree. New scenes/objects you need go under this worktree's `ispatialgym/assets/` or an
output dir of yours.

**Environments.** Install and upgrade only in `ispatialgym-vectorized-envs`. If you ever re-install
`omnigibson`/`bddl`/`ispatialgym` editable, point them at *this* worktree
(`pip install -e <worktree>/Behavior-1K/OmniGibson --no-deps --no-build-isolation`, etc.). Never run
`pip`/`conda` against `ispatialgym`.

**Git (both repos).** Commit only on `vectorized-envs` (in `…/ispatialgym-vectorized-envs` and in its
`Behavior-1K/`). Never check out, reset, rebase, force-move, or delete `survey`, `ispatialgym`,
`main`, or any other branch; never `git worktree remove|prune|move`; never `git gc --prune`,
`reflog expire`, or edit `.git/info/exclude` (it is shared across worktrees). Avoid `git stash` (the
stash is shared across worktrees); if you must, pop it immediately in the same worktree and never
`stash drop|clear`. Do not push unless asked. Do not merge into `survey`/`ispatialgym` — that is the
owner's decision.

**Paths.** Write outputs under this worktree (`datasets/`, `runlogs/`) or `/tmp/vecenv-*`. Do not
touch `/tmp/isg*`, `/tmp/isg_push_under/`, `/tmp/isg_dispatch_export`, `/tmp/band_collect_loop.sh`.
Do not run the tracked scripts that hardcode the main checkout's absolute path — they would read and
write the main tree: `test_scene_loadable.py`, `runlogs/isg_run_tests.sh`,
`runlogs/isg_run_tests_20260709.sh`, `runlogs/grasp_exp_20260904/scripts/*.py`,
`runlogs/grasp_library_20260905/scripts/*.py`,
`ispatialgym/data_collection/notebooks/generate_bddl_camera_relocalization.ipynb`. Re-check with
`git grep -l -F '/home/minyeongk/Desktop/projects/ispatialgym/' -- . ':!Behavior-1K'` (the remaining
hits are data files; reading them is fine).

**Shared caches.** Do not delete or rewrite `~/.cache/ispatialgym`, `~/.cache/ov`, `~/.cache/warp`,
`~/.cache/torch`, `~/.triton`, or `~/.cache/huggingface`. Your own `Behavior-1K/OmniGibson/appdata`
and `~/.cache/ispatialgym-vecenv` you may clear freely. Do not modify `~/.bashrc`, conda config, or
`~/.ssh`.

## Things that are fine

- Editing OmniGibson/bddl source under `Behavior-1K/` here — it is your own worktree and branch; the
  pilot uses the main clone's. Note the first Isaac Sim launch here is slower (cold shader cache in
  your own `appdata`).
- Running headless smoke tests here, e.g.
  `python -m ispatialgym.sampling.sample_instances --problem camera_relocalization-0 --n-instances 1 --max-templates 1 --seed 0 --out /tmp/vecenv-smoke --overwrite`
  (subject to the GPU check above).
- `pytest` on the no-simulator tests (`ispatialgym/tests/test_problem_parsing.py`,
  `ispatialgym/tests/test_geometry_and_targets.py`).

When in doubt about whether an action reaches outside this worktree, don't do it — report it and ask.
