# AGENTS.md — the iSpatialGym eval release

This checkout (`/vision/u/minyeong/projects/ispatialgym-eval`, branch `ispatialgym-eval`) is the eval release: the
iSpatialGym code plus the 83 eval instances, and only the data that loading, replaying and the oracle need for them.
Development (the sampler's grounding, new tasks, demo production) happens in the main `ispatialgym` checkout and its
worktrees.

## Names

Task family → task → subtask → variant → instance, e.g. `alignment` → `alignment_axial` → `alignment_axial_board_game` →
`alignment-axial-board_game-breakfast_table` → instance 0. A subtask is a task with its objects by category (no model or
scene object ids), the support left out (`ispatialgym.eval.names.subtask_name`); `isg_customs.json` is keyed by subtask. A
variant is the sampler's grounded setup of a subtask, its objects and support: a BDDL problem's `(:variant ...)`, an
instance file's `task_variant`, a sidecar's and the template metadata's `variant`. A subtask may have several variants;
the release has one per subtask. Use these names in code, docs and output. Files written before 2026-10-14 (the eval runs,
the main checkout's samples) spell the variant the old way; the readers marked "legacy spelling" accept both
(`datasets/conversions/2026-10-14_alias_to_variant/NOTES.md`).

## What is where

| path | what |
|---|---|
| `Behavior-1K/datasets/ispatialgym-eval-instances/` (untracked) | the 83 instances, laid out like `2026-challenge-task-instances` next to it; `metadata/episodes.jsonl` = each instance's demo and its length; no BDDL (an instance names its problem in `ispatialgym/assets/bddl/`); its `README.md` describes the layout |
| `Behavior-1K/datasets/ispatialgym-eval-demos/` (untracked) | the instances' demos (`demo.hdf5`), the paths of the instances' |
| `ispatialgym/eval/instances.py` | the dataset index: name + instance indices → instance files, demo, step budget (`list`, `show`); goal images (`set-goal`) |
| `ispatialgym/eval/names.py` | task family / task / subtask names from a variant |
| `ispatialgym/oracle/run_isg.py` | load / replay / oracle / policy runs (`--task NAME --instance-indices ...`) |
| `scripts/eval_instances/` | the exporter (`export_counted.py`), the release's runs (`release_runs.tsv`), `env.sh` and the GPU job script |
| `ispatialgym/assets/` | BDDL problems (numbered from 0 per activity), iddl files, `isg_customs.json`, `icl/revolution_axes.json` and the custom assets of the 83 |
| `runlogs/grasp_library_20260905/` | the grasp library entries of the 83, plus 7 the oracle unit tests use (`cands`, `results`, `review`) |
| `datasets/conversions/` (untracked) | records of the conversions (BDDL re-indexing, subtask names, the dataset index before them; the rename to "variant": tool, notes, diff) |
| `datasets/release_records/` (untracked, internal) | where each goal image came from (`goal_images/<instance id>-goal.json`: replay, success, sha256, size, camera); not published with the dataset |
| `ispatialgym/harness/`, `scripts/harness/` | the policy harness (separate work in progress) |

## Environment and jobs

- Conda env `ispatialgym-vectorized-envs`; `scripts/eval_instances/env.sh` sets `PYTHONPATH` to this checkout and its
  `Behavior-1K/` and `OMNIGIBSON_DATA_PATH` to `Behavior-1K/datasets` (symlinks to the shared BEHAVIOR data, plus
  `ispatialgym-assets` → `ispatialgym/assets/datasets/ispatialgym-assets`, and the two eval datasets). Never write into
  the shared datasets. In a batch job it also sets up the warm Kit app data (and puts `HOME` inside it); in an
  interactive allocation's shell (the editor's shells run in one) it does not (`ISG_KIT_APPDATA=1` / `0` forces it).
- Isaac Sim runs go through the job script (`sbatch scripts/eval_instances/job_run_isg.sh ...`), submitted from the
  repository root. It uses the warm Kit app-data helper (`/vision/u/minyeong/projects/cluster_tools/kit_appdata.sh`) and
  excludes viscam7 / viscam11.
- `run_isg` step budget: `--max-steps` (the actions; default for a dataset instance `max(300, 2 x` its demo's length`)`,
  else 3000), `--settle-steps` (default 0.2 x `--max-steps`), `--wallclock` (default 1 s x `--max-steps`); `--settle-eval`
  defaults to off for `--mode policy` (the binary goal at the end of the settle window decides) and on for oracle / replay.
- No-simulator tests: `python -m pytest -q ispatialgym/tests`.

## Rules

- The datasets are written only by `export_counted.py` and `python -m ispatialgym.eval.instances set-goal`. A goal image
  is the final frame (`final_rgb.png`) of a successful replay of the instance's demo: the eval run's re-run replay, else
  its original replay, or a replay you name with `--run` (another image size); never a frame rendered straight at the
  demo's final state (darker: the renderer's bounced light depends on what it rendered before) and never the demo's own
  frame. `set-goal` registers it in the instance file and refreshes the sidecar's `tro_sha256`. `run_isg` writes its runs
  to `--out` (default `./runs`) and skips its goal / sidecar write-backs for dataset instances.
- Never read or commit `.env`. Check what you stage; do not push unless asked.
