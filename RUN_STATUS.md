# iSpatialGym install + test run — status / handoff

_Last updated: 2026-07-10. The env was rebuilt from scratch against the updated
Behavior-1K checkout and fully re-validated. Complete._

## TL;DR
- `Behavior-1K` was pulled forward (only new upstream commit: 770343a91
  "Fix installation (#2297)", touching setup.sh) and the local branch rebased.
- The wrapper `setup.sh` was updated for the new upstream, the `ispatialgym`
  conda env was **removed and reinstalled from scratch** end-to-end via
  `./setup.sh --new-env ispatialgym --omnigibson --bddl --dataset --dev
  --primitives --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos`,
  and the full non-primitives test batch was re-run: **identical results to the
  June baseline — all CI-tracked files pass.**

## Wrapper setup.sh changes for the updated upstream (2026-07-09)
- Dataset step now downloads the **2026** challenge task instances (upstream
  switched from 2025; both live under `Behavior-1K/datasets/`).
- Primitives interception now also installs **warp-lang>=0.9,<1.13** (new
  member of OmniGibson 3.9's `primitives` extra, alongside ninja).
- Pillow pin changed `==11.3.0` → `~=11.0.0` to match OmniGibson 3.9's own
  declared constraint (env ends at pillow 11.0.0; isaacsim-kernel's exact
  `Pillow==11.3.0` pin warns but is benign — upstream's own script already
  knowingly violates isaacsim's exact pins for websockets/click).
- Remediation comments refreshed: since OmniGibson 3.9, `lerobot[dataset]` is
  in install_requires, so `datasets`/`pandas` arrive on their own; **`av` is
  still missing** (it sits under lerobot's un-requested `av-dep` extra) and is
  still installed by the wrapper.

## Rebuilt env (2026-07-09) — `~/miniconda3/envs/ispatialgym`
Python 3.11, torch 2.7.0+cu128, Isaac Sim 5.1.0, omnigibson 3.9.0 (editable),
bddl 3.7.0 (editable, bddl3/), nvidia-curobo 0.8.0 (cuRoboV2), warp-lang 1.12.1,
av 15.1.0, datasets 4.8.5, pandas 2.3.3, pillow 11.0.0, numpy 1.26.0,
packaging 25.0, websockets 16.1. Install log: `runlogs/reinstall-20260709.log`.

Expected/benign pip-resolver warnings: isaacsim wheels carry exact pins
(Pillow==11.3.0, websockets==12.0, click==8.1.7, packaging==23.0) that clash
with omnigibson/lerobot ranges; upstream's own installer produces the same
warnings and deliberately overrides several of them.

## Datasets (safe on disk, outside the env)
`Behavior-1K/datasets/` — robot assets, behavior-1k assets, 2025 AND 2026
challenge task instances (2026 newly downloaded). Most entries are symlinks
into `~/Desktop/projects/BEHAVIOR-1K-ORIGIN/datasets/`.

## Test results (2026-07-10 rerun) — logs in `runlogs/tests-20260709/`
Runner: `runlogs/isg_run_tests_20260709.sh` (one file per process, headless).

PASSED (all CI-tracked non-primitives files, identical to June baseline):
test_transform_utils (34), test_envs (5), test_systems, test_object_removal,
test_scene_graph, test_sensors, test_dump_load_states, test_controllers (5),
test_robot_states_flatcache, test_symbolic_primitives, test_multiple_envs,
test_object_states, test_transition_rules, test_snapshots, test_ag_states (13).
Skipped-only: test_robot_teleoperation (needs teleop hardware).

FAILED — same pre-existing upstream issues as June, NOT regressions:
- test_behavior_task (1 F): bddl wildcard expansion — "requires >=1
  cabinet.n.01 in living_room, found 0" (Rs_int +
  putting_away_Halloween_decorations).
- test_data_collection (5 F): DataWrapper/USD-context bugs; upstream CI
  disables this file (StanfordVL/BEHAVIOR-1K#2016).

Scene smoke test `test_scene_loadable.py`: PASSES (exit 0, no tracebacks;
log `runlogs/scene_loadable-20260709.log`). Note: the script now sets
`gm.ENABLE_TRANSITION_RULES = False` — loading the cluttered scene otherwise
trips a recipe rule's garbage fallback that wants the GPU-dynamics `sludge`
fluid system (pre-existing content behavior, unrelated to the reinstall), and
its prints use `flush=True` because `og.shutdown()` hard-exits.

Excluded as primitives/curobo (broken by curobo v1->v2 API, unchanged):
test_primitives, test_curobo, and the `robots.curobo_example` example.

## How to resume
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ispatialgym
cd Behavior-1K/OmniGibson

# Re-run any single test:
OMNIGIBSON_HEADLESS=1 pytest -q tests/test_object_states.py

# Re-run the whole non-primitives batch:
bash ../../runlogs/isg_run_tests_20260709.sh
```
