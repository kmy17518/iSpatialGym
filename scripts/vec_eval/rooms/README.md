# `scripts/vec_eval/rooms` — the rooms-override analysis scripts (campaign rooms, 2026-10-27/28)

Copies of the four analysis scripts of campaign rooms (`$V/campaigns/rooms/{analyze_visibility,make_gate_units,compare_gate,
decide_gate}.py`, `$V=/vision/u/minyeong/projects/isg_home/vec_eval85`) with the campaign's hard-coded paths turned into
arguments. The logic is unchanged; every default is the campaign's layout, so running a script with no arguments on the
cluster reproduces the campaign's outputs. The procedure they belong to — what a room override is, the two passes, the file
format, how the evaluators consume it, the adoption rule — is `docs/ROOMS_OVERRIDE_HOWTO.md`.

Not copied: `finish.sh` (the campaign's Vast / R2 teardown and mirror; fleet-specific) and `ROOMS_REPORT.md`
(`$V/docs/ROOMS_REPORT.md`). The simulator side of the procedure is not a script here: it is `run_isg_vec` of the rooms
worktree (`--seg-visibility`, `--dump-frames-spec` (vec_next; `--dump-frames` in `$R`), `--load-rooms <list>`, `--rooms-override`; see the how-to, §0).

## Common arguments

| argument | default | meaning |
|---|---|---|
| `--root` | `$ISG_VEC_ROOT`, else `/vision/u/minyeong/projects/isg_home/vec_eval85` | the vec_eval root: `<root>/instances/{task_ids.json,eval/<task>.json,all/<task>.json}` |
| `--campaign` | `<root>/campaigns/rooms` | the campaign folder: `seg/`, `gate/`, `summary/`, `units.json` under it |
| `--instances` | `<root>/instances` | the instance lists folder (`scripts/vec_eval/build_instance_lists.py --out <root>` writes it) |

The template meta a script needs (`task_rooms`, `load_room_instances`, `scene`, `variant`) is read from the eval instance's
TRO path in `<instances>/eval/<task>.json`: `<template dir>/<template id>-meta.json` next to `.../json/<...>_instances/`.

## Order and what each does

1. **Segmentation pass** (not here): `run_isg_vec --seg-visibility ...` per alias, every demo, N=1 →
   `<campaign>/seg/<alias>/<episode>/rooms_visibility.json` (how-to §2.1).

2. **`analyze_visibility.py`** — per task: a neighbour room of the template (`load_room_instances` minus `task_rooms`) is
   *seen* when its non-always-loaded objects covered ≥ `--min-frac` of a frame (default 0.0002 = 104 px of 720²) in any
   sampled step of any demo; candidate rooms = task rooms + seen neighbours + `--keep` rooms (pass `--keep garden_0`: the
   exterior is never dropped since the sunflower gate).
   Inputs: `<campaign>/seg/<task>/*/rooms_visibility.json`, `<instances>/eval/<task>.json` (→ template meta),
   `<instances>/task_ids.json` (the task list) unless `--tasks a,b`.
   Outputs: `<campaign>/summary/visibility.md` (table), `<campaign>/summary/rooms_override_candidate.json`
   (`{min_frac, generated, overrides: {task: {rooms, task_rooms, template_rooms, dropped, scene, demos, seen, seen_not_in_template}}}`).

3. **`make_gate_units.py`** — for every candidate task with a non-empty `dropped` (or all with `--all`): one fleet unit of
   kind `vec_runs` = three `run_isg_vec` replays on ONE host in sequence (`template`, `override` = `--load-rooms <candidate
   rooms>`, `template_repeat` = the host's noise floor) of the first three instances of `<instances>/all/<task>.json`
   (eval + 2 val), each `--num-envs 1 --dump-frames-spec 0-60:5,25 --record-view tiled --tile-px 240 --external-size 240
   --grid-video --limit-batches 3`. Tasks in `TWO_WORLDS` (stockpot) get one run per world group (`<mode>_wg0`, `<mode>_wg1`).
   Inputs: `--candidate` (default `<campaign>/summary/rooms_override_candidate.json`), `<instances>/all/<task>.json`.
   Output: appends to `--units` (default `<campaign>/units.json`; created when missing), units sorted by (priority, −frames);
   uploads nothing. The `vec_runs` kind exists in the rooms worktree's `eval_worker.py` (`run_cmd_unit`), not in this
   checkout's fleet worker; without it, run a unit's three `runs` by hand: `run_isg_vec --out <campaign>/gate/<task>/<name>
   <args>` (how-to §2.2).

4. **`compare_gate.py --task <alias> --a template --b override`** and **`--a template --b template_repeat --no-dataset`** —
   per episode common to both runs: the verdicts (`success`, `symbolic_success`, `geometric_success`,
   `geometric_success_step`, …) and `verdicts_equal`, the trajectory deviation (`compare_slots.diff`: max position m /
   rotation rad per tracked key), and per dumped step and camera the mean |Δ| at the dump resolution and at 224² bilinear
   (the policy input), the fraction of pixels differing by > 2 / > 8, the max. With `--dataset` (default on) the val
   episodes' head frames are also measured against the recorded demo frames of the LeRobot dataset (`--dataset-root`,
   default `/vision/u/minyeong/projects/isg_home/train85/datasets/isg-1020-b1k`; `build_instance_lists.decode_frame`).
   Inputs: `<campaign>/gate/<task>/{<a>,<b>}/<episode>/{meta.json, trajectory.npz, frames/step_NNNNN_<cam>.png}`,
   `<instances>/all/<task>.json`, `compare_slots.py` + `build_instance_lists.py` from `--scripts-dir` (default: this
   checkout's `scripts/vec_eval`).
   Output: `<campaign>/gate/<task>/gate_<a>_vs_<b>.json` (`{task, a, b, episodes: {name: {verdict_a, verdict_b,
   verdicts_equal, rooms_a, rooms_b, traj, frames: {cam: {rows: [{step, mean_abs, mean_abs_224, max_224, frac_gt2, frac_gt8,
   max}], ...}}, head_vs_demo}}, summary}`) and one printed line.

5. **`decide_gate.py`** — the rule per gated task: physics = `verdicts_equal` and trajectory ≤ 0.02 m / 0.05 rad
   (`POS_TOL`, `ROT_TOL`); frames = for head / left / right on the steps ≤ 60, the per-step excess of the override's mean
   |Δ|₂₂₄ over the repeat's (both vs the template) has mean ≤ 0.25/255 and p90 ≤ 0.6/255 (`EXCESS_MEAN`, `EXCESS_MAX`).
   Later frames and the external camera are reported, not decisive.
   Inputs: `<campaign>/gate/<task>/gate_template_vs_{override,template_repeat}.json`, `--candidate`,
   `<instances>/eval/<task>.json` (→ the template's `variant`).
   Outputs: `<campaign>/summary/gate.md` (table + before / after / repeat video index), `<campaign>/summary/rooms_override.json`
   (passing tasks only: `{generated, source, rule, overrides: {alias: {rooms, variant, template_rooms, dropped, scene,
   evidence}}}`; `--source` names the campaign), `<campaign>/summary/gate_rows.json` (every gated task, pass or fail).

Typical invocation for one new alias (the how-to, §8):

```bash
R=scripts/vec_eval/rooms; ROOT=/vision/u/minyeong/projects/isg_home/vec_eval85; C=$ROOT/campaigns/rooms_<date>
python $R/analyze_visibility.py --root $ROOT --campaign $C --tasks <alias> --keep garden_0
python $R/make_gate_units.py    --root $ROOT --campaign $C --tasks <alias>        # or run the three replays by hand
python $R/compare_gate.py       --root $ROOT --campaign $C --task <alias> --a template --b override
python $R/compare_gate.py       --root $ROOT --campaign $C --task <alias> --a template --b template_repeat --no-dataset
python $R/decide_gate.py        --root $ROOT --campaign $C --source "campaign rooms_<date> (segmentation pass + same-host gate)"
```

## Changes vs the campaign copies

- `V`, `C`, `R`, `D`, `DATASET` constants → `--root` / `--campaign` / `--instances` / `--seg` / `--gate` / `--out` /
  `--units` / `--candidate` / `--scripts-dir` / `--dataset-root`, defaults as above (`D` of `analyze_visibility.py` was unused
  and is gone).
- `analyze_visibility.py --tasks a,b`: the task list without `<instances>/task_ids.json` (a new alias need not be in it).
- `make_gate_units.py`: `units.json` is created as `{"units": [], "plan": ""}` when missing (the campaign appended to the
  segmentation units' file); the instance paths written into the units are made absolute.
- `compare_gate.py`: `compare_slots` is imported after the arguments are parsed (from `--scripts-dir`); the old boolean
  `--dataset / --no-dataset` is kept and the dataset path is `--dataset-root`.
- `decide_gate.py --source TEXT`: the `source` field of `rooms_override.json` (was the 2026-10-28 campaign string, still the default).
- `__import__("time")` → `import time`. Nothing else.
