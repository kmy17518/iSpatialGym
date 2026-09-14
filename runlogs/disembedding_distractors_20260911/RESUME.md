# Resume notes (updated 2026-09-12 10:50)

## State

* Render phase DONE: 786 unique candidate models rendered from the disembedding init stance (`renders/`, 783 ok;
  `chalk/{dcosbg,hpnnry,fdmgww}` have no asset on disk). Grids composed (`grids/same`, `grids/same_contact`,
  `grids/diff`, `grids/pool_other`).
* Look-alike review DONE for all 39 categories (`groups/*.json`), related-category check DONE (`groups_other/*.json`
  -> `diff_exclusions.json`, 24 exclusions, target-specific).
* Preliminary pools written (`assemble_pools.py --prelim --reserve-other 4` -> `distractor_pools.json`,
  `distractors_section.json`, `distractor_pools.md`, `pairs.json` = 1031 (target, distractor) pairs).
* Interim sheet update done (rows say "sampler validation IN PROGRESS"): `select/task_object_selection.{csv,xlsx,html}`
  (backup of the pre-update CSV: `/tmp/isg_dis_distractors/task_object_selection.before_pools.csv`).
* Sampler validation RUNNING since 10:46: `scripts/run_phase.sh validate --pairs pairs.json` (log
  `/tmp/isg_dis_distractors/logs/validate.log`, wrapper log `validate.wrapper.log`, results in `results/`, ~1 pair
  per 10-30 s plus ~4 min of IK tables per category; expected ~6 h). Resumable: relaunch the same command
  (`--skip-done` is implied by the wrapper). bar_soap: both test targets fail the template setup on the test table
  (no graspable + placeable cell for a 3 cm flat soap) -> its pools stay render-checked only.

## To finish (after `validate.wrapper.log` says `validate complete`)

```bash
cd /home/minyeongk/Desktop/projects/ispatialgym; D=runlogs/disembedding_distractors_20260911
~/miniconda3/envs/ispatialgym/bin/python $D/scripts/assemble_pools.py --reserve-other 4      # final pools (validated only)
python3 $D/scripts/update_selection_csv.py                                                  # sheet (system python has openpyxl)
```
Then update `README.md` (status section) with the numbers from `distractor_pools.md`, and decide about
`ispatialgym/assets/task_specs/task_custom_lists.json`: `distractors_section.json` is a drop-in replacement of the
`disembedding-0.distractors` section written by the parallel agent (same format; `other_models` is supported by
`spec.py`).

## Sampler fix made here

`ispatialgym/sampling/tasks/disembedding.py::_tune_occluder` looked the target's hull / radius up by BDDL instance
name (stored by role) and treated the trays as obstacles -> every occluded draw raised KeyError (all disembedding
instances generated so far were unoccluded). Fixed + CHANGELOG entry.
