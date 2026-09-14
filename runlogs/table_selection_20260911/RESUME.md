# Table pre-selection run - status

**Complete (2026-09-12 07:00).** Every pass finished: main sweep over the 7 scenes, the disembedding second pass, the
IORS-under pairs and a 15-attempt re-run of the low-yield supports. `results/` holds 1114 runs (185 viable);
`task_object_selection.csv` / `.xlsx` / `.html` were updated by `scripts/update_selection_csv.py`.

## Re-running / extending

The frozen sampler copy lives in `/tmp/isg_table_sel/snapshot` (see README "Sampler version"); the generated
templates / TROs in `/tmp/isg_table_sel/<scene>/<tag>/`. Both are outside the repo and may be gone after a reboot;
`results/*.json` keep every measurement needed for the sheet.

```bash
cd /home/minyeongk/Desktop/projects/ispatialgym
# redo one (support, problem): delete its result, then
ISG_TABLE_MAXP=2 runlogs/table_selection_20260911/scripts/queue_scenes.sh "<scene>" --include-low       # table problems
ISG_TABLE_MAXP=2 runlogs/table_selection_20260911/scripts/queue_scenes.sh "<scene>" --under             # IORS-1 pairs
# more attempts for a low-yield support
runlogs/table_selection_20260911/scripts/validate_tables.py --scene <scene> --tables <name> --problems <problem> \
    --max-attempts 30 --retries 3 --out runlogs/table_selection_20260911 --work /tmp/isg_table_sel --skip-done
# aggregate + sheet (openpyxl: base-env python)
python3 runlogs/table_selection_20260911/scripts/aggregate_results.py
~/miniconda3/bin/python3 runlogs/table_selection_20260911/scripts/update_selection_csv.py
```

Notes: `queue_scenes.sh` keeps at most `ISG_TABLE_MAXP` sampler processes and waits for >= 9 GB free GPU memory
(other agents share the GPU; a process takes 7.5-10 GB). Progress: `grep table_selection /tmp/isg_table_sel/logs/<scene>.log`.
Sampler issue found (not fixed in the repo, fixed in the snapshot): `disembedding.py::_tune_occluder` indexes
`self.hull` / `self.radius` with the target's BDDL instance name and iterates tray instance names, while the dicts are
keyed by role -> `KeyError` whenever an occluded instance gets an occluder placed.
