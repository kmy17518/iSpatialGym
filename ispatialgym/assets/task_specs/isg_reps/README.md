# ground_isg (v1 bridge) output

200 grounded problems (slots: representative); one spec per alias under `specs/`, jobs in `jobs.json`.

| subtask | problems |
|---|---|
| alignment-axial | 10 |
| alignment-world_horizontal | 10 |
| alignment-world_vertical | 10 |
| camera_relocalization-obstructed | 17 |
| camera_relocalization-standard | 1 |
| configuration_matching-articulation_close_large_scale | 6 |
| configuration_matching-articulation_close_small_scale | 5 |
| configuration_matching-articulation_open_large_scale | 6 |
| configuration_matching-articulation_open_small_scale | 5 |
| configuration_matching-attachment | 6 |
| configuration_matching-block_design | 1 |
| disembedding | 30 |
| inter_object_relation_satisfaction-inside | 5 |
| inter_object_relation_satisfaction-nextto | 5 |
| inter_object_relation_satisfaction-ontop_large_scale | 5 |
| inter_object_relation_satisfaction-ontop_small_scale | 5 |
| inter_object_relation_satisfaction-touching | 4 |
| inter_object_relation_satisfaction-under_large_scale | 5 |
| inter_object_relation_satisfaction-under_small_scale | 4 |
| mental_rotation | 30 |
| object_scaling | 30 |

Provenance: grounded from the representative selection at commit `2c7b2a0` (`icl/roles/representatives.csv` of that
commit; the selection is re-run after every icl rebuild and may differ — re-run `python -m ispatialgym.sampling.v1_bridge`
to refresh the specs; the fleet run of 2026-09-15 used the files in this directory). `assignment_w*.json` = which pod ran
which job (`scripts/runpod/dispatch_samp.py`), `jobs_demo_*.json` = the `run_isg` demo jobs per pod
(`scripts/runpod/make_demo_jobs.py`). Results index: `runlogs/isg_reps_20260915/` (`scripts/isg_reps/assemble_results.py`).
