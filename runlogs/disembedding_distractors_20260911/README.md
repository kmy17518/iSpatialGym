# Disembedding distractor pools per target category (2026-09-11/12)

Goal: for each of the 39 object categories that have disembedding test instances (`disembedding-0 / target` in
`runlogs/grasp_library_20260905/select/task_object_selection.csv`), a **same-category distractor pool** and an
**other-category distractor pool**, 20 candidate models per category in total, where

* same-category models that **look identical from the robot's initial head-camera view** are marked as one
  *look-alike group* (a group counts as one candidate; when a member is the target, the other members are never drawn
  as distractors - the robot has to recognise the target from that first image),
* other-category distractors are semantically sensible (forks and knives for spoons, rice / sugar boxes for cereal
  boxes, ...), and
* every pool member was validated with the instance sampler (the initialization script): one disembedding instance
  was generated with it as the distractor of the category's test target.

## Pipeline (`scripts/`)

1. `build_candidates.py` (no simulator) -> `candidates.json`. Same-category candidates = every ready BEHAVIOR model
   of the category (341, 317 fit the tray) with the grasp-library / selection data of the disembedding round and the
   geometry-duplicate verdicts of `select/duplicates.md`; other-category candidates = ready models of hand-curated
   related categories (`RELATED`, tier 1 = closest, ~120 categories), at most 3 per category, interleaved so the
   list stays diverse. Size filter for the test source tray `tray/gsxbym` (0.31 x 0.44 m): footprint (+2 cm)
   <= 0.045 m^2 (0.051 for same-category), longest side <= 0.30 m, height <= 0.36 m, >= 3 cm.
2. `validate_distractors.py --phase render` (OmniGibson, house_single_floor dining room, test table + trays):
   the disembedding template of the test spec is set up (`DisembeddingSampler.setup`: stance, trays, viewing trunk),
   then every unique candidate model (786) is placed alone at the centre of the source tray, settled for 60 physics
   steps and photographed by the head camera at the init stance at 4 yaws (same-category candidates) or 2 yaws. Crops
   of the tray region: `renders/<category>__<model>__y<k>.png`; settling / visibility check: `renders/meta/*.json`.
   Launcher `run_phase.sh` waits for free GPU memory (the GPU is shared with the table-selection and world-alignment
   runs) and restarts after crashes; the entry point is named `dis_validate_tables.py` so that the table-selection
   queue counts it as one of its two GPU slots.
3. `compose_grids.py` -> `grids/same/<category>__p<k>.png` (one row per model: label + the 4 yaw crops at native
   head-camera resolution), `grids/same_contact/`, `grids/diff/` (contact sheets). Reviewer agents judge the
   look-alike groups per category following `review_contract.md` -> `groups/<category>.json`.
4. `assemble_pools.py` -> `distractor_pools.json`, `distractors_section.json` (spec form), `distractor_pools.md`.
   Pool rule: same-category look-alike groups first (ranked: test picks, selection score, AG-valid grasps, untested;
   at most 10 groups, every validated member of a selected group listed), related-category models fill up to 20
   (tier order, at most 3 per category, exact duplicates folded). Validated extras are `reserve`.
5. `validate_distractors.py --phase validate --pairs <pools+reserves>`: for every target category the template is set
   up once with the test pick as target (`DisembeddingSampler.setup`, the pick-1 model itself is validated as a
   distractor of pick 2), then each candidate is swapped in as the only distractor and `TaskSampler.run(1)` generates
   one instance exactly like `sample_instances.py` (sample -> apply -> settle -> BEHAVIOR validation -> `:init` true ->
   goal false -> occlusion measurement -> goal feasibility by teleport + settle). `results/<target>__<cat>-<model>.json`
   holds status, rejections, occlusion, footprint budget and the instance summary digest; TROs / summaries under
   `/tmp/isg_dis_distractors/`. Step 4 is re-run on the results so that only validated models remain in the pools.
6. `update_selection_csv.py` writes the pools into `task_object_selection.csv` (+ `.xlsx` / `.html`).

## Status

See the end of this file (filled in when the run completes).
