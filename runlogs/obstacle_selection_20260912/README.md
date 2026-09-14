# Obstacle object selection for camera_relocalization-0 (2026-09-12)

Goal (request of 2026-09-12): some `camera_relocalization-0` instances should get an **obstacle** - an extra object
dropped on the floor roughly midway along the robot's init -> goal path, so that the robot has to steer around it.
Select the obstacle categories and models: ready assets, natural on the floor of at least one of the seven candidate
scenes (`house_single_floor`, `house_double_floor_upper`, `house_double_floor_lower`, `Rs_int`, `restaurant_diner`,
`hotel_suite_large`, `office_cubicles_right`), common-sense things a robot avoids while navigating (bins, potted
plants, ...) that also make sense in the middle of the path, and not so large (sofa) that they cannot be sampled on
the floor or would block the whole path. Results are recorded in
`runlogs/grasp_library_20260905/select/task_object_selection.csv` (+ `.xlsx` / `.html`; rows
`camera_relocalization-0 / obstacle` and `camera_relocalization-0 / floor`, replacing the old single
`(no task objects)` row of the problem).

The obstacle is never touched, so - like the world-alignment selections - there is no grasp test; the selection is a
render-and-review of the assets in their canonical frame.

## Method

1. **Candidates** (`scripts/build_candidates.py` -> `candidate_categories.json`, `candidate_models.json`,
   `scene_presence.json`, `candidate_scan.json`; ispatialgym env): 96 categories picked by hand from (a) a geometry
   scan of every BEHAVIOR category with ready models (median canonical bbox: largest footprint side 0.12-0.9 m, height
   0.10-1.7 m -> 312 categories in `candidate_scan.json`) and (b) the object inventory of the seven scenes (every
   `<scene>_*.json` variant: which categories already stand in which rooms). Groups: bins, plants, containers (boxes,
   baskets, sacks), luggage, cleaning, seats (stools, small chairs), toys / pets, floor-standing appliances, floor
   clutter (shoes, pillows). Each category carries a hand annotation of the scenes it is natural in. 296 ready models
   (all ready models of a category up to a cap of 12; 16 for the flagship categories, 6-8 for the very large,
   low-priority ones such as pillow / gym_shoe / straight_chair / swivel_chair; models already standing in one of the
   seven scenes are kept first, the rest is a seeded random subset).
2. **Renders** (`runlogs/world_alignment_selection_20260911/scripts/render_models.py`, base conda env, 38 s for all
   models): four views per model in its canonical frame with an axis gizmo; `renders/strips/<cat>__<model>.png`,
   grids of 6 in `renders/grids/`, bboxes in `renders/render_meta.json`.
3. **Review** (`review_contract.md`, 8 parallel reviewer agents, `assign/rev_XX.json` from
   `scripts/make_assignments.py`, outputs `review/<category>.json`): per model V visual 0-5, N naturalness as a
   mid-path floor obstacle 0-3 (3 = commonly stands in walkable space and is unremarkable mid-path; 2 = plausible but
   normally against a wall / small clutter; 1 = odd; 0 = not floor-standing or not in these scenes), S size fit 0-2
   (2 = footprint 0.20-0.55 m and height 0.20-1.10 m; 1 = footprint 0.12-0.20 / 0.55-0.80 m or height 0.10-0.20 /
   1.10-1.70 m; 0 = beyond; grounded in the R1Pro chassis half-diagonal 0.33 m and ~1 m corridors), `rest_ok` (stands
   stably in its canonical pose, +z up; balls and lying / hanging models fail); `score = 2V + 2N + S` (max 18),
   `valid = V >= 3, N >= 2, S >= 1, rest_ok`; plus the scenes the category is natural in and a category verdict.
   Reviewer notes worth keeping: the sizes were read from the strip headers (visual-mesh extents); they match the
   knowledge-base bboxes recorded in the sheet within 5 % for every model except `water_cooler/essxzr` (glass bottle
   omitted by the renderer, not selected). The first assignment lists contained two grids of other reviewers'
   categories through prefix matching (`plant_pot_*` -> plant_pot_stand, `bucket_*` -> bucket_of_paint); the
   reviewers skipped them, and `make_assignments.py` now matches category names exactly.
4. **Selection** (`scripts/select_obstacles.py` -> `selection.json`, `selection.md`, `spec_fragments.json`):
   categories ranked by the mean score of their two best valid models (single valid model: score - 2; `weak`
   verdict: - 1); round 1 = the 10 best with >= 2 valid models, round 2 = the next 5 with >= 2 valid, round 3 = the
   next 5 with >= 1 valid. Rounds are ranking tiers: the obstacle is never manipulated, so no overlap rule with other
   tasks' categories is applied (overlaps from `scripts/used_categories.py` -> `used_categories.json` are only
   recorded; none of the 20 selected categories is a target elsewhere, `carton` is an articulation category of
   configuration_matching-1/-2). Two documented overrides (`OVERRIDES` in the script) for obstacle-type diversity:
   `music_stool` (round 1 -> 2) swapped with `vacuum` (2 -> 1) because `taboret` already covers stools, and
   `highchair` (both models are counter stools) replaced in round 3 by `backpack` (score 16, natural in all seven
   scenes, single ready model). Every other reviewed category is ranked under "not selected" in `selection.md`
   (58 more categories have >= 1 valid model; 116 valid models outside the selection) so the pool can be extended.
5. **Duplicates** (`runlogs/world_alignment_selection_20260911/scripts/wa_duplicates.py --models valid_models.json`
   -> `duplicates.json`): geometry signatures of the 109 valid models of the 20 selected categories: no exact
   duplicates, 3 texture-variant pairs (trash_can djgllo / gilsji and uknjdm / vasiit, wicker_basket dgkhyn / unasxd),
   flagged in the sheet's `duplicates` column.
6. **Sheet** (`scripts/update_selection_csv.py`): rewrites the `camera_relocalization-0` block of the sheet (header
   `obstacle` row with the method, every valid model of the 20 categories as pick 1 / pick 2 / alternate / candidate,
   `pair` = the scenes the category is natural in with `*` for scenes whose files already contain the category, a
   `floor` row) and regenerates the xlsx / html with the shared renderer; legend item in `select/legend_extra.json`.
   Idempotent; the rest of the sheet is untouched (verified byte-identical).

## Outcome (96 categories / 296 models reviewed; 78 categories with >= 1 valid model, 225 valid models)

| round | categories (valid models; pick 1, pick 2) |
|---|---|
| 1 | trash_can (15; cdzyew, vasiit), hamper (15; clziqw, xjftlf), straight_chair (8 of 51 ready; amgwaw, eospnr), taboret (6; hugvml, ysrhuw), footstool (2; mysfnz, sghekm), ottoman (4; miftfy, xcmniq), poinsettia (3; qdcipo, eefcsd), pot_plant (12 of 27 ready; imhfnc, kxmvco), public_trash_can (10; duugbb, glzckq), vacuum (3; wikhik, bdmsbr) |
| 2 | music_stool (2; dqvlff, bjtbrp), wicker_basket (5; drxfmr, jqfftz), carton (4; cdmmwy, ylrxhe), teddy_bear (4; faxvha, mvwvlv), serving_cart (2; heqtlk, vvtcby) |
| 3 | alphabet_abacus (1; wojwvu), doll (1; fhdgrp), toy_car (3; ksijjl, nhtywr), swivel_chair (8 of 56 ready; jhipgz, aqobyj), backpack (1; gvbiwl) |

109 valid models are listed in the sheet (20 pick 1, 17 pick 2, 14 alternates, 58 candidates). Per-category scenes,
scores, bboxes and reviewer notes: `selection.md` and the sheet rows.

Caveats:
* The sampler has no obstacle role yet (`ispatialgym/sampling/tasks/camera_relocalization.py` binds the floor only;
  `problem0.bddl` has floor + agent). `spec_fragments.json` holds a ready-made block for
  `task_custom_lists.json["camera_relocalization-0"]`: `roles.obstacle.models` (the picks), `_pool` (every valid
  model per category), `_natural_scenes` (where each category may be drawn), `_rounds`. Nothing was merged into the
  spec.
* `swivel_chair` footprints (0.59-0.75 m) and the larger ottomans / serving carts are borderline for a ~1 m corridor:
  drop them in open rooms. Teddy bears lie on their backs (stable, low: 0.19-0.33 m); `carton` pick 1 is a small box
  (0.19 m tall), pick 2 a 0.64 m box. `alphabet_abacus`, `doll` and `backpack` are single-model categories.
* Balls (soccer_ball, volleyball, bowling_ball), the loose plants (marigold, pottable_dahlia), folding_chair (folded
  canonical poses), magazine_rack (desktop holders), garage-only tools and the > 0.8 m items (stroller, pet_bed,
  cooler, camera_tripod) have no valid model; see the `drop` rows of `selection.md`.
* The category pool is deliberately wide (96); the not-selected ranking includes many good single-model categories
  (tote, ice_bucket, litter_box, suitcase, compost_bin, dehumidifier, electric_fan, water_dispenser, ...) and
  borderline-size classics (bucket, box_of_fruit, floor_lamp, pillow, gym_shoe) if more variety is wanted later.

## Files touched outside this directory

* `runlogs/grasp_library_20260905/select/task_object_selection.{csv,xlsx,html}` (camera_relocalization-0 block only),
  `runlogs/grasp_library_20260905/select/legend_extra.json` (one legend item added).
* Nothing committed. Backup of the sheet before the update: `/tmp/task_object_selection.backup_obst.csv`.
