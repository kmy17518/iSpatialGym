# World-alignment object and wall selection (2026-09-11)

Goal: for `world_alignment-0` (table surface: an object on a table is yawed back into alignment with the table edge) and
`world_alignment-1` (world vertical: an object hinged on a wall at its centre of mass is pushed back upright) select the
target object categories / models of selection rounds 1-3, and for `world_alignment-1` find the wall regions of the
seven candidate scenes where the instance sampler can initialise the task. Results are recorded in
`runlogs/grasp_library_20260905/select/task_object_selection.csv` (+ `.xlsx` / `.html`; rows `world_alignment-0 / target`,
`world_alignment-1 / target`, `world_alignment-1 / wall`).

Rounds (request of 2026-09-11): round 1 = 10 categories per task, not overlapping any other task's round-1 categories,
>= 2 valid models; round 2 = 5 categories, may overlap other tasks' round-1 categories, >= 2 valid models; round 3 = 5
categories, any overlap, >= 1 valid model. `used_categories.json` (`scripts/used_categories.py`) lists what the other
tasks use per round (grasp-library selection manifest + partner / spec rows of the sheet).

## Objects

1. **Candidates** (`candidate_categories.json`, `candidate_models.json`): 117 categories picked from the BEHAVIOR
   category list by name and canonical bbox (table-flat objects with a readable front: books, keyboards, pans,
   boards, boxes, tools, shoes ...; wall items: paintings, pictures, clocks, signs, boards, instruments, racks ...),
   up to 16 ready models per category (459 models).
2. **Renders** (`scripts/render_models.py`, base conda env: usd-core + trimesh + pyrender/EGL, no OmniGibson, 55 s
   for all models): four views per model in its canonical frame (`from +x`, `from -x`, `from -y`, top-down) with an
   RGB axis gizmo, textured from the assets' diffuse maps (glass parts omitted). `renders/strips/<cat>__<model>.png`,
   grids of 3 models in `renders/grids/`.
3. **Review** (`review_contract.md`, 8 parallel reviewer agents, `assign/rev_XX.json`, outputs `review/<category>.json`
   / `<category>__wa1.json`): per model V visual 0-5, O orientation readability 0-3 (table) or U up readability 0-3
   (wall), F table fit 0-2 or H hangability 0-2, `score = 2V + 2O + F` / `2V + 2U + H` (max 18), `valid = V >= 3,
   O/U >= 2, F/H >= 1`, plus the canonical frame (`long_axis` / `front_axis` / `rest_ok`; `front_axis` / `up_axis` /
   `thin_axis`) and a category verdict.
4. **Selection** (`scripts/select_wa_objects.py` -> `selection.json`, `selection.md`): categories ranked by the mean
   score of their two best valid models, rounds filled per the rules (the spec-pinned keyboard / painting in round 1;
   wa0 and wa1 disjoint in rounds 1-2), a few documented manual overrides (`OVERRIDES` in the script: gym_shoe <->
   newspaper between rounds 1 / 2, paper-thin menus replaced by clipboard / valentine_wreath in round 3).
5. **Duplicates** (`scripts/wa_duplicates.py` -> `duplicates.json`): geometry signatures of the 194 valid models of the
   selected categories (same method as `select/duplicates.md`): no exact duplicates; 15 texture-variant clusters
   (36 models) flagged in the sheet's `duplicates` column.

Outcome (see `selection.md` for every category, model, score and frame):

| task | round 1 | round 2 | round 3 |
|---|---|---|---|
| world_alignment-0 (table surface) | paperback_book, hardback, magazine, comic_book, letter, bag_of_chips, frying_pan, keyboard, catalog, newspaper | board_game, gym_shoe, notebook, boxed_ink_cartridge, carton_of_eggs | calculator, chopping_board, digital_scale, notepad, clipboard |
| world_alignment-1 (world vertical) | wall_clock, painting, picture, signpost, decorative_sign, fire_extinguisher, guitar, hanger, picture_frame, towel_rack | license_plate, umbrella, paper_towel_dispenser (only 3 categories with >= 2 valid models remain) | poster, dartboard, whiteboard, first_aid_kit, valentine_wreath |

Sampler 1.4.1 smoke tests (2026-09-12, `/tmp/wa_test_spec.json`): `world_alignment-1` with `picture/jmvdxm` (canonical
frame flat, `front_axis +z`, `up_axis -x`) and `painting/dacihj` on the validated Rs_int entryway region via
`fixed.wall` / `fixed.wall_regions` (3 spots seeded from the region, one instance per template, picture hangs upright
in the render); `world_alignment-0` with `hardback/agbqsy` (`long_axis x`) and `keyboard/injrgc` on the test table
(one instance each). `rooms_for` in `sample_instances.py` now skips the `wall_regions` list when collecting rooms.

Caveats: the BEHAVIOR library has few wall-hangable categories with one readable upright and >= 2 ready models, so
`world_alignment-1` round 1 already includes weak categories (picture_frame = tabletop easel frames, towel_rack =
bars with a subtle cue) and round 2 has only three. Categories whose canonical frame lies flat (picture, license_plate,
guitar, hanger, dartboard, fire_extinguisher, umbrella, valentine_wreath, first_aid_kit) need the sampler's
`generation.up_axis` / per-category `front_axis` (sampler 1.4.1) - the frames are in `selection.md` and in the sheet's
`pair` column. Mirrors have no ready model (all carry unprocessed complaints) and were not usable.

## Walls (world_alignment-1)

`scripts/survey_walls.py` (one OmniGibson process per scene, one room load per indoor room, restart wrapper
`scripts/run_survey_all.sh`, templates / TROs under `/tmp/isg_wall_survey`):

1. Dense wall spots: horizontal rays at the hanging height (1.45 m) from a 0.3 m lattice of the room's free floor in
   24 directions, hits on `walls` whose front lies in the room, one spot per 0.25 m of wall.
2. The sampler's own checks per spot (`WorldVerticalSampler`): flat patch and 0.35 m front clearance at square sizes
   0.4-1.2 m (`max_patch_m`) and for the reference target (painting/dacihj, 0.50 x 0.42 m); traversable, in-room,
   whole-body collision-free base poses on the 0.6-0.95 m arc (capped at 12 per spot for the survey); right-arm IK
   pushing reach of the aligned target's lower corners. Viable spots are merged into regions per wall segment.
3. One full sampler instance per region (`TaskSampler.run`: init pose + robot on the arc, settle, BEHAVIOR
   validation, `:init` true, goal false, >= 5 objects visible from the head camera, hinge drift, goal feasible by
   teleport + settle, TRO written); a second spot of the region is tried when the first fails; long regions also get
   their quarter points.

`scripts/aggregate_walls.py` -> `wall_candidates.json` (per scene / room / wall the regions with geometry, size
admitted, bases, validation; `scenes.<scene>.task_spec` = the `scenes` block for `task_custom_lists.json` with
`fixed.wall` / `fixed.wall_regions`, which sampler 1.4.1 understands), `wall_candidates.md` (tables, rejection reasons
per room), `maps/<scene>_walls.png` (floor plan with every spot, region and validated instance).
Per-room evidence: `results/<scene>__<room>.json`. Outcome (2026-09-12): 54 indoor rooms surveyed in the 7 scenes, 155 regions, 78 validated by a full instance (85.9 m of wall): house_single_floor 7 rooms / 21 regions, house_double_floor_lower 3 / 10, house_double_floor_upper 4 / 10, Rs_int 1 / 1 (entryway; the living-room regions fail with `scene_disturbed`, the arc base displaces the coffee table), hotel_suite_large 2 / 7, office_cubicles_right 8 / 25, restaurant_diner 1 / 4 (bar; the dining room regions fail `too_few_visible` / `grasp_unreachable`). `DONE.json` (written by `scripts/finalize.py`) holds the timestamp and the summary of everything produced.

## Sheet update

`scripts/update_selection_csv.py` rewrites the three world-alignment blocks of `task_object_selection.csv` and
regenerates the xlsx / html with the shared renderer (`runlogs/table_selection_20260911/scripts/update_selection_csv.py`,
legend additions in `select/legend_extra.json`).

## Addendum 2026-10-11: tablespoon (alignment-axial#05)

The new sheet (`ispatialgym/assets/icl/roles/representatives.csv`, owner 2026-10-10) makes `tablespoon/tqqwys` the
alignment-axial#05 target (variation_geometry `pltxjb`, held-out `ubuhoz`), a category this selection never reviewed.
Same renderer and contract: `scripts/render_models.py` unchanged (four canonical-frame views per model; run in an env
with `pxr` + `pyrender` on USDs decrypted beforehand), `review/tablespoon.json` (wa0 scores; V as the grasp-library
review of the same models gave it), the three models appended to `renders/render_meta.json`. `frying_pan/jpzusm` was
re-rendered as the reference (handle at -x, reviewed front -x; its `render_meta.json` entry came out identical).
Frame of all three: long axis `x`, bowl at -x with its opening toward +z, handle toward +x, so `front_axis` = `+x` =
the handle end (the frying pans' rule: the handle faces the robot); `rest_ok`: the canonical pose (+z up) rests on the
bowl back and the handle end. Cross-checked on the meshes (surface profile along x): bowl 5.7 / 4.4 / 2.6 cm wide at
-x, handle 0.8-2.1 / 1.3-2.1 / 0.6-1.2 cm wide toward +x (tqqwys / pltxjb / ubuhoz), the bowl's centre 4-7 mm below its
rim. The icl rows are `alignment-axial / target` in `roles/candidates.csv` (`frame_front +x`, `frame_up +z`,
`frame_long x`). No contact rows: `require/contacts.csv` stays header-only (B6), as for every alignment target.
