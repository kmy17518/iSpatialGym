# World-horizontal object selection (2026-09-12)

Goal: create `world_alignment-2` (variant `world_horizontal`), the sibling of world vertical (`world_alignment-1`,
`runlogs/world_alignment_selection_20260911`): an object hinged flat on a wall at its centre of mass starts tilted in
the wall plane and is pushed back to **level**. This run selects the target categories / models of rounds 1-3, reuses
the world-vertical wall regions, records both in `runlogs/grasp_library_20260905/select/task_object_selection.csv`
(+ `.xlsx` / `.html`; rows `world_alignment-2 (world_horizontal)` / `target`, `wall`, `floor`) and smoke-tests the
new sampler. Code / docs of the problem: `ispatialgym/sampling/CHANGELOG.md` 1.5.0, `docs/instance_sampling_spec.md`
§5.9, `docs/task_bddl_formats.md` §6c, `docs/task_definition_and_evaluation.md`.

Rounds (rules of the 2026-09-11 request applied to the new problem): round 1 = up to 10 categories used by no other
task's round 1 (>= 2 valid models), round 2 = up to 5 categories that may overlap other tasks' round-1 categories
(>= 2 valid), round 3 = up to 5 categories with any overlap (>= 1 valid). `used_categories.json`
(`scripts/used_categories.py`) lists what the other tasks use per round, **including** the table-surface and
world-vertical rounds of the world-alignment run.

## Objects

1. **Candidates** (`scripts/build_candidates.py` -> `candidate_models.json`, `candidate_categories.json`,
   `candidate_scan.json`): a geometry scan of every BEHAVIOR category with ready models (median bbox long extent
   0.2-1.5 m, long / mid >= 1.4, thin <= 0.30 m: 373 categories) plus a name search for wall / rod vocabulary, then
   picked by hand: 40 categories, 135 ready models - wall-native elongated items (towel_rack, hook_rack, shelf,
   wall_mounted_tv, whiteboard, bulletin_board, address, signs, license_plate, hanger, radiator, range_hood,
   air_conditioner, wall_mounted_light, frames / boards, landscape pictures and paintings) and rod-like objects that
   are displayed horizontally on wall hooks (baseball_bat, golf_club, fishing_rod, skateboard_deck, longboard, guitar,
   trumpet, violin, tennis_racket, rolling_pin, ruler, drumstick, poker, axe, umbrella, handle, rake, shovel). Per
   category only models that can hang in front of the head camera were kept (`MODEL_FILTER`: <= ~1.4 m, no tall
   vertical radiators / shelving units, no mini rulers).
2. **Renders** (`../world_alignment_selection_20260911/scripts/render_models.py`, base conda env, 14 s): four views per
   model in its canonical frame with the RGB axis gizmo, `renders/strips/<cat>__<model>.png`, grids of 3 in
   `renders/grids/` (134 strips; `shelf/apiaid` has no visual mesh).
3. **Review** (`review_contract.md`, 8 parallel reviewer agents, `assign/rev_XX.json`, outputs `review/<category>.json`):
   per model V visual 0-5, **L level readability 0-3** (one dominant axis / edge that must be horizontal: 3 = bar /
   rail / rod / shelf / board with aspect >= 2:1 / a text line; 2 = aspect 1.5-2:1 or a content cue; 1 = ambiguous
   which edge is level; 0 = round / square), **H hangability 0-2** (flat against the wall, 0.25-0.90 m along the level
   axis, normal wall item or commonly displayed on wall hooks), `score = 2V + 2L + H` (max 18), `valid = V >= 3,
   L >= 2, H >= 1`, plus the canonical frame `front_axis` (out of the wall), `level_axis` (along the wall when level),
   `up_axis`, `thin_axis` and `symmetric_180` (looks the same upside down), and a category verdict. Unlike world
   vertical a 180-degree ambiguity is allowed (horizontal is undirected); a 90-degree one (square) is not.
4. **Selection** (`scripts/select_wh_objects.py` -> `selection.json`, `selection.md`, `spec_fragments.json`): categories
   ranked by the mean score of their two best valid models (single valid model: - 2, weak verdict: - 1), rounds filled
   per the rules (round 1 first from categories used by no other task at all, then from categories only used in
   other tasks' rounds 2-3). One documented override (`OVERRIDES`): `wall_mounted_light` is excluded - the reviewer
   gave H = 1 to the two 0.20 m-deep track lights and flagged it; the contract sets H = 0 above 0.15 m, which
   leaves only the vertical tube sconce valid (< 2 models). The same reviewer note applies to `trumpet` (0.16 m
   deep, single model), which is not selected anyway.
5. **Duplicates** (`../world_alignment_selection_20260911/scripts/wa_duplicates.py` -> `duplicates.json`): geometry
   signatures of the 69 valid models of the selected categories: no exact duplicates, 6 texture-variant pairs
   (picture x 2, radiator, ruler, signpost, wall_mounted_tv) flagged in the sheet's `duplicates` column.

Outcome (every category, model, score and frame in `selection.md`; 18 categories, 69 valid models):

| round | categories (cat score) |
|---|---|
| 1 (8; no further category with >= 2 valid models is free of other tasks' round 1) | rolling_pin (17), baseball_bat (16), shelf (15), radiator (14), drumstick (14, weak), wall_mounted_tv (13.5), umbrella (13), whiteboard (12.5) |
| 2 (5) | license_plate (18), picture (16), signpost (16), towel_rack (16), hanger (15) - all world-vertical round-1/2 categories |
| 3 (5) | decorative_sign (15), guitar (14), ruler (14, weak), address (14, 1 model), axe (14, 1 model) |

Not selected (22 categories): 15 with a single valid model (shovel, tennis_racket, bookcase_shelf, bulletin_board,
handle, hook_rack, longboard, rake, trumpet, poster, board, golf_club, poker, skateboard_deck, violin), painting
(2 valid, landscape canvases of aspect ~1.4 with weak level cues), wall_mounted_light (override above) and 5 drops
(air_conditioner, fishing_rod, frame, range_hood, wall_mounted_loudspeaker: boxy, over 1.3 m or no readable level
edge).

Caveats: the BEHAVIOR library has few flat wall items with a dominant horizontal axis and >= 2 ready models, so
round 1 already contains a weak category (drumstick = hairline 1.2-1.5 cm rods; umbrella and hanger hang flat
only as a display, H = 1) and the archetypal wall bars
(towel_rack), text plates (signpost, license_plate) and landscape pictures sit in round 2 because world vertical
already uses them in its rounds 1-2. Rods (bats, pins, drumsticks) have an arbitrary front face; `front_axis`
was read as the side with the most detail. Each target row of the sheet notes the flat wall patch the model needs
(its extents swept between a 45 deg tilt and level + 0.10 m), to be compared with the wall rows' patch sizes
(0.6-1.2 m): the compact picks (towel_rack 0.58 m, rolling_pin 0.52 m, license_plate 0.44 m, decorative_sign
0.46 m, ...) fit every validated region, the wide ones (guitar/cybisc 1.20 m, picture/chfljy 1.18 m,
whiteboard/bqqmtc 1.17 m, shelf/xogqqm 1.01 m, baseball_bat/yalgei 0.92 m) only the largest.

## Walls

Same walls as world vertical (request: "the wall can be the same"): the 78 regions validated by the wall survey of
`runlogs/world_alignment_selection_20260911` (`wall_candidates.json`, `scenes.<scene>.task_spec` = ready-to-paste
`fixed.wall` / `fixed.wall_regions` blocks) are written as the `wall` rows of the new block, each annotated with the
horizontal pick-1 models whose swept extent fits its flat / clear patch (`scripts/update_selection_csv.py`). The
horizontal sampler seeds its spots along the same regions and re-checks flat patch and 0.35 m front clearance for
the target's swept extents (spec §5.9 item 4), so no new survey was run; a region that admits the painting reference
target does not necessarily admit a 1 m rail (the annotation says which do).

## Sampler and smoke test

`WorldHorizontalSampler` (`ispatialgym/sampling/tasks/world_alignment.py`, registered for `world_alignment-2`,
`bddl_writer` variant `world_horizontal`, template `assets/bddl/activity_definitions/world_alignment/problem2.bddl`)
subclasses the world-vertical sampler: `generation.level_axis` is checked against the frame derived from
`front_axis` / `up_axis`, `generation.symmetric_180` adds the half turn about the front axis to the goal's
`orientation_equivalences`, and the wall patch tests use the swept extents (`generation.swept_patch`, default true;
`template_meta.swept_dims`). Test spec (`ispatialgym/assets/task_specs/task_custom_lists.json`) pins
`towel_rack/{kqrmrh, crivoa}` on house_single_floor living_room_0 (wall found by the sampler, as for world vertical).

Smoke test (2026-09-12, `smoke/test_spec.json`, Rs_int entryway validated region via `fixed.wall_regions`,
`OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.sample_instances --spec smoke/test_spec.json --problem
world_alignment-2 --scene Rs_int --n-instances 1 --out /tmp/isg_wh_test --overwrite`, then `render_instances`):
`towel_rack/kqrmrh` (level 0.47 x 0.06 m, swept patch 0.48 x 0.38 m, symmetric_180 -> `orientation_equivalences`
`[[1, 0, 0, 0]]`, instance at -24 deg after two `grasp_unreachable` attempts) and `hook_rack/mdnfjr` (front `-x`,
level 0.96 x 0.13 m, swept patch 0.97 x 0.77 m, instance at -32 deg, first attempt), 7 s per template; TROs, summaries
and init renders in `smoke/` (the bar / rail hang flat on the wall, tilted, robot on the arc). Unit tests:
`ispatialgym/tests/test_problem_parsing.py` (59 passed, includes the new problem file).

## Sheet update

`scripts/update_selection_csv.py` inserts / rewrites the three `world_alignment-2 (world_horizontal)` blocks right
after the world-vertical blocks of `task_object_selection.csv` (all other rows byte-identical), merges two legend
items into `select/legend_extra.json` and regenerates the xlsx / html with the shared renderer
(`runlogs/table_selection_20260911/scripts/update_selection_csv.py`). Re-run order after changing reviews:
`select_wh_objects.py` -> `wa_duplicates.py --models <valid models json> --out duplicates.json` ->
`update_selection_csv.py` (base env for the last two).

## Decisions taken without a check-in (adjust if wanted)

* Problem id `world_alignment-2`, variant `world_horizontal`; hanging height, arc, deviation range (15-45 deg),
  tolerance (0.08 rad) and `min_visible_objects` copied from world vertical.
* 180-degree-symmetric models are accepted (flagged `symmetric 180`, evaluated with the flip as an equivalence);
  the level goal for models with a readable top keeps them upright.
* World-vertical categories may enter world horizontal's round 2 (the general round-2 rule), unlike the wa0 / wa1
  disjointness kept when those two were selected together; round 1 stays free of every other task's round 1.
* Test spec pins `towel_rack` (a round-2 category here, but the most natural wall bar and the template synset)
  rather than the top-scoring round-1 category (`rolling_pin`, a kitchen item in a living room).
* `wall_mounted_light` excluded by override (contract thickness rule); drop the override in
  `scripts/select_wh_objects.py` and re-run the three scripts to restore it as the 9th round-1 category.
