# IORS-4 (touching) target / partner selection (2026-09-11)

Goal: choose the object categories and models of the `target` and `partner` roles of
`inter_object_relation_satisfaction-4` (alias `touching`: the target is pushed - or grasped and set down - until it is in
rigid contact with the partner, both released; see `ispatialgym/docs/task_bddl_formats.md` section 7 and the sampler
`ispatialgym/sampling/tasks/iors.py`, which puts the partner on the robot's left with a 10-25 cm gap, translates the target
sideways until the hulls touch and rejects goals where the partner moves > 5 mm), with the same target / partner rules as
IORS-3 nextto, and record them in `runlogs/grasp_library_20260905/select/task_object_selection.csv` (+ `.xlsx` / `.html`).

## Rules

| | target | partner |
|---|---|---|
| ready (no unprocessed KB complaint), single rigid link, on disk | yes | yes |
| pinchable in the rest pose: smallest horizontal extent <= 8.9 cm (antipodal pipeline width limit) | yes (the spec-pinned `can_of_baking_mix` is 9.4-10.4 cm: push-only, kept because pinned) | not required |
| stays put when pushed / bumped: flat base (>= 25 % of the footprint in contact, or a foot >= 0.6 x / 0.45 x the body width), height <= 3 x / 2.5 x the smallest horizontal extent, no rolling body | yes | yes |
| size: long side <= 30 cm, height <= 30 cm | yes | fits every scene support validated for IORS by `runlogs/table_selection_20260911` + the test table (8 supports): short side <= min depth - 6 cm = 33.6 cm, long side <= min length / 2 = 38.6 cm, height <= 35 cm |
| semantically paired | target -> partner pairs below | |

Rounds (the user's rules): round 1 = 4 categories per role, unused by any other task's round 1 (checked against the
grasp-library selection manifest, every role of the sheet, the spec pins and the parallel world-alignment candidate list),
>= 2 distinct valid models; round 2 = 2 categories per role that may overlap with another task's round-1 categories,
>= 2 valid models (chosen among round-1 categories not yet reused by a third task); round 3 = 2 categories per role with
any overlap, single-model categories allowed.

## Result

| round | target -> partner | theme | target models (valid / ready) | partner models (valid / ready) |
|---|---|---|---|---|
| 1 | can_of_baking_mix -> board_game | test-round spec pair (pinned) | 6 / 6 (2 geometries, push-only 9.4-10.4 cm cans) | 16 / 16 |
| 1 | box_of_baking_powder -> kitchen_analog_scale | baking | 2 / 2 (one tin, two prints) | 2 / 2 |
| 1 | box_of_coffee -> jar_of_grains | pantry | 2 / 2 | 12 / 12 (one 22 cm jar, 12 fillings) |
| 1 | box_of_vegetable_juice -> pop_case | beverages | 2 / 2 | 6 / 13 (7 fridge packs are 39.4 cm long) |
| 2 | box_of_sanitary_napkins -> box_of_tissues | bathroom paper goods (disembedding r1 categories) | 9 / 9 | 4 / 4 (one soft pack, 4 prints) |
| 2 | carton_of_milk -> box_of_cookies | milk and cookies (disembedding r1 categories) | 3 / 4 (slim 4.5 cm brick tips) | 14 / 14 |
| 3 | teacup -> teapot | tea set | 3 / 6 (3 rounded-base cups rock) | 3 / 3 (plain grey; `kettle` is the textured alternative) |
| 3 | bottle_of_medicine -> first_aid_kit | medicine | 7 / 8 (the cardboard box tips) | 1 / 1 |

Picks (pick 1, pick 2): can_of_baking_mix blrqqz, orwvfx (pinned); board_game eoywpy, gcfkqa (pinned); box_of_baking_powder
vzgrlv, zevydc; kitchen_analog_scale zpmpoq, lpjpez; box_of_coffee cjtadw, mreyla; jar_of_grains vuolos, uydvnv;
box_of_vegetable_juice jsnnlv, qgjdbn; pop_case rzhvvk, gwgaqb; box_of_sanitary_napkins tkrzkb, acaivd; box_of_tissues
uglbjc, zutrxn; carton_of_milk kszoro, atyqub; box_of_cookies dlhqft, apjokz; teacup cpozxi, oxfzfe; teapot foaehs,
jlalfc; bottle_of_medicine zfbnjh, kqkwoq; first_aid_kit flctdz.

Targets of rounds 2-3 reuse the tabletop grasp test of their category (disembedding r1 / r2 and IORS-1 r2 result keys,
AG-valid and reviewer-clean counts, selection scores and reviewer notes from `select/out/*.json`); the round-1 targets are
fresh categories without a grasp test (the touching oracle pushes; a pick-and-place is optional) and were checked on
textured renders. Weak points: `can_of_baking_mix` is not pinchable (push-only); `jar_of_grains` and `box_of_tissues` are a
single geometry with texture variants; `teapot` models are untextured grey; `box_of_vegetable_juice` cartons are small
(4.6-6.5 cm wide, 11-13 cm tall, aspect up to 2.8) and should be pushed low.

Considered and not taken: paintbrush -> bucket_of_paint (brush 1 cm thick), toilet_paper -> tissue_dispenser (untextured
rolls / plain dispensers), can_of_beans (two ids, one asset), glass bottles and flasks (breakable, round bases), rolling
produce and pencil cases, knives and other < 2.5 cm thick objects; test_tube_rack, shoe_box, storage_box, gift_box,
desk_phone, loudspeaker, plant_pot, bucket_of_paint, hiking_boot are usable partners without a fresh pinchable target.

## Files

* `scripts/used_categories.py` -> `used_categories.json`: categories used by the other tasks per round.
* `scripts/candidate_pool.py` -> `candidate_pool.json`, `pool_table.txt`: every rigid non-scene category with ready
  single-link models, median bbox, per-model target / partner geometric flags, validated supports and limits.
* `render_models*.json` + `runlogs/world_alignment_selection_20260911/scripts/render_models.py` -> `renders/strips`,
  `renders/grids`: textured 4-view renders (usd-core + pyrender) of 496 candidate models.
* `scripts/model_geometry.py` -> `model_geometry.json`: visual-mesh bbox, base contact fraction / base hull size, vertex
  hash and diffuse-texture md5 per candidate model (duplicates).
* `scripts/select_touching.py` -> `touching_selection.json` (per-model evidence, rules, rejected models with reasons) and
  the rewritten IORS-4 `target` / `partner` blocks of the sheet (idempotent; xlsx / html regenerated with the
  table-selection `render()`).
