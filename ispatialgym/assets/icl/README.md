# icl — iSpatialGym candidate list

The candidate list that `ground_isg` grounds the iddl files with (`ispatialgym/assets/iddl/`). Built from the selection
runlogs by `scripts/icl/build_icl.py` (repeatable; re-run after editing the sources, never edit the CSVs by hand — add an
override in the script instead). Every row starts with `status = untested`: the old sampler's "viable" verdicts are
kept as provenance (`source_selection`) only; `ground_isg` writes the real per‑scene status into the task customs
list (whitelists / blacklists / constraints).

Categories are BEHAVIOR category names (synsets only appear in the grounded BDDL); `display_name` is the phrase used
in language annotations.

## Files

### `roles.csv` — one row per candidate of a (subtask, role)

| column | meaning |
|---|---|
| `subtask` | iddl subtask id (`disembedding`, `inter_object_relation_satisfaction-ontop_small_scale`, …) |
| `role` | iddl role: `target`, `targets`, `partner`, `source`, `destination`, `distractors`, `obstacles`, `support`, `room` |
| `kind` | `object` (one spawned instance), `group` (member candidate of a group role), `fixed` (scene object / wall region / room) |
| `category`, `model` | BEHAVIOR category and model id; for `fixed` rows `model` is the scene object name (or the wall object / room) |
| `display_name`, `synset` | language‑annotation phrase; KB synset of the category |
| `scene`, `room` | fixed rows: where the object is |
| `representative`, `test`, `rank`, `source_selection` | `pick 1` / pinned rows are the representative models (`ground_isg` tests them first); `pick 1` + `pick 2` + pinned are the test set; `rank` 0 pinned, 1 pick 1, 2 pick 2, 3 alternate, 4 candidate/viable/partner, 5 pool, 6 reserve |
| `status` | `untested` for every row (see above) |
| `grasp_key` | grasp‑library key (`runlogs/grasp_library_20260905/{cands,results,review}`) when the role is grasped; empty = no grasps yet (touching round‑1 targets are flagged in `note`) |
| `scale` | per‑axis scale of the grasp‑tested variant (scaled categories such as `garlic_clove`, `pencil`, `french_fries`) |
| `bbox_x/y/z` | canonical bounding box (m) from the KB inventory |
| `natural_scenes` | obstacles: scenes where the category is natural |
| `frame_front`, `frame_up`, `frame_level`, `frame_long`, `symmetric_180` | alignment targets: reviewed canonical frame (`+x` …), long axis, half‑turn symmetry |
| `joint`, `handle_key`, `handle_kind`, `handle_annotated`, `instance_scale` | articulated targets: joint(s) (`joint:status` list for scene objects), `artic/handles/<key>.json` key and whether it exists, the scene instance's scale relative to the canonical model (handles must be rescaled) |
| `region_id`, `p0`, `p1`, `normal`, `hang_z`, `length_m`, `max_object_m` | wall regions (`<room>/<wall>/<rNN>`, world xy at hang height, normal into the room) |
| `top_z`, `world_ext_x/y`, `pos_x/y`, `yaw_deg` | supports / big partners: geometry from `runlogs/table_selection_20260911/inventory.csv` |
| `dataset`, `count` | non‑BEHAVIOR assets (`ispatialgym-assets` for the block), group member count |
| `ready` | 1 = no unprocessed KB complaint |
| `note`, `source` | reviewer note (trimmed), provenance file |

Supports: the union of every scene object that was `spec`/`viable` for any task in the old sheet (31 objects in 6
scenes) is listed for every tabletop subtask; the floor of every scene is an admissible init support for the
large‑scale IORS subtasks. Wall regions: the 78 validated regions of the 2026‑09‑11 survey, for both wall subtasks.
Large partners: every bed / sofa / desk / breakfast table (+ countertop for ontop) of the inventory. Scene articulated
targets: every host object of the articulation swap tests, with `handle_annotated` and `instance_scale`.

### `pairs.csv` — how roles correlate

| relation | a → b |
|---|---|
| `partner_of` | target category (or model, attachment) → partner category / model; `attach_type` and `ready` for attachment; large‑scale IORS partners are categories (scene object chosen by `ground_isg`) |
| `distractor_same_category` | disembedding target category → same‑category distractor model |
| `distractor_other_category` | disembedding target category → other‑category distractor `category_b/model_b`, `tier` 1 = closest |

### `lookalikes.csv`

Look‑alike groups (`group_id`, `category`, `model`): members cannot be told apart from the robot's init view and are
never drawn as each other's distractors (from `task_custom_lists.json`, `disembedding-0.distractors.lookalike_groups`).

### `combos.csv` — explicit combinations

`kind = spec`: the current test set (`task_custom_lists.json` pins), `test = 1`; `kind = representative`: one combo per
(subtask, target category) with the representative target, the pick‑1 partner of its paired category and the test
support — what `ground_isg` runs first. `assignment` is `role=category/model;role=scene:scene_object;…`
(`*4` = group count, `(joint)` = articulation joint, `scene:room/*` = any validated wall region of the room).

### `grasps.csv` — every validated grasp of every referenced key

`finger_q1/2` (finger joint positions at closure), `width_m` (object width at the pinch), `pad_clearance_m` =
(0.10 − width)/2 (how far each open pad sticks out beyond the object; disembedding distractor spacing), CoM offset,
antipodal / coverage scores, tilt, reviewer verdict and flags (`one_finger` tolerated).
