# icl — iSpatialGym candidate list

The candidate list that `ground_isg` grounds the iddl files with (`ispatialgym/assets/iddl/`). Built from the selection
runlogs by `scripts/icl/build_icl.py` (repeatable, ~10 s in the `ispatialgym` env; re-run after editing the sources,
never edit the CSVs by hand — add an override in the script instead). Categories are BEHAVIOR category names (synsets
only appear in the grounded BDDL); `display_name` is the phrase used in language annotations.

Two folders, one per question `ground_isg` asks:

```
icl/
  roles/                          which concrete categories and models can play which iddl role
    candidates.csv                one row per candidate of a (subtask, role): kind, category, model, marks, per-model facts
    constraints_compatible.csv    which candidates go together (target -> partner, target -> distractor pools)
    constraints_conflicting.csv   which candidates must not appear together / as each other (look-alike groups)
  require/                        the artefacts the iddl :require block points at — one compact row per computed pose
    grasps.csv                    (grasps <role> <pipeline>)   validated library grasps
    handles.csv                   (handle <role>)              handle / edge pinches of articulated objects
    contacts.csv                  (contacts <role>)            push / brace / stop contacts of the alignment targets (B6, empty until then)
    lib/                          symlinks to the libraries the `path` columns refer to
```

Every candidate row starts with `status = untested`: the old sampler's "viable" verdicts are kept as provenance
(`source_selection`) only; `ground_isg` writes the real per‑scene status into the task customs list (whitelists /
blacklists / constraints). What the icl does **not** hold: how many members a group has, which model is drawn when,
spawn scales, chosen joints — those are task customs list parameters (`generation`, `roles.<role>.*`).

## `roles/`

### `candidates.csv` — one row per candidate of a (subtask, role)

| column | meaning |
|---|---|
| `subtask` | iddl subtask id (`disembedding`, `inter_object_relation_satisfaction-ontop_small_scale`, …) |
| `role` | iddl role: `target`, `targets`, `partner`, `source`, `destination`, `distractors`, `obstacles`, `support`, `room` (the iddl files declare no roles; a role's rows here define it for the subtask; `room` rows bound a subtask to one scene / room and bind the implicit `floor`) |
| `kind` | how `ground_isg` grounds the role: `sampleable` (one spawned instance, model drawn per template), `group` (member candidate of a group role; member count and identical‑vs‑unique draw are customs list parameters, `generation.groups.<role>`), `scene` (scene object / wall region / room bound per scene) |
| `category`, `model` | BEHAVIOR category and model id; for `scene` rows `model` is the scene object name (or the wall object / room) |
| `display_name`, `synset` | language‑annotation phrase; KB synset of the category |
| `scene`, `room` | scene rows: where the object is |
| `representative`, `test`, `rank`, `source_selection` | `representative = 1`: the model of the category `ground_isg` tests first (pick 1 / pinned; the test support, test partner, test room and the wall regions of the test room for scene rows; attachment targets: the best‑ranked male with a READY partner); `test = 1`: the test set (pick 1 + pick 2 + the v1 pins); `rank` 0 pinned, 1 pick 1, 2 pick 2, 3 alternate, 4 candidate/viable/partner, 5 pool, 6 reserve. Explicit test combinations are the cross product of the `test` rows admitted by `constraints_compatible.csv` — there is no separate combinations sheet |
| `status` | `untested` for every row (see above) |
| `grasp_key` | preferred grasp‑library key when the role is grasped; the `(grasps <role> <pipeline>)` requirement is checked against `require/grasps.csv`, not this column; empty = no grasps yet (touching round‑1 targets are flagged in `note`) |
| `scale` | per‑axis scale of the grasp‑tested variant (scaled categories such as `garlic_clove`, `pencil`, `french_fries`) |
| `bbox_x/y/z` | canonical bounding box (m) from the KB inventory |
| `natural_scenes` | obstacles: scenes where the category is natural |
| `frame_front`, `frame_up`, `frame_level`, `frame_long`, `symmetric_180` | alignment targets: reviewed canonical frame (`+x` …), long axis, half‑turn symmetry (reviewer's call, world‑horizontal targets only; kept as is — compare with `symmetry`) |
| `symmetry`, `symmetry_near` | rotational symmetry of the model in its canonical frame (`runlogs/symmetry_20260915`, `scripts/symmetry/survey_symmetry.py`; `build_icl.py --symmetry`): comma‑separated generator tags — `x2 y2 z2` half turns about the frame axes, `x4 y4 z4` quarter turns, `x3 y3 z3 x5 y5 z5 x6 y6 z6` 3‑/5‑/6‑fold, `xinf yinf zinf` body of revolution about that axis, `full` sphere (a plain standing cylinder = `zinf x2`, a plain box = `x2 y2 z2`, a plain ball = `full`). `symmetry` = rotations under which geometry **and** texture are identical (mental‑rotation targets must not be sampled with such a rotation; every orientation‑evaluated target gets them as `orientation_equivalences`); `symmetry_near` = rotations that are geometrically near‑symmetric or texture‑borderline and await the review round (not equivalences yet). `none` = surveyed, nothing found; empty = not surveyed (scene rows, the block, the trays, articulated targets). Rows with a grasp‑tested `scale` carry the verdict at that scale |
| `joint`, `handle_key`, `handle_kind`, `handle_annotated`, `instance_scale` | articulated targets: joint(s) (`joint:status` list for scene objects), handle key and whether a canonical annotation exists, the scene instance's scale relative to the canonical model. The `(handle <role>)` requirement is checked against `require/handles.csv`; the joint actually used is a customs list value (`roles.<role>.joint`) |
| `region_id`, `p0`, `p1`, `normal`, `hang_z`, `length_m`, `max_object_m` | wall regions (`<room>/<wall>/<rNN>`, world xy at hang height, normal into the room) |
| `top_z`, `world_ext_x/y`, `pos_x/y`, `yaw_deg` | supports / big partners: geometry from `runlogs/table_selection_20260911/inventory.csv` |
| `dataset` | non‑BEHAVIOR assets (`ispatialgym-assets` for the block) |
| `ready` | 1 = no unprocessed KB complaint |
| `note`, `source` | reviewer note (trimmed), provenance file |

Supports: the union of every scene object that was `spec`/`viable` for any task in the old sheet (31 objects in 6
scenes) is listed for every tabletop subtask; the floor of every scene is an admissible init support for the
large‑scale IORS subtasks. Wall regions: the 78 validated regions of the 2026‑09‑11 survey, for both wall subtasks.
Large partners: every bed / sofa / desk / breakfast table (+ countertop for ontop) of the inventory. Scene articulated
targets: every host object of the articulation swap tests, with `handle_annotated` and `instance_scale`. These scene
rows are seeds until A5 regenerates them from the scene inventory (the v1 pin `desk_xhjsub_0` as under(large‑scale)
support has no row yet — the builder reports every pin without a row).

### `constraints_compatible.csv` — which candidates go together

| relation | a → b |
|---|---|
| `partner_of` | target category (or model, attachment) → partner category / model; `attach_type` and `ready` for attachment; large‑scale IORS partners are categories (scene object chosen by `ground_isg`) |
| `distractor_same_category` | disembedding target category → same‑category distractor model |
| `distractor_other_category` | disembedding target category → other‑category distractor `category_b/model_b`, `tier` 1 = closest |

`ground_isg` only grounds category combinations that appear here (never a cartesian product of the candidate lists).

### `constraints_conflicting.csv` — which candidates must not appear together

`relation` (`lookalike` for now; A6 adds `exact_duplicate` / `same_geometry`), `group_id`, `category`, `model`,
`reason`, `source`. Members of one look‑alike group cannot be told apart from the robot's init view and are never
drawn as each other's distractors, never both into one distractor set, and never as a target–partner pair
(`docs/instance_sampling_spec.md` §3.12).

## `require/`

One row per computed pose, so `ground_isg` can filter models (and pick joints / poses) without opening the libraries.
All three sheets start with the same core columns; sheet‑specific columns follow.

| core column | meaning |
|---|---|
| `key` | artefact key (`<category>__<model>[__suffix]` for grasps, `<category>__<model>` for handles / contacts) |
| `category`, `model` | the object |
| `scene`, `instance` | empty for a canonical model / spawned object; the scene and scene object name for a per‑instance (scaled) annotation |
| `require` | the iddl `:require` form the row satisfies: `grasps:floating` / `grasps:tabletop` / `grasps:insert` / `grasps:attachment`, `handle`, `contacts` |
| `kind` | detail of the pose: grasps `<library pipeline>/<constraint>` (`graspgen/floating`, `a1/tabletop_insert`, `a1/flower_stem_top`, …); handles `bar` / `knob` / `edge_door` / `edge_lid` / `edge_drawer_top` / a reviewer's manual kind; contacts `push` / `brace` / `stop_edge` |
| `name` | grasp name; handle candidate `<joint>#A..D` (protruding handles as lettered on the review strips), `<joint>#F` (edge fallback), `<joint>#M` (reviewer‑placed pinch), `<joint>#-` (no candidate); contact name |
| `position`, `axis_approach`, `axis_close` | object `base_link` frame, metres: the eef origin (grasps) or the contact point (handles, contacts); a = approach = eef +z; c = closing / pinch axis = eef +y |
| `width_m` | object width at the pinch (grasps), handle / leaf thickness (handles) |
| `verdict` | grasps: `ok` / `flag` / `unsure` (`none` = key without a clean grasp); handles: `handle` (bar / knob), `edge` (exposed edge fallback), `edge_blocked` (edge hidden by the base or a neighbour), `manual` (reviewer pinch), `manual_unscaled` (reviewer pinch of the canonical model on a non‑unit‑scale instance — rescale before use), `skip` (reviewer: infeasible / asset issue), `ungraspable`; contacts: B6 verdict |
| `flags`, `note` | reviewer flags (`one_finger` is tolerated) and notes |
| `path` | repo‑relative library file the row summarises (also reachable through `lib/`) |
| `source` | library / run directory |

**Admissibility for `ground_isg`** (a model without an admissible row is blacklisted with the `missing_*` reason):

- `(grasps <role> <pipeline>)` — ≥ 1 row with `require = grasps:<pipeline>` and `verdict ∈ {ok, flag}` for the model.
- `(handle <role>)` — ≥ 1 row with `verdict ∈ {handle, edge, manual}` (`review_graspable ≠ False`) for the model (spawned
  targets) or for the (scene, instance) (scene targets); the joint of that row is a candidate for `roles.<role>.joint`.
  Instance rows are the B4 geometric proposals until the C3 review downgrades them; `width_m` must fall in the pinch
  range (15–80 mm) after scaling.
- `(contacts <role>)` — ≥ 1 `push` row for the model (plus the brace / stop rows the task uses).

### `grasps.csv` — every validated grasp of every referenced key

Extras: `quat_xyzw` (eef orientation in the object frame), `finger_q1/2` (finger joint positions at closure),
`pad_clearance_m` = (0.10 − width)/2 (how far each open pad sticks out beyond the object; disembedding distractor
spacing), `com_perp_m`, `antipodal`, `coverage`, `tilt_deg`, `rest_quat_wxyz` (rest pose the object was grasped in),
`scale`, `task`, `pipeline` (the iddl pipeline). Library: `lib/grasps` → `runlogs/grasp_library_20260905`
(`cands/<key>.yaml` candidates, `results/<key>.json` outcomes = `path`, `review/` verdicts). Key suffixes: none / `__f`
floating, `__t` tabletop, `__i` insert, `__fl` flower stem (insert), `__a` attachment, `__s` / `__sf` scaled variants.

### `handles.csv` — handle / edge pinches of articulated objects

Canonical annotations (`lib/handles` → `runlogs/grasp_library_20260905/artic/handles/<key>.json`, 100 models) with
the reviewer verdicts of `artic/artic_results.csv` (`review_graspable`, `review_pinch`, `chosen` = the reviewer's
marker) and `artic/manual_handles.json` (manual pinches `#M`, `use`, `skip`); plus the per‑instance scaled annotations
of the 137 scene articulated objects (`lib/scene_inventory` → `runlogs/scene_inventory_20260915/<scene>/artic/<name>.json`,
B4) with `instance_scale`, `position_world`, `height_world_m`, `front_free_m` / `front_free_eroded_m`. Extras:
`joint`, `joint_type`, `joint_kind` (door / drawer / flap), `status` (the joint's geometric status), `chosen`,
`exposed`, `length_m`, `depth_m`, `pull_dir`, `axis_dir`, `hinge_point`, `lower`, `upper`, `closed_q`.

### `contacts.csv` — push / brace / stop contacts of the alignment targets (B6)

Header‑only until `runlogs/contacts_<date>/<category>__<model>.json` exist; the builder then links `lib/contacts` and
reads `{"key", "category", "model", "scale", "contacts": [{"name", "role": push | brace | stop_edge, "position",
"approach", "closing", "direction", "shape_class", "verdict", "flags", "note"}]}` (object `base_link` frame). Extras:
`contact_role`, `direction` (push tangent / brace direction), `shape_class`, `scale`.
