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

Every sampleable / group candidate row starts with `status = untested`: the old sampler's "viable" verdicts are kept as
provenance (`source_selection`) only; `ground_isg` writes the real per‑scene status into the task customs list
(whitelists / blacklists / constraints). The scene rows regenerated from the B4 scene inventory (`build_icl.py
--scene-inventory`, A5‑lite) carry the geometric verdict instead: `geometric_ok` or `geometric_reject:<reason>` (rooms
and wall regions stay `untested`, no B4 data for them). What the icl does **not** hold: how many members a group has,
which model is drawn when, spawn scales, chosen joints — those are task customs list parameters (`generation`,
`roles.<role>.*`).

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
| `status` | sampleable / group rows, rooms, wall regions: `untested` (see above). Scene rows from the B4 inventory: `geometric_ok` = admitted by the geometric filters below, `geometric_reject:<reason>` = rejected but kept so nothing is silently lost (`representative` / `test` empty; every failing check is listed in `note` as `rejects=…`). Reasons — supports: `support_low` / `support_high` (top z outside 0.55–1.20 m), `support_small` (top polygon < 0.12 m² or hull short side < 0.30 m), `no_free_edge` (no edge with a ≥ 0.60 m free‑floor run at stand‑off 0.45, chairs within 1 m removed), `chair`; large‑scale supports also `no_near_partner` (no admitted partner of the subtask within 3.0 m geodesic); partners: `no_face_raster`, `under_clearance` (under: maximum under‑clearance < 0.30 m); scene articulated targets: `joint_kind_<kind>`, `ungraspable` (B4 status), `handle_low` / `handle_high` (pinch outside 0.30–1.70 m), `front_blocked` / `front_free_unknown` (eroded free floor in front of the face < 0.60 m), `pinch_narrow` / `pinch_wide` (width outside 15–80 mm after the instance scale) |
| `grasp_key` | preferred grasp‑library key when the role is grasped; the `(grasps <role> <pipeline>)` requirement is checked against `require/grasps.csv`, not this column; empty = no grasps yet (touching round‑1 targets are flagged in `note`) |
| `scale` | per‑axis scale of the grasp‑tested variant (scaled categories such as `garlic_clove`, `pencil`, `french_fries`) |
| `bbox_x/y/z` | canonical bounding box (m) from the KB inventory |
| `natural_scenes` | obstacles: scenes where the category is natural |
| `frame_front`, `frame_up`, `frame_level`, `frame_long`, `symmetric_180` | alignment targets: reviewed canonical frame (`+x` …), long axis, half‑turn symmetry about the alignment axis (wall normal = front axis for the wall subtasks, up axis for axial): `1` = symmetric, empty = the selection reviewer did not flag it (world‑horizontal targets only), `0` = the symmetry review round judged the half turn broken. The survey's disagreements are listed in `runlogs/symmetry_20260915/report.md`; the value is only overwritten where the review round answered that (subtask, model) explicitly (`preserved` → `1`, `broken` → `0`) |
| `symmetry`, `symmetry_near` | rotational symmetry of the model in its canonical frame (`runlogs/symmetry_20260915`, `scripts/symmetry/survey_symmetry.py`; `build_icl.py --symmetry`): comma‑separated generator tags — `x2 y2 z2` half turns about the frame axes, `x4 y4 z4` quarter turns, `x3 y3 z3 x5 y5 z5 x6 y6 z6` 3‑/5‑/6‑fold, `xinf yinf zinf` body of revolution about that axis, `full` sphere (a plain standing cylinder = `zinf x2`, a plain box = `x2 y2 z2`, a plain ball = `full`). `symmetry` = rotations under which geometry **and** texture are identical (mental‑rotation targets must not be sampled with such a rotation; every orientation‑evaluated target gets them as `orientation_equivalences`); `symmetry_near` = rotations that are geometrically near‑symmetric or texture‑borderline and await the review round (not equivalences yet). `none` = surveyed, nothing found; empty = not surveyed (scene rows, the block, the trays, articulated targets). Rows with a grasp‑tested `scale` carry the verdict at that scale. Review round: `runlogs/symmetry_20260915/review_contract.md` + `assign/rev_XX.json` → reviewer files `review/rev_XX.json` → `scripts/symmetry/aggregate_review.py` → `review_verdicts.json`; when that file sits next to `symmetry.json` the builder uses the reviewed values (`preserved` tags joined `symmetry`, `broken` tags left `symmetry_near`, `unsure` stays in `symmetry_near`) for the reviewed models |
| `joint`, `handle_key`, `handle_kind`, `handle_annotated`, `instance_scale` | articulated targets: the joint (small‑scale: the reviewed joint; scene targets: one row per (scene object, joint) of the B4 inventory), handle key, the pinch kind the row was judged on (`bar` / `knob`, or the exposed edge fallback `edge_door` / `edge_drawer_top` / `edge_lid` when the protruding handles fail the pinch filters — flagged in `note`), whether a canonical annotation exists, the scene instance's per‑axis scale relative to the canonical model (B4 measurement). `note` carries the B4 pinch (`h=` world height, `w=` width in mm), `front_free=<raw>(<eroded>)` and the prior canonical review. The `(handle <role>)` requirement is checked against `require/handles.csv`; the joint actually used is a customs list value (`roles.<role>.joint`) |
| `region_id`, `p0`, `p1`, `normal`, `hang_z`, `length_m`, `max_object_m` | wall regions (`<room>/<wall>/<rNN>`, world xy at hang height, normal into the room) |
| `top_z`, `world_ext_x/y`, `pos_x/y`, `yaw_deg` | supports / big partners: top (face) raster height, world AABB extents and centre, yaw of the top raster frame — from the B4 support / partner JSON (`runlogs/table_selection_20260911/inventory.csv` when the builder runs without `--scene-inventory`). Support `note`: `B4: top z, area, short side; edges=<edge>(run at 0.35/0.45/0.55)…` = the admitted edges whose free run holds at all three stand‑offs, `edges_0.45_only=` the ones admitted at 0.45 only, `best_run=`, `chairs<=1m=`, `occupants=`; large‑scale supports add `near_partners=<partner>(geodesic m)`. Partner `note`: face cells, `best_run_0.45=`, `under_clearance_max=` (under), `supports<=3m=` |
| `dataset` | non‑BEHAVIOR assets (`ispatialgym-assets` for the block) |
| `ready` | 1 = no unprocessed KB complaint |
| `note`, `source` | reviewer note (trimmed), provenance file |

Mental rotation (spec §3.13 rule 1, decision 2026‑09‑15): the builder drops every `mental_rotation` / `target` row whose
`symmetry` is a confirmed symmetry (`≠ none`; empty and `symmetry_near`‑only rows stay until a review verdict confirms the
tag, then the drop applies on the next build) and hands a dropped `representative` / `test` mark to the next‑ranked
remaining model of the category (`rank`, then row order; the test set is refilled to its previous size; the promoted row's
`note` starts with `promoted: replaces <model> (<tags>)`; a category left empty is only reported). The dropped rows are
kept in `runlogs/symmetry_20260915/dropped_mental_rotation.csv` (candidates columns + `reason`); the models stay
candidates for every other role.

Scene rows (A5‑lite, `build_icl.py --scene-inventory runlogs/scene_inventory_20260915`, on by default when the
directory exists; `--scene-inventory none` falls back to the hand‑copied seeds of A2): regenerated from the B4 scene
inventory (`summary.json` + `<scene>/{supports,partners,artic}/<name>.json`) with the geometric filters of
`docs/instance_sampling_spec.md` §3.2 / §5.6 / §6.12, one `geometric_ok` or `geometric_reject:<reason>` row each:

- **Supports** — every B4 support candidate with a top raster (158 objects; the 16 without free cells are skipped and
  reported) is listed for every tabletop subtask (disembedding, block design, articulation small‑scale, attachment,
  alignment‑axial, IORS ontop / under small‑scale, inside, nextto, touching, and the two large‑scale IORS subtasks):
  top z 0.55–1.20 m, top polygon ≥ 0.12 m² with a hull short side ≥ 0.30 m, ≥ 1 edge whose free‑floor run at
  stand‑off 0.45 m (chairs within 1 m removed) is ≥ 0.60 m; edges holding at all three stand‑offs 0.35 / 0.45 / 0.55
  are the preferred ones (`edges=` in `note`). Chairs are never supports; categories never used before (ottoman,
  bench, shelf, nightstand, cubicle, bar, …) are candidates when they pass. For the large‑scale IORS subtasks a support
  must in addition lie within 3.0 m geodesic (eroded map, B4 partner records) of an admitted partner of that subtask
  (`near_partners=`); the floor of every scene stays an admissible init support there (`untested`).
- **Large partners** — ontop: every bed / sofa / desk / breakfast table / countertop of the B4 partner inventory with
  a face raster; under: bed / sofa / desk / breakfast table with a maximum under‑clearance ≥ 0.30 m
  (`under_clearance_max=` in `note`, the others `geometric_reject:under_clearance`). B4 only inventories fixed‑base
  partners, so the movable breakfast tables / sofas of the old table‑selection inventory have no partner row.
- **Scene articulated targets** — one row per (scene object, joint) of the B4 artic inventory: joint kind door /
  drawer / flap, B4 handle status `handle` or `edge`, a pinch (protruding handle, else the exposed edge fallback) at
  0.30–1.70 m world height, ≥ 0.60 m eroded free floor in front of the face, width 15–80 mm after the instance scale.
- **Rooms** and **wall regions** (the 78 validated regions of the 2026‑09‑11 survey, both wall subtasks) are unchanged
  and `untested`.

The v1 pins are kept where the object survives the filter (test support `breakfast_table_rhjoby_0`, under partner
`bed_rrcvaq_0`, under support `desk_xhjsub_0`); a pin whose row is rejected is reported by the builder and left
unmarked (`breakfast_table_rhjoby_0` as under(large‑scale) support: no admitted under partner within 3 m).

### `representatives.csv`, `candidates_extra.csv`, `representatives_report.md` — the representative selection (A9)

Written by `scripts/icl/select_representatives.py` (deterministic; re-run after every `build_icl.py`). One row per
(pick, role, slot): `family, subtask, pick` (`<subtask>#NN`), `role`, `slot` — `representative` (the model
`ground_isg` / `sample_isg` / the oracle run first; training), `variation_texture` (V‑OBJ: same geometry, other
texture), `variation_geometry` (VB‑MOBJ / VB‑ISC: other geometry of the same category; scene roles: another support /
partner / scene instance), `heldout` (generalization; **global per category** — the same model in every subtask, never
drawn in any other role) — then `category, model, scene, name` (scene object), `joint`, `kind`, `grasp_key` (the
pipeline key of the subtask), `scale`, `geometry_group` / `exact_group` (texture‑variant / duplicate ids from the
reviews), `status`, `note`. `candidates_extra.csv` = candidate rows the icl did not have (`candidates.csv` columns +
`plan`; `status = incomplete:<reason>`), merged by `ground_isg` until a survey validates them. The report lists per
subtask which variation / generalization axes are unavailable and why, the slot‑availability table and the plan for
the incomplete candidates. `ispatialgym/sampling/ground_isg.py` (v1 bridge) grounds the picks into
`ispatialgym/assets/task_specs/isg_reps/`.

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
