# iSpatialGym instance sampling and initialization rules (v2)

Status: authoritative rule set for the sampling rebuild (`ground_isg` → `sample_isg` → `multiply_isg`), 2026-09-15.
Supersedes `instance_sampling_spec_v1.md` (sampler 1.x; kept for measured facts). Companions: `task_bddl_formats.md`
(what an instance evaluates), `task_definition_and_evaluation.md` (artefacts, customs list, deterministic loading),
`oracle_primitives_design.md` (the demo oracles the rules must stay consistent with), `ispatialgym/assets/iddl/README.md`
(iddl grammar), `TODO.md` (who is doing what).

Numbers marked `[pilot]` are the design values; the pilot (TODO E1/E2) verifies or replaces them.

## 0. Pipeline and principles

```
Behavior-1K:  sample_b1k   : bddl + task customs list ─sample→   1 template
              multiply_b1k : 1 template               ─multiply→ N instances

iSpatialGym:  ground_isg   : iddl + icl               ─ground→   bddl per category combination + task customs list
                                                                 (+ template & instance 0 of the representative models)
              sample_isg   : bddl + task customs list ─sample→   N templates (model combinations), each with instance 0
              multiply_isg : 1 template               ─multiply→ N instances
              run_isg      : instances                ─oracle→   demos (hdf5 + LeRobot v3), QA, evaluation
```

- **iddl** (`ispatialgym/assets/iddl/`): one file per subtask, BDDL over roles, no synsets (see its README).
- **icl** (`ispatialgym/assets/icl/`): candidate categories and models per role, how roles correlate (partner for
  this target, distractor pools for this target, female for this male), which combinations are *test* combinations,
  supports and wall regions per scene, per-model frame axes / scale / grasp keys.
- **task customs list** (`task_custom_lists.json`, section 2.3): per alias, per-scene whitelists and blacklists of
  models, bindings of the scene roles, generation parameters (group sizes and draw mode, scales, joints, counts,
  tolerances) and the *constraints* every successful sampling run adds.

Principles, in the order they are traded off:

1. **Correctness of the saved instance**: `:init` true, binary goal false, robot and every object settled and
   whole-body collision-free at init and at the goal, goal physically feasible without the robot, every TRO target
   consistent with the BDDL.
2. **NO FALLBACKS.** A check that fails rejects the attempt (or the template); the only alternatives are the
   enumerated candidate sets written in this document (edges, stand-offs, trunks, feasible grasp sets, brace kinds,
   partner candidates). Nothing silently substitutes another arm pose, another grasp pipeline, another support or
   another relation. `view_arm_fallbacks`, the `ready` arm fallback, `init_depth_fallbacks`, the bed-top init
   fallback and the base-yaw fallback of v1 are gone.
3. **Geometry first, one verification.** Placements come from support geometry, the camera model and cached
   reachability sets (reach volume, reach bands); random draws choose among pre-filtered candidates; each draw is
   confirmed once with the exact IK.
4. **Sampling must guarantee oracle yield.** Every gripper pose the scripted policy will need (grasp, pre-grasp,
   place, push and brace contacts, articulation waypoints, in-hand reorientations, base-follow poses) is IK- and
   collision-checked during sampling and recorded for the oracle (section 7). Whatever the oracle does, the sampler
   has checked — by construction, not by two code bases agreeing: the oracle's program for the instance
   (`expand(alias, sidecar)`, `oracle_primitives_design.md` §3.7) is expanded from the completed draft sidecar at
   the end of sampling and every pose and waypoint of every action is checked once more (the program check, 3.5),
   and sampler and oracle derive those poses from one conventions module (oracle doc §3.5).
5. **Identical robot configuration across instances** except where a rule samples it (base pose, trunk when the
   rule says "continuous", the grasp draw).
6. **Every instance makes sampling cheaper**: constraints learned from successful instances are written back into
   the customs list (section 9) and read as priors by later runs.

## 1. Artefacts and layout

```text
<out_root>/
  task_custom_lists.json                          the customs list used (copied; ground_isg / sample_isg append to the source)
  <alias>/                                        e.g. inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table
    activity_definitions/<activity>/problem<N>.bddl  the grounded BDDL (one per alias; N from problems.json)
    <template_id>/                                e.g. house_single_floor__target-apple-omzprq__partner-plate-kkjiko
      template.json / template-partial_rooms.json
      template_meta.json                          bindings, rooms, cached geometry, spec hash, timings, stances
      instances/
        <instance_id>-tro_state.json              TRO (BEHAVIOR object states + robot_poses + ispatialgym block)
        <instance_id>.oracle.json                 oracle sidecar (section 7): stances, contacts, grasps, waypoints, plans
        <instance_id>.summary.json                what was sampled / measured (debug)
```

- `template_id = <scene>__<role>-<category>-<model>[__...]` for the spawned roles; scene roles are recorded in
  `template_meta.json` and in the alias.
- `instance_id = <alias>__<template_id>__<NNNN>`.
- The TRO keeps BEHAVIOR's layout and the `isg-tro-v1` block exactly as in `task_bddl_formats.md`. Everything the
  oracle needs but the evaluator does not is in the **oracle sidecar**, a sibling file that carries the TRO's sha256;
  the public release may omit it.
- The activity directories are `camera_relocalization`, `object_scaling`, `mental_rotation`, `disembedding`,
  `configuration_matching`, `alignment`, `inter_object_relation_satisfaction`. Definition ids are allocated per alias
  in `activity_definitions/<activity>/problems.json` (alias ↔ id, contiguous), because the BEHAVIOR knowledge base
  enumerates `problemN.bddl`.

## 2. Inputs

### 2.1 iddl

See `ispatialgym/assets/iddl/README.md`. An iddl file is the BDDL skeleton over bare role names (`:alias`, optional
`:groups` composites, `:init`, `:goal`, `:measure`) plus a `:require` block naming the icl artefacts a role's models
must have — `(grasps <role> floating|tabletop|insert|attachment)`, `(contacts <role>)`, `(handle <role>)` — which
`ground_isg` applies as admissibility filters. It declares no roles, kinds, categories, counts or scales: role kinds
and candidates are the icl's (`roles/candidates.csv`), draw parameters the customs list's (`generation`), everything about
placement and installed constraints (hinges) is in sections 3–6 here. Roles in use: `target`, `targets` (group),
`partner`, `source`, `destination`, `distractors` (group with `same_category` / `other_category` pools), `obstacles`
(group, obstructed camera relocalization only), `support`, implicit `floor` and `agent`.

### 2.2 icl (candidate list)

`ispatialgym/assets/icl/` — categories, never synsets; schema in its `README.md`. Two folders: `roles/` matches
concrete categories and models to the iddl roles, `require/` holds the artefacts the `:require` block points at.

| sheet | one row per | key columns |
|---|---|---|
| `roles/candidates.csv` | (subtask, role, category, model) | `kind` — `sampleable` (one spawned instance), `group` (member candidate of a group role), `scene` (scene object / wall region / room bound per scene; supports, walls, scene articulated objects, big partners, `room` bounds); `representative` (the model of the category tested first by `ground_isg`; for scene rows the test support / partner / room / wall regions), `test` (test-set models incl. the v1 pins — explicit test combinations are the `test` rows admitted by `constraints_compatible.csv`, there is no combinations sheet), `natural_scenes` (where an obstacle category may be drawn), `grasp_key`, `scale` (grasp-tested variant), frame axes (`frame_front`, `frame_up`, `frame_level`, `frame_long`, `symmetric_180`), articulation (`joint`, `handle_key`, `handle_annotated`, `instance_scale`), wall-region / support geometry, `display_name`, `status` (`untested` until `ground_isg` writes the per-scene verdicts into the customs list), `note` |
| `roles/constraints_compatible.csv` | (subtask, relation, role_a, category_a[/model_a], role_b, category_b[/model_b]) | which candidates go together: target→partner (`partner_of`, attachment `attach_type`, `ready`), target→distractor pool (`distractor_same_category` / `distractor_other_category`, tier) |
| `roles/constraints_conflicting.csv` | (relation, group, category, model) | which candidates must not appear together / as each other: `lookalike` groups (section 3.12; A6 adds duplicates / same-geometry) |
| `require/grasps.csv` | (grasp key, grasp name) | shared core (`key`, `category`, `model`, `scene`, `instance`, `require = grasps:<pipeline>`, `kind`, `name`, `position`, `axis_approach`, `axis_close`, `width_m`, `verdict`, `flags`, `note`, `path`, `source`) + `quat_xyzw`, `finger_q1/2`, `pad_clearance_m = (0.10 − width)/2`, `com_perp_m`, `antipodal`, `coverage`, `tilt_deg`, `rest_quat_wxyz`, `scale`, `task`, `pipeline` — what `(grasps …)` is checked against |
| `require/handles.csv` | (key, joint, handle candidate) | shared core (`require = handle`; `name` `<joint>#A..D` / `#F` edge fallback / `#M` reviewer pinch; `verdict` `handle` / `edge` / `edge_blocked` / `manual` / `manual_unscaled` / `skip` / `ungraspable`) + `joint`, `joint_type`, `joint_kind`, `status`, `chosen`, `exposed`, `length_m`, `depth_m`, `pull_dir`, `axis_dir`, `hinge_point`, joint limits, and for the B4 scene instances `instance_scale`, `position_world`, `height_world_m`, `front_free_m` / `front_free_eroded_m` — what `(handle …)` is checked against |
| `require/contacts.csv` | (key, contact) | shared core (`require = contacts`; `kind` push / brace / stop_edge) + `contact_role`, `direction`, `shape_class`, `scale` — what `(contacts …)` is checked against; header-only until B6 writes `runlogs/contacts_<date>/` |

`require/lib/` holds relative symlinks to the libraries the `path` columns refer to (`grasps` → the grasp library,
`handles` → its `artic/handles`, `scene_inventory` → the B4 inventory, `contacts` once B6 exists). Everything that
used to be a "viable" list (supports, walls, articulated objects) is a *candidate* here until `ground_isg` re-derives
it.

**Row admissibility that `ground_isg` enforces before anything is spawned** (each failure is a blacklist entry with
the reason in parentheses; the selection runlogs applied these by hand, here they are checks):

- `ready = 1` — no unprocessed BEHAVIOR-KB complaint, asset on disk, single rigid link for rigid roles
  (`not_ready`). Scene rows are exempt (they are what the scene contains).
- A manipulated role (grasped or pushed) needs a `graspable` model: ≥ 1 admissible grasp / handle / contact row for
  the pipeline the iddl asks for (`missing_grasps:<pipeline>`, `missing_handle`, `missing_contacts`). The
  zero-grasp categories of the grasp library (cap without knob, lid, license_plate, dartboard, briefcase,
  french_fries, cinnamon_stick, …) therefore only appear as partner, distractor or obstacle.
- **Grasp pipeline per subtask** (the iddl `(grasps <role> <pipeline>)` form; no fallback to another pipeline):
  `floating` (`__f`) for object scaling and mental rotation; `tabletop` (`__t`) for disembedding and IORS ontop /
  under / nextto / touching; `insert` (`__i`, flowers `__fl`) for IORS inside; `attachment` (`__a`) for the
  attachment male. `handle_half` (`__h`) rows count as `floating` for knives. A model whose only rows belong to
  another pipeline is `missing_grasps:<pipeline>`.
- **Reviewer verdicts**: a grasp row is admissible when `verdict = ok`, or `flag` with flags ⊆ {`one_finger`};
  `end_grasp`, `gap`, `fragile`, `penetration`, `task_violation` exclude the row. `one_finger` is an artefact of
  assisted grasping (the fingers are position-controlled independently: the first pad to touch pushes the object into
  the other pad and keeps closing), it happens in demos too and is not a grasp failure.
- **Spawn scale = grasp-tested scale**: a row with a non-empty `scale` (the thin objects re-run at a per-axis scale
  that makes their thinnest extent 20 mm: pencil, garlic_clove, french_fries, cinnamon_stick, two caps) is spawned at
  exactly that scale (`roles.<role>.scale` in the customs list, written by `ground_isg`), never at the canonical
  scale, and its rest pose, symmetry (3.13) and every per-model cache (3.10) are keyed by the scale vector.
- Alignment targets need a reviewed frame record (`frame_front` / `frame_up` or `frame_level`, `frame_long`) —
  otherwise `missing_frame`; wall targets additionally need `symmetric_180` filled (0 or 1).

### 2.3 Task customs list

One entry per alias. Scene-independent model pools are the icl's job; the customs list holds **how models are drawn**
(`generation`: group sizes and draw mode, spawn scales, chosen joints, counts, tolerances), what a scene changes
(`scenes`: bindings of the scene roles, per-scene whitelists / blacklists) and what sampling learned (`constraints`).

```json
{
  "inter_object_relation_satisfaction-nextto-spatula-soup_ladle-breakfast_table": {
    "iddl": "inter_object_relation_satisfaction/nextto.iddl",
    "definition_id": 3,
    "roles": {
      "target":  {"category": "spatula",    "sampleable": true,  "require": {"grasps": "tabletop"}, "representative": "hkvsep"},
      "partner": {"category": "soup_ladle", "sampleable": true,  "representative": "cfzneb"},
      "support": {"category": "breakfast_table", "sampleable": false},
      "floor":   {"sampleable": false}
    },
    "scenes": {
      "house_single_floor": {
        "rooms": ["dining_room_0"],
        "bindings": {"support": ["breakfast_table_rhjoby_0"], "floor": {"room": "dining_room_0"}},
        "whitelist": {"target": {"spatula": ["hkvsep", "ktkigm", "ymvrct"]}, "partner": {"soup_ladle": ["cfzneb", "hgvzdm"]}},
        "blacklist": {"target": {"spatula": {"pdqgzq": "ground_isg 2026-09-20: grasp_unreachable at every focus cell",
                                             "qwbmxs": "ground_isg 2026-09-20: missing_grasps:tabletop"}}}
      },
      "restaurant_diner": {"...": "one block per candidate scene; unbounded scenes list all seven"}
    },
    "generation": {
      "templates_max": 20, "instances_per_template": 5,
      "tolerances": {"position": 0.03, "orientation": 0.20},
      "groups": {},
      "path_buckets": null
    },
    "constraints": [
      {"scope": {"scene": "house_single_floor", "support": "breakfast_table_rhjoby_0"},
       "type": "stance", "edges": ["e2"], "stand_off": [0.45, 0.55], "lateral": [-0.30, 0.60],
       "trunk": {"translate": [0.5, 0.75], "tilt": [-0.6, -0.3]}, "hard": false,
       "source": {"instance": "...__0003", "sampler": "2.0.0"}},
      {"scope": {"target": "spatula/hkvsep", "support_height": 0.80},
       "type": "grasp_set", "trunk": [0.5, -0.3], "cell": [0.62, -0.18], "feasible": [[3, "grasp_1"], [4, "grasp_1"], [3, "grasp_4"]],
       "hard": false, "source": {"instance": "...__0003"}}
    ]
  }
}
```

- `roles.<role>`: the grounded category, `sampleable` (`true` = spawned from the whitelist per template; `false` =
  a scene role bound per scene under `scenes.<scene>.bindings`), the copied `require` entries of the iddl
  (`grasps: <pipeline>`, `contacts: true`, `handle: true` — the loader and oracle read the grasp pipeline from here),
  the representative model, and the per-role draw parameters that are not group-related: `scale` (per-model spawn
  scale map: the configured scale of the alignment targets, `{"keyboard/hgtbot": 0.8}`, and the grasp-tested scale
  of the thin models copied from the icl, `{"pencil/xyz": [1.0, 3.6, 5.1]}`; absent = 1), `joint` / `handle_key`
  (the one joint of an articulation target, per model or per scene object name), and `dataset` for models that are
  not BEHAVIOR assets (`"dataset": "ispatialgym-assets"` for the block and any custom tray; the synset must be
  registered in the ispatialgym BDDL domain).
- `whitelist` / `blacklist` are **per scene**; a scene block exists for every candidate scene (`house_single_floor`,
  `house_double_floor_upper`, `house_double_floor_lower`, `Rs_int`, `hotel_suite_large`, `restaurant_diner`,
  `office_cubicles_right`) unless the icl bounds the scene (`room` rows: camera relocalization and the held-object
  tasks are `house_single_floor` / `living_room_0` only for now). Whitelists start as the full icl candidate pool per
  category minus the models rejected by the iddl `:require` filters (blacklisted as `missing_grasps:<pipeline>`,
  `missing_contacts`, `missing_handle`) and minus models removed by a failed representative test (blacklisted with
  the reason); scene-dependent objects (supports, walls, scene articulated objects, big partners) are listed per
  scene under `bindings`.
- `generation.groups.<role>` for every group role: `size` (an integer or `[min, max]`, drawn per template) and
  `draw` — `unique` (pairwise distinct models, look-alike exclusions of section 3.12 on top; the default) or
  `same_model` (all members one model). Current values: `targets` `{size: 4, draw: same_model}` (block design),
  `distractors` `{size: [1, 3], draw: unique}`, `obstacles` `{size: 1, draw: unique}`. The size is part of the alias
  (`block_4`, `bottle_of_beer_3_…`), so changing it grounds a new problem instead of re-grounding the old one.
- **Seeding.** `ground_isg` creates the entry: `generation` (and the `roles.<role>.scale` / `joint` values) is
  copied from the existing entry when the alias was grounded before, otherwise from the per-subtask defaults in
  `ispatialgym/sampling/generation_defaults.py`, whose values are the numbers written in this document (4 blocks, 1–3
  distractors, one obstacle, tolerances of section 8, path buckets below, `require_goal_handle: auto`, `arm: auto`).
  There are no defaults hidden in the samplers.
- `generation.path_buckets` (camera relocalization only): `[{"max_m": 3, "n": 10}, {"max_m": 5, "n": 10},
  {"max_m": 10, "n": 5}]`; buckets are half-open ranges of the geodesic path length `(0, 3]`, `(3, 5]`, `(5, 10]`;
  `multiply_isg --path-buckets 3:10,5:10,10:5` overrides; the bucket is recorded in the sidecar and used in the
  instance-level language.
- **Constraints** are typed, scoped records that *narrow* a sampling distribution (priors), never replace a check.
  Types: `stance` (edges, stand-off, lateral, trunk ranges), `grasp_set` (feasible `(yaw, grasp)` per trunk and
  cell), `joint_set` (stable set S, `passive_from`), `placement` (a BDDL-like literal with numeric ranges:
  `(ontop target support :depth [0.10, 0.40] :lateral [-0.30, 0.30])`), `scale`, `band_ref` (reach-band cache key),
  `contacts` (verified push / brace contacts per model), `blacklist` (with reason). Every record has `scope`
  (any of `scene`, `support`, `partner`, `target`, `model`, `support_height`, `trunk`), `hard` (a hard record is a
  rejection rule, e.g. a blacklist; soft records are priors) and `source` (instance id, sampler version, date).
  `sample_isg` reads them before drawing; `sample_isg` and `multiply_isg` append what each successful instance
  confirmed.

## 3. Shared rules (every task)

### 3.1 Scenes and scene variants

Candidates: the seven 2026-challenge scenes above. Task rooms = the rooms of the bound fixed objects (+ `rooms`);
`load_room_instances` adds the neighbours (segmentation footprint within 0.6 m) as in v1. **The sampler loads the
same room set** (task rooms + neighbours, `SimWorld.load_adjacent` default on, `ISG_LOAD_ADJACENT=0` opts out), so
sampling, evaluation and the oracle share one collision and visual world: a stance or path validated in sampling
cannot meet furniture in the demo that did not exist while sampling, and segmentation renders see the neighbours'
objects through doorways and glass instead of sky. v1 loaded the task rooms only because of the renderer crash
described in 3.4; that guard no longer depends on the room set. Table tasks sample in the
stable scene minus the chairs within 1.0 m of the support (`scene_variants`, `freed_footprints` recorded and applied
by evaluation and by the oracle's navigation map). The obstructed camera-relocalization variant records the obstacle
footprint as `blocked_footprints` in the same structure.

### 3.2 Support and placement

- An object placed on a surface has its **whole footprint** (AABB projected after the sampled yaw) inside the support
  polygon (top face of the collision geometry, not the AABB) shrunk by 2 cm (0.5 cm on a tray floor).
- Rest orientation = canonical up, or the grasp library's rest quaternion for models that lie on a side
  (`require/grasps.csv rest_quat_wxyz` of the drawn grasp row — the grasps were validated in that pose, so a model
  whose rows carry a rest quaternion is always spawned in it), times the sampled yaw; z = support top + bottom
  offset + 3 mm. Objects with a non-empty icl `scale` are spawned at that scale (2.2).
- Footprints keep ≥ 3 cm clearance (target–partner) unless a rule states otherwise; nothing is placed inside the
  robot's or another object's AABB.
- **Support admissibility** (re-derived by `ground_isg` from the B4 inventory, never trusted from the icl): a single
  flat top link, top height 0.55–1.20 m (`support_low` / `support_high`), top polygon ≥ 0.30 m on its short side and
  ≥ 0.12 m² (`support_small`), at least one edge with a free-floor run for the stances of section 4 (`no_free_edge`),
  in a room the robot can stand in. Chairs are never supports; a cubicle desk is one object with several tops, each
  top is a separate support entry. Small-scale articulation targets taller than 0.6 m use the floor (5.4).

### 3.3 Robot

- R1Pro, JoyLo controller config (`r1pro_model.joylo_robot_config`), grasping mode assisted, head camera
  `robot_r1:zed_link:Camera:0`, 1080 × 1080, hfov = vfov = 99.2°.
- **Untucked is the only arm posture** (`untucked_default_joint_pos`, hands 0.64 m ahead of the base at 1.20 m at
  translate 0.5) — at init, for the idle arm at every manipulation stance and as the locked posture in every IK
  screen. Grippers open (0.05). Exceptions: camera relocalization (JoyLo reset pose: trunk 0.5, arms zero) and the
  held-object tasks (trunk and left arm at the JoyLo reset pose, right arm from the fixed hold, section 6.3).
- Trunk = JoyLo (translate t ∈ [0, 1] on the keyframe curve, tilt k on `torso_joint3`). Grid for caches: translate
  {0, 0.25, 0.5, 0.75, 0.95} × tilt {0, −0.3, −0.6, −0.9} (+ −1.2 when the support height h < 0.6 m). Translate 1.0
  self-collides in cuRobo for every arm posture (measured 2026-09-15, reach-volume run), so 0.95 is the deepest grid
  trunk everywhere (Phase 2 scans, Prompt 0/1 scans, camera-relocalization goal range clamp). Camera and
  hand offsets as functions of t (`x_c(t)`, `z_c(t)`, camera pitch, hand height below the camera) come from FK once
  per robot and are cached; the constants quoted in section 4 are their design values `[pilot]`.
- Base frame: x ahead, lateral + = robot's left, z above the floor. Face frame for large-scale tasks: n = outward
  unit normal of the manipulation face, P = reference point, d = distance of the base centre from the face plane
  along n, λ = lateral offset from P along the face tangent (+ = robot's left when facing the face).
- Whole-body collision freedom: cuRobo spheres against every collision mesh of the loaded rooms, 1 cm margin for
  non-floor links (2 cm where a rule says so).
- **Parking** (object settling, goal-feasibility checks, template resets): the robot is parked tucked at a free cell
  inside the task room's floor polygon, ≥ 1 m from every spawned object and support (a base parked over an unloaded
  room falls through; the v1 room-centre `PARK_POSITION` is replaced by this rule).

### 3.4 Camera model and visibility

Pinhole model of the head camera (fx = fy = 459 px, cx = cy = 540). "In frame (p %)" = every AABB corner (or hull
point where the rule says hull) of the object projects inside the image with a margin of p % of the image size.
Optical axis "within θ of a point" = angle between the optical axis and the ray to the point. Renders are used only
where a measurement needs them (visible-object counts, occlusion ratio, held-object self-occlusion, one confirmation
render of the disembedding destination width).

**Measurement renders (`seg_instance`) hide nothing.** v1 hid every `light` / `lamp` / `chandelier` object for the
whole run and loaded the task rooms only, because Isaac Sim 5.1 segfaulted on `seg_instance` renders with a light
fixture and glass in the head camera's view (and, with the neighbours loaded, on views of the kitchen or the garden of
`house_single_floor`). The trigger is the RTX flag `/rtx/raytracing/showLights` (OmniGibson turns it on at launch): it
draws the shapes of the light sources inside the fixture models, and the instance-segmentation reduction node
(`omni.replicator.core.InstanceSegmentationReduction`, the C++ node behind `seg_instance` and `seg_semantic`) cannot map
their instance ids — pixels around them come out `unlabelled` (a third of the image in some living-room views) and with
enough of them in view the node indexes out of bounds. `SimWorld` sets the flag to 0 for the sampler process
(`_set_measurement_render_settings`); nothing is hidden during physics, rgb observations, measurement renders or
`scene.save`, light fixtures and glass stay visible and are labelled normally, and the measured views are what
evaluation renders (only the glowing emitter shape is absent from the sampler's own rgb; illumination is unchanged).
Measured 2026-09-15 on `house_single_floor` with the neighbours loaded: 8 living-room views (sliding door + garden,
kitchen, dining room, corridor, high window) × 4–6 renders crash-free with zero unlabelled pixels; the v1 guard still
crashed on the kitchen view, the viewer-camera workaround of BEHAVIOR-1K #2312 (robot-camera render products lacking
the instance-mapping table) does not cover this crash, and translucency off does not either. Consequences for the
rules: light fixtures are labelled objects now, so the structural-category exclusion (`downlight`, `room_light`; 6.1)
is what keeps them out of visibility counts; objects of the neighbours seen through glass count as visible where a
rule counts non-structural objects. `ISG_HIDE_CATEGORY_SUBSTRINGS` (default empty) remains as an emergency override.

**Settled reads.** The annotator data lags the scene: a pose or visibility change reaches the renderer one frame
late, and the buffer a sensor read returns trails the last rendered frame by a further, GPU-load-dependent one to two
frames (measured 2026-09-15: change, three renders, read → the old view; read again without rendering → the same;
under load even render, read, render, read → the old view). v1's fixed "three renders then read" therefore reported
the state of the *previous* measurement whenever two measurements followed each other (e.g. the reference silhouette
with the distractors hidden came out equal to the occluded one, and the tray counts right after it carried the
hidden state). `SimWorld.seg_instance` now renders at least four frames and keeps rendering one frame at a time
until two consecutive fetches of the raw segmentation buffer are identical (the scene is static during a measurement,
so a settled buffer repeats exactly; cap eight frames), then reads the remapped observation. Cost ≈ 0.3 s per
measurement at 1080² (≈ 0.17 s before). Verified on a sampled disembedding instance: with the same robot and object
poses the target, source-tray and destination-tray pixel counts are identical with and without the neighbours
loaded (100596 / 59332 / 15447 px), and repeated measurements agree exactly.

### 3.5 Checks common to every task

At init **and** at the goal configuration: whole body collision-free, robot and all objects settled; objects on a
surface fully supported (3.2); when a gripper pose near the goal is known, its IK feasibility is checked (section 3.6);
then the validation steps of v1 §3.6, unchanged: settle (90 physics steps + 10), BEHAVIOR `validate_task` checks
(scene unchanged 5 cm / 0.05 rad, velocities ~0, `:init` true, stability over 1 + 10 steps), binary goal **false** at
init, goal feasibility by teleport + settle (120 steps) **with the robot parked** (tucked, room centre) and every
binary goal atom true, drift ≤ tolerance, TRO written and validated against the BDDL. Every attempt and every
template starts from the stable scene (`reset_scene_objects`, robot parked).

**Program check** (new in v2, TODO D2 / F6): once the draft sidecar is complete, the sampler expands the oracle's
program for the alias (`expand(alias, sidecar)`, `oracle_primitives_design.md` §3.7) and IK- and collision-checks
every pose and waypoint that every action's `targets()` returns — in the recorded posture, trunk and stance, with the
carried object attached where the action carries one (screening IK at waypoints, exact IK at end poses, 3.6). This is
the check principle 4 refers to. A pose that fails here is a bug in the rule that recorded it; the attempt is
rejected (`program_check_failed`, reported per action name), never patched. `validate_instances.py` (D5) runs the same
expansion on written instances. Trajectories *between* key poses are not checked here: the oracle plans them at run
time (oracle doc §3.9).

Details kept from v1 that the checks depend on:

- **Predicate cache**: OmniGibson caches kinematic predicates (`ontop`, `inside`, `touching`, `nextto`) per object
  pair and treats a sleeping object as unchanged without comparing its pose. Every truth table (`:init`, binary goal
  at init, goal feasibility) is evaluated only after the objects were woken or re-placed in the same step; a
  predicate read across a settle without a wake is a bug, not a check.
- **Room membership**: every spawned object is registered with the room it stands in (`in_rooms`), so partial-room
  loading and evaluation keep it; `(ontop agent floor)` is evaluated after 3 physics steps (the base needs contact)
  and holds when the base straddles two floor pieces of an open-plan room.
- **Reload validation** is a pipeline stage, not optional: every template and every instance is re-loaded in a fresh
  process from its TRO, settled 60 steps, and must reproduce the stored state — robot base within 2 mm / 0.02 rad,
  every object within 1 mm / 0.005 rad, joints within ½ tolerance, `:init` true and the binary goal false; a contact
  sheet (init view, goal view when the goal is materialised) is rendered per instance for the review rounds. Failing
  instances are deleted and counted (`reload_failed`), never patched.
- **Report**: `summary.json` per run keeps the v1 fields — the `:init` truth table, the goal truth table at init
  (all false) and at the goal check (all true), rejections per rule with counts and the first three examples, phase
  timings, kit / cache counters, and the seeds of every template and instance (`problems.<problem>.<template>.
  {n_written, time_s}` is the layout `bench_report.py` reads).

### 3.6 IK conventions

- Screening: 12-seed batched IK per pose; exact confirmation: 32-seed IK. Locked joints = the untucked init
  configuration at the stance's trunk; the idle arm is untucked and collision-checked.
- Pre-grasp / pre-contact = the pose moved 0.10 m back along −a (a = approach = eef +z). Handle grasps: eef origin =
  contact − 0.012·a. Push / brace contacts: pad centre on the contact point, eef origin = point − `CONTACT_CENTER_Z`·a.
- "Support ignored for the grasp pose only": the support (table, tray floor, wall, mattress) is removed from the
  collision world for the final grasp / contact pose and for nothing else; the target object is ignored for the
  final contact pose unless the rule says otherwise; every other object (partner, distractors, scene) stays.
- Every arm the oracle will use (right, left, both) has its cuRobo embodiment; the arm rule of the task decides.

### 3.7 Navigation feasibility (init stance → manipulation stance → placement stance)

Stances lie 0.35–1.1 m from a face, inside BEHAVIOR's eroded band, so the eroded navigation map cannot be used
end to end:

1. If the goal stance lies in the same connected region of the **locally un-eroded** map (the room's erosion mask
   minus the strip in front of the support / face) as the start stance, compute the shortest path on the un-eroded
   map and validate the whole path with whole-body collision checks at the **navigation posture** (untucked arms,
   carried object attached, trunk as recorded) at 0.1 m strides.
2. Otherwise plan on the eroded map between the nearest eroded-free cells of the two stances and connect each stance
   to the global path with a straight local segment, collision-checked the same way.

The path (waypoints, posture, length) is recorded; the oracle drives exactly this path. Path length limits: Prompt 0
step 6 (≤ 3 m), IORS large-scale B → C (≤ `max_partner_geodesic` + 1 m), camera relocalization buckets.

### 3.8 Stance transitions (swept trunk motion)

Wherever two consecutive recorded stances differ in trunk (init → manipulation, B → C, manipulation → placement),
the trunk change is executed at a **transition stand-off** `s_T`: find the smallest `s_T ∈ {s, s + 0.1, …, s + 0.5}`
(along −n from the arriving stance, `s` its stand-off) at which the swept trunk motion (5 samples between the two
trunk values, arms untucked, carried object attached) is whole-body collision-free; record
`transition = {stand_off: s_T, order: drive-back → trunk → drive-in}`; reject the candidate when none exists. This
replaces the oracle's "torso first or drive first" guess and its `ready_arms` fallback.

### 3.9 Waypoints and gripper visibility

**Gripper-in-FOV rule (b) — owner decision 2026‑09‑16 17:10.** The acting arm's fingertips must stay in the head camera's
frame (pinhole projection, 5 % margin) **continuously over every grasp → release span**: from the moment the pre‑grasp pose
is reached, through grasp, lift, the carry to the pre‑place, the descent and the release step; the retreat after a release
is exempt until the next pre‑grasp. Each grasp → release pair is its own span (disembedding's grasps, block design's
blocks, the flip‑on‑table re‑grasp); contact spans (brace / push) run from the pre‑contact to the end of the contact leg.
Consequence for the recorded motion: the **carry leg** from the lift end to the pre‑place is a straight Cartesian leg at
constant orientation (both ends are recorded in frame, so the projected path is in frame by construction); when that leg
is infeasible the sampler records a `carry` waypoint (`grasps[k].carry`, core `dc6a1a5`) — never a free trajopt swing. Implemented oracle‑side in `166005ff` (per‑arm spans opened at the pre‑grasp / pre‑contact, closed after the release / contact leg, checkpointed; `Place.targets["carry"]` straight leg through the recorded waypoint): the nextto regression instance goes from 114 consecutive out‑of‑frame steps at the place transit (verdict `gripper_out_of_fov`) to 0. B1K's replay QA
(`FieldOfViewMetric`, `gripper_changes_outside_fov_limit=0`) only requires the gripper in view at the grasp / release
toggles; ours is deliberately stricter (oracle doc §6.4).


- Motions the oracle executes as paths are checked at intermediate waypoints, not only at the endpoints: SLERP of
  in-hand rotations and the straight camera-ray motion of the held-object tasks (3 waypoints), push arcs of the
  alignment tasks (δ/2 and the goal), articulation trajectories (q_k), block flips (pick tilt → place tilt). Screening
  IK per waypoint, exact IK at the endpoints. `waypoint_policy` in the sidecar records which waypoints were checked.
- Every recorded contact / grasp pose is FK-projected into the init‑image or manipulation‑image frame; the eef at the
  final pose must be in frame (5 %) for the camera of the stance the oracle uses there (hard QA rule "gripper in FOV").

### 3.10 Caches

- **eef reach volume** (robot-only, section 5.1): once per arm; `ISPATIALGYM_REACH_VOLUME`.
- **Reach bands** (per support + object, section 4.1): two factors cached separately — the object-independent factor
  (trunk collision check, lattice, shoulder-distance prune; key = support name + top height + trunk grid + posture)
  and the object factor (grasp mask; key = model + grasp set + rest orientation + objects placed before it).
- **Feasible grasp sets F** per (trunk, cell) found in Phase 3 are stored in the band cache; a cell with empty F is
  removed from the band.
- Stance survivors of Prompt 0 steps 1–3 per (scene, target, init state).
- `ISPATIALGYM_SAMPLING_CACHE` for the exact-IK results of v1 stays.
- Every per-model cache key (grasp masks, held-object visibility, rest pose, symmetry) includes the spawn scale
  vector; a canonical-scale entry is never reused for a scaled variant (2.2).

### 3.11 What is recorded for the oracle

Section 7 lists the sidecar schema. Rule: anything the oracle would otherwise have to search for — stances, trunk
values, grasp `(yaw, grasp_key, grasp_name)`, pre-/grasp/place eef poses, push and brace contacts with directions,
handle waypoints `h(q_k)`, `follow_mode`, transitions, paths, block order and per-block grasp/flip data, path bucket,
per-object image-space descriptors for the language annotations — is written at sampling time.

Two things are deliberately **not** recorded (decision 2026-09-15, oracle doc §3.7 / §3.9): the action *sequence*
— it is the oracle's per-subtask program (oracle doc §6.2), which both `run_isg` and the program check of 3.5 expand
from these facts — and the joint-space trajectories between key poses, which the oracle plans at run time
(`plan_failed` is its own QA category).

### 3.12 Look-alikes, exact duplicates and texture variants

Three relations between models of one category, all recorded in the icl and all used by the samplers:

| relation | definition (source) | icl | rule |
|---|---|---|---|
| **exact duplicate** | identical geometry *and* textures under different model ids (`runlogs/grasp_library_20260905/select/duplicates.md` + `duplicates/`, `runlogs/{world_alignment_selection_20260911,world_horizontal_selection_20260912,obstacle_selection_20260912}/duplicates.json` `exact_groups`) | `roles/candidates.csv: duplicate_of` (the group's first model) | one member is the candidate; the others are never spawned, never counted as "another model" when `sample_isg` iterates over the whitelist, never both in a test set |
| **texture / colour variant** (same geometry) | identical geometry, different textures (`same_geometry` / `same_shape` in the same sources) | `roles/candidates.csv: geometry_group` | both are candidates (they look different); they count once for *geometry* coverage statistics, separately for model iteration; never two variants as the two test instances of a category (pair rule) |
| **look-alike** | models that cannot be told apart from the robot's **init view** (exact duplicates + same-shape variants whose only difference is a small print, label or shade, judged on renders at the init distance where objects are 20–130 px tall; `roles/constraints_conflicting.csv`, 55 groups today, disembedding categories only) | `roles/constraints_conflicting.csv` (`group_id`) | the task-semantics notion: no instance may ask for an object that another object in the scene could be confused with |

Rules that use them:

1. **Disembedding.** For a target `C/m` the same-category pool is `pools[C].same_category` minus `m` and minus every
   model in `m`'s look-alike group; the other-category pool likewise (groups may cross categories). The distractor set
   of a template contains **at most one model per look-alike group** and no exact duplicates: no distractor is a
   look-alike of the target, and no two distractors are look-alikes of each other (a look-alike group counts as one
   candidate when the set is drawn). An empty pool for the requested kind is not a fallback case: the alias fixes the
   distractor categories, so `ground_isg` only grounds category combinations whose pools are non-empty after the
   exclusions.
2. **Same-category or look-alike target–partner pairs** (IORS nextto / touching / ontop, e.g. salt shaker – pepper
   shaker, teacup – teapot): a pair whose two models share a look-alike group is rejected (the instruction would be
   ambiguous); a same-category pair of distinguishable models is allowed, and both objects get a spatial descriptor in
   the sidecar (`language.objects.*.descriptor`) so the annotations can name them.
3. **Groups of identical members** are the intended exception: blocks (`generation.groups.targets.draw =
   same_model`), and — if ever wanted — several distractors of one model; `distractors_group` members are unique
   models (`draw = unique`, the default).
4. **Model iteration** in `sample_isg` ("iterate over all models before reusing one") counts exact duplicates as one
   model and texture variants as distinct models.
5. **Which models are the representative and the test models of a category** (a constraint on `build_icl.py`, not a
   runtime rule). Every category has one *representative* model (the one `ground_isg` tests first) and two *test*
   models (the ones used for the benchmark's test instances, the reviewers' pick 1 and pick 2). After deduplication:
   the representative is the surviving member of pick 1's exact-duplicate group, and the two test models must be two
   different geometries — never an exact duplicate or a recolour of each other, otherwise the "two test models" would
   be one object with two labels (the selection contract's pair rule already asked reviewers to pick a different-looking
   second model; this makes it a check).

Status note (not a rule): the init-view look-alike review ("cannot be told apart from the robot's initial view") was
done only for the 39 disembedding categories, because that is where distractors are drawn. For the subtasks where two
objects of the same or a related category can appear together for other reasons — IORS pairs such as salt shaker /
pepper shaker, teacup / teapot, jar of spaghetti sauce / canned food — and for the obstacle and alignment categories,
only the geometry-based flags (exact duplicate, same geometry) exist today. Until TODO A6 delivers that review, rule 2
treats two models as look-alikes when one is the other's exact duplicate or texture variant (`duplicate_of` or the same
`geometry_group`); that is a stand-in, weaker than the reviewed judgement (two differently shaped shakers are not
caught, two recolours of one shaker are).

### 3.13 Rotational symmetry and `orientation_equivalences`

Source: the `symmetry` / `symmetry_near` columns of `roles/candidates.csv` (survey `runlogs/symmetry_20260915`,
TODO A8: geometry by Chamfer test and texture by rendering the object against its rotated copy under flat lighting,
evaluated at the grasp-tested `scale` when the row has one). Tags name the generators of the rotations under which
the object looks identical: `x2 y2 z2` (half turns about a canonical axis), `x4 z4 …` (quarter turns), `z3 z5 z6`,
`xinf yinf zinf` (bodies of revolution), `full` (spheres); `none` = surveyed and asymmetric; empty = not surveyed.
`symmetry_near` lists candidates that are geometrically or texturally borderline and not confirmed; such a tag
counts as asymmetric (a symmetry only takes effect once it is confirmed). The review round is closed (2026-09-15,
`runlogs/symmetry_20260915/review_summary.md`): 374 borderline entries in 16 batches were judged by human
reviewers and 3 coordinator overrides applied, so `symmetry` holds the confirmed set — 380 of the 1480 surveyed
models are symmetric, 1100 are `none`, and 18 keep a `symmetry_near` entry: 17 `flat_render` models (assets whose
reflection / emissive map carries the visible pattern, which the albedo render cannot show — their tags stay in
`symmetry_near` until they are rendered with their reflection maps) and one `unsure` verdict (`ruler/ujizkx` `y2`).

1. **Mental rotation targets must be asymmetric** (decision 2026-09-15): rotationally symmetric models are
   **removed from the candidate list** — `build_icl.py` drops every `mental_rotation` / `target` row whose
   `symmetry ≠ none` and promotes the next-ranked model of the category to `representative` / `test` when a pick was
   dropped (the drop is listed in the build report and the row's `note`). The five symmetric picks of the selection
   sheet (`teacup/zdvgol`, `teacup/cpozxi`, `teacup/tfzfam`, `hook/wmvozo`, `cookie_cutter/lqrfzo`) and the four
   the review round confirmed (`pickle/ifterk`, `pickle/iobrqp`, `tongs/igsklx`, `tongs/gbjqyn`) are dropped
   (`runlogs/symmetry_20260915/dropped_mental_rotation.csv`, 9 rows). The models stay candidates for every
   other role. `ground_isg` additionally rejects a symmetric model that reaches it through a customs-list override
   (`symmetric_target`). Object scaling has no such rule (its goal is a depth, symmetric bodies are welcome).
2. **Equivalences from the tags**: for every orientation-evaluated target (held object, disembedding measure,
   IORS small-scale, attachment members, alignment) `ground_isg` writes `orientation_equivalences` as the rotation
   group generated by the `symmetry` tags, in the object's canonical frame: finite groups are enumerated (a `z2 x2`
   box gives 4 elements, `z4 x2` gives 8), continuous axes are enumerated at 5° steps (72 quaternions per axis, finer
   than any orientation tolerance in section 8), `full` marks the target `orientation_metric: null`. Blocks keep no
   equivalences (six distinct faces). Wall targets: the reviewed `symmetric_180` column stays the source of the
   half-turn about the wall normal; the survey's disagreements went to the review round as explicit questions and
   the column was corrected from its outcome, not from the geometry alone (21 rows answered, 7 values changed:
   `clipboard/gmxyfo`, `towel_rack/crivoa`, `towel_rack/kqrmrh`, `whiteboard/gvkgsi` → `1`; `shelf/iruewm`,
   `shelf/qrmiur`, `shelf/wvkefp` → `0`).
3. **Look-alikes**: a rotational symmetry is not a look-alike relation (3.12); the two columns are independent.

### 3.14 What the TRO must let `run_isg` restore (loader contract, TODO F1)

The sampler only guarantees what the loader can reproduce. Beyond the BEHAVIOR object states and `robot_poses`,
`load_tro_instance` restores, after **every** `scene.reset()`:

- `constraints` — the COM hinges of the alignment tasks (revolute joint parent = support root link or wall, child =
  target, anchor = COM, axis as recorded) are re-created **with the recorded hinge resistance** (§7: `friction` /
  `damping` of the wall hinge as the joint's angular drive; absent = frictionless, the axial table hinge); the table
  hinge sits 1 mm below the top so the object cannot rock; a hinge that is missing at evaluation is a loader error,
  not a free-spinning object.
- `grasps` — the held object of object scaling / mental rotation: the object is placed at
  `grasps[0].object_pose_in_eef_frame`, the recorded finger joints are applied and the assisted grasp is attached
  before the first policy step; an attach failure fails the load (`grasp_restore_failed`), it never silently starts
  ungrasped.
- `attachments` — the named male / female meta-link pair and the collision filter between the two members, both at
  load and when the policy attaches during the episode (the v1 open point is closed by the loader, not by sampling).
- `articulation_targets` and the initial joint values of every non-fixed joint of the target, with the moving link
  held during the settle so lids and doors do not drift before step 0.
- `orientation_equivalences` (3.13), `pose_targets` metrics / frames / tolerances, `groups` with `target_slots` and
  relations, `initial_occlusion` (6.4), and the `evaluation_profile` (`isg-v1` unless the customs list overrides:
  camera key `robot_r1:zed_link:Camera:0`, 1080 × 1080, 40 mm aperture, `n_render_iterations = 1`).
- Nothing is re-sampled at load. The reload validation of 3.5 is the test of this contract.

## 4. Tabletop initialization

Applies to every task whose objects rest on a fixed support: disembedding, block design, attachment, axial
alignment, IORS ontop / under / inside / nextto / touching (small-scale), and — with the substitutions of section 5.4
— small-scale articulation. One procedure replaces the per-task stance and view code of v1. Arms always untucked,
grippers open; trunk JoyLo (t, k). Chairs within 1.0 m of the support are removed (3.1). Coordinates in the base frame;
h = top height, s = stand-off.

### 4.1 Reach-band script

**Band scope (owner decision 2026‑09‑16, A1).** Bands are per arm. Besides the right‑arm grasp / generic‑pinch bands,
a **left‑arm generic contact band** is computed per support and 2 cm object‑height class: closed gripper, pads on the
near / far / left / right face at ¾ h or a top press, two closing axes, eef = point − CONTACT_CENTER_Z·a (the contact
set depends on the contact height and approach, not on the object's grasps, so one band serves a height class). The
generic push / pinch set has the approach pitches {0, 30, 45, 60}° (the pitched classes keep an IK‑feasible push arc
at far cells, 0.9–1.0 m). Every generic / contact band records a per‑cell class bitmask (`meta.classes`). When a
trunk self‑collides with untucked arms the script retries it with the alternative idle posture (A6) and records the
posture used.

```
python -m ispatialgym.sampling.reach_bands --scene S --support T --target CATEGORY/MODEL [--grasp-key K]
```

For one (scene support, grasped object) it computes, in base coordinates at one concrete stance (edge midpoint,
s = 0.45), for every trunk of the grid (3.3):

1. Whole-body collision check of the trunk with untucked arms, no objects; skip colliding trunks.
2. Lattice (8 cm) over the near strip at the object's grasp height; drop cells farther from the right shoulder than
   arm reach + largest grasp offset (geometric, no IK).
3. Library-grasp mask on the remaining cells: object posed at 8 yaws × every library grasp, pre-grasp 10 cm back
   along the grasp approach; 12-seed batched IK; locked joints = the untucked init pose at that trunk; support ignored
   for the grasp pose only. `(yaw, grasp)` pairs are evaluated in rounds — best-ranked grasps and robot-facing yaws
   first — with early exit per cell (one feasible pair marks the cell). Objects without library grasps (blocks; axial
   push poses) use the generic top-down pinch (2 wrist yaws × 2 approach pitches at the cell) as their grasp set.
4. Output per trunk: reachable cells (x ahead, lateral); plus the union extent [λ_min, λ_max] over all trunks.

Cache key: 3.10 (two factors). Not the base pose; the stand-off is not a dimension (a different stand-off only
changes which cells fall on the top). Expected cost: 5–20 s per surviving trunk, 1–4 min cold per (support, object).
`sample_isg` calls the function on a cache miss.

### 4.2 Phase 1a — edge and stand-off

Any edge of the top polygon (raycast raster), no preference; yaw = inward normal; s = 0.45 m from the top edge, plus
the AABB overhang beyond it when ≥ 5 cm.

### 4.3 Phase 1b — init trunk height (tilt 0)

At tilt 0 the camera pitch is constant (≈ 22.6° down `[pilot]`), the camera is x_c ≈ 0.22 m ahead of the base
`[pilot]`, its height z_c(t) decreases monotonically with t, and the untucked hands are 0.19 m below the camera
`[pilot]` (all four from the FK tables of 3.3). Workspace = band union ∩ top of this edge; d_ws, d_near = depth from
the edge of its centre / near edge; H_max = tallest object to be placed; φ_m = image-border angle including the
visibility margin.

```
Δ_aim = (s + d_ws − x_c)                          # 45° elevation onto the workspace centre
Δ_min = H_max + 0.10 + 0.19                       # hands ≥ 10 cm above the tallest object
Δ_max = (s + d_near − x_c) · tan(22.6° + φ_m)     # workspace near edge inside the frame
Δ     = clamp(Δ_aim, Δ_min, Δ_max);  t* = z_c⁻¹(h + Δ), clamped to [0, 1]
```

Record `init_stance = {stand_off, translate: t*, tilt: 0}`. From this camera compute b (lateral offset of the image
border at the source-tray depth) and R_img (surface distance from the target to the image border on a given side).

### 4.4 Phase 1c — lateral position

1. Margins. w_x = half the larger horizontal bbox side of object x (random yaw); [λ_min, λ_max] = band union extent
   (λ_min ≤ 0 right end, λ_max left end) — for two‑arm tasks the union of the working arm's band and the left contact
   band (A1; `[pilot]` on `breakfast_table_rhjoby_0` this makes block design 1.21 m and axial 1.46 m, so only the
   2.9 m edge admits them; the implementation keeps the working‑arm band as the default and the union behind
   `ISG_MARGIN_UNION=1` until the owner confirms); W_p = the partner's top-down AABB extent along the edge (canonical
   orientation); D_p = maximum edge-to-edge gap target–partner (the partner's side is random, so both margins carry
   it); safety 0.02.

   D_p = min(W_p + 0.03, R_img) for ontop, under, inside, touching and nextto; nextto additionally rejects the pair
   when D_p < g_NT + 0.03, with g_NT = mean(dims_target + dims_partner)/6 (element-wise sum of the two AABB extents,
   then the mean — OmniGibson's `NextTo` threshold).

   | task | m_R (robot's right) | m_L (robot's left) |
   |---|---|---|
   | ontop, under, inside, nextto, touching | \|λ_min\| + w_t + D_p + 2·w_p + 0.02 | max(0, λ_max) + w_t + D_p + 2·w_p + 0.02 |
   | attachment | \|λ_min\| + max(w_target, w_partner) + 0.02 | max(0, λ_max) + max(w_target, w_partner) + 0.02 |
   | axial alignment | \|λ_min\| + w_diag + 0.02, w_diag = ½√(long² + short²) | max(0, λ_max) + w_diag + 0.02 |
   | block design | \|λ_min\| + 0.0125 + 0.026 + 0.02 | max(0, λ_max) + 0.0125 + 0.02 |
   | disembedding | b + 0.437 + 0.02 | b + 0.437 + 0.02 |

   (block design: half edge + one pitch to the robot's right of the anchor; disembedding: 0.437 = the two fixed trays'
   half lengths along the edge + gap, recompute if the trays change.)
2. Lateral position uniform over [−L/2 + m_R, L/2 − m_L] (L = edge length); skip the edge if m_R + m_L > L. Valid
   when (1) the base centre is on free floor; (2) the resulting configuration — base (x, y, yaw), trunk (t*, 0),
   untucked arms, grippers open — is whole-body collision-free (it is the instance's **init pose**); if not, move t
   within [Δ_min, Δ_max] (recompute b, R_img and the margins) or redraw the lateral position; (3) base + torso links
   + head are collision-free (cuRobo, 1 cm) at translates 0, 0.25, …, 1.0, tilt 0, arms excluded.

### 4.5 Phase 2 — manipulation stance

Grasps are executed from this stance without re-posing the arms, so each candidate trunk uses its own band. Scan
stand-off {0.35, 0.45, 0.55} × translate {0, 0.25, 0.5, 0.75, 0.95} × tilt continuous from 0 to the `torso_joint3`
limit (same edge and lateral as the init stance):

1. Whole-body collision check without objects; drop colliding candidates (this also prunes trunks whose bands are
   never requested).
2. Band = the grid trunk with the same translate and the nearest tilt; focus candidates = its cells inside the top
   eroded by the target's half-extent + 0.02 (band shifted with the stand-off).
3. Keep candidates where the optical axis is within 8° of the focus (if none, within 20°) and the target AABB is in
   frame (5 %). Disembedding: the whole source tray (centred on the focus, long axis along the edge) in frame (5 %).
   Block design: the focus is the centre of a region of band cells with ≥ n_blocks + 2 cells at spacing edge·√2 +
   0.03, the 2 × 2 goal grid (pitch = edge + 1 mm) still on the top, region in frame (8 %).
4. Draw uniformly among the survivors in the best angle class (never always the smallest angle: the target would land
   on the same cell every instance); sample tilt continuously within the surviving band. Record
   `manipulation_stance = {stand_off, translate, tilt, focus, base_pose2d}`.
5. Transition check init → manipulation (3.8); on failure take the next survivor.

### 4.6 Phase 3 — sample objects

**Joint two‑arm placement (A1).** For a two‑arm task (axial stop edge, attachment partner brace, block row / column
braces, touching partner brace) the working point must lie in a right‑band cell and the brace point in a left contact
band cell **of the same trunk**, the brace's contact class passing where the band records classes; then the exact
two‑arm IK of step 6, the touched objects ignored only for the contact poses and the combined check. The push / stop
approach pitch follows the cell's passing classes (shallowest first); 45° / 60° are tried before rejecting. A stop
blade (axial: only when the joint placement exists — otherwise push‑only, 6.10) stands `finger_profile(h above the fingertips) + 5 mm` outside the goal face — the closed R1Pro finger is a
wedge: outer face 8.4 mm from the closing plane over the last 3 cm of the tip, 18 mm above, palm beyond ≈ 7 cm — on
the part of the face the turning object does not sweep before the goal, and is checked against the target at its
goal pose (3 mm margin); the push contact keeps its ignore. `[pilot]` finding: on `breakfast_table_rhjoby_0` the
right band ends at λ ≤ 0.0 and the left contact band starts at λ ≥ 0.08 at the manipulation trunks, and the sweep‑clear
part of a stop face lies within ≈ 7 cm of the pushed end, so push and stop points share one lateral — the two‑arm
axial stop is at the robot's limit there (0 / 8 with both bands).

1. **Target init pose.** Screen the 8 fixed yaws (45° steps) × library grasps at the focus cell with the 12-seed IK
   from the manipulation stance → feasible set F; store F in the band cache under (trunk, cell) (empty F removes the
   cell). Draw (yaw, grasp) uniformly from F, add a uniform jitter of ±22.5° to the yaw, place the target within 5 cm
   of the focus with its footprint on the top (2 cm margin). Derive the eef poses from the drawn grasp at the actual
   object pose — grasp = object pose ∘ T_obj→eef, pre-grasp 10 cm back along the approach — and confirm both with the
   exact IK from the manipulation stance (idle arm untucked and locked; support ignored for the grasp pose only). On
   failure draw the next element of F without replacement; reject the attempt when F is exhausted. Record
   `(yaw, grasp_key, grasp_name)` and both eef poses.
   - *Disembedding*: source tray at the focus, aligned with the edge; target inside it (yaw as above, footprint on
     the tray floor, 0.5 cm margin). Destination tray on the left or right (random), aligned with the edge and the
     source tray (same depth), at the lateral position where exactly 4 cm of it is inside the **init** image (border
     at tray depth = camera-frame depth × tan 49.6°, confirmed by one segmentation render), ≥ 4 cm from the source
     tray, footprint on the top.
   - *Blocks*: one block per region cell drawn at random with ≥ 3 cm gaps between footprints, distinct up faces,
     **no block's up face is the opposite of its goal face** (section 6.5), and the **visible-face rule** (kept from
     v1 §5.4): every pattern used in the goal is visible in the init image — a block whose goal face is its up face
     shows it on top; a block that flips gets the coarse yaw (multiple of 90°, before the jitter) that turns its goal
     face toward the robot, so the goal face is a robot-facing side face. Each goal face must be ≥ 50 % visible from
     the init stance (ray test on a 3 × 3 grid of face points against the other blocks and the scene; the front row
     hides the lower part of a back-row block's front face at the camera's elevation) — otherwise redraw the cells.
     The two rules do not compete: the flip rule fixes *which* face (up or an adjacent face) is the goal face, the
     visible-face rule fixes the *yaw* about the vertical; for 2 × 2 the four distinct up faces plus four
     robot-facing side faces always suffice, and a pattern that is on no up face is adjacent to at least three of the
     up faces (only its opposite is not), so a block that can show it always exists. Yaw jitter as before on top of
     the coarse yaw; reach = the cube grasp set of 6.5, confirmed per block with the exact IK for pick **and** place.
   - *Axial alignment*: yaw = aligned yaw ± U(20°, 70°) with random sign, footprint on the top, at least one end of
     the long axis inside the band and its top-down push pose confirmed with the exact IK; the stop edge for the other
     arm (6.10) reachable.
2. **Partner** on a random side of the target with edge-to-edge gap g: ontop / under / inside g ∈ [0.03, D_p];
   nextto g ∈ [g_NT + 0.03, D_p]; touching g ∈ [0.03, D_p] **on the robot's left** (the right hand pushes toward it,
   the left hand braces its far side, section 6.11); attachment: partner on a cell of the attach-reach band (every
   attachment pose reachable ignoring collision with the target), gap ≥ 0.04, centre distance ≥ 0.08. Footprints on
   the top, ≥ 3 cm apart.
3. **Distractors** (disembedding) on the source tray, each ≥ x from the target along the drawn grasp's pinch axis,
   x = (open width 0.10 − pinch width of the drawn grasp)/2 + 0.01 (`Grasp.pad_clearance` + margin), so the grasp
   stays possible with the distractors placed; they may occlude the target up to **30 %** of its visible area
   (segmentation render, distractors visible vs hidden); no other occlusion bucket exists.
4. Every sampled object fully in frame (5 %) in the init image except the disembedding destination tray.
5. Whole-body collision check at the init pose and at the manipulation stance with the objects placed; on failure
   take the next Phase-2 survivor before rejecting the attempt.
6. **Brace / second-arm poses** where the oracle uses them (block design row-left / column-top brace, attachment
   partner brace, axial stop edge, touching partner brace): pre-contact + contact IK with the other arm, the working
   arm locked at its solution, combined whole-body check (section 6 per task).
7. Then the existing pipeline (3.5): settle, BEHAVIOR validation, `:init` true, binary goal false, goal feasibility by
   teleport + settle, TRO + sidecar with `init_stance`, `manipulation_stance`, transitions, grasps, contacts.

### 4.7 Disembedding placement stance

The carried target is placed at the destination centre from a **placement stance**: the manipulation stance shifted
along the edge so that the destination centre is at the focus (same stand-off, translate, tilt). Record
`placement_stance`; check: base on free floor and whole-body collision-free with the trays and distractors; the
place pose (goal target pose ∘ T_obj→eef of the drawn grasp) and its pre-place (10 cm up along −a) pass the exact IK
from the placement stance (tray and support ignored for the place pose only); the lateral drive between the two
stances is a straight segment validated per 3.7 with the object attached at the **carry height** (lift ≥ tallest
distractor + tray rim + 3 cm above the source floor). The drawn `(yaw, grasp)` must pass both the pick and the place
check (draw the next element of F otherwise).

## 5. Large-scale initialization (shared robot init for scene targets)

Applies to every task whose target is a fixed scene object or is hinged to a fixed support: articulation
(large-scale, open / close) — the scene articulated object is the target; world vertical / horizontal — the target
hung on the wall support. Arms untucked, grippers open; trunk JoyLo (t, k). Face frame as in 3.3. Untucked hands sit
0.64 m ahead of the base at 1.20 m, so a single-arm stance closer than ≈ 0.7 m to a face fails the whole-body check;
rely on the check, not on a bound.

### 5.1 eef reach volume (robot-only cache)

```
python -m ispatialgym.sampling.eef_reach_volume compute --arm right|left|both [--all | --translates … --tilts …] --out DIR
python -m ispatialgym.sampling.eef_reach_volume merge --out DIR
```

Object- and scene-independent: large-scale contacts are handle bars / knobs and push or brace contacts whose eef
orientations are fixed by geometry, so one cache serves every scene and scale. For every trunk of the grid and each
arm: (1) whole-body self-collision check with untucked arms (floor plane only); (2) 5 cm lattice of eef positions in
the base frame, x ∈ [0.25, 0.90], λ ∈ [−0.70, 0.70], z per class group, pruned to positions within arm length + 0.05
of that arm's shoulder; (3) orientation classes (a = approach = eef +z, c = closing axis = eef y): `fwd_pinch_v`
(a +x, c +z) and `fwd_pinch_h` (a +x, c +y) for handles / knobs (z ∈ [0.20, 1.70]); `push_up` (a +z), `push_down`
(a −z), `push_left` (a +y), `push_right` (a −y), all c = +x, pads parallel to the wall (z ∈ [0.90, 1.90]); pre-pose =
pose − 0.10·a; a cell is feasible when pose and pre-pose pass the 12-seed IK with the untucked init at that trunk as
locked joints, floor as the only obstacle; (4) output per (arm, trunk, class): boolean mask;
`reachable(arm, trunk, class, p_base)` = nearest-cell lookup.

Lookup of a world eef pose (p, R) at a candidate base (b, ψ): p_base = R_z(−ψ)(p − b); class = the one whose (a, c)
lie within 20° of R_z(−ψ)·(a_world, c_world) with **c compared up to sign** (the closing axis is undirected); push
classes: nearest of the four; no class → infeasible for screening. Cache key: robot model + arm + posture + trunk
grid + lattice + classes + IK settings. Cost: 1–3 min per (trunk, arm) → 20–60 min cold per arm, once per robot
(sharded per (arm, trunk) across pods).

### 5.2 Prompt 0 — stance skeleton

**Phase 1 — init pose.**

1a. Approach ray: bearing β ~ U(−β_max, β_max) (per task) about n through P; base positions b(d) = P_xy + R_z(β)·n_xy·d,
base yaw faces P, tilt 0 throughout.

1b. Minimum distance with the whole target in view. Target extent = AABB corners of the target in its init state
(articulation: joints at q_init, an open door counts; alignment: the hull at the init tilt δ). Scan d from d_min in
5 cm steps to d_max; at each d, for t ∈ {0, 0.25, 0.5, 0.75, 0.95}: (1) base centre on free floor (un-eroded map) and
inside the task room; (2) all target corners in frame (5 %) at camera FK (b(d), t, 0); (3) whole-body collision-free
(1 cm) with untucked arms and open grippers, world synced with the target in its init state; (4) base + torso links
+ head collision-free at every t ∈ {0, …, 1.0}, tilt 0, arms excluded. d* = the first d with a feasible t; t* = the
feasible t whose optical axis is closest to the target centre (tie → smaller t): step back until the object fits,
then forward while it still fits. No feasible d → redraw β (≤ 6), then reject `no_stance`. Record
`init_stance = {face_normal, bearing_deg, distance: d*, translate: t*, tilt: 0, base_pose2d}`.

**Phase 2 — manipulation stance** (contact set, arm rule and focus per task). Base square to the face (yaw = −n).
Scan d ∈ {0.35, 0.40, …, 0.90} × λ ∈ Λ_task (5 cm) × translate {0, 0.25, 0.5, 0.75, 0.95} × tilt continuous from 0 to
the `torso_joint3` limit (band = the grid trunk with the same translate and the nearest tilt):

1. Base on free floor, in room; collision-free (1 cm) against the scene with the target in its init state. Single-arm
   tasks: the idle arm untucked is included; two-arm tasks: arms excluded (IK-checked in step 5).
2. Every required contact of the task's contact set (pose + pre-pose, per arm) feasible in the reach volume at this
   base and trunk.
3. Visibility: the task's manipulation focus in frame (5 %) and the optical axis within 8° of it (if none, 20°). The
   whole target need not be in frame.
4. Draw uniformly among the survivors in the best angle class; sample tilt continuously within the surviving band.
   Record `manipulation_stance = {stand_off: d, lateral: λ, translate, tilt, focus, base_pose2d, arms, contacts}`.
5. Exact IK from the drawn stance for the whole contact set (locked trunk; idle arm untucked and locked; the
   contacted object ignored for the final contact pose only unless the task says otherwise; world re-synced per
   waypoint wherever the target moves). Failure → next survivor; exhausted → reject.
6. Path init → manipulation stance per 3.7, length ≤ 3 m; transition per 3.8.

Cache per (scene, target, init state): survivors of steps 1–3 that do not depend on continuous per-instance draws;
key = scene + target name + init state (joint value quantised 0.01, or none) + trunk grid + lattice + arm rule. Per
instance: the draw, step 5, step 6. Variety comes from β, the step-4 draw and the continuous tilt.

### 5.3 Prompt 1 — articulation, large-scale (open / close)

Target bound to a scene articulated object (`bindings.target` names, from the `roles/candidates.csv` scene rows of the task rooms:
cabinets, fridges, ovens, dishwashers, drawers, doors, iddl `(handle target)` satisfied); one joint per object
(`roles.target.joint`, `roles.target.handle_key`).
Handle annotation `artic/handles/<key>.json` → joint, contact, pinch_axis, approach_dir (target frame, closed
configuration), kind (bar / knob / edge / fallback), outward_normal, axis_dir_obj, hinge_point_obj — **scaled by the
scene instance's per-axis scale** before use (scene instances are non-uniformly scaled; the pinch width must stay
within 15–80 mm after scaling). Single arm; the object holds itself, no brace arm. Uses Prompt 0 with:

- **Joint states.** Stable set S: 7 fractions of `open_fraction_range` ([0.5, 0.95]) + the upper limit, each held 120
  physics steps, kept when it stays within ½ tolerance (revolute 0.10 rad, prismatic 0.02 m). S empty → reject the
  template. Objects whose partial opening never holds settle to the same angle every time: that angle is recorded
  per object and is the only open target (open) or the only open init (close). Open: q_init = lower limit on every
  joint, q_goal ~ U(S). Close: chosen joint q_init ~ U(S) (must satisfy OmniGibson `Open`), others closed, q_goal =
  lower limit.
- **Handle trajectory h(q)**: contact / pinch / approach of the closed configuration moved with the link — revolute:
  rotation about (hinge_point_obj, axis_dir_obj) by q − q_lower; prismatic: translation along axis_dir_obj. Eef pose
  at q: approach along approach_dir(q), closing along pinch_axis(q), origin = contact(q) − 0.012·approach, pre-grasp
  0.10 m back.
- **Waypoints** q_k = q_init + k·(q_goal − q_init), k ∈ {0.25, 0.5, 0.75, 1.0}. Required handle poses
  (`generation.require_goal_handle: auto`): q_init plus every q_k up to and including the first k from which the link
  reaches q_goal passively (set q_k, 120 steps, within tolerance of q_goal; probed once per template, per q_init for
  close); if no k passes, all waypoints including q_goal are required. `true` / `false` force q_goal in / out. Summary:
  `handle_reach = {q: reachable}` for the four waypoints and `passive_from` = the first passing k.
- **Arm rule**: the arm on the handle's side of the face centre P — handle > 0.10 m to the robot's left of P (facing
  the face) → left arm, > 0.10 m right → right arm, else right; `generation.arm` overrides. Requires the left-arm
  cuRobo embodiment.
- **Phase 1**: β_max = 20°, d_min = 0.5 m, d_max = 2.5 m; the whole target at q_init in frame (5 %; an open door counts).
- **Phase 2** (d scan extended to {0.35, …, 1.10}: an open door protrudes): λ measured from the handle contact's
  projection onto the face; right arm Λ = [−0.10, 0.60] (base to the left of the handle), left arm Λ = [−0.60, 0.10].
  Contact set screened by the volume: h(q_init) grasp + pre-grasp; class `fwd_pinch_v` if |pinch_axis(q_init)·ẑ| > 0.7
  else `fwd_pinch_h` (knob: either). Focus = contact(q_init); visibility: contact(q_init) and the moving face patch
  (contact ± 0.10 m along the pinch axis) in frame (5 %), optical axis within 8° (else 20°). Step 5 exact IK:
  [pre-grasp, grasp] at q_init, then the grasp pose at each required q_k with the world re-synced at q_k (target
  ignored for the grasp poses only). Cache per (scene, target name, joint, q_init) (close: one entry per element of S).
- **Articulation class and follow mode (owner decision 2026‑09‑16, A4).** The handle is grasped in both directions and
  the pinch is tried with both closing signs (the recorded pinch axis carries the sign). Class per joint:
  `drawer` (prismatic) — **base** mode as before: with the arm locked at its q_init IK solution q*, the eef pose relative
  to the base is fixed, so b_k = h(q_k) ∘ T_eef→base(q*); per required q_k: b_k on free floor (un-eroded), whole-body
  collision-free with the target at q_k (the grasped link ignored for the gripper only), base path b_0 → b_k per 3.7;
  `cabinet` (revolute door or flap of a furniture piece: cabinets, fridge, dishwasher, oven, …) — the manipulation
  stance stands on the **hinge side** (Λ toward the hinge, the base between handle and hinge; the arm on the handle's
  side of that stance), `follow_mode: arm`, and the goal set of a stance is S restricted to the joint values the arm
  reaches from it (h(q) on a 10° / 5 cm grid from the closed joint up to the first unreachable value, the target
  ignored for the grasp only); when no element of S is reachable, the reach limit itself is the goal provided it opens
  ≥ 50° (0.10 m) and holds passively — a 50–70° opening is a valid configuration-matching goal; `reachable_fraction`
  per stance in the template meta and the articulation block; `room_door` (revolute door leaf) — **hybrid** follow:
  per handle waypoint a base pose near the current base on free floor, collision-free, with feasible arm IK, minimum
  displacement (not implemented yet: `Reject("room_door_hybrid_todo")`). The rigid eef→base sweep is no longer used
  for revolute furniture doors (the pilot: the base orbits the hinge at ≈ 1 m through 55–110° into the kitchen).
  Record `door_class`, `hinge_side`, `follow_mode ∈ {arm, base, hybrid}`, `goal_set`, `q_reach`, `base_follow`
  (base mode only).
- **Phase 3**: nothing is spawned. Joints at q_init, `settle_free = []`, the moving link held at q_init during
  settling; after settling the joint may not have drifted > ½ tolerance; `articulation_targets` = every non-fixed joint
  at its init value except the chosen one at q_goal, tolerance per type; then the pipeline (`:init` true, binary goal
  false, goal feasibility: set q_goal, settle 120 steps, all joints within tolerance, robot parked). Sidecar:
  `init_stance`, `manipulation_stance` (arm, handle class, focus), `handle_waypoints` = h(q) for the four waypoints
  (contact, approach, pinch, world frame), `handle_reach`, `passive_from`, `follow_mode`, `base_follow`.

Kept from v1: stable-angle probe, tolerances, `Open` semantics, whole target in frame at init, pre-grasp 10 cm / eef
offset 1.2 cm. Replaced: `scene_distances` × lateral list and the drive-forward {0, 0.15, 0.3} (subsumed by the d
scan), `view_translates` (Phase 1b), tucked reach arms, right-arm-only. **Dropped**: the v1 model-swap path
(`scene_assign`, a candidate articulated model swapped into a scene host slot) — v2 large-scale targets are the
scene's own objects, nothing is swapped in.

Handle rows without a protruding handle (icl `require/handles.csv`): an **edge pinch** (`<joint>#F`, verdict `edge`)
is the pinch across the leaf thickness `width_m` at the recorded point of the exposed edge (the free edge of a lid
or door, the top lip of a drawer front), pinch axis = the leaf normal, approach = into the edge along the leaf plane,
same 1.2 cm eef offset and 10 cm pre-grasp; `edge_blocked` rows are not admissible. A reviewer pinch (`#M`,
`manual`) is used as recorded; `manual_unscaled` (pinch placed on the canonical model of a non-unit-scale scene
instance) is rescaled by the instance's per-axis scale before use and its `width_m` re-checked against the 15–80 mm
pinch range. Rows with verdict `skip` or `ungraspable` exclude the joint.

### 5.4 Prompt 2 — articulation, small-scale (open / close)

Tabletop Phases 1–2 for stance and view (support = the fixed `support`, or the floor for objects taller than 0.6 m);
everything object-related follows this section, which wins over section 4 where they differ — small-scale
articulation is not a pick-and-place (some objects have no grasp, only an edge pinch). Roles: `target` (spawned
articulated object; `roles.target.joint`, `handle_key`), scene `support`. Handle annotation, joint states, S (probed
on a band cell of the top), h(q), waypoints and the auto required-waypoint rule as in 5.3.

- **Target yaw is not free**: the facing direction f (target frame: the moving face's outward normal; for lids the
  direction from the hinge line to the target centre, so the lid swings away from the robot) points at the robot:
  yaw = face_yaw(edge) + U(−0.1, 0.1).
- **Arms.** Handle arm = the arm on the handle's side of the target centre (> 0.05 m; else right); brace arm = the
  other. The brace only keeps the body in place: it acts on the side opposite to the pushing direction.
- **Push axis and brace contact.** v = unit horizontal component of the handle's motion at q_init toward the first
  required waypoint (dh/dq · sign(q_goal − q_init)); if the horizontal component is < 30 % of the motion (a lid lifted
  straight up), v = the horizontal direction from the hinge line to the handle contact. Brace candidates on the base
  link (never the moving link), first reachable wins (`brace.kind` recorded): (1) `opposite_face`: the base-link face
  whose outward normal is closest to v, contact at the face centroid shifted ¼ of the face width toward the brace
  arm's side, a = −v (into the face), c = +y_base; (2) `top_press`: downward press on the exposed top of the base link
  (raycast from above with the moving link hidden at q_init; exposed cell nearest the brace arm's side), a = −ẑ,
  c = +y_base — table friction holds the body when the opposite face is unreachable (drawer pushed closed). For both:
  contact ≥ 2 cm outside the moving link's swept AABB over the required waypoints; pad centre on the face, eef origin
  = point − `CONTACT_CENTER_Z`·a, pre-contact = eef − 0.10·a; one pose (the base link does not move).
- **Reach-band script**: `reach_bands --scene S --support T --target toolbox/redmtl --handle-key toolbox__redmtl
  --joint j_lid --joint-values <lower>,<S>,<waypoints> --brace` — grasp set = the handle [pre-grasp, grasp] at each
  listed q, target posed at the cell with yaw ∈ {face_yaw − 0.1, face_yaw, face_yaw + 0.1} (3 yaws, no library
  grasps), screened with the handle arm (brace arm untucked); the brace candidates [pre, contact] screened with the
  brace arm (handle arm untucked). Output per (trunk, cell): the set of joint values whose handle poses are feasible
  plus the feasible brace kinds; a cell is in the band for a task direction when the required waypoints ⊆
  feasible(cell) and ≥ 1 brace kind is feasible. Cache key as in 4.1 with handle key + joint values + brace in place
  of the grasp set.
- **Phase 1b**: H_max = target height at q_init (an open lid counts). Margins: m_R = |λ_min| + w_a + 0.02,
  m_L = max(0, λ_max) + w_a + 0.02, w_a = half the larger horizontal bbox side at q_init. In-frame margin for the
  target stays `view_margin` = 0.03 on the exact hull (the toolbox is 0.6 m wide).
- **Phase 2**: stand-off {0.35, 0.45, 0.55}; band cells filtered for the task direction; visibility: the whole target
  at q_init (hull, `view_margin`), the handle contact and the brace point in frame; optical axis within 8° (else 20°)
  of the target centre.
- **Phase 3**: place the target at the focus cell (± 5 cm), yaw as above, footprint on the top (2 cm), joint at
  q_init. Exact 32-seed IK from the manipulation stance, in order: brace arm (handle arm untucked and locked;
  pre-contact against the full world, contact with the target ignored; first feasible candidate), then the handle
  arm with the brace arm locked at its solution ([pre-grasp, grasp] at q_init, support and target ignored for the
  grasp pose only; grasp poses at the required waypoints with the target re-posed for each world sync), then a
  whole-body collision check of the combined configuration against the scene (target excluded: both hands touch it).
  Settle with the joint held at q_init; drift ≤ ½ tolerance; `pose_targets[target]` measured (0.03 / 0.20) and
  `articulation_targets`; pipeline. Sidecar: `init_stance`, `manipulation_stance` (handle arm, brace = {kind, face,
  point, direction}), `handle_waypoints`, `handle_reach`, `passive_from`, `follow_mode: arm`.

Removed from v1: `lateral_center` −0.15, `lateral_half` 0.15, `depth_range` (band), `view_prefer_translate` 0,
`approach_forward`, `stand_offs` [0.45, 0.55].

### 5.5 Alignment — world vertical and world horizontal

Roles: `target` (hung; iddl `(contacts target)`), scene `support` (the wall: `bindings.support` = allowed wall object
names, `bindings.support_regions` = the surveyed wall segments). The hinge below is an initialization rule of this
section, not an iddl requirement. Prompt 0 with the wall as the face (n = wall normal, P = hinge
point on the wall at hang height). Arc lattice, single right-arm push into the wall and `view_translates` of v1 are
replaced by the Phase 1 ray + Phase 2 lattice and the tangential push + brace below.

Kept (target side): wall spots from `bindings.support_regions` (0.25 m seeding) or the 24-ray search; flat patch 5 × 5
rays within 1 cm over (w + 0.10) × (h + 0.10) (horizontal: swept extents over the max deviation, `swept_patch`);
front clearance 0.35 m; in-room test 0.8 m in front; one wall per template (most spots); frame from `front_axis` /
`up_axis` / `level_axis` (dict form), `symmetric_180`; hang pose (up axis along world z, front axis along n, centre at
height 1.45 m, thickness/2 + 0.01 off the wall); hinge at the COM with axis n, parent = the support's wall link (TRO
`constraints`, with the §7 hinge friction / damping); init in-plane angle δ = sign · U(15°, 25°) (owner decision 2026‑09‑16, A8: |δ| > 25° is beyond the
one-stroke push of the R1Pro; a two-stroke push for larger tilts is recommended later); the template filter uses
δ = ±25°; goal = level pose (rotation 0), orientation tolerance
0.08, equivalences (`symmetric_180` half turn + explicit); `settle_free = []`; hinge drift ≤ 0.03 rad after settling;
≥ `min_visible_objects` (5) objects with ≥ 200 px, structure included, in the init render; whole tilted target in frame
at init (5 %).

**Frame** (reviewed, never defaulted; owner 2026‑09‑16 22:55): `ground_isg` writes the icl frame of every whitelisted
wall target (`roles/candidates.csv` `frame_front` / `frame_up`, and `frame_level` for world horizontal) into the alias as
`generation.front_axis` / `up_axis` / `level_axis` in the per-model dict form (`{"license_plate/nwfbxn": "+z", …}`;
dormant / superseded aliases of the subtask included). The sampler builds the hang pose from that record only —
`Reject("missing_frame")` when the alias carries no entry for the spawned model (no default frame inside the sampler:
the painting-shaped `+x` / `+z` default hung the `+z` / `+y` / `+x` license plate table-flat, run 0916‑212037) — and
records the frame used in `template_facts.frame` and the instance summary (`frame`). Unit test: for every reviewed wall
target the hang quaternion puts the up axis along world +z and the front axis along n; a flat object's thickness axis
ends horizontal.

**Scale** (configured, not searched): `roles.target.scale` in the customs list ([sx, sy, sz] per model, or the dict
form keyed by `<category>/<model>`, `<category>`, `default`; uniform recommended). Hull, COM, dimensions, swept
extents and wall spots are computed from the spawned (scaled) target; the reach cache is robot-only, pushing needs no
grasp re-validation. When the template fails with `push_unreachable` / `no_stance`, a diagnostic pass over uniform
s ∈ {0.9, 0.8, …, 0.5} on the same spots puts the first feasible s into the rejection message and
`template_meta.scale_hint`; it is never applied automatically — it is copied into the customs list by hand (or by the
icl review, TODO B6).

**Contacts**: push arm + brace arm. ℓ_δ = the target's level axis (side = up × front) rotated by δ in the wall plane;
C = COM. E_low / E_high = mean of the hull points within 3 cm of the two extremes along ℓ_δ (E_low the lower one), both
on the mid-thickness plane (thickness/2 + 0.01 off the wall). Reviewed contact candidates per model
(`contacts/<category>__<model>.json`, TODO B6) override the geometric E_low / brace points where present.

- Push arm at E_low: a = t(E_low) = unit tangent of the circle about C through E_low, signed so the low end rises
  (t·ẑ > 0); c = n; pad centre at E_low, eef origin = E_low − `CONTACT_CENTER_Z`·a; pre-contact = eef − 0.10·a; goal
  contact = the same material point at the level pose, a recomputed there; arc waypoint at δ/2 (3.9). Class = nearest
  push class of t. Push arm = the arm on E_low's side.
- Brace arm (a stop that arrests the rotation at the level pose; placed before the push, does not move; the other
  arm). Candidates in order, first reachable wins (`brace.kind`): (1) `below_high`: under the high end at its level
  position — B = mean of the level-pose hull points on E_high's side within 3 cm of the lowest z; a = +ẑ, c = n;
  fingertip face 5 mm below B; (2) `above_low`: over the low end at its level position — B = mean of the level-pose
  hull points on E_low's side within 3 cm of the highest z; a = −ẑ, c = n; fingertip face 5 mm above B; (3)
  `side_high`: against the outer edge of the high end at its level position — B = mean of the level-pose hull points
  on E_high's side within 3 cm of the outermost lateral coordinate, upper half; a = horizontal direction from B toward
  C, c = n; fingertip face 5 mm outside B; admitted only when the target's height ≥ half its width (frames). For every
  candidate: pre-contact = eef − 0.10·a; at init the brace hand must be ≥ 2 cm clear of the tilted hull (exact IK
  checks the brace with the target not ignored); class = nearest push class of a.

**Phase 1**: β_max = 35°, d_min = 0.5 m, d_max = 1.5 m; the spot is drawn per instance weighted by its number of
Phase-2 survivors.

**Phase 2**: Λ = [−0.40, 0.40] about C; arms excluded from the stance collision check (two-arm task). Contact set:
push arm {pre-contact, init contact, δ/2 contact, goal contact}; brace arm ≥ 1 candidate {pre-contact, contact}.
Focus = C; visibility: C, E_low and the chosen B in frame (5 %), optical axis within 8° (else 20°) of C. Step 5, exact
IK in order: brace arm (other arm untucked and locked; target not ignored; first candidate whose pre-contact and
contact pass), then push arm with the brace arm locked at its solution (target ignored for the push contact poses
only), then one whole-body collision check of the combined configuration (both arms placed, tilted target in the
world). Record `manipulation_stance` with `push_arm`, `brace_arm`, `contacts = {push: {pre, init, mid, goal,
direction}, brace: {kind, point, direction}}`, `deviation_deg`, `scale`. Cache per (scene, spot): step-1 survivors
(δ-, target- and scale-independent because arms are excluded); steps 2–3 are per-instance lookups. Template filter: a
spot is kept when ≥ 1 candidate passes steps 1–3 for δ = +30° and for δ = −30° at the configured scale; none left →
reject `no_stance` (with `scale_hint`).

**Phase 3**: hang the target at δ with the hinge, `settle_free = []`; pipeline (hinge drift, visible-object count,
`:init` true, goal false, goal feasibility by teleport to the level pose + settle). Sidecar: `init_stance`,
`manipulation_stance`, `push_contacts`, `constraints`, `scale`.

**Contacts near the wall (`[pilot]` 2026‑09‑16, implemented `2e29921` / `1f921e5` / `464259c` / `20cfe86`).** Every solved
contact configuration (brace pre / contact, push pre / init / goal, the combined brace + push) is re‑checked with the
wall present at margin 0 after the wall‑free IK: the pads may sit 2 cm off the wall, a wrist or wrist camera inside it
fails. The closing axis may lie in the wall plane (the gripper's thin side toward the wall; along n the palm penetrates
the wall at either wrist turn); the brace / push approach is pitched 30° / 45° off the wall toward the room (0° last;
with the approach in the wall plane the forearm lies along the wall 2 cm from it and penetrates it). Brace / push key
poses require the fingertips in the head camera frame (5 %) from the manipulation stance. The loose angle class stays
as the step‑5 fall‑back pool behind the best class. `[pilot]`: `painting_s070` writes an instance in 2 attempts,
`license_plate` (horizontal) at attempt 0.

**Two-arm stance posture (A6).** The manipulation stance records the idle-arm / drive-in posture it was checked with
(`arms_q`, `idle_posture`, `fingers`; the wall uses `tucked` with closed pads: hanging arms clear the wall at
d = 0.5–0.55 m where untucked or ready hands do not), self-collision on; the drive-in s_T → s is checked with that
posture, the path init → s_T with the init posture (3.8 order: s_T first). Step 5 solves the brace approach segment
(pre → contact, 3 poses) exactly so every pose has a witness; the wall is removed from the collision world along the
whole brace chain and the whole push chain (the target only for the push poses); the program check applies the same
ignore sets per waypoint.

### 5.6 IORS large-scale — three stances (ontop, under)

Partner = a large fixed scene object (ontop: bed, sofa, desk, breakfast_table, countertop; under: bed, sofa, desk,
breakfast_table — filtered per target height class by the under-clearance rule below). Roles: `target` (spawned,
library grasps), fixed `partner`, fixed `support` (the init surface), `floor`. Three recorded stances: **A** init pose
and **B** manipulation stance at the init surface, **C** placement stance at the partner.

**Init surface** — an explicit choice of `ground_isg` (it is in the alias), never a runtime fallback. Selection order
(owner decision 2026‑09‑16, A5; minimise floor inits): under — (a) a support object whose near AABB side lies within
`max_partner_geodesic` (3.0 m, eroded map) of the partner's near AABB side → (b) **the partner's own top** ("take it
from the bed and put it under the bed"; alias `…-<partner>-<partner>`, `bindings.support == bindings.partner`) → (c)
**the floor** beside the partner (`support = floor`); ontop — (a) support → (c) floor (the own top would hold at
init). `ground_isg` emits the later classes as fallback aliases (`fallback_of`, `fallback_rank`); the scheduler queues
a fallback only after the primary yields nothing; within an alias the candidates stay in the same class (no runtime
class switching; cross-category init surfaces are separate aliases, VB-ISC). Own top: A/B by the tabletop phases with
the partner as the support (the partner-visibility redraw does not apply), C at the same partner. Per template one
candidate of the grounded class is drawn uniformly; if A–C cannot be built on it, the remaining candidates *of the
same class* are tried in random order; none → reject the template.

**Stance A** — tabletop Phase 1 at the support, target the only object on it, plus the **goal region visible** (owner
decision 2026‑09‑16 10:47, ontop): the partner's top‑face patch around the goal cell and the partner's nearest bottom
edge in frame (≥ 5 %), together with the init surface and the target — this replaces "≥ 4 of the partner's 8 AABB
corners + centre" (which no countertop pair could satisfy). If the lateral draw fails this, redraw the
lateral position (≤ 6), then the next edge, then the next candidate surface. Record `init_stance`.

**Stance B** — tabletop Phase 2 at the support, same edge as A; band from the reach-band script for (support,
target). Record `manipulation_stance`. A → B per 3.7 / 3.8.

**Floor init** (support = floor): anchor = the target's floor cell c_t, drawn uniformly among floor raster cells
(2 cm) that are free floor on the un-eroded map, ≥ 2 cm outside every object's AABB, outside the partner's footprint
(the goal must be false at init) and within geodesic 3 m of the partner's near AABB side. Band: reach-band script with
h = floor z (`--support floor --target C/M --grasp-key K`; the grid adds tilt −1.2; no edge, so the band is the near
strip on the floor in base coordinates). Stance B: base facing c_t at distance ∈ {0.35, 0.45, 0.55} (bearing uniform),
trunk from the floor band (c_t must be a band cell), whole-body collision-free (idle arm untucked), target in frame
(5 %), optical axis within 8° (else 20°) of c_t. Stance A: base on the eroded free map at 0.6–1.5 m from c_t, facing
it (bearing uniform), tilt 0, t* from Phase 1b with h = floor z, s = 0 and d_ws = the distance to c_t; the target at
c_t and the partner (≥ 4 / 8 corners) in frame; collision-free; path A → B exists.

**Stance C** — placement stance (per relation; base square to the partner edge, stand-off {0.35, 0.45, 0.55}, lateral
uniform along the edge minus margins, trunk from the band, uniform draw among survivors in the best angle class):

- ontop: placement top = the partner's largest horizontal face raster (mattress / seat; not the AABB top —
  headboards), stand-off measured from the AABB where the top raster is inset (bed frames). Band from the reach-band
  script for (partner-as-support, target). Edge = the partner edge nearest the target's init pose; focus / **goal
  cell** = the band cell inside the top eroded by the target's half-extent + 0.05 that is nearest to the init pose
  (constant safety margin for every partner candidate); the target AABB at the goal cell in frame (5 %), optical axis
  within 8° (else 20°).
- under: placement top = the floor strip under the partner: floor raster (2 cm) cells inside the partner's footprint
  shrunk by the target's half-extent + 0.05, where an upward ray from the cell hits the partner ≥ target height + 0.03
  above the floor (this is exactly OmniGibson's `Under` test: the ray from the target's AABB-centre column), within
  0.45 m of the partner's near AABB edge. Band from the reach-band script with h = floor z (`--support floor --under
  <partner>`; tilt −1.2 added; the partner's frame is an obstacle, the floor ignored for the grasp pose only). Base
  square to the partner's near AABB edge nearest the init pose; visibility: the near edge's bottom line at the goal
  cell's lateral position ± (target half-extent + 0.10 m) in frame (5 %) and the optical axis within 20° of the goal
  cell (the cell itself is hidden; the earlier "two bottom AABB corners" rule is unsatisfiable for bed / sofa edges of
  ± 1 m at s = 0.45 — 65° off-axis). **Goal cell** = the band
  cell nearest the init pose.

**Visibility per stance (owner decision 2026‑09‑16 10:47; implemented `31afbc9` / `42a7c3b`):** stance A sees the
target on the init surface (tabletop Phase 1) and the partner's *relevant region*: ontop — the partner's top‑face patch
(± 0.10 m) around the goal cell the placement will use and the partner's bottom edge nearest the init pose, both in
frame (5 %); under — the nearest bottom edge (replaces "≥ 4 of the partner's 8 AABB corners and its centre"). Stance B
has no partner rule (its target‑in‑frame / Phase‑2 rules stay; A → B stays a recorded path / transition). Stance C,
ontop: the target at the goal cell in frame (5 %) — **no optical‑axis cone** (the camera is rigid on the torso and the
stance cannot stand back beyond the band; `[pilot]` sofa: 13/13 candidates in frame and unoccluded at 30–31° off
axis, so the 20° cone was the blocker). Under (placement variant): the near‑edge bottom line at the goal lane in frame
and the optical axis within 20° of the hidden goal cell. Push‑under: the drop footprint in frame, optical axis within
45° (the crouched head hangs over the lane).

Record `placement_stance = {stand_off, lateral, translate, tilt, focus / goal_cell, base_pose2d, grasp}`. B → C path
per 3.7 (≤ `max_partner_geodesic` + 1 m) with the object attached; transition per 3.8 (with the partner as init
surface the retreat `s_T` before lowering the trunk toward the floor band is what makes C reachable).

**Coupling B ↔ C — one grasp for pick and place.** The target is carried without re-grasping, so the (yaw, grasp)
drawn at B must also be feasible at C: F_init = the Phase 3 feasible set at B's focus cell (or c_t); F_goal = the
library grasps screened at the goal cell for the target's goal yaw (= its init world yaw) with the 12-seed IK from C
(cached under (trunk, cell)); draw (yaw, grasp) uniformly from F_init ∩ F_goal (empty → next C survivor, then next B
survivor, then reject). Exact IK at B (pre-grasp + grasp, the init surface ignored for the grasp pose only) and at C
(pre-place + place at the goal pose; ontop: the partner's top ignored for the place pose only; under: only the floor
ignored). Record `(yaw, grasp_key, grasp_name)` in both stances.

**Push‑under (large‑scale under; owner decision 2026‑09‑16 06:50 item 7, implemented `25e94d3` … `d7ca530`).** Goal
cell = a footprint‑edge floor cell with the target centre 5–10 cm inside the partner's footprint (0.07; the `under`
predicate needs only the centre column), clear height ≥ target height + 3 cm. The place is a **drop** (`place_kind:
drop`) with the object bottom 10 cm above the floor on a free cell half‑extent + 0.10 m outside the edge (3 cm was
kinematically out of reach), followed by a **straight floor‑level push** (closed gripper, blade behind the target,
stand‑off from the finger profile, approach pitched 0–60° toward the push) from the drop position to the goal cell,
3 waypoints, exact IK from stance C with the floor and the target ignored and the partner present, recorded as
`contacts.push` (`kind: push_under`, `init` / `mid` / `goal`, `direction`, `distance_m`, `push_height_m`, `pitch_deg`,
`blade_standoff_m`, `drop_cell`, `goal_cell`). Stance C: base square to the edge (stand‑off 0.5–0.8 m) or **beside the
lane** facing along the edge with the partner on the right (lane 0.35–0.45 m ahead), floor trunks (translate ≥ 0.75,
tilt ≤ −0.9), the floor mesh removed from the stance / program checks (the ground slab stays); the deepest trunk is
tried first. Program: `… → place(drop) → retreat → push → retreat`; the oracle's push may touch floor and target, never
the partner; `under` is read in the epilogue. `[pilot]`: a floor‑level push is at the edge of the arm's envelope
(shoulder ≥ 0.79 m at the deepest collision‑free crouch, arm 0.79 m); ground‑like objects with footprints > 15 m²
(the garden lawn) no longer block floor cells.

**Objects.** Target at B's focus cell (± 5 cm, footprint on the init surface with 2 cm margin; floor init: at c_t)
with the drawn yaw; no small partner. Goal pose: ontop → target root at the goal cell, init orientation, z = partner
face + bottom offset + 1 cm, feasibility by drop + settle, `OnTop(target, partner)` true; under → target at the goal
cell, init orientation, z = floor + bottom offset + 3 mm, `Under(target, partner)` true. Partner unchanged during the
goal settle (≤ 5 mm). Whole-body collision checks: at A and B with the target at init; at C with the target at its
goal pose. Pose targets measured for the target (0.03 / 0.20); pipeline. Sidecar: `init_stance`,
`manipulation_stance`, `placement_stance`, `goal_cell`, `init_surface` (support name, `partner` or `floor`),
`path_length_m` (A → B, B → C), transitions.

Kept from v1: partner candidate categories, the ≥ 4 / 8 corners rule, `max_partner_geodesic`, the under-clearance rule,
partner unchanged ≤ 5 mm at the goal, tolerances, init surface = largest horizontal face. Replaced: `stand_offs`
[0.45, 0.35, 0.55, 0.65] + `try_all_edges` → any edge / stand-off scan; the "nearest floor point under the bed" goal →
a reachable band cell from a recorded placement stance; the name-sorted candidate order → uniform draw within the
grounded init-surface class.

## 6. Per-task rules

Init = the sampled configuration, goal = what the TRO targets describe. Everything the oracle needs is recorded.

### 6.1 Camera relocalization — standard

- Agent init: base = random free cell of the **eroded** traversability map (task room `living_room_0` of
  `house_single_floor` for now) + yaw ~ U(−π, π); trunk and arms = JoyLo reset joint pos (trunk 0.5, arms zero,
  grippers open).
- Agent goal: base = free cell of the eroded map with a path from init (`get_shortest_path`, eroded map) and geodesic
  length inside the instance's **path bucket** (`generation.path_buckets`, half-open ranges (0, 3], (3, 5], (5, 10] m);
  arms stay as at init unless the selected (translate, tilt) makes the arms self-collide → untucked (the oracle lifts
  the arms before adjusting the torso; recorded as `goal_arms`); trunk (translate, tilt) from the widely used JoyLo
  range (`goal_translate_range` [0, 1.25], `goal_tilt_range` [−0.5, 0.5] `[pilot]`); the camera forward-axis angle
  between init and goal ≤ 60°; goal render: ≥ 3 non-structural objects each ≥ 200 px, ≥ 30 % of them visible from the
  init; both configurations whole-body collision-free (2 cm).
- **Object counting** (kept from v1): an object counts when its segmentation mask has ≥ `min_object_pixels = 200`
  pixels; structural categories never count — `walls`, `floors`, `ceilings`, `window`, `door` frames / leaves,
  `light` / `lamp` fixtures, `room_light`, `downlight`, curtains fixed to the structure — nor does the robot; two
  texture variants of one geometry standing in the room are two objects (they are distinguishable in RGB). The
  shared fraction is computed on that object set.
- TRO: `pose_targets[agent]` = goal camera sensor pose (5 cm / 0.10 rad); sidecar: goal base pose, trunk, arms, path
  (3.7), bucket.

### 6.2 Camera relocalization — obstructed

As 6.1 plus one obstacle (group `obstacles`, size 1 for now; category natural in the scene per the icl): placed on
the eroded shortest path so that the shortest path on the map **with the obstacle's footprint blocked** differs
(`blocked_footprints` recorded like `freed_footprints`, applied by evaluation and the oracle); the **new** path must
stay inside the bucket; the obstacle must be inside the head camera's frame (5 %) at some pose along the new path
**before the path departs from the unobstructed shortest path** — i.e. at one of the waypoints the two paths share
(the divergence waypoint included), with the base heading along the unobstructed path and the JoyLo reset trunk used
at init (the oracle navigates first, then adjusts the torso). Decision 2026‑09‑15 (owner): an obstacle that enters the
frame only after the robot has gone around it (e.g. at the last waypoint, where the base takes the goal yaw and looks
back) is not admissible — a policy could then "know" the detour without ever having seen its cause. With several
obstacles each one is checked against the shortest path before it was placed. The sidecar / summary record the
divergence waypoint index next to the view waypoint index. The obstacle is never touched; it is `ontop floor` in
`:init` and its displacement is checked at the goal settle.

**Obstacle admissibility** (the selection criteria of `runlogs/obstacle_selection_20260912`, now checks): the
category is natural on the floor of the scene (`natural_scenes`), the model stands stably in its canonical +z-up
pose (settled 60 steps, drift ≤ 5 mm; balls, lying and hanging models are out), footprint 0.12–0.80 m and height
0.10–1.70 m (preferred 0.20–0.55 m / 0.20–1.10 m, relative to the 0.33 m chassis half-diagonal and ~1 m
corridors), and it **diverts without closing**: the blocked-map path is ≥ 0.3 m longer than the free path, still
inside the bucket, and the corridor around the obstacle keeps ≥ 0.75 m of free width on one side (the eroded map
must stay connected). The obstacle is not a look-alike of any object the init view or the goal view counts (3.12).

### 6.3 Object scaling and mental rotation (held object)

- Agent init: base = random free cell (eroded map) + yaw ~ U(−π, π); trunk + left arm = JoyLo reset joint pos; right
  arm = the **fixed hold** solved once per robot: pad centre on the head camera's optical axis at depth 0.50 m, trunk
  (0.5, 0), closing axis rolled 15° from vertical `[pilot]` (no depth or trunk fallbacks: the hold is a constant
  written into the customs list as a `hold` constraint). Grasp = a library grasp (floating pipeline) realised as in v1
  §5.2 (finger advance 1 mm/step, 4 mm press), pose written to `grasps[0].object_pose_in_eef_frame`; both init and
  goal need ≥ 40 % of the object visible (self-occlusion render) and the whole object in frame.
- Goal (object scaling): depth d1 on a 5 cm grid over [0.35, 0.85] m with |d1 − d0| ≥ 0.10 m, same orientation; target
  frame = camera sensor, `position_metric = depth`, tolerance 20 % of |Δd| with a 2 cm floor (section 8).
- Goal (mental rotation): rotation about a sampled axis by U(30°, 120°), position unchanged; world orientation target,
  tolerance 0.15 rad, equivalences per model (3.13; symmetric models are not admitted as mental-rotation targets).
- Feasibility: the goal arm configuration is solved with the **right-arm embodiment only** (the oracle moves the arm,
  not the torso); the motion between init and goal (straight line along the camera ray / SLERP about the object) is
  checked at 3 intermediate waypoints (3.9); whole-body collision-free with the object attached at every checked pose.
- Kept from v1 §5.2: the goal options (depth grid × orientation set, or axis × angle set) are precomputed **once per
  template** — they depend on the hold and the object, not on the base pose — and each instance draws from the
  precomputed feasible set; an assisted-grasp attach that fails at the hold for a library row is a rejection of that
  row (`grasp_failed`, the next row is tried), never a fallback to an AABB heuristic; the realised grasp (finger
  joint positions after the press, object pose in the eef frame, the grasp key / name) is what the TRO stores and the
  loader restores (3.14). Self-occlusion is measured with one segmentation render each at init and at the goal with
  everything but the object and the gripper hidden.
- Sidecar: hold constant, grasp, goal arm configuration, waypoints.

### 6.4 Disembedding

Init per section 4 (+ 4.7). Goal: target at the **centre of the destination tray** with its init orientation
(measured, 0.03 / 0.20); no distractor on the destination tray; feasibility by teleport + settle. Sidecar:
`init_stance`, `manipulation_stance`, `placement_stance`, transitions, carry height, grasp with both eef poses, place
eef pose, occlusion ratio, per-object image-space descriptors.

Numbers and records kept from v1 §5.3:

- Source-tray placement raster 1 cm over the tray's **inner floor** (floor polygon minus the wall thickness), 0.5 cm
  footprint margin (3.2); the distractor set's footprints, each grown by its `pad_clearance_m`, may cover at most
  **0.45** of the inner floor (`tray_budget`), so the target stays pickable from above.
- Occlusion: `external_occlusion_ratio = 1 − V / A` from two segmentation renders at the init stance — the complete
  scene, and the same view with the other candidate objects hidden (A = the target's own visible area, so
  self-occlusion is excluded); the target's projection must lie fully inside a 5 % image margin; ratio ≤ 0.30 and
  visible area ≥ `min_visible_pixels = 400`; the destination tray must be visible in one confirmation render (its
  floor ≥ 30 % unoccluded). The TRO keeps the v1 `initial_occlusion` record (`ratio`, `visible_area`,
  `reference_area`, `occluder_ids`, `camera_key`; the `bucket` field is dropped) and the loader re-measures it after
  the deterministic settle (tolerance 0.05 on the ratio).
- Tray semantics: destination membership is `(ontop target destination)` only; `inside` never gates or excludes;
  distractors must satisfy `(not (ontop distractor destination))`.

### 6.5 Configuration matching — block design (goal and cube grasp rule)

**Owner decision 2026‑09‑16 17:10.** First demos run with the goal face up at init (`generation.block_no_flip: true`, no
flip; the yaw‑align stays). **Flip‑on‑the‑table sub‑chain (required end state, prep workstream):** a block whose goal face
is not up is placed on a free cell of the region in the intermediate orientation, re‑grasped with the other pitch class
and placed at its slot — grasp → place(intermediate) → grasp → place, every pose in frame and recorded like any other
grasp / place (two grasp records per flipped block, each its own FOV span, §3.9); the carried‑tilt flip (pick tilt →
place tilt) is no longer the flip mechanism.

*Implementation (`block_design.py`, `0ec0153d`, behind `generation.block_table_flip`, default off):* two quarter turns
about different horizontal axes compose to a 120° body‑diagonal rotation and two about the same axis to 180°, so the only
two‑leg chain through a stable intermediate pose is **yaw first**: leg 1 grasps the block at init, turns it in hand by
ψ + k·90° (k ∈ {1, 3, 2}: an orbit about the vertical, the goal face moves to a side or to the back) and places it on a
free region cell (≥ spacing from every other block's init pose and slot, nearest the block's own cell first); leg 2
re‑grasps it there with the pinch **along** the remaining flip axis and turns it by the one quarter turn about that
horizontal table axis that brings the goal face up (both senses offered to the IK), placing it at the slot. The block's
goal orientation becomes the resulting table‑aligned quaternion (the pattern constrains the up face only; `target_slots`
updated). Each leg passes pre‑pick + pick exact IK, pre‑place + place exact IK with the block attached and the in‑hand
orbit waypoints. Record shape (proposed to the oracle agent): the block record's top‑level `pick / pre_pick / place /
pre_place / waypoints / tilt_*` mirror the **last** leg, `flip_mode: "table"`, `legs: [leg, leg]` in execution order with
per‑leg `face_pair, tilt_pick_deg, tilt_place_deg, yaw_deg, pick, pre_pick, place, pre_place, waypoints, carry_height,
rotation_axis, rotation_deg, block_pose_after`, and `intermediate_pose`; `program_block_design` iterates `legs` when
present (grasp → place(intermediate) → grasp → place). Braces are not attempted after a table flip in this prep
(`brace: none`).


- Goal pattern from `generation.patterns` (or `--variant`); the top-left block in the init image is the anchor: its
  position stays, its goal orientation is the table-aligned orientation with the pattern's top face up reached by the
  **smallest rotation** from its init orientation — yaw-align (|ψ| ≤ 45°) then one flip about a horizontal table axis
  a that brings the goal face up (adjacent face; the init up-face is never the opposite of the goal face, enforced in
  Phase 3); no extra quarter turns. The other blocks are placed relative to the anchor (row first, then column;
  pitch = edge + 1 mm toward the robot's right and toward the robot) with the same smallest-rotation rule per block.
  `layout_reached` gates, `pose_reached` is measured; `assignment.cost = layout`.
- **Cube grasp rule** (the same grasp serves pick and place). The cube grasp set = pinch across one of the 3 face
  pairs × approach along one of the 4 perpendicular faces × tilt about the pinch axis ∈ {0, ±30°, ±45°, ±60°} (pads
  stay flat only for tilts about the pinch axis). The in-hand transform is rigid, so `d_pick = R_a(∓90°)·d_place`
  for a block that flips: pinch axis ∥ a; place approach tilted +θ about a toward the free side of the grid, pick
  approach tilted θ − 90° (θ = 45° gives two symmetric 45° grasps; 30/60 are the alternatives); exactly one tilt sign
  has a downward pick approach. Blocks without a flip use the top-down pinch with the yaw. Sampling screens (pick,
  place) pairs with the 12-seed IK from the manipulation stance with the already-placed blocks (in placement order) in
  the collision world, confirms with the 32-seed IK, and records per block: face pair, tilt pair, ψ, a, sign, place
  order, pick / place eef poses.
- **Brace points**: after each placement the oracle pushes the block left (row) and up (column) with the right hand
  while the closed left gripper braces the already-placed blocks (row-left / column-top): the left-arm brace poses at
  the goal grid (pre-contact + contact) are IK-checked from the manipulation stance and recorded.
- Tolerances: position 0.4 × edge (≈ 1 cm), orientation 0.15 rad (section 8).

### 6.6 Configuration matching — articulation (large-scale, open / close)

Section 5.3. The object may have several viable joints; exactly one is chosen (customs list). Open: goal joint angle
= q_goal ∈ S (or the object's settle angle when partial openings never hold — recorded per object); close: init
angle ∈ S, goal = closed limit. Pose unchanged (scene object).

### 6.7 Configuration matching — articulation (small-scale, open / close)

**Owner decision 2026‑09‑16 17:10.** The manipulation‑stance cone of §4.5 is relaxed for small articulation: the loose
class is 25° / focus‑in‑frame (the same shape as the ontop‑large C rule; the hinged jar's best focus lies 24° off the
optical axis and 25° admits 42 foci over 3 stand‑offs).

*Implementation (`articulation_small.py`, `aea6bd0a`):* `phase2_manipulation_stance(..., angle_classes_deg=(8°, 25°))` with
`generation.phase2_tight_class_deg` / `phase2_loose_class_deg` overriding the two classes per alias; the in‑frame hook stays
the small‑articulation one (target hull + handle contact + at least one near / own‑side brace point in frame), i.e. the
loose class is "focus within 25° of the optical axis and the target in frame", the same shape as the ontop‑large C rule.

*Brace search (`d4546de2` … `6643f1be`):* handle grasp first (pre + grasp), then the brace arm's pre‑contact + contact
with the handle arm locked, and the reverse order when that fails; brace candidates = the base link's four side faces
(vertical / horizontal closing, either sign) plus a pitched 45° press on each face's upper part, ranked by B3's passing
classes → in the left contact band → near / own side; when no pair passes from the manipulation stance, **second
stances** (lateral ±0.10 / ±0.20 m, stand‑off kept or 10 cm closer, trunk kept or (0.5, 0) / (0.25, −0.3); the target must
be a handle‑band cell, in frame from the new camera; ≤ 10 evaluated) — the passing one becomes the recorded manipulation
stance. Per‑attempt budget 300 s. `BRACE_FAILURE` in the run log carries the per‑stance numbers.

*D‑008 (prepared behind `generation.brace_optional`, default off):* when no pair passes anywhere, the instance is
**handle‑only** — `contacts.brace` absent, `stances.manipulation.arms = [handle_arm]`, `summary.brace = "none"`,
`articulation.brace_kind = null`; the oracle's `program_articulation_small` must accept that sidecar (REQUIRED_BLOCKS:
contacts.brace optional, as for touching / axial). Hinged jar numbers (run 0916‑205838): handle grasp alone 1/9 stances,
left‑face brace alone 66/108, two‑arm 0/108; no left contact band for the 0.20 m height class.


Section 5.4. Root pose measured only (0.03 / 0.20); joint goals as 6.6.

### 6.8 Configuration matching — attachment

**Owner decision 2026‑09‑16 17:10 — the brace stays.** When the left brace on the female and the mating pose are not both
reachable from the grasp stance, the sampler records a **second base pose** after the male grasp (lateral trunk move +
torso adjust, both arms' IK verified there, all poses in frame) and the sidecar carries it as a recorded transition
between grasp and attach; the oracle executes that recorded move holding the object — no runtime re‑planning of the
stance (no‑fallback principle). Only when the joint placement has no cell after that is the brace optional (as axial /
block design).

*Implementation (`attachment.py`, `58afae21`):* after the male grasp at the manipulation stance the partner placement,
attach pose, SLERP reorientation and left brace are derived **from the grasp stance first** (A1 joint placement required:
partner cell in the attach‑reach band and its brace face point in the left contact band at that trunk, two‑arm exact IK
with the attach arm locked). Failing that, the **second base pose** search: the grasp stance shifted along the edge by
±0.10 / ±0.20 / ±0.30 m with the trunk kept or moved to (0.5, 0) / (0.25, −0.3), on free floor and whole‑body
collision‑free, at most 6 evaluated; from each the same derivation with the carried male (the lift pose translated with
the base; SLERP waypoints checked from that stance; attach and brace eef poses in frame from that stance's camera). The
sidecar records the stance as `stances[].name = "attach"` (`carried = male`), the base move as `paths[]`
`manipulation → attach` with `carried = male` (local path checked with the male attached) and, when the trunk differs,
`transitions[]` `manipulation → attach` with `holding = male`; the grasp record's `pre_place` / `place` (attach poses) and
`waypoints` belong to the attach stance (`grasps[].extras.attach_stance`). The oracle program (7deabbed) travels
`_travel(sc, "manipulation", "attach", carried=male)` before the reorientation when the `attach` stance exists. Only when
no partner cell satisfies the joint placement from any of these stances is the attach derived brace‑less
(`summary.brace = "none"`, `contacts.brace` absent); a brace‑less sidecar is not yet accepted by `Sidecar.problems()`
(REQUIRED_BLOCKS) — smoke runs use `ISG_PROGRAM_CHECK=stub` for that case only. Per‑attempt budget 360 s.

*Follow‑ups (`bd30c0cf`, `deb0093c`):* brace face candidates for a vertical insertion = the partner's near, robot‑left and
far faces (the near face alone left 22/40 partner cells off the left band); manipulation candidates ordered by the focus
depth (≤ 0.62 m base‑forward first) and the lift's carry height laddered 12 / 8 / 5 cm. *D‑009 (prepared, default off):*
every class‑0 (8°) focus of the lid–teapot alias lies 0.77–0.81 m base‑forward, where the lift / SLERP have no IK —
A `generation.phase2_loose_class_deg` (20–25°) or B `generation.phase2_class1_when_deep_m` (0.65: `phase2_manipulation_
stance(class1_when_deep_m=…)` admits the class‑1 survivors when every class‑0 focus is deeper than that; §4.5 general rule).


Init per section 4 with: partner (female) upright on a cell of the attach-reach band, target (male) at the focus with
the attachment pipeline's library grasps. Goal: partner unchanged; target attached at
`T_partner ∘ T_F(partner) ∘ inv(T_M(target))` (meta-link alignment, verified with `AttachedTo.set_value(True)`; the
pair's collisions filtered as in v1). Checks recorded: pre-attach pose 10 cm back along the **female meta-link's
insertion axis** (not the grasp approach), the attach pose with the target ignored, the in-hand reorientation from the
grasp pose to the attach pose at 3 SLERP waypoints, and the **partner brace**: the other gripper closed on the
partner's face opposite to the insertion direction (pre-contact + contact IK with the target arm locked at the attach
solution; combined whole-body check). Group target: fixed assignment, relation partner → target, tolerances 0.01 /
0.15, symmetry equivalences as in v1.

Kept from v1 §5.6, now explicit: the male gets the `attachable` ability at spawn; the TRO names the exact male /
female meta-link pair (BEHAVIOR serialises only the parent); the collision filter between the two members is applied
before the goal check and restored by the loader at load and at in-episode attachment (3.14); the mating-axis
symmetry test (rotate the child about the female axis in 5 mm-slab steps, accept a rotation when ≥ 90 % of the
slabs match) writes `orientation_equivalences` for **both** group members; a partner whose KB entry is not ready
is not a candidate (2.2). Only the male is grasped; the attachment grasp rows (`__a`) keep every gripper sphere
≥ 1 cm from the male meta-link and the contact off the mating side.

### 6.9 Alignment — world vertical / world horizontal

Section 5.5. Goal: position as at init (hinge), orientation = aligned (upright / level), tolerance 0.08 rad. The
aligned orientation is defined by the target's **reviewed frame** (`generation.front_axis` / `up_axis` / `level_axis`
per model, written by `ground_isg` from the icl; §5.5 "Frame"): upright / level = the reviewed up axis along world +z
with the front axis along the wall normal. An alias without the frame record for its model is rejected
(`missing_frame`), never sampled with a default.

### 6.10 Alignment — axial

Tabletop sections 4.2–4.6 with the axial rules: target yaw = aligned yaw ± U(20°, 70°); a vertical hinge through the
COM (TRO constraint, parent = support) installed after settling, its anchor sunk 1 mm below the top so the object
cannot rock, drift ≤ 1 mm / 0.005 rad over the 60-step check; push contact = the long-axis end inside the band,
generic top-down push pose, arc waypoints at δ/2 and the goal (3.9); **stop edge** for the other arm = the target's
edge (left or right, depending on the spin direction to the goal) that protrudes most along that direction at the
goal orientation, braced with the closed left gripper: pre-contact + contact IK at the manipulation stance, combined
whole-body check; every right-hand push pose IK-feasible; `roles.target.scale` may be configured to make the reach
conditions satisfiable (diagnostic `scale_hint` as in 5.5). Goal: position unchanged, orientation = long axis
parallel to the support's front edge, front toward the robot; tolerance 0.10 rad; equivalences per model (3.13). Screening
uses the left-arm reach volume (push classes at the top height + object thickness) before the exact IK.

**Reference edge visible** (kept from v1 §5.7): the support's front edge is the alignment reference, so a sufficient
run of it must be in the init image — of 9 points sampled uniformly along the front edge over the lateral window
[λ − 0.30, λ + 0.30] m around the target, ≥ 3 are inside the frame (5 % margin) and unoccluded (ray from the camera
to the point, the target itself may not hide them: the points in front of the target are the ones that count), from
the init stance (Phase 1) and from the manipulation stance (Phase 2). Rejection reason `edge_not_visible`; the
template retries with the next lateral position (4.4) before the next edge.

**Push‑only axial (owner decision 2026‑09‑16 08:10).** The stop‑edge brace is optional. The sampler records
`contacts.brace` only when a joint two‑arm placement exists (4.6: work point in the right band AND brace point in the
left contact band at the same trunk, sweep test passed); otherwise the instance is a push‑only axial with the right‑arm
push record alone (`summary.brace: present | none`; the missing brace is not a rejection), and the oracle's push closure
(hold at the goal until the object is still, residual re‑pushed once along the extended arc, back‑off before the
retreat) is the stop. `[pilot]`: on `breakfast_table_rhjoby_0` the two bands overlap only at λ ≈ 0, so axial instances
there are push‑only; the pilot's blocked axial instance succeeded with the closure (final error 0.027 rad).

### 6.11 IORS — ontop / under / inside / nextto / touching (small-scale)

Init per section 4. Goal: partner unchanged; target orientation as at init (inside: not evaluated,
`orientation_metric: null`); position: ontop / under(small) → target centred on the partner (under(small): the
partner ends under the target, BDDL `(under ?partner ?target)`); inside → centred on the partner's fillable-volume
floor (drop + settle; standing orientation allowed for stem-grasped objects); nextto / touching → the position
closest to the init pose that satisfies the predicate (nextto: AABB gap < g_NT along the lateral axis, adjacency rays;
touching: translated toward the partner until rigid contact, partner moved ≤ 5 mm). The place pose (goal pose ∘
T_obj→eef of the drawn grasp) and its pre-place are IK-checked from the manipulation stance. **Touching**: the partner
is on the robot's left; the oracle places the target at the nextto position and pushes it toward the partner with the
right hand while the closed left gripper braces the partner's far (left) side: the push contact on the target's right
face, its direction and distance, and the brace pre-contact + contact (left arm) are IK-checked and recorded.

**Pair admissibility** (`ground_isg`, once per (target, partner) pair; the selection rounds applied these by hand):

- ontop / under(small): the partner's top face admits the target footprint with ≥ 1 cm margin — tier `any_yaw`
  (fits at every yaw; preferred), `aligned` (fits only with the long axes aligned: the goal yaw set is restricted to
  those yaws), `centred` (fits only centred: the goal offset is 0); no tier → `partner_too_small`.
- inside: opening ≥ target footprint + 1 cm at the target's standing or lying orientation, and a **drop test**
  (target released 5 mm above the rim at the `fillable` centroid, settled 120 steps): `Inside` true and the target
  protrudes above the rim by ≤ half its height (`does_not_fit_inside`). Flowers (stem grasps) use the standing
  orientation.
- nextto: both fit side by side inside the reach band with the g_NT gap, and the objects' centre heights overlap
  (the taller one's base below the shorter one's top; `nextto_height_mismatch`) so `NextTo` can be true.
- touching: **partner stability** — base contact ≥ 25 % of its footprint (or a foot ≥ 0.6 × / 0.45 × the body width),
  height ≤ 2.5 × its smallest horizontal extent, no rolling body (`partner_unstable`); **target pushability** —
  smallest horizontal extent ≤ 8.9 cm when it is grasped (pinch width), long side ≤ 30 cm, height ≤ 30 cm, flat base
  (`target_not_pushable`).

Goal numbers kept from v1 §5.10: ontop / under(small) z = partner top + target bottom offset + 1 cm drop; inside drop
from partner rim + 5 mm, xy at the `fillable` meta-link centroid; touching nudges of 1 mm (≤ 25) with physics steps
between them; partner displacement ≤ 5 mm at the goal; a fixed partner gets no `pose_settled` atom.

**Goal predicates `under` / `ontop` (D-007, option A, 2026-09-16).** OmniGibson's `Under` / `OnTop` shoot one vertical ray
from the supporting object's own surface point into the resting object; on a physically correct placement the verdict
flips with sub-millimetre contact (the teacup 1.4 mm from its goal, 2.7° tilt, `under` never true). Both the sampler's
goal check and the task evaluator (`run_isg`, the policy-evaluation path) evaluate these two atoms through
`ispatialgym.sampling.predicates.evaluate_predicate`: OmniGibson's verdict, **or** probe rays across the contact patch
(footprint centre + a ring at half the resting object's footprint radius; each ray starts 3 mm below the supporting
surface found under that point and goes up, any hit on the resting object counts), **or** the geometric branch (resting
AABB bottom within 8 mm of the supporting surface under its footprint centre, footprint centre inside the supporting
top, up axis within 15° of vertical — concave partners whose collision mesh lets rays through). The deciding branch is
recorded per atom (`goal_check.binary_branch` in the instance summary, `meta.binary_branch` in the demo). The release-
height rules above (1 cm / half the well depth above the raycast surface) still hold: the wrapper decides the verdict,
not the placement.

### 6.12 IORS — ontop / under (large-scale)

Section 5.6. Goal: partner unchanged; target orientation as at init; position = the goal cell (nearest band cell to
the init pose inside the eroded placement face, margin = half-extent + 0.05), verified `OnTop` / `Under`.
Under-large partner admissibility: the underside clearance at the goal cell ≥ target height + 0.12 m (gripper
below the wrist) + 0.05 m, checked on the B4 under-clearance map (desks and breakfast tables at 0.70–0.75 m qualify,
coffee tables do not; `under_clearance`), and the target's footprint fits between the legs with 5 cm margin.

## 7. Oracle sidecar (`<iid>.oracle.json`)

```json
{
  "schema": "isg-oracle-v1", "instance_id": "...", "tro_sha256": "...",
  "task_family": "inter_object_relation_satisfaction", "alias": "...", "path_bucket_m": null,
  "stances": {
    "init":         {"base_pose2d": [x, y, yaw], "translate": 0.5, "tilt": 0.0, "arms": "untucked", "stand_off": 0.45, "edge": "e2"},
    "manipulation": {"base_pose2d": [...], "translate": 0.75, "tilt": -0.42, "stand_off": 0.35, "focus": [x, y, z], "arms": ["right"]},
    "placement":    {"...": "IORS large-scale / disembedding only"}
  },
  "transitions": [{"from": "init", "to": "manipulation", "stand_off": 0.55, "order": ["drive-back", "trunk", "drive-in"]}],
  "paths": [{"from": "init", "to": "manipulation", "mode": "local|global", "waypoints": [[x, y, yaw], ...], "length_m": 0.4, "posture": "untucked", "carried": null}],
  "grasps": [{"object": "spatula.n.01_1", "arm": "right", "grasp_key": "spatula__hkvsep__t", "grasp_name": "grasp_1", "yaw": 1.23,
              "pre_grasp": {"position": [], "orientation": []}, "grasp": {"...": ""}, "pre_place": {"...": ""}, "place": {"...": ""}, "carry_height": 0.12}],
  "contacts": {"push": {"arm": "right", "pre": {}, "init": {}, "mid": {}, "goal": {}, "direction": [], "distance_m": 0.06},
               "brace": {"arm": "left", "kind": "below_high", "pre": {}, "contact": {}, "direction": []}},
  "articulation": {"joint": "j_link_0", "q_init": 0.0, "q_goal": 1.2, "waypoints": {"0.25": {"contact": [], "approach": [], "pinch": []}},
                   "handle_reach": {"0.25": true}, "passive_from": 0.75, "follow_mode": "base", "base_follow": {"0.25": [x, y, yaw]}},
  "blocks": [{"instance": "toy_dice.n.01_2", "order": 0, "face_pair": "x", "tilt_pick_deg": -45, "tilt_place_deg": 45, "yaw_deg": 12.0, "flip_axis": [0, 1, 0], "flip_sign": 1, "pick": {}, "place": {}, "brace_row": {}, "brace_col": {}}],
  "goal": {"base_pose2d": [...], "translate": 0.8, "tilt": -0.25, "arms": "same|untucked"},
  "language": {"objects": {"spatula.n.01_1": {"display_name": "spatula", "descriptor": "the spatula on the left"}}, "goal_numbers": {"move_left_cm": 6.0}},
  "waypoint_policy": "checked"
}
```

Only the blocks a task uses are present. `run_isg --mode oracle` requires the sidecar; the evaluator never reads it.

The sidecar carries **facts, not an action list** (decision 2026-09-15): the sequence is the oracle's program for the
alias (`oracle_primitives_design.md` §3.7, one function per row of its §6.2), expanded from these blocks by `run_isg`
and by the sampler's program check (3.5). One schema therefore serves every subtask, and a new subtask adds a program,
not a schema version. Each block is consumed by a named action of the oracle doc §3.6 (`paths` → `Navigate`,
`transitions` → `Transition`, `grasps` → `Grasp` / `Lift` / `Place`, `contacts.push` → `Push`, `contacts.brace` →
`Brace`, `articulation` → `BaseFollow`, `blocks` → the block-design program, `goal` → `SetTorso`, `language` → every
action's annotation).

**TRO `constraints` record (the hinge the loader re-creates, 3.14).** `{"type": "revolute", "child", "parent",
"parent_link", "anchor_position", "axis", "limits": null, "friction", "damping", "drive", "resistance_facts"}`.
`friction` (N·m, Coulomb torque) and `damping` (N·m·s/rad, viscous) are the **hinge resistance** of the wall alignment
hinge (owner 2026‑09‑16 22:05): the hinge sits at the COM, so gravity exerts no restoring torque and a frictionless
joint keeps whatever angular velocity the last contact left (vertical painting demo 0916‑194735: 0.25–0.6 rad/s for
the whole settle window, `not_settled`). The sampler sets both from the target's inertia about the hinge axis
I_n = m (w² + h²) / 12 (mass of the spawned target, canonical width × height perpendicular to the front axis):
`friction = I_n · 10 rad/s²` (`generation.hinge_stop_decel_rad_s2`; pod probe 2026‑09‑16 23:24 on the painting_s070
instance, 2.25 kg, I_n = 0.040 kg·m²: a 0.5 / 1.0 rad/s kick settles in 0.43 / 0.53 s with 0.022 / 0.080 rad of drift —
at 5 rad/s² 0.040 / 0.155 rad; the push works against ≈ 1.6 N at the pad), `damping = I_n / 0.02 s`
(`generation.hinge_viscous_tau_s`: the Coulomb regime reaches down to friction / damping = 0.2 rad/s, then a 20 ms
exponential creep). The loader applies them as a `UsdPhysics.DriveAPI` angular drive on the joint (stiffness 0, target
velocity 0, `damping` per deg/s = damping · π/180, `maxForce` = friction): PhysX's implicit force-limited damped
spring gives torque = clamp(−damping · ω, ±friction) — recorded verbatim in `drive`; `physxJoint:jointFriction` is not
usable (articulation joints only; the hinge is a plain joint). `resistance_facts` = the mass, extents and inertia the
values came from. The axial table hinge (6.10) records neither key: frictionless as before.

## 8. Error tolerances

Gating tolerances follow the perceptual quantity a subtask tests; measured-only tasks report absolute and
size-normalised errors.

| subtask | gating target | tolerance |
|---|---|---|
| camera relocalization | camera pose | 0.05 m / 0.10 rad (fixed) |
| object scaling | depth | max(0.02 m, 0.20·\|Δd\|) |
| mental rotation | orientation | 0.15 rad (+ equivalences) |
| block design | layout relations | position 0.4·edge (≈ 0.010 m), orientation 0.15 rad |
| articulation | joint | max(0.05 rad, 0.10·\|q_goal − q_init\|) revolute; max(0.01 m, 0.10·\|Δq\|) prismatic |
| attachment | layout relation | 0.01 m / 0.15 rad |
| alignment vertical / horizontal | orientation | 0.08 rad |
| alignment axial | orientation | 0.10 rad |
| disembedding, IORS | measured only | 0.03 m / 0.20 rad; override threshold for "goal := settled pose" = max(0.01 m, 0.25 × smallest horizontal extent) / 0.20 rad |

The tolerances are written into the TRO per target and pinned by the evaluation profile `isg-v1` together with the
camera configuration and `n_render_iterations = 1` (`task_definition_and_evaluation.md`, observation timing);
`generation.tolerances` in the customs list overrides per alias, never per instance. Continuous
`orientation_equivalences` are enumerated at 5° (3.13), finer than every orientation tolerance above.

## 9. `ground_isg`, `sample_isg`, `multiply_isg`

**`ground_isg`** (iddl + icl → BDDL + customs list). For each iddl file and each category combination admitted by the
icl (`roles/constraints_compatible.csv`; `--test-only` / `--representative-only` restrict the candidates to the `test = 1` /
`representative = 1` rows of `roles/candidates.csv`): write the BDDL (alias, synsets from the KB, groups),
the customs entry (per-scene whitelists = full icl pools minus the `:require` rejects — `missing_grasps:<pipeline>` /
`missing_contacts` / `missing_handle` in the blacklist —, bindings of the scene roles, `generation` seeded from the
existing entry or `generation_defaults.py`, section 2.3), then test the
combination with **one representative model per category** (`roles/candidates.csv`) on every candidate scene / support / partner:
run the full initialization once (template + instance 0). Success → the template and instance 0 are kept, the
constraints learned are appended; failure → the representative model is moved to that scene's blacklist with the
rejection reason, and the combination is retried with the next representative (icl order) once; a combination with
no successful representative anywhere is written with an empty whitelist and flagged. The alias is never a simple
cartesian product of category lists: only correlated pairs from `roles/constraints_compatible.csv` are grounded.

**`sample_isg`** (BDDL + customs → N templates). N = min(#valid model combinations, `templates_max`), minus one when
`ground_isg` already produced the representative template (its pair is removed from the pool). Model draw: iterate
over all whitelisted models before reusing one (exact duplicates count as one model, texture variants as distinct
models, section 3.12); group members per `generation.groups.<role>` (`size` drawn per template, `draw = unique`
pairwise distinct — the default — or `same_model` for the blocks), unique models across roles within a template;
disembedding distractors from
the pools of the target's category minus the target's look-alike group, at most one model per look-alike group in the
set; look-alike target–partner pairs rejected. Each template: full
initialization → instance 0; failure → the model(s) responsible move to the blacklist (reason recorded); every
success appends constraints (stance, grasp_set, joint_set, placement, contacts) with `source`.

**`multiply_isg`** (template → N instances). N minus the instance 0 that exists; per instance the per-instance parts
of the rules (draws, exact IK, transitions, paths) with the constraints as priors; camera relocalization fills the
path buckets; every success appends constraints; rejections are counted per reason (`summary.rejections`).

## 10. `[pilot]` numbers

x_c(t), z_c(t), camera pitch at tilt 0, hand height below the camera (FK tables); band extents with untucked idle arms
per support height; brace reach of the second arm (blocks, attachment, axial, touching, alignment); base-follow
corridors; the held-object hold (depth 0.50, roll 15°); `goal_translate_range` / `goal_tilt_range` of camera
relocalization; the 45° cube-grasp tilts vs 30/60; `max_partner_geodesic`; the 0.437 disembedding margin.

## 11. Changes from v1 (summary)

Replaced: canonical front-edge stance, `view_prefer_translate`, `view_translates`, `view_arm_fallbacks`, tucked reach
arms (`REACH_ARMS`), `approach_forward` / drive-forward, `scene_distances` × lateral list, arc lattice (4 radii × 7
bearings), right-arm-only, disembedding base-yaw fallback and occluded / unoccluded buckets, "nearest floor point under
the bed", bed-top init fallback, `stand_offs` lists, `init_depth_fallbacks` / `hold_trunks`, per-template BDDL
generation, `<activity>-<def_id>` spec keys.

Kept: scene variants (chairs), support polygon and rest orientations, settle / validation / goal-feasibility pipeline,
IK seeds, grasp library access, hinge constraints, attachment meta-link alignment and collision filter, stable-angle
probe, `Open` semantics, occlusion measurement (as a ≤ 30 % check), visible-object counts, TRO schema `isg-tro-v1`.
