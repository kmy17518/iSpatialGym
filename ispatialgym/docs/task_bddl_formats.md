# BDDL formats for iSpatialGym

Status: implemented for evaluation (2026-09-06); updated 2026-09-15 for the v2 sampling rebuild (grounded BDDL per
category combination, `alignment` activity, oracle sidecar); 20 subtasks since the retirement of world horizontal
(2026-09-23). See `instance_sampling_spec.md` for the rules that
produce instances and `ispatialgym/assets/iddl/README.md` for the role-level problem definitions the BDDL is grounded
from.

This document specifies the BDDL problem files for the seven public iSpatialGym
tasks and the task-specific TRO metadata each one needs. Every task family has
one activity directory; the 20 subtasks are distinguished by an `(:variant ...)`
whose name also carries the grounded object categories (section "Files, problem
names, and variants"). The shipped files under
`ispatialgym/assets/bddl/activity_definitions/` are structural templates with
representative synsets; `ground_isg` writes the concrete problems.

The problem files live in `ispatialgym/assets/bddl/activity_definitions/` and
are parsed, expanded and compiled by `ispatialgym.activity`; the evaluator is
`ispatialgym.tasks.ISpatialGymTask` (see "Evaluation: symbolic and geometric success" below). The
parts of the TRO that only concern instance *restoration* (grasps, hinge
constraints, attachment links, occlusion measurements) are specified here but
not yet applied by the loader; see "Implementation status" at the end.

## Evaluation: symbolic and geometric success

Every task is scored per step (2026-10-14, `ispatialgym.eval.metrics`; reported in `info["done"]["ispatialgym"]`
of `env.step` and summarized by `ispatialgym.eval.episode.summarize_episode`):

1. **Symbolic success** is the BDDL `:goal`: `G(t)`, every goal atom except `pose_settled`, holds. `pose_reached`
   and `layout_reached` atoms compare exactly what the TRO target enables (full pose, the head-camera frame
   position, orientation only, joint positions, or relative poses between group members) within the target's
   `isg-v2` tolerance (`ispatialgym.eval.tolerance_profile`).
2. **Geometric success**, the primary metric, is symbolic success plus every `(:measure ...)` atom within its
   tolerance at the same step(s) (`M(t)`). An atom may be in `:goal`, in `:measure`, or in both.
3. **Settle-eval** decides the step(s): off, the first `t` with `G(t)` (geometric: and `M(t)`); on, the first
   `t` where every `pose_settled` atom holds (`S(t)`: its window of `settle_steps` steps ending at `t` is full
   and within the drift thresholds) and `G` (geometric: `G` and `M`) held at every step of that window.

Both modes are tracked; the run's mode (`run_isg --settle-eval`, on for oracle / replay, off for policy) gives
`task.success`, `official_success` and the episode's termination (its geometric success), the other is reported
as `other_mode`. The errors of every `pose_reached` / `layout_reached` atom are reported each step with their
tolerances; the episode's `errors` / `error_metric` are taken at the run mode's geometric success, or at the last
step without it (see "Common metrics and logging" in `task_definition_and_evaluation.md`).

## Shared conventions

### Files, problem names, and variants

Each public task is one activity directory:

```text
activity_definitions/<activity>/problem0.bddl
activity_definitions/<activity>/problem1.bddl
...
```

- Problem `N` is named `<activity>-N`. BEHAVIOR's knowledge base already
  enumerates every `problemN.bddl` in a directory (`get_instance_count`), and
  `BehaviorTask(activity_definition_id=N)` loads it. `ground_isg` allocates `N`
  per variant, contiguously, and records the mapping in
  `activity_definitions/<activity>/problems.json` (variant ↔ id).
- Every problem declares `(:variant <name>)` directly after `:domain`. The variant
  names the **subtask and the grounded categories**:
  `<family>-<subtask>-<category slots>` (table below), e.g.
  `inter_object_relation_satisfaction-nextto_small_scale-spatula-soup_ladle-breakfast_table`.
  The parser records it as `task_variant`, the TRO's `task_variant` must equal
  it, and results are reported per variant. Group slots list `<category>_<count>`
  tokens (`bottle_of_beer_3_bottle_of_sake_1`); the block design uses the custom
  block token `block_4`. One variant = one BDDL = one category combination;
  templates of that variant differ only by models.
- Every problem ends with a `(:measure ...)` section (after `:goal`) listing the
  `pose_reached` / `layout_reached` atoms that **geometric success** requires
  within tolerance:

  ```lisp
  (:measure
      (pose_reached ?mug.n.04_1)
  )
  ```

  Each atom must name a declared instance or group and needs the same TRO
  target a goal atom would need; it may also appear in the goal (2026-10-14:
  then it is part of the symbolic goal too). Its errors are reported like those
  of a goal atom (`errors`, with `"in_goal"` / `"in_measure"` and its
  `tolerances`), the atoms are listed in `info["done"]["ispatialgym"]["measured"]`,
  and their combination gives the episode's `error_metric` (`poses`, `layout`,
  `pose` or `layout_only`; any other combination is an error at load).
- Stock `parse_problem` prints and ignores unrecognized sections. The
  iSpatialGym parser (`ispatialgym.activity.problem`) implements `:variant`,
  `:groups` and `:measure` explicitly, validates every symbol against
  `:objects` / `:groups` and the domain predicates, and rejects a file whose
  group symbols would otherwise be undefined at scope resolution.
- BEHAVIOR's `sample_b1k_tasks.py` hard-codes `activity_definition_id=0` and
  keys `task_custom_lists.json` (scene, room types, model pools) by activity
  name. iSpatialGym generation reads its own customs list keyed by **variant**
  (see "Task spec (customs list)" in `task_definition_and_evaluation.md`),
  because subtasks in one activity use disjoint object pools and may need
  different scenes.

### Domain

```lisp
(define (domain ispatialgym)
    (:requirements :strips :adl)
    (:predicates
        (open ?obj1)
        (ontop ?obj1 ?obj2)
        (nextto ?obj1 ?obj2)
        (under ?obj1 ?obj2)
        (touching ?obj1 ?obj2)
        (inside ?obj1 ?obj2)
        (attached ?obj1 ?obj2)
        (grasped ?agent ?obj)
        (pose_reached ?obj1)
        (pose_settled ?obj1)
        (layout_reached ?obj1)
    )
)
```

`layout_reached` takes either a declared group (parts are the member objects)
or an articulated object (parts are its links, parametrized by its joints).
The domain file is `bddl3/bddl/activity_definitions/domain_ispatialgym.bddl`;
`layout_reached` is registered as `bddl.predicates.LayoutReached` and mapped to
`omnigibson.object_states.LayoutReached` (articulated objects) or to the group
entity of `ispatialgym.eval.group_entity` (groups). The BEHAVIOR domain is
unchanged.

### Object names

The examples use representative BEHAVIOR synsets such as `mug.n.04` and
`table.n.02`. A concrete activity may substitute another valid synset, but every
declared instance keeps the normal form:

```text
<synset>_<instance number> - <synset>
```

Roles such as "target" and "distractor" are conveyed by which instances appear
in which predicates (and by groups), not by variants such as `target_1`.

Block design uses `toy_dice.n.01` for the custom Kohs block (`toy_dice/isgblk`
in the `ispatialgym-assets` dataset); every declared synset must exist in the
BEHAVIOR knowledge base, so the custom asset is registered under an existing
category and the variant carries the token `block_4` instead.

Group names are derived from the iddl group role: `targets` → `targets_group.n.01_1`
(blocks), `distractors` → `distractors_group.n.01_1`, the attachment assembly
`target partner - assembly` → `assembly_group.n.01_1`, `obstacles` →
`obstacles_group.n.01_1`.

### Groups

```text
(:groups
    MEMBER MEMBER ... - GROUP_NAME
    MEMBER ...        - GROUP_NAME
)
```

- The typed-list shape is the same as `:objects` (`members - name`), so the
  same tokenizer loop parses it into `{group_name: [members]}`. This is why the
  members-first order is recommended over `group - members`.
- Group names follow the instance-name shape (`targets_group.n.01_1`) so they can
  appear wherever an instance can, but they are not synsets: nothing is loaded
  for them, ontology checks skip them, and they must not be declared in
  `:objects`.
- Members must be declared in `:objects`; a member belongs to at most one group;
  groups do not nest. Members may share a synset (blocks) or differ (camera and
  tripod). Whether members are interchangeable is stated in the TRO
  `assignment.mode`, not in BDDL.

Groups are expanded on the parsed nested lists before compilation:

| Literal | Expansion |
| --- | --- |
| `(P g x)` or `(P x g)` for a pointwise predicate (`ontop`, `inside`, `touching`, `nextto`, `under`, `grasped`, `pose_settled`, ...) | conjunction over members |
| `(not (P g x))` | conjunction over members of `(not (P m x))`: no member satisfies `P` |
| `(layout_reached g)`, `(pose_reached g)` | aggregate: evaluated once on the group with one shared member-to-slot assignment |
| a group inside `forall`, `exists`, or `forpairs` | not allowed in v1 |
| a literal with two group arguments | not allowed in v1 |

The negative rule is deliberately not `(not (and ...))`; it is the meaning
needed for "no block is grasped" and "no distractor is in the tray".

At runtime the group symbol is bound to a nonphysical `GroupEntity` in the task
scope; only the aggregate atoms ever reach it.

### Initial and goal conditions

- `inroom` is intentionally absent. Concrete room instances and object bindings
  come from the TRO before scene construction.
- Initial-condition references have no `?` prefix; goal references do.
- Numeric poses, links or sensor keys, tolerances, target slots, relations,
  joint targets, camera metadata, and occlusion measurements never appear in
  BDDL. They come from the TRO.
- Every `pose_settled` atom is a direct child of the top-level goal `and` (the
  parser rejects nested ones). The remaining goal atoms are `G(t)`; the
  `pose_settled` atoms are `S(t)`, which settle-eval on requires (see
  "Evaluation" above).
- `pose_reached`, `pose_settled` and `layout_reached` never appear in `:init`.
- Objects are enumerated explicitly; no wildcards. A different object count is
  a different problem file.

### Released versus held endpoints

`ontop`, `inside`, `attached`, `pose_reached`, `layout_reached`, and
`pose_settled` do not imply that the robot released an object. Tasks whose
answer is a completed scene configuration therefore include:

```lisp
(not (grasped ?agent.n.01_1 ?object))
```

Object Scaling and Mental Rotation are the opposite: their goals require the
object to remain grasped.

## Shared TRO extension

The existing BEHAVIOR TRO JSON stores each task-relevant object's serialized
state under its BDDL instance name and the robot root pose under `robot_poses`.
Keep those fields. Add one reserved top-level `ispatialgym` object:

```json
{
  "mug.n.04_1": "existing BEHAVIOR serialized object state, unchanged",
  "robot_poses": "existing BEHAVIOR robot root pose, unchanged",
  "ispatialgym": {
    "schema_version": "isg-tro-v1",
    "activity": "object_scaling",
    "activity_definition_id": 0,
    "task_variant": null,
    "instance_id": "object_scaling-000001",
    "generation_seed": 1234,
    "evaluation_profile": "isg-v1",
    "bddl_sha256": "SHA256",
    "task_spec_sha256": "SHA256",
    "layout_template_id": "TEMPLATE_ID_OR_NULL",
    "scene": {
      "model": "SCENE_MODEL",
      "floor": 0,
      "template_path": "path/to/materialized_scene_template.json",
      "template_sha256": "SHA256",
      "load_room_instances": ["ROOM_INSTANCE"],
      "required_scene_objects": ["floor.n.01_1"]
    },
    "object_bindings": {
      "mug.n.04_1": {
        "scene_object_name": "SCENE_OBJECT_NAME",
        "category": "mug",
        "model": "MODEL_ID"
      }
    },
    "robot_state": {
      "bddl_instance": "agent.n.01_1",
      "model": "r1pro",
      "joint_positions": {"JOINT_NAME": 0.0},
      "joint_velocities": {"JOINT_NAME": 0.0},
      "controller_targets": {"CONTROLLER_NAME": [0.0]}
    },
    "goal_observation": {
      "sensor_key": "robot_r1:zed_link:Camera:0",
      "rgb_path": "path/to/goal.png",
      "rgb_sha256": "SHA256",
      "resolution": [1080, 1080],
      "intrinsics": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
      "camera_pose_world": {
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0]
      }
    },
    "pose_targets": {}
  }
}
```

The per-task snippets below show only the fields they add inside `ispatialgym`;
the common fields above are always present. Uppercase placeholders and zero
values are explanatory and must be replaced in real files.

### Common rules

- The loader must consume the reserved `ispatialgym` key before the existing
  loop that treats every non-`robot_poses` top-level key as a BDDL object;
  stock `Evaluator.load_task_instance()` would otherwise fail on it.
- `activity_definition_id` equals the problem file number; `task_variant`
  equals the problem's `:variant` (or is `null`).
- `scene` is read before environment construction because partial-room loading
  happens before the rest of the TRO is applied.
- `object_bindings` records the exact materialized model for every physical
  BDDL instance except the agent. It is a validation record, not a sampling
  request. Groups are not listed here.
- `robot_state` is keyed by joint name and covers every controllable joint plus
  controller targets, so restoration does not depend on array order.
- `goal_observation.sensor_key`, `resolution`, and `intrinsics` must equal the
  runtime head-camera configuration. OmniGibson's default is 128x128; the
  BEHAVIOR 2026 eval config (`omnigibson/eval/r1pro.yaml`) uses 1080x1080 with a
  40 mm horizontal aperture on `robot_r1:zed_link:Camera:0`. The evaluation
  profile (`isg-v1`) pins these together with `n_render_iterations: 1`, the
  BEHAVIOR observation convention adopted for demos, goal images and
  evaluation (see "Observation timing" in `task_definition_and_evaluation.md`).
- `goal_observation.rgb_path` is captured with the robot and every task object
  stationary at the goal configuration after at least three renders, so the
  frame is current despite the renderer's two-frame readback latency;
  `camera_pose_world` is the physics-synced sensor prim pose at that moment,
  which is also what `pose_reached` evaluates.
- Quaternions are `[x, y, z, w]`; positions and position tolerances are in
  meters; orientation tolerances and revolute joint values are in radians;
  prismatic joint values are in meters.
- Every `pose_reached` argument, in the goal or in `:measure`, has exactly one
  `pose_targets` entry, or is a group whose `groups` entry has `target_slots`
  and `pose_tolerances`. Every `layout_reached` argument, in the goal or in
  `:measure`, is a group whose `groups` entry has `layout_relations` and
  `layout_tolerances`, or an articulated object listed under
  `articulation_targets`.
- Conversely, every `pose_targets`, `groups` and `articulation_targets` entry
  must correspond to such an atom; a target for an entity that no atom
  mentions is rejected. The BDDL is the single statement of *what* is
  evaluated (symbolic success in the goal, geometric success in `:measure`); the
  TRO supplies the targets. `ispatialgym.eval.tro.validate_tro_against_problem`
  enforces both directions together with activity / definition id / variant
  agreement. This is how Disembedding, Configuration Matching and Inter-object
  Relation Satisfaction judge poses geometrically while their symbolic goal is
  relational.
- The BDDL decides which entities must satisfy `pose_settled`; the TRO does not
  repeat that list, and the settle-eval mode is an evaluator switch.
- The tolerances are the `isg-v2` profile's (2026-10-14, `ispatialgym.eval.tolerance_profile`), computed at load from
  each target's first-step error and written over the TRO's in memory; the TRO's tolerance fields stay (they are
  required) but are superseded at evaluation, and per-subtask overrides live in `isg_customs.json`
  (`evaluation.tolerances`).

### Scalar pose target

```json
{
  "pose_targets": {
    "mug.n.04_1": {
      "link": null,
      "frame": "world",
      "position": [0.0, 0.0, 0.0],
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "position_metric": "full",
      "orientation_metric": "full",
      "position_tolerance": 0.0,
      "orientation_tolerance": 0.0,
      "orientation_equivalences": []
    }
  }
}
```

- `link: null` uses the entity pose API (root link; base footprint for the
  robot). A non-null value must be an exact runtime sensor key or link name of
  the instance and must resolve without fallback. Sensors are evaluated from the
  sensor prim's world pose (synchronized with physics, unlike the render-side
  camera transform).
- `frame` is `"world"`, a BDDL instance (`"table.n.02_1"`: its root pose), or
  `"<instance>::<link or sensor key>"` (`"agent.n.01_1::robot_r1:zed_link:Camera:0"`:
  the head camera). The loader resolves the instance to the bound scene object.
- `position_metric` is `"full"` (3-D position error), `"depth"` (only the
  distance along the frame's optical axis; the frame must be a camera sensor,
  USD convention: optical axis = -Z) or `null` (position not evaluated).
  `orientation_metric` is `"full"` (angular distance) or `null`. At least one
  metric must be enabled; the corresponding `position` / `orientation` and
  tolerance are then required. When a metric field is omitted it defaults to
  `"full"` if the value is given and `null` otherwise.
- `orientation_equivalences` lists object-frame quaternions `e` under which the
  object is visually identical (`q * e`); the evaluator snaps the measured
  orientation to the closest equivalent before computing the error. Empty means
  exact.
- Position error is reported in meters (for `"depth"`: `|depth - goal_depth|`),
  orientation error in radians. The orientation error is frame independent: a
  fixed frame change multiplies both the measured and the goal orientation by
  the same rotation, which leaves their angular distance unchanged, so a
  table-frame and a world-frame orientation target agree as long as the table
  does not move between instance generation and evaluation.

### Group target

Used by `pose_reached(g)` and `layout_reached(g)`:

```json
{
  "groups": {
    "GROUP_NAME": {
      "members": ["INSTANCE_A", "INSTANCE_B"],
      "assignment": {"mode": "exchangeable"},
      "orientation_equivalences": [],
      "target_slots": {
        "SLOT_ID": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
      },
      "pose_tolerances": {"position": 0.0, "orientation": 0.0},
      "layout_relations": [
        {
          "from": "SLOT_ID",
          "to": "OTHER_SLOT_ID",
          "position": [0.0, 0.0, 0.0],
          "orientation": [0.0, 0.0, 0.0, 1.0]
        }
      ],
      "layout_tolerances": {"position": 0.0, "orientation": 0.0}
    }
  }
}
```

- `assignment.mode` is `exchangeable` (one-to-one matching computed at
  evaluation time) or `fixed`, which adds `"member_to_slot": {"INSTANCE": "SLOT_ID"}`.
- For `exchangeable` groups, `assignment.cost` selects the matching cost:
  `"layout"` (default whenever `layout_relations` exist) picks the permutation
  whose relations best reproduce the stored relative poses (total relation
  position error, then orientation error), which is invariant to a rigid motion
  of the whole arrangement; `"slot_position"` picks the minimum total slot
  position error. Ties are broken by member name, then slot id, so the
  assignment is deterministic. `"layout"` is what block design needs once
  `pose_reached` no longer gates success: a correct design displaced or rotated
  on the table must still be matched to the right slots.
- An `exchangeable` group may add `"pinned": {"INSTANCE": "SLOT_ID"}`: members
  held to one slot while the others are matched among the remaining slots. A
  pinned member is a reference of the layout that is not interchangeable with
  the others, such as the puzzle board of block design's board variant: its
  relations to the blocks make `layout_reached` judge the blocks relative to
  the board.
- `pose_reached(g)`: every member's world pose is within `pose_tolerances` of
  its assigned slot (requires `target_slots` and `pose_tolerances`).
- `layout_reached(g)`: for every relation, the pose of the member assigned to
  `to`, expressed in the frame of the member assigned to `from`, is within
  `layout_tolerances` of the stored relative pose. Relations are between the
  parts themselves, so they hold wherever the whole arrangement sits;
  `pose_reached` supplies the absolute placement. `orientation_equivalences`
  are applied to both ends of a relation (`E_from^-1 * T_rel * E_to`, which also
  rotates the relative position), so symmetric blocks are compared modulo their
  symmetry group.
- One assignment is computed per evaluation step, shared by both predicates,
  and logged (`info["done"]["ispatialgym"]["group_assignments"]`).
  `target_slots` has exactly one slot per member and is required whenever
  relations exist (relations reference slot ids), even if only
  `layout_reached` gates success; the slot errors are then reported if
  `(pose_reached g)` is listed in `:measure` (which also requires
  `pose_tolerances`).
- Each relation must equal `T_from^-1 * T_to` of the slot poses (validated on
  load). The generator emits all slot pairs unless the evaluation profile
  selects a subset, so no anchor slot is privileged.
- Reported errors: `pose_reached` -> max member position / orientation error;
  `layout_reached` -> max relation position / orientation error.

## Public task, subtask and variant mapping

| Family (activity dir) | Subtask | Variant template | Scene roles in the variant (icl `kind = scene`) |
| --- | --- | --- | --- |
| `camera_relocalization` | standard | `camera_relocalization-standard` | – |
| `camera_relocalization` | obstructed | `camera_relocalization-obstructed-{obstacles}` | obstacle category (one for now) |
| `object_scaling` | – | `object_scaling-{target}` | – |
| `mental_rotation` | – | `mental_rotation-{target}` | – |
| `disembedding` | – | `disembedding-{target}-{distractors}-{support}` | support category |
| `configuration_matching` | block design | `configuration_matching-block_design-block_4-{support}` | support |
| `configuration_matching` | articulation open, large-scale | `configuration_matching-articulation_open_large_scale-{target}` | target = scene articulated object category |
| `configuration_matching` | articulation close, large-scale | `configuration_matching-articulation_close_large_scale-{target}` | idem |
| `configuration_matching` | articulation open, small-scale | `configuration_matching-articulation_open_small_scale-{target}-{support}` | support (table or floor) |
| `configuration_matching` | articulation close, small-scale | `configuration_matching-articulation_close_small_scale-{target}-{support}` | idem |
| `configuration_matching` | attachment | `configuration_matching-attachment-{target}-{partner}-{support}` | support |
| `alignment` | world vertical | `alignment-world_vertical-{target}-{support}` | support = wall |
| `alignment` | axial | `alignment-axial-{target}-{support}` | support = table |
| `inter_object_relation_satisfaction` | ontop, large-scale | `inter_object_relation_satisfaction-ontop_large_scale-{target}-{partner}-{support}` | partner = scene object; support = table or floor |
| `inter_object_relation_satisfaction` | ontop, small-scale | `inter_object_relation_satisfaction-ontop_small_scale-{target}-{partner}-{support}` | support |
| `inter_object_relation_satisfaction` | under, large-scale | `inter_object_relation_satisfaction-under_large_scale-{target}-{partner}-{support}` | partner = scene object; support = table, floor or the partner |
| `inter_object_relation_satisfaction` | under, small-scale | `inter_object_relation_satisfaction-under_small_scale-{target}-{partner}-{support}` | support |
| `inter_object_relation_satisfaction` | inside / nextto / touching | `inter_object_relation_satisfaction-{inside,nextto,touching}-{target}-{partner}-{support}` | support |

Definition ids `N` are per variant (`problems.json`); the examples below keep the
representative synsets of the structural templates. The former names
`world_alignment` (activity), `table_surface` (variant) and `iors` are retired:
the activity directory is `alignment`, the table variant is `axial`, and the IORS
activity keeps its full name.

## 1. Camera Relocalization

Path: `activity_definitions/camera_relocalization/problem0.bddl`

```lisp
(define (problem camera_relocalization-0)
    (:domain ispatialgym)

    (:objects
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (pose_reached ?agent.n.01_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (pose_reached ?agent.n.01_1)
    )
)
```

Additional TRO specification:

```json
{
  "pose_targets": {
    "agent.n.01_1": {
      "link": "robot_r1:zed_link:Camera:0",
      "frame": "world",
      "position": [0.0, 0.0, 0.0],
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "position_metric": "full",
      "orientation_metric": "full",
      "position_tolerance": 0.0,
      "orientation_tolerance": 0.0
    }
  }
}
```

- Symbolic success: the head-camera pose is within both the position and the
  orientation tolerance of the goal camera pose; geometric success: the same
  atom in `:measure` (the `isg-v2` large group: 0.05 m / 0.15 rad at scale 1).
- The target is the head-camera sensor pose, so `link` is the exact sensor key,
  and it must equal `goal_observation.sensor_key`; `camera_pose_world` must
  equal the target pose.
- The full initial R1Pro configuration in `robot_state` matters here: the same
  camera pose depends on torso joints, not only on `robot_poses`.
- The initial state must not already satisfy `pose_reached`.
- The geodesic path-length bucket of the instance ((0, 3], (3, 5], (5, 10] m) is
  a generation parameter (`generation.path_buckets`) recorded in the oracle
  sidecar, not part of the BDDL or the variant.

### 1b. Obstructed (`camera_relocalization-obstructed-{obstacles}`)

One obstacle object (group `obstacles_group.n.01_1`, one member for now) stands
on the floor on the eroded shortest path from the init to the goal so that the
shortest path changes; it is never touched.

```lisp
(define (problem camera_relocalization-1)
    (:domain ispatialgym)

    (:variant camera_relocalization-obstructed-trash_can_1)

    (:objects
        trash_can.n.01_1 - trash_can.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:groups
        trash_can.n.01_1 - obstacles_group.n.01_1
    )

    (:init
        (ontop obstacles_group.n.01_1 floor.n.01_1)
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (pose_reached ?agent.n.01_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (pose_reached ?agent.n.01_1)
    )
)
```

The TRO is that of the standard problem; the obstacle's footprint is recorded
as `scene_variant.blocked_footprints` (the counterpart of the chair variant's
`freed_footprints`) so that evaluation and the oracle's navigation map block it.

## 2. Object Scaling

Path: `activity_definitions/object_scaling/problem0.bddl`

```lisp
(define (problem object_scaling-0)
    (:domain ispatialgym)

    (:objects
        mug.n.04_1 - mug.n.04
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop agent.n.01_1 floor.n.01_1)
        (grasped agent.n.01_1 mug.n.04_1)
    )

    (:goal
        (and
            (grasped ?agent.n.01_1 ?mug.n.04_1)
            (pose_reached ?mug.n.04_1)
            (pose_settled ?mug.n.04_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (pose_reached ?mug.n.04_1)
    )
)
```

Additional TRO specification:

```json
{
  "grasps": [
    {
      "agent": "agent.n.01_1",
      "arm": "LEFT_OR_RIGHT",
      "object": "mug.n.04_1",
      "object_pose_in_eef_frame": {
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0]
      }
    }
  ],
  "pose_targets": {
    "mug.n.04_1": {
      "link": null,
      "frame": "agent.n.01_1::robot_r1:zed_link:Camera:0",
      "position": [0.0, 0.0, -0.6],
      "orientation": null,
      "position_metric": "depth",
      "orientation_metric": null,
      "position_tolerance": 0.0,
      "orientation_tolerance": null
    }
  }
}
```

- Symbolic and geometric success: the object is still grasped and its root
  position in the head-camera frame matches the goal within
  `position_tolerance`. `position` is the object's root position in the camera
  frame at goal time; the TRO's `position_metric` says `"depth"` (only `-z`, the
  apparent size), and since 2026-10-14 the evaluation compares the **full**
  camera-frame position (the loader sets the metric to `"full"`; micro group:
  max(0.01 m, 10 % of the initial error)). The orientation is not evaluated.
- `object_pose_in_eef_frame` is the object's root-link pose expressed in the
  end-effector link frame of `arm` (`T_eef^-1 * T_obj`). The loader restores
  `robot_state`, places the object at `T_eef * T_rel`, re-establishes the grasp
  according to the robot's `grasping_mode`, and then verifies `grasped`.
- Camera/image data and allowed controls are environment configuration.
  Structurally this BDDL is identical to Mental Rotation; the target metric
  (`depth` here, orientation there) and allowed controls distinguish the tasks.

## 3. Mental Rotation

Path: `activity_definitions/mental_rotation/problem0.bddl`

```lisp
(define (problem mental_rotation-0)
    (:domain ispatialgym)

    (:objects
        mug.n.04_1 - mug.n.04
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop agent.n.01_1 floor.n.01_1)
        (grasped agent.n.01_1 mug.n.04_1)
    )

    (:goal
        (and
            (grasped ?agent.n.01_1 ?mug.n.04_1)
            (pose_reached ?mug.n.04_1)
            (pose_settled ?mug.n.04_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (pose_reached ?mug.n.04_1)
    )
)
```

Additional TRO specification:

```json
{
  "grasps": [
    {
      "agent": "agent.n.01_1",
      "arm": "LEFT_OR_RIGHT",
      "object": "mug.n.04_1",
      "object_pose_in_eef_frame": {
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.0, 0.0, 0.0, 1.0]
      }
    }
  ],
  "pose_targets": {
    "mug.n.04_1": {
      "link": null,
      "frame": "world",
      "position": null,
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "position_metric": null,
      "orientation_metric": "full",
      "position_tolerance": null,
      "orientation_tolerance": 0.0,
      "orientation_equivalences": []
    }
  }
}
```

- Symbolic and geometric success: the object is still grasped and its
  world-frame orientation is within `orientation_tolerance` of the goal (micro
  group: max(0.05 rad, 10 % of the initial error)). The position is not
  evaluated.
- `orientation_equivalences` stays empty when the exact physical orientation is
  required; list the visually equivalent quaternions only if symmetric objects
  are admitted.

## 4. Disembedding

The template uses a low-rim source tray (`tray.n.01_1`) and a distinct
destination tray (`tray.n.01_2`), one target (`mug.n.04_1`), and a distractor
group. A table-only layout would be a second problem file with its own variant.

Path: `activity_definitions/disembedding/problem0.bddl`

```lisp
(define (problem disembedding-0)
    (:domain ispatialgym)

    (:objects
        mug.n.04_1 mug.n.04_2 mug.n.04_3 mug.n.04_4 - mug.n.04
        tray.n.01_1 tray.n.01_2 - tray.n.01
        table.n.02_1 - table.n.02
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:groups
        mug.n.04_2 mug.n.04_3 mug.n.04_4 - distractors_group.n.01_1
    )

    (:init
        (ontop tray.n.01_1 table.n.02_1)
        (ontop tray.n.01_2 table.n.02_1)
        (ontop mug.n.04_1 tray.n.01_1)
        (ontop distractors_group.n.01_1 tray.n.01_1)
        (not (ontop mug.n.04_1 tray.n.01_2))
        (not (ontop distractors_group.n.01_1 tray.n.01_2))
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (ontop ?mug.n.04_1 ?tray.n.01_2)
            (not (ontop ?distractors_group.n.01_1 ?tray.n.01_2))
            (not (grasped ?agent.n.01_1 ?mug.n.04_1))
            (pose_settled ?mug.n.04_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (pose_reached ?mug.n.04_1)
    )
)
```

Additional TRO specification:

```json
{
  "pose_targets": {
    "mug.n.04_1": {
      "link": null,
      "frame": "world",
      "position": [0.0, 0.0, 0.0],
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "position_metric": "full",
      "orientation_metric": "full",
      "position_tolerance": 0.0,
      "orientation_tolerance": 0.0
    }
  },
  "initial_occlusion": {
    "profile": "OCCLUSION_PROFILE_ID",
    "bucket": "OCCLUDED_OR_UNOCCLUDED",
    "sensor_key": "robot_r1:zed_link:Camera:0",
    "target": "mug.n.04_1",
    "reference_visible_pixels": 0,
    "visible_pixels": 0,
    "external_occlusion_ratio": 0.0,
    "occluders": ["mug.n.04_2"]
  }
}
```

- Symbolic success is purely relational: the target is `ontop` the destination
  tray, no distractor is (the group literal expands to one negative `ontop` per
  distractor), and the target is released. The target's exact pictured pose is
  listed in `:measure`: geometric success also needs it within tolerance (small
  group: 0.03 m / 0.10 rad).
- Occlusion is a generation constraint, not a goal. The dataset-level profile
  holds the probability or quota and the acceptance thresholds; the TRO stores
  only the assigned bucket and the realized measurement so the instance can be
  revalidated after reload.

## 5. Configuration Matching

One activity, four problem files, three variants.

### 5a. Block design (`problem0`, variant `block_design`)

```lisp
(define (problem configuration_matching-0)
    (:domain ispatialgym)

    (:variant configuration_matching-block_design-block_4-table)

    (:objects
        toy_dice.n.01_1 toy_dice.n.01_2 toy_dice.n.01_3 toy_dice.n.01_4 - toy_dice.n.01
        table.n.02_1 - table.n.02
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:groups
        toy_dice.n.01_1 toy_dice.n.01_2 toy_dice.n.01_3 toy_dice.n.01_4 - targets_group.n.01_1
    )

    (:init
        (ontop targets_group.n.01_1 table.n.02_1)
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (ontop ?targets_group.n.01_1 ?table.n.02_1)
            (layout_reached ?targets_group.n.01_1)
            (not (grasped ?agent.n.01_1 ?targets_group.n.01_1))
            (pose_settled ?targets_group.n.01_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (pose_reached ?targets_group.n.01_1)
        (layout_reached ?targets_group.n.01_1)
    )
)
```

Additional TRO specification:

```json
{
  "groups": {
    "targets_group.n.01_1": {
      "members": [
        "toy_dice.n.01_1",
        "toy_dice.n.01_2",
        "toy_dice.n.01_3",
        "toy_dice.n.01_4"
      ],
      "assignment": {"mode": "exchangeable", "cost": "layout"},
      "orientation_equivalences": [],
      "target_slots": {
        "slot_0": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        "slot_1": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        "slot_2": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        "slot_3": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
      },
      "pose_tolerances": {"position": 0.0, "orientation": 0.0},
      "layout_relations": [
        {"from": "slot_0", "to": "slot_1", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        {"from": "slot_0", "to": "slot_2", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        {"from": "slot_0", "to": "slot_3", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        {"from": "slot_1", "to": "slot_2", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        {"from": "slot_1", "to": "slot_3", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        {"from": "slot_2", "to": "slot_3", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
      ],
      "layout_tolerances": {"position": 0.0, "orientation": 0.0}
    }
  }
}
```

- Symbolic success: every block is on the table and released, and
  `layout_reached` holds, i.e. the relative arrangement matches (each relation
  is one block's pose in another block's frame). A correctly built design that
  is displaced on the table therefore still succeeds symbolically;
  `target_slots` record the pictured absolute placement, and geometric success
  needs the `(:measure ...)` atoms within tolerance: the relations
  (`layout_reached`, also in the goal; micro group 0.01 m / 0.05 rad) and the
  absolute placement (`pose_reached ?targets_group.n.01_1`; small group). The
  episode's error metric is the layout report's (case `layout`).
- `orientation_equivalences` matters here: an unmarked cube has 24 equivalent
  orientations, and patterned blocks usually keep some. Leaving it empty makes
  the orientation check exact (the Kohs block has six distinct faces).
- The goal orientation of every block is the table-aligned orientation with the
  pattern's face up that is reached by the **smallest rotation** from its init
  orientation (yaw-align, then one flip about a horizontal table axis); the
  anchor is the block that is top-left in the init image and keeps its position
  (`instance_sampling_spec.md` §6.5).
- `(ontop ?targets_group.n.01_1 ?table.n.02_1)` restricts the template to a
  single layer. For stacked designs, drop it and let the layout relations carry
  the block-on-block structure.

### 5b. Articulation (four subtasks: open / close × large-scale / small-scale)

Large-scale: the target is a **scene** articulated object (cabinet, fridge,
oven, dishwasher, drawer, door) bound per scene; nothing is spawned. Small-scale:
the target is a spawned articulated object (toolbox, carton, hinged jar) that
stands on a support (a table, or the floor for objects taller than 0.6 m); its
BDDL adds `(ontop target support)` to `:init` and `:goal` and measures the root
pose (`(:measure (pose_reached ?target))`). Only the `open` literals differ
between the open and close directions. Exactly one joint of the object is the
task joint; every other non-fixed joint is listed at its init value.

```lisp
(define (problem configuration_matching-1)
    (:domain ispatialgym)

    (:variant configuration_matching-articulation_open_large_scale-cabinet)

    (:objects
        cabinet.n.01_1 - cabinet.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (not (open cabinet.n.01_1))
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (open ?cabinet.n.01_1)
            (not (grasped ?agent.n.01_1 ?cabinet.n.01_1))
            (pose_settled ?cabinet.n.01_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (layout_reached ?cabinet.n.01_1)
    )
)
```

```lisp
(define (problem configuration_matching-2)
    (:domain ispatialgym)

    (:variant configuration_matching-articulation_close_large_scale-cabinet)

    (:objects
        cabinet.n.01_1 - cabinet.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (open cabinet.n.01_1)
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (not (open ?cabinet.n.01_1))
            (not (grasped ?agent.n.01_1 ?cabinet.n.01_1))
            (pose_settled ?cabinet.n.01_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (layout_reached ?cabinet.n.01_1)
    )
)
```

Additional TRO specification (both problems):

```json
{
  "articulation_targets": {
    "cabinet.n.01_1": {
      "joints": {
        "JOINT_NAME": {"position": 0.0, "tolerance": 0.0}
      }
    }
  }
}
```

- Symbolic success: the `open` guard holds in the requested direction
  (BEHAVIOR's `open`) and the cabinet is released; geometric success: every
  listed joint is within tolerance of its target (`layout_reached` in
  `:measure`; micro group per joint: max(0.05 rad, 10 % of the joint's initial
  error), prismatic max(0.01 m, 10 %)).
- `layout_reached(cabinet)` is the object's internal configuration: each
  moving link's pose relative to the base link, which the joint positions
  parametrize exactly. The evaluator therefore compares joint positions, one
  target and tolerance per joint, and all must hold. This handles any number of
  independently moving links, with a fixed part-to-target mapping because links
  are not interchangeable.
- Every non-fixed joint of the object must be listed, stationary ones at their
  initial value, so an unlisted drawer or door cannot drift unnoticed (the
  loader rejects a target that omits one).
- Reported errors: revolute joints contribute to the layout orientation error
  (radians), prismatic joints to the layout position error (meters); the
  maximum over joints is reported.
- A movable articulated object (laptop, box) may additionally carry a
  `pose_targets` entry for its root placement, licensed by a
  `(pose_reached ?obj)` atom in `:measure` (the small-scale variant, small
  group; error metric case `layout`). The fixed-base cabinet here has none
  (case `layout_only`).
- `open` is kept as the semantic guard. BEHAVIOR's `Open` only checks that some
  relevant joint has crossed a small threshold, so it cannot express the exact
  pictured configuration by itself.
- The initial articulation is not in `articulation_targets`: it is the standard
  top-level BEHAVIOR object state for `cabinet.n.01_1`, whose `joint_pos` and
  `joint_vel` fields `load_state` restores. For `problem2` that initial state is
  the per-instance sampled opening; for `problem1` it is the closed
  configuration.
- In the closing direction the targets are the canonical closed joint limits,
  derived from the same openable-direction metadata that `Open` uses, so they
  are identical for every instance of that object. `layout_reached` still
  matters there: `(not (open ...))` accepts anything below `Open`'s 5% of joint
  range (about 4.5 degrees on a 90-degree door) and only for annotated joints,
  whereas the joint targets enforce closure within the profile tolerance and
  supply the reported joint error.

### 5c. Attachment (`problem3`, variant `attachment`)

```lisp
(define (problem configuration_matching-3)
    (:domain ispatialgym)

    (:variant configuration_matching-attachment-digital_camera-camera_tripod-table)

    (:objects
        digital_camera.n.01_1 - digital_camera.n.01
        camera_tripod.n.01_1 - camera_tripod.n.01
        table.n.02_1 - table.n.02
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:groups
        digital_camera.n.01_1 camera_tripod.n.01_1 - assembly_group.n.01_1
    )

    (:init
        (ontop assembly_group.n.01_1 table.n.02_1)
        (not (attached digital_camera.n.01_1 camera_tripod.n.01_1))
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (attached ?digital_camera.n.01_1 ?camera_tripod.n.01_1)
            (ontop ?camera_tripod.n.01_1 ?table.n.02_1)
            (layout_reached ?assembly_group.n.01_1)
            (not (grasped ?agent.n.01_1 ?assembly_group.n.01_1))
            (pose_settled ?assembly_group.n.01_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (pose_reached ?assembly_group.n.01_1)
        (layout_reached ?assembly_group.n.01_1)
    )
)
```

Additional TRO specification:

```json
{
  "groups": {
    "assembly_group.n.01_1": {
      "members": ["digital_camera.n.01_1", "camera_tripod.n.01_1"],
      "assignment": {
        "mode": "fixed",
        "member_to_slot": {
          "digital_camera.n.01_1": "camera",
          "camera_tripod.n.01_1": "tripod"
        }
      },
      "orientation_equivalences": [],
      "target_slots": {
        "camera": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
        "tripod": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
      },
      "pose_tolerances": {"position": 0.0, "orientation": 0.0},
      "layout_relations": [
        {"from": "tripod", "to": "camera", "position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
      ],
      "layout_tolerances": {"position": 0.0, "orientation": 0.0}
    }
  },
  "attachment_targets": [
    {
      "child": "digital_camera.n.01_1",
      "parent": "camera_tripod.n.01_1",
      "child_link": "MALE_ATTACHMENT_LINK",
      "parent_link": "FEMALE_ATTACHMENT_LINK",
      "joint_type": "fixed"
    }
  ]
}
```

- Symbolic success: the camera is `attached` to the tripod, the tripod stands
  on the table, `layout_reached` holds (the camera pose in the tripod frame
  matches the pictured one), and both parts are released. `target_slots` record
  where the assembled pair stood in the goal image; geometric success needs the
  `:measure` atoms within tolerance: the relation again and both world poses.
- `attached` verifies a real BEHAVIOR attachment joint; `attachment_targets`
  pins which meta-link pair must be used, since BEHAVIOR serializes only the
  parent object. Replace the pair only with assets that expose compatible
  male/female attachment meta links.
- An assembly with more parts lists every directed `attached` edge, one
  attachment target per edge, and all parts in the group.

## 6. Alignment (activity directory `alignment`, formerly `world_alignment`)

One activity, two subtasks (the world-horizontal variant, formerly definition 2 /
variant `alignment-world_horizontal-{target}-{support}`, was retired on 2026-09-23).

### 6a. Axial (`alignment-axial-{target}-{support}`, formerly `table_surface`)

```lisp
(define (problem alignment-0)
    (:domain ispatialgym)

    (:variant alignment-axial-fishing_rod-table)

    (:objects
        fishing_rod.n.01_1 - fishing_rod.n.01
        table.n.02_1 - table.n.02
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop fishing_rod.n.01_1 table.n.02_1)
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (ontop ?fishing_rod.n.01_1 ?table.n.02_1)
            (pose_reached ?fishing_rod.n.01_1)
            (not (grasped ?agent.n.01_1 ?fishing_rod.n.01_1))
            (pose_settled ?fishing_rod.n.01_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (pose_reached ?fishing_rod.n.01_1)
    )
)
```

Additional TRO specification:

```json
{
  "pose_targets": {
    "fishing_rod.n.01_1": {
      "link": null,
      "frame": "world",
      "position": null,
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "position_metric": null,
      "orientation_metric": "full",
      "position_tolerance": null,
      "orientation_tolerance": 0.0,
      "orientation_equivalences": []
    }
  }
}
```

- Symbolic and geometric success: the rod is on the table, released, and its
  orientation is within `orientation_tolerance` of the goal (micro group); where
  it lies on the table is not evaluated. Because the orientation error is frame independent (see "Scalar
  pose target"), a world-frame goal and a table-frame goal are the same check
  for a static table.
- `fishing_rod.n.01` is representative; the chosen model needs an unambiguous
  canonical frame and must be manipulable by R1Pro. List
  `orientation_equivalences` (e.g. the 180-degree flip of a plain rod) if the
  model is symmetric.

### 6b. World vertical (`alignment-world_vertical-{target}-{support}`)

A painting hangs on a wall through a revolute joint whose axis passes through
the painting's center of mass; the goal is the straight (world-vertical) pose.

```lisp
(define (problem alignment-1)
    (:domain ispatialgym)

    (:variant alignment-world_vertical-painting-wall)

    (:objects
        painting.n.01_1 - painting.n.01
        wall.n.01_1 - wall.n.01
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (pose_reached ?painting.n.01_1)
            (not (grasped ?agent.n.01_1 ?painting.n.01_1))
            (pose_settled ?painting.n.01_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (pose_reached ?painting.n.01_1)
    )
)
```

Additional TRO specification:

```json
{
  "constraints": [
    {
      "type": "revolute",
      "child": "painting.n.01_1",
      "parent": "wall.n.01_1",
      "parent_link": "WALL_LINK",
      "anchor_position": [0.0, 0.0, 0.0],
      "axis": [0.0, 0.0, 0.0],
      "limits": null
    }
  ],
  "pose_targets": {
    "painting.n.01_1": {
      "link": null,
      "frame": "world",
      "position": null,
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "position_metric": null,
      "orientation_metric": "full",
      "position_tolerance": null,
      "orientation_tolerance": 0.0
    }
  }
}
```

- Symbolic and geometric success: the painting is released and its
  orientation is within `orientation_tolerance` of the straight pose (micro
  group). The position is fixed by the hinge and is not evaluated.
- The hinge is a TRO constraint, not a BDDL literal. `attached` is deliberately
  not used: BEHAVIOR attachment requires male/female meta links and moves the
  child so its meta link coincides with the parent's, which is not the
  center-of-mass pivot this task needs.
- `anchor_position` is the pivot in world coordinates (the painting's center of
  mass at generation time); `axis` is the unit hinge axis in world coordinates
  (the wall normal). The loader creates the joint between `parent_link` and the
  painting root after both are placed, using the same joint-creation utility as
  `AttachedTo`, and recreates it after every reset because it is not part of
  BEHAVIOR's serialized object state.
- `wall.n.01_1` is declared so the constraint parent has a scope binding.
  `(touching painting.n.01_1 wall.n.01_1)` may be added to `:init` and `:goal`
  only if the hinge guarantees contact.
- The painting has one degree of freedom, so the full world-frame target
  effectively evaluates the in-plane angle; no axis-only predicate is needed.

## 7. Inter-object Relation Satisfaction

One activity, seven subtasks: `ontop_large_scale`, `ontop_small_scale`,
`under_large_scale`, `under_small_scale`, `inside`, `nextto`, `touching`. BDDL
cannot parameterize a predicate name, so each relation is its own problem; the
large-scale variants bind the partner to a fixed scene object (bed, sofa, desk,
breakfast table, countertop) and the target's init support is a table, the floor,
or (under only) the partner itself.

```lisp
(define (problem inter_object_relation_satisfaction-3)
    (:domain ispatialgym)

    (:variant inter_object_relation_satisfaction-nextto_small_scale-apple-bowl-table)

    (:objects
        apple.n.01_1 - apple.n.01
        bowl.n.01_1 - bowl.n.01
        table.n.02_1 - table.n.02
        floor.n.01_1 - floor.n.01
        agent.n.01_1 - agent.n.01
    )

    (:init
        (ontop apple.n.01_1 table.n.02_1)
        (ontop bowl.n.01_1 table.n.02_1)
        (ontop agent.n.01_1 floor.n.01_1)
    )

    (:goal
        (and
            (nextto ?apple.n.01_1 ?bowl.n.01_1)
            (not (grasped ?agent.n.01_1 ?apple.n.01_1))
            (not (grasped ?agent.n.01_1 ?bowl.n.01_1))
            (pose_settled ?apple.n.01_1)
            (pose_settled ?bowl.n.01_1)
            (pose_settled ?agent.n.01_1)
        )
    )

    (:measure
        (pose_reached ?apple.n.01_1)
        (pose_reached ?bowl.n.01_1)
    )
)
```

| Subtask | Goal relation | Init support literals |
| --- | --- | --- |
| `ontop_large_scale` | `(ontop ?target ?partner)` | `(ontop target support)`, `(not (ontop target partner))`; partner fixed |
| `ontop_small_scale` | `(ontop ?target ?partner)` | both `ontop support` |
| `under_large_scale` | `(under ?target ?partner)` | `(ontop target support)`, `(not (under target partner))`; partner fixed; support may be the partner |
| `under_small_scale` | **`(under ?partner ?target)`** | both `ontop support`, `(not (under partner target))` |
| `inside` | `(inside ?target ?partner)` | both `ontop support`, `(not (inside target partner))` |
| `nextto` | `(nextto ?target ?partner)` | both `ontop support`, `(not (nextto target partner))` |
| `touching` | `(touching ?target ?partner)` | both `ontop support`, `(not (touching target partner))` |

The argument order of the two `under` subtasks differs on purpose: in the
small-scale version the manipulated target (glass, teacup, fries, bottle) is set
down on the partner (coaster, saucer, platter, tray), so the *partner* ends under
the target. Fixed scene partners have no `pose_settled` atom (kinematic prims
report no velocity) and no `pose_reached` measurement.

Additional TRO specification (all seven; the partner entry only for spawned partners):

```json
{
  "pose_targets": {
    "apple.n.01_1": {
      "link": null,
      "frame": "world",
      "position": [0.0, 0.0, 0.0],
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "position_metric": "full",
      "orientation_metric": "full",
      "position_tolerance": 0.0,
      "orientation_tolerance": 0.0
    },
    "bowl.n.01_1": {
      "link": null,
      "frame": "world",
      "position": [0.0, 0.0, 0.0],
      "orientation": [0.0, 0.0, 0.0, 1.0],
      "position_metric": "full",
      "orientation_metric": "full",
      "position_tolerance": 0.0,
      "orientation_tolerance": 0.0
    }
  }
}
```

- Symbolic success: the relation predicate holds and both objects are released.
  The two `:measure` atoms and their pose targets record the exact pictured
  configuration: geometric success needs both within tolerance (small group for
  the small-scale tasks, large for the large-scale ones); the error metric is
  the maximum over both objects (case `poses`). The loader checks that
  `task_variant` names the relation in the goal.
- `inside` sets `orientation_metric: null` on the target's pose target (the
  target's goal orientation is whatever settles in the container; this admits
  flowers stood in vases).
- The apple/bowl pair is representative. Each problem must use synsets that can
  physically realize its relation, and initial support literals may differ
  (an `under` reference may need a raised object).
- `open` is excluded; articulation belongs to Configuration Matching.

## BDDL content deliberately omitted

Part of the TRO, environment configuration, generation configuration, or
evaluator, never BDDL:

- Scene model, floor, partial-room load scope, and object-model ids.
- Initial numeric object and robot state, grasps, attachments, and hinge
  constraints.
- Goal images and camera intrinsics/extrinsics.
- Pose targets, evaluated links or sensors, tolerances, target slots, layout
  relations, joint targets, and the member-to-slot assignment.
- Disembedding occlusion profile, bucket, and measurements.
- Allowed robot actuator groups, time limits, split metadata, seeds, and
  oracle-only information (stances, contacts, grasps, waypoints, paths: the
  oracle sidecar `<instance_id>.oracle.json`, `instance_sampling_spec.md` §7).
- The settle-eval mode and its window (`settle_steps`), and the tolerance scale.

## Implementation status

Implemented (`ispatialgym/activity`, `ispatialgym/eval`, `ispatialgym/tasks`,
plus the narrow Behavior-1K extensions `bddl.predicates.LayoutReached`,
`domain_ispatialgym.bddl`, `omnigibson.object_states.LayoutReached`, and the
`depth` / orientation-only / equivalence modes of
`omnigibson.object_states.PoseReached`):

1. Parser with `:variant`, `:groups` and the `:measure` section (an atom may also be in the goal), group
   expansion per the table above, validation of symbols, predicates, arities,
   `pose_settled` placement and the `:init` exclusions; `task_variant` recorded
   from the variant.
2. `isg-tro-v1` validation of the evaluation targets: schema version, ids,
   variant / definition agreement, array lengths, unit quaternions, positive
   tolerances, metric / tolerance consistency, one slot per member, fixed
   assignment bijections, relation consistency with slots, and a one-to-one
   correspondence between targets and the `pose_reached` / `layout_reached`
   atoms of the goal and of `:measure`.
3. Nonphysical group scope entities with the shared member-to-slot assignment
   (`layout` or `slot_position` cost), group `pose_reached` / `layout_reached`,
   articulated `layout_reached`, and camera-frame depth targets.
4. `ISpatialGymTask`: explicit object bindings (no sampling), target
   configuration from the TRO, symbolic / geometric success under both settle-eval modes (per-target errors with
   tolerances, the error metric, snapshots at each first success), BEHAVIOR-compatible `task.success` /
   `goal_status`, and `load_tro_instance` (object states, robot root pose and joint positions, settle, snapshot);
   the `isg-v2` tolerance profile (`ispatialgym.eval.tolerance_profile`).
5. Tests: `ispatialgym/tests/test_problem_parsing.py` and
   `test_geometry_and_targets.py` (no simulator; every shipped problem file),
   `test_sim_evaluator.py` (headless OmniGibson: pose metrics on a mug and on
   the R1Pro head camera, articulated cabinet, dice group, and the
   `world_alignment-0` episode flow).

Not yet implemented (instance restoration / generation):

1. Pre-reading `ispatialgym.scene` before environment construction (the loader
   currently takes the environment config and the object bindings explicitly).
2. Grasp restoration from `grasps`, attachment-link restoration from
   `attachment_targets`, hinge restoration from `constraints` (recreated after
   every reset), and `robot_state.controller_targets`; `load_tro_instance`
   logs a warning when these records are present.
3. Recomputing and validating Disembedding's `initial_occlusion` after settling.
4. The v2 generation pipeline (`ground_isg` / `sample_isg` / `multiply_isg`,
   `instance_sampling_spec.md`) with the 20 subtasks, grounded BDDL per variant,
   `problems.json`, the `alignment` activity rename and the oracle sidecar.
5. `initial_occlusion`, `goal_observation` and hash fields are carried through
   but not checked beyond their presence.
