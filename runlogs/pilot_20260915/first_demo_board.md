# First-demo board (2026-09-16 06:57 PDT)

**0 / 21 subtasks with a valid demo.**

state ∈ {no instance, instance, demo run, valid demo}; a demo is valid only when its code commit contains every fix-log entry affecting the subtask (earlier successes are `pre-fix`).

| subtask | state | blocking item | owner | fixing commits | last verdict | artefact (meta.json) | last change |
|---|---|---|---|---|---|---|---|
| camera_relocalization-standard | demo run (pre-fix success) | - | - | 7a340e4 |  | runlogs/pilot_20260915/demos/camera_relocalization-standard__house_single_floor__0000/…/meta.json | 2026-09-16 03:00 PDT |
| camera_relocalization-obstructed | demo run (pre-fix success) | tracker corner fix d4a1745 not yet re-demoed | oracle | 7a340e4, d4a1745 |  | runlogs/runpod/pilot/*/demo__v2__camera_relocalization-obstructed-*/…/meta.json | 2026-09-16 01:40 PDT |
| object_scaling | demo run (pre-fix success) | - | - | 7a340e4 |  | runlogs/pilot_20260915/demos/object_scaling-*/…/meta.json | 2026-09-16 03:00 PDT |
| mental_rotation | demo run (pre-fix success) | - | - | 7a340e4 |  | runlogs/pilot_20260915/demos/mental_rotation-*/…/meta.json | 2026-09-16 03:00 PDT |
| disembedding | no instance | 4a distractor budget (D1), 4b tray 1.25 (D1+tabletop), 9 band keying (bands) | D1 / tabletop / bands | 7a340e4, 9744b34, d4a1745 |  |  |  |
| configuration_matching-block_design | no instance | pick spacing 10.1 cm smoke; left band + A1 placement | tabletop | 4fd4c89, 7a340e4, d4a1745 |  |  |  |
| configuration_matching-attachment | no instance | left band + A1 placement (skateboard edge_too_short = yield) | tabletop | 4fd4c89, 7a340e4, d4a1745 |  |  |  |
| configuration_matching-articulation_open_small_scale | no instance | brace band empty (§5.4) → left contact band wiring; floor_support_todo | tabletop | 7a340e4 |  |  |  |
| configuration_matching-articulation_close_small_scale | no instance | same as open | tabletop | 7a340e4 |  |  |  |
| configuration_matching-articulation_open_large_scale | no instance | item 6 roomy picks (A3 + D1), A4 cabinet class landed | A3 / D1 / large-scale | 6abf9da, 7a340e4 |  |  |  |
| configuration_matching-articulation_close_large_scale | no instance | same as open | A3 / D1 / large-scale | 6abf9da, 7a340e4 |  |  |  |
| alignment-world_vertical | no instance | program_check brace ignore set (core), hanger blacklist (D1) | core / D1 | 7a340e4 |  |  |  |
| alignment-world_horizontal | no instance | same as vertical | core / D1 | 7a340e4 |  |  |  |
| alignment-axial | no instance | two-arm stop at the robot's limit (decision), blade stand-off landed | tabletop / owner | 4fd4c89, 7a340e4, d4a1745, e4a1b5c |  |  |  |
| inter_object_relation_satisfaction-ontop_small_scale | demo run (pre-fix success) | - | - | 7a340e4, d4a1745 |  | runlogs/pilot_20260915/demos/inter_object_relation_satisfaction-ontop_small_scale-apple-plate-…__0000/…/meta.json | 2026-09-16 04:37 PDT |
| inter_object_relation_satisfaction-under_small_scale | no instance | re-run after e4a1b5c (stalled approach finish) | oracle | 7a340e4, d4a1745, e4a1b5c |  |  |  |
| inter_object_relation_satisfaction-inside | no instance | re-sample after 160c518 (release height) + drop executor | core / oracle | 160c518, 7a340e4, d4a1745, e4a1b5c |  |  |  |
| inter_object_relation_satisfaction-nextto | demo run (pre-fix success) | near-side goal 160c518 (future sampling) | - | 160c518, 7a340e4, d4a1745 |  | runlogs/pilot_20260915/demos/inter_object_relation_satisfaction-nextto-jar_of_spaghetti_sauce-…__0000/…/meta.json | 2026-09-16 04:37 PDT |
| inter_object_relation_satisfaction-touching | no instance | left band + joint gate (core landed), blade stand-off | core / tabletop | 160c518, 7a340e4, d4a1745, e4a1b5c |  |  |  |
| inter_object_relation_satisfaction-ontop_large_scale | no instance | T2/T3 supports or floor class (new machinery) | large-scale | 7a340e4 |  |  |  |
| inter_object_relation_satisfaction-under_large_scale | no instance | item 7 push-under (sampler + program + oracle) | large-scale / oracle | 7a340e4 |  |  |  |

## Fix log

| time | commit | subtasks | what | owner |
|---|---|---|---|---|
| 2026-09-16 06:57 PDT | `d4a1745` | camera_relocalization-obstructed, inter_object_relation_satisfaction-ontop_small_scale, inter_object_relation_satisfaction-nextto, inter_object_relation_satisfaction-inside, inter_object_relation_satisfaction-under_small_scale, alignment-axial, disembedding, configuration_matching-block_design, configuration_matching-attachment, inter_object_relation_satisfaction-touching | QA gripper_out_of_fov arms on first frame entry; base tracker corner-aware (25-50 cm -> 1 cm) | oracle |
| 2026-09-16 06:57 PDT | `e4a1b5c` | inter_object_relation_satisfaction-inside, inter_object_relation_satisfaction-under_small_scale, alignment-axial, inter_object_relation_satisfaction-touching | epilogue rotation only with an orientation target; push closure; stalled approach finished by arm_to | oracle |
| 2026-09-16 06:57 PDT | `160c518` | inter_object_relation_satisfaction-nextto, inter_object_relation_satisfaction-inside, inter_object_relation_satisfaction-touching | nextto near-side goal; inside release 2 cm above the rim (drop) | core |
| 2026-09-16 06:57 PDT | `7a340e4` | * | sim_utils: parked objects keep gravity off; local_hull PhysX->Fabric sync | core |
| 2026-09-16 06:57 PDT | `9744b34` | disembedding | tray measured off a re-parked body, rest check, edges per lateral run | disembedding |
| 2026-09-16 06:57 PDT | `4fd4c89` | alignment-axial, configuration_matching-block_design, configuration_matching-attachment | A1 joint two-arm placement, finger-profile blade stand-offs, brace vs target at goal | tabletop |
| 2026-09-16 06:57 PDT | `6abf9da` | configuration_matching-articulation_open_large_scale, configuration_matching-articulation_close_large_scale | cuRobo mesh cache 8192 | coordinator |
