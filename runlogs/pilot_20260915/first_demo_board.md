# First-demo board (2026-09-16 07:26 PDT)

**1 / 21 subtasks with a valid demo.**

state ∈ {no instance, instance, demo run, valid demo}; a demo is valid only when its code commit contains every fix-log entry affecting the subtask (earlier successes are `pre-fix`).

| subtask | state | blocking item | owner | fixing commits | last verdict | artefact (meta.json) | last change |
|---|---|---|---|---|---|---|---|
| camera_relocalization-standard | demo run (pre-fix success) | - | - | 7a340e4 |  | runlogs/pilot_20260915/demos/camera_relocalization-standard__house_single_floor__0000/…/meta.json | 2026-09-16 03:00 PDT |
| camera_relocalization-obstructed | valid demo | tracker corner fix d4a1745 not yet re-demoed | oracle | 7a340e4, d4a1745 | success | runlogs/pilot_20260915/first_demo/camera_relocalization-obstructed/0916-065800/demo/camera_relocalization-obstructed-footstool_1__house_single_floor__obstacles-footstool_1__0000/meta.json | 2026-09-16 07:23 PDT |
| object_scaling | demo run (pre-fix success) | - | - | 7a340e4 |  | runlogs/pilot_20260915/demos/object_scaling-*/…/meta.json | 2026-09-16 03:00 PDT |
| mental_rotation | demo run (pre-fix success) | - | - | 7a340e4 |  | runlogs/pilot_20260915/demos/mental_rotation-*/…/meta.json | 2026-09-16 03:00 PDT |
| disembedding | no instance | 4a distractor budget (D1), 4b tray 1.25 (D1+tabletop), 9 band keying (bands) | D1 / tabletop / bands | 2cec470, 5552ee8, 7a340e4, 9744b34, d4a1745 |  |  |  |
| configuration_matching-block_design | no instance | pick spacing 10.1 cm smoke; left band + A1 placement | tabletop | 4fd4c89, 5552ee8, 7a340e4, d4a1745 |  |  |  |
| configuration_matching-attachment | no instance | left band + A1 placement (skateboard edge_too_short = yield) | tabletop | 4fd4c89, 5552ee8, 7a340e4, d4a1745 |  |  |  |
| configuration_matching-articulation_open_small_scale | no instance | brace band empty (§5.4) → left contact band wiring; floor_support_todo | tabletop | 7a340e4, fcf7bfc |  |  |  |
| configuration_matching-articulation_close_small_scale | no instance | same as open | tabletop | 7a340e4, fcf7bfc |  |  |  |
| configuration_matching-articulation_open_large_scale | no instance | item 6 roomy picks (A3 + D1), A4 cabinet class landed | A3 / D1 / large-scale | 6abf9da, 7a340e4, c161a04 |  |  |  |
| configuration_matching-articulation_close_large_scale | no instance | same as open | A3 / D1 / large-scale | 6abf9da, 7a340e4, c161a04 |  |  |  |
| alignment-world_vertical | no instance | program_check brace ignore set (core), hanger blacklist (D1) | core / D1 | 4eb66f5, 7a340e4 |  |  |  |
| alignment-world_horizontal | no instance | same as vertical | core / D1 | 4eb66f5, 7a340e4 |  |  |  |
| alignment-axial | no instance | two-arm stop at the robot's limit (decision), blade stand-off landed | tabletop / owner | 27b27c1, 4fd4c89, 7a340e4, d4a1745, e4a1b5c |  |  |  |
| inter_object_relation_satisfaction-ontop_small_scale | demo run (pre-fix success) | - | - | 5552ee8, 7a340e4, d4a1745 |  | runlogs/pilot_20260915/demos/inter_object_relation_satisfaction-ontop_small_scale-apple-plate-…__0000/…/meta.json | 2026-09-16 04:37 PDT |
| inter_object_relation_satisfaction-under_small_scale | no instance | re-run after e4a1b5c (stalled approach finish) | oracle | 5552ee8, 7a340e4, d4a1745, e4a1b5c |  |  |  |
| inter_object_relation_satisfaction-inside | no instance | re-sample after 160c518 (release height) + drop executor | core / oracle | 160c518, 5552ee8, 7a340e4, d4a1745, e4a1b5c |  |  |  |
| inter_object_relation_satisfaction-nextto | demo run (pre-fix success) | near-side goal 160c518 (future sampling) | - | 160c518, 5552ee8, 7a340e4, d4a1745 |  | runlogs/pilot_20260915/demos/inter_object_relation_satisfaction-nextto-jar_of_spaghetti_sauce-…__0000/…/meta.json | 2026-09-16 04:37 PDT |
| inter_object_relation_satisfaction-touching | no instance | left band + joint gate (core landed), blade stand-off | core / tabletop | 160c518, 7a340e4, d4a1745, e4a1b5c |  |  |  |
| inter_object_relation_satisfaction-ontop_large_scale | no instance | T2/T3 supports or floor class (new machinery) | large-scale | 7a340e4 |  |  |  |
| inter_object_relation_satisfaction-under_large_scale | no instance | item 7 push-under (sampler + program + oracle) | large-scale / oracle | 7a340e4, 8140025 |  |  |  |

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
| 2026-09-16 07:09 PDT | `2cec470` | disembedding | source tray at roles.source.scale (1.25) placed stance-relative (inner-floor centre on the focus cell, edge-aligned); band key tray/gsxbym@1.25:focus:edge; budget from generation.distractor_budget | tabletop |
| 2026-09-16 07:09 PDT | `fcf7bfc` | configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale | brace on a base-link face from the left contact band (handle-only workspace band; B3 brace bands empty) | tabletop |
| 2026-09-16 07:09 PDT | `4fd4c89` | configuration_matching-block_design | region spacing 10.1 cm (open fingers vs neighbouring cubes); blades from the finger profile; anchor on a left-band cell | tabletop |
| 2026-09-16 07:09 PDT | `27b27c1` | alignment-axial | stop blade from the finger profile on the unswept part of the stop face, brace checked against the goal-posed target | tabletop |
| 2026-09-16 07:26 PDT | `4eb66f5` | alignment-world_vertical, alignment-world_horizontal | hanger agrpio/nadesv blacklisted, next-model fallback | D1 |
| 2026-09-16 07:26 PDT | `c161a04` | configuration_matching-articulation_open_large_scale, configuration_matching-articulation_close_large_scale | re-grounded on the roomy A3 picks (rvunhj_0, bamfsz_0 drawer, fridge jtqazu_1, top cabinet tynnnw_1, oven amblrk_0) | D1+A3 |
| 2026-09-16 07:26 PDT | `8140025` | inter_object_relation_satisfaction-under_large_scale | push-under sampler: goal cell at the footprint edge, drop outside, straight floor push record | large-scale |
| 2026-09-16 07:26 PDT | `5552ee8` | inter_object_relation_satisfaction-under_small_scale, inter_object_relation_satisfaction-ontop_small_scale, inter_object_relation_satisfaction-nextto, inter_object_relation_satisfaction-inside, disembedding, configuration_matching-attachment, configuration_matching-block_design | finishing-motion excursion bound also for witness goals (pre_grasp/grasp witnesses in different IK branches) | oracle |
