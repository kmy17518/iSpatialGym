# Status 2026-09-16 21:22 PDT

**7 / 21 subtasks with a valid demo.** balance $25.97   fleet spend/h $20.992   spent so far $662.06 of cap $1000   recharges=28

## Runs in flight

| pod | name | role | run | since | ETA |
|---|---|---|---|---|---|
| pjf6451ahyqxll | smoke2-a40-1 | pool | run ? () | agent run |  |
| h4enpm19l0t3ng | smoke2-a40-0 | pool | run ? () | agent run |  |
| 1c88jsy46ubecf | pilot2-a40-6 | pool | run ? () | agent run |  |
| gvua02xnwm4ti6 | pilot2-a40-9 | pool | demo-only alignment-axial-board_game-breakfast_table (alignment-axial) | agent run |  |
| xnam1nh3y2706e | smoke-l40s-3 | pool | run alignment-world_horizontal-license_plate-walls (alignment-world_horizontal) | 21:20 PDT | ~22:35 |
| 29s7u374iifjbj | pilot2-a40-10 | pool | demo-only alignment-world_vertical-painting_s070-walls (alignment-world_vertical) | 20:58 PDT | ~22:13 |
| htvfrubebp6tnj | pilot2-a40-3 | pool | run configuration_matching-articulation_open_large_scale-dishwasher (CM articulation_open_large_scale) | 20:24 PDT | ~21:39 |
| bpu6l9bffwaf1y | pilot2-l40s-11 | pool | run configuration_matching-articulation_open_small_scale-hinged_jar-breakfast_table (CM articulation_open_small_scale) | 20:58 PDT | ~22:13 |
| oliefq3ymuexna | pilot2-a40-1 | pool | run configuration_matching-attachment-lid-teapot-breakfast_table (CM attachment) | 20:58 PDT | ~22:13 |
| 6pdrunfn78iofy | pilot2-a40-4 | pool | run configuration_matching-block_design-block_4-breakfast_table (CM block_design) | 21:20 PDT | ~22:35 |
| 5gm563m9wgnrwc | pilot2-l40s-10 | pool | run inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table (IORS nextto) | 21:20 PDT | ~22:35 |
| cy0f5t6i58i08v | pilot2-l40s-3 | pool | demo-only inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table (IORS touching) | agent run |  |
| s7m4h2ay2slau4 | pilot2-l40s-4 | pool | run inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed (IORS under_large_scale) | 21:13 PDT | ~22:28 |
| local GPU | 4090 | local | run inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-countertop-floor | | |
| uilwpybjxnekg0 | | band | 1 band job(s) running (pool 26 (9 free), bands 1 (0 idle), inflight 10, launched 2, ready 0, band remaining 0 / unassigned 0) | | |

## Idle pods: 13 pool, band 0

reason: ready queue empty for the pool: 2 waiting for a fix (CM block_design, IORS under_small_scale); 0 valid; 10 in flight; coverage paused

## Queued runs: 0


## Board

| subtask | state | blocking | owner |
|---|---|---|---|
| camera_relocalization-standard | valid demo (pre-166005f) | - | - |
| camera_relocalization-obstructed | valid demo (pre-166005f) | tracker corner fix d4a1745 not yet re-demoed | oracle |
| object_scaling | valid demo (pre-166005f) | - | - |
| mental_rotation | valid demo (pre-166005f) | - | - |
| disembedding | valid demo (pre-bcc937d,166005f) | - | - |
| CM block_design | no instance | D-006 default A applied 21:17 (block_gap_m 0.006, 8c2a2bc) — loop runs next tick; flip-on-table prep 0ec0153d (sampler) + 0ee927f6 (program) | loop / tabletop |
| CM attachment | no instance | program travels the recorded attach stance (6baaaa3), PROGRAM_BLOCKS brace optional; re-run 0916-205838 in flight | loop |
| CM articulation_open_small_scale | no instance | 25° cone landed (aea6bd0a): Phase 2 survives; now contact_unreachable at the brace pre-contact (no brace face + handle grasp passes the two- | tabletop |
| CM articulation_close_small_scale | no instance | same as open | tabletop |
| CM articulation_open_large_scale | no instance | all five A3 units handle_unreachable x8 (fridge, dishwasher, oven, top_cabinet, no_top); Rs_int drawer running; large-scale: object-inclusiv | large-scale |
| CM articulation_close_large_scale | no instance | same as open; Rs_int drawer close = Phase-1 no_stance (base_ok instrumented 1c829ae) | large-scale |
| alignment-world_vertical | demo run | oracle chain landed (standoff, 3-D arc about the hook, loaded bounds): demo runs to the end in tolerance (0.052 rad) but not_settled — the f | large-scale |
| alignment-world_horizontal | demo run | push plan_failed = self-collision left_realsense_link vs right_arm_link7 (13 mm) along every path: the sidecar's push witnesses have the bra | large-scale |
| alignment-axial | demo run (pre-fix success) | re-run 20:40 (oracle retreat changes 35be0ee/3eb8e51): plan_failed at retreat → primary oracle | oracle |
| IORS ontop_small_scale | valid demo (pre-166005f) | - | - |
| IORS under_small_scale | demo run (pre-fix success) | D-007 default A applied 21:17: robust Under evaluator → core; teacup placements are physically correct (1.4 mm, 2.7°) | core |
| IORS inside | valid demo (pre-166005f) | valid on f5ae607; re-opened by the core's targeted 1ba5e74 (open pads clear of the support: the failing sprout grasp had its fingertip at th | loop |
| IORS nextto | demo run (pre-fix success) | closure ramp + pre-shape (bcc937d): 0.386 → 0.038 rad; the release still flings the pencil (blocked lower pad snaps open in the retreat) → o | oracle-B |
| IORS touching | demo run | first two instances written (push-only program 0a81f739); 0916-202403 executed the whole chain, not_reached: the recorded 18 mm push left th | core / oracle |
| IORS ontop_large_scale | no instance | countertop pair = geometric yield (partner behind stance A) → floor fallback alias queued (local); sofa-bar re-run with the B→C path limit g | large-scale / loop |
| IORS under_large_scale | no instance | push-under lanes 0.45–0.6 m ahead + edge-end laterals + frontal offset 0.25 m right (7162fe7, 8c93d0b, 2cfce47) — re-run queued/running | large-scale / loop |

## Open decisions: none

