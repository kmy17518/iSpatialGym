# Status 2026-09-16 21:52 PDT

**10 / 21 subtasks with a valid demo.** balance $42.14   fleet spend/h $12.884   spent so far $669.13 of cap $1000   recharges=29

## Runs in flight

| pod | name | role | run | since | ETA |
|---|---|---|---|---|---|
| 1c88jsy46ubecf | pilot2-a40-6 | pool | run ? () | agent run |  |
| h4enpm19l0t3ng | smoke2-a40-0 | pool | run ? () | agent run |  |
| 5gm563m9wgnrwc | pilot2-l40s-10 | pool | run alignment-world_vertical-painting_s070-walls (alignment-world_vertical) | 21:49 PDT | ~23:04 |
| 29s7u374iifjbj | pilot2-a40-10 | pool | run configuration_matching-articulation_close_small_scale-hinged_jar-breakfast_table (CM articulation_close_small_scale) | 21:42 PDT | ~22:57 |
| pjf6451ahyqxll | smoke2-a40-1 | pool | run configuration_matching-articulation_open_large_scale-dishwasher (CM articulation_open_large_scale) | 21:34 PDT | ~22:49 |
| bpu6l9bffwaf1y | pilot2-l40s-11 | pool | run configuration_matching-articulation_open_small_scale-hinged_jar-breakfast_table (CM articulation_open_small_scale) | 20:58 PDT | ~22:13 |
| oliefq3ymuexna | pilot2-a40-1 | pool | run configuration_matching-attachment-lid-teapot-breakfast_table (CM attachment) | 20:58 PDT | ~22:13 |
| 6pdrunfn78iofy | pilot2-a40-4 | pool | run configuration_matching-block_design-block_4-breakfast_table (CM block_design) | 21:42 PDT | ~22:57 |
| bl91tuqop45ila | pilot2-l40s-2 | pool | run inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table (IORS touching) | 21:49 PDT | ~23:04 |
| s7m4h2ay2slau4 | pilot2-l40s-4 | pool | run inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed (IORS under_large_scale) | 21:13 PDT | ~22:28 |
| local GPU | 4090 | local | run inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-countertop-floor | | |
| uilwpybjxnekg0, m48tuedy95bu05, rzcqsre0b89792, 8lp72savyk8co4, 1qzna64k3s9t8s, z2qdfyfyrgcjys, xnam1nh3y2706e, cy0f5t6i58i08v | | band | 8 band job(s) running (pool 13 (1 free), bands 8 (0 idle), inflight 9, launched 2, ready 0, band remaining 2 / unassigned 0) | | |

## Idle pods: 3 pool, band 0

reason: ready queue empty for the pool: 1 waiting for a fix (alignment-world_horizontal); 1 valid; 9 in flight; coverage paused

## Queued runs: 0


## Board

| subtask | state | blocking | owner |
|---|---|---|---|
| camera_relocalization-standard | valid demo (pre-166005f) | - | - |
| camera_relocalization-obstructed | valid demo (pre-166005f) | tracker corner fix d4a1745 not yet re-demoed | oracle |
| object_scaling | valid demo (pre-166005f) | - | - |
| mental_rotation | valid demo (pre-166005f) | - | - |
| disembedding | valid demo (pre-bcc937d,166005f,de4e723) | - | - |
| CM block_design | no instance | D-006 default A applied 21:17 (block_gap_m 0.006, 8c2a2bc) — loop runs next tick; flip-on-table prep 0ec0153d (sampler) + 0ee927f6 (program) | loop / tabletop |
| CM attachment | no instance | program travels the recorded attach stance (6baaaa3), PROGRAM_BLOCKS brace optional; re-run 0916-205838 in flight | loop |
| CM articulation_open_small_scale | no instance | 25° cone landed (aea6bd0a): Phase 2 survives; now contact_unreachable at the brace pre-contact (no brace face + handle grasp passes the two- | tabletop |
| CM articulation_close_small_scale | no instance | same as open | tabletop |
| CM articulation_open_large_scale | no instance | all five A3 units handle_unreachable x8 (fridge, dishwasher, oven, top_cabinet, no_top); Rs_int drawer running; large-scale: object-inclusiv | large-scale |
| CM articulation_close_large_scale | no instance | same as open; Rs_int drawer close = Phase-1 no_stance (base_ok instrumented 1c829ae) | large-scale |
| alignment-world_vertical | demo run | oracle chain landed (standoff, 3-D arc about the hook, loaded bounds): demo runs to the end in tolerance (0.052 rad) but not_settled — the f | large-scale |
| alignment-world_horizontal | demo run | push plan_failed = self-collision left_realsense_link vs right_arm_link7 (13 mm) along every path: the sidecar's push witnesses have the bra | large-scale |
| alignment-axial | valid demo | re-run 20:40 (oracle retreat changes 35be0ee/3eb8e51): plan_failed at retreat → primary oracle | oracle |
| IORS ontop_small_scale | valid demo (pre-de4e723) | - | - |
| IORS under_small_scale | valid demo (pre-de4e723) | D-007 A landed (core 5555526: robust under/ontop predicate, branch logged) — re-run queued | loop |
| IORS inside | valid demo (pre-166005f,de4e723) | valid on f5ae607; re-opened by the core's targeted 1ba5e74 (open pads clear of the support: the failing sprout grasp had its fingertip at th | loop |
| IORS nextto | valid demo | closure ramp + pre-shape (bcc937d): 0.386 → 0.038 rad; the release still flings the pencil (blocked lower pad snaps open in the retreat) → o | oracle-B |
| IORS touching | demo run | push spans the measured gap at the place + tolerance margin (core a6d7123); pre-grasp joint jump / grasp plan → primary oracle; re-run queue | loop / oracle |
| IORS ontop_large_scale | no instance | countertop pair = geometric yield (partner behind stance A) → floor fallback alias queued (local); sofa-bar re-run with the B→C path limit g | large-scale / loop |
| IORS under_large_scale | no instance | push-under lanes 0.45–0.6 m ahead + edge-end laterals + frontal offset 0.25 m right (7162fe7, 8c93d0b, 2cfce47) — re-run queued/running | large-scale / loop |

## Open decisions: none

