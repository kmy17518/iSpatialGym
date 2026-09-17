# Status 2026-09-16 19:52 PDT

**6 / 21 subtasks with a valid demo.** balance $32.47   fleet spend/h $20.992   spent so far $632.31 of cap $1000   recharges=27

## Runs in flight

| pod | name | role | run | since | ETA |
|---|---|---|---|---|---|
| 5gm563m9wgnrwc | pilot2-l40s-10 | pool | run ? () | agent run |  |
| xnam1nh3y2706e | smoke-l40s-3 | pool | run ? () | agent run |  |
| 6pdrunfn78iofy | pilot2-a40-4 | pool | run ? () | agent run |  |
| m48tuedy95bu05 | smoke2-l40s-0 | pool | demo-only alignment-world_vertical-painting_s070-walls (alignment-world_vertical) | agent run |  |
| bl91tuqop45ila | pilot2-l40s-2 | pool | run configuration_matching-articulation_close_large_scale-bottom_cabinet (CM articulation_close_large_scale) | 19:51 PDT | ~21:06 |
| htvfrubebp6tnj | pilot2-a40-3 | pool | run configuration_matching-articulation_open_large_scale-dishwasher (CM articulation_open_large_scale) | 19:17 PDT | ~20:32 |
| 1c88jsy46ubecf | pilot2-a40-6 | pool | run configuration_matching-attachment-lid-teapot-breakfast_table (CM attachment) | 19:17 PDT | ~20:32 |
| pjf6451ahyqxll | smoke2-a40-1 | pool | run configuration_matching-block_design-block_4-breakfast_table (CM block_design) | 19:07 PDT | ~20:22 |
| 29s7u374iifjbj | pilot2-a40-10 | pool | run inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table (IORS inside) | 19:26 PDT | ~20:41 |
| oliefq3ymuexna | pilot2-a40-1 | pool | run inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table (IORS nextto) | 19:26 PDT | ~20:41 |
| gvua02xnwm4ti6 | pilot2-a40-9 | pool | run inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table (IORS ontop_small_scale) | 19:26 PDT | ~20:41 |
| h4enpm19l0t3ng | smoke2-a40-0 | pool | run inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed (IORS under_large_scale) | 19:07 PDT | ~20:22 |
| uilwpybjxnekg0 | | band | 1 band job(s) running (pool 26 (15 free), bands 1 (0 idle), inflight 9, launched 1, ready 0, band remaining 0 / unassigned 0) | | |

## Idle pods: 14 pool, band 0

reason: ready queue empty for the pool: 2 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale); 4 waiting for a fix (CM articulation_close_large_scale, alignment-world_vertical, alignment-world_horizontal, IORS touching); 6 valid; 8 in flight; coverage paused
holds: CM articulation_open_small_scale: owner 17:10 item 3 decided (25 deg cone) - tabletop implementing; lifts on its targeted fix-log entr; CM articulation_close_small_scale: same as open (25 deg cone)

## Queued runs: 0


## Board

| subtask | state | blocking | owner |
|---|---|---|---|
| camera_relocalization-standard | valid demo | - | - |
| camera_relocalization-obstructed | valid demo | tracker corner fix d4a1745 not yet re-demoed | oracle |
| object_scaling | valid demo | - | - |
| mental_rotation | valid demo | - | - |
| disembedding | valid demo | - | - |
| CM block_design | no instance | no-flip run 19:03: target_placement x8 → tabletop adding the place pre-pose diagnostic | tabletop |
| CM attachment | no instance | 19:06: program_check_failed x8 on the second-base-pose sidecar (transitions[].holding) → oracle program / program_check must accept it (tabl | oracle / tabletop |
| CM articulation_open_small_scale | no instance | owner 17:10 (3): Phase-2 cone 25° / focus-in-frame for small articulation (tabletop, 20 min) → hinged_jar loop | tabletop |
| CM articulation_close_small_scale | no instance | same as open (25° cone) | tabletop |
| CM articulation_open_large_scale | no instance | all five A3 units handle_unreachable x8 (fridge, dishwasher, oven, top_cabinet, no_top); Rs_int drawer running; large-scale: object-inclusiv | large-scale |
| CM articulation_close_large_scale | no instance | same as open; Rs_int drawer close = Phase-1 no_stance (base_ok instrumented 1c829ae) | large-scale |
| alignment-world_vertical | demo run | oracle re-runs on 4b2bb60 / 9345c8a: push [execution] again (17:06) → oracle | oracle |
| alignment-world_horizontal | demo run | oracle re-run on 4b2bb60: push plan_failed again (16:56) → oracle | oracle |
| alignment-axial | valid demo | - | - |
| IORS ontop_small_scale | demo run (pre-fix success) | valid on 6b856ed 18:59; re-opened by the core's 02c700d (two-tier grasp draw) — re-run scheduled (D-005 asks whether blanket fixes should ke | core / loop |
| IORS under_small_scale | demo run (pre-fix success) | valid on 6b856ed 19:10 (release surface 5851c8b); re-opened by 02c700d — re-run scheduled (D-005) | core / loop |
| IORS inside | demo run (pre-fix success) | 18:53 failed_grasp at grasp (sprout spins out of the pinch?) → core diagnosing with the new grasp facts; 02c700d re-run scheduled | core / loop |
| IORS nextto | demo run (pre-fix success) | valid on 6b856ed 19:06 (full placed_orientation gate passed); re-opened by 02c700d — re-run scheduled (D-005); FOV rule (b) monitor still pe | core / loop |
| IORS touching | no instance | core 6de8803: brace optional (push chain required) — napkins loop launched 18:12; oracle program_touching must accept the brace-less sidecar | loop / oracle |
| IORS ontop_large_scale | no instance | countertop pair = geometric yield (partner behind stance A) → floor fallback alias queued (local); sofa-bar re-run with the B→C path limit g | large-scale / loop |
| IORS under_large_scale | no instance | push-under lanes 0.45–0.6 m ahead + edge-end laterals + frontal offset 0.25 m right (7162fe7, 8c93d0b, 2cfce47) — re-run queued/running | large-scale / loop |

## Open decisions: none

