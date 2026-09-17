# Status 2026-09-16 20:22 PDT

**7 / 21 subtasks with a valid demo.** balance $46.96   fleet spend/h $20.992   spent so far $641.07 of cap $1000   recharges=28

## Runs in flight

| pod | name | role | run | since | ETA |
|---|---|---|---|---|---|
| m48tuedy95bu05 | smoke2-l40s-0 | pool | run ? () | agent run |  |
| 6pdrunfn78iofy | pilot2-a40-4 | pool | run configuration_matching-articulation_close_large_scale-bottom_cabinet (CM articulation_close_large_scale) | 20:17 PDT | ~21:32 |
| 1c88jsy46ubecf | pilot2-a40-6 | pool | run configuration_matching-attachment-lid-teapot-breakfast_table (CM attachment) | 19:17 PDT | ~20:32 |
| pjf6451ahyqxll | smoke2-a40-1 | pool | run configuration_matching-block_design-block_4-breakfast_table (CM block_design) | 19:07 PDT | ~20:22 |
| 5gm563m9wgnrwc | pilot2-l40s-10 | pool | run inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table (IORS nextto) | 20:17 PDT | ~21:32 |
| oliefq3ymuexna | pilot2-a40-1 | pool | demo-only inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table (IORS nextto) | 20:17 PDT | ~21:32 |
| h4enpm19l0t3ng | smoke2-a40-0 | pool | run inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed (IORS under_large_scale) | 19:07 PDT | ~20:22 |
| uilwpybjxnekg0 | | band | 1 band job(s) running (pool 26 (18 free), bands 1 (0 idle), inflight 5, launched 2, ready 0, band remaining 0 / unassigned 0) | | |

## Idle pods: 19 pool, band 0

reason: ready queue empty for the pool: 2 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale); 4 waiting for a fix (alignment-world_vertical, alignment-world_horizontal, IORS under_small_scale, IORS ontop_large_scale); 8 valid; 5 in flight; coverage paused
holds: CM articulation_open_small_scale: owner 17:10 item 3 decided (25 deg cone) - tabletop implementing; lifts on its targeted fix-log entr; CM articulation_close_small_scale: same as open (25 deg cone)

## Queued runs: 0


## Board

| subtask | state | blocking | owner |
|---|---|---|---|
| camera_relocalization-standard | valid demo | - | - |
| camera_relocalization-obstructed | valid demo | tracker corner fix d4a1745 not yet re-demoed | oracle |
| object_scaling | valid demo | - | - |
| mental_rotation | valid demo | - | - |
| disembedding | valid demo (pre-bcc937d) | - | - |
| CM block_design | no instance | DECISION D-006: last block cannot be pinched into 1 mm gaps (finger-clearance gate 4c66d53a) — 6 mm gaps / oracle push-only tail / 3 blocks; | owner (D-006) / tabletop |
| CM attachment | no instance | second base pose (58afae2) + brace faces near/left/far (bd30c0cf) + oracle program travels the recorded stance, brace optional (6baaaa3b) —  | loop / tabletop |
| CM articulation_open_small_scale | no instance | 25° cone landed (aea6bd0a): Phase 2 survives; now contact_unreachable at the brace pre-contact (no brace face + handle grasp passes the two- | tabletop |
| CM articulation_close_small_scale | no instance | same as open | tabletop |
| CM articulation_open_large_scale | no instance | all five A3 units handle_unreachable x8 (fridge, dishwasher, oven, top_cabinet, no_top); Rs_int drawer running; large-scale: object-inclusiv | large-scale |
| CM articulation_close_large_scale | no instance | same as open; Rs_int drawer close = Phase-1 no_stance (base_ok instrumented 1c829ae) | large-scale |
| alignment-world_vertical | demo run | oracle chain landed (standoff, 3-D arc about the hook, loaded bounds): demo runs to the end in tolerance (0.052 rad) but not_settled — the f | large-scale |
| alignment-world_horizontal | demo run | push plan_failed = self-collision left_realsense_link vs right_arm_link7 (13 mm) along every path: the sidecar's push witnesses have the bra | large-scale |
| alignment-axial | demo run (pre-fix success) | - | - |
| IORS ontop_small_scale | valid demo (pre-bcc937d) | valid on 6b856ed 18:59; re-opened by the core's 02c700d (two-tier grasp draw) — re-run scheduled (D-005 asks whether blanket fixes should ke | core / loop |
| IORS under_small_scale | demo run (pre-fix success) | valid on 6b856ed 19:10 (release surface 5851c8b); re-opened by 02c700d — re-run scheduled (D-005) | core / loop |
| IORS inside | valid demo (pre-bcc937d) | 18:53 failed_grasp at grasp (sprout spins out of the pinch?) → core diagnosing with the new grasp facts; 02c700d re-run scheduled | core / loop |
| IORS nextto | demo run (pre-fix success) | oracle re-run 0916-181440: relation held, then the opening finger swept the pencil 9 cm at release (pencil 3 cm from the marker inside the o | core / oracle-B |
| IORS touching | demo run | core 6de8803: brace optional (push chain required) — napkins loop launched 18:12; oracle program_touching must accept the brace-less sidecar | loop / oracle |
| IORS ontop_large_scale | no instance | countertop pair = geometric yield (partner behind stance A) → floor fallback alias queued (local); sofa-bar re-run with the B→C path limit g | large-scale / loop |
| IORS under_large_scale | no instance | push-under lanes 0.45–0.6 m ahead + edge-end laterals + frontal offset 0.25 m right (7162fe7, 8c93d0b, 2cfce47) — re-run queued/running | large-scale / loop |

## Open decisions: D-006

- D-006 (open) CM block_design: the last block of the 2×2 grid cannot be pinched into 1 mm gaps (two occupied sides) — widen the gaps to 6 mm, or let the oracle push the last block into place? — default: A at 21:00 PDT if unanswered (first demo), B stays the required end-state question for the full 2×2 with 1 mm gaps
