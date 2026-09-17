# Status 2026-09-16 17:36 PDT

**6 / 21 subtasks with a valid demo.** balance $30.03   fleet spend/h $20.992   spent so far $588.24 of cap $1000   recharges=25

## Runs in flight

| pod | name | role | run | since | ETA |
|---|---|---|---|---|---|
| 88hr6mw4y5i1h8 | smoke2-l40s-1 | pool | demo-only alignment-world_horizontal-license_plate-walls (alignment-world_horizontal) | agent run |  |
| xnam1nh3y2706e | smoke-l40s-3 | pool | demo-only alignment-world_vertical-painting_s070-walls (alignment-world_vertical) | agent run |  |
| bpu6l9bffwaf1y | pilot2-l40s-11 | pool | run configuration_matching-articulation_close_large_scale-bottom_cabinet (CM articulation_close_large_scale) | 17:19 PDT | ~18:34 |
| z2qdfyfyrgcjys | pilot2-a40-0 | pool | run configuration_matching-articulation_open_large_scale-bottom_cabinet (CM articulation_open_large_scale) | 17:19 PDT | ~18:34 |
| h7thjdjnyh8qcx | pilot2-l40s-0 | pool | run configuration_matching-block_design-block_4-breakfast_table (CM block_design) | 17:23 PDT | ~18:38 |
| 6c5712ggii2ihf | pilot2-a40-2 | pool | run inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table (IORS ontop_small_scale) | 17:19 PDT | ~18:34 |
| htvfrubebp6tnj | pilot2-a40-3 | pool | run inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed (IORS under_large_scale) | 17:19 PDT | ~18:34 |
| oliefq3ymuexna | pilot2-a40-1 | pool | run inter_object_relation_satisfaction-under_small_scale-teacup-saucer-breakfast_table (IORS under_small_scale) | 17:23 PDT | ~18:38 |
| h4enpm19l0t3ng | smoke2-a40-0 | pool | run mental_rotation-banana (mental_rotation) | 17:13 PDT | ~18:28 |

## Idle pods: 18 pool, band 0

reason: ready queue empty for the pool: 5 subtask(s) on hold for an owner decision (configuration_matching-attachment, configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale, inter_object_relation_satisfaction-touching, inter_object_relation_satisfaction-nextto); 1 waiting for a fix (IORS ontop_large_scale); 4 valid; 11 in flight; coverage paused
holds: CM attachment: owner 17:10 item 2 decided (brace stays; second base pose after the grasp) - tabletop implementing; ; CM articulation_open_small_scale: owner 17:10 item 3 decided (25 deg cone) - tabletop implementing; lifts on its targeted fix-log entr; CM articulation_close_small_scale: same as open (25 deg cone); IORS touching: owner 17:10: brace optional (as axial) - core (iors_small + program_check) + oracle (program) implem; IORS nextto: owner 17:10 item 4: closure-roll investigation (agent 52b4fb5f, <= 3 h) then either the closure fix 

## Queued runs: 0


## Board

| subtask | state | blocking | owner |
|---|---|---|---|
| camera_relocalization-standard | valid demo | - | - |
| camera_relocalization-obstructed | valid demo | tracker corner fix d4a1745 not yet re-demoed | oracle |
| object_scaling | valid demo | - | - |
| mental_rotation | demo run (pre-fix success) | - | - |
| disembedding | valid demo | - | - |
| CM block_design | no instance | owner 17:10 (1): first demo with block_no_flip (35506bf) — loop running (scheduler 17:24); flip-on-the-table sub-chain = required end state, | tabletop / oracle |
| CM attachment | no instance | owner 17:10 (2): brace stays — second base pose recorded after the male grasp, sidecar transition, oracle executes it holding the object; br | tabletop → oracle |
| CM articulation_open_small_scale | no instance | owner 17:10 (3): Phase-2 cone 25° / focus-in-frame for small articulation (tabletop, 20 min) → hinged_jar loop | tabletop |
| CM articulation_close_small_scale | no instance | same as open (25° cone) | tabletop |
| CM articulation_open_large_scale | no instance | HEAD (floor 2893a09 + A3 05ce671): open handle_unreachable x8 (step 5), close no_manipulation_stance x8 (Phase 2) — large-scale diagnosing;  | large-scale |
| CM articulation_close_large_scale | no instance | HEAD (floor 2893a09 + A3 05ce671): no_manipulation_stance x8 (Phase 2) — large-scale diagnosing; open alias running | large-scale |
| alignment-world_vertical | demo run | oracle re-runs on 4b2bb60 / 9345c8a: push [execution] again (17:06) → oracle | oracle |
| alignment-world_horizontal | demo run | oracle re-run on 4b2bb60: push plan_failed again (16:56) → oracle | oracle |
| alignment-axial | valid demo | - | - |
| IORS ontop_small_scale | demo run (pre-fix success) | - | - |
| IORS under_small_scale | demo run (pre-fix success) | core 6307dd7: the cup rotated in the hand at the grasp (grasp_3 closing axis 28° off horizontal) → flush pinches only + release at half the  | core / loop |
| IORS inside | valid demo | - | - |
| IORS nextto | demo run (pre-fix success) | owner 17:10 (4): placed_tilt withdrawn (roll happens during finger closure, 24.6°); closure-roll investigation on a pod (≤ 3 h, agent 52b4fb | oracle / investigation |
| IORS touching | no instance | owner 17:10: brace optional (as axial) — core (iors_small + program_check) + oracle (program); loop on the fix | core → oracle |
| IORS ontop_large_scale | no instance | countertop pair: goal_region_not_visible at the only free edge (+u) now logged; sofa-bar target_placement x8 → large-scale | large-scale |
| IORS under_large_scale | no instance | blade 6–9 cm (d073579): target_placement x8 (new category) → large-scale | large-scale |

## Open decisions: D-004

- D-004 (open) CM articulation_open_large_scale: rvunhj_0 (cubicle bottom cabinet, office_cubicles_right) after the floor fix: open = step-5 handle_unreachable ×8, close = no_manipulation_stance ×8 — keep working the arm-only class or blacklist rvunhj_0 + rvunhj_5? — default: A until the 0427202 verdict lands (≤ 1 h); if it is still handle_unreachable, run B's fallback aliases in parallel and keep rvunhj_0 open
