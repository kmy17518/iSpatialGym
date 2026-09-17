# Status 2026-09-16 18:21 PDT

**5 / 21 subtasks with a valid demo.** balance $38.95   fleet spend/h $20.992   spent so far $602.56 of cap $1000   recharges=26

## Runs in flight

| pod | name | role | run | since | ETA |
|---|---|---|---|---|---|
| m48tuedy95bu05 | smoke2-l40s-0 | pool | run ? () | agent run |  |
| nfgjwzgjajpr5u | pilot2-l40s-7 | pool | run ? () | agent run |  |
| h4enpm19l0t3ng | smoke2-a40-0 | pool | demo-only alignment-world_vertical-painting_s070-walls (alignment-world_vertical) | agent run |  |
| 1c88jsy46ubecf | pilot2-a40-6 | pool | run configuration_matching-articulation_open_large_scale-dishwasher (CM articulation_open_large_scale) | agent run |  |
| z2qdfyfyrgcjys | pilot2-a40-0 | pool | run configuration_matching-articulation_open_large_scale-oven (CM articulation_open_large_scale) | agent run |  |
| bpu6l9bffwaf1y | pilot2-l40s-11 | pool | run configuration_matching-articulation_open_large_scale-top_cabinet (CM articulation_open_large_scale) | agent run |  |
| pjf6451ahyqxll | smoke2-a40-1 | pool | run configuration_matching-attachment-lid-teapot-breakfast_table (CM attachment) | 17:38 PDT | ~18:53 |
| h7thjdjnyh8qcx | pilot2-l40s-0 | pool | run configuration_matching-block_design-block_4-breakfast_table (CM block_design) | 17:23 PDT | ~18:38 |
| 6c5712ggii2ihf | pilot2-a40-2 | pool | run disembedding-bottle_of_milk-bottle_of_milk_1-breakfast_table (disembedding) | 18:12 PDT | ~19:27 |
| 6pdrunfn78iofy | pilot2-a40-4 | pool | run inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table (IORS inside) | 18:12 PDT | ~19:27 |
| 88hr6mw4y5i1h8 | smoke2-l40s-1 | pool | demo-only inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table (IORS nextto) | agent run |  |
| rzcqsre0b89792 | pilot2-a40-5 | pool | run inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table (IORS ontop_small_scale) | 18:12 PDT | ~19:27 |
| 5gm563m9wgnrwc | pilot2-l40s-10 | pool | run inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table (IORS touching) | 18:12 PDT | ~19:27 |
| htvfrubebp6tnj | pilot2-a40-3 | pool | run inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed (IORS under_large_scale) | 17:19 PDT | ~18:34 |
| usqruny80uunl6 | pilot2-l40s-9 | pool | run inter_object_relation_satisfaction-under_small_scale-teacup-saucer-breakfast_table (IORS under_small_scale) | 18:12 PDT | ~19:27 |
| uilwpybjxnekg0, xnam1nh3y2706e, oliefq3ymuexna, bl91tuqop45ila, cy0f5t6i58i08v, s7m4h2ay2slau4, 4udkfzz52mzpih, 8lp72savyk8co4, gvua02xnwm4ti6, 29s7u374iifjbj, z830mv63m0wjt7, 1qzna64k3s9t8s | | band | 12 band job(s) running (pool 15 (0 free), bands 12 (0 idle), inflight 9, launched 5, ready 0, band remaining 37 / unassigned 31) | | |

## Idle pods: 0 pool, band 0

holds: CM articulation_open_small_scale: owner 17:10 item 3 decided (25 deg cone) - tabletop implementing; lifts on its targeted fix-log entr; CM articulation_close_small_scale: same as open (25 deg cone); IORS nextto: owner 17:10 item 4: closure-roll investigation (agent 52b4fb5f, <= 3 h) then either the closure fix 

## Queued runs: 1

- prio 16 run inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-countertop-floor — next scheduler tick (≤ 5 min), ETA ~75 min after launch

## Board

| subtask | state | blocking | owner |
|---|---|---|---|
| camera_relocalization-standard | valid demo | - | - |
| camera_relocalization-obstructed | valid demo | tracker corner fix d4a1745 not yet re-demoed | oracle |
| object_scaling | valid demo | - | - |
| mental_rotation | valid demo | - | - |
| disembedding | demo run (pre-fix success) | - | - |
| CM block_design | no instance | owner 17:10 (1): first demo with block_no_flip (35506bf) — loop running (scheduler 17:24); flip-on-the-table sub-chain = required end state, | tabletop / oracle |
| CM attachment | no instance | tabletop 58afae2: second base pose between grasp and attach (paths carried, Transition.holding bd10c72) — lid-teapot loop running since 17:3 | loop / tabletop |
| CM articulation_open_small_scale | no instance | owner 17:10 (3): Phase-2 cone 25° / focus-in-frame for small articulation (tabletop, 20 min) → hinged_jar loop | tabletop |
| CM articulation_close_small_scale | no instance | same as open (25° cone) | tabletop |
| CM articulation_open_large_scale | no instance | D-004: large-scale verdict = blacklist rvunhj_0 (8 mm bar / 7 mm knobs < 15 mm pinch → edge-pinch fallback, follow ≤ 25°) unless the owner r | owner (D-004) / large-scale |
| CM articulation_close_large_scale | no instance | same as open (close sweep follows ≤ 25° from 90° on rvunhj_0; 9a3e678 records the honest fraction); Rs_int drawer queued | owner (D-004) / large-scale |
| alignment-world_vertical | demo run | oracle re-runs on 4b2bb60 / 9345c8a: push [execution] again (17:06) → oracle | oracle |
| alignment-world_horizontal | demo run | oracle re-run on 4b2bb60: push plan_failed again (16:56) → oracle | oracle |
| alignment-axial | valid demo | - | - |
| IORS ontop_small_scale | demo run (pre-fix success) | - | - |
| IORS under_small_scale | demo run (pre-fix success) | core 6307dd7: the cup rotated in the hand at the grasp (grasp_3 closing axis 28° off horizontal) → flush pinches only + release at half the  | core / loop |
| IORS inside | demo run (pre-fix success) | - | - |
| IORS nextto | demo run (pre-fix success) | owner 17:10 (4): placed_tilt withdrawn (roll happens during finger closure, 24.6°); closure-roll investigation on a pod (≤ 3 h, agent 52b4fb | oracle / investigation |
| IORS touching | no instance | core 6de8803: brace optional (push chain required) — napkins loop launched 18:12; oracle program_touching must accept the brace-less sidecar | loop / oracle |
| IORS ontop_large_scale | no instance | countertop pair = geometric yield (partner behind stance A) → floor fallback alias queued (local); sofa-bar re-run with the B→C path limit g | large-scale / loop |
| IORS under_large_scale | no instance | push-under lanes 0.45–0.6 m ahead + edge-end laterals + frontal offset 0.25 m right (7162fe7, 8c93d0b, 2cfce47) — re-run queued/running | large-scale / loop |

## Open decisions: D-004

- D-004 (open) CM articulation_open_large_scale: rvunhj_0 (cubicle bottom cabinet, office_cubicles_right) after the floor fix: open = step-5 handle_unreachable ×8, close = no_manipulation_stance ×8 — keep working the arm-only class or blacklist rvunhj_0 + rvunhj_5? — default: A until the 0427202 verdict lands (≤ 1 h); if it is still handle_unreachable, run B's fallback aliases in parallel and keep rvunhj_0 open
