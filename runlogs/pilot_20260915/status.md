# Status 2026-09-16 19:22 PDT

**9 / 21 subtasks with a valid demo.** balance $42.98   fleet spend/h $20.992   spent so far $621.81 of cap $1000   recharges=27

## Runs in flight

| pod | name | role | run | since | ETA |
|---|---|---|---|---|---|
| bl91tuqop45ila | pilot2-l40s-2 | pool | run ? () | agent run |  |
| 5gm563m9wgnrwc | pilot2-l40s-10 | pool | run ? () | agent run |  |
| m48tuedy95bu05 | smoke2-l40s-0 | pool | run ? () | agent run |  |
| 6pdrunfn78iofy | pilot2-a40-4 | pool | demo-only alignment-world_vertical-painting_s070-walls (alignment-world_vertical) | agent run |  |
| htvfrubebp6tnj | pilot2-a40-3 | pool | run configuration_matching-articulation_open_large_scale-dishwasher (CM articulation_open_large_scale) | 19:17 PDT | ~20:32 |
| 1c88jsy46ubecf | pilot2-a40-6 | pool | run configuration_matching-attachment-lid-teapot-breakfast_table (CM attachment) | 19:17 PDT | ~20:32 |
| pjf6451ahyqxll | smoke2-a40-1 | pool | run configuration_matching-block_design-block_4-breakfast_table (CM block_design) | 19:07 PDT | ~20:22 |
| h4enpm19l0t3ng | smoke2-a40-0 | pool | run inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed (IORS under_large_scale) | 19:07 PDT | ~20:22 |
| uilwpybjxnekg0, xnam1nh3y2706e, oliefq3ymuexna, cy0f5t6i58i08v, s7m4h2ay2slau4, 4udkfzz52mzpih, 8lp72savyk8co4, gvua02xnwm4ti6, 29s7u374iifjbj, z830mv63m0wjt7, 1qzna64k3s9t8s, bpu6l9bffwaf1y, usqruny80uunl6, 6c5712ggii2ihf, 88hr6mw4y5i1h8, z2qdfyfyrgcjys, h7thjdjnyh8qcx, nfgjwzgjajpr5u, rzcqsre0b89792 | | band | 19 band job(s) running (pool 8 (0 free), bands 19 (0 idle), inflight 4, launched 2, ready 0, band remaining 11 / unassigned 6) | | |

## Idle pods: 0 pool, band 0

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
| CM block_design | no instance | owner 17:10 (1): first demo with block_no_flip (35506bf) — loop running (scheduler 17:24); flip-on-the-table sub-chain = required end state, | tabletop / oracle |
| CM attachment | no instance | tabletop 58afae2: second base pose between grasp and attach (paths carried, Transition.holding bd10c72) — lid-teapot loop running since 17:3 | loop / tabletop |
| CM articulation_open_small_scale | no instance | owner 17:10 (3): Phase-2 cone 25° / focus-in-frame for small articulation (tabletop, 20 min) → hinged_jar loop | tabletop |
| CM articulation_close_small_scale | no instance | same as open (25° cone) | tabletop |
| CM articulation_open_large_scale | no instance | D-004 = B (18:30): rvunhj_0/5 blacklisted (f8e9aa7); candidates fridge (handle_unreachable x8), dishwasher / oven / top_cabinet (running), b | large-scale |
| CM articulation_close_large_scale | no instance | same as open; Rs_int drawer close = Phase-1 no_stance (base_ok instrumented 1c829ae) | large-scale |
| alignment-world_vertical | demo run | oracle re-runs on 4b2bb60 / 9345c8a: push [execution] again (17:06) → oracle | oracle |
| alignment-world_horizontal | demo run | oracle re-run on 4b2bb60: push plan_failed again (16:56) → oracle | oracle |
| alignment-axial | valid demo | - | - |
| IORS ontop_small_scale | valid demo | - | - |
| IORS under_small_scale | valid demo | core 6307dd7: the cup rotated in the hand at the grasp (grasp_3 closing axis 28° off horizontal) → flush pinches only + release at half the  | core / loop |
| IORS inside | demo run (pre-fix success) | - | - |
| IORS nextto | valid demo | owner 17:10 (4): placed_tilt withdrawn (roll happens during finger closure, 24.6°); closure-roll investigation on a pod (≤ 3 h, agent 52b4fb | oracle / investigation |
| IORS touching | no instance | core 6de8803: brace optional (push chain required) — napkins loop launched 18:12; oracle program_touching must accept the brace-less sidecar | loop / oracle |
| IORS ontop_large_scale | no instance | countertop pair = geometric yield (partner behind stance A) → floor fallback alias queued (local); sofa-bar re-run with the B→C path limit g | large-scale / loop |
| IORS under_large_scale | no instance | push-under lanes 0.45–0.6 m ahead + edge-end laterals + frontal offset 0.25 m right (7162fe7, 8c93d0b, 2cfce47) — re-run queued/running | large-scale / loop |

## Open decisions: none

