# Status 2026-09-16 23:19 PDT

**8 / 21 subtasks with a valid demo.** balance $47.82   fleet spend/h $12.884   spent so far $687.38 of cap $1000   recharges=30

## Runs in flight

| pod | name | role | run | since | ETA |
|---|---|---|---|---|---|
| bpu6l9bffwaf1y | pilot2-l40s-11 | pool | run alignment-world_horizontal-license_plate-walls (alignment-world_horizontal) | 23:14 PDT | ~00:29 |
| bl91tuqop45ila | pilot2-l40s-2 | pool | run alignment-world_vertical-painting_s070-walls (alignment-world_vertical) | 23:14 PDT | ~00:29 |
| 1c88jsy46ubecf | pilot2-a40-6 | pool | run configuration_matching-articulation_close_large_scale-bottom_cabinet (CM articulation_close_large_scale) | 23:14 PDT | ~00:29 |
| s7m4h2ay2slau4 | pilot2-l40s-4 | pool | run configuration_matching-articulation_close_small_scale-hinged_jar-breakfast_table (CM articulation_close_small_scale) | 23:14 PDT | ~00:29 |
| pjf6451ahyqxll | smoke2-a40-1 | pool | run configuration_matching-articulation_open_large_scale-dishwasher (CM articulation_open_large_scale) | 22:56 PDT | ~00:11 |
| h4enpm19l0t3ng | smoke2-a40-0 | pool | run configuration_matching-articulation_open_small_scale-hinged_jar-breakfast_table (CM articulation_open_small_scale) | 23:02 PDT | ~00:17 |
| oliefq3ymuexna | pilot2-a40-1 | pool | run configuration_matching-attachment-lid-teapot-breakfast_table (CM attachment) | 22:44 PDT | ~23:59 |
| htvfrubebp6tnj | pilot2-a40-3 | pool | run configuration_matching-block_design-block_4-breakfast_table (CM block_design) | 23:08 PDT | ~00:23 |
| gvua02xnwm4ti6 | pilot2-a40-9 | pool | run inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table (IORS ontop_small_scale) | 22:44 PDT | ~23:59 |
| 29s7u374iifjbj | pilot2-a40-10 | pool | run inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table (IORS touching) | 23:14 PDT | ~00:29 |
| nfgjwzgjajpr5u | pilot2-l40s-7 | pool | demo-only inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table (IORS touching) | 23:14 PDT | ~00:29 |
| local GPU | 4090 | local | run inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-countertop-floor | | |
| m48tuedy95bu05, rzcqsre0b89792, 8lp72savyk8co4, 1qzna64k3s9t8s | | band | 4 band job(s) running (pool 17 (3 free), bands 4 (0 idle), inflight 8, launched 6, ready 0, band remaining 0 / unassigned 0) | | |

## Idle pods: 6 pool, band 0

reason: ready queue empty for the pool: 0 valid; 7 in flight; coverage paused

## Queued runs: 0


## Board

| subtask | state | blocking | owner |
|---|---|---|---|
| camera_relocalization-standard | valid demo (pre-166005f,dde4f11) | - | - |
| camera_relocalization-obstructed | valid demo (pre-166005f,dde4f11) | tracker corner fix d4a1745 not yet re-demoed | oracle |
| object_scaling | valid demo (pre-166005f,dde4f11) | - | - |
| mental_rotation | valid demo (pre-166005f,dde4f11) | - | - |
| disembedding | valid demo (pre-bcc937d,166005f,de4e723,ab60baa,dde4f11) | - | - |
| CM block_design | no instance | D-006 default A applied 21:17 (block_gap_m 0.006, 8c2a2bc) — loop runs next tick; flip-on-table prep 0ec0153d (sampler) + 0ee927f6 (program) | loop / tabletop |
| CM attachment | demo run | re-run 0916-205838: each attempt burns the 360 s budget in the second-base-pose search — the SLERP's first waypoint (carried lift, 0.70–0.82 | tabletop |
| CM articulation_open_small_scale | no instance | D-008 A applied (e5fbced: brace_optional on the hinged_jar variants; program 59f6426f) — handle-only loop runs next tick | loop |
| CM articulation_close_small_scale | no instance | same as open | loop |
| CM articulation_open_large_scale | no instance | first instance + demo through grasp(handle) on the Rs_int drawer bamfsz_1 (probe; pick applied 0589d9d) — needs core program_check (handle g | core / oracle / owner (D-011) |
| CM articulation_close_large_scale | no instance | same pick (bamfsz_1); close waits on the open fixes | loop |
| alignment-world_vertical | demo run | painting demo in tolerance, not_settled (undamped hook) → large-scale-B: hinge friction/damping in the constraints record + TRO loader (owne | large-scale-B |
| alignment-world_horizontal | demo run | owner 22:55: hang frame bug — no wall variant carries front/up/level axes, the sampler defaults to the painting frame (+x/+z) → the plate hang | large-scale-B |
| alignment-axial | valid demo (pre-dde4f11) | - | - |
| IORS ontop_small_scale | demo run (pre-fix success) | - | - |
| IORS under_small_scale | valid demo (pre-de4e723,ab60baa,dde4f11) | valid (a6d7123, 21:46). D-007 branch: mechanism verified (2e889f3), a passing re-run pending the primary oracle's carry-leg regression (30-w | oracle |
| IORS inside | valid demo (pre-166005f,de4e723,ab60baa,dde4f11) | valid on f5ae607; re-opened by the core's targeted 1ba5e74 (open pads clear of the support: the failing sprout grasp had its fingertip at th | loop |
| IORS nextto | demo run (pre-fix success) | - | - |
| IORS touching | demo run | plan_failed at grasp = sampler witnesses off by a constant world offset (9–13 mm) and on joint-2's limit → core: witnesses at the recorded b | core / oracle |
| IORS ontop_large_scale | no instance | countertop-floor: setup fixed (c5040c8/b931b31/729a1ec/1dd3820/f0684cb), now B↔C coupling F_init ∩ F_goal empty → large-scale; sofa-bar limi | large-scale |
| IORS under_large_scale | no instance | D-010 (default C 23:30): sandal/bed F_goal 0 (matches B3); apple-desk-desk no_init_surface at setup; apple-desk-floor + pencil-sofa-sofa lau | owner (D-010) / large-scale |

## Open decisions: D-010, D-011

- D-010 (open) IORS under_large_scale: sandal / bed_rrcvaq_0 push-under: F_goal = 0 at every side-lane drop after four lane variants — is this pick a yield; should apple-desk-desk carry the under-large first demo? — default: C at 23:30 PDT if unanswered (apple-desk already running; the large-scale agent picks the sandal diagnosis up after the Rs_int drawer)
- D-011 (open) CM articulation_open_large_scale: door units under arm-only follow reach 9-33 % of the swing (oven 0 %), below the 50 deg minimum: first large-articulation demo = the Rs_int drawer, doors deferred to the hybrid base-follow? — default: A at 23:45 PDT if unanswered
