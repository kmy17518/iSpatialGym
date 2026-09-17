# Idle pods (2026-09-16 18:12 PDT)

pool: 21 pods, 11 busy (locks held), 10 free; first-demo runs in flight: 9 (pending a lock: 0); ready queue: 0 entries
bands: 6 pods, 0 busy, 0 idle; manifests ['jobs_shampoo_countertops.json', 'jobs_w3_heldout.json']: 43 remaining (not done / not running), 37 unassigned
local GPU: busy (reserved for ontop large-scale; runs in flight: 1)

| pod | name | role | idle since | why nothing was schedulable |
|---|---|---|---|---|
| usqruny80uunl6 | pilot2-l40s-9 | pool | 2026-09-16 17:04 PDT | ready queue empty for the pool: 3 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale, inter_object_relation_satisfaction-nextto); 3 waiting for a fix (alignment-world_horizontal, IORS under_small_scale, IORS touching); 8 valid; 5 in flight; coverage paused |
| 5gm563m9wgnrwc | pilot2-l40s-10 | pool | 2026-09-16 17:04 PDT | ready queue empty for the pool: 3 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale, inter_object_relation_satisfaction-nextto); 3 waiting for a fix (alignment-world_horizontal, IORS under_small_scale, IORS touching); 8 valid; 5 in flight; coverage paused |
| 6pdrunfn78iofy | pilot2-a40-4 | pool | 2026-09-16 17:04 PDT | ready queue empty for the pool: 3 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale, inter_object_relation_satisfaction-nextto); 3 waiting for a fix (alignment-world_horizontal, IORS under_small_scale, IORS touching); 8 valid; 5 in flight; coverage paused |
| rzcqsre0b89792 | pilot2-a40-5 | pool | 2026-09-16 17:04 PDT | ready queue empty for the pool: 3 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale, inter_object_relation_satisfaction-nextto); 3 waiting for a fix (alignment-world_horizontal, IORS under_small_scale, IORS touching); 8 valid; 5 in flight; coverage paused |
| 4udkfzz52mzpih | pilot2-a40-7 | pool | 2026-09-16 17:04 PDT | ready queue empty for the pool: 3 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale, inter_object_relation_satisfaction-nextto); 3 waiting for a fix (alignment-world_horizontal, IORS under_small_scale, IORS touching); 8 valid; 5 in flight; coverage paused |
| 8lp72savyk8co4 | pilot2-a40-8 | pool | 2026-09-16 17:04 PDT | ready queue empty for the pool: 3 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale, inter_object_relation_satisfaction-nextto); 3 waiting for a fix (alignment-world_horizontal, IORS under_small_scale, IORS touching); 8 valid; 5 in flight; coverage paused |
| gvua02xnwm4ti6 | pilot2-a40-9 | pool | 2026-09-16 17:04 PDT | ready queue empty for the pool: 3 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale, inter_object_relation_satisfaction-nextto); 3 waiting for a fix (alignment-world_horizontal, IORS under_small_scale, IORS touching); 8 valid; 5 in flight; coverage paused |
| 29s7u374iifjbj | pilot2-a40-10 | pool | 2026-09-16 17:04 PDT | ready queue empty for the pool: 3 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale, inter_object_relation_satisfaction-nextto); 3 waiting for a fix (alignment-world_horizontal, IORS under_small_scale, IORS touching); 8 valid; 5 in flight; coverage paused |
| z830mv63m0wjt7 | pilot2-a40-11 | pool | 2026-09-16 17:04 PDT | ready queue empty for the pool: 3 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale, inter_object_relation_satisfaction-nextto); 3 waiting for a fix (alignment-world_horizontal, IORS under_small_scale, IORS touching); 8 valid; 5 in flight; coverage paused |
| 1qzna64k3s9t8s | pilot2-l40s-1 | pool | 2026-09-16 17:24 PDT | ready queue empty for the pool: 3 subtask(s) on hold for an owner decision (configuration_matching-articulation_open_small_scale, configuration_matching-articulation_close_small_scale, inter_object_relation_satisfaction-nextto); 3 waiting for a fix (alignment-world_horizontal, IORS under_small_scale, IORS touching); 8 valid; 5 in flight; coverage paused |

## Subtasks not in the queue

| subtask | why |
|---|---|
| camera_relocalization-standard | valid demo on the board |
| camera_relocalization-obstructed | valid demo on the board |
| object_scaling | valid demo on the board |
| mental_rotation | valid demo on the board |
| disembedding | valid demo on the board |
| configuration_matching-block_design | run in flight (pid 869723, run configuration_matching-block_design-block_4-breakfast_table) |
| configuration_matching-attachment | run in flight (pid 905641, run configuration_matching-attachment-lid-teapot-breakfast_table) |
| configuration_matching-articulation_open_small_scale | hold: owner 17:10 item 3 decided (25 deg cone) - tabletop implementing; lifts on its targeted fix-log entry |
| configuration_matching-articulation_close_small_scale | hold: same as open (25 deg cone) |
| configuration_matching-articulation_open_large_scale | run in flight (pid 926872, run configuration_matching-articulation_open_large_scale-dishwasher) |
| configuration_matching-articulation_close_large_scale | already in the ready queue |
| alignment-world_vertical | run in flight (pid 937724, demo-only alignment-world_vertical-painting_s070-walls) |
| alignment-world_horizontal | last run (18:01 PDT) already carried every fix and did not pass - waiting for a fix: oracle re-run on 4b2bb60: push plan_failed again (16:56) → oracle (owner: oracle) |
| alignment-axial | valid demo on the board |
| inter_object_relation_satisfaction-ontop_small_scale | valid demo on the board |
| inter_object_relation_satisfaction-under_small_scale | last run (17:58 PDT) already carried every fix and did not pass - waiting for a fix: core 6307dd7: the cup rotated in the hand at the grasp (grasp_3 closing axis 28° off horizontal) → flush pinches only + release at half the well depth; loop run (owner: core / loop) |
| inter_object_relation_satisfaction-inside | valid demo on the board |
| inter_object_relation_satisfaction-nextto | hold: owner 17:10 item 4: closure-roll investigation (agent 52b4fb5f, <= 3 h) then either the closure fix or cylinder equivalences in the gate; item 5 FOV rule (b) re-run follows |
| inter_object_relation_satisfaction-touching | last run (17:54 PDT) already carried every fix and did not pass - waiting for a fix: owner 17:10: brace optional (as axial) — core (iors_small + program_check) + oracle (program); loop on the fix (owner: core → oracle) |
| inter_object_relation_satisfaction-ontop_large_scale | already in the ready queue |
| inter_object_relation_satisfaction-under_large_scale | run in flight (pid 861918, run inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed) |

## Ready queue

| prio | state | subtask | kind | alias | note |
|---|---|---|---|---|---|
| 15 | running | configuration_matching-articulation_close_large_scale | run | configuration_matching-articulation_close_large_scale-bottom_cabinet | Rs_int drawer: Phase 1 base_ok diagnostics |
| 15 | running | inter_object_relation_satisfaction-ontop_large_scale | run (local) | inter_object_relation_satisfaction-ontop_large_scale-box_of_shampoo-sofa-bar | B->C path limit 6 m (4df6d0ac): the sofa run failed only on path_too_long 4.7-5. |
| 55 | running | configuration_matching-block_design | run | configuration_matching-block_design-block_4-breakfast_table | auto: new fix(es) 4f01c3c, 6931c6e, c88a3e3, 01cb9b2, 561f343, d03b21d, 9729372, |
| 56 | running | configuration_matching-attachment | run | configuration_matching-attachment-lid-teapot-breakfast_table | auto: new fix(es) 5e30334, 5e30334, 625759b, bcc1478, 58afae2 since the last run |
| 70 | running | inter_object_relation_satisfaction-under_large_scale | run | inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed | auto: new fix(es) 7162fe7, 8c93d0b since the last run |
