# E1 pilot — yield audit (2026-09-16 02:10 PDT, after the C0 stop)

Source: 21 pilot pods, `done.txt` + `summary.json` + `run.log` of every job dir (`runlogs/runpod/pilot/live/`, full artefacts in
`runlogs/runpod/pilot/<pod>/`), `progress.json` (ledger per alias), `killed_at_stop.json`. 197 aliases × (1 template × 2 instances,
`--max-attempts 10`): **complete 76, partial 1, zero 25, crashed 59, killed 7, never started 29 → 151 instances**.

## Red flags (owner)

1. **Demo QA `gripper_out_of_fov` fires at the idle posture — 6/6 tabletop demos failed (IORS nextto)**: the verdict comes 5 steps
   into `grasp` (u = 1108 > 1080, depth 0.45) before the arm has moved: the right hand in its idle posture at the manipulation stance
   is just outside the right image border. The sampler guarantees pre-grasp / grasp / place poses in frame, not the idle pose →
   the monitor must arm once the fingertips first enter the frame within the action (or after the pre-pose is reached). Oracle bug;
   blocks every manipulation demo. (`runlogs/runpod/pilot/pilot2-a40-0/demo__v2__inter_object_relation_satisfaction-nextto-*/*/meta.json`)
2. **Obstructed camera demos: 7 success / 5 `collision` during `navigate`, always instance `0001` of a template (0000 succeeds)**:
   `wheel_motor_link1` vs the obstacle itself (footstool, serving cart) or `torso_link1` vs a sofa. Suspects: the base tracker cutting
   the detour corner (corridor ≥ 0.75 m leaves ~12 cm per side), or a per-template artefact applied to the second instance
   (`blocked_footprints` / obstacle pose restore). Needs the hdf5 base trajectory vs the sidecar path. Sampler-or-oracle bug.
3. **Disembedding 0/26 — the source tray is placed ≈ 24 cm below the table top** (`no raycast hits on the tray`, 12) or its inner
   floor raster is nearly empty (`inner floor too small`, 10). Module bug (agent 8db7c96f on it). **Blocks the reschedule of 30 aliases.**
4. **Axial 2/5 complete at 14 attempts per instance, 3 zero** — every rejection is `target_placement`: the left stop‑edge brace reaches
   across the centre line (`contact_unreachable`) or no long‑axis end lands in the band (`grasp_unreachable`) → decisions A1 (left band,
   joint placement, pitched push class). Block design 0/1 for the same reason (`target_placement` 20).
5. **Wall alignment 0/20**: `push_unreachable` 167, `no_manipulation_stance` 25, 7 aliases `no wall spot passes Phase 2 at δ = ±30°`
   (plus 5 crashes on the missing reach volume, fixed) → A6 (idle‑arm posture), A8 (δ 15–25°), B (scale_hint fallback).
6. **Articulation 0/17**: large `no_stance` 250 / `handle_unreachable` 24 (+2 aliases crashed on the cuRobo mesh cache, fixed 4096) → A4;
   small: `carton-floor` grounding contract (`bindings.support must list scene objects` ×2, bug), toolbox `no_workspace` (no brace‑band
   cell on the top at s = 0.45 ×3, band/height check needed), `missing_handle` 8.4 mm ×2 and blender `no_stable_state` ×2 (yield → fallback).
7. **IORS large 0/8**: `no_init_surface` (desk/sofa/bar/countertop: no free edge at s = 0.45; floor candidates fail) → A5 + A3.
8. Held‑out / never started tail: 29 never started (object_scaling 8, mental_rotation 3, axial 5, attachment 4, disembedding 4, …),
   7 killed at the stop — all re‑queued by the new manifest.

## -1 / crash classification

| cause | aliases | class | status |
|---|---|---|---|
| `reach_volume.npz` not on the pods (git‑ignored) | 5 wall | bug | fixed `e26746f`, re‑run |
| cuRobo `Cannot add mesh, cache is full (2048)` | 4 large articulation | bug | fixed (4096), re‑run |
| axial `Draft` shadowed by a tensor | 1 | bug | fixed, re‑run |
| disembedding tray below the top / empty inner floor | 22 | bug | **open — blocks** (agent 8db7c96f) |
| small articulation `carton-floor`: `bindings.support` contract | 2 | bug (grounding ↔ module) | open, small |
| toolbox brace band: no cell on the top at s = 0.45 | 3 | bug or band scope | open, check |
| wall `no wall spot … δ = ±30°` | 7 | spec (A8 / A6 / scale_hint) | decisions |
| IORS large `no_init_surface` | 8 | spec (A5 / A3) | decisions |
| attachment `edge_too_short` 1.65 > 1.16, no attach‑reach survivor | 2 | spec (A1 / A3) | decisions |
| `missing_handle` 8.4 mm, blender `no_stable_state`, `does_not_fit_inside`, `target_not_pushable` | 6 | yield → fallback alias (B) | decisions |

Bugs that block the reschedule: the disembedding tray (30 aliases), the QA fov rule (every manipulation demo), the obstructed
navigation collision (half the camera demos). The other crashes are fixed or need only the decisions.

## Working well
mental_rotation 25/25 (1.1 attempts per instance, median 435 s per alias incl. scene load), object_scaling 19/19 (1.1), obstructed
16/17 (3.2), standard 1/1, IORS ontop‑small 5/5 (2.7), nextto 4/5 + 1 partial (2.8), under‑small 2/2 (1.5), inside 2/5 (14.5).
Demos: mental_rotation 1/1, obstructed 7/12 (see flag 2), nextto 0/6 (flag 1).


197 aliases; states: complete 76, crashed 59, killed 7, never_started 29, partial 1, zero 25

| subtask | aliases | ran | n_written 0/1/2 | crashed | killed / never | median wall s | max wall s | attempts per instance | dominant attempt rejections (excl. scene_reset) | demos ok / run |
|---|---|---|---|---|---|---|---|---|---|---|
| alignment-axial | 10 | 5 | 3/0/2 | 0 | 5 | 3131 | 3725 | 14.2 | target_placement 53 | 0 / 0 |
| alignment-world_horizontal | 10 | 10 | 10/0/0 | 7 | 0 | 207 | 2350 | - | push_unreachable 48, no_manipulation_stance 12, no_stance 4 | 0 / 0 |
| alignment-world_vertical | 10 | 10 | 10/0/0 | 7 | 0 | 375 | 2835 | - | push_unreachable 119, no_manipulation_stance 13, unstable_init 8, no_stance 5 | 0 / 0 |
| camera_relocalization-obstructed | 17 | 17 | 1/0/16 | 0 | 0 | 942 | 1824 | 3.2 | obstacle 39, goal_sampling 24, init_false 4, program_check_failed 3 | 7 / 12 |
| camera_relocalization-standard | 1 | 1 | 0/0/1 | 0 | 0 | 393 | 393 | 2.5 | goal_sampling 1, program_check_failed 1, init_false 1 | 0 / 0 |
| configuration_matching-articulation_close_large_scale | 6 | 6 | 6/0/0 | 1 | 0 | 1092 | 2690 | - | no_stance 154, no_manipulation_stance 26 | 0 / 0 |
| configuration_matching-articulation_close_small_scale | 5 | 5 | 5/0/0 | 5 | 0 | 186 | 598 | - | no_stance 2, spec 1, no_stable_state 1, missing_handle 1 | 0 / 0 |
| configuration_matching-articulation_open_large_scale | 6 | 3 | 3/0/0 | 1 | 3 | 3407 | 4667 | - | no_stance 96, handle_unreachable 24 | 0 / 0 |
| configuration_matching-articulation_open_small_scale | 5 | 4 | 4/0/0 | 4 | 1 | 71 | 140 | - | no_stable_state 1, spec 1, missing_handle 1, no_stance 1 | 0 / 0 |
| configuration_matching-attachment | 6 | 2 | 2/0/0 | 2 | 4 | 1361 | 1586 | - | no_stance 2 | 0 / 0 |
| configuration_matching-block_design | 1 | 1 | 1/0/0 | 0 | 0 | 2858 | 2858 | - | target_placement 20 | 0 / 0 |
| disembedding | 30 | 26 | 26/0/0 | 22 | 4 | 50 | 137 | - | support 22 | 0 / 0 |
| inter_object_relation_satisfaction-inside | 5 | 5 | 3/0/2 | 1 | 0 | 381 | 1039 | 14.5 | place_unreachable 28, grasp_unreachable 22, program_check_failed 2, stance_collision_with_objects 2 | 0 / 0 |
| inter_object_relation_satisfaction-nextto | 5 | 5 | 0/1/4 | 0 | 0 | 282 | 727 | 2.8 | place_unreachable 10, grasp_unreachable 5, unstable_init 1 | 0 / 6 |
| inter_object_relation_satisfaction-ontop_large_scale | 4 | 4 | 4/0/0 | 4 | 0 | 27 | 45 | - | no_init_surface 4 | 0 / 0 |
| inter_object_relation_satisfaction-ontop_small_scale | 5 | 5 | 0/0/5 | 0 | 0 | 484 | 929 | 2.7 | program_check_failed 5, grasp_unreachable 4, place_unreachable 4, unstable_init 2 | 0 / 0 |
| inter_object_relation_satisfaction-touching | 3 | 2 | 2/0/0 | 1 | 1 | 631 | 1206 | - | place_unreachable 12, grasp_unreachable 7, goal_infeasible 1, target_not_pushable 1 | 0 / 0 |
| inter_object_relation_satisfaction-under_large_scale | 4 | 4 | 4/0/0 | 4 | 0 | 26 | 26 | - | no_init_surface 4 | 0 / 0 |
| inter_object_relation_satisfaction-under_small_scale | 4 | 2 | 0/0/2 | 0 | 2 | 174 | 228 | 1.5 | goal_infeasible 2 | 0 / 0 |
| mental_rotation | 30 | 25 | 0/0/25 | 0 | 5 | 435 | 642 | 1.1 | grasp_failed 2, goal_infeasible 1 | 1 / 1 |
| object_scaling | 30 | 19 | 0/0/19 | 0 | 11 | 347 | 610 | 1.1 | goal_infeasible 2 | 0 / 0 |

## alignment-axial

Candidate-level (log) rejections: `[target_placement] candidate #: [grasp_unreachable] neither long#axis end inside the band has an IK#feasible ` ×27; `[target_placement] candidate #: [contact_unreachable] stop#edge brace with the push arm locked: left (brace):` ×21; `[target_placement] candidate #: [contact_unreachable] stop#edge brace with the push arm locked: combined conf` ×12
- `alignment-axial-catalog-breakfast_table`: **complete** n=2 wall=3472 pod=pilot2-l40s-2 rej={'target_placement': 6}   demos: -
- `alignment-axial-board_game-breakfast_table`: **complete** n=2 wall=3725 pod=pilot2-l40s-2 rej={'target_placement': 7}   demos: -
- `alignment-axial-frying_pan-breakfast_table`: **never_started** n=0 wall=None pod=pilot2-a40-11 rej={}   demos: -
- `alignment-axial-bag_of_chips-breakfast_table`: **never_started** n=0 wall=None pod=pilot2-a40-8 rej={}   demos: -
- `alignment-axial-gym_shoe-breakfast_table`: **never_started** n=0 wall=None pod=pilot2-l40s-0 rej={}   demos: -
- `alignment-axial-hardback-breakfast_table`: **never_started** n=0 wall=None pod=pilot2-l40s-1 rej={}   demos: -
- `alignment-axial-keyboard-breakfast_table`: **never_started** n=0 wall=None pod=pilot2-l40s-3 rej={}   demos: -
- `alignment-axial-paperback_book-breakfast_table`: **zero** n=0 wall=None pod=pilot2-l40s-10 rej={}   demos: -
- `alignment-axial-letter-breakfast_table`: **zero** n=0 wall=2789 pod=pilot2-l40s-11 rej={'target_placement': 20}   demos: -
- `alignment-axial-magazine-breakfast_table`: **zero** n=0 wall=1753 pod=pilot2-l40s-9 rej={'target_placement': 20}   demos: -

## alignment-world_horizontal

Crashes (Traceback): `FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/ispatialgym/reach_volume/reach_volume.npz'` ×3
Setup errors: `setup rejected: [no_stance] alignment-world_horizontal-picture-walls: no wall spot passes Phase 2 steps 1–3 fo` ×1; `setup rejected: [no_stance] alignment-world_horizontal-shelf-walls: no wall spot passes Phase 2 steps 1–3 for ` ×1; `setup rejected: [no_stance] alignment-world_horizontal-wall_mounted_tv-walls: no wall spot passes Phase 2 step` ×1; `setup rejected: [no_stance] alignment-world_horizontal-umbrella-walls: no wall spot passes Phase 2 steps 1–3 f` ×1
Candidate-level (log) rejections: `[push_unreachable] # Phase# survivors (δ #°, {'brace_kinds_clear': ['below_high', 'above_low'], 'brace_kinds_` ×32; `[push_unreachable] # Phase# survivors (δ +#°, {'brace_kinds_clear': ['below_high', 'above_low'], 'brace_kinds` ×16; `[no_manipulation_stance] spot # δ #°: no Phase# survivor ({'brace_kinds_clear': ['below_high', 'above_low'], 'brace` ×12
- `alignment-world_horizontal-decorative_sign-walls`: **crashed** n=0 wall=74 pod=pilot2-a40-0 rej={} exc=FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/ispatialgy  demos: -
- `alignment-world_horizontal-picture-walls`: **crashed** n=0 wall=290 pod=pilot2-a40-0 rej={'no_stance': 1}  err=setup rejected: [no_stance] alignment-world_horizontal-picture-walls: no wall sp demos: -
- `alignment-world_horizontal-hanger-walls`: **crashed** n=0 wall=80 pod=pilot2-a40-1 rej={} exc=FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/ispatialgy  demos: -
- `alignment-world_horizontal-license_plate-walls`: **crashed** n=0 wall=71 pod=pilot2-a40-2 rej={} exc=FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/ispatialgy  demos: -
- `alignment-world_horizontal-shelf-walls`: **crashed** n=0 wall=188 pod=pilot2-a40-2 rej={'no_stance': 1}  err=setup rejected: [no_stance] alignment-world_horizontal-shelf-walls: no wall spot demos: -
- `alignment-world_horizontal-wall_mounted_tv-walls`: **crashed** n=0 wall=128 pod=pilot2-a40-5 rej={'no_stance': 1}  err=setup rejected: [no_stance] alignment-world_horizontal-wall_mounted_tv-walls: no demos: -
- `alignment-world_horizontal-umbrella-walls`: **crashed** n=0 wall=226 pod=pilot2-a40-8 rej={'no_stance': 1}  err=setup rejected: [no_stance] alignment-world_horizontal-umbrella-walls: no wall s demos: -
- `alignment-world_horizontal-radiator-walls`: **zero** n=0 wall=977 pod=pilot2-a40-1 rej={'push_unreachable': 17, 'no_manipulation_stance': 3}   demos: -
- `alignment-world_horizontal-signpost-walls`: **zero** n=0 wall=2350 pod=pilot2-a40-3 rej={'push_unreachable': 17, 'no_manipulation_stance': 3}   demos: -
- `alignment-world_horizontal-towel_rack-walls`: **zero** n=0 wall=1260 pod=pilot2-a40-4 rej={'push_unreachable': 14, 'no_manipulation_stance': 6}   demos: -

## alignment-world_vertical

Crashes (Traceback): `FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/ispatialgym/reach_volume/reach_volume.npz'` ×2
Setup errors: `setup rejected: [no_stance] alignment-world_vertical-fire_extinguisher-walls: no wall spot passes Phase 2 step` ×1; `setup rejected: [no_stance] alignment-world_vertical-guitar-walls: no wall spot passes Phase 2 steps 1–3 for δ` ×1; `setup rejected: [no_stance] alignment-world_vertical-painting-walls: no wall spot passes Phase 2 steps 1–3 for` ×1; `setup rejected: [no_stance] alignment-world_vertical-paper_towel_dispenser-walls: no wall spot passes Phase 2 ` ×1; `setup rejected: [no_stance] alignment-world_vertical-picture-walls: no wall spot passes Phase 2 steps 1–3 for ` ×1
Candidate-level (log) rejections: `[push_unreachable] # Phase# survivors (δ #°, {'brace_kinds_clear': ['below_high', 'above_low'], 'brace_kinds_` ×52; `[push_unreachable] # Phase# survivors (δ +#°, {'brace_kinds_clear': ['below_high', 'above_low'], 'brace_kinds` ×47; `[no_manipulation_stance] spot # δ #°: no Phase# survivor ({'brace_kinds_clear': ['below_high', 'above_low'], 'brace` ×13; `[push_unreachable] # Phase# survivors (δ #°, {'brace_kinds_clear': ['above_low', 'side_high'], 'brace_kinds_b` ×8; `[unstable_init] target_hanger_agrpio_# moving (v=# m/s, w=# rad/s)` ×8; `[push_unreachable] # Phase# survivors (δ #°, {'brace_kinds_clear': ['below_high', 'above_low', 'side_high'], ` ×4
- `alignment-world_vertical-signpost-walls`: **crashed** n=0 wall=46 pod=pilot2-a40-0 rej={} exc=FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/ispatialgy  demos: -
- `alignment-world_vertical-wall_clock-walls`: **crashed** n=0 wall=52 pod=pilot2-a40-1 rej={} exc=FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/ispatialgy  demos: -
- `alignment-world_vertical-fire_extinguisher-walls`: **crashed** n=0 wall=358 pod=pilot2-a40-10 rej={'no_stance': 1}  err=setup rejected: [no_stance] alignment-world_vertical-fire_extinguisher-walls: no demos: -
- `alignment-world_vertical-guitar-walls`: **crashed** n=0 wall=633 pod=pilot2-a40-11 rej={'no_stance': 1}  err=setup rejected: [no_stance] alignment-world_vertical-guitar-walls: no wall spot  demos: -
- `alignment-world_vertical-painting-walls`: **crashed** n=0 wall=229 pod=pilot2-l40s-3 rej={'no_stance': 1}  err=setup rejected: [no_stance] alignment-world_vertical-painting-walls: no wall spo demos: -
- `alignment-world_vertical-paper_towel_dispenser-walls`: **crashed** n=0 wall=142 pod=pilot2-l40s-7 rej={'no_stance': 1}  err=setup rejected: [no_stance] alignment-world_vertical-paper_towel_dispenser-walls demos: -
- `alignment-world_vertical-picture-walls`: **crashed** n=0 wall=392 pod=pilot2-l40s-9 rej={'no_stance': 1}  err=setup rejected: [no_stance] alignment-world_vertical-picture-walls: no wall spot demos: -
- `alignment-world_vertical-picture_frame-walls`: **zero** n=0 wall=2080 pod=pilot2-a40-6 rej={'push_unreachable': 20}   demos: -
- `alignment-world_vertical-hanger-walls`: **zero** n=0 wall=2835 pod=pilot2-l40s-0 rej={'push_unreachable': 39, 'no_manipulation_stance': 13, 'unstable_init': 8}   demos: -
- `alignment-world_vertical-license_plate-walls`: **zero** n=0 wall=2322 pod=pilot2-l40s-1 rej={'push_unreachable': 60}   demos: -

## camera_relocalization-obstructed

Candidate-level (log) rejections: `[goal_sampling] no goal candidate passed collision / path / bucket / visibility (# rendered; {'path_collis` ×18; `[init_false] {'satisfied': [#], 'unsatisfied': [#]}` ×4; `[obstacle] no cell of the shortest path admits music_stool#n#_# (# cells tried: {'unstable': #}; free` ×4; `[obstacle] no cell of the shortest path admits music_stool#n#_# (# cells tried: {'unstable': #, 'shor` ×4; `[program_check_failed] base pose (#, #, #) collides at trunk (#, #) {'action': 'navigate'}` ×3; `[goal_sampling] no goal candidate passed collision / path / bucket / visibility (# rendered; {'goal_object` ×3
- `camera_relocalization-obstructed-straight_chair_1`: **complete** n=2 wall=970 pod=pilot2-a40-0 rej={'obstacle': 2, 'goal_sampling': 1}   demos: success, oracle_failed/collision
- `camera_relocalization-obstructed-taboret_1`: **complete** n=2 wall=858 pod=pilot2-a40-0 rej={'obstacle': 2, 'goal_sampling': 1}   demos: killed
- `camera_relocalization-obstructed-swivel_chair_1`: **complete** n=2 wall=1119 pod=pilot2-a40-1 rej={'obstacle': 2, 'goal_sampling': 1}   demos: success, success
- `camera_relocalization-obstructed-teddy_bear_1`: **complete** n=2 wall=889 pod=pilot2-a40-1 rej={'obstacle': 2, 'goal_sampling': 1}   demos: -
- `camera_relocalization-obstructed-carton_1`: **complete** n=2 wall=966 pod=pilot2-a40-10 rej={'obstacle': 2, 'goal_sampling': 1}   demos: killed
- `camera_relocalization-obstructed-footstool_1`: **complete** n=2 wall=942 pod=pilot2-a40-11 rej={'obstacle': 2, 'goal_sampling': 1}   demos: success, oracle_failed/collision
- `camera_relocalization-obstructed-toy_car_1`: **complete** n=2 wall=1557 pod=pilot2-a40-2 rej={'goal_sampling': 1, 'obstacle': 1}   demos: -
- `camera_relocalization-obstructed-trash_can_1`: **complete** n=2 wall=719 pod=pilot2-a40-3 rej={'goal_sampling': 1, 'program_check_failed': 1, 'obstacle': 1}   demos: killed
- `camera_relocalization-obstructed-vacuum_1`: **complete** n=2 wall=857 pod=pilot2-a40-4 rej={'goal_sampling': 1, 'init_false': 1, 'obstacle': 1}   demos: -
- `camera_relocalization-obstructed-wicker_basket_1`: **complete** n=2 wall=1531 pod=pilot2-a40-5 rej={'goal_sampling': 1, 'obstacle': 1}   demos: killed
- `camera_relocalization-obstructed-hamper_1`: **complete** n=2 wall=1379 pod=pilot2-l40s-0 rej={'obstacle': 2, 'goal_sampling': 1}   demos: -
- `camera_relocalization-obstructed-serving_cart_1`: **complete** n=2 wall=1187 pod=pilot2-l40s-10 rej={'obstacle': 4, 'goal_sampling': 2, 'init_false': 1}   demos: success, oracle_failed/collision
- `camera_relocalization-obstructed-ottoman_1`: **complete** n=2 wall=577 pod=pilot2-l40s-2 rej={'goal_sampling': 1, 'obstacle': 1, 'init_false': 1}   demos: -
- `camera_relocalization-obstructed-poinsettia_1`: **complete** n=2 wall=467 pod=pilot2-l40s-3 rej={'goal_sampling': 1, 'program_check_failed': 1, 'obstacle': 1}   demos: success, oracle_failed/collision
- `camera_relocalization-obstructed-pot_plant_1`: **complete** n=2 wall=394 pod=pilot2-l40s-4 rej={'goal_sampling': 1, 'program_check_failed': 1, 'obstacle': 1}   demos: -
- `camera_relocalization-obstructed-public_trash_can_1`: **complete** n=2 wall=443 pod=pilot2-l40s-7 rej={'goal_sampling': 1, 'obstacle': 1, 'init_false': 1}   demos: success, oracle_failed/collision
- `camera_relocalization-obstructed-music_stool_1`: **zero** n=0 wall=1824 pod=pilot2-l40s-1 rej={'obstacle': 13, 'goal_sampling': 7}   demos: -

## camera_relocalization-standard

Candidate-level (log) rejections: `[goal_sampling] no goal candidate passed collision / path / bucket / visibility (# rendered; {'path_collis` ×1; `[program_check_failed] base pose (#, #, #) collides at trunk (#, #) {'action': 'navigate'}` ×1; `[init_false] {'satisfied': [], 'unsatisfied': [#]}` ×1
- `camera_relocalization-standard`: **complete** n=2 wall=393 pod=pilot2-a40-6 rej={'goal_sampling': 1, 'program_check_failed': 1, 'init_false': 1}   demos: -

## configuration_matching-articulation_close_large_scale

Crashes (Traceback): `ValueError: Cannot add mesh, cache is full (2048 meshes)` ×2
Candidate-level (log) rejections: `[no_stance] Phase #: no (d, t) on # approach rays (bearings [#, #, #, #, #, #, #]°, d #–# m) sees the ` ×154; `[no_manipulation_stance] Phase # left no survivor for bottom_cabinet_no_top_gjrero_#/j_link_# at q_init # ({'step#'` ×20; `[no_manipulation_stance] Phase # left no survivor for bottom_cabinet_fancyy_#/j_door# at q_init # ({'step#': #, 'st` ×6
- `configuration_matching-articulation_close_large_scale-door`: **crashed** n=0 wall=46 pod=pilot2-a40-11 rej={} exc=ValueError: Cannot add mesh, cache is full (2048 meshes)  demos: -
- `configuration_matching-articulation_close_large_scale-fridge`: **zero** n=0 wall=2138 pod=pilot2-a40-0 rej={'no_stance': 60}   demos: -
- `configuration_matching-articulation_close_large_scale-oven`: **zero** n=0 wall=2690 pod=pilot2-a40-1 rej={'no_stance': 60}   demos: -
- `configuration_matching-articulation_close_large_scale-dishwasher`: **zero** n=0 wall=592 pod=pilot2-a40-10 rej={'no_stance': 20}   demos: -
- `configuration_matching-articulation_close_large_scale-bottom_cabinet`: **zero** n=0 wall=1160 pod=pilot2-a40-8 rej={'no_stance': 14, 'no_manipulation_stance': 6}   demos: -
- `configuration_matching-articulation_close_large_scale-bottom_cabinet_no_top`: **zero** n=0 wall=1025 pod=pilot2-a40-9 rej={'no_manipulation_stance': 20}   demos: -

## configuration_matching-articulation_close_small_scale

Setup errors: `setup rejected: [no_stance] edge +v: [no_workspace] breakfast_table_rhjoby_0 edge +v: no band cell of /root/is` ×2; `setup rejected: [spec] configuration_matching-articulation_close_small_scale-carton-floor: scenes.house_single` ×1; `setup rejected: [no_stable_state] target_blender_cwkvib_0/j_lid holds no open value passively` ×1; `setup rejected: [missing_handle] pinch width 8.4 mm outside [15.0, 80.0] mm after scaling` ×1
- `configuration_matching-articulation_close_small_scale-toolbox-breakfast_table`: **crashed** n=0 wall=186 pod=pilot2-a40-10 rej={'no_stance': 1}  err=setup rejected: [no_stance] edge +v: [no_workspace] breakfast_table_rhjoby_0 edg demos: -
- `configuration_matching-articulation_close_small_scale-carton-floor`: **crashed** n=0 wall=12 pod=pilot2-a40-7 rej={'spec': 1}  err=setup rejected: [spec] configuration_matching-articulation_close_small_scale-car demos: -
- `configuration_matching-articulation_close_small_scale-hinged_jar-breakfast_table`: **crashed** n=0 wall=221 pod=pilot2-a40-8 rej={'no_stance': 1}  err=setup rejected: [no_stance] edge +v: [no_workspace] breakfast_table_rhjoby_0 edg demos: -
- `configuration_matching-articulation_close_small_scale-blender-breakfast_table`: **crashed** n=0 wall=598 pod=pilot2-a40-9 rej={'no_stable_state': 1}  err=setup rejected: [no_stable_state] target_blender_cwkvib_0/j_lid holds no open va demos: -
- `configuration_matching-articulation_close_small_scale-laptop-breakfast_table`: **crashed** n=0 wall=35 pod=pilot2-a40-9 rej={'missing_handle': 1}  err=setup rejected: [missing_handle] pinch width 8.4 mm outside [15.0, 80.0] mm afte demos: -

## configuration_matching-articulation_open_large_scale

Crashes (Traceback): `ValueError: Cannot add mesh, cache is full (2048 meshes)` ×2
Candidate-level (log) rejections: `[no_stance] Phase #: no (d, t) on # approach rays (bearings [#, #, #, #, #, #, #]°, d #–# m) sees the ` ×96; `[handle_unreachable] # Phase# survivors, none passes step # / base#follow / step #: handle_unreachable: [handle` ×24
- `configuration_matching-articulation_open_large_scale-door`: **crashed** n=0 wall=74 pod=pilot2-a40-5 rej={} exc=ValueError: Cannot add mesh, cache is full (2048 meshes)  demos: -
- `configuration_matching-articulation_open_large_scale-bottom_cabinet_no_top`: **killed** n=0 wall=None pod=pilot2-a40-3 rej={}   demos: -
- `configuration_matching-articulation_open_large_scale-dishwasher`: **never_started** n=0 wall=None pod=pilot2-a40-4 rej={}   demos: -
- `configuration_matching-articulation_open_large_scale-fridge`: **never_started** n=0 wall=None pod=pilot2-a40-6 rej={}   demos: -
- `configuration_matching-articulation_open_large_scale-bottom_cabinet`: **zero** n=0 wall=3407 pod=pilot2-a40-2 rej={'no_stance': 36, 'handle_unreachable': 24}   demos: -
- `configuration_matching-articulation_open_large_scale-oven`: **zero** n=0 wall=4667 pod=pilot2-a40-7 rej={'no_stance': 60}   demos: -

## configuration_matching-articulation_open_small_scale

Setup errors: `setup rejected: [no_stable_state] target_blender_cwkvib_0/j_lid holds no open value passively` ×1; `setup rejected: [spec] configuration_matching-articulation_open_small_scale-carton-floor: scenes.house_single_` ×1; `setup rejected: [missing_handle] pinch width 8.4 mm outside [15.0, 80.0] mm after scaling` ×1; `setup rejected: [no_stance] edge +v: [no_workspace] breakfast_table_rhjoby_0 edge +v: no band cell of /root/is` ×1
- `configuration_matching-articulation_open_small_scale-blender-breakfast_table`: **crashed** n=0 wall=140 pod=pilot2-a40-11 rej={'no_stable_state': 1}  err=setup rejected: [no_stable_state] target_blender_cwkvib_0/j_lid holds no open va demos: -
- `configuration_matching-articulation_open_small_scale-carton-floor`: **crashed** n=0 wall=6 pod=pilot2-l40s-0 rej={'spec': 1}  err=setup rejected: [spec] configuration_matching-articulation_open_small_scale-cart demos: -
- `configuration_matching-articulation_open_small_scale-laptop-breakfast_table`: **crashed** n=0 wall=38 pod=pilot2-l40s-10 rej={'missing_handle': 1}  err=setup rejected: [missing_handle] pinch width 8.4 mm outside [15.0, 80.0] mm afte demos: -
- `configuration_matching-articulation_open_small_scale-toolbox-breakfast_table`: **crashed** n=0 wall=104 pod=pilot2-l40s-2 rej={'no_stance': 1}  err=setup rejected: [no_stance] edge +v: [no_workspace] breakfast_table_rhjoby_0 edg demos: -
- `configuration_matching-articulation_open_small_scale-hinged_jar-breakfast_table`: **never_started** n=0 wall=None pod=None rej={}   demos: -

## configuration_matching-attachment

Setup errors: `setup rejected: [no_stance] edge +v: [no_manipulation_stance] Phase 2 left no survivor with attach-reach cells` ×1; `setup rejected: [no_stance] edge +v: [edge_too_short] edge +v: m_R + m_L = 1.65 > L = 1.16 ({'kind': 'attachme` ×1
- `configuration_matching-attachment-dip_candle-candle_holder-breakfast_table`: **crashed** n=0 wall=1137 pod=pilot2-a40-4 rej={'no_stance': 1}  err=setup rejected: [no_stance] edge +v: [no_manipulation_stance] Phase 2 left no su demos: -
- `configuration_matching-attachment-skateboard_wheel_assembly-skateboard_deck-breakfast_table`: **crashed** n=0 wall=1586 pod=pilot2-a40-7 rej={'no_stance': 1}  err=setup rejected: [no_stance] edge +v: [edge_too_short] edge +v: m_R + m_L = 1.65  demos: -
- `configuration_matching-attachment-cap-lemonade_bottle-breakfast_table`: **never_started** n=0 wall=None pod=pilot2-a40-2 rej={}   demos: -
- `configuration_matching-attachment-cork-vinegar_bottle-breakfast_table`: **never_started** n=0 wall=None pod=pilot2-a40-3 rej={}   demos: -
- `configuration_matching-attachment-lid-teapot-breakfast_table`: **never_started** n=0 wall=None pod=pilot2-a40-5 rej={}   demos: -
- `configuration_matching-attachment-light_bulb-table_lamp-breakfast_table`: **never_started** n=0 wall=None pod=pilot2-a40-6 rej={}   demos: -

## configuration_matching-block_design

Candidate-level (log) rejections: `[target_placement] candidate #: [grasp_unreachable] block toy_dice#n#_#: no pick passes the screening IK; can` ×11; `[target_placement] candidate #: [place_unreachable] block toy_dice#n#_#: no cube grasp passes pick + place (+` ×4; `[target_placement] candidate #: [layout] goal faces below # visible: {#: #}; candidate #: [place_unreachable]` ×3; `[target_placement] candidate #: [layout] goal faces below # visible: {#: #}; candidate #: [grasp_unreachable]` ×1; `[target_placement] candidate #: [layout] goal faces below # visible: {#: #}; candidate #: [layout] goal faces` ×1
- `configuration_matching-block_design-block_4-breakfast_table`: **zero** n=0 wall=2858 pod=pilot2-a40-9 rej={'target_placement': 20}   demos: -

## disembedding

Setup errors: `setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (1015/1015 rays hit ['/World/scene` ×3; `setup rejected: [support] source_tray_gsxbym_0: inner floor too small (3 cells)` ×3; `setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (1530/1530 rays hit ['/World/scene` ×2; `setup rejected: [support] source_tray_gsxbym_0: inner floor too small (10 cells)` ×2; `setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (1575/1575 rays hit ['/World/scene` ×2
- `disembedding-box_of_cereal-box_of_cereal_2_box_of_baking_mix_1-breakfast_table`: **crashed** n=0 wall=55 pod=pilot2-a40-0 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (153 demos: -
- `disembedding-box_of_cookies-box_of_cookies_2_bag_of_cookies_1-breakfast_table`: **crashed** n=0 wall=58 pod=pilot2-a40-1 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (101 demos: -
- `disembedding-conditioner_dispenser-bottle_of_cologne_1_bottle_of_dish_soap_1_bottle_of_lotion_1-breakfast_table`: **crashed** n=0 wall=64 pod=pilot2-a40-10 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (133 demos: -
- `disembedding-box_of_rice-bag_of_flour_1_box_of_baking_mix_1_box_of_baking_powder_1-breakfast_table`: **crashed** n=0 wall=64 pod=pilot2-a40-2 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: inner floor too small (10 cells) demos: -
- `disembedding-box_of_sanitary_napkins-box_of_sanitary_napkins_2_bag_of_breadcrumbs_1-breakfast_table`: **crashed** n=0 wall=60 pod=pilot2-a40-3 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: inner floor too small (8 cells) demos: -
- `disembedding-box_of_tissues-bag_of_cookies_1_bag_of_cream_cheese_1_bag_of_ice_cream_1-breakfast_table`: **crashed** n=0 wall=137 pod=pilot2-a40-5 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: inner floor too small (3 cells) demos: -
- `disembedding-can_of_soda-can_of_soda_2_can_of_baking_mix_1-breakfast_table`: **crashed** n=0 wall=59 pod=pilot2-a40-6 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: inner floor too small (11 cells) demos: -
- `disembedding-carton_of_milk-bottle_of_coconut_milk_1_bottle_of_milk_1_bottle_of_soy_milk_1-breakfast_table`: **crashed** n=0 wall=69 pod=pilot2-a40-7 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (157 demos: -
- `disembedding-cellulose_tape-biscuit_1_butter_cookie_1_cap_1-breakfast_table`: **crashed** n=0 wall=78 pod=pilot2-a40-8 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: inner floor too small (3 cells) demos: -
- `disembedding-chili-asparagus_1_bacon_1_banana_1-breakfast_table`: **crashed** n=0 wall=63 pod=pilot2-a40-9 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (986 demos: -
- `disembedding-box_of_baking_mix-box_of_baking_mix_2_box_of_baking_powder_1-breakfast_table`: **crashed** n=0 wall=49 pod=pilot2-l40s-10 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (153 demos: -
- `disembedding-toy_dice-toy_dice_2_bar_soap_1-breakfast_table`: **crashed** n=0 wall=48 pod=pilot2-l40s-10 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (101 demos: -
- `disembedding-box_of_cane_sugar-box_of_cane_sugar_2_bag_of_flour_1-breakfast_table`: **crashed** n=0 wall=32 pod=pilot2-l40s-11 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (157 demos: -
- `disembedding-video_game-video_game_2_board_game_1-breakfast_table`: **crashed** n=0 wall=34 pod=pilot2-l40s-11 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: inner floor too small (3 cells) demos: -
- `disembedding-bottle_of_beer-bottle_of_beer_2_bottle_of_champagne_1-breakfast_table`: **crashed** n=0 wall=33 pod=pilot2-l40s-2 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (130 demos: -
- `disembedding-pack_of_cigarettes-pack_of_cigarettes_2_bandage_1-breakfast_table`: **crashed** n=0 wall=32 pod=pilot2-l40s-2 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: inner floor too small (0 cells) demos: -
- `disembedding-pencil_case-chocolate_bar_1_flashlight_1_granola_bar_1-breakfast_table`: **crashed** n=0 wall=39 pod=pilot2-l40s-3 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: inner floor too small (7 cells) demos: -
- `disembedding-bottle_of_milk-bottle_of_milk_2_bottle_of_apple_juice_1-breakfast_table`: **crashed** n=0 wall=31 pod=pilot2-l40s-4 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (101 demos: -
- `disembedding-soap_dispenser-soap_dispenser_2_bleaching_agent_atomizer_1-breakfast_table`: **crashed** n=0 wall=34 pod=pilot2-l40s-4 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: inner floor too small (10 cells) demos: -
- `disembedding-bottle_of_shampoo-bottle_of_cologne_1_bottle_of_conditioner_1_bottle_of_dish_soap_1-breakfast_table`: **crashed** n=0 wall=36 pod=pilot2-l40s-7 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (130 demos: -
- `disembedding-bottle_of_wine-bottle_of_wine_2_bottle_of_beer_1-breakfast_table`: **crashed** n=0 wall=49 pod=pilot2-l40s-9 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: inner floor too small (12 cells) demos: -
- `disembedding-teaspoon-teaspoon_2_chopstick_1-breakfast_table`: **crashed** n=0 wall=50 pod=pilot2-l40s-9 rej={'support': 1}  err=setup rejected: [support] source_tray_gsxbym_0: no raycast hits on the tray (105 demos: -
- `disembedding-bar_soap-bar_soap_2_biscuit_1-breakfast_table`: **never_started** n=0 wall=None pod=pilot2-l40s-0 rej={}   demos: -
- `disembedding-beaker-beaker_2_beer_glass_1-breakfast_table`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `disembedding-mug-mug_2_beaker_1-breakfast_table`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `disembedding-notebook-notebook_2_board_game_1-breakfast_table`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `disembedding-mouse-mouse_2_bar_soap_1-breakfast_table`: **zero** n=0 wall=None pod=pilot2-a40-11 rej={}   demos: -
- `disembedding-box_of_shampoo-box_of_shampoo_2_bottle_of_shampoo_1-breakfast_table`: **zero** n=0 wall=None pod=pilot2-a40-4 rej={}   demos: -
- `disembedding-bottle_of_medicine-bottle_of_medicine_2_bottle_of_antihistamines_1-breakfast_table`: **zero** n=0 wall=None pod=pilot2-l40s-3 rej={}   demos: -
- `disembedding-tablefork-tablefork_2_chopstick_1-breakfast_table`: **zero** n=0 wall=None pod=pilot2-l40s-7 rej={}   demos: -

## inter_object_relation_satisfaction-inside

Setup errors: `setup rejected: [does_not_fit_inside] target 0.08x0.19 vs opening 0.10x0.10` ×1
Candidate-level (log) rejections: `[place_unreachable] pre#place / place fail the exact IK from the manipulation stance with the object attached ` ×28; `[grasp_unreachable] pre#grasp / grasp at the settled target pose fail the exact IK` ×22; `[stance_collision_with_objects] stances ['manipulation'] collide with the objects placed (trunks [(#, #)])` ×2; `[program_check_failed] approach: waypoints [#] of # fail the screening IK (stance manipulation, arm right, carrie` ×1; `[program_check_failed] retreat: waypoints [#] of # fail the screening IK (stance manipulation, arm right, carried` ×1
- `inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table`: **complete** n=2 wall=381 pod=pilot2-l40s-3 rej={'grasp_unreachable': 7, 'program_check_failed': 2}   demos: -
- `inter_object_relation_satisfaction-inside-hard_candy-hingeless_jar-breakfast_table`: **complete** n=2 wall=312 pod=pilot2-l40s-4 rej={'place_unreachable': 5}   demos: -
- `inter_object_relation_satisfaction-inside-sunflower-vase-breakfast_table`: **crashed** n=0 wall=38 pod=pilot2-l40s-10 rej={'does_not_fit_inside': 1}  err=setup rejected: [does_not_fit_inside] target 0.08x0.19 vs opening 0.10x0.10 demos: -
- `inter_object_relation_satisfaction-inside-ice_cube-beer_glass-breakfast_table`: **zero** n=0 wall=696 pod=pilot2-l40s-7 rej={'place_unreachable': 12, 'grasp_unreachable': 8}   demos: -
- `inter_object_relation_satisfaction-inside-quartz-saucepot-breakfast_table`: **zero** n=0 wall=1039 pod=pilot2-l40s-9 rej={'place_unreachable': 11, 'grasp_unreachable': 7, 'stance_collision_with_objects': 2}   demos: -

## inter_object_relation_satisfaction-nextto

Candidate-level (log) rejections: `[place_unreachable] pre#place / place fail the exact IK from the manipulation stance with the object attached ` ×10; `[grasp_unreachable] pre#grasp / grasp at the settled target pose fail the exact IK` ×5; `[unstable_init] target_pencil_hmawzy_# moving (v=# m/s, w=# rad/s)` ×1
- `inter_object_relation_satisfaction-nextto-jar_of_spaghetti_sauce-canned_food-breakfast_table`: **complete** n=2 wall=214 pod=pilot2-a40-0 rej={}   demos: oracle_failed/gripper_out_of_fov, oracle_failed/gripper_out_of_fov
- `inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table`: **complete** n=2 wall=249 pod=pilot2-a40-1 rej={}   demos: oracle_failed/gripper_out_of_fov, oracle_failed/gripper_out_of_fov
- `inter_object_relation_satisfaction-nextto-parsnip-carrot-breakfast_table`: **complete** n=2 wall=299 pod=pilot2-a40-2 rej={'place_unreachable': 1}   demos: -
- `inter_object_relation_satisfaction-nextto-cinnamon_stick-star_anise-breakfast_table`: **complete** n=2 wall=282 pod=pilot2-l40s-11 rej={'place_unreachable': 4}   demos: oracle_failed/gripper_out_of_fov, oracle_failed/gripper_out_of_fov
- `inter_object_relation_satisfaction-nextto-pencil-ruler-breakfast_table`: **partial** n=1 wall=727 pod=pilot2-a40-3 rej={'grasp_unreachable': 5, 'place_unreachable': 5, 'unstable_init': 1}   demos: -

## inter_object_relation_satisfaction-ontop_large_scale

Setup errors: `setup rejected: [no_init_surface] inter_object_relation_satisfaction-ontop_large_scale-beaker-bed-desk: no sup` ×1; `setup rejected: [no_init_surface] inter_object_relation_satisfaction-ontop_large_scale-box_of_shampoo-sofa-bar` ×1; `setup rejected: [no_init_surface] inter_object_relation_satisfaction-ontop_large_scale-pencil-desk-floor: no f` ×1; `setup rejected: [no_init_surface] inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-count` ×1
- `inter_object_relation_satisfaction-ontop_large_scale-beaker-bed-desk`: **crashed** n=0 wall=27 pod=pilot2-a40-0 rej={'no_init_surface': 1}  err=setup rejected: [no_init_surface] inter_object_relation_satisfaction-ontop_large demos: -
- `inter_object_relation_satisfaction-ontop_large_scale-box_of_shampoo-sofa-bar`: **crashed** n=0 wall=45 pod=pilot2-a40-1 rej={'no_init_surface': 1}  err=setup rejected: [no_init_surface] inter_object_relation_satisfaction-ontop_large demos: -
- `inter_object_relation_satisfaction-ontop_large_scale-pencil-desk-floor`: **crashed** n=0 wall=28 pod=pilot2-a40-2 rej={'no_init_surface': 1}  err=setup rejected: [no_init_surface] inter_object_relation_satisfaction-ontop_large demos: -
- `inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-countertop-countertop`: **crashed** n=0 wall=24 pod=pilot2-a40-3 rej={'no_init_surface': 1}  err=setup rejected: [no_init_surface] inter_object_relation_satisfaction-ontop_large demos: -

## inter_object_relation_satisfaction-ontop_small_scale

Candidate-level (log) rejections: `[grasp_unreachable] pre#grasp / grasp at the settled target pose fail the exact IK` ×4; `[place_unreachable] pre#place / place fail the exact IK from the manipulation stance with the object attached ` ×4; `[program_check_failed] retreat: waypoints [#] of # fail the screening IK (stance manipulation, arm right, carried` ×3; `[unstable_init] target_coffee_cup_siksnl_# moving (v=# m/s, w=# rad/s)` ×2; `[stance_collision_with_objects] stances ['manipulation'] collide with the objects placed (trunks [(#, #)])` ×2; `[program_check_failed] approach: waypoints [#] of # fail the screening IK (stance manipulation, arm right, carrie` ×1
- `inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table`: **complete** n=2 wall=510 pod=pilot2-a40-4 rej={}   demos: -
- `inter_object_relation_satisfaction-ontop_small_scale-coffee_cup-saucer-breakfast_table`: **complete** n=2 wall=929 pod=pilot2-a40-5 rej={'grasp_unreachable': 4, 'place_unreachable': 4, 'program_check_failed': 2}   demos: -
- `inter_object_relation_satisfaction-ontop_small_scale-half_bagel-plate-breakfast_table`: **complete** n=2 wall=236 pod=pilot2-a40-6 rej={}   demos: -
- `inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table`: **complete** n=2 wall=428 pod=pilot2-a40-7 rej={'stance_collision_with_objects': 2, 'program_check_failed': 1}   demos: -
- `inter_object_relation_satisfaction-ontop_small_scale-pickle-cutting_board-breakfast_table`: **complete** n=2 wall=484 pod=pilot2-a40-8 rej={'program_check_failed': 2}   demos: -

## inter_object_relation_satisfaction-touching

Setup errors: `setup rejected: [target_not_pushable] target 0.10x0.13x0.07` ×1
Candidate-level (log) rejections: `[place_unreachable] pre#place / place fail the exact IK from the manipulation stance with the object attached ` ×12; `[grasp_unreachable] pre#grasp / grasp at the settled target pose fail the exact IK` ×7; `[goal_infeasible] unsatisfied ["['touching', '?carton__of__milk#n#_#', '?box__of__cookies#n#_#']"]; goal obj` ×1
- `inter_object_relation_satisfaction-touching-teacup-teapot-breakfast_table`: **crashed** n=0 wall=56 pod=pilot2-a40-11 rej={'target_not_pushable': 1}  err=setup rejected: [target_not_pushable] target 0.10x0.13x0.07 demos: -
- `inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table`: **killed** n=0 wall=None pod=pilot2-a40-9 rej={}   demos: -
- `inter_object_relation_satisfaction-touching-carton_of_milk-box_of_cookies-breakfast_table`: **zero** n=0 wall=1206 pod=pilot2-a40-10 rej={'place_unreachable': 12, 'grasp_unreachable': 7, 'goal_infeasible': 1}   demos: -

## inter_object_relation_satisfaction-under_large_scale

Setup errors: `setup rejected: [no_init_surface] inter_object_relation_satisfaction-under_large_scale-apple-desk-floor: no fl` ×1; `setup rejected: [no_init_surface] inter_object_relation_satisfaction-under_large_scale-marker-desk-floor: no f` ×1; `setup rejected: [no_init_surface] inter_object_relation_satisfaction-under_large_scale-pencil-sofa-desk: no su` ×1; `setup rejected: [no_init_surface] inter_object_relation_satisfaction-under_large_scale-sandal-bed-desk: no sup` ×1
- `inter_object_relation_satisfaction-under_large_scale-apple-desk-floor`: **crashed** n=0 wall=26 pod=pilot2-a40-4 rej={'no_init_surface': 1}  err=setup rejected: [no_init_surface] inter_object_relation_satisfaction-under_large demos: -
- `inter_object_relation_satisfaction-under_large_scale-marker-desk-floor`: **crashed** n=0 wall=24 pod=pilot2-a40-5 rej={'no_init_surface': 1}  err=setup rejected: [no_init_surface] inter_object_relation_satisfaction-under_large demos: -
- `inter_object_relation_satisfaction-under_large_scale-pencil-sofa-desk`: **crashed** n=0 wall=25 pod=pilot2-a40-6 rej={'no_init_surface': 1}  err=setup rejected: [no_init_surface] inter_object_relation_satisfaction-under_large demos: -
- `inter_object_relation_satisfaction-under_large_scale-sandal-bed-desk`: **crashed** n=0 wall=26 pod=pilot2-a40-7 rej={'no_init_surface': 1}  err=setup rejected: [no_init_surface] inter_object_relation_satisfaction-under_large demos: -

## inter_object_relation_satisfaction-under_small_scale

Candidate-level (log) rejections: `[goal_infeasible] unsatisfied ["['under', '?saucer#n#_#', '?teacup#n#_#']"]; goal objects {'target_teacup_wo` ×2
- `inter_object_relation_satisfaction-under_small_scale-french_fries-platter-breakfast_table`: **complete** n=2 wall=120 pod=pilot2-l40s-2 rej={}   demos: -
- `inter_object_relation_satisfaction-under_small_scale-teacup-saucer-breakfast_table`: **complete** n=2 wall=228 pod=pilot2-l40s-3 rej={'goal_infeasible': 2}   demos: -
- `inter_object_relation_satisfaction-under_small_scale-beer_glass-coaster-breakfast_table`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `inter_object_relation_satisfaction-under_small_scale-bottle_of_supplements-tray-breakfast_table`: **never_started** n=0 wall=None pod=None rej={}   demos: -

## mental_rotation

Candidate-level (log) rejections: `[grasp_failed] assisted grasping did not attach the object (both pads in contact)` ×2; `[goal_infeasible] no precomputed (grasp, goal) confirmed at this base pose: grasp_#: hold collides with the ` ×1
- `mental_rotation-cellulose_tape_dispenser`: **complete** n=2 wall=486 pod=pilot2-a40-0 rej={}   demos: -
- `mental_rotation-teaspoon`: **complete** n=2 wall=642 pod=pilot2-a40-0 rej={'grasp_failed': 2}   demos: -
- `mental_rotation-coffee_cup`: **complete** n=2 wall=521 pod=pilot2-a40-1 rej={}   demos: -
- `mental_rotation-toothbrush`: **complete** n=2 wall=504 pod=pilot2-a40-1 rej={}   demos: -
- `mental_rotation-pickle`: **complete** n=2 wall=438 pod=pilot2-a40-10 rej={}   demos: -
- `mental_rotation-potato`: **complete** n=2 wall=409 pod=pilot2-a40-11 rej={}   demos: -
- `mental_rotation-cookie_cutter`: **complete** n=2 wall=435 pod=pilot2-a40-2 rej={}   demos: -
- `mental_rotation-toy_figure`: **complete** n=2 wall=442 pod=pilot2-a40-2 rej={}   demos: -
- `mental_rotation-crab`: **complete** n=2 wall=456 pod=pilot2-a40-3 rej={}   demos: -
- `mental_rotation-garlic_clove`: **complete** n=2 wall=470 pod=pilot2-a40-4 rej={}   demos: -
- `mental_rotation-ginger_root`: **complete** n=2 wall=442 pod=pilot2-a40-5 rej={}   demos: -
- `mental_rotation-hammer`: **complete** n=2 wall=463 pod=pilot2-a40-6 rej={}   demos: -
- `mental_rotation-hook`: **complete** n=2 wall=520 pod=pilot2-a40-7 rej={}   demos: -
- `mental_rotation-mouse`: **complete** n=2 wall=512 pod=pilot2-a40-8 rej={}   demos: -
- `mental_rotation-briefcase`: **complete** n=2 wall=315 pod=pilot2-l40s-10 rej={}   demos: success
- `mental_rotation-tablespoon`: **complete** n=2 wall=298 pod=pilot2-l40s-10 rej={}   demos: -
- `mental_rotation-carving_knife`: **complete** n=2 wall=239 pod=pilot2-l40s-11 rej={}   demos: -
- `mental_rotation-teacup`: **complete** n=2 wall=223 pod=pilot2-l40s-11 rej={}   demos: -
- `mental_rotation-spatula`: **complete** n=2 wall=277 pod=pilot2-l40s-3 rej={'goal_infeasible': 1}   demos: -
- `mental_rotation-baby_shoe`: **complete** n=2 wall=220 pod=pilot2-l40s-4 rej={}   demos: -
- `mental_rotation-statue`: **complete** n=2 wall=218 pod=pilot2-l40s-4 rej={}   demos: -
- `mental_rotation-banana`: **complete** n=2 wall=219 pod=pilot2-l40s-7 rej={}   demos: -
- `mental_rotation-sunflower`: **complete** n=2 wall=215 pod=pilot2-l40s-7 rej={}   demos: -
- `mental_rotation-bell_pepper`: **complete** n=2 wall=367 pod=pilot2-l40s-9 rej={}   demos: -
- `mental_rotation-tablefork`: **complete** n=2 wall=352 pod=pilot2-l40s-9 rej={}   demos: -
- `mental_rotation-walker`: **killed** n=0 wall=None pod=pilot2-a40-3 rej={}   demos: -
- `mental_rotation-soup_ladle`: **killed** n=0 wall=None pod=pilot2-l40s-2 rej={}   demos: -
- `mental_rotation-mug`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `mental_rotation-quartz`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `mental_rotation-sandal`: **never_started** n=0 wall=None pod=None rej={}   demos: -

## object_scaling

Candidate-level (log) rejections: `[goal_infeasible] no precomputed (grasp, goal) confirmed at this base pose: grasp_#: hold collides with the ` ×2
- `object_scaling-cap`: **complete** n=2 wall=520 pod=pilot2-a40-0 rej={}   demos: -
- `object_scaling-cherry_tomato`: **complete** n=2 wall=610 pod=pilot2-a40-1 rej={}   demos: -
- `object_scaling-bottle_of_beer`: **complete** n=2 wall=340 pod=pilot2-a40-10 rej={}   demos: -
- `object_scaling-soda_cup`: **complete** n=2 wall=357 pod=pilot2-a40-10 rej={}   demos: -
- `object_scaling-bottle_of_catsup`: **complete** n=2 wall=336 pod=pilot2-a40-11 rej={}   demos: -
- `object_scaling-toy_dice`: **complete** n=2 wall=335 pod=pilot2-a40-11 rej={}   demos: -
- `object_scaling-hard_candy`: **complete** n=2 wall=361 pod=pilot2-a40-2 rej={}   demos: -
- `object_scaling-bar_soap`: **complete** n=2 wall=347 pod=pilot2-a40-5 rej={}   demos: -
- `object_scaling-lid`: **complete** n=2 wall=364 pod=pilot2-a40-5 rej={'goal_infeasible': 1}   demos: -
- `object_scaling-beaker`: **complete** n=2 wall=347 pod=pilot2-a40-6 rej={}   demos: -
- `object_scaling-beefsteak_tomato`: **complete** n=2 wall=401 pod=pilot2-a40-7 rej={}   demos: -
- `object_scaling-pack_of_cigarettes`: **complete** n=2 wall=344 pod=pilot2-a40-7 rej={}   demos: -
- `object_scaling-beeswax_candle`: **complete** n=2 wall=383 pod=pilot2-a40-8 rej={}   demos: -
- `object_scaling-potato`: **complete** n=2 wall=385 pod=pilot2-a40-8 rej={}   demos: -
- `object_scaling-can_of_soda`: **complete** n=2 wall=265 pod=pilot2-l40s-10 rej={}   demos: -
- `object_scaling-canned_food`: **complete** n=2 wall=167 pod=pilot2-l40s-11 rej={}   demos: -
- `object_scaling-bottle_of_sake`: **complete** n=2 wall=192 pod=pilot2-l40s-3 rej={}   demos: -
- `object_scaling-bottle_of_water`: **complete** n=2 wall=191 pod=pilot2-l40s-4 rej={'goal_infeasible': 1}   demos: -
- `object_scaling-bottle_of_wine`: **complete** n=2 wall=164 pod=pilot2-l40s-7 rej={}   demos: -
- `object_scaling-apple`: **killed** n=0 wall=None pod=pilot2-a40-4 rej={}   demos: -
- `object_scaling-marker`: **killed** n=0 wall=None pod=pilot2-a40-6 rej={}   demos: -
- `object_scaling-box_of_cereal`: **killed** n=0 wall=None pod=pilot2-l40s-9 rej={}   demos: -
- `object_scaling-bell_pepper`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `object_scaling-bottle_of_medicine`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `object_scaling-bottle_of_olive_oil`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `object_scaling-bottle_of_pop`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `object_scaling-jar_of_spaghetti_sauce`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `object_scaling-lemon`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `object_scaling-shampoo_dispenser`: **never_started** n=0 wall=None pod=None rej={}   demos: -
- `object_scaling-wineglass`: **never_started** n=0 wall=None pod=None rej={}   demos: -
