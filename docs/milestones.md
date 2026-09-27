# Milestones

Tagged states of this repository and what passed on them.

## `first-demo-v2.0` — the owner's v1.2 findings fixed; IK attempt gate and twin grasps; 28/28 on one commit (2026-09-27)

- **Tag / commit:** `first-demo-v2.0` → `29a94212` on `rush-20260917-2100` (`first-demo-v1.2` + the other sessions' commits of
  2026-09-26 evening + the 15 first-demo v2 commits `9ffc758e` … `29a94212`).
- **Phase A.** Each owner finding on v1.2 was fixed so that it switches on only in its situation, detected at run time, and was
  proven with a demo against its reference run before any global change (job markers `runlogs/jobs/v2_a*_20260927/`):

  | | finding (v1.2 run) | change (commit) | evidence (v1.2 → after) |
  |---|---|---|---|
  | A1 | world_vertical: the fist pushed / braced into the wall, the painting looked smaller then bigger, it did not end against the wall, an idle wait before the retreat, sampled at attempt 7 of 8 (0926-065156) | the wall contacts act in the wall plane: the fist's finger side on the low end's corner, square to its circle; the brace a finger-side stop at the goal edge; in-plane pre-contacts; the target hangs 3 mm off the wall (`9ffc758e`, `b2216a7e`; shared: `brace_targets_at_goal` skips braces the sampler placed at the goal) | 0927-012025: painting swing-back after the push 4.03° → 0; torso bent 0.028 rad → 0; head camera moved 34.3 → 0.1 mm (the "smaller then bigger"); steps after the painting stopped 96 → 27; final tilt 4.27° → 0.42°; the painting 1 cm closer to the wall; attempt 7 → 1 (sweep: 0). The arm swap is the instance's tilt sign (push arm on the low side), not a regression |
  | A2 | dishwasher close dragged the base during the follow (0926-065206) | arm-only follow whenever the arm alone reaches every waypoint and the retreat, else sequential arm → base → arm (`928d66e1`) | 0927-012451: arm-only follow of 79 steps, base 0.000 m (v1.2: hybrid, 198 steps, 0.20 m net / 0.33 m path; reference 65 steps) |
  | A3 | blender close: two pushes with a return to the untucked posture (0926-065158) | close-push search: the fingertip test in the search, the stance's own solutions pinned, the carried arm seeds the next push, same-trunk tie-break (`cd967a1c`) | 0927-012453: one push of 152 steps, hand 0.24 m (v1.2: 492 + 441 steps with 3.1 / 3.2 rad elbow swings). The sweep's attempt-0 instance needs two pushes: the second is reached directly (retreat 6 cm, transition 22 cm, largest joint excursion 0.31 rad; v1.2: 1.05 m, 3.14 rad) |
  | A4 | carton close: the final movement unlike 0922-120145 (0926-065204) | the push-close re-stance carries the arms to the next pre-contact (program-check departure arms); pushes on braked links stream at 0.10 m/s (`cd967a1c`) | 0927-012455: pushes 147 / 117 steps (reference 151 / 114, v1.2 293 / 259); final approach a 21-waypoint refinement (reference 21; v1.2 81 after a 19 cm fist lift); final push at the goal after 52 steps (reference 49, v1.2 134); the left hand back to idle as in the reference |
  | A5 | blender_eyedvd: the first open tumbles the blender and the demo was accepted (0926-073955) | detection: the `root_moved` demo gate (a free-standing articulated root within 10 mm / 0.04 rad at every step, 5 mm / 0.02 rad at the end, `149609a6`); fix: the stage-2 ride-on ends at q(follow_to) (`149609a6`) and the follow unwind stops re-reading once the free body moves (`6434682d`). The gate failed every carton-lever run: lever fist camera up (`743d9584`), lever fist slides out before it lifts (`6434682d`), a hand holding a press may touch its object during the other hand's transit (`5228f2a8`), the pinch's transit takes the witness branch when the detour's branch cannot reach the grasp (`879df877`) | the gate fails the v1.2 replay (0927-014528, tipped 2.62°). Blender replay 0927-022020: body peak 9.3 mm / 2.57° → 2.5 mm / 0.93°, end 7.8 mm / 2.33° → 1.4 mm / 0.66°. Carton lever 0927-024931: peak 16.0 mm / 1.49° → 2.5 mm / 0.59°, end 8.8 mm / 2.99° → 1.5 mm / 0.82°; the lever push moves the box 1.4 mm (was 12.0) |
  | A6 | brussels sprouts hang above the bowl after the fingers open (0926-075013) | a joint-less held object is let go at the first opening increment of the staged release (`279c3b45`) | 0927-014548: the fall starts 1 step after the first opening (was 39 steps = 1.3 s) |
  | A7 | kielbasa: the hand moves in an arch, then fixes itself (0926-074242) | a pre-grasp plan that clears the protected objects but detours more than 1.4× the via-above's path takes the via-above (`279c3b45`) | 0927-014608: grasp 529 → 391 steps, hand path 1.37 → 1.00 m, peak +19 → +12 cm. The direct plan passes 0.8 mm from the kielbasa, so a detour stays: the via-above |
  | A8 | axial: bad instances X (arc untrackable) and Z (retreat at a joint limit) | the program check replays a free-object arc push as the oracle runs it: straight leg from the pre-contact witness, the true arc, the escape retreat (`65c5e150`; the arc's limit margin is reported, not a rejection, `65f69055`) | the replay accepts X, Y and Z: X fails dynamically (the board lags the pad), so `_flipped_push_chain_check` stays and rejects X; Z's retreat now succeeds (0927-013443). Sweep: axial 3/3 at attempt 0. Neither oracle fallback (other-branch pre-contact, retreat planned transit) fired in the 43 sweep runs; both are kept |

  Tried and reverted on evidence: a closed-loop follow of a free body (the blender turned 6.9°), an edge-on brace blade that
  seeks the box face (the lever push pivoted the carton about it, 16 mm / −6.5°), and a joint-limit-margin rejection in the
  push replay (it rejected the good axial Y instances).
- **Phase B** (global changes, then one sweep):
  - **B1** (`934af755`): the random joint-limit re-solves run only from attempt 1 of an instance (re-applied by hand from
    `pilot-wave-10..12`). A rescued solution lies in another IK branch and seeded the attempt's later candidates.
  - **B2** (`29a94212`): twin library pinches. The R1Pro gripper is C2-symmetric except for the wrist camera, and its wrist is
    spherical, so a pinch turned 180° about its approach axis is the same grasp held with (q5 ∓ π, −q6, −q7). No
    library-pinch path used it before. The twin is chosen by feasibility, then the joint-limit margin (capped at 0.30 rad,
    deciding beyond 0.10 rad), then continuity (0.30 rad), and must keep the wrist camera 5 mm above the support. It is
    recorded as `<row>~r180`; `generation.grasp_twins` (default on).
  - **B3:** the other sessions' commits since v1.2 (object-scaling depth draw and drive targets, mental-rotation drive
    targets, held-start program check, `validate_instances` stance arms) get their first sweep here.
- **Sweep** `v2_sweep_20260927-1` on `29a94212` (worktree `ispatialgym-v2sweep`, 250 kg chassis, v1.0 budgets):
  - phase 1: 28/28 (the 13 most exposed aliases × 2–3 copies);
  - phase 2: 15/15;
  - 43/43 green in total.
- **Measured against the v1.2 demos:**
  - **Attempts:** attempt 0 for 21/28 aliases (v1.2: 18/28). Earlier: world_vertical 7 → 0, cork 6 → 1, apple 2 → 0,
    axial / blender close / sandal 1 → 0. Later: footstool 1 → 2, cap 2 → 3, lid 0 → 1, milk 0 → 1.
  - **Twins:** recorded for 6 aliases (shampoo and kielbasa: the row infeasible; bar soap, sprouts, cork, apple: margin).
  - **Witness margins:** the smallest IK-witness joint-limit margin rose on 13 aliases (bar soap 0.029 → 0.186 rad, sprouts
    0.025 → 0.180, sandal 0.008 → 0.102, cork 0.015 → 0.073, world_vertical 0.056 → 0.190) and fell on 7 (apple 0.109 →
    0.002, teacup 0.107 → 0.044, cap 0.153 → 0.065, lid 0.052 → 0.023, kielbasa 0.085 → 0.056).
  - **Random re-solves** in the recorded attempt: cork 92, lid 26, cap 26, sprouts 10 (all attempts ≥ 1).
  - **Smoothness:** no new joint jumps (largest per-step arm-joint change 0.041 rad in both). A twin chosen on margin swings
    the wrist further: the arm-joint path grows by 3.4–6.6 rad for bar soap, sprouts, shampoo, apple and kielbasa (cork:
    −4.9).
  - **Demo length:** 33598 demo steps in total (v1.2: 34693).

| alias | demo on this commit (collected) | other copies (phase 1) | sampling attempt (v1.2) | steps (v1.2) |
|---|---|---|---|---|
| `alignment-axial-board_game-breakfast_table` | 0927-030114 | 0927-030116, 0927-030118 | 0 (1) | 563 (563) |
| `alignment-world_vertical-painting_s070-walls` | 0927-030124 | 0927-030126 | 0 (7) | 1148 (1280) |
| `camera_relocalization-obstructed-footstool_1` | 0927-040237 | – | 2 (1) | 999 (624) |
| `camera_relocalization-standard` | 0927-040239 | – | 1 (1) | 400 (400) |
| `configuration_matching-articulation_close_large_scale-dishwasher` | 0927-030128 | 0927-031302 | 0 (0) | 567 (694) |
| `configuration_matching-articulation_close_large_scale-door` | 0927-040233 | – | 0 (0) | 2350 (2350) |
| `configuration_matching-articulation_close_small_scale-blender-breakfast_table` | 0927-032548 | 0927-032645 | 0 (1) | 857 (1566) |
| `configuration_matching-articulation_close_small_scale-carton_cdmmwy-breakfast_table` | 0927-032700 | 0927-033044 | 0 (0) | 1464 (1522) |
| `configuration_matching-articulation_open_large_scale-dishwasher` | 0927-031305 | 0927-031356 | 0 (0) | 1234 (1235) |
| `configuration_matching-articulation_open_large_scale-door` | 0927-040235 | – | 0 (0) | 2246 (2277) |
| `configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table` | 0927-031502 | 0927-031601 | 0 (0) | 936 (1068) |
| `configuration_matching-articulation_open_small_scale-carton_cdmmwy_lever-breakfast_table` | 0927-031846 | 0927-031916, 0927-032506 | 0 (0) | 1604 (1457) |
| `configuration_matching-attachment-cap-lemonade_bottle-breakfast_table` | 0927-040231 | – | 3 (2) | 927 (1006) |
| `configuration_matching-attachment-cork-wine_bottle-breakfast_table` | 0927-040228 | – | 1 (6) | 887 (1013) |
| `configuration_matching-attachment-lid-teapot-breakfast_table` | 0927-040241 | – | 1 (0) | 917 (891) |
| `configuration_matching-block_design-block_4-breakfast_table_board` | 0927-040243 | – | 0 (0) | 3936 (3772) |
| `disembedding-bar_soap-bar_soap_2_biscuit_1-breakfast_table` | 0927-034001 | 0927-034021 | 0 (0) | 929 (863) |
| `disembedding-bottle_of_milk-bottle_of_milk_1-breakfast_table` | 0927-040245 | – | 1 (0) | 881 (906) |
| `inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table` | 0927-034023 | 0927-034137 | 1 (1) | 969 (886) |
| `inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table` | 0927-040247 | – | 0 (0) | 806 (999) |
| `inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-countertop-floor` | 0927-030109 | 0927-030112 | 0 (0) | 1545 (1972) |
| `inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table` | 0927-041454 | – | 0 (2) | 960 (944) |
| `inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table` | 0927-034205 | 0927-034228 | 0 (0) | 945 (1136) |
| `inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table` | 0927-041624 | – | 0 (0) | 1638 (1653) |
| `inter_object_relation_satisfaction-under_large_scale-sandal-breakfast_table-breakfast_table` | 0927-030120 | 0927-030122 | 0 (1) | 2768 (2493) |
| `inter_object_relation_satisfaction-under_small_scale-teacup-saucer-breakfast_table` | 0927-041703 | – | 0 (0) | 891 (918) |
| `mental_rotation-banana` | 0927-041741 | – | 0 (0) | 108 (108) |
| `object_scaling-canned_food` | 0927-042025 | – | 0 (0) | 123 (97) |

- **Unit tests** on this commit: 1285 passed; the same 4 known failures as v1.0.
- **Collected demos:** `runlogs/first-demo-v2.0/` (the 28 runs above copied, `videos/`, `manifest.json`). The videos are also
  linked from `runlogs/pilot_20260915/first_demo_latest_success_videos/manifest.json`.
- **Known limits at this tag:**
  - **Carton lever pinch:** the first closing does not attach (both pads are on the flap, but no assisted grasp forms within
    60 steps); the retry attaches. Its transit into the witness branch takes the protection's via-above.
  - **Flipped-witness check:** it depends on the raw IK branch, which differs between processes. `validate_instances`
    rejects the axial Y instances that sampling accepts.
  - **Twin wrist travel:** twin grasps chosen on margin lengthen the wrist travel (above), and 7 aliases drew instances with
    a smaller witness margin than v1.2.
  - **Oracle fallbacks:** both are still in the code. Neither fired in any v2 run; removing them needs another sweep.
  - **Close pushes:** the attempt-0 draws need more than one push segment (blender 2, carton 3). Each later segment is
    reached directly from the previous one's end.

## `first-demo-v1.2` — held-object demos end without the wrist sag (2026-09-26)

- **Tag / commit:** `first-demo-v1.2` → `d53060db` on `rush-20260917-2100` (`first-demo-v1.1` + its docs commit + `d53060db`).
- **Change** (owner decision 2026-09-26 15:45): when the program's last action moves an arm that holds an object and does not
  place it (the held-object reorient), that arm keeps its hold: neither the jam probe nor the epilogue relaxes it. Before,
  each relax let the payload sag by the wrist's steady offset under the load: the SPAM demo's hand turned about 4° at the
  end, and the 0.022 rad this put into the first still window triggered the 100-step second window (150 still steps).
- **Evidence on this commit** (job `held_keep_hold_20260926`, demo-only replays of the v1.1 instances; the sampler is
  unchanged since `6c9873a4`):

  | alias | demo on this code | instance (v1.1 run) | steps: move + still |
  |---|---|---|---|
  | `object_scaling-canned_food` | 0926-154820 | 0926-151156 | 47 + 50 (was 56 + 150) |
  | `mental_rotation-banana` | 0926-155334 | 0926-152026 | 58 + 50 (unchanged; the banana barely sagged) |

  After the move the SPAM hand now moves at most 0.2 mm / 0.06° (was 5 mm / 4°); both demos succeed and settle.
- **The other 26 aliases** end with a free arm (23 retreat, 2 set_torso, 1 push_close), which the change does not touch;
  their `first-demo-v1.0` runs stand.
- **Unit tests** on this commit: 1233 passed; the same 4 known failures.
- **Collected demos:** `runlogs/first-demo-v1.2/` (the two replays copied, the 26 others linked into `runlogs/first-demo-v1.0/`).

## `first-demo-v1.1` — object scaling held with the fingers horizontal; the SPAM tin replaces the beaker (2026-09-26)

- **Tag / commit:** `first-demo-v1.1` → `6c9873a4` on `rush-20260917-2100` (`first-demo-v1.0` + the milestone docs commits +
  `6c9873a4`).
- **Change** (owner decision 2026-09-26 15:00): object scaling's fixed hold rolls the closing axis 90° (fingers horizontal,
  closing toward the robot) on all 30 object_scaling aliases, so an object pinched across its side faces the camera with that
  side; mental rotation keeps 15°. The object_scaling demo of the set is `object_scaling-canned_food` (the SPAM tin, its logo
  face toward the camera) instead of `object_scaling-beaker` (a plain white cup). The B7 visibility survey, rendered at 15°,
  applies only at that hold; at 90° every grasp row is measured by the renders.
- **Evidence on this commit** (job `object_scaling_p90_rep_20260926`, `first_demo.py run`, 250 kg chassis, both sampled at
  attempt 0):

  | alias | sampling + demo run on this code | replaces |
  |---|---|---|
  | `object_scaling-canned_food` | 0926-151156 | `object_scaling-beaker` (v1.0 run 0926-075332) |
  | `mental_rotation-banana` | 0926-152026 | its v1.0 run 0926-072342 (re-run because `held_object.py` changed) |

- **The other 26 aliases** keep their `first-demo-v1.0` runs (table below): between `b7234b1b` and `6c9873a4` only the 30
  object_scaling customs entries (`generation.hold.roll_deg`), the object_scaling defaults, `held_object.py` (held-object tasks
  only), its tests and the spec changed.
- **How the hold and the object were chosen:** `runlogs/hold_roll_20260926/` (roll +90 / −90 vs 15 on can_of_soda and beaker:
  the hold solves from the default posture at both rolls, 4/4 demos succeed, reachable goal depths unchanged), textured
  candidates at 15° in `runlogs/first-demo-v1.0/object_scaling_candidates/` and at +90° in
  `runlogs/hold_roll_20260926/candidates_p90/`.
- **Unit tests** on this commit: 1231 passed; the same 4 known failures as v1.0.
- **Collected demos:** `runlogs/first-demo-v1.1/` (the two new runs copied, the 26 others linked into `runlogs/first-demo-v1.0/`).
- **Known limits (beyond v1.0's):** from the 0.50 m hold only the goal depths 0.35 / 0.40 m pass the screening IK, at every
  roll (0.60–0.85 m fail for every grasp), so object scaling only brings the object 10–15 cm closer; the front finger crosses
  the held object (≥ 40 % stays visible, checked at init and goal).

## `first-demo-v1.0` — all 28 first-demo aliases pass sampling + demo (2026-09-26)

- **Tag / commit:** `first-demo-v1.0` → `b7234b1b` on `rush-20260917-2100`.
- **Evidence:** the code of this commit (`ispatialgym/`, `scripts/`) is identical, tree for tree, to branch `face-travel-w16`
  (`5bdbe1de`). On that code every alias below was sampled and then demoed successfully in one full run
  (`python scripts/isg/first_demo.py run --alias <alias>`, 250 kg chassis): job `pilot_face_travel_20260926-3`, 28/28,
  2026-09-26 06:51–08:18 PDT.
- The same code without the face_travel base-path style (`pilot-wave-16`, `7aefab00`) passed 31/31 as well (job
  `pilot_wave_20260925-16`: the 28 aliases, extra phase-1 copies of axial and world_vertical, and the cork replay of instance
  0921-030246).
- **Unit tests** on this commit: 1229 passed; 4 known failures deselected (`test_stance_sidecar_keeps_extras_and_recorded_configuration`,
  `test_transition_standoff_enumeration`, `test_push_under_record_and_geometry_constants`,
  `test_push_under_attempt_budget_rejects_time_budget_with_the_stage`).
- **Sampling budgets:** 8 attempts per alias, except cap 16, cork 20 and shampoo 16 (timeout 14400 s).

| alias | sampling + demo run on this code (`pilot_face_travel_20260926-3`) | drive legs (recorded style) | run on `pilot-wave-16` |
|---|---|---|---|
| `alignment-axial-board_game-breakfast_table` | 0926-073330 | no drive leg | 0926-063516 |
| `alignment-world_vertical-painting_s070-walls` | 0926-065156 | init → manipulation: recorded | 0926-063509 |
| `camera_relocalization-obstructed-footstool_1` | 0926-065208 | init → goal: face_travel | 0926-065120 |
| `camera_relocalization-standard` | 0926-074252 | init → goal: face_travel | 0926-071256 |
| `configuration_matching-articulation_close_large_scale-dishwasher` | 0926-065206 | init → manipulation: recorded | 0926-065118 |
| `configuration_matching-articulation_close_large_scale-door` | 0926-065137 | init → manipulation: face_travel | 0926-065051 |
| `configuration_matching-articulation_close_small_scale-blender-breakfast_table` | 0926-065158 | no drive leg | 0926-065110 |
| `configuration_matching-articulation_close_small_scale-carton_cdmmwy-breakfast_table` | 0926-065204 | no drive leg | 0926-065116 |
| `configuration_matching-articulation_open_large_scale-dishwasher` | 0926-073458 | init → manipulation: recorded | 0926-070445 |
| `configuration_matching-articulation_open_large_scale-door` | 0926-065140 | init → manipulation: face_travel | 0926-065054 |
| `configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table` | 0926-073955 | no drive leg | 0926-070835 |
| `configuration_matching-articulation_open_small_scale-carton_cdmmwy_lever-breakfast_table` | 0926-073750 | no drive leg | 0926-070636 |
| `configuration_matching-attachment-cap-lemonade_bottle-breakfast_table` | 0926-065150 | no drive leg | 0926-065104 |
| `configuration_matching-attachment-cork-wine_bottle-breakfast_table` | 0926-065146 | no drive leg | 0926-065100 |
| `configuration_matching-attachment-lid-teapot-breakfast_table` | 0926-065202 | no drive leg | 0926-065114 |
| `configuration_matching-block_design-block_4-breakfast_table_board` | 0926-065148 | init → manipulation: recorded; manipulation_2 → manipulation_3: recorded | 0926-065102 |
| `disembedding-bar_soap-bar_soap_2_biscuit_1-breakfast_table` | 0926-065154 | manipulation → placement (carried): recorded | 0926-065108 |
| `disembedding-bottle_of_milk-bottle_of_milk_1-breakfast_table` | 0926-065152 | manipulation → placement (carried): recorded | 0926-065106 |
| `inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table` | 0926-075013 | no drive leg | 0926-071855 |
| `inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table` | 0926-074719 | no drive leg | 0926-071633 |
| `inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-countertop-floor` | 0926-065142 | init → manipulation: face_travel; manipulation → placement (carried): recorded | 0926-065056 |
| `inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table` | 0926-073844 | no drive leg | 0926-070834 |
| `inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table` | 0926-074242 | no drive leg | 0926-070913 |
| `inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table` | 0926-074810 | manipulation → push (carried): recorded | 0926-071724 |
| `inter_object_relation_satisfaction-under_large_scale-sandal-breakfast_table-breakfast_table` | 0926-065144 | init → manipulation: recorded; manipulation → placement (carried): recorded | 0926-065058 |
| `inter_object_relation_satisfaction-under_small_scale-teacup-saucer-breakfast_table` | 0926-065200 | no drive leg | 0926-065112 |
| `mental_rotation-banana` | 0926-072342 | no drive leg | 0926-065122 |
| `object_scaling-beaker` | 0926-075332 | no drive leg | 0926-072149 |

**Collected demos:** copies of the 28 run directories above, one video link per alias and a `manifest.json` are in
`runlogs/first-demo-v1.0/`.

**Artefacts** (local to the pilot machine; `runlogs/` is not tracked): each run lives in
`runlogs/pilot_20260915/first_demo/<subtask>/<run>/` (`remote.log` / `local.log`, `inst/` with the TRO and its sidecar, `demo/` with
`meta.json` and `head_rgb.mp4`); every run is a row of `runlogs/pilot_20260915/first_demo/ledger.jsonl`; the job markers are in
`runlogs/jobs/pilot_face_travel_20260926-3/`, `runlogs/jobs/pilot_wave_20260925-16/` and `runlogs/jobs/pilot_overnight_20260926/`
(the overnight report); the videos are linked from `runlogs/pilot_20260915/first_demo_latest_success_videos/manifest.json`.

**Known limits at this tag:** world_vertical finds its instance only at its last allowed attempt (7 of 8); axial can draw two
infeasible instances, which are rejected (a flipped push pre-contact witness, program check) or caught (a retreat at a joint
limit, planned-transit fallback) rather than prevented; the face_travel style is recorded for door close / open and shampoo's
init → B drive, the other opted-in legs fall back to the recorded polyline, and carried legs stay recorded (their 1 cm schedule
probe collided in 79 of 185 checks).
