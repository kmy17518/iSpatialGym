# Milestones

Tagged states of this repository and what passed on them.

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
