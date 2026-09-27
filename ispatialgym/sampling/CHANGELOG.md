# Instance sampler changelog

Version = `ispatialgym.sampling.SAMPLER_VERSION` (also written into every `sampling_report_*.json`).

## 1.5.0 (unreleased) - one arm pose per task, scene variants without chairs, goal check without the robot

- **Three 09-16..09-24 fixes narrowed to their geometry** (owner 2026-09-25; nothing reverted). (3) `iors_small.recentre_pinches`
  (453b3065) re-centres a pinch only when the hull chord between the pads at the re-centred pose (`pinch_chord`, the
  attachment module's band) matches the library pinch width - else the original chord - within
  `pinch_recentre_width_tol_m` 5 mm / `pinch_recentre_width_tol_frac` 25 %: the nextto marker spans the same chord
  anywhere along the closing axis and is still re-centred; the teacup's rim pinch moved onto the cup's centre put the
  13.7 mm pre-shape over the interior (0925-075114 / 114310 `failed_grasp`) and keeps its 45 mm offset. Facts:
  `template_facts.grasp_pinch_recentre`; `grasps_recentred` lists the grasps actually moved. (4) `attachment._attach_clearance`
  (703a3837) takes the partner's top under the FINGERTIPS - the partner hull within `TIP_RADIAL_BAND_M` 4.25 mm of each
  tip's radial distance from the insertion axis (the tips at ± pinch width / 2; the depth may lie below the female link) -
  not the highest point within the male's radius: the 51 mm cap's tips hang over the lemonade bottle's shoulder, the
  neck lip inside them no longer floors the lift (0925-092708: 8/8 `fingers_in_partner`); the cork's tips stand over the
  rim and keep their floor; a male inserted the other way up than it rests gets the clearance as a lift ceiling; no
  partner hull under the tips = no partner floor (`_fingers_in_partner` and the finger-envelope floor still guard).
  (5) `attachment.stable_rest_quat` (superseded the same day: see "Male rest orientation" below - the library rest is kept
  unless physics toppled it).
  Also `IKChecker.track_chain(ignore_start, ground_at_start)`: the first 3.3 cm of a continuation off a contact are
  checked with the start's supports / the released object removed and the ground slab off (the IORS floor-class lift,
  0925-115205: the dispenser's spheres in the slab 1 cm off the floor).
- **The carry leg is a tracked chain too** (owner 2026-09-25; teacup 0925-141122 retreat[4] `the recorded IK witness lies
  4.58 rad away`, lift end -> pre-place 4.30 rad apart in the sidecar): the oracle reaches the pre-place by tracking the
  carry from where the lift left the hand (`execute_place`, `carry_leg "straight"`) and stays in the lift's branch, while
  the check solved `pre_place` as a fresh single from the stance posture (`_SamplerState` had no `eef_pose`, so
  `Place.targets()` returned `carry: []`) and chained the descent, the place and both retreats from THAT branch.
  `_SamplerState.eef_pose(arm)` is now the FK of `tracker.arm_end_q[arm]` on the current stance (raising when unknown, so
  `targets()` keeps its recorded-pose fallbacks; `Lift.targets()` / `Retreat._segment` start from it too, within the
  witness residual of the recorded pose; push / brace retreats are unchanged, their arm has no recorded end), and a `Place`
  is checked in motion order (`_place_order`): `carry` first as a continuation from the arm's configuration (the start pose
  prepended - `line_waypoints` leaves it out and the orientation SLERP defeats `_chain_starts`; `carry_via` is part of it),
  `pre_place` = the carry's end, `descent` from it, `place` = the descent's end, then the retreat leg. The carry is checked
  against the world QA sees (partner, neighbours and the destination support present); only the first `APPROACH_WAYPOINT_M`
  off the lift end leave the grasp's support out (`_carry_start_ignores`). A carried navigate / transition keeps `arm_end_q`
  of the arm holding the object (the oracle freezes it through carried legs) and drops the other arms, so the chain is
  tracked after a carried drive as well (IORS large-scale, touching, disembedding; IORS large-scale's `carry_end_q` seed is
  redundant now); `_keeps_arms` travels keep everything as before. A carry whose tracking stalls beyond `FREE_LEG_STALL_M`
  rejects the draw, as the oracle's finish would.
- **Fingertips in frame at every in-between pose** (owner 2026-09-25, between pilot waves 2 and 3; cap 0925-191101: the
  demo's `gripper_out_of_fov` at `reorient.waypoints[3]` - the fingertip point at v = 65 px of the 480 px image, QA's
  bound 54 px). `program_check.fingertips_in_frame` projects the fingertip point `C.fingertip(eef)` of every checked pose of
  a watched action into the stance camera (`IKChecker.camera_pose`) and requires it inside the image with QA's 5 %
  margin (`FINGERTIP_FRAME_MARGIN`) plus the execution slack of `FINGERTIP_FRAME_SLACK_M` 2 cm at the point's depth (the
  tracked motion lands within that of the recorded pose); reject `gripper_not_in_frame` naming `action.key[i/n]`, u, v
  and depth. Contact poses (grasp / place / contact, `CONTACT_POSE_KEYS`) keep the `eef_in_frame` box, which is not
  extended to in-between poses (offline it rejects three QA-passing wave-2 demos: open blender push.waypoints[2], world
  vertical brace.pre / approach / push.waypoints[2], close blender push.pre). Offline on the 22 wave-2 successes + the cap:
  the cap rejected at its reorient, the 22 accepted. `ISG_FINGERTIP_FRAME_CHECK=off` disables it.
- **Ground re-enable respects an active ignore context** (shampoo 0925-145936 / 174935 / 205946: the floor-class lift's
  first dense pose 10 mm off the floor `1/11 collide` against `floors_glwobe_0`, the task's own floor, although the ignore
  set held it). `R1ProCuRobo.check_collision(margin > 0)` toggled the ground obstacles back on for the padded query and
  left them on, undoing the `ik._Ignoring` context that had removed the floor mesh for the leg; `_ground_obstacles_enabled`
  now skips the meshes an active context holds off (`backend._held_off`, registered by `_Ignoring`). The IORS large-scale
  module's own lift check (`iors_large.lift_arm_q`) tracks the lift as a continuation of the grasp witness with the same
  ground handling (floor class: slab and ground meshes off) instead of a fresh end-pose IK. And `reorient.derived` is no
  longer checked as a second chain when the recorded key poses exist (`_skipped_target_keys`): the oracle tracks the
  recorded `waypoints`; the re-derived list from a fresh IK seed rejected the floor-class carry (0925-210939).
- **face_travel base-path style for uncarried multi-corner drives** (owner 2026-09-26, overnight plan item 6, after the
  all-green recorded wave). `navigation.plan_path(style=...)` with `generation.base_path_style` (default `"recorded"`; opted
  in: IORS large, the four large articulations, the alignment families): `"face_travel"` builds the shared
  `robot.base_path.BasePath` schedule of the path and whole-body-checks its `stride_poses(STRIDE_M)` with the same posture,
  carried object, margin and yaw-lag envelope as the polyline (`yaw_lag_envelope`, `face_travel_poses`); a schedule that
  clears is recorded `paths[].style = "face_travel"` (the oracle drives it as such), one whose stride collides falls back to
  the polyline check and is recorded `"recorded"` - never a rejection because of the schedule, so acceptance does not drop.
  Applies to the uncarried legs (`core.record_path`: IORS large init -> B; `prompt0.connect`: the large-articulation and
  wall-alignment approach); carried legs stay recorded, their schedule only probed at 1 cm and tallied
  (`FACE_TRAVEL_CARRIED_PROBE`, log `FACE_TRAVEL carried-leg probe`); transitions, base-follow and the drive with an arm
  schedule are untouched. `first_demo.py run --sample-only` re-samples an alias for the sidecar comparison.
- **IORS large: the A -> B drive is re-swept with the target placed** (owner 2026-09-26; shampoo 0926-015905, wave 8 phase 1:
  navigate[0] failed twice, 'wheel_motor_link1 in contact with the shampoo dispenser' - the recorded A -> B line passed 5.6 cm
  from the dispenser standing on the floor). `record_leg_a_b` runs before the target has a pose (it is parked), so the drive
  was never swept against where the target ends up: harmless on a table, not for a target on the floor B drives up to.
  `_leg_ab_clear_of_target`: once a (yaw bucket, grasp) gives the target's pose at B, the recorded A -> B path (its strides +
  the yaw-lag envelope, the trunk and arms it was checked with, the floor relaxation of the original check) is swept once
  with the target placed there; a collision drops that bucket (`B: the A -> B drive passes the placed target`) and the next
  draw is tried. A target clear of the drive changes nothing.
- **IORS large: a C whose place the placement camera cannot see is skipped in the C loop** (owner 2026-09-26; shampoo, waves
  6-8: 8 of 22 attempts sampled a whole instance for 7-13 min and were then rejected in `goal_check` 'place eef pose outside
  the placement image (5 %)'). The goal check's own test (`IKChecker.eef_in_frame(place_eef, camera_pose_at(C))`, ontop
  relation, not the push-under drop) is applied to each C candidate once its place IK holds - at the release pose and at the
  resting pose `goal_drop_m` (1 cm) below it, since the goal check tests the settled goal (wave 9 0926-023738 still failed it
  with only the release pose tested); such a C is counted as `C: place eef outside the placement image` and the next C is
  tried within the attempt. An instance whose C was in frame is unchanged.
- **Joint-limit re-solves from random configurations** (owner 2026-09-26; bar soap 0926-011542, wave 7 phase 1: no instance in 8
  attempts - every solution of the place pose within 2-12 mrad of `right_arm_joint6`'s limit - while copy 0926-011544 of the
  same alias and seed sampled at attempt 0 and waves 3 / 6 passed; wave 5's 8/8 failure was the same flakiness). The four
  re-solves of `IKChecker.away_from_limits` start from the remembered seeds and `q0` - near-identical starts; when they all stay
  within `JOINT_LIMIT_MARGIN`, up to `LIMIT_RETRY_RANDOM` 12 more re-solves start from random arm configurations (a generator
  seeded by the pose, so reproducible). Runs only where the candidate would otherwise be rejected `ik_joint_limit`.
  **From attempt 1 on only** (`LIMIT_RETRY_RANDOM_FROM_ATTEMPT`, first-demo v2 B1, re-applied from pilot-wave-12): a solution
  rescued from a random start lies in another IK branch and is remembered as a seed (`IKChecker.solutions`, cleared per
  attempt), so the attempt's later candidates were solved in that branch too - sandal's attempt 0 (0926-023744) ended
  without an instance, and axial's flipped push pre-contact witnesses came from it; the dishwasher close's dragging hybrid
  follow passed the program check only through such a rescue. The core sets `ikc.attempt` per attempt: attempt 0 solves as
  before the re-solves existed, every later attempt with them (verified before: sandal 0926-030813 and bar soap 0926-031126
  found their instance at attempt 0, job `ik_attempt_gate_20260926`).
- **Twin grasps: a library pinch is also considered turned 180 deg about its approach axis** (first-demo v2 B2). The R1Pro
  gripper is C2-symmetric (finger meshes, joint origins and drives; only the wrist camera is not) and its wrist is spherical,
  so `T o R_z(pi)` is the same grasp with the fingers swapped, held with (q5 -/+ pi, -q6, -q7). No library-pinch path used it
  (handles, pushes / braces and cube grasps already try both signs). `tabletop.pick_twin` compares the row and its twin per
  drawn pinch (`phase3_target`: iors_small / disembedding / attachment targets; the iors_large B stage with a C fallback;
  `iors_small.reselect_grasp`): feasibility, then the joint-limit margin (capped at 0.30 rad, deciding beyond a 0.10 rad gain),
  then continuity (the pre-grasp witness closer to the stance posture by more than 0.30 rad); ties keep the row. A twin must
  keep the wrist camera 5 mm above the support top at the grasp pose. It is recorded as `<row>~r180` with its own
  `eef_in_object` (the oracle executes it unchanged; `conventions.grasp_entry` resolves the alias), `grasps[].wrist_flip` /
  `twin_choice`, and the instance summary's `kit` counts the choices and the joint-limit re-solves. `generation.grasp_twins`
  (default on; off = the previous behaviour exactly).
- **The wrist turn of every contact by one rule** (owner 2026-09-27, first-demo v2.1; replaces B2's capped-margin rule). The
  hand is C2-symmetric about its approach axis except for the wrist camera, so wherever a generator yields a contact in both
  wrist turns - a library pinch of a free object and its twin (`tabletop.pick_twin`), the tabletop / block-design pushes, the
  braces and blades (touching, attachment, axial, wall, block-design pack / row / column, the small-articulation base-link
  braces), the push-under blade, the handle pinches (small and large articulation), the block-design cube picks (a pick and
  its twin grouped, ordered on the pick IK the chain search runs first) - `orient.choose` decides between them.
  Each turn is evaluated with the generator's own checks; each solution (and the witnesses a failed chain left - B2's rescue
  path) also offers the other turn in its shoulder / elbow branch (`IKChecker.twin_q`, verified when it is the one to try).
  Order: joint-limit safe (every configuration of the action's chain at least `JOINT_LIMIT_MARGIN` 0.05 rad from every arm
  limit) before unsafe, neither safe: more clearance first; then the smaller largest joint change from where the arm starts,
  the generator's own order within `REACH_TOL` 0.30 rad (its first turn: camera away / camera up, the library row). The first
  option that passes wins; a `downstream` check (the pushes with the brace arm locked at it) failing moves on to the next
  option. Where two turns compete the wrist camera (`<arm>_realsense_link`) must keep `CAMERA_MARGIN` 5 mm from the world
  with the contacted objects present; a generator's single turn (the camera-up lever fist, the camera-up flip) is its hard
  rule and is not re-checked. B2's 5 mm camera-above-support test applies to the library row as well. The large articulation
  orders its two pinch signs by their step-5a solutions without a flipped branch (the follow solves the grasp again from the
  stance). The axial push scores its contact configuration from the pre witness (tracked, else the plain IK) - no new
  rejection. Records: `orient` in the contact extras (every option's margin, nearest joint, reach, verdict; `against_default`
  when the chosen turn is not the generator's first), grasps `twin_choice.orient`, the summary `kit` counters
  `orient_choices`, `orient_reason.*`, `orient_source.map`, `orient_against_default`. Tests: `test_orient.py`,
  `test_grasp_twins.py`.
- **Oracle: a retreat leg that fails is replaced by a planned transit to its end pose** (owner 2026-09-26; axial 0926-061151 /
  061806, wave 15: after a successful push the escape leg reached `right_arm_joint4`'s lower limit at waypoint 3/11 from the
  push's end configuration, both attempts). Only where the leg fails `plan_failed` short of both the partial acceptance and
  the contact escape - the action failed there before - `execute_retreat` plans a transit (`arm_to`, pose goal, the leg's
  ignore set) to the leg's end pose from where the hand is; its failure fails the action as before.
  `ExecuteConfig.retreat_plan_fallback`; notes `retreat_planned_fallback`.
- **Program check: a push whose pre-contact witness sits in another branch is rejected** (owner 2026-09-26; axial
  0926-043254 wave 10 = 0926-051919 wave 12 = wave 13's copies, one instance: the push pre-contact witness 3.10 rad from the
  branch the raw IK reaches from the stance posture; trajopt never reached it (5 of 5, only through the board) and the
  oracle's closed-loop arc failed from EITHER branch - reached via the point above it, 0926-050248, or in cuRobo's branch,
  0926-053900 - while every pose of the chain had an IK solution of its own). Only in that situation (a free-object push whose
  pre-contact witness lies more than `FLIPPED_PUSH_WITNESS_RAD` 2.0 rad - a flip; world vertical's green instances keep 0.73 rad and
  were rejected at 0.6 (wave 15 0926-061803) - from the raw IK - no joint-limit re-solves - of the
  pre-contact seeded at the stance posture) the instance is rejected `approach_untrackable` and the sampler draws another
  candidate (axial's other instance passed, 0926-051915 / 051917). A dense track of the chain was tried first and is no
  proxy: it passed from the flipped witness (0926-060408 / 060411). A push whose witness is in the stance posture's branch
  is checked as before.
- **Oracle: a push pre-contact keeps cuRobo's branch where the witness's cannot be planned** (owner 2026-09-26; axial
  0926-043254 wave 10 / 0926-051919 wave 12: the push pre-contact witness 3.10 rad from cuRobo's branch - from a random
  joint-limit re-solve and from the seeded IK alike - trajopt reached it only through the board, 5 rounds, and the attempt's
  pose-goal fallback was refused; reached via the point above it, the push arc was untrackable from it, 0926-050248).
  `Motions.arm_to(accept_other_branch=True)` keeps the other-branch pose-goal plan at exactly that refusal; `execute_push`
  passes it for the pre-contact (`reach_pre_pose`): a push's legs are Cartesian and closed-loop from wherever the hand is
  (it already drops the witness whenever the pre-pose is lowered or moved). Logged `OTHER BRANCH accepted`; demo meta
  `witness_branch_accepted`.
- **Protected objects during transits** (owner 2026-09-26, overnight plan item 2; `oracle/protection.py`). Teacup
  0925-222212 / 235559 (frames 220-340): on the way to the cup the hand passed low over the saucer and shoved it ~20 cm
  against the cup before the grasp - the planner keeps the finger SPHERES >= 1 cm off, the real fingers stick out past
  them, the saucer is 1.5 cm tall; the same instance had passed in wave 2 by a hair (an old near-miss). QA recorded no
  collision because a grasp is a manipulation action from its first step: the acting arm's finger / gripper links are exempt
  for the whole action, transit included. **Protected** = the objects the goal refers to in positive atoms that the task
  placed (teacup -> saucer and cup, disembedding -> the destination tray, block design -> the blocks and the board) plus the
  action's own target until its intended contact begins; distractors (negated atoms only), source containers and furniture
  (scene names `<category>_<model>_<k>`: tables, doors, walls) keep today's handling; per-alias override
  `isg_customs.json` `"protection": {"exclude"/"include": [bddl names]}` + `_protection_note`. **Plan-time**
  (`Motions.protecting`, oracle only): every trajectory `arm_to` plans inside a transit (to the pre-grasp, the pre-place, the
  pre-contact) and the carry's tracked path are checked with the moving arm's spheres - a carried object's included
  (`R1ProCuRobo.robot_spheres_world`, forward kinematics only) - against the protected objects' boxes inflated by 2 cm
  (`INFLATION_M`, the sphere-vs-finger gap), the threshold lowered to the transit endpoints' own clearance (a pre-contact
  3 cm off a painting may be approached that close, no closer). A plan that clears runs unchanged; one that does not is
  re-planned via the point above its goal (the hand rises 12 cm where it is, a plan to 12 cm above the goal in the goal's
  IK branch, straight down; the carry: up, across, down), else re-planned directly up to 3 times. **Run-time** (QA transit
  span): during a transit any NEW contact of a robot link (the acting fingers included) or held object with a protected
  object is `protected_contact`, a protected object moving > 1 cm is `protected_moved`; the executor rolls back and the
  retry takes the via-above at once. Nothing global changes (no margin, activation distance or planner config); no protected
  object near the transit = neither check fires and the motion is exactly as before. Every firing is logged
  (`PROTECT plan-check fired: task=... action=... object=... clearance=...` / `PROTECT run-check fired`) and lands in
  `meta.protection.fired`; `meta.protected_objects` lists the set (`ISG_PROTECT=off` disables both checks).
- **A place ignores its partner only** (owner 2026-09-26: "ignore only what the motion is meant to touch, only on the leg
  where it touches"). An on / into / under / attach place ignored the partner plus `contact_supports(partner)` - its
  contact-reported supports, i.e. the table and the floor whenever the partner was awake: the teacup's place ignored
  breakfast_table + floors in wave 3 (the transit had shoved the saucer awake), only the saucer in wave 2 (saucer asleep).
  Now the partner alone, whatever its sleep state; next-to / touching places keep ignoring the shared support.
- **Braced-push ladder: brace hit vs free short** (owner 2026-09-26; world vertical 0925-193257 / 225052 / 235643: every rung
  ended "short" with brace hits 0 - the pad stops at the goal, the painting sags back 0.015-0.035 rad once it leaves; moving
  the brace outward cannot change that, the rungs never improved the residual (0.024-0.044 of the 0.08 tolerance) and wave 3's
  fourth push failed to plan twice). `Executor.push_outcome`: a short with a brace hit keeps the ladder (brace outward); a
  short with NO brace hit inside the tolerance is accepted without improvement rungs (the epilogue's settled check
  decides); out of tolerance it is `short_free` - the mirror of `push_shorten_on_free_overshoot`: the arc is lengthened and
  the pad aims past the goal by the measured sag (`execute.aim_push_past_goal`, `contact.extra["aim_past_goal_rad"]`,
  `push_extend_on_free_short`), the brace unchanged. The fallback to the in-tolerance rung when an improvement rung fails
  outright stays as the safety net (unit-tested with a forced plan failure).
- **Path sweeps cover the tracker's yaw lag** (owner 2026-09-25; shampoo 0925-211757 / 222156: the idle left finger
  brushed `wardrobe_pyuwgc_0` for 2 steps 38 steps into the carried 36 cm shuffle to the countertop stance - a 57 deg
  turn then 90 deg in place - with the base 4 mm off the path line and its yaw 6 deg BEHIND the interpolated one; the
  carried drive had been swept at the alias's 3 cm `carry_path_margin_m` along the interpolated yaws and passed).
  `navigation.stride_poses(yaw_envelope=True)` repeats every stride pose at the yaws the reference held over the last
  `YAW_LAG_LOOKBACK_M` 0.25 m of path (the angular hull of those yaws and the interpolated one, every
  `YAW_ENVELOPE_STEP_RAD` 10 deg), and `plan_path` / the program check's travel re-sweep (`_sweep_travel_record`) sweep
  that envelope - the oracle's `BaseTrajectoryTracker` (0.8 rad/s, 1.5 rad/s^2, kp_yaw 2: a ~0.5 s time constant at
  0.3-0.5 m/s) lags what the reference did over the last ~0.2 m, a long straight run has caught up. A first cut swept back
  to the whole segment's start yaw and rejected every init -> manipulation path of the shampoo's utility room (verification
  0925-230222, wave 4 aborted before a result). Each offset is capped at `YAW_LAG_MAX_RAD` 12 deg (2026-09-26: the look-back
  hull alone swept up to ~40 deg behind on the utility room's 36 cm / 57 deg shuffle and rejected most shampoo paths in wave 6
  - 0926-001742, 3 attempts in 42 min, none written, where wave 3 had sampled at attempt 0 in 407 s; the measured lag of the
  brush it guards against was 6 deg, the tracker's yaw feed-forward keeping the lag a fraction of the reference's change).
  The recorded strides are unchanged (`n_strides`); a straight path with a constant yaw sweeps the same poses as before.
- **Planner activation distance stays at 1 cm** (2026-09-25; teacup 0925-222212, wave 3: the pre-grasp transit planned at the
  1 cm `collision_activation_distance` swept the TCP 2.7 cm over the table through the 1.5 cm saucer 30 cm beside the teacup
  and pushed it 20 cm - `under(saucer, teacup)` was lost before the grasp). 2 cm was tried in wave 5 (f52992eb / 4ec17e77) and
  reverted the same night: one `MotionPlannerCfg` configures the IK solvers and the trajopt alike, so the sampler's IK
  solutions changed too (bar soap 0925-235551: 8/8 attempts `place_unreachable: ik_joint_limit`, an alias green in waves
  2 and 3), and the transit still brushed the saucer (0925-235559: 12 cm). Open: the transit needs a via-above leg (or the
  partner kept out of the arm's approach corridor), not a margin.
- **Rejected manipulation candidates move down the order** (owner 2026-09-25; cap 0925-220515: 45 phase-2 survivors,
  and all 8 attempts tried the same 6 best-scored stances - candidates 6 / 21 / 36 rejected 20 times over for the attach
  pose outside the manipulation image, no transition or the lift without IK, 39 stances never looked at, no instance).
  `attachment._candidate_order`: every rejection a candidate collected in an earlier attempt of the template adds
  `CANDIDATE_REJECT_PENALTY` (1.0, more than any depth / trunk preference) to its score, so the next attempt's window of
  six reaches the untried stances; the untried keep the depth / trunk order.
- **A grasp lift a floor demands is not optional** (owner 2026-09-25; cork 0925-195316 / 205211 and their replays
  0925-213602 / 215025: attach refused at 15-17 mm / 4.4-4.9°). The partner-top floor asked for a 30.9 mm lift of the
  library pinch (fingertips 25.9 mm below the rim at the attach otherwise), the lifted grasp had no IK at the chosen stance
  (`ik_joint_limit: right_arm_joint5 within 34 mrad of its limit`) and `_apply_grasp_height_extra` fell back to the
  library height ("IK failed with the lift; library height kept") - the instance was written with the fingers ending inside
  the neck; in the demo the re-aim leg at the pre-place already jammed (`right_arm_joint3` 0.055 rad short of its target,
  1.5 rad from any limit: a contact), the hand ended 14 mm short, the base nudge and three correction passes oscillated
  at 13-18 mm. The 09-21 success carried the full 31 mm lift (`grasp_height_extra_reason: full`). The cap's instances
  (0925-191101 / 204950) had the same fallback for the finger-envelope floor (14.4 mm: the fingertips 14 mm into the table
  at the library height; `right_arm_joint4` near its limit with the lift) - 204950 `failed_grasp`, the pads on the table.
  Now an IK failure with a lift a floor demands rejects the candidate (`Reject("lift_unreachable")`, counter of the same name) and the next stance /
  grasp is tried; the library height is kept only when no floor asked for the lift. Hardening on the oracle side from the
  same runs: `ExecuteConfig.attach_leg_soft_rot_tol` 0.01 rad and `attach_leg_limit_margin_rad` 0.05 on the re-aim, the
  attach descent and `_attach_at_place`'s correction passes (`Motions.arm_along(soft_rot_tol, limit_margin)`); a pass
  stops only when neither the position nor the rotation improves (`attach_correct_stop_rot_rad` 0.005); the program check
  tracks an attach place's carry and descent with the same `RETREAT_LIMIT_MARGIN_RAD` 0.05; the jam probe
  (`Motions._relax_if_jammed`) names the worst joint, its distance to its limits and the arm's current contact pairs
  (robot-robot included) in the log and the `jams` notes; `MotionsConfig.witness_plan_retries` 2 -> 4 (cap 0925-204950:
  three joint-plan rounds failed, the pose fallback in another branch was refused, the next attempt planned it in one).
- **Attachment partner floor: the pair is classified** (owner 2026-09-25 16:41; cork 0925-145307: item 4's band of 4.25 mm
  around each fingertip's radius met no hull vertex between the wine bottle's rim circle and its shoulder, so the "partner top
  under the tips" was a point 20 cm down the body - the rim floor vanished, the lift shrank from the 09-21 success's 31 mm
  fingertip pinch to a 14 mm pad pinch and the attach was refused twice at 15-16 mm). `_attach_clearance` now classifies the
  pair from the male's radius against the partner hull at the female link's height: `into_opening` (the male fits the
  opening: the cork) and `on_rim` (between the opening and the outer radius: a lid on a thick rim) get the 09-20 rim floor
  back (fingertips >= 5 mm above the highest partner point within the male's radius, never below the link); `over_partner`
  (the male is wider than the rim: the cap) keeps item 4's relief, except when the point found under the tips lies further
  below the link than the male's own length AND the fingers' reach (`TIP_REACH_BELOW_LINK_M` 6 cm - a wide cap's tips do hang
  over a shoulder 3-5 cm under the mouth, further than the 11 mm cap is long): then the rim floor again. Class and numbers on
  `template_facts.attach_pair_class` (r_male, r_rim, r_opening, male length, depth under the tips / rim, the rule used) for
  every attachment alias. The cork's library end topples in ~half of PhysX's settles (probe: 3/6 baseline, 0/6 on its other
  end; a self-excited rock 4 -> 6 -> 8 -> 10 -> 110 deg): the rest stays the library's and the cork runs with 20 attempts.
- **Male rest orientation: the library's, unless physics toppled it** (owner 2026-09-25; cap 0925-144739: the geometry-only
  flip of 2026-09-24 turned the flat cap open side up for a 1.6x "better" tipping margin and 8/8 draws had no reachable
  library grasp). `attachment.stable_rest_quat` keeps the library rest by default; it turns the male 180° only when the
  nudge test (`check_rest_stability`) toppled it on that end in a previous draw of the template (`_rest_flip_forced`), the
  turn is about the insertion direction's horizontal projection (so a vertical insertion - cork, cap - is never turned: the
  attachment program does no in-hand 180° turn) and at least one library grasp still approaches from above afterwards.
  Tipping margins (link centre of mass when PhysX has it) are reported in `template_facts.male_rest_orientation`.
- **Program check: legs tracked from where the arm is; travel with the arm as left** (owner 2026-09-24; block design
  0921-050358 action 15 "the sidecar's pre-pose and end-pose witnesses are in different IK branches", finish rejected 3.2
  rad away; the 371-step re-pose ceremony of a 7 cm stance hop). (1) `program_check.check_action_targets` checks an
  action's eef targets as ONE chain the way the oracle runs them: a chain whose first pose is where the arm already is
  (the retreat off the place configuration, the lift off the grasp, a `Retreat` off the previous action's end -
  `_Tracker.arm_end_q`, every family) is tracked by differential IK from that configuration (`IKChecker.continue_chain`:
  no re-solve of the first pose; a stall is finished by the IK of the end pose seeded at the stall and accepted only
  within the oracle's 0.6 rad finish bound `FINISH_MAX_EXCURSION`, else `program_check_failed`; a `Retreat` action that
  stalls within the oracle's `retreat_partial_accept_m` 4 cm is accepted as the partial retreat the oracle makes, with
  no witness recorded); a chain that leaves from a recorded single witness (`pre_place` -> descent, `pre_grasp` ->
  approach) starts at that witness (`exact(q_first=...)`), and when the approach re-tracks backward from the end's
  branch the pre witness follows the chain's start; a single pose that ends a chain (`place`, `grasp`, a brace `contact`)
  takes the chain's tracked end (`_chain_order`). Witness dicts carry `chain` / `branch` notes; the report entry
  `partial_leg`. (2) An EMPTY-HANDED travel after an arm action (the consecutive navigate / transition actions) is swept
  again with the acting arm where the last motion left it, then lifted 5 / 10 cm (`choose_departure_arms`,
  `_sweep_travel_record`: the path over its strides at its trunk, an arrival transition's sweep / drive-in and a
  departure's drive-back / sweep at their recorded s_T, the released objects at their release poses); the first
  candidate that keeps every leg free is written as `paths[].arms_q` / `transitions[].arms_q` (+ `arms_source`), the
  tracker keeps the arm through the travel, and the oracle re-poses only an arm that deviates from those joints
  (`execute_navigate`, `move_idle_arms(arms_q=...)`; `Navigate` / `Transition` targets gain `arms_q`). No candidate
  free: the records stay as swept and the oracle re-poses to the stance posture as before. `ISG_DEPARTURE_ARMS=off`
  disables (2). Report: `departure_arms[]`. Details settled by the first re-sampling passes (2026-09-25): the straight
  legs list `n` poses AFTER their start (`line_waypoints`), so a chain leaves from `2 p0 - p1` (`_chain_starts`; the
  pre witness 3.3 cm away was outside the 3 cm remembered-solution radius - the very reason the first listed pose was
  re-solved into a random branch); the continuation solver runs with the oracle's acceptance (soft waypoint tolerance
  1 cm / 0.03 rad, 100 iterations: without it 8/8 block-design attempts stalled the lift at 4 mm where the oracle's
  lift goes on); the grasped object's surface comes from the module's `grasp_ignore_objects` when it has one (IORS
  floor class: the floor, which `support_objects` leaves out); the oracle executes the converged prefix of a finishing
  leg that stalls AT a joint limit without a witness before failing it, so `execute_retreat` accepts the partial
  retreat (block design 0925-062801 retreat[15] failed outright; 0925-110340 accepted 23 mm short). Block design
  0925-110340: 521 travel steps (919 in the demo of record 0921-050358; the 7 cm stance hop 210 vs 371), no arm move at
  the three later travels, 3898 steps in all (4264).
- **Camera relocalization bucket = the geodesic** (2026-09-24). The four bucket decisions (goal sampling, the
  per-cell obstacle try, the final blocked-map path, the re-plan after the settle) read the eroded-map shortest-path
  length `kit.geodesic` returns (spec §2.3 / §6.1), not `paths[].length_m`: since the record became the line-of-sight
  shortcut (2026-09-22) its polyline is up to ~8 % shorter than the grid geodesic the pilot's dense records had, so an
  instance near a bound could change bucket. `path_length_m` stays in the summary as information; the goal candidate's
  bucket is now checked before its path is collision-checked.
- **World horizontal retired** (2026-09-23). The `alignment-world_horizontal` subtask (v1 `world_alignment-2`, alias
  `world_horizontal`) is removed everywhere: the v2 sampler `alignment_wall.py` registers world vertical only (the
  horizontal `swept_patch` extents, `level_axis` frame key and the `V1_DEFINITION_ID` 2 are gone), the v1
  `WorldHorizontalSampler` / `bddl_writer` definition 2, the oracle program / sidecar / program-check registries,
  `generation_defaults`, `v2.problem.SUBTASK_FAMILY`, `v1_bridge`, `ground_isg` (`WALL_SUBTASKS`,
  `SCALE_FALLBACK_SUBTASKS`, `frame_level` is no longer read), the iddl file, the 22 alignment BDDL problems
  (legacy `problem2` + 21 grounded aliases; the remaining alignment problems are renumbered contiguously and the
  customs `definition_id`s follow), the 21 customs entries, the `isg_reps` v1-bridge snapshot, the icl candidate /
  representative rows (+ report) and the revolution-axes table entries only that subtask used. 20 subtasks remain.
- **Neighbouring rooms loaded while sampling, nothing hidden for renders** (B10). `SimWorld.load_adjacent` defaults
  to on (`ISG_LOAD_ADJACENT=0` opts out), so the sampler, evaluation and the oracle load the same room set (collision
  and visual world parity). The v1 `seg_instance` crash (light fixture + glass in view; kitchen / garden with the
  neighbours loaded) was bisected to the RTX flag `/rtx/raytracing/showLights` that OmniGibson enables at launch: the
  light-source shapes it draws get instance ids the replicator's `InstanceSegmentationReduction` node cannot map
  (`unlabelled` regions, then an out-of-bounds index). `SimWorld._set_measurement_render_settings` sets the flag to 0;
  the light-fixture hiding (`ISG_HIDE_CATEGORY_SUBSTRINGS`, default now empty) is an emergency override only, and
  label maps no longer contain `unlabelled` regions in the affected views. Measured on house_single_floor with the
  neighbours loaded: 8 living-room views x 4-6 renders crash-free; the v1 guard still crashed on the kitchen view,
  the viewer-camera workaround of BEHAVIOR-1K #2312 and translucency off did not help. Spec 3.1 / 3.4.
- **Settled segmentation reads.** A sensor read returns a buffer one to two rendered frames old (load-dependent), and
  a scene change reaches the renderer one frame late, so v1's "three renders then read" in `SimWorld.seg_instance`
  reported the state of the *previous* measurement when measurements followed each other (the distractor-hidden
  reference silhouette equalled the occluded one; the tray counts read next carried the hidden state).
  `seg_instance` now renders >= 4 frames and continues one frame at a time until two consecutive raw-buffer fetches
  are identical (cap 8), then reads; `rgb()` reads after the same settling. ~0.3 s per measurement instead of
  ~0.17 s. Same-pose measurements are now identical across repeats and across room sets. Spec 3.4.
- **World horizontal (`world_alignment-2`, alias `world_horizontal`)**: new problem (BDDL template
  `world_alignment/problem2.bddl`, `bddl_writer`, `WorldHorizontalSampler` in `tasks/world_alignment.py`, spec entry,
  docs spec §5.9 / `task_bddl_formats.md` §6c). Same wall hinge, arc and pushing rules as world vertical; the target
  is an elongated wall item whose long axis / reference edge must end up **level**. Differences: the optional
  `generation.level_axis` is checked against the frame derived from `front_axis` / `up_axis`;
  `generation.symmetric_180` (bool or per-model dict) adds the half turn about the front axis to the target's
  `orientation_equivalences` (a plain bar is level either way); the flat-patch / front-clearance tests use the
  extents swept by the object between the largest init tilt and level (`generation.swept_patch`, default true;
  `template_meta.swept_dims`), because a tilted 1 m rail needs a 1.0 x 0.8 m clear patch, not 1.06 x 0.23 m.
  `_HingedSampler.axis_param` now builds on `target_param` (same per-model lookup for non-axis values).
- **Data**: `runlogs/world_horizontal_selection_20260912` - object rounds 1-3 for world horizontal (135 candidate
  models of 40 elongated wall / rod categories, 134 rendered, 8 reviewers, `review_contract.md` with V / L / H scores,
  `selection.md`: 18 categories / 69 valid models), recorded in `select/task_object_selection.csv` (+ xlsx / html) as
  `world_alignment-2 (world_horizontal)` target / wall / floor rows; the wall rows reuse the 78 validated regions of
  the 1.4.1 survey, annotated with the picks whose swept extent fits each region. Test spec pins
  `towel_rack/{kqrmrh, crivoa}`; smoke-tested on the Rs_int entryway region (towel bar and hook rail, one instance
  each, renders in the runlog's `smoke/`).
- **Disembedding occluded instances were impossible**: `_tune_occluder` looked the target's hull / radius up under
  its BDDL instance name (they are stored under the role key `target`) and treated the trays as obstacles, so every
  occluded draw raised `KeyError` (all 11 generated disembedding instances so far are unoccluded). Found by the
  distractor validation (`runlogs/disembedding_distractors_20260911`); fixed.
- **`generation.init_arms`** (default `untucked`, BEHAVIOR's reset pose) fixes the init arm configuration per task;
  every sampler routes through `TaskSampler.init_arms()` (camera relocalization used tucked, the held tasks' free arm
  used ready before). `generation.view_arm_fallbacks` is empty by default: no silent fallback to `ready`, a stance that
  collides with the chosen arm pose rejects the template instead, so all instances of a task share the arm pose.
  `template_meta.init_arms` records it. Within a template the whole robot configuration except the base is identical
  across instances (trunk translate / tilt come from the template's view solution).
- **Scene variants** (`sampling/scene_variants.py`). Table tasks (a fixed `table` role, or
  `generation.remove_chairs_near_tables: true`) sample in the stable scene minus the chairs within
  `generation.chair_distance` (1.0 m) of the work table(s) (`CHAIR_CATEGORIES`; `generation.chair_categories`
  overrides). The variant json is written to `<out>/scene_variants/<scene>__<hash>.json` and is the scene file the
  sampler loads, templates are merged from it (`template.json` has no chairs) and the reload checks compare against
  it. The traversability PNG is baked with the furniture, so the removed objects' footprints (canonical bounding box,
  yaw, `base_link_offset`) are freed in the loaded map where `floor_trav_no_obj_0.png` has floor and re-blocked under
  remaining objects (the table); `template_meta.scene_variant.freed_footprints` records them and
  `scene_variants.patch_trav_map_from_meta(scene, meta)` applies the same patch at evaluation time (the validator
  does; the oracle's base planner must, or it treats the chair spots as blocked). On the test dining table 6
  `straight_chair`s are removed (64 traversability cells freed outside the table footprint); the disembedding stance
  is then square to the edge (`yaw_adjust_deg` 0) and the two v1.3 validation failures are gone - the destination
  tray had been resting against the back of `straight_chair_jgoolz_1` (probe: the only non-table object within 2 cm
  of the tray), which only showed as drift with the neighbouring rooms loaded.
- Disembedding places the target on the destination tray at the reachable floor cell closest to the tray centre
  (`template_meta.dest_place_offset_m`) instead of exactly the centre: the arm's reach over a table is a narrow band
  at a fixed depth from the edge and whether it crosses the centre depends on the tray's depth; every long-edge stance
  had failed with `unreachable` because of it.
- The generic goal check parks the robot (tucked, room centre) while the goal objects are teleported and settled:
  the goal must be feasible *excluding* the robot, and a hand hovering where the goal object lands ejects it. Seen
  in disembedding after the chairs were removed: the square stance put the destination tray 0.65 m to the right,
  exactly under the right hand of a 51 deg forward-tilted torso, and 54 of 55 goal checks threw the box on the floor.
  Disembedding records whether a hand hovers in the column above a tray (`template_meta.hands_over_tray`;
  `generation.require_hands_clear` turns it into a rejection - on the 1.16 m table end with 1.2 m of untucked arm span
  it rejects every stance, so it is off by default).
- Neighbouring rooms: gardens are no longer excluded (`ISG_ADJACENT_EXCLUDE` default empty) - rgb rendering with the
  garden loaded is fine, only the sampler's `seg_instance` renders crash with it in view. A "load the neighbours but
  hide their prims" sampler mode was tried and dropped: hiding ~200 prims leaves the instance-segmentation annotator
  with no labels at all (every pixel `unlabelled`), re-creating the annotator does not recover it. The remaining way
  to keep a base off open-plan room boundaries is `RobotKit.trav_free(in_task_rooms=True)` (opt-in).

## 1.4.1 (2026-09-11) - world alignment: per-category object frames, spec wall regions

- **World vertical (`world_alignment-1`)**: the hung object's frame is no longer assumed to be "thin x, upright";
  `generation.front_axis` (axis out of the wall, default `+x`) and the new `generation.up_axis` (default `+z`) define
  it, and both (like the table-surface `front_axis` / `long_axis`) accept a dict keyed by `<category>/<model>`,
  `<category>` or `default`. `_hang_quat` maps (front, side, up) onto (wall normal, wall tangent, world up);
  `_object_dims` measures thickness / width / height along those axes. Behaviour for the default frame (paintings)
  is unchanged. Needed by the round 1-3 world-vertical categories whose canonical frame lies flat (picture,
  license_plate, guitar, hanger, dartboard, fire_extinguisher, ...; frames per model in
  `runlogs/world_alignment_selection_20260911/selection.md`).
- **Spec wall restriction**: `scenes.<scene>.fixed.wall` (allowed wall object names) and
  `scenes.<scene>.fixed.wall_regions` (surveyed segments `{room, wall, p0, p1, normal, hang_z}`) restrict / seed the
  wall spots; without them the 24-ray search from the room centre is used as before. When several walls of a room
  have usable spots, the template now binds the wall with the most spots (was: the first hit).
- **Data**: `runlogs/world_alignment_selection_20260911` - object rounds 1-3 for both world-alignment problems
  (pyrender renders of 459 models in their canonical frame, 8 reviewers, `review_contract.md`, `selection.md`) and
  the wall survey of every indoor room of the 7 scenes (`wall_candidates.{json,md}`, `maps/`), recorded in
  `select/task_object_selection.csv` (+ xlsx / html).

## 1.4.0 (2026-09-12) - disembedding distractor pools per target category, look-alike exclusion

- **Spec: `distractors` section replaces the `same_category_pool` / `distractor_pool` roles** of `disembedding-0`
  (`spec.DistractorSpec`, spec §2.2 / §5.3). `pools.<category>.same_category` lists the models of that category that may
  be drawn as same-category distractors, `pools.<category>.other_categories` names the categories whose pools form the
  other-category pool, `lookalike_groups` lists `<category>/<model>` sets that cannot be told apart from the robot's
  initial view, `families` documents how `other_categories` were derived. The loader checks that every pooled model
  is a ready model of its category, that `other_categories` reference pools, that look-alike members are pooled (or
  targets) and that every target category has a pool. `ProblemSpec.distractor_spec()` parses the section lazily from
  `raw` for scripts that rebuild a `ProblemSpec` field by field (`validate_tables.py`).
- **Sampler**: for a target `C/m` the same-category pool is `pools[C].same_category` minus `m` **and minus every model
  that shares a look-alike group with `m`**; the other-category pool is the union of the pools of
  `pools[C].other_categories`, again minus look-alikes (groups may cross categories). An empty requested pool falls
  back to the other one (logged; `template_meta.distractors.fallback`). `--variant same_category` /
  `--variant other_category` force the pool. `template_meta.distractors` records pool, requested pool, pool size,
  pool categories and the excluded look-alikes next to the drawn models.
- **Data**: `runlogs/grasp_library_20260905/scripts/build_disembedding_pools.py` builds the section for all 39
  disembedding categories (242 pooled models = every AG-valid disembedding model except one V<=1 asset; 5 visual
  families: cartons, packs, bottles, vessels, small_items; 55 look-alike groups = the exact-duplicate groups of
  `select/duplicates.md` plus same-shape variants that differ only by a small print / label / shade, judged on the
  cluster renders and grasp strips) and records everything in `select/disembedding_distractor_pools.json` and in
  the `same_category_pool` / `distractor_pool` rows of `select/task_object_selection.csv` (+ xlsx / html; every
  disembedding target row carries its pool sizes in `pair` and a `LOOK-ALIKE` flag in `duplicates`).
  Same-category pools that are always empty (single-model `jar_of_bath_salt`; `toy_dice`, `cookie_cutter`,
  `tortilla_chips`, whose members all look alike at ~20-30 px) always fall back to the other-category pool.
- `sample_instances.py` replicates templates (`templates_per_scene`) for problems with a `distractors` section (the
  pool roles that used to trigger the replication are gone).

## 1.3.0 (2026-09-12) - realised grasps, neighbouring rooms, square disembedding stances

- **Held-object grasps are realised at the library pose.** Previously the gripper was simply commanded to close on
  the free-floating object: the two R1Pro finger drives are not equally strong, the first pad shoved the object into
  the other pad and the object ended 1-3 cm / 9-14 deg away from the validated grasp with one pad off the object (the
  library's `one_finger` artefact; visible as a slanted bottle / gap on the toy figure). `_establish_grasp` now advances
  each finger by 1 mm per step until it touches the object (object kept still), holds the contact opening with a 4 mm
  press while the assisted-grasp window elapses, and leaves the fingers at the contact positions. Measured: object
  pose in the eef frame within 0.3-4 mm / 1 deg of the library pose (`summary.grasp_realised`), fingers symmetric
  (e.g. [0.021, 0.016]). Reviewer-flagged (`one_finger`) grasps are only used when no clean grasp has an option;
  grasps whose pads cannot both reach the object are rejected (`grasp_failed`) and the per-template statistics steer
  away from them.
- **Neighbouring rooms.** `SimWorld.adjacent` (rooms sharing a wall / doorway with the task rooms, from the scene's
  room-instance map dilated by 0.6 m; outdoor rooms excluded) is recorded as `template_meta.load_room_instances`
  together with the task rooms, so evaluation, validation and rendering load the neighbours (walls and furniture
  behind doorways and windows instead of sky). The sampler itself keeps loading the task rooms only
  (`ISG_LOAD_ADJACENT=1` to change): Isaac Sim 5.1's synthetic-data graph segfaults on house_single_floor as soon as
  the kitchen or garden are in the head camera's `seg_instance` view (and when their prims are hidden); bisected with
  probes, not fixable from our side. `validate_instances.py` renders rgb only for the same reason. Category-based
  fixed roles (tables, paintings) are restricted to the task rooms; cuRobo obstacles are filtered to the task rooms'
  extent + 1 m (`SimWorld.obstacle_filter`, `R1ProCuRobo.obstacle_filter`) - with the neighbours loaded the mesh cache
  (2048) overflowed.
- **Disembedding.** Round 1 of the stance search keeps the robot square to the edge on every edge (preferred first,
  then by length) and bisects the destination tray's *lateral* position until exactly `dest_visible_width` of it is in
  view; the base-yaw fallback is round 2 and is logged as a warning (`template_meta.yaw_adjust_deg`). The grasp /
  placement feasibility tables are now part of the stance acceptance, so a stance whose placement pose is blocked
  (chairs along the long edges of the dining table) falls through instead of failing the template. The distractor set
  is drawn within a footprint budget (`distractor_area_fraction` 0.45 of the source tray) - replicated templates used
  to fail every placement with three large boxes. On the test dining table the ends are the only chair-free edges and
  the 10.5 deg yaw fallback is still what makes "destination barely visible" satisfiable.
- `render_instances.py` defaults: 640 px tiles and 10 render iterations (384 px / 4 iterations looked blurred; the
  RTX real-time denoiser needs a few frames after a teleport).
- Template setup starts from a clean world: non-task scene objects are reset and the robot is parked (tucked, room
  centre) before the fixed roles are bound. Left at the previous template's stance, the robot's untucked arms hovered
  over the table and the support raster's downward rays hit them (32 "occupied" cells on the dining table), so the
  disembedding source tray had no supported position after the attachment template - a template-order dependence
  present since v1.0.
- `validate_instances.py` steps three times before evaluating `:init` (the robot base starts a few mm above the floor
  and `ontop agent floor` needs the contact; with the neighbours loaded the disembedding stance, whose rear wheels
  rest on the living-room floor across the open-plan boundary, read false after one step and true after three).
  `RobotKit.trav_free(in_task_rooms=True)` can additionally require the whole base footprint inside the task rooms
  (off by default: it removed every disembedding stance on the test table).
- Spec: §3.2 notes that BEHAVIOR's `sample_kinematics` orientation sampling reduces to canonical-up + yaw for the
  2026 assets (every `metadata.json` has `orientations: []`), §3.3 robot facts, §5.2 grasp realisation, §5.3 stance
  search order.
- Status: all 15 problems sample (30 / 30 instances, 7 min of template time after the scene loads); reload
  validation with the neighbouring rooms 28 / 30. Open: the two disembedding instances report the destination tray
  drifting 1.2 cm / 2.6 deg over 60 steps when the neighbours are loaded (identical tray poses validated with zero
  drift in v1.1 with the task room only) - to be probed for a neighbouring-room prim touching the tray.

## 1.2.0 (2026-09-11) - scene-reset guarantee, on-disk evidence, init renders

- **Every attempt starts from the stable scene.** `SimWorld.reset_scene_objects` reloads the state of every non-task
  scene object from the scene json before each attempt and right before `template.json` is written, so instances differ
  from the scene json only in the task-relevant objects and the robot. Previously only the 5 cm / 0.05 rad
  `check_scene_unchanged` guard existed; smaller creep did leak into templates (measured on the v1.1 outputs: a living
  room floor lamp 3-5 mm off in four templates, everything else identical). Reset counts are logged
  (`rejections.scene_reset`).
- `validate_instances.py` compares every non-task object of `template.json` with the stable scene json (1 mm /
  0.005 rad) and fails the template otherwise (`template_scene_deviations`).
- Per-instance summaries now carry the BDDL truth tables: `bddl_at_init` (init conditions all true, every goal atom
  with its value; the binary goal is false) and `goal_check.goal` (every binary goal atom true at the teleported and
  settled goal). Held-object tasks evaluate the complete goal expression at the goal configuration as well (previously
  `pose_reached` only); a `grasped` atom that reads false right after the teleport is logged, any other false atom
  rejects the draft.
- New `render_instances.py`: one PNG per instance in the BEHAVIOR replay layout (left wrist | head | right wrist on top,
  the replay's over-the-shoulder external camera below) plus a per-problem contact sheet under `<out>/renders/`.
  Objects the loader cannot restore yet (assisted grasps, hinge constraints) are posed from the TRO with gravity off.
  30 / 30 v1.1 instances rendered and inspected.
- Spec §3.3 corrected: the R1Pro articulation root (`base_footprint_x`) is anchored to the world by a fixed joint at
  the creation pose and never moves while the simulator plays; `set_position_orientation` only rewrites the six
  virtual joints (measured). The base sinks through the prismatic z joint when parked over an unloaded room, hence the
  parking rule.

## 1.1.0 (2026-09-11) - efficiency pass (same outputs, same checks)

Profiling (`template_meta.timings.*_kit`, `summary.phase_s` / `summary.kit`, new in this version) showed three hot
spots; each fix keeps the accuracy checks unchanged:

- **On-disk cache for the IK precomputations** (`sampling/cache.py`, `$ISPATIALGYM_SAMPLING_CACHE`, default
  `~/.cache/ispatialgym/sampling`, `off` to disable). Reach grids and grasp combination tables are keyed by the scene
  (model + loaded rooms), the placed spawned objects, the stance, the lattice, the query poses and the IK settings.
  Re-running a spec (more instances of an existing template) skips the IK entirely: IORS ontop template 80 s -> 17.5 s.
- **Held-object precomputation uses the batched IK solver's solutions** instead of exact single solves, grouped by
  trunk configuration (every new set of locked joints costs seconds). The torso embodiment now only needs its 0.2 s
  batched solver instead of the 50-90 s planner build. Template precompute: object scaling 62 s -> 19 s, mental
  rotation 232 s -> 20 s.
- **Assisted-grasp establishment without the environment pipeline**: `robot.apply_action` + a non-rendering simulator
  step (`SimWorld.control_step`) instead of `env.step`, which rendered and read six 1080^2 sensor buffers per step.
  Held instance 7.2 s -> 4.0 s. Full steps that precede predicate evaluation keep rendering: the physics -> USD sync
  that happens there is what refreshes the AABB-based predicates (a non-rendering variant made `Inside` fail).
- Bodies teleported through the physics API are woken explicitly (`SimWorld.activate`, held-object placement,
  disembedding drafts): a sleeping body keeps its old render transform and is then missing from segmentation renders.
- Default reach translates are `[0.5, 0.75]` (deeper "sitting" translates never added a reachable cell on tables and
  each costs a full IK pass); specs can still list more.
- Disembedding: an occluded draw with no room in front of the target falls back to an unoccluded instance instead of
  failing the attempt, and distractors that still occlude an unoccluded target are re-drawn (up to 3 times) before the
  attempt is given up.

Measured facts added: cuRobo's exact single IK is ~100 ms (arm) / ~750 ms (arm + torso) per call, while the batched
12-seed solver costs a few ms per pose plus a fixed refresh per distinct trunk configuration.

## 1.0.0 (2026-09-11) - first complete version

Everything in `ispatialgym/docs/instance_sampling_spec.md`:

- Task spec loader (`spec.py`), concrete BDDL writer (`bddl_writer.py`), TRO writer (`tro_writer.py`),
  grasp-library access (`grasp_library.py`), knowledge helpers (`knowledge.py`), geometry (`geometry.py`).
- Simulator layer (`sim_utils.py`): partial-room scene loading, gravity-free object parking, support-top rasters from
  raycasts (largest horizontal face, not the AABB top), BEHAVIOR-style settle / stability / scene-unchanged / drift
  checks, template saving (partial + merged).
- Robot layer (`robot_utils.py`): cuRobo whole-body collision checks, camera FK, view solving, single and batched IK
  (`R1ProCuRobo.solve_ik_batch`, 12-seed screening / 32-seed exact), reach grids, traversability, geodesics.
- Samplers for all 15 problems (`tasks/`): camera relocalization, object scaling / mental rotation (held object with
  assisted-grasp establishment and per-template precomputed goal options), disembedding (destination tray made
  "barely visible" by turning the base, occlusion tuning by seg renders), block design (Kohs blocks), articulation
  (open / close, passive-stability probe of joint angles, drive-forward reach), attachment (meta-link alignment,
  revolute symmetry equivalences, collision filter), world alignment table / vertical (COM hinges as TRO
  `constraints`), inter-object relations (ontop / under / inside / nextto / touching).
- `sample_instances.py` CLI (several problems and scenes per process), `validate_instances.py` reload validator.
- Kohs block asset pipeline (`assets/objects/kohs_block/`): Blender build, URDF -> USD conversion into
  `ispatialgym-assets/objects/toy_dice/isgblk`.

Facts established while validating on house_single_floor (all folded into defaults / the spec):

- cuRobo IK costs ~1 ms per (seed, problem) regardless of batching; screening uses 12 seeds, exact checks 32.
- R1Pro right-arm reach over a 0.80 m table from a 0.45 m stance: a strip 0.30-0.34 m deep, lateral -0.44..-0.03 m
  (robot's right); deeper trunk translates add nothing. Reach checks lock the arms tucked ("sitting").
- The head camera sits ~0.3 m ahead of the base and JoyLo's tilt is clamped, so a held object 0.55 m along the optical
  axis is out of arm reach; 0.50 m works at trunk (0.5, 0) with a 15 deg roll of the closing axis.
- Isaac Sim 5.1 `seg_instance` segfaults when a light fixture and glass are both in view; light-like categories are
  hidden during sampling (templates unaffected).
- OmniGibson's `AttachedTo` no longer filters child / parent collisions; an attached cap on a bottle neck jitters unless
  the pair is filtered.
- The toolbox lid (`toolbox/redmtl`) falls fully open from every angle in [85, 170] deg: its only passive open target is
  the 170 deg limit.
- A holonomic R1Pro keeps its articulation root at the spawn pose; every base pose lives in the six virtual joints.

- Partial-room scene loading skips scene-file objects without a room assignment, so spawned objects are saved with
  `in_rooms` = the loaded rooms and evaluation configs must supply the robot themselves (the validator does).
- `pose_settled` is not emitted for fixed scene partners (kinematic prims have no velocity to read).

Validation (`validate_instances.py`, fresh process per template, `ISpatialGymTask.load_tro_instance`): 29 / 29
instances of the 15 test problems reload with the init conditions true, the binary goal false, no drift over 60 extra
steps and the robot pose / joints within 2 mm / 0.02 rad of the TRO; the plain-teleport goal check passes wherever it
applies. Assisted grasps and COM hinges are not restored by the loader yet (reported as skipped).

Efficiency baseline (one template, 2 instances each, RTX 4090): scene load 35-45 s, cuRobo build 13 s (+50 s for the
torso embodiment when a held task needs it), template precompute 20-90 s (IK grids), instances 3-8 s for table tasks,
7 s camera relocalization, ~10 s held-object instances (assisted-grasp establishment through the environment step).
Whole test set (15 problems, 2 instances each, 3 scene loads): ~30 min sampling + ~15 min validation.
