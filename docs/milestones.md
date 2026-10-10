# Milestones

Tagged states of this repository and what passed on them.

## `v3.0-attach-contact` — attachment by contact: the pair collides until the task's rule joins it, released, touching and aligned; the oracle hovers, settles, aligns, touches down and opens at once (2026-10-10)

- **Commits** (untagged, local, branch `v3.0-attach-contact` off `5d54b735`): `a4895689` (the rule, the join modes, the filter at the
  join, the `AttachedTo` overrides, the oracle's hover / alignment / instant release), `d9937bce` (the hover settles on arrival before
  the first measurement), `ebdb93f0` (`snap` is the default join), then the touchdown before the release. Behavior-1K unchanged
  (`ddbd8ff24`). The gripper A/B test's decision (owner): keep the contact-hold closure everywhere (closing to -1 slides the part
  1-2 cm before the assisted grasp attaches); the instant opening stays; never a partial opening.
- **Why** (owner): the oracle carried the part with partly opened fingers while the assisted grasp held it (+0.125 for the whole
  descent, then +0.91 -> a second attach -> +1); the release hook and the per-step `attached` check joined the part wherever it was
  within 5 cm / 15 deg (under -1 the cap 18 mm off failed, the lid 8.9 mm and the cork 6.6 mm passed the 1 cm layout tolerance off
  their seats); the pair's collision was filtered from load, so a part could only be seated by moving it through its partner.
- **The rule** (`sampling/predicates.py`, shared by oracle and evaluator): a registered pair joins only when the part is not held
  (or is being released), in PhysX contact with its partner (`RigidContactAPI`, this step), and aligned modulo its symmetry - 4 mm
  across the seat axis, 0.1 rad, -4..23 mm along it (the axis: the female +z unless more than 45 deg from the partner-centroid ->
  female direction - the sampler's rule; the cork's female z is horizontal). The release hook joins only a part already touching
  (it defers to that step's own check, after its physics); no mid-air joins. `ISG_ATTACH_JOIN_MODE`: `snap` (the part first moved
  onto its seat at the nearest symmetric equivalent; the default since the demos below, owner 2026-10-10) or `in_place` (the
  joint at the resting pose; the cap and the cork rest on the rim beyond the 1 cm tolerance). `AttachedTo._update`
  does nothing for registered members (it joined on contact at 5 cm / 15 deg in SO(3) and snapped a lid 4 cm out of the hand);
  `_attach` / `_detach` of a registered pair go through the same joint and filter. Every join is logged (`meta["attach_joins"]`:
  trigger, contact, offsets, snap size). No other alias uses `attached`: 20 configuration_matching problems have it in the goal,
  their 19 customs aliases all use `attachment.iddl`; of the 28 v3.0 aliases only cap, cork and lid.
- **Collision**: the load-time filters are gone (sampler v1/v2, `run_isg._restore_attachment_filters`); the pair's filtered pairs
  are added at the join and removed at a detach (state loads queue the edit for the next check). Editing them while the simulation
  plays invalidates PhysX's tensor views (`update_handles()` then segfaulted); the stock route (dump state, stop, edit, play, load
  state) takes 2.7-3.3 s, keeps robot, partner and joints exactly and restarts `og.sim.current_time`. The fixed joint does not
  filter the pair (joined in the seated pose without the filter the part is thrown 83-190 mm); joined and filtered it holds within
  0.006 mm / 0.0007 rad of its partner over 60 steps (all three). The sampler's goal check filters before seating the part.
- **Step 0** (r5 instances, collision on): seated, the hulls interpenetrate - cap 20 convex pairs, deepest vertex 3.95 mm, 10.2 mm
  of lift to separate; cork 8 / 2.72 / 18.1 mm; lid 10 / 4.91 / 4.91 mm. Dropped aligned from a few mm (no robot) each rests on the
  rim: cap +10.19 mm (lateral 0.01 mm, angle <= 0.0004 rad), cork +17.9 mm (0.11-0.13 mm, 0.014-0.018 rad, tilt 1.0-2.7 deg), lid
  +4.85 mm (0.13-0.63 mm, 0.002 rad, tilt 0.1 deg); none tipped, bounce <= 0.14 mm. So `in_place` can pass for the lid only (cap and
  cork rest beyond the 1 cm tolerance) and a snap is ~10 / 18 / 5 mm.
- **The oracle's attach place**: contact hold through carry and descent to a hover pose where the lowest point of the part and of
  the fingertips is 3 mm above the partner's top under them; settle there (the descent arrives at 3 mm/step and `a4895689` let the
  part go at that speed); lateral + yaw alignment modulo symmetry (the correction regime: 4 passes, targets 2 mm / 0.02 rad); the
  touchdown (owner 2026-10-10, after the cap below): the hand straight down the seat axis at 1 cm/s until PhysX reports the part in
  contact with its partner (at most the part's clearance + 4 mm), 3 steps still, the contact hold kept; a touchdown that fails (its
  leg raises, or no contact within the depth) releases where the hand is - a plain drop; +1 in one step (a touching part is joined
  by the release step's own check); hold until the joint exists (at most 45 steps); retreat (raised by the hover). Removed:
  `clear_fingers_over_partner`, `gripper_open_short`, `_attach_at_place`, `_attach_after_release` and their knobs.
- **Demos + goal-judged replays** (`d9937bce`, r5 instances, `runs_settle`): every join came from the per-step check, in contact,
  after the release step (the release hook saw no contact at the release in every run). Hover 13.2 / 21.3 / 9.4 mm above the seat,
  fingertips 7.9 / 24.9 / 21.4 mm above the partner (the part's 3 mm binds); all aligned on arrival (<= 0.9 mm, 0.005 rad), no pass.
  - lid: `in_place` pass (4.8 mm / 0.001 rad), `snap` pass (0.13 mm, snap 4.9 mm); contact and join 1 step after the release step;
    tilt <= 0.2 deg; replays pass, identical to the demos.
  - cork: `in_place` fails as step 0 says (17.8 mm, `not_reached`), `snap` pass (0.93 mm, snap 17.9 mm in one frame); join 1 step
    after the release; replays identical (snap pass, in_place fail).
  - cap: the instant open kicks it. From the hand at rest the first attempt shot it 4.2 mm sideways in the release step and 7.8 mm
    by the next (tilt 2.7 deg); it landed flat 7.6 mm off the axis, was never joined (correctly: > 4 mm), the place's postcondition
    failed and the executor rolled back; the retry (from the restored checkpoint, the assisted grasp re-created) kicked it up 3.8 mm
    and 17.7 deg, it fell back 2.9 mm off and joined 3 steps after the release: `snap` pass (0.18 mm, snap 10.7 mm), `in_place`
    `not_reached` (10.6 mm). The replays reproduce the first attempt's sideways kick (no rollback in a replay) and fail (12.8-13.0 mm,
    never joined). The fingertips pinch the dome below its widest point; the cork and the lid do not move at the release. Under the
    old rule the release hook joined the part at the release instant, which hid any kick.
  - `a4895689` (no settle): the same outcomes; the snap cap on svl10 passed in one attempt and its replay passed.
- **The cap's release, sandbox experiment** (`ebdb93f0` + a wrapper of `_release_into_join`, `runs_exp`, snap, hover 3 and 5 mm):
  *unload* (the fingers 0.2 mm past the wider one's opening, 5 steps, the assisted grasp holding - they moved 0.3 / 1.0 mm), then +1:
  the cap fell its clearance and moved <= 1.0 mm aside, tilt <= 1.4 deg, joined 1 step after the release - the contact hold's 1 mm
  squeeze let go at once was the kick; *touchdown* (down the seat axis at 1 cm/s to contact, 3.2 / 5.2 mm in 11 / 17 steps), then +1:
  joined in the release step (the release hook deferred to the step's check), <= 1.4 mm aside, tilt 0.2 deg; *both* likewise. All
  six in one attempt (no rollback); every replay passed, identical to its demo. The owner chose the touchdown, a failed one falling
  back to a plain drop - never a partial opening (the unload is one).
- **Tests**: `d9937bce` 1487 passed, 1 skipped, the 4 pre-existing failures (`test_v2_core` x2, `test_v2_large_scale` x2) - the
  baseline's 1477 + 13 new tests in `test_attach_contact.py` - 3 removed with the old attach place. On svl nodes one more test skips
  (`a4895689` on svl10: 1486 passed, 2 skipped; `c9522ff8` on svl7: 1476 / 2).

## `v3.0-reconciled` audit fixes — no stored force is let go of at once, drops come down before an instant release, a free-standing lid follows its arc; a visibly wobbling demo is discarded (2026-10-09/10)

- **Commits** (untagged, local, on `v3.0-reconciled`): `06abb98a` (unload, push-close / grasp stops, the instant drop release, the
  partner clearance, `TorsoLoad` + the `robot_wobble` gate, QA categories), `157775e5` (the follow tests' stubs), `bdb9ec76` (a fixed-base
  handle is released unloaded; the follow's unwinds watch the torso every step), `e39c30bc` (notes reach `meta.json`; its ramp exception for
  the sandal was reverted by `9b422a29`), `9b422a29` (drops come down before the release), `855546a1` (a free-standing body's handle
  follow turns rigidly about the start hinge; the opened fingers bound a drop's descent), `f704954b` (a descent whose plan jumps is tried
  shorter), `56cb2e8e` (customs: the sandal drops from 0.15 m, the (0.95, -1.2) trunk first), `c9522ff8` (the same as the default of
  every under_large_scale alias, and for the sibling breakfast-table sandal). Behavior-1K unchanged (`ddbd8ff24`): the assisted grasp's release-by-command rule - what demo collection and evaluation share -
  is untouched; every change is in how the oracle commands the robot.
- **The owner's audit** (r4 demos, `ab47a362`) - cause, fix, before -> after on the same instance (largest |torso command - measured|,
  deg; torso speed while its command and the base are still, deg/s; head-camera motion within 10 still steps, mm; finger force on the
  support, N):
  - *door close* ("unacceptable"): the push-close's base drag drove the fist 6.7 cm on into the closed leaf; torso_joint1 sat 4.2 deg
    behind its target and sprang forward at the back-off. The arc and the drag stop at the closed leaf (within 0.004 rad) or once the
    trunk is pressed 0.3 deg; arm + torso are unloaded before the fist backs off: 4.24 -> 0.05 deg, 9.7 -> 0.6 deg/s, 58 -> 0.1 mm.
  - *dishwasher close*: the arm's unwind left it a rigid link and the closed door's load went into the torso (0.24 deg), released at the
    retreat. The unwinds take the torso along; a fixed-base handle is released only after unloading: 0.24 -> 0.07 deg, 3.2 -> 0.8 mm.
  - *door open*: the swinging leaf pulled the torso 0.38 deg past its target, released by the gripper: same fixes, 0.38 -> 0.09 deg,
    7.3 -> 2.6 deg/s, 9.2 -> 1.9 mm.
  - *blender open* (owner 2026-10-09 night): the arm-mode follow streamed the chords between the five recorded handle poses (0.4 rad
    apart), pulling the lid off its circle - torso 0.41 deg, the free-standing body tipped 0.95 deg. Re-aiming the poses for the body's
    live pose made it run away (4.6 deg, 13.9 mm: `root_moved`, the hand followed the body it tipped); an arc snapped onto each recorded
    pose jerked the hand by the grasp offset. The hand now turns rigidly about the hinge where it stood when the follow began, through the
    recorded joint values (`_follow_handle_poses`): 0.41 -> 0.025 deg, 5.4 -> 0.8 deg/s, 6.8 -> 0.26 mm, body tilt 0.95 -> 0.34 deg, slide
    2.3 -> 1.1 mm (old vs new code on r6's instance). Carton lever (the other arm-mode follow) unchanged in outcome (1341 steps, tilt
    0.67 -> 0.54 deg); blender close and carton close close with pushes and run byte-identically.
  - *brussels sprouts*: (1) not a wake issue - the assisted grasp lets go on the full open command only and the opening ramp left the
    sprout hanging between open fingers for 13 steps: every free drop now opens at once (1 step). (2) The pinch aims the fingertips 24 mm
    below the sphere; the hand pressed the table up to 196 N. The approach stops once a finger presses the support (trunk 0.1 deg off)
    and is unloaded before the closure: 0.81 -> 0.14 deg, 18 -> 3.6 mm, 196 -> 50 N (apple: 113 -> 85 N). (3) The drop comes down 4.6 cm
    (bounded by the opened fingers over the bowl's rim, 1.5 cm clear) and falls from 5.8 cm instead of 10.5.
  - *touching*: the opening pad on the partner's side swept 15 mm toward the box of tissues and shoved it 5.3 mm. A next-to / touching
    place moves away from the partner until the fully opened fingers keep 5 mm (raycasts along the finger slide axes, at most 3 cm) and
    the push keeps its recorded end: the partner does not move at the release.
  - *sandal* (instant release for every drop, owner 2026-10-09 night): released 24 cm above its rest at once it tumbled 7 deg, bounced
    14.6 deg and landed 14 mm aside; the push then turned it 60 deg (`placed_orientation`). A drop now first comes straight down from the
    recorded drop pose (`lower_drop`: until its bottom is 2 cm above the surface under it, at most 20 cm, collision-checked with only the
    held object left out, the opened fingers 1.5 cm clear; a leg whose plan jumps is tried shorter). On the instances drawn with
    `drop_height_m` 0.25 the wrist flips 5-9 cm down, so it comes 6.8 cm: released 16.7 cm up, landed 10 mm aside, success with the
    yaw 0.7 rad off (tolerance 0.8) - marginal. With the alias drawn at `drop_height_m` 0.15 and the deeper drop trunk (0.95, -1.2) first
    (a customs copy) it comes the full 11.7 cm: released 1.7 cm above its rest, landed 1.2 mm aside, push yaw 0.075 rad off - success
    (sampler: instance 0 at attempt 0 in 1420 s). The goal-judged replay passed too
    (3172 steps). Seed 3 at 0.15 likewise (11.7 cm down, landed 1.0 mm aside, yaw 0.073 rad off; replay passed); at
    `drop_height_m` 0.10 seeds 0 / 3 come 6.8 / 6.9 cm down to the same 1.6-1.7 cm release (1.2 / 1.1 mm, 0.024 / 0.105 rad; replays
    passed) (`runs/r11sandal_drop*`). The shipped customs entry has `drop_height_m` 0.15 with the (0.95, -1.2) trunk first
    (`56cb2e8e`; drawn from it unchanged, seed 0 reproduces the customs copy's demo exactly: `runs/r12sandal_shipped_s0`). `c9522ff8`
    (owner 2026-10-10) makes both the default of every under_large_scale alias (`PUSH_UNDER_DROP_HEIGHT_M` 0.15, was 0.10;
    `PUSH_UNDER_DROP_TRUNKS`, (0.95, -1.2) before (0.5, -0.6)) and gives the sibling `...sandal-breakfast_table-floor` (was 0.25) the
    same entry. Of the 28 only the breakfast-table sandal is an under alias, and it already had both. The ten aliases without an
    override (apple / marker under the desk, pencil under the sofa, sandal under the bed) have no push-under tuning; first draws on the
    new default (`runs/r13default_*`, seed 0): apple and marker on the desk or the floor and the pencil on the sofa or the floor are
    rejected before any drop (no free edge / no workspace on the desk or sofa top, no stance C or A for a floor init, the pencil's
    grasp unreachable at B). The sandal under the bed samples at attempt 0 with the drop stance (0.95, -1.2) and 0.15 m; its descent
    stops at 5.8 cm (the 11.9 and 8.9 cm legs jump 0.25 rad at the wrist), it falls 8 cm and lands 5.4 mm aside, yaw +3.7 deg - then
    the untuned push turns it 93 deg in its last third (4.4 cm short, extended): `placed_orientation` (the default 0.2 rad).
- **The gate**: `executor.TorsoLoad` (`meta.torso_load`); `robot_wobble` (on with `--demo-gating auto` in oracle mode) discards a demo
  above 1 deg. After the fixes the largest over the 28 is 0.16 deg (the apple's and the sandal's grasps).
- **Found on the way**: `goal_occluded` was not a registered QA category (the in-run occlusion gate would have raised); the sandal's
  grasp fails its finishing swing (3.02-3.07 rad, both attempts) on Turing GPUs - Titan RTX on viscam2 and svl - with the original code
  `ab47a362` as with the new, and passes on Ampere / Ada (A5000, A6000, 3090, L40S): the GPU, not the code or the rewritten instance.
- **Validation** (viscam, overflow to svl's Titan RTX nodes; sample -> demo -> goal-judged replay): A/B demos on the r4/r5/r6 instances
  (`ab5`-`ab11`); r5 (`06abb98a`, all 28) 27/28 and r6 (`bdb9ec76`, the 8 articulation aliases) 8/8 for the earlier fixes; tests
  (`f704954b`, `c9522ff8`; GPU node): 1476 passed, the 4 pre-existing failures. On svl: r5 lid / blender open / world vertical's replay as on viscam.
  Two r5 replays on viscam12 hit a cffi version clash in numba's cache load (Kit's bundled cffi) before any action; re-run, both passed.

## `v3.0-reconciled` — production (`v2.0-vectorized`), the multienv speedups (`v2.0-multienv`) and `first-demo-v3.0` on one branch; the goal-occlusion screen before generation; the 28 v3.0 aliases sampled, demonstrated and replayed through the evaluator on one commit (2026-10-08)

- **Branch / commits** (untagged, local): iSpatialGym `v3.0-reconciled` from `v2.0-multienv` (`33072fca`): `aac50a6e` merges `v2.0-vectorized`
  (release by the gripper command `998cd55b`, Brace.follow `7140a215`, Push.follow `f6385a89`, the attach release `7c48588d` /
  `75d2cd72` / `8292f12e`, the FixedJoint task rule `5913aa30`); `e39ffd35` merges `rush-20260917-2100` (`first-demo-v3.0` `e4bc51ab`,
  the pinned reach bands `452082a8`, docs); then `3882cb49` defaults, `3917264c` `4af874f2` `037e9853` `7a24b174` `0e3c1d82` `ab47a362`
  goal occlusion, `c2a0c5d0` prod_job speedups, `0fdc2259` `--stream-qa log`, `75bc1728` carton flap release, `fc3e410d` replay stream
  budget. Behavior-1K `v3.0-reconciled`: `a4e7724fe` merges `v2.0-vectorized` (release wake `a0e81fb68`, release callbacks `4eee45a42`,
  `ag_moving_link_joint_type` `2d26311f5`) into `v2.0-multienv` (numpy change detection `0fb7df693`) without conflicts; `ddbd8ff24`.
- **Conflicts** (both sides kept): `Executor.restore` (production's stale head image and physics-only transform sync; v3.0's
  release-check trim to the checkpoint step); `ispatialgym_task.py` (multienv's shared goal evaluation; production's
  `AG_MOVING_LINK_JOINT_TYPE = "FixedJoint"`); `execute_place` (v3.0 F4's `attach_hold_kept` window now spans production's release by
  command - the alignment passes, `gripper_open_short`, the second `_attach_at_place`, the open command and `_release` -, then
  `_attach_after_release`; `_attach_at_place`, `Motions.gripper` and `gripper_open_short` byte-identical to production's); `run_isg.py`
  (v3.0's `--load-only` with production's checks, its exit returns `(exit_code, result)`; production's meta keys). The reach bands are the
  v3.0 pin exactly: production's extra carton `3ef569c1` and milk `e964cb25` bands share the pinned ones' lookup factors
  (`ReachBands.find` picks by mtime) and are removed.
- **Grasp release, demo collection = evaluation:** the three Behavior-1K release commits are in; the FixedJoint rule is the task's
  (oracle, replay and policy meet it); `_patch_ag_joint_type` / `handle_grasp_fixed_ag` stay removed; the v3.0 oracle diff adds no
  privileged release or attach; `release_grasp_immediately` is left only in rollback restores, the loader's grasp setup and `run_isg_v1`.
- **Defaults:** production's (`3882cb49`): shard format `legacy`, unseeded attach sphere fit; multienv's `nodepth_nvenc` and seeded fit
  stay opt-ins.
- **Speedups** (every run below): `OMP_NUM_THREADS=1`, `OMNIGIBSON_CHANGE_DETECTION_NUMPY=1` and `ISPATIALGYM_FAST_GEOMETRY` /
  `FAST_SUPPORT` / `FAST_ASSIGN` / `SHARED_GOAL_EVAL` / `GC_FREEZE` / `BATCHED_WORLD_SYNC` = 1; `EXECUTOR_MESH_CACHE=room` off (not
  bitwise). `scripts/isg/prod_job.sh` sets them by default (`PROD_SPEEDUPS=0` opts out) and records them in the job provenance.
- **Goal occlusion before generation.** The rule is unchanged: a `target_*` / `partner_*` / `targets_*` entity at least 50% hidden in the
  settled last head-camera frame invalidates the demo (box screen, 2 cm depth tolerance; the with / without-robot panel screen for
  `dishwasher_*` / `door_*`; camera relocalization exempt), now in `ispatialgym.eval.goal_occlusion` (simulator-free, parity-tested
  against `occ.py`).
  - *Sampler* (`core.check_goal_occlusion`, right after an instance is written, before its demo): the goal entities at the program
    check's settled poses, the goal joints, the robot at the program's final configuration (the last action's retreat witness at its
    stance), one rendering step, the settled head-camera depth (`SimWorld.depth_linear` -> `ispatialgym.oracle.head_depth`: a kept extra
    annotator read until two consecutive reads agree; the camera's modalities never change), the screen. Occluded: the instance's files
    are removed and the attempt is rejected (`goal_occluded`). 2-3.5 s per instance; fail-open; `ISPATIALGYM_GOAL_OCCLUSION=off` disables.
  - *Oracle* (`demo_gating` `goal_occlusion`, default in oracle mode): the same screen on the demo's settled last frame (verdict
    `goal_occluded`, `meta.goal_occlusion`), so a miss of the prediction never becomes a demo. *Offline:* `scripts/isg/goal_occlusion_check.py`.
  - *Validation* against production's post-hoc screen (`prod_0927`, 61 instances of 5 aliases): TP 11 / FN 1 / FP 0 / TN 47, 2 not
    predicted (no end witness; target outside the predicted frame); median |predicted - measured| 0.011. The miss (milk 1409, 0.478 vs
    0.575) and the largest gaps (brussels sprouts, up to 0.34) are where the demo's end departs from the plan's. On the 28 below the
    prediction is within 0.01 for most (kielbasa 0.221 / 0.219, teacup 0.263 / 0.266) and off where the demo ends elsewhere than the
    program check (canned food 0.429 / 0.0: the reorient witness leaves the camera 0.67 m from the can, the demo 0.42 m). A wrong
    prediction costs a resample (false reject) or reaches the oracle's gate (miss).
- **Found and fixed on the way:**
  - *Carton lever* (`75bc1728`): the staged release of a pinched sheet (`edge_lid`) cannot let go by command under the FixedJoint grasp -
    the hinge blocks the radial escape the deferred second stage waits for (latent since `998cd55b`: "AG did not let go at the open
    command"). The retreat now opens fully at the hand's pose: flap error 0.0028 rad, 1 release / 0 violations.
  - *Replay through the evaluator:* strict replay stops at the first QA verdict, and nothing is intended in a stream, so contact-rich
    demos end at a "collision" on every branch - `--stream-qa log` (`0fdc2259`) records the monitor verdicts and lets the goal decide.
    The stream's QA action budget was an oracle action's 3000 steps whatever `--max-steps` said, so no demo longer than that could be
    replayed (block design 3676 / 3707, v3.0's door close 3028); it is now the episode budget (`fc3e410d`; oracle actions keep 3000).
  - *Behavior-1K `ddbd8ff24`:* `og.cleanup()` aborted on `OSError` ("Directory not empty") when Kit's temp directory is on NFS (a node
    with a full `/tmp`): a finished sample exited 139.
- **Results** (viscam via Slurm, frozen snapshots, private HOME / caches per job; sample = `sample_isg` seed 0 with v3.0's budgets; demo
  = `run_isg --mode oracle --video --video-size 540 --dev`; replay = `run_isg --mode replay --stream-qa log --no-goal-override` on a copy
  of the post-demo instance, judged by the goal). Round r4, all on `ab47a362` + Behavior-1K `ddbd8ff24` (block design's replay on
  `fc3e410d`, which changes only the replay / policy budget): **28/28 sampled, demonstrated and replayed**. The sandal row is
  `r4sandal_s0`: r4's own sandal draw failed at its grasp (plan_failed: the sidecar's pre-pose and end-pose witnesses 3.04 rad apart in
  different IK branches - the signature of v3.0's canary red), and viscam's nodes run the IORS-large search ~2.5x slower than v3.0's pods
  (107 vs 43 s per exhausted C try), so the sandal ran with its attempt budget 2400 -> 6000 s through a customs copy (`ISG_CUSTOMS`; the
  oracle reads the shipped list): seeds 0 / 1 / 2 all green (2753 / 2686 / 2848 steps, replays 3304 / 3224 / 3418), and v3.0's own sandal
  instance on the same code gave 2754 / 3305. Round r3 (`4af874f2`, replays on `0fdc2259` / `fc3e410d`) went 26/28 before the carton fix
  (carton lever red; its sandal sampler spent three 2400 s attempts).

  | alias | v3.0 | r4 demo | r4 replay | occlusion predicted / measured | r3 demo / replay |
  |---|---|---|---|---|---|
  | block design | 3562 | 3676 | 4412 | 0.087 / 0.108 | 3707 / 4449 |
  | door close | 3028 | 2477 | 2973 | 0.0 / 0.0 | 2556 / 3068 |
  | sandal (`r4sandal_s0`) | 2754 | 2753 | 3304 | 0.075 / 0.081 | no instance |
  | door open | 2201 | 2222 | 2667 | 0.102 / 0.0 | 2222 / 2667 |
  | touching | 1638 | 1624 | 1949 | 0.029 / 0.035 | 1624 / 1949 |
  | dishwasher open | 1557 | 1557 | 1869 | 0.0 / 0.0 | 1557 / 1869 |
  | shampoo | 1510 | 1571 | 1886 | 0.0 / 0.0 | 1571 / 1886 |
  | carton lever | 1316 | 1341 | 1610 | 0.099 / 0.091 | failed (1077) |
  | carton close | 1265 | 1262 | 1515 | 0.008 / 0.010 | 1262 / 1515 |
  | world vertical | 1050 | 1050 | 1260 | 0.0 / 0.0 | 1050 / 1260 |
  | camera relocalization, obstructed | 999 | 624 | 749 | exempt | 624 / 749 |
  | kielbasa | 942 | 946 | 1136 | 0.221 / 0.219 | 946 / 1136 |
  | bar soap | 940 | 756 | 908 | 0.0 / 0.0 | 789 / 947 |
  | lid | 938 | 1018 | 1222 | 0.0 / 0.011 | 1018 / 1222 |
  | blender open | 936 | 938 | 1126 | 0.055 / 0.024 | 938 / 1126 |
  | brussels sprouts | 934 | 945 | 1134 | 0.006 / 0.003 | 945 / 1134 |
  | milk | 904 | 987 | 1185 | 0.080 / 0.008 | 987 / 1185 |
  | cap | 899 | 783 | 940 | 0.0 / 0.0 | 899 / 1079 |
  | cork | 893 | 914 | 1097 | 0.0 / 0.0 | 914 / 1097 |
  | blender close | 885 | 881 | 1058 | 0.042 / 0.043 | 881 / 1058 |
  | apple | 879 | 884 | 1061 | 0.056 / 0.056 | 884 / 1061 |
  | teacup | 872 | 874 | 1049 | 0.263 / 0.266 | 874 / 1049 |
  | marker | 681 | 698 | 838 | 0.014 / 0.004 | 698 / 838 |
  | camera relocalization, standard | 563 | 400 | 480 | exempt | 400 / 480 |
  | dishwasher close | 546 | 546 | 656 | 0.162 / 0.0 | 546 / 656 |
  | axial | 518 | 518 | 622 | 0.0 / 0.0 | 518 / 622 |
  | canned food | 123 | 123 | 173 | 0.429 / 0.0 | 123 / 173 |
  | banana | 108 | 108 | 158 | 0.149 / 0.193 | 108 / 158 |

- **Tests** (`fc3e410d`, GPU node): 1465 passed, 2 skipped, 4 failed - the same 4 fail on both parents' tips
  (`test_v2_core` stance sidecar / transition standoff, `test_v2_large_scale` push-under constants / attempt budget).
- **Not done:** nothing pushed; production not switched (`isg_prod`, its snapshots and `isg_home/tmp` untouched). Each sandbox job
  starts Kit cold (~5 min before the first frame, up to 7 min of scene import); production's warm worker homes do not pay that.

## `first-demo-v3.0` — the owner's five findings on v2.2 fixed, each only in its own situation; 28/28 green, scoped under the owner's 23:05 rules (2026-09-30)

- **Tag / commit:** `first-demo-v3.0` → `e4bc51ab` on `rush-20260917-2100` (`first-demo-v2.2` and its docs / manifest / TODO commits +
  `35be7cba` F4, `91ebb50b` F2, `1178c510` F1, `123179c9` F5 - oracle - and `e4bc51ab` F3 - sampling; one commit per fix). After it:
  `7ba60e3b` the reel builder (`scripts/isg/build_demo_reel.py`), the latest-success manifest (`fe9a6e69`), these docs and TODO.md.
- **How the 28 demos were made** (owner rules 2026-09-27 23:05, as v2.2): each fix declares its situation predicate and its code paths and is
  otherwise a strict no-op that adds a fired / not-fired note. Every predicate was evaluated in shadow over the 28 v2.2 release instances
  (`runlogs/first-demo-v3.0/reports/scoping.md`): 9 aliases exposed - world vertical (F1); door open (F2, F5); carton lever (F3, a sampling
  change: re-sampled); cork, cap, lid (F4); marker, dishwasher open, dishwasher close (F5). Phase A proved each fix on its v2.2 instance
  (job markers `runlogs/jobs/v30_f*_20260930/`); the canaries `v30_canary_20260930` on `e4bc51ab` went 42/43 (the red is pre-existing,
  below); the freeze `v30_sweep_20260930-1` on `e4bc51ab` re-ran the 9 exposed aliases ×2: 18/18 green (both copies step for step on
  seven aliases; the cap 899 / 812 and the lid 938 / 909 drew differently on the two machines). The other 19 aliases keep their v2.2
  release run (made on `e6c62b73` / kielbasa `bc78d77f`); `made_on`, the scoping evidence and the arm are in every manifest entry
  (`runlogs/first-demo-v3.0/manifest.json`).
- **The fixes** (each: the situation, the code path; Phase A = the fix on its v2.2 instance, a demo-only replay - F3 a fresh sample - with
  head + overhead video):

  | | finding (v2.2 run) | situation -> change (commit) | Phase A proof (v2.2 -> v3.0) |
  |---|---|---|---|
  | F1 | world vertical 0928-043205: the transition tucked both untucked arms (the wall stance's recorded posture), the brace and the push started from tucked | a WORKING arm the transition would put into the arriving stance's NAMED posture (`arm_posture`, no transition `arms_q`) whose next action there is a brace / push stays as it is when, the kept arms as they are: the sweep (trunk, drive-in) is collision-free with the protection distance, each kept arm's straight joint route to its contact's standoff / first witness is collision-free (the earlier arm at its contact) and the later arm as it is clears the earlier arm's contact witnesses; collision checks, FK and differential IK only (`execute.keep_contact_arms`, `Executor.stance_contact`; `1178c510`) | 0930-175423 (local): both arms kept; transition 226 -> 65 steps; before the brace (right) 10.35 -> 7.11 rad joint path, 2.32 -> 0.82 m hand path; before the push (left) 8.70 -> 5.54 rad, 2.26 -> 0.63 m; demo 1149 -> 1050 steps |
  | F2 | door open 0928-042225: after the first release the ramp onto `arm_q_retreat` turned joint 1 by 3.4 rad (the hand rose ~0.65 m and came back), the re-grasp plan swung back | a release / move / re-grasp whose `arm_q_retreat` lies > 1.0 rad from the measured joints after the back-off (70 logged ramps of 38 runs: 0.012-0.595 rad; this instance 3.505) and whose drive holds the retreat posture keeps the arm where the back-off left it when the drive along `base_path` with it is collision-free (margin 0.01 m) and it keeps 0.02 m from the self-collision padding; either check failing ramps as before (`execute.keep_retreat_arm`; `91ebb50b`) | 0930-175755 (pod): kept at the first release (3.436 rad), the other two segments ramp as before (0.163 / 0.485 rad); largest joint excursion first release -> second grasp 3.505 -> 0.58 rad, hand rise 0.684 -> 0.005 m, base_follow 1882 -> 1316 steps; demo 2876 -> 2201 steps |
  | F3 | carton lever 0928-043925: the grasp transit swung the hand ~56 cm right and 20 cm up (brace at joint 2's limit, pinch in the other shoulder solution; the oracle's detour needs the brace's solution to reach the pinch) | the two-flap lever brace is re-solved nearest the pinch's pre-grasp witness with a 0.05 rad joint-limit margin, the lever and the pinch re-checked beside it, when its solution does not reach the pinch the oracle's way (retreat, up-across-down detour, approach - differential IK) or stands within 0.05 rad of a joint limit (`articulation_small.brace_in_pinch_solution` / `lever_detour_reaches`; `e4bc51ab`) | 0930-182056 (fresh sample, pod): re-solved 3.03 rad away, margin 0.003 -> 0.099 rad; the transit takes the 'cartesian detour 6 cm above'; brace -> grasp excursion 3.167 -> 1.023 rad, hand lateral excursion 0.58 -> 0.001 m, grasp 425 -> 194 steps; demo 1604 -> 1316 steps. Control 0930-180217 (the detour check alone): the v2.2 instance again, identical to v2.2 |
  | F4 | cork 0928-041525: seated within 0.7 mm / 0.4 deg, then two jam relaxes of the loaded wrist (after the descent, in the correction pass) moved it; attached 3.8 mm / 1.7 deg | an attach place with the part in hand: the holding arm joins `keep_hold_arms` from the descent until the attachment exists (`execute.attach_hold_kept`; `35be7cba`) | 0930-174300 (local): offset before the attach 3.57 mm / 1.66 deg -> 0.93 / 0.46, at the joint 3.81 / 1.66 -> 0.93 / 0.46, correction passes 1 -> 0, holding-arm relaxes from the descent to the attachment 2 -> 0 (in the whole place 3 -> 1: the one left ends the move to the re-aimed pre-place, before the descent); place 160 -> 112 steps |
  | F5 | nextto marker 0928-050212: the grasp transit dropped the hand ~9 cm below the pre-grasp beside the robot, then rose to it: two descents | a plain grasp transit whose planned hand path (FK before it streams) dips > 7 cm below both ends (the 28 v2.2 transits: free-object pinches 0-5.6 cm, the marker 9.0, the handle grasps 13-25) runs as a Cartesian leg - straight, else up-over-down; tracked forward, else back from the plan's end after an in-place adjustment <= 0.6 rad moving the hand <= 2 cm - when it is collision- and protection-checked, ends in the plan's IK branch and the approach tracks from it; otherwise the plan runs as before (`Motions.transit_dip_gate`; `123179c9`) | 0930-181938 (local): the straight leg tracked back from the plan's end (in-place adjustment 0.32 rad / 1.9 mm first); hand dip 8.7 -> 0 cm, grasp 365 -> 253 steps; demo 740 -> 681 steps |

- **Scoping in the freeze** (each fix fired exactly where the shadow said, in every replay, fresh sample and freeze copy): F1 on world
  vertical (both arms kept); F2 on door open (the first release); F3 on the carton lever's fresh samples (its v2.2 instance's replays run
  the recorded brace: 1604 steps, as v2.2); F4 on cork, cap and lid; F5 on the marker (backward straight leg), door open (forward,
  13.2 cm planned dip) and dishwasher close (forward, 15.4 cm). Dishwasher open is exposed (its plan dips 24.5 cm) but no leg qualifies -
  the four variants stall 13-15 mm short - so it runs v2.2's plan (1557 steps, as v2.2). Strict no-op evidence for the others: blender
  close (F1's literal predicate matched it) replayed on `e4bc51ab` = v2.2 step for step (885); the canary replays of the non-exposed
  fragile aliases repeat v2.2 (carton close 1265, milk 904, shampoo 1510; block design 3580 / 3567 vs 3562, run-to-run as before; the
  sandal 2756 / 2756 vs 2754: every action up to the push-under step for step, then the push 290 vs 289 and the settle 460 vs 459 - no
  v3.0 path wraps the push, F1 'no arm to re-pose', F5 evaluated 6.1 cm <= 7 cm).
- **No-op replays** (evidence only, after the freeze, local RTX 4090; `runlogs/first-demo-v3.0/reports/noop_replays.md`): the 14
  aliases neither exposed nor canaried, replayed from their v2.2 instances on `e4bc51ab` (`v30_noop_replay_20260930`): 14/14 green,
  13/14 step for step as v2.2 (every action and the settle), no v3.0 note fired. Axial's push came out 341 steps vs 343 (515 vs 518 in
  all, settle 86 vs 87). A/B `v30_noop_ab_20260930` on `bc78d77f`, local RTX 4090: the axial replays as 515 / [48, 341, 40] / 86 and
  the sandal as 2756 / push 290 / 460 - equal per action to their `e4bc51ab` replays. Both differences from the v2.2 runs (fresh sample
  + demo on a pod) are replay / machine variance at the base, not v3.0; every earlier axial run (three fresh runs, two commits, two
  pods) gave 518.
- **Results vs v2.2** (`runlogs/first-demo-v3.0/reports/measurements.md`):
  - *F1 world vertical:* transition 226 -> 65 steps; the tuck skipped for both arms (the sweep, the route to the contact and the later
    arm's hold clear); before the brace (right) 10.35 -> 7.11 rad / 2.317 -> 0.815 m hand path / 383 -> 282 steps; before the push (left)
    8.70 -> 5.54 rad / 2.259 -> 0.629 m / 592 -> 511 steps.
  - *F2 door open:* largest joint excursion first release -> second grasp 3.505 rad (joint 1) -> 0.58 rad (joint 7); hand rise 0.684 ->
    0.005 m; base_follow 1882 -> 1316 steps.
  - *F3 carton lever:* brace -> grasp excursion 3.167 rad (joint 1) -> 1.023 rad (joint 7); hand lateral excursion 0.58 -> 0.001 m;
    pre_grasp_transit 'witness branch (the detour's branch does not reach the grasp)' -> 'cartesian detour 6 cm above'; grasp 425 -> 194
    steps.
  - *F4 attach places:* cork offset before the attach 3.57 mm / 1.66 deg -> 0.93 / 0.46, at the joint 3.81 / 1.66 -> 0.93 / 0.46,
    holding-arm relaxes from the descent to the attachment 2 -> 0, correction passes 1 -> 0; lid 0.91 / 1.01 -> 0.27 / 0.28, relaxes
    1 -> 0; cap (another draw) 0.69 / 0.32 -> 0.29 / 0.03, relaxes 0 -> 0. Jams in the whole place (the carry leg and the move to the
    re-aimed pre-place included, before F4's window): cork 3 -> 1, lid 3 -> 2, cap 0 -> 1.
  - *F5 grasp transits:* marker hand dip 8.7 -> 0 cm, grasp 365 -> 253 steps; door open 12.9 -> 0 cm (239 -> 243); dishwasher close
    15.2 -> 0 cm (257 -> 233); dishwasher open 24.1 cm, unchanged (no leg qualifies).
  - *Demo length:* 33441 demo steps in total (v2.2 34617). Shorter: door open 2876 -> 2201, carton lever 1604 -> 1316, world vertical
    1149 -> 1050, marker 740 -> 681, cork 950 -> 893, dishwasher close 567 -> 546, lid 950 -> 938; cap 864 -> 899 (another draw,
    attempt 2 -> 1); dishwasher open 1557 and the 19 kept aliases unchanged.
  - *Attempts:* attempt 0 for 18/28 aliases (v2.2 18/28); cap 2 -> 1, the others unchanged.
  - *Smoothness* (largest per-step arm-joint change / arm joint path, rad): the set's largest stays 0.048 (door close, kept). Re-runs:
    door open 0.036 -> 0.038 / 53.8 -> 38.4, world vertical 0.020 / 24.1 -> 17.7, carton lever 0.030 -> 0.037 / 38.0 -> 34.1, cork
    0.038 -> 0.030 / 16.1 -> 15.8; up on dishwasher close 0.036 -> 0.041 / 12.1 -> 14.3 and marker 0.027 -> 0.029 / 15.5 -> 16.2 (the
    Cartesian legs), cap 15.5 -> 17.7 (another draw).
  - *Wall time* (job, s): door open 3089 -> 1128, world vertical 855 -> 563, carton lever 1568 -> 1288, cork 2226 -> 1686, cap 2689 ->
    2232, dishwasher open 1411 -> 557; up on marker 784 -> 1074, lid 883 -> 1538, dishwasher close 645 -> 675 (pod-to-pod demo walls
    vary with the GPU).
  - *Success:* 28/28 (v2.2 28/28).
- **Canaries** (`v30_canary_20260930`, `e4bc51ab`): the 9 exposed aliases + the fragile sandal, shampoo, milk, block design and carton
  close - replays of their v2.2 release instances on type-A pods (L40S / RTX 6000 Ada / L40) 14/14 and type-B (RTX 5090 / RTX 4090)
  14/14, fresh samples 13/14, plus the blender close no-op replay. Determinism: the fresh samples of the exposed aliases draw their v2.2
  instance (seed 0, sampling unchanged for them; the carton lever is re-sampled by F3) and the demos repeat across pod types (door open
  2201 on three pods, cork 893, world vertical 1050 / 1049). Red: the sandal's fresh sample (0930-183356, local RTX 4090) - plan_failed at
  its grasp (the protection's via-above and its finishing swing refused, -0.0228 / -0.0149 m, 3.05 / 3.03 rad), v2.1's / v2.2's known
  pod-dependent signature, on another instance than v2.2's (the IORS-large sampler is time-budgeted, its draw depends on the machine); no
  v3.0 path acted (F1 'no arm to re-pose', F5 5.9 cm <= 7 cm); the same instance on `bc78d77f` on the same GPU fails identically
  (`v30_sandal_ab_20260930`, 0930-192835) -> pre-existing.
- **Unit tests** on `e4bc51ab`: 1409 passed, 1 skipped; the same 4 known failures (identical on `bc78d77f`). New: `test_v30_fixes.py`
  (15 tests: each fix's predicate, its code path and its not-fired note).
- **Collected demos:** `runlogs/first-demo-v3.0/` (28 runs, `videos/` = head camera, `videos_overhead/` = the `--dev` overhead camera,
  `manifest.json` with `arm` / `made_on` / `scoping` per entry, `reports/` with the decision log and the driver scripts). Reels
  `runlogs/tmp/first-demo-v3.0_all_demos_with_overhead.mp4` (+ `.txt` index) and `runlogs/tmp/first-demo-v3.0_all_demos.mp4` (18:35, one
  chapter per demo). The head videos are also linked from `runlogs/pilot_20260915/first_demo_latest_success_videos/manifest.json`.
- **Pinned for reproduction** (pushed to GitHub):
  - code: tag `first-demo-v3.0`;
  - reach bands the runs read: commit `452082a8` on this branch, the tag plus the 8 `.npz` it lacks (shampoo floor and countertop,
    the blender cwkvib / eyedvd and carton cdmmwy handles, bar soap, milk, sandal: only their `.json` were committed, the pods got
    the `.npz` with remote_run's working-tree sync). Check out the tag, then
    `git checkout 452082a8 -- runlogs/reach_bands_20260915/bands/`;
  - simulator: unchanged, tag `ispatialgym-first-demo-v2.0` of `kmy17518/BEHAVIOR-1K` (`14e06b710`; `setup.sh` clones the
    branch, `ISPATIALGYM_B1K_BRANCH=ispatialgym-first-demo-v2.0` pins the tag). Assets, robot-asset checksums and package
    versions as at `first-demo-v2.0`;
  - demos: branch `first-demo-v3.0-artifacts` / tag `first-demo-v3.0-demos`, an orphan branch with no code history. It holds
    the collection and `runlogs/first-demo-v3.0/pins/PIN.json`: package versions, the sampling budgets, robot-asset checksums,
    the local sampler cache, the ledger rows and job markers; its README has the rebuild commands.
- **Known limits:**
  - **Dishwasher open** still dips 24.5 cm in its grasp transit: the straight and up-over-down legs both stall 13-15 mm short, forward and
    backward, so F5 keeps the plan (the owner's fallback).
  - **F5's legs** trade joint path for the hand's height on two transits: dishwasher close +2.2 rad (largest step 0.041 rad), marker
    +0.7 rad; the marker's backward leg starts with a 0.32 rad / 1.9 mm in-place adjustment at the idle arm's shoulder singularity.
  - **F4's window** is the descent to the attachment only: the jam probe may still relax the holding arm at the end of the carry leg or
    of the move to the re-aimed pre-place (cork 1, lid 2, cap 1 in the freeze runs; 0.007-0.02 rad of joint target each).
  - **F1, F2 and F5 decide at run time:** the shadow bounds exposure from the recorded instances; their collision / tracking checks run
    in the demo (deterministic: collision checks, FK, differential IK - no cuRobo random stream).
  - **F3** covers the two-flap lever brace only (the `two_flap_v4` path); the other brace samplers keep their solutions.
  - **Pre-existing:** the sandal's grasp via-above on some draws / pods (fails identically on `bc78d77f`); block design's run-to-run
    step counts; a replay of a recorded instance can run a closed-loop push a step or two differently from its fresh run (axial, the
    sandal; identical on `bc78d77f`).
  - **Not in v3.0** (owner): camera relocalization's goal tilt, the sandal's via-above approach, D6 (the real left-arm bands).

## `first-demo-v2.2` — the left arm as one more dimension of the orientation chooser; mirrored left bands; 28/28 green, scoped under the owner's 23:05 rules (2026-09-28)

- **Tag / commit:** `first-demo-v2.2` → `bc78d77f` on `rush-20260917-2100` (`first-demo-v2.1` `5d6a7ff5` + `2b6e0904` the left arm +
  `e6c62b73` / `db1b7555` / `1d8aa61d` engineering track + `bc78d77f` an oracle fix).
- **How the 28 demos were made** (owner rules 2026-09-27 23:05): a full resampling sweep `v22_sweep_20260928-1` on `e6c62b73` - phase 1 =
  the 11 in-scope aliases ×2 + touching + cap, phase 2 = the other 15 once, one queue longest first - went 37/39: both kielbasa copies
  failed on one instance (seed 0 is deterministic on one commit). The fix `bc78d77f` is scoped (below); its exposed alias was re-run on
  `bc78d77f` (`v22_sweep_20260928-2`, kielbasa ×2: 2/2 green, 0928-080032 / 080036, 942 steps each, the pre-grasp witness reached through the ready posture). 27 aliases keep their green run of `-1`; `made_on`, the scoping evidence
  and the arm are in every manifest entry (`runlogs/first-demo-v2.2/manifest.json`).
- **The rule** (owner 2026-09-27; `sampling/v2/orient.py`, `tabletop.py`, the task modules): per arm-bound segment (grasp → carry →
  place → release, the push-under blade, a held object's hold) the other arm is offered when it has no role besides the segment and the
  object is free (`tabletop.arm_offer`: reads the template and the BDDL goal only). Candidates are arm × wrist turn × configuration, each
  solved with that arm's IK and checked against that arm's wrist camera; order: success, joint-limit safe (≥ 0.05 rad on every
  configuration of the segment's chains, else more clearance), easier reach from that arm's own start posture; the right arm with the
  generator's turn is the default and wins within 0.30 rad. Stance first: phase 2 takes foci from every offered arm's band and ranks by
  whichever arm reaches better; the other arm is screened only where its band covers the target. The fallback spans the segment. The arm
  is also ranked on the whole held chain after v2.1's `held_turns` (`tabletop.held_arms`). Records: `grasp.extras.arm_offer` (fired /
  not fired and why), `arm_choice` in the orient decision, `summary.arm`, `notes.arm_offer` on the executor's grasp action.
- **Mirrored left bands** (owner 13:01): the right tabletop / floor bands mirrored in memory (`ReachBands.mirrored` / `load_arm`: cells
  (x, λ) → (x, −λ), yaw → −yaw, extent [a, b] → [−b, −a], shoulders and idle postures to the other side; meta `mirrored_from: right` +
  `mirror_source`); never written, never computed in-process; a band of the arm's own replaces a mirror with no code change. Used: 8
  tabletop bands on breakfast_table_rhjoby_0 (marker, milk, sprouts, bar soap, apple, kielbasa, teacup, sandal) and the shampoo floor
  band (`reports/mirrored_bands.md`).
- **Results vs v2.1** (`runlogs/first-demo-v2.2/reports/measurements.md`, per copy):
  - *Arm:* the left arm on milk (2/2), shampoo (2/2), apple (2/2), bar soap (1/2); the right on sprouts, marker, teacup, kielbasa, sandal,
    banana, canned food; touching - offer evaluated, not fired - and cap / the 15 phase-2 aliases (not in path) unchanged. Every left
    choice is "only arm feasible at the draw": phase 2 ranked the stance by the best arm's reach and only the left arm reached from it; no
    draw had both arms feasible at the same pose. Program and annotations name the acting hand in every run.
  - *Right joint-2 limit lines (sampling log):* apple 19 → 0, milk 9 → 0, shampoo 0 → 0, marker 6 → 3, cap 21 → 13; up on bar soap 2 → 8,
    sprouts 11 → 24, teacup 2 → 6, sandal 17 → 26 (more stances explored).
  - *Smallest witness margin (rad):* sprouts 0.018 → 0.263, teacup 0.044 → 0.087, shampoo 0.002 → 0.009, sandal 0.122 → 0.141; down on
    milk 0.141 → 0.065, apple 0.083 → 0.047, marker 0.131 → 0.061, bar soap 0.181 → 0.045 (left) / 0.001 (right copy).
  - *Attempts:* shampoo 3 → 0, milk 1 → 0; up on sprouts 0 → 2, bar soap 0 → 2 / 1, teacup 0 → 1, sandal 0 → 1.
  - *Demo steps / smoothness (largest per-step arm change, arm path rad):* shampoo 1715 → 1510 (0.041 → 0.040, 18.7 → 16.4), milk
    890 → 904 (0.033 → 0.028, 12.6 → 10.8), apple 761 → 879 (0.021 → 0.036, 15.9 → 20.0), bar soap 876 → 940 left / 778 right.
  - *Wall time:* median 994.5 s (v2.1) → 1104 s; IORS large sandal 3001 s → 3318 / 5032 s (its sampling 1497 s → up to 4599 s: the left
    arm's coupling runs where the right arm's failed, and attempt 0 spends its 2400 s budget).
  - *Mirrored-band candidates that failed the exact IK:* 170/236 = 72 % in the sweep (canaries 79/116, 30/54).
- **Canaries** (`v22_canary_20260928`, on `e6c62b73`): the fragile aliases + the 11 exposed ones - replays of the v2.1 release instances on
  L40S (13/13) and RTX 5090 (12/13), fresh samples 13/15; the task-specific test (fresh apple and sandal, where the owner expected the
  left arm, + controls touching and cap vs v2.1): apple left, sandal right (only arm feasible), touching identical to v2.1 (strict no-op),
  cap unchanged. Reds, all right-arm instances the v2.1 oracle fails identically: the sandal 5090 replay (v2.1's own red canary, A/B
  `v21_sandal_ab_20260927`), kielbasa (A/B `v22_kielbasa_ab_20260928` on `5d6a7ff5`), a sandal instance's push-under joint jumps (A/B
  `v22_sandal_ab_20260928`).
- **The kielbasa fix, scoped per the owner's rules** (`bc78d77f`, `oracle/motions.py` `arm_to`):
  - *Predicate:* every joint-space round to a recorded witness failed and the pose-goal fallback failed or ended past
    `witness_branch_max_jump` without `accept_other_branch` - exactly where `arm_to` raised `plan_failed`; read from the planner outcomes it
    already has.
  - *Code paths:* when it fires, start → posture → witness as two joint-space plans through `witness_via_postures` (ready, then untucked),
    streamed like the direct plan (protection gate, residual close); none → `plan_failed` as before. Otherwise a strict no-op (the notes
    gain `witness_via_posture: "not fired"`). No sampling path.
  - *Exposure* (log scan of every v2.1 / v2.2 sweep and canary run): 4 runs, all failed before - the kielbasa instance ×3 and cap
    0927-221834 (v2.1); never on a green run (none of the 28 v2.1 release entries, none of the 37 green `-1` runs).
  - *Canaries* (`v22_canary_20260928-2`, on `bc78d77f`): replays of the `-1` instances of the fragile aliases + the kielbasa instance on two
    pod sets 12/12, fresh samples 5/6; kielbasa succeeds through the ready posture in all three runs (942 steps). Red: block design's fresh
    sample (not_reached at the final layout check) - the same instance as its green `-1` run, the fix never fired in it: the known
    block-design fragility.
- **Engineering track** (owner 2026-09-27, separate from the evidence path): `oracle/start_state.py` fingerprints a loaded instance's
  start state (robot base / joints / velocities / controller goals / assisted grasps, task objects' poses and joints, hinges, constraints,
  attachments, render settings; `run_isg` writes it to `meta.start_state`); `first_demo.py run --warm` runs the sampler and the demo in one
  Isaac process (stage 1: `og.clear()` between them), keeps a pristine TRO copy and loads it in a fresh process on the same pod. 8/8
  representative aliases equal (`v22_warm_20260928`; 7 bit-identical, banana equal on every value - its sha differed only by an object's
  repr address, fixed `1d8aa61d`). Stage 1 saves one process boot, not the scene load (warm load 86.8 s vs fresh 93.1 s median); stage 2
  (the demo adopting the sampler's world) is next. Warm runs live under `first_demo_warm/`; release evidence stays on fresh processes
  (`reports/warm_vs_fresh.md`).
- **Known limits:**
  - The left bands are mirrors of the right ones (9 bands, list above): 72 % of mirrored candidates fail the exact IK (the mirror does not
    map library grasp indices), so the left arm wins only where it is the only arm feasible at the draw; the owner computes the real left
    bands (TODO D6) - they replace the mirrors with no code change.
  - IORS large with both arms is slow: sandal samples up to 3× longer (both arms' couplings per stance B; the carry-pose search repeats per
    C candidate); `held_arms` is not offered for IORS large and held objects.
  - Pre-existing oracle limits surfaced by v2.2's new draws, identical on `5d6a7ff5`: a sandal instance's push-under joint jumps; the
    sandal's RTX 5090 replay; block design's final layout (run-to-run).
  - Kielbasa's grasp is slow with the fix (job wall 1818 / 2116 s vs 706 s on v2.1): five joint-space rounds and the pose-goal fallback
    fail before the posture route; the sampler still records a witness no direct plan reaches.
  - Scope unchanged: touching (the other arm pushes / braces), attachment, the carton lever, block design, axial, world_vertical,
    articulation and camera relocalization never offer the left arm.

## `first-demo-v2.1` — the owner's v2.0 findings fixed; orientation chooser and held-chain ranking; 28/28 green, scoped under the owner's 23:05 rules (2026-09-28)

- **Tag / commit:** `first-demo-v2.1` → `5d6a7ff5` on `rush-20260917-2100` (`first-demo-v2.0` + the orientation-chooser session's
  commits `6313cfe0` … `576c8ef1` + this session's `06f45e24` … `5d6a7ff5`).
- **How the 28 demos were made** (owner decisions 2026-09-27 23:05, rule 5): 25 aliases keep their green run of the sweep
  `v21_sweep_20260927-6` on `6ef3110f`; the last fix (`8e26243f` + `5d6a7ff5`, milk) is scoped, its exposed aliases (milk,
  kielbasa) and the alias -6 lost to a pod's TTL (object scaling) were re-run on the frozen `5d6a7ff5` (`v21_sweep_20260927-8`).
  `made_on` and the scoping evidence are in every manifest entry (`runlogs/first-demo-v2.1/manifest.json`).
- **Phase A.** Each owner finding on v2.0 was fixed so that it switches on only in its situation, detected at run time, and was
  proven with a demo (head + overhead video) against its reference before the sweep (job markers `runlogs/jobs/v21_*_20260927/`):

  | | finding (v2.0 run) | change (commit) | evidence (v2.0 → after) |
  |---|---|---|---|
  | A1 | cap: the attach offset before the snap (0927-040231; reference 0920-235831) | an attach place whose fingers, swept from the pre-place to the seat by raycasts, would meet the partner before the part seats opens them just enough (≤ 4 mm per finger, the assisted grasp keeps the cap; owner 14:46 accepted), then the descent is re-aimed for the part's settled in-hand pose (`11104778`, `8fd4da68`) | 0927-151320: offset before the snap 5.24 mm / 0.041 rad (Δz −5.1 mm) → 0.18 mm / 0.00009 rad (reference 1.1 mm / 0.0036); correction passes 1 → 0 (reference 0); the bottle moved 2.95 mm / 0.025 rad during the attach → unmoved; place 166 → 129 steps (reference 100); demo 927 → 882 steps |
  | A2 | block design: before the third grasp the gripper swings out to the right and back (0927-040243) | a travel leg keeps an empty arm that holds the recorded hand pose in another IK branch, when the leg stays clear and the arm's next **grasp** tracks from there (IK nearer than the recorded witness, no closer to a joint limit); the grasp's chain then runs on the kept branch (`b33352b3`; grasps only `8b572a12`) | 0927-152526 (pod): the transition before the third grasp 382 → 80 steps (hand 2.8 mm from the recorded pose, joints 3.39 rad from `arms_q`); the demo 3936 → 3563 steps |
  | A3 | bar soap: "the wrist camera contacts the table during grasp & release and pushes against it" (0927-034001) | not the camera (0 contacts): the two finger links pressed 91–154 N into the source tray at the closure and ~108 N into the destination tray at the release, torso_joint1 bent 7.6 mrad. A library pinch whose fingers would reach below the object's bottom + 4 mm is lifted by the deficit - the grasp raised, the approach ramped up to it from the recorded pre-grasp - when the object is set down or attached next, the pads stay 5 mm below its top, the faces at the lifted pads' end slope ≤ 20° and the approach tracks from the pre-grasp witness; a support place whose object hangs lower in the hand than recorded is raised by the difference (`48964d04`; gates `85887729` `c20fcb9d` `503be065` `ddc28f69` `8e26243f` `5d6a7ff5`) | torso deflection at the grasp / release 7.57 / 7.62 → 0.06 / 0.08 mrad; grasp lifted 14.4 mm, place raised 14.8–15.6 mm; no finger–tray contact (contact dump); demo 929 → 876 steps |

- **Other changes since v2.0:**
  - **Orientation chooser** (the other session, `6313cfe0` … `576c8ef1`): both wrist turns of a contact are proposals; order =
    success > joint-limit safe (0.05 rad on every configuration of the chain) > easier reach, the generator's order within
    0.30 rad. Details in `runlogs/jobs/v21_orient_chooser_20260927/`.
  - **Held-chain ranking** (owner 14:46, `346a5144`): a library pinch's turn is ranked on its whole held chain (lift, carry,
    pre-place, descent, place) - ranking only, never a rejection; IORS large records "not offered".
  - **Carton close brace** (`056538be`): a push-close far-face stop whose fingertips the push line passes < 1 cm inside moves
    onto the push line (the chooser's seed-0 instance failed `root_moved`: 0.0215 rad → 0.0020 rad, 0927-145609).
  - **Harness** (`06f45e24`, `c827b3c4`): `first_demo.py run --run-isg-args=--dev` on every demo (head + overhead video;
    `meta.json` records `external_video` / `external_video_error`). Cost, same machine and instance: +6.7 % demo time
    (278.3 → 297.0 s, kielbasa 0927-170717 / 0927-171505).
  - **Joint jump** (`5174c989`, `6ef3110f`): an `arm_along` leg whose differential IK jumps past `max_joint_step` is solved once
    more at half the waypoint spacing, except the via-above's rise, which keeps its own planned-lift fallback (shampoo's carry:
    0.21–0.28 rad jumps at 1 cm, tracked at 5 mm).
  - **Witness posture** (`052de5eb`, not proven): after a grasp's pre-grasp transit, when the approach tracks only from the
    pre-grasp witness, a protection-checked joint plan moves the arm into it. It never acted: on the sandal the plan passes the
    sandal's protection box (−22.8 mm) and is refused.
  - **Milk: no lift onto curved faces** (`8e26243f`, `5d6a7ff5`; scoped below).
- **Sweeps and the last fix** (history in `runlogs/first-demo-v2.1/reports/freezes.md`): -2 `8b572a12`, -3 `ddc28f69`,
  -4 `052de5eb`, -5 `5174c989` stopped or failed on the sandal, shampoo and the pre-existing block-design drop, each fixed and
  re-frozen; -6 `6ef3110f`: 44 of 46 green, phase 1 33/33; milk failed `placed_orientation` 0.483 rad (A3 lifted the pinch of the
  bottle lying on its side 13.1 mm; the lifted pads reached only 12 mm above its widest section and the bottle rolled 14.4°
  under the closure - not in the carry: the place was raised for it), object scaling lost to a pod's TTL.
- **The milk fix, scoped per the owner's rule 2:**
  - *Predicate:* inside A3's lift, after every existing check passed - the faces of the grasped object at the lifted pads'
    fingertip-side end slope more than 20° from vertical (two raycasts along the closing axis: PhysX scene queries, no state
    change, no IK / planner call, no cuRobo random stream).
  - *Code paths:* when it fires, that grasp runs the recorded (unlifted) pinch and the same object's place sees the recorded
    in-hand pose; otherwise a strict no-op. No sampling path.
  - *Exposure* (shadow over the recorded instances; demo-only replays on `5d6a7ff5`, identical to `6ef3110f` unless it fires):
    over all 142 v2.1 sweep runs the predicate can only be evaluated on three instances (bar soap, milk, kielbasa); it fires on
    milk (slope 0.875) and kielbasa (0.638), not on bar soap (0.0). On the v2.0 collection it fires on milk, kielbasa and apple
    (0.543); bar soap no; marker, touching, teacup, cap, cork, lid never reach it (teacup and marker replays reproduce v2.0's
    step counts exactly).
  - *Canaries* (job `v21_canary_20260927`, on `5d6a7ff5`): the fragile aliases sandal, shampoo, milk, block design, carton close and the
    exposed kielbasa, each replayed from its -6 instance on an L40S and an RTX 5090 pod plus a fresh sample - 17 of 18 green. The
    sandal's RTX 5090 replay failed at its grasp; the same replay on the base commit on the same pod fails identically (job
    `v21_sandal_ab_20260927`: the same clearances and swings), and the fix is not evaluated on the sandal's drop-bound grasp - so
    it is pre-existing and the freeze went ahead (a decision against the letter of rule 3, recorded here). The first canary launch
    was discarded: concurrent waves started runs of one subtask within a second and shared run directories; the driver now
    spaces launches ≥ 2 s.
  - *Re-run on the frozen `5d6a7ff5`:* `v21_sweep_20260927-8`: milk ×1 (0928-013557, the predicate fires: the recorded grasp), kielbasa ×2 (0928-013601,
    0928-013605, fires), object scaling ×1 (0928-013609; its -6 run was lost to a pod's TTL, the predicate never runs for it) -
    4/4 green. The other 25 aliases keep their -6 run: the fix changes no sampling path and its predicate does not fire on their
    instances (bar soap: slope 0.0; the rest never reach it). Shampoo's -6 phase 1 has one finished copy (the second was running
    when -6 was stopped); its canaries are green.
- **Owner rules checked after phase 1:** the tie-breaker's against-the-generator choices (cap's `~r180` turns for reach) were
  written and their demos passed - the rule stays; close-push fist: carton close took the default fist, blender close the
  handle's own fist as the only feasible one (the twin dropped downstream), all green - kept as is.
- **Measured against v2.0:**
  - **Demo length:** 34130 demo steps in total (v2.0 33598). Shorter: block design 3936 → 3646 (A2), apple 960 → 761, carton
    close 1464 → 1265, cap 927 → 858, footstool 999 → 624; longer on new instances: door close 2350 → 3028, door open 2246 →
    2876, dishwasher open 1234 → 1557, shampoo 1545 → 1715.
  - **Attempts:** attempt 0 for 21/28 aliases (v2.0 21/28). Earlier: cap 3 → 1, cork 1 → 0, sprouts 1 → 0, footstool 2 → 1;
    later: axial 0 → 1, shampoo 0 → 3.
  - **Orientation decisions** in the collected runs: 16 only feasible, 10 for reach, 3 default; 3 of 29 in the wrist-flipped
    branch; 2 against the generator's order, both the only feasible turn.
  - **Torso deflection at the grasp / release** (mrad): bar soap 7.57 / 7.62 → 0.06 / 0.08, apple 4.61 / 2.02 → 0.08 / 0.09,
    marker release 3.95 → 0.67; sandal 12.1 → 10.3 and milk 0.6 → 0.75, kielbasa 0.9 → 1.95 (their recorded grasps); **sprouts
    0.06 → 14** (the chooser's library turn puts the fingers 24 mm below the object's bottom and a drop follows, so A3 does not
    lift it - see known limits).
  - **Witness margins:** up on apple (0.002 → 0.083 rad), kielbasa (0.056 → 0.101), cap (0.065 → 0.087), lid (0.023 → 0.039);
    down on sprouts (0.18 → 0.018), axial (0.107 → 0.056), cork (0.073 → 0.051), world_vertical (0.19 → 0.157).
  - **Random re-solves** in the recorded attempt: axial 87, cap 76, shampoo 72, lid 56 (all attempts ≥ 1).
  - **Smoothness:** the largest per-step arm-joint change is 0.048 rad (door close; v2.0's largest 0.041).
  - **Wall time:** the `--dev` overhead camera costs +6.7 % demo time on one machine; pod-to-pod demo walls vary with the GPU.
  - Full table: `runlogs/first-demo-v2.1/reports/measurements.md`.

| alias | demo (made on) | other copies | sampling attempt (v2.0) | steps (v2.0) |
|---|---|---|---|---|
| `alignment-axial-board_game-breakfast_table` | 0927-212340 (`6ef3110f`) | 0927-212705 | 1 (0) | 518 (563) |
| `alignment-world_vertical-painting_s070-walls` | 0927-205225 (`6ef3110f`) | 0927-205922 | 0 (0) | 1149 (1148) |
| `camera_relocalization-obstructed-footstool_1` | 0927-211957 (`6ef3110f`) | – | 1 (2) | 624 (999) |
| `camera_relocalization-standard` | 0927-211908 (`6ef3110f`) | – | 1 (1) | 400 (400) |
| `configuration_matching-articulation_close_large_scale-dishwasher` | 0927-210926 (`6ef3110f`) | – | 0 (0) | 567 (567) |
| `configuration_matching-articulation_close_large_scale-door` | 0927-202321 (`6ef3110f`) | – | 0 (0) | 3028 (2350) |
| `configuration_matching-articulation_close_small_scale-blender-breakfast_table` | 0927-210241 (`6ef3110f`) | 0927-210444 | 0 (0) | 885 (857) |
| `configuration_matching-articulation_close_small_scale-carton_cdmmwy-breakfast_table` | 0927-202325 (`6ef3110f`) | 0927-202327 | 0 (0) | 1265 (1464) |
| `configuration_matching-articulation_open_large_scale-dishwasher` | 0927-211314 (`6ef3110f`) | – | 0 (0) | 1557 (1234) |
| `configuration_matching-articulation_open_large_scale-door` | 0927-202323 (`6ef3110f`) | – | 0 (0) | 2876 (2246) |
| `configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table` | 0927-212718 (`6ef3110f`) | 0927-213014 | 0 (0) | 936 (936) |
| `configuration_matching-articulation_open_small_scale-carton_cdmmwy_lever-breakfast_table` | 0927-202329 (`6ef3110f`) | 0927-202333, 0927-202331 | 0 (0) | 1604 (1604) |
| `configuration_matching-attachment-cap-lemonade_bottle-breakfast_table` | 0927-202307 (`6ef3110f`) | 0927-202311, 0927-202309 | 1 (3) | 858 (927) |
| `configuration_matching-attachment-cork-wine_bottle-breakfast_table` | 0927-205045 (`6ef3110f`) | – | 0 (1) | 950 (887) |
| `configuration_matching-attachment-lid-teapot-breakfast_table` | 0927-204513 (`6ef3110f`) | – | 1 (1) | 950 (917) |
| `configuration_matching-block_design-block_4-breakfast_table_board` | 0927-202313 (`6ef3110f`) | 0927-202315 | 0 (0) | 3646 (3936) |
| `disembedding-bar_soap-bar_soap_2_biscuit_1-breakfast_table` | 0927-213135 (`6ef3110f`) | 0927-213222, 0927-213504 | 0 (0) | 876 (929) |
| `disembedding-bottle_of_milk-bottle_of_milk_1-breakfast_table` | 0928-013557 (`5d6a7ff5`) | – | 1 (1) | 890 (881) |
| `inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table` | 0927-210516 (`6ef3110f`) | 0927-210632 | 0 (1) | 892 (969) |
| `inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table` | 0927-211238 (`6ef3110f`) | – | 0 (0) | 806 (806) |
| `inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-countertop-floor` | 0927-211643 (`6ef3110f`) | – | 3 (0) | 1715 (1545) |
| `inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table` | 0927-212056 (`6ef3110f`) | 0927-212310 | 0 (0) | 761 (960) |
| `inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table` | 0928-013601 (`5d6a7ff5`) | 0928-013605 | 0 (0) | 863 (945) |
| `inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table` | 0927-205001 (`6ef3110f`) | 0927-205038 | 0 (0) | 1638 (1638) |
| `inter_object_relation_satisfaction-under_large_scale-sandal-breakfast_table-breakfast_table` | 0927-202317 (`6ef3110f`) | 0927-202319 | 0 (0) | 2754 (2768) |
| `inter_object_relation_satisfaction-under_small_scale-teacup-saucer-breakfast_table` | 0927-210941 (`6ef3110f`) | – | 0 (0) | 891 (891) |
| `mental_rotation-banana` | 0927-210824 (`6ef3110f`) | – | 0 (0) | 108 (108) |
| `object_scaling-canned_food` | 0928-013609 (`5d6a7ff5`) | – | 0 (0) | 123 (123) |

- **Unit tests** on this commit: 1335 passed; the same 4 known failures as v1.0.
- **Collected demos:** `runlogs/first-demo-v2.1/` (28 runs copied, `videos/` = head camera, `videos_overhead/` = the `--dev`
  overhead camera under the same names, `manifest.json` with `made_on` / `scoping` per entry, `reports/`). The head videos
  are also linked from `runlogs/pilot_20260915/first_demo_latest_success_videos/manifest.json`.
- **Known limits at this tag:**
  - **Sandal grasp:** its pre-grasp transit takes the protection's via-above; on some pods the first plan succeeds with the
    supports present and the via-above's pose-goal point descends into another IK branch - the approach stalls 56 mm short
    in both attempts. v2.0's L40S pods, the local machine and the L40S canaries take a transit that works; the RTX 5090 canary
    replay of the -6 instance fails, and the same replay on the base commit on the same pod fails identically (the fix is not
    evaluated there). The witness-posture rule meant for it does not act (above).
  - **Sandal wobble:** a drop-bound pinch keeps its recorded grasp, so the sandal's fingers still press the table at its grasp
    as in v2.0 (171 N, 12.1 mrad torso deflection).
  - **Sprouts wobble:** the chooser's library turn puts the fingers 24 mm below the sprouts' bottom; a drop follows, so A3
    does not lift it and the torso deflects 14 mrad at the grasp (v2.0's turn: 0.06). Its witness margin is 0.018 rad (v2.0
    0.18); the held-chain ranking found no other feasible turn.
  - **Curved-face pinches** (milk, kielbasa, apple's v2.0 instance) keep their recorded grasp: A3 does not lift them, so their
    fingertips press the support as in v2.0.
  - **Block design release:** every board place releases its die ~2.3 cm above its goal (the die sits ~2.3 cm higher in the
    hand than the sampler's model; the fingertips already stand at their release clearance) - as in v2.0. A die that slides
    ~9 mm lands tilted on the raised slot's edge (-2 #2); it landed flat in every other run.
  - **Scoping of earlier fixes:** the owner's rule 2 came after them. A2's situation check calls cuRobo IK (it consumes the
    random stream when evaluated); the witness-posture rule's action does too (its predicate is two differential-IK checks).
  - **A1:** the fingers open up to ~4 mm per side over the partner while the assisted grasp holds the cap (owner accepted).
  - **A2:** only a grasp as the kept arm's next contact; pushes and braces keep the recorded posture.
  - **Held-chain ranking:** not offered for IORS large; the chooser session's single-turn contact sites (top-rim brace, prep
    and stage-2 pushes, the carton lever fist, held-object grasps, the block-design table flip) are not routed through it.

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
  - **Twins:** evaluated on 13 aliases, recorded on 6. Four rescue a library row that was infeasible at a joint limit:
    shampoo (joint 5 within 5 mrad; twin margin 0.48 rad), apple (joint 2, 3 mrad; 0.07), cork (joint 5, 32 mrad; 0.058),
    kielbasa (joint 7, 33 mrad; 0.055). Two win on margin: bar soap (0.007 → 0.19 rad), sprouts (0.063 → 0.18). Inside the
    searches the twin also rescued draws that were not the recorded one (shampoo 16, sandal 8, milk 2). Pushes, braces,
    blades, handles and cube grasps are not affected: they already tried both wrist turns.
  - **Witness margins:** the smallest IK-witness joint-limit margin rose on 13 aliases (bar soap 0.029 → 0.186 rad, sprouts
    0.025 → 0.180, sandal 0.008 → 0.102, cork 0.015 → 0.073, world_vertical 0.056 → 0.190) and fell on 7 (apple 0.109 →
    0.002, teacup 0.107 → 0.044, cap 0.153 → 0.065, lid 0.052 → 0.023, kielbasa 0.085 → 0.056).
  - **Random re-solves** in the recorded attempt: cork 92, lid 26, cap 26, sprouts 10 (all attempts ≥ 1).
  - **Smoothness:** no new joint jumps (largest per-step arm-joint change 0.041 rad in both). Within the same draw, a twin
    picked on margin starts further from the stance posture: the pre-grasp swing grows 1.42 → 3.10 rad (bar soap) and
    2.42 → 3.37 rad (sprouts).
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
- **Pinned for reproduction** (pushed to GitHub):
  - code: tag `first-demo-v2.0`;
  - reach bands the sampler read: commit `20c89d11` on this branch;
  - simulator: tag `ispatialgym-first-demo-v2.0` of `kmy17518/BEHAVIOR-1K` (`14e06b710`). It downloads behavior-1k-assets
    3.9.0 and robot assets 3.8.2 (the pods); the local machine had 3.9.0rc9 / 3.8.2;
  - demos: branch `first-demo-v2.0-artifacts` / tag `first-demo-v2.0-demos`, an orphan branch with no code history. It holds
    the collection and `runlogs/first-demo-v2.0/pins/PIN.json`: package versions, robot-asset checksums, the local sampler
    cache, and the sweep's ledger rows, job markers and driver.
- **Known limits at this tag:**
  - **Carton lever pinch:** the first closing does not attach (both pads are on the flap, but no assisted grasp forms within
    60 steps); the retry attaches. Its transit into the witness branch takes the protection's via-above.
  - **Flipped-witness check:** it depends on the raw IK branch, which differs between processes. `validate_instances`
    rejects the axial Y instances that sampling accepts.
  - **Twin wrist travel:** a twin picked on margin lengthens the swing to the pre-grasp (above), and 7 aliases drew instances
    with a smaller witness margin than v1.2. There was no run with twins off, so the twins' share of the attempt and time
    changes is not isolated from B1.
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
