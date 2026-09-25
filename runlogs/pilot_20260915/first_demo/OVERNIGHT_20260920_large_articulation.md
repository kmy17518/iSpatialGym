# Overnight 2026-09-19/20 — large-scale articulation (dishwasher, door)

Owner instructions (2026-09-19 23:18): (1) a dishwasher demo wider than 21°, the oracle working below and above 21°;
(2) door experiment: grasp, pull the base straight back with the eef following, at the arm's limit release, closed
gripper on the *inside* of the leaf, push outward (base to the left first if needed); (3) large-scale **close** demos
for the dishwasher (inverse of open) and the door (push from outside if (2) works, else push + grasp, else the inverse);
joints damped / counterbalanced so they stay where released.

All runs under `runlogs/pilot_20260915/first_demo/<subtask>/<run id>/{inst,demo,remote.log}`; every demo has
`head_rgb.mp4`, `meta.json`, `demo.hdf5`. Pods: B `zitmtum94orfh6` (dishwasher, push-close), C `1hh1gvuj7jy95n` (door).

## Demos (all `success: true`, `binary_success` held to the last step)

| task | run | angles | follow | notes |
|---|---|---|---|---|
| dishwasher open (shallow) | `configuration_matching-articulation_open_large_scale/0919-213438-counterbalance` | 0 → 20.7° | arm | counterbalance (gravity off on `link_2`, damping 5) |
| dishwasher open (wide) | `configuration_matching-articulation_open_large_scale/0920-002330-dishwasher_wide` | 0 → 31.5° (stopped 30.9°) | hybrid, 1 base move | same physics; the same oracle executes both modes |
| dishwasher close | `configuration_matching-articulation_close_large_scale/0920-014713-dishwasher_close` | 18.7° → 0.6° | arm (hybrid search fell back) | inverse of the shallow open, same grasp |
| door open | `configuration_matching-articulation_open_large_scale/0919-220054-door` | 0 → 48° | hybrid, 7 base moves | garden clutter removed, right arm |
| door close | `configuration_matching-articulation_close_large_scale/0920-011636-door_close` | 49.2° → 1.1° | hybrid, 8 base moves (1.04 m) | the inverse of the open demo (grasp the open leaf from the garden, follow it shut) |

Joints stay where released in every demo: the dishwasher through the counterbalance (gravity off on the leaf +
viscous damping — `constraints.apply_counterbalance`, re-applied by the loader from the TRO), the door through its
stock friction on a vertical axis (every stable-set probe holds). The pass criterion of the demos (goal band held
through release / retreat / settle) is exactly "the joint stays where the hand left it".

## Why the dishwasher tops out at ~31° with one grasp

The pinch on the bar is a rigid attachment: the hand must pitch with the door. Past ~28–31° every base pose within
0.8 m fails IK (`0919-232626-dishwasher_wide-36to50deg`: 119/119 IK rejections at 22–45° from six stances; the reach
volume's pinch class tolerates 20° of pitch). Moving the base cannot fix an orientation limit. Wider needs a second
grasp (release at ~25°, re-grasp the tilted handle, pull on) — the close sampler shows a grasp at ≤ 20° open is
feasible from a fresh stance, at 27–36° it is not (`0920-004610-dishwasher_close-27to36deg`: `step2_reach 0`). A push
does not help here: the only face whose push opens a bottom-hinged door further is the inside face, i.e. a hook-pull.

## Door pull-back / push experiment (`scripts/experiments/door_pull_push.py`, `ISG_PROGRAM_HOOK`)

`run_isg --mode oracle` with `ISG_PROGRAM_HOOK=<file>:hook` keeps navigate/transition/grasp of the recorded program
and replaces the follow by the experimental action; outcomes in `meta.json → actions[3].notes.pull_push`.

| run | pull | outcome |
|---|---|---|
| `0920-000304-door_pullpush-lagstop` | rigid arm, 0.15 m/s | stopped at 7° after 0.10 m: wind-up read before the door caught up |
| `0920-010009-door_pullpush-rigid` | rigid, settled, 0.08 m/s | 10.6° after 0.15 m; wind-up 0.24 rad ≈ the door angle — a rigid hand must yaw with the leaf |
| `0920-022239-door_pullpush-compliant7` | all 7 joints re-targeted | 21.5° after 0.40 m, but the arm went slack (extension 0.72 → 0.83 m) while the door stalled at 19° |
| `0920-030007 / 032722` | joints 1–4 rigid, wrist compliant | 18.9° after 0.30 m: the compliant wrist bends instead of pulling; the push stances were then wrongly clamped to the bounding box of the sampler's recorded poses (y ≥ 4.38); the inside-face pose with the hand aimed back at the robot is unreachable (1.1–1.4 rad off) |
| `0920-041046-door_pullpush-hook-unreachable` (final) | as above + hook contact (hand along the leaf, finger flank on the inside face) | 18.9°; the hook pose is unreachable from all six doorway-front stances (178–219 mm / 0.6–0.9 rad off, obstacles ignored) |

Conclusion so far: a straight base pull opens this door to ~19–21° whatever the arm compliance (rigid: wrist wind-up =
door angle; compliant: the pull force is lost in the wrist). The inside-face pose with the hand aimed back at the robot
is outside the arm's workspace. **Correction (owner, 2026-09-20 18:43):** the "1.65 m garden strip / side wall at
y ≈ 4.1" was wrong — it came from a Phase 1 debugging comment about the depth to the garden furniture before
``remove_objects``; the deck is open on the free-edge (−y) side of the door. The experiments had clamped every stance to
the bounding box of the poses the sampler happened to record (y ≥ 4.38), which is why the wedge-entry and hook poses were
never tried from beside the door. The only real wall contact was on the hinge side (wrist camera, y ≈ 5.7); the
left-gripper contact at y = 4.20 was the untucked arm touching the door wall. The clamp is removed; the arms travel in
the "ready" posture; the beside-the-door runs are reported in the afternoon section.

## Door push-close experiment (`hook_close` on the close instance, door at 49°)

| run | outcome |
|---|---|
| `0920-022454-door_pushclose-jamb` | stance on the hinge side put the wrist camera into the wall beside the hinge (y ≈ 5.7; collision monitor aborts the action) |
| `0920-024158-door_pushclose-air` | stance + pre-pose fine; the arc "completed" with the door unmoved: the face plane estimated from the handle contacts is ~6 cm off |
| `0920-030008 / 032726` | guarded approach: no contact in 16 cm; then a diff-IK stall at 13 mm with no joint at a limit |
| `0920-041049-door_pushclose-reachstall` | the stall counted as contact, the arc pushed from there: the arc completed freely, door unmoved — the stall was the arm's reach limit, not the leaf |
| `0920-044526-door_pushclose-glass` | closer stances (0.46 / 0.40 m unreachable, 0.54 reached): arc free again; the contact point (10 cm inside the free edge, 5 cm below the handle) lies on the glass pane |
| `0920-051131-door_pushclose` (final) | contact on the stile at handle height, 4 cm inside the edge: arc free again, door at 49.2° |

Conclusion of the night runs: the outer-face push never made contact with the leaf in five runs — the hinge-side
stance hits the wall beside the hinge, and the leaf plane used for the contact point was fitted through the *handle*
contacts (~6 cm in front of the face; corrected in the afternoon). The door close that works is the **inverse** of the
open (grasp the open leaf's handle, hybrid follow it shut, `0920-011636-door_close`); the opening-side hook-close is
re-run without the stance clamp in the afternoon section.

## 2026-09-20 afternoon: smoothness (owner: "smooth like a good teleoperator", fewest base moves)

Measured on `demo.hdf5` positions (the PhysX joint-velocity channel is noisy under contact and was misread earlier: the
"grasp jolt" reported in the morning does not exist — base, camera and eef are still to 0.1 mm at the grasp).

Sources of the wide-demo wobble and the fixes (`motions.py`, `execute.py`):
1. hybrid base segment: 0.10 m at 0.15 m/s carrying a 0.5 rad wrist change — the arm ramped by the base's progress index
   → segment duration ≥ max|Δq| / `follow_arm_speed` (0.35 rad/s), base accel ≤ `follow_base_accel` (0.4 m/s²), and a
   slew limit on the ramp fraction (the tracker's index runs ahead during the yaw blend at a segment start).
2. arm-only dense steps: a straight joint-space ramp between two IK solutions leaves the handle arc → the recorded eef
   poses are tracked by differential IK along the **true arc** about the joint axis (`joint_axis_from_contacts`,
   `arc_between`; the dishwasher hinge is recovered to < 1 mm from the sidecar contacts).
3. goal-band stop: closing the motion generator cut the base velocity in one step → cooperative stop (`stop_requested`):
   the base decelerates, the arm target freezes.
4. release: one-step unwind + immediate retreat → 12-step gradual unwind, 12-step hold after the release.

| wide dishwasher demo | follow steps | eef speed max | eef accel p95 / max | arm joint speed p95 | base accel max | trunk at release |
|---|---|---|---|---|---|---|
| original `0920-002330` | 64 | 0.44 m/s | 2.1 / 5.7 m/s² | 1.07 rad/s | 2.3 m/s² | 0.40 rad/s |
| final `0920-184003-dishwasher_wide_final` (see below) | ~500 | 0.13 | 0.3 / ~4 | 0.33 | 0.4 | 0.05 |

The residual single-step eef accelerations are a 0.1 s stick-slip of the loaded wrist roll against the door's stock
friction just before the goal band; the base-pitch impulse at the assisted-grasp deletion (0.25 rad/s, one step) is
inherent to removing a rigid fixed joint and unchanged.

Fewest base moves: `hybrid_follow.base_move_policy = "lookahead"` — when the arm alone cannot reach the next pose, the
base candidates are scored by how many following poses are then reachable arm-only (cheap IK probe), the best wins.
Door: 7 → 5 base moves (`0920-173049-door-lookahead`; this door's handle sweeps 0.5 m sideways, so from any base pose
the arm rarely reaches more than one further 10° step — the geometry, not the policy, sets the count). Dishwasher: 1 move
either way.

Door experiments of the afternoon (owner's suggestions), first pass **with the wrong stance clamp** (y ≥ 4.38): pull to
the opening side + wedge push — six clamped stances reached, the wedge-entry hand pose (along the leaf, ~60° off the
clamped headings) IK-unreachable (`0920-150800-door_wedgepush-entry-unreachable`); hook-close from the opening side —
48 entry poses over 6 clamped stances × 4 heights × 2 closings, best miss 8.6 mm / 0.046 rad against 5 mm / 0.05 rad
(`0920-154838`, `0920-162802`). Second pass with the clamp removed (stances beside the door, y ≈ 3.8–4.3): see the
runs `*-door_wedgepush` / `*-door_hookclose` after 18:46 and the report below.

## Sampler / oracle changes made tonight (all in the working tree)

* `articulation_large.py`: `arm_reach_limit` also requires the **retreat pose** (`reach_check_retreat`, default on) —
  7/8 dishwasher attempts had died at the program check's retreat; `stable_set_include_full` (drop the fully-open
  entry no follow can reach); close tasks aim the manipulation face at the **open handle** (`manip_face_from_open_handle`:
  the reach-volume screen classifies the grasp in the base frame within 20°, so every stance was discarded before IK);
  `prompt0.init_tilts` plumbing.
* `prompt0.py`: Phase 1 scans trunk **tilts** (`init_tilts`): the open dishwasher door fits the frame only from
  1.15 m upright, where the base backs into the kitchen bar.
* `articulation_hybrid.py`: follow-pose residual judged at the search's 5 mm (`witness_residual_follow_m`); the
  tracker rejects base poses from which the fingertips leave the head camera frame (`check_fov`, `fov_margin`) — the
  first dishwasher-close demo failed QA `gripper_out_of_fov`.
* `run_isg.py`: `ISG_PROGRAM_HOOK=path.py:callable` rewrites the expanded program (experiments only; recorded in meta).
* Experiment customs `runlogs/smoke_inputs/artic_large_exp/customs.json`: aliases `dishwasher_wide` (hybrid,
  S = 27/31.5/36°), close dishwasher (q_init 16–19°, init tilts) and close door (d_max 3.5 m, init tilts); door aliases
  carry `joint_limit_margin_rad 0.025`, `hybrid_follow.hard_margin_rad 0.025`, `remove_objects` (garden armchairs,
  coffee table, plant) — none of this is in the shared `isg_customs.json` yet.

## 2026-09-21 night: owner decisions 23:58 — sequential follow (A), push-close from the outer face (B), left-arm wedge (C)

Pods D `7rytyml4reqdfn` (dishwasher) and E `ioc20so9f8vxmf` (door), TTL ~09:50 PDT. Comparison demos (hybrid):
`0920-184638-dishwasher_wide_final` (1 base move, follow 7.7 s, eef accel p95/max 0.48/2.03 m/s², base accel max 0.41),
`0920-184644-door_lookahead_smooth` (3 base moves, 13.2 s, 0.74/3.04, 0.57). Metrics: `/tmp/isg_artic/smooth_metrics.py`
(30 Hz positions of `demo.hdf5` over the `base_follow` span; a "base move" = |v| > 2 cm/s).

### A. Sequential follow — sampler (`articulation_hybrid.py: FollowSearch.run_sequential`, policy `"sequential"`)

Arm segment = the dense grid advanced while the arm-only IK (joint-limit margin, collision, fov) succeeds; the segment
ends at the last pose whose **release** is feasible; one base move (release → drive with the arm frozen → re-grasp)
chosen by the look-ahead; repeat. Typed segments in `articulation.base_follow_path.segments`
(`arm` / `release_move_regrasp`, the anchored move dropped per the 23:58 decision); the flat pose list, `base_follow`,
`waypoints` and `retreat_arm_q` (the search's retreat witness, now verified directly by the program check) as before.

What it took to get from "one pose per segment" to the owner's one-move path (all in the working tree, experiment customs only):

| iteration | finding | change |
|---|---|---|
| `0921-011609 / 020027 / 022826` | segments cut to one 8° pose each (3+ moves for 31°): the 0.10 m retreat along −a has no IK once the handle has come close — the elbow folds past its limit | **retreat ladder** `release_retreat_m = [0.10, 0.07, 0.05]`: the release retreats by the first distance with an IK (the path's final pose keeps the program's 0.10 m); recorded per segment (`retreat_m`, `retreat_eef`), the executor tracks the recorded pose |
| `0921-034952` | a 1 cm "base move" scored look-ahead 3: the look-ahead counted arm-reachable poses, the segment was then cut by the retreat | look-ahead counts poses **up to the last releasable one**; `min_move_m 0.10 / min_move_deg 10` |
| `0921-022828 / 030920` (door) | the drawn stance reached 10° but no retreat from there: a zero-length first segment → grasp, release, move | a stance with no releasable first segment is rejected in seconds (`hybrid_follow_infeasible`) and every Phase-2 survivor gets a draw (not 8) |
| `0921-030918` | `TypeError: ignoring() got 'ground'` — the pods carried a mixed tree (another agent's `program_check.py` edit ahead of its `ik.py`) | full-package sync |
| `0921-041232` (door) | 30 s per look-ahead candidate: `pose_joint` re-synced all 4155 collision meshes of the house at every joint pose | `update_world(only_objects=[target])` — the target's meshes alone (`RobotKit.sync_world(only_objects)`, `pose_joint`) |
| `0921-041230 / 044216` | demo: after the release the re-grasp plan refused its start state — `right_arm_link5 – torso_link4` 9 mm inside cuRobo's self-collision padding, at the sampler's own retreat witness (the sampler checks the whole body with the recorded trunk; the oracle plans the arm with the trunk where it really is) | sampler: every search IK solution (and the re-grasp / hold configurations) must clear the padding by `self_clearance_m 0.02` (`R1ProCuRobo.self_collision_pairs(margin)`); executor: after the diff-IK retreat the arm ramps onto the recorded retreat witness, and a start-state refusal of the pre-grasp plan falls back to a joint-space ramp onto the recorded pre-grasp witness (`_ramp_arm_to_q`) |

### A. Results

| demo | path | base moves | follow | eef accel p95 / max (working arm) | base accel max | video |
|---|---|---|---|---|---|---|
| `0921-044216-dishwasher_wide_seq_replay` (instance `0921-041230-dishwasher_wide`) | arm 0 → 15.8°, release (retreat 7 cm) → 0.14 m straight back → re-grasp, arm 15.8 → 31.5° (goal 31.5° ± 3.1°, ended 31.0°, zero creep) | **1** (release / re-grasp; the hybrid demo also needed 1) | 501 steps = 16.7 s incl. the release / re-grasp (hybrid: 7.7 s) | 0.69 / 3.30 m/s² (hybrid 0.48 / 2.03) | 0.85 m/s² (hybrid 0.41: the tracker's stop, not the arm) | `.../0921-044216-dishwasher_wide_seq_replay/demo/*/head_rgb.mp4` |

| **`0921-050314-dishwasher_wide`** (clean re-sample with the self-clearance; **the sequential demo of record**) | arm 0 → 7.9°, release (retreat 5 cm) → 0.22 m back → re-grasp, arm 7.9 → 31.5° (3 poses; ended 30.7°, creep +0.0005 rad) | **1** | 540 steps = 18.0 s | 0.58 / 3.19 m/s² | 0.75 m/s² | `.../0921-050314-dishwasher_wide/demo/*/head_rgb.mp4` |

The eef maxima sit at the release (the assisted-grasp joint deletion, one step) and the base stop; between them the arm
segments run at the follow speed. The first demo (`0921-044216`) used the executor's **fallback** (the pre-grasp plan
refused its start state, the ramp onto the recorded pre-grasp witness took over — `[WARNING] ... ramped onto the recorded
pre-grasp witness`); the clean one (`0921-050314`, sampled with the 2 cm self-clearance) ran without a warning: the arm
ramped 0.022 rad onto the retreat witness, the pre-grasp plan started from it. With the self-clearance the first
segment shrank to one pose (the 15.8° pose's retreat is within 2 cm of the trunk padding) and the look-ahead put the
single base move 0.22 m back instead of 0.14 — still one move, the second segment then covers 8 → 31.5°.

Door (`0921-045829-door`, sampled in 633 s once `pose_joint` stopped re-syncing the whole house): arm 0 → 12.3°, move
0.14 m, arm 12.3 → 36.9° (4 poses), move 0.14 m, arm → 43.0°, move 0.14 m, arm → 49.2° = **3 base moves** (the hybrid
comparison: 3; the geometry — this handle sweeps 0.5 m sideways — sets the count past 37°). Its first demo failed QA
`gripper_out_of_fov` during the first drive: the retreated hand sits 0.32 m below the head camera; per the QA rule (b)
the gripper span closes at the release and re-opens at the next pre-grasp, which the sequential executor now does
(`hooks["fov"](False)` after `_release`, `True` after the pre-grasp).

| demo | path | base moves | follow | eef accel p95 / max (right) | base accel max | video |
|---|---|---|---|---|---|---|
| **`0921-055946-door_seq_replay`** (instance `0921-045829-door`; **the sequential door demo**) | arm 0 → 12.3°, release/move/re-grasp ×3 (0.14 m each, retreat 0.10 m, hand frozen at the retreat during the drives), arm 12.3 → 36.9 → 43.0 → 49.2° (ended 48.6°, goal 49.2° ± 4.9°, zero creep) | **3** (all release / re-grasp; hybrid: 3) | 1345 steps = 44.8 s (hybrid 13.2 s: each release / drive / re-grasp cycle costs ~10 s) | 0.55 / 3.53 m/s² (hybrid 0.74 / 3.04) | 0.85 m/s² (hybrid 0.57) | `.../0921-055946-door_seq_replay/demo/*/head_rgb.mp4` |

No executor fallback fired in the door demo (the diff-IK tracked every arc; the arm ramped 0.22 / 0.20 / 0.56 rad onto
the recorded retreat witnesses after the diff-IK retreats — the null-space drift the witness ramp exists for).

### B. Door close by pushing the outer face — sequential (`door_pull_push.py: PushClose / hook_close`)

`configuration_matching-articulation_close_large_scale/0921-003144-door_pushclose_seq3` — **success**, the door (49.2°)
closed by pushing in 5 chunks, 2608 steps, no grasp. Per chunk: the stance is chosen BEFORE driving by the oracle
backend's IK on the contact pose and the arc waypoints (front and side candidates, 20 per chunk), the base drives there
with the arms in "ready", then arm only: guarded approach along −m onto the outer face (corrected leaf plane, 6 cm handle
stand-off), arc push to the arm's limit, back off.

| chunk | stance (x, y, yaw) | contact + arc waypoints reachable | predicted landing | actual |
|---|---|---|---|---|
| 0 | front (−5.72, 5.15, −2.64) | contact + 1/10 | 47.2° | 49.2 → 46.7° |
| 1 | front (−5.80, 5.03, −2.74) | contact + 6/10 | 34.7° | 46.7 → 34.3° |
| 2 | front (−5.86, 4.81, −2.96) | contact + 6/10 | 22.3° | 34.3 → 21.4° |
| 3 | front (−5.98, 4.59, 3.10) | contact + 6/10 | 9.4° | 21.4 → 8.5° |
| 4 | front (−6.04, 4.42, 2.94) | contact + 4/4 | 0.0° | 8.5 → 0.0° (closed) |

Contact evidence: the leaf moved under every push (the guarded approach's stall counted as contact only when a further
1 cm push moved the door — it did in all five chunks); the collision monitor logged no pair. Video
`.../0921-003144-door_pushclose_seq3/demo/*/head_rgb.mp4`. The side stances were scored but never won (a front stance
0.6 m out reached more of the arc each time).

### C. Left-arm wedge push after the pull

`0921-000629-door_wedgepush`: the arm-mode pull from the hybrid instance's stance opened the door 5.0° only (the
differential IK gave up on the 5 → 10° arc: 10 mm off at waypoint 4/7, no joint at a limit) — below the 8° the hand
needs behind the leaf; no push. Re-run on the **sequential** door instance (`0921-045829-door`, whose first arm segment is
the sampler-proved 0 → 12.3° arm-only pull; the pull now tracks that recorded segment, a diff-IK give-up ramps onto the
recorded IK witness, then 5° steps on to the arm's limit), then release, base to the free-edge side (arms "ready"), left
hand into the wedge (hook point 5 cm inside the free edge, entry gated by the wrist camera), swing +20°, optional 0.25 m
outward drag.

`0921-063243-door_wedgepush_seq` (its predecessor `0921-060044-…-ghosthand` died when the fist closed ON the handle after a
failed back-off and the assisted grasp re-engaged — the door then followed the base; the fist now waits until the hand is
6 cm clear, the "ready" tuck ignores the door, the FOV span closes at the release):

| phase | result |
|---|---|
| pull (arm-mode: the recorded segment 0 → 6.1 → 12.2°, then 5° arc steps) | **22.1°** at joint 3's limit (margin 0.027 rad); no witness ramp was needed; grasp held throughout |
| release | the 0.10 m back-off along −a is unreachable at the arm's limit (diff-IK 11.8 mm; planned IK best 13.1 mm even without obstacles) → the hand stayed at the handle with the gripper open, arms to "ready", then the drive |
| reposition | first stance candidate reached: (−6.285, 3.868, 2.141), beside the door on the free-edge side, heading ≈ the wedge axis (123°) |
| entry (left arm) | pre pose reached at the first try (pre +0.12 m, pads vertical, best IK miss 0); slid 5.7 cm into the wedge, stopped by the wrist-camera gate at the free-edge plane; 29 cm wedge width at 22° |
| swing +20° about the hinge (arm only) | stalled after one waypoint (`joint jump 0.24 rad at waypoint 6`): +0.8° (22.1 → 23.0°) — the arm-only arc from beside the door runs into a wrist singularity, as in `0920-202450` |
| fallback: base 0.25 m outward (+m) with the hand hooked | 23.0 → **41.65°** (+19.5° ≈ the 20° target), door settled at 41.65°, zero creep; collision-monitor contact pairs: **none** |

So the pull-then-wedge-push route opens this door 0 → 41.6° with **one** base move that carries the leaf (plus the
reposition), against 3 release / re-grasp moves for the sequential follow — but the arm-only swing does not work from
beside the door (twice now); it is the base drag with the hooked hand that moves the leaf. The instance's binary goal
(49.2° ± 4.9°) is not reached by the experiment (41.65°: `oracle_failed / not_reached` in `meta.json`), the experiment's
own verdict is "push opened the door further". Video `.../0921-063243-door_wedgepush_seq/demo/*/head_rgb.mp4`.

### Where things stand (07:15)

* **A** done: `0921-050314-dishwasher_wide` (sequential, 1 base move) and `0921-055946-door_seq_replay` (sequential,
  3 base moves), both `success`, alongside the hybrid comparison demos. Sampler + executor changes in the working tree
  (`articulation_hybrid.py`, `articulation_large.py`, `program_check.py`, `execute.py`, `curobo_planner.py`,
  `robot_utils.py`); experiment customs only — nothing in `isg_customs.json`.
* **B** done: `0921-003144-door_pushclose_seq3`, closed by pushing in 5 chunks, predicted landings within 1.3° of the actual.
* **C** done as an experiment: pull 22.1° → wedge + drag 41.65°; the arm-only swing is the part that does not work.
* Open: the door's sequential follow takes 45 s (3 release / re-grasp cycles of ~10 s) against 13 s for the hybrid
  tracker; the release retreat at the arm's limit is often unreachable along −a (C's pull; A uses the retreat ladder
  and never releases where no retreat exists).

### 2026-09-21 23:50 — the chassis wobble and the BEHAVIOR-1K evaluator's 250 kg base

The wobble in `0921-050314-dishwasher_wide` is the chassis yielding to the loaded arm: the R1Pro base stands on three
bounding-sphere wheels (z / rx / ry virtual joints undriven, gravity off above the base), so at the URDF's 47.6 kg the
pull's reaction tips it. The BEHAVIOR-1K evaluator sets `base_footprint_link.mass = 250` for r1 / r1pro
(`omnigibson/eval/evaluator.py: EVAL_BASE_LINK_MASS`); we now do the same in both loaders (`r1pro_model.apply_eval_base_mass`,
called from `run_isg.InstanceLoader.build_env` and `sim_utils.SimWorld`; `ISG_BASE_MASS_KG=0` keeps the URDF mass; recorded in
`meta.json → loader.base_link_mass_kg` and the sampler's `summary.json → world`). Replay of the same instance, locally:
`0921-235247-dishwasher_wide_mass250_local`.

| | 47.6 kg (`0921-050314`) | 250 kg (`0921-235247-dishwasher_wide_mass250_local`) |
|---|---|---|
| chassis tilt during the follow (max / p95) | 0.87° / 0.78° | **0.01° / 0.01°** |
| chassis lift | 4.8 mm | **0** |
| base rotation rate at the release | 11.4°/s | **1.0°/s** |
| torso_joint1 deflection under the pull (max) | 0.01° | **2.6°** |
| head-camera rise during the pull | 10 mm | 19 mm |
| head-camera rotation rate at the release | 11.6°/s | 25.5°/s |
| eef accel p95 / max, door angle, success | 0.58 / 3.19 m/s², 32.3°, yes | 0.65 / 3.39 m/s², 32.4°, yes |

The heavy base removes the chassis motion entirely, but the head moves *more*: the same wrench now bends torso_joint1
(100 N·m cap) by 2.6° instead of tipping the chassis, and it springs back at the release. The load therefore is not a base
problem — it is the arm's wrench on the door: 1e7-stiff position drives saturate at their 18–55 N·m caps on any millimetre of
mismatch between the tracked arc and where the rigid assisted grasp lets the hand be, i.e. ~100 N at the hand at 30°. The
fix that removes the wobble is a compliant loaded arm (lower drive stiffness or effort caps during arm segments — the
counterbalanced door needs a few N) and / or re-anchoring each arc step at the measured hand pose; the 250 kg base stays
because it is the grader's robot.

### 2026-09-22 11:20 — the wobble's real cause, and an evaluator-replicable fix (owner: "would eval replicate that?")

Drive-level compliance is **not** replicable: the evaluator (`eval/r1pro.yaml`) runs the arms and trunk as
`JointController(position, use_impedances: false)` with no `isaac_kp/kd`, i.e. PhysX drives at `DEFAULT_ISAAC_KP = 1e7`,
`DEFAULT_ISAAC_KD = 1e5` and the URDF effort caps (arm 55/55/25/18 N·m, torso 100); the policy's only channel is the 23-dim
action (base velocity, trunk / arm absolute joint targets, grippers). So the compliance has to be in the commanded
targets. Three variants on the 250 kg dishwasher instance, all local replays of `0921-050314`:

| 250 kg base | open-loop path from the recorded poses (`0921-235247`) | per-step closed loop (`0922-113724-dishwasher_wide_closedloop_local`) | **re-anchored per pose (`0922-115754-dishwasher_wide_reanchored_local`)** |
|---|---|---|---|
| torso_joint1 deflection max | 2.56° | 4.26° | **0.05°** |
| head-camera rise during the pull | 19 mm | 36 mm | **0.5 mm** |
| release snap: camera rotation rate / speed | 25.5°/s / 0.37 m/s | 38°/s / 0.56 m/s | **1.0°/s / 0.001 m/s** |
| arm wind-up at the end (unwind) | 0.049 rad | 0.037 rad | **0.016 rad** |
| eef speed max / accel p95 | 0.17 m/s / 0.65 m/s² | 0.39 / 0.48 (bursts) | 0.17 / 0.49 |
| door, success | 32.4°, yes | 30.2°, yes | 31.3°, yes |

What was wrong: the fingers meet the bar 13 mm before the recorded grasp pose and the approach finishes on contact, so the
hand attaches **20 mm above the recorded eef pose** (radius from the hinge 0.612 m against the recorded arc's 0.592 m). Every
recorded follow target therefore sat 2 cm off the true handle circle; with 1e7-stiff, effort-capped drives that is a permanent
saturation (~100 N at the hand) — the wrench that tipped the light chassis and, on the heavy one, bent the torso, and that
snapped back at the release. (`arc_between` even tripped its own 2 cm "not on one circle" fallback and tracked straight to the
recorded poses.) The per-step closed loop removed the radial error but made every 1.7 mm correction a saturated burst — a
30 Hz pulse train, worse. **The fix** (`ExecuteConfig.follow_arm_tracking = "reanchored"`, the default): each recorded pose's
arc is the hand's ACTUAL pose (FK of the measured joints) rotated about the fitted hinge by the measured joint delta to the
pose's q, streamed open-loop at the follow speed; the recorded target positions are used only for the joint value and the axis
sense (`arc_axis_sign`). The arm segments now carry the grasp offset along (21 mm at every pose, recorded per pose in the
notes), the drives run in their linear regime, and nothing is stored to snap at the release. The remaining eef accelerations
(2.6–4.2 m/s² one step each) sit at the base drive's start and the re-grasp attach, not in the pulls. Fully replicable: it is
a property of the recorded joint-target stream (targets consistent with where the hand is), on the evaluator's stock drives
and 250 kg base. `Motions.arm_arc_closed_loop` stays available (`follow_arm_tracking = "closed_loop"`) as the measured
negative. Door replay with the same executor: `*-door_seq_reanchored_local` (below).

Door check (`0922-120634-door_seq_reanchored_local`, 250 kg + re-anchored, local): success, 0 → 49.2°, 3 release / re-grasp moves, follow 45.1 s; its
grasp offset is only 2 mm (a vertical bar: the fingers close where the sampler put them), so the door demo never had the
mismatch — torso deflection 0.25° → 0.33°, release snap 1.4 → 1.3°/s, eef accel p95 0.55 → 0.54: unchanged, as expected.
The fix is neutral where the grasp lands on the recorded pose and decisive where it does not. Both demos of record are now
the local ones (evaluator physics: 250 kg base, stock drives): `0922-115754-dishwasher_wide_reanchored_local` and `0922-120634-door_seq_reanchored_local`.
