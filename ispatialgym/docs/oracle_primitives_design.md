# iSpatialGym oracle primitives for R1Pro: design and implementation notes

Status: first implementation validated in simulation, 2026-09-04; section 6 rewritten 2026-09-15 for the v2 sampling
rebuild (recorded plans from the oracle sidecar, no fallbacks); section 3 rewritten 2026-09-15 06:50 for the v2
**structure** (conventions shared with the sampler, decision-free motions, actions as the unit of checkpoint /
annotation / QA, pure per-subtask programs, one executor; owner decision). Code: `ispatialgym/oracle/` holds the v1
classes until TODO F2/F5/F6 land; section 3.2 lists the fate of every v1 skill.

This document explains how the scripted (oracle) demonstration policies for the seven
iSpatialGym tasks are organised, why the primitive set is arranged in layers rather
than as a flat per-task list, which unit carries checkpoints, annotations and QA in v2 (the
action, section 3), how joint groups are restricted when planning with cuRobo, how base /
torso motions are made to look like JoyLo teleoperation, and what was learned while getting
it to run.

## 1. Facts that constrain the design

| Fact | Consequence |
| --- | --- |
| The `ispatialgym` env has **cuRobo v0.8.0 (cuRoboV2)**, a rewrite with a new public API (`curobo.motion_planner.MotionPlanner`, `curobo.kinematics`, `curobo.scene`, ...). OmniGibson's `action_primitives/curobo.py` targets the v1 fork API and cannot even import. | We write our own thin cuRobo v2 backend instead of patching OmniGibson. Nothing under `Behavior-1K/` is modified. |
| cuRoboV2 loads robots from **URDF only**. The R1Pro URDF has neither the six holonomic virtual joints (`base_footprint_{x,y,z,rx,ry,rz}_joint`) OmniGibson adds in the USD, nor the `left/right_eef_link` frames; its wheel/steer joints are movable in the URDF but fixed in the USD. | `r1pro_model.generate_holonomic_urdf()` writes a derived URDF (cached in `~/.cache/ispatialgym/r1pro/`): virtual base chain `base_footprint_x -> ... -> base_footprint_rz -> base_link` exactly as OmniGibson builds it, `{left,right}_eef_link` (6 cm below the gripper link, 180 deg about y, from the USD), `attached_object_{left,right}_eef_link` for grasped-object spheres, wheel/steer joints fixed. Collision spheres and self-collision pairs are reused from the shipped v1 yaml. **FK of every link matches OmniGibson to < 1 um** (`/tmp` validation), including the joints whose URDF axis is negative, so no sign remapping is needed. |
| OmniGibson's R1Pro joint vector is **28-D, breadth-first**, arms *interleaved* (`left_arm_joint1, right_arm_joint1, ...`). cuRobo orders joints by kinematic-tree traversal. | Never assume contiguous blocks; always go through `robot.arm_control_idx` etc. and reorder cuRobo `JointState`s by name. |
| The BEHAVIOR-challenge / JoyLo action is **23-D**: `base (3: vx, vy, wz in base frame, normalised to [-1,1] -> 0.75 m/s, 0.75 m/s, 1.0 rad/s)`, `trunk (4 abs)`, `arm_left (7 abs)`, `gripper_left (1, +1 open / -1 close)`, `arm_right (7)`, `gripper_right (1)`. JoyLo drives the base in **velocity** mode and the trunk through a 1-D *trunk translate* scalar interpolating three torso keyframes plus a tilt offset on `torso_joint3`. | The oracle emits exactly this action space, every step (`action.ActionState`), so demonstrations are drop-in training data and distributionally close to human JoyLo demos. `r1pro_model.joylo_robot_config()` gives the matching OmniGibson robot config. |
| Control at 30 Hz, physics 120 Hz. | cuRobo trajectories are interpolated at `dt = 1/30` and streamed one waypoint per env step. |

## 2. Design principle: oracle actions must be policy actions

Every primitive is a Python generator yielding **23-D JoyLo actions**, one per environment step;
nothing else touches robot state during execution. Privileged information is used only for
*deciding* (targets, IK, planning, object poses), never for *doing*. `set_obj_position(...)` with
physics disabled and the like are therefore instance-setup utilities (e.g. restoring the initial
grasp for Object Scaling / Mental Rotation from a TRO), not primitives.

In v2 the deciding / doing split is structural, not just a rule: the deciding happened in the
sampler and arrives in the sidecar; Layer 1 (motions) contains no deciding at all, and the only
run-time decision left is the trajopt between recorded key poses (3.9).

## 3. Structure (v2, 2026-09-15): conventions, motions, actions, programs, executor

v1 organised the oracle as backend / skills / task oracles: the skills decided *and* did (stance search, grasp
candidates, fallback cascades) and every task was a hand-written class with its own retry loop. v2 moves every
decision into the sampler and the sidecar (spec principle 4, §3.11), which changes the unit the code is organised
around: the oracle executes a recorded plan, and the per-action concerns of section 6 (checkpoint, one retry, two
annotations, hard-failure QA) attach to the **action** — not to the task and not to the motion. The v2 layering:

```
Layer 0   backend        r1pro_model, curobo_planner.R1ProCuRobo, action.ActionState, trunk.TrunkState,
                         base_tracker — unchanged from v1
          conventions    the geometry both the sampler and the oracle derive poses from (3.5); imported by
                         ispatialgym.sampling and ispatialgym.oracle alike
Layer 1   motions        decision-free executors of one recorded target each (3.2): track_path, trunk_to,
                         arm_to, arm_along, gripper, base_follow, hold, settle
Layer 2   actions        the unit of checkpoint / annotation / QA (3.6): Navigate, Transition, SetTorso, Grasp,
                         Lift, Reorient, Place, Push, Brace, BaseFollow, Retreat — targets() from a sidecar block
                         via the conventions; execute() yields 23-D actions via Layer 1; postcondition(); annotate()
Layer 3   programs       one pure function per row of 6.2: expand(alias, sidecar) -> list[Action] (3.7); no simulator
Executor  run_isg        one loop for the oracle / replay / policy modes (3.8): load -> expand -> per action:
                         checkpoint -> execute under the QA monitors -> postcondition -> one retry -> annotate;
                         then the epilogue (success, no-op window, settled, goal image, TRO goal override) -> record
```

Rules that keep the layers apart:

* **Layer 1 makes no decisions.** A motion receives one fully specified target (a path, a trunk pair, an eef pose
  or joint vector, a waypoint list, a gripper command) and either reaches it or raises. No candidate sets, no
  alternative embodiments, no re-stancing. Everything of that kind in v1's `Skills` (3.2, column "v2") moves to
  the sampler or is deleted.
* **Layer 2 is the only layer the executor sees.** Checkpoints, annotations and QA exemptions (push / brace need no
  grasp check) are per action; a motion is never checkpointed or annotated on its own.
* **Layer 3 is pure.** `expand` reads the sidecar and returns actions with their targets resolved; it never touches
  the simulator, so it is unit-tested against sample sidecars without Isaac Sim, and the sampler calls it (3.7).
* **Sampler and oracle share the conventions, not copies of them.** The sampler checks the *outputs* of the
  conventions functions; the oracle tracks the *same outputs* (3.5). v1 had `Skills.q_with_base_pose` and
  `RobotKit.q_at_base_pose`, `Skills.grasp_candidates` and `RobotKit.grasp_candidates_at`, and
  `camera_offset_in_zed` defined in a task oracle and imported by the sampler; none of that survives.
* **Correction is not a fallback.** Closing the loop on the tracking error of a recorded target (re-tracking the
  residual base error after a path, re-solving the arm goal from the settled state, the corrective navigation of
  the camera-relocalization oracle) is control and stays. Choosing a different target, embodiment, grasp or
  stance because the recorded one failed is a fallback and is gone (spec principle 2).

For reference, the v1 layering as implemented in `ispatialgym/oracle/` today (runnable until F2 lands):

```
Layer 2  task oracles   tasks/camera_relocalization.py, tasks/held_object.py (object scaling, mental rotation)
Layer 1  motor skills   skills.Skills: navigate_to, set_torso, ready_arms, move_eef(s), move_joints,
                        move_eef_cartesian, rotate_eef, set_gripper, grasp, place, place_object_at,
                        push, brace, stance_for / approach
Layer 0  backend        curobo_planner.R1ProCuRobo (planners per joint group, world sync, attach,
                        FK / IK / trajopt / differential IK), action.ActionState (23-D assembly),
                        trunk (JoyLo keyframe map), base_tracker (joystick-like velocity tracker)
```

### 3.1 Frames

Pose arguments are `FramedPose(pos, quat_xyzw, frame)` with `frame` in `world`, `base`
(robot `base_link`), `eef:<arm>`, `obj:<name>`, or a plain world `(pos, quat)`. This replaces
the `mode=global/local` flags.

### 3.2 Layer 1: the v1 skills and their v2 fate

All v1 skills yield 23-D actions; the "How" column is the v1 implementation (kept as the record of what works in
simulation), the "v2" column says what becomes of each one. The v2 motions are `track_path(waypoints, posture)`,
`trunk_to(translate, tilt)`, `arm_to(arm | arms, pose | q)`, `arm_along(arm, waypoints, speed)`, `gripper(arm,
close)`, `base_follow(b_k, arm)`, `hold(steps)`, `settle()`; nothing else is public in Layer 1.

| Skill | Joints that move | How (v1) | v2 |
| --- | --- | --- | --- |
| `navigate_to(pose2d)` | base x, y, yaw | cuRobo `base` embodiment plans a smooth collision-free `(x, y, yaw)(t)` for the *whole body* at its current arm/torso configuration (fallback: traversability-map A* / straight line, always whole-body collision-checked). `base_tracker` turns it into base-frame velocities: feed-forward + pose feedback, clipped to joystick limits, acceleration-limited, exponentially smoothed. Translation and rotation happen simultaneously - no translate-rotate-translate. | → `track_path`: the tracker follows the **recorded** path (spec §3.7); the cuRobo base plan and the trav-map / straight-line cascade (`plan_base`) are deleted. Used by `Navigate` and `Transition`. |
| `set_torso(translate, tilt, delta)` | torso 1-3 (+3) | JoyLo keyframe curve (`trunk.py`), scalars rate-limited exactly like holding the JoyLo controls. Swept motion is whole-body collision-checked; if hanging arms would hit the base it first calls `ready_arms`. | → `trunk_to`: keyframe curve and rates stay; the swept check and the `ready_arms` fallback go (the sampler swept the motion at the recorded stand-off, spec §3.8). Used by `Transition` and `SetTorso`. |
| `ready_arms()` | arms | Planned motion to `ARM_READY_Q` (hands in front, elbows bent; found by IK with max joint-limit margin), with lift-first and Cartesian fallbacks. What a JoyLo operator does before working. | **deleted.** Arms are untucked at init (spec §3.6); where the sidecar says `goal.arms: untucked` the lift is `arm_to` with the recorded joint vector. |
| `move_eef(arm, pose, torso=False)` / `move_eefs({arm: pose})` / `move_joints(q)` | 7 (14) arm joints, +4 torso if `torso` | cuRobo IK + trajopt, self/scene collision, grasped object attached; interpolated at 1/30 s, streamed, hold-to-converge. | → `arm_to`: trajopt with the trunk locked — the one planner left at run time (3.9). `torso=True` is removed. |
| `move_eef_cartesian(arm, pose / delta / waypoints, torso=False)` | arm (+torso) | Dense straight-line / arc waypoints -> damped-least-squares differential IK (finite-difference Jacobians from batched cuRobo FK, saturated-joint handling) -> collision validated -> streamed at a configurable eef speed. Guarantees the path shape trajopt does not. | → `arm_along`: differential IK through a **given** waypoint list; the `pose` / `delta_pos` forms become conventions functions that produce the list (3.5). `torso=True` is removed. Used by every contact motion. |
| `rotate_eef(arm, delta_rot, pivot=eef/object/point)` | arm (+torso) | Special case of the above with SLERP about a pivot; `pivot="object"` keeps the held object centred (Mental Rotation). | → conventions `orbit_waypoints` / `slerp_waypoints` fed to `arm_along`. Used by `Reorient`. |
| `set_gripper / open_gripper / close_gripper` | fingers | Streams +1/-1 until fingers stop or assisted grasping engages. | → `gripper(arm, close)`, unchanged. |
| `grasp(arm, obj, approach=top/front)` | composite | `grasp_candidates` (vertical + 30/45 deg tilted approaches x wrist yaws, fingertips kept above the object bottom) -> first IK-feasible grasp + pre-grasp -> planned pre-grasp -> straight approach -> close -> attach to planner -> lift. | → `Grasp` action (3.6): pre-grasp and grasp from the sidecar `grasps[]` entry via `grasp_pose` / `pre_pose`; `grasp_candidates` / `grasp_pose_for` are deleted; the lift is its own `Lift` action. |
| `place(arm, eef_pose)` / `place_object_at(arm, object_pose)` | composite | Reachability check (repositions via `stance_for` if needed) -> planned pose above -> straight descent (planned fallback) -> release -> retreat. `drop_height` avoids pushing the object into the support. | → `Place` + `Retreat` actions: recorded pre-place / place; `reposition` and the planned-descent fallback are deleted. |
| `push(arm, contact, direction, distance)` | composite | Closed gripper, planned standoff, straight motion through the contact. | → `Push` action: recorded pre-contact, contact, arc waypoints (δ/2, goal), distance. |
| `brace(arm, pose)` | other arm | Closed gripper + planned move: support hand for World Alignment / articulation / attachment. | → `Brace` action: recorded kind, pre-contact, contact; held until the working arm's `Retreat`. |
| `stance_for(target)` / `approach(target)` | base + trunk | Chooses base pose (facing the target, target on the arm's side) and JoyLo trunk level from candidates that are whole-body collision-free *and* have IK-feasible grasp + pre-grasp, then executes trunk/drive in a collision-safe order. Mirrors how operators position the robot before reaching. | **deleted.** Stances are the sampler's (spec §4–6) and arrive as `stances{}`; `go_to_stance`'s "torso first or drive first" guess becomes the recorded `Transition` (spec §3.8). |
| `plan_base`, `base_reference_collision_free`, `q_with_base_pose` | - | base path planning and its whole-body validation | `q_with_base_pose` → conventions (3.5); the rest is deleted (paths are recorded). |
| `hold`, `settle`, `execute_joint_trajectory`, `track_base`, `_wait_converged` | - | emitters and trajectory streaming | Layer-1 internals, unchanged. |

### 3.3 Restricting the joints that move (cuRobo)

cuRobo plans over `cspace.joint_names` minus `lock_joints`; locked joints become fixed
transforms at load time, so a planner has a fixed active set. We build one `MotionPlanner`
per **embodiment** (`r1pro_model.EMBODIMENTS`) from the same robot dictionary with different
`lock_joints`, and before every query refresh the locked joints' fixed transforms from the
simulator (`R1ProCuRobo.refresh_locked_joints`, the v1 trick ported to
`KinematicsParams.fixed_transforms`). Everything outside the active set is guaranteed not to
move. All planners share one scene collision checker (world loaded once, poses updated
incrementally).

| Embodiment | Active joints | Tool frames | Used by |
| --- | --- | --- | --- |
| `base` | `base_footprint_{x,y,rz}` | `base_link` | `navigate_to` |
| `right_arm` / `left_arm` | 7 | that eef | most manipulation |
| `both_arms` | 14 | both eefs | `ready_arms`, bimanual |
| `right_arm_torso` / `left_arm_torso` | 4 + 7 | eef | v1 fallback for hard reaches (the held-object oracles used it when the arm alone could not track); **not used by the v2 oracles** (spec §6.3: the arm moves, not the torso), kept for sampler queries |
| `whole_body` | 3 + 4 + 14 | eefs, `zed_link`, `base_link`, `torso_link4` | FK / collision queries (kinematics-only, no planner needed) |
| `base_torso` | 3 + 4 | `zed_link`, ... | viewpoint queries |

Per-embodiment `velocity_scale`, `max_acceleration`, `max_jerk` (in `EMBODIMENTS`) set the
planned motion speed; the defaults are deliberately slow (arms 0.25 x URDF limits, base
0.45 m/s) to look like teleoperation and should be tuned against human JoyLo statistics.

Design decision: for demos the torso is **not** freed while reaching. JoyLo operators lower the
trunk along the keyframe curve, then reach. The sampler chooses the trunk level (v1: `stance_for`),
`trunk_to` applies it, and the arm is planned with the torso locked. v1 used `*_arm_torso` as a
fallback when a motion was otherwise infeasible (e.g. Object Scaling toward the camera); v2 has no
such fallback — the sampler solved the goal with the arm-only embodiment (spec §6.3).

Unused tool frames of a multi-frame planner get a zero-weight `ToolPoseCriteria` (cuRobo then
treats them as trivially converged); joint-space goals use `plan_cspace`.

### 3.4 Execution model

* Motions and actions are generators yielding 23-D actions; composites `yield from`. The **executor** (3.8) steps
  the env (`env.step(action, n_render_iterations=1)`); `skills.run(env, gen)` is the v1 driver and stays only for
  the v1 classes.
* `ActionState` holds targets (trunk q, both arm q, gripper commands); every yielded action
  fills all 23 dims, so position-controlled joints never drift. Base velocity defaults to zero.
* Convergence checks with step budgets raise `SkillError(reason, ...)`; `reason` is the QA category of 6.4
  (`plan_failed`, `execution`, `collision`, `failed_grasp`, `timeout`, ...). Nothing inside a motion or an action
  retries with other parameters (v1 did); the executor retries the whole action once from its checkpoint.
* Collision world: all non-visual-only object collision meshes (expressed in the robot root
  frame, fixed during an episode) plus a ground slab; the grasped object is sphere-fitted,
  attached to `attached_object_<arm>_eef_link` and removed from the world. Sampler and oracle must load the
  same world (spec review item 3.1: room-set parity is a sampler-side fix in `sim_utils.py`).

### 3.5 Conventions shared with the sampler

One module (`ispatialgym/robot/conventions.py`, TODO F5; today spread over `oracle/skills.py`,
`oracle/tasks/camera_relocalization.py`, `oracle/poses.py`, `oracle/trunk.py` and `sampling/robot_utils.py`) holds
every function that turns sidecar facts into poses. The sampler validates the *outputs* of these functions; the
oracle tracks the *same outputs*; neither package may re-implement one. Every function is pure (tensors in, tensors
out, frames as in 3.1); the simulator is read only by the caller.

| Function | Used by |
| --- | --- |
| `grasp_pose(obj_pose, grasp_key, grasp_name, yaw)` → eef pose from the grasp-library entry | sampler Phase 3, `Grasp`, `Place` (carry transform) |
| `pre_pose(pose, dist=0.10)` = pose moved back along −a (spec §3.6); `handle_eef(contact)` = contact − 0.012·a; `contact_eef(point, normal)` = point − `CONTACT_CENTER_Z`·a | sampler brace / push / handle IKs; `Grasp`, `Push`, `Brace`, `BaseFollow` |
| `line_waypoints(p0, p1, n)`, `slerp_waypoints(q0, q1, n)`, `orbit_waypoints(pivot, delta, n)`, `arc_waypoints(contact, centre, δ)` (δ/2 and goal), `handle_waypoints(h, q_k)` — the waypoint policy of spec §3.9 | sampler waypoint IKs; the `arm_along` targets of `Grasp`, `Place`, `Push`, `Reorient`, `BaseFollow` (arm mode) and the held-object programs |
| `transition_poses(stance_from, stance_to, s_T)` → drive-back pose, trunk pair, drive-in pose (spec §3.8) | sampler transition sweep; `Transition` |
| `base_follow_poses(h, q_k, T_eef_base)` = `h(q_k) ∘ T_eef→base(q*)` (spec §5.3) | sampler base-follow feasibility; `BaseFollow` |
| `trunk.torso_q(translate, tilt)`, `translate_to_torso_q`, `torso_q_to_translate_tilt` (JoyLo keyframes) | everywhere the trunk is set or read |
| `camera_offset_in_zed(robot)`, `camera_pose(q)` | sampler view solve and goal image; `SetTorso`; the gripper-in-FOV monitor |
| `q_with_base_pose(q, x, y, yaw)` (root-frame virtual base joints) | sampler stance checks; `track_path`'s collision monitor |
| `cube_grasp(block_pose, face_pair, tilt_pick, yaw)`, `flip_waypoints(...)` (spec §6.5) | block-design sampler; `Grasp` / `Reorient` for blocks |
| gripper constants `FINGER_OPEN`, `CONTACT_CENTER_Z`, fingertip length | sampler; `gripper`; the contact functions |

### 3.6 Layer 2: actions

An action is one row of the executor loop: it owns exactly one checkpoint, one annotation pair and one QA verdict.
The contract (`ispatialgym/oracle/actions.py`, TODO F6 declares, F2 executes):

```python
class Action:
    name: str                          # "grasp", "push", ...
    arms: tuple[str, ...]              # arms involved (QA: gripper-in-FOV; failed-grasp exemption)
    objects: tuple[str, ...]           # bddl instances involved (annotation; intended-contact whitelist)
    qa_exempt: frozenset[str]          # e.g. {"failed_grasp"} for Push / Brace
    def targets(self, sidecar, state) -> dict           # poses / waypoints / paths via the conventions (3.5); pure
    def execute(self, motions, state) -> Iterator[th.Tensor]  # Layer-1 motions only; yields 23-D actions
    def postcondition(self, state) -> str | None        # None = ok, else a QA category of 6.4
    def annotate(self, sidecar) -> tuple[str, str]      # fine-grained, with numbers (6.3)
```

The vocabulary (the rows of 6.2 are compositions of these and nothing else):

| Action | Sidecar block | Motions | Postcondition |
| --- | --- | --- | --- |
| `Navigate(path)` | `paths[i]` (waypoints, posture, carried) | `track_path` | base within tracker tolerance of the last waypoint; whole-body collision-free along the way (monitor) |
| `Transition(t)` | `transitions[i]` + the two stances | `track_path` (drive-back) → `trunk_to` → `track_path` (drive-in) | trunk at the target pair, base at the arriving stance |
| `SetTorso(goal)` | `goal.{translate, tilt, arms}` | (`arm_to` untucked when `goal.arms` says so) → `trunk_to` | trunk converged; camera relocalization: head-camera pose within tolerance, one corrective `track_path` on the residual allowed (correction, not fallback) |
| `Grasp(g)` | `grasps[i]` | `gripper(open)` → `arm_to(pre_grasp)` → `arm_along(approach)` → `gripper(close)` | assisted grasp attached to the object (`failed_grasp` otherwise); backend `attach_object` |
| `Lift(g)` | `grasps[i].carry_height` | `arm_along(vertical)` | eef at the carry height, object orientation kept |
| `Reorient(w)` | `grasps[i]` / `blocks[i]` waypoints | `arm_along(orbit / slerp / flip waypoints)` | object orientation within tolerance of the waypoint end |
| `Place(g)` | `grasps[i].{pre_place, place}` | `arm_to(pre_place)` → `arm_along(descent)` → `gripper(open)` → `arm_along(retreat −a)` | object released (`release_grasp_immediately`), backend `detach_object`, object settled on the support |
| `Push(c)` | `contacts.push` | `gripper(close)` → `arm_to(pre)` → `arm_along(init → mid → goal)` | eef reached the goal contact; the pushed object moved (`no_motion` otherwise) |
| `Brace(c)` | `contacts.brace` | `gripper(close)` → `arm_to(pre)` → `arm_along(pre → contact)` | eef at the contact; held (no further command) until the working arm's `Retreat` |
| `BaseFollow(a)` | `articulation.{base_follow, waypoints, follow_mode, passive_from}` | `base_follow(b_k)` with the arm locked, or `arm_along(h(q_k))` in arm mode | joint at `q_goal`, or `passive_from` reached and the joint continued passively |
| `Retreat(arm)` | the preceding action's retreat vector (−a, 0.10 m) or "away a little" after a place | `gripper(open)` if it was closed for a brace → `arm_along(retreat)` | eef clear of the object by the retreat distance |

Actions read privileged state between yields (object poses, `_ag_obj_in_hand`, joint positions) to evaluate their
postcondition and to close the loop on tracking error; they never choose between alternatives.

### 3.7 Layer 3: programs and `expand`

`expand(alias, sidecar) -> list[Action]` (`ispatialgym/oracle/programs.py`, TODO F6) dispatches on the alias's
subtask to one function per row of 6.2 (`program_block_design(sidecar)`, `program_iors_touching(sidecar)`, …).
Each returns the action list with the loops and conditionals of its row resolved from the sidecar (block order,
`brace_col` present or not, `passive_from`, `follow_mode`, the placement stance of disembedding / IORS large-scale).
The sidecar therefore carries **facts, not the sequence** (owner decision 2026-09-15; spec §7): one schema serves
every subtask, the sequence lives in one readable place next to the table it must match, the loops are Python rather
than a JSON DSL, and a new subtask adds a program, not a schema version.

Two callers besides the executor:

* **The sampler's program check** (spec §3.5, TODO D2): once the draft sidecar is complete, `expand` it and IK- and
  collision-check every pose and waypoint that every action's `targets()` returns, in the recorded posture with the
  carried object attached. This is how spec principle 4 ("whatever the oracle does, the sampler has checked") holds
  by construction rather than by two code bases agreeing.
* **`validate_instances.py`** (TODO D5) runs the same expansion on written instances.

Tests: every program is unit-tested against one hand-written sidecar per subtask (action names, arms, object sets,
target shapes) without a simulator; the conventions are tested against the FK numbers of section 5.

### 3.8 Executor (`run_isg`)

One loop serves the three modes of 6.5; only the action source differs (a program, a LeRobot episode, a policy):

```
load TRO + sidecar + scene variant (loader contract, spec §3.14); verify sidecar.tro_sha256
actions = expand(alias, sidecar)
for action in actions:
    checkpoint = snapshot(simulator state, executor state)
    for attempt in (0, 1):
        verdict = run(action)                 # steps the env; the QA monitors of 6.4 watch every step
        if verdict is None: verdict = action.postcondition(state)
        if verdict is None: break
        restore(checkpoint); log(action, verdict, attempt)
    else: fail the episode with verdict
    record action.annotate(sidecar)
epilogue: binary success -> max(50, 20 % of the action steps) no-op -> pose-settled -> goal image ->
          if success within tolerance: TRO goal poses overridden with the settled values (6.2)
```

* **Checkpoint contents.** The simulator state (scene state including the assisted-grasp bookkeeping of the robot;
  verify that `scene.save_state` / `load_state` round-trip it, section 8) **and** the executor state: `ActionState`
  hold targets, `TrunkState`, the backend's attached objects, the gripper commands, the tracker's reference index.
  Restoring the sim without the executor state resumes with stale hold targets or a planner that still thinks the
  object is attached — the reason executor state can no longer live as loose attributes on `Skills`.
* **One retry**, then the episode fails with the action's category (6.4). The retry re-runs `targets()` from the
  restored state, so tracking error accumulated before the checkpoint is not inherited.
* **The epilogue belongs to the executor, not to an action.** The TRO goal override makes `run_isg --mode oracle`
  a producer of evaluation targets; it is the only place where the demo pipeline writes into the TRO. The loader
  and `ISpatialGymTask` never read the sidecar.
* **Stepping is the executor's** (`env.step(action, n_render_iterations=1)`), so the recorder (6.5) and the QA
  monitors see the same frames in all three modes.

### 3.9 The one planner left at run time

Between recorded key poses the arm still moves by cuRobo trajopt (`arm_to`: pre-grasp, pre-place, pre-contact,
the untucked lift), planned at execution time from the settled state with the trunk locked. The sampler proves IK
and collision feasibility of the key poses and waypoints, not the existence of a trajopt solution between them, so a
trajopt failure is not a sampler bug. Decision 2026-09-15 (owner, on the agent's recommendation): trajopt stays at
run time — recording joint-space trajectories would enlarge the sidecar, cost one trajopt per candidate pair during
sampling, and still not start from the settled state — and `plan_failed` is its own QA category (6.4), retried
once from the checkpoint and counted separately in the QA statistics so that a sampler-side cause (a key pose at
the collision margin) can be told from a planner-side one. If the pilot (E1) shows a `plan_failed` rate the retry
does not absorb, the alternative is to record `q_traj` per key-pose pair in the sidecar and make `arm_to` a tracker.

## 4. cuRobo 0.8.0 facts and workarounds found on the way

These cost most of the time and are worth knowing:

1. **Warp shadowing.** Isaac Sim's Kit fast importer maps `warp.*` module names to its bundled
   `omni.warp.core` (warp 1.8) regardless of the pip package already imported. cuRobo 0.8 needs
   pip `warp-lang`. `curobo_planner` eagerly imports the pip submodules cuRobo uses and installs
   a `sys.meta_path` finder in front of Kit's (`pin_pip_warp_after_launch`). Import
   `ispatialgym.oracle.*` (or `warp`) **before** `omnigibson`.
2. **Mesh SDF bug for small meshes.** `compute_local_sdf_with_grad` queries
   `wp.mesh_query_point` with `max_distance = half bounding-box diagonal` and returns that value
   as the distance when nothing is found. Any robot sphere with radius larger than a mesh's
   half-diagonal (15 cm base spheres vs a 5 cm object) is then "in collision" everywhere. The
   query radius must be >= max(half-diagonal, sphere radius + activation distance); the second
   term is what upstream forgot. Workaround: raise the stored mesh `dims` (only used for that
   query) to >= 0.4 m, i.e. a 0.35 m query radius that exceeds the largest robot sphere
   (0.18 m) plus the largest margin we use (`_patch_mesh_query_dims`). Larger meshes are untouched.
3. `SceneData.update_obstacle_pose / enable_obstacle` only catch `RuntimeError`, but the cuboid
   store raises `ValueError` for unknown names -> address the mesh store directly.
4. `RobotSceneCollision.validate` is broken (passes a tensor where a state is expected);
   `TrajOptSolver.attachment_manager` does not exist. We build `SceneCollisionCost` /
   `SelfCollisionCost` and `AttachmentManager` ourselves on a dedicated `Kinematics` that shares
   the planner's tensors.
5. Switching a planner between pose goals and joint goals needs a CUDA-graph re-capture, which
   cuRobo only allows with `curobo.runtime.cuda_graph_reset = True` (set on the *public* module).
6. IK results: use `result.solution` (batch x seeds x dof) with `result.success`; `js_solution`
   has a different layout. IK/trajopt also only report `success`, no reason.
7. The shipped self-collision ignore list misses `{arm}_arm_link1 <-> {arm}_arm_link3`, which
   overlap by 3 mm in the straight-arm pose (v2 has no tolerance slack) -> added.
8. The wheels touch the floor (3 mm sphere clearance) and the base body sits ~1 cm above it, so
   any proximity margin flags them against the ground plane *and* against BEHAVIOR's floor meshes
   at z = 0. `check_collision(margin=...)` therefore exempts only the pairs (ground-contact link,
   ground-like obstacle): ground-contact links are `robot.floor_touching_base_link_names` +
   `base_link`; ground-like obstacles are the ground slab and objects of the categories in
   `GROUND_CATEGORIES` (floors, carpet, mat, ...). The margin is evaluated with those obstacles
   switched off for those links only, so the base still keeps its full margin against walls and
   furniture, and actual penetration is always checked for every pair. The ground slab stays at
   z = 0 like the floor meshes. (Verified: base 3 cm from a wall is flagged at a 3 cm margin but
   not at 1 cm; standing on the floor is never flagged; sinking into the floor always is.)

## 5. Lessons from the simulation runs (`/tmp` tests, not in the repo)

### 5.1 Validated end to end

Empty scene, table (0.72 m) with a 4.5 cm block: `ready_arms` -> `set_torso` (0.5 -> 0.9 ->
tilt -0.15 -> back; joints reach the keyframe values exactly) -> `navigate_to((0.35,-0.35,0.35))`
(1.5 cm / 0.00 rad error, translation and rotation simultaneous, |v| <= 0.42 m/s) ->
`approach(block)` (stance (0.20,-0.21,0.39), trunk 0.5) -> `grasp` (assisted grasp engaged) ->
`rotate_eef` 60 deg about the object (60.0 deg measured) -> `move_eef_cartesian` up 10 cm
(10.0 cm) -> `place_object_at` (auto-reposition, 2 mm final error).

Task oracles: Camera Relocalization to a goal built from (base, translate 0.8, tilt -0.25):
viewpoint solve recovers it (residual 0.0008), final camera error 1.6 cm / 0.0 deg in 12 s of
sim time. Object Scaling x1.3 apparent size: camera-object distance 0.462 -> 0.355 m (target
0.355). Mental Rotation 77 deg: 0.33 deg final error.

### 5.2 R1Pro geometry that the oracles must respect

Trunk translate vs shoulder height and how far the bent torso ("knee") protrudes in front of
the base (arms in the ready pose):

| translate | shoulder (x, z) | ready hand (x, z) | knee front x |
| --- | --- | --- | --- |
| 0.0 | (0.13, 1.41) | (0.48, 0.96) | 0.29 |
| 0.5 | (0.16, 1.23) | (0.51, 0.78) | 0.36 |
| 0.75 | (0.14, 1.09) | (0.49, 0.64) | 0.40 |
| 1.0 | (0.10, 0.92) | (0.45, 0.48) | 0.41 |

Consequences: with the default trunk (0.5) the base cannot get closer than ~0.37 m to a table
edge, so top-down grasps on a 0.72 m table only work within ~15 cm of the edge; lowering the
trunk further puts the hands below the table; vertical wrist-down grasps far from the shoulder
exceed the arm's reach, hence the tilted (30/45 deg) grasp candidates. This is why `stance_for`
exists and why instance generators should validate reachability with `Skills.reachable` /
`stance_for` when sampling object and goal poses.

## 6. Task oracles (v2, 2026-09-15)

The oracles are the second half of the sampling rules in `instance_sampling_spec.md`: every pose, stance, contact
and waypoint they use was IK- and collision-checked at sampling time and is read from the instance's **oracle
sidecar** (`<instance_id>.oracle.json`, spec §7). An oracle never searches for a grasp, a stance or a path; if a
recorded plan fails in execution that is a QA failure to be fixed in the sampler, not something to fall back from.
Motions look like JoyLo teleoperation: base velocities with simultaneous translation and rotation, trunk along the
keyframe curve, arms planned with the trunk locked. The structure (conventions, motions, actions, programs,
executor) is section 3; this section is the per-action and per-subtask content.

### 6.1 Actions in v2 (behaviour per action; the contract is 3.6)

| Action (v1 skill) | v2 behaviour |
| --- | --- |
| `Navigate` (`navigate_to`) | follows the **recorded path** (spec §3.7): un-eroded shortest path validated with whole-body checks at the navigation posture when both stances lie in one locally un-eroded region, otherwise eroded-map global path + collision-checked local connections; the traversability map carries the scene variant's `freed_footprints` (chairs) and `blocked_footprints` (obstacle). Erosion for planning uses the footprint of the posture actually held (untucked arms, carried object). |
| `Transition` / `SetTorso` (`set_torso`, `go_to_stance`) | no `ready_arms` fallback. Trunk changes happen at the recorded **transition stand-off** (spec §3.8): drive back to `s_T`, move the trunk, drive in (`Transition`). `SetTorso` is the camera-relocalization goal trunk; arms are lifted to untucked only where the sidecar says so (`goal.arms`, self-collision at the goal trunk). |
| `Grasp` (`grasp`) | uses the recorded `(grasp_key, grasp_name, yaw)` → pre-grasp and grasp eef poses; approach = straight 10 cm along the grasp axis; assisted grasp must attach (hard QA failure otherwise, except for push / brace actions). The following `Lift` keeps the object orientation. **Roll-free closure (owner 2026‑09‑16 17:10 item 4, closure probe `runlogs/closure_probe_20260916/report.md`):** before the descent the fingers are pre-shaped to the library pinch width / 2 + 10 mm (`MotionsConfig.gripper_preshape_margin`; the fully open lower pad of a tilted grasp stood on the table, stopped the approach 9 mm short and closed 3 steps late), and the closure ramps the 1‑D gripper command at 2.25 mm per control step (`gripper_close_rate`, ¼ of the free speed) so both pads reach the object in the same step — the stock step command let the free pad knock the un-pinched cylinder along the table and roll it 19–77° (demo 0916‑152503: 24.6°); the ramp gave 0.1 / 2.9 / −3.2° on the marker, −0.3° with the pre-shape. Opening and the push / brace closures keep the step command; the assisted-grasp rule is unchanged. |
| `Place` + `Retreat` (`place` / `place_object_at`) | recorded pre-place / place eef poses; straight descent; release; retreat 10 cm along −a; then "move contacting grippers away a little". **Staged, rate-limited release (owner 2026‑09‑16 21:18; `Motions.gripper`, no `execute_place` change):** every opening ramps at `MotionsConfig.gripper_open_rate` (2.25 mm/step); the release of a held object opens first only to the pre-shape width — pinch / 2 + 10 mm, capped by the sampler's `grasps[k].release_sweep.release_opening_max_m` / 2 (core `385de1e`), never under pinch / 2 + 5 mm (`Executor.release_stage_opening`) — and the full opening is deferred until the hand has moved ≥ 3 cm from the release pose (`gripper_release_clearance_m`; the place's retreat leg completes it through `_emit`, at the latest after 300 steps). Nextto 0916‑201902 / 0916‑201730: the lower pad of the tilted grasp was blocked by the table at 16–20 mm, stayed there ~60 steps and snapped to 50 mm during the retreat, flinging the pencil 5.2 cm (or the released marker 26°). Notes: `actions[].notes.release` (staged / full_open events), `release_stage_m`. |
| `Push` (`push`) | closed gripper at the recorded contact, along the recorded direction / arc waypoints (δ/2, goal), distance from the sidecar. |
| `Brace` (`brace`) | closed gripper at the recorded brace contact (kind, point, direction), placed **before** the working arm moves, held until the working arm retreats. |
| `BaseFollow` (new) | articulation: with the handle grasped and the arm locked, track the recorded base poses `b_k` (`base_follow`) while the joint moves; `follow_mode: arm` uses `arm_along` over `h(q_k)` instead; base mode may fall to arm mode per waypoint only where the sampler recorded both. |
| `Reorient` (new) | in-hand yaw / flip / SLERP with the object attached, through the recorded waypoints (`orbit_waypoints` / `slerp_waypoints` / `flip_waypoints` of 3.5 fed to `arm_along`, pivot = object centre). |
| checkpoints | per action, by the executor (3.8): after every successfully executed action the simulator and executor state are snapshotted so the episode can roll back and retry the action once before it is declared a failure. |

### 6.2 Programs per subtask (Layer 3)

Each row is one `program_<subtask>(sidecar)` function behind `expand` (3.7); the row is what the function must
produce, with its loops and conditionals resolved from the sidecar. Every program ends with the executor's epilogue
(3.8): task success check → `max(50, 20 % of the action steps)` no-op steps → pose-settled check → goal image
captured → if the binary goal holds and the measured errors are within tolerance, the goal poses of the TRO are
overridden with the settled values.

| Subtask | Sequence |
| --- | --- |
| camera relocalization — standard | `navigate_to` the goal base (recorded path, bucket length) → `set_torso` to the goal (translate, tilt); if the goal trunk self-collides with the reset arms, lift the arms to untucked first (recorded `goal_arms`). |
| camera relocalization — obstructed | as standard on the map with the obstacle blocked; the obstacle appears in the head camera along the path. |
| object scaling | move the held object along the camera centre ray to the recorded depth (arm only; 3 recorded waypoints), object orientation unchanged. |
| mental rotation | rotate the held object to the goal orientation about its own centre (arm only; SLERP waypoints), position unchanged. |
| disembedding | init → manipulation stance (transition) → `grasp` at the recorded pose → lift to the carry height (orientation kept) → drive sideways along the edge to the **placement stance** (recorded path) → `place` at the destination centre (orientation kept) → retreat. |
| block design | init → manipulation stance → for each block in the recorded order (anchor first): `grasp` (recorded cube grasp: pinch ∥ flip axis, pick tilt) → lift → `reorient_in_hand` (smallest yaw to the table axes, then one 90° flip about the pinch axis to the place tilt) → `place` at its slot with a small safety margin (anchor: its init position) → push it left against the placed row with the right hand while the closed left gripper braces the row-left blocks (recorded brace) → push it up while the left gripper braces the column-top blocks (where recorded) → retreat both grippers. |
| articulation — large-scale open / close | init → manipulation stance (transition) → `grasp` the handle at h(q_init) (both directions) → move the joint: `base_follow` along the recorded `b_k` (vertical-axis doors, drawers) or `move_eef_cartesian` along `h(q_k)` (horizontal-axis flaps), up to `passive_from` or q_goal → release → retreat. |
| articulation — small-scale open / close | init → manipulation stance → `brace` the base link (recorded kind) with the brace arm → `grasp` the handle with the handle arm → move the arm along `h(q_k)` to the goal (or `passive_from`) → release → retreat both. |
| attachment | init → manipulation stance → `grasp` the target → lift → `reorient_in_hand` toward the attach orientation while moving to the pre-attach pose (10 cm back along the female insertion axis) → `brace` the partner on its face opposite to the insertion direction with the other gripper (closed) → insert along the axis until `attached` → release → retreat both. |
| alignment — world vertical / horizontal | init → manipulation stance → close both grippers → `brace` at the recorded stop (`below_high` / `above_low` / `side_high`) → `push` at E_low along the arc (δ/2, goal) with the arm only until the level / upright orientation → retreat both. |
| alignment — axial | init → manipulation stance → close both grippers → left gripper `brace` at the recorded stop edge → right gripper `push` at the long-axis end along the arc to the aligned yaw → retreat both. |
| IORS ontop / under (large-scale) | init A → manipulation B (transition) → `grasp` → lift (orientation kept) → `navigate_to` placement C with the object attached (recorded path, retreat transition before lowering the trunk toward the floor band for under) → `place` at the goal cell → retreat. |
| IORS ontop / under / inside / nextto (small-scale) | init → manipulation stance → `grasp` → lift (orientation kept) → `place` at the recorded goal pose (centre of the partner; nearest satisfying position for nextto; inside: drop into the fillable volume) → retreat. |
| IORS touching | as nextto to the recorded nextto position, then `brace` the partner's far side with the left gripper and `push` the target toward the partner with the right gripper until `touching` → retreat both. |

### 6.3 Language annotations

Each oracle action yields (1) a fine-grained annotation over the involved objects' `display_name`s, disambiguated by
the per-object descriptors recorded at sampling time ("the cereal box on the left"), and (2) the same with the
concrete numbers of the action ("push the towel rack up 3 cm at its left end"). Each demo carries (a) the task family
name, (b) the family-level description shared by all instances, (c) the template-level description with the involved
categories, and (d) the instance-level description with the goal numbers (a replacement for the goal image).
Descriptors must single out the object among the objects in view: when two texture variants of one geometry (or two
look-alikes, spec §3.12) are both visible, the descriptor carries a colour / print word or a spatial relation
("the blue mug", "the mug nearer the tray"), never the bare `display_name`.

### 6.4 Real-time QA (hard failures, `run_isg`)

Binary success not reached, measured error above tolerance, robot or object motion after the final no-op window,
collision (whole body vs scene / objects other than the intended contacts), ghost hand (assisted-grasp artefact),
prolonged pause (except the final stop), gripper out of the head camera's FOV during a manipulation action, failed
grasp for grasp actions (not checked for push / brace actions), and a trajopt failure between recorded key poses
(`plan_failed`, 3.9). A hard failure ends the episode; the checkpoint of the last successful action allows one retry
of the failed action (3.8). "Failed grasp" means the assisted grasp did not attach (both pads never touched the
object); an asymmetric closure — one finger at 0 mm, the other at 30–50 mm — is the normal `one_finger` artefact of
position-controlled fingers under assisted grasping and is not a failure.

Categories, as the executor's verdict / `SkillError.reason` (F3 fixes the final list): `not_reached` (binary goal),
`tolerance`, `not_settled`, `collision`, `ghost_hand`, `paused`, `gripper_out_of_fov`, `failed_grasp`, `no_motion`
(push / articulation moved nothing), `plan_failed`, `execution` (a motion did not converge), `timeout`. QA
statistics are reported per category and per action name, so a sampler-side cause and a planner-side cause of the
same symptom stay distinguishable.

Owner decision 2026‑09‑16 17:10 — **gripper‑in‑FOV rule (b), continuous per grasp → release span** (supersedes the
per‑action arming below): for each acting arm the monitor is enabled when the pre‑grasp pose is reached and stays enabled
through grasp, lift, the carry to the pre‑place, the descent and the release step; disabled after the release (retreat)
until the next pre‑grasp; each grasp → release pair is its own span (disembedding's grasps, block design's blocks, the
flip‑on‑table re‑grasp); contact spans (brace / push) run pre‑contact → end of the contact leg. The 5‑consecutive‑steps
threshold and the fingertip pinhole projection with the 5 % margin stay. The carry leg (lift end → pre‑place) is a
straight Cartesian leg at constant orientation (or a recorded `carry` waypoint) — no free trajopt swing. Regression
reference: nextto 0916‑152503 (transit to x 7.25, 106/429 place steps out of frame) must fail `gripper_out_of_fov` at the
place transit under the rule and pass after the carry‑leg change. **B1K comparison:** the replay QA
(`joylo/gello/utils/qa_utils.py::FieldOfViewMetric`, `gripper_changes_outside_fov_limit=0`) requires the gripper in view
only at grasp / release toggles, measured as any‑gripper‑link visibility by instance segmentation, and would pass that
episode; ours is deliberately stricter. **Implementation (oracle, 2026‑09‑16 20:xx):** `qa.Monitors` keeps one **span
per arm** (`fov_span`), opened / closed by the execute functions through `ctx["hooks"]["fov"]` (`set_fov_enabled` acts
on the action's arms): Grasp opens it after `arm_to(pre_grasp)`; Lift, Reorient, Articulate, BaseFollow and a Transition
/ Navigate made while holding leave it alone; Place closes it right after the assisted grasp has released (its own
straight retreat and the Retreat action run with the span closed); `Retreat(release=True)` closes it after the release;
Push / Brace open it after the pre‑contact and close it after the contact leg (the push's arc + extension + back‑off, the
brace approach). The monitor runs during every action (`fov_monitored` is true for all; a replay stream passes
`fov=False`); the acting arm's frames while its span is closed are `fov_idle_out` events. Spans are part of the
`Checkpoint` (`fov_spans`) and restored on a retry. **Carry leg:** `Place.targets["carry"]` is the straight leg from the
hand's actual pose at the start of the place (the lift end) to the recorded pre‑place — orientation SLERPed, a pose every
5 cm — through the sampler's `grasps[k].carry` waypoint when one is recorded (core `dc6a1a5`); `execute_place` tracks it
with differential IK and the free‑leg `arm_to` finish (as lift / retreat), `notes.carry_leg`; `run_isg --carry-leg
trajopt` keeps the former free transit as the regression reference. **Placed‑object gate:** `placed_orientation` (full orientation error vs the
recorded tolerance) is the demo gate and the `run_isg --mode oracle` default; the `placed_tilt` variant is withdrawn (the
nextto roll happened during finger closure, not at release; the code path stays selectable, unused). **Cylinder
equivalence (owner fallback, applied 2026‑09‑16 20:00 after the closure investigation ended without its criterion —
`runlogs/closure_probe_20260916/report.md`):** inside this gate an object whose model has a *continuous symmetry about
its long axis* in the ICL symmetry survey is compared **modulo rotation about that axis** — the gate error is the angle
between its long axis now and at the recorded goal (nearest recorded equivalence), against the same tolerance; the
evaluator's full error is reported alongside (`demo_gating.placed_orientation.<obj>.full_error`). The tag table is
`ispatialgym/assets/icl/revolution_axes.json` (`scripts/symmetry/revolution_axes.py` over `runlogs/symmetry_20260915/
symmetry.json`: a body‑of‑revolution tag `xinf` / `yinf` / `zinf` in `symmetry`, `symmetry_near`, `geom_exact` or
`geom_near` about the model's long axis — largest native extent ≥ 1.5 × the middle one; geometry alone counts, a printed
label breaks the visual symmetry but not the rolling): markers, pencils, bottles, cans, shakers, candles (262 models);
plates, bowls, boxes and every other object keep the full error (`ExecutorConfig.demo_revolution_axes` switches the
equivalence off). The TRO targets and the policy evaluation are untouched.

Owner decisions 2026‑09‑16: **`gripper_out_of_fov` is armed unconditionally** from the pre‑pose of every manipulation
action (grasp, lift, place, push, brace) to the end of the action — a leg whose fingertips never enter the frame is
itself a verdict (`fov_first_entry_step: null`); the transit to the pre‑pose and the retreat legs stay unmonitored, and
a recorded pre‑pose on the 5 % margin is a sampler defect (the sampler's frame check includes the pre‑poses).
**Placed‑object orientation gates demo collection only**: `run_isg --mode oracle` (`--demo-gating placed_orientation`,
default on for oracle runs, off for replay / policy) fails a demo whose placed object ends outside its recorded
orientation tolerance (`placed_orientation`); the TRO targets keep `gating: false`, so policy evaluation is unchanged;
position stays non‑gating. **Brush warning (conditional, wall alignment)**: if, with the recorded drive‑in posture
applied to every free arm, an idle hand (no acting arm, nothing held) still touches the task support (wall / table)
during `transition` / `set_torso` briefly (< 3 × the monitor's consecutive‑step threshold) with no penetration and no
displacement of the support, that contact is a *brush warning* in `meta.json` / `annotations.json`, not a failure;
every other contact (held object, non‑support objects, acting‑arm contacts outside the finger exemption) stays a
`collision` verdict. **Brace retry ladder**: a braced push that ends outside tolerance restores the checkpoint before
the brace and re‑braces at +5 / +10 mm outward (target touched the brace before the goal: `short`) or −5 / −10 mm
inward (residual sign flipped: `overshoot`), re‑pushes the residual, records `meta.brace_retry`; two retries, then
`tolerance`.

### 6.5 `run_isg` modes and recording

One script loads an instance (TRO + sidecar + scene variant), then either (i) feeds actions from a policy, (ii)
replays actions from a LeRobot episode, or (iii) runs the oracle (`expand` + the executor loop of 3.8); the same
loader, executor loop and evaluator serve all three, and the recorder hangs off the executor's `env.step`.
Recording modes: demos → hdf5 replayable by the BEHAVIOR tooling **and** the LeRobot v3 ingredients (observations,
actions, annotations) written by the same wrapper (no second replay pass; see
`OmniGibson/omnigibson/envs/lerobot_data_wrapper.py`); QA-only → the rollout observation video of `replay_data.py`.
Goal images (instance conditioning) are captured by `run_isg` with the robot and every task object stationary at the
goal configuration, after ≥ 3 renders (10, as the playback wrapper does), with the physics-synced sensor pose stored
as `goal_observation.camera_pose_world`, the evaluation camera profile (`isg-v1`), and nothing hidden: goal images
are rgb and never needed the sampler's segmentation crash guard (spec §3.4, B10 — the guard is a renderer flag for
the sampler's `seg_instance` renders, no prim is hidden anywhere).

## 7. Running it

The snippet below is the **v1** API and still runs. The v2 entry point is `run_isg --mode oracle <instance>` (TODO
F1), which loads the TRO + sidecar, expands the alias's program and runs it through the executor (3.8); none of the
v2 modules (`robot/conventions.py`, `oracle/actions.py`, `oracle/programs.py`, `run_isg`) exists yet.

```bash
conda activate ispatialgym
pip install -e /home/minyeongk/Desktop/projects/ispatialgym --no-deps   # once
```

```python
from ispatialgym.oracle import r1pro_model as M           # import the oracle stack before omnigibson
from ispatialgym.oracle.skills import Skills, run
from ispatialgym.oracle.tasks import CameraRelocalizationOracle
import omnigibson as og

env = og.Environment(configs={"scene": {"type": "Scene"}, "robots": [M.joylo_robot_config("robot")]})
sk = Skills(env)                       # builds cuRobo planners (right_arm, base) ~40 s incl. warm-up
run(env, sk.ready_arms())
run(env, sk.navigate_to((1.0, 0.5, 0.0)))
run(env, sk.approach(obj)); run(env, sk.grasp("right", obj))
run(env, CameraRelocalizationOracle(sk).run(goal_camera_pose))
```

Logging: the `ispatialgym` logger prints to stdout (`ISPATIALGYM_LOG_LEVEL`).

## 8. Open items

* Speed / smoothness tuning against human JoyLo statistics (embodiment velocity scales,
  tracker gains, trunk rates).
* The v2 cut (TODO F5 → F6 → F2): conventions out of `skills.py` / `tasks/camera_relocalization.py` /
  `sampling/robot_utils.py` into one module; `Skills` reduced to the Layer-1 motions of 3.2; the action vocabulary
  and the 14 programs; the three v1 oracle classes retired once `run_isg --mode oracle` reproduces their section 5.1
  results on the same goals.
* Checkpoint mechanism (3.8): verify that OmniGibson's `scene.save_state()` / `load_state()` round-trips the
  assisted-grasp bookkeeping (`_ag_obj_in_hand`, AG constraints) and the position-controller targets; otherwise the
  checkpoint has to store and re-apply them explicitly.
* Gripper-in-FOV monitor (spec §3.9, 6.4): FK-project both eefs into the head camera every step through the
  conventions' `camera_pose`; 5 % frame margin; only during manipulation actions.
* `plan_failed` rate in the pilot (3.9) decides whether `arm_to` stays a planner or becomes a tracker of recorded
  `q_traj`.
* Superseded by the sidecar: grasp candidates for arbitrary assets (grasp library via `grasps[]`), handle /
  attachment meta-link lookup (icl `require/handles.csv`, TRO `attachments`), arc-following helper (recorded arc
  waypoints). Still wanted: a smoother waypoint-following profile for `track_path` (heading toward the path, blend
  to the goal yaw at the end).
* Planner build time (~10-15 s per embodiment) and GPU memory: build lazily, one process per
  worker.
