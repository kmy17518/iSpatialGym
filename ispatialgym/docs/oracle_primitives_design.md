# iSpatialGym oracle primitives for R1Pro: design and implementation notes

Status: first implementation validated in simulation, 2026-09-04. Code: `ispatialgym/oracle/`.

This document explains how the scripted (oracle) demonstration policies for the seven
iSpatialGym tasks are organised, why the primitive set is arranged hierarchically rather
than as a flat per-task list, how joint groups are restricted when planning with cuRobo,
how base / torso motions are made to look like JoyLo teleoperation, and what was learned
while getting it to run.

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

## 3. Hierarchy instead of a flat per-task list

The brainstormed table has ~15 primitives, several of which are the same motion with a
different frame (`move_eef global/local`, `translate_eef_along_axis`, `rotate_eef`, `push`,
`release(xyz, xyzw)` ...). Three layers remove the duplication and isolate everything
cuRobo-specific in one place:

```
Layer 2  task oracles   tasks/camera_relocalization.py, tasks/held_object.py (object scaling, mental
                        rotation); disembedding, configuration matching, world alignment and IORS are
                        short compositions of the skills below (section 6)
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

### 3.2 Layer 1 skills (all yield 23-D actions)

| Skill | Joints that move | How | Replaces (brainstorm) |
| --- | --- | --- | --- |
| `navigate_to(pose2d)` | base x, y, yaw | cuRobo `base` embodiment plans a smooth collision-free `(x, y, yaw)(t)` for the *whole body* at its current arm/torso configuration (fallback: traversability-map A* / straight line, always whole-body collision-checked). `base_tracker` turns it into base-frame velocities: feed-forward + pose feedback, clipped to joystick limits, acceleration-limited, exponentially smoothed. Translation and rotation happen simultaneously - no translate-rotate-translate. | `navigate` |
| `set_torso(translate, tilt, delta)` | torso 1-3 (+3) | JoyLo keyframe curve (`trunk.py`), scalars rate-limited exactly like holding the JoyLo controls. Swept motion is whole-body collision-checked; if hanging arms would hit the base it first calls `ready_arms`. | `raise_torso`, `tilt_torso` |
| `ready_arms()` | arms | Planned motion to `ARM_READY_Q` (hands in front, elbows bent; found by IK with max joint-limit margin), with lift-first and Cartesian fallbacks. What a JoyLo operator does before working. | - |
| `move_eef(arm, pose, torso=False)` / `move_eefs({arm: pose})` / `move_joints(q)` | 7 (14) arm joints, +4 torso if `torso` | cuRobo IK + trajopt, self/scene collision, grasped object attached; interpolated at 1/30 s, streamed, hold-to-converge. | `move_eef`, bimanual |
| `move_eef_cartesian(arm, pose / delta / waypoints, torso=False)` | arm (+torso) | Dense straight-line / arc waypoints -> damped-least-squares differential IK (finite-difference Jacobians from batched cuRobo FK, saturated-joint handling) -> collision validated -> streamed at a configurable eef speed. Guarantees the path shape trajopt does not. | `translate_eef_along_axis`, `push`, articulation follow |
| `rotate_eef(arm, delta_rot, pivot=eef/object/point)` | arm (+torso) | Special case of the above with SLERP about a pivot; `pivot="object"` keeps the held object centred (Mental Rotation). | `rotate_eef` |
| `set_gripper / open_gripper / close_gripper` | fingers | Streams +1/-1 until fingers stop or assisted grasping engages. | `grasp`, `close_eef`, `release` |
| `grasp(arm, obj, approach=top/front)` | composite | `grasp_candidates` (vertical + 30/45 deg tilted approaches x wrist yaws, fingertips kept above the object bottom) -> first IK-feasible grasp + pre-grasp -> planned pre-grasp -> straight approach -> close -> attach to planner -> lift. | `grasp` |
| `place(arm, eef_pose)` / `place_object_at(arm, object_pose)` | composite | Reachability check (repositions via `stance_for` if needed) -> planned pose above -> straight descent (planned fallback) -> release -> retreat. `drop_height` avoids pushing the object into the support. | `release(xyz, xyzw)` |
| `push(arm, contact, direction, distance)` | composite | Closed gripper, planned standoff, straight motion through the contact. | `push`, world-alignment nudges |
| `brace(arm, pose)` | other arm | Closed gripper + planned move: support hand for World Alignment / articulation / attachment. | bimanual support |
| `stance_for(target)` / `approach(target)` | base + trunk | Chooses base pose (facing the target, target on the arm's side) and JoyLo trunk level from candidates that are whole-body collision-free *and* have IK-feasible grasp + pre-grasp, then executes trunk/drive in a collision-safe order. Mirrors how operators position the robot before reaching. | (new; needed, see 5.2) |

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
| `right_arm_torso` / `left_arm_torso` | 4 + 7 | eef | fallback for hard reaches (held-object oracles use it when the arm alone cannot track) |
| `whole_body` | 3 + 4 + 14 | eefs, `zed_link`, `base_link`, `torso_link4` | FK / collision queries (kinematics-only, no planner needed) |
| `base_torso` | 3 + 4 | `zed_link`, ... | viewpoint queries |

Per-embodiment `velocity_scale`, `max_acceleration`, `max_jerk` (in `EMBODIMENTS`) set the
planned motion speed; the defaults are deliberately slow (arms 0.25 x URDF limits, base
0.45 m/s) to look like teleoperation and should be tuned against human JoyLo statistics.

Design decision: for demos the torso is **not** freed while reaching. JoyLo operators lower the
trunk along the keyframe curve, then reach. `stance_for` chooses the trunk level, `set_torso`
applies it, and the arm is planned with the torso locked. `*_arm_torso` is used only as a
fallback when a motion is otherwise infeasible (e.g. Object Scaling toward the camera).

Unused tool frames of a multi-frame planner get a zero-weight `ToolPoseCriteria` (cuRobo then
treats them as trivially converged); joint-space goals use `plan_cspace`.

### 3.4 Execution model

* Skills are generators; `skills.run(env, gen)` steps the env. Composites `yield from`.
* `ActionState` holds targets (trunk q, both arm q, gripper commands); every yielded action
  fills all 23 dims, so position-controlled joints never drift. Base velocity defaults to zero.
* Convergence checks with step budgets raise `SkillError(reason, ...)`; task oracles retry with
  other parameters.
* Collision world: all non-visual-only object collision meshes (expressed in the robot root
  frame, fixed during an episode) plus a ground slab; the grasped object is sphere-fitted,
  attached to `attached_object_<arm>_eef_link` and removed from the world.

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

## 6. Task oracles

| Task | Script |
| --- | --- |
| Camera Relocalization | `tasks.CameraRelocalizationOracle`: `solve_viewpoint` (grid + refinement over trunk translate/tilt for a planar base pose) -> `go_to_stance` -> `set_torso(tilt)` -> corrective `navigate_to` on the residual. |
| Object Scaling | `tasks.ObjectScalingOracle`: target position on the camera->object ray (from a distance or an apparent-size ratio) -> straight-line eef translation, fallbacks arm+torso / planned. |
| Mental Rotation | `tasks.MentalRotationOracle`: rotation to the target orientation in <= 60 deg chunks, `rotate_eef(pivot="object")`, same fallbacks. |
| Disembedding | `approach(target)` -> `grasp(top)` -> `place_object_at(tray pose)` (or `approach` the tray first). |
| Configuration Matching / block design | per slot: `approach` (if needed) -> `grasp(top)` -> `place_object_at(slot pose)`. |
| Configuration Matching / articulated opening | `brace` (left) where useful -> `grasp(handle)` -> `move_eef_cartesian(waypoints on the joint's arc/line from the privileged joint axis)` -> release. |
| Configuration Matching / attachment | `grasp(child)` -> `move_eef(above attach site)` -> `move_eef_cartesian(insert)` -> release; left hand `brace`s the parent. |
| World Alignment | wall: `brace` left gripper on the frame edge -> `push` along an arc about the pivot (waypoints via `rotate_eef` math) with feedback on the object's measured angle; desk: planar arc. |
| IORS | ontop/under/inside: `grasp` -> `place_object_at(goal)`; touching/nextto: `push` or `grasp` + `place_object_at`. |

The last five need per-asset information (handle/attach points, tray poses) that the TRO
schema will provide; the skills they need exist.

## 7. Running it

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
* Grasp candidates for arbitrary BEHAVIOR assets (currently AABB-based), handle / attachment
  meta-link lookup for Configuration Matching, arc-following helper for articulation and World
  Alignment.
* Long-range navigation in full scenes uses the traversability-map fallback; a smoother
  waypoint-following profile (heading toward the path, blend to the goal yaw at the end) would
  look more human.
* Planner build time (~10-15 s per embodiment) and GPU memory: build lazily, one process per
  worker.
