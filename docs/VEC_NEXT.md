# vec_next — the integrated vectorized evaluator (2026-10-29)

Worktree `ispatialgym-eval-vec-next` (branch `vec-eval-next` at f4c0d23c; working files = the vec_speed worktree's state of 2026-10-29 21:55
plus the changes below, uncommitted like the campaign worktrees). Behavior-1K 6a6194600 + `scripts/vec_eval/b1k_overlay/` (apply with
`scripts/vec_eval/apply_b1k_overlay.sh <Behavior-1K dir>`). Record: `$V/campaigns/vec_next/{DECISIONS.md,verify/,runs/}` on the cluster,
`$V/docs/EVAL_RECORD.md` §11 entry of 2026-10-29 22:15.

The whole campaign (rounds A-F, the restore fix, the build options, the evaluated success step, every experiment with its script and
result) and how to replicate it: [scripts/vec_eval/next/README.md](../scripts/vec_eval/next/README.md). This file stops at round D.

## 1. What changed, and where

| item | where | default / switch |
|---|---|---|
| World build no longer linear in the clone count: an asset's preparation (decrypt, hash, asset-path rewrite, articulation-root pre-apply, re-export) runs once per process and every later object made from it - the clones' copies, scene 0's second preparation - replays the derived scale / kinematic_only / `articulated__` prefix and references the same temp USD | overlay `objects/usd_object.py` (`_prepare_to_load`, `_prepare_cache_key`, `_PREPARED_ASSETS`) | on; `ISG_VEC_PREPARE_CACHE=0` = stock |
| A clone's object loop runs inside one `batched_usd_edits()` scope: one USD→Fabric stage sync per clone instead of one at the exit of every `editing_usd()` block (every API apply, xform rewrite, attribute write; 42 % of an N=8 build and growing with the stage). Scene 0 keeps the stock per-edit sync, so the single-env path is unchanged | overlay `scenes/scene_base.py` (`_load_scene_prim_with_objects`) | on for scenes idx > 0; `ISG_VEC_BATCH_USD_EDITS=0` = stock |
| Collision-mesh volume / centre of mass / oblong check computed once per process per (scene-relative mesh path, scale) - by scene 0 - and replayed for the clones (the trimesh conversions were 20 % of an N=8 build) | overlay `prims/rigid_prim.py` (`update_meshes`, `_ISG_MESH_CACHE`) | on; `ISG_VEC_MESH_CACHE=0` = stock |
| Build-phase timers (`ISG_VEC_BUILD_TIMER <phase> <s>` lines: env.load_scene / load_robots / reset, sim.scene_load[i] / import_play / scene_initialize[i], scene[i].prebuilt_usd / systems / add_objects / init_load_state / init_save) | overlay `utils/isg_build_timer.py` + wrappers in `envs/env_base.py`, `simulator.py`, `scenes/scene_base.py` | off; `ISG_VEC_BUILD_TIMERS=1` |
| The dataset's 996 particle systems are no longer instantiated per scene: names listed once per process, a scene's system created on first access | overlay `scenes/scene_base.py` (`_LazySystems`, `_load_systems`) | on; `ISG_VEC_LAZY_SYSTEMS=0` = stock |
| Assisted grasp: one finger-contact view shared by every robot, its contact data read once per physics step (the k-th concurrent grasp no longer fails k times) | overlay `robots/robot.py` (co-location worktree 9f151f78, copied as is) | on; `ISG_AG_SHARED_CONTACT_VIEW=0` = stock |
| Rooms override: a world loads the rooms its demos can see instead of the template's task + adjacent rooms, for the 14 aliases that passed the campaign-rooms gate | `run_isg.rooms_for_template` (by template variant, both evaluators), `vec_loader.VecInstanceLoader(load_rooms=list)`, `vec_rooms.py`, driver `--rooms-override` / `--load-rooms`, worker `proc_env` | `ISG_ROOMS_OVERRIDE=<file>`; the worker defaults to `scripts/vec_eval/rooms/rooms_override.json`; how-to for new templates: `docs/ROOMS_OVERRIDE_HOWTO.md`, tools `scripts/vec_eval/rooms/` |
| Clone placement: centred grid, 5 m margin (the 10-27 fix: free-standing objects at world x ≥ 157 m crept in yaw under the +x/+y grid) | driver `--layout grid-centered --scene-margin 5` | default; the 10-23 production placement = `--layout grid --scene-margin 10` |
| Wall clock: episodes are limited by their step budget only; a hung simulator is caught by a stall guard (no batch step for 30 min → exit 75, `RUN_ISG_VEC_STALL`) | driver `--wallclock 0` (unchanged), new `--stall-timeout 1800`; `vec_stream.VecStreamExecutor(stall_timeout_s)` | default |
| Comparator angle wrap: joint deviations are `min(|x-y|, 2π-|x-y|)` (`max_abs_unwrapped` kept) | `scripts/vec_eval/compare_slots.py` (vec_speed fix 2) | always |
| Speed levers adopted as defaults: per-step QA monitors only with `--qa`; video composed on the sink thread; lean step (no per-object `update_visuals` loop); no external camera in `--mode policy` (replay keeps it) | driver defaults (`--monitors`, `--video-compose thread`, `--lean-step`, `--external-camera`) | `--monitors`, `--video-compose cpu`, `--no-lean-step`, `--external-camera` restore the 10-23 path |
| **Small-object trembling, part 1 (sleep threshold):** `physxRigidBody` / `physxArticulation:sleepThreshold` of every free non-articulated object = 0.005 (the PhysX default; OmniGibson sets 5e-5 on everything). At 5e-5 a placed light object's TGS residual (kielbasa pack: 0.1-0.8 rad/s) exceeds the threshold, so it never sleeps and, by contact, neither do the platter and the free table under it; in some clone configurations (not a function of the clone's coordinates) the rocking never decays and the `pose_settled` gate fails as `not_settled`. At 0.005 that stack sleeps within 0.4 s of the placement in every slot (kielbasa N=2 same-host experiment: stock FAIL 24 mrad, 0.005 / 0.01 / 0.02 PASS). Articulations and the robot keep 5e-5. | driver `--rigid-sleep-threshold` → `vec_loader._apply_physics_overrides` (the scene's objects, after the build) and `vec_loader._apply_rigid_sleep_threshold` (each slot's task objects, at every TRO restore - the TRO adds them after the build; every slot's `meta.loader.rigid_sleep_threshold` lists the objects set / skipped); both passes are `oracle/sleep_assist.py`'s, which `run_isg` applies at the same points since 2026-11-02 (train-fast-vecnext a977ac9e) | default 0.005; `--rigid-sleep-threshold 0` = stock |
| **Small-object trembling, part 2 (the sleep assist):** the teapot (lid-on-teapot alias) never sleeps at ANY threshold up to 0.05: its 56-vertex convex-hull base chatters on the bar's 56-vertex top hull at 30-60 mrad/s under OmniGibson's 32 position / 1 velocity TGS iterations - a 120 Hz contact chatter with no net motion (tilt < 1 mrad at N=1) that in some clones grows into rocking (teapot 0.7-31 mrad between slots at the end, `settled` only in 3 of 8). The lid is innocent: the attachment snaps it 5 mm into the rim opening where it touches nothing (contact partners printed by `probe_stack.py`). PhysX's sleep filter sums the per-frame velocities (`ScBodySim.cpp updateWakeCounter`) and never fires on it, while a forced `obj.sleep()` sticks (probe F1: nothing wakes the pair). Solver iterations only move the odds: 48/8 and 64/8 slept at N=1 (the drop came to rest before the chatter ignited) but 64/16, 128/16 and 64/8 with a 6 mm drop did not, and at N=8 64/8 left 5 of 8 slots rocking up to 10 mrad/step (`experiments/teapot_iters_n8/i64v8`). Stabilization segfaults in this Kit even when enabled before the first play; damping / restitution / filtered-pair edits mid-play invalidate the PhysX tensor views (segfault). **The fix is deterministic rest detection (the sleep assist):** every free, non-articulated task object that is awake, has moved less than 1 mm / 3 mrad over the last 30 control steps, does not touch the robot and has no end effector within 5 cm of its bounding box is put to sleep (`oracle/sleep_assist.py` `SleepAssist.step`, called by `SlotRecorder.record`; PhysX putToSleep); a slept object is woken again as soon as an end effector comes within 5 cm of its box - without that wake the gripper (an articulation link) does not wake a slept body in time and the contact resolves a deep penetration at once (round D's first attempt: the brussels sprout spun 180-385 mrad per step when the gripper came back to the bowl; a rigid lid landing on the slept teapot woke it cleanly). Two more guards from the second attempt: objects heavier than 5 kg (free furniture) are never assist-slept, and a body is not slept while an awake tracked free body rests on it (touching and higher) - putting the breakfast table to sleep under the awake platter + kielbasa pack jolted the pack 20-94 mrad per step and left it 0.43 rad off in 2 of 8 slots. The proximity test counts only a MOVING end effector (≥ 0.3 mm per control step): a gripper parked 3.6 cm from the released pack must not block its sleep for the whole tail. Kielbasa itself stays marginal for a different reason: the retreating gripper brushes the pack at step 749 in the demo itself, the re-settle runs to step 805-807 and the PoseSettled window (velocity-still on every one of the 159 steps of the settle phase) barely fits - round C's N=1 passed at 948 by a hair, other hosts / slots do not. On the stock trajectories the rest test is met in every slot 10-50 steps after the release, before the rocking grows (`assist_window.py`). Teapot N=1: teapot asleep at 30, woken by the passing arm at 292 / 628, lid asleep at 806, teapot at 814, success at 1002 like the demo; N=8 (8 identical slots, lambda and cluster): 8/8 success at 1002, strict comparator PASS (stock: 3/8, rocking up to 31 mrad between slots). Brussels sprouts (the kick case) with the final code: N=1 trajectory identical to stock step for step, N=8 8/8 at 1183, strict PASS. Clamp mode (zero the velocities every control step, stay awake) held the teapot stack in 5 of 8 slots only. | driver `--sleep-assist W,POS_MM,ROT_MRAD` (`ISG_SLEEP_ASSIST`; per-object sleep steps in `meta.sleep_assist`); `--rigid-solver-iterations POS,VEL` kept as an opt-in knob (`rigid_solver_position_iterations` / `rigid_solver_velocity_iterations`, task objects at the restore + scene objects at the build); `scripts/vec_eval/probe_stack.py` (the lid dropped onto the teapot, auto-attach, per-step asleep / speed / tilt / contact partners, knobs) | defaults `--sleep-assist 30,1,3 --sleep-assist-mode sleep --sleep-assist-wake-radius 0.05 --sleep-assist-max-mass 5` (adopted 2026-10-30; round D); `--sleep-assist off` = stock; `--sleep-assist-mode clamp` = the velocity clamp |
| Physics experiment knobs: `--physics-override KEY=VALUE` (solver, broadphase, gpu_dynamics, num_threads, enhanced_determinism, sleep_threshold, solver_position_iterations, solver_velocity_iterations, stabilization), `--scene-offsets X,Y,Z;...` (explicit clone offsets), `ISG_SLEEP_TRACE=1` (per-slot sleep states and speeds of the tracked objects in the replay log), `scripts/vec_eval/probe_sleep.py` | driver, loader, overlay `scenes/scene_base.py` | off |
| Tiled rendering: untouched (policy cameras: no-go, `docs` of campaign vec_speed); `--tiled --tiled-group name+external` for verification / benchmark units only | overlay `sensors/tiled_sensor.py`, driver `--tiled*` | off |

Kept per environment on purpose: the Python object wrappers (predicates, TRO restores, hinge constraints, the assisted grasp and the N=1
bit-identity gate depend on them). The per-step pose readback was already one PhysX tensor view over every scene (`RigidBodyViewAPI`,
pattern `/World/scene_*/*/*`, read once per step), so the "Isaac Lab style" change is the build side only; GPU-pipeline physics was not
adopted because it voids the N=1 reference gate.

## 2. Build time (lambda RTX 4090, centred grid 5 m, same demo in every slot)

Kit app-data (shader + texture cache) is per Behavior-1K checkout; every number below used the same warm app-data (the speed checkout's) unless
marked cold. "Before" = the vec_speed worktree's code; change 1 = memoised asset preparation + lazy systems; change 2 = + batched Fabric sync
for clone loads + collision-mesh cache. Replay = the eval instance's demo in every slot, open loop.

| world | N | before | change 1 | change 2 | env-steps/s before → after |
|---|---|---|---|---|---|
| apple (house_double_floor_lower, 244 objects / scene) | 1 | 70.3 s | 66.1 s | 66.1 s (scene 0 is the stock path) | 14.7 → 16.8 |
| apple | 8 | 595.4 s (first N=8 in a fresh checkout: partly cold) | 512.3 s | **355.0 s** | 26.1 → 31.9 |
| hamper (house_single_floor, 426 objects / scene) | 1 | 110.9 s | 106.4 s | 106.4 s | 12.8 → 14.5 |
| hamper | 8 | 1041.4 s | 954.8 s | **621.6 s** | 21.1 → 22.9 |

Per clone (apple N=8, phase timers): objects 36-43 s growing with the stage → 14 s flat; robot 8.5 s (unchanged: its post-load reads its own
just-added prims from Fabric, so it keeps the per-edit sync); PhysX play share ~5 s; reset / sensors / initialize ~3 s. The remaining slope is
≈ 30 s per clone on the 4090 (was ≈ 55 s), i.e. an N=8 world costs ~5 N=1 worlds instead of ~8.

Where the build time went before change 2 (py-spy, apple N=8): 42 % in `Simulator.editing_usd.__exit__` = `SynchronizeToFabric()` of the whole
stage at the exit of every small USD edit (API applies, xform rewrites, attribute writes), 20 % in the trimesh conversions of `RigidPrim.update_meshes`,
15 % robot loads, 8 % `play`. cProfile is blind past the Kit launch (it resets the profiler hooks); py-spy must launch the process (ptrace_scope 1).

Exactness: the world state after the build (`dump_state` of every scene) is bit-identical between two builds of the same code; scene 0 is
bit-identical between change 2 and stock; a clone's state differs by ≤ 1e-5 m / 1e-12 velocity residuals (the xform round trips of its prims
read the Fabric synced before the batch instead of after each parent's op rewrite: a different float rounding of the same transform at a ~50 m
clone offset). Replays at N=8 vs N=1 pass the gate for apple and hamper.

Cluster (A40 / L40S, 48 GB): the same worlds build 2-3x slower when the assets are read from NFS (round-1 jobs whose staging fell back: N=8
1400-2600 s) than from the node-local asset cache (`verify.sbatch` stages the bundle once per node: 450-1530 s before change 2).

## 3. Re-verification of the 53 aliases at N=8

`$V/campaigns/vec_next/jobs/verify.sbatch` (one alias per 48 GB viscam GPU: n1, n8same, compare → `verify/<alias>.json`, the 10-23 record's
verdict referenced); `scripts/vec_eval/next/verify_record.py` composes the verdict file. Pass rule unchanged (`verify_rules.py`): identical
verdicts and success steps within 5 steps, trajectories within 2 cm / 0.05 rad, angle-wrapped joints.

Four rounds: A = the change-1 overlay (`$V/campaigns/vec_next/{runs,verify}/`, 35 aliases, interrupted when change 2 was ready; 28 PASS), B = the final
overlay with the stock sleep threshold (`Behavior-1K-vec-next-b-6a6194600`; `{runs_B,verify_B}/`, all 53: 47 PASS, the 6 FAIL = the trembling light
objects + the tablespoon comparator case), C = the same with the sleep-threshold default (`{runs_C,verify_C}/`, all 53) plus the N=1 parity phase:
the new N=1 vs round B's N=1 under the 10-23 pass rule (identical verdicts / success steps, trajectories within 2 cm / 0.05 rad); D = C plus the
sleep assist (`--sleep-assist 30,1,3`, sleep mode with the 5 cm wake, stock solver iterations; `{runs_D,verify_D}/`, all 53, N=1 parity vs round B; the
first attempt without the wake is kept as `{runs_D_sleepmode,verify_D_sleepmode}/`, 11 + 3 verdicts; the second, without the supporter guards, was
cancelled before its first verdict). Round C: 53 verdicts, 47
PASS strict / 50 relaxed (comparator-only: cap yaw, shaker yaw, spoon roll); FAIL teapot (rocking, 2-5 slots unsettled), sunflower (lean set at
the placement, 0.27 rad, N=1 parity FAIL too), kielbasa (its pack still at 5e-5 in that round). Round D result (2026-10-30 17:50 PDT, `verify_D/SUMMARY.md`): **53 aliases, 49 PASS strict, 51 PASS relaxed**; N=1 parity vs round B's
stock-physics N=1: 50 of 53. FAIL strict + relaxed: sunflower (lean set at the placement, 0.27 rad), kielbasa (the demo's own settle margin after
the gripper brushes the pack at step 749). Comparator-only (relaxed PASS): cap-lemonade (cap yaw), tablespoon (spoon roll). Fixed vs round C:
the teapot lid (3/8 → 8/8, strict PASS) and the salt shaker. 12 aliases the 10-23 record had verified below N=8 pass at 8.
