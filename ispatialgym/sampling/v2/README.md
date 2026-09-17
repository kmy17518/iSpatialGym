# `ispatialgym.sampling.v2` — the v2 sampler (spec `docs/instance_sampling_spec.md`, TODO D2–D4)

Module map and the contracts between the pieces. Several agents implement the modules in parallel; the names below
are the interfaces they build against. Every rule number refers to `docs/instance_sampling_spec.md` unless it says
"oracle doc" (`docs/oracle_primitives_design.md`).

```
sampling/
  iddl.py, customs.py, generation_defaults.py, bddl_from_iddl.py, ground_isg.py     D1 (grounding; no simulator)
  reach_bands.py                                                                     B3 (cache the sampler reads)
  sample_isg.py, multiply_isg.py                                                     D4 (CLIs over v2.core)
  v2/
    problem.py      GroundedProblem: one customs entry + its BDDL + scene bindings; TemplateChoice (models per role)
    stances.py      Stance / Transition / Path records (sidecar-compatible), edge frames, base-frame helpers
    ik.py           §3.6 IK conventions: screening (12 seeds) / exact (32 seeds), locked idle arm, ignore lists,
                    §3.9 waypoint checks, eef-in-frame; the only place cuRobo IK is called from v2
    navigation.py   §3.7 paths (locally un-eroded / eroded global + local connections, whole-body checks at
                    0.1 m strides), §3.8 stance transitions (swept trunk at s_T)
    tabletop.py     §4 Phases 1a / 1b / 1c / 2 / 3 (+ 4.7 disembedding placement stance) as functions on a sampler
    prompt0.py      §5.2 large-scale stance skeleton (Phase 1 ray, Phase 2 lattice, caches)
    program_check.py §3.5 program check: expand(alias, draft sidecar) -> IK/collision-check every target
    core.py         SamplerV2 (extends the v1 TaskSampler pipeline: settle / validate / goal feasibility / TRO)
                    with the sidecar draft, transitions, program check, reload validation, constraints write-back
    tasks/          one module per subtask family, registered with @register("<subtask id>")
      camera_relocalization.py  §6.1 / §6.2
      held_object.py            §6.3 (object scaling, mental rotation)
      disembedding.py           §6.4 (+ §4.7)
      block_design.py           §6.5
      articulation_large.py     §5.3 / §6.6
      articulation_small.py     §5.4 / §6.7
      attachment.py             §6.8
      alignment_wall.py         §5.5 / §6.9
      alignment_axial.py        §6.10
      iors_small.py             §6.11 (ontop, under, inside, nextto, touching)
      iors_large.py             §5.6 / §6.12 (ontop, under)
```

## Contracts

1. **Inputs** are the D1 artefacts only: the customs entry (`customs.py`, §2.3 schema — `roles`, `scenes.<scene>.
   {rooms, bindings, whitelist, blacklist}`, `generation`, `constraints`) and the grounded BDDL
   (`activity_definitions/<activity>/problem<N>.bddl`, id from `problems.json`). No defaults hidden in a sampler:
   every number comes from `generation` (seeded by `generation_defaults.py`).
2. **Geometry** comes from `ispatialgym.robot.conventions` (F5) — grasp / pre / contact poses, waypoints,
   transitions, base-follow, trunk keyframes, camera pose. v2 never re-implements one of those functions.
3. **Records** are the sidecar dataclasses of `ispatialgym.oracle.sidecar` (F6, spec §7). The sampler fills a
   *draft* `Sidecar` while it samples; `core.write_instance` finalises it (`tro_sha256`) next to the TRO.
4. **Checks**: every eef pose the oracle will use is IK-checked here (screening at waypoints, exact at endpoints,
   §3.6 / §3.9); the last stage of every attempt is the **program check** (`program_check.run(alias, sidecar)`:
   `oracle.programs.expand` → `Action.targets()` → IK / collision per target in the recorded posture with the
   carried object attached). A failure is `Reject("program_check_failed", action=<name>)`; nothing is patched.
5. **No fallbacks** (principle 2): candidate sets are enumerated (edges, stand-offs, trunks, feasible grasp sets,
   brace kinds, partner candidates); a failed check rejects the attempt / template with a named reason. The v1
   helpers with fallbacks (`view_arm_fallbacks`, `init_depth_fallbacks`, `stance_for`, base-yaw fallback) are not
   used.
6. **Caches**: reach volume (`sampling.eef_reach_volume.ReachVolume.load()`), reach bands
   (`sampling.reach_bands.ReachBands.load(...)`, computed on a cache miss by calling the same function the CLI
   runs), Prompt-0 survivors per (scene, target, init state), the exact-IK cache of v1 (`ISPATIALGYM_SAMPLING_CACHE`).
   Every per-model key includes the spawn scale vector (§3.10).
7. **Pipeline per attempt** (`core.SamplerV2.run_instance`): sample (task module) → apply draft → settle + BEHAVIOR
   validation (`:init` true, binary goal false) → goal feasibility (teleport + settle, robot parked per §3.3) →
   program check → TRO + sidecar → (D5) reload validation in a fresh process. Rejections are counted per reason
   and reported in `summary.json` as in v1 (`problems.<problem>.<template>.{n_written, time_s}` layout kept).
8. **Constraints write-back** (§2.3 / §9): each successful instance appends typed, scoped records (`stance`,
   `grasp_set`, `joint_set`, `placement`, `contacts`, `band_ref`) with `source` to the customs entry;
   `sample_isg` / `multiply_isg` read them as priors before drawing.

## How a task module plugs in

```python
from ispatialgym.sampling.v2.core import SamplerV2, register

@register("inter_object_relation_satisfaction-ontop_small_scale")
class IORSOntopSmall(SamplerV2):
    def bind_scene_roles(self): ...        # supports / partners / walls / rooms from customs.scenes[scene].bindings
    def sample_template(self): ...         # per-template facts: bands, stable joint set S, pair admissibility
    def sample_instance(self, rng): ...    # per-instance draws -> Draft (object poses, stances, sidecar blocks)
    def goal(self, draft): ...             # goal poses / joint targets -> pose_targets / articulation_targets
```

`tabletop.py` and `prompt0.py` expose the shared phases as functions taking the sampler (`tabletop.phase1_edge_and_standoff(s, ...)`,
`tabletop.phase1b_init_trunk(s, ...)`, `tabletop.phase1c_lateral(s, margins)`, `tabletop.phase2_manipulation_stance(s, ...)`,
`tabletop.phase3_target(s, ...)`; `prompt0.phase1_init_pose(s, face, beta_max, d_min, d_max)`, `prompt0.phase2_manipulation(s, face, contacts, lam_range)`),
so a task module composes them and adds its own rules (§6).

## Core API (D2 / D4, `core.py` and friends) — what a task module gets

`SamplerV2(ctx: ContextV2)` extends the v1 `TaskSampler`; `ContextV2(problem: GroundedProblem, choice: TemplateChoice, world, kit,
out_dir, base_seed, customs_sha256, variant)`. Attributes a task module uses:

| attribute / method | what |
|---|---|
| `s.gp` (`GroundedProblem`) | the customs entry: `gp.roles[role]` (`category`, `kind`, `grasp_pipeline()`, `scale_of()`), `gp.gen(key)`, `gp.rooms(scene)`, `gp.bindings(scene)`, `gp.constraints_for(type, **scope)` (priors), `gp.tolerances()`, `gp.bddl` (ISGProblem, None for the v1 facade) |
| `s.choice` (`TemplateChoice`) | `choice.models[role]` (`ModelChoice`: category, model, dataset, grasp_key, scale), `choice.groups[role]` (member list), `choice.bindings`, `choice.scene` |
| `s.sidecar` (`stances.SidecarDraft`) | `add_stance(Stance)`, `add_transition`, `add_path`, `add_grasp(GraspRecord)`, `add_contact(ContactRecord)`, `set_goal(base_pose2d, translate, tilt, arms)`, `describe(instance, display_name, descriptor, **goal_numbers)`, `articulation` / `blocks` dicts; reset before every attempt |
| `s.ikc` (`ik.IKChecker`) | `q_locked(stance)`, `stance_collision_free(stance)`, `stances_collision_free([...])`, `screen(poses, stance, arm, ignore)`, `screen_pairs(pairs, ...)`, `exact(poses, stance, arm, ignore_for_last, scale_key)`, `exact_both_arms(work, other, stance, ...)`, `waypoints_ok(list, stance, arm, policy, ...)`, `eef_in_frame(pose, cam_pose)`, `feasible_sets(cells, yaws, grasps, pose_at, stance, ...)` (F per cell) |
| `s.record_transition(frm, to, carried, arm)` | §3.8 sweep -> `Transition` in the sidecar or `Reject("no_transition")` |
| `s.record_path(frm, to, carried, arm, max_length_m, blocked)` | §3.7 path -> `PathRecord` or `Reject(no_path / path_collision / path_too_long)` |
| `s.park_robot()` | §3.3 parking (free cell in the task rooms, >= 1 m from the task objects) |
| `s.camera_pose_at(stance)`, `s.stance_q(stance)` | FK helpers |
| v1 helpers still available | `support_top(obj)`, `frustum_ok(cam, pts)`, `hull_corners_world(hull, pose)`, `world.footprint / bottom_offset / local_hull`, `bind_scene_object`, `spawn_role`, `tol(key, default)` |

Hooks (override): `bind_scene_roles()`, `sample_template()`, `sample_instance(rng) -> Draft`, `goal(draft)`, `after_goal_check(draft)`,
`record_constraints(draft) -> [records]`, `support_objects()`, `v1_role_aliases()` (facade only). Per attempt `core.run_instances`
runs: reset scene + park -> `sample` (= `sidecar.reset_instance()`, `sample_instance`, `goal`) -> `apply_draft` -> `settle_and_validate`
-> `check_goal_feasible` (parked) -> `after_goal_check` -> `program_check` -> `save_template` (first instance) -> TRO + `<iid>.oracle.json`
+ `<iid>.summary.json` -> `constraints_write_back`. Every `Reject(reason)` is counted (first three examples kept in `template_meta.json`
and the CLI `summary.json`).

`ik.py` also holds the F5 adapter (`pre_pose`, `grasp_pose`, `contact_eef`, `handle_eef`, `line_waypoints`): it forwards to
`ispatialgym.robot.conventions` when that module exists and otherwise applies the documented formulas — task modules call the adapter,
never the formulas.

### tabletop.py — signatures the reference task (`tasks/iors_small.py`) calls

```python
phase1_edge_and_standoff(s, top: SupportTop, rng, stand_off=0.45) -> (EdgeFrame, Stance)      # §4.2; Stance("init", ..., translate 0.5, tilt 0)
phase1b_init_trunk(s, init: Stance, H_max: float) -> Stance                                     # §4.3; init_stance with t*, tilt 0
phase1c_lateral(s, init: Stance, margins: dict, rng) -> Stance                                  # §4.4; margins = the task's row (w_t, w_p, W_p, D_p_max, safety, relation, g_NT)
phase2_manipulation_stance(s, init: Stance, target_half_extent: float) -> list[Stance]           # §4.5; survivors of the best angle class, .focus set, tilt continuous
phase3_target(s, manip: Stance, rng, grasps, hull, rest_quat, ignore_for_grasp) -> (obj_pose, GraspRecord)   # §4.6 step 1 (F at the focus, exact IK)
```

`iors_small.py` draws the manipulation stance uniformly among the Phase-2 survivors and checks the §4.5 step 5 transition itself;
it also calls `tabletop.edge_candidates` (per-edge retry when Phase 1c rejects an edge), `library_grasps`, `place_partner`,
`brace_second_arm`, `in_frame` / `hull_corners_world` / `height_of` / `half_extent` / `hull_extents`. The TEMP stand-in is gone
(tabletop landed in `957c767`); `iors_small._tabletop()` remains as the accessor `iors_large` imports.

### Stance configuration contract (what the checks lock, seed and attach) — 2026-09-15 evening

- `Stance.q` is the **recorded configuration**: `IKChecker.q_locked(stance)` takes the arm joints and the finger opening
  from it (the base virtual joints are always re-derived for the current robot root); without `q`, the posture named by
  `stance.arms` (`"untucked"` default, `"tucked"` / `"reset"` = JoyLo reset arms, `"ready"`) with open grippers, or
  `stance.fingers`. So the §3.3 exceptions are expressed by the task module setting `q` (held-object hold) or
  `arms="reset"` (camera relocalization) on the stance — no overrides in the modules are needed any more (the held-object
  module's TEMP overrides can go).
- `Stance.carried` names the bddl instance in the hand at that stance; `core.check_stances_with_objects`, the
  transition re-sweep and the program check attach it **posed in the hand** per its grasp record's `eef_in_object`
  (`navigation.attached(kit, arm, obj, eef_in_object=...)` — never at the object's current pose), with its own meshes
  ignored. `program_check._Tracker` starts with `stances["init"].carried`.
- `core.record_path` / `record_transition` look the carried object up in `sidecar.grasps` (`carried` defaults to the
  stances' `carried`) and attach it the same way; `record_path(..., force_global=True)` plans on the eroded map end to
  end; paths are checked at the **departing** stance's trunk (the trunk changes at s_T after the drive) with the
  departing stance's arms; `trunk=` overrides.
- Push / brace contacts are IK-checked with **closed** pads (`fingers=0.0` in `exact` / `waypoints_ok` / `screen`; the
  program check does this for `push` / `brace` actions); `base_follow` in base mode skips the handle poses (reached by
  the base, not from the stance); every program-check solve runs with `cache=False`.
- "Arms excluded" collision checks (§4.4 step 3, prompt0): cuRobo checks the whole sphere model and cannot mask links;
  the accepted stand-in is *arms hanging (tucked, inside the base column) with the self-collision term off*
  (`tabletop.trunk_sweep_free`) — documented here as the approximation everyone uses.
- `Stance.to_sidecar()` writes the task's `extras` next to the §7 keys (`goal_cell`, `init_surface`, `path_length_m`,
  `bearing_deg`, `deviation_deg`, ...); the F6 `Sidecar` keeps them in `extra`. Keys starting with `_` stay private.

### A6 — idle-arm posture (owner decision 2026-09-16; §3.6)

- `IKChecker.stances_collision_free(stances)` retries a one-arm manipulation stance whose untucked idle arm collides with
  the alternative idle postures (`IDLE_POSTURES = untucked, ready (ARM_READY_Q), tucked`) and records the one that passed in
  `Stance.idle_posture`; `q_locked` / the transition sweep / `check_stances_with_objects` / the program check lock the idle
  arm in that posture (`ik.stance_arm_vectors`). Init stances (`arms = "untucked"`, tilt 0) and two-arm stances are never
  re-postured; a stance with a recorded `q` keeps it.
- **Sidecar fields (proposed to the oracle agent; F6 keeps unknown keys in `Stance.extra`):** `stances.<name>.idle_posture`
  (`"untucked" | "ready" | "tucked"`, written for every one-arm stance) and `stances.<name>.arms_q = {"right": [7], "left":
  [7]}` (the joint values the checks used for both arms; `fingers` when not the open default). The oracle moves the idle
  arm to `arms_q[idle]` before the trunk motion of the transition (precedent: camera relocalization `goal.arms`).
  `stances.<name>.arms` stays `"untucked"` or the working-arm list (the F6 validator rejects other strings).
- The stance *search* lives in `tabletop.phase2_manipulation_stance` / `prompt0`; they call `stances_collision_free`, so
  the retry applies there without changes (the band script does the same per trunk — B3).

### A1 — joint work / brace placement (owner decision 2026-09-16)

`stances.in_band(point_base, cells)` / `stances.brace_placement_admissible(work_base, brace_base, right_cells, left_cells)`
(pure, 6 cm cell tolerance) gate a two-arm placement: the pushed end / work point ∈ right band AND the brace point ∈ left
contact band *of the same trunk*, then the exact two-arm IK (`IKChecker.exact_both_arms(..., fingers=FINGER_CLOSED)`).
`iors_small` applies it to touching (`touching_points` = target right face at the place position / partner far face) as a **preference** (owner 2026-09-16 17:10, the axial rule): a partner position that passes the joint gate enables the brace; the push chain alone is the requirement, `contacts.brace` is recorded only when the gate passed and a two-arm candidate passes the exact IK (`summary.brace.present: true|false` with the gate verdict / the failed candidates); push-only instances have a right-arm manipulation stance and no `contacts.brace` (program check: `contacts.push` required, `contacts.brace` optional).
The left contact band is looked up by `iors_small.left_contact_band_cells` against the APIs to come: `tabletop.ensure_
contact_bands(s, arm="left", height=h)` then `reach_bands.ReachBands.load(scene, support, mode="contact", arm="left",
height_class=h)`; until one exists the gate is recorded as `template_facts.left_contact_band = "unavailable"` and the
two-arm IK alone decides.

### C5 — several aliases per process (`sample_isg --alias A --alias B …`)

The world is reused when the **world key** `(scene, task rooms, removed chairs)` matches (`sample_isg.world_key`; log line
`world reused for <alias>` / `world loaded for <alias> in Ns`); a failing alias is caught per alias (`summary.json`
`aliases.<alias>.{time_s, world, n_written, templates, error}`, `errors[]`) and the chunk continues; `summary.json` /
`SAMPLING_REPORT` carry `wall_time_s` and `world.{n_loads, load_time_s}` for the wall-time split.
`python -m ispatialgym.sampling.sample_isg --world-key --alias A --alias B` prints `alias<TAB>scene|rooms=…|removed=…` per
(alias, candidate scene) without a simulator (18 s for the imports), so the scheduler can group aliases by key.

### Report layout changes (2026-09-16)

- `scene_reset` is a **counter**, not a rejection: `template_summary` / `template_meta.json` carry `counters.scene_reset`;
  `rejections` holds rejected attempts only (a legacy `rejections.scene_reset` is moved on read).
- The program check records the joint solution of every checked end pose: `summary.program_check.actions[].witnesses
  {<target key>: {pose, q (28), n_waypoints}}` and the sidecar `ik_witnesses[]` (per action: `stance`, `arm`, `carried`,
  `targets`); D5's reload re-check verifies them as witnesses (FK within 5 mm / 0.05 rad + collision) instead of re-solving
  the randomised IK — the same rule `IKChecker.exact` applies to its own remembered solutions.

### Sidecar semantics the task modules must keep (F6 programs read them)

- `grasps[].yaw` is the drawn **object** yaw about world z relative to the rest orientation (§4.6); the recorded `grasp` /
  `pre_grasp` poses are `compose(object init pose, T_obj→eef)` — a consumer that re-derives the grasp from the settled object pose
  passes `yaw = 0` to `conventions.grasp_pose` (its docstring says so); `grasps[].place_on` = the bddl instance the object is
  placed on / into (the F6 IORS programs require `pre_grasp`, `carry_height`, `place`, `place_on`).
- `contacts.push.object` = the pushed instance, `contacts.brace.object` = the braced instance (`program_iors_touching` checks
  push.object == grasps[0].object and brace.object == grasps[0].place_on).
- Every stance the program visits has a `transitions[]` entry from the previous one when the trunk differs, and a `paths[]`
  entry when the base moves without a transition (`programs._travel` rejects the sidecar otherwise).
- `stances.<name>.arms` is `"untucked"` (init) or the list of working arms; the program check runs every action's targets from
  the stance the program is at (tracked through `Navigate.path.to` / `Transition.transition.to`) with the object a preceding
  `Grasp` picked up attached (the grasp record's `eef_in_object` puts it into the hand).

## Running (D4)

```
OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.sample_isg --alias <alias> [--customs ispatialgym/assets/task_specs/isg_customs.json]
    [--scene S] [--templates N] [--instances-per-template N] [--seed 0] [--out <root>] [--max-attempts 30] [--no-write-back]
OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.multiply_isg --template <root>/<alias>/<template_id> --instances N [--path-buckets 3:10,5:10,10:5]
```

Reach bands: `sample_isg` / `multiply_isg` point `ISPATIALGYM_REACH_BANDS` at `runlogs/reach_bands_20260915/bands/` (the fleet
results B3 collects) when the directory exists and the variable is unset; `ReachBands.load` searches it first, then the runlog
root and `~/.cache/ispatialgym/reach_bands` — a hit logs `reach band cache hit ...` in `tabletop.ensure_bands` and skips the
in-process computation. The BDDL is written both at `<root>/<alias>/activity_definitions/...` (§1) and inside the template
directory (`run_isg.build_env_config` and `validate_instances` read the latter). The `hold` constraint type written by the
held-object module is added to `customs.CONSTRAINT_TYPES` by the D1 owner.

`sample_isg` copies the customs list to `<root>/task_custom_lists.json`, writes `<root>/<alias>/activity_definitions/<activity>/problem<N>.bddl`,
`<root>/<alias>/<template_id>/{template.json, template-partial_rooms.json, template_meta.json, instances/<iid>-tro_state.json,
<iid>.oracle.json, <iid>.summary.json}` and `<root>/summary.json`, and prints one `SAMPLING_REPORT {...}` line. Blacklists and
constraints learned go back into the source customs file (§1) unless `--no-write-back`; the `<root>` copy always carries them.
On the shared smoke pods (`scripts/runpod/remote_run.py`) pass explicit paths (`--out $SMOKE_OUT/run1`, `--customs ...`); the
reach band of a (support, object) pair is computed in-process on a cache miss (~10 min per pair on an L40S, then cached under
`~/.cache/ispatialgym/reach_bands`). `ISG_PROGRAM_CHECK=stub` limits the program check to the structural test (development only).

## Backend facts the checks depend on (found in the smoke runs, 2026-09-15)

- **cuRobo IK seeds are random** (`solve_ik` uses the seed configuration plus random seeds), so a marginal pose that passed
  once can fail when re-solved — the program check re-solves every recorded pose. `IKChecker` remembers the solution of
  every successful exact solve (per attempt) and seeds a re-solve of the same / a nearby pose (3 cm, 0.15 rad) with it;
  D5's `validate_instances` program check will need the same (or recorded joint solutions in the sidecar) to be stable.
- **`R1ProCuRobo.attach_object` fits 5–7 spheres to a small object and the fit is random per call**; the attached link is
  exempt from self-collision only against the finger links (`r1pro_model` self-collision ignore), not against
  `<arm>_gripper_link` / the wrist, so a grasp that puts the object deep in the gripper self-collides once attached —
  every attached check (lift, carry, place) then fails whatever the pose. `navigation.attached` seeds torch around the
  fit (one geometry per attempt) and `IKChecker.attachable(obj, T_obj→eef, arm)` screens the grasp set once per template
  (`iors_small` drops such grasps: `template_facts.grasps_dropped_not_attachable`). Owner of `r1pro_model` /
  `curobo_planner`: exempting `attached_object_<arm>_eef_link` against `<arm>_gripper_link` (as cuRobo's stock configs do)
  and a finer fit would make those grasps usable.
- `RobotKit.ignoring` re-syncs the whole world twice per use (0.5 s locally, up to 13 s on an L40S pod); `IKChecker.
  ignoring` switches the objects' meshes off by name instead (§3.6 "support ignored for the grasp pose only").
- Every `q_at_base_pose` configuration is relative to the robot's articulation root at call time: a stance's 28-D vector
  must not be cached across a teleport (`SamplerV2.stance_q` recomputes).

## Smoke (2026-09-15, pool pod L40S, run `194648-593f2d`)

`sample_isg --alias inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table --templates 1
--instances-per-template 1` → template + instance 0000 in 191 s (4 attempts: `place_unreachable`, `stance_collision_with_objects`,
`grasp_unreachable` once each), full checks (`checks.attached_object = true`, program check `expand` over `transition, grasp, lift,
place, retreat`); `multiply_isg --instances 1` continued the template (same `template.json` sha, band cache hit) and wrote 0001 at
its first attempt (80 s); D5's `validate_instances` reloaded both in a fresh process: `n_ok 2, n_fail 0, reload_failed 0`. Sidecar
`tro_sha256` verified against the TRO. The earlier runs of the day (0 of 30–40 attempts) fell to the backend facts below and to the
partner gap rule: cutting the §4.6 step-2 gap range per side to the manipulation trunk's reach band (the goal cell must be
reachable) took `place_unreachable` from 15–21 to ≤ 5 per 40 attempts.

### Multi-alias chunk smoke (2026-09-16, pool pod `htvfrubebp6tnj`, run `021545-064eb8`)

`sample_isg --alias …ontop_small_scale-apple-plate-breakfast_table --alias …nextto-marker-colored_pencil-breakfast_table --alias
mental_rotation-banana --templates 1 --instances-per-template 1 --max-attempts 6`: 3 / 3 aliases wrote their instance, wall 2090 s,
two world loads (1096 s: 562 + 534 s on a cold pod — dining room vs living room are different world keys), the second alias
**reused** the first world (134 s for the whole alias vs 961 / 984 s for the loading ones; template + instance 122 s); fleet
reach-band cache hits for apple and marker. Sidecars carry `idle_posture` / `arms_q` per stance and `ik_witnesses` per action.

### Simulator facts fixed in `sim_utils.py` (2026-09-16, from the disembedding root cause `9744b34`)

- **Parked objects fell**: `RigidPrimView.disable_gravities` writes the PhysX view while playing and USD only when stopped;
  every later `spawn()` does `og.sim.stop()/play()`, which rebuilds PhysX from USD → gravity back on for everything parked
  before, and there is no floor under the parking slots. `SimWorld.set_gravity(obj, enabled)` now writes the USD
  `PhysxRigidBodyAPI.disableGravity` attribute too (persistent across stop/play), `park()` / `activate()` use it and
  `spawn()` re-parks every earlier spawned object after `play()`. `navigation.attached` restores the object's *previous*
  gravity state (`gravity_disabled_in_usd`) instead of re-enabling it blindly.
- **`local_hull` mixed frames**: `collision_boundary_points_world` is transformed with the render-side (Fabric) link
  pose, refreshed only by a render or a teleport, while the root pose is PhysX; after physics-only steps a moving body's
  hull was shifted by the motion since the last render (+6.5 cm for a falling tray → `bottom_offset` wrong → tray inside
  the table slab). `local_hull` now calls `SimWorld.sync_render_transforms()` (`og.sim.sync_physx_to_fabric()`, a render
  frame as fallback) before reading. Task modules may drop their re-park workarounds once they run on this version.
- `--log-level` of the CLIs now wins over `ISPATIALGYM_LOG_LEVEL` (`sample_isg.configure_logging`: the `ispatialgym` logger and
  its stdout handler are re-levelled; third-party loggers stay at INFO).

### IORS small-scale goal rules (owner review 2026-09-16)

- **nextto / touching goal side** (`stances.nearer_side_lateral`, used by `iors_small.goal_pose`): the target goes on the
  partner side that faces its init position (never lifted over the partner); `summary.goal_side` = `left` / `right`
  (robot frame). Touching keeps §4.6 step 2 (partner on the robot's left → target on the partner's right, push toward
  +tangent, brace on the partner's left face); a target starting left of the partner is `Reject("touching_side")`. After the
  goal settle the nextto AABB gap (OmniGibson `NextTo` distance, PhysX poses + hulls) is recorded as `summary.nextto_gap_m`
  and must be `< g_NT`.
- **inside release height** (`iors_small.release_pose`): the recorded `place` is the release above the rim — the object is
  centred on the container and raised until the lowest point of the *open* gripper (`gripper_corners_world`: finger
  envelope `PAD_Z_RANGE[0]..FINGERTIP_LENGTH` along a, ±(`GRIPPER_OPEN_M`/2 + 1.2 cm) along c) clears `p_hull_w[:, 2].max()`
  by `release_clearance_m` (default 0.02); `drop_height_m` (default 5 mm) stays the floor of the object's bottom above the
  rim. `goal_poses` = the release pose (the goal check drops from there), the recorded goal = the settled pose, `pre_place`
  = 10 cm back along −a, both re-solved by the program check with the partner **not** ignored. The target footprint at the
  release must lie in the opening (outer extents − `opening_margin_m`) else `Reject("does_not_fit_inside")`. Facts in
  `grasps[].{place_kind: "drop", rim_z, drop_height_m, finger_clearance_m, place_lowest_finger_z}` and `summary.release`.
  The oracle side (`Place(kind="drop")`: open at the recorded pose, retreat straight up along −a) is the oracle agent's.

### Program check ignore sets, transition re-sweep, screening witnesses (large-scale A4/A5/A8, 2026-09-16)

- `program_check._ignore_for`: **brace** removes the supports (wall / stop) for **every** waypoint of the approach chain
  (as `push` does — the chain runs 2 cm along the wall); the braced object is removed at the contact pose only. A module
  may record exactly what it ignored on the contact / grasp record (`ignore_every` / `ignore_last`: bddl instance names
  or the token `"supports"`); the program check then uses those lists verbatim.
- `navigation.transition`: at each candidate s_T the trunk sweep **and the drive-in s_T → s** (strides at the arriving
  trunk) are checked with the *arriving* stance's arms (as `prompt0.connect`); `min_standoff` keeps a re-sweep from
  lowering the recorded s_T; `path_end_standoff` (where the recorded path ends — `core.path_end_standoff`) adds the extra
  drive-back from the path's end at the *departing* trunk and arms when s_T grows. `check_stances_with_objects` re-sweeps
  with both (the recorded path is never left ending beyond the transition's drive-back pose).
- `IKChecker.screen`: a remembered solution of the same pose (the sampler's exact solve) is verified as a witness before
  the 12-seed batch; misses are retried seeded from the nearest remembered solution and from the caller's `seeds`.
  `waypoints_ok` now solves the endpoints first and screens the intermediate waypoints seeded from them; the found
  screening solutions are remembered (deterministic re-check in the program check / `validate_instances`).

### Branch-consistent approaches (oracle finding, 2026-09-16)

The under-small french-fries sidecar had pre_grasp / grasp witnesses 3.16 rad apart (different IK branches): the 10 cm
straight approach was trackable in neither. `IKChecker.exact` now treats a straight same-orientation segment between
consecutive chain poses (`is_approach`: ≤ 0.25 m, same orientation within 0.05 rad — pre → grasp / place / contact, lifts,
the wall brace chain) as an **approach tracked from the previous solution** (`track_approach` → `backend.solve_ik_path`,
1 cm steps, ≤ 0.2 rad per step, ≤ 0.6 rad between consecutive §3.9 waypoints, intermediate configurations collision-free
against the world, the end one with the end pose's ignore set). When the forward tracking fails the end pose is solved on
its own and the segment tracked backward; the pre witness then moves to the end's branch. Neither → the chain fails with
`approach_untrackable` (`IKResult.code`), which `iors_small` (grasp at the settled pose, lift, place descent) and the
program check raise as their own reject reason. `waypoints_ok` hands straight chains to `exact` (one tracked motion with
`ignore_all` removed throughout). `IKChecker.ignoring` contexts nest (refcounted per mesh).

### Per-role spawn scale (owner item 4b, 2026-09-16)

`roles.<role>.scale` — `{"<cat>/<model>": s, "default": s}` (a bare scalar / `[sx, sy, sz]` = `default`) — resolves through
`problem.resolve_scale` (`<cat>/<model>` → model id → `default`) into `ModelChoice.scale` for every sampleable / group
role (D1's `roles.source.scale = 1.25` for the disembedding tray included); `spawn_role` passes it to `SimWorld.spawn`.
The template id carries the scale (`source-tray-gsxbym-s1.25`, `problem.scale_tag`; unscaled ids unchanged), so scaled
and unscaled draws never share a directory; band keys use the same `ModelChoice.scale` (`tray/gsxbym*1.25@…`).

### Two-arm check and the touching brace (2026-09-16)

`IKChecker.exact_both_arms` now ignores the objects the two final poses touch in its combined check (`ignore_for_last` ∪
`ignore_other_last`) and offers `brace_only_check=(ignored, margin)` — the same steps as `tabletop.brace_second_arm`,
which can call it again. `iors_small.touching_contacts` builds the brace like axial's stop brace
(`touching_brace_candidates`): pads on the partner's far face `STOP_CLEARANCE` (5 mm) outside it (closing vertical /
horizontal, either wrist turn), then the top-down blade beside the face with `tabletop.blade_standoff` over the face
height (pitches per `approach_pitch_order`); thin partners (< 4 cm) try the blade first. The record is a `tabletop.BraceRecord`
with `blade_half_thickness_m`, `stop_clearance_m`, `blade_standoff_m`, `pose_class` (`summary.brace` mirrors it).

### Floor height per task room (2026-09-16, large-articulation diagnosis)

`trav_map.floor_heights[0]` is OmniGibson's default 0.0 in every scene, but a room's slab may top out elsewhere
(`office_cubicles_right` / `shared_office_0`: `floors_ovpbsn_0` collision top at **z +0.0071**; the parked robot settles at
base z 0.012). Every check stood `base_link` at `0.0 + NOMINAL_BASE_Z`, so the wheel spheres sank 5 mm into the slab and
cuRobo's penetration query (no ground-like exemption, unlike the proximity margin) failed **every** configuration in the
room. `SimWorld.measure_floor_z(xy)` now measures the walkable surface: downward raycast at the park cell (robot bodies
ignored, hit within 0.3 m of the nominal height) → the room's `floors` collision-hull top → its AABB top → the nominal
value; cached in `floor_z_by_room`, logged with the delta, and `floor_probe` checks the settled base z − `NOMINAL_BASE_Z`
against it (2 mm; written to the template meta under `floor`). `SimWorld.floor_z`, `RobotKit.floor_z` (`q_config` /
`q_at_base_pose`), `teleport_robot`, `park_robot`, `prompt0.base_z` and the band script follow; `SamplerV2.park_robot_v2`
re-measures at the task room's park cell.

### Owner decisions 2026-09-16 10:15 (A–D)

- **A. Approached object along the approach**: the grasped target / placed-on partner / touched object is removed from
  the world along the **whole** tracked segment (`exact(..., approached=[...])`, `waypoints_ok(..., approached=...)`; the
  program check passes it per action; without it `IKChecker.approached_objects(end_pose)` infers the spawned objects whose
  AABB grown by 3 cm holds the pad centre / fingertip). Supports stay last-waypoint (3.3 cm tail), the pre pose keeps the
  full world.
- **B. Pre-poses in the frame check**: pre-grasp / pre-place / pre-contact and the contact poses go through
  `eef_in_frame` (`iors_small.require_in_frame`); `camera_pose_at` / `IKChecker.camera_pose` use the shared
  `conventions.camera_pose` FK (base on the measured floor, the kit's measured sensor offset) — the oracle's FOV monitor
  model.
- **C. Release above the actual surface**: `SimWorld.surface_z_under(xy, from_z, ignore=[target])` (downward raycast, robot
  and ``ignore`` bodies excluded → `{z, body, normal}`); `iors_small.goal_pose` (ontop / under) releases
  `release_above_surface_m` = 1 cm above it (saucer well, plate centre) instead of `goal_drop` above the hull top, once the
  partner stands at its pose; `release_surface_z` in the grasp record / summary. Disembedding destination, attachment
  place and block slots should adopt the same helper.
- **D. Up-axis guard**: `SamplerV2.check_goal_upright` after the goal settle — every goal object's up axis within
  `goal_tip_max_deg` (15°) of its goal pose's, else `Reject("goal_tipped")`; `summary.goal_tilt_deg`, counter `goal_tipped`.

### FOV rule (b): carry waypoint on demand (owner 2026-09-16 17:10 item 5, spec §3.9)

The gripper-in-FOV monitor is continuous from the pre-grasp to the release, so the carry leg lift end → pre-place must
stay in frame; the oracle drives it as a straight Cartesian leg at constant orientation (both ends recorded in frame ⇒
in frame). `SamplerV2.verify_carry_legs` (after `after_goal_check`) screens that leg with the object attached
(`waypoints_ok`, 5 waypoints, supports removed, partner and scene in the world); when it fails, one carry waypoint —
the midpoint raised 5–20 cm / shifted ±5–10 cm laterally, inside the manipulation image, both sub-legs trackable — is
recorded as **`grasps[k].carry: [pose]`** (`waypoint_policy` notes it, `summary.carry` per object), else
`Reject("carry_infeasible")`. Skipped when the place is made from a `placement` stance (large-scale: base move) or the
orientation changes along the carry (held-object rotation).

### Open-jaw footprint at the release (owner 2026-09-16 20:29)

nextto re-run 0916-181440: the opening finger swept the pencil 9 cm. `iors_small.open_jaw_clearance(place_eef, pts)` =
signed clearance of the partner's hull from the box the opened fingers sweep from the release up through the retreat
(eef frame: a from `PAD_Z_RANGE[0] − PRE_DIST` to `FINGERTIP_LENGTH`, c ±(`GRIPPER_OPEN_M`/2 + finger thickness + 1 cm),
x ±(finger half width + 1 cm)). nextto (`avoid_release_sweep` in `goal_pose`): a partner inside moves the place along
the nextto bound (gap up to g_NT − 3 mm), then — when the target's orientation is free — the place yaw turns the jaw
away (±90°, 180°); `release_opening_max_m` = the largest symmetric opening that still clears the partner by 1 cm (for the
oracle's pre-shaped release); when even pinch + 1 cm would sweep it → `Reject("release_sweep")`. Facts (`grasps[k].release_sweep`,
`summary.release_sweep`) for nextto and touching (touching: at the nextto place the push starts from, facts only).

### Robust `under` / `ontop` (D-007 option A, 2026-09-16 21:17)

`ispatialgym/sampling/predicates.py` — `evaluate_predicate(predicate_cls, *entities)`, the drop-in both `TaskSampler._evaluate`
(goal check) and `ispatialgym/tasks/ispatialgym_task.py` (`run_isg` / policy evaluation) call instead of OmniGibson's
`evaluate_bddl_predicate`. Branches for `under` / `ontop`: OmniGibson → probe rays across the contact patch (footprint centre +
ring at half the footprint radius, 3 mm below the supporting surface, up) → geometry (bottom gap ≤ 8 mm, centre inside the
support, up axis ≤ 15°). `reset_branches()` / `branches()` / `decided_by()`; the sampler writes `goal_check.binary_branch`
(+ the last log entries), the task exposes `predicate_branches()` / `reset_predicate_branches()` for `meta.binary_branch`.

## Status (D2 / D4, 2026-09-15) and gaps

Implemented: §2.3 customs reading + write-back (`problem.py`, reads D1's `isg_customs.json`: `roles.<role>.{grasp_keys,
orientation_equivalences, scale, joint}`, per-scene whitelists / blacklists, `generation`, `constraints`), §9 model draw and
`sample_isg` / `multiply_isg` (`iter_template_choices`, whitelist pruning after a failed template, `SAMPLING_REPORT`, `summary.json`),
§1 artefact layout, §3.2 placement helpers (v1 `SupportTop`, 2 cm margin, rest quaternion, 3 mm drop), §3.3 untucked posture /
trunk grid / **parking rule** (`navigation.park_pose`), §3.4 frustum test (5 %), §3.5 pipeline (settle, BEHAVIOR validation, goal
feasibility with the robot parked, **§4.6 step 5** whole-body check of every recorded stance with the objects placed + transitions
re-swept, **program check** through `oracle.programs.expand` (F6) with stance / carried-object tracking; `ISG_PROGRAM_CHECK=stub`
keeps the structural check only), predicate-cache handling (v1), report layout (rejections per reason with the first three examples),
§3.6 IK conventions (`ik.py`, forwarding to `robot.conventions` F5), §3.7 paths (`navigation.plan_path`), §3.8 transitions
(`navigation.transition`, both endpoints swept), §3.9 waypoint policy + eef-in-frame, §3.10 exact-IK and feasible-set caches keyed
with the spawn scale, §3.11 / §7 sidecar draft (F6-compatible dicts), §3.13 `orientation_equivalences` into `pose_targets`
(`customs.equivalence_group`), §6.11 rules in `tasks/iors_small.py` (pair admissibility tiers, gap rule, goal per relation, place IK
with the object attached, inside protrusion test, touching push + brace, `grasp_set` / `placement` / `band_ref` / `stance` records).

Not implemented yet (owner / item): §3.5 **reload validation** in a fresh process and the contact sheet (D5 `validate_instances.py` —
`core` writes everything it needs; the v1 validator still loads the artefacts because `template_meta.problem` stays
`<activity>-<def_id>` and the BDDL is also written inside the template directory); §3.12 the look-alike index reads
`constraints_conflicting.csv` + `representatives.csv` (`exact_group` / `geometry_group`) — `candidates.csv` has no `duplicate_of` /
`geometry_group` columns yet (A6), so texture variants outside the representatives sheet are not recognised; §6.11 inside: the drop
test is the goal-feasibility settle + the protrusion rule, the fillable-volume centroid comes from the container meta link (no
opening-shape test beyond the footprint); §3.8 the sweep samples both endpoints (6 configurations with F5's `trunk_sweep(endpoints=
True)` semantics; spec says "5 samples between"); the carry height of IORS is `max(0.12, partner height + 0.03)` (no rule in §6.11
— flagged for the spec); camera-relocalization path buckets are plumbed (`generation.path_buckets`, `--path-buckets`,
`sidecar.path_bucket_m`) and filled by the camera-relocalization module, not by core.
