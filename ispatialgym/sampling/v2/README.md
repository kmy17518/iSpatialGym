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
