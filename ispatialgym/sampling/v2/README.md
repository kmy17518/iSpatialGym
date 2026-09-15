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
