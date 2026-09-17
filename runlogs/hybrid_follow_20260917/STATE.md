# Hybrid base-follow for large-scale articulation doors — state at the 2026-09-17 02:15 fleet shutdown

Owner decision D-011 C (2026-09-17 01:00): implement the hybrid base-follow (spec §5.3 A4 `room_door` rule, extended to every
door class behind `generation.follow_mode_override: "hybrid"`). Agent: hybrid-follow (this file written 02:30 PDT on the
owner's shutdown request; no probe launched after 02:09).

## What landed (all committed, working tree clean for these paths)

| sha | what |
|---|---|
| `7445ea71` | `ispatialgym/sampling/v2/tasks/articulation_hybrid.py` (new; the whole search + program-check verifier), the two hooks in `articulation_large.py`, the `program_check.py` hybrid branch + end-stance override, oracle pure reads (`sidecar.py` `FOLLOW_MODES += "hybrid"` + validation of `base_follow_path`; `actions.py` `BaseFollow.targets()` hybrid branch, `TARGET_KEYS += "follow_path"`, annotate), 6 unit tests in `tests/test_v2_large_scale.py`, `tests/test_sidecar_programs._check_value` allows `follow_path`. Full simulator-free suite: 1171 passed. |
| `c7b595ab` | docs: spec §5.3 A4 paragraph rewritten for the implemented hybrid; oracle doc `BaseFollow` row (hybrid targets, pending `Motions.base_follow_hybrid`). |
| `35940ba8` | `articulation_hybrid.py`: margin-aware IK (`_ik`: every successful seed, the closest-to-seed solution with joint-limit margin ≥ `min_margin_rad` 0.15, else the best-margin one ≥ 0.05) and the **retreat pose required at the final pose** (`check_retreat`: `pre_pose(h(q_last), 0.10)` must be reachable from the final base). Motivation: probe 0917-014213 below. |

Not landed on purpose (owner rule: executor changes only if a pure read): `oracle/execute.py` — see "Next step".

Probe inputs: `runlogs/smoke_inputs/hybrid_follow/customs_hybrid.json` = TEMP copy of `assets/task_specs/isg_customs.json` with
`generation.follow_mode_override: "hybrid"` on all 14 `configuration_matching-articulation_{open,close}_large_scale-*` aliases
(used via `export ISG_CUSTOMS=...` in the remote command). Launcher used: `/tmp/launch_hybrid_probe.sh <alias> <seed> <max_attempts>`
(not in the repo; it ran `remote_run.py --timeout 5400 --pull '$SMOKE_OUT' runlogs/hybrid_follow_20260917/<unit>/<run id>` with
`sample_isg --templates 1 --instances-per-template 1 --no-write-back --out $SMOKE_OUT/inst; then run_isg --mode oracle --video` on
the first written TRO). `launches.txt` lists run id → alias → local dir.

## How the hybrid follow works (module `articulation_hybrid.py`)

Single call site: `ArticulationLarge._solve_from` → `AH.search_follow_path(self, manip, q_star, h0, q_init, q_goal, fractions,
required, fraction_key)` when `self.mode == "hybrid"` (same `except Reject` fallback to arm mode as `_base_follow` when the arm
alone reaches every required waypoint). `self.mode` = A4 class rule (`room_door` → hybrid) or the override; the hinge-side stance
rule (Λ toward the hinge, the other arm) now applies to `mode == "arm"` only — hybrid keeps the per-arm default Λ, so
`hinge_side` is recorded `None` in hybrid runs. The `room_door_hybrid_todo` Reject is gone (the `door` units run hybrid).

Search (`FollowSearch.run`): dense joint grid `dense_values(q_init, q_last, marks, step)` — every §5.3 fraction interval up to the
last **required** waypoint split into equal sub-steps ≤ 10° (`step_revolute_deg`) / 5 cm; per q_i: `pose_joint(q_i)` (leaf at
q_i in the sim + cuRobo world), `h(q_i)` via the sampler (`pinch_sign` / `bar_offset` respected); (1) arm-only IK from the current
base b_{i−1} (skipped when the leaf footprint enters the base disc or the handle is beyond the torso-top reach bound); (2) else
candidates `lattice_candidates(b_prev, b_rigid, ...)`: interpolations b_{i−1} → b_rigid(q_i) = h(q_i) ∘ T_eef→base(q*) at α ∈
{0.25, 0.5, 0.75, 1} plus a 0.10 m lattice within 0.50 m × yaws between the current yaw and the rigid yaw (15° steps, ±15° extra),
sorted by displacement cost |Δxy| + 0.30·|Δyaw|; each candidate: `trav_free` (un-eroded floor), base disc (0.30) clear of the
fixed body's planar AABBs in the 0.05–1.20 m height band (`body_boxes`), base disc (0.33) clear of the leaf footprint
(`LeafFootprint`: the closed leaf's AABB corners rotated about the hinge / translated along the axis; falls back to the world AABB
when the analytic centroid leaves it), reach pre-filter, then `solve()`: `_ik` (32-seed cuRobo IK, all solutions, margin-aware
pick), `polish_witness` (≤ 5 mm), whole-body `collision_free` at `COLLISION_MARGIN` with the **leaf links removed but the body
present** (`_LinkSet` proxy through `IKChecker.ignoring`; base mode removed the whole object), at the final pose also the retreat
IK; then `path_ok`: 0.1 m strides b_{i−1} → b_i with the arm joints interpolated, whole-body collision-free, leaf (both joint
values) clear of every stride. Cap 40 IK calls per step, 600 s per search. Failure → `Reject("hybrid_follow_infeasible", ...
followed <fraction> of the swing ...)`; `template_meta.hybrid_follow` (per-step log, stats) and `hybrid_follow_fraction[stance]`
+ `hybrid_follow_fraction_max` keep the reached fraction.

Program check (`check_program_follow`, called from `program_check.check_action_targets` when `targets["mode"] == "hybrid"`):
every recorded pose re-verified as a witness (joint at q_i, `q_locked(stance at b_i, arm_q)`, polish → FK residual ≤ 2 mm,
collision-free with leaf removed / body present, floor, leaf footprint, margin ≥ 0.025, the base path between consecutive poses,
`base_follow[k]` == the path pose at q_k); witnesses `handle_poses[i]` + `follow_end`; `_Tracker.stance_obj` =
`end_stance_from_witness(follow_end)` so the **retreat is checked from where the base ends** (`check_action_targets(stance_obj=)`),
cleared by the next navigate / transition.

## Record shape (sidecar `articulation` block; mirrors the drawer's `follow_mode base`)

```json
"articulation": {
  "...": "object / joint / joint_type / q_init / q_goal / waypoints{k} / handle_reach{k} / passive_from as before",
  "follow_mode": "hybrid",
  "base_follow": {"0.25": [x, y, yaw], "0.5": [x, y, yaw], "0.75": [x, y, yaw], "1.0": [x, y, yaw]},
  "base_follow_path": {
    "mode": "hybrid", "arm": "left", "trunk": [translate, tilt],
    "step": {"revolute_rad": 0.1745} | {"prismatic_m": 0.05},
    "arm_q_order": "arm joints 1..7 (r1pro_model.ARM_JOINTS[arm])",
    "idle_arm": "right", "idle_arm_q": [7 floats], "fingers": 0.05,
    "poses": [
      {"k": 0.0, "q": q_init, "base_pose2d": [x, y, yaw], "arm_q": [7 floats], "eef": {"position": [3], "orientation": [4 xyzw]},
       "fk_residual_m": ..., "joint_limit_margin_rad": ..., "base_step_m": 0.0, "base_step_rad": 0.0, "chosen": "stance", "n_ik": 0},
      {"k": 0.083, "q": ..., "base_pose2d": [...], "arm_q": [...], "eef": {...}, "base_step_m": 0.1, "base_step_rad": 0.262,
       "chosen": "arm_only | alpha=0.5 | lattice", "n_ik": 12},
      "... one entry per dense step, k non-decreasing, the last = the last required waypoint ..."
    ],
    "stats": {"n_poses", "n_base_moves", "base_path_m", "base_turn_rad", "max_step_m", "max_step_rad", "n_ik", "followed_fraction",
              "arm_only_fraction", "total_displacement_m", "time_s", "r_max_m", "ignored": "leaf links", "leaf_links": [...],
              "n_body_boxes", "leaf_footprint": "analytic"},
    "knobs": {"... generation.hybrid_follow.* as used ..."}
  }
}
```
`base_follow[k]` = only the fractions the oracle drives through (k ≤ passive_from), exactly the path's pose at q_k (the program
check enforces it). `summary.hybrid_follow` = the `stats` dict; `summary.follow_mode` = "hybrid". The oracle side reads it:
`Sidecar.validate()` accepts `follow_mode: hybrid` (requires `base_follow` per followed fraction + `base_follow_path.poses[]`
≥ 2 with k from 0, `base_pose2d[3]`, `arm_q[7]`); `BaseFollow.targets()` returns `base_poses` = the **recorded** `base_follow[k]`
(base mode recomputes them from the rigid transform — wrong for hybrid) and `follow_path` = `base_follow_path.poses`.

Knobs (`generation.hybrid_follow`): `step_revolute_deg` 10, `step_prismatic_m` 0.05, `alphas` [0.25, 0.5, 0.75, 1.0],
`lattice_step_m` 0.10, `lattice_radius_m` 0.50, `yaw_step_deg` 15, `yaw_extra_deg` 15, `yaw_weight_m_per_rad` 0.30,
`max_ik_per_step` 40, `min_margin_rad` 0.15, `check_retreat` true, `reach_fraction` 1.0, `reach_extra_m` 0.10, `min_reach_m` 0.15,
`base_radius_m` 0.33, `body_radius_m` 0.30, `body_height_m` 1.20, `budget_s` 600, `record_eef` true
(`generation.hybrid_follow_record_eef` false drops the per-pose eef from the record).

## Probes (all with `customs_hybrid.json`; local dirs under `runlogs/hybrid_follow_20260917/<unit>/<run id>/`, `remote.log` = the
streamed pod stdout; the pod-side `/root/smoke/<remote run id>` dirs are collected by the coordinator to
`runlogs/runpod/collected_20260917/<pod>/smoke/`)

| unit (alias suffix) | pod / remote run id | local run id | code | state at 02:28 | follow fraction |
|---|---|---|---|---|---|
| open top_cabinet (house_double_floor_lower tynnnw_1, j_rightdoor, arm left) | `h4enpm19l0t3ng` / `014216-867a08` | `0917-014213` | `7445ea71` tree (no margin / retreat fix) | attempts 0, 1, 2, 3 each **followed the whole swing** (k 1.00 = q_goal 1.824 rad ≈ 104.5°, 12 dense steps, 7–8 base moves of 0.10–0.20 m / 15° yaw, ≈ 0.9 m base path) and **passed the follow's program check**; each was rejected at the **retreat** from the path's end: `ik_joint_limit` left_arm_joint7 within 14 / 18 / 7 mrad, attempt 3 left_arm_joint6 within 46 mrad (`stance manipulation -> follow_end [7.06, -0.01, -1.571]`). Attempt 4 in the search (k 0.33) when this was written. No instance. | **1.00** (search); instance blocked by the retreat |
| open top_cabinet, seed 1 | `pjf6451ahyqxll` / `020829-0dd28f` | `0917-020825` | `35940ba8` (margin-aware IK + retreat at the last pose) | attempt 0 in the search, k 0.50 of q_goal 1.920 rad at 02:28 (11 Phase-2 survivors, class 1). The run that would tell whether the retreat fix yields an instance. | ≥ 0.50 so far |
| open dishwasher (house_single_floor ynrhuk_0, j_link_2 horizontal axis, arm right) | `1c88jsy46ubecf` / `014218-5d7d3d` | `0917-014215` | `7445ea71` tree | attempts 0, 1 rejected `handle_unreachable` after ≈ 690 s each (10 Phase-2 survivors × 2 pinch signs: every hybrid search ends `hybrid_follow_infeasible`); attempt 2+ running. Per search the follow reaches k 0.25–0.42 (q 0.39–0.65 rad of 1.571): the arm follows to k 0.25–0.33 alone, one base move, then no candidate — the handle of a down-swinging door drops below the arm's reach at the fixed trunk (rejections `ik`, some `leaf` = the leaf's footprint over the base). No instance. | **0.33–0.42** (arm-only today: 0.11–0.33) |
| open fridge (restaurant_diner jtqazu_1, j_link_0, arm left) | `5gm563m9wgnrwc` / `014219-63db2f` | `0917-014217` | `7445ea71` tree | **finished, 0 written**: `handle_unreachable` ×8 at step 5a (the [pre-grasp, grasp] exact IK of the left arm fails from all 4 Phase-2 survivors of the best class, both pinch signs) — upstream of the follow, the hybrid search never started. S = [1.48, 1.71, 1.93, 2.15, 2.97]. | — (grasp itself unreachable) |
| close top_cabinet | `5gm563m9wgnrwc` / `020849-e2e0a6` | `0917-020847` | `35940ba8` | **finished, 0 written**: `no_stance` ×7 (Phase 1: no init pose on the 7 approach rays sees the whole open target collision-free) + `no_manipulation_stance` ×1 — the known close-variant Phase-1 problem, upstream of the follow. | — |
| open door (house_single_floor door_crtrpa_0, j_leaf, room sauna_0; the A4 `room_door` class rule → hybrid without override) | `6pdrunfn78iofy` / `020851-f1e671` | `0917-020849` | `35940ba8` | scene loading at 02:28 (no sampler output yet). | — |

Demo verdicts: none — no instance was written, so `run_isg` never ran. (Had one been written, the executor would have taken the
`arm_along(handle_poses)` branch for mode "hybrid" and failed at the arm's reach limit — see "Next step".) No `first_demo.py fix`
entry was logged (the instruction was to log when an instance is written; the code changes leave arm-mode instances unaffected —
hybrid runs only under the override or for `room_door`, which was a template Reject before).

Timing seen: scene load 220–450 s; template (stable set + passive probes) ≈ 2–5 min; one hybrid search over 12 dense steps
≈ 2–4 min (150–250 IK calls, `pose_joint` world re-sync per step); the follow's program check ≈ 1 min.

## Findings

1. **The hybrid follow works for a vertical-axis cabinet door**: the top cabinet's full 104° swing is followed with ≈ 8 small base
   moves (the arm-only follow reached 9–18 %). The base moves chosen are the min-displacement lattice steps (0.10–0.14 m or a 15° yaw),
   not the α-interpolations toward the rigid sweep — expected from the cost ordering.
2. **The retreat from the path's end is the blocker** (0917-014213, 4/4 attempts): at ≈ 104° the wrist (joint 7, once joint 6) sits
   within 7–46 mrad of its limit for every IK solution of `pre_pose(h(q_last))`, so the retreat check (`away_from_limits`, 0.05 rad
   margin) rejects. `35940ba8` requires the retreat IK at the final pose inside the search (the base moves at the last step until it
   works) and prefers solutions with ≥ 0.15 rad margin along the whole path — unverified at shutdown (run 0917-020825 was mid-search).
   If it still fails: try the other pinch sign first at the last steps, or let the final pose be chosen with a larger `min_margin_rad`,
   or end the follow one dense step earlier when `passive_from` allows (the joint continues on its own).
3. **Horizontal-axis doors (dishwasher, oven) need a trunk DoF**: the handle drops as the door swings down; the base cannot help
   with height and the search stalls at 33–42 %. Extension: add `translate` (and maybe `tilt`) to the candidate set (`kit.with_trunk`
   in `solve()`, the trunk recorded per pose, the executor ramping the trunk with the base and arm) — not started.
4. Fridge (restaurant_diner) and close top_cabinet fail upstream of the follow (grasp IK / Phase 1) — the same as in the arm-mode
   runs of 09-16; not hybrid problems.

## Next step (in order)

1. **Oracle executor** (primary oracle agent 7deabbed; described in the `7445ea71` commit message): `execute_base_follow` must
   dispatch `targets["mode"] == "hybrid"` to a new `Motions.base_follow_hybrid(targets["follow_path"], action.arm)`: per consecutive
   pair of `poses`, drive the base `b_{i−1} → b_i` (the `PathTracker` / `_base_reference` at `base_follow_speed`) while ramping
   `state.arm_target[arm]` from the current arm joints to `poses[i]["arm_q"]` over the **same progress along the segment** (interpolate
   by the tracker's progress, not by time; consecutive poses are ≤ 10° of joint motion apart so the eef stays within a few mm of the
   handle arc), then `settle`; the first pose is the manipulation stance at the grasp (start from the current joints), the trunk
   stays; the `Retreat` after it runs from the current state as today. Until then a hybrid instance's demo fails `plan_failed` in
   `arm_along` at the arm's reach limit.
2. Re-run `open_large_scale-top_cabinet` with `customs_hybrid.json` on `35940ba8`+ (seed 1 was mid-search; seeds 0/1, 8 attempts,
   ≈ 45 min per run incl. the 7 min scene load) → the first hybrid instance → `run_isg` demo once (1) has landed → fix-log
   `python scripts/isg/first_demo.py fix --commit <sha> --subtasks configuration_matching-articulation_open_large_scale
   configuration_matching-articulation_close_large_scale --what "..." --owner hybrid-follow`.
3. Read the collected `/root/smoke/{014216-867a08, 020829-0dd28f, 014218-5d7d3d, 020851-f1e671}` dirs (`inst/.../summary.json`,
   `template_meta.hybrid_follow` in the template dir) for the per-step logs of the searches that were cut off.
4. Decide with the owner whether the hybrid becomes the default for the `cabinet` class (customs `follow_mode_override` per alias, or
   `follow_mode_by_class("cabinet") → "hybrid"` in `articulation_large.py`), and whether the trunk DoF (finding 3) is worth it for
   dishwasher / oven.

## What a fresh agent must know

- Owner decision D-011 C = this work; the primary large-scale agent (f0a45699) edits `articulation_large.py` for the chained-open
  variant — keep the two hybrid hunks there (`_sample_template` override + `_solve_from` call site) when merging; the primary oracle
  agent (7deabbed) owns `oracle/execute.py` / `motions.py` (the pending motion); `program_check.py` is core's (D2) — the hybrid branch
  and `stance_obj` are small and self-contained.
- The unit tests run without a simulator: `pytest ispatialgym/tests/test_v2_large_scale.py -k hybrid` (6 tests) and
  `ispatialgym/tests/test_sidecar_programs.py` (targets keys incl. `follow_path`).
- To probe: pods via `scripts/runpod/remote_run.py` (pool `runlogs/runpod/smoke/pods.txt`; the fleet is shut down as of 02:15 — do not
  launch until the owner reopens the pool); the customs copy with the override is `runlogs/smoke_inputs/hybrid_follow/customs_hybrid.json`
  (`export ISG_CUSTOMS=<that path>` before `python -m ispatialgym.sampling.sample_isg --alias <alias> ...`).
- Reading a run: `grep "hybrid follow" remote.log` gives one line per dense step (`chosen`, base step, IK calls, rejected candidates
  by reason `floor / body / leaf / reach / ik / path / ik_cap`); a search that fails leaves `hybrid_follow_infeasible` in the
  attempt's `handle_unreachable` summary (the last 4 errors) and the reached fraction in `template_meta.hybrid_follow`.
