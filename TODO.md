# iSpatialGym sampling + demo rebuild — shared TODO

Single source of truth for every agent / person working on the rebuild (iddl → icl → `ground_isg` → `sample_isg` →
`multiply_isg` → `run_isg`). The full design was settled in the design thread of 2026‑09‑14 (see *Decisions register*
below and the docs listed under A3 once rewritten). Keep this file current; nothing else tracks who does what.

## How to use this file (protocol)

1. **Before starting an item**: re-read this file, set the item's `status` to `in_progress`, put your handle in `owner`
   (a stable handle such as `agent-main`, `agent-rv`, `minyeong`), and add a line to the *Activity log*.
2. **When done**: set `status` to `done`, fill `artefact` (path, commit, or runlog dir), add an Activity‑log line.
   Partial results: `blocked` + one line saying what blocks it.
3. **Edit surgically** (single-line replacements, never rewrite the file); several agents edit the same working tree.
   Commit `TODO.md` on its own (`git commit -m "todo: ..." TODO.md`) so the history of the list stays readable.
4. New items go into the phase they belong to with the next free id (`A5`, `B9`, ...). Do not renumber.
5. Dependencies (`needs`) are hard: do not start an item whose dependencies are not `done` unless the note says the
   overlap is safe.
6. Status vocabulary: `todo` | `in_progress` | `blocked` | `done` | `dropped`.

## Shared resources — rules

- **RunPod** (account key in `.env`, never print it): create pods only with `python scripts/runpod/rp.py create --gpu ...
  --name <purpose>-<detail> --purpose <purpose> --ttl-h <h>`; start `python -u scripts/runpod/collect.py --purpose
  <purpose> --remote-dir /root/<dir>` for every purpose so results are pulled before the TTL kills the pod; **one**
  `rp.py watchdog` runs on this machine (TTL / idle / cap enforcement; check `pgrep -af "rp.py watchdog"`, restart with
  `nohup python3 scripts/runpod/rp.py watchdog --interval 300 --idle-min 15 --idle-grace-min 30 >> runlogs/runpod/watchdog.log 2>&1 &`
  if it is gone). Spend cap **$200** (`rp.py --cap`), auto‑recharge is +$25→$50 at $25; every `RECHARGE` line in
  `runlogs/runpod/billing.log` must be reported to the owner. Terminate pods when their job is collected (collect.py
  does it) — pods that idle bill money. One sampler/survey process per pod (measured: more processes per pod are not
  faster; see *Pitfalls*).
- **`.env`**: `RUNPOD_API_KEY`, `HF_TOKEN`, `GH_TOKEN`, `WANDB_API_KEY`; git‑ignored; pass secrets to pods only via
  `rp.py bootstrap` (stdin → root‑only file, shredded after the install).
- **Repo**: commit small, message prefixes `todo:`, `docs:`, `icl:`, `iddl:`, `sampling:`, `runpod:`, `oracle:`.
  `runlogs/runpod/` is git‑ignored (fleet state, billing, raw pulls); survey *results* meant for the pipeline go under
  `runlogs/<survey>_<date>/` and are committed like the earlier selection runlogs.
- **Caches** shared by the samplers: `~/.cache/ispatialgym/{sampling,reach_volume}` (`ISPATIALGYM_SAMPLING_CACHE`,
  `ISPATIALGYM_REACH_VOLUME`). Survey outputs that samplers read at runtime must be copied into the repo
  (`runlogs/...`) or the cache dir named in the item's artefact.

## Phases and items

Columns: id | item | needs | where | owner | status | artefact / notes

### A. Specification artefacts (CPU, local) — first; B/D depend on the names and fields

| id | item | needs | where | owner | status | artefact / notes |
|---|---|---|---|---|---|---|
| A1 | **iddl**: 21 `.iddl` files under `ispatialgym/assets/iddl/<family>/<subtask>.iddl` + `ispatialgym/assets/iddl/README.md` (grammar: roles, groups, implicit `agent`/`floor`, predicate argument order per subtask, e.g. `(under ?partner ?target)` for under(small‑scale)) | – | local | agent-main | done | `ispatialgym/assets/iddl/` (README = grammar + grounding rules; role kinds object/group/fixed/agent; `:options` grasps/push/articulated/attach/hinge/scale/scene/group_size/same_model). Parser is part of D1. |
| A2 | **icl** (iSpatialGym candidate list): convert `runlogs/grasp_library_20260905/select/task_object_selection.csv` → `ispatialgym/assets/icl/{roles,pairs,combos,grasps}.csv` with the new subtask ids / role names, categories + `display_name`, representative (`pick 1`) and test (`pick 1`+`pick 2`) marks, per‑scene naturalness, every old "viable" list downgraded to `candidate` (status `untested`); supports = union of the 31 scene objects that appear as `spec`/`viable` in any task + the 78 wall regions; large‑scale partners from `runlogs/table_selection_20260911/inventory.csv` (`under_partner`); per‑grasp `width` / `finger_q` from the grasp library results; `combos.csv` carries the explicit combinations to test (`test=1`) | – | local | | todo | script `scripts/icl/build_icl.py` (repeatable) |
| A3 | **Docs rewrite**: `task_bddl_formats.md` (aliases, `alignment` activity, under argument order, inside `orientation_metric: null`, obstacle objects in the obstructed BDDL, `targets_group`), `instance_sampling_spec.md` → new rule set (tabletop Phases 1a/1b/2/3 + margin table, Prompt 0/1/2, three stances, stance‑transition sweep, base‑follow, brace IKs, waypoint IKs, eef‑in‑frame, disembedding placement stance, navigation rule, `blocked_footprints`, path buckets), `oracle_primitives_design.md` (demo sequences incl. cube grasp rule, base‑primary articulation), `task_definition_and_evaluation.md` (customs‑list fields: constraints, per‑scene whitelists, `path_buckets`, oracle sidecar). Pilot‑dependent numbers marked `[pilot]` | A1 | local | agent-main | in_progress | |
| A4 | **BDDL plumbing**: rename activity dir `world_alignment` → `alignment`; `bddl_writer` for the new problems (ontop/under large‑scale, under small‑scale argument order, obstructed with obstacle objects); alias ↔ definition‑id registry `activity_definitions/<activity>/problems.json`; tests updated | A1 | local | | todo | |

### B. Survey scripts (write locally, smoke‑test on the local 4090) — independent of A, run in parallel

| id | item | needs | where | owner | status | artefact / notes |
|---|---|---|---|---|---|---|
| B1 | `ispatialgym/sampling/eef_reach_volume.py` (robot‑only eef reach volume, 6 orientation classes, sharded per (arm, trunk)) | – | local | agent-rv | done | commit `bba3413`; smoke test ok (local 4090: 258 feasible cells per trunk in 32 s) |
| B2 | **Full reach volume run**: 2 arms × 20 trunks, merge → `reach_volume.npz` + `meta.json`; copy into the repo/cache location the samplers read | B1 | pods (10 × 4090, purpose `rv`) | agent-rv | in_progress | started 06:42 UTC 09‑15, TTL 3 h, `collect.py --purpose rv` pulling to `runlogs/runpod/rv/`; final location TBD by owner (write it here) |
| B3 | `ispatialgym/sampling/reach_bands.py` — tabletop reach bands (lattice 8 cm, library‑grasp mask in rounds with early exit, generic pinch for blocks / push poses), two‑factor cache (support+trunk part vs object+grasp part), `--brace` mode for small‑scale articulation, `--support floor [--under <partner>]` for IORS large‑scale, output per trunk + union extent | – | local (smoke) → pods | | todo | |
| B4 | **Scene inventories**: (a) articulated scene objects per scene with instance scale, joints, handle annotations *scaled to the instance* (incl. the 13 host models without annotation: `door_ktydvs_0`, `oven_wuinhm_0`, `bottom_cabinet_no_top_gjrero_0/4`, `bottom_cabinet_fancyy_0`, `bottom_cabinet_jhymlr_0`, `bottom_cabinet_nddvba_0`, `bottom_cabinet_no_top_ncgmqr_0`, `bottom_cabinet_rvunhj_0`, `top_cabinet_tynnnw_0`, `door_vudhlc_0`, `door_mkcndz_0`, `openable_window_ithrgo_0`) + review renders; (b) `supports/<scene>__<name>.json` (top polygon, edges, free‑floor run per edge, overhang, chairs removed) for the 31 supports; (c) large partners (largest horizontal face raster, under‑clearance map) for beds/sofas/desks/breakfast tables/countertops | – | pods (per scene) | | todo | |
| B5 | **Wall region re‑survey** with the Prompt‑0 lattice (β ray, d/λ scan, both arms, brace) using the reach volume; per region: survivors of steps 1–3 for δ = ±30° with the reference targets | B2 | pods (per scene) | | todo | |
| B6 | `contacts/<category>__<model>.json` for alignment targets (E_low/E_high, brace kinds by shape class, no‑go regions) + review renders; per‑model scale hints (diagnostic pass s ∈ {0.9 … 0.5}); axial: pushable ends + stop edges | B2 | pods | | todo | |
| B7 | Held‑object visibility cache per (model, grasp) for object scaling / mental rotation (hold pose constant: depth 0.50, trunk (0.5, 0), roll 15°) | – | pods | | todo | |
| B8 | `Grasp.pad_clearance` (open width 0.10 − `meta.width`)/2 in `grasp_library.py`; expose `finger_q`, `width` in icl `grasps.csv` | – | local | | todo | trivial |
| B9 | Camera‑relocalization room inventory (`house_single_floor/living_room_0` only for now) and obstacle footprint check ("diverts the eroded shortest path") | – | local | | todo | |

### C. Survey run on pods

| id | item | needs | where | owner | status | artefact / notes |
|---|---|---|---|---|---|---|
| C1 | `scripts/runpod/fleet.py`: N single‑process pods from a job manifest (per‑scene queues), bootstrap, run, collect, terminate; bandwidth probe in `bootstrap.sh` (abort hosts < 20 MB/s); `allowedCudaVersions` guard for drivers 595/610 | – | local | | todo | pieces exist in `rp.py` / `collect.py`; manifest runner missing |
| C2 | Run B3–B7 across ~10 pods + local (≈ 2.5–3 h per pass) or ~25 pods for ≈ 1 h (bake the Docker image first if > 10 pods) | B3–B7, C1 | pods | | todo | pod count = owner's decision (same cost either way) |
| C3 | Agent‑review rounds on the renders of B4/B6 (scene handles, alignment contacts, odd support edges) → verdicts back into the icl | C2 | local (review agents) | | todo | review contracts like `runlogs/grasp_library_20260905/artic/handle_review_contract.md` |

### D. Sampler rewrite (local code; needs A done, consumes B/C outputs)

| id | item | needs | where | owner | status | artefact / notes |
|---|---|---|---|---|---|---|
| D1 | `ground_isg`: iddl + icl → BDDL (alias per category combination) + task customs list (per‑scene whitelists/blacklists, constraints records, `path_buckets`, isg fields); representative‑model combination test; template + instance 0; oracle sidecar `<iid>.oracle.json` | A1–A4, B2 | local + pods | | todo | |
| D2 | Sampler core: tabletop Phases 1a/1b/2/3 (margin table), Prompt 0/1/2, IORS three stances A/B/C with B↔C grasp coupling, stance‑transition sweep (retreat `s_T`), navigation rule (locally un‑eroded path + whole‑body checks / eroded global + local connections), intermediate waypoint IKs, eef‑in‑frame check, `blocked_footprints`, recording of every stance / contact / grasp into the sidecar | D1 | local | | todo | replaces `tasks/base.py` stance code; `REACH_ARMS` tucked and `view_arm_fallbacks` removed |
| D3 | Task samplers: disembedding (placement stance, distractor spacing `x`, ≤ 30 % occlusion), block design (cube grasp set: pinch ∥ flip axis, ±45° tilt pair, smallest rotation, init up‑face ≠ opposite of goal face, left brace points), articulation small (handle + brace arms, 3 yaws) / large (base‑follow per joint type, handle grasp both directions, scaled handles), attachment (+ partner brace, pre‑attach along insertion axis), alignment vertical/horizontal (tangential push + brace, arc waypoints), axial (+ left brace at stop edge, scale), IORS ×7 (under small argument order, inside orientation null, touching partner on robot's left + brace), camera relocalization standard / obstructed (path buckets), held objects (fixed hold, arm‑only goal IK, SLERP/ray waypoints) | D2 | local | | todo | |
| D4 | `sample_isg` (N templates per BDDL, unique models, whitelist pruning on failure, constraint recording) and `multiply_isg` (N instances, bucket counts) | D1–D3 | local | | todo | |
| D5 | `validate_instances.py` / `render_instances.py` for the new artefacts (sidecar, `alignment` activity, new problems) | D4 | local | | todo | |

### E. Pilot

| id | item | needs | where | owner | status | artefact / notes |
|---|---|---|---|---|---|---|
| E1 | `ground_isg` pilot: representative models for all 21 subtasks × scenes → whitelists, yield per rule; fix low‑yield rules | D1–D5, C2, C3 | pods | | todo | |
| E2 | Freeze the `[pilot]` numbers in the docs (x_c(t), camera pitch, band extents, brace reach, base‑follow corridors) | E1 | local | | todo | |

### F. `run_isg` (demo / replay / policy, record, eval)

| id | item | needs | where | owner | status | artefact / notes |
|---|---|---|---|---|---|---|
| F1 | Unified loader + evaluation (TRO + sidecar, scene variant patch, `ISpatialGymTask`), three modes (oracle / lerobot replay / policy) | D5 | local | | todo | |
| F2 | Oracles per task on the sidecar (navigate with the new rule, `set_torso` without arm fallback, recorded grasps, place, push/brace, base‑follow articulation, block yaw‑flip‑place), JoyLo‑like motion | F1 | local | | todo | |
| F3 | Real‑time QA hard failures (binary success, tolerance, motion, collision, ghost hand, prolonged pause, gripper in FOV, failed grasp except push/brace oracles); checkpoints / rollback per oracle action | F2 | local | | todo | |
| F4 | Recording: hdf5 + LeRobot v3 in one wrapper (`omnigibson/envs/lerobot_data_wrapper.py`), video‑only mode, three annotation levels (family, template categories, per‑instance numbers), goal‑pose override + goal image after `max(50, 20 %)` no‑op steps | F1 | local | | todo | |

### G. Production

| id | item | needs | where | owner | status | artefact / notes |
|---|---|---|---|---|---|---|
| G1 | `ground_isg → sample_isg → multiply_isg` on the fleet for all subtasks | E1, D4 | pods | | todo | |
| G2 | Demo collection with `run_isg`, QA statistics, LeRobot shards | F4, G1 | pods | | todo | |

## Decisions register (apply everywhere; details in the design thread and, after A3, in the docs)

- Names: `iddl` (files), `icl` (candidate list), stages `ground_isg → sample_isg → multiply_isg`, demo/eval script `run_isg`.
- Alias families: `camera_relocalization-{standard,obstructed}`, `object_scaling-{target}`, `mental_rotation-{target}`,
  `disembedding-{target}-{distractors}-{support}`, `configuration_matching-{block_design-block_4-{support} | articulation_{open,close}_{large,small}_scale-… | attachment-{target}-{partner}-{support}}`,
  `alignment-{world_vertical,world_horizontal,axial}-{target}-{support}`, `inter_object_relation_satisfaction-{ontop_large_scale,ontop_small_scale,under_large_scale,under_small_scale,inside,nextto,touching}-{target}-{partner}-{support}`.
  Activity dirs: `alignment` (renamed), `inter_object_relation_satisfaction` (kept). Block = custom `toy_dice/isgblk` (KB synset `toy_dice.n.01`).
- Roles: `target`, `targets` (list), `partner`, `source`, `destination`, `distractors` (+ `_same_category` / `_other_category` pools), `obstacles` (obstructed camera relocalization only), `support`; implicit `agent`, `floor`. Categories in the icl, synsets only in BDDL.
- Scope for now: camera relocalization in `house_single_floor/living_room_0`; one obstacle; fixed trays (`gsxbym` source, `hvlfig` destination); 4 blocks; inside has no orientation metric; large‑scale IORS partners from the CSV (ontop: countertop, desk, bed, sofa, breakfast_table; under: bed, sofa, desk, breakfast_table); partner may be the init surface for under(large‑scale); floor init is an explicit support choice, never a runtime fallback; NO FALLBACKS anywhere in initialization.
- Robot: untucked arms everywhere except camera relocalization / held objects (JoyLo reset pose; held: right arm from the fixed hold); 12‑seed screening IK, 32‑seed exact IK; trunk grid translate {0, .25, .5, .75, 1.0} × tilt {0, −.3, −.6, −.9} (+ −1.2 when h < 0.6 m).
- Margin table (Phase 1b), navigation rule, stance transitions, base‑follow per joint type, cube grasp rule, brace IKs, waypoint IKs, eef‑in‑frame, path buckets, oracle sidecar: as written in the design thread; A3 transcribes them.
- Every existing "viable" list (supports, walls, artic) is a *candidate* list until `ground_isg` re‑derives it.

## Pitfalls (measured)

- RunPod: hosts with driver 595/610 and the RTX PRO 6000 Blackwell segfault at Isaac Sim startup → `allowedCudaVersions` 12.8/12.9/13.0; `EUR‑IS‑2` pulls GitHub at 120 KB/s; vCPU allotment per pod varies 9–36 for the same GPU type (`RUNPOD_CPU_COUNT` in `/proc/1/environ`); SSH sessions do not inherit the container env.
- Throughput: the sampler is CPU‑bound (GPU 5–10 %, 7–8 GB per process); K=2 on a 4090 = K=1; K=4 on L40S/A6000 is slower than K=1 → **one process per pod**; host CPU decides (EPYC 9354: 208 s/job, EPYC 7642: 302 s, Xeon 6342: 384 s for the same job). Sampling report layout: `problems.<problem>.<template>.{n_written,time_s}`.
- Secrets: never on a command line (they end up in `ps` and tracebacks); `rp.py bootstrap` uses stdin.
- Sampling: `Under` = upward ray from the target's AABB‑centre column hits the partner; `NextTo` bound = mean(dims_a + dims_b)/6 (element‑wise sum) + horizontal adjacency rays.

## Activity log (append‑only, newest last)

- 2026‑09‑14 17:38 PDT agent-main: RunPod tooling created (`scripts/runpod/rp.py`, `bootstrap.sh`, `bench.sh`); benchmark pods 4090 / L40S / A6000 / A40 / PRO 6000 (PRO 6000 unusable). GH/HF tokens leaked into the transcript on the first launch; owner chose not to rotate.
- 2026‑09‑14 19:29 PDT: auto‑recharge +$24.76 (balance 24.99 → 49.75).
- 2026‑09‑14 21:19–22:59 PDT agent-main: second benchmark round (bench2/bench3), results in `runlogs/runpod/bench/`, conclusions in *Pitfalls*; `collect.py` added; `eef_reach_volume.py` written (B1) and committed (`bba3413`).
- 2026‑09‑14 23:42 PDT agent-rv: B2 started — 10 × RTX 4090 pods `rv-{left,right}-t{0,0.25,0.5,0.75,1.0}`, TTL 3 h, `collect.py --purpose rv` running.
- 2026‑09‑15 00:05 PDT agent-main: this file created; A1 claimed.
- 2026‑09‑15 00:20 PDT agent-main: A1 done (iddl README + 21 files, commit `iddl:`). Next: A3 (rule doc) — A2 (icl) and B3 (reach_bands) are free for other agents.
- 2026‑09‑15 00:21 PDT agent-main: A3 claimed (docs rewrite).
