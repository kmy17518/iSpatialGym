# Representative run of 2026‑09‑15 (G0: instances + demos for the representative pairs)

Owner request (07:44 PDT): fill the missing candidates with quick geometric filters, select ~30 representative object
models per task family (≥ 5 per subtask), then run everything end to end with **only** those representative pairs and
report what survives. This directory is the index of that run; the heavy artefacts (TROs, `demo.hdf5`, `head_rgb.mp4`)
stay under the git‑ignored `runlogs/runpod/samp/<pod>/<job id>/` pulled from the fleet.

Files: `summary.md` (per‑subtask table + every pick without an instance, with the rejection reason), `instances.csv`
(one row per written instance, path, demo status), `jobs.csv` (one row per sampling job), `demos.csv` (one row per
`run_isg` run). Regenerate with `python scripts/isg_reps/assemble_results.py`.

## What was built today (all on branch `survey`)

| step | artefact | commit |
|---|---|---|
| candidates filled by geometry (A5‑lite) | `build_icl.py --scene-inventory`: 64 admitted supports × 13 tabletop subtasks, 35 / 16 large partners, 84 (object, joint) articulated scene targets; `status = geometric_ok / geometric_reject:<reason>` in `icl/roles/candidates.csv` | `c0bc29e` |
| representative selection (A9) | `scripts/icl/select_representatives.py` → `icl/roles/representatives.csv` (199 picks; slots representative / variation_texture / variation_geometry / **global held‑out**), `representatives_report.md` (unavailable axes per subtask, slot availability, validation plan), `candidates_extra.csv` (635 filled large‑scale IORS target rows, `incomplete:*`) | `8e3dbac` … `a495dd5` |
| `ground_isg` v1 bridge (D0) | `ispatialgym/sampling/ground_isg.py`: one v1 spec per grounded alias (`ispatialgym/assets/task_specs/isg_reps/specs/`, 200 problems, all 21 subtasks) + `jobs.json` | `2c7b2a0` |
| v1 sampler gaps closed | obstructed camera relocalization (obstacle on the eroded shortest path, `blocked_footprints`, re‑planned path, obstacle in view) `8c3f56d`; IORS under(small) goal + `(under ?partner ?target)`, ontop(large) without the partner‑top init fallback, large‑scale articulation with scaled handles + B4 `handle_file` `5f6fe7d` | |
| `run_isg` lite (F0) | `ispatialgym/oracle/run_isg.py`: load instance → v1 oracle → evaluator → `demo.hdf5` + `head_rgb.mp4` + `meta.json`; obstacle footprints blocked in the planner map | `1818bf1`, `+` |
| fleet tooling | `scripts/runpod/dispatch_samp.py` (queue runner per pod, waves, rebalance, CMD jobs, rsync of the exported HEAD), `samp_summary.py`, `make_demo_jobs.py`, `scripts/isg_reps/assemble_results.py` | |

## Fleet

19 RunPod SECURE pods (7 × A40, 11 × L40S, 1 × RTX 4090; COMMUNITY had no stock), one sampler process per pod, TTL 4–5 h,
`collect.py --purpose samp`. A 20th pod (RTX 4090, host driver **610.57**, created with `--cuda` = any) segfaulted at Isaac
Sim start on every job (rc 139 after 18 s) and was terminated — the `allowedCudaVersions` filter of `rp.py create` is
there for a reason. Pods cannot pull the private repo after the bootstrap (token shredded): the dispatcher rsyncs the
committed HEAD instead. Spend ≈ $100 for the day (3 auto‑recharges of ~$24 today: 02:29, 15:51, 17:26 UTC).

## Results

See `summary.md`. Final (12:53 PDT, fleet terminated): 196 / 200 sampling jobs done (4 still running at the TTL),
**151 of 200 picks with instances, 302 instances**; 103 `run_isg` demos: object scaling 35 / 35 success, mental
rotation 37 / 37 success, camera relocalization 12 success / 19 QA‑failed (v1 base planner: `plan_failed` 17,
`collision` 2; the binary goal is reached in several of the failed runs — see `demos.csv`). Spend for the day $93
(4 auto‑recharges).

### Where the v1 rules lose picks (the reasons the v2 rules exist)

- **disembedding** 17 / 30 picks: `no_support` at `breakfast_table_rhjoby_0` (the v1 stance needs the source tray in
  view with the destination tray barely visible on the same edge — small targets: chili, mug, toy_dice, tablefork),
  `distractor_placement`, `occlusion`.
- **large‑scale IORS** ontop 0 / 5, under 1 / 5: `no_stance` at the init supports (desk_xhjsub_0, countertop_ikwqer_0,
  bar_udatjt_0) — the v1 stance search has fixed stand‑offs / laterals; v2 §5.6 (three stances + reach bands) is the fix.
- **large‑scale articulation** 1 / 12 (door_crtrpa_0 close): `unreachable` — v1's in‑scene stance / IK search with
  tucked‑arm seeds fails on kitchen cabinets, the fridge and the oven even where the robot‑only reach volume says the
  handle is reachable; one `Cannot add mesh, cache is full (2048)` (cuRobo mesh cache with the neighbouring rooms
  loaded, B10).
- **small‑scale articulation** 4 / 10: `no_support` (stance) and `artic_blender_cwkvib_0/j_lid holds no` stable angle.
- **attachment** 2 / 7: `partner_placement` (attach‑reach band), `male_cork_ncxgpe_0 has no male attachment link`
  (icl mark fixed: the representative must mate with the chosen female — `a495dd5`+).
- **touching** 3 / 4: `first-aid_kit.n.01` — a hyphenated synset breaks the BDDL instance parser (bug, not a rule).
- **camera relocalization demos**: the v1 oracle's base planner (cuRobo trajopt → traversability fallback) fails or
  brushes furniture; the sampler's validated path is not yet handed to the oracle (v2 sidecar `paths`, F1/F2).
- **held‑object demos**: `pose_settled ?agent` never holds because PhysX reports 0.1–0.2 rad/s on the wrist joints while
  the pads press the assisted‑grasp object (evaluator threshold 0.02 rad/s); binary success is unaffected.

### Not covered today

Variation / held‑out slots were pinned (`representatives.csv`) but only the `representative` slot was sampled
(`ground_isg --slots variation_geometry,heldout` grounds the rest). No `multiply_isg` (2 instances per template). The
21 v2 rules (reach bands, three stances, program check, sidecar) are untouched — this run used the v1 samplers through
the bridge, which is why the large‑scale subtasks lose most picks.
