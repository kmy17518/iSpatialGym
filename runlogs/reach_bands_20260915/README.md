# Tabletop reach bands (R1Pro, untucked arms) — 2026-09-15 (TODO B3)

Produced by `python -m ispatialgym.sampling.reach_bands` for the owner's representative picks
(`ispatialgym/assets/icl/roles/representatives.csv`, slot `representative`; the `variation_geometry` slot is the second
wave, `jobs_w2.json`). For one (scene support, grasped object) and every trunk of the grid (translate {0, .25, .5, .75,
.95} × tilt {0, −.3, −.6, −.9} (+ −1.2 when the top is below 0.6 m)) at one concrete stance (edge midpoint, stand-off
0.45 m + AABB overhang ≥ 5 cm, yaw = inward normal): whole-body collision check of the trunk, an 8 cm lattice over the
near strip of the top, the shoulder-distance prune, then the library-grasp mask (object at 8 yaws × every library grasp,
pre-grasp 10 cm back along the approach, 12-seed batched cuRobo IK, untucked idle arm locked, support ignored for the
grasp pose only) in rounds of 8 (yaw, grasp) pairs with early exit per cell. Spec: `instance_sampling_spec.md` §4.1,
§3.10, §4.5–4.6 (feasible sets), §5.4 (`--brace`), §5.6 (`--support floor [--under]`).

## Files

* `bands/<mode>__<scene>__<support>[-under-<partner>]__<category>-<model>__<key>.{npz,json}` — one band per job:
  the `.json` carries the factors (scene, support, object, grasp key, spawn scale, placed objects, partner), the stance
  (`stance.base_pose2d`, edge, overhang), the top raster summary, per-trunk statistics (`trunks[]`: collides, cells,
  rounds, pairs, time) and the union extent; the `.npz` holds per trunk `t<translate>_k<tilt>` = `(N, 2)` cells in base
  coordinates (x ahead, lateral + = robot's left) and `t<translate>_k<tilt>__pairs` = `(M, 3)` int `[cell index,
  yaw index into meta.yaws, grasp index into meta.grasp_names]` = the (yaw, grasp) pairs found feasible for the cell
  in the round that marked it (seed of the feasible set F of spec §4.6; `yaw index = -1` + candidate index for the
  generic pinch). Brace bands keep `trunks[].feasible = [{cell, yaw_offset, q_indices, q_feasible, brace_kinds}]` in
  the `.json` (the set of joint values whose handle poses are feasible per cell, spec §5.4).
* `jobs.json` / `jobs_w2.json` — the manifests (`scripts/reach_bands/make_jobs.py`; `cmd` = the pod command, `args` =
  the local call); `jobs_notes.txt` lists the picks that got no job and why.
* `results.json` / `results.csv` — per-job outcome (pod, wall time, trunks, cells, union extent, error); the table
  below is generated from them by `scripts/reach_bands/collect_bands.py`.
* `smoke/` — the four local smoke runs (`pack_of_kielbasa/dgodnn` on `breakfast_table_rhjoby_0`, see below).

## Reading a band (the sampler)

```python
from ispatialgym.sampling.reach_bands import ReachBands
rb = ReachBands.load("house_single_floor", "breakfast_table_rhjoby_0", "pack_of_kielbasa/dgodnn", "pack_of_kielbasa__dgodnn__t")
rb.band(0.5, 0.0)                       # (N, 2) cells of the grid trunk with the same translate and the nearest tilt
rb.union_extent()                       # (lambda_min, lambda_max) over every trunk (Phase 1c margins)
rb.feasible(0.5, 0.0, cell)             # [(yaw_offset, grasp_name), ...] recorded for that cell; yaw_world = base_yaw + yaw_offset
rb.to_world(cells, base_pose2d)         # base-frame cells -> world xy at a concrete stance (Phase 2: band shifted with the stand-off)
rb.meta["stance"], rb.meta["trunks"]    # the stance it was computed at, per-trunk statistics
ReachBands.load(scene, "floor", obj, key, under="bed_rrcvaq_0")          # IORS under band (floor under the partner)
ReachBands.load(scene, support, target="toolbox/redmtl", joint="j_lid")  # brace band; rb.brace_feasible(t, k)
ReachBands.load(scene, support, "toy_dice/isgblk", "generic")            # generic-pinch band (blocks, axial targets)
ReachBands.load(scene, support, obj, key, placed=["tray/gsxbym"])        # disembedding (lattice = source tray floor)
```

`ReachBands.load` searches `ISPATIALGYM_REACH_BANDS` (if set), then this directory's `bands/`, then
`~/.cache/ispatialgym/reach_bands`, matching the meta fields (never the hashed file name); a miss raises
`FileNotFoundError` — `sample_isg` then runs the script (spec §4.1). `ReachBands.from_files(path_npz)` reads one file.

## 2026-09-16 additions (owner decisions A1 / A2 / A6, commit `54097c0`)

**Left-arm generic CONTACT bands** (`--arm left --generic-contact --height H`, files `contact__<scene>__<support>__left_h<class>__<key>`):
one band per (support, object height class) — heights binned to 2 cm (`height_class(h) = round(h / 0.02) * 0.02`), the
21 brace-task objects of the representative slot on `breakfast_table_rhjoby_0` fall into 8 classes (0.02: blocks, lids,
caps, books, magazine, letter, keyboard, catalog; 0.04: cork, hardback; 0.06: frying pan; 0.08: light bulb, bag of chips,
teacup; 0.10: box of sanitary napkins, bottle of medicine; 0.14: skateboard wheel assembly, gym shoe; 0.16: dip candle;
0.22: carton of milk). The **cell is the contact point** (x ahead, lateral, base frame at the same edge stance as the
right-hand bands: edge midpoint, s = 0.45 + overhang), not the object centre: placement computes the brace point on the
object's face and looks it up. Pose set (`CONTACT_CLASSES`, bit order of the per-cell bitmask), closed gripper
(`fingers = 0`), the idle right arm locked in the trunk's posture, pre-contact 10 cm back along −a against the full
world, contact pose with the support ignored (§3.6), eef origin = point − `CONTACT_CENTER_Z`·a (pad centre on the point):

| bit | class | approach a (base frame) | closing axis c | contact height |
|---|---|---|---|---|
| 0 / 1 | `near_v` / `near_h` | +x (pads on the face toward the robot, pushing away) | up / lateral | top + ¾ h |
| 2 / 3 | `far_v` / `far_h` | −x (pads on the far face, pulling toward the robot) | up / lateral | top + ¾ h |
| 4 / 5 | `left_v` / `left_h` | −y (pads on the robot-left face, pushing to the robot's right: row-left / touching-partner braces) | up / forward | top + ¾ h |
| 6 / 7 | `right_v` / `right_h` | +y (pads on the robot-right face, pushing left) | up / forward | top + ¾ h |
| 8 / 9 | `top_y0` / `top_y90` | −z (top press) | lateral / forward | top + h |

Smoke (h = 0.10, trunk (0.5, 0), 8 cm): 34 cells, union extent [−0.16, +0.56] (the left band lies to the robot's left,
mirroring the right band's [−0.56, +0.08]); classes passing: `near_h` 28, `top_y90` 20, `near_v` 13, `left_h` 8,
`left_v` 5, `top_y0` 5, `far_*` / `right_*` 0 (pulling faces and the robot-right face are out of the left arm's reach).

**Pitched push / pinch classes** (`--generic-pinch`): the generic set is now 2 wrist yaws × pitches {0, 30, 45, 60}° (the
approach vector tilted from straight down toward the robot's forward direction, i.e. the hand comes from the robot's
side; `GENERIC_CLASSES = p00_y0, p00_y90, p30_y0, …, p60_y90`, pitch-major). Every generic / contact band stores per
trunk `t<..>__classes` = `(N,)` int bitmask (bit k = class `meta["classes"][k]` passed at the cell); the `pairs` array
still lists `(cell, -1, class)` for every passing class. The axial / block bands were recomputed with the extended set
(new cache keys; the old two-pitch bands were removed).

**Idle-arm posture retry (A6)**: a trunk whose untucked configuration is not collision-free is retried with
`ARM_READY_Q` (`ready`) and then `tucked` (`POSTURES`); the posture used is recorded per trunk (`trunks[].posture`,
`trunks[].idle_arm_q = {left: [7], right: [7]}`) and returned by `ReachBands.posture(translate, tilt)`; the stance
selection uses the same retry. Measured on the floor band (`childs_room_0`, apple): (0.5, −1.2) and (0.75, −1.2) are
unblocked by `tucked` (`ready` still collides), (0.95, −0.9) and (0.95, −1.2) collide with every posture at a floor
stance (torso / head within the 1 cm margin of the floor mesh), and at the table stance the crouched trunks collide with
the table whatever the arms do. The `brace` mode and the v2 `articulation_small` module's `brace_candidates` consumer
keep working (`BraceCandidate` unpacks as `(kind, pair)`, `.q_prefix` carries the joint-value prefix).

**Loader API (additions, backward compatible)**::

    rb_r = ReachBands.load(scene, support, obj, grasp_key)                                   # right-hand band (unchanged)
    rb_l = ReachBands.load(scene, support, arm="left", contact_set="generic_contact", height=h)  # left contact band of the object's height class
    rb_l.union_extent()                       # (λmin, λmax) of the left band -> §4.4 margins use the union of both bands' extents
    rb_l.classes_at(translate, tilt, brace_pt)  # ["near_h", "top_y90", ...] contact classes feasible at the brace point (nearest cell)
    rb_l.band_for_class(translate, tilt, "left_h")  # cells where one class passes
    rb_r.classes_at(translate, tilt, push_pt)   # generic push classes at a cell, e.g. ["p00_y0", "p30_y0", "p45_y0"] -> prefer the shallowest pitch
    rb.posture(translate, tilt)               # {"posture": "tucked", "idle_arm_q": {...}} - the idle-arm posture the trunk needs
    rb.class_names                            # bit order of the bitmask
    rb.classes[(translate, tilt)]             # raw (N,) bitmask aligned with rb.cells[(translate, tilt)]

**Measured 2026-09-16 (pilot pods, `breakfast_table_rhjoby_0`)**:

* Left contact bands (8 height classes, 16–21 min each): 17 of 20 trunks are usable (13 untucked, 3 `ready`, 1 `tucked`
  — the posture retry unlocks 4 trunks that collided with the table with untucked arms), 432–815 cells, union extent
  λ ∈ [0.00, +0.56] for the flat classes (0.02 / 0.04 m) and [−0.16, +0.56] from 0.06 m up — the left band lies to the
  robot's **left**, the right-hand bands ([−0.56, +0.08]) to its right, so a two-arm placement has to sit near the
  centre line (overlap λ ∈ [−0.16, +0.08]) with the brace point left of the work point. Classes (cells summed over
  trunks, class 0.10 m): `near_h` 528, `top_y90` 371, `left_h` 243, `near_v` 150, `top_y0` 124, `left_v` 69, `far_*` 0,
  `right_*` 0; for objects ≤ 4 cm only the `near_*` faces and the top presses pass (the side contact at ¾ × 2–4 cm puts
  the closed gripper into the table), `left_h` passes from 6 cm, `left_v` from 8 cm; `far_*` (pulling toward the robot)
  never, `right_*` only 2–6 cells for objects ≥ 14 cm.
* Pitched push classes (11 axial / block generic bands recomputed): 699–718 cells per band vs 427–463 with the two-pitch
  set (17 trunks vs 13 thanks to the posture retry, plus the far cells). Pass rate per class over all band cells by
  distance ahead of the base (`p00` = straight down, `p30/45/60` = approach tilted toward the robot's side; `y0`/`y90` =
  wrist yaw):

  | x ahead | cells | p00_y0 | p00_y90 | p30_y0 | p30_y90 | p45_y0 | p45_y90 | p60_y0 | p60_y90 | only 45/60° |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 0.4–0.5 | 570 | 0.75 | 0.61 | 0.45 | 0.28 | 0.41 | 0.32 | 0.39 | 0.52 | 0.20 |
  | 0.5–0.6 | 599 | 0.76 | 0.65 | 0.60 | 0.46 | 0.50 | 0.38 | 0.44 | 0.46 | 0.08 |
  | 0.6–0.7 | 676 | 0.70 | 0.64 | 0.66 | 0.54 | 0.60 | 0.48 | 0.52 | 0.48 | 0.03 |
  | 0.7–0.8 | 684 | 0.60 | 0.55 | 0.67 | 0.58 | 0.63 | 0.54 | 0.59 | 0.52 | 0.04 |
  | 0.8–0.9 | 1077 | 0.43 | 0.45 | 0.70 | 0.65 | 0.72 | 0.66 | 0.71 | 0.63 | 0.18 |
  | 0.9–1.0 | 336 | 0.17 | 0.21 | 0.69 | 0.66 | 0.85 | 0.79 | 0.88 | 0.86 | 0.31 |
  | 1.0–1.1 | 199 | 0.00 | 0.00 | 0.57 | 0.54 | 0.84 | 0.83 | 1.00 | 0.92 | 0.43 |

  Straight-down pushes die beyond 0.9 m; 31–43 % of the cells at 0.9–1.1 m are reachable **only** with the 45° / 60°
  pitches — placement should pick the shallowest passing pitch per cell (`classes_at`).
* Coverage (`scripts/reach_bands/check_coverage.py`, pilot manifest `runlogs/pilot_20260915/jobs_sample.json`, 197
  aliases): 110 aliases need no tabletop band, 74 (support, object, grasp set) needs are covered by a band with cells, 16
  by an **empty** band (8 small-articulation brace bands — see below —, 3 `desk_xhjsub_0` init / placement bands, 4
  `--under` bands, the `sofa_lugrhk_1` placement band), **0 missing** (`coverage.json`).
* Toolbox `no_workspace`: `brace__…toolbox-redmtl__f7a311f9f055b593` was computed at the right height (z = top 0.8126 m,
  stand-off 0.45, 225 lattice cells on the top, 123–197 within reach per trunk) and the handle grasp is feasible at 30–70
  (cell, yaw) pairs per trunk for translate 0–0.5, but **no brace contact of §5.4 is IK-feasible** (`opposite_face` =
  the far face of the lid hinged at the back, `top_press` excluded by the swept-AABB rule), so the band is genuinely
  empty at s = 0.45 — a yield problem of the brace rule, not of the band; the left contact band of the toolbox's height
  class (0.14 / 0.16 m: `near_*`, `left_*`, top presses) is the replacement the joint placement should use.

Joint placement (owner A1): work point ∈ `rb_r.band(t, k)` **and** brace point ∈ `rb_l.band(t, k)` at the same trunk, then
the exact two-arm IK; the trunk's `posture()` is the idle-arm posture to record in the stance. `BandComputer.compute`
now returns `result["classes"]` as well — in-process band writers (`sampling.v2.tabletop.compute_bands`) should save it
as `t<..>__classes` and copy `meta["classes"]`, and pass `arm=` / set `bc.contact_height` + `bc.fingers = 0` for a left
contact band.

## Smoke (local RTX 4090, `--smoke`: 16 cm lattice, trunks (0.5, 0) and (0.25, −0.3))

* **Shoulder-FK fix verified**: the band computation's shoulder position (cuRobo FK of `torso_link4` + the constant
  `right_arm_link1` offset measured once on the simulated robot) against the simulated robot's `right_arm_link1` world
  position at the same joint state: **1.4 mm** at trunk (0.5, 0) and **1.4 mm** at (0.95, −0.9) — the same 1.4 mm as
  `torso_link4` and the eef themselves (the parked robot sits 1.4 mm below `NOMINAL_BASE_Z`), i.e. the FK chain is
  right (the URDF puts `right_arm_link1` at `torso_link4` + (−0.0005, −0.1707, 0.3030) through the fixed
  `right_arm_base_joint`). Shoulder height above the floor: 1.231 m at (0.5, 0), 0.794 m at (0.95, −0.9); stretched arm
  (shoulder → eef, all arm joints zero) 0.793 m.
* **Band sanity** (`pack_of_kielbasa/dgodnn`, tabletop grasps, top at 0.813 m, stance 0.45 m from the +v edge): at
  16 cm, 9 cells at (0.5, 0) and 12 at (0.25, −0.3), x 0.48–0.80 m ahead of the base (3–35 cm into the table),
  lateral −0.64..0.00 (robot's right, as for the right arm), no cell behind the base, shoulder → cell distance
  0.53–0.78 m, every cell with ≥ 1 recorded feasible (yaw, grasp) pair. At 8 cm (smoke 3, exhaustive schedule): 31 and
  39 cells, lateral −0.56..+0.08, union extent [−0.56, +0.08].
* **Timing** (the reason for the schedule changes below): per trunk 93 s / 63 s at 16 cm with the original loop
  (4 pairs per round, `RobotKit.ignoring` = two world re-syncs per round, 400 meshes); 69 s / 45 s at 8 cm with the
  final schedule but exhaustive rounds; **the IK throughput is ~20 poses/s** (48 ms per 12-seed pose, 2328 poses in
  113 s of IK) regardless of the mesh count (72 vs 28 meshes: same), so a full 8 cm job = ~20 trunks × 45–70 s +
  2.5 min scene load ≈ **15–25 min** (spec §4.1 estimated 1–4 min). Coarse-to-fine (16 cm first, refine near the
  reachable coarse cells) was tried and dropped: 54 s vs 45 s per trunk and one isolated cell lost.

Schedule in the committed script (all recorded in `meta.schedule`): rounds of 8 (yaw, grasp) pairs (best-ranked grasp ×
the 8 yaws, robot-facing first); IK chunks of 128 poses with the last chunk sized 16/32/64/128 (no padding waste);
poses whose eef or pre-grasp origin lies farther than arm length + 3 cm from the shoulder are skipped without IK; a cell
none of whose first 8 pre-grasp poses is IK-feasible is dropped (`n_dropped_no_pre`; on the smoke job the dropped cells
had no pre-feasible pose in the following rounds either, so the band was identical — the approximation can lose far-rim
cells that only a lower-ranked grasp with another approach direction reaches); the collision world is restricted to the
objects within 1.6 m of the base (below 2.1 m; floors dropped for tabletop bands, the cuRobo ground slab stays); the
support is switched off for the grasp poses by disabling its meshes directly (no world re-sync).

## Fleet run (RunPod SECURE, purpose `rb`, 2026-09-15 21:20 → 2026-09-16 UTC)

Manifests: `jobs.json` (95 jobs, `representative` slot: 70 tabletop picks on `breakfast_table_rhjoby_0` — 30 disembedding
with the source tray placed, 1 block design, 6 attachment targets, 10 axial targets (generic pinch), 23 small-scale IORS
targets —, 5 small-scale articulation brace jobs, 20 large-scale IORS bands: 5 placement bands with the partner as the
support, 5 `--support floor --under` bands, 10 init bands (7 tabletop supports, 3 floor rooms)), `jobs_w2.json` (90
jobs, `variation_geometry` slot, mostly `countertop_ikwqer_0`), `jobs_w3_heldout.json` (77 jobs, `heldout` slot, mostly
`desk_pfsjbc_0` in `office_cubicles_right`). Pods (all SECURE, one process each, 2.5–3 h TTL): round 1 = 5 × A40 +
3 × L40S (21:20–00:20), round 2 = 5 × A40 + 2 × L40S (22:20–01:20), round 3 = 3 × L40S (23:07–02:07), round 4 = 4 × L40S
(00:40–03:10), round 5 = 3 × L40S (01:07–03:37), round 6 = 1 × L40S + 2 × RTX 6000 Ada (02:18, killed 04:03), round 7 =
6 × L40S (04:17, the leftovers of waves 1–2); ≤ 15 live at a time after the first TTL cliff (A40 hosts at load average
30–50 took 2–4× longer per job, so later rounds were L40S only). `dispatch_samp.py` queues (`w01_` representative,
`w04_` variation, `w05_` heldout, `w08_` re-runs, `w09_` brace on L40S only), `scripts/reach_bands/redistribute.sh`
re-balanced the queues after every TTL cliff (`fleet_remaining.py`: a job is done when a `.npz` exists on some pod or
in the pulled copies), `pull_loop.sh` rsynced `/root/samp` every 8–10 min so that little was lost when a pod
self-terminated. The pulled copies live in the git-ignored `runlogs/runpod/rb/<pod name>/<job id>/` (+ `rb_local/`).

Incidents: (1) the first dispatch ran **stale code** on the pods — `dispatch_samp.py` did not sync the new
`ispatialgym/robot/` package that `robot_utils.py` imports since F5 — and all 95 wave-1 jobs died with `ImportError`
after 10–19 s (89 recorded in `done.txt`, 6 still queued); the package was pushed by hand, the dispatcher fixed by the
coordinator (`7384318`), and every job re-queued (**95 re-runs**). (2) The first brace run crashed in
`brace_candidates` (`G.rotate` on an `(8, 3)` corner array, fixed in `fc5eadf`; the same commit records per cell the set
of joint values whose handle poses are feasible and the longest joint-value prefix the brace contact is valid for,
instead of requiring every joint value). (3) Two brace jobs on A40 pods hit the 50 min `timeout` (blender 1613 s +
timeout, carton 3002 s, ~250 s per trunk vs 60 s on an L40S); brace jobs then ran on L40S only with a 90 min timeout.
(4) Rounds 3–5 were bootstrapped from the GitHub `survey` branch (at `69e0ffe`, i.e. the **original** `reach_bands.py`
of `644495e`: 4 pairs per round, no `--scale`, two world re-syncs per round) and were dispatched with `--no-sync`, so
until 02:10 UTC they ran the original script: `--scale` jobs failed at argparse (re-queued), the others took 40–80 min
and produced valid but pair-seed-less bands (15; all recomputed with the committed script in round 7 — `collect_bands.py`
prefers a current-script result when both exist). Fixed by `--sync-only` + `--restart-runner --force`. (5) At 04:03 UTC the
account-wide `rp.py watchdog` (still running with its default `--cap 200`) terminated every pod of the account
(`reason=cap`), including the round-6 pods mid-run; the last pull was at 03:56, so at most 3 finished jobs were lost
(they are in the re-run set). Local runs (`scripts/reach_bands/run_local.sh`, shared RTX 4090, one process at a time,
< 12 GB rule) did the `desk_xhjsub_0` / countertop / sofa init bands.

Timing per job (wall incl. the scene load): L40S tabletop median 8.7 min (4–32), RTX 6000 Ada 8 min, A40 tabletop
median 17.8 min (12–40), floor / under bands 5 min (L40S) – 10 min (A40) (few cells survive the shoulder prune), brace
24 min (19–34, L40S). Fleet spend for the reach bands ≈ $75 (39 pods created over 7 rounds, 7 terminated unused within
minutes: slow-network hosts, the fleet cap, one L40 outside the allowed GPU list).

**Outcome (2026-09-16 06:45 UTC)**: 262 manifest jobs → **218 bands** (179 with cells, 39 empty): the `representative`
slot 95 / 95 (77 with cells; the 18 empty ones are the 5 brace bands, 7 floor / under bands and the 6 `desk_xhjsub_0` /
`sofa_lugrhk_1` bands explained below), the `variation_geometry` slot 90 / 90 (74 with cells), the `heldout` slot 33 / 77
(28 with cells) — the remaining 44 heldout jobs (41 tabletop on `desk_pfsjbc_0` in `office_cubicles_right`, 3 floor
bands) were queued when the account-wide cap termination hit and were not re-run (`jobs_remaining_w3.json` lists them;
`sample_isg` computes a missing band on the fly, spec §4.1). Every committed band comes from the committed script
(`meta.schedule` present); the 15 bands first produced by the original script were recomputed in round 7.

## Findings for the spec (measured)

* **Tabletop bands are wide and consistent**: on `breakfast_table_rhjoby_0` (top 0.81 m, stance 0.45 m from the +v edge)
  every library-grasp object gets 325–608 cells over the 13 collision-free trunks (the 7 crouched trunks (0.25, −0.9),
  (0.5, −0.9), (0.75, −0.6/−0.9), (0.95, −0.3/−0.6/−0.9) collide with the table at stand-off 0.45 with untucked arms),
  union extent λ ∈ [−0.56, 0] to [−0.56, +0.16] (the right arm reaches 0–56 cm to the robot's right, ≤ 16 cm to its
  left), x = 0.48–0.80 m ahead (3–35 cm into the top). Generic-pinch bands (blocks, axial targets): 427–463 cells,
  [−0.56, 0]. Disembedding bands (lattice = the `tray/gsxbym` floor, 8 cm behind the edge): 64–143 cells, λ ∈ [−0.16,
  0..0.08] — the tray floor is 0.40 m wide, the right hand reaches its right half.
* **Support geometry decides more than the object**: `countertop_ikwqer_0` / `fjkase_0` / `kelker_0` (0.89–0.93 m): 9–10
  free trunks, 265–474 cells, λ ∈ [−0.80, +0.08]; `bar_udatjt_0` (1.05 m): 5 free trunks, 108 cells; beds as ontop
  partners (`bed_rrcvaq_0`, `bed_gxfipj_0`, 0.49 m, tilt −1.2 added): 21/25 free trunks, 400–446 cells; `sofa_lugrhk_1`
  (seat 0.40 m): 16 free trunks but **0 cells** (the seat lies beyond the armrest overhang); **`desk_xhjsub_0`
  (house_single_floor, childs_room_0): every Phase-1a stance collides** — its AABB extends 0.66 m beyond the exposed top
  (+u edge), so the spec's stand-off + overhang puts the base 1.11 m from the top edge into the bathroom furniture next
  to it; all 4 init bands and the 2 `--under` bands (the AABB-edge stance collides at every trunk too) are empty → the
  3 under-large picks and 2 ontop-large picks using this desk cannot be initialized under §4.2 / §5.6 as written (the
  overhang rule should use the top's own AABB, or the pick needs another support).
* **Floor is out of reach**: every `--support floor --room R` init band (5 jobs, 21/25 free trunks) and 3 of 5
  `--under` bands are empty: from a standing base the eef cannot reach the floor 0.45–0.9 m ahead (the stretched arm
  is 0.79 m from a shoulder ≥ 0.79 m above the floor at the deepest collision-free crouch; the reach volume's pinch band
  bottoms out at z 0.25 m). Only tall objects under a bed at stand-off 0.45 work (`briefcase/lwbctq` under
  `bed_rrcvaq_0`: 16 cells at translate 0.95 / tilt −0.6…−0.9, λ ∈ [−0.56, −0.16]; `sandal` under the same bed: 0).
  §5.6's floor init and under placement need a smaller stand-off (0.35), a bent-over posture, or a push.
* **Small-scale articulation brace rule yields nothing**: for all five targets the handle is reachable (50–160
  (cell, yaw) combinations per trunk with ≥ 1 feasible joint value) but **no brace contact is IK-feasible**: the
  `opposite_face` of §5.4 is the face the handle motion points toward = the far face of a lid hinged at the back, so the
  brace hand would have to reach behind the object and point back at the robot; `top_press` is excluded by the ≥ 2 cm
  swept-AABB rule because the lid sweeps over the base link's top (laptop: the base link is entirely under the lid, no
  candidate at all). The bands record the handle-feasible pairs (`n_handle_only_pairs`) so the sampler can initialize
  without a brace if the rule is relaxed.
* **Cost**: 15–25 min per (support, object) on an L40S — 5× the §4.1 estimate; the two-factor cache pays off only
  across objects sharing a support (the support factor is < 10 % of a job), so a multi-object mode (one scene load,
  many objects) is the next speed-up.

<!-- results -->

## Results per job (regenerated by `scripts/reach_bands/collect_bands.py`)

Status: {'not run': 124}

| job | kind | status | pod | wall s | trunks free/total | cells | union extent [λmin, λmax] | note |
|---|---|---|---|---|---|---|---|---|
| `rb__contact__restaurant_diner__bar_xxftww_0__h0.02__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__bar_xxftww_0__h0.04__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__bar_xxftww_0__h0.10__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__bar_xxftww_0__h0.12__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__commercial_kitchen_table_vxvtec_0__h0.02__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__commercial_kitchen_table_vxvtec_0__h0.04__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__commercial_kitchen_table_vxvtec_0__h0.10__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__commercial_kitchen_table_vxvtec_0__h0.12__left` | contact | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__apple-bwteqh` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__bag_of_chips-ennnjj__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__bar_soap-utgixp__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__beaker-exzsal__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__board_game-frgdqh__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__bottle_of_beer-rdnopv__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__bottle_of_medicine-qqsukh` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__bottle_of_medicine-qqsukh__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__bottle_of_milk-czblzn__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__bottle_of_shampoo-hlkpwd__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__box_of_baking_mix-fnbgwd__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__box_of_cane_sugar-rvsivw__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__box_of_cereal-sgirte__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__box_of_cookies-levwda__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__box_of_rice-yyheuw__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__box_of_sanitary_napkins-arqcyu` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__box_of_sanitary_napkins-arqcyu__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__box_of_shampoo-tzghev__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__box_of_tissues-xwstls__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__brussels_sprouts-vdamtq` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__can_of_soda-lugwcz__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__carton_of_milk-kklgxk__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__cellulose_tape-sklkyc__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__chili-xhbpqh__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__cinnamon_stick-qmlyim__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__coffee_cup-fgizgn` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__conditioner_dispenser-udeflt__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__cork-lseuwf` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__dip_candle-gbaons` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__french_fries-oeterg__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__frying_pan-aewpzn__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__gym_shoe-arigtn__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__hard_candy-vyxhmv` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__hardback-bqvvvx__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__ice_cube-kdlafa` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__jar_of_spaghetti_sauce-efrxze` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__keyboard-nddmmy__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__letter-bdlpry__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__light_bulb-ymomhw` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__magazine-ehdwsd__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__marker-qemval` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__mouse-eosngb__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__mug-ehnmxj__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__notebook-dlhxxl__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__pack_of_cigarettes-ppbmak__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__pack_of_kielbasa-fwqdoy` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__paperback_book-vciijb__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__parsnip-fwmatc` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__pencil-fnlbpu__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__pencil_case-acikzo__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__pickle-gytwes` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__quartz-drqtci` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__soap_dispenser-sinntg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__tablefork-yuihyg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__teacup-cpozxi` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__teaspoon-nxddfp__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__toy_dice-bjrxkj__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__bar_xxftww_0__video_game-lfdpse__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__apple-bwteqh` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__bag_of_chips-ennnjj__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__bar_soap-utgixp__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__beaker-exzsal__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__board_game-frgdqh__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__bottle_of_beer-rdnopv__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__bottle_of_medicine-qqsukh` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__bottle_of_medicine-qqsukh__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__bottle_of_milk-czblzn__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__bottle_of_shampoo-hlkpwd__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__box_of_baking_mix-fnbgwd__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__box_of_cane_sugar-rvsivw__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__box_of_cereal-sgirte__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__box_of_cookies-levwda__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__box_of_rice-yyheuw__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__box_of_sanitary_napkins-arqcyu` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__box_of_sanitary_napkins-arqcyu__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__box_of_shampoo-tzghev__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__box_of_tissues-xwstls__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__brussels_sprouts-vdamtq` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__can_of_soda-lugwcz__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__carton_of_milk-kklgxk__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__cellulose_tape-sklkyc__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__chili-xhbpqh__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__cinnamon_stick-qmlyim__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__coffee_cup-fgizgn` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__conditioner_dispenser-udeflt__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__cork-lseuwf` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__dip_candle-gbaons` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__french_fries-oeterg__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__frying_pan-aewpzn__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__gym_shoe-arigtn__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__hard_candy-vyxhmv` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__hardback-bqvvvx__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__ice_cube-kdlafa` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__jar_of_spaghetti_sauce-efrxze` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__keyboard-nddmmy__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__letter-bdlpry__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__light_bulb-ymomhw` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__magazine-ehdwsd__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__marker-qemval` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__mouse-eosngb__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__mug-ehnmxj__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__notebook-dlhxxl__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__pack_of_cigarettes-ppbmak__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__pack_of_kielbasa-fwqdoy` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__paperback_book-vciijb__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__parsnip-fwmatc` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__pencil-fnlbpu__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__pencil_case-acikzo__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__pickle-gytwes` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__quartz-drqtci` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__soap_dispenser-sinntg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__tablefork-yuihyg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__teacup-cpozxi` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__teaspoon-nxddfp__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__toy_dice-bjrxkj__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__restaurant_diner__commercial_kitchen_table_vxvtec_0__video_game-lfdpse__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
