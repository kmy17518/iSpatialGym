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
ReachBands.load(scene, support, obj, key, placed=["tray/gsxbym"])        # disembedding, legacy fixed tray (lattice = its floor)
ReachBands.load(scene, support, obj, key, placed=[placed_spec("tray/gsxbym", 1.25)])   # disembedding, stance-relative tray at 1.25 (item 9, see below)
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

## 2026-09-16 (2): queue order (owner 7a), stance-relative placed-tray bands (owner 9 / 4b), commit `634f423`

**Queue order set 2026-09-16 ~15:10 UTC (item 7a)** on the 15 band pods (`runlogs/runpod/pilot/pods_bands.txt`: 8 L40S
`pilot2-l40s-*` + 7 A40 `pilot2-a40-*` that joined at 14:26 UTC). Procedure: `dispatch_samp.py --restart-runner --force
--no-start` (8 running A3 jobs killed and dropped from the queues; `--force` does not kill a `reach_bands` process, so
`scripts/reach_bands/fleet_run.py` killed the orphans and emptied the stale `w03`–`w10` queues, which held the var / held
waves twice), commit + `--sync-only`, then `fleet_remaining.py` (a job is done when a `.npz` exists on some pod or in the
pulled copies — `--rebalance` alone cannot tell finished band jobs apart because the runner records `n_written=-1` for
every CMD job) → two plain dispatches in weight order (`fleet_remaining.py --weights`; the dispatcher writes the queue
files in manifest order):

| wave | pods | jobs | content |
|---|---|---|---|
| `w11` | 8 L40S | **30** | disembedding right-arm bands, `breakfast_table_rhjoby_0`, source tray `tray/gsxbym*1.25@focus_centred_edge_aligned` (`jobs_disemb_tray125.json` = `jobs_q11_disemb.json`, weight 9) |
| `w12` | 15 | **285** | `jobs_q12_rest.json`: A3 rep remaining 6 (weight 3: `countertop_kelker_0` / `conference_table_fpkmyw_0`, 150 / 156 already done), A6 desk / sofa re-runs 8 (2.5), A3 var 147 (2: `bar_udatjt_0` / `desk_mdhelw_1`), A3 held 124 (1: `bar_xxftww_0` / `commercial_kitchen_table_vxvtec_0`); 19 per pod |

Nothing already finished is recomputed (175 of the 460 A3 / A6 manifest jobs had band files). ETA: the 30 disembedding
bands ≈ 2 h (4 per L40S at ~30 min each, see below), then the var / held waves (A40 pods take 2–4× longer per job).

**Left-band coverage per left-brace subtask on `breakfast_table_rhjoby_0`** (the 24 committed left contact bands = 8
height classes × 3 supports; on the breakfast table classes 0.02 / 0.04 / 0.06 / 0.08 / 0.10 / 0.14 / 0.16 / 0.22, all with
cells). Heights from the band metas' `object_extent`; class = `height_class(h)`; the right-hand band of every object
exists with cells (coverage 74 covered / 0 missing for the pilot manifest, `coverage.json`):

| subtask | objects (height → class) | left band |
|---|---|---|
| alignment-axial | bag_of_chips 0.090→0.08, board_game 0.025→0.02, catalog 0.006→0.02, frying_pan 0.062→0.06, gym_shoe 0.145→0.14, hardback 0.031→0.04, keyboard 0.021→0.02, letter 0.014→0.02, magazine 0.010→0.02, paperback_book 0.008→0.02 | all 10 covered |
| configuration_matching-attachment | cap 0.011→0.02, cork 0.045→0.04, dip_candle 0.169→0.16, lid 0.028→0.02, light_bulb 0.071→0.08, skateboard_wheel_assembly 0.137→0.14 | all 6 covered |
| configuration_matching-block_design | toy_dice/isgblk 0.025→0.02 | covered |
| IORS touching | box_of_sanitary_napkins 0.109→0.10, carton_of_milk 0.218→0.22, teacup 0.072→0.08 | all 3 covered |
| small articulation (brace band empty, §5.4; left band of the target's height class is the replacement) | blender 0.394→0.40 (base link 0.23→0.22), hinged_jar 0.204→0.20 (base 0.086→0.08), laptop 0.020→0.02, toolbox 0.352→0.36 (base 0.21→0.22) | laptop covered; blender / hinged_jar / toolbox: nearest class 0.22 (the brace contact is on the base link, so the base-link height applies: 0.22 / 0.08 / 0.22 all exist); the sampler's `load_left_contact_band` accepts a class within 4 cm |

→ every alias of the five left-brace subtasks on the breakfast table can enter the demo loop now as far as bands are
concerned (20 of 20 objects + the 4 small-articulation targets). Missing bands elsewhere (all deferred): the D1 fallback
aliases (`*_s070` axial at scale 0.7: 10 generic bands per support; `blender/eyedvd`, `hinged_jar/vzwhbg` brace bands),
`check_coverage.py` now writes their `--scale` into the miss jobs.

**Placed-tray band keying (item 9).** `--placed` specs are `CATEGORY/MODEL[*SCALE][@MODE]` (`reach_bands.PlacedSpec`,
modes `edge` = the legacy fixed tray of the 2026-09-15 bands, `focus_centred_edge_aligned` = stance-relative). The
agreed spec string of the disembedding source tray at owner item 4b's scale is

    tray/gsxbym*1.25@focus_centred_edge_aligned        == reach_bands.placed_spec("tray/gsxbym", 1.25)

(the tuple `("tray/gsxbym", 1.25, "focus_centred_edge_aligned")`, the list scale `[1.25, 1.25, 1.25]` and the string all
canonicalise to it; `"tray/gsxbym"` alone = `tray/gsxbym@edge`, so the old bands keep matching the old requests).
`ReachBands.find` / `load` compare the canonical strings (`placed_specs` in the meta; metas without the field are read
from their `placed` entries); unknown modes are kept verbatim (a module's own in-process band round-trips).

*What the focus band computes*: for every 8 cm lattice cell c of the near strip of the top (the plain bands' lattice,
base coordinates at the band stance), the tray stands with its **inner-floor centroid on c** and its **long footprint
axis along the edge** (`yaw = tangent_yaw if extent_x >= extent_y else tangent_yaw − π/2`, the sampler's `_tray_yaw`
rule), z = top + `bottom_offset(hull, quat)` + 3 mm (settled), and the grasped object stands on the tray floor at c
(`z_support` = the measured floor z, `placed[].floor_z`). The (yaw, grasp) rounds run as for the plain bands with the
tray hidden; every cell that passes is re-checked with the tray's collision meshes moved to that cell's pose
(`MovingPlaced`: per-mesh pose update in the checker, no world re-sync; pre-grasp only — the grasp pose ignores the
support and the tray, §3.6). The lattice is **not** filtered by the tray's fit on the top: a cell near the front edge is
the stance-relative geometry of a closer stance, so the sampler shifts the band with its stand-off and applies its own
fit / in-frame checks (as `disembedding._source_tray_in_frame` does). Metas: `placed[]` = `spec, scale, mode, pose,
pose_settled, floor_z, floor_centre_local, yaw, yaw_rule, pose_rule, n_lattice_top, n_lattice_fit_at_band_stance`;
per trunk `n_placed_rejected_pairs / n_placed_rejected_cells / placed_verify_s`; `--smoke` adds `placed[].self_check`
(moved-mesh poses vs the simulator: 3.0 mm / 0.02° on the 1.25 tray).

*Contract for the tabletop agent (`sampling/v2/tasks/disembedding.py`)*: request
`ReachBands.load(scene, support, obj, grasp_key, placed=[RB.placed_spec(f"{src.category}/{src.model}", src.scale)], scale=…)`
with `src.scale` = the customs `roles.source.scale` (D1 writes `{"tray/gsxbym": 1.25, "default": 1.25}`; a list or a
number both canonicalise to `*1.25`); place the source tray with its inner-floor centroid on the focus and the yaw rule
above (what `_tray_pose_centred` does), the target at the focus (`phase3_target`, within 5 cm), and keep the fit /
in-frame checks at the stance. An in-process fallback (`compute_bands`) should store the same canonical string in
`placed_specs` (its whole-top lattice at the tray-floor z is the same approximation minus the per-cell tray check).

*Measured (local RTX 4090, tray 1.25, 8 cm)*: `mug/ycbmug` trunks (0.5, 0) 21 cells, (0.5, −0.3) 31, (0, −0.6) 28 — x
0.48–0.80, λ ∈ [−0.56, 0], i.e. the plain bands' shape (the legacy fixed-tray band had 1–9 cells per trunk because its
lattice was the tray floor only, 0.53–0.85 m ahead); the tray rejected 0–1 pre-grasp pairs and no cell per trunk
(3–7 s of verification); `box_of_cereal/iucshm` (0.5, −0.3): 27 cells, 1 pair rejected. 75–120 s per trunk → a full job
≈ 25–35 min on an L40S incl. the scene load. Since core's `7a340e4` (spawn re-parks every earlier spawn) the target
spawn moved the placed tray to the parking spot; `restore_placed()` puts it back (the `edge` mode is an obstacle again).

**Under-edge floor band (owner item 7, push-under support; started after items 1–2 were queued).** New mode
`--support floor --under PARTNER --under-edge --generic-pinch [--object C/M] --posture tucked --fingers 0`: the stance is
the partner-edge stance (`stance_under`: base square to the partner's near AABB edge, facing inward = the push direction,
`--stand-off` s), the lattice = the free floor strip in front of that edge from `UNDER_EDGE_OUT_M` = 0.50 m outside it
(the drop cells of the large-scale sampler: half extent + 10 cm outside) to `UNDER_EDGE_IN_M` = 0.30 m under it (goal
cells, kept when the raycast clearance ≥ object height + 3 cm), the classes = the pitched push set
(`GENERIC_CLASSES`, eef at the object-centre height `floor + max(0.6 h, 3 cm)`), closed gripper (`--fingers 0`, the
blade), idle arm tucked (`--posture` = the posture order to try; recorded per trunk). Loader:
`ReachBands.load(scene, "floor", obj, "generic", under=partner, under_edge=True)` (`under_edge=False` = the plain
`--under` bands, `None` = either); file `floor__<scene>__floor-under-edge-<partner>__<obj>__<key>`; meta `under_edge`
= `{out_m, in_m, clearance_extra_m, n_outside, n_under_near, n_under_clear, object_height_m, edge_depth_m, n_lattice}`.

*Measured (local, `sandal/chgcrm` under `bed_rrcvaq_0`, the pilot's under-large pick, 8 cm, trunks (0.5/0.75/0.95,
−0.9/−1.2))*: at s = 0.45 the band is **empty** — (0.95, ·) collide with the bed at every posture, (0.75, −0.9) and
(0.5, −1.2) have every push pose farther than arm length + 3 cm from the shoulder (eef 4.3 cm above the floor), (0.75,
−1.2) screens 152 poses on 84 cells and none is IK-feasible (the crouched torso over the 0.49 m bed edge). The s = 0.65
run (the large-scale agent's stance range 0.5–0.7, all postures) is reported below this table when it finishes. The
push-under record's blade poses (`iors_large._push_under_chain`: eef at `floor + clamp(h/2, 4.5, 8 cm)`, approach pitched
0–60° toward the push, ±wrist) have the same eef height as the band's push classes, so the band is the right screen once
a stance with reachable floor cells exists; the finding matches §5.6's note (floor init / under placement need a bent
posture or a closer stand-off) — for the demo the large-scale agent's exact-IK chain per candidate stance stays the path.

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

Status: {'empty band': 34, 'not run': 323, 'ok': 337, 'failed': 6}

| job | kind | status | pod | wall s | trunks free/total | cells | union extent [λmin, λmax] | note |
|---|---|---|---|---|---|---|---|---|
| `rb__brace__house_single_floor__breakfast_table_rhjoby_0__blender-cwkvib-j_lid` | brace | empty band | rb7-l40s-4 | 2012 | 13/20 | 0 | null |  |
| `rb__brace__house_single_floor__breakfast_table_rhjoby_0__carton-causya-j_link_2` | brace | empty band | rb7-l40s-5 | 2004 | 13/20 | 0 | null |  |
| `rb__brace__house_single_floor__breakfast_table_rhjoby_0__hinged_jar-gqtsam-j_cap` | brace | empty band | rb-l40s-0 | 1141 | 13/20 | 0 | null |  |
| `rb__brace__house_single_floor__breakfast_table_rhjoby_0__laptop-nvulcs-j_link_0` | brace | empty band | rb-l40s-1 | 1167 | 13/20 | 0 | null |  |
| `rb__brace__house_single_floor__breakfast_table_rhjoby_0__toolbox-redmtl-j_lid` | brace | empty band | rb-l40s-2 | 1443 | 13/20 | 0 | null |  |
| `rb__brace__house_single_floor__countertop_ikwqer_0__blender-eyedvd-j_lid` | brace | empty band | rb7-l40s-3 | 1820 | 9/20 | 0 | null |  |
| `rb__brace__house_single_floor__countertop_ikwqer_0__carton-msfzpz-j_link_0` | brace | empty band | rb7-l40s-4 | 1460 | 9/20 | 0 | null |  |
| `rb__brace__house_single_floor__countertop_ikwqer_0__hinged_jar-vzwhbg-j_rb1` | brace | empty band | rb7-l40s-5 | 1371 | 9/20 | 0 | null |  |
| `rb__contact__house_single_floor__bar_udatjt_0__h0.02__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__house_single_floor__bar_udatjt_0__h0.04__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__house_single_floor__bar_udatjt_0__h0.06__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__house_single_floor__bar_udatjt_0__h0.08__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__house_single_floor__bar_udatjt_0__h0.10__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__house_single_floor__bar_udatjt_0__h0.12__left` | contact | ok | pilot2-l40s-7 | 963 | 11/20 | 641 | [-0.1599999964237213, 0.800000011920929] |  |
| `rb__contact__house_single_floor__bar_udatjt_0__h0.14__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__house_single_floor__bar_udatjt_0__h0.18__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__house_single_floor__breakfast_table_rhjoby_0__h0.02__left` | contact | ok | pilot2-l40s-0 | 1098 | 17/20 | 432 | [0.0, 0.5600000023841858] |  |
| `rb__contact__house_single_floor__breakfast_table_rhjoby_0__h0.04__left` | contact | ok | pilot2-l40s-1 | 1073 | 17/20 | 440 | [0.0, 0.5600000023841858] |  |
| `rb__contact__house_single_floor__breakfast_table_rhjoby_0__h0.06__left` | contact | ok | pilot2-l40s-2 | 986 | 17/20 | 726 | [-0.1599999964237213, 0.5600000023841858] |  |
| `rb__contact__house_single_floor__breakfast_table_rhjoby_0__h0.08__left` | contact | ok | pilot2-l40s-3 | 1043 | 17/20 | 774 | [-0.1599999964237213, 0.5600000023841858] |  |
| `rb__contact__house_single_floor__breakfast_table_rhjoby_0__h0.10__left` | contact | ok | pilot2-l40s-4 | 1039 | 17/20 | 786 | [-0.1599999964237213, 0.5600000023841858] |  |
| `rb__contact__house_single_floor__breakfast_table_rhjoby_0__h0.14__left` | contact | ok | pilot2-l40s-7 | 1075 | 17/20 | 792 | [-0.1599999964237213, 0.5600000023841858] |  |
| `rb__contact__house_single_floor__breakfast_table_rhjoby_0__h0.16__left` | contact | ok | pilot2-l40s-9 | 1233 | 17/20 | 800 | [-0.1599999964237213, 0.5600000023841858] |  |
| `rb__contact__house_single_floor__breakfast_table_rhjoby_0__h0.22__left` | contact | ok | pilot2-l40s-10 | 1276 | 17/20 | 815 | [-0.1599999964237213, 0.5600000023841858] |  |
| `rb__contact__house_single_floor__countertop_kelker_0__h0.02__left` | contact | ok | pilot2-l40s-0 | 986 | 17/20 | 203 | [0.0, 0.800000011920929] |  |
| `rb__contact__house_single_floor__countertop_kelker_0__h0.04__left` | contact | ok | pilot2-l40s-1 | 925 | 17/20 | 236 | [0.0, 0.800000011920929] |  |
| `rb__contact__house_single_floor__countertop_kelker_0__h0.06__left` | contact | ok | pilot2-l40s-2 | 889 | 17/20 | 367 | [-0.1599999964237213, 0.800000011920929] |  |
| `rb__contact__house_single_floor__countertop_kelker_0__h0.08__left` | contact | ok | pilot2-l40s-3 | 954 | 17/20 | 387 | [-0.23999999463558197, 0.800000011920929] |  |
| `rb__contact__house_single_floor__countertop_kelker_0__h0.10__left` | contact | ok | pilot2-l40s-4 | 943 | 17/20 | 437 | [-0.23999999463558197, 0.800000011920929] |  |
| `rb__contact__house_single_floor__countertop_kelker_0__h0.14__left` | contact | ok | pilot2-l40s-7 | 939 | 17/20 | 465 | [-0.23999999463558197, 0.800000011920929] |  |
| `rb__contact__house_single_floor__countertop_kelker_0__h0.16__left` | contact | ok | pilot2-l40s-9 | 1094 | 17/20 | 465 | [-0.23999999463558197, 0.800000011920929] |  |
| `rb__contact__house_single_floor__countertop_kelker_0__h0.22__left` | contact | ok | pilot2-l40s-10 | 1116 | 17/20 | 463 | [-0.23999999463558197, 0.800000011920929] |  |
| `rb__contact__office_cubicles_right__conference_table_fpkmyw_0__h0.02__left` | contact | ok | pilot2-l40s-0 | 759 | 18/20 | 467 | [0.0, 0.800000011920929] |  |
| `rb__contact__office_cubicles_right__conference_table_fpkmyw_0__h0.04__left` | contact | ok | pilot2-l40s-1 | 728 | 18/20 | 483 | [0.0, 0.800000011920929] |  |
| `rb__contact__office_cubicles_right__conference_table_fpkmyw_0__h0.06__left` | contact | ok | pilot2-l40s-2 | 716 | 18/20 | 720 | [-0.1599999964237213, 0.800000011920929] |  |
| `rb__contact__office_cubicles_right__conference_table_fpkmyw_0__h0.08__left` | contact | ok | pilot2-l40s-3 | 763 | 18/20 | 758 | [-0.1599999964237213, 0.800000011920929] |  |
| `rb__contact__office_cubicles_right__conference_table_fpkmyw_0__h0.10__left` | contact | ok | pilot2-l40s-4 | 767 | 18/20 | 819 | [-0.1599999964237213, 0.800000011920929] |  |
| `rb__contact__office_cubicles_right__conference_table_fpkmyw_0__h0.14__left` | contact | ok | pilot2-l40s-7 | 819 | 18/20 | 869 | [-0.1599999964237213, 0.800000011920929] |  |
| `rb__contact__office_cubicles_right__conference_table_fpkmyw_0__h0.16__left` | contact | ok | pilot2-l40s-9 | 915 | 18/20 | 902 | [-0.1599999964237213, 0.800000011920929] |  |
| `rb__contact__office_cubicles_right__conference_table_fpkmyw_0__h0.22__left` | contact | ok | pilot2-l40s-10 | 950 | 18/20 | 971 | [-0.1599999964237213, 0.800000011920929] |  |
| `rb__contact__office_cubicles_right__desk_mdhelw_1__h0.02__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__office_cubicles_right__desk_mdhelw_1__h0.04__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__office_cubicles_right__desk_mdhelw_1__h0.06__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__office_cubicles_right__desk_mdhelw_1__h0.08__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__office_cubicles_right__desk_mdhelw_1__h0.10__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__office_cubicles_right__desk_mdhelw_1__h0.12__left` | contact | ok | pilot2-l40s-7 | 764 | 17/20 | 991 | [-0.1599999964237213, 0.800000011920929] |  |
| `rb__contact__office_cubicles_right__desk_mdhelw_1__h0.14__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__office_cubicles_right__desk_mdhelw_1__h0.18__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__bar_xxftww_0__h0.02__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__bar_xxftww_0__h0.04__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__bar_xxftww_0__h0.10__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__bar_xxftww_0__h0.12__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__commercial_kitchen_table_vxvtec_0__h0.02__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__commercial_kitchen_table_vxvtec_0__h0.04__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__commercial_kitchen_table_vxvtec_0__h0.10__left` | contact | not run |  |  | / |  |  |  |
| `rb__contact__restaurant_diner__commercial_kitchen_table_vxvtec_0__h0.12__left` | contact | not run |  |  | / |  |  |  |
| `rb__floor__Rs_int__floor-under-bed_zrumze_0__briefcase-uquevd` | floor | empty band | rb2-l40s-0 | 243 | 18/25 | 0 | null |  |
| `rb__floor__Rs_int__floor-under-bed_zrumze_0__sandal-spcddb` | floor | empty band | rb2-a40-0 | 327 | 18/25 | 0 | null |  |
| `rb__floor__house_double_floor_lower__floor-under-sofa_frlxgz_0__pencil-vpzhth__scaled` | floor | empty band | rb-l40s-0 | 308 | 0/25 | 0 | null |  |
| `rb__floor__house_double_floor_upper__floor-bedroom_0__apple-bwteqh` | floor | ok | rb5-l40s-1 | 315 | 25/25 | 47 | [-0.47999998927116394, -0.07999999821186066] |  |
| `rb__floor__house_double_floor_upper__floor-bedroom_0__marker-qemval` | floor | ok | rb5-l40s-0 | 276 | 25/25 | 25 | [-0.4000000059604645, -0.1599999964237213] |  |
| `rb__floor__house_double_floor_upper__floor-bedroom_0__pencil-fnlbpu__scaled` | floor | not run |  |  | / |  |  |  |
| `rb__floor__house_double_floor_upper__floor-under-desk_aduafr_0__apple-bwteqh` | floor | empty band | rb5-l40s-2 | 227 | 7/25 | 0 | null |  |
| `rb__floor__house_double_floor_upper__floor-under-desk_aduafr_0__marker-qemval` | floor | empty band | rb2-l40s-0 | 168 | 7/25 | 0 | null |  |
| `rb__floor__house_double_floor_upper__floor-under-sofa_ukfaaz_0__pencil-fnlbpu__scaled` | floor | not run |  |  | / |  |  |  |
| `rb__floor__house_double_floor_upper__floor-under-sofa_ukfaaz_0__pencil-lykqoh__scaled` | floor | empty band | rb-a40b-0 | 664 | 25/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-childs_room_0__apple-omzprq` | floor | empty band | pilot2-l40s-1 | 326 | 23/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-childs_room_0__marker-gldosk` | floor | empty band | pilot2-l40s-2 | 286 | 23/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-childs_room_0__pencil-gbzzcg__scaled` | floor | empty band | pilot2-l40s-3 | 315 | 23/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-childs_room_1__apple-zlxfxt` | floor | empty band | pilot2-l40s-3 | 356 | 23/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-childs_room_1__marker-kjxpdi` | floor | empty band | pilot2-l40s-4 | 292 | 23/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-childs_room_1__pencil-lruiot__scaled` | floor | empty band | pilot2-l40s-7 | 299 | 23/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-under-bed_rrcvaq_0__briefcase-lwbctq` | floor | ok | pilot2-l40s-4 | 603 | 23/25 | 24 | [-0.6399999856948853, -0.1599999964237213] |  |
| `rb__floor__house_single_floor__floor-under-bed_rrcvaq_0__briefcase-psvdau` | floor | not run |  |  | / |  |  |  |
| `rb__floor__house_single_floor__floor-under-bed_rrcvaq_0__sandal-chgcrm` | floor | empty band | pilot2-l40s-7 | 301 | 23/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-under-bed_rrcvaq_0__sandal-shdahi` | floor | empty band | rb4-l40s-0 | 299 | 21/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-under-desk_bhkhxo_0__apple-zlxfxt` | floor | empty band | rb2-a40b-0 | 518 | 0/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-under-desk_bhkhxo_0__marker-kjxpdi` | floor | empty band | rb2-a40b-1 | 596 | 0/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-under-desk_xhjsub_0__apple-omzprq` | floor | empty band | rb-l40s-1 | 291 | 0/25 | 0 | null |  |
| `rb__floor__house_single_floor__floor-under-desk_xhjsub_0__marker-gldosk` | floor | empty band | pilot2-l40s-10 | 392 | 15/25 | 0 | null |  |
| `rb__tabletop__Rs_int__countertop_tpuwys_0__shampoo_dispenser-ndrvvm` | tabletop | ok | rb6-l40s-0 | 348 | 5/20 | 106 | [-0.7200000286102295, 0.0] |  |
| `rb__tabletop__house_double_floor_upper__bed_tjfzsb_0__beaker-exzsal` | tabletop | ok | rb6-6000ada-0 | 745 | 22/25 | 534 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_double_floor_upper__bed_tjfzsb_0__pen-gqwdor` | tabletop | ok | rb6-6000ada-1 | 635 | 22/25 | 362 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_double_floor_upper__desk_aduafr_0__pencil-fnlbpu__scaled` | tabletop | ok | rb5-l40s-1 | 356 | 7/20 | 213 | [-0.7200000286102295, 0.0] |  |
| `rb__tabletop__house_double_floor_upper__desk_aduafr_0__pencil-lykqoh__scaled` | tabletop | ok | rb-a40b-0 | 785 | 7/20 | 231 | [-0.7200000286102295, 0.07999999821186066] |  |
| `rb__tabletop__house_double_floor_upper__desk_aduafr_0__pencil-vpzhth__scaled` | tabletop | ok | rb2-l40s-0 | 485 | 7/20 | 232 | [-0.7200000286102295, 0.07999999821186066] |  |
| `rb__tabletop__house_double_floor_upper__sofa_ukfaaz_0__box_of_shampoo-tzghev` | tabletop | empty band | rb5-l40s-0 | 233 | 22/25 | 0 | null |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__apple-netbsb` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__bag_of_chips-bryahw__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__bar_soap-feqemg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__beaker-sfvswx__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__beer_glass-rxscji` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__board_game-eoywpy__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__bottle_of_beer-dcwvkg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__bottle_of_medicine-egondf__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__bottle_of_medicine-zfbnjh` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__bottle_of_milk-ddvsgl__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__bottle_of_shampoo-dvrzmy__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__bottle_of_supplements-kgreql` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__bottle_of_wine-wcokej__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__box_of_cane_sugar-qvjiti__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__box_of_cereal-ykuyqb__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__box_of_cookies-dlhqft__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__box_of_rice-fvulow__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__box_of_sanitary_napkins-acaivd__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__box_of_sanitary_napkins-zqinib` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__box_of_shampoo-bclpiq` | tabletop | ok | rb-a40b-1 | 1156 | 5/20 | 108 | [-0.7200000286102295, 0.0] |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__box_of_shampoo-nijidu` | tabletop | ok | rb2-a40-0 | 1071 | 5/20 | 88 | [-0.7200000286102295, 0.0] |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__box_of_shampoo-nijidu__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__box_of_shampoo-tzghev` | tabletop | ok | rb2-l40sb-0 | 516 | 5/20 | 91 | [-0.7200000286102295, 0.0] |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__brussels_sprouts-siodbb` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__can_of_soda-evcxlr__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__cap-hbafeb` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__carton_of_milk-atyqub__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__carton_of_milk-kszoro` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__catalog-uilzqm__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__cellulose_tape-gchdhk__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__chili-rafkbt__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__cinnamon_stick-kdaxdy__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__coffee_cup-ckkwmj` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__conditioner_dispenser-uncljo__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__cork-ncxgpe` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__dip_candle-ghctun` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__french_fries-xshwvn__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__frying_pan-sfbdjn__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__gym_shoe-qxecbh__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__half_bagel-hsvakn` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__hard_candy-yuatcg` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__hardback-agbqsy__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__ice_cube-tqsquu` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__jar_of_spaghetti_sauce-fyocnu` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__keyboard-jtdyix__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__letter-qhqhwk__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__lid-zkderg` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__light_bulb-sxkjea` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__magazine-zfeoeu__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__marker-kjxpdi` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__mouse-pljjtu__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__mug-waqrdy__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__notebook-yibctu__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__pack_of_cigarettes-qabdgg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__pack_of_kielbasa-femhoz` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__paperback_book-aubynm__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__parsnip-invznw` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__pencil-rzscao__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__pencil_case-cqxkgl__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__pickle-ifterk` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__quartz-poplsu` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__skateboard_wheel_assembly-acmbvb` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__soap_dispenser-cgxrwd__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__sunflower-mdpoct` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__tablefork-uodzrq__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__teacup-tfzfam` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__teacup-wopjex` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__teaspoon-piqabk__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__toy_dice-ievnsq__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bar_udatjt_0__video_game-ppnzqm__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__bed_gxfipj_0__beaker-dtjmai` | tabletop | ok | rb2-a40-1 | 1664 | 21/25 | 631 | [-0.800000011920929, 0.1599999964237213] |  |
| `rb__tabletop__house_single_floor__bed_gxfipj_0__pen-hiyeki` | tabletop | ok | rb2-a40b-0 | 1395 | 21/25 | 401 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__bed_rrcvaq_0__beaker-fyrkzs` | tabletop | ok | rb-l40s-1 | 1037 | 21/25 | 446 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__bed_rrcvaq_0__pen-gzrbrx` | tabletop | ok | rb-l40s-2 | 833 | 21/25 | 438 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__apple-zlxfxt` | tabletop | ok | rb-a40b-2 | 1642 | 13/20 | 426 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__bag_of_chips-dwkdko__gp` | tabletop | ok | pilot2-l40s-0 | 1151 | 17/20 | 709 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__bar_soap-ofargb__placed-tray-gsxbym` | tabletop | ok | rb2-a40b-1 | 1693 | 13/20 | 143 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__beaker-jdijek__placed-tray-gsxbym` | tabletop | ok | rb-l40s-0 | 482 | 13/20 | 78 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__beer_glass-mxsliu` | tabletop | ok | rb-l40s-1 | 920 | 13/20 | 600 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__board_game-qgvsnb__gp` | tabletop | ok | pilot2-l40s-1 | 1073 | 17/20 | 699 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__bottle_of_beer-meqliv__placed-tray-gsxbym` | tabletop | ok | rb-l40s-1 | 459 | 13/20 | 97 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__bottle_of_medicine-kasbsy` | tabletop | ok | rb-a40b-0 | 1819 | 13/20 | 341 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__bottle_of_medicine-syfpak__placed-tray-gsxbym` | tabletop | ok | rb-l40s-2 | 474 | 13/20 | 81 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__bottle_of_milk-mrejrs__placed-tray-gsxbym` | tabletop | ok | rb-a40-0 | 914 | 13/20 | 100 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__bottle_of_shampoo-stjjjm__placed-tray-gsxbym` | tabletop | ok | rb-a40-1 | 909 | 13/20 | 75 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__bottle_of_supplements-oqakev` | tabletop | ok | rb2-l40s-0 | 1348 | 13/20 | 350 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__bottle_of_wine-lacaoa__placed-tray-gsxbym` | tabletop | ok | rb-l40s-0 | 444 | 13/20 | 64 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__box_of_baking_mix-fmewsz__placed-tray-gsxbym` | tabletop | ok | rb-l40s-1 | 468 | 13/20 | 77 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__box_of_cane_sugar-gpbobo__placed-tray-gsxbym` | tabletop | ok | rb-l40s-2 | 455 | 13/20 | 76 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__box_of_cereal-iucshm__placed-tray-gsxbym` | tabletop | ok | rb-a40b-0 | 890 | 13/20 | 76 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__box_of_cookies-apjokz__placed-tray-gsxbym` | tabletop | ok | rb-a40b-1 | 924 | 13/20 | 81 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__box_of_rice-pstqvm__placed-tray-gsxbym` | tabletop | ok | rb-a40b-2 | 979 | 13/20 | 78 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__box_of_sanitary_napkins-humkqw` | tabletop | ok | rb2-a40-0 | 2085 | 13/20 | 334 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__box_of_sanitary_napkins-tkrzkb__placed-tray-gsxbym` | tabletop | ok | rb2-l40s-0 | 574 | 13/20 | 76 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__box_of_shampoo-bclpiq__placed-tray-gsxbym` | tabletop | ok | rb2-a40-0 | 866 | 13/20 | 80 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__box_of_tissues-uglbjc__placed-tray-gsxbym` | tabletop | ok | rb2-a40-1 | 1443 | 13/20 | 79 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__brussels_sprouts-mbkrxe` | tabletop | ok | rb2-a40-1 | 1610 | 13/20 | 484 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__can_of_soda-bfrzvk__placed-tray-gsxbym` | tabletop | ok | rb2-a40b-0 | 1377 | 13/20 | 74 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__cap-xxqyrt` | tabletop | ok | rb2-a40b-0 | 1681 | 13/20 | 521 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__carton_of_milk-kszoro__placed-tray-gsxbym` | tabletop | ok | rb2-a40b-1 | 874 | 13/20 | 81 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__carton_of_milk-xmugpm` | tabletop | ok | rb-a40-0 | 1552 | 13/20 | 328 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__catalog-aygcnt__gp` | tabletop | ok | pilot2-l40s-2 | 1005 | 17/20 | 699 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__cellulose_tape-kavsnx__placed-tray-gsxbym` | tabletop | ok | rb2-l40sb-0 | 446 | 13/20 | 90 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__chili-agecro__placed-tray-gsxbym` | tabletop | ok | rb2-a40c | 875 | 13/20 | 90 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__cinnamon_stick-qxpzdm__scaled` | tabletop | ok | rb-a40-1 | 1600 | 13/20 | 444 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__coffee_cup-siksnl` | tabletop | ok | rb-l40s-0 | 803 | 13/20 | 330 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__conditioner_dispenser-njqnuo__placed-tray-gsxbym` | tabletop | ok | rb2-a40-1 | 964 | 13/20 | 154 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__cork-uyceta` | tabletop | ok | rb-l40s-1 | 750 | 13/20 | 333 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__dip_candle-poexjo` | tabletop | ok | rb-l40s-0 | 849 | 13/20 | 576 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__french_fries-kxpfcj__scaled` | tabletop | ok | rb-l40s-1 | 717 | 13/20 | 425 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__frying_pan-jpzusm__gp` | tabletop | ok | pilot2-l40s-3 | 1029 | 17/20 | 699 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__gym_shoe-kmcbym__gp` | tabletop | ok | pilot2-l40s-4 | 1017 | 17/20 | 718 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__half_bagel-gywbkh` | tabletop | ok | rb-a40b-1 | 1590 | 13/20 | 608 | [-0.5600000023841858, 0.1599999964237213] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__hard_candy-vwegtt` | tabletop | ok | rb-l40s-2 | 677 | 13/20 | 362 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__hardback-acbrnv__gp` | tabletop | ok | pilot2-l40s-7 | 1013 | 17/20 | 699 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__ice_cube-ujmgkb` | tabletop | ok | rb2-l40s-0 | 849 | 13/20 | 360 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__jar_of_spaghetti_sauce-exmsmm` | tabletop | ok | rb-a40b-2 | 2429 | 13/20 | 325 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__keyboard-injrgc__gp` | tabletop | ok | pilot2-l40s-9 | 1160 | 17/20 | 699 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__letter-oheufk__gp` | tabletop | ok | pilot2-l40s-10 | 1136 | 17/20 | 699 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__lid-tnwisp` | tabletop | ok | rb2-l40s-0 | 836 | 13/20 | 377 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__light_bulb-kfmkwd` | tabletop | ok | rb2-l40sb-0 | 712 | 13/20 | 420 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__magazine-tfovwy__gp` | tabletop | ok | pilot2-l40s-0 | 1127 | 17/20 | 699 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__marker-xakoic` | tabletop | ok | rb2-a40-1 | 1589 | 13/20 | 536 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__mouse-wwyopp__placed-tray-gsxbym` | tabletop | ok | rb-l40s-0 | 475 | 13/20 | 83 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__mug-ycbmug__placed-tray-gsxbym` | tabletop | ok | rb-l40s-0 | 478 | 13/20 | 73 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__notebook-umxsmd__placed-tray-gsxbym` | tabletop | ok | rb-a40b-2 | 989 | 13/20 | 84 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__pack_of_cigarettes-ghbqav__placed-tray-gsxbym` | tabletop | ok | rb-l40s-1 | 460 | 13/20 | 80 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__pack_of_kielbasa-dgodnn` | tabletop | ok | rb2-a40b-0 | 1543 | 13/20 | 493 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__paperback_book-jhaltj__gp` | tabletop | ok | pilot2-l40s-1 | 1060 | 17/20 | 699 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__parsnip-osqisk` | tabletop | ok | rb2-a40b-1 | 1410 | 13/20 | 389 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__pencil-hmawzy__scaled` | tabletop | ok | rb-l40s-2 | 688 | 13/20 | 367 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__pencil_case-otcmih__placed-tray-gsxbym` | tabletop | ok | rb-l40s-2 | 519 | 13/20 | 113 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__pickle-iobrqp` | tabletop | ok | rb-a40b-0 | 1772 | 13/20 | 555 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__quartz-vfuilh` | tabletop | ok | rb2-l40sb-0 | 798 | 13/20 | 558 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__skateboard_wheel_assembly-jljqtt` | tabletop | ok | rb2-a40c | 1482 | 13/20 | 354 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__soap_dispenser-zwudcz__placed-tray-gsxbym` | tabletop | ok | rb2-a40b-0 | 837 | 13/20 | 76 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__sunflower-dkajbh` | tabletop | ok | rb7-l40s-4 | 637 | 13/20 | 479 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__tablefork-pqrtkc__placed-tray-gsxbym` | tabletop | ok | rb2-a40b-1 | 958 | 13/20 | 127 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__teacup-oxfzfe` | tabletop | ok | rb7-l40s-5 | 756 | 13/20 | 479 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__teacup-wopjex` | tabletop | ok | rb7-l40s-3 | 885 | 13/20 | 553 | [-0.5600000023841858, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__teaspoon-qdepdj__placed-tray-gsxbym` | tabletop | ok | rb-a40b-2 | 1001 | 13/20 | 128 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__toy_dice-hbkqqp__placed-tray-gsxbym` | tabletop | ok | rb2-l40s-0 | 567 | 13/20 | 80 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__toy_dice-isgblk__gp` | tabletop | ok | pilot2-l40s-2 | 989 | 17/20 | 699 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__house_single_floor__breakfast_table_rhjoby_0__video_game-ceaxsd__placed-tray-gsxbym` | tabletop | ok | rb2-a40-0 | 891 | 13/20 | 83 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_fjkase_0__shampoo_dispenser-zoyyia` | tabletop | failed | rb-a40-1 | 14 | / |  |  | rc=1 |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__apple-netbsb` | tabletop | ok | rb4-l40s-1 | 659 | 9/20 | 290 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__bag_of_chips-bryahw__gp` | tabletop | ok | rb-l40s-1 | 535 | 9/20 | 387 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__bar_soap-feqemg__placed-tray-gsxbym` | tabletop | ok | rb7-l40s-4 | 378 | 9/20 | 75 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__beaker-sfvswx__placed-tray-gsxbym` | tabletop | ok | rb2-a40-0 | 821 | 9/20 | 92 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__beer_glass-rxscji` | tabletop | ok | rb7-l40s-5 | 897 | 9/20 | 512 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__board_game-eoywpy__gp` | tabletop | ok | rb4-l40s-0 | 513 | 9/20 | 379 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__bottle_of_beer-dcwvkg__placed-tray-gsxbym` | tabletop | ok | rb2-a40-1 | 779 | 9/20 | 51 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__bottle_of_medicine-egondf__placed-tray-gsxbym` | tabletop | ok | rb2-l40s-0 | 522 | 9/20 | 53 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__bottle_of_medicine-zfbnjh` | tabletop | ok | rb2-l40sb-0 | 679 | 9/20 | 279 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__bottle_of_milk-ddvsgl__placed-tray-gsxbym` | tabletop | ok | rb2-a40-0 | 808 | 9/20 | 59 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__bottle_of_shampoo-dvrzmy__placed-tray-gsxbym` | tabletop | ok | rb2-a40-1 | 851 | 9/20 | 85 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__bottle_of_supplements-kgreql` | tabletop | ok | rb2-a40c | 1567 | 9/20 | 327 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__bottle_of_wine-wcokej__placed-tray-gsxbym` | tabletop | ok | rb2-a40b-0 | 736 | 9/20 | 49 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__box_of_cane_sugar-qvjiti__placed-tray-gsxbym` | tabletop | ok | rb2-a40b-1 | 781 | 9/20 | 55 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__box_of_cereal-ykuyqb__placed-tray-gsxbym` | tabletop | ok | rb2-l40sb-0 | 401 | 9/20 | 52 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__box_of_cookies-dlhqft__placed-tray-gsxbym` | tabletop | ok | rb2-a40c | 854 | 9/20 | 57 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__box_of_rice-fvulow__placed-tray-gsxbym` | tabletop | ok | rb7-l40s-3 | 424 | 9/20 | 51 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__box_of_sanitary_napkins-acaivd__placed-tray-gsxbym` | tabletop | ok | rb7-l40s-4 | 367 | 9/20 | 54 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__box_of_sanitary_napkins-zqinib` | tabletop | ok | rb2-l40s-0 | 811 | 9/20 | 265 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__box_of_shampoo-nijidu__placed-tray-gsxbym` | tabletop | ok | rb7-l40s-5 | 372 | 9/20 | 45 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__brussels_sprouts-siodbb` | tabletop | ok | rb7-l40s-3 | 727 | 9/20 | 322 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__can_of_soda-evcxlr__placed-tray-gsxbym` | tabletop | ok | rb2-l40sb-0 | 388 | 9/20 | 53 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__cap-hbafeb` | tabletop | ok | rb5-l40s-1 | 780 | 9/20 | 294 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__carton_of_milk-atyqub__placed-tray-gsxbym` | tabletop | ok | rb7-l40s-3 | 430 | 9/20 | 53 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__carton_of_milk-kszoro` | tabletop | ok | rb7-l40s-4 | 686 | 9/20 | 263 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__catalog-uilzqm__gp` | tabletop | ok | rb2-a40b-1 | 1029 | 9/20 | 384 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__cellulose_tape-gchdhk__placed-tray-gsxbym` | tabletop | ok | rb5-l40s-1 | 485 | 9/20 | 103 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__chili-rafkbt__placed-tray-gsxbym` | tabletop | ok | rb7-l40s-5 | 397 | 9/20 | 86 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__cinnamon_stick-kdaxdy__scaled` | tabletop | ok | rb2-l40sb-0 | 694 | 9/20 | 383 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__coffee_cup-ckkwmj` | tabletop | ok | rb2-a40c | 1608 | 9/20 | 387 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__conditioner_dispenser-uncljo__placed-tray-gsxbym` | tabletop | ok | rb7-l40s-3 | 421 | 9/20 | 54 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__cork-ncxgpe` | tabletop | ok | rb2-l40s-0 | 951 | 9/20 | 451 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__dip_candle-ghctun` | tabletop | ok | rb7-l40s-4 | 847 | 9/20 | 479 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__french_fries-xshwvn__scaled` | tabletop | ok | rb5-l40s-0 | 757 | 9/20 | 308 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__frying_pan-sfbdjn__gp` | tabletop | ok | rb7-l40s-4 | 495 | 9/20 | 387 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__gym_shoe-qxecbh__gp` | tabletop | ok | rb2-l40sb-0 | 517 | 9/20 | 402 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__half_bagel-hsvakn` | tabletop | ok | rb7-l40s-5 | 835 | 9/20 | 606 | [-0.800000011920929, 0.1599999964237213] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__hard_candy-yuatcg` | tabletop | ok | rb4-l40s-3 | 879 | 9/20 | 466 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__hardback-agbqsy__gp` | tabletop | ok | rb7-l40s-3 | 522 | 9/20 | 379 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__ice_cube-tqsquu` | tabletop | ok | rb4-l40s-1 | 734 | 9/20 | 389 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__jar_of_spaghetti_sauce-fyocnu` | tabletop | ok | rb4-l40s-2 | 1472 | 9/20 | 265 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__keyboard-jtdyix__gp` | tabletop | ok | rb4-l40s-1 | 485 | 9/20 | 379 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__letter-qhqhwk__gp` | tabletop | ok | rb2-l40s-0 | 605 | 9/20 | 379 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__lid-zkderg` | tabletop | ok | rb4-l40s-0 | 743 | 9/20 | 367 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__light_bulb-sxkjea` | tabletop | ok | rb6-l40s-0 | 1155 | 9/20 | 340 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__magazine-zfeoeu__gp` | tabletop | ok | rb4-l40s-1 | 1156 | 9/20 | 379 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__marker-kjxpdi` | tabletop | ok | rb2-a40b-1 | 1506 | 9/20 | 451 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__mouse-pljjtu__placed-tray-gsxbym` | tabletop | ok | rb7-l40s-5 | 386 | 9/20 | 78 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__mug-waqrdy__placed-tray-gsxbym` | tabletop | ok | rb4-l40s-3 | 398 | 9/20 | 80 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__notebook-yibctu__placed-tray-gsxbym` | tabletop | ok | rb5-l40s-0 | 414 | 9/20 | 63 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__pack_of_cigarettes-qabdgg__placed-tray-gsxbym` | tabletop | ok | rb2-l40s-0 | 499 | 9/20 | 54 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__pack_of_kielbasa-femhoz` | tabletop | ok | rb2-l40sb-0 | 808 | 9/20 | 407 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__paperback_book-aubynm__gp` | tabletop | ok | rb2-a40c | 1069 | 9/20 | 384 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__parsnip-invznw` | tabletop | ok | rb4-l40s-0 | 1492 | 9/20 | 458 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__pencil-rzscao__scaled` | tabletop | ok | rb7-l40s-3 | 717 | 9/20 | 313 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__pencil_case-cqxkgl__placed-tray-gsxbym` | tabletop | ok | rb2-l40sb-0 | 397 | 9/20 | 55 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__pickle-ifterk` | tabletop | ok | rb5-l40s-0 | 1560 | 9/20 | 461 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__quartz-poplsu` | tabletop | ok | rb4-l40s-3 | 739 | 9/20 | 490 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__shampoo_dispenser-kkbhzw` | tabletop | ok | rb5-l40s-1 | 1095 | 9/20 | 515 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__shampoo_dispenser-ndrvvm` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__shampoo_dispenser-zoyyia` | tabletop | failed | rb-l40s-0 | 11 | / |  |  | rc=1 |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__skateboard_wheel_assembly-acmbvb` | tabletop | ok | rb6-6000ada-0 | 1532 | 9/20 | 529 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__soap_dispenser-cgxrwd__placed-tray-gsxbym` | tabletop | ok | rb7-l40s-4 | 370 | 9/20 | 49 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__sunflower-mdpoct` | tabletop | ok | rb5-l40s-2 | 1948 | 9/20 | 519 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__tablefork-uodzrq__placed-tray-gsxbym` | tabletop | ok | rb4-l40s-3 | 452 | 9/20 | 88 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__teacup-tfzfam` | tabletop | ok | rb6-6000ada-1 | 1689 | 9/20 | 523 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__teacup-wopjex` | tabletop | ok | rb5-l40s-1 | 798 | 9/20 | 487 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__teaspoon-piqabk__placed-tray-gsxbym` | tabletop | ok | rb4-l40s-2 | 1201 | 9/20 | 67 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__toy_dice-ievnsq__placed-tray-gsxbym` | tabletop | ok | rb2-l40sb-0 | 410 | 9/20 | 56 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_ikwqer_0__video_game-ppnzqm__placed-tray-gsxbym` | tabletop | ok | rb5-l40s-2 | 502 | 9/20 | 56 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__apple-zlxfxt` | tabletop | ok | pilot2-l40s-0 | 1327 | 17/20 | 405 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__bag_of_chips-dwkdko__gp` | tabletop | ok | pilot2-l40s-1 | 1098 | 17/20 | 542 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__bar_soap-ofargb__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-2 | 1298 | 16/20 | 51 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__beaker-jdijek__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-3 | 692 | 16/20 | 99 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__beer_glass-mxsliu` | tabletop | ok | pilot2-l40s-4 | 1221 | 17/20 | 624 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__board_game-qgvsnb__gp` | tabletop | ok | pilot2-l40s-7 | 1023 | 17/20 | 512 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__bottle_of_beer-meqliv__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-9 | 778 | 16/20 | 93 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__bottle_of_medicine-kasbsy` | tabletop | ok | pilot2-l40s-10 | 1227 | 17/20 | 307 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__bottle_of_medicine-syfpak__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-0 | 775 | 16/20 | 103 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__bottle_of_milk-mrejrs__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-1 | 914 | 16/20 | 64 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__bottle_of_shampoo-stjjjm__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-2 | 628 | 16/20 | 74 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__bottle_of_supplements-oqakev` | tabletop | ok | pilot2-l40s-3 | 1123 | 17/20 | 318 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__bottle_of_wine-lacaoa__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-4 | 611 | 16/20 | 48 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__box_of_baking_mix-fmewsz__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-7 | 625 | 16/20 | 84 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__box_of_cane_sugar-gpbobo__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-9 | 772 | 16/20 | 84 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__box_of_cereal-iucshm__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-10 | 742 | 16/20 | 52 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__box_of_cookies-apjokz__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-0 | 758 | 16/20 | 94 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__box_of_rice-pstqvm__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-1 | 656 | 16/20 | 74 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__box_of_sanitary_napkins-humkqw` | tabletop | ok | pilot2-l40s-2 | 1092 | 17/20 | 298 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__box_of_sanitary_napkins-tkrzkb__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-3 | 660 | 16/20 | 91 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__box_of_shampoo-bclpiq__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-4 | 622 | 16/20 | 78 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__box_of_tissues-uglbjc__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-7 | 617 | 16/20 | 91 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__brussels_sprouts-mbkrxe` | tabletop | ok | pilot2-l40s-9 | 1371 | 17/20 | 473 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__can_of_soda-bfrzvk__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-10 | 809 | 16/20 | 78 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__cap-xxqyrt` | tabletop | ok | pilot2-l40s-0 | 1603 | 17/20 | 502 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__carton_of_milk-kszoro__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-1 | 668 | 16/20 | 81 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__carton_of_milk-xmugpm` | tabletop | ok | pilot2-l40s-2 | 1032 | 17/20 | 226 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__catalog-aygcnt__gp` | tabletop | ok | pilot2-l40s-3 | 1052 | 17/20 | 513 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__cellulose_tape-kavsnx__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-4 | 973 | 16/20 | 31 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__chili-agecro__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-7 | 898 | 16/20 | 54 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__cinnamon_stick-qxpzdm__scaled` | tabletop | ok | pilot2-l40s-9 | 1363 | 17/20 | 431 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__coffee_cup-siksnl` | tabletop | ok | pilot2-l40s-10 | 1298 | 17/20 | 318 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__conditioner_dispenser-njqnuo__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-0 | 916 | 16/20 | 196 | [-0.23999999463558197, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__cork-uyceta` | tabletop | ok | pilot2-l40s-1 | 1232 | 17/20 | 329 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__dip_candle-poexjo` | tabletop | ok | pilot2-l40s-2 | 1194 | 17/20 | 580 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__french_fries-kxpfcj__scaled` | tabletop | ok | pilot2-l40s-3 | 1231 | 17/20 | 397 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__frying_pan-jpzusm__gp` | tabletop | ok | pilot2-l40s-4 | 1025 | 17/20 | 518 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__gym_shoe-kmcbym__gp` | tabletop | ok | pilot2-l40s-7 | 1061 | 17/20 | 557 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__half_bagel-gywbkh` | tabletop | ok | pilot2-l40s-9 | 1513 | 17/20 | 602 | [-0.800000011920929, 0.1599999964237213] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__hard_candy-vwegtt` | tabletop | ok | pilot2-l40s-10 | 1166 | 17/20 | 356 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__hardback-acbrnv__gp` | tabletop | ok | pilot2-l40s-0 | 1147 | 17/20 | 512 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__ice_cube-ujmgkb` | tabletop | ok | pilot2-l40s-1 | 1246 | 17/20 | 334 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__jar_of_spaghetti_sauce-exmsmm` | tabletop | ok | pilot2-l40s-2 | 1038 | 17/20 | 275 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__keyboard-injrgc__gp` | tabletop | ok | pilot2-l40s-3 | 1058 | 17/20 | 513 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__letter-oheufk__gp` | tabletop | ok | pilot2-l40s-4 | 1039 | 17/20 | 513 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__lid-tnwisp` | tabletop | ok | pilot2-l40s-7 | 1145 | 17/20 | 370 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__light_bulb-kfmkwd` | tabletop | ok | pilot2-l40s-9 | 1274 | 17/20 | 414 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__magazine-tfovwy__gp` | tabletop | ok | pilot2-l40s-10 | 1162 | 17/20 | 513 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__marker-xakoic` | tabletop | ok | pilot2-l40s-0 | 1387 | 17/20 | 502 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__mouse-wwyopp__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-1 | 705 | 16/20 | 103 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__mug-ycbmug__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-2 | 658 | 16/20 | 88 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__notebook-umxsmd__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-3 | 979 | 16/20 | 21 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__pack_of_cigarettes-ghbqav__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-4 | 918 | 16/20 | 21 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__pack_of_kielbasa-dgodnn` | tabletop | ok | pilot2-l40s-7 | 1299 | 17/20 | 481 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__paperback_book-jhaltj__gp` | tabletop | ok | pilot2-l40s-9 | 1162 | 17/20 | 513 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__parsnip-osqisk` | tabletop | ok | pilot2-l40s-10 | 1333 | 17/20 | 346 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__pencil-hmawzy__scaled` | tabletop | ok | pilot2-l40s-0 | 1226 | 17/20 | 344 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__pencil_case-otcmih__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-1 | 868 | 16/20 | 125 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__pickle-iobrqp` | tabletop | ok | pilot2-l40s-2 | 1342 | 17/20 | 526 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__quartz-vfuilh` | tabletop | ok | pilot2-l40s-3 | 1310 | 17/20 | 570 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__shampoo_dispenser-kkbhzw` | tabletop | ok | rb2-l40sb-0 | 716 | 10/20 | 315 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__skateboard_wheel_assembly-jljqtt` | tabletop | ok | pilot2-l40s-4 | 1023 | 17/20 | 285 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__soap_dispenser-zwudcz__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-7 | 648 | 16/20 | 83 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__sunflower-dkajbh` | tabletop | ok | pilot2-l40s-9 | 1285 | 17/20 | 466 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__tablefork-pqrtkc__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-10 | 1069 | 16/20 | 68 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__teacup-oxfzfe` | tabletop | ok | pilot2-l40s-0 | 1256 | 17/20 | 459 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__teacup-wopjex` | tabletop | ok | pilot2-l40s-1 | 1218 | 17/20 | 570 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__teaspoon-qdepdj__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-2 | 932 | 16/20 | 75 | [-0.23999999463558197, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__toy_dice-hbkqqp__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-3 | 986 | 16/20 | 21 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__toy_dice-isgblk__gp` | tabletop | ok | pilot2-l40s-4 | 1021 | 17/20 | 512 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__house_single_floor__countertop_kelker_0__video_game-ceaxsd__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-7 | 917 | 16/20 | 22 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__house_single_floor__desk_bhkhxo_0__pencil-lruiot__scaled` | tabletop | empty band | rb5-l40s-1 | 296 | 0/20 | 0 | null |  |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__beaker-dtjmai` | tabletop | empty band | rb4-l40s-3 | 697 | 0/20 | 0 | null |  |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__beaker-exzsal` | tabletop | empty band | rb5-l40s-2 | 384 | 0/20 | 0 | null |  |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__beaker-fyrkzs` | tabletop | failed | rb-l40s-1 | 11 | / |  |  | rc=1 |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__briefcase-lwbctq` | tabletop | failed | rb-l40s-2 | 11 | / |  |  | rc=1 |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__briefcase-psvdau` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__briefcase-uquevd` | tabletop | empty band | rb5-l40s-0 | 347 | 0/20 | 0 | null |  |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__pen-gqwdor` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__pen-gzrbrx` | tabletop | failed | rb-a40b-0 | 13 | / |  |  | rc=1 |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__pen-hiyeki` | tabletop | empty band | rb7-l40s-5 | 253 | 0/20 | 0 | null |  |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__pencil-gbzzcg__scaled` | tabletop | empty band | rb-a40b-1 | 1359 | 0/20 | 0 | null |  |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__sandal-chgcrm` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__sandal-shdahi` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__house_single_floor__desk_xhjsub_0__sandal-spcddb` | tabletop | empty band | rb5-l40s-2 | 386 | 0/20 | 0 | null |  |
| `rb__tabletop__house_single_floor__sofa_lugrhk_1__box_of_shampoo-bclpiq` | tabletop | failed | rb-a40-0 | 17 | / |  |  | rc=1 |
| `rb__tabletop__house_single_floor__sofa_wnzdke_0__box_of_shampoo-nijidu` | tabletop | ok | rb7-l40s-3 | 781 | 15/20 | 342 | [-0.5600000023841858, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__apple-zlxfxt` | tabletop | ok | pilot2-l40s-9 | 850 | 18/20 | 661 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__bag_of_chips-dwkdko__gp` | tabletop | ok | pilot2-l40s-10 | 692 | 18/20 | 782 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__bar_soap-ofargb__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-0 | 383 | 18/20 | 167 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__beaker-jdijek__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-1 | 331 | 18/20 | 100 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__beer_glass-mxsliu` | tabletop | ok | pilot2-l40s-2 | 988 | 18/20 | 988 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__board_game-qgvsnb__gp` | tabletop | ok | pilot2-l40s-3 | 645 | 18/20 | 747 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__bottle_of_beer-meqliv__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-4 | 288 | 18/20 | 132 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__bottle_of_medicine-kasbsy` | tabletop | ok | pilot2-l40s-7 | 776 | 18/20 | 535 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__bottle_of_medicine-syfpak__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-9 | 365 | 18/20 | 107 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__bottle_of_milk-mrejrs__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-10 | 330 | 18/20 | 131 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__bottle_of_shampoo-stjjjm__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-0 | 324 | 18/20 | 94 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__bottle_of_supplements-oqakev` | tabletop | ok | pilot2-l40s-1 | 923 | 18/20 | 599 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__bottle_of_wine-lacaoa__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-2 | 277 | 18/20 | 93 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__box_of_baking_mix-fmewsz__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-3 | 307 | 18/20 | 102 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__box_of_cane_sugar-gpbobo__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-4 | 289 | 18/20 | 103 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__box_of_cereal-iucshm__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-7 | 260 | 18/20 | 102 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__box_of_cookies-apjokz__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-9 | 333 | 18/20 | 101 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__box_of_rice-pstqvm__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-10 | 329 | 18/20 | 104 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__box_of_sanitary_napkins-humkqw` | tabletop | ok | pilot2-l40s-0 | 863 | 18/20 | 528 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__box_of_sanitary_napkins-tkrzkb__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-1 | 293 | 18/20 | 101 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__box_of_shampoo-bclpiq__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-2 | 275 | 18/20 | 107 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__box_of_tissues-uglbjc__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-3 | 294 | 18/20 | 100 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__brussels_sprouts-mbkrxe` | tabletop | ok | pilot2-l40s-4 | 802 | 18/20 | 753 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__can_of_soda-bfrzvk__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-7 | 284 | 18/20 | 93 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__cap-xxqyrt` | tabletop | ok | pilot2-l40s-9 | 1040 | 18/20 | 789 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__carton_of_milk-kszoro__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-10 | 327 | 18/20 | 108 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__carton_of_milk-xmugpm` | tabletop | ok | pilot2-l40s-0 | 845 | 18/20 | 573 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__catalog-aygcnt__gp` | tabletop | ok | pilot2-l40s-1 | 653 | 18/20 | 747 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__cellulose_tape-kavsnx__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-2 | 271 | 18/20 | 115 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__chili-agecro__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-3 | 298 | 18/20 | 109 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__cinnamon_stick-qxpzdm__scaled` | tabletop | ok | pilot2-l40s-4 | 822 | 18/20 | 697 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__coffee_cup-siksnl` | tabletop | ok | pilot2-l40s-7 | 873 | 18/20 | 537 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__conditioner_dispenser-njqnuo__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-9 | 377 | 18/20 | 180 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__cork-uyceta` | tabletop | ok | pilot2-l40s-10 | 887 | 18/20 | 529 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__dip_candle-poexjo` | tabletop | ok | pilot2-l40s-0 | 979 | 18/20 | 934 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__french_fries-kxpfcj__scaled` | tabletop | ok | pilot2-l40s-1 | 754 | 18/20 | 684 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__frying_pan-jpzusm__gp` | tabletop | ok | pilot2-l40s-2 | 628 | 18/20 | 758 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__gym_shoe-kmcbym__gp` | tabletop | ok | pilot2-l40s-3 | 682 | 18/20 | 834 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__half_bagel-gywbkh` | tabletop | ok | pilot2-l40s-4 | 886 | 18/20 | 960 | [-0.800000011920929, 0.1599999964237213] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__hard_candy-vwegtt` | tabletop | ok | pilot2-l40s-7 | 650 | 18/20 | 578 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__hardback-acbrnv__gp` | tabletop | ok | pilot2-l40s-9 | 682 | 18/20 | 747 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__ice_cube-ujmgkb` | tabletop | ok | pilot2-l40s-10 | 838 | 18/20 | 588 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__jar_of_spaghetti_sauce-exmsmm` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__keyboard-injrgc__gp` | tabletop | ok | pilot2-l40s-1 | 656 | 18/20 | 747 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__letter-oheufk__gp` | tabletop | ok | pilot2-l40s-2 | 629 | 18/20 | 747 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__lid-tnwisp` | tabletop | ok | pilot2-l40s-3 | 819 | 18/20 | 593 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__light_bulb-kfmkwd` | tabletop | ok | pilot2-l40s-4 | 736 | 18/20 | 681 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__magazine-tfovwy__gp` | tabletop | ok | pilot2-l40s-7 | 629 | 18/20 | 747 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__marker-xakoic` | tabletop | ok | pilot2-l40s-9 | 865 | 18/20 | 819 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__mouse-wwyopp__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-10 | 353 | 18/20 | 113 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__mug-ycbmug__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__notebook-umxsmd__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-1 | 292 | 18/20 | 104 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__pack_of_cigarettes-ghbqav__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-2 | 274 | 18/20 | 105 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__pack_of_kielbasa-dgodnn` | tabletop | ok | pilot2-l40s-3 | 884 | 18/20 | 750 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__paperback_book-jhaltj__gp` | tabletop | ok | pilot2-l40s-4 | 627 | 18/20 | 747 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__parsnip-osqisk` | tabletop | ok | pilot2-l40s-7 | 721 | 18/20 | 612 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__pencil-hmawzy__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__pencil_case-otcmih__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-10 | 375 | 18/20 | 142 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__pickle-iobrqp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__quartz-vfuilh` | tabletop | ok | pilot2-l40s-1 | 900 | 18/20 | 892 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__skateboard_wheel_assembly-jljqtt` | tabletop | ok | pilot2-l40s-2 | 757 | 18/20 | 616 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__soap_dispenser-zwudcz__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-3 | 304 | 18/20 | 94 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__sunflower-dkajbh` | tabletop | ok | pilot2-l40s-4 | 675 | 18/20 | 782 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__tablefork-pqrtkc__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-7 | 296 | 18/20 | 145 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__teacup-oxfzfe` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__teacup-wopjex` | tabletop | ok | pilot2-l40s-10 | 929 | 18/20 | 956 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__teaspoon-qdepdj__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__toy_dice-hbkqqp__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-1 | 303 | 18/20 | 105 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__toy_dice-isgblk__gp` | tabletop | ok | pilot2-l40s-2 | 627 | 18/20 | 747 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__conference_table_fpkmyw_0__video_game-ceaxsd__placed-tray-gsxbym` | tabletop | ok | pilot2-l40s-3 | 291 | 18/20 | 105 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__apple-netbsb` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__bag_of_chips-bryahw__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__bar_soap-feqemg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__beaker-sfvswx__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__beer_glass-rxscji` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__board_game-eoywpy__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__bottle_of_beer-dcwvkg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__bottle_of_medicine-egondf__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__bottle_of_medicine-zfbnjh` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__bottle_of_milk-ddvsgl__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__bottle_of_shampoo-dvrzmy__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__bottle_of_supplements-kgreql` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__bottle_of_wine-wcokej__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__box_of_cane_sugar-qvjiti__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__box_of_cereal-ykuyqb__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__box_of_cookies-dlhqft__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__box_of_rice-fvulow__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__box_of_sanitary_napkins-acaivd__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__box_of_sanitary_napkins-zqinib` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__box_of_shampoo-nijidu__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__brussels_sprouts-siodbb` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__can_of_soda-evcxlr__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__cap-hbafeb` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__carton_of_milk-atyqub__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__carton_of_milk-kszoro` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__catalog-uilzqm__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__cellulose_tape-gchdhk__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__chili-rafkbt__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__cinnamon_stick-kdaxdy__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__coffee_cup-ckkwmj` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__conditioner_dispenser-uncljo__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__cork-ncxgpe` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__dip_candle-ghctun` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__french_fries-xshwvn__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__frying_pan-sfbdjn__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__gym_shoe-qxecbh__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__half_bagel-hsvakn` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__hard_candy-yuatcg` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__hardback-agbqsy__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__ice_cube-tqsquu` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__jar_of_spaghetti_sauce-fyocnu` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__keyboard-jtdyix__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__letter-qhqhwk__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__lid-zkderg` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__light_bulb-sxkjea` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__magazine-zfeoeu__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__marker-kjxpdi` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__mouse-pljjtu__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__mug-waqrdy__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__notebook-yibctu__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__pack_of_cigarettes-qabdgg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__pack_of_kielbasa-femhoz` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__paperback_book-aubynm__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__parsnip-invznw` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__pencil-rzscao__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__pencil_case-cqxkgl__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__pickle-ifterk` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__quartz-poplsu` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__skateboard_wheel_assembly-acmbvb` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__soap_dispenser-cgxrwd__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__sunflower-mdpoct` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__tablefork-uodzrq__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__teacup-tfzfam` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__teacup-wopjex` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__teaspoon-piqabk__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__toy_dice-ievnsq__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_mdhelw_1__video_game-ppnzqm__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__apple-bwteqh` | tabletop | ok | rb6-l40s-0 | 611 | 13/20 | 561 | [-0.800000011920929, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__bag_of_chips-ennnjj__gp` | tabletop | ok | rb6-6000ada-0 | 495 | 13/20 | 496 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__bar_soap-utgixp__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__beaker-exzsal__placed-tray-gsxbym` | tabletop | ok | rb6-l40s-0 | 238 | 13/20 | 105 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__board_game-frgdqh__gp` | tabletop | ok | rb6-6000ada-1 | 493 | 13/20 | 478 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__bottle_of_beer-rdnopv__placed-tray-gsxbym` | tabletop | ok | rb6-6000ada-0 | 313 | 13/20 | 83 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__bottle_of_medicine-qqsukh` | tabletop | ok | rb5-l40s-1 | 591 | 13/20 | 399 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__bottle_of_medicine-qqsukh__placed-tray-gsxbym` | tabletop | ok | rb6-6000ada-1 | 319 | 13/20 | 81 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__bottle_of_milk-czblzn__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__bottle_of_shampoo-hlkpwd__placed-tray-gsxbym` | tabletop | ok | rb5-l40s-0 | 379 | 13/20 | 104 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__box_of_baking_mix-fnbgwd__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__box_of_cane_sugar-rvsivw__placed-tray-gsxbym` | tabletop | ok | rb5-l40s-2 | 421 | 13/20 | 80 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__box_of_cereal-sgirte__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__box_of_cookies-levwda__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__box_of_rice-yyheuw__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__box_of_sanitary_napkins-arqcyu` | tabletop | ok | rb5-l40s-0 | 450 | 13/20 | 375 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__box_of_sanitary_napkins-arqcyu__placed-tray-gsxbym` | tabletop | ok | rb6-l40s-0 | 224 | 13/20 | 73 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__box_of_shampoo-tzghev__placed-tray-gsxbym` | tabletop | ok | rb6-6000ada-0 | 301 | 13/20 | 74 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__box_of_tissues-xwstls__placed-tray-gsxbym` | tabletop | ok | rb6-6000ada-1 | 309 | 13/20 | 74 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__brussels_sprouts-vdamtq` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__can_of_soda-lugwcz__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__carton_of_milk-kklgxk__placed-tray-gsxbym` | tabletop | ok | rb5-l40s-0 | 234 | 13/20 | 70 | [-0.1599999964237213, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__cellulose_tape-sklkyc__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__chili-xhbpqh__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__cinnamon_stick-qmlyim__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__coffee_cup-fgizgn` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__conditioner_dispenser-udeflt__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__cork-lseuwf` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__dip_candle-gbaons` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__french_fries-oeterg__scaled` | tabletop | ok | rb6-l40s-0 | 436 | 13/20 | 533 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__frying_pan-aewpzn__gp` | tabletop | ok | rb6-6000ada-0 | 352 | 13/20 | 507 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__gym_shoe-arigtn__gp` | tabletop | ok | rb6-6000ada-1 | 366 | 13/20 | 507 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__hard_candy-vyxhmv` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__hardback-bqvvvx__gp` | tabletop | ok | rb5-l40s-0 | 264 | 13/20 | 478 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__ice_cube-kdlafa` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__jar_of_spaghetti_sauce-efrxze` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__keyboard-nddmmy__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__letter-bdlpry__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__light_bulb-ymomhw` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__magazine-ehdwsd__gp` | tabletop | ok | rb6-l40s-0 | 272 | 13/20 | 478 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__marker-qemval` | tabletop | ok | rb6-6000ada-0 | 482 | 13/20 | 459 | [-0.800000011920929, 0.0] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__mouse-eosngb__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__mug-ehnmxj__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__notebook-dlhxxl__placed-tray-gsxbym` | tabletop | ok | rb6-l40s-0 | 273 | 13/20 | 139 | [-0.1599999964237213, 0.07999999821186066] |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__pack_of_cigarettes-ppbmak__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__pack_of_kielbasa-fwqdoy` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__paperback_book-vciijb__gp` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__parsnip-fwmatc` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__pencil-fnlbpu__scaled` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__pencil_case-acikzo__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__pickle-gytwes` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__quartz-drqtci` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__soap_dispenser-sinntg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__tablefork-yuihyg__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__teacup-cpozxi` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__teaspoon-nxddfp__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__toy_dice-bjrxkj__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
| `rb__tabletop__office_cubicles_right__desk_pfsjbc_0__video_game-lfdpse__placed-tray-gsxbym` | tabletop | not run |  |  | / |  |  |  |
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
