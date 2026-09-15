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

<!-- results -->
