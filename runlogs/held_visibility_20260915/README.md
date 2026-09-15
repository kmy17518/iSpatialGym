# Held-object visibility cache (TODO B7) — 2026-09-15

Per-(model, grasp) visibility of the held object at the **fixed hold** of the held-object tasks (spec §6.3, §3.10), the
prior the v2 sampler (`ispatialgym/sampling/v2/tasks/held_object.py`) reads before it realises a library grasp: rows
whose surveyed visible fraction is below `min_visible_fraction` (0.40) or whose object leaves the frame are skipped
(`held_object.prior_admits`); rows the survey does not know are realised and measured in the sampler as before.

## Method (`scripts/held_visibility/survey.py`)

* **Picks**: every `target` row of `icl/roles/representatives.csv` for `object_scaling` / `mental_rotation`, slots
  `representative`, `variation_geometry`, `heldout`, `variation_texture` → 182 distinct models (15 shared by both
  subtasks). Spawn scale = the pick's `scale` (the grasp-tested scale of the thin models, keys `__sf`).
* **Grasp rows**: the validated `floating` rows of the grasp library (`icl/require/grasps.csv` indexes them: `require =
  grasps:floating`, verdict `ok` or `flag` with flags ⊆ {`one_finger`}) → 869 rows; keys `<cat>__<model>` (GraspGen),
  `__f`, `__sf` (scaled), `__h` (knife handle-half).
* **World**: an empty OmniGibson `Scene` (ground plane) with the R1Pro at the origin, headless, 1080² head camera
  (`robot_r1:zed_link:Camera:0`, 40 mm aperture), `/rtx/raytracing/showLights = 0` as in the sampler.
* **Hold**: exactly the sampler's constant — trunk (0.5, 0) (JoyLo reset), left arm zero, right arm from one cuRobo IK
  of `held_object.hold_eef_pose`: pad centre on the optical axis at depth 0.50 m, approach = base +y (the robot's
  left), closing axis = world down rolled 15° about the approach. Solution (base frame): eef origin
  `[0.7012, 0.0118, 1.2000]`, quaternion `[-0.7011, 0.0923, 0.0923, 0.7011]` (xyzw) — identical to the hold the v1
  representative run found (`runlogs/isg_reps_20260915`). The survey writes it into the sampler's hold cache
  (`~/.cache/ispatialgym/held_hold.json`, key `19590eb8f1c9dcea`).
* **Realisation**: object at `eef ∘ inv(T_obj→eef)` of the library row, fingers at the library's closed `finger_q`,
  gravity off; **no assisted grasping** (the sampler realises every grasp it uses, v1 §5.2 — the attach pulls the
  object in by a few millimetres, which does not change the visibility class).
* **Measurement**: two settled `seg_instance` renders of the head camera (`SimWorld.object_pixels`): the object's
  pixels with the robot visible (`visible_pixels`) and with the robot hidden (`reference_pixels`);
  `visible_fraction = visible / reference`. `in_frame`: every corner of the AABB of the object's collision hull inside
  the image with the 5 % margin. `root_depth_m`: the object root along the optical axis.
* Cost: 0.9 s per grasp (median), ~0.5 s per model spawn / removal, 25 s scene load; 182 models in 15.5 min on the
  local 4090 (one process; the first run died at model 10 from a GPU OOM caused by two other simulator processes —
  `--resume` continued it, `--redo-errors` re-ran the three `__sf` models after the key filter learnt the suffix).

## Outcome (`visibility.json`, schema `held-visibility-v1`)

| | |
|---|---|
| models | 182 (object_scaling 464 rows, mental_rotation 382, both 15 + 8) |
| grasp rows | 869 measured, 0 errors |
| admitted (visible ≥ 0.40 and in frame) | **869 / 869** |
| below 0.40 | 0 |
| out of frame (5 %) | 0 |
| visible fraction | min 0.449, p5 0.793, p25 0.933, median 0.989, max 1.000 |
| distribution | [0.4, 0.6): 5 · [0.6, 0.8): 40 · [0.8, 0.9): 97 · [0.9, 0.95): 110 · ≥ 0.95: 617 |
| reference silhouette | 163 px (tablefork/flexrc) … 41 315 px; median 2 862 px |
| flags | 817 clean, 52 `one_finger` |

The hold puts every surveyed object fully into the image; the gripper approaches from the robot's right and hides at
most half of the object. The least visible rows are thin utensils held across the pads (the pads cover the handle):

| model | grasp | visible | px (vis/ref) | root depth m |
|---|---|---|---|---|
| teaspoon/prbhag | grasp_4 | 0.449 | 128/285 | 0.520 |
| soup_ladle/gxsrhh | grasp_4 | 0.500 | 1364/2726 | 0.566 |
| spatula/pxwpuw | grasp_1 | 0.553 | 1858/3359 | 0.570 |
| teaspoon/prbhag | grasp_3 | 0.556 | 203/365 | 0.486 |
| hook/yfikis | grasp_3 | 0.595 | 260/437 | 0.503 |
| teaspoon/prbhag | grasp_5 | 0.613 | 195/318 | 0.497 |
| soup_ladle/gxsrhh | grasp_3 | 0.619 | 1257/2030 | 0.569 |
| hook/yfikis | grasp_4 | 0.634 | 289/456 | 0.500 |

Models whose best row is below 0.80: teaspoon/nxddfp (0.68), tablefork/yuihyg (0.73), toothbrush/vkrjps (0.78),
teaspoon/prbhag (0.79), tablespoon/pltxjb (0.79). Small silhouettes (a few hundred px: forks, spoons, cherry tomatoes)
are where a 40 % rule bites first once the goal motion moves the object; the sampler re-measures at init and at the
goal in the scene.

## Files

* `visibility.json` — `hold` (the constant + solution), `summary` (totals, `per_model`), `records[]` (one per grasp
  row: `category`, `model`, `scale`, `grasp_key`, `grasp_name`, `verdict`, `flags`, `icl_admissible`, `slots`,
  `subtasks`, `visible_pixels`, `reference_pixels`, `visible_fraction`, `in_frame`, `root_depth_m`, `seconds`).
* `README.md` — this file. The run logs (`survey*.log`) stay local (Isaac console noise).

Re-run: `OMNIGIBSON_HEADLESS=1 python scripts/held_visibility/survey.py --out runlogs/held_visibility_20260915/visibility.json`
(`--resume` to continue, `--models cat/model,...` to restrict, `--redo-errors` to re-run models whose rows all failed).
Every per-model record carries the spawn scale (§3.10: a scaled variant is a different key).
