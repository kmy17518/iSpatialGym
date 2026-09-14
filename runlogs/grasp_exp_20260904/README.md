# Grasp-proposal experiment (2026-09-04): antipodal sampler + filter vs GraspGen

Objects: 10 categories sampled (seed 2026) from the 91 movable, gripper-sized (2nd-smallest extent <= 9.5 cm)
task-relevant models of the 100 BEHAVIOR-2026 challenge task templates
(`datasets/2026-challenge-task-instances/scenes/*/json/*_task_*_0_0_template.json`, `metadata.task.inst_to_name`).

Approach 1 (`scripts/approach1_sampler_filter.py`, ispatialgym env, no OmniGibson):
`isaacsim.replicator.grasping.sampler_utils.sample_antipodal` on the exported collision mesh (4000 candidates x 12
orientations, aperture 8.5 cm) -> approach within 60 deg of vertical -> object width along the closing axis in
[6 mm, 8.5 cm] and both contact normals within 30 deg of the closing axis (antipodal check; without it the wedge-shaped
pie produced 5 "valid" top grasps that all slipped out in simulation) -> retreat along the approach so the fingertips stay
above the support plane while the contact stays on the pads -> R1Pro gripper spheres (open) >= 2 mm from the object
-> rank by clearance, NMS, top-5.

Approach 2 (`scripts/approach2_graspgen.py`, separate `graspgen` conda env, NVlabs/GraspGen Franka-Panda checkpoint,
pure diffusion planner): 2000 surface points -> grasps in the Panda hand frame -> converted to `right_eef_link`
(`approach2_convert.py`: Rz(90 deg) because GraspGen closes along x, ours along y; TCP depth 0.1053 m, pad centre
-0.0119 m). Two selections: raw top-5 by confidence (400 samples -> top 100) and "table" top-5 (2000 -> top 500, approach
<= 60 deg, fingertips above the support plane).

AG check (`scripts/ag_check_render.py`, OmniGibson headless): R1Pro gripper pointing down, object teleported into the open
gripper at the stored relative pose (gravity off), gripper closed for up to 60 control steps, success = OmniGibson
assisted grasp attaches the object and it is still held after 30 steps with gravity on.

| object | A1 gen time | A1 sampled->pass | A1 AG | GraspGen inference (400->100) | raw top-100 approaching from above | A2 raw AG | GraspGen inference (2000->500) | top-500 above+table ok | A2 table AG |
|---|---|---|---|---|---|---|---|---|---|
| brussels_sprouts_hkwyzk | 106 ms | 2028 -> 153 | 5/5 | 461 ms (first call) | 29 | 5/5 | 490 ms | 75 | 5/5 |
| mushroom_zlukhz | 141 ms | 1920 -> 46 | 5/5 | 79 ms | 12 | 5/5 | 236 ms | 9 | 5/5 |
| tube_of_toothpaste_vsclsj | 54 ms | 2052 -> 16 | 5/5 | 78 ms | 20 | 5/5 | 243 ms | 9 | 5/5 |
| vidalia_onion_buyxll | 212 ms | 1872 -> 142 | 5/5 | 79 ms | 14 | 5/5 | 244 ms | 58 | 3/5 |
| broccolini_rlsytp | 149 ms | 1992 -> 57 | 5/5 | 77 ms | 28 | 5/5 | 248 ms | 20 | 5/5 |
| half_apple_pie_grcock | 49 ms | 2088 -> 0 | 0/0 | 78 ms | 0 | 5/5 (side pinches) | 370 ms | 0 | 0/0 |
| steel_wool_egeolq | 260 ms | 2520 -> 107 | 5/5 | 78 ms | 2 | 5/5 | 243 ms | 0 | 0/0 |
| toy_figure_orunry | 74 ms | 2196 -> 5 | 4/4 | 77 ms | 1 | 5/5 | 245 ms | 0 | 0/0 |
| tablefork_flexrc | 74 ms | 1332 -> 28 | 5/5 | 78 ms | 2 | 1/5 | 245 ms | 0 | 0/0 |
| zucchini_udvfaz | 201 ms | 3876 -> 301 | 5/5 | 79 ms | 44 | 5/5 | 242 ms | 70 | 5/5 |
| total | mean 132 ms | | 44/44 | mean 116 ms (+1.2 s model load) | 152/1000 | 46/50 | mean 281 ms | | 28/30 |

AG check cost: 1.63 s per grasp (124 grasps). Files: `strips/<object>.png` (rows: A1, A2 raw, A2 table; green = AG
success), `grasps/*.yaml` (isaac_grasp, eef in object base_link frame), `*_results.json`, `category_table.json`
(per-category ready/rigid/openable/attachment/scene data used for the task-object consultation).

## Follow-up (same day): best-grasp ranking heuristic and floating-object grasps

`scripts/rank_grasps.py` scores every stored candidate with
`5*f_CoM + 1*antipodal + 1*coverage + 0.5*width + 1*clearance + 3*vertical` where `f_CoM = exp(-d_perp / (0.25*object_size))`
(d_perp = distance from the object's centre of mass to the pinch axis through the two contacts), tabletop grasps steeper than
40 deg are rejected, and top-5 are picked with NMS. Sets: `grasps/a1best_*.yaml` (A1 candidates, tabletop) and
`grasps/a2best_*.yaml` (GraspGen candidates for floating-object tasks: no table/approach constraint, gripper-object
collision filter, score + 2*confidence). AG results (`ag_results_rank_v1.json`, `ag_results_rank2.json`, strips in
`strips_ranking/`, rows = A1 clearance-ranked / A1 CoM v1 (tilt<=60, weak verticality) / A1 CoM v2 / GraspGen floating):

| ranking | AG success | median CoM offset of the picked grasps |
|---|---|---|
| A1 ranked by clearance (original) | 44/44 | 26 mm (pendulum grasps at the end of zucchini/broccolini/steel wool: 60-130 mm) |
| A1 ranked by CoM v1 (tilt <= 60 deg, vertical weight 0.5) | 37/44 | ~15 mm; failures were the 52-60 deg approaches it favoured on small objects |
| A1 ranked by CoM v2 (tilt <= 40 deg, size-normalised CoM, vertical weight 3) | 42/44 | 20 mm; 2 failures on broccolini (irregular collision pieces) |
| GraspGen floating, CoM + confidence | 48/48 | 2 mm |

`curated_category_plan.md`: 92 categories (20/19/19/9/9/4x4) hand-curated from the ranked pools in
`task_category_candidates.md`, with ready counts and graspability per category.

## GraspGen scoring variants (floating-object setting), `strips_graspgen_scoring/`

Rows: top-5 by confidence only (46/50 AG) | CoM heuristic + 2*confidence (48/48) | confidence band (max-0.03) then CoM (48/48;
`grasps/a2band_*.yaml`). Confidence is GraspGen's discriminator output (stability under the simulated shake test); within an
object it spans only 0.02-0.06 and is uncorrelated with CoM offset (Spearman -0.26..+0.36). It does encode a functional prior:
on the fork and the steel-wool scrubber the top-confidence grasps are on the handle, while CoM-based rankings grasp the strip /
neck. On produce (onion, zucchini, brussels sprouts) confidence is flat and CoM is what picks a sensible grasp. Recommendation:
choose the mode per category from the BDDL taxonomy (implement/tool/cutlery/utensil/device -> confidence-first, CoM tie-break;
food/produce/plant/plaything/footwear -> confidence gate then CoM), with a manual override list (e.g. steel_wool has a handle
but no taxonomy ancestors).
