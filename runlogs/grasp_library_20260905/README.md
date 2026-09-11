# Grasp library run (2026-09-05): 582 models, 5 validated grasps each

Inputs: `curated_category_plan.md` (91 categories), cap 12 models/category -> `models.json` (582 models; 285 in-hand
models on the GraspGen pipeline, 297 tabletop models on the antipodal-sampler pipeline). Articulation is not included
(per-link handle grasps are a separate job).

## Pipeline (`scripts/`)
1. `export_meshes_batch.py` - collision meshes + metadata (attachment meta-links) from OmniGibson, 2 parallel sessions, 100 s.
2. `graspgen_batch.py` (graspgen env) - GraspGen Franka-Panda diffusion planner, 2000 -> top-500 grasps per model, 247 ms/model.
3. `gen_candidates.py` - task/object-specific constraints + ranking -> top-12 candidates per model (`cands/*.yaml`, isaac_grasp):
   - tabletop (disembedding, IORS ontop/under/nextto, inside): antipodal sampler, approach <= 40 deg, width 3-89 mm, contact
     normals within 30 deg of the pinch axis, fingertips may dip <= 8 mm into the support plane with the contact no further
     out than the fingertip, object-aware retreat for knobs/rims, CoM-v2 ranking; if the canonical pose yields nothing the
     object is stood on a side face (`rest_pose` = x_down/y_down/x_up, stability ratio >= 0.2) - 25 models needed this.
   - attachment_free (lid, cap, cork, dip_candle, light_bulb, wreath, dartboard, license_plate, webcam): all gripper spheres
     >= 1 cm from the attachment meta-link point and the contact not on the mating side; ranking prefers distance from it.
   - flower_stem_top (sunflower -> vase): contact in the top 45% of the stem axis, pinch across the stem, preference for
     just below the head.
   - floating (GraspGen, object scaling / mental rotation): per-category mode H (confidence-first, tools with handles),
     V (vessels), C (confidence gate then CoM).
   Embree returned only the self-hit for rays starting on a face (box meshes gave 0 samples) -> origins offset 0.1 mm.
4. `ag_validate_render.py` - object teleported into the open gripper at the candidate pose, gripper closed, success =
   assisted grasp attaches and holds under gravity; 3 external views + right wrist camera per valid grasp; stop at 5.
   Parallel OmniGibson sessions: 1 process 1.3 s/grasp, 3 concurrent processes 2.7 s/grasp each (1.45x throughput, GPU
   bound). Full run: 3 shards, 84 min wall clock. 3337 candidates tested, 2442 passed (73%).
5. `make_model_strips.py` - one PNG per model (`strips/`), 5 rows x [profile, opposite profile, oblique-below, wrist cam].
6. Review: 17 parallel reviewer agents, rubric in `review_prompt.md`, outputs in `review/*.json`; `aggregate.py` ->
   `aggregate.json`, `aggregate_report.txt`.

## Results
| task | models | AG >= 1 | AG = 5 | lenient clean >= 1 | lenient clean >= 3 | lenient clean = 5 | strict clean >= 1 |
|---|---|---|---|---|---|---|---|
| object_scaling | 162 | 161 | 153 | 156 | 150 | 144 | 146 |
| mental_rotation | 123 | 119 | 111 | 116 | 107 | 98 | 112 |
| disembedding | 131 | 122 | 118 | 119 | 119 | 115 | 90 |
| attachment | 42 | 22 | 20 | 20 | 19 | 19 | 20 |
| IORS (16 categories) | 124 | 83 | 72 | 68 | 59 | 48 | 62 |
| **all** | **582** | **507** | **474** | **479** | **454** | **424** | **430** |

"strict" removes every reviewer-flagged grasp; "lenient" ignores the `one_finger` flag (117 models), which is an artefact of
assisted grasping in OmniGibson: the fingers are position-controlled independently, the first pad to touch pushes the
object into the other pad, the AG fixed joint attaches to the eef link, and the closing finger then continues to its
target (finger 2 ends at 0 mm in 46% of valid grasps while finger 1 stays at 30-50 mm). Without an object the fingers
close symmetrically (checked). The same thing will happen in demos, so it is not a property of the grasp pose.

Zero valid grasps (75 models): pencil 12/12, french_fries 10/11, garlic_clove 9/10, cinnamon_stick 7/8 (objects <= 1 cm:
the pads close fully without both touching), cap 8/12 (6-14 mm discs 7.5 cm wide: geometric candidates exist but slip),
lid 6/12 and notebook 6/12 (>= 9 cm wide, no knob), license_plate 5/5, dartboard 1/1, briefcase 3/7 (flat/wide, cannot be
pinched off a table), statue 2/10, plus singletons.

Other real flags: `end_grasp` 32 models (quartz 5, marker 4, canned_food 3, jar_of_spaghetti_sauce 3, ...), `gap` 21
(marker 4, tablefork 3, briefcase 3: coarse collision hulls), `fragile` 19 (sunflower 12: petal pinches; hard_candy,
tortilla_chips), `penetration` 2, `task_violation` 2 (one lid, one box).

Per-category table: `aggregate_report.txt`. Per-grasp flags and notes: `review/*.json`. Grasps: `cands/<category>__<model>.yaml`
(all 12 ranked candidates, isaac_grasp, eef in the object base_link frame; `rest_quat_wxyz` when the object must stand on a
side face) with the AG outcome per candidate in `results/<key>.json`.

## Round 2 (same day): thickness scaling + constraint reruns
* `build_scaled_jobs.py` -> `models_scaled.json`: 49 sub-2 cm models (pencil, french_fries, garlic_clove, cinnamon_stick, 8 caps)
  re-run at a per-axis scale that makes the thinnest extent 20 mm (cross-section only for elongated shapes, uniform for
  cloves, thickness only for caps). Their outputs use the key suffix `__s` (`results/`, `cands/`, `strips/`).
* `gen_candidates.py` changes: lever-arm cap (CoM offset along the pad-width axis <= 40% of the length), `handle_half`
  constraint for knives, flower window 20-70% with pinch width <= 30 mm, features / lever arm at the actual contact height,
  closing-sweep check (`pad_sweep`). `models_rerun.json` = the 124 models of the 15 affected categories; their round-1
  outputs are in `superseded/v1`, intermediate reruns in `superseded/v2`, `superseded/v3`.
* Reviews `review/wave3_*`..`wave5_*`; `aggregate.py` takes the latest review per model. `best_per_model.json` = the
  variant (scaled or not) to use per model with its scale vector.
* Outcome: pencil 12/12 and garlic_clove 10/10 work at the new scale; fries, cinnamon sticks and caps do not (curved /
  hollow / disc shapes). Knife and quartz end grasps are gone, sunflower grasps are stem pinches. Totals: 535 AG-valid,
  510 with >= 1 clean grasp, 455 with 5 clean grasps (of 582). Details in `plan_vs_validated.md`, section 5.

## Selection rounds 2 and 3 (same day, evening)
* `scripts/build_rounds.py` -> `manifest.json` (every (task, category, model) selection with its round: 1 = original plan, 2 = category
  reused from another task, 3 = never-used category, single-model categories allowed) and `jobs_round23.json` (467 new grasp tests:
  198 GraspGen floating, 269 sampler). Round-2 models are chosen least-used-first so no model is selected more than twice
  (669 models once, 245 twice). Grasp tests are skipped only when pipeline + constraint + scale already exist (bell_pepper, potato).
  New result keys carry a method suffix: `__f` floating, `__t` tabletop, `__i` insert, `__fl` flower, `__a` attachment_free,
  `__h` handle_half (`__sf` = scaled + floating).
* `scripts/aggregate_manifest.py` -> `selection_manifest.csv` / `.json` (per selection: round, task, relation, partner, result key,
  AG-valid, clean, flags) and `rounds_summary.md` (per task x round and per category tables). Reviews: `review/wave6_*`.
* Constraint addition: per-category flower window (`FLOWER_WINDOW`, lily 55-85%).

## Test-instance selection (two models per task x category)
* Contract: `select/select_contract.md` (rubric: G grasp quality 0-5, V visual quality 0-5, T task fit 0-3, R robustness 0-2;
  score = 2G + 2V + T + R; pair rule prefers a visually different second model; hard exclusions for no clean grasp / broken asset).
* Inputs: `select/groups/<task>__<category>__r<round>.json` (candidates with strips, prior review notes, bbox, scale);
  `select/assign/sel_XX.json` (29 reviewer assignments of <= 34 strips). Outputs: `select/out/<group>.json` (per-candidate scores
  and notes, picks with reasons, alternate, warnings), aggregated by `scripts/aggregate_selection.py` into
  `select/test_instances.csv` / `.json` / `.md`.

## Articulated objects (two full passes done 2026-09-07)
See `artic/ARTIC_RESUME.md` for the pipeline, results (13 joints on 11 models opened by the robot) and how to rerun. Inventory, handle inference,
renders, reviewer verdicts, scene assignments and the two validation harnesses are in `artic/` and `scripts/artic_*.py`.
