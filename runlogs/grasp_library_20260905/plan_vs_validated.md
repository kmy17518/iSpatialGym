# Curated category plan vs. validated grasp library (2026-09-05)

Scope: the 91 graspable categories of `curated_category_plan.md` (articulation excluded), capped at 12 ready models per
category -> 582 models. Each model went through candidate generation (task-specific constraints), assisted-grasp (AG)
validation in OmniGibson (up to 12 candidates tested, stop at 5 valid) and a visual review of every valid grasp by 17
reviewer agents. Definitions used below:

* **AG-valid model**: at least one candidate attached under assisted grasping and stayed attached under gravity.
* **clean grasp (lenient)**: AG-valid and not flagged by the reviewer, ignoring the `one_finger` flag (an assisted-grasp
  closing artefact, see failure mode F5).
* **clean grasp (strict)**: AG-valid and carrying no reviewer flag at all.
* A category is called **ok** with >= 5 models having >= 1 clean grasp, **thin** with 3-4, **weak** with 1-2, **drop** with 0.

> **Round 2 (same day, later):** thin objects were re-tested at >= 2 cm thickness and the flower / lever-arm / knife constraint
> changes were re-run. Sections 1-4 are the round-1 numbers kept for reference; section 5 has the updated tables and the
> working scales. Net effect: 510 / 582 models with >= 1 clean grasp (was 479), 23 categories with >= 9 clean models (was 21).

## 1. Headline: plan vs. what survived

| task | planned categories | planned models (cap 12) | models AG-valid | models with >= 1 clean grasp (lenient) | with >= 3 clean | with 5 clean | strict >= 1 | categories ok / thin / weak / drop |
|---|---|---|---|---|---|---|---|---|
| object scaling | 20 | 162 | 161 | 156 | 150 | 144 | 146 | 17 / 3 / 0 / 0 |
| mental rotation | 19 | 123 | 119 | 116 | 107 | 98 | 112 | 12 / 4 / 3 / 0 |
| disembedding | 19 | 131 | 122 | 119 | 119 | 115 | 90 | 8 / 10 / 1 / 0 |
| attachment | 9 | 42 | 22 | 20 | 19 | 19 | 20 | 1 / 4 / 1 / 3 |
| IORS (16 categories) | 16 | 124 | 83 | 68 | 59 | 48 | 62 | 9 / 0 / 3 / 4 |
| **total** | **83 (91 slots)** | **582** | **507 (87%)** | **479 (82%)** | **454 (78%)** | **424 (73%)** | **430 (74%)** | 47 / 21 / 8 / 7 |

Categories that met the original ">= 9 ready models" bar *and* still have >= 9 clean models after validation: 21
(object scaling 8, mental rotation 5, disembedding 6, IORS 2, attachment 0). The plan's weakness was never the grasp
pipeline but the asset pool: most categories entered with 3-8 ready models, so any loss is visible immediately.

## 2. Per-category outcome

Columns: ready = models in the plan (pre-cap), tested = models run (cap 12), AG = AG-valid, L1/L3/L5 = models with >= 1 / >= 3 / 5
clean grasps (lenient), S1 = models with >= 1 strictly clean grasp, flags = models carrying each reviewer flag.

### object scaling (GraspGen, floating)
| category | ready | tested | AG | L1 | L3 | L5 | S1 | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|
| bottle_of_pop | 37 | 12 | 12 | 12 | 12 | 12 | 12 | ok | |
| bottle_of_wine | 27 | 12 | 12 | 12 | 12 | 12 | 12 | ok | |
| bottle_of_beer | 14 | 12 | 12 | 12 | 12 | 12 | 10 | ok | one_finger 7 |
| wineglass | 14 | 12 | 12 | 11 | 11 | 11 | 9 | ok | one_finger 3, fragile 1 (rim pinch) |
| lemon | 13 | 12 | 12 | 12 | 12 | 12 | 12 | ok | |
| bottle_of_water | 12 | 12 | 12 | 12 | 10 | 10 | 12 | ok | end_grasp 1 (cap pinch) |
| beaker | 10 | 10 | 10 | 9 | 9 | 9 | 9 | ok | gap 1 (hemispherical bowl: filled hull) |
| can_of_soda | 9 | 9 | 9 | 9 | 9 | 9 | 9 | ok | |
| ice_cube | 8 | 8 | 8 | 8 | 8 | 6 | 8 | ok | |
| bottle_of_medicine | 8 | 8 | 8 | 8 | 8 | 8 | 7 | ok | one_finger 4 |
| apple | 8 | 8 | 7 | 7 | 7 | 6 | 7 | ok | 1 model without candidates |
| canned_food | 8 | 8 | 8 | 6 | 5 | 4 | 6 | ok | end_grasp 3 (tapered rim / lid pinches) |
| jar_of_spaghetti_sauce | 6 | 6 | 6 | 6 | 5 | 3 | 6 | ok | end_grasp 3 (bottom-end pinches) |
| beeswax_candle | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok | |
| bottle_of_essential_oil | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok | |
| bottle_of_catsup | 5 | 5 | 5 | 5 | 5 | 5 | 2 | ok | one_finger 4 |
| potato | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok | |
| bell_pepper | 5 | 5 | 5 | 4 | 3 | 3 | 4 | thin | end_grasp 2, fragile 1 (stem pinches) |
| bottle_of_black_pepper | 4 | 4 | 4 | 4 | 4 | 4 | 2 | thin | one_finger 3 |
| soda_cup | 4 | 4 | 4 | 4 | 3 | 3 | 4 | thin | one_finger 2 |

### mental rotation (GraspGen, floating)
| category | ready | tested | AG | L1 | L3 | L5 | S1 | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|
| toy_figure | 35 | 12 | 12 | 11 | 11 | 10 | 10 | ok | one_finger 3, end_grasp 1 (ring stacker held by the top figure), penetration 1 (6 mm tail fin) |
| walker | 12 | 12 | 12 | 12 | 12 | 12 | 9 | ok | one_finger 7 |
| coffee_cup | 11 | 11 | 11 | 11 | 11 | 11 | 11 | ok | |
| mug | 10 | 10 | 10 | 10 | 10 | 9 | 10 | ok | gap 1 |
| statue | 10 | 10 | 8 | 8 | 8 | 7 | 8 | ok | 2 models without candidates (23 cm bodies) |
| spatula | 9 | 9 | 9 | 9 | 9 | 7 | 9 | ok | gap 2 |
| teaspoon | 9 | 9 | 8 | 8 | 4 | 4 | 8 | ok | 1 model without candidates |
| mouse | 7 | 7 | 7 | 7 | 6 | 5 | 7 | ok | end_grasp 2 (thin-shell tip) |
| teacup | 6 | 6 | 6 | 6 | 6 | 6 | 6 | ok | |
| soup_ladle | 5 | 5 | 5 | 5 | 5 | 3 | 5 | ok | gap 2 |
| hook | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok | unsure 3 (3 cm chrome disc, contact not readable) |
| crab | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok | |
| carving_knife | 5 | 5 | 5 | 3 | 1 | 1 | 3 | thin | end_grasp 2 models / 10 grasps (blade-tip pinches, handle 12-15 cm away) |
| tablefork | 5 | 5 | 4 | 4 | 2 | 1 | 4 | thin | gap 3, penetration 1 (tines; coarse hull) |
| cellulose_tape_dispenser | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin | |
| tongs | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin | |
| vegetable_peeler | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak (pool) | |
| hammer | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak (pool) | |
| banana | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak (pool) | |

### disembedding (sampler, tabletop)
| category | ready | tested | AG | L1 | L3 | L5 | S1 | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|
| toy_dice | 20 | 12 | 12 | 12 | 12 | 12 | 12 | ok | |
| pack_of_cigarettes | 16 | 12 | 12 | 12 | 12 | 11 | 11 | ok | one_finger 6, gap 1 |
| video_game | 12 | 12 | 12 | 12 | 12 | 12 | 10 | ok | one_finger 8 |
| box_of_cereal | 14 | 12 | 12 | 12 | 12 | 12 | 4 | ok | one_finger 11 (stood on a side face) |
| box_of_sanitary_napkins | 9 | 9 | 9 | 9 | 9 | 9 | 9 | ok | one_finger 2 |
| box_of_baking_mix | 8 | 8 | 8 | 8 | 8 | 8 | 0 | ok | one_finger 8 |
| box_of_cookies | 14 | 12 | 11 | 10 | 10 | 10 | 5 | ok | one_finger 7, task_violation 1 |
| notebook | 34 | 12 | 6 | 6 | 6 | 6 | 6 | ok | 6 models without candidates (>= 9 cm wide) |
| bar_soap | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin | |
| box_of_shampoo | 4 | 4 | 4 | 4 | 4 | 4 | 0 | thin | one_finger 4 |
| box_of_cane_sugar | 4 | 4 | 4 | 4 | 4 | 4 | 3 | thin | |
| box_of_tissues | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin | |
| carton_of_milk | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin | |
| chili | 4 | 4 | 4 | 4 | 4 | 3 | 4 | thin | end_grasp 1 |
| pencil_case | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin | |
| cookie_cutter | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin | |
| cellulose_tape | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin | |
| tortilla_chips | 6 | 6 | 5 | 3 | 3 | 1 | 3 | thin | fragile 1, end_grasp 1 |
| box_of_yogurt | 3 | 3 | 2 | 2 | 2 | 2 | 2 | weak | 1 model without candidates |

### configuration matching - attachment (sampler, attachment_free)
| category | ready | tested | AG | L1 | L3 | L5 | S1 | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|
| lid | 48 | 12 | 6 | 5 | 5 | 5 | 5 | ok | 5 models without candidates (>= 9 cm, no knob); one_finger 3, task_violation 1 (rim-edge pinch) |
| cap | 28 | 12 | 4 | 4 | 4 | 4 | 4 | thin | 8 models slip in AG (6-14 mm thick discs) |
| dip_candle | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin | |
| cork | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin | |
| light_bulb | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin | |
| webcam | 1 | 1 | 1 | 1 | 0 | 0 | 1 | weak | gap 1 (hull larger than the ball) |
| wreath | 1 | 1 | 1 | 0 | 0 | 0 | 0 | drop | fragile (foliage pinch) |
| license_plate | 5 | 5 | 0 | 0 | 0 | 0 | 0 | drop | 30 x 15 x 0.4 cm lying flat: no candidates |
| dartboard | 1 | 1 | 0 | 0 | 0 | 0 | 0 | drop | 45 cm disc: no candidates |

### IORS (sampler, tabletop / insert / flower)
| relation | category | ready | tested | AG | L1 | L3 | L5 | S1 | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| ontop | pen | 6 | 6 | 6 | 6 | 6 | 6 | 6 | ok | |
| ontop | shampoo_dispenser | 7 | 7 | 7 | 7 | 7 | 7 | 5 | ok | one_finger 7 |
| ontop | cherry_tomato | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok | |
| ontop | pickle | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok | |
| under | french_fries | 11 | 11 | 1 | 0 | 0 | 0 | 0 | drop | 6-13 mm: slips before both pads register |
| under | beer_glass | 3 | 3 | 3 | 2 | 2 | 2 | 0 | weak | fragile (rim pinches), one_finger 2 |
| under | sandal | 9 | 9 | 9 | 9 | 8 | 7 | 7 | ok | one_finger 5, gap 2 (strap hulls) |
| under | briefcase | 7 | 7 | 4 | 2 | 1 | 1 | 2 | weak | gap 3, end_grasp 2 (top-edge pinches, hull larger than visual) |
| inside | garlic_clove | 10 | 10 | 1 | 0 | 0 | 0 | 0 | drop | 0.5-1.3 cm: slips |
| inside | sunflower | 18 | 12 | 12 | 7 | 4 | 0 | 7 | ok (L1) | fragile 12 (petal pinches), one_finger 8 |
| inside | quartz | 6 | 6 | 6 | 2 | 2 | 1 | 2 | weak | end_grasp 5 (tip pinches on a 2 x 8 cm crystal) |
| inside | hard_candy | 5 | 5 | 5 | 5 | 4 | 3 | 5 | ok | fragile 2 (wrapper twist) |
| nextto | marker | 20 | 12 | 12 | 12 | 9 | 5 | 12 | ok | end_grasp 4 (nib/cap), gap 4 |
| nextto | pencil | 69 | 12 | 0 | 0 | 0 | 0 | 0 | drop | 4-7 mm: pads close fully, no double contact |
| nextto | cinnamon_stick | 8 | 8 | 1 | 0 | 0 | 0 | 0 | drop | 8-11 mm |
| nextto | parsnip | 6 | 6 | 6 | 6 | 6 | 6 | 6 | ok | |

## 3. Failure modes

Counts refer to models unless stated. Stage 1 = candidate generation (geometry), stage 2 = assisted-grasp test in
simulation, stage 3 = visual review (2457 grasp rows reviewed, 573 flagged = 23%, of which 396 carry only `one_finger`).

**F1. Too wide to pinch off a table (stage 1, 18 models).** notebook 6, lid 5, box_of_cookies 1, box_of_yogurt 1, statue 2,
apple 1, teaspoon 1, plus F2 below. The R1Pro opens 97 mm; with pad margins the usable pinch width is ~89 mm, so a flat
object lying on its largest face must be < 9 cm across its second dimension or must have a protrusion (knob) to pinch.
Standing boxes on a side face rescued 25 models (cereal, baking mix, cane sugar, shampoo, tissues) but not books, wide lids
or the 9.4 cm cookie box. Fix: prefer models by width when choosing the 12 per category (the plan counted "pinch by thickness",
which only works for floating objects), or accept that these categories keep 5-6 models.

**F2. Flat sheets and discs (stage 1, 7 models).** license_plate 5 (30 x 15 x 0.4 cm), dartboard (45 cm disc), briefcase 2.
No top-down pinch exists; they would have to start leaning against something. Drop from attachment or hand them over.

**F3. Sub-centimetre objects slip under assisted grasping (stage 2, 38 models).** pencil 12/12, french_fries 10/11,
garlic_clove 9/10, cinnamon_stick 7/8. The candidates were geometrically fine (all had 12), but the pads closed completely
without both fingers registering contact for the 0.3 s AG window. Below ~1 cm the R1Pro fingers cannot hold an object under
AG. Replace these four categories (marker, pen, chili, hard_candy, cherry_tomato and parsnip all worked at 1.3-3.5 cm).

**F4. Thin discs slip (stage 2, 9 models).** cap 8/12 (6-14 mm thick, 7.5 cm wide), 1 lid, 1 tortilla chip. Related to F3:
the pinch is across the disc edge with only the fingertip. Keep the 4 caps that hold (thicker screw caps) or grasp caps
standing on edge.

**F5. `one_finger` closure pattern (stage 3, 117 models, 430 grasp rows, 396 as the only flag).** One finger stays at the
open limit (50 mm) while the other closes to the centre, with the object pressed against the open pad. Traced to the
assisted-grasp mechanics, not to the grasp pose: the two finger joints are position-controlled independently, the first pad
to touch pushes the free object into the other pad, the AG fixed joint attaches the object to the eef link, and the closing
finger continues to its target (finger 2 ends at 0 mm in 46% of valid grasps). Without an object both fingers close
symmetrically (checked). Demos will show exactly this behaviour, so it is not a library defect; it does make the closed
gripper look off-centre in renders. Mitigation if wanted: freeze the finger targets at the AG attach step.

**F6. End / tip grasps (stage 3, 32 models, 83 rows).** quartz 17 rows (tip pinches on a 2 x 8 cm crystal), carving_knife 10
(blade tip with the handle 12-15 cm away - the H mode trusted GraspGen confidence, which favours the thin blade),
bell_pepper 9 (stem), briefcase 6, jar_of_spaghetti_sauce 6 and canned_food 5 (bottom/lid rims - the confidence gate kept
end pinches the CoM term could not outvote), toy_figure 5, marker 5 (nib/cap), bottle_of_water 3 (cap). Fix: hard cap on
the CoM lever arm (e.g. <= 40% of the object length) for non-handle objects, and for knives constrain the contact to the
handle half explicitly instead of relying on confidence.

**F7. Gap between pad and visual mesh (stage 3, 21 models, 37 rows).** briefcase 7, beaker 5 (hemispherical bowl exported
as a filled hull: the inner pad stops 13 mm from the 4 mm wall), tablefork 5 (tines), marker 5, sandal 4 (straps), webcam 3,
soup_ladle 3, spatula 2. Cause: convex-decomposition collision hulls larger than the visual geometry; the fingers stop on
the hull and appear to float. Only fixable per asset (better collision meshes) or by excluding those models.

**F8. Fragile pinches (stage 3, 19 models, 62 rows).** sunflower 42 rows (all 12 models: the `flower_stem_top` preference
window of 12-35% from the head end landed on the flower head, so the grasps pinch petals/disc rather than the stem just
below the head), beer_glass 5 and wineglass 5 (rim pinches), bell_pepper 4 (stem), hard_candy 4 (wrapper twist),
tortilla_chips 1, wreath 1 (foliage). Fix for flowers: require the pinch width to be a stem width (<= 15 mm) and move the
window to 30-45% from the head end.

**F9. Penetration (2 rows) and task violations (6 rows).** tablefork and toy_figure tail fin (hull smaller than visual);
box_of_cookies 5 rows and one lid where the pinch covered the placed / mating face. Rare; handle per model.

## 4. What this means for the plan

* Drop or replace now: pencil, french_fries, garlic_clove, cinnamon_stick (F3), license_plate, dartboard, wreath (F2/F8).
  The working categories set the bar: pen (11 mm) held in 6/6, marker (14 mm) 12/12, chili (13 mm) 4/4, so ~11 mm is the
  practical minimum thickness. Replacements that clear it (ready models in brackets): nextto -> pipe_cleaner 29 mm (5),
  lipstick 20 mm (3), carrot 30 mm (3), spinach 26 mm (6); not ruler 2 mm, colored_pencil 5 mm, chopstick 5 mm,
  table_knife 6 mm, straw 6 mm, crayon 10 mm (borderline). inside -> chestnut 16 mm (4), date 18 mm (3), half_lime 38 mm (4),
  dreidel 15 mm (3); not sliced_lemon 9 mm or star_anise 5 mm. under -> tablespoon 18 mm (4) on a platter. attachment ->
  keep lid (5 usable), cap (4), cork, dip_candle, light_bulb, webcam; nothing in the pool replaces the wall decorations.
* Thin categories (3-4 usable models) are a pool problem, not a grasp problem: 21 categories, mostly disembedding boxes and
  the kitchen tools in mental rotation.
* Constraint changes worth a rerun: flower window (F8), lever-arm cap for jars/cans/quartz (F6), knives -> handle half (F6);
  each is a parameter change in `gen_candidates.py` followed by re-validating only the affected categories (~100 models,
  ~15 min with 3 shards).
* Model selection for tabletop categories should rank by second-smallest extent (width), not thickness (F1).

## 5. Round 2: thickness scaling and constraint reruns

### 5.1 Scaling thin objects to >= 2 cm

Rule (`scripts/build_scaled_jobs.py`, scales are per-axis in the object's canonical frame and passed to
`DatasetObject(scale=...)`, so the collision mesh, visual mesh and AG test all see the same enlarged object):

* elongated objects (longest / thinnest extent > 3): scale only the two cross-section axes up to 20 mm and keep the length.
  A 12.4 cm pencil becomes a 2 x 2 cm chunky pencil rather than a 50 cm giant one; a fry becomes a steak fry; a cinnamon
  stick keeps its 10-16 cm length;
* compact objects (garlic cloves, 0.5-0.9 cm thick, 1.3-2.5 cm long): uniform scale so the thinnest extent is 20 mm. The
  shape is kept; the cloves end up 4.4-5.4 cm long, large but still recognisable;
* flat discs (the 8 caps that slipped): thickness axis only, 13 -> 20 mm.

49 scaled variants were exported, run through candidate generation, AG validation (batch of 108 models with the constraint
reruns below, 3 shards, 22 min) and the same reviewer process (`review/wave3_*`, `wave4_*`). Outcome per model
(orig AG = valid grasps at scale 1; AG / clean = valid / reviewer-clean grasps at the new scale; works = >= 3 clean grasps):

| category | model | rule | scale (x, y, z) | bbox mm before -> after | orig AG | AG | clean | flags | verdict |
|---|---|---|---|---|---|---|---|---|---|
| pencil | lykqoh | cross_section | [1.0, 3.596, 5.119] | [124, 6, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| pencil | rzscao | cross_section | [1.0, 3.867, 5.113] | [124, 5, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| pencil | mvmxfo | cross_section | [1.0, 3.327, 5.106] | [124, 6, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| pencil | gbzzcg | cross_section | [1.0, 3.901, 5.106] | [124, 5, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| pencil | jntybl | cross_section | [1.0, 3.96, 5.098] | [124, 5, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| pencil | vpzhth | cross_section | [1.0, 3.361, 5.096] | [124, 6, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| pencil | lruiot | cross_section | [1.0, 3.603, 5.085] | [124, 6, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| pencil | heqkmr | cross_section | [1.0, 3.345, 5.081] | [124, 6, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| pencil | fnlbpu | cross_section | [1.0, 4.217, 5.069] | [124, 5, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| pencil | vedvit | cross_section | [1.0, 3.451, 5.058] | [124, 6, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| pencil | vvofwr | cross_section | [1.0, 3.473, 5.014] | [124, 6, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| pencil | hmawzy | cross_section | [1.0, 3.231, 4.998] | [124, 6, 4] -> [124, 20, 20] | 0 | 5 | 5 |  | works |
| garlic_clove | zpzrac | uniform | [3.79, 3.79, 3.79] | [9, 13, 5] -> [35, 51, 20] | 0 | 5 | 5 |  | works |
| garlic_clove | qaqrqd | uniform | [3.746, 3.746, 3.746] | [9, 13, 5] -> [33, 50, 20] | 0 | 5 | 5 |  | works |
| garlic_clove | xotobt | uniform | [3.724, 3.724, 3.724] | [9, 13, 5] -> [33, 50, 20] | 0 | 5 | 5 | one_finger | works |
| garlic_clove | hpzvmi | uniform | [3.61, 3.61, 3.61] | [9, 13, 6] -> [32, 48, 20] | 0 | 5 | 5 |  | works |
| garlic_clove | saombi | cross_section | [1.268, 1.0, 3.446] | [16, 27, 6] -> [20, 27, 20] | 4 | 5 | 5 |  | works |
| garlic_clove | zpybkr | uniform | [3.33, 3.33, 3.33] | [8, 13, 6] -> [27, 45, 20] | 0 | 5 | 5 |  | works |
| garlic_clove | faeyay | uniform | [3.329, 3.329, 3.329] | [9, 13, 6] -> [29, 44, 20] | 0 | 5 | 5 |  | works |
| garlic_clove | flcbrn | uniform | [3.276, 3.276, 3.276] | [9, 13, 6] -> [29, 44, 20] | 0 | 4 | 4 |  | works |
| garlic_clove | wyyhmi | uniform | [2.518, 2.518, 2.518] | [10, 20, 8] -> [25, 49, 20] | 0 | 5 | 5 |  | works |
| garlic_clove | jwhnfc | uniform | [2.196, 2.196, 2.196] | [15, 25, 9] -> [33, 54, 20] | 0 | 2 | 1 | penetration | partial |
| french_fries | jvhnyc | cross_section | [1.0, 1.545, 3.364] | [69, 13, 6] -> [69, 20, 20] | 0 | 0 | 0 |  | fails |
| french_fries | jhskfy | cross_section | [1.0, 2.055, 2.56] | [83, 10, 8] -> [83, 20, 20] | 0 | 1 | 0 | penetration | AG only |
| french_fries | nfgqwu | cross_section | [1.0, 1.891, 2.458] | [82, 11, 8] -> [82, 20, 20] | 0 | 3 | 0 | end_grasp | AG only |
| french_fries | uqsuef | cross_section | [1.0, 2.155, 2.458] | [82, 9, 8] -> [82, 20, 20] | 0 | 1 | 0 | penetration | AG only |
| french_fries | xzsoub | cross_section | [1.0, 2.176, 2.414] | [59, 9, 8] -> [59, 20, 20] | 0 | 3 | 0 | end_grasp, penetration | AG only |
| french_fries | rremmp | cross_section | [1.0, 1.635, 2.395] | [75, 12, 8] -> [76, 20, 20] | 0 | 0 | 0 |  | fails |
| french_fries | oeterg | cross_section | [1.0, 1.871, 2.385] | [49, 11, 8] -> [49, 20, 20] | 0 | 5 | 1 | end_grasp, one_finger | partial |
| french_fries | kxpfcj | cross_section | [1.0, 1.646, 2.306] | [67, 12, 9] -> [67, 20, 20] | 2 | 5 | 2 | end_grasp | partial |
| french_fries | xshwvn | cross_section | [1.0, 2.227, 1.618] | [59, 9, 12] -> [59, 20, 20] | 0 | 5 | 3 | end_grasp, one_finger | works |
| french_fries | kjqhuo | cross_section | [1.0, 1.446, 2.117] | [79, 14, 9] -> [79, 20, 20] | 0 | 3 | 0 | penetration | AG only |
| french_fries | xlkkxg | cross_section | [1.0, 1.657, 1.991] | [65, 12, 10] -> [66, 20, 20] | 0 | 1 | 0 | penetration | AG only |
| cinnamon_stick | hjhcpm | cross_section | [1.0, 1.0, 2.508] | [21, 98, 8] -> [21, 98, 20] | 0 | 2 | 0 | penetration | AG only |
| cinnamon_stick | cdkjfo | cross_section | [1.647, 1.0, 2.423] | [12, 113, 8] -> [20, 113, 20] | 0 | 2 | 0 | penetration | AG only |
| cinnamon_stick | qxpzdm | cross_section | [1.0, 1.0, 1.968] | [21, 161, 10] -> [21, 161, 20] | 0 | 5 | 5 |  | works |
| cinnamon_stick | qsqvgk | cross_section | [1.056, 1.0, 1.876] | [19, 140, 11] -> [20, 140, 20] | 0 | 3 | 0 | penetration | AG only |
| cinnamon_stick | kdaxdy | cross_section | [1.0, 1.0, 1.87] | [21, 131, 11] -> [22, 131, 20] | 0 | 2 | 2 |  | partial |
| cinnamon_stick | smfuqz | cross_section | [1.207, 1.0, 1.8] | [17, 137, 11] -> [20, 137, 20] | 0 | 3 | 0 | penetration | AG only |
| cinnamon_stick | bmbjdf | cross_section | [1.628, 1.0, 1.462] | [12, 92, 14] -> [20, 92, 20] | 0 | 1 | 0 | end_grasp, penetration | AG only |
| cinnamon_stick | qmlyim | cross_section | [1.054, 1.0, 1.326] | [19, 143, 15] -> [20, 143, 20] | 1 | 2 | 1 | penetration | partial |
| cap | dwpcld | thickness_only | [1.0, 1.0, 1.529] | [76, 76, 13] -> [76, 76, 20] | 0 | 0 | 0 |  | fails |
| cap | qcfsnv | thickness_only | [1.0, 1.0, 1.529] | [76, 76, 13] -> [76, 76, 20] | 0 | 0 | 0 |  | fails |
| cap | jybxvq | thickness_only | [1.0, 1.0, 1.529] | [76, 76, 13] -> [76, 76, 20] | 0 | 0 | 0 |  | fails |
| cap | ytjxqn | thickness_only | [1.0, 1.0, 1.529] | [76, 76, 13] -> [76, 76, 20] | 0 | 0 | 0 |  | fails |
| cap | ciopwh | thickness_only | [1.0, 1.0, 1.529] | [76, 76, 13] -> [76, 76, 20] | 0 | 0 | 0 |  | fails |
| cap | clhquh | thickness_only | [1.0, 1.0, 1.529] | [76, 76, 13] -> [76, 76, 20] | 0 | 2 | 2 | one_finger | partial |
| cap | wwbayp | thickness_only | [1.0, 1.0, 1.529] | [76, 76, 13] -> [76, 76, 20] | 0 | 0 | 0 |  | fails |
| cap | yqvild | thickness_only | [1.0, 1.0, 1.529] | [76, 76, 13] -> [76, 76, 20] | 0 | 1 | 1 | one_finger | partial |

Summary by category:

| category | models | rule | AG >= 1 | clean >= 1 | clean >= 3 | 5 clean | reading |
|---|---|---|---|---|---|---|---|
| pencil | 12 | cross-section x3.2-5.1 -> 20 x 20 mm | 12 | 12 | 12 | 12 | works for every model, no flags; keep pencil at this scale |
| garlic_clove | 10 | uniform x2.2-3.8 | 10 | 10 | 9 | 8 | works; one model (jwhnfc, the least scaled) has hull penetration |
| french_fries | 11 | cross-section x1.4-3.4 | 9 | 3 | 1 | 0 | does not work: fries are curved strips, so axis scaling does not make the pinch 2 cm thick (pads still close to 2-7 mm) and the hull differs from the visual (penetration 5, end_grasp 5) |
| cinnamon_stick | 8 | thickness axis x1.3-2.5 | 8 | 3 | 1 | 1 | does not work: rolled-bark shells, the pads close on the curved wall (penetration 6) |
| cap | 8 | thickness x1.53 -> 20 mm | 2 | 2 | 0 | 0 | does not work: a 7.6 cm disc lying flat still slips (pinch across the diameter is at the fingertips) |

So the 2 cm rule fixes prismatic shapes (pencil, cloves) and does not fix curved / hollow / disc shapes. For the failing
ones the replacement pool is small (unused categories with >= 5 ready single-link models, >= 15 mm thick, <= 85 mm wide,
3-25 cm long): water_glass (20), pack_of_kielbasa (16), bottle_of_milk (6), spinach (6), soap_dispenser (6),
pipe_cleaner (5). Suggested swaps: `under: platter / french_fries` -> `pack_of_kielbasa` (or spinach); `nextto:
marker / cinnamon_stick` -> `pipe_cleaner` (or soap_dispenser); caps: keep the 4 unscaled caps that hold plus the 2 scaled
ones (6 models).

### 5.2 Constraint changes and what they did

Changes in `scripts/gen_candidates.py`, applied to the 15 categories that carried end-grasp / flower flags in round 1
(124 models re-generated, re-validated, re-rendered and re-reviewed; round-1 outputs are kept under `superseded/v1`):

1. **Lever-arm cap** (`LEVER_CAP = 0.40`): the CoM offset from the pinch, measured along the pad-width axis, must be
   <= 40% of the longest extent. Applied to the tabletop sampler and to the CoM-ranked GraspGen modes (not to the H mode).
2. **Knives -> `handle_half`**: the thick end of the long axis is detected as the handle (thickness profile), contacts are
   restricted to the handle half, ranked confidence-first with a preference for the middle of the handle.
3. **Flowers**: pinch width <= 30 mm (stem or the receptacle behind the head, never petals), window 20-70% of the
   head-to-stem-end length, preference centre at 35%.
4. Two pipeline bugs found while doing this, both affecting only thin tabletop objects: the stored width / antipodal
   features and the lever arm were computed at the nominal pad centre, which sits above a thin object after the
   table-clearance retreat (features came out `inf`, the lever arm was inflated by the retreat and killed 94% of chili
   candidates). Both now use the actual contact height. And a **closing-sweep check** (`pad_sweep`): rays from the open pad
   faces inward over the pad footprint; a candidate is dropped if the pads would meet another feature (flower head, jar rim)
   before the intended contact. This is what finally turned the sunflower grasps into stem pinches.

Effect (models with >= 1 / >= 3 / 5 clean grasps; round 1 -> round 2; flags are reviewer flags on the final grasps):

| category | models | round 1 L1/L3/L5 | round 2 L1/L3/L5 | round-2 flags | reading |
|---|---|---|---|---|---|
| carving_knife | 5 | 3 / 1 / 1 | 5 / 5 / 5 | none (25 / 25 rows clean) | handle_half works: all grasps on the handle next to the bolster |
| quartz | 6 | 2 / 2 / 1 | 5 / 5 / 3 | end_grasp 1 | lever cap removed the tip pinches |
| bell_pepper | 5 | 4 / 3 / 3 | 5 / 5 / 4 | end_grasp 1 | stem pinches gone |
| briefcase | 7 | 2 / 1 / 1 | 3 / 3 / 3 | none | 4 models still have no valid grasp (F1/F2) |
| jar_of_spaghetti_sauce | 6 | 6 / 5 / 3 | 6 / 6 / 4 | end_grasp 2 | most rim pinches gone |
| canned_food | 8 | 6 / 5 / 4 | 6 / 5 / 4 | end_grasp 2 | one model lost AG; 2 still pinch a rim |
| mouse | 7 | 7 / 6 / 5 | 7 / 7 / 6 | end_grasp 1 | |
| bottle_of_water | 12 | 12 / 10 / 10 | 12 / 10 / 10 | none | unchanged |
| toy_figure | 12 | 11 / 11 / 10 | 11 / 10 / 9 | end_grasp 3 | figurines are pinched at limbs/heads either way |
| marker | 12 | 12 / 9 / 5 | 12 / 10 / 5 | end_grasp 4, penetration 3 | unchanged |
| chili | 4 | 4 / 4 / 3 | 4 / 4 / 4 | none | unchanged after the contact-height fix (the first rerun had 0 clean) |
| tortilla_chips | 6 | 3 / 3 / 1 | 2 / 2 / 1 | gap 1 | |
| lid | 12 | 5 / 5 / 5 | 4 / 4 / 3 | gap 2, one_finger 2 | one model lost AG |
| statue | 10 | 8 / 8 / 7 | 4 / 4 / 4 | end_grasp 4, fragile 2 | grasps barely changed; the round-2 reviewer flagged head/arm pinches the round-1 reviewer accepted |
| sunflower | 12 | 7 / 4 / 0 (petal pinches) | 7 / 4 / 2 (stem pinches) | penetration 1 | the 7 usable models are now real stem grasps (pads close to 9-11 mm with the head hanging free); 3 models have no stem candidate that survives the sweep check and 2 slip on the 9 mm stem |

Reviewer variance is visible in this table (statue, bottle_of_water): each wave is a different reviewer, and end-grasp
judgements on figurines and bottle necks differ between them. For the final numbers the latest review of each model is
used (`aggregate.py` now sorts review files so later waves override earlier ones, ignores reviews of grasps that no
longer exist, and the duplicate half-batch reviews of stalled batches live in `review/redundant/` and are not counted).

### 5.3 Updated headline (best variant per model: scaled variant used where it is the better one)

| task | categories | models | AG-valid | >= 1 clean (lenient) | >= 3 clean | 5 clean | strict >= 1 | categories ok / thin / weak / drop |
|---|---|---|---|---|---|---|---|---|
| object scaling | 20 | 162 | 160 | 157 | 153 | 145 | 146 | 18 / 2 / 0 / 0 |
| mental rotation | 19 | 123 | 119 | 114 | 107 | 99 | 111 | 12 / 4 / 3 / 0 |
| disembedding | 19 | 131 | 120 | 118 | 118 | 116 | 89 | 8 / 9 / 2 / 0 |
| attachment | 9 | 42 | 23 | 21 | 18 | 17 | 20 | 1 / 4 / 1 / 3 |
| IORS | 16 | 124 | 113 | 100 | 90 | 78 | 94 | 12 / 3 / 1 / 0 |
| **total** | **83** | **582** | **535 (92%)** | **510 (88%)** | **486 (84%)** | **455 (78%)** | **460 (79%)** | 51 / 22 / 7 / 3 |

Round 1 -> round 2: AG-valid 507 -> 535, >= 1 clean 479 -> 510, 5 clean 424 -> 455; categories with >= 9 clean models
21 -> 23 (pencil and garlic_clove join at the new scale). IORS drops from 4 "drop" categories to 0: pencil and
garlic_clove are rescued by scaling, sunflower by the stem constraint, and french_fries / cinnamon_stick now have 3 clean
models each (weak/thin) but should still be swapped as suggested in 5.1. The 3 remaining drops are all attachment
(license_plate, dartboard, wreath) and are asset-pool problems.

Artifacts: scaled job list `models_scaled.json` (scale vectors per model), `best_per_model.json` (which variant to use per
model, with its scale), strips for the scaled models `strips/<category>__<model>__s.png`, round-1/2/3 outputs of the re-run
models under `superseded/v1..v3`, review files `review/wave3_* .. wave5_*`.
