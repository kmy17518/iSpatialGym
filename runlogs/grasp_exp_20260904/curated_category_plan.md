# Curated category plan (2026-09-04)

ready = no unprocessed QA complaint (BEHAVIOR knowledge-base definition); pinch = thinnest extent <= 8.5 cm; width-fit = two extents <= 9.5 cm (top-down body pinch); bbox = median over ready models [m], canonical frame.

## object_scaling (20) - prefer symmetric

| # | category | ready | pinch | width-fit | median bbox | GraspGen mode |
|---|---|---|---|---|---|---|
| 1 | bottle_of_pop | 37 | 34 | 36 | [0.062, 0.062, 0.17] | C |
| 2 | bottle_of_wine | 27 | 23 | 24 | [0.077, 0.078, 0.275] | C |
| 3 | bottle_of_black_pepper | 4 | 4 | 4 | [0.061, 0.06, 0.074] | C |
| 4 | bottle_of_beer | 14 | 7 | 11 | [0.079, 0.079, 0.236] | C |
| 5 | wineglass | 14 | 7 | 8 | [0.085, 0.085, 0.157] | V |
| 6 | lemon | 13 | 13 | 13 | [0.098, 0.07, 0.07] | C |
| 7 | bottle_of_water | 12 | 10 | 10 | [0.074, 0.071, 0.225] | C |
| 8 | beaker | 10 | 7 | 7 | [0.078, 0.078, 0.081] | V |
| 9 | can_of_soda | 9 | 9 | 9 | [0.075, 0.075, 0.135] | C |
| 10 | ice_cube | 8 | 8 | 8 | [0.022, 0.022, 0.021] | C |
| 11 | bottle_of_medicine | 8 | 8 | 8 | [0.051, 0.051, 0.092] | C |
| 12 | apple | 8 | 6 | 6 | [0.082, 0.074, 0.074] | C |
| 13 | canned_food | 8 | 4 | 4 | [0.087, 0.099, 0.133] | C |
| 14 | jar_of_spaghetti_sauce | 6 | 1 | 6 | [0.086, 0.085, 0.129] | C |
| 15 | beeswax_candle | 5 | 5 | 5 | [0.07, 0.07, 0.206] | C (override: taxonomy says device) |
| 16 | bottle_of_essential_oil | 5 | 5 | 5 | [0.022, 0.022, 0.06] | C |
| 17 | bottle_of_catsup | 5 | 5 | 4 | [0.054, 0.053, 0.207] | C |
| 18 | bell_pepper | 5 | 5 | 5 | [0.094, 0.066, 0.069] | C |
| 19 | potato | 5 | 5 | 5 | [0.081, 0.069, 0.064] | C |
| 20 | soda_cup | 4 | 2 | 4 | [0.087, 0.087, 0.117] | V |

8/20 with >= 9 ready models. Alternates: round_bottom_flask (4), toy_dice (20), pillar_candle (4), bottle_of_sake (4), bottle_of_olive_oil (4), beefsteak_tomato (4), half_lime (4), pine_cone (4), spray_can (3), salt_shaker (3), pepper_shaker (3), beer_glass (3), pear (3), eggplant (3), zucchini (3), shiitake (3), vidalia_onion (3), bottle_of_salsa (3), pencil (69), crayon (12), marker (20)

GraspGen pipeline (object starts in hand). Scoring mode per category from the BDDL taxonomy: C = confidence gate then CoM (default here: bottles, cans, produce, plaything), V = vessel (wineglass, beaker, soda_cup: default CoM), H = confidence-first (none needed; beeswax_candle is tagged device.n.01 by the taxonomy -> override to C). Rotationally symmetric bodies; vase (32) and pillar_candle are too wide for the gripper; cap/lid reserved for attachment; toy_dice moved to disembedding.

## mental_rotation (19) - must NOT be symmetric

| # | category | ready | pinch | width-fit | median bbox | GraspGen mode |
|---|---|---|---|---|---|---|
| 1 | toy_figure | 35 | 23 | 6 | [0.15, 0.139, 0.079] | C |
| 2 | carving_knife | 5 | 5 | 5 | [0.277, 0.037, 0.024] | H |
| 3 | walker | 12 | 10 | 9 | [0.223, 0.08, 0.086] | C |
| 4 | coffee_cup | 11 | 11 | 11 | [0.095, 0.074, 0.071] | V |
| 5 | mug | 10 | 7 | 5 | [0.087, 0.118, 0.083] | V |
| 6 | statue | 10 | 4 | 2 | [0.234, 0.141, 0.235] | C |
| 7 | spatula | 9 | 9 | 7 | [0.345, 0.076, 0.03] | H |
| 8 | teaspoon | 9 | 9 | 9 | [0.131, 0.029, 0.016] | H |
| 9 | mouse | 7 | 7 | 7 | [0.109, 0.059, 0.03] | H (device; harmless) |
| 10 | teacup | 6 | 4 | 6 | [0.089, 0.081, 0.064] | V |
| 11 | tablefork | 5 | 5 | 5 | [0.201, 0.028, 0.016] | H |
| 12 | soup_ladle | 5 | 2 | 2 | [0.326, 0.087, 0.087] | H (override: taxonomy says vessel) |
| 13 | hook | 5 | 5 | 5 | [0.044, 0.066, 0.041] | H |
| 14 | crab | 5 | 5 | 0 | [0.154, 0.141, 0.031] | C |
| 15 | cellulose_tape_dispenser | 3 | 3 | 3 | [0.112, 0.05, 0.068] | V |
| 16 | tongs | 3 | 3 | 3 | [0.247, 0.038, 0.026] | H |
| 17 | vegetable_peeler | 2 | 2 | 2 | [0.18, 0.083, 0.028] | H |
| 18 | hammer | 2 | 2 | 0 | [0.286, 0.104, 0.05] | H |
| 19 | banana | 2 | 2 | 1 | [0.191, 0.107, 0.041] | C |

7/19 with >= 9 ready models. Alternates: sunflower (18), sandal (9), paintbrush (2), high_heel (2), baby_shoe (2), cream_pitcher (2), toothbrush (4), gaming_controller (1), stapler (1), plier (1), drill (1), corkscrew (1), mallet (1), trowel (1), hairbrush (1), scrub_brush (1), watch (1), whistle (1), pruner (1), open_end_wrench (1), shears (1), dustpan (1)

GraspGen pipeline (object starts in hand). Modes: H = confidence-first (handled implements: fork, spoon, spatula, hammer, tongs, peeler, ladle), V = vessels (mug, cups, ladle-like containers; default CoM so the body stays at the wrist), C = confidence gate then CoM. 3-D shapes with no rotational symmetry; bounding-box heuristics cannot detect this reliably, so the list is curated by semantics. sandal moved to IORS-under and sunflower to IORS-inside; statue (10 ready, 4 pinchable, 23 cm) and carving_knife (5 ready, blade+handle, mirror-symmetric only) take their slots. toy_figure needs per-model curation (35 ready, 23 pinchable). Only 8 categories have >= 9 ready models.

## disembedding (19) - settles on a flat surface

| # | category | ready | pinch | width-fit | median bbox |
|---|---|---|---|---|---|
| 1 | toy_dice | 20 | 14 | 14 | [0.041, 0.041, 0.041] |
| 2 | pack_of_cigarettes | 16 | 16 | 16 | [0.11, 0.055, 0.016] |
| 3 | video_game | 12 | 12 | 12 | [0.061, 0.063, 0.007] |
| 4 | box_of_cereal | 14 | 14 | 2 | [0.052, 0.127, 0.21] |
| 5 | box_of_sanitary_napkins | 9 | 9 | 3 | [0.07, 0.096, 0.109] |
| 6 | box_of_baking_mix | 8 | 8 | 0 | [0.05, 0.115, 0.153] |
| 7 | notebook | 34 | 34 | 6 | [0.232, 0.223, 0.012] |
| 8 | tortilla_chips | 6 | 6 | 6 | [0.089, 0.072, 0.022] |
| 9 | box_of_cookies | 14 | 13 | 4 | [0.205, 0.189, 0.051] |
| 10 | box_of_yogurt | 3 | 2 | 2 | [0.065, 0.064, 0.054] |
| 11 | bar_soap | 4 | 4 | 4 | [0.101, 0.088, 0.034] |
| 12 | box_of_shampoo | 4 | 4 | 4 | [0.056, 0.092, 0.206] |
| 13 | box_of_cane_sugar | 4 | 3 | 1 | [0.08, 0.113, 0.171] |
| 14 | box_of_tissues | 4 | 4 | 0 | [0.107, 0.127, 0.075] |
| 15 | carton_of_milk | 4 | 3 | 1 | [0.084, 0.115, 0.166] |
| 16 | chili | 4 | 4 | 4 | [0.064, 0.026, 0.013] |
| 17 | pencil_case | 3 | 3 | 3 | [0.063, 0.193, 0.06] |
| 18 | cookie_cutter | 3 | 3 | 3 | [0.056, 0.056, 0.02] |
| 19 | cellulose_tape | 3 | 3 | 3 | [0.05, 0.05, 0.017] |

7/19 with >= 9 ready models. Alternates: soap_dispenser (6), bottle_of_milk (6), post_it (3), wall_socket (4), seashell (4), sliced_lemon (5), dice (3), box_of_rice (3), box_of_granola_bars (3), box_of_oatmeal (3), boxed_ink_cartridge (3), half_lime (4)

A1 pipeline (object rests on a table/tray), so every entry must be pinchable IN ITS RESTING POSE: lying flat with the second-smallest extent <= 9.5 cm (dice, cigarette pack, game cartridge, soap, tape, cookie cutter, chili, the 6 small notebooks) or a box that stands stably on its narrow face with that face <= 9 cm wide (cereal 5.2, baking mix 5.0, sanitary napkins 7.0, cane sugar 8.0, tissues 7.5, milk carton 8.4, shampoo box 5.6, pencil case 6.3) or a small box (box_of_yogurt 6.5 x 6.4 x 5.4). Dropped from the previous list because a flat book/board cannot be pinched from above (needs a finger under it): hardback, paperback_book, letter, board_game, chopping_board, magazine, comic_book, notepad, chocolate_bar. Counts are 'ready models that pass this pose test', see disembedding_table_graspable.json. Only 5 categories have >= 9 such models.

## configuration_matching - articulation (9)

| # | category | ready | pinch | width-fit | median bbox |
|---|---|---|---|---|---|
| 1 | bottom_cabinet | 64 | - | - | [0.306, 0.555, 0.722] |
| 2 | top_cabinet | 17 | - | - | [0.25, 0.565, 0.404] |
| 3 | bottom_cabinet_no_top | 16 | - | - | [0.453, 0.453, 0.605] |
| 4 | fridge | 8 | - | - | [0.686, 0.636, 1.678] |
| 5 | dishwasher | 7 | - | - | [0.571, 0.526, 0.706] |
| 6 | carton | 6 | - | - | [0.431, 0.476, 0.344] |
| 7 | oven | 5 | - | - | [0.525, 0.527, 0.536] |
| 8 | clothes_dryer | 5 | - | - | [0.628, 0.605, 0.766] |
| 9 | hinged_jar | 3 | - | - | [0.154, 0.133, 0.204] |

3/9 with >= 9 ready models. Alternates: washer (5), recycling_bin (3), laptop (1), toolbox (1), microwave (0)

openable + articulated. Scene objects have a ready model in the 7 challenge scenes: bottom_cabinet (house_single_floor xqmipa, house_double_floor_lower qacthv, house_double_floor_upper nddvba, Rs_int bamfsz/dajebq/slgzfc, restaurant_diner olgoza, office immwzb/rvunhj), top_cabinet (Rs_int fqhdne/lsyzkh, office fqitbw), bottom_cabinet_no_top (house_double_floor_upper ncgmqr), fridge (house_single_floor dszchb, Rs_int xyejdx, restaurant jtqazu/juwaoh), dishwasher (ynrhuk single/lower, Rs_int xlmier), oven (ffitak single/lower), clothes_dryer (single xsuyua). carton and hinged_jar are non-scene openables. washer/microwave models in the scenes are NOT ready; laptop has one ready model (Rs_int nvulcs).

## configuration_matching - attachment (9)

| # | category | ready | pinch | width-fit | median bbox |
|---|---|---|---|---|---|
| 1 | lid | 48 | 42 | 5 | [0.2, 0.2, 0.048] |
| 2 | cap | 28 | 28 | 28 | [0.052, 0.052, 0.018] |
| 3 | license_plate | 5 | 5 | 0 | [0.308, 0.155, 0.004] |
| 4 | dip_candle | 4 | 4 | 4 | [0.025, 0.027, 0.147] |
| 5 | cork | 3 | 3 | 3 | [0.019, 0.018, 0.034] |
| 6 | light_bulb | 3 | 2 | 2 | [0.113, 0.07, 0.069] |
| 7 | wreath | 1 | 1 | 0 | [0.2, 0.198, 0.042] |
| 8 | dartboard | 1 | - | - | [0.451, 0.451, 0.017] |
| 9 | webcam | 1 | 1 | 1 | [0.069, 0.091, 0.065] |

2/9 with >= 9 ready models. Alternates: bow (2), holly_decoration (1), fire_alarm (5), antlers (2), menu (6), address (1), poster (1), skateboard_wheel_assembly (2), sticker (1)

Table-graspability in the resting pose: cap, cork, dip_candle, light_bulb, webcam and the wreath (ring cross-section) are fine; a lid lying knob-up must be grasped by its knob (check per model that the knob is in the collision mesh, otherwise rest it rim-up); license_plate (30 x 15 x 0.4 cm) and dartboard (45 cm disc) cannot be pinched off a table and would have to start leaning against something or be handed over. graspable (male) part -> partner: lid -> hingeless_jar x4, storage_box x3, teapot x2, kettle x2, canister, tupperware, stockpot, pressure_cooker, crock_pot, steamer_basket, compost_bin, toy_box, trash_can; cap -> bottle_of_whiskey x4, bottle_of_water, bottle_of_vodka, oil_bottle, jar_of_clove, jar_of_tumeric, soda_cup, paper_cup, bird_feeder, deodorant_stick, round_bottom_flask (+ one generic cap for 19 bottle categories and one for 9 cartons); license_plate -> car (2 ready cars, present in the house scenes); dip_candle -> candle_holder (5 ready); cork -> wine_bottle / rum_bottle / vinegar_bottle; light_bulb -> table_lamp; wreath / dartboard / bow / holly_decoration / fire_alarm / antlers / menu / address / poster -> wall_nail (scene object, 1 ready); webcam -> monitor; skateboard_wheel_assembly -> skateboard_deck; sticker -> license_plate. Only lid and cap have >= 9 ready models; everything else is 1-5.

## IORS - ontop (4)

| # | category | ready | pinch | width-fit | median bbox |
|---|---|---|---|---|---|
| 1 | pen | 6 | 6 | 6 | [0.153, 0.011, 0.011] |
| 2 | shampoo_dispenser | 7 | 6 | 7 | [0.067, 0.052, 0.145] |
| 3 | cherry_tomato | 5 | 5 | 5 | [0.034, 0.024, 0.022] |
| 4 | pickle | 5 | 5 | 5 | [0.035, 0.1, 0.034] |

0/4 with >= 9 ready models. Alternates: french_fries (11), sliced_lemon (5), garlic_clove (10), bottle_of_milk (6), pack_of_kielbasa (16)

pen -> desk (scene), shampoo_dispenser -> countertop (scene), cherry_tomato -> plate (46), pickle -> cutting_board (5).

## IORS - under (4)

| # | category | ready | pinch | width-fit | median bbox |
|---|---|---|---|---|---|
| 1 | french_fries | 11 | 11 | 11 | [0.069, 0.011, 0.008] |
| 2 | beer_glass | 3 | 2 | 2 | [0.06, 0.06, 0.117] |
| 3 | sandal | 9 | 7 | 5 | [0.237, 0.091, 0.059] |
| 4 | briefcase | 7 | 3 | 0 | [0.284, 0.332, 0.097] |

2/4 with >= 9 ready models. Alternates: toy_figure (small under furniture, disjoint models from mental_rotation) (None), walker (12), tennis_ball + baseball + pool_ball (1 model each) (None), toy_dice (20), sliced_lemon (5), tablespoon (4)

two small-under-small (the TOP object is grasped and placed on the bottom one): platter (6) under french_fries (11); coaster (2) or saucer (4) under beer_glass (3). Two small-under-big (the small object is grasped and placed under furniture with enough clearance: desk 0.75 m, breakfast_table 0.70 m, pedestal_table 0.69 m, bed; not coffee_table 0.41 m): sandal (9 ready, 5 fit by width, 7 pinchable) under bed / desk; briefcase (7 ready) under desk - its body is 9.7 cm thick so only 3 models pinch by thickness, the rest need a handle grasp (verify the handle is in the collision mesh). sandal therefore leaves mental_rotation (walker keeps footwear covered there; statue takes the slot).

## IORS - inside (4)

| # | category | ready | pinch | width-fit | median bbox |
|---|---|---|---|---|---|
| 1 | garlic_clove | 10 | 10 | 10 | [0.009, 0.013, 0.006] |
| 2 | sunflower | 18 | 18 | 14 | [0.191, 0.08, 0.07] |
| 3 | quartz | 6 | 5 | 5 | [0.03, 0.095, 0.026] |
| 4 | hard_candy | 5 | 5 | 5 | [0.023, 0.055, 0.021] |

2/4 with >= 9 ready models. Alternates: lily (10 models, not 'ready' only because of an automatic non-VRay material flag; renders fine) (None), dahlia_flower (2, same flag, renders fine) (None), pen (6), plant_stem (3), cinnamon_stick (8), chopstick (5), sliced_lemon (5), french_fries (11), spinach (6)

garlic_clove -> bowl (39), sunflower -> vase (32, fillable), quartz -> saucepot (12), hard_candy -> hingeless_jar (8). sunflower therefore leaves mental_rotation (carving_knife takes the slot).

Flowers: there is no tidy shared synset. In the BDDL/WordNet taxonomy sunflower.n.01 and dahlia.n.01 sit under flower.n.01 (together with marigold, poinsettia, rose), while lily.n.01 and tulip.n.01 sit under liliaceous_plant.n.01 / bulbous_plant.n.01; the lowest common ancestor of all three is vascular_plant.n.01, which also covers trees and grass, so it cannot be used as the task synset. Treat 'flower' as a benchmark-level group {sunflower 18, lily 10 (flagged), dahlia_flower 2 (flagged)} instead; marigold (10 ready) and poinsettia are bushy 12-25 cm bodies with no pinchable stem, so they do not help.

Vase heights (TODO when building instances): OmniGibson's Inside tests the flower's AABB centre against the vase's fillable volume, so the centre must be below the rim: flower length <= ~2 x vase interior depth. Plan: pick the 9 tallest TABLE-scale vases (heights 0.27-0.47 m: aegxpb 0.269, meetii 0.301, hliauj 0.315, saenda 0.417, tjrbxv 0.473, ...; skip the four floor vases of 0.77-1.5 m, 37-53 cm wide) and rescale each flower so that its centre ends up inside the vase (e.g. sunflower 0.11-0.24 m fits as is; lily 0.42-0.47 m needs ~0.6-0.9 x scale for a 0.3 m vase, or a 0.47 m vase). Keep stems >= 6 mm thick after scaling so they remain pinchable.

## IORS - nextto (4)

| # | category | ready | pinch | width-fit | median bbox |
|---|---|---|---|---|---|
| 1 | marker | 20 | 20 | 20 | [0.154, 0.017, 0.014] |
| 2 | pencil | 69 | 69 | 69 | [0.129, 0.007, 0.006] |
| 3 | cinnamon_stick | 8 | 8 | 8 | [0.019, 0.134, 0.011] |
| 4 | parsnip | 6 | 6 | 6 | [0.05, 0.23, 0.047] |

2/4 with >= 9 ready models. Alternates: ruler (5), colored_pencil (20), star_anise (3), carrot (3), whisk (5), bottle_of_milk (6)

marker <-> colored_pencil (20/20), pencil <-> ruler (69/5), cinnamon_stick <-> star_anise (8/3), parsnip <-> carrot (6/3). Rods roll a little on a table; the vegetable pair settles best.

## Overlaps between tasks

none
