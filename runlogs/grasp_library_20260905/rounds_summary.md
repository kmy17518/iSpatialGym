# Category selection rounds 1-3: outcome

round 1 = original plan (validated in the first two runs); round 2 = categories reused from another task (grasps re-tested only where the grasp method differs); round 3 = never-used categories, single-model categories allowed.
L1 / L3 / L5 = models with >= 1 / >= 3 / 5 reviewer-clean grasps (lenient: `one_finger` ignored); S1 = models with >= 1 strictly clean grasp. Articulation rows have no grasp test (handle grasps on articulated links are a separate pipeline).

## Per task and round

| task | round | categories | models | grasp tests run | AG-valid | L1 | L3 | L5 | S1 | categories ok / thin / weak / drop |
|---|---|---|---|---|---|---|---|---|---|---|
| object_scaling | 1 | 20 | 162 | 162 | 160 | 157 | 153 | 145 | 146 | 18 / 2 / 0 / 0 |
| object_scaling | 2 | 10 | 93 | 93 | 86 | 85 | 80 | 72 | 83 | 9 / 1 / 0 / 0 |
| object_scaling | 3 | 10 | 33 | 33 | 32 | 29 | 28 | 27 | 26 | 0 / 8 / 1 / 1 |
| mental_rotation | 1 | 19 | 123 | 123 | 119 | 114 | 107 | 99 | 111 | 12 / 4 / 3 / 0 |
| mental_rotation | 2 | 10 | 63 | 53 | 59 | 53 | 46 | 39 | 53 | 7 / 2 / 1 / 0 |
| mental_rotation | 3 | 10 | 19 | 19 | 18 | 18 | 18 | 13 | 18 | 0 / 2 / 8 / 0 |
| disembedding | 1 | 19 | 131 | 131 | 120 | 118 | 118 | 116 | 89 | 8 / 9 / 2 / 0 |
| disembedding | 2 | 10 | 94 | 94 | 89 | 89 | 88 | 84 | 80 | 10 / 0 / 0 / 0 |
| disembedding | 3 | 10 | 36 | 36 | 34 | 33 | 31 | 27 | 33 | 2 / 5 / 3 / 0 |
| configuration_matching | 1 | 9 | 42 | 42 | 23 | 21 | 18 | 17 | 20 | 1 / 4 / 1 / 3 |
| configuration_matching | 2 | 2 | 2 | 2 | 1 | 0 | 0 | 0 | 0 | 0 / 0 / 0 / 2 |
| configuration_matching | 3 | 10 | 11 | 11 | 7 | 7 | 6 | 5 | 7 | 0 / 0 / 6 / 4 |
| articulation | 1 | 9 | 70 | n/a | | | | | | |
| articulation | 3 | 10 | 30 | n/a | | | | | | |
| IORS | 1 | 16 | 124 | 124 | 113 | 100 | 90 | 78 | 94 | 12 / 3 / 1 / 0 |
| IORS | 2 | 10 | 84 | 84 | 79 | 78 | 74 | 67 | 73 | 9 / 1 / 0 / 0 |
| IORS | 3 | 10 | 42 | 42 | 37 | 35 | 32 | 30 | 33 | 1 / 4 / 5 / 0 |

## Per category

### object_scaling

| round | relation | category | partner / note | models | tests | AG | L1 | L3 | L5 | S1 | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  | bottle_of_pop |  | 12 | 12 | 12 | 12 | 12 | 12 | 12 | ok |  |
| 1 |  | bottle_of_wine |  | 12 | 12 | 12 | 12 | 12 | 11 | 11 | ok | end_grasp 1, one_finger 1 |
| 1 |  | bottle_of_black_pepper |  | 4 | 4 | 4 | 4 | 4 | 4 | 2 | thin | one_finger 3 |
| 1 |  | bottle_of_beer |  | 12 | 12 | 12 | 12 | 12 | 12 | 10 | ok | one_finger 7 |
| 1 |  | wineglass |  | 12 | 12 | 12 | 11 | 11 | 11 | 9 | ok | one_finger 3, fragile 1 |
| 1 |  | lemon |  | 12 | 12 | 12 | 12 | 12 | 12 | 12 | ok | one_finger 1 |
| 1 |  | bottle_of_water |  | 12 | 12 | 12 | 12 | 10 | 10 | 12 | ok |  |
| 1 |  | beaker |  | 10 | 10 | 10 | 9 | 9 | 9 | 9 | ok | gap 1 |
| 1 |  | can_of_soda |  | 9 | 9 | 9 | 9 | 9 | 9 | 9 | ok |  |
| 1 |  | ice_cube |  | 8 | 8 | 8 | 8 | 8 | 6 | 8 | ok |  |
| 1 |  | bottle_of_medicine |  | 8 | 8 | 8 | 8 | 8 | 8 | 7 | ok | one_finger 4 |
| 1 |  | apple |  | 8 | 8 | 7 | 7 | 7 | 6 | 7 | ok |  |
| 1 |  | canned_food |  | 8 | 8 | 7 | 6 | 5 | 4 | 6 | ok | end_grasp 2 |
| 1 |  | jar_of_spaghetti_sauce |  | 6 | 6 | 6 | 6 | 6 | 4 | 6 | ok | end_grasp 2 |
| 1 |  | beeswax_candle |  | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok |  |
| 1 |  | bottle_of_essential_oil |  | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok |  |
| 1 |  | bottle_of_catsup |  | 5 | 5 | 5 | 5 | 5 | 5 | 2 | ok | one_finger 4 |
| 1 |  | bell_pepper |  | 5 | 5 | 5 | 5 | 5 | 4 | 5 | ok | end_grasp 1 |
| 1 |  | potato |  | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok |  |
| 1 |  | soda_cup |  | 4 | 4 | 4 | 4 | 3 | 3 | 4 | thin | one_finger 2 |
| 2 |  | lid | round disc | 12 | 12 | 12 | 12 | 12 | 10 | 12 | ok | end_grasp 1 |
| 2 |  | cap | round | 12 | 12 | 12 | 12 | 12 | 10 | 10 | ok | task_violation 1, one_finger 3, end_grasp 1 |
| 2 |  | toy_dice | cube | 12 | 12 | 6 | 6 | 6 | 6 | 6 | ok |  |
| 2 |  | marker | rod | 12 | 12 | 11 | 11 | 6 | 2 | 11 | ok | penetration 1 |
| 2 |  | pack_of_cigarettes | box | 12 | 12 | 12 | 12 | 12 | 12 | 12 | ok |  |
| 2 |  | box_of_cereal | box | 12 | 12 | 12 | 12 | 12 | 12 | 12 | ok |  |
| 2 |  | shampoo_dispenser | round body | 7 | 7 | 7 | 6 | 6 | 6 | 6 | ok | end_grasp 1, fragile 1 |
| 2 |  | hard_candy | wrapped, mirror-symmetric | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok |  |
| 2 |  | cherry_tomato | round | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok |  |
| 2 |  | bar_soap | rounded bar | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin |  |
| 3 |  | spray_can |  | 3 | 3 | 3 | 3 | 3 | 2 | 3 | thin |  |
| 3 |  | bottle_of_salsa |  | 3 | 3 | 3 | 0 | 0 | 0 | 0 | drop | end_grasp 3 |
| 3 |  | bottle_of_olive_oil |  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin |  |
| 3 |  | bottle_of_sake |  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin |  |
| 3 |  | beefsteak_tomato |  | 4 | 4 | 4 | 4 | 3 | 3 | 3 | thin | gap 1, one_finger 3 |
| 3 |  | pine_cone |  | 4 | 4 | 3 | 3 | 3 | 3 | 2 | thin | one_finger 1 |
| 3 |  | pear |  | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin | one_finger 1 |
| 3 |  | orange |  | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak |  |
| 3 |  | vidalia_onion |  | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin |  |
| 3 |  | shiitake |  | 3 | 3 | 3 | 3 | 3 | 3 | 2 | thin | one_finger 3 |

### mental_rotation

| round | relation | category | partner / note | models | tests | AG | L1 | L3 | L5 | S1 | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  | toy_figure |  | 12 | 12 | 12 | 11 | 10 | 9 | 11 | ok | end_grasp 3, one_finger 1 |
| 1 |  | carving_knife |  | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok |  |
| 1 |  | walker |  | 12 | 12 | 12 | 12 | 12 | 12 | 9 | ok | one_finger 7 |
| 1 |  | coffee_cup |  | 11 | 11 | 11 | 11 | 11 | 11 | 11 | ok |  |
| 1 |  | mug |  | 10 | 10 | 10 | 10 | 10 | 9 | 10 | ok | gap 1 |
| 1 |  | statue |  | 10 | 10 | 8 | 4 | 4 | 4 | 4 | thin | end_grasp 4, fragile 2, one_finger 1 |
| 1 |  | spatula |  | 9 | 9 | 9 | 9 | 9 | 7 | 9 | ok | gap 2 |
| 1 |  | teaspoon |  | 9 | 9 | 8 | 8 | 4 | 4 | 8 | ok |  |
| 1 |  | mouse |  | 7 | 7 | 7 | 7 | 7 | 6 | 7 | ok | end_grasp 1 |
| 1 |  | teacup |  | 6 | 6 | 6 | 6 | 6 | 6 | 6 | ok |  |
| 1 |  | tablefork |  | 5 | 5 | 4 | 4 | 2 | 1 | 4 | thin | gap 3, penetration 1 |
| 1 |  | soup_ladle |  | 5 | 5 | 5 | 5 | 5 | 3 | 5 | ok | gap 2 |
| 1 |  | hook |  | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok | unsure 3 |
| 1 |  | crab |  | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok |  |
| 1 |  | cellulose_tape_dispenser |  | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin |  |
| 1 |  | tongs |  | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin |  |
| 1 |  | vegetable_peeler |  | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak |  |
| 1 |  | hammer |  | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak |  |
| 1 |  | banana |  | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak |  |
| 2 |  | sandal |  | 9 | 9 | 9 | 8 | 8 | 8 | 8 | ok | fragile 1, one_finger 1, unsure 2 |
| 2 |  | briefcase | handle | 7 | 7 | 3 | 3 | 1 | 1 | 3 | thin | end_grasp 1, gap 2 |
| 2 |  | sunflower |  | 12 | 12 | 12 | 11 | 8 | 5 | 11 | ok | fragile 7, one_finger 3 |
| 2 |  | garlic_clove | scaled x2.2-3.8 | 10 | 10 | 10 | 6 | 4 | 3 | 6 | ok | penetration 6, unsure 1, gap 1 |
| 2 |  | quartz |  | 6 | 6 | 6 | 6 | 6 | 6 | 6 | ok | one_finger 1 |
| 2 |  | pickle |  | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok |  |
| 2 |  | webcam |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 | weak |  |
| 2 |  | cookie_cutter | shaped | 3 | 3 | 3 | 3 | 3 | 1 | 3 | thin | end_grasp 1, gap 2 |
| 2 |  | bell_pepper | same floating grasps as object scaling | 5 | 0 | 5 | 5 | 5 | 4 | 5 | ok | end_grasp 1 |
| 2 |  | potato | same floating grasps as object scaling | 5 | 0 | 5 | 5 | 5 | 5 | 5 | ok |  |
| 3 |  | toothbrush | H | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin |  |
| 3 |  | tablespoon | H | 4 | 4 | 3 | 3 | 3 | 0 | 3 | thin | end_grasp 1 |
| 3 |  | high_heel |  | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak |  |
| 3 |  | baby_shoe |  | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak | one_finger 2 |
| 3 |  | ginger_root |  | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak | one_finger 1 |
| 3 |  | gaming_controller |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 | weak |  |
| 3 |  | stapler |  | 1 | 1 | 1 | 1 | 1 | 1 | 1 | weak |  |
| 3 |  | drill | H | 1 | 1 | 1 | 1 | 1 | 1 | 1 | weak |  |
| 3 |  | mallet | H | 1 | 1 | 1 | 1 | 1 | 0 | 1 | weak | gap 1 |
| 3 |  | cleaver | handle half | 1 | 1 | 1 | 1 | 1 | 0 | 1 | weak | gap 1 |

### disembedding

| round | relation | category | partner / note | models | tests | AG | L1 | L3 | L5 | S1 | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  | toy_dice |  | 12 | 12 | 12 | 12 | 12 | 12 | 12 | ok |  |
| 1 |  | pack_of_cigarettes |  | 12 | 12 | 12 | 12 | 12 | 11 | 11 | ok | one_finger 6, gap 1 |
| 1 |  | video_game |  | 12 | 12 | 12 | 12 | 12 | 12 | 10 | ok | one_finger 8 |
| 1 |  | box_of_cereal |  | 12 | 12 | 12 | 12 | 12 | 12 | 4 | ok | one_finger 11 |
| 1 |  | box_of_sanitary_napkins |  | 9 | 9 | 9 | 9 | 9 | 9 | 9 | ok | one_finger 2 |
| 1 |  | box_of_baking_mix |  | 8 | 8 | 8 | 8 | 8 | 8 | 0 | ok | one_finger 8 |
| 1 |  | notebook |  | 12 | 12 | 6 | 6 | 6 | 6 | 6 | ok |  |
| 1 |  | tortilla_chips |  | 6 | 6 | 3 | 2 | 2 | 1 | 2 | weak | gap 1 |
| 1 |  | box_of_cookies |  | 12 | 12 | 11 | 10 | 10 | 10 | 5 | ok | one_finger 7, task_violation 1 |
| 1 |  | box_of_yogurt |  | 3 | 3 | 2 | 2 | 2 | 2 | 2 | weak |  |
| 1 |  | bar_soap |  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin | one_finger 1 |
| 1 |  | box_of_shampoo |  | 4 | 4 | 4 | 4 | 4 | 4 | 0 | thin | one_finger 4 |
| 1 |  | box_of_cane_sugar |  | 4 | 4 | 4 | 4 | 4 | 4 | 3 | thin | one_finger 1 |
| 1 |  | box_of_tissues |  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin |  |
| 1 |  | carton_of_milk |  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin |  |
| 1 |  | chili |  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin |  |
| 1 |  | pencil_case |  | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin | one_finger 2 |
| 1 |  | cookie_cutter |  | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin |  |
| 1 |  | cellulose_tape |  | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin |  |
| 2 |  | bottle_of_pop | stands | 12 | 12 | 10 | 10 | 10 | 8 | 10 | ok | gap 2 |
| 2 |  | bottle_of_wine | stands | 12 | 12 | 12 | 12 | 12 | 11 | 12 | ok | end_grasp 1 |
| 2 |  | bottle_of_beer | stands | 12 | 12 | 12 | 12 | 12 | 12 | 12 | ok |  |
| 2 |  | can_of_soda | stands | 9 | 9 | 9 | 9 | 9 | 9 | 9 | ok |  |
| 2 |  | bottle_of_medicine | stands | 8 | 8 | 8 | 8 | 8 | 8 | 1 | ok | one_finger 7 |
| 2 |  | mug | stands | 10 | 10 | 10 | 10 | 10 | 10 | 10 | ok |  |
| 2 |  | mouse | flat | 7 | 7 | 7 | 7 | 7 | 7 | 6 | ok | one_finger 4 |
| 2 |  | teaspoon | lies flat | 9 | 9 | 7 | 7 | 6 | 6 | 6 | ok | one_finger 1 |
| 2 |  | tablefork | lies flat | 5 | 5 | 5 | 5 | 5 | 4 | 5 | ok |  |
| 2 |  | beaker | stands | 10 | 10 | 9 | 9 | 9 | 9 | 9 | ok |  |
| 3 |  | soap_dispenser |  | 6 | 6 | 6 | 6 | 6 | 5 | 6 | ok | fragile 1 |
| 3 |  | bottle_of_milk |  | 6 | 6 | 6 | 6 | 6 | 5 | 6 | ok |  |
| 3 |  | box_of_rice | stands on narrow face | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin |  |
| 3 |  | box_of_granola_bars |  | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin |  |
| 3 |  | box_of_oatmeal |  | 3 | 3 | 2 | 2 | 1 | 1 | 2 | weak |  |
| 3 |  | pencil_holder |  | 4 | 4 | 4 | 3 | 2 | 2 | 3 | thin | end_grasp 2 |
| 3 |  | bottle_of_shampoo |  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin |  |
| 3 |  | conditioner_dispenser |  | 3 | 3 | 3 | 3 | 3 | 2 | 3 | thin | gap 1 |
| 3 |  | jar_of_bath_salt |  | 2 | 2 | 1 | 1 | 1 | 0 | 1 | weak | gap 1 |
| 3 |  | box_of_butter |  | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak |  |

### configuration_matching

| round | relation | category | partner / note | models | tests | AG | L1 | L3 | L5 | S1 | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  | lid |  | 12 | 12 | 5 | 4 | 4 | 3 | 3 | thin | one_finger 2, gap 2 |
| 1 |  | cap |  | 12 | 12 | 6 | 6 | 4 | 4 | 6 | ok |  |
| 1 |  | license_plate |  | 5 | 5 | 0 | 0 | 0 | 0 | 0 | drop |  |
| 1 |  | dip_candle |  | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin |  |
| 1 |  | cork |  | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin |  |
| 1 |  | light_bulb |  | 3 | 3 | 3 | 3 | 3 | 3 | 3 | thin |  |
| 1 |  | wreath |  | 1 | 1 | 1 | 0 | 0 | 0 | 0 | drop | fragile 1, one_finger 1 |
| 1 |  | dartboard |  | 1 | 1 | 0 | 0 | 0 | 0 | 0 | drop |  |
| 1 |  | webcam |  | 1 | 1 | 1 | 1 | 0 | 0 | 1 | weak | gap 1, one_finger 1 |
| 2 |  | bottle_of_water | F part: partner cap must be fixed | 1 | 1 | 1 | 0 | 0 | 0 | 0 | drop | task_violation 1 |
| 2 |  | soda_cup | F part: partner cap must be fixed | 1 | 1 | 0 | 0 | 0 | 0 | 0 | drop |  |
| 3 |  | blackberry | M | 1 | 1 | 1 | 1 | 1 | 1 | 1 | weak |  |
| 3 |  | holly_decoration | M -> wall_nail | 1 | 1 | 1 | 1 | 1 | 0 | 1 | weak | fragile 1, one_finger 1 |
| 3 |  | trout | M -> fishing_rod | 1 | 1 | 1 | 1 | 1 | 1 | 1 | weak |  |
| 3 |  | antlers | M -> wall_nail | 1 | 1 | 0 | 0 | 0 | 0 | 0 | drop |  |
| 3 |  | skateboard_wheel_assembly | M -> skateboard_deck | 2 | 2 | 2 | 2 | 1 | 1 | 2 | weak | end_grasp 1 |
| 3 |  | fire_alarm | M | 1 | 1 | 0 | 0 | 0 | 0 | 0 | drop |  |
| 3 |  | bow | M | 1 | 1 | 0 | 0 | 0 | 0 | 0 | drop |  |
| 3 |  | cabinet_door | M -> cabinet_base | 1 | 1 | 1 | 1 | 1 | 1 | 1 | weak |  |
| 3 |  | bookcase_baseboard | M -> bookcase_side | 1 | 1 | 1 | 1 | 1 | 1 | 1 | weak |  |
| 3 |  | menu | M -> displaywall; 1 mm sheet, expected to fail like license_plate | 1 | 1 | 0 | 0 | 0 | 0 | 0 | drop |  |

### IORS

| round | relation | category | partner / note | models | tests | AG | L1 | L3 | L5 | S1 | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  | pen |  | 6 | 6 | 6 | 6 | 6 | 6 | 6 | ok |  |
| 1 |  | shampoo_dispenser |  | 7 | 7 | 7 | 7 | 7 | 7 | 5 | ok | one_finger 7 |
| 1 |  | cherry_tomato |  | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok |  |
| 1 |  | pickle |  | 5 | 5 | 5 | 5 | 5 | 5 | 5 | ok |  |
| 1 |  | french_fries |  | 11 | 11 | 9 | 3 | 3 | 3 | 3 | thin | penetration 5, end_grasp 1 |
| 1 |  | beer_glass |  | 3 | 3 | 3 | 2 | 2 | 2 | 0 | weak | one_finger 2, fragile 1 |
| 1 |  | sandal |  | 9 | 9 | 9 | 9 | 8 | 7 | 7 | ok | one_finger 5, gap 2 |
| 1 |  | briefcase |  | 7 | 7 | 3 | 3 | 3 | 3 | 3 | thin |  |
| 1 |  | garlic_clove |  | 10 | 10 | 10 | 10 | 9 | 8 | 10 | ok | one_finger 1, penetration 1 |
| 1 |  | sunflower |  | 12 | 12 | 7 | 7 | 4 | 2 | 7 | ok | penetration 1 |
| 1 |  | quartz |  | 6 | 6 | 6 | 5 | 5 | 3 | 5 | ok | end_grasp 1, one_finger 1 |
| 1 |  | hard_candy |  | 5 | 5 | 5 | 5 | 4 | 3 | 5 | ok | fragile 2, one_finger 1 |
| 1 |  | marker |  | 12 | 12 | 12 | 12 | 10 | 5 | 12 | ok | end_grasp 4, penetration 3 |
| 1 |  | pencil |  | 12 | 12 | 12 | 12 | 12 | 12 | 12 | ok |  |
| 1 |  | cinnamon_stick |  | 8 | 8 | 8 | 3 | 1 | 1 | 3 | thin | penetration 6, end_grasp 1 |
| 1 |  | parsnip |  | 6 | 6 | 6 | 6 | 6 | 6 | 6 | ok |  |
| 2 | ontop | apple | plate | 8 | 8 | 6 | 6 | 6 | 6 | 6 | ok | one_finger 3 |
| 2 | ontop | coffee_cup | saucer | 11 | 11 | 11 | 11 | 9 | 9 | 11 | ok | one_finger 1 |
| 2 | ontop | canned_food | countertop (scene) | 8 | 8 | 5 | 5 | 5 | 4 | 2 | ok | one_finger 3 |
| 2 | under | teacup | saucer under teacup (teacup grasped) | 6 | 6 | 6 | 6 | 6 | 6 | 6 | ok | unsure 1 |
| 2 | under | toy_figure | under bed (small under big) | 12 | 12 | 12 | 11 | 10 | 7 | 9 | ok | unsure 1, end_grasp 2, one_finger 3 |
| 2 | inside | lemon | bowl | 12 | 12 | 12 | 12 | 12 | 12 | 12 | ok | one_finger 1 |
| 2 | inside | ice_cube | beer_glass | 8 | 8 | 8 | 8 | 7 | 4 | 8 | ok |  |
| 2 | inside | chili | bowl | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin |  |
| 2 | nextto | spatula | soup_ladle | 9 | 9 | 9 | 9 | 9 | 9 | 9 | ok |  |
| 2 | nextto | jar_of_spaghetti_sauce | canned_food | 6 | 6 | 6 | 6 | 6 | 6 | 6 | ok |  |
| 3 | ontop | pack_of_kielbasa | platter | 12 | 12 | 12 | 12 | 12 | 12 | 12 | ok |  |
| 3 | ontop | half_bagel | plate | 2 | 2 | 2 | 2 | 2 | 2 | 2 | weak | one_finger 2 |
| 3 | ontop | baguette | cutting_board | 1 | 1 | 1 | 1 | 1 | 1 | 1 | weak | one_finger 1 |
| 3 | under | bottle_of_supplements | tray under bottle (bottle grasped) | 2 | 2 | 2 | 2 | 2 | 1 | 2 | weak | gap 1, one_finger 1 |
| 3 | under | tennis_ball | under couch / bed | 1 | 1 | 1 | 1 | 1 | 1 | 1 | weak |  |
| 3 | inside | lily | vase (flower_stem_top) | 10 | 10 | 6 | 4 | 2 | 1 | 4 | thin | penetration 1, fragile 1, gap 1, one_finger 1 |
| 3 | inside | half_lime | bowl | 4 | 4 | 4 | 4 | 4 | 4 | 4 | thin | one_finger 4 |
| 3 | inside | brussels_sprouts | bowl | 4 | 4 | 4 | 4 | 4 | 4 | 3 | thin | one_finger 3 |
| 3 | nextto | salt_shaker | pepper_shaker | 3 | 3 | 2 | 2 | 2 | 2 | 1 | weak | one_finger 2 |
| 3 | nextto | pepper_shaker | salt_shaker | 3 | 3 | 3 | 3 | 2 | 2 | 3 | thin | end_grasp 1, penetration 1, one_finger 2 |

### articulation

| round | relation | category | partner / note | models | tests | AG | L1 | L3 | L5 | S1 | verdict | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | | bottom_cabinet |  | 12 | n/a | | | | | | | |
| 1 | | top_cabinet |  | 12 | n/a | | | | | | | |
| 1 | | bottom_cabinet_no_top |  | 12 | n/a | | | | | | | |
| 1 | | fridge |  | 8 | n/a | | | | | | | |
| 1 | | dishwasher |  | 7 | n/a | | | | | | | |
| 1 | | carton |  | 6 | n/a | | | | | | | |
| 1 | | oven |  | 5 | n/a | | | | | | | |
| 1 | | clothes_dryer |  | 5 | n/a | | | | | | | |
| 1 | | hinged_jar |  | 3 | n/a | | | | | | | |
| 3 | | door | scene; all 7 challenge scenes | 12 | n/a | | | | | | | |
| 3 | | openable_window | scene; house_single_floor | 4 | n/a | | | | | | | |
| 3 | | sliding_door | scene; house_single_floor | 3 | n/a | | | | | | | |
| 3 | | recycling_bin |  | 3 | n/a | | | | | | | |
| 3 | | blender |  | 2 | n/a | | | | | | | |
| 3 | | cedar_chest |  | 2 | n/a | | | | | | | |
| 3 | | toolbox |  | 1 | n/a | | | | | | | |
| 3 | | cooler |  | 1 | n/a | | | | | | | |
| 3 | | violin_case |  | 1 | n/a | | | | | | | |
| 3 | | laptop | Rs_int nvulcs | 1 | n/a | | | | | | | |

## Model usage balance

Distinct models selected: 914. Selection counts per model: 1x: 669, 2x: 245.
Categories used by more than one task: apple, bar_soap, beaker, bell_pepper, bottle_of_beer, bottle_of_medicine, bottle_of_pop, bottle_of_water, bottle_of_wine, box_of_cereal, briefcase, can_of_soda, canned_food, cap, cherry_tomato, chili, coffee_cup, cookie_cutter, garlic_clove, hard_candy, ice_cube, jar_of_spaghetti_sauce, lemon, lid, marker, mouse, mug, pack_of_cigarettes, pickle, potato, quartz, sandal, shampoo_dispenser, soda_cup, spatula, sunflower, tablefork, teacup, teaspoon, toy_dice, toy_figure, webcam.

## Notes on the selection

* Round 2 picks stay disjoint across tasks (a category is reused by at most one extra task), and inside a reused category the least-used models are taken first, so no model is selected more than twice.
* Round 2 grasp tests were skipped only where the grasp method was identical (bell_pepper and potato: GraspGen floating grasps already validated for object scaling are reused for mental rotation). Everything else in round 2 changes pipeline or constraint (tabletop -> floating, floating -> tabletop, tabletop -> insert / attachment_free) and was re-tested.
* Attachment round 2 has only 2 categories: no category used by another task carries a male attachment meta-link; bottle_of_water and soda_cup are female parts (the cap attaches to them) and both failed (task_violation / no AG). Attachment round 3 has 10 categories but 9 of them have exactly one model with an attachment link; menu / bow / fire_alarm / antlers are sheets or thin discs and fail as license_plate did.
* Articulation round 2 is empty (no category used by another task is openable + articulated); round 3 lists 10 unused openable articulated categories (door, openable_window, sliding_door with models in the challenge scenes; recycling_bin, blender, cedar_chest, toolbox, cooler, violin_case, laptop). No grasp tests: handle grasps on articulated links are a separate pipeline.
* Round 3 flowers: lily needed its own stem window (55-85% of the head-to-stem axis, blooms and leaves occupy the top 60%); 6 of 10 models produce stem candidates, 4 hold.
* Round 3 categories with a single ready model are counted as `weak` by the verdict rule (>= 1 clean model); the user allowed single-model categories in this round, so `weak` is acceptable there.
