# Scene inventories (B4)

Produced by `python -m ispatialgym.sampling.scene_inventory --scene <scene>` (one process per scene, full scene, in place).
Per scene: `supports/<name>.json` (top raster, hull, four edges with free-floor runs at stand-offs 0.35 / 0.45 / 0.55 m with the
chairs within 1 m removed, walls in front, chairs, occupants), `partners/<name>.json` (same + under-clearance map + supports within
3.5 m with geodesic distance), `artic/<name>.json` (instance scale, joints measured in the instance frame, handle candidates of the
canonical heuristic, world heights, free floor in front, prior reviewer verdict of the canonical model) and `artic/strips/<name>.png`
(review renders: overview / per-joint zoom with lettered markers / opened 60 %).

Edge run = longest contiguous lateral run (m) of base positions on free floor in front of the edge at stand-off 0.45 m (chairs removed).

## house_single_floor

### supports (49)

| name | category | rooms | top z | area m² | free cells | best edge run 0.45 (edge, m, free frac) | blocked in front | chairs ≤ 1 m | in union |
|---|---|---|---|---|---|---|---|---|---|
| coffee_table_wgtaip_0 | coffee_table | garden_0 | 0.77 | 4.31 | 10764 | +u, 2.35, 0.97 | - | 10 |  |
| bar_udatjt_0 | bar | kitchen_0 | 1.05 | 3.44 | 8605 | +u, 3.00, 1.00 | - | 0 | yes |
| breakfast_table_rhjoby_0 | breakfast_table | dining_room_0 | 0.81 | 3.38 | 8463 | +u, 2.95, 1.00 | fixed_window | 6 | yes |
| countertop_ikwqer_0 | countertop | utility_room_0 | 0.93 | 2.65 | 6620 | -u, 4.45, 1.00 | walls | 0 | yes |
| countertop_kelzer_0 | countertop | kitchen_0 | 0.89 | 1.78 | 4453 | +u, 3.10, 1.00 | - | 0 |  |
| countertop_fjkase_0 | countertop | utility_room_0 | 0.92 | 1.61 | 4026 | +u, 2.45, 1.00 | walls | 0 | yes |
| countertop_kelker_0 | countertop | kitchen_0 | 0.89 | 1.44 | 3592 | -u, 3.40, 0.99 | walls | 0 |  |
| coffee_table_qtwqqb_0 | coffee_table | living_room_1 | 0.49 | 1.42 | 3547 | +v, 0.85, 1.00 | walls | 0 | yes |
| desk_bhkhxo_0 | desk | childs_room_1 | 0.77 | 1.18 | 2940 | +u, 1.85, 1.00 | fixed_window | 0 |  |
| coffee_table_rlsebe_0 | coffee_table | living_room_0 | 0.27 | 0.92 | 2308 | +u, 1.00, 1.00 | - | 0 |  |
| desk_xhjsub_0 | desk | childs_room_0 | 0.78 | 0.83 | 2088 | +u, 1.50, 1.00 | - | 1 |  |
| bottom_cabinet_fancyy_0 | bottom_cabinet | kitchen_0 | 2.09 | 0.76 | 1897 | +u, 1.25, 1.00 | - | 0 |  |
| shelf_amyper_1 | shelf | corridor_0 | 0.40 | 0.59 | 1485 | +u, 1.75, 1.00 | - | 0 |  |
| shelf_amyper_2 | shelf | corridor_0 | 0.81 | 0.59 | 1485 | +u, 1.75, 1.00 | - | 0 |  |
| shelf_amyper_3 | shelf | corridor_0 | 1.22 | 0.59 | 1485 | +u, 1.75, 1.00 | - | 0 |  |
| shelf_amyper_4 | shelf | corridor_0 | 1.63 | 0.59 | 1485 | +u, 1.75, 1.00 | - | 0 |  |
| shelf_amyper_5 | shelf | corridor_0 | 2.04 | 0.59 | 1485 | +u, 1.75, 1.00 | - | 0 |  |
| shelf_amyper_10 | shelf | corridor_0 | 2.04 | 0.59 | 1479 | +u, 1.75, 1.00 | - | 0 |  |
| shelf_amyper_6 | shelf | corridor_0 | 0.40 | 0.59 | 1479 | +u, 1.75, 1.00 | - | 0 |  |
| shelf_amyper_7 | shelf | corridor_0 | 0.81 | 0.59 | 1479 | +u, 1.75, 1.00 | - | 0 |  |
| shelf_amyper_8 | shelf | corridor_0 | 1.22 | 0.59 | 1479 | +u, 1.75, 1.00 | - | 0 |  |
| shelf_amyper_9 | shelf | corridor_0 | 1.63 | 0.59 | 1479 | +u, 1.75, 1.00 | - | 0 |  |
| desk_nbpuns_0 | desk | childs_room_2 | 0.55 | 0.58 | 1447 | +u, 1.10, 1.00 | bed | 1 |  |
| ottoman_lsaxch_0 | ottoman | garden_0 | 0.28 | 0.55 | 1385 | -v, 0.85, 1.00 | - | 0 |  |
| ottoman_lsaxch_1 | ottoman | garden_0 | 0.28 | 0.55 | 1385 | -v, 0.85, 1.00 | - | 0 |  |
| ottoman_lsaxch_3 | ottoman | garden_0 | 0.28 | 0.55 | 1385 | -v, 0.85, 1.00 | - | 0 |  |
| ottoman_lsaxch_4 | ottoman | garden_0 | 0.28 | 0.55 | 1385 | +v, 0.85, 1.00 | - | 0 |  |
| ottoman_lsaxch_2 | ottoman | garden_0 | 0.28 | 0.55 | 1384 | -v, 0.85, 1.00 | - | 0 |  |
| bottom_cabinet_xqmipa_0 | bottom_cabinet | bathroom_0 | 0.74 | 0.50 | 1254 | +u, 1.35, 1.00 | - | 0 | yes |
| bench_pfjnnj_0 | bench | bedroom_0 | 0.35 | 0.44 | 1102 | +u, 1.15, 1.00 | - | 0 |  |
| countertop_sjxber_0 | countertop | utility_room_0 | 0.93 | 0.43 | 1078 | -u, 1.00, 1.00 | walls | 0 | yes |
| bottom_cabinet_rhdbzv_0 | bottom_cabinet | entryway_0 | 0.37 | 0.31 | 784 | +u, 1.00, 1.00 | - | 0 |  |
| bottom_cabinet_rhdbzv_1 | bottom_cabinet | entryway_0 | 0.37 | 0.31 | 784 | +u, 1.00, 1.00 | - | 0 |  |
| nightstand_wbxekb_1 | nightstand | bedroom_0 | 0.35 | 0.30 | 755 | +u, 0.80, 1.00 | fixed_window | 0 |  |
| shelf_fjozhc_0 | shelf | bathroom_1 | 1.06 | 0.30 | 745 | +u, 2.50, 0.91 | walls | 0 |  |
| nightstand_wbxekb_0 | nightstand | bedroom_0 | 0.35 | 0.29 | 735 | +u, 0.80, 1.00 | fixed_window | 0 |  |
| shelf_pfusrd_0 | shelf | kitchen_0 | 1.83 | 0.28 | 702 | +u, 0.80, 1.00 | - | 0 |  |
| ottoman_wkmbra_0 | ottoman | entryway_0 | 0.34 | 0.22 | 550 | +u, 0.50, 1.00 | fixed_window | 0 |  |
| nightstand_wbxekb_2 | nightstand | childs_room_1 | 0.35 | 0.18 | 439 | +v, 0.50, 1.00 | walls | 0 |  |
| bottom_cabinet_no_top_gjeoer_0 | bottom_cabinet_no_top | kitchen_0 | 0.85 | 0.00 | 0 | -, -, - | - | 0 |  |
| bottom_cabinet_no_top_gjrero_0 | bottom_cabinet_no_top | utility_room_0 | 0.88 | 0.00 | 0 | -, -, - | - | 0 |  |
| bottom_cabinet_no_top_gjrero_1 | bottom_cabinet_no_top | utility_room_0 | 0.88 | 0.00 | 0 | -, -, - | - | 0 |  |
| bottom_cabinet_no_top_gjrero_2 | bottom_cabinet_no_top | utility_room_0 | 0.88 | 0.00 | 0 | -, -, - | - | 0 |  |
| bottom_cabinet_no_top_gjrero_3 | bottom_cabinet_no_top | utility_room_0 | 0.88 | 0.00 | 0 | -, -, - | - | 0 |  |
| bottom_cabinet_no_top_gjrero_4 | bottom_cabinet_no_top | utility_room_0 | 0.88 | 0.00 | 0 | -, -, - | - | 0 |  |
| bottom_cabinet_no_top_rkgjer_0 | bottom_cabinet_no_top | kitchen_0 | 0.85 | 0.00 | 0 | -, -, - | - | 0 |  |
| shelf_amyper_0 | shelf | corridor_0 | 2.45 | 0.00 | 0 | -, -, - | - | 0 |  |
| shelf_amyper_11 | shelf | corridor_0 | 2.45 | 0.00 | 0 | -, -, - | - | 0 |  |
| shelf_pfusrd_1 | shelf | kitchen_0 | 1.53 | 0.00 | 0 | -, -, - | - | 0 |  |

### large partners (16)

| name | category | rooms | face z | face cells | under ≥ 0.15 m | under ≥ 0.25 m | max clearance | best edge run 0.45 | supports ≤ 3 m (geodesic) |
|---|---|---|---|---|---|---|---|---|---|
| bed_gxfipj_0 | bed | bedroom_0 | 0.49 | 8160 | 0.05 | 0.00 | 0.17 | 1.65 | bench_pfjnnj_0 (0.8) |
| bed_ivdnny_0 | bed | childs_room_2 | 0.42 | 3167 | 0.88 | 0.00 | 0.24 | 1.55 | - |
| bed_rrcvaq_0 | bed | childs_room_0 | 0.49 | 5466 | 0.98 | 0.70 | 0.46 | 2.00 | shelf_fjozhc_0 (0.1); desk_xhjsub_0 (0.44) |
| bed_rrcvaq_1 | bed | childs_room_1 | 0.49 | 5365 | 0.99 | 0.70 | 0.46 | 1.45 | - |
| countertop_fjkase_0 | countertop | utility_room_0 | 0.92 | 4026 | 0.52 | 0.52 | 0.88 | 2.45 | bottom_cabinet_no_top_gjrero_1 (1.24); countertop_ikwqer_0 (1.32); bottom_cabinet_no_top_gjrero_0 (1.35); bottom_cabinet_no_top_gjrero_2 (1.37); bottom_cabinet_no_top_gjrero_4 (1.39); countertop_sjxber_0 (1.39); bottom_cabinet_no_top_gjrero_3 (1.43) |
| countertop_ikwqer_0 | countertop | utility_room_0 | 0.93 | 6620 | 0.13 | 0.13 | 0.89 | 4.45 | countertop_fjkase_0 (1.32) |
| countertop_kelker_0 | countertop | kitchen_0 | 0.89 | 3592 | 0.00 | 0.00 | 0.00 | 3.40 | bottom_cabinet_no_top_rkgjer_0 (0.42); countertop_kelzer_0 (0.42); bar_udatjt_0 (0.68); shelf_pfusrd_1 (1.07); shelf_pfusrd_0 (1.07); breakfast_table_rhjoby_0 (2.52) |
| countertop_kelzer_0 | countertop | kitchen_0 | 0.89 | 4453 | 0.42 | 0.42 | 0.35 | 3.10 | countertop_kelker_0 (0.42); bottom_cabinet_no_top_gjeoer_0 (0.57); bar_udatjt_0 (0.72) |
| countertop_sjxber_0 | countertop | utility_room_0 | 0.93 | 1078 | 0.00 | 0.00 | 0.00 | 1.00 | countertop_fjkase_0 (1.39) |
| desk_bhkhxo_0 | desk | childs_room_1 | 0.77 | 2940 | 0.54 | 0.54 | 0.67 | 1.85 | - |
| desk_nbpuns_0 | desk | childs_room_2 | 0.55 | 1447 | 0.85 | 0.79 | 0.52 | 1.10 | - |
| desk_xhjsub_0 | desk | childs_room_0 | 0.78 | 2088 | 0.07 | 0.07 | 0.76 | 1.50 | - |
| sofa_hiphpn_0 | sofa | living_room_0 | 0.40 | 5839 | 0.00 | 0.00 | 0.23 | 2.75 | coffee_table_rlsebe_0 (0.62) |
| sofa_lugrhk_0 | sofa | living_room_0 | 0.40 | 3981 | 0.00 | 0.00 | 0.01 | 2.35 | - |
| sofa_lugrhk_1 | sofa | living_room_0 | 0.40 | 3984 | 0.00 | 0.00 | 0.01 | 2.35 | coffee_table_rlsebe_0 (0.1); bar_udatjt_0 (0.6); bottom_cabinet_fancyy_0 (0.67); breakfast_table_rhjoby_0 (1.47); shelf_pfusrd_1 (1.81); shelf_pfusrd_0 (1.81) |
| sofa_wnzdke_0 | sofa | living_room_0 | 0.71 | 1191 | 0.00 | 0.00 | 0.01 | 1.15 | coffee_table_rlsebe_0 (0.14); bottom_cabinet_fancyy_0 (0.24); bar_udatjt_0 (0.82) |

### articulated objects (47)

| name | category/model | scale | rooms | joint | kind | status | best handle (kind, height m, width mm) | front free m (eroded) | canonical annotation | prior verdict | strip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bottom_cabinet_fancyy_0 | bottom_cabinet/fancyy | [1.0001, 1.0, 1.0] | kitchen_0 | j_door1 | door | handle | bar, 1.50, 22 | 1.25 (0.95) | NO | - | artic/strips/bottom_cabinet_fancyy_0.png |
| bottom_cabinet_fancyy_0 | bottom_cabinet/fancyy | [1.0001, 1.0, 1.0] | kitchen_0 | j_door2 | door | handle | bar, 0.81, 22 | 3.00 (0.95) | NO | - | artic/strips/bottom_cabinet_fancyy_0.png |
| bottom_cabinet_fancyy_0 | bottom_cabinet/fancyy | [1.0001, 1.0, 1.0] | kitchen_0 | j_door3 | door | handle | bar, 0.81, 22 | 1.25 (0.95) | NO | - | artic/strips/bottom_cabinet_fancyy_0.png |
| bottom_cabinet_no_top_gjeoer_0 | bottom_cabinet_no_top/gjeoer | [1.0, 1.0, 1.0] | kitchen_0 | j_doorleft1 | door | ungraspable | edge_door (not exposed), 0.79 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_no_top_gjeoer_0.png |
| bottom_cabinet_no_top_gjeoer_0 | bottom_cabinet_no_top/gjeoer | [1.0, 1.0, 1.0] | kitchen_0 | j_doorleft2 | door | ungraspable | edge_door (not exposed), 0.79 | 1.30 (1.00) | NO | - | artic/strips/bottom_cabinet_no_top_gjeoer_0.png |
| bottom_cabinet_no_top_gjeoer_0 | bottom_cabinet_no_top/gjeoer | [1.0, 1.0, 1.0] | kitchen_0 | j_doorright1 | door | ungraspable | edge_door (not exposed), 0.47 | 3.00 (1.00) | NO | - | artic/strips/bottom_cabinet_no_top_gjeoer_0.png |
| bottom_cabinet_no_top_gjeoer_0 | bottom_cabinet_no_top/gjeoer | [1.0, 1.0, 1.0] | kitchen_0 | j_doorright2 | door | ungraspable | edge_door (not exposed), 0.79 | 1.30 (1.00) | NO | - | artic/strips/bottom_cabinet_no_top_gjeoer_0.png |
| bottom_cabinet_no_top_gjrero_0 | bottom_cabinet_no_top/gjrero | [0.9998, 1.3945, 1.2035] | utility_room_0 | j_link_0 | door | handle | knob, 0.54, 23 | 2.00 (1.60) | NO | - | artic/strips/bottom_cabinet_no_top_gjrero_0.png |
| bottom_cabinet_no_top_gjrero_0 | bottom_cabinet_no_top/gjrero | [0.9998, 1.3945, 1.2035] | utility_room_0 | j_link_1 | door | handle | knob, 0.54, 23 | 2.00 (1.70) | NO | - | artic/strips/bottom_cabinet_no_top_gjrero_0.png |
| bottom_cabinet_no_top_gjrero_1 | bottom_cabinet_no_top/gjrero | [0.9998, 1.3945, 1.2035] | utility_room_0 | j_link_0 | door | handle | knob, 0.54, 23 | 1.90 (1.60) | NO | - | artic/strips/bottom_cabinet_no_top_gjrero_1.png |
| bottom_cabinet_no_top_gjrero_1 | bottom_cabinet_no_top/gjrero | [0.9998, 1.3945, 1.2035] | utility_room_0 | j_link_1 | door | handle | knob, 0.54, 23 | 1.90 (1.60) | NO | - | artic/strips/bottom_cabinet_no_top_gjrero_1.png |
| bottom_cabinet_no_top_gjrero_2 | bottom_cabinet_no_top/gjrero | [0.9998, 1.3945, 1.2035] | utility_room_0 | j_link_0 | door | handle | knob, 0.54, 23 | 2.00 (1.70) | NO | - | artic/strips/bottom_cabinet_no_top_gjrero_2.png |
| bottom_cabinet_no_top_gjrero_2 | bottom_cabinet_no_top/gjrero | [0.9998, 1.3945, 1.2035] | utility_room_0 | j_link_1 | door | handle | knob, 0.54, 23 | 2.00 (1.60) | NO | - | artic/strips/bottom_cabinet_no_top_gjrero_2.png |
| bottom_cabinet_no_top_gjrero_3 | bottom_cabinet_no_top/gjrero | [0.9998, 1.3945, 1.2035] | utility_room_0 | j_link_0 | door | handle | knob, 0.54, 23 | 2.00 (1.70) | NO | - | artic/strips/bottom_cabinet_no_top_gjrero_3.png |
| bottom_cabinet_no_top_gjrero_3 | bottom_cabinet_no_top/gjrero | [0.9998, 1.3945, 1.2035] | utility_room_0 | j_link_1 | door | handle | knob, 0.54, 23 | 2.00 (1.70) | NO | - | artic/strips/bottom_cabinet_no_top_gjrero_3.png |
| bottom_cabinet_no_top_gjrero_4 | bottom_cabinet_no_top/gjrero | [0.661, 1.3524, 1.2035] | utility_room_0 | j_link_0 | door | handle | knob, 0.54, 22 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_no_top_gjrero_4.png |
| bottom_cabinet_no_top_gjrero_4 | bottom_cabinet_no_top/gjrero | [0.661, 1.3524, 1.2035] | utility_room_0 | j_link_1 | door | handle | knob, 0.54, 22 | 3.00 (0.00) | NO | - | artic/strips/bottom_cabinet_no_top_gjrero_4.png |
| bottom_cabinet_no_top_rkgjer_0 | bottom_cabinet_no_top/rkgjer | [1.0, 1.0, 0.9999] | kitchen_0 | j_doorleft1 | door | ungraspable | edge_door (not exposed), 0.79 | 1.30 (1.00) | NO | - | artic/strips/bottom_cabinet_no_top_rkgjer_0.png |
| bottom_cabinet_no_top_rkgjer_0 | bottom_cabinet_no_top/rkgjer | [1.0, 1.0, 0.9999] | kitchen_0 | j_doorleft2 | door | ungraspable | edge_door (not exposed), 0.79 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_no_top_rkgjer_0.png |
| bottom_cabinet_no_top_rkgjer_0 | bottom_cabinet_no_top/rkgjer | [1.0, 1.0, 0.9999] | kitchen_0 | j_doorright1 | door | ungraspable | edge_door (not exposed), 0.79 | 1.30 (1.00) | NO | - | artic/strips/bottom_cabinet_no_top_rkgjer_0.png |
| bottom_cabinet_no_top_rkgjer_0 | bottom_cabinet_no_top/rkgjer | [1.0, 1.0, 0.9999] | kitchen_0 | j_doorright2 | door | ungraspable | edge_door (not exposed), 0.79 | 3.00 (0.00) | NO | - | artic/strips/bottom_cabinet_no_top_rkgjer_0.png |
| bottom_cabinet_rhdbzv_0 | bottom_cabinet/rhdbzv | [0.8493, 0.4163, 0.9044] | entryway_0 | j_left_drawer | drawer | handle | bar, 0.34, 7 | 1.70 (1.40) | NO | - | artic/strips/bottom_cabinet_rhdbzv_0.png |
| bottom_cabinet_rhdbzv_0 | bottom_cabinet/rhdbzv | [0.8493, 0.4163, 0.9044] | entryway_0 | j_right_drawer | drawer | handle | bar, 0.34, 7 | 0.40 (1.40) | NO | - | artic/strips/bottom_cabinet_rhdbzv_0.png |
| bottom_cabinet_rhdbzv_1 | bottom_cabinet/rhdbzv | [0.8493, 0.4163, 0.9044] | entryway_0 | j_left_drawer | drawer | handle | bar, 0.34, 7 | 1.75 (1.15) | NO | - | artic/strips/bottom_cabinet_rhdbzv_1.png |
| bottom_cabinet_rhdbzv_1 | bottom_cabinet/rhdbzv | [0.8493, 0.4163, 0.9044] | entryway_0 | j_right_drawer | drawer | handle | bar, 0.34, 7 | 3.00 (1.55) | NO | - | artic/strips/bottom_cabinet_rhdbzv_1.png |
| bottom_cabinet_xqmipa_0 | bottom_cabinet/xqmipa | [0.548, 0.5592, 0.9078] | bathroom_0 | j_bottom_left_drawer | drawer | handle | bar, 0.34, 9 | 3.00 (0.40) | NO | - | artic/strips/bottom_cabinet_xqmipa_0.png |
| bottom_cabinet_xqmipa_0 | bottom_cabinet/xqmipa | [0.548, 0.5592, 0.9078] | bathroom_0 | j_bottom_right_drawer | drawer | handle | bar, 0.34, 9 | 3.00 (0.40) | NO | - | artic/strips/bottom_cabinet_xqmipa_0.png |
| bottom_cabinet_xqmipa_0 | bottom_cabinet/xqmipa | [0.548, 0.5592, 0.9078] | bathroom_0 | j_left_left_leaf | door | handle | bar, 0.64, 9 | 1.85 (0.85) | NO | - | artic/strips/bottom_cabinet_xqmipa_0.png |
| bottom_cabinet_xqmipa_0 | bottom_cabinet/xqmipa | [0.548, 0.5592, 0.9078] | bathroom_0 | j_left_right_leaf | door | handle | bar, 0.64, 9 | 3.00 (1.05) | NO | - | artic/strips/bottom_cabinet_xqmipa_0.png |
| bottom_cabinet_xqmipa_0 | bottom_cabinet/xqmipa | [0.548, 0.5592, 0.9078] | bathroom_0 | j_middle_left_drawer | drawer | handle | bar, 0.49, 9 | 3.00 (0.40) | NO | - | artic/strips/bottom_cabinet_xqmipa_0.png |
| bottom_cabinet_xqmipa_0 | bottom_cabinet/xqmipa | [0.548, 0.5592, 0.9078] | bathroom_0 | j_middle_right_drawer | drawer | handle | bar, 0.49, 9 | 3.00 (0.40) | NO | - | artic/strips/bottom_cabinet_xqmipa_0.png |
| bottom_cabinet_xqmipa_0 | bottom_cabinet/xqmipa | [0.548, 0.5592, 0.9078] | bathroom_0 | j_right_left_leaf | door | handle | bar, 0.64, 9 | 0.65 (0.35) | NO | - | artic/strips/bottom_cabinet_xqmipa_0.png |
| bottom_cabinet_xqmipa_0 | bottom_cabinet/xqmipa | [0.548, 0.5592, 0.9078] | bathroom_0 | j_right_right_leaf | door | handle | bar, 0.64, 9 | 0.65 (0.00) | NO | - | artic/strips/bottom_cabinet_xqmipa_0.png |
| bottom_cabinet_xqmipa_0 | bottom_cabinet/xqmipa | [0.548, 0.5592, 0.9078] | bathroom_0 | j_top_left_drawer | drawer | handle | bar, 0.64, 9 | 3.00 (0.40) | NO | - | artic/strips/bottom_cabinet_xqmipa_0.png |
| bottom_cabinet_xqmipa_0 | bottom_cabinet/xqmipa | [0.548, 0.5592, 0.9078] | bathroom_0 | j_top_right_drawer | drawer | handle | bar, 0.64, 9 | 3.00 (0.40) | NO | - | artic/strips/bottom_cabinet_xqmipa_0.png |
| clothes_dryer_xsuyua_0 | clothes_dryer/xsuyua | [1.1082, 0.9749, 1.0521] | utility_room_0 | j_link_0 | door | ungraspable | edge_door (not exposed), 0.46 | 1.85 (1.55) | yes | none/infeasible_2_reviews | artic/strips/clothes_dryer_xsuyua_0.png |
| dishwasher_ynrhuk_0 | dishwasher/ynrhuk | [1.0597, 1.1054, 1.0615] | kitchen_0 | j_link_2 | flap | handle | bar, 0.76, 24 | 1.30 (1.00) | yes | A/grasped_not_opened | artic/strips/dishwasher_ynrhuk_0.png |
| door_crtrpa_0 | door/crtrpa | [1.0001, 1.0, 1.0] | sauna_0,garden_0 | j_leaf | door | handle | bar, 1.04, 20 | 1.65 (1.35) | yes | B/grasped_not_opened | artic/strips/door_crtrpa_0.png |
| door_eduavq_0 | door/eduavq | [0.9998, 1.0, 1.0] | garage_0,garden_0 | j_leaf | door | handle | knob, 1.11, 20 | 3.00 (3.00) | NO | - | artic/strips/door_eduavq_0.png |
| door_fnvpyu_0 | door/fnvpyu | [1.0003, 1.0, 1.0] | corridor_0,garden_0 | j_leaf | door | handle | knob, 1.10, 20 | 3.00 (3.00) | NO | - | artic/strips/door_fnvpyu_0.png |
| door_iuuykr_0 | door/iuuykr | [1.0001, 1.0, 1.0] | garage_0,garden_0 | j_left_leaf | door | handle | bar, 0.82, 19 | 2.00 (1.70) | NO | - | artic/strips/door_iuuykr_0.png |
| door_iuuykr_0 | door/iuuykr | [1.0001, 1.0, 1.0] | garage_0,garden_0 | j_right_leaf | door | handle | knob, 0.89, 19 | 2.05 (1.75) | NO | - | artic/strips/door_iuuykr_0.png |
| door_kwbnhy_0 | door/kwbnhy | [1.0002, 0.9999, 1.0] | childs_room_0,garden_0 | j_leaf | door | handle | knob, 1.11, 20 | 3.00 (3.00) | NO | - | artic/strips/door_kwbnhy_0.png |
| door_kwbnhy_1 | door/kwbnhy | [1.0077, 0.9999, 1.0] | corridor_0,garden_0 | j_leaf | door | handle | knob, 1.11, 20 | 3.00 (3.00) | NO | - | artic/strips/door_kwbnhy_1.png |
| door_kwbnhy_2 | door/kwbnhy | [0.9947, 0.9999, 1.0] | childs_room_2,garden_0 | j_leaf | door | handle | knob, 1.10, 20 | 1.80 (0.70) | NO | - | artic/strips/door_kwbnhy_2.png |
| door_kwbnhy_3 | door/kwbnhy | [1.089, 0.9999, 1.0] | childs_room_1,garden_0 | j_leaf | door | handle | knob, 1.10, 20 | 3.00 (3.00) | NO | - | artic/strips/door_kwbnhy_3.png |
| door_tfdrhh_0 | door/tfdrhh | [1.0001, 1.0, 1.0] | entryway_0,garden_0 | j_leaf | door | handle | knob, 1.09, 34 | 1.85 (1.55) | NO | - | artic/strips/door_tfdrhh_0.png |
| door_tonlzz_0 | door/tonlzz | [1.0001, 1.0, 1.0] | dining_room_0,living_room_1 | j_leaf | door | handle | bar, 1.04, 20 | 3.00 (1.20) | NO | - | artic/strips/door_tonlzz_0.png |
| fridge_dszchb_0 | fridge/dszchb | [1.0292, 0.9617, 1.0212] | kitchen_0 | j_link_0 | door | handle | bar, 1.12, 30 | 1.30 (1.00) | yes | A/unreachable | artic/strips/fridge_dszchb_0.png |
| microwave_abzvij_0 | microwave/abzvij | [1.0001, 0.9999, 0.9999] | kitchen_0 | j_link_0 | door | ungraspable | edge_door (not exposed), 1.08 | 1.35 (1.05) | NO | - | artic/strips/microwave_abzvij_0.png |
| openable_window_eyvzfu_0 | openable_window/eyvzfu | [0.9999, 1.0, 1.0] | bathroom_0,garden_0 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.59 | 3.00 (3.00) | yes | none/collision | artic/strips/openable_window_eyvzfu_0.png |
| oven_ffitak_0 | oven/ffitak | [0.9999, 1.0001, 1.0001] | kitchen_0 | j_door | flap | handle | bar, 1.30, 9 | 1.25 (0.95) | yes | B/unreachable | artic/strips/oven_ffitak_0.png |
| oven_ffitak_0 | oven/ffitak | [0.9999, 1.0001, 1.0001] | kitchen_0 | j_rack1 | drawer | ungraspable | edge_drawer_top (not exposed), 1.04 | 1.30 (1.00) | yes | F/unreachable | artic/strips/oven_ffitak_0.png |
| oven_ffitak_0 | oven/ffitak | [0.9999, 1.0001, 1.0001] | kitchen_0 | j_rack2 | drawer | ungraspable | edge_drawer_top (not exposed), 0.93 | 1.30 (1.00) | yes | F/unreachable | artic/strips/oven_ffitak_0.png |
| sliding_door_lrvyvc_0 | sliding_door/lrvyvc | [0.9999, 1.0, 1.0] | living_room_0,garden_0 | j_movingpart | drawer | edge | edge_drawer_top, 2.68 | 0.00 (0.00) | yes | none/model_error | artic/strips/sliding_door_lrvyvc_0.png |
| sliding_door_tprpvb_0 | sliding_door/tprpvb | [1.0007, 1.0, 1.0] | corridor_0,bedroom_0 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 0.00 (0.00) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_0.png |
| sliding_door_tprpvb_1 | sliding_door/tprpvb | [1.0007, 1.0632, 1.0] | corridor_0,childs_room_1 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 3.00 (0.00) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_1.png |
| sliding_door_tprpvb_10 | sliding_door/tprpvb | [1.0007, 1.0019, 1.0] | bedroom_0,bathroom_1 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 3.00 (3.00) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_10.png |
| sliding_door_tprpvb_11 | sliding_door/tprpvb | [1.0007, 1.0888, 1.0] | corridor_0,childs_room_0 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 3.00 (3.00) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_11.png |
| sliding_door_tprpvb_12 | sliding_door/tprpvb | [1.0007, 1.2042, 1.0] | corridor_0 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 0.00 (0.00) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_12.png |
| sliding_door_tprpvb_2 | sliding_door/tprpvb | [1.0007, 0.8283, 1.0] | childs_room_2,closet_3 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 0.00 (0.00) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_2.png |
| sliding_door_tprpvb_3 | sliding_door/tprpvb | [1.0007, 0.9632, 1.0] | childs_room_0,closet_2 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 0.00 (0.00) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_3.png |
| sliding_door_tprpvb_4 | sliding_door/tprpvb | [1.0007, 0.7484, 1.0] | bedroom_0,closet_0 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 0.00 (3.00) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_4.png |
| sliding_door_tprpvb_5 | sliding_door/tprpvb | [1.0007, 1.0908, 1.0] | corridor_0,bathroom_0 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 0.00 (2.40) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_5.png |
| sliding_door_tprpvb_6 | sliding_door/tprpvb | [1.0007, 1.0997, 1.0] | corridor_0,childs_room_2 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 3.00 (2.85) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_6.png |
| sliding_door_tprpvb_7 | sliding_door/tprpvb | [1.0007, 1.1156, 1.0] | corridor_0,utility_room_0 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 3.00 (3.00) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_7.png |
| sliding_door_tprpvb_8 | sliding_door/tprpvb | [1.0007, 0.7851, 1.0] | childs_room_1,closet_4 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 0.00 (0.00) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_8.png |
| sliding_door_tprpvb_9 | sliding_door/tprpvb | [1.0007, 0.9854, 1.0] | bedroom_0,closet_1 | j_movingpart | drawer | ungraspable | edge_drawer_top (not exposed), 2.28 | 0.00 (0.00) | yes | none/unreachable | artic/strips/sliding_door_tprpvb_9.png |
| toilet_udiezm_0 | toilet/udiezm | [1.0, 1.0, 1.0] | bathroom_1 | j_flap | flap | edge | edge_lid, 0.90 | 0.35 (0.00) | NO | - | artic/strips/toilet_udiezm_0.png |
| toilet_udiezm_1 | toilet/udiezm | [1.0, 1.0, 1.0] | bathroom_0 | j_flap | flap | edge | edge_lid, 0.90 | 3.00 (2.85) | NO | - | artic/strips/toilet_udiezm_1.png |
| top_cabinet_lkxmne_0 | top_cabinet/lkxmne | [0.9999, 1.0, 1.0] | kitchen_0 | j_leftdoor | door | ungraspable | edge_door (not exposed), 1.79 | 0.00 (0.00) | NO | - | artic/strips/top_cabinet_lkxmne_0.png |
| top_cabinet_lkxmne_0 | top_cabinet/lkxmne | [0.9999, 1.0, 1.0] | kitchen_0 | j_rightdoor | door | ungraspable | edge_door (not exposed), 1.50 | 3.00 (0.00) | NO | - | artic/strips/top_cabinet_lkxmne_0.png |
| top_cabinet_lkxmne_1 | top_cabinet/lkxmne | [0.9999, 1.0, 1.0] | kitchen_0 | j_leftdoor | door | ungraspable | edge_door (not exposed), 1.79 | 3.00 (3.00) | NO | - | artic/strips/top_cabinet_lkxmne_1.png |
| top_cabinet_lkxmne_1 | top_cabinet/lkxmne | [0.9999, 1.0, 1.0] | kitchen_0 | j_rightdoor | door | ungraspable | edge_door (not exposed), 1.50 | 1.55 (1.25) | NO | - | artic/strips/top_cabinet_lkxmne_1.png |
| top_cabinet_lkxmne_2 | top_cabinet/lkxmne | [0.9999, 1.0, 1.0] | kitchen_0 | j_leftdoor | door | ungraspable | edge_door (not exposed), 1.79 | 1.55 (1.25) | NO | - | artic/strips/top_cabinet_lkxmne_2.png |
| top_cabinet_lkxmne_2 | top_cabinet/lkxmne | [0.9999, 1.0, 1.0] | kitchen_0 | j_rightdoor | door | ungraspable | edge_door (not exposed), 1.50 | 1.75 (1.25) | NO | - | artic/strips/top_cabinet_lkxmne_2.png |
| washer_ynwamu_0 | washer/ynwamu | [1.0, 1.0, 1.0001] | utility_room_0 | j_door | door | handle | knob, 0.30, 64 | 2.00 (1.70) | NO | - | artic/strips/washer_ynwamu_0.png |
Review needed (no canonical handle annotation, graspable joint): 21 objects in 16 distinct (model, scale) groups: bottom_cabinet/fancyy x1 scale [1.0, 1.0, 1.0]; bottom_cabinet/rhdbzv x2 scale [0.85, 0.42, 0.9]; bottom_cabinet/xqmipa x1 scale [0.55, 0.56, 0.91]; bottom_cabinet_no_top/gjrero x1 scale [0.66, 1.35, 1.2]; bottom_cabinet_no_top/gjrero x4 scale [1.0, 1.39, 1.2]; door/eduavq x1 scale [1.0, 1.0, 1.0]; door/fnvpyu x1 scale [1.0, 1.0, 1.0]; door/iuuykr x1 scale [1.0, 1.0, 1.0]; door/kwbnhy x1 scale [0.99, 1.0, 1.0]; door/kwbnhy x1 scale [1.0, 1.0, 1.0]; door/kwbnhy x1 scale [1.01, 1.0, 1.0]; door/kwbnhy x1 scale [1.09, 1.0, 1.0]; door/tfdrhh x1 scale [1.0, 1.0, 1.0]; door/tonlzz x1 scale [1.0, 1.0, 1.0]; toilet/udiezm x2 scale [1.0, 1.0, 1.0]; washer/ynwamu x1 scale [1.0, 1.0, 1.0]

## house_double_floor_upper

### supports (15)

| name | category | rooms | top z | area m² | free cells | best edge run 0.45 (edge, m, free frac) | blocked in front | chairs ≤ 1 m | in union |
|---|---|---|---|---|---|---|---|---|---|
| desk_uqcmzf_0 | desk | bedroom_2 | 0.72 | 1.75 | 4382 | +v, 1.60, 1.00 | - | 1 |  |
| coffee_table_osroux_0 | coffee_table | living_room_0 | 0.36 | 1.17 | 2920 | +u, 1.45, 1.00 | - | 0 |  |
| desk_aduafr_0 | desk | bedroom_0 | 0.77 | 0.84 | 2100 | +u, 1.50, 1.00 | - | 0 | yes |
| desk_iotfzl_0 | desk | living_room_0 | 0.74 | 0.72 | 1791 | -u, 1.40, 1.00 | - | 1 | yes |
| bottom_cabinet_jhymlr_0 | bottom_cabinet | bathroom_0 | 0.88 | 0.45 | 1113 | +u, 1.20, 1.00 | - | 0 | yes |
| nightstand_roxnbe_1 | nightstand | bedroom_1 | 0.70 | 0.42 | 1053 | +u, 0.80, 1.00 | - | 0 |  |
| nightstand_roxnbe_0 | nightstand | bedroom_1 | 0.70 | 0.41 | 1028 | +u, 0.80, 1.00 | - | 0 |  |
| nightstand_uzxrlg_0 | nightstand | bedroom_2 | 0.35 | 0.34 | 838 | +u, 0.80, 1.00 | - | 0 |  |
| bottom_cabinet_nddvba_0 | bottom_cabinet | bedroom_2 | 0.72 | 0.31 | 779 | +u, 0.80, 1.00 | - | 1 | yes |
| nightstand_aykehx_0 | nightstand | bedroom_0 | 0.31 | 0.31 | 773 | +u, 0.65, 1.00 | - | 0 |  |
| shelf_sapyzl_1 | shelf | bedroom_1 | 0.97 | 0.22 | 550 | -u, 1.55, 1.00 | - | 0 |  |
| shelf_sapyzl_0 | shelf | bedroom_1 | 1.50 | 0.22 | 540 | -u, 1.55, 1.00 | - | 0 |  |
| ottoman_qsnawv_0 | ottoman | bedroom_1 | 0.43 | 0.19 | 469 | +u, 0.45, 1.00 | - | 0 |  |
| serving_cart_vvtcby_0 | serving_cart | bathroom_0 | 0.58 | 0.16 | 393 | +u, 0.50, 1.00 | - | 0 |  |
| bottom_cabinet_no_top_ncgmqr_0 | bottom_cabinet_no_top | bedroom_2 | 0.67 | 0.00 | 2 | +u, 0.05, 1.00 | - | 1 |  |

### large partners (7)

| name | category | rooms | face z | face cells | under ≥ 0.15 m | under ≥ 0.25 m | max clearance | best edge run 0.45 | supports ≤ 3 m (geodesic) |
|---|---|---|---|---|---|---|---|---|---|
| bed_klcjcy_0 | bed | bedroom_0 | 0.45 | 5982 | 0.00 | 0.00 | 0.15 | 1.60 | desk_aduafr_0 (1.34) |
| bed_siuxgz_0 | bed | bedroom_1 | 0.47 | 9132 | 0.00 | 0.00 | 0.06 | 1.65 | shelf_sapyzl_1 (0.1); shelf_sapyzl_0 (0.1); bottom_cabinet_jhymlr_0 (0.1); nightstand_uzxrlg_0 (0.84) |
| bed_tjfzsb_0 | bed | bedroom_2 | 0.47 | 6799 | 0.09 | 0.00 | 0.20 | 2.00 | bottom_cabinet_nddvba_0 (0.64); bottom_cabinet_no_top_ncgmqr_0 (0.72); ottoman_qsnawv_0 (0.74); shelf_sapyzl_1 (2.26); shelf_sapyzl_0 (2.26) |
| desk_aduafr_0 | desk | bedroom_0 | 0.77 | 2100 | 0.38 | 0.38 | 0.75 | 1.50 | coffee_table_osroux_0 (0.67); nightstand_aykehx_0 (0.68); serving_cart_vvtcby_0 (1.41) |
| desk_iotfzl_0 | desk | living_room_0 | 0.74 | 1791 | 0.95 | 0.95 | 0.71 | 1.40 | coffee_table_osroux_0 (0.14) |
| desk_uqcmzf_0 | desk | bedroom_2 | 0.72 | 4382 | 0.50 | 0.50 | 0.70 | 1.60 | nightstand_uzxrlg_0 (1.38) |
| sofa_ukfaaz_0 | sofa | living_room_0 | 0.46 | 7451 | 0.00 | 0.00 | 0.46 | 3.25 | desk_iotfzl_0 (0.84); desk_aduafr_0 (2.17) |

### articulated objects (10)

| name | category/model | scale | rooms | joint | kind | status | best handle (kind, height m, width mm) | front free m (eroded) | canonical annotation | prior verdict | strip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [1.6129, 1.7242, 2.0689] | bathroom_0 | j_link_0 | drawer | handle | bar, 0.42, 5 | 2.70 (0.50) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [1.6129, 1.7242, 2.0689] | bathroom_0 | j_link_1 | drawer | handle | bar, 0.69, 5 | 2.70 (0.50) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [1.6129, 1.7242, 2.0689] | bathroom_0 | j_link_2 | drawer | handle | bar, 0.87, 5 | 2.70 (0.50) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [1.6129, 1.7242, 2.0689] | bathroom_0 | j_link_3 | drawer | handle | bar, 0.42, 5 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [1.6129, 1.7242, 2.0689] | bathroom_0 | j_link_4 | drawer | handle | bar, 0.69, 5 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [1.6129, 1.7242, 2.0689] | bathroom_0 | j_link_5 | drawer | handle | bar, 0.87, 5 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_nddvba_0 | bottom_cabinet/nddvba | [0.9191, 1.793, 1.7552] | bedroom_2 | j_link_1 | drawer | handle | bar, 0.18, 15 | 1.60 (1.30) | NO | - | artic/strips/bottom_cabinet_nddvba_0.png |
| bottom_cabinet_nddvba_0 | bottom_cabinet/nddvba | [0.9191, 1.793, 1.7552] | bedroom_2 | j_link_2 | drawer | handle | bar, 0.42, 15 | 1.60 (1.30) | NO | - | artic/strips/bottom_cabinet_nddvba_0.png |
| bottom_cabinet_nddvba_0 | bottom_cabinet/nddvba | [0.9191, 1.793, 1.7552] | bedroom_2 | j_link_3 | drawer | handle | bar, 0.54, 15 | 1.60 (1.30) | NO | - | artic/strips/bottom_cabinet_nddvba_0.png |
| bottom_cabinet_nddvba_0 | bottom_cabinet/nddvba | [0.9191, 1.793, 1.7552] | bedroom_2 | j_link_4 | drawer | handle | knob, 0.66, 13 | 1.60 (1.30) | NO | - | artic/strips/bottom_cabinet_nddvba_0.png |
| bottom_cabinet_no_top_ncgmqr_0 | bottom_cabinet_no_top/ncgmqr | [0.9999, 1.0, 1.0] | bedroom_2 | j_drawer0 | drawer | handle | bar, 0.66, 14 | 2.95 (2.65) | NO | - | artic/strips/bottom_cabinet_no_top_ncgmqr_0.png |
| bottom_cabinet_no_top_ncgmqr_0 | bottom_cabinet_no_top/ncgmqr | [0.9999, 1.0, 1.0] | bedroom_2 | j_drawer1 | drawer | handle | bar, 0.48, 14 | 2.95 (2.65) | NO | - | artic/strips/bottom_cabinet_no_top_ncgmqr_0.png |
| bottom_cabinet_no_top_ncgmqr_0 | bottom_cabinet_no_top/ncgmqr | [0.9999, 1.0, 1.0] | bedroom_2 | j_drawer2 | drawer | handle | bar, 0.29, 14 | 2.95 (2.65) | NO | - | artic/strips/bottom_cabinet_no_top_ncgmqr_0.png |
| door_dlncrt_0 | door/dlncrt | [1.0002, 1.0, 1.0] | bathroom_0 | j_leaf | door | handle | knob, 1.02, 38 | 0.00 (0.00) | yes | A/model_error | artic/strips/door_dlncrt_0.png |
| door_pevmgl_0 | door/pevmgl | [1.0004, 1.0, 1.0] | bathroom_0 | j_leaf | door | handle | knob, 1.02, 38 | 1.60 (0.00) | NO | - | artic/strips/door_pevmgl_0.png |
| door_pxhbim_0 | door/pxhbim | [1.0003, 1.0, 1.0] | living_room_0,bathroom_1 | j_leaf | door | handle | knob, 1.10, 29 | 2.25 (1.95) | NO | - | artic/strips/door_pxhbim_0.png |
| door_pxhbim_1 | door/pxhbim | [1.0003, 1.0, 1.0] | living_room_0,bedroom_0 | j_leaf | door | handle | knob, 1.10, 29 | 3.00 (3.00) | NO | - | artic/strips/door_pxhbim_1.png |
| door_pxhbim_2 | door/pxhbim | [1.0003, 1.0, 1.0] | living_room_0,bedroom_1 | j_leaf | door | handle | knob, 1.10, 29 | 3.00 (3.00) | NO | - | artic/strips/door_pxhbim_2.png |
| door_pxhbim_3 | door/pxhbim | [1.0003, 1.0, 1.0] | living_room_0,bedroom_2 | j_leaf | door | handle | knob, 1.10, 29 | 1.05 (0.75) | NO | - | artic/strips/door_pxhbim_3.png |
| toilet_udiezm_0 | toilet/udiezm | [1.0, 1.0, 1.0] | bathroom_0 | j_flap | flap | edge | edge_lid, 0.90 | 0.00 (0.00) | NO | - | artic/strips/toilet_udiezm_0.png |
Review needed (no canonical handle annotation, graspable joint): 9 objects in 6 distinct (model, scale) groups: bottom_cabinet/jhymlr x1 scale [1.61, 1.72, 2.07]; bottom_cabinet/nddvba x1 scale [0.92, 1.79, 1.76]; bottom_cabinet_no_top/ncgmqr x1 scale [1.0, 1.0, 1.0]; door/pevmgl x1 scale [1.0, 1.0, 1.0]; door/pxhbim x4 scale [1.0, 1.0, 1.0]; toilet/udiezm x1 scale [1.0, 1.0, 1.0]

## house_double_floor_lower

### supports (8)

| name | category | rooms | top z | area m² | free cells | best edge run 0.45 (edge, m, free frac) | blocked in front | chairs ≤ 1 m | in union |
|---|---|---|---|---|---|---|---|---|---|
| breakfast_table_xftrki_0 | breakfast_table | kitchen_0 | 0.68 | 1.38 | 3450 | +u, 1.50, 1.00 | - | 4 |  |
| coffee_table_koagbh_0 | coffee_table | living_room_0 | 0.42 | 1.36 | 3403 | +u, 1.65, 1.00 | - | 0 |  |
| bar_egwapq_0 | bar | kitchen_0 | 0.89 | 1.10 | 2756 | -u, 2.45, 1.00 | walls | 0 |  |
| bottom_cabinet_rhdbzv_0 | bottom_cabinet | living_room_0 | 0.41 | 0.89 | 2223 | +u, 2.35, 1.00 | - | 0 |  |
| bar_byvbuc_0 | bar | kitchen_0 | 0.89 | 0.67 | 1686 | -u, 2.10, 1.00 | walls | 0 |  |
| countertop_tpuwys_0 | countertop | living_room_0 | 1.38 | 0.66 | 1653 | -u, 1.75, 1.00 | - | 0 |  |
| shelf_vanltx_0 | shelf | living_room_0 | 1.30 | 0.43 | 1080 | -u, 1.20, 1.00 | walls | 0 |  |
| bottom_cabinet_qacthv_0 | bottom_cabinet | bathroom_0 | 0.81 | 0.29 | 725 | +u, 1.15, 1.00 | walls | 0 |  |

### large partners (2)

| name | category | rooms | face z | face cells | under ≥ 0.15 m | under ≥ 0.25 m | max clearance | best edge run 0.45 | supports ≤ 3 m (geodesic) |
|---|---|---|---|---|---|---|---|---|---|
| countertop_tpuwys_0 | countertop | living_room_0 | 1.38 | 1653 | 0.00 | 0.00 | 0.00 | 1.75 | coffee_table_koagbh_0 (1.25); shelf_vanltx_0 (1.37) |
| sofa_frlxgz_0 | sofa | living_room_0 | 0.46 | 6775 | 0.02 | 0.01 | 0.43 | 3.30 | breakfast_table_xftrki_0 (2.41) |

### articulated objects (16)

| name | category/model | scale | rooms | joint | kind | status | best handle (kind, height m, width mm) | front free m (eroded) | canonical annotation | prior verdict | strip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bottom_cabinet_qacthv_0 | bottom_cabinet/qacthv | [1.6584, 1.6618, 2.2723] | bathroom_0 | j_link_0 | drawer | ungraspable | edge_drawer_top (not exposed), 0.28 | 1.15 (0.85) | NO | - | artic/strips/bottom_cabinet_qacthv_0.png |
| bottom_cabinet_qacthv_0 | bottom_cabinet/qacthv | [1.6584, 1.6618, 2.2723] | bathroom_0 | j_link_1 | drawer | ungraspable | edge_drawer_top (not exposed), 0.28 | 1.15 (0.85) | NO | - | artic/strips/bottom_cabinet_qacthv_0.png |
| bottom_cabinet_qacthv_0 | bottom_cabinet/qacthv | [1.6584, 1.6618, 2.2723] | bathroom_0 | j_link_2 | drawer | ungraspable | edge_drawer_top (not exposed), 0.27 | 1.25 (0.95) | NO | - | artic/strips/bottom_cabinet_qacthv_0.png |
| bottom_cabinet_rhdbzv_0 | bottom_cabinet/rhdbzv | [0.9999, 1.0, 0.9999] | living_room_0 | j_left_drawer | drawer | handle | bar, 0.37, 8 | 2.05 (1.75) | NO | - | artic/strips/bottom_cabinet_rhdbzv_0.png |
| bottom_cabinet_rhdbzv_0 | bottom_cabinet/rhdbzv | [0.9999, 1.0, 0.9999] | living_room_0 | j_right_drawer | drawer | handle | bar, 0.37, 8 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_rhdbzv_0.png |
| dishwasher_ynrhuk_0 | dishwasher/ynrhuk | [1.1037, 1.1392, 1.0696] | kitchen_0 | j_link_2 | flap | handle | bar, 0.77, 24 | 2.65 (2.35) | yes | A/grasped_not_opened | artic/strips/dishwasher_ynrhuk_0.png |
| door_bexenl_0 | door/bexenl | [0.9998, 1.0, 1.0] | garage_0,corridor_0 | j_leaf | door | handle | bar, 1.10, 36 | 1.45 (1.15) | yes | A/opened | artic/strips/door_bexenl_0.png |
| door_bexenl_1 | door/bexenl | [0.9984, 1.0, 1.0] | corridor_0,bathroom_0 | j_leaf | door | handle | bar, 1.10, 36 | 1.40 (1.10) | yes | A/opened | artic/strips/door_bexenl_1.png |
| door_vudhlc_0 | door/vudhlc | [0.9998, 1.0, 1.0] | corridor_0,garden_0 | j_leaf | door | handle | knob, 1.27, 20 | 2.35 (2.05) | NO | - | artic/strips/door_vudhlc_0.png |
| door_vudhlc_1 | door/vudhlc | [0.9998, 1.0, 1.0] | corridor_0,garden_0 | j_leaf | door | handle | knob, 1.27, 20 | 2.00 (1.70) | NO | - | artic/strips/door_vudhlc_1.png |
| door_vudhlc_2 | door/vudhlc | [0.9998, 1.0, 1.0] | kitchen_0,garden_0 | j_leaf | door | handle | knob, 1.27, 20 | 3.00 (3.00) | NO | - | artic/strips/door_vudhlc_2.png |
| door_vudhlc_3 | door/vudhlc | [0.9998, 1.0, 1.0] | living_room_0,garden_0 | j_leaf | door | handle | knob, 1.27, 20 | 3.00 (3.00) | NO | - | artic/strips/door_vudhlc_3.png |
| fridge_petcxr_0 | fridge/petcxr | [1.0, 1.0, 1.0] | kitchen_0 | j_left_door | door | handle | bar, 1.58, 32 | 3.00 (3.00) | NO | - | artic/strips/fridge_petcxr_0.png |
| fridge_petcxr_0 | fridge/petcxr | [1.0, 1.0, 1.0] | kitchen_0 | j_right_door | door | handle | bar, 1.58, 32 | 0.95 (0.45) | NO | - | artic/strips/fridge_petcxr_0.png |
| microwave_hjjxmi_0 | microwave/hjjxmi | [0.9999, 1.0, 0.9998] | kitchen_0 | j_link_0 | door | handle | bar, 1.00, 14 | 3.00 (3.00) | NO | - | artic/strips/microwave_hjjxmi_0.png |
| oven_ffitak_0 | oven/ffitak | [1.2165, 0.9933, 0.9864] | kitchen_0 | j_door | flap | handle | bar, 0.68, 9 | 3.00 (3.00) | yes | B/unreachable | artic/strips/oven_ffitak_0.png |
| oven_ffitak_0 | oven/ffitak | [1.2165, 0.9933, 0.9864] | kitchen_0 | j_rack1 | drawer | ungraspable | edge_drawer_top (not exposed), 0.43 | 3.00 (3.00) | yes | F/unreachable | artic/strips/oven_ffitak_0.png |
| oven_ffitak_0 | oven/ffitak | [1.2165, 0.9933, 0.9864] | kitchen_0 | j_rack2 | drawer | ungraspable | edge_drawer_top (not exposed), 0.32 | 3.00 (3.00) | yes | F/unreachable | artic/strips/oven_ffitak_0.png |
| toilet_udiezm_0 | toilet/udiezm | [1.0, 1.0, 1.0] | bathroom_0 | j_flap | flap | edge | edge_lid, 0.93 | 0.60 (3.00) | NO | - | artic/strips/toilet_udiezm_0.png |
| top_cabinet_tynnnw_0 | top_cabinet/tynnnw | [1.0, 1.0, 1.0] | kitchen_0 | j_leftdoor | door | handle | bar, 1.48, 31 | 2.50 (0.00) | NO | - | artic/strips/top_cabinet_tynnnw_0.png |
| top_cabinet_tynnnw_0 | top_cabinet/tynnnw | [1.0, 1.0, 1.0] | kitchen_0 | j_rightdoor | door | handle | knob, 1.48, 27 | 0.00 (0.00) | NO | - | artic/strips/top_cabinet_tynnnw_0.png |
| top_cabinet_tynnnw_1 | top_cabinet/tynnnw | [1.0, 1.0, 1.0] | kitchen_0 | j_leftdoor | door | handle | bar, 1.48, 31 | 2.90 (2.60) | NO | - | artic/strips/top_cabinet_tynnnw_1.png |
| top_cabinet_tynnnw_1 | top_cabinet/tynnnw | [1.0, 1.0, 1.0] | kitchen_0 | j_rightdoor | door | handle | knob, 1.48, 27 | 3.00 (3.00) | NO | - | artic/strips/top_cabinet_tynnnw_1.png |
| washer_ziomqg_0 | washer/ziomqg | [1.0, 1.0, 1.0] | bathroom_0 | j_door | door | handle | knob, 0.73, 77 | 2.20 (1.90) | NO | - | artic/strips/washer_ziomqg_0.png |
Review needed (no canonical handle annotation, graspable joint): 11 objects in 7 distinct (model, scale) groups: bottom_cabinet/rhdbzv x1 scale [1.0, 1.0, 1.0]; door/vudhlc x4 scale [1.0, 1.0, 1.0]; fridge/petcxr x1 scale [1.0, 1.0, 1.0]; microwave/hjjxmi x1 scale [1.0, 1.0, 1.0]; toilet/udiezm x1 scale [1.0, 1.0, 1.0]; top_cabinet/tynnnw x2 scale [1.0, 1.0, 1.0]; washer/ziomqg x1 scale [1.0, 1.0, 1.0]

## Rs_int

### supports (11)

| name | category | rooms | top z | area m² | free cells | best edge run 0.45 (edge, m, free frac) | blocked in front | chairs ≤ 1 m | in union |
|---|---|---|---|---|---|---|---|---|---|
| ottoman_ycfbsd_0 | ottoman | living_room_0 | 0.39 | 0.77 | 1925 | +v, 0.90, 1.00 | walls | 1 |  |
| bottom_cabinet_jhymlr_0 | bottom_cabinet | living_room_0 | 0.75 | 0.77 | 1919 | +u, 1.65, 1.00 | - | 1 | yes |
| breakfast_table_skczfi_0 | breakfast_table | living_room_0 | 0.76 | 0.75 | 1885 | +u, 1.00, 1.00 | - | 3 | yes |
| breakfast_table_skczfi_1 | breakfast_table | living_room_0 | 0.76 | 0.71 | 1787 | +u, 1.45, 1.00 | - | 1 | yes |
| countertop_tpuwys_0 | countertop | kitchen_0,living_room_0 | 1.08 | 0.58 | 1440 | -u, 1.40, 0.78 | - | 2 |  |
| bottom_cabinet_slgzfc_0 | bottom_cabinet | entryway_0 | 0.84 | 0.54 | 1350 | +u, 1.10, 1.00 | - | 1 | yes |
| coffee_table_fqluyq_0 | coffee_table | living_room_0 | 0.40 | 0.52 | 1309 | -u, 1.25, 1.00 | - | 0 |  |
| bottom_cabinet_dajebq_0 | bottom_cabinet | bedroom_0 | 1.29 | 0.19 | 476 | +u, 0.55, 1.00 | - | 0 |  |
| bottom_cabinet_bamfsz_1 | bottom_cabinet | living_room_0 | 0.73 | 0.17 | 435 | +u, 0.45, 1.00 | agent | 0 | yes |
| bottom_cabinet_bamfsz_0 | bottom_cabinet | kitchen_0 | 0.88 | 0.17 | 426 | -v, 0.20, 0.31 | dishwasher | 0 |  |
| bottom_cabinet_slgzfc_1 | bottom_cabinet | kitchen_0 | 0.87 | 0.08 | 195 | -v, 0.60, 1.00 | - | 1 |  |

### large partners (2)

| name | category | rooms | face z | face cells | under ≥ 0.15 m | under ≥ 0.25 m | max clearance | best edge run 0.45 | supports ≤ 3 m (geodesic) |
|---|---|---|---|---|---|---|---|---|---|
| bed_zrumze_0 | bed | bedroom_0 | 0.62 | 7744 | 0.05 | 0.02 | 0.72 | 1.65 | breakfast_table_skczfi_0 (1.77) |
| countertop_tpuwys_0 | countertop | kitchen_0,living_room_0 | 1.08 | 1440 | 0.20 | 0.20 | 1.04 | 1.40 | breakfast_table_skczfi_0 (0.34); bottom_cabinet_bamfsz_1 (0.34); bottom_cabinet_slgzfc_0 (0.44); coffee_table_fqluyq_0 (0.84); bottom_cabinet_jhymlr_0 (1.71) |

### articulated objects (22)

| name | category/model | scale | rooms | joint | kind | status | best handle (kind, height m, width mm) | front free m (eroded) | canonical annotation | prior verdict | strip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bottom_cabinet_bamfsz_0 | bottom_cabinet/bamfsz | [1.4322, 1.6915, 1.6014] | kitchen_0 | j_link_1 | drawer | handle | bar, 0.38, 19 | 3.00 (2.30) | yes | A/unreachable | artic/strips/bottom_cabinet_bamfsz_0.png |
| bottom_cabinet_bamfsz_0 | bottom_cabinet/bamfsz | [1.4322, 1.6915, 1.6014] | kitchen_0 | j_link_2 | drawer | handle | knob, 0.60, 20 | 3.00 (2.30) | yes | A/unreachable | artic/strips/bottom_cabinet_bamfsz_0.png |
| bottom_cabinet_bamfsz_0 | bottom_cabinet/bamfsz | [1.4322, 1.6915, 1.6014] | kitchen_0 | j_link_3 | drawer | handle | bar, 0.71, 20 | 3.00 (2.30) | yes | B/unreachable | artic/strips/bottom_cabinet_bamfsz_0.png |
| bottom_cabinet_bamfsz_0 | bottom_cabinet/bamfsz | [1.4322, 1.6915, 1.6014] | kitchen_0 | j_link_4 | drawer | handle | bar, 0.82, 20 | 3.00 (2.30) | yes | A/unreachable | artic/strips/bottom_cabinet_bamfsz_0.png |
| bottom_cabinet_bamfsz_1 | bottom_cabinet/bamfsz | [0.9254, 1.0426, 1.2737] | living_room_0 | j_link_1 | drawer | handle | bar, 0.33, 15 | 3.00 (0.40) | yes | A/unreachable | artic/strips/bottom_cabinet_bamfsz_1.png |
| bottom_cabinet_bamfsz_1 | bottom_cabinet/bamfsz | [0.9254, 1.0426, 1.2737] | living_room_0 | j_link_2 | drawer | handle | knob, 0.51, 16 | 3.00 (0.40) | yes | A/unreachable | artic/strips/bottom_cabinet_bamfsz_1.png |
| bottom_cabinet_bamfsz_1 | bottom_cabinet/bamfsz | [0.9254, 1.0426, 1.2737] | living_room_0 | j_link_3 | drawer | handle | knob, 0.59, 16 | 3.00 (0.40) | yes | B/unreachable | artic/strips/bottom_cabinet_bamfsz_1.png |
| bottom_cabinet_bamfsz_1 | bottom_cabinet/bamfsz | [0.9254, 1.0426, 1.2737] | living_room_0 | j_link_4 | drawer | handle | bar, 0.68, 16 | 3.00 (0.40) | yes | A/unreachable | artic/strips/bottom_cabinet_bamfsz_1.png |
| bottom_cabinet_dajebq_0 | bottom_cabinet/dajebq | [1.2926, 1.077, 1.4045] | bedroom_0 | j_link_0 | door | handle | bar, 0.58, 6 | 0.65 (0.35) | yes | C/unreachable | artic/strips/bottom_cabinet_dajebq_0.png |
| bottom_cabinet_dajebq_0 | bottom_cabinet/dajebq | [1.2926, 1.077, 1.4045] | bedroom_0 | j_link_1 | door | handle | bar, 0.83, 6 | 0.65 (0.35) | yes | B/unreachable | artic/strips/bottom_cabinet_dajebq_0.png |
| bottom_cabinet_dajebq_0 | bottom_cabinet/dajebq | [1.2926, 1.077, 1.4045] | bedroom_0 | j_link_2 | door | handle | bar, 0.58, 6 | 0.65 (0.00) | yes | C/unreachable | artic/strips/bottom_cabinet_dajebq_0.png |
| bottom_cabinet_dajebq_0 | bottom_cabinet/dajebq | [1.2926, 1.077, 1.4045] | bedroom_0 | j_link_3 | door | handle | bar, 0.83, 6 | 0.65 (0.00) | yes | B/unreachable | artic/strips/bottom_cabinet_dajebq_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.3135, 2.4063, 2.0004] | living_room_0 | j_link_0 | drawer | handle | bar, 0.31, 5 | 2.30 (1.00) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.3135, 2.4063, 2.0004] | living_room_0 | j_link_1 | drawer | handle | bar, 0.57, 5 | 2.30 (1.00) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.3135, 2.4063, 2.0004] | living_room_0 | j_link_2 | drawer | handle | bar, 0.74, 5 | 2.30 (1.00) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.3135, 2.4063, 2.0004] | living_room_0 | j_link_3 | drawer | handle | bar, 0.31, 5 | 1.40 (1.10) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.3135, 2.4063, 2.0004] | living_room_0 | j_link_4 | drawer | handle | bar, 0.57, 5 | 1.40 (1.10) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.3135, 2.4063, 2.0004] | living_room_0 | j_link_5 | drawer | handle | bar, 0.74, 5 | 1.40 (1.10) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_slgzfc_0 | bottom_cabinet/slgzfc | [1.1548, 2.5209, 1.5468] | entryway_0 | j_link_1 | drawer | handle | bar, 0.16, 13 | 2.65 (1.05) | NO | - | artic/strips/bottom_cabinet_slgzfc_0.png |
| bottom_cabinet_slgzfc_0 | bottom_cabinet/slgzfc | [1.1548, 2.5209, 1.5468] | entryway_0 | j_link_2 | drawer | handle | bar, 0.37, 13 | 2.65 (1.05) | NO | - | artic/strips/bottom_cabinet_slgzfc_0.png |
| bottom_cabinet_slgzfc_0 | bottom_cabinet/slgzfc | [1.1548, 2.5209, 1.5468] | entryway_0 | j_link_3 | drawer | handle | bar, 0.58, 13 | 2.65 (1.05) | NO | - | artic/strips/bottom_cabinet_slgzfc_0.png |
| bottom_cabinet_slgzfc_0 | bottom_cabinet/slgzfc | [1.1548, 2.5209, 1.5468] | entryway_0 | j_link_4 | drawer | handle | bar, 0.79, 13 | 2.65 (1.05) | NO | - | artic/strips/bottom_cabinet_slgzfc_0.png |
| bottom_cabinet_slgzfc_1 | bottom_cabinet/slgzfc | [1.3906, 0.3767, 1.6013] | kitchen_0 | j_link_1 | drawer | handle | bar, 0.17, 14 | 2.70 (0.00) | NO | - | artic/strips/bottom_cabinet_slgzfc_1.png |
| bottom_cabinet_slgzfc_1 | bottom_cabinet/slgzfc | [1.3906, 0.3767, 1.6013] | kitchen_0 | j_link_2 | drawer | handle | bar, 0.38, 14 | 2.70 (0.00) | NO | - | artic/strips/bottom_cabinet_slgzfc_1.png |
| bottom_cabinet_slgzfc_1 | bottom_cabinet/slgzfc | [1.3906, 0.3767, 1.6013] | kitchen_0 | j_link_3 | drawer | handle | bar, 0.60, 14 | 2.70 (0.00) | NO | - | artic/strips/bottom_cabinet_slgzfc_1.png |
| bottom_cabinet_slgzfc_1 | bottom_cabinet/slgzfc | [1.3906, 0.3767, 1.6013] | kitchen_0 | j_link_4 | drawer | handle | bar, 0.82, 14 | 2.70 (0.00) | NO | - | artic/strips/bottom_cabinet_slgzfc_1.png |
| dishwasher_xlmier_0 | dishwasher/xlmier | [1.2022, 1.1235, 0.7561] | kitchen_0 | j_link_1 | flap | handle | bar, 1.21, 15 | 3.00 (0.00) | yes | A/unreachable | artic/strips/dishwasher_xlmier_0.png |
| door_ktydvs_0 | door/ktydvs | [0.9542, 0.9485, 1.0492] | bedroom_0,bathroom_0 | j_link_2 | door | handle | knob, 1.17, 48 | 1.20 (0.80) | NO | - | artic/strips/door_ktydvs_0.png |
| door_lvgliq_0 | door/lvgliq | [0.7535, 1.0876, 1.0838] | entryway_0 | j_link_2 | door | handle | knob, 1.17, 8 | 0.00 (0.00) | NO | - | artic/strips/door_lvgliq_0.png |
| fridge_xyejdx_0 | fridge/xyejdx | [1.0556, 1.2552, 0.9016] | kitchen_0 | j_link_0 | door | handle | bar, 1.34, 29 | 1.00 (0.50) | yes | A/unreachable | artic/strips/fridge_xyejdx_0.png |
| fridge_xyejdx_0 | fridge/xyejdx | [1.0556, 1.2552, 0.9016] | kitchen_0 | j_link_1 | door | handle | bar, 0.96, 29 | 1.00 (0.50) | yes | A/unreachable | artic/strips/fridge_xyejdx_0.png |
| microwave_abzvij_0 | microwave/abzvij | [1.2896, 1.3002, 1.0157] | kitchen_0 | j_link_0 | door | ungraspable | edge_door (not exposed), 1.43 | 3.00 (2.75) | NO | - | artic/strips/microwave_abzvij_0.png |
| openable_window_ithrgo_0 | openable_window/ithrgo | [1.3135, 1.1064, 0.9675] | kitchen_0 | j_link_0 | drawer | ungraspable | edge_drawer_top (not exposed), 1.08 | 0.00 (0.00) | NO | - | artic/strips/openable_window_ithrgo_0.png |
| openable_window_ithrgo_0 | openable_window/ithrgo | [1.3135, 1.1064, 0.9675] | kitchen_0 | j_link_1 | drawer | ungraspable | edge_drawer_top (not exposed), 1.08 | 0.00 (0.00) | NO | - | artic/strips/openable_window_ithrgo_0.png |
| openable_window_ithrgo_1 | openable_window/ithrgo | [1.6716, 1.3682, 0.9675] | living_room_0 | j_link_0 | drawer | ungraspable | edge_drawer_top (not exposed), 1.08 | 0.00 (0.00) | NO | - | artic/strips/openable_window_ithrgo_1.png |
| openable_window_ithrgo_1 | openable_window/ithrgo | [1.6716, 1.3682, 0.9675] | living_room_0 | j_link_1 | drawer | ungraspable | edge_drawer_top (not exposed), 1.08 | 0.00 (0.00) | NO | - | artic/strips/openable_window_ithrgo_1.png |
| openable_window_ithrgo_2 | openable_window/ithrgo | [2.0297, 1.3682, 0.9675] | living_room_0 | j_link_0 | drawer | ungraspable | edge_drawer_top (not exposed), 1.08 | 3.00 (2.95) | NO | - | artic/strips/openable_window_ithrgo_2.png |
| openable_window_ithrgo_2 | openable_window/ithrgo | [2.0297, 1.3682, 0.9675] | living_room_0 | j_link_1 | drawer | ungraspable | edge_drawer_top (not exposed), 1.08 | 0.00 (0.00) | NO | - | artic/strips/openable_window_ithrgo_2.png |
| openable_window_ulnafj_0 | openable_window/ulnafj | [1.5084, 1.8207, 0.6507] | bedroom_0 | j_link_0 | drawer | edge | edge_drawer_top, 1.86 | 0.00 (0.00) | NO | - | artic/strips/openable_window_ulnafj_0.png |
| openable_window_ulnafj_0 | openable_window/ulnafj | [1.5084, 1.8207, 0.6507] | bedroom_0 | j_link_1 | drawer | ungraspable | edge_drawer_top (not exposed), 1.86 | 0.00 (0.00) | NO | - | artic/strips/openable_window_ulnafj_0.png |
| oven_wuinhm_0 | oven/wuinhm | [1.6359, 1.3271, 1.4872] | kitchen_0 | j_link_1 | flap | handle | bar, 0.52, 33 | 2.60 (2.30) | NO | - | artic/strips/oven_wuinhm_0.png |
| oven_wuinhm_0 | oven/wuinhm | [1.6359, 1.3271, 1.4872] | kitchen_0 | j_link_2 | flap | handle | bar, 0.84, 29 | 2.60 (2.30) | NO | - | artic/strips/oven_wuinhm_0.png |
| toilet_vtqdev_0 | toilet/vtqdev | [1.4042, 1.3224, 1.3099] | bathroom_0 | j_link_0 | flap | ungraspable | edge_lid (not exposed), 0.63 | - (-) | NO | - | artic/strips/toilet_vtqdev_0.png |
| toilet_vtqdev_0 | toilet/vtqdev | [1.4042, 1.3224, 1.3099] | bathroom_0 | j_link_1 | drawer | edge | edge_drawer_top, 0.66 | - (-) | NO | - | artic/strips/toilet_vtqdev_0.png |
| top_cabinet_fqhdne_0 | top_cabinet/fqhdne | [1.8757, 2.117, 1.8406] | kitchen_0 | j_link_0 | door | handle | bar, 1.79, 24 | 1.25 (2.55) | yes | A/model_error | artic/strips/top_cabinet_fqhdne_0.png |
| top_cabinet_lsyzkh_0 | top_cabinet/lsyzkh | [1.5608, 1.2891, 1.8992] | kitchen_0 | j_link_1 | door | handle | bar, 2.17, 10 | 1.20 (3.00) | yes | A/model_error | artic/strips/top_cabinet_lsyzkh_0.png |
| top_cabinet_lsyzkh_1 | top_cabinet/lsyzkh | [1.562, 1.4655, 1.8992] | kitchen_0 | j_link_1 | door | handle | bar, 2.17, 12 | 1.20 (0.90) | yes | A/model_error | artic/strips/top_cabinet_lsyzkh_1.png |
| top_cabinet_lsyzkh_2 | top_cabinet/lsyzkh | [1.1926, 1.4285, 1.8884] | kitchen_0 | j_link_1 | door | handle | bar, 2.13, 11 | 0.90 (0.00) | yes | A/model_error | artic/strips/top_cabinet_lsyzkh_2.png |
| top_cabinet_lsyzkh_3 | top_cabinet/lsyzkh | [1.1926, 1.4285, 1.8884] | kitchen_0 | j_link_1 | door | handle | bar, 2.13, 11 | 0.90 (0.50) | yes | A/model_error | artic/strips/top_cabinet_lsyzkh_3.png |
Review needed (no canonical handle annotation, graspable joint): 8 objects in 8 distinct (model, scale) groups: bottom_cabinet/jhymlr x1 scale [2.31, 2.41, 2.0]; bottom_cabinet/slgzfc x1 scale [1.15, 2.52, 1.55]; bottom_cabinet/slgzfc x1 scale [1.39, 0.38, 1.6]; door/ktydvs x1 scale [0.95, 0.95, 1.05]; door/lvgliq x1 scale [0.75, 1.09, 1.08]; openable_window/ulnafj x1 scale [1.51, 1.82, 0.65]; oven/wuinhm x1 scale [1.64, 1.33, 1.49]; toilet/vtqdev x1 scale [1.4, 1.32, 1.31]

## hotel_suite_large

### supports (15)

| name | category | rooms | top z | area m² | free cells | best edge run 0.45 (edge, m, free frac) | blocked in front | chairs ≤ 1 m | in union |
|---|---|---|---|---|---|---|---|---|---|
| stand_dbskvf_0 | stand | bedroom_0 | 0.41 | 2.34 | 5842 | +u, 5.70, 0.99 | - | 1 |  |
| desk_ohjotf_0 | desk | bedroom_0 | 0.68 | 0.96 | 2403 | +u, 1.65, 1.00 | - | 1 | yes |
| shelf_rglkkf_0 | shelf | bathroom_0 | 0.66 | 0.83 | 2064 | +u, 1.70, 0.97 | - | 0 | yes |
| ottoman_hnmndv_0 | ottoman | bedroom_0 | 0.47 | 0.61 | 1518 | +u, 1.45, 1.00 | - | 1 |  |
| ottoman_hnmndv_1 | ottoman | bedroom_0 | 0.47 | 0.61 | 1518 | +u, 1.45, 1.00 | - | 0 |  |
| shelf_nbmidu_0 | shelf | bedroom_0 | 1.28 | 0.53 | 1333 | +u, 1.50, 0.97 | - | 0 |  |
| nightstand_maeowc_1 | nightstand | bedroom_0 | 0.45 | 0.32 | 806 | -u, 0.65, 1.00 | walls | 0 |  |
| shelf_aqtmhm_0 | shelf | bedroom_0 | 0.91 | 0.31 | 778 | +u, 2.25, 0.98 | - | 0 |  |
| nightstand_maeowc_0 | nightstand | bedroom_0 | 0.45 | 0.31 | 769 | +u, 0.65, 1.00 | - | 0 |  |
| nightstand_maeowc_2 | nightstand | bedroom_0 | 0.45 | 0.31 | 769 | +u, 0.65, 1.00 | - | 0 |  |
| shelf_xogqqm_0 | shelf | bathroom_0 | 1.46 | 0.25 | 624 | +u, 0.85, 1.00 | - | 0 |  |
| shelf_xogqqm_2 | shelf | bathroom_0 | 0.64 | 0.25 | 624 | +u, 0.85, 1.00 | - | 0 |  |
| shelf_xogqqm_3 | shelf | bathroom_0 | 0.22 | 0.25 | 624 | +u, 0.85, 1.00 | - | 0 |  |
| shelf_cbidyu_0 | shelf | bathroom_0 | 0.27 | 0.05 | 114 | +u, 2.00, 0.87 | wall_mounted_sink | 0 |  |
| shelf_xogqqm_1 | shelf | bathroom_0 | 1.08 | 0.03 | 84 | +u, 0.85, 1.00 | shelf | 0 |  |

### large partners (2)

| name | category | rooms | face z | face cells | under ≥ 0.15 m | under ≥ 0.25 m | max clearance | best edge run 0.45 | supports ≤ 3 m (geodesic) |
|---|---|---|---|---|---|---|---|---|---|
| bed_kddswt_0 | bed | bedroom_0 | 0.51 | 7836 | 0.00 | 0.00 | 0.01 | 1.60 | stand_dbskvf_0 (0.68) |
| bed_kddswt_1 | bed | bedroom_0 | 0.51 | 7836 | 0.00 | 0.00 | 0.01 | 1.60 | shelf_xogqqm_3 (0.44); shelf_xogqqm_1 (0.44); shelf_xogqqm_2 (0.44); shelf_xogqqm_0 (0.44); shelf_rglkkf_0 (1.33) |

### articulated objects (3)

| name | category/model | scale | rooms | joint | kind | status | best handle (kind, height m, width mm) | front free m (eroded) | canonical annotation | prior verdict | strip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| door_uptpdr_0 | door/uptpdr | [0.9999, 1.0, 1.0] | bathroom_0 | j_leaf | door | ungraspable | edge_door (not exposed), 0.66 | 3.00 (3.00) | NO | - | artic/strips/door_uptpdr_0.png |
| door_ydokma_0 | door/ydokma | [1.0001, 1.0, 1.0] | bathroom_0 | j_leaf | door | handle | knob, 0.86, 46 | 3.00 (3.00) | NO | - | artic/strips/door_ydokma_0.png |
| toilet_aapttl_0 | toilet/aapttl | [1.1838, 1.02, 1.5284] | bathroom_0 | j_link_2 | flap | ungraspable | edge_lid (not exposed), 0.48 | - (-) | NO | - | artic/strips/toilet_aapttl_0.png |
Review needed (no canonical handle annotation, graspable joint): 1 objects in 1 distinct (model, scale) groups: door/ydokma x1 scale [1.0, 1.0, 1.0]

## restaurant_diner

### supports (25)

| name | category | rooms | top z | area m² | free cells | best edge run 0.45 (edge, m, free frac) | blocked in front | chairs ≤ 1 m | in union |
|---|---|---|---|---|---|---|---|---|---|
| bar_xxftww_0 | bar | bar_0,dining_room_0 | 1.08 | 4.85 | 12130 | +u, 4.00, 0.93 | - | 1 |  |
| commercial_kitchen_table_vxvtec_0 | commercial_kitchen_table | kitchen_0 | 0.87 | 2.24 | 5600 | +u, 2.80, 1.00 | - | 0 |  |
| booth_xzrpar_0 | booth | dining_room_0 | 0.75 | 0.92 | 2309 | -v, 0.70, 1.00 | - | 0 | yes |
| booth_xzrpar_1 | booth | dining_room_0 | 0.75 | 0.92 | 2309 | -v, 0.70, 1.00 | - | 0 | yes |
| booth_xzrpar_2 | booth | dining_room_0 | 0.75 | 0.92 | 2307 | -v, 0.70, 1.00 | - | 1 | yes |
| booth_xzrpar_3 | booth | dining_room_0 | 0.75 | 0.92 | 2307 | -v, 0.70, 1.00 | - | 0 | yes |
| pedestal_table_nlfgwe_0 | pedestal_table | dining_room_0 | 0.78 | 0.89 | 2222 | +u, 1.10, 1.00 | - | 3 | yes |
| pedestal_table_nlfgwe_1 | pedestal_table | dining_room_0 | 0.78 | 0.89 | 2222 | -u, 1.10, 1.00 | - | 2 | yes |
| pedestal_table_nlfgwe_2 | pedestal_table | dining_room_0 | 0.78 | 0.89 | 2222 | -u, 1.10, 1.00 | - | 2 | yes |
| bottom_cabinet_olgoza_0 | bottom_cabinet | bar_0 | 0.84 | 0.88 | 2205 | +u, 2.10, 1.00 | - | 0 | yes |
| metal_bottom_cabinet_bbjuqu_0 | metal_bottom_cabinet | kitchen_0 | 0.86 | 0.84 | 2100 | +u, 1.40, 1.00 | - | 0 |  |
| metal_bottom_cabinet_bbjuqu_1 | metal_bottom_cabinet | kitchen_0 | 0.86 | 0.84 | 2100 | +u, 1.40, 1.00 | - | 0 |  |
| metal_bottom_cabinet_bbjuqu_2 | metal_bottom_cabinet | kitchen_0 | 0.86 | 0.84 | 2100 | +u, 1.40, 1.00 | - | 0 |  |
| metal_bottom_cabinet_bbjuqu_3 | metal_bottom_cabinet | kitchen_0 | 0.86 | 0.84 | 2100 | +u, 1.40, 1.00 | - | 0 |  |
| commercial_kitchen_table_nisnrl_0 | commercial_kitchen_table | kitchen_0 | 0.87 | 0.76 | 1904 | +u, 1.15, 1.00 | - | 0 |  |
| shelf_xuhsfd_0 | shelf | kitchen_0 | 2.04 | 0.74 | 1860 | +u, 2.50, 1.00 | - | 0 |  |
| shelf_xuhsfd_1 | shelf | kitchen_0 | 2.12 | 0.74 | 1860 | +u, 2.50, 1.00 | - | 0 |  |
| bench_xwphjd_6 | bench | dining_room_0 | 0.40 | 0.59 | 1476 | -v, 0.55, 1.00 | - | 1 |  |
| bench_xwphjd_7 | bench | dining_room_0 | 0.40 | 0.59 | 1476 | -v, 0.55, 1.00 | - | 0 |  |
| bench_xwphjd_0 | bench | dining_room_0 | 0.40 | 0.59 | 1464 | -u, 1.25, 1.00 | - | 0 |  |
| bench_xwphjd_5 | bench | dining_room_0 | 0.40 | 0.59 | 1464 | -v, 0.60, 1.00 | - | 0 |  |
| bench_xwphjd_1 | bench | dining_room_0 | 0.40 | 0.57 | 1427 | +v, 0.55, 1.00 | - | 0 |  |
| bench_xwphjd_2 | bench | dining_room_0 | 0.40 | 0.57 | 1427 | -u, 1.25, 1.00 | - | 0 |  |
| bench_xwphjd_3 | bench | dining_room_0 | 0.40 | 0.57 | 1415 | +v, 0.60, 1.00 | - | 0 |  |
| bench_xwphjd_4 | bench | dining_room_0 | 0.40 | 0.57 | 1415 | -u, 0.65, 0.92 | - | 0 |  |

### large partners (0)

| name | category | rooms | face z | face cells | under ≥ 0.15 m | under ≥ 0.25 m | max clearance | best edge run 0.45 | supports ≤ 3 m (geodesic) |
|---|---|---|---|---|---|---|---|---|---|

### articulated objects (7)

| name | category/model | scale | rooms | joint | kind | status | best handle (kind, height m, width mm) | front free m (eroded) | canonical annotation | prior verdict | strip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| door_adejun_0 | door/adejun | [1.0001, 1.0, 1.0] | dining_room_0 | j_left_leaf | door | handle | bar, 0.92, 67 | 0.00 (0.00) | yes | none/model_error | artic/strips/door_adejun_0.png |
| door_adejun_0 | door/adejun | [1.0001, 1.0, 1.0] | dining_room_0 | j_right_leaf | door | handle | bar, 2.07, 68 | 0.00 (0.00) | yes | none/model_error | artic/strips/door_adejun_0.png |
| door_cvvumy_0 | door/cvvumy | [1.0002, 1.0, 1.0] | bathroom_0 | j_left_leaf | door | ungraspable | edge_door (not exposed), 1.54 | 3.00 (3.00) | yes | none/infeasible_2_reviews | artic/strips/door_cvvumy_0.png |
| door_cvvumy_0 | door/cvvumy | [1.0002, 1.0, 1.0] | bathroom_0 | j_right_leaf | door | ungraspable | edge_door (not exposed), 1.40 | 3.00 (3.00) | yes | none/infeasible_2_reviews | artic/strips/door_cvvumy_0.png |
| door_mkcndz_0 | door/mkcndz | [1.0002, 0.9999, 1.0] | kitchen_0 | j_leaf | door | ungraspable | edge_door (not exposed), 0.92 | 1.75 (1.45) | NO | - | artic/strips/door_mkcndz_0.png |
| fridge_jtqazu_0 | fridge/jtqazu | [1.6483, 2.0141, 1.2681] | kitchen_0 | j_link_0 | door | handle | knob, 1.16, 63 | 3.00 (2.20) | yes | A/opened | artic/strips/fridge_jtqazu_0.png |
| fridge_jtqazu_1 | fridge/jtqazu | [1.6483, 2.0141, 1.2681] | kitchen_0 | j_link_0 | door | handle | knob, 1.16, 63 | 3.00 (3.00) | yes | A/opened | artic/strips/fridge_jtqazu_1.png |
| fridge_juwaoh_0 | fridge/juwaoh | [1.0691, 1.3773, 0.941] | bar_0 | j_link_0 | door | ungraspable | edge_door (not exposed), 0.63 | 1.00 (0.40) | yes | F/unreachable | artic/strips/fridge_juwaoh_0.png |
| oven_amblrk_0 | oven/amblrk | [1.0, 1.0001, 1.0] | kitchen_0 | j_door_leaf1 | door | handle | knob, 1.22, 21 | 3.00 (3.00) | NO | - | artic/strips/oven_amblrk_0.png |
| oven_amblrk_0 | oven/amblrk | [1.0, 1.0001, 1.0] | kitchen_0 | j_door_leaf2 | door | handle | knob, 0.64, 21 | 3.00 (3.00) | NO | - | artic/strips/oven_amblrk_0.png |
Review needed (no canonical handle annotation, graspable joint): 1 objects in 1 distinct (model, scale) groups: oven/amblrk x1 scale [1.0, 1.0, 1.0]

## office_cubicles_right

### supports (35)

| name | category | rooms | top z | area m² | free cells | best edge run 0.45 (edge, m, free frac) | blocked in front | chairs ≤ 1 m | in union |
|---|---|---|---|---|---|---|---|---|---|
| conference_table_fpkmyw_0 | conference_table | meeting_room_0 | 0.67 | 5.20 | 12996 | +u, 3.45, 1.00 | walls | 10 | yes |
| desk_pfsjbc_0 | desk | lobby_0 | 0.77 | 3.54 | 8846 | +u, 4.25, 1.00 | walls | 2 | yes |
| desk_mdhelw_0 | desk | private_office_3 | 0.79 | 2.58 | 6439 | -u, 2.80, 1.00 | - | 3 | yes |
| desk_mdhelw_1 | desk | private_office_2 | 0.79 | 2.58 | 6439 | +u, 2.80, 1.00 | walls | 3 | yes |
| desk_mdhelw_2 | desk | private_office_0 | 0.79 | 2.58 | 6439 | +u, 2.80, 1.00 | - | 3 |  |
| desk_mdhelw_3 | desk | private_office_1 | 0.79 | 2.58 | 6439 | +u, 2.80, 1.00 | walls | 3 | yes |
| bottom_cabinet_jhymlr_0 | bottom_cabinet | copy_room_0 | 1.02 | 2.05 | 5119 | +u, 3.45, 0.99 | - | 0 |  |
| cubicle_hzalfx_0 | cubicle | shared_office_0 | 0.78 | 1.28 | 3192 | +v, 1.40, 1.00 | - | 2 |  |
| cubicle_hzalfx_2 | cubicle | shared_office_0 | 0.78 | 1.28 | 3192 | +v, 1.40, 1.00 | - | 3 |  |
| cubicle_hzalfx_4 | cubicle | shared_office_0 | 0.78 | 1.28 | 3192 | +v, 1.40, 1.00 | - | 3 |  |
| cubicle_hzalfx_1 | cubicle | shared_office_0 | 0.78 | 1.26 | 3163 | +v, 1.40, 1.00 | cubicle | 2 |  |
| cubicle_hzalfx_3 | cubicle | shared_office_0 | 0.78 | 1.26 | 3163 | +v, 1.40, 1.00 | cubicle | 3 |  |
| cubicle_hzalfx_5 | cubicle | shared_office_0 | 0.78 | 1.26 | 3163 | +v, 1.40, 1.00 | - | 3 |  |
| cubicle_deceti_1 | cubicle | shared_office_0 | 0.78 | 1.26 | 3156 | -v, 1.40, 1.00 | - | 3 |  |
| cubicle_deceti_3 | cubicle | shared_office_0 | 0.78 | 1.26 | 3156 | -v, 1.40, 1.00 | - | 3 |  |
| cubicle_deceti_0 | cubicle | shared_office_0 | 0.78 | 1.25 | 3127 | -v, 1.40, 1.00 | - | 3 |  |
| cubicle_deceti_2 | cubicle | shared_office_0 | 0.78 | 1.25 | 3127 | -v, 1.40, 1.00 | - | 3 |  |
| countertop_zkhelh_0 | countertop | bathroom_0 | 0.78 | 1.12 | 2797 | +u, 4.45, 1.00 | - | 0 |  |
| coffee_table_mtchfd_0 | coffee_table | lobby_0 | 0.45 | 0.44 | 1093 | +u, 0.75, 1.00 | - | 0 |  |
| bottom_cabinet_rvunhj_0 | bottom_cabinet | shared_office_0 | 1.57 | 0.34 | 840 | +u, 0.80, 1.00 | - | 1 |  |
| bottom_cabinet_rvunhj_1 | bottom_cabinet | shared_office_0 | 1.57 | 0.34 | 840 | +u, 0.80, 1.00 | - | 1 |  |
| bottom_cabinet_rvunhj_2 | bottom_cabinet | shared_office_0 | 1.57 | 0.34 | 840 | +u, 0.80, 1.00 | - | 1 |  |
| bottom_cabinet_rvunhj_3 | bottom_cabinet | shared_office_0 | 1.57 | 0.34 | 840 | +u, 0.80, 1.00 | - | 1 |  |
| bottom_cabinet_rvunhj_4 | bottom_cabinet | shared_office_0 | 1.57 | 0.34 | 840 | +u, 0.80, 1.00 | - | 1 |  |
| bottom_cabinet_rvunhj_5 | bottom_cabinet | shared_office_0 | 1.57 | 0.34 | 840 | +u, 0.80, 1.00 | - | 0 |  |
| bottom_cabinet_immwzb_6 | bottom_cabinet | shared_office_0 | 0.67 | 0.00 | 8 | +u, 0.15, 1.00 | - | 2 |  |
| bottom_cabinet_immwzb_7 | bottom_cabinet | shared_office_0 | 0.67 | 0.00 | 8 | +u, 0.15, 1.00 | - | 1 |  |
| bottom_cabinet_immwzb_8 | bottom_cabinet | shared_office_0 | 0.67 | 0.00 | 8 | +u, 0.15, 1.00 | - | 2 |  |
| bottom_cabinet_immwzb_9 | bottom_cabinet | shared_office_0 | 0.67 | 0.00 | 8 | +u, 0.15, 1.00 | - | 1 |  |
| bottom_cabinet_immwzb_0 | bottom_cabinet | shared_office_0 | 0.71 | 0.00 | 0 | -, -, - | - | 2 |  |
| bottom_cabinet_immwzb_1 | bottom_cabinet | shared_office_0 | 0.71 | 0.00 | 0 | -, -, - | - | 1 |  |
| bottom_cabinet_immwzb_2 | bottom_cabinet | shared_office_0 | 0.71 | 0.00 | 0 | -, -, - | - | 2 |  |
| bottom_cabinet_immwzb_3 | bottom_cabinet | shared_office_0 | 0.71 | 0.00 | 0 | -, -, - | - | 1 |  |
| bottom_cabinet_immwzb_4 | bottom_cabinet | shared_office_0 | 0.71 | 0.00 | 0 | -, -, - | - | 2 |  |
| bottom_cabinet_immwzb_5 | bottom_cabinet | shared_office_0 | 0.71 | 0.00 | 0 | -, -, - | - | 1 |  |

### large partners (6)

| name | category | rooms | face z | face cells | under ≥ 0.15 m | under ≥ 0.25 m | max clearance | best edge run 0.45 | supports ≤ 3 m (geodesic) |
|---|---|---|---|---|---|---|---|---|---|
| countertop_zkhelh_0 | countertop | bathroom_0 | 0.78 | 2797 | 1.00 | 1.00 | 0.69 | 4.45 | - |
| desk_mdhelw_0 | desk | private_office_3 | 0.79 | 6439 | 0.82 | 0.81 | 0.69 | 2.80 | - |
| desk_mdhelw_1 | desk | private_office_2 | 0.79 | 6439 | 0.81 | 0.80 | 0.69 | 2.80 | - |
| desk_mdhelw_2 | desk | private_office_0 | 0.79 | 6439 | 0.81 | 0.80 | 0.68 | 2.80 | - |
| desk_mdhelw_3 | desk | private_office_1 | 0.79 | 6439 | 0.82 | 0.81 | 0.69 | 2.80 | - |
| desk_pfsjbc_0 | desk | lobby_0 | 0.77 | 8846 | 0.59 | 0.58 | 0.67 | 4.25 | - |

### articulated objects (32)

| name | category/model | scale | rooms | joint | kind | status | best handle (kind, height m, width mm) | front free m (eroded) | canonical annotation | prior verdict | strip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bottom_cabinet_immwzb_0 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_0 | door | handle | knob, 0.19, 7 | 2.00 (1.70) | NO | - | artic/strips/bottom_cabinet_immwzb_0.png |
| bottom_cabinet_immwzb_0 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_2 | drawer | handle | bar, 0.67, 13 | 2.00 (1.70) | NO | - | artic/strips/bottom_cabinet_immwzb_0.png |
| bottom_cabinet_immwzb_1 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_0 | door | handle | knob, 0.19, 7 | 2.00 (1.70) | NO | - | artic/strips/bottom_cabinet_immwzb_1.png |
| bottom_cabinet_immwzb_1 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_2 | drawer | handle | bar, 0.67, 13 | 2.00 (1.70) | NO | - | artic/strips/bottom_cabinet_immwzb_1.png |
| bottom_cabinet_immwzb_2 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_0 | door | handle | knob, 0.19, 7 | 2.65 (1.55) | NO | - | artic/strips/bottom_cabinet_immwzb_2.png |
| bottom_cabinet_immwzb_2 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_2 | drawer | handle | bar, 0.67, 13 | 2.65 (1.55) | NO | - | artic/strips/bottom_cabinet_immwzb_2.png |
| bottom_cabinet_immwzb_3 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_0 | door | handle | knob, 0.19, 7 | 2.65 (1.75) | NO | - | artic/strips/bottom_cabinet_immwzb_3.png |
| bottom_cabinet_immwzb_3 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_2 | drawer | handle | bar, 0.67, 13 | 2.65 (1.75) | NO | - | artic/strips/bottom_cabinet_immwzb_3.png |
| bottom_cabinet_immwzb_4 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_0 | door | handle | knob, 0.19, 7 | 2.70 (1.60) | NO | - | artic/strips/bottom_cabinet_immwzb_4.png |
| bottom_cabinet_immwzb_4 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_2 | drawer | handle | bar, 0.67, 13 | 2.70 (1.60) | NO | - | artic/strips/bottom_cabinet_immwzb_4.png |
| bottom_cabinet_immwzb_5 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_0 | door | handle | knob, 0.19, 7 | 2.70 (1.80) | NO | - | artic/strips/bottom_cabinet_immwzb_5.png |
| bottom_cabinet_immwzb_5 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_2 | drawer | handle | bar, 0.67, 13 | 2.70 (1.80) | NO | - | artic/strips/bottom_cabinet_immwzb_5.png |
| bottom_cabinet_immwzb_6 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_0 | door | handle | knob, 0.19, 7 | 2.60 (1.50) | NO | - | artic/strips/bottom_cabinet_immwzb_6.png |
| bottom_cabinet_immwzb_6 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_2 | drawer | handle | bar, 0.67, 13 | 2.60 (1.50) | NO | - | artic/strips/bottom_cabinet_immwzb_6.png |
| bottom_cabinet_immwzb_7 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_0 | door | handle | knob, 0.19, 7 | 2.60 (1.70) | NO | - | artic/strips/bottom_cabinet_immwzb_7.png |
| bottom_cabinet_immwzb_7 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_2 | drawer | handle | bar, 0.67, 13 | 2.60 (1.70) | NO | - | artic/strips/bottom_cabinet_immwzb_7.png |
| bottom_cabinet_immwzb_8 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_0 | door | handle | knob, 0.19, 7 | 2.60 (1.60) | NO | - | artic/strips/bottom_cabinet_immwzb_8.png |
| bottom_cabinet_immwzb_8 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_2 | drawer | handle | bar, 0.67, 13 | 2.60 (1.60) | NO | - | artic/strips/bottom_cabinet_immwzb_8.png |
| bottom_cabinet_immwzb_9 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_0 | door | handle | knob, 0.19, 7 | 2.60 (1.80) | NO | - | artic/strips/bottom_cabinet_immwzb_9.png |
| bottom_cabinet_immwzb_9 | bottom_cabinet/immwzb | [1.3301, 0.9117, 1.3046] | shared_office_0 | j_link_2 | drawer | handle | bar, 0.67, 13 | 2.60 (1.80) | NO | - | artic/strips/bottom_cabinet_immwzb_9.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.4429, 5.1141, 2.72] | copy_room_0 | j_link_0 | drawer | handle | bar, 0.41, 6 | 1.75 (1.45) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.4429, 5.1141, 2.72] | copy_room_0 | j_link_1 | drawer | handle | bar, 0.77, 6 | 1.75 (1.45) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.4429, 5.1141, 2.72] | copy_room_0 | j_link_2 | drawer | handle | bar, 1.00, 6 | 1.75 (1.45) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.4429, 5.1141, 2.72] | copy_room_0 | j_link_3 | drawer | handle | bar, 0.41, 6 | 1.75 (1.45) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.4429, 5.1141, 2.72] | copy_room_0 | j_link_4 | drawer | handle | bar, 0.77, 6 | 1.75 (1.45) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_jhymlr_0 | bottom_cabinet/jhymlr | [2.4429, 5.1141, 2.72] | copy_room_0 | j_link_5 | drawer | handle | bar, 1.00, 6 | 1.75 (1.45) | NO | - | artic/strips/bottom_cabinet_jhymlr_0.png |
| bottom_cabinet_rvunhj_0 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_0 | door | handle | knob, 0.61, 7 | 3.00 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_0.png |
| bottom_cabinet_rvunhj_0 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_1 | door | handle | knob, 1.04, 7 | 3.00 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_0.png |
| bottom_cabinet_rvunhj_0 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_2 | door | handle | knob, 0.61, 7 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_0.png |
| bottom_cabinet_rvunhj_0 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_3 | door | handle | knob, 1.04, 7 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_0.png |
| bottom_cabinet_rvunhj_1 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_0 | door | handle | knob, 0.61, 7 | 3.00 (0.70) | NO | - | artic/strips/bottom_cabinet_rvunhj_1.png |
| bottom_cabinet_rvunhj_1 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_1 | door | handle | knob, 1.04, 7 | 3.00 (0.70) | NO | - | artic/strips/bottom_cabinet_rvunhj_1.png |
| bottom_cabinet_rvunhj_1 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_2 | door | handle | knob, 0.61, 7 | 0.80 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_1.png |
| bottom_cabinet_rvunhj_1 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_3 | door | handle | knob, 1.04, 7 | 0.80 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_1.png |
| bottom_cabinet_rvunhj_2 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_0 | door | handle | knob, 0.61, 7 | 3.00 (0.60) | NO | - | artic/strips/bottom_cabinet_rvunhj_2.png |
| bottom_cabinet_rvunhj_2 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_1 | door | handle | knob, 1.04, 7 | 3.00 (0.60) | NO | - | artic/strips/bottom_cabinet_rvunhj_2.png |
| bottom_cabinet_rvunhj_2 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_2 | door | handle | knob, 0.61, 7 | 1.00 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_2.png |
| bottom_cabinet_rvunhj_2 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_3 | door | handle | knob, 1.04, 7 | 1.00 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_2.png |
| bottom_cabinet_rvunhj_3 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_0 | door | handle | knob, 0.61, 7 | 1.00 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_3.png |
| bottom_cabinet_rvunhj_3 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_1 | door | handle | knob, 1.04, 7 | 1.00 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_3.png |
| bottom_cabinet_rvunhj_3 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_2 | door | handle | knob, 0.61, 7 | 3.00 (0.70) | NO | - | artic/strips/bottom_cabinet_rvunhj_3.png |
| bottom_cabinet_rvunhj_3 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_3 | door | handle | knob, 1.04, 7 | 3.00 (0.70) | NO | - | artic/strips/bottom_cabinet_rvunhj_3.png |
| bottom_cabinet_rvunhj_4 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_0 | door | handle | knob, 0.61, 7 | 3.00 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_4.png |
| bottom_cabinet_rvunhj_4 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_1 | door | handle | knob, 1.04, 7 | 3.00 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_4.png |
| bottom_cabinet_rvunhj_4 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_2 | door | handle | knob, 0.61, 7 | 0.60 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_4.png |
| bottom_cabinet_rvunhj_4 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_3 | door | handle | knob, 1.04, 7 | 0.60 (0.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_4.png |
| bottom_cabinet_rvunhj_5 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_0 | door | handle | knob, 0.61, 7 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_5.png |
| bottom_cabinet_rvunhj_5 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_1 | door | handle | knob, 1.04, 7 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_5.png |
| bottom_cabinet_rvunhj_5 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_2 | door | handle | knob, 0.61, 7 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_5.png |
| bottom_cabinet_rvunhj_5 | bottom_cabinet/rvunhj | [1.6025, 1.5341, 1.7222] | shared_office_0 | j_link_3 | door | handle | knob, 1.04, 7 | 3.00 (3.00) | NO | - | artic/strips/bottom_cabinet_rvunhj_5.png |
| door_ggcyib_0 | door/ggcyib | [1.0003, 1.0, 1.0] | corridor_0,private_office_3 | j_leaf | door | handle | knob, 1.00, 50 | 3.00 (1.70) | NO | - | artic/strips/door_ggcyib_0.png |
| door_ggcyib_1 | door/ggcyib | [1.0003, 1.0, 1.0] | corridor_0,private_office_1 | j_leaf | door | handle | knob, 1.00, 50 | 2.00 (1.70) | NO | - | artic/strips/door_ggcyib_1.png |
| door_ggcyib_2 | door/ggcyib | [1.0003, 1.0, 1.0] | corridor_0,empty_room_0 | j_leaf | door | handle | knob, 1.01, 50 | 2.00 (1.70) | NO | - | artic/strips/door_ggcyib_2.png |
| door_ggcyib_3 | door/ggcyib | [1.0003, 1.0, 1.0] | corridor_0,private_office_0 | j_leaf | door | handle | knob, 1.01, 50 | 2.00 (1.70) | NO | - | artic/strips/door_ggcyib_3.png |
| door_ggcyib_4 | door/ggcyib | [1.0003, 1.0, 1.0] | corridor_0,conference_hall_0 | j_leaf | door | handle | knob, 1.00, 50 | 1.85 (1.55) | NO | - | artic/strips/door_ggcyib_4.png |
| door_ggcyib_5 | door/ggcyib | [1.0003, 1.0, 1.0] | corridor_0,meeting_room_0 | j_leaf | door | handle | knob, 1.00, 50 | 1.85 (1.55) | NO | - | artic/strips/door_ggcyib_5.png |
| door_lztyfn_0 | door/lztyfn | [0.9997, 1.0, 1.0] | bathroom_0 | j_door | door | handle | knob, 0.86, 56 | 2.10 (1.80) | NO | - | artic/strips/door_lztyfn_0.png |
| door_pwoerr_0 | door/pwoerr | [1.0002, 1.0, 1.0] | corridor_0,private_office_2 | j_leaf | door | handle | bar, 1.00, 21 | 3.00 (3.00) | NO | - | artic/strips/door_pwoerr_0.png |
| door_pwoerr_1 | door/pwoerr | [1.0002, 1.0, 1.0] | corridor_0,conference_hall_0 | j_leaf | door | handle | bar, 1.00, 21 | 1.85 (1.55) | NO | - | artic/strips/door_pwoerr_1.png |
| toilet_udiezm_0 | toilet/udiezm | [0.9173, 0.9084, 0.328] | bathroom_0 | j_flap | flap | edge | edge_lid, 0.47 | 0.00 (0.00) | NO | - | artic/strips/toilet_udiezm_0.png |
| toilet_udiezm_1 | toilet/udiezm | [0.9173, 0.9084, 0.328] | bathroom_0 | j_flap | flap | edge | edge_lid, 0.47 | 0.00 (0.00) | NO | - | artic/strips/toilet_udiezm_1.png |
| toilet_udiezm_2 | toilet/udiezm | [0.9173, 0.9084, 0.328] | bathroom_0 | j_flap | flap | edge | edge_lid, 0.47 | 0.00 (0.00) | NO | - | artic/strips/toilet_udiezm_2.png |
| toilet_udiezm_3 | toilet/udiezm | [0.9173, 0.9084, 0.328] | bathroom_0 | j_flap | flap | edge | edge_lid, 0.47 | 0.00 (0.00) | NO | - | artic/strips/toilet_udiezm_3.png |
| toilet_udiezm_4 | toilet/udiezm | [0.9173, 0.9084, 0.328] | bathroom_0 | j_flap | flap | edge | edge_lid, 0.47 | 0.00 (0.00) | NO | - | artic/strips/toilet_udiezm_4.png |
| top_cabinet_fqitbw_0 | top_cabinet/fqitbw | [1.7096, 6.1928, 6.9626] | copy_room_0 | j_link_1 | flap | ungraspable | edge_lid (not exposed), 1.96 | 2.05 (1.75) | yes | A/unreachable | artic/strips/top_cabinet_fqitbw_0.png |
Review needed (no canonical handle annotation, graspable joint): 31 objects in 7 distinct (model, scale) groups: bottom_cabinet/immwzb x10 scale [1.33, 0.91, 1.3]; bottom_cabinet/jhymlr x1 scale [2.44, 5.11, 2.72]; bottom_cabinet/rvunhj x6 scale [1.6, 1.53, 1.72]; door/ggcyib x6 scale [1.0, 1.0, 1.0]; door/lztyfn x1 scale [1.0, 1.0, 1.0]; door/pwoerr x2 scale [1.0, 1.0, 1.0]; toilet/udiezm x5 scale [0.92, 0.91, 0.33]

