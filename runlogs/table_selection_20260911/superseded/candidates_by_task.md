# Table / work-surface candidates per task (sampler-validated)

Status `ok` = `TaskSampler.setup()` succeeded on that support and `TaskSampler.run()` wrote >= 1 instance (init settled and BEHAVIOR-validated, `:init` true, goal false at init, goal feasible by teleport + settle). Details: `results_summary.csv`, `results/<scene>__<table>__<problem>.json`.

## disembedding-0

### house_double_floor_upper
- (no viable support)

  rejected:
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral +0.05, footprint u 1.85..2.29 vs support (-2. (tags: no_support, support, no_stance)
  - bottom_cabinet_nddvba_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_nddvba_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral -0.04, footprint u 4.28..4.71 vs support (-4. (tags: no_support, support, no_stance)
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_aduafr_0 [desk]: setup_failed - no_support: [no_support] desk_aduafr_0: [support] untucked: [support] edge +u lateral +0.00: [unreachable] no (cell, yaw, grasp) is both graspable on the source tray and placeable on the destination tray (tags: no_support, support, unreachable, no_stance)
  - desk_iotfzl_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_uqcmzf_0 [desk]: setup_failed - no_support: [no_support] desk_uqcmzf_0: [support] untucked: [support] edge +v lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral -0.05, footprint u 3.00..3.44 vs support (2.48177552223 (tags: no_support, support, no_stance)
  - nightstand_roxnbe_0 [nightstand]: setup_failed - no_support: [no_support] nightstand_roxnbe_0: [support] untucked: [support] edge +u: [no_stance] no collision-free stance around nightstand_roxnbe_0 (tags: no_support, support, no_stance)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - ottoman_qsnawv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_qsnawv_0: [support] untucked: [support] edge +v: [no_stance] no collision-free stance around ottoman_qsnawv_0 (tags: no_support, support, no_stance)
  - serving_cart_vvtcby_0 [serving_cart]: setup_failed - no_support: [no_support] serving_cart_vvtcby_0: [support] untucked: [support] edge +u: [no_stance] no collision-free stance around serving_cart_vvtcby_0 (tags: no_support, support, no_stance)

### Rs_int
- (no viable support)

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - no stance around the support (precheck) and inter_object_relation_satisfaction-0 failed at setup
  - bottom_cabinet_slgzfc_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_slgzfc_0: [support] untucked: [support] edge +u lateral +0.00: [support] no lateral room for the destination tray on either side (support bounds (-1.8401092290878296, -1.3601082563400269, -2.7 (tags: no_support, support, ('right', -0.58), ('left', 0.58), no_stance)
  - countertop_tpuwys_0 [countertop]: setup_failed - no_support: [no_support] countertop_tpuwys_0: [support] untucked: [support] edge +u: [no_stance] no collision-free stance around countertop_tpuwys_0 (tags: no_support, support, no_stance)

### hotel_suite_large
- (no viable support)

  rejected:
  - desk_ohjotf_0 [desk]: setup_failed - no_support: [no_support] desk_ohjotf_0: [support] untucked: [support] edge +u lateral +0.00: [support] destination tray visibility out of range (dest 10473 px, source 0 px) (tags: no_support, support, no_stance)
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_0: [support] untucked: [support] edge +u lateral +0.00: [support] tray_source_tray_gsxbym_0: no raycast hits (tags: no_support, support, no_stance)
  - ottoman_hnmndv_1 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_rglkkf_0 [shelf]: setup_failed - no_support: [no_support] shelf_rglkkf_0: [support] untucked: [support] edge +u lateral +0.00: [support] no view / side keeps the source tray in view with the destination tray barely visible ({'views': 40, 'collision_free': 17, 'src_ (tags: no_support, support, ('right', -0.62), ('left', 0.62), unreachable, no_stance)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### restaurant_diner
- (no viable support)

  rejected:
  - bar_xxftww_0 [bar]: error - error: RuntimeError('stack expects a non-empty TensorList')
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_2 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_2: [support] untucked: [support] edge +u: [no_stance] no traversable stance around bench_xwphjd_2 (tags: no_support, support, no_stance)
  - bench_xwphjd_3 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_3: [support] untucked: [support] edge +u: [no_stance] no traversable stance around bench_xwphjd_3 (tags: no_support, support, no_stance)
  - bench_xwphjd_4 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_4: [support] untucked: [support] edge +u: [no_stance] no traversable stance around bench_xwphjd_4 (tags: no_support, support, no_stance)
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - booth_xzrpar_0 [booth]: setup_failed - spec: [spec] no 'distractors' pool for target category 'box_of_cereal' (spec section distractors.pools)
  - booth_xzrpar_1 [booth]: setup_failed - spec: [spec] no 'distractors' pool for target category 'box_of_cereal' (spec section distractors.pools)
  - booth_xzrpar_2 [booth]: setup_failed - spec: [spec] no 'distractors' pool for target category 'box_of_cereal' (spec section distractors.pools)
  - booth_xzrpar_3 [booth]: setup_failed - spec: [spec] no 'distractors' pool for target category 'box_of_cereal' (spec section distractors.pools)
  - bottom_cabinet_olgoza_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_olgoza_0: [support] untucked: [support] edge +u lateral +0.00: [unreachable] no (cell, yaw, grasp) is both graspable on the source tray and placeable on the destination tray (tags: no_support, support, unreachable, no_stance)
  - pedestal_table_nlfgwe_0 [pedestal_table]: setup_failed - spec: [spec] no 'distractors' pool for target category 'box_of_cereal' (spec section distractors.pools)
  - pedestal_table_nlfgwe_1 [pedestal_table]: setup_failed - spec: [spec] no 'distractors' pool for target category 'box_of_cereal' (spec section distractors.pools)
  - pedestal_table_nlfgwe_2 [pedestal_table]: setup_failed - spec: [spec] no 'distractors' pool for target category 'box_of_cereal' (spec section distractors.pools)

## configuration_matching-0

### house_double_floor_upper
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.88 m, 1113 free cells; occupants ['picture_xnuynn_0']; stance edge +u base (1.39, 2.07, yaw 0.00); trunk translate 0.75 tilt -0.30; n_reachable_cells 15
- **bottom_cabinet_nddvba_0** [bottom_cabinet/nddvba, rooms bedroom_2]: 1 instance(s) in 3 attempt(s), rejections {'layout': 2}; top z 0.72 m, 779 free cells; stance edge +u base (4.49, -3.86, yaw -1.57); trunk translate 0.75 tilt -0.55; n_reachable_cells 11

  rejected:
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_aduafr_0 [desk]: no_instances - template ok, no instance in 6 attempts: layout x6
  - desk_iotfzl_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_uqcmzf_0 [desk]: setup_failed - no_support: [no_support] desk_uqcmzf_0: [support] placement region empty (tags: no_support, support)
  - nightstand_roxnbe_0 [nightstand]: no_instances - template ok, no instance in 7 attempts: layout x6, scene_reset x1
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - ottoman_qsnawv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_qsnawv_0: [unreachable] only 0 reachable cells (tags: no_support, unreachable)
  - serving_cart_vvtcby_0 [serving_cart]: setup_failed - no_support: [no_support] serving_cart_vvtcby_0: [unreachable] only 0 reachable cells (tags: no_support, unreachable)

### Rs_int
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 15
- **countertop_tpuwys_0** [countertop/tpuwys, rooms kitchen_0;living_room_0]: 1 instance(s) in 31 attempt(s), rejections {'scene_reset': 26, 'scene_disturbed': 4}; top z 1.08 m, 1440 free cells; stance edge -u base (-0.86, 2.13, yaw -1.57); trunk translate 0.50 tilt -0.15; n_reachable_cells 12

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - no stance around the support (precheck) and inter_object_relation_satisfaction-0 failed at setup

### hotel_suite_large
- (no viable support)

  rejected:
  - desk_ohjotf_0 [desk]: no_instances - template ok, no instance in 13 attempts: scene_reset x7, layout x6
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_0: [support] placement region empty (tags: no_support, support)
  - ottoman_hnmndv_1 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_rglkkf_0 [shelf]: no_instances - template ok, no instance in 6 attempts: layout x6
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### restaurant_diner
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'layout': 1}; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 11
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.50 tilt -0.45; n_reachable_cells 7
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'layout': 1}; top z 0.78 m, 2222 free cells; occupants ['straight_chair_nntxvr_0']; stance edge +u base (-2.38, 2.12, yaw 3.14); trunk translate 0.75 tilt -0.65; n_reachable_cells 10
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 3 attempt(s), rejections {'layout': 2}; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.75 tilt -0.65; n_reachable_cells 15
- **pedestal_table_nlfgwe_2** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -6.73, yaw -0.00); trunk translate 0.75 tilt -0.65; n_reachable_cells 16

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [support] placement region empty (tags: no_support, support)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_2 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_2: [unreachable] only 0 reachable cells (tags: no_support, unreachable)
  - bench_xwphjd_3 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_3: [unreachable] only 0 reachable cells (tags: no_support, unreachable)
  - bench_xwphjd_4 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_4: [unreachable] only 0 reachable cells (tags: no_support, unreachable)
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - booth_xzrpar_0 [booth]: no_instances - template ok, no instance in 6 attempts: layout x6
  - booth_xzrpar_1 [booth]: no_instances - template ok, no instance in 6 attempts: layout x6
  - booth_xzrpar_3 [booth]: no_instances - template ok, no instance in 6 attempts: layout x6

## configuration_matching-1

### house_double_floor_upper
- (no viable support)

  rejected:
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [support] stand_off 0.45: 0 visible of 30 cells, none supported+visible (view translate 0.0) (tags: no_support, support)
  - bottom_cabinet_nddvba_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_nddvba_0: [support] stand_off 0.45: 0 visible of 27 cells, none supported+visible (view translate 0.75) (tags: no_support, support)
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_aduafr_0 [desk]: setup_failed - no_support: [no_support] desk_aduafr_0: [unreachable] handle not reachable from any of 9 cells (e.g. handle at [-5.4426350593566895, -0.05020727962255478, 1.1287872791290283], base (-5.236870765686035, 0.8917053937911987, -1.5753529 (tags: no_support, unreachable, -5.4426350593566895, -0.05020727962255478, 1.1287872791290283, 0.0, 0.25, 0.5)
  - desk_iotfzl_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_uqcmzf_0 [desk]: setup_failed - no_support: [no_support] desk_uqcmzf_0: [support] stand_off 0.45: placement region empty (tags: no_support, support)
  - nightstand_roxnbe_0 [nightstand]: no_instances - template ok, no instance in 7 attempts: target_unsupported x4, out_of_view x2, scene_reset x1
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - ottoman_qsnawv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_qsnawv_0: [support] stand_off 0.45: 0 visible of 5 cells, none supported+visible (view translate 0.75) (tags: no_support, support)
  - serving_cart_vvtcby_0 [serving_cart]: setup_failed - no_support: [no_support] serving_cart_vvtcby_0: [support] stand_off 0.45: placement region empty (tags: no_support, support)

### Rs_int
- (no viable support)

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - no stance around the support (precheck) and inter_object_relation_satisfaction-0 failed at setup
  - bottom_cabinet_slgzfc_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_slgzfc_0: [unreachable] handle not reachable from any of 0 cells (e.g. handle at None, base (0.8101062774658203, 2.2857182025909424, 8.662515256217028e-07), translates [0.0, 0.25, 0.5]) (tags: no_support, unreachable, 0.0, 0.25, 0.5)

### hotel_suite_large
- (no viable support)

  rejected:
  - desk_ohjotf_0 [desk]: setup_failed - no_support: [no_support] desk_ohjotf_0: [support] stand_off 0.45: 0 visible of 40 cells, none supported+visible (view translate 0.0) (tags: no_support, support)
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_0: [support] stand_off 0.45: placement region empty (tags: no_support, support)
  - ottoman_hnmndv_1 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_rglkkf_0 [shelf]: setup_failed - no_support: [no_support] shelf_rglkkf_0: [support] stand_off 0.45: 0 visible of 105 cells, none supported+visible (view translate 0.75) (tags: no_support, support)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### restaurant_diner
- (no viable support)

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [support] stand_off 0.45: placement region empty (tags: no_support, support)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_2 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_2: [support] stand_off 0.45: placement region empty (tags: no_support, support)
  - bench_xwphjd_3 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_3: [support] stand_off 0.45: placement region empty (tags: no_support, support)
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - booth_xzrpar_0 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_0: [unreachable] handle not reachable from any of 2 cells (e.g. handle at [0.2267158329486847, 2.6330230236053467, 1.1091846227645874], base (-0.5766842365264893, 2.6547231674194336, -1.62920684 (tags: no_support, unreachable, 0.2267158329486847, 2.6330230236053467, 1.1091846227645874, 0.0, 0.25, 0.5)
  - booth_xzrpar_1 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_1: [unreachable] handle not reachable from any of 2 cells (e.g. handle at [0.22671595215797424, 0.382412314414978, 1.1091842651367188], base (-0.5766841173171997, 0.40411248803138733, -1.6292068 (tags: no_support, unreachable, 0.22671595215797424, 0.382412314414978, 1.1091842651367188, 0.0, 0.25, 0.5)
  - booth_xzrpar_2 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_2: [unreachable] handle not reachable from any of 2 cells (e.g. handle at [0.09517262130975723, -8.772573471069336, 1.1091824769973755], base (0.11687394976615906, -7.909173011779785, -1.5707974 (tags: no_support, unreachable, 0.09517262130975723, -8.772573471069336, 1.1091824769973755, 0.0, 0.25, 0.5)
  - booth_xzrpar_3 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_3: [unreachable] handle not reachable from any of 2 cells (e.g. handle at [-2.155437707901001, -8.772574424743652, 1.109182596206665], base (-2.1337361335754395, -7.909174919128418, -1.570797431 (tags: no_support, unreachable, -2.155437707901001, -8.772574424743652, 1.109182596206665, 0.0, 0.25, 0.5)
  - bottom_cabinet_olgoza_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_olgoza_0: [unreachable] handle not reachable from any of 1 cells (e.g. handle at [-4.61296272277832, -2.013155460357666, 1.1971007585525513], base (-3.808410882949829, -2.014829158782959, -3.1 (tags: no_support, unreachable, -4.61296272277832, -2.013155460357666, 1.1971007585525513, 0.0, 0.25, 0.5)
  - pedestal_table_nlfgwe_0 [pedestal_table]: setup_failed - no_support: [no_support] pedestal_table_nlfgwe_0: [support] stand_off 0.45: 0 visible of 54 cells, none supported+visible (view translate 0.0) (tags: no_support, support)
  - pedestal_table_nlfgwe_1 [pedestal_table]: setup_failed - no_support: [no_support] pedestal_table_nlfgwe_1: [unreachable] handle not reachable from any of 0 cells (e.g. handle at None, base (-0.32638320326805115, -4.161439418792725, -0.00028657914131617347), translates [0.0, 0.25, 0.5]) (tags: no_support, unreachable, 0.0, 0.25, 0.5)

## configuration_matching-2

### house_double_floor_upper
- (no viable support)

  rejected:
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_nddvba_0 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - configuration_matching-1 produced no instance on this support
  - desk_aduafr_0 [desk]: skipped - configuration_matching-1 produced no instance on this support
  - desk_iotfzl_0 [desk]: skipped - configuration_matching-1 produced no instance on this support
  - desk_uqcmzf_0 [desk]: skipped - configuration_matching-1 produced no instance on this support
  - nightstand_roxnbe_0 [nightstand]: skipped - configuration_matching-1 produced no instance on this support
  - nightstand_roxnbe_1 [nightstand]: skipped - configuration_matching-1 produced no instance on this support
  - ottoman_qsnawv_0 [ottoman]: skipped - configuration_matching-1 produced no instance on this support
  - serving_cart_vvtcby_0 [serving_cart]: skipped - configuration_matching-1 produced no instance on this support

### Rs_int
- (no viable support)

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_bamfsz_1 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_slgzfc_0 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - breakfast_table_skczfi_0 [breakfast_table]: skipped - configuration_matching-1 produced no instance on this support
  - breakfast_table_skczfi_1 [breakfast_table]: skipped - configuration_matching-1 produced no instance on this support
  - countertop_tpuwys_0 [countertop]: skipped - configuration_matching-1 produced no instance on this support

### hotel_suite_large
- (no viable support)

  rejected:
  - desk_ohjotf_0 [desk]: skipped - configuration_matching-1 produced no instance on this support
  - nightstand_maeowc_0 [nightstand]: skipped - configuration_matching-1 produced no instance on this support
  - nightstand_maeowc_1 [nightstand]: skipped - configuration_matching-1 produced no instance on this support
  - nightstand_maeowc_2 [nightstand]: skipped - configuration_matching-1 produced no instance on this support
  - ottoman_hnmndv_0 [ottoman]: skipped - configuration_matching-1 produced no instance on this support
  - ottoman_hnmndv_1 [ottoman]: skipped - configuration_matching-1 produced no instance on this support
  - shelf_aqtmhm_0 [shelf]: skipped - configuration_matching-1 produced no instance on this support
  - shelf_rglkkf_0 [shelf]: skipped - configuration_matching-1 produced no instance on this support
  - shelf_xogqqm_1 [shelf]: skipped - configuration_matching-1 produced no instance on this support
  - shelf_xogqqm_2 [shelf]: skipped - configuration_matching-1 produced no instance on this support

### restaurant_diner
- (no viable support)

  rejected:
  - bar_xxftww_0 [bar]: skipped - configuration_matching-1 produced no instance on this support
  - bench_xwphjd_0 [bench]: skipped - configuration_matching-1 produced no instance on this support
  - bench_xwphjd_1 [bench]: skipped - configuration_matching-1 produced no instance on this support
  - bench_xwphjd_2 [bench]: skipped - configuration_matching-1 produced no instance on this support
  - bench_xwphjd_3 [bench]: skipped - configuration_matching-1 produced no instance on this support
  - bench_xwphjd_4 [bench]: skipped - configuration_matching-1 produced no instance on this support
  - bench_xwphjd_5 [bench]: skipped - configuration_matching-1 produced no instance on this support
  - bench_xwphjd_6 [bench]: skipped - configuration_matching-1 produced no instance on this support
  - bench_xwphjd_7 [bench]: skipped - configuration_matching-1 produced no instance on this support
  - booth_xzrpar_0 [booth]: skipped - configuration_matching-1 produced no instance on this support
  - booth_xzrpar_1 [booth]: skipped - configuration_matching-1 produced no instance on this support
  - booth_xzrpar_2 [booth]: skipped - configuration_matching-1 produced no instance on this support
  - booth_xzrpar_3 [booth]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_olgoza_0 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - pedestal_table_nlfgwe_0 [pedestal_table]: skipped - configuration_matching-1 produced no instance on this support
  - pedestal_table_nlfgwe_1 [pedestal_table]: skipped - configuration_matching-1 produced no instance on this support

## configuration_matching-3

### house_double_floor_upper
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.88 m, 1113 free cells; occupants ['picture_xnuynn_0']; stance edge +u base (1.39, 2.07, yaw 0.00); trunk translate 0.75 tilt -0.30; n_reachable_cells 12; n_grasp_cells 12
- **bottom_cabinet_nddvba_0** [bottom_cabinet/nddvba, rooms bedroom_2]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.72 m, 779 free cells; stance edge +u base (4.49, -3.86, yaw -1.57); trunk translate 0.75 tilt -0.55; n_reachable_cells 11; n_grasp_cells 11
- **desk_aduafr_0** [desk/aduafr, rooms bedroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 2100 free cells; stance edge +u base (-5.43, 0.79, yaw -1.58); trunk translate 0.50 tilt -0.70; n_reachable_cells 7; n_grasp_cells 7
- **nightstand_roxnbe_0** [nightstand/roxnbe, rooms bedroom_1]: 1 instance(s) in 2 attempt(s), rejections {'scene_reset': 1}; top z 0.70 m, 1028 free cells; occupants ['table_lamp_erfbgw_0']; stance edge +u base (5.97, 3.04, yaw 1.57); trunk translate 0.00 tilt -0.90; arms ready; n_reachable_cells 9; n_grasp_cells 7

  rejected:
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_iotfzl_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_uqcmzf_0 [desk]: setup_failed - no_support: [no_support] desk_uqcmzf_0: [unreachable] no reachable cell for the male part (tags: no_support, unreachable)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - ottoman_qsnawv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_qsnawv_0: [unreachable] no reachable cell for the male part (tags: no_support, unreachable)
  - serving_cart_vvtcby_0 [serving_cart]: setup_failed - no_support: [no_support] serving_cart_vvtcby_0: [unreachable] no reachable cell for the male part (tags: no_support, unreachable)

### Rs_int
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 13; n_grasp_cells 13

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - no stance around the support (precheck) and inter_object_relation_satisfaction-0 failed at setup
  - countertop_tpuwys_0 [countertop]: no_instances - template ok, no instance in 45 attempts: scene_reset x37, partner_placement x5, scene_disturbed x3

### hotel_suite_large
- **desk_ohjotf_0** [desk/ohjotf, rooms bedroom_0]: 1 instance(s) in 7 attempt(s), rejections {'scene_reset': 4, 'scene_disturbed': 1, 'target_placement': 1}; top z 0.68 m, 2403 free cells; stance edge +v base (-1.71, -4.63, yaw 0.00); trunk translate 0.00 tilt -0.90; n_reachable_cells 7; n_grasp_cells 7
- **shelf_rglkkf_0** [shelf/rglkkf, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.66 m, 2064 free cells; stance edge +u base (0.96, 3.27, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 7; n_grasp_cells 7

  rejected:
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_0: [unreachable] no reachable cell for the male part (tags: no_support, unreachable)
  - ottoman_hnmndv_1 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.75 tilt -0.70; n_reachable_cells 9; n_grasp_cells 9
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 5 attempt(s), rejections {'partner_placement': 4}; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.75 tilt -0.70; n_reachable_cells 9; n_grasp_cells 9
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.70; n_reachable_cells 9; n_grasp_cells 9
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.70; n_reachable_cells 8; n_grasp_cells 6
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.75 tilt -0.20; n_reachable_cells 5; n_grasp_cells 5
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; occupants ['straight_chair_nntxvr_0']; stance edge +u base (-2.38, 2.12, yaw 3.14); trunk translate 0.75 tilt -0.65; n_reachable_cells 9; n_grasp_cells 8
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 5 attempt(s), rejections {'partner_placement': 4}; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.75 tilt -0.65; n_reachable_cells 11; n_grasp_cells 9
- **pedestal_table_nlfgwe_2** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -6.73, yaw -0.00); trunk translate 0.75 tilt -0.65; n_reachable_cells 11; n_grasp_cells 9

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [support] placement region empty (tags: no_support, support)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_2 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_2: [unreachable] no reachable cell for the male part (tags: no_support, unreachable)
  - bench_xwphjd_3 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_3: [unreachable] no reachable cell for the male part (tags: no_support, unreachable)
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

## world_alignment-0

### house_double_floor_upper
- (no viable support)

  rejected:
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - bottom_cabinet_nddvba_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_nddvba_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_aduafr_0 [desk]: setup_failed - no_support: [no_support] desk_aduafr_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - desk_iotfzl_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_uqcmzf_0 [desk]: setup_failed - no_support: [no_support] desk_uqcmzf_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - nightstand_roxnbe_0 [nightstand]: setup_failed - no_support: [no_support] nightstand_roxnbe_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - ottoman_qsnawv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_qsnawv_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - serving_cart_vvtcby_0 [serving_cart]: setup_failed - no_support: [no_support] serving_cart_vvtcby_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)

### Rs_int
- (no viable support)

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - no stance around the support (precheck) and inter_object_relation_satisfaction-0 failed at setup
  - bottom_cabinet_slgzfc_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_slgzfc_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - countertop_tpuwys_0 [countertop]: setup_failed - no_support: [no_support] countertop_tpuwys_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)

### hotel_suite_large
- **desk_ohjotf_0** [desk/ohjotf, rooms bedroom_0]: 1 instance(s) in 3 attempt(s), rejections {'scene_reset': 2}; top z 0.68 m, 2403 free cells; stance edge +v base (-1.71, -4.63, yaw 0.00); trunk translate 0.50 tilt -0.55

  rejected:
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - ottoman_hnmndv_1 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_rglkkf_0 [shelf]: setup_failed - no_support: [no_support] shelf_rglkkf_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.50 tilt -0.60
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.50 tilt -0.60
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.50 tilt -0.60
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.50 tilt -0.60
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; occupants ['straight_chair_nntxvr_0']; stance edge +u base (-2.38, 2.12, yaw 3.14); trunk translate 0.25 tilt -0.70
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.25 tilt -0.70

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_2 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_2: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - bench_xwphjd_3 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_3: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bottom_cabinet_olgoza_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_olgoza_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)

## inter_object_relation_satisfaction-0

### house_double_floor_upper
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.88 m, 1113 free cells; occupants ['picture_xnuynn_0']; stance edge +u base (1.39, 2.07, yaw 0.00); trunk translate 0.75 tilt -0.25; n_reachable_cells 12; n_grasp_cells 11
- **bottom_cabinet_nddvba_0** [bottom_cabinet/nddvba, rooms bedroom_2]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.72 m, 779 free cells; stance edge +u base (4.49, -3.86, yaw -1.57); trunk translate 0.75 tilt -0.55; n_reachable_cells 6; n_grasp_cells 6
- **desk_aduafr_0** [desk/aduafr, rooms bedroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 2100 free cells; stance edge +u base (-5.43, 0.79, yaw -1.58); trunk translate 0.25 tilt -0.75; n_reachable_cells 8; n_grasp_cells 8

  rejected:
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: setup_failed - no_support: [no_support] bottom_cabinet_no_top_ncgmqr_0: [support] bottom_cabinet_no_top_ncgmqr_0 has too little free top surface (tags: no_support, support)
  - desk_iotfzl_0 [desk]: setup_failed - no_support: [no_support] desk_iotfzl_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around desk_iotfzl_0 (tags: no_support, no_stance, stance_collision)
  - desk_uqcmzf_0 [desk]: setup_failed - no_support: [no_support] desk_uqcmzf_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - nightstand_roxnbe_0 [nightstand]: no_instances - template ok, no instance in 7 attempts: target_placement x4, partner_placement x2, scene_reset x1
  - nightstand_roxnbe_1 [nightstand]: setup_failed - no_support: [no_support] nightstand_roxnbe_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around nightstand_roxnbe_1 (tags: no_support, no_stance, stance_collision)
  - ottoman_qsnawv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_qsnawv_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - serving_cart_vvtcby_0 [serving_cart]: setup_failed - no_support: [no_support] serving_cart_vvtcby_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable, stance_collision)

### Rs_int
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 12; n_grasp_cells 9

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_bamfsz_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no traversable stance around bottom_cabinet_bamfsz_0 (tags: no_support, no_stance, stance_collision)
  - countertop_tpuwys_0 [countertop]: no_instances - template ok, no instance in 40 attempts: scene_reset x32, partner_placement x7, target_placement x1

### hotel_suite_large
- **desk_ohjotf_0** [desk/ohjotf, rooms bedroom_0]: 1 instance(s) in 7 attempt(s), rejections {'scene_reset': 4, 'target_placement': 2}; top z 0.68 m, 2403 free cells; stance edge +v base (-1.71, -4.63, yaw -0.00); trunk translate 0.00 tilt -0.85; n_reachable_cells 8; n_grasp_cells 6
- **shelf_rglkkf_0** [shelf/rglkkf, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.66 m, 2064 free cells; stance edge +u base (0.96, 3.27, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 4; n_grasp_cells 3

  rejected:
  - nightstand_maeowc_0 [nightstand]: setup_failed - no_support: [no_support] nightstand_maeowc_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around nightstand_maeowc_0 (tags: no_support, no_stance, stance_collision)
  - nightstand_maeowc_1 [nightstand]: setup_failed - no_support: [no_support] nightstand_maeowc_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around nightstand_maeowc_1 (tags: no_support, no_stance, stance_collision)
  - nightstand_maeowc_2 [nightstand]: setup_failed - no_support: [no_support] nightstand_maeowc_2: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around nightstand_maeowc_2 (tags: no_support, no_stance, stance_collision)
  - ottoman_hnmndv_0 [ottoman]: no_instances - template ok, no instance in 12 attempts: scene_reset x6, partner_placement x3, target_placement x3
  - ottoman_hnmndv_1 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_1: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - shelf_aqtmhm_0 [shelf]: setup_failed - no_support: [no_support] shelf_aqtmhm_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around shelf_aqtmhm_0 (tags: no_support, no_stance, stance_collision)
  - shelf_xogqqm_1 [shelf]: setup_failed - no_support: [no_support] shelf_xogqqm_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around shelf_xogqqm_1 (tags: no_support, no_stance, stance_collision, support)
  - shelf_xogqqm_2 [shelf]: setup_failed - no_support: [no_support] shelf_xogqqm_2: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around shelf_xogqqm_2 (tags: no_support, no_stance, stance_collision, unreachable)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 7; n_grasp_cells 6
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 7; n_grasp_cells 6
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 4; n_grasp_cells 4
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 4; n_grasp_cells 4
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.50 tilt -0.45; n_reachable_cells 7; n_grasp_cells 7
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; occupants ['straight_chair_nntxvr_0']; stance edge +u base (-2.38, 2.12, yaw 3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 8; n_grasp_cells 8
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 9; n_grasp_cells 9
- **pedestal_table_nlfgwe_2** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -6.73, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 9; n_grasp_cells 9

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, unreachable)
  - bench_xwphjd_0 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around bench_xwphjd_0 (tags: no_support, no_stance, stance_collision)
  - bench_xwphjd_1 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (33 candidates) (tags: no_support, no_stance, stance_collision, unreachable)
  - bench_xwphjd_2 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_2: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_3 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_3: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_4 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_4: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_5 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_5: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_6 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_6: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around bench_xwphjd_6 (tags: no_support, no_stance, stance_collision)
  - bench_xwphjd_7 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_7: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)

## inter_object_relation_satisfaction-1

## inter_object_relation_satisfaction-2

### house_double_floor_upper
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.88 m, 1113 free cells; occupants ['picture_xnuynn_0']; stance edge +u base (1.39, 2.07, yaw 0.00); trunk translate 0.75 tilt -0.25; n_reachable_cells 13; n_grasp_cells 12
- **bottom_cabinet_nddvba_0** [bottom_cabinet/nddvba, rooms bedroom_2]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.72 m, 779 free cells; stance edge +u base (4.49, -3.86, yaw -1.57); trunk translate 0.75 tilt -0.55; n_reachable_cells 11; n_grasp_cells 11
- **desk_aduafr_0** [desk/aduafr, rooms bedroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 2100 free cells; stance edge +u base (-5.43, 0.79, yaw -1.58); trunk translate 0.25 tilt -0.75; n_reachable_cells 9; n_grasp_cells 9
- **nightstand_roxnbe_0** [nightstand/roxnbe, rooms bedroom_1]: 1 instance(s) in 4 attempt(s), rejections {'scene_reset': 1, 'target_placement': 2}; top z 0.70 m, 1028 free cells; occupants ['table_lamp_erfbgw_0']; stance edge +u base (5.97, 3.04, yaw 1.57); trunk translate 0.00 tilt -0.90; arms ready; n_reachable_cells 11; n_grasp_cells 11

  rejected:
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_iotfzl_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_uqcmzf_0 [desk]: setup_failed - no_support: [no_support] desk_uqcmzf_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - ottoman_qsnawv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_qsnawv_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - serving_cart_vvtcby_0 [serving_cart]: setup_failed - no_support: [no_support] serving_cart_vvtcby_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable, stance_collision)

### Rs_int
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 14; n_grasp_cells 14

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - no stance around the support (precheck) and inter_object_relation_satisfaction-0 failed at setup

### hotel_suite_large
- **desk_ohjotf_0** [desk/ohjotf, rooms bedroom_0]: 1 instance(s) in 11 attempt(s), rejections {'scene_reset': 6, 'scene_disturbed': 1, 'target_placement': 3}; top z 0.68 m, 2403 free cells; stance edge +v base (-1.71, -4.63, yaw 0.00); trunk translate 0.00 tilt -0.85; n_reachable_cells 8; n_grasp_cells 7
- **shelf_rglkkf_0** [shelf/rglkkf, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.66 m, 2064 free cells; stance edge +u base (0.96, 3.27, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 10; n_grasp_cells 8

  rejected:
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - ottoman_hnmndv_1 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 11; n_grasp_cells 9
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 11; n_grasp_cells 9
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 9; n_grasp_cells 9
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 9; n_grasp_cells 9
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.50 tilt -0.45; n_reachable_cells 7; n_grasp_cells 7
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; occupants ['straight_chair_nntxvr_0']; stance edge +u base (-2.38, 2.12, yaw 3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 10; n_grasp_cells 10
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 14; n_grasp_cells 13

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, unreachable)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_2 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_2: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_3 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_3: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

## inter_object_relation_satisfaction-3

### house_double_floor_upper
- **desk_aduafr_0** [desk/aduafr, rooms bedroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 2100 free cells; stance edge +u base (-5.43, 0.79, yaw -1.58); trunk translate 0.25 tilt -0.75; n_reachable_cells 7; n_grasp_cells 7
- **nightstand_roxnbe_0** [nightstand/roxnbe, rooms bedroom_1]: 1 instance(s) in 8 attempt(s), rejections {'scene_reset': 2, 'partner_placement': 3, 'unstable_init': 1, 'target_placement': 1}; top z 0.70 m, 1028 free cells; occupants ['table_lamp_erfbgw_0']; stance edge +u base (5.97, 3.04, yaw 1.57); trunk translate 0.00 tilt -0.90; arms ready; n_reachable_cells 11; n_grasp_cells 11

  rejected:
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: no_instances - template ok, no instance in 8 attempts: target_placement x5, partner_placement x3
  - bottom_cabinet_nddvba_0 [bottom_cabinet]: no_instances - template ok, no instance in 6 attempts: target_placement x6
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_iotfzl_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_uqcmzf_0 [desk]: setup_failed - no_support: [no_support] desk_uqcmzf_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - ottoman_qsnawv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_qsnawv_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - serving_cart_vvtcby_0 [serving_cart]: setup_failed - no_support: [no_support] serving_cart_vvtcby_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable, stance_collision)

### Rs_int
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 5 attempt(s), rejections {'partner_placement': 2, 'target_placement': 2}; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 7; n_grasp_cells 7

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - no stance around the support (precheck) and inter_object_relation_satisfaction-0 failed at setup

### hotel_suite_large
- **shelf_rglkkf_0** [shelf/rglkkf, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.66 m, 2064 free cells; stance edge +u base (0.96, 3.27, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 1; n_grasp_cells 1

  rejected:
  - desk_ohjotf_0 [desk]: no_instances - template ok, no instance in 13 attempts: scene_reset x7, target_placement x6
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - ottoman_hnmndv_1 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 3 attempt(s), rejections {'partner_placement': 2}; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 4; n_grasp_cells 4
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 5 attempt(s), rejections {'partner_placement': 4}; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 4; n_grasp_cells 4
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 4 attempt(s), rejections {'partner_placement': 3}; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 1; n_grasp_cells 1
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 1; n_grasp_cells 1
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.50 tilt -0.45; n_reachable_cells 6; n_grasp_cells 6
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 3 attempt(s), rejections {'partner_placement': 2}; top z 0.78 m, 2222 free cells; occupants ['straight_chair_nntxvr_0']; stance edge +u base (-2.38, 2.12, yaw 3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 4; n_grasp_cells 4

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, unreachable)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_2 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_2: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_3 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_3: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - pedestal_table_nlfgwe_1 [pedestal_table]: no_instances - template ok, no instance in 6 attempts: partner_placement x4, unstable_init x2

## inter_object_relation_satisfaction-4

### house_double_floor_upper
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms bathroom_0]: 1 instance(s) in 4 attempt(s), rejections {'goal_infeasible': 3}; top z 0.88 m, 1113 free cells; occupants ['picture_xnuynn_0']; stance edge +u base (1.39, 2.07, yaw 0.00); trunk translate 0.75 tilt -0.25; n_reachable_cells 12
- **bottom_cabinet_nddvba_0** [bottom_cabinet/nddvba, rooms bedroom_2]: 1 instance(s) in 6 attempt(s), rejections {'target_placement': 4, 'goal_infeasible': 1}; top z 0.72 m, 779 free cells; stance edge +u base (4.49, -3.86, yaw -1.57); trunk translate 0.75 tilt -0.55; n_reachable_cells 13
- **desk_aduafr_0** [desk/aduafr, rooms bedroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 2100 free cells; stance edge +u base (-5.43, 0.79, yaw -1.58); trunk translate 0.25 tilt -0.75; n_reachable_cells 10
- **nightstand_roxnbe_0** [nightstand/roxnbe, rooms bedroom_1]: 1 instance(s) in 2 attempt(s), rejections {'scene_reset': 1}; top z 0.70 m, 1028 free cells; occupants ['table_lamp_erfbgw_0']; stance edge +u base (5.97, 3.04, yaw 1.57); trunk translate 0.00 tilt -0.90; arms ready; n_reachable_cells 11

  rejected:
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_iotfzl_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - desk_uqcmzf_0 [desk]: setup_failed - no_support: [no_support] desk_uqcmzf_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - ottoman_qsnawv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_qsnawv_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - serving_cart_vvtcby_0 [serving_cart]: no_instances - template ok, no instance in 8 attempts: target_placement x8

### Rs_int
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 3 attempt(s), rejections {'goal_infeasible': 2}; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 14

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - no stance around the support (precheck) and inter_object_relation_satisfaction-0 failed at setup

### hotel_suite_large
- **shelf_rglkkf_0** [shelf/rglkkf, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.66 m, 2064 free cells; stance edge +u base (0.96, 3.27, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 8

  rejected:
  - desk_ohjotf_0 [desk]: no_instances - template ok, no instance in 15 attempts: scene_reset x9, scene_disturbed x4, target_placement x1, partner_placement x1
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - ottoman_hnmndv_1 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 3 attempt(s), rejections {'target_placement': 1, 'goal_infeasible': 1}; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 10
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 10
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 8
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 4 attempt(s), rejections {'goal_infeasible': 1, 'partner_placement': 1, 'target_placement': 1}; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 8
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.50 tilt -0.45; n_reachable_cells 7
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 3 attempt(s), rejections {'goal_infeasible': 2}; top z 0.78 m, 2222 free cells; occupants ['straight_chair_nntxvr_0']; stance edge +u base (-2.38, 2.12, yaw 3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 9
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 13

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, unreachable)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support (precheck: none either)
  - bench_xwphjd_2 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_2: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_3 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_3: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

