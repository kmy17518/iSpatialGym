# Table / work-surface candidates per task (sampler-validated)

Status `ok` = `TaskSampler.setup()` succeeded on that support and `TaskSampler.run()` wrote >= 1 instance (init settled and BEHAVIOR-validated, `:init` true, goal false at init, goal feasible by teleport + settle). Details: `results_summary.csv`, `results/<scene>__<table>__<problem>.json`.

## disembedding-0

### house_single_floor
- **breakfast_table_rhjoby_0** [breakfast_table/rhjoby, rooms dining_room_0]: 1 instance(s) in 7 attempt(s), rejections {'unstable_init': 3, 'goal_infeasible': 2, 'distractor_placement': 1}; top z 0.81 m, 8463 free cells; stance edge -v base (9.19, 3.16, yaw 3.14); trunk translate 0.75 tilt -0.40; n_reachable_cells 6; n_grasp_cells 1

  rejected:
  - bar_udatjt_0 [bar]: setup_failed - no_support: [no_support] bar_udatjt_0: [support] untucked: [support] edge +u lateral +0.00: [unreachable] no (cell, yaw, grasp) is both graspable on the source tray and placeable on the destination tray (tags: no_support, support, unreachable, no_stance, ('right', -0.54), ('left', 0.54))
  - bottom_cabinet_no_top_gjeoer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_1 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_2 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_3 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_4 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_rkgjer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_xqmipa_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_xqmipa_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral +0.00, footprint u -11.53..-11.10 vs support  (tags: no_support, support, no_stance)
  - coffee_table_qtwqqb_0 [coffee_table]: setup_failed - no_support: [no_support] coffee_table_qtwqqb_0: [support] untucked: [support] edge +u: [no_stance] no traversable stance around coffee_table_qtwqqb_0 (tags: no_support, support, no_stance)
  - countertop_fjkase_0 [countertop]: setup_failed - no_support: [no_support] countertop_fjkase_0: [support] untucked: [support] edge +u: [no_stance] no collision-free stance around countertop_fjkase_0 (tags: no_support, support, no_stance, unreachable)
  - countertop_ikwqer_0 [countertop]: setup_failed - no_support: [no_support] countertop_ikwqer_0: [support] untucked: [support] edge +u lateral +0.00: [unreachable] no (cell, yaw, grasp) is both graspable on the source tray and placeable on the destination tray (tags: no_support, support, unreachable, ('left', 2.24), ('left', 2.34), no_stance)
  - countertop_kelker_0 [countertop]: setup_failed - no_support: [no_support] countertop_kelker_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral -0.05, footprint u 0.50..0.94 vs support (3.73717 (tags: no_support, support, no_stance)
  - countertop_kelzer_0 [countertop]: setup_failed - no_support: [no_support] countertop_kelzer_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral -0.00, footprint u 5.95..6.39 vs support (-1.8297 (tags: no_support, support, no_stance)
  - countertop_sjxber_0 [countertop]: setup_failed - no_support: [no_support] countertop_sjxber_0: [support] untucked: [support] edge +u lateral +0.00: [support] no lateral room for the destination tray on either side (support bounds (-2.2146389484405518, -1.792198657989502, -24.18548 (tags: no_support, support, no_stance, ('left', 0.48))
  - desk_bhkhxo_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_nbpuns_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_xhjsub_0 [desk]: setup_failed - no_support: [no_support] desk_xhjsub_0: [support] untucked: [support] edge +u lateral +0.00: [support] no view / side keeps the source tray in view with the destination tray barely visible ({'views': 40, 'collision_free': 33, 'src_i (tags: no_support, support, ('left', 0.6), no_stance)
  - shelf_amyper_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_6 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_7 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### house_double_floor_lower
- (no viable support)

  rejected:
  - bar_byvbuc_0 [bar]: setup_failed - no_support: [no_support] bar_byvbuc_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral +0.49, footprint u 7.03..7.46 vs support (-0.97410464286 (tags: no_support, support, no_stance)
  - bar_egwapq_0 [bar]: setup_failed - no_support: [no_support] bar_egwapq_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral -0.05, footprint u -0.04..0.39 vs support (-8.4064989089 (tags: no_support, support, no_stance)
  - bottom_cabinet_qacthv_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_qacthv_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral +0.00, footprint u -2.70..-2.26 vs support (- (tags: no_support, support, no_stance)
  - bottom_cabinet_rhdbzv_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_rhdbzv_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral +0.00, footprint u 4.27..4.71 vs support (-6. (tags: no_support, support, no_stance)
  - breakfast_table_xftrki_0 [breakfast_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - coffee_table_koagbh_0 [coffee_table]: setup_failed - no_support: [no_support] coffee_table_koagbh_0: [support] untucked: [support] edge +u lateral +0.00: [support] no view / side keeps the source tray in view with the destination tray barely visible ({'views': 32, 'collision_free': 21 (tags: no_support, support, ('right', -0.52), ('left', 0.64), no_stance)

### house_double_floor_upper
- (no viable support)

  rejected:
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral +0.05, footprint u 1.85..2.29 vs support (-2. (tags: no_support, support, no_stance)
  - bottom_cabinet_nddvba_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_nddvba_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral -0.04, footprint u 4.28..4.71 vs support (-4. (tags: no_support, support, no_stance)
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - desk_aduafr_0 [desk]: setup_failed - no_support: [no_support] desk_aduafr_0: [support] untucked: [support] edge +u lateral +0.00: [unreachable] no (cell, yaw, grasp) is both graspable on the source tray and placeable on the destination tray (tags: no_support, support, unreachable, no_stance)
  - desk_iotfzl_0 [desk]: setup_failed - no_support: [no_support] desk_iotfzl_0: [support] untucked: [support] edge +u: [no_stance] no traversable stance around desk_iotfzl_0 (tags: no_support, support, no_stance)
  - desk_uqcmzf_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - nightstand_roxnbe_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_qsnawv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_qsnawv_0: [support] untucked: [support] edge +v: [no_stance] no collision-free stance around ottoman_qsnawv_0 (tags: no_support, support, no_stance)
  - serving_cart_vvtcby_0 [serving_cart]: setup_failed - no_support: [no_support] serving_cart_vvtcby_0: [support] untucked: [support] edge +u: [no_stance] no collision-free stance around serving_cart_vvtcby_0 (tags: no_support, support, no_stance)

### Rs_int
- (no viable support)

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bottom_cabinet_bamfsz_1 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_bamfsz_1: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral -0.00, footprint u 0.22..0.65 vs support (-1. (tags: no_support, support, no_stance)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral -0.05, footprint u -1.77..-1.33 vs support (- (tags: no_support, support, no_stance)
  - bottom_cabinet_slgzfc_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_slgzfc_0: [support] untucked: [support] edge +u lateral +0.00: [support] no lateral room for the destination tray on either side (support bounds (-1.8401092290878296, -1.3601082563400269, -2.7 (tags: no_support, support, ('right', -0.58), ('left', 0.58), no_stance)
  - breakfast_table_skczfi_0 [breakfast_table]: setup_failed - no_support: [no_support] breakfast_table_skczfi_0: [support] untucked: [support] edge +u lateral +0.00: [support] no lateral room for the destination tray on either side (support bounds (-1.8613483905792236, -1.0913903713226318, -0. (tags: no_support, support, no_stance)
  - breakfast_table_skczfi_1 [breakfast_table]: setup_failed - no_support: [no_support] breakfast_table_skczfi_1: [support] untucked: [support] edge +u lateral +0.00: [support] no view / side keeps the source tray in view with the destination tray barely visible ({'views': 40, 'collision_free': (tags: no_support, support, ('right', -0.72), no_stance, ('right', -0.82))
  - countertop_tpuwys_0 [countertop]: setup_failed - no_support: [no_support] countertop_tpuwys_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral -0.05, footprint u 0.54..0.98 vs support (-1.6803 (tags: no_support, support, no_stance)

### hotel_suite_large
- (no viable support)

  rejected:
  - desk_ohjotf_0 [desk]: setup_failed - no_support: [no_support] desk_ohjotf_0: [support] untucked: [support] edge +u lateral +0.00: [support] no view / side keeps the source tray in view with the destination tray barely visible ({'views': 40, 'collision_free': 17, 'src_i (tags: no_support, support, ('right', -0.52), ('left', 0.64), unreachable, no_stance)
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: error - error: RuntimeError('stack expects a non-empty TensorList')
  - ottoman_hnmndv_1 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_1: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral -0.05, footprint u 0.03..0.47 vs support (-0.0371266 (tags: no_support, support, no_stance)
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_rglkkf_0 [shelf]: setup_failed - no_support: [no_support] shelf_rglkkf_0: [support] untucked: [support] edge +u lateral +0.00: [support] no view / side keeps the source tray in view with the destination tray barely visible ({'views': 40, 'collision_free': 17, 'src_ (tags: no_support, support, ('right', -0.62), ('left', 0.62), unreachable, no_stance)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### office_cubicles_right
- **desk_mdhelw_0** [desk/mdhelw, rooms private_office_3]: 1 instance(s) in 7 attempt(s), rejections {'distractor_placement': 6}; top z 0.79 m, 6439 free cells; stance edge +u base (16.45, -13.63, yaw 1.57); trunk translate 0.00 tilt -0.90; n_reachable_cells 5; n_grasp_cells 1

  rejected:
  - bottom_cabinet_immwzb_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_1 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_2 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_3 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_4 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_5 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_6 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_7 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_8 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_9 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - coffee_table_mtchfd_0 [coffee_table]: setup_failed - no_support: [no_support] coffee_table_mtchfd_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral +0.00, footprint u -15.32..-14.88 vs support (4 (tags: no_support, support, no_stance)
  - conference_table_fpkmyw_0 [conference_table]: setup_failed - no_support: [no_support] conference_table_fpkmyw_0: [support] untucked: [support] edge +u lateral +0.00: [unreachable] no (cell, yaw, grasp) is both graspable on the source tray and placeable on the destination tray (tags: no_support, support, unreachable, ('right', -0.52), ('left', 0.52))
  - countertop_zkhelh_0 [countertop]: setup_failed - no_support: [no_support] countertop_zkhelh_0: [support] untucked: [support] edge +u lateral +0.00: [support] source tray not fully supported at the canonical position (stance lateral -0.05, footprint u -16.67..-16.23 vs support (-9. (tags: no_support, support, no_stance)
  - cubicle_deceti_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_4 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_5 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_mdhelw_1 [desk]: setup_failed - no_support: [no_support] desk_mdhelw_1: [support] untucked: [support] edge +u lateral +0.00: [unreachable] no (cell, yaw, grasp) is both graspable on the source tray and placeable on the destination tray (tags: no_support, support, unreachable, no_stance)
  - desk_mdhelw_2 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_mdhelw_3 [desk]: setup_failed - no_support: [no_support] desk_mdhelw_3: [support] untucked: [support] edge +u lateral +0.00: [unreachable] no (cell, yaw, grasp) is both graspable on the source tray and placeable on the destination tray (tags: no_support, support, unreachable, no_stance)
  - desk_pfsjbc_0 [desk]: setup_failed - no_support: [no_support] desk_pfsjbc_0: [support] untucked: [support] edge +u lateral +0.00: [unreachable] no (cell, yaw, grasp) is both graspable on the source tray and placeable on the destination tray (tags: no_support, support, unreachable, no_stance, ('right', -0.62))

### restaurant_diner
- (no viable support)

  rejected:
  - bar_xxftww_0 [bar]: error - error: RuntimeError('stack expects a non-empty TensorList')
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_2 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_2: [support] untucked: [support] edge +u: [no_stance] no traversable stance around bench_xwphjd_2 (tags: no_support, support, no_stance)
  - bench_xwphjd_3 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_3: [support] untucked: [support] edge +u: [no_stance] no traversable stance around bench_xwphjd_3 (tags: no_support, support, no_stance)
  - bench_xwphjd_4 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_4: [support] untucked: [support] edge +u: [no_stance] no traversable stance around bench_xwphjd_4 (tags: no_support, support, no_stance)
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - booth_xzrpar_0 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_0: [support] untucked: [support] edge +u: [no_stance] no traversable stance around booth_xzrpar_0 (tags: no_support, support, no_stance)
  - booth_xzrpar_1 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_1: [support] untucked: [support] edge +u: [no_stance] no traversable stance around booth_xzrpar_1 (tags: no_support, support, no_stance)
  - booth_xzrpar_2 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_2: [support] untucked: [support] edge +u: [no_stance] no traversable stance around booth_xzrpar_2 (tags: no_support, support, no_stance)
  - booth_xzrpar_3 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_3: [support] untucked: [support] edge +u: [no_stance] no traversable stance around booth_xzrpar_3 (tags: no_support, support, no_stance)
  - bottom_cabinet_olgoza_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_olgoza_0: [support] untucked: [support] edge +u lateral +0.00: [support] destination tray visibility out of range (dest 9043 px, source 0 px) (tags: no_support, support, no_stance)
  - commercial_kitchen_table_nisnrl_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_vxvtec_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_0 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_1 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_2 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_3 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - pedestal_table_nlfgwe_0 [pedestal_table]: setup_failed - no_support: [no_support] pedestal_table_nlfgwe_0: [support] untucked: [support] edge +v lateral +0.00: [support] no lateral room for the destination tray on either side (support bounds (-3.890573740005493, -2.830573558807373, 1.5905 (tags: no_support, support, no_stance)
  - pedestal_table_nlfgwe_1 [pedestal_table]: setup_failed - no_support: [no_support] pedestal_table_nlfgwe_1: [support] untucked: [support] edge +u: [no_stance] no traversable stance around pedestal_table_nlfgwe_1 (tags: no_support, support, no_stance)
  - pedestal_table_nlfgwe_2 [pedestal_table]: setup_failed - no_support: [no_support] pedestal_table_nlfgwe_2: [support] untucked: [support] edge +u: [no_stance] no traversable stance around pedestal_table_nlfgwe_2 (tags: no_support, support, no_stance)

## configuration_matching-0

### house_single_floor
- **bar_udatjt_0** [bar/udatjt, rooms kitchen_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 1.05 m, 8605 free cells; stance edge +v base (9.23, 0.27, yaw 3.14); trunk translate 0.75 tilt 0.10; n_reachable_cells 16
- **breakfast_table_rhjoby_0** [breakfast_table/rhjoby, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'grasp_unreachable': 1}; top z 0.81 m, 8463 free cells; stance edge -v base (9.19, 3.54, yaw 3.14); trunk translate 0.75 tilt -0.40; n_reachable_cells 15
- **countertop_fjkase_0** [countertop/fjkase, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.92 m, 4026 free cells; occupants ['wardrobe_pyuwgc_0']; stance edge -u base (22.70, 0.62, yaw 3.14); trunk translate 0.75 tilt -0.20; n_reachable_cells 15
- **countertop_ikwqer_0** [countertop/ikwqer, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.93 m, 6620 free cells; occupants ['fixed_window_qvqecb_0', 'tabletop_sink_vekaaf_0']; stance edge +u base (23.74, -0.11, yaw -0.00); trunk translate 0.75 tilt -0.20; n_reachable_cells 17
- **countertop_sjxber_0** [countertop/sjxber, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.93 m, 1078 free cells; stance edge +u base (23.66, -1.40, yaw -1.57); trunk translate 0.75 tilt -0.20; n_reachable_cells 16

  rejected:
  - bottom_cabinet_no_top_gjeoer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_1 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_2 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_3 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_4 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_rkgjer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_xqmipa_0 [bottom_cabinet]: no_instances - template ok, no instance in 15 attempts: layout x15
  - coffee_table_qtwqqb_0 [coffee_table]: setup_failed - no_support: [no_support] coffee_table_qtwqqb_0: [unreachable] only 0 reachable cells (tags: no_support, unreachable)
  - countertop_kelker_0 [countertop]: setup_failed - no_support: [no_support] countertop_kelker_0: [support] placement region empty (tags: no_support, support)
  - countertop_kelzer_0 [countertop]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - desk_bhkhxo_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_nbpuns_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_xhjsub_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_amyper_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_6 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_7 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### house_double_floor_lower
- (no viable support)

  rejected:
  - bar_byvbuc_0 [bar]: setup_failed - no_support: [no_support] bar_byvbuc_0: [unreachable] only 0 reachable cells (tags: no_support, unreachable)
  - bar_egwapq_0 [bar]: setup_failed - no_support: [no_support] bar_egwapq_0: [support] placement region empty (tags: no_support, support)
  - bottom_cabinet_qacthv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bottom_cabinet_rhdbzv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - breakfast_table_xftrki_0 [breakfast_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - coffee_table_koagbh_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### house_double_floor_upper
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.88 m, 1113 free cells; occupants ['picture_xnuynn_0']; stance edge +u base (1.39, 2.07, yaw 0.00); trunk translate 0.75 tilt -0.30; n_reachable_cells 15
- **bottom_cabinet_nddvba_0** [bottom_cabinet/nddvba, rooms bedroom_2]: 1 instance(s) in 2 attempt(s), rejections {'layout': 1}; top z 0.72 m, 779 free cells; stance edge +u base (4.49, -3.86, yaw -1.57); trunk translate 0.75 tilt -0.55; n_reachable_cells 11
- **desk_aduafr_0** [desk/aduafr, rooms bedroom_0]: 1 instance(s) in 8 attempt(s), rejections {'layout': 7}; top z 0.77 m, 2100 free cells; stance edge +u base (-5.43, 0.79, yaw -1.58); trunk translate 0.25 tilt -0.80; n_reachable_cells 8

  rejected:
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - desk_iotfzl_0 [desk]: setup_failed - no_support: [no_support] desk_iotfzl_0: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (47 candidates) (tags: no_support, stance_collision)
  - desk_uqcmzf_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - nightstand_roxnbe_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_qsnawv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - serving_cart_vvtcby_0 [serving_cart]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### Rs_int
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 15
- **breakfast_table_skczfi_0** [breakfast_table/skczfi, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.76 m, 1885 free cells; stance edge +u base (0.64, 0.36, yaw 0.01); trunk translate 0.75 tilt -0.50; n_reachable_cells 11

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bottom_cabinet_bamfsz_1 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_bamfsz_1: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (48 candidates) (tags: no_support, stance_collision)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (48 candidates) (tags: no_support, stance_collision)
  - breakfast_table_skczfi_1 [breakfast_table]: setup_failed - no_support: [no_support] breakfast_table_skczfi_1: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (48 candidates) (tags: no_support, stance_collision)
  - countertop_tpuwys_0 [countertop]: no_instances - template ok, no instance in 15 attempts: init_false x15

### hotel_suite_large
- (no viable support)

  rejected:
  - desk_ohjotf_0 [desk]: no_instances - template ok, no instance in 15 attempts: layout x15
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - ottoman_hnmndv_1 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_1: [unreachable] only 0 reachable cells (tags: no_support, unreachable)
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_rglkkf_0 [shelf]: no_instances - template ok, no instance in 15 attempts: layout x15
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### office_cubicles_right
- **desk_mdhelw_0** [desk/mdhelw, rooms private_office_3]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -12.73, yaw -3.14); trunk translate 0.75 tilt -0.55; n_reachable_cells 13
- **desk_mdhelw_1** [desk/mdhelw, rooms private_office_2]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.79 m, 6439 free cells; stance edge -v base (18.02, -8.39, yaw -3.14); trunk translate 0.75 tilt -0.55; n_reachable_cells 13
- **desk_mdhelw_3** [desk/mdhelw, rooms private_office_1]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -5.06, yaw -3.14); trunk translate 0.75 tilt -0.55; n_reachable_cells 13

  rejected:
  - bottom_cabinet_immwzb_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_1 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_2 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_3 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_4 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_5 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_6 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_7 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_8 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_9 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - coffee_table_mtchfd_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - conference_table_fpkmyw_0 [conference_table]: no_instances - template ok, no instance in 15 attempts: layout x15
  - countertop_zkhelh_0 [countertop]: setup_failed - no_support: [no_support] countertop_zkhelh_0: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (48 candidates) (tags: no_support, stance_collision)
  - cubicle_deceti_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_4 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_5 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_mdhelw_2 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_pfsjbc_0 [desk]: no_instances - template ok, no instance in 15 attempts: layout x15

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.75 tilt -0.65; n_reachable_cells 13
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 4 attempt(s), rejections {'layout': 3}; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.75 tilt -0.65; n_reachable_cells 13
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 4 attempt(s), rejections {'layout': 3}; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 11
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 9
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.50 tilt -0.45; n_reachable_cells 7
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge +v base (-3.36, 3.10, yaw -1.57); trunk translate 0.75 tilt -0.50; n_reachable_cells 11
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.75 tilt -0.65; n_reachable_cells 15
- **pedestal_table_nlfgwe_2** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 3 attempt(s), rejections {'layout': 2}; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -6.73, yaw -0.00); trunk translate 0.75 tilt -0.65; n_reachable_cells 15

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [support] placement region empty (tags: no_support, support)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_2 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_3 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_nisnrl_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_vxvtec_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_0 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_1 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_2 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_3 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

## configuration_matching-1

### house_single_floor
- **breakfast_table_rhjoby_0** [breakfast_table/rhjoby, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.81 m, 8463 free cells; stance edge -u base (7.29, 2.37, yaw 1.57); trunk translate 0.75 tilt -0.05; drive forward [0.2]; n_reachable_cells 3

  rejected:
  - bar_udatjt_0 [bar]: setup_failed - no_support: [no_support] bar_udatjt_0: [unreachable] handle not reachable from any of 6 cells (e.g. handle at [8.465109825134277, 0.28713566064834595, 1.4059076309204102], base (9.329936981201172, 0.2678448259830475, 3.1390495181347 (tags: no_support, unreachable, 8.465109825134277, 0.28713566064834595, 1.4059076309204102, 0.0, 0.25, 0.5)
  - bottom_cabinet_no_top_gjeoer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_1 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_2 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_3 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_4 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_rkgjer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_xqmipa_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_xqmipa_0: [support] stand_off 0.45: 0 visible of 30 cells, none supported+visible (view translate 0.75) (tags: no_support, support)
  - coffee_table_qtwqqb_0 [coffee_table]: setup_failed - no_support: [no_support] coffee_table_qtwqqb_0: [unreachable] handle not reachable from any of 3 cells (e.g. handle at [1.2160460948944092, 4.780270099639893, 0.8430641293525696], base (1.2577643394470215, 5.642963409423828, -1.5682 (tags: no_support, unreachable, 1.2160460948944092, 4.780270099639893, 0.8430641293525696, 0.0, 0.25, 0.5)
  - countertop_fjkase_0 [countertop]: setup_failed - no_support: [no_support] countertop_fjkase_0: [unreachable] handle not reachable from any of 2 cells (e.g. handle at [21.936439514160156, 0.5415071845054626, 1.271114706993103], base (22.80303382873535, 0.5181719064712524, 3.1390488 (tags: no_support, unreachable, 21.936439514160156, 0.5415071845054626, 1.271114706993103, 0.0, 0.25, 0.5)
  - countertop_ikwqer_0 [countertop]: setup_failed - no_support: [no_support] countertop_ikwqer_0: [unreachable] handle not reachable from any of 1 cells (e.g. handle at [24.4649600982666, -0.13863307237625122, 1.2829140424728394], base (23.637117385864258, -0.10525951534509659, -0.00 (tags: no_support, unreachable, 24.4649600982666, -0.13863307237625122, 1.2829140424728394, 0.0, 0.25, 0.5)
  - countertop_kelker_0 [countertop]: setup_failed - no_support: [no_support] countertop_kelker_0: [support] stand_off 0.45: placement region empty (tags: no_support, support)
  - countertop_kelzer_0 [countertop]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - countertop_sjxber_0 [countertop]: setup_failed - no_support: [no_support] countertop_sjxber_0: [unreachable] handle not reachable from any of 1 cells (e.g. handle at [23.61809730529785, -2.107102155685425, 1.2829136848449707], base (23.655609130859375, -1.3023715019226074, -1.5733 (tags: no_support, unreachable, 23.61809730529785, -2.107102155685425, 1.2829136848449707, 0.0, 0.25, 0.5)
  - desk_bhkhxo_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_nbpuns_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_xhjsub_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_amyper_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_6 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_7 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### house_double_floor_lower
- (no viable support)

  rejected:
  - bar_byvbuc_0 [bar]: setup_failed - no_support: [no_support] bar_byvbuc_0: [support] stand_off 0.45: placement region empty (tags: no_support, support)
  - bar_egwapq_0 [bar]: setup_failed - no_support: [no_support] bar_egwapq_0: [support] stand_off 0.45: placement region empty (tags: no_support, support)
  - bottom_cabinet_qacthv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bottom_cabinet_rhdbzv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - breakfast_table_xftrki_0 [breakfast_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - coffee_table_koagbh_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### house_double_floor_upper
- (no viable support)

  rejected:
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [support] stand_off 0.45: 0 visible of 30 cells, none supported+visible (view translate 0.0) (tags: no_support, support)
  - bottom_cabinet_nddvba_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_nddvba_0: [support] stand_off 0.45: 0 visible of 27 cells, none supported+visible (view translate 0.75) (tags: no_support, support)
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - desk_aduafr_0 [desk]: setup_failed - no_support: [no_support] desk_aduafr_0: [unreachable] handle not reachable from any of 8 cells (e.g. handle at [-5.422635078430176, -0.05020727962255478, 1.1287872791290283], base (-5.236870765686035, 0.8917053937911987, -1.57535293 (tags: no_support, unreachable, -5.422635078430176, -0.05020727962255478, 1.1287872791290283, 0.0, 0.25, 0.5)
  - desk_iotfzl_0 [desk]: setup_failed - no_support: [no_support] desk_iotfzl_0: [support] stand_off 0.45: 0 visible of 120 cells, none supported+visible (view translate 0.75) (tags: no_support, support)
  - desk_uqcmzf_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - nightstand_roxnbe_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_qsnawv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - serving_cart_vvtcby_0 [serving_cart]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### Rs_int
- (no viable support)

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bottom_cabinet_bamfsz_1 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_bamfsz_1: [support] stand_off 0.45: 0 visible of 3 cells, none supported+visible (view translate 0.75) (tags: no_support, support)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [support] [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (48 candidates) (tags: no_support, support, stance_collision)
  - bottom_cabinet_slgzfc_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_slgzfc_0: [unreachable] handle not reachable from any of 0 cells (e.g. handle at None, base (0.8101062774658203, 2.2857182025909424, 8.662515256217028e-07), translates [0.0, 0.25, 0.5]) (tags: no_support, unreachable, 0.0, 0.25, 0.5)
  - breakfast_table_skczfi_0 [breakfast_table]: setup_failed - no_support: [no_support] breakfast_table_skczfi_0: [unreachable] handle not reachable from any of 0 cells (e.g. handle at None, base (0.5367545485496521, 0.3553491234779358, 0.013137252966373756), translates [0.0, 0.25, 0.5]) (tags: no_support, unreachable, 0.0, 0.25, 0.5)
  - breakfast_table_skczfi_1 [breakfast_table]: setup_failed - no_support: [no_support] breakfast_table_skczfi_1: [unreachable] handle not reachable from any of 3 cells (e.g. handle at [0.36238789558410645, -3.8400814533233643, 1.1141337156295776], base (0.4297318458557129, -2.976639986038208,  (tags: no_support, unreachable, 0.36238789558410645, -3.8400814533233643, 1.1141337156295776, 0.0, 0.25, 0.5)
  - countertop_tpuwys_0 [countertop]: setup_failed - no_support: [no_support] countertop_tpuwys_0: [support] stand_off 0.45: placement region empty (tags: no_support, support)

### hotel_suite_large
- (no viable support)

  rejected:
  - desk_ohjotf_0 [desk]: setup_failed - no_support: [no_support] desk_ohjotf_0: [support] stand_off 0.45: 0 visible of 40 cells, none supported+visible (view translate 0.75) (tags: no_support, support)
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - ottoman_hnmndv_1 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_1: [support] stand_off 0.45: placement region empty (tags: no_support, support)
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_rglkkf_0 [shelf]: setup_failed - no_support: [no_support] shelf_rglkkf_0: [support] stand_off 0.45: 0 visible of 105 cells, none supported+visible (view translate 0.75) (tags: no_support, support)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### office_cubicles_right
- **desk_pfsjbc_0** [desk/pfsjbc, rooms lobby_0]: 1 instance(s) in 6 attempt(s), rejections {'out_of_view': 5}; top z 0.77 m, 8846 free cells; occupants ['walls_wcngqp_0']; stance edge +v base (1.89, -15.50, yaw 3.14); trunk translate 0.00 tilt -0.55; drive forward [0.2]; n_reachable_cells 1

  rejected:
  - bottom_cabinet_immwzb_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_1 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_2 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_3 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_4 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_5 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_6 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_7 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_8 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_9 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - coffee_table_mtchfd_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - conference_table_fpkmyw_0 [conference_table]: setup_failed - no_support: [no_support] conference_table_fpkmyw_0: [unreachable] handle not reachable from any of 4 cells (e.g. handle at [6.15805721282959, 14.832605361938477, 1.0290371179580688], base (6.229757308959961, 15.696004867553711, -1.5 (tags: no_support, unreachable, 6.15805721282959, 14.832605361938477, 1.0290371179580688, 0.0, 0.25, 0.5)
  - countertop_zkhelh_0 [countertop]: setup_failed - no_support: [no_support] countertop_zkhelh_0: [support] stand_off 0.45: placement region empty (tags: no_support, support)
  - cubicle_deceti_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_4 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_5 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_mdhelw_0 [desk]: setup_failed - no_support: [no_support] desk_mdhelw_0: [unreachable] handle not reachable from any of 3 cells (e.g. handle at [17.175443649291992, -12.718043327331543, 1.1406528949737549], base (18.03884506225586, -12.72973346710205, -3.1415870944 (tags: no_support, unreachable, 17.175443649291992, -12.718043327331543, 1.1406528949737549, 0.0, 0.25, 0.5)
  - desk_mdhelw_1 [desk]: setup_failed - no_support: [no_support] desk_mdhelw_1: [unreachable] handle not reachable from any of 3 cells (e.g. handle at [17.253700256347656, -8.376163482666016, 1.1406530141830444], base (18.117103576660156, -8.387853622436523, -3.1415868878 (tags: no_support, unreachable, 17.253700256347656, -8.376163482666016, 1.1406530141830444, 0.0, 0.25, 0.5)
  - desk_mdhelw_2 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_mdhelw_3 [desk]: setup_failed - no_support: [no_support] desk_mdhelw_3: [unreachable] handle not reachable from any of 3 cells (e.g. handle at [17.175443649291992, -5.051483154296875, 1.1406526565551758], base (18.038846969604492, -5.063172340393066, -3.1415870944 (tags: no_support, unreachable, 17.175443649291992, -5.051483154296875, 1.1406526565551758, 0.0, 0.25, 0.5)

### restaurant_diner
- (no viable support)

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [support] stand_off 0.45: placement region empty (tags: no_support, support)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_2 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_3 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - booth_xzrpar_0 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_0: [unreachable] handle not reachable from any of 2 cells (e.g. handle at [0.2267158329486847, 2.6330230236053467, 1.1091846227645874], base (-0.5766842365264893, 2.6547231674194336, -1.62920684 (tags: no_support, unreachable, 0.2267158329486847, 2.6330230236053467, 1.1091846227645874, 0.0, 0.25, 0.5)
  - booth_xzrpar_1 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_1: [unreachable] handle not reachable from any of 2 cells (e.g. handle at [0.22671595215797424, 0.382412314414978, 1.1091842651367188], base (-0.5766841173171997, 0.40411248803138733, -1.6292068 (tags: no_support, unreachable, 0.22671595215797424, 0.382412314414978, 1.1091842651367188, 0.0, 0.25, 0.5)
  - booth_xzrpar_2 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_2: [unreachable] handle not reachable from any of 2 cells (e.g. handle at [0.09517262130975723, -8.772573471069336, 1.1091824769973755], base (0.11687394976615906, -7.909173011779785, -1.5707974 (tags: no_support, unreachable, 0.09517262130975723, -8.772573471069336, 1.1091824769973755, 0.0, 0.25, 0.5)
  - booth_xzrpar_3 [booth]: setup_failed - no_support: [no_support] booth_xzrpar_3: [unreachable] handle not reachable from any of 2 cells (e.g. handle at [-2.155437707901001, -8.772574424743652, 1.109182596206665], base (-2.1337361335754395, -7.909174919128418, -1.570797431 (tags: no_support, unreachable, -2.155437707901001, -8.772574424743652, 1.109182596206665, 0.0, 0.25, 0.5)
  - bottom_cabinet_olgoza_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_olgoza_0: [unreachable] handle not reachable from any of 1 cells (e.g. handle at [-4.61296272277832, -2.013155460357666, 1.1971007585525513], base (-3.808410882949829, -2.014829158782959, -3.1 (tags: no_support, unreachable, -4.61296272277832, -2.013155460357666, 1.1971007585525513, 0.0, 0.25, 0.5)
  - commercial_kitchen_table_nisnrl_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_vxvtec_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_0 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_1 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_2 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_3 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - pedestal_table_nlfgwe_0 [pedestal_table]: setup_failed - no_support: [no_support] pedestal_table_nlfgwe_0: [support] stand_off 0.45: 0 visible of 54 cells, none supported+visible (view translate 0.0) (tags: no_support, support)
  - pedestal_table_nlfgwe_1 [pedestal_table]: setup_failed - no_support: [no_support] pedestal_table_nlfgwe_1: [unreachable] handle not reachable from any of 0 cells (e.g. handle at None, base (-0.32638251781463623, -4.161527633666992, -0.0002636909580918), translates [0.0, 0.25, 0.5]) (tags: no_support, unreachable, 0.0, 0.25, 0.5)
  - pedestal_table_nlfgwe_2 [pedestal_table]: setup_failed - no_support: [no_support] pedestal_table_nlfgwe_2: [unreachable] handle not reachable from any of 0 cells (e.g. handle at None, base (-0.32638224959373474, -6.733242034912109, -0.00025415421499872746), translates [0.0, 0.25, 0.5]) (tags: no_support, unreachable, 0.0, 0.25, 0.5)

## configuration_matching-2

### house_single_floor
- **breakfast_table_rhjoby_0** [breakfast_table/rhjoby, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'target_unsupported': 1}; top z 0.81 m, 8463 free cells; stance edge -u base (7.29, 2.37, yaw 1.57); trunk translate 0.75 tilt -0.05; drive forward [0.2]; n_reachable_cells 3

  rejected:
  - bar_udatjt_0 [bar]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_no_top_gjeoer_0 [bottom_cabinet_no_top]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_no_top_gjrero_0 [bottom_cabinet_no_top]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_no_top_gjrero_1 [bottom_cabinet_no_top]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_no_top_gjrero_2 [bottom_cabinet_no_top]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_no_top_gjrero_3 [bottom_cabinet_no_top]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_no_top_gjrero_4 [bottom_cabinet_no_top]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_no_top_rkgjer_0 [bottom_cabinet_no_top]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_xqmipa_0 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - coffee_table_qtwqqb_0 [coffee_table]: skipped - configuration_matching-1 produced no instance on this support
  - countertop_fjkase_0 [countertop]: skipped - configuration_matching-1 produced no instance on this support
  - countertop_ikwqer_0 [countertop]: skipped - configuration_matching-1 produced no instance on this support
  - countertop_kelker_0 [countertop]: skipped - configuration_matching-1 produced no instance on this support
  - countertop_kelzer_0 [countertop]: skipped - configuration_matching-1 produced no instance on this support
  - countertop_sjxber_0 [countertop]: skipped - configuration_matching-1 produced no instance on this support
  - desk_bhkhxo_0 [desk]: skipped - configuration_matching-1 produced no instance on this support
  - desk_nbpuns_0 [desk]: skipped - configuration_matching-1 produced no instance on this support
  - desk_xhjsub_0 [desk]: skipped - configuration_matching-1 produced no instance on this support
  - shelf_amyper_1 [shelf]: skipped - configuration_matching-1 produced no instance on this support
  - shelf_amyper_2 [shelf]: skipped - configuration_matching-1 produced no instance on this support
  - shelf_amyper_6 [shelf]: skipped - configuration_matching-1 produced no instance on this support
  - shelf_amyper_7 [shelf]: skipped - configuration_matching-1 produced no instance on this support

### house_double_floor_lower
- (no viable support)

  rejected:
  - bar_byvbuc_0 [bar]: skipped - configuration_matching-1 produced no instance on this support
  - bar_egwapq_0 [bar]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_qacthv_0 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_rhdbzv_0 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - breakfast_table_xftrki_0 [breakfast_table]: skipped - configuration_matching-1 produced no instance on this support
  - coffee_table_koagbh_0 [coffee_table]: skipped - configuration_matching-1 produced no instance on this support

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

### office_cubicles_right
- **desk_pfsjbc_0** [desk/pfsjbc, rooms lobby_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 8846 free cells; occupants ['walls_wcngqp_0']; stance edge +v base (1.89, -15.50, yaw 3.14); trunk translate 0.00 tilt -0.55; drive forward [0.2]; n_reachable_cells 1

  rejected:
  - bottom_cabinet_immwzb_0 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_immwzb_1 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_immwzb_2 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_immwzb_3 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_immwzb_4 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_immwzb_5 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_immwzb_6 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_immwzb_7 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_immwzb_8 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_immwzb_9 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - coffee_table_mtchfd_0 [coffee_table]: skipped - configuration_matching-1 produced no instance on this support
  - conference_table_fpkmyw_0 [conference_table]: skipped - configuration_matching-1 produced no instance on this support
  - countertop_zkhelh_0 [countertop]: skipped - configuration_matching-1 produced no instance on this support
  - cubicle_deceti_0 [cubicle]: skipped - configuration_matching-1 produced no instance on this support
  - cubicle_deceti_1 [cubicle]: skipped - configuration_matching-1 produced no instance on this support
  - cubicle_deceti_2 [cubicle]: skipped - configuration_matching-1 produced no instance on this support
  - cubicle_deceti_3 [cubicle]: skipped - configuration_matching-1 produced no instance on this support
  - cubicle_hzalfx_0 [cubicle]: skipped - configuration_matching-1 produced no instance on this support
  - cubicle_hzalfx_1 [cubicle]: skipped - configuration_matching-1 produced no instance on this support
  - cubicle_hzalfx_2 [cubicle]: skipped - configuration_matching-1 produced no instance on this support
  - cubicle_hzalfx_3 [cubicle]: skipped - configuration_matching-1 produced no instance on this support
  - cubicle_hzalfx_4 [cubicle]: skipped - configuration_matching-1 produced no instance on this support
  - cubicle_hzalfx_5 [cubicle]: skipped - configuration_matching-1 produced no instance on this support
  - desk_mdhelw_0 [desk]: skipped - configuration_matching-1 produced no instance on this support
  - desk_mdhelw_1 [desk]: skipped - configuration_matching-1 produced no instance on this support
  - desk_mdhelw_2 [desk]: skipped - configuration_matching-1 produced no instance on this support
  - desk_mdhelw_3 [desk]: skipped - configuration_matching-1 produced no instance on this support

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
  - commercial_kitchen_table_nisnrl_0 [commercial_kitchen_table]: skipped - configuration_matching-1 produced no instance on this support
  - commercial_kitchen_table_vxvtec_0 [commercial_kitchen_table]: skipped - configuration_matching-1 produced no instance on this support
  - metal_bottom_cabinet_bbjuqu_0 [metal_bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - metal_bottom_cabinet_bbjuqu_1 [metal_bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - metal_bottom_cabinet_bbjuqu_2 [metal_bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - metal_bottom_cabinet_bbjuqu_3 [metal_bottom_cabinet]: skipped - configuration_matching-1 produced no instance on this support
  - pedestal_table_nlfgwe_0 [pedestal_table]: skipped - configuration_matching-1 produced no instance on this support
  - pedestal_table_nlfgwe_1 [pedestal_table]: skipped - configuration_matching-1 produced no instance on this support
  - pedestal_table_nlfgwe_2 [pedestal_table]: skipped - configuration_matching-1 produced no instance on this support

## configuration_matching-3

### house_single_floor
- **bar_udatjt_0** [bar/udatjt, rooms kitchen_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 1.05 m, 8605 free cells; stance edge +v base (9.23, 0.27, yaw 3.14); trunk translate 0.75 tilt 0.10; n_reachable_cells 12; n_grasp_cells 12
- **bottom_cabinet_xqmipa_0** [bottom_cabinet/xqmipa, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.74 m, 1254 free cells; stance edge +u base (21.66, 11.32, yaw -3.14); trunk translate 0.00 tilt -0.90; n_reachable_cells 8; n_grasp_cells 7
- **breakfast_table_rhjoby_0** [breakfast_table/rhjoby, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.81 m, 8463 free cells; stance edge -v base (9.19, 3.54, yaw 3.14); trunk translate 0.75 tilt -0.40; n_reachable_cells 11; n_grasp_cells 11
- **countertop_fjkase_0** [countertop/fjkase, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.92 m, 4026 free cells; occupants ['wardrobe_pyuwgc_0']; stance edge -u base (22.70, 0.62, yaw 3.14); trunk translate 0.75 tilt -0.20; n_reachable_cells 12; n_grasp_cells 11
- **countertop_ikwqer_0** [countertop/ikwqer, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.93 m, 6620 free cells; occupants ['fixed_window_qvqecb_0', 'tabletop_sink_vekaaf_0']; stance edge +u base (23.74, -0.11, yaw -0.00); trunk translate 0.75 tilt -0.20; n_reachable_cells 15; n_grasp_cells 15
- **countertop_sjxber_0** [countertop/sjxber, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.93 m, 1078 free cells; stance edge +u base (23.66, -1.40, yaw -1.57); trunk translate 0.75 tilt -0.20; n_reachable_cells 14; n_grasp_cells 12

  rejected:
  - bottom_cabinet_no_top_gjeoer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_1 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_2 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_3 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_4 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_rkgjer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - coffee_table_qtwqqb_0 [coffee_table]: setup_failed - no_support: [no_support] coffee_table_qtwqqb_0: [unreachable] no reachable cell for the male part (tags: no_support, unreachable)
  - countertop_kelker_0 [countertop]: setup_failed - no_support: [no_support] countertop_kelker_0: [support] placement region empty (tags: no_support, support)
  - countertop_kelzer_0 [countertop]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - desk_bhkhxo_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_nbpuns_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_xhjsub_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_amyper_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_6 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_7 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### house_double_floor_lower
- (no viable support)

  rejected:
  - bar_byvbuc_0 [bar]: setup_failed - no_support: [no_support] bar_byvbuc_0: [unreachable] no reachable cell for the male part (tags: no_support, unreachable)
  - bar_egwapq_0 [bar]: setup_failed - no_support: [no_support] bar_egwapq_0: [support] placement region empty (tags: no_support, support)
  - bottom_cabinet_qacthv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bottom_cabinet_rhdbzv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - breakfast_table_xftrki_0 [breakfast_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - coffee_table_koagbh_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### house_double_floor_upper
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.88 m, 1113 free cells; occupants ['picture_xnuynn_0']; stance edge +u base (1.39, 2.07, yaw 0.00); trunk translate 0.75 tilt -0.30; n_reachable_cells 12; n_grasp_cells 12
- **bottom_cabinet_nddvba_0** [bottom_cabinet/nddvba, rooms bedroom_2]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.72 m, 779 free cells; stance edge +u base (4.49, -3.86, yaw -1.57); trunk translate 0.75 tilt -0.55; n_reachable_cells 11; n_grasp_cells 11
- **desk_aduafr_0** [desk/aduafr, rooms bedroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 2100 free cells; stance edge +u base (-5.43, 0.79, yaw -1.58); trunk translate 0.50 tilt -0.70; n_reachable_cells 7; n_grasp_cells 7

  rejected:
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - desk_iotfzl_0 [desk]: setup_failed - no_support: [no_support] desk_iotfzl_0: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (44 candidates) (tags: no_support, stance_collision)
  - desk_uqcmzf_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - nightstand_roxnbe_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_qsnawv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - serving_cart_vvtcby_0 [serving_cart]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### Rs_int
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 13; n_grasp_cells 13
- **breakfast_table_skczfi_0** [breakfast_table/skczfi, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.76 m, 1885 free cells; stance edge +u base (0.64, 0.36, yaw 0.01); trunk translate 0.75 tilt -0.50; n_reachable_cells 10; n_grasp_cells 10

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bottom_cabinet_bamfsz_1 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_bamfsz_1: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (47 candidates) (tags: no_support, stance_collision)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (45 candidates) (tags: no_support, stance_collision)
  - breakfast_table_skczfi_1 [breakfast_table]: setup_failed - no_support: [no_support] breakfast_table_skczfi_1: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (46 candidates) (tags: no_support, stance_collision)
  - countertop_tpuwys_0 [countertop]: no_instances - template ok, no instance in 15 attempts: init_false x7, partner_placement x5, target_placement x3

### hotel_suite_large
- **shelf_rglkkf_0** [shelf/rglkkf, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.66 m, 2064 free cells; stance edge +u base (0.96, 3.27, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 7; n_grasp_cells 7

  rejected:
  - desk_ohjotf_0 [desk]: setup_failed - no_support: [no_support] desk_ohjotf_0: [support] no visible cells (tags: no_support, support)
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - ottoman_hnmndv_1 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_1: [unreachable] no reachable cell for the male part (tags: no_support, unreachable)
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### office_cubicles_right
- **conference_table_fpkmyw_0** [conference_table/fpkmyw, rooms meeting_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.67 m, 12996 free cells; stance edge +v base (6.23, 15.60, yaw -1.57); trunk translate 0.00 tilt -0.90; n_reachable_cells 6; n_grasp_cells 6
- **desk_mdhelw_0** [desk/mdhelw, rooms private_office_3]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -12.73, yaw -3.14); trunk translate 0.75 tilt -0.60; n_reachable_cells 9; n_grasp_cells 9
- **desk_mdhelw_1** [desk/mdhelw, rooms private_office_2]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.79 m, 6439 free cells; stance edge -v base (18.02, -8.39, yaw -3.14); trunk translate 0.75 tilt -0.60; n_reachable_cells 10; n_grasp_cells 10
- **desk_mdhelw_3** [desk/mdhelw, rooms private_office_1]: 1 instance(s) in 2 attempt(s), rejections {'target_placement': 1}; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -5.06, yaw -3.14); trunk translate 0.75 tilt -0.60; n_reachable_cells 11; n_grasp_cells 11
- **desk_pfsjbc_0** [desk/pfsjbc, rooms lobby_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 8846 free cells; occupants ['walls_wcngqp_0']; stance edge +v base (1.79, -15.50, yaw 3.14); trunk translate 0.00 tilt -0.90; n_reachable_cells 11; n_grasp_cells 11

  rejected:
  - bottom_cabinet_immwzb_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_1 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_2 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_3 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_4 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_5 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_6 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_7 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_8 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_9 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - coffee_table_mtchfd_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - countertop_zkhelh_0 [countertop]: setup_failed - no_support: [no_support] countertop_zkhelh_0: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (46 candidates) (tags: no_support, stance_collision)
  - cubicle_deceti_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_4 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_5 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_mdhelw_2 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 5 attempt(s), rejections {'partner_placement': 4}; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.75 tilt -0.70; n_reachable_cells 9; n_grasp_cells 9
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.75 tilt -0.70; n_reachable_cells 9; n_grasp_cells 9
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 5 attempt(s), rejections {'partner_placement': 4}; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.70; n_reachable_cells 9; n_grasp_cells 9
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 3 attempt(s), rejections {'partner_placement': 2}; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.70; n_reachable_cells 8; n_grasp_cells 6
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.75 tilt -0.20; n_reachable_cells 5; n_grasp_cells 5
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge +v base (-3.36, 3.10, yaw -1.57); trunk translate 0.75 tilt -0.50; n_reachable_cells 10; n_grasp_cells 10
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 4 attempt(s), rejections {'partner_placement': 3}; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.75 tilt -0.65; n_reachable_cells 11; n_grasp_cells 9
- **pedestal_table_nlfgwe_2** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -6.73, yaw -0.00); trunk translate 0.75 tilt -0.65; n_reachable_cells 11; n_grasp_cells 9

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [support] placement region empty (tags: no_support, support)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_2 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_3 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_nisnrl_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_vxvtec_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_0 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_1 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_2 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_3 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

## world_alignment-0

### house_single_floor
- **bar_udatjt_0** [bar/udatjt, rooms kitchen_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 1.05 m, 8605 free cells; stance edge +v base (9.23, 0.27, yaw 3.14); trunk translate 0.50 tilt -0.10
- **breakfast_table_rhjoby_0** [breakfast_table/rhjoby, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.81 m, 8463 free cells; stance edge -v base (9.19, 3.54, yaw 3.14); trunk translate 0.75 tilt -0.25

  rejected:
  - bottom_cabinet_no_top_gjeoer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_1 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_2 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_3 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_4 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_rkgjer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_xqmipa_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_xqmipa_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - coffee_table_qtwqqb_0 [coffee_table]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - countertop_fjkase_0 [countertop]: setup_failed - no_support: [no_support] countertop_fjkase_0: [support] front edge barely in view (2/9 samples) (tags: no_support, support)
  - countertop_ikwqer_0 [countertop]: setup_failed - no_support: [no_support] countertop_ikwqer_0: [support] front edge barely in view (1/9 samples) (tags: no_support, support)
  - countertop_kelker_0 [countertop]: setup_failed - no_support: [no_support] countertop_kelker_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - countertop_kelzer_0 [countertop]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - countertop_sjxber_0 [countertop]: setup_failed - no_support: [no_support] countertop_sjxber_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - desk_bhkhxo_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_nbpuns_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_xhjsub_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_amyper_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_6 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_7 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### house_double_floor_lower
- (no viable support)

  rejected:
  - bar_byvbuc_0 [bar]: setup_failed - no_support: [no_support] bar_byvbuc_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - bar_egwapq_0 [bar]: setup_failed - no_support: [no_support] bar_egwapq_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - bottom_cabinet_qacthv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bottom_cabinet_rhdbzv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - breakfast_table_xftrki_0 [breakfast_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - coffee_table_koagbh_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### house_double_floor_upper
- (no viable support)

  rejected:
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - bottom_cabinet_nddvba_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_nddvba_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - desk_aduafr_0 [desk]: setup_failed - no_support: [no_support] desk_aduafr_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - desk_iotfzl_0 [desk]: setup_failed - no_support: [no_support] desk_iotfzl_0: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (48 candidates) (tags: no_support, stance_collision)
  - desk_uqcmzf_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - nightstand_roxnbe_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_qsnawv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - serving_cart_vvtcby_0 [serving_cart]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### Rs_int
- (no viable support)

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bottom_cabinet_bamfsz_1 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_bamfsz_1: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (48 candidates) (tags: no_support, stance_collision)
  - bottom_cabinet_slgzfc_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_slgzfc_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - breakfast_table_skczfi_0 [breakfast_table]: no_instances - template ok, no instance in 15 attempts: init_false x15
  - breakfast_table_skczfi_1 [breakfast_table]: setup_failed - no_support: [no_support] breakfast_table_skczfi_1: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (48 candidates) (tags: no_support, stance_collision)
  - countertop_tpuwys_0 [countertop]: setup_failed - no_support: [no_support] countertop_tpuwys_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)

### hotel_suite_large
- **desk_ohjotf_0** [desk/ohjotf, rooms bedroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.68 m, 2403 free cells; stance edge +v base (-1.71, -4.63, yaw 0.00); trunk translate 0.75 tilt -0.45

  rejected:
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - ottoman_hnmndv_1 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_1: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_rglkkf_0 [shelf]: setup_failed - no_support: [no_support] shelf_rglkkf_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### office_cubicles_right
- **conference_table_fpkmyw_0** [conference_table/fpkmyw, rooms meeting_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.67 m, 12996 free cells; stance edge +v base (6.23, 15.60, yaw -1.57); trunk translate 0.50 tilt -0.70
- **desk_mdhelw_0** [desk/mdhelw, rooms private_office_3]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -12.73, yaw -3.14); trunk translate 0.50 tilt -0.55
- **desk_mdhelw_1** [desk/mdhelw, rooms private_office_2]: 1 instance(s) in 2 attempt(s), rejections {'grasp_unreachable': 1}; top z 0.79 m, 6439 free cells; stance edge -v base (18.02, -8.39, yaw -3.14); trunk translate 0.50 tilt -0.55
- **desk_mdhelw_3** [desk/mdhelw, rooms private_office_1]: 1 instance(s) in 2 attempt(s), rejections {'grasp_unreachable': 1}; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -5.06, yaw -3.14); trunk translate 0.50 tilt -0.55
- **desk_pfsjbc_0** [desk/pfsjbc, rooms lobby_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 8846 free cells; occupants ['walls_wcngqp_0']; stance edge +v base (1.79, -15.50, yaw 3.14); trunk translate 0.50 tilt -0.50

  rejected:
  - bottom_cabinet_immwzb_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_1 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_2 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_3 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_4 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_5 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_6 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_7 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_8 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_9 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - coffee_table_mtchfd_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - countertop_zkhelh_0 [countertop]: setup_failed - no_support: [no_support] countertop_zkhelh_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - cubicle_deceti_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_4 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_5 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_mdhelw_2 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.50 tilt -0.60
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.50 tilt -0.60
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.50 tilt -0.60
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.50 tilt -0.60
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'grasp_unreachable': 1}; top z 0.78 m, 2222 free cells; stance edge +v base (-3.36, 3.10, yaw -1.57); trunk translate 0.25 tilt -0.70
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.25 tilt -0.70
- **pedestal_table_nlfgwe_2** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -6.73, yaw -0.00); trunk translate 0.25 tilt -0.70

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_2 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_3 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_olgoza_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_olgoza_0: [support] placement region empty (object too large for the support depth) (tags: no_support, support)
  - commercial_kitchen_table_nisnrl_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_vxvtec_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_0 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_1 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_2 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_3 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

## inter_object_relation_satisfaction-0

### house_single_floor
- **bar_udatjt_0** [bar/udatjt, rooms kitchen_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 1.05 m, 8605 free cells; stance edge +v base (9.23, 0.27, yaw 3.14); trunk translate 0.75 tilt 0.10; n_reachable_cells 14; n_grasp_cells 12
- **bottom_cabinet_xqmipa_0** [bottom_cabinet/xqmipa, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.74 m, 1254 free cells; stance edge +u base (21.66, 11.32, yaw -3.14); trunk translate 0.00 tilt -0.90; n_reachable_cells 10; n_grasp_cells 8
- **breakfast_table_rhjoby_0** [breakfast_table/rhjoby, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.81 m, 8463 free cells; stance edge -v base (9.19, 3.54, yaw 3.14); trunk translate 0.75 tilt -0.40; n_reachable_cells 12; n_grasp_cells 10
- **coffee_table_qtwqqb_0** [coffee_table/qtwqqb, rooms living_room_1]: 1 instance(s) in 2 attempt(s), rejections {'target_placement': 1}; top z 0.49 m, 3547 free cells; stance edge -v base (1.26, 5.44, yaw -1.57); trunk translate 0.00 tilt -0.90; stand-off 0.35; n_reachable_cells 3; n_grasp_cells 3
- **countertop_fjkase_0** [countertop/fjkase, rooms utility_room_0]: 1 instance(s) in 3 attempt(s), rejections {'target_placement': 2}; top z 0.92 m, 4026 free cells; occupants ['wardrobe_pyuwgc_0']; stance edge -u base (22.70, 0.62, yaw 3.14); trunk translate 0.75 tilt -0.20; n_reachable_cells 13; n_grasp_cells 11
- **countertop_ikwqer_0** [countertop/ikwqer, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.93 m, 6620 free cells; occupants ['fixed_window_qvqecb_0', 'tabletop_sink_vekaaf_0']; stance edge +u base (23.74, -0.11, yaw -0.00); trunk translate 0.75 tilt -0.15; n_reachable_cells 16; n_grasp_cells 15
- **countertop_sjxber_0** [countertop/sjxber, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.93 m, 1078 free cells; stance edge +u base (23.66, -1.40, yaw -1.57); trunk translate 0.75 tilt -0.15; n_reachable_cells 16; n_grasp_cells 13

  rejected:
  - bottom_cabinet_no_top_gjeoer_0 [bottom_cabinet_no_top]: setup_failed - no_support: [no_support] bottom_cabinet_no_top_gjeoer_0: [support] bottom_cabinet_no_top_gjeoer_0 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_no_top_gjrero_0 [bottom_cabinet_no_top]: setup_failed - no_support: [no_support] bottom_cabinet_no_top_gjrero_0: [support] bottom_cabinet_no_top_gjrero_0 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_no_top_gjrero_1 [bottom_cabinet_no_top]: setup_failed - no_support: [no_support] bottom_cabinet_no_top_gjrero_1: [support] bottom_cabinet_no_top_gjrero_1 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_no_top_gjrero_2 [bottom_cabinet_no_top]: setup_failed - no_support: [no_support] bottom_cabinet_no_top_gjrero_2: [support] bottom_cabinet_no_top_gjrero_2 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_no_top_gjrero_3 [bottom_cabinet_no_top]: setup_failed - no_support: [no_support] bottom_cabinet_no_top_gjrero_3: [support] bottom_cabinet_no_top_gjrero_3 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_no_top_gjrero_4 [bottom_cabinet_no_top]: setup_failed - no_support: [no_support] bottom_cabinet_no_top_gjrero_4: [support] bottom_cabinet_no_top_gjrero_4 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_no_top_rkgjer_0 [bottom_cabinet_no_top]: setup_failed - no_support: [no_support] bottom_cabinet_no_top_rkgjer_0: [support] bottom_cabinet_no_top_rkgjer_0 has too little free top surface (tags: no_support, support)
  - countertop_kelker_0 [countertop]: setup_failed - no_support: [no_support] countertop_kelker_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support)
  - countertop_kelzer_0 [countertop]: setup_failed - no_support: [no_support] countertop_kelzer_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - desk_bhkhxo_0 [desk]: setup_failed - no_support: [no_support] desk_bhkhxo_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around desk_bhkhxo_0 (tags: no_support, no_stance, stance_collision)
  - desk_nbpuns_0 [desk]: setup_failed - no_support: [no_support] desk_nbpuns_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around desk_nbpuns_0 (tags: no_support, no_stance, stance_collision)
  - desk_xhjsub_0 [desk]: setup_failed - no_support: [no_support] desk_xhjsub_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable, stance_collision)
  - shelf_amyper_1 [shelf]: setup_failed - no_support: [no_support] shelf_amyper_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around shelf_amyper_1 (tags: no_support, no_stance, stance_collision)
  - shelf_amyper_2 [shelf]: setup_failed - no_support: [no_support] shelf_amyper_2: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around shelf_amyper_2 (tags: no_support, no_stance, stance_collision)
  - shelf_amyper_6 [shelf]: setup_failed - no_support: [no_support] shelf_amyper_6: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around shelf_amyper_6 (tags: no_support, no_stance, stance_collision)
  - shelf_amyper_7 [shelf]: setup_failed - no_support: [no_support] shelf_amyper_7: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around shelf_amyper_7 (tags: no_support, no_stance, stance_collision)

### house_double_floor_lower
- (no viable support)

  rejected:
  - bar_byvbuc_0 [bar]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - bar_egwapq_0 [bar]: setup_failed - no_support: [no_support] bar_egwapq_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, stance_collision)
  - bottom_cabinet_qacthv_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_qacthv_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable, stance_collision)
  - bottom_cabinet_rhdbzv_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_rhdbzv_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable, stance_collision)
  - breakfast_table_xftrki_0 [breakfast_table]: setup_failed - no_support: [no_support] breakfast_table_xftrki_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no traversable stance around breakfast_table_xftrki_0 (tags: no_support, no_stance, stance_collision)
  - coffee_table_koagbh_0 [coffee_table]: setup_failed - no_support: [no_support] coffee_table_koagbh_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)

### house_double_floor_upper
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.88 m, 1113 free cells; occupants ['picture_xnuynn_0']; stance edge +u base (1.39, 2.07, yaw 0.00); trunk translate 0.75 tilt -0.25; n_reachable_cells 12; n_grasp_cells 11
- **bottom_cabinet_nddvba_0** [bottom_cabinet/nddvba, rooms bedroom_2]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.72 m, 779 free cells; stance edge +u base (4.49, -3.86, yaw -1.57); trunk translate 0.75 tilt -0.55; n_reachable_cells 6; n_grasp_cells 6
- **desk_aduafr_0** [desk/aduafr, rooms bedroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 2100 free cells; stance edge +u base (-5.43, 0.79, yaw -1.58); trunk translate 0.25 tilt -0.75; n_reachable_cells 8; n_grasp_cells 8
- **desk_iotfzl_0** [desk/iotfzl, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.74 m, 1792 free cells; occupants ['table_lamp_lshxmu_0']; stance edge -u base (-2.11, -2.99, yaw -0.00); trunk translate 0.25 tilt -0.75; stand-off 0.55; n_reachable_cells 7; n_grasp_cells 6

  rejected:
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: setup_failed - no_support: [no_support] bottom_cabinet_no_top_ncgmqr_0: [support] bottom_cabinet_no_top_ncgmqr_0 has too little free top surface (tags: no_support, support)
  - desk_uqcmzf_0 [desk]: setup_failed - no_support: [no_support] desk_uqcmzf_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - nightstand_roxnbe_0 [nightstand]: setup_failed - no_support: [no_support] nightstand_roxnbe_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around nightstand_roxnbe_0 (tags: no_support, no_stance, stance_collision)
  - nightstand_roxnbe_1 [nightstand]: setup_failed - no_support: [no_support] nightstand_roxnbe_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around nightstand_roxnbe_1 (tags: no_support, no_stance, stance_collision)
  - ottoman_qsnawv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_qsnawv_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - serving_cart_vvtcby_0 [serving_cart]: setup_failed - no_support: [no_support] serving_cart_vvtcby_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable, stance_collision)

### Rs_int
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 1919 free cells; occupants ['loudspeaker_bmpdyv_0', 'pot_plant_jatssq_1', 'standing_tv_udotid_0']; stance edge +u base (0.74, -1.55, yaw 0.00); trunk translate 0.50 tilt -0.60; stand-off 0.55; n_reachable_cells 3; n_grasp_cells 3
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 12; n_grasp_cells 9
- **breakfast_table_skczfi_0** [breakfast_table/skczfi, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.76 m, 1885 free cells; stance edge +u base (0.64, 0.36, yaw 0.01); trunk translate 0.75 tilt -0.50; n_reachable_cells 5; n_grasp_cells 4
- **breakfast_table_skczfi_1** [breakfast_table/skczfi, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.76 m, 1787 free cells; stance edge +u base (0.43, -2.98, yaw -1.57); trunk translate 0.75 tilt -0.35; stand-off 0.55; n_reachable_cells 8; n_grasp_cells 6

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_bamfsz_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no traversable stance around bottom_cabinet_bamfsz_0 (tags: no_support, no_stance, stance_collision)
  - bottom_cabinet_bamfsz_1 [bottom_cabinet]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - countertop_tpuwys_0 [countertop]: no_instances - template ok, no instance in 15 attempts: partner_placement x7, target_placement x6, init_false x2

### hotel_suite_large
- **desk_ohjotf_0** [desk/ohjotf, rooms bedroom_0]: 1 instance(s) in 2 attempt(s), rejections {'target_placement': 1}; top z 0.68 m, 2403 free cells; stance edge +v base (-1.71, -4.63, yaw -0.00); trunk translate 0.75 tilt -0.70; n_reachable_cells 6; n_grasp_cells 5
- **shelf_rglkkf_0** [shelf/rglkkf, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.66 m, 2064 free cells; stance edge +u base (0.96, 3.27, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 4; n_grasp_cells 3

  rejected:
  - nightstand_maeowc_0 [nightstand]: setup_failed - no_support: [no_support] nightstand_maeowc_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around nightstand_maeowc_0 (tags: no_support, no_stance, stance_collision)
  - nightstand_maeowc_1 [nightstand]: setup_failed - no_support: [no_support] nightstand_maeowc_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around nightstand_maeowc_1 (tags: no_support, no_stance, stance_collision)
  - nightstand_maeowc_2 [nightstand]: setup_failed - no_support: [no_support] nightstand_maeowc_2: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around nightstand_maeowc_2 (tags: no_support, no_stance, stance_collision)
  - ottoman_hnmndv_0 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - ottoman_hnmndv_1 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_1: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - shelf_aqtmhm_0 [shelf]: setup_failed - no_support: [no_support] shelf_aqtmhm_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around shelf_aqtmhm_0 (tags: no_support, no_stance, stance_collision)
  - shelf_xogqqm_1 [shelf]: setup_failed - no_support: [no_support] shelf_xogqqm_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around shelf_xogqqm_1 (tags: no_support, no_stance, stance_collision, support)
  - shelf_xogqqm_2 [shelf]: setup_failed - no_support: [no_support] shelf_xogqqm_2: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around shelf_xogqqm_2 (tags: no_support, no_stance, stance_collision, unreachable)

### office_cubicles_right
- **conference_table_fpkmyw_0** [conference_table/fpkmyw, rooms meeting_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.67 m, 12996 free cells; stance edge +v base (6.23, 15.60, yaw -1.57); trunk translate 0.00 tilt -0.90; n_reachable_cells 10; n_grasp_cells 8
- **desk_mdhelw_0** [desk/mdhelw, rooms private_office_3]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -12.73, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 7; n_grasp_cells 7
- **desk_mdhelw_1** [desk/mdhelw, rooms private_office_2]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.79 m, 6439 free cells; stance edge -v base (18.02, -8.39, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 7; n_grasp_cells 7
- **desk_mdhelw_3** [desk/mdhelw, rooms private_office_1]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -5.06, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 9; n_grasp_cells 8
- **desk_pfsjbc_0** [desk/pfsjbc, rooms lobby_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 8846 free cells; occupants ['walls_wcngqp_0']; stance edge +v base (1.79, -15.50, yaw 3.14); trunk translate 0.75 tilt -0.45; n_reachable_cells 8; n_grasp_cells 8

  rejected:
  - bottom_cabinet_immwzb_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_immwzb_0: [support] bottom_cabinet_immwzb_0 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_immwzb_1 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_immwzb_1: [support] bottom_cabinet_immwzb_1 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_immwzb_2 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_immwzb_2: [support] bottom_cabinet_immwzb_2 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_immwzb_3 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_immwzb_3: [support] bottom_cabinet_immwzb_3 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_immwzb_4 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_immwzb_4: [support] bottom_cabinet_immwzb_4 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_immwzb_5 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_immwzb_5: [support] bottom_cabinet_immwzb_5 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_immwzb_6 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_immwzb_6: [support] bottom_cabinet_immwzb_6 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_immwzb_7 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_immwzb_7: [support] bottom_cabinet_immwzb_7 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_immwzb_8 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_immwzb_8: [support] bottom_cabinet_immwzb_8 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_immwzb_9 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_immwzb_9: [support] bottom_cabinet_immwzb_9 has too little free top surface (tags: no_support, support)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around bottom_cabinet_jhymlr_0 (tags: no_support, no_stance, stance_collision, unreachable)
  - coffee_table_mtchfd_0 [coffee_table]: setup_failed - no_support: [no_support] coffee_table_mtchfd_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - countertop_zkhelh_0 [countertop]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - cubicle_deceti_0 [cubicle]: setup_failed - no_support: [no_support] cubicle_deceti_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around cubicle_deceti_0 (tags: no_support, no_stance, stance_collision)
  - cubicle_deceti_1 [cubicle]: setup_failed - no_support: [no_support] cubicle_deceti_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around cubicle_deceti_1 (tags: no_support, no_stance, stance_collision)
  - cubicle_deceti_2 [cubicle]: setup_failed - no_support: [no_support] cubicle_deceti_2: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around cubicle_deceti_2 (tags: no_support, no_stance, stance_collision)
  - cubicle_deceti_3 [cubicle]: setup_failed - no_support: [no_support] cubicle_deceti_3: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around cubicle_deceti_3 (tags: no_support, no_stance, stance_collision)
  - cubicle_hzalfx_0 [cubicle]: setup_failed - no_support: [no_support] cubicle_hzalfx_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around cubicle_hzalfx_0 (tags: no_support, no_stance, stance_collision)
  - cubicle_hzalfx_1 [cubicle]: setup_failed - no_support: [no_support] cubicle_hzalfx_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around cubicle_hzalfx_1 (tags: no_support, no_stance, stance_collision)
  - cubicle_hzalfx_2 [cubicle]: setup_failed - no_support: [no_support] cubicle_hzalfx_2: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around cubicle_hzalfx_2 (tags: no_support, no_stance, stance_collision)
  - cubicle_hzalfx_3 [cubicle]: setup_failed - no_support: [no_support] cubicle_hzalfx_3: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around cubicle_hzalfx_3 (tags: no_support, no_stance, stance_collision)
  - cubicle_hzalfx_4 [cubicle]: setup_failed - no_support: [no_support] cubicle_hzalfx_4: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around cubicle_hzalfx_4 (tags: no_support, no_stance, stance_collision)
  - cubicle_hzalfx_5 [cubicle]: setup_failed - no_support: [no_support] cubicle_hzalfx_5: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around cubicle_hzalfx_5 (tags: no_support, no_stance, stance_collision)
  - desk_mdhelw_2 [desk]: setup_failed - no_support: [no_support] desk_mdhelw_2: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around desk_mdhelw_2 (tags: no_support, no_stance, stance_collision)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 7; n_grasp_cells 6
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 7; n_grasp_cells 6
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 4; n_grasp_cells 4
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 4; n_grasp_cells 4
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.50 tilt -0.45; n_reachable_cells 7; n_grasp_cells 7
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge +u base (-2.38, 2.12, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 8; n_grasp_cells 8
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 9; n_grasp_cells 9
- **pedestal_table_nlfgwe_2** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -6.73, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 9; n_grasp_cells 9

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, unreachable)
  - bench_xwphjd_0 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around bench_xwphjd_0 (tags: no_support, no_stance, stance_collision)
  - bench_xwphjd_1 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (33 candidates) (tags: no_support, no_stance, stance_collision, unreachable)
  - bench_xwphjd_2 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_2: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_3 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_3: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_4 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_4: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - bench_xwphjd_5 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_5: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (33 candidates) (tags: no_support, no_stance, stance_collision, unreachable)
  - bench_xwphjd_6 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_6: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around bench_xwphjd_6 (tags: no_support, no_stance, stance_collision)
  - bench_xwphjd_7 [bench]: setup_failed - no_support: [no_support] bench_xwphjd_7: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (33 candidates) (tags: no_support, no_stance, stance_collision, unreachable)
  - commercial_kitchen_table_nisnrl_0 [commercial_kitchen_table]: setup_failed - no_support: [no_support] commercial_kitchen_table_nisnrl_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around commercial_kitchen_table_nisnrl_0 (tags: no_support, no_stance, stance_collision)
  - commercial_kitchen_table_vxvtec_0 [commercial_kitchen_table]: setup_failed - no_support: [no_support] commercial_kitchen_table_vxvtec_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around commercial_kitchen_table_vxvtec_0 (tags: no_support, no_stance, stance_collision)
  - metal_bottom_cabinet_bbjuqu_0 [metal_bottom_cabinet]: setup_failed - no_support: [no_support] metal_bottom_cabinet_bbjuqu_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around metal_bottom_cabinet_bbjuqu_0 (tags: no_support, no_stance, stance_collision)
  - metal_bottom_cabinet_bbjuqu_1 [metal_bottom_cabinet]: setup_failed - no_support: [no_support] metal_bottom_cabinet_bbjuqu_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around metal_bottom_cabinet_bbjuqu_1 (tags: no_support, no_stance, stance_collision)
  - metal_bottom_cabinet_bbjuqu_2 [metal_bottom_cabinet]: setup_failed - no_support: [no_support] metal_bottom_cabinet_bbjuqu_2: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around metal_bottom_cabinet_bbjuqu_2 (tags: no_support, no_stance, stance_collision)
  - metal_bottom_cabinet_bbjuqu_3 [metal_bottom_cabinet]: setup_failed - no_support: [no_support] metal_bottom_cabinet_bbjuqu_3: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around metal_bottom_cabinet_bbjuqu_3 (tags: no_support, no_stance, stance_collision)

## inter_object_relation_satisfaction-1

### house_single_floor
- (no viable support)
- bottom_cabinet_xqmipa_0 (partner bed_rrcvaq_1): instances were written but the sampler bound another support (see results json)
- desk_bhkhxo_0 (partner bed_rrcvaq_1): instances were written but the sampler bound another support (see results json)
- desk_xhjsub_0 (partner bed_rrcvaq_0): instances were written but the sampler bound another support (see results json)

  rejected:
  - bar_udatjt_0 (partner sofa_lugrhk_1) [bar]: setup_failed - no_support: [no_support] bar_udatjt_0: [no_stance] stand_off 0.45 edge None: [partner_not_visible] sofa_lugrhk_1 corners in view: 0/8 (tags: no_support, no_stance, partner_not_visible, stance_collision, partner_far, unreachable)
  - bar_udatjt_0 (partner sofa_wnzdke_0) [bar]: no_instances - template ok, no instance in 6 attempts: under_goal x6
  - bottom_cabinet_xqmipa_0 (partner bed_ivdnny_0) [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_xqmipa_0: [no_stance] stand_off 0.45 edge None: [partner_not_visible] bed_ivdnny_0 corners in view: 0/8 (tags: no_support, no_stance, partner_not_visible, stance_collision, unreachable)
  - coffee_table_qtwqqb_0 (partner sofa_bskyog_0) [coffee_table]: setup_failed - no_support: [no_support] coffee_table_qtwqqb_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable, stance_collision, partner_not_visible)
  - coffee_table_qtwqqb_0 (partner sofa_cozrjc_0) [coffee_table]: setup_failed - no_support: [no_support] coffee_table_qtwqqb_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable, stance_collision, partner_not_visible, partner_far)
  - coffee_table_qtwqqb_0 (partner sofa_cozrjc_1) [coffee_table]: no_instances - template ok, no instance in 6 attempts: under_goal x6
  - desk_nbpuns_0 (partner bed_ivdnny_0) [desk]: setup_failed - no_support: [no_support] desk_nbpuns_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [no_stance] no collision-free stance around desk_nbpuns_0 (tags: no_support, no_stance, stance_collision, unreachable)

### house_double_floor_lower
- (no viable support)

  rejected:
  - coffee_table_koagbh_0 (partner sofa_frlxgz_0) [coffee_table]: setup_failed - no_support: [no_support] coffee_table_koagbh_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable, stance_collision, support)

### house_double_floor_upper
- (no viable support)

  rejected:
  - bottom_cabinet_jhymlr_0 (partner bed_siuxgz_0) [bottom_cabinet]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - bottom_cabinet_nddvba_0 (partner bed_tjfzsb_0) [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_nddvba_0: [no_stance] stand_off 0.45 edge None: [partner_not_visible] bed_tjfzsb_0 corners in view: 0/8 (tags: no_support, no_stance, partner_not_visible, stance_collision, unreachable)
  - bottom_cabinet_no_top_ncgmqr_0 (partner bed_tjfzsb_0) [bottom_cabinet_no_top]: setup_failed - no_support: [no_support] bottom_cabinet_no_top_ncgmqr_0: [support] bottom_cabinet_no_top_ncgmqr_0 has too little free top surface (tags: no_support, support, no_stance, unreachable, stance_collision)
  - desk_aduafr_0 (partner bed_klcjcy_0) [desk]: setup_failed - no_support: [no_support] desk_aduafr_0: [no_stance] stand_off 0.45 edge None: [partner_not_visible] bed_klcjcy_0 corners in view: 0/8 (tags: no_support, no_stance, partner_not_visible, stance_collision, unreachable)
  - desk_aduafr_0 (partner sofa_ukfaaz_0) [desk]: setup_failed - no_support: [no_support] desk_aduafr_0: [no_stance] stand_off 0.45 edge None: [partner_not_visible] sofa_ukfaaz_0 corners in view: 2/8 (tags: no_support, no_stance, partner_not_visible, stance_collision, partner_far, support)
  - desk_iotfzl_0 (partner sofa_ukfaaz_0) [desk]: setup_failed - no_support: [no_support] desk_iotfzl_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (37 candidates) (tags: no_support, no_stance, stance_collision, partner_not_visible, support)
  - desk_uqcmzf_0 (partner bed_tjfzsb_0) [desk]: setup_failed - no_support: [no_support] desk_uqcmzf_0: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable, stance_collision, partner_not_visible)
  - nightstand_roxnbe_0 (partner bed_siuxgz_0) [nightstand]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - nightstand_roxnbe_1 (partner bed_siuxgz_0) [nightstand]: no_instances - template ok, no instance in 15 attempts: target_placement x15

### Rs_int
- (no viable support)

  rejected:
  - bottom_cabinet_bamfsz_1 (partner bed_zrumze_0) [bottom_cabinet]: no_instances - template ok, no instance in 6 attempts: under_goal x6
  - bottom_cabinet_bamfsz_1 (partner sofa_mnfbbh_0) [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_bamfsz_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (37 candidates) (tags: no_support, no_stance, stance_collision, partner_not_visible)
  - bottom_cabinet_jhymlr_0 (partner sofa_mnfbbh_0) [bottom_cabinet]: setup_failed - no_support: [no_support] bottom_cabinet_jhymlr_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (38 candidates) (tags: no_support, no_stance, stance_collision, partner_not_visible)
  - breakfast_table_skczfi_0 (partner sofa_mnfbbh_0) [breakfast_table]: setup_failed - no_support: [no_support] breakfast_table_skczfi_0: [no_stance] stand_off 0.45 edge None: [partner_not_visible] sofa_mnfbbh_0 corners in view: 0/8 (tags: no_support, no_stance, partner_not_visible, stance_collision)
  - breakfast_table_skczfi_1 (partner sofa_mnfbbh_0) [breakfast_table]: setup_failed - no_support: [no_support] breakfast_table_skczfi_1: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (38 candidates) (tags: no_support, no_stance, stance_collision, partner_not_visible)
  - countertop_tpuwys_0 (partner bed_zrumze_0) [countertop]: no_instances - template ok, no instance in 6 attempts: under_goal x6
  - countertop_tpuwys_0 (partner sofa_mnfbbh_0) [countertop]: setup_failed - no_support: [no_support] countertop_tpuwys_0: [no_stance] stand_off 0.45 edge None: [stance_collision] untucked: [stance_collision] no collision-free viewing trunk configuration (2 candidates) (tags: no_support, no_stance, stance_collision)

### hotel_suite_large
- (no viable support)

  rejected:
  - desk_ohjotf_0 (partner bed_kddswt_0) [desk]: no_instances - template ok, no instance in 6 attempts: under_goal x6
  - nightstand_maeowc_0 (partner bed_kddswt_0) [nightstand]: no_instances - template ok, no instance in 6 attempts: under_goal x6
  - nightstand_maeowc_0 (partner bed_kddswt_1) [nightstand]: no_instances - template ok, no instance in 6 attempts: under_goal x6
  - nightstand_maeowc_1 (partner bed_kddswt_0) [nightstand]: no_instances - template ok, no instance in 6 attempts: under_goal x6
  - nightstand_maeowc_1 (partner bed_kddswt_1) [nightstand]: no_instances - template ok, no instance in 6 attempts: under_goal x6
  - nightstand_maeowc_2 (partner bed_kddswt_0) [nightstand]: no_instances - template ok, no instance in 6 attempts: under_goal x6
  - nightstand_maeowc_2 (partner bed_kddswt_1) [nightstand]: no_instances - template ok, no instance in 6 attempts: under_goal x6

## inter_object_relation_satisfaction-2

### house_single_floor
- **bar_udatjt_0** [bar/udatjt, rooms kitchen_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 1.05 m, 8605 free cells; stance edge +v base (9.23, 0.27, yaw 3.14); trunk translate 0.75 tilt 0.10; n_reachable_cells 14; n_grasp_cells 14
- **bottom_cabinet_xqmipa_0** [bottom_cabinet/xqmipa, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.74 m, 1254 free cells; stance edge +u base (21.66, 11.32, yaw -3.14); trunk translate 0.00 tilt -0.90; n_reachable_cells 10; n_grasp_cells 8
- **breakfast_table_rhjoby_0** [breakfast_table/rhjoby, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.81 m, 8463 free cells; stance edge -v base (9.19, 3.54, yaw 3.14); trunk translate 0.75 tilt -0.40; n_reachable_cells 14; n_grasp_cells 12
- **coffee_table_qtwqqb_0** [coffee_table/qtwqqb, rooms living_room_1]: 1 instance(s) in 2 attempt(s), rejections {'target_placement': 1}; top z 0.49 m, 3547 free cells; stance edge -v base (1.26, 5.44, yaw -1.57); trunk translate 0.00 tilt -0.90; stand-off 0.35; n_reachable_cells 3; n_grasp_cells 3
- **countertop_fjkase_0** [countertop/fjkase, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.92 m, 4026 free cells; occupants ['wardrobe_pyuwgc_0']; stance edge -u base (22.70, 0.62, yaw 3.14); trunk translate 0.75 tilt -0.20; n_reachable_cells 14; n_grasp_cells 14
- **countertop_ikwqer_0** [countertop/ikwqer, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.93 m, 6620 free cells; occupants ['fixed_window_qvqecb_0', 'tabletop_sink_vekaaf_0']; stance edge +u base (23.74, -0.11, yaw -0.00); trunk translate 0.75 tilt -0.15; n_reachable_cells 18; n_grasp_cells 18
- **countertop_sjxber_0** [countertop/sjxber, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.93 m, 1078 free cells; stance edge +u base (23.66, -1.40, yaw -1.57); trunk translate 0.75 tilt -0.15; n_reachable_cells 17; n_grasp_cells 14

  rejected:
  - bottom_cabinet_no_top_gjeoer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_1 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_2 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_3 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_4 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_rkgjer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - countertop_kelker_0 [countertop]: setup_failed - no_support: [no_support] countertop_kelker_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support)
  - countertop_kelzer_0 [countertop]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - desk_bhkhxo_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_nbpuns_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_xhjsub_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_amyper_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_6 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_7 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### house_double_floor_lower
- (no viable support)

  rejected:
  - bar_byvbuc_0 [bar]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - bar_egwapq_0 [bar]: setup_failed - no_support: [no_support] bar_egwapq_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, stance_collision)
  - bottom_cabinet_qacthv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bottom_cabinet_rhdbzv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - breakfast_table_xftrki_0 [breakfast_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - coffee_table_koagbh_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### house_double_floor_upper
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.88 m, 1113 free cells; occupants ['picture_xnuynn_0']; stance edge +u base (1.39, 2.07, yaw 0.00); trunk translate 0.75 tilt -0.25; n_reachable_cells 13; n_grasp_cells 12
- **bottom_cabinet_nddvba_0** [bottom_cabinet/nddvba, rooms bedroom_2]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.72 m, 779 free cells; stance edge +u base (4.49, -3.86, yaw -1.57); trunk translate 0.75 tilt -0.55; n_reachable_cells 11; n_grasp_cells 11
- **desk_aduafr_0** [desk/aduafr, rooms bedroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 2100 free cells; stance edge +u base (-5.43, 0.79, yaw -1.58); trunk translate 0.25 tilt -0.75; n_reachable_cells 9; n_grasp_cells 9
- **desk_iotfzl_0** [desk/iotfzl, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.74 m, 1792 free cells; occupants ['table_lamp_lshxmu_0']; stance edge -u base (-2.11, -2.99, yaw -0.00); trunk translate 0.25 tilt -0.75; stand-off 0.55; n_reachable_cells 7; n_grasp_cells 7

  rejected:
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - desk_uqcmzf_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - nightstand_roxnbe_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_qsnawv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - serving_cart_vvtcby_0 [serving_cart]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### Rs_int
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 1919 free cells; occupants ['loudspeaker_bmpdyv_0', 'pot_plant_jatssq_1', 'standing_tv_udotid_0']; stance edge +u base (0.74, -1.55, yaw 0.00); trunk translate 0.50 tilt -0.60; stand-off 0.55; n_reachable_cells 4; n_grasp_cells 4
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 14; n_grasp_cells 14
- **breakfast_table_skczfi_0** [breakfast_table/skczfi, rooms living_room_0]: 1 instance(s) in 2 attempt(s), rejections {'target_placement': 1}; top z 0.76 m, 1885 free cells; stance edge +u base (0.64, 0.36, yaw 0.01); trunk translate 0.75 tilt -0.50; n_reachable_cells 11; n_grasp_cells 11
- **breakfast_table_skczfi_1** [breakfast_table/skczfi, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.76 m, 1787 free cells; stance edge +u base (0.43, -2.98, yaw -1.57); trunk translate 0.75 tilt -0.35; stand-off 0.55; n_reachable_cells 8; n_grasp_cells 8

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bottom_cabinet_bamfsz_1 [bottom_cabinet]: no_instances - template ok, no instance in 15 attempts: partner_placement x15
  - countertop_tpuwys_0 [countertop]: no_instances - template ok, no instance in 15 attempts: init_false x10, target_placement x5

### hotel_suite_large
- **desk_ohjotf_0** [desk/ohjotf, rooms bedroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.68 m, 2403 free cells; stance edge +v base (-1.71, -4.63, yaw 0.00); trunk translate 0.75 tilt -0.70; n_reachable_cells 8; n_grasp_cells 7
- **shelf_rglkkf_0** [shelf/rglkkf, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.66 m, 2064 free cells; stance edge +u base (0.96, 3.27, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 10; n_grasp_cells 8

  rejected:
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - ottoman_hnmndv_1 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### office_cubicles_right
- **conference_table_fpkmyw_0** [conference_table/fpkmyw, rooms meeting_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.67 m, 12996 free cells; stance edge +v base (6.23, 15.60, yaw -1.57); trunk translate 0.00 tilt -0.90; n_reachable_cells 10; n_grasp_cells 10
- **desk_mdhelw_0** [desk/mdhelw, rooms private_office_3]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -12.73, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 10; n_grasp_cells 10
- **desk_mdhelw_1** [desk/mdhelw, rooms private_office_2]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.79 m, 6439 free cells; stance edge -v base (18.02, -8.39, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 9; n_grasp_cells 9
- **desk_mdhelw_3** [desk/mdhelw, rooms private_office_1]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -5.06, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 11; n_grasp_cells 11
- **desk_pfsjbc_0** [desk/pfsjbc, rooms lobby_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 8846 free cells; occupants ['walls_wcngqp_0']; stance edge +v base (1.79, -15.50, yaw 3.14); trunk translate 0.75 tilt -0.45; n_reachable_cells 10; n_grasp_cells 10

  rejected:
  - bottom_cabinet_immwzb_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_1 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_2 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_3 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_4 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_5 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_6 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_7 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_8 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_9 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - coffee_table_mtchfd_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - countertop_zkhelh_0 [countertop]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - cubicle_deceti_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_4 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_5 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_mdhelw_2 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 11; n_grasp_cells 9
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 11; n_grasp_cells 9
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 9; n_grasp_cells 9
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 9; n_grasp_cells 9
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.50 tilt -0.45; n_reachable_cells 7; n_grasp_cells 7
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'target_placement': 1}; top z 0.78 m, 2222 free cells; stance edge +u base (-2.38, 2.12, yaw 3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 10; n_grasp_cells 10
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 14; n_grasp_cells 13
- **pedestal_table_nlfgwe_2** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -6.73, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 14; n_grasp_cells 14

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, unreachable)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_2 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_3 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_nisnrl_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_vxvtec_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_0 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_1 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_2 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_3 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

## inter_object_relation_satisfaction-3

### house_single_floor
- **bar_udatjt_0** [bar/udatjt, rooms kitchen_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 1.05 m, 8605 free cells; stance edge +v base (9.23, 0.27, yaw 3.14); trunk translate 0.75 tilt 0.10; n_reachable_cells 13; n_grasp_cells 13
- **bottom_cabinet_xqmipa_0** [bottom_cabinet/xqmipa, rooms bathroom_0]: 1 instance(s) in 2 attempt(s), rejections {'target_placement': 1}; top z 0.74 m, 1254 free cells; stance edge +u base (21.66, 11.32, yaw -3.14); trunk translate 0.00 tilt -0.90; n_reachable_cells 10; n_grasp_cells 10
- **breakfast_table_rhjoby_0** [breakfast_table/rhjoby, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.81 m, 8463 free cells; stance edge -v base (9.19, 3.54, yaw 3.14); trunk translate 0.75 tilt -0.40; n_reachable_cells 5; n_grasp_cells 5
- **countertop_fjkase_0** [countertop/fjkase, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.92 m, 4026 free cells; occupants ['wardrobe_pyuwgc_0']; stance edge -u base (22.70, 0.62, yaw 3.14); trunk translate 0.75 tilt -0.20; n_reachable_cells 9; n_grasp_cells 9
- **countertop_ikwqer_0** [countertop/ikwqer, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.93 m, 6620 free cells; occupants ['fixed_window_qvqecb_0', 'tabletop_sink_vekaaf_0']; stance edge +u base (23.74, -0.11, yaw -0.00); trunk translate 0.75 tilt -0.15; n_reachable_cells 11; n_grasp_cells 11
- **countertop_sjxber_0** [countertop/sjxber, rooms utility_room_0]: 1 instance(s) in 3 attempt(s), rejections {'partner_placement': 2}; top z 0.93 m, 1078 free cells; stance edge +u base (23.66, -1.40, yaw -1.57); trunk translate 0.75 tilt -0.15; n_reachable_cells 13; n_grasp_cells 13

  rejected:
  - bottom_cabinet_no_top_gjeoer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_1 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_2 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_3 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_4 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_rkgjer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - coffee_table_qtwqqb_0 [coffee_table]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - countertop_kelker_0 [countertop]: setup_failed - no_support: [no_support] countertop_kelker_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support)
  - countertop_kelzer_0 [countertop]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - desk_bhkhxo_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_nbpuns_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_xhjsub_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_amyper_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_6 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_7 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### house_double_floor_lower
- (no viable support)

  rejected:
  - bar_byvbuc_0 [bar]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - bar_egwapq_0 [bar]: setup_failed - no_support: [no_support] bar_egwapq_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, stance_collision)
  - bottom_cabinet_qacthv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bottom_cabinet_rhdbzv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - breakfast_table_xftrki_0 [breakfast_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - coffee_table_koagbh_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### house_double_floor_upper
- **desk_aduafr_0** [desk/aduafr, rooms bedroom_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.77 m, 2100 free cells; stance edge +u base (-5.43, 0.79, yaw -1.58); trunk translate 0.25 tilt -0.75; n_reachable_cells 7; n_grasp_cells 7
- **desk_iotfzl_0** [desk/iotfzl, rooms living_room_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.74 m, 1791 free cells; occupants ['table_lamp_lshxmu_0']; stance edge -u base (-2.11, -2.99, yaw -0.00); trunk translate 0.25 tilt -0.75; stand-off 0.55; n_reachable_cells 6; n_grasp_cells 6

  rejected:
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: no_instances - template ok, no instance in 15 attempts: target_placement x10, partner_placement x5
  - bottom_cabinet_nddvba_0 [bottom_cabinet]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - desk_uqcmzf_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - nightstand_roxnbe_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_qsnawv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - serving_cart_vvtcby_0 [serving_cart]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### Rs_int
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms living_room_0]: 1 instance(s) in 3 attempt(s), rejections {'partner_placement': 2}; top z 0.75 m, 1919 free cells; occupants ['loudspeaker_bmpdyv_0', 'pot_plant_jatssq_1', 'standing_tv_udotid_0']; stance edge +u base (0.74, -1.55, yaw 0.00); trunk translate 0.50 tilt -0.60; stand-off 0.55; n_reachable_cells 3; n_grasp_cells 3
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 5 attempt(s), rejections {'partner_placement': 2, 'target_placement': 2}; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 7; n_grasp_cells 7
- **breakfast_table_skczfi_0** [breakfast_table/skczfi, rooms living_room_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.76 m, 1885 free cells; stance edge +u base (0.64, 0.36, yaw 0.01); trunk translate 0.75 tilt -0.50; n_reachable_cells 3; n_grasp_cells 3
- **breakfast_table_skczfi_1** [breakfast_table/skczfi, rooms living_room_0]: 1 instance(s) in 2 attempt(s), rejections {'target_placement': 1}; top z 0.76 m, 1787 free cells; stance edge +u base (0.43, -2.98, yaw -1.57); trunk translate 0.75 tilt -0.35; stand-off 0.55; n_reachable_cells 5; n_grasp_cells 5

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bottom_cabinet_bamfsz_1 [bottom_cabinet]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - countertop_tpuwys_0 [countertop]: no_instances - template ok, no instance in 15 attempts: partner_placement x14, target_placement x1

### hotel_suite_large
- **desk_ohjotf_0** [desk/ohjotf, rooms bedroom_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.68 m, 2403 free cells; stance edge +v base (-1.71, -4.63, yaw 0.00); trunk translate 0.75 tilt -0.70; n_reachable_cells 1; n_grasp_cells 1
- **shelf_rglkkf_0** [shelf/rglkkf, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.66 m, 2064 free cells; stance edge +u base (0.96, 3.27, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 1; n_grasp_cells 1

  rejected:
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - ottoman_hnmndv_1 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### office_cubicles_right
- **conference_table_fpkmyw_0** [conference_table/fpkmyw, rooms meeting_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.67 m, 12996 free cells; stance edge +v base (6.23, 15.60, yaw -1.57); trunk translate 0.00 tilt -0.90; n_reachable_cells 10; n_grasp_cells 10
- **desk_mdhelw_0** [desk/mdhelw, rooms private_office_3]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -12.73, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 4; n_grasp_cells 4
- **desk_mdhelw_1** [desk/mdhelw, rooms private_office_2]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.79 m, 6439 free cells; stance edge -v base (18.02, -8.39, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 4; n_grasp_cells 4
- **desk_mdhelw_3** [desk/mdhelw, rooms private_office_1]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -5.06, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 5; n_grasp_cells 5
- **desk_pfsjbc_0** [desk/pfsjbc, rooms lobby_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 8846 free cells; occupants ['walls_wcngqp_0']; stance edge +v base (1.79, -15.50, yaw 3.14); trunk translate 0.75 tilt -0.45; n_reachable_cells 4; n_grasp_cells 4

  rejected:
  - bottom_cabinet_immwzb_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_1 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_2 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_3 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_4 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_5 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_6 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_7 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_8 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_9 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - coffee_table_mtchfd_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - countertop_zkhelh_0 [countertop]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - cubicle_deceti_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_4 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_5 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_mdhelw_2 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 4; n_grasp_cells 4
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 4; n_grasp_cells 4
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 1; n_grasp_cells 1
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 3 attempt(s), rejections {'partner_placement': 2}; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 1; n_grasp_cells 1
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.50 tilt -0.45; n_reachable_cells 6; n_grasp_cells 6
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.78 m, 2222 free cells; stance edge +v base (-3.36, 3.10, yaw -1.57); trunk translate 0.75 tilt -0.50; n_reachable_cells 4; n_grasp_cells 4
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 5; n_grasp_cells 5
- **pedestal_table_nlfgwe_2** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -6.73, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 5; n_grasp_cells 5

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, unreachable)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_2 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_3 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_nisnrl_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_vxvtec_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_0 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_1 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_2 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_3 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

## inter_object_relation_satisfaction-4

### house_single_floor
- **bar_udatjt_0** [bar/udatjt, rooms kitchen_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 1.05 m, 8605 free cells; stance edge +v base (9.23, 0.27, yaw 3.14); trunk translate 0.75 tilt 0.10; n_reachable_cells 14
- **bottom_cabinet_xqmipa_0** [bottom_cabinet/xqmipa, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.74 m, 1254 free cells; stance edge +u base (21.66, 11.32, yaw -3.14); trunk translate 0.00 tilt -0.90; n_reachable_cells 11
- **breakfast_table_rhjoby_0** [breakfast_table/rhjoby, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.81 m, 8463 free cells; stance edge -v base (9.19, 3.54, yaw 3.14); trunk translate 0.75 tilt -0.40; n_reachable_cells 13
- **coffee_table_qtwqqb_0** [coffee_table/qtwqqb, rooms living_room_1]: 1 instance(s) in 2 attempt(s), rejections {'target_placement': 1}; top z 0.49 m, 3547 free cells; stance edge -v base (1.26, 5.54, yaw -1.57); trunk translate 0.00 tilt -0.90; n_reachable_cells 2
- **countertop_fjkase_0** [countertop/fjkase, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.92 m, 4026 free cells; occupants ['wardrobe_pyuwgc_0']; stance edge -u base (22.70, 0.62, yaw 3.14); trunk translate 0.75 tilt -0.20; n_reachable_cells 14
- **countertop_ikwqer_0** [countertop/ikwqer, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.93 m, 6620 free cells; occupants ['fixed_window_qvqecb_0', 'tabletop_sink_vekaaf_0']; stance edge +u base (23.74, -0.11, yaw -0.00); trunk translate 0.75 tilt -0.15; n_reachable_cells 18
- **countertop_sjxber_0** [countertop/sjxber, rooms utility_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.93 m, 1078 free cells; stance edge +u base (23.66, -1.40, yaw -1.57); trunk translate 0.75 tilt -0.15; n_reachable_cells 16

  rejected:
  - bottom_cabinet_no_top_gjeoer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_1 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_2 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_3 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_gjrero_4 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_no_top_rkgjer_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - countertop_kelker_0 [countertop]: setup_failed - no_support: [no_support] countertop_kelker_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support)
  - countertop_kelzer_0 [countertop]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - desk_bhkhxo_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_nbpuns_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_xhjsub_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - shelf_amyper_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_6 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_amyper_7 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### house_double_floor_lower
- (no viable support)

  rejected:
  - bar_byvbuc_0 [bar]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - bar_egwapq_0 [bar]: setup_failed - no_support: [no_support] bar_egwapq_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, stance_collision)
  - bottom_cabinet_qacthv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bottom_cabinet_rhdbzv_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - breakfast_table_xftrki_0 [breakfast_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - coffee_table_koagbh_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### house_double_floor_upper
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms bathroom_0]: 1 instance(s) in 4 attempt(s), rejections {'goal_infeasible': 3}; top z 0.88 m, 1113 free cells; occupants ['picture_xnuynn_0']; stance edge +u base (1.39, 2.07, yaw 0.00); trunk translate 0.75 tilt -0.25; n_reachable_cells 12
- **bottom_cabinet_nddvba_0** [bottom_cabinet/nddvba, rooms bedroom_2]: 1 instance(s) in 4 attempt(s), rejections {'partner_placement': 1, 'goal_infeasible': 1, 'target_placement': 1}; top z 0.72 m, 779 free cells; stance edge +u base (4.49, -3.86, yaw -1.57); trunk translate 0.75 tilt -0.55; n_reachable_cells 13
- **desk_aduafr_0** [desk/aduafr, rooms bedroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.77 m, 2100 free cells; stance edge +u base (-5.43, 0.79, yaw -1.58); trunk translate 0.25 tilt -0.75; n_reachable_cells 10
- **desk_iotfzl_0** [desk/iotfzl, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.74 m, 1791 free cells; occupants ['table_lamp_lshxmu_0']; stance edge -u base (-2.11, -2.99, yaw -0.00); trunk translate 0.25 tilt -0.75; stand-off 0.55; n_reachable_cells 7

  rejected:
  - bottom_cabinet_no_top_ncgmqr_0 [bottom_cabinet_no_top]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - desk_uqcmzf_0 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - nightstand_roxnbe_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_roxnbe_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_qsnawv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - serving_cart_vvtcby_0 [serving_cart]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off

### Rs_int
- **bottom_cabinet_bamfsz_1** [bottom_cabinet/bamfsz, rooms living_room_0]: 1 instance(s) in 5 attempt(s), rejections {'scene_disturbed': 3, 'goal_infeasible': 1}; top z 0.73 m, 435 free cells; occupants ['pot_plant_udqjui_0']; stance edge +u base (-1.00, -0.45, yaw -3.13); trunk translate 0.50 tilt -0.60; stand-off 0.55; n_reachable_cells 2
- **bottom_cabinet_jhymlr_0** [bottom_cabinet/jhymlr, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 1919 free cells; occupants ['loudspeaker_bmpdyv_0', 'pot_plant_jatssq_1', 'standing_tv_udotid_0']; stance edge +u base (0.74, -1.55, yaw 0.00); trunk translate 0.50 tilt -0.60; stand-off 0.55; n_reachable_cells 5
- **bottom_cabinet_slgzfc_0** [bottom_cabinet/slgzfc, rooms entryway_0]: 1 instance(s) in 3 attempt(s), rejections {'goal_infeasible': 2}; top z 0.84 m, 1350 free cells; stance edge +u base (0.91, 2.29, yaw 0.00); trunk translate 0.75 tilt -0.35; n_reachable_cells 14
- **breakfast_table_skczfi_0** [breakfast_table/skczfi, rooms living_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.76 m, 1885 free cells; stance edge +u base (0.64, 0.36, yaw 0.01); trunk translate 0.75 tilt -0.50; n_reachable_cells 9
- **breakfast_table_skczfi_1** [breakfast_table/skczfi, rooms living_room_0]: 1 instance(s) in 3 attempt(s), rejections {'goal_infeasible': 2}; top z 0.76 m, 1787 free cells; stance edge +u base (0.43, -2.98, yaw -1.57); trunk translate 0.75 tilt -0.35; stand-off 0.55; n_reachable_cells 9

  rejected:
  - bottom_cabinet_bamfsz_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - countertop_tpuwys_0 [countertop]: no_instances - template ok, no instance in 15 attempts: target_placement x7, init_false x5, partner_placement x3

### hotel_suite_large
- **desk_ohjotf_0** [desk/ohjotf, rooms bedroom_0]: 1 instance(s) in 3 attempt(s), rejections {'partner_placement': 1, 'target_placement': 1}; top z 0.68 m, 2403 free cells; stance edge +v base (-1.71, -4.63, yaw 0.00); trunk translate 0.75 tilt -0.70; n_reachable_cells 7
- **shelf_rglkkf_0** [shelf/rglkkf, rooms bathroom_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.66 m, 2064 free cells; stance edge +u base (0.96, 3.27, yaw -1.57); trunk translate 0.75 tilt -0.65; n_reachable_cells 8

  rejected:
  - nightstand_maeowc_0 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_1 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - nightstand_maeowc_2 [nightstand]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - ottoman_hnmndv_0 [ottoman]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - ottoman_hnmndv_1 [ottoman]: setup_failed - no_support: [no_support] ottoman_hnmndv_1: [no_stance] stand_off 0.45 edge None: [unreachable] no reachable cell on the support (tags: no_support, no_stance, unreachable)
  - shelf_aqtmhm_0 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - shelf_xogqqm_1 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - shelf_xogqqm_2 [shelf]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)

### office_cubicles_right
- **conference_table_fpkmyw_0** [conference_table/fpkmyw, rooms meeting_room_0]: 1 instance(s) in 2 attempt(s), rejections {'goal_infeasible': 1}; top z 0.67 m, 12996 free cells; stance edge +v base (6.23, 15.60, yaw -1.57); trunk translate 0.00 tilt -0.90; n_reachable_cells 12
- **desk_mdhelw_0** [desk/mdhelw, rooms private_office_3]: 1 instance(s) in 2 attempt(s), rejections {'goal_infeasible': 1}; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -12.73, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 10
- **desk_mdhelw_1** [desk/mdhelw, rooms private_office_2]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.79 m, 6439 free cells; stance edge -v base (18.02, -8.39, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 11
- **desk_mdhelw_3** [desk/mdhelw, rooms private_office_1]: 1 instance(s) in 3 attempt(s), rejections {'target_placement': 1, 'goal_infeasible': 1}; top z 0.79 m, 6439 free cells; stance edge +v base (17.94, -5.06, yaw -3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 11
- **desk_pfsjbc_0** [desk/pfsjbc, rooms lobby_0]: 1 instance(s) in 2 attempt(s), rejections {'goal_infeasible': 1}; top z 0.77 m, 8846 free cells; occupants ['walls_wcngqp_0']; stance edge +v base (1.79, -15.50, yaw 3.14); trunk translate 0.75 tilt -0.45; n_reachable_cells 11

  rejected:
  - bottom_cabinet_immwzb_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_1 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_2 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_3 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_4 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_5 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_6 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_7 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_8 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_immwzb_9 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bottom_cabinet_jhymlr_0 [bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - coffee_table_mtchfd_0 [coffee_table]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - countertop_zkhelh_0 [countertop]: no_instances - template ok, no instance in 15 attempts: target_placement x15
  - cubicle_deceti_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_deceti_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_0 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_1 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_2 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_3 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_4 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - cubicle_hzalfx_5 [cubicle]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - desk_mdhelw_2 [desk]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

### restaurant_diner
- **booth_xzrpar_0** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 2.65, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 10
- **booth_xzrpar_1** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 3 attempt(s), rejections {'target_placement': 2}; top z 0.75 m, 2309 free cells; stance edge -v base (-0.48, 0.40, yaw -0.00); trunk translate 0.75 tilt -0.60; n_reachable_cells 10
- **booth_xzrpar_2** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.75 m, 2307 free cells; stance edge -v base (0.12, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 8
- **booth_xzrpar_3** [booth/xzrpar, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'partner_placement': 1}; top z 0.75 m, 2307 free cells; stance edge -v base (-2.13, -8.01, yaw -1.57); trunk translate 0.75 tilt -0.60; n_reachable_cells 8
- **bottom_cabinet_olgoza_0** [bottom_cabinet/olgoza, rooms bar_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.84 m, 2205 free cells; stance edge +u base (-3.81, -2.01, yaw -3.14); trunk translate 0.50 tilt -0.45; n_reachable_cells 7
- **pedestal_table_nlfgwe_0** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 1 attempt(s), rejections none; top z 0.78 m, 2222 free cells; stance edge +u base (-2.38, 2.12, yaw 3.14); trunk translate 0.75 tilt -0.50; n_reachable_cells 9
- **pedestal_table_nlfgwe_1** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 4 attempt(s), rejections {'goal_infeasible': 2, 'target_placement': 1}; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -4.16, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 13
- **pedestal_table_nlfgwe_2** [pedestal_table/nlfgwe, rooms dining_room_0]: 1 instance(s) in 2 attempt(s), rejections {'goal_infeasible': 1}; top z 0.78 m, 2222 free cells; stance edge -u base (-0.23, -6.73, yaw -0.00); trunk translate 0.75 tilt -0.50; n_reachable_cells 13

  rejected:
  - bar_xxftww_0 [bar]: setup_failed - no_support: [no_support] bar_xxftww_0: [no_stance] stand_off 0.45 edge None: [support] placement region empty (tags: no_support, no_stance, support, unreachable)
  - bench_xwphjd_0 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_1 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_2 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_3 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_4 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off
  - bench_xwphjd_5 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - bench_xwphjd_6 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - bench_xwphjd_7 [bench]: skipped - inter_object_relation_satisfaction-0 found no stance / no reachable cell on the support top at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_nisnrl_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - commercial_kitchen_table_vxvtec_0 [commercial_kitchen_table]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_0 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_1 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_2 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)
  - metal_bottom_cabinet_bbjuqu_3 [metal_bottom_cabinet]: skipped - inter_object_relation_satisfaction-0 found no stance around the support at any stand-off (precheck: no stance either)

