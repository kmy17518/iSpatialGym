# Wall candidates for world_alignment-1 (world vertical)

Survey: `scripts/survey_walls.py` (dense wall spots every 0.25 m at 1.45 m height, the sampler's own flat-patch / front-clearance / arc-base / pushing-reach checks, then one full sampler instance per region). Reference target painting/dacihj (0.50 x 0.42 m). `max patch` = largest flat + unobstructed square (m); `bases` = collision-free arc bases over the region's spots, `reach` = those with pushing reach of a lower corner; `instance` = full instance written by the sampler at the region's middle spot (rejections otherwise).

## house_single_floor

| room | wall object | region | p0 -> p1 (x, y) | normal | length m | max patch m | spots | bases / reach | instance |
|---|---|---|---|---|---|---|---|---|---|
| bathroom_0 | walls_uxgkxd_0 | r00 | (20.84, 11.16) -> (20.84, 11.49) | (1.00, 0.00) | 0.58 | 0.6 | 2 | 7 / 4 | yes |
| bathroom_1 | walls_lbxvwa_0 | r00 | (24.92, 21.06) -> (24.92, 21.06) | (-1.00, 0.00) | 0.25 | 0.8 | 1 | 3 / 2 | yes |
| bathroom_1 | walls_olxjsy_0 | r01 | (24.37, 21.71) -> (24.37, 21.71) | (0.00, -1.00) | 0.25 | 0.8 | 1 | 1 / 1 | yes |
| bedroom_0 | walls_chsrxr_0 | r00 | (20.24, 23.27) -> (20.24, 23.27) | (0.00, -1.00) | 0.25 | 0.8 | 1 | 2 / 1 | no ({'grasp_unreachable': 6}) |
| bedroom_0 | walls_hzhlyw_0 | r01 | (21.40, 25.90) -> (22.45, 25.90) | (0.00, -1.00) | 1.30 | 1.2 | 5 | 27 / 14 | yes |
| childs_room_0 | walls_ipoatq_0 | r00 | (22.05, 19.29) -> (22.05, 19.29) | (0.00, -1.00) | 0.25 | 1.0 | 1 | 2 / 1 | yes |
| childs_room_0 | walls_uxcfbp_0 | r01 | (21.77, 16.54) -> (21.77, 16.54) | (0.00, 1.00) | 0.25 | 0.8 | 1 | 6 / 3 | yes |
| childs_room_0 | walls_vcgulk_0 | r02 | (21.22, 19.29) -> (21.22, 19.29) | (0.00, -1.00) | 0.25 | 0.6 | 1 | 2 / 2 | yes |
| childs_room_1 | walls_ahpkeh_0 | r00 | (20.89, 15.14) -> (21.95, 15.14) | (0.00, -1.00) | 1.30 | 0.0 | 5 | 25 / 13 | no ({'grasp_unreachable': 5, 'too_few_visible': 1}; {'grasp_unreachable': 6}) |
| childs_room_1 | walls_snssjy_0 | r01 | (22.95, 15.14) -> (22.95, 15.14) | (0.00, -1.00) | 0.25 | 0.8 | 1 | 6 / 3 | yes |
| childs_room_2 | walls_cjxcfc_0 | r00 | (24.23, 6.50) -> (24.23, 6.50) | (-1.00, 0.00) | 0.25 | 0.6 | 1 | 6 / 3 | yes |
| childs_room_2 | walls_frgwwc_0 | r01 | (22.94, 5.43) -> (22.94, 5.43) | (0.00, 1.00) | 0.25 | 0.8 | 1 | 6 / 3 | yes |
| childs_room_2 | walls_tvjrao_0 | r02 | (22.00, 5.46) -> (21.02, 5.46) | (0.00, 1.00) | 1.22 | 0.8 | 5 | 30 / 15 | yes |
| closet_0 | walls_ggtriw_0, walls_hzhlyw_0 | - | - | - | - | - | 13 spots, 2 flat + clear, 0 viable | - | no viable region |
| closet_1 | walls_khkkca_0, walls_thipzw_0 | - | - | - | - | - | 7 spots, 0 flat + clear, 0 viable | - | no viable region |
| closet_2 | walls_ahpkeh_0, walls_noclci_0, walls_uxcfbp_0, walls_wtpbpe_0, walls_ybcyqk_0 | - | - | - | - | - | 23 spots, 0 flat + clear, 0 viable | - | no viable region |
| closet_3 | walls_frgwwc_0, walls_nhvigl_0, walls_tlcifm_0, walls_vmsoas_0 | - | - | - | - | - | 31 spots, 1 flat + clear, 0 viable | - | no viable region |
| closet_4 | walls_mianco_0, walls_noclci_0, walls_snssjy_0, walls_uplnmr_0 | - | - | - | - | - | 21 spots, 0 flat + clear, 0 viable | - | no viable region |
| corridor_0 | walls_bpbucg_0 | r00 | (18.08, 9.58) -> (18.08, 11.62) | (1.00, 0.00) | 2.54 | 0.8 | 5 | 30 / 15 | yes |
| corridor_0 | walls_dhsptd_0 | r01 | (18.11, 1.61) -> (18.11, 1.61) | (1.00, 0.00) | 0.50 | 0.6 | 1 | 4 / 3 | not tested |
| corridor_0 | walls_duepvi_0 | r02 | (20.63, -2.28) -> (20.63, -2.28) | (0.00, 1.00) | 0.50 | 0.6 | 1 | 6 / 3 | not tested |
| corridor_0 | walls_evgaml_0 | r03 | (21.34, 1.30) -> (21.34, -1.51) | (-1.00, 0.00) | 3.32 | 1.2 | 7 | 42 / 21 | yes |
| corridor_0 | walls_hczqhb_0 | r04 | (20.15, 18.35) -> (20.15, 17.97) | (-1.00, 0.00) | 0.88 | 0.6 | 2 | 10 / 5 | not tested |
| corridor_0 | walls_isnclr_0 | r05 | (18.11, -1.80) -> (18.11, -1.80) | (1.00, 0.00) | 0.50 | 1.0 | 1 | 3 / 3 | not tested |
| corridor_0 | walls_kkltnz_0 | r06 | (24.77, 2.48) -> (24.32, 2.48) | (0.00, 1.00) | 0.95 | 0.6 | 2 | 12 / 6 | not tested |
| corridor_0 | walls_nhvigl_0 | r07 | (22.49, 4.69) -> (22.49, 4.69) | (-1.00, 0.00) | 0.50 | 1.2 | 1 | 3 / 2 | not tested |
| corridor_0 | walls_pozods_0 | r08 | (19.98, 9.65) -> (19.98, 9.65) | (0.00, -1.00) | 0.50 | 0.6 | 1 | 6 / 3 | not tested |
| corridor_0 | walls_qpkfgi_0 | r09 | (18.11, -0.51) -> (18.11, 0.79) | (1.00, 0.00) | 1.81 | 0.6 | 4 | 24 / 12 | not tested |
| corridor_0 | walls_qsvtbs_0 | r10 | (20.77, 2.48) -> (19.82, 2.48) | (0.00, 1.00) | 1.44 | 0.8 | 3 | 18 / 9 | not tested |
| corridor_0 | walls_qsvtbs_0 | r11 | (20.20, 2.19) -> (20.50, 2.19) | (0.00, -1.00) | 0.80 | 1.2 | 2 | 12 / 6 | not tested |
| corridor_0 | walls_sgqckc_0 | r12 | (18.92, 23.27) -> (18.92, 23.27) | (0.00, -1.00) | 0.50 | 0.6 | 1 | 6 / 3 | not tested |
| corridor_0 | walls_thipzw_0 | r13 | (19.64, 21.45) -> (19.64, 19.95) | (-1.00, 0.00) | 2.00 | 0.6 | 4 | 24 / 12 | yes |
| corridor_0 | walls_tvjrao_0 | r14 | (20.33, 5.32) -> (21.82, 5.32) | (0.00, -1.00) | 1.99 | 1.2 | 4 | 24 / 12 | not tested |
| corridor_0 | walls_urngpl_0 | r15 | (17.71, 3.95) -> (17.71, 3.95) | (0.00, -1.00) | 0.50 | 0.8 | 1 | 6 / 2 | not tested |
| corridor_0 | walls_vmsoas_0 | r16 | (22.94, 3.91) -> (22.94, 3.91) | (0.00, -1.00) | 0.50 | 0.6 | 1 | 1 / 1 | not tested |
| dining_room_0 | walls_lxqjnd_0 | r00 | (4.66, 3.92) -> (4.66, 4.12) | (1.00, 0.00) | 0.44 | 0.6 | 2 | 5 / 4 | yes |
| entryway_0 | walls_nfmhed_0 | r00 | (1.02, 2.25) -> (2.20, 2.25) | (0.00, -1.00) | 1.43 | 0.6 | 5 | 19 / 11 | no ({'scene_reset': 2, 'grasp_unreachable': 5, 'scene_disturbed': 1}) |
| entryway_0 | walls_nfmhed_0 | r01 | (2.66, 2.25) -> (3.10, 2.25) | (0.00, -1.00) | 0.69 | 1.2 | 3 | 6 / 3 | no ({'no_stance': 6}; {'no_stance': 6}) |
| entryway_0 | walls_onwdhr_0 | r02 | (-0.50, 2.25) -> (-0.44, 2.25) | (0.00, -1.00) | 0.31 | 0.6 | 2 | 6 / 4 | no ({'no_stance': 6}; {'no_stance': 6, 'scene_reset': 1}) |
| entryway_0 | walls_wbjwyy_0 | r03 | (-0.93, -1.56) -> (-0.93, -0.97) | (1.00, 0.00) | 0.84 | 0.6 | 4 | 23 / 12 | no ({'scene_reset': 3, 'grasp_unreachable': 5, 'scene_disturbed': 1}; {'no_stance': 6, 'scene_reset': 1}) |
| entryway_0 | walls_xbotyv_0 | r04 | (2.43, 0.89) -> (2.43, 0.89) | (-1.00, 0.00) | 0.25 | 0.6 | 1 | 6 / 3 | no ({'scene_disturbed': 4, 'scene_reset': 4, 'grasp_unreachable': 2}) |
| garage_0 | walls_somuet_0 | r00 | (-14.78, 16.18) -> (-14.70, 17.82) | (1.00, -0.05) | 1.89 | 1.0 | 8 | 40 / 21 | no ({'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 3, 'too_few_visible': 3}) |
| garage_0 | walls_somuet_0 | r01 | (-11.61, 11.08) -> (-11.69, 9.61) | (-1.00, 0.05) | 1.72 | 0.8 | 7 | 40 / 21 | no ({'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 4, 'too_few_visible': 2}) |
| garage_0 | walls_somuet_0 | r02 | (-15.07, 10.18) -> (-15.01, 11.51) | (1.00, -0.05) | 1.58 | 0.6 | 7 | 42 / 21 | no ({'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 4, 'too_few_visible': 2}) |
| garage_0 | walls_somuet_0 | r03 | (-11.65, 17.83) -> (-11.68, 17.12) | (-1.00, 0.05) | 0.96 | 0.8 | 4 | 24 / 12 | no ({'grasp_unreachable': 4, 'too_few_visible': 2}; {'grasp_unreachable': 3, 'too_few_visible': 3}) |
| garage_0 | walls_somuet_0 | r04 | (-12.15, 8.37) -> (-12.45, 8.20) | (-0.51, 0.86) | 0.59 | 0.8 | 3 | 16 / 9 | no ({'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 4, 'too_few_visible': 2}) |
| garage_0 | walls_somuet_0 | r05 | (-13.46, 8.24) -> (-13.68, 8.44) | (0.68, 0.74) | 0.55 | 1.0 | 3 | 16 / 9 | no ({'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 4, 'too_few_visible': 2}) |
| garage_0 | walls_somuet_0 | r06 | (-11.71, 9.17) -> (-11.71, 9.17) | (-1.00, 0.05) | 0.25 | 1.0 | 1 | 5 / 3 | no ({'grasp_unreachable': 2, 'too_few_visible': 4}) |
| kitchen_0 | walls_cnqnew_0, walls_fglkmr_0, walls_ivezzr_0, walls_nfmhed_0, walls_udqkcs_0 | - | - | - | - | - | 32 spots, 3 flat + clear, 0 viable | - | no viable region |
| living_room_0 | walls_dhsptd_0 | r00 | (17.64, 2.11) -> (17.64, 1.59) | (-1.00, 0.00) | 0.77 | 0.6 | 3 | 18 / 9 | yes |
| living_room_0 | walls_urngpl_0 | r01 | (16.85, 3.95) -> (17.51, 3.95) | (0.00, -1.00) | 0.91 | 0.6 | 4 | 24 / 12 | yes |
| living_room_0 | walls_urngpl_0 | r02 | (14.21, 3.95) -> (14.51, 3.95) | (0.00, -1.00) | 0.55 | 0.8 | 3 | 18 / 9 | yes |
| living_room_1 | walls_lxqjnd_0 | r00 | (4.45, 4.44) -> (4.45, 3.96) | (-1.00, 0.00) | 0.73 | 0.6 | 3 | 18 / 9 | yes |
| living_room_1 | walls_nfmhed_0 | r01 | (3.82, 2.53) -> (3.22, 2.53) | (0.00, 1.00) | 0.85 | 1.2 | 4 | 18 / 10 | no ({'scene_reset': 1, 'too_few_visible': 5, 'grasp_unreachable': 1}; {'too_few_visible': 4, 'grasp_unreachable': 2}) |
| sauna_0 | walls_hajmoj_0 | - | - | - | - | - | 179 spots, 44 flat + clear, 0 viable | - | no viable region |
| utility_room_0 | walls_kkltnz_0 | r00 | (23.72, 2.19) -> (23.80, 2.19) | (0.00, -1.00) | 0.33 | 0.6 | 2 | 12 / 6 | yes |
| utility_room_0 | walls_zufhrv_0 | r01 | (23.93, -2.28) -> (23.63, -2.28) | (0.00, 1.00) | 0.55 | 0.8 | 2 | 9 / 5 | yes |

Rooms with a validated region: bathroom_0, bedroom_0, childs_room_2, corridor_0, living_room_0, living_room_1, utility_room_0. Map: `maps/house_single_floor_walls.png`.

## house_double_floor_lower

| room | wall object | region | p0 -> p1 (x, y) | normal | length m | max patch m | spots | bases / reach | instance |
|---|---|---|---|---|---|---|---|---|---|
| bathroom_0 | walls_fwsbwf_0 | r00 | (-4.34, -0.15) -> (-4.34, -0.15) | (1.00, 0.00) | 0.25 | 0.8 | 1 | 1 / 1 | yes |
| bathroom_0 | walls_owxeej_0 | r01 | (-1.56, 0.30) -> (-1.56, -0.16) | (-1.00, 0.00) | 0.70 | 0.8 | 3 | 9 / 6 | yes |
| bathroom_0 | walls_xyrbgy_0 | r02 | (-3.35, -0.98) -> (-3.65, -0.98) | (0.00, 1.00) | 0.55 | 0.8 | 2 | 4 / 4 | yes |
| corridor_0 | walls_abtbfo_0 | r00 | (0.80, 1.66) -> (0.80, -0.54) | (-1.00, 0.00) | 2.45 | 0.8 | 10 | 57 / 29 | no ({'init_false': 5, 'grasp_unreachable': 1}; {'init_false': 3, 'grasp_unreachable': 3}; {'init_false': 3, 'grasp_unreachable': 3}; None AssertionError('/World/scene_0/controllable__r1pro__robot_r1/base_link orientati) |
| corridor_0 | walls_fwsbwf_0 | r01 | (-4.34, 1.69) -> (-4.34, 1.69) | (1.00, 0.00) | 0.25 | 1.0 | 1 | 2 / 2 | yes |
| corridor_0 | walls_qeofms_0 | r02 | (-2.89, 0.85) -> (-3.68, 0.85) | (0.00, 1.00) | 1.04 | 0.8 | 4 | 21 / 12 | yes |
| corridor_0 | walls_weybsr_0 | r03 | (-3.13, 2.23) -> (-2.43, 2.23) | (0.00, -1.00) | 0.95 | 0.6 | 4 | 24 / 12 | yes |
| corridor_0 | walls_weybsr_0 | r04 | (-2.03, 2.23) -> (-1.90, 2.23) | (0.00, -1.00) | 0.38 | 0.6 | 2 | 11 / 6 | yes |
| garage_0 | walls_cvdpaq_0 | r00 | (-4.75, 2.45) -> (-5.66, 2.45) | (0.00, 1.00) | 1.16 | 0.6 | 5 | 30 / 15 | yes |
| garage_0 | walls_oqrbxo_0 | r01 | (-1.55, 5.33) -> (-1.55, 3.23) | (-1.00, 0.00) | 2.36 | 0.8 | 10 | 60 / 30 | no ({'too_few_visible': 4, 'grasp_unreachable': 2}; {'too_few_visible': 4, 'grasp_unreachable': 2}; {'too_few_visible': 4, 'grasp_unreachable': 2}; {'too_few_visible': 4, 'grasp_unreachable': 2}) |
| garage_0 | walls_weybsr_0 | r02 | (-2.06, 2.45) -> (-2.96, 2.45) | (0.00, 1.00) | 1.15 | 0.8 | 4 | 19 / 10 | yes |
| kitchen_0 | walls_cxsmhr_0, walls_dlfarz_0, walls_jemunw_0, walls_jzwmhx_0, walls_lblwvg_0, walls_umqqmh_0, walls_upttgq_0 | - | - | - | - | - | 34 spots, 0 flat + clear, 0 viable | - | no viable region |
| living_room_0 | walls_dlfarz_0 | r00 | (6.42, 3.65) -> (6.42, 3.65) | (-1.00, 0.00) | 0.25 | 0.6 | 1 | 3 / 3 | yes |

Rooms with a validated region: bathroom_0, corridor_0, garage_0. Map: `maps/house_double_floor_lower_walls.png`.

## house_double_floor_upper

| room | wall object | region | p0 -> p1 (x, y) | normal | length m | max patch m | spots | bases / reach | instance |
|---|---|---|---|---|---|---|---|---|---|
| bathroom_0 | walls_ynhwyn_0 | r00 | (-1.21, 1.34) -> (-1.51, 1.34) | (0.00, 1.00) | 0.55 | 0.6 | 3 | 9 / 8 | yes |
| bathroom_0 | walls_ynhwyn_0 | r01 | (1.54, 1.34) -> (1.49, 1.34) | (0.00, 1.00) | 0.31 | 1.0 | 2 | 2 / 2 | yes |
| bedroom_0 | walls_hddwcd_0 | r00 | (-6.28, 1.78) -> (-6.28, 2.08) | (1.00, 0.00) | 0.55 | 1.0 | 2 | 7 / 5 | yes |
| bedroom_0 | walls_jpdrrk_0 | r01 | (-2.63, 1.87) -> (-2.63, 1.87) | (-1.00, 0.00) | 0.25 | 0.8 | 1 | 6 / 3 | yes |
| bedroom_0 | walls_lempwq_0 | r02 | (-4.20, -0.30) -> (-4.20, -0.30) | (0.00, 1.00) | 0.25 | 0.6 | 1 | 1 / 1 | yes |
| bedroom_1 | walls_eguptx_0 | r00 | (6.47, 1.67) -> (6.47, 1.48) | (-1.00, 0.00) | 0.44 | 0.8 | 2 | 6 / 3 | no ({'grasp_unreachable': 6, 'scene_reset': 1}) |
| bedroom_1 | walls_wjblrc_0 | r01 | (5.40, -0.30) -> (3.00, -0.30) | (-0.00, 1.00) | 2.65 | 0.8 | 11 | 57 / 30 | no ({'scene_reset': 4, 'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 3, 'too_few_visible': 3, 'scene_reset': 3}; {'grasp_unreachable': 4, 'too_few_visible': 2, 'scene_reset': 2}; {'grasp_unreachable': 4, 'too_few_visible': 2, 'scene_reset': 2}) |
| bedroom_2 | walls_rscyjh_0 | r00 | (6.47, -2.76) -> (6.47, -3.90) | (-1.00, 0.00) | 1.39 | 0.6 | 5 | 24 / 13 | yes |
| bedroom_2 | walls_vastci_0 | r01 | (2.46, -1.54) -> (2.46, -1.40) | (1.00, 0.00) | 0.39 | 0.6 | 2 | 11 / 6 | yes |
| bedroom_2 | walls_wjblrc_0 | r02 | (3.13, -0.49) -> (3.13, -0.49) | (0.00, -1.00) | 0.25 | 0.8 | 1 | 2 / 2 | yes |
| living_room_0 | walls_lempwq_0 | r00 | (-5.81, -0.49) -> (-2.93, -0.49) | (0.00, -1.00) | 3.12 | 0.6 | 13 | 73 / 37 | yes |
| living_room_0 | walls_ynhwyn_0 | r01 | (0.81, 1.15) -> (1.72, 1.15) | (0.00, -1.00) | 1.15 | 1.0 | 4 | 19 / 10 | yes |

Rooms with a validated region: bathroom_0, bedroom_0, bedroom_2, living_room_0. Map: `maps/house_double_floor_upper_walls.png`.

## Rs_int

| room | wall object | region | p0 -> p1 (x, y) | normal | length m | max patch m | spots | bases / reach | instance |
|---|---|---|---|---|---|---|---|---|---|
| bathroom_0 | walls_axhynp_0, walls_exiopd_0, walls_gxvfxw_0, walls_xlgydu_0 | - | - | - | - | - | 18 spots, 0 flat + clear, 0 viable | - | no viable region |
| bedroom_0 | walls_exiopd_0, walls_gxvfxw_0, walls_gxynpx_0, walls_xlgydu_0 | - | - | - | - | - | 39 spots, 1 flat + clear, 0 viable | - | no viable region |
| entryway_0 | walls_bcjmdz_0 | r00 | (1.87, 2.77) -> (1.87, 2.28) | (-1.00, 0.00) | 0.74 | 1.2 | 3 | 6 / 6 | yes |
| kitchen_0 | walls_exiopd_0, walls_gxvfxw_0 | - | - | - | - | - | 16 spots, 0 flat + clear, 0 viable | - | no viable region |
| living_room_0 | walls_bcjmdz_0 | r00 | (1.87, -2.76) -> (1.87, -3.11) | (-1.00, 0.00) | 0.59 | 0.6 | 2 | 10 / 7 | no ({'scene_reset': 32, 'grasp_unreachable': 5, 'scene_disturbed': 1}; {'scene_reset': 32, 'grasp_unreachable': 5, 'scene_disturbed': 1}) |
| living_room_0 | walls_bcjmdz_0 | r01 | (1.87, 1.12) -> (1.87, 1.12) | (-1.00, 0.00) | 0.25 | 0.6 | 1 | 1 / 1 | no ({'scene_reset': 32, 'grasp_unreachable': 5, 'scene_disturbed': 1}) |

Rooms with a validated region: entryway_0. Map: `maps/Rs_int_walls.png`.

## hotel_suite_large

| room | wall object | region | p0 -> p1 (x, y) | normal | length m | max patch m | spots | bases / reach | instance |
|---|---|---|---|---|---|---|---|---|---|
| bathroom_0 | walls_aisjjz_0 | r00 | (-1.78, 5.29) -> (-1.78, 5.29) | (-1.00, 0.00) | 0.25 | 0.6 | 1 | 12 / 3 | yes |
| bathroom_0 | walls_gjtspo_0 | r01 | (1.34, 4.94) -> (1.34, 4.94) | (-0.01, -1.00) | 0.25 | 0.8 | 1 | 12 / 9 | yes |
| bathroom_0 | walls_sdrwoa_0 | r02 | (-3.22, 3.14) -> (-3.22, 5.12) | (1.00, 0.00) | 2.23 | 1.0 | 9 | 69 / 46 | no ({'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 3, 'too_few_visible': 3}) |
| bathroom_0 | walls_znbklu_0 | r03 | (-1.69, 2.58) -> (-2.71, 2.58) | (0.00, 1.00) | 1.27 | 0.8 | 5 | 37 / 32 | yes |
| bedroom_0 | walls_irxipd_0 | r00 | (2.24, -2.64) -> (2.24, -3.85) | (-1.00, 0.00) | 1.46 | 0.6 | 6 | 40 / 31 | yes |
| bedroom_0 | walls_irxipd_0 | r01 | (2.24, 0.41) -> (2.24, -0.46) | (-1.00, 0.00) | 1.12 | 0.6 | 4 | 13 / 13 | yes |
| bedroom_0 | walls_xftoqy_0 | r02 | (-2.30, 1.22) -> (-2.30, 1.91) | (1.00, 0.00) | 0.94 | 0.6 | 4 | 18 / 16 | yes |
| bedroom_0 | walls_xftoqy_0 | r03 | (-2.30, -4.20) -> (-2.30, -3.69) | (1.00, 0.00) | 0.75 | 0.6 | 3 | 4 / 4 | yes |
| bedroom_0 | walls_xftoqy_0 | r04 | (-2.30, 0.79) -> (-2.30, 0.79) | (1.00, 0.00) | 0.25 | 0.6 | 1 | 2 / 1 | no ({'too_few_visible': 4, 'scene_reset': 5, 'grasp_unreachable': 2}) |

Rooms with a validated region: bathroom_0, bedroom_0. Map: `maps/hotel_suite_large_walls.png`.

## office_cubicles_right

| room | wall object | region | p0 -> p1 (x, y) | normal | length m | max patch m | spots | bases / reach | instance |
|---|---|---|---|---|---|---|---|---|---|
| bathroom_0 | walls_ajkcwv_0 | r00 | (17.07, 5.39) -> (17.07, 6.17) | (1.00, 0.00) | 1.03 | 0.8 | 4 | 32 / 21 | yes |
| bathroom_0 | walls_ajkcwv_0 | r01 | (16.92, 6.12) -> (16.92, 5.47) | (-1.00, -0.00) | 0.91 | 1.0 | 4 | 34 / 24 | yes |
| bathroom_0 | walls_hvqsio_0 | r02 | (14.17, 6.00) -> (14.17, 6.52) | (1.00, 0.00) | 0.77 | 1.0 | 4 | 33 / 21 | yes |
| bathroom_0 | walls_hvqsio_0 | r03 | (14.17, 5.48) -> (14.17, 5.53) | (1.00, 0.00) | 0.31 | 1.0 | 2 | 12 / 8 | yes |
| bathroom_0 | walls_rwfind_0 | r04 | (14.61, 4.79) -> (14.61, 4.79) | (0.00, 1.00) | 0.25 | 0.6 | 1 | 1 / 1 | yes |
| bathroom_0 | walls_ybneja_0 | r05 | (19.00, 9.70) -> (19.00, 9.70) | (0.00, -1.00) | 0.25 | 1.0 | 1 | 1 / 1 | yes |
| conference_hall_0 | walls_bpehln_0 | r00 | (-4.55, 10.87) -> (-4.55, 13.75) | (1.00, 0.00) | 3.13 | 1.2 | 13 | 114 / 86 | yes |
| conference_hall_0 | walls_bpehln_0 | r01 | (-4.55, 14.23) -> (-4.55, 14.95) | (1.00, 0.00) | 0.98 | 1.2 | 4 | 29 / 21 | not tested |
| conference_hall_0 | walls_uvdsip_0 | r02 | (-4.10, 15.91) -> (-4.10, 16.02) | (1.00, 0.00) | 0.36 | 0.6 | 2 | 24 / 18 | not tested |
| conference_hall_0 | walls_vkjkbk_0 | r03 | (0.46, 17.16) -> (2.76, 17.16) | (0.00, -1.00) | 2.55 | 0.8 | 11 | 85 / 64 | yes |
| conference_hall_0 | walls_vkjkbk_0 | r04 | (-3.54, 17.16) -> (-1.36, 17.16) | (0.00, -1.00) | 2.43 | 1.0 | 10 | 75 / 57 | no ({'too_few_visible': 4, 'grasp_unreachable': 2}; {'too_few_visible': 4, 'grasp_unreachable': 2}; {'too_few_visible': 4, 'grasp_unreachable': 2}) |
| conference_hall_0 | walls_vkjkbk_0 | r05 | (-0.49, 17.05) -> (-0.49, 17.05) | (0.00, -1.00) | 0.25 | 0.6 | 1 | 12 / 8 | not tested |
| conference_hall_0 | walls_wlvcmv_0 | r06 | (3.28, 16.18) -> (3.28, 15.88) | (-1.00, 0.00) | 0.55 | 0.6 | 2 | 24 / 10 | not tested |
| conference_hall_0 | walls_wxbrqm_0 | r07 | (3.73, 12.48) -> (3.73, 10.71) | (-1.00, 0.00) | 2.02 | 1.0 | 8 | 63 / 46 | not tested |
| copy_room_0 | walls_iexwiy_0, walls_nmbhlo_0, walls_plhcyg_0, walls_shkrsf_0, walls_tuemkh_0, walls_vjendd_0 | - | - | - | - | - | 50 spots, 30 flat + clear, 0 viable | - | no viable region |
| corridor_0 | walls_ajqvoa_0 | r00 | (-2.90, -8.09) -> (-2.90, -8.09) | (1.00, 0.00) | 1.25 | 1.2 | 1 | 6 / 3 | not tested |
| corridor_0 | walls_dqybhv_0 | r01 | (12.83, -13.60) -> (5.33, -13.60) | (0.00, 1.00) | 8.75 | 0.6 | 7 | 42 / 21 | not tested |
| corridor_0 | walls_gcfmvy_0 | r02 | (1.74, -11.20) -> (6.54, -11.20) | (0.00, -1.00) | 6.05 | 1.2 | 5 | 30 / 15 | not tested |
| corridor_0 | walls_gkewle_0 | r03 | (-2.90, -1.29) -> (-2.90, 7.41) | (1.00, 0.00) | 9.95 | 1.2 | 8 | 48 / 24 | yes |
| corridor_0 | walls_gqvjdy_0 | r04 | (6.10, 8.20) -> (1.00, 8.20) | (-0.00, 1.00) | 6.35 | 1.2 | 5 | 30 / 15 | not tested |
| corridor_0 | walls_iexwiy_0 | r05 | (9.45, -11.20) -> (10.65, -11.20) | (0.00, -1.00) | 2.45 | 1.2 | 2 | 12 / 6 | not tested |
| corridor_0 | walls_kkyrqq_0 | r06 | (0.30, -4.83) -> (0.30, -4.83) | (-1.00, 0.00) | 1.25 | 0.6 | 1 | 6 / 3 | not tested |
| corridor_0 | walls_kzncyd_0 | r07 | (-2.90, -5.25) -> (-2.90, -5.25) | (1.00, 0.00) | 1.25 | 1.2 | 1 | 6 / 3 | not tested |
| corridor_0 | walls_llrlhg_0 | r08 | (13.85, 6.62) -> (13.85, -2.16) | (-1.00, 0.00) | 10.03 | 1.2 | 8 | 48 / 24 | no ({'init_false': 3, 'grasp_unreachable': 3}; {'init_false': 3, 'grasp_unreachable': 3}; {'init_false': 3, 'grasp_unreachable': 3}; {'init_false': 3, 'grasp_unreachable': 3}) |
| corridor_0 | walls_llrlhg_0 | r09 | (13.85, 9.24) -> (13.85, 9.24) | (-1.00, 0.00) | 1.25 | 1.2 | 1 | 6 / 3 | not tested |
| corridor_0 | walls_rlchwi_0 | r10 | (0.30, 7.31) -> (0.30, -1.39) | (-1.00, 0.00) | 9.95 | 1.2 | 8 | 48 / 24 | yes |
| corridor_0 | walls_rlygpv_0 | r11 | (11.85, 5.82) -> (11.85, 5.82) | (1.00, 0.00) | 1.25 | 0.6 | 1 | 6 / 3 | not tested |
| corridor_0 | walls_tqjsqg_0 | r12 | (-4.75, 9.50) -> (-4.75, 9.50) | (1.00, 0.00) | 1.25 | 1.0 | 1 | 2 / 1 | not tested |
| corridor_0 | walls_tsqvif_0 | r13 | (9.43, 10.01) -> (13.00, 10.01) | (0.00, -1.00) | 4.82 | 1.2 | 4 | 24 / 12 | not tested |
| corridor_0 | walls_vsmckp_0 | r14 | (0.30, -8.48) -> (0.30, -8.48) | (-1.00, 0.00) | 1.25 | 0.6 | 1 | 6 / 2 | not tested |
| corridor_0 | walls_wcngqp_0 | r15 | (-2.87, -12.69) -> (-2.87, -12.69) | (1.00, 0.00) | 1.25 | 1.2 | 1 | 6 / 3 | not tested |
| corridor_0 | walls_wcngqp_0 | r16 | (-2.87, -10.46) -> (-2.87, -10.46) | (1.00, 0.00) | 1.25 | 0.6 | 1 | 6 / 3 | not tested |
| corridor_0 | walls_xdmmwd_0 | r17 | (10.30, 8.20) -> (7.90, 8.20) | (0.00, 1.00) | 3.65 | 1.2 | 3 | 18 / 9 | not tested |
| corridor_0 | walls_zkpumg_0 | r18 | (13.85, -6.88) -> (13.85, -6.88) | (-1.00, 0.00) | 1.25 | 1.2 | 1 | 6 / 3 | not tested |
| empty_room_0 | walls_isrini_0, walls_kydyel_0, walls_ptxwgq_0, walls_rlygpv_0, walls_swfzhs_0, walls_wzpmnh_0, walls_xdmmwd_0 | - | - | - | - | - | 167 spots, 34 flat + clear, 0 viable | - | no viable region |
| lobby_0 | walls_dzwthn_0 | r00 | (3.53, -17.20) -> (0.29, -17.20) | (0.00, 1.00) | 3.48 | 1.2 | 14 | 72 / 36 | no ({'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 2, 'too_few_visible': 4}; {'grasp_unreachable': 4, 'too_few_visible': 2}; {'grasp_unreachable': 5, 'too_few_visible': 1}) |
| lobby_0 | walls_gqtmou_0 | r01 | (4.57, -13.90) -> (4.57, -14.07) | (-1.00, 0.00) | 0.42 | 0.6 | 2 | 10 / 6 | yes |
| lobby_0 | walls_wcngqp_0 | r02 | (-2.87, -14.03) -> (-2.87, -13.73) | (1.00, 0.00) | 0.55 | 1.2 | 3 | 18 / 9 | yes |
| lobby_0 | walls_wcngqp_0 | r03 | (-2.87, -14.63) -> (-2.87, -14.46) | (1.00, 0.00) | 0.43 | 0.6 | 2 | 9 / 4 | no ({'grasp_unreachable': 3, 'too_few_visible': 3}) |
| meeting_room_0 | walls_dmcumz_0 | r00 | (4.90, 17.16) -> (7.04, 17.16) | (0.00, -1.00) | 2.39 | 1.0 | 10 | 57 / 29 | yes |
| meeting_room_0 | walls_tuiizs_0 | r01 | (8.15, 16.45) -> (8.15, 15.19) | (-1.00, 0.00) | 1.51 | 1.0 | 6 | 31 / 17 | no ({'too_few_visible': 4, 'grasp_unreachable': 2}; {'too_few_visible': 5, 'grasp_unreachable': 1}) |
| meeting_room_0 | walls_tuiizs_0 | r02 | (8.15, 11.83) -> (8.15, 10.69) | (-1.00, 0.00) | 1.39 | 1.0 | 6 | 24 / 14 | yes |
| meeting_room_0 | walls_wlvcmv_0 | r03 | (4.30, 15.91) -> (4.30, 16.02) | (1.00, 0.00) | 0.36 | 0.6 | 2 | 12 / 6 | not tested |
| meeting_room_0 | walls_wxbrqm_0 | r04 | (3.85, 10.87) -> (3.85, 12.85) | (1.00, 0.00) | 2.23 | 1.2 | 9 | 42 / 21 | yes |
| meeting_room_0 | walls_wxbrqm_0 | r05 | (3.85, 13.45) -> (3.85, 14.53) | (1.00, 0.00) | 1.32 | 1.2 | 6 | 16 / 9 | not tested |
| private_office_0 | walls_dcyrxu_0, walls_kydyel_0, walls_ptxwgq_0, walls_rlygpv_0, walls_wlwppv_0 | - | - | - | - | - | 43 spots, 30 flat + clear, 0 viable | - | no viable region |
| private_office_1 | walls_cmjgnh_0 | r00 | (17.12, -3.00) -> (18.62, -3.00) | (0.00, -1.00) | 1.75 | 1.2 | 7 | 34 / 18 | no ({'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 4, 'too_few_visible': 2}) |
| private_office_1 | walls_cmjgnh_0 | r01 | (14.46, -3.00) -> (15.14, -3.00) | (0.00, -1.00) | 0.92 | 0.8 | 4 | 10 / 7 | yes |
| private_office_1 | walls_fldlqu_0 | r02 | (15.26, -6.59) -> (15.15, -6.59) | (0.00, 1.00) | 0.36 | 0.6 | 2 | 8 / 4 | yes |
| private_office_1 | walls_vaphll_0 | r03 | (19.36, -3.87) -> (19.36, -5.52) | (-1.00, 0.00) | 1.90 | 1.0 | 8 | 48 / 24 | no ({'grasp_unreachable': 2, 'too_few_visible': 4}; {'grasp_unreachable': 3, 'too_few_visible': 3}) |
| private_office_1 | walls_zplhuj_0 | r04 | (18.27, -6.25) -> (18.02, -6.25) | (0.00, 1.00) | 0.49 | 0.6 | 2 | 12 / 6 | yes |
| private_office_2 | walls_fldlqu_0 | r00 | (15.14, -6.89) -> (15.14, -6.89) | (0.00, -1.00) | 0.25 | 0.6 | 1 | 6 / 3 | not tested |
| private_office_2 | walls_iuufle_0 | r01 | (19.36, -7.94) -> (19.36, -9.82) | (-1.00, 0.00) | 2.13 | 0.6 | 9 | 53 / 27 | no ({'scene_reset': 1, 'too_few_visible': 3, 'grasp_unreachable': 3}; {'too_few_visible': 2, 'grasp_unreachable': 4}; {'too_few_visible': 3, 'grasp_unreachable': 3}; {'too_few_visible': 3, 'grasp_unreachable': 3}) |
| private_office_2 | walls_mrghjk_0 | r02 | (18.87, -10.61) -> (17.07, -10.61) | (0.00, 1.00) | 2.05 | 0.6 | 8 | 39 / 20 | no ({'too_few_visible': 3, 'grasp_unreachable': 3}; {'too_few_visible': 2, 'grasp_unreachable': 4}; {'too_few_visible': 3, 'grasp_unreachable': 3}; {'too_few_visible': 4, 'grasp_unreachable': 2}) |
| private_office_2 | walls_mrghjk_0 | r03 | (15.27, -10.61) -> (14.97, -10.61) | (0.00, 1.00) | 0.55 | 1.2 | 3 | 11 / 7 | yes |
| private_office_2 | walls_mrghjk_0 | r04 | (14.56, -10.61) -> (14.56, -10.61) | (0.00, 1.00) | 0.25 | 1.0 | 1 | 3 / 2 | not tested |
| private_office_2 | walls_zplhuj_0 | r05 | (18.12, -7.25) -> (18.25, -7.25) | (0.00, -1.00) | 0.38 | 0.8 | 2 | 12 / 5 | yes |
| private_office_3 | walls_mrghjk_0 | r00 | (17.22, -10.71) -> (18.70, -10.71) | (0.00, -1.00) | 1.72 | 1.0 | 7 | 30 / 18 | yes |
| private_office_3 | walls_mrghjk_0 | r01 | (14.50, -10.71) -> (15.10, -10.71) | (0.00, -1.00) | 0.85 | 1.0 | 3 | 9 / 6 | yes |
| private_office_3 | walls_nbtgcz_0 | r02 | (13.95, -14.05) -> (13.95, -13.75) | (1.00, 0.00) | 0.55 | 0.6 | 3 | 18 / 9 | yes |
| private_office_3 | walls_nbtgcz_0 | r03 | (13.95, -14.46) -> (13.95, -14.46) | (1.00, 0.00) | 0.25 | 1.0 | 1 | 2 / 2 | not tested |
| private_office_3 | walls_ocyncu_0 | r04 | (16.84, -15.04) -> (14.74, -15.04) | (0.00, 1.00) | 2.35 | 1.2 | 10 | 57 / 30 | yes |
| private_office_3 | walls_qbdext_0 | r05 | (17.47, -13.88) -> (17.47, -14.26) | (-1.00, 0.00) | 0.64 | 0.6 | 3 | 18 / 9 | yes |
| private_office_3 | walls_woavqu_0 | r06 | (19.36, -11.98) -> (19.36, -12.02) | (-1.00, 0.00) | 0.29 | 1.0 | 2 | 10 / 6 | no ({'grasp_unreachable': 3, 'too_few_visible': 3}; {'grasp_unreachable': 3, 'too_few_visible': 3}) |
| private_office_3 | walls_woavqu_0 | r07 | (19.25, -13.01) -> (19.25, -13.01) | (-1.00, 0.00) | 0.25 | 0.6 | 1 | 2 / 1 | not tested |
| private_office_3 | walls_ymuaax_0 | r08 | (18.64, -13.54) -> (18.64, -13.54) | (0.00, 1.00) | 0.25 | 0.6 | 1 | 5 / 3 | not tested |
| shared_office_0 | walls_dcyrxu_0, walls_gxjccg_0, walls_plhcyg_0, walls_tuemkh_0, walls_wlwppv_0 | - | - | - | - | - | 47 spots, 17 flat + clear, 0 viable | - | no viable region |

Rooms with a validated region: bathroom_0, conference_hall_0, corridor_0, lobby_0, meeting_room_0, private_office_1, private_office_2, private_office_3. Map: `maps/office_cubicles_right_walls.png`.

## restaurant_diner

| room | wall object | region | p0 -> p1 (x, y) | normal | length m | max patch m | spots | bases / reach | instance |
|---|---|---|---|---|---|---|---|---|---|
| bar_0 | walls_ssudnh_0 | r00 | (-4.79, -2.77) -> (-4.79, -1.59) | (1.00, 0.00) | 1.43 | 0.4 | 6 | 33 / 17 | yes |
| bathroom_0 | walls_hcdfmx_0 | r00 | (-5.15, 5.76) -> (-5.15, 5.76) | (1.00, -0.00) | 0.25 | 0.6 | 1 | 1 / 1 | no ({'too_few_visible': 4, 'grasp_unreachable': 2}) |
| bathroom_0 | walls_ztyvde_0 | r01 | (-2.15, 5.70) -> (-2.15, 5.70) | (-1.00, 0.00) | 0.25 | 0.8 | 1 | 6 / 3 | yes |
| corridor_0 | walls_vnnkzu_0 | r00 | (-3.95, 4.25) -> (-3.95, 4.25) | (1.00, 0.00) | 0.25 | 0.6 | 1 | 1 / 1 | yes |
| dining_room_0 | walls_fjxunn_0 | r00 | (-4.79, -8.35) -> (-4.79, -5.88) | (1.00, 0.00) | 2.72 | 1.2 | 11 | 64 / 32 | no ({'too_few_visible': 3, 'grasp_unreachable': 3}; {'too_few_visible': 3, 'grasp_unreachable': 3}; {'too_few_visible': 3, 'grasp_unreachable': 3}; {'too_few_visible': 3, 'grasp_unreachable': 3}) |
| dining_room_0 | walls_xtujiy_0 | r01 | (-4.16, 3.45) -> (-4.16, 3.45) | (1.00, 0.00) | 0.25 | 0.6 | 1 | 1 / 1 | yes |
| kitchen_0 | walls_bftgiq_0, walls_hbnjvp_0, walls_jmnoau_0, walls_opceer_0 | - | - | - | - | - | 105 spots, 52 flat + clear, 0 viable | - | no viable region |

Rooms with a validated region: bar_0. Map: `maps/restaurant_diner_walls.png`.

