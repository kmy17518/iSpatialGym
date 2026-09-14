# Articulated objects: handle inference, review and opening tests

100 models, 159 openable joints. Heuristic handle status: edge 25, handle 89, ungraspable 45.
Review verdicts (chosen marker): F 15, none 61, A 63, B 10, C 10; graspable=false: 57.

## Opening tests

157 joints tested so far. Outcome: asset_issue 4, collision 4, grasp_no_ag 3, grasped_not_opened 4, infeasible_2_reviews 33, model_error 26, no_handle 1, opened 13, other 3, planning 2, unreachable 64.

| category | model | joint | kind | setting | handle (review) | height m | reachable | trunk | AG | dq / target | opened | shifts | result / error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blender | cwkvib | j_lid | flap | table/floor | F edge | 1.107 |  | 0.5 | False | 0.0 / 0.524 | False |  | grasp_no_ag: grasp: [grasp] o_blender__cwkvib not in hand after closing gripper {'in_hand': None} |
| blender | eyedvd | j_lid | flap | table/floor | none edge | 1.079 |  | 0.5 | False | 0.0 / 0.785 | False |  | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| bottom_cabinet | bamfsz | j_link_1 | drawer | scene:Rs_int | A horizontal_bar | 0.562 | False |  | False | 0.0 / 0.15 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | bamfsz | j_link_2 | drawer | scene:Rs_int | A horizontal_bar | 0.884 | False |  | False | 0.0 / 0.15 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | bamfsz | j_link_3 | drawer | scene:Rs_int | B horizontal_bar | 1.044 | False |  | False | 0.0 / 0.15 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | bamfsz | j_link_4 | drawer | scene:Rs_int | A horizontal_bar | 1.205 | False |  | False | 0.0 / 0.15 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | bycegi | j_link_0 | door | scene:Rs_int | C vertical_bar | 0.402 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | cjcyed | j_link_1 | drawer | scene:house_double_floor_upper | none none |  | None |  | False | 0.0 / None | False | 0 | no_handle: no graspable handle |
| bottom_cabinet | dajebq | j_link_0 | door | scene:Rs_int | C vertical_bar | 0.335 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | dajebq | j_link_1 | door | scene:Rs_int | B vertical_bar | 0.47 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | dajebq | j_link_2 | door | scene:Rs_int | C vertical_bar | 0.335 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | dajebq | j_link_3 | door | scene:Rs_int | B vertical_bar | 0.47 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | dsbcxl | j_link_1 | drawer | scene:Rs_int | A horizontal_bar | 0.366 | False |  | False | 0.0 / 0.15 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | dsbcxl | j_link_2 | drawer | scene:Rs_int | C horizontal_bar | 0.785 | True | 0.5 | False | 0.0 / 0.15 | False | 0 | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| bottom_cabinet | eupgpt | j_link_0 | door | scene:Rs_int | A vertical_bar | 0.358 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | eupgpt | j_link_1 | door | scene:Rs_int | C vertical_bar | 0.622 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | gcrjan | j_link_0 | door | scene:house_double_floor_upper | B vertical_bar | 0.339 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | gcrjan | j_link_1 | door | scene:house_double_floor_upper | A vertical_bar | 0.66 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | gcrjan | j_link_2 | door | scene:house_double_floor_upper | A vertical_bar | 0.271 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | gcrjan | j_link_3 | door | scene:house_double_floor_upper | A vertical_bar | 0.657 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | getjzo | j_link_1 | flap | scene:house_double_floor_upper | A horizontal_bar | 0.035 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | glefdh | j_link_0 | door | scene:house_single_floor | C vertical_bar | 1.186 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | glefdh | j_link_1 | door | scene:house_single_floor | C vertical_bar | 1.186 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | glefdh | j_link_2 | door | scene:house_single_floor | A vertical_bar | 1.797 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | glefdh | j_link_3 | door | scene:house_single_floor | A vertical_bar | 1.66 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | planning: planning: [planning] no collision-free base path to (8.914261817932129, -0.861605763435363 |
| bottom_cabinet | gvtucm | j_link_0 | drawer | scene:office_cubicles_right | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| bottom_cabinet | gvtucm | j_link_1 | drawer | scene:office_cubicles_right | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| bottom_cabinet | gvtucm | j_link_2 | drawer | scene:office_cubicles_right | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| bottom_cabinet | gvtucm | j_link_3 | drawer | scene:office_cubicles_right | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| bottom_cabinet | hnxyev | j_link_0 | flap | scene:house_double_floor_upper | F edge | 0.661 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | grasp_no_ag: grasp: [grasp] o_bottom_cabinet__hnxyev not in hand after closing gripper {'in_hand': None |
| bottom_cabinet | hnxyev | j_link_1 | drawer | scene:house_double_floor_upper | A knob | 0.171 | False |  | False | 0.0 / 0.15 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet | songes | j_bottomdrawer | drawer | scene:house_double_floor_upper | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| bottom_cabinet | songes | j_middledrawer | drawer | scene:house_double_floor_upper | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| bottom_cabinet | songes | j_topdrawer | drawer | scene:house_double_floor_upper | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| bottom_cabinet_no_top | apjbrw | j_link_0 | door | scene:house_double_floor_upper | A horizontal_bar | 0.679 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | bmsclc | j_link_0 | door | scene:house_double_floor_upper | A horizontal_bar | 0.657 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | exgntg | j_link_0 | door | scene:house_single_floor | A vertical_bar | 0.454 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | exgntg | j_link_1 | door | scene:house_single_floor | A vertical_bar | 0.46 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | exgntg | j_link_2 | drawer | scene:house_single_floor | A horizontal_bar | 0.84 | True | 0.25 | False | 0.0 / 0.13 | False | 0 | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| bottom_cabinet_no_top | fmelap | j_link_0 | door | scene:house_single_floor | A horizontal_bar | 0.649 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | fmelap | j_link_1 | door | scene:house_single_floor | A horizontal_bar | 0.649 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | motill | j_link_0 | door | scene:house_double_floor_upper | A knob | 0.666 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | ozzcvs | j_link_0 | door | scene:house_single_floor | A horizontal_bar |  |  |  |  |  /  |  |  | model_error: ValueError("'/World/scene_0/o_bottom_cabinet_no_top__fmelap/joints/j_link_0' is not in lis |
| bottom_cabinet_no_top | ozzcvs | j_link_1 | door | scene:house_single_floor | A horizontal_bar |  |  |  |  |  /  |  |  | model_error: ValueError("'/World/scene_0/o_bottom_cabinet_no_top__fmelap/joints/j_link_0' is not in lis |
| bottom_cabinet_no_top | ozzcvs | j_link_2 | drawer | scene:house_single_floor | A horizontal_bar |  |  |  |  |  /  |  |  | model_error: ValueError("'/World/scene_0/o_bottom_cabinet_no_top__fmelap/joints/j_link_0' is not in lis |
| bottom_cabinet_no_top | ozzcvs | j_link_3 | drawer | scene:house_single_floor | A horizontal_bar |  |  |  |  |  /  |  |  | model_error: ValueError("'/World/scene_0/o_bottom_cabinet_no_top__fmelap/joints/j_link_0' is not in lis |
| bottom_cabinet_no_top | pluwfl | j_link_0 | door | scene:house_single_floor | A vertical_bar | 0.547 | True | 0.5 | True | 0.4495 / 0.785 | True | 0 | opened: None |
| bottom_cabinet_no_top | pluwfl | j_link_1 | drawer | scene:house_single_floor | A horizontal_bar | 0.784 | True | 0.5 | True | 0.1434 / 0.15 | True | 0 | opened: None |
| bottom_cabinet_no_top | qohxjq | j_link_0 | drawer | scene:house_double_floor_upper | none none | 0.205 | False |  | False | 0.0 / 0.15 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | qohxjq | j_link_1 | drawer | scene:house_double_floor_upper | none none | 0.435 | False |  | False | 0.0 / 0.15 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | qohxjq | j_link_2 | drawer | scene:house_double_floor_upper | F edge | 0.651 | False |  | False | 0.0 / 0.15 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | qudfwe | j_link_0 | door | scene:house_single_floor | A knob | 0.621 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | qudfwe | j_link_1 | door | scene:house_single_floor | A knob | 0.622 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | qudfwe | j_link_2 | drawer | scene:house_single_floor | A horizontal_bar | 0.785 | True | 0.5 | False | 0.0 / 0.15 | False | 0 | other: execution: [execution] joints did not converge (max err 0.158 rad) |
| bottom_cabinet_no_top | qudfwe | j_link_3 | drawer | scene:house_single_floor | A horizontal_bar | 0.784 | True | 0.5 | True | 0.1431 / 0.15 | True | 0 | opened: grasp: [grasp] o_bottom_cabinet_no_top__qudfwe not in hand after closing gripper {'in_hand |
| bottom_cabinet_no_top | spojpj | j_link_0 | door | scene:house_single_floor | A vertical_bar | 0.674 | True | 0.5 | True | 0.2765 / 0.785 | False | 0 | grasped_not_opened: None |
| bottom_cabinet_no_top | spojpj | j_link_1 | door | scene:house_single_floor | A vertical_bar | 0.674 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| bottom_cabinet_no_top | ufhpbn | j_link_0 | drawer | scene:house_double_floor_upper | A horizontal_bar | 0.666 | False |  | False | 0.0 / 0.15 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| bottom_cabinet_no_top | vdedzt | j_link_0 | door | scene:house_single_floor | A vertical_bar | 0.738 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | collision: collision: [collision] trunk motion would collide (sample 1) {'translate': 0.5, 'tilt': No |
| bottom_cabinet_no_top | vdedzt | j_link_1 | door | scene:house_single_floor | A vertical_bar | 0.734 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | collision: collision: [collision] trunk motion would collide (sample 1) {'translate': 0.5, 'tilt': No |
| carton | causya | j_link_0 | flap | table/floor | none none |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| carton | causya | j_link_2 | flap | table/floor | F edge | 0.673 |  | 0.5 | False | 0.0 / 0.785 | False |  | other: execution: [execution] base tracking timed out (pos err 0.631 m, yaw err 1.704 rad) |
| carton | cdmmwy | j_left_lid | flap | table/floor | none none |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| carton | cdmmwy | j_right_lid | flap | table/floor | none none |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| carton | hhlmbi | j_link0 | flap | table/floor | none none |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| carton | libote | j_link0 | flap | table/floor | none none |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| carton | msfzpz | j_link_0 | flap | table/floor | F edge | 0.669 |  | 0.5 | True | 0.8016 / 0.785 | True |  | opened: None |
| carton | ylrxhe | j_link0 | flap | table/floor | none none |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| carton | ylrxhe | j_link1 | flap | table/floor | none none |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| cedar_chest | fwstpx | j_up | flap | table/floor | none none |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| cedar_chest | gbdzls | j_up | flap | table/floor | none none |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| clothes_dryer | fhtvuq | j_link_0 | door | scene:house_single_floor | none none | 0.445 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| clothes_dryer | smcyys | j_dof_rootd_aa001_r | door | scene:house_single_floor | none none | 0.219 | False |  | False | 0.0 / 0.725 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| clothes_dryer | wivnic | j_dof_rootd_aa001_r | door | scene:house_single_floor | none none | 0.651 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| clothes_dryer | xsuyua | j_link_0 | door | scene:house_single_floor | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| clothes_dryer | zlmnfg | j_link_0 | door | scene:house_single_floor | none none | 0.77 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | grasp_no_ag: grasp: [grasp] o_clothes_dryer__zlmnfg not in hand after closing gripper {'in_hand': None} |
| cooler | pnkiag | j_lid | flap | table/floor | none none | 1.258 |  | 0.25 | False | 0.0 / 0.785 | False |  | other: execution: [execution] joints did not converge (max err 0.168 rad) |
| dishwasher | cgugie | j_link_0 | flap | scene:house_single_floor | A horizontal_bar | 0.577 | True | 0.5 | True | 0.0001 / 0.785 | False | 0 | grasped_not_opened: None |
| dishwasher | dngvvi | j_link_0 | flap | scene:house_double_floor_lower | F edge | 0.783 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| dishwasher | dugpby | j_link_0 | flap | scene:house_double_floor_lower | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| dishwasher | tnlzww | j_link_4 | flap | scene:house_single_floor | A horizontal_bar | 0.708 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | collision: collision: [collision] trunk motion would collide (sample 1) {'translate': 0.5, 'tilt': No |
| dishwasher | xlmier | j_link_1 | flap | scene:Rs_int | A horizontal_bar | 1.21 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| dishwasher | ynrhuk | j_link_2 | flap | scene:house_single_floor | A horizontal_bar | 0.757 | True | 0.5 | True | 0.3768 / 0.785 | False | 0 | grasped_not_opened: None |
| dishwasher | znfvmj | j_link_2 | flap | scene:house_double_floor_lower | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| door | adejun | j_left_leaf | door | scene:restaurant_diner | none none |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [7.95, 0.1, 0.81]') |
| door | adejun | j_right_leaf | door | scene:restaurant_diner | none none |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [7.95, 0.1, 0.81]') |
| door | aleker | j_door1 | door | scene:restaurant_diner | none vertical_bar |  | None |  | False | 0.0 / None | False | 0 | asset_issue: no graspable handle |
| door | aleker | j_door2 | door | scene:restaurant_diner | none vertical_bar |  | None |  | False | 0.0 / None | False | 0 | asset_issue: no graspable handle |
| door | aleker | j_door3 | door | scene:restaurant_diner | none vertical_bar |  | None |  | False | 0.0 / None | False | 0 | asset_issue: no graspable handle |
| door | aleker | j_door4 | door | scene:restaurant_diner | none vertical_bar |  | None |  | False | 0.0 / None | False | 0 | asset_issue: no graspable handle |
| door | bandpe | j_leaf0 | door | scene:house_double_floor_lower | none none |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [10.95, 0.11, 1.13]') |
| door | bandpe | j_leaf1 | door | scene:house_double_floor_lower | none none |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [10.95, 0.11, 1.13]') |
| door | bexenl | j_leaf | door | scene:house_double_floor_lower | A lever | 1.091 | True | 0.5 | True | 0.7768 / 0.785 | True | 0 | opened: None |
| door | cbgtjl | j_leaf | door | scene:restaurant_diner | A lever | 0.958 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| door | ceddpg | j_link_1 | door | scene:Rs_int | A knob | 0.839 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | planning: planning: [planning] no collision-free base path to (-3.03570294380188, 1.0386492013931274 |
| door | chonxi | j_link_0 | door | scene:house_double_floor_lower | none none |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [10.74, 0.16, 1.11]') |
| door | crtrpa | j_leaf | door | scene:house_single_floor | B vertical_bar | 1.031 | True | 0.0 | True | 0.2997 / 0.785 | False | 0 | grasped_not_opened: None |
| door | cvvumy | j_left_leaf | door | scene:restaurant_diner | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| door | cvvumy | j_right_leaf | door | scene:restaurant_diner | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| door | dlncrt | j_leaf | door | scene:house_double_floor_upper | A knob |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [11.64, 0.16, 1.08]') |
| door | dqnbsj | j_leaf | door | scene:restaurant_diner | none vertical_bar |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [2.9, 0.14, 0.79]') |
| door | duymuw | j_leaf | door | scene:house_double_floor_lower | A lever |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [9.29, 0.16, 1.09]') |
| fridge | dszchb | j_link_0 | door | scene:house_single_floor | A vertical_bar | 1.117 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| fridge | dxwbae | j_left_leaf | door | scene:restaurant_diner | A vertical_bar | 1.015 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| fridge | dxwbae | j_right_leaf | door | scene:restaurant_diner | A vertical_bar | 1.015 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| fridge | hivvdf | j_link_0 | door | scene:Rs_int | C vertical_bar | 0.982 | True | 0.25 | False | 0.0 / 0.785 | False | 0 | unreachable: planning: [planning] no collision-free base path to (-0.6062722206115723, 2.22174692153930 |
| fridge | hzgqdn | j_link_0 | door | scene:Rs_int | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| fridge | jtqazu | j_link_0 | door | scene:restaurant_diner | A vertical_bar | 1.093 | True | 0.5 | True | 0.7855 / 0.785 | True | 0 | opened: None |
| fridge | juwaoh | j_link_0 | door | scene:restaurant_diner | F edge | 0.926 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| fridge | lleghp | j_shelf0 | drawer | scene:house_single_floor | none none |  |  |  |  |  /  |  |  | model_error: AssertionError('Some matrices are singular and cannot be decomposed') |
| fridge | lleghp | j_shelf1 | drawer | scene:house_single_floor | none none |  |  |  |  |  /  |  |  | model_error: AssertionError('Some matrices are singular and cannot be decomposed') |
| fridge | lleghp | j_shelf2 | drawer | scene:house_single_floor | none none |  |  |  |  |  /  |  |  | model_error: AssertionError('Some matrices are singular and cannot be decomposed') |
| fridge | lleghp | j_shelf3 | drawer | scene:house_single_floor | none none |  |  |  |  |  /  |  |  | model_error: AssertionError('Some matrices are singular and cannot be decomposed') |
| fridge | xyejdx | j_link_0 | door | scene:Rs_int | A vertical_bar | 1.335 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| fridge | xyejdx | j_link_1 | door | scene:Rs_int | A vertical_bar | 0.956 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| hinged_jar | gqtsam | j_cap | flap | table/floor | F edge | 0.938 |  | 0.5 | False | 0.0 / 0.785 | False |  | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| hinged_jar | jnjtrl | j_lid | flap | table/floor | none none |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| hinged_jar | vzwhbg | j_rb1 | flap | table/floor | F edge | 0.952 |  | 0.5 | True | 0.8016 / 0.785 | True |  | opened: grasp: [grasp] o_hinged_jar__vzwhbg not in hand after closing gripper {'in_hand': None} |
| laptop | nvulcs | j_link_0 | flap | table/floor | F edge | 0.766 |  |  | False | 0.0 / 0.785 | False |  | unreachable: unreachable: no stance |
| openable_window | cbudyv | j_leaf0 | door | scene:Rs_int | A vertical_bar | 1.624 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| openable_window | cbudyv | j_leaf1 | door | scene:Rs_int | B vertical_bar | 1.625 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| openable_window | cbudyv | j_leaf2 | door | scene:Rs_int | A vertical_bar | 1.623 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| openable_window | dpngol | j_left_leaf | drawer | scene:house_single_floor | none none |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [87.22, 0.06, 1.31]') |
| openable_window | dpngol | j_right_leaf | drawer | scene:house_single_floor | none none |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [87.22, 0.06, 1.31]') |
| openable_window | eyvzfu | j_movingpart | drawer | scene:house_single_floor | none none | 0.953 | True | 0.5 | False | 0.0 / 0.15 | False | 0 | collision: collision: [collision] trunk motion would collide (sample 1) {'translate': 0.5, 'tilt': No |
| openable_window | qdkcdn | j_dof_rootd_aa002_t | drawer | scene:Rs_int | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| oven | ffitak | j_door | flap | scene:house_single_floor | B horizontal_bar | 1.291 | True | 0.5 | False | 0.0 / 0.785 | False | 0 | unreachable: planning: [planning] move_eef(right) failed: IK failed: goal pose unreachable for the acti |
| oven | ffitak | j_rack1 | drawer | scene:house_single_floor | F horizontal_bar | 1.035 | False |  | False | 0.0 / 0.111 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| oven | ffitak | j_rack2 | drawer | scene:house_single_floor | F horizontal_bar | 0.924 | False |  | False | 0.0 / 0.111 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| oven | kfftgk | j_dof_rootd_aa001_r | flap | scene:house_double_floor_lower | B horizontal_bar | 0.698 | True | 0.5 | True | 0.4532 / 0.785 | True | 0 | opened: None |
| oven | leqtlc | j_dof_rootd_aa001_r | flap | scene:house_double_floor_lower | C horizontal_bar | 0.621 | True | 0.5 | True | 0.5963 / 0.785 | True | 0 | opened: None |
| oven | nkxhvf | j_link_4 | flap | scene:Rs_int | A horizontal_bar |  |  |  |  |  /  |  |  | model_error: AssertionError('Some matrices are singular and cannot be decomposed') |
| oven | rwuazb | j_dof_rootd_aa001_r | flap | scene:house_double_floor_lower | A horizontal_bar | 0.663 | True | 0.5 | True | 0.5354 / 0.785 | True | 0 | opened: planning: [planning] move_eef_cartesian(right) failed: differential IK did not converge at |
| recycling_bin | mdojox | j_link_2 | flap | table/floor | F edge |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| recycling_bin | nuoypc | j_link_2 | flap | table/floor | F edge |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| recycling_bin | pdmzhv | j_link_2 | flap | table/floor | F edge |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |
| sliding_door | hgcxno | j_left_leaf | drawer | scene:house_single_floor | none none |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [4.94, 0.01, 0.89]') |
| sliding_door | hgcxno | j_right_leaf | drawer | scene:house_single_floor | none none |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [4.94, 0.01, 0.89]') |
| sliding_door | lrvyvc | j_movingpart | drawer | scene:house_single_floor | none none |  |  |  |  |  /  |  |  | model_error: RuntimeError('swap skipped: host/test aspect mismatch, scale factors [0.18, 0.43, 0.86]') |
| sliding_door | tprpvb | j_movingpart | drawer | scene:house_single_floor | none edge | 1.155 | False |  | False | 0.0 / 0.15 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| toolbox | redmtl | j_lid | flap | table/floor | A horizontal_bar | 1.092 |  | 0.5 | True | 0.5949 / 0.785 | True |  | opened: None |
| top_cabinet | dhguym | j_link_1 | door | scene:Rs_int | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| top_cabinet | dmwxyl | j_link_0 | door | scene:house_double_floor_lower | A horizontal_bar | 1.49 | True | 0.5 | True | 0.7835 / 0.785 | True | 0 | opened: None |
| top_cabinet | dmwxyl | j_link_1 | door | scene:house_double_floor_lower | A horizontal_bar | 1.49 | True | 0.5 | True | 0.9705 / 0.785 | True | 0 | opened: None |
| top_cabinet | eobsmt | j_link_0 | door | scene:Rs_int | B vertical_bar |  |  |  |  |  /  |  |  | model_error: ValueError("'/World/scene_0/o_top_cabinet__dhguym/joints/j_link_1' is not in list") |
| top_cabinet | fqhdne | j_link_0 | door | scene:Rs_int | A vertical_bar |  |  |  |  |  /  |  |  | model_error: ValueError("'/World/scene_0/o_top_cabinet__dhguym/joints/j_link_1' is not in list") |
| top_cabinet | fqitbw | j_link_1 | flap | scene:office_cubicles_right | A knob | 1.98 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| top_cabinet | jvdbxh | j_link_0 | door | scene:Rs_int | A vertical_bar | 2.049 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| top_cabinet | lsyzkh | j_link_1 | door | scene:Rs_int | A vertical_bar |  |  |  |  |  /  |  |  | model_error: RuntimeError('Exception occurred during pre-physics step') |
| top_cabinet | nkrgez | j_link_0 | door | scene:house_double_floor_lower | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| top_cabinet | nkrgez | j_link_1 | door | scene:house_double_floor_lower | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| top_cabinet | oztrra | j_link_0 | flap | scene:Rs_int | A horizontal_bar |  |  |  |  |  /  |  |  | model_error: RuntimeError('Exception occurred during post-physics step') |
| top_cabinet | qiausz | j_dof_rootd_ba001_r | door | scene:Rs_int | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| top_cabinet | qiausz | j_dof_rootd_bb001_r | door | scene:Rs_int | none none |  | None |  | False | 0.0 / None | False | 0 | infeasible_2_reviews: no graspable handle |
| top_cabinet | tactqn | j_link_0 | door | scene:Rs_int | C horizontal_bar | 1.931 | False |  | False | 0.0 / 0.785 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| top_cabinet | twjqua | j_link_0 | drawer | scene:Rs_int | none none | 2.073 | False |  | False | 0.0 / 0.12 | False | 0 | unreachable: unreachable: no collision-free stance with IK to the handle |
| violin_case | jlasop | j_lid | flap | table/floor | none none |  |  |  | False | 0.0 / None | False |  | infeasible_2_reviews: no graspable handle |

## Viable models (11; at least one joint opened by the robot)

* bottom_cabinet_no_top / pluwfl: j_link_0 (0.4495 / 0.785, scene:house_single_floor), j_link_1 (0.1434 / 0.15, scene:house_single_floor)
* bottom_cabinet_no_top / qudfwe: j_link_3 (0.1431 / 0.15, scene:house_single_floor)
* carton / msfzpz: j_link_0 (0.8016 / 0.785, table/floor)
* door / bexenl: j_leaf (0.7768 / 0.785, scene:house_double_floor_lower)
* fridge / jtqazu: j_link_0 (0.7855 / 0.785, scene:restaurant_diner)
* hinged_jar / vzwhbg: j_rb1 (0.8016 / 0.785, table/floor)
* oven / kfftgk: j_dof_rootd_aa001_r (0.4532 / 0.785, scene:house_double_floor_lower)
* oven / leqtlc: j_dof_rootd_aa001_r (0.5963 / 0.785, scene:house_double_floor_lower)
* oven / rwuazb: j_dof_rootd_aa001_r (0.5354 / 0.785, scene:house_double_floor_lower)
* toolbox / redmtl: j_lid (0.5949 / 0.785, table/floor)
* top_cabinet / dmwxyl: j_link_0 (0.7835 / 0.785, scene:house_double_floor_lower), j_link_1 (0.9705 / 0.785, scene:house_double_floor_lower)

## Handles to exclude (unreachable or ungraspable)

* blender / eyedvd / j_lid (flap): unreachable - Marker A is a small tab beside the rear hinge (no leverage). Real grasp: pinch the lid's front free edge across its thickness and lift.
* bottom_cabinet / bamfsz / j_link_1 (drawer): unreachable - Slim horizontal rail pull near the top of the bottom drawer front, spanning most of its width; 12 mm thick, 23 mm proud.
* bottom_cabinet / bamfsz / j_link_2 (drawer): unreachable - Slim horizontal rail pull across the drawer front; A is the rail middle, B and C are its end pieces.
* bottom_cabinet / bamfsz / j_link_3 (drawer): unreachable - Slim horizontal rail pull across the drawer front; B is the rail middle, A and C are its end pieces.
* bottom_cabinet / bamfsz / j_link_4 (drawer): unreachable - Slim horizontal rail pull across the top drawer front; 12 mm thick, 20 mm proud.
* bottom_cabinet / bycegi / j_link_0 (door): unreachable - Thin vertical wire pull near the free -y edge; C is the bar middle, A/B its end posts; hinge on +y side.
* bottom_cabinet / cjcyed / j_link_1 (drawer): no_handle - Sliding door panel (not a drawer), no handle; F sits atop the leading edge with a pinch axis across the panel width, unusable.
* bottom_cabinet / dajebq / j_link_0 (door): unreachable - Thin vertical bar pull at the centre gap, lower +y door; C is the bar middle, A/B its end caps; hinge at outer +y edge.
* bottom_cabinet / dajebq / j_link_1 (door): unreachable - Thin vertical bar pull at the centre gap, upper +y door; B is the bar middle, A its end cap; hinge at outer +y edge.
* bottom_cabinet / dajebq / j_link_2 (door): unreachable - Thin vertical bar pull at the centre gap, lower -y door; C is the bar middle, A/B its end caps; hinge at outer -y edge.
* bottom_cabinet / dajebq / j_link_3 (door): unreachable - Thin vertical bar pull at the centre gap, upper -y door; B is the bar middle, A its end cap; hinge at outer -y edge.
* bottom_cabinet / dsbcxl / j_link_1 (drawer): unreachable - Long thin horizontal rod pull across the bottom drawer front, 26 mm proud; A is the rod middle, B and C are its end posts.
* bottom_cabinet / dsbcxl / j_link_2 (drawer): unreachable - Long thin horizontal rod pull across the top drawer front, 26 mm proud; C is the rod middle, A and B are its end posts.
* bottom_cabinet / eupgpt / j_link_0 (door): unreachable - Thin vertical wire pull near the free -y edge, lower door; A is the bar middle, B/C its end posts; hinge at +y.
* bottom_cabinet / eupgpt / j_link_1 (door): unreachable - Thin vertical wire pull near the free -y edge, upper door; C is the bar middle, A/B its end posts; hinge at +y.
* bottom_cabinet / gcrjan / j_link_0 (door): unreachable - Vertical bar pull at the centre gap, lower -y door; B is the bar middle, A/C its end pieces; hinge at outer -y edge.
* bottom_cabinet / gcrjan / j_link_1 (door): unreachable - Vertical bar pull at the centre gap, upper -y door; A is the bar middle, B its top end piece; hinge at outer -y edge.
* bottom_cabinet / gcrjan / j_link_2 (door): unreachable - Vertical bar pull at the centre gap, lower +y door; A is the bar middle, B its top end piece; hinge at outer +y edge.
* bottom_cabinet / gcrjan / j_link_3 (door): unreachable - Vertical bar pull at the centre gap, upper +y door; A is the bar middle, B its top end piece; hinge at outer +y edge.
* bottom_cabinet / getjzo / j_link_1 (flap): unreachable - Small horizontal pull at the bottom +y corner of the top-hinged flap; bottom edge swings out and up. Bottom-edge fallback unusable (floor level).
* bottom_cabinet / glefdh / j_link_0 (door): unreachable - Vertical bar pull at the centre seam of the lower right (+y) door; C is mid-bar, A/B its end posts; hinge on outer +y edge
* bottom_cabinet / glefdh / j_link_1 (door): unreachable - Vertical bar pull at the centre seam of the lower left (-y) door; C is mid-bar, A/B its end posts; hinge on outer -y edge
* bottom_cabinet / glefdh / j_link_2 (door): unreachable - Vertical bar pull at the centre seam of the upper right (+y) door; A is mid-bar, B its lower end post; hinge on +y edge
* bottom_cabinet / gvtucm / j_link_0 (drawer): infeasible_2_reviews - Flush wooden sliding-door panel (upper left, inner track); no handle, top edge runs hidden inside the frame track, F marker occluded
* bottom_cabinet / gvtucm / j_link_1 (drawer): infeasible_2_reviews - Flush wooden sliding-door panel (upper right, outer track); no handle, top edge enclosed, free vertical edge only ~11 mm proud of neighbour panel
* bottom_cabinet / gvtucm / j_link_2 (drawer): infeasible_2_reviews - Flush wooden sliding-door panel (lower right, outer track); no handle, top edge hidden under the middle rail
* bottom_cabinet / gvtucm / j_link_3 (drawer): infeasible_2_reviews - Flush wooden sliding-door panel (lower left, inner track); no handle, top edge hidden under the middle rail, nothing to pinch
* bottom_cabinet / hnxyev / j_link_1 (drawer): unreachable - Two round wooden knobs (~35 mm, 20 mm stand-off) on the low drawer front; A is the left (-y) knob, B the right, both real
* bottom_cabinet / songes / j_bottomdrawer (drawer): infeasible_2_reviews - Thin ~6 cm bar pull at mid-height inside the recessed panel, unmarked; never protrudes past the frame plane, too shallow to pinch; F edge enclosed
* bottom_cabinet / songes / j_middledrawer (drawer): infeasible_2_reviews - Thin ~6 cm bar pull at mid-height inside the recessed panel, unmarked; sits within the frame plane, nothing to get fingertips behind; F edge enclosed
* bottom_cabinet / songes / j_topdrawer (drawer): infeasible_2_reviews - Thin ~6 cm bar pull at mid-height inside the recessed panel, unmarked; sits within the frame plane; top edge hidden under the top rail
* bottom_cabinet_no_top / apjbrw / j_link_0 (door): unreachable - Small 4 cm tab pull (15 mm thick, 17 mm stand-off) at the door's top corner beside the free +y edge; hinge on -y edge
* bottom_cabinet_no_top / bmsclc / j_link_0 (door): unreachable - Dark 8 cm horizontal bar pull (12 mm thick, 22 mm stand-off) centred near the top of the narrow door; hinge on +y edge
* bottom_cabinet_no_top / exgntg / j_link_0 (door): unreachable - Vertical bar pull (17 cm, 30 mm stand-off) beside centre seam on the right (+y) door; A mid-bar, B/C end posts; hinge on +y edge
* bottom_cabinet_no_top / exgntg / j_link_1 (door): unreachable - Vertical bar pull (17 cm, 30 mm stand-off) beside centre seam on the left (-y) door; A mid-bar, B/C end posts; hinge on -y edge
* bottom_cabinet_no_top / exgntg / j_link_2 (drawer): unreachable - Long horizontal bar pull (34 cm, 16 mm thick, 26 mm stand-off) centred on the top drawer front; A is mid-bar; drawer slides toward +x
* bottom_cabinet_no_top / fmelap / j_link_0 (door): unreachable - Thin bow pull (7 cm, 9 mm wire, 18 mm stand-off) at top centre of right (+y) door; A on bow; hinge on +y edge
* bottom_cabinet_no_top / fmelap / j_link_1 (door): unreachable - Thin 6 mm wire bow pull (7 cm, 18 mm stand-off) at top centre of left (-y) door; A on bow; hinge on -y edge
* bottom_cabinet_no_top / motill / j_link_0 (door): unreachable - Small dark knob/tab (27 mm tall, 7 mm wide, 16 mm stand-off) at top centre of the narrow door; hinge on +y edge
* bottom_cabinet_no_top / qohxjq / j_link_0 (drawer): unreachable - Small inset horizontal pull at centre of bottom drawer front, protrudes ~11 mm (X); top edge F is covered by the drawer above.
* bottom_cabinet_no_top / qohxjq / j_link_1 (drawer): unreachable - Small inset horizontal pull at centre of middle drawer front, protrudes ~12 mm (X); top edge F hidden under the top drawer.
* bottom_cabinet_no_top / qohxjq / j_link_2 (drawer): unreachable - Centre inset pull too shallow (X); cabinet has no top, so pinch the top drawer front's exposed top edge (~17 mm thick) from above.
* bottom_cabinet_no_top / qudfwe / j_link_0 (door): unreachable - Round knob (~24 mm, protrudes ~17 mm) high on the right door near its free centre edge; hinge on the outer right edge.
* bottom_cabinet_no_top / qudfwe / j_link_1 (door): unreachable - Round knob (~24 mm, protrudes ~17 mm) high on the left door near its free centre edge; hinge on the outer left edge.
* bottom_cabinet_no_top / spojpj / j_link_1 (door): unreachable - Long vertical bar pull (~200 mm, protrudes ~27 mm) near the left door's free centre edge; A at mid-bar; hinge on outer left edge.
* bottom_cabinet_no_top / ufhpbn / j_link_0 (drawer): unreachable - Horizontal bar pull (~82 mm, protrudes ~31 mm) near the top of the tall pull-out front; A on the bar, B is its end post.
* carton / causya / j_link_0 (flap): infeasible_2_reviews - F floats in empty air ~0.7 m beside the box; link_0 flap bbox is displaced outside the carton, no visible geometry to pinch.
* carton / cdmmwy / j_left_lid (flap): infeasible_2_reviews - No handle; paper-thin (~1 mm) flap whose free edge meets the other flap flush at the centre seam, no gap to slip a fingertip under.
* carton / cdmmwy / j_right_lid (flap): infeasible_2_reviews - No handle; paper-thin (~1 mm) flap whose free edge meets the other flap flush at the centre seam, no gap to slip a fingertip under.
* carton / hhlmbi / j_link0 (flap): infeasible_2_reviews - No handle; shoebox-style lid skirt sits flush over the box walls, overhanging only ~4 mm, nothing to hook a fingertip under.
* carton / libote / j_link0 (flap): infeasible_2_reviews - No handle; skirted shoebox lid whose rim overhangs the box wall by only ~2 mm, nothing to hook a fingertip under at the front edge.
* carton / ylrxhe / j_link0 (flap): infeasible_2_reviews - No handle; flap free edge lies at the centre seam butted against the other flap, underside inside the box, no gap for a fingertip
* carton / ylrxhe / j_link1 (flap): infeasible_2_reviews - No handle; flap free edge lies at the centre seam butted against the other flap, underside inside the box, no gap for a fingertip
* cedar_chest / fwstpx / j_up (flap): infeasible_2_reviews - No handle; domed trunk lid ~88 mm thick at its free edge, skirt flush with body; only a lock hasp at front centre
* cedar_chest / gbdzls / j_up (flap): infeasible_2_reviews - No handle; flat trunk lid ~48 mm thick, skirt flush with the body, no lip to get a finger under; lock hasp at front centre
* clothes_dryer / fhtvuq / j_link_0 (door): unreachable - No handle; flat black round porthole door sits against the front panel, its rim has no clearance behind it to get a fingertip in
* clothes_dryer / smcyys / j_dof_rootd_aa001_r (door): unreachable - No handle; raised ring rim sits against the front panel, no clearance behind it. Vertical block right of the door is cabinet latch hardware
* clothes_dryer / wivnic / j_dof_rootd_aa001_r (door): unreachable - No handle; porthole door with thick raised ring rim sits in a recessed front panel, no clearance behind the leaf edge; rim maybe pinchable radially
* clothes_dryer / xsuyua / j_link_0 (door): infeasible_2_reviews - No handle; bowl-shaped porthole door 118 mm thick, black bezel nearly flush with the cabinet front, wide (~8 cm) rim, nothing to get fingertips behind
* clothes_dryer / zlmnfg / j_link_0 (door): grasp_no_ag - No handle; glossy bowl-shaped porthole door 127 mm thick, bezel protrudes ~2 cm but leaf edge has no clearance behind; X bumps are bezel facets
* cooler / pnkiag / j_lid (flap): other - No lid handle; front-centre grey latch is body hardware; 93 mm thick lid skirt sits flush over the body front, no exposed lip to grip
* dishwasher / dngvvi / j_link_0 (flap): unreachable - No handle, flat panel; pinch door top edge (43 mm) from above, rear finger needs clearance behind it. Bottom hinge, folds toward viewer.
* dishwasher / dugpby / j_link_0 (flap): infeasible_2_reviews - No handle; flush slab door, back face inside the cabinet opening, top edge below cabinet top, nothing to pinch. Bottom hinge, folds toward viewer.
* dishwasher / xlmier / j_link_1 (flap): unreachable - Horizontal pull bar along the door top, 19 mm thick, 38 mm proud; A/C central segments, B/D outer ends. Bottom hinge, folds toward viewer.
* dishwasher / znfvmj / j_link_2 (flap): infeasible_2_reviews - Recessed rectangular pull pocket near the top centre of the door; no protrusion to pinch. Top edge sits under the fixed control panel. Bottom hinge.
* door / adejun / j_left_leaf (door): model_error - No handle; glazed leaf. A/D are full-height meeting/hinge stiles, B/C bottom/top rails, all frame mouldings. Hinge +y side, swings away from viewer.
* door / adejun / j_right_leaf (door): model_error - No handle; glazed leaf. A/B are full-height hinge/meeting stiles, C/D bottom/top rails, all frame mouldings. Hinge -y side, swings away from viewer.
* door / aleker / j_door1 (door): asset_issue - Vertical pull bar on two standoffs near the free (outer, -y) edge at mid-height, not marked. Hinged at the door1/door2 mullion, swings away from viewer.
* door / aleker / j_door2 (door): asset_issue - Vertical pull bar on two standoffs near the free centre edge (y~0) at mid-height, not marked. Hinged at the door1/door2 mullion, swings away from viewer.
* door / aleker / j_door3 (door): asset_issue - Vertical pull bar on two standoffs near the free centre edge (y~0) at mid-height, not marked. Hinged at the door3/door4 mullion, swings away from viewer.
* door / aleker / j_door4 (door): asset_issue - Vertical pull bar on two standoffs near the free (outer, +y) edge at mid-height, not marked. Hinged at the door3/door4 mullion, swings away from viewer.
* door / bandpe / j_leaf0 (door): model_error - No knob/pull; A marks the full-width push band (~0.3 m tall plate), not a pinchable bar. Hinge at outer -y edge, opens away from viewer.
* door / bandpe / j_leaf1 (door): model_error - No knob/pull; A marks the full-width push band (~0.3 m tall plate), not a pinchable bar. Hinge at outer +y edge, opens away from viewer.
* door / cbgtjl / j_leaf (door): unreachable - Black lever at mid-height near free (+y) edge; A = outside lever, B = inside lever (wrong side). Hinge -y side, swings toward viewer.
* door / chonxi / j_link_0 (door): model_error - No handle; plain flush slab recessed ~10 mm inside its frame, free (+y) edge abuts post so F edge not exposed. Hinge -y side.
* door / cvvumy / j_left_leaf (door): infeasible_2_reviews - No protruding handle; only flush/recessed vertical pull plate near the meeting edge at ~1 m height. Meeting edge not exposed. Hinge on outer (-y) edge.
* door / cvvumy / j_right_leaf (door): infeasible_2_reviews - No protruding handle; only flush/recessed vertical pull plate near the meeting edge at ~1 m height. Meeting edge not exposed. Hinge on outer (+y) edge.
* fridge / dszchb / j_link_0 (door): unreachable - Black vertical bar (26 cm long, 31 mm thick, 51 mm standoff) at mid-height near free (-y) edge; A at its middle. Hinge +y side.
* fridge / dxwbae / j_left_leaf (door): unreachable - Vertical bar handle on the left glass door beside the centre gap, mid-height; door hinged on the far left (-y) edge.
* fridge / dxwbae / j_right_leaf (door): unreachable - Vertical bar handle on the right glass door beside the centre gap, mid-height; door hinged on the far right (+y) edge.
* fridge / hivvdf / j_link_0 (door): unreachable - Full-height vertical bar handle along the door's free left edge; C is the bar middle, A/B are its end mounts. Hinge on the right (+y).
* fridge / hzgqdn / j_link_0 (door): infeasible_2_reviews - No handle: plain flush door slab with interior bins; free edge measures 213 mm thick, far beyond pinch range. Top rail is body hardware.
* fridge / juwaoh / j_link_0 (door): unreachable - No handle; pinch the white door frame (31 mm) across its thickness at the free left edge, mid-height. Overlay door, clearance behind edge uncertain.
* fridge / lleghp / j_shelf0 (drawer): model_error - Top sliding wire shelf, inside the cabinet behind the closed glass door; 13 mm front wire pinchable from above only once the door is open.
* fridge / lleghp / j_shelf1 (drawer): model_error - Bottom sliding wire shelf, inside the cabinet behind the closed glass door; 13 mm front wire pinchable from above only once the door is open.
* fridge / lleghp / j_shelf2 (drawer): model_error - Lower-middle sliding wire shelf, inside the cabinet behind the closed glass door; 13 mm front wire pinchable from above only once the door is open.
* fridge / lleghp / j_shelf3 (drawer): model_error - Upper-middle sliding wire shelf, inside the cabinet behind the closed glass door; 13 mm front wire pinchable from above only once the door is open.
* fridge / xyejdx / j_link_0 (door): unreachable - Vertical bar handle on the upper door's free left edge at mid-door height; grey X ends are its flush mounts. Hinge on the right (+y).
* fridge / xyejdx / j_link_1 (door): unreachable - Vertical bar handle on the lower door's free left edge at mid-door height; grey X ends are its flush mounts. Hinge on the right (+y).
* hinged_jar / gqtsam / j_cap (flap): unreachable - No handle; pinch the lid's overhanging front (+x) free edge across its 17 mm thickness. Lid hinged at the back (-x); release wire clamp first.
* hinged_jar / jnjtrl / j_lid (flap): infeasible_2_reviews - No handle; lid rim sits flush with the jar body with no undercut for a fingertip; wire clamp at the free edge holds it shut.
* laptop / nvulcs / j_link_0 (flap): unreachable - No handle; pinch the lid's free front edge (8 mm thick, slight overhang past the base); lid hinged at the rear, free edge rises.
* openable_window / cbudyv / j_leaf0 (door): unreachable - Lever-style window handle at the leaf's right edge; A is the hanging lever grip, B the hub above it. Left-hinged, swings toward viewer.
* openable_window / cbudyv / j_leaf1 (door): unreachable - Lever-style window handle at the leaf's right edge; B is the hanging lever grip, A the hub above it. Left-hinged, swings toward viewer.
* openable_window / cbudyv / j_leaf2 (door): unreachable - Lever-style window handle at the leaf's right edge; A is the hanging lever grip, B the hub above it. Left-hinged, swings toward viewer.
* openable_window / dpngol / j_left_leaf (drawer): model_error - No handle; bare 17 mm sliding leaf frame, back face nearly touching the fixed frame, edges captured in tracks; nothing to pinch.
* openable_window / dpngol / j_right_leaf (drawer): model_error - No handle; bare 17 mm sliding leaf frame, back face nearly touching the fixed frame, edges captured in tracks; nothing to pinch.
* openable_window / eyvzfu / j_movingpart (drawer): collision - No handle; bare sliding leaf frame, leading stile ~15 cm deep (too thick to pinch), top edge inside the head track.
* openable_window / qdkcdn / j_dof_rootd_aa002_t (drawer): infeasible_2_reviews - No handle; flush wooden sash recessed ~1 cm behind the frame, all edges in tracks or against the fixed sash; thin muntins not pinchable.
* oven / ffitak / j_door (flap): unreachable - Horizontal bar handle across the top of the drop-down door (bottom hinge); B is the bar's middle, A/C its end caps. Shallow 17 mm standoff.
* oven / ffitak / j_rack1 (drawer): unreachable - Front wire rail of the wire rack (~9 mm); pinch it from the front. Only reachable after the door is opened.
* oven / ffitak / j_rack2 (drawer): unreachable - Front wire rail of the lower wire rack (~9 mm); pinch it from the front. Only reachable after the door is opened.
* recycling_bin / mdojox / j_link_2 (flap): infeasible_2_reviews - Handle-less wheelie-bin lid hinged at the rear; F is the front edge, but it rests on the body rim/comb lip with no underside to reach.
* recycling_bin / nuoypc / j_link_2 (flap): infeasible_2_reviews - Handle-less lid hinged at the rear; F marks the front skirt edge, but the body rim/comb lip sits directly beneath, no finger fits under.
* recycling_bin / pdmzhv / j_link_2 (flap): infeasible_2_reviews - Handle-less lid hinged at the rear; F is the 12 mm front lip, flush with the body front face, nothing to get a fingertip under.
* sliding_door / hgcxno / j_left_leaf (drawer): model_error - No handle; framed glass leaf about 170 mm deep per bbox, too thick to pinch. F marks the top edge hidden inside the header.
* sliding_door / hgcxno / j_right_leaf (drawer): model_error - No handle; framed glass leaf about 165 mm deep per bbox, too thick to pinch. F marks the top edge hidden inside the header.
* sliding_door / lrvyvc / j_movingpart (drawer): model_error - No handle; only grip is the leading vertical stile, but the leaf frame is about 156 mm deep per bbox, too thick to pinch.
* sliding_door / tprpvb / j_movingpart (drawer): unreachable - No handle; pinch the panel's leading vertical edge (y-min side, toward the empty rail half) across its 47 mm thickness at about 1.1 m.
* top_cabinet / dhguym / j_link_1 (door): infeasible_2_reviews - No real handle; A sits on the raised panel moulding's right stile (18 mm proud), a makeshift grip at best. Inset leaf, edge not pinchable.
* top_cabinet / fqitbw / j_link_1 (flap): unreachable - Small knob/latch at bottom-right corner of glass-paned flap; ~12 mm protrusion, marginal; flap hinged at top, bottom edge swings up
* top_cabinet / jvdbxh / j_link_0 (door): unreachable - Long vertical bar handle (~156 mm) on door's free right edge, mid-height; B/C are its end posts; hinge on left (-y) edge
* top_cabinet / nkrgez / j_link_0 (door): infeasible_2_reviews - No handle; flush plank door hinged at outer right (+y) edge; free inner edge butts against central shelf unit, not pinchable
* top_cabinet / nkrgez / j_link_1 (door): infeasible_2_reviews - No handle; flush plank door hinged at outer left (-y) edge; free inner edge butts against central shelf unit, not pinchable
* top_cabinet / qiausz / j_dof_rootd_ba001_r (door): infeasible_2_reviews - No handle; thin (9 mm) raised-panel door hinged at outer left (-y) edge; free centre edge not exposed, nothing to pinch
* top_cabinet / qiausz / j_dof_rootd_bb001_r (door): infeasible_2_reviews - No handle; thin (9 mm) raised-panel door hinged at outer right (+y) edge; free centre edge not exposed, nothing to pinch
* top_cabinet / tactqn / j_link_0 (door): unreachable - Long horizontal bar (~176 mm) along door's bottom edge; C is the bar's middle, A/B its end posts; hinge on right (+y) edge
* top_cabinet / twjqua / j_link_0 (drawer): unreachable - Sliding door misread as drawer; F on top-left edge is useless. Possible small vertical pull at right edge, mid-height, protrusion unverified
* violin_case / jlasop / j_lid (flap): infeasible_2_reviews - No lid handle; lid rim sits flush on base rim, no lip to pinch; three front latches must open first. F marks lid's front edge