# Survey re-run diff: runlogs/symmetry_20260915/superseded/v1/symmetry.json -> current

## Texture-class flips per (model, tag) with a texture score in both runs

| tag | preserved -> broken | preserved -> borderline | borderline -> broken | borderline -> preserved | broken -> borderline | broken -> preserved |
|---|---|---|---|---|---|---|
| `x2` | 113 | 24 | 105 | 9 | 2 | 13 |
| `y2` | 77 | 5 | 99 | 12 | 4 | 11 |
| `z2` | 133 | 32 | 126 | 16 | 7 | 0 |
| `x4` | 25 | 19 | 30 | 2 | 1 | 3 |
| `y4` | 11 | 1 | 25 | 2 | 1 | 0 |
| `z4` | 85 | 19 | 65 | 6 | 2 | 0 |
| `x3` | 28 | 16 | 27 | 2 | 0 | 1 |
| `y3` | 12 | 1 | 20 | 0 | 1 | 0 |
| `z3` | 84 | 16 | 52 | 5 | 1 | 0 |
| `x5` | 21 | 21 | 31 | 0 | 1 | 2 |
| `y5` | 11 | 1 | 21 | 3 | 1 | 0 |
| `z5` | 98 | 14 | 57 | 6 | 2 | 0 |
| `x6` | 35 | 19 | 26 | 0 | 0 | 2 |
| `y6` | 11 | 2 | 22 | 3 | 2 | 0 |
| `z6` | 89 | 15 | 58 | 7 | 2 | 0 |
| `xinf` | 22 | 13 | 29 | 3 | 1 | 11 |
| `yinf` | 9 | 1 | 18 | 1 | 1 | 0 |
| `zinf` | 77 | 13 | 48 | 4 | 1 | 0 |
| `full` | 3 | 0 | 2 | 0 | 0 | 0 |

Totals: borderline -> broken: 861, borderline -> preserved: 81, broken -> borderline: 30, broken -> preserved: 43, preserved -> borderline: 232, preserved -> broken: 944

## Final verdicts

338 models changed `symmetry` / `symmetry_near`. Tags removed from `symmetry`: `z2` 43, `zinf` 29, `x2` 28, `z4` 19, `y2` 12, `z6` 12, `z5` 11, `x4` 7, `x6` 7, `xinf` 5, `x5` 4, `z3` 3, `y4` 2, `yinf` 1. Tags added to `symmetry`: none.

88 models gained a `symmetry_near` tag (5 of them were `none` without a near entry before).

| key | before `symmetry` / `symmetry_near` | now `symmetry` / `symmetry_near` |
|---|---|---|
| `trash_can__cdzyew` | `zinf` / `-` | `none` / `-` |
| `trash_can__eahqyq` | `z2` / `-` | `none` / `-` |
| `trash_can__hqdnjz` | `zinf` / `-` | `z4 z5 z6` / `zinf` |
| `trash_can__wkxtxh` | `zinf` / `-` | `z2` / `-` |
| `trash_can__wklill` | `z2` / `-` | `none` / `-` |
| `hamper__clziqw` | `zinf` / `-` | `none` / `-` |
| `hamper__xjftlf` | `z2` / `-` | `none` / `-` |
| `hamper__pdzaca` | `none` / `zinf` | `none` / `-` |
| `hamper__iofciz` | `z2` / `-` | `none` / `-` |
| `hamper__pcyjbu` | `z2` / `-` | `none` / `-` |
| `hamper__pqzyxx` | `z2` / `-` | `none` / `-` |
| `hamper__sijioo` | `none` / `z3 z5` | `none` / `z3` |
| `hamper__ssfvij` | `z2` / `-` | `none` / `-` |
| `hamper__xhdtjn` | `zinf` / `-` | `z4 z5 z6` / `zinf` |
| `taboret__hugvml` | `none` / `z3` | `none` / `-` |
| `taboret__ysrhuw` | `z4` / `-` | `none` / `z2` |
| `taboret__ehaclu` | `none` / `z2` | `none` / `-` |
| `taboret__wjrpwg` | `z3` / `-` | `none` / `z3` |
| `taboret__vzzmir` | `z3` / `-` | `none` / `-` |
| `taboret__bciekv` | `z4` / `-` | `none` / `z4` |
| `footstool__sghekm` | `none` / `z2` | `none` / `-` |
| `ottoman__miftfy` | `z4` / `-` | `none` / `-` |
| `ottoman__xcmniq` | `z2` / `-` | `none` / `z2` |
| `ottoman__qsnawv` | `x4 y4 z4` / `-` | `none` / `z4` |
| `ottoman__ycfbsd` | `z4` / `-` | `none` / `-` |
| `public_trash_can__hxsyxo` | `none` / `z5 z6` | `none` / `-` |
| `public_trash_can__leazin` | `z4 z5` / `zinf` | `none` / `-` |
| `public_trash_can__gvnfgj` | `zinf` / `-` | `z6` / `-` |
| `public_trash_can__aefcem` | `z4` / `zinf` | `none` / `-` |
| `public_trash_can__xkqkbf` | `z2` / `zinf` | `z2` / `-` |
| `wicker_basket__jqfftz` | `none` / `z4` | `none` / `-` |
| `wicker_basket__tsjvyu` | `none` / `z4` | `none` / `-` |
| `carton__cdmmwy` | `x2 y2 z2` / `-` | `none` / `-` |
| `carton__ylrxhe` | `none` / `y2 z4` | `none` / `-` |
| `bottle_of_water__sxffiv` | `none` / `z2` | `none` / `-` |
| `bottle_of_water__migvlt` | `z2` / `zinf` | `none` / `-` |
| `bottle_of_water__cyzaue` | `z5 z6` / `zinf` | `none` / `z6` |
| `bottle_of_water__emquat` | `z4 z5 z6` / `zinf` | `none` / `-` |
| `bottle_of_water__lkpaas` | `z5 z6` / `zinf` | `none` / `-` |
| `bottle_of_water__rmtdxh` | `none` / `z5 z6` | `none` / `-` |
| `bottle_of_water__eeaimz` | `z2` / `zinf` | `none` / `-` |
| `lid__muczdt` | `zinf` / `-` | `none` / `-` |
| `lid__fnqhbz` | `z2` / `z5 z6` | `z2` / `-` |
| `lid__kdswwr` | `none` / `x2 y2 z2` | `none` / `-` |
| `bottle_of_wine__nhmtvt` | `none` / `z5 z6` | `none` / `-` |
| `bottle_of_wine__bmudli` | `zinf` / `-` | `none` / `-` |
| `bottle_of_wine__cldder` | `none` / `zinf` | `none` / `-` |
| `cap__ghwjwe` | `zinf` / `-` | `z4 z5 z6` / `zinf` |
| `can_of_soda__evcxlr` | `none` / `zinf` | `none` / `-` |
| `can_of_soda__wbrrad` | `z4` / `zinf` | `none` / `-` |
| `bottle_of_beer__fgzjnb` | `z6` / `zinf` | `none` / `-` |
| `bottle_of_beer__nfzzqc` | `none` / `zinf` | `none` / `-` |
| `bottle_of_beer__zlmwyn` | `zinf` / `-` | `none` / `z4 z5 z6` |
| `canned_food__foetdd` | `z2` / `z4` | `none` / `-` |
| `lemon__bmvxas` | `x6` / `y2 xinf` | `none` / `-` |
| `lemon__dsacfa` | `none` / `y2 x5 x6` | `none` / `-` |
| `lemon__asdada` | `xinf` / `y2 z2` | `none` / `x6` |
| `lemon__gkrfha` | `x6` / `y2 xinf` | `none` / `-` |
| `lemon__hoghjl` | `none` / `y2 x6` | `none` / `-` |
| `lemon__kvgfdl` | `x4 x5 x6` / `y2 xinf` | `none` / `x5 x6` |
| `lemon__ovbcff` | `none` / `y2 x6` | `none` / `-` |
| `lemon__peqwoa` | `x6` / `y2 xinf` | `none` / `-` |
| `lemon__pffkae` | `x5 x6` / `y2 xinf` | `none` / `-` |
| `lemon__pqweio` | `none` / `y2 x4 x5 x6` | `none` / `-` |
| `shampoo_dispenser__thurmw` | `z4` / `-` | `none` / `-` |
| `shampoo_dispenser__eblcnw` | `none` / `zinf` | `none` / `-` |
| `shampoo_dispenser__kkbhzw` | `z4 z5 z6` / `zinf` | `none` / `-` |
| `shampoo_dispenser__eywylh` | `none` / `z4` | `none` / `-` |
| `box_of_cereal__ykuyqb` | `none` / `x2 y2 z2` | `none` / `-` |
| `beaker__effbnc` | `z6` / `zinf` | `none` / `zinf` |
| `bottle_of_medicine__qqsukh` | `none` / `z6` | `none` / `-` |
| `apple__agveuv` | `none` / `x5 x6` | `none` / `x6` |
| `beeswax_candle__nhdnje` | `zinf` / `-` | `none` / `zinf` |
| `beeswax_candle__oimgmh` | `zinf` / `x2 y2` | `z4 z5 z6` / `zinf` |
| `beeswax_candle__ouzkdj` | `zinf` / `-` | `none` / `zinf` |
| `beeswax_candle__nxewyk` | `zinf` / `-` | `none` / `zinf` |
| `beeswax_candle__pfewit` | `x2 zinf` / `-` | `none` / `zinf` |
| `bottle_of_pop__twtsry` | `z2` / `zinf` | `z2` / `-` |
| `bottle_of_catsup__bcqfxb` | `z2` / `-` | `none` / `z2` |
| `beefsteak_tomato__pnrdxh` | `none` / `z6` | `none` / `-` |
| `bottle_of_olive_oil__cqycjk` | `zinf` / `-` | `none` / `-` |
| `bottle_of_black_pepper__honise` | `none` / `z4` | `none` / `-` |
| `spray_can__bgalkx` | `zinf` / `-` | `z6` / `-` |
| `spray_can__tehmzy` | `zinf` / `-` | `z5 z6` / `z4` |
| `vidalia_onion__ldeuli` | `none` / `xinf` | `none` / `-` |
| `vidalia_onion__dejqhf` | `xinf` / `-` | `none` / `xinf` |
| `pack_of_cigarettes__kienyo` | `z2` / `-` | `none` / `-` |
| `pack_of_cigarettes__ajphvf` | `none` / `y2` | `none` / `-` |
| `pack_of_cigarettes__ohpqua` | `none` / `z2` | `none` / `-` |
| `pack_of_cigarettes__ppbmak` | `none` / `x2 y2 z2` | `none` / `-` |
| `pack_of_cigarettes__tngixy` | `z2` / `x2 y2` | `z2` / `-` |
| `pack_of_cigarettes__wjfbhw` | `z2` / `-` | `none` / `z2` |
| `orange__ucstpm` | `x2 zinf` / `full` | `none` / `z6` |
| `toy_dice__hbkqqp` | `none` / `x2 y2 z2` | `none` / `z2` |
| `toy_dice__cmqtch` | `none` / `x2 y2 z4` | `none` / `-` |
| `bottle_of_essential_oil__wansva` | `none` / `zinf` | `none` / `z5 z6` |
| `bottle_of_essential_oil__eyyhld` | `none` / `zinf` | `none` / `z4 z5 z6` |
| `bottle_of_essential_oil__xvqshn` | `none` / `zinf` | `none` / `z6` |
| `bottle_of_salsa__nafwlf` | `zinf` / `-` | `none` / `z4 z5 z6` |
| `toy_figure__jddrck` | `none` / `z4` | `none` / `-` |
| `teacup__cpozxi` | `z5` / `z6` | `none` / `z5` |
| `cookie_cutter__lqrfzo` | `z2` / `-` | `none` / `z4` |
| `bottle_of_wine__oldpru` | `z6` / `zinf` | `none` / `-` |
| `box_of_cookies__jtalxs` | `none` / `y2` | `none` / `-` |
| `box_of_cookies__lwhbid` | `none` / `x2` | `none` / `-` |
| `notebook__zkmmnf` | `x2 y2 z2` / `-` | `x2 y2` / `z2` |
| `soap_dispenser__cgxrwd` | `none` / `zinf` | `none` / `-` |
| `soap_dispenser__ijxawq` | `none` / `zinf` | `none` / `-` |
| `soap_dispenser__jmulpc` | `none` / `z2` | `none` / `-` |
| `box_of_cereal__ihedod` | `none` / `x2 y2 z2` | `none` / `-` |
| `bottle_of_milk__ddvsgl` | `none` / `x2` | `none` / `-` |
| `bottle_of_milk__qsnnbp` | `none` / `x5 x6` | `none` / `-` |
| `bottle_of_milk__utfsmp` | `x4` / `-` | `none` / `-` |
| `box_of_sanitary_napkins__cmnpuv` | `none` / `z2` | `none` / `-` |
| `box_of_sanitary_napkins__mxdefq` | `none` / `z2` | `none` / `-` |
| `box_of_cane_sugar__rvsivw` | `none` / `z4` | `none` / `-` |
| `cellulose_tape__gchdhk` | `z6` / `x2 z4 z5` | `none` / `-` |
| `conditioner_dispenser__njqnuo` | `x4 x5 x6` / `xinf` | `none` / `-` |
| `conditioner_dispenser__udeflt` | `none` / `zinf` | `none` / `-` |
| `pencil_case__acikzo` | `none` / `z2` | `none` / `-` |
| `pack_of_cigarettes__ghbqav` | `none` / `z2` | `none` / `-` |
| `pack_of_cigarettes__qabdgg` | `z2` / `-` | `none` / `z2` |
| `pack_of_cigarettes__rmrubd` | `z2` / `-` | `none` / `-` |
| `bottle_of_pop__sevoto` | `none` / `z2` | `none` / `-` |
| `bottle_of_shampoo__stjjjm` | `none` / `z6` | `none` / `-` |
| `bottle_of_shampoo__lvvkhx` | `none` / `z6` | `none` / `-` |
| `box_of_butter__mfjuil` | `none` / `z2` | `none` / `-` |
| `toy_dice__kmagva` | `none` / `x2 y2 z2` | `none` / `z2` |
| `toy_dice__krpdji` | `none` / `y2` | `none` / `x2` |
| `toy_dice__mqvaaa` | `none` / `x2 z2 y4` | `none` / `-` |
| `toy_dice__rbzyrv` | `none` / `x2 z2 y4` | `none` / `z2` |
| `toy_dice__xfqqhh` | `none` / `x2 y2 z2` | `none` / `z2` |
| `box_of_baking_mix__lgusnr` | `none` / `z2` | `none` / `-` |
| `box_of_tissues__zutrxn` | `none` / `x2` | `none` / `-` |
| `box_of_tissues__xwstls` | `none` / `x2` | `none` / `-` |
| `box_of_granola_bars__tqwhvz` | `none` / `z2` | `none` / `-` |
| `box_of_granola_bars__bqeeki` | `none` / `z2` | `none` / `-` |
| `box_of_granola_bars__awombj` | `none` / `z2` | `none` / `-` |
| `bottle_of_wine__dijgvu` | `z2` / `zinf` | `none` / `-` |
| `notebook__eijrqw` | `none` / `y2` | `none` / `-` |
| `notebook__ffdjzp` | `x2 y2 z2` / `-` | `none` / `-` |
| `notebook__gqkaqu` | `x2 y2 z2` / `-` | `none` / `x2` |
| `bottle_of_pop__bqretx` | `none` / `z4 z3 z5` | `none` / `-` |
| `bottle_of_pop__cmqubs` | `none` / `z2` | `none` / `-` |
| `bottle_of_pop__kviogj` | `none` / `zinf` | `none` / `-` |
| `bottle_of_pop__wuuoes` | `zinf` / `-` | `z2` / `z4 z3` |
| `bottle_of_pop__zjbqvb` | `zinf` / `-` | `z2` / `z4` |
| `box_of_yogurt__jtjcrx` | `none` / `zinf` | `none` / `-` |
| `hardback__acaxtl` | `x2 y2 z2` / `-` | `y2` / `-` |
| `hardback__aceozs` | `y2` / `-` | `none` / `-` |
| `pack_of_chocolate_bar__tidcst` | `none` / `x2` | `none` / `-` |
| `box_of_coffee__cjtadw` | `none` / `z2` | `none` / `-` |
| `can_of_corn__alphlq` | `z2` / `-` | `none` / `z2` |
| `can_of_icetea__ifrjsc` | `none` / `zinf` | `none` / `-` |
| `can_of_corn__kwwlfn` | `z2` / `-` | `none` / `z2` |
| `goblet__nawrfs` | `none` / `zinf` | `none` / `-` |
| `water_glass__cdteyb` | `zinf` / `-` | `none` / `-` |
| `bottle_of_ginger_beer__zkocwb` | `z2` / `z5 z6` | `none` / `-` |
| `lemonade_bottle__yprkek` | `z4 z5` / `-` | `none` / `-` |
| `soda_water_bottle__upfssc` | `z4 z5` / `-` | `none` / `-` |
| `chopstick__bcwuwv` | `none` / `-` | `none` / `x4 x5 x6` |
| `marker__bwwkmb` | `none` / `x5 x6` | `none` / `x6` |
| `pencil__bfkwhb` | `xinf` / `y2 z2` | `xinf` / `-` |
| `table_knife__eebpap` | `none` / `x2` | `none` / `-` |
| `table_knife__jxdfyy` | `none` / `x2` | `none` / `-` |
| `pen__gzrbrx` | `xinf` / `-` | `none` / `xinf` |
| `lipstick__albshz` | `y4` / `y5` | `none` / `-` |
| `folder__lktggf` | `y2` / `-` | `none` / `-` |
| `notepad__rvevdj` | `x2` / `-` | `none` / `-` |
| `bottle_of_dish_soap__bnmixt` | `z2` / `-` | `none` / `-` |
| `bottle_of_lotion__tkryrh` | `z4 z6` / `zinf` | `none` / `-` |
| `bottle_of_face_cream__dztaed` | `zinf` / `x2 y2` | `none` / `-` |
| `box_of_barley__lxpwnw` | `none` / `z2` | `none` / `-` |
| `bottle_of_ginger__drqhzo` | `none` / `zinf` | `none` / `z6` |
| `bottle_of_ground_cloves__vzamzb` | `zinf` / `-` | `z4 z5 z6` / `zinf` |
| `bottle_of_ground_mace__lgpxro` | `none` / `zinf` | `none` / `-` |
| `bottle_of_ground_nutmeg__qebruq` | `none` / `zinf` | `none` / `z5 z6` |
| `bottle_of_mustard_seeds__grryaf` | `none` / `zinf` | `none` / `-` |
| `bottle_of_paprika__nrdgrp` | `zinf` / `-` | `z4 z5 z6` / `zinf` |
| `bottle_of_sage__bterim` | `none` / `-` | `none` / `z5 z6` |
| `bottle_of_seasoning__vqwqqv` | `none` / `zinf` | `none` / `z5 z6` |
| `bottle_of_onion_powder__xruqod` | `none` / `zinf` | `none` / `-` |
| `bottle_of_poppy_seeds__xdtrgi` | `none` / `zinf` | `none` / `-` |
| `bottle_of_tea_leaves__pfbtus` | `none` / `z2` | `none` / `zinf` |
| `jar_of_curry_powder__ozhftu` | `none` / `zinf` | `none` / `z4 z5 z6` |
| `bottle_of_coconut_milk__idenxg` | `none` / `z4 z5 z6` | `none` / `-` |
| `bottle_of_coconut_water__lsixio` | `z4` / `zinf` | `none` / `z2` |
| `bottle_of_tea__iladfg` | `none` / `z4 z3 z5` | `none` / `-` |
| `bottle_of_tea__yatmrs` | `none` / `z2` | `none` / `-` |
| `bag_of_flour__fntqmd` | `z2` / `-` | `none` / `-` |
| `jar_of_sugar__bholak` | `z2` / `-` | `none` / `-` |
| `carton_of_orange_juice__brryuo` | `none` / `z2` | `none` / `-` |
| `carton_of_pineapple_juice__vzueyg` | `none` / `z6` | `none` / `-` |
| `asparagus__eodozo` | `none` / `-` | `none` / `y6` |
| `carrot__aucrah` | `none` / `xinf` | `none` / `x6` |
| `asparagus__xguktb` | `none` / `-` | `none` / `y5 y6` |
| `carrot__qhmmmx` | `none` / `xinf` | `none` / `x4 x3 x5` |
| `box_of_rum__lnjbjj` | `x2 y2 z4` / `-` | `z2` / `-` |
| `box_of_whiskey__ftapxo` | `none` / `x2 y2 z2` | `none` / `-` |
| `hamburger_bun__fqxumh` | `none` / `zinf` | `none` / `z4 z5 z6` |
| `hamburger_bun__jeygaq` | `none` / `z4 z5 z6` | `none` / `-` |
| `bottle_of_disinfectant__faedff` | `none` / `z2` | `none` / `-` |
| `flashlight__jtrbkc` | `none` / `x6` | `none` / `-` |
| `bread_slice__pfggnm` | `none` / `y2 z4` | `none` / `-` |
| `bandage__riftxh` | `z2` / `x2 y2` | `z2` / `-` |
| `bottle_of_carrot_juice__jkuhio` | `none` / `z2` | `none` / `-` |
| `bottle_of_fruit_punch__azcigi` | `none` / `z2` | `none` / `-` |
| `bottle_of_papaya_juice__nmfmxy` | `none` / `z2` | `none` / `-` |
| `paper_bag__bzsxgw` | `z2` / `-` | `none` / `-` |
| `vase__aegxpb` | `none` / `zinf` | `none` / `-` |
| `vase__atgnsc` | `none` / `zinf` | `none` / `-` |
| `tissue_dispenser__nflhrf` | `x2` / `-` | `none` / `-` |
| `jar_of_mayonnaise__ktmjvg` | `none` / `zinf` | `none` / `-` |
| `jar_of_tumeric__miivhi` | `none` / `z4` | `none` / `-` |
| `cap__iqeyba` | `zinf` / `-` | `z4 z5 z6` / `zinf` |
| `dip_candle__poexjo` | `z6` / `x2 z4 z5` | `none` / `x2 z4 z5 z6` |
| `dip_candle__mxjejs` | `x2 zinf` / `-` | `zinf` / `x2 y2` |
| `cabinet_door__ertkre` | `none` / `x2` | `none` / `-` |
| `table_lamp__ehjsdz` | `z2` / `-` | `none` / `-` |
| `tomato_paste_can__krarex` | `z4 z5` / `-` | `none` / `-` |
| `paperback_book__jhaltj` | `none` / `x2` | `none` / `-` |
| `paperback_book__aubynm` | `none` / `x2` | `none` / `-` |
| `paperback_book__phsgun` | `none` / `x2` | `none` / `-` |
| `paperback_book__vciijb` | `none` / `x2` | `none` / `-` |
| `paperback_book__bisnup` | `x2` / `-` | `none` / `-` |
| `paperback_book__eqzjjk` | `none` / `x2` | `none` / `-` |
| `paperback_book__ixwtnb` | `x2` / `-` | `none` / `-` |
| `paperback_book__ncjkcg` | `none` / `x2` | `none` / `-` |
| `paperback_book__wfcxjv` | `x2` / `-` | `none` / `-` |
| `hardback__bgtbdp` | `none` / `x2` | `none` / `-` |
| `letter__oheufk` | `none` / `x2 y2 z2` | `none` / `-` |
| `letter__dyvllf` | `none` / `z2` | `none` / `-` |
| `letter__csncjv` | `none` / `z2` | `none` / `-` |
| `letter__esmoua` | `none` / `x2 y2 z2` | `none` / `-` |
| `letter__qrykxc` | `none` / `z2` | `none` / `-` |
| `keyboard__jtdyix` | `none` / `x2` | `none` / `-` |
| `keyboard__uwacaq` | `z2` / `x2 y2` | `z2` / `-` |
| `newspaper__edarlp` | `none` / `x2` | `none` / `-` |
| `carton_of_eggs__stxfxb` | `x2` / `-` | `none` / `-` |
| `carton_of_eggs__rixhgu` | `none` / `z2` | `none` / `-` |
| `chopping_board__psabiv` | `none` / `x2` | `none` / `-` |
| `notepad__awcyha` | `none` / `x2` | `none` / `-` |
| `notepad__hirdfd` | `x2` / `-` | `none` / `-` |
| `clipboard__envjqe` | `none` / `z2` | `none` / `-` |
| `clipboard__gmxyfo` | `x2 y2 z2` / `-` | `x2 z2` / `-` |
| `wall_clock__zpctvp` | `none` / `x4 x6` | `none` / `-` |
| `wall_clock__jerval` | `none` / `x2` | `none` / `-` |
| `wall_clock__xevvel` | `x4` / `x6` | `none` / `-` |
| `painting__ejhanr` | `none` / `x2` | `none` / `-` |
| `painting__gardkz` | `none` / `x2` | `none` / `-` |
| `painting__eexydp` | `none` / `x2` | `none` / `-` |
| `signpost__vwhtlm` | `z2` / `-` | `none` / `-` |
| `guitar__hpqjrq` | `none` / `x2` | `none` / `-` |
| `hanger__anobre` | `none` / `x2` | `none` / `-` |
| `towel_rack__nhlijv` | `none` / `z2` | `none` / `-` |
| `rolling_pin__smnklu` | `x2 yinf` / `-` | `x2 z2 y4 y5 y6` / `yinf` |
| `baseball_bat__yalgei` | `none` / `xinf` | `none` / `-` |
| `baseball_bat__spcxrt` | `xinf` / `-` | `none` / `-` |
| `shelf__qrmiur` | `none` / `x2 y2 z2` | `none` / `-` |
| `shelf__iruewm` | `none` / `x2 y2 z2` | `none` / `-` |
| `shelf__kxvbui` | `x2 y2 z2` / `-` | `x2` / `-` |
| `radiator__arsjbe` | `none` / `y2` | `none` / `-` |
| `radiator__dwmloz` | `x2` / `-` | `none` / `-` |
| `radiator__xnoiwp` | `none` / `y2` | `none` / `-` |
| `radiator__wntwxx` | `y2` / `-` | `none` / `-` |
| `drumstick__jauzlc` | `z2 x4 x5 x6` / `y2 xinf` | `x4 x5 x6` / `-` |
| `wall_mounted_tv__ggfznh` | `x2` / `-` | `none` / `-` |
| `wall_mounted_tv__dxyfxf` | `x2` / `-` | `none` / `-` |
| `wall_mounted_tv__ylvjhb` | `x2` / `-` | `none` / `-` |
| `wall_mounted_tv__sodbbr` | `x2` / `-` | `none` / `-` |
| `umbrella__hdfkbc` | `none` / `x4` | `none` / `-` |
| `umbrella__zxnjwz` | `xinf` / `-` | `none` / `-` |
| `whiteboard__gvkgsi` | `x2` / `-` | `none` / `-` |
| `picture__watgtj` | `none` / `z2` | `none` / `-` |
| `towel_rack__crivoa` | `x2` / `-` | `none` / `-` |
| `towel_rack__quznqx` | `none` / `x2 yinf` | `none` / `x2 y2 z2` |
| `decorative_sign__hgonwj` | `x2` / `-` | `none` / `x2` |
| `ruler__ujizkx` | `none` / `x2 y2 z2` | `none` / `y2` |
| `ruler__vmnaih` | `none` / `x2 z2` | `none` / `-` |
| `address__pbfbxj` | `none` / `x2 y2` | `none` / `-` |
| `pack_of_kielbasa__dgodnn` | `x2` / `-` | `none` / `-` |
| `pack_of_kielbasa__rvhpal` | `none` / `x2` | `none` / `-` |
| `pack_of_kielbasa__fwqdoy` | `x2` / `-` | `none` / `x2` |
| `pack_of_kielbasa__bzlsuh` | `x2` / `-` | `none` / `-` |
| `pack_of_kielbasa__nhkuzn` | `x2` / `-` | `none` / `-` |
| `pen__hiyeki` | `x4 x5 x6` / `-` | `x4 x3 x5` / `xinf` |
| `pen__enfumt` | `none` / `xinf` | `none` / `x2` |
| `cutting_board__tcdrzs` | `y2` / `x2 z2` | `none` / `x2 y2 z2` |
| `plate__luhkiz` | `z4 z5 z6` / `zinf` | `none` / `z4 z6` |
| `plate__bgxzec` | `zinf` / `x2 y2` | `zinf` / `-` |
| `plate__mtetqm` | `z4` / `-` | `z2` / `z4` |
| `plate__spppps` | `none` / `z2` | `none` / `-` |
| `platter__hnlivs` | `z2` / `-` | `none` / `-` |
| `platter__kiiamx` | `z2` / `-` | `none` / `-` |
| `platter__ekjpdj` | `z2` / `-` | `none` / `-` |
| `toy_figure__gqspnb` | `none` / `x2` | `none` / `-` |
| `tennis_ball__rgekxe` | `y2 x4 z5 z6` / `full` | `x2` / `-` |
| `tray__hjxczh` | `x2 y2 z2` / `-` | `z2` / `-` |
| `tray__mkdcha` | `z2` / `x2 y2` | `z2` / `-` |
| `lemon__pwqedq` | `none` / `y2 x4 x5 x6` | `none` / `-` |
| `half_lime__dtyjkb` | `none` / `zinf` | `none` / `z6` |
| `half_lime__wdadtn` | `none` / `zinf` | `none` / `z2` |
| `bowl__belcml` | `z4` / `zinf` | `z4` / `-` |
| `bowl__eawgwj` | `z4` / `z6` | `z4` / `-` |
| `saucepot__wfryvm` | `z2` / `-` | `none` / `-` |
| `saucepot__vqtkwq` | `z2` / `-` | `none` / `-` |
| `saucepot__chjetk` | `z2` / `zinf` | `z2` / `-` |
| `vase__gopbrh` | `zinf` / `-` | `z4` / `-` |
| `cinnamon_stick__qxpzdm` | `none` / `x2 y2 z2` | `none` / `x2 z2` |
| `cinnamon_stick__qmlyim` | `none` / `x2 z2` | `none` / `x2` |
| `cinnamon_stick__qsqvgk` | `none` / `x2 y2 z2` | `none` / `z2` |
| `cinnamon_stick__smfuqz` | `none` / `x2 yinf` | `none` / `z2 y6` |
| `pencil__hmawzy` | `z2 x5 x6` / `-` | `x5 x6` / `y2 xinf` |
| `pencil__heqkmr` | `x6` / `-` | `x6` / `y2 xinf` |
| `pencil__lruiot` | `x5 x6` / `xinf` | `x5 x6` / `y2 xinf` |
| `pencil__lykqoh` | `x4 x5 x6` / `-` | `x4 x5 x6` / `y2 xinf` |
| `pencil__vpzhth` | `x4 x5 x6` / `-` | `x4 x5 x6` / `y2 xinf` |
| `pencil__vvofwr` | `x5 x6` / `xinf` | `x5 x6` / `y2 xinf` |
| `pencil__mvmxfo` | `none` / `-` | `none` / `xinf` |
| `pencil__vedvit` | `z2 xinf` / `-` | `y2 xinf` / `-` |
| `marker__vrjwkg` | `none` / `xinf` | `none` / `x3` |
| `marker__vjjsul` | `none` / `xinf` | `none` / `x4 x3 x5` |
| `pepper_shaker__tgcwtr` | `none` / `zinf` | `none` / `z5 z6` |
| `pepper_shaker__ykujyn` | `zinf` / `x2 y2` | `z4 z5 z6` / `zinf` |
| `salt_shaker__iomwtn` | `z3` / `zinf` | `none` / `z4 z5 z6` |
| `colored_pencil__bkdqwb` | `x4 x5 x6` / `-` | `x4 x5 x6` / `y2 xinf` |
| `colored_pencil__deuvcx` | `x4 x5 x6` / `-` | `x6` / `y2 xinf` |
| `colored_pencil__egvqng` | `z2 x4 x5 x6` / `-` | `x4 x5 x6` / `y2 xinf` |
| `colored_pencil__jdqvdl` | `z2 x4 x5 x6` / `-` | `z2 x4 x5 x6` / `y2 xinf` |
| `colored_pencil__kadvlg` | `x4 x5 x6` / `-` | `x4 x5 x6` / `y2 xinf` |
| `colored_pencil__kgmapz` | `x4 x5 x6` / `xinf` | `x4 x5 x6` / `y2 xinf` |
| `colored_pencil__kjwoqm` | `x4 x5 x6` / `-` | `x4 x5 x6` / `y2 xinf` |
| `colored_pencil__qrxemk` | `x4 x5 x6` / `xinf` | `x4 x5 x6` / `y2 xinf` |
| `can_of_baking_mix__orwvfx` | `z2` / `zinf` | `none` / `-` |
| `can_of_baking_mix__rxorlp` | `z2` / `zinf` | `none` / `z2` |
| `pop_case__gwgaqb` | `none` / `y2` | `none` / `-` |
| `pop_case__qcfxec` | `none` / `z2` | `none` / `-` |
| `pop_case__pldtgy` | `none` / `x2` | `none` / `-` |

## (model, tag) that were texture-preserved and are not any more (1176)

| key | tag | now | MAD | p95 | p99 | blob (x 6x6 mm) | blob mean /255 | view |
|---|---|---|---|---|---|---|---|---|
| `trash_can__cdzyew` | `z2` | broken | 0.8 | 1.6 | 13 | 55.7 | 53 | -x |
| `trash_can__cdzyew` | `z4` | broken | 0.9 | 1.7 | 29 | 55.7 | 53 | -x |
| `trash_can__cdzyew` | `z3` | broken | 0.9 | 2.0 | 28 | 57.6 | 54 | -y |
| `trash_can__cdzyew` | `z5` | broken | 1.0 | 2.0 | 28 | 55.9 | 53 | -x |
| `trash_can__cdzyew` | `z6` | broken | 0.9 | 1.9 | 24 | 55.9 | 53 | -x |
| `trash_can__cdzyew` | `zinf` | broken | 0.9 | 1.9 | 22 | 55.6 | 53 | -x |
| `trash_can__vasiit` | `z4` | broken | 1.4 | 5.9 | 41 | 393.8 | 20 | +x |
| `trash_can__vasiit` | `z3` | broken | 1.5 | 6.9 | 44 | 399.3 | 23 | -x |
| `trash_can__vasiit` | `z5` | broken | 1.3 | 6.6 | 37 | 117.1 | 19 | +x |
| `trash_can__vasiit` | `z6` | broken | 1.3 | 6.6 | 35 | 56.1 | 19 | -x |
| `trash_can__eahqyq` | `z2` | broken | 1.0 | 2.6 | 18 | 88.3 | 26 | +z |
| `trash_can__cjmezk` | `z2` | broken | 0.5 | 3.1 | 13 | 0.2 | 21 | -z |
| `trash_can__cjmezk` | `z4` | broken | 0.6 | 4.6 | 11 | 0.3 | 0 | +y |
| `trash_can__cjmezk` | `z3` | broken | 0.6 | 5.6 | 13 | 0.2 | 0 | -y |
| `trash_can__cjmezk` | `z5` | broken | 0.5 | 4.0 | 11 | 0.2 | 0 | +y |
| `trash_can__cjmezk` | `z6` | broken | 0.5 | 3.1 | 11 | 0.2 | 0 | -y |
| `trash_can__cjmezk` | `zinf` | broken | 0.6 | 5.2 | 13 | 0.3 | 0 | +y |
| `trash_can__gxajos` | `z2` | broken | 0.4 | 0.1 | 23 | 83.6 | 40 | -z |
| `trash_can__gxajos` | `z4` | broken | 0.5 | 0.1 | 26 | 76.2 | 34 | +x |
| `trash_can__gxajos` | `z3` | broken | 0.4 | 0.1 | 22 | 121.9 | 38 | +x |
| `trash_can__gxajos` | `z5` | broken | 0.5 | 0.1 | 27 | 86.0 | 33 | -y |
| `trash_can__gxajos` | `z6` | broken | 0.5 | 0.1 | 27 | 69.6 | 34 | -y |
| `trash_can__gxajos` | `zinf` | broken | 0.4 | 0.1 | 22 | 102.4 | 36 | +x |
| `trash_can__hqdnjz` | `zinf` | borderline | 0.6 | 2.0 | 5 | 0.4 | 0 | -z |
| `trash_can__uknjdm` | `z2` | broken | 0.5 | 3.9 | 11 | 0.2 | 0 | +x |
| `trash_can__uknjdm` | `z4` | broken | 0.4 | 2.0 | 11 | 0.5 | 0 | +x |
| `trash_can__uknjdm` | `z3` | broken | 0.5 | 3.4 | 11 | 0.3 | 0 | -y |
| `trash_can__uknjdm` | `z5` | broken | 0.4 | 2.0 | 10 | 0.3 | 0 | +y |
| `trash_can__uknjdm` | `z6` | broken | 0.4 | 1.9 | 10 | 0.4 | 0 | +x |
| `trash_can__uknjdm` | `zinf` | broken | 0.5 | 3.2 | 12 | 0.3 | 0 | +x |
| `trash_can__wkxtxh` | `z4` | broken | 1.0 | 4.5 | 10 | 0.5 | 16 | +x |
| `trash_can__wkxtxh` | `z3` | broken | 1.0 | 4.3 | 10 | 0.5 | 16 | +x |
| `trash_can__wkxtxh` | `z5` | broken | 1.0 | 4.5 | 10 | 0.5 | 16 | +x |
| `trash_can__wkxtxh` | `z6` | broken | 1.0 | 4.4 | 10 | 0.5 | 16 | +x |
| `trash_can__wkxtxh` | `zinf` | broken | 1.0 | 4.4 | 10 | 0.5 | 16 | +x |
| `trash_can__wklill` | `z2` | broken | 1.1 | 2.9 | 4 | 0.0 | 0 |  |
| `hamper__clziqw` | `z2` | broken | 2.5 | 6.0 | 14 | 168.2 | 19 | -z |
| `hamper__clziqw` | `z4` | broken | 2.6 | 6.0 | 13 | 76.1 | 20 | -z |
| `hamper__clziqw` | `z3` | broken | 2.6 | 6.0 | 14 | 160.0 | 18 | -z |
| `hamper__clziqw` | `z5` | broken | 2.5 | 6.0 | 12 | 109.7 | 23 | -z |
| `hamper__clziqw` | `z6` | broken | 2.5 | 6.1 | 12 | 81.1 | 21 | -z |
| `hamper__clziqw` | `zinf` | broken | 2.6 | 6.0 | 14 | 322.1 | 19 | -z |
| `hamper__xjftlf` | `z2` | broken | 1.6 | 4.2 | 8 | 12.8 | 13 | +x |
| `hamper__iofciz` | `z2` | broken | 0.4 | 2.4 | 7 | 0.1 | 8 | -z |
| `hamper__pcyjbu` | `z2` | broken | 1.6 | 3.7 | 8 | 112.0 | 27 | +y |
| `hamper__pqzyxx` | `z2` | broken | 1.4 | 3.5 | 7 | 8.5 | 13 | -x |
| `hamper__wdsnjy` | `x2` | broken | 1.5 | 3.5 | 5 | 0.0 | 0 |  |
| `hamper__wdsnjy` | `y2` | broken | 1.6 | 3.4 | 4 | 0.0 | 0 |  |
| `hamper__ssfvij` | `z2` | broken | 0.2 | 0.6 | 5 | 0.0 | 0 |  |
| `hamper__ssfvij` | `z4` | broken | 0.2 | 0.7 | 5 | 0.0 | 0 |  |
| `hamper__xhdtjn` | `zinf` | borderline | 0.9 | 2.5 | 4 | 0.3 | 1 | +z |
| `taboret__ysrhuw` | `z2` | borderline | 1.2 | 4.7 | 7 | 0.0 | 0 |  |
| `taboret__ysrhuw` | `z4` | broken | 1.2 | 4.9 | 8 | 48.2 | 24 | +z |
| `taboret__ysrhuw` | `z5` | broken | 1.5 | 5.2 | 9 | 56.7 | 24 | +z |
| `taboret__ehaclu` | `z2` | broken | 0.3 | 1.0 | 8 | 8.7 | 24 | +z |
| `taboret__wjrpwg` | `z3` | borderline | 1.9 | 8.1 | 12 | 0.9 | 0 | +x |
| `taboret__vzzmir` | `z2` | broken | 0.7 | 1.9 | 24 | 69.0 | 20 | -z |
| `taboret__vzzmir` | `z4` | broken | 0.6 | 1.6 | 16 | 26.7 | 20 | -z |
| `taboret__vzzmir` | `z3` | broken | 0.8 | 2.2 | 25 | 53.6 | 24 | -z |
| `taboret__vzzmir` | `z5` | broken | 0.9 | 2.0 | 30 | 81.3 | 22 | -z |
| `taboret__vzzmir` | `z6` | broken | 1.0 | 2.3 | 33 | 105.2 | 22 | -z |
| `taboret__vzzmir` | `zinf` | broken | 0.8 | 2.1 | 29 | 45.9 | 20 | -z |
| `taboret__bciekv` | `z2` | borderline | 1.7 | 8.2 | 13 | 0.8 | 9 | +z |
| `taboret__bciekv` | `z4` | borderline | 1.6 | 7.5 | 12 | 0.4 | 14 | +z |
| `ottoman__miftfy` | `z2` | broken | 1.6 | 5.4 | 8 | 1.8 | 10 | -z |
| `ottoman__miftfy` | `z4` | broken | 1.7 | 5.6 | 8 | 7.2 | 16 | -z |
| `ottoman__xcmniq` | `z2` | borderline | 1.6 | 3.9 | 5 | 0.0 | 0 |  |
| `ottoman__qsnawv` | `x2` | broken | 0.5 | 1.1 | 3 | 0.2 | 2 | +z |
| `ottoman__qsnawv` | `y2` | broken | 0.5 | 1.1 | 3 | 0.2 | 1 | -z |
| `ottoman__qsnawv` | `z2` | borderline | 0.3 | 0.7 | 1 | 0.0 | 0 |  |
| `ottoman__qsnawv` | `x4` | broken | 0.4 | 1.0 | 2 | 0.2 | 1 | -z |
| `ottoman__qsnawv` | `y4` | broken | 0.4 | 1.0 | 2 | 0.2 | 1 | -z |
| `ottoman__qsnawv` | `z4` | borderline | 0.3 | 0.7 | 1 | 0.2 | 5 | -z |
| `ottoman__ycfbsd` | `z2` | broken | 1.1 | 2.7 | 4 | 1.0 | 1 | -z |
| `ottoman__ycfbsd` | `z4` | broken | 1.1 | 2.7 | 4 | 1.8 | 16 | -z |
| `public_trash_can__glzckq` | `z5` | broken | 2.2 | 14.4 | 46 | 173.0 | 61 | +x |
| `public_trash_can__fkosow` | `x2` | broken | 1.1 | 0.1 | 41 | 228.6 | 52 | +x |
| `public_trash_can__fkosow` | `y2` | broken | 1.0 | 0.1 | 38 | 123.3 | 50 | -x |
| `public_trash_can__hxsyxo` | `z2` | broken | 0.8 | 0.7 | 24 | 128.0 | 35 | +x |
| `public_trash_can__hxsyxo` | `z4` | broken | 0.7 | 0.3 | 24 | 128.0 | 35 | +x |
| `public_trash_can__hxsyxo` | `z3` | broken | 0.8 | 0.7 | 24 | 128.0 | 35 | +x |
| `public_trash_can__hxsyxo` | `z5` | broken | 0.8 | 0.6 | 27 | 128.0 | 35 | +x |
| `public_trash_can__hxsyxo` | `z6` | broken | 0.8 | 0.7 | 27 | 128.0 | 35 | +x |
| `public_trash_can__hxsyxo` | `zinf` | broken | 0.6 | 0.4 | 23 | 192.6 | 34 | +x |
| `public_trash_can__leazin` | `z2` | broken | 2.0 | 5.9 | 9 | 0.4 | 6 | +y |
| `public_trash_can__leazin` | `z4` | broken | 2.2 | 5.7 | 9 | 0.9 | 2 | -z |
| `public_trash_can__leazin` | `z5` | broken | 2.4 | 6.0 | 9 | 0.0 | 0 |  |
| `public_trash_can__gvnfgj` | `x2` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `public_trash_can__gvnfgj` | `y2` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `public_trash_can__gvnfgj` | `z4` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `public_trash_can__gvnfgj` | `z5` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `public_trash_can__gvnfgj` | `zinf` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `public_trash_can__aefcem` | `z2` | broken | 1.9 | 5.5 | 11 | 55.6 | 15 | +z |
| `public_trash_can__aefcem` | `z4` | broken | 1.9 | 5.5 | 11 | 35.7 | 13 | +z |
| `public_trash_can__aefcem` | `z3` | broken | 2.5 | 8.2 | 21 | 69.1 | 17 | -z |
| `public_trash_can__aefcem` | `z6` | broken | 2.5 | 8.5 | 21 | 57.2 | 18 | -z |
| `music_stool__dqvlff` | `z2` | broken | 1.2 | 2.8 | 15 | 43.3 | 30 | +y |
| `music_stool__dqvlff` | `z4` | broken | 1.8 | 14.1 | 16 | 98.8 | 29 | +y |
| `carton__cdmmwy` | `x2` | broken | 1.5 | 3.0 | 21 | 120.5 | 23 | -x |
| `carton__cdmmwy` | `y2` | broken | 1.5 | 3.0 | 21 | 120.7 | 23 | -x |
| `carton__cdmmwy` | `z2` | broken | 1.4 | 2.8 | 19 | 115.6 | 24 | +x |
| `backpack__gvbiwl` | `z2` | broken | 2.1 | 6.0 | 21 | 30.7 | 16 | +z |
| `bottle_of_water__migvlt` | `z2` | broken | 2.5 | 12.3 | 19 | 11.7 | 23 | +y |
| `bottle_of_water__cyzaue` | `z2` | broken | 0.6 | 2.0 | 10 | 5.1 | 19 | +z |
| `bottle_of_water__cyzaue` | `z4` | broken | 0.6 | 2.4 | 8 | 3.8 | 17 | +z |
| `bottle_of_water__cyzaue` | `z3` | broken | 0.5 | 2.0 | 8 | 5.8 | 17 | +z |
| `bottle_of_water__cyzaue` | `z5` | broken | 0.5 | 2.1 | 7 | 4.1 | 14 | +z |
| `bottle_of_water__cyzaue` | `z6` | borderline | 0.4 | 1.8 | 7 | 3.2 | 13 | +z |
| `bottle_of_water__cyzaue` | `zinf` | broken | 0.6 | 2.4 | 8 | 5.6 | 16 | +z |
| `bottle_of_water__emquat` | `z2` | broken | 1.6 | 9.6 | 24 | 15.5 | 17 | -z |
| `bottle_of_water__emquat` | `z4` | broken | 1.4 | 7.9 | 20 | 17.5 | 14 | -z |
| `bottle_of_water__emquat` | `z3` | broken | 1.6 | 9.0 | 22 | 17.8 | 14 | -z |
| `bottle_of_water__emquat` | `z5` | broken | 1.3 | 7.0 | 18 | 16.3 | 12 | -z |
| `bottle_of_water__emquat` | `z6` | broken | 1.2 | 6.4 | 17 | 15.0 | 11 | -z |
| `bottle_of_water__emquat` | `zinf` | broken | 1.7 | 9.6 | 24 | 21.8 | 37 | +z |
| `bottle_of_water__lkpaas` | `x2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `y2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `z2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `z4` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `z3` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `z5` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `z6` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `zinf` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `z2` | broken | 0.3 | 1.4 | 7 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `z4` | broken | 0.3 | 1.5 | 7 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `z3` | broken | 0.3 | 1.6 | 7 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `z5` | broken | 0.3 | 1.4 | 7 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `z6` | broken | 0.2 | 1.2 | 6 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `zinf` | broken | 0.3 | 1.6 | 7 | 0.0 | 0 |  |
| `bottle_of_water__eeaimz` | `z2` | broken | 2.6 | 12.6 | 20 | 22.8 | 26 | +y |
| `bottle_of_water__cytqio` | `z2` | broken | 0.2 | 1.1 | 6 | 0.0 | 0 |  |
| `bottle_of_water__cytqio` | `z4` | broken | 0.2 | 0.9 | 6 | 0.0 | 0 |  |
| `bottle_of_water__cytqio` | `z3` | broken | 0.2 | 1.0 | 6 | 0.0 | 0 |  |
| `bottle_of_water__cytqio` | `z5` | broken | 0.2 | 0.7 | 6 | 0.0 | 0 |  |
| `bottle_of_water__cytqio` | `z6` | broken | 0.2 | 0.7 | 5 | 0.0 | 0 |  |
| `bottle_of_water__cytqio` | `zinf` | broken | 0.2 | 1.0 | 6 | 0.0 | 0 |  |
| `lid__muczdt` | `x2` | broken | 0.4 | 1.0 | 5 | 0.0 | 0 |  |
| `lid__muczdt` | `y2` | broken | 0.4 | 1.0 | 6 | 0.0 | 0 |  |
| `lid__muczdt` | `z2` | broken | 0.3 | 1.0 | 4 | 0.0 | 0 |  |
| `lid__muczdt` | `z4` | broken | 0.3 | 1.0 | 4 | 0.0 | 0 |  |
| `lid__muczdt` | `z3` | broken | 0.3 | 1.0 | 4 | 0.0 | 0 |  |
| `lid__muczdt` | `z5` | broken | 0.3 | 1.0 | 4 | 0.0 | 0 |  |
| `lid__muczdt` | `z6` | broken | 0.3 | 1.0 | 4 | 0.0 | 0 |  |
| `lid__muczdt` | `zinf` | broken | 0.3 | 1.0 | 4 | 0.0 | 0 |  |
| `lid__xdyddo` | `x2` | broken | 0.6 | 1.5 | 2 | 0.0 | 0 |  |
| `lid__xdyddo` | `y2` | broken | 0.6 | 1.4 | 2 | 0.0 | 0 |  |
| `lid__xdyddo` | `z3` | broken | 0.6 | 1.4 | 2 | 0.0 | 0 |  |
| `lid__xdyddo` | `z6` | broken | 0.6 | 1.3 | 2 | 0.0 | 0 |  |
| `lid__fnqhbz` | `z4` | broken | 0.2 | 1.0 | 7 | 0.0 | 0 |  |
| `lid__fnqhbz` | `z3` | broken | 0.3 | 1.0 | 7 | 0.0 | 0 |  |
| `lid__fnqhbz` | `z5` | broken | 0.3 | 1.0 | 7 | 0.0 | 0 |  |
| `lid__fnqhbz` | `z6` | broken | 0.3 | 1.1 | 7 | 0.0 | 0 |  |
| `lid__fnqhbz` | `zinf` | broken | 0.2 | 0.9 | 7 | 0.0 | 0 |  |
| `bottle_of_wine__bmudli` | `x2` | broken | 2.2 | 15.4 | 39 | 22.1 | 16 | +x |
| `bottle_of_wine__bmudli` | `z2` | broken | 1.6 | 9.6 | 26 | 20.1 | 18 | +x |
| `bottle_of_wine__bmudli` | `z4` | broken | 1.3 | 7.4 | 24 | 17.6 | 17 | +y |
| `bottle_of_wine__bmudli` | `z3` | broken | 1.4 | 9.0 | 25 | 22.5 | 17 | +y |
| `bottle_of_wine__bmudli` | `z5` | broken | 1.2 | 6.7 | 22 | 13.7 | 17 | +y |
| `bottle_of_wine__bmudli` | `z6` | broken | 1.1 | 6.6 | 22 | 9.7 | 16 | +x |
| `bottle_of_wine__bmudli` | `zinf` | broken | 1.6 | 10.1 | 26 | 25.3 | 17 | -x |
| `marker__zvvovh` | `xinf` | borderline | 0.5 | 2.7 | 5 | 0.7 | 11 | +x |
| `marker__slgheo` | `x2` | borderline | 0.9 | 4.1 | 12 | 1.7 | 16 | -x |
| `marker__slgheo` | `x4` | borderline | 0.7 | 3.2 | 8 | 1.4 | 15 | -x |
| `marker__slgheo` | `x3` | borderline | 0.7 | 3.5 | 9 | 1.4 | 17 | -x |
| `marker__slgheo` | `x5` | borderline | 0.6 | 3.0 | 7 | 1.3 | 13 | -x |
| `marker__slgheo` | `x6` | borderline | 0.6 | 2.7 | 6 | 1.2 | 11 | -x |
| `marker__slgheo` | `xinf` | borderline | 0.7 | 3.4 | 9 | 1.4 | 16 | -x |
| `marker__erhlbb` | `x2` | borderline | 0.4 | 2.2 | 5 | 1.0 | 12 | +x |
| `marker__erhlbb` | `x4` | borderline | 0.4 | 2.0 | 5 | 0.0 | 0 |  |
| `marker__erhlbb` | `x3` | borderline | 0.4 | 2.4 | 5 | 0.5 | 12 | +x |
| `marker__erhlbb` | `x5` | borderline | 0.3 | 1.8 | 4 | 0.0 | 0 |  |
| `marker__erhlbb` | `xinf` | borderline | 0.4 | 2.0 | 5 | 1.2 | 12 | +x |
| `marker__jlvneb` | `x2` | borderline | 0.6 | 3.3 | 7 | 1.2 | 13 | +x |
| `marker__jlvneb` | `x4` | borderline | 0.5 | 2.8 | 6 | 0.8 | 11 | -x |
| `marker__jlvneb` | `x3` | borderline | 0.6 | 3.1 | 7 | 0.9 | 12 | +x |
| `marker__jlvneb` | `x5` | borderline | 0.5 | 2.5 | 6 | 0.7 | 11 | -x |
| `marker__jlvneb` | `x6` | borderline | 0.4 | 2.4 | 5 | 0.4 | 11 | -x |
| `marker__jlvneb` | `xinf` | borderline | 0.6 | 2.9 | 7 | 1.4 | 13 | +x |
| `marker__kjxpdi` | `z2` | borderline | 0.4 | 2.3 | 5 | 0.0 | 0 |  |
| `marker__mcpkkq` | `x2` | borderline | 0.9 | 4.6 | 10 | 1.2 | 16 | +x |
| `marker__mcpkkq` | `x4` | borderline | 0.7 | 3.8 | 8 | 1.1 | 12 | -x |
| `marker__mcpkkq` | `x3` | borderline | 0.8 | 4.3 | 9 | 1.0 | 13 | -x |
| `marker__mcpkkq` | `x5` | borderline | 0.6 | 3.4 | 7 | 1.1 | 11 | -x |
| `marker__mcpkkq` | `x6` | borderline | 0.6 | 3.2 | 7 | 1.0 | 10 | -x |
| `marker__mcpkkq` | `xinf` | borderline | 0.8 | 3.9 | 9 | 1.4 | 15 | +x |
| `cap__ghwjwe` | `z2` | borderline | 1.0 | 4.2 | 9 | 0.5 | 9 | -y |
| `cap__ghwjwe` | `zinf` | borderline | 1.0 | 4.2 | 9 | 0.8 | 9 | +y |
| `cap__uxpeaz` | `z3` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `cap__uxpeaz` | `z5` | borderline | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `cap__uxpeaz` | `z6` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `cap__uxpeaz` | `zinf` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `can_of_soda__bfrzvk` | `z4` | borderline | 1.5 | 4.1 | 12 | 1.1 | 23 | -z |
| `can_of_soda__bfrzvk` | `z3` | borderline | 1.5 | 4.3 | 9 | 0.3 | 22 | -z |
| `can_of_soda__bfrzvk` | `z5` | borderline | 1.3 | 3.9 | 11 | 0.6 | 27 | -z |
| `can_of_soda__bfrzvk` | `zinf` | borderline | 1.6 | 4.4 | 12 | 0.9 | 21 | -z |
| `can_of_soda__wbrrad` | `z2` | broken | 1.7 | 7.4 | 15 | 9.3 | 22 | -z |
| `can_of_soda__wbrrad` | `z4` | broken | 1.4 | 5.8 | 14 | 8.8 | 30 | -x |
| `can_of_soda__wbrrad` | `z3` | broken | 2.1 | 9.5 | 23 | 10.0 | 29 | +y |
| `can_of_soda__wbrrad` | `z5` | broken | 2.0 | 8.5 | 23 | 9.9 | 28 | +y |
| `can_of_soda__wbrrad` | `z6` | broken | 1.9 | 7.8 | 22 | 8.8 | 29 | +y |
| `can_of_soda__wbrrad` | `zinf` | broken | 2.3 | 10.0 | 25 | 17.2 | 21 | -z |
| `bottle_of_beer__fgzjnb` | `z6` | broken | 2.5 | 16.3 | 25 | 43.3 | 25 | +y |
| `bottle_of_beer__zlmwyn` | `z2` | broken | 1.9 | 6.2 | 10 | 1.9 | 17 | +y |
| `bottle_of_beer__zlmwyn` | `z4` | borderline | 1.4 | 5.8 | 10 | 1.9 | 17 | -y |
| `bottle_of_beer__zlmwyn` | `z3` | broken | 1.6 | 5.9 | 10 | 1.9 | 17 | -y |
| `bottle_of_beer__zlmwyn` | `z5` | borderline | 1.2 | 5.5 | 10 | 1.9 | 17 | -y |
| `bottle_of_beer__zlmwyn` | `z6` | borderline | 1.1 | 5.4 | 10 | 1.9 | 17 | -y |
| `bottle_of_beer__zlmwyn` | `zinf` | broken | 1.9 | 6.2 | 10 | 1.9 | 17 | -y |
| `canned_food__foetdd` | `z2` | broken | 1.7 | 7.4 | 15 | 12.6 | 17 | -z |
| `lemon__bmvxas` | `x2` | broken | 1.3 | 3.0 | 6 | 7.6 | 26 | +x |
| `lemon__bmvxas` | `x3` | broken | 2.3 | 5.0 | 7 | 6.9 | 26 | +x |
| `lemon__bmvxas` | `x6` | broken | 2.3 | 4.9 | 7 | 5.9 | 18 | +x |
| `lemon__vuwehu` | `x2` | broken | 0.9 | 1.9 | 5 | 3.1 | 25 | +x |
| `lemon__vuwehu` | `y2` | borderline | 0.9 | 1.6 | 4 | 3.0 | 24 | -x |
| `lemon__vuwehu` | `z2` | broken | 0.9 | 1.7 | 4 | 1.2 | 31 | +x |
| `lemon__vuwehu` | `x4` | broken | 0.9 | 1.6 | 4 | 4.7 | 25 | +x |
| `lemon__vuwehu` | `x3` | broken | 0.9 | 1.9 | 4 | 3.0 | 24 | +x |
| `lemon__vuwehu` | `x5` | broken | 0.8 | 1.6 | 4 | 4.5 | 25 | +x |
| `lemon__vuwehu` | `x6` | broken | 0.8 | 1.6 | 4 | 4.2 | 26 | +x |
| `lemon__vuwehu` | `xinf` | broken | 0.9 | 1.7 | 5 | 4.7 | 25 | +x |
| `lemon__dsacfa` | `x2` | broken | 1.8 | 3.8 | 9 | 6.7 | 29 | +x |
| `lemon__asdada` | `x2` | broken | 1.3 | 3.1 | 5 | 6.1 | 16 | +x |
| `lemon__asdada` | `x4` | broken | 2.5 | 5.3 | 7 | 5.9 | 14 | +x |
| `lemon__asdada` | `x3` | broken | 2.3 | 5.0 | 7 | 5.5 | 16 | +x |
| `lemon__asdada` | `x5` | broken | 2.4 | 5.0 | 7 | 4.2 | 13 | +x |
| `lemon__asdada` | `x6` | borderline | 2.2 | 4.7 | 6 | 3.5 | 13 | +x |
| `lemon__asdada` | `xinf` | broken | 1.6 | 3.9 | 6 | 6.1 | 17 | +x |
| `lemon__gkrfha` | `x2` | broken | 1.5 | 3.2 | 8 | 8.3 | 29 | +x |
| `lemon__gkrfha` | `x3` | broken | 2.4 | 4.9 | 8 | 7.0 | 30 | +x |
| `lemon__gkrfha` | `x6` | broken | 2.3 | 4.9 | 7 | 6.1 | 23 | +x |
| `lemon__hoghjl` | `x2` | broken | 1.9 | 4.1 | 8 | 6.7 | 30 | +x |
| `lemon__kvgfdl` | `x2` | broken | 1.2 | 3.1 | 5 | 6.0 | 18 | +x |
| `lemon__kvgfdl` | `x4` | broken | 2.4 | 5.2 | 7 | 4.8 | 16 | +x |
| `lemon__kvgfdl` | `x3` | broken | 2.3 | 5.0 | 6 | 5.6 | 17 | +x |
| `lemon__kvgfdl` | `x5` | borderline | 2.3 | 5.0 | 6 | 2.1 | 15 | +x |
| `lemon__kvgfdl` | `x6` | borderline | 2.2 | 4.7 | 6 | 1.8 | 14 | +x |
| `lemon__peqwoa` | `x2` | broken | 1.5 | 3.6 | 7 | 9.2 | 27 | +x |
| `lemon__peqwoa` | `x3` | broken | 2.4 | 5.3 | 7 | 8.2 | 25 | +x |
| `lemon__peqwoa` | `x6` | broken | 2.3 | 5.1 | 7 | 7.1 | 19 | +x |
| `lemon__pffkae` | `x2` | broken | 1.5 | 3.6 | 6 | 9.5 | 26 | +x |
| `lemon__pffkae` | `x3` | broken | 2.4 | 5.2 | 7 | 8.7 | 24 | +x |
| `lemon__pffkae` | `x5` | broken | 2.4 | 5.3 | 7 | 7.3 | 20 | +x |
| `lemon__pffkae` | `x6` | broken | 2.3 | 4.9 | 7 | 7.0 | 19 | +x |
| `lemon__pqweio` | `x2` | broken | 1.9 | 4.2 | 8 | 8.5 | 30 | +x |
| `lemon__pqweio` | `x6` | broken | 2.3 | 4.9 | 7 | 6.5 | 27 | +x |
| `shampoo_dispenser__thurmw` | `z2` | broken | 0.2 | 0.2 | 2 | 4.2 | 18 | +z |
| `shampoo_dispenser__thurmw` | `z4` | broken | 0.2 | 0.3 | 1 | 4.1 | 18 | +z |
| `shampoo_dispenser__thurmw` | `z5` | broken | 0.5 | 3.1 | 9 | 3.3 | 21 | +z |
| `shampoo_dispenser__eblcnw` | `z2` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__eblcnw` | `z4` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__eblcnw` | `z3` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__eblcnw` | `z5` | broken | 0.2 | 0.8 | 3 | 0.0 | 0 |  |
| `shampoo_dispenser__eblcnw` | `z6` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `shampoo_dispenser__eblcnw` | `zinf` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `shampoo_dispenser__zoyyia` | `z2` | broken | 0.7 | 3.6 | 17 | 4.0 | 11 | +z |
| `shampoo_dispenser__zoyyia` | `z4` | borderline | 0.5 | 2.8 | 11 | 2.5 | 16 | +z |
| `shampoo_dispenser__zoyyia` | `z5` | broken | 0.5 | 2.7 | 9 | 1.9 | 16 | +z |
| `shampoo_dispenser__ndrvvm` | `z2` | broken | 0.2 | 1.1 | 5 | 0.0 | 0 |  |
| `shampoo_dispenser__ndrvvm` | `z4` | broken | 0.2 | 1.0 | 5 | 0.0 | 0 |  |
| `shampoo_dispenser__ndrvvm` | `z3` | broken | 0.2 | 1.1 | 5 | 0.0 | 0 |  |
| `shampoo_dispenser__ndrvvm` | `z5` | broken | 0.2 | 0.9 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__ndrvvm` | `z6` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__ndrvvm` | `zinf` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__nrthyl` | `z2` | borderline | 0.5 | 2.9 | 10 | 2.3 | 10 | +z |
| `shampoo_dispenser__nrthyl` | `z4` | borderline | 0.4 | 2.3 | 8 | 1.6 | 13 | +z |
| `shampoo_dispenser__nrthyl` | `z3` | broken | 0.4 | 3.0 | 9 | 1.6 | 13 | +z |
| `shampoo_dispenser__nrthyl` | `z5` | broken | 0.4 | 2.5 | 7 | 1.5 | 14 | +z |
| `shampoo_dispenser__nrthyl` | `z6` | broken | 0.3 | 2.2 | 7 | 1.6 | 14 | +z |
| `shampoo_dispenser__nrthyl` | `zinf` | broken | 0.3 | 2.2 | 7 | 1.7 | 14 | +z |
| `shampoo_dispenser__kkbhzw` | `z2` | broken | 0.4 | 1.2 | 6 | 0.0 | 0 |  |
| `shampoo_dispenser__kkbhzw` | `z4` | broken | 0.5 | 1.0 | 5 | 0.0 | 0 |  |
| `shampoo_dispenser__kkbhzw` | `z3` | broken | 0.5 | 1.1 | 6 | 0.0 | 0 |  |
| `shampoo_dispenser__kkbhzw` | `z5` | broken | 0.4 | 1.0 | 5 | 0.0 | 0 |  |
| `shampoo_dispenser__kkbhzw` | `z6` | broken | 0.4 | 1.0 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__kkbhzw` | `zinf` | broken | 0.5 | 1.1 | 5 | 0.0 | 0 |  |
| `shampoo_dispenser__eywylh` | `z2` | broken | 1.2 | 5.1 | 21 | 11.3 | 43 | +z |
| `shampoo_dispenser__eywylh` | `z4` | broken | 1.3 | 5.5 | 24 | 13.2 | 46 | +z |
| `beaker__effbnc` | `z6` | borderline | 2.8 | 8.4 | 17 | 1.9 | 17 | +z |
| `beaker__dtjmai` | `z3` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `beaker__dtjmai` | `z6` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `apple__obixxh` | `x2` | borderline | 2.1 | 5.6 | 10 | 2.3 | 16 | -x |
| `apple__obixxh` | `y2` | broken | 2.2 | 5.8 | 7 | 0.3 | 10 | -x |
| `apple__obixxh` | `z2` | broken | 2.2 | 5.7 | 10 | 2.3 | 16 | -x |
| `apple__obixxh` | `x4` | borderline | 2.4 | 5.8 | 10 | 2.3 | 16 | -x |
| `apple__obixxh` | `y4` | broken | 2.3 | 5.4 | 11 | 2.0 | 18 | -x |
| `apple__obixxh` | `z4` | broken | 2.4 | 5.8 | 10 | 2.3 | 16 | -y |
| `apple__obixxh` | `x3` | broken | 2.3 | 5.7 | 10 | 2.3 | 16 | -x |
| `apple__obixxh` | `y3` | broken | 2.2 | 5.6 | 10 | 2.2 | 15 | +x |
| `apple__obixxh` | `z3` | broken | 2.4 | 5.8 | 11 | 2.2 | 16 | -y |
| `apple__obixxh` | `x5` | borderline | 2.4 | 5.5 | 10 | 2.3 | 16 | -x |
| `apple__obixxh` | `y5` | broken | 2.2 | 5.3 | 10 | 2.1 | 16 | +z |
| `apple__obixxh` | `z5` | broken | 2.3 | 5.7 | 10 | 2.2 | 17 | -x |
| `apple__obixxh` | `x6` | broken | 2.3 | 5.4 | 10 | 4.7 | 16 | -x |
| `apple__obixxh` | `y6` | borderline | 2.2 | 5.2 | 11 | 2.0 | 17 | -x |
| `apple__obixxh` | `z6` | broken | 2.3 | 5.6 | 10 | 2.2 | 16 | -x |
| `apple__obixxh` | `xinf` | broken | 2.3 | 5.5 | 10 | 4.5 | 16 | -x |
| `apple__obixxh` | `yinf` | broken | 1.6 | 4.2 | 9 | 4.3 | 16 | -x |
| `apple__obixxh` | `zinf` | broken | 1.6 | 4.1 | 9 | 4.4 | 16 | -x |
| `apple__zlxfxt` | `x2` | borderline | 2.1 | 5.6 | 10 | 2.3 | 16 | -x |
| `apple__zlxfxt` | `y2` | broken | 2.2 | 5.8 | 7 | 0.2 | 13 | -x |
| `apple__zlxfxt` | `z2` | broken | 2.2 | 5.7 | 10 | 2.3 | 16 | -x |
| `apple__zlxfxt` | `x4` | borderline | 2.4 | 5.8 | 10 | 2.3 | 16 | -x |
| `apple__zlxfxt` | `y4` | broken | 2.3 | 5.4 | 11 | 2.0 | 18 | -x |
| `apple__zlxfxt` | `z4` | broken | 2.4 | 5.8 | 10 | 2.3 | 16 | -y |
| `apple__zlxfxt` | `x3` | broken | 2.4 | 5.7 | 10 | 2.3 | 16 | -x |
| `apple__zlxfxt` | `y3` | broken | 2.2 | 5.5 | 10 | 2.3 | 16 | +x |
| `apple__zlxfxt` | `z3` | broken | 2.4 | 5.7 | 11 | 2.3 | 16 | -y |
| `apple__zlxfxt` | `x5` | borderline | 2.4 | 5.5 | 10 | 2.3 | 16 | -x |
| `apple__zlxfxt` | `y5` | broken | 2.2 | 5.3 | 10 | 2.1 | 16 | +z |
| `apple__zlxfxt` | `z5` | broken | 2.3 | 5.7 | 10 | 2.1 | 17 | -x |
| `apple__zlxfxt` | `x6` | broken | 2.3 | 5.4 | 10 | 4.7 | 16 | -x |
| `apple__zlxfxt` | `y6` | broken | 2.2 | 5.2 | 11 | 2.0 | 17 | -x |
| `apple__zlxfxt` | `z6` | broken | 2.3 | 5.6 | 10 | 2.2 | 16 | -x |
| `apple__zlxfxt` | `yinf` | broken | 1.6 | 4.2 | 9 | 4.3 | 16 | -x |
| `apple__zlxfxt` | `zinf` | broken | 1.6 | 4.1 | 9 | 4.3 | 17 | -x |
| `beeswax_candle__nhdnje` | `z2` | borderline | 0.3 | 1.2 | 3 | 2.7 | 13 | +z |
| `beeswax_candle__nhdnje` | `z4` | borderline | 1.3 | 5.3 | 6 | 1.7 | 12 | +z |
| `beeswax_candle__nhdnje` | `z3` | borderline | 1.2 | 5.3 | 6 | 2.3 | 12 | +z |
| `beeswax_candle__nhdnje` | `z5` | borderline | 1.2 | 5.3 | 6 | 1.3 | 10 | +z |
| `beeswax_candle__nhdnje` | `z6` | borderline | 1.1 | 5.3 | 6 | 1.1 | 10 | +z |
| `beeswax_candle__nhdnje` | `zinf` | borderline | 0.6 | 3.0 | 4 | 2.7 | 13 | +z |
| `beeswax_candle__oimgmh` | `z2` | borderline | 0.4 | 1.7 | 3 | 1.2 | 13 | +z |
| `beeswax_candle__oimgmh` | `z3` | borderline | 0.9 | 4.9 | 6 | 0.5 | 13 | +z |
| `beeswax_candle__oimgmh` | `zinf` | borderline | 0.6 | 2.9 | 4 | 1.2 | 13 | +z |
| `beeswax_candle__ouzkdj` | `z2` | borderline | 0.9 | 4.6 | 5 | 1.6 | 12 | +z |
| `beeswax_candle__ouzkdj` | `z4` | borderline | 0.9 | 5.0 | 6 | 1.8 | 12 | +z |
| `beeswax_candle__ouzkdj` | `z3` | borderline | 0.9 | 5.0 | 6 | 2.0 | 12 | +z |
| `beeswax_candle__ouzkdj` | `z5` | borderline | 0.9 | 5.2 | 6 | 1.6 | 11 | +z |
| `beeswax_candle__ouzkdj` | `z6` | borderline | 0.9 | 5.2 | 6 | 1.4 | 11 | +z |
| `beeswax_candle__ouzkdj` | `zinf` | borderline | 1.0 | 5.0 | 6 | 1.9 | 13 | +z |
| `beeswax_candle__nxewyk` | `z2` | borderline | 0.5 | 3.3 | 6 | 2.8 | 15 | +z |
| `beeswax_candle__nxewyk` | `z4` | borderline | 0.6 | 3.9 | 6 | 2.1 | 13 | +z |
| `beeswax_candle__nxewyk` | `z3` | borderline | 0.6 | 4.0 | 6 | 2.5 | 14 | +z |
| `beeswax_candle__nxewyk` | `z5` | borderline | 0.6 | 3.8 | 6 | 1.7 | 12 | +z |
| `beeswax_candle__nxewyk` | `z6` | borderline | 0.5 | 3.6 | 6 | 1.1 | 12 | +z |
| `beeswax_candle__nxewyk` | `zinf` | borderline | 0.6 | 3.8 | 6 | 2.8 | 15 | +z |
| `beeswax_candle__pfewit` | `x2` | broken | 1.9 | 6.9 | 42 | 12.7 | 29 | +y |
| `beeswax_candle__pfewit` | `y2` | broken | 1.9 | 7.1 | 42 | 12.7 | 29 | -x |
| `beeswax_candle__pfewit` | `z2` | borderline | 0.3 | 1.2 | 3 | 1.3 | 14 | +z |
| `beeswax_candle__pfewit` | `z4` | borderline | 0.3 | 1.3 | 3 | 1.9 | 15 | +z |
| `beeswax_candle__pfewit` | `z3` | borderline | 0.3 | 1.3 | 3 | 1.9 | 15 | +z |
| `beeswax_candle__pfewit` | `z5` | borderline | 0.3 | 1.2 | 3 | 1.6 | 14 | +z |
| `beeswax_candle__pfewit` | `z6` | borderline | 0.3 | 1.2 | 3 | 1.4 | 13 | +z |
| `beeswax_candle__pfewit` | `zinf` | borderline | 0.3 | 1.3 | 3 | 2.0 | 15 | +z |
| `bottle_of_pop__twtsry` | `z4` | broken | 0.7 | 5.4 | 13 | 10.0 | 18 | -z |
| `bottle_of_pop__twtsry` | `z3` | broken | 0.6 | 4.3 | 11 | 8.9 | 17 | -z |
| `bottle_of_pop__twtsry` | `z5` | broken | 0.7 | 5.0 | 14 | 8.1 | 18 | -z |
| `bottle_of_pop__twtsry` | `z6` | broken | 0.7 | 4.8 | 13 | 7.1 | 18 | -z |
| `bottle_of_pop__twtsry` | `zinf` | broken | 0.7 | 5.2 | 13 | 10.3 | 17 | -z |
| `bell_pepper__ihctxa` | `x6` | broken | 2.0 | 4.0 | 30 | 17.3 | 28 | +x |
| `bottle_of_catsup__ahoiqe` | `x2` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `bottle_of_catsup__ahoiqe` | `y2` | broken | 0.2 | 0.8 | 3 | 0.0 | 0 |  |
| `bottle_of_catsup__bcqfxb` | `z2` | borderline | 0.3 | 0.9 | 10 | 2.6 | 9 | +x |
| `bar_soap__feqemg` | `z2` | borderline | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `beefsteak_tomato__pnrdxh` | `x2` | broken | 0.9 | 3.5 | 24 | 15.2 | 36 | +z |
| `beefsteak_tomato__pnrdxh` | `y2` | broken | 0.9 | 3.7 | 25 | 15.2 | 36 | +z |
| `beefsteak_tomato__pnrdxh` | `z2` | broken | 0.7 | 3.3 | 22 | 20.4 | 31 | +z |
| `beefsteak_tomato__pnrdxh` | `z4` | broken | 0.6 | 3.0 | 18 | 19.1 | 27 | +z |
| `beefsteak_tomato__pnrdxh` | `z3` | broken | 0.7 | 3.1 | 21 | 19.6 | 29 | +z |
| `beefsteak_tomato__pnrdxh` | `z5` | broken | 0.6 | 2.7 | 16 | 18.4 | 25 | +z |
| `beefsteak_tomato__pnrdxh` | `z6` | broken | 0.5 | 2.6 | 14 | 18.0 | 22 | +z |
| `beefsteak_tomato__pnrdxh` | `zinf` | broken | 0.7 | 3.4 | 22 | 20.5 | 31 | +z |
| `beefsteak_tomato__altlfz` | `x2` | broken | 2.0 | 4.5 | 6 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `y2` | broken | 2.1 | 4.3 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `z2` | broken | 2.2 | 4.0 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `x4` | broken | 2.1 | 4.4 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `y4` | broken | 2.0 | 4.1 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `z4` | broken | 2.0 | 4.1 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `x3` | broken | 2.0 | 3.9 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `y3` | broken | 2.1 | 4.4 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `z3` | broken | 2.2 | 4.3 | 6 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `x5` | broken | 2.2 | 4.5 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `y5` | broken | 2.1 | 3.9 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `z5` | broken | 2.0 | 4.1 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `x6` | broken | 2.2 | 4.5 | 6 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `y6` | broken | 2.1 | 4.0 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `z6` | broken | 2.1 | 4.3 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `xinf` | broken | 2.2 | 4.4 | 6 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `yinf` | broken | 2.0 | 3.9 | 5 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `zinf` | broken | 2.2 | 4.3 | 6 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `full` | broken | 2.1 | 4.3 | 6 | 0.0 | 0 |  |
| `beefsteak_tomato__ogpans` | `y2` | broken | 1.3 | 8.1 | 28 | 14.0 | 25 | +z |
| `beefsteak_tomato__ogpans` | `z2` | broken | 1.5 | 12.3 | 28 | 21.5 | 24 | +z |
| `bottle_of_olive_oil__cqycjk` | `z2` | broken | 1.3 | 9.3 | 29 | 18.0 | 26 | +x |
| `bottle_of_olive_oil__cqycjk` | `z4` | broken | 1.1 | 7.1 | 28 | 9.4 | 23 | -x |
| `bottle_of_olive_oil__cqycjk` | `z3` | broken | 0.9 | 3.7 | 26 | 9.4 | 23 | -x |
| `bottle_of_olive_oil__cqycjk` | `z5` | broken | 1.4 | 9.8 | 29 | 17.1 | 26 | +x |
| `bottle_of_olive_oil__cqycjk` | `z6` | broken | 1.4 | 10.4 | 29 | 19.4 | 27 | +x |
| `bottle_of_olive_oil__cqycjk` | `zinf` | broken | 1.4 | 10.3 | 29 | 23.9 | 24 | -y |
| `bottle_of_olive_oil__ksxqkk` | `z2` | broken | 1.6 | 9.0 | 17 | 15.8 | 24 | -x |
| `spray_can__bgalkx` | `x2` | broken | 0.2 | 0.6 | 4 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `y2` | broken | 0.2 | 0.7 | 4 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `z2` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `z4` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `z3` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `z5` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `zinf` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `spray_can__tehmzy` | `z2` | broken | 1.2 | 2.9 | 7 | 4.0 | 11 | +z |
| `spray_can__tehmzy` | `z4` | borderline | 1.4 | 3.1 | 6 | 1.9 | 10 | +z |
| `spray_can__tehmzy` | `z3` | borderline | 1.5 | 3.4 | 7 | 3.0 | 10 | +z |
| `spray_can__tehmzy` | `zinf` | broken | 1.4 | 3.3 | 8 | 4.0 | 11 | +z |
| `vidalia_onion__dejqhf` | `x2` | borderline | 1.5 | 3.5 | 5 | 1.2 | 14 | +x |
| `vidalia_onion__dejqhf` | `y2` | broken | 1.8 | 4.6 | 10 | 1.2 | 15 | -x |
| `vidalia_onion__dejqhf` | `z2` | broken | 1.9 | 4.6 | 10 | 1.1 | 13 | +x |
| `vidalia_onion__dejqhf` | `x4` | borderline | 1.7 | 4.0 | 5 | 1.1 | 13 | +x |
| `vidalia_onion__dejqhf` | `y4` | broken | 1.8 | 4.5 | 10 | 1.2 | 13 | -z |
| `vidalia_onion__dejqhf` | `z4` | broken | 1.8 | 4.4 | 10 | 1.2 | 13 | +y |
| `vidalia_onion__dejqhf` | `x3` | borderline | 1.6 | 3.9 | 5 | 0.6 | 10 | +x |
| `vidalia_onion__dejqhf` | `y3` | broken | 1.9 | 4.6 | 9 | 1.0 | 13 | +x |
| `vidalia_onion__dejqhf` | `z3` | broken | 1.9 | 4.8 | 11 | 1.8 | 14 | +y |
| `vidalia_onion__dejqhf` | `x5` | borderline | 1.6 | 3.8 | 5 | 1.4 | 13 | +x |
| `vidalia_onion__dejqhf` | `y5` | broken | 1.7 | 4.2 | 10 | 1.3 | 13 | -z |
| `vidalia_onion__dejqhf` | `z5` | broken | 1.8 | 4.4 | 11 | 1.3 | 12 | +x |
| `vidalia_onion__dejqhf` | `x6` | borderline | 1.6 | 3.9 | 5 | 1.3 | 13 | +x |
| `vidalia_onion__dejqhf` | `y6` | broken | 1.7 | 4.0 | 10 | 1.1 | 12 | +x |
| `vidalia_onion__dejqhf` | `z6` | broken | 1.7 | 4.3 | 10 | 1.3 | 13 | +x |
| `vidalia_onion__dejqhf` | `xinf` | borderline | 1.6 | 3.8 | 5 | 1.4 | 15 | +x |
| `vidalia_onion__dejqhf` | `yinf` | broken | 1.4 | 3.4 | 9 | 0.9 | 16 | +x |
| `vidalia_onion__dejqhf` | `zinf` | broken | 1.6 | 3.8 | 10 | 1.3 | 13 | +x |
| `vidalia_onion__dejqhf` | `full` | broken | 1.6 | 4.1 | 9 | 1.1 | 14 | +x |
| `pack_of_cigarettes__kienyo` | `z2` | broken | 1.3 | 2.8 | 12 | 4.5 | 17 | +x |
| `pack_of_cigarettes__wjfbhw` | `z2` | borderline | 1.7 | 4.9 | 7 | 1.8 | 12 | +x |
| `orange__ucstpm` | `x2` | broken | 1.8 | 3.3 | 8 | 4.6 | 22 | +z |
| `orange__ucstpm` | `y2` | broken | 1.8 | 3.7 | 8 | 4.6 | 22 | -z |
| `orange__ucstpm` | `z2` | broken | 1.7 | 3.4 | 5 | 6.5 | 16 | +z |
| `orange__ucstpm` | `x4` | broken | 1.6 | 3.3 | 10 | 4.4 | 22 | +z |
| `orange__ucstpm` | `y4` | broken | 2.0 | 3.9 | 8 | 4.5 | 22 | +z |
| `orange__ucstpm` | `z4` | broken | 1.8 | 3.9 | 5 | 5.7 | 14 | +z |
| `orange__ucstpm` | `x3` | broken | 1.7 | 3.4 | 10 | 4.7 | 22 | +z |
| `orange__ucstpm` | `y3` | broken | 1.9 | 3.8 | 9 | 4.4 | 22 | +z |
| `orange__ucstpm` | `z3` | broken | 1.8 | 3.6 | 6 | 6.2 | 16 | +z |
| `orange__ucstpm` | `x5` | broken | 1.7 | 3.5 | 11 | 4.7 | 22 | -y |
| `orange__ucstpm` | `y5` | broken | 2.0 | 3.9 | 10 | 4.8 | 22 | +x |
| `orange__ucstpm` | `z5` | broken | 1.8 | 3.8 | 5 | 5.2 | 13 | +z |
| `orange__ucstpm` | `x6` | broken | 1.7 | 3.7 | 12 | 4.8 | 23 | -y |
| `orange__ucstpm` | `y6` | broken | 1.9 | 3.8 | 12 | 5.0 | 22 | +x |
| `orange__ucstpm` | `z6` | borderline | 1.7 | 3.6 | 5 | 2.5 | 12 | +z |
| `orange__ucstpm` | `xinf` | broken | 1.2 | 2.7 | 12 | 10.2 | 22 | +z |
| `orange__ucstpm` | `yinf` | broken | 1.2 | 2.3 | 12 | 9.6 | 22 | +z |
| `orange__ucstpm` | `zinf` | broken | 1.7 | 3.4 | 5 | 6.6 | 16 | +z |
| `orange__ucstpm` | `full` | broken | 1.7 | 3.5 | 12 | 5.2 | 22 | -y |
| `bottle_of_salsa__nafwlf` | `z2` | broken | 1.2 | 4.5 | 8 | 6.2 | 12 | +z |
| `bottle_of_salsa__nafwlf` | `z4` | borderline | 1.1 | 4.3 | 8 | 1.8 | 14 | +x |
| `bottle_of_salsa__nafwlf` | `z3` | borderline | 1.3 | 4.4 | 8 | 2.3 | 10 | +z |
| `bottle_of_salsa__nafwlf` | `z5` | borderline | 1.2 | 4.3 | 8 | 1.5 | 11 | +z |
| `bottle_of_salsa__nafwlf` | `z6` | borderline | 1.3 | 4.3 | 7 | 1.5 | 13 | +x |
| `bottle_of_salsa__nafwlf` | `zinf` | broken | 1.3 | 4.4 | 8 | 7.2 | 11 | +z |
| `coffee_cup__dkxddg` | `z4` | broken | 1.1 | 6.4 | 19 | 9.8 | 22 | +z |
| `coffee_cup__dkxddg` | `z3` | broken | 1.2 | 7.6 | 21 | 9.7 | 23 | +z |
| `coffee_cup__dkxddg` | `z5` | broken | 1.0 | 5.6 | 17 | 9.5 | 20 | +z |
| `coffee_cup__dkxddg` | `z6` | broken | 0.9 | 4.8 | 14 | 9.2 | 18 | +z |
| `coffee_cup__ykuftq` | `x2` | broken | 1.0 | 5.2 | 13 | 7.8 | 17 | +y |
| `coffee_cup__rypdvd` | `z4` | broken | 0.4 | 2.8 | 8 | 4.5 | 12 | +z |
| `coffee_cup__rypdvd` | `z3` | broken | 0.5 | 3.1 | 9 | 4.5 | 12 | +z |
| `coffee_cup__rypdvd` | `z5` | broken | 0.4 | 2.5 | 7 | 7.0 | 12 | +z |
| `coffee_cup__rypdvd` | `z6` | broken | 0.3 | 2.3 | 6 | 4.1 | 11 | +z |
| `coffee_cup__fgizgn` | `z4` | broken | 0.1 | 0.9 | 3 | 0.0 | 0 |  |
| `coffee_cup__fgizgn` | `z5` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `coffee_cup__fgizgn` | `z6` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `z2` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `z4` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `z3` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `z5` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `z6` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `zinf` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `z2` | broken | 0.1 | 0.9 | 3 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `z4` | broken | 0.1 | 0.9 | 3 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `z3` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `z5` | broken | 0.1 | 0.9 | 3 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `z6` | broken | 0.1 | 0.9 | 3 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `zinf` | broken | 0.1 | 1.0 | 3 | 0.0 | 0 |  |
| `mug__ppzttc` | `z4` | broken | 1.2 | 7.6 | 11 | 7.4 | 13 | +x |
| `mug__ppzttc` | `z3` | broken | 1.1 | 7.5 | 10 | 4.0 | 13 | -y |
| `mug__ppzttc` | `z5` | broken | 1.2 | 8.0 | 15 | 16.0 | 20 | -y |
| `mug__ppzttc` | `z6` | broken | 1.2 | 8.2 | 15 | 13.1 | 21 | -y |
| `mug__lgxhsc` | `z2` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `mug__lgxhsc` | `z4` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `mug__lgxhsc` | `z3` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `mug__lgxhsc` | `z5` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `mug__lgxhsc` | `z6` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `mug__lgxhsc` | `zinf` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `mug__waqrdy` | `z5` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__waqrdy` | `z6` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__ehnmxj` | `z2` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `mug__ehnmxj` | `z4` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `mug__ehnmxj` | `z3` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `mug__ehnmxj` | `z5` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `mug__ehnmxj` | `z6` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__ehnmxj` | `zinf` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `mug__fapsrj` | `z2` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `mug__fapsrj` | `z4` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `mug__fapsrj` | `z3` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `mug__fapsrj` | `z5` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__fapsrj` | `z6` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__fapsrj` | `zinf` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `mug__ntgftr` | `z4` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__ntgftr` | `z5` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__ntgftr` | `z6` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__ppdqbj` | `y2` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `mug__jgethp` | `z2` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `mug__jgethp` | `z4` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__jgethp` | `z3` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `mug__jgethp` | `z5` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__jgethp` | `z6` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__jgethp` | `zinf` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `statue__cxsvwj` | `z2` | broken | 0.2 | 0.9 | 5 | 0.0 | 0 |  |
| `statue__ponfvo` | `z2` | broken | 2.1 | 12.7 | 19 | 13.0 | 26 | +z |
| `mouse__cduwda` | `z2` | borderline | 0.5 | 0.9 | 12 | 3.3 | 19 | -x |
| `mouse__fjljxp` | `x2` | broken | 0.2 | 1.0 | 4 | 0.0 | 0 |  |
| `mouse__fjljxp` | `z2` | broken | 0.2 | 1.4 | 5 | 0.0 | 0 |  |
| `mouse__pljjtu` | `z2` | borderline | 0.5 | 1.0 | 13 | 3.0 | 19 | +x |
| `mouse__ihpnjx` | `x2` | broken | 0.2 | 1.6 | 5 | 0.0 | 0 |  |
| `mouse__eosngb` | `z2` | broken | 0.5 | 1.2 | 17 | 0.6 | 9 | +x |
| `quartz__klgses` | `x2` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `quartz__klgses` | `y2` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `quartz__klgses` | `z2` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `quartz__klgses` | `y4` | broken | 0.0 | 0.3 | 1 | 0.0 | 0 |  |
| `quartz__klgses` | `y3` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `quartz__klgses` | `y5` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `quartz__klgses` | `y6` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `quartz__klgses` | `yinf` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `quartz__kilkpz` | `x2` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__kilkpz` | `y2` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__kilkpz` | `z2` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__kilkpz` | `y4` | broken | 0.0 | 0.3 | 1 | 0.0 | 0 |  |
| `quartz__kilkpz` | `y3` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__kilkpz` | `y5` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `quartz__kilkpz` | `y6` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__vfuilh` | `y5` | broken | 0.0 | 0.3 | 1 | 0.0 | 0 |  |
| `quartz__vfuilh` | `y6` | broken | 0.0 | 0.3 | 1 | 0.0 | 0 |  |
| `quartz__drqtci` | `x2` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__drqtci` | `y2` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `quartz__drqtci` | `z2` | borderline | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__drqtci` | `y4` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__drqtci` | `y3` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `quartz__drqtci` | `y5` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__drqtci` | `y6` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `quartz__drqtci` | `yinf` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__poplsu` | `x2` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__poplsu` | `y2` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__poplsu` | `z2` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__poplsu` | `y3` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__poplsu` | `y5` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__poplsu` | `y6` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `quartz__poplsu` | `yinf` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `teacup__oxfzfe` | `z2` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `teacup__oxfzfe` | `z4` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `teacup__oxfzfe` | `z3` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `teacup__oxfzfe` | `z5` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `teacup__oxfzfe` | `z6` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `teacup__oxfzfe` | `zinf` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `teacup__vckahe` | `z5` | broken | 1.5 | 10.2 | 22 | 10.6 | 11 | -y |
| `teacup__vckahe` | `z6` | broken | 1.3 | 8.9 | 19 | 9.2 | 10 | -y |
| `teacup__cpozxi` | `z5` | borderline | 1.3 | 5.0 | 9 | 2.5 | 14 | -z |
| `teacup__wopjex` | `z2` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `teacup__wopjex` | `z4` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `teacup__wopjex` | `z3` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `teacup__wopjex` | `z5` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `teacup__wopjex` | `z6` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `teacup__wopjex` | `zinf` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `carving_knife__fqqbop` | `x2` | broken | 0.6 | 3.7 | 9 | 0.0 | 0 |  |
| `carving_knife__fqqbop` | `x6` | broken | 0.7 | 4.0 | 8 | 0.0 | 0 |  |
| `carving_knife__zaihmr` | `x6` | broken | 0.6 | 3.8 | 8 | 0.0 | 0 |  |
| `carving_knife__awvoox` | `x2` | broken | 0.8 | 4.0 | 15 | 3.7 | 14 | -x |
| `carving_knife__awvoox` | `x6` | broken | 0.6 | 3.5 | 8 | 0.9 | 9 | -x |
| `hook__akwlam` | `z2` | broken | 0.1 | 0.4 | 1 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `x2` | broken | 1.2 | 3.0 | 5 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `x4` | broken | 0.8 | 2.7 | 4 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `x3` | broken | 0.9 | 2.6 | 5 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `x5` | borderline | 0.8 | 2.7 | 4 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `x6` | borderline | 0.7 | 2.4 | 4 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `xinf` | broken | 0.9 | 2.7 | 5 | 0.0 | 0 |  |
| `briefcase__lwbctq` | `x2` | broken | 0.2 | 0.3 | 3 | 4.9 | 18 | -x |
| `briefcase__psvdau` | `z2` | broken | 0.3 | 0.4 | 12 | 0.5 | 10 | -y |
| `cookie_cutter__lqrfzo` | `z2` | borderline | 1.6 | 10.0 | 30 | 2.7 | 18 | +x |
| `hammer__fcrlbo` | `x2` | broken | 1.1 | 7.5 | 17 | 0.7 | 12 | -x |
| `bottle_of_wine__oldpru` | `z6` | broken | 2.5 | 11.7 | 12 | 0.0 | 0 |  |
| `box_of_cookies__nirfva` | `y2` | broken | 0.1 | 0.3 | 1 | 0.0 | 0 |  |
| `notebook__zkmmnf` | `z2` | borderline | 0.6 | 3.7 | 14 | 0.3 | 13 | +y |
| `soap_dispenser__zwudcz` | `z2` | borderline | 0.8 | 4.8 | 9 | 1.2 | 11 | +z |
| `soap_dispenser__cgxrwd` | `z2` | broken | 0.2 | 0.9 | 4 | 0.0 | 0 |  |
| `soap_dispenser__cgxrwd` | `z4` | broken | 0.2 | 0.7 | 3 | 0.0 | 0 |  |
| `soap_dispenser__cgxrwd` | `z3` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `soap_dispenser__cgxrwd` | `z5` | broken | 0.2 | 0.7 | 4 | 0.0 | 0 |  |
| `soap_dispenser__cgxrwd` | `z6` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `soap_dispenser__cgxrwd` | `zinf` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `soap_dispenser__sinntg` | `x2` | broken | 2.3 | 17.1 | 47 | 32.5 | 34 | -x |
| `soap_dispenser__sinntg` | `z2` | broken | 2.1 | 11.7 | 48 | 17.5 | 30 | +z |
| `soap_dispenser__ijxawq` | `z2` | broken | 0.7 | 2.9 | 18 | 10.3 | 24 | +z |
| `soap_dispenser__ijxawq` | `z4` | broken | 0.6 | 2.4 | 13 | 4.8 | 40 | +z |
| `soap_dispenser__ijxawq` | `z3` | broken | 0.7 | 2.7 | 15 | 4.7 | 40 | +z |
| `soap_dispenser__ijxawq` | `z5` | broken | 0.5 | 1.9 | 10 | 4.7 | 41 | +z |
| `soap_dispenser__ijxawq` | `z6` | broken | 0.5 | 2.0 | 9 | 4.8 | 40 | +z |
| `soap_dispenser__ijxawq` | `zinf` | broken | 0.9 | 3.0 | 20 | 11.5 | 26 | +z |
| `soap_dispenser__jmulpc` | `z2` | broken | 0.6 | 1.7 | 21 | 41.2 | 27 | +z |
| `bottle_of_milk__utfsmp` | `x2` | broken | 1.6 | 8.6 | 23 | 12.6 | 24 | +x |
| `bottle_of_milk__utfsmp` | `x4` | broken | 1.6 | 8.6 | 22 | 9.3 | 22 | +x |
| `bottle_of_milk__utfsmp` | `x5` | broken | 1.6 | 9.1 | 21 | 6.7 | 23 | +x |
| `cellulose_tape__gchdhk` | `z6` | broken | 2.0 | 12.7 | 23 | 7.9 | 19 | +x |
| `conditioner_dispenser__njqnuo` | `x2` | broken | 0.4 | 1.1 | 5 | 0.0 | 0 |  |
| `conditioner_dispenser__njqnuo` | `x4` | broken | 0.6 | 1.1 | 5 | 0.0 | 0 |  |
| `conditioner_dispenser__njqnuo` | `x3` | broken | 0.6 | 1.1 | 5 | 0.0 | 0 |  |
| `conditioner_dispenser__njqnuo` | `x5` | broken | 0.6 | 1.1 | 5 | 0.0 | 0 |  |
| `conditioner_dispenser__njqnuo` | `x6` | broken | 0.5 | 1.0 | 4 | 0.0 | 0 |  |
| `conditioner_dispenser__njqnuo` | `xinf` | broken | 0.4 | 1.1 | 5 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `x2` | broken | 0.2 | 1.2 | 4 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `z2` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `z4` | broken | 0.2 | 0.9 | 3 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `z3` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `z5` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `z6` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `zinf` | broken | 0.2 | 0.9 | 3 | 0.0 | 0 |  |
| `pack_of_cigarettes__qabdgg` | `z2` | borderline | 0.8 | 1.9 | 9 | 3.2 | 15 | +x |
| `pack_of_cigarettes__rmrubd` | `z2` | broken | 1.9 | 8.3 | 40 | 59.3 | 37 | +x |
| `pencil_holder__jkuiqg` | `x2` | broken | 0.1 | 0.3 | 1 | 0.0 | 0 |  |
| `pencil_holder__jkuiqg` | `y2` | broken | 0.1 | 0.3 | 1 | 0.0 | 0 |  |
| `jar_of_bath_salt__dtrjxi` | `z2` | broken | 0.5 | 1.6 | 14 | 9.9 | 10 | +z |
| `jar_of_bath_salt__dtrjxi` | `z5` | broken | 0.3 | 1.2 | 7 | 0.0 | 0 |  |
| `bottle_of_wine__dijgvu` | `z2` | broken | 2.0 | 13.4 | 26 | 17.0 | 24 | -z |
| `teaspoon__mgdfcz` | `x2` | broken | 0.4 | 2.4 | 4 | 0.0 | 0 |  |
| `teaspoon__mgdfcz` | `z2` | broken | 0.6 | 2.9 | 4 | 0.0 | 0 |  |
| `notebook__ffdjzp` | `x2` | broken | 1.2 | 6.0 | 7 | 5.5 | 29 | -x |
| `notebook__ffdjzp` | `y2` | broken | 1.0 | 6.0 | 7 | 7.1 | 34 | -x |
| `notebook__ffdjzp` | `z2` | broken | 1.3 | 6.0 | 8 | 4.6 | 32 | -x |
| `notebook__gqkaqu` | `x2` | borderline | 0.7 | 5.1 | 14 | 0.3 | 16 | -x |
| `notebook__gqkaqu` | `y2` | broken | 0.9 | 5.3 | 21 | 23.8 | 17 | +y |
| `notebook__gqkaqu` | `z2` | broken | 1.3 | 8.3 | 30 | 24.5 | 21 | +y |
| `bottle_of_pop__wuuoes` | `z4` | borderline | 0.3 | 1.2 | 4 | 2.3 | 18 | +z |
| `bottle_of_pop__wuuoes` | `z3` | borderline | 0.7 | 3.9 | 11 | 3.8 | 17 | +z |
| `bottle_of_pop__wuuoes` | `z5` | broken | 0.7 | 4.1 | 11 | 4.8 | 17 | +z |
| `bottle_of_pop__wuuoes` | `z6` | broken | 0.7 | 3.8 | 10 | 4.6 | 17 | +z |
| `bottle_of_pop__wuuoes` | `zinf` | broken | 0.5 | 2.8 | 8 | 4.4 | 17 | +z |
| `bottle_of_pop__zjbqvb` | `z4` | borderline | 0.2 | 1.0 | 2 | 2.0 | 17 | +z |
| `bottle_of_pop__zjbqvb` | `z3` | broken | 0.5 | 2.7 | 10 | 3.9 | 19 | -z |
| `bottle_of_pop__zjbqvb` | `z5` | broken | 0.5 | 2.8 | 11 | 4.2 | 20 | -z |
| `bottle_of_pop__zjbqvb` | `z6` | broken | 0.5 | 2.6 | 10 | 4.0 | 19 | -z |
| `bottle_of_pop__zjbqvb` | `zinf` | broken | 0.6 | 2.9 | 11 | 4.5 | 19 | -z |
| `jar_of_bath_salt__erglwp` | `z5` | broken | 0.4 | 1.5 | 2 | 0.0 | 0 |  |
| `bottle_of_champagne__npsevx` | `z6` | broken | 2.3 | 14.6 | 46 | 38.5 | 44 | -z |
| `bottle_of_whiskey__aipxlu` | `x2` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `bottle_of_whiskey__aipxlu` | `y2` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `bottle_of_whiskey__aipxlu` | `z5` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `bottle_of_whiskey__hejqyo` | `x2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_whiskey__hejqyo` | `y2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_whiskey__hejqyo` | `z5` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `case_of_eyeshadow__zgervc` | `x2` | borderline | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `hardback__acaxtl` | `x2` | broken | 0.4 | 2.9 | 6 | 0.0 | 0 |  |
| `hardback__acaxtl` | `z2` | broken | 0.4 | 3.1 | 7 | 0.0 | 0 |  |
| `hardback__aceozs` | `y2` | broken | 0.9 | 5.6 | 19 | 33.6 | 18 | -y |
| `can_of_corn__alphlq` | `z2` | borderline | 0.9 | 3.5 | 7 | 1.1 | 12 | +z |
| `can_of_corn__kwwlfn` | `z2` | borderline | 1.5 | 6.8 | 10 | 0.4 | 11 | +z |
| `water_glass__cdteyb` | `z2` | broken | 0.6 | 3.2 | 6 | 0.0 | 0 |  |
| `water_glass__cdteyb` | `z4` | broken | 0.6 | 3.0 | 7 | 0.0 | 0 |  |
| `water_glass__cdteyb` | `z3` | broken | 0.7 | 3.4 | 8 | 0.0 | 0 |  |
| `water_glass__cdteyb` | `z5` | broken | 0.6 | 3.0 | 7 | 0.0 | 0 |  |
| `water_glass__cdteyb` | `z6` | broken | 0.6 | 3.0 | 7 | 0.0 | 0 |  |
| `water_glass__cdteyb` | `zinf` | broken | 0.6 | 3.0 | 7 | 0.0 | 0 |  |
| `water_glass__elwfms` | `x2` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `water_glass__elwfms` | `y2` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `bottle_of_ginger_beer__zkocwb` | `z2` | broken | 1.8 | 12.7 | 21 | 5.4 | 15 | +x |
| `lemonade_bottle__yprkek` | `z4` | broken | 1.2 | 6.7 | 9 | 0.0 | 0 |  |
| `lemonade_bottle__yprkek` | `z5` | broken | 0.6 | 5.4 | 8 | 0.0 | 0 |  |
| `soda_water_bottle__upfssc` | `z4` | broken | 1.2 | 6.7 | 9 | 0.0 | 0 |  |
| `soda_water_bottle__upfssc` | `z5` | broken | 0.6 | 5.4 | 8 | 0.0 | 0 |  |
| `marker__ablbdc` | `x4` | borderline | 0.4 | 2.1 | 4 | 0.0 | 0 |  |
| `marker__ablbdc` | `x3` | broken | 0.4 | 2.3 | 5 | 0.0 | 0 |  |
| `marker__ablbdc` | `xinf` | broken | 0.4 | 2.0 | 4 | 0.0 | 0 |  |
| `marker__bwwkmb` | `x2` | broken | 0.6 | 3.2 | 8 | 1.2 | 15 | +x |
| `marker__bwwkmb` | `x4` | broken | 0.5 | 2.7 | 7 | 0.9 | 11 | -x |
| `marker__bwwkmb` | `x3` | broken | 0.6 | 3.2 | 8 | 0.9 | 13 | +x |
| `marker__bwwkmb` | `x5` | broken | 0.4 | 2.4 | 6 | 0.8 | 11 | -x |
| `marker__bwwkmb` | `x6` | borderline | 0.4 | 2.3 | 6 | 0.7 | 10 | -x |
| `marker__bwwkmb` | `xinf` | broken | 0.7 | 3.3 | 9 | 1.0 | 14 | +x |
| `table_knife__eebpap` | `x2` | broken | 0.4 | 2.5 | 5 | 0.0 | 0 |  |
| `table_knife__eebpap` | `y2` | broken | 0.5 | 2.8 | 5 | 0.0 | 0 |  |
| `table_knife__eebpap` | `z2` | broken | 0.3 | 1.9 | 4 | 0.0 | 0 |  |
| `table_knife__eebpap` | `x3` | broken | 0.6 | 3.2 | 6 | 0.0 | 0 |  |
| `table_knife__eebpap` | `x5` | broken | 0.5 | 3.0 | 6 | 0.0 | 0 |  |
| `table_knife__eebpap` | `x6` | broken | 0.5 | 2.8 | 5 | 0.0 | 0 |  |
| `table_knife__eebpap` | `xinf` | broken | 0.6 | 3.0 | 6 | 0.0 | 0 |  |
| `table_knife__jxdfyy` | `x2` | broken | 0.4 | 2.1 | 5 | 0.0 | 0 |  |
| `table_knife__jxdfyy` | `y2` | broken | 0.5 | 3.1 | 6 | 0.0 | 0 |  |
| `table_knife__jxdfyy` | `z2` | broken | 0.5 | 2.9 | 6 | 0.0 | 0 |  |
| `pen__gzrbrx` | `x2` | borderline | 0.6 | 2.6 | 6 | 1.0 | 11 | -x |
| `pen__gzrbrx` | `x4` | borderline | 0.5 | 2.2 | 6 | 2.0 | 13 | -x |
| `pen__gzrbrx` | `x3` | borderline | 0.5 | 2.5 | 7 | 2.0 | 14 | -x |
| `pen__gzrbrx` | `x5` | borderline | 0.4 | 2.1 | 6 | 1.4 | 12 | -x |
| `pen__gzrbrx` | `x6` | borderline | 0.4 | 1.9 | 5 | 0.5 | 11 | -x |
| `pen__gzrbrx` | `xinf` | borderline | 0.6 | 2.6 | 7 | 1.3 | 11 | -x |
| `table_knife__lrdmpf` | `x3` | broken | 0.6 | 2.7 | 4 | 0.0 | 0 |  |
| `table_knife__lrdmpf` | `x6` | broken | 0.5 | 2.4 | 4 | 0.0 | 0 |  |
| `clipper__befwbq` | `x2` | broken | 1.4 | 4.2 | 14 | 0.0 | 0 |  |
| `clipper__befwbq` | `y2` | broken | 1.6 | 5.2 | 13 | 0.0 | 0 |  |
| `clipper__befwbq` | `z2` | broken | 1.5 | 4.9 | 11 | 0.0 | 0 |  |
| `clipper__befwbq` | `x4` | broken | 2.0 | 5.4 | 13 | 0.0 | 0 |  |
| `clipper__befwbq` | `x3` | broken | 2.0 | 5.3 | 14 | 0.0 | 0 |  |
| `lipstick__albshz` | `x2` | broken | 2.1 | 12.4 | 31 | 10.2 | 11 | +z |
| `lipstick__albshz` | `y2` | broken | 1.9 | 12.1 | 28 | 10.2 | 11 | +z |
| `lipstick__albshz` | `y4` | broken | 2.1 | 12.3 | 30 | 10.2 | 11 | +z |
| `lipstick__albshz` | `y3` | broken | 2.2 | 11.9 | 29 | 10.2 | 11 | +z |
| `lock__pngvzn` | `x2` | broken | 0.5 | 2.9 | 10 | 0.0 | 0 |  |
| `soap_dish__gnndfy` | `z4` | broken | 2.4 | 11.2 | 11 | 0.0 | 0 |  |
| `soap_dish__gnndfy` | `z3` | broken | 2.0 | 11.1 | 11 | 0.0 | 0 |  |
| `soap_dish__gnndfy` | `z5` | broken | 2.0 | 11.0 | 11 | 0.0 | 0 |  |
| `soap_dish__gnndfy` | `z6` | broken | 1.9 | 10.8 | 11 | 0.0 | 0 |  |
| `soap_dish__gnndfy` | `zinf` | broken | 2.3 | 11.2 | 11 | 0.0 | 0 |  |
| `folder__lktggf` | `y2` | broken | 1.7 | 14.1 | 35 | 110.1 | 41 | -y |
| `notepad__rvevdj` | `x2` | broken | 1.1 | 4.9 | 27 | 40.6 | 22 | +y |
| `bottle_of_dish_soap__bnmixt` | `z2` | broken | 1.5 | 11.9 | 23 | 50.0 | 26 | +x |
| `bottle_of_lotion__tkryrh` | `z2` | broken | 1.3 | 7.4 | 11 | 4.1 | 18 | +z |
| `bottle_of_lotion__tkryrh` | `z4` | broken | 1.2 | 6.3 | 11 | 6.9 | 25 | +z |
| `bottle_of_lotion__tkryrh` | `z3` | broken | 1.8 | 8.8 | 18 | 7.4 | 24 | +z |
| `bottle_of_lotion__tkryrh` | `z5` | broken | 1.6 | 7.6 | 18 | 5.9 | 23 | +z |
| `bottle_of_lotion__tkryrh` | `z6` | broken | 1.5 | 6.8 | 17 | 9.8 | 23 | +z |
| `bottle_of_lotion__tkryrh` | `zinf` | broken | 1.3 | 5.9 | 13 | 9.6 | 23 | +z |
| `tube_of_lotion__ixfuxm` | `z3` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `tube_of_lotion__ixfuxm` | `z5` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `tube_of_lotion__ixfuxm` | `z6` | broken | 0.2 | 1.0 | 2 | 0.0 | 0 |  |
| `tube_of_lotion__ixfuxm` | `zinf` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `bottle_of_face_cream__dztaed` | `z2` | broken | 1.1 | 7.2 | 13 | 4.3 | 11 | +y |
| `bottle_of_face_cream__dztaed` | `z4` | broken | 0.9 | 6.3 | 13 | 7.7 | 11 | +z |
| `bottle_of_face_cream__dztaed` | `z3` | broken | 1.0 | 6.8 | 13 | 9.7 | 12 | +z |
| `bottle_of_face_cream__dztaed` | `z5` | broken | 0.9 | 6.1 | 12 | 5.2 | 12 | +z |
| `bottle_of_face_cream__dztaed` | `z6` | broken | 0.8 | 5.4 | 12 | 11.7 | 12 | +z |
| `bottle_of_face_cream__dztaed` | `zinf` | broken | 1.0 | 7.0 | 13 | 12.9 | 12 | +z |
| `bottle_of_perfume__ipurzb` | `z3` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `bottle_of_perfume__ipurzb` | `z5` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `bottle_of_perfume__ipurzb` | `z6` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `z2` | broken | 0.9 | 3.0 | 4 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `z4` | broken | 1.0 | 3.5 | 5 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `z3` | broken | 1.0 | 3.5 | 5 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `z5` | broken | 1.0 | 3.5 | 5 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `z6` | borderline | 1.0 | 3.4 | 5 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `zinf` | broken | 1.0 | 3.4 | 5 | 0.0 | 0 |  |
| `bottle_of_ground_cloves__vzamzb` | `z3` | borderline | 1.0 | 3.0 | 5 | 0.0 | 0 |  |
| `bottle_of_ground_cloves__vzamzb` | `zinf` | borderline | 1.0 | 3.1 | 4 | 1.0 | 9 | -z |
| `bottle_of_ground_mace__lgpxro` | `z2` | broken | 0.8 | 2.6 | 4 | 0.4 | 10 | -z |
| `bottle_of_ground_mace__lgpxro` | `z4` | broken | 0.8 | 2.6 | 4 | 0.0 | 0 |  |
| `bottle_of_ground_mace__lgpxro` | `z3` | broken | 0.8 | 2.7 | 4 | 0.0 | 0 |  |
| `bottle_of_ground_mace__lgpxro` | `z5` | broken | 0.8 | 2.6 | 4 | 0.0 | 0 |  |
| `bottle_of_ground_mace__lgpxro` | `z6` | broken | 0.8 | 2.8 | 4 | 1.4 | 10 | -z |
| `bottle_of_ground_mace__lgpxro` | `zinf` | broken | 0.9 | 2.8 | 4 | 1.8 | 10 | -z |
| `bottle_of_ground_nutmeg__qebruq` | `z2` | broken | 0.8 | 2.9 | 5 | 0.0 | 0 |  |
| `bottle_of_ground_nutmeg__qebruq` | `z4` | broken | 1.0 | 3.5 | 5 | 0.0 | 0 |  |
| `bottle_of_ground_nutmeg__qebruq` | `z3` | broken | 1.0 | 3.5 | 5 | 0.0 | 0 |  |
| `bottle_of_ground_nutmeg__qebruq` | `z5` | borderline | 1.0 | 3.4 | 5 | 0.0 | 0 |  |
| `bottle_of_ground_nutmeg__qebruq` | `z6` | borderline | 0.9 | 3.3 | 5 | 0.0 | 0 |  |
| `bottle_of_ground_nutmeg__qebruq` | `zinf` | broken | 1.0 | 3.4 | 5 | 0.0 | 0 |  |
| `bottle_of_paprika__nrdgrp` | `z3` | borderline | 1.0 | 3.1 | 5 | 0.0 | 0 |  |
| `bottle_of_paprika__nrdgrp` | `zinf` | borderline | 1.0 | 3.1 | 5 | 1.1 | 9 | -z |
| `bottle_of_seasoning__vqwqqv` | `z2` | broken | 0.8 | 2.9 | 5 | 0.0 | 0 |  |
| `bottle_of_seasoning__vqwqqv` | `z4` | broken | 1.0 | 3.5 | 5 | 0.0 | 0 |  |
| `bottle_of_seasoning__vqwqqv` | `z3` | broken | 1.0 | 3.5 | 5 | 0.0 | 0 |  |
| `bottle_of_seasoning__vqwqqv` | `z5` | borderline | 1.0 | 3.4 | 5 | 0.0 | 0 |  |
| `bottle_of_seasoning__vqwqqv` | `z6` | borderline | 0.9 | 3.3 | 5 | 0.0 | 0 |  |
| `bottle_of_seasoning__vqwqqv` | `zinf` | broken | 1.0 | 3.4 | 5 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `x2` | broken | 1.5 | 3.4 | 4 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `y2` | broken | 1.5 | 3.4 | 4 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `z3` | broken | 1.7 | 4.0 | 5 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `z5` | borderline | 1.5 | 3.7 | 5 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `z6` | broken | 1.7 | 4.0 | 5 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `zinf` | broken | 1.7 | 4.1 | 5 | 0.0 | 0 |  |
| `jar_of_curry_powder__ozhftu` | `z2` | borderline | 0.8 | 2.7 | 4 | 0.0 | 0 |  |
| `jar_of_curry_powder__ozhftu` | `z4` | borderline | 0.8 | 2.7 | 4 | 3.8 | 10 | -z |
| `jar_of_curry_powder__ozhftu` | `z3` | broken | 0.9 | 2.9 | 5 | 2.9 | 10 | -z |
| `jar_of_curry_powder__ozhftu` | `z5` | borderline | 0.8 | 2.7 | 5 | 2.8 | 10 | -z |
| `jar_of_curry_powder__ozhftu` | `z6` | borderline | 0.8 | 2.8 | 4 | 1.7 | 10 | -z |
| `jar_of_curry_powder__ozhftu` | `zinf` | broken | 0.9 | 2.8 | 4 | 3.9 | 10 | -z |
| `bottle_of_coconut_water__lsixio` | `z2` | borderline | 1.2 | 5.8 | 10 | 3.7 | 15 | -x |
| `bottle_of_coconut_water__lsixio` | `z4` | broken | 2.4 | 15.4 | 31 | 17.9 | 25 | +x |
| `bottle_of_milkshake__naxqya` | `z2` | broken | 0.3 | 0.6 | 11 | 3.8 | 21 | +z |
| `bottle_of_milkshake__naxqya` | `z4` | broken | 0.3 | 0.6 | 11 | 4.0 | 24 | -y |
| `bottle_of_milkshake__naxqya` | `z3` | broken | 0.3 | 0.5 | 11 | 3.3 | 20 | +z |
| `bottle_of_milkshake__naxqya` | `z5` | broken | 0.3 | 0.9 | 10 | 6.1 | 19 | +z |
| `bottle_of_milkshake__naxqya` | `z6` | broken | 0.3 | 0.6 | 11 | 5.6 | 20 | +z |
| `bottle_of_milkshake__naxqya` | `zinf` | broken | 0.3 | 0.8 | 8 | 11.4 | 19 | +z |
| `egg__qopwgx` | `z2` | borderline | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `egg__qopwgx` | `x4` | borderline | 0.1 | 0.8 | 2 | 0.0 | 0 |  |
| `egg__qopwgx` | `x3` | broken | 0.1 | 0.8 | 2 | 0.0 | 0 |  |
| `egg__qopwgx` | `xinf` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `bag_of_flour__fntqmd` | `z2` | broken | 0.8 | 2.7 | 15 | 13.0 | 11 | -z |
| `jar_of_sugar__bholak` | `z2` | broken | 0.3 | 0.2 | 9 | 4.4 | 17 | +x |
| `jug_of_milk__xbvdpc` | `z2` | broken | 0.3 | 1.0 | 2 | 0.0 | 0 |  |
| `jug_of_milk__xbvdpc` | `z4` | broken | 0.4 | 1.0 | 2 | 0.0 | 0 |  |
| `hotdog_frank__qijvmi` | `y2` | broken | 0.9 | 4.8 | 6 | 0.0 | 0 |  |
| `carrot__qhmmmx` | `x2` | borderline | 1.6 | 3.7 | 6 | 0.6 | 13 | -x |
| `carrot__qhmmmx` | `x4` | borderline | 1.4 | 3.5 | 6 | 1.4 | 17 | -x |
| `carrot__qhmmmx` | `x3` | borderline | 1.8 | 3.8 | 6 | 0.9 | 14 | -x |
| `carrot__qhmmmx` | `x5` | borderline | 1.6 | 3.5 | 6 | 1.3 | 14 | -x |
| `carrot__qhmmmx` | `x6` | broken | 1.8 | 3.5 | 5 | 1.1 | 14 | -x |
| `carrot__qhmmmx` | `xinf` | broken | 1.1 | 2.2 | 3 | 0.0 | 0 |  |
| `box_of_rum__lnjbjj` | `x2` | broken | 1.0 | 5.1 | 18 | 57.4 | 17 | +x |
| `box_of_rum__lnjbjj` | `y2` | broken | 1.0 | 5.1 | 18 | 57.0 | 17 | -x |
| `box_of_rum__lnjbjj` | `z4` | broken | 1.6 | 10.7 | 19 | 44.2 | 16 | +x |
| `bottle_of_disinfectant__faedff` | `z2` | broken | 1.4 | 4.1 | 6 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `x2` | broken | 0.5 | 3.1 | 12 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `x4` | broken | 0.4 | 2.7 | 11 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `x3` | broken | 0.4 | 2.7 | 10 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `x5` | broken | 0.4 | 2.5 | 10 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `x6` | broken | 0.4 | 2.3 | 9 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `xinf` | broken | 0.5 | 3.1 | 12 | 0.0 | 0 |  |
| `scone__gwdhyj` | `z2` | broken | 0.2 | 1.2 | 4 | 0.0 | 0 |  |
| `bottle_of_conditioner__teafxb` | `x2` | broken | 1.7 | 14.0 | 29 | 19.9 | 16 | +x |
| `paper_bag__bzsxgw` | `x2` | broken | 2.4 | 6.4 | 9 | 0.0 | 0 |  |
| `paper_bag__bzsxgw` | `y2` | broken | 2.4 | 6.4 | 9 | 0.0 | 0 |  |
| `paper_bag__bzsxgw` | `z2` | broken | 1.1 | 3.5 | 7 | 0.0 | 0 |  |
| `toy_box__kvithq` | `z4` | broken | 1.1 | 5.1 | 17 | 60.5 | 15 | +z |
| `vase__atgnsc` | `x2` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `vase__atgnsc` | `y2` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `vase__atgnsc` | `z2` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `vase__atgnsc` | `z4` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `vase__atgnsc` | `z3` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `vase__atgnsc` | `z5` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `vase__atgnsc` | `z6` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `vase__atgnsc` | `zinf` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `tissue_dispenser__frdbld` | `z2` | broken | 0.1 | 0.6 | 4 | 0.0 | 0 |  |
| `tissue_dispenser__nflhrf` | `x2` | broken | 0.6 | 1.8 | 20 | 60.0 | 17 | +x |
| `can_of_bay_leaves__ppwvjf` | `z2` | broken | 2.0 | 12.7 | 21 | 6.2 | 19 | -z |
| `can_of_bay_leaves__ppwvjf` | `z4` | broken | 2.0 | 10.9 | 19 | 3.9 | 17 | -z |
| `can_of_bay_leaves__ppwvjf` | `z3` | broken | 1.9 | 10.5 | 20 | 6.2 | 16 | -z |
| `can_of_bay_leaves__ppwvjf` | `z5` | broken | 2.0 | 11.0 | 19 | 4.7 | 18 | -z |
| `can_of_bay_leaves__ppwvjf` | `z6` | broken | 2.1 | 10.7 | 19 | 5.0 | 17 | -z |
| `cap__iqeyba` | `zinf` | borderline | 1.8 | 4.6 | 13 | 0.7 | 13 | -z |
| `dip_candle__poexjo` | `x2` | borderline | 0.2 | 0.9 | 3 | 1.3 | 16 | +z |
| `dip_candle__poexjo` | `y2` | broken | 0.3 | 1.2 | 4 | 1.3 | 16 | +z |
| `dip_candle__poexjo` | `z2` | broken | 0.2 | 1.1 | 3 | 1.2 | 14 | +z |
| `dip_candle__poexjo` | `z4` | borderline | 0.2 | 1.2 | 3 | 1.5 | 16 | +z |
| `dip_candle__poexjo` | `z3` | broken | 0.2 | 1.2 | 3 | 1.3 | 14 | +z |
| `dip_candle__poexjo` | `z5` | borderline | 0.2 | 1.2 | 3 | 1.5 | 16 | +z |
| `dip_candle__poexjo` | `z6` | borderline | 0.2 | 1.1 | 3 | 1.5 | 15 | +z |
| `dip_candle__poexjo` | `zinf` | broken | 0.2 | 1.2 | 3 | 1.2 | 14 | +z |
| `dip_candle__ghctun` | `x2` | borderline | 0.2 | 0.6 | 3 | 1.0 | 17 | -z |
| `dip_candle__ghctun` | `y2` | borderline | 0.2 | 0.6 | 3 | 0.9 | 16 | -z |
| `dip_candle__mxjejs` | `x2` | borderline | 0.2 | 1.0 | 4 | 1.3 | 17 | -z |
| `dip_candle__mxjejs` | `y2` | borderline | 0.2 | 0.9 | 4 | 1.3 | 18 | -z |
| `light_bulb__ymomhw` | `x4` | broken | 0.5 | 4.1 | 10 | 0.1 | 6 | -y |
| `light_bulb__ymomhw` | `x5` | broken | 0.5 | 3.6 | 10 | 0.2 | 6 | -y |
| `light_bulb__ymomhw` | `x6` | broken | 0.5 | 3.4 | 10 | 0.2 | 6 | -y |
| `blackberry__tvpfvb` | `y2` | borderline | 2.0 | 6.0 | 46 | 3.2 | 23 | +x |
| `blackberry__tvpfvb` | `z2` | borderline | 2.0 | 5.8 | 46 | 3.0 | 24 | +x |
| `blackberry__tvpfvb` | `x4` | borderline | 0.7 | 3.1 | 8 | 0.9 | 10 | +x |
| `blackberry__tvpfvb` | `y4` | borderline | 2.1 | 6.5 | 46 | 3.4 | 23 | -z |
| `blackberry__tvpfvb` | `z4` | borderline | 2.0 | 5.8 | 45 | 3.1 | 22 | +y |
| `blackberry__tvpfvb` | `x3` | borderline | 0.8 | 3.4 | 8 | 0.7 | 11 | +x |
| `blackberry__tvpfvb` | `y3` | borderline | 1.9 | 4.9 | 43 | 3.0 | 24 | +x |
| `blackberry__tvpfvb` | `z3` | borderline | 2.1 | 6.1 | 46 | 3.2 | 23 | +y |
| `blackberry__tvpfvb` | `x5` | borderline | 0.7 | 2.8 | 7 | 0.8 | 10 | +x |
| `blackberry__tvpfvb` | `y5` | borderline | 2.1 | 6.6 | 47 | 3.3 | 23 | -z |
| `blackberry__tvpfvb` | `z5` | broken | 2.2 | 7.8 | 46 | 4.8 | 22 | +x |
| `blackberry__tvpfvb` | `x6` | borderline | 0.7 | 2.9 | 7 | 0.4 | 9 | +x |
| `blackberry__tvpfvb` | `y6` | borderline | 2.1 | 6.0 | 46 | 3.0 | 24 | +x |
| `blackberry__tvpfvb` | `z6` | broken | 2.2 | 7.9 | 46 | 5.0 | 24 | +x |
| `blackberry__tvpfvb` | `xinf` | borderline | 0.7 | 3.1 | 9 | 0.9 | 10 | +x |
| `bookcase_baseboard__hlhneo` | `y3` | broken | 2.2 | 5.6 | 8 | 0.0 | 0 |  |
| `bookcase_baseboard__hlhneo` | `y6` | broken | 1.9 | 5.0 | 8 | 0.0 | 0 |  |
| `table_lamp__ehjsdz` | `z2` | broken | 0.2 | 0.9 | 3 | 0.0 | 0 |  |
| `table_lamp__ehjsdz` | `z4` | broken | 0.8 | 1.8 | 27 | 206.1 | 24 | -z |
| `table_lamp__ehjsdz` | `z3` | broken | 0.9 | 2.1 | 27 | 271.9 | 24 | -z |
| `table_lamp__ehjsdz` | `z5` | broken | 0.7 | 1.6 | 27 | 151.9 | 24 | -z |
| `table_lamp__ehjsdz` | `z6` | broken | 0.6 | 1.3 | 27 | 132.4 | 24 | -z |
| `table_lamp__ehjsdz` | `zinf` | broken | 0.9 | 2.0 | 27 | 376.7 | 24 | -z |
| `teapot__mvrhya` | `z2` | broken | 0.2 | 1.4 | 6 | 0.0 | 0 |  |
| `teapot__mvrhya` | `z4` | broken | 0.2 | 1.0 | 5 | 0.0 | 0 |  |
| `teapot__mvrhya` | `z3` | broken | 0.2 | 1.0 | 5 | 0.0 | 0 |  |
| `teapot__mvrhya` | `z5` | broken | 0.2 | 1.0 | 5 | 0.0 | 0 |  |
| `teapot__mvrhya` | `z6` | broken | 0.2 | 1.0 | 5 | 0.0 | 0 |  |
| `teapot__mvrhya` | `zinf` | broken | 0.2 | 1.0 | 5 | 0.0 | 0 |  |
| `tomato_paste_can__krarex` | `z4` | broken | 1.2 | 6.7 | 9 | 0.0 | 0 |  |
| `tomato_paste_can__krarex` | `z5` | broken | 0.6 | 5.4 | 8 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `z2` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `z4` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `z3` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `z5` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `z6` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `zinf` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `paperback_book__bisnup` | `x2` | broken | 1.3 | 7.6 | 23 | 28.2 | 19 | -y |
| `paperback_book__ixwtnb` | `x2` | broken | 2.3 | 13.3 | 40 | 47.8 | 22 | -z |
| `paperback_book__wfcxjv` | `x2` | broken | 1.8 | 6.7 | 19 | 12.8 | 15 | -x |
| `frying_pan__jpzusm` | `z5` | broken | 0.2 | 0.1 | 10 | 0.4 | 16 | +x |
| `frying_pan__jpzusm` | `z6` | broken | 0.2 | 0.1 | 7 | 0.0 | 0 |  |
| `frying_pan__mhndon` | `z5` | broken | 0.2 | 0.0 | 10 | 1.6 | 0 | +z |
| `frying_pan__mhndon` | `z6` | broken | 0.2 | 0.0 | 7 | 1.6 | 0 | +z |
| `keyboard__uwacaq` | `x2` | broken | 1.2 | 10.0 | 10 | 0.0 | 0 |  |
| `keyboard__uwacaq` | `y2` | broken | 1.2 | 10.0 | 10 | 0.0 | 0 |  |
| `carton_of_eggs__stxfxb` | `x2` | broken | 1.6 | 3.6 | 37 | 85.3 | 49 | -x |
| `notepad__awcyha` | `x2` | broken | 1.3 | 8.1 | 23 | 85.3 | 30 | +y |
| `notepad__hirdfd` | `x2` | broken | 0.6 | 2.6 | 17 | 42.4 | 18 | +y |
| `clipboard__gmxyfo` | `y2` | broken | 0.3 | 0.6 | 13 | 0.0 | 0 |  |
| `clipboard__gmxyfo` | `z3` | broken | 0.2 | 0.3 | 8 | 0.0 | 0 |  |
| `clipboard__gmxyfo` | `z6` | broken | 0.2 | 0.4 | 8 | 0.0 | 0 |  |
| `wall_clock__xevvel` | `x2` | broken | 2.3 | 5.9 | 8 | 0.0 | 0 |  |
| `wall_clock__xevvel` | `x4` | broken | 2.3 | 6.0 | 8 | 0.0 | 0 |  |
| `wall_clock__xevvel` | `x6` | broken | 2.2 | 5.8 | 8 | 0.0 | 0 |  |
| `signpost__vwhtlm` | `z2` | broken | 1.6 | 11.0 | 31 | 83.0 | 28 | +z |
| `decorative_sign__rypeha` | `x2` | broken | 0.4 | 2.2 | 6 | 0.0 | 0 |  |
| `decorative_sign__rypeha` | `y2` | broken | 0.2 | 1.3 | 4 | 0.0 | 0 |  |
| `decorative_sign__rypeha` | `z2` | broken | 0.4 | 2.5 | 6 | 0.0 | 0 |  |
| `towel_rack__nhlijv` | `z2` | broken | 0.4 | 0.9 | 4 | 33.0 | 38 | -z |
| `umbrella__ziqzqo` | `x2` | broken | 1.8 | 4.0 | 7 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `y2` | broken | 1.8 | 4.2 | 9 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `z2` | broken | 2.0 | 4.5 | 9 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `x4` | broken | 1.5 | 3.7 | 7 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `x3` | broken | 1.8 | 4.3 | 7 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `x5` | broken | 1.7 | 3.9 | 7 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `x6` | broken | 1.5 | 3.9 | 6 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `xinf` | broken | 1.4 | 3.6 | 6 | 0.0 | 0 |  |
| `rolling_pin__smnklu` | `yinf` | borderline | 1.4 | 4.7 | 7 | 0.8 | 13 | +x |
| `baseball_bat__spcxrt` | `x2` | broken | 1.6 | 4.0 | 7 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `y2` | broken | 1.7 | 4.4 | 8 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `z2` | broken | 1.7 | 4.4 | 8 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `x4` | broken | 1.6 | 4.2 | 6 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `x3` | broken | 1.6 | 4.0 | 6 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `x5` | broken | 1.6 | 4.0 | 6 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `x6` | broken | 1.6 | 4.0 | 6 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `xinf` | broken | 1.6 | 4.2 | 7 | 0.0 | 0 |  |
| `shelf__kxvbui` | `y2` | broken | 1.1 | 2.7 | 4 | 0.0 | 0 |  |
| `shelf__kxvbui` | `z2` | broken | 1.1 | 2.8 | 4 | 0.0 | 0 |  |
| `radiator__arsjbe` | `y2` | broken | 0.7 | 2.0 | 9 | 37.8 | 43 | +y |
| `radiator__dwmloz` | `x2` | broken | 1.1 | 2.7 | 4 | 0.0 | 0 |  |
| `radiator__dwmloz` | `y2` | broken | 1.1 | 2.7 | 4 | 0.0 | 0 |  |
| `radiator__dwmloz` | `z2` | broken | 1.2 | 2.8 | 4 | 0.0 | 0 |  |
| `radiator__xnoiwp` | `y2` | broken | 0.8 | 2.4 | 9 | 40.2 | 45 | +y |
| `radiator__wntwxx` | `x2` | broken | 0.5 | 1.8 | 7 | 7.7 | 54 | +z |
| `radiator__wntwxx` | `y2` | broken | 0.3 | 0.5 | 2 | 7.2 | 40 | -y |
| `radiator__wntwxx` | `z2` | broken | 0.6 | 1.9 | 8 | 7.9 | 45 | +y |
| `drumstick__jauzlc` | `z2` | broken | 1.4 | 3.7 | 7 | 0.0 | 0 |  |
| `drumstick__jauzlc` | `x3` | borderline | 1.9 | 3.9 | 7 | 0.0 | 0 |  |
| `wall_mounted_tv__ggfznh` | `x2` | broken | 2.4 | 7.8 | 40 | 152.7 | 28 | +x |
| `wall_mounted_tv__dxyfxf` | `x2` | broken | 0.4 | 1.8 | 7 | 0.6 | 19 | +x |
| `wall_mounted_tv__ylvjhb` | `x2` | broken | 0.4 | 2.0 | 5 | 0.0 | 0 |  |
| `wall_mounted_tv__sodbbr` | `x2` | broken | 0.4 | 1.7 | 7 | 0.6 | 19 | +y |
| `umbrella__zxnjwz` | `x2` | broken | 1.0 | 2.8 | 22 | 12.2 | 38 | -x |
| `umbrella__zxnjwz` | `x4` | broken | 0.8 | 2.4 | 19 | 10.5 | 32 | +x |
| `umbrella__zxnjwz` | `x3` | broken | 0.9 | 2.6 | 21 | 10.7 | 36 | +x |
| `umbrella__zxnjwz` | `x5` | broken | 0.8 | 2.4 | 19 | 9.4 | 29 | +x |
| `umbrella__zxnjwz` | `x6` | broken | 0.6 | 2.1 | 15 | 8.6 | 26 | +x |
| `umbrella__zxnjwz` | `xinf` | broken | 0.9 | 2.5 | 22 | 13.4 | 38 | -x |
| `whiteboard__gvkgsi` | `x2` | broken | 1.2 | 5.4 | 39 | 134.4 | 17 | +x |
| `towel_rack__crivoa` | `x2` | broken | 0.7 | 4.1 | 14 | 3.8 | 15 | +y |
| `towel_rack__crivoa` | `y2` | broken | 1.1 | 7.4 | 16 | 3.6 | 15 | +y |
| `towel_rack__crivoa` | `z2` | broken | 1.0 | 7.3 | 16 | 5.2 | 17 | -y |
| `towel_rack__crivoa` | `y4` | broken | 1.0 | 6.7 | 15 | 6.4 | 15 | +y |
| `towel_rack__crivoa` | `y3` | broken | 1.0 | 7.1 | 15 | 2.3 | 16 | -y |
| `towel_rack__crivoa` | `y5` | broken | 1.0 | 6.8 | 15 | 7.8 | 16 | +y |
| `towel_rack__crivoa` | `y6` | broken | 1.1 | 7.1 | 16 | 7.8 | 16 | +y |
| `towel_rack__crivoa` | `yinf` | broken | 1.0 | 7.0 | 16 | 3.8 | 17 | +y |
| `decorative_sign__hgonwj` | `x2` | borderline | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `decorative_sign__hgonwj` | `y2` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `decorative_sign__hgonwj` | `z2` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `pack_of_kielbasa__dgodnn` | `x2` | broken | 1.0 | 3.3 | 6 | 6.2 | 18 | -x |
| `pack_of_kielbasa__fwqdoy` | `x2` | borderline | 0.2 | 0.5 | 1 | 3.9 | 17 | +x |
| `pack_of_kielbasa__bzlsuh` | `x2` | broken | 0.3 | 0.5 | 2 | 10.2 | 23 | -x |
| `pack_of_kielbasa__nhkuzn` | `x2` | broken | 0.1 | 0.3 | 1 | 6.4 | 24 | +x |
| `pen__hiyeki` | `x6` | borderline | 0.8 | 3.2 | 15 | 0.3 | 12 | -x |
| `pen__enfumt` | `x2` | borderline | 0.4 | 2.0 | 3 | 0.0 | 0 |  |
| `pen__enfumt` | `y2` | broken | 0.6 | 2.3 | 3 | 0.0 | 0 |  |
| `pen__enfumt` | `z2` | broken | 0.2 | 1.2 | 2 | 0.0 | 0 |  |
| `pen__enfumt` | `x4` | broken | 0.3 | 1.6 | 3 | 0.0 | 0 |  |
| `pen__enfumt` | `x3` | broken | 0.3 | 1.7 | 3 | 0.0 | 0 |  |
| `pen__enfumt` | `x6` | borderline | 0.3 | 1.4 | 3 | 0.0 | 0 |  |
| `pen__enfumt` | `xinf` | broken | 0.3 | 1.6 | 3 | 0.0 | 0 |  |
| `cutting_board__tcdrzs` | `y2` | borderline | 2.6 | 7.0 | 10 | 0.0 | 0 |  |
| `cutting_board__aibvew` | `y2` | broken | 2.2 | 7.3 | 12 | 0.7 | 9 | -y |
| `plate__kkjiko` | `z5` | broken | 1.3 | 13.8 | 14 | 89.7 | 30 | +y |
| `plate__luhkiz` | `z2` | borderline | 1.1 | 5.6 | 11 | 2.5 | 12 | +y |
| `plate__luhkiz` | `z4` | borderline | 1.2 | 6.0 | 10 | 2.9 | 11 | +x |
| `plate__luhkiz` | `z3` | borderline | 1.3 | 6.8 | 13 | 3.7 | 15 | -x |
| `plate__luhkiz` | `z5` | broken | 2.0 | 10.7 | 19 | 4.1 | 15 | -z |
| `plate__luhkiz` | `z6` | borderline | 1.2 | 6.1 | 11 | 2.6 | 13 | -x |
| `plate__bgxzec` | `x2` | broken | 0.1 | 0.4 | 4 | 0.0 | 0 |  |
| `plate__bgxzec` | `y2` | broken | 0.1 | 0.4 | 4 | 0.0 | 0 |  |
| `plate__odmjdd` | `z3` | broken | 0.1 | 0.3 | 3 | 0.0 | 0 |  |
| `plate__odmjdd` | `z5` | broken | 0.1 | 0.4 | 4 | 0.0 | 0 |  |
| `plate__odmjdd` | `z6` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `plate__vtjwof` | `x2` | broken | 0.1 | 0.5 | 4 | 0.0 | 0 |  |
| `plate__vtjwof` | `y2` | broken | 0.1 | 0.5 | 4 | 0.0 | 0 |  |
| `plate__mtetqm` | `z4` | borderline | 0.2 | 0.9 | 5 | 0.0 | 0 |  |
| `plate__mtetqm` | `z3` | broken | 0.2 | 1.3 | 6 | 0.0 | 0 |  |
| `plate__mtetqm` | `z5` | broken | 0.2 | 1.2 | 6 | 0.0 | 0 |  |
| `plate__mtetqm` | `z6` | broken | 0.2 | 1.3 | 6 | 0.0 | 0 |  |
| `plate__mtetqm` | `zinf` | broken | 0.2 | 1.2 | 6 | 0.0 | 0 |  |
| `platter__hnlivs` | `z2` | broken | 0.2 | 0.9 | 4 | 0.0 | 0 |  |
| `platter__kiiamx` | `z2` | broken | 2.1 | 17.4 | 26 | 173.1 | 26 | +z |
| `platter__ekjpdj` | `z2` | broken | 2.5 | 18.3 | 24 | 235.9 | 24 | +z |
| `platter__iadlti` | `z2` | broken | 0.2 | 0.7 | 4 | 0.0 | 0 |  |
| `platter__iadlti` | `z4` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `platter__iadlti` | `z5` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `platter__iadlti` | `z6` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `platter__iadlti` | `zinf` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `toy_figure__gqspnb` | `x2` | broken | 1.6 | 5.7 | 19 | 7.0 | 29 | +y |
| `toy_figure__gqspnb` | `x4` | broken | 2.4 | 9.1 | 27 | 7.0 | 29 | -x |
| `toy_figure__gqspnb` | `x6` | broken | 2.4 | 8.5 | 22 | 7.2 | 31 | -x |
| `tennis_ball__rgekxe` | `y2` | broken | 2.2 | 8.4 | 14 | 14.4 | 13 | -x |
| `tennis_ball__rgekxe` | `x4` | broken | 2.4 | 8.4 | 14 | 14.1 | 12 | -x |
| `tennis_ball__rgekxe` | `z5` | broken | 2.3 | 10.7 | 15 | 13.7 | 13 | -y |
| `tennis_ball__rgekxe` | `z6` | broken | 2.3 | 10.9 | 15 | 13.2 | 12 | -y |
| `tray__hjxczh` | `x2` | broken | 0.8 | 2.4 | 4 | 0.0 | 0 |  |
| `tray__hjxczh` | `y2` | broken | 1.6 | 3.9 | 6 | 0.0 | 0 |  |
| `tray__zcmnji` | `x2` | broken | 0.7 | 2.9 | 9 | 0.0 | 0 |  |
| `tray__zcmnji` | `y2` | broken | 0.6 | 2.6 | 6 | 0.0 | 0 |  |
| `tray__spopfj` | `x2` | broken | 0.1 | 0.0 | 3 | 0.0 | 0 |  |
| `tray__spopfj` | `y2` | broken | 0.1 | 0.0 | 3 | 0.0 | 0 |  |
| `tray__jpcflq` | `x2` | broken | 0.1 | 0.3 | 4 | 0.0 | 0 |  |
| `tray__jpcflq` | `y2` | broken | 0.1 | 0.3 | 4 | 0.0 | 0 |  |
| `tray__thkphg` | `x2` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `tray__thkphg` | `y2` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `tray__zsddtq` | `x2` | broken | 0.1 | 0.3 | 3 | 0.0 | 0 |  |
| `tray__zsddtq` | `y2` | broken | 0.1 | 0.3 | 3 | 0.0 | 0 |  |
| `tray__zsddtq` | `z3` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `tray__zsddtq` | `z5` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `tray__zsddtq` | `z6` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `tray__mkdcha` | `x2` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `tray__mkdcha` | `y2` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `tray__hvlfig` | `x2` | broken | 0.1 | 0.0 | 1 | 0.0 | 0 |  |
| `tray__hvlfig` | `y2` | broken | 0.1 | 0.0 | 1 | 0.0 | 0 |  |
| `lemon__pwqedq` | `x2` | broken | 1.6 | 3.4 | 9 | 11.1 | 31 | +x |
| `bowl__belcml` | `z3` | broken | 0.2 | 0.3 | 5 | 0.0 | 0 |  |
| `bowl__belcml` | `z5` | broken | 0.2 | 0.3 | 5 | 0.0 | 0 |  |
| `bowl__belcml` | `z6` | broken | 0.1 | 0.3 | 4 | 0.0 | 0 |  |
| `bowl__belcml` | `zinf` | broken | 0.2 | 0.3 | 5 | 0.0 | 0 |  |
| `bowl__eawgwj` | `z3` | broken | 0.2 | 1.3 | 6 | 0.0 | 0 |  |
| `bowl__eawgwj` | `z5` | broken | 0.2 | 1.3 | 6 | 0.0 | 0 |  |
| `bowl__eawgwj` | `z6` | broken | 0.2 | 1.3 | 5 | 0.0 | 0 |  |
| `bowl__eawgwj` | `zinf` | broken | 0.2 | 1.3 | 6 | 0.0 | 0 |  |
| `hingeless_jar__lvuvbf` | `x2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `hingeless_jar__lvuvbf` | `y2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `hingeless_jar__bqpmsv` | `x2` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `hingeless_jar__bqpmsv` | `y2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `hingeless_jar__wcqjew` | `x2` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `hingeless_jar__wcqjew` | `y2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `hingeless_jar__mefezc` | `x2` | broken | 0.1 | 0.1 | 2 | 0.0 | 0 |  |
| `hingeless_jar__mefezc` | `y2` | broken | 0.1 | 0.1 | 2 | 0.0 | 0 |  |
| `saucepot__tfzijn` | `z4` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `saucepot__tfzijn` | `z3` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `saucepot__tfzijn` | `z5` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `saucepot__tfzijn` | `z6` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `saucepot__tfzijn` | `zinf` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `z2` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `z4` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `z3` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `z5` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `z6` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `zinf` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `saucepot__vqtkwq` | `z2` | broken | 0.3 | 0.4 | 11 | 45.0 | 18 | +z |
| `saucepot__vqtkwq` | `z4` | broken | 0.2 | 0.3 | 6 | 0.7 | 11 | -z |
| `saucepot__vqtkwq` | `z3` | broken | 0.2 | 0.3 | 8 | 10.1 | 17 | +z |
| `saucepot__vqtkwq` | `z5` | broken | 0.2 | 0.3 | 6 | 0.5 | 12 | +z |
| `saucepot__vqtkwq` | `z6` | broken | 0.2 | 0.3 | 5 | 0.5 | 12 | +z |
| `saucepot__vqtkwq` | `zinf` | broken | 0.2 | 0.3 | 5 | 0.0 | 0 |  |
| `saucepot__uvzmss` | `z4` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `saucepot__uvzmss` | `z3` | broken | 0.1 | 0.1 | 2 | 0.0 | 0 |  |
| `saucepot__uvzmss` | `z5` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `saucepot__uvzmss` | `z6` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `saucepot__uvzmss` | `zinf` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `saucepot__chjetk` | `z4` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `saucepot__chjetk` | `z3` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `saucepot__chjetk` | `z5` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `saucepot__chjetk` | `z6` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `saucepot__chjetk` | `zinf` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `vase__gopbrh` | `x2` | broken | 0.3 | 1.7 | 4 | 0.0 | 0 |  |
| `vase__gopbrh` | `y2` | broken | 0.3 | 1.7 | 4 | 0.0 | 0 |  |
| `vase__gopbrh` | `z3` | broken | 0.2 | 1.1 | 3 | 0.0 | 0 |  |
| `vase__gopbrh` | `z5` | broken | 0.2 | 1.1 | 3 | 0.0 | 0 |  |
| `vase__gopbrh` | `z6` | broken | 0.2 | 1.2 | 3 | 0.0 | 0 |  |
| `vase__gopbrh` | `zinf` | broken | 0.2 | 1.2 | 3 | 0.0 | 0 |  |
| `vase__stqkvx` | `x2` | broken | 0.2 | 0.6 | 4 | 0.0 | 0 |  |
| `vase__stqkvx` | `y2` | broken | 0.2 | 0.6 | 4 | 0.0 | 0 |  |
| `pencil__hmawzy` | `z2` | borderline | 1.5 | 13.3 | 20 | 0.3 | 32 | +x |
| `pencil__hmawzy__scaled` | `z2` | broken | 1.7 | 14.8 | 23 | 7.1 | 29 | +x |
| `pencil__hmawzy__scaled` | `x6` | broken | 0.9 | 6.3 | 15 | 2.7 | 20 | -x |
| `pencil__rzscao__scaled` | `x2` | broken | 0.7 | 4.1 | 14 | 2.8 | 17 | -x |
| `pencil__rzscao__scaled` | `x4` | broken | 0.9 | 6.2 | 16 | 2.7 | 23 | -x |
| `pencil__rzscao__scaled` | `x3` | broken | 0.7 | 4.4 | 14 | 2.5 | 17 | -x |
| `pencil__rzscao__scaled` | `x5` | borderline | 0.9 | 6.5 | 15 | 2.9 | 23 | -x |
| `pencil__rzscao__scaled` | `x6` | borderline | 0.9 | 6.1 | 15 | 2.7 | 22 | -x |
| `pencil__rzscao__scaled` | `xinf` | broken | 0.8 | 5.4 | 15 | 3.4 | 23 | -x |
| `pencil__gbzzcg__scaled` | `x2` | borderline | 1.1 | 4.9 | 26 | 3.4 | 23 | -x |
| `pencil__gbzzcg__scaled` | `x4` | broken | 1.3 | 7.9 | 26 | 3.8 | 28 | -x |
| `pencil__gbzzcg__scaled` | `x3` | broken | 1.2 | 7.7 | 26 | 4.5 | 28 | -x |
| `pencil__gbzzcg__scaled` | `x5` | broken | 1.1 | 6.6 | 24 | 3.2 | 26 | -x |
| `pencil__gbzzcg__scaled` | `x6` | borderline | 1.0 | 5.6 | 22 | 3.3 | 16 | -z |
| `pencil__gbzzcg__scaled` | `xinf` | broken | 1.2 | 5.9 | 28 | 6.6 | 30 | -x |
| `pencil__fnlbpu__scaled` | `x2` | borderline | 0.8 | 4.1 | 18 | 2.8 | 14 | +y |
| `pencil__fnlbpu__scaled` | `x4` | borderline | 1.0 | 6.4 | 18 | 3.0 | 18 | -x |
| `pencil__fnlbpu__scaled` | `x3` | borderline | 0.8 | 4.4 | 16 | 2.5 | 21 | -x |
| `pencil__fnlbpu__scaled` | `x5` | borderline | 1.0 | 6.0 | 18 | 2.7 | 17 | -x |
| `pencil__fnlbpu__scaled` | `x6` | borderline | 0.9 | 5.3 | 17 | 3.4 | 13 | -z |
| `pencil__fnlbpu__scaled` | `xinf` | borderline | 0.8 | 5.0 | 17 | 3.3 | 14 | -z |
| `pencil__heqkmr__scaled` | `x6` | broken | 1.3 | 6.1 | 27 | 4.6 | 25 | -x |
| `pencil__jntybl__scaled` | `x2` | borderline | 1.0 | 4.3 | 21 | 3.3 | 22 | -x |
| `pencil__jntybl__scaled` | `x4` | broken | 1.3 | 8.3 | 22 | 5.6 | 24 | -x |
| `pencil__jntybl__scaled` | `x3` | broken | 1.2 | 7.8 | 23 | 4.6 | 26 | -x |
| `pencil__jntybl__scaled` | `x5` | borderline | 1.1 | 7.1 | 20 | 3.2 | 23 | -x |
| `pencil__jntybl__scaled` | `x6` | broken | 1.0 | 6.2 | 19 | 4.7 | 19 | -x |
| `pencil__jntybl__scaled` | `xinf` | broken | 1.3 | 8.3 | 23 | 5.8 | 25 | -x |
| `pencil__lruiot__scaled` | `x2` | broken | 1.3 | 5.8 | 29 | 5.7 | 31 | -x |
| `pencil__lruiot__scaled` | `x5` | broken | 1.3 | 8.6 | 26 | 3.0 | 23 | -x |
| `pencil__lruiot__scaled` | `x6` | broken | 1.2 | 7.8 | 24 | 3.4 | 21 | -x |
| `pencil__lykqoh__scaled` | `x2` | broken | 1.2 | 5.3 | 26 | 5.7 | 30 | -x |
| `pencil__lykqoh__scaled` | `x5` | broken | 1.1 | 6.9 | 23 | 3.0 | 24 | -x |
| `pencil__lykqoh__scaled` | `x6` | broken | 1.1 | 6.6 | 22 | 2.9 | 23 | -x |
| `pencil__vpzhth__scaled` | `x6` | broken | 1.2 | 6.3 | 24 | 2.8 | 21 | -x |
| `pencil__vvofwr__scaled` | `x2` | broken | 1.1 | 5.8 | 23 | 4.7 | 29 | -x |
| `pencil__vvofwr__scaled` | `x5` | broken | 1.2 | 7.7 | 21 | 5.2 | 27 | -x |
| `pencil__vvofwr__scaled` | `x6` | broken | 1.1 | 7.1 | 21 | 3.3 | 24 | -x |
| `pencil__vedvit__scaled` | `x2` | broken | 0.7 | 3.5 | 12 | 0.9 | 10 | -y |
| `pencil__vedvit__scaled` | `y2` | broken | 1.5 | 12.4 | 27 | 2.0 | 19 | +y |
| `pencil__vedvit__scaled` | `z2` | broken | 1.6 | 14.0 | 28 | 1.9 | 19 | +y |
| `pencil__vedvit__scaled` | `x5` | broken | 0.8 | 4.5 | 17 | 2.8 | 17 | -x |
| `pencil__vedvit__scaled` | `x6` | broken | 0.7 | 4.0 | 15 | 2.3 | 16 | -x |
| `marker__xakoic` | `x2` | borderline | 0.7 | 3.4 | 7 | 1.2 | 16 | +x |
| `marker__xakoic` | `x4` | borderline | 0.6 | 3.0 | 6 | 1.0 | 12 | -x |
| `marker__xakoic` | `x3` | borderline | 0.6 | 3.3 | 7 | 1.0 | 14 | +x |
| `marker__xakoic` | `x5` | borderline | 0.5 | 2.7 | 6 | 1.0 | 11 | -x |
| `marker__xakoic` | `x6` | borderline | 0.5 | 2.6 | 5 | 0.9 | 10 | -x |
| `marker__xakoic` | `xinf` | borderline | 0.6 | 3.1 | 7 | 1.4 | 15 | +x |
| `marker__gldosk` | `x2` | borderline | 0.7 | 3.3 | 11 | 1.2 | 16 | +x |
| `marker__gldosk` | `x4` | borderline | 0.6 | 2.8 | 8 | 1.0 | 12 | -x |
| `marker__gldosk` | `x3` | borderline | 0.7 | 3.2 | 10 | 0.9 | 14 | +x |
| `marker__gldosk` | `x5` | borderline | 0.5 | 2.4 | 7 | 1.0 | 11 | -x |
| `marker__gldosk` | `x6` | borderline | 0.5 | 2.3 | 6 | 0.9 | 10 | -x |
| `marker__gldosk` | `xinf` | borderline | 0.7 | 2.9 | 9 | 1.5 | 17 | +x |
| `marker__vrjwkg` | `x2` | broken | 0.7 | 3.5 | 7 | 1.1 | 11 | -x |
| `marker__vrjwkg` | `x4` | broken | 0.5 | 2.8 | 6 | 0.4 | 10 | -x |
| `marker__vrjwkg` | `x3` | borderline | 0.6 | 3.1 | 7 | 1.0 | 10 | -x |
| `marker__vrjwkg` | `x5` | broken | 0.5 | 2.7 | 6 | 0.0 | 0 |  |
| `marker__vrjwkg` | `x6` | broken | 0.5 | 2.5 | 6 | 0.0 | 0 |  |
| `marker__vrjwkg` | `xinf` | broken | 0.7 | 3.4 | 7 | 0.9 | 11 | -x |
| `marker__vjjsul` | `x2` | broken | 0.9 | 4.4 | 9 | 1.6 | 16 | -x |
| `marker__vjjsul` | `x4` | borderline | 0.7 | 3.5 | 8 | 1.3 | 14 | -x |
| `marker__vjjsul` | `x3` | borderline | 0.8 | 4.0 | 8 | 1.5 | 17 | -x |
| `marker__vjjsul` | `x5` | borderline | 0.6 | 3.3 | 7 | 1.2 | 12 | -x |
| `marker__vjjsul` | `x6` | broken | 0.6 | 3.1 | 7 | 1.0 | 11 | -x |
| `marker__vjjsul` | `xinf` | broken | 0.6 | 3.0 | 7 | 1.0 | 12 | +x |
| `marker__oyjbhy` | `x2` | borderline | 0.8 | 4.3 | 12 | 1.5 | 19 | -x |
| `marker__oyjbhy` | `x4` | borderline | 0.7 | 3.5 | 8 | 1.3 | 15 | -x |
| `marker__oyjbhy` | `x3` | borderline | 0.8 | 3.9 | 9 | 1.6 | 18 | -x |
| `marker__oyjbhy` | `x5` | borderline | 0.7 | 3.3 | 7 | 1.2 | 13 | -x |
| `marker__oyjbhy` | `x6` | borderline | 0.6 | 3.0 | 7 | 1.0 | 11 | -x |
| `marker__oyjbhy` | `xinf` | borderline | 0.8 | 4.0 | 10 | 1.6 | 19 | -x |
| `marker__mznenj` | `x2` | borderline | 0.8 | 3.8 | 8 | 1.2 | 17 | +x |
| `marker__mznenj` | `x4` | borderline | 0.6 | 3.1 | 7 | 1.1 | 13 | -x |
| `marker__mznenj` | `x3` | borderline | 0.7 | 3.5 | 7 | 1.0 | 14 | -x |
| `marker__mznenj` | `x5` | borderline | 0.6 | 2.9 | 6 | 1.1 | 11 | -x |
| `marker__mznenj` | `x6` | borderline | 0.5 | 2.7 | 6 | 1.1 | 11 | -x |
| `marker__mznenj` | `xinf` | borderline | 0.7 | 3.3 | 8 | 1.4 | 16 | +x |
| `pepper_shaker__ykujyn` | `zinf` | borderline | 1.3 | 4.4 | 7 | 0.4 | 12 | -z |
| `pepper_shaker__stbqbt` | `z3` | borderline | 0.5 | 1.5 | 3 | 0.0 | 0 |  |
| `salt_shaker__iomwtn` | `z3` | borderline | 1.9 | 5.0 | 8 | 3.9 | 20 | -z |
| `salt_shaker__fonexa` | `x2` | broken | 0.9 | 3.1 | 25 | 9.8 | 29 | -z |
| `salt_shaker__fonexa` | `y2` | broken | 0.9 | 3.1 | 25 | 9.9 | 29 | -z |
| `carrot__nktmff` | `z2` | broken | 1.5 | 3.2 | 5 | 0.0 | 0 |  |
| `colored_pencil__deuvcx` | `x4` | borderline | 1.2 | 4.4 | 13 | 0.3 | 41 | -x |
| `colored_pencil__deuvcx` | `x5` | borderline | 1.1 | 4.3 | 12 | 0.3 | 37 | -x |
| `colored_pencil__egvqng` | `z2` | borderline | 1.5 | 15.3 | 23 | 0.7 | 35 | +x |
| `can_of_baking_mix__orwvfx` | `z2` | broken | 1.0 | 5.9 | 15 | 15.9 | 16 | +z |
| `can_of_baking_mix__rxorlp` | `z2` | borderline | 0.8 | 5.2 | 12 | 1.3 | 12 | +z |
| `teapot__foaehs` | `x2` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `y2` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `z2` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `x4` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `z4` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `x3` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `z3` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `x5` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `z5` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `z6` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `zinf` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `teapot__jlalfc` | `x2` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `teapot__jlalfc` | `y2` | broken | 0.1 | 0.3 | 3 | 0.0 | 0 |  |
| `teapot__jlalfc` | `z2` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `teapot__jlalfc` | `z4` | broken | 0.1 | 0.2 | 3 | 0.0 | 0 |  |
| `teapot__jlalfc` | `z3` | broken | 0.1 | 0.3 | 3 | 0.0 | 0 |  |
| `teapot__jlalfc` | `z5` | broken | 0.1 | 0.2 | 3 | 0.0 | 0 |  |
| `teapot__jlalfc` | `z6` | broken | 0.1 | 0.2 | 3 | 0.0 | 0 |  |
| `teapot__jlalfc` | `zinf` | broken | 0.1 | 0.3 | 3 | 0.0 | 0 |  |

## Agreement with the reviewer verdicts (review/*.json, judged on the current texture class)

- DISAGREE reviewer broken / survey preserved: 19
- DISAGREE reviewer preserved / survey broken: 77
- agree (broken): 224
- agree (preserved): 47
- n/a (unsure or not scored): 4
- reviewer broken / survey borderline: 39
- reviewer preserved / survey borderline: 44

| batch | key | tag | reviewer | survey | note |
|---|---|---|---|---|---|
| rev_00 | `apple__agveuv` | `x5` | preserved | broken | Lumpy but uniformly green apple: stem and calyx sit on the x axis and stay put, no skin bl |
| rev_00 | `baseball_bat__yalgei` | `xinf` | preserved | broken | Unprinted wooden bat: the grain profile along the barrel is identical in both rows and onl |
| rev_00 | `beefsteak_tomato__pnrdxh` | `z6` | preserved | broken | Uniformly dark-red lumpy body of revolution; the small green calyx sits on the z axis and  |
| rev_00 | `bottle_of_black_pepper__honise` | `z4` | preserved | broken | Glass body absent; the plain round cap and the square block of peppercorns map onto themse |
| rev_01 | `bottle_of_lotion__tkryrh` | `zinf` | preserved | broken | Unprinted dark-navy 12-gon bottle with a plain black cap; the rotated silhouette differs o |
| rev_01 | `bottle_of_milk__ddvsgl` | `x2` | preserved | broken | The same red 'MILK / niu-nai + cow logo' print is on both z faces and the same small red t |
| rev_01 | `bottle_of_mustard_seeds__grryaf` | `zinf` | preserved | broken | Plain black lid and a uniform seed-speckle body texture with no label or readable feature; |
| rev_01 | `bottle_of_pop__sevoto` | `z2` | preserved | broken | The red/orange logo and swoosh are printed on both front and back of the grey label and th |
| rev_01 | `bottle_of_poppy_seeds__xdtrgi` | `zinf` | preserved | broken | Black lid and a uniform brown seed-speckle body with no label; the speckle is noise and th |
| rev_01 | `bottle_of_seasoning__vqwqqv` | `zinf` | preserved | broken | Black lid and a uniform brown body with no label or print; round shape, silhouettes match  |
| rev_01 | `bottle_of_water__cyzaue` | `zinf` | preserved | broken | Unprinted black hexagonal-prism bottle (44 faces) with a plain cap; only the polygon corne |
| rev_01 | `bottle_of_water__emquat` | `zinf` | preserved | broken | Unprinted grey-blue body with a round white-topped cap and concentric base, silhouettes id |
| rev_01 | `bottle_of_water__lkpaas` | `zinf` | preserved | broken | All-glass bottle drawn in a uniform tint, judged on shape only: a body of revolution with  |
| rev_02 | `bowl__belcml` | `zinf` | preserved | broken | Uniform black octagonal bowl with no print, seam or interior feature; only the polygon cor |
| rev_02 | `bowl__eawgwj` | `z6` | preserved | broken | Uniform grey octagonal bowl; the 60° row differs only by the misaligned polygon corners at |
| rev_02 | `box_of_cookies__jtalxs` | `y2` | preserved | broken | Top and bottom faces carry the same print (green logo, red character, curved COOKIES band) |
| rev_02 | `box_of_cookies__lwhbid` | `x2` | preserved | broken | Top and bottom carry the same cartoon print aligned under the flip, the two long sides the |
| rev_02 | `bread_slice__pfggnm` | `y2` | preserved | broken | Both faces are the same tan crumb with a uniform dark crust rim and the outline is mirror- |
| rev_03 | `can_of_soda__wbrrad` | `zinf` | preserved | broken | Uniform dark-red body and a plain grey lid with no readable tab or print; only the facet c |
| rev_03 | `canned_food__foetdd` | `z4` | preserved | broken | All four sides carry the same uniform dark vertical-streak texture with no knot or mark, t |
| rev_03 | `carrot__aucrah` | `xinf` | preserved | broken | Uniform dark-brown tapered rod with no visible skin texture, stem or bend; every camera sh |
| rev_03 | `carrot__qhmmmx` | `xinf` | preserved | broken | Uniform brown body; the green leaf tuft is a three-dimensional spray around the axis, so t |
| rev_03 | `cellulose_tape__gchdhk` | `x2` | preserved | broken | Both flat faces of the plain dark-grey roll look alike; the brown seen through the hole is |
| rev_03 | `cellulose_tape__gchdhk` | `z4` | preserved | broken | Plain dark-grey roll with a uniform brown inner wall; no mark on the faces or rim moves wi |
| rev_03 | `cellulose_tape__gchdhk` | `z5` | preserved | broken | Plain dark-grey roll with a uniform brown inner wall; no mark on the faces or rim moves wi |
| rev_03 | `chopping_board__psabiv` | `x2` | preserved | broken | Flipped over about x the board shows the same plank layout (the lighter plank stays in the |
| rev_03 | `cinnamon_stick__qmlyim` | `z2` | preserved | broken | Uniform unprinted quill with alike forked ends; the end swap leaves the same bar, the resi |
| rev_03 | `cinnamon_stick__qmlyim__scaled` | `x2` | preserved | broken | Uniform unprinted quill; the scaled stick stays on its axis with no wobble and flipped ove |
| rev_03 | `cinnamon_stick__qsqvgk` | `x2` | preserved | broken | Uniform unprinted flat quill with alike forked ends; flipped over it shows the same bar fr |
| rev_03 | `cinnamon_stick__qsqvgk` | `y2` | preserved | broken | Uniform unprinted quill; the half turn about its long axis leaves the same bar, the residu |
| rev_03 | `cinnamon_stick__qsqvgk__scaled` | `y6` | preserved | broken | Unprinted quill; the 60-degree roll changes its apparent thickness by a couple of millimet |
| rev_03 | `cinnamon_stick__qxpzdm` | `y2` | preserved | broken | Uniform unprinted quill; the half turn about its long axis only reverses a bow of about 2  |
| rev_04 | `colored_pencil__acumgp` | `y2` | broken | preserved | The end-to-end flip swaps the sharpened tip with the flat butt end, clearly visible as the |
| rev_04 | `colored_pencil__bnsqcn` | `y2` | broken | preserved | The sharpened tip and the flat end swap ends under the flip, which a person reads at once. |
| rev_04 | `colored_pencil__jnccuz` | `y2` | broken | preserved | The flip moves the pointed tip to the opposite end while the flat end takes its place. |
| rev_04 | `colored_pencil__nssris` | `y2` | broken | preserved | The sharpened tip swaps places with the flat butt end under the end-to-end flip. |
| rev_04 | `dip_candle__gbaons` | `x2` | broken | preserved | The flip puts the wick at the bottom and reverses the taper, so the candle is visibly upsi |
| rev_04 | `dip_candle__gbaons` | `y2` | broken | preserved | The flip about y also swaps the wick end with the base and reverses the taper. |
| rev_04 | `drumstick__jauzlc` | `xinf` | preserved | broken | The stick body is a plain uniform wood-coloured body of revolution with no print, so rolli |
| rev_04 | `flashlight__jtrbkc` | `x6` | preserved | broken | The flashlight renders as a uniform black body of revolution with no visible switch, clip  |
| rev_04 | `footstool__sghekm` | `z2` | preserved | broken | The X-frame legs and the regular dotted seat pattern are the same on both sides, and the o |
| rev_05 | `hamburger_bun__fqxumh` | `zinf` | preserved | broken | Uniform dark-brown top and sides; the underside shows only a faint, roughly concentric mot |
| rev_05 | `hamburger_bun__jeygaq` | `z4` | preserved | broken | Geometrically round; the cut-face crumb mottling is an isotropic noise pattern with a roug |
| rev_05 | `hamburger_bun__jeygaq` | `z5` | preserved | broken | Same as z4: round bun, uniform crust, cut-face crumb texture is unreadable noise with no d |
| rev_05 | `hamburger_bun__jeygaq` | `z6` | preserved | broken | Same as z4: round bun, uniform crust, cut-face crumb texture is unreadable noise with no d |
| rev_05 | `hamper__sijioo` | `z5` | preserved | broken | Same basket, the wire spacing repeats exactly after 72°; no seam or feature anywhere, so t |
| rev_05 | `hardback__bgtbdp` | `x2` | preserved | broken | Front and back covers carry the identical grey cover / green band / maroon label design at |
| rev_05 | `jar_of_curry_powder__ozhftu` | `zinf` | preserved | broken | Round jar with a uniform fine-grained powder texture and a plain dark lid (the glass is om |
| rev_05 | `keyboard__uwacaq` | `z2` | broken | preserved | symmetric_180 (alignment-axial), judged about the +z (up) axis: the faint grey key-legend  |
| rev_05 | `lemon__bmvxas` | `xinf` | preserved | broken | The scar and the nipple sit on the long axis and stay put under any turn about x (the scar |
| rev_05 | `lemon__dsacfa` | `x5` | preserved | broken | Scar and nipple sit on the long axis and do not move under the 72° turn, skin uniform; the |
| rev_05 | `lemon__dsacfa` | `x6` | preserved | broken | Scar and nipple on the axis, uniform skin; the 60° turn shifts the outline by ~2 mm only,  |
| rev_05 | `lemon__gkrfha` | `xinf` | preserved | broken | Scar and nipple sit on the long axis and stay put under any turn about x, skin uniform; th |
| rev_05 | `lemon__hoghjl` | `x6` | preserved | broken | Scar and nipple on the axis, uniform skin; the lemon's own axis is slightly tilted against |
| rev_05 | `lemon__kvgfdl` | `xinf` | preserved | broken | Scar and nipple sit on the long axis and stay put under any turn about x, uniform skin, ex |
| rev_05 | `lemon__ovbcff` | `x6` | preserved | broken | Scar and nipple on the axis, uniform skin; the slight tilt of the lemon's own axis against |
| rev_06 | `lemon__peqwoa` | `xinf` | preserved | broken | Body of revolution about its long x axis: the navel scar sits on the axis and stays put at |
| rev_06 | `lemon__pffkae` | `xinf` | preserved | broken | Round about its long axis: the on-axis navel scar does not move at x101, the skin colour i |
| rev_06 | `lemon__pqweio` | `x4` | preserved | broken | Rolling the lemon 90° about its long axis changes nothing a person could see: the navel sc |
| rev_06 | `lemon__pqweio` | `x5` | preserved | broken | Same as x4: on-axis navel scar, uniform skin, no readable feature moves under a 72° roll. |
| rev_06 | `lemon__pqweio` | `x6` | preserved | broken | Same as x4: on-axis navel scar, uniform skin, the 60° roll is invisible. |
| rev_06 | `lemon__pwqedq` | `x4` | preserved | broken | Roll about the long axis: the on-axis navel scar stays in place, the skin is uniform, the  |
| rev_06 | `lemon__pwqedq` | `x5` | preserved | broken | Same as x4: nothing readable moves under a 72° roll about the long axis. |
| rev_06 | `lemon__pwqedq` | `x6` | preserved | broken | Same as x4: the 60° roll about the long axis is invisible. |
| rev_06 | `lid__qferch` | `zinf` | broken | preserved | A bar-shaped handle/ridge across the top (a faint line in the +z tile, a bump in the side  |
| rev_06 | `marker__ablbdc` | `x5` | broken | preserved | The one-sided cap clip moves with the 72° roll (visible in the −y tile after the roll, abs |
| rev_06 | `marker__ablbdc` | `x6` | broken | preserved | The one-sided cap clip moves with the 60° roll and appears at the cap's lower edge instead |
| rev_06 | `marker__erhlbb` | `x6` | broken | preserved | The one-sided cap clip moves with the 60° roll (appears in the −y tile instead of the +z t |
| rev_07 | `marker__zvvovh` | `x4` | broken | preserved | The thin light pocket clip on the cap swings to another side of the barrel (seen along the |
| rev_07 | `marker__zvvovh` | `x5` | broken | preserved | Same cap clip: after the 72-degree turn it shows on the +y side instead of the +z side, a  |
| rev_07 | `marker__zvvovh` | `x6` | broken | preserved | Same cap clip: after the 60-degree turn it is visible from +y and gone from +z, so the mar |
| rev_07 | `notebook__eijrqw` | `y2` | preserved | broken | Both covers are plain black and the spiral stays along the same long edge after the half t |
| rev_07 | `pack_of_chocolate_bar__bxgewr` | `z2` | broken | preserved | The brand text printed on the wrapper reads upside down after the half turn about z, on bo |
| rev_07 | `pack_of_chocolate_bar__tidcst` | `x2` | preserved | broken | The wrapper carries the same print (roundel emblem at the same end, dot row and swoosh) on |
| rev_07 | `pack_of_cigarettes__ohpqua` | `z2` | preserved | broken | Both end faces carry the same black band plus red chevron, the long sides the same stripe  |
| rev_07 | `pack_of_kielbasa__rvhpal` | `x2` | preserved | broken | The label band's heart emblem is printed twice around the sausage so it appears identicall |
| rev_07 | `paperback_book__aubynm` | `x2` | preserved | broken | The back cover carries the same sketch and title strip as the front, aligned under the fli |
| rev_07 | `paperback_book__eqzjjk` | `x2` | preserved | broken | The same bold cover graphic and title strip appear on both covers in the same orientation, |
| rev_07 | `paperback_book__jhaltj` | `x2` | preserved | broken | The 'DIE' roundel and title strip are identical on the front and back covers, so the flipp |
| rev_07 | `paperback_book__ncjkcg` | `x2` | preserved | broken | The same cover image panel and title strip are on both covers in the same orientation; not |
| rev_07 | `paperback_book__phsgun` | `x2` | preserved | broken | Identical roundel-and-title print on both covers, so the flip about x leaves all six views |
| rev_07 | `paperback_book__vciijb` | `x2` | preserved | broken | Both covers carry the same print in the same orientation and the spine stays on the same e |
| rev_08 | `pencil__rzscao` | `y2` | broken | preserved | The end-to-end flip swaps the sharpened conical tip with the flat cut end: in the +y/-y an |
| rev_08 | `pencil_case__acikzo` | `z2` | preserved | broken | Both ends carry the same wire loop, the zipper runs the full length and the brown/red band |
| rev_08 | `pepper_shaker__tgcwtr` | `zinf` | preserved | broken | Black cap, uniformly speckled dark-green body (noise texture, not a print) and plain black |
| rev_08 | `plate__bgxzec` | `x2` | preserved | broken | Uniform grey 8 mm-thick disc: top and bottom faces are identical in colour and the flipped |
| rev_08 | `plate__bgxzec` | `y2` | preserved | broken | Same as x2: plain grey disc, both faces alike, the y180 tiles are indistinguishable from t |
| rev_08 | `plate__luhkiz` | `zinf` | preserved | broken | Dark grey plate with a fine, uniform beaded black rim (dozens of identical beads, no print |
| rev_08 | `public_trash_can__leazin` | `zinf` | preserved | broken | A cage of about twenty identical, evenly spaced bars and slats with no label, opening or s |
| rev_08 | `salt_shaker__iomwtn` | `zinf` | preserved | broken | Plain dark turned-wood mill with a black knob: no label, seam or readable grain from any s |
| rev_09 | `towel_rack__crivoa` | `x2` | preserved | broken | Judged the half turn about the +x (front) axis: a plain black bar with seven identical peg |
| rev_10 | `toy_dice__bjrxkj` | `x2` | broken | preserved | The fish outline on the +x/-x faces is rotated 180 degrees in place (tail swaps sides, bel |
| rev_10 | `toy_dice__bjrxkj` | `z2` | broken | preserved | The side fish and I faces do match, but the 6 on the top (and bottom) face is turned into  |
| rev_10 | `vase__aegxpb` | `zinf` | preserved | broken | Round dark vase with horizontal banded texture that runs continuously around the body; onl |
| rev_10 | `vase__stqkvx` | `zinf` | broken | preserved | All-glass vase judged by shape: the rim is scalloped with six lobes clearly visible from a |
| rev_10 | `whiteboard__gvkgsi` | `x2` | preserved | broken | Plain white board in a uniform thin frame with a plain grey back and no tray, print or asy |
