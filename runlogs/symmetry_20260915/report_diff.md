# Survey re-run diff: runlogs/symmetry_20260915/superseded/v1/symmetry.json -> current

## Texture-class flips per (model, tag) with a texture score in both runs

| tag | preserved -> broken | preserved -> borderline | borderline -> broken | borderline -> preserved | broken -> borderline | broken -> preserved |
|---|---|---|---|---|---|---|
| `x2` | 119 | 24 | 105 | 9 | 2 | 5 |
| `y2` | 79 | 5 | 102 | 9 | 3 | 3 |
| `z2` | 132 | 35 | 126 | 16 | 8 | 0 |
| `x4` | 33 | 19 | 30 | 2 | 0 | 3 |
| `y4` | 11 | 1 | 26 | 2 | 1 | 0 |
| `z4` | 85 | 19 | 65 | 6 | 3 | 0 |
| `x3` | 40 | 16 | 27 | 2 | 0 | 1 |
| `y3` | 12 | 1 | 22 | 0 | 1 | 0 |
| `z3` | 84 | 16 | 52 | 5 | 1 | 0 |
| `x5` | 22 | 20 | 31 | 0 | 0 | 2 |
| `y5` | 11 | 1 | 21 | 3 | 0 | 0 |
| `z5` | 98 | 14 | 57 | 6 | 2 | 0 |
| `x6` | 36 | 17 | 26 | 0 | 0 | 1 |
| `y6` | 11 | 2 | 22 | 3 | 2 | 0 |
| `z6` | 89 | 15 | 58 | 6 | 2 | 0 |
| `xinf` | 28 | 13 | 29 | 3 | 1 | 4 |
| `yinf` | 9 | 1 | 19 | 0 | 1 | 0 |
| `zinf` | 77 | 11 | 48 | 4 | 1 | 0 |
| `full` | 3 | 0 | 2 | 0 | 0 | 0 |

Totals: borderline -> broken: 868, borderline -> preserved: 76, broken -> borderline: 28, broken -> preserved: 19, preserved -> borderline: 230, preserved -> broken: 979

## Final verdicts

352 models changed `symmetry` / `symmetry_near`. Tags removed from `symmetry`: `z2` 50, `zinf` 36, `x2` 31, `z4` 19, `x4` 15, `y2` 14, `z6` 12, `z5` 11, `x6` 7, `xinf` 5, `x5` 4, `z3` 3, `y4` 2, `yinf` 1. Tags added to `symmetry`: none.

101 models gained a `symmetry_near` tag (4 of them were `none` without a near entry before).

| key | before `symmetry` / `symmetry_near` | now `symmetry` / `symmetry_near` |
|---|---|---|
| `trash_can__cdzyew` | `zinf` / `-` | `none` / `-` |
| `trash_can__eahqyq` | `z2` / `-` | `none` / `-` |
| `trash_can__wkxtxh` | `zinf` / `-` | `z2` / `-` |
| `trash_can__djgllo` | `zinf` / `-` | `none` / `zinf` |
| `trash_can__gilsji` | `zinf` / `-` | `none` / `zinf` |
| `trash_can__wklill` | `z2` / `-` | `none` / `-` |
| `hamper__clziqw` | `zinf` / `-` | `none` / `-` |
| `hamper__xjftlf` | `z2` / `-` | `none` / `-` |
| `hamper__pdzaca` | `none` / `zinf` | `none` / `-` |
| `hamper__iofciz` | `z2` / `-` | `none` / `-` |
| `hamper__pcyjbu` | `z2` / `-` | `none` / `-` |
| `hamper__pqzyxx` | `z2` / `-` | `none` / `-` |
| `hamper__sijioo` | `none` / `z3 z5` | `none` / `z3` |
| `hamper__ssfvij` | `z2` / `-` | `none` / `-` |
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
| `lid__wtjthb` | `zinf` / `-` | `none` / `zinf` |
| `lid__muczdt` | `zinf` / `-` | `none` / `-` |
| `lid__fnqhbz` | `z2` / `z5 z6` | `z2` / `-` |
| `lid__kdswwr` | `none` / `x2 y2 z2` | `none` / `-` |
| `lid__snklci` | `z2` / `-` | `none` / `z2` |
| `bottle_of_wine__nhmtvt` | `none` / `z5 z6` | `none` / `-` |
| `bottle_of_wine__bmudli` | `zinf` / `-` | `none` / `-` |
| `bottle_of_wine__cldder` | `none` / `zinf` | `none` / `-` |
| `bottle_of_wine__uoykwx` | `zinf` / `-` | `none` / `zinf` |
| `bottle_of_wine__jdqahn` | `zinf` / `-` | `none` / `zinf` |
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
| `notebook__zkmmnf` | `x2 y2 z2` / `-` | `none` / `x2 y2 z2` |
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
| `hardback__acaxtl` | `x2 y2 z2` / `-` | `none` / `y2` |
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
| `marker__bwwkmb` | `none` / `x5 x6` | `none` / `x6` |
| `pencil__bfkwhb` | `xinf` / `y2 z2` | `xinf` / `-` |
| `table_knife__eebpap` | `none` / `x2` | `none` / `-` |
| `table_knife__jxdfyy` | `none` / `x2` | `none` / `-` |
| `pen__gzrbrx` | `xinf` / `-` | `none` / `xinf` |
| `table_knife__lrdmpf` | `x2` / `y2 z2` | `none` / `-` |
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
| `bottle_of_sage__bterim` | `none` / `-` | `none` / `z4 z5 z6` |
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
| `asparagus__xguktb` | `none` / `-` | `none` / `y6` |
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
| `canister__xcppkc` | `z2` / `-` | `none` / `z2` |
| `vase__aegxpb` | `none` / `zinf` | `none` / `-` |
| `vase__atgnsc` | `none` / `zinf` | `none` / `-` |
| `tissue_dispenser__nflhrf` | `x2` / `-` | `none` / `-` |
| `jar_of_mayonnaise__ktmjvg` | `none` / `zinf` | `none` / `-` |
| `jar_of_tumeric__miivhi` | `none` / `z4` | `none` / `-` |
| `cap__iqeyba` | `zinf` / `-` | `z4 z5 z6` / `zinf` |
| `dip_candle__poexjo` | `z6` / `x2 z4 z5` | `none` / `x2 z4 z5 z6` |
| `dip_candle__ghctun` | `zinf` / `-` | `none` / `zinf` |
| `dip_candle__mxjejs` | `x2 zinf` / `-` | `none` / `x2 zinf` |
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
| `keyboard__uwacaq` | `z2` / `x2 y2` | `none` / `z2` |
| `newspaper__edarlp` | `none` / `x2` | `none` / `-` |
| `carton_of_eggs__stxfxb` | `x2` / `-` | `none` / `-` |
| `carton_of_eggs__rixhgu` | `none` / `z2` | `none` / `-` |
| `chopping_board__psabiv` | `none` / `x2` | `none` / `-` |
| `notepad__awcyha` | `none` / `x2` | `none` / `-` |
| `notepad__hirdfd` | `x2` / `-` | `none` / `-` |
| `clipboard__envjqe` | `none` / `z2` | `none` / `-` |
| `clipboard__gmxyfo` | `x2 y2 z2` / `-` | `none` / `x2 z2` |
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
| `tray__zcmnji` | `zinf` / `-` | `none` / `zinf` |
| `tray__mkdcha` | `z2` / `x2 y2` | `z2` / `-` |
| `lemon__pwqedq` | `none` / `y2 x4 x5 x6` | `none` / `-` |
| `half_lime__dtyjkb` | `none` / `zinf` | `none` / `z6` |
| `half_lime__wdadtn` | `none` / `zinf` | `none` / `z2` |
| `bowl__belcml` | `z4` / `zinf` | `z4` / `-` |
| `bowl__eawgwj` | `z4` / `z6` | `z4` / `-` |
| `saucepot__tfzijn` | `z2` / `-` | `none` / `z2` |
| `saucepot__wfryvm` | `z2` / `-` | `none` / `-` |
| `saucepot__vqtkwq` | `z2` / `-` | `none` / `-` |
| `saucepot__uvzmss` | `z2` / `-` | `none` / `z2` |
| `saucepot__chjetk` | `z2` / `zinf` | `z2` / `-` |
| `vase__gopbrh` | `zinf` / `-` | `z4` / `-` |
| `cinnamon_stick__qxpzdm` | `none` / `x2 y2 z2` | `none` / `x2 z2` |
| `cinnamon_stick__qmlyim` | `none` / `x2 z2` | `none` / `x2` |
| `cinnamon_stick__qsqvgk` | `none` / `x2 y2 z2` | `none` / `z2` |
| `cinnamon_stick__cdkjfo` | `none` / `x2 yinf` | `none` / `x2 z2 y4 y5 y6` |
| `cinnamon_stick__smfuqz` | `none` / `x2 yinf` | `none` / `z2 y6` |
| `pencil__hmawzy` | `z2 x5 x6` / `-` | `x5 x6` / `z2 x4` |
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
| `salt_shaker__fonexa` | `zinf` / `-` | `none` / `zinf` |
| `colored_pencil__acumgp` | `x4 x5 x6` / `y2 xinf` | `x5 x6` / `z2` |
| `colored_pencil__bkdqwb` | `x4 x5 x6` / `-` | `x5 x6` / `z2` |
| `colored_pencil__bnsqcn` | `z2 x4 x5 x6` / `y2 xinf` | `z2 x4 x5 x6` / `-` |
| `colored_pencil__deuvcx` | `x4 x5 x6` / `-` | `x6` / `x4 x5` |
| `colored_pencil__egvqng` | `z2 x4 x5 x6` / `-` | `x5 x6` / `z2` |
| `colored_pencil__jdqvdl` | `z2 x4 x5 x6` / `-` | `x5 x6` / `z2` |
| `colored_pencil__jnccuz` | `x4 x5 x6` / `y2 xinf` | `x5 x6` / `z2` |
| `colored_pencil__kadvlg` | `x4 x5 x6` / `-` | `x5 x6` / `z2` |
| `colored_pencil__kgmapz` | `x4 x5 x6` / `xinf` | `x5 x6` / `-` |
| `colored_pencil__nssris` | `z2 x4 x5 x6` / `y2 xinf` | `z2 x4 x5 x6` / `-` |
| `colored_pencil__qrxemk` | `x4 x5 x6` / `xinf` | `x5 x6` / `z2` |
| `can_of_baking_mix__orwvfx` | `z2` / `zinf` | `none` / `-` |
| `can_of_baking_mix__rxorlp` | `z2` / `zinf` | `none` / `z2` |
| `pop_case__gwgaqb` | `none` / `y2` | `none` / `-` |
| `pop_case__qcfxec` | `none` / `z2` | `none` / `-` |
| `pop_case__pldtgy` | `none` / `x2` | `none` / `-` |

## (model, tag) that were texture-preserved and are not any more (1209)

| key | tag | now | MAD | p95 | p99 | blob (x 6x6 mm) | blob mean /255 | view |
|---|---|---|---|---|---|---|---|---|
| `trash_can__cdzyew` | `z2` | broken | 2.0 | 3.5 | 33 | 55.6 | 133 | -x |
| `trash_can__cdzyew` | `z4` | broken | 2.3 | 3.8 | 73 | 55.7 | 134 | -x |
| `trash_can__cdzyew` | `z3` | broken | 2.3 | 4.5 | 70 | 57.6 | 134 | -y |
| `trash_can__cdzyew` | `z5` | broken | 2.3 | 4.7 | 71 | 55.8 | 134 | -x |
| `trash_can__cdzyew` | `z6` | broken | 2.2 | 4.2 | 60 | 55.9 | 134 | -x |
| `trash_can__cdzyew` | `zinf` | broken | 2.2 | 4.4 | 56 | 55.5 | 134 | -x |
| `trash_can__vasiit` | `z4` | broken | 3.4 | 13.9 | 103 | 388.5 | 51 | +x |
| `trash_can__vasiit` | `z3` | broken | 3.6 | 16.1 | 110 | 402.9 | 59 | +x |
| `trash_can__vasiit` | `z5` | broken | 3.2 | 15.7 | 93 | 117.4 | 48 | +x |
| `trash_can__vasiit` | `z6` | broken | 3.1 | 15.8 | 86 | 55.9 | 48 | -x |
| `trash_can__eahqyq` | `z2` | broken | 2.5 | 6.2 | 44 | 88.2 | 64 | +z |
| `trash_can__cjmezk` | `z2` | broken | 1.3 | 6.7 | 33 | 0.1 | 35 | -z |
| `trash_can__cjmezk` | `z4` | broken | 1.4 | 10.4 | 28 | 0.0 | 0 |  |
| `trash_can__cjmezk` | `z3` | broken | 1.5 | 12.9 | 31 | 0.0 | 0 |  |
| `trash_can__cjmezk` | `z5` | broken | 1.3 | 8.9 | 28 | 0.0 | 0 |  |
| `trash_can__cjmezk` | `z6` | broken | 1.1 | 7.1 | 27 | 0.0 | 0 |  |
| `trash_can__cjmezk` | `zinf` | broken | 1.5 | 11.9 | 32 | 0.1 | 35 | -z |
| `trash_can__gxajos` | `z2` | broken | 1.1 | 0.1 | 59 | 110.7 | 97 | -z |
| `trash_can__gxajos` | `z4` | broken | 1.2 | 0.1 | 66 | 76.4 | 86 | +x |
| `trash_can__gxajos` | `z3` | broken | 1.1 | 0.1 | 56 | 122.1 | 96 | +x |
| `trash_can__gxajos` | `z5` | broken | 1.3 | 0.2 | 68 | 86.2 | 83 | -y |
| `trash_can__gxajos` | `z6` | broken | 1.3 | 0.2 | 68 | 69.6 | 87 | -y |
| `trash_can__gxajos` | `zinf` | broken | 1.0 | 0.1 | 55 | 102.6 | 91 | +x |
| `trash_can__uknjdm` | `z2` | broken | 1.2 | 8.7 | 27 | 0.1 | 35 | -z |
| `trash_can__uknjdm` | `z4` | broken | 1.0 | 4.3 | 27 | 0.0 | 0 |  |
| `trash_can__uknjdm` | `z3` | broken | 1.1 | 7.7 | 26 | 0.0 | 0 |  |
| `trash_can__uknjdm` | `z5` | broken | 0.9 | 4.4 | 26 | 0.0 | 0 |  |
| `trash_can__uknjdm` | `z6` | broken | 0.9 | 4.2 | 24 | 0.1 | 34 | -z |
| `trash_can__uknjdm` | `zinf` | broken | 1.2 | 6.7 | 31 | 0.0 | 0 |  |
| `trash_can__wkxtxh` | `z4` | broken | 2.5 | 10.9 | 26 | 0.8 | 24 | +x |
| `trash_can__wkxtxh` | `z3` | broken | 2.4 | 10.5 | 26 | 0.8 | 24 | -y |
| `trash_can__wkxtxh` | `z5` | broken | 2.5 | 11.1 | 26 | 0.8 | 24 | -y |
| `trash_can__wkxtxh` | `z6` | broken | 2.4 | 10.8 | 26 | 0.8 | 24 | -y |
| `trash_can__wkxtxh` | `zinf` | broken | 2.5 | 10.8 | 26 | 0.8 | 24 | -y |
| `trash_can__wklill` | `z2` | broken | 2.9 | 7.3 | 10 | 0.0 | 0 |  |
| `hamper__clziqw` | `z2` | broken | 6.3 | 15.0 | 36 | 168.4 | 46 | -z |
| `hamper__clziqw` | `z4` | broken | 6.4 | 15.1 | 33 | 75.6 | 50 | -z |
| `hamper__clziqw` | `z3` | broken | 6.4 | 15.0 | 35 | 160.9 | 46 | -z |
| `hamper__clziqw` | `z5` | broken | 6.3 | 15.0 | 30 | 109.3 | 58 | -z |
| `hamper__clziqw` | `z6` | broken | 6.4 | 15.2 | 30 | 81.2 | 52 | -z |
| `hamper__clziqw` | `zinf` | broken | 6.4 | 15.1 | 35 | 322.8 | 48 | -z |
| `hamper__xjftlf` | `z2` | broken | 3.9 | 10.4 | 19 | 13.0 | 34 | +x |
| `hamper__iofciz` | `z2` | broken | 0.4 | 2.4 | 7 | 0.0 | 0 |  |
| `hamper__pcyjbu` | `z2` | broken | 4.0 | 9.2 | 19 | 112.1 | 68 | +y |
| `hamper__pqzyxx` | `z2` | broken | 3.6 | 8.7 | 16 | 17.7 | 42 | +x |
| `hamper__wdsnjy` | `x2` | broken | 3.7 | 8.6 | 12 | 0.0 | 0 |  |
| `hamper__wdsnjy` | `y2` | broken | 3.9 | 8.5 | 11 | 0.0 | 0 |  |
| `hamper__ssfvij` | `z2` | broken | 0.4 | 0.8 | 11 | 0.0 | 0 |  |
| `hamper__ssfvij` | `z4` | broken | 0.4 | 0.9 | 11 | 0.0 | 0 |  |
| `taboret__ysrhuw` | `z2` | borderline | 2.8 | 10.9 | 16 | 0.0 | 0 |  |
| `taboret__ysrhuw` | `z4` | broken | 2.9 | 11.9 | 19 | 48.0 | 61 | +z |
| `taboret__ysrhuw` | `z5` | broken | 3.5 | 12.4 | 20 | 56.4 | 61 | +z |
| `taboret__ehaclu` | `z2` | broken | 0.6 | 1.7 | 14 | 8.7 | 61 | +z |
| `taboret__wjrpwg` | `z3` | borderline | 4.6 | 19.7 | 29 | 0.7 | 24 | -z |
| `taboret__vzzmir` | `z2` | broken | 1.7 | 4.2 | 61 | 69.4 | 52 | -z |
| `taboret__vzzmir` | `z4` | broken | 1.3 | 3.7 | 39 | 26.7 | 51 | -z |
| `taboret__vzzmir` | `z3` | broken | 1.8 | 4.5 | 64 | 53.6 | 61 | -z |
| `taboret__vzzmir` | `z5` | broken | 2.1 | 4.6 | 75 | 81.7 | 55 | -z |
| `taboret__vzzmir` | `z6` | broken | 2.4 | 4.9 | 83 | 105.4 | 56 | -z |
| `taboret__vzzmir` | `zinf` | broken | 2.0 | 4.6 | 73 | 46.3 | 50 | -z |
| `taboret__bciekv` | `z2` | borderline | 4.3 | 20.3 | 32 | 0.7 | 26 | +z |
| `taboret__bciekv` | `z4` | borderline | 4.0 | 18.6 | 30 | 0.4 | 36 | +z |
| `ottoman__miftfy` | `z2` | broken | 4.1 | 13.4 | 19 | 1.8 | 24 | -z |
| `ottoman__miftfy` | `z4` | broken | 4.4 | 13.8 | 21 | 7.1 | 38 | -z |
| `ottoman__xcmniq` | `z2` | borderline | 3.9 | 9.7 | 13 | 0.0 | 0 |  |
| `ottoman__qsnawv` | `x2` | broken | 0.7 | 1.9 | 7 | 0.0 | 0 |  |
| `ottoman__qsnawv` | `y2` | broken | 0.7 | 1.9 | 6 | 0.0 | 0 |  |
| `ottoman__qsnawv` | `z2` | borderline | 0.2 | 0.8 | 2 | 0.0 | 0 |  |
| `ottoman__qsnawv` | `x4` | broken | 0.5 | 1.7 | 3 | 0.0 | 0 |  |
| `ottoman__qsnawv` | `y4` | broken | 0.5 | 1.7 | 4 | 0.0 | 0 |  |
| `ottoman__qsnawv` | `z4` | borderline | 0.2 | 0.8 | 2 | 0.0 | 0 |  |
| `ottoman__ycfbsd` | `z2` | broken | 2.6 | 6.7 | 10 | 0.8 | 54 | -z |
| `ottoman__ycfbsd` | `z4` | broken | 2.7 | 6.6 | 10 | 1.8 | 40 | -z |
| `public_trash_can__glzckq` | `z5` | broken | 5.5 | 35.8 | 117 | 173.0 | 154 | +x |
| `public_trash_can__fkosow` | `x2` | broken | 2.6 | 0.1 | 103 | 228.9 | 130 | +x |
| `public_trash_can__fkosow` | `y2` | broken | 2.4 | 0.1 | 96 | 123.8 | 126 | -x |
| `public_trash_can__hxsyxo` | `z2` | broken | 2.1 | 1.2 | 60 | 128.7 | 88 | +x |
| `public_trash_can__hxsyxo` | `z4` | broken | 1.8 | 0.5 | 60 | 128.7 | 88 | +x |
| `public_trash_can__hxsyxo` | `z3` | broken | 1.9 | 1.0 | 60 | 128.7 | 88 | +x |
| `public_trash_can__hxsyxo` | `z5` | broken | 2.0 | 1.0 | 68 | 128.7 | 88 | +x |
| `public_trash_can__hxsyxo` | `z6` | broken | 2.0 | 1.1 | 68 | 128.7 | 88 | +x |
| `public_trash_can__hxsyxo` | `zinf` | broken | 1.5 | 0.7 | 57 | 192.6 | 85 | +x |
| `public_trash_can__leazin` | `z2` | broken | 4.9 | 14.0 | 21 | 0.9 | 7 | +y |
| `public_trash_can__leazin` | `z4` | broken | 5.6 | 14.0 | 21 | 0.4 | 63 | -z |
| `public_trash_can__leazin` | `z5` | broken | 6.0 | 14.6 | 21 | 0.0 | 0 |  |
| `public_trash_can__gvnfgj` | `x2` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `public_trash_can__gvnfgj` | `y2` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `public_trash_can__gvnfgj` | `z4` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `public_trash_can__gvnfgj` | `z5` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `public_trash_can__gvnfgj` | `zinf` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `public_trash_can__aefcem` | `z2` | broken | 4.6 | 13.9 | 27 | 55.0 | 37 | +z |
| `public_trash_can__aefcem` | `z4` | broken | 4.7 | 13.9 | 27 | 33.7 | 31 | +z |
| `public_trash_can__aefcem` | `z3` | broken | 6.1 | 20.4 | 53 | 63.2 | 41 | -z |
| `public_trash_can__aefcem` | `z6` | broken | 6.2 | 21.1 | 53 | 71.0 | 44 | -z |
| `music_stool__dqvlff` | `z2` | broken | 2.7 | 6.7 | 38 | 43.3 | 76 | +y |
| `music_stool__dqvlff` | `z4` | broken | 4.4 | 35.4 | 40 | 98.8 | 72 | +x |
| `carton__cdmmwy` | `x2` | broken | 3.7 | 7.4 | 53 | 120.7 | 57 | -x |
| `carton__cdmmwy` | `y2` | broken | 3.7 | 7.5 | 53 | 120.7 | 57 | -x |
| `carton__cdmmwy` | `z2` | broken | 3.3 | 6.9 | 46 | 115.7 | 60 | +x |
| `backpack__gvbiwl` | `z2` | broken | 5.2 | 14.9 | 54 | 30.9 | 40 | +z |
| `bottle_of_water__migvlt` | `z2` | broken | 6.4 | 30.6 | 47 | 11.6 | 56 | +y |
| `bottle_of_water__cyzaue` | `z2` | broken | 1.4 | 4.0 | 24 | 5.1 | 49 | +z |
| `bottle_of_water__cyzaue` | `z4` | broken | 1.4 | 4.6 | 18 | 3.8 | 42 | +z |
| `bottle_of_water__cyzaue` | `z3` | broken | 1.3 | 4.0 | 20 | 5.7 | 42 | +z |
| `bottle_of_water__cyzaue` | `z5` | broken | 1.2 | 4.2 | 17 | 4.1 | 36 | +z |
| `bottle_of_water__cyzaue` | `z6` | borderline | 1.1 | 3.6 | 16 | 3.2 | 33 | +z |
| `bottle_of_water__cyzaue` | `zinf` | broken | 1.4 | 4.6 | 20 | 5.5 | 40 | +z |
| `bottle_of_water__emquat` | `z2` | broken | 3.8 | 22.7 | 58 | 15.6 | 43 | -z |
| `bottle_of_water__emquat` | `z4` | broken | 3.1 | 18.4 | 50 | 17.5 | 34 | -z |
| `bottle_of_water__emquat` | `z3` | broken | 3.7 | 21.4 | 54 | 17.7 | 35 | -z |
| `bottle_of_water__emquat` | `z5` | broken | 2.9 | 16.0 | 45 | 16.4 | 31 | -z |
| `bottle_of_water__emquat` | `z6` | broken | 2.7 | 14.5 | 41 | 15.0 | 28 | -z |
| `bottle_of_water__emquat` | `zinf` | broken | 3.9 | 22.8 | 58 | 21.6 | 93 | +z |
| `bottle_of_water__lkpaas` | `x2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `y2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `z2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `z4` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `z3` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `z5` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `z6` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `bottle_of_water__lkpaas` | `zinf` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `z2` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `z4` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `z3` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `z5` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `z6` | broken | 0.1 | 0.6 | 2 | 0.0 | 0 |  |
| `bottle_of_water__rmtdxh` | `zinf` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `bottle_of_water__eeaimz` | `z2` | broken | 6.5 | 32.0 | 51 | 22.8 | 64 | +y |
| `bottle_of_water__cytqio` | `z2` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `bottle_of_water__cytqio` | `z4` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `bottle_of_water__cytqio` | `z3` | broken | 0.1 | 0.4 | 3 | 0.0 | 0 |  |
| `bottle_of_water__cytqio` | `z5` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `bottle_of_water__cytqio` | `z6` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `bottle_of_water__cytqio` | `zinf` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `lid__muczdt` | `x2` | broken | 0.3 | 1.0 | 4 | 0.0 | 0 |  |
| `lid__muczdt` | `y2` | broken | 0.3 | 1.0 | 4 | 0.0 | 0 |  |
| `lid__muczdt` | `z2` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `lid__muczdt` | `z4` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `lid__muczdt` | `z3` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `lid__muczdt` | `z5` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `lid__muczdt` | `z6` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `lid__muczdt` | `zinf` | broken | 0.2 | 1.0 | 3 | 0.0 | 0 |  |
| `lid__xdyddo` | `x2` | broken | 1.3 | 3.4 | 5 | 0.0 | 0 |  |
| `lid__xdyddo` | `y2` | broken | 1.3 | 3.3 | 5 | 0.0 | 0 |  |
| `lid__xdyddo` | `z3` | broken | 1.3 | 3.2 | 5 | 0.0 | 0 |  |
| `lid__xdyddo` | `z6` | broken | 1.2 | 3.0 | 4 | 0.0 | 0 |  |
| `lid__fnqhbz` | `z4` | broken | 0.2 | 0.7 | 5 | 0.0 | 0 |  |
| `lid__fnqhbz` | `z3` | broken | 0.2 | 0.8 | 5 | 0.0 | 0 |  |
| `lid__fnqhbz` | `z5` | broken | 0.2 | 0.8 | 5 | 0.0 | 0 |  |
| `lid__fnqhbz` | `z6` | broken | 0.2 | 0.9 | 5 | 0.0 | 0 |  |
| `lid__fnqhbz` | `zinf` | broken | 0.2 | 0.8 | 5 | 0.0 | 0 |  |
| `bottle_of_wine__bmudli` | `x2` | broken | 5.1 | 37.6 | 100 | 22.2 | 41 | +x |
| `bottle_of_wine__bmudli` | `z2` | broken | 3.8 | 22.5 | 65 | 20.1 | 46 | +x |
| `bottle_of_wine__bmudli` | `z4` | broken | 3.0 | 15.0 | 60 | 17.6 | 43 | +y |
| `bottle_of_wine__bmudli` | `z3` | broken | 3.4 | 20.3 | 65 | 22.5 | 44 | +y |
| `bottle_of_wine__bmudli` | `z5` | broken | 2.7 | 12.9 | 56 | 13.9 | 43 | +y |
| `bottle_of_wine__bmudli` | `z6` | broken | 2.6 | 11.4 | 55 | 9.8 | 40 | +x |
| `bottle_of_wine__bmudli` | `zinf` | broken | 3.9 | 24.2 | 66 | 25.5 | 45 | -x |
| `marker__zvvovh` | `xinf` | borderline | 0.7 | 3.4 | 10 | 0.7 | 27 | +x |
| `marker__slgheo` | `x2` | borderline | 1.8 | 7.4 | 30 | 1.7 | 40 | -x |
| `marker__slgheo` | `x4` | borderline | 1.4 | 6.4 | 20 | 1.4 | 37 | -x |
| `marker__slgheo` | `x3` | borderline | 1.5 | 6.8 | 22 | 1.4 | 42 | -x |
| `marker__slgheo` | `x5` | borderline | 1.3 | 5.9 | 16 | 1.3 | 31 | -x |
| `marker__slgheo` | `x6` | borderline | 1.2 | 5.5 | 13 | 1.2 | 28 | -x |
| `marker__slgheo` | `xinf` | borderline | 1.5 | 6.6 | 20 | 1.4 | 39 | -x |
| `marker__erhlbb` | `x2` | borderline | 0.6 | 3.0 | 10 | 1.0 | 29 | +x |
| `marker__erhlbb` | `x4` | borderline | 0.5 | 2.6 | 7 | 0.0 | 0 |  |
| `marker__erhlbb` | `x3` | borderline | 0.6 | 2.9 | 9 | 0.5 | 29 | +x |
| `marker__erhlbb` | `x5` | borderline | 0.5 | 2.3 | 6 | 0.0 | 0 |  |
| `marker__erhlbb` | `xinf` | borderline | 0.6 | 2.6 | 8 | 1.2 | 30 | +x |
| `marker__jlvneb` | `x2` | borderline | 1.1 | 5.6 | 16 | 1.2 | 33 | +x |
| `marker__jlvneb` | `x4` | borderline | 0.9 | 4.6 | 12 | 0.8 | 29 | -x |
| `marker__jlvneb` | `x3` | borderline | 1.1 | 5.2 | 15 | 0.9 | 30 | +x |
| `marker__jlvneb` | `x5` | borderline | 0.8 | 4.2 | 10 | 0.7 | 27 | -x |
| `marker__jlvneb` | `xinf` | borderline | 1.1 | 5.0 | 14 | 1.4 | 33 | +x |
| `marker__kjxpdi` | `z2` | borderline | 0.4 | 2.3 | 5 | 0.0 | 0 |  |
| `marker__mcpkkq` | `x2` | borderline | 1.8 | 8.4 | 25 | 1.2 | 40 | +x |
| `marker__mcpkkq` | `x4` | borderline | 1.5 | 7.1 | 18 | 1.1 | 31 | -x |
| `marker__mcpkkq` | `x3` | borderline | 1.7 | 7.9 | 22 | 1.0 | 33 | -x |
| `marker__mcpkkq` | `x5` | borderline | 1.3 | 6.6 | 16 | 1.1 | 28 | -x |
| `marker__mcpkkq` | `x6` | borderline | 1.3 | 6.6 | 14 | 1.0 | 26 | -x |
| `marker__mcpkkq` | `xinf` | borderline | 1.7 | 7.5 | 21 | 1.4 | 38 | +x |
| `cap__ghwjwe` | `z2` | borderline | 2.4 | 10.5 | 23 | 0.5 | 22 | -y |
| `cap__ghwjwe` | `zinf` | borderline | 2.3 | 10.5 | 23 | 0.8 | 21 | +y |
| `cap__uxpeaz` | `z3` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `cap__uxpeaz` | `z5` | borderline | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `cap__uxpeaz` | `z6` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `cap__uxpeaz` | `zinf` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `can_of_soda__bfrzvk` | `z4` | borderline | 3.5 | 9.5 | 28 | 1.1 | 59 | -z |
| `can_of_soda__bfrzvk` | `z3` | borderline | 3.6 | 10.5 | 20 | 0.3 | 55 | -z |
| `can_of_soda__bfrzvk` | `z5` | borderline | 3.1 | 8.5 | 25 | 0.6 | 67 | -z |
| `can_of_soda__bfrzvk` | `zinf` | borderline | 3.8 | 10.6 | 28 | 0.9 | 53 | -z |
| `can_of_soda__wbrrad` | `z2` | broken | 4.2 | 18.5 | 39 | 9.3 | 55 | -z |
| `can_of_soda__wbrrad` | `z4` | broken | 3.5 | 14.5 | 35 | 8.8 | 75 | -x |
| `can_of_soda__wbrrad` | `z3` | broken | 5.2 | 23.8 | 57 | 10.0 | 72 | +y |
| `can_of_soda__wbrrad` | `z5` | broken | 4.9 | 21.2 | 59 | 10.0 | 69 | +y |
| `can_of_soda__wbrrad` | `z6` | broken | 4.6 | 19.2 | 55 | 8.9 | 72 | +y |
| `can_of_soda__wbrrad` | `zinf` | broken | 5.5 | 24.9 | 62 | 17.2 | 52 | -z |
| `bottle_of_beer__fgzjnb` | `z6` | broken | 6.1 | 41.0 | 62 | 46.7 | 61 | +x |
| `bottle_of_beer__zlmwyn` | `z2` | broken | 4.7 | 15.4 | 25 | 1.9 | 43 | +y |
| `bottle_of_beer__zlmwyn` | `z4` | borderline | 3.3 | 14.5 | 24 | 1.9 | 43 | -y |
| `bottle_of_beer__zlmwyn` | `z3` | broken | 3.9 | 14.9 | 24 | 1.9 | 43 | -y |
| `bottle_of_beer__zlmwyn` | `z5` | borderline | 2.9 | 13.5 | 24 | 1.9 | 43 | -y |
| `bottle_of_beer__zlmwyn` | `z6` | borderline | 2.7 | 12.9 | 24 | 1.9 | 43 | -y |
| `bottle_of_beer__zlmwyn` | `zinf` | broken | 4.6 | 15.3 | 26 | 1.9 | 43 | -y |
| `canned_food__foetdd` | `z2` | broken | 4.3 | 18.3 | 37 | 12.6 | 43 | -z |
| `lemon__bmvxas` | `x2` | broken | 3.1 | 6.9 | 13 | 7.6 | 64 | +x |
| `lemon__bmvxas` | `x3` | broken | 5.8 | 12.3 | 17 | 6.9 | 64 | +x |
| `lemon__bmvxas` | `x6` | broken | 5.6 | 12.1 | 16 | 5.9 | 45 | +x |
| `lemon__vuwehu` | `x2` | broken | 2.1 | 4.0 | 6 | 3.1 | 62 | +x |
| `lemon__vuwehu` | `y2` | borderline | 2.1 | 3.5 | 5 | 3.0 | 60 | -x |
| `lemon__vuwehu` | `z2` | broken | 2.0 | 3.8 | 5 | 1.2 | 77 | +x |
| `lemon__vuwehu` | `x4` | broken | 2.0 | 3.5 | 5 | 4.7 | 62 | +x |
| `lemon__vuwehu` | `x3` | broken | 2.1 | 3.9 | 6 | 3.0 | 61 | +x |
| `lemon__vuwehu` | `x5` | broken | 1.9 | 3.4 | 5 | 4.5 | 64 | +x |
| `lemon__vuwehu` | `x6` | broken | 1.8 | 3.2 | 5 | 4.2 | 66 | +x |
| `lemon__vuwehu` | `xinf` | broken | 2.0 | 3.6 | 5 | 4.7 | 63 | +x |
| `lemon__dsacfa` | `x2` | broken | 4.2 | 9.0 | 20 | 6.7 | 73 | +x |
| `lemon__asdada` | `x2` | broken | 3.1 | 7.5 | 11 | 6.1 | 41 | +x |
| `lemon__asdada` | `x4` | broken | 6.2 | 13.3 | 18 | 5.9 | 36 | +x |
| `lemon__asdada` | `x3` | broken | 5.8 | 12.4 | 16 | 5.5 | 41 | +x |
| `lemon__asdada` | `x5` | broken | 5.9 | 12.5 | 16 | 4.2 | 33 | +x |
| `lemon__asdada` | `x6` | borderline | 5.5 | 11.8 | 16 | 3.5 | 32 | +x |
| `lemon__asdada` | `xinf` | broken | 3.9 | 9.5 | 14 | 6.1 | 42 | +x |
| `lemon__gkrfha` | `x2` | broken | 3.6 | 7.8 | 19 | 8.3 | 73 | +x |
| `lemon__gkrfha` | `x3` | broken | 5.9 | 12.2 | 19 | 7.0 | 75 | +x |
| `lemon__gkrfha` | `x6` | broken | 5.7 | 12.2 | 17 | 6.1 | 58 | +x |
| `lemon__hoghjl` | `x2` | broken | 4.6 | 9.8 | 17 | 6.7 | 74 | +x |
| `lemon__kvgfdl` | `x2` | broken | 3.1 | 7.6 | 11 | 6.0 | 44 | +x |
| `lemon__kvgfdl` | `x4` | broken | 6.1 | 13.1 | 17 | 4.8 | 40 | +x |
| `lemon__kvgfdl` | `x3` | broken | 5.7 | 12.4 | 16 | 5.6 | 42 | +x |
| `lemon__kvgfdl` | `x5` | borderline | 5.9 | 12.5 | 16 | 2.1 | 37 | +x |
| `lemon__kvgfdl` | `x6` | borderline | 5.5 | 11.6 | 15 | 1.8 | 34 | +x |
| `lemon__peqwoa` | `x2` | broken | 3.5 | 8.3 | 16 | 9.1 | 68 | +x |
| `lemon__peqwoa` | `x3` | broken | 5.9 | 12.9 | 18 | 8.2 | 63 | +x |
| `lemon__peqwoa` | `x6` | broken | 5.8 | 12.5 | 17 | 7.0 | 48 | +x |
| `lemon__pffkae` | `x2` | broken | 3.6 | 8.5 | 15 | 9.5 | 64 | +x |
| `lemon__pffkae` | `x3` | broken | 5.9 | 12.8 | 18 | 8.7 | 60 | +x |
| `lemon__pffkae` | `x5` | broken | 6.1 | 13.1 | 17 | 7.2 | 51 | +x |
| `lemon__pffkae` | `x6` | broken | 5.7 | 12.3 | 17 | 7.0 | 46 | +x |
| `lemon__pqweio` | `x2` | broken | 4.5 | 10.0 | 17 | 8.4 | 74 | +x |
| `lemon__pqweio` | `x6` | broken | 5.7 | 12.0 | 17 | 6.5 | 67 | +x |
| `shampoo_dispenser__thurmw` | `z2` | broken | 0.5 | 0.5 | 5 | 4.2 | 45 | +z |
| `shampoo_dispenser__thurmw` | `z4` | broken | 0.4 | 0.6 | 2 | 4.2 | 45 | +z |
| `shampoo_dispenser__thurmw` | `z5` | broken | 1.3 | 7.8 | 23 | 3.3 | 53 | +z |
| `shampoo_dispenser__eblcnw` | `z2` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `shampoo_dispenser__eblcnw` | `z4` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `shampoo_dispenser__eblcnw` | `z3` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `shampoo_dispenser__eblcnw` | `z5` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `shampoo_dispenser__eblcnw` | `z6` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `shampoo_dispenser__eblcnw` | `zinf` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `shampoo_dispenser__zoyyia` | `z2` | borderline | 1.6 | 6.6 | 41 | 3.9 | 28 | +z |
| `shampoo_dispenser__zoyyia` | `z4` | borderline | 1.1 | 4.6 | 27 | 2.4 | 40 | +z |
| `shampoo_dispenser__zoyyia` | `z5` | broken | 1.0 | 4.2 | 23 | 1.9 | 40 | +z |
| `shampoo_dispenser__ndrvvm` | `z2` | broken | 0.2 | 1.0 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__ndrvvm` | `z4` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__ndrvvm` | `z3` | broken | 0.2 | 0.9 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__ndrvvm` | `z5` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__ndrvvm` | `z6` | broken | 0.2 | 0.8 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__ndrvvm` | `zinf` | broken | 0.2 | 0.7 | 4 | 0.0 | 0 |  |
| `shampoo_dispenser__nrthyl` | `z2` | borderline | 0.9 | 4.0 | 23 | 2.0 | 24 | +z |
| `shampoo_dispenser__nrthyl` | `z4` | borderline | 0.7 | 3.2 | 16 | 1.6 | 33 | +z |
| `shampoo_dispenser__nrthyl` | `z3` | broken | 0.8 | 4.1 | 19 | 1.5 | 32 | +z |
| `shampoo_dispenser__nrthyl` | `z5` | broken | 0.6 | 3.5 | 13 | 1.5 | 35 | +z |
| `shampoo_dispenser__nrthyl` | `z6` | broken | 0.6 | 3.2 | 11 | 1.6 | 35 | +z |
| `shampoo_dispenser__nrthyl` | `zinf` | broken | 0.5 | 3.0 | 10 | 1.7 | 36 | +z |
| `shampoo_dispenser__kkbhzw` | `z2` | broken | 0.2 | 1.0 | 1 | 0.0 | 0 |  |
| `shampoo_dispenser__kkbhzw` | `z4` | broken | 0.3 | 1.0 | 1 | 0.0 | 0 |  |
| `shampoo_dispenser__kkbhzw` | `z3` | broken | 0.3 | 1.0 | 1 | 0.0 | 0 |  |
| `shampoo_dispenser__kkbhzw` | `z5` | broken | 0.3 | 1.0 | 2 | 0.0 | 0 |  |
| `shampoo_dispenser__kkbhzw` | `z6` | broken | 0.3 | 1.0 | 1 | 0.0 | 0 |  |
| `shampoo_dispenser__kkbhzw` | `zinf` | broken | 0.3 | 1.0 | 1 | 0.0 | 0 |  |
| `shampoo_dispenser__eywylh` | `z2` | broken | 2.9 | 12.5 | 54 | 11.3 | 109 | +z |
| `shampoo_dispenser__eywylh` | `z4` | broken | 3.1 | 13.6 | 60 | 13.2 | 114 | +z |
| `beaker__effbnc` | `z6` | borderline | 6.9 | 20.9 | 42 | 1.9 | 42 | +z |
| `beaker__dtjmai` | `z3` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `beaker__dtjmai` | `z6` | broken | 0.0 | 0.1 | 0 | 0.0 | 0 |  |
| `apple__obixxh` | `x2` | borderline | 5.1 | 13.6 | 26 | 2.3 | 40 | -x |
| `apple__obixxh` | `y2` | broken | 5.3 | 14.2 | 18 | 0.3 | 26 | -x |
| `apple__obixxh` | `z2` | broken | 5.4 | 13.9 | 24 | 2.3 | 40 | -x |
| `apple__obixxh` | `x4` | borderline | 5.9 | 14.2 | 25 | 2.3 | 40 | -x |
| `apple__obixxh` | `y4` | broken | 5.5 | 13.1 | 27 | 2.0 | 44 | -x |
| `apple__obixxh` | `z4` | broken | 5.8 | 14.3 | 26 | 2.3 | 40 | -y |
| `apple__obixxh` | `x3` | broken | 5.7 | 14.1 | 26 | 2.3 | 40 | -x |
| `apple__obixxh` | `y3` | broken | 5.5 | 13.7 | 25 | 2.3 | 39 | +x |
| `apple__obixxh` | `z3` | broken | 5.8 | 14.1 | 26 | 2.3 | 41 | -y |
| `apple__obixxh` | `x5` | borderline | 5.8 | 13.3 | 25 | 2.3 | 40 | -x |
| `apple__obixxh` | `y5` | broken | 5.5 | 12.9 | 26 | 2.1 | 41 | +z |
| `apple__obixxh` | `z5` | broken | 5.7 | 13.9 | 26 | 2.2 | 42 | -x |
| `apple__obixxh` | `x6` | broken | 5.6 | 13.2 | 25 | 4.7 | 40 | -x |
| `apple__obixxh` | `y6` | borderline | 5.4 | 12.7 | 27 | 2.0 | 40 | +z |
| `apple__obixxh` | `z6` | broken | 5.7 | 13.7 | 25 | 2.2 | 41 | -x |
| `apple__obixxh` | `xinf` | broken | 5.5 | 13.4 | 24 | 4.6 | 40 | -x |
| `apple__obixxh` | `yinf` | broken | 4.0 | 10.3 | 23 | 4.3 | 40 | -x |
| `apple__obixxh` | `zinf` | broken | 3.9 | 9.8 | 23 | 4.4 | 41 | -x |
| `apple__zlxfxt` | `x2` | borderline | 5.1 | 13.6 | 26 | 2.3 | 40 | -x |
| `apple__zlxfxt` | `y2` | broken | 5.3 | 14.2 | 18 | 0.2 | 34 | -x |
| `apple__zlxfxt` | `z2` | broken | 5.4 | 13.9 | 24 | 2.3 | 40 | -x |
| `apple__zlxfxt` | `x4` | borderline | 5.9 | 14.2 | 26 | 2.3 | 40 | -x |
| `apple__zlxfxt` | `y4` | broken | 5.5 | 13.2 | 27 | 2.0 | 44 | -x |
| `apple__zlxfxt` | `z4` | broken | 5.8 | 14.3 | 26 | 2.3 | 40 | -y |
| `apple__zlxfxt` | `x3` | broken | 5.7 | 14.1 | 26 | 2.3 | 40 | -x |
| `apple__zlxfxt` | `y3` | broken | 5.5 | 13.7 | 25 | 2.3 | 39 | +x |
| `apple__zlxfxt` | `z3` | broken | 5.8 | 14.0 | 26 | 2.3 | 41 | -y |
| `apple__zlxfxt` | `x5` | borderline | 5.8 | 13.3 | 25 | 2.3 | 39 | -x |
| `apple__zlxfxt` | `y5` | broken | 5.5 | 12.9 | 26 | 2.1 | 41 | +z |
| `apple__zlxfxt` | `z5` | broken | 5.7 | 13.9 | 26 | 2.1 | 42 | -x |
| `apple__zlxfxt` | `x6` | broken | 5.6 | 13.2 | 25 | 4.7 | 40 | -x |
| `apple__zlxfxt` | `y6` | broken | 5.4 | 12.8 | 27 | 2.0 | 40 | +z |
| `apple__zlxfxt` | `z6` | broken | 5.7 | 13.7 | 25 | 2.2 | 41 | -x |
| `apple__zlxfxt` | `yinf` | broken | 4.0 | 10.3 | 23 | 4.4 | 40 | -x |
| `apple__zlxfxt` | `zinf` | broken | 3.9 | 9.8 | 23 | 4.4 | 41 | -x |
| `beeswax_candle__nhdnje` | `z2` | borderline | 0.7 | 2.8 | 8 | 2.8 | 34 | +z |
| `beeswax_candle__nhdnje` | `z4` | borderline | 3.1 | 13.1 | 15 | 1.7 | 29 | +z |
| `beeswax_candle__nhdnje` | `z3` | borderline | 3.0 | 13.2 | 15 | 2.3 | 31 | +z |
| `beeswax_candle__nhdnje` | `z5` | borderline | 3.0 | 13.2 | 15 | 1.3 | 26 | +z |
| `beeswax_candle__nhdnje` | `z6` | borderline | 2.8 | 13.1 | 15 | 1.1 | 25 | +z |
| `beeswax_candle__nhdnje` | `zinf` | borderline | 1.6 | 7.5 | 9 | 2.7 | 34 | +z |
| `beeswax_candle__oimgmh` | `z2` | borderline | 0.9 | 4.0 | 8 | 1.2 | 31 | +z |
| `beeswax_candle__oimgmh` | `z3` | borderline | 2.3 | 12.2 | 15 | 0.5 | 32 | +z |
| `beeswax_candle__oimgmh` | `zinf` | borderline | 1.4 | 7.1 | 9 | 1.2 | 32 | +z |
| `beeswax_candle__ouzkdj` | `z2` | borderline | 2.1 | 11.5 | 14 | 1.6 | 31 | +z |
| `beeswax_candle__ouzkdj` | `z4` | borderline | 2.3 | 12.6 | 15 | 1.8 | 29 | +z |
| `beeswax_candle__ouzkdj` | `z3` | borderline | 2.3 | 12.6 | 15 | 2.0 | 30 | +z |
| `beeswax_candle__ouzkdj` | `z5` | borderline | 2.3 | 12.9 | 15 | 1.6 | 28 | +z |
| `beeswax_candle__ouzkdj` | `z6` | borderline | 2.2 | 13.0 | 15 | 1.4 | 27 | +z |
| `beeswax_candle__ouzkdj` | `zinf` | borderline | 2.4 | 12.6 | 15 | 1.9 | 32 | +z |
| `beeswax_candle__nxewyk` | `z2` | borderline | 1.3 | 8.1 | 14 | 2.8 | 37 | +z |
| `beeswax_candle__nxewyk` | `z4` | borderline | 1.5 | 9.7 | 15 | 2.1 | 32 | +z |
| `beeswax_candle__nxewyk` | `z3` | borderline | 1.6 | 9.9 | 15 | 2.5 | 36 | +z |
| `beeswax_candle__nxewyk` | `z5` | borderline | 1.4 | 9.4 | 15 | 1.7 | 31 | +z |
| `beeswax_candle__nxewyk` | `z6` | borderline | 1.3 | 8.9 | 14 | 1.1 | 30 | +z |
| `beeswax_candle__nxewyk` | `zinf` | borderline | 1.5 | 9.5 | 15 | 2.8 | 37 | +z |
| `beeswax_candle__pfewit` | `x2` | broken | 4.7 | 16.1 | 104 | 12.7 | 73 | +y |
| `beeswax_candle__pfewit` | `y2` | broken | 4.6 | 16.4 | 104 | 12.7 | 73 | -x |
| `beeswax_candle__pfewit` | `z2` | borderline | 0.7 | 3.0 | 7 | 1.3 | 36 | +z |
| `beeswax_candle__pfewit` | `z4` | borderline | 0.7 | 3.1 | 7 | 1.9 | 37 | +z |
| `beeswax_candle__pfewit` | `z3` | borderline | 0.7 | 3.1 | 7 | 1.9 | 38 | +z |
| `beeswax_candle__pfewit` | `z5` | borderline | 0.7 | 2.8 | 7 | 1.6 | 35 | +z |
| `beeswax_candle__pfewit` | `z6` | borderline | 0.7 | 2.9 | 7 | 1.4 | 33 | +z |
| `beeswax_candle__pfewit` | `zinf` | borderline | 0.8 | 3.3 | 7 | 2.0 | 37 | +z |
| `bottle_of_pop__twtsry` | `z4` | broken | 1.3 | 6.4 | 32 | 9.9 | 44 | -z |
| `bottle_of_pop__twtsry` | `z3` | broken | 1.1 | 4.8 | 27 | 8.9 | 41 | -z |
| `bottle_of_pop__twtsry` | `z5` | broken | 1.3 | 5.6 | 34 | 8.1 | 46 | -z |
| `bottle_of_pop__twtsry` | `z6` | broken | 1.3 | 5.2 | 33 | 7.1 | 45 | -z |
| `bottle_of_pop__twtsry` | `zinf` | broken | 1.3 | 5.8 | 31 | 10.3 | 43 | -z |
| `bell_pepper__ihctxa` | `x6` | broken | 4.7 | 8.1 | 74 | 17.3 | 71 | +x |
| `bottle_of_catsup__ahoiqe` | `x2` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `bottle_of_catsup__ahoiqe` | `y2` | broken | 0.1 | 0.9 | 3 | 0.0 | 0 |  |
| `bottle_of_catsup__bcqfxb` | `z2` | borderline | 0.7 | 2.1 | 24 | 2.6 | 23 | +x |
| `bar_soap__feqemg` | `z2` | borderline | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `beefsteak_tomato__pnrdxh` | `x2` | broken | 2.0 | 3.7 | 61 | 15.2 | 90 | +z |
| `beefsteak_tomato__pnrdxh` | `y2` | broken | 2.0 | 4.0 | 61 | 15.2 | 90 | +z |
| `beefsteak_tomato__pnrdxh` | `z2` | broken | 1.7 | 4.1 | 55 | 20.4 | 77 | +z |
| `beefsteak_tomato__pnrdxh` | `z4` | broken | 1.4 | 3.4 | 45 | 19.1 | 67 | +z |
| `beefsteak_tomato__pnrdxh` | `z3` | broken | 1.5 | 3.7 | 51 | 19.6 | 73 | +z |
| `beefsteak_tomato__pnrdxh` | `z5` | broken | 1.2 | 3.0 | 40 | 18.3 | 62 | +z |
| `beefsteak_tomato__pnrdxh` | `z6` | broken | 1.1 | 2.8 | 36 | 18.0 | 56 | +z |
| `beefsteak_tomato__pnrdxh` | `zinf` | broken | 1.7 | 4.2 | 55 | 20.4 | 77 | +z |
| `beefsteak_tomato__altlfz` | `x2` | broken | 4.9 | 11.3 | 14 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `y2` | broken | 5.2 | 10.7 | 13 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `z2` | broken | 5.4 | 9.8 | 13 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `x4` | broken | 5.3 | 11.0 | 13 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `y4` | broken | 5.1 | 10.0 | 12 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `z4` | broken | 5.0 | 10.1 | 13 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `x3` | broken | 5.0 | 9.6 | 12 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `y3` | broken | 5.2 | 10.8 | 13 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `z3` | broken | 5.4 | 10.5 | 14 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `x5` | broken | 5.4 | 11.2 | 14 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `y5` | broken | 5.1 | 9.6 | 13 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `z5` | broken | 5.0 | 10.1 | 13 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `x6` | broken | 5.4 | 11.1 | 14 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `y6` | broken | 5.1 | 9.9 | 13 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `z6` | broken | 5.1 | 10.5 | 13 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `xinf` | broken | 5.4 | 11.0 | 14 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `yinf` | broken | 4.9 | 9.6 | 12 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `zinf` | broken | 5.5 | 10.6 | 14 | 0.0 | 0 |  |
| `beefsteak_tomato__altlfz` | `full` | broken | 5.2 | 10.7 | 15 | 0.0 | 0 |  |
| `beefsteak_tomato__ogpans` | `y2` | broken | 3.0 | 20.2 | 70 | 14.0 | 62 | +z |
| `beefsteak_tomato__ogpans` | `z2` | broken | 3.5 | 30.8 | 70 | 21.5 | 60 | +z |
| `bottle_of_olive_oil__cqycjk` | `z2` | broken | 3.4 | 23.3 | 72 | 18.0 | 65 | +x |
| `bottle_of_olive_oil__cqycjk` | `z4` | broken | 2.9 | 17.9 | 69 | 9.4 | 57 | -x |
| `bottle_of_olive_oil__cqycjk` | `z3` | broken | 2.3 | 9.1 | 65 | 9.4 | 57 | -x |
| `bottle_of_olive_oil__cqycjk` | `z5` | broken | 3.5 | 24.4 | 71 | 17.1 | 66 | +x |
| `bottle_of_olive_oil__cqycjk` | `z6` | broken | 3.6 | 26.0 | 72 | 19.3 | 66 | +x |
| `bottle_of_olive_oil__cqycjk` | `zinf` | broken | 3.6 | 25.7 | 71 | 23.9 | 60 | -y |
| `bottle_of_olive_oil__ksxqkk` | `z2` | broken | 3.4 | 18.9 | 43 | 17.2 | 60 | -x |
| `spray_can__bgalkx` | `x2` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `y2` | broken | 0.1 | 0.6 | 3 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `z2` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `z4` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `z3` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `z5` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `spray_can__bgalkx` | `zinf` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `spray_can__tehmzy` | `z2` | borderline | 3.0 | 6.9 | 17 | 4.0 | 26 | +z |
| `spray_can__tehmzy` | `z4` | borderline | 3.5 | 7.6 | 14 | 1.8 | 24 | +z |
| `spray_can__tehmzy` | `z3` | borderline | 3.5 | 7.8 | 14 | 3.0 | 25 | +z |
| `spray_can__tehmzy` | `zinf` | broken | 3.2 | 7.2 | 17 | 4.0 | 27 | +z |
| `vidalia_onion__dejqhf` | `x2` | borderline | 3.8 | 8.7 | 13 | 1.2 | 36 | +x |
| `vidalia_onion__dejqhf` | `y2` | broken | 4.6 | 11.4 | 25 | 1.2 | 32 | +x |
| `vidalia_onion__dejqhf` | `z2` | broken | 4.7 | 11.3 | 24 | 1.1 | 32 | +x |
| `vidalia_onion__dejqhf` | `x4` | borderline | 4.1 | 9.9 | 13 | 1.1 | 34 | +x |
| `vidalia_onion__dejqhf` | `y4` | broken | 4.4 | 10.4 | 25 | 1.2 | 34 | -z |
| `vidalia_onion__dejqhf` | `z4` | broken | 4.4 | 10.3 | 24 | 1.3 | 32 | +y |
| `vidalia_onion__dejqhf` | `x3` | borderline | 3.9 | 9.7 | 13 | 0.6 | 27 | +x |
| `vidalia_onion__dejqhf` | `y3` | broken | 4.5 | 10.9 | 21 | 1.0 | 33 | +x |
| `vidalia_onion__dejqhf` | `z3` | broken | 4.7 | 11.2 | 26 | 1.8 | 36 | +y |
| `vidalia_onion__dejqhf` | `x5` | borderline | 4.1 | 9.5 | 13 | 1.4 | 33 | +x |
| `vidalia_onion__dejqhf` | `y5` | broken | 4.2 | 9.5 | 24 | 1.3 | 32 | -z |
| `vidalia_onion__dejqhf` | `z5` | broken | 4.3 | 10.3 | 26 | 1.3 | 30 | +x |
| `vidalia_onion__dejqhf` | `x6` | borderline | 4.1 | 9.7 | 13 | 1.3 | 33 | +x |
| `vidalia_onion__dejqhf` | `y6` | broken | 4.1 | 9.2 | 23 | 1.1 | 31 | +x |
| `vidalia_onion__dejqhf` | `z6` | broken | 4.2 | 10.1 | 25 | 1.3 | 32 | +x |
| `vidalia_onion__dejqhf` | `xinf` | borderline | 4.0 | 9.3 | 13 | 1.4 | 38 | +x |
| `vidalia_onion__dejqhf` | `yinf` | broken | 3.3 | 7.9 | 21 | 0.9 | 39 | +x |
| `vidalia_onion__dejqhf` | `zinf` | broken | 3.9 | 9.1 | 23 | 1.4 | 32 | +x |
| `vidalia_onion__dejqhf` | `full` | broken | 3.9 | 9.5 | 24 | 1.1 | 35 | +x |
| `pack_of_cigarettes__kienyo` | `z2` | broken | 3.0 | 7.0 | 31 | 4.3 | 42 | +x |
| `pack_of_cigarettes__wjfbhw` | `z2` | borderline | 4.2 | 12.2 | 18 | 1.8 | 30 | +x |
| `orange__ucstpm` | `x2` | broken | 4.4 | 7.9 | 20 | 4.6 | 55 | +z |
| `orange__ucstpm` | `y2` | broken | 4.4 | 8.8 | 20 | 4.6 | 55 | -z |
| `orange__ucstpm` | `z2` | broken | 4.1 | 8.4 | 11 | 6.5 | 41 | +z |
| `orange__ucstpm` | `x4` | broken | 4.0 | 7.8 | 25 | 4.4 | 54 | +z |
| `orange__ucstpm` | `y4` | broken | 4.8 | 9.4 | 18 | 4.5 | 55 | +z |
| `orange__ucstpm` | `z4` | broken | 4.4 | 9.4 | 12 | 5.7 | 34 | +z |
| `orange__ucstpm` | `x3` | broken | 4.2 | 8.0 | 24 | 4.7 | 55 | +z |
| `orange__ucstpm` | `y3` | broken | 4.6 | 9.0 | 21 | 4.4 | 54 | +z |
| `orange__ucstpm` | `z3` | broken | 4.3 | 8.7 | 11 | 6.2 | 39 | +z |
| `orange__ucstpm` | `x5` | broken | 4.2 | 8.3 | 27 | 4.7 | 55 | -y |
| `orange__ucstpm` | `y5` | broken | 4.8 | 9.2 | 25 | 4.8 | 55 | +x |
| `orange__ucstpm` | `z5` | broken | 4.3 | 9.2 | 12 | 5.2 | 32 | +z |
| `orange__ucstpm` | `x6` | broken | 4.2 | 8.9 | 30 | 4.8 | 57 | -y |
| `orange__ucstpm` | `y6` | broken | 4.6 | 8.9 | 29 | 5.0 | 54 | +x |
| `orange__ucstpm` | `z6` | borderline | 4.1 | 8.8 | 11 | 2.5 | 30 | +z |
| `orange__ucstpm` | `xinf` | broken | 2.9 | 6.4 | 30 | 10.2 | 55 | +z |
| `orange__ucstpm` | `yinf` | broken | 2.8 | 5.3 | 29 | 9.6 | 55 | +z |
| `orange__ucstpm` | `zinf` | broken | 4.1 | 8.2 | 11 | 6.5 | 41 | +z |
| `orange__ucstpm` | `full` | broken | 4.2 | 8.4 | 31 | 5.2 | 54 | -y |
| `bottle_of_salsa__nafwlf` | `z2` | broken | 2.9 | 10.3 | 19 | 6.2 | 30 | +z |
| `bottle_of_salsa__nafwlf` | `z4` | borderline | 2.6 | 10.0 | 18 | 1.8 | 35 | +x |
| `bottle_of_salsa__nafwlf` | `z3` | borderline | 3.2 | 10.4 | 19 | 2.4 | 26 | +z |
| `bottle_of_salsa__nafwlf` | `z5` | borderline | 3.0 | 10.0 | 18 | 1.6 | 29 | +z |
| `bottle_of_salsa__nafwlf` | `z6` | borderline | 3.1 | 10.1 | 18 | 1.5 | 33 | +x |
| `bottle_of_salsa__nafwlf` | `zinf` | broken | 3.1 | 10.3 | 19 | 7.3 | 28 | +z |
| `coffee_cup__dkxddg` | `z4` | broken | 2.5 | 16.2 | 46 | 9.8 | 55 | +z |
| `coffee_cup__dkxddg` | `z3` | broken | 2.8 | 19.3 | 53 | 9.7 | 59 | +z |
| `coffee_cup__dkxddg` | `z5` | broken | 2.2 | 14.1 | 41 | 9.6 | 49 | +z |
| `coffee_cup__dkxddg` | `z6` | broken | 1.9 | 11.8 | 34 | 9.3 | 44 | +z |
| `coffee_cup__ykuftq` | `x2` | broken | 2.2 | 11.8 | 34 | 7.8 | 43 | +y |
| `coffee_cup__rypdvd` | `z4` | broken | 0.9 | 4.9 | 21 | 4.5 | 29 | +z |
| `coffee_cup__rypdvd` | `z3` | broken | 1.0 | 5.7 | 24 | 4.5 | 30 | +z |
| `coffee_cup__rypdvd` | `z5` | broken | 0.8 | 4.0 | 17 | 7.2 | 29 | +z |
| `coffee_cup__rypdvd` | `z6` | broken | 0.7 | 3.4 | 15 | 4.5 | 27 | +z |
| `coffee_cup__fgizgn` | `z4` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `coffee_cup__fgizgn` | `z5` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `coffee_cup__fgizgn` | `z6` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `z2` | broken | 0.1 | 0.1 | 1 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `z4` | broken | 0.1 | 0.1 | 1 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `z3` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `z5` | broken | 0.1 | 0.1 | 1 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `z6` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `coffee_cup__siksnl` | `zinf` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `z2` | broken | 0.1 | 0.4 | 1 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `z4` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `z3` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `z5` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `z6` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `coffee_cup__ibhhfj` | `zinf` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__ppzttc` | `z4` | broken | 2.8 | 19.0 | 26 | 7.6 | 34 | +x |
| `mug__ppzttc` | `z3` | broken | 2.6 | 18.6 | 26 | 4.4 | 36 | -y |
| `mug__ppzttc` | `z5` | broken | 2.8 | 19.6 | 39 | 16.0 | 50 | -y |
| `mug__ppzttc` | `z6` | broken | 2.9 | 20.2 | 37 | 13.1 | 52 | -y |
| `mug__lgxhsc` | `z2` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `mug__lgxhsc` | `z4` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `mug__lgxhsc` | `z3` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `mug__lgxhsc` | `z5` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `mug__lgxhsc` | `z6` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `mug__lgxhsc` | `zinf` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `mug__waqrdy` | `z5` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `mug__waqrdy` | `z6` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `mug__ehnmxj` | `z2` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `mug__ehnmxj` | `z4` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `mug__ehnmxj` | `z3` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `mug__ehnmxj` | `z5` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `mug__ehnmxj` | `z6` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `mug__ehnmxj` | `zinf` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `mug__fapsrj` | `z2` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `mug__fapsrj` | `z4` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `mug__fapsrj` | `z3` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `mug__fapsrj` | `z5` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `mug__fapsrj` | `z6` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `mug__fapsrj` | `zinf` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `mug__ntgftr` | `z4` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `mug__ntgftr` | `z5` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `mug__ntgftr` | `z6` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `mug__ppdqbj` | `y2` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `mug__jgethp` | `z2` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `mug__jgethp` | `z4` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `mug__jgethp` | `z3` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `mug__jgethp` | `z5` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `mug__jgethp` | `z6` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `mug__jgethp` | `zinf` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `statue__cxsvwj` | `z2` | broken | 0.2 | 1.1 | 4 | 0.0 | 0 |  |
| `statue__ponfvo` | `z2` | broken | 5.1 | 31.6 | 46 | 12.9 | 65 | +z |
| `mouse__cduwda` | `z2` | borderline | 1.0 | 0.6 | 31 | 3.3 | 47 | -x |
| `mouse__fjljxp` | `x2` | broken | 0.3 | 0.5 | 11 | 0.0 | 0 |  |
| `mouse__fjljxp` | `z2` | broken | 0.4 | 1.1 | 13 | 0.0 | 0 |  |
| `mouse__pljjtu` | `z2` | borderline | 1.1 | 0.6 | 32 | 3.0 | 47 | +x |
| `mouse__ihpnjx` | `x2` | broken | 0.3 | 0.7 | 12 | 0.0 | 0 |  |
| `mouse__eosngb` | `z2` | broken | 1.0 | 1.5 | 43 | 2.1 | 27 | +x |
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
| `teacup__oxfzfe` | `z2` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `teacup__oxfzfe` | `z4` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `teacup__oxfzfe` | `z3` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `teacup__oxfzfe` | `z5` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `teacup__oxfzfe` | `z6` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `teacup__oxfzfe` | `zinf` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `teacup__vckahe` | `z5` | broken | 3.6 | 25.2 | 54 | 10.6 | 26 | -y |
| `teacup__vckahe` | `z6` | broken | 3.1 | 22.0 | 47 | 9.3 | 24 | -y |
| `teacup__cpozxi` | `z5` | borderline | 3.3 | 12.6 | 22 | 2.5 | 36 | -z |
| `teacup__wopjex` | `z2` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `teacup__wopjex` | `z4` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `teacup__wopjex` | `z3` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `teacup__wopjex` | `z5` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `teacup__wopjex` | `z6` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `teacup__wopjex` | `zinf` | broken | 0.1 | 0.4 | 1 | 0.0 | 0 |  |
| `carving_knife__fqqbop` | `x2` | broken | 0.7 | 3.4 | 11 | 0.0 | 0 |  |
| `carving_knife__fqqbop` | `x6` | broken | 0.8 | 3.6 | 10 | 0.0 | 0 |  |
| `carving_knife__zaihmr` | `x6` | broken | 0.6 | 3.2 | 8 | 0.0 | 0 |  |
| `carving_knife__awvoox` | `x2` | broken | 1.5 | 5.8 | 37 | 3.7 | 35 | -x |
| `carving_knife__awvoox` | `x6` | broken | 1.0 | 5.0 | 18 | 0.4 | 22 | -x |
| `hook__akwlam` | `z2` | broken | 0.1 | 0.6 | 2 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `x2` | broken | 1.6 | 4.9 | 5 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `x4` | broken | 0.7 | 4.4 | 5 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `x3` | broken | 1.1 | 4.9 | 5 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `x5` | broken | 0.5 | 3.4 | 5 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `x6` | broken | 0.4 | 2.4 | 4 | 0.0 | 0 |  |
| `toothbrush__nbgpwg` | `xinf` | broken | 1.5 | 4.9 | 5 | 0.0 | 0 |  |
| `briefcase__lwbctq` | `x2` | broken | 0.2 | 0.5 | 3 | 4.9 | 46 | -x |
| `briefcase__psvdau` | `z2` | broken | 0.6 | 0.5 | 28 | 0.5 | 25 | -y |
| `cookie_cutter__lqrfzo` | `z2` | borderline | 4.1 | 25.3 | 74 | 2.7 | 45 | +x |
| `hammer__fcrlbo` | `x2` | broken | 2.0 | 7.8 | 44 | 0.7 | 30 | -x |
| `bottle_of_wine__oldpru` | `z6` | broken | 5.9 | 29.0 | 29 | 0.0 | 0 |  |
| `box_of_cookies__nirfva` | `y2` | broken | 0.2 | 0.7 | 2 | 0.0 | 0 |  |
| `notebook__zkmmnf` | `z2` | borderline | 1.4 | 8.2 | 35 | 0.3 | 31 | +y |
| `soap_dispenser__zwudcz` | `z2` | borderline | 2.1 | 11.9 | 22 | 1.2 | 27 | +z |
| `soap_dispenser__cgxrwd` | `z2` | broken | 0.2 | 0.8 | 3 | 0.0 | 0 |  |
| `soap_dispenser__cgxrwd` | `z4` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `soap_dispenser__cgxrwd` | `z3` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `soap_dispenser__cgxrwd` | `z5` | broken | 0.1 | 0.7 | 3 | 0.0 | 0 |  |
| `soap_dispenser__cgxrwd` | `z6` | broken | 0.1 | 0.6 | 2 | 0.0 | 0 |  |
| `soap_dispenser__cgxrwd` | `zinf` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `soap_dispenser__sinntg` | `x2` | broken | 5.6 | 42.4 | 117 | 32.5 | 85 | -x |
| `soap_dispenser__sinntg` | `z2` | broken | 5.0 | 29.0 | 119 | 17.3 | 75 | +z |
| `soap_dispenser__ijxawq` | `z2` | broken | 1.7 | 5.4 | 44 | 10.3 | 61 | +z |
| `soap_dispenser__ijxawq` | `z4` | broken | 1.4 | 4.1 | 32 | 4.8 | 100 | +z |
| `soap_dispenser__ijxawq` | `z3` | broken | 1.5 | 4.8 | 37 | 4.7 | 101 | +z |
| `soap_dispenser__ijxawq` | `z5` | broken | 1.2 | 3.1 | 26 | 4.7 | 102 | +z |
| `soap_dispenser__ijxawq` | `z6` | broken | 1.2 | 3.2 | 23 | 4.8 | 100 | +z |
| `soap_dispenser__ijxawq` | `zinf` | broken | 2.1 | 5.6 | 50 | 11.5 | 66 | +z |
| `soap_dispenser__jmulpc` | `z2` | broken | 1.3 | 1.6 | 54 | 41.1 | 69 | +z |
| `bottle_of_milk__utfsmp` | `x2` | broken | 4.0 | 21.5 | 56 | 12.8 | 59 | +x |
| `bottle_of_milk__utfsmp` | `x4` | broken | 3.9 | 21.4 | 54 | 9.4 | 54 | +x |
| `bottle_of_milk__utfsmp` | `x5` | broken | 4.0 | 22.8 | 53 | 6.8 | 58 | +x |
| `cellulose_tape__gchdhk` | `z6` | broken | 5.0 | 31.8 | 59 | 7.9 | 48 | +x |
| `conditioner_dispenser__njqnuo` | `x2` | broken | 0.3 | 1.0 | 2 | 0.0 | 0 |  |
| `conditioner_dispenser__njqnuo` | `x4` | broken | 0.4 | 1.0 | 2 | 0.0 | 0 |  |
| `conditioner_dispenser__njqnuo` | `x3` | broken | 0.4 | 1.0 | 2 | 0.0 | 0 |  |
| `conditioner_dispenser__njqnuo` | `x5` | broken | 0.5 | 1.0 | 2 | 0.0 | 0 |  |
| `conditioner_dispenser__njqnuo` | `x6` | broken | 0.4 | 1.0 | 2 | 0.0 | 0 |  |
| `conditioner_dispenser__njqnuo` | `xinf` | broken | 0.3 | 1.0 | 2 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `x2` | broken | 0.4 | 2.1 | 6 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `z2` | broken | 0.3 | 1.5 | 5 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `z4` | broken | 0.3 | 1.4 | 5 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `z3` | broken | 0.3 | 1.4 | 5 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `z5` | broken | 0.3 | 1.3 | 5 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `z6` | broken | 0.3 | 1.4 | 6 | 0.0 | 0 |  |
| `conditioner_dispenser__udeflt` | `zinf` | broken | 0.3 | 1.5 | 6 | 0.0 | 0 |  |
| `pack_of_cigarettes__qabdgg` | `z2` | borderline | 1.8 | 4.6 | 23 | 3.1 | 38 | +x |
| `pack_of_cigarettes__rmrubd` | `z2` | broken | 4.6 | 20.6 | 99 | 59.3 | 92 | +x |
| `pencil_holder__jkuiqg` | `x2` | broken | 0.1 | 0.3 | 1 | 0.0 | 0 |  |
| `pencil_holder__jkuiqg` | `y2` | broken | 0.1 | 0.3 | 1 | 0.0 | 0 |  |
| `jar_of_bath_salt__dtrjxi` | `z2` | broken | 1.1 | 2.8 | 35 | 9.9 | 26 | +z |
| `jar_of_bath_salt__dtrjxi` | `z5` | broken | 0.6 | 2.2 | 18 | 0.0 | 0 |  |
| `bottle_of_wine__dijgvu` | `z2` | broken | 5.1 | 33.5 | 65 | 17.0 | 60 | -z |
| `teaspoon__mgdfcz` | `x2` | broken | 0.3 | 1.7 | 3 | 0.0 | 0 |  |
| `teaspoon__mgdfcz` | `z2` | broken | 0.4 | 2.1 | 3 | 0.0 | 0 |  |
| `notebook__ffdjzp` | `x2` | broken | 2.4 | 16.9 | 18 | 5.5 | 73 | -x |
| `notebook__ffdjzp` | `y2` | broken | 1.7 | 13.0 | 18 | 7.1 | 85 | -x |
| `notebook__ffdjzp` | `z2` | broken | 2.9 | 17.0 | 17 | 4.6 | 81 | -x |
| `notebook__gqkaqu` | `x2` | borderline | 1.7 | 12.2 | 35 | 0.3 | 40 | -x |
| `notebook__gqkaqu` | `y2` | broken | 2.1 | 13.2 | 52 | 23.8 | 42 | +y |
| `notebook__gqkaqu` | `z2` | broken | 3.2 | 20.6 | 74 | 24.5 | 52 | +y |
| `bottle_of_pop__wuuoes` | `z4` | borderline | 0.6 | 2.6 | 9 | 2.3 | 44 | +z |
| `bottle_of_pop__wuuoes` | `z3` | borderline | 1.4 | 7.0 | 24 | 3.9 | 42 | +z |
| `bottle_of_pop__wuuoes` | `z5` | broken | 1.4 | 7.4 | 25 | 4.8 | 41 | +z |
| `bottle_of_pop__wuuoes` | `z6` | broken | 1.4 | 7.0 | 24 | 4.6 | 43 | +z |
| `bottle_of_pop__wuuoes` | `zinf` | broken | 1.1 | 5.1 | 18 | 4.4 | 42 | +z |
| `bottle_of_pop__zjbqvb` | `z4` | borderline | 0.4 | 2.4 | 5 | 2.0 | 43 | +z |
| `bottle_of_pop__zjbqvb` | `z3` | broken | 1.2 | 4.9 | 26 | 3.9 | 47 | -z |
| `bottle_of_pop__zjbqvb` | `z5` | broken | 1.2 | 5.1 | 27 | 4.2 | 50 | -z |
| `bottle_of_pop__zjbqvb` | `z6` | broken | 1.1 | 5.0 | 25 | 4.0 | 48 | -z |
| `bottle_of_pop__zjbqvb` | `zinf` | broken | 1.2 | 5.2 | 28 | 4.6 | 48 | -z |
| `jar_of_bath_salt__erglwp` | `z5` | broken | 1.0 | 3.6 | 4 | 0.0 | 0 |  |
| `bottle_of_champagne__npsevx` | `z6` | broken | 5.3 | 35.3 | 114 | 38.5 | 110 | -z |
| `bottle_of_whiskey__aipxlu` | `x2` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `bottle_of_whiskey__aipxlu` | `y2` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `bottle_of_whiskey__aipxlu` | `z5` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `bottle_of_whiskey__hejqyo` | `x2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_whiskey__hejqyo` | `y2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `bottle_of_whiskey__hejqyo` | `z5` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `case_of_eyeshadow__zgervc` | `x2` | borderline | 0.1 | 0.7 | 2 | 0.0 | 0 |  |
| `hardback__acaxtl` | `x2` | broken | 0.8 | 6.0 | 13 | 0.0 | 0 |  |
| `hardback__acaxtl` | `z2` | broken | 0.9 | 6.4 | 15 | 0.0 | 0 |  |
| `hardback__aceozs` | `y2` | broken | 2.2 | 13.3 | 48 | 34.4 | 45 | -y |
| `can_of_corn__alphlq` | `z2` | borderline | 2.3 | 8.7 | 17 | 1.3 | 31 | +z |
| `can_of_corn__kwwlfn` | `z2` | borderline | 3.7 | 16.9 | 26 | 0.4 | 26 | +z |
| `water_glass__cdteyb` | `z2` | broken | 0.5 | 2.6 | 6 | 0.0 | 0 |  |
| `water_glass__cdteyb` | `z4` | broken | 0.5 | 2.5 | 6 | 0.0 | 0 |  |
| `water_glass__cdteyb` | `z3` | broken | 0.6 | 3.0 | 7 | 0.0 | 0 |  |
| `water_glass__cdteyb` | `z5` | broken | 0.5 | 2.6 | 6 | 0.0 | 0 |  |
| `water_glass__cdteyb` | `z6` | broken | 0.6 | 2.8 | 6 | 0.0 | 0 |  |
| `water_glass__cdteyb` | `zinf` | broken | 0.5 | 2.7 | 6 | 0.0 | 0 |  |
| `water_glass__elwfms` | `x2` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `water_glass__elwfms` | `y2` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `bottle_of_ginger_beer__zkocwb` | `z2` | broken | 4.5 | 31.7 | 51 | 5.4 | 37 | +x |
| `lemonade_bottle__yprkek` | `z4` | broken | 0.2 | 1.0 | 2 | 0.0 | 0 |  |
| `lemonade_bottle__yprkek` | `z5` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `soda_water_bottle__upfssc` | `z4` | broken | 0.2 | 1.0 | 2 | 0.0 | 0 |  |
| `soda_water_bottle__upfssc` | `z5` | broken | 0.0 | 0.2 | 1 | 0.0 | 0 |  |
| `marker__ablbdc` | `x4` | borderline | 0.5 | 2.5 | 6 | 0.0 | 0 |  |
| `marker__ablbdc` | `x3` | broken | 0.6 | 2.7 | 8 | 0.0 | 0 |  |
| `marker__ablbdc` | `xinf` | broken | 0.5 | 2.6 | 7 | 0.0 | 0 |  |
| `marker__bwwkmb` | `x2` | broken | 1.2 | 5.3 | 21 | 1.2 | 36 | +x |
| `marker__bwwkmb` | `x4` | broken | 1.0 | 4.6 | 15 | 0.8 | 28 | -x |
| `marker__bwwkmb` | `x3` | broken | 1.2 | 5.2 | 19 | 0.9 | 32 | +x |
| `marker__bwwkmb` | `x5` | broken | 0.8 | 4.2 | 12 | 0.8 | 27 | -x |
| `marker__bwwkmb` | `x6` | borderline | 0.8 | 4.0 | 11 | 0.7 | 26 | -x |
| `marker__bwwkmb` | `xinf` | broken | 1.3 | 5.4 | 21 | 1.0 | 35 | +x |
| `table_knife__eebpap` | `x2` | broken | 0.4 | 2.1 | 4 | 0.0 | 0 |  |
| `table_knife__eebpap` | `y2` | broken | 0.4 | 2.3 | 4 | 0.0 | 0 |  |
| `table_knife__eebpap` | `z2` | broken | 0.3 | 1.5 | 4 | 0.0 | 0 |  |
| `table_knife__eebpap` | `x3` | broken | 0.5 | 2.6 | 5 | 0.0 | 0 |  |
| `table_knife__eebpap` | `x5` | broken | 0.4 | 2.4 | 5 | 0.0 | 0 |  |
| `table_knife__eebpap` | `x6` | broken | 0.4 | 2.2 | 4 | 0.0 | 0 |  |
| `table_knife__eebpap` | `xinf` | broken | 0.5 | 2.4 | 5 | 0.0 | 0 |  |
| `table_knife__jxdfyy` | `x2` | broken | 0.4 | 2.1 | 5 | 0.0 | 0 |  |
| `table_knife__jxdfyy` | `y2` | broken | 0.5 | 3.1 | 6 | 0.0 | 0 |  |
| `table_knife__jxdfyy` | `z2` | broken | 0.5 | 2.9 | 6 | 0.0 | 0 |  |
| `pen__gzrbrx` | `x2` | borderline | 0.9 | 4.4 | 15 | 1.0 | 29 | -x |
| `pen__gzrbrx` | `x4` | borderline | 0.9 | 4.1 | 15 | 2.0 | 32 | -x |
| `pen__gzrbrx` | `x3` | borderline | 1.0 | 4.5 | 18 | 2.0 | 36 | -x |
| `pen__gzrbrx` | `x5` | borderline | 0.8 | 3.6 | 13 | 1.4 | 29 | -x |
| `pen__gzrbrx` | `x6` | borderline | 0.7 | 3.1 | 11 | 0.5 | 38 | +y |
| `pen__gzrbrx` | `xinf` | borderline | 0.9 | 4.5 | 15 | 1.3 | 28 | -x |
| `table_knife__lrdmpf` | `x2` | broken | 0.2 | 1.0 | 2 | 0.0 | 0 |  |
| `table_knife__lrdmpf` | `y2` | broken | 0.6 | 2.0 | 13 | 0.0 | 0 |  |
| `table_knife__lrdmpf` | `z2` | broken | 0.5 | 1.7 | 12 | 0.0 | 0 |  |
| `table_knife__lrdmpf` | `x3` | broken | 0.3 | 1.4 | 3 | 0.0 | 0 |  |
| `table_knife__lrdmpf` | `x6` | broken | 0.3 | 1.3 | 4 | 0.0 | 0 |  |
| `clipper__befwbq` | `x2` | broken | 3.3 | 9.5 | 36 | 0.0 | 0 |  |
| `clipper__befwbq` | `y2` | broken | 3.8 | 12.4 | 33 | 0.0 | 0 |  |
| `clipper__befwbq` | `z2` | broken | 3.5 | 11.4 | 27 | 0.0 | 0 |  |
| `clipper__befwbq` | `x4` | broken | 4.6 | 12.7 | 32 | 0.0 | 0 |  |
| `clipper__befwbq` | `x3` | broken | 4.6 | 12.5 | 35 | 0.0 | 0 |  |
| `lipstick__albshz` | `x2` | broken | 5.1 | 31.3 | 77 | 10.3 | 29 | +z |
| `lipstick__albshz` | `y2` | broken | 4.9 | 30.7 | 70 | 10.3 | 29 | +z |
| `lipstick__albshz` | `y4` | broken | 5.4 | 31.1 | 75 | 10.3 | 29 | +z |
| `lipstick__albshz` | `y3` | broken | 5.3 | 30.2 | 73 | 10.3 | 29 | +z |
| `lock__pngvzn` | `x2` | broken | 1.0 | 4.0 | 25 | 0.0 | 0 |  |
| `soap_dish__gnndfy` | `z4` | broken | 6.0 | 28.6 | 29 | 0.0 | 0 |  |
| `soap_dish__gnndfy` | `z3` | broken | 4.8 | 28.2 | 29 | 0.0 | 0 |  |
| `soap_dish__gnndfy` | `z5` | broken | 4.8 | 27.8 | 29 | 0.0 | 0 |  |
| `soap_dish__gnndfy` | `z6` | broken | 4.6 | 27.5 | 29 | 0.0 | 0 |  |
| `soap_dish__gnndfy` | `zinf` | broken | 5.7 | 28.5 | 29 | 0.0 | 0 |  |
| `folder__lktggf` | `y2` | broken | 4.1 | 35.3 | 86 | 109.9 | 101 | -y |
| `notepad__rvevdj` | `x2` | broken | 2.4 | 8.6 | 67 | 40.6 | 56 | +y |
| `bottle_of_dish_soap__bnmixt` | `z2` | broken | 3.7 | 29.6 | 58 | 49.9 | 64 | +x |
| `bottle_of_lotion__tkryrh` | `z2` | broken | 3.3 | 18.7 | 28 | 4.1 | 46 | +z |
| `bottle_of_lotion__tkryrh` | `z4` | broken | 2.9 | 15.7 | 28 | 6.8 | 62 | +z |
| `bottle_of_lotion__tkryrh` | `z3` | broken | 4.3 | 22.0 | 43 | 7.4 | 59 | +z |
| `bottle_of_lotion__tkryrh` | `z5` | broken | 4.0 | 18.8 | 45 | 5.9 | 58 | +z |
| `bottle_of_lotion__tkryrh` | `z6` | broken | 3.7 | 16.7 | 43 | 9.8 | 58 | +z |
| `bottle_of_lotion__tkryrh` | `zinf` | broken | 3.1 | 14.2 | 32 | 9.6 | 56 | +z |
| `tube_of_lotion__ixfuxm` | `z3` | broken | 0.2 | 0.9 | 1 | 0.0 | 0 |  |
| `tube_of_lotion__ixfuxm` | `z5` | broken | 0.2 | 0.9 | 1 | 0.0 | 0 |  |
| `tube_of_lotion__ixfuxm` | `z6` | broken | 0.2 | 0.9 | 1 | 0.0 | 0 |  |
| `tube_of_lotion__ixfuxm` | `zinf` | broken | 0.2 | 0.9 | 1 | 0.0 | 0 |  |
| `bottle_of_face_cream__dztaed` | `z2` | broken | 2.6 | 17.5 | 32 | 4.3 | 26 | +y |
| `bottle_of_face_cream__dztaed` | `z4` | broken | 2.2 | 15.5 | 31 | 7.7 | 28 | +z |
| `bottle_of_face_cream__dztaed` | `z3` | broken | 2.5 | 16.8 | 33 | 9.8 | 29 | +z |
| `bottle_of_face_cream__dztaed` | `z5` | broken | 2.1 | 15.0 | 31 | 5.2 | 31 | +z |
| `bottle_of_face_cream__dztaed` | `z6` | broken | 1.9 | 13.3 | 30 | 11.8 | 29 | +z |
| `bottle_of_face_cream__dztaed` | `zinf` | broken | 2.5 | 17.1 | 33 | 13.1 | 30 | +z |
| `bottle_of_perfume__ipurzb` | `z3` | broken | 0.4 | 1.0 | 3 | 0.0 | 0 |  |
| `bottle_of_perfume__ipurzb` | `z5` | broken | 0.4 | 1.0 | 1 | 0.0 | 0 |  |
| `bottle_of_perfume__ipurzb` | `z6` | broken | 0.4 | 1.0 | 1 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `z2` | broken | 2.1 | 7.2 | 11 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `z4` | broken | 2.5 | 8.7 | 12 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `z3` | broken | 2.4 | 8.7 | 13 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `z5` | broken | 2.4 | 8.5 | 12 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `z6` | borderline | 2.3 | 8.3 | 12 | 0.0 | 0 |  |
| `bottle_of_ginger__drqhzo` | `zinf` | broken | 2.4 | 8.4 | 12 | 0.0 | 0 |  |
| `bottle_of_ground_cloves__vzamzb` | `z3` | borderline | 2.3 | 7.5 | 11 | 0.0 | 0 |  |
| `bottle_of_ground_cloves__vzamzb` | `zinf` | borderline | 2.4 | 7.8 | 11 | 0.0 | 0 |  |
| `bottle_of_ground_mace__lgpxro` | `z2` | broken | 1.9 | 6.3 | 10 | 0.1 | 25 | -z |
| `bottle_of_ground_mace__lgpxro` | `z4` | broken | 1.9 | 6.3 | 9 | 0.0 | 0 |  |
| `bottle_of_ground_mace__lgpxro` | `z3` | broken | 2.0 | 6.7 | 10 | 0.0 | 0 |  |
| `bottle_of_ground_mace__lgpxro` | `z5` | broken | 1.9 | 6.4 | 10 | 0.0 | 0 |  |
| `bottle_of_ground_mace__lgpxro` | `z6` | broken | 2.0 | 6.7 | 10 | 1.3 | 24 | -z |
| `bottle_of_ground_mace__lgpxro` | `zinf` | broken | 2.1 | 6.8 | 10 | 1.7 | 25 | -z |
| `bottle_of_ground_nutmeg__qebruq` | `z2` | broken | 2.0 | 7.0 | 11 | 0.0 | 0 |  |
| `bottle_of_ground_nutmeg__qebruq` | `z4` | broken | 2.4 | 8.6 | 12 | 0.0 | 0 |  |
| `bottle_of_ground_nutmeg__qebruq` | `z3` | broken | 2.4 | 8.4 | 12 | 0.0 | 0 |  |
| `bottle_of_ground_nutmeg__qebruq` | `z5` | borderline | 2.3 | 8.3 | 12 | 0.0 | 0 |  |
| `bottle_of_ground_nutmeg__qebruq` | `z6` | borderline | 2.2 | 8.1 | 12 | 0.0 | 0 |  |
| `bottle_of_ground_nutmeg__qebruq` | `zinf` | broken | 2.4 | 8.3 | 12 | 0.0 | 0 |  |
| `bottle_of_paprika__nrdgrp` | `z3` | borderline | 2.3 | 7.6 | 11 | 0.0 | 0 |  |
| `bottle_of_paprika__nrdgrp` | `zinf` | borderline | 2.4 | 7.8 | 11 | 0.0 | 0 |  |
| `bottle_of_seasoning__vqwqqv` | `z2` | broken | 2.0 | 7.0 | 11 | 0.0 | 0 |  |
| `bottle_of_seasoning__vqwqqv` | `z4` | broken | 2.4 | 8.6 | 12 | 0.0 | 0 |  |
| `bottle_of_seasoning__vqwqqv` | `z3` | broken | 2.4 | 8.4 | 12 | 0.0 | 0 |  |
| `bottle_of_seasoning__vqwqqv` | `z5` | borderline | 2.3 | 8.3 | 12 | 0.0 | 0 |  |
| `bottle_of_seasoning__vqwqqv` | `z6` | borderline | 2.2 | 8.1 | 12 | 0.0 | 0 |  |
| `bottle_of_seasoning__vqwqqv` | `zinf` | broken | 2.4 | 8.3 | 12 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `x2` | broken | 3.6 | 8.2 | 10 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `y2` | broken | 3.7 | 8.3 | 11 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `z3` | broken | 4.1 | 9.6 | 12 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `z5` | borderline | 3.7 | 8.9 | 11 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `z6` | broken | 4.0 | 9.8 | 12 | 0.0 | 0 |  |
| `bottle_of_pumpkin_pie_spice__oindrv` | `zinf` | broken | 4.0 | 9.8 | 13 | 0.0 | 0 |  |
| `jar_of_curry_powder__ozhftu` | `z2` | borderline | 1.9 | 6.4 | 10 | 0.0 | 0 |  |
| `jar_of_curry_powder__ozhftu` | `z4` | borderline | 2.1 | 6.6 | 11 | 3.7 | 26 | -z |
| `jar_of_curry_powder__ozhftu` | `z3` | broken | 2.1 | 7.0 | 11 | 2.9 | 26 | -z |
| `jar_of_curry_powder__ozhftu` | `z5` | borderline | 2.0 | 6.6 | 11 | 2.7 | 25 | -z |
| `jar_of_curry_powder__ozhftu` | `z6` | borderline | 2.0 | 6.9 | 11 | 1.5 | 24 | -z |
| `jar_of_curry_powder__ozhftu` | `zinf` | broken | 2.1 | 6.8 | 11 | 3.8 | 26 | -z |
| `bottle_of_coconut_water__lsixio` | `z2` | borderline | 2.9 | 14.4 | 25 | 3.8 | 37 | -x |
| `bottle_of_coconut_water__lsixio` | `z4` | broken | 6.1 | 38.6 | 78 | 18.0 | 62 | +x |
| `bottle_of_milkshake__naxqya` | `z2` | broken | 0.8 | 1.0 | 27 | 3.8 | 50 | +z |
| `bottle_of_milkshake__naxqya` | `z4` | broken | 0.8 | 1.2 | 26 | 3.9 | 59 | -y |
| `bottle_of_milkshake__naxqya` | `z3` | broken | 0.7 | 0.8 | 27 | 3.3 | 49 | +z |
| `bottle_of_milkshake__naxqya` | `z5` | broken | 0.7 | 1.0 | 25 | 6.1 | 46 | +z |
| `bottle_of_milkshake__naxqya` | `z6` | broken | 0.8 | 1.3 | 26 | 5.6 | 49 | +z |
| `bottle_of_milkshake__naxqya` | `zinf` | broken | 0.7 | 1.0 | 20 | 11.4 | 48 | +z |
| `egg__qopwgx` | `z2` | borderline | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `egg__qopwgx` | `x4` | borderline | 0.1 | 0.4 | 1 | 0.0 | 0 |  |
| `egg__qopwgx` | `x3` | broken | 0.1 | 0.4 | 1 | 0.0 | 0 |  |
| `egg__qopwgx` | `xinf` | broken | 0.1 | 0.4 | 1 | 0.0 | 0 |  |
| `bag_of_flour__fntqmd` | `z2` | broken | 2.0 | 6.7 | 36 | 11.7 | 33 | -z |
| `jar_of_sugar__bholak` | `z2` | broken | 0.9 | 1.0 | 22 | 4.4 | 42 | +x |
| `jug_of_milk__xbvdpc` | `z2` | broken | 0.3 | 1.0 | 2 | 0.0 | 0 |  |
| `jug_of_milk__xbvdpc` | `z4` | broken | 0.4 | 1.0 | 2 | 0.0 | 0 |  |
| `hotdog_frank__qijvmi` | `y2` | broken | 1.7 | 9.6 | 13 | 0.0 | 0 |  |
| `carrot__qhmmmx` | `x2` | borderline | 3.9 | 8.7 | 13 | 0.6 | 33 | -x |
| `carrot__qhmmmx` | `x4` | borderline | 3.4 | 8.2 | 13 | 1.4 | 41 | -x |
| `carrot__qhmmmx` | `x3` | borderline | 4.3 | 8.9 | 14 | 0.9 | 36 | -x |
| `carrot__qhmmmx` | `x5` | borderline | 3.9 | 8.2 | 13 | 1.3 | 36 | -x |
| `carrot__qhmmmx` | `x6` | broken | 4.4 | 8.4 | 12 | 1.1 | 35 | -x |
| `carrot__qhmmmx` | `xinf` | broken | 2.7 | 5.3 | 7 | 0.0 | 0 |  |
| `box_of_rum__lnjbjj` | `x2` | broken | 2.4 | 12.9 | 46 | 58.1 | 43 | +x |
| `box_of_rum__lnjbjj` | `y2` | broken | 2.4 | 12.9 | 46 | 57.9 | 42 | -x |
| `box_of_rum__lnjbjj` | `z4` | broken | 3.7 | 26.6 | 48 | 43.7 | 40 | +x |
| `bottle_of_disinfectant__faedff` | `z2` | broken | 3.3 | 9.6 | 13 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `x2` | broken | 0.5 | 3.0 | 11 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `x4` | broken | 0.4 | 2.6 | 11 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `x3` | broken | 0.4 | 2.6 | 10 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `x5` | broken | 0.4 | 2.5 | 10 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `x6` | broken | 0.4 | 2.2 | 9 | 0.0 | 0 |  |
| `flashlight__jtrbkc` | `xinf` | broken | 0.5 | 3.0 | 11 | 0.0 | 0 |  |
| `scone__gwdhyj` | `z2` | broken | 0.1 | 0.8 | 3 | 0.0 | 0 |  |
| `bottle_of_conditioner__teafxb` | `x2` | broken | 4.1 | 34.3 | 72 | 19.8 | 40 | +x |
| `paper_bag__bzsxgw` | `x2` | broken | 5.8 | 15.4 | 22 | 0.0 | 0 |  |
| `paper_bag__bzsxgw` | `y2` | broken | 5.8 | 15.2 | 22 | 0.0 | 0 |  |
| `paper_bag__bzsxgw` | `z2` | broken | 2.5 | 8.1 | 14 | 0.0 | 0 |  |
| `toy_box__kvithq` | `z4` | broken | 2.5 | 12.2 | 41 | 61.0 | 37 | +z |
| `vase__atgnsc` | `x2` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `vase__atgnsc` | `y2` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `vase__atgnsc` | `z2` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vase__atgnsc` | `z4` | broken | 0.1 | 0.1 | 1 | 0.0 | 0 |  |
| `vase__atgnsc` | `z3` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vase__atgnsc` | `z5` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vase__atgnsc` | `z6` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vase__atgnsc` | `zinf` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `tissue_dispenser__frdbld` | `z2` | broken | 0.2 | 0.7 | 4 | 0.0 | 0 |  |
| `tissue_dispenser__nflhrf` | `x2` | broken | 1.4 | 2.5 | 51 | 60.1 | 43 | +x |
| `can_of_bay_leaves__ppwvjf` | `z2` | broken | 4.9 | 31.8 | 52 | 6.4 | 47 | -z |
| `can_of_bay_leaves__ppwvjf` | `z4` | broken | 4.8 | 27.2 | 47 | 3.9 | 43 | -z |
| `can_of_bay_leaves__ppwvjf` | `z3` | broken | 4.6 | 26.0 | 50 | 6.4 | 42 | -z |
| `can_of_bay_leaves__ppwvjf` | `z5` | broken | 4.9 | 27.5 | 49 | 4.8 | 46 | -z |
| `can_of_bay_leaves__ppwvjf` | `z6` | broken | 4.9 | 26.4 | 49 | 5.4 | 46 | -z |
| `cap__iqeyba` | `zinf` | borderline | 4.4 | 11.5 | 31 | 0.6 | 32 | -z |
| `dip_candle__poexjo` | `x2` | borderline | 0.3 | 0.6 | 2 | 1.3 | 41 | +z |
| `dip_candle__poexjo` | `y2` | broken | 0.3 | 0.8 | 3 | 1.3 | 41 | +z |
| `dip_candle__poexjo` | `z2` | broken | 0.2 | 0.7 | 2 | 1.2 | 34 | +z |
| `dip_candle__poexjo` | `z4` | borderline | 0.3 | 0.8 | 3 | 1.5 | 39 | +z |
| `dip_candle__poexjo` | `z3` | broken | 0.3 | 0.8 | 2 | 1.3 | 35 | +z |
| `dip_candle__poexjo` | `z5` | borderline | 0.3 | 0.8 | 2 | 1.5 | 39 | +z |
| `dip_candle__poexjo` | `z6` | borderline | 0.3 | 0.7 | 2 | 1.5 | 36 | +z |
| `dip_candle__poexjo` | `zinf` | broken | 0.2 | 0.8 | 2 | 1.2 | 34 | +z |
| `dip_candle__ghctun` | `x2` | borderline | 0.3 | 0.4 | 2 | 1.0 | 41 | -z |
| `dip_candle__ghctun` | `y2` | borderline | 0.3 | 0.5 | 2 | 0.9 | 40 | -z |
| `dip_candle__mxjejs` | `x2` | borderline | 0.2 | 0.7 | 3 | 1.3 | 42 | -z |
| `dip_candle__mxjejs` | `y2` | borderline | 0.2 | 0.6 | 3 | 1.3 | 44 | -z |
| `light_bulb__ymomhw` | `x4` | broken | 1.1 | 6.4 | 27 | 0.1 | 14 | -y |
| `light_bulb__ymomhw` | `x5` | broken | 1.0 | 5.3 | 26 | 0.1 | 13 | -y |
| `light_bulb__ymomhw` | `x6` | broken | 0.9 | 5.3 | 24 | 0.1 | 13 | -y |
| `blackberry__tvpfvb` | `y2` | borderline | 4.8 | 14.2 | 115 | 3.2 | 57 | +x |
| `blackberry__tvpfvb` | `z2` | borderline | 4.7 | 13.4 | 115 | 3.0 | 60 | +x |
| `blackberry__tvpfvb` | `x4` | borderline | 1.6 | 6.5 | 20 | 0.9 | 25 | +x |
| `blackberry__tvpfvb` | `y4` | borderline | 5.0 | 14.2 | 117 | 3.4 | 57 | -z |
| `blackberry__tvpfvb` | `z4` | borderline | 4.8 | 11.9 | 114 | 3.1 | 56 | +y |
| `blackberry__tvpfvb` | `x3` | borderline | 1.7 | 6.9 | 20 | 0.7 | 27 | +x |
| `blackberry__tvpfvb` | `y3` | borderline | 4.4 | 9.7 | 109 | 3.0 | 61 | +x |
| `blackberry__tvpfvb` | `z3` | borderline | 4.9 | 12.8 | 115 | 3.2 | 58 | +y |
| `blackberry__tvpfvb` | `x5` | borderline | 1.4 | 5.8 | 17 | 0.8 | 25 | +x |
| `blackberry__tvpfvb` | `y5` | borderline | 5.0 | 14.4 | 117 | 3.3 | 57 | -z |
| `blackberry__tvpfvb` | `z5` | broken | 5.1 | 19.1 | 116 | 4.8 | 54 | +x |
| `blackberry__tvpfvb` | `x6` | borderline | 1.5 | 6.3 | 16 | 0.5 | 22 | +x |
| `blackberry__tvpfvb` | `y6` | borderline | 4.9 | 12.9 | 115 | 3.0 | 60 | +x |
| `blackberry__tvpfvb` | `z6` | broken | 5.1 | 19.7 | 116 | 5.0 | 61 | +x |
| `blackberry__tvpfvb` | `xinf` | borderline | 1.6 | 6.3 | 22 | 0.9 | 25 | +x |
| `bookcase_baseboard__hlhneo` | `y3` | broken | 5.3 | 13.1 | 17 | 0.0 | 0 |  |
| `bookcase_baseboard__hlhneo` | `y6` | broken | 4.4 | 11.6 | 16 | 0.0 | 0 |  |
| `table_lamp__ehjsdz` | `z2` | broken | 0.3 | 1.6 | 6 | 0.0 | 0 |  |
| `table_lamp__ehjsdz` | `z4` | broken | 2.1 | 3.4 | 69 | 208.3 | 61 | -z |
| `table_lamp__ehjsdz` | `z3` | broken | 2.2 | 4.4 | 69 | 275.3 | 62 | -z |
| `table_lamp__ehjsdz` | `z5` | broken | 1.8 | 2.9 | 69 | 303.6 | 60 | -z |
| `table_lamp__ehjsdz` | `z6` | broken | 1.5 | 2.3 | 69 | 204.8 | 60 | -z |
| `table_lamp__ehjsdz` | `zinf` | broken | 2.2 | 4.2 | 69 | 382.2 | 61 | -z |
| `teapot__mvrhya` | `z2` | broken | 0.1 | 0.3 | 3 | 0.0 | 0 |  |
| `teapot__mvrhya` | `z4` | broken | 0.1 | 0.2 | 3 | 0.0 | 0 |  |
| `teapot__mvrhya` | `z3` | broken | 0.1 | 0.2 | 3 | 0.0 | 0 |  |
| `teapot__mvrhya` | `z5` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `teapot__mvrhya` | `z6` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `teapot__mvrhya` | `zinf` | broken | 0.1 | 0.3 | 3 | 0.0 | 0 |  |
| `tomato_paste_can__krarex` | `z4` | broken | 0.2 | 1.0 | 2 | 0.0 | 0 |  |
| `tomato_paste_can__krarex` | `z5` | broken | 0.0 | 0.3 | 1 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `z2` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `z4` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `z3` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `z5` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `z6` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `vinegar_bottle__hbsbwt` | `zinf` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `paperback_book__bisnup` | `x2` | broken | 3.2 | 18.5 | 58 | 28.2 | 48 | -y |
| `paperback_book__ixwtnb` | `x2` | broken | 5.8 | 32.9 | 100 | 47.9 | 55 | -z |
| `paperback_book__wfcxjv` | `x2` | broken | 4.3 | 16.2 | 46 | 12.8 | 38 | -x |
| `frying_pan__jpzusm` | `z5` | broken | 0.4 | 0.3 | 12 | 0.0 | 0 |  |
| `frying_pan__jpzusm` | `z6` | broken | 0.3 | 0.2 | 8 | 0.0 | 0 |  |
| `frying_pan__mhndon` | `z5` | broken | 0.3 | 0.0 | 12 | 0.5 | 20 | -x |
| `frying_pan__mhndon` | `z6` | broken | 0.3 | 0.0 | 7 | 0.5 | 30 | -x |
| `keyboard__uwacaq` | `x2` | broken | 3.1 | 26.8 | 27 | 0.0 | 0 |  |
| `keyboard__uwacaq` | `y2` | broken | 3.1 | 26.8 | 27 | 0.0 | 0 |  |
| `carton_of_eggs__stxfxb` | `x2` | broken | 3.9 | 9.0 | 93 | 85.4 | 122 | -x |
| `notepad__awcyha` | `x2` | broken | 3.2 | 19.7 | 56 | 85.3 | 75 | +y |
| `notepad__hirdfd` | `x2` | broken | 1.4 | 5.8 | 41 | 42.3 | 45 | +y |
| `clipboard__gmxyfo` | `y2` | broken | 0.7 | 0.8 | 32 | 0.0 | 0 |  |
| `clipboard__gmxyfo` | `z3` | broken | 0.4 | 0.4 | 11 | 0.0 | 0 |  |
| `clipboard__gmxyfo` | `z6` | broken | 0.4 | 0.5 | 11 | 0.0 | 0 |  |
| `wall_clock__xevvel` | `x2` | broken | 5.6 | 14.4 | 20 | 0.0 | 0 |  |
| `wall_clock__xevvel` | `x4` | broken | 5.7 | 14.6 | 20 | 0.0 | 0 |  |
| `wall_clock__xevvel` | `x6` | broken | 5.4 | 14.0 | 19 | 0.0 | 0 |  |
| `signpost__vwhtlm` | `z2` | broken | 3.9 | 27.7 | 79 | 83.3 | 71 | +z |
| `decorative_sign__rypeha` | `x2` | broken | 0.5 | 2.6 | 9 | 0.0 | 0 |  |
| `decorative_sign__rypeha` | `y2` | broken | 0.4 | 1.9 | 8 | 0.0 | 0 |  |
| `decorative_sign__rypeha` | `z2` | broken | 0.6 | 3.0 | 11 | 0.0 | 0 |  |
| `towel_rack__nhlijv` | `z2` | broken | 0.8 | 1.4 | 4 | 33.0 | 97 | -z |
| `umbrella__ziqzqo` | `x2` | broken | 4.2 | 9.4 | 12 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `y2` | broken | 4.1 | 9.7 | 14 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `z2` | broken | 4.5 | 10.6 | 14 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `x4` | broken | 3.4 | 8.4 | 12 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `x3` | broken | 4.0 | 10.1 | 14 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `x5` | broken | 3.8 | 9.1 | 13 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `x6` | broken | 3.4 | 9.1 | 12 | 0.0 | 0 |  |
| `umbrella__ziqzqo` | `xinf` | broken | 3.0 | 8.1 | 12 | 0.0 | 0 |  |
| `rolling_pin__smnklu` | `yinf` | borderline | 3.5 | 11.7 | 18 | 0.7 | 33 | +x |
| `baseball_bat__spcxrt` | `x2` | broken | 3.5 | 8.8 | 13 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `y2` | broken | 3.7 | 9.6 | 13 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `z2` | broken | 3.7 | 9.0 | 13 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `x4` | broken | 3.7 | 9.5 | 13 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `x3` | broken | 3.6 | 9.1 | 12 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `x5` | broken | 3.6 | 9.0 | 12 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `x6` | broken | 3.6 | 9.0 | 13 | 0.0 | 0 |  |
| `baseball_bat__spcxrt` | `xinf` | broken | 3.5 | 9.3 | 13 | 0.0 | 0 |  |
| `shelf__kxvbui` | `y2` | broken | 2.6 | 6.6 | 9 | 0.0 | 0 |  |
| `shelf__kxvbui` | `z2` | broken | 2.7 | 6.9 | 10 | 0.0 | 0 |  |
| `radiator__arsjbe` | `y2` | broken | 1.6 | 3.9 | 20 | 37.6 | 108 | +y |
| `radiator__dwmloz` | `x2` | broken | 2.6 | 6.4 | 9 | 0.0 | 0 |  |
| `radiator__dwmloz` | `y2` | broken | 2.6 | 6.4 | 9 | 0.0 | 0 |  |
| `radiator__dwmloz` | `z2` | broken | 2.7 | 6.6 | 9 | 0.0 | 0 |  |
| `radiator__xnoiwp` | `y2` | broken | 1.8 | 5.2 | 22 | 40.1 | 113 | +y |
| `radiator__wntwxx` | `x2` | broken | 0.6 | 1.0 | 4 | 7.8 | 133 | +z |
| `radiator__wntwxx` | `y2` | broken | 0.6 | 0.8 | 3 | 7.1 | 102 | -y |
| `radiator__wntwxx` | `z2` | broken | 0.9 | 1.1 | 5 | 7.9 | 113 | +y |
| `drumstick__jauzlc` | `z2` | broken | 2.7 | 6.1 | 7 | 0.0 | 0 |  |
| `drumstick__jauzlc` | `x3` | borderline | 4.1 | 8.0 | 9 | 0.0 | 0 |  |
| `wall_mounted_tv__ggfznh` | `x2` | broken | 6.1 | 19.7 | 99 | 153.1 | 70 | +x |
| `wall_mounted_tv__dxyfxf` | `x2` | broken | 0.7 | 4.0 | 15 | 0.0 | 0 |  |
| `wall_mounted_tv__ylvjhb` | `x2` | broken | 1.3 | 4.5 | 9 | 0.0 | 0 |  |
| `wall_mounted_tv__sodbbr` | `x2` | broken | 0.7 | 3.9 | 15 | 0.0 | 0 |  |
| `umbrella__zxnjwz` | `x2` | broken | 2.1 | 5.0 | 54 | 12.1 | 96 | -x |
| `umbrella__zxnjwz` | `x4` | broken | 1.7 | 4.4 | 47 | 10.4 | 81 | +x |
| `umbrella__zxnjwz` | `x3` | broken | 1.8 | 4.8 | 50 | 10.8 | 90 | +x |
| `umbrella__zxnjwz` | `x5` | broken | 1.6 | 4.5 | 45 | 9.3 | 72 | +x |
| `umbrella__zxnjwz` | `x6` | broken | 1.3 | 4.0 | 32 | 8.6 | 65 | +x |
| `umbrella__zxnjwz` | `xinf` | broken | 2.0 | 4.7 | 52 | 13.4 | 95 | -x |
| `whiteboard__gvkgsi` | `x2` | broken | 3.1 | 12.9 | 99 | 134.7 | 43 | +x |
| `towel_rack__crivoa` | `x2` | broken | 1.1 | 4.7 | 32 | 3.8 | 37 | +y |
| `towel_rack__crivoa` | `y2` | broken | 1.4 | 9.2 | 32 | 3.6 | 37 | +y |
| `towel_rack__crivoa` | `z2` | broken | 1.4 | 9.0 | 31 | 5.3 | 42 | -y |
| `towel_rack__crivoa` | `y4` | broken | 1.2 | 7.4 | 21 | 6.4 | 37 | +y |
| `towel_rack__crivoa` | `y3` | broken | 1.3 | 8.2 | 27 | 2.2 | 40 | -y |
| `towel_rack__crivoa` | `y5` | broken | 1.3 | 7.4 | 22 | 7.8 | 41 | +y |
| `towel_rack__crivoa` | `y6` | broken | 1.3 | 8.3 | 23 | 7.9 | 41 | +y |
| `towel_rack__crivoa` | `yinf` | broken | 1.3 | 8.3 | 24 | 4.1 | 43 | +y |
| `decorative_sign__hgonwj` | `x2` | borderline | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `decorative_sign__hgonwj` | `y2` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `decorative_sign__hgonwj` | `z2` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `pack_of_kielbasa__dgodnn` | `x2` | broken | 2.3 | 7.6 | 14 | 6.2 | 45 | -x |
| `pack_of_kielbasa__fwqdoy` | `x2` | borderline | 0.5 | 1.3 | 4 | 3.9 | 43 | +x |
| `pack_of_kielbasa__bzlsuh` | `x2` | broken | 0.7 | 1.1 | 5 | 10.2 | 58 | -x |
| `pack_of_kielbasa__nhkuzn` | `x2` | broken | 0.3 | 0.7 | 2 | 6.4 | 60 | +x |
| `pen__hiyeki` | `x6` | borderline | 1.6 | 7.9 | 37 | 0.3 | 30 | -x |
| `pen__enfumt` | `x2` | borderline | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `pen__enfumt` | `y2` | broken | 0.2 | 1.0 | 2 | 0.0 | 0 |  |
| `pen__enfumt` | `z2` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `pen__enfumt` | `x4` | broken | 0.1 | 0.6 | 2 | 0.0 | 0 |  |
| `pen__enfumt` | `x3` | broken | 0.1 | 0.8 | 2 | 0.0 | 0 |  |
| `pen__enfumt` | `x6` | borderline | 0.1 | 0.6 | 2 | 0.0 | 0 |  |
| `pen__enfumt` | `xinf` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `cutting_board__tcdrzs` | `y2` | borderline | 6.5 | 17.3 | 24 | 0.0 | 0 |  |
| `cutting_board__aibvew` | `y2` | broken | 5.4 | 18.1 | 31 | 0.7 | 22 | -y |
| `plate__kkjiko` | `z5` | broken | 3.2 | 34.1 | 35 | 89.7 | 76 | -x |
| `plate__luhkiz` | `z2` | borderline | 2.6 | 13.5 | 26 | 2.4 | 28 | +y |
| `plate__luhkiz` | `z4` | borderline | 2.8 | 14.7 | 25 | 2.9 | 28 | +x |
| `plate__luhkiz` | `z3` | borderline | 3.1 | 16.4 | 32 | 3.6 | 37 | -x |
| `plate__luhkiz` | `z5` | broken | 4.9 | 26.7 | 49 | 4.1 | 36 | -z |
| `plate__luhkiz` | `z6` | borderline | 2.8 | 14.9 | 27 | 2.6 | 33 | -x |
| `plate__bgxzec` | `x2` | broken | 0.1 | 0.3 | 1 | 0.0 | 0 |  |
| `plate__bgxzec` | `y2` | broken | 0.1 | 0.3 | 1 | 0.0 | 0 |  |
| `plate__odmjdd` | `z3` | broken | 0.1 | 0.3 | 1 | 0.0 | 0 |  |
| `plate__odmjdd` | `z5` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `plate__odmjdd` | `z6` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `plate__vtjwof` | `x2` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `plate__vtjwof` | `y2` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `plate__mtetqm` | `z4` | borderline | 0.1 | 0.3 | 1 | 0.0 | 0 |  |
| `plate__mtetqm` | `z3` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `plate__mtetqm` | `z5` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `plate__mtetqm` | `z6` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `plate__mtetqm` | `zinf` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `platter__hnlivs` | `z2` | broken | 0.3 | 1.4 | 5 | 0.0 | 0 |  |
| `platter__kiiamx` | `z2` | broken | 5.2 | 43.5 | 65 | 173.4 | 66 | +z |
| `platter__ekjpdj` | `z2` | broken | 6.3 | 45.9 | 61 | 235.8 | 59 | +z |
| `platter__iadlti` | `z2` | broken | 0.3 | 1.1 | 4 | 0.0 | 0 |  |
| `platter__iadlti` | `z4` | broken | 0.3 | 1.1 | 4 | 0.0 | 0 |  |
| `platter__iadlti` | `z5` | broken | 0.3 | 1.1 | 4 | 0.0 | 0 |  |
| `platter__iadlti` | `z6` | broken | 0.3 | 1.2 | 4 | 0.0 | 0 |  |
| `platter__iadlti` | `zinf` | broken | 0.3 | 1.1 | 4 | 0.0 | 0 |  |
| `toy_figure__gqspnb` | `x2` | broken | 4.0 | 13.9 | 48 | 7.0 | 73 | +y |
| `toy_figure__gqspnb` | `x4` | broken | 5.9 | 22.7 | 68 | 7.1 | 74 | -x |
| `toy_figure__gqspnb` | `x6` | broken | 5.9 | 20.8 | 56 | 7.2 | 77 | -x |
| `tennis_ball__rgekxe` | `y2` | broken | 5.5 | 20.9 | 34 | 14.4 | 31 | -x |
| `tennis_ball__rgekxe` | `x4` | broken | 5.9 | 21.0 | 34 | 14.1 | 31 | -x |
| `tennis_ball__rgekxe` | `z5` | broken | 5.8 | 26.7 | 38 | 13.7 | 32 | -y |
| `tennis_ball__rgekxe` | `z6` | broken | 5.6 | 27.3 | 38 | 13.2 | 31 | -y |
| `tray__hjxczh` | `x2` | broken | 1.8 | 5.6 | 9 | 0.0 | 0 |  |
| `tray__hjxczh` | `y2` | broken | 3.8 | 9.5 | 13 | 0.0 | 0 |  |
| `tray__zcmnji` | `x2` | broken | 2.0 | 7.1 | 19 | 0.0 | 0 |  |
| `tray__zcmnji` | `y2` | broken | 1.7 | 6.5 | 14 | 0.0 | 0 |  |
| `tray__spopfj` | `x2` | broken | 0.1 | 0.0 | 3 | 0.0 | 0 |  |
| `tray__spopfj` | `y2` | broken | 0.1 | 0.0 | 3 | 0.0 | 0 |  |
| `tray__jpcflq` | `x2` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `tray__jpcflq` | `y2` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `tray__thkphg` | `x2` | broken | 0.1 | 0.2 | 3 | 0.0 | 0 |  |
| `tray__thkphg` | `y2` | broken | 0.1 | 0.2 | 3 | 0.0 | 0 |  |
| `tray__zsddtq` | `x2` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `tray__zsddtq` | `y2` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `tray__zsddtq` | `z3` | broken | 0.2 | 0.7 | 3 | 0.0 | 0 |  |
| `tray__zsddtq` | `z5` | broken | 0.1 | 0.6 | 2 | 0.0 | 0 |  |
| `tray__zsddtq` | `z6` | broken | 0.1 | 0.6 | 2 | 0.0 | 0 |  |
| `tray__mkdcha` | `x2` | broken | 0.2 | 0.6 | 2 | 0.0 | 0 |  |
| `tray__mkdcha` | `y2` | broken | 0.2 | 0.7 | 2 | 0.0 | 0 |  |
| `tray__hvlfig` | `x2` | broken | 0.1 | 0.0 | 1 | 0.0 | 0 |  |
| `tray__hvlfig` | `y2` | broken | 0.1 | 0.0 | 1 | 0.0 | 0 |  |
| `lemon__pwqedq` | `x2` | broken | 3.8 | 8.0 | 20 | 11.1 | 77 | +x |
| `bowl__belcml` | `z3` | broken | 0.1 | 0.3 | 4 | 0.0 | 0 |  |
| `bowl__belcml` | `z5` | broken | 0.1 | 0.3 | 4 | 0.0 | 0 |  |
| `bowl__belcml` | `z6` | broken | 0.1 | 0.4 | 4 | 0.0 | 0 |  |
| `bowl__belcml` | `zinf` | broken | 0.1 | 0.4 | 4 | 0.0 | 0 |  |
| `bowl__eawgwj` | `z3` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `bowl__eawgwj` | `z5` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `bowl__eawgwj` | `z6` | broken | 0.1 | 0.5 | 3 | 0.0 | 0 |  |
| `bowl__eawgwj` | `zinf` | broken | 0.1 | 0.5 | 2 | 0.0 | 0 |  |
| `hingeless_jar__lvuvbf` | `x2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `hingeless_jar__lvuvbf` | `y2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `hingeless_jar__bqpmsv` | `x2` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `hingeless_jar__bqpmsv` | `y2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `hingeless_jar__wcqjew` | `x2` | broken | 0.0 | 0.0 | 0 | 0.0 | 0 |  |
| `hingeless_jar__wcqjew` | `y2` | broken | 0.0 | 0.0 | 1 | 0.0 | 0 |  |
| `hingeless_jar__mefezc` | `x2` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `hingeless_jar__mefezc` | `y2` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `saucepot__tfzijn` | `z4` | broken | 0.3 | 0.7 | 1 | 0.0 | 0 |  |
| `saucepot__tfzijn` | `z3` | broken | 0.3 | 0.7 | 1 | 0.0 | 0 |  |
| `saucepot__tfzijn` | `z5` | broken | 0.3 | 0.7 | 1 | 0.0 | 0 |  |
| `saucepot__tfzijn` | `z6` | broken | 0.3 | 0.7 | 1 | 0.0 | 0 |  |
| `saucepot__tfzijn` | `zinf` | broken | 0.3 | 0.7 | 1 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `z2` | broken | 0.2 | 0.8 | 7 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `z4` | broken | 0.2 | 0.7 | 5 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `z3` | broken | 0.2 | 0.7 | 7 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `z5` | broken | 0.2 | 0.7 | 6 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `z6` | broken | 0.2 | 0.7 | 6 | 0.0 | 0 |  |
| `saucepot__wfryvm` | `zinf` | broken | 0.2 | 0.7 | 5 | 0.0 | 0 |  |
| `saucepot__vqtkwq` | `z2` | broken | 0.7 | 0.5 | 23 | 44.7 | 45 | +z |
| `saucepot__vqtkwq` | `z4` | broken | 0.4 | 0.4 | 13 | 0.5 | 39 | +z |
| `saucepot__vqtkwq` | `z3` | broken | 0.5 | 0.4 | 17 | 10.0 | 43 | +z |
| `saucepot__vqtkwq` | `z5` | broken | 0.4 | 0.4 | 11 | 0.5 | 30 | +z |
| `saucepot__vqtkwq` | `z6` | broken | 0.3 | 0.4 | 9 | 0.5 | 30 | +z |
| `saucepot__vqtkwq` | `zinf` | broken | 0.3 | 0.4 | 9 | 0.0 | 0 |  |
| `saucepot__uvzmss` | `z4` | broken | 0.3 | 0.7 | 5 | 0.0 | 0 |  |
| `saucepot__uvzmss` | `z3` | broken | 0.3 | 0.6 | 4 | 0.0 | 0 |  |
| `saucepot__uvzmss` | `z5` | broken | 0.4 | 0.7 | 5 | 0.0 | 0 |  |
| `saucepot__uvzmss` | `z6` | broken | 0.3 | 0.6 | 5 | 0.0 | 0 |  |
| `saucepot__uvzmss` | `zinf` | broken | 0.4 | 0.7 | 5 | 0.0 | 0 |  |
| `saucepot__chjetk` | `z4` | broken | 0.1 | 0.2 | 1 | 0.0 | 0 |  |
| `saucepot__chjetk` | `z3` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `saucepot__chjetk` | `z5` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `saucepot__chjetk` | `z6` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `saucepot__chjetk` | `zinf` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `vase__gopbrh` | `x2` | broken | 0.2 | 0.9 | 3 | 0.0 | 0 |  |
| `vase__gopbrh` | `y2` | broken | 0.2 | 0.9 | 2 | 0.0 | 0 |  |
| `vase__gopbrh` | `z3` | broken | 0.1 | 0.7 | 2 | 0.0 | 0 |  |
| `vase__gopbrh` | `z5` | broken | 0.1 | 0.7 | 2 | 0.0 | 0 |  |
| `vase__gopbrh` | `z6` | broken | 0.1 | 0.6 | 2 | 0.0 | 0 |  |
| `vase__gopbrh` | `zinf` | broken | 0.2 | 0.7 | 2 | 0.0 | 0 |  |
| `vase__stqkvx` | `x2` | broken | 0.1 | 0.4 | 2 | 0.0 | 0 |  |
| `vase__stqkvx` | `y2` | broken | 0.1 | 0.3 | 2 | 0.0 | 0 |  |
| `pencil__hmawzy` | `z2` | borderline | 3.3 | 32.8 | 48 | 0.3 | 81 | +x |
| `pencil__hmawzy__scaled` | `z2` | broken | 4.1 | 37.1 | 59 | 7.1 | 74 | +x |
| `pencil__hmawzy__scaled` | `x6` | broken | 1.9 | 15.4 | 37 | 2.7 | 50 | -x |
| `pencil__rzscao__scaled` | `x2` | broken | 1.6 | 9.4 | 36 | 2.8 | 42 | -x |
| `pencil__rzscao__scaled` | `x4` | broken | 1.9 | 15.3 | 39 | 2.7 | 58 | -x |
| `pencil__rzscao__scaled` | `x3` | broken | 1.6 | 10.3 | 36 | 2.5 | 43 | -x |
| `pencil__rzscao__scaled` | `x5` | borderline | 1.9 | 16.1 | 38 | 2.9 | 58 | -x |
| `pencil__rzscao__scaled` | `x6` | borderline | 1.8 | 15.3 | 37 | 2.7 | 55 | -x |
| `pencil__rzscao__scaled` | `xinf` | broken | 1.8 | 13.3 | 38 | 3.4 | 57 | -x |
| `pencil__gbzzcg__scaled` | `x2` | borderline | 2.5 | 12.0 | 66 | 3.4 | 59 | -x |
| `pencil__gbzzcg__scaled` | `x4` | broken | 2.8 | 19.9 | 65 | 3.8 | 69 | -x |
| `pencil__gbzzcg__scaled` | `x3` | broken | 2.8 | 19.3 | 66 | 4.5 | 71 | -x |
| `pencil__gbzzcg__scaled` | `x5` | broken | 2.5 | 16.5 | 61 | 3.2 | 65 | -x |
| `pencil__gbzzcg__scaled` | `x6` | borderline | 2.3 | 13.9 | 55 | 3.3 | 40 | -z |
| `pencil__gbzzcg__scaled` | `xinf` | broken | 2.7 | 14.7 | 69 | 6.6 | 77 | -x |
| `pencil__fnlbpu__scaled` | `x2` | borderline | 1.8 | 9.4 | 45 | 2.8 | 36 | +y |
| `pencil__fnlbpu__scaled` | `x4` | borderline | 2.3 | 16.1 | 45 | 2.9 | 46 | -x |
| `pencil__fnlbpu__scaled` | `x3` | borderline | 1.9 | 9.2 | 41 | 2.5 | 53 | -x |
| `pencil__fnlbpu__scaled` | `x5` | borderline | 2.1 | 15.1 | 45 | 2.7 | 42 | -x |
| `pencil__fnlbpu__scaled` | `x6` | borderline | 2.0 | 12.7 | 43 | 3.4 | 33 | -z |
| `pencil__fnlbpu__scaled` | `xinf` | borderline | 1.9 | 11.5 | 42 | 3.3 | 35 | -z |
| `pencil__heqkmr__scaled` | `x6` | broken | 3.0 | 15.6 | 68 | 4.6 | 64 | -x |
| `pencil__jntybl__scaled` | `x2` | borderline | 2.1 | 9.6 | 54 | 3.3 | 56 | -x |
| `pencil__jntybl__scaled` | `x4` | broken | 2.8 | 20.8 | 56 | 5.6 | 60 | -x |
| `pencil__jntybl__scaled` | `x3` | broken | 2.8 | 19.5 | 58 | 4.6 | 66 | -x |
| `pencil__jntybl__scaled` | `x5` | borderline | 2.5 | 18.0 | 51 | 3.2 | 59 | -x |
| `pencil__jntybl__scaled` | `x6` | broken | 2.3 | 15.9 | 47 | 4.7 | 49 | -x |
| `pencil__jntybl__scaled` | `xinf` | broken | 2.9 | 20.8 | 57 | 5.8 | 63 | -x |
| `pencil__lruiot__scaled` | `x2` | broken | 2.8 | 14.4 | 73 | 5.7 | 77 | -x |
| `pencil__lruiot__scaled` | `x5` | broken | 3.0 | 21.4 | 64 | 3.0 | 59 | -x |
| `pencil__lruiot__scaled` | `x6` | broken | 2.8 | 19.6 | 61 | 3.3 | 54 | -x |
| `pencil__lykqoh__scaled` | `x2` | broken | 2.6 | 12.5 | 64 | 5.7 | 74 | -x |
| `pencil__lykqoh__scaled` | `x5` | broken | 2.5 | 17.2 | 57 | 3.0 | 60 | -x |
| `pencil__lykqoh__scaled` | `x6` | broken | 2.4 | 16.3 | 54 | 2.9 | 57 | -x |
| `pencil__vpzhth__scaled` | `x6` | broken | 2.6 | 15.4 | 60 | 2.8 | 53 | -x |
| `pencil__vvofwr__scaled` | `x2` | broken | 2.4 | 14.3 | 56 | 4.7 | 72 | -x |
| `pencil__vvofwr__scaled` | `x5` | broken | 2.7 | 19.3 | 53 | 5.2 | 66 | -x |
| `pencil__vvofwr__scaled` | `x6` | broken | 2.5 | 17.8 | 52 | 3.2 | 59 | -x |
| `pencil__vedvit__scaled` | `x2` | broken | 1.3 | 7.7 | 31 | 0.6 | 59 | -x |
| `pencil__vedvit__scaled` | `y2` | broken | 3.5 | 30.9 | 67 | 2.0 | 49 | +y |
| `pencil__vedvit__scaled` | `z2` | broken | 4.0 | 34.5 | 68 | 1.9 | 48 | +y |
| `pencil__vedvit__scaled` | `x5` | broken | 1.7 | 10.9 | 42 | 2.8 | 42 | -x |
| `pencil__vedvit__scaled` | `x6` | broken | 1.5 | 9.4 | 37 | 2.3 | 41 | -x |
| `marker__xakoic` | `x2` | borderline | 1.2 | 5.6 | 14 | 1.2 | 41 | +x |
| `marker__xakoic` | `x4` | borderline | 1.0 | 4.8 | 10 | 1.0 | 30 | -x |
| `marker__xakoic` | `x3` | borderline | 1.1 | 5.3 | 13 | 1.0 | 36 | +x |
| `marker__xakoic` | `x5` | borderline | 0.9 | 4.4 | 9 | 1.0 | 28 | -x |
| `marker__xakoic` | `x6` | borderline | 0.8 | 4.3 | 8 | 0.9 | 26 | -x |
| `marker__xakoic` | `xinf` | borderline | 1.1 | 5.0 | 12 | 1.4 | 38 | +x |
| `marker__gldosk` | `x2` | borderline | 1.6 | 6.7 | 27 | 1.2 | 40 | +x |
| `marker__gldosk` | `x4` | borderline | 1.3 | 5.9 | 19 | 1.0 | 31 | -x |
| `marker__gldosk` | `x3` | borderline | 1.5 | 6.6 | 25 | 0.9 | 34 | +x |
| `marker__gldosk` | `x5` | borderline | 1.1 | 5.3 | 16 | 1.0 | 28 | -x |
| `marker__gldosk` | `x6` | borderline | 1.1 | 5.2 | 15 | 0.9 | 26 | -x |
| `marker__gldosk` | `xinf` | borderline | 1.5 | 6.0 | 22 | 1.5 | 43 | +x |
| `marker__vrjwkg` | `x2` | broken | 1.1 | 5.5 | 13 | 1.1 | 27 | -x |
| `marker__vrjwkg` | `x4` | broken | 0.9 | 4.4 | 10 | 0.4 | 25 | -x |
| `marker__vrjwkg` | `x3` | borderline | 1.0 | 4.9 | 11 | 1.0 | 26 | -x |
| `marker__vrjwkg` | `x5` | broken | 0.8 | 4.2 | 9 | 0.0 | 0 |  |
| `marker__vrjwkg` | `x6` | broken | 0.8 | 3.9 | 9 | 0.0 | 0 |  |
| `marker__vrjwkg` | `xinf` | broken | 1.1 | 5.5 | 13 | 0.9 | 27 | -x |
| `marker__vjjsul` | `x2` | broken | 1.8 | 8.0 | 23 | 1.6 | 40 | -x |
| `marker__vjjsul` | `x4` | borderline | 1.4 | 7.0 | 15 | 1.3 | 35 | -x |
| `marker__vjjsul` | `x3` | borderline | 1.6 | 7.5 | 18 | 1.5 | 41 | -x |
| `marker__vjjsul` | `x5` | borderline | 1.3 | 6.4 | 14 | 1.2 | 30 | -x |
| `marker__vjjsul` | `x6` | broken | 1.3 | 6.1 | 14 | 1.0 | 27 | -x |
| `marker__vjjsul` | `xinf` | broken | 1.2 | 5.8 | 13 | 1.0 | 30 | +x |
| `marker__oyjbhy` | `x2` | borderline | 1.8 | 8.1 | 29 | 1.5 | 48 | -x |
| `marker__oyjbhy` | `x4` | borderline | 1.5 | 7.4 | 18 | 1.3 | 37 | -x |
| `marker__oyjbhy` | `x3` | borderline | 1.7 | 7.7 | 22 | 1.6 | 45 | -x |
| `marker__oyjbhy` | `x5` | borderline | 1.5 | 7.0 | 17 | 1.2 | 32 | -x |
| `marker__oyjbhy` | `x6` | borderline | 1.4 | 6.5 | 16 | 1.0 | 28 | -x |
| `marker__oyjbhy` | `xinf` | borderline | 1.8 | 7.9 | 24 | 1.6 | 46 | -x |
| `marker__mznenj` | `x2` | borderline | 1.4 | 6.6 | 17 | 1.2 | 42 | +x |
| `marker__mznenj` | `x4` | borderline | 1.1 | 5.6 | 13 | 1.1 | 32 | -x |
| `marker__mznenj` | `x3` | borderline | 1.3 | 6.2 | 16 | 1.0 | 35 | -x |
| `marker__mznenj` | `x5` | borderline | 1.0 | 5.2 | 11 | 1.1 | 28 | -x |
| `marker__mznenj` | `x6` | borderline | 1.0 | 5.2 | 10 | 1.0 | 26 | -x |
| `marker__mznenj` | `xinf` | borderline | 1.3 | 6.0 | 15 | 1.4 | 40 | +x |
| `pepper_shaker__ykujyn` | `zinf` | borderline | 3.1 | 10.8 | 16 | 0.3 | 31 | -z |
| `pepper_shaker__stbqbt` | `z3` | borderline | 0.8 | 2.3 | 3 | 0.0 | 0 |  |
| `salt_shaker__iomwtn` | `z3` | borderline | 4.7 | 12.4 | 19 | 3.9 | 51 | -z |
| `salt_shaker__fonexa` | `x2` | broken | 1.9 | 2.7 | 63 | 9.9 | 73 | -z |
| `salt_shaker__fonexa` | `y2` | broken | 1.9 | 2.7 | 63 | 9.9 | 73 | -z |
| `carrot__nktmff` | `z2` | broken | 3.6 | 7.5 | 9 | 0.0 | 0 |  |
| `colored_pencil__acumgp` | `x2` | broken | 0.8 | 1.4 | 17 | 0.0 | 0 |  |
| `colored_pencil__acumgp` | `x4` | broken | 0.9 | 1.5 | 21 | 0.0 | 0 |  |
| `colored_pencil__acumgp` | `x3` | broken | 0.8 | 1.5 | 19 | 0.0 | 0 |  |
| `colored_pencil__acumgp` | `xinf` | broken | 0.7 | 1.5 | 10 | 0.0 | 0 |  |
| `colored_pencil__bkdqwb` | `x4` | broken | 1.6 | 6.4 | 25 | 0.0 | 0 |  |
| `colored_pencil__bkdqwb` | `x3` | broken | 1.5 | 6.2 | 21 | 0.0 | 0 |  |
| `colored_pencil__bnsqcn` | `x2` | broken | 0.7 | 2.8 | 13 | 0.0 | 0 |  |
| `colored_pencil__bnsqcn` | `y2` | broken | 1.3 | 7.3 | 29 | 0.0 | 0 |  |
| `colored_pencil__bnsqcn` | `x3` | broken | 0.6 | 2.0 | 14 | 0.0 | 0 |  |
| `colored_pencil__bnsqcn` | `xinf` | broken | 0.6 | 2.1 | 7 | 0.0 | 0 |  |
| `colored_pencil__deuvcx` | `x4` | borderline | 2.0 | 8.7 | 32 | 0.3 | 102 | -x |
| `colored_pencil__deuvcx` | `x3` | broken | 1.8 | 8.2 | 27 | 0.2 | 111 | -x |
| `colored_pencil__deuvcx` | `x5` | borderline | 1.8 | 8.8 | 31 | 0.3 | 92 | -x |
| `colored_pencil__egvqng` | `z2` | borderline | 3.6 | 37.1 | 56 | 0.7 | 87 | +x |
| `colored_pencil__egvqng` | `x4` | broken | 1.2 | 4.2 | 19 | 0.0 | 0 |  |
| `colored_pencil__egvqng` | `x3` | broken | 1.1 | 3.9 | 13 | 0.0 | 0 |  |
| `colored_pencil__jdqvdl` | `z2` | borderline | 3.1 | 36.4 | 44 | 0.3 | 71 | +x |
| `colored_pencil__jdqvdl` | `x4` | broken | 1.0 | 3.7 | 19 | 0.0 | 0 |  |
| `colored_pencil__jdqvdl` | `x3` | broken | 1.1 | 5.2 | 22 | 0.0 | 0 |  |
| `colored_pencil__jnccuz` | `x2` | broken | 1.2 | 5.6 | 15 | 0.0 | 0 |  |
| `colored_pencil__jnccuz` | `x4` | broken | 1.2 | 5.5 | 21 | 0.0 | 0 |  |
| `colored_pencil__jnccuz` | `x3` | broken | 1.1 | 4.8 | 18 | 0.0 | 0 |  |
| `colored_pencil__jnccuz` | `xinf` | broken | 1.0 | 3.8 | 14 | 0.0 | 0 |  |
| `colored_pencil__kadvlg` | `x4` | broken | 1.3 | 4.4 | 27 | 0.0 | 0 |  |
| `colored_pencil__kadvlg` | `x3` | broken | 1.2 | 3.8 | 20 | 0.0 | 0 |  |
| `colored_pencil__kgmapz` | `x4` | broken | 1.8 | 8.7 | 31 | 0.2 | 87 | -x |
| `colored_pencil__kgmapz` | `x3` | broken | 1.7 | 7.5 | 25 | 0.0 | 0 |  |
| `colored_pencil__kgmapz` | `xinf` | broken | 1.6 | 7.4 | 24 | 0.0 | 0 |  |
| `colored_pencil__kjwoqm` | `x3` | broken | 1.6 | 6.0 | 22 | 0.0 | 0 |  |
| `colored_pencil__nssris` | `x2` | broken | 0.6 | 2.9 | 12 | 0.0 | 0 |  |
| `colored_pencil__nssris` | `x3` | broken | 0.7 | 2.5 | 19 | 0.0 | 0 |  |
| `colored_pencil__nssris` | `xinf` | broken | 0.5 | 1.9 | 7 | 0.0 | 0 |  |
| `colored_pencil__qrxemk` | `x2` | broken | 1.6 | 9.5 | 22 | 0.0 | 0 |  |
| `colored_pencil__qrxemk` | `x4` | broken | 1.7 | 10.5 | 24 | 0.0 | 0 |  |
| `colored_pencil__qrxemk` | `x3` | broken | 1.6 | 8.9 | 28 | 0.2 | 87 | -x |
| `colored_pencil__qrxemk` | `xinf` | broken | 1.6 | 8.2 | 30 | 0.2 | 89 | -x |
| `can_of_baking_mix__orwvfx` | `z2` | broken | 2.5 | 14.8 | 39 | 16.0 | 41 | +z |
| `can_of_baking_mix__rxorlp` | `z2` | borderline | 2.0 | 12.9 | 29 | 1.3 | 31 | +z |
| `teapot__foaehs` | `x2` | broken | 0.5 | 1.0 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `y2` | broken | 0.5 | 1.0 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `z2` | broken | 0.5 | 1.0 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `x4` | broken | 0.5 | 1.0 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `z4` | broken | 0.4 | 1.0 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `x3` | broken | 0.5 | 1.0 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `z3` | broken | 0.5 | 1.0 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `x5` | broken | 0.5 | 1.0 | 3 | 0.0 | 0 |  |
| `teapot__foaehs` | `z5` | broken | 0.4 | 1.0 | 2 | 0.0 | 0 |  |
| `teapot__foaehs` | `z6` | broken | 0.4 | 1.0 | 2 | 0.0 | 0 |  |
| `teapot__foaehs` | `zinf` | broken | 0.4 | 1.0 | 3 | 0.0 | 0 |  |
| `teapot__jlalfc` | `x2` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `teapot__jlalfc` | `y2` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `teapot__jlalfc` | `z2` | broken | 0.1 | 0.2 | 2 | 0.0 | 0 |  |
| `teapot__jlalfc` | `z4` | broken | 0.1 | 0.1 | 2 | 0.0 | 0 |  |
| `teapot__jlalfc` | `z3` | broken | 0.1 | 0.1 | 2 | 0.0 | 0 |  |
| `teapot__jlalfc` | `z5` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `teapot__jlalfc` | `z6` | broken | 0.0 | 0.1 | 1 | 0.0 | 0 |  |
| `teapot__jlalfc` | `zinf` | broken | 0.1 | 0.1 | 2 | 0.0 | 0 |  |

## Agreement with the reviewer verdicts (review/*.json, judged on the current texture class)

- DISAGREE reviewer broken / survey preserved: 20
- DISAGREE reviewer preserved / survey broken: 100
- agree (broken): 240
- agree (preserved): 47
- n/a (unsure or not scored): 4
- reviewer broken / survey borderline: 74
- reviewer preserved / survey borderline: 106

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
| rev_03 | `cinnamon_stick__cdkjfo` | `yinf` | preserved | broken | Uniform unprinted quill that is nearly round; rolled about its axis it shows the same thin |
| rev_03 | `cinnamon_stick__qmlyim` | `z2` | preserved | broken | Uniform unprinted quill with alike forked ends; the end swap leaves the same bar, the resi |
| rev_03 | `cinnamon_stick__qmlyim__scaled` | `x2` | preserved | broken | Uniform unprinted quill; the scaled stick stays on its axis with no wobble and flipped ove |
| rev_03 | `cinnamon_stick__qsqvgk` | `x2` | preserved | broken | Uniform unprinted flat quill with alike forked ends; flipped over it shows the same bar fr |
| rev_03 | `cinnamon_stick__qsqvgk` | `y2` | preserved | broken | Uniform unprinted quill; the half turn about its long axis leaves the same bar, the residu |
| rev_03 | `cinnamon_stick__qsqvgk__scaled` | `y6` | preserved | broken | Unprinted quill; the 60-degree roll changes its apparent thickness by a couple of millimet |
| rev_03 | `cinnamon_stick__qxpzdm` | `y2` | preserved | broken | Uniform unprinted quill; the half turn about its long axis only reverses a bow of about 2  |
| rev_04 | `colored_pencil__acumgp` | `xinf` | preserved | broken | Rolling about the long axis leaves an unprinted, uniformly coloured pencil body with the t |
| rev_04 | `colored_pencil__bnsqcn` | `xinf` | preserved | broken | No print or colour change anywhere around the uniform pencil body, so any roll about the l |
| rev_04 | `colored_pencil__jnccuz` | `xinf` | preserved | broken | The body is unprinted and uniformly coloured, so a roll about the long axis is invisible a |
| rev_04 | `colored_pencil__kgmapz` | `xinf` | preserved | broken | Uniform unprinted body with the tip staying at the same end; nothing around the pencil mar |
| rev_04 | `colored_pencil__nssris` | `xinf` | preserved | broken | No print or marking around the uniform body, so any rotation about the long axis looks ide |
| rev_04 | `colored_pencil__qrxemk` | `xinf` | preserved | broken | Uniform unprinted pencil body, tip stays at the same end; the roll is invisible. |
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
| rev_09 | `table_knife__lrdmpf` | `y2` | preserved | broken | Thin uniform grey spreader whose two rounded ends are alike; after the end swap the outlin |
| rev_09 | `table_knife__lrdmpf` | `z2` | preserved | broken | Same as y2: both ends of the flat spreader look the same and the half turn about z leaves  |
| rev_09 | `towel_rack__crivoa` | `x2` | preserved | broken | Judged the half turn about the +x (front) axis: a plain black bar with seven identical peg |
| rev_10 | `toy_dice__bjrxkj` | `x2` | broken | preserved | The fish outline on the +x/-x faces is rotated 180 degrees in place (tail swaps sides, bel |
| rev_10 | `toy_dice__bjrxkj` | `z2` | broken | preserved | The side fish and I faces do match, but the 6 on the top (and bottom) face is turned into  |
| rev_10 | `vase__aegxpb` | `zinf` | preserved | broken | Round dark vase with horizontal banded texture that runs continuously around the body; onl |
| rev_10 | `vase__stqkvx` | `zinf` | broken | preserved | All-glass vase judged by shape: the rim is scalloped with six lobes clearly visible from a |
| rev_10 | `whiteboard__gvkgsi` | `x2` | preserved | broken | Plain white board in a uniform thin frame with a plain grey back and no tray, print or asy |
| rev_11 | `asparagus__xguktb` | `y5` | preserved | broken | Rotation about the stalk's long axis keeps tip and cut end in place; the stalk is nearly s |
| rev_12 | `chopstick__bcwuwv` | `x4` | preserved | broken | Unprinted two-tone chopstick (dark tip, light body) rolled 90° about its long axis: no fla |
| rev_12 | `chopstick__bcwuwv` | `x5` | preserved | broken | Roll by 72° about the long axis of the same unprinted two-tone stick: colours, taper and t |
| rev_12 | `chopstick__bcwuwv` | `x6` | preserved | broken | Roll by 60° about the long axis of the same unprinted two-tone stick: colours, taper and t |
| rev_12 | `cinnamon_stick__qxpzdm__scaled` | `y5` | broken | preserved | The 72° roll about the long axis turns the bark slit off the top face: the forked notch se |
| rev_12 | `cinnamon_stick__qxpzdm__scaled` | `y6` | broken | preserved | The 60° roll about the long axis turns the bark slit off the top face: the forked notch of |
| rev_12 | `colored_pencil__bkdqwb` | `xinf` | preserved | broken | Roll about the long axis of an unprinted uniform-colour pencil: the tip stays at the same  |
| rev_12 | `colored_pencil__deuvcx` | `xinf` | preserved | broken | Roll about the long axis of an unprinted uniform-colour pencil: the tip stays at the same  |
| rev_12 | `colored_pencil__egvqng` | `xinf` | preserved | broken | Roll about the long axis of an unprinted uniform-colour pencil: the tip stays at the same  |
| rev_12 | `colored_pencil__jdqvdl` | `xinf` | preserved | broken | Roll about the long axis of an unprinted uniform-colour pencil: the tip stays at the same  |
| rev_12 | `colored_pencil__kadvlg` | `xinf` | preserved | broken | Roll about the long axis of an unprinted uniform-colour pencil: the tip stays at the same  |
| rev_12 | `colored_pencil__kjwoqm` | `xinf` | preserved | broken | Roll about the long axis of an unprinted uniform-colour pencil: the tip stays at the same  |
| rev_13 | `pencil__hmawzy` | `xinf` | preserved | broken | Unprinted golden pencil; the x131 row is identical apart from a ~1-2 mm wobble of the thin |
| rev_13 | `pencil__lykqoh` | `y2` | broken | preserved | The pale sharpened tip swaps from the -x to the +x end after the half turn about y, visibl |
| rev_14 | `pencil__vpzhth` | `y2` | broken | preserved | The end-to-end flip about y swaps the sharpened, tapering tip with the flat-cut end; the t |
| rev_14 | `pencil__vvofwr` | `y2` | broken | preserved | The end-to-end flip about y swaps the sharpened, tapering tip with the flat-cut end, clear |
| rev_14 | `shelf__kxvbui` | `z2` | preserved | broken | Plain black rectangular slab with no visible edge feature: the z180 tiles match the origin |
| rev_14 | `wall_mounted_tv__ggfznh` | `x2` | preserved | broken | Plain grey screen in a black bezel with a plain black back: the flipped tiles match to sin |
| rev_14 | `wall_mounted_tv__ylvjhb` | `x2` | preserved | broken | Simple box TV with a plain grey screen, a bezel equal top and bottom to under 1 mm and a p |
