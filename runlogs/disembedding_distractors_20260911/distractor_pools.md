# Disembedding distractor pools (per target category)

Pool = 20 candidate models per target category: same-category look-alike groups first (a group counts once, at most 10 groups), related-category models fill the rest (at most 3 per category). Every listed model rested stably on the source tray in the head-camera render check and produced one sampler instance as the distractor of the category's test target (`results/`). `reserve` = validated but beyond the pool size.

| target category | same groups (eligible) | same groups in pool | same models in pool | other models | total | reserve same / other | note |
|---|---|---|---|---|---|---|---|
| bar_soap | 4 | 4 | 4 | 16 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| beaker | 7 | 7 | 8 | 13 | 20 | 0 / 4 |  |
| bottle_of_beer | 8 | 8 | 13 | 12 | 20 | 0 / 4 |  |
| bottle_of_medicine | 7 | 7 | 7 | 13 | 20 | 0 / 4 |  |
| bottle_of_milk | 6 | 6 | 6 | 14 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| bottle_of_pop | 26 | 10 | 16 | 10 | 20 | 18 / 3 |  |
| bottle_of_shampoo | 3 | 3 | 4 | 17 | 20 | 0 / 4 |  |
| bottle_of_wine | 21 | 10 | 12 | 10 | 20 | 14 / 3 |  |
| box_of_baking_mix | 2 | 2 | 8 | 18 | 20 | 0 / 4 |  |
| box_of_butter | 2 | 2 | 2 | 17 | 19 | 0 / 0 | SHORT |
| box_of_cane_sugar | 4 | 4 | 4 | 16 | 20 | 0 / 4 |  |
| box_of_cereal | 11 | 10 | 13 | 10 | 20 | 1 / 4 |  |
| box_of_cookies | 10 | 10 | 10 | 10 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| box_of_granola_bars | 3 | 3 | 3 | 17 | 20 | 0 / 0 |  |
| box_of_oatmeal | 0 | 0 | 0 | 18 | 18 | 0 / 0 | SHORT |
| box_of_rice | 3 | 3 | 3 | 17 | 20 | 0 / 2 |  |
| box_of_sanitary_napkins | 9 | 9 | 9 | 11 | 20 | 0 / 4 |  |
| box_of_shampoo | 4 | 4 | 4 | 16 | 20 | 0 / 4 |  |
| box_of_tissues | 4 | 4 | 4 | 16 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| box_of_yogurt | 3 | 3 | 3 | 17 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| can_of_soda | 9 | 9 | 9 | 11 | 20 | 0 / 4 |  |
| carton_of_milk | 1 | 1 | 1 | 0 | 1 | 0 / 0 | SHORT |
| cellulose_tape | 3 | 3 | 3 | 17 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| chili | 2 | 2 | 2 | 18 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| conditioner_dispenser | 2 | 2 | 3 | 18 | 20 | 0 / 4 |  |
| cookie_cutter | 3 | 3 | 3 | 17 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| jar_of_bath_salt | 1 | 1 | 2 | 19 | 20 | 0 / 4 |  |
| mouse | 4 | 4 | 6 | 16 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| mug | 8 | 8 | 10 | 12 | 20 | 0 / 4 |  |
| notebook | 13 | 10 | 11 | 10 | 20 | 3 / 4 | template setup FAILED: render-only evidence |
| pack_of_cigarettes | 16 | 10 | 10 | 10 | 20 | 6 / 4 | template setup FAILED: render-only evidence |
| pencil_case | 3 | 3 | 3 | 17 | 20 | 0 / 2 |  |
| pencil_holder | 1 | 1 | 1 | 0 | 1 | 0 / 0 | SHORT |
| soap_dispenser | 6 | 6 | 6 | 14 | 20 | 0 / 4 |  |
| tablefork | 4 | 4 | 5 | 16 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| teaspoon | 5 | 5 | 8 | 15 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| tortilla_chips | 4 | 4 | 5 | 16 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| toy_dice | 7 | 7 | 19 | 13 | 20 | 0 / 4 | template setup FAILED: render-only evidence |
| video_game | 11 | 10 | 10 | 10 | 20 | 1 / 4 | template setup FAILED: render-only evidence |

## bar_soap (test target ofargb)
Same-category pool: ofargb [g1], feqemg [g2], ozifwa [g3], utgixp [g4]
Other-category pool: sponge/aewrov, soap_dish/gnndfy, mouse/cduwda, deodorant_stick/albeyc, sugar_cookie/cgkiti, butter_cookie/kukrla, chocolate_bar/amutpr, pack_of_chocolate_bar/bxgewr, biscuit/ukcwqw, cheesecake/epmobi, granola_bar/iurfsq, potato/bpwohr, lemon/asdada, match_box/xpnfjz, lighter/zsxfjz, coaster/arjpcz

## beaker (test target jdijek)
Look-alike groups (init view): {fyrkzs, zycgen} - exact dup: identical dark chrome 8 cm cylinders, same highlights and bright tray reflection at every yaw
Same-category pool: jdijek [g1], zycgen [g2], fyrkzs [g2], rhohgs [g3], uobdoq [g4], exzsal [g5], effbnc [g6], sfvswx [g7]
Other-category pool: water_glass/cdteyb, mug/ycbmug, coffee_cup/ckkwmj, teacup/cpozxi, soda_cup/fsfsas, paper_cup/guobeq, beer_glass/lymciz, wineglass/aakcyj, goblet/nawrfs, water_glass/edfzlt, mug/ppzttc, coffee_cup/dkxddg, teacup/oxfzfe
Same-category candidates dropped: beaker/dtjmai (wide shallow grey bowl, blurry 3k px; does not read as a beaker); beaker/fxrsyi (dark untextured square tube/box, does not read as a beaker (already excluded))

## bottle_of_beer (test target meqliv)
Look-alike groups (init view): {crfcwo, meqliv} - Exact duplicates: same 9x9x22 green bottle, gold cap, teal label with dark round emblem at every yaw.; {fgzjnb, gxxbhh} - Same 7x7x22 geometry, dark green glass, gold cap, dark warm-toned wrap label with round emblem; maroon vs brown hue only, hidden at yaw 180.; {dcwvkg, vhscym} - Exact duplicates: same 11x11x29 green bottle with yellow IPA label and yellow neck label.; {kqskmv, qepxvl, rdnopv} - Slim black 6-cm bottles with tan/gold decorated body label and small tan neck label; label-away yaws show a plain black bottle. Heights 25/23/21 cm.
Same-category pool: meqliv [g1], fgzjnb [g2], zbsxro [g3], crfcwo [g1], dcwvkg [g4], kqskmv [g5], vhscym [g4], gxxbhh [g2], qepxvl [g5], rdnopv [g5], saslsh [g6], phdimo [g7], zlmwyn [g8]
Other-category pool: bottle_of_pop/ribekf, bottle_of_wine/lacaoa, bottle_of_water/atvnqy, bottle_of_champagne/mjpyvh, lemonade_bottle/yprkek, bottle_of_sake/pwtebq, bottle_of_rum/bnvoux, bottle_of_whiskey/aipxlu, bottle_of_ginger_beer/zkocwb, bottle_of_soda/eqyqlx, soda_water_bottle/upfssc, bottle_of_pop/wrgmdt

## bottle_of_medicine (test target zfbnjh)
Same-category pool: zfbnjh [g1], kqkwoq [g2], syfpak [g3], egondf [g4], fmfwng [g5], kasbsy [g6], qqsukh [g7]
Other-category pool: bottle_of_supplements/kgreql, bottle_of_antihistamines/agavwx, bottle_of_black_pepper/ejtiig, bottle_of_paprika/nrdgrp, bottle_of_sage/bterim, bottle_of_seasoning/vqwqqv, bottle_of_ground_nutmeg/qebruq, bottle_of_ground_cloves/vzamzb, bottle_of_ground_mace/lgpxro, bottle_of_mustard_seeds/grryaf, bottle_of_sesame_seeds/lzndie, bottle_of_fennel/ihlkfu, bottle_of_ginger/drqhzo
Same-category candidates dropped: bottle_of_medicine/hvocpc (tall thin orange rectangular carton (4x4x11), not a bottle)

## bottle_of_milk (test target mrejrs)
Same-category pool: mrejrs [g1], ddvsgl [g2], qsnnbp [g3], czblzn [g4], utfsmp [g5], debqen [g6]
Other-category pool: carton_of_milk/kszoro, bottle_of_water/atvnqy, bottle_of_pop/ribekf, bottle_of_soy_milk/mcjlhs, bottle_of_coconut_milk/idenxg, carton_of_soy_milk/orgyvw, box_of_almond_milk/oiiqwq, bottle_of_milkshake/naxqya, bottle_of_orange_juice/edltwh, bottle_of_apple_juice/xvrbdy, bottle_of_cranberry_juice/heoxnw, bottle_of_tea/iladfg, bottle_of_coconut_water/lsixio, bottle_of_soda/eqyqlx

## bottle_of_pop (test target ribekf)
Look-alike groups (init view): {nepfjl, ubazru, xoldze} - Dark bottle, red cap, red cola wrap label with white script; differ only by small text and a thin pale band.; {mhzttm, nyupqh, sevoto, vfjhav} - Dark bottle, grey cap, pale blue-grey label with small red/blue logos; vfjhav's red front panel hides at side yaws.; {pyrics, wmqhul, zxambx} - Dark bottle, yellow cap, blue wrap label with small yellow/green fruit graphics; graphics and blue shade differ only subtly.; {haoywb, hjrwqb, uwmchl} - Magenta body and cap, magenta label with small text; only text colour and a thin grey band differ.; {oeipjl, pfjzkn} - Same slim dark bottle, navy cap, dark label with red Tribal-Cola print at every yaw; only label shade differs slightly.
Same-category pool: ribekf [g1], wrgmdt [g2], xoldze [g3], ghxeqz [g4], nyupqh [g5], vfjhav [g5], zxambx [g6], mdznsn [g7], sevoto [g5], bqretx [g8], cmqubs [g9], dlqmit [g10], nepfjl [g3], pyrics [g6], ubazru [g3], wmqhul [g6]
Other-category pool: bottle_of_water/atvnqy, bottle_of_beer/fgzjnb, soda_water_bottle/upfssc, bottle_of_soda/eqyqlx, lemonade_bottle/yprkek, bottle_of_ginger_beer/zkocwb, bottle_of_tea/iladfg, bottle_of_orange_juice/edltwh, bottle_of_cranberry_juice/heoxnw, bottle_of_fruit_punch/azcigi
Same-category candidates dropped: bottle_of_pop/mhzttm (no_instances: rejections {'target_placement': 1, 'unstable_i); bottle_of_pop/gkwdyt (Deformed, crumpled translucent red mesh; does not read as an intact pop bottle.); bottle_of_pop/hjrwqb (no_instances: rejections {'target_placement': 1, 'unstable_i)

## bottle_of_shampoo (test target stjjjm)
Look-alike groups (init view): {lvvkhx, stjjjm} - Same grey 6x6x17 cm cylinder bottle and cap; only faint label accent colour differs, and the label faces away at most yaws
Same-category pool: stjjjm [g1], dvrzmy [g2], lvvkhx [g1], hlkpwd [g3]
Other-category pool: shampoo_dispenser/eblcnw, soap_dispenser/zwudcz, bottle_of_conditioner/teafxb, conditioner_atomizer/quqgcu, bottle_of_lotion/tkryrh, tube_of_lotion/ixfuxm, bottle_of_dish_soap/bnmixt, deodorant_atomizer/utjfug, deodorant_stick/albeyc, detergent_atomizer/rhfpsb, water_atomizer/lfarai, bottle_of_perfume/ipurzb, bottle_of_cologne/lyipur, bottle_of_face_cream/dztaed, box_of_shampoo/bclpiq, spray_bottle/asztxi, shampoo_dispenser/eywylh

## bottle_of_wine (test target lacaoa)
Look-alike groups (init view): {lacaoa, njoeiz} - Identical dark champagne bottles with full gold-foil neck, same height and pixel count (geom cluster c2); no label difference visible at any yaw.; {jdqahn, wcokej} - Dark grey-green Burgundy-shape bottle with tan cork, ~24-25 cm; only faint low-contrast label patches, indistinguishable at every yaw.; {hnkiog, vecadn} - Dark (green / near-black) Bordeaux bottle, red foil cap, cream label on one side only; 24 vs 27 cm, same look when label faces away.; {ivuvbv, pgfplm, qvsnjo} - Same 7x7x24 dark-green Bordeaux bottle with pale (white/cream-gold) foil cap; only a one-sided paper label differs, hidden at yaws 90/180.
Same-category pool: lacaoa [g1], wcokej [g2], njoeiz [g1], xkjyby [g3], yaxufs [g4], oldpru [g5], rztiuc [g6], ynfvhg [g7], fzuiwy [g8], klbwkf [g9], vmdwln [g10], jdqahn [g2]
Other-category pool: bottle_of_champagne/mjpyvh, bottle_of_beer/fgzjnb, bottle_of_sake/pwtebq, bottle_of_whiskey/aipxlu, bottle_of_vodka/nwegts, bottle_of_gin/qzgcdx, bottle_of_tequila/gbozpr, bottle_of_champagne/npsevx, bottle_of_beer/zbsxro, bottle_of_sake/swlykk
Same-category candidates dropped: bottle_of_wine/bmudli (Settles at 16 deg tilt; visibly leaning in every crop, looks toppled / broken base.)

## box_of_baking_mix (test target fmewsz)
Look-alike groups (init view): {fmewsz, ltccey, luwlfd} - Identical 5x12x15 white carton with red top band on every face; only the round food photo/title on the wide faces differs, hidden edge-on.; {fnbgwd, lgusnr, qkvqkm, uhvkxj, xcxrek} - Same white 5x12x15 carton, no band; narrow faces near-blank white with a small warm colour patch, so edge-on views are interchangeable; photos differ only front-on.
Same-category pool: fmewsz [g1], lgusnr [g2], uhvkxj [g2], fnbgwd [g2], ltccey [g1], luwlfd [g1], qkvqkm [g2], xcxrek [g2]
Other-category pool: box_of_cereal/iucshm, box_of_rice/pstqvm, box_of_cane_sugar/gpbobo, box_of_brown_sugar/kqyevo, box_of_flour/ylezpk, box_of_baking_powder/vzgrlv, box_of_baking_soda/pskrgy, box_of_crackers/cmdigf, box_of_barley/lxpwnw, box_of_lasagna/oaonbr, box_of_coffee/cjtadw, box_of_granola_bars/tqwhvz, box_of_cookies/apjokz, box_of_candy/gvreky, can_of_baking_mix/blrqqz, pack_of_pasta/gwqhko, box_of_cereal/ykuyqb, box_of_rice/fvulow

## box_of_butter (test target oixapu)
Same-category pool: oixapu [g1], mfjuil [g2]
Other-category pool: butter/ovirmc, bag_of_cream_cheese/dxrsol, box_of_cookies/apjokz, box_of_granola_bars/tqwhvz, box_of_baking_mix/fmewsz, pack_of_chocolate_bar/bxgewr, box_of_candy/gvreky, box_of_crackers/cmdigf, box_of_coffee/cjtadw, bag_of_cookies/bbyvsc, cheesecake/epmobi, bag_of_ice_cream/himfgn, box_of_granola_bars/bqeeki, box_of_baking_mix/lgusnr, pack_of_chocolate_bar/tidcst, bag_of_cookies/ikivgk, bag_of_ice_cream/jwalbo

## box_of_cane_sugar (test target gpbobo)
Same-category pool: gpbobo [g1], qvjiti [g2], cqlofx [g3], rvsivw [g4]
Other-category pool: box_of_brown_sugar/kqyevo, box_of_flour/ylezpk, bag_of_flour/fntqmd, box_of_rice/pstqvm, box_of_cereal/iucshm, box_of_baking_mix/fmewsz, box_of_baking_powder/vzgrlv, box_of_baking_soda/pskrgy, box_of_barley/lxpwnw, box_of_crackers/cmdigf, box_of_coffee/cjtadw, box_of_lasagna/oaonbr, jar_of_sugar/bholak, box_of_cookies/apjokz, box_of_rice/fvulow, box_of_cereal/ykuyqb

## box_of_cereal (test target iucshm)
Look-alike groups (init view): {ihedod, yorray} - Light beige dorset-style boxes of near-identical size and layout; tan vs cream shade and the front graphic are the only cues, sides/back look alike.; {sgirte, wdkcbu, wrlalk} - Near-black dorset-style boxes, same size and layout; only the front graphic differs, sides and back are identical dark panels with a tiny logo.
Same-category pool: iucshm [g1], ykuyqb [g2], surzft [g3], hcwbzw [g4], hosfhj [g5], ihedod [g6], qirvmd [g7], sgirte [g8], tiykku [g9], wdkcbu [g8], yorray [g6], pfwgkm [g10], wrlalk [g8]
Other-category pool: box_of_rice/pstqvm, box_of_cane_sugar/gpbobo, box_of_baking_mix/fmewsz, box_of_crackers/cmdigf, box_of_granola_bars/tqwhvz, box_of_cookies/apjokz, box_of_barley/lxpwnw, box_of_flour/ylezpk, box_of_brown_sugar/kqyevo, box_of_lasagna/oaonbr

## box_of_cookies (test target apjokz)
Same-category pool: apjokz [g1], dlhqft [g2], nirfva [g3], jtalxs [g4], nnfisz [g5], sjcnoq [g6], hdgsxz [g7], lwhbid [g8], fsdjun [g9], qkzrdd [g11]
Other-category pool: box_of_crackers/cmdigf, box_of_candy/gvreky, bag_of_cookies/bbyvsc, box_of_cereal/iucshm, box_of_granola_bars/tqwhvz, box_of_baking_mix/fmewsz, box_of_coffee/cjtadw, box_of_candy/jodwfv, bag_of_cookies/ikivgk, box_of_cereal/ykuyqb
Same-category candidates dropped: box_of_cookies/swilxz (octagonal teal tub with ice-cream lid print; reads as an ice-cream tub, not a cookie box)

## box_of_granola_bars (test target tqwhvz)
Same-category pool: tqwhvz [g1], bqeeki [g2], awombj [g3]
Other-category pool: box_of_baking_mix/fmewsz, box_of_crackers/cmdigf, box_of_cereal/iucshm, box_of_candy/gvreky, chocolate_bar/amutpr, pack_of_chocolate_bar/bxgewr, granola_bar/iurfsq, box_of_butter/oixapu, box_of_coffee/cjtadw, box_of_rice/pstqvm, box_of_cane_sugar/gpbobo, bag_of_cookies/bbyvsc, box_of_baking_powder/vzgrlv, box_of_baking_mix/lgusnr, box_of_cereal/ykuyqb, chocolate_bar/bcfudr, pack_of_chocolate_bar/tidcst

## box_of_oatmeal (test target jtqeef)
Same-category pool: -
Other-category pool: box_of_rice/pstqvm, box_of_cane_sugar/gpbobo, bag_of_flour/fntqmd, box_of_flour/ylezpk, box_of_brown_sugar/kqyevo, box_of_barley/lxpwnw, box_of_crackers/cmdigf, can_of_oatmeal/qyukhm, bag_of_snacks/pkwgid, bag_of_tea/jjweyi, bag_of_breadcrumbs/nvhvxe, box_of_baking_powder/vzgrlv, box_of_coffee/cjtadw, box_of_lasagna/oaonbr, paper_bag/bzsxgw, box_of_cereal/ykuyqb, box_of_rice/fvulow, box_of_cane_sugar/qvjiti
Same-category candidates dropped: box_of_oatmeal/jtqeef (UNSTABLE (tilt 17.2 deg): the bag fell over / slumped on the tray.); box_of_oatmeal/rabeel (no_instances: rejections {'distractor_placement': 3, 'target); box_of_oatmeal/zkggxm (no_instances: rejections {'distractor_placement': 3, 'target)

## box_of_rice (test target pstqvm)
Same-category pool: pstqvm [g1], fvulow [g2], yyheuw [g3]
Other-category pool: box_of_cereal/iucshm, box_of_cane_sugar/gpbobo, box_of_brown_sugar/kqyevo, box_of_flour/ylezpk, box_of_baking_mix/fmewsz, box_of_barley/lxpwnw, box_of_crackers/cmdigf, box_of_lasagna/oaonbr, box_of_baking_powder/vzgrlv, box_of_coffee/cjtadw, bag_of_flour/fntqmd, box_of_granola_bars/tqwhvz, box_of_cereal/ykuyqb, box_of_cane_sugar/qvjiti, box_of_baking_mix/lgusnr, box_of_baking_powder/zevydc, box_of_coffee/mreyla

## box_of_sanitary_napkins (test target tkrzkb)
Same-category pool: tkrzkb [g1], acaivd [g2], zqinib [g3], arqcyu [g4], humkqw [g5], skltgp [g6], ykpeja [g7], cmnpuv [g8], mxdefq [g9]
Other-category pool: box_of_tissues/uglbjc, bag_of_cookies/bbyvsc, bag_of_ice_cream/himfgn, bag_of_jerky/wblype, bag_of_snacks/pkwgid, bag_of_tea/jjweyi, bag_of_breadcrumbs/nvhvxe, bag_of_cream_cheese/dxrsol, box_of_shampoo/bclpiq, box_of_cereal/iucshm, box_of_cane_sugar/gpbobo

## box_of_shampoo (test target bclpiq)
Same-category pool: bclpiq [g1], nijidu [g2], jccyom [g3], tzghev [g4]
Other-category pool: box_of_cereal/iucshm, box_of_rice/pstqvm, box_of_cane_sugar/gpbobo, box_of_baking_mix/fmewsz, box_of_crackers/cmdigf, box_of_coffee/cjtadw, box_of_granola_bars/tqwhvz, box_of_tissues/uglbjc, box_of_sanitary_napkins/tkrzkb, bottle_of_shampoo/stjjjm, shampoo_dispenser/eblcnw, soap_dispenser/zwudcz, box_of_whiskey/ftapxo, box_of_rum/lnjbjj, box_of_champagne/rjsibn, box_of_cereal/ykuyqb

## box_of_tissues (test target uglbjc)
Same-category pool: uglbjc [g1], zutrxn [g2], ntbrtz [g3], xwstls [g4]
Other-category pool: box_of_sanitary_napkins/tkrzkb, tissue_dispenser/frdbld, paper_liners/yvuilk, bag_of_cookies/bbyvsc, bag_of_jerky/wblype, bag_of_ice_cream/himfgn, bag_of_cream_cheese/dxrsol, box_of_cookies/apjokz, box_of_candy/gvreky, box_of_coffee/cjtadw, sponge/aewrov, box_of_butter/oixapu, box_of_baking_mix/fmewsz, paper_bag/bzsxgw, bag_of_snacks/pkwgid, box_of_sanitary_napkins/acaivd

## box_of_yogurt (test target jkemqc)
Same-category pool: jkemqc [g1], znhjgm [g2], jtjcrx [g3]
Other-category pool: cup_of_yogurt/kihdsj, cupcake/pfwrlq, teacup/cpozxi, coffee_cup/ckkwmj, paper_cup/guobeq, soda_cup/fsfsas, beaker/jdijek, mug/ycbmug, jar_of_jam/vxjbjf, jar_of_mayonnaise/ktmjvg, jar_of_puree/raagvm, jar_of_bath_salt/dtrjxi, bottle_of_face_cream/dztaed, bottle_of_sour_cream/pmawft, can_of_corn/alphlq, can_of_baking_mix/blrqqz, canned_food/acgdtc

## can_of_soda (test target opivig)
Same-category pool: opivig [g1], itolcg [g2], bfrzvk [g3], evcxlr [g4], frewxk [g5], lugwcz [g6], wbrrad [g7], xmjfcg [g8], ttxyui [g9]
Other-category pool: can_of_icetea/ifrjsc, can_of_beans/kclbuu, canned_food/acgdtc, can_of_tomatoes/ckdouu, can_of_tomato_paste/sqqdzb, can_of_sardines/dpgmry, can_of_corn/alphlq, can_of_oatmeal/qyukhm, can_of_baking_mix/blrqqz, spray_can/bgalkx, can_of_beans/ojqgjz

## carton_of_milk (test target kszoro)
Same-category pool: kszoro [g1]
Other-category pool: -
Same-category candidates dropped: carton_of_milk/atyqub (no sampler result); carton_of_milk/xmugpm (no sampler result); carton_of_milk/kklgxk (no sampler result)

## cellulose_tape (test target kavsnx)
Same-category pool: kavsnx [g1], gchdhk [g2], sklkyc [g3]
Other-category pool: coaster/arjpcz, cookie_cutter/jpscvj, cork/lseuwf, watch/ghobzn, sugar_cookie/cgkiti, butter_cookie/kukrla, chocolate_chip_cookie/ggpang, biscuit/ukcwqw, cellulose_tape_dispenser/budhaz, peppermint/pbrxfe, hamburger_bun/fqxumh, cookie_cutter/lqrfzo, cap/ciopwh, cork/ncxgpe, sugar_cookie/hlyeaq, chocolate_chip_cookie/oyhoth, cellulose_tape_dispenser/fetnry

## chili (test target agecro)
Look-alike groups (init view): {pbbkpz, xhbpqh} - Both 4 cm curved red chilies with green stem; at ~100 px they are the same red blob at every yaw
Same-category pool: agecro [g1], rafkbt [g2]
Other-category pool: carrot/aucrah, cucumber/wcvwye, pickle/gytwes, green_bean/dyjvwv, asparagus/eodozo, candy_cane/qfomjj, cigar/kslnuf, radish/bpmjki, date/kdfasq, banana/vvyyyv, celery/rqvtqi, bacon/eclhdh, pickle/ifterk, green_bean/lvbjws, hotdog_frank/qijvmi, asparagus/npggjn, radish/vztgpr, date/lskefl
Same-category candidates dropped: chili/pbbkpz (Segmentation 0 px at yaw 0 though visible; 4 cm chili renders as a few-pixel red blob); chili/xhbpqh (no sampler result)

## conditioner_dispenser (test target njqnuo)
Look-alike groups (init view): {udeflt, uncljo} - both plain white upright pump bottles, red ball pump, same 19 cm height, no print; udeflt only somewhat slimmer with softer shoulder
Same-category pool: njqnuo [g1], uncljo [g2], udeflt [g2]
Other-category pool: soap_dispenser/zwudcz, shampoo_dispenser/eblcnw, bottle_of_shampoo/stjjjm, conditioner_atomizer/quqgcu, bottle_of_lotion/tkryrh, tube_of_lotion/ixfuxm, bottle_of_dish_soap/bnmixt, spray_bottle/asztxi, water_atomizer/lfarai, detergent_atomizer/rhfpsb, deodorant_atomizer/utjfug, bottle_of_perfume/ipurzb, bottle_of_cologne/lyipur, deodorant_stick/albeyc, soap_dispenser/cgxrwd, shampoo_dispenser/eywylh, bottle_of_shampoo/dvrzmy, soap_dispenser/edinmg

## cookie_cutter (test target jpscvj)
Same-category pool: jpscvj [g1], lqrfzo [g2], fvxiun [g3]
Other-category pool: cellulose_tape/kavsnx, coaster/arjpcz, cork/lseuwf, sugar_cookie/cgkiti, butter_cookie/kukrla, chocolate_chip_cookie/ggpang, biscuit/ukcwqw, watch/ghobzn, cupcake/pfwrlq, toy_dice/eqwazv, hamburger_bun/fqxumh, cellulose_tape/gchdhk, cap/ciopwh, cork/ncxgpe, sugar_cookie/hlyeaq, chocolate_chip_cookie/oyhoth, toy_dice/hbkqqp

## jar_of_bath_salt (test target dtrjxi)
Look-alike groups (init view): {dtrjxi, erglwp} - Same 7x7x7 square beige jar with black round lid, same pixel count; only a subtle beige-vs-cream body shade differs, invisible at this size.
Same-category pool: dtrjxi [g1], erglwp [g1]
Other-category pool: jar_of_jam/vxjbjf, jar_of_mayonnaise/ktmjvg, jar_of_puree/raagvm, jar_of_cumin/tsktnz, jar_of_curry_powder/ozhftu, jar_of_dill_seed/reavkt, jar_of_orange_sauce/xfxfig, jar_of_spaghetti_sauce/efrxze, jar_of_sugar/bholak, jar_of_clove/dlvall, jar_of_tumeric/miivhi, bottle_of_black_pepper/ejtiig, bottle_of_paprika/nrdgrp, bottle_of_sage/bterim, bottle_of_seasoning/vqwqqv, bottle_of_ground_nutmeg/qebruq, bottle_of_ground_cloves/vzamzb, bottle_of_ground_mace/lgpxro, bottle_of_mustard_seeds/grryaf

## mouse (test target cduwda)
Look-alike groups (init view): {cduwda, pljjtu} - Identical white 12x7x4 mice: same domed shape, shading and small dark sensor dot in every crop.; {fjljxp, ihpnjx} - Exact geometry duplicates; identical white flat egg-shaped mice in every crop.
Same-category pool: cduwda [g1], wwyopp [g2], pljjtu [g1], fjljxp [g3], eosngb [g4], ihpnjx [g3]
Other-category pool: webcam/fabztm, calculator/kwmmty, stapler/kgnsas, bar_soap/ofargb, potato/bpwohr, wallet/lagjae, sunglasses/gqawld, watch/ghobzn, charger/bapkyh, keys/olymng, lock/pngvzn, soap_dish/gnndfy, bar_soap/feqemg, potato/ljfitc, sunglasses/nxfifs, soap_dish/ownvqj
Same-category candidates dropped: mouse/oydobn (3x2x1 cm, 273 px dark blob; does not read as a mouse.)

## mug (test target ycbmug)
Look-alike groups (init view): {waqrdy, yxaapv} - Both tall flared light-grey latte mugs with small handle, same size and colour; only slight base taper differs; {ehnmxj, fapsrj} - Geometry check proved exact duplicates; identical squat light-grey mug with D handle at every yaw
Same-category pool: ycbmug [g1], ppzttc [g2], waqrdy [g3], ehnmxj [g4], fapsrj [g4], ntgftr [g5], ppdqbj [g6], yxaapv [g3], jgethp [g7], lgxhsc [g8]
Other-category pool: coffee_cup/ckkwmj, teacup/cpozxi, beaker/jdijek, water_glass/cdteyb, soda_cup/fsfsas, paper_cup/guobeq, beer_glass/lymciz, goblet/nawrfs, coffee_cup/dkxddg, teacup/oxfzfe, beaker/fyrkzs, water_glass/edfzlt

## notebook (test target umxsmd)
Look-alike groups (init view): {dlhxxl, zkpswz} - Same 12x9x2 cm black hardcover with elastic band and white page block; identical appearance at all four yaws
Same-category pool: umxsmd [g1], yibctu [g2], ksxknp [g3], dlhxxl [g4], zkpswz [g4], zkmmnf [g5], aanuhi [g6], eijrqw [g7], ffdjzp [g8], gqkaqu [g9], jxabmj [g10]
Other-category pool: hardback/acaxtl, paperback_book/xxknda, magazine/ajitiv, folder/lktggf, notepad/oaikde, board_game/eoywpy, video_game/ceaxsd, hardback/aceozs, magazine/tfovwy, notepad/rvevdj
Same-category candidates dropped: notebook/xdnswn (Label reports 0 segmentation px at yaw 0 although rendered; near-untextured dark-grey slab)

## pack_of_cigarettes (test target ppbmak)
Same-category pool: ppbmak [g1], ghbqav [g2], ipmejr [g3], cfmlhf [g4], gzxueq [g5], qldhnm [g6], ajphvf [g7], dthkix [g8], hmhqkq [g9], qabdgg [g10]
Other-category pool: match_box/xpnfjz, lighter/zsxfjz, pack_of_chocolate_bar/bxgewr, chocolate_bar/amutpr, granola_bar/iurfsq, video_game/ceaxsd, wallet/lagjae, case_of_eyeshadow/zgervc, cigar/kslnuf, bar_soap/ofargb

## pencil_case (test target otcmih)
Same-category pool: otcmih [g1], cqxkgl [g2], acikzo [g3]
Other-category pool: pencil_holder/jyetcl, wallet/lagjae, sunglasses/gqawld, tube_of_toothpaste/vsclsj, granola_bar/iurfsq, chocolate_bar/amutpr, pack_of_chocolate_bar/bxgewr, pencil_holder/jkuiqg, sunglasses/nxfifs, chocolate_bar/bcfudr, pack_of_chocolate_bar/tidcst, sunglasses/qiwncl, chocolate_bar/dxnisi, pack_of_chocolate_bar/wgrkhn, box_of_cookies/apjokz, bar_soap/ofargb, video_game/ceaxsd

## pencil_holder (test target jyetcl)
Look-alike groups (init view): {jyetcl, taixcy} - Same asset (exact dup) in two yaws: tall tan wooden rectangular open box, identical wood grain and proportions; {jkuiqg, mrrpay} - Same asset (exact dup): squat plain grey metal-look open box, identical shade and proportions at every yaw
Same-category pool: jyetcl [g1]
Other-category pool: -
Same-category candidates dropped: pencil_holder/taixcy (no sampler result); pencil_holder/mrrpay (no sampler result); pencil_holder/jkuiqg (no sampler result)

## soap_dispenser (test target zwudcz)
Same-category pool: zwudcz [g1], cgxrwd [g2], edinmg [g3], sinntg [g4], ijxawq [g5], jmulpc [g6]
Other-category pool: shampoo_dispenser/eblcnw, bottle_of_shampoo/stjjjm, bottle_of_lotion/tkryrh, tube_of_lotion/ixfuxm, bottle_of_dish_soap/bnmixt, spray_bottle/asztxi, conditioner_atomizer/quqgcu, water_atomizer/lfarai, detergent_atomizer/rhfpsb, deodorant_atomizer/utjfug, bleaching_agent_atomizer/arxjfa, conditioner_dispenser/uncljo, shampoo_dispenser/eywylh, bottle_of_shampoo/dvrzmy

## tablefork (test target hktwlm)
Look-alike groups (init view): {flexrc, pqrtkc} - Identical pale matte-silver 20 cm forks: same slender flared handle, same four-tine head, same size and shading at every yaw.
Same-category pool: hktwlm [g1], yuihyg [g2], flexrc [g3], pqrtkc [g3], uodzrq [g4]
Other-category pool: teaspoon/qdepdj, tablespoon/pltxjb, table_knife/eebpap, tongs/gbjqyn, toothbrush/ifxlss, marker/ablbdc, teaspoon/piqabk, tablespoon/tqqwys, table_knife/jxdfyy, chopstick/bcwuwv, tongs/igsklx, toothbrush/nbgpwg, pen/gqwdor, pencil/bfkwhb, marker/bwwkmb, teaspoon/nxddfp

## teaspoon (test target qdepdj)
Look-alike groups (init view): {fgeimy, nxddfp, prbhag, qdepdj} - Same ~13 cm plain silver teaspoon, thin handle and round bowl, identical proportions at every yaw; fgeimy differs only by a slightly shinier finish.
Same-category pool: qdepdj [g1], piqabk [g2], nxddfp [g1], prbhag [g1], vgbcmo [g4], eilfij [g5], fgeimy [g1], mgdfcz [g6]
Other-category pool: tablespoon/pltxjb, tablefork/hktwlm, table_knife/eebpap, toothbrush/ifxlss, marker/ablbdc, tablespoon/tqqwys, tablefork/yuihyg, table_knife/jxdfyy, chopstick/bcwuwv, toothbrush/nbgpwg, pen/gqwdor, pencil/bfkwhb, marker/bwwkmb, tablespoon/ubuhoz, tablefork/flexrc
Same-category candidates dropped: teaspoon/bjzgmj (UNSTABLE, tilt 151 deg: flipped over on the tray; also 25 cm serving-spoon size.)

## tortilla_chips (test target aiohfb)
Look-alike groups (init view): {aiohfb, herjym} - Same 7x6 cm yellow speckled saddle chip: identical colour, curl and size; every yaw renders near-identically.
Same-category pool: aiohfb [g1], herjym [g1], dlkrgq [g3], kdjrni [g4], yreyjs [g5]
Other-category pool: chip/obgeiz, biscuit/ukcwqw, sugar_cookie/cgkiti, butter_cookie/kukrla, chocolate_chip_cookie/ggpang, cheese_danish/hkdtwp, pack_of_chocolate_bar/bxgewr, chocolate_bar/amutpr, granola_bar/iurfsq, peppermint/pbrxfe, peppermint_candy/dqpiet, candy_cane/qfomjj, bread_slice/pfggnm, hamburger_bun/fqxumh, bacon/eclhdh, scone/gwdhyj
Same-category candidates dropped: tortilla_chips/mhbnsy (Large translucent ghost shell surrounds the chip at every yaw (alpha/geometry artifact).)

## toy_dice (test target eqwazv)
Look-alike groups (init view): {akguod, bjrxkj, cmqtch, eqwazv, gcevml, hbkqqp, ievnsq, kmagva, krpdji, mqvaaa, nmhsvp, rbzyrv, wyxlhc, xfqqhh} - Identical 4 cm wooden alphabet blocks; coloured letter frame hides at yaws 0/180 leaving only plain engraved wood; hbkqqp=kmagva=xfqqhh and cmqtch=mqvaaa exact dups
Same-category pool: eqwazv [g1], hbkqqp [g1], ievnsq [g1], bjrxkj [g1], cmqtch [g1], gcevml [g1], kmagva [g1], krpdji [g1], mqvaaa [g1], rbzyrv [g1], xfqqhh [g1], dupdoy [g2], haqumx [g3], meyqyk [g4], nmhsvp [g1], pzbapa [g5], qtcfxi [g6], wyxlhc [g1], zublep [g7]
Other-category pool: toy_box/kvithq, match_box/xpnfjz, box_of_yogurt/jkemqc, jar_of_bath_salt/dtrjxi, can_of_baking_mix/blrqqz, bar_soap/ofargb, cellulose_tape/kavsnx, cookie_cutter/jpscvj, video_game/ceaxsd, box_of_tissues/uglbjc, box_of_butter/oixapu, keys/olymng, lock/pngvzn
Same-category candidates dropped: toy_dice/akguod (large translucent yellow stray geometry around the block at every yaw (broken mesh))

## video_game (test target ceaxsd)
Same-category pool: ceaxsd [g1], ppnzqm [g2], coiafo [g3], lfdpse [g4], yyvkel [g5], apjpos [g6], ckugyi [g7], pujfzk [g8], quzjuo [g9], qsyyql [g10]
Other-category pool: board_game/eoywpy, hardback/acaxtl, pack_of_cigarettes/ppbmak, match_box/xpnfjz, wallet/lagjae, case_of_eyeshadow/zgervc, calculator/kwmmty, chocolate_bar/amutpr, pack_of_chocolate_bar/bxgewr, board_game/frgdqh
