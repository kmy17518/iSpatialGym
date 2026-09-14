# Test instances: two models per (task, category)

Selected by parallel reviewer agents under `select_contract.md` (score = 2*G + 2*V + T + R, max 25; G grasp quality, V visual quality, T task fit, R robustness). 175 of 175 groups have a selection; 149 groups have two picks, 19 have one, 7 have none.

## object_scaling

| round | relation | category | cands | pick 1 | score | pick 2 | score | alternate | conf | warnings |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  | apple | 7 | omzprq | 25 | agveuv | 25 | obixxh | high |  |
| 1 |  | beaker | 10 | rhohgs | 21 | zycgen | 21 | fyrkzs | medium | jdijek, rhohgs, uobdoq, zycgen and fyrkzs share bbox 78x78x81 and are the same tapered-tumbler mesh with different mater |
| 1 |  | beeswax_candle | 5 | nhdnje | 19 | oimgmh | 17 | ouzkdj | medium | Every candidate offers only one grasp pose (horizontal mid-body diameter pinch at tilt ~90), so G is capped at 3 group-w |
| 1 |  | bell_pepper | 5 | ukkycp | 25 | ggurxn | 25 | ihctxa | high |  |
| 1 |  | bottle_of_beer | 12 | gxxbhh | 25 | kqskmv | 21 | fgzjnb | high |  |
| 1 |  | bottle_of_black_pepper | 4 | honise | 23 | zybfok | 17 | ejtiig | medium | ejtiig, ydzzrv and zybfok look like the same squat black jar; their 50/12 closures (gap ~62 mm) match the lid diameter,  |
| 1 |  | bottle_of_catsup | 5 | hvxkso | 23 | dmyfdf | 21 | ialodu | medium | All candidates except bcqfxb rely on one-finger closures (object pressed against the open pad); pose diversity is limite |
| 1 |  | bottle_of_essential_oil | 5 | wansva | 25 | eyyhld | 25 | xvqshn | high | All five candidates are the same 22x22x60 mm mesh with different label colours and tie at 25; ranking among them is by p |
| 1 |  | bottle_of_medicine | 8 | kqkwoq | 24 | zfbnjh | 24 | egondf | high | hvocpc is a medicine box, not a bottle (V=2). Reported width 24-27 mm is wrong for the ~48 mm bottles (body fills the pa |
| 1 |  | bottle_of_pop | 12 | jnoksl | 19 | ubazru | 19 | oghrnk | medium | Every candidate offers only one grasp pose (horizontal mid-body pinch, tilt ~90, CoM offset <= 10 mm); in-hand transform |
| 1 |  | bottle_of_water | 12 | sxffiv | 23 | atvnqy | 23 | migvlt | high |  |
| 1 |  | bottle_of_wine | 12 | nhmtvt | 23 | hnkiog | 21 | bmudli | medium | bmudli ties nhmtvt at 23 but is another dark cork bottle with white text; pair rule preferred the different-looking hnki |
| 1 |  | can_of_soda | 9 | opivig | 21 | itolcg | 19 | bfrzvk | medium | All candidates except lugwcz offer only one grasp pose type (horizontal diameter pinch), so G is capped at 3 for the flo |
| 1 |  | canned_food | 7 | ycbspm | 22 | byakrm | 19 | qhgdys | medium | Category is mixed: foetdd is a pasta box and cbmndg/pkopdw are 100 mm jars wider than the gripper opening. | pkopdw has  |
| 1 |  | ice_cube | 8 | xupugj | 23 | pjgror | 23 | dueotq | high | All eight candidates are near-identical ~21 mm glassy cubes; the pair cannot span different appearances, so ranking is b |
| 1 |  | jar_of_spaghetti_sauce | 6 | zahyds | 23 | xpasvb | 20 | exmsmm | medium | All 85 mm jars (yqnmoy, fyocnu, xpasvb, efrxze, exmsmm) render the glass body as a black mirror-like surface and nearly  |
| 1 |  | lemon | 12 | bmvxas | 23 | vuwehu | 21 | dsacfa | medium | Ten candidates are the same textured low-poly lemon asset at 61-73 mm and tie at 23; #1 chosen by pinch symmetry and rep |
| 1 |  | potato | 5 | lqjear | 25 | bpwohr | 23 | ljfitc | medium | ljfitc ties lqjear at 25 but is a near-identical yellow oval potato, so it was passed over for #2 in favour of a differe |
| 1 |  | soda_cup | 4 | fsfsas | 25 | vicaqs | 21 | lpanoc | high | gnzegv has only one valid grasp and an untextured white asset (G=1, V=2); lpanoc is a matte duplicate of the fsfsas shap |
| 1 |  | wineglass | 12 | oadvet | 21 | yfzibn | 21 | vxmzmq | medium | ezsdil is not selectable: all five rows pinch the fragile rim lip (n_clean 0, confirmed visually). | No candidate offers |
| 2 |  | bar_soap | 4 | feqemg | 19 | ozifwa | 19 | ofargb | medium | Every candidate repeats a single pinch pose (same contact region and approach), so grasp diversity is capped at G=3 for  |
| 2 |  | box_of_cereal | 12 | wrlalk | 23 | hosfhj | 23 | yorray | medium | Seven candidates tie at 23: every model offers five clean pinches but only front/back face flips of one approach, so G i |
| 2 |  | cap | 12 | rgtedj | 23 | gmdwwe | 23 | gxkbcd | medium | Four candidates tie at 23 (rgtedj, mgirzi, gxkbcd, gmdwwe); tie broken by validity rate, pose spread and appearance | gm |
| 2 |  | cherry_tomato | 5 | dqjuub | 23 | eukpfb | 23 | dqvtgm | medium | All grasps are 19-36/0 one-finger fingertip presses on 19-34 mm objects; four of five candidates are oval grape-style to |
| 2 |  | hard_candy | 5 | qptjgv | 25 | yuatcg | 25 | ojznyf | medium | All candidates share one low-poly wrapped-candy mesh style (~21 mm body, 55 mm with twists) and differ mainly in wrapper |
| 2 |  | lid | 12 | wtjthb | 19 | niarsj | 18 | egfobe | medium | All lids offer only one grasp pose (rim pinch, hangs vertically), so G is capped at 3 for every candidate; in-hand trans |
| 2 |  | marker | 11 | rsxbso | 25 | hmlofs | 25 | zvvovh | high | Picks #1 and #2 are near-identical white whiteboard markers (black cap, orange text); if appearance spread is wanted, al |
| 2 |  | pack_of_cigarettes | 12 | kienyo | 21 | ajphvf | 21 | ohpqua | medium | Group splits into two mesh families: 125x55x80 chunky boxes (odd depth, banded side texture) and 110x55x16 slim packs (p |
| 2 |  | shampoo_dispenser | 7 | thurmw | 23 | eblcnw | 21 | zoyyia | medium | zoyyia ties #1 on score (23) but is the same cube shape in black; eblcnw (21) taken as #2 for shape diversity and rotati |
| 2 |  | toy_dice | 6 | eqwazv | 25 | wyxlhc | 25 | hbkqqp | high | All six candidates are 41 mm wooden alphabet blocks (letter/animal/number faces), not pipped dice; every row is a 43/0 a |
| 3 |  | beefsteak_tomato | 4 | pnrdxh | 21 | altlfz | 21 | ogpans | medium | Three of four assets are visibly low-poly (faceted); altlfz shows no calyx and reads as a plain orange sphere at wrist-c |
| 3 |  | bottle_of_olive_oil | 4 | cqycjk | 23 | wztvie | 21 | luikop | medium | wztvie offers only one repeated pose with edge contact on its square body; cqycjk's label text is not olive oil, though  |
| 3 |  | bottle_of_sake | 4 | swlykk | 25 | vfxfuj | 19 | utwglw | medium | utwglw, vfxfuj and pwtebq are held only by the neck in one repeated pose; vfxfuj's identity as sake is weaker (wine-styl |
| 3 |  | bottle_of_salsa | 3 | dvxotd | 11 | nafwlf | 11 | mavope | low | No candidate has a clean grasp: every row is an end/rim pinch on the 250 mm bottle with the body hanging 53-100 mm from  |
| 3 |  | orange | 2 | ucstpm | 23 | nlygru | 18 |  | medium | Group has only 2 candidates; both selected. | nlygru: pinch width 81-83 mm is already in the borderline band; scaling th |
| 3 |  | pear | 3 | ngtwvy | 24 | clvnqe | 23 | vutulb | high |  |
| 3 |  | pine_cone | 3 | bmvhss | 25 | vbqhas | 24 | fmskdx | medium | vbqhas: every grasp is a one-finger closure with the moving finger at 1-6 mm; if that artefact is unacceptable, substitu |
| 3 |  | shiitake | 3 | lrejyh | 24 | oirawt | 22 | iwrtus | medium | Small category (47-52 mm): one finger bottoms out at 0 in every row of every candidate; grasps hold only at the fingerti |
| 3 |  | spray_can | 3 | buxqkr | 18 | bgalkx | 15 | tehmzy | low | bgalkx has visible geometry defects (glossy red/green hemispheres protruding from the body at the pinch line) and an unt |
| 3 |  | vidalia_onion | 3 | ldeuli | 23 | buyxll | 21 | dejqhf | medium | Two of three candidates (buyxll, dejqhf) are red onions, not Vidalia-coloured; the pair therefore mixes a yellow and a r |

## mental_rotation

| round | relation | category | cands | pick 1 | score | pick 2 | score | alternate | conf | warnings |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  | banana | 2 | znakxm | 22 | vvyyyv | 20 |  | medium | Group has only 2 candidates; both selected. | Both models: the closing finger reaches 0 mm on every row (banana pressed  |
| 1 |  | carving_knife | 5 | alekva | 23 | fqqbop | 21 | zaihmr | medium | fqqbop is serrated and gixmnd has a wide cleaver-like blade; both are less typical carving-knife shapes. |
| 1 |  | cellulose_tape_dispenser | 3 | fetnry | 23 | budhaz | 23 | yyekns | high | All three candidates share one mesh (bbox 112x50x68 mm); the pair differs only by colour. | Every row is an off-centre 5 |
| 1 |  | coffee_cup | 11 | dkxddg | 23 | ykuftq | 23 | rypdvd | high |  |
| 1 |  | crab | 5 | ohjsdr | 23 | tjkiva | 20 | eqfklx | medium | tjkiva pinches are on 13-15 mm leg segments (R=1) and the mesh is low-poly; eqfklx is the safer but look-alike fallback. |
| 1 |  | hammer | 2 | fcrlbo | 20 | bbzlel | 18 |  | medium | Group has only 2 candidates; both selected. | bbzlel is a large rubber mallet (341 mm) rather than a canonical hammer an |
| 1 |  | hook | 5 | hewhtl | 19 | yfikis | 19 | ddbsdk | medium | Every hook is pinched on a 6-12 mm feature, so robustness is marginal group-wide; yfikis is tiny (44x66 mm) and may be h |
| 1 |  | mouse | 7 | cduwda | 21 | fjljxp | 21 | pljjtu | medium | Grasp diversity is moderate for all conventional mice: every row pinches across the height, varying only position and si |
| 1 |  | mug | 10 | ppzttc | 22 | lgxhsc | 20 | waqrdy | medium | 7 of 10 candidates (fapsrj, ehnmxj, ycbmug, jgethp, ppdqbj, yxaapv, ntgftr) offer only rim pinches from above on a 3-11  |
| 1 |  | soup_ladle | 5 | cfzneb | 23 | hgvzdm | 21 | rqnufc | high | hgvzdm is a uniform flat-grey material (plausible silicone ladle, but no texture detail); gxsrhh and tmdqpb have collisi |
| 1 |  | spatula | 9 | hkvsep | 23 | pdqgzq | 20 | ymvrct | medium | Several assets are monochrome flat grey (ktkigm, pdqgzq, pxwpuw); scored V=3, not excluded. | Many rows across the group |
| 1 |  | statue | 8 | tuldri | 24 | wrtqik | 22 | oxgpfk | high | cxsvwj, noykhv, ponfvo and qbfkek have no clean grasp (all end/protrusion pinches 80-150 mm from CoM) and are not select |
| 1 |  | tablefork | 4 | hktwlm | 21 | flexrc | 19 | pqrtkc | medium | every fork is 3-8 mm thick at the pinch so R is 0 for all candidates; fingers close to 0 on one side in every row | flex |
| 1 |  | teacup | 6 | zdvgol | 20 | oxfzfe | 19 | vckahe | medium | All candidates except zdvgol are held only by 3-11 mm rim/shell pinches (R 0-1); zdvgol and cpozxi are handle-less and n |
| 1 |  | teaspoon | 8 | eilfij | 21 | prbhag | 19 | vgbcmo | medium | Every candidate is pinched across 3-7 mm of spoon thickness with fingers nearly closed, so R=0 group-wide; eilfij is tab |
| 1 |  | tongs | 3 | rcpjld | 22 | igsklx | 20 | gbjqyn | medium | igsklx and gbjqyn render as plain light grey - materials may be missing. |
| 1 |  | toy_figure | 12 | zuntus | 25 | vcmcoi | 25 | gnofgv | high |  |
| 1 |  | vegetable_peeler | 2 | njgqao | 20 | wqtpjz | 15 |  | medium | Group has only 2 candidates; both selected. | wqtpjz: weak grasp set - row 2 pinches the 4 mm blade arm, rows 3-4 clamp  |
| 1 |  | walker | 12 | glrsok | 25 | txpihh | 24 | scbatq | high |  |
| 2 |  | bell_pepper | 5 | ukkycp | 23 | ihctxa | 23 | ggurxn | medium | All rows are 50/x one-finger closures (assisted grasp); ihctxa width metric reflects a hollow shell, pads still span the |
| 2 |  | briefcase | 3 | uquevd | 21 | lwbctq | 16 | psvdau | medium | lwbctq (#2) has only 2 clean rows, both similar top-body pinches; the three handle rows stop on a collision hull far out |
| 2 |  | cookie_cutter | 3 | jpscvj | 22 | fvxiun | 20 | lqrfzo | high |  |
| 2 |  | garlic_clove | 10 | xotobt | 23 | zpybkr | 21 | zpzrac | medium | hpzvmi, faeyay, flcbrn, jwhnfc: collision hull far smaller than the visual mesh, fingers close through the clove in ever |
| 2 |  | pickle | 5 | ifterk | 23 | iobrqp | 23 | vuspsy | medium | All pickles share the same low-poly family with faceted ends; straight pickles are near-symmetric about their long axis, |
| 2 |  | potato | 5 | lqjear | 25 | bpwohr | 23 | ljfitc | high | Potatoes are only weakly asymmetric; ngjeog is near-spherical and unsuitable for reading orientation. Strip headers say  |
| 2 |  | quartz | 6 | klgses | 25 | kilkpz | 23 | vfuilh | high |  |
| 2 |  | sandal | 9 | uhzfis | 25 | zafskl | 25 | chgcrm | high | tkfdsk: no clean row seen (every pinch is sole-plus-strap with 34-38 mm opening on a 13 mm sole); not selectable. | txki |
| 2 |  | sunflower | 12 | ysogni | 22 | zdmbjt | 20 | vkgdsl | medium | vkgdsl and zdmbjt share bbox 223x90x77 and look identical (same mesh, different grasp sets); pibfxd and mdpoct likewise  |
| 2 |  | webcam | 1 | fabztm | 19 |  |  |  | medium | Only 1 candidate in the group; fewer than 2 models selectable. | Rows 1, 2 and 5 pinch across the protruding lens dome ( |
| 3 |  | baby_shoe | 2 | tnbjlw | 21 | jzwnri | 18 |  | medium | Group has only 2 candidates; both selected. | Both models are the same pink star sneaker design with identical bbox (127 |
| 3 |  | cleaver | 1 | vpmydv | 18 |  |  |  | medium | Only 1 candidate in the group; fewer than 2 models selectable. | vpmydv: row 5 is not clean (33 mm opening on an 11 mm s |
| 3 |  | drill | 1 | nzgmza | 24 |  |  |  | high | Only 1 candidate in the group; fewer than 2 models selectable. | Rows 4-5 grasp the chuck and rear housing 76-88 mm from |
| 3 |  | gaming_controller | 1 | yofqbq | 24 |  |  |  | high | Only 1 candidate in the group; fewer than 2 models selectable. | Rows 2-5 all pinch one grip 59-63 mm from CoM; pose div |
| 3 |  | ginger_root | 2 | wqcasl | 22 | izvukv | 18 |  | medium | Group has only 2 candidates; both selected. | Both models: one finger closes fully to 0 mm on every row (small object pr |
| 3 |  | high_heel | 2 | omckzr | 20 | htqifz | 20 |  | medium | Group has only 2 candidates; both selected. | Both models appear to be the same black strappy heel (identical bbox 197 x |
| 3 |  | mallet | 1 | wpswgw | 18 |  |  |  | medium | Only 1 candidate in the group; fewer than 2 models selectable. | wpswgw: row 2 is not clean (pads stop 28 mm apart on a  |
| 3 |  | stapler | 1 | kgnsas | 16 |  |  |  | medium | Only 1 candidate in the group; fewer than 2 models selectable. | kgnsas is a mini stapler (87 x 18 x 45 mm) modelled in  |
| 3 |  | tablespoon | 3 | tqqwys | 20 | pltxjb | 18 | ubuhoz | medium | All spoons are thin at the pinch (4-11 mm); pltxjb appears untextured (flat white) - verify material before use. |
| 3 |  | toothbrush | 4 | vkrjps | 22 | qbkrzg | 18 | nbgpwg | medium | All toothbrushes are 4-11 mm thin at the pinch; robustness is borderline or marginal for the whole category. |

## disembedding

| round | relation | category | cands | pick 1 | score | pick 2 | score | alternate | conf | warnings |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  | bar_soap | 4 | ofargb | 22 | feqemg | 22 | ozifwa | medium | Both picks pinch at 85-87 mm, close to the 97 mm gripper opening (one finger stays at the open limit), so width margin i |
| 1 |  | box_of_baking_mix | 8 | fmewsz | 24 | lgusnr | 24 | uhvkxj | high | All rows in every candidate are 50/0 one-finger closures (closing finger at its limit), so R=1 across the group; the obj |
| 1 |  | box_of_cane_sugar | 4 | gpbobo | 25 | qvjiti | 25 | cqlofx | high | rvsivw is a plain unprinted soft bag rather than a box (V 3), 89 mm wide nearly fills the gripper | cqlofx rows close th |
| 1 |  | box_of_cereal | 12 | iucshm | 25 | ykuyqb | 25 | surzft | medium | 11 of 12 candidates tie at 25 under the rubric; pair chosen for grasp centring and appearance spread (large classic box  |
| 1 |  | box_of_cookies | 11 | apjokz | 23 | dlhqft | 22 | nirfva | high |  |
| 1 |  | box_of_sanitary_napkins | 9 | tkrzkb | 23 | acaivd | 23 | zqinib | medium | Seven candidates tie at 23 with the same 70 mm side pinch (50/21-23); #1 chosen for most central grasps and sharpest pri |
| 1 |  | box_of_shampoo | 4 | bclpiq | 25 | nijidu | 25 | jccyom | medium | every grasp in this group is a one-sided top-edge pinch (50/5-6) with the box pressed against the open-limit pad; both p |
| 1 |  | box_of_tissues | 4 | uglbjc | 23 | zutrxn | 21 | ntbrtz | medium | all four candidates share one soft-pack mesh (not a rigid box) with plain white side faces and low-resolution prints | n |
| 1 |  | box_of_yogurt | 2 | jkemqc | 23 | znhjgm | 21 |  | medium | Group has only 2 candidates; both selected. | Both models appear to share the same cup mesh (bbox 64-65 x 64 x 54 mm); t |
| 1 |  | carton_of_milk | 4 | kszoro | 22 | atyqub | 22 | xmugpm | medium | all three gable cartons are 84-86 mm wide at the pinch (borderline R 1) and every row rests one finger at the open limit |
| 1 |  | cellulose_tape | 3 | kavsnx | 23 | gchdhk | 23 | sklkyc | medium |  |
| 1 |  | chili | 4 | agecro | 24 | rafkbt | 24 | pbbkpz | high | xhbpqh and pbbkpz are 39-43 mm long and 6-9 mm thick: marginal pinch width (R 0), size outliers, nearly invisible in pro |
| 1 |  | cookie_cutter | 3 | jpscvj | 23 | lqrfzo | 21 | fvxiun | medium |  |
| 1 |  | notebook | 6 | umxsmd | 24 | yibctu | 22 | ksxknp | medium | All lying notebooks are held by width squeezes with one finger idle at the open limit (2-3 mm nominal slack); dlhxxl/zkp |
| 1 |  | pack_of_cigarettes | 12 | ppbmak | 25 | ghbqav | 23 | ipmejr | medium | Most candidates' top-ranked row is a one-finger (50/x) push with the pack ~20 mm off the gripper centre; only ppbmak, gh |
| 1 |  | pencil_case | 3 | otcmih | 24 | cqxkgl | 24 | acikzo | high | All three are near-circular cylinders lying on their side; they can roll slightly before settling (T scored 2 rather tha |
| 1 |  | tortilla_chips | 3 | aiohfb | 19 | mhbnsy | 17 |  | medium | Both selected chips are held by a 4-10 mm edge with fingers nearly fully closed (R=0); expect fragile grasps. | herjym:  |
| 1 |  | toy_dice | 12 | eqwazv | 21 | hbkqqp | 21 | ievnsq | medium | All 12 candidates are wooden alphabet blocks (letter/animal/number faces), not pipped dice; they are colour/letter varia |
| 1 |  | video_game | 12 | ceaxsd | 25 | ppnzqm | 25 | coiafo | high |  |
| 2 |  | beaker | 9 | jdijek | 23 | zycgen | 23 | fyrkzs | medium | Four candidates tie at 23 (jdijek, rhohgs, zycgen, fyrkzs); they share one tumbler mesh (78x78x81) with different materi |
| 2 |  | bottle_of_beer | 12 | meqliv | 25 | fgzjnb | 24 | zbsxro | high | phdimo is a transparent unlabeled bottle (V 2): selectable only as last resort | dcwvkg/vhscym and qepxvl/rdnopv and crf |
| 2 |  | bottle_of_medicine | 8 | zfbnjh | 25 | kqkwoq | 25 | syfpak | high | Seven candidates are the same white Jarrow bottle mesh with different labels and all score 25; ordering among them is by |
| 2 |  | bottle_of_pop | 10 | ribekf | 23 | wrgmdt | 21 | xoldze | medium | All ten candidates are low-poly with visible faceting; dark bodies render nearly flat black in the wrist camera. mdznsn  |
| 2 |  | bottle_of_wine | 12 | lacaoa | 25 | wcokej | 23 | njoeiz | high |  |
| 2 |  | can_of_soda | 9 | opivig | 25 | itolcg | 23 | bfrzvk | medium | All nine candidates close only one finger (50/13-29); the can is pushed against the open pad but stays between both pads |
| 2 |  | mouse | 7 | cduwda | 23 | wwyopp | 21 | pljjtu | high | oydobn is an unscaled 31 mm model (about quarter size) and is not selectable as a mouse test instance despite clean fing |
| 2 |  | mug | 10 | ycbmug | 23 | ppzttc | 21 | waqrdy | medium | Every rim-pinch candidate grasps a 5-10 mm wall (R 0); ycbmug is held at 50/46, only ~4 mm from full gripper opening | f |
| 2 |  | tablefork | 5 | hktwlm | 23 | yuihyg | 20 | flexrc | medium | Neck pinches of uodzrq, pqrtkc and flexrc are only 6-10 mm wide (R 0); yuihyg renders washed-out pale. |
| 2 |  | teaspoon | 7 | qdepdj | 22 | piqabk | 21 | nxddfp | medium | All teaspoons pinch 3-13 mm sections; robustness low across the category. | eilfij has no clean row (single wedged 48/0  |
| 3 |  | bottle_of_milk | 6 | mrejrs | 23 | ddvsgl | 21 | qsnnbp | medium | mrejrs closes at both joint limits (50/0) with zero margin; ddvsgl and qsnnbp textures are low-resolution; utfsmp appear |
| 3 |  | bottle_of_shampoo | 4 | stjjjm | 23 | dvrzmy | 22 | lvvkhx | high | stjjjm/lvvkhx and dvrzmy/hlkpwd are near-duplicate meshes differing only in label | tube grasps close one finger to 0 (R |
| 3 |  | box_of_butter | 2 | oixapu | 22 | mfjuil | 20 |  | medium | Both textures are low-resolution/blurry; mfjuil's blue print does not read as a butter box. | Every row closes one finge |
| 3 |  | box_of_granola_bars | 3 | tqwhvz | 25 | bqeeki | 25 | awombj | high |  |
| 3 |  | box_of_oatmeal | 2 | jtqeef | 22 | rabeel | 14 |  | medium | Group has only 2 candidates; both selected. | rabeel: only 2 valid grasps; row 2 is a corner pinch 42 mm off CoM with 33 |
| 3 |  | box_of_rice | 3 | pstqvm | 25 | fvulow | 25 | yyheuw | high |  |
| 3 |  | conditioner_dispenser | 3 | njqnuo | 22 | uncljo | 19 | udeflt | medium | njqnuo rests on a near-circular 75x78 mm side and could roll on a flat tray; uncljo/udeflt are held only by the pump hea |
| 3 |  | jar_of_bath_salt | 1 | dtrjxi | 17 |  |  |  | medium | Only 1 candidate in the group (erglwp excluded with no valid grasp); fewer than 2 models selectable. | dtrjxi: top-ranke |
| 3 |  | pencil_holder | 4 | jyetcl | 23 | taixcy | 23 | mrrpay | medium | jyetcl and taixcy are the same wooden mesh in two orientations, so the pair does not span the category's appearance | jk |
| 3 |  | soap_dispenser | 6 | zwudcz | 25 | cgxrwd | 23 | edinmg | high |  |

## configuration_matching

| round | relation | category | cands | pick 1 | score | pick 2 | score | alternate | conf | warnings | attachment counterpart(s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 |  | cap | 6 | hbafeb | 24 | xxqyrt | 21 | vnpjfn | medium | xxqyrt grasps sit at both joint limits (zero closing margin); tilt-30 grasps on vnpjfn, clhquh and yqvild let the cap sw | hbafeb -> cola_bottle/oyqdtz (not ready) [type hbafebparent] ; xxqyrt -> lemonade_bottle/yprkek [type xxqyrtparent], soda_water_bottle/upfssc [type xxqyrtparent], tomato_paste_can/krarex [type xxqyrtparent], water_bottle/lzdzkk (not ready) [type xxqyrtparent] |
| 1 |  | cork | 3 | ncxgpe | 23 | uyceta | 21 | lseuwf | medium | uyceta (#2) has a flat untextured tan material; chosen over near-identical lseuwf for appearance diversity per the pair  | ncxgpe -> wine_bottle/inkqch [type corkbottle] ; uyceta -> vinegar_bottle/hbsbwt [type uycetaparent] |
| 1 |  | dartboard | 0 |  |  |  |  |  | high | no model with an AG-valid grasp: zjenlc: no candidates |  |
| 1 |  | dip_candle | 4 | poexjo | 19 |  |  |  | medium | gbaons is a fat white pillar-shaped cylinder with no visible wick, an outlier shape for a dip candle; poexjo is the tape | poexjo -> candle_holder/wiufnv [type dipcandle] |
| 1 |  | license_plate | 0 |  |  |  |  |  | high | no model with an AG-valid grasp: nwfbxn: no candidates; naclac: no candidates; hjdaqn: no candidates; nuqnlr: no candida |  |
| 1 |  | lid | 5 | tnwisp | 17 | zkderg | 16 | qhlsdq | medium | ugfmpq not selectable: n_clean 0 and no clean row seen (fin never pinched, pads on rim). | Group is weak overall: only t | tnwisp -> teapot/mvrhya [type tnwispparent] ; zkderg -> kettle/vjbldp [type zkdergparent] |
| 1 |  | light_bulb | 3 | kfmkwd | 25 |  |  |  | high | re-selected after attachment-link check: ['sxkjea'] have no attachment meta-link and can never attach | kfmkwd -> table_lamp/ehjsdz [type bulblamp] |
| 1 |  | webcam | 1 | fabztm | 13 |  |  |  | low | Only one candidate in the group; fewer than 2 models selectable. | fabztm: collision hull is much larger than the visual | fabztm -> monitor/qvpbge [type webcamcomputer] |
| 1 |  | wreath | 1 | rkrxno | 7 |  |  |  | low | Hard-exclusion fallback: rkrxno has no clean grasp (fragile foliage pinch, antipodal 0.12, coverage 0.00) and was select | rkrxno -> recreational_vehicle/doiucd [type displaywall], wall_nail/wlnail [type displaywall] |
| 2 |  | bottle_of_water | 1 | emquat | 16 |  |  |  | low | Single candidate: all five grasps are the same shoulder pinch with the neck opening between the fingers under the palm ( | emquat -> cap/gmdwwe [type gmdwweparent, this model is the female part] |
| 2 |  | soda_cup | 0 |  |  |  |  |  | high | no model with an AG-valid grasp: vicaqs: no AG-valid grasp |  |
| 3 |  | antlers | 0 |  |  |  |  |  | high | no model with an AG-valid grasp: nlsanr: no candidates |  |
| 3 |  | blackberry | 1 | tvpfvb | 23 |  |  |  | medium | Only one candidate in the group (no excluded models either), so no second test instance exists for blackberry. | Asset r | tvpfvb -> bush/waafqw (not ready) [type berrybush] |
| 3 |  | bookcase_baseboard | 1 | hlhneo | 25 |  |  |  | medium | Only one candidate in the group; fewer than 2 test instances selectable | Every row is a 50/19 one-finger closure (finge | hlhneo -> bookcase_side/bxfkjj [type shelfbase] |
| 3 |  | bow | 0 |  |  |  |  |  | high | no model with an AG-valid grasp: fhchql: no AG-valid grasp |  |
| 3 |  | cabinet_door | 1 | ertkre | 21 |  |  |  | medium | Only one candidate in the group, so only one test instance can be selected. | Pinch spans an 8 mm panel with one finger  | ertkre -> cabinet_base/dcmhhh [type doorcabinet] |
| 3 |  | fire_alarm | 0 |  |  |  |  |  | high | no model with an AG-valid grasp: dkwmmf: no candidates |  |
| 3 |  | holly_decoration | 1 | shmkir | 21 |  |  |  | medium | Only one candidate in the group, so only one test instance could be selected. | All clean grasps are 3-5 mm twig pinches | shmkir -> recreational_vehicle/doiucd [type displaywall], wall_nail/wlnail [type displaywall] |
| 3 |  | menu | 0 |  |  |  |  |  | high | no model with an AG-valid grasp: keaqlh: no candidates |  |
| 3 |  | skateboard_wheel_assembly | 2 | jljqtt | 23 | acmbvb | 14 |  | medium | acmbvb: rank-1 grasp is an axial end-wheel pinch (wheel could spin, truck hangs ~80 mm off-axis); the only sound central | jljqtt -> skateboard_deck/tgfidu [type jljqttparent] ; acmbvb -> skateboard_deck/tgfidu [type acmbvbparent] |
| 3 |  | trout | 1 | kqpeke | 25 |  |  |  | high | Only one candidate in this group; a second test instance cannot be selected. | Rows 4-5 pinch the rear body 44-50 mm beh | kqpeke -> fishing_rod/zhmrtc [type fishrod] |

## IORS

| round | relation | category | cands | pick 1 | score | pick 2 | score | alternate | conf | warnings |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | inside | garlic_clove | 10 | faeyay | 24 | qaqrqd | 24 | hpzvmi | medium | Every row in this group closes one finger to 0 mm (fingertip pinch on ~25 mm cloves), so R=1 throughout; jwhnfc has an u |
| 1 | inside | hard_candy | 5 | yuatcg | 22 | ojznyf | 20 | qptjgv | medium |  |
| 1 | inside | quartz | 6 | kilkpz | 23 | vfuilh | 23 | poplsu | high | klgses has no clean row (all end pinches) and is not selectable; tgqodl cluster relies on one-finger 82 mm spans with un |
| 1 | inside | sunflower | 7 | dkajbh | 24 | mdpoct | 23 | urpskm | high | Rank-1 V1 panel is blank for pibfxd, mdpoct and dkajbh (strip rendering, object out of frame); V2-V4 confirm those grasp |
| 1 | nextto | cinnamon_stick | 8 | qxpzdm | 23 | kdaxdy | 19 | qmlyim | high | hjhcpm, cdkjfo, qsqvgk, smfuqz and bmbjdf have collision hulls far smaller than the visual quill (fingers close to 0-5 m |
| 1 | nextto | marker | 12 | xakoic | 22 | mcpkkq | 22 | qemval | medium | All markers are 12-18 mm thick so every pinch closes one finger to 0 mm (R=1); qemval is the only distinct shape/variant |
| 1 | nextto | parsnip | 6 | osqisk | 25 | invznw | 23 | jidpjl | high |  |
| 1 | nextto | pencil | 12 | hmawzy | 24 | rzscao | 24 | gbzzcg | medium | All twelve are SCALED variants (cross-section ~x3.2-4.2 / ~x5.0); every row is an asymmetric closure with one finger at  |
| 1 | ontop | cherry_tomato | 5 | dqjuub | 25 | dqvtgm | 21 | hmsqrh | high | Every row in this group reads x/0: one finger reaches its closed limit and the tomato is pinched off-centre in the gripp |
| 1 | ontop | pen | 6 | gzrbrx | 24 | hiyeki | 24 | ubwjqa | high | All pens pinch at 9-13 mm, so R=1 for every candidate (inherent to the category). | nvyicz transparent barrel and enfumt |
| 1 | ontop | pickle | 5 | iobrqp | 23 | rxuimv | 23 | vuspsy | medium |  |
| 1 | ontop | shampoo_dispenser | 7 | zoyyia | 21 | kkbhzw | 19 | thurmw | medium | Most rows in this group are 50/x one-finger assisted closures; only zoyyia, thurmw, nrthyl (row 2) and kkbhzw (row 2) sh |
| 1 | under | beer_glass | 3 | mxsliu | 20 | rxscji | 18 |  | low | lymciz not selectable: all five grasps pinch the fragile rim lip (no clean row); its 106 mm diameter exceeds the gripper |
| 1 | under | briefcase | 3 | lwbctq | 23 | uquevd | 21 | psvdau | medium | uquevd is a soft handle-less felt sleeve rather than a classic briefcase; psvdau's 87-88 mm pinch is at the edge of the  |
| 1 | under | french_fries | 9 | kxpfcj | 23 | xshwvn | 22 | oeterg | high | Six of nine candidates have collision hulls much thinner than the scaled visual mesh (pads penetrate or fingers close to |
| 1 | under | sandal | 9 | uhzfis | 24 | chgcrm | 24 | zafskl | high | Both picks are tan/olive leather sandals pinched across an 82-90 mm sole with one finger fully open; robustness is borde |
| 2 | inside | chili | 4 | rafkbt | 25 | agecro | 25 | pbbkpz | high | xhbpqh (6 mm) and pbbkpz (9 mm) are very thin: fingers close to 6-10 mm, below the 10 mm robustness floor (R 0); xhbpqh  |
| 2 | inside | ice_cube | 8 | ujmgkb | 25 | dueotq | 25 | tqsquu | high | The seven frosty cubes look nearly identical, so the pair cannot span much appearance variety; only kdlafa (clear glass  |
| 2 | inside | lemon | 12 | ovbcff | 25 | gkrfha | 23 | dsacfa | medium |  |
| 2 | nextto | jar_of_spaghetti_sauce | 6 | zahyds | 23 | efrxze | 22 | xpasvb | high | Five dark-jar candidates share one model with different labels and pinch at 85-87 mm, close to the ~97 mm gripper openin |
| 2 | nextto | spatula | 9 | hkvsep | 25 | ktkigm | 23 | ymvrct | high | ymvrct ties hkvsep at 25 but shares the wood-handle/dark-head look; pair rule preferred the different-looking ktkigm (23 |
| 2 | ontop | apple | 6 | omzprq | 25 | agveuv | 25 | zlxfxt | high | obixxh and zlxfxt are the same green apple asset; obixxh's top row is a 50/8 one-finger closure 21 mm off-centre. | netb |
| 2 | ontop | canned_food | 5 | ycbspm | 24 | qhgdys | 21 | byakrm | medium | Every row in this group is a one-finger closure (50/x): the object sits 14-22 mm off the gripper centre against the open |
| 2 | ontop | coffee_cup | 11 | siksnl | 23 | ckkwmj | 23 | xjdyon | high |  |
| 2 | under | teacup | 6 | cpozxi | 22 | oxfzfe | 22 | wopjex | medium | All candidates are thin rim pinches (3-15 mm gap), robustness borderline for the whole group; tfzfam, zdvgol, wopjex and |
| 2 | under | toy_figure | 12 | cobfpg | 25 | nncqfn | 25 | psfkaj | high | kitqln not selectable: n_clean 0 and only grasp is a 12 mm ear pinch 78 mm from CoM | usnnxe has a single valid grasp (G |
| 3 | inside | brussels_sprouts | 4 | mbkrxe | 25 | siodbb | 23 | hkwyzk | medium | vdamtq closes every row at both finger limits (48-49/0) with the sprout entirely in one half of the opening; hkwyzk rows |
| 3 | inside | half_lime | 4 | iwjqqq | 23 | dtyjkb | 23 | wdadtn | medium | Most rows of every candidate are 50/x assisted one-finger pushes (lime ~20 mm off gripper centre) but the dome stays hel |
| 3 | inside | lily | 6 | yltekp | 23 | relerp | 22 | mohrqo | medium | jfeimu and jobzwo have no clean grasp (hull penetration / foliage squeeze) and are not selectable. | mohrqo and jtouad h |
| 3 | nextto | pepper_shaker | 3 | tgcwtr | 23 | ykujyn | 21 | stbqbt | medium | ykujyn and tgcwtr look like near-identical grinder assets (same 67x67x160 mm bbox, navy cap, speckled body), so the sele |
| 3 | nextto | salt_shaker | 2 | iomwtn | 24 | fonexa | 19 |  | medium | fonexa: grinder spike protrudes below the base, so the shaker may rest tilted on the table; every grasp is a 50/0 one-fi |
| 3 | ontop | baguette | 1 | xydhpd | 23 |  |  |  | high | Only one candidate in the group; no second test instance is available for baguette in this round. | Rows 1-4 are one_fin |
| 3 | ontop | half_bagel | 2 | gywbkh | 22 | hsvakn | 22 |  | medium | only two candidates in the group; both selected (short) | every row of both candidates closes one finger to 0 (R 1) and  |
| 3 | ontop | pack_of_kielbasa | 12 | dgodnn | 25 | femhoz | 23 | rvhpal | medium |  |
| 3 | under | bottle_of_supplements | 2 | oqakev | 23 | kgreql | 20 |  | medium | Only 2 candidates in the group, both selected. oqakev: all five rows are the same one-finger (50/17) shoulder pinch and  |
| 3 | under | tennis_ball | 1 | rgekxe | 25 |  |  |  | high | Only one candidate in this group (round 3), so no second test instance can be selected; the single model itself has no w |

## Models selected for more than one task

apple/agveuv (IORS, object_scaling), apple/omzprq (IORS, object_scaling), bar_soap/feqemg (disembedding, object_scaling), beaker/zycgen (disembedding, object_scaling), bell_pepper/ukkycp (mental_rotation, object_scaling), bottle_of_medicine/kqkwoq (disembedding, object_scaling), bottle_of_medicine/zfbnjh (disembedding, object_scaling), briefcase/lwbctq (IORS, mental_rotation), briefcase/uquevd (IORS, mental_rotation), can_of_soda/itolcg (disembedding, object_scaling), can_of_soda/opivig (disembedding, object_scaling), canned_food/ycbspm (IORS, object_scaling), cherry_tomato/dqjuub (IORS, object_scaling), chili/agecro (IORS, disembedding), chili/rafkbt (IORS, disembedding), cookie_cutter/jpscvj (disembedding, mental_rotation), hard_candy/yuatcg (IORS, object_scaling), jar_of_spaghetti_sauce/zahyds (IORS, object_scaling), mouse/cduwda (disembedding, mental_rotation), mug/ppzttc (disembedding, mental_rotation), pickle/iobrqp (IORS, mental_rotation), potato/bpwohr (mental_rotation, object_scaling), potato/lqjear (mental_rotation, object_scaling), quartz/kilkpz (IORS, mental_rotation), sandal/uhzfis (IORS, mental_rotation), spatula/hkvsep (IORS, mental_rotation), tablefork/hktwlm (disembedding, mental_rotation), teacup/oxfzfe (IORS, mental_rotation), toy_dice/eqwazv (disembedding, object_scaling), webcam/fabztm (configuration_matching, mental_rotation)
