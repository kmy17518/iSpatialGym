## Per-tag counts (1480 models surveyed, 0 loader failures)

| tag | `symmetry` (final) | `symmetry_near` (review) | geometry exact (before texture) |
|---|---|---|---|
| `x2` | 17 | 33 | 278 |
| `y2` | 15 | 18 | 267 |
| `z2` | 72 | 56 | 364 |
| `x4` | 9 | 9 | 31 |
| `y4` | 2 | 3 | 17 |
| `z4` | 29 | 19 | 82 |
| `x3` | 1 | 3 | 1 |
| `y3` | 0 | 0 | 2 |
| `z3` | 1 | 4 | 5 |
| `x5` | 19 | 10 | 22 |
| `y5` | 1 | 1 | 1 |
| `z5` | 9 | 15 | 55 |
| `x6` | 20 | 11 | 24 |
| `y6` | 1 | 4 | 1 |
| `z6` | 14 | 20 | 64 |
| `xinf` | 8 | 16 | 42 |
| `yinf` | 0 | 2 | 2 |
| `zinf` | 96 | 44 | 266 |
| `full` | 1 | 0 | 2 |
| `none` | 1237 | 0 | 624 |

Models with a final symmetry: 243; asymmetric (`none`): 1237; with a `symmetry_near` entry to review: 190; with at least one geometric candidate (contact sheet rendered): 965.

## Flat renders

301 of 1480 models render (near-)uniform (mode share >= 0.85 or colour std <= 6/255): 261 are genuinely plain (`plain_albedo`: the texels the mesh uses are uniform in every map), 40 are `flat_render` (a reflection / emissive / diffuse map with structure the albedo render cannot show) - their 17 models with a symmetry were demoted to `symmetry_near`.

| key | render std | mode share | hidden maps | `symmetry` (demoted to near) |
|---|---|---|---|---|
| `trash_can__djgllo` | 3.5 | 0.99 | reflection | `zinf` |
| `trash_can__gilsji` | 3.4 | 0.99 | reflection | `zinf` |
| `hamper__ssfvij` | 3.2 | 0.96 | reflection | `-` |
| `public_trash_can__fkosow` | 21.7 | 0.97 | reflection | `-` |
| `serving_cart__vvtcby` | 34.4 | 0.91 | reflection | `-` |
| `lid__wtjthb` | 12.8 | 0.90 | reflection | `zinf` |
| `lid__snklci` | 8.1 | 0.90 | reflection | `z2` |
| `bottle_of_wine__uoykwx` | 5.9 | 0.85 | reflection | `zinf` |
| `bottle_of_wine__jdqahn` | 19.2 | 0.88 | reflection | `zinf` |
| `shampoo_dispenser__zoyyia` | 25.4 | 0.85 | reflection | `-` |
| `shampoo_dispenser__nrthyl` | 21.8 | 0.89 | reflection | `z4` |
| `beefsteak_tomato__ogpans` | 11.7 | 0.95 | reflection | `-` |
| `mug__lgxhsc` | 1.9 | 0.99 | reflection | `-` |
| `mouse__fjljxp` | 3.2 | 0.96 | reflection | `-` |
| `mouse__ihpnjx` | 2.9 | 0.97 | reflection | `-` |
| `mouse__eosngb` | 5.9 | 0.96 | reflection | `-` |
| `toothbrush__nbgpwg` | 2.8 | 0.80 | reflection | `-` |
| `notebook__dlhxxl` | 60.4 | 0.85 | reflection | `-` |
| `notebook__zkpswz` | 61.0 | 0.86 | reflection | `-` |
| `notebook__zkmmnf` | 36.4 | 0.92 | reflection | `x2 y2 z2` |
| `notebook__gqkaqu` | 37.9 | 0.91 | reflection | `x2` |
| `notebook__qwhiqc` | 58.9 | 0.89 | reflection | `-` |
| `hardback__acaxtl` | 3.8 | 0.90 | reflection | `y2` |
| `hardback__aceozs` | 23.9 | 0.87 | reflection | `-` |
| `lipstick__albshz` | 15.7 | 0.88 | reflection | `-` |
| `bag_of_snacks__pkwgid` | 11.2 | 0.86 | diffuse | `-` |
| `canister__xcppkc` | 4.6 | 0.96 | reflection | `z2` |
| `dip_candle__poexjo` | 10.3 | 0.90 | reflection | `x2 z4 z5 z6` |
| `dip_candle__ghctun` | 9.4 | 0.95 | reflection | `zinf` |
| `dip_candle__mxjejs` | 6.2 | 0.90 | reflection | `x2 zinf` |
| `light_bulb__ymomhw` | 7.8 | 0.89 | reflection | `-` |
| `frying_pan__snvhlz` | 4.6 | 0.98 | reflection | `-` |
| `keyboard__uwacaq` | 8.0 | 0.88 | reflection | `z2` |
| `clipboard__gmxyfo` | 5.7 | 0.98 | reflection | `x2 z2` |
| `fire_extinguisher__bejaeu` | 33.4 | 0.86 | reflection | `-` |
| `tray__zcmnji` | 6.1 | 0.89 | reflection | `zinf` |
| `saucepot__tfzijn` | 3.5 | 0.99 | reflection | `z2` |
| `saucepot__uvzmss` | 3.6 | 0.98 | reflection | `z2` |
| `saucepot__kvgaar` | 4.7 | 0.98 | reflection | `-` |
| `salt_shaker__fonexa` | 11.3 | 0.94 | reflection | `zinf` |

## Scaled variants

41 models have a grasp-tested spawn scale; the symmetry at that scale differs from the canonical model for 21:

- `french_fries__oeterg` scale [1.0, 1.871, 2.385]: canonical `none` / near `z2` -> scaled `none` / near `-`
- `french_fries__nfgqwu` scale [1.0, 1.891, 2.458]: canonical `none` / near `z2` -> scaled `none` / near `-`
- `french_fries__uqsuef` scale [1.0, 2.155, 2.458]: canonical `none` / near `z2` -> scaled `none` / near `-`
- `cinnamon_stick__qxpzdm` scale [1.0, 1.0, 1.968]: canonical `none` / near `x2 z2` -> scaled `none` / near `x2 z2 y5 y6`
- `cinnamon_stick__kdaxdy` scale [1.0, 1.0, 1.87]: canonical `none` / near `x2 y2 z2` -> scaled `none` / near `y2`
- `cinnamon_stick__qmlyim` scale [1.054, 1.0, 1.326]: canonical `none` / near `x2` -> scaled `none` / near `-`
- `cinnamon_stick__cdkjfo` scale [1.647, 1.0, 2.423]: canonical `none` / near `x2 z2 y4 y5 y6` -> scaled `none` / near `-`
- `cinnamon_stick__smfuqz` scale [1.207, 1.0, 1.8]: canonical `none` / near `z2 y6` -> scaled `none` / near `z2`
- `cinnamon_stick__hjhcpm` scale [1.0, 1.0, 2.508]: canonical `none` / near `z2` -> scaled `none` / near `-`
- `pencil__hmawzy` scale [1.0, 3.231, 4.998]: canonical `x5 x6` / near `z2 x4` -> scaled `none` / near `-`
- `pencil__rzscao` scale [1.0, 3.867, 5.113]: canonical `xinf` / near `y2 z2` -> scaled `none` / near `-`
- `pencil__gbzzcg` scale [1.0, 3.901, 5.106]: canonical `xinf` / near `-` -> scaled `none` / near `-`
- `pencil__fnlbpu` scale [1.0, 4.217, 5.069]: canonical `xinf` / near `-` -> scaled `none` / near `xinf`
- `pencil__heqkmr` scale [1.0, 3.345, 5.081]: canonical `x6` / near `y2 xinf` -> scaled `none` / near `-`
- `pencil__jntybl` scale [1.0, 3.96, 5.098]: canonical `xinf` / near `-` -> scaled `none` / near `-`
- `pencil__lruiot` scale [1.0, 3.603, 5.085]: canonical `x5 x6` / near `y2 xinf` -> scaled `none` / near `-`
- `pencil__lykqoh` scale [1.0, 3.596, 5.119]: canonical `x4 x5 x6` / near `y2 xinf` -> scaled `none` / near `-`
- `pencil__vpzhth` scale [1.0, 3.361, 5.096]: canonical `x4 x5 x6` / near `y2 xinf` -> scaled `none` / near `-`
- `pencil__vvofwr` scale [1.0, 3.473, 5.014]: canonical `x5 x6` / near `y2 xinf` -> scaled `none` / near `-`
- `pencil__mvmxfo` scale [1.0, 3.327, 5.106]: canonical `none` / near `xinf` -> scaled `none` / near `-`
- `pencil__vedvit` scale [1.0, 3.451, 5.058]: canonical `y2 xinf` / near `-` -> scaled `none` / near `-`

## Sanity table

| category | models | most common `symmetry` | most common geometry-exact | expectation | match |
|---|---|---|---|---|---|
| bowl | 12 | `zinf` x6, `none` x3, `z4` x2 | `zinf` x9, `z4` x2, `z2` x1 | zinf (plain bowls; a decorated rim breaks it) | |
| plate | 12 | `zinf` x5, `none` x4, `z4` x2 | `zinf` x9, `z4` x3 | zinf | |
| mug | 10 | `none` x10 | `none` x10 | none (handle) | |
| teacup | 6 | `none` x4, `zinf` x2 | `zinf` x3, `none` x3 | zinf if handle-less, else none | |
| teapot | 3 | `none` x3 | `none` x3 | none | |
| sunflower | 17 | `none` x17 | `none` x17 | none | |
| pencil | 13 | `xinf` x5, `x5 x6` x3, `x4 x5 x6` x2 | `y2 xinf` x12, `z2 x4 x5 x6` x1 | xinf/yinf/zinf about the long axis if unprinted; a print breaks it | |
| bottle_of_medicine | 8 | `none` x8 | `zinf` x4, `none` x2, `z5 z6` x1 | zinf geometry; the label breaks the texture -> none | |
| can_of_baking_mix | 6 | `z2` x4, `none` x2 | `zinf` x3, `x2 zinf` x3 | zinf geometry; label -> none | |
| can_of_soda | 9 | `none` x7, `zinf` x1, `z2` x1 | `z4` x4, `zinf` x3, `z2` x1 | zinf geometry; label -> none | |
| board_game | 16 | `none` x16 | `x2 y2 z2` x13, `none` x1, `x4 y4 z4` x1 | x2 y2 z2 box geometry; artwork -> none | |
| box_of_cookies | 14 | `none` x12, `x2` x1, `y2` x1 | `x2 y2 z2` x6, `x2 y2 z4` x4, `z2` x1 | box geometry; print -> none | |
| box_of_cereal | 14 | `none` x14 | `x2 y2 z2` x14 | box geometry; print -> none | |
| apple | 7 | `none` x7 | `none` x7 | none / near zinf (stem) | |
| orange | 2 | `none` x1, `full` x1 | `x2 zinf` x1, `full` x1 | full or zinf geometry; texture decides | |
| tennis_ball | 1 | `x2` x1 | `full` x1 | full (plain) or none (seam) | |
| toy_dice | 20 | `none` x19, `y2` x1 | `x2 z2 y4` x9, `none` x6, `x2 y2 z4` x3 | cube geometry x4 y4 z4; pips -> none | |
| jar_of_grains | 12 | `none` x12 | `none` x12 | zinf geometry; label -> none | |
| cap | 19 | `zinf` x11, `z4 z5 z6` x2, `z2` x2 | `zinf` x12, `z2` x2, `z5 z6` x1 | zinf | |
| lid | 17 | `zinf` x7, `z2` x5, `none` x4 | `zinf` x9, `z2` x6, `x2 y2 z2` x1 | zinf | |
| keyboard | 6 | `none` x6 | `none` x4, `y2` x1, `z2` x1 | none | |
| painting | 11 | `none` x11 | `x2 y2 z2` x6, `x2` x2, `x4` x2 | none | |
| beaker | 10 | `zinf` x6, `z2` x2, `none` x1 | `zinf` x7, `z2` x2, `z4` x1 | zinf | |
| bar_soap | 4 | `x2 y2 z2` x2, `none` x1, `x2 zinf` x1 | `x2 y2 z2` x2, `none` x1, `x2 zinf` x1 | zinf or x2 y2 z2 | |
| wineglass | 12 | `zinf` x12 | `zinf` x12 | zinf | |
| egg | 1 | `none` x1 | `none` x1 | zinf or xinf (about the long axis) | |
| wall_clock | 5 | `none` x5 | `xinf` x2, `y2 xinf` x2, `x4 x3` x1 | none (numerals) | |
| vase | 6 | `none` x2, `z4` x2, `z2` x1 | `z4 z5 z6` x1, `none` x1, `z4` x1 | zinf unless decorated | |
| platter | 6 | `none` x3, `z2` x2, `z3` x1 | `z2` x5, `z3` x1 | zinf or x2 y2 z2 (oval) | |
| tray | 12 | `z2` x8, `none` x3, `z4` x1 | `z2` x9, `x2 y2 z2` x1, `zinf` x1 | x2 y2 z2 (plain rectangular) or none | |

## `mental_rotation` targets with a symmetry (3 of 196; exclude or give `orientation_equivalences`)

| category | model | `symmetry` | `symmetry_near` |
|---|---|---|---|
| teacup | zdvgol | `zinf` | `-` |
| teacup | tfzfam | `zinf` | `-` |
| hook | wmvozo | `zinf` | `-` |

`mental_rotation` targets with only a `symmetry_near` entry (review before use): 6: `teacup/cpozxi` (z5), `pickle/ifterk` (y4), `pickle/iobrqp` (y4), `cookie_cutter/lqrfzo` (z4), `tongs/igsklx` (x2), `tongs/gbjqyn` (x2)

## `symmetric_180` (alignment targets) vs the survey

The half turn that matters is about the alignment axis: the wall normal = the frame's front axis for the wall subtasks, the
vertical (`frame_up`, else the canonical `z`) for `alignment-axial` (the object is yawed on the table). `symmetric_180` is only
set for `alignment-world_horizontal` targets in the selection sheet (empty = the reviewer did not call the object symmetric),
so every axial / vertical row below is a symmetry the reviewers did not record rather than a contradiction.

| subtask | category | model | half-turn axis | `symmetric_180` | survey half turn | survey `symmetry` | `symmetry_near` |
|---|---|---|---|---|---|---|---|
| alignment-axial | clipboard | gmxyfo | +z (up) | 1 | near | `none` | `x2 z2` |
| alignment-world_horizontal | rolling_pin | mwrdae | +x (front) | 1 | near | `none` | `x2 yinf` |
| alignment-world_horizontal | shelf | kxvbui | +z (front) | 1 | no | `x2` | `-` |
| alignment-world_horizontal | wall_mounted_tv | ggfznh | +x (front) | 1 | no | `none` | `-` |
| alignment-world_horizontal | wall_mounted_tv | dxyfxf | +x (front) | 1 | no | `none` | `-` |
| alignment-world_horizontal | wall_mounted_tv | ylvjhb | +x (front) | 1 | no | `none` | `-` |
| alignment-world_horizontal | wall_mounted_tv | sodbbr | +x (front) | 1 | no | `none` | `-` |
| alignment-world_horizontal | whiteboard | gvkgsi | +x (front) | 1 | no | `none` | `-` |
| alignment-world_horizontal | towel_rack | crivoa | +x (front) | 1 | no | `none` | `-` |
| alignment-world_horizontal | towel_rack | quznqx | +x (front) | 1 | near | `none` | `x2 y2 z2` |
| alignment-world_horizontal | decorative_sign | hgonwj | +x (front) | 1 | near | `none` | `x2` |

11 alignment-target rows disagree (11 flagged `symmetric_180 = 1` without a confirmed survey half turn, 0 with a survey half turn the sheet does not record).

## Loader / render failures

Geometry stage: 0 failures; texture stage: 0 failures.


## Review list (190 models with a `symmetry_near` entry)

| model | `symmetry` | `symmetry_near` | why | sheet |
|---|---|---|---|---|
| `trash_can__djgllo` | `none` | `zinf` | zinf: geom exact (mean 0.14% frac2 0.0%), tex preserved (MAD 0.0) | runlogs/symmetry_20260915/renders/trash_can__djgllo.jpg |
| `trash_can__gilsji` | `none` | `zinf` | zinf: geom exact (mean 0.14% frac2 0.0%), tex preserved (MAD 0.0) | runlogs/symmetry_20260915/renders/trash_can__gilsji.jpg |
| `hamper__sijioo` | `none` | `z3` | z3: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 7.9) | runlogs/symmetry_20260915/renders/hamper__sijioo.jpg |
| `taboret__ysrhuw` | `none` | `z2` | z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 2.8) | runlogs/symmetry_20260915/renders/taboret__ysrhuw.jpg |
| `taboret__wjrpwg` | `none` | `z3` | z3: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 4.6) | runlogs/symmetry_20260915/renders/taboret__wjrpwg.jpg |
| `taboret__bciekv` | `none` | `z4` | z4: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 4.0) | runlogs/symmetry_20260915/renders/taboret__bciekv.jpg |
| `ottoman__xcmniq` | `none` | `z2` | z2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/ottoman__xcmniq.jpg |
| `ottoman__qsnawv` | `none` | `z4` | z4: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 0.2) | runlogs/symmetry_20260915/renders/ottoman__qsnawv.jpg |
| `bottle_of_water__cyzaue` | `none` | `z6` | z6: geom exact (mean 0.15% frac2 0.0%), tex borderline (MAD 1.1) | runlogs/symmetry_20260915/renders/bottle_of_water__cyzaue.jpg |
| `lid__wtjthb` | `none` | `zinf` | zinf: geom exact (mean 0.07% frac2 0.0%), tex preserved (MAD 0.7) | runlogs/symmetry_20260915/renders/lid__wtjthb.jpg |
| `lid__qferch` | `z2` | `zinf` | zinf: geom near (mean 0.18% frac2 2.0%), tex preserved (MAD 1.3) | runlogs/symmetry_20260915/renders/lid__qferch.jpg |
| `lid__snklci` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex preserved (MAD 0.1) | runlogs/symmetry_20260915/renders/lid__snklci.jpg |
| `bottle_of_wine__uoykwx` | `none` | `zinf` | zinf: geom exact (mean 0.14% frac2 0.0%), tex preserved (MAD 0.2) | runlogs/symmetry_20260915/renders/bottle_of_wine__uoykwx.jpg |
| `bottle_of_wine__jdqahn` | `none` | `zinf` | zinf: geom exact (mean 0.18% frac2 0.0%), tex preserved (MAD 0.4) | runlogs/symmetry_20260915/renders/bottle_of_wine__jdqahn.jpg |
| `marker__zvvovh` | `none` | `x4 x5 x6` | x4: geom near (mean 0.61% frac2 0.3%), tex preserved (MAD 0.6); x5: geom near (mean 0.55% frac2 0.4%), tex preserved (MAD 0.5); x6: geom near (mean 0.50% frac2 0.5%), tex preserved (MAD 0.5) | runlogs/symmetry_20260915/renders/marker__zvvovh.jpg |
| `marker__slgheo` | `none` | `x4 x5 x6` | x4: geom near (mean 0.60% frac2 0.3%), tex borderline (MAD 1.4); x5: geom near (mean 0.54% frac2 0.3%), tex borderline (MAD 1.3); x6: geom near (mean 0.49% frac2 0.5%), tex borderline (MAD 1.2) | runlogs/symmetry_20260915/renders/marker__slgheo.jpg |
| `marker__erhlbb` | `none` | `x4 x5 x6` | x4: geom near (mean 0.57% frac2 0.5%), tex borderline (MAD 0.5); x5: geom near (mean 0.52% frac2 0.1%), tex borderline (MAD 0.5); x6: geom near (mean 0.47% frac2 0.2%), tex preserved (MAD 0.4) | runlogs/symmetry_20260915/renders/marker__erhlbb.jpg |
| `marker__jlvneb` | `none` | `xinf` | xinf: geom near (mean 0.75% frac2 0.9%), tex borderline (MAD 1.1) | runlogs/symmetry_20260915/renders/marker__jlvneb.jpg |
| `marker__mcpkkq` | `none` | `xinf` | xinf: geom near (mean 0.79% frac2 0.2%), tex borderline (MAD 1.7) | runlogs/symmetry_20260915/renders/marker__mcpkkq.jpg |
| `cap__rgtedj` | `z5 z6` | `zinf` | zinf: geom near (mean 0.43% frac2 0.0%), tex preserved (MAD 0.2) | runlogs/symmetry_20260915/renders/cap__rgtedj.jpg |
| `cap__ghwjwe` | `z4 z5 z6` | `zinf` | zinf: geom exact (mean 0.38% frac2 0.0%), tex borderline (MAD 2.3) | runlogs/symmetry_20260915/renders/cap__ghwjwe.jpg |
| `cap__ygsmgm` | `z6` | `zinf` | zinf: geom near (mean 0.34% frac2 1.0%), tex preserved (MAD 0.1) | runlogs/symmetry_20260915/renders/cap__ygsmgm.jpg |
| `can_of_soda__bfrzvk` | `z2` | `zinf` | zinf: geom near (mean 0.71% frac2 0.0%), tex borderline (MAD 3.8) | runlogs/symmetry_20260915/renders/can_of_soda__bfrzvk.jpg |
| `can_of_soda__lugwcz` | `none` | `zinf` | zinf: geom exact (mean 0.21% frac2 0.0%), tex borderline (MAD 9.9) | runlogs/symmetry_20260915/renders/can_of_soda__lugwcz.jpg |
| `bottle_of_beer__zlmwyn` | `none` | `z4 z5 z6` | z4: geom exact (mean 0.21% frac2 0.0%), tex borderline (MAD 3.3); z5: geom exact (mean 0.19% frac2 0.0%), tex borderline (MAD 2.9); z6: geom exact (mean 0.17% frac2 0.0%), tex borderline (MAD 2.7) | runlogs/symmetry_20260915/renders/bottle_of_beer__zlmwyn.jpg |
| `lemon__vuwehu` | `none` | `y2` | y2: geom near (mean 0.67% frac2 0.8%), tex borderline (MAD 2.1) | runlogs/symmetry_20260915/renders/lemon__vuwehu.jpg |
| `lemon__asdada` | `none` | `x6` | x6: geom exact (mean 0.23% frac2 0.0%), tex borderline (MAD 5.5) | runlogs/symmetry_20260915/renders/lemon__asdada.jpg |
| `lemon__kvgfdl` | `none` | `x5 x6` | x5: geom exact (mean 0.26% frac2 0.0%), tex borderline (MAD 5.9); x6: geom exact (mean 0.26% frac2 0.0%), tex borderline (MAD 5.5) | runlogs/symmetry_20260915/renders/lemon__kvgfdl.jpg |
| `lemon__bbvonm` | `none` | `y2 z2 x5 x6` | y2: geom near (mean 0.56% frac2 0.0%), tex preserved (MAD 0.1); z2: geom near (mean 0.70% frac2 0.1%), tex preserved (MAD 0.1); x5: geom near (mean 0.69% frac2 1.8%), tex preserved (MAD 0.1); x6: geom near (mean 0.63% frac2 0.9%), tex preserved (MAD 0.1) | runlogs/symmetry_20260915/renders/lemon__bbvonm.jpg |
| `shampoo_dispenser__nrthyl` | `none` | `z4` | z4: geom near (mean 0.53% frac2 0.3%), tex borderline (MAD 0.7) | runlogs/symmetry_20260915/renders/shampoo_dispenser__nrthyl.jpg |
| `beaker__effbnc` | `none` | `zinf` | zinf: geom exact (mean 0.31% frac2 0.0%), tex borderline (MAD 9.4) | runlogs/symmetry_20260915/renders/beaker__effbnc.jpg |
| `apple__agveuv` | `none` | `x6` | x6: geom near (mean 0.37% frac2 1.7%), tex borderline (MAD 9.3) | runlogs/symmetry_20260915/renders/apple__agveuv.jpg |
| `beeswax_candle__nhdnje` | `none` | `zinf` | zinf: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 1.6) | runlogs/symmetry_20260915/renders/beeswax_candle__nhdnje.jpg |
| `beeswax_candle__oimgmh` | `z4 z5 z6` | `zinf` | zinf: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 1.4) | runlogs/symmetry_20260915/renders/beeswax_candle__oimgmh.jpg |
| `beeswax_candle__ouzkdj` | `none` | `zinf` | zinf: geom exact (mean 0.09% frac2 0.0%), tex borderline (MAD 2.4) | runlogs/symmetry_20260915/renders/beeswax_candle__ouzkdj.jpg |
| `beeswax_candle__nxewyk` | `none` | `zinf` | zinf: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 1.5) | runlogs/symmetry_20260915/renders/beeswax_candle__nxewyk.jpg |
| `beeswax_candle__pfewit` | `none` | `zinf` | zinf: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 0.8) | runlogs/symmetry_20260915/renders/beeswax_candle__pfewit.jpg |
| `bottle_of_catsup__bcqfxb` | `none` | `z2` | z2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 0.7) | runlogs/symmetry_20260915/renders/bottle_of_catsup__bcqfxb.jpg |
| `bar_soap__feqemg` | `none` | `z2` | z2: geom near (mean 0.72% frac2 1.9%), tex borderline (MAD 0.0) | runlogs/symmetry_20260915/renders/bar_soap__feqemg.jpg |
| `cherry_tomato__dqjuub` | `none` | `z5 z6` | z5: geom near (mean 0.32% frac2 2.6%), tex preserved (MAD 1.2); z6: geom near (mean 0.28% frac2 1.6%), tex preserved (MAD 1.1) | runlogs/symmetry_20260915/renders/cherry_tomato__dqjuub.jpg |
| `bottle_of_black_pepper__ydzzrv` | `none` | `zinf` | zinf: geom near (mean 0.53% frac2 2.2%), tex borderline (MAD 5.3) | runlogs/symmetry_20260915/renders/bottle_of_black_pepper__ydzzrv.jpg |
| `ice_cube__dueotq` | `none` | `z4` | z4: geom near (mean 0.58% frac2 2.2%), tex preserved (MAD 0.0) | runlogs/symmetry_20260915/renders/ice_cube__dueotq.jpg |
| `ice_cube__lkirej` | `none` | `z4` | z4: geom near (mean 0.59% frac2 2.2%), tex preserved (MAD 0.0) | runlogs/symmetry_20260915/renders/ice_cube__lkirej.jpg |
| `spray_can__tehmzy` | `z5 z6` | `z4` | z4: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.5) | runlogs/symmetry_20260915/renders/spray_can__tehmzy.jpg |
| `vidalia_onion__buyxll` | `none` | `xinf` | xinf: geom near (mean 0.53% frac2 1.3%), tex borderline (MAD 11.9) | runlogs/symmetry_20260915/renders/vidalia_onion__buyxll.jpg |
| `vidalia_onion__dejqhf` | `none` | `xinf` | xinf: geom exact (mean 0.21% frac2 0.6%), tex borderline (MAD 4.0) | runlogs/symmetry_20260915/renders/vidalia_onion__dejqhf.jpg |
| `pack_of_cigarettes__wjfbhw` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 4.2) | runlogs/symmetry_20260915/renders/pack_of_cigarettes__wjfbhw.jpg |
| `orange__ucstpm` | `none` | `z6` | z6: geom exact (mean 0.42% frac2 0.0%), tex borderline (MAD 4.1) | runlogs/symmetry_20260915/renders/orange__ucstpm.jpg |
| `toy_dice__eqwazv` | `none` | `z2` | z2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 6.1) | runlogs/symmetry_20260915/renders/toy_dice__eqwazv.jpg |
| `toy_dice__hbkqqp` | `none` | `z2` | z2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 6.2) | runlogs/symmetry_20260915/renders/toy_dice__hbkqqp.jpg |
| `toy_dice__bjrxkj` | `y2` | `x2 z2` | x2: geom exact (mean 0.15% frac2 0.0%), tex preserved (MAD 5.3); z2: geom exact (mean 0.12% frac2 0.0%), tex preserved (MAD 5.1) | runlogs/symmetry_20260915/renders/toy_dice__bjrxkj.jpg |
| `toy_dice__nmhsvp` | `none` | `x2` | x2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 5.6) | runlogs/symmetry_20260915/renders/toy_dice__nmhsvp.jpg |
| `bottle_of_essential_oil__wansva` | `none` | `z5 z6` | z5: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 7.1); z6: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 6.3) | runlogs/symmetry_20260915/renders/bottle_of_essential_oil__wansva.jpg |
| `bottle_of_essential_oil__eyyhld` | `none` | `z4 z5 z6` | z4: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 9.0); z5: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 7.8); z6: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 7.0) | runlogs/symmetry_20260915/renders/bottle_of_essential_oil__eyyhld.jpg |
| `bottle_of_essential_oil__xvqshn` | `none` | `z6` | z6: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 6.2) | runlogs/symmetry_20260915/renders/bottle_of_essential_oil__xvqshn.jpg |
| `bottle_of_essential_oil__xhoipk` | `none` | `zinf` | zinf: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 11.5) | runlogs/symmetry_20260915/renders/bottle_of_essential_oil__xhoipk.jpg |
| `bottle_of_essential_oil__yjxvpg` | `none` | `zinf` | zinf: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 9.1) | runlogs/symmetry_20260915/renders/bottle_of_essential_oil__yjxvpg.jpg |
| `bottle_of_salsa__nafwlf` | `none` | `z4 z5 z6` | z4: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 2.6); z5: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 3.0); z6: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/bottle_of_salsa__nafwlf.jpg |
| `teacup__cpozxi` | `none` | `z5` | z5: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 3.3) | runlogs/symmetry_20260915/renders/teacup__cpozxi.jpg |
| `pickle__ifterk` | `none` | `y4` | y4: geom exact (mean 0.43% frac2 0.0%), tex borderline (MAD 8.5) | runlogs/symmetry_20260915/renders/pickle__ifterk.jpg |
| `pickle__iobrqp` | `none` | `y4` | y4: geom exact (mean 0.39% frac2 0.0%), tex borderline (MAD 9.1) | runlogs/symmetry_20260915/renders/pickle__iobrqp.jpg |
| `cookie_cutter__lqrfzo` | `none` | `z4` | z4: geom exact (mean 0.24% frac2 0.0%), tex borderline (MAD 5.1) | runlogs/symmetry_20260915/renders/cookie_cutter__lqrfzo.jpg |
| `tongs__igsklx` | `none` | `x2` | x2: geom exact (mean 0.19% frac2 0.0%), tex borderline (MAD 9.9) | runlogs/symmetry_20260915/renders/tongs__igsklx.jpg |
| `tongs__gbjqyn` | `none` | `x2` | x2: geom exact (mean 0.20% frac2 0.0%), tex borderline (MAD 10.5) | runlogs/symmetry_20260915/renders/tongs__gbjqyn.jpg |
| `notebook__zkmmnf` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.11% frac2 0.0%), tex preserved (MAD 0.8); y2: geom exact (mean 0.15% frac2 0.0%), tex preserved (MAD 1.0); z2: geom exact (mean 0.15% frac2 0.0%), tex borderline (MAD 1.4) | runlogs/symmetry_20260915/renders/notebook__zkmmnf.jpg |
| `soap_dispenser__zwudcz` | `z4` | `zinf` | zinf: geom exact (mean 0.09% frac2 0.1%), tex borderline (MAD 6.8) | runlogs/symmetry_20260915/renders/soap_dispenser__zwudcz.jpg |
| `soap_dispenser__edinmg` | `none` | `zinf` | zinf: geom exact (mean 0.25% frac2 0.9%), tex borderline (MAD 9.5) | runlogs/symmetry_20260915/renders/soap_dispenser__edinmg.jpg |
| `cellulose_tape__sklkyc` | `zinf` | `x2 y2` | x2: geom exact (mean 0.36% frac2 0.0%), tex borderline (MAD 11.0); y2: geom exact (mean 0.23% frac2 0.0%), tex borderline (MAD 10.8) | runlogs/symmetry_20260915/renders/cellulose_tape__sklkyc.jpg |
| `pack_of_cigarettes__qabdgg` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 1.8) | runlogs/symmetry_20260915/renders/pack_of_cigarettes__qabdgg.jpg |
| `toy_dice__ievnsq` | `none` | `z2` | z2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 6.1) | runlogs/symmetry_20260915/renders/toy_dice__ievnsq.jpg |
| `toy_dice__akguod` | `none` | `z2` | z2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 6.1) | runlogs/symmetry_20260915/renders/toy_dice__akguod.jpg |
| `toy_dice__kmagva` | `none` | `z2` | z2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 5.9) | runlogs/symmetry_20260915/renders/toy_dice__kmagva.jpg |
| `toy_dice__krpdji` | `none` | `x2` | x2: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 6.5) | runlogs/symmetry_20260915/renders/toy_dice__krpdji.jpg |
| `toy_dice__rbzyrv` | `none` | `z2` | z2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 6.2) | runlogs/symmetry_20260915/renders/toy_dice__rbzyrv.jpg |
| `toy_dice__xfqqhh` | `none` | `z2` | z2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 5.8) | runlogs/symmetry_20260915/renders/toy_dice__xfqqhh.jpg |
| `notebook__gqkaqu` | `none` | `x2` | x2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 1.7) | runlogs/symmetry_20260915/renders/notebook__gqkaqu.jpg |
| `bottle_of_pop__wuuoes` | `z2` | `z4 z3` | z4: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 0.6); z3: geom exact (mean 0.38% frac2 0.0%), tex borderline (MAD 1.4) | runlogs/symmetry_20260915/renders/bottle_of_pop__wuuoes.jpg |
| `bottle_of_pop__zjbqvb` | `z2` | `z4` | z4: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 0.4) | runlogs/symmetry_20260915/renders/bottle_of_pop__zjbqvb.jpg |
| `case_of_eyeshadow__zgervc` | `none` | `y2` | y2: geom near (mean 0.55% frac2 0.1%), tex preserved (MAD 0.1) | runlogs/symmetry_20260915/renders/case_of_eyeshadow__zgervc.jpg |
| `hardback__acaxtl` | `none` | `y2` | y2: geom exact (mean 0.13% frac2 0.0%), tex preserved (MAD 0.7) | runlogs/symmetry_20260915/renders/hardback__acaxtl.jpg |
| `pack_of_chocolate_bar__bxgewr` | `none` | `z2` | z2: geom exact (mean 0.07% frac2 0.0%), tex preserved (MAD 6.4) | runlogs/symmetry_20260915/renders/pack_of_chocolate_bar__bxgewr.jpg |
| `can_of_corn__alphlq` | `none` | `z2` | z2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 2.3) | runlogs/symmetry_20260915/renders/can_of_corn__alphlq.jpg |
| `can_of_corn__kwwlfn` | `none` | `z2` | z2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 3.7) | runlogs/symmetry_20260915/renders/can_of_corn__kwwlfn.jpg |
| `marker__ablbdc` | `none` | `x4 x5 x6` | x4: geom near (mean 0.60% frac2 2.0%), tex borderline (MAD 0.5); x5: geom near (mean 0.54% frac2 0.0%), tex preserved (MAD 0.5); x6: geom near (mean 0.49% frac2 0.2%), tex preserved (MAD 0.5) | runlogs/symmetry_20260915/renders/marker__ablbdc.jpg |
| `marker__bwwkmb` | `none` | `x6` | x6: geom near (mean 0.56% frac2 0.1%), tex borderline (MAD 0.8) | runlogs/symmetry_20260915/renders/marker__bwwkmb.jpg |
| `pen__gzrbrx` | `none` | `xinf` | xinf: geom exact (mean 0.40% frac2 0.0%), tex borderline (MAD 0.9) | runlogs/symmetry_20260915/renders/pen__gzrbrx.jpg |
| `bottle_of_fennel__ihlkfu` | `none` | `zinf` | zinf: geom exact (mean 0.20% frac2 0.7%), tex borderline (MAD 3.5) | runlogs/symmetry_20260915/renders/bottle_of_fennel__ihlkfu.jpg |
| `bottle_of_ginger__drqhzo` | `none` | `z6` | z6: geom near (mean 0.24% frac2 1.9%), tex borderline (MAD 2.3) | runlogs/symmetry_20260915/renders/bottle_of_ginger__drqhzo.jpg |
| `bottle_of_ground_cloves__vzamzb` | `z4 z5 z6` | `zinf` | zinf: geom exact (mean 0.20% frac2 0.7%), tex borderline (MAD 2.4) | runlogs/symmetry_20260915/renders/bottle_of_ground_cloves__vzamzb.jpg |
| `bottle_of_ground_nutmeg__qebruq` | `none` | `z5 z6` | z5: geom near (mean 0.25% frac2 2.0%), tex borderline (MAD 2.3); z6: geom near (mean 0.24% frac2 1.9%), tex borderline (MAD 2.2) | runlogs/symmetry_20260915/renders/bottle_of_ground_nutmeg__qebruq.jpg |
| `bottle_of_paprika__nrdgrp` | `z4 z5 z6` | `zinf` | zinf: geom exact (mean 0.20% frac2 0.7%), tex borderline (MAD 2.4) | runlogs/symmetry_20260915/renders/bottle_of_paprika__nrdgrp.jpg |
| `bottle_of_sage__bterim` | `none` | `z4 z5 z6` | z4: geom exact (mean 0.46% frac2 0.0%), tex borderline (MAD 9.9); z5: geom exact (mean 0.42% frac2 0.1%), tex borderline (MAD 9.8); z6: geom exact (mean 0.40% frac2 0.2%), tex borderline (MAD 9.8) | runlogs/symmetry_20260915/renders/bottle_of_sage__bterim.jpg |
| `bottle_of_seasoning__vqwqqv` | `none` | `z5 z6` | z5: geom near (mean 0.25% frac2 2.0%), tex borderline (MAD 2.3); z6: geom near (mean 0.24% frac2 1.9%), tex borderline (MAD 2.2) | runlogs/symmetry_20260915/renders/bottle_of_seasoning__vqwqqv.jpg |
| `bottle_of_sesame_seeds__lzndie` | `none` | `zinf` | zinf: geom exact (mean 0.22% frac2 0.7%), tex borderline (MAD 4.0) | runlogs/symmetry_20260915/renders/bottle_of_sesame_seeds__lzndie.jpg |
| `bottle_of_tea_leaves__pfbtus` | `none` | `zinf` | zinf: geom exact (mean 0.20% frac2 0.7%), tex borderline (MAD 9.4) | runlogs/symmetry_20260915/renders/bottle_of_tea_leaves__pfbtus.jpg |
| `jar_of_curry_powder__ozhftu` | `none` | `z4 z5 z6` | z4: geom near (mean 0.24% frac2 2.0%), tex borderline (MAD 2.1); z5: geom near (mean 0.24% frac2 2.0%), tex borderline (MAD 2.0); z6: geom near (mean 0.24% frac2 1.9%), tex borderline (MAD 2.0) | runlogs/symmetry_20260915/renders/jar_of_curry_powder__ozhftu.jpg |
| `jar_of_dill_seed__reavkt` | `none` | `zinf` | zinf: geom near (mean 0.60% frac2 1.7%), tex borderline (MAD 6.6) | runlogs/symmetry_20260915/renders/jar_of_dill_seed__reavkt.jpg |
| `bottle_of_coconut_water__lsixio` | `none` | `z2` | z2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 2.9) | runlogs/symmetry_20260915/renders/bottle_of_coconut_water__lsixio.jpg |
| `paper_liners__yvuilk` | `z4 z5 z6` | `zinf` | zinf: geom near (mean 0.69% frac2 2.3%), tex preserved (MAD 0.1) | runlogs/symmetry_20260915/renders/paper_liners__yvuilk.jpg |
| `biscuit__ukcwqw` | `none` | `z2` | z2: geom exact (mean 0.08% frac2 0.0%), tex preserved (MAD 5.7) | runlogs/symmetry_20260915/renders/biscuit__ukcwqw.jpg |
| `butter_cookie__kukrla` | `z2` | `x2 zinf` | x2: geom near (mean 0.61% frac2 1.4%), tex preserved (MAD 6.4); zinf: geom near (mean 0.30% frac2 1.6%), tex preserved (MAD 4.3) | runlogs/symmetry_20260915/renders/butter_cookie__kukrla.jpg |
| `coaster__arjpcz` | `none` | `x2 zinf` | x2: geom exact (mean 0.07% frac2 0.0%), tex preserved (MAD 4.1); zinf: geom exact (mean 0.07% frac2 0.0%), tex preserved (MAD 4.1) | runlogs/symmetry_20260915/renders/coaster__arjpcz.jpg |
| `sugar_cookie__cgkiti` | `none` | `x2 zinf` | x2: geom near (mean 0.51% frac2 0.0%), tex borderline (MAD 16.2); zinf: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 7.6) | runlogs/symmetry_20260915/renders/sugar_cookie__cgkiti.jpg |
| `sponge__klwueh` | `none` | `z2` | z2: geom exact (mean 0.09% frac2 0.0%), tex borderline (MAD 7.0) | runlogs/symmetry_20260915/renders/sponge__klwueh.jpg |
| `sugar_cookie__hlyeaq` | `none` | `x2 zinf` | x2: geom near (mean 0.51% frac2 0.0%), tex borderline (MAD 16.2); zinf: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 7.7) | runlogs/symmetry_20260915/renders/sugar_cookie__hlyeaq.jpg |
| `asparagus__eodozo` | `none` | `y6` | y6: geom exact (mean 0.38% frac2 0.1%), tex borderline (MAD 24.0) | runlogs/symmetry_20260915/renders/asparagus__eodozo.jpg |
| `carrot__aucrah` | `none` | `x6` | x6: geom exact (mean 0.38% frac2 0.0%), tex borderline (MAD 7.5) | runlogs/symmetry_20260915/renders/carrot__aucrah.jpg |
| `green_bean__dyjvwv` | `none` | `x2` | x2: geom near (mean 0.58% frac2 1.0%), tex borderline (MAD 9.0) | runlogs/symmetry_20260915/renders/green_bean__dyjvwv.jpg |
| `green_bean__lvbjws` | `none` | `x2` | x2: geom near (mean 0.56% frac2 0.9%), tex borderline (MAD 9.0) | runlogs/symmetry_20260915/renders/green_bean__lvbjws.jpg |
| `asparagus__xguktb` | `none` | `y6` | y6: geom near (mean 0.58% frac2 0.7%), tex borderline (MAD 23.8) | runlogs/symmetry_20260915/renders/asparagus__xguktb.jpg |
| `carrot__qhmmmx` | `none` | `x4 x3 x5` | x4: geom near (mean 0.49% frac2 2.2%), tex borderline (MAD 3.4); x3: geom near (mean 0.56% frac2 2.9%), tex borderline (MAD 4.3); x5: geom near (mean 0.44% frac2 2.7%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/carrot__qhmmmx.jpg |
| `cork__lseuwf` | `none` | `zinf` | zinf: geom exact (mean 0.31% frac2 0.0%), tex preserved (MAD 6.0) | runlogs/symmetry_20260915/renders/cork__lseuwf.jpg |
| `cork__ncxgpe` | `none` | `zinf` | zinf: geom exact (mean 0.20% frac2 0.0%), tex preserved (MAD 6.1) | runlogs/symmetry_20260915/renders/cork__ncxgpe.jpg |
| `hamburger_bun__fqxumh` | `none` | `z4 z5 z6` | z4: geom near (mean 0.56% frac2 0.3%), tex borderline (MAD 6.8); z5: geom exact (mean 0.48% frac2 0.2%), tex borderline (MAD 6.7); z6: geom exact (mean 0.44% frac2 0.2%), tex borderline (MAD 6.7) | runlogs/symmetry_20260915/renders/hamburger_bun__fqxumh.jpg |
| `canister__xcppkc` | `none` | `z2` | z2: geom exact (mean 0.12% frac2 0.0%), tex preserved (MAD 0.1) | runlogs/symmetry_20260915/renders/canister__xcppkc.jpg |
| `cap__iqeyba` | `z4 z5 z6` | `zinf` | zinf: geom exact (mean 0.19% frac2 0.0%), tex borderline (MAD 4.4) | runlogs/symmetry_20260915/renders/cap__iqeyba.jpg |
| `dip_candle__poexjo` | `none` | `x2 z4 z5 z6` | x2: geom near (mean 0.29% frac2 1.9%), tex borderline (MAD 0.3); z4: geom near (mean 0.59% frac2 0.3%), tex borderline (MAD 0.3); z5: geom near (mean 0.49% frac2 0.1%), tex borderline (MAD 0.3); z6: geom exact (mean 0.42% frac2 0.0%), tex borderline (MAD 0.3) | runlogs/symmetry_20260915/renders/dip_candle__poexjo.jpg |
| `dip_candle__ghctun` | `none` | `zinf` | zinf: geom exact (mean 0.09% frac2 0.0%), tex preserved (MAD 0.1) | runlogs/symmetry_20260915/renders/dip_candle__ghctun.jpg |
| `dip_candle__gbaons` | `zinf` | `x2 y2` | x2: geom near (mean 0.25% frac2 2.6%), tex preserved (MAD 0.3); y2: geom near (mean 0.25% frac2 2.6%), tex preserved (MAD 0.3) | runlogs/symmetry_20260915/renders/dip_candle__gbaons.jpg |
| `dip_candle__mxjejs` | `none` | `x2 zinf` | x2: geom exact (mean 0.10% frac2 0.4%), tex borderline (MAD 0.2); zinf: geom exact (mean 0.05% frac2 0.0%), tex preserved (MAD 0.1) | runlogs/symmetry_20260915/renders/dip_candle__mxjejs.jpg |
| `cabinet_base__dcmhhh` | `none` | `x2` | x2: geom exact (mean 0.17% frac2 0.0%), tex borderline (MAD 8.6) | runlogs/symmetry_20260915/renders/cabinet_base__dcmhhh.jpg |
| `letter__bdlpry` | `none` | `z2` | z2: geom exact (mean 0.39% frac2 0.0%), tex borderline (MAD 15.9) | runlogs/symmetry_20260915/renders/letter__bdlpry.jpg |
| `letter__igvnic` | `none` | `z2` | z2: geom exact (mean 0.06% frac2 0.0%), tex borderline (MAD 15.9) | runlogs/symmetry_20260915/renders/letter__igvnic.jpg |
| `letter__jafvfh` | `none` | `z2` | z2: geom exact (mean 0.06% frac2 0.0%), tex borderline (MAD 14.4) | runlogs/symmetry_20260915/renders/letter__jafvfh.jpg |
| `keyboard__uwacaq` | `none` | `z2` | z2: geom exact (mean 0.17% frac2 0.0%), tex preserved (MAD 3.6) | runlogs/symmetry_20260915/renders/keyboard__uwacaq.jpg |
| `clipboard__gmxyfo` | `none` | `x2 z2` | x2: geom exact (mean 0.10% frac2 0.0%), tex preserved (MAD 0.6); z2: geom exact (mean 0.09% frac2 0.0%), tex preserved (MAD 0.9) | runlogs/symmetry_20260915/renders/clipboard__gmxyfo.jpg |
| `rolling_pin__mwrdae` | `none` | `x2 yinf` | x2: geom exact (mean 0.05% frac2 0.0%), tex borderline (MAD 13.7); yinf: geom exact (mean 0.09% frac2 0.0%), tex borderline (MAD 14.9) | runlogs/symmetry_20260915/renders/rolling_pin__mwrdae.jpg |
| `rolling_pin__smnklu` | `x2 z2 y4 y5 y6` | `yinf` | yinf: geom exact (mean 0.06% frac2 0.0%), tex borderline (MAD 3.5) | runlogs/symmetry_20260915/renders/rolling_pin__smnklu.jpg |
| `drumstick__lkvzmj` | `y2 z2 x4 x5 x6` | `xinf` | xinf: geom exact (mean 0.04% frac2 0.0%), tex preserved (MAD 6.6) | runlogs/symmetry_20260915/renders/drumstick__lkvzmj.jpg |
| `towel_rack__quznqx` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.13% frac2 0.0%), tex preserved (MAD 4.1); y2: geom exact (mean 0.03% frac2 0.0%), tex preserved (MAD 3.5); z2: geom exact (mean 0.13% frac2 0.0%), tex preserved (MAD 3.6) | runlogs/symmetry_20260915/renders/towel_rack__quznqx.jpg |
| `decorative_sign__hgonwj` | `none` | `x2` | x2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 0.1) | runlogs/symmetry_20260915/renders/decorative_sign__hgonwj.jpg |
| `ruler__ujizkx` | `none` | `y2` | y2: geom exact (mean 0.03% frac2 0.0%), tex borderline (MAD 18.3) | runlogs/symmetry_20260915/renders/ruler__ujizkx.jpg |
| `pack_of_kielbasa__fwqdoy` | `none` | `x2` | x2: geom exact (mean 0.06% frac2 0.0%), tex borderline (MAD 0.5) | runlogs/symmetry_20260915/renders/pack_of_kielbasa__fwqdoy.jpg |
| `pen__hiyeki` | `x4 x3 x5` | `xinf` | xinf: geom near (mean 0.53% frac2 0.0%), tex borderline (MAD 1.8) | runlogs/symmetry_20260915/renders/pen__hiyeki.jpg |
| `pen__enfumt` | `none` | `x2` | x2: geom near (mean 0.78% frac2 2.1%), tex borderline (MAD 0.1) | runlogs/symmetry_20260915/renders/pen__enfumt.jpg |
| `cutting_board__tcdrzs` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 14.4); y2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 6.5); z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 14.3) | runlogs/symmetry_20260915/renders/cutting_board__tcdrzs.jpg |
| `cutting_board__nsvnai` | `y2` | `x2 z2` | x2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 11.8); z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 11.7) | runlogs/symmetry_20260915/renders/cutting_board__nsvnai.jpg |
| `cutting_board__jfrbuz` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 11.3); y2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 11.2); z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 9.8) | runlogs/symmetry_20260915/renders/cutting_board__jfrbuz.jpg |
| `cutting_board__aibvew` | `none` | `x2` | x2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 15.1) | runlogs/symmetry_20260915/renders/cutting_board__aibvew.jpg |
| `plate__luhkiz` | `none` | `z4 z6` | z4: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 2.8); z6: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 2.8) | runlogs/symmetry_20260915/renders/plate__luhkiz.jpg |
| `plate__mtetqm` | `z2` | `z4` | z4: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 0.1) | runlogs/symmetry_20260915/renders/plate__mtetqm.jpg |
| `french_fries__oeterg` | `none` | `z2` | z2: geom near (mean 0.81% frac2 2.9%), tex borderline (MAD 11.7) | runlogs/symmetry_20260915/renders/french_fries__oeterg.jpg |
| `french_fries__nfgqwu` | `none` | `z2` | z2: geom exact (mean 0.45% frac2 0.6%), tex borderline (MAD 10.5) | runlogs/symmetry_20260915/renders/french_fries__nfgqwu.jpg |
| `french_fries__uqsuef` | `none` | `z2` | z2: geom near (mean 0.54% frac2 1.3%), tex borderline (MAD 17.3) | runlogs/symmetry_20260915/renders/french_fries__uqsuef.jpg |
| `tray__xzcnjq` | `none` | `z2` | z2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 21.1) | runlogs/symmetry_20260915/renders/tray__xzcnjq.jpg |
| `tray__zcmnji` | `none` | `zinf` | zinf: geom exact (mean 0.07% frac2 0.0%), tex preserved (MAD 2.3) | runlogs/symmetry_20260915/renders/tray__zcmnji.jpg |
| `half_lime__dtyjkb` | `none` | `z6` | z6: geom near (mean 0.52% frac2 0.0%), tex borderline (MAD 12.2) | runlogs/symmetry_20260915/renders/half_lime__dtyjkb.jpg |
| `half_lime__wdadtn` | `none` | `z2` | z2: geom near (mean 0.54% frac2 0.0%), tex borderline (MAD 9.5) | runlogs/symmetry_20260915/renders/half_lime__wdadtn.jpg |
| `half_lime__mxmmcp` | `none` | `zinf` | zinf: geom near (mean 0.42% frac2 1.1%), tex borderline (MAD 12.9) | runlogs/symmetry_20260915/renders/half_lime__mxmmcp.jpg |
| `bowl__jpvcjv` | `none` | `z3` | z3: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 13.3) | runlogs/symmetry_20260915/renders/bowl__jpvcjv.jpg |
| `saucepot__tfzijn` | `none` | `z2` | z2: geom exact (mean 0.18% frac2 0.0%), tex preserved (MAD 0.3) | runlogs/symmetry_20260915/renders/saucepot__tfzijn.jpg |
| `saucepot__uvzmss` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex preserved (MAD 0.2) | runlogs/symmetry_20260915/renders/saucepot__uvzmss.jpg |
| `vase__stqkvx` | `z6` | `zinf` | zinf: geom near (mean 0.58% frac2 0.0%), tex preserved (MAD 0.1) | runlogs/symmetry_20260915/renders/vase__stqkvx.jpg |
| `cinnamon_stick__qxpzdm` | `none` | `x2 z2` | x2: geom near (mean 0.41% frac2 0.0%), tex borderline (MAD 3.7); z2: geom exact (mean 0.41% frac2 0.0%), tex preserved (MAD 3.6) | runlogs/symmetry_20260915/renders/cinnamon_stick__qxpzdm.jpg |
| `cinnamon_stick__kdaxdy` | `none` | `x2 y2 z2` | x2: geom near (mean 0.42% frac2 0.9%), tex preserved (MAD 4.0); y2: geom near (mean 0.51% frac2 0.4%), tex preserved (MAD 2.1); z2: geom near (mean 0.53% frac2 0.6%), tex preserved (MAD 4.0) | runlogs/symmetry_20260915/renders/cinnamon_stick__kdaxdy.jpg |
| `cinnamon_stick__qmlyim` | `none` | `x2` | x2: geom near (mean 0.47% frac2 0.8%), tex borderline (MAD 3.7) | runlogs/symmetry_20260915/renders/cinnamon_stick__qmlyim.jpg |
| `cinnamon_stick__qsqvgk` | `none` | `z2` | z2: geom exact (mean 0.35% frac2 0.0%), tex preserved (MAD 3.4) | runlogs/symmetry_20260915/renders/cinnamon_stick__qsqvgk.jpg |
| `cinnamon_stick__cdkjfo` | `none` | `x2 z2 y4 y5 y6` | x2: geom near (mean 0.51% frac2 0.6%), tex borderline (MAD 3.4); z2: geom exact (mean 0.44% frac2 0.0%), tex preserved (MAD 3.4); y4: geom near (mean 0.64% frac2 0.5%), tex preserved (MAD 2.7); y5: geom near (mean 0.60% frac2 0.5%), tex preserved (MAD 2.9); y6: geom near (mean 0.55% frac2 0.6%), tex preserved (MAD 3.1) | runlogs/symmetry_20260915/renders/cinnamon_stick__cdkjfo.jpg |
| `cinnamon_stick__smfuqz` | `none` | `z2 y6` | z2: geom exact (mean 0.37% frac2 0.0%), tex preserved (MAD 3.5); y6: geom near (mean 0.53% frac2 1.0%), tex borderline (MAD 3.4) | runlogs/symmetry_20260915/renders/cinnamon_stick__smfuqz.jpg |
| `cinnamon_stick__hjhcpm` | `none` | `z2` | z2: geom near (mean 0.52% frac2 0.5%), tex preserved (MAD 3.7) | runlogs/symmetry_20260915/renders/cinnamon_stick__hjhcpm.jpg |
| `pencil__hmawzy` | `x5 x6` | `z2 x4` | z2: geom exact (mean 0.10% frac2 0.6%), tex borderline (MAD 3.3); x4: geom exact (mean 0.36% frac2 0.0%), tex preserved (MAD 1.5) | runlogs/symmetry_20260915/renders/pencil__hmawzy.jpg |
| `pencil__rzscao` | `xinf` | `y2 z2` | y2: geom exact (mean 0.26% frac2 0.7%), tex preserved (MAD 4.1); z2: geom exact (mean 0.08% frac2 0.7%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/pencil__rzscao.jpg |
| `pencil__heqkmr` | `x6` | `y2 xinf` | y2: geom exact (mean 0.45% frac2 0.7%), tex borderline (MAD 7.9); xinf: geom exact (mean 0.43% frac2 0.0%), tex preserved (MAD 2.5) | runlogs/symmetry_20260915/renders/pencil__heqkmr.jpg |
| `pencil__lruiot` | `x5 x6` | `y2 xinf` | y2: geom exact (mean 0.35% frac2 0.7%), tex borderline (MAD 6.6); xinf: geom exact (mean 0.33% frac2 0.0%), tex preserved (MAD 1.9) | runlogs/symmetry_20260915/renders/pencil__lruiot.jpg |
| `pencil__lykqoh` | `x4 x5 x6` | `y2 xinf` | y2: geom exact (mean 0.35% frac2 0.7%), tex preserved (MAD 5.7); xinf: geom exact (mean 0.33% frac2 0.0%), tex preserved (MAD 1.7) | runlogs/symmetry_20260915/renders/pencil__lykqoh.jpg |
| `pencil__vpzhth` | `x4 x5 x6` | `y2 xinf` | y2: geom exact (mean 0.45% frac2 0.7%), tex preserved (MAD 5.8); xinf: geom exact (mean 0.43% frac2 0.0%), tex preserved (MAD 1.9) | runlogs/symmetry_20260915/renders/pencil__vpzhth.jpg |
| `pencil__vvofwr` | `x5 x6` | `y2 xinf` | y2: geom exact (mean 0.40% frac2 0.7%), tex preserved (MAD 4.5); xinf: geom exact (mean 0.38% frac2 0.0%), tex preserved (MAD 1.1) | runlogs/symmetry_20260915/renders/pencil__vvofwr.jpg |
| `pencil__mvmxfo` | `none` | `xinf` | xinf: geom exact (mean 0.44% frac2 0.0%), tex preserved (MAD 2.5) | runlogs/symmetry_20260915/renders/pencil__mvmxfo.jpg |
| `marker__xakoic` | `none` | `xinf` | xinf: geom near (mean 0.86% frac2 1.6%), tex borderline (MAD 1.1) | runlogs/symmetry_20260915/renders/marker__xakoic.jpg |
| `marker__gldosk` | `none` | `x4 x5 x6` | x4: geom near (mean 0.57% frac2 0.8%), tex borderline (MAD 1.3); x5: geom near (mean 0.52% frac2 0.0%), tex borderline (MAD 1.1); x6: geom near (mean 0.47% frac2 0.2%), tex borderline (MAD 1.1) | runlogs/symmetry_20260915/renders/marker__gldosk.jpg |
| `marker__vrjwkg` | `none` | `x3` | x3: geom near (mean 0.80% frac2 0.6%), tex borderline (MAD 1.0) | runlogs/symmetry_20260915/renders/marker__vrjwkg.jpg |
| `marker__vjjsul` | `none` | `x4 x3 x5` | x4: geom near (mean 0.67% frac2 0.6%), tex borderline (MAD 1.4); x3: geom near (mean 0.77% frac2 0.2%), tex borderline (MAD 1.6); x5: geom near (mean 0.60% frac2 0.9%), tex borderline (MAD 1.3) | runlogs/symmetry_20260915/renders/marker__vjjsul.jpg |
| `marker__oyjbhy` | `none` | `xinf` | xinf: geom near (mean 0.70% frac2 2.3%), tex borderline (MAD 1.8) | runlogs/symmetry_20260915/renders/marker__oyjbhy.jpg |
| `marker__mznenj` | `none` | `xinf` | xinf: geom near (mean 0.81% frac2 0.3%), tex borderline (MAD 1.3) | runlogs/symmetry_20260915/renders/marker__mznenj.jpg |
| `pepper_shaker__tgcwtr` | `none` | `z5 z6` | z5: geom exact (mean 0.23% frac2 0.0%), tex borderline (MAD 7.5); z6: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 7.5) | runlogs/symmetry_20260915/renders/pepper_shaker__tgcwtr.jpg |
| `pepper_shaker__ykujyn` | `z4 z5 z6` | `zinf` | zinf: geom exact (mean 0.24% frac2 0.0%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/pepper_shaker__ykujyn.jpg |
| `pepper_shaker__stbqbt` | `none` | `z5 z6` | z5: geom near (mean 0.55% frac2 2.1%), tex preserved (MAD 0.8); z6: geom near (mean 0.52% frac2 0.1%), tex preserved (MAD 0.9) | runlogs/symmetry_20260915/renders/pepper_shaker__stbqbt.jpg |
| `salt_shaker__iomwtn` | `none` | `z4 z5 z6` | z4: geom exact (mean 0.17% frac2 0.0%), tex borderline (MAD 7.5); z5: geom exact (mean 0.15% frac2 0.0%), tex borderline (MAD 9.1); z6: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 9.4) | runlogs/symmetry_20260915/renders/salt_shaker__iomwtn.jpg |
| `salt_shaker__fonexa` | `none` | `zinf` | zinf: geom exact (mean 0.37% frac2 0.0%), tex preserved (MAD 0.2) | runlogs/symmetry_20260915/renders/salt_shaker__fonexa.jpg |
| `carrot__nktmff` | `none` | `x2` | x2: geom near (mean 0.61% frac2 0.3%), tex preserved (MAD 7.1) | runlogs/symmetry_20260915/renders/carrot__nktmff.jpg |
| `colored_pencil__acumgp` | `x5 x6` | `z2` | z2: geom exact (mean 0.09% frac2 0.8%), tex borderline (MAD 4.0) | runlogs/symmetry_20260915/renders/colored_pencil__acumgp.jpg |
| `colored_pencil__bkdqwb` | `x5 x6` | `z2` | z2: geom exact (mean 0.09% frac2 0.8%), tex borderline (MAD 5.5) | runlogs/symmetry_20260915/renders/colored_pencil__bkdqwb.jpg |
| `colored_pencil__deuvcx` | `x6` | `x4 x5` | x4: geom exact (mean 0.36% frac2 0.0%), tex borderline (MAD 2.0); x5: geom exact (mean 0.29% frac2 0.0%), tex borderline (MAD 1.8) | runlogs/symmetry_20260915/renders/colored_pencil__deuvcx.jpg |
| `colored_pencil__egvqng` | `x5 x6` | `z2` | z2: geom exact (mean 0.09% frac2 0.8%), tex borderline (MAD 3.6) | runlogs/symmetry_20260915/renders/colored_pencil__egvqng.jpg |
| `colored_pencil__jdqvdl` | `x5 x6` | `z2` | z2: geom exact (mean 0.09% frac2 0.8%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/colored_pencil__jdqvdl.jpg |
| `colored_pencil__jnccuz` | `x5 x6` | `z2` | z2: geom exact (mean 0.08% frac2 0.7%), tex borderline (MAD 4.1) | runlogs/symmetry_20260915/renders/colored_pencil__jnccuz.jpg |
| `colored_pencil__kadvlg` | `x5 x6` | `z2` | z2: geom exact (mean 0.08% frac2 0.8%), tex borderline (MAD 5.0) | runlogs/symmetry_20260915/renders/colored_pencil__kadvlg.jpg |
| `colored_pencil__qrxemk` | `x5 x6` | `z2` | z2: geom exact (mean 0.09% frac2 0.8%), tex borderline (MAD 5.4) | runlogs/symmetry_20260915/renders/colored_pencil__qrxemk.jpg |
| `salt_shaker__qowouj` | `z4 z6` | `zinf` | zinf: geom near (mean 0.52% frac2 0.0%), tex preserved (MAD 0.5) | runlogs/symmetry_20260915/renders/salt_shaker__qowouj.jpg |
| `can_of_baking_mix__rxorlp` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 2.0) | runlogs/symmetry_20260915/renders/can_of_baking_mix__rxorlp.jpg |
