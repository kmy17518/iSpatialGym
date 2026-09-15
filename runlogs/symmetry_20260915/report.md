## Per-tag counts (1480 models surveyed, 0 loader failures)

| tag | `symmetry` (final) | `symmetry_near` (review) | geometry exact (before texture) |
|---|---|---|---|
| `x2` | 47 | 82 | 278 |
| `y2` | 28 | 60 | 267 |
| `z2` | 117 | 78 | 364 |
| `x4` | 24 | 9 | 31 |
| `y4` | 3 | 4 | 17 |
| `z4` | 41 | 20 | 82 |
| `x3` | 0 | 0 | 1 |
| `y3` | 0 | 0 | 2 |
| `z3` | 4 | 5 | 5 |
| `x5` | 23 | 12 | 22 |
| `y5` | 0 | 1 | 1 |
| `z5` | 13 | 14 | 55 |
| `x6` | 27 | 17 | 24 |
| `y6` | 0 | 0 | 1 |
| `z6` | 17 | 16 | 64 |
| `xinf` | 13 | 29 | 42 |
| `yinf` | 1 | 4 | 2 |
| `zinf` | 132 | 76 | 266 |
| `full` | 1 | 2 | 2 |
| `none` | 1103 | 0 | 624 |

Models with a final symmetry: 377; asymmetric (`none`): 1103; with a `symmetry_near` entry to review: 302; with at least one geometric candidate (contact sheet rendered): 965.

## Scaled variants

41 models have a grasp-tested spawn scale; the symmetry at that scale differs from the canonical model for 21:

- `french_fries__oeterg` scale [1.0, 1.871, 2.385]: canonical `none` / near `z2` -> scaled `none` / near `-`
- `french_fries__nfgqwu` scale [1.0, 1.891, 2.458]: canonical `none` / near `z2` -> scaled `none` / near `-`
- `french_fries__uqsuef` scale [1.0, 2.155, 2.458]: canonical `none` / near `z2` -> scaled `none` / near `-`
- `cinnamon_stick__qxpzdm` scale [1.0, 1.0, 1.968]: canonical `none` / near `x2 y2 z2` -> scaled `none` / near `x2 yinf`
- `cinnamon_stick__kdaxdy` scale [1.0, 1.0, 1.87]: canonical `none` / near `x2 y2 z2` -> scaled `none` / near `y2`
- `cinnamon_stick__qmlyim` scale [1.054, 1.0, 1.326]: canonical `none` / near `x2 z2` -> scaled `none` / near `x2`
- `cinnamon_stick__qsqvgk` scale [1.056, 1.0, 1.876]: canonical `none` / near `x2 y2 z2` -> scaled `none` / near `z2 y6`
- `cinnamon_stick__cdkjfo` scale [1.647, 1.0, 2.423]: canonical `none` / near `x2 yinf` -> scaled `none` / near `-`
- `cinnamon_stick__smfuqz` scale [1.207, 1.0, 1.8]: canonical `none` / near `x2 yinf` -> scaled `none` / near `x2 z2 y5 y6`
- `cinnamon_stick__hjhcpm` scale [1.0, 1.0, 2.508]: canonical `none` / near `z2` -> scaled `none` / near `-`
- `pencil__hmawzy` scale [1.0, 3.231, 4.998]: canonical `z2 x5 x6` / near `-` -> scaled `none` / near `z2`
- `pencil__rzscao` scale [1.0, 3.867, 5.113]: canonical `xinf` / near `y2 z2` -> scaled `none` / near `z2`
- `pencil__gbzzcg` scale [1.0, 3.901, 5.106]: canonical `xinf` / near `-` -> scaled `none` / near `-`
- `pencil__fnlbpu` scale [1.0, 4.217, 5.069]: canonical `xinf` / near `-` -> scaled `none` / near `xinf`
- `pencil__heqkmr` scale [1.0, 3.345, 5.081]: canonical `x6` / near `-` -> scaled `none` / near `-`
- `pencil__jntybl` scale [1.0, 3.96, 5.098]: canonical `xinf` / near `-` -> scaled `none` / near `-`
- `pencil__lruiot` scale [1.0, 3.603, 5.085]: canonical `x5 x6` / near `xinf` -> scaled `none` / near `-`
- `pencil__lykqoh` scale [1.0, 3.596, 5.119]: canonical `x4 x5 x6` / near `-` -> scaled `none` / near `-`
- `pencil__vpzhth` scale [1.0, 3.361, 5.096]: canonical `x4 x5 x6` / near `-` -> scaled `none` / near `-`
- `pencil__vvofwr` scale [1.0, 3.473, 5.014]: canonical `x5 x6` / near `xinf` -> scaled `none` / near `-`
- `pencil__vedvit` scale [1.0, 3.451, 5.058]: canonical `z2 xinf` / near `-` -> scaled `none` / near `z2`

## Sanity table

| category | models | most common `symmetry` | most common geometry-exact | expectation | match |
|---|---|---|---|---|---|
| bowl | 12 | `zinf` x6, `none` x3, `z4` x2 | `zinf` x9, `z4` x2, `z2` x1 | zinf (plain bowls; a decorated rim breaks it) | |
| plate | 12 | `zinf` x5, `z4` x3, `none` x3 | `zinf` x9, `z4` x3 | zinf | |
| mug | 10 | `none` x10 | `none` x10 | none (handle) | |
| teacup | 6 | `none` x3, `zinf` x2, `z5` x1 | `zinf` x3, `none` x3 | zinf if handle-less, else none | |
| teapot | 3 | `none` x3 | `none` x3 | none | |
| sunflower | 17 | `none` x17 | `none` x17 | none | |
| pencil | 13 | `xinf` x5, `x5 x6` x2, `x4 x5 x6` x2 | `y2 xinf` x12, `z2 x4 x5 x6` x1 | xinf/yinf/zinf about the long axis if unprinted; a print breaks it | |
| bottle_of_medicine | 8 | `none` x8 | `zinf` x4, `none` x2, `z5 z6` x1 | zinf geometry; the label breaks the texture -> none | |
| can_of_baking_mix | 6 | `z2` x6 | `zinf` x3, `x2 zinf` x3 | zinf geometry; label -> none | |
| can_of_soda | 9 | `none` x6, `zinf` x1, `z2` x1 | `z4` x4, `zinf` x3, `z2` x1 | zinf geometry; label -> none | |
| board_game | 16 | `none` x16 | `x2 y2 z2` x13, `none` x1, `x4 y4 z4` x1 | x2 y2 z2 box geometry; artwork -> none | |
| box_of_cookies | 14 | `none` x12, `x2` x1, `y2` x1 | `x2 y2 z2` x6, `x2 y2 z4` x4, `z2` x1 | box geometry; print -> none | |
| box_of_cereal | 14 | `none` x14 | `x2 y2 z2` x14 | box geometry; print -> none | |
| apple | 7 | `none` x7 | `none` x7 | none / near zinf (stem) | |
| orange | 2 | `x2 zinf` x1, `full` x1 | `x2 zinf` x1, `full` x1 | full or zinf geometry; texture decides | |
| tennis_ball | 1 | `y2 x4 z5 z6` x1 | `full` x1 | full (plain) or none (seam) | |
| toy_dice | 20 | `none` x19, `y2` x1 | `x2 z2 y4` x9, `none` x6, `x2 y2 z4` x3 | cube geometry x4 y4 z4; pips -> none | |
| jar_of_grains | 12 | `none` x12 | `none` x12 | zinf geometry; label -> none | |
| cap | 19 | `zinf` x13, `z2` x2, `z5 z6` x1 | `zinf` x12, `z2` x2, `z5 z6` x1 | zinf | |
| lid | 17 | `zinf` x9, `z2` x6, `none` x1 | `zinf` x9, `z2` x6, `x2 y2 z2` x1 | zinf | |
| keyboard | 6 | `none` x5, `z2` x1 | `none` x4, `y2` x1, `z2` x1 | none | |
| painting | 11 | `none` x11 | `x2 y2 z2` x6, `x2` x2, `x4` x2 | none | |
| beaker | 10 | `zinf` x6, `z2` x2, `z6` x1 | `zinf` x7, `z2` x2, `z4` x1 | zinf | |
| bar_soap | 4 | `x2 y2 z2` x2, `none` x1, `x2 zinf` x1 | `x2 y2 z2` x2, `none` x1, `x2 zinf` x1 | zinf or x2 y2 z2 | |
| wineglass | 12 | `zinf` x12 | `zinf` x12 | zinf | |
| egg | 1 | `none` x1 | `none` x1 | zinf or xinf (about the long axis) | |
| wall_clock | 5 | `none` x4, `x4` x1 | `xinf` x2, `y2 xinf` x2, `x4 x3` x1 | none (numerals) | |
| vase | 6 | `none` x2, `z4` x1, `zinf` x1 | `z4 z5 z6` x1, `none` x1, `z4` x1 | zinf unless decorated | |
| platter | 6 | `z2` x5, `z3` x1 | `z2` x5, `z3` x1 | zinf or x2 y2 z2 (oval) | |
| tray | 12 | `z2` x7, `none` x2, `x2 y2 z2` x1 | `z2` x9, `x2 y2 z2` x1, `zinf` x1 | x2 y2 z2 (plain rectangular) or none | |

## `mental_rotation` targets with a symmetry (5 of 196; exclude or give `orientation_equivalences`)

| category | model | `symmetry` | `symmetry_near` |
|---|---|---|---|
| teacup | zdvgol | `zinf` | `-` |
| teacup | cpozxi | `z5` | `z6` |
| teacup | tfzfam | `zinf` | `-` |
| hook | wmvozo | `zinf` | `-` |
| cookie_cutter | lqrfzo | `z2` | `-` |

`mental_rotation` targets with only a `symmetry_near` entry (review before use): 5: `toy_figure/jddrck` (z4), `pickle/ifterk` (y4), `pickle/iobrqp` (y4), `tongs/igsklx` (x2), `tongs/gbjqyn` (x2)

## `symmetric_180` (alignment targets) vs the survey

The half turn that matters is about the alignment axis: the wall normal = the frame's front axis for the wall subtasks, the
vertical (`frame_up`, else the canonical `z`) for `alignment-axial` (the object is yawed on the table). `symmetric_180` is only
set for `alignment-world_horizontal` targets in the selection sheet (empty = the reviewer did not call the object symmetric),
so every axial / vertical row below is a symmetry the reviewers did not record rather than a contradiction.

| subtask | category | model | half-turn axis | `symmetric_180` | survey half turn | survey `symmetry` | `symmetry_near` |
|---|---|---|---|---|---|---|---|
| alignment-axial | keyboard | uwacaq | +z (up) | - | yes | `z2` | `x2 y2` |
| alignment-axial | notebook | ffdjzp | +z (up) | - | yes | `x2 y2 z2` | `-` |
| alignment-axial | notebook | gqkaqu | +z (up) | - | yes | `x2 y2 z2` | `-` |
| alignment-axial | clipboard | gmxyfo | +z (up) | - | yes | `x2 y2 z2` | `-` |
| alignment-world_vertical | wall_clock | xevvel | +x (front) | - | yes | `x4` | `x6` |
| alignment-world_vertical | towel_rack | kqrmrh | +x (front) | - | yes | `x2` | `-` |
| alignment-world_horizontal | rolling_pin | mwrdae | +x (front) | 1 | near | `none` | `x2 yinf` |
| alignment-world_horizontal | shelf | qrmiur | +z (front) | 1 | near | `none` | `x2 y2 z2` |
| alignment-world_horizontal | shelf | iruewm | +z (front) | 1 | near | `none` | `x2 y2 z2` |
| alignment-world_horizontal | shelf | wvkefp | +z (front) | 1 | no | `none` | `-` |
| alignment-world_horizontal | radiator | dwmloz | +x (front) | - | yes | `x2` | `-` |
| alignment-world_horizontal | drumstick | jauzlc | +z (front) | - | yes | `z2 x4 x5 x6` | `y2 xinf` |
| alignment-world_horizontal | whiteboard | gvkgsi | +x (front) | - | yes | `x2` | `-` |
| alignment-world_horizontal | towel_rack | crivoa | +x (front) | - | yes | `x2` | `-` |
| alignment-world_horizontal | towel_rack | quznqx | +x (front) | 1 | near | `none` | `x2 yinf` |

15 alignment-target rows disagree (5 flagged `symmetric_180 = 1` without a confirmed survey half turn, 10 with a survey half turn the sheet does not record).

## Loader / render failures

Geometry stage: 0 failures; texture stage: 0 failures.


## Review list (302 models with a `symmetry_near` entry)

| model | `symmetry` | `symmetry_near` | why | sheet |
|---|---|---|---|---|
| `hamper__pdzaca` | `none` | `zinf` | zinf: geom near (mean 0.56% frac2 2.1%), tex borderline (MAD 4.1) | runlogs/symmetry_20260915/renders/hamper__pdzaca.jpg |
| `hamper__sijioo` | `none` | `z3 z5` | z3: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.8); z5: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.5) | runlogs/symmetry_20260915/renders/hamper__sijioo.jpg |
| `taboret__hugvml` | `none` | `z3` | z3: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 3.3) | runlogs/symmetry_20260915/renders/taboret__hugvml.jpg |
| `taboret__ehaclu` | `none` | `z2` | z2: geom near (mean 0.43% frac2 2.0%), tex preserved (MAD 1.1) | runlogs/symmetry_20260915/renders/taboret__ehaclu.jpg |
| `footstool__sghekm` | `none` | `z2` | z2: geom near (mean 0.47% frac2 0.0%), tex borderline (MAD 4.9) | runlogs/symmetry_20260915/renders/footstool__sghekm.jpg |
| `public_trash_can__hxsyxo` | `none` | `z5 z6` | z5: geom near (mean 0.23% frac2 2.4%), tex preserved (MAD 0.9); z6: geom near (mean 0.21% frac2 2.0%), tex preserved (MAD 1.0) | runlogs/symmetry_20260915/renders/public_trash_can__hxsyxo.jpg |
| `public_trash_can__leazin` | `z4 z5` | `zinf` | zinf: geom exact (mean 0.42% frac2 0.0%), tex borderline (MAD 4.1) | runlogs/symmetry_20260915/renders/public_trash_can__leazin.jpg |
| `public_trash_can__aefcem` | `z4` | `zinf` | zinf: geom near (mean 0.41% frac2 1.1%), tex borderline (MAD 3.2) | runlogs/symmetry_20260915/renders/public_trash_can__aefcem.jpg |
| `public_trash_can__xkqkbf` | `z2` | `zinf` | zinf: geom exact (mean 0.31% frac2 0.7%), tex borderline (MAD 3.8) | runlogs/symmetry_20260915/renders/public_trash_can__xkqkbf.jpg |
| `wicker_basket__jqfftz` | `none` | `z4` | z4: geom exact (mean 0.18% frac2 0.1%), tex borderline (MAD 3.8) | runlogs/symmetry_20260915/renders/wicker_basket__jqfftz.jpg |
| `wicker_basket__tsjvyu` | `none` | `z4` | z4: geom near (mean 0.35% frac2 2.5%), tex borderline (MAD 7.7) | runlogs/symmetry_20260915/renders/wicker_basket__tsjvyu.jpg |
| `carton__ylrxhe` | `none` | `y2 z4` | y2: geom exact (mean 0.27% frac2 0.0%), tex borderline (MAD 11.6); z4: geom exact (mean 0.17% frac2 0.0%), tex borderline (MAD 6.1) | runlogs/symmetry_20260915/renders/carton__ylrxhe.jpg |
| `bottle_of_water__sxffiv` | `none` | `z2` | z2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 2.9) | runlogs/symmetry_20260915/renders/bottle_of_water__sxffiv.jpg |
| `bottle_of_water__migvlt` | `z2` | `zinf` | zinf: geom exact (mean 0.46% frac2 0.0%), tex borderline (MAD 4.0) | runlogs/symmetry_20260915/renders/bottle_of_water__migvlt.jpg |
| `bottle_of_water__cyzaue` | `z5 z6` | `zinf` | zinf: geom near (mean 0.66% frac2 0.0%), tex preserved (MAD 1.1) | runlogs/symmetry_20260915/renders/bottle_of_water__cyzaue.jpg |
| `bottle_of_water__emquat` | `z4 z5 z6` | `zinf` | zinf: geom near (mean 0.50% frac2 0.0%), tex preserved (MAD 2.2) | runlogs/symmetry_20260915/renders/bottle_of_water__emquat.jpg |
| `bottle_of_water__lkpaas` | `z5 z6` | `zinf` | zinf: geom near (mean 0.46% frac2 1.1%), tex preserved (MAD 0.0) | runlogs/symmetry_20260915/renders/bottle_of_water__lkpaas.jpg |
| `bottle_of_water__rmtdxh` | `none` | `z5 z6` | z5: geom near (mean 0.53% frac2 2.7%), tex preserved (MAD 0.5); z6: geom near (mean 0.47% frac2 1.9%), tex preserved (MAD 0.5) | runlogs/symmetry_20260915/renders/bottle_of_water__rmtdxh.jpg |
| `bottle_of_water__eeaimz` | `z2` | `zinf` | zinf: geom exact (mean 0.46% frac2 0.0%), tex borderline (MAD 4.1) | runlogs/symmetry_20260915/renders/bottle_of_water__eeaimz.jpg |
| `lid__qferch` | `z2` | `zinf` | zinf: geom near (mean 0.18% frac2 2.0%), tex preserved (MAD 0.7) | runlogs/symmetry_20260915/renders/lid__qferch.jpg |
| `lid__fnqhbz` | `z2` | `z5 z6` | z5: geom near (mean 0.46% frac2 2.8%), tex preserved (MAD 0.5); z6: geom near (mean 0.42% frac2 2.2%), tex preserved (MAD 0.5) | runlogs/symmetry_20260915/renders/lid__fnqhbz.jpg |
| `lid__kdswwr` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 11.2); y2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 11.1); z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 4.8) | runlogs/symmetry_20260915/renders/lid__kdswwr.jpg |
| `bottle_of_wine__nhmtvt` | `none` | `z5 z6` | z5: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 3.1); z6: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 3.0) | runlogs/symmetry_20260915/renders/bottle_of_wine__nhmtvt.jpg |
| `bottle_of_wine__cldder` | `none` | `zinf` | zinf: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 6.9) | runlogs/symmetry_20260915/renders/bottle_of_wine__cldder.jpg |
| `marker__zvvovh` | `none` | `x4 x5 x6` | x4: geom near (mean 0.61% frac2 0.3%), tex preserved (MAD 0.9); x5: geom near (mean 0.55% frac2 0.4%), tex preserved (MAD 0.9); x6: geom near (mean 0.50% frac2 0.5%), tex preserved (MAD 0.9) | runlogs/symmetry_20260915/renders/marker__zvvovh.jpg |
| `marker__slgheo` | `none` | `x4 x5 x6` | x4: geom near (mean 0.60% frac2 0.3%), tex preserved (MAD 1.4); x5: geom near (mean 0.54% frac2 0.3%), tex preserved (MAD 1.3); x6: geom near (mean 0.49% frac2 0.5%), tex preserved (MAD 1.2) | runlogs/symmetry_20260915/renders/marker__slgheo.jpg |
| `marker__erhlbb` | `none` | `x4 x5 x6` | x4: geom near (mean 0.57% frac2 0.5%), tex preserved (MAD 0.9); x5: geom near (mean 0.52% frac2 0.1%), tex preserved (MAD 0.8); x6: geom near (mean 0.47% frac2 0.2%), tex preserved (MAD 0.8) | runlogs/symmetry_20260915/renders/marker__erhlbb.jpg |
| `marker__jlvneb` | `none` | `xinf` | xinf: geom near (mean 0.75% frac2 0.9%), tex preserved (MAD 1.1) | runlogs/symmetry_20260915/renders/marker__jlvneb.jpg |
| `marker__mcpkkq` | `none` | `xinf` | xinf: geom near (mean 0.79% frac2 0.2%), tex preserved (MAD 1.3) | runlogs/symmetry_20260915/renders/marker__mcpkkq.jpg |
| `cap__rgtedj` | `z5 z6` | `zinf` | zinf: geom near (mean 0.43% frac2 0.0%), tex preserved (MAD 0.7) | runlogs/symmetry_20260915/renders/cap__rgtedj.jpg |
| `cap__ygsmgm` | `z6` | `zinf` | zinf: geom near (mean 0.34% frac2 1.0%), tex preserved (MAD 0.2) | runlogs/symmetry_20260915/renders/cap__ygsmgm.jpg |
| `can_of_soda__bfrzvk` | `z2` | `zinf` | zinf: geom near (mean 0.71% frac2 0.0%), tex preserved (MAD 2.1) | runlogs/symmetry_20260915/renders/can_of_soda__bfrzvk.jpg |
| `can_of_soda__lugwcz` | `none` | `zinf` | zinf: geom exact (mean 0.21% frac2 0.0%), tex borderline (MAD 5.4) | runlogs/symmetry_20260915/renders/can_of_soda__lugwcz.jpg |
| `can_of_soda__evcxlr` | `none` | `zinf` | zinf: geom near (mean 0.69% frac2 0.0%), tex borderline (MAD 4.5) | runlogs/symmetry_20260915/renders/can_of_soda__evcxlr.jpg |
| `can_of_soda__wbrrad` | `z4` | `zinf` | zinf: geom near (mean 0.63% frac2 0.0%), tex preserved (MAD 2.9) | runlogs/symmetry_20260915/renders/can_of_soda__wbrrad.jpg |
| `bottle_of_beer__fgzjnb` | `z6` | `zinf` | zinf: geom exact (mean 0.28% frac2 0.0%), tex borderline (MAD 4.4) | runlogs/symmetry_20260915/renders/bottle_of_beer__fgzjnb.jpg |
| `bottle_of_beer__nfzzqc` | `none` | `zinf` | zinf: geom exact (mean 0.22% frac2 0.0%), tex borderline (MAD 4.6) | runlogs/symmetry_20260915/renders/bottle_of_beer__nfzzqc.jpg |
| `canned_food__foetdd` | `z2` | `z4` | z4: geom exact (mean 0.37% frac2 0.0%), tex borderline (MAD 4.3) | runlogs/symmetry_20260915/renders/canned_food__foetdd.jpg |
| `lemon__bmvxas` | `x6` | `y2 xinf` | y2: geom exact (mean 0.44% frac2 0.0%), tex borderline (MAD 3.6); xinf: geom near (mean 0.53% frac2 0.1%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/lemon__bmvxas.jpg |
| `lemon__vuwehu` | `none` | `y2` | y2: geom near (mean 0.67% frac2 0.8%), tex preserved (MAD 1.3) | runlogs/symmetry_20260915/renders/lemon__vuwehu.jpg |
| `lemon__dsacfa` | `none` | `y2 x5 x6` | y2: geom near (mean 0.50% frac2 0.0%), tex borderline (MAD 3.6); x5: geom near (mean 0.71% frac2 0.7%), tex borderline (MAD 3.2); x6: geom near (mean 0.63% frac2 0.3%), tex borderline (MAD 3.0) | runlogs/symmetry_20260915/renders/lemon__dsacfa.jpg |
| `lemon__asdada` | `xinf` | `y2 z2` | y2: geom exact (mean 0.42% frac2 0.0%), tex borderline (MAD 3.6); z2: geom exact (mean 0.48% frac2 0.0%), tex borderline (MAD 3.5) | runlogs/symmetry_20260915/renders/lemon__asdada.jpg |
| `lemon__gkrfha` | `x6` | `y2 xinf` | y2: geom near (mean 0.60% frac2 0.0%), tex borderline (MAD 3.6); xinf: geom near (mean 0.81% frac2 2.3%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/lemon__gkrfha.jpg |
| `lemon__hoghjl` | `none` | `y2 x6` | y2: geom near (mean 0.48% frac2 0.0%), tex borderline (MAD 3.6); x6: geom near (mean 0.78% frac2 2.5%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/lemon__hoghjl.jpg |
| `lemon__kvgfdl` | `x4 x5 x6` | `y2 xinf` | y2: geom near (mean 0.51% frac2 0.0%), tex borderline (MAD 3.6); xinf: geom exact (mean 0.38% frac2 0.0%), tex borderline (MAD 3.0) | runlogs/symmetry_20260915/renders/lemon__kvgfdl.jpg |
| `lemon__ovbcff` | `none` | `y2 x6` | y2: geom near (mean 0.56% frac2 0.0%), tex borderline (MAD 3.5); x6: geom near (mean 0.81% frac2 1.9%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/lemon__ovbcff.jpg |
| `lemon__peqwoa` | `x6` | `y2 xinf` | y2: geom exact (mean 0.47% frac2 0.0%), tex borderline (MAD 3.5); xinf: geom near (mean 0.56% frac2 0.2%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/lemon__peqwoa.jpg |
| `lemon__pffkae` | `x5 x6` | `y2 xinf` | y2: geom near (mean 0.61% frac2 0.0%), tex borderline (MAD 3.6); xinf: geom near (mean 0.54% frac2 0.1%), tex borderline (MAD 3.0) | runlogs/symmetry_20260915/renders/lemon__pffkae.jpg |
| `lemon__pqweio` | `none` | `y2 x4 x5 x6` | y2: geom near (mean 0.69% frac2 1.3%), tex borderline (MAD 3.5); x4: geom near (mean 0.81% frac2 2.5%), tex borderline (MAD 3.2); x5: geom near (mean 0.69% frac2 0.4%), tex borderline (MAD 3.0); x6: geom near (mean 0.61% frac2 0.3%), tex preserved (MAD 2.9) | runlogs/symmetry_20260915/renders/lemon__pqweio.jpg |
| `lemon__bbvonm` | `none` | `y2 z2 x5 x6` | y2: geom near (mean 0.56% frac2 0.0%), tex preserved (MAD 0.3); z2: geom near (mean 0.70% frac2 0.1%), tex preserved (MAD 0.3); x5: geom near (mean 0.69% frac2 1.8%), tex preserved (MAD 0.3); x6: geom near (mean 0.63% frac2 0.9%), tex preserved (MAD 0.3) | runlogs/symmetry_20260915/renders/lemon__bbvonm.jpg |
| `shampoo_dispenser__eblcnw` | `none` | `zinf` | zinf: geom near (mean 0.52% frac2 1.8%), tex preserved (MAD 0.6) | runlogs/symmetry_20260915/renders/shampoo_dispenser__eblcnw.jpg |
| `shampoo_dispenser__nrthyl` | `none` | `z4` | z4: geom near (mean 0.53% frac2 0.3%), tex preserved (MAD 0.6) | runlogs/symmetry_20260915/renders/shampoo_dispenser__nrthyl.jpg |
| `shampoo_dispenser__kkbhzw` | `z4 z5 z6` | `zinf` | zinf: geom near (mean 0.42% frac2 1.4%), tex preserved (MAD 0.7) | runlogs/symmetry_20260915/renders/shampoo_dispenser__kkbhzw.jpg |
| `shampoo_dispenser__eywylh` | `none` | `z4` | z4: geom near (mean 0.28% frac2 1.6%), tex preserved (MAD 1.8) | runlogs/symmetry_20260915/renders/shampoo_dispenser__eywylh.jpg |
| `box_of_cereal__ykuyqb` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 5.5); y2: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 5.4); z2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 5.4) | runlogs/symmetry_20260915/renders/box_of_cereal__ykuyqb.jpg |
| `beaker__effbnc` | `z6` | `zinf` | zinf: geom exact (mean 0.31% frac2 0.0%), tex borderline (MAD 4.8) | runlogs/symmetry_20260915/renders/beaker__effbnc.jpg |
| `bottle_of_medicine__qqsukh` | `none` | `z6` | z6: geom exact (mean 0.39% frac2 0.0%), tex borderline (MAD 6.2) | runlogs/symmetry_20260915/renders/bottle_of_medicine__qqsukh.jpg |
| `apple__agveuv` | `none` | `x5 x6` | x5: geom near (mean 0.43% frac2 2.9%), tex borderline (MAD 4.2); x6: geom near (mean 0.37% frac2 1.7%), tex borderline (MAD 4.6) | runlogs/symmetry_20260915/renders/apple__agveuv.jpg |
| `beeswax_candle__oimgmh` | `zinf` | `x2 y2` | x2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.7); y2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.7) | runlogs/symmetry_20260915/renders/beeswax_candle__oimgmh.jpg |
| `bottle_of_pop__twtsry` | `z2` | `zinf` | zinf: geom near (mean 0.81% frac2 0.3%), tex preserved (MAD 0.9) | runlogs/symmetry_20260915/renders/bottle_of_pop__twtsry.jpg |
| `bar_soap__feqemg` | `none` | `z2` | z2: geom near (mean 0.72% frac2 1.9%), tex preserved (MAD 0.2) | runlogs/symmetry_20260915/renders/bar_soap__feqemg.jpg |
| `beefsteak_tomato__pnrdxh` | `none` | `z6` | z6: geom near (mean 0.66% frac2 2.8%), tex preserved (MAD 0.7) | runlogs/symmetry_20260915/renders/beefsteak_tomato__pnrdxh.jpg |
| `cherry_tomato__dqjuub` | `none` | `z5 z6` | z5: geom near (mean 0.32% frac2 2.6%), tex preserved (MAD 0.8); z6: geom near (mean 0.28% frac2 1.6%), tex preserved (MAD 0.7) | runlogs/symmetry_20260915/renders/cherry_tomato__dqjuub.jpg |
| `bottle_of_black_pepper__honise` | `none` | `z4` | z4: geom near (mean 0.65% frac2 1.8%), tex borderline (MAD 5.7) | runlogs/symmetry_20260915/renders/bottle_of_black_pepper__honise.jpg |
| `bottle_of_black_pepper__ydzzrv` | `none` | `zinf` | zinf: geom near (mean 0.53% frac2 2.2%), tex borderline (MAD 5.4) | runlogs/symmetry_20260915/renders/bottle_of_black_pepper__ydzzrv.jpg |
| `ice_cube__dueotq` | `none` | `z4` | z4: geom near (mean 0.58% frac2 2.2%), tex preserved (MAD 0.0) | runlogs/symmetry_20260915/renders/ice_cube__dueotq.jpg |
| `ice_cube__lkirej` | `none` | `z4` | z4: geom near (mean 0.59% frac2 2.2%), tex preserved (MAD 0.0) | runlogs/symmetry_20260915/renders/ice_cube__lkirej.jpg |
| `vidalia_onion__ldeuli` | `none` | `xinf` | xinf: geom exact (mean 0.47% frac2 0.7%), tex borderline (MAD 5.4) | runlogs/symmetry_20260915/renders/vidalia_onion__ldeuli.jpg |
| `vidalia_onion__buyxll` | `none` | `xinf` | xinf: geom near (mean 0.53% frac2 1.3%), tex borderline (MAD 6.7) | runlogs/symmetry_20260915/renders/vidalia_onion__buyxll.jpg |
| `pack_of_cigarettes__ajphvf` | `none` | `y2` | y2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 11.0) | runlogs/symmetry_20260915/renders/pack_of_cigarettes__ajphvf.jpg |
| `pack_of_cigarettes__ohpqua` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/pack_of_cigarettes__ohpqua.jpg |
| `pack_of_cigarettes__ppbmak` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 7.0); y2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 7.1); z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 5.6) | runlogs/symmetry_20260915/renders/pack_of_cigarettes__ppbmak.jpg |
| `pack_of_cigarettes__tngixy` | `z2` | `x2 y2` | x2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.3); y2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.4) | runlogs/symmetry_20260915/renders/pack_of_cigarettes__tngixy.jpg |
| `orange__ucstpm` | `x2 zinf` | `full` | full: geom near (mean 0.67% frac2 0.1%), tex preserved (MAD 2.3) | runlogs/symmetry_20260915/renders/orange__ucstpm.jpg |
| `toy_dice__eqwazv` | `none` | `z2` | z2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 4.1) | runlogs/symmetry_20260915/renders/toy_dice__eqwazv.jpg |
| `toy_dice__hbkqqp` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 3.6); y2: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 4.6); z2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/toy_dice__hbkqqp.jpg |
| `toy_dice__bjrxkj` | `y2` | `x2 z2` | x2: geom exact (mean 0.15% frac2 0.0%), tex borderline (MAD 3.7); z2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 3.6) | runlogs/symmetry_20260915/renders/toy_dice__bjrxkj.jpg |
| `toy_dice__cmqtch` | `none` | `x2 y2 z4` | x2: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 4.2); y2: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 4.9); z4: geom exact (mean 0.20% frac2 0.0%), tex borderline (MAD 5.8) | runlogs/symmetry_20260915/renders/toy_dice__cmqtch.jpg |
| `toy_dice__nmhsvp` | `none` | `x2` | x2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 3.8) | runlogs/symmetry_20260915/renders/toy_dice__nmhsvp.jpg |
| `bottle_of_essential_oil__wansva` | `none` | `zinf` | zinf: geom exact (mean 0.15% frac2 0.0%), tex borderline (MAD 4.6) | runlogs/symmetry_20260915/renders/bottle_of_essential_oil__wansva.jpg |
| `bottle_of_essential_oil__eyyhld` | `none` | `zinf` | zinf: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 4.9) | runlogs/symmetry_20260915/renders/bottle_of_essential_oil__eyyhld.jpg |
| `bottle_of_essential_oil__xvqshn` | `none` | `zinf` | zinf: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 4.5) | runlogs/symmetry_20260915/renders/bottle_of_essential_oil__xvqshn.jpg |
| `bottle_of_essential_oil__xhoipk` | `none` | `zinf` | zinf: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 5.5) | runlogs/symmetry_20260915/renders/bottle_of_essential_oil__xhoipk.jpg |
| `bottle_of_essential_oil__yjxvpg` | `none` | `zinf` | zinf: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 4.5) | runlogs/symmetry_20260915/renders/bottle_of_essential_oil__yjxvpg.jpg |
| `toy_figure__jddrck` | `none` | `z4` | z4: geom near (mean 0.59% frac2 2.6%), tex borderline (MAD 6.4) | runlogs/symmetry_20260915/renders/toy_figure__jddrck.jpg |
| `teacup__cpozxi` | `z5` | `z6` | z6: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 4.8) | runlogs/symmetry_20260915/renders/teacup__cpozxi.jpg |
| `pickle__ifterk` | `none` | `y4` | y4: geom exact (mean 0.43% frac2 0.0%), tex borderline (MAD 4.7) | runlogs/symmetry_20260915/renders/pickle__ifterk.jpg |
| `pickle__iobrqp` | `none` | `y4` | y4: geom exact (mean 0.39% frac2 0.0%), tex borderline (MAD 4.8) | runlogs/symmetry_20260915/renders/pickle__iobrqp.jpg |
| `tongs__igsklx` | `none` | `x2` | x2: geom exact (mean 0.19% frac2 0.0%), tex borderline (MAD 4.8) | runlogs/symmetry_20260915/renders/tongs__igsklx.jpg |
| `tongs__gbjqyn` | `none` | `x2` | x2: geom exact (mean 0.20% frac2 0.0%), tex borderline (MAD 5.4) | runlogs/symmetry_20260915/renders/tongs__gbjqyn.jpg |
| `bottle_of_wine__oldpru` | `z6` | `zinf` | zinf: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 5.4) | runlogs/symmetry_20260915/renders/bottle_of_wine__oldpru.jpg |
| `box_of_cookies__jtalxs` | `none` | `y2` | y2: geom exact (mean 0.15% frac2 0.0%), tex borderline (MAD 3.2) | runlogs/symmetry_20260915/renders/box_of_cookies__jtalxs.jpg |
| `box_of_cookies__lwhbid` | `none` | `x2` | x2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 8.3) | runlogs/symmetry_20260915/renders/box_of_cookies__lwhbid.jpg |
| `soap_dispenser__zwudcz` | `z4` | `zinf` | zinf: geom exact (mean 0.09% frac2 0.1%), tex borderline (MAD 4.3) | runlogs/symmetry_20260915/renders/soap_dispenser__zwudcz.jpg |
| `soap_dispenser__cgxrwd` | `none` | `zinf` | zinf: geom near (mean 0.53% frac2 1.8%), tex preserved (MAD 0.5) | runlogs/symmetry_20260915/renders/soap_dispenser__cgxrwd.jpg |
| `soap_dispenser__edinmg` | `none` | `zinf` | zinf: geom exact (mean 0.25% frac2 0.9%), tex borderline (MAD 4.5) | runlogs/symmetry_20260915/renders/soap_dispenser__edinmg.jpg |
| `soap_dispenser__ijxawq` | `none` | `zinf` | zinf: geom near (mean 0.45% frac2 1.5%), tex preserved (MAD 1.3) | runlogs/symmetry_20260915/renders/soap_dispenser__ijxawq.jpg |
| `soap_dispenser__jmulpc` | `none` | `z2` | z2: geom near (mean 0.51% frac2 2.4%), tex preserved (MAD 0.9) | runlogs/symmetry_20260915/renders/soap_dispenser__jmulpc.jpg |
| `box_of_cereal__ihedod` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 5.2); y2: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 5.2); z2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 5.1) | runlogs/symmetry_20260915/renders/box_of_cereal__ihedod.jpg |
| `bottle_of_milk__ddvsgl` | `none` | `x2` | x2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 3.7) | runlogs/symmetry_20260915/renders/bottle_of_milk__ddvsgl.jpg |
| `bottle_of_milk__qsnnbp` | `none` | `x5 x6` | x5: geom near (mean 0.70% frac2 1.6%), tex borderline (MAD 3.6); x6: geom exact (mean 0.46% frac2 0.0%), tex borderline (MAD 3.2) | runlogs/symmetry_20260915/renders/bottle_of_milk__qsnnbp.jpg |
| `box_of_sanitary_napkins__cmnpuv` | `none` | `z2` | z2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 4.2) | runlogs/symmetry_20260915/renders/box_of_sanitary_napkins__cmnpuv.jpg |
| `box_of_sanitary_napkins__mxdefq` | `none` | `z2` | z2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 4.1) | runlogs/symmetry_20260915/renders/box_of_sanitary_napkins__mxdefq.jpg |
| `box_of_cane_sugar__rvsivw` | `none` | `z4` | z4: geom near (mean 0.58% frac2 0.5%), tex borderline (MAD 4.4) | runlogs/symmetry_20260915/renders/box_of_cane_sugar__rvsivw.jpg |
| `cellulose_tape__gchdhk` | `z6` | `x2 z4 z5` | x2: geom exact (mean 0.31% frac2 0.0%), tex borderline (MAD 6.2); z4: geom exact (mean 0.25% frac2 0.0%), tex borderline (MAD 3.6); z5: geom exact (mean 0.22% frac2 0.0%), tex borderline (MAD 3.2) | runlogs/symmetry_20260915/renders/cellulose_tape__gchdhk.jpg |
| `cellulose_tape__sklkyc` | `zinf` | `x2 y2` | x2: geom exact (mean 0.36% frac2 0.0%), tex borderline (MAD 5.0); y2: geom exact (mean 0.23% frac2 0.0%), tex borderline (MAD 4.9) | runlogs/symmetry_20260915/renders/cellulose_tape__sklkyc.jpg |
| `conditioner_dispenser__njqnuo` | `x4 x5 x6` | `xinf` | xinf: geom near (mean 0.48% frac2 1.7%), tex preserved (MAD 0.8) | runlogs/symmetry_20260915/renders/conditioner_dispenser__njqnuo.jpg |
| `conditioner_dispenser__udeflt` | `none` | `zinf` | zinf: geom near (mean 0.55% frac2 2.2%), tex preserved (MAD 1.3) | runlogs/symmetry_20260915/renders/conditioner_dispenser__udeflt.jpg |
| `pencil_case__acikzo` | `none` | `z2` | z2: geom exact (mean 0.25% frac2 0.0%), tex borderline (MAD 7.7) | runlogs/symmetry_20260915/renders/pencil_case__acikzo.jpg |
| `pack_of_cigarettes__ghbqav` | `none` | `z2` | z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 5.5) | runlogs/symmetry_20260915/renders/pack_of_cigarettes__ghbqav.jpg |
| `bottle_of_pop__sevoto` | `none` | `z2` | z2: geom near (mean 0.28% frac2 1.6%), tex borderline (MAD 4.1) | runlogs/symmetry_20260915/renders/bottle_of_pop__sevoto.jpg |
| `bottle_of_shampoo__stjjjm` | `none` | `z6` | z6: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.3) | runlogs/symmetry_20260915/renders/bottle_of_shampoo__stjjjm.jpg |
| `bottle_of_shampoo__lvvkhx` | `none` | `z6` | z6: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.2) | runlogs/symmetry_20260915/renders/bottle_of_shampoo__lvvkhx.jpg |
| `box_of_butter__mfjuil` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 8.2) | runlogs/symmetry_20260915/renders/box_of_butter__mfjuil.jpg |
| `toy_dice__ievnsq` | `none` | `z2` | z2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 4.0) | runlogs/symmetry_20260915/renders/toy_dice__ievnsq.jpg |
| `toy_dice__akguod` | `none` | `z2` | z2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 4.1) | runlogs/symmetry_20260915/renders/toy_dice__akguod.jpg |
| `toy_dice__kmagva` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 4.5); y2: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 3.5); z2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 3.7) | runlogs/symmetry_20260915/renders/toy_dice__kmagva.jpg |
| `toy_dice__krpdji` | `none` | `y2` | y2: geom exact (mean 0.17% frac2 0.0%), tex borderline (MAD 3.8) | runlogs/symmetry_20260915/renders/toy_dice__krpdji.jpg |
| `toy_dice__mqvaaa` | `none` | `x2 z2 y4` | x2: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 5.1); z2: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 4.3); y4: geom exact (mean 0.20% frac2 0.0%), tex borderline (MAD 5.9) | runlogs/symmetry_20260915/renders/toy_dice__mqvaaa.jpg |
| `toy_dice__rbzyrv` | `none` | `x2 z2 y4` | x2: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 5.3); z2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 3.9); y4: geom exact (mean 0.19% frac2 0.0%), tex borderline (MAD 5.3) | runlogs/symmetry_20260915/renders/toy_dice__rbzyrv.jpg |
| `toy_dice__xfqqhh` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 4.5); y2: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 3.5); z2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 3.7) | runlogs/symmetry_20260915/renders/toy_dice__xfqqhh.jpg |
| `box_of_baking_mix__lgusnr` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 4.6) | runlogs/symmetry_20260915/renders/box_of_baking_mix__lgusnr.jpg |
| `box_of_tissues__zutrxn` | `none` | `x2` | x2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 4.2) | runlogs/symmetry_20260915/renders/box_of_tissues__zutrxn.jpg |
| `box_of_tissues__xwstls` | `none` | `x2` | x2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 4.1) | runlogs/symmetry_20260915/renders/box_of_tissues__xwstls.jpg |
| `box_of_granola_bars__tqwhvz` | `none` | `z2` | z2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 3.3) | runlogs/symmetry_20260915/renders/box_of_granola_bars__tqwhvz.jpg |
| `box_of_granola_bars__bqeeki` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.7) | runlogs/symmetry_20260915/renders/box_of_granola_bars__bqeeki.jpg |
| `box_of_granola_bars__awombj` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.0) | runlogs/symmetry_20260915/renders/box_of_granola_bars__awombj.jpg |
| `bottle_of_wine__dijgvu` | `z2` | `zinf` | zinf: geom exact (mean 0.41% frac2 0.0%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/bottle_of_wine__dijgvu.jpg |
| `notebook__eijrqw` | `none` | `y2` | y2: geom exact (mean 0.21% frac2 0.0%), tex borderline (MAD 3.5) | runlogs/symmetry_20260915/renders/notebook__eijrqw.jpg |
| `bottle_of_pop__bqretx` | `none` | `z4 z3 z5` | z4: geom exact (mean 0.22% frac2 0.5%), tex borderline (MAD 5.0); z3: geom near (mean 0.18% frac2 1.3%), tex borderline (MAD 4.8); z5: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 4.7) | runlogs/symmetry_20260915/renders/bottle_of_pop__bqretx.jpg |
| `bottle_of_pop__cmqubs` | `none` | `z2` | z2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 4.5) | runlogs/symmetry_20260915/renders/bottle_of_pop__cmqubs.jpg |
| `bottle_of_pop__kviogj` | `none` | `zinf` | zinf: geom near (mean 0.28% frac2 1.6%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/bottle_of_pop__kviogj.jpg |
| `box_of_yogurt__jtjcrx` | `none` | `zinf` | zinf: geom near (mean 0.48% frac2 0.0%), tex borderline (MAD 6.5) | runlogs/symmetry_20260915/renders/box_of_yogurt__jtjcrx.jpg |
| `case_of_eyeshadow__zgervc` | `none` | `y2` | y2: geom near (mean 0.55% frac2 0.1%), tex preserved (MAD 0.5) | runlogs/symmetry_20260915/renders/case_of_eyeshadow__zgervc.jpg |
| `pack_of_chocolate_bar__bxgewr` | `none` | `z2` | z2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 3.5) | runlogs/symmetry_20260915/renders/pack_of_chocolate_bar__bxgewr.jpg |
| `pack_of_chocolate_bar__tidcst` | `none` | `x2` | x2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 3.6) | runlogs/symmetry_20260915/renders/pack_of_chocolate_bar__tidcst.jpg |
| `box_of_coffee__cjtadw` | `none` | `z2` | z2: geom exact (mean 0.43% frac2 0.0%), tex borderline (MAD 7.5) | runlogs/symmetry_20260915/renders/box_of_coffee__cjtadw.jpg |
| `can_of_icetea__ifrjsc` | `none` | `zinf` | zinf: geom near (mean 0.63% frac2 0.0%), tex borderline (MAD 6.0) | runlogs/symmetry_20260915/renders/can_of_icetea__ifrjsc.jpg |
| `goblet__nawrfs` | `none` | `zinf` | zinf: geom exact (mean 0.20% frac2 0.7%), tex borderline (MAD 6.5) | runlogs/symmetry_20260915/renders/goblet__nawrfs.jpg |
| `bottle_of_ginger_beer__zkocwb` | `z2` | `z5 z6` | z5: geom near (mean 0.64% frac2 0.0%), tex borderline (MAD 5.1); z6: geom near (mean 0.59% frac2 0.0%), tex borderline (MAD 4.9) | runlogs/symmetry_20260915/renders/bottle_of_ginger_beer__zkocwb.jpg |
| `marker__ablbdc` | `none` | `x4 x5 x6` | x4: geom near (mean 0.60% frac2 2.0%), tex preserved (MAD 0.9); x5: geom near (mean 0.54% frac2 0.0%), tex preserved (MAD 1.0); x6: geom near (mean 0.49% frac2 0.2%), tex preserved (MAD 0.9) | runlogs/symmetry_20260915/renders/marker__ablbdc.jpg |
| `marker__bwwkmb` | `none` | `x5 x6` | x5: geom near (mean 0.63% frac2 1.9%), tex preserved (MAD 0.9); x6: geom near (mean 0.56% frac2 0.1%), tex preserved (MAD 0.9) | runlogs/symmetry_20260915/renders/marker__bwwkmb.jpg |
| `pencil__bfkwhb` | `xinf` | `y2 z2` | y2: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 4.5); z2: geom exact (mean 0.16% frac2 0.0%), tex borderline (MAD 4.6) | runlogs/symmetry_20260915/renders/pencil__bfkwhb.jpg |
| `table_knife__eebpap` | `none` | `x2` | x2: geom near (mean 0.39% frac2 1.2%), tex preserved (MAD 1.3) | runlogs/symmetry_20260915/renders/table_knife__eebpap.jpg |
| `table_knife__jxdfyy` | `none` | `x2` | x2: geom near (mean 0.25% frac2 1.4%), tex preserved (MAD 1.0) | runlogs/symmetry_20260915/renders/table_knife__jxdfyy.jpg |
| `table_knife__lrdmpf` | `x2` | `y2 z2` | y2: geom near (mean 0.68% frac2 1.2%), tex preserved (MAD 1.2); z2: geom near (mean 0.68% frac2 0.9%), tex preserved (MAD 1.3) | runlogs/symmetry_20260915/renders/table_knife__lrdmpf.jpg |
| `lipstick__albshz` | `y4` | `y5` | y5: geom near (mean 0.57% frac2 0.0%), tex borderline (MAD 2.6) | runlogs/symmetry_20260915/renders/lipstick__albshz.jpg |
| `bottle_of_lotion__tkryrh` | `z4 z6` | `zinf` | zinf: geom near (mean 0.53% frac2 0.0%), tex preserved (MAD 2.3) | runlogs/symmetry_20260915/renders/bottle_of_lotion__tkryrh.jpg |
| `bottle_of_face_cream__dztaed` | `zinf` | `x2 y2` | x2: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 11.9); y2: geom exact (mean 0.20% frac2 0.0%), tex borderline (MAD 11.8) | runlogs/symmetry_20260915/renders/bottle_of_face_cream__dztaed.jpg |
| `box_of_barley__lxpwnw` | `none` | `z2` | z2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 5.4) | runlogs/symmetry_20260915/renders/box_of_barley__lxpwnw.jpg |
| `bottle_of_fennel__ihlkfu` | `none` | `zinf` | zinf: geom exact (mean 0.20% frac2 0.7%), tex borderline (MAD 3.8) | runlogs/symmetry_20260915/renders/bottle_of_fennel__ihlkfu.jpg |
| `bottle_of_ginger__drqhzo` | `none` | `zinf` | zinf: geom near (mean 0.28% frac2 2.2%), tex preserved (MAD 2.1) | runlogs/symmetry_20260915/renders/bottle_of_ginger__drqhzo.jpg |
| `bottle_of_ground_mace__lgpxro` | `none` | `zinf` | zinf: geom near (mean 0.25% frac2 2.1%), tex preserved (MAD 2.3) | runlogs/symmetry_20260915/renders/bottle_of_ground_mace__lgpxro.jpg |
| `bottle_of_ground_nutmeg__qebruq` | `none` | `zinf` | zinf: geom near (mean 0.28% frac2 2.2%), tex preserved (MAD 2.1) | runlogs/symmetry_20260915/renders/bottle_of_ground_nutmeg__qebruq.jpg |
| `bottle_of_mustard_seeds__grryaf` | `none` | `zinf` | zinf: geom near (mean 0.25% frac2 2.1%), tex borderline (MAD 4.3) | runlogs/symmetry_20260915/renders/bottle_of_mustard_seeds__grryaf.jpg |
| `bottle_of_seasoning__vqwqqv` | `none` | `zinf` | zinf: geom near (mean 0.28% frac2 2.2%), tex preserved (MAD 2.1) | runlogs/symmetry_20260915/renders/bottle_of_seasoning__vqwqqv.jpg |
| `bottle_of_sesame_seeds__lzndie` | `none` | `zinf` | zinf: geom exact (mean 0.22% frac2 0.7%), tex borderline (MAD 4.4) | runlogs/symmetry_20260915/renders/bottle_of_sesame_seeds__lzndie.jpg |
| `bottle_of_onion_powder__xruqod` | `none` | `zinf` | zinf: geom near (mean 0.24% frac2 2.2%), tex borderline (MAD 4.5) | runlogs/symmetry_20260915/renders/bottle_of_onion_powder__xruqod.jpg |
| `bottle_of_poppy_seeds__xdtrgi` | `none` | `zinf` | zinf: geom near (mean 0.25% frac2 2.1%), tex borderline (MAD 4.2) | runlogs/symmetry_20260915/renders/bottle_of_poppy_seeds__xdtrgi.jpg |
| `bottle_of_tea_leaves__pfbtus` | `none` | `z2` | z2: geom near (mean 0.20% frac2 1.1%), tex borderline (MAD 4.2) | runlogs/symmetry_20260915/renders/bottle_of_tea_leaves__pfbtus.jpg |
| `jar_of_curry_powder__ozhftu` | `none` | `zinf` | zinf: geom near (mean 0.26% frac2 2.1%), tex preserved (MAD 2.4) | runlogs/symmetry_20260915/renders/jar_of_curry_powder__ozhftu.jpg |
| `jar_of_dill_seed__reavkt` | `none` | `zinf` | zinf: geom near (mean 0.60% frac2 1.7%), tex borderline (MAD 6.3) | runlogs/symmetry_20260915/renders/jar_of_dill_seed__reavkt.jpg |
| `bottle_of_coconut_milk__idenxg` | `none` | `z4 z5 z6` | z4: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 4.4); z5: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 4.2); z6: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 4.1) | runlogs/symmetry_20260915/renders/bottle_of_coconut_milk__idenxg.jpg |
| `bottle_of_coconut_water__lsixio` | `z4` | `zinf` | zinf: geom exact (mean 0.41% frac2 0.0%), tex borderline (MAD 3.3) | runlogs/symmetry_20260915/renders/bottle_of_coconut_water__lsixio.jpg |
| `bottle_of_tea__iladfg` | `none` | `z4 z3 z5` | z4: geom exact (mean 0.19% frac2 0.0%), tex borderline (MAD 5.0); z3: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 4.8); z5: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 5.4) | runlogs/symmetry_20260915/renders/bottle_of_tea__iladfg.jpg |
| `bottle_of_tea__yatmrs` | `none` | `z2` | z2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 5.1) | runlogs/symmetry_20260915/renders/bottle_of_tea__yatmrs.jpg |
| `paper_liners__yvuilk` | `z4 z5 z6` | `zinf` | zinf: geom near (mean 0.69% frac2 2.3%), tex preserved (MAD 0.6) | runlogs/symmetry_20260915/renders/paper_liners__yvuilk.jpg |
| `biscuit__ukcwqw` | `none` | `z2` | z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 3.6) | runlogs/symmetry_20260915/renders/biscuit__ukcwqw.jpg |
| `butter_cookie__kukrla` | `z2` | `x2 zinf` | x2: geom near (mean 0.61% frac2 1.4%), tex borderline (MAD 3.6); zinf: geom near (mean 0.30% frac2 1.6%), tex preserved (MAD 3.0) | runlogs/symmetry_20260915/renders/butter_cookie__kukrla.jpg |
| `coaster__arjpcz` | `none` | `x2 zinf` | x2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 3.1); zinf: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 3.3) | runlogs/symmetry_20260915/renders/coaster__arjpcz.jpg |
| `sugar_cookie__cgkiti` | `none` | `x2 zinf` | x2: geom near (mean 0.51% frac2 0.0%), tex borderline (MAD 8.2); zinf: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 8.6) | runlogs/symmetry_20260915/renders/sugar_cookie__cgkiti.jpg |
| `sponge__klwueh` | `none` | `z2` | z2: geom exact (mean 0.09% frac2 0.0%), tex borderline (MAD 4.4) | runlogs/symmetry_20260915/renders/sponge__klwueh.jpg |
| `sugar_cookie__hlyeaq` | `none` | `x2 zinf` | x2: geom near (mean 0.51% frac2 0.0%), tex borderline (MAD 8.2); zinf: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 8.6) | runlogs/symmetry_20260915/renders/sugar_cookie__hlyeaq.jpg |
| `carton_of_orange_juice__brryuo` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 5.2) | runlogs/symmetry_20260915/renders/carton_of_orange_juice__brryuo.jpg |
| `carton_of_pineapple_juice__vzueyg` | `none` | `z6` | z6: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 8.0) | runlogs/symmetry_20260915/renders/carton_of_pineapple_juice__vzueyg.jpg |
| `carrot__aucrah` | `none` | `xinf` | xinf: geom near (mean 0.60% frac2 0.9%), tex borderline (MAD 3.8) | runlogs/symmetry_20260915/renders/carrot__aucrah.jpg |
| `green_bean__dyjvwv` | `none` | `x2` | x2: geom near (mean 0.58% frac2 1.0%), tex borderline (MAD 5.0) | runlogs/symmetry_20260915/renders/green_bean__dyjvwv.jpg |
| `green_bean__lvbjws` | `none` | `x2` | x2: geom near (mean 0.56% frac2 0.9%), tex borderline (MAD 4.9) | runlogs/symmetry_20260915/renders/green_bean__lvbjws.jpg |
| `carrot__qhmmmx` | `none` | `xinf` | xinf: geom near (mean 0.61% frac2 2.7%), tex preserved (MAD 2.8) | runlogs/symmetry_20260915/renders/carrot__qhmmmx.jpg |
| `box_of_whiskey__ftapxo` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.09% frac2 0.0%), tex borderline (MAD 4.1); y2: geom exact (mean 0.09% frac2 0.0%), tex borderline (MAD 8.6); z2: geom exact (mean 0.09% frac2 0.0%), tex borderline (MAD 8.8) | runlogs/symmetry_20260915/renders/box_of_whiskey__ftapxo.jpg |
| `cork__lseuwf` | `none` | `zinf` | zinf: geom exact (mean 0.31% frac2 0.0%), tex borderline (MAD 4.5) | runlogs/symmetry_20260915/renders/cork__lseuwf.jpg |
| `cork__ncxgpe` | `none` | `zinf` | zinf: geom exact (mean 0.20% frac2 0.0%), tex borderline (MAD 4.6) | runlogs/symmetry_20260915/renders/cork__ncxgpe.jpg |
| `hamburger_bun__fqxumh` | `none` | `zinf` | zinf: geom near (mean 0.60% frac2 0.3%), tex borderline (MAD 4.0) | runlogs/symmetry_20260915/renders/hamburger_bun__fqxumh.jpg |
| `hamburger_bun__jeygaq` | `none` | `z4 z5 z6` | z4: geom exact (mean 0.34% frac2 0.3%), tex borderline (MAD 8.1); z5: geom exact (mean 0.27% frac2 0.0%), tex borderline (MAD 7.6); z6: geom exact (mean 0.26% frac2 0.0%), tex borderline (MAD 7.5) | runlogs/symmetry_20260915/renders/hamburger_bun__jeygaq.jpg |
| `bottle_of_disinfectant__faedff` | `none` | `z2` | z2: geom near (mean 0.18% frac2 1.1%), tex preserved (MAD 1.8) | runlogs/symmetry_20260915/renders/bottle_of_disinfectant__faedff.jpg |
| `flashlight__jtrbkc` | `none` | `x6` | x6: geom near (mean 0.58% frac2 0.9%), tex preserved (MAD 0.7) | runlogs/symmetry_20260915/renders/flashlight__jtrbkc.jpg |
| `bread_slice__pfggnm` | `none` | `y2 z4` | y2: geom exact (mean 0.32% frac2 0.0%), tex borderline (MAD 6.1); z4: geom near (mean 0.49% frac2 1.8%), tex borderline (MAD 7.1) | runlogs/symmetry_20260915/renders/bread_slice__pfggnm.jpg |
| `bandage__riftxh` | `z2` | `x2 y2` | x2: geom exact (mean 0.17% frac2 0.0%), tex borderline (MAD 6.1); y2: geom exact (mean 0.17% frac2 0.0%), tex borderline (MAD 6.1) | runlogs/symmetry_20260915/renders/bandage__riftxh.jpg |
| `bottle_of_carrot_juice__jkuhio` | `none` | `z2` | z2: geom exact (mean 0.09% frac2 0.0%), tex borderline (MAD 3.7) | runlogs/symmetry_20260915/renders/bottle_of_carrot_juice__jkuhio.jpg |
| `bottle_of_fruit_punch__azcigi` | `none` | `z2` | z2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 4.7) | runlogs/symmetry_20260915/renders/bottle_of_fruit_punch__azcigi.jpg |
| `bottle_of_papaya_juice__nmfmxy` | `none` | `z2` | z2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 8.1) | runlogs/symmetry_20260915/renders/bottle_of_papaya_juice__nmfmxy.jpg |
| `vase__aegxpb` | `none` | `zinf` | zinf: geom near (mean 0.44% frac2 1.6%), tex borderline (MAD 6.5) | runlogs/symmetry_20260915/renders/vase__aegxpb.jpg |
| `vase__atgnsc` | `none` | `zinf` | zinf: geom near (mean 0.38% frac2 2.1%), tex preserved (MAD 0.5) | runlogs/symmetry_20260915/renders/vase__atgnsc.jpg |
| `jar_of_mayonnaise__ktmjvg` | `none` | `zinf` | zinf: geom exact (mean 0.32% frac2 0.0%), tex borderline (MAD 3.6) | runlogs/symmetry_20260915/renders/jar_of_mayonnaise__ktmjvg.jpg |
| `jar_of_tumeric__miivhi` | `none` | `z4` | z4: geom near (mean 0.30% frac2 1.9%), tex borderline (MAD 3.2) | runlogs/symmetry_20260915/renders/jar_of_tumeric__miivhi.jpg |
| `dip_candle__poexjo` | `z6` | `x2 z4 z5` | x2: geom near (mean 0.29% frac2 1.9%), tex preserved (MAD 0.5); z4: geom near (mean 0.59% frac2 0.3%), tex preserved (MAD 0.5); z5: geom near (mean 0.49% frac2 0.1%), tex preserved (MAD 0.5) | runlogs/symmetry_20260915/renders/dip_candle__poexjo.jpg |
| `dip_candle__gbaons` | `zinf` | `x2 y2` | x2: geom near (mean 0.25% frac2 2.6%), tex preserved (MAD 0.7); y2: geom near (mean 0.25% frac2 2.6%), tex preserved (MAD 0.8) | runlogs/symmetry_20260915/renders/dip_candle__gbaons.jpg |
| `cabinet_door__ertkre` | `none` | `x2` | x2: geom exact (mean 0.09% frac2 0.6%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/cabinet_door__ertkre.jpg |
| `cabinet_base__dcmhhh` | `none` | `x2` | x2: geom exact (mean 0.17% frac2 0.0%), tex borderline (MAD 3.4) | runlogs/symmetry_20260915/renders/cabinet_base__dcmhhh.jpg |
| `paperback_book__jhaltj` | `none` | `x2` | x2: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/paperback_book__jhaltj.jpg |
| `paperback_book__aubynm` | `none` | `x2` | x2: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 4.6) | runlogs/symmetry_20260915/renders/paperback_book__aubynm.jpg |
| `paperback_book__phsgun` | `none` | `x2` | x2: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 3.0) | runlogs/symmetry_20260915/renders/paperback_book__phsgun.jpg |
| `paperback_book__vciijb` | `none` | `x2` | x2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 3.4) | runlogs/symmetry_20260915/renders/paperback_book__vciijb.jpg |
| `paperback_book__eqzjjk` | `none` | `x2` | x2: geom exact (mean 0.20% frac2 0.0%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/paperback_book__eqzjjk.jpg |
| `paperback_book__ncjkcg` | `none` | `x2` | x2: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 3.6) | runlogs/symmetry_20260915/renders/paperback_book__ncjkcg.jpg |
| `hardback__bgtbdp` | `none` | `x2` | x2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/hardback__bgtbdp.jpg |
| `letter__oheufk` | `none` | `x2 y2 z2` | x2: geom near (mean 0.69% frac2 1.9%), tex borderline (MAD 9.1); y2: geom near (mean 0.68% frac2 1.9%), tex borderline (MAD 7.5); z2: geom exact (mean 0.14% frac2 0.0%), tex borderline (MAD 7.9) | runlogs/symmetry_20260915/renders/letter__oheufk.jpg |
| `letter__dyvllf` | `none` | `z2` | z2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 8.0) | runlogs/symmetry_20260915/renders/letter__dyvllf.jpg |
| `letter__bdlpry` | `none` | `z2` | z2: geom exact (mean 0.39% frac2 0.0%), tex borderline (MAD 6.7) | runlogs/symmetry_20260915/renders/letter__bdlpry.jpg |
| `letter__csncjv` | `none` | `z2` | z2: geom near (mean 0.58% frac2 0.1%), tex borderline (MAD 7.8) | runlogs/symmetry_20260915/renders/letter__csncjv.jpg |
| `letter__esmoua` | `none` | `x2 y2 z2` | x2: geom near (mean 0.52% frac2 0.0%), tex borderline (MAD 4.3); y2: geom near (mean 0.52% frac2 0.0%), tex borderline (MAD 7.3); z2: geom exact (mean 0.06% frac2 0.0%), tex borderline (MAD 7.8) | runlogs/symmetry_20260915/renders/letter__esmoua.jpg |
| `letter__igvnic` | `none` | `z2` | z2: geom exact (mean 0.06% frac2 0.0%), tex borderline (MAD 6.5) | runlogs/symmetry_20260915/renders/letter__igvnic.jpg |
| `letter__qrykxc` | `none` | `z2` | z2: geom exact (mean 0.06% frac2 0.0%), tex borderline (MAD 6.8) | runlogs/symmetry_20260915/renders/letter__qrykxc.jpg |
| `letter__jafvfh` | `none` | `z2` | z2: geom exact (mean 0.06% frac2 0.0%), tex borderline (MAD 6.0) | runlogs/symmetry_20260915/renders/letter__jafvfh.jpg |
| `keyboard__jtdyix` | `none` | `x2` | x2: geom near (mean 0.58% frac2 2.9%), tex borderline (MAD 9.1) | runlogs/symmetry_20260915/renders/keyboard__jtdyix.jpg |
| `keyboard__uwacaq` | `z2` | `x2 y2` | x2: geom near (mean 0.60% frac2 0.0%), tex preserved (MAD 1.7); y2: geom near (mean 0.61% frac2 0.0%), tex preserved (MAD 1.7) | runlogs/symmetry_20260915/renders/keyboard__uwacaq.jpg |
| `newspaper__edarlp` | `none` | `x2` | x2: geom near (mean 0.28% frac2 1.2%), tex borderline (MAD 6.3) | runlogs/symmetry_20260915/renders/newspaper__edarlp.jpg |
| `carton_of_eggs__rixhgu` | `none` | `z2` | z2: geom exact (mean 0.09% frac2 0.0%), tex borderline (MAD 8.0) | runlogs/symmetry_20260915/renders/carton_of_eggs__rixhgu.jpg |
| `chopping_board__psabiv` | `none` | `x2` | x2: geom exact (mean 0.09% frac2 0.0%), tex borderline (MAD 5.5) | runlogs/symmetry_20260915/renders/chopping_board__psabiv.jpg |
| `notepad__awcyha` | `none` | `x2` | x2: geom near (mean 0.46% frac2 1.6%), tex preserved (MAD 1.7) | runlogs/symmetry_20260915/renders/notepad__awcyha.jpg |
| `clipboard__envjqe` | `none` | `z2` | z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 5.7) | runlogs/symmetry_20260915/renders/clipboard__envjqe.jpg |
| `wall_clock__zpctvp` | `none` | `x4 x6` | x4: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 4.5); x6: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 4.5) | runlogs/symmetry_20260915/renders/wall_clock__zpctvp.jpg |
| `wall_clock__jerval` | `none` | `x2` | x2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 6.5) | runlogs/symmetry_20260915/renders/wall_clock__jerval.jpg |
| `wall_clock__xevvel` | `x4` | `x6` | x6: geom near (mean 0.20% frac2 1.2%), tex preserved (MAD 3.0) | runlogs/symmetry_20260915/renders/wall_clock__xevvel.jpg |
| `painting__ejhanr` | `none` | `x2` | x2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 4.7) | runlogs/symmetry_20260915/renders/painting__ejhanr.jpg |
| `painting__gardkz` | `none` | `x2` | x2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 5.9) | runlogs/symmetry_20260915/renders/painting__gardkz.jpg |
| `painting__eexydp` | `none` | `x2` | x2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 4.8) | runlogs/symmetry_20260915/renders/painting__eexydp.jpg |
| `guitar__hpqjrq` | `none` | `x2` | x2: geom near (mean 0.47% frac2 1.4%), tex borderline (MAD 5.3) | runlogs/symmetry_20260915/renders/guitar__hpqjrq.jpg |
| `hanger__anobre` | `none` | `x2` | x2: geom near (mean 0.12% frac2 1.7%), tex borderline (MAD 5.7) | runlogs/symmetry_20260915/renders/hanger__anobre.jpg |
| `towel_rack__nhlijv` | `none` | `z2` | z2: geom near (mean 0.19% frac2 2.2%), tex preserved (MAD 1.1) | runlogs/symmetry_20260915/renders/towel_rack__nhlijv.jpg |
| `rolling_pin__mwrdae` | `none` | `x2 yinf` | x2: geom exact (mean 0.05% frac2 0.0%), tex borderline (MAD 5.8); yinf: geom exact (mean 0.09% frac2 0.0%), tex borderline (MAD 7.8) | runlogs/symmetry_20260915/renders/rolling_pin__mwrdae.jpg |
| `baseball_bat__yalgei` | `none` | `xinf` | xinf: geom exact (mean 0.18% frac2 0.0%), tex borderline (MAD 5.1) | runlogs/symmetry_20260915/renders/baseball_bat__yalgei.jpg |
| `shelf__qrmiur` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.24% frac2 0.0%), tex borderline (MAD 6.7); y2: geom exact (mean 0.22% frac2 0.0%), tex borderline (MAD 8.0); z2: geom exact (mean 0.25% frac2 0.0%), tex borderline (MAD 9.2) | runlogs/symmetry_20260915/renders/shelf__qrmiur.jpg |
| `shelf__iruewm` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 3.5); y2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 3.4); z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 3.5) | runlogs/symmetry_20260915/renders/shelf__iruewm.jpg |
| `radiator__arsjbe` | `none` | `y2` | y2: geom near (mean 0.35% frac2 2.4%), tex preserved (MAD 2.0) | runlogs/symmetry_20260915/renders/radiator__arsjbe.jpg |
| `radiator__xnoiwp` | `none` | `y2` | y2: geom near (mean 0.29% frac2 1.4%), tex preserved (MAD 2.1) | runlogs/symmetry_20260915/renders/radiator__xnoiwp.jpg |
| `drumstick__jauzlc` | `z2 x4 x5 x6` | `y2 xinf` | y2: geom exact (mean 0.29% frac2 0.0%), tex borderline (MAD 3.0); xinf: geom exact (mean 0.24% frac2 0.0%), tex borderline (MAD 3.3) | runlogs/symmetry_20260915/renders/drumstick__jauzlc.jpg |
| `drumstick__lkvzmj` | `y2 z2 x4 x5 x6` | `xinf` | xinf: geom exact (mean 0.04% frac2 0.0%), tex borderline (MAD 3.4) | runlogs/symmetry_20260915/renders/drumstick__lkvzmj.jpg |
| `umbrella__hdfkbc` | `none` | `x4` | x4: geom exact (mean 0.05% frac2 0.0%), tex borderline (MAD 7.9) | runlogs/symmetry_20260915/renders/umbrella__hdfkbc.jpg |
| `picture__watgtj` | `none` | `z2` | z2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 4.3) | runlogs/symmetry_20260915/renders/picture__watgtj.jpg |
| `towel_rack__quznqx` | `none` | `x2 yinf` | x2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 4.3); yinf: geom near (mean 0.74% frac2 0.0%), tex borderline (MAD 4.3) | runlogs/symmetry_20260915/renders/towel_rack__quznqx.jpg |
| `ruler__ujizkx` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.04% frac2 0.0%), tex borderline (MAD 6.7); y2: geom exact (mean 0.03% frac2 0.0%), tex borderline (MAD 8.1); z2: geom exact (mean 0.04% frac2 0.0%), tex borderline (MAD 5.9) | runlogs/symmetry_20260915/renders/ruler__ujizkx.jpg |
| `ruler__vmnaih` | `none` | `x2 z2` | x2: geom exact (mean 0.04% frac2 0.0%), tex borderline (MAD 10.2); z2: geom exact (mean 0.04% frac2 0.0%), tex borderline (MAD 6.8) | runlogs/symmetry_20260915/renders/ruler__vmnaih.jpg |
| `address__pbfbxj` | `none` | `x2 y2` | x2: geom exact (mean 0.21% frac2 0.6%), tex borderline (MAD 5.6); y2: geom near (mean 0.84% frac2 0.8%), tex borderline (MAD 7.7) | runlogs/symmetry_20260915/renders/address__pbfbxj.jpg |
| `pack_of_kielbasa__rvhpal` | `none` | `x2` | x2: geom exact (mean 0.06% frac2 0.0%), tex borderline (MAD 4.2) | runlogs/symmetry_20260915/renders/pack_of_kielbasa__rvhpal.jpg |
| `pen__enfumt` | `none` | `xinf` | xinf: geom near (mean 0.77% frac2 2.9%), tex preserved (MAD 1.3) | runlogs/symmetry_20260915/renders/pen__enfumt.jpg |
| `cutting_board__tcdrzs` | `y2` | `x2 z2` | x2: geom exact (mean 0.12% frac2 0.0%), tex borderline (MAD 6.0); z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 5.7) | runlogs/symmetry_20260915/renders/cutting_board__tcdrzs.jpg |
| `cutting_board__nsvnai` | `y2` | `x2 z2` | x2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 4.8); z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 4.8) | runlogs/symmetry_20260915/renders/cutting_board__nsvnai.jpg |
| `cutting_board__jfrbuz` | `none` | `x2 y2 z2` | x2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 4.6); y2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 4.5); z2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/cutting_board__jfrbuz.jpg |
| `cutting_board__aibvew` | `none` | `x2` | x2: geom exact (mean 0.08% frac2 0.0%), tex borderline (MAD 6.4) | runlogs/symmetry_20260915/renders/cutting_board__aibvew.jpg |
| `plate__luhkiz` | `z4 z5 z6` | `zinf` | zinf: geom exact (mean 0.15% frac2 0.0%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/plate__luhkiz.jpg |
| `plate__bgxzec` | `zinf` | `x2 y2` | x2: geom near (mean 0.64% frac2 0.9%), tex preserved (MAD 0.5); y2: geom near (mean 0.64% frac2 0.9%), tex preserved (MAD 0.5) | runlogs/symmetry_20260915/renders/plate__bgxzec.jpg |
| `plate__spppps` | `none` | `z2` | z2: geom exact (mean 0.07% frac2 0.0%), tex borderline (MAD 6.4) | runlogs/symmetry_20260915/renders/plate__spppps.jpg |
| `toy_figure__gqspnb` | `none` | `x2` | x2: geom near (mean 0.32% frac2 2.2%), tex preserved (MAD 2.2) | runlogs/symmetry_20260915/renders/toy_figure__gqspnb.jpg |
| `french_fries__oeterg` | `none` | `z2` | z2: geom near (mean 0.81% frac2 2.9%), tex borderline (MAD 5.4) | runlogs/symmetry_20260915/renders/french_fries__oeterg.jpg |
| `french_fries__nfgqwu` | `none` | `z2` | z2: geom exact (mean 0.45% frac2 0.6%), tex borderline (MAD 5.8) | runlogs/symmetry_20260915/renders/french_fries__nfgqwu.jpg |
| `french_fries__uqsuef` | `none` | `z2` | z2: geom near (mean 0.54% frac2 1.3%), tex borderline (MAD 8.7) | runlogs/symmetry_20260915/renders/french_fries__uqsuef.jpg |
| `tennis_ball__rgekxe` | `y2 x4 z5 z6` | `full` | full: geom exact (mean 0.34% frac2 0.0%), tex borderline (MAD 3.7) | runlogs/symmetry_20260915/renders/tennis_ball__rgekxe.jpg |
| `tray__xzcnjq` | `none` | `z2` | z2: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 8.5) | runlogs/symmetry_20260915/renders/tray__xzcnjq.jpg |
| `tray__mkdcha` | `z2` | `x2 y2` | x2: geom near (mean 0.65% frac2 1.1%), tex preserved (MAD 0.7); y2: geom near (mean 0.65% frac2 1.1%), tex preserved (MAD 0.7) | runlogs/symmetry_20260915/renders/tray__mkdcha.jpg |
| `lemon__pwqedq` | `none` | `y2 x4 x5 x6` | y2: geom near (mean 0.54% frac2 0.0%), tex borderline (MAD 3.4); x4: geom near (mean 0.61% frac2 0.3%), tex borderline (MAD 3.2); x5: geom near (mean 0.53% frac2 0.1%), tex borderline (MAD 3.1); x6: geom exact (mean 0.47% frac2 0.0%), tex borderline (MAD 3.0) | runlogs/symmetry_20260915/renders/lemon__pwqedq.jpg |
| `half_lime__dtyjkb` | `none` | `zinf` | zinf: geom near (mean 0.59% frac2 2.8%), tex borderline (MAD 6.9) | runlogs/symmetry_20260915/renders/half_lime__dtyjkb.jpg |
| `half_lime__wdadtn` | `none` | `zinf` | zinf: geom near (mean 0.57% frac2 2.6%), tex borderline (MAD 6.9) | runlogs/symmetry_20260915/renders/half_lime__wdadtn.jpg |
| `half_lime__mxmmcp` | `none` | `zinf` | zinf: geom near (mean 0.42% frac2 1.1%), tex borderline (MAD 6.6) | runlogs/symmetry_20260915/renders/half_lime__mxmmcp.jpg |
| `bowl__belcml` | `z4` | `zinf` | zinf: geom near (mean 0.70% frac2 2.1%), tex preserved (MAD 0.3) | runlogs/symmetry_20260915/renders/bowl__belcml.jpg |
| `bowl__eawgwj` | `z4` | `z6` | z6: geom near (mean 0.75% frac2 0.6%), tex preserved (MAD 0.4) | runlogs/symmetry_20260915/renders/bowl__eawgwj.jpg |
| `bowl__jpvcjv` | `none` | `z3` | z3: geom exact (mean 0.11% frac2 0.0%), tex borderline (MAD 8.0) | runlogs/symmetry_20260915/renders/bowl__jpvcjv.jpg |
| `saucepot__chjetk` | `z2` | `zinf` | zinf: geom near (mean 0.20% frac2 2.2%), tex preserved (MAD 0.3) | runlogs/symmetry_20260915/renders/saucepot__chjetk.jpg |
| `vase__stqkvx` | `z6` | `zinf` | zinf: geom near (mean 0.58% frac2 0.0%), tex preserved (MAD 0.3) | runlogs/symmetry_20260915/renders/vase__stqkvx.jpg |
| `cinnamon_stick__qxpzdm` | `none` | `x2 y2 z2` | x2: geom near (mean 0.41% frac2 0.0%), tex borderline (MAD 3.8); y2: geom near (mean 0.45% frac2 1.0%), tex borderline (MAD 3.4); z2: geom exact (mean 0.41% frac2 0.0%), tex borderline (MAD 3.8) | runlogs/symmetry_20260915/renders/cinnamon_stick__qxpzdm.jpg |
| `cinnamon_stick__kdaxdy` | `none` | `x2 y2 z2` | x2: geom near (mean 0.42% frac2 0.9%), tex borderline (MAD 3.9); y2: geom near (mean 0.51% frac2 0.4%), tex borderline (MAD 3.2); z2: geom near (mean 0.53% frac2 0.6%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/cinnamon_stick__kdaxdy.jpg |
| `cinnamon_stick__qmlyim` | `none` | `x2 z2` | x2: geom near (mean 0.47% frac2 0.8%), tex borderline (MAD 3.8); z2: geom near (mean 0.51% frac2 2.3%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/cinnamon_stick__qmlyim.jpg |
| `cinnamon_stick__qsqvgk` | `none` | `x2 y2 z2` | x2: geom near (mean 0.60% frac2 1.9%), tex borderline (MAD 3.8); y2: geom near (mean 0.59% frac2 1.8%), tex borderline (MAD 3.6); z2: geom exact (mean 0.35% frac2 0.0%), tex borderline (MAD 3.8) | runlogs/symmetry_20260915/renders/cinnamon_stick__qsqvgk.jpg |
| `cinnamon_stick__cdkjfo` | `none` | `x2 yinf` | x2: geom near (mean 0.51% frac2 0.6%), tex borderline (MAD 3.8); yinf: geom near (mean 0.64% frac2 0.8%), tex borderline (MAD 3.8) | runlogs/symmetry_20260915/renders/cinnamon_stick__cdkjfo.jpg |
| `cinnamon_stick__smfuqz` | `none` | `x2 yinf` | x2: geom near (mean 0.42% frac2 0.8%), tex borderline (MAD 3.8); yinf: geom near (mean 0.66% frac2 1.3%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/cinnamon_stick__smfuqz.jpg |
| `cinnamon_stick__hjhcpm` | `none` | `z2` | z2: geom near (mean 0.52% frac2 0.5%), tex borderline (MAD 3.7) | runlogs/symmetry_20260915/renders/cinnamon_stick__hjhcpm.jpg |
| `pencil__rzscao` | `xinf` | `y2 z2` | y2: geom exact (mean 0.26% frac2 0.7%), tex borderline (MAD 3.5); z2: geom exact (mean 0.08% frac2 0.7%), tex borderline (MAD 2.5) | runlogs/symmetry_20260915/renders/pencil__rzscao.jpg |
| `pencil__lruiot` | `x5 x6` | `xinf` | xinf: geom exact (mean 0.33% frac2 0.0%), tex borderline (MAD 3.1) | runlogs/symmetry_20260915/renders/pencil__lruiot.jpg |
| `pencil__vvofwr` | `x5 x6` | `xinf` | xinf: geom exact (mean 0.38% frac2 0.0%), tex borderline (MAD 3.0) | runlogs/symmetry_20260915/renders/pencil__vvofwr.jpg |
| `marker__xakoic` | `none` | `xinf` | xinf: geom near (mean 0.86% frac2 1.6%), tex preserved (MAD 1.2) | runlogs/symmetry_20260915/renders/marker__xakoic.jpg |
| `marker__gldosk` | `none` | `x4 x5 x6` | x4: geom near (mean 0.57% frac2 0.8%), tex preserved (MAD 1.1); x5: geom near (mean 0.52% frac2 0.0%), tex preserved (MAD 1.1); x6: geom near (mean 0.47% frac2 0.2%), tex preserved (MAD 1.1) | runlogs/symmetry_20260915/renders/marker__gldosk.jpg |
| `marker__vrjwkg` | `none` | `xinf` | xinf: geom near (mean 0.87% frac2 2.7%), tex preserved (MAD 1.1) | runlogs/symmetry_20260915/renders/marker__vrjwkg.jpg |
| `marker__vjjsul` | `none` | `xinf` | xinf: geom near (mean 0.82% frac2 0.4%), tex preserved (MAD 1.3) | runlogs/symmetry_20260915/renders/marker__vjjsul.jpg |
| `marker__oyjbhy` | `none` | `xinf` | xinf: geom near (mean 0.70% frac2 2.3%), tex preserved (MAD 1.3) | runlogs/symmetry_20260915/renders/marker__oyjbhy.jpg |
| `marker__mznenj` | `none` | `xinf` | xinf: geom near (mean 0.81% frac2 0.3%), tex preserved (MAD 1.2) | runlogs/symmetry_20260915/renders/marker__mznenj.jpg |
| `pepper_shaker__tgcwtr` | `none` | `zinf` | zinf: geom exact (mean 0.23% frac2 0.0%), tex borderline (MAD 4.7) | runlogs/symmetry_20260915/renders/pepper_shaker__tgcwtr.jpg |
| `pepper_shaker__ykujyn` | `zinf` | `x2 y2` | x2: geom exact (mean 0.32% frac2 0.0%), tex borderline (MAD 10.2); y2: geom exact (mean 0.32% frac2 0.0%), tex borderline (MAD 10.2) | runlogs/symmetry_20260915/renders/pepper_shaker__ykujyn.jpg |
| `pepper_shaker__stbqbt` | `none` | `z5 z6` | z5: geom near (mean 0.55% frac2 2.1%), tex preserved (MAD 1.6); z6: geom near (mean 0.52% frac2 0.1%), tex preserved (MAD 1.6) | runlogs/symmetry_20260915/renders/pepper_shaker__stbqbt.jpg |
| `salt_shaker__iomwtn` | `z3` | `zinf` | zinf: geom exact (mean 0.21% frac2 0.0%), tex borderline (MAD 4.2) | runlogs/symmetry_20260915/renders/salt_shaker__iomwtn.jpg |
| `carrot__nktmff` | `none` | `x2` | x2: geom near (mean 0.61% frac2 0.3%), tex borderline (MAD 3.3) | runlogs/symmetry_20260915/renders/carrot__nktmff.jpg |
| `colored_pencil__acumgp` | `x4 x5 x6` | `y2 xinf` | y2: geom near (mean 0.50% frac2 1.0%), tex borderline (MAD 3.4); xinf: geom near (mean 0.48% frac2 0.0%), tex preserved (MAD 2.2) | runlogs/symmetry_20260915/renders/colored_pencil__acumgp.jpg |
| `colored_pencil__bnsqcn` | `z2 x4 x5 x6` | `y2 xinf` | y2: geom near (mean 0.50% frac2 1.0%), tex preserved (MAD 2.3); xinf: geom near (mean 0.48% frac2 0.0%), tex preserved (MAD 1.9) | runlogs/symmetry_20260915/renders/colored_pencil__bnsqcn.jpg |
| `colored_pencil__jnccuz` | `x4 x5 x6` | `y2 xinf` | y2: geom near (mean 0.50% frac2 1.0%), tex borderline (MAD 3.6); xinf: geom near (mean 0.48% frac2 0.0%), tex preserved (MAD 2.6) | runlogs/symmetry_20260915/renders/colored_pencil__jnccuz.jpg |
| `colored_pencil__kgmapz` | `x4 x5 x6` | `xinf` | xinf: geom near (mean 0.49% frac2 0.0%), tex preserved (MAD 2.5) | runlogs/symmetry_20260915/renders/colored_pencil__kgmapz.jpg |
| `colored_pencil__nssris` | `z2 x4 x5 x6` | `y2 xinf` | y2: geom near (mean 0.50% frac2 1.0%), tex borderline (MAD 2.9); xinf: geom near (mean 0.48% frac2 0.0%), tex preserved (MAD 1.9) | runlogs/symmetry_20260915/renders/colored_pencil__nssris.jpg |
| `colored_pencil__qrxemk` | `x4 x5 x6` | `xinf` | xinf: geom near (mean 0.51% frac2 0.0%), tex preserved (MAD 2.6) | runlogs/symmetry_20260915/renders/colored_pencil__qrxemk.jpg |
| `salt_shaker__qowouj` | `z4 z6` | `zinf` | zinf: geom near (mean 0.52% frac2 0.0%), tex preserved (MAD 0.8) | runlogs/symmetry_20260915/renders/salt_shaker__qowouj.jpg |
| `can_of_baking_mix__orwvfx` | `z2` | `zinf` | zinf: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 4.9) | runlogs/symmetry_20260915/renders/can_of_baking_mix__orwvfx.jpg |
| `can_of_baking_mix__rxorlp` | `z2` | `zinf` | zinf: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.8) | runlogs/symmetry_20260915/renders/can_of_baking_mix__rxorlp.jpg |
| `pop_case__gwgaqb` | `none` | `y2` | y2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 4.6) | runlogs/symmetry_20260915/renders/pop_case__gwgaqb.jpg |
| `pop_case__qcfxec` | `none` | `z2` | z2: geom exact (mean 0.13% frac2 0.0%), tex borderline (MAD 4.2) | runlogs/symmetry_20260915/renders/pop_case__qcfxec.jpg |
| `pop_case__pldtgy` | `none` | `x2` | x2: geom exact (mean 0.10% frac2 0.0%), tex borderline (MAD 3.9) | runlogs/symmetry_20260915/renders/pop_case__pldtgy.jpg |
