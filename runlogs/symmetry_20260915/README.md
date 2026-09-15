# Rotational-symmetry survey of the icl models (2026-09-15)

Which rotations of the canonical object frame leave a model looking identical — geometry **and** texture? Feeds two
rules of `ispatialgym/docs/instance_sampling_spec.md`: a mental-rotation target must not be rotationally symmetric
under the sampled rotation (the policy could not tell the goal orientation apart), and every orientation-evaluated
target gets `orientation_equivalences` (rotations under which the object looks identical) in its task record so a
symmetric placement is not scored as an error. Result: the `symmetry` / `symmetry_near` columns of
`ispatialgym/assets/icl/roles/candidates.csv` (`scripts/icl/build_icl.py --symmetry runlogs/symmetry_20260915/symmetry.json`,
the default when the file exists).

Scripts: `scripts/symmetry/` (`survey_symmetry.py` stages `models` / `geom` / `texture` / `report`, `symcore.py`
rotations + scores + thresholds, `meshload.py` offline USD loader factored out of the world-alignment render script).
`base` conda env (usd-core `pxr`, trimesh 4.11, scipy 1.17, pyrender 0.1.45 / EGL); no OmniGibson.

## Model set

Every distinct (category, model) of `roles/candidates.csv` with `kind` ∈ {`sampleable`, `group`} except the
block-design block (`ispatialgym-assets`), the disembedding trays (`source` / `destination`) and the small-scale
articulation targets (articulated): **1480 models in 392 categories**, 41 of them with a grasp-tested spawn `scale`
(evaluated at that scale as well). Per (subtask, role):

| subtask | role | models |
|---|---|---|
| camera_relocalization-obstructed | obstacles | 109 |
| object_scaling | target | 278 |
| mental_rotation | target | 196 |
| disembedding | target | 243 |
| disembedding | distractors | 603 |
| configuration_matching-attachment | target / partner | 30 / 19 |
| alignment-axial | target | 125 |
| alignment-world_vertical | target | 69 |
| alignment-world_horizontal | target | 69 |
| IORS ontop_small_scale | target / partner | 42 / 25 |
| IORS ontop_large_scale | target | 18 |
| IORS under_large_scale | target | 25 |
| IORS under_small_scale | target / partner | 20 / 23 |
| IORS inside | target / partner | 66 / 33 |
| IORS nextto | target / partner | 58 / 42 |
| IORS touching | target / partner | 34 / 58 |

(`models.json`; a model used by several roles is surveyed once — the column sums exceed 1480.)

## Method

**Frame.** The canonical object frame of the USD (the frame the sampler places objects in and the KB bounding boxes
refer to). Rotation axes = the frame axes through the bbox centre **and** through the surface centroid; the better
centre is kept per rotation (a 3- or 5-fold body's bbox centre is not on its axis, the centroid always is). Only
rotations about the frame axes are tested: a symmetry about a tilted axis is not found (this is what makes the
non-uniformly scaled pencils asymmetric, see *Scaled variants*).

**Geometry** (`geom/<key>.json`, CPU, 24 workers, 997 s for 1480 + 41 scaled models). Visual meshes of the USD
(collision / meta links skipped, OmniGlass parts **included** — a wineglass is all glass), 400 000 area-weighted
surface points as the reference (`cKDTree`), a 20 000-point subset as the query. For a rotation R about centre c the
score is the symmetric Chamfer distance — nearest-neighbour distance of R·P to the reference and of R⁻¹·P to the
reference (= P to R·P) — normalised by the bbox diagonal: `mean`, `p95` and `frac2` (fraction of points displaced by
more than 2 % of the diagonal). Distances are capped at 10 % of the diagonal and a 2 000-point one-directional
pre-test rejects rotations with a mean above 3 % (KD-tree queries far from every surface point are 20–30× slower). The
sampling noise floor (a fresh 20 000-point sample against the reference) is 0.09 % median, ≤ 0.2 % max.

Candidate rotations and tags: 180° about x, y, z (`x2 y2 z2`); 90° (`x4 y4 z4`); 120° / 72° / 60° about every axis
(`x3 y3 z3 x5 y5 z5 x6 y6 z6`); revolution about every axis (`xinf yinf zinf`: 8 fixed angles 17°, 39°, 53°, 101°, 131°,
163°, 197°, 251° — none a multiple of 30 / 45 / 60 / 72 / 90 —, all must pass); spherical (`full`: 8 fixed uniformly
random rotations). A tag's score is its worst rotation. Accepted tags are reduced to generators
(`symcore.reduce_tags`): `full` absorbs everything, `ainf` absorbs every finite fold about `a` and one of the two
perpendicular half turns (a plain cylinder → `zinf x2`), `a6` absorbs `a3` / `a2`, `a4` absorbs `a2`; finite folds about
different axes are kept (a box → `x2 y2 z2`); two coprime folds about one axis (`z5 z6`) mean the mesh is a 30-gon of
revolution whose `zinf` test failed on a detail (a ribbed cap, a seam) — the group they generate is the equivalence.

**Texture** (`tex/<key>.json`, GPU, 3 processes; final pass 2026-09-15 10:30, 40 min for the 1207 models with a
rotation within the near mean threshold; the v1 pass took 4.3 min at 192 px). pyrender `RenderFlags.FLAT` (colour =
albedo texture, no lighting, so shading cannot break a symmetry), white background, six fixed cameras (from ±x, ±y at
25° elevation and 20° azimuth offset; from ±z tilted 25°, so no face is edge-on), distance 1.55 × diagonal, **384 px**
rendered at 768 px and box-downsampled (bodies ≥ 4.5× longer than wide — pencils, rods — at 768 px). *Brightness bug,
found and fixed on 2026-09-15 (reviewer reports of "featureless black" trays and clipboards):* trimesh's
`SimpleMaterial` default diffuse colour (102, 102, 102) multiplied every texture in pyrender, so all textured renders
of the v1 pass and of the first v2 pass were at 40 % of their albedo brightness (a white cup rendered mid-grey, a dark
tray black); `meshload.py` now neutralises the factor, the renders are in true albedo units and the thresholds tuned on
the dark renders were rescaled by × 2.5 (`symcore.ALBEDO_FIX`) so the validated behaviour is kept — the brightness fix
alone flipped 81 (model, tag) texture classes (63 preserved → broken, finer quantisation of bright textures) and 37
models' verdicts (`superseded/v2a/` holds the dark-render outputs). The model and the model rotated by every rotation
of the candidate tag are rendered with the same cameras and compared on the pixels **fully covered by the object in
both renders** (partially covered silhouette pixels mix the background in and move with the sub-pixel position of the
edge; for an exact geometric symmetry the two silhouettes coincide anyway — the raw union-mask `mad_union` and the mask
IoU are recorded too). Both renders are blurred with a Gaussian of σ = 1.0 mm at the render scale (a person at
0.5–1.5 m does not resolve pixel-scale speckle, grain or mottle), then:

- `mad` / `p95` / `p99` of the per-pixel channel-mean |diff| (the v1 statistics, on the blurred renders);
- the **local-feature test** (v2): D = 3 × 3 median of the channel-max |diff|; a pixel is *hot* when D > 62/255 (a
  high-contrast feature; 25/255 in the dark-render units the rule was tuned in) **or** when the local standard
  deviation of D over a 5 mm window exceeds 15/255 (6/255 dark units: a low-contrast but *structured* feature —
  dark-on-dark text, a pale print; a smooth baked-shading gradient has D of the same size but no local structure);
  the hot mask is opened with a 2.5 mm disc (facet edges of low-poly bodies, rim corners and grain streaks are thinner
  than that, print strokes and logos are not; high-contrast pixels with D > 150/255 — a black stroke on a light face —
  are exempt down to a 1 mm opening) and its largest connected blob is measured: `blob_px`, its share of the view
  (`blob_frac`), the camera (`blob_view`), its mean D (`blob_d`),
  which test lit it (`blob_kind`), its side in mm (`blob_mm`) and `n6` = the pixel area of a 6 × 6 mm patch at the
  render scale (`m_per_px` = 2 · dist · tan(fov/2) / size);
- the **silhouette test** (v2): the XOR of the two coverage masks — a part of the object present in one render and
  absent in the other: a pencil tip swapped with the flat end, a pump nozzle, a swing-top bail, a keyboard flipped
  face-down — opened with a 4 mm disc (`sil_px`, a shape difference at least that thick) and a 3 mm disc
  (`sil_px_weak`), largest blob in pixels, `sil_view`, `sil_mm`. The v1 percentile statistics only saw pixels covered
  in both renders and were blind to these (the first review round found the end-to-end flips of coloured pencils,
  drumsticks, dip candles and the pump heads of dispensers "preserved" by the survey).

OmniGlass parts are left out of the texture renders (they are transparent in Isaac Sim: the clock face under its glass
cover is what the policy sees — rendered opaque, the glass hid the numerals and two numeral clocks came out `xinf`); an
all-glass model is rendered in a uniform tint, its visual symmetry is its geometric one. Textures larger than 1024 px
are downsampled on load.

*Rejected in the v2 tuning* (code paths kept behind options, off by default): (1) a global per-view **contrast
normalisation** (object 2nd–98th percentiles stretched to 0–255, gain ≤ 4 or ≤ 2): it amplifies smooth baked shading
and stochastic texture exactly like a print — a dark-grey 12-gon beaker, a mottled apple, a wooden bat and a jar of
speckled seeds (all `preserved` by the reviewers) came out `broken`; the structural local-std test achieves the
intended sensitivity for dark-on-dark labels without it. (2) **Depth matching** (pixels whose depth differs by > 1.5 %
of the diagonal excluded as "geometry"): after a symmetry rotation a pixel legitimately shows another part of the
object — a thin painting flipped by `y2` shows its white back at a depth that differs by its thickness and the mask
hid the picture (`painting/fwuole` y2 came out preserved).

**Thresholds** (all fractions of the bbox diagonal, resp. of 255; tuned on the cases below):

| class | rule |
|---|---|
| geometry `exact` | mean ≤ 0.5 % and p95 ≤ 1.2 % and frac2 ≤ 1 % |
| geometry `near` | mean ≤ 1.5 % and frac2 ≤ 3 % (and not exact) |
| texture `preserved` | MAD < 7.5 and p95 < 50 and p99 < 150 (albedo /255; = 3 / 20 / 60 in the dark-render units the rule was tuned in) and blob < n6 / 4 (no structured difference ≥ 3 × 3 mm) and no ≥ 3 mm-thick silhouette difference of ≥ 6 × 6 mm |
| texture `borderline` | MAD ≤ 30 and p95 ≤ 62 and p99 ≤ 150 (12 / 25 / 60 dark units) and blob < 4 · n6 and not (blob ≥ n6 with mean D ≥ 75, i.e. 30 dark units) and no ≥ 4 mm-thick silhouette difference of ≥ 6 × 6 mm — and not preserved |
| texture `broken` | otherwise: a difference blob ≥ 4 × (6 × 6 mm), or ≥ 6 × 6 mm with mean D ≥ 75/255, or a ≥ 4 mm-thick silhouette difference of ≥ 6 × 6 mm, or percentiles beyond the borderline band |
| `flat_render` | the six base views are (near-)uniform (modal colour ≥ 85 % of the object pixels or colour std ≤ 6/255) **and** a map the albedo render cannot show carries structure (reflection / emissive used-texel std > 0.03, or diffuse texels that vary while the render does not): every `symmetry` of such a model is held in `symmetry_near`, no review is issued for it, until a render with those maps exists. A uniform render of uniform maps is `plain_albedo` (a genuinely plain object; the render is faithful) |

The spec's mean thresholds (0.5 % / 1.5 %) are kept; `frac2` was added because a mean over the whole surface dilutes a
small asymmetric part — a mug handle is 6 % of the surface displaced by 20 % of the diagonal and scores a mean of
1.25–1.4 % under z4 / z5 / z6 (would be "near"), frac2 = 6–24 %; a 16-gon bowl rotated by 17° has frac2 = 2 %. The
texture percentile rules are the v1 ones (MAD < 3 / p95 < 20 / p99 < 60 for `preserved`; the borderline band closed
by p95 ≤ 25 and p99 ≤ 60 because a muted label moves the mean by only 7–12/255 while ≥ 5 % of the pixels differ by
> 25/255). The v2 blob rule was tuned on: `bottle_of_beer/fgzjnb` z6 (near-black bottle, dark-red logo; v1 MAD 2.8,
p95 18, p99 30 → *preserved*; v2 blob 43 × n6 → broken), `bottle_of_face_cream/dztaed` zinf ("Put your Own Label"
text wrapping the side, contrast ≈ 15/255; v2 blob 13 × n6 by the structural test → broken),
`bottle_of_coconut_water/lsixio` z4 (a glyph on two faces; blob 18 × n6 → broken), medicine bottle, essential-oil
bottles, bandage pad, board games (broken); plain teacup / plate / bowl / cap / bar of soap / pencil / wineglass /
biscuit / seed jar, the twice-printed `can_of_baking_mix/blrqqz`, `can_of_coffee/zubwua` and `box_of_baking_mix/fmewsz`
(preserved); silhouette controls: drumstick / keyboard / cinnamon-stick flips, a dispenser's pump nozzle under yaw, a
swing-top bottle (broken); plate, bowl, teacup, pencil about its axis (preserved). Known false brokens of the rule —
stochastic textures and slight shape irregularities that are asymmetric pixel by pixel but not to a person: the
mottled `apple/agveuv` x5, the wood-grain `baseball_bat/yalgei` xinf, the wicker `hamper/pqzyxx` z2, the calyx of
`beefsteak_tomato/pnrdxh` z6, the wire basket `hamper/djrdum` z2 (blobs 4–18 × n6 with mean D 13–22, or wire
silhouettes); the 12-gon `beaker/effbnc` zinf, the lopsided `bar_soap/feqemg` z2 (2.5 mm silhouette crescent) and the
mottled `lemon/asdada` x6 land in borderline. Known misses (features below what pixel statistics can separate from
rendering noise): faint engraved line drawings and digits on the letter blocks (`toy_dice/bjrxkj` x2 / z2, D ≈ 12–14
after the blur, reviewers: broken), a wrapper's faint brand text (`pack_of_chocolate_bar/bxgewr` z2), a marker's thin
cap clip (`marker/zvvovh`, < 3 mm), a thin pencil's end-to-end flip (`colored_pencil/acumgp` y2, tip 4 mm) and a dip
candle's taper reversal — these stay preserved / borderline. The reviewer verdicts of the first round settle the
reviewed ones (a human verdict wins in the aggregation); the rest err on the strict side — a false `broken` costs an
equivalence, a false `preserved` would relax the evaluation.

**Final verdict** (`report` stage, re-runnable without new renders): `symmetry` = geometry exact ∩ texture preserved;
`symmetry_near` = geometry exact ∩ texture borderline, plus geometry near ∩ texture preserved-or-borderline (minus
what `symmetry` already implies); both reduced to generators, `none` when empty. A model with a spawn `scale`
(`scale` in `symmetry.json`) gets `symmetry_scaled` / `symmetry_near_scaled` from the scaled mesh; `build_icl.py`
writes the scaled verdict into the rows whose `scale` matches.

## Outputs

- `symmetry.json` — `{"<category>__<model>": {geom_exact, geom_near, texture_preserved, symmetry, symmetry_near,
  scores: {tag: {geom: {mean, p95, frac2}, centre, worst, tex: {mad, p95, p99, mad_union, iou, worst}, tex_class}},
  floor, diag, bbox, scale, symmetry_scaled, symmetry_near_scaled, scaled: {...}, uses, render, n_faces, n_meshes}}`
  (scores only for the rotations that passed the 3 % pre-test; raw tag lists in `*_raw`).
- `symmetry.csv` — one row per model: category, model, symmetry, symmetry_near, geom_exact, geom_near,
  texture_preserved, scale, symmetry_scaled, symmetry_near_scaled, floor_mean, best tag with its geometry / texture
  scores, n_faces, render, uses.
- `report.md` — the tables below in full (per-tag counts, scaled variants, sanity table, mental-rotation targets,
  `symmetric_180` comparison, failures, the 302-model review list with the reason per tag).
- `renders/<category>__<model>[__scaled].jpg` — contact sheet for every model with a geometric candidate (984 sheets,
  66 MB, each ≤ 100 KB): top row the model from the six cameras, one row per shown tag with the worst rotation of the
  tag and its geometry / texture scores and classes (as of the final thresholds), up to 10 rows (reduced exact + near
  tags, texture-borderline tags, then the other rendered tags).
- `models.json`, `geom/`, `tex/` — the model set and the per-model raw scores (re-tune thresholds in `symcore.py`, run
  `report` again: no recomputation needed).
- `dropped_mental_rotation.csv` — written by `scripts/icl/build_icl.py`: the `mental_rotation` / `target` rows removed
  from `roles/candidates.csv` because their `symmetry` is confirmed (spec §3.13 rule 1; candidates columns + `reason`).
- `superseded/v1/` — the first pass (192 px, percentile statistics only, 40 %-brightness renders): `symmetry.json`,
  `symmetry.csv`, `report.md`, the per-model `tex/` records and the forced-row file; `superseded/v2a/` — the first v2
  pass (blob + silhouette tests, still at 40 % brightness); `report_diff.md` compares v1 with the current pass;
  `assign/render_layout.json` is the v1 sheet layout the re-renders preserved for the issued review batches.
- `flat_render.json` (`survey_symmetry.py flatcheck`) — per model the colour statistics of the six base renders
  (std, modal colour and its share, luminance), the statistics of the texels the mesh actually uses in its diffuse /
  reflection / metalness / emissive maps, the material shader ids and texture inputs, and the `flat` verdict.

## Results (v2 pass, final 2026-09-15 10:30; the v1 pass is under `superseded/v1/`, the dark-render v2 pass under `superseded/v2a/`)

1480 models surveyed, 0 loader failures (the 18 all-glass wineglasses / bottles that the world-alignment loader
dropped are loaded here), 0 render failures. Survey alone: **243 models have a final `symmetry`, 1237 are `none`, 190
have a `symmetry_near` entry** (v1: 377 / 1103 / 302; 17 of the near models are `flat_render` demotions, see below);
965 models have at least one geometric candidate (sheet). With the reviewer verdicts of the first round applied
(`review_verdicts.json`, still the rev_00–10 aggregation): 362 symmetric models, 79 with a `symmetry_near` entry.

| tag | `symmetry` (survey) | `symmetry_near` (review) | geometry exact (before texture) |
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
| `none` | 1237 | – | 624 |

(The `symmetry_near` column counts a tag once per model; 62 of the near tags on 24 models are *relaxations held for
review*: the v2 criterion calls them preserved but the v1 pass did not — a re-tuned criterion may only tighten on its
own, `relaxed_awaiting_review` in `symmetry.json`.)

Texture is the main filter: 856 models are geometrically symmetric under some rotation, 261 stay symmetric with their
texture before the review (labels, prints, wood grain, seams, tips break the rest).

### v1 → v2 diff (`report_diff.md`)

The first review batch found false positives in the v1 *final* verdicts — `bottle_of_beer/fgzjnb` z6 (near-black
bottle, dark-on-dark logo: MAD 2.8, p95 18, p99 30 → preserved), `bottle_of_face_cream/dztaed` zinf (low-contrast text
wrapping the side), `bottle_of_coconut_water/lsixio` z4 (a glyph on two faces). The v2 texture stage (384 px,
perceptual blur, blob + silhouette tests, see *Method*) re-scored every (model, tag) with a geometric candidate:

| flip of the texture class | (model, tag) pairs |
|---|---|
| preserved → broken | 979 |
| preserved → borderline | 230 |
| borderline → broken | 868 |
| borderline → preserved | 76 (held in `symmetry_near` as relaxations, see above) |
| broken → borderline | 28 |
| broken → preserved | 19 (held in `symmetry_near`) |

352 models changed their final verdict; tags removed from `symmetry`: `z2` 50, `zinf` 36, `x2` 31, `z4` 19, `x4` 15,
`y2` 14, `z6` 12, `z5` 11, `x6` 7, `xinf` 5, `x5` 4, `z3` 3, `y4` 2, `yinf` 1; none added. 101 models gained a
`symmetry_near` tag (4 of them were `none` without a near entry before); the three reported false positives are
`broken` now (blobs 47 ×, 13 × and 18 × a 6 × 6 mm patch).

**Agreement with the reviewer verdicts** (all 454 verdicts of `rev_00..10`, judged on the final texture class): 287
agree (240 broken, 47 preserved), 120 disagree — 100 where the survey is the stricter side (reviewer preserved / survey
broken: mottled apple, wooden bat, tomato calyx, peppercorn block, wicker hampers, low-poly bottles whose facet corners
the reviewers accepted, both-sides-alike prints the survey finds slightly misaligned) and 20 the other way (reviewer
broken / survey preserved: the faint engraved letter blocks, a wrapper's faint text, marker cap clips, pencil / candle
end-to-end flips — features below what the pixel statistics separate from rendering noise), 180 land in `borderline`,
4 are `unsure`. The human verdict wins in the aggregation either way (the reviewers judged the 40 %-brightness sheets;
the re-rendered sheets are brighter with the same layout).

### Flat renders (`flat_render.json`, `survey_symmetry.py flatcheck`)

Reviewers reported `clipboard/gmxyfo` and `tray/mkdcha` as featureless black. Both have a diffuse map that the loader
does apply (their used texels are uniform dark grey, 41/255 and 38/33/28), and both were rendered at 40 % brightness by
the material bug above — they are dark, plain, glossy objects (VRay material: black-ish diffuse, the visible colour in
the simulator comes from a uniform reflection map, 119 resp. 160/255), not untextured ones. Across the survey, **301 of
1480 models render (near-)uniform** after the fix (modal colour ≥ 85 % of the object pixels or colour std ≤ 6/255;
median luminance 0.75 — mostly white cups, plates, caps, lids): **261 are `plain_albedo`** (the texels the mesh uses are
uniform in every map, 48 of them all-glass models drawn in a uniform tint — the render is faithful, their symmetries
stand) and **40 are `flat_render`** — a reflection map with structure (used-texel std > 0.03; 39 models: trash cans,
hampers, notebooks, mice, saucepots, dip candles, the clipboard, `keyboard/uwacaq` …) or, once, diffuse texels that vary
while the render does not (`bag_of_snacks/pkwgid`, the only loader-side suspect; its varying texels are on faces the six
cameras do not see). Every `symmetry` of a `flat_render` model is held in `symmetry_near` and no review is issued for
it until a render with the reflection maps exists (17 models demoted: `trash_can/djgllo`, `trash_can/gilsji`,
`lid/wtjthb`, `lid/snklci`, `bottle_of_wine/uoykwx`, `bottle_of_wine/jdqahn`, `notebook/zkmmnf`, `hardback/acaxtl`,
`canister/xcppkc`, `dip_candle/ghctun`, `dip_candle/mxjejs`, `keyboard/uwacaq`, `clipboard/gmxyfo`, `tray/zcmnji`,
`saucepot/tfzijn`, `saucepot/uvzmss`, `salt_shaker/fonexa`; the other 23 had no symmetry to hold); the aggregation
holds reviewer `preserved` verdicts on them the same way. All 1480 assets use the `OmniGibsonVRayMtl` MDL (114 with an
`OmniGlass` part); no asset lacked a bound diffuse texture, none failed to decrypt or resolve.

### Sanity table (~30 categories with an obvious answer; the survey's most common verdicts, before the review)

| category | models | most common `symmetry` | most common geometry-exact | expectation | match |
|---|---|---|---|---|---|
| bowl | 12 | `zinf` ×6, `none` ×3, `z4` ×2 | `zinf` ×9, `z4` ×2, `z2` ×1 | zinf (plain bowls; a decorated rim breaks it) | yes (the `none` ones are decorated; `z4` = 8-gon meshes whose `zinf` the silhouette test now calls broken — the reviewer's "polygon count" call restores it) |
| plate | 12 | `zinf` ×5, `none` ×4, `z4` ×2 | `zinf` ×9, `z4` ×3 | zinf | yes (`z4` = polygonal meshes, `none` = patterned) |
| mug | 10 | `none` ×10 | `none` ×10 | none (handle) | yes |
| teacup | 6 | `none` ×4, `zinf` ×2 | `zinf` ×3, `none` ×3 | zinf if handle-less, else none | yes (`teacup/cpozxi`'s 5× repeating pattern is `z5` borderline now → review) |
| teapot | 3 | `none` ×3 | `none` ×3 | none | yes |
| sunflower | 17 | `none` ×17 | `none` ×17 | none | yes |
| pencil | 13 | `xinf` ×5, `x5 x6` ×3, `x4 x5 x6` ×2 | `y2 xinf` ×12, `z2 x4 x5 x6` ×1 | revolution about the long axis (x) if unprinted | yes; the end-to-end flip `y2` is geometrically exact (a 4 mm tip on a 19 cm pencil) and only partly caught by the silhouette test → review / reviewer verdicts |
| bottle_of_medicine | 8 | `none` ×8 | `zinf` ×4, `none` ×2, `z5 z6` ×1 | zinf geometry; the label breaks it → none | yes |
| can_of_baking_mix | 6 | `z2` ×4, `none` ×2 | `zinf` ×3, `x2 zinf` ×3 | zinf geometry; label → none | asset-specific: four cans carry the label twice (front = back) → `z2`; on two the lid photo / fine print differs under the half turn → none |
| can_of_soda | 9 | `none` ×7, `zinf` ×1, `z2` ×1 | `z4` ×4, `zinf` ×3, `z2` ×1 | zinf geometry; label → none | yes (one unlabelled can) |
| board_game | 16 | `none` ×16 | `x2 y2 z2` ×13 | box geometry; artwork → none | yes |
| box_of_cookies | 14 | `none` ×12, `x2` ×1, `y2` ×1 | `x2 y2 z2` ×6, `x2 y2 z4` ×4 | box geometry; print → none | yes (two boxes with an identical top / bottom or front / back print) |
| box_of_cereal | 14 | `none` ×14 | `x2 y2 z2` ×14 | box geometry; print → none | yes |
| apple | 7 | `none` ×7 | `none` ×7 | none / near zinf (stem) | yes |
| orange | 2 | `none` ×1, `full` ×1 | `x2 zinf` ×1, `full` ×1 | full or zinf; texture decides | yes (the oblate one's stem dot now breaks it) |
| tennis_ball | 1 | `x2` | `full` | full (plain) or none (seam) | the seam breaks everything but one half turn (low contrast, MAD 2–4) → `x2` stays; reviewer: `full` broken |
| toy_dice | 20 | `none` ×19, `y2` ×1 | `x2 z2 y4` ×9, `none` ×6 | cube geometry; pips / letters → none | yes; the faint engraved `toy_dice/bjrxkj` y2 stays (reviewer: broken → removed by the verdict) |
| jar_of_grains | 12 | `none` ×12 | `none` ×12 | zinf geometry; label → none | none already in geometry (not bodies of revolution about a frame axis) |
| cap | 19 | `zinf` ×11, `z4 z5 z6` ×2, `z2` ×2 | `zinf` ×12, `z2` ×2 | zinf | yes (`z4 z5 z6` = 12-gon caps whose `zinf` the silhouette test holds for review) |
| lid | 17 | `zinf` ×8, `z2` ×6, `none` ×2 | `zinf` ×9, `z2` ×6 | zinf | yes (`z2` = oval / rectangular lids) |
| keyboard | 6 | `none` ×5, `z2` ×1 | `none` ×4, `y2` ×1, `z2` ×1 | none | `keyboard/uwacaq` (dark slab, faint legends) keeps `z2` in the survey; the reviewer read the legends → broken, applied |
| painting | 11 | `none` ×11 | `x2 y2 z2` ×6, `x2` ×2, `x4` ×2 | none | yes |
| beaker | 10 | `zinf` ×6, `z2` ×2, `none` ×1 | `zinf` ×7 | zinf | yes (the dark 12-gon `beaker/effbnc` is borderline: facet shading) |
| bar_soap | 4 | `x2 y2 z2` ×2, `none` ×1, `x2 zinf` ×1 | same | zinf or x2 y2 z2 | yes (`bar_soap/feqemg` z2 borderline: a 2.5 mm silhouette crescent; reviewer preserved) |
| wineglass | 12 | `zinf` ×12 | `zinf` ×12 | zinf | yes (all glass, geometry only) |
| egg | 1 | `none` | `none` | zinf about the long axis | no: the egg mesh is not a body of revolution about a frame axis (tilted / bumpy) |
| wall_clock | 5 | `none` ×5 | `xinf` ×2, `y2 xinf` ×2, `x4 x3` ×1 | none (numerals) | yes (the open-work 12-marker clock `xevvel` is borderline now) |
| vase | 6 | `none` ×2, `z4` ×2, `z2` ×1 | mixed | zinf unless decorated | yes |
| platter | 6 | `none` ×3, `z2` ×2, `z3` ×1 | `z2` ×5, `z3` ×1 | z2 (oval) | partly: three oval platters lost `z2` to faint rim details |
| tray | 12 | `z2` ×8, `none` ×2, `zinf` ×1 | `z2` ×9 | z2 / x2 y2 z2 (rectangular) | yes |

### `mental_rotation` targets with a symmetry (removed from the candidate list, spec §3.13 rule 1)

Survey: 3 of 196 — `teacup/zdvgol` (`zinf`), `teacup/tfzfam` (`zinf`), `hook/wmvozo` (`zinf`). Review-confirmed near
tags: `pickle/ifterk` (y4, pick 1), `pickle/iobrqp` (y4, pick 2), `tongs/igsklx` (x2, pick 2), `tongs/gbjqyn` (x2).
`build_icl.py` drops the seven (`dropped_mental_rotation.csv`) and promotes `teacup/oxfzfe` → representative,
`teacup/vckahe` → test, `pickle/vuspsy` → representative + test, `pickle/rxuimv` → test (tongs keeps its one remaining
test model). `teacup/cpozxi` (z5, v1: symmetric) and `cookie_cutter/lqrfzo` (v1: `z2`) are candidates again — v2 puts
them in `symmetry_near` (supplementary batches). The reviewer notes of the selection sheet agree with the dropped ones
("handle-less, near-symmetric", "fully rotationally symmetric", "nearly rotationally symmetric, tiny outlier").

### `symmetric_180` (alignment targets) vs the survey

The half turn that matters is about the alignment axis: the wall normal = `frame_front` for the wall subtasks, the
vertical (`frame_up`, else canonical z) for `alignment-axial`. `symmetric_180` was set for 14 `alignment-world_horizontal`
rows only. The survey's disagreements went to the review round as explicit questions; the answered ones now set the
column (`preserved` → 1, `broken` → 0; `review_verdicts.json` → `build_icl.py`): `clipboard/gmxyfo` → 1,
`towel_rack/kqrmrh` → 1, `towel_rack/crivoa` → 1, `whiteboard/gvkgsi` → 1 (unflagged, symmetric), `shelf/qrmiur`,
`shelf/iruewm`, `shelf/wvkefp` → 0 (flagged 1, the reviewers judged the planks' knots / end lips visible),
`keyboard/uwacaq`, `notebook/ffdjzp`, `notebook/gqkaqu`, `wall_clock/xevvel`, `radiator/dwmloz`, `drumstick/jauzlc` → 0
(the survey's half turn was rejected by the reviewers); `rolling_pin/mwrdae` and `towel_rack/quznqx` stay at the sheet's
1 (reviewer verdicts preserved). The remaining v2 disagreements (14 rows, 6 of them new) are questions in the
supplementary batches (`report.md` lists the current table).

### Scaled variants

41 models carry a grasp-tested spawn scale (garlic cloves uniform, `cap/xxqyrt` z-stretched, french fries,
cinnamon sticks and pencils non-uniform). The verdict changes for 21 of them, all non-uniform: the pencils lose
`xinf` / `x5 x6` (the canonical pencil axis is slightly tilted against the frame's x axis; the y × 3.4–4.2, z × 5.0–5.1
scale that makes them 2 cm thick amplifies the tilt into a visible wobble under every rotation about the frame axis —
the spawned rod is still symmetric about its own axis, but not about a frame axis, which is what
`orientation_equivalences` are expressed in), cinnamon sticks and fries change their near set. `build_icl.py` writes
the scaled verdict into the rows that carry the matching `scale`.

## Runtime

`models` < 1 s; `geom` 997 s on 24 CPU workers (≈ 0.7 s wall per model, 16 core-seconds: 400k + 20k + 20k surface
samples, ≈ 20 coarse and ≈ 5–40 full KD-tree passes); `texture` v2 41 min on 3 GPU processes (≈ 0.3 GB each, 1207
models, 6 × (1 + Σ rotations) renders at 768 px + the blur / median / local-std / opening filters — CPU-bound, GPU
utilisation ≈ 15 %; the v1 pass at 192 px took 4.3 min); `flatcheck` 4 min on 3 GPU processes (192 px base views of
all 1480 models + used-texel sampling); `report` / `diff` 2 s each. Everything is idempotent (`--force` recomputes);
the full survey is ≈ 1 h on the local 4090 box.

## Review round

- `review_contract.md` — the reviewer's question (identical from the robot's head camera at 0.5–1.5 m?), the three
  verdicts `preserved` / `broken` / `unsure`, rules of thumb, the tag / axis convention, the output schema.
- `assign/rev_00.json … rev_10.json` (`scripts/symmetry/prepare_review.py`, first round on the v1 survey) — 11 batches
  of ≤ 29 entries, 319 entries: the 302 v1 models with a `symmetry_near` entry, 9 spawn-scaled variants
  (`<key>__scaled`, judged on `renders/<key>__scaled.jpg`) and 8 further alignment targets of the 15 v1 `symmetric_180`
  disagreement rows; 454 candidate tags. **All 11 are reviewed** (`review/rev_00..10.json`: 282 broken, 168 preserved,
  4 unsure). The v2 re-render kept every issued sheet's row order and displayed rotations
  (`assign/render_layout.json`), so the batches stayed valid; the labels and the top-row renders were refreshed.
- `assign/rev_11.json … rev_14.json` (`prepare_review.py --supplement`) — 4 supplementary batches, 94 entries / 136
  candidate tags no earlier batch asked about: the v2 `symmetry_near` tags (tags that moved from preserved to
  borderline, the relaxations held for review, newly borderline models) and 6 new `symmetric_180` questions; 1 scaled
  variant. 55 of the 94 entries are models no earlier batch covered, 39 add tags to already reviewed models.
  `assign/rev_15.json` — 11 entries / 16 tags that the brightness fix moved into `symmetry_near` (pencil end-to-end
  flips and quarter turns, a spice bottle, a cinnamon stick). `flat_render` models are left out of every batch.
- `review/overrides.json` — coordinator decisions (`{"<key>": {"<tag>": {"verdict", "note", "by": "coordinator"}}}`),
  applied by `aggregate_review.py` after the reviewer files (an override wins and is listed in `review_summary.md`);
  first entry: `bottle_of_water/cyzaue` `zinf` → broken (hexagonal prism, `z6` stays).
- `scripts/symmetry/aggregate_review.py` merges `review/*.json` into `review_verdicts.json` + `review_summary.md`
  (`preserved` → joins `symmetry`, `broken` → leaves `symmetry` and `symmetry_near`, `unsure` → stays / moves to
  `symmetry_near`; `symmetric_180` value per alignment subtask; unreviewed and partially reviewed entries; verdicts on
  tags the v2 survey no longer lists as candidates are kept, applied and flagged — 304 of the 454, the v2 criterion is
  that much stricter). `build_icl.py --symmetry` uses `review_verdicts.json` when it sits next to `symmetry.json`.

## Caveats for the review round

- Only frame-axis rotations are tested; a symmetry about a tilted axis (egg, scaled pencils) is reported as `none`.
- Low-poly meshes (181 models < 200 faces) are polygons of revolution: the finite fold is exact; `zinf` lands in
  `symmetry_near` (geometry near) or, when the corner sagitta exceeds 4 mm (an 8-gon bowl), fails the v2 silhouette
  test — the reviewer's rule 1 ("polygon count is not a difference") restores it, and the aggregation keeps such
  verdicts even though the survey no longer lists the tag as a candidate.
- Flat shading hides relief; a feature that is geometry only (key caps, embossing) is judged by the geometric score.
- OmniGlass parts are opaque in the geometry sample but absent from the texture renders (transparent in Isaac Sim);
  refraction effects are not modelled.
- Texture statistics are computed on the blurred, fully covered pixels of both renders; shape differences are caught
  by the silhouette test (≥ 3–4 mm thick, ≥ 6 × 6 mm), thinner ones (a pencil tip, a wick, a marker clip) are not —
  the union-mask MAD (`mad_union`) and the mask IoU are recorded for reference.
- Pixel statistics cannot separate faint engraved / low-contrast prints (D ≲ 15/255 after the blur) or stochastic
  textures (mottle, grain, weave) from rendering noise; the former stay preserved / borderline, the latter become
  broken — the review verdicts settle both where they exist.
