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

**Texture** (`tex/<key>.json`, GPU, 3 processes, 4.3 min for the 1207 models with a rotation within the near mean
threshold). pyrender `RenderFlags.FLAT` (colour = albedo texture, no lighting, so shading cannot break a symmetry),
white background, six fixed cameras (from ±x, ±y at 25° elevation and 20° azimuth offset; from ±z tilted 25°, so no
face is edge-on), distance 1.55 × diagonal, 192 px rendered at 384 px and box-downsampled (bodies ≥ 4.5× longer than
wide — pencils, rods — at 384 px, since a 19 cm pencil is 4 px thick at 192 px). The model and the model rotated by
every rotation of the candidate tag are rendered with the same cameras and compared: `mad` / `p95` / `p99` of the
per-pixel channel-mean |diff| over the pixels **fully covered by the object in both renders** (partially covered
silhouette pixels mix the background in and move with the sub-pixel position of the edge; for an exact geometric
symmetry the two silhouettes coincide anyway — the raw union-mask `mad_union` and the mask IoU are recorded too, and
the union MAD is what a 16-gon bowl's `zinf` fails on: IoU 0.945, union MAD 12/255, texture MAD 0.3/255). OmniGlass
parts are left out of the texture renders (they are transparent in Isaac Sim: the clock face under its glass cover is
what the policy sees — rendered opaque, the glass hid the numerals and two numeral clocks came out `xinf`); an
all-glass model is rendered in a uniform tint, its visual symmetry is its geometric one. Textures larger than 1024 px
are downsampled on load.

**Thresholds** (all fractions of the bbox diagonal, resp. of 255; tuned on the cases below):

| class | rule |
|---|---|
| geometry `exact` | mean ≤ 0.5 % and p95 ≤ 1.2 % and frac2 ≤ 1 % |
| geometry `near` | mean ≤ 1.5 % and frac2 ≤ 3 % (and not exact) |
| texture `preserved` | MAD < 3 and p95 < 20 and p99 < 60 |
| texture `borderline` | MAD ≤ 12 and p95 ≤ 25 and p99 ≤ 60 (and not preserved) |
| texture `broken` | otherwise |

The spec's mean thresholds (0.5 % / 1.5 %) are kept; `frac2` was added because a mean over the whole surface dilutes a
small asymmetric part — a mug handle is 6 % of the surface displaced by 20 % of the diagonal and scores a mean of
1.25–1.4 % under z4 / z5 / z6 (would be "near"), frac2 = 6–24 %; a 16-gon bowl rotated by 17° has frac2 = 2 %. The
texture rule keeps the spec's MAD < 3 / p95 < 20 for `preserved` and adds `p99 < 60`: features that cover 1–5 % of the
object's pixels (die pips, letters, clock numerals, small logos) barely move the mean and p95 (numeral clock: MAD 5.5,
p95 23, p99 71). The borderline band (spec: MAD in [3, 12]) is closed by p95 ≤ 25 and p99 ≤ 60 because a muted label
on a medicine bottle or a wrapper logo moves the mean by only 7–12/255 while ≥ 5 % of the pixels differ by > 25/255.
Tuning cases: plain bowl / plate / cap / beaker / plain box → preserved (MAD 0.1–1.1); can with a one-sided label,
medicine bottle, cereal / cookie boxes, board games, letter block (under 90°), magazine cover, licence plate →
broken; letter block under 180° (low-contrast grey letters: MAD 4, p95 20, p99 48), tennis-ball seam (MAD 3–4, p95
13–18), pencil tip under an end-to-end flip, candle wick → borderline (review). A can whose label is printed twice
(front = back) is genuinely `z2` (six `can_of_baking_mix` models, `can_of_coffee`, boxes of baking mix).

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

## Results

1480 models surveyed, 0 loader failures (the 18 all-glass wineglasses / bottles that the world-alignment loader
dropped are loaded here), 0 render failures. **377 models have a final `symmetry`, 1103 are `none`, 302 have a
`symmetry_near` entry to review**; 965 models have at least one geometric candidate (sheet).

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
| `none` | 1103 | – | 624 |

Texture is the main filter: 856 models are geometrically symmetric under some rotation, 377 stay symmetric with their
texture (labels, prints, wood grain, seams break the rest). The review list splits into 228 tag entries geometry-exact /
texture-borderline (low-contrast features: seams, tips, wicks, grey letters), 111 geometry-near / texture-preserved
(mostly low-poly bodies of revolution: 181 models have < 200 faces; 36 models are a finite fold with `zinf` near, e.g.
the 8-gon `bowl/belcml` = `z4` + near `zinf`) and 90 geometry-near / texture-borderline.

### Sanity table (~30 categories with an obvious answer; the most common verdicts)

| category | models | most common `symmetry` | most common geometry-exact | expectation | match |
|---|---|---|---|---|---|
| bowl | 12 | `zinf` ×6, `none` ×3, `z4` ×2 | `zinf` ×9, `z4` ×2, `z2` ×1 | zinf (plain bowls; a decorated rim breaks it) | yes (the `none` ones are decorated, `z4` = 8-gon meshes with `zinf` near) |
| plate | 12 | `zinf` ×5, `z4` ×3, `none` ×3 | `zinf` ×9, `z4` ×3 | zinf | yes (`z4` = polygonal meshes, `none` = patterned) |
| mug | 10 | `none` ×10 | `none` ×10 | none (handle) | yes |
| teacup | 6 | `none` ×3, `zinf` ×2, `z5` ×1 | `zinf` ×3, `none` ×3 | zinf if handle-less, else none | yes (`z5` = handle-less cup with a 5× repeating pattern, reviewer: "near-symmetric") |
| teapot | 3 | `none` ×3 | `none` ×3 | none | yes |
| sunflower | 17 | `none` ×17 | `none` ×17 | none | yes |
| pencil | 13 | `xinf` ×5, `x5 x6` ×2, `x4 x5 x6` ×2 | `y2 xinf` ×12, `z2 x4 x5 x6` ×1 | revolution about the long axis (x) if unprinted | yes; the end-to-end flip `y2` is geometrically exact (a 4 mm tip on a 19 cm pencil) but the tip colour makes it borderline → review |
| bottle_of_medicine | 8 | `none` ×8 | `zinf` ×4, `none` ×2, `z5 z6` ×1 | zinf geometry; the label breaks it → none | yes |
| can_of_baking_mix | 6 | `z2` ×6 | `zinf` ×3, `x2 zinf` ×3 | zinf geometry; label → none | asset-specific: the label is printed twice (front = back), so `z2` is right; `zinf` broken as expected |
| can_of_soda | 9 | `none` ×6, `zinf` ×1, `z2` ×1 | `z4` ×4, `zinf` ×3, `z2` ×1 | zinf geometry; label → none | yes (one unlabelled can) |
| board_game | 16 | `none` ×16 | `x2 y2 z2` ×13 | box geometry; artwork → none | yes |
| box_of_cookies | 14 | `none` ×12, `x2` ×1, `y2` ×1 | `x2 y2 z2` ×6, `x2 y2 z4` ×4 | box geometry; print → none | yes (two boxes with an identical top / bottom or front / back print) |
| box_of_cereal | 14 | `none` ×14 | `x2 y2 z2` ×14 | box geometry; print → none | yes |
| apple | 7 | `none` ×7 | `none` ×7 | none / near zinf (stem) | yes |
| orange | 2 | `x2 zinf` ×1, `full` ×1 | same | full or zinf; texture decides | yes (one is slightly oblate: `full` geometry near, its stem dot too small to break the texture) |
| tennis_ball | 1 | `y2 x4 z5 z6` | `full` | full (plain) or none (seam) | debatable: the seam is low-contrast (MAD 2–4, p99 23–25), some tags fall on each side of MAD 3 → `full` in `symmetry_near`, review |
| toy_dice | 20 | `none` ×19, `y2` ×1 | `x2 z2 y4` ×9, `none` ×6 | cube geometry; pips / letters → none | yes (letter blocks; the geometry-exact set says the block meshes are not perfect cubes) |
| jar_of_grains | 12 | `none` ×12 | `none` ×12 | zinf geometry; label → none | none already in geometry (the jars are not bodies of revolution about a frame axis: z2 mean 0.85 %, zinf 1.4 %, frac2 above the near bound) |
| cap | 19 | `zinf` ×13, `z2` ×2, `z5 z6` ×1 | `zinf` ×12, `z2` ×2 | zinf | yes |
| lid | 17 | `zinf` ×9, `z2` ×6, `none` ×1 | `zinf` ×9, `z2` ×6 | zinf | yes (`z2` = oval / rectangular lids) |
| keyboard | 6 | `none` ×5, `z2` ×1 | `none` ×4, `y2` ×1, `z2` ×1 | none | yes; `keyboard/uwacaq` is the reviewer's "very dark flat slab with faint key texture, little relief" → `z2` is real for that asset (axial alignment cannot tell its front from its back) |
| painting | 11 | `none` ×11 | `x2 y2 z2` ×6, `x2` ×2, `x4` ×2 | none | yes |
| beaker | 10 | `zinf` ×6, `z2` ×2, `z6` ×1 | `zinf` ×7 | zinf | yes |
| bar_soap | 4 | `x2 y2 z2` ×2, `none` ×1, `x2 zinf` ×1 | same | zinf or x2 y2 z2 | yes |
| wineglass | 12 | `zinf` ×12 | `zinf` ×12 | zinf | yes (all glass, geometry only) |
| egg | 1 | `none` | `none` | zinf about the long axis | no: the egg mesh is not a body of revolution about a frame axis (tilted / bumpy) |
| wall_clock | 5 | `none` ×4, `x4` ×1 | `xinf` ×2, `y2 xinf` ×2 | none (numerals) | yes for the four numeral faces; `wall_clock/xevvel` is an open-work face with 12 look-alike hour markers (geometry `x4 x3`), its `x4` sits exactly on the MAD 3 threshold (`x3` borderline) → review |
| vase | 6 | `none` ×2, `z4` ×1, `zinf` ×1 | mixed | zinf unless decorated | yes |
| platter | 6 | `z2` ×5, `z3` ×1 | same | z2 (oval) | yes |
| tray | 12 | `z2` ×7, `none` ×2, `x2 y2 z2` ×1 | `z2` ×9 | z2 / x2 y2 z2 (rectangular) | yes (`z2` only when the underside differs from the top) |

### `mental_rotation` targets with a symmetry (must be excluded or get `orientation_equivalences`)

5 of 196: `teacup/zdvgol` (`zinf`), `teacup/cpozxi` (`z5`, near `z6`), `teacup/tfzfam` (`zinf`), `hook/wmvozo`
(`zinf`), `cookie_cutter/lqrfzo` (`z2`). Five more have only a `symmetry_near` entry and should be reviewed before
use: `toy_figure/jddrck` (z4), `pickle/ifterk` (y4), `pickle/iobrqp` (y4), `tongs/igsklx` (x2), `tongs/gbjqyn` (x2).
The reviewer notes of the selection sheet agree ("handle-less, near-symmetric", "fully rotationally symmetric",
"nearly rotationally symmetric, tiny outlier").

### `symmetric_180` (alignment targets) vs the survey

The half turn that matters is about the alignment axis: the wall normal = `frame_front` for the wall subtasks, the
vertical (`frame_up`, else canonical z) for `alignment-axial`. `symmetric_180` is set for 14 `alignment-world_horizontal`
rows only; the column is left untouched. 15 rows disagree:

| subtask | category | model | half-turn axis | `symmetric_180` | survey | `symmetry` | `symmetry_near` |
|---|---|---|---|---|---|---|---|
| alignment-axial | keyboard | uwacaq | +z (up) | – | yes | `z2` | `x2 y2` |
| alignment-axial | notebook | ffdjzp | +z (up) | – | yes | `x2 y2 z2` | – |
| alignment-axial | notebook | gqkaqu | +z (up) | – | yes | `x2 y2 z2` | – |
| alignment-axial | clipboard | gmxyfo | +z (up) | – | yes | `x2 y2 z2` | – |
| alignment-world_vertical | wall_clock | xevvel | +x (front) | – | yes | `x4` | `x6` |
| alignment-world_vertical | towel_rack | kqrmrh | +x (front) | – | yes | `x2` | – |
| alignment-world_horizontal | rolling_pin | mwrdae | +x (front) | 1 | near | `none` | `x2 yinf` |
| alignment-world_horizontal | shelf | qrmiur | +z (front) | 1 | near | `none` | `x2 y2 z2` |
| alignment-world_horizontal | shelf | iruewm | +z (front) | 1 | near | `none` | `x2 y2 z2` |
| alignment-world_horizontal | shelf | wvkefp | +z (front) | 1 | no | `none` | – |
| alignment-world_horizontal | radiator | dwmloz | +x (front) | – | yes | `x2` | – |
| alignment-world_horizontal | drumstick | jauzlc | +z (front) | – | yes | `z2 x4 x5 x6` | `y2 xinf` |
| alignment-world_horizontal | whiteboard | gvkgsi | +x (front) | – | yes | `x2` | – |
| alignment-world_horizontal | towel_rack | crivoa | +x (front) | – | yes | `x2` | – |
| alignment-world_horizontal | towel_rack | quznqx | +x (front) | 1 | near | `none` | `x2 yinf` |

Five rows are flagged `symmetric_180 = 1` without a confirmed survey half turn (four are `near`: the geometry is
exact but wood grain / knots make the texture borderline, MAD 4–9; `shelf/wvkefp` is geometrically asymmetric, z2
frac2 5 %, and its texture broken); ten rows have a survey half turn the sheet does not record (plain notebooks, a
clipboard, a plain whiteboard, radiator, towel racks, the open-work clock, the dark keyboard — the axial and
world-vertical reviewers never filled the column, so these are additions rather than contradictions). The 9 flagged
rows that agree are not listed.

### Scaled variants

41 models carry a grasp-tested spawn scale (garlic cloves uniform, `cap/xxqyrt` z-stretched, french fries,
cinnamon sticks and pencils non-uniform). The verdict changes for 21 of them (11 in `symmetry`, the rest in
`symmetry_near`), all non-uniform: the pencils lose `xinf` / `x5 x6` (the canonical pencil axis is slightly tilted
against the frame's x axis; the y × 3.4–4.2, z × 5.0–5.1 scale that makes them 2 cm thick amplifies the tilt into a
visible wobble under every rotation about the frame axis — the spawned rod is still symmetric about its own axis, but
not about a frame axis, which is what `orientation_equivalences` are expressed in), cinnamon sticks and fries change
their near set. `build_icl.py` writes the scaled verdict into the rows that carry the matching `scale`.

## Runtime

`models` < 1 s; `geom` 997 s on 24 CPU workers (≈ 0.7 s wall per model, 16 core-seconds: 400k + 20k + 20k surface
samples, ≈ 20 coarse and ≈ 5–40 full KD-tree passes); `texture` 4.3 min on 3 GPU processes (≈ 0.3 GB each, 1207
models, 6 × (1 + Σ rotations) renders per model; 1.5–3 s for the 384 px pencils); `report` 2 s. Everything is
idempotent (`--force` recomputes); the full survey is ≈ 25 min on the local 4090 box.

## Review round (prepared, not yet run)

- `review_contract.md` — the reviewer's question (identical from the robot's head camera at 0.5–1.5 m?), the three
  verdicts `preserved` / `broken` / `unsure`, rules of thumb, the tag / axis convention, the output schema.
- `assign/rev_00.json … rev_10.json` (`scripts/symmetry/prepare_review.py`) — 11 batches of ≤ 29 entries, 319 entries in
  all: the 302 models with a `symmetry_near` entry, the 9 spawn-scaled variants with a `symmetry_near_scaled` entry
  (`<key>__scaled`, judged on `renders/<key>__scaled.jpg`) and the 8 further alignment targets of the 15
  `symmetric_180` disagreement rows (the other 7 were already in the set); 454 candidate tags, 15 entries carry a
  `symmetric_180` question. Each entry: key, sheet, candidate tags with the survey's scores / classes and the sheet row
  to look at, uses, scale, bbox. `assign/index.json` lists the batches; `assign/review_tags.json` the one sheet that had
  to be rendered for the round (`shelf/wvkefp`, no geometric candidate, its `z2` row forced).
- Reviewers write `review/rev_XX.json`; `scripts/symmetry/aggregate_review.py` merges them into `review_verdicts.json`
  + `review_summary.md` (`preserved` → joins `symmetry`, `broken` → leaves `symmetry_near`, `unsure` → stays;
  `symmetric_180` value per alignment subtask), and `build_icl.py --symmetry` uses `review_verdicts.json` when it sits
  next to `symmetry.json`.

## Caveats for the review round

- Only frame-axis rotations are tested; a symmetry about a tilted axis (egg, scaled pencils) is reported as `none`.
- Low-poly meshes (181 models < 200 faces) are polygons of revolution: the finite fold is exact, `zinf` lands in
  `symmetry_near` with a preserved texture — a reviewer can promote it when the render rows look identical.
- Flat shading hides relief; a feature that is geometry only (key caps, embossing) is judged by the geometric score.
- OmniGlass parts are opaque in the geometry sample but absent from the texture renders (transparent in Isaac Sim);
  refraction effects are not modelled.
- Texture scores are computed on fully covered pixels of both renders; the union-mask MAD (`mad_union`) and the mask
  IoU are recorded for reference.
