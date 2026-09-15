# Symmetry review — reviewer contract

You review the borderline results of the rotational-symmetry survey of the icl object models
(`runlogs/symmetry_20260915/README.md` describes the survey). Your batch is one file `assign/rev_XX.json`; you write one
file `review/rev_XX.json`. Do not edit anything else, do not re-render, do not change the sheets.

## The question

For every candidate rotation of every model in your batch: **would a person looking at the object through the robot's
head camera (0.5–1.5 m away, 1080 px image) see the object as identical after that rotation?** Identical means: nothing
in shape, colour or print tells the rotated object apart from the original — the policy could not tell the goal
orientation from the start orientation, and the evaluator must accept the rotated placement as correct.

## Verdicts

| verdict | meaning |
|---|---|
| `preserved` | a person at 0.5–1.5 m could not tell the rotated object from the original: this rotation becomes an `orientation_equivalence` of the model |
| `broken` | a person could tell them apart (a print, a seam, a logo, a handle, a lid, a different end, a visibly different silhouette ...) |
| `unsure` | the sheet does not let you decide (too small, ambiguous, rendering artefact); say why in the note |

Be decisive when you can; `unsure` keeps the rotation in the review queue, it is not a soft `preserved`.

## What you look at

The contact sheet of the model, `runlogs/symmetry_20260915/renders/<key>.jpg` (`sheet` in your assignment). Layout:

- header: `category/model`, bbox in metres (x × y × z), the survey's geometric verdict;
- **top row** = the original object from six fixed cameras: from **+x, +y, −x, −y** (25° above the horizon, 20° offset
  in azimuth) and from **+z** (above) and **−z** (below), tilted 25°;
- **every further row** = the object rotated by the candidate named at the left, seen from the **same six cameras**.
  Compare each tile with the tile above it in the top row. The label gives the tag, the rotation shown (`z163` = 163°
  about z — for `zinf` / `full` the worst of the tested angles is shown; a row that an earlier batch was issued with
  keeps its rotation, then `scores <rotation>` names the angle the scores belong to), the survey's texture class, the
  geometry scores (mean / p95 / frac2 = fraction of surface points displaced by > 2 % of the bbox diagonal) and the
  texture scores (MAD / p95 / p99 of the pixel difference in /255 over the pixels covered in both renders; `blob` = the
  largest connected patch of difference the survey found, its side in mm, the camera it was seen from and its pixel
  count over the pixel count of a 6 × 6 mm patch; the union-mask MAD and the silhouette IoU). Renders are flat-shaded
  (albedo only, no lighting): shading can never differ, only shape and print.

Your assignment lists per model the **candidate tags** to judge (`candidates`), with `sheet_row` = the label of the row
to look at, and the survey's reasoning (`survey_geometry`, `survey_texture`, `asked_because`). Judge only those tags;
rows of other tags are context.

## Tags = rotations of the canonical object frame

The canonical frame is the object's asset frame (the frame the sampler spawns it in; bbox x × y × z in the header;
for most standing objects z is up, `frame_front` / `frame_up` of the alignment targets name the front and up axes).

| tag | rotation |
|---|---|
| `x2`, `y2`, `z2` | half turn (180°) about the frame's x / y / z axis |
| `x4`, `y4`, `z4` | quarter turn (90°) |
| `x3`, `y3`, `z3` / `x5`, `y5`, `z5` / `x6`, `y6`, `z6` | 120° / 72° / 60° |
| `xinf`, `yinf`, `zinf` | **any** angle about that axis (body of revolution); the row shows one odd angle, e.g. `z163` |
| `full` | any rotation at all (a sphere); the row shows one random rotation |

A tag is `preserved` only if *every* rotation it stands for is invisible: `zinf` means any yaw, so a single label,
seam or dent anywhere around the object breaks it; `z2` only needs the two opposite sides to look alike.

## Rules of thumb

1. **Polygon count is not a difference.** A low-poly body of revolution (an 8-, 12-, 16-gon bowl, cup, bottle) whose
   rotated silhouette differs only by the position of the polygon corners is `preserved` for `zinf` — a person sees a
   round object either way. The survey put these into the review because the geometry score is slightly off
   ("geometry near", small frac2, texture MAD ≈ 0).
2. **Small features count if a person could read the orientation from them at 1 m**: a seam, a small print, a faint
   logo, a stem, a pull tab, a spout, a wick, non-repeating wood grain or a knot → `broken`. Below that (a texture noise
   pattern, a uniform grain, a speck that is not visible in the sheet) → `preserved`.
3. **Both sides alike is a real symmetry.** Many assets carry the same label twice (front = back), a lid identical to
   the base, a cover printed on both sides: then `z2` (or the flip) is genuinely `preserved` even though the object is
   "labelled". Check the tiles, not the category name.
4. **Repeating patterns.** A pattern that repeats n times around the object gives the n-fold tag (`z5` for a cup with
   five identical motifs) but breaks `zinf`.
5. **Glass.** OmniGlass parts are transparent in the simulator and were left out of the texture renders: a bottle
   with a glass body shows only its label and cap floating in the render; the glass body itself is included in the
   geometry score. Judge the visible parts; an all-glass model is drawn in a uniform tint — judge its shape only
   (a plain wineglass under `zinf` is `preserved`).
6. **Thin / elongated objects** (pencils, sticks, rods) are rendered at twice the resolution but still thin: an
   end-to-end flip (`y2` / `z2` of a pencil lying along x) that swaps a pointed tip with a flat end is `broken`
   if the ends differ in shape or colour in the tiles; a rotation about the long axis (`xinf`) of an unprinted hexagonal
   pencil is `preserved`.
7. **Silhouette mismatch** in a row (the rotated object visibly larger / smaller / shifted) means the geometry is not
   symmetric about that frame axis — `broken` unless it is only the polygon-corner effect of rule 1. The scaled
   variants (`<key>__scaled`, spawned at the listed per-axis `scale`) often show this: a non-uniform scale amplifies a
   slight tilt of the object's own axis against the frame axis, and the rotated rod visibly wobbles.
8. **`symmetric_180` questions** (models with a non-empty `symmetric_180` list): the tag named there is the half turn
   about the alignment axis (the wall normal = `front` axis for the wall subtasks, the `up` axis for axial alignment).
   Your verdict on that tag decides the icl column (`preserved` → `symmetric_180 = 1`, `broken` → `0`, `unsure` → the
   reviewer's value stays). In the tag's note say who was right (the selection reviewer's value or the survey) and
   about which axis you judged.
9. Ignore rendering artefacts: one-pixel differences along edges, JPEG noise, the white background. Ignore anything you
   cannot see in the sheet — if a decision would need a bigger picture, answer `unsure`.
10. Judge the object as it is in the asset, not the real-world object ("a soda can normally has one label" is not an
    argument; the tiles are).

## Output

Write **exactly one** JSON file, `runlogs/symmetry_20260915/review/<batch_id>.json` (`output` in your assignment; create
the `review/` directory if needed), with one entry per model of your batch and one entry per candidate tag:

```json
{
  "<category>__<model>": {
    "<tag>": {"verdict": "preserved" | "broken" | "unsure", "note": "short reason (one sentence)"},
    ...
  },
  "<category>__<model>__scaled": {
    "<tag>": {"verdict": "...", "note": "..."}
  },
  ...
}
```

- keys = the `key` fields of your assignment (scaled variants keep their `__scaled` suffix), tags = exactly the
  `candidates` listed for that key — every model and every candidate must appear, no extra tags, no extra keys;
- `verdict` is one of the three strings; `note` is a short free-text reason (mention the feature that breaks or the
  reason nothing does; for `symmetric_180` tags also "reviewer right" / "survey right" and the axis);
- do not include scores, do not copy the assignment; do not touch `symmetry.json`, `symmetry.csv` or the sheets.

`scripts/symmetry/aggregate_review.py` merges the `review/*.json` files: `preserved` tags join the model's `symmetry`
(they become `orientation_equivalences`), `broken` tags leave `symmetry` and `symmetry_near`, `unsure` tags stay in
`symmetry_near` for a second look. Your verdict stands even if a later re-run of the survey re-classified the tag
(such verdicts are listed separately in the aggregation summary).

Supplementary batches (`rev_11` and up) were issued after the survey's texture test was tightened (higher-resolution
renders, a local-feature test that catches small and low-contrast prints): they contain only the (model, tag)
candidates no earlier batch asked about; the sheets of the earlier batches kept their row order, only the labels and
the top-row renders were refreshed.

Brightness: the sheets issued to `rev_00`–`rev_10` (and the first `rev_11`–`rev_14` renders) showed every textured
model at 40 % of its albedo brightness (a renderer material default; a white cup looked mid-grey, dark trays black).
The sheets were re-rendered at true brightness on 2026-09-15 with the same row layout. Verdicts already given stand;
if a model you judged looked "featureless black", its dark parts are now visible — the coordinator may re-ask.
`review/overrides.json` holds coordinator decisions that win over reviewer verdicts (schema as above plus `"by"`).
