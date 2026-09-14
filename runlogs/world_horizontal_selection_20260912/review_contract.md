# World-horizontal object review contract (v1, 2026-09-12)

You review rendered BEHAVIOR-1K object models for the iSpatialGym problem `wa2` = **world_alignment-2 (world
horizontal)**. It is the sibling of world vertical (`world_alignment-1`, runlogs/world_alignment_selection_20260911):
the object hangs **flat against a wall** through a revolute joint at its centre of mass (axis = wall normal), starts
rotated 15-45 deg in the wall plane and must be pushed by one of its lower ends back to the **level** orientation. The
robot never grasps the object. Only the orientation is evaluated, so a model is useful only if a human (and a policy)
can read ONE horizontal direction off the object: the axis or edge that has to be level. Unlike world vertical, a
180-degree ambiguity does NOT hurt (a plain bar is level either way); a 90-degree ambiguity (square) does.

Renders: `renders/grids/<category>_<k>.png` (3 models per image, one strip per model). Each strip = four views of the
model in its **canonical frame** (the frame the sampler places it in): `from +x`, `from -x`, `from -y` (25 deg above the
horizon) and `top (+z, up=+y)` (looking down, image up = +y, image right = +x). An axis gizmo is drawn at the bbox
centre: **red = +x, green = +y, blue = +z** (arrow head = positive end). The header gives the canonical bbox
(x, y, z extents in metres). Glass parts are omitted by the renderer; reflections and normal maps are not rendered, so
a dark or flat-looking surface is not a defect by itself, but missing textures (uniform grey `textured 0`),
zero-thickness sheets (`z 0.000`) and broken geometry are.

How the object will hang: its **thin axis along the wall normal** (flat against the wall, like a sign), its
**level axis along the wall** (horizontal when aligned), centre 1.45 m above the floor, robot 0.6-0.95 m in front
with the head camera on the object. A shelf plank or a bat therefore hangs like a board / a rod on display hooks, not
in its functional pose; judge hangability for that presentation.

## Scores (integers)

* **V - visual quality 0-5.** 5 = complete sharp textures, clean geometry, immediately recognisable, representative of
  the category; subtract per defect (untextured parts, missing faces, spikes, floating pieces, odd outlier shape).
  0-1 = unusable.
* **L - level readability 0-3.** 3 = the object has ONE dominant axis or edge that must obviously be horizontal and a
  15 deg tilt would be seen at a glance from the front: a bar, rail, rod, shelf, a board / screen / sign with aspect
  >= 2:1, a line of text. 2 = the level axis is unique and readable but subtler: aspect 1.5-2:1, or the horizontal
  comes from content (text baseline, a row of hooks, a horizon or a frame edge of a landscape picture) rather than the
  silhouette. 1 = ambiguous which edge should be level (aspect < 1.5 without a strong content cue, a portrait object
  whose only horizontal cue is small) or the object's natural hanging is vertical so a level long axis looks wrong.
  0 = round / symmetric under 90 deg / no readable horizontal.
* **H - hangability 0-2.** 2 = flat back, thickness <= 0.08 m, 0.25-0.90 m along the level axis, a normal wall item
  or an object commonly displayed on wall hooks / racks (bat, rolling pin, guitar, tools on a workshop wall). 1 =
  borderline: 0.15-0.25 m or 0.90-1.30 m along the level axis, thickness 0.08-0.15 m, unusual on a wall (a shelf
  plank hung flat, a hanger, a kitchen utensil), or a round rod whose front face is arbitrary. 0 = not hangable:
  < 0.15 m, > 1.30 m, thickness > 0.15 m, boxy / deep (a chimney hood, a deep AC unit), degenerate geometry.
* **score = 2V + 2L + H** (max 18). **valid = V >= 3 and L >= 2 and H >= 1.**

## Canonical frame (read from the gizmo)

* `front_axis` = `+x`, `-x`, `+y`, `-y`, `+z` or `-z`: the axis that points out of the wall toward the room, i.e. the
  face the robot sees (the printed / detailed face; for a flat object it is one of the two directions of the thin
  axis; for a round rod pick the direction that shows the most detail, e.g. the logo).
* `level_axis` = `x`, `y` or `z` (unsigned): the canonical axis that lies horizontal along the wall when the object
  hangs level - the long axis of a bar / rod / shelf, the long edge of a landscape board, the short edge of a
  portrait poster, the bar of a hanger, the baseline of a text sign.
* `up_axis` = signed axis pointing to world up when the object hangs correctly, when it has a readable top / bottom
  (hook of a hanger up, brackets of a shelf down, marker tray at the bottom, text upright, handle of a poker down).
  When the object looks the same upside down (plain bar, blank screen, rolling pin) still give one - prefer the one
  that keeps the model's canonical +z up - and set `symmetric_180` to `true`.
* `symmetric_180` = `true` / `false`: whether a 180-degree turn about the front axis leaves the object looking the same.
* `thin_axis` = the bbox axis with the smallest extent.
* `front_axis`, `level_axis` and `up_axis` must name three different axes.

## Procedure

1. Open every grid image of every category you were assigned (all `renders/grids/<category>_*.png`). Do not skip any.
2. Score every model of the category (every strip you saw).
3. Write ONE json per category to `review/<category>.json` (create the directory if needed):

```json
{
  "category": "towel_rack", "task": "wa2", "n_models": 6, "n_opened": 6,
  "models": [
    {"model": "kqrmrh", "V": 4, "L": 3, "H": 2, "score": 16, "valid": true,
     "front_axis": "+x", "level_axis": "y", "up_axis": "+z", "symmetric_180": false, "thin_axis": "x",
     "note": "<= 25 words"}
  ],
  "picks": ["kqrmrh", "crivoa"], "alternate": "pqhsry",
  "category_verdict": "good" | "weak" | "drop",
  "category_note": "<= 40 words: what the category looks like, typical size, caveats"
}
```

`picks` = the two highest-scoring valid models; if two candidates are within 2 points, prefer for pick 2 the one that
looks different from pick 1 (colour / content / shape) so the pair spans the category. `alternate` = the next valid
model or null. `category_verdict`: `good` = >= 2 valid models and the category clearly has one readable horizontal;
`weak` = only one valid model, or the level cue is marginal for most models; `drop` = no valid model or the category
does not fit the task (square, round, wrong size, not natural on a wall).

Rules: `score` must equal the formula, every model of every grid must appear, JSON only (exactly these keys). Do not
modify any other file. Finish with a two-line summary: categories done, and which ones you rated `drop` or `weak`.
