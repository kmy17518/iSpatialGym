# World-alignment object review contract (v1, 2026-09-11)

You review rendered BEHAVIOR-1K object models for the two iSpatialGym **World Alignment** problems. The robot never
grasps these objects; it pushes them so that their orientation matches a goal image. Both problems evaluate the
orientation only, so a model is only useful if a human (and a policy) can read ONE correct orientation off the object.

Renders: `renders/grids/<category>_<k>.png` (3 models per image, one strip per model). Each strip = four views of the
model in its **canonical frame** (the frame the sampler places it in): `from +x`, `from -x`, `from -y` (25 deg above the
horizon) and `top (+z, up=+y)` (looking down, image up = +y, image right = +x). An axis gizmo is drawn at the bbox
centre: **red = +x, green = +y, blue = +z** (arrow head = positive end). The header gives the canonical bbox
(x, y, z extents in metres). Glass parts are omitted by the renderer (a clock face under glass therefore shows without
its glass); reflections and normal maps are not rendered, so a dark or flat-looking surface is not a defect by itself,
but missing textures (uniform grey `textured 0`), zero-thickness sheets (`z 0.000`) and broken geometry are.

## A. `wa0` = world_alignment-0 (table surface)

The object rests on a table in front of the robot; a vertical hinge through its centre of mass lets it yaw only. It
starts rotated 20-70 deg away from the aligned pose and must be pushed back so that its long axis is parallel to the
table edge and its "front" faces the robot. The category must have a **single** aligned yaw that is obvious from above
and from the robot's oblique view (keyboard keys + spacebar, cover text of a book, the handle of a pan, the clip of a
clipboard, the display of a calculator, the toe of a shoe...).

Score every model (integers):

* **V - visual quality 0-5.** 5 = complete sharp textures, clean geometry, immediately recognisable, representative of
  the category; subtract per defect (untextured parts, missing faces, spikes, floating pieces, odd outlier shape).
  0-1 = unusable.
* **O - orientation readability 0-3.** 3 = from the top view the aligned yaw is unique AND obvious (asymmetric print,
  text, key layout, handle, clearly asymmetric silhouette). 2 = unique, but the cue is smaller (a logo, a spine, a
  hinge, a cable, a clasp). 1 = a 180-degree ambiguity remains (plain box without readable print, symmetric tray, a
  plain rectangle) or the long axis itself is hard to read. 0 = round / square / plain: no readable yaw.
* **F - table fit 0-2.** 2 = long axis 0.12-0.50 m, height <= 0.10 m, rests flat and stable on its natural base in the
  canonical pose (+z up), natural on a table / desk / counter. 1 = borderline: long axis 0.08-0.12 or 0.50-0.60 m,
  height 0.10-0.20 m, slightly unusual on a table (a shoe, a tool), or a curved but stable bottom. 0 = rolls
  (cylinder / rod), < 0.08 m, > 0.60 m, taller than it is long, or unnatural on a table (floor / wall / garden item).
* **score = 2V + 2O + F** (max 18). **valid = V >= 3 and O >= 2 and F >= 1.**

Also read the canonical frame: `long_axis` = `x` or `y` (the canonical axis along the longest horizontal extent),
`front_axis` = `+x`, `-x`, `+y` or `-y` (the side that must face the robot when aligned: the spacebar side of a
keyboard, the bottom edge of readable cover text, the handle end of a pan, the clip side of a clipboard; for a book
the bottom of the cover), and `rest_ok` = true when the canonical pose (+z up) is the object's natural resting pose on
a table (false if it stands on an edge / is upside down / would fall over).

## B. `wa1` = world_alignment-1 (world vertical)

The object hangs on a wall through a revolute joint at its centre of mass (axis = wall normal), starts rotated 15-45
deg in the wall plane and must be pushed back to the straight (upright) orientation. The whole object has to fit the
head camera image from 0.6-0.95 m and the robot pushes one of its lower corners. The category must have ONE readable
upright orientation (picture content, clock numerals, text, a hook at the top, an asymmetric silhouette) and be
natural on a wall of a house / hotel suite / office / diner.

* **V - visual quality 0-5** as above.
* **U - up readability 0-3.** 3 = the upright orientation is obvious at a glance from the front (a scene, a face,
  numerals, text). 2 = readable but subtler (a horizon, a hook, a marker tray, an asymmetric ornament). 1 = a
  180-degree (or 90-degree for squares) ambiguity remains (abstract symmetric pattern, plain mirror, plain frame).
  0 = round / symmetric, no readable up.
* **H - hangability 0-2.** 2 = flat back, thickness <= 0.08 m, 0.25-0.90 m across, a normal wall item. 1 = borderline:
  0.15-0.25 m or 0.90-1.30 m across, thickness 0.08-0.15 m, or unusual on a wall (a tool, a hanger, a tray).
  0 = not hangable, < 0.15 m, > 1.30 m, not flat, or degenerate geometry.
* **score = 2V + 2U + H** (max 18). **valid = V >= 3 and U >= 2 and H >= 1.**

Read the canonical frame: `front_axis` = the axis pointing out of the wall toward the room, i.e. the face carrying the
content (`+x` for the paintings: the `from +x` view shows the picture, `from -x` the back); `up_axis` = the canonical
axis that is the object's up when it hangs (`+z` when the model already stands upright like the paintings; for a
model lying flat, thin along z, read the content in the top view: image up = `+y`, so a picture whose content is
upright in the top view has `up_axis = +y`, one whose content is upside down has `-y`, sideways `+x` / `-x`).
`thin_axis` = the bbox axis with the smallest extent.

## Procedure

1. Open every grid image of every category you were assigned (all `renders/grids/<category>_*.png`). Do not skip any.
2. Score every model of the category (every strip you saw).
3. Write ONE json per category to `review/<category>.json` (create the directory if needed):

```json
{
  "category": "keyboard", "task": "wa0", "n_models": 6, "n_opened": 6,
  "models": [
    {"model": "hgtbot", "V": 4, "O": 3, "F": 2, "score": 16, "valid": true,
     "long_axis": "y", "front_axis": "+x", "rest_ok": true, "note": "<= 25 words"}
  ],
  "picks": ["hgtbot", "injrgc"], "alternate": "jtdyix",
  "category_verdict": "good" | "weak" | "drop",
  "category_note": "<= 40 words: what the category looks like, typical size, caveats"
}
```

For `wa1` replace `O`/`F`/`long_axis`/`rest_ok` by `U`/`H`/`up_axis`/`thin_axis`. `picks` = the two highest-scoring
valid models; if two candidates are within 2 points, prefer for pick 2 the one that looks different from pick 1
(colour / content / shape) so the pair spans the category. `alternate` = the next valid model or null.
`category_verdict`: `good` = >= 2 valid models and the category clearly has one readable orientation; `weak` = only
one valid model, or the orientation cue is marginal for most models; `drop` = no valid model or the category does not
fit the task (rolls, symmetric, wrong size, not natural on a table / wall). A category listed for both tasks gets two
files: `review/<category>.json` for its `wa0` scores and `review/<category>__wa1.json` for `wa1`.

Rules: `score` must equal the formula, every model of every grid must appear, JSON only (exactly these keys). Do not
modify any other file. Finish with a two-line summary: categories done, and which ones you rated `drop` or `weak`.
