# Obstacle object review contract (v1, 2026-09-12)

You review rendered BEHAVIOR-1K object models as candidate **navigation obstacles** for the iSpatialGym problem
`camera_relocalization-0` (task id `obst`). In that task the robot (R1Pro, wheeled base ~0.66 m across the chassis
diagonal, head camera ~1.5 m up) starts at one base pose and has to drive to the base / trunk configuration whose head
camera pose matches a goal image; init and goal are up to 5 m apart (10 m geodesic) in one room. Some instances will
get one extra object dropped **on the floor, roughly midway along the robot's path**, so that the robot has to steer
around it. The object is never touched. A good obstacle is therefore something that (1) commonly stands on the floor in
the walkable part of at least one of the seven candidate scenes - `house_single_floor`, `house_double_floor_upper`,
`house_double_floor_lower`, `Rs_int` (a flat), `restaurant_diner`, `hotel_suite_large`, `office_cubicles_right` - and
looks unremarkable when met in the middle of a room or corridor (a bin, a potted plant, a cardboard box, a laundry
hamper, a bucket, a suitcase, a toy, a vacuum cleaner, a stool), (2) is big enough that the base cannot roll over it
and the head camera sees it, yet small enough to be placed on a free floor cell without blocking a ~1 m corridor, and
(3) rests stably on the floor in its canonical pose (the sampler drops the model as it is: canonical +z up).

Renders: `renders/grids/<category>_<k>.png` (up to 6 models per image, one strip per model). Each strip = four views of
the model in its **canonical frame**: `from +x`, `from -x`, `from -y` (25 deg above the horizon) and `top (+z, up=+y)`.
An axis gizmo is drawn at the bbox centre: **red = +x, green = +y, blue = +z** (arrow head = positive end). The header
gives the canonical bbox (x, y, z extents in metres) - use it for the size score. Glass parts are omitted by the
renderer; reflections and normal maps are not rendered, so a dark or flat-looking surface is not a defect by itself, but
missing textures (uniform grey, `textured 0`), missing faces, spikes, floating pieces and broken geometry are.

Per category you also get (in your assignment file) the hand annotation `natural_scenes` (which of the seven scenes the
category was proposed for) and `in_scene` (the scenes whose BEHAVIOR scene files already contain the category, with the
room types). Use them as hints; what you see decides (e.g. a "plant_pot" model that is a hanging planter, a "balloon"
without a weight, a "folding_chair" model that is folded and lying down).

## Scores (integers)

* **V - visual quality 0-5.** 5 = complete sharp textures, clean geometry, immediately recognisable as its category,
  representative shape; subtract per defect (untextured parts, missing faces, spikes, floating pieces, odd outlier).
  0-1 = unusable.
* **N - naturalness as a mid-path floor obstacle 0-3.** 3 = an object commonly found standing on the floor in the
  walkable space of at least one of the seven scenes and unremarkable when met in the middle of a room or corridor:
  waste bins, potted plants, cardboard boxes, laundry hampers / baskets, buckets, luggage and bags, toys, a vacuum
  cleaner, a stool or a pulled-out chair, a cart. 2 = plausible on the floor but normally against a wall or next to
  furniture (floor lamp, fan, heater, water cooler, coat rack, shoe rack, speaker, printer) or a small item credible as
  floor clutter (shoes, a pillow, a bag). 1 = possible but odd in the middle of the path in these scenes (a counter
  appliance, a display rack, a keg outside a bar, a tyre in a living room). 0 = does not belong on the floor of any of
  the seven scenes, or the model is not the floor-standing form of its category (a hanging planter, a wall item, a
  folded chair lying flat, a lid).
* **S - size fit 0-2.** Read the bbox in the header. 2 = largest footprint side (max of x, y) 0.20-0.55 m AND height (z)
  0.20-1.10 m. 1 = borderline: footprint 0.12-0.20 m or 0.55-0.80 m, or height 0.10-0.20 m or 1.10-1.70 m. 0 = footprint
  > 0.80 m or < 0.12 m, height < 0.10 m or > 1.70 m (blocks a corridor / is furniture, or is too small to matter).
* **rest_ok - true / false.** true = canonical +z is up and the model stands on a flat base, legs or wheels so that it
  stays where it is dropped. false = the model lies on its side, hangs, balances on a round bottom or an edge, would
  roll (balls: set false and say "rolls" in the note) or its canonical up axis is not +z.
* **score = 2V + 2N + S** (max 18). **valid = V >= 3 and N >= 2 and S >= 1 and rest_ok.**

## Procedure

1. Open every grid image of every category you were assigned (all `renders/grids/<category>_*.png`). Do not skip any.
2. Score every model of the category (every strip you saw).
3. Write ONE json per category to `review/<category>.json` (create the directory if needed):

```json
{
  "category": "trash_can", "task": "obst", "n_models": 6, "n_opened": 6,
  "models": [
    {"model": "abcdef", "V": 4, "N": 3, "S": 2, "rest_ok": true, "score": 16, "valid": true,
     "note": "<= 25 words: what it looks like, size, any defect"}
  ],
  "picks": ["abcdef", "ghijkl"], "alternate": "mnopqr",
  "natural_scenes": ["house_single_floor", "Rs_int", "office_cubicles_right"],
  "category_verdict": "good" | "weak" | "drop",
  "category_note": "<= 40 words: what the category looks like, typical size, where it is natural, caveats"
}
```

`picks` = the two highest-scoring valid models; if two candidates are within 2 points, prefer for pick 2 the one that
looks different from pick 1 (colour / shape / variant) so the pair spans the category. `alternate` = the next valid
model or null. `natural_scenes` = the subset of the seven scene ids in which the category is natural as a floor
obstacle (start from the hint, drop or add scenes as you see fit). `category_verdict`: `good` = >= 2 valid models and
the category is a natural floor obstacle (N = 3 for its typical model); `weak` = only one valid model, or N = 2 is the
best the category gets, or the size is borderline for most models; `drop` = no valid model or the category does not
fit (not floor-standing, wrong size, not natural in any of the seven scenes).

Rules: `score` must equal the formula, `valid` must follow the rule, every model of every grid must appear, JSON only
(exactly these keys). Do not modify any other file. Finish with a two-line summary: categories done, and which ones
you rated `drop` or `weak`.
