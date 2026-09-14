# Related-category distractor check (disembedding other-category pools)

Goal: for one target category, look at the related-category models selected as its other-category distractors and
flag the ones that should NOT be used. A distractor is an object that sits on the tray next to the target; it must be a
plausible everyday object of a related kind (forks / knives next to a spoon, rice and sugar boxes next to a cereal
box), intact and recognisable. The semantic relation was chosen by category; you judge the individual assets.

## Inputs

`grids/pool_other/<category>__p<k>.png`: 5 x 3 tiles per page, each tile = one model standing alone at the centre of the
source tray, seen by the robot's head camera at the disembedding init stance (yaw 0). Label: `<category>/<model>`,
`tier` (1 = closest related category, 2 = broader, 3 = last resort), `pool` (in the 20) or `reserve`, the grasp-round
status, and the render check (segmentation pixels, settled tilt). Open EVERY page of the category.

## Flag a model when

* it is broken or untextured (flat grey / black / missing material, spikes, holes, floating parts),
* it does not read as its category or as any everyday object at this resolution,
* it is a poor distractor for the target category (a completely unrelated kind of object; e.g. a hammer next to
  cereal boxes) - be lenient: related-but-different is exactly what is wanted, only flag clear mismatches,
* it lies toppled / rolled / deformed on the tray.

Do not flag models for being small, plain, or for having a low pixel count alone.

## Output (one JSON per category) -> `groups_other/<category>.json`

```json
{
  "category": "teaspoon",
  "pages_opened": ["grids/pool_other/teaspoon__p1.png", "grids/pool_other/teaspoon__p2.png"],
  "n_models": 28,
  "exclude": [
    {"model": "<category>/<model>", "reason": "<= 20 words"}
  ],
  "notes": "<= 60 words"
}
```

JSON only, exactly these keys. Do not write or modify any other file. Finish with a two-line summary: categories done,
pages you could not open.
