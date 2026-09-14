# Look-alike review contract (disembedding same-category distractors)

Goal: for one target category, decide which candidate models **cannot be told apart from the robot's initial head-camera
view**. In the disembedding task the target and its distractors stand on a tray ~0.7-0.9 m in front of the robot; the
robot must recognise the target from that first image. If two models look the same there, one must never be the
distractor of the other. Several reviewers work in parallel; follow this contract exactly.

## Inputs

* Your assignment lists grid pages `grids/same/<category>__p<k>.png` (open EVERY page of the category before deciding;
  groups may span pages). One row per model: label (model id, selection / score of the grasp-library round, canonical
  bbox, exact-duplicate / geometry-cluster note, segmentation pixels at yaw 0, settled tilt, `stable` / `UNSTABLE`) and
  the crops of the object standing alone at the tray centre, seen by the head camera at the init stance, at yaws 0 / 90
  / 180 / 270 deg (native 1080 px head-camera resolution, all rows scaled to the same height; a small object that is
  upscaled looks blurry for everyone - that blur is what the robot sees).
* `exact dup:` in a label means the geometry check already proved the two assets identical (same mesh and texture).
  Keep those together; you may merge further.

## Definition

Two models are **look-alikes** when, shown one of them on the tray at an arbitrary yaw, you could not say which of the
two it is: same overall shape and proportions, same colours / material, and any print or label difference is either
invisible at this resolution or only on faces that a random yaw may hide. Judge from the crops, not from the model
ids or from what you know about the assets.

* Different colour scheme, clearly different graphics, different size (>~15 %), different shape / proportions, a
  missing / extra part (cap, handle, spout) -> NOT look-alikes.
* Same colour scheme and shape, differing only by small text, a small logo, a subtle shade, or a label that faces
  away at some yaws -> look-alikes.
* Look-alike is transitive within a group: if A~B and B~C, list {A, B, C} as one group.
* A model may be in at most one group; models that resemble nothing else are simply not listed.

Also flag models that are **unusable as clutter**: untextured flat-grey / black assets, broken geometry, a model that
does not read as its category at all, or a model that fell over / rolled (`UNSTABLE` in the label) - these are
reported separately and excluded from the pools.

## Output (one JSON per category) -> `groups/<category>.json`

```json
{
  "category": "box_of_cereal",
  "n_models": 14,
  "pages_opened": ["grids/same/box_of_cereal__p1.png", "grids/same/box_of_cereal__p2.png"],
  "groups": [
    {"members": ["hcwbzw", "tiykku"], "reason": "<= 25 words: what makes them indistinguishable at this view", "confidence": "high" | "medium" | "low"}
  ],
  "unusable": [
    {"model": "abcdef", "reason": "<= 20 words"}
  ],
  "notes": "<= 60 words: anything the pool builder should know (e.g. all models look alike because they are 20 px)"
}
```

Rules: `n_models` = rows you saw across all pages; every model id you list must appear in the grids; `pages_opened`
must list every page of the category; groups need >= 2 members; write JSON only, exactly these keys, nothing else.
Do not write or modify any other file. Finish your batch with a two-line summary: categories done, and any category
where you could not open a page.
