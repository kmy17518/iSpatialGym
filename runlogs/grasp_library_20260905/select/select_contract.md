# Test-instance selection contract (v1)

Goal: for every (task, category) group of the grasp library, pick the TWO object models that will become the test instances of
that category for that task. A test instance must (a) have the best grasps for the task and (b) look good: a clean,
representative, well-textured asset. You are one of several reviewers working in parallel; follow this contract exactly so
that the results are comparable.

## Inputs

* Your assignment file lists group JSON files: `/tmp/glib/select/groups/<task>__<category>__r<round>.json`.
* Each group JSON has: `task`, `relation` (IORS only), `partner`, `round`, `category`, `requirements` (task-specific text),
  `candidates` (models with at least one AG-valid grasp) and `excluded` (models with no valid grasp: do NOT open their strips).
* Each candidate has: `model`, `key`, `strip` (PNG path), `n_valid`, `n_clean` (grasps not flagged by the earlier reviewer,
  `one_finger` ignored), `n_clean_strict`, `flags`, `grasp_notes` (earlier reviewer's per-row verdicts), `object_note`,
  `bbox_mm` (x, y, z of the canonical model), `scale` (per-axis scale of the tested variant, null = original size),
  `ready`, `pipeline`, `constraint`.

Strip anatomy: header = `category / model   task   pipeline (mode)   constraint   rest pose   [SCALED x[...]]   valid k/n`.
One row per validated grasp, best-ranked first. Columns: V1 profile along the pinch axis, V2 opposite profile, V3 oblique
from below, V4 right wrist camera. Left label: `fingers closed at a/b mm` (a, b = the two finger joint positions; 50 = fully
open, 0 = fully closed; a snug two-sided pinch reads roughly (open - width)/2 on both, e.g. 17/17 for a 60 mm object;
`50/x` means one finger never moved and the object is pressed against the open pad), `width` (object width at the nominal
pinch), `CoM offset`, `tilt`, `antipodal`, `coverage`, `GraspGen conf` (floating) or `stem t` (flowers).

## Procedure

1. Read the group JSON.
2. Open EVERY candidate strip with the Read tool (PNG). Do not skip any; do not open `excluded` strips.
3. Score every candidate with the rubric below. The earlier reviewer's flags/notes are hints only; what you see decides.
4. Rank by score, apply the pair rule, write the output JSON for the group, move to the next group.

## Rubric (integers)

**G - grasp quality, 0-5.** Start from the number of rows that are clean and natural: both pads on the object, contact
centred on the body or on the handle for handled objects, no visible gap between a pad and the object (collision hull larger
than the visual), no pad sunk into the object (hull smaller), no pinch on a fragile part (petals, rim of a glass, stem of a
pepper), no tip/end pinch far from the centre of mass on a handle-less object. Cap at 5. Then:
* floating tasks (`object_scaling`, `mental_rotation`): these tasks need several different object-in-hand transforms, so
  require DIVERSITY - at least 3 clean rows with visibly different contact regions or approach directions; if the clean rows
  are near-duplicates of one pose, cap G at 3.
* tabletop tasks (`disembedding`, `IORS`, `configuration_matching`): the BEST single grasp matters most; G = 5 requires the
  top-ranked row to be clean and natural plus at least 2 other clean rows; if the top row is not clean, G <= 3.
* `one_finger` closures (`50/x`) are an assisted-grasp artefact and are not a defect by themselves; judge them by whether the
  object is still held at a sensible place. If the object is pushed so far that it hangs off one pad edge, count the row as
  not clean.

**T - task fit, 0-3.** 3 = the selected grasps are exactly what the task needs; 0 = they conflict with it.
* `configuration_matching` (attachment): the mating side (the part that attaches: cap bottom, lid rim, bulb socket, cork
  body, the nail side of decorations) must be fully uncovered by the pads.
* `IORS inside` and flowers: pinch from above / on the upper stem so the lower body can go down into the container; a
  flower must be held by the stem or the receptacle behind the head with the head free.
* `IORS ontop / under / nextto` and `disembedding`: the object bottom stays free so it can be placed; the object is visibly
  stable in its resting pose (V1/V2 show it sitting, not toppling).
* floating tasks: any stable body grasp is fine (3); handle grasps on tools count as fine.

**V - visual quality, 0-5.** 5 = sharp, complete textures (no black / flat-grey untextured faces, no missing materials),
clean geometry (no spikes, holes, interpenetrating parts, floating pieces), immediately recognisable as the category, and a
representative shape and size for the category (not an odd outlier). Subtract for each defect. 0-1 = unusable as a test
instance. Use V1-V4 together; V4 (wrist camera) shows the texture closest.

**R - robustness, 0-2.** 2 = pinch width comfortably within 15-80 mm and the object is >= 15 mm thick at the pinch (read
`width` and `fingers closed`); 1 = borderline (width 10-15 mm or 80-89 mm, or fingers closing to <= 5 mm); 0 = marginal.

**score = 2*G + 2*V + T + R** (maximum 25).

## Hard exclusions

* `n_clean` = 0 in the input AND you also see no clean row -> not selectable (score it, but do not select it unless the
  group would otherwise have fewer than 2 picks; then select with a warning).
* V <= 1 -> not selectable (broken or untextured asset), same fallback rule.
* SCALED variants are selectable (they are the dataset variant for that category); mention the scale in the reason.

## Pair rule

* Pick #1 = the highest score.
* Pick #2 = the highest score among the rest, except: if two remaining candidates are within 2 points of each other, prefer the
  one that looks DIFFERENT from #1 (another colour / shape / variant) so the pair spans the category's appearance. Never pick
  a candidate with G <= 1 as #2 while one with G >= 3 remains.
* Groups with <= 2 candidates: score them, select all that are selectable, set `"short": true`, and say in `warnings` what is
  weak.
* Also name the best `alternate` (third choice) when one exists.

## Output (one JSON per group) -> `/tmp/glib/select/out/<group_id>.json`

```json
{
  "group": "<task>__<category>__r<round>", "task": "...", "category": "...", "round": 1, "relation": "ontop" | null,
  "n_candidates": 7, "n_opened": 7,
  "candidates": [
    {"model": "abcdef", "key": "category__abcdef", "G": 5, "T": 3, "V": 4, "R": 2, "score": 23,
     "note": "<= 25 words: what the grasps and the asset look like, any defect"}
  ],
  "selected": [
    {"model": "...", "key": "...", "rank": 1, "reason": "<= 30 words"},
    {"model": "...", "key": "...", "rank": 2, "reason": "<= 30 words"}
  ],
  "alternate": {"model": "...", "key": "..."} | null,
  "short": false, "warnings": [], "confidence": "high" | "medium" | "low"
}
```

Rules: `candidates` must list every candidate of the input (all with `n_valid >= 1`), `n_opened` must equal
`n_candidates`, scores are integers, `score` must equal `2*G + 2*V + T + R`. Write JSON only, with exactly these keys. Do
not write or modify any other file. If a strip cannot be opened, put the model in `warnings` and score it from the metadata
with `confidence: "low"`. Finish your batch with a two-line summary: groups done, and any group where fewer than 2 models
were selectable.
