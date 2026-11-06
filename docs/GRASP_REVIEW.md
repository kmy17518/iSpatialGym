# Grasp-library v2 review instructions (agents and image reviewers)

Review one model entry at a time: `models/<entry id>/{candidates.json, validation.json, library.json}` and the rendered frames when present.
Write `review/<batch>.json` as `{"batch": ..., "models": [{"key": <entry id>, "grasps": [...], "families": [...], "object_note": ...}]}`.

## Per validated grasp (`library.json` `grasps[]`)

```json
{"name": "top_3", "verdict": "ok | reject", "flags": ["one_finger", "off_target", "rim_not_knob", ...], "note": "..."}
```

- Compare the INTENDED contact (the candidate's pad centre) with the ACHIEVED one (`validation.json` `trials[].achieved`:
  `T_object_eef_at_attach`, `off_target_m`). An assisted grasp that formed on the rim or the body while the candidate targeted a knob
  is `off_target` even when it held.
- Check the forces (`trials[].forces`): finger / table load, the object's load on the table relative to its weight.

## Symmetry annotations (REQUIRED for every grasp)

For each validated grasp, state its symmetry family explicitly - the reach maps and the sampler use it:

```json
{"name": "top_3", "symmetry": {"type": "rotational | translational | none",
                               "frame": "object", "axis": [x, y, z], "center": [x, y, z],
                               "interval": [lo, hi], "units": "rad | m",
                               "agrees_with_detected": true, "note": "rim of the cup, interrupted by the handle between 1.9 and 2.6 rad"}}
```

- **Rotational** (a cup rim, a round lid's edge, a bottle's body): the axis and a point on it in the object frame, and the angular
  interval (relative to this grasp) over which the same pinch exists - stop where a handle, spout or label edge interrupts it.
- **Translational** (a straight handle or bar, a rod): the axis and the interval (m, relative to this grasp) along which it slides.
- **none**: a grasp on a unique feature.
- Compare with the detected family (`candidates.json` `families[<family_id>]`) and say whether it agrees; correct the axis / centre /
  interval when it does not.
- ONE family is ONE candidate: only its representative was physics-tested (`physics: representative_only`). Do not describe the family
  as physically validated; describe it as identified geometrically / by review.

## Per model

`object_note`: the rest the object took (`rests.json`: is it the intended canonical / designated rest? a second rest?), anything about
the asset (collision pieces that do not match the visual, a hollow shell).
