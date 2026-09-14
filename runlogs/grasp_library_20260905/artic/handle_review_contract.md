# Handle review contract (articulated objects, v1)

Goal: for every openable joint of every articulated object model, confirm WHERE a two-finger parallel gripper (R1Pro, 97 mm max
opening, 55 mm long pads) must grasp to open the door / drawer / lid, and HOW (pinch orientation), based on the rendered strip.
A geometric heuristic already proposes candidates; you validate or correct it. Several reviewers work in parallel: follow this
contract exactly.

## Inputs

* Your batch file lists model JSONs `/tmp/glib/artic/handles/<category>__<model>.json` and the matching strip
  `/tmp/glib/artic/strips/<category>__<model>.png`.
* Strip layout: `overview (closed)` = whole object, all markers; one `zoom` panel per openable joint showing ONLY that joint's
  markers (title = joint name, kind, heuristic status); `opened 60 %` = all openable joints moved 60 % towards open (markers stay
  where they were in the closed state, so you can see the opening motion relative to the markers).
* Markers (spheres): **A / B / C / D** (red / green / blue / orange) = candidate handle spots found as geometry protruding from
  the movable link's face; **F** (yellow) = fallback edge grasp (lid / leaf / drawer-front edge) proposed when nothing protrudes;
  **X** (grey) = protrusion too shallow to pinch (< 12 mm).
* Model JSON: per joint `kind` (drawer / door / flap), `status` (handle / edge / ungraspable), `handles` (A, B, ... in order,
  with `contact`, `pinch_axis`, `width` = handle thickness the fingers close on, `length`, `depth` = protrusion), `fallback`,
  `outward_normal`, `axis_dir_obj`, `hinge_point_obj`, `open_sign`, link bbox. Object frame: x, y, z in metres, z up, object
  upright, joints closed.

## What to decide per joint

1. **Which marker is the right place to grasp to open it?** Choose one letter, or `"none"`. The right place is the real handle /
   knob / pull bar of THAT movable part, or (for handle-less lids and flaps) an exposed edge the fingers can pinch across. Reject
   markers that sit on decorative mouldings, panel frames, the top lip of a drawer front spanning its full width, hinges, or on
   a neighbouring part. If two markers are both real handles (a long bar marked in the middle and at its ends), prefer the
   middle (`A` is usually the middle).
2. **Pinch orientation**: `"horizontal_bar"` (fingers close vertically across a horizontal bar), `"vertical_bar"` (fingers
   close horizontally), `"knob"` (either), `"edge"` (fingers close across the thickness of a lid / leaf / drawer front),
   `"lever"` (door lever handle: fingers close across the lever's thickness), or `"none"`.
3. **Opening motion**, from the `opened 60 %` panel: `"pull"` (the part comes towards the marker side / the viewer),
   `"swing"` (a door/lid rotates about a hinge; say which side the hinge is on if you can), `"lift"` (a top lid rises),
   `"slide"` (sliding door / window), `"push"` (moves away from the viewer), or `"unclear"`.
4. **Graspable by pinching?** true / false. False when the handle is flush / recessed (no protrusion to get the fingertips
   behind), when the only marker is on the wrong side (e.g. inside the cabinet), when the part is a glass pane with no frame
   edge, or when the movable part is too thick to pinch (> 85 mm) and has no handle.
5. **Where is the real handle if no marker is right?** Free-text description in object terms, e.g. "recessed pull along the
   top edge of the drawer front", "round knob at mid-height on the right edge, not marked", "no handle; flush panel".
6. **Height concern**: note if the grasp spot is above 1.5 m or below 0.25 m in the closed state (read z from the JSON contact +
   remember the object stands on z = 0 with its AABB; the strip header gives the AABB height), since the robot may not reach.
7. **Confidence**: high / medium / low.

## Output (one JSON per model) -> `/tmp/glib/artic/review/<category>__<model>.json`

```json
{
  "key": "bottom_cabinet__bamfsz", "category": "bottom_cabinet", "model": "bamfsz", "strip_opened": true,
  "joints": [
    {"joint": "j_link_1", "kind": "drawer", "chosen": "A" | "B" | "C" | "D" | "F" | "none",
     "pinch": "horizontal_bar" | "vertical_bar" | "knob" | "edge" | "lever" | "none",
     "motion": "pull" | "swing" | "lift" | "slide" | "push" | "unclear",
     "graspable": true, "handle_description": "<= 25 words", "height_concern": "" | "<= 15 words", "confidence": "high" | "medium" | "low"}
  ],
  "object_note": "<= 30 words: overall quality, wrong-side markers, anything the heuristic clearly got wrong"
}
```

Rules: one entry per joint listed in the model JSON (same joint names, same order); open every strip with the Read tool before
answering; JSON only, exactly these keys; do not write or modify any other file. Finish your batch with a two-line summary:
models done, and the joints you marked `graspable: false` or `chosen: "none"`.
