# Grasp proposal contract, ladder step 3 (`isg-grasp-proposal/1`)

Goal: an object in the v2 grasp library still has **0 physics-valid grasps** after rounds 1 and 2. You propose where a two-finger
parallel gripper (R1Pro: 97 mm max opening, usable pinch width 3-89 mm, pads ~20 mm wide and 50 mm long, the fingertips 18.8 mm past
the eef origin) should pinch the object **as it rests on its support**, and from where it comes in. Your proposals then go through
the SAME geometric gates and the SAME physics validation as every generated candidate - nothing is relaxed for you, so propose pinches a
real gripper could do, not ones that only look plausible. If there is no pinchable spot from this rest, say so (`feasible: false`).
Several proposers may work in parallel: follow this contract exactly. (This is the grasp-library version of the articulation
library's handle review, `runlogs/grasp_library_20260905/artic/handle_review2_contract.md`.)

## Inputs (per entry, in the batch directory)

* `inputs/<id>.json`
  * `geometry`: the object's collision pieces in the **rest frame** (z up, the support plane at `support_z_mm`, millimetres): extents,
    bbox, CoM, mass, per piece bbox / centroid / volume, the vertical axis of round objects (`plate_axis_xy_mm`), principal axes,
    attachment mates (point + mating direction) and meta-link names.
  * `task`: what the grasp is for (constraint, its rule, the aliases / subtasks, the support height).
  * `gripper`: the eef frame and the numbers above (`pad_span_along_approach_mm`, `pad_centre_mm`, `max_contact_toward_tips_mm`).
  * `history`: rounds 1-2 - per pool the gate survivors (`n_pool / n_tilt -> n_width_antipodal -> n_support -> n_object_clear ->
    n_sweep -> n_lever -> n_selected`), the geometry reasons of the selected candidates and the physics reasons.
  * `markers`: the heuristic's proposals = the rounds' **near misses** (A, B, ...): `contact_mm`, `approach`, `closing_axis`,
    `width_mm`, `contact_offset_mm`, the gate that rejected it and the measured values it rejected it on.
* `strips/<id>.png`: top / front / side views and two vertical sections through the object's axis (mm grid, brown = the support
  line), an oblique view, the markers lettered (dot = pinch centre, segment = closing axis over the pinch width, arrow = approach).

## The gates your proposal must pass (all in `ispatialgym/grasping/candidates.py`, then physics in `validate_tabletop.py`)

1. width 3-89 mm along the closing axis through the contact, and the surfaces there within the antipodal cone of the closing axis;
2. support: only the fingers may come within 2 mm of the support; palm, wrist camera and wrist stay clear. A near-vertical pinch
   retreats along its approach (the contact may move up to 28.8 mm toward the fingertips, then it is rejected); a side pinch (approach
   >= 70 deg from vertical) is lifted along the support normal (at most 30 mm) and must still pinch the object there;
3. the hand's collision spheres 1 mm clear of the object, from the pre-grasp (100 mm back along the approach) to the grasp;
4. the closing sweep: the pads must meet the object first at the pinch, not further out;
5. lever: the pinch line within 0.4 x the object size of the CoM;
6. the constraint (attachment: contact on the far side of the mating face, no finger past it);
7. the finger blades (exact hulls) outside the object's other pieces down to the first touch;
8. physics: the oracle's pre-shape (width / 2 + 10 mm per finger) and closure, assisted grasp formed, held when the support drops 10 cm,
   no hand part on the support, no pressing the object into the support (load <= 2 x weight).

## What to propose (1-8 proposals, best first)

* `contact_mm`: [x, y, z] the pinch centre - midway between where the two pads touch the object - in the rest frame (mm).
* `closing_axis`: [x, y, z] the direction the fingers close along (any length; the sign does not matter).
* `approach`: [x, y, z] the direction the hand moves onto the object (eef +z; [0, 0, -1] = straight down). It is made orthogonal to
  the closing axis; the two must be more than 60 deg apart.
* `pad_offset_mm`: 1-5 values in [0, 28.8]: how far toward the fingertips from the pad centre the contact sits (0 = the pad
  centre; 20-28 = pinch with the fingertips, e.g. a thin rim close to the support). Side pinches (approach >= 70 deg from vertical) use
  only their 0 value.
* `expand`: null (default: tilts 0 / +-10 deg about the closing axis, turns 0 / +-15 deg about the approach, slides 0 / +-3 mm along
  the pads' width) or `{"tilt_deg": [...], "yaw_deg": [...], "slide_mm": [...]}` (each within +-45; at most 400 poses with the offsets).
* `from_marker`: the marker letter you started from, or null.
* `rationale`: <= 30 words: what you pinch and why it passes the gates that rejected the rounds' candidates.
* `confidence`: high / medium / low.

## Output (one JSON per entry) -> `proposals/<id>.json`

```json
{"contract": "isg-grasp-proposal/1", "id": "<entry id>", "feasible": true,
 "proposals": [{"from_marker": "A", "contact_mm": [21.8, 0.2, 9.0], "closing_axis": [1, 0, 0], "approach": [0, 0, -1],
                "pad_offset_mm": [20, 24, 28], "expand": null, "rationale": "<= 30 words", "confidence": "medium"}],
 "object_note": "<= 40 words: the object, why the rounds failed, what you changed"}
```

Rules: open the strip (PNG) before answering; JSON only, exactly these keys; `proposals` may be empty only when `feasible` is false;
coordinates in the rest frame in mm; do not write or modify any other file. Finish with a two-line summary: entries done, and the ones
you marked infeasible.
