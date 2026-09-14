# Handle review, round 2: propose your own grasp spot (v1)

These are the openable parts that the first review marked "no handle / not graspable" or where no marker was right. The
geometric heuristic found nothing usable, so YOU propose where a two-finger parallel gripper (R1Pro: 97 mm max opening, pads
~55 mm long, fingers ~15 mm thick) should pinch to open the part, and how. If there really is no pinchable spot, say so.

## Inputs (per joint)
* Strip `/tmp/glib/artic/strips2/<key>__<joint>.png`, three panels:
  1. **outer face** of the movable part, seen from outside, with four sphere markers on the corners of the part's outer face:
     **TL red, TR green, BL blue, BR yellow** and **C grey** at the face centre. "Left/right/top/bottom" below always refer to
     THIS panel (for top lids the outer face is the lid top seen from the front-above: "top" = far edge, "bottom" = near edge).
  2. a view from the side / above, 3. the part opened 60 %.
* Header: part size w x h x d (metres: width left-right, height bottom-top, depth = thickness along the outer normal) and
  the first reviewer's description.
* The joint list JSON `/tmp/glib/artic/nohandle_joints.json` (kind: drawer / door / flap; type: Prismatic / Revolute).

## What to propose
Choose the spot a human would grab to open the part and express it in the outer-face frame:
* `a` = fraction along the bottom edge from BL (0) to BR (1); `b` = fraction from the bottom edge (0) to the top edge (1).
  Example: the middle of the top edge = a 0.5, b 1.0; a knob at the right, mid-height = a 0.9, b 0.5.
* `face`: `"outer"` (the marked face), `"top_edge"`, `"bottom_edge"`, `"left_edge"`, `"right_edge"` (pinch ACROSS the part's
  thickness at that edge - only if the edge is exposed, i.e. nothing flush next to it), or `"none"`.
* `pinch`: `"horizontal"` (fingers close left-right), `"vertical"` (fingers close up-down), `"thickness"` (fingers straddle the
  panel/lid thickness - use with an edge face).
* `approach`: `"front"` (gripper moves into the outer face), `"front_down45"`, `"above"` (straight down), `"below"`,
  `"left"`, `"right"` (gripper comes from that side, parallel to the face).
* `depth_mm`: how far behind the outer face plane the pad centre should sit (0 = on the face; for an edge pinch 30-40 mm
  inside the part; negative if the spot protrudes in front of the face).
* `expected_motion`: `"pull"`, `"lift"`, `"swing"`, `"slide"`, `"push"`.
* `feasible`: true if you believe the pinch can physically hold the part (a >= 15 mm thick lip / knob / bar, or a panel edge
  10-85 mm thick that is exposed on both sides); false for flush panels, 4 mm flaps, recessed pulls with no lip.
* `note`: <= 25 words on what you are grasping and why it should work (or why not).
* `confidence`: high / medium / low.

## Output (one JSON per joint) -> `/tmp/glib/artic/review2/<key>__<joint>.json`
```json
{"key": "carton__hhlmbi", "joint": "j_link0", "feasible": true,
 "proposal": {"face": "bottom_edge", "a": 0.5, "b": 0.0, "pinch": "thickness", "approach": "front", "depth_mm": 30,
              "expected_motion": "lift"},
 "alternative": {"face": "outer", "a": 0.5, "b": 0.5, "pinch": "horizontal", "approach": "front", "depth_mm": 0, "expected_motion": "pull"} | null,
 "note": "<= 25 words", "confidence": "medium"}
```
Rules: open every strip with the Read tool; exactly these keys; `proposal` may be null only when `feasible` is false; do not
write or modify other files. Finish with a two-line summary: joints done, and how many you marked infeasible.
