You are reviewing robot grasp candidates for a manipulation benchmark. You will be given a list of PNG strip images.
Each strip shows ONE object model and up to 5 grasps that passed a physics "assisted grasp" test with an R1Pro
parallel-jaw gripper (two fingers, opening 0..97 mm; assisted grasping means a virtual joint attaches the object once both
fingers touch it, so physically weak grasps can still pass). Your job is to flag grasps that a careful roboticist would
NOT accept as natural, physically plausible grasps.

STRIP LAYOUT
- Header: category / model, task, pipeline, constraint, rest pose, "valid k/n tested".
- One row per grasp (top to bottom = rank 1..5). Left label per row: tilt (approach angle from vertical, only meaningful for
  tabletop objects), CoM offset (distance from the object's centre of mass to the pinch axis, mm), width (object width
  between the pads, mm), "fingers closed at a/b mm" (finger joint positions after closing: 50 = fully open, 0 = fully
  closed; two similar small values = symmetric pinch on the object; one value stuck at ~50 while the other is small
  means the object was pushed against one finger and the pads are not both at the nominal contact), antipodal / coverage
  scores, GraspGen confidence (in-hand tasks), stem t (flowers: 0 = flower head end, 1 = stem end), attach dist
  (attachment parts: distance from the fingers to the mating point, mm).
- Columns: V1 profile view (pinch axis left-right; best view for gaps / penetration between pads and object),
  V2 opposite profile, V3 oblique view from below (best view for WHERE on the object the grasp is), V4 right wrist
  camera (looks along the approach direction from the gripper; for large objects it is saturated and uninformative,
  rely on V1-V3 then).

WHAT TO FLAG (use these exact flag names; several may apply)
- end_grasp: pinch at an extreme end/tip/corner of an elongated or large object so that, without assisted grasping, the
  object would tip out or slip (long lever arm, tiny contact on the very edge). A pinch on a handle of a tool
  (fork, spatula, hammer, ladle) is NOT an end grasp - that is the natural grasp.
- gap: visible daylight between a finger pad and the object at the contact, i.e. the fingers stopped on an invisible,
  coarser collision hull and the visual mesh is not actually touched (object appears to float between the pads).
- penetration: a finger visibly sinks into the visual mesh (collision hull smaller than the visual object).
- one_finger: object clearly pressed against one finger only / strongly off-centre between the pads (typically
  fingers closed at ~50 / small).
- task_violation: the grasp blocks what the task needs: for attachment parts (lid, cap, cork, candle, light bulb,
  webcam, wreath) the mating side must stay free (a lid/cap must be held from the top/side, a cork/candle/bulb by the
  end that does NOT go into the partner); for flowers going into a vase the grasp must be high on the stem or on the
  head so the stem end is free and the flower can be inserted upright; for objects that will be placed (ontop /
  under / inside / disembedding) the fingers must not extend below the object's bottom face.
- fragile: pinching a thin protruding part that would break/deform (flower petals crushed, a single leaf, a thin rim of
  a glass at its very lip) when a better body grasp is obviously available.
- unsure: image does not allow a judgement (occlusion, object too small to see).

Do NOT flag: asymmetric-looking but firmly closed pinches; grasps that are merely non-optimal; mild tilt; grasps on
handles; small objects entirely hidden between the pads when V1/V2 show both pads touching it.

OUTPUT
Write one JSON file with the path given below, containing:
{ "batch": "<batch id>", "models": [ { "key": "<category>__<model> (from the filename without .png)",
    "n_rows": <number of grasp rows shown>, "grasps": [ {"rank": 1, "verdict": "ok"|"flag"|"unsure", "flags": [...], "note": "<=12 words"}, ... ],
    "object_note": "<optional, <=20 words, e.g. collision hull much bigger than visual mesh>" }, ... ] }
Review EVERY image in your list (open each one; do not skip). Be consistent: a row with no problems is "ok" with flags [].
Do not write anything else to disk. When finished, reply with a two-line summary (number of models, number of flagged grasps).
