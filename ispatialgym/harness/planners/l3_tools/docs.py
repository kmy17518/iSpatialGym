"""Model-facing operating notes for R1Pro on the Cartesian surface (the counterpart of RoboProbe's ARX X5 notes).

What the tool descriptions, the system message or the state block already say is left out rather than repeated in a
second wording: the tools own the action mechanics and the bounds, the system message owns what the model receives, and
these notes own what those cannot express (frame, geometry, grasp semantics, gripper polarity). Every number is one of
:mod:`core.pose`'s constants, formatted here, so the prose cannot drift from the surface."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from ispatialgym.harness.core.pose import FINGERTIPS_BEYOND_GRASP_M, MAX_JAW_OPENING_M, PAD_LENGTH_M

MOUNTING_HEADER = "R1Pro mounting"

R1PRO_EEF_DOCS = f"""A wheeled R1Pro: a holonomic base, a torso that lowers and tilts and carries
the head camera, and two 7-DoF arms, prefixed left_ and right_, each with a
parallel-jaw gripper and a camera on its wrist. Base frame: origin on the floor
at the centre of the base, +x forward, +y left, +z up. It moves with the base,
so after move_base every position refers to the new pose; the arms are mounted
on the torso, so move_torso moves both shoulders and the head camera (see the
mounting note below).
Every position, commanded and reported, is the point between the jaws where an
object is held. To grasp something, name that object's own position: no tool
offset is yours to add.
The finger pads are {PAD_LENGTH_M * 1000:.0f} mm long along the tool axis and the commanded point is
their centre, so the fingertips reach {FINGERTIPS_BEYOND_GRASP_M * 1000:.0f} mm past it; open, the pads are
{MAX_JAW_OPENING_M * 1000:.0f} mm apart.
Gripper 0 is fully closed, 1 is fully open, and commanding 0 always closes as
far as the object allows. Closing with both pads on an object grasps it; a
held object is let go only by opening fully, to 1.
Each wrist camera rides its own arm, so which views you have is itself
something you can command. When the head camera is blocked, or the working
arm's own wrist sits too close to show what it is doing, move the idle arm
until its wrist looks at the work: a motion that only buys a view is worth its
env steps when the alternative is aiming blind. The head camera rises and
falls with the torso, and tilting the torso pitches its view.
On anything that is not a simple block, where you take hold of it decides how
hard the rest of the task is. Choose the grasp that leaves the part you still
have to insert, pour from, hang or set down clear of the jaws and pointing the
way it will need to go, rather than whichever face is easiest to close on.
Scale every motion to how close it is to touching something. Full-size moves
belong in free space; once the jaws are within about a centimetre of the
object, the table, or wherever a held object is going down, keep corrections
to a few millimetres per call and re-read the cameras between them. A carried
object extends the hand, so measure that centimetre from the edges of whatever
is held, not from the jaws.
When a task is built from several placements, the order they are done in is
part of the task rather than a detail. Work out which piece has to be in place
before which, and do first whatever the rest will stand on, reach over, or
need the room beside.
Nothing plans around the scene: the arm planner avoids only the robot's own
body and the floor, and the base drives straight to the requested pose without
avoiding furniture, so clear the way yourself.
The advertised per-dimension bounds only reject gross magnitude errors: a pose
inside them can still be refused as unreachable by the planner, and that
refusal spends a turn and gains no env steps, so keep targets near the current
grasp point. A refusal is likelier the more one call asks for, so travel to
the new position first and turn the tool in a later call rather than doing
both at once. Large rotations are more reliable in two or three steps than in
one. Both arms named in one call are planned as a unit and refused as one: a
pose the planner will not take for either arm leaves the other standing still
as well, so move one arm at a time whenever either target is uncertain. A
target out of reach where the robot stands may come within reach after
move_base or move_torso."""


def format_mounting_notes(mounting: Mapping[str, Sequence[float]]) -> str:
    """Robot-only positions in the base frame as the episode starts (shoulders, head camera)."""
    lines = [f"{MOUNTING_HEADER} (base frame as the episode starts; the shoulders and the head camera move with the torso):"]
    for name, p in mounting.items():
        lines.append(f"- {name}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m")
    return "\n".join(lines) if len(lines) > 1 else ""


def build_r1pro_docs(mounting: Optional[Mapping[str, Sequence[float]]] = None) -> str:
    appendix = format_mounting_notes(mounting or {})
    return R1PRO_EEF_DOCS + ("\n\n" + appendix if appendix else "")
