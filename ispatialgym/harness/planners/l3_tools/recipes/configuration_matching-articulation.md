# Configuration Matching: Articulation

iSpatialGym subtask description and success check.
The live Goal instruction and goal image are authoritative for this instance.
The environment's own check judges the episode; this text does not replace it.

## Description

An articulated object (a door, a cabinet, an appliance, a box with a lid) must be opened or closed until it matches the
goal image. Large-scale variants use a piece of the scene's own furniture; small-scale variants use an object standing
on a support.

## Success check

The object is open (or closed, in a closing task) in the requested direction, every moving joint is within the task's
tolerance of its position in the goal, and the object is released.

The episode ends as soon as the check first passes. The robot is then held still for a settling period, and the episode
succeeds only if the check still passes at its end with the robot and the object at rest.
