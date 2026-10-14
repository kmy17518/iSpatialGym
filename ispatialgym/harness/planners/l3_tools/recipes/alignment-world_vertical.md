# Alignment: World Vertical

iSpatialGym subtask description and success check.
The live Goal instruction and goal image are authoritative for this instance.
The environment's own check judges the episode; this text does not replace it.

## Description

An object (for example a painting) hangs on a wall on a pivot at its centre and can only turn in the plane of the wall.
It must be turned until it hangs as in the goal image.

## Success check

The object is released and its orientation is within the task's tolerance of the goal. Its position is fixed by the
pivot and is not checked.

The episode ends as soon as the check first passes. The robot is then held still for a settling period, and the episode
succeeds only if the check still passes at its end with the robot and the object at rest.
