# Camera Relocalization

iSpatialGym subtask description and success check.
The live Goal instruction and goal image are authoritative for this instance.
The environment's own check judges the episode; this text does not replace it.

## Description

The robot starts elsewhere in the home. It must move until its head camera sees the view in the goal image: the same
place and the same viewing direction. The head camera's pose depends on the base pose and on the torso (its height and
tilt), not on the base alone. In the obstructed variant an object stands on the shortest way from the start to the goal.

## Success check

The head camera's position and its orientation are both within the task's tolerances of the goal camera pose. Nothing
else is checked.

The episode ends as soon as the check first passes. The robot is then held still for a settling period, and the episode
succeeds only if the check still passes at its end with the robot at rest.
