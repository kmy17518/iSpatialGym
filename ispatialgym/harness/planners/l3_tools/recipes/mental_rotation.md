# Mental Rotation

iSpatialGym subtask description and success check.
The live Goal instruction and goal image are authoritative for this instance.
The environment's own check judges the episode; this text does not replace it.

## Description

The robot starts holding an object in one gripper. It must turn the object until its orientation matches the goal
image, and keep holding it.

## Success check

The object is still held, and its orientation in the world is within the task's tolerance of the goal orientation. Its
position is not checked. For some symmetric objects, orientations that look the same also count.

The episode ends as soon as the check first passes. The robot is then held still for a settling period, and the episode
succeeds only if the check still passes at its end with the robot and the object at rest.
