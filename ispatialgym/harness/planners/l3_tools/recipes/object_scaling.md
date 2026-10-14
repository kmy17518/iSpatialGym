# Object Scaling

iSpatialGym subtask description and success check.
The live Goal instruction and goal image are authoritative for this instance.
The environment's own check judges the episode; this text does not replace it.

## Description

The robot starts holding an object in one gripper. It must move the object toward or away from the head camera until
the object appears the same size in the head camera as in the goal image, and keep holding it.

## Success check

The object is still held, and its depth along the head camera's viewing axis (how far in front of the camera it is) is
within the task's tolerance of the goal depth. Where the object sits across the image and how it is turned are not
checked.

The episode ends as soon as the check first passes. The robot is then held still for a settling period, and the episode
succeeds only if the check still passes at its end with the robot and the object at rest.
