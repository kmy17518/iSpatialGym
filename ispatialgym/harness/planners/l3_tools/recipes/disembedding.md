# Disembedding

iSpatialGym subtask description and success check.
The live Goal instruction and goal image are authoritative for this instance.
The environment's own check judges the episode; this text does not replace it.

## Description

The target object sits among other objects on a source tray, possibly partly hidden by them. The robot must take only
the target and put it onto the other, destination tray, as in the goal image.

## Success check

The target rests on top of the destination tray and is released, and none of the other objects is on top of the
destination tray. Where exactly the target lies on the tray is not checked.

The episode ends as soon as the check first passes. The robot is then held still for a settling period, and the episode
succeeds only if the check still passes at its end with the robot and the objects at rest.
