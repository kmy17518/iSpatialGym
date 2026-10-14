# Alignment: Axial

iSpatialGym subtask description and success check.
The live Goal instruction and goal image are authoritative for this instance.
The environment's own check judges the episode; this text does not replace it.

## Description

An object lies on a table and must be turned until its orientation matches the goal image.

## Success check

The object is on the table and released, and its orientation is within the task's tolerance of the goal. Where it lies
on the table is not checked. For some symmetric objects, orientations that look the same also count.

The episode ends as soon as the check first passes. The robot is then held still for a settling period, and the episode
succeeds only if the check still passes at its end with the robot and the object at rest.
