# Inter-object Relation Satisfaction

iSpatialGym subtask description and success check.
The live Goal instruction and goal image are authoritative for this instance.
The environment's own check judges the episode; this text does not replace it.

## Description

The robot must bring one object (the target) into a spatial relation with another (the partner), as in the goal image:
on top of it, under it, inside it, next to it, or touching it. In the small-scale "under" variant the target goes on top
of the partner, so the partner ends up under the target; in the large-scale variants the partner is a piece of the
scene's furniture.

## Success check

The relation named in the Goal holds and both objects are released. The exact poses in the goal image are not checked.

The episode ends as soon as the check first passes. The robot is then held still for a settling period, and the episode
succeeds only if the check still passes at its end with the robot and both objects at rest.
