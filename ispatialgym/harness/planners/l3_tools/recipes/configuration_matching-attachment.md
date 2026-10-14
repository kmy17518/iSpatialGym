# Configuration Matching: Attachment

iSpatialGym subtask description and success check.
The live Goal instruction and goal image are authoritative for this instance.
The environment's own check judges the episode; this text does not replace it.

## Description

One part must be attached to another (for example a camera onto a tripod) in the pose shown in the goal image.

## Success check

The part is attached to its partner (a real attachment, not just resting on it), the partner stands on the support,
the part's pose relative to the partner is within the task's tolerance of the goal, and both are released. Where the
assembled pair stands is not checked.

The episode ends as soon as the check first passes. The robot is then held still for a settling period, and the episode
succeeds only if the check still passes at its end with the robot and both parts at rest.
