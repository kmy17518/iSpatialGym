# Configuration Matching: Block Design

iSpatialGym subtask description and success check.
The live Goal instruction and goal image are authoritative for this instance.
The environment's own check judges the episode; this text does not replace it.

## Description

Four interchangeable blocks lie on a table. The robot must arrange them as in the goal image.

## Success check

Every block is on the table and released, and the blocks' arrangement relative to each other matches the goal: each
block's pose relative to the others is within the task's tolerance, the blocks being interchangeable and a block's
symmetric orientations counting alike. Where the arrangement sits on the table is not checked.

The episode ends as soon as the check first passes. The robot is then held still for a settling period, and the episode
succeeds only if the check still passes at its end with the robot and the blocks at rest.
