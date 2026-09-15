# Robot-only eef reach volume (R1Pro, untucked arms) — 2026-09-15

Produced by `python -m ispatialgym.sampling.eef_reach_volume` (B1/B2 of `TODO.md`): for every trunk of the JoyLo grid
(translate {0, 0.25, 0.5, 0.75, 0.95} x tilt {0, -0.3, -0.6, -0.9}) and each arm, a boolean mask over a 5 cm lattice of
eef positions in the base frame (x 0.25..0.90, lateral -0.70..0.70, z 0.20..1.70 for the pinch classes / 0.90..1.90 for
the push classes) telling whether the pose and its pre-pose (0.10 m back along the approach) pass the 12-seed batched
cuRobo IK with the untucked init configuration as locked joints and the floor plane as the only obstacle.

* `reach_volume.npz` + `meta.json`: merged volume read by `ispatialgym.sampling.eef_reach_volume.ReachVolume.load()`
  (this directory is the default location; `ISPATIALGYM_REACH_VOLUME` overrides).
* `shards/`: one file per (arm, translate, tilt); the eight `t1.00` shards carry `collides=True`.

Measured while producing it:

* JoyLo translate 1.0 (the "downward" keyframe) self-collides in cuRobo's sphere model for every arm posture
  (untucked / tucked / ready); untucked arms are collision-free up to 0.95, hanging (tucked) arms collide from 0.75.
  The deepest grid trunk is therefore 0.95.
* One 5 cm shard = 59,276 lattice points, ~35-45 % pruned by the shoulder-distance rule, ~4.6-6k feasible cells;
  8-10 min on the local RTX 4090, ~15 min on a RunPod 4090 (12 vCPU). Pod and local results are bit-identical.
* Left and right arm volumes are mirror images only to 98.2 % (21,947 of 1.19 M cells differ), so both arms are
  computed rather than mirrored.
* Right arm, translate 0.5 / tilt 0: forward pinches reachable for x 0.25-0.90, lateral -0.70..-0.05, z 0.70-1.70;
  crouched (0.95, -0.9) the pinch band reaches down to z 0.25.
