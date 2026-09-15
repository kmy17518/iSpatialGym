"""Robot-side conventions shared by the sampler and the oracle (oracle doc §3.5, TODO F5).

* ``conventions`` - the pure geometry both ``ispatialgym.sampling`` and ``ispatialgym.oracle`` derive poses from:
  frames, eef axis conventions, gripper constants, grasp-library -> eef poses, pre / contact poses, waypoint
  generators, stance transitions, base-follow, JoyLo trunk keyframes, head-camera FK, cube grasps.

Nothing in this package imports OmniGibson, Isaac Sim or cuRobo.
"""
