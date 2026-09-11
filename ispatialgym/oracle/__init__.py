"""Oracle (scripted, privileged) demonstration policies for R1Pro in iSpatialGym.

Layering (see ``ispatialgym/docs/oracle_primitives_design.md``):

* ``r1pro_model``     - static robot model: joint groups, JoyLo trunk keyframes, derived URDF and
                        cuRobo v2 robot dictionaries.
* ``curobo_planner``  - cuRobo v2 backend: one planner per joint group ("embodiment"), world sync,
                        grasped-object attachment, FK / IK / trajectory planning.
* ``action``          - 23-D JoyLo action assembly with hold targets.
* ``trunk``           - JoyLo trunk translate / tilt <-> torso joint mapping.
* ``base_tracker``    - holonomic velocity tracker (joystick-like) for planned base paths.
* ``skills``          - Layer-1 motor skills, generators yielding actions.
* ``tasks``           - Layer-2 task oracles composed from skills.
"""

import logging
import os
import sys

# OmniGibson / Isaac Sim reconfigure the root logger at launch; give the oracle stack its own
# stdout handler so progress messages stay visible (level via ISPATIALGYM_LOG_LEVEL, default INFO).
_log = logging.getLogger("ispatialgym")
if not _log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(levelname)s %(name)s] %(message)s"))
    _log.addHandler(_h)
    _log.setLevel(os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    _log.propagate = False
