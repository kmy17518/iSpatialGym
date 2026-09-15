"""Oracle (scripted, privileged) demonstration policies for R1Pro in iSpatialGym.

Layering (``ispatialgym/docs/oracle_primitives_design.md`` §3, v2):

* Layer 0 backend  ``r1pro_model`` (static robot model, embodiments, derived URDF), ``curobo_planner`` (cuRobo v2:
  planners per embodiment, world sync, attach / detach, FK / IK / trajopt / differential IK), ``action``
  (23-D JoyLo action assembly), ``trunk`` (JoyLo trunk state), ``base_tracker`` (joystick-like base tracker);
  the shared geometry lives in ``ispatialgym.robot.conventions``.
* Layer 1 motions  ``motions.Motions``: ``track_path``, ``trunk_to``, ``arm_to``, ``arm_along``, ``gripper``,
  ``base_follow``, ``hold``, ``settle`` — decision-free generators of 23-D actions, ``SkillError(reason)`` on
  non-convergence.
* Layer 2 actions  ``actions`` (the vocabulary, ``targets()`` / ``annotate()``) + ``execute`` (the
  ``EXECUTORS[ActionClass] = (execute, postcondition)`` table).
* Layer 3 programs ``programs.expand(alias, sidecar)`` from the ``sidecar`` facts.
* Executor         ``executor.Executor`` (checkpoint → execute under ``qa.Monitors`` → postcondition → one retry →
  annotate; epilogue), ``recorder.Recorder`` (hdf5 / LeRobot / mp4 / annotations), ``run_isg`` (the driver).
* v1 (kept runnable): ``skills.Skills``, ``tasks``, ``run_isg_v1``.

Everything here is exported lazily (PEP 562) so that importing the package stays free of the simulator and of
import cycles with ``ispatialgym.robot.conventions``.
"""

import importlib
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

_EXPORTS = {
    # Layer 1
    "Motions": "ispatialgym.oracle.motions",
    "MotionsConfig": "ispatialgym.oracle.motions",
    "SkillError": "ispatialgym.oracle.motions",
    # Layer 2
    "EXECUTORS": "ispatialgym.oracle.execute",
    "ExecuteConfig": "ispatialgym.oracle.execute",
    "run_action": "ispatialgym.oracle.execute",
    "check_postcondition": "ispatialgym.oracle.execute",
    "Action": "ispatialgym.oracle.actions",
    "VOCABULARY": "ispatialgym.oracle.actions",
    # Layer 3
    "expand": "ispatialgym.oracle.programs",
    "Sidecar": "ispatialgym.oracle.sidecar",
    "SidecarError": "ispatialgym.oracle.sidecar",
    # executor / QA / recording
    "Executor": "ispatialgym.oracle.executor",
    "ExecutorConfig": "ispatialgym.oracle.executor",
    "SimState": "ispatialgym.oracle.executor",
    "Monitors": "ispatialgym.oracle.qa",
    "QAConfig": "ispatialgym.oracle.qa",
    "Verdict": "ispatialgym.oracle.qa",
    "CATEGORIES": "ispatialgym.oracle.qa",
    "Recorder": "ispatialgym.oracle.recorder",
    "RecorderConfig": "ispatialgym.oracle.recorder",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
