"""Shared pytest hooks for the iSpatialGym tests.

The non-sim tests never import OmniGibson; the sim tests (``test_sim_evaluator.py``) launch Isaac Sim, which must be
shut down explicitly at the end of the session (same pattern as OmniGibson's own ``tests/conftest.py``).
"""

import sys


def pytest_unconfigure(config):
    og = sys.modules.get("omnigibson")
    if og is not None and getattr(og, "app", None) is not None:
        # Isaac Sim's app.close() ends the process; make sure pytest's report has reached the terminal first.
        sys.stdout.flush()
        sys.stderr.flush()
        og.shutdown()
