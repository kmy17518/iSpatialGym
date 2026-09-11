"""iSpatialGym: spatial-intelligence robotics benchmark on BEHAVIOR-1K / OmniGibson."""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_ROOT = os.path.join(REPO_ROOT, "ispatialgym", "assets")

__all__ = ["REPO_ROOT", "ASSETS_ROOT"]
