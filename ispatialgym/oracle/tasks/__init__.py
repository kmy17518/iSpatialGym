"""Layer-2 task oracles: short scripts composing :class:`ispatialgym.oracle.skills.Skills`.

Each oracle takes its goal from privileged state (TRO targets) and yields JoyLo actions.
"""

from ispatialgym.oracle.tasks.camera_relocalization import CameraRelocalizationOracle, solve_viewpoint
from ispatialgym.oracle.tasks.held_object import MentalRotationOracle, ObjectScalingOracle

__all__ = ["CameraRelocalizationOracle", "solve_viewpoint", "MentalRotationOracle", "ObjectScalingOracle"]
