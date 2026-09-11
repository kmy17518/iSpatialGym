"""iSpatialGym evaluation layer: TRO target schema, group entities, and episode result summaries.

``ispatialgym.eval.geometry`` and ``ispatialgym.eval.targets`` only depend on torch; the simulator-facing pieces
(``group_entity``, ``tro.apply_targets_to_scope``, ``episode``) import OmniGibson lazily.
"""

from ispatialgym.eval.targets import (
    SCHEMA_VERSION,
    TRO_KEY,
    ArticulationTarget,
    GroupTarget,
    ISGTroMeta,
    PoseTarget,
    TROValidationError,
)
from ispatialgym.eval.tro import TROFile, load_tro, validate_tro_against_problem

__all__ = [
    "SCHEMA_VERSION",
    "TRO_KEY",
    "ArticulationTarget",
    "GroupTarget",
    "ISGTroMeta",
    "PoseTarget",
    "TROFile",
    "TROValidationError",
    "load_tro",
    "validate_tro_against_problem",
]
