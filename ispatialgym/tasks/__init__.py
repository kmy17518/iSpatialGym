"""iSpatialGym task classes. Importing this package registers ``ISpatialGymTask`` in OmniGibson's task registry so
that ``{"task": {"type": "ISpatialGymTask", ...}}`` works in an environment config."""

from ispatialgym.tasks.ispatialgym_task import ISpatialGymTask

__all__ = ["ISpatialGymTask"]
