"""v2 sampler package (spec docs/instance_sampling_spec.md; layout and contracts in README.md).

Import side effects are avoided: the task modules register themselves when ``ispatialgym.sampling.v2.tasks`` is
imported (``load_tasks()``), so that ``ground_isg`` can import ``v2.problem`` without OmniGibson.
"""

from __future__ import annotations

SAMPLER_VERSION = "2.0.0-dev"


def load_tasks() -> None:
    """Import every task module so that the ``@register`` decorators run (needs the simulator stack importable)."""
    import importlib
    import pkgutil

    from ispatialgym.sampling.v2 import tasks as _tasks

    for m in pkgutil.iter_modules(_tasks.__path__):
        importlib.import_module(f"{_tasks.__name__}.{m.name}")
