"""Per-subtask description and success check, as task recipes (RoboProbe's ``TASK RECIPE`` of the Goal turn).

Each file says what the subtask asks for and what the environment's check measures, from the task's BDDL goal and TRO
targets (``ispatialgym/docs/task_bddl_formats.md``) and the executor's epilogue. Advice that holds whatever the task is
belongs in the embodiment notes instead, which ride every prompt."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

RECIPE_DIR = Path(__file__).resolve().parent


def recipe_key(alias: Optional[str]) -> Optional[str]:
    """``family`` or ``family-variant`` of an alias, the recipe file's name."""
    if not alias:
        return None
    parts = alias.split("-")
    fam = parts[0]
    if fam == "configuration_matching" and len(parts) > 1:
        kind = parts[1]
        return f"{fam}-" + ("articulation" if kind.startswith("articulation_") else kind)
    if fam == "alignment" and len(parts) > 1:
        return f"{fam}-{parts[1]}"
    return fam


def task_recipe(alias: Optional[str]) -> Optional[Tuple[str, str]]:
    """``(key, text)`` of the alias's recipe, or None when it has none."""
    key = recipe_key(alias)
    if not key or Path(key).name != key:
        return None
    path = RECIPE_DIR / f"{key}.md"
    return (key, path.read_text(encoding="utf-8")) if path.is_file() else None
