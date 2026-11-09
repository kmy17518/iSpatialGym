"""Per-family defaults of environment switches (i7-t03 round 8, LEAD DECISION #39 under the owner's mandate "fix all the bugs and speed up /
make yield better the slow low yield aliases").

A switch's environment variable wins when it is set ("1" / "on" / "true" = on, anything else - e.g. "0" - = off). When it is unset, the switch
is on only for a sampler of the SHELF TIP family (``IORSSmall.shelf_tip()``: the shelf layout with ``generation.shelf_tip_release`` - only
touching_03's customs entry carries it, and its counted templates' customs copies too) and only for the names in :data:`SHELF_TIP_DEFAULTS`.
Every other alias - touching_01 / 02 / 04, every IORS small / large alias - reads unset as off, as before. A stand-in without ``shelf_tip``
(or no sampler) reads unset as off."""
from __future__ import annotations

import os
from typing import Any, Tuple

#: the switches ON by default for the shelf tip family (touching_03); each name was added in its own commit (i7-t03 r8)
SHELF_TIP_DEFAULTS: Tuple[str, ...] = ("ISPATIALGYM_SHELF_FREE_START", "ISPATIALGYM_SHELF_JOINT_STANCE", "ISPATIALGYM_SHELF_RELEASE_IN_GAP", "ISPATIALGYM_SHELF_STANCE_SHIFT")


def shelf_tip_family(s: Any) -> bool:
    """``s`` samples the shelf tip layout (touching_03)."""
    fn = getattr(s, "shelf_tip", None) if s is not None else None
    if not callable(fn):
        return False
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 - a stand-in without the shelf start fields
        return False


def alias_defaults(s: Any = None) -> Tuple[str, ...]:
    """The switch names on by default for ``s`` (:data:`SHELF_TIP_DEFAULTS` for the shelf tip family, else ())."""
    return SHELF_TIP_DEFAULTS if shelf_tip_family(s) else ()


def switch_on(env: str, s: Any = None) -> bool:
    """The switch ``env``: the environment's explicit value, else on when ``s``'s family defaults list it."""
    v = os.environ.get(env, "").strip().lower()
    if v:
        return v in ("1", "on", "true")
    return env in alias_defaults(s)
